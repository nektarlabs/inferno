use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
};

use ::metal::{Buffer, CommandBufferRef, ComputePipelineState, Device};
use common::{Error, Result};

use crate::{DeviceRopeTable, DeviceRouterTopK, GgufKQuant};

use super::{
    arena::MetalArena,
    buffers::{require_f32_capacity, u32_buffer, u8_buffer_no_copy, ImmutableF32BufferCache},
    command::{encode_1d_threadgroups_args, KernelArg},
    laguna_views::MetalLagunaViews,
    library::MetalLibrary,
    pipeline::compute_pipeline,
};

const K_BLOCK_VALUES: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q6_K_BLOCK_BYTES: usize = 210;
const SIMD_LANES: usize = 32;
const SIMDGROUPS_PER_THREADGROUP: usize = 4;
// One accumulator per lane was fastest across the real XS projection shapes.
const PROJECTION_ROWS_PER_SIMDGROUP: usize = 1;
// One row keeps gate/up register pressure low enough to improve real decode.
const EXPERT_GATE_UP_ROWS_PER_SIMDGROUP: usize = 1;
// Top-8 down already carries eight dot products; one row preserves occupancy.
const EXPERT_DOWN_ROWS_PER_SIMDGROUP: usize = 1;
const EXPERT_DOWN_PARALLEL_SIMDGROUPS: usize = 4;
const KERNEL_SOURCE: &str = include_str!("kernels/laguna_xs_kernels.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct WeightKey {
    address: usize,
    byte_len: usize,
}

struct WeightBuffer {
    storage: Buffer,
    byte_offset: usize,
}

pub(crate) struct MetalLagunaXs {
    pipelines: Mutex<Option<LagunaXsPipelines>>,
    arena: MetalArena,
    weights: Mutex<HashMap<WeightKey, Buffer>>,
    f32_weights: ImmutableF32BufferCache,
    laguna_views: Arc<MetalLagunaViews>,
}

struct LagunaXsPipelines {
    q4_matvec: ComputePipelineState,
    q6_matvec: ComputePipelineState,
    q4_decode_matvec: ComputePipelineState,
    q6_decode_matvec: ComputePipelineState,
    q4_decode_matvec_residuals: ComputePipelineState,
    q6_decode_matvec_residuals: ComputePipelineState,
    q4_gate_up_swiglu_decode: ComputePipelineState,
    router_top8_decode: ComputePipelineState,
    qk_norm_rope_pair_decode: ComputePipelineState,
    rms_norm_decode: ComputePipelineState,
    q4_attention_projections: ComputePipelineState,
    q6_value_attention_projections: ComputePipelineState,
    q4_embedding: ComputePipelineState,
    q4_expert_gate_up: ComputePipelineState,
    q4_expert_down: ComputePipelineState,
    q6_expert_down: ComputePipelineState,
    q4_expert_down_parallel: ComputePipelineState,
    q6_expert_down_parallel: ComputePipelineState,
}

impl MetalLagunaXs {
    pub(crate) fn new(arena: MetalArena, laguna_views: Arc<MetalLagunaViews>) -> Self {
        Self {
            pipelines: Mutex::new(None),
            arena,
            weights: Mutex::new(HashMap::new()),
            f32_weights: ImmutableF32BufferCache::default(),
            laguna_views,
        }
    }

    fn pipelines(&self, device: &Device) -> Result<MutexGuard<'_, Option<LagunaXsPipelines>>> {
        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| Error::backend("Laguna XS Metal pipeline lock poisoned"))?;
        if pipelines.is_none() {
            let library = MetalLibrary::compile_source(device, KERNEL_SOURCE)?;
            *pipelines = Some(LagunaXsPipelines {
                q4_matvec: compute_pipeline(device, &library, "laguna_xs_q4_k_matvec_f32_kernel")?,
                q6_matvec: compute_pipeline(device, &library, "laguna_xs_q6_k_matvec_f32_kernel")?,
                q4_decode_matvec: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_q4_k_decode_matvec_f32_kernel",
                )?,
                q6_decode_matvec: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_q6_k_decode_matvec_f32_kernel",
                )?,
                q4_decode_matvec_residuals: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_q4_k_decode_matvec_residuals_f32_kernel",
                )?,
                q6_decode_matvec_residuals: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_q6_k_decode_matvec_residuals_f32_kernel",
                )?,
                q4_gate_up_swiglu_decode: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_q4_gate_up_swiglu_decode_f32_kernel",
                )?,
                router_top8_decode: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_router_top8_decode_f32_kernel",
                )?,
                qk_norm_rope_pair_decode: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_qk_norm_rope_pair_decode_f32_kernel",
                )?,
                rms_norm_decode: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_rms_norm_decode_f32_kernel",
                )?,
                q4_attention_projections: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_q4_attention_projections_f32_kernel",
                )?,
                q6_value_attention_projections: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_q6_value_attention_projections_f32_kernel",
                )?,
                q4_embedding: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_q4_k_embedding_f32_kernel",
                )?,
                q4_expert_gate_up: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_q4_expert_gate_up_f32_kernel",
                )?,
                q4_expert_down: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_q4_expert_down_sum_f32_kernel",
                )?,
                q6_expert_down: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_q6_expert_down_sum_f32_kernel",
                )?,
                q4_expert_down_parallel: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_q4_expert_down_parallel_f32_kernel",
                )?,
                q6_expert_down_parallel: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_q6_expert_down_parallel_f32_kernel",
                )?,
            });
        }
        Ok(pipelines)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_matvec(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        quant: GgufKQuant,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        validate_k_dimensions(row_count, in_features, out_features)?;
        let expected_input_len = row_count
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("Laguna XS matvec input length overflow"))?;
        if input_len != expected_input_len {
            return Err(Error::backend(format!(
                "Laguna XS matvec expected {expected_input_len} input values, got {input_len}"
            )));
        }
        require_f32_capacity(input, input_len, "Laguna XS matvec input")?;
        validate_weight_len(quant, weights, in_features, out_features)?;

        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Laguna XS matvec output length overflow"))?;
        let output = self.arena.empty_f32(output_len)?;
        let weight = self.weight_buffer(device, weights)?;
        let pipelines = self.pipelines(device)?;
        let pipelines = pipelines
            .as_ref()
            .ok_or_else(|| Error::backend("Laguna XS Metal pipelines were not initialized"))?;
        let paired_decode = row_count == 1 && out_features.is_multiple_of(2);
        let pipeline = match (quant, paired_decode) {
            (GgufKQuant::Q4K, true) => &pipelines.q4_decode_matvec,
            (GgufKQuant::Q6K, true) => &pipelines.q6_decode_matvec,
            (GgufKQuant::Q4K, false) => &pipelines.q4_matvec,
            (GgufKQuant::Q6K, false) => &pipelines.q6_matvec,
        };
        let output_row_groups = if paired_decode {
            out_features / 2
        } else {
            out_features.div_ceil(PROJECTION_ROWS_PER_SIMDGROUP)
        };
        let simdgroup_count = row_count
            .checked_mul(output_row_groups)
            .ok_or_else(|| Error::backend("Laguna XS matvec SIMD-group count overflow"))?;
        let threadgroup_count = simdgroup_count.div_ceil(SIMDGROUPS_PER_THREADGROUP);
        let args = [
            KernelArg::BufferOffset(&weight.storage, weight.byte_offset),
            KernelArg::Buffer(input),
            KernelArg::Buffer(&output),
            KernelArg::U32(as_u32(row_count, "row count")?),
            KernelArg::U32(as_u32(in_features, "input width")?),
            KernelArg::U32(as_u32(out_features, "output width")?),
        ];
        encode_1d_threadgroups_args(
            command_buffer,
            pipeline,
            &args,
            threadgroup_count,
            SIMDGROUPS_PER_THREADGROUP * SIMD_LANES,
        )?;
        Ok(output)
    }

    pub(crate) fn encode_router_topk(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        router_logits: &Buffer,
        router_logits_len: usize,
        correction_bias: &[f32],
        routed_scaling_factor: f32,
    ) -> Result<DeviceRouterTopK> {
        const EXPERT_COUNT: usize = 256;
        const TOP_K: usize = 8;
        if router_logits_len != EXPERT_COUNT {
            return Err(Error::backend(format!(
                "Laguna XS decode router requires {EXPERT_COUNT} logits, got {router_logits_len}"
            )));
        }
        if correction_bias.len() != EXPERT_COUNT {
            return Err(Error::backend(format!(
                "Laguna XS router correction bias requires {EXPERT_COUNT} values, got {}",
                correction_bias.len()
            )));
        }
        if !routed_scaling_factor.is_finite() {
            return Err(Error::backend(
                "Laguna XS router scaling factor must be finite",
            ));
        }
        require_f32_capacity(router_logits, router_logits_len, "Laguna XS router logits")?;

        let expert_ids = self.arena.empty_u32(TOP_K)?;
        let expert_weights = self.arena.empty_f32(TOP_K)?;
        let token_indices = self.arena.empty_u32(TOP_K)?;
        let correction_bias = self.f32_weights.get(device, correction_bias)?;
        let pipelines = self.pipelines(device)?;
        let pipelines = pipelines
            .as_ref()
            .ok_or_else(|| Error::backend("Laguna XS Metal pipelines were not initialized"))?;
        let args = [
            KernelArg::Buffer(router_logits),
            KernelArg::Buffer(&correction_bias),
            KernelArg::Buffer(&expert_ids),
            KernelArg::Buffer(&expert_weights),
            KernelArg::Buffer(&token_indices),
            KernelArg::F32(routed_scaling_factor),
        ];
        encode_1d_threadgroups_args(
            command_buffer,
            &pipelines.router_top8_decode,
            &args,
            1,
            SIMD_LANES,
        )?;
        Ok(DeviceRouterTopK {
            token_count: 1,
            expert_count: EXPERT_COUNT,
            top_k: TOP_K,
            token_indices,
            expert_ids,
            expert_weights,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_qk_norm_rope_pair(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        query: &Buffer,
        query_len: usize,
        key: &Buffer,
        key_len: usize,
        query_norm_weight: &[f32],
        key_norm_weight: &[f32],
        query_head_count: usize,
        position_offset: usize,
        eps: f32,
        table: &DeviceRopeTable,
    ) -> Result<(Buffer, Buffer)> {
        const HEAD_DIM: usize = 128;
        const KEY_HEAD_COUNT: usize = 8;
        if !matches!(query_head_count, 48 | 64) || !matches!(table.rotary_dim, 64 | 128) {
            return Err(Error::backend(format!(
                "Laguna XS decode Q/K norm+RoPE requires 48|64 query heads and rotary dim 64|128, got heads={query_head_count}, rotary={}",
                table.rotary_dim
            )));
        }
        let expected_query_len = query_head_count
            .checked_mul(HEAD_DIM)
            .ok_or_else(|| Error::backend("Laguna XS query length overflow"))?;
        let expected_key_len = KEY_HEAD_COUNT * HEAD_DIM;
        if query_len != expected_query_len || key_len != expected_key_len {
            return Err(Error::backend(format!(
                "Laguna XS decode Q/K lengths must be {expected_query_len}/{expected_key_len}, got {query_len}/{key_len}"
            )));
        }
        if query_norm_weight.len() != HEAD_DIM || key_norm_weight.len() != HEAD_DIM {
            return Err(Error::backend(format!(
                "Laguna XS Q/K norm weights must both contain {HEAD_DIM} values, got {}/{}",
                query_norm_weight.len(),
                key_norm_weight.len()
            )));
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err(Error::backend(format!(
                "Laguna XS Q/K RMSNorm epsilon must be finite and positive, got {eps}"
            )));
        }
        require_f32_capacity(query, query_len, "Laguna XS query norm+RoPE input")?;
        require_f32_capacity(key, key_len, "Laguna XS key norm+RoPE input")?;

        let query_output = self.arena.empty_f32(query_len)?;
        let key_output = self.arena.empty_f32(key_len)?;
        let query_norm_weight = self.f32_weights.get(device, query_norm_weight)?;
        let key_norm_weight = self.f32_weights.get(device, key_norm_weight)?;
        let pipelines = self.pipelines(device)?;
        let pipelines = pipelines
            .as_ref()
            .ok_or_else(|| Error::backend("Laguna XS Metal pipelines were not initialized"))?;
        let args = [
            KernelArg::Buffer(query),
            KernelArg::Buffer(key),
            KernelArg::Buffer(&query_norm_weight),
            KernelArg::Buffer(&key_norm_weight),
            KernelArg::Buffer(&table.inverse_frequency),
            KernelArg::Buffer(&query_output),
            KernelArg::Buffer(&key_output),
            KernelArg::U32(as_u32(query_head_count, "query head count")?),
            KernelArg::U32(as_u32(table.rotary_dim, "rotary dimension")?),
            KernelArg::U32(as_u32(position_offset, "position offset")?),
            KernelArg::F32(eps),
            KernelArg::F32(table.attention_factor),
        ];
        encode_1d_threadgroups_args(
            command_buffer,
            &pipelines.qk_norm_rope_pair_decode,
            &args,
            query_head_count + KEY_HEAD_COUNT,
            SIMD_LANES,
        )?;
        Ok((query_output, key_output))
    }

    pub(crate) fn encode_rms_norm(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        input: &Buffer,
        input_len: usize,
        weight: &[f32],
        eps: f32,
    ) -> Result<Buffer> {
        const HIDDEN_SIZE: usize = 2_048;
        const THREADS: usize = 256;
        if input_len != HIDDEN_SIZE || weight.len() != HIDDEN_SIZE {
            return Err(Error::backend(format!(
                "Laguna XS decode RMSNorm requires input/weight lengths {HIDDEN_SIZE}, got {input_len}/{}",
                weight.len()
            )));
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err(Error::backend(format!(
                "Laguna XS decode RMSNorm epsilon must be finite and positive, got {eps}"
            )));
        }
        require_f32_capacity(input, input_len, "Laguna XS RMSNorm input")?;

        let output = self.arena.empty_f32(HIDDEN_SIZE)?;
        let weight = self.f32_weights.get(device, weight)?;
        let pipelines = self.pipelines(device)?;
        let pipelines = pipelines
            .as_ref()
            .ok_or_else(|| Error::backend("Laguna XS Metal pipelines were not initialized"))?;
        let args = [
            KernelArg::Buffer(input),
            KernelArg::Buffer(&weight),
            KernelArg::Buffer(&output),
            KernelArg::F32(eps),
        ];
        encode_1d_threadgroups_args(
            command_buffer,
            &pipelines.rms_norm_decode,
            &args,
            1,
            THREADS,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_matvec_residuals(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        quant: GgufKQuant,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        residual_a: &Buffer,
        residual_b: Option<&Buffer>,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        validate_k_dimensions(row_count, in_features, out_features)?;
        if row_count != 1 || !out_features.is_multiple_of(2) {
            return Err(Error::backend(format!(
                "Laguna XS fused decode matvec requires one row and an even output width, got rows={row_count}, out={out_features}"
            )));
        }
        let expected_input_len = row_count
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("Laguna XS fused matvec input length overflow"))?;
        if input_len != expected_input_len {
            return Err(Error::backend(format!(
                "Laguna XS fused matvec expected {expected_input_len} input values, got {input_len}"
            )));
        }
        require_f32_capacity(input, input_len, "Laguna XS fused matvec input")?;
        require_f32_capacity(
            residual_a,
            out_features,
            "Laguna XS fused matvec first residual",
        )?;
        if let Some(residual_b) = residual_b {
            require_f32_capacity(
                residual_b,
                out_features,
                "Laguna XS fused matvec second residual",
            )?;
        }
        validate_weight_len(quant, weights, in_features, out_features)?;

        let output = self.arena.empty_f32(out_features)?;
        let weight = self.weight_buffer(device, weights)?;
        let pipelines = self.pipelines(device)?;
        let pipelines = pipelines
            .as_ref()
            .ok_or_else(|| Error::backend("Laguna XS Metal pipelines were not initialized"))?;
        let pipeline = match quant {
            GgufKQuant::Q4K => &pipelines.q4_decode_matvec_residuals,
            GgufKQuant::Q6K => &pipelines.q6_decode_matvec_residuals,
        };
        let second_residual = residual_b.unwrap_or(residual_a);
        let residual_count = if residual_b.is_some() { 2_u32 } else { 1_u32 };
        let args = [
            KernelArg::BufferOffset(&weight.storage, weight.byte_offset),
            KernelArg::Buffer(input),
            KernelArg::Buffer(residual_a),
            KernelArg::Buffer(second_residual),
            KernelArg::Buffer(&output),
            KernelArg::U32(as_u32(row_count, "row count")?),
            KernelArg::U32(as_u32(in_features, "input width")?),
            KernelArg::U32(as_u32(out_features, "output width")?),
            KernelArg::U32(residual_count),
        ];
        let simdgroup_count = out_features / 2;
        encode_1d_threadgroups_args(
            command_buffer,
            pipeline,
            &args,
            simdgroup_count.div_ceil(SIMDGROUPS_PER_THREADGROUP),
            SIMDGROUPS_PER_THREADGROUP * SIMD_LANES,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_gate_up_swiglu(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        gate_weights: &[u8],
        up_weights: &[u8],
        input: &Buffer,
        input_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        validate_k_dimensions(row_count, in_features, out_features)?;
        if row_count != 1 {
            return Err(Error::backend(format!(
                "Laguna XS fused gate/up decode requires one row, got {row_count}"
            )));
        }
        if input_len != in_features {
            return Err(Error::backend(format!(
                "Laguna XS fused gate/up expected {in_features} input values, got {input_len}"
            )));
        }
        require_f32_capacity(input, input_len, "Laguna XS fused gate/up input")?;
        validate_weight_len(GgufKQuant::Q4K, gate_weights, in_features, out_features)?;
        validate_weight_len(GgufKQuant::Q4K, up_weights, in_features, out_features)?;

        let output = self.arena.empty_f32(out_features)?;
        let gate = self.weight_buffer(device, gate_weights)?;
        let up = self.weight_buffer(device, up_weights)?;
        let pipelines = self.pipelines(device)?;
        let pipelines = pipelines
            .as_ref()
            .ok_or_else(|| Error::backend("Laguna XS Metal pipelines were not initialized"))?;
        let args = [
            KernelArg::BufferOffset(&gate.storage, gate.byte_offset),
            KernelArg::BufferOffset(&up.storage, up.byte_offset),
            KernelArg::Buffer(input),
            KernelArg::Buffer(&output),
            KernelArg::U32(as_u32(row_count, "row count")?),
            KernelArg::U32(as_u32(in_features, "input width")?),
            KernelArg::U32(as_u32(out_features, "output width")?),
        ];
        encode_1d_threadgroups_args(
            command_buffer,
            &pipelines.q4_gate_up_swiglu_decode,
            &args,
            out_features.div_ceil(SIMDGROUPS_PER_THREADGROUP),
            SIMDGROUPS_PER_THREADGROUP * SIMD_LANES,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_attention_projections(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        query_weights: &[u8],
        key_weights: &[u8],
        value_weights: &[u8],
        value_quant: GgufKQuant,
        gate_weights: &[u8],
        input: &Buffer,
        input_len: usize,
        in_features: usize,
        query_features: usize,
        key_features: usize,
        value_features: usize,
        gate_features: usize,
    ) -> Result<[Buffer; 4]> {
        if input_len != in_features {
            return Err(Error::backend(format!(
                "Laguna XS fused attention expected one {in_features}-value decode row, got {input_len} values"
            )));
        }
        require_f32_capacity(input, input_len, "Laguna XS fused attention input")?;
        if [query_features, key_features, value_features, gate_features]
            .into_iter()
            .any(|features| !features.is_multiple_of(2))
        {
            return Err(Error::backend(format!(
                "Laguna XS fused attention projection widths must be even, got Q={query_features}, K={key_features}, V={value_features}, gate={gate_features}"
            )));
        }
        validate_weight_len(GgufKQuant::Q4K, query_weights, in_features, query_features)?;
        validate_weight_len(GgufKQuant::Q4K, key_weights, in_features, key_features)?;
        validate_weight_len(value_quant, value_weights, in_features, value_features)?;
        validate_weight_len(GgufKQuant::Q4K, gate_weights, in_features, gate_features)?;

        let query_output = self.arena.empty_f32(query_features)?;
        let key_output = self.arena.empty_f32(key_features)?;
        let value_output = self.arena.empty_f32(value_features)?;
        let gate_output = self.arena.empty_f32(gate_features)?;
        let query = self.weight_buffer(device, query_weights)?;
        let key = self.weight_buffer(device, key_weights)?;
        let value = self.weight_buffer(device, value_weights)?;
        let gate = self.weight_buffer(device, gate_weights)?;
        let pipelines = self.pipelines(device)?;
        let pipelines = pipelines
            .as_ref()
            .ok_or_else(|| Error::backend("Laguna XS Metal pipelines were not initialized"))?;
        let pipeline = match value_quant {
            GgufKQuant::Q4K => &pipelines.q4_attention_projections,
            GgufKQuant::Q6K => &pipelines.q6_value_attention_projections,
        };
        let total_features = query_features
            .checked_add(key_features)
            .and_then(|features| features.checked_add(value_features))
            .and_then(|features| features.checked_add(gate_features))
            .ok_or_else(|| Error::backend("Laguna XS attention projection width overflow"))?;
        let simdgroup_count = total_features / 2;
        let args = [
            KernelArg::BufferOffset(&query.storage, query.byte_offset),
            KernelArg::BufferOffset(&key.storage, key.byte_offset),
            KernelArg::BufferOffset(&value.storage, value.byte_offset),
            KernelArg::BufferOffset(&gate.storage, gate.byte_offset),
            KernelArg::Buffer(input),
            KernelArg::Buffer(&query_output),
            KernelArg::Buffer(&key_output),
            KernelArg::Buffer(&value_output),
            KernelArg::Buffer(&gate_output),
            KernelArg::U32(as_u32(in_features, "input width")?),
            KernelArg::U32(as_u32(query_features, "query width")?),
            KernelArg::U32(as_u32(key_features, "key width")?),
            KernelArg::U32(as_u32(value_features, "value width")?),
            KernelArg::U32(as_u32(gate_features, "gate width")?),
        ];
        encode_1d_threadgroups_args(
            command_buffer,
            pipeline,
            &args,
            simdgroup_count.div_ceil(SIMDGROUPS_PER_THREADGROUP),
            SIMDGROUPS_PER_THREADGROUP * SIMD_LANES,
        )?;
        Ok([query_output, key_output, value_output, gate_output])
    }

    pub(crate) fn encode_q4_embedding(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        weights: &[u8],
        token_ids: &[u32],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Buffer> {
        if token_ids.is_empty() || vocab_size == 0 || !hidden_size.is_multiple_of(K_BLOCK_VALUES) {
            return Err(Error::backend(format!(
                "Laguna XS embedding dimensions are invalid: tokens={}, vocab={vocab_size}, hidden={hidden_size}",
                token_ids.len()
            )));
        }
        validate_weight_len(GgufKQuant::Q4K, weights, hidden_size, vocab_size)?;
        if token_ids
            .iter()
            .any(|token_id| *token_id as usize >= vocab_size)
        {
            return Err(Error::backend(format!(
                "Laguna XS embedding token ID exceeds vocabulary size {vocab_size}"
            )));
        }

        let output_len = token_ids
            .len()
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("Laguna XS embedding output length overflow"))?;
        let output = self.arena.empty_f32(output_len)?;
        let token_ids = u32_buffer(device, token_ids)?;
        let weight = self.weight_buffer(device, weights)?;
        let pipelines = self.pipelines(device)?;
        let pipelines = pipelines
            .as_ref()
            .ok_or_else(|| Error::backend("Laguna XS Metal pipelines were not initialized"))?;
        let args = [
            KernelArg::BufferOffset(&weight.storage, weight.byte_offset),
            KernelArg::Buffer(&token_ids),
            KernelArg::Buffer(&output),
            KernelArg::U32(as_u32(output_len / hidden_size, "token count")?),
            KernelArg::U32(as_u32(vocab_size, "vocabulary size")?),
            KernelArg::U32(as_u32(hidden_size, "hidden size")?),
        ];
        let threads_per_group = 256;
        encode_1d_threadgroups_args(
            command_buffer,
            &pipelines.q4_embedding,
            &args,
            output_len.div_ceil(threads_per_group),
            threads_per_group,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_moe(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: GgufKQuant,
        input: &Buffer,
        input_len: usize,
        routing: &DeviceRouterTopK,
        in_features: usize,
        intermediate_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        let token_count = routing.token_count;
        let top_k = routing.top_k;
        let expert_count = routing.expert_count;
        if token_count == 0 || top_k == 0 || top_k > expert_count {
            return Err(Error::backend(format!(
                "Laguna XS routing dimensions are invalid: tokens={token_count}, top_k={top_k}, experts={expert_count}"
            )));
        }
        let expected_input_len = token_count
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("Laguna XS MoE input length overflow"))?;
        if input_len != expected_input_len {
            return Err(Error::backend(format!(
                "Laguna XS MoE expected {expected_input_len} input values, got {input_len}"
            )));
        }
        require_f32_capacity(input, input_len, "Laguna XS MoE input")?;
        validate_weight_len(
            GgufKQuant::Q4K,
            gate_weights,
            in_features,
            intermediate_features
                .checked_mul(expert_count)
                .ok_or_else(|| Error::backend("Laguna XS gate expert rows overflow"))?,
        )?;
        validate_weight_len(
            GgufKQuant::Q4K,
            up_weights,
            in_features,
            intermediate_features
                .checked_mul(expert_count)
                .ok_or_else(|| Error::backend("Laguna XS up expert rows overflow"))?,
        )?;
        validate_weight_len(
            down_quant,
            down_weights,
            intermediate_features,
            out_features
                .checked_mul(expert_count)
                .ok_or_else(|| Error::backend("Laguna XS down expert rows overflow"))?,
        )?;

        let assignment_count = routing.assignment_count()?;
        let intermediate_len = assignment_count
            .checked_mul(intermediate_features)
            .ok_or_else(|| Error::backend("Laguna XS MoE intermediate length overflow"))?;
        let output_len = token_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Laguna XS MoE output length overflow"))?;
        let intermediate = self.arena.empty_f32(intermediate_len)?;
        let output = self.arena.empty_f32(output_len)?;
        let gate = self.weight_buffer(device, gate_weights)?;
        let up = self.weight_buffer(device, up_weights)?;
        let down = self.weight_buffer(device, down_weights)?;
        let pipelines = self.pipelines(device)?;
        let pipelines = pipelines
            .as_ref()
            .ok_or_else(|| Error::backend("Laguna XS Metal pipelines were not initialized"))?;

        let gate_args = [
            KernelArg::BufferOffset(&gate.storage, gate.byte_offset),
            KernelArg::BufferOffset(&up.storage, up.byte_offset),
            KernelArg::Buffer(input),
            KernelArg::Buffer(&routing.token_indices),
            KernelArg::Buffer(&routing.expert_ids),
            KernelArg::Buffer(&routing.expert_weights),
            KernelArg::Buffer(&intermediate),
            KernelArg::U32(as_u32(assignment_count, "assignment count")?),
            KernelArg::U32(as_u32(expert_count, "expert count")?),
            KernelArg::U32(as_u32(in_features, "input width")?),
            KernelArg::U32(as_u32(intermediate_features, "intermediate width")?),
        ];
        let gate_simdgroups = assignment_count
            .checked_mul(intermediate_features.div_ceil(EXPERT_GATE_UP_ROWS_PER_SIMDGROUP))
            .ok_or_else(|| Error::backend("Laguna XS gate/up SIMD-group count overflow"))?;
        encode_1d_threadgroups_args(
            command_buffer,
            &pipelines.q4_expert_gate_up,
            &gate_args,
            gate_simdgroups.div_ceil(SIMDGROUPS_PER_THREADGROUP),
            SIMDGROUPS_PER_THREADGROUP * SIMD_LANES,
        )?;

        let down_args = [
            KernelArg::BufferOffset(&down.storage, down.byte_offset),
            KernelArg::Buffer(&routing.expert_ids),
            KernelArg::Buffer(&intermediate),
            KernelArg::Buffer(&output),
            KernelArg::U32(as_u32(token_count, "token count")?),
            KernelArg::U32(as_u32(top_k, "top-k")?),
            KernelArg::U32(as_u32(expert_count, "expert count")?),
            KernelArg::U32(as_u32(intermediate_features, "intermediate width")?),
            KernelArg::U32(as_u32(out_features, "output width")?),
        ];
        if token_count == 1 && top_k == 8 {
            let down_pipeline = match down_quant {
                GgufKQuant::Q4K => &pipelines.q4_expert_down_parallel,
                GgufKQuant::Q6K => &pipelines.q6_expert_down_parallel,
            };
            encode_1d_threadgroups_args(
                command_buffer,
                down_pipeline,
                &down_args,
                out_features,
                EXPERT_DOWN_PARALLEL_SIMDGROUPS * SIMD_LANES,
            )?;
        } else {
            let down_simdgroups = token_count
                .checked_mul(out_features.div_ceil(EXPERT_DOWN_ROWS_PER_SIMDGROUP))
                .ok_or_else(|| Error::backend("Laguna XS down SIMD-group count overflow"))?;
            let down_pipeline = match down_quant {
                GgufKQuant::Q4K => &pipelines.q4_expert_down,
                GgufKQuant::Q6K => &pipelines.q6_expert_down,
            };
            encode_1d_threadgroups_args(
                command_buffer,
                down_pipeline,
                &down_args,
                down_simdgroups.div_ceil(SIMDGROUPS_PER_THREADGROUP),
                SIMDGROUPS_PER_THREADGROUP * SIMD_LANES,
            )?;
        }
        Ok(output)
    }

    fn weight_buffer(&self, device: &Device, bytes: &[u8]) -> Result<WeightBuffer> {
        if let Some(binding) = self.laguna_views.binding(bytes)? {
            return Ok(WeightBuffer {
                storage: binding.buffer,
                byte_offset: binding.byte_offset,
            });
        }

        let key = WeightKey {
            address: bytes.as_ptr() as usize,
            byte_len: bytes.len(),
        };
        let mut weights = self
            .weights
            .lock()
            .map_err(|_| Error::backend("Laguna XS Metal weight cache lock poisoned"))?;
        if let Some(buffer) = weights.get(&key) {
            return Ok(WeightBuffer {
                storage: buffer.clone(),
                byte_offset: 0,
            });
        }
        let buffer = u8_buffer_no_copy(device, bytes)?;
        weights.insert(key, buffer.clone());
        Ok(WeightBuffer {
            storage: buffer,
            byte_offset: 0,
        })
    }
}

fn validate_k_dimensions(row_count: usize, in_features: usize, out_features: usize) -> Result<()> {
    if row_count == 0 || out_features == 0 || !in_features.is_multiple_of(K_BLOCK_VALUES) {
        return Err(Error::backend(format!(
            "Laguna XS K-quant matvec dimensions are invalid: rows={row_count}, in={in_features}, out={out_features}"
        )));
    }
    Ok(())
}

fn validate_weight_len(
    quant: GgufKQuant,
    weights: &[u8],
    in_features: usize,
    out_features: usize,
) -> Result<()> {
    let block_bytes = match quant {
        GgufKQuant::Q4K => Q4_K_BLOCK_BYTES,
        GgufKQuant::Q6K => Q6_K_BLOCK_BYTES,
    };
    let expected = out_features
        .checked_mul(in_features / K_BLOCK_VALUES)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(|| Error::backend("Laguna XS K-quant weight size overflow"))?;
    if weights.len() != expected {
        return Err(Error::backend(format!(
            "Laguna XS {quant:?} weights must contain {expected} bytes, got {}",
            weights.len()
        )));
    }
    Ok(())
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::backend(format!("Laguna XS {label} exceeds Metal u32")))
}

#[cfg(test)]
mod tests {
    use crate::{Backend, GgufKQuant, MetalBackend};
    use common::F32Tensor;
    use gguf::GgufQuantBlockKind;

    use super::{K_BLOCK_VALUES, Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES};

    #[test]
    fn q4_k_matvec_matches_a_known_constant_block() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let row_count = 2;
        let out_features = 4;
        let weights = repeat_block(&q4_block(2), out_features);
        let input = F32Tensor::new(
            vec![1.0 / K_BLOCK_VALUES as f32; row_count * K_BLOCK_VALUES],
            [row_count, K_BLOCK_VALUES],
        )
        .unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();

        let output = backend
            .gguf_k_matvec_device(
                GgufKQuant::Q4K,
                &weights,
                &input,
                row_count,
                K_BLOCK_VALUES,
                out_features,
            )
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();

        assert_eq!(output.dims(), &[row_count, out_features]);
        for value in output.values() {
            assert!((value - 2.0).abs() <= 1e-5, "Q4_K value={value}");
        }
    }

    #[test]
    fn q6_k_matvec_matches_a_known_constant_block() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let row_count = 2;
        let out_features = 3;
        let weights = repeat_block(&q6_block(1), out_features);
        let input = F32Tensor::new(
            vec![1.0 / K_BLOCK_VALUES as f32; row_count * K_BLOCK_VALUES],
            [row_count, K_BLOCK_VALUES],
        )
        .unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();

        let output = backend
            .gguf_k_matvec_device(
                GgufKQuant::Q6K,
                &weights,
                &input,
                row_count,
                K_BLOCK_VALUES,
                out_features,
            )
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();

        assert_eq!(output.dims(), &[row_count, out_features]);
        for value in output.values() {
            assert!((value - 1.0).abs() <= 1e-5, "Q6_K value={value}");
        }
    }

    #[test]
    fn q4_k_prefill_mma_matches_the_rust_decoder() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let row_count = 8;
        let out_features = 32;
        let block = patterned_q4_block();
        let decoded = GgufQuantBlockKind::Q4K.decode_block(&block).unwrap();
        let weights = repeat_block(&block, out_features);
        let input_values = (0..row_count)
            .flat_map(|row| {
                patterned_input()
                    .into_iter()
                    .map(move |value| value * (row + 1) as f32 / row_count as f32)
            })
            .collect::<Vec<_>>();
        let expected = (0..row_count)
            .map(|row| {
                dot(
                    &decoded,
                    &input_values[row * K_BLOCK_VALUES..(row + 1) * K_BLOCK_VALUES],
                )
            })
            .collect::<Vec<_>>();
        let input = F32Tensor::new(input_values, [row_count, K_BLOCK_VALUES]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();

        let output = backend
            .gguf_k_matvec_device(
                GgufKQuant::Q4K,
                &weights,
                &input,
                row_count,
                K_BLOCK_VALUES,
                out_features,
            )
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();

        assert_eq!(output.dims(), &[row_count, out_features]);
        for row in 0..row_count {
            for column in 0..out_features {
                assert_prefill_close(
                    output.values()[row * out_features + column],
                    expected[row],
                    "Q4_K prefill MMA",
                );
            }
        }
    }

    #[test]
    fn q6_k_prefill_mma_matches_the_rust_decoder() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let row_count = 8;
        let out_features = 32;
        let block = patterned_q6_block();
        let decoded = GgufQuantBlockKind::Q6K.decode_block(&block).unwrap();
        let weights = repeat_block(&block, out_features);
        let input_values = (0..row_count)
            .flat_map(|row| {
                patterned_input()
                    .into_iter()
                    .map(move |value| value * (row + 1) as f32 / row_count as f32)
            })
            .collect::<Vec<_>>();
        let expected = (0..row_count)
            .map(|row| {
                dot(
                    &decoded,
                    &input_values[row * K_BLOCK_VALUES..(row + 1) * K_BLOCK_VALUES],
                )
            })
            .collect::<Vec<_>>();
        let input = F32Tensor::new(input_values, [row_count, K_BLOCK_VALUES]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();

        let output = backend
            .gguf_k_matvec_device(
                GgufKQuant::Q6K,
                &weights,
                &input,
                row_count,
                K_BLOCK_VALUES,
                out_features,
            )
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();

        assert_eq!(output.dims(), &[row_count, out_features]);
        for row in 0..row_count {
            for column in 0..out_features {
                assert_prefill_close(
                    output.values()[row * out_features + column],
                    expected[row],
                    "Q6_K prefill MMA",
                );
            }
        }
    }

    #[test]
    fn q4_k_prefill_mma_fuses_one_or_two_residuals() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let row_count = 8;
        let out_features = 32;
        let weights = repeat_block(&q4_block(2), out_features);
        let input = F32Tensor::new(
            vec![1.0 / K_BLOCK_VALUES as f32; row_count * K_BLOCK_VALUES],
            [row_count, K_BLOCK_VALUES],
        )
        .unwrap();
        let residual_a = F32Tensor::new(
            vec![3.0; row_count * out_features],
            [row_count, out_features],
        )
        .unwrap();
        let residual_b = F32Tensor::new(
            vec![5.0; row_count * out_features],
            [row_count, out_features],
        )
        .unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let residual_a = backend
            .device_upload_f32_tensor(&residual_a)
            .unwrap()
            .unwrap();
        let residual_b = backend
            .device_upload_f32_tensor(&residual_b)
            .unwrap()
            .unwrap();

        let one = backend
            .laguna_xs_k_matvec_add_device(
                GgufKQuant::Q4K,
                &weights,
                &input,
                &residual_a,
                row_count,
                K_BLOCK_VALUES,
                out_features,
            )
            .unwrap()
            .unwrap();
        let two = backend
            .laguna_xs_k_matvec_add2_device(
                GgufKQuant::Q4K,
                &weights,
                &input,
                &residual_a,
                &residual_b,
                row_count,
                K_BLOCK_VALUES,
                out_features,
            )
            .unwrap()
            .unwrap();

        let one = backend.device_download_f32_tensor(&one).unwrap();
        let two = backend.device_download_f32_tensor(&two).unwrap();
        assert!(one
            .values()
            .iter()
            .all(|value| (*value - 5.0).abs() <= 1e-5));
        assert!(two
            .values()
            .iter()
            .all(|value| (*value - 10.0).abs() <= 1e-5));
    }

    #[test]
    fn q4_k_matvec_matches_the_rust_decoder_for_packed_scales_and_minima() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let weights = patterned_q4_block();
        let input_values = patterned_input();
        let expected = dot(
            &GgufQuantBlockKind::Q4K.decode_block(&weights).unwrap(),
            &input_values,
        );
        let input = F32Tensor::new(input_values, [1, K_BLOCK_VALUES]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();

        let output = backend
            .gguf_k_matvec_device(GgufKQuant::Q4K, &weights, &input, 1, K_BLOCK_VALUES, 1)
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();

        assert_close(output.values()[0], expected, "Q4_K patterned matvec");
    }

    #[test]
    fn q4_decode_matvec_matches_multiple_packed_blocks() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let in_features = K_BLOCK_VALUES * 2;
        let out_features = 3;
        let input_values = (0..in_features)
            .map(|index| ((index % 17) as f32 - 8.0) / 9.0)
            .collect::<Vec<_>>();
        let mut weights = Vec::new();
        let mut expected = Vec::new();
        for row in 0..out_features {
            let blocks = [q4_block((row + 1) as u8), q4_block((row + 5) as u8)];
            expected.push(
                blocks
                    .iter()
                    .enumerate()
                    .map(|(block_index, block)| {
                        dot(
                            &GgufQuantBlockKind::Q4K.decode_block(block).unwrap(),
                            &input_values
                                [block_index * K_BLOCK_VALUES..(block_index + 1) * K_BLOCK_VALUES],
                        )
                    })
                    .sum::<f32>(),
            );
            weights.extend(blocks.into_iter().flatten());
        }
        let input = F32Tensor::new(input_values, [1, in_features]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();

        let output = backend
            .gguf_k_matvec_device(
                GgufKQuant::Q4K,
                &weights,
                &input,
                1,
                in_features,
                out_features,
            )
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();

        assert_eq!(output.dims(), &[1, out_features]);
        for (index, (actual, expected)) in output.values().iter().zip(expected.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                &format!("Q4_K interleaved decode row {index}"),
            );
        }
    }

    #[test]
    fn q6_k_matvec_matches_the_rust_decoder_for_all_packed_bit_planes() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let weights = patterned_q6_block();
        let input_values = patterned_input();
        let expected = dot(
            &GgufQuantBlockKind::Q6K.decode_block(&weights).unwrap(),
            &input_values,
        );
        let input = F32Tensor::new(input_values, [1, K_BLOCK_VALUES]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();

        let output = backend
            .gguf_k_matvec_device(GgufKQuant::Q6K, &weights, &input, 1, K_BLOCK_VALUES, 1)
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();

        assert_close(output.values()[0], expected, "Q6_K patterned matvec");
    }

    #[test]
    fn q6_decode_matvec_matches_multiple_packed_blocks() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let in_features = K_BLOCK_VALUES * 2;
        let out_features = 4;
        let input_values = (0..in_features)
            .map(|index| ((index % 19) as f32 - 9.0) / 10.0)
            .collect::<Vec<_>>();
        let mut weights = Vec::new();
        let mut expected = Vec::new();
        for row in 0..out_features {
            let blocks = [q6_block((row + 1) as u8), q6_block((row + 5) as u8)];
            expected.push(
                blocks
                    .iter()
                    .enumerate()
                    .map(|(block_index, block)| {
                        dot(
                            &GgufQuantBlockKind::Q6K.decode_block(block).unwrap(),
                            &input_values
                                [block_index * K_BLOCK_VALUES..(block_index + 1) * K_BLOCK_VALUES],
                        )
                    })
                    .sum::<f32>(),
            );
            weights.extend(blocks.into_iter().flatten());
        }
        let input = F32Tensor::new(input_values, [1, in_features]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();

        let output = backend
            .gguf_k_matvec_device(
                GgufKQuant::Q6K,
                &weights,
                &input,
                1,
                in_features,
                out_features,
            )
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();

        assert_eq!(output.dims(), &[1, out_features]);
        for (index, (actual, expected)) in output.values().iter().zip(expected.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                &format!("Q6_K packed decode row {index}"),
            );
        }
    }

    #[test]
    fn decode_matvec_fuses_one_or_two_residuals() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let out_features = 4;
        let input = F32Tensor::new(
            vec![1.0 / K_BLOCK_VALUES as f32; K_BLOCK_VALUES],
            [1, K_BLOCK_VALUES],
        )
        .unwrap();
        let residual_a = F32Tensor::new(vec![1.0; out_features], [1, out_features]).unwrap();
        let residual_b = F32Tensor::new(vec![2.0; out_features], [1, out_features]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let residual_a = backend
            .device_upload_f32_tensor(&residual_a)
            .unwrap()
            .unwrap();
        let residual_b = backend
            .device_upload_f32_tensor(&residual_b)
            .unwrap()
            .unwrap();

        for (quant, weights, projected) in [
            (
                GgufKQuant::Q4K,
                repeat_block(&q4_block(2), out_features),
                2.0_f32,
            ),
            (
                GgufKQuant::Q6K,
                repeat_block(&q6_block(1), out_features),
                1.0_f32,
            ),
        ] {
            let output = backend
                .laguna_xs_k_matvec_add_device(
                    quant,
                    &weights,
                    &input,
                    &residual_a,
                    1,
                    K_BLOCK_VALUES,
                    out_features,
                )
                .unwrap()
                .unwrap();
            let output = backend.device_download_f32_tensor(&output).unwrap();
            for value in output.values() {
                assert_close(*value, projected + 1.0, "fused first residual");
            }

            let output = backend
                .laguna_xs_k_matvec_add2_device(
                    quant,
                    &weights,
                    &input,
                    &residual_a,
                    &residual_b,
                    1,
                    K_BLOCK_VALUES,
                    out_features,
                )
                .unwrap()
                .unwrap();
            let output = backend.device_download_f32_tensor(&output).unwrap();
            for value in output.values() {
                assert_close(*value, projected + 3.0, "fused second residual");
            }
        }
    }

    #[test]
    fn decode_gate_up_fuses_swiglu() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let out_features = 4;
        let expected = 2.0 / (1.0 + (-1.0_f32).exp());

        for in_features in [K_BLOCK_VALUES, 2_048] {
            let input = F32Tensor::new(
                vec![1.0 / in_features as f32; in_features],
                [1, in_features],
            )
            .unwrap();
            let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
            let blocks_per_row = in_features / K_BLOCK_VALUES;
            let gate = repeat_block(&q4_block(1), out_features * blocks_per_row);
            let up = repeat_block(&q4_block(2), out_features * blocks_per_row);

            let output = backend
                .laguna_xs_q4_gate_up_swiglu_device(
                    &gate,
                    &up,
                    &input,
                    1,
                    in_features,
                    out_features,
                )
                .unwrap()
                .unwrap();
            let output = backend.device_download_f32_tensor(&output).unwrap();

            assert_eq!(output.dims(), &[1, out_features]);
            for value in output.values() {
                assert_close(*value, expected, "fused decode SwiGLU");
            }
        }
    }

    #[test]
    fn fused_attention_projections_match_independent_q4_and_q6_values() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let query_features = 4;
        let key_features = 2;
        let value_features = 2;
        let gate_features = 2;
        for in_features in [K_BLOCK_VALUES, 2_048] {
            let input = F32Tensor::new(
                vec![1.0 / in_features as f32; in_features],
                [1, in_features],
            )
            .unwrap();
            let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
            let blocks_per_row = in_features / K_BLOCK_VALUES;
            let query = repeat_block(&q4_block(2), query_features * blocks_per_row);
            let key = repeat_block(&q4_block(3), key_features * blocks_per_row);
            let gate = repeat_block(&q4_block(4), gate_features * blocks_per_row);

            for (value, value_quant, expected_value) in [
                (
                    repeat_block(&q4_block(5), value_features * blocks_per_row),
                    GgufKQuant::Q4K,
                    5.0_f32,
                ),
                (
                    repeat_block(&q6_block(1), value_features * blocks_per_row),
                    GgufKQuant::Q6K,
                    1.0_f32,
                ),
            ] {
                let [query_output, key_output, value_output, gate_output] = backend
                    .laguna_xs_attention_projections_device(
                        &query,
                        &key,
                        &value,
                        value_quant,
                        &gate,
                        &input,
                        1,
                        in_features,
                        query_features,
                        key_features,
                        value_features,
                        gate_features,
                    )
                    .unwrap()
                    .unwrap();
                for (output, expected) in [
                    (query_output, 2.0_f32),
                    (key_output, 3.0_f32),
                    (value_output, expected_value),
                    (gate_output, 4.0_f32),
                ] {
                    let output = backend.device_download_f32_tensor(&output).unwrap();
                    for value in output.values() {
                        assert_close(*value, expected, "fused attention projection");
                    }
                }
            }
        }
    }

    #[test]
    fn q4_k_embedding_gathers_only_requested_rows() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let mut weights = q4_block(2);
        weights.extend(q4_block(3));

        let output = backend
            .q4_k_embedding_device(&weights, &[1, 0], &[1, 2], 2, K_BLOCK_VALUES)
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();

        assert_eq!(output.dims(), &[1, 2, K_BLOCK_VALUES]);
        assert!(output.values()[..K_BLOCK_VALUES]
            .iter()
            .all(|value| (*value - 3.0).abs() <= 1e-5));
        assert!(output.values()[K_BLOCK_VALUES..]
            .iter()
            .all(|value| (*value - 2.0).abs() <= 1e-5));
    }

    #[test]
    fn q4_gate_up_routed_moe_supports_q4_and_q6_down_projections() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let expert_count = 2;
        let top_k = 2;
        let width = K_BLOCK_VALUES;
        let input = F32Tensor::new(vec![1.0 / width as f32; width], [1, width]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let logits = F32Tensor::new(vec![0.0_f32; expert_count], [1, expert_count]).unwrap();
        let logits = backend.device_upload_f32_tensor(&logits).unwrap().unwrap();
        let routing = backend
            .moe_router_topk_resident_device(&logits, &vec![0.0; expert_count], top_k, true, 1.0)
            .unwrap()
            .unwrap();
        let gate = q4_expert_matrix(&[1, 2], width, width);
        let up = q4_expert_matrix(&[1, 2], width, width);

        let expected = 128.0 / (1.0 + (-1.0_f32).exp()) + 512.0 / (1.0 + (-2.0_f32).exp());
        for (down, down_quant) in [
            (q4_expert_matrix(&[1, 1], width, width), GgufKQuant::Q4K),
            (q6_expert_matrix(&[1, 1], width, width), GgufKQuant::Q6K),
        ] {
            let output = backend
                .laguna_xs_gguf_moe_device(
                    &gate, &up, &down, down_quant, &input, &routing, width, width, width,
                )
                .unwrap()
                .unwrap();
            let output = backend.device_download_f32_tensor(&output).unwrap();

            assert_eq!(output.dims(), &[1, width]);
            for value in output.values() {
                let tolerance = expected.abs() * 1e-5;
                assert!(
                    (value - expected).abs() <= tolerance,
                    "MoE {down_quant:?} value={value}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn top8_routed_moe_parallel_down_matches_the_known_sum() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let expert_count = 8;
        let top_k = 8;
        let in_features = K_BLOCK_VALUES;
        let intermediate_features = K_BLOCK_VALUES * 2;
        let out_features = K_BLOCK_VALUES;
        let input = F32Tensor::new(
            vec![1.0 / in_features as f32; in_features],
            [1, in_features],
        )
        .unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let logits = F32Tensor::new(vec![0.0_f32; expert_count], [1, expert_count]).unwrap();
        let logits = backend.device_upload_f32_tensor(&logits).unwrap().unwrap();
        let routing = backend
            .moe_router_topk_resident_device(&logits, &vec![0.0; expert_count], top_k, true, 1.0)
            .unwrap()
            .unwrap();
        let expert_values = (1_u8..=expert_count as u8).collect::<Vec<_>>();
        let gate = q4_expert_matrix(&expert_values, in_features, intermediate_features);
        let up = q4_expert_matrix(&expert_values, in_features, intermediate_features);
        let expected = expert_values
            .iter()
            .map(|value| {
                let value = f32::from(*value);
                (intermediate_features as f32 / top_k as f32) * value * value
                    / (1.0 + (-value).exp())
            })
            .sum::<f32>();

        for (down, down_quant) in [
            (
                q4_expert_matrix(&vec![1; expert_count], intermediate_features, out_features),
                GgufKQuant::Q4K,
            ),
            (
                q6_expert_matrix(&vec![1; expert_count], intermediate_features, out_features),
                GgufKQuant::Q6K,
            ),
        ] {
            let output = backend
                .laguna_xs_gguf_moe_device(
                    &gate,
                    &up,
                    &down,
                    down_quant,
                    &input,
                    &routing,
                    in_features,
                    intermediate_features,
                    out_features,
                )
                .unwrap()
                .unwrap();
            let output = backend.device_download_f32_tensor(&output).unwrap();

            assert_eq!(output.dims(), &[1, out_features]);
            for value in output.values() {
                let tolerance = expected.abs() * 1e-5;
                assert!(
                    (value - expected).abs() <= tolerance,
                    "top-8 MoE {down_quant:?} value={value}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn top8_routed_moe_prefill_matches_the_known_sum() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let token_count = 8;
        let expert_count = 256;
        let top_k = 8;
        let width = K_BLOCK_VALUES;
        let input = F32Tensor::new(
            (0..token_count)
                .flat_map(|token| {
                    let value = (token + 1) as f32 / width as f32;
                    std::iter::repeat_n(value, width)
                })
                .collect(),
            [token_count, width],
        )
        .unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let logits = F32Tensor::new(
            vec![0.0_f32; token_count * expert_count],
            [token_count, expert_count],
        )
        .unwrap();
        let logits = backend.device_upload_f32_tensor(&logits).unwrap().unwrap();
        let routing = backend
            .moe_router_topk_resident_device(&logits, &vec![0.0; expert_count], top_k, true, 1.0)
            .unwrap()
            .unwrap();
        let expert_values = vec![1_u8; expert_count];
        let gate = q4_expert_matrix(&expert_values, width, width);
        let up = q4_expert_matrix(&expert_values, width, width);
        let expected = (1..=token_count)
            .map(|value| {
                let value = value as f32;
                width as f32 * value * value / (1.0 + (-value).exp())
            })
            .collect::<Vec<_>>();

        for (down, down_quant) in [
            (
                q4_expert_matrix(&expert_values, width, width),
                GgufKQuant::Q4K,
            ),
            (
                q6_expert_matrix(&expert_values, width, width),
                GgufKQuant::Q6K,
            ),
        ] {
            let output = backend
                .laguna_xs_gguf_moe_device(
                    &gate, &up, &down, down_quant, &input, &routing, width, width, width,
                )
                .unwrap()
                .unwrap();
            let output = backend.device_download_f32_tensor(&output).unwrap();

            assert_eq!(output.dims(), &[token_count, width]);
            for token in 0..token_count {
                for value in &output.values()[token * width..(token + 1) * width] {
                    let expected = expected[token];
                    let tolerance = expected.abs() * 3e-3;
                    assert!(
                        (value - expected).abs() <= tolerance,
                        "top-8 prefill MoE {down_quant:?} token={token}, value={value}, expected={expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn specialized_decode_top8_matches_the_reference_router() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let expert_count = 256;
        let top_k = 8;
        let logits = (0..expert_count)
            .map(|expert| (expert as f32 / 64.0) - 2.0)
            .collect::<Vec<_>>();
        let logits = F32Tensor::new(logits, [1, expert_count]).unwrap();
        let logits = backend.device_upload_f32_tensor(&logits).unwrap().unwrap();
        let mut correction_bias = vec![0.0_f32; expert_count];
        correction_bias[3] = 2.0;
        correction_bias[7] = 1.9;
        correction_bias[11] = 1.8;

        let reference = backend
            .moe_router_topk_resident_device(&logits, &correction_bias, top_k, true, 1.25)
            .unwrap()
            .unwrap();
        let specialized = backend
            .laguna_xs_router_topk_device(&logits, &correction_bias, top_k, true, 1.25)
            .unwrap()
            .unwrap();

        let reference_ids = backend
            .moe_router_expert_ids_device(&reference)
            .unwrap()
            .unwrap();
        let specialized_ids = backend
            .moe_router_expert_ids_device(&specialized)
            .unwrap()
            .unwrap();
        assert_eq!(specialized_ids, reference_ids);

        let reference_weights = unsafe {
            std::slice::from_raw_parts(reference.expert_weights.contents().cast::<f32>(), top_k)
                .to_vec()
        };
        let specialized_weights = unsafe {
            std::slice::from_raw_parts(specialized.expert_weights.contents().cast::<f32>(), top_k)
                .to_vec()
        };
        for (actual, expected) in specialized_weights.iter().zip(&reference_weights) {
            assert_close(*actual, *expected, "specialized router weight");
        }
    }

    #[test]
    fn vectorized_decode_rms_norm_matches_the_reference_kernel() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let hidden_size = 2_048;
        let eps = 1e-6;
        let input = F32Tensor::new(
            (0..hidden_size)
                .map(|index| ((index % 31) as f32 - 15.0) / 17.0)
                .collect(),
            [1, hidden_size],
        )
        .unwrap();
        let weight = F32Tensor::new(
            (0..hidden_size)
                .map(|index| 0.75 + (index % 19) as f32 / 32.0)
                .collect(),
            [hidden_size],
        )
        .unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let reference = backend
            .rms_norm_device(&input, &weight, eps)
            .unwrap()
            .unwrap();
        let specialized = backend
            .laguna_xs_rms_norm_device(&input, &weight, eps)
            .unwrap()
            .unwrap();
        let reference = backend.device_download_f32_tensor(&reference).unwrap();
        let specialized = backend.device_download_f32_tensor(&specialized).unwrap();

        assert_eq!(specialized.dims(), &[1, hidden_size]);
        for (actual, expected) in specialized.values().iter().zip(reference.values()) {
            assert_close(*actual, *expected, "vectorized decode RMSNorm");
        }
    }

    #[test]
    fn vectorized_decode_qk_norm_rope_matches_the_reference_pair() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let head_dim = 128;
        let key_heads = 8;
        let eps = 1e-6;
        let query_norm = F32Tensor::new(
            (0..head_dim)
                .map(|index| 0.75 + (index % 13) as f32 / 32.0)
                .collect(),
            [head_dim],
        )
        .unwrap();
        let key_norm = F32Tensor::new(
            (0..head_dim)
                .map(|index| 0.9 + (index % 7) as f32 / 40.0)
                .collect(),
            [head_dim],
        )
        .unwrap();

        for (query_heads, rotary_dim) in [(48, 64), (64, 128)] {
            let query = F32Tensor::new(
                (0..query_heads * head_dim)
                    .map(|index| ((index % 29) as f32 - 14.0) / 17.0)
                    .collect(),
                [1, 1, query_heads, head_dim],
            )
            .unwrap();
            let key = F32Tensor::new(
                (0..key_heads * head_dim)
                    .map(|index| ((index % 23) as f32 - 11.0) / 13.0)
                    .collect(),
                [1, 1, key_heads, head_dim],
            )
            .unwrap();
            let query = backend.device_upload_f32_tensor(&query).unwrap().unwrap();
            let key = backend.device_upload_f32_tensor(&key).unwrap().unwrap();
            let inverse_frequency = (0..rotary_dim / 2)
                .map(|index| 1.0 / (10_000.0_f32).powf((2 * index) as f32 / rotary_dim as f32))
                .collect::<Vec<_>>();
            let table = backend
                .prepare_rope_table(&inverse_frequency, rotary_dim, 1.125)
                .unwrap()
                .unwrap();

            let reference = backend
                .laguna_qk_rms_norm_rope_pair_device(
                    &query,
                    &key,
                    &query_norm,
                    &key_norm,
                    eps,
                    37,
                    &table,
                )
                .unwrap()
                .unwrap();
            let vectorized = backend
                .laguna_xs_qk_rms_norm_rope_pair_device(
                    &query,
                    &key,
                    &query_norm,
                    &key_norm,
                    eps,
                    37,
                    &table,
                )
                .unwrap()
                .unwrap();

            for ((actual, expected), label) in [
                ((vectorized.0, reference.0), "vectorized query norm+RoPE"),
                ((vectorized.1, reference.1), "vectorized key norm+RoPE"),
            ] {
                let actual = backend.device_download_f32_tensor(&actual).unwrap();
                let expected = backend.device_download_f32_tensor(&expected).unwrap();
                assert_eq!(actual.dims(), expected.dims());
                for (actual, expected) in actual.values().iter().zip(expected.values()) {
                    assert_close(*actual, *expected, label);
                }
            }
        }
    }

    fn repeat_block(block: &[u8], row_count: usize) -> Vec<u8> {
        (0..row_count).flat_map(|_| block.iter().copied()).collect()
    }

    fn q4_block(quant: u8) -> Vec<u8> {
        assert!(quant <= 15);
        let mut block = vec![0_u8; Q4_K_BLOCK_BYTES];
        block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        block[4..16].fill(1);
        block[16..].fill(quant | (quant << 4));
        block
    }

    fn q4_expert_matrix(expert_quants: &[u8], in_features: usize, out_features: usize) -> Vec<u8> {
        assert!(in_features.is_multiple_of(K_BLOCK_VALUES));
        let blocks_per_expert = out_features * (in_features / K_BLOCK_VALUES);
        let mut weights =
            Vec::with_capacity(expert_quants.len() * blocks_per_expert * Q4_K_BLOCK_BYTES);
        for quant in expert_quants {
            let block = q4_block(*quant);
            for _ in 0..blocks_per_expert {
                weights.extend_from_slice(&block);
            }
        }
        weights
    }

    fn q6_block(value: u8) -> Vec<u8> {
        assert!(value <= 31);
        let quant = value + 32;
        let low = quant & 0x0f;
        let high = quant >> 4;
        let mut block = vec![0_u8; Q6_K_BLOCK_BYTES];
        block[..128].fill(low | (low << 4));
        block[128..192].fill(high | (high << 2) | (high << 4) | (high << 6));
        block[192..208].fill(1);
        block[208..].copy_from_slice(&0x3c00_u16.to_le_bytes());
        block
    }

    fn q6_expert_matrix(expert_values: &[u8], in_features: usize, out_features: usize) -> Vec<u8> {
        assert!(in_features.is_multiple_of(K_BLOCK_VALUES));
        let blocks_per_expert = out_features * (in_features / K_BLOCK_VALUES);
        let mut weights =
            Vec::with_capacity(expert_values.len() * blocks_per_expert * Q6_K_BLOCK_BYTES);
        for value in expert_values {
            let block = q6_block(*value);
            for _ in 0..blocks_per_expert {
                weights.extend_from_slice(&block);
            }
        }
        weights
    }

    fn patterned_q4_block() -> Vec<u8> {
        let group_scales = [1_u8, 7, 15, 23, 31, 39, 47, 63];
        let group_minima = [0_u8, 3, 9, 15, 17, 33, 49, 63];
        let mut scales = [0_u8; 12];
        for group in 0..4 {
            scales[group] = group_scales[group];
            scales[group + 4] = group_minima[group];
        }
        for group in 4..8 {
            scales[group + 4] = (group_scales[group] & 0x0f) | ((group_minima[group] & 0x0f) << 4);
            scales[group - 4] |= (group_scales[group] >> 4) << 6;
            scales[group] |= (group_minima[group] >> 4) << 6;
        }

        let mut block = vec![0_u8; Q4_K_BLOCK_BYTES];
        block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        block[2..4].copy_from_slice(&0x3800_u16.to_le_bytes());
        block[4..16].copy_from_slice(&scales);
        for pair in 0..4 {
            for lane in 0..32 {
                let low = ((lane + pair) & 15) as u8;
                let high = ((31 - lane + pair) & 15) as u8;
                block[16 + pair * 32 + lane] = low | (high << 4);
            }
        }
        block
    }

    fn patterned_q6_block() -> Vec<u8> {
        let mut block = vec![0_u8; Q6_K_BLOCK_BYTES];
        for (index, value) in block[..128].iter_mut().enumerate() {
            *value = ((index * 29 + 7) & 0xff) as u8;
        }
        for (index, value) in block[128..192].iter_mut().enumerate() {
            *value = ((index * 53 + 11) & 0xff) as u8;
        }
        for (index, value) in block[192..208].iter_mut().enumerate() {
            *value = (index as i8 - 8) as u8;
        }
        block[208..].copy_from_slice(&0x3800_u16.to_le_bytes());
        block
    }

    fn patterned_input() -> Vec<f32> {
        (0..K_BLOCK_VALUES)
            .map(|index| ((index % 11) as f32 - 5.0) / 7.0)
            .collect()
    }

    fn dot(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(left, right)| left * right)
            .sum()
    }

    fn assert_close(actual: f32, expected: f32, label: &str) {
        let tolerance = 1e-4_f32.max(expected.abs() * 1e-5);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }

    fn assert_prefill_close(actual: f32, expected: f32, label: &str) {
        let tolerance = 2e-2_f32.max(expected.abs() * 2e-3);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}
