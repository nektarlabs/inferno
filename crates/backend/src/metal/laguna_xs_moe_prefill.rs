use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
};

use ::metal::{Buffer, CommandBufferRef, ComputePipelineState, Device};
use common::{Error, Result};

use crate::{DeviceRouterTopK, GgufKQuant};

use super::{
    arena::MetalArena,
    buffers::{require_f32_capacity, u8_buffer_no_copy},
    command::{
        encode_1d_threadgroups_args, encode_3d_threadgroups_args,
        encode_threadgroups_indirect_args, KernelArg,
    },
    laguna_views::MetalLagunaViews,
    library::MetalLibrary,
    pipeline::compute_pipeline,
};

const BLOCK_VALUES: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q6_K_BLOCK_BYTES: usize = 210;
const TOP_K: usize = 8;
const EXPERT_COUNT: usize = 256;
const OUTPUT_TILE: usize = 32;
const SMALL_ASSIGNMENT_TILE: usize = 8;
const MEDIUM_ASSIGNMENT_TILE: usize = 16;
const LARGE_ASSIGNMENT_TILE: usize = 32;
const MATMUL_THREADS: usize = 128;
const MAP_THREADS: usize = 256;
const COMBINE_THREADS: usize = 256;
// Staging and result storage are used at different times. Reusing the same
// threadgroup memory keeps more prompt GEMM threadgroups resident on Metal.
const GATE_UP_SHARED_BYTES: usize = 8 * 1024;
const DOWN_SHARED_BYTES: usize = 4 * 1024;
const INDIRECT_ARGUMENT_U32S: usize = 18;
const MIN_TOKENS: usize = 4;
const KERNEL_SOURCE: &str = include_str!("kernels/laguna_xs_moe_prefill_kernels.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct WeightKey {
    address: usize,
    byte_len: usize,
}

struct WeightBuffer {
    storage: Buffer,
    byte_offset: usize,
}

/// Expert-major prompt path for Laguna XS routed experts.
///
/// The top-8 token-major routing result is compacted into per-expert assignment
/// tiles. Each selected expert then processes up to 32 prompt tokens at once.
/// Single-token decode remains in `MetalLagunaXs`.
pub(crate) struct MetalLagunaXsMoePrefill {
    pipelines: Mutex<Option<Pipelines>>,
    arena: MetalArena,
    weights: Mutex<HashMap<WeightKey, Buffer>>,
    laguna_views: Arc<MetalLagunaViews>,
}

struct Pipelines {
    build_map: ComputePipelineState,
    q4_gate_up_small: ComputePipelineState,
    q4_gate_up_medium: ComputePipelineState,
    q4_gate_up_large: ComputePipelineState,
    q4_down_small: ComputePipelineState,
    q4_down_medium: ComputePipelineState,
    q4_down_large: ComputePipelineState,
    q6_down_small: ComputePipelineState,
    q6_down_medium: ComputePipelineState,
    q6_down_large: ComputePipelineState,
    combine: ComputePipelineState,
}

impl MetalLagunaXsMoePrefill {
    pub(crate) fn new(arena: MetalArena, laguna_views: Arc<MetalLagunaViews>) -> Self {
        Self {
            pipelines: Mutex::new(None),
            arena,
            weights: Mutex::new(HashMap::new()),
            laguna_views,
        }
    }

    pub(crate) fn supports(routing: &DeviceRouterTopK) -> bool {
        routing.token_count >= MIN_TOKENS
            && routing.top_k == TOP_K
            && routing.expert_count == EXPERT_COUNT
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
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
        self.validate(
            gate_weights,
            up_weights,
            down_weights,
            down_quant,
            input,
            input_len,
            routing,
            in_features,
            intermediate_features,
            out_features,
        )?;

        let token_count = routing.token_count;
        let assignment_count = routing.assignment_count()?;
        let expert_counts = self.arena.empty_u32(EXPERT_COUNT)?;
        let assignment_map = self.arena.empty_u32(
            EXPERT_COUNT
                .checked_mul(token_count)
                .ok_or_else(|| Error::backend("Laguna XS assignment map overflow"))?,
        )?;
        let small_work_tiles = self.arena.empty_u32(
            EXPERT_COUNT
                .checked_mul(4)
                .ok_or_else(|| Error::backend("Laguna XS small work tile buffer overflow"))?,
        )?;
        let medium_work_tiles = self.arena.empty_u32(
            EXPERT_COUNT
                .checked_mul(4)
                .ok_or_else(|| Error::backend("Laguna XS medium work tile buffer overflow"))?,
        )?;
        let maximum_large_work_tiles = EXPERT_COUNT
            .checked_mul(token_count.div_ceil(LARGE_ASSIGNMENT_TILE))
            .ok_or_else(|| Error::backend("Laguna XS work tile count overflow"))?;
        let large_work_tiles = self.arena.empty_u32(
            maximum_large_work_tiles
                .checked_mul(4)
                .ok_or_else(|| Error::backend("Laguna XS large work tile buffer overflow"))?,
        )?;
        let indirect_arguments = self.arena.empty_u32(INDIRECT_ARGUMENT_U32S)?;
        let intermediate = self.arena.empty_f16(
            assignment_count
                .checked_mul(intermediate_features)
                .ok_or_else(|| Error::backend("Laguna XS intermediate buffer overflow"))?,
        )?;
        let assignment_output = self.arena.empty_f32(
            assignment_count
                .checked_mul(out_features)
                .ok_or_else(|| Error::backend("Laguna XS assignment output overflow"))?,
        )?;
        let output_len = token_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Laguna XS MoE output overflow"))?;
        let output = self.arena.empty_f32(output_len)?;
        let pipelines = self.pipelines(device)?;
        let pipelines = pipelines.as_ref().ok_or_else(|| {
            Error::backend("Laguna XS expert-prefill pipelines were not initialized")
        })?;

        encode_3d_threadgroups_args(
            command_buffer,
            &pipelines.build_map,
            &[
                KernelArg::Buffer(&routing.expert_ids),
                KernelArg::Buffer(&expert_counts),
                KernelArg::Buffer(&assignment_map),
                KernelArg::Buffer(&small_work_tiles),
                KernelArg::Buffer(&medium_work_tiles),
                KernelArg::Buffer(&large_work_tiles),
                KernelArg::Buffer(&indirect_arguments),
                KernelArg::U32(as_u32(SMALL_ASSIGNMENT_TILE, "small assignment tile")?),
                KernelArg::U32(as_u32(MEDIUM_ASSIGNMENT_TILE, "medium assignment tile")?),
                KernelArg::U32(as_u32(token_count, "token count")?),
                KernelArg::U32(as_u32(
                    intermediate_features / OUTPUT_TILE,
                    "gate output tiles",
                )?),
                KernelArg::U32(as_u32(out_features / OUTPUT_TILE, "down output tiles")?),
            ],
            [1, 1, 1],
            MAP_THREADS,
            0,
        )?;

        let gate = self.weight_buffer(device, gate_weights)?;
        let up = self.weight_buffer(device, up_weights)?;
        let down = self.weight_buffer(device, down_weights)?;
        let gate_row_bytes = quantized_row_bytes(in_features, Q4_K_BLOCK_BYTES)?;
        let gate_expert_stride = intermediate_features
            .checked_mul(gate_row_bytes)
            .ok_or_else(|| Error::backend("Laguna XS gate expert stride overflow"))?;
        let down_block_bytes = match down_quant {
            GgufKQuant::Q4K => Q4_K_BLOCK_BYTES,
            GgufKQuant::Q6K => Q6_K_BLOCK_BYTES,
        };
        let down_row_bytes = quantized_row_bytes(intermediate_features, down_block_bytes)?;
        let down_expert_stride = out_features
            .checked_mul(down_row_bytes)
            .ok_or_else(|| Error::backend("Laguna XS down expert stride overflow"))?;

        encode_threadgroups_indirect_args(
            command_buffer,
            &pipelines.q4_gate_up_small,
            &[
                KernelArg::BufferOffset(&gate.storage, gate.byte_offset),
                KernelArg::BufferOffset(&up.storage, up.byte_offset),
                KernelArg::Buffer(input),
                KernelArg::Buffer(&assignment_map),
                KernelArg::Buffer(&small_work_tiles),
                KernelArg::Buffer(&routing.expert_weights),
                KernelArg::Buffer(&intermediate),
                KernelArg::U32(as_u32(in_features, "input width")?),
                KernelArg::U32(as_u32(intermediate_features, "intermediate width")?),
                KernelArg::U32(as_u32(token_count, "token count")?),
                KernelArg::U32(as_u32(gate_row_bytes, "gate row bytes")?),
                KernelArg::U32(as_u32(gate_expert_stride, "gate expert stride")?),
            ],
            &indirect_arguments,
            0,
            MATMUL_THREADS,
            GATE_UP_SHARED_BYTES,
        )?;
        encode_threadgroups_indirect_args(
            command_buffer,
            &pipelines.q4_gate_up_medium,
            &[
                KernelArg::BufferOffset(&gate.storage, gate.byte_offset),
                KernelArg::BufferOffset(&up.storage, up.byte_offset),
                KernelArg::Buffer(input),
                KernelArg::Buffer(&assignment_map),
                KernelArg::Buffer(&medium_work_tiles),
                KernelArg::Buffer(&routing.expert_weights),
                KernelArg::Buffer(&intermediate),
                KernelArg::U32(as_u32(in_features, "input width")?),
                KernelArg::U32(as_u32(intermediate_features, "intermediate width")?),
                KernelArg::U32(as_u32(token_count, "token count")?),
                KernelArg::U32(as_u32(gate_row_bytes, "gate row bytes")?),
                KernelArg::U32(as_u32(gate_expert_stride, "gate expert stride")?),
            ],
            &indirect_arguments,
            6 * std::mem::size_of::<u32>(),
            MATMUL_THREADS,
            GATE_UP_SHARED_BYTES,
        )?;
        encode_threadgroups_indirect_args(
            command_buffer,
            &pipelines.q4_gate_up_large,
            &[
                KernelArg::BufferOffset(&gate.storage, gate.byte_offset),
                KernelArg::BufferOffset(&up.storage, up.byte_offset),
                KernelArg::Buffer(input),
                KernelArg::Buffer(&assignment_map),
                KernelArg::Buffer(&large_work_tiles),
                KernelArg::Buffer(&routing.expert_weights),
                KernelArg::Buffer(&intermediate),
                KernelArg::U32(as_u32(in_features, "input width")?),
                KernelArg::U32(as_u32(intermediate_features, "intermediate width")?),
                KernelArg::U32(as_u32(token_count, "token count")?),
                KernelArg::U32(as_u32(gate_row_bytes, "gate row bytes")?),
                KernelArg::U32(as_u32(gate_expert_stride, "gate expert stride")?),
            ],
            &indirect_arguments,
            12 * std::mem::size_of::<u32>(),
            MATMUL_THREADS,
            GATE_UP_SHARED_BYTES,
        )?;
        let (small_down_pipeline, medium_down_pipeline, large_down_pipeline) = match down_quant {
            GgufKQuant::Q4K => (
                &pipelines.q4_down_small,
                &pipelines.q4_down_medium,
                &pipelines.q4_down_large,
            ),
            GgufKQuant::Q6K => (
                &pipelines.q6_down_small,
                &pipelines.q6_down_medium,
                &pipelines.q6_down_large,
            ),
        };
        encode_threadgroups_indirect_args(
            command_buffer,
            small_down_pipeline,
            &[
                KernelArg::BufferOffset(&down.storage, down.byte_offset),
                KernelArg::Buffer(&intermediate),
                KernelArg::Buffer(&assignment_map),
                KernelArg::Buffer(&small_work_tiles),
                KernelArg::Buffer(&assignment_output),
                KernelArg::U32(as_u32(intermediate_features, "intermediate width")?),
                KernelArg::U32(as_u32(out_features, "output width")?),
                KernelArg::U32(as_u32(token_count, "token count")?),
                KernelArg::U32(as_u32(down_row_bytes, "down row bytes")?),
                KernelArg::U32(as_u32(down_expert_stride, "down expert stride")?),
            ],
            &indirect_arguments,
            3 * std::mem::size_of::<u32>(),
            MATMUL_THREADS,
            DOWN_SHARED_BYTES,
        )?;
        encode_threadgroups_indirect_args(
            command_buffer,
            medium_down_pipeline,
            &[
                KernelArg::BufferOffset(&down.storage, down.byte_offset),
                KernelArg::Buffer(&intermediate),
                KernelArg::Buffer(&assignment_map),
                KernelArg::Buffer(&medium_work_tiles),
                KernelArg::Buffer(&assignment_output),
                KernelArg::U32(as_u32(intermediate_features, "intermediate width")?),
                KernelArg::U32(as_u32(out_features, "output width")?),
                KernelArg::U32(as_u32(token_count, "token count")?),
                KernelArg::U32(as_u32(down_row_bytes, "down row bytes")?),
                KernelArg::U32(as_u32(down_expert_stride, "down expert stride")?),
            ],
            &indirect_arguments,
            9 * std::mem::size_of::<u32>(),
            MATMUL_THREADS,
            DOWN_SHARED_BYTES,
        )?;
        encode_threadgroups_indirect_args(
            command_buffer,
            large_down_pipeline,
            &[
                KernelArg::BufferOffset(&down.storage, down.byte_offset),
                KernelArg::Buffer(&intermediate),
                KernelArg::Buffer(&assignment_map),
                KernelArg::Buffer(&large_work_tiles),
                KernelArg::Buffer(&assignment_output),
                KernelArg::U32(as_u32(intermediate_features, "intermediate width")?),
                KernelArg::U32(as_u32(out_features, "output width")?),
                KernelArg::U32(as_u32(token_count, "token count")?),
                KernelArg::U32(as_u32(down_row_bytes, "down row bytes")?),
                KernelArg::U32(as_u32(down_expert_stride, "down expert stride")?),
            ],
            &indirect_arguments,
            15 * std::mem::size_of::<u32>(),
            MATMUL_THREADS,
            DOWN_SHARED_BYTES,
        )?;

        encode_1d_threadgroups_args(
            command_buffer,
            &pipelines.combine,
            &[
                KernelArg::Buffer(&assignment_output),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(token_count, "token count")?),
                KernelArg::U32(as_u32(out_features, "output width")?),
            ],
            output_len.div_ceil(COMBINE_THREADS),
            COMBINE_THREADS,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn validate(
        &self,
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
    ) -> Result<()> {
        if !Self::supports(routing) {
            return Err(Error::backend(format!(
                "Laguna XS expert prefill requires at least {MIN_TOKENS} tokens, top-{TOP_K}, and {EXPERT_COUNT} experts; got tokens={}, top_k={}, experts={}",
                routing.token_count, routing.top_k, routing.expert_count
            )));
        }
        if !in_features.is_multiple_of(BLOCK_VALUES)
            || !intermediate_features.is_multiple_of(BLOCK_VALUES)
            || !intermediate_features.is_multiple_of(OUTPUT_TILE)
            || !out_features.is_multiple_of(OUTPUT_TILE)
        {
            return Err(Error::backend(format!(
                "Laguna XS expert prefill dimensions are unsupported: in={in_features}, intermediate={intermediate_features}, out={out_features}"
            )));
        }
        let expected_input_len = routing
            .token_count
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("Laguna XS MoE input length overflow"))?;
        if input_len != expected_input_len {
            return Err(Error::backend(format!(
                "Laguna XS expert prefill expected {expected_input_len} input values, got {input_len}"
            )));
        }
        require_f32_capacity(input, input_len, "Laguna XS expert prefill input")?;

        let gate_row_bytes = quantized_row_bytes(in_features, Q4_K_BLOCK_BYTES)?;
        let gate_expert_stride = intermediate_features
            .checked_mul(gate_row_bytes)
            .ok_or_else(|| Error::backend("Laguna XS gate stride overflow"))?;
        validate_weight_len("gate", gate_weights, gate_expert_stride)?;
        validate_weight_len("up", up_weights, gate_expert_stride)?;

        let down_block_bytes = match down_quant {
            GgufKQuant::Q4K => Q4_K_BLOCK_BYTES,
            GgufKQuant::Q6K => Q6_K_BLOCK_BYTES,
        };
        let down_row_bytes = quantized_row_bytes(intermediate_features, down_block_bytes)?;
        let down_expert_stride = out_features
            .checked_mul(down_row_bytes)
            .ok_or_else(|| Error::backend("Laguna XS down stride overflow"))?;
        validate_weight_len("down", down_weights, down_expert_stride)
    }

    fn pipelines(&self, device: &Device) -> Result<MutexGuard<'_, Option<Pipelines>>> {
        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| Error::backend("Laguna XS expert-prefill pipeline lock poisoned"))?;
        if pipelines.is_none() {
            let library = MetalLibrary::compile_source(device, KERNEL_SOURCE)?;
            *pipelines = Some(Pipelines {
                build_map: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_prefill_build_expert_map_kernel",
                )?,
                q4_gate_up_small: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_prefill_q4_gate_up_small_mma_kernel",
                )?,
                q4_gate_up_medium: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_prefill_q4_gate_up_medium_mma_kernel",
                )?,
                q4_gate_up_large: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_prefill_q4_gate_up_large_mma_kernel",
                )?,
                q4_down_small: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_prefill_q4_down_small_mma_kernel",
                )?,
                q4_down_medium: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_prefill_q4_down_medium_mma_kernel",
                )?,
                q4_down_large: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_prefill_q4_down_large_mma_kernel",
                )?,
                q6_down_small: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_prefill_q6_down_small_mma_kernel",
                )?,
                q6_down_medium: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_prefill_q6_down_medium_mma_kernel",
                )?,
                q6_down_large: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_prefill_q6_down_large_mma_kernel",
                )?,
                combine: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_prefill_combine_f32_kernel",
                )?,
            });
        }
        Ok(pipelines)
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
            .map_err(|_| Error::backend("Laguna XS prefill weight cache lock poisoned"))?;
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

fn quantized_row_bytes(features: usize, block_bytes: usize) -> Result<usize> {
    if !features.is_multiple_of(BLOCK_VALUES) {
        return Err(Error::backend(format!(
            "Laguna XS feature count {features} is not divisible by {BLOCK_VALUES}"
        )));
    }
    (features / BLOCK_VALUES)
        .checked_mul(block_bytes)
        .ok_or_else(|| Error::backend("Laguna XS quantized row length overflow"))
}

fn validate_weight_len(label: &str, weights: &[u8], expert_stride: usize) -> Result<()> {
    let expected = EXPERT_COUNT
        .checked_mul(expert_stride)
        .ok_or_else(|| Error::backend("Laguna XS expert payload overflow"))?;
    if weights.len() != expected {
        return Err(Error::backend(format!(
            "Laguna XS {label} experts must contain {expected} bytes, got {}",
            weights.len()
        )));
    }
    Ok(())
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::backend(format!("Laguna XS {label} exceeds Metal u32")))
}
