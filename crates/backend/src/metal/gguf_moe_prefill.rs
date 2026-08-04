use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use ::metal::{Buffer, CommandBufferRef, ComputePipelineState, Device};
use common::{Error, Result};

use crate::{DeviceRouterTopK, GgufExpertQuant};

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

const Q2_BLOCK_BYTES: usize = 84;
const Q3_BLOCK_BYTES: usize = 110;
const BLOCK_VALUES: usize = 256;
const TOP_K: usize = 10;
const OUTPUT_TILE: usize = 64;
const ASSIGNMENT_TILE: usize = 32;
const MATMUL_THREADS: usize = 128;
// Ping-pong staging is larger than the final result tile for this geometry.
const GATE_UP_SHARED_BYTES: usize = 20 * 1024;
const DOWN_SHARED_BYTES: usize = 12 * 1024;
const CAST_THREADS: usize = 256;
const CAST_VECTOR_WIDTH: usize = 4;
const COMBINE_THREADS: usize = 256;
const INDIRECT_ARGUMENT_U32S: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct WeightKey {
    address: usize,
    byte_len: usize,
}

struct WeightBuffer {
    storage: Buffer,
    byte_offset: usize,
}

/// Expert-major quantized GEMM used only for multi-token Laguna GGUF prefill.
///
/// The established `MetalGgufMoe` implementation remains the single-token
/// decode path. This dispatcher groups Laguna's top-10 assignments by expert,
/// then
/// evaluates 64 output rows against up to 32 routed prompt rows per Metal
/// threadgroup.
pub(crate) struct MetalGgufMoePrefill {
    cast_input: ComputePipelineState,
    build_map: ComputePipelineState,
    q2_gate_up: ComputePipelineState,
    q2_down: ComputePipelineState,
    q3_gate_up: ComputePipelineState,
    q3_down: ComputePipelineState,
    combine: ComputePipelineState,
    arena: MetalArena,
    weights: Mutex<HashMap<WeightKey, Buffer>>,
    laguna_views: Arc<MetalLagunaViews>,
}

impl MetalGgufMoePrefill {
    pub(crate) fn new(
        device: &Device,
        library: &MetalLibrary,
        arena: MetalArena,
        laguna_views: Arc<MetalLagunaViews>,
    ) -> Result<Self> {
        Ok(Self {
            cast_input: compute_pipeline(device, library, "laguna_prefill_cast_f32_f16_kernel")?,
            build_map: compute_pipeline(device, library, "laguna_prefill_build_expert_map_kernel")?,
            q2_gate_up: compute_pipeline(device, library, "laguna_prefill_q2_gate_up_mma_kernel")?,
            q2_down: compute_pipeline(device, library, "laguna_prefill_q2_down_mma_kernel")?,
            q3_gate_up: compute_pipeline(device, library, "laguna_prefill_q3_gate_up_mma_kernel")?,
            q3_down: compute_pipeline(device, library, "laguna_prefill_q3_down_mma_kernel")?,
            combine: compute_pipeline(device, library, "laguna_prefill_combine_f32_kernel")?,
            arena,
            weights: Mutex::new(HashMap::new()),
            laguna_views,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        quant: GgufExpertQuant,
        input: &Buffer,
        input_len: usize,
        routing: &DeviceRouterTopK,
        in_features: usize,
        intermediate_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.validate(
            input,
            input_len,
            routing,
            in_features,
            intermediate_features,
            out_features,
        )?;
        let token_count = routing.token_count;
        let expert_count = routing.expert_count;
        let assignment_count = routing.assignment_count()?;
        let input_f16 = self.arena.empty_f16(input_len)?;
        let block_bytes = match quant {
            GgufExpertQuant::Q2K => Q2_BLOCK_BYTES,
            GgufExpertQuant::Q3K => Q3_BLOCK_BYTES,
        };
        let gate_row_bytes = quantized_row_bytes(in_features, block_bytes)?;
        let down_row_bytes = quantized_row_bytes(intermediate_features, block_bytes)?;
        let gate_expert_stride = intermediate_features
            .checked_mul(gate_row_bytes)
            .ok_or_else(|| Error::backend("Laguna prefill gate stride overflow"))?;
        let down_expert_stride = out_features
            .checked_mul(down_row_bytes)
            .ok_or_else(|| Error::backend("Laguna prefill down stride overflow"))?;
        validate_weight_len("gate", gate_weights, expert_count, gate_expert_stride)?;
        validate_weight_len("up", up_weights, expert_count, gate_expert_stride)?;
        validate_weight_len("down", down_weights, expert_count, down_expert_stride)?;

        let expert_counts = self.arena.empty_u32(expert_count)?;
        let assignment_map_len = expert_count
            .checked_mul(token_count)
            .ok_or_else(|| Error::backend("Laguna prefill assignment map overflow"))?;
        let assignment_map = self.arena.empty_u32(assignment_map_len)?;
        let maximum_work_tiles = expert_count
            .checked_mul(token_count.div_ceil(ASSIGNMENT_TILE))
            .ok_or_else(|| Error::backend("Laguna prefill work tile count overflow"))?;
        let work_tiles = self.arena.empty_u32(
            maximum_work_tiles
                .checked_mul(4)
                .ok_or_else(|| Error::backend("Laguna prefill work tile buffer overflow"))?,
        )?;
        let indirect_arguments = self.arena.empty_u32(INDIRECT_ARGUMENT_U32S)?;
        let intermediate_len = assignment_count
            .checked_mul(intermediate_features)
            .ok_or_else(|| Error::backend("Laguna prefill intermediate overflow"))?;
        let intermediate = self.arena.empty_f16(intermediate_len)?;
        let assignment_output_len = assignment_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Laguna prefill assignment output overflow"))?;
        let assignment_output = self.arena.empty_f32(assignment_output_len)?;
        let output_len = token_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Laguna prefill output overflow"))?;
        let output = self.arena.empty_f32(output_len)?;

        encode_1d_threadgroups_args(
            command_buffer,
            &self.cast_input,
            &[
                KernelArg::Buffer(input),
                KernelArg::Buffer(&input_f16),
                KernelArg::U32(as_u32(input_len, "input values")?),
            ],
            (input_len / CAST_VECTOR_WIDTH).div_ceil(CAST_THREADS),
            CAST_THREADS,
        )?;

        let map_args = [
            KernelArg::Buffer(&routing.expert_ids),
            KernelArg::Buffer(&expert_counts),
            KernelArg::Buffer(&assignment_map),
            KernelArg::Buffer(&work_tiles),
            KernelArg::Buffer(&indirect_arguments),
            KernelArg::U32(as_u32(token_count, "tokens")?),
            KernelArg::U32(as_u32(expert_count, "experts")?),
            KernelArg::U32(as_u32(
                intermediate_features.div_ceil(OUTPUT_TILE),
                "gate output tiles",
            )?),
            KernelArg::U32(as_u32(
                out_features.div_ceil(OUTPUT_TILE),
                "down output tiles",
            )?),
        ];
        encode_3d_threadgroups_args(
            command_buffer,
            &self.build_map,
            &map_args,
            [1, 1, 1],
            expert_count,
            0,
        )?;

        let gate = self.weight_buffer(device, gate_weights)?;
        let up = self.weight_buffer(device, up_weights)?;
        let down = self.weight_buffer(device, down_weights)?;
        let gate_args = [
            KernelArg::BufferOffset(&gate.storage, gate.byte_offset),
            KernelArg::BufferOffset(&up.storage, up.byte_offset),
            KernelArg::Buffer(&input_f16),
            KernelArg::Buffer(&assignment_map),
            KernelArg::Buffer(&work_tiles),
            KernelArg::Buffer(&routing.expert_weights),
            KernelArg::Buffer(&intermediate),
            KernelArg::U32(as_u32(in_features, "input width")?),
            KernelArg::U32(as_u32(intermediate_features, "intermediate width")?),
            KernelArg::U32(as_u32(token_count, "tokens")?),
            KernelArg::U32(as_u32(gate_row_bytes, "gate row bytes")?),
            KernelArg::U32(as_u32(gate_expert_stride, "gate expert stride")?),
        ];
        let (gate_pipeline, down_pipeline) = match quant {
            GgufExpertQuant::Q2K => (&self.q2_gate_up, &self.q2_down),
            GgufExpertQuant::Q3K => (&self.q3_gate_up, &self.q3_down),
        };
        encode_threadgroups_indirect_args(
            command_buffer,
            gate_pipeline,
            &gate_args,
            &indirect_arguments,
            0,
            MATMUL_THREADS,
            GATE_UP_SHARED_BYTES,
        )?;

        let down_args = [
            KernelArg::BufferOffset(&down.storage, down.byte_offset),
            KernelArg::Buffer(&intermediate),
            KernelArg::Buffer(&assignment_map),
            KernelArg::Buffer(&work_tiles),
            KernelArg::Buffer(&assignment_output),
            KernelArg::U32(as_u32(intermediate_features, "intermediate width")?),
            KernelArg::U32(as_u32(out_features, "output width")?),
            KernelArg::U32(as_u32(token_count, "tokens")?),
            KernelArg::U32(as_u32(down_row_bytes, "down row bytes")?),
            KernelArg::U32(as_u32(down_expert_stride, "down expert stride")?),
        ];
        encode_threadgroups_indirect_args(
            command_buffer,
            down_pipeline,
            &down_args,
            &indirect_arguments,
            3 * std::mem::size_of::<u32>(),
            MATMUL_THREADS,
            DOWN_SHARED_BYTES,
        )?;

        let combine_args = [
            KernelArg::Buffer(&assignment_output),
            KernelArg::Buffer(&output),
            KernelArg::U32(as_u32(token_count, "tokens")?),
            KernelArg::U32(as_u32(out_features, "output width")?),
        ];
        encode_1d_threadgroups_args(
            command_buffer,
            &self.combine,
            &combine_args,
            output_len.div_ceil(COMBINE_THREADS),
            COMBINE_THREADS,
        )?;
        Ok(output)
    }

    fn validate(
        &self,
        input: &Buffer,
        input_len: usize,
        routing: &DeviceRouterTopK,
        in_features: usize,
        intermediate_features: usize,
        out_features: usize,
    ) -> Result<()> {
        if routing.token_count < 2 || routing.top_k != TOP_K || routing.expert_count != 256 {
            return Err(Error::backend(format!(
                "Laguna prefill GEMM requires multi-token top-10 routing over 256 experts, got tokens={}, top_k={}, experts={}",
                routing.token_count, routing.top_k, routing.expert_count
            )));
        }
        let expected_input_len = routing
            .token_count
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("Laguna prefill input overflow"))?;
        if input_len != expected_input_len {
            return Err(Error::backend(format!(
                "Laguna prefill expected {expected_input_len} input values, got {input_len}"
            )));
        }
        require_f32_capacity(input, input_len, "Laguna prefill input")?;
        if !in_features.is_multiple_of(BLOCK_VALUES)
            || !intermediate_features.is_multiple_of(BLOCK_VALUES)
            || out_features == 0
        {
            return Err(Error::backend(format!(
                "Laguna prefill dimensions must use 256-wide quant blocks, got in={in_features}, intermediate={intermediate_features}, out={out_features}"
            )));
        }
        Ok(())
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
            .map_err(|_| Error::backend("Laguna prefill weight cache lock poisoned"))?;
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
            "Laguna prefill feature count {features} is not divisible by {BLOCK_VALUES}"
        )));
    }
    (features / BLOCK_VALUES)
        .checked_mul(block_bytes)
        .ok_or_else(|| Error::backend("Laguna prefill row byte length overflow"))
}

fn validate_weight_len(
    label: &str,
    bytes: &[u8],
    expert_count: usize,
    expert_stride: usize,
) -> Result<()> {
    let expected = expert_count
        .checked_mul(expert_stride)
        .ok_or_else(|| Error::backend("Laguna prefill expert payload overflow"))?;
    if bytes.len() != expected {
        return Err(Error::backend(format!(
            "Laguna prefill {label} payload must contain {expected} bytes, got {}",
            bytes.len()
        )));
    }
    Ok(())
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::backend(format!("Laguna prefill {label} exceeds Metal u32")))
}
