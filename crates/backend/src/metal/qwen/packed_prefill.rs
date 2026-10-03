use std::sync::Mutex;

use ::metal::{Buffer, CommandBufferRef, ComputePipelineState, Device};
use common::{Error, Result};

use crate::DeviceQwenMlxW4Matrix;

use super::super::{
    arena::MetalArena,
    buffers::{empty_f16_buffer, empty_f32_buffer, empty_u32_buffer},
    command::{encode_1d_threadgroups_args, KernelArg},
    library::MetalLibrary,
    pipeline::compute_pipeline,
};

const SOURCE: &str = include_str!("kernels/packed_prefill.metal");
const RESIDENT_SOURCE: &str = include_str!("kernels/resident_prefill.metal");
const REPACK: &str = "qwen_packed_prefill_repack_kernel";
const LINEAR: &str = "qwen_splitk_prefill_linear_bf16_kernel";
const GATE_UP: &str = "qwen_splitk_prefill_gate_up_bf16_kernel";
const REDUCE: &str = "qwen_splitk_prefill_reduce_bf16_kernel";
const REDUCE_ADD: &str = "qwen_splitk_prefill_reduce_add_bf16_kernel";
const REDUCE_SWIGLU: &str = "qwen_splitk_prefill_reduce_swiglu_bf16_kernel";
const MIN_ROWS: usize = 64;
const MIN_RESIDENT_ROWS: usize = 256;
const GROUP_SIZE: usize = 64;
const VALUES_PER_WORD: usize = 8;
const K_PARTS: usize = 8;
const LINEAR_TOKEN_ROWS: usize = 32;
const LINEAR_OUTPUTS: usize = 32;
const LINEAR_THREADS: usize = 128;
const REDUCE_THREADS: usize = 256;
const REPACK_THREADS: usize = 256;
const REPACK_WORD_TILE: usize = 32;

#[cfg(test)]
#[path = "packed_prefill_tests.rs"]
mod tests;

#[derive(Default)]
struct ScratchSlot {
    packed: Option<Buffer>,
    scales: Option<Buffer>,
    biases: Option<Buffer>,
    partials: Option<Buffer>,
}

#[derive(Clone)]
struct PackedBuffers {
    packed: Buffer,
    scales: Buffer,
    biases: Buffer,
}

pub(super) struct MetalQwenPackedPrefill {
    repack: ComputePipelineState,
    linear: ComputePipelineState,
    gate_up: ComputePipelineState,
    resident_gate_up: ComputePipelineState,
    reduce: ComputePipelineState,
    reduce_add: ComputePipelineState,
    reduce_swiglu: ComputePipelineState,
    scratch: Mutex<[ScratchSlot; 2]>,
    device: Device,
    arena: MetalArena,
}

impl MetalQwenPackedPrefill {
    pub(super) fn new(device: &Device, arena: MetalArena) -> Result<Self> {
        let library =
            MetalLibrary::compile_source(device, &format!("{SOURCE}\n{RESIDENT_SOURCE}"))?;
        Ok(Self {
            repack: compute_pipeline(device, &library, REPACK)?,
            linear: compute_pipeline(device, &library, LINEAR)?,
            gate_up: compute_pipeline(device, &library, GATE_UP)?,
            resident_gate_up: compute_pipeline(
                device,
                &library,
                "qwen_resident_prefill_gate_up_kernel",
            )?,
            reduce: compute_pipeline(device, &library, REDUCE)?,
            reduce_add: compute_pipeline(device, &library, REDUCE_ADD)?,
            reduce_swiglu: compute_pipeline(device, &library, REDUCE_SWIGLU)?,
            scratch: Mutex::new([ScratchSlot::default(), ScratchSlot::default()]),
            device: device.to_owned(),
            arena,
        })
    }

    pub(super) fn supports_linear(matrix: &DeviceQwenMlxW4Matrix, row_count: usize) -> bool {
        row_count >= MIN_ROWS
            && matrix.rows.is_multiple_of(LINEAR_OUTPUTS)
            && matrix.columns.is_multiple_of(GROUP_SIZE)
            && matrix.columns.is_multiple_of(256)
    }

    pub(super) fn supports_gate_up(matrix: &DeviceQwenMlxW4Matrix, row_count: usize) -> bool {
        row_count >= MIN_ROWS
            && matrix.rows.is_multiple_of(LINEAR_OUTPUTS)
            && matrix.columns.is_multiple_of(GROUP_SIZE)
            && matrix.columns.is_multiple_of(256)
    }

    pub(super) fn encode_linear_into(
        &self,
        command_buffer: &CommandBufferRef,
        matrix: &DeviceQwenMlxW4Matrix,
        input: &Buffer,
        output: &Buffer,
        row_count: usize,
    ) -> Result<()> {
        let packed = self.repack_matrix(command_buffer, matrix, 0, LINEAR_OUTPUTS)?;
        self.encode_packed_linear(
            command_buffer,
            &packed,
            input,
            output,
            row_count,
            matrix.columns,
            matrix.rows,
        )
    }

    pub(super) fn encode_linear_add_into(
        &self,
        command_buffer: &CommandBufferRef,
        matrix: &DeviceQwenMlxW4Matrix,
        input: &Buffer,
        residual: &Buffer,
        output: &Buffer,
        row_count: usize,
    ) -> Result<()> {
        let packed = self.repack_matrix(command_buffer, matrix, 0, LINEAR_OUTPUTS)?;
        let (partials, output_len) = self.encode_packed_linear_partials(
            command_buffer,
            &packed,
            input,
            row_count,
            matrix.columns,
            matrix.rows,
        )?;
        self.encode_reduce_add(command_buffer, &partials, residual, output, output_len)
    }

    pub(super) fn encode_gate_up(
        &self,
        command_buffer: &CommandBufferRef,
        gate: &DeviceQwenMlxW4Matrix,
        up: &DeviceQwenMlxW4Matrix,
        input: &Buffer,
        row_count: usize,
    ) -> Result<Buffer> {
        if gate.rows != up.rows || gate.columns != up.columns {
            return Err(Error::backend("Qwen packed prefill gate/up shapes differ"));
        }
        let gate_packed = self.repack_matrix(command_buffer, gate, 0, LINEAR_OUTPUTS)?;
        let up_packed = self.repack_matrix(command_buffer, up, 1, LINEAR_OUTPUTS)?;
        let output_len = row_count
            .checked_mul(gate.rows)
            .ok_or_else(|| Error::backend("Qwen prefill gate/up output size overflow"))?;
        let output = self.arena.empty_f16(output_len)?;
        self.encode_packed_gate_up(
            command_buffer,
            &gate_packed,
            &up_packed,
            input,
            &output,
            row_count,
            gate.columns,
            gate.rows,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_packed_gate_up(
        &self,
        command_buffer: &CommandBufferRef,
        gate_packed: &PackedBuffers,
        up_packed: &PackedBuffers,
        input: &Buffer,
        output: &Buffer,
        row_count: usize,
        input_width: usize,
        output_width: usize,
    ) -> Result<()> {
        // Large prompts supply enough tiles without splitting K across threadgroups.
        if row_count >= MIN_RESIDENT_ROWS {
            let threadgroups = row_count
                .div_ceil(LINEAR_TOKEN_ROWS)
                .checked_mul(output_width / LINEAR_OUTPUTS)
                .ok_or_else(|| {
                    Error::backend("Qwen resident prefill threadgroup count overflow")
                })?;
            return encode_1d_threadgroups_args(
                command_buffer,
                &self.resident_gate_up,
                &[
                    KernelArg::Buffer(&gate_packed.packed),
                    KernelArg::Buffer(&gate_packed.scales),
                    KernelArg::Buffer(&gate_packed.biases),
                    KernelArg::Buffer(&up_packed.packed),
                    KernelArg::Buffer(&up_packed.scales),
                    KernelArg::Buffer(&up_packed.biases),
                    KernelArg::Buffer(input),
                    KernelArg::Buffer(output),
                    KernelArg::U32(as_u32(row_count, "row count")?),
                    KernelArg::U32(as_u32(input_width, "input width")?),
                    KernelArg::U32(as_u32(output_width, "output width")?),
                ],
                threadgroups,
                LINEAR_THREADS,
            );
        }
        let output_len = row_count
            .checked_mul(output_width)
            .ok_or_else(|| Error::backend("Qwen prefill gate/up output size overflow"))?;
        let partial_len = output_len
            .checked_mul(K_PARTS)
            .ok_or_else(|| Error::backend("Qwen prefill gate/up partial size overflow"))?;
        let gate_partials = self.partial_scratch(0, partial_len)?;
        let up_partials = self.partial_scratch(1, partial_len)?;
        let threadgroups = row_count
            .div_ceil(LINEAR_TOKEN_ROWS)
            .checked_mul(output_width / LINEAR_OUTPUTS)
            .and_then(|count| count.checked_mul(K_PARTS))
            .ok_or_else(|| Error::backend("Qwen prefill gate/up threadgroup count overflow"))?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.gate_up,
            &[
                KernelArg::Buffer(&gate_packed.packed),
                KernelArg::Buffer(&gate_packed.scales),
                KernelArg::Buffer(&gate_packed.biases),
                KernelArg::Buffer(&up_packed.packed),
                KernelArg::Buffer(&up_packed.scales),
                KernelArg::Buffer(&up_packed.biases),
                KernelArg::Buffer(input),
                KernelArg::Buffer(&gate_partials),
                KernelArg::Buffer(&up_partials),
                KernelArg::U32(as_u32(row_count, "row count")?),
                KernelArg::U32(as_u32(input_width, "input width")?),
                KernelArg::U32(as_u32(output_width, "output width")?),
            ],
            threadgroups,
            LINEAR_THREADS,
        )?;
        self.encode_reduce_swiglu(
            command_buffer,
            &gate_partials,
            &up_partials,
            output,
            output_len,
        )
    }

    fn repack_matrix(
        &self,
        command_buffer: &CommandBufferRef,
        matrix: &DeviceQwenMlxW4Matrix,
        slot_index: usize,
        output_tile: usize,
    ) -> Result<PackedBuffers> {
        self.repack_raw(
            command_buffer,
            &matrix.weight,
            matrix.weight_byte_offset,
            &matrix.scales,
            matrix.scale_byte_offset,
            &matrix.biases,
            matrix.bias_byte_offset,
            matrix.columns,
            matrix.rows,
            slot_index,
            output_tile,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn repack_raw(
        &self,
        command_buffer: &CommandBufferRef,
        source_packed: &Buffer,
        packed_byte_offset: usize,
        source_scales: &Buffer,
        scale_byte_offset: usize,
        source_biases: &Buffer,
        bias_byte_offset: usize,
        input_width: usize,
        output_width: usize,
        slot_index: usize,
        output_tile: usize,
    ) -> Result<PackedBuffers> {
        if output_tile != LINEAR_OUTPUTS
            || !output_width.is_multiple_of(LINEAR_OUTPUTS)
            || !input_width.is_multiple_of(REPACK_WORD_TILE * VALUES_PER_WORD)
        {
            return Err(Error::backend(
                "Qwen prefill repack requires complete 32-by-32 word tiles",
            ));
        }
        let words_per_row = input_width / VALUES_PER_WORD;
        let groups_per_row = input_width / GROUP_SIZE;
        let packed_count = output_width
            .checked_mul(words_per_row)
            .ok_or_else(|| Error::backend("Qwen packed prefill weight size overflow"))?;
        let parameter_count = output_width
            .checked_mul(groups_per_row)
            .ok_or_else(|| Error::backend("Qwen packed prefill parameter size overflow"))?;
        let buffers = self.scratch(slot_index, packed_count, parameter_count)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.repack,
            &[
                KernelArg::BufferOffset(source_packed, packed_byte_offset),
                KernelArg::BufferOffset(source_scales, scale_byte_offset),
                KernelArg::BufferOffset(source_biases, bias_byte_offset),
                KernelArg::Buffer(&buffers.packed),
                KernelArg::Buffer(&buffers.scales),
                KernelArg::Buffer(&buffers.biases),
                KernelArg::U32(as_u32(input_width, "input width")?),
                KernelArg::U32(as_u32(output_width, "output width")?),
                KernelArg::U32(as_u32(output_tile, "output tile")?),
            ],
            packed_count.div_ceil(LINEAR_OUTPUTS * REPACK_WORD_TILE),
            REPACK_THREADS,
        )?;
        Ok(buffers)
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_linear_raw(
        &self,
        command_buffer: &CommandBufferRef,
        source_packed: &Buffer,
        source_scales: &Buffer,
        source_biases: &Buffer,
        input: &Buffer,
        output: &Buffer,
        row_count: usize,
        input_width: usize,
        output_width: usize,
    ) -> Result<()> {
        let packed = self.repack_raw(
            command_buffer,
            source_packed,
            0,
            source_scales,
            0,
            source_biases,
            0,
            input_width,
            output_width,
            0,
            LINEAR_OUTPUTS,
        )?;
        self.encode_packed_linear(
            command_buffer,
            &packed,
            input,
            output,
            row_count,
            input_width,
            output_width,
        )
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_linear_add_raw(
        &self,
        command_buffer: &CommandBufferRef,
        source_packed: &Buffer,
        source_scales: &Buffer,
        source_biases: &Buffer,
        input: &Buffer,
        residual: &Buffer,
        output: &Buffer,
        row_count: usize,
        input_width: usize,
        output_width: usize,
    ) -> Result<()> {
        let packed = self.repack_raw(
            command_buffer,
            source_packed,
            0,
            source_scales,
            0,
            source_biases,
            0,
            input_width,
            output_width,
            0,
            LINEAR_OUTPUTS,
        )?;
        let (partials, output_len) = self.encode_packed_linear_partials(
            command_buffer,
            &packed,
            input,
            row_count,
            input_width,
            output_width,
        )?;
        self.encode_reduce_add(command_buffer, &partials, residual, output, output_len)
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_gate_up_raw(
        &self,
        command_buffer: &CommandBufferRef,
        gate_packed: &Buffer,
        gate_scales: &Buffer,
        gate_biases: &Buffer,
        up_packed: &Buffer,
        up_scales: &Buffer,
        up_biases: &Buffer,
        input: &Buffer,
        output: &Buffer,
        row_count: usize,
        input_width: usize,
        output_width: usize,
    ) -> Result<()> {
        let gate = self.repack_raw(
            command_buffer,
            gate_packed,
            0,
            gate_scales,
            0,
            gate_biases,
            0,
            input_width,
            output_width,
            0,
            LINEAR_OUTPUTS,
        )?;
        let up = self.repack_raw(
            command_buffer,
            up_packed,
            0,
            up_scales,
            0,
            up_biases,
            0,
            input_width,
            output_width,
            1,
            LINEAR_OUTPUTS,
        )?;
        self.encode_packed_gate_up(
            command_buffer,
            &gate,
            &up,
            input,
            output,
            row_count,
            input_width,
            output_width,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_packed_linear(
        &self,
        command_buffer: &CommandBufferRef,
        packed: &PackedBuffers,
        input: &Buffer,
        output: &Buffer,
        row_count: usize,
        input_width: usize,
        output_width: usize,
    ) -> Result<()> {
        let output_len = row_count
            .checked_mul(output_width)
            .ok_or_else(|| Error::backend("Qwen tiled prefill output size overflow"))?;
        let (partials, partial_output_len) = self.encode_packed_linear_partials(
            command_buffer,
            packed,
            input,
            row_count,
            input_width,
            output_width,
        )?;
        if partial_output_len != output_len {
            return Err(Error::backend(
                "Qwen packed prefill linear output lengths differ",
            ));
        }
        self.encode_reduce(command_buffer, &partials, output, output_len)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_packed_linear_partials(
        &self,
        command_buffer: &CommandBufferRef,
        packed: &PackedBuffers,
        input: &Buffer,
        row_count: usize,
        input_width: usize,
        output_width: usize,
    ) -> Result<(Buffer, usize)> {
        let output_len = row_count
            .checked_mul(output_width)
            .ok_or_else(|| Error::backend("Qwen packed prefill output size overflow"))?;
        let partial_len = output_len
            .checked_mul(K_PARTS)
            .ok_or_else(|| Error::backend("Qwen packed prefill partial size overflow"))?;
        let partials = self.partial_scratch(0, partial_len)?;
        let threadgroups = row_count
            .div_ceil(LINEAR_TOKEN_ROWS)
            .checked_mul(output_width / LINEAR_OUTPUTS)
            .and_then(|count| count.checked_mul(K_PARTS))
            .ok_or_else(|| Error::backend("Qwen packed prefill threadgroup count overflow"))?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.linear,
            &[
                KernelArg::Buffer(&packed.packed),
                KernelArg::Buffer(&packed.scales),
                KernelArg::Buffer(&packed.biases),
                KernelArg::Buffer(input),
                KernelArg::Buffer(&partials),
                KernelArg::U32(as_u32(row_count, "row count")?),
                KernelArg::U32(as_u32(input_width, "input width")?),
                KernelArg::U32(as_u32(output_width, "output width")?),
            ],
            threadgroups,
            LINEAR_THREADS,
        )?;
        Ok((partials, output_len))
    }

    fn encode_reduce(
        &self,
        command_buffer: &CommandBufferRef,
        partials: &Buffer,
        output: &Buffer,
        output_len: usize,
    ) -> Result<()> {
        encode_1d_threadgroups_args(
            command_buffer,
            &self.reduce,
            &[
                KernelArg::Buffer(partials),
                KernelArg::Buffer(output),
                KernelArg::U32(as_u32(output_len, "output length")?),
            ],
            output_len.div_ceil(REDUCE_THREADS),
            REDUCE_THREADS,
        )
    }

    fn encode_reduce_add(
        &self,
        command_buffer: &CommandBufferRef,
        partials: &Buffer,
        residual: &Buffer,
        output: &Buffer,
        output_len: usize,
    ) -> Result<()> {
        encode_1d_threadgroups_args(
            command_buffer,
            &self.reduce_add,
            &[
                KernelArg::Buffer(partials),
                KernelArg::Buffer(residual),
                KernelArg::Buffer(output),
                KernelArg::U32(as_u32(output_len, "residual output length")?),
            ],
            output_len.div_ceil(REDUCE_THREADS),
            REDUCE_THREADS,
        )
    }

    fn encode_reduce_swiglu(
        &self,
        command_buffer: &CommandBufferRef,
        gate_partials: &Buffer,
        up_partials: &Buffer,
        output: &Buffer,
        output_len: usize,
    ) -> Result<()> {
        if !output_len.is_multiple_of(4) {
            return Err(Error::backend(format!(
                "Qwen fused SwiGLU output length must be divisible by 4, got {output_len}"
            )));
        }
        let output_vector_count = output_len / 4;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.reduce_swiglu,
            &[
                KernelArg::Buffer(gate_partials),
                KernelArg::Buffer(up_partials),
                KernelArg::Buffer(output),
                KernelArg::U32(as_u32(output_vector_count, "SwiGLU vector count")?),
            ],
            output_vector_count.div_ceil(REDUCE_THREADS),
            REDUCE_THREADS,
        )
    }

    fn partial_scratch(&self, slot_index: usize, len: usize) -> Result<Buffer> {
        let bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| Error::backend("Qwen prefill partial scratch size overflow"))?;
        let mut slots = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Qwen prefill scratch lock poisoned"))?;
        let slot = slots
            .get_mut(slot_index)
            .ok_or_else(|| Error::backend("Qwen prefill partial scratch slot is invalid"))?;
        if slot
            .partials
            .as_ref()
            .is_none_or(|buffer| buffer.length() < bytes as u64)
        {
            slot.partials = Some(empty_f32_buffer(&self.device, len)?);
        }
        // GPU-only scratch: each tracked reduction reads it before the next
        // projection overwrites it on the same ordered command queue.
        slot.partials
            .clone()
            .ok_or_else(|| Error::backend("Qwen prefill partial scratch is unavailable"))
    }

    fn scratch(
        &self,
        slot_index: usize,
        packed_count: usize,
        parameter_count: usize,
    ) -> Result<PackedBuffers> {
        let mut slots = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Qwen packed prefill scratch lock poisoned"))?;
        let slot = slots
            .get_mut(slot_index)
            .ok_or_else(|| Error::backend("Qwen packed prefill scratch slot is invalid"))?;
        ensure_u32_capacity(&self.device, &mut slot.packed, packed_count)?;
        ensure_f16_capacity(&self.device, &mut slot.scales, parameter_count)?;
        ensure_f16_capacity(&self.device, &mut slot.biases, parameter_count)?;
        Ok(PackedBuffers {
            packed: slot
                .packed
                .as_ref()
                .cloned()
                .ok_or_else(|| Error::backend("Qwen packed weight scratch is unavailable"))?,
            scales: slot
                .scales
                .as_ref()
                .cloned()
                .ok_or_else(|| Error::backend("Qwen packed scale scratch is unavailable"))?,
            biases: slot
                .biases
                .as_ref()
                .cloned()
                .ok_or_else(|| Error::backend("Qwen packed bias scratch is unavailable"))?,
        })
    }
}

fn ensure_u32_capacity(device: &Device, buffer: &mut Option<Buffer>, len: usize) -> Result<()> {
    let required_bytes = len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| Error::backend("Qwen packed weight byte size overflow"))?;
    if buffer
        .as_ref()
        .is_none_or(|buffer| buffer.length() < required_bytes as u64)
    {
        *buffer = Some(empty_u32_buffer(device, len)?);
    }
    Ok(())
}

fn ensure_f16_capacity(device: &Device, buffer: &mut Option<Buffer>, len: usize) -> Result<()> {
    let required_bytes = len
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| Error::backend("Qwen packed parameter byte size overflow"))?;
    if buffer
        .as_ref()
        .is_none_or(|buffer| buffer.length() < required_bytes as u64)
    {
        *buffer = Some(empty_f16_buffer(device, len)?);
    }
    Ok(())
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::backend(format!("Qwen packed prefill {label} exceeds Metal u32")))
}
