use ::metal::{Buffer, CommandBufferRef, ComputePipelineState, Device};
use common::{Error, Result};

use super::{
    arena::MetalArena,
    buffers::{
        read_f16_buffer_as_f32, read_f32_buffer, require_byte_capacity, require_f32_capacity,
    },
    command::encode_1d,
    library::MetalLibrary,
    pipeline::compute_pipeline,
};

const F32_TO_F16_KERNEL: &str = "f32_to_f16_kernel";
const Q8_ROWS_TO_F32_KERNEL: &str = "q8_rows_to_f32_kernel";

pub(crate) struct MetalCast {
    f32_to_f16: ComputePipelineState,
    q8_rows_to_f32: ComputePipelineState,
    arena: MetalArena,
}

impl MetalCast {
    pub(crate) fn new(device: &Device, library: &MetalLibrary, arena: MetalArena) -> Result<Self> {
        Ok(Self {
            f32_to_f16: compute_pipeline(device, library, F32_TO_F16_KERNEL)?,
            q8_rows_to_f32: compute_pipeline(device, library, Q8_ROWS_TO_F32_KERNEL)?,
            arena,
        })
    }

    pub(crate) fn encode_f32_to_f16(
        &self,
        command_buffer: &CommandBufferRef,
        _device: &Device,
        input: &Buffer,
        len: usize,
    ) -> Result<Buffer> {
        let output = self.arena.empty_f16(len)?;
        let len_buffer = u32_scalar_len(&self.arena, len, "batched f32 to f16 cast")?;
        require_f32_capacity(input, len, "batched f32 to f16 input")?;
        encode_1d(
            command_buffer,
            &self.f32_to_f16,
            &[input, &output, &len_buffer],
            len,
        )?;
        Ok(output)
    }

    pub(crate) fn encode_q8_rows_to_f32(
        &self,
        command_buffer: &CommandBufferRef,
        _device: &Device,
        payload: &Buffer,
        payload_len: usize,
        row_count: usize,
        dim: usize,
    ) -> Result<Buffer> {
        validate_q8_rows_payload(payload, payload_len, row_count, dim)?;
        let output_len = row_count
            .checked_mul(dim)
            .ok_or_else(|| Error::backend("Q8 row f32 output length overflow"))?;
        let output = self.arena.empty_f32(output_len)?;
        let row_count_buffer = u32_scalar_len(&self.arena, row_count, "Q8 row decode row_count")?;
        let dim_buffer = u32_scalar_len(&self.arena, dim, "Q8 row decode dim")?;
        encode_1d(
            command_buffer,
            &self.q8_rows_to_f32,
            &[payload, &output, &row_count_buffer, &dim_buffer],
            output_len,
        )?;
        Ok(output)
    }

    pub(crate) fn read_f16_as_f32(&self, input: &Buffer, len: usize) -> Result<Vec<f32>> {
        read_f16_buffer_as_f32(input, len)
    }

    pub(crate) fn read_f32(&self, input: &Buffer, len: usize) -> Result<Vec<f32>> {
        read_f32_buffer(input, len)
    }
}

fn u32_scalar_len(arena: &MetalArena, len: usize, context: &str) -> Result<Buffer> {
    let len = u32::try_from(len)
        .map_err(|_| Error::backend(format!("{context} length exceeds Metal u32 limit")))?;
    arena.u32(len)
}

fn validate_q8_rows_payload(
    payload: &Buffer,
    payload_len: usize,
    row_count: usize,
    dim: usize,
) -> Result<()> {
    if row_count == 0 || dim == 0 {
        return Err(Error::backend(
            "Q8 row decode requires positive row_count and dim",
        ));
    }
    let row_bytes = dim
        .checked_add(4)
        .ok_or_else(|| Error::backend("Q8 row byte length overflow"))?;
    let expected_payload_len = row_count
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::backend("Q8 row payload length overflow"))?;
    if payload_len != expected_payload_len {
        return Err(Error::backend(format!(
            "Q8 row payload length mismatch: expected {expected_payload_len}, got {payload_len}"
        )));
    }
    require_byte_capacity(payload, payload_len, "Q8 row payload")
}
