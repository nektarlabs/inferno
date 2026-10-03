use ::metal::{Buffer, CommandBufferRef, ComputePipelineState, Device};
use common::{Error, Result};

use crate::DeviceQwenMlxW4Matrix;

use super::packed_prefill::MetalQwenPackedPrefill;

use super::super::{
    arena::MetalArena,
    buffers::{require_byte_capacity, require_f16_capacity, write_u32_buffer},
    command::{encode_1d_threadgroups_args, KernelArg},
    library::MetalLibrary,
    pipeline::compute_pipeline,
};

const SOURCE: &str = include_str!("kernels/mlx_w4.metal");
const LINEAR: &str = "qwen_mlx_w4_linear_bf16_kernel";
const VERIFY5_EXACT_LINEAR: &str = "qwen_mlx_w4_verify5_exact_linear_bf16_kernel";
const VERIFY8_EXACT_LINEAR: &str = "qwen_mlx_w4_verify8_exact_linear_bf16_kernel";
const VERIFY5_EXACT_LINEAR_ADD: &str = "qwen_mlx_w4_verify5_exact_linear_add_bf16_kernel";
const VERIFY8_EXACT_LINEAR_ADD: &str = "qwen_mlx_w4_verify8_exact_linear_add_bf16_kernel";
const PREFILL_MATRIX_LINEAR: &str = "qwen_mlx_w4_prefill_matrix_linear_bf16_kernel";
const PREFILL_WIDE_LINEAR: &str = "qwen_mlx_w4_prefill_wide_linear_bf16_kernel";
const GATE_UP: &str = "qwen_mlx_w4_gate_up_swiglu_bf16_kernel";
const VERIFY5_EXACT_GATE_UP: &str = "qwen_mlx_w4_verify5_exact_gate_up_swiglu_bf16_kernel";
const VERIFY8_EXACT_GATE_UP: &str = "qwen_mlx_w4_verify8_exact_gate_up_swiglu_bf16_kernel";
const PREFILL_MATRIX_GATE_UP: &str = "qwen_mlx_w4_prefill_matrix_gate_up_swiglu_bf16_kernel";
const DFLASH_TOP_K_LOCAL: &str = "qwen_mlx_w4_dflash_top_k_local_kernel";
const TOP_K_MERGE: &str = "qwen_mlx_w4_top_k_merge_kernel";
const EMBEDDING: &str = "qwen_mlx_w4_embedding_bf16_kernel";
const ARGMAX: &str = "qwen_mlx_w4_argmax_bf16_kernel";
const GROUP_SIZE: usize = 64;
const VALUES_PER_WORD: usize = 8;
const ROW_TILE: usize = 4;
const PREFILL_MATRIX_ROWS: usize = 8;
const PREFILL_LINEAR_OUTPUTS: usize = 16;
const PREFILL_GATE_UP_OUTPUTS: usize = 8;
const PREFILL_MIN_ROWS: usize = 16;
const PREFILL_WIDE_ROWS: usize = 16;
const PREFILL_WIDE_MIN_ROWS: usize = 64;
const PREFILL_WIDE_THREADS: usize = 512;
const QWEN_HIDDEN_SIZE: usize = 5_120;
const QWEN_INTERMEDIATE_SIZE: usize = 17_408;
const VERIFY_ROWS: usize = 5;
const VERIFY8_ROWS: usize = 8;
const LINEAR_THREADS: usize = 128;
const TOP_K: usize = 16;
const TOP_K_VOCAB_TILE: usize = 256;

#[cfg(test)]
#[path = "verification_tests.rs"]
mod verification_tests;

pub(super) struct MetalQwenMlxW4 {
    linear: ComputePipelineState,
    verify5_exact_linear: ComputePipelineState,
    verify8_exact_linear: ComputePipelineState,
    verify5_exact_linear_add: ComputePipelineState,
    verify8_exact_linear_add: ComputePipelineState,
    prefill_matrix_linear: ComputePipelineState,
    prefill_wide_linear: ComputePipelineState,
    gate_up: ComputePipelineState,
    verify5_exact_gate_up: ComputePipelineState,
    verify8_exact_gate_up: ComputePipelineState,
    prefill_matrix_gate_up: ComputePipelineState,
    packed_prefill: MetalQwenPackedPrefill,
    dflash_top_k_local: ComputePipelineState,
    top_k_merge: ComputePipelineState,
    embedding: ComputePipelineState,
    argmax: ComputePipelineState,
    arena: MetalArena,
}

impl MetalQwenMlxW4 {
    pub(super) fn new(device: &Device, arena: MetalArena) -> Result<Self> {
        let library = MetalLibrary::compile_source(device, SOURCE)?;
        Ok(Self {
            linear: compute_pipeline(device, &library, LINEAR)?,
            verify5_exact_linear: compute_pipeline(device, &library, VERIFY5_EXACT_LINEAR)?,
            verify8_exact_linear: compute_pipeline(device, &library, VERIFY8_EXACT_LINEAR)?,
            verify5_exact_linear_add: compute_pipeline(device, &library, VERIFY5_EXACT_LINEAR_ADD)?,
            verify8_exact_linear_add: compute_pipeline(device, &library, VERIFY8_EXACT_LINEAR_ADD)?,
            prefill_matrix_linear: compute_pipeline(device, &library, PREFILL_MATRIX_LINEAR)?,
            prefill_wide_linear: compute_pipeline(device, &library, PREFILL_WIDE_LINEAR)?,
            gate_up: compute_pipeline(device, &library, GATE_UP)?,
            verify5_exact_gate_up: compute_pipeline(device, &library, VERIFY5_EXACT_GATE_UP)?,
            verify8_exact_gate_up: compute_pipeline(device, &library, VERIFY8_EXACT_GATE_UP)?,
            prefill_matrix_gate_up: compute_pipeline(device, &library, PREFILL_MATRIX_GATE_UP)?,
            packed_prefill: MetalQwenPackedPrefill::new(device, arena.clone())?,
            dflash_top_k_local: compute_pipeline(device, &library, DFLASH_TOP_K_LOCAL)?,
            top_k_merge: compute_pipeline(device, &library, TOP_K_MERGE)?,
            embedding: compute_pipeline(device, &library, EMBEDDING)?,
            argmax: compute_pipeline(device, &library, ARGMAX)?,
            arena,
        })
    }

    pub(super) fn encode_linear(
        &self,
        command_buffer: &CommandBufferRef,
        matrix: &DeviceQwenMlxW4Matrix,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Buffer> {
        validate_linear(matrix, input, input_len, row_count)?;
        let output_len = row_count
            .checked_mul(matrix.rows)
            .ok_or_else(|| Error::backend("Qwen MLX W4 output length overflow"))?;
        let output = self.arena.empty_f16(output_len)?;
        self.encode_linear_into(command_buffer, matrix, input, &output, row_count, 0, 1)?;
        Ok(output)
    }

    pub(super) fn encode_linear_add(
        &self,
        command_buffer: &CommandBufferRef,
        matrix: &DeviceQwenMlxW4Matrix,
        input: &Buffer,
        residual: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Option<Buffer>> {
        validate_linear(matrix, input, input_len, row_count)?;
        let exact_verify = matches!(row_count, VERIFY_ROWS | VERIFY8_ROWS);
        if !exact_verify && !MetalQwenPackedPrefill::supports_linear(matrix, row_count) {
            return Ok(None);
        }
        let output_len = row_count
            .checked_mul(matrix.rows)
            .ok_or_else(|| Error::backend("Qwen MLX W4 residual output length overflow"))?;
        let required_bytes = output_len
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| Error::backend("Qwen MLX W4 residual byte size overflow"))?;
        if residual.length() < required_bytes as u64 {
            return Err(Error::backend(format!(
                "Qwen MLX W4 residual buffer is too small: need {required_bytes} bytes, got {}",
                residual.length()
            )));
        }
        let output = self.arena.empty_f16(output_len)?;
        if exact_verify {
            let (pipeline, output_tile) = if row_count == VERIFY_ROWS {
                (&self.verify5_exact_linear_add, 4)
            } else {
                (&self.verify8_exact_linear_add, 4)
            };
            encode_1d_threadgroups_args(
                command_buffer,
                pipeline,
                &[
                    KernelArg::BufferOffset(&matrix.weight, matrix.weight_byte_offset),
                    KernelArg::BufferOffset(&matrix.scales, matrix.scale_byte_offset),
                    KernelArg::BufferOffset(&matrix.biases, matrix.bias_byte_offset),
                    KernelArg::Buffer(input),
                    KernelArg::Buffer(residual),
                    KernelArg::Buffer(&output),
                    KernelArg::U32(as_u32(matrix.columns, "input width")?),
                    KernelArg::U32(as_u32(matrix.rows, "output width")?),
                ],
                matrix.rows.div_ceil(output_tile),
                LINEAR_THREADS,
            )?;
            return Ok(Some(output));
        }
        self.packed_prefill.encode_linear_add_into(
            command_buffer,
            matrix,
            input,
            residual,
            &output,
            row_count,
        )?;
        Ok(Some(output))
    }

    fn encode_linear_into(
        &self,
        command_buffer: &CommandBufferRef,
        matrix: &DeviceQwenMlxW4Matrix,
        input: &Buffer,
        output: &Buffer,
        row_count: usize,
        first_input_row: usize,
        input_row_stride: usize,
    ) -> Result<()> {
        if row_count == VERIFY8_ROWS && first_input_row == 0 && input_row_stride == 1 {
            return encode_1d_threadgroups_args(
                command_buffer,
                &self.verify8_exact_linear,
                &[
                    KernelArg::BufferOffset(&matrix.weight, matrix.weight_byte_offset),
                    KernelArg::BufferOffset(&matrix.scales, matrix.scale_byte_offset),
                    KernelArg::BufferOffset(&matrix.biases, matrix.bias_byte_offset),
                    KernelArg::Buffer(input),
                    KernelArg::Buffer(output),
                    KernelArg::U32(as_u32(matrix.columns, "input width")?),
                    KernelArg::U32(as_u32(matrix.rows, "output width")?),
                ],
                matrix.rows.div_ceil(4),
                LINEAR_THREADS,
            );
        }
        if row_count == VERIFY_ROWS && first_input_row == 0 && input_row_stride == 1 {
            return encode_1d_threadgroups_args(
                command_buffer,
                &self.verify5_exact_linear,
                &[
                    KernelArg::BufferOffset(&matrix.weight, matrix.weight_byte_offset),
                    KernelArg::BufferOffset(&matrix.scales, matrix.scale_byte_offset),
                    KernelArg::BufferOffset(&matrix.biases, matrix.bias_byte_offset),
                    KernelArg::Buffer(input),
                    KernelArg::Buffer(output),
                    KernelArg::U32(as_u32(matrix.columns, "input width")?),
                    KernelArg::U32(as_u32(matrix.rows, "output width")?),
                ],
                matrix.rows.div_ceil(4),
                LINEAR_THREADS,
            );
        }
        if row_count >= PREFILL_MIN_ROWS
            && first_input_row == 0
            && input_row_stride == 1
            && supports_prefill_matrix(matrix)
        {
            if MetalQwenPackedPrefill::supports_linear(matrix, row_count) {
                return self.packed_prefill.encode_linear_into(
                    command_buffer,
                    matrix,
                    input,
                    output,
                    row_count,
                );
            }
            if row_count >= PREFILL_WIDE_MIN_ROWS
                && matrix.rows == QWEN_HIDDEN_SIZE
                && matrix.columns == QWEN_INTERMEDIATE_SIZE
            {
                let threadgroups = row_count
                    .div_ceil(PREFILL_WIDE_ROWS)
                    .checked_mul(matrix.rows.div_ceil(PREFILL_LINEAR_OUTPUTS))
                    .ok_or_else(|| {
                        Error::backend("Qwen MLX W4 wide prefill threadgroup count overflow")
                    })?;
                return encode_1d_threadgroups_args(
                    command_buffer,
                    &self.prefill_wide_linear,
                    &[
                        KernelArg::BufferOffset(&matrix.weight, matrix.weight_byte_offset),
                        KernelArg::BufferOffset(&matrix.scales, matrix.scale_byte_offset),
                        KernelArg::BufferOffset(&matrix.biases, matrix.bias_byte_offset),
                        KernelArg::Buffer(input),
                        KernelArg::Buffer(output),
                        KernelArg::U32(as_u32(row_count, "row count")?),
                        KernelArg::U32(as_u32(matrix.columns, "input width")?),
                        KernelArg::U32(as_u32(matrix.rows, "output width")?),
                    ],
                    threadgroups,
                    PREFILL_WIDE_THREADS,
                );
            }
            let threadgroups = row_count
                .div_ceil(PREFILL_MATRIX_ROWS)
                .checked_mul(matrix.rows.div_ceil(PREFILL_LINEAR_OUTPUTS))
                .ok_or_else(|| {
                    Error::backend("Qwen MLX W4 matrix prefill threadgroup count overflow")
                })?;
            return encode_1d_threadgroups_args(
                command_buffer,
                &self.prefill_matrix_linear,
                &[
                    KernelArg::BufferOffset(&matrix.weight, matrix.weight_byte_offset),
                    KernelArg::BufferOffset(&matrix.scales, matrix.scale_byte_offset),
                    KernelArg::BufferOffset(&matrix.biases, matrix.bias_byte_offset),
                    KernelArg::Buffer(input),
                    KernelArg::Buffer(output),
                    KernelArg::U32(as_u32(row_count, "row count")?),
                    KernelArg::U32(as_u32(matrix.columns, "input width")?),
                    KernelArg::U32(as_u32(matrix.rows, "output width")?),
                ],
                threadgroups,
                256,
            );
        }
        let threadgroups = matrix
            .rows
            .checked_mul(row_count.div_ceil(ROW_TILE))
            .ok_or_else(|| Error::backend("Qwen MLX W4 threadgroup count overflow"))?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.linear,
            &[
                KernelArg::BufferOffset(&matrix.weight, matrix.weight_byte_offset),
                KernelArg::BufferOffset(&matrix.scales, matrix.scale_byte_offset),
                KernelArg::BufferOffset(&matrix.biases, matrix.bias_byte_offset),
                KernelArg::Buffer(input),
                KernelArg::Buffer(output),
                KernelArg::U32(as_u32(row_count, "row count")?),
                KernelArg::U32(as_u32(matrix.columns, "input width")?),
                KernelArg::U32(as_u32(matrix.rows, "output width")?),
                KernelArg::U32(as_u32(first_input_row, "first input row")?),
                KernelArg::U32(as_u32(input_row_stride, "input row stride")?),
            ],
            threadgroups,
            LINEAR_THREADS,
        )
    }

    /// Projects up to seven DFlash draft rows and retains only the exact
    /// top-16 logits for each row. This avoids materializing the full
    /// `[rows, vocabulary]` logits tensor used only by the selector.
    pub(super) fn encode_top_k(
        &self,
        command_buffer: &CommandBufferRef,
        matrix: &DeviceQwenMlxW4Matrix,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
        top_k: usize,
    ) -> Result<(Buffer, Buffer)> {
        validate_linear(matrix, input, input_len, row_count)?;
        if !(1..=7).contains(&row_count) {
            return Err(Error::backend(format!(
                "Qwen MLX W4 DFlash top-k requires 1..=7 rows, got {row_count}"
            )));
        }
        if top_k != TOP_K {
            return Err(Error::backend(format!(
                "Qwen MLX W4 fused top-k requires top_k={TOP_K}, got {top_k}"
            )));
        }

        let tile_count = matrix.rows.div_ceil(TOP_K_VOCAB_TILE);
        let local_count = row_count
            .checked_mul(tile_count)
            .and_then(|count| count.checked_mul(TOP_K))
            .ok_or_else(|| Error::backend("Qwen MLX W4 local top-k size overflow"))?;
        let local_ids = self.arena.empty_u32(local_count)?;
        let local_values = self.arena.empty_f32(local_count)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.dflash_top_k_local,
            &[
                KernelArg::BufferOffset(&matrix.weight, matrix.weight_byte_offset),
                KernelArg::BufferOffset(&matrix.scales, matrix.scale_byte_offset),
                KernelArg::BufferOffset(&matrix.biases, matrix.bias_byte_offset),
                KernelArg::Buffer(input),
                KernelArg::Buffer(&local_ids),
                KernelArg::Buffer(&local_values),
                KernelArg::U32(as_u32(row_count, "DFlash rows")?),
                KernelArg::U32(as_u32(matrix.columns, "input width")?),
                KernelArg::U32(as_u32(matrix.rows, "vocabulary size")?),
                KernelArg::U32(as_u32(tile_count, "vocabulary tile count")?),
            ],
            tile_count * row_count.div_ceil(ROW_TILE),
            LINEAR_THREADS,
        )?;

        let output_count = row_count
            .checked_mul(top_k)
            .ok_or_else(|| Error::backend("Qwen MLX W4 top-k output size overflow"))?;
        let candidate_ids = self.arena.empty_u32(output_count)?;
        let candidate_values = self.arena.empty_f32(output_count)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.top_k_merge,
            &[
                KernelArg::Buffer(&local_ids),
                KernelArg::Buffer(&local_values),
                KernelArg::Buffer(&candidate_ids),
                KernelArg::Buffer(&candidate_values),
                KernelArg::U32(as_u32(row_count, "top-k rows")?),
                KernelArg::U32(as_u32(tile_count, "top-k tile count")?),
            ],
            row_count,
            LINEAR_THREADS,
        )?;
        Ok((candidate_ids, candidate_values))
    }

    pub(super) fn encode_pair(
        &self,
        command_buffer: &CommandBufferRef,
        first: &DeviceQwenMlxW4Matrix,
        second: &DeviceQwenMlxW4Matrix,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<(Buffer, Buffer)> {
        if first.columns != second.columns {
            return Err(Error::backend("Qwen MLX W4 paired input widths differ"));
        }
        let first_output =
            self.encode_linear(command_buffer, first, input, input_len, row_count)?;
        let second_output =
            self.encode_linear(command_buffer, second, input, input_len, row_count)?;
        Ok((first_output, second_output))
    }

    pub(super) fn encode_qkv(
        &self,
        command_buffer: &CommandBufferRef,
        query: &DeviceQwenMlxW4Matrix,
        key: &DeviceQwenMlxW4Matrix,
        value: &DeviceQwenMlxW4Matrix,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<(Buffer, Buffer, Buffer)> {
        if query.columns != key.columns || query.columns != value.columns {
            return Err(Error::backend("Qwen MLX W4 Q/K/V input widths differ"));
        }
        Ok((
            self.encode_linear(command_buffer, query, input, input_len, row_count)?,
            self.encode_linear(command_buffer, key, input, input_len, row_count)?,
            self.encode_linear(command_buffer, value, input, input_len, row_count)?,
        ))
    }

    pub(super) fn encode_gate_up(
        &self,
        command_buffer: &CommandBufferRef,
        gate: &DeviceQwenMlxW4Matrix,
        up: &DeviceQwenMlxW4Matrix,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Buffer> {
        if gate.rows != up.rows || gate.columns != up.columns {
            return Err(Error::backend("Qwen MLX W4 gate/up shapes differ"));
        }
        validate_linear(gate, input, input_len, row_count)?;
        validate_matrix_ranges(up, "Qwen MLX W4 up")?;
        let output_len = row_count
            .checked_mul(gate.rows)
            .ok_or_else(|| Error::backend("Qwen MLX W4 SwiGLU output length overflow"))?;
        let output = self.arena.empty_f16(output_len)?;
        if row_count == VERIFY8_ROWS {
            encode_1d_threadgroups_args(
                command_buffer,
                &self.verify8_exact_gate_up,
                &[
                    KernelArg::BufferOffset(&gate.weight, gate.weight_byte_offset),
                    KernelArg::BufferOffset(&gate.scales, gate.scale_byte_offset),
                    KernelArg::BufferOffset(&gate.biases, gate.bias_byte_offset),
                    KernelArg::BufferOffset(&up.weight, up.weight_byte_offset),
                    KernelArg::BufferOffset(&up.scales, up.scale_byte_offset),
                    KernelArg::BufferOffset(&up.biases, up.bias_byte_offset),
                    KernelArg::Buffer(input),
                    KernelArg::Buffer(&output),
                    KernelArg::U32(as_u32(gate.columns, "input width")?),
                    KernelArg::U32(as_u32(gate.rows, "output width")?),
                ],
                gate.rows.div_ceil(2),
                LINEAR_THREADS,
            )?;
            return Ok(output);
        }
        if row_count == VERIFY_ROWS {
            encode_1d_threadgroups_args(
                command_buffer,
                &self.verify5_exact_gate_up,
                &[
                    KernelArg::BufferOffset(&gate.weight, gate.weight_byte_offset),
                    KernelArg::BufferOffset(&gate.scales, gate.scale_byte_offset),
                    KernelArg::BufferOffset(&gate.biases, gate.bias_byte_offset),
                    KernelArg::BufferOffset(&up.weight, up.weight_byte_offset),
                    KernelArg::BufferOffset(&up.scales, up.scale_byte_offset),
                    KernelArg::BufferOffset(&up.biases, up.bias_byte_offset),
                    KernelArg::Buffer(input),
                    KernelArg::Buffer(&output),
                    KernelArg::U32(as_u32(gate.columns, "input width")?),
                    KernelArg::U32(as_u32(gate.rows, "output width")?),
                ],
                gate.rows.div_ceil(2),
                LINEAR_THREADS,
            )?;
            return Ok(output);
        }
        if row_count >= PREFILL_MIN_ROWS && supports_prefill_matrix(gate) {
            if MetalQwenPackedPrefill::supports_gate_up(gate, row_count) {
                return self.packed_prefill.encode_gate_up(
                    command_buffer,
                    gate,
                    up,
                    input,
                    row_count,
                );
            }
            let threadgroups = row_count
                .div_ceil(PREFILL_MATRIX_ROWS)
                .checked_mul(gate.rows.div_ceil(PREFILL_GATE_UP_OUTPUTS))
                .ok_or_else(|| {
                    Error::backend("Qwen MLX W4 matrix prefill SwiGLU threadgroup count overflow")
                })?;
            encode_1d_threadgroups_args(
                command_buffer,
                &self.prefill_matrix_gate_up,
                &[
                    KernelArg::BufferOffset(&gate.weight, gate.weight_byte_offset),
                    KernelArg::BufferOffset(&gate.scales, gate.scale_byte_offset),
                    KernelArg::BufferOffset(&gate.biases, gate.bias_byte_offset),
                    KernelArg::BufferOffset(&up.weight, up.weight_byte_offset),
                    KernelArg::BufferOffset(&up.scales, up.scale_byte_offset),
                    KernelArg::BufferOffset(&up.biases, up.bias_byte_offset),
                    KernelArg::Buffer(input),
                    KernelArg::Buffer(&output),
                    KernelArg::U32(as_u32(row_count, "row count")?),
                    KernelArg::U32(as_u32(gate.columns, "input width")?),
                    KernelArg::U32(as_u32(gate.rows, "output width")?),
                ],
                threadgroups,
                256,
            )?;
            return Ok(output);
        }
        let threadgroups = gate
            .rows
            .checked_mul(row_count.div_ceil(ROW_TILE))
            .ok_or_else(|| Error::backend("Qwen MLX W4 SwiGLU threadgroup count overflow"))?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.gate_up,
            &[
                KernelArg::BufferOffset(&gate.weight, gate.weight_byte_offset),
                KernelArg::BufferOffset(&gate.scales, gate.scale_byte_offset),
                KernelArg::BufferOffset(&gate.biases, gate.bias_byte_offset),
                KernelArg::BufferOffset(&up.weight, up.weight_byte_offset),
                KernelArg::BufferOffset(&up.scales, up.scale_byte_offset),
                KernelArg::BufferOffset(&up.biases, up.bias_byte_offset),
                KernelArg::Buffer(input),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(row_count, "row count")?),
                KernelArg::U32(as_u32(gate.columns, "input width")?),
                KernelArg::U32(as_u32(gate.rows, "output width")?),
            ],
            threadgroups,
            LINEAR_THREADS,
        )?;
        Ok(output)
    }

    pub(super) fn encode_embedding(
        &self,
        command_buffer: &CommandBufferRef,
        matrix: &DeviceQwenMlxW4Matrix,
        token_ids: &[u32],
    ) -> Result<Buffer> {
        if token_ids.is_empty() || token_ids.iter().any(|id| *id as usize >= matrix.rows) {
            return Err(Error::backend(
                "Qwen MLX W4 embedding token is out of range",
            ));
        }
        let ids = self.arena.empty_u32(token_ids.len())?;
        write_u32_buffer(&ids, token_ids)?;
        self.encode_embedding_from_device(command_buffer, matrix, &ids, token_ids.len())
    }

    pub(super) fn encode_embedding_from_device(
        &self,
        command_buffer: &CommandBufferRef,
        matrix: &DeviceQwenMlxW4Matrix,
        token_ids: &Buffer,
        token_count: usize,
    ) -> Result<Buffer> {
        validate_matrix_ranges(matrix, "Qwen MLX W4 embedding")?;
        require_byte_capacity(
            token_ids,
            token_count * std::mem::size_of::<u32>(),
            "Qwen MLX W4 token IDs",
        )?;
        let output_len = token_count
            .checked_mul(matrix.columns)
            .ok_or_else(|| Error::backend("Qwen MLX W4 embedding length overflow"))?;
        let output = self.arena.empty_f16(output_len)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.embedding,
            &[
                KernelArg::BufferOffset(&matrix.weight, matrix.weight_byte_offset),
                KernelArg::BufferOffset(&matrix.scales, matrix.scale_byte_offset),
                KernelArg::BufferOffset(&matrix.biases, matrix.bias_byte_offset),
                KernelArg::Buffer(token_ids),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(token_count, "token count")?),
                KernelArg::U32(as_u32(matrix.columns, "hidden size")?),
            ],
            output_len.div_ceil(256),
            256,
        )?;
        Ok(output)
    }

    pub(super) fn encode_argmax(
        &self,
        command_buffer: &CommandBufferRef,
        matrix: &DeviceQwenMlxW4Matrix,
        hidden: &Buffer,
        hidden_len: usize,
        row_count: usize,
    ) -> Result<Buffer> {
        let logits = self.encode_linear(command_buffer, matrix, hidden, hidden_len, row_count)?;
        let ids = self.arena.empty_u32(row_count)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.argmax,
            &[
                KernelArg::Buffer(&logits),
                KernelArg::Buffer(&ids),
                KernelArg::U32(as_u32(row_count, "argmax rows")?),
                KernelArg::U32(as_u32(matrix.rows, "vocabulary size")?),
            ],
            row_count,
            256,
        )?;
        Ok(ids)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_last_argmax(
        &self,
        command_buffer: &CommandBufferRef,
        matrix: &DeviceQwenMlxW4Matrix,
        hidden: &Buffer,
        batch: usize,
        sequence_length: usize,
    ) -> Result<Buffer> {
        let input_rows = batch
            .checked_mul(sequence_length)
            .ok_or_else(|| Error::backend("Qwen MLX W4 hidden row count overflow"))?;
        validate_linear(matrix, hidden, input_rows * matrix.columns, input_rows)?;
        let logits = self.arena.empty_f16(batch * matrix.rows)?;
        self.encode_linear_into(
            command_buffer,
            matrix,
            hidden,
            &logits,
            batch,
            sequence_length - 1,
            sequence_length,
        )?;
        let ids = self.arena.empty_u32(batch)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.argmax,
            &[
                KernelArg::Buffer(&logits),
                KernelArg::Buffer(&ids),
                KernelArg::U32(as_u32(batch, "argmax batch")?),
                KernelArg::U32(as_u32(matrix.rows, "vocabulary size")?),
            ],
            batch,
            256,
        )?;
        Ok(ids)
    }
}

fn validate_linear(
    matrix: &DeviceQwenMlxW4Matrix,
    input: &Buffer,
    input_len: usize,
    row_count: usize,
) -> Result<()> {
    if row_count == 0 || input_len != row_count.saturating_mul(matrix.columns) {
        return Err(Error::backend(format!(
            "Qwen MLX W4 input length {input_len} does not match [{row_count},{}]",
            matrix.columns
        )));
    }
    require_f16_capacity(input, input_len, "Qwen MLX W4 input")?;
    validate_matrix_ranges(matrix, "Qwen MLX W4 matrix")
}

fn validate_matrix_ranges(matrix: &DeviceQwenMlxW4Matrix, label: &str) -> Result<()> {
    if matrix.rows == 0
        || matrix.columns == 0
        || !matrix.columns.is_multiple_of(GROUP_SIZE)
        || matrix.groups_per_row != matrix.columns / GROUP_SIZE
    {
        return Err(Error::backend(format!("invalid {label} dimensions")));
    }
    let weight_bytes = matrix
        .rows
        .checked_mul(matrix.columns / VALUES_PER_WORD)
        .and_then(|words| words.checked_mul(std::mem::size_of::<u32>()))
        .ok_or_else(|| Error::backend(format!("{label} weight size overflow")))?;
    let parameter_bytes = matrix
        .rows
        .checked_mul(matrix.groups_per_row)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .ok_or_else(|| Error::backend(format!("{label} parameter size overflow")))?;
    require_range(
        &matrix.weight,
        matrix.weight_byte_offset,
        weight_bytes,
        label,
    )?;
    require_range(
        &matrix.scales,
        matrix.scale_byte_offset,
        parameter_bytes,
        label,
    )?;
    require_range(
        &matrix.biases,
        matrix.bias_byte_offset,
        parameter_bytes,
        label,
    )
}

fn supports_prefill_matrix(matrix: &DeviceQwenMlxW4Matrix) -> bool {
    matrix.columns.is_multiple_of(256)
}

fn require_range(buffer: &Buffer, offset: usize, bytes: usize, label: &str) -> Result<()> {
    let end = offset
        .checked_add(bytes)
        .ok_or_else(|| Error::backend(format!("{label} range overflow")))?;
    require_byte_capacity(buffer, end, label)
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::backend(format!("{label} does not fit u32")))
}

#[cfg(test)]
mod tests {
    use ::metal::{MTLCommandBufferStatus, MTLResourceOptions};

    use super::super::super::buffers::{read_bf16_buffer_as_f32, read_f32_buffer, read_u32_buffer};
    use super::*;

    const TEST_ROWS: usize = 5;
    const TEST_INPUT_WIDTH: usize = 64;
    const TEST_OUTPUT_WIDTH: usize = 3;
    const MATRIX_INPUT_WIDTH: usize = 512;
    const MATRIX_OUTPUT_WIDTH: usize = 32;
    const TOP_K_TEST_VOCAB: usize = 512;

    #[test]
    fn affine_w4_linear_matches_cpu_reference() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let arena = MetalArena::new(&device).unwrap();
        let w4 = MetalQwenMlxW4::new(&device, arena.clone()).unwrap();
        let input = test_input(TEST_ROWS);
        let packed = packed_rows(3);
        let scales = bf16_bits(&[0.125, -0.0625, 0.25]);
        let biases = bf16_bits(&[-0.5, 0.75, -1.0]);
        let options = MTLResourceOptions::StorageModeShared;
        let input_buffer = buffer(&device, &bf16_bits(&input), options);
        let packed_buffer = buffer(&device, &packed, options);
        let scale_buffer = buffer(&device, &scales, options);
        let bias_buffer = buffer(&device, &biases, options);
        let output = arena.empty_f16(TEST_ROWS * TEST_OUTPUT_WIDTH).unwrap();
        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        encode_1d_threadgroups_args(
            command_buffer,
            &w4.linear,
            &[
                KernelArg::Buffer(&packed_buffer),
                KernelArg::Buffer(&scale_buffer),
                KernelArg::Buffer(&bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&output),
                KernelArg::U32(TEST_ROWS as u32),
                KernelArg::U32(TEST_INPUT_WIDTH as u32),
                KernelArg::U32(TEST_OUTPUT_WIDTH as u32),
                KernelArg::U32(0),
                KernelArg::U32(1),
            ],
            TEST_OUTPUT_WIDTH * TEST_ROWS.div_ceil(ROW_TILE),
            LINEAR_THREADS,
        )
        .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);

        let actual = read_bf16_buffer_as_f32(&output, TEST_ROWS * TEST_OUTPUT_WIDTH).unwrap();
        let verify_output = arena.empty_f16(TEST_ROWS * TEST_OUTPUT_WIDTH).unwrap();
        let command_buffer = queue.new_command_buffer();
        encode_1d_threadgroups_args(
            command_buffer,
            &w4.verify5_exact_linear,
            &[
                KernelArg::Buffer(&packed_buffer),
                KernelArg::Buffer(&scale_buffer),
                KernelArg::Buffer(&bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&verify_output),
                KernelArg::U32(TEST_INPUT_WIDTH as u32),
                KernelArg::U32(TEST_OUTPUT_WIDTH as u32),
            ],
            TEST_OUTPUT_WIDTH.div_ceil(4),
            LINEAR_THREADS,
        )
        .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        let verify_actual =
            read_bf16_buffer_as_f32(&verify_output, TEST_ROWS * TEST_OUTPUT_WIDTH).unwrap();
        assert_eq!(verify_actual, actual);
        let expected = affine_reference(
            &input,
            &packed,
            &[0.125, -0.0625, 0.25],
            &[-0.5, 0.75, -1.0],
        );
        assert_bf16_close(&actual, &expected);
    }

    #[test]
    fn affine_w4_gate_up_swiglu_matches_cpu_reference() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let arena = MetalArena::new(&device).unwrap();
        let w4 = MetalQwenMlxW4::new(&device, arena.clone()).unwrap();
        let input = test_input(TEST_ROWS);
        let gate_packed = packed_rows(5);
        let up_packed = packed_rows(11);
        let gate_scales_f32 = [0.0625, -0.125, 0.25];
        let gate_biases_f32 = [-0.25, 0.5, -0.75];
        let up_scales_f32 = [-0.03125, 0.125, 0.0625];
        let up_biases_f32 = [0.375, -0.5, 0.125];
        let options = MTLResourceOptions::StorageModeShared;
        let input_buffer = buffer(&device, &bf16_bits(&input), options);
        let gate_packed_buffer = buffer(&device, &gate_packed, options);
        let up_packed_buffer = buffer(&device, &up_packed, options);
        let gate_scale_buffer = buffer(&device, &bf16_bits(&gate_scales_f32), options);
        let gate_bias_buffer = buffer(&device, &bf16_bits(&gate_biases_f32), options);
        let up_scale_buffer = buffer(&device, &bf16_bits(&up_scales_f32), options);
        let up_bias_buffer = buffer(&device, &bf16_bits(&up_biases_f32), options);
        let output = arena.empty_f16(TEST_ROWS * TEST_OUTPUT_WIDTH).unwrap();
        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        encode_1d_threadgroups_args(
            command_buffer,
            &w4.gate_up,
            &[
                KernelArg::Buffer(&gate_packed_buffer),
                KernelArg::Buffer(&gate_scale_buffer),
                KernelArg::Buffer(&gate_bias_buffer),
                KernelArg::Buffer(&up_packed_buffer),
                KernelArg::Buffer(&up_scale_buffer),
                KernelArg::Buffer(&up_bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&output),
                KernelArg::U32(TEST_ROWS as u32),
                KernelArg::U32(TEST_INPUT_WIDTH as u32),
                KernelArg::U32(TEST_OUTPUT_WIDTH as u32),
            ],
            TEST_OUTPUT_WIDTH * TEST_ROWS.div_ceil(ROW_TILE),
            LINEAR_THREADS,
        )
        .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);

        let actual = read_bf16_buffer_as_f32(&output, TEST_ROWS * TEST_OUTPUT_WIDTH).unwrap();
        let verify_output = arena.empty_f16(TEST_ROWS * TEST_OUTPUT_WIDTH).unwrap();
        let command_buffer = queue.new_command_buffer();
        encode_1d_threadgroups_args(
            command_buffer,
            &w4.verify5_exact_gate_up,
            &[
                KernelArg::Buffer(&gate_packed_buffer),
                KernelArg::Buffer(&gate_scale_buffer),
                KernelArg::Buffer(&gate_bias_buffer),
                KernelArg::Buffer(&up_packed_buffer),
                KernelArg::Buffer(&up_scale_buffer),
                KernelArg::Buffer(&up_bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&verify_output),
                KernelArg::U32(TEST_INPUT_WIDTH as u32),
                KernelArg::U32(TEST_OUTPUT_WIDTH as u32),
            ],
            TEST_OUTPUT_WIDTH.div_ceil(2),
            LINEAR_THREADS,
        )
        .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        let verify_actual =
            read_bf16_buffer_as_f32(&verify_output, TEST_ROWS * TEST_OUTPUT_WIDTH).unwrap();
        assert_eq!(verify_actual, actual);
        let gate = affine_reference(&input, &gate_packed, &gate_scales_f32, &gate_biases_f32);
        let up = affine_reference(&input, &up_packed, &up_scales_f32, &up_biases_f32);
        let expected = gate
            .into_iter()
            .zip(up)
            .map(|(gate, up)| {
                let gate = round_bf16(gate);
                let up = round_bf16(up);
                let activated = round_bf16(gate / (1.0 + (-gate).exp()));
                round_bf16(activated * up)
            })
            .collect::<Vec<_>>();
        assert_bf16_close(&actual, &expected);
    }

    #[test]
    fn verify8_exact_kernels_match_generic_bitwise() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let arena = MetalArena::new(&device).unwrap();
        let w4 = MetalQwenMlxW4::new(&device, arena.clone()).unwrap();
        let input = test_input(VERIFY8_ROWS);
        let gate_packed = packed_rows(5);
        let up_packed = packed_rows(11);
        let gate_scales = [0.0625, -0.125, 0.25];
        let gate_biases = [-0.25, 0.5, -0.75];
        let up_scales = [-0.03125, 0.125, 0.0625];
        let up_biases = [0.375, -0.5, 0.125];
        let options = MTLResourceOptions::StorageModeShared;
        let input_buffer = buffer(&device, &bf16_bits(&input), options);
        let gate_packed_buffer = buffer(&device, &gate_packed, options);
        let up_packed_buffer = buffer(&device, &up_packed, options);
        let gate_scale_buffer = buffer(&device, &bf16_bits(&gate_scales), options);
        let gate_bias_buffer = buffer(&device, &bf16_bits(&gate_biases), options);
        let up_scale_buffer = buffer(&device, &bf16_bits(&up_scales), options);
        let up_bias_buffer = buffer(&device, &bf16_bits(&up_biases), options);
        let linear_output = arena.empty_f16(VERIFY8_ROWS * TEST_OUTPUT_WIDTH).unwrap();
        let generic_output = arena.empty_f16(VERIFY8_ROWS * TEST_OUTPUT_WIDTH).unwrap();
        let gate_up_output = arena.empty_f16(VERIFY8_ROWS * TEST_OUTPUT_WIDTH).unwrap();
        let generic_gate_up_output = arena.empty_f16(VERIFY8_ROWS * TEST_OUTPUT_WIDTH).unwrap();
        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        encode_1d_threadgroups_args(
            command_buffer,
            &w4.verify8_exact_linear,
            &[
                KernelArg::Buffer(&gate_packed_buffer),
                KernelArg::Buffer(&gate_scale_buffer),
                KernelArg::Buffer(&gate_bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&linear_output),
                KernelArg::U32(TEST_INPUT_WIDTH as u32),
                KernelArg::U32(TEST_OUTPUT_WIDTH as u32),
            ],
            TEST_OUTPUT_WIDTH.div_ceil(2),
            LINEAR_THREADS,
        )
        .unwrap();
        encode_1d_threadgroups_args(
            command_buffer,
            &w4.linear,
            &[
                KernelArg::Buffer(&gate_packed_buffer),
                KernelArg::Buffer(&gate_scale_buffer),
                KernelArg::Buffer(&gate_bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&generic_output),
                KernelArg::U32(VERIFY8_ROWS as u32),
                KernelArg::U32(TEST_INPUT_WIDTH as u32),
                KernelArg::U32(TEST_OUTPUT_WIDTH as u32),
                KernelArg::U32(0),
                KernelArg::U32(1),
            ],
            TEST_OUTPUT_WIDTH * VERIFY8_ROWS.div_ceil(ROW_TILE),
            LINEAR_THREADS,
        )
        .unwrap();
        encode_1d_threadgroups_args(
            command_buffer,
            &w4.verify8_exact_gate_up,
            &[
                KernelArg::Buffer(&gate_packed_buffer),
                KernelArg::Buffer(&gate_scale_buffer),
                KernelArg::Buffer(&gate_bias_buffer),
                KernelArg::Buffer(&up_packed_buffer),
                KernelArg::Buffer(&up_scale_buffer),
                KernelArg::Buffer(&up_bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&gate_up_output),
                KernelArg::U32(TEST_INPUT_WIDTH as u32),
                KernelArg::U32(TEST_OUTPUT_WIDTH as u32),
            ],
            TEST_OUTPUT_WIDTH,
            LINEAR_THREADS,
        )
        .unwrap();
        encode_1d_threadgroups_args(
            command_buffer,
            &w4.gate_up,
            &[
                KernelArg::Buffer(&gate_packed_buffer),
                KernelArg::Buffer(&gate_scale_buffer),
                KernelArg::Buffer(&gate_bias_buffer),
                KernelArg::Buffer(&up_packed_buffer),
                KernelArg::Buffer(&up_scale_buffer),
                KernelArg::Buffer(&up_bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&generic_gate_up_output),
                KernelArg::U32(VERIFY8_ROWS as u32),
                KernelArg::U32(TEST_INPUT_WIDTH as u32),
                KernelArg::U32(TEST_OUTPUT_WIDTH as u32),
            ],
            TEST_OUTPUT_WIDTH * VERIFY8_ROWS.div_ceil(ROW_TILE),
            LINEAR_THREADS,
        )
        .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);

        let linear_actual =
            read_bf16_buffer_as_f32(&linear_output, VERIFY8_ROWS * TEST_OUTPUT_WIDTH).unwrap();
        let generic_actual =
            read_bf16_buffer_as_f32(&generic_output, VERIFY8_ROWS * TEST_OUTPUT_WIDTH).unwrap();
        assert_eq!(linear_actual, generic_actual);
        let linear_expected = affine_reference(&input, &gate_packed, &gate_scales, &gate_biases);
        assert_bf16_close(&linear_actual, &linear_expected);

        let gate = affine_reference(&input, &gate_packed, &gate_scales, &gate_biases);
        let up = affine_reference(&input, &up_packed, &up_scales, &up_biases);
        let gate_up_expected = gate
            .into_iter()
            .zip(up)
            .map(|(gate, up)| {
                let gate = round_bf16(gate);
                let up = round_bf16(up);
                let activated = round_bf16(gate / (1.0 + (-gate).exp()));
                round_bf16(activated * up)
            })
            .collect::<Vec<_>>();
        let gate_up_actual =
            read_bf16_buffer_as_f32(&gate_up_output, VERIFY8_ROWS * TEST_OUTPUT_WIDTH).unwrap();
        let generic_gate_up_actual =
            read_bf16_buffer_as_f32(&generic_gate_up_output, VERIFY8_ROWS * TEST_OUTPUT_WIDTH)
                .unwrap();
        assert_eq!(gate_up_actual, generic_gate_up_actual);
        assert_bf16_close(&gate_up_actual, &gate_up_expected);
    }

    #[test]
    fn exact_verify_linear_add_matches_two_step_bf16_path_bitwise() {
        for row_count in [VERIFY_ROWS, VERIFY8_ROWS] {
            assert_exact_verify_linear_add(row_count);
        }
    }

    #[test]
    fn fused_top_k_matches_materialized_logits_for_dflash_rows() {
        for row_count in [1, 3, 4, 7] {
            assert_fused_top_k_matches_materialized_logits(row_count);
        }
    }

    #[test]
    fn prefill_matrix_kernels_match_cpu_reference() {
        const ROWS: usize = 17;

        let Some(device) = Device::system_default() else {
            return;
        };
        let arena = MetalArena::new(&device).unwrap();
        let w4 = MetalQwenMlxW4::new(&device, arena.clone()).unwrap();
        let input = test_input_for(ROWS, MATRIX_INPUT_WIDTH);
        let gate_packed = packed_matrix(MATRIX_OUTPUT_WIDTH, MATRIX_INPUT_WIDTH, 5);
        let up_packed = packed_matrix(MATRIX_OUTPUT_WIDTH, MATRIX_INPUT_WIDTH, 11);
        let parameter_count = MATRIX_OUTPUT_WIDTH * (MATRIX_INPUT_WIDTH / GROUP_SIZE);
        let gate_scales = (0..parameter_count)
            .map(|index| round_bf16(((index % 5) as f32 + 1.0) / 64.0))
            .collect::<Vec<_>>();
        let gate_biases = (0..parameter_count)
            .map(|index| round_bf16((index % 7) as f32 / 32.0 - 0.09375))
            .collect::<Vec<_>>();
        let up_scales = (0..parameter_count)
            .map(|index| round_bf16(-((index % 3) as f32 + 1.0) / 128.0))
            .collect::<Vec<_>>();
        let up_biases = (0..parameter_count)
            .map(|index| round_bf16((index % 11) as f32 / 64.0 - 0.0625))
            .collect::<Vec<_>>();
        let options = MTLResourceOptions::StorageModeShared;
        let input_buffer = buffer(&device, &bf16_bits(&input), options);
        let gate_packed_buffer = buffer(&device, &gate_packed, options);
        let up_packed_buffer = buffer(&device, &up_packed, options);
        let gate_scale_buffer = buffer(&device, &bf16_bits(&gate_scales), options);
        let gate_bias_buffer = buffer(&device, &bf16_bits(&gate_biases), options);
        let up_scale_buffer = buffer(&device, &bf16_bits(&up_scales), options);
        let up_bias_buffer = buffer(&device, &bf16_bits(&up_biases), options);
        let output_len = ROWS * MATRIX_OUTPUT_WIDTH;
        let linear_output = arena.empty_f16(output_len).unwrap();
        let gate_up_output = arena.empty_f16(output_len).unwrap();
        let wide_linear_output = arena.empty_f16(output_len).unwrap();
        let generic_linear_output = arena.empty_f16(output_len).unwrap();
        let generic_gate_up_output = arena.empty_f16(output_len).unwrap();
        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        encode_1d_threadgroups_args(
            command_buffer,
            &w4.prefill_matrix_linear,
            &[
                KernelArg::Buffer(&gate_packed_buffer),
                KernelArg::Buffer(&gate_scale_buffer),
                KernelArg::Buffer(&gate_bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&linear_output),
                KernelArg::U32(ROWS as u32),
                KernelArg::U32(MATRIX_INPUT_WIDTH as u32),
                KernelArg::U32(MATRIX_OUTPUT_WIDTH as u32),
            ],
            ROWS.div_ceil(PREFILL_MATRIX_ROWS)
                * MATRIX_OUTPUT_WIDTH.div_ceil(PREFILL_LINEAR_OUTPUTS),
            256,
        )
        .unwrap();
        encode_1d_threadgroups_args(
            command_buffer,
            &w4.prefill_matrix_gate_up,
            &[
                KernelArg::Buffer(&gate_packed_buffer),
                KernelArg::Buffer(&gate_scale_buffer),
                KernelArg::Buffer(&gate_bias_buffer),
                KernelArg::Buffer(&up_packed_buffer),
                KernelArg::Buffer(&up_scale_buffer),
                KernelArg::Buffer(&up_bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&gate_up_output),
                KernelArg::U32(ROWS as u32),
                KernelArg::U32(MATRIX_INPUT_WIDTH as u32),
                KernelArg::U32(MATRIX_OUTPUT_WIDTH as u32),
            ],
            ROWS.div_ceil(PREFILL_MATRIX_ROWS)
                * MATRIX_OUTPUT_WIDTH.div_ceil(PREFILL_GATE_UP_OUTPUTS),
            256,
        )
        .unwrap();
        encode_1d_threadgroups_args(
            command_buffer,
            &w4.prefill_wide_linear,
            &[
                KernelArg::Buffer(&gate_packed_buffer),
                KernelArg::Buffer(&gate_scale_buffer),
                KernelArg::Buffer(&gate_bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&wide_linear_output),
                KernelArg::U32(ROWS as u32),
                KernelArg::U32(MATRIX_INPUT_WIDTH as u32),
                KernelArg::U32(MATRIX_OUTPUT_WIDTH as u32),
            ],
            ROWS.div_ceil(PREFILL_WIDE_ROWS) * MATRIX_OUTPUT_WIDTH.div_ceil(PREFILL_LINEAR_OUTPUTS),
            PREFILL_WIDE_THREADS,
        )
        .unwrap();
        encode_1d_threadgroups_args(
            command_buffer,
            &w4.linear,
            &[
                KernelArg::Buffer(&gate_packed_buffer),
                KernelArg::Buffer(&gate_scale_buffer),
                KernelArg::Buffer(&gate_bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&generic_linear_output),
                KernelArg::U32(ROWS as u32),
                KernelArg::U32(MATRIX_INPUT_WIDTH as u32),
                KernelArg::U32(MATRIX_OUTPUT_WIDTH as u32),
                KernelArg::U32(0),
                KernelArg::U32(1),
            ],
            MATRIX_OUTPUT_WIDTH * ROWS.div_ceil(ROW_TILE),
            LINEAR_THREADS,
        )
        .unwrap();
        encode_1d_threadgroups_args(
            command_buffer,
            &w4.gate_up,
            &[
                KernelArg::Buffer(&gate_packed_buffer),
                KernelArg::Buffer(&gate_scale_buffer),
                KernelArg::Buffer(&gate_bias_buffer),
                KernelArg::Buffer(&up_packed_buffer),
                KernelArg::Buffer(&up_scale_buffer),
                KernelArg::Buffer(&up_bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&generic_gate_up_output),
                KernelArg::U32(ROWS as u32),
                KernelArg::U32(MATRIX_INPUT_WIDTH as u32),
                KernelArg::U32(MATRIX_OUTPUT_WIDTH as u32),
            ],
            MATRIX_OUTPUT_WIDTH * ROWS.div_ceil(ROW_TILE),
            LINEAR_THREADS,
        )
        .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);

        let linear_expected = affine_matrix_reference(
            &input,
            &gate_packed,
            &gate_scales,
            &gate_biases,
            MATRIX_INPUT_WIDTH,
            MATRIX_OUTPUT_WIDTH,
        );
        let linear_actual = read_bf16_buffer_as_f32(&linear_output, output_len).unwrap();
        assert_bf16_close(&linear_actual, &linear_expected);
        let generic_linear_actual =
            read_bf16_buffer_as_f32(&generic_linear_output, output_len).unwrap();
        assert_eq!(linear_actual, generic_linear_actual);
        let wide_linear_actual = read_bf16_buffer_as_f32(&wide_linear_output, output_len).unwrap();
        assert_eq!(wide_linear_actual, generic_linear_actual);

        let gate = linear_expected;
        let up = affine_matrix_reference(
            &input,
            &up_packed,
            &up_scales,
            &up_biases,
            MATRIX_INPUT_WIDTH,
            MATRIX_OUTPUT_WIDTH,
        );
        let expected = gate
            .into_iter()
            .zip(up)
            .map(|(gate, up)| {
                let gate = round_bf16(gate);
                let up = round_bf16(up);
                let activated = round_bf16(gate / (1.0 + (-gate).exp()));
                round_bf16(activated * up)
            })
            .collect::<Vec<_>>();
        let actual = read_bf16_buffer_as_f32(&gate_up_output, output_len).unwrap();
        assert_bf16_close(&actual, &expected);
        let generic_actual = read_bf16_buffer_as_f32(&generic_gate_up_output, output_len).unwrap();
        assert_eq!(actual, generic_actual);
    }

    #[test]
    fn packed_prefill_kernels_match_generic_path_bit_for_bit() {
        const MATRIX_INPUT_WIDTH: usize = 512;
        const OUTPUT_WIDTH: usize = 64;

        let Some(device) = Device::system_default() else {
            return;
        };
        let arena = MetalArena::new(&device).unwrap();
        let w4 = MetalQwenMlxW4::new(&device, arena.clone()).unwrap();
        for row_count in [65, 257] {
            let input = test_input_for(row_count, MATRIX_INPUT_WIDTH);
            let output_len = row_count * OUTPUT_WIDTH;
            let residual = (0..output_len)
                .map(|index| round_bf16((index % 17) as f32 / 32.0 - 0.25))
                .collect::<Vec<_>>();
            let gate_packed = packed_matrix(OUTPUT_WIDTH, MATRIX_INPUT_WIDTH, 5);
            let up_packed = packed_matrix(OUTPUT_WIDTH, MATRIX_INPUT_WIDTH, 11);
            let parameter_count = OUTPUT_WIDTH * (MATRIX_INPUT_WIDTH / GROUP_SIZE);
            let gate_scales = (0..parameter_count)
                .map(|index| round_bf16(((index % 5) as f32 + 1.0) / 64.0))
                .collect::<Vec<_>>();
            let gate_biases = (0..parameter_count)
                .map(|index| round_bf16((index % 7) as f32 / 32.0 - 0.09375))
                .collect::<Vec<_>>();
            let up_scales = (0..parameter_count)
                .map(|index| round_bf16(-((index % 3) as f32 + 1.0) / 128.0))
                .collect::<Vec<_>>();
            let up_biases = (0..parameter_count)
                .map(|index| round_bf16((index % 11) as f32 / 64.0 - 0.0625))
                .collect::<Vec<_>>();
            let options = MTLResourceOptions::StorageModeShared;
            let input_buffer = buffer(&device, &bf16_bits(&input), options);
            let gate_packed_buffer = buffer(&device, &gate_packed, options);
            let up_packed_buffer = buffer(&device, &up_packed, options);
            let gate_scale_buffer = buffer(&device, &bf16_bits(&gate_scales), options);
            let gate_bias_buffer = buffer(&device, &bf16_bits(&gate_biases), options);
            let up_scale_buffer = buffer(&device, &bf16_bits(&up_scales), options);
            let up_bias_buffer = buffer(&device, &bf16_bits(&up_biases), options);
            let residual_buffer = buffer(&device, &bf16_bits(&residual), options);
            let packed_linear_output = arena.empty_f16(output_len).unwrap();
            let packed_linear_add_output = arena.empty_f16(output_len).unwrap();
            let packed_gate_up_output = arena.empty_f16(output_len).unwrap();
            let generic_linear_output = arena.empty_f16(output_len).unwrap();
            let generic_gate_up_output = arena.empty_f16(output_len).unwrap();
            let queue = device.new_command_queue();
            let command_buffer = queue.new_command_buffer();
            w4.packed_prefill
                .encode_linear_raw(
                    command_buffer,
                    &gate_packed_buffer,
                    &gate_scale_buffer,
                    &gate_bias_buffer,
                    &input_buffer,
                    &packed_linear_output,
                    row_count,
                    MATRIX_INPUT_WIDTH,
                    OUTPUT_WIDTH,
                )
                .unwrap();
            w4.packed_prefill
                .encode_linear_add_raw(
                    command_buffer,
                    &gate_packed_buffer,
                    &gate_scale_buffer,
                    &gate_bias_buffer,
                    &input_buffer,
                    &residual_buffer,
                    &packed_linear_add_output,
                    row_count,
                    MATRIX_INPUT_WIDTH,
                    OUTPUT_WIDTH,
                )
                .unwrap();
            w4.packed_prefill
                .encode_gate_up_raw(
                    command_buffer,
                    &gate_packed_buffer,
                    &gate_scale_buffer,
                    &gate_bias_buffer,
                    &up_packed_buffer,
                    &up_scale_buffer,
                    &up_bias_buffer,
                    &input_buffer,
                    &packed_gate_up_output,
                    row_count,
                    MATRIX_INPUT_WIDTH,
                    OUTPUT_WIDTH,
                )
                .unwrap();
            encode_1d_threadgroups_args(
                command_buffer,
                &w4.linear,
                &[
                    KernelArg::Buffer(&gate_packed_buffer),
                    KernelArg::Buffer(&gate_scale_buffer),
                    KernelArg::Buffer(&gate_bias_buffer),
                    KernelArg::Buffer(&input_buffer),
                    KernelArg::Buffer(&generic_linear_output),
                    KernelArg::U32(row_count as u32),
                    KernelArg::U32(MATRIX_INPUT_WIDTH as u32),
                    KernelArg::U32(OUTPUT_WIDTH as u32),
                    KernelArg::U32(0),
                    KernelArg::U32(1),
                ],
                OUTPUT_WIDTH * row_count.div_ceil(ROW_TILE),
                LINEAR_THREADS,
            )
            .unwrap();
            encode_1d_threadgroups_args(
                command_buffer,
                &w4.gate_up,
                &[
                    KernelArg::Buffer(&gate_packed_buffer),
                    KernelArg::Buffer(&gate_scale_buffer),
                    KernelArg::Buffer(&gate_bias_buffer),
                    KernelArg::Buffer(&up_packed_buffer),
                    KernelArg::Buffer(&up_scale_buffer),
                    KernelArg::Buffer(&up_bias_buffer),
                    KernelArg::Buffer(&input_buffer),
                    KernelArg::Buffer(&generic_gate_up_output),
                    KernelArg::U32(row_count as u32),
                    KernelArg::U32(MATRIX_INPUT_WIDTH as u32),
                    KernelArg::U32(OUTPUT_WIDTH as u32),
                ],
                OUTPUT_WIDTH * row_count.div_ceil(ROW_TILE),
                LINEAR_THREADS,
            )
            .unwrap();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);

            let packed_linear = read_bf16_buffer_as_f32(&packed_linear_output, output_len).unwrap();
            let generic_linear =
                read_bf16_buffer_as_f32(&generic_linear_output, output_len).unwrap();
            assert_eq!(packed_linear, generic_linear);
            let packed_linear_add =
                read_bf16_buffer_as_f32(&packed_linear_add_output, output_len).unwrap();
            let expected_linear_add = generic_linear
                .iter()
                .zip(&residual)
                .map(|(projected, residual)| round_bf16(projected + residual))
                .collect::<Vec<_>>();
            assert_eq!(packed_linear_add, expected_linear_add);

            let packed_gate_up =
                read_bf16_buffer_as_f32(&packed_gate_up_output, output_len).unwrap();
            let generic_gate_up =
                read_bf16_buffer_as_f32(&generic_gate_up_output, output_len).unwrap();
            assert_eq!(packed_gate_up, generic_gate_up);
        }
    }

    fn assert_fused_top_k_matches_materialized_logits(row_count: usize) {
        let Some(device) = Device::system_default() else {
            return;
        };
        let arena = MetalArena::new(&device).unwrap();
        let w4 = MetalQwenMlxW4::new(&device, arena.clone()).unwrap();
        let input = test_input_for(row_count, TEST_INPUT_WIDTH);
        let packed = packed_matrix(TOP_K_TEST_VOCAB, TEST_INPUT_WIDTH, 17);
        let scales = (0..TOP_K_TEST_VOCAB)
            .map(|index| round_bf16(((index % 11) as f32 + 1.0) / 128.0))
            .collect::<Vec<_>>();
        let biases = (0..TOP_K_TEST_VOCAB)
            .map(|index| round_bf16((index % 13) as f32 / 64.0 - 0.09375))
            .collect::<Vec<_>>();
        let options = MTLResourceOptions::StorageModeShared;
        let input_buffer = buffer(&device, &bf16_bits(&input), options);
        let packed_buffer = buffer(&device, &packed, options);
        let scale_buffer = buffer(&device, &bf16_bits(&scales), options);
        let bias_buffer = buffer(&device, &bf16_bits(&biases), options);
        let logits = arena.empty_f16(row_count * TOP_K_TEST_VOCAB).unwrap();
        let tile_count = TOP_K_TEST_VOCAB.div_ceil(TOP_K_VOCAB_TILE);
        let local_count = row_count * tile_count * TOP_K;
        let local_ids = arena.empty_u32(local_count).unwrap();
        let local_values = arena.empty_f32(local_count).unwrap();
        let candidate_ids = arena.empty_u32(row_count * TOP_K).unwrap();
        let candidate_values = arena.empty_f32(row_count * TOP_K).unwrap();
        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        encode_1d_threadgroups_args(
            command_buffer,
            &w4.linear,
            &[
                KernelArg::Buffer(&packed_buffer),
                KernelArg::Buffer(&scale_buffer),
                KernelArg::Buffer(&bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&logits),
                KernelArg::U32(row_count as u32),
                KernelArg::U32(TEST_INPUT_WIDTH as u32),
                KernelArg::U32(TOP_K_TEST_VOCAB as u32),
                KernelArg::U32(0),
                KernelArg::U32(1),
            ],
            TOP_K_TEST_VOCAB * row_count.div_ceil(ROW_TILE),
            LINEAR_THREADS,
        )
        .unwrap();
        encode_1d_threadgroups_args(
            command_buffer,
            &w4.dflash_top_k_local,
            &[
                KernelArg::Buffer(&packed_buffer),
                KernelArg::Buffer(&scale_buffer),
                KernelArg::Buffer(&bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&local_ids),
                KernelArg::Buffer(&local_values),
                KernelArg::U32(row_count as u32),
                KernelArg::U32(TEST_INPUT_WIDTH as u32),
                KernelArg::U32(TOP_K_TEST_VOCAB as u32),
                KernelArg::U32(tile_count as u32),
            ],
            tile_count * row_count.div_ceil(ROW_TILE),
            LINEAR_THREADS,
        )
        .unwrap();
        encode_1d_threadgroups_args(
            command_buffer,
            &w4.top_k_merge,
            &[
                KernelArg::Buffer(&local_ids),
                KernelArg::Buffer(&local_values),
                KernelArg::Buffer(&candidate_ids),
                KernelArg::Buffer(&candidate_values),
                KernelArg::U32(row_count as u32),
                KernelArg::U32(tile_count as u32),
            ],
            row_count,
            LINEAR_THREADS,
        )
        .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);

        let logits = read_bf16_buffer_as_f32(&logits, row_count * TOP_K_TEST_VOCAB).unwrap();
        let actual_ids = read_u32_buffer(&candidate_ids, row_count * TOP_K).unwrap();
        let actual_values = read_f32_buffer(&candidate_values, row_count * TOP_K).unwrap();
        for row in 0..row_count {
            let mut expected = logits[row * TOP_K_TEST_VOCAB..(row + 1) * TOP_K_TEST_VOCAB]
                .iter()
                .copied()
                .enumerate()
                .collect::<Vec<_>>();
            expected.sort_unstable_by(|(left_id, left), (right_id, right)| {
                right.total_cmp(left).then_with(|| left_id.cmp(right_id))
            });
            for rank in 0..TOP_K {
                let actual = row * TOP_K + rank;
                assert_eq!(actual_ids[actual] as usize, expected[rank].0);
                assert_eq!(actual_values[actual], expected[rank].1);
            }
        }
    }

    fn test_input(rows: usize) -> Vec<f32> {
        (0..rows * TEST_INPUT_WIDTH)
            .map(|index| round_bf16(((index * 13 % 97) as f32 - 48.0) / 32.0))
            .collect()
    }

    fn assert_exact_verify_linear_add(row_count: usize) {
        const OUTPUT_WIDTH: usize = 9;

        let Some(device) = Device::system_default() else {
            return;
        };
        let arena = MetalArena::new(&device).unwrap();
        let w4 = MetalQwenMlxW4::new(&device, arena.clone()).unwrap();
        let input = test_input(row_count);
        let residual = (0..row_count * OUTPUT_WIDTH)
            .map(|index| round_bf16(((index * 17 % 31) as f32 - 15.0) / 16.0))
            .collect::<Vec<_>>();
        let packed = packed_rows_with_width(3, OUTPUT_WIDTH);
        let scales = (0..OUTPUT_WIDTH)
            .map(|row| 0.03125 * (row + 1) as f32)
            .collect::<Vec<_>>();
        let biases = (0..OUTPUT_WIDTH)
            .map(|row| -0.5 + 0.125 * row as f32)
            .collect::<Vec<_>>();
        let options = MTLResourceOptions::StorageModeShared;
        let input_buffer = buffer(&device, &bf16_bits(&input), options);
        let residual_buffer = buffer(&device, &bf16_bits(&residual), options);
        let packed_buffer = buffer(&device, &packed, options);
        let scale_buffer = buffer(&device, &bf16_bits(&scales), options);
        let bias_buffer = buffer(&device, &bf16_bits(&biases), options);
        let output_len = row_count * OUTPUT_WIDTH;
        let projected = arena.empty_f16(output_len).unwrap();
        let generic = arena.empty_f16(output_len).unwrap();
        let fused = arena.empty_f16(output_len).unwrap();
        let (linear_pipeline, fused_pipeline, output_tile) = if row_count == VERIFY_ROWS {
            (&w4.verify5_exact_linear, &w4.verify5_exact_linear_add, 4)
        } else {
            (&w4.verify8_exact_linear, &w4.verify8_exact_linear_add, 4)
        };
        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        encode_1d_threadgroups_args(
            command_buffer,
            linear_pipeline,
            &[
                KernelArg::Buffer(&packed_buffer),
                KernelArg::Buffer(&scale_buffer),
                KernelArg::Buffer(&bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&projected),
                KernelArg::U32(TEST_INPUT_WIDTH as u32),
                KernelArg::U32(OUTPUT_WIDTH as u32),
            ],
            OUTPUT_WIDTH.div_ceil(output_tile),
            LINEAR_THREADS,
        )
        .unwrap();
        encode_1d_threadgroups_args(
            command_buffer,
            &w4.linear,
            &[
                KernelArg::Buffer(&packed_buffer),
                KernelArg::Buffer(&scale_buffer),
                KernelArg::Buffer(&bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&generic),
                KernelArg::U32(row_count as u32),
                KernelArg::U32(TEST_INPUT_WIDTH as u32),
                KernelArg::U32(OUTPUT_WIDTH as u32),
                KernelArg::U32(0),
                KernelArg::U32(1),
            ],
            OUTPUT_WIDTH * row_count.div_ceil(ROW_TILE),
            LINEAR_THREADS,
        )
        .unwrap();
        encode_1d_threadgroups_args(
            command_buffer,
            fused_pipeline,
            &[
                KernelArg::Buffer(&packed_buffer),
                KernelArg::Buffer(&scale_buffer),
                KernelArg::Buffer(&bias_buffer),
                KernelArg::Buffer(&input_buffer),
                KernelArg::Buffer(&residual_buffer),
                KernelArg::Buffer(&fused),
                KernelArg::U32(TEST_INPUT_WIDTH as u32),
                KernelArg::U32(OUTPUT_WIDTH as u32),
            ],
            OUTPUT_WIDTH.div_ceil(output_tile),
            LINEAR_THREADS,
        )
        .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);

        let projected = read_bf16_buffer_as_f32(&projected, output_len).unwrap();
        let generic = read_bf16_buffer_as_f32(&generic, output_len).unwrap();
        let fused = read_bf16_buffer_as_f32(&fused, output_len).unwrap();
        assert_eq!(projected, generic);
        let expected = projected
            .into_iter()
            .zip(residual)
            .map(|(projected, residual)| round_bf16(projected + residual))
            .collect::<Vec<_>>();
        assert_eq!(fused, expected);
    }

    fn packed_rows(seed: usize) -> Vec<u32> {
        packed_rows_with_width(seed, TEST_OUTPUT_WIDTH)
    }

    fn packed_rows_with_width(seed: usize, output_width: usize) -> Vec<u32> {
        (0..output_width)
            .flat_map(|row| {
                (0..TEST_INPUT_WIDTH / VALUES_PER_WORD).map(move |word| {
                    (0..VALUES_PER_WORD).fold(0_u32, |packed, nibble| {
                        let column = word * VALUES_PER_WORD + nibble;
                        let value = ((row * 7 + column * seed + 3) % 16) as u32;
                        packed | (value << (nibble * 4))
                    })
                })
            })
            .collect()
    }

    fn test_input_for(rows: usize, width: usize) -> Vec<f32> {
        (0..rows * width)
            .map(|index| round_bf16(((index * 13 % 97) as f32 - 48.0) / 128.0))
            .collect()
    }

    fn packed_matrix(output_width: usize, input_width: usize, seed: usize) -> Vec<u32> {
        (0..output_width)
            .flat_map(|row| {
                (0..input_width / VALUES_PER_WORD).map(move |word| {
                    (0..VALUES_PER_WORD).fold(0_u32, |packed, nibble| {
                        let column = word * VALUES_PER_WORD + nibble;
                        let value = ((row * 7 + column * seed + 3) % 16) as u32;
                        packed | (value << (nibble * 4))
                    })
                })
            })
            .collect()
    }

    fn affine_matrix_reference(
        input: &[f32],
        packed: &[u32],
        scales: &[f32],
        biases: &[f32],
        input_width: usize,
        output_width: usize,
    ) -> Vec<f32> {
        let groups_per_row = input_width / GROUP_SIZE;
        (0..input.len() / input_width)
            .flat_map(|input_row| {
                (0..output_width).map(move |output_row| {
                    (0..input_width).fold(0.0_f32, |sum, column| {
                        let word = packed[output_row * (input_width / VALUES_PER_WORD)
                            + column / VALUES_PER_WORD];
                        let quantized = (word >> ((column % VALUES_PER_WORD) * 4)) & 0x0f;
                        let parameter = output_row * groups_per_row + column / GROUP_SIZE;
                        let weight =
                            round_bf16(quantized as f32 * scales[parameter] + biases[parameter]);
                        sum + input[input_row * input_width + column] * weight
                    })
                })
            })
            .collect()
    }

    fn affine_reference(input: &[f32], packed: &[u32], scales: &[f32], biases: &[f32]) -> Vec<f32> {
        (0..input.len() / TEST_INPUT_WIDTH)
            .flat_map(|input_row| {
                (0..TEST_OUTPUT_WIDTH).map(move |output_row| {
                    (0..TEST_INPUT_WIDTH).fold(0.0_f32, |sum, column| {
                        let word = packed[output_row * (TEST_INPUT_WIDTH / VALUES_PER_WORD)
                            + column / VALUES_PER_WORD];
                        let quantized = (word >> ((column % VALUES_PER_WORD) * 4)) & 0x0f;
                        let weight = quantized as f32 * scales[output_row] + biases[output_row];
                        sum + input[input_row * TEST_INPUT_WIDTH + column] * weight
                    })
                })
            })
            .collect()
    }

    fn bf16_bits(values: &[f32]) -> Vec<u16> {
        values
            .iter()
            .map(|value| (round_bf16(*value).to_bits() >> 16) as u16)
            .collect()
    }

    fn round_bf16(value: f32) -> f32 {
        let bits = value.to_bits();
        let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
        f32::from_bits(rounded & 0xffff_0000)
    }

    fn buffer<T>(device: &Device, values: &[T], options: MTLResourceOptions) -> ::metal::Buffer {
        device.new_buffer_with_data(
            values.as_ptr().cast(),
            std::mem::size_of_val(values) as u64,
            options,
        )
    }

    fn assert_bf16_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let expected = round_bf16(*expected);
            let tolerance = (expected.abs() * 0.01).max(0.03125);
            assert!(
                (actual - expected).abs() <= tolerance,
                "W4 output {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }
}
