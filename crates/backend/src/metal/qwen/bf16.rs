use ::metal::{Buffer, CommandBufferRef, ComputePipelineState, Device};
use common::{Error, Result};

use crate::DeviceQwenBf16Tensor;

use super::super::{
    arena::MetalArena,
    buffers::{require_byte_capacity, require_f16_capacity},
    command::{encode_1d_threadgroups_args, KernelArg},
    library::MetalLibrary,
    pipeline::compute_pipeline,
};

const KERNEL_SOURCE: &str = include_str!("kernels/bf16.metal");
const EMBEDDING_KERNEL: &str = "qwen_bf16_embedding_kernel";
const RMS_NORM_KERNEL: &str = "qwen_bf16_rms_norm_kernel";
const RMS_NORM_STANDARD_KERNEL: &str = "qwen_bf16_rms_norm_standard_kernel";
const LINEAR_KERNEL: &str = "qwen_bf16_linear_kernel";
const ADD_KERNEL: &str = "qwen_bf16_add_kernel";
const CONCAT_LAST_KERNEL: &str = "qwen_bf16_concat_last_kernel";
const COPY_ROW_KERNEL: &str = "qwen_bf16_copy_row_kernel";
const LAST_TOKEN_LOGITS_KERNEL: &str = "qwen_bf16_last_token_logits_kernel";
const ALL_TOKEN_LOGITS_KERNEL: &str = "qwen_bf16_all_token_logits_kernel";
const VERIFY_ROWS_LOGITS_KERNEL: &str = "qwen_bf16_verify_rows_logits_kernel";
const VERIFY3_LOGITS_KERNEL: &str = "qwen_bf16_verify3_logits_kernel";
const ARGMAX_KERNEL: &str = "qwen_bf16_argmax_kernel";
const COMPACT_DRAFT_LOGITS_KERNEL: &str = "qwen_bf16_compact_draft_logits_kernel";
const COMPACT_DRAFT_ARGMAX_KERNEL: &str = "qwen_bf16_compact_draft_argmax_kernel";
const SIMD_LANES: usize = 32;
const MAX_VERIFY_ROWS: usize = 9;
const COMPACT_DRAFT_PREFIX: usize = 98_304;
const COMPACT_DRAFT_CONTROL_START: usize = 248_044;
const COMPACT_DRAFT_CONTROL_END: usize = 248_070;
const COMPACT_DRAFT_VOCAB: usize =
    COMPACT_DRAFT_PREFIX + COMPACT_DRAFT_CONTROL_END - COMPACT_DRAFT_CONTROL_START;

pub(super) struct MetalQwenBf16 {
    embedding: ComputePipelineState,
    rms_norm: ComputePipelineState,
    rms_norm_standard: ComputePipelineState,
    linear: ComputePipelineState,
    add: ComputePipelineState,
    concat_last: ComputePipelineState,
    copy_row: ComputePipelineState,
    last_token_logits: ComputePipelineState,
    all_token_logits: ComputePipelineState,
    verify_rows_logits: ComputePipelineState,
    verify3_logits: ComputePipelineState,
    argmax: ComputePipelineState,
    compact_draft_logits: ComputePipelineState,
    compact_draft_argmax: ComputePipelineState,
    arena: MetalArena,
}

impl MetalQwenBf16 {
    pub(super) fn new(device: &Device, arena: MetalArena) -> Result<Self> {
        let library = MetalLibrary::compile_source(device, KERNEL_SOURCE)?;
        let embedding = compute_pipeline(device, &library, EMBEDDING_KERNEL)?;
        let rms_norm = compute_pipeline(device, &library, RMS_NORM_KERNEL)?;
        let rms_norm_standard = compute_pipeline(device, &library, RMS_NORM_STANDARD_KERNEL)?;
        let linear = compute_pipeline(device, &library, LINEAR_KERNEL)?;
        let add = compute_pipeline(device, &library, ADD_KERNEL)?;
        let concat_last = compute_pipeline(device, &library, CONCAT_LAST_KERNEL)?;
        let copy_row = compute_pipeline(device, &library, COPY_ROW_KERNEL)?;
        let last_token_logits = compute_pipeline(device, &library, LAST_TOKEN_LOGITS_KERNEL)?;
        let all_token_logits = compute_pipeline(device, &library, ALL_TOKEN_LOGITS_KERNEL)?;
        let verify_rows_logits = compute_pipeline(device, &library, VERIFY_ROWS_LOGITS_KERNEL)?;
        let verify3_logits = compute_pipeline(device, &library, VERIFY3_LOGITS_KERNEL)?;
        let argmax = compute_pipeline(device, &library, ARGMAX_KERNEL)?;
        let compact_draft_logits = compute_pipeline(device, &library, COMPACT_DRAFT_LOGITS_KERNEL)?;
        let compact_draft_argmax = compute_pipeline(device, &library, COMPACT_DRAFT_ARGMAX_KERNEL)?;
        require_simd_width(&rms_norm, RMS_NORM_KERNEL)?;
        require_simd_width(&rms_norm_standard, RMS_NORM_STANDARD_KERNEL)?;
        require_simd_width(&linear, LINEAR_KERNEL)?;
        require_simd_width(&last_token_logits, LAST_TOKEN_LOGITS_KERNEL)?;
        require_simd_width(&all_token_logits, ALL_TOKEN_LOGITS_KERNEL)?;
        require_simd_width(&verify_rows_logits, VERIFY_ROWS_LOGITS_KERNEL)?;
        require_simd_width(&verify3_logits, VERIFY3_LOGITS_KERNEL)?;
        require_simd_width(&compact_draft_logits, COMPACT_DRAFT_LOGITS_KERNEL)?;
        Ok(Self {
            embedding,
            rms_norm,
            rms_norm_standard,
            linear,
            add,
            concat_last,
            copy_row,
            last_token_logits,
            all_token_logits,
            verify_rows_logits,
            verify3_logits,
            argmax,
            compact_draft_logits,
            compact_draft_argmax,
            arena,
        })
    }

    pub(super) fn encode_embedding(
        &self,
        command_buffer: &CommandBufferRef,
        embedding: &DeviceQwenBf16Tensor,
        token_ids: &[u32],
    ) -> Result<Buffer> {
        let [vocab_size, _hidden_size]: [usize; 2] = embedding
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| Error::backend("Qwen embedding must be rank 2"))?;
        if token_ids.is_empty() {
            return Err(Error::backend("Qwen embedding requires token IDs"));
        }
        if let Some(token_id) = token_ids
            .iter()
            .copied()
            .find(|token_id| *token_id as usize >= vocab_size)
        {
            return Err(Error::backend(format!(
                "Qwen token ID {token_id} exceeds vocabulary {vocab_size}"
            )));
        }
        let token_id_buffer = self.arena.empty_u32(token_ids.len())?;
        super::super::buffers::write_u32_buffer(&token_id_buffer, token_ids)?;
        self.encode_embedding_from_device(
            command_buffer,
            embedding,
            &token_id_buffer,
            token_ids.len(),
        )
    }

    pub(super) fn encode_embedding_from_device(
        &self,
        command_buffer: &CommandBufferRef,
        embedding: &DeviceQwenBf16Tensor,
        token_ids: &Buffer,
        token_count: usize,
    ) -> Result<Buffer> {
        let [_vocab_size, hidden_size]: [usize; 2] = embedding
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| Error::backend("Qwen embedding must be rank 2"))?;
        if token_count == 0 {
            return Err(Error::backend("Qwen embedding requires token IDs"));
        }
        require_tensor_range(embedding, "Qwen embedding")?;
        require_byte_capacity(
            token_ids,
            token_count * std::mem::size_of::<u32>(),
            "Qwen device token IDs",
        )?;
        let output_len = token_count
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("Qwen embedding output length overflow"))?;
        let output = self.arena.empty_f16(output_len)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.embedding,
            &[
                KernelArg::BufferOffset(&embedding.buffer, embedding.byte_offset),
                KernelArg::Buffer(token_ids),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(hidden_size, "embedding hidden size")?),
                KernelArg::U32(as_u32(output_len, "embedding output length")?),
            ],
            output_len.div_ceil(256),
            256,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_rms_norm(
        &self,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        input_len: usize,
        weight: &DeviceQwenBf16Tensor,
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<Buffer> {
        if weight.shape != [hidden_size] {
            return Err(Error::backend(format!(
                "Qwen RMSNorm weight must be [{hidden_size}], got {:?}",
                weight.shape
            )));
        }
        let expected = rows
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("Qwen RMSNorm input length overflow"))?;
        if rows == 0 || hidden_size == 0 || input_len != expected || !eps.is_finite() || eps <= 0.0
        {
            return Err(Error::backend(format!(
                "invalid Qwen RMSNorm rows={rows}, hidden={hidden_size}, input={input_len}, eps={eps}"
            )));
        }
        require_f16_capacity(input, input_len, "Qwen RMSNorm input")?;
        require_tensor_range(weight, "Qwen RMSNorm weight")?;
        let output = self.arena.empty_f16(input_len)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.rms_norm,
            &[
                KernelArg::Buffer(input),
                KernelArg::BufferOffset(&weight.buffer, weight.byte_offset),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(rows, "RMSNorm rows")?),
                KernelArg::U32(as_u32(hidden_size, "RMSNorm hidden size")?),
                KernelArg::F32(eps),
            ],
            rows,
            SIMD_LANES,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_rms_norm_standard(
        &self,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        input_len: usize,
        weight: &DeviceQwenBf16Tensor,
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<Buffer> {
        if weight.shape != [hidden_size] {
            return Err(Error::backend(format!(
                "Qwen standard RMSNorm weight must be [{hidden_size}], got {:?}",
                weight.shape
            )));
        }
        let expected = rows
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("Qwen standard RMSNorm input length overflow"))?;
        if rows == 0 || hidden_size == 0 || input_len != expected || !eps.is_finite() || eps <= 0.0
        {
            return Err(Error::backend(format!(
                "invalid Qwen standard RMSNorm rows={rows}, hidden={hidden_size}, input={input_len}, eps={eps}"
            )));
        }
        require_f16_capacity(input, input_len, "Qwen standard RMSNorm input")?;
        require_tensor_range(weight, "Qwen standard RMSNorm weight")?;
        let output = self.arena.empty_f16(input_len)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.rms_norm_standard,
            &[
                KernelArg::Buffer(input),
                KernelArg::BufferOffset(&weight.buffer, weight.byte_offset),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(rows, "standard RMSNorm rows")?),
                KernelArg::U32(as_u32(hidden_size, "standard RMSNorm hidden size")?),
                KernelArg::F32(eps),
            ],
            rows,
            SIMD_LANES,
        )?;
        Ok(output)
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_rms_norm_test_raw(
        &self,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        input_len: usize,
        weight: &Buffer,
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<Buffer> {
        let expected = rows
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("Qwen test RMSNorm input length overflow"))?;
        if rows == 0 || hidden_size == 0 || input_len != expected || !eps.is_finite() || eps <= 0.0
        {
            return Err(Error::backend("invalid Qwen test RMSNorm dimensions"));
        }
        require_f16_capacity(input, input_len, "Qwen test RMSNorm input")?;
        require_f16_capacity(weight, hidden_size, "Qwen test RMSNorm weight")?;
        let output = self.arena.empty_f16(input_len)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.rms_norm,
            &[
                KernelArg::Buffer(input),
                KernelArg::Buffer(weight),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(rows, "test RMSNorm rows")?),
                KernelArg::U32(as_u32(hidden_size, "test RMSNorm hidden size")?),
                KernelArg::F32(eps),
            ],
            rows,
            SIMD_LANES,
        )?;
        Ok(output)
    }

    pub(super) fn encode_linear(
        &self,
        command_buffer: &CommandBufferRef,
        matrix: &DeviceQwenBf16Tensor,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Buffer> {
        let [out_features, in_features]: [usize; 2] = matrix
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| Error::backend("Qwen BF16 linear matrix must be rank 2"))?;
        let expected = row_count
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("Qwen BF16 linear input length overflow"))?;
        if row_count == 0 || input_len != expected {
            return Err(Error::backend(format!(
                "Qwen BF16 linear [{row_count},{in_features}] requires {expected} input values, got {input_len}"
            )));
        }
        require_f16_capacity(input, input_len, "Qwen BF16 linear input")?;
        require_tensor_range(matrix, "Qwen BF16 linear weight")?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Qwen BF16 linear output length overflow"))?;
        let output = self.arena.empty_f16(output_len)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.linear,
            &[
                KernelArg::BufferOffset(&matrix.buffer, matrix.byte_offset),
                KernelArg::Buffer(input),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(row_count, "linear rows")?),
                KernelArg::U32(as_u32(in_features, "linear input width")?),
                KernelArg::U32(as_u32(out_features, "linear output width")?),
            ],
            output_len,
            SIMD_LANES,
        )?;
        Ok(output)
    }

    pub(super) fn encode_add(
        &self,
        command_buffer: &CommandBufferRef,
        left: &Buffer,
        right: &Buffer,
        element_count: usize,
    ) -> Result<Buffer> {
        if element_count == 0 {
            return Err(Error::backend("Qwen BF16 add requires input values"));
        }
        require_f16_capacity(left, element_count, "Qwen BF16 add left input")?;
        require_f16_capacity(right, element_count, "Qwen BF16 add right input")?;
        let output = self.arena.empty_f16(element_count)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.add,
            &[
                KernelArg::Buffer(left),
                KernelArg::Buffer(right),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(element_count, "add element count")?),
            ],
            element_count.div_ceil(256),
            256,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_concat_last(
        &self,
        command_buffer: &CommandBufferRef,
        left: &Buffer,
        right: &Buffer,
        row_count: usize,
        left_width: usize,
        right_width: usize,
    ) -> Result<Buffer> {
        if row_count == 0 || left_width == 0 || right_width == 0 {
            return Err(Error::backend(
                "Qwen BF16 concat dimensions must be positive",
            ));
        }
        let left_len = row_count
            .checked_mul(left_width)
            .ok_or_else(|| Error::backend("Qwen concat left length overflow"))?;
        let right_len = row_count
            .checked_mul(right_width)
            .ok_or_else(|| Error::backend("Qwen concat right length overflow"))?;
        let output_len = left_len
            .checked_add(right_len)
            .ok_or_else(|| Error::backend("Qwen concat output length overflow"))?;
        require_f16_capacity(left, left_len, "Qwen concat left input")?;
        require_f16_capacity(right, right_len, "Qwen concat right input")?;
        let output = self.arena.empty_f16(output_len)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.concat_last,
            &[
                KernelArg::Buffer(left),
                KernelArg::Buffer(right),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(row_count, "concat rows")?),
                KernelArg::U32(as_u32(left_width, "concat left width")?),
                KernelArg::U32(as_u32(right_width, "concat right width")?),
            ],
            output_len.div_ceil(256),
            256,
        )?;
        Ok(output)
    }

    pub(super) fn encode_copy_row(
        &self,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        row_count: usize,
        row_width: usize,
        row_index: usize,
    ) -> Result<Buffer> {
        if row_count == 0 || row_width == 0 || row_index >= row_count {
            return Err(Error::backend(format!(
                "invalid Qwen copy row {row_index} from [{row_count},{row_width}]"
            )));
        }
        require_f16_capacity(input, row_count * row_width, "Qwen row input")?;
        let output = self.arena.empty_f16(row_width)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.copy_row,
            &[
                KernelArg::Buffer(input),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(row_index, "row index")?),
                KernelArg::U32(as_u32(row_width, "row width")?),
            ],
            row_width.div_ceil(256),
            256,
        )?;
        Ok(output)
    }

    pub(super) fn encode_last_token_argmax(
        &self,
        command_buffer: &CommandBufferRef,
        output_weight: &DeviceQwenBf16Tensor,
        hidden_states: &Buffer,
        batch_size: usize,
        sequence_length: usize,
        hidden_size: usize,
    ) -> Result<Buffer> {
        let [vocab_size, weight_hidden]: [usize; 2] = output_weight
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| Error::backend("Qwen output weight must be rank 2"))?;
        if weight_hidden != hidden_size {
            return Err(Error::backend(format!(
                "Qwen output weight hidden size must be {hidden_size}, got {weight_hidden}"
            )));
        }
        require_tensor_range(output_weight, "Qwen output weight")?;
        self.encode_last_token_argmax_raw(
            command_buffer,
            &output_weight.buffer,
            output_weight.byte_offset,
            vocab_size,
            hidden_states,
            batch_size,
            sequence_length,
            hidden_size,
        )
    }

    pub(super) fn encode_all_token_argmax(
        &self,
        command_buffer: &CommandBufferRef,
        output_weight: &DeviceQwenBf16Tensor,
        hidden_states: &Buffer,
        row_count: usize,
        hidden_size: usize,
    ) -> Result<Buffer> {
        let [vocab_size, weight_hidden]: [usize; 2] = output_weight
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| Error::backend("Qwen output weight must be rank 2"))?;
        if row_count == 0 || weight_hidden != hidden_size {
            return Err(Error::backend(format!(
                "Qwen all-row output requires positive rows and hidden {hidden_size}, got rows={row_count}, weight hidden={weight_hidden}"
            )));
        }
        require_tensor_range(output_weight, "Qwen output weight")?;
        require_f16_capacity(
            hidden_states,
            row_count * hidden_size,
            "Qwen all-row hidden states",
        )?;
        let logits = self.arena.empty_f16(row_count * vocab_size)?;
        let token_ids = self.arena.empty_u32(row_count)?;
        let (logits_pipeline, logits_threadgroups) = match row_count {
            3 => (&self.verify3_logits, vocab_size),
            2..=MAX_VERIFY_ROWS => (&self.verify_rows_logits, vocab_size),
            _ => (&self.all_token_logits, row_count * vocab_size),
        };
        encode_1d_threadgroups_args(
            command_buffer,
            logits_pipeline,
            &[
                KernelArg::BufferOffset(&output_weight.buffer, output_weight.byte_offset),
                KernelArg::Buffer(hidden_states),
                KernelArg::Buffer(&logits),
                KernelArg::U32(as_u32(row_count, "output rows")?),
                KernelArg::U32(as_u32(hidden_size, "output hidden size")?),
                KernelArg::U32(as_u32(vocab_size, "vocabulary size")?),
            ],
            logits_threadgroups,
            SIMD_LANES,
        )?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.argmax,
            &[
                KernelArg::Buffer(&logits),
                KernelArg::Buffer(&token_ids),
                KernelArg::U32(as_u32(row_count, "argmax rows")?),
                KernelArg::U32(as_u32(vocab_size, "argmax vocabulary size")?),
            ],
            row_count,
            256,
        )?;
        Ok(token_ids)
    }

    pub(super) fn encode_compact_draft_argmax(
        &self,
        command_buffer: &CommandBufferRef,
        output_weight: &DeviceQwenBf16Tensor,
        hidden_states: &Buffer,
        batch_size: usize,
        sequence_length: usize,
        hidden_size: usize,
    ) -> Result<Buffer> {
        let [vocab_size, weight_hidden]: [usize; 2] = output_weight
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| Error::backend("Qwen output weight must be rank 2"))?;
        if batch_size == 0
            || sequence_length == 0
            || hidden_size == 0
            || vocab_size < COMPACT_DRAFT_CONTROL_END
            || weight_hidden != hidden_size
        {
            return Err(Error::backend(format!(
                "invalid Qwen compact draft output shape weight={:?}, hidden=[{batch_size},{sequence_length},{hidden_size}]",
                output_weight.shape
            )));
        }
        require_tensor_range(output_weight, "Qwen compact draft output weight")?;
        require_f16_capacity(
            hidden_states,
            batch_size * sequence_length * hidden_size,
            "Qwen compact draft hidden states",
        )?;
        let logits = self.arena.empty_f16(batch_size * COMPACT_DRAFT_VOCAB)?;
        let token_ids = self.arena.empty_u32(batch_size)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.compact_draft_logits,
            &[
                KernelArg::BufferOffset(&output_weight.buffer, output_weight.byte_offset),
                KernelArg::Buffer(hidden_states),
                KernelArg::Buffer(&logits),
                KernelArg::U32(as_u32(batch_size, "compact draft batch")?),
                KernelArg::U32(as_u32(sequence_length, "compact draft sequence")?),
                KernelArg::U32(as_u32(hidden_size, "compact draft hidden size")?),
            ],
            batch_size * COMPACT_DRAFT_VOCAB,
            SIMD_LANES,
        )?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.compact_draft_argmax,
            &[
                KernelArg::Buffer(&logits),
                KernelArg::Buffer(&token_ids),
                KernelArg::U32(as_u32(batch_size, "compact draft argmax batch")?),
            ],
            batch_size,
            256,
        )?;
        Ok(token_ids)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_last_token_argmax_raw(
        &self,
        command_buffer: &CommandBufferRef,
        output_weight: &Buffer,
        weight_byte_offset: usize,
        vocab_size: usize,
        hidden_states: &Buffer,
        batch_size: usize,
        sequence_length: usize,
        hidden_size: usize,
    ) -> Result<Buffer> {
        if batch_size == 0 || sequence_length == 0 || hidden_size == 0 || vocab_size == 0 {
            return Err(Error::backend(
                "Qwen output argmax dimensions must be positive",
            ));
        }
        let hidden_values = batch_size
            .checked_mul(sequence_length)
            .and_then(|values| values.checked_mul(hidden_size))
            .ok_or_else(|| Error::backend("Qwen output hidden-state size overflow"))?;
        let weight_values = vocab_size
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("Qwen output weight size overflow"))?;
        require_f16_capacity(hidden_states, hidden_values, "Qwen output hidden states")?;
        let weight_end = weight_byte_offset
            .checked_add(weight_values * std::mem::size_of::<u16>())
            .ok_or_else(|| Error::backend("Qwen output weight range overflow"))?;
        require_byte_capacity(output_weight, weight_end, "Qwen output weight")?;

        let logits = self.arena.empty_f16(batch_size * vocab_size)?;
        let token_ids = self.arena.empty_u32(batch_size)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.last_token_logits,
            &[
                KernelArg::BufferOffset(output_weight, weight_byte_offset),
                KernelArg::Buffer(hidden_states),
                KernelArg::Buffer(&logits),
                KernelArg::U32(as_u32(batch_size, "output batch")?),
                KernelArg::U32(as_u32(sequence_length, "output sequence length")?),
                KernelArg::U32(as_u32(hidden_size, "output hidden size")?),
                KernelArg::U32(as_u32(vocab_size, "vocabulary size")?),
            ],
            batch_size * vocab_size,
            SIMD_LANES,
        )?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.argmax,
            &[
                KernelArg::Buffer(&logits),
                KernelArg::Buffer(&token_ids),
                KernelArg::U32(as_u32(batch_size, "argmax batch")?),
                KernelArg::U32(as_u32(vocab_size, "argmax vocabulary size")?),
            ],
            batch_size,
            256,
        )?;
        Ok(token_ids)
    }
}

fn require_tensor_range(tensor: &DeviceQwenBf16Tensor, label: &str) -> Result<()> {
    let byte_len = tensor
        .element_count()?
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| Error::backend(format!("{label} byte length overflow")))?;
    let end = tensor
        .byte_offset
        .checked_add(byte_len)
        .ok_or_else(|| Error::backend(format!("{label} range overflow")))?;
    require_byte_capacity(&tensor.buffer, end, label)
}

fn require_simd_width(pipeline: &ComputePipelineState, label: &str) -> Result<()> {
    let width = pipeline.thread_execution_width() as usize;
    if width != SIMD_LANES {
        return Err(Error::backend(format!(
            "{label} requires a {SIMD_LANES}-lane SIMD group, device reports {width}"
        )));
    }
    Ok(())
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::backend(format!("Qwen {label} exceeds u32")))
}

#[cfg(test)]
mod tests {
    use std::slice;

    use ::metal::{MTLCommandBufferStatus, MTLResourceOptions};

    use super::*;

    #[test]
    fn rms_norm_preserves_the_mlx_bf16_boundary_before_weighting() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let Ok(arena) = MetalArena::new(&device) else {
            return;
        };
        let Ok(bf16) = MetalQwenBf16::new(&device, arena) else {
            return;
        };
        let width = 128;
        let eps = 1.0e-6_f32;
        let input_values = (0..width)
            .map(|index| ((index * 17 % 97) as f32 - 48.0) / 31.0)
            .collect::<Vec<_>>();
        let weight_values = (0..width)
            .map(|index| ((index * 11 % 29) as f32 - 14.0) / 64.0)
            .collect::<Vec<_>>();
        let input = bf16_buffer(&device, &input_values);
        let weight = bf16_buffer(&device, &weight_values);
        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        let output = bf16
            .encode_rms_norm_test_raw(command_buffer, &input, width, &weight, 1, width, eps)
            .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);

        let input_values = input_values
            .iter()
            .map(|value| bf16_value(bf16_bits(*value)))
            .collect::<Vec<_>>();
        let weight_values = weight_values
            .iter()
            .map(|value| bf16_value(bf16_bits(*value)))
            .collect::<Vec<_>>();
        let inverse_rms =
            (input_values.iter().map(|value| value * value).sum::<f32>() / width as f32 + eps)
                .sqrt()
                .recip();
        let expected = input_values
            .iter()
            .zip(&weight_values)
            .map(|(value, weight)| {
                let normalized = bf16_value(bf16_bits(value * inverse_rms));
                bf16_bits(normalized * (1.0 + weight))
            })
            .collect::<Vec<_>>();
        let incorrectly_fused = input_values
            .iter()
            .zip(&weight_values)
            .map(|(value, weight)| bf16_bits(value * inverse_rms * (1.0 + weight)))
            .collect::<Vec<_>>();
        assert_ne!(
            expected, incorrectly_fused,
            "fixture must detect the missing BF16 boundary"
        );

        let actual = unsafe { slice::from_raw_parts(output.contents().cast::<u16>(), width) };
        assert_eq!(actual, expected);
    }

    #[test]
    fn output_argmax_reads_only_the_last_token_and_uses_stable_ties() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let Ok(arena) = MetalArena::new(&device) else {
            return;
        };
        let Ok(bf16) = MetalQwenBf16::new(&device, arena) else {
            return;
        };
        let weights = [
            1.0_f32, 0.0, 0.0, 0.0, // token 0
            3.0, 0.0, 0.0, 0.0, // token 1
            3.0, 0.0, 0.0, 0.0, // token 2, tied with token 1
            2.0, 0.0, 0.0, 0.0, // token 3
        ];
        let hidden = [
            100.0_f32, 0.0, 0.0, 0.0, // ignored first token
            1.0, 0.0, 0.0, 0.0, // selected last token
        ];
        let weight = bf16_buffer(&device, &weights);
        let hidden = bf16_buffer(&device, &hidden);
        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        let output = bf16
            .encode_last_token_argmax_raw(command_buffer, &weight, 0, 4, &hidden, 1, 2, 4)
            .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        let ids = unsafe { slice::from_raw_parts(output.contents().cast::<u32>(), 1) };
        assert_eq!(ids, &[1]);
    }

    #[test]
    fn verification_output_head_reuses_weights_and_preserves_stable_argmax() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let Ok(arena) = MetalArena::new(&device) else {
            return;
        };
        let Ok(bf16) = MetalQwenBf16::new(&device, arena) else {
            return;
        };
        let rows = 5;
        let hidden_size = 4;
        let vocab_size = 4;
        let weights = [
            1.0_f32, 0.0, 0.0, 0.0, // token 0
            0.0, 1.0, 0.0, 0.0, // token 1
            0.0, 0.0, 1.0, 0.0, // token 2
            0.0, 0.0, 0.0, 1.0, // token 3
        ];
        let hidden = [
            4.0_f32, 1.0, 2.0, 3.0, // token 0
            1.0, 4.0, 2.0, 3.0, // token 1
            1.0, 2.0, 4.0, 3.0, // token 2
            1.0, 2.0, 3.0, 4.0, // token 3
            1.0, 1.0, 1.0, 1.0, // stable tie selects token 0
        ];
        let weight = bf16_buffer(&device, &weights);
        let hidden_buffer = bf16_buffer(&device, &hidden);
        let logits = bf16.arena.empty_f16(rows * vocab_size).unwrap();
        let token_ids = bf16.arena.empty_u32(rows).unwrap();
        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        encode_1d_threadgroups_args(
            command_buffer,
            &bf16.verify_rows_logits,
            &[
                KernelArg::Buffer(&weight),
                KernelArg::Buffer(&hidden_buffer),
                KernelArg::Buffer(&logits),
                KernelArg::U32(rows as u32),
                KernelArg::U32(hidden_size as u32),
                KernelArg::U32(vocab_size as u32),
            ],
            vocab_size,
            SIMD_LANES,
        )
        .unwrap();
        encode_1d_threadgroups_args(
            command_buffer,
            &bf16.argmax,
            &[
                KernelArg::Buffer(&logits),
                KernelArg::Buffer(&token_ids),
                KernelArg::U32(rows as u32),
                KernelArg::U32(vocab_size as u32),
            ],
            rows,
            256,
        )
        .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);

        let actual_logits =
            unsafe { slice::from_raw_parts(logits.contents().cast::<u16>(), rows * vocab_size) };
        let expected_logits = hidden
            .iter()
            .map(|value| ((value.to_bits() + 0x7fff + ((value.to_bits() >> 16) & 1)) >> 16) as u16)
            .collect::<Vec<_>>();
        assert_eq!(actual_logits, expected_logits);
        let ids = unsafe { slice::from_raw_parts(token_ids.contents().cast::<u32>(), rows) };
        assert_eq!(ids, &[0, 1, 2, 3, 0]);
    }

    #[test]
    fn three_row_output_head_preserves_logits_and_argmax() {
        assert_specialized_output_head(3, &[0, 1, 2]);
    }

    fn assert_specialized_output_head(rows: usize, expected_ids: &[u32]) {
        let Some(device) = Device::system_default() else {
            return;
        };
        let Ok(arena) = MetalArena::new(&device) else {
            return;
        };
        let Ok(bf16) = MetalQwenBf16::new(&device, arena) else {
            return;
        };
        let hidden_size = 4;
        let vocab_size = 4;
        let weights = [
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let hidden = [
            4.0_f32, 1.0, 2.0, 3.0, 1.0, 4.0, 2.0, 3.0, 1.0, 2.0, 4.0, 3.0, 1.0, 2.0, 3.0, 4.0,
        ];
        let hidden = &hidden[..rows * hidden_size];
        let weight = bf16_buffer(&device, &weights);
        let hidden_buffer = bf16_buffer(&device, &hidden);
        let logits = bf16.arena.empty_f16(rows * vocab_size).unwrap();
        let token_ids = bf16.arena.empty_u32(rows).unwrap();
        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        encode_1d_threadgroups_args(
            command_buffer,
            &bf16.verify3_logits,
            &[
                KernelArg::Buffer(&weight),
                KernelArg::Buffer(&hidden_buffer),
                KernelArg::Buffer(&logits),
                KernelArg::U32(rows as u32),
                KernelArg::U32(hidden_size as u32),
                KernelArg::U32(vocab_size as u32),
            ],
            vocab_size,
            SIMD_LANES,
        )
        .unwrap();
        encode_1d_threadgroups_args(
            command_buffer,
            &bf16.argmax,
            &[
                KernelArg::Buffer(&logits),
                KernelArg::Buffer(&token_ids),
                KernelArg::U32(rows as u32),
                KernelArg::U32(vocab_size as u32),
            ],
            rows,
            256,
        )
        .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);

        let actual_logits =
            unsafe { slice::from_raw_parts(logits.contents().cast::<u16>(), rows * vocab_size) };
        let expected_logits = hidden
            .iter()
            .map(|value| ((value.to_bits() + 0x7fff + ((value.to_bits() >> 16) & 1)) >> 16) as u16)
            .collect::<Vec<_>>();
        assert_eq!(actual_logits, expected_logits);
        let ids = unsafe { slice::from_raw_parts(token_ids.contents().cast::<u32>(), rows) };
        assert_eq!(ids, expected_ids);
    }

    #[test]
    fn compact_draft_output_maps_control_tokens_to_full_vocabulary_ids() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let Ok(arena) = MetalArena::new(&device) else {
            return;
        };
        let Ok(bf16) = MetalQwenBf16::new(&device, arena) else {
            return;
        };
        let hidden_size = 4;
        let vocab_size = COMPACT_DRAFT_CONTROL_END;
        let mut weights = vec![0_u16; vocab_size * hidden_size];
        weights[7 * hidden_size] = bf16_bits(2.0);
        weights[248_045 * hidden_size] = bf16_bits(3.0);
        let weight = device.new_buffer_with_data(
            weights.as_ptr().cast(),
            (weights.len() * std::mem::size_of::<u16>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let hidden = bf16_buffer(&device, &[1.0, 0.0, 0.0, 0.0]);
        let logits = bf16.arena.empty_f16(COMPACT_DRAFT_VOCAB).unwrap();
        let token_ids = bf16.arena.empty_u32(1).unwrap();
        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        encode_1d_threadgroups_args(
            command_buffer,
            &bf16.compact_draft_logits,
            &[
                KernelArg::Buffer(&weight),
                KernelArg::Buffer(&hidden),
                KernelArg::Buffer(&logits),
                KernelArg::U32(1),
                KernelArg::U32(1),
                KernelArg::U32(hidden_size as u32),
            ],
            COMPACT_DRAFT_VOCAB,
            SIMD_LANES,
        )
        .unwrap();
        encode_1d_threadgroups_args(
            command_buffer,
            &bf16.compact_draft_argmax,
            &[
                KernelArg::Buffer(&logits),
                KernelArg::Buffer(&token_ids),
                KernelArg::U32(1),
            ],
            1,
            256,
        )
        .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        let ids = unsafe { slice::from_raw_parts(token_ids.contents().cast::<u32>(), 1) };
        assert_eq!(ids, &[248_045]);
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        let bytes = values
            .iter()
            .flat_map(|value| {
                let bits = value.to_bits();
                let rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
                rounded.to_le_bytes()
            })
            .collect::<Vec<_>>();
        device.new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }

    fn bf16_value(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }
}
