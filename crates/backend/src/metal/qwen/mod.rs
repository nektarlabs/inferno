mod attention;
mod bf16;
mod delta;
mod delta_prefill;
mod dflash;
mod mlx_w4;
mod packed_prefill;
mod views;

use std::sync::OnceLock;

use ::metal::Device;
use common::{Error, Result};
use inferno_io::{SafeTensorDtype, SafeTensorHandle, SafeTensorInfo};

use crate::{
    DFlashAttentionCache, DeviceDFlashW4Matrix, DeviceQwenBf16Tensor, DeviceQwenMlxW4Matrix,
    QwenFullAttentionCache, QwenLinearAttentionCache,
};

use self::attention::MetalQwenAttention;
use self::bf16::MetalQwenBf16;
use self::delta::MetalQwenDelta;
use self::dflash::MetalDFlash;
use self::mlx_w4::MetalQwenMlxW4;
use self::views::QwenShardViews;
use super::arena::MetalArena;

pub(crate) struct MetalQwen {
    views: QwenShardViews,
    arena: MetalArena,
    attention: OnceLock<std::result::Result<MetalQwenAttention, String>>,
    bf16: OnceLock<std::result::Result<MetalQwenBf16, String>>,
    delta: OnceLock<std::result::Result<MetalQwenDelta, String>>,
    dflash: OnceLock<std::result::Result<MetalDFlash, String>>,
    mlx_w4: OnceLock<std::result::Result<MetalQwenMlxW4, String>>,
}

impl MetalQwen {
    pub(crate) fn new(arena: MetalArena) -> Self {
        Self {
            views: QwenShardViews::new(),
            arena,
            attention: OnceLock::new(),
            bf16: OnceLock::new(),
            delta: OnceLock::new(),
            dflash: OnceLock::new(),
            mlx_w4: OnceLock::new(),
        }
    }

    pub(crate) fn prepare_bf16_tensor(
        &self,
        device: &Device,
        tensor: SafeTensorHandle,
    ) -> Result<DeviceQwenBf16Tensor> {
        if tensor.info().dtype != SafeTensorDtype::Bf16 {
            return Err(Error::backend(format!(
                "Qwen BF16 tensor {} has dtype {:?}",
                tensor.info().name,
                tensor.info().dtype
            )));
        }
        let element_count = tensor
            .info()
            .shape
            .iter()
            .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or_else(|| Error::backend("Qwen BF16 tensor element count overflow"))?;
        let expected_bytes = element_count
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| Error::backend("Qwen BF16 tensor byte count overflow"))?;
        if tensor.info().byte_len != expected_bytes as u64 {
            return Err(Error::backend(format!(
                "Qwen BF16 tensor {} requires {expected_bytes} bytes, got {}",
                tensor.info().name,
                tensor.info().byte_len
            )));
        }
        let binding = self.views.binding(device, &tensor)?;
        Ok(DeviceQwenBf16Tensor {
            shape: tensor.info().shape.clone(),
            byte_offset: binding.byte_offset,
            buffer: binding.buffer,
            _owner: tensor,
        })
    }

    pub(crate) fn prepare_mlx_w4_matrix(
        &self,
        device: &Device,
        weight: SafeTensorHandle,
        scales: SafeTensorHandle,
        biases: SafeTensorHandle,
        rows: usize,
        columns: usize,
    ) -> Result<DeviceQwenMlxW4Matrix> {
        validate_mlx_w4_layout(weight.info(), scales.info(), biases.info(), rows, columns)?;
        let weight_binding = self.views.binding(device, &weight)?;
        let scale_binding = self.views.binding(device, &scales)?;
        let bias_binding = self.views.binding(device, &biases)?;
        Ok(DeviceQwenMlxW4Matrix {
            rows,
            columns,
            groups_per_row: columns / 64,
            weight_byte_offset: weight_binding.byte_offset,
            scale_byte_offset: scale_binding.byte_offset,
            bias_byte_offset: bias_binding.byte_offset,
            weight: weight_binding.buffer,
            scales: scale_binding.buffer,
            biases: bias_binding.buffer,
            _weight_owner: weight,
            _scale_owner: scales,
            _bias_owner: biases,
        })
    }

    pub(crate) fn encode_mlx_w4_linear(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        matrix: &DeviceQwenMlxW4Matrix,
        input: &::metal::Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<::metal::Buffer> {
        self.mlx_w4(device)?
            .encode_linear(command_buffer, matrix, input, input_len, row_count)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_mlx_w4_linear_add(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        matrix: &DeviceQwenMlxW4Matrix,
        input: &::metal::Buffer,
        residual: &::metal::Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Option<::metal::Buffer>> {
        self.mlx_w4(device)?.encode_linear_add(
            command_buffer,
            matrix,
            input,
            residual,
            input_len,
            row_count,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_mlx_w4_gate_up(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        gate: &DeviceQwenMlxW4Matrix,
        up: &DeviceQwenMlxW4Matrix,
        input: &::metal::Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<::metal::Buffer> {
        self.mlx_w4(device)?
            .encode_gate_up(command_buffer, gate, up, input, input_len, row_count)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_mlx_w4_qkv(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        query: &DeviceQwenMlxW4Matrix,
        key: &DeviceQwenMlxW4Matrix,
        value: &DeviceQwenMlxW4Matrix,
        input: &::metal::Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<(::metal::Buffer, ::metal::Buffer, ::metal::Buffer)> {
        self.mlx_w4(device)?.encode_qkv(
            command_buffer,
            query,
            key,
            value,
            input,
            input_len,
            row_count,
        )
    }

    pub(crate) fn encode_mlx_w4_embedding(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        matrix: &DeviceQwenMlxW4Matrix,
        token_ids: &[u32],
    ) -> Result<::metal::Buffer> {
        self.mlx_w4(device)?
            .encode_embedding(command_buffer, matrix, token_ids)
    }

    pub(crate) fn encode_mlx_w4_embedding_from_device(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        matrix: &DeviceQwenMlxW4Matrix,
        token_ids: &::metal::Buffer,
        token_count: usize,
    ) -> Result<::metal::Buffer> {
        self.mlx_w4(device)?.encode_embedding_from_device(
            command_buffer,
            matrix,
            token_ids,
            token_count,
        )
    }

    pub(crate) fn encode_mlx_w4_argmax(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        matrix: &DeviceQwenMlxW4Matrix,
        hidden: &::metal::Buffer,
        hidden_len: usize,
        row_count: usize,
    ) -> Result<::metal::Buffer> {
        self.mlx_w4(device)?
            .encode_argmax(command_buffer, matrix, hidden, hidden_len, row_count)
    }

    pub(crate) fn encode_mlx_w4_last_argmax(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        matrix: &DeviceQwenMlxW4Matrix,
        hidden: &::metal::Buffer,
        batch: usize,
        sequence_length: usize,
    ) -> Result<::metal::Buffer> {
        self.mlx_w4(device)?.encode_last_argmax(
            command_buffer,
            matrix,
            hidden,
            batch,
            sequence_length,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_mlx_w4_linear_attention(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        input: &::metal::Buffer,
        input_len: usize,
        qkv: &DeviceQwenMlxW4Matrix,
        gate: &DeviceQwenMlxW4Matrix,
        input_a: &DeviceQwenMlxW4Matrix,
        input_b: &DeviceQwenMlxW4Matrix,
        conv1d: &DeviceQwenBf16Tensor,
        a_log: &DeviceQwenBf16Tensor,
        dt_bias: &DeviceQwenBf16Tensor,
        norm: &DeviceQwenBf16Tensor,
        cache: &QwenLinearAttentionCache,
        row_count: usize,
        sequence_length: usize,
        eps: f32,
    ) -> Result<::metal::Buffer> {
        let w4 = self.mlx_w4(device)?;
        let (mixed_qkv, gate) =
            w4.encode_pair(command_buffer, qkv, gate, input, input_len, row_count)?;
        let (input_a, input_b) = w4.encode_pair(
            command_buffer,
            input_a,
            input_b,
            input,
            input_len,
            row_count,
        )?;
        self.delta(device)?.encode(
            command_buffer,
            &mixed_qkv,
            &gate,
            &input_a,
            &input_b,
            conv1d,
            a_log,
            dt_bias,
            norm,
            cache,
            sequence_length,
            eps,
        )
    }

    pub(crate) fn encode_bf16_embedding(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        embedding: &DeviceQwenBf16Tensor,
        token_ids: &[u32],
    ) -> Result<::metal::Buffer> {
        self.bf16(device)?
            .encode_embedding(command_buffer, embedding, token_ids)
    }

    pub(crate) fn encode_bf16_rms_norm(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        input: &::metal::Buffer,
        input_len: usize,
        weight: &DeviceQwenBf16Tensor,
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<::metal::Buffer> {
        self.bf16(device)?.encode_rms_norm(
            command_buffer,
            input,
            input_len,
            weight,
            rows,
            hidden_size,
            eps,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_bf16_rms_norm_standard(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        input: &::metal::Buffer,
        input_len: usize,
        weight: &DeviceQwenBf16Tensor,
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<::metal::Buffer> {
        self.bf16(device)?.encode_rms_norm_standard(
            command_buffer,
            input,
            input_len,
            weight,
            rows,
            hidden_size,
            eps,
        )
    }

    pub(crate) fn encode_bf16_linear(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        matrix: &DeviceQwenBf16Tensor,
        input: &::metal::Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<::metal::Buffer> {
        self.bf16(device)?
            .encode_linear(command_buffer, matrix, input, input_len, row_count)
    }

    pub(crate) fn encode_bf16_add(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        left: &::metal::Buffer,
        right: &::metal::Buffer,
        element_count: usize,
    ) -> Result<::metal::Buffer> {
        self.bf16(device)?
            .encode_add(command_buffer, left, right, element_count)
    }

    pub(crate) fn encode_bf16_embedding_from_device(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        embedding: &DeviceQwenBf16Tensor,
        token_ids: &::metal::Buffer,
        token_count: usize,
    ) -> Result<::metal::Buffer> {
        self.bf16(device)?.encode_embedding_from_device(
            command_buffer,
            embedding,
            token_ids,
            token_count,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_bf16_concat_last(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        left: &::metal::Buffer,
        right: &::metal::Buffer,
        row_count: usize,
        left_width: usize,
        right_width: usize,
    ) -> Result<::metal::Buffer> {
        self.bf16(device)?.encode_concat_last(
            command_buffer,
            left,
            right,
            row_count,
            left_width,
            right_width,
        )
    }

    pub(crate) fn encode_bf16_copy_row(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        input: &::metal::Buffer,
        row_count: usize,
        row_width: usize,
        row_index: usize,
    ) -> Result<::metal::Buffer> {
        self.bf16(device)?
            .encode_copy_row(command_buffer, input, row_count, row_width, row_index)
    }

    pub(crate) fn encode_bf16_last_token_argmax(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        output_weight: &DeviceQwenBf16Tensor,
        hidden_states: &::metal::Buffer,
        batch_size: usize,
        sequence_length: usize,
        hidden_size: usize,
    ) -> Result<::metal::Buffer> {
        self.bf16(device)?.encode_last_token_argmax(
            command_buffer,
            output_weight,
            hidden_states,
            batch_size,
            sequence_length,
            hidden_size,
        )
    }

    pub(crate) fn encode_bf16_all_token_argmax(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        output_weight: &DeviceQwenBf16Tensor,
        hidden_states: &::metal::Buffer,
        row_count: usize,
        hidden_size: usize,
    ) -> Result<::metal::Buffer> {
        self.bf16(device)?.encode_all_token_argmax(
            command_buffer,
            output_weight,
            hidden_states,
            row_count,
            hidden_size,
        )
    }

    pub(crate) fn encode_bf16_compact_draft_argmax(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        output_weight: &DeviceQwenBf16Tensor,
        hidden_states: &::metal::Buffer,
        batch_size: usize,
        sequence_length: usize,
        hidden_size: usize,
    ) -> Result<::metal::Buffer> {
        self.bf16(device)?.encode_compact_draft_argmax(
            command_buffer,
            output_weight,
            hidden_states,
            batch_size,
            sequence_length,
            hidden_size,
        )
    }

    pub(crate) fn encode_dflash_pack_features(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        features: [&::metal::Buffer; 5],
        row_count: usize,
    ) -> Result<::metal::Buffer> {
        self.dflash(device)?
            .encode_pack_features(command_buffer, features, row_count)
    }

    pub(crate) fn encode_dflash_copy_rows(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        input: &::metal::Buffer,
        row_start: usize,
        row_count: usize,
        row_width: usize,
    ) -> Result<::metal::Buffer> {
        self.dflash(device)?.encode_copy_rows(
            command_buffer,
            input,
            row_start,
            row_count,
            row_width,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_dflash_rms_norm(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        input: &::metal::Buffer,
        input_len: usize,
        weight: &DeviceQwenBf16Tensor,
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<::metal::Buffer> {
        self.dflash(device)?.encode_rms_norm(
            command_buffer,
            input,
            input_len,
            weight,
            rows,
            hidden_size,
            eps,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_dflash_rms_norm_suffix(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        input: &::metal::Buffer,
        input_len: usize,
        weight: &DeviceQwenBf16Tensor,
        total_rows: usize,
        suffix_rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<::metal::Buffer> {
        self.dflash(device)?.encode_rms_norm_suffix(
            command_buffer,
            input,
            input_len,
            weight,
            total_rows,
            suffix_rows,
            hidden_size,
            eps,
        )
    }

    pub(crate) fn encode_dflash_linear(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        matrix: &DeviceQwenBf16Tensor,
        input: &::metal::Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<::metal::Buffer> {
        self.dflash(device)?
            .encode_linear(command_buffer, matrix, input, input_len, row_count)
    }

    pub(crate) fn encode_dflash_quantize_w4(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        source: &DeviceQwenBf16Tensor,
        rows: usize,
        columns: usize,
        group_size: usize,
    ) -> Result<DeviceDFlashW4Matrix> {
        self.dflash(device)?.encode_quantize_w4(
            device,
            command_buffer,
            source,
            rows,
            columns,
            group_size,
        )
    }

    pub(crate) fn encode_dflash_w4_linear(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        matrix: &DeviceDFlashW4Matrix,
        input: &::metal::Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<::metal::Buffer> {
        self.dflash(device)?
            .encode_w4_linear(command_buffer, matrix, input, input_len, row_count)
    }

    pub(crate) fn encode_dflash_w4_gate_up_swiglu(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        gate: &DeviceDFlashW4Matrix,
        up: &DeviceDFlashW4Matrix,
        input: &::metal::Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<::metal::Buffer> {
        self.dflash(device)?.encode_w4_gate_up_swiglu(
            command_buffer,
            gate,
            up,
            input,
            input_len,
            row_count,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_dflash_dynamic_conv(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        input: &::metal::Buffer,
        dynamic: &::metal::Buffer,
        base_kernel: &DeviceQwenBf16Tensor,
        rows: usize,
        sequence_length: usize,
        hidden_size: usize,
        stage: usize,
        kernel_size: usize,
        group_size: usize,
    ) -> Result<::metal::Buffer> {
        self.dflash(device)?.encode_dynamic_conv(
            command_buffer,
            input,
            dynamic,
            base_kernel,
            rows,
            sequence_length,
            hidden_size,
            stage,
            kernel_size,
            group_size,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_dflash_dynamic_conv_residual(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        input: &::metal::Buffer,
        dynamic: &::metal::Buffer,
        base_kernel: &DeviceQwenBf16Tensor,
        residual: &::metal::Buffer,
        rows: usize,
        sequence_length: usize,
        hidden_size: usize,
        kernel_size: usize,
        group_size: usize,
    ) -> Result<::metal::Buffer> {
        self.dflash(device)?.encode_dynamic_conv_residual(
            command_buffer,
            input,
            dynamic,
            base_kernel,
            residual,
            rows,
            sequence_length,
            hidden_size,
            kernel_size,
            group_size,
        )
    }

    pub(crate) fn encode_dflash_gate_up_swiglu(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        gate: &DeviceQwenBf16Tensor,
        up: &DeviceQwenBf16Tensor,
        input: &::metal::Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<::metal::Buffer> {
        self.dflash(device)?.encode_gate_up_swiglu(
            command_buffer,
            gate,
            up,
            input,
            input_len,
            row_count,
        )
    }

    pub(crate) fn create_dflash_attention_cache(
        &self,
        device: &Device,
        batch: usize,
        capacity_tokens: usize,
    ) -> Result<DFlashAttentionCache> {
        self.dflash(device)?
            .create_attention_cache(device, batch, capacity_tokens)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_dflash_attention(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        query: &::metal::Buffer,
        context_key: &::metal::Buffer,
        context_value: &::metal::Buffer,
        proposal_key: &::metal::Buffer,
        proposal_value: &::metal::Buffer,
        query_norm: &DeviceQwenBf16Tensor,
        key_norm: &DeviceQwenBf16Tensor,
        cache: &DFlashAttentionCache,
        batch: usize,
        proposal_length: usize,
        context_length: usize,
        context_position_start: usize,
        rope_theta: f32,
        sliding_window: usize,
    ) -> Result<::metal::Buffer> {
        self.dflash(device)?.encode_attention(
            command_buffer,
            query,
            context_key,
            context_value,
            proposal_key,
            proposal_value,
            query_norm,
            key_norm,
            cache,
            batch,
            proposal_length,
            context_length,
            context_position_start,
            rope_theta,
            sliding_window,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_dflash_select_candidates(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        output_weight: &DeviceQwenBf16Tensor,
        hidden_states: &::metal::Buffer,
        projected_hidden: &::metal::Buffer,
        predecessor_codebook: &DeviceQwenBf16Tensor,
        successor_codebook: &DeviceQwenBf16Tensor,
        row_count: usize,
        anchor_token: u32,
        top_k: usize,
    ) -> Result<::metal::Buffer> {
        self.dflash(device)?.encode_select_candidates(
            command_buffer,
            output_weight,
            hidden_states,
            projected_hidden,
            predecessor_codebook,
            successor_codebook,
            row_count,
            anchor_token,
            top_k,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_dflash_mlx_w4_select_candidates(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        output_weight: &DeviceQwenMlxW4Matrix,
        hidden_states: &::metal::Buffer,
        hidden_len: usize,
        projected_hidden: &::metal::Buffer,
        predecessor_codebook: &DeviceQwenBf16Tensor,
        successor_codebook: &DeviceQwenBf16Tensor,
        row_count: usize,
        anchor_token: u32,
        top_k: usize,
    ) -> Result<::metal::Buffer> {
        let (candidate_ids, candidate_logits) = self.mlx_w4(device)?.encode_top_k(
            command_buffer,
            output_weight,
            hidden_states,
            hidden_len,
            row_count,
            top_k,
        )?;
        self.dflash(device)?.encode_path_selector(
            command_buffer,
            &candidate_ids,
            &candidate_logits,
            projected_hidden,
            predecessor_codebook,
            successor_codebook,
            row_count,
            anchor_token,
            top_k,
        )
    }

    pub(crate) fn create_full_attention_cache(
        &self,
        device: &Device,
        batch: usize,
        capacity_tokens: usize,
    ) -> Result<QwenFullAttentionCache> {
        self.attention(device)?
            .create_cache(device, batch, capacity_tokens)
    }

    pub(crate) fn create_linear_attention_cache(
        &self,
        device: &Device,
        batch: usize,
    ) -> Result<QwenLinearAttentionCache> {
        self.delta(device)?.create_cache(device, batch, false)
    }

    pub(crate) fn create_speculative_linear_attention_cache(
        &self,
        device: &Device,
        batch: usize,
    ) -> Result<QwenLinearAttentionCache> {
        self.delta(device)?.create_cache(device, batch, true)
    }

    pub(crate) fn encode_restore_linear_attention_checkpoint(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        cache: &QwenLinearAttentionCache,
        checkpoint_index: usize,
    ) -> Result<()> {
        self.delta(device)?
            .encode_restore(command_buffer, cache, checkpoint_index)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_full_attention(
        &self,
        device: &Device,
        command_buffer: &::metal::CommandBufferRef,
        query_gate: &::metal::Buffer,
        key: &::metal::Buffer,
        value: &::metal::Buffer,
        query_norm: &DeviceQwenBf16Tensor,
        key_norm: &DeviceQwenBf16Tensor,
        cache: &QwenFullAttentionCache,
        row_count: usize,
        sequence_length: usize,
        rope_theta: f32,
        rotary_dim: usize,
        norm_weight_has_unit_offset: bool,
    ) -> Result<::metal::Buffer> {
        self.attention(device)?.encode(
            command_buffer,
            query_gate,
            key,
            value,
            query_norm,
            key_norm,
            cache,
            row_count,
            sequence_length,
            rope_theta,
            rotary_dim,
            norm_weight_has_unit_offset,
        )
    }

    fn attention(&self, device: &Device) -> Result<&MetalQwenAttention> {
        self.attention
            .get_or_init(|| {
                MetalQwenAttention::new(device, self.arena.clone())
                    .map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|message| Error::backend(message.clone()))
    }

    fn bf16(&self, device: &Device) -> Result<&MetalQwenBf16> {
        self.bf16
            .get_or_init(|| {
                MetalQwenBf16::new(device, self.arena.clone()).map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|message| Error::backend(message.clone()))
    }

    fn delta(&self, device: &Device) -> Result<&MetalQwenDelta> {
        self.delta
            .get_or_init(|| {
                MetalQwenDelta::new(device, self.arena.clone()).map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|message| Error::backend(message.clone()))
    }

    fn dflash(&self, device: &Device) -> Result<&MetalDFlash> {
        self.dflash
            .get_or_init(|| {
                MetalDFlash::new(device, self.arena.clone()).map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|message| Error::backend(message.clone()))
    }

    fn mlx_w4(&self, device: &Device) -> Result<&MetalQwenMlxW4> {
        self.mlx_w4
            .get_or_init(|| {
                MetalQwenMlxW4::new(device, self.arena.clone()).map_err(|error| error.to_string())
            })
            .as_ref()
            .map_err(|message| Error::backend(message.clone()))
    }
}

fn validate_mlx_w4_layout(
    weight: &SafeTensorInfo,
    scales: &SafeTensorInfo,
    biases: &SafeTensorInfo,
    rows: usize,
    columns: usize,
) -> Result<()> {
    const GROUP_SIZE: usize = 64;
    const VALUES_PER_WORD: usize = 8;
    if rows == 0 || columns == 0 || !columns.is_multiple_of(GROUP_SIZE) {
        return Err(Error::backend(format!(
            "Qwen MLX W4 matrix [{rows},{columns}] must have positive dimensions and a width divisible by {GROUP_SIZE}"
        )));
    }
    require_tensor(
        weight,
        SafeTensorDtype::U32,
        &[rows, columns / VALUES_PER_WORD],
        "MLX W4 weight",
    )?;
    let parameter_shape = [rows, columns / GROUP_SIZE];
    require_tensor(
        scales,
        SafeTensorDtype::Bf16,
        &parameter_shape,
        "MLX W4 scales",
    )?;
    require_tensor(
        biases,
        SafeTensorDtype::Bf16,
        &parameter_shape,
        "MLX W4 biases",
    )
}

fn require_tensor(
    info: &SafeTensorInfo,
    dtype: SafeTensorDtype,
    shape: &[usize],
    label: &str,
) -> Result<()> {
    if info.dtype != dtype || info.shape != shape {
        return Err(Error::backend(format!(
            "Qwen {label} {} must be {dtype:?} {shape:?}, got {:?} {:?}",
            info.name, info.dtype, info.shape
        )));
    }
    let expected_bytes = shape
        .iter()
        .try_fold(dtype.byte_width(), |bytes, dimension| {
            bytes.checked_mul(*dimension)
        })
        .ok_or_else(|| Error::backend(format!("Qwen {label} byte count overflow")))?;
    if info.byte_len != expected_bytes as u64 {
        return Err(Error::backend(format!(
            "Qwen {label} {} must contain {expected_bytes} bytes, got {}",
            info.name, info.byte_len
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use crate::{Backend, MetalBackend};
    use inferno_io::SafeTensorModel;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn zero_copy_bf16_embedding_norm_and_linear_stay_in_one_device_chain() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model_dir = write_bf16_fixture();
        let model = SafeTensorModel::open(&model_dir).unwrap();
        let embedding = backend
            .prepare_qwen_bf16_tensor(model.tensor("embedding").unwrap())
            .unwrap()
            .unwrap();
        let norm = backend
            .prepare_qwen_bf16_tensor(model.tensor("norm").unwrap())
            .unwrap()
            .unwrap();
        let linear = backend
            .prepare_qwen_bf16_tensor(model.tensor("linear").unwrap())
            .unwrap()
            .unwrap();

        let hidden = backend
            .qwen_bf16_embedding_device(&embedding, &[1], &[1, 1])
            .unwrap()
            .unwrap();
        assert_eq!(hidden.dims(), [1, 1, 128]);
        let hidden_values = backend.device_download_f32_tensor(&hidden).unwrap();
        assert!(
            hidden_values.values().iter().all(|value| *value == 2.0),
            "embedding output starts with {:?}",
            &hidden_values.values()[..8]
        );
        let normalized = backend
            .qwen_bf16_rms_norm_device(&hidden, &norm, 1e-6)
            .unwrap()
            .unwrap();
        let normalized_values = backend.device_download_f32_tensor(&normalized).unwrap();
        assert!(
            normalized_values
                .values()
                .iter()
                .all(|value| (*value - 1.0).abs() <= 0.01),
            "RMSNorm output starts with {:?}",
            &normalized_values.values()[..8]
        );
        let projected = backend
            .qwen_bf16_linear_device(&linear, &normalized)
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&projected).unwrap();

        assert_eq!(output.dims(), [1, 1, 128]);
        assert!(output
            .values()
            .iter()
            .all(|value| (*value - 1.0).abs() <= 0.01));
    }

    #[test]
    fn resident_bf16_copy_matches_zero_copy_values() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model_dir = write_bf16_fixture();
        let model = SafeTensorModel::open(&model_dir).unwrap();
        let mapped = backend
            .prepare_qwen_bf16_tensor(model.tensor("embedding").unwrap())
            .unwrap()
            .unwrap();
        let resident = backend
            .prepare_qwen_bf16_tensor_resident(model.tensor("embedding").unwrap())
            .unwrap()
            .unwrap();

        assert_eq!(resident.byte_offset, 0);
        let mapped_hidden = backend
            .qwen_bf16_embedding_device(&mapped, &[1], &[1, 1])
            .unwrap()
            .unwrap();
        let resident_hidden = backend
            .qwen_bf16_embedding_device(&resident, &[1], &[1, 1])
            .unwrap()
            .unwrap();
        let mapped_values = backend.device_download_f32_tensor(&mapped_hidden).unwrap();
        let resident_values = backend
            .device_download_f32_tensor(&resident_hidden)
            .unwrap();

        assert_eq!(resident_values.dims(), [1, 1, 128]);
        assert_eq!(resident_values.values(), mapped_values.values());
    }

    #[test]
    fn dflash_direct_rms_norm_compiles_and_does_not_add_one_to_weight() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model_dir = write_bf16_fixture();
        let model = SafeTensorModel::open(&model_dir).unwrap();
        let embedding = backend
            .prepare_qwen_bf16_tensor(model.tensor("embedding").unwrap())
            .unwrap()
            .unwrap();
        let zero_norm = backend
            .prepare_qwen_bf16_tensor(model.tensor("norm").unwrap())
            .unwrap()
            .unwrap();
        let hidden = backend
            .qwen_bf16_embedding_device(&embedding, &[1], &[1, 1])
            .unwrap()
            .unwrap();
        let normalized = backend
            .dflash_rms_norm_device(&hidden, &zero_norm, 1e-6)
            .unwrap()
            .unwrap();
        let values = backend.device_download_f32_tensor(&normalized).unwrap();

        assert!(values.values().iter().all(|value| *value == 0.0));
    }

    #[test]
    fn dflash_row_tiled_linear_matches_qwen_and_fuses_swiglu() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model_dir = write_bf16_fixture();
        let model = SafeTensorModel::open(&model_dir).unwrap();
        let embedding = backend
            .prepare_qwen_bf16_tensor(model.tensor("embedding").unwrap())
            .unwrap()
            .unwrap();
        let linear = backend
            .prepare_qwen_bf16_tensor(model.tensor("linear").unwrap())
            .unwrap()
            .unwrap();
        let token_ids = vec![1_u32; 19];
        let hidden = backend
            .qwen_bf16_embedding_device(&embedding, &token_ids, &[1, token_ids.len()])
            .unwrap()
            .unwrap();

        let reference = backend
            .qwen_bf16_linear_device(&linear, &hidden)
            .unwrap()
            .unwrap();
        let tiled = backend
            .dflash_bf16_linear_device(&linear, &hidden)
            .unwrap()
            .unwrap();
        let reference = backend.device_download_f32_tensor(&reference).unwrap();
        let tiled = backend.device_download_f32_tensor(&tiled).unwrap();
        assert_eq!(tiled.dims(), [1, 19, 128]);
        assert_eq!(tiled.values(), reference.values());

        let activated = backend
            .dflash_bf16_gate_up_swiglu_device(&linear, &linear, &hidden)
            .unwrap()
            .unwrap();
        let activated = backend.device_download_f32_tensor(&activated).unwrap();
        assert_eq!(activated.dims(), [1, 19, 128]);
        assert!(activated
            .values()
            .iter()
            .all(|value| (*value - 3.515625).abs() <= 0.01));
    }

    #[test]
    fn dflash_w4_quantization_and_row_tiled_projection_are_numerically_sane() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model_dir = write_bf16_fixture();
        let model = SafeTensorModel::open(&model_dir).unwrap();
        let embedding = backend
            .prepare_qwen_bf16_tensor(model.tensor("embedding").unwrap())
            .unwrap()
            .unwrap();
        let weight = backend
            .prepare_dflash_w4_matrix(model.tensor("linear").unwrap(), 128, 128, 64)
            .unwrap()
            .unwrap();
        backend.device_flush().unwrap();
        assert_eq!(weight.storage_bytes().unwrap(), 128 * 64 + 128 * 2 * 2 * 2);

        let token_ids = vec![1_u32; 19];
        let hidden = backend
            .qwen_bf16_embedding_device(&embedding, &token_ids, &[1, token_ids.len()])
            .unwrap()
            .unwrap();

        let projected = backend
            .dflash_w4_linear_device(&weight, &hidden)
            .unwrap()
            .unwrap();
        let projected = backend.device_download_f32_tensor(&projected).unwrap();
        let affine_projection = 1.5703125_f32;
        assert_eq!(projected.dims(), [1, 19, 128]);
        assert!(
            projected
                .values()
                .iter()
                .all(|value| (*value - affine_projection).abs() <= 0.01),
            "affine W4 projection starts with {:?}",
            &projected.values()[..8]
        );

        let activated = backend
            .dflash_w4_gate_up_swiglu_device(&weight, &weight, &hidden)
            .unwrap()
            .unwrap();
        let activated = backend.device_download_f32_tensor(&activated).unwrap();
        let expected_activation =
            affine_projection / (1.0 + (-affine_projection).exp()) * affine_projection;
        assert_eq!(activated.dims(), [1, 19, 128]);
        assert!(activated
            .values()
            .iter()
            .all(|value| (*value - expected_activation).abs() <= 0.02));
    }

    #[test]
    fn dflash_fused_dynamic_convolution_residual_is_bit_exact() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model_dir = write_bf16_fixture();
        let model = SafeTensorModel::open(&model_dir).unwrap();
        let embedding = backend
            .prepare_qwen_bf16_tensor(model.tensor("embedding").unwrap())
            .unwrap()
            .unwrap();
        let base_kernel = backend
            .prepare_qwen_bf16_tensor(model.tensor("conv").unwrap())
            .unwrap()
            .unwrap();
        let dynamic_projection = backend
            .prepare_qwen_bf16_tensor(model.tensor("dynamic").unwrap())
            .unwrap()
            .unwrap();
        let token_ids = vec![1_u32; 7];
        let input = backend
            .qwen_bf16_embedding_device(&embedding, &token_ids, &[1, token_ids.len()])
            .unwrap()
            .unwrap();
        let dynamic = backend
            .dflash_bf16_linear_device(&dynamic_projection, &input)
            .unwrap()
            .unwrap();

        let convolved = backend
            .dflash_dynamic_conv_device(&input, &dynamic, &base_kernel, 1, 2, 16)
            .unwrap()
            .unwrap();
        let reference = backend
            .qwen_bf16_add_device(&input, &convolved)
            .unwrap()
            .unwrap();
        let fused = backend
            .dflash_dynamic_conv_residual_device(&input, &dynamic, &base_kernel, &input, 2, 16)
            .unwrap()
            .unwrap();
        let reference = backend.device_download_f32_tensor(&reference).unwrap();
        let fused = backend.device_download_f32_tensor(&fused).unwrap();

        assert_eq!(fused.dims(), [1, 7, 128]);
        assert_eq!(fused.values(), reference.values());
    }

    #[test]
    fn dflash_suffix_rms_norm_matches_full_norm_then_copy_bit_for_bit() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let model_dir = write_bf16_fixture();
        let model = SafeTensorModel::open(&model_dir).unwrap();
        let embedding = backend
            .prepare_qwen_bf16_tensor(model.tensor("embedding").unwrap())
            .unwrap()
            .unwrap();
        let norm = backend
            .prepare_qwen_bf16_tensor(model.tensor("direct_norm").unwrap())
            .unwrap()
            .unwrap();
        let token_ids = [0_u32, 1, 1, 0, 1, 0, 1, 1];
        let hidden = backend
            .qwen_bf16_embedding_device(&embedding, &token_ids, &[1, token_ids.len()])
            .unwrap()
            .unwrap();

        let full = backend
            .dflash_rms_norm_device(&hidden, &norm, 1e-6)
            .unwrap()
            .unwrap();
        let reference = backend
            .dflash_bf16_suffix_rows_device(&full, 7)
            .unwrap()
            .unwrap();
        let fused = backend
            .dflash_rms_norm_suffix_device(&hidden, &norm, 1e-6, 7)
            .unwrap()
            .unwrap();
        let reference = backend.device_download_f32_tensor(&reference).unwrap();
        let fused = backend.device_download_f32_tensor(&fused).unwrap();

        assert_eq!(fused.dims(), [1, 7, 128]);
        assert_eq!(fused.values(), reference.values());
    }

    fn write_bf16_fixture() -> std::path::PathBuf {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let model_dir =
            std::env::temp_dir().join(format!("inferno-qwen-bf16-{}-{id}", std::process::id()));
        fs::create_dir_all(&model_dir).unwrap();
        let embedding = [vec![0.0_f32; 128], vec![2.0_f32; 128]].concat();
        let linear = (0..128 * 128)
            .map(|index| if index / 128 == index % 128 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        // Official Qwen RMSNorm stores an additive delta: scale = 1 + weight.
        let norm = vec![0.0_f32; 128];
        let conv = [
            vec![0.0_f32; 2 * 128],
            vec![0.5_f32; 128],
            vec![0.25_f32; 128],
        ]
        .concat();
        let dynamic = vec![0.0_f32; 32 * 128];
        let direct_norm = (0..128)
            .map(|index| 0.5_f32 + index as f32 / 256.0)
            .collect::<Vec<_>>();
        let embedding_bytes = bf16_bytes(&embedding);
        let linear_bytes = bf16_bytes(&linear);
        let norm_bytes = bf16_bytes(&norm);
        let conv_bytes = bf16_bytes(&conv);
        let dynamic_bytes = bf16_bytes(&dynamic);
        let direct_norm_bytes = bf16_bytes(&direct_norm);
        let linear_start = embedding_bytes.len();
        let norm_start = linear_start + linear_bytes.len();
        let conv_start = norm_start + norm_bytes.len();
        let dynamic_start = conv_start + conv_bytes.len();
        let direct_norm_start = dynamic_start + dynamic_bytes.len();
        let data_len = direct_norm_start + direct_norm_bytes.len();
        let mut header = format!(
            "{{\"embedding\":{{\"dtype\":\"BF16\",\"shape\":[2,128],\"data_offsets\":[0,{}]}},\"linear\":{{\"dtype\":\"BF16\",\"shape\":[128,128],\"data_offsets\":[{linear_start},{norm_start}]}},\"norm\":{{\"dtype\":\"BF16\",\"shape\":[128],\"data_offsets\":[{norm_start},{conv_start}]}},\"conv\":{{\"dtype\":\"BF16\",\"shape\":[2,2,128],\"data_offsets\":[{conv_start},{dynamic_start}]}},\"dynamic\":{{\"dtype\":\"BF16\",\"shape\":[32,128],\"data_offsets\":[{dynamic_start},{direct_norm_start}]}},\"direct_norm\":{{\"dtype\":\"BF16\",\"shape\":[128],\"data_offsets\":[{direct_norm_start},{data_len}]}}}}",
            embedding_bytes.len()
        );
        while !header.len().is_multiple_of(8) {
            header.push(' ');
        }
        let mut shard = Vec::new();
        shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
        shard.extend_from_slice(header.as_bytes());
        shard.extend_from_slice(&embedding_bytes);
        shard.extend_from_slice(&linear_bytes);
        shard.extend_from_slice(&norm_bytes);
        shard.extend_from_slice(&conv_bytes);
        shard.extend_from_slice(&dynamic_bytes);
        shard.extend_from_slice(&direct_norm_bytes);
        fs::write(model_dir.join("weights.safetensors"), shard).unwrap();
        fs::write(
            model_dir.join("model.safetensors.index.json"),
            format!(
                "{{\"metadata\":{{\"total_size\":{data_len}}},\"weight_map\":{{\"embedding\":\"weights.safetensors\",\"linear\":\"weights.safetensors\",\"norm\":\"weights.safetensors\",\"conv\":\"weights.safetensors\",\"dynamic\":\"weights.safetensors\",\"direct_norm\":\"weights.safetensors\"}}}}"
            ),
        )
        .unwrap();
        model_dir
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| {
                let bits = value.to_bits();
                let rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
                rounded.to_le_bytes()
            })
            .collect()
    }
}
