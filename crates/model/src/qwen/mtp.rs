use backend::{Backend, DeviceQwenTokenIds, DeviceValue, QwenFullAttentionCache};
use common::{DType, Error, Result};
use config::QwenConfig;

use super::{
    forward_full_attention_residual_device, forward_mlp_residual_device, QwenDeviceMtpWeights,
    QwenDeviceRootWeights,
};

const HIDDEN_SIZE: usize = 5_120;

pub struct QwenMtpState {
    attention: QwenFullAttentionCache,
}

impl QwenMtpState {
    pub fn create<B: Backend>(backend: &B, batch: usize, capacity_tokens: usize) -> Result<Self> {
        let attention = backend
            .create_qwen_full_attention_cache(batch, capacity_tokens)?
            .ok_or_else(|| Error::backend("Qwen MTP attention state requires native Metal"))?;
        Ok(Self { attention })
    }

    pub fn length(&self) -> usize {
        self.attention.length()
    }

    pub fn storage_bytes(&self) -> Result<usize> {
        self.attention.storage_bytes()
    }

    /// Drops speculative MTP rows while retaining the committed prefix.
    pub fn rewind_speculative(&mut self, rows: usize) -> Result<()> {
        if rows > 0 {
            self.attention.rewind(rows)?;
        }
        Ok(())
    }
}

/// Runs the official one-layer MTP head and returns post-`mtp.norm` BF16 rows.
///
/// `backbone_hidden` is the backbone's normalized hidden state at each committed
/// position. `next_token_ids` supplies the token immediately following each row.
/// The resulting shape is `[B,T,5120]` and remains resident on Metal.
pub fn forward_mtp_hidden_device<B: Backend>(
    backend: &B,
    config: &QwenConfig,
    root: &QwenDeviceRootWeights,
    weights: &QwenDeviceMtpWeights,
    state: &mut QwenMtpState,
    backbone_hidden: &DeviceValue,
    next_token_ids: &[u32],
) -> Result<DeviceValue> {
    let [batch, sequence_length] = validate_inputs(backbone_hidden, next_token_ids)?;
    if state.attention.batch() != batch {
        return Err(Error::cache(format!(
            "Qwen MTP batch {batch} does not match state batch {}",
            state.attention.batch()
        )));
    }

    let embeddings = backend
        .qwen_matrix_embedding_device(&root.embedding, next_token_ids, &[batch, sequence_length])?
        .ok_or_else(|| Error::backend("Qwen MTP embedding requires native Metal"))?;
    forward_with_embeddings(
        backend,
        config,
        weights,
        state,
        backbone_hidden,
        &embeddings,
        batch,
        sequence_length,
    )
}

/// Runs one chained MTP position without reading its token ID on the CPU.
pub fn forward_mtp_hidden_from_device_tokens<B: Backend>(
    backend: &B,
    config: &QwenConfig,
    root: &QwenDeviceRootWeights,
    weights: &QwenDeviceMtpWeights,
    state: &mut QwenMtpState,
    backbone_hidden: &DeviceValue,
    next_token_ids: &DeviceQwenTokenIds,
) -> Result<DeviceValue> {
    let [batch, sequence_length] = validate_hidden(backbone_hidden, next_token_ids.len())?;
    if state.attention.batch() != batch {
        return Err(Error::cache(format!(
            "Qwen MTP batch {batch} does not match state batch {}",
            state.attention.batch()
        )));
    }
    let embeddings = backend
        .qwen_matrix_embedding_from_device_tokens(
            &root.embedding,
            next_token_ids,
            &[batch, sequence_length],
        )?
        .ok_or_else(|| Error::backend("Qwen MTP device embedding requires native Metal"))?;
    forward_with_embeddings(
        backend,
        config,
        weights,
        state,
        backbone_hidden,
        &embeddings,
        batch,
        sequence_length,
    )
}

#[allow(clippy::too_many_arguments)]
fn forward_with_embeddings<B: Backend>(
    backend: &B,
    config: &QwenConfig,
    weights: &QwenDeviceMtpWeights,
    state: &mut QwenMtpState,
    backbone_hidden: &DeviceValue,
    embeddings: &DeviceValue,
    _batch: usize,
    _sequence_length: usize,
) -> Result<DeviceValue> {
    let normalized_embedding = backend
        .qwen_bf16_rms_norm_device(
            embeddings,
            &weights.pre_fc_embedding_norm,
            config.text_config.rms_norm_eps as f32,
        )?
        .ok_or_else(|| Error::backend("Qwen MTP embedding RMSNorm requires native Metal"))?;
    let normalized_hidden = backend
        .qwen_bf16_rms_norm_device(
            backbone_hidden,
            &weights.pre_fc_hidden_norm,
            config.text_config.rms_norm_eps as f32,
        )?
        .ok_or_else(|| Error::backend("Qwen MTP hidden RMSNorm requires native Metal"))?;
    let concatenated = backend
        .qwen_bf16_concat_last_device(&normalized_embedding, &normalized_hidden)?
        .ok_or_else(|| Error::backend("Qwen MTP fusion concat requires native Metal"))?;
    let fused = backend
        .qwen_matrix_linear_device(&weights.fusion, &concatenated)?
        .ok_or_else(|| Error::backend("Qwen MTP fusion projection requires native Metal"))?;

    let eps = config.text_config.rms_norm_eps as f32;
    let normalized = backend
        .qwen_bf16_rms_norm_device(&fused, &weights.layer.input_norm, eps)?
        .ok_or_else(|| Error::backend("Qwen MTP input RMSNorm requires native Metal"))?;
    let attention_output = forward_full_attention_residual_device(
        backend,
        &fused,
        &normalized,
        &weights.layer.attention,
        &mut state.attention,
        config.text_config.rope_parameters.rope_theta as f32,
        config.rotary_dim(),
    )?;
    let normalized = backend
        .qwen_bf16_rms_norm_device(&attention_output, &weights.layer.post_attention_norm, eps)?
        .ok_or_else(|| Error::backend("Qwen MTP post-attention RMSNorm requires native Metal"))?;
    let output =
        forward_mlp_residual_device(backend, &attention_output, &normalized, &weights.layer.mlp)?;
    backend
        .qwen_bf16_rms_norm_device(&output, &weights.final_norm, eps)?
        .ok_or_else(|| Error::backend("Qwen MTP final RMSNorm requires native Metal"))
}

fn validate_inputs(hidden: &DeviceValue, token_ids: &[u32]) -> Result<[usize; 2]> {
    validate_hidden(hidden, token_ids.len())
}

fn validate_hidden(hidden: &DeviceValue, token_count: usize) -> Result<[usize; 2]> {
    if hidden.dtype() != DType::BF16 || hidden.dims().len() != 3 {
        return Err(Error::model(format!(
            "Qwen MTP hidden states must be BF16 [B,T,{HIDDEN_SIZE}], got {:?} {:?}",
            hidden.dtype(),
            hidden.dims()
        )));
    }
    let [batch, sequence_length, hidden_size]: [usize; 3] = hidden
        .dims()
        .try_into()
        .map_err(|_| Error::model("Qwen MTP hidden shape must have rank 3"))?;
    if batch == 0 || sequence_length == 0 || hidden_size != HIDDEN_SIZE {
        return Err(Error::model(format!(
            "Qwen MTP hidden states must be [B,T,{HIDDEN_SIZE}], got {:?}",
            hidden.dims()
        )));
    }
    let expected = batch
        .checked_mul(sequence_length)
        .ok_or_else(|| Error::model("Qwen MTP token count overflow"))?;
    if token_count != expected {
        return Err(Error::model(format!(
            "Qwen MTP hidden shape {:?} requires {expected} next-token IDs, got {}",
            hidden.dims(),
            token_count
        )));
    }
    Ok([batch, sequence_length])
}
