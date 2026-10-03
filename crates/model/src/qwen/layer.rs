use backend::{Backend, DeviceValue, QwenFullAttentionCache, QwenLinearAttentionCache};
use common::{Error, Result};
use config::{QwenConfig, QwenLayerKind};

use super::{
    forward_full_attention_residual_device, forward_linear_attention_residual_device,
    forward_mlp_residual_with_norm_device, QwenDeviceAttentionWeights, QwenDeviceLayerWeights,
};

pub enum QwenLayerState {
    Linear(QwenLinearAttentionCache),
    Full(QwenFullAttentionCache),
}

impl QwenLayerState {
    pub fn processed_tokens(&self) -> usize {
        match self {
            Self::Linear(state) => state.processed_tokens(),
            Self::Full(state) => state.length(),
        }
    }

    pub fn storage_bytes(&self) -> Result<usize> {
        match self {
            Self::Linear(state) => state.storage_bytes(),
            Self::Full(state) => state.storage_bytes(),
        }
    }
}

pub struct QwenModelState {
    batch: usize,
    capacity_tokens: usize,
    layers: Vec<QwenLayerState>,
}

impl QwenModelState {
    /// Starts an independent sequence without reallocating device storage.
    pub fn reset<B: Backend>(&mut self, backend: &B) -> Result<()> {
        for layer in &mut self.layers {
            match layer {
                QwenLayerState::Linear(cache) => {
                    backend.reset_qwen_linear_attention_cache(cache)?
                }
                QwenLayerState::Full(cache) => cache.reset(),
            }
        }
        Ok(())
    }

    pub fn create<B: Backend>(
        backend: &B,
        config: &QwenConfig,
        batch: usize,
        capacity_tokens: usize,
    ) -> Result<Self> {
        Self::create_internal(backend, config, batch, capacity_tokens, false)
    }

    pub fn create_speculative<B: Backend>(
        backend: &B,
        config: &QwenConfig,
        batch: usize,
        capacity_tokens: usize,
    ) -> Result<Self> {
        Self::create_internal(backend, config, batch, capacity_tokens, true)
    }

    fn create_internal<B: Backend>(
        backend: &B,
        config: &QwenConfig,
        batch: usize,
        capacity_tokens: usize,
        speculative: bool,
    ) -> Result<Self> {
        if batch == 0 || capacity_tokens == 0 {
            return Err(Error::cache(format!(
                "Qwen state dimensions must be positive, got batch={batch}, capacity={capacity_tokens}"
            )));
        }
        let layers = config
            .text_config
            .layer_types
            .iter()
            .map(|kind| match kind {
                QwenLayerKind::LinearAttention => (if speculative {
                    backend.create_qwen_speculative_linear_attention_cache(batch)
                } else {
                    backend.create_qwen_linear_attention_cache(batch)
                })?
                .map(QwenLayerState::Linear)
                .ok_or_else(|| Error::backend("Qwen DeltaNet state requires native Metal")),
                QwenLayerKind::FullAttention => backend
                    .create_qwen_full_attention_cache(batch, capacity_tokens)?
                    .map(QwenLayerState::Full)
                    .ok_or_else(|| {
                        Error::backend("Qwen full-attention state requires native Metal")
                    }),
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            batch,
            capacity_tokens,
            layers,
        })
    }

    /// Keeps `primary + accepted_drafts` from the most recent verify block.
    pub fn restore_speculative_prefix<B: Backend>(
        &mut self,
        backend: &B,
        accepted_drafts: usize,
        draft_count: usize,
    ) -> Result<()> {
        if accepted_drafts >= draft_count || draft_count == 0 {
            return Err(Error::cache(format!(
                "invalid Qwen speculative rollback accepted={accepted_drafts}, drafts={draft_count}"
            )));
        }
        let rejected_rows = draft_count - accepted_drafts;
        for layer in &mut self.layers {
            match layer {
                QwenLayerState::Linear(cache) => {
                    if !backend.restore_qwen_linear_attention_checkpoint(
                        cache,
                        accepted_drafts,
                        rejected_rows,
                    )? {
                        return Err(Error::backend(
                            "Qwen speculative rollback requires native Metal",
                        ));
                    }
                }
                QwenLayerState::Full(cache) => cache.rewind(rejected_rows)?,
            }
        }
        Ok(())
    }

    pub fn accept_speculative_rows(&mut self) {
        for layer in &mut self.layers {
            if let QwenLayerState::Linear(cache) = layer {
                cache.clear_checkpoint();
            }
        }
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn capacity_tokens(&self) -> usize {
        self.capacity_tokens
    }

    pub fn layers(&self) -> &[QwenLayerState] {
        &self.layers
    }

    pub fn storage_bytes(&self) -> Result<usize> {
        self.layers.iter().try_fold(0_usize, |total, layer| {
            total
                .checked_add(layer.storage_bytes()?)
                .ok_or_else(|| Error::cache("Qwen model state size overflow"))
        })
    }

    pub(crate) fn layer_mut(&mut self, layer_index: usize) -> Result<&mut QwenLayerState> {
        self.layers
            .get_mut(layer_index)
            .ok_or_else(|| Error::cache(format!("Qwen state has no layer {layer_index}")))
    }
}

pub fn forward_layer_device<B: Backend>(
    backend: &B,
    config: &QwenConfig,
    hidden_states: &DeviceValue,
    weights: &QwenDeviceLayerWeights,
    state: &mut QwenLayerState,
) -> Result<DeviceValue> {
    let eps = config.text_config.rms_norm_eps as f32;
    let attention_output = match (&weights.attention, state) {
        (QwenDeviceAttentionWeights::Linear(attention_weights), QwenLayerState::Linear(state)) => {
            let normalized = backend
                .qwen_bf16_rms_norm_standard_device(hidden_states, &weights.input_norm, eps)?
                .ok_or_else(|| Error::backend("Qwen input RMSNorm requires native Metal"))?;
            profile_boundary(backend, weights.layer_index, "input_norm")?;
            forward_linear_attention_residual_device(
                backend,
                hidden_states,
                &normalized,
                attention_weights,
                state,
                eps,
            )?
        }
        (QwenDeviceAttentionWeights::Full(attention_weights), QwenLayerState::Full(state)) => {
            let normalized = backend
                .qwen_bf16_rms_norm_standard_device(hidden_states, &weights.input_norm, eps)?
                .ok_or_else(|| Error::backend("Qwen input RMSNorm requires native Metal"))?;
            profile_boundary(backend, weights.layer_index, "input_norm")?;
            forward_full_attention_residual_device(
                backend,
                hidden_states,
                &normalized,
                attention_weights,
                state,
                config.text_config.rope_parameters.rope_theta as f32,
                config.rotary_dim(),
            )?
        }
        (QwenDeviceAttentionWeights::Linear(_), QwenLayerState::Full(_)) => {
            return Err(Error::cache(format!(
                "Qwen layer {} requires DeltaNet state, got full-attention KV state",
                weights.layer_index
            )));
        }
        (QwenDeviceAttentionWeights::Full(_), QwenLayerState::Linear(_)) => {
            return Err(Error::cache(format!(
                "Qwen layer {} requires full-attention KV state, got DeltaNet state",
                weights.layer_index
            )));
        }
    };
    profile_boundary(backend, weights.layer_index, "attention")?;
    let output = forward_mlp_residual_with_norm_device(
        backend,
        &attention_output,
        &weights.post_attention_norm,
        eps,
        &weights.mlp,
    )?;
    profile_boundary(backend, weights.layer_index, "post_norm")?;
    profile_boundary(backend, weights.layer_index, "mlp")?;
    Ok(output)
}

fn profile_boundary<B: Backend>(backend: &B, layer_index: usize, stage: &str) -> Result<()> {
    if tracing::enabled!(
        target: "inferno::metal::profile",
        tracing::Level::TRACE
    ) {
        backend.device_profile_boundary(&format!("qwen.layer.{layer_index}.{stage}"))?;
    }
    Ok(())
}
