use backend::{Backend, DeviceValue};
use common::{DType, Error, Result};
use config::{LagunaConfig, LagunaMlpKind};

use super::{
    forward_attention, forward_dense_mlp_residual, forward_sparse_mlp_residual,
    LagunaAttentionCache, LagunaDeviceLayerMlpWeights, LagunaDeviceLayerWeights,
    LagunaDeviceRopeTables, LagunaExpertCache, LagunaExpertPrefetchPool, LagunaWeightIndex,
};

/// Executes one Laguna transformer layer while keeping hidden states on Metal.
///
/// Shapes remain `[B,T,3072]` through both residual branches:
///
/// 1. RMSNorm -> attention -> residual add.
/// 2. RMSNorm -> dense or top-10 sparse MLP -> residual add.
///
/// Dense layers can remain in one open Metal batch. Sparse layers synchronize
/// once to read router IDs, then overlap selected-expert SSD prefetch with the
/// shared expert and cache-hit kernels.
#[allow(clippy::too_many_arguments)]
pub fn forward_layer<B: Backend>(
    config: &LagunaConfig,
    hidden_states: &DeviceValue,
    weights: &LagunaDeviceLayerWeights,
    rope_tables: &LagunaDeviceRopeTables,
    cache: &mut LagunaAttentionCache,
    weight_index: &LagunaWeightIndex,
    expert_cache: &mut LagunaExpertCache,
    expert_prefetch_pool: &LagunaExpertPrefetchPool,
    backend: &B,
) -> Result<DeviceValue> {
    validate_layer_input(config, hidden_states, weights)?;

    let attention_input = required(
        "Laguna pre-attention RMSNorm",
        backend.rms_norm_device(
            hidden_states,
            &weights.input_norm,
            config.rms_norm_eps as f32,
        )?,
    )?;
    let attention_output = forward_attention(
        config,
        weights.layer_index,
        &attention_input,
        &weights.attention,
        rope_tables,
        cache,
        backend,
    )?;
    let post_attention = required(
        "Laguna attention residual add",
        backend.add_device(hidden_states, &attention_output)?,
    )?;
    let mlp_input = required(
        "Laguna post-attention RMSNorm",
        backend.rms_norm_device(
            &post_attention,
            &weights.post_attention_norm,
            config.rms_norm_eps as f32,
        )?,
    )?;

    match (config.mlp_kind(weights.layer_index), &weights.mlp) {
        (Some(LagunaMlpKind::Dense), LagunaDeviceLayerMlpWeights::Dense(dense)) => {
            forward_dense_mlp_residual(config, &mlp_input, &post_attention, dense, backend)
        }
        (Some(LagunaMlpKind::Sparse), LagunaDeviceLayerMlpWeights::Moe(moe)) => {
            forward_sparse_mlp_residual(
                config,
                weights.layer_index,
                &mlp_input,
                &post_attention,
                moe,
                weight_index,
                expert_cache,
                expert_prefetch_pool,
                backend,
            )
        }
        (Some(expected), actual) => Err(Error::weights(format!(
            "Laguna layer {} expects {expected:?} MLP weights, got {}",
            weights.layer_index,
            mlp_variant_name(actual)
        ))),
        (None, _) => Err(Error::config(format!(
            "Laguna layer {} is outside the configured MLP schedule",
            weights.layer_index
        ))),
    }
}

fn validate_layer_input(
    config: &LagunaConfig,
    hidden_states: &DeviceValue,
    weights: &LagunaDeviceLayerWeights,
) -> Result<()> {
    if weights.layer_index >= config.num_hidden_layers {
        return Err(Error::model(format!(
            "Laguna layer index {} must be below {}",
            weights.layer_index, config.num_hidden_layers
        )));
    }
    if hidden_states.dtype() != DType::F32 {
        return Err(Error::model(format!(
            "Laguna layer input must be F32, got {:?}",
            hidden_states.dtype()
        )));
    }
    let [batch, tokens, hidden] = hidden_states.dims() else {
        return Err(Error::model(format!(
            "Laguna layer input must be [B,T,{}], got {:?}",
            config.hidden_size,
            hidden_states.dims()
        )));
    };
    if *batch == 0 || *tokens == 0 || *hidden != config.hidden_size {
        return Err(Error::model(format!(
            "Laguna layer input must be non-empty [B,T,{}], got {:?}",
            config.hidden_size,
            hidden_states.dims()
        )));
    }
    for (label, norm) in [
        ("input norm", &weights.input_norm),
        ("post-attention norm", &weights.post_attention_norm),
    ] {
        if norm.dims() != [config.hidden_size] {
            return Err(Error::weights(format!(
                "Laguna layer {} {label} must be [{}], got {:?}",
                weights.layer_index,
                config.hidden_size,
                norm.dims()
            )));
        }
    }
    Ok(())
}

fn mlp_variant_name(weights: &LagunaDeviceLayerMlpWeights) -> &'static str {
    match weights {
        LagunaDeviceLayerMlpWeights::Dense(_) => "dense",
        LagunaDeviceLayerMlpWeights::Moe(_) => "sparse",
    }
}

fn required(label: &str, value: Option<DeviceValue>) -> Result<DeviceValue> {
    value.ok_or_else(|| Error::backend(format!("{label} requires native Metal")))
}
