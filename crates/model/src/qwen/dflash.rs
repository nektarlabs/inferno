use backend::{Backend, DFlashAttentionCache, DeviceQwenTokenIds, DeviceValue};
use common::{DType, Error, Result};
use config::DFlashConfig;

use super::{
    DFlashDeviceAttentionWeights, DFlashDeviceConvWeights, DFlashDeviceLayerWeights,
    DFlashDeviceMlpWeights, DFlashDeviceWeights, QwenDeviceRootWeights,
};

const HIDDEN_SIZE: usize = 5_120;
const TARGET_FEATURE_WIDTH: usize = 25_600;
const MAX_BLOCK_SIZE: usize = 8;

pub struct DFlashState {
    layers: Vec<DFlashAttentionCache>,
}

impl DFlashState {
    pub fn create<B: Backend>(backend: &B, config: &DFlashConfig, batch: usize) -> Result<Self> {
        let capacity = config
            .sliding_window
            .checked_sub(1)
            .ok_or_else(|| Error::cache("DFlash2 sliding window must exceed one token"))?;
        let layers = (0..config.num_hidden_layers)
            .map(|_| {
                backend
                    .create_dflash_attention_cache(batch, capacity)?
                    .ok_or_else(|| Error::backend("DFlash2 attention cache requires native Metal"))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    pub fn storage_bytes(&self) -> Result<usize> {
        self.layers.iter().try_fold(0_usize, |total, cache| {
            total
                .checked_add(cache.storage_bytes()?)
                .ok_or_else(|| Error::cache("DFlash2 state size overflow"))
        })
    }
}

/// Produces one DFlash2 block while leaving all activations and selected token
/// IDs on Metal. `block_tokens[0]` is the accepted anchor token and the
/// remaining positions are the DFlash mask token.
#[allow(clippy::too_many_arguments)]
pub fn forward_dflash_proposals_device<B: Backend>(
    backend: &B,
    config: &DFlashConfig,
    target_root: &QwenDeviceRootWeights,
    weights: &DFlashDeviceWeights,
    state: &mut DFlashState,
    target_features: &DeviceValue,
    context_position_start: usize,
    block_tokens: &[u32],
) -> Result<DeviceQwenTokenIds> {
    validate_inputs(config, weights, state, target_features, block_tokens)?;
    let [batch, context_length, _]: [usize; 3] = target_features
        .dims()
        .try_into()
        .map_err(|_| Error::model("DFlash2 target features must have rank 3"))?;
    let block_length = block_tokens.len();
    let eps = config.rms_norm_eps as f32;

    let fused_context = backend
        .dflash_w4_linear_device(&weights.target_fusion, target_features)?
        .ok_or_else(|| Error::backend("DFlash2 target fusion requires native Metal"))?;
    let fused_context = backend
        .dflash_rms_norm_device(&fused_context, &weights.target_hidden_norm, eps)?
        .ok_or_else(|| Error::backend("DFlash2 target feature RMSNorm requires native Metal"))?;
    profile_boundary(backend, "dflash.target_fusion")?;
    let mut hidden = backend
        .qwen_matrix_embedding_device(&target_root.embedding, block_tokens, &[batch, block_length])?
        .ok_or_else(|| Error::backend("DFlash2 noise embedding requires native Metal"))?;

    for (layer_index, layer) in weights.layers.iter().enumerate() {
        if layer.layer_index != layer_index {
            return Err(Error::weights(format!(
                "DFlash2 layer slot {layer_index} contains layer {}",
                layer.layer_index
            )));
        }
        let cache = state
            .layers
            .get_mut(layer_index)
            .ok_or_else(|| Error::cache(format!("DFlash2 state has no layer {layer_index}")))?;
        hidden = forward_layer(
            backend,
            config,
            &hidden,
            &fused_context,
            context_length,
            context_position_start,
            layer,
            cache,
            eps,
        )?;
        profile_boundary(backend, &format!("dflash.layer.{layer_index}"))?;
        backend.device_submit()?;
    }

    let proposal_rows = block_length - 1;
    let selected_hidden = backend
        .dflash_rms_norm_suffix_device(&hidden, &weights.final_norm, eps, proposal_rows)?
        .ok_or_else(|| Error::backend("DFlash2 suffix RMSNorm requires native Metal"))?;
    let projected_hidden = backend
        .dflash_w4_linear_device(&weights.selector_hidden_projection, &selected_hidden)?
        .ok_or_else(|| Error::backend("DFlash2 selector projection requires native Metal"))?;
    let selected = backend
        .dflash_select_candidates_for_output_device(
            &target_root.output,
            &selected_hidden,
            &projected_hidden,
            &weights.predecessor_codebook,
            &weights.successor_codebook,
            block_tokens[0],
            config.dflash_config.selector_top_k,
        )?
        .ok_or_else(|| Error::backend("DFlash2 path selector requires native Metal"))?;
    profile_boundary(backend, "dflash.selector")?;
    Ok(selected)
}

fn profile_boundary<B: Backend>(backend: &B, stage: &str) -> Result<()> {
    if tracing::enabled!(
        target: "inferno::metal::profile",
        tracing::Level::TRACE
    ) {
        backend.device_profile_boundary(stage)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn forward_layer<B: Backend>(
    backend: &B,
    config: &DFlashConfig,
    residual: &DeviceValue,
    context: &DeviceValue,
    _context_length: usize,
    context_position_start: usize,
    weights: &DFlashDeviceLayerWeights,
    cache: &mut DFlashAttentionCache,
    eps: f32,
) -> Result<DeviceValue> {
    let normalized = backend
        .dflash_rms_norm_device(residual, &weights.input_norm, eps)?
        .ok_or_else(|| Error::backend("DFlash2 input RMSNorm requires native Metal"))?;
    let (attention_input, attention_dynamic) =
        prepare_dynamic_conv(backend, config, &normalized, &weights.attention_conv)?;
    let attention = forward_attention(
        backend,
        config,
        &attention_input,
        context,
        context_position_start,
        &weights.attention,
        cache,
    )?;
    let attention_residual = finish_dynamic_conv_residual(
        backend,
        config,
        &attention,
        &attention_dynamic,
        &weights.attention_conv,
        residual,
    )?;

    let normalized = backend
        .dflash_rms_norm_device(&attention_residual, &weights.post_attention_norm, eps)?
        .ok_or_else(|| Error::backend("DFlash2 post-attention RMSNorm requires native Metal"))?;
    let (mlp_input, mlp_dynamic) =
        prepare_dynamic_conv(backend, config, &normalized, &weights.mlp_conv)?;
    let mlp = forward_mlp(backend, &mlp_input, &weights.mlp)?;
    finish_dynamic_conv_residual(
        backend,
        config,
        &mlp,
        &mlp_dynamic,
        &weights.mlp_conv,
        &attention_residual,
    )
}

fn prepare_dynamic_conv<B: Backend>(
    backend: &B,
    config: &DFlashConfig,
    input: &DeviceValue,
    weights: &DFlashDeviceConvWeights,
) -> Result<(DeviceValue, DeviceValue)> {
    let dynamic = backend
        .dflash_w4_linear_device(&weights.kernel_projection, input)?
        .ok_or_else(|| Error::backend("DFlash2 dynamic kernel projection requires native Metal"))?;
    let convolved = backend
        .dflash_dynamic_conv_device(
            input,
            &dynamic,
            &weights.base_kernel,
            0,
            config.dflash_config.conv_kernel_size,
            config.dflash_config.conv_group_size,
        )?
        .ok_or_else(|| Error::backend("DFlash2 input convolution requires native Metal"))?;
    Ok((convolved, dynamic))
}

fn finish_dynamic_conv_residual<B: Backend>(
    backend: &B,
    config: &DFlashConfig,
    input: &DeviceValue,
    dynamic: &DeviceValue,
    weights: &DFlashDeviceConvWeights,
    residual: &DeviceValue,
) -> Result<DeviceValue> {
    backend
        .dflash_dynamic_conv_residual_device(
            input,
            dynamic,
            &weights.base_kernel,
            residual,
            config.dflash_config.conv_kernel_size,
            config.dflash_config.conv_group_size,
        )?
        .ok_or_else(|| Error::backend("DFlash2 fused output convolution requires native Metal"))
}

#[allow(clippy::too_many_arguments)]
fn forward_attention<B: Backend>(
    backend: &B,
    config: &DFlashConfig,
    input: &DeviceValue,
    context: &DeviceValue,
    context_position_start: usize,
    weights: &DFlashDeviceAttentionWeights,
    cache: &mut DFlashAttentionCache,
) -> Result<DeviceValue> {
    let query = backend
        .dflash_w4_linear_device(&weights.query, input)?
        .ok_or_else(|| Error::backend("DFlash2 query projection requires native Metal"))?;
    let context_key = backend
        .dflash_w4_linear_device(&weights.key, context)?
        .ok_or_else(|| Error::backend("DFlash2 context key projection requires native Metal"))?;
    let context_value = backend
        .dflash_w4_linear_device(&weights.value, context)?
        .ok_or_else(|| Error::backend("DFlash2 context value projection requires native Metal"))?;
    let proposal_key = backend
        .dflash_w4_linear_device(&weights.key, input)?
        .ok_or_else(|| Error::backend("DFlash2 proposal key projection requires native Metal"))?;
    let proposal_value = backend
        .dflash_w4_linear_device(&weights.value, input)?
        .ok_or_else(|| Error::backend("DFlash2 proposal value projection requires native Metal"))?;
    let attention = backend
        .dflash_attention_device(
            &query,
            &context_key,
            &context_value,
            &proposal_key,
            &proposal_value,
            &weights.query_norm,
            &weights.key_norm,
            cache,
            context_position_start,
            config.rope_parameters.rope_theta as f32,
            config.sliding_window,
        )?
        .ok_or_else(|| Error::backend("DFlash2 attention requires native Metal"))?;
    backend
        .dflash_w4_linear_device(&weights.output, &attention)?
        .ok_or_else(|| Error::backend("DFlash2 attention output requires native Metal"))
}

fn forward_mlp<B: Backend>(
    backend: &B,
    input: &DeviceValue,
    weights: &DFlashDeviceMlpWeights,
) -> Result<DeviceValue> {
    let activated = backend
        .dflash_w4_gate_up_swiglu_device(&weights.gate, &weights.up, input)?
        .ok_or_else(|| Error::backend("DFlash2 SwiGLU requires native Metal"))?;
    backend
        .dflash_w4_linear_device(&weights.down, &activated)?
        .ok_or_else(|| Error::backend("DFlash2 down projection requires native Metal"))
}

fn validate_inputs(
    config: &DFlashConfig,
    weights: &DFlashDeviceWeights,
    state: &DFlashState,
    target_features: &DeviceValue,
    block_tokens: &[u32],
) -> Result<()> {
    if target_features.dtype() != DType::BF16
        || target_features.dims().len() != 3
        || target_features.dims()[0] != 1
        || target_features.dims()[1] == 0
        || target_features.dims()[2] != TARGET_FEATURE_WIDTH
    {
        return Err(Error::model(format!(
            "DFlash2 target features must be BF16 [1,S,{TARGET_FEATURE_WIDTH}], got {:?} {:?}",
            target_features.dtype(),
            target_features.dims()
        )));
    }
    if block_tokens.len() < 2 || block_tokens.len() > MAX_BLOCK_SIZE {
        return Err(Error::model(format!(
            "DFlash2 block length must be in 2..={MAX_BLOCK_SIZE}, got {}",
            block_tokens.len()
        )));
    }
    if block_tokens[1..]
        .iter()
        .any(|token| *token != config.dflash_config.mask_token_id)
    {
        return Err(Error::model(
            "DFlash2 proposal positions must contain the configured mask token",
        ));
    }
    if weights.layers.len() != config.num_hidden_layers
        || state.layers.len() != config.num_hidden_layers
    {
        return Err(Error::model(format!(
            "DFlash2 requires {} layers, got {} weight layers and {} state layers",
            config.num_hidden_layers,
            weights.layers.len(),
            state.layers.len()
        )));
    }
    if config.hidden_size != HIDDEN_SIZE || config.target_feature_width()? != TARGET_FEATURE_WIDTH {
        return Err(Error::config("unsupported DFlash2 hidden dimensions"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_contract_requires_anchor_followed_by_masks() {
        let mask = 248_070;
        let valid = [7, mask, mask];
        assert_eq!(valid[1..].iter().filter(|token| **token != mask).count(), 0);
        assert!(valid.len() <= MAX_BLOCK_SIZE);
    }
}
