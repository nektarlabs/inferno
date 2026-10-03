use backend::{Backend, DeviceQwenTokenIds, DeviceValue};
use common::{Error, Result};
use config::{DFlashConfig, QwenConfig};

use super::{forward_layer_device, QwenDeviceWeights, QwenModelState};

const HIDDEN_SIZE: usize = 5_120;

/// Runs the official text embedding, all 64 decoder layers, and final norm.
pub fn forward_hidden_device<B: Backend>(
    backend: &B,
    config: &QwenConfig,
    weights: &QwenDeviceWeights,
    state: &mut QwenModelState,
    token_ids: &[u32],
    token_shape: &[usize],
) -> Result<DeviceValue> {
    let (hidden_states, captures) = forward_hidden_impl::<false, B>(
        backend,
        config,
        weights,
        state,
        token_ids,
        token_shape,
        &[],
        0,
    )?;
    debug_assert!(captures.is_empty());
    Ok(hidden_states)
}

/// Runs the target Qwen model and captures the five intermediate layer
/// outputs required by DFlash2. This is a separate entrypoint so the default
/// MTP path neither retains intermediate buffers nor packs target features.
pub fn forward_hidden_with_dflash_features_device<B: Backend>(
    backend: &B,
    config: &QwenConfig,
    dflash_config: &DFlashConfig,
    weights: &QwenDeviceWeights,
    state: &mut QwenModelState,
    token_ids: &[u32],
    token_shape: &[usize],
) -> Result<(DeviceValue, DeviceValue)> {
    let target_layer_ids = &dflash_config.dflash_config.target_layer_ids;
    if dflash_config.num_target_layers != config.text_config.num_hidden_layers
        || dflash_config.hidden_size != config.text_config.hidden_size
    {
        return Err(Error::config(
            "DFlash2 target-layer contract does not match the loaded Qwen model",
        ));
    }
    let (hidden_states, captures) = forward_hidden_impl::<true, B>(
        backend,
        config,
        weights,
        state,
        token_ids,
        token_shape,
        target_layer_ids,
        dflash_config.sliding_window - 1,
    )?;
    if captures.len() != target_layer_ids.len() {
        return Err(Error::model(format!(
            "DFlash2 expected {} target features, captured {}",
            target_layer_ids.len(),
            captures.len()
        )));
    }
    let target_features = backend
        .dflash_pack_target_features_device(&captures)?
        .ok_or_else(|| Error::backend("DFlash2 target feature packing requires native Metal"))?;
    Ok((hidden_states, target_features))
}

#[allow(clippy::too_many_arguments)]
fn forward_hidden_impl<const CAPTURE_DFLASH_FEATURES: bool, B: Backend>(
    backend: &B,
    config: &QwenConfig,
    weights: &QwenDeviceWeights,
    state: &mut QwenModelState,
    token_ids: &[u32],
    token_shape: &[usize],
    target_layer_ids: &[usize],
    capture_max_rows: usize,
) -> Result<(DeviceValue, Vec<DeviceValue>)> {
    validate_forward(config, weights, state, token_ids, token_shape)?;
    let mut hidden_states = backend
        .qwen_matrix_embedding_device(&weights.root.embedding, token_ids, token_shape)?
        .ok_or_else(|| Error::backend("Qwen embedding requires native Metal"))?;
    if hidden_states.dims() != [token_shape[0], token_shape[1], HIDDEN_SIZE] {
        return Err(Error::model(format!(
            "Qwen embedding returned shape {:?}",
            hidden_states.dims()
        )));
    }
    let mut captures = if CAPTURE_DFLASH_FEATURES {
        Vec::with_capacity(target_layer_ids.len())
    } else {
        Vec::new()
    };
    for (layer_index, layer_weights) in weights.layers.iter().enumerate() {
        if layer_weights.layer_index != layer_index {
            return Err(Error::weights(format!(
                "Qwen layer slot {layer_index} contains weights for layer {}",
                layer_weights.layer_index
            )));
        }
        hidden_states = forward_layer_device(
            backend,
            config,
            &hidden_states,
            layer_weights,
            state.layer_mut(layer_index)?,
        )?;
        if CAPTURE_DFLASH_FEATURES && target_layer_ids.contains(&layer_index) {
            let capture = backend
                .dflash_bf16_suffix_rows_device(&hidden_states, capture_max_rows)?
                .ok_or_else(|| {
                    Error::backend("DFlash2 target feature capture requires native Metal")
                })?;
            captures.push(capture);
        }
        backend.device_submit()?;
    }

    let output = backend
        .qwen_bf16_rms_norm_standard_device(
            &hidden_states,
            &weights.root.final_norm,
            config.text_config.rms_norm_eps as f32,
        )?
        .ok_or_else(|| Error::backend("Qwen final RMSNorm requires native Metal"))?;
    if tracing::enabled!(
        target: "inferno::metal::profile",
        tracing::Level::TRACE
    ) {
        backend.device_profile_boundary(&format!(
            "qwen.final_norm.rows{}",
            token_shape[0] * token_shape[1]
        ))?;
    }
    Ok((output, captures))
}

pub fn greedy_next_tokens_device<B: Backend>(
    backend: &B,
    weights: &QwenDeviceWeights,
    hidden_states: &DeviceValue,
) -> Result<Vec<u32>> {
    backend
        .qwen_matrix_last_token_argmax(&weights.root.output, hidden_states)?
        .ok_or_else(|| Error::backend("Qwen greedy output head requires native Metal"))
}

pub fn draft_next_token_device_handle<B: Backend>(
    backend: &B,
    weights: &QwenDeviceWeights,
    hidden_states: &DeviceValue,
) -> Result<DeviceQwenTokenIds> {
    backend
        .qwen_matrix_compact_draft_argmax_device(&weights.root.output, hidden_states)?
        .ok_or_else(|| Error::backend("Qwen compact draft output requires native Metal"))
}

pub fn greedy_all_tokens_device<B: Backend>(
    backend: &B,
    weights: &QwenDeviceWeights,
    hidden_states: &DeviceValue,
) -> Result<Vec<u32>> {
    backend
        .qwen_matrix_all_token_argmax(&weights.root.output, hidden_states)?
        .ok_or_else(|| Error::backend("Qwen all-row greedy output requires native Metal"))
}

pub fn forward_greedy_device<B: Backend>(
    backend: &B,
    config: &QwenConfig,
    weights: &QwenDeviceWeights,
    state: &mut QwenModelState,
    token_ids: &[u32],
    token_shape: &[usize],
) -> Result<Vec<u32>> {
    let hidden_states =
        forward_hidden_device(backend, config, weights, state, token_ids, token_shape)?;
    greedy_next_tokens_device(backend, weights, &hidden_states)
}

fn validate_forward(
    config: &QwenConfig,
    weights: &QwenDeviceWeights,
    state: &QwenModelState,
    token_ids: &[u32],
    token_shape: &[usize],
) -> Result<()> {
    let [batch, sequence_length] = validate_token_shape(token_shape, token_ids.len())?;
    if batch != state.batch() {
        return Err(Error::cache(format!(
            "Qwen token batch {} does not match state batch {}",
            batch,
            state.batch()
        )));
    }
    if sequence_length > state.capacity_tokens() {
        return Err(Error::cache(format!(
            "Qwen input length {} exceeds state capacity {}",
            sequence_length,
            state.capacity_tokens()
        )));
    }
    if weights.layers.len() != config.text_config.num_hidden_layers
        || state.layers().len() != config.text_config.num_hidden_layers
    {
        return Err(Error::model(format!(
            "Qwen forward requires {} layers, got {} weight layers and {} state layers",
            config.text_config.num_hidden_layers,
            weights.layers.len(),
            state.layers().len()
        )));
    }
    Ok(())
}

fn validate_token_shape(shape: &[usize], id_count: usize) -> Result<[usize; 2]> {
    let [batch, sequence_length]: [usize; 2] = shape.try_into().map_err(|_| {
        Error::model(format!(
            "Qwen token shape must be positive [B,T], got {shape:?}"
        ))
    })?;
    if batch == 0 || sequence_length == 0 {
        return Err(Error::model(format!(
            "Qwen token shape must be positive [B,T], got {shape:?}"
        )));
    }
    let expected = batch
        .checked_mul(sequence_length)
        .ok_or_else(|| Error::model("Qwen token count overflow"))?;
    if expected != id_count {
        return Err(Error::model(format!(
            "Qwen token shape {shape:?} requires {expected} IDs, got {id_count}"
        )));
    }
    Ok([batch, sequence_length])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_matrix_token_shapes_before_device_execution() {
        assert!(validate_token_shape(&[4], 4).is_err());
        assert!(validate_token_shape(&[1, 4], 3).is_err());
        validate_token_shape(&[1, 4], 4).unwrap();
    }
}
