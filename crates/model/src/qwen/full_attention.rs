use backend::{Backend, DeviceValue, QwenFullAttentionCache};
use common::{DType, Error, Result};

use super::QwenDeviceFullAttentionWeights;

const HIDDEN_SIZE: usize = 5_120;
const QUERY_GATE_WIDTH: usize = 12_288;
const KEY_VALUE_WIDTH: usize = 1_024;
const ATTENTION_OUTPUT_WIDTH: usize = 6_144;
const HEAD_DIM: usize = 256;

/// Runs one official Qwen3.8 full-attention token mixer and adds the residual.
/// All activations and K/V state remain in Metal buffers.
pub fn forward_full_attention_residual_device<B: Backend>(
    backend: &B,
    residual: &DeviceValue,
    normalized_input: &DeviceValue,
    weights: &QwenDeviceFullAttentionWeights,
    cache: &mut QwenFullAttentionCache,
    rope_theta: f32,
    rotary_dim: usize,
) -> Result<DeviceValue> {
    validate_inputs(residual, normalized_input, weights)?;
    let (query_gate, key, value) = backend
        .qwen_matrix_qkv_device(
            &weights.query,
            &weights.key,
            &weights.value,
            normalized_input,
        )?
        .ok_or_else(|| Error::backend("Qwen Q/K/V projection requires native Metal"))?;
    profile_boundary(backend, normalized_input, "qwen.full.qkv")?;
    let attention = backend
        .qwen_full_attention_device(
            &query_gate,
            &key,
            &value,
            &weights.query_norm,
            &weights.key_norm,
            cache,
            rope_theta,
            rotary_dim,
            false,
        )?
        .ok_or_else(|| Error::backend("Qwen full attention requires native Metal"))?;
    profile_boundary(backend, normalized_input, "qwen.full.core")?;
    if let Some(output) =
        backend.qwen_matrix_linear_add_device(&weights.output, &attention, residual)?
    {
        profile_boundary(backend, normalized_input, "qwen.full.output_residual")?;
        return Ok(output);
    }
    let projected = backend
        .qwen_matrix_linear_device(&weights.output, &attention)?
        .ok_or_else(|| Error::backend("Qwen attention output projection requires native Metal"))?;
    profile_boundary(backend, normalized_input, "qwen.full.output")?;
    let output = backend
        .qwen_bf16_add_device(residual, &projected)?
        .ok_or_else(|| Error::backend("Qwen attention residual add requires native Metal"))?;
    profile_boundary(backend, normalized_input, "qwen.full.residual")?;
    Ok(output)
}

fn profile_boundary<B: Backend>(backend: &B, input: &DeviceValue, stage: &str) -> Result<()> {
    if tracing::enabled!(
        target: "inferno::metal::profile",
        tracing::Level::TRACE
    ) {
        let rows = input.dims()[..input.dims().len() - 1]
            .iter()
            .product::<usize>();
        backend.device_profile_boundary(&format!("{stage}.rows{rows}"))?;
    }
    Ok(())
}

fn validate_inputs(
    residual: &DeviceValue,
    normalized_input: &DeviceValue,
    weights: &QwenDeviceFullAttentionWeights,
) -> Result<()> {
    validate_dimensions(
        residual.dims(),
        normalized_input.dims(),
        normalized_input.dtype(),
        [weights.query.rows(), weights.query.columns()],
        [weights.key.rows(), weights.key.columns()],
        [weights.value.rows(), weights.value.columns()],
        [weights.output.rows(), weights.output.columns()],
        weights.query_norm.shape(),
        weights.key_norm.shape(),
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_dimensions(
    residual_shape: &[usize],
    input_shape: &[usize],
    input_dtype: DType,
    query_shape: [usize; 2],
    key_shape: [usize; 2],
    value_shape: [usize; 2],
    output_shape: [usize; 2],
    query_norm_shape: &[usize],
    key_norm_shape: &[usize],
) -> Result<()> {
    if residual_shape != input_shape || input_shape.len() != 3 || input_shape[2] != HIDDEN_SIZE {
        return Err(Error::model(format!(
            "Qwen full attention requires matching [B,T,{HIDDEN_SIZE}] inputs, got {residual_shape:?} and {input_shape:?}"
        )));
    }
    if input_dtype != DType::BF16 {
        return Err(Error::model(format!(
            "Qwen full attention input must be BF16, got {input_dtype:?}"
        )));
    }
    let expected = [
        ("query/gate", query_shape, [QUERY_GATE_WIDTH, HIDDEN_SIZE]),
        ("key", key_shape, [KEY_VALUE_WIDTH, HIDDEN_SIZE]),
        ("value", value_shape, [KEY_VALUE_WIDTH, HIDDEN_SIZE]),
        (
            "output",
            output_shape,
            [HIDDEN_SIZE, ATTENTION_OUTPUT_WIDTH],
        ),
    ];
    for (label, actual, required) in expected {
        if actual != required {
            return Err(Error::model(format!(
                "Qwen {label} matrix must be {required:?}, got {actual:?}"
            )));
        }
    }
    if query_norm_shape != [HEAD_DIM] || key_norm_shape != [HEAD_DIM] {
        return Err(Error::model(format!(
            "Qwen Q/K norms must be [{HEAD_DIM}], got {query_norm_shape:?} and {key_norm_shape:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_official_full_attention_shapes() {
        validate_dimensions(
            &[1, 4, HIDDEN_SIZE],
            &[1, 4, HIDDEN_SIZE],
            DType::BF16,
            [QUERY_GATE_WIDTH, HIDDEN_SIZE],
            [KEY_VALUE_WIDTH, HIDDEN_SIZE],
            [KEY_VALUE_WIDTH, HIDDEN_SIZE],
            [HIDDEN_SIZE, ATTENTION_OUTPUT_WIDTH],
            &[HEAD_DIM],
            &[HEAD_DIM],
        )
        .unwrap();
    }

    #[test]
    fn rejects_a_non_gated_query_projection() {
        let error = validate_dimensions(
            &[1, 1, HIDDEN_SIZE],
            &[1, 1, HIDDEN_SIZE],
            DType::BF16,
            [ATTENTION_OUTPUT_WIDTH, HIDDEN_SIZE],
            [KEY_VALUE_WIDTH, HIDDEN_SIZE],
            [KEY_VALUE_WIDTH, HIDDEN_SIZE],
            [HIDDEN_SIZE, ATTENTION_OUTPUT_WIDTH],
            &[HEAD_DIM],
            &[HEAD_DIM],
        )
        .unwrap_err();

        assert!(error.to_string().contains("query/gate"));
    }
}
