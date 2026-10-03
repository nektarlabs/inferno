use backend::{Backend, DeviceValue, QwenLinearAttentionCache};
use common::{DType, Error, Result};

use super::QwenDeviceLinearAttentionWeights;

const HIDDEN_SIZE: usize = 5_120;
const QKV_WIDTH: usize = 10_240;
const VALUE_WIDTH: usize = 6_144;
const VALUE_HEADS: usize = 48;
const HEAD_DIM: usize = 128;

/// Runs one official Qwen3.8 Gated DeltaNet mixer and adds its residual.
pub fn forward_linear_attention_residual_device<B: Backend>(
    backend: &B,
    residual: &DeviceValue,
    normalized_input: &DeviceValue,
    weights: &QwenDeviceLinearAttentionWeights,
    cache: &mut QwenLinearAttentionCache,
    eps: f32,
) -> Result<DeviceValue> {
    validate_inputs(residual, normalized_input, weights)?;
    let mixed = backend
        .qwen_mlx_w4_linear_attention_device(
            normalized_input,
            &weights.qkv,
            &weights.gate,
            &weights.input_a,
            &weights.input_b,
            &weights.conv1d,
            &weights.a_log,
            &weights.dt_bias,
            &weights.norm,
            cache,
            eps,
        )?
        .ok_or_else(|| Error::backend("Qwen Gated DeltaNet requires native Metal"))?;
    profile_boundary(backend, normalized_input, "qwen.linear.mixer")?;
    if let Some(output) =
        backend.qwen_matrix_linear_add_device(&weights.output, &mixed, residual)?
    {
        profile_boundary(backend, normalized_input, "qwen.linear.output_residual")?;
        return Ok(output);
    }
    let projected = backend
        .qwen_matrix_linear_device(&weights.output, &mixed)?
        .ok_or_else(|| Error::backend("Qwen DeltaNet output projection requires native Metal"))?;
    profile_boundary(backend, normalized_input, "qwen.linear.output")?;
    let output = backend
        .qwen_bf16_add_device(residual, &projected)?
        .ok_or_else(|| Error::backend("Qwen DeltaNet residual add requires native Metal"))?;
    profile_boundary(backend, normalized_input, "qwen.linear.residual")?;
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
    weights: &QwenDeviceLinearAttentionWeights,
) -> Result<()> {
    let expected_conv_shape = [QKV_WIDTH, 4, 1];
    validate_dimensions(
        residual.dims(),
        normalized_input.dims(),
        normalized_input.dtype(),
        [weights.qkv.rows(), weights.qkv.columns()],
        [weights.gate.rows(), weights.gate.columns()],
        [weights.output.rows(), weights.output.columns()],
        &[weights.input_a.rows(), weights.input_a.columns()],
        &[weights.input_b.rows(), weights.input_b.columns()],
        weights.conv1d.shape(),
        &expected_conv_shape,
        weights.a_log.shape(),
        weights.dt_bias.shape(),
        weights.norm.shape(),
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_dimensions(
    residual_shape: &[usize],
    input_shape: &[usize],
    input_dtype: DType,
    qkv_shape: [usize; 2],
    gate_shape: [usize; 2],
    output_shape: [usize; 2],
    input_a_shape: &[usize],
    input_b_shape: &[usize],
    conv_shape: &[usize],
    expected_conv_shape: &[usize],
    a_log_shape: &[usize],
    dt_bias_shape: &[usize],
    norm_shape: &[usize],
) -> Result<()> {
    if residual_shape != input_shape || input_shape.len() != 3 || input_shape[2] != HIDDEN_SIZE {
        return Err(Error::model(format!(
            "Qwen linear attention requires matching [B,T,{HIDDEN_SIZE}] inputs, got {residual_shape:?} and {input_shape:?}"
        )));
    }
    if input_dtype != DType::BF16 {
        return Err(Error::model(format!(
            "Qwen linear attention input must be BF16, got {input_dtype:?}"
        )));
    }
    for (label, actual, expected) in [
        ("QKV", qkv_shape, [QKV_WIDTH, HIDDEN_SIZE]),
        ("gate", gate_shape, [VALUE_WIDTH, HIDDEN_SIZE]),
        ("output", output_shape, [HIDDEN_SIZE, VALUE_WIDTH]),
    ] {
        if actual != expected {
            return Err(Error::model(format!(
                "Qwen DeltaNet {label} matrix must be {expected:?}, got {actual:?}"
            )));
        }
    }
    for (label, actual, expected) in [
        ("input A", input_a_shape, &[VALUE_HEADS, HIDDEN_SIZE][..]),
        ("input B", input_b_shape, &[VALUE_HEADS, HIDDEN_SIZE][..]),
        ("conv1d", conv_shape, expected_conv_shape),
        ("A_log", a_log_shape, &[VALUE_HEADS][..]),
        ("dt_bias", dt_bias_shape, &[VALUE_HEADS][..]),
        ("norm", norm_shape, &[HEAD_DIM][..]),
    ] {
        if actual != expected {
            return Err(Error::model(format!(
                "Qwen DeltaNet {label} tensor must be {expected:?}, got {actual:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_official_linear_attention_shapes() {
        validate_dimensions(
            &[1, 4, HIDDEN_SIZE],
            &[1, 4, HIDDEN_SIZE],
            DType::BF16,
            [QKV_WIDTH, HIDDEN_SIZE],
            [VALUE_WIDTH, HIDDEN_SIZE],
            [HIDDEN_SIZE, VALUE_WIDTH],
            &[VALUE_HEADS, HIDDEN_SIZE],
            &[VALUE_HEADS, HIDDEN_SIZE],
            &[QKV_WIDTH, 1, 4],
            &[QKV_WIDTH, 1, 4],
            &[VALUE_HEADS],
            &[VALUE_HEADS],
            &[HEAD_DIM],
        )
        .unwrap();
    }

    #[test]
    fn rejects_full_attention_projection_shapes() {
        let error = validate_dimensions(
            &[1, 1, HIDDEN_SIZE],
            &[1, 1, HIDDEN_SIZE],
            DType::BF16,
            [12_288, HIDDEN_SIZE],
            [VALUE_WIDTH, HIDDEN_SIZE],
            [HIDDEN_SIZE, VALUE_WIDTH],
            &[VALUE_HEADS, HIDDEN_SIZE],
            &[VALUE_HEADS, HIDDEN_SIZE],
            &[QKV_WIDTH, 1, 4],
            &[QKV_WIDTH, 1, 4],
            &[VALUE_HEADS],
            &[VALUE_HEADS],
            &[HEAD_DIM],
        )
        .unwrap_err();

        assert!(error.to_string().contains("QKV"));
    }
}
