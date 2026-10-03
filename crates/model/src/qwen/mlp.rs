use backend::{Backend, DeviceQwenBf16Tensor, DeviceQwenMatrix, DeviceValue};
use common::{Error, Result};

use super::QwenDeviceMlpWeights;

/// Runs Qwen3.8's MLX W4 SwiGLU MLP while keeping every activation on
/// Metal. The returned tensor has the same shape as `input`.
pub fn forward_mlp_device<B: Backend>(
    backend: &B,
    input: &DeviceValue,
    weights: &QwenDeviceMlpWeights,
) -> Result<DeviceValue> {
    validate_mlp_shapes(input.dims(), &weights.gate, &weights.up, &weights.down)?;
    let activated = backend
        .qwen_matrix_gate_up_swiglu_device(&weights.gate, &weights.up, input)?
        .ok_or_else(|| Error::backend("Qwen SwiGLU requires native Metal"))?;
    profile_boundary(backend, input, "gate_up")?;
    let output = backend
        .qwen_matrix_linear_device(&weights.down, &activated)?
        .ok_or_else(|| Error::backend("Qwen down projection requires native Metal"))?;
    profile_boundary(backend, input, "down")?;
    Ok(output)
}

/// Runs the MLP and adds its output to the residual stream on Metal.
pub fn forward_mlp_residual_device<B: Backend>(
    backend: &B,
    residual: &DeviceValue,
    normalized_input: &DeviceValue,
    weights: &QwenDeviceMlpWeights,
) -> Result<DeviceValue> {
    if residual.dims() != normalized_input.dims() {
        return Err(Error::model(format!(
            "Qwen MLP residual shape {:?} does not match normalized input {:?}",
            residual.dims(),
            normalized_input.dims()
        )));
    }
    let projected = forward_mlp_device(backend, normalized_input, weights)?;
    let output = backend
        .qwen_bf16_add_device(residual, &projected)?
        .ok_or_else(|| Error::backend("Qwen BF16 residual add requires native Metal"))?;
    profile_boundary(backend, normalized_input, "residual")?;
    Ok(output)
}

/// Runs the post-attention RMSNorm and MLP without materializing the
/// normalized BF16 activation when the backend supports the fused Qwen path.
pub fn forward_mlp_residual_with_norm_device<B: Backend>(
    backend: &B,
    residual: &DeviceValue,
    norm_weight: &DeviceQwenBf16Tensor,
    eps: f32,
    weights: &QwenDeviceMlpWeights,
) -> Result<DeviceValue> {
    validate_mlp_shapes(residual.dims(), &weights.gate, &weights.up, &weights.down)?;
    let normalized = backend
        .qwen_bf16_rms_norm_standard_device(residual, norm_weight, eps)?
        .ok_or_else(|| Error::backend("Qwen post-attention RMSNorm requires native Metal"))?;
    let activated = backend
        .qwen_matrix_gate_up_swiglu_device(&weights.gate, &weights.up, &normalized)?
        .ok_or_else(|| Error::backend("Qwen SwiGLU requires native Metal"))?;
    profile_boundary(backend, residual, "gate_up")?;
    if let Some(output) =
        backend.qwen_matrix_linear_add_device(&weights.down, &activated, residual)?
    {
        profile_boundary(backend, residual, "down_residual")?;
        return Ok(output);
    }
    let projected = backend
        .qwen_matrix_linear_device(&weights.down, &activated)?
        .ok_or_else(|| Error::backend("Qwen down projection requires native Metal"))?;
    profile_boundary(backend, residual, "down")?;
    let output = backend
        .qwen_bf16_add_device(residual, &projected)?
        .ok_or_else(|| Error::backend("Qwen BF16 residual add requires native Metal"))?;
    profile_boundary(backend, residual, "residual")?;
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
        backend.device_profile_boundary(&format!("qwen.mlp.{stage}.rows{rows}"))?;
    }
    Ok(())
}

fn validate_mlp_shapes(
    input_shape: &[usize],
    gate: &DeviceQwenMatrix,
    up: &DeviceQwenMatrix,
    down: &DeviceQwenMatrix,
) -> Result<()> {
    validate_mlp_dimensions(
        input_shape,
        gate.rows(),
        gate.columns(),
        up.rows(),
        up.columns(),
        down.rows(),
        down.columns(),
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_mlp_dimensions(
    input_shape: &[usize],
    gate_rows: usize,
    gate_columns: usize,
    up_rows: usize,
    up_columns: usize,
    down_rows: usize,
    down_columns: usize,
) -> Result<()> {
    let Some(&hidden_size) = input_shape.last() else {
        return Err(Error::model("Qwen MLP input shape is empty"));
    };
    if input_shape.contains(&0) {
        return Err(Error::model(format!(
            "Qwen MLP input shape must be positive, got {input_shape:?}"
        )));
    }
    if gate_rows != up_rows || gate_columns != up_columns {
        return Err(Error::model(format!(
            "Qwen MLP gate/up shapes must match, got [{gate_rows},{gate_columns}] and [{up_rows},{up_columns}]"
        )));
    }
    if gate_columns != hidden_size || down_columns != gate_rows || down_rows != hidden_size {
        return Err(Error::model(format!(
            "Qwen MLP requires gate/up [{gate_rows},{hidden_size}] and down [{hidden_size},{gate_rows}], got gate [{gate_rows},{gate_columns}] and down [{down_rows},{down_columns}]"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_official_qwen_mlp_shapes() {
        validate_mlp_dimensions(&[1, 4, 5_120], 17_408, 5_120, 17_408, 5_120, 5_120, 17_408)
            .unwrap();
    }

    #[test]
    fn rejects_a_down_projection_with_the_wrong_axis_order() {
        let error =
            validate_mlp_dimensions(&[1, 4, 5_120], 17_408, 5_120, 17_408, 5_120, 17_408, 5_120)
                .unwrap_err();

        assert!(error.to_string().contains("down [5120,17408]"));
    }
}
