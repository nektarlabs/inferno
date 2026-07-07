use common::{Error, Result};

pub(crate) const Q2_K_BLOCK_VALUES: usize = 256;
pub(crate) const Q2_K_BLOCK_BYTES: usize = 84;
pub(crate) const Q8_0_BLOCK_VALUES: usize = 32;
pub(crate) const Q8_0_BLOCK_BYTES: usize = 34;

pub(crate) fn validate_matmul_f32(
    lhs: &[f32],
    rhs: &[f32],
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<()> {
    if rows == 0 {
        return Err(Error::backend("matmul rows must be greater than zero"));
    }
    if inner == 0 {
        return Err(Error::backend("matmul inner must be greater than zero"));
    }
    if cols == 0 {
        return Err(Error::backend("matmul cols must be greater than zero"));
    }

    let expected_lhs_len = rows
        .checked_mul(inner)
        .ok_or_else(|| Error::backend("matmul lhs length overflow"))?;
    if lhs.len() != expected_lhs_len {
        return Err(Error::backend(format!(
            "matmul lhs shape mismatch: expected [{rows}, {inner}] = {expected_lhs_len} values, got {}",
            lhs.len()
        )));
    }

    let expected_rhs_len = inner
        .checked_mul(cols)
        .ok_or_else(|| Error::backend("matmul rhs length overflow"))?;
    if rhs.len() != expected_rhs_len {
        return Err(Error::backend(format!(
            "matmul rhs shape mismatch: expected [{inner}, {cols}] = {expected_rhs_len} values, got {}",
            rhs.len()
        )));
    }
    debug_assert_finite_values("matmul lhs", lhs);
    debug_assert_finite_values("matmul rhs", rhs);

    Ok(())
}

pub(crate) fn validate_linear_f32(
    input: &[f32],
    weight: &[f32],
    rows: usize,
    in_features: usize,
    out_features: usize,
) -> Result<()> {
    if rows == 0 {
        return Err(Error::backend("linear rows must be greater than zero"));
    }
    if in_features == 0 {
        return Err(Error::backend(
            "linear in_features must be greater than zero",
        ));
    }
    if out_features == 0 {
        return Err(Error::backend(
            "linear out_features must be greater than zero",
        ));
    }

    let expected_input_len = rows
        .checked_mul(in_features)
        .ok_or_else(|| Error::backend("linear input length overflow"))?;
    if input.len() != expected_input_len {
        return Err(Error::backend(format!(
            "linear input shape mismatch: expected [{rows}, {in_features}] = {expected_input_len} values, got {}",
            input.len()
        )));
    }

    let expected_weight_len = out_features
        .checked_mul(in_features)
        .ok_or_else(|| Error::backend("linear weight length overflow"))?;
    if weight.len() != expected_weight_len {
        return Err(Error::backend(format!(
            "linear weight shape mismatch: expected [{out_features}, {in_features}] = {expected_weight_len} values, got {}",
            weight.len()
        )));
    }
    debug_assert_finite_values("linear input", input);
    debug_assert_finite_values("linear weight", weight);

    Ok(())
}

pub(crate) fn validate_swiglu_f32(gate: &[f32], up: &[f32], value_count: usize) -> Result<()> {
    if value_count == 0 {
        return Err(Error::backend(
            "SwiGLU value_count must be greater than zero",
        ));
    }
    if gate.len() != value_count {
        return Err(Error::backend(format!(
            "SwiGLU gate shape mismatch: expected {value_count} values, got {}",
            gate.len()
        )));
    }
    if up.len() != value_count {
        return Err(Error::backend(format!(
            "SwiGLU up shape mismatch: expected {value_count} values, got {}",
            up.len()
        )));
    }
    debug_assert_finite_values("SwiGLU gate", gate);
    debug_assert_finite_values("SwiGLU up", up);

    Ok(())
}

pub(crate) fn validate_add_f32(lhs: &[f32], rhs: &[f32], value_count: usize) -> Result<()> {
    if value_count == 0 {
        return Err(Error::backend("add value_count must be greater than zero"));
    }
    if lhs.len() != value_count {
        return Err(Error::backend(format!(
            "add lhs shape mismatch: expected {value_count} values, got {}",
            lhs.len()
        )));
    }
    if rhs.len() != value_count {
        return Err(Error::backend(format!(
            "add rhs shape mismatch: expected {value_count} values, got {}",
            rhs.len()
        )));
    }
    debug_assert_finite_values("add lhs", lhs);
    debug_assert_finite_values("add rhs", rhs);

    Ok(())
}

pub(crate) fn validate_select_last_token_f32(
    hidden_states: &[f32],
    batch_count: usize,
    token_count: usize,
    hidden_size: usize,
) -> Result<()> {
    if batch_count == 0 {
        return Err(Error::backend(
            "select_last_token batch_count must be greater than zero",
        ));
    }
    if token_count == 0 {
        return Err(Error::backend(
            "select_last_token token_count must be greater than zero",
        ));
    }
    if hidden_size == 0 {
        return Err(Error::backend(
            "select_last_token hidden_size must be greater than zero",
        ));
    }
    let expected_len = batch_count
        .checked_mul(token_count)
        .and_then(|value| value.checked_mul(hidden_size))
        .ok_or_else(|| Error::backend("select_last_token input length overflow"))?;
    if hidden_states.len() != expected_len {
        return Err(Error::backend(format!(
            "select_last_token input shape mismatch: expected [{batch_count}, {token_count}, {hidden_size}] = {expected_len} values, got {}",
            hidden_states.len()
        )));
    }
    debug_assert_finite_values("select_last_token input", hidden_states);

    Ok(())
}

pub(crate) fn validate_heads_to_attention_layout_f32(
    input: &[f32],
    batch_count: usize,
    token_count: usize,
    head_count: usize,
    head_dim: usize,
) -> Result<()> {
    validate_layout_4d(
        "heads_to_attention_layout",
        input,
        batch_count,
        token_count,
        head_count,
        head_dim,
    )
}

pub(crate) fn validate_merge_attention_heads_f32(
    input: &[f32],
    batch_count: usize,
    head_count: usize,
    token_count: usize,
    head_dim: usize,
) -> Result<()> {
    validate_layout_4d(
        "merge_attention_heads",
        input,
        batch_count,
        head_count,
        token_count,
        head_dim,
    )
}

pub(crate) fn validate_split_rope_tail_f32(
    input: &[f32],
    batch_count: usize,
    token_count: usize,
    head_count: usize,
    no_rope_dim: usize,
    rope_dim: usize,
) -> Result<()> {
    if no_rope_dim == 0 {
        return Err(Error::backend(
            "split_rope_tail no_rope_dim must be greater than zero",
        ));
    }
    if rope_dim == 0 {
        return Err(Error::backend(
            "split_rope_tail rope_dim must be greater than zero",
        ));
    }
    let total_dim = no_rope_dim
        .checked_add(rope_dim)
        .ok_or_else(|| Error::backend("split_rope_tail total dim overflow"))?;
    validate_layout_4d(
        "split_rope_tail",
        input,
        batch_count,
        token_count,
        head_count,
        total_dim,
    )
}

pub(crate) fn validate_split_kv_mqa_f32(
    input: &[f32],
    batch_count: usize,
    token_count: usize,
    kv_lora_rank: usize,
    rope_dim: usize,
) -> Result<()> {
    if batch_count == 0 {
        return Err(Error::backend(
            "split_kv_mqa batch_count must be greater than zero",
        ));
    }
    if token_count == 0 {
        return Err(Error::backend(
            "split_kv_mqa token_count must be greater than zero",
        ));
    }
    if kv_lora_rank == 0 {
        return Err(Error::backend(
            "split_kv_mqa kv_lora_rank must be greater than zero",
        ));
    }
    if rope_dim == 0 {
        return Err(Error::backend(
            "split_kv_mqa rope_dim must be greater than zero",
        ));
    }
    let total_dim = kv_lora_rank
        .checked_add(rope_dim)
        .ok_or_else(|| Error::backend("split_kv_mqa total dim overflow"))?;
    let expected_len = batch_count
        .checked_mul(token_count)
        .and_then(|value| value.checked_mul(total_dim))
        .ok_or_else(|| Error::backend("split_kv_mqa input length overflow"))?;
    if input.len() != expected_len {
        return Err(Error::backend(format!(
            "split_kv_mqa input shape mismatch: expected [{batch_count}, {token_count}, {total_dim}] = {expected_len} values, got {}",
            input.len()
        )));
    }
    debug_assert_finite_values("split_kv_mqa input", input);

    Ok(())
}

pub(crate) fn validate_combine_rope_tail_f32(
    no_rope: &[f32],
    rope: &[f32],
    batch_count: usize,
    token_count: usize,
    head_count: usize,
    rope_head_count: usize,
    no_rope_dim: usize,
    rope_dim: usize,
) -> Result<()> {
    if rope_head_count != 1 && rope_head_count != head_count {
        return Err(Error::backend(format!(
            "combine_rope_tail rope_head_count must be 1 or head_count {head_count}, got {rope_head_count}"
        )));
    }
    if no_rope_dim == 0 {
        return Err(Error::backend(
            "combine_rope_tail no_rope_dim must be greater than zero",
        ));
    }
    if rope_dim == 0 {
        return Err(Error::backend(
            "combine_rope_tail rope_dim must be greater than zero",
        ));
    }
    validate_layout_4d(
        "combine_rope_tail_no_rope",
        no_rope,
        batch_count,
        token_count,
        head_count,
        no_rope_dim,
    )?;
    validate_layout_4d(
        "combine_rope_tail_rope",
        rope,
        batch_count,
        token_count,
        rope_head_count,
        rope_dim,
    )
}

fn validate_layout_4d(
    name: &str,
    input: &[f32],
    dim0: usize,
    dim1: usize,
    dim2: usize,
    dim3: usize,
) -> Result<()> {
    if dim0 == 0 || dim1 == 0 || dim2 == 0 || dim3 == 0 {
        return Err(Error::backend(format!(
            "{name} dimensions must be greater than zero, got [{dim0}, {dim1}, {dim2}, {dim3}]"
        )));
    }

    let expected_len = dim0
        .checked_mul(dim1)
        .and_then(|value| value.checked_mul(dim2))
        .and_then(|value| value.checked_mul(dim3))
        .ok_or_else(|| Error::backend(format!("{name} input length overflow")))?;
    if input.len() != expected_len {
        return Err(Error::backend(format!(
            "{name} input shape mismatch: expected [{dim0}, {dim1}, {dim2}, {dim3}] = {expected_len} values, got {}",
            input.len()
        )));
    }
    debug_assert_finite_values(name, input);

    Ok(())
}

pub(crate) fn validate_rms_norm_f32(
    input: &[f32],
    weight: &[f32],
    rows: usize,
    hidden_size: usize,
    eps: f32,
) -> Result<()> {
    validate_rms_norm_buffer(input.len(), weight, rows, hidden_size, eps)?;
    debug_assert_finite_values("RMSNorm input", input);
    debug_assert_finite_values("RMSNorm weight", weight);

    Ok(())
}

/// Shape-only RMSNorm validation for a device-resident input. The weight is
/// still a host slice (norm weights always are), so it is checked fully; the
/// input is checked by length only.
pub(crate) fn validate_rms_norm_buffer(
    input_len: usize,
    weight: &[f32],
    rows: usize,
    hidden_size: usize,
    eps: f32,
) -> Result<()> {
    if rows == 0 {
        return Err(Error::backend("RMSNorm rows must be greater than zero"));
    }
    if hidden_size == 0 {
        return Err(Error::backend(
            "RMSNorm hidden_size must be greater than zero",
        ));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(Error::backend(
            "RMSNorm eps must be finite and greater than zero",
        ));
    }

    let expected_input_len = rows
        .checked_mul(hidden_size)
        .ok_or_else(|| Error::backend("RMSNorm input length overflow"))?;
    if input_len != expected_input_len {
        return Err(Error::backend(format!(
            "RMSNorm input shape mismatch: expected [{rows}, {hidden_size}] = {expected_input_len} values, got {input_len}"
        )));
    }
    if weight.len() != hidden_size {
        return Err(Error::backend(format!(
            "RMSNorm weight shape mismatch: expected [{hidden_size}], got [{}]",
            weight.len()
        )));
    }

    Ok(())
}

pub(crate) fn validate_q2_k_matvec_f32(
    weights: &[u8],
    input: &[f32],
    row_count: usize,
    in_features: usize,
    out_features: usize,
) -> Result<usize> {
    let blocks_per_row =
        validate_q2_k_matvec_buffer(weights, input.len(), row_count, in_features, out_features)?;
    debug_assert_finite_values("Q2_K matvec input", input);

    Ok(blocks_per_row)
}

pub(crate) fn validate_q2_k_matvec_buffer(
    weights: &[u8],
    input_len: usize,
    row_count: usize,
    in_features: usize,
    out_features: usize,
) -> Result<usize> {
    if row_count == 0 {
        return Err(Error::backend(
            "Q2_K matvec row_count must be greater than zero",
        ));
    }
    if in_features == 0 {
        return Err(Error::backend(
            "Q2_K matvec in_features must be greater than zero",
        ));
    }
    if out_features == 0 {
        return Err(Error::backend(
            "Q2_K matvec out_features must be greater than zero",
        ));
    }
    if in_features % Q2_K_BLOCK_VALUES != 0 {
        return Err(Error::backend(format!(
            "Q2_K matvec in_features {in_features} must be divisible by {Q2_K_BLOCK_VALUES}"
        )));
    }

    let blocks_per_row = in_features / Q2_K_BLOCK_VALUES;
    let expected_input_len = row_count
        .checked_mul(in_features)
        .ok_or_else(|| Error::backend("Q2_K matvec input length overflow"))?;
    if input_len != expected_input_len {
        return Err(Error::backend(format!(
            "Q2_K matvec input shape mismatch: expected [{row_count}, {in_features}] = {expected_input_len} values, got {}",
            input_len
        )));
    }

    let expected_weight_bytes = out_features
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(Q2_K_BLOCK_BYTES))
        .ok_or_else(|| Error::backend("Q2_K matvec weight byte length overflow"))?;
    if weights.len() != expected_weight_bytes {
        return Err(Error::backend(format!(
            "Q2_K matvec weight byte mismatch: expected {expected_weight_bytes} bytes, got {}",
            weights.len()
        )));
    }
    Ok(blocks_per_row)
}

pub(crate) fn validate_q2_k_gate_up_swiglu_f32(
    gate_weights: &[u8],
    up_weights: &[u8],
    input: &[f32],
    row_count: usize,
    in_features: usize,
    out_features: usize,
) -> Result<usize> {
    let gate_blocks =
        validate_q2_k_matvec_f32(gate_weights, input, row_count, in_features, out_features)?;
    let up_blocks =
        validate_q2_k_matvec_f32(up_weights, input, row_count, in_features, out_features)?;
    if gate_blocks != up_blocks {
        return Err(Error::backend(format!(
            "Q2_K gate/up SwiGLU block mismatch: gate has {gate_blocks}, up has {up_blocks}"
        )));
    }

    Ok(gate_blocks)
}

pub(crate) fn validate_q8_0_matvec_f32(
    weights: &[u8],
    input: &[f32],
    row_count: usize,
    in_features: usize,
    out_features: usize,
) -> Result<usize> {
    validate_quantized_matvec_shape(
        "Q8_0 matvec",
        weights,
        input,
        row_count,
        in_features,
        out_features,
        Q8_0_BLOCK_VALUES,
        Q8_0_BLOCK_BYTES,
        false,
    )
}

pub(crate) fn validate_q8_0_transposed_matvec_f32(
    weights: &[u8],
    input: &[f32],
    row_count: usize,
    in_features: usize,
    out_features: usize,
) -> Result<usize> {
    validate_quantized_matvec_shape(
        "Q8_0 transposed matvec",
        weights,
        input,
        row_count,
        in_features,
        out_features,
        Q8_0_BLOCK_VALUES,
        Q8_0_BLOCK_BYTES,
        true,
    )
}

pub(crate) fn validate_q2_k_transposed_matvec_f32(
    weights: &[u8],
    input: &[f32],
    row_count: usize,
    in_features: usize,
    out_features: usize,
) -> Result<usize> {
    if row_count == 0 {
        return Err(Error::backend(
            "Q2_K transposed matvec row_count must be greater than zero",
        ));
    }
    if in_features == 0 {
        return Err(Error::backend(
            "Q2_K transposed matvec in_features must be greater than zero",
        ));
    }
    if out_features == 0 {
        return Err(Error::backend(
            "Q2_K transposed matvec out_features must be greater than zero",
        ));
    }
    if out_features % Q2_K_BLOCK_VALUES != 0 {
        return Err(Error::backend(format!(
            "Q2_K transposed matvec out_features {out_features} must be divisible by {Q2_K_BLOCK_VALUES}"
        )));
    }

    let blocks_per_input_row = out_features / Q2_K_BLOCK_VALUES;
    let expected_input_len = row_count
        .checked_mul(in_features)
        .ok_or_else(|| Error::backend("Q2_K transposed matvec input length overflow"))?;
    if input.len() != expected_input_len {
        return Err(Error::backend(format!(
            "Q2_K transposed matvec input shape mismatch: expected [{row_count}, {in_features}] = {expected_input_len} values, got {}",
            input.len()
        )));
    }

    let expected_weight_bytes = in_features
        .checked_mul(blocks_per_input_row)
        .and_then(|blocks| blocks.checked_mul(Q2_K_BLOCK_BYTES))
        .ok_or_else(|| Error::backend("Q2_K transposed matvec weight byte length overflow"))?;
    if weights.len() != expected_weight_bytes {
        return Err(Error::backend(format!(
            "Q2_K transposed matvec weight byte mismatch: expected {expected_weight_bytes} bytes, got {}",
            weights.len()
        )));
    }
    debug_assert_finite_values("Q2_K transposed matvec input", input);

    Ok(blocks_per_input_row)
}

fn validate_quantized_matvec_shape(
    name: &str,
    weights: &[u8],
    input: &[f32],
    row_count: usize,
    in_features: usize,
    out_features: usize,
    block_values: usize,
    block_bytes: usize,
    transposed: bool,
) -> Result<usize> {
    let blocks_per_row = validate_quantized_matvec_buffer_shape(
        name,
        weights.len(),
        input.len(),
        row_count,
        in_features,
        out_features,
        block_values,
        block_bytes,
        transposed,
    )?;
    debug_assert_finite_values(name, input);

    Ok(blocks_per_row)
}

/// Shape-only validation for quantized matvec ops whose input already lives in
/// a GPU buffer. Identical to `validate_quantized_matvec_shape` except it takes
/// lengths instead of slices and cannot (deliberately does not) scan values for
/// finiteness — device-resident inputs have no host copy to scan.
#[allow(clippy::too_many_arguments)]
fn validate_quantized_matvec_buffer_shape(
    name: &str,
    weight_bytes: usize,
    input_len: usize,
    row_count: usize,
    in_features: usize,
    out_features: usize,
    block_values: usize,
    block_bytes: usize,
    transposed: bool,
) -> Result<usize> {
    if row_count == 0 {
        return Err(Error::backend(format!(
            "{name} row_count must be greater than zero"
        )));
    }
    if in_features == 0 {
        return Err(Error::backend(format!(
            "{name} in_features must be greater than zero"
        )));
    }
    if out_features == 0 {
        return Err(Error::backend(format!(
            "{name} out_features must be greater than zero"
        )));
    }

    let quantized_width = if transposed {
        out_features
    } else {
        in_features
    };
    if quantized_width % block_values != 0 {
        return Err(Error::backend(format!(
            "{name} quantized width {quantized_width} must be divisible by {block_values}"
        )));
    }
    let blocks_per_row = quantized_width / block_values;

    let expected_input_len = row_count
        .checked_mul(in_features)
        .ok_or_else(|| Error::backend(format!("{name} input length overflow")))?;
    if input_len != expected_input_len {
        return Err(Error::backend(format!(
            "{name} input shape mismatch: expected [{row_count}, {in_features}] = {expected_input_len} values, got {input_len}"
        )));
    }

    let quantized_rows = if transposed {
        in_features
    } else {
        out_features
    };
    let expected_weight_bytes = quantized_rows
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(|| Error::backend(format!("{name} weight byte length overflow")))?;
    if weight_bytes != expected_weight_bytes {
        return Err(Error::backend(format!(
            "{name} weight byte mismatch: expected {expected_weight_bytes} bytes, got {weight_bytes}"
        )));
    }

    Ok(blocks_per_row)
}

pub(crate) fn validate_q2_k_transposed_matvec_buffer(
    weights: &[u8],
    input_len: usize,
    row_count: usize,
    in_features: usize,
    out_features: usize,
) -> Result<usize> {
    validate_quantized_matvec_buffer_shape(
        "Q2_K transposed matvec",
        weights.len(),
        input_len,
        row_count,
        in_features,
        out_features,
        Q2_K_BLOCK_VALUES,
        Q2_K_BLOCK_BYTES,
        true,
    )
}

pub(crate) fn validate_q8_0_matvec_buffer(
    weights: &[u8],
    input_len: usize,
    row_count: usize,
    in_features: usize,
    out_features: usize,
) -> Result<usize> {
    validate_quantized_matvec_buffer_shape(
        "Q8_0 matvec",
        weights.len(),
        input_len,
        row_count,
        in_features,
        out_features,
        Q8_0_BLOCK_VALUES,
        Q8_0_BLOCK_BYTES,
        false,
    )
}

pub(crate) fn validate_q8_0_transposed_matvec_buffer(
    weights: &[u8],
    input_len: usize,
    row_count: usize,
    in_features: usize,
    out_features: usize,
) -> Result<usize> {
    validate_quantized_matvec_buffer_shape(
        "Q8_0 transposed matvec",
        weights.len(),
        input_len,
        row_count,
        in_features,
        out_features,
        Q8_0_BLOCK_VALUES,
        Q8_0_BLOCK_BYTES,
        true,
    )
}

pub(crate) fn validate_moe_gather_tokens_f32(
    flat_tokens: &[f32],
    token_indices: &[u32],
    token_count: usize,
    hidden_size: usize,
    assignment_count: usize,
) -> Result<()> {
    if token_count == 0 {
        return Err(Error::backend(
            "MoE gather token_count must be greater than zero",
        ));
    }
    if hidden_size == 0 {
        return Err(Error::backend(
            "MoE gather hidden_size must be greater than zero",
        ));
    }
    if assignment_count == 0 {
        return Err(Error::backend(
            "MoE gather assignment_count must be greater than zero",
        ));
    }

    let expected_flat_len = token_count
        .checked_mul(hidden_size)
        .ok_or_else(|| Error::backend("MoE gather flat token length overflow"))?;
    if flat_tokens.len() != expected_flat_len {
        return Err(Error::backend(format!(
            "MoE gather flat_tokens shape mismatch: expected [{token_count}, {hidden_size}] = {expected_flat_len} values, got {}",
            flat_tokens.len()
        )));
    }
    if token_indices.len() != assignment_count {
        return Err(Error::backend(format!(
            "MoE gather token_indices shape mismatch: expected [{assignment_count}], got [{}]",
            token_indices.len()
        )));
    }
    debug_assert_finite_values("MoE gather flat_tokens", flat_tokens);
    if let Some(token_index) = token_indices
        .iter()
        .copied()
        .find(|token_index| *token_index as usize >= token_count)
    {
        return Err(Error::backend(format!(
            "MoE gather token index {token_index} is outside token_count {token_count}"
        )));
    }

    Ok(())
}

pub(crate) fn validate_moe_weighted_index_add_combine_f32(
    accumulator: &[f32],
    token_indices: &[u32],
    expert_outputs: &[f32],
    expert_weights: &[f32],
    token_count: usize,
    hidden_size: usize,
    assignment_count: usize,
) -> Result<()> {
    if token_count == 0 {
        return Err(Error::backend(
            "MoE combine token_count must be greater than zero",
        ));
    }
    if hidden_size == 0 {
        return Err(Error::backend(
            "MoE combine hidden_size must be greater than zero",
        ));
    }
    if assignment_count == 0 {
        return Err(Error::backend(
            "MoE combine assignment_count must be greater than zero",
        ));
    }

    let expected_accumulator_len = token_count
        .checked_mul(hidden_size)
        .ok_or_else(|| Error::backend("MoE combine accumulator length overflow"))?;
    if accumulator.len() != expected_accumulator_len {
        return Err(Error::backend(format!(
            "MoE combine accumulator shape mismatch: expected [{token_count}, {hidden_size}] = {expected_accumulator_len} values, got {}",
            accumulator.len()
        )));
    }
    if token_indices.len() != assignment_count {
        return Err(Error::backend(format!(
            "MoE combine token_indices shape mismatch: expected [{assignment_count}], got [{}]",
            token_indices.len()
        )));
    }
    if expert_weights.len() != assignment_count {
        return Err(Error::backend(format!(
            "MoE combine expert_weights shape mismatch: expected [{assignment_count}], got [{}]",
            expert_weights.len()
        )));
    }

    let expected_expert_output_len = assignment_count
        .checked_mul(hidden_size)
        .ok_or_else(|| Error::backend("MoE combine expert output length overflow"))?;
    if expert_outputs.len() != expected_expert_output_len {
        return Err(Error::backend(format!(
            "MoE combine expert_outputs shape mismatch: expected [{assignment_count}, {hidden_size}] = {expected_expert_output_len} values, got {}",
            expert_outputs.len()
        )));
    }
    debug_assert_finite_values("MoE combine accumulator", accumulator);
    debug_assert_finite_values("MoE combine expert_outputs", expert_outputs);
    debug_assert_finite_values("MoE combine expert_weights", expert_weights);
    if let Some(token_index) = token_indices
        .iter()
        .copied()
        .find(|token_index| *token_index as usize >= token_count)
    {
        return Err(Error::backend(format!(
            "MoE combine token index {token_index} is outside token_count {token_count}"
        )));
    }

    Ok(())
}

pub(crate) fn validate_attention_scores_f32(
    q: &[f32],
    k: &[f32],
    batch_count: usize,
    head_count: usize,
    query_tokens: usize,
    key_tokens: usize,
    head_dim: usize,
) -> Result<()> {
    if batch_count == 0 {
        return Err(Error::backend(
            "attention batch_count must be greater than zero",
        ));
    }
    if head_count == 0 {
        return Err(Error::backend(
            "attention head_count must be greater than zero",
        ));
    }
    if query_tokens == 0 {
        return Err(Error::backend(
            "attention query_tokens must be greater than zero",
        ));
    }
    if key_tokens == 0 {
        return Err(Error::backend(
            "attention key_tokens must be greater than zero",
        ));
    }
    if head_dim == 0 {
        return Err(Error::backend(
            "attention head_dim must be greater than zero",
        ));
    }

    let expected_q_len = batch_count
        .checked_mul(head_count)
        .and_then(|value| value.checked_mul(query_tokens))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| Error::backend("attention q length overflow"))?;
    if q.len() != expected_q_len {
        return Err(Error::backend(format!(
            "attention q shape mismatch: expected [{batch_count}, {head_count}, {query_tokens}, {head_dim}] = {expected_q_len} values, got {}",
            q.len()
        )));
    }

    let expected_k_len = batch_count
        .checked_mul(head_count)
        .and_then(|value| value.checked_mul(key_tokens))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| Error::backend("attention k length overflow"))?;
    if k.len() != expected_k_len {
        return Err(Error::backend(format!(
            "attention k shape mismatch: expected [{batch_count}, {head_count}, {key_tokens}, {head_dim}] = {expected_k_len} values, got {}",
            k.len()
        )));
    }
    debug_assert_finite_values("attention q", q);
    debug_assert_finite_values("attention k", k);

    Ok(())
}

pub(crate) fn validate_attention_values_f32(
    probs: &[f32],
    values: &[f32],
    batch_count: usize,
    head_count: usize,
    query_tokens: usize,
    key_tokens: usize,
    value_dim: usize,
) -> Result<()> {
    if batch_count == 0 {
        return Err(Error::backend(
            "attention values batch_count must be greater than zero",
        ));
    }
    if head_count == 0 {
        return Err(Error::backend(
            "attention values head_count must be greater than zero",
        ));
    }
    if query_tokens == 0 {
        return Err(Error::backend(
            "attention values query_tokens must be greater than zero",
        ));
    }
    if key_tokens == 0 {
        return Err(Error::backend(
            "attention values key_tokens must be greater than zero",
        ));
    }
    if value_dim == 0 {
        return Err(Error::backend(
            "attention values value_dim must be greater than zero",
        ));
    }

    let expected_probs_len = batch_count
        .checked_mul(head_count)
        .and_then(|value| value.checked_mul(query_tokens))
        .and_then(|value| value.checked_mul(key_tokens))
        .ok_or_else(|| Error::backend("attention probabilities length overflow"))?;
    if probs.len() != expected_probs_len {
        return Err(Error::backend(format!(
            "attention probabilities shape mismatch: expected [{batch_count}, {head_count}, {query_tokens}, {key_tokens}] = {expected_probs_len} values, got {}",
            probs.len()
        )));
    }

    let expected_values_len = batch_count
        .checked_mul(head_count)
        .and_then(|value| value.checked_mul(key_tokens))
        .and_then(|value| value.checked_mul(value_dim))
        .ok_or_else(|| Error::backend("attention value tensor length overflow"))?;
    if values.len() != expected_values_len {
        return Err(Error::backend(format!(
            "attention values shape mismatch: expected [{batch_count}, {head_count}, {key_tokens}, {value_dim}] = {expected_values_len} values, got {}",
            values.len()
        )));
    }
    debug_assert_finite_values("attention probabilities", probs);
    debug_assert_finite_values("attention values", values);

    Ok(())
}

pub(crate) fn validate_attention_causal_softmax_f32(
    scores: &[f32],
    batch_count: usize,
    head_count: usize,
    query_tokens: usize,
    key_tokens: usize,
    past_tokens: usize,
) -> Result<()> {
    if batch_count == 0 {
        return Err(Error::backend(
            "attention causal softmax batch_count must be greater than zero",
        ));
    }
    if head_count == 0 {
        return Err(Error::backend(
            "attention causal softmax head_count must be greater than zero",
        ));
    }
    if query_tokens == 0 {
        return Err(Error::backend(
            "attention causal softmax query_tokens must be greater than zero",
        ));
    }
    if key_tokens == 0 {
        return Err(Error::backend(
            "attention causal softmax key_tokens must be greater than zero",
        ));
    }
    if past_tokens
        .checked_add(query_tokens)
        .filter(|expected_key_tokens| *expected_key_tokens == key_tokens)
        .is_none()
    {
        return Err(Error::backend(format!(
            "attention causal softmax expects key_tokens == past_tokens + query_tokens, got key_tokens={key_tokens}, past_tokens={past_tokens}, query_tokens={query_tokens}"
        )));
    }

    let expected_scores_len = batch_count
        .checked_mul(head_count)
        .and_then(|value| value.checked_mul(query_tokens))
        .and_then(|value| value.checked_mul(key_tokens))
        .ok_or_else(|| Error::backend("attention causal softmax score length overflow"))?;
    if scores.len() != expected_scores_len {
        return Err(Error::backend(format!(
            "attention causal softmax scores shape mismatch: expected [{batch_count}, {head_count}, {query_tokens}, {key_tokens}] = {expected_scores_len} values, got {}",
            scores.len()
        )));
    }
    debug_assert_finite_values("attention causal softmax scores", scores);

    Ok(())
}

pub(crate) fn validate_rope_slice_f32(
    input: &[f32],
    batch_count: usize,
    token_count: usize,
    head_count: usize,
    rope_dim: usize,
    position_offset: usize,
    theta: f32,
) -> Result<()> {
    validate_rope_slice_buffer(
        input.len(),
        batch_count,
        token_count,
        head_count,
        rope_dim,
        position_offset,
        theta,
    )?;
    debug_assert_finite_values("RoPE input", input);

    Ok(())
}

/// Shape-only RoPE validation for a device-resident input (length instead of
/// slice; no finiteness scan — see `validate_quantized_matvec_buffer_shape`).
pub(crate) fn validate_rope_slice_buffer(
    input_len: usize,
    batch_count: usize,
    token_count: usize,
    head_count: usize,
    rope_dim: usize,
    position_offset: usize,
    theta: f32,
) -> Result<()> {
    if batch_count == 0 {
        return Err(Error::backend("RoPE batch_count must be greater than zero"));
    }
    if token_count == 0 {
        return Err(Error::backend("RoPE token_count must be greater than zero"));
    }
    if head_count == 0 {
        return Err(Error::backend("RoPE head_count must be greater than zero"));
    }
    if rope_dim == 0 {
        return Err(Error::backend("RoPE rope_dim must be greater than zero"));
    }
    if rope_dim % 2 != 0 {
        return Err(Error::backend(format!(
            "RoPE rope_dim must be even, got {rope_dim}"
        )));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(Error::backend(
            "RoPE theta must be finite and greater than zero",
        ));
    }
    position_offset
        .checked_add(token_count - 1)
        .ok_or_else(|| Error::backend("RoPE position range overflow"))?;

    let expected_input_len = batch_count
        .checked_mul(token_count)
        .and_then(|value| value.checked_mul(head_count))
        .and_then(|value| value.checked_mul(rope_dim))
        .ok_or_else(|| Error::backend("RoPE input length overflow"))?;
    if input_len != expected_input_len {
        return Err(Error::backend(format!(
            "RoPE input shape mismatch: expected [{batch_count}, {token_count}, {head_count}, {rope_dim}] = {expected_input_len} values, got {input_len}"
        )));
    }

    Ok(())
}

pub(crate) fn debug_assert_finite_values(name: &str, values: &[f32]) {
    debug_assert!(
        values.iter().all(|value| value.is_finite()),
        "{name} contains non-finite values"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_matmul_shape() {
        validate_matmul_f32(&[1.0; 6], &[0.5; 12], 2, 3, 4).unwrap();
    }

    #[test]
    fn rejects_bad_matmul_rhs_len() {
        let err = validate_matmul_f32(&[1.0; 6], &[0.5; 11], 2, 3, 4)
            .expect_err("bad matmul rhs length should fail");

        assert!(err.to_string().contains("rhs shape mismatch"));
    }

    #[test]
    fn accepts_valid_linear_shape() {
        validate_linear_f32(&[1.0; 6], &[0.5; 12], 2, 3, 4).unwrap();
    }

    #[test]
    fn rejects_bad_linear_weight_len() {
        let err = validate_linear_f32(&[1.0; 6], &[0.5; 11], 2, 3, 4)
            .expect_err("bad linear weight length should fail");

        assert!(err.to_string().contains("weight shape mismatch"));
    }

    #[test]
    fn accepts_valid_swiglu_shape() {
        validate_swiglu_f32(&[1.0; 6], &[0.5; 6], 6).unwrap();
    }

    #[test]
    fn rejects_bad_swiglu_up_len() {
        let err = validate_swiglu_f32(&[1.0; 6], &[0.5; 5], 6)
            .expect_err("bad SwiGLU up length should fail");

        assert!(err.to_string().contains("up shape mismatch"));
    }

    #[test]
    fn accepts_valid_add_shape() {
        validate_add_f32(&[1.0; 6], &[0.5; 6], 6).unwrap();
    }

    #[test]
    fn rejects_bad_add_rhs_len() {
        let err =
            validate_add_f32(&[1.0; 6], &[0.5; 5], 6).expect_err("bad rhs length should fail");

        assert!(err.to_string().contains("rhs shape mismatch"));
    }

    #[test]
    fn accepts_valid_select_last_token_shape() {
        validate_select_last_token_f32(&[1.0; 12], 2, 3, 2).unwrap();
    }

    #[test]
    fn accepts_valid_heads_to_attention_layout_shape() {
        validate_heads_to_attention_layout_f32(&[1.0; 24], 1, 3, 2, 4).unwrap();
    }

    #[test]
    fn accepts_valid_merge_attention_heads_shape() {
        validate_merge_attention_heads_f32(&[1.0; 24], 1, 2, 3, 4).unwrap();
    }

    #[test]
    fn accepts_valid_split_rope_tail_shape() {
        validate_split_rope_tail_f32(&[1.0; 24], 1, 3, 2, 2, 2).unwrap();
    }

    #[test]
    fn accepts_valid_split_kv_mqa_shape() {
        validate_split_kv_mqa_f32(&[1.0; 18], 1, 3, 4, 2).unwrap();
    }

    #[test]
    fn accepts_valid_combine_rope_tail_shape_with_shared_rope_head() {
        validate_combine_rope_tail_f32(&[1.0; 12], &[0.5; 6], 1, 3, 2, 1, 2, 2).unwrap();
    }

    #[test]
    fn accepts_valid_combine_rope_tail_shape_with_per_head_rope() {
        validate_combine_rope_tail_f32(&[1.0; 12], &[0.5; 12], 1, 3, 2, 2, 2, 2).unwrap();
    }

    #[test]
    fn rejects_bad_select_last_token_len() {
        let err = validate_select_last_token_f32(&[1.0; 11], 2, 3, 2)
            .expect_err("bad input length should fail");

        assert!(err.to_string().contains("input shape mismatch"));
    }

    #[test]
    fn rejects_bad_heads_to_attention_layout_len() {
        let err = validate_heads_to_attention_layout_f32(&[1.0; 23], 1, 3, 2, 4)
            .expect_err("bad heads_to_attention_layout length should fail");

        assert!(err.to_string().contains("shape mismatch"));
    }

    #[test]
    fn rejects_bad_combine_rope_tail_head_count() {
        let err = validate_combine_rope_tail_f32(&[1.0; 12], &[0.5; 18], 1, 3, 2, 3, 2, 2)
            .expect_err("unsupported rope head count should fail");

        assert!(err.to_string().contains("rope_head_count"));
    }

    #[test]
    fn rejects_bad_split_kv_mqa_len() {
        let err = validate_split_kv_mqa_f32(&[1.0; 17], 1, 3, 4, 2)
            .expect_err("bad split_kv_mqa length should fail");

        assert!(err.to_string().contains("shape mismatch"));
    }

    #[test]
    fn accepts_valid_rms_norm_shape() {
        validate_rms_norm_f32(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0], 2, 2, 1e-5).unwrap();
    }

    #[test]
    fn rejects_wrong_input_len() {
        let err = validate_rms_norm_f32(&[1.0, 2.0, 3.0], &[1.0, 1.0], 2, 2, 1e-5)
            .expect_err("shape mismatch should fail");

        assert!(err.to_string().contains("input shape mismatch"));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "RMSNorm weight contains non-finite values")]
    fn rejects_non_finite_weight() {
        let _ = validate_rms_norm_f32(&[1.0, 2.0], &[f32::NAN, 1.0], 1, 2, 1e-5);
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn release_builds_do_not_scan_non_finite_weight() {
        validate_rms_norm_f32(&[1.0, 2.0], &[f32::NAN, 1.0], 1, 2, 1e-5).unwrap();
    }

    #[test]
    fn accepts_valid_q2_k_matvec_shape() {
        let weights = vec![0_u8; Q2_K_BLOCK_BYTES * 2];
        let input = vec![0.5_f32; Q2_K_BLOCK_VALUES];
        let blocks_per_row =
            validate_q2_k_matvec_f32(&weights, &input, 1, Q2_K_BLOCK_VALUES, 2).unwrap();

        assert_eq!(blocks_per_row, 1);
    }

    #[test]
    fn rejects_bad_q2_k_weight_bytes() {
        let input = vec![0.5_f32; Q2_K_BLOCK_VALUES];
        let err = validate_q2_k_matvec_f32(&[0_u8; 16], &input, 1, Q2_K_BLOCK_VALUES, 2)
            .expect_err("bad Q2 byte count should fail");

        assert!(err.to_string().contains("weight byte mismatch"));
    }

    #[test]
    fn rejects_bad_q2_k_input_width() {
        let err = validate_q2_k_matvec_f32(&[0_u8; Q2_K_BLOCK_BYTES], &[0.5_f32; 128], 1, 128, 1)
            .expect_err("bad Q2 input width should fail");

        assert!(err.to_string().contains("divisible"));
    }

    #[test]
    fn accepts_valid_q2_k_transposed_matvec_shape() {
        let weights = vec![0_u8; Q2_K_BLOCK_BYTES * 2];
        let input = vec![0.5_f32; 2];
        let blocks_per_input_row =
            validate_q2_k_transposed_matvec_f32(&weights, &input, 1, 2, Q2_K_BLOCK_VALUES).unwrap();

        assert_eq!(blocks_per_input_row, 1);
    }

    #[test]
    fn rejects_bad_q2_k_transposed_output_width() {
        let err = validate_q2_k_transposed_matvec_f32(
            &[0_u8; Q2_K_BLOCK_BYTES],
            &[0.5_f32; 2],
            1,
            2,
            128,
        )
        .expect_err("bad Q2 transposed output width should fail");

        assert!(err.to_string().contains("divisible"));
    }

    #[test]
    fn accepts_valid_moe_combine_shape() {
        validate_moe_weighted_index_add_combine_f32(
            &[0.0; 6],
            &[0, 2],
            &[1.0, 2.0, 3.0, 4.0],
            &[0.25, 0.75],
            3,
            2,
            2,
        )
        .unwrap();
    }

    #[test]
    fn accepts_valid_moe_gather_shape() {
        validate_moe_gather_tokens_f32(&[0.0; 6], &[0, 2], 3, 2, 2).unwrap();
    }

    #[test]
    fn rejects_moe_gather_bad_token_index() {
        let err = validate_moe_gather_tokens_f32(&[0.0; 4], &[2], 2, 2, 1)
            .expect_err("bad token index should fail");

        assert!(err.to_string().contains("outside token_count"));
    }

    #[test]
    fn rejects_moe_combine_bad_token_index() {
        let err = validate_moe_weighted_index_add_combine_f32(
            &[0.0; 4],
            &[2],
            &[1.0, 2.0],
            &[1.0],
            2,
            2,
            1,
        )
        .expect_err("bad token index should fail");

        assert!(err.to_string().contains("outside token_count"));
    }

    #[test]
    fn accepts_valid_attention_scores_shape() {
        validate_attention_scores_f32(&[0.5; 8], &[0.25; 12], 1, 2, 2, 3, 2).unwrap();
    }

    #[test]
    fn rejects_attention_scores_bad_k_len() {
        let err = validate_attention_scores_f32(&[0.5; 8], &[0.25; 10], 1, 2, 2, 3, 2)
            .expect_err("bad k length should fail");

        assert!(err.to_string().contains("k shape mismatch"));
    }

    #[test]
    fn accepts_valid_attention_values_shape() {
        validate_attention_values_f32(&[0.5; 12], &[0.25; 18], 1, 2, 2, 3, 3).unwrap();
    }

    #[test]
    fn rejects_attention_values_bad_value_len() {
        let err = validate_attention_values_f32(&[0.5; 12], &[0.25; 17], 1, 2, 2, 3, 3)
            .expect_err("bad value length should fail");

        assert!(err.to_string().contains("attention values shape mismatch"));
    }

    #[test]
    fn accepts_valid_attention_causal_softmax_shape() {
        validate_attention_causal_softmax_f32(&[0.5; 12], 1, 2, 2, 3, 1).unwrap();
    }

    #[test]
    fn rejects_attention_causal_softmax_bad_past_tokens() {
        let err = validate_attention_causal_softmax_f32(&[0.5; 12], 1, 2, 2, 3, 0)
            .expect_err("bad past_tokens should fail");

        assert!(err.to_string().contains("past_tokens"));
    }

    #[test]
    fn rejects_attention_causal_softmax_bad_score_len() {
        let err = validate_attention_causal_softmax_f32(&[0.5; 11], 1, 2, 2, 3, 1)
            .expect_err("bad score length should fail");

        assert!(err.to_string().contains("scores shape mismatch"));
    }

    #[test]
    fn accepts_valid_rope_shape() {
        validate_rope_slice_f32(&[0.5; 16], 1, 2, 2, 4, 0, 10_000.0).unwrap();
    }

    #[test]
    fn rejects_rope_odd_dim() {
        let err = validate_rope_slice_f32(&[0.5; 3], 1, 1, 1, 3, 0, 10_000.0)
            .expect_err("odd rope_dim should fail");

        assert!(err.to_string().contains("even"));
    }

    #[test]
    fn rejects_rope_bad_theta() {
        let err = validate_rope_slice_f32(&[0.5; 4], 1, 1, 1, 4, 0, 0.0)
            .expect_err("bad theta should fail");

        assert!(err.to_string().contains("theta"));
    }
}
