use std::cmp::Ordering;

use backend::{Backend, BackendCapabilities, DeviceValue};
use common::{validate_exact_shape, Error, F32Tensor, Result, Shape};
use config::{Config, IndexerLayerKind};
use gguf::{GgmlType, GgufFile};

use crate::{profile, IndexerIndex, QuantizedLinear, TensorLoadReport, TensorRef, WeightLoader};

const INDEXER_LAYER_NORM_EPS: f32 = 1e-6;

#[derive(Debug)]
pub struct DsaIndexer<'a> {
    layer_index: usize,
    wq_b: QuantizedLinear<'a>,
    wk: QuantizedLinear<'a>,
    weights_proj: F32Tensor,
    k_norm_weight: F32Tensor,
    k_norm_bias: F32Tensor,
    load_report: DsaIndexerLoadReport,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DsaIndexerLoadReport {
    pub backend: BackendCapabilities,
    pub layer_index: usize,
    pub layer_kind: IndexerLayerKind,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub top_k: usize,
    pub wq_b_shape: Shape,
    pub wk_shape: Shape,
    pub weights_proj: TensorLoadReport,
    pub k_norm_weight: TensorLoadReport,
    pub k_norm_bias: TensorLoadReport,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DsaTopKSelection {
    pub batch: usize,
    pub query_tokens: usize,
    pub key_tokens: usize,
    pub top_k: usize,
    pub token_indices: Vec<u32>,
    pub current_key: F32Tensor,
}

#[derive(Debug)]
pub struct DsaDeviceTopKSelection {
    pub batch: usize,
    pub query_tokens: usize,
    pub key_tokens: usize,
    pub top_k: usize,
    pub token_indices: Vec<u32>,
    pub current_key_device: DeviceValue,
}

impl DsaTopKSelection {
    pub fn dims(&self) -> [usize; 3] {
        [self.batch, self.query_tokens, self.top_k]
    }
}

impl<'a> DsaIndexer<'a> {
    pub fn open<B: Backend>(
        gguf: &'a GgufFile,
        config: &Config,
        layer_index: usize,
        index: &IndexerIndex,
        q_lora_rank: usize,
        backend: &B,
        output_chunk_rows: usize,
    ) -> Result<Option<Self>> {
        let layer_kind = if layer_index == config.num_layers && config.num_nextn_predict_layers == 1
        {
            IndexerLayerKind::Full
        } else {
            config.indexer_layer_kind(layer_index).ok_or_else(|| {
                Error::gguf(format!(
                    "GLM-5.2 DSA indexer layer {layer_index} exceeds num_layers {}",
                    config.num_layers
                ))
            })?
        };
        if layer_kind == IndexerLayerKind::Shared {
            return Ok(None);
        }
        if !config.indexer_rope_interleave {
            return Err(Error::config(
                "Inferno currently targets GLM-5.2 interleaved indexer RoPE only",
            ));
        }

        validate_ref_shape(
            "dsa_indexer_k_norm_weight",
            &index.k_norm_weight,
            &[config.index_head_dim],
            GgmlType::F32,
        )?;
        validate_ref_shape(
            "dsa_indexer_k_norm_bias",
            &index.k_norm_bias,
            &[config.index_head_dim],
            GgmlType::F32,
        )?;
        validate_ref_shape(
            "dsa_indexer_proj",
            &index.proj,
            &[config.hidden_size, config.index_n_heads],
            GgmlType::F32,
        )?;
        validate_ref_shape(
            "dsa_indexer_attn_k",
            &index.attn_k,
            &[config.hidden_size, config.index_head_dim],
            GgmlType::Q8_0,
        )?;
        validate_ref_shape(
            "dsa_indexer_attn_q_b",
            &index.attn_q_b,
            &[
                q_lora_rank,
                config
                    .index_n_heads
                    .checked_mul(config.index_head_dim)
                    .ok_or_else(|| Error::config("DSA indexer q output width overflow"))?,
            ],
            GgmlType::Q8_0,
        )?;

        let loader = WeightLoader::new(gguf);
        let weights_proj = loader.load_tensor_as_f32_tensor(&index.proj)?;
        let k_norm_weight = loader.load_tensor_as_f32_tensor(&index.k_norm_weight)?;
        let k_norm_bias = loader.load_tensor_as_f32_tensor(&index.k_norm_bias)?;
        let wq_b = QuantizedLinear::open(
            gguf,
            &index.attn_q_b,
            q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            output_chunk_rows,
        )?;
        let wk = QuantizedLinear::open(
            gguf,
            &index.attn_k,
            config.hidden_size,
            config.index_head_dim,
            output_chunk_rows,
        )?;

        let load_report = DsaIndexerLoadReport {
            backend: backend.capabilities(),
            layer_index,
            layer_kind,
            n_heads: config.index_n_heads,
            head_dim: config.index_head_dim,
            rope_dim: config.qk_rope_dim,
            top_k: config.dsa_index_topk,
            wq_b_shape: Shape::new(vec![
                config.index_n_heads * config.index_head_dim,
                q_lora_rank,
            ]),
            wk_shape: Shape::new(vec![config.index_head_dim, config.hidden_size]),
            weights_proj: weights_proj.report,
            k_norm_weight: k_norm_weight.report,
            k_norm_bias: k_norm_bias.report,
        };

        Ok(Some(Self {
            layer_index,
            wq_b,
            wk,
            weights_proj: weights_proj.tensor,
            k_norm_weight: k_norm_weight.tensor,
            k_norm_bias: k_norm_bias.tensor,
            load_report,
        }))
    }

    pub fn load_report(&self) -> &DsaIndexerLoadReport {
        &self.load_report
    }

    pub fn key_f32<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        position_offset: usize,
        backend: &B,
    ) -> Result<F32Tensor> {
        let dims = hidden_states.dims();
        validate_exact_shape(
            "dsa_indexer_hidden_states",
            dims,
            &[dims[0], dims[1], config.hidden_size],
        )?;
        let raw_key = profile::run_layer_stage(self.layer_index, "dsa_indexer.wk", || {
            self.wk.forward_f32_tensor(hidden_states, backend)
        })?;
        let normed_key = layer_norm_last_dim(
            &raw_key,
            &self.k_norm_weight,
            &self.k_norm_bias,
            INDEXER_LAYER_NORM_EPS,
        )?;
        apply_interleaved_rope_to_key(
            &normed_key,
            config.qk_rope_dim,
            position_offset,
            config.rope_theta as f32,
        )
    }

    pub fn key_device<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &DeviceValue,
        position_offset: usize,
        backend: &B,
    ) -> Result<Option<DeviceValue>> {
        let dims = hidden_states.dims();
        validate_exact_shape(
            "dsa_indexer_device_hidden_states",
            dims,
            &[dims[0], dims[1], config.hidden_size],
        )?;
        let raw_key = crate::try_device!(profile::run_layer_stage(
            self.layer_index,
            "dsa_indexer.wk_device",
            || self.wk.forward_device(hidden_states, backend),
        ));
        backend.dsa_index_key_device(
            &raw_key,
            &self.k_norm_weight,
            &self.k_norm_bias,
            config.qk_rope_dim,
            position_offset,
            config.rope_theta as f32,
        )
    }

    pub fn select_decode_topk_device<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &DeviceValue,
        q_resid: &DeviceValue,
        cached_index_keys: &DeviceValue,
        position_offset: usize,
        backend: &B,
    ) -> Result<Option<DsaDeviceTopKSelection>> {
        let hidden_dims = hidden_states.dims();
        validate_exact_shape(
            "dsa_indexer_device_decode_hidden_states",
            hidden_dims,
            &[hidden_dims[0], 1, config.hidden_size],
        )?;
        let batch = hidden_dims[0];
        validate_exact_shape(
            "dsa_indexer_device_decode_q_resid",
            q_resid.dims(),
            &[batch, 1, self.wq_b_input_features()?],
        )?;
        validate_exact_shape(
            "dsa_indexer_device_cached_keys",
            cached_index_keys.dims(),
            &[batch, position_offset, config.index_head_dim],
        )?;

        let current_key =
            crate::try_device!(self.key_device(config, hidden_states, position_offset, backend));
        let q_raw = crate::try_device!(profile::run_layer_stage(
            self.layer_index,
            "dsa_indexer.wq_b_device",
            || self.wq_b.forward_device(q_resid, backend),
        ));
        let key_tokens = position_offset
            .checked_add(1)
            .ok_or_else(|| Error::model("DSA device indexer key token count overflow"))?;
        let top_k = config.dsa_index_topk.min(key_tokens);
        let token_indices = backend
            .dsa_decode_topk_device(
                hidden_states,
                &q_raw,
                cached_index_keys,
                &current_key,
                &self.weights_proj,
                config.index_n_heads,
                config.index_head_dim,
                config.qk_rope_dim,
                position_offset,
                config.rope_theta as f32,
                top_k,
            )?
            .ok_or_else(|| Error::backend("native Metal DSA decode top-k is required"))?;

        Ok(Some(DsaDeviceTopKSelection {
            batch,
            query_tokens: 1,
            key_tokens,
            top_k,
            token_indices,
            current_key_device: current_key,
        }))
    }

    pub fn select_decode_topk<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        q_resid: &F32Tensor,
        cached_index_keys: &F32Tensor,
        position_offset: usize,
        backend: &B,
    ) -> Result<DsaTopKSelection> {
        let hidden_dims = hidden_states.dims();
        validate_exact_shape(
            "dsa_indexer_decode_hidden_states",
            hidden_dims,
            &[hidden_dims[0], 1, config.hidden_size],
        )?;
        let batch = hidden_dims[0];
        validate_exact_shape(
            "dsa_indexer_decode_q_resid",
            q_resid.dims(),
            &[batch, 1, self.wq_b_input_features()?],
        )?;
        validate_exact_shape(
            "dsa_indexer_cached_keys",
            cached_index_keys.dims(),
            &[batch, position_offset, config.index_head_dim],
        )?;

        let current_key = self.key_f32(config, hidden_states, position_offset, backend)?;
        let key_tokens = position_offset
            .checked_add(1)
            .ok_or_else(|| Error::model("DSA indexer key token count overflow"))?;
        let keys = concat_decode_key(cached_index_keys, &current_key, key_tokens)?;

        let q = profile::run_layer_stage(self.layer_index, "dsa_indexer.wq_b", || {
            self.wq_b.forward_f32_tensor(q_resid, backend)
        })?;
        let q = apply_interleaved_rope_to_query(
            &q,
            config.index_n_heads,
            config.index_head_dim,
            config.qk_rope_dim,
            position_offset,
            config.rope_theta as f32,
        )?;
        let weights = indexer_weights(hidden_states, &self.weights_proj, config.index_n_heads)?;
        let top_k = config.dsa_index_topk.min(key_tokens);
        let token_indices = select_topk_indices(
            &q,
            &keys,
            &weights,
            batch,
            config.index_n_heads,
            config.index_head_dim,
            key_tokens,
            top_k,
        )?;

        Ok(DsaTopKSelection {
            batch,
            query_tokens: 1,
            key_tokens,
            top_k,
            token_indices,
            current_key,
        })
    }

    fn wq_b_input_features(&self) -> Result<usize> {
        self.load_report
            .wq_b_shape
            .dims()
            .get(1)
            .copied()
            .ok_or_else(|| Error::model("DSA indexer q input shape missing"))
    }
}

fn validate_ref_shape(
    context: &str,
    tensor_ref: &TensorRef,
    expected: &[usize],
    expected_type: GgmlType,
) -> Result<()> {
    let dims = tensor_ref
        .dims
        .iter()
        .map(|dim| {
            usize::try_from(*dim).map_err(|_| {
                Error::gguf(format!(
                    "GGUF tensor {} dimension {dim} does not fit usize",
                    tensor_ref.name
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    validate_exact_shape(context, &dims, expected)?;
    if tensor_ref.ty != expected_type {
        return Err(Error::gguf(format!(
            "{context} tensor {} must be {expected_type}, got {}",
            tensor_ref.name, tensor_ref.ty
        )));
    }
    Ok(())
}

fn layer_norm_last_dim(
    input: &F32Tensor,
    weight: &F32Tensor,
    bias: &F32Tensor,
    eps: f32,
) -> Result<F32Tensor> {
    let dims = input.dims();
    let hidden = *dims
        .last()
        .ok_or_else(|| Error::model("DSA indexer layer norm input must have rank >= 1"))?;
    validate_exact_shape("dsa_indexer_k_norm_weight", weight.dims(), &[hidden])?;
    validate_exact_shape("dsa_indexer_k_norm_bias", bias.dims(), &[hidden])?;

    let row_count = input.values().len() / hidden;
    let mut output = vec![0.0_f32; input.values().len()];
    for row in 0..row_count {
        let row_start = row * hidden;
        let row_values = &input.values()[row_start..row_start + hidden];
        let mean = row_values.iter().copied().sum::<f32>() / hidden as f32;
        let variance = row_values
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f32>()
            / hidden as f32;
        let inv_std = 1.0_f32 / (variance + eps).sqrt();
        for dim in 0..hidden {
            output[row_start + dim] =
                ((row_values[dim] - mean) * inv_std * weight.values()[dim]) + bias.values()[dim];
        }
    }
    F32Tensor::new(output, dims.to_vec())
}

fn apply_interleaved_rope_to_query(
    q: &F32Tensor,
    heads: usize,
    head_dim: usize,
    rope_dim: usize,
    position_offset: usize,
    theta: f32,
) -> Result<F32Tensor> {
    let dims = q.dims();
    validate_exact_shape("dsa_indexer_query", dims, &[dims[0], 1, heads * head_dim])?;
    let mut output = q.values().to_vec();
    for batch in 0..dims[0] {
        for head in 0..heads {
            let base = (batch * heads + head) * head_dim;
            rotate_interleaved_pair_range(
                &mut output[base..base + head_dim],
                rope_dim,
                position_offset,
                theta,
            )?;
        }
    }
    F32Tensor::new(output, vec![dims[0], 1, heads, head_dim])
}

fn apply_interleaved_rope_to_key(
    key: &F32Tensor,
    rope_dim: usize,
    position_offset: usize,
    theta: f32,
) -> Result<F32Tensor> {
    let dims = key.dims();
    let batch = dims[0];
    let tokens = dims[1];
    let head_dim = dims[2];
    let mut output = key.values().to_vec();
    for batch_index in 0..batch {
        for token_index in 0..tokens {
            let base = (batch_index * tokens + token_index) * head_dim;
            rotate_interleaved_pair_range(
                &mut output[base..base + head_dim],
                rope_dim,
                position_offset + token_index,
                theta,
            )?;
        }
    }
    F32Tensor::new(output, dims.to_vec())
}

fn rotate_interleaved_pair_range(
    values: &mut [f32],
    rope_dim: usize,
    position: usize,
    theta: f32,
) -> Result<()> {
    if rope_dim % 2 != 0 {
        return Err(Error::model("DSA indexer RoPE dim must be even"));
    }
    if rope_dim > values.len() {
        return Err(Error::model(format!(
            "DSA indexer RoPE dim {rope_dim} exceeds head dim {}",
            values.len()
        )));
    }
    let pair_count = rope_dim / 2;
    let mut rotated = vec![0.0_f32; rope_dim];
    for pair in 0..pair_count {
        let even = pair * 2;
        let odd = even + 1;
        let freq = 1.0_f32 / theta.powf((2 * pair) as f32 / rope_dim as f32);
        let angle = position as f32 * freq;
        let cos = angle.cos();
        let sin = angle.sin();
        let x0 = values[even];
        let x1 = values[odd];
        rotated[pair] = (x0 * cos) - (x1 * sin);
        rotated[pair_count + pair] = (x1 * cos) + (x0 * sin);
    }
    values[..rope_dim].copy_from_slice(&rotated);
    Ok(())
}

fn indexer_weights(
    hidden_states: &F32Tensor,
    weights_proj: &F32Tensor,
    heads: usize,
) -> Result<Vec<f32>> {
    let dims = hidden_states.dims();
    let batch = dims[0];
    let hidden = dims[2];
    validate_exact_shape(
        "dsa_indexer_weights_proj",
        weights_proj.dims(),
        &[hidden, heads],
    )?;
    let scale = (heads as f32).sqrt().recip();
    let mut output = vec![0.0_f32; batch * heads];
    for batch_index in 0..batch {
        for head in 0..heads {
            let mut sum = 0.0_f32;
            for dim in 0..hidden {
                sum += hidden_states.values()[(batch_index * hidden) + dim]
                    * weights_proj.values()[(dim * heads) + head];
            }
            output[(batch_index * heads) + head] = sum * scale;
        }
    }
    Ok(output)
}

fn concat_decode_key(
    past: &F32Tensor,
    current: &F32Tensor,
    key_tokens: usize,
) -> Result<F32Tensor> {
    let dims = current.dims();
    let batch = dims[0];
    let head_dim = dims[2];
    validate_exact_shape("dsa_indexer_current_key", dims, &[batch, 1, head_dim])?;
    let past_tokens = key_tokens - 1;
    validate_exact_shape(
        "dsa_indexer_past_key",
        past.dims(),
        &[batch, past_tokens, head_dim],
    )?;
    let mut values = Vec::with_capacity(batch * key_tokens * head_dim);
    for batch_index in 0..batch {
        let past_start = batch_index * past_tokens * head_dim;
        values.extend_from_slice(&past.values()[past_start..past_start + past_tokens * head_dim]);
        let current_start = batch_index * head_dim;
        values.extend_from_slice(&current.values()[current_start..current_start + head_dim]);
    }
    F32Tensor::new(values, vec![batch, key_tokens, head_dim])
}

#[allow(clippy::too_many_arguments)]
fn select_topk_indices(
    q: &F32Tensor,
    keys: &F32Tensor,
    weights: &[f32],
    batch: usize,
    heads: usize,
    head_dim: usize,
    key_tokens: usize,
    top_k: usize,
) -> Result<Vec<u32>> {
    validate_exact_shape(
        "dsa_indexer_q_for_topk",
        q.dims(),
        &[batch, 1, heads, head_dim],
    )?;
    validate_exact_shape(
        "dsa_indexer_keys_for_topk",
        keys.dims(),
        &[batch, key_tokens, head_dim],
    )?;
    validate_exact_shape(
        "dsa_indexer_weights_for_topk",
        &[weights.len()],
        &[batch * heads],
    )?;
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = Vec::with_capacity(batch * top_k);
    for batch_index in 0..batch {
        let mut scores = Vec::with_capacity(key_tokens);
        for token in 0..key_tokens {
            let mut score = 0.0_f32;
            for head in 0..heads {
                let mut dot = 0.0_f32;
                let q_base = ((batch_index * heads + head) * head_dim) as usize;
                let k_base = (batch_index * key_tokens + token) * head_dim;
                for dim in 0..head_dim {
                    dot += q.values()[q_base + dim] * keys.values()[k_base + dim];
                }
                score += weights[(batch_index * heads) + head] * (dot * scale).max(0.0);
            }
            scores.push((token, score));
        }
        scores.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        for (token, _) in scores.into_iter().take(top_k) {
            output.push(u32::try_from(token).map_err(|_| {
                Error::model(format!("DSA top-k token index {token} does not fit u32"))
            })?);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_norm_last_dim_applies_weight_and_bias() {
        let input = F32Tensor::new(vec![1.0, 3.0], [1, 1, 2]).unwrap();
        let weight = F32Tensor::new(vec![2.0, 2.0], [2]).unwrap();
        let bias = F32Tensor::new(vec![0.5, -0.5], [2]).unwrap();

        let output = layer_norm_last_dim(&input, &weight, &bias, 1e-6).unwrap();

        assert_eq!(output.dims(), &[1, 1, 2]);
        assert!((output.values()[0] - -1.499999).abs() < 1e-4);
        assert!((output.values()[1] - 1.499999).abs() < 1e-4);
    }

    #[test]
    fn topk_selection_uses_weighted_relu_scores_and_stable_ties() {
        let q = F32Tensor::new(vec![1.0, 0.0, 0.0, 1.0], [1, 1, 2, 2]).unwrap();
        let keys = F32Tensor::new(
            vec![
                1.0, 0.0, // token 0: head 0 wins
                0.0, 1.0, // token 1: head 1 wins
                -1.0, 0.0, // token 2: negative dot is clamped to zero
                1.0, 0.0, // token 3: tie with token 0
            ],
            [1, 4, 2],
        )
        .unwrap();
        let weights = vec![2.0, 1.0];

        let topk = select_topk_indices(&q, &keys, &weights, 1, 2, 2, 4, 3).unwrap();

        assert_eq!(topk, vec![0, 3, 1]);
    }

    #[test]
    fn query_rope_returns_head_shaped_tensor() {
        let q = F32Tensor::new(vec![1.0, 0.0, 0.0, 1.0], [1, 1, 4]).unwrap();

        let output = apply_interleaved_rope_to_query(&q, 2, 2, 2, 0, 10_000.0).unwrap();

        assert_eq!(output.dims(), &[1, 1, 2, 2]);
        assert_eq!(output.values(), &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn interleaved_rope_writes_rotary_slice_in_half_split_order() {
        let q = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 4]).unwrap();

        let output = apply_interleaved_rope_to_query(&q, 1, 4, 4, 0, 10_000.0).unwrap();

        assert_eq!(output.dims(), &[1, 1, 1, 4]);
        assert_eq!(output.values(), &[1.0, 3.0, 2.0, 4.0]);
    }
}
