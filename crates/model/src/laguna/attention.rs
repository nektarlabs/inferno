use backend::{Backend, DeviceBf16Matrix, DeviceValue, LagunaFp8KvCache, LagunaKvRetention};
use common::{DType, Error, Result};
use config::{LagunaAttentionKind, LagunaConfig};

use super::{LagunaDeviceAttentionWeights, LagunaDeviceRopeTables};

const HEAD_DIM: usize = 128;
const KV_HEADS: usize = 8;

/// K/V state owned by one Laguna transformer layer.
///
/// Full-attention layers receive the requested context capacity. Sliding
/// layers always allocate exactly 512 token slots and overwrite them as a
/// ring. The wrapped cache never leaves Metal during inference.
#[derive(Debug)]
pub struct LagunaAttentionCache {
    layer_index: usize,
    inner: LagunaFp8KvCache,
}

impl LagunaAttentionCache {
    pub fn prepare<B: Backend>(
        config: &LagunaConfig,
        layer_index: usize,
        batch: usize,
        full_capacity_tokens: usize,
        key_scale: f32,
        value_scale: f32,
        backend: &B,
    ) -> Result<Self> {
        if full_capacity_tokens == 0 || full_capacity_tokens > config.max_position_embeddings {
            return Err(Error::cache(format!(
                "Laguna full KV capacity must be within 1..={}, got {full_capacity_tokens}",
                config.max_position_embeddings
            )));
        }
        let attention_kind = config
            .attention_kind(layer_index)
            .ok_or_else(|| Error::config(format!("Laguna layer {layer_index} is out of range")))?;
        let (retention, capacity_tokens) =
            cache_spec(attention_kind, full_capacity_tokens, config.sliding_window);
        let inner = backend
            .prepare_laguna_fp8_kv_cache(batch, capacity_tokens, retention, key_scale, value_scale)?
            .ok_or_else(|| Error::backend("Laguna FP8 KV cache requires native Metal"))?;
        Ok(Self { layer_index, inner })
    }

    pub fn layer_index(&self) -> usize {
        self.layer_index
    }

    pub fn total_tokens(&self) -> usize {
        self.inner.total_tokens()
    }

    pub fn stored_tokens(&self) -> usize {
        self.inner.stored_tokens()
    }

    pub fn capacity_tokens(&self) -> usize {
        self.inner.capacity_tokens()
    }

    pub fn storage_bytes(&self) -> Result<usize> {
        self.inner.storage_bytes()
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub(super) fn grow_full_capacity<B: Backend>(
        &mut self,
        capacity_tokens: usize,
        backend: &B,
    ) -> Result<()> {
        if self.inner.retention() == LagunaKvRetention::Sliding
            || capacity_tokens <= self.inner.capacity_tokens()
        {
            return Ok(());
        }
        if !backend.grow_laguna_fp8_kv_cache(&mut self.inner, capacity_tokens)? {
            return Err(Error::backend("Laguna FP8 KV growth requires native Metal"));
        }
        Ok(())
    }
}

/// Executes one complete Laguna attention stage without leaving the shared
/// Metal batch.
///
/// Input is the pre-attention normalized hidden state `[B,T,3072]`. The
/// backend encodes Q/K/V/gate projections, Q/K RMSNorm+RoPE, fused causal GQA,
/// FP8 cache append, and the output projection in order. The result remains
/// `[B,T,3072]` and device-resident.
pub fn forward_attention<B: Backend>(
    config: &LagunaConfig,
    layer_index: usize,
    hidden_states: &DeviceValue,
    weights: &LagunaDeviceAttentionWeights,
    rope_tables: &LagunaDeviceRopeTables,
    cache: &mut LagunaAttentionCache,
    backend: &B,
) -> Result<DeviceValue> {
    if cache.layer_index != layer_index {
        return Err(Error::cache(format!(
            "Laguna layer {layer_index} received KV cache for layer {}",
            cache.layer_index
        )));
    }
    if hidden_states.dtype() != DType::F32 {
        return Err(Error::backend(format!(
            "Laguna attention hidden states must be F32, got {:?}",
            hidden_states.dtype()
        )));
    }
    let [batch, token_count, hidden_size] = hidden_states.dims() else {
        return Err(Error::model(format!(
            "Laguna attention input must be [B,T,3072], got {:?}",
            hidden_states.dims()
        )));
    };
    if *batch != cache.inner.batch() || *token_count == 0 || *hidden_size != config.hidden_size {
        return Err(Error::model(format!(
            "Laguna attention input/cache mismatch: input={:?}, cache_batch={}",
            hidden_states.dims(),
            cache.inner.batch()
        )));
    }
    let query_heads = config
        .query_heads(layer_index)
        .ok_or_else(|| Error::config(format!("Laguna layer {layer_index} is out of range")))?;
    validate_weight_shapes(weights, config.hidden_size, query_heads)?;
    let position_offset = cache.inner.total_tokens();

    let projections = backend
        .laguna_attention_projections_device(
            &weights.query,
            &weights.key,
            &weights.value,
            &weights.gate,
            hidden_states,
        )?
        .ok_or_else(|| Error::backend("Laguna Q/K/V/gate projections require native Metal"))?;
    let query = projections.query;
    let key = projections.key;
    let value = projections.value;
    let gate = projections.gate;

    let rope_table = match config.attention_kind(layer_index) {
        Some(LagunaAttentionKind::FullAttention) => &rope_tables.full_attention,
        Some(LagunaAttentionKind::SlidingAttention) => &rope_tables.sliding_attention,
        None => {
            return Err(Error::config(format!(
                "Laguna layer {layer_index} is outside the attention schedule"
            )))
        }
    };
    let (query, key) = backend
        .laguna_qk_rms_norm_rope_pair_device(
            &query,
            &key,
            &weights.query_norm,
            &weights.key_norm,
            config.rms_norm_eps as f32,
            position_offset,
            rope_table,
        )?
        .ok_or_else(|| Error::backend("Laguna fused Q/K RMSNorm/RoPE requires native Metal"))?;
    let context = required(
        "Laguna gated GQA",
        backend.laguna_gated_gqa_attention_device(&query, &key, &value, &gate, &mut cache.inner)?,
    )?
    .reshape(vec![*batch, *token_count, query_heads * HEAD_DIM])?;
    required(
        "Laguna attention output projection",
        backend.bf16_linear_device(&weights.output, &context)?,
    )
}

fn cache_spec(
    attention_kind: LagunaAttentionKind,
    full_capacity_tokens: usize,
    sliding_window: usize,
) -> (LagunaKvRetention, usize) {
    match attention_kind {
        LagunaAttentionKind::FullAttention => (LagunaKvRetention::Full, full_capacity_tokens),
        LagunaAttentionKind::SlidingAttention => (LagunaKvRetention::Sliding, sliding_window),
    }
}

fn validate_weight_shapes(
    weights: &LagunaDeviceAttentionWeights,
    hidden_size: usize,
    query_heads: usize,
) -> Result<()> {
    let query_width = query_heads
        .checked_mul(HEAD_DIM)
        .ok_or_else(|| Error::model("Laguna query width overflow"))?;
    let matrices: [(&str, &DeviceBf16Matrix, usize, usize); 5] = [
        ("query", &weights.query, query_width, hidden_size),
        ("key", &weights.key, KV_HEADS * HEAD_DIM, hidden_size),
        ("value", &weights.value, KV_HEADS * HEAD_DIM, hidden_size),
        ("gate", &weights.gate, query_heads, hidden_size),
        ("output", &weights.output, hidden_size, query_width),
    ];
    for (label, matrix, rows, columns) in matrices {
        if matrix.rows() != rows || matrix.columns() != columns {
            return Err(Error::weights(format!(
                "Laguna {label} matrix must be [{rows},{columns}], got [{},{}]",
                matrix.rows(),
                matrix.columns()
            )));
        }
    }
    if weights.query_norm.dims() != [HEAD_DIM] || weights.key_norm.dims() != [HEAD_DIM] {
        return Err(Error::weights(format!(
            "Laguna Q/K norm weights must be [{HEAD_DIM}], got {:?}/{:?}",
            weights.query_norm.dims(),
            weights.key_norm.dims()
        )));
    }
    Ok(())
}

fn required(label: &str, value: Option<DeviceValue>) -> Result<DeviceValue> {
    value.ok_or_else(|| Error::backend(format!("{label} requires native Metal")))
}

#[cfg(test)]
mod tests {
    use super::cache_spec;
    use backend::LagunaKvRetention;
    use config::LagunaAttentionKind;

    #[test]
    fn cache_spec_keeps_global_context_and_bounds_sliding_context() {
        assert_eq!(
            cache_spec(LagunaAttentionKind::FullAttention, 32_768, 512),
            (LagunaKvRetention::Full, 32_768)
        );
        assert_eq!(
            cache_spec(LagunaAttentionKind::SlidingAttention, 32_768, 512),
            (LagunaKvRetention::Sliding, 512)
        );
    }
}
