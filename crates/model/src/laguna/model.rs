use std::path::Path;

use backend::{Backend, DeviceValue};
use common::{Error, Result};
use config::LagunaConfig;

use super::{
    forward_layer, LagunaAttentionCache, LagunaDeviceWeights, LagunaExpertCache,
    LagunaExpertCacheMetrics, LagunaExpertPrefetchPool, LagunaWeightIndex, LagunaWeightSummary,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LagunaTokenOutput {
    pub token_id: u32,
    pub token_score: f32,
}

/// Immutable Laguna S 2.1 INT4 model state.
///
/// Always-used BF16 matrices are prepared once on Metal. Routed INT4 experts
/// remain indexed Safetensors ranges and become direct Metal views only after
/// the router selects them.
#[derive(Debug)]
pub struct LagunaModel {
    config: LagunaConfig,
    index: LagunaWeightIndex,
    weights: LagunaDeviceWeights,
    expert_prefetch_pool: LagunaExpertPrefetchPool,
}

impl LagunaModel {
    pub fn open<B: Backend>(
        model_dir: impl AsRef<Path>,
        config: LagunaConfig,
        backend: &B,
    ) -> Result<Self> {
        if !backend.device_values_supported() {
            return Err(Error::backend(
                "Laguna S 2.1 INT4 inference requires native device-resident Metal execution",
            ));
        }
        let index = LagunaWeightIndex::open(model_dir, &config)?;
        let expert_prefetch_pool = LagunaExpertPrefetchPool::new()?;
        let weights = LagunaDeviceWeights::prepare(&index, &config, backend)?;
        Ok(Self {
            config,
            index,
            weights,
            expert_prefetch_pool,
        })
    }

    pub fn config(&self) -> &LagunaConfig {
        &self.config
    }

    pub fn weight_summary(&self) -> LagunaWeightSummary {
        self.index.summary
    }

    pub fn prepared_matrix_bytes(&self) -> u64 {
        self.weights.prepared_matrix_bytes
    }

    pub fn new_session<B: Backend>(
        &self,
        batch: usize,
        context_capacity: usize,
        expert_cache_capacity: usize,
        backend: &B,
    ) -> Result<LagunaSession> {
        self.validate_session_shape(batch, context_capacity)?;
        let attention_caches = self.prepare_attention_caches(batch, context_capacity, backend)?;
        Ok(LagunaSession {
            batch,
            context_capacity,
            attention_caches,
            expert_cache: LagunaExpertCache::new(
                expert_cache_capacity,
                self.config.num_experts_per_tok,
                self.index.summary.bytes_per_expert,
                self.index.summary.routed_layer_count,
                self.index.summary.experts_per_layer,
            )?,
        })
    }

    /// Starts a fresh sequence while retaining the session's routed-expert
    /// cache. KV buffers are reused when large enough and replaced only when a
    /// later request needs a larger context.
    pub fn prepare_session<B: Backend>(
        &self,
        session: &mut LagunaSession,
        batch: usize,
        context_capacity: usize,
        backend: &B,
    ) -> Result<()> {
        self.validate_session_shape(batch, context_capacity)?;
        if session.batch != batch {
            return Err(Error::runtime(format!(
                "Laguna session batch {} cannot be reused for batch {batch}",
                session.batch
            )));
        }
        if context_capacity <= session.context_capacity {
            session.reset_sequence();
            return Ok(());
        }

        let attention_caches = self.prepare_attention_caches(batch, context_capacity, backend)?;
        session.attention_caches = attention_caches;
        session.context_capacity = context_capacity;
        Ok(())
    }

    /// Expands the active sequence's full-attention KV buffers without
    /// discarding already cached tokens. Sliding-window buffers stay fixed.
    pub fn grow_session_capacity<B: Backend>(
        &self,
        session: &mut LagunaSession,
        context_capacity: usize,
        backend: &B,
    ) -> Result<()> {
        self.validate_session_shape(session.batch, context_capacity)?;
        if context_capacity <= session.context_capacity {
            return Ok(());
        }
        for cache in &mut session.attention_caches {
            cache.grow_full_capacity(context_capacity, backend)?;
        }
        session.context_capacity = context_capacity;
        Ok(())
    }

    /// Appends one prompt/decode chunk and returns the greedy next token.
    ///
    /// Input IDs have logical shape `[1,T]`. Embeddings and hidden states are
    /// `[1,T,3072]`; the last hidden row is projected to `[1,vocab_size]`.
    /// The argmax sink is the only synchronization after the layer stack.
    pub fn forward_next_token<B: Backend>(
        &self,
        session: &mut LagunaSession,
        token_ids: &[u32],
        backend: &B,
    ) -> Result<LagunaTokenOutput> {
        let hidden_states = self.forward_hidden_states(session, token_ids, backend)?;

        let last_hidden = required(
            "Laguna last-token selection",
            backend.select_last_token_device(&hidden_states)?,
        )?;
        let normalized = required(
            "Laguna final RMSNorm",
            backend.rms_norm_device(
                &last_hidden,
                &self.weights.root.final_norm,
                self.config.rms_norm_eps as f32,
            )?,
        )?;
        let logits = required(
            "Laguna output projection",
            backend.bf16_linear_device(&self.weights.root.output, &normalized)?,
        )?
        .reshape(vec![session.batch, self.config.vocab_size])?;
        let (token_ids, token_scores) = backend
            .argmax_rows_f32_device(&logits, self.config.vocab_size)?
            .ok_or_else(|| Error::backend("Laguna greedy argmax requires native Metal"))?;
        let ([token_id], [token_score]) = (token_ids.as_slice(), token_scores.as_slice()) else {
            return Err(Error::backend(format!(
                "Laguna batch-1 argmax returned {}/{} results",
                token_ids.len(),
                token_scores.len()
            )));
        };
        session.validate_after_forward(token_ids.len())?;
        Ok(LagunaTokenOutput {
            token_id: *token_id,
            token_score: *token_score,
        })
    }

    /// Appends a bounded prompt chunk without running the final norm, output
    /// projection, or argmax. KV writes are submitted asynchronously so the
    /// next chunk can be encoded behind them on the same Metal queue.
    pub fn prefill_chunk<B: Backend>(
        &self,
        session: &mut LagunaSession,
        token_ids: &[u32],
        backend: &B,
    ) -> Result<()> {
        let hidden_states = self.forward_hidden_states(session, token_ids, backend)?;
        backend.device_submit()?;
        drop(hidden_states);
        Ok(())
    }

    fn forward_hidden_states<B: Backend>(
        &self,
        session: &mut LagunaSession,
        token_ids: &[u32],
        backend: &B,
    ) -> Result<DeviceValue> {
        validate_token_ids(&self.config, token_ids)?;
        session.validate_before_forward(self.config.num_hidden_layers, token_ids.len())?;

        let mut hidden_states = required(
            "Laguna embedding lookup",
            backend.bf16_embedding_device(
                &self.weights.root.embedding,
                token_ids,
                &[session.batch, token_ids.len()],
            )?,
        )?;
        for ((layer, cache), expected_index) in self
            .weights
            .layers
            .iter()
            .zip(&mut session.attention_caches)
            .zip(0..self.config.num_hidden_layers)
        {
            if layer.layer_index != expected_index || cache.layer_index() != expected_index {
                return Err(Error::model(format!(
                    "Laguna layer/cache order mismatch at position {expected_index}: layer={}, cache={}",
                    layer.layer_index,
                    cache.layer_index()
                )));
            }
            hidden_states = forward_layer(
                &self.config,
                &hidden_states,
                layer,
                &self.weights.rope,
                cache,
                &self.index,
                &mut session.expert_cache,
                &self.expert_prefetch_pool,
                backend,
            )?;
        }
        let _ = session.position()?;
        Ok(hidden_states)
    }

    fn validate_session_shape(&self, batch: usize, context_capacity: usize) -> Result<()> {
        if batch != 1 {
            return Err(Error::runtime(format!(
                "Laguna runtime currently supports batch 1, got {batch}"
            )));
        }
        if context_capacity == 0 || context_capacity > self.config.max_position_embeddings {
            return Err(Error::runtime(format!(
                "Laguna session context capacity must be within 1..={}, got {context_capacity}",
                self.config.max_position_embeddings
            )));
        }
        Ok(())
    }

    fn prepare_attention_caches<B: Backend>(
        &self,
        batch: usize,
        context_capacity: usize,
        backend: &B,
    ) -> Result<Vec<LagunaAttentionCache>> {
        let mut attention_caches = Vec::with_capacity(self.config.num_hidden_layers);
        for layer in &self.weights.layers {
            attention_caches.push(LagunaAttentionCache::prepare(
                &self.config,
                layer.layer_index,
                batch,
                context_capacity,
                layer.attention.key_scale,
                layer.attention.value_scale,
                backend,
            )?);
        }
        Ok(attention_caches)
    }
}

/// Mutable per-sequence state. It can be reset without discarding reusable
/// expert weights, so repeated turns avoid cold expert loads where possible.
#[derive(Debug)]
pub struct LagunaSession {
    batch: usize,
    context_capacity: usize,
    attention_caches: Vec<LagunaAttentionCache>,
    expert_cache: LagunaExpertCache,
}

impl LagunaSession {
    pub fn position(&self) -> Result<usize> {
        let Some(first) = self.attention_caches.first() else {
            return Err(Error::cache("Laguna session has no attention caches"));
        };
        let position = first.total_tokens();
        if self
            .attention_caches
            .iter()
            .any(|cache| cache.total_tokens() != position)
        {
            return Err(Error::cache(
                "Laguna attention caches disagree on the current sequence position",
            ));
        }
        Ok(position)
    }

    pub fn context_capacity(&self) -> usize {
        self.context_capacity
    }

    pub fn expert_cache_metrics(&self) -> LagunaExpertCacheMetrics {
        self.expert_cache.metrics()
    }

    pub fn resize_expert_cache_capacity(&mut self, capacity_experts: usize) -> Result<()> {
        self.expert_cache.resize_capacity(capacity_experts)
    }

    /// Starts a new sequence while retaining routed experts already in RAM.
    pub fn reset_sequence(&mut self) {
        for cache in &mut self.attention_caches {
            cache.reset();
        }
    }

    pub fn clear_expert_cache(&mut self) {
        self.expert_cache.clear();
    }

    fn validate_before_forward(&self, layer_count: usize, token_count: usize) -> Result<()> {
        if self.attention_caches.len() != layer_count {
            return Err(Error::cache(format!(
                "Laguna session has {} layer caches, expected {layer_count}",
                self.attention_caches.len()
            )));
        }
        let end = self
            .position()?
            .checked_add(token_count)
            .ok_or_else(|| Error::cache("Laguna sequence position overflow"))?;
        if end > self.context_capacity {
            return Err(Error::cache(format!(
                "Laguna sequence end {end} exceeds session capacity {}",
                self.context_capacity
            )));
        }
        Ok(())
    }

    fn validate_after_forward(&self, output_rows: usize) -> Result<()> {
        if output_rows != self.batch {
            return Err(Error::backend(format!(
                "Laguna output row count {output_rows} does not match batch {}",
                self.batch
            )));
        }
        let _ = self.position()?;
        Ok(())
    }
}

fn validate_token_ids(config: &LagunaConfig, token_ids: &[u32]) -> Result<()> {
    if token_ids.is_empty() {
        return Err(Error::model(
            "Laguna forward requires at least one token ID",
        ));
    }
    if let Some(token_id) = token_ids
        .iter()
        .copied()
        .find(|token_id| *token_id as usize >= config.vocab_size)
    {
        return Err(Error::model(format!(
            "Laguna token ID {token_id} is outside vocabulary size {}",
            config.vocab_size
        )));
    }
    Ok(())
}

fn required(label: &str, value: Option<DeviceValue>) -> Result<DeviceValue> {
    value.ok_or_else(|| Error::backend(format!("{label} requires native Metal")))
}
