use std::path::Path;

use backend::{
    Backend, DeviceRopeTable, DeviceValue, GgufExpertQuant, LagunaF16KvCache, LagunaKvRetention,
};
use common::{Error, F32Tensor, Result};
use config::{LagunaAttentionKind, LagunaConfig};
use gguf::{GgmlType, GgufQuantBlockKind, GgufTensorAdvice, GgufTensorInfo};

use super::{
    LagunaGgufDense, LagunaGgufIndex, LagunaGgufLayer, LagunaGgufMlp, LagunaGgufMoe,
    LagunaTokenOutput,
};

/// Layers per command-buffer submission.
///
/// Submitting each layer as it is encoded lets the GPU start on it while the CPU
/// encodes the next one. Measured on an M4 Max at decode, throughput falls off
/// monotonically as this grows — 24 layers per submission costs about 16% — so
/// the extra submissions pay for themselves. Submitting more often than once per
/// layer measured flat.
const LAYER_SUBMISSION_CHUNK: usize = 1;

#[derive(Debug)]
pub struct LagunaGgufModel {
    config: LagunaConfig,
    index: LagunaGgufIndex,
    prepared: PreparedWeights,
}

#[derive(Debug)]
struct PreparedWeights {
    final_norm: F32Tensor,
    full_rope: DeviceRopeTable,
    sliding_rope: DeviceRopeTable,
    layers: Vec<PreparedLayer>,
}

#[derive(Debug)]
struct PreparedLayer {
    input_norm: F32Tensor,
    post_attention_norm: F32Tensor,
    query_norm: F32Tensor,
    key_norm: F32Tensor,
    moe: Option<PreparedMoe>,
}

#[derive(Debug)]
struct PreparedMoe {
    router: F32Tensor,
    correction_bias: Vec<f32>,
}

#[derive(Debug)]
struct AttentionCache {
    layer_index: usize,
    inner: LagunaF16KvCache,
}

#[derive(Debug)]
pub struct LagunaGgufSession {
    batch: usize,
    context_capacity: usize,
    attention_caches: Vec<AttentionCache>,
}

impl LagunaGgufModel {
    pub fn open<B: Backend>(
        gguf_path: impl AsRef<Path>,
        config: LagunaConfig,
        backend: &B,
    ) -> Result<Self> {
        if !backend.device_values_supported() {
            return Err(Error::backend(
                "Antirez Laguna GGUF inference requires native device-resident Metal execution",
            ));
        }
        let index = LagunaGgufIndex::open(gguf_path, &config)?;
        apply_memory_advice(&index)?;
        let view_report = backend
            .prepare_laguna_gguf_views(
                index.mapped_bytes(),
                index.tensor_data_offset()?,
                index.max_tensor_storage_byte_len()?,
            )?
            .ok_or_else(|| Error::backend("Laguna Q2/Q3 GGUF model views require native Metal"))?;
        tracing::debug!(
            target: "inferno::laguna",
            view_count = view_report.view_count,
            model_gb = view_report.model_bytes as f64 / 1024_f64.powi(3),
            mapped_view_gb = view_report.view_bytes as f64 / 1024_f64.powi(3),
            max_view_gb = view_report.max_view_bytes as f64 / 1024_f64.powi(3),
            warmup_samples = view_report.warmup_samples,
            "prepared persistent Laguna Q2/Q3 Metal model views"
        );
        let prepared = PreparedWeights::new(&index, &config, backend)?;
        Ok(Self {
            config,
            index,
            prepared,
        })
    }

    pub fn config(&self) -> &LagunaConfig {
        &self.config
    }

    pub fn new_session<B: Backend>(
        &self,
        batch: usize,
        context_capacity: usize,
        backend: &B,
    ) -> Result<LagunaGgufSession> {
        self.validate_session_shape(batch, context_capacity)?;
        Ok(LagunaGgufSession {
            batch,
            context_capacity,
            attention_caches: self.prepare_attention_caches(batch, context_capacity, backend)?,
        })
    }

    pub fn prepare_session<B: Backend>(
        &self,
        session: &mut LagunaGgufSession,
        batch: usize,
        context_capacity: usize,
        backend: &B,
    ) -> Result<()> {
        self.validate_session_shape(batch, context_capacity)?;
        if session.batch != batch {
            return Err(Error::runtime(format!(
                "Laguna GGUF session batch {} cannot be reused for batch {batch}",
                session.batch
            )));
        }
        if context_capacity <= session.context_capacity {
            session.reset_sequence();
            return Ok(());
        }
        session.attention_caches =
            self.prepare_attention_caches(batch, context_capacity, backend)?;
        session.context_capacity = context_capacity;
        Ok(())
    }

    pub fn grow_session_capacity<B: Backend>(
        &self,
        session: &mut LagunaGgufSession,
        context_capacity: usize,
        backend: &B,
    ) -> Result<()> {
        self.validate_session_shape(session.batch, context_capacity)?;
        if context_capacity <= session.context_capacity {
            return Ok(());
        }
        for cache in &mut session.attention_caches {
            if cache.inner.retention() == LagunaKvRetention::Full
                && !backend.grow_laguna_f16_kv_cache(&mut cache.inner, context_capacity)?
            {
                return Err(Error::backend(
                    "Laguna GGUF F16 KV growth requires native Metal",
                ));
            }
        }
        session.context_capacity = context_capacity;
        Ok(())
    }

    pub fn forward_next_token<B: Backend>(
        &self,
        session: &mut LagunaGgufSession,
        token_ids: &[u32],
        backend: &B,
    ) -> Result<LagunaTokenOutput> {
        let hidden = self.forward_hidden_states(session, token_ids, backend)?;
        let last = required(
            "Laguna GGUF last-token selection",
            backend.select_last_token_device(&hidden)?,
        )?;
        let normalized = required(
            "Laguna GGUF final RMSNorm",
            backend.rms_norm_device(
                &last,
                &self.prepared.final_norm,
                self.config.rms_norm_eps as f32,
            )?,
        )?;
        let output = self.q8_bytes(&self.index.root.output)?;
        let (token_id, token_score) = backend
            .laguna_q8_0_matvec_argmax_device(
                output,
                &normalized,
                self.config.hidden_size,
                self.config.vocab_size,
            )?
            .ok_or_else(|| {
                Error::backend("Laguna GGUF Q8_0 output-head argmax requires native Metal")
            })?;
        session.validate_after_forward(1)?;
        Ok(LagunaTokenOutput {
            token_id,
            token_score,
        })
    }

    pub fn prefill_chunk<B: Backend>(
        &self,
        session: &mut LagunaGgufSession,
        token_ids: &[u32],
        backend: &B,
    ) -> Result<()> {
        let hidden = self.forward_hidden_states(session, token_ids, backend)?;
        backend.device_submit()?;
        drop(hidden);
        Ok(())
    }

    fn forward_hidden_states<B: Backend>(
        &self,
        session: &mut LagunaGgufSession,
        token_ids: &[u32],
        backend: &B,
    ) -> Result<DeviceValue> {
        validate_token_ids(&self.config, token_ids)?;
        session.validate_before_forward(self.config.num_hidden_layers, token_ids.len())?;
        let embedding = self.q8_bytes(&self.index.root.embedding)?;
        let mut hidden = required(
            "Laguna GGUF embedding",
            backend.q8_0_embedding_device(
                embedding,
                token_ids,
                &[session.batch, token_ids.len()],
                self.config.vocab_size,
                self.config.hidden_size,
            )?,
        )?;

        for layer_index in 0..self.config.num_hidden_layers {
            let layer = self.index.layers.get(layer_index).ok_or_else(|| {
                Error::weights(format!("missing Laguna GGUF layer {layer_index}"))
            })?;
            let prepared = self.prepared.layers.get(layer_index).ok_or_else(|| {
                Error::weights(format!("missing prepared Laguna GGUF layer {layer_index}"))
            })?;
            let cache = session
                .attention_caches
                .get_mut(layer_index)
                .ok_or_else(|| Error::cache(format!("missing Laguna GGUF cache {layer_index}")))?;
            hidden = self.forward_layer(&hidden, layer, prepared, cache, backend)?;
            if should_submit_after_layer(layer_index, self.config.num_hidden_layers) {
                backend.device_submit()?;
            }
        }
        Ok(hidden)
    }

    fn forward_layer<B: Backend>(
        &self,
        hidden: &DeviceValue,
        layer: &LagunaGgufLayer,
        prepared: &PreparedLayer,
        cache: &mut AttentionCache,
        backend: &B,
    ) -> Result<DeviceValue> {
        let row_count = hidden.element_count()? / self.config.hidden_size;
        let normalized = required(
            "Laguna GGUF attention RMSNorm",
            backend.rms_norm_device(
                hidden,
                &prepared.input_norm,
                self.config.rms_norm_eps as f32,
            )?,
        )?;
        let attention = &layer.attention;
        let query_heads = self
            .config
            .query_heads(layer.layer_index)
            .ok_or_else(|| Error::config("Laguna GGUF layer has no query-head count"))?;
        let query_width = query_heads
            .checked_mul(self.config.head_dim)
            .ok_or_else(|| Error::model("Laguna GGUF query width overflow"))?;
        let kv_width = self.config.key_value_width();
        let [query, key, value, gate] = match backend.laguna_q8_0_attention_projections_device(
            self.q8_bytes(&attention.query)?,
            self.q8_bytes(&attention.key)?,
            self.q8_bytes(&attention.value)?,
            self.q8_bytes(&attention.gate)?,
            &normalized,
            row_count,
            self.config.hidden_size,
            query_width,
            kv_width,
            kv_width,
            query_heads,
        )? {
            Some(projections) => projections,
            None => {
                let (query, key) = required_pair(
                    "Laguna GGUF Q/K projection",
                    backend.q8_0_matvec_pair_device(
                        self.q8_bytes(&attention.query)?,
                        self.q8_bytes(&attention.key)?,
                        &normalized,
                        row_count,
                        self.config.hidden_size,
                        query_width,
                        kv_width,
                    )?,
                )?;
                let (value, gate) = required_pair(
                    "Laguna GGUF V/gate projection",
                    backend.q8_0_matvec_pair_device(
                        self.q8_bytes(&attention.value)?,
                        self.q8_bytes(&attention.gate)?,
                        &normalized,
                        row_count,
                        self.config.hidden_size,
                        kv_width,
                        query_heads,
                    )?,
                )?;
                [query, key, value, gate]
            }
        };
        let [batch, tokens, _] = hidden.dims() else {
            return Err(Error::model(format!(
                "Laguna GGUF hidden state must be [B,T,H], got {:?}",
                hidden.dims()
            )));
        };
        let query = query.reshape(vec![*batch, *tokens, query_heads, self.config.head_dim])?;
        let key = key.reshape(vec![
            *batch,
            *tokens,
            self.config.num_key_value_heads,
            self.config.head_dim,
        ])?;
        let value = value.reshape(vec![
            *batch,
            *tokens,
            self.config.num_key_value_heads,
            self.config.head_dim,
        ])?;
        let gate = gate.reshape(vec![*batch, *tokens, query_heads])?;
        let rope = match self.config.attention_kind(layer.layer_index) {
            Some(LagunaAttentionKind::FullAttention) => &self.prepared.full_rope,
            Some(LagunaAttentionKind::SlidingAttention) => &self.prepared.sliding_rope,
            None => return Err(Error::config("Laguna GGUF attention layer is out of range")),
        };
        let position = cache.inner.total_tokens();
        let (query, key) = required_pair(
            "Laguna GGUF Q/K RMSNorm and RoPE",
            backend.laguna_qk_rms_norm_rope_pair_device(
                &query,
                &key,
                &prepared.query_norm,
                &prepared.key_norm,
                self.config.rms_norm_eps as f32,
                position,
                rope,
            )?,
        )?;
        let context = required(
            "Laguna GGUF F16 attention",
            backend.laguna_gated_gqa_f16_attention_device(
                &query,
                &key,
                &value,
                &gate,
                &mut cache.inner,
            )?,
        )?
        .reshape(vec![*batch, *tokens, query_width])?;
        let post_attention = required(
            "Laguna GGUF attention output and residual",
            backend.q8_0_matvec_add_device(
                self.q8_bytes(&attention.output)?,
                &context,
                hidden,
                row_count,
                query_width,
                self.config.hidden_size,
            )?,
        )?;
        let mlp_input = required(
            "Laguna GGUF MLP RMSNorm",
            backend.rms_norm_device(
                &post_attention,
                &prepared.post_attention_norm,
                self.config.rms_norm_eps as f32,
            )?,
        )?;
        match (&layer.mlp, &prepared.moe) {
            (LagunaGgufMlp::Dense(dense), None) => {
                self.forward_dense(dense, &mlp_input, &post_attention, row_count, backend)
            }
            (LagunaGgufMlp::Moe(moe), Some(prepared_moe)) => self.forward_moe(
                moe,
                prepared_moe,
                &mlp_input,
                &post_attention,
                row_count,
                backend,
            ),
            _ => Err(Error::weights(format!(
                "Laguna GGUF layer {} prepared MLP kind mismatch",
                layer.layer_index
            ))),
        }
    }

    fn forward_dense<B: Backend>(
        &self,
        dense: &LagunaGgufDense,
        input: &DeviceValue,
        residual: &DeviceValue,
        row_count: usize,
        backend: &B,
    ) -> Result<DeviceValue> {
        let activated = required(
            "Laguna GGUF dense gate/up",
            backend.q8_0_gate_up_swiglu_device(
                self.q8_bytes(&dense.gate)?,
                self.q8_bytes(&dense.up)?,
                input,
                row_count,
                self.config.hidden_size,
                self.config.intermediate_size,
            )?,
        )?;
        required(
            "Laguna GGUF dense down and residual",
            backend.q8_0_matvec_add_device(
                self.q8_bytes(&dense.down)?,
                &activated,
                residual,
                row_count,
                self.config.intermediate_size,
                self.config.hidden_size,
            )?,
        )
    }

    fn forward_moe<B: Backend>(
        &self,
        moe: &LagunaGgufMoe,
        prepared: &PreparedMoe,
        input: &DeviceValue,
        residual: &DeviceValue,
        row_count: usize,
        backend: &B,
    ) -> Result<DeviceValue> {
        let flat_input = input.reshape(vec![row_count, self.config.hidden_size])?;
        let logits = required(
            "Laguna GGUF router",
            backend.linear_f32_device(&flat_input, &prepared.router)?,
        )?;
        let routing = backend
            .moe_router_topk_resident_device(
                &logits,
                &prepared.correction_bias,
                self.config.num_experts_per_tok,
                self.config.norm_topk_prob,
                self.config.moe_routed_scaling_factor as f32,
            )?
            .ok_or_else(|| Error::backend("Laguna GGUF routing requires native Metal"))?;
        let routed_gate = self.index.quantized_storage(&moe.routed_gate)?;
        let routed_up = self.index.quantized_storage(&moe.routed_up)?;
        let routed_down = self.index.quantized_storage(&moe.routed_down)?;
        if routed_gate.block != routed_up.block || routed_gate.block != routed_down.block {
            return Err(Error::weights(
                "Laguna GGUF routed gate/up/down quantization types disagree",
            ));
        }
        let quant = match routed_gate.block {
            GgufQuantBlockKind::Q2K => GgufExpertQuant::Q2K,
            GgufQuantBlockKind::Q3K => GgufExpertQuant::Q3K,
            GgufQuantBlockKind::Q8_0 => {
                return Err(Error::weights(
                    "Laguna GGUF routed experts must be Q2_K or Q3_K",
                ))
            }
        };
        let routed = required(
            "Laguna GGUF routed experts",
            backend.laguna_gguf_moe_device(
                routed_gate.bytes,
                routed_up.bytes,
                routed_down.bytes,
                quant,
                &flat_input,
                &routing,
                self.config.hidden_size,
                self.config.moe_intermediate_size,
                self.config.hidden_size,
            )?,
        )?;
        let shared_activated = required(
            "Laguna GGUF shared expert gate/up",
            backend.q8_0_gate_up_swiglu_device(
                self.q8_bytes(&moe.shared.gate)?,
                self.q8_bytes(&moe.shared.up)?,
                &flat_input,
                row_count,
                self.config.hidden_size,
                self.config.shared_expert_intermediate_size,
            )?,
        )?;
        required(
            "Laguna GGUF shared expert down, routed combine, and residual",
            backend.laguna_q8_0_matvec_add2_device(
                self.q8_bytes(&moe.shared.down)?,
                &shared_activated,
                &routed,
                residual,
                row_count,
                self.config.shared_expert_intermediate_size,
                self.config.hidden_size,
            )?,
        )
    }

    fn q8_bytes<'a>(&'a self, tensor: &'a GgufTensorInfo) -> Result<&'a [u8]> {
        let storage = self.index.quantized_storage(tensor)?;
        if storage.block != GgufQuantBlockKind::Q8_0 {
            return Err(Error::weights(format!(
                "Laguna GGUF tensor {} must be Q8_0, got {}",
                tensor.name, storage.block
            )));
        }
        Ok(storage.bytes)
    }

    fn prepare_attention_caches<B: Backend>(
        &self,
        batch: usize,
        context_capacity: usize,
        backend: &B,
    ) -> Result<Vec<AttentionCache>> {
        (0..self.config.num_hidden_layers)
            .map(|layer_index| {
                let (retention, capacity) = match self.config.attention_kind(layer_index) {
                    Some(LagunaAttentionKind::FullAttention) => {
                        (LagunaKvRetention::Full, context_capacity)
                    }
                    Some(LagunaAttentionKind::SlidingAttention) => {
                        (LagunaKvRetention::Sliding, self.config.sliding_window)
                    }
                    None => {
                        return Err(Error::config(format!(
                            "Laguna GGUF layer {layer_index} is out of range"
                        )))
                    }
                };
                let inner = backend
                    .prepare_laguna_f16_kv_cache(batch, capacity, retention)?
                    .ok_or_else(|| {
                        Error::backend("Laguna GGUF F16 KV cache requires native Metal")
                    })?;
                Ok(AttentionCache { layer_index, inner })
            })
            .collect()
    }

    fn validate_session_shape(&self, batch: usize, context_capacity: usize) -> Result<()> {
        if batch != 1 {
            return Err(Error::runtime(format!(
                "Laguna GGUF runtime currently supports batch 1, got {batch}"
            )));
        }
        if context_capacity == 0 || context_capacity > self.config.max_position_embeddings {
            return Err(Error::runtime(format!(
                "Laguna GGUF context capacity must be within 1..={}, got {context_capacity}",
                self.config.max_position_embeddings
            )));
        }
        Ok(())
    }
}

impl PreparedWeights {
    fn new<B: Backend>(
        index: &LagunaGgufIndex,
        config: &LagunaConfig,
        backend: &B,
    ) -> Result<Self> {
        let full_rope = prepare_rope(config, 0, backend)?;
        let sliding_rope = prepare_rope(config, 1, backend)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in &index.layers {
            let moe = match &layer.mlp {
                LagunaGgufMlp::Dense(_) => None,
                LagunaGgufMlp::Moe(moe) => Some(PreparedMoe {
                    router: F32Tensor::new(
                        index.f32_values(&moe.router)?,
                        [config.num_experts, config.hidden_size],
                    )?,
                    correction_bias: index.f32_values(&moe.correction_bias)?,
                }),
            };
            layers.push(PreparedLayer {
                input_norm: vector(index, &layer.attention.input_norm, config.hidden_size)?,
                post_attention_norm: vector(index, &layer.post_attention_norm, config.hidden_size)?,
                query_norm: vector(index, &layer.attention.query_norm, config.head_dim)?,
                key_norm: vector(index, &layer.attention.key_norm, config.head_dim)?,
                moe,
            });
        }
        Ok(Self {
            final_norm: vector(index, &index.root.final_norm, config.hidden_size)?,
            full_rope,
            sliding_rope,
            layers,
        })
    }
}

impl LagunaGgufSession {
    pub fn position(&self) -> Result<usize> {
        let Some(first) = self.attention_caches.first() else {
            return Err(Error::cache("Laguna GGUF session has no attention caches"));
        };
        let position = first.inner.total_tokens();
        if self
            .attention_caches
            .iter()
            .any(|cache| cache.inner.total_tokens() != position)
        {
            return Err(Error::cache(
                "Laguna GGUF attention caches disagree on sequence position",
            ));
        }
        Ok(position)
    }

    pub fn context_capacity(&self) -> usize {
        self.context_capacity
    }

    pub fn reset_sequence(&mut self) {
        for cache in &mut self.attention_caches {
            cache.inner.reset();
        }
    }

    fn validate_before_forward(&self, layer_count: usize, token_count: usize) -> Result<()> {
        if self.attention_caches.len() != layer_count
            || self
                .attention_caches
                .iter()
                .enumerate()
                .any(|(index, cache)| cache.layer_index != index)
        {
            return Err(Error::cache(
                "Laguna GGUF session cache layer ordering is invalid",
            ));
        }
        let end = self
            .position()?
            .checked_add(token_count)
            .ok_or_else(|| Error::cache("Laguna GGUF sequence position overflow"))?;
        if end > self.context_capacity {
            return Err(Error::cache(format!(
                "Laguna GGUF sequence end {end} exceeds capacity {}",
                self.context_capacity
            )));
        }
        Ok(())
    }

    fn validate_after_forward(&self, output_rows: usize) -> Result<()> {
        if output_rows != self.batch {
            return Err(Error::backend(format!(
                "Laguna GGUF output row count {output_rows} does not match batch {}",
                self.batch
            )));
        }
        let _ = self.position()?;
        Ok(())
    }
}

fn apply_memory_advice(index: &LagunaGgufIndex) -> Result<()> {
    for tensor in index.layers.iter().flat_map(layer_tensors).chain([
        &index.root.embedding,
        &index.root.final_norm,
        &index.root.output,
    ]) {
        let advice = if tensor.name.contains("_exps.weight") {
            GgufTensorAdvice::Random
        } else {
            GgufTensorAdvice::WillNeed
        };
        index.advise(tensor, advice)?;
    }
    Ok(())
}

fn layer_tensors(layer: &LagunaGgufLayer) -> Vec<&GgufTensorInfo> {
    let attention = &layer.attention;
    let mut tensors = vec![
        &attention.input_norm,
        &attention.query,
        &attention.key,
        &attention.value,
        &attention.gate,
        &attention.query_norm,
        &attention.key_norm,
        &attention.output,
        &layer.post_attention_norm,
    ];
    match &layer.mlp {
        LagunaGgufMlp::Dense(dense) => {
            tensors.extend([&dense.gate, &dense.up, &dense.down]);
        }
        LagunaGgufMlp::Moe(moe) => {
            tensors.extend([
                &moe.router,
                &moe.correction_bias,
                &moe.routed_gate,
                &moe.routed_up,
                &moe.routed_down,
                &moe.shared.gate,
                &moe.shared.up,
                &moe.shared.down,
            ]);
        }
    }
    tensors
}

fn prepare_rope<B: Backend>(
    config: &LagunaConfig,
    layer_index: usize,
    backend: &B,
) -> Result<DeviceRopeTable> {
    let frequencies = config.rope_frequencies(layer_index)?;
    backend
        .prepare_rope_table(
            &frequencies.inverse_frequencies,
            frequencies.rotary_dim,
            frequencies.attention_factor,
        )?
        .ok_or_else(|| Error::backend("Laguna GGUF RoPE table requires native Metal"))
}

fn vector(index: &LagunaGgufIndex, tensor: &GgufTensorInfo, len: usize) -> Result<F32Tensor> {
    if tensor.ty != GgmlType::F32 || tensor.dims != [len as u64] {
        return Err(Error::weights(format!(
            "Laguna GGUF vector {} must be F32 [{len}], got {} {:?}",
            tensor.name, tensor.ty, tensor.dims
        )));
    }
    F32Tensor::new(index.f32_values(tensor)?, [len])
}

fn validate_token_ids(config: &LagunaConfig, token_ids: &[u32]) -> Result<()> {
    if token_ids.is_empty() {
        return Err(Error::model(
            "Laguna GGUF forward requires at least one token ID",
        ));
    }
    if let Some(token_id) = token_ids
        .iter()
        .copied()
        .find(|token_id| *token_id as usize >= config.vocab_size)
    {
        return Err(Error::model(format!(
            "Laguna GGUF token ID {token_id} is outside vocabulary {}",
            config.vocab_size
        )));
    }
    Ok(())
}

fn required(label: &str, value: Option<DeviceValue>) -> Result<DeviceValue> {
    value.ok_or_else(|| Error::backend(format!("{label} requires native Metal")))
}

fn required_pair(
    label: &str,
    value: Option<(DeviceValue, DeviceValue)>,
) -> Result<(DeviceValue, DeviceValue)> {
    value.ok_or_else(|| Error::backend(format!("{label} requires native Metal")))
}

fn should_submit_after_layer(layer_index: usize, layer_count: usize) -> bool {
    let completed_layers = layer_index + 1;
    completed_layers < layer_count && completed_layers.is_multiple_of(LAYER_SUBMISSION_CHUNK)
}

#[cfg(test)]
mod tests {
    use super::should_submit_after_layer;

    #[test]
    fn submits_complete_laguna_layer_chunks_but_not_the_final_layer() {
        let submission_layers = (0..48)
            .filter(|layer_index| should_submit_after_layer(*layer_index, 48))
            .collect::<Vec<_>>();

        // The final layer is left for the caller to submit alongside the output
        // head, so it must not appear here.
        assert_eq!(
            submission_layers,
            (0..47).step_by(super::LAYER_SUBMISSION_CHUNK).collect::<Vec<_>>()
        );
    }
}
