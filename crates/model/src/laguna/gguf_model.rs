use std::path::Path;

use backend::{
    Backend, DeviceRopeTable, DeviceValue, GgufExpertQuant, GgufKQuant, LagunaF16KvCache,
    LagunaKvRetention,
};
use common::{Error, F32Tensor, Result};
use config::{LagunaAttentionKind, LagunaConfig};
use gguf::{GgmlType, GgufQuantBlockKind, GgufTensorAdvice, GgufTensorInfo};

use super::{
    LagunaGgufDense, LagunaGgufFlavor, LagunaGgufIndex, LagunaGgufLayer, LagunaGgufMlp,
    LagunaGgufMoe, LagunaTokenOutput,
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
    pub const fn flavor(&self) -> super::LagunaGgufFlavor {
        self.index.flavor()
    }

    pub fn open<B: Backend>(
        gguf_path: impl AsRef<Path>,
        config: LagunaConfig,
        backend: &B,
    ) -> Result<Self> {
        if !backend.device_values_supported() {
            return Err(Error::backend(
                "Laguna GGUF inference requires native device-resident Metal execution",
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
            .ok_or_else(|| Error::backend("Laguna GGUF model views require native Metal"))?;
        tracing::debug!(
            target: "inferno::laguna",
            view_count = view_report.view_count,
            model_gb = view_report.model_bytes as f64 / 1024_f64.powi(3),
            mapped_view_gb = view_report.view_bytes as f64 / 1024_f64.powi(3),
            max_view_gb = view_report.max_view_bytes as f64 / 1024_f64.powi(3),
            warmup_samples = view_report.warmup_samples,
            flavor = ?index.flavor(),
            "prepared persistent Laguna Metal model views"
        );
        let prepared = PreparedWeights::new(&index, &config, backend)?;
        if index.flavor() == LagunaGgufFlavor::Xs21Q4KM {
            let prepared_bytes = prepare_xs_dense_prefill_weights(&index, &config, backend)?;
            backend.device_flush()?;
            tracing::debug!(
                target: "inferno::laguna",
                prepared_gb = prepared_bytes as f64 / 1024_f64.powi(3),
                "prepared Laguna XS dense FP16 prefill weights"
            );
        }
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
            self.rms_norm_device(
                &last,
                &self.prepared.final_norm,
                1,
                self.config.rms_norm_eps as f32,
                backend,
            )?,
        )?;
        let (token_id, token_score) = match self.flavor() {
            LagunaGgufFlavor::S21Q2Q3 => {
                let output = self.q8_bytes(&self.index.root.output)?;
                backend
                    .laguna_q8_0_matvec_argmax_device(
                        output,
                        &normalized,
                        self.config.hidden_size,
                        self.config.vocab_size,
                    )?
                    .ok_or_else(|| {
                        Error::backend(
                            "Laguna S GGUF Q8_0 output-head argmax requires native Metal",
                        )
                    })?
            }
            LagunaGgufFlavor::Xs21Q4KM => {
                let scores = self
                    .xs_matvec(
                        &self.index.root.output,
                        &normalized,
                        1,
                        self.config.hidden_size,
                        self.config.vocab_size,
                        backend,
                    )?
                    .reshape(vec![self.config.vocab_size])?;
                backend
                    .argmax_f32_device(&scores)?
                    .ok_or_else(|| Error::backend("Laguna XS argmax requires native Metal"))?
            }
        };
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
        let token_shape = [session.batch, token_ids.len()];
        let mut hidden = match self.flavor() {
            LagunaGgufFlavor::S21Q2Q3 => required(
                "Laguna S GGUF embedding",
                backend.q8_0_embedding_device(
                    self.q8_bytes(&self.index.root.embedding)?,
                    token_ids,
                    &token_shape,
                    self.config.vocab_size,
                    self.config.hidden_size,
                )?,
            )?,
            LagunaGgufFlavor::Xs21Q4KM => required(
                "Laguna XS GGUF embedding",
                backend.q4_k_embedding_device(
                    self.k_bytes(&self.index.root.embedding)?.1,
                    token_ids,
                    &token_shape,
                    self.config.vocab_size,
                    self.config.hidden_size,
                )?,
            )?,
        };
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
            if row_is_prefill(&hidden)?
                && tracing::enabled!(
                    target: "inferno::metal::profile",
                    tracing::Level::TRACE
                )
            {
                backend.device_profile_boundary("laguna.prefill.mlp_tail")?;
            } else if should_submit_after_layer(layer_index, self.config.num_hidden_layers) {
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
            self.rms_norm_device(
                hidden,
                &prepared.input_norm,
                row_count,
                self.config.rms_norm_eps as f32,
                backend,
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
        let [query, key, value, gate] = match self.flavor() {
            LagunaGgufFlavor::S21Q2Q3 => {
                match backend.laguna_q8_0_attention_projections_device(
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
                            "Laguna S GGUF Q/K projection",
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
                            "Laguna S GGUF V/gate projection",
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
                }
            }
            LagunaGgufFlavor::Xs21Q4KM => {
                let (query_quant, query_weights) = self.k_bytes(&attention.query)?;
                let (key_quant, key_weights) = self.k_bytes(&attention.key)?;
                let (value_quant, value_weights) = self.k_bytes(&attention.value)?;
                let (gate_quant, gate_weights) = self.k_bytes(&attention.gate)?;
                if query_quant != GgufKQuant::Q4K
                    || key_quant != GgufKQuant::Q4K
                    || gate_quant != GgufKQuant::Q4K
                {
                    return Err(Error::weights(format!(
                        "Laguna XS Q/K/gate projections must be Q4_K, got {query_quant:?}/{key_quant:?}/{gate_quant:?}"
                    )));
                }
                match backend.laguna_xs_attention_projections_device(
                    query_weights,
                    key_weights,
                    value_weights,
                    value_quant,
                    gate_weights,
                    &normalized,
                    row_count,
                    self.config.hidden_size,
                    query_width,
                    kv_width,
                    kv_width,
                    query_heads,
                )? {
                    Some(projections) => projections,
                    None => [
                        self.xs_matvec(
                            &attention.query,
                            &normalized,
                            row_count,
                            self.config.hidden_size,
                            query_width,
                            backend,
                        )?,
                        self.xs_matvec(
                            &attention.key,
                            &normalized,
                            row_count,
                            self.config.hidden_size,
                            kv_width,
                            backend,
                        )?,
                        self.xs_matvec(
                            &attention.value,
                            &normalized,
                            row_count,
                            self.config.hidden_size,
                            kv_width,
                            backend,
                        )?,
                        self.xs_matvec(
                            &attention.gate,
                            &normalized,
                            row_count,
                            self.config.hidden_size,
                            query_heads,
                            backend,
                        )?,
                    ],
                }
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
        let xs_qk = if self.flavor() == LagunaGgufFlavor::Xs21Q4KM {
            backend.laguna_xs_qk_rms_norm_rope_pair_device(
                &query,
                &key,
                &prepared.query_norm,
                &prepared.key_norm,
                self.config.rms_norm_eps as f32,
                position,
                rope,
            )?
        } else {
            None
        };
        let (query, key) = match xs_qk {
            Some(pair) => pair,
            None => required_pair(
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
            )?,
        };
        profile_prefill_boundary(
            backend,
            row_count,
            "laguna.prefill.attention_projection_and_rope",
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
        profile_prefill_boundary(backend, row_count, "laguna.prefill.attention_core")?;
        let post_attention = match self.flavor() {
            LagunaGgufFlavor::S21Q2Q3 => required(
                "Laguna S GGUF attention output and residual",
                backend.q8_0_matvec_add_device(
                    self.q8_bytes(&attention.output)?,
                    &context,
                    hidden,
                    row_count,
                    query_width,
                    self.config.hidden_size,
                )?,
            )?,
            LagunaGgufFlavor::Xs21Q4KM => self.xs_matvec_add(
                &attention.output,
                &context,
                hidden,
                row_count,
                query_width,
                self.config.hidden_size,
                backend,
            )?,
        };
        profile_prefill_boundary(backend, row_count, "laguna.prefill.attention_output")?;
        let mlp_input = required(
            "Laguna GGUF MLP RMSNorm",
            self.rms_norm_device(
                &post_attention,
                &prepared.post_attention_norm,
                row_count,
                self.config.rms_norm_eps as f32,
                backend,
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

    fn rms_norm_device<B: Backend>(
        &self,
        input: &DeviceValue,
        weight: &F32Tensor,
        row_count: usize,
        eps: f32,
        backend: &B,
    ) -> Result<Option<DeviceValue>> {
        if self.flavor() == LagunaGgufFlavor::Xs21Q4KM && row_count == 1 {
            if let Some(output) = backend.laguna_xs_rms_norm_device(input, weight, eps)? {
                return Ok(Some(output));
            }
        }
        backend.rms_norm_device(input, weight, eps)
    }

    fn forward_dense<B: Backend>(
        &self,
        dense: &LagunaGgufDense,
        input: &DeviceValue,
        residual: &DeviceValue,
        row_count: usize,
        backend: &B,
    ) -> Result<DeviceValue> {
        match self.flavor() {
            LagunaGgufFlavor::S21Q2Q3 => {
                let activated = required(
                    "Laguna S GGUF dense gate/up",
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
                    "Laguna S GGUF dense down and residual",
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
            LagunaGgufFlavor::Xs21Q4KM => {
                let activated = self.xs_gate_up_swiglu(
                    &dense.gate,
                    &dense.up,
                    input,
                    row_count,
                    self.config.hidden_size,
                    self.config.intermediate_size,
                    backend,
                )?;
                self.xs_matvec_add(
                    &dense.down,
                    &activated,
                    residual,
                    row_count,
                    self.config.intermediate_size,
                    self.config.hidden_size,
                    backend,
                )
            }
        }
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
        let xs_routing = if self.flavor() == LagunaGgufFlavor::Xs21Q4KM {
            backend.laguna_xs_router_topk_device(
                &logits,
                &prepared.correction_bias,
                self.config.num_experts_per_tok,
                self.config.norm_topk_prob,
                self.config.moe_routed_scaling_factor as f32,
            )?
        } else {
            None
        };
        let routing = match xs_routing {
            Some(routing) => routing,
            None => backend
                .moe_router_topk_resident_device(
                    &logits,
                    &prepared.correction_bias,
                    self.config.num_experts_per_tok,
                    self.config.norm_topk_prob,
                    self.config.moe_routed_scaling_factor as f32,
                )?
                .ok_or_else(|| Error::backend("Laguna GGUF routing requires native Metal"))?,
        };
        profile_prefill_boundary(backend, row_count, "laguna.prefill.router")?;
        let routed_gate = self.index.quantized_storage(&moe.routed_gate)?;
        let routed_up = self.index.quantized_storage(&moe.routed_up)?;
        let routed_down = self.index.quantized_storage(&moe.routed_down)?;
        match self.flavor() {
            LagunaGgufFlavor::S21Q2Q3 => {
                if routed_gate.block != routed_up.block || routed_gate.block != routed_down.block {
                    return Err(Error::weights(
                        "Laguna S GGUF routed gate/up/down quantization types disagree",
                    ));
                }
                let quant = match routed_gate.block {
                    GgufQuantBlockKind::Q2K => GgufExpertQuant::Q2K,
                    GgufQuantBlockKind::Q3K => GgufExpertQuant::Q3K,
                    other => {
                        return Err(Error::weights(format!(
                            "Laguna S routed experts must be Q2_K or Q3_K, got {other}"
                        )))
                    }
                };
                let routed = required(
                    "Laguna S GGUF routed experts",
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
                profile_prefill_boundary(backend, row_count, "laguna.prefill.routed_experts")?;
                let shared_activated = required(
                    "Laguna S GGUF shared expert gate/up",
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
                    "Laguna S GGUF shared expert down, routed combine, and residual",
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
            LagunaGgufFlavor::Xs21Q4KM => {
                if routed_gate.block != GgufQuantBlockKind::Q4K
                    || routed_up.block != GgufQuantBlockKind::Q4K
                {
                    return Err(Error::weights(format!(
                        "Laguna XS routed gate/up must be Q4_K, got {}/{}",
                        routed_gate.block, routed_up.block
                    )));
                }
                let down_quant = k_quant(routed_down.block)?;
                let routed = required(
                    "Laguna XS GGUF routed experts",
                    backend.laguna_xs_gguf_moe_device(
                        routed_gate.bytes,
                        routed_up.bytes,
                        routed_down.bytes,
                        down_quant,
                        &flat_input,
                        &routing,
                        self.config.hidden_size,
                        self.config.moe_intermediate_size,
                        self.config.hidden_size,
                    )?,
                )?;
                profile_prefill_boundary(backend, row_count, "laguna.prefill.routed_experts")?;
                let shared_activated = self.xs_gate_up_swiglu(
                    &moe.shared.gate,
                    &moe.shared.up,
                    &flat_input,
                    row_count,
                    self.config.hidden_size,
                    self.config.shared_expert_intermediate_size,
                    backend,
                )?;
                let flat_residual = residual.reshape(vec![row_count, self.config.hidden_size])?;
                self.xs_matvec_add2(
                    &moe.shared.down,
                    &shared_activated,
                    &routed,
                    &flat_residual,
                    row_count,
                    self.config.shared_expert_intermediate_size,
                    self.config.hidden_size,
                    backend,
                )
                .and_then(|output| output.reshape(residual.dims().to_vec()))
            }
        }
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

    fn k_bytes<'a>(&'a self, tensor: &'a GgufTensorInfo) -> Result<(GgufKQuant, &'a [u8])> {
        let storage = self.index.quantized_storage(tensor)?;
        Ok((k_quant(storage.block)?, storage.bytes))
    }

    fn xs_matvec<B: Backend>(
        &self,
        tensor: &GgufTensorInfo,
        input: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features: usize,
        backend: &B,
    ) -> Result<DeviceValue> {
        let (quant, weights) = self.k_bytes(tensor)?;
        required(
            &format!("Laguna XS {} projection", tensor.name),
            backend.gguf_k_matvec_device(
                quant,
                weights,
                input,
                row_count,
                in_features,
                out_features,
            )?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn xs_gate_up_swiglu<B: Backend>(
        &self,
        gate: &GgufTensorInfo,
        up: &GgufTensorInfo,
        input: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features: usize,
        backend: &B,
    ) -> Result<DeviceValue> {
        let (gate_quant, gate_weights) = self.k_bytes(gate)?;
        let (up_quant, up_weights) = self.k_bytes(up)?;
        if gate_quant != GgufKQuant::Q4K || up_quant != GgufKQuant::Q4K {
            return Err(Error::weights(format!(
                "Laguna XS gate/up must be Q4_K, got {gate_quant:?}/{up_quant:?}"
            )));
        }
        if let Some(output) = backend.laguna_xs_q4_gate_up_swiglu_device(
            gate_weights,
            up_weights,
            input,
            row_count,
            in_features,
            out_features,
        )? {
            return Ok(output);
        }
        let gate = self.xs_matvec(gate, input, row_count, in_features, out_features, backend)?;
        let up = self.xs_matvec(up, input, row_count, in_features, out_features, backend)?;
        required(
            "Laguna XS gate/up SwiGLU",
            backend.swiglu_device(&gate, &up)?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn xs_matvec_add<B: Backend>(
        &self,
        tensor: &GgufTensorInfo,
        input: &DeviceValue,
        residual: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features: usize,
        backend: &B,
    ) -> Result<DeviceValue> {
        let (quant, weights) = self.k_bytes(tensor)?;
        if let Some(output) = backend.laguna_xs_k_matvec_add_device(
            quant,
            weights,
            input,
            residual,
            row_count,
            in_features,
            out_features,
        )? {
            return Ok(output);
        }
        let projected =
            self.xs_matvec(tensor, input, row_count, in_features, out_features, backend)?;
        required(
            "Laguna XS projection residual",
            backend.add_device(&projected, residual)?,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn xs_matvec_add2<B: Backend>(
        &self,
        tensor: &GgufTensorInfo,
        input: &DeviceValue,
        residual_a: &DeviceValue,
        residual_b: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features: usize,
        backend: &B,
    ) -> Result<DeviceValue> {
        let (quant, weights) = self.k_bytes(tensor)?;
        if let Some(output) = backend.laguna_xs_k_matvec_add2_device(
            quant,
            weights,
            input,
            residual_a,
            residual_b,
            row_count,
            in_features,
            out_features,
        )? {
            return Ok(output);
        }
        let projected =
            self.xs_matvec(tensor, input, row_count, in_features, out_features, backend)?;
        let combined = required(
            "Laguna XS projection first residual",
            backend.add_device(&projected, residual_a)?,
        )?;
        required(
            "Laguna XS projection second residual",
            backend.add_device(&combined, residual_b)?,
        )
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

fn prepare_xs_dense_prefill_weights<B: Backend>(
    index: &LagunaGgufIndex,
    config: &LagunaConfig,
    backend: &B,
) -> Result<u64> {
    let mut prepared_bytes = 0_u64;
    for layer in &index.layers {
        let query_heads = config
            .query_heads(layer.layer_index)
            .ok_or_else(|| Error::config("Laguna XS layer has no query-head count"))?;
        let query_width = query_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| Error::model("Laguna XS query width overflow"))?;
        let kv_width = config.key_value_width();
        for (tensor, input_width, output_width) in [
            (&layer.attention.query, config.hidden_size, query_width),
            (&layer.attention.key, config.hidden_size, kv_width),
            (&layer.attention.value, config.hidden_size, kv_width),
            (&layer.attention.gate, config.hidden_size, query_heads),
            (&layer.attention.output, query_width, config.hidden_size),
        ] {
            prepared_bytes = prepared_bytes
                .checked_add(prepare_xs_dense_prefill_weight(
                    index,
                    tensor,
                    input_width,
                    output_width,
                    backend,
                )?)
                .ok_or_else(|| Error::backend("Laguna XS prepared weight byte count overflow"))?;
        }

        let shared_or_dense = match &layer.mlp {
            LagunaGgufMlp::Dense(dense) => (
                &dense.gate,
                &dense.up,
                &dense.down,
                config.intermediate_size,
            ),
            LagunaGgufMlp::Moe(moe) => (
                &moe.shared.gate,
                &moe.shared.up,
                &moe.shared.down,
                config.shared_expert_intermediate_size,
            ),
        };
        for tensor in [shared_or_dense.0, shared_or_dense.1] {
            prepared_bytes = prepared_bytes
                .checked_add(prepare_xs_dense_prefill_weight(
                    index,
                    tensor,
                    config.hidden_size,
                    shared_or_dense.3,
                    backend,
                )?)
                .ok_or_else(|| Error::backend("Laguna XS prepared weight byte count overflow"))?;
        }
        prepared_bytes = prepared_bytes
            .checked_add(prepare_xs_dense_prefill_weight(
                index,
                shared_or_dense.2,
                shared_or_dense.3,
                config.hidden_size,
                backend,
            )?)
            .ok_or_else(|| Error::backend("Laguna XS prepared weight byte count overflow"))?;
    }
    Ok(prepared_bytes)
}

fn prepare_xs_dense_prefill_weight<B: Backend>(
    index: &LagunaGgufIndex,
    tensor: &GgufTensorInfo,
    in_features: usize,
    out_features: usize,
    backend: &B,
) -> Result<u64> {
    let storage = index.quantized_storage(tensor)?;
    let quant = k_quant(storage.block)?;
    if !backend.prepare_laguna_xs_mps_prefill_weight(
        quant,
        storage.bytes,
        in_features,
        out_features,
    )? {
        return Err(Error::backend(
            "Laguna XS dense FP16 prefill preparation requires Metal Performance Shaders",
        ));
    }
    in_features
        .checked_mul(out_features)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| Error::backend("Laguna XS prepared weight size overflow"))
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
        let advice = if index.flavor() == LagunaGgufFlavor::S21Q2Q3
            && tensor.name.contains("_exps.weight")
        {
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

fn k_quant(block: GgufQuantBlockKind) -> Result<GgufKQuant> {
    match block {
        GgufQuantBlockKind::Q4K => Ok(GgufKQuant::Q4K),
        GgufQuantBlockKind::Q6K => Ok(GgufKQuant::Q6K),
        other => Err(Error::weights(format!(
            "Laguna XS projection must be Q4_K or Q6_K, got {other}"
        ))),
    }
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

fn row_is_prefill(hidden: &DeviceValue) -> Result<bool> {
    let row_count = hidden.element_count()? / hidden.dims().last().copied().unwrap_or(1);
    Ok(row_count > 1)
}

fn profile_prefill_boundary<B: Backend>(backend: &B, row_count: usize, label: &str) -> Result<()> {
    if row_count > 1
        && tracing::enabled!(
            target: "inferno::metal::profile",
            tracing::Level::TRACE
        )
    {
        backend.device_profile_boundary(label)?;
    }
    Ok(())
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
            (0..47)
                .step_by(super::LAYER_SUBMISSION_CHUNK)
                .collect::<Vec<_>>()
        );
    }
}
