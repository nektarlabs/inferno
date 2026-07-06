use backend::{Backend, BackendCapabilities};
use common::Tensor;
use common::{validate_exact_shape, Error, F32Tensor, PagedKvView, Result, Shape};
use config::Config;
use gguf::{GgmlType, GgufFile};

use crate::{
    EmbeddingLookupOutput, EmbeddingLookupReport, EmbeddingTable, GreedyReport, Index,
    IndexSummary, LayerKvCacheTensors, LayerStack, LayerStackForwardReport, LayerStackLoadReport,
    LogitsOutput, LogitsReport, OutputHead, OutputHeadLoadReport,
};

pub const DEFAULT_GGUF_OUTPUT_CHUNK_ROWS: usize = 16_384;

#[derive(Debug)]
pub struct Model<'a> {
    index: Index,
    embedding_table: EmbeddingTable<'a>,
    layer_stack: LayerStack<'a>,
    output_head: OutputHead<'a>,
    hidden_size: usize,
    max_context: usize,
    load_report: ModelLoadReport,
    /// Set after the batched device decode path declines once (a weight or
    /// backend without a device kernel), so later tokens skip straight to the
    /// eager path instead of re-encoding work that will be thrown away.
    device_decode_disabled: std::sync::atomic::AtomicBool,
}

#[derive(Debug)]
pub struct ModelHiddenOutput {
    pub hidden_states: Tensor,
    pub layer_kv_cache: Vec<LayerKvCacheTensors>,
    pub report: ModelHiddenReport,
}

#[derive(Debug)]
pub struct ModelLogitsOutput {
    pub hidden_states: Tensor,
    pub layer_kv_cache: Vec<LayerKvCacheTensors>,
    pub logits: Tensor,
    pub report: ModelLogitsReport,
}

#[derive(Debug)]
pub struct ModelGreedyOutput {
    pub hidden_states: Tensor,
    pub layer_kv_cache: Vec<LayerKvCacheTensors>,
    pub token_id: u32,
    pub token_score: f32,
    pub report: ModelGreedyReport,
}

#[derive(Debug)]
pub struct ModelTokenOutput {
    pub layer_kv_cache: Vec<LayerKvCacheTensors>,
    pub token_id: u32,
    pub token_score: f32,
}

#[derive(Debug)]
struct ModelHiddenTensors {
    hidden_states: Tensor,
    layer_kv_cache: Vec<LayerKvCacheTensors>,
}

#[derive(Debug)]
struct ModelHiddenF32Tensors {
    hidden_states: F32Tensor,
    layer_kv_cache: Vec<LayerKvCacheTensors>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelLoadReport {
    pub backend: BackendCapabilities,
    pub architecture: String,
    pub index: IndexSummary,
    pub token_embedding_tensor_name: String,
    pub token_embedding_tensor_type: GgmlType,
    pub token_embedding_shape: Shape,
    pub final_norm_tensor_name: String,
    pub layer_stack: LayerStackLoadReport,
    pub output_head: OutputHeadLoadReport,
    pub output_chunk_rows: usize,
    pub max_context: usize,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelHiddenReport {
    pub input_ids_shape: Shape,
    pub embedding: EmbeddingLookupReport,
    pub layer_stack: LayerStackForwardReport,
    pub output_hidden_states_shape: Shape,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelLogitsReport {
    pub hidden: ModelHiddenReport,
    pub selected_hidden_states_shape: Shape,
    pub logits: LogitsReport,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelGreedyReport {
    pub hidden: ModelHiddenReport,
    pub selected_hidden_states_shape: Shape,
    pub greedy: GreedyReport,
}

impl<'a> Model<'a> {
    pub fn open_from_gguf<B: Backend>(
        gguf: &'a GgufFile,
        config: &Config,
        backend: &B,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        let index = Index::from_gguf(gguf, config)?;
        Self::open_from_index(gguf, config, index, backend, output_chunk_rows)
    }

    pub fn open_from_index<B: Backend>(
        gguf: &'a GgufFile,
        config: &Config,
        index: Index,
        backend: &B,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        let embedding_table = EmbeddingTable::open(gguf, config, &index.root.token_embedding)?;
        let layer_stack = LayerStack::open(gguf, config, &index, backend, output_chunk_rows)?;
        let layer_stack_report = layer_stack.load_report().clone();
        let output_head = OutputHead::open(gguf, config, &index.root, backend, output_chunk_rows)?;
        let output_head_report = output_head.load_report().clone();

        let load_report = ModelLoadReport {
            backend: backend.capabilities(),
            architecture: index.architecture.clone(),
            index: index.summary.clone(),
            token_embedding_tensor_name: index.root.token_embedding.name.clone(),
            token_embedding_tensor_type: index.root.token_embedding.ty,
            token_embedding_shape: Shape::new(
                index
                    .root
                    .token_embedding
                    .dims
                    .iter()
                    .map(|dim| *dim as usize)
                    .collect::<Vec<_>>(),
            ),
            final_norm_tensor_name: index.root.final_norm.name.clone(),
            layer_stack: layer_stack_report,
            output_head: output_head_report,
            output_chunk_rows,
            max_context: config.max_context,
            limitations: vec![
                "direct GGUF path executes token embedding, GLM layer stack, final norm, and output head"
                    .to_string(),
                "paged KV ownership remains in the runtime above this model".to_string(),
                "output projection is chunked to avoid full vocabulary matrix dequantization"
                    .to_string(),
            ],
        };

        Ok(Self {
            index,
            embedding_table,
            layer_stack,
            output_head,
            hidden_size: config.hidden_size,
            max_context: config.max_context,
            load_report,
            device_decode_disabled: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Forces the per-op (eager) decode path by disabling the batched
    /// device-resident path. Escape hatch for debugging and for A/B
    /// comparison of the two decode routes.
    pub fn disable_device_decode(&self) {
        self.device_decode_disabled
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn load_report(&self) -> &ModelLoadReport {
        &self.load_report
    }

    pub fn index(&self) -> &Index {
        &self.index
    }

    pub fn max_context(&self) -> usize {
        self.max_context
    }

    pub fn embed_input_ids<B: Backend>(
        &self,
        input_ids: &[u32],
        backend: &B,
    ) -> Result<EmbeddingLookupOutput> {
        self.embedding_table.lookup_with_backend(input_ids, backend)
    }

    pub fn decode_logits_from_hidden<B: Backend>(
        &self,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<LogitsOutput> {
        self.output_head.decode_logits(hidden_states, backend)
    }

    pub fn prefill_last_logits<B: Backend>(
        &self,
        config: &Config,
        input_ids: &[u32],
        backend: &B,
    ) -> Result<ModelLogitsOutput> {
        let hidden = self.forward_hidden_states(config, input_ids, backend)?;
        self.last_token_logits(hidden, backend)
    }

    pub fn prefill_greedy<B: Backend>(
        &self,
        config: &Config,
        input_ids: &[u32],
        backend: &B,
    ) -> Result<ModelGreedyOutput> {
        let hidden = self.forward_hidden_states(config, input_ids, backend)?;
        self.last_token_greedy(hidden, backend)
    }

    pub fn prefill_next_token<B: Backend>(
        &self,
        config: &Config,
        input_ids: &[u32],
        backend: &B,
    ) -> Result<ModelTokenOutput> {
        if backend.capabilities().custom_kernels {
            let hidden = self.forward_hidden_f32_tensors(config, input_ids, backend)?;
            return self.last_token_only_f32(hidden, backend);
        }

        let hidden = self.forward_hidden_tensors(config, input_ids, backend)?;
        self.last_token_only(hidden, backend)
    }

    pub fn decode_step_with_past_kv_provider<B, F>(
        &self,
        config: &Config,
        decode_token_ids: &[u32],
        backend: &B,
        past_kv_for_layer: F,
    ) -> Result<ModelLogitsOutput>
    where
        B: Backend,
        F: FnMut(usize) -> Result<Option<(F32Tensor, F32Tensor)>>,
    {
        if decode_token_ids.is_empty() {
            return Err(Error::model(
                "GLM-5.2 GGUF decode step requires at least one token id",
            ));
        }
        let hidden = self.forward_hidden_states_with_past_kv_provider(
            config,
            decode_token_ids,
            backend,
            past_kv_for_layer,
        )?;
        self.last_token_logits(hidden, backend)
    }

    pub fn decode_step_greedy_with_past_kv_provider<B, F>(
        &self,
        config: &Config,
        decode_token_ids: &[u32],
        backend: &B,
        past_kv_for_layer: F,
    ) -> Result<ModelGreedyOutput>
    where
        B: Backend,
        F: FnMut(usize) -> Result<Option<(F32Tensor, F32Tensor)>>,
    {
        if decode_token_ids.is_empty() {
            return Err(Error::model(
                "GLM-5.2 GGUF greedy decode step requires at least one token id",
            ));
        }
        let hidden = self.forward_hidden_states_with_past_kv_provider(
            config,
            decode_token_ids,
            backend,
            past_kv_for_layer,
        )?;
        self.last_token_greedy(hidden, backend)
    }

    pub fn decode_next_token_with_past_kv_provider<B, F>(
        &self,
        config: &Config,
        decode_token_ids: &[u32],
        backend: &B,
        past_kv_for_layer: F,
    ) -> Result<ModelTokenOutput>
    where
        B: Backend,
        F: FnMut(usize) -> Result<Option<(F32Tensor, F32Tensor)>>,
    {
        if decode_token_ids.is_empty() {
            return Err(Error::model(
                "GLM-5.2 GGUF next-token decode step requires at least one token id",
            ));
        }
        if backend.capabilities().custom_kernels {
            let hidden = self.forward_hidden_f32_tensors_with_past_kv_provider(
                config,
                decode_token_ids,
                backend,
                past_kv_for_layer,
            )?;
            return self.last_token_only_f32(hidden, backend);
        }

        let hidden = self.forward_hidden_tensors_with_past_kv_provider(
            config,
            decode_token_ids,
            backend,
            past_kv_for_layer,
        )?;
        self.last_token_only(hidden, backend)
    }

    pub fn decode_next_token_with_paged_kv_provider<'kv, B, F>(
        &self,
        config: &Config,
        decode_token_ids: &[u32],
        backend: &B,
        mut past_kv_for_layer: F,
    ) -> Result<ModelTokenOutput>
    where
        B: Backend,
        F: FnMut(usize) -> Result<Option<PagedKvView<'kv>>>,
    {
        if decode_token_ids.is_empty() {
            return Err(Error::model(
                "GLM-5.2 GGUF native paged next-token decode requires at least one token id",
            ));
        }
        if !backend.capabilities().custom_kernels {
            return Err(Error::backend(
                "native paged decode requires the Metal Q2 backend",
            ));
        }

        // Prefer the batched device-resident path: one shared command buffer
        // per stretch of GPU work instead of one commit+wait per kernel.
        if decode_token_ids.len() == 1
            && backend.device_values_supported()
            && !self
                .device_decode_disabled
                .load(std::sync::atomic::Ordering::Relaxed)
        {
            match self.decode_next_token_paged_device(
                config,
                decode_token_ids,
                backend,
                &mut past_kv_for_layer,
            )? {
                Some(output) => return Ok(output),
                None => {
                    // Discard any partially encoded work, then permanently
                    // fall back — the unsupported component will not change
                    // between tokens.
                    backend.device_flush()?;
                    self.device_decode_disabled
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    tracing::warn!(
                        "device-batched decode path unavailable for this artifact; \
                         falling back to per-op decode"
                    );
                }
            }
        }

        let hidden = self.forward_hidden_f32_tensors_with_paged_kv_provider(
            config,
            decode_token_ids,
            backend,
            past_kv_for_layer,
        )?;
        self.last_token_only_f32(hidden, backend)
    }

    /// Batched device-resident single-token decode. Embeds on the host (a
    /// table lookup), then runs every layer and the output head with
    /// GPU-resident hidden states; the fused argmax at the end flushes the
    /// final batch and yields the token.
    fn decode_next_token_paged_device<'kv, B, F>(
        &self,
        config: &Config,
        decode_token_ids: &[u32],
        backend: &B,
        past_kv_for_layer: &mut F,
    ) -> Result<Option<ModelTokenOutput>>
    where
        B: Backend,
        F: FnMut(usize) -> Result<Option<PagedKvView<'kv>>>,
    {
        let embedding = self.embedding_table.lookup_f32(decode_token_ids)?;
        let stack = crate::try_device!(self.layer_stack.forward_decode_device(
            config,
            &embedding.hidden_states,
            backend,
            past_kv_for_layer,
        ));

        let token = match self
            .output_head
            .decode_token_device(&stack.hidden_states, backend)?
        {
            Some(token) => token,
            None => {
                // The head has no device path (e.g. a non-Q2_K output
                // projection): download the final hidden states and finish on
                // the eager head. The layer stack still ran fully batched.
                let hidden_states = backend.device_download_f32_tensor(&stack.hidden_states)?;
                let selected = require_native(
                    "select_last_token",
                    backend.select_last_token_f32_tensor(&hidden_states)?,
                )?;
                self.output_head.decode_token_f32(&selected, backend)?
            }
        };

        Ok(Some(ModelTokenOutput {
            layer_kv_cache: stack.layer_kv_cache,
            token_id: token.token_id,
            token_score: token.token_score,
        }))
    }

    fn forward_hidden_states<B: Backend>(
        &self,
        config: &Config,
        input_ids: &[u32],
        backend: &B,
    ) -> Result<ModelHiddenOutput> {
        self.forward_hidden_states_with_past_kv_provider(config, input_ids, backend, |_| Ok(None))
    }

    fn forward_hidden_states_with_past_kv_provider<B, F>(
        &self,
        config: &Config,
        input_ids: &[u32],
        backend: &B,
        past_kv_for_layer: F,
    ) -> Result<ModelHiddenOutput>
    where
        B: Backend,
        F: FnMut(usize) -> Result<Option<(F32Tensor, F32Tensor)>>,
    {
        let embedding = self
            .embedding_table
            .lookup_with_backend(input_ids, backend)?;
        let input_shape = embedding.report.input_ids_shape.clone();
        let stack_output = self.layer_stack.forward_with_past_kv_provider(
            config,
            &embedding.hidden_states,
            backend,
            past_kv_for_layer,
        )?;
        let layer_kv_cache = stack_output.layer_kv_cache;
        let hidden_states = stack_output.hidden_states;

        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF model hidden states must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        validate_exact_shape(
            "gguf_model_output_hidden_size",
            &[dims[2]],
            &[self.hidden_size],
        )?;

        Ok(ModelHiddenOutput {
            report: ModelHiddenReport {
                input_ids_shape: input_shape,
                embedding: embedding.report,
                layer_stack: stack_output.report,
                output_hidden_states_shape: Shape::new(hidden_states.dims().to_vec()),
            },
            hidden_states,
            layer_kv_cache,
        })
    }

    fn forward_hidden_tensors<B: Backend>(
        &self,
        config: &Config,
        input_ids: &[u32],
        backend: &B,
    ) -> Result<ModelHiddenTensors> {
        self.forward_hidden_tensors_with_past_kv_provider(config, input_ids, backend, |_| Ok(None))
    }

    fn forward_hidden_tensors_with_past_kv_provider<B, F>(
        &self,
        config: &Config,
        input_ids: &[u32],
        backend: &B,
        past_kv_for_layer: F,
    ) -> Result<ModelHiddenTensors>
    where
        B: Backend,
        F: FnMut(usize) -> Result<Option<(F32Tensor, F32Tensor)>>,
    {
        let embedding = self
            .embedding_table
            .lookup_with_backend(input_ids, backend)?;
        let stack_output = self.layer_stack.forward_tensors_with_past_kv_provider(
            config,
            &embedding.hidden_states,
            backend,
            past_kv_for_layer,
        )?;
        let hidden_states = stack_output.hidden_states;

        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF model hidden states must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        validate_exact_shape(
            "gguf_model_output_hidden_size",
            &[dims[2]],
            &[self.hidden_size],
        )?;

        Ok(ModelHiddenTensors {
            hidden_states,
            layer_kv_cache: stack_output.layer_kv_cache,
        })
    }

    fn forward_hidden_f32_tensors<B: Backend>(
        &self,
        config: &Config,
        input_ids: &[u32],
        backend: &B,
    ) -> Result<ModelHiddenF32Tensors> {
        self.forward_hidden_f32_tensors_with_past_kv_provider(config, input_ids, backend, |_| {
            Ok(None)
        })
    }

    fn forward_hidden_f32_tensors_with_past_kv_provider<B, F>(
        &self,
        config: &Config,
        input_ids: &[u32],
        backend: &B,
        past_kv_for_layer: F,
    ) -> Result<ModelHiddenF32Tensors>
    where
        B: Backend,
        F: FnMut(usize) -> Result<Option<(F32Tensor, F32Tensor)>>,
    {
        let embedding = self.embedding_table.lookup_f32(input_ids)?;
        let stack_output = self.layer_stack.forward_f32_input_with_past_kv_provider(
            config,
            &embedding.hidden_states,
            backend,
            past_kv_for_layer,
        )?;
        let hidden_states = stack_output.hidden_states;

        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF native model hidden states must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        validate_exact_shape(
            "gguf_native_model_output_hidden_size",
            &[dims[2]],
            &[self.hidden_size],
        )?;

        Ok(ModelHiddenF32Tensors {
            hidden_states,
            layer_kv_cache: stack_output.layer_kv_cache,
        })
    }

    fn forward_hidden_f32_tensors_with_paged_kv_provider<'kv, B, F>(
        &self,
        config: &Config,
        input_ids: &[u32],
        backend: &B,
        past_kv_for_layer: F,
    ) -> Result<ModelHiddenF32Tensors>
    where
        B: Backend,
        F: FnMut(usize) -> Result<Option<PagedKvView<'kv>>>,
    {
        let embedding = self.embedding_table.lookup_f32(input_ids)?;
        let stack_output = self.layer_stack.forward_f32_input_with_paged_kv_provider(
            config,
            &embedding.hidden_states,
            backend,
            past_kv_for_layer,
        )?;
        let hidden_states = stack_output.hidden_states;

        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF native paged model hidden states must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        validate_exact_shape(
            "gguf_native_paged_model_output_hidden_size",
            &[dims[2]],
            &[self.hidden_size],
        )?;

        Ok(ModelHiddenF32Tensors {
            hidden_states,
            layer_kv_cache: stack_output.layer_kv_cache,
        })
    }

    fn last_token_logits<B: Backend>(
        &self,
        hidden: ModelHiddenOutput,
        backend: &B,
    ) -> Result<ModelLogitsOutput> {
        let dims = hidden.hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF logits input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        if tokens == 0 {
            return Err(Error::model(
                "GLM-5.2 GGUF logits require at least one token",
            ));
        }
        validate_exact_shape(
            "gguf_model_logits_hidden_size",
            &[dims[2]],
            &[self.hidden_size],
        )?;

        let selected_hidden_states = backend.select_last_token(&hidden.hidden_states)?;
        validate_exact_shape(
            "gguf_model_selected_hidden_states",
            selected_hidden_states.dims(),
            &[batch, 1, self.hidden_size],
        )?;
        let logits = self
            .output_head
            .decode_logits(&selected_hidden_states, backend)?;

        Ok(ModelLogitsOutput {
            hidden_states: hidden.hidden_states,
            layer_kv_cache: hidden.layer_kv_cache,
            logits: logits.logits,
            report: ModelLogitsReport {
                hidden: hidden.report,
                selected_hidden_states_shape: Shape::new(selected_hidden_states.dims().to_vec()),
                logits: logits.report,
            },
        })
    }

    fn last_token_greedy<B: Backend>(
        &self,
        hidden: ModelHiddenOutput,
        backend: &B,
    ) -> Result<ModelGreedyOutput> {
        let dims = hidden.hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF greedy input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        if tokens == 0 {
            return Err(Error::model(
                "GLM-5.2 GGUF greedy decode requires at least one token",
            ));
        }
        validate_exact_shape(
            "gguf_model_greedy_hidden_size",
            &[dims[2]],
            &[self.hidden_size],
        )?;

        let selected_hidden_states = backend.select_last_token(&hidden.hidden_states)?;
        validate_exact_shape(
            "gguf_model_selected_hidden_states",
            selected_hidden_states.dims(),
            &[batch, 1, self.hidden_size],
        )?;
        let greedy = self
            .output_head
            .decode_greedy(&selected_hidden_states, backend)?;

        Ok(ModelGreedyOutput {
            hidden_states: hidden.hidden_states,
            layer_kv_cache: hidden.layer_kv_cache,
            token_id: greedy.token_id,
            token_score: greedy.token_score,
            report: ModelGreedyReport {
                hidden: hidden.report,
                selected_hidden_states_shape: Shape::new(selected_hidden_states.dims().to_vec()),
                greedy: greedy.report,
            },
        })
    }

    fn last_token_only<B: Backend>(
        &self,
        hidden: ModelHiddenTensors,
        backend: &B,
    ) -> Result<ModelTokenOutput> {
        if backend.capabilities().custom_kernels {
            return self.last_token_only_native(hidden, backend);
        }

        let dims = hidden.hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF next-token input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        if tokens == 0 {
            return Err(Error::model(
                "GLM-5.2 GGUF next-token decode requires at least one token",
            ));
        }
        validate_exact_shape(
            "gguf_model_next_token_hidden_size",
            &[dims[2]],
            &[self.hidden_size],
        )?;

        let selected_hidden_states = backend.select_last_token(&hidden.hidden_states)?;
        validate_exact_shape(
            "gguf_model_next_token_selected_hidden_states",
            selected_hidden_states.dims(),
            &[batch, 1, self.hidden_size],
        )?;
        let token = self
            .output_head
            .decode_token(&selected_hidden_states, backend)?;

        Ok(ModelTokenOutput {
            layer_kv_cache: hidden.layer_kv_cache,
            token_id: token.token_id,
            token_score: token.token_score,
        })
    }

    fn last_token_only_native<B: Backend>(
        &self,
        hidden: ModelHiddenTensors,
        backend: &B,
    ) -> Result<ModelTokenOutput> {
        let hidden_states = tensor_to_f32_tensor(&hidden.hidden_states)?;
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF native next-token input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        if tokens == 0 {
            return Err(Error::model(
                "GLM-5.2 GGUF native next-token decode requires at least one token",
            ));
        }
        validate_exact_shape(
            "gguf_model_native_next_token_hidden_size",
            &[dims[2]],
            &[self.hidden_size],
        )?;

        let selected_hidden_states = require_native(
            "select_last_token",
            backend.select_last_token_f32_tensor(&hidden_states)?,
        )?;
        validate_exact_shape(
            "gguf_model_native_next_token_selected_hidden_states",
            selected_hidden_states.dims(),
            &[batch, 1, self.hidden_size],
        )?;
        let token = self
            .output_head
            .decode_token_f32(&selected_hidden_states, backend)?;

        Ok(ModelTokenOutput {
            layer_kv_cache: hidden.layer_kv_cache,
            token_id: token.token_id,
            token_score: token.token_score,
        })
    }

    fn last_token_only_f32<B: Backend>(
        &self,
        hidden: ModelHiddenF32Tensors,
        backend: &B,
    ) -> Result<ModelTokenOutput> {
        let dims = hidden.hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF native next-token input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        if tokens == 0 {
            return Err(Error::model(
                "GLM-5.2 GGUF native next-token decode requires at least one token",
            ));
        }
        validate_exact_shape(
            "gguf_model_native_next_token_hidden_size",
            &[dims[2]],
            &[self.hidden_size],
        )?;

        let selected_hidden_states = require_native(
            "select_last_token",
            backend.select_last_token_f32_tensor(&hidden.hidden_states)?,
        )?;
        validate_exact_shape(
            "gguf_model_native_next_token_selected_hidden_states",
            selected_hidden_states.dims(),
            &[batch, 1, self.hidden_size],
        )?;
        let token = self
            .output_head
            .decode_token_f32(&selected_hidden_states, backend)?;

        Ok(ModelTokenOutput {
            layer_kv_cache: hidden.layer_kv_cache,
            token_id: token.token_id,
            token_score: token.token_score,
        })
    }
}

fn tensor_to_f32_tensor(tensor: &Tensor) -> Result<F32Tensor> {
    let dims = tensor.dims().to_vec();
    let values = tensor
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    F32Tensor::new(values, dims)
}

fn require_native<T>(operation: &str, value: Option<T>) -> Result<T> {
    value.ok_or_else(|| {
        Error::backend(format!(
            "native Metal {operation} is required for the GLM-5.2 Q2 model path"
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use backend::MetalBackend;
    use common::{Device, Tensor};
    use config::Config;
    use gguf::{
        GgmlType, GgufFile, GgufMetadataValueType, GGML_Q2_K_BLOCK_BYTES, GGML_Q8_0_BLOCK_BYTES,
        GGUF_MAGIC, GGUF_VERSION_V3,
    };

    use super::*;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn opens_q2_gguf_model_shell_and_runs_root_primitives_from_fixture() {
        let path = write_gguf_model_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config(256, 8);
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();

        let model = Model::open_from_gguf(&gguf, &config, &backend, 4).unwrap();
        let embedding = model.embed_input_ids(&[1, 2, 1], &backend).unwrap();
        let logits = model
            .decode_logits_from_hidden(
                &Tensor::from_vec(vec![1.0_f32; 256], (1, 1, 256), &Device::Cpu).unwrap(),
                &backend,
            )
            .unwrap();

        assert_eq!(model.load_report().architecture, "glm-dsa");
        assert_eq!(model.load_report().index.tensor_count, 3);
        assert_eq!(model.load_report().index.dense_layer_count, 0);
        assert_eq!(model.load_report().token_embedding_shape.dims(), &[256, 8]);
        assert_eq!(model.max_context(), 32);
        assert_eq!(model.index().root.output.name, "output.weight");
        assert_eq!(embedding.hidden_states.dims(), &[1, 3, 256]);
        assert_eq!(embedding.report.unique_token_count, 2);
        assert!(embedding.report.avoided_full_tensor_decode);
        assert_eq!(logits.logits.dims(), &[1, 8]);
        assert_eq!(logits.report.output_projection_chunk_count, 2);
        assert_eq!(
            model
                .load_report()
                .output_head
                .full_output_source_payload_bytes,
            GGML_Q2_K_BLOCK_BYTES * 8
        );
    }

    #[test]
    fn opens_q2_gguf_model_shell_and_runs_root_primitives() {
        let path = write_gguf_model_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config(256, 8);
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();

        let model = Model::open_from_gguf(&gguf, &config, &backend, 3).unwrap();
        let embedding = model.embed_input_ids(&[0, 7], &backend).unwrap();
        let logits = model
            .decode_logits_from_hidden(
                &Tensor::from_vec(vec![1.0_f32; 2 * 256], (2, 1, 256), &Device::Cpu).unwrap(),
                &backend,
            )
            .unwrap();

        assert_eq!(
            model.load_report().token_embedding_tensor_type,
            GgmlType::Q2K
        );
        assert_eq!(
            model
                .load_report()
                .output_head
                .full_output_source_payload_bytes,
            GGML_Q2_K_BLOCK_BYTES * 8
        );
        assert_eq!(embedding.hidden_states.dims(), &[1, 2, 256]);
        assert_eq!(logits.logits.dims(), &[2, 8]);
        assert_eq!(logits.report.output_projection_chunk_count, 3);
    }

    #[test]
    fn rejects_decode_logits_with_prefill_hidden_states() {
        let path = write_gguf_model_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config(256, 8);
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let model = Model::open_from_gguf(&gguf, &config, &backend, 4).unwrap();
        let hidden_states =
            Tensor::from_vec(vec![1.0_f32; 2 * 256], (1, 2, 256), &Device::Cpu).unwrap();

        let err = model
            .decode_logits_from_hidden(&hidden_states, &backend)
            .expect_err("decode logits should reject prefill shape");

        assert!(err
            .to_string()
            .contains("gguf_output_head_decode_hidden_states"));
    }

    #[test]
    fn q2_model_prefill_runs_embedding_stack_and_logits() {
        let path = write_full_gguf_model_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = full_config();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let model = Model::open_from_gguf(&gguf, &config, &backend, 4).unwrap();

        let output = model
            .prefill_last_logits(&config, &[1, 2], &backend)
            .unwrap();

        assert_eq!(model.load_report().layer_stack.total_layers, 2);
        assert_eq!(model.load_report().layer_stack.loaded_dense_layers, 1);
        assert_eq!(model.load_report().layer_stack.loaded_sparse_layers, 1);
        assert_eq!(output.hidden_states.dims(), &[1, 2, 256]);
        assert_eq!(output.layer_kv_cache.len(), 2);
        assert_eq!(output.layer_kv_cache[0].cache_k.dims(), &[1, 2, 2, 256]);
        assert_eq!(output.layer_kv_cache[1].cache_k.dims(), &[1, 2, 2, 256]);
        assert_eq!(output.logits.dims(), &[1, 8]);
        assert_eq!(
            output.report.hidden.embedding.input_ids_shape.dims(),
            &[1, 2]
        );
        assert_eq!(
            output
                .report
                .hidden
                .layer_stack
                .output_hidden_states_shape
                .dims(),
            &[1, 2, 256]
        );
        assert_eq!(output.report.hidden.layer_stack.dense_block_count, 1);
        assert_eq!(output.report.hidden.layer_stack.sparse_block_count, 1);
        assert_eq!(output.report.hidden.layer_stack.loaded_expert_requests, 2);
        assert_eq!(
            output.report.selected_hidden_states_shape.dims(),
            &[1, 1, 256]
        );
        assert_eq!(output.report.logits.logits_shape.dims(), &[1, 8]);
    }

    #[test]
    fn q2_model_prefill_greedy_matches_full_logits_without_materializing_logits() {
        let path = write_full_gguf_model_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = full_config();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let model = Model::open_from_gguf(&gguf, &config, &backend, 4).unwrap();
        let logits = model
            .prefill_last_logits(&config, &[1, 2], &backend)
            .unwrap();

        let greedy = model.prefill_greedy(&config, &[1, 2], &backend).unwrap();
        let token = model
            .prefill_next_token(&config, &[1, 2], &backend)
            .unwrap();

        let (expected_token_id, expected_score) =
            argmax_row(&logits.logits.to_vec2::<f32>().unwrap()[0]);
        assert_eq!(greedy.token_id, expected_token_id);
        assert_eq!(greedy.token_score, expected_score);
        assert_eq!(token.token_id, expected_token_id);
        assert_eq!(token.token_score, expected_score);
        assert_eq!(greedy.hidden_states.dims(), &[1, 2, 256]);
        assert_eq!(greedy.layer_kv_cache.len(), 2);
        assert_eq!(token.layer_kv_cache.len(), 2);
        assert_eq!(greedy.report.greedy.logits_shape.dims(), &[1, 8]);
        assert!(!greedy.report.greedy.materialized_full_logits);
    }

    #[test]
    fn q2_model_decode_uses_past_layer_kv() {
        let path = write_full_gguf_model_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = full_config();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let model = Model::open_from_gguf(&gguf, &config, &backend, 4).unwrap();
        let prefill = model
            .prefill_last_logits(&config, &[1, 2], &backend)
            .unwrap();
        let mut provider_calls = 0_usize;

        let decode = model
            .decode_step_with_past_kv_provider(&config, &[3], &backend, |layer_index| {
                provider_calls += 1;
                Ok(prefill
                    .layer_kv_cache
                    .iter()
                    .find(|entry| entry.layer_index == layer_index)
                    .map(|entry| (entry.cache_k.clone(), entry.cache_v.clone())))
            })
            .unwrap();
        let token = model
            .decode_next_token_with_past_kv_provider(&config, &[3], &backend, |layer_index| {
                Ok(prefill
                    .layer_kv_cache
                    .iter()
                    .find(|entry| entry.layer_index == layer_index)
                    .map(|entry| (entry.cache_k.clone(), entry.cache_v.clone())))
            })
            .unwrap();

        assert_eq!(provider_calls, 2);
        assert_eq!(
            model.load_report().token_embedding_tensor_type,
            GgmlType::Q2K
        );
        assert_eq!(decode.hidden_states.dims(), &[1, 1, 256]);
        assert_eq!(decode.layer_kv_cache.len(), 2);
        assert_eq!(decode.layer_kv_cache[0].cache_k.dims(), &[1, 2, 1, 256]);
        assert_eq!(decode.layer_kv_cache[1].cache_k.dims(), &[1, 2, 1, 256]);
        assert_eq!(token.layer_kv_cache.len(), 2);
        assert_eq!(token.layer_kv_cache[0].cache_k.dims(), &[1, 2, 1, 256]);
        assert_eq!(token.layer_kv_cache[1].cache_k.dims(), &[1, 2, 1, 256]);
        assert_eq!(decode.logits.dims(), &[1, 8]);
        assert_eq!(
            decode.report.hidden.layer_stack.max_attention_past_tokens,
            2
        );
    }

    fn argmax_row(row: &[f32]) -> (u32, f32) {
        let mut best_token_id = 0_u32;
        let mut best_score = f32::NEG_INFINITY;
        for (token_id, score) in row.iter().copied().enumerate() {
            if score > best_score {
                best_token_id = token_id as u32;
                best_score = score;
            }
        }
        (best_token_id, best_score)
    }

    fn tiny_config(hidden_size: usize, vocab_size: usize) -> Config {
        Config {
            model_type: "glm_moe_dsa".to_string(),
            hidden_size,
            num_layers: 0,
            dense_layers: 0,
            sparse_moe_layers: Some(0),
            vocab_size,
            attention_heads: 1,
            qk_head_dim: hidden_size,
            qk_no_rope_dim: hidden_size,
            qk_rope_dim: 0,
            v_head_dim: Some(hidden_size),
            num_routed_experts: 1,
            experts_per_token: 1,
            moe_intermediate_size: hidden_size,
            num_shared_experts: 1,
            moe_groups: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
            scoring_func: "softmax".to_string(),
            topk_method: "greedy".to_string(),
            max_context: 32,
            dsa_index_topk: 1,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000_000.0,
        }
    }

    fn full_config() -> Config {
        Config {
            model_type: "glm_moe_dsa".to_string(),
            hidden_size: 256,
            num_layers: 2,
            dense_layers: 1,
            sparse_moe_layers: Some(1),
            vocab_size: 8,
            attention_heads: 2,
            qk_head_dim: 256,
            qk_no_rope_dim: 128,
            qk_rope_dim: 128,
            v_head_dim: Some(256),
            num_routed_experts: 4,
            experts_per_token: 2,
            moe_intermediate_size: 256,
            num_shared_experts: 1,
            moe_groups: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 2.5,
            scoring_func: "sigmoid".to_string(),
            topk_method: "noaux_tc".to_string(),
            max_context: 32,
            dsa_index_topk: 1,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000_000.0,
        }
    }

    fn write_gguf_model_fixture(ty: GgmlType) -> PathBuf {
        let path = unique_temp_file("model");
        let mut writer = GgufWriter::new();
        writer.header(3, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);

        let embedding_offset = 0_u64;
        let embedding_bytes = quantized_payload_bytes(ty, 8);
        let norm_offset = align_u64(embedding_offset + embedding_bytes, 32);
        let norm_bytes = 256_u64 * 4;
        let output_offset = align_u64(norm_offset + norm_bytes, 32);
        let output_bytes = quantized_payload_bytes(ty, 8);

        writer.tensor_info("token_embd.weight", &[256, 8], ty, embedding_offset);
        writer.tensor_info("output_norm.weight", &[256], GgmlType::F32, norm_offset);
        writer.tensor_info("output.weight", &[256, 8], ty, output_offset);
        writer.pad_to(32);

        write_quantized_payload(&mut writer, ty, 8);
        writer.pad_to_absolute_data_offset(norm_offset);
        for _ in 0..256 {
            writer.bytes(&1.0_f32.to_le_bytes());
        }
        writer.pad_to_absolute_data_offset(output_offset);
        write_quantized_payload(&mut writer, ty, 8);
        writer.pad_to_absolute_data_offset(output_offset + output_bytes);
        writer.finish_to(path)
    }

    fn write_full_gguf_model_fixture(ty: GgmlType) -> PathBuf {
        let path = unique_temp_file("full-model");
        let mut specs = vec![
            TensorSpec::quant("token_embd.weight", vec![256, 8], ty, None),
            TensorSpec::f32("output_norm.weight", vec![256], None),
            TensorSpec::quant("output.weight", vec![256, 8], ty, None),
        ];
        insert_dense_layer(&mut specs, 0, ty);
        insert_sparse_layer(&mut specs, 1, ty);
        write_specs_gguf(path, &specs)
    }

    fn insert_dense_layer(specs: &mut Vec<TensorSpec>, layer: usize, ty: GgmlType) {
        insert_attention(specs, layer, ty);
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_gate.weight"),
            vec![256, 512],
            ty,
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_up.weight"),
            vec![256, 512],
            ty,
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_down.weight"),
            vec![512, 256],
            ty,
            None,
        ));
    }

    fn insert_sparse_layer(specs: &mut Vec<TensorSpec>, layer: usize, ty: GgmlType) {
        insert_attention(specs, layer, ty);
        specs.push(TensorSpec::f32(
            format!("blk.{layer}.exp_probs_b.bias"),
            vec![4],
            Some(vec![0.0, 0.4, 0.2, 0.8]),
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_gate_inp.weight"),
            vec![256, 4],
            ty,
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_gate_shexp.weight"),
            vec![256, 256],
            ty,
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_up_shexp.weight"),
            vec![256, 256],
            ty,
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_down_shexp.weight"),
            vec![256, 256],
            ty,
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_gate_exps.weight"),
            vec![256, 256, 4],
            ty,
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_up_exps.weight"),
            vec![256, 256, 4],
            ty,
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_down_exps.weight"),
            vec![256, 256, 4],
            ty,
            None,
        ));
    }

    fn insert_attention(specs: &mut Vec<TensorSpec>, layer: usize, ty: GgmlType) {
        specs.push(TensorSpec::f32(
            format!("blk.{layer}.attn_norm.weight"),
            vec![256],
            None,
        ));
        specs.push(TensorSpec::f32(
            format!("blk.{layer}.ffn_norm.weight"),
            vec![256],
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_q_a.weight"),
            vec![256, 256],
            ty,
            None,
        ));
        specs.push(TensorSpec::f32(
            format!("blk.{layer}.attn_q_a_norm.weight"),
            vec![256],
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_q_b.weight"),
            vec![256, 512],
            ty,
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_kv_a_mqa.weight"),
            vec![256, 384],
            ty,
            None,
        ));
        specs.push(TensorSpec::f32(
            format!("blk.{layer}.attn_kv_a_norm.weight"),
            vec![256],
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_k_b.weight"),
            vec![128, 256, 2],
            GgmlType::Q8_0,
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_v_b.weight"),
            vec![256, 256, 2],
            GgmlType::Q8_0,
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_output.weight"),
            vec![512, 256],
            ty,
            None,
        ));
    }

    #[derive(Debug)]
    struct TensorSpec {
        name: String,
        dims: Vec<u64>,
        ty: GgmlType,
        values: Option<Vec<f32>>,
    }

    impl TensorSpec {
        fn f32(name: impl Into<String>, dims: Vec<u64>, values: Option<Vec<f32>>) -> Self {
            Self {
                name: name.into(),
                dims,
                ty: GgmlType::F32,
                values,
            }
        }

        fn quant(
            name: impl Into<String>,
            dims: Vec<u64>,
            ty: GgmlType,
            values: Option<Vec<f32>>,
        ) -> Self {
            Self {
                name: name.into(),
                dims,
                ty,
                values,
            }
        }

        fn payload_len(&self) -> u64 {
            match self.ty {
                GgmlType::F32 => self.dims.iter().product::<u64>() * 4,
                GgmlType::Q2K => self.dims.iter().product::<u64>() / 256 * GGML_Q2_K_BLOCK_BYTES,
                GgmlType::Q8_0 => self.dims.iter().product::<u64>() / 32 * GGML_Q8_0_BLOCK_BYTES,
                other => panic!("unsupported fixture tensor type {other}"),
            }
        }
    }

    fn write_specs_gguf(path: PathBuf, specs: &[TensorSpec]) -> PathBuf {
        let mut writer = GgufWriter::new();
        writer.header(specs.len() as u64, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);

        let mut offset = 0_u64;
        let offsets = specs
            .iter()
            .map(|spec| {
                let current = offset;
                offset = align_u64(current + spec.payload_len(), 32);
                current
            })
            .collect::<Vec<_>>();
        for (spec, offset) in specs.iter().zip(offsets.iter().copied()) {
            writer.tensor_info(&spec.name, &spec.dims, spec.ty, offset);
        }
        writer.pad_to(32);

        for (spec, offset) in specs.iter().zip(offsets.iter().copied()) {
            writer.pad_to_absolute_data_offset(offset);
            match spec.ty {
                GgmlType::F32 => {
                    let element_count = spec.dims.iter().product::<u64>() as usize;
                    let values = spec
                        .values
                        .clone()
                        .unwrap_or_else(|| vec![1.0_f32; element_count]);
                    assert_eq!(values.len(), element_count);
                    for value in values {
                        writer.bytes(&value.to_le_bytes());
                    }
                }
                GgmlType::Q2K | GgmlType::Q8_0 => {
                    writer.bytes(&vec![0_u8; spec.payload_len() as usize]);
                }
                other => panic!("unsupported fixture tensor type {other}"),
            }
        }
        writer.pad_to_absolute_data_offset(offset);
        writer.finish_to(path)
    }

    fn write_quantized_payload(writer: &mut GgufWriter, ty: GgmlType, rows: usize) {
        for row in 0..rows {
            match ty {
                GgmlType::Q2K => writer.bytes(&q2_k_block(0xe4, row as u8 + 1)),
                other => panic!("unsupported fixture tensor type {other}"),
            }
        }
    }

    fn quantized_payload_bytes(ty: GgmlType, rows: u64) -> u64 {
        match ty {
            GgmlType::Q2K => GGML_Q2_K_BLOCK_BYTES * rows,
            other => panic!("unsupported fixture tensor type {other}"),
        }
    }

    fn q2_k_block(quant_byte: u8, min_shift: u8) -> Vec<u8> {
        let mut block = Vec::with_capacity(GGML_Q2_K_BLOCK_BYTES as usize);
        for scale in 1_u8..=16 {
            block.push((scale & 0x0f) | ((min_shift & 0x0f) << 4));
        }
        block.extend(std::iter::repeat_n(quant_byte, 64));
        block.extend_from_slice(&0x3c00_u16.to_le_bytes());
        block.extend_from_slice(&0x3800_u16.to_le_bytes());
        block
    }

    struct GgufWriter {
        bytes: Vec<u8>,
        data_start: Option<usize>,
    }

    impl GgufWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                data_start: None,
            }
        }

        fn header(&mut self, tensor_count: u64, metadata_kv_count: u64) {
            self.bytes.extend_from_slice(GGUF_MAGIC);
            self.u32(GGUF_VERSION_V3);
            self.u64(tensor_count);
            self.u64(metadata_kv_count);
        }

        fn metadata_key(&mut self, key: &str) {
            self.string(key);
        }

        fn tensor_info(&mut self, name: &str, dims: &[u64], ty: GgmlType, offset: u64) {
            self.string(name);
            self.u32(dims.len() as u32);
            for dim in dims {
                self.u64(*dim);
            }
            self.u32(ty.code());
            self.u64(offset);
        }

        fn string(&mut self, value: &str) {
            self.u64(value.len() as u64);
            self.bytes.extend_from_slice(value.as_bytes());
        }

        fn u32(&mut self, value: u32) {
            self.bytes.extend_from_slice(&value.to_le_bytes());
        }

        fn u64(&mut self, value: u64) {
            self.bytes.extend_from_slice(&value.to_le_bytes());
        }

        fn pad_to(&mut self, alignment: usize) {
            let remainder = self.bytes.len() % alignment;
            if remainder != 0 {
                self.bytes
                    .resize(self.bytes.len() + alignment - remainder, 0);
            }
            self.data_start = Some(self.bytes.len());
        }

        fn pad_to_absolute_data_offset(&mut self, offset: u64) {
            let target = self.data_start.unwrap() + offset as usize;
            if self.bytes.len() < target {
                self.bytes.resize(target, 0);
            }
        }

        fn bytes(&mut self, bytes: &[u8]) {
            self.bytes.extend_from_slice(bytes);
        }

        fn finish_to(self, path: PathBuf) -> PathBuf {
            fs::write(&path, self.bytes).unwrap();
            path
        }
    }

    fn align_u64(value: u64, alignment: u64) -> u64 {
        let remainder = value % alignment;
        if remainder == 0 {
            value
        } else {
            value + alignment - remainder
        }
    }

    fn unique_temp_file(label: &str) -> PathBuf {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("model-{label}-{}-{id}", std::process::id()))
    }
}
