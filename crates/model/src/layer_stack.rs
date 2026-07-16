use backend::{Backend, BackendCapabilities};
use common::Tensor;
use common::{validate_exact_shape, Error, F32Tensor, PagedKvView, Result, Shape};
use config::Config;
use gguf::GgufFile;
use tracing::{debug, warn};

use crate::{
    kv_types::{cache_tensor_from_host, cache_tensor_to_host},
    profile, DenseBlock, DenseBlockLoadReport, Index, LayerDeviceKvCacheTensors, LayerKind,
    LayerKvCacheTensors, SparseBlock, SparseBlockLoadReport,
};

#[derive(Debug)]
pub struct LayerStack<'a> {
    hidden_size: usize,
    layers: Vec<RuntimeLayer<'a>>,
    report: LayerStackLoadReport,
}

#[derive(Debug)]
enum RuntimeLayer<'a> {
    Dense(DenseBlock<'a>),
    Sparse(SparseBlock<'a>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerRuntimeKind {
    DenseBlock,
    SparseBlock,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerRuntimeReport {
    pub layer_index: usize,
    pub layer_kind: LayerKind,
    pub runtime_kind: LayerRuntimeKind,
    pub dense_block: Option<DenseBlockLoadReport>,
    pub sparse_block: Option<SparseBlockLoadReport>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerStackLoadReport {
    pub backend: BackendCapabilities,
    pub total_layers: usize,
    pub loaded_dense_layers: usize,
    pub loaded_sparse_layers: usize,
    pub output_chunk_rows: usize,
    pub layers: Vec<LayerRuntimeReport>,
    pub limitations: Vec<String>,
}

#[derive(Debug)]
pub struct LayerStackForwardOutput {
    pub hidden_states: Tensor,
    pub layer_kv_cache: Vec<LayerKvCacheTensors>,
    pub report: LayerStackForwardReport,
}

#[derive(Debug)]
pub struct LayerStackForwardTensors {
    pub hidden_states: Tensor,
    pub layer_kv_cache: Vec<LayerKvCacheTensors>,
}

#[derive(Debug)]
pub struct LayerStackForwardF32Tensors {
    pub hidden_states: F32Tensor,
    pub layer_kv_cache: Vec<LayerKvCacheTensors>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerStackForwardReport {
    pub input_hidden_states_shape: Shape,
    pub output_hidden_states_shape: Shape,
    pub executed_layer_count: usize,
    pub dense_block_count: usize,
    pub sparse_block_count: usize,
    pub kv_cache_layer_count: usize,
    pub layer_k_cache_shape: Option<Shape>,
    pub layer_v_cache_shape: Option<Shape>,
    pub max_attention_past_tokens: usize,
    pub loaded_expert_requests: usize,
    pub routed_source_payload_bytes_read: u64,
    pub routed_peak_decoded_f32_bytes: u64,
}

impl<'a> LayerStack<'a> {
    pub fn open<B: Backend>(
        gguf: &'a GgufFile,
        config: &Config,
        index: &Index,
        backend: &B,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        validate_exact_shape(
            "gguf_layer_stack_index_layer_count",
            &[index.layers.len()],
            &[config.num_layers],
        )?;

        let mut layers = Vec::with_capacity(index.layers.len());
        let mut layer_reports = Vec::with_capacity(index.layers.len());
        let mut loaded_dense_layers = 0_usize;
        let mut loaded_sparse_layers = 0_usize;

        for layer in &index.layers {
            match layer.kind {
                LayerKind::Dense => {
                    let block = DenseBlock::open(gguf, config, layer, backend, output_chunk_rows)?;
                    let report = block.load_report().clone();
                    loaded_dense_layers += 1;
                    layer_reports.push(LayerRuntimeReport {
                        layer_index: layer.layer_index,
                        layer_kind: LayerKind::Dense,
                        runtime_kind: LayerRuntimeKind::DenseBlock,
                        dense_block: Some(report),
                        sparse_block: None,
                    });
                    layers.push(RuntimeLayer::Dense(block));
                }
                LayerKind::SparseMoe => {
                    let block = SparseBlock::open(gguf, config, layer, backend, output_chunk_rows)?;
                    let report = block.load_report().clone();
                    loaded_sparse_layers += 1;
                    layer_reports.push(LayerRuntimeReport {
                        layer_index: layer.layer_index,
                        layer_kind: LayerKind::SparseMoe,
                        runtime_kind: LayerRuntimeKind::SparseBlock,
                        dense_block: None,
                        sparse_block: Some(report),
                    });
                    layers.push(RuntimeLayer::Sparse(block));
                }
            }
        }

        Ok(Self {
            hidden_size: config.hidden_size,
            layers,
            report: LayerStackLoadReport {
                backend: backend.capabilities(),
                total_layers: layer_reports.len(),
                loaded_dense_layers,
                loaded_sparse_layers,
                output_chunk_rows,
                layers: layer_reports,
                limitations: vec![
                    "GGUF layer stack executes dense prefix layers and sparse MoE layers in GLM order"
                        .to_string(),
                    "paged KV ownership remains in the runtime; this stack returns per-layer K/V tensors"
                        .to_string(),
                    "native Metal kernels are required for the GLM-5.2 Q2 generation path"
                        .to_string(),
                    "DSA sparse decode is active for native sparse layers; absorbed MLA Metal attention is still pending".to_string(),
                ],
            },
        })
    }

    pub fn load_report(&self) -> &LayerStackLoadReport {
        &self.report
    }

    #[cfg(test)]
    pub fn forward<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<LayerStackForwardOutput> {
        self.forward_with_past_kv_provider(config, hidden_states, backend, |_| Ok(None))
    }

    pub fn forward_with_past_kv_provider<B, F>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
        mut past_kv_for_layer: F,
    ) -> Result<LayerStackForwardOutput>
    where
        B: Backend,
        F: FnMut(usize) -> Result<Option<(F32Tensor, F32Tensor)>>,
    {
        validate_hidden_states(self.hidden_size, hidden_states)?;

        let input_shape = Shape::new(hidden_states.dims().to_vec());
        let mut current = hidden_states.clone();
        let mut layer_kv_cache = Vec::with_capacity(self.layers.len());
        let mut dense_block_count = 0_usize;
        let mut sparse_block_count = 0_usize;
        let mut layer_k_cache_shape = None;
        let mut layer_v_cache_shape = None;
        let mut max_attention_past_tokens = 0_usize;
        let mut loaded_expert_requests = 0_usize;
        let mut routed_source_payload_bytes_read = 0_u64;
        let mut routed_peak_decoded_f32_bytes = 0_u64;

        for layer in &self.layers {
            match layer {
                RuntimeLayer::Dense(block) => {
                    let layer_index = block.load_report().layer_index;
                    let past_kv = past_kv_to_host(past_kv_for_layer(layer_index)?, backend)?;
                    let past_kv = past_kv
                        .as_ref()
                        .map(|(cache_k, cache_v)| (cache_k, cache_v));
                    let output = block.forward_with_past_kv(config, &current, backend, past_kv)?;
                    dense_block_count += 1;
                    max_attention_past_tokens =
                        max_attention_past_tokens.max(output.report.attention_past_tokens);
                    capture_first_kv_shapes(
                        &mut layer_k_cache_shape,
                        &mut layer_v_cache_shape,
                        &output.cache_k,
                        &output.cache_v,
                    );
                    layer_kv_cache.push(LayerKvCacheTensors {
                        layer_index: output.report.layer_index,
                        layer_kind: LayerKind::Dense,
                        cache_k: cache_tensor_from_host(&output.cache_k)?,
                        cache_v: cache_tensor_from_host(&output.cache_v)?,
                        index_key: None,
                    });
                    current = output.hidden_states;
                }
                RuntimeLayer::Sparse(block) => {
                    let layer_index = block.load_report().layer_index;
                    let past_kv = past_kv_to_host(past_kv_for_layer(layer_index)?, backend)?;
                    let past_kv = past_kv
                        .as_ref()
                        .map(|(cache_k, cache_v)| (cache_k, cache_v));
                    let output = block.forward_with_past_kv(config, &current, backend, past_kv)?;
                    sparse_block_count += 1;
                    max_attention_past_tokens =
                        max_attention_past_tokens.max(output.report.attention_past_tokens);
                    loaded_expert_requests =
                        loaded_expert_requests.saturating_add(output.report.loaded_expert_count);
                    routed_source_payload_bytes_read = routed_source_payload_bytes_read
                        .saturating_add(output.report.routed_source_payload_bytes_read);
                    routed_peak_decoded_f32_bytes = routed_peak_decoded_f32_bytes
                        .saturating_add(output.report.routed_peak_decoded_f32_bytes);
                    capture_first_kv_shapes(
                        &mut layer_k_cache_shape,
                        &mut layer_v_cache_shape,
                        &output.cache_k,
                        &output.cache_v,
                    );
                    layer_kv_cache.push(LayerKvCacheTensors {
                        layer_index: output.report.layer_index,
                        layer_kind: LayerKind::SparseMoe,
                        cache_k: cache_tensor_from_host(&output.cache_k)?,
                        cache_v: cache_tensor_from_host(&output.cache_v)?,
                        index_key: None,
                    });
                    current = output.hidden_states;
                }
            }
        }

        Ok(LayerStackForwardOutput {
            report: LayerStackForwardReport {
                input_hidden_states_shape: input_shape,
                output_hidden_states_shape: Shape::new(current.dims().to_vec()),
                executed_layer_count: layer_kv_cache.len(),
                dense_block_count,
                sparse_block_count,
                kv_cache_layer_count: layer_kv_cache.len(),
                layer_k_cache_shape,
                layer_v_cache_shape,
                max_attention_past_tokens,
                loaded_expert_requests,
                routed_source_payload_bytes_read,
                routed_peak_decoded_f32_bytes,
            },
            hidden_states: current,
            layer_kv_cache,
        })
    }

    #[cfg(test)]
    pub fn forward_tensors<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<LayerStackForwardTensors> {
        self.forward_tensors_with_past_kv_provider(config, hidden_states, backend, |_| Ok(None))
    }

    pub fn forward_tensors_with_past_kv_provider<B, F>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
        mut past_kv_for_layer: F,
    ) -> Result<LayerStackForwardTensors>
    where
        B: Backend,
        F: FnMut(usize) -> Result<Option<(F32Tensor, F32Tensor)>>,
    {
        if backend.capabilities().custom_kernels {
            return self.forward_f32_tensors_with_past_kv_provider(
                config,
                hidden_states,
                backend,
                past_kv_for_layer,
            );
        }

        validate_hidden_states(self.hidden_size, hidden_states)?;

        let mut current = hidden_states.clone();
        let mut layer_kv_cache = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            match layer {
                RuntimeLayer::Dense(block) => {
                    let layer_index = block.load_report().layer_index;
                    let past_kv = past_kv_for_layer(layer_index)?;
                    let past_kv = past_kv
                        .as_ref()
                        .map(|(cache_k, cache_v)| (cache_k, cache_v));
                    let output = profile::run_layer_stage(layer_index, "dense", || {
                        block.forward_tensors_with_past_kv(config, &current, backend, past_kv)
                    })?;
                    debug!(
                        layer_index,
                        layer_kind = "dense",
                        "GLM-5.2 Q2 layer completed"
                    );
                    layer_kv_cache.push(LayerKvCacheTensors {
                        layer_index,
                        layer_kind: LayerKind::Dense,
                        cache_k: output.cache_k,
                        cache_v: output.cache_v,
                        index_key: output.index_key,
                    });
                    current = output.hidden_states;
                }
                RuntimeLayer::Sparse(block) => {
                    let layer_index = block.load_report().layer_index;
                    let past_kv = past_kv_for_layer(layer_index)?;
                    let past_kv = past_kv
                        .as_ref()
                        .map(|(cache_k, cache_v)| (cache_k, cache_v));
                    let output = profile::run_layer_stage(layer_index, "sparse_moe", || {
                        block.forward_tensors_with_past_kv(config, &current, backend, past_kv)
                    })?;
                    debug!(
                        layer_index,
                        layer_kind = "sparse_moe",
                        "GLM-5.2 Q2 layer completed"
                    );
                    layer_kv_cache.push(LayerKvCacheTensors {
                        layer_index,
                        layer_kind: LayerKind::SparseMoe,
                        cache_k: output.cache_k,
                        cache_v: output.cache_v,
                        index_key: output.index_key,
                    });
                    current = output.hidden_states;
                }
            }
        }

        Ok(LayerStackForwardTensors {
            hidden_states: current,
            layer_kv_cache,
        })
    }

    fn forward_f32_tensors_with_past_kv_provider<B, F>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
        past_kv_for_layer: F,
    ) -> Result<LayerStackForwardTensors>
    where
        B: Backend,
        F: FnMut(usize) -> Result<Option<(F32Tensor, F32Tensor)>>,
    {
        validate_hidden_states(self.hidden_size, hidden_states)?;

        let hidden_states = tensor_to_f32_tensor(hidden_states)?;
        let output = self.forward_f32_input_with_past_kv_provider(
            config,
            &hidden_states,
            backend,
            past_kv_for_layer,
        )?;

        Ok(LayerStackForwardTensors {
            hidden_states: tensor_from_f32_tensor(output.hidden_states, backend.device())?,
            layer_kv_cache: output.layer_kv_cache,
        })
    }

    pub fn forward_f32_input_with_past_kv_provider<B, F>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        backend: &B,
        mut past_kv_for_layer: F,
    ) -> Result<LayerStackForwardF32Tensors>
    where
        B: Backend,
        F: FnMut(usize) -> Result<Option<(F32Tensor, F32Tensor)>>,
    {
        validate_hidden_states_f32(self.hidden_size, hidden_states)?;

        let mut current = hidden_states.clone();
        let mut layer_kv_cache = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            match layer {
                RuntimeLayer::Dense(block) => {
                    let layer_index = block.load_report().layer_index;
                    let past_kv = past_kv_for_layer(layer_index)?;
                    let past_kv = past_kv
                        .as_ref()
                        .map(|(cache_k, cache_v)| (cache_k, cache_v));
                    let output = profile::run_layer_stage(layer_index, "dense", || {
                        block.forward_f32_tensors_with_past_kv(config, &current, backend, past_kv)
                    })?;
                    debug!(
                        layer_index,
                        layer_kind = "dense",
                        "GLM-5.2 Q2 native layer completed"
                    );
                    layer_kv_cache.push(LayerKvCacheTensors {
                        layer_index,
                        layer_kind: LayerKind::Dense,
                        cache_k: output.cache_k,
                        cache_v: output.cache_v,
                        index_key: output.index_key,
                    });
                    current = output.hidden_states;
                }
                RuntimeLayer::Sparse(block) => {
                    let layer_index = block.load_report().layer_index;
                    let past_kv = past_kv_for_layer(layer_index)?;
                    let past_kv = past_kv
                        .as_ref()
                        .map(|(cache_k, cache_v)| (cache_k, cache_v));
                    let output = profile::run_layer_stage(layer_index, "sparse_moe", || {
                        block.forward_f32_tensors_with_past_kv(config, &current, backend, past_kv)
                    })?;
                    debug!(
                        layer_index,
                        layer_kind = "sparse_moe",
                        "GLM-5.2 Q2 native layer completed"
                    );
                    layer_kv_cache.push(LayerKvCacheTensors {
                        layer_index,
                        layer_kind: LayerKind::SparseMoe,
                        cache_k: output.cache_k,
                        cache_v: output.cache_v,
                        index_key: output.index_key,
                    });
                    current = output.hidden_states;
                }
            }
        }

        Ok(LayerStackForwardF32Tensors {
            hidden_states: current,
            layer_kv_cache,
        })
    }

    pub fn forward_sparse_f32_input_with_past_kv_provider<B, F, I>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        backend: &B,
        mut past_kv_for_layer: F,
        mut index_keys_for_layer: I,
    ) -> Result<LayerStackForwardF32Tensors>
    where
        B: Backend,
        F: FnMut(usize) -> Result<Option<(F32Tensor, F32Tensor)>>,
        I: FnMut(usize) -> Result<Option<F32Tensor>>,
    {
        validate_hidden_states_f32(self.hidden_size, hidden_states)?;

        let mut current = hidden_states.clone();
        let mut layer_kv_cache = Vec::with_capacity(self.layers.len());
        let mut last_dsa_selection: Option<Vec<u32>> = None;

        for layer in &self.layers {
            match layer {
                RuntimeLayer::Dense(block) => {
                    let layer_index = block.load_report().layer_index;
                    let past_kv = past_kv_for_layer(layer_index)?;
                    let past_kv = past_kv
                        .as_ref()
                        .map(|(cache_k, cache_v)| (cache_k, cache_v));
                    let output = profile::run_layer_stage(layer_index, "dense", || {
                        block.forward_f32_tensors_with_past_kv(config, &current, backend, past_kv)
                    })?;
                    debug!(
                        layer_index,
                        layer_kind = "dense",
                        "GLM-5.2 Q2 native layer completed"
                    );
                    layer_kv_cache.push(LayerKvCacheTensors {
                        layer_index,
                        layer_kind: LayerKind::Dense,
                        cache_k: output.cache_k,
                        cache_v: output.cache_v,
                        index_key: output.index_key,
                    });
                    current = output.hidden_states;
                }
                RuntimeLayer::Sparse(block) => {
                    let layer_index = block.load_report().layer_index;
                    let past_kv = past_kv_for_layer(layer_index)?;
                    let past_kv = past_kv
                        .as_ref()
                        .map(|(cache_k, cache_v)| (cache_k, cache_v));
                    let cached_index_keys = if block.has_dsa_indexer() {
                        index_keys_for_layer(layer_index)?
                    } else {
                        None
                    };
                    let (output, next_shared_selection) =
                        profile::run_layer_stage(layer_index, "sparse_moe", || {
                            block.forward_sparse_f32_tensors_with_past_kv(
                                config,
                                &current,
                                backend,
                                past_kv,
                                cached_index_keys.as_ref(),
                                last_dsa_selection.as_deref(),
                            )
                        })?;
                    if let Some(selection) = next_shared_selection {
                        last_dsa_selection = Some(selection);
                    }
                    debug!(
                        layer_index,
                        layer_kind = "sparse_moe",
                        "GLM-5.2 Q2 native sparse layer completed"
                    );
                    layer_kv_cache.push(LayerKvCacheTensors {
                        layer_index,
                        layer_kind: LayerKind::SparseMoe,
                        cache_k: output.cache_k,
                        cache_v: output.cache_v,
                        index_key: output.index_key,
                    });
                    current = output.hidden_states;
                }
            }
        }

        Ok(LayerStackForwardF32Tensors {
            hidden_states: current,
            layer_kv_cache,
        })
    }

    pub fn forward_f32_input_with_paged_kv_provider<'kv, B, F>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        backend: &B,
        mut past_kv_for_layer: F,
    ) -> Result<LayerStackForwardF32Tensors>
    where
        B: Backend,
        F: FnMut(usize) -> Result<Option<PagedKvView<'kv>>>,
    {
        validate_hidden_states_f32(self.hidden_size, hidden_states)?;

        let mut current = hidden_states.clone();
        let mut layer_kv_cache = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            match layer {
                RuntimeLayer::Dense(block) => {
                    let layer_index = block.load_report().layer_index;
                    let past_kv = past_kv_for_layer(layer_index)?;
                    let output = profile::run_layer_stage(layer_index, "dense", || {
                        block.forward_f32_tensors_with_paged_past_kv(
                            config, &current, backend, past_kv,
                        )
                    })?;
                    debug!(
                        layer_index,
                        layer_kind = "dense",
                        "GLM-5.2 Q2 native paged layer completed"
                    );
                    layer_kv_cache.push(LayerKvCacheTensors {
                        layer_index,
                        layer_kind: LayerKind::Dense,
                        cache_k: output.cache_k,
                        cache_v: output.cache_v,
                        index_key: output.index_key,
                    });
                    current = output.hidden_states;
                }
                RuntimeLayer::Sparse(block) => {
                    let layer_index = block.load_report().layer_index;
                    let past_kv = past_kv_for_layer(layer_index)?;
                    let output = profile::run_layer_stage(layer_index, "sparse_moe", || {
                        block.forward_f32_tensors_with_paged_past_kv(
                            config, &current, backend, past_kv,
                        )
                    })?;
                    debug!(
                        layer_index,
                        layer_kind = "sparse_moe",
                        "GLM-5.2 Q2 native paged layer completed"
                    );
                    layer_kv_cache.push(LayerKvCacheTensors {
                        layer_index,
                        layer_kind: LayerKind::SparseMoe,
                        cache_k: output.cache_k,
                        cache_v: output.cache_v,
                        index_key: output.index_key,
                    });
                    current = output.hidden_states;
                }
            }
        }

        Ok(LayerStackForwardF32Tensors {
            hidden_states: current,
            layer_kv_cache,
        })
    }

    /// Batched device-resident decode driver: uploads the embedding once,
    /// threads GPU-resident hidden states through every layer, and returns
    /// them still on the device so the output head can consume them without a
    /// host round-trip. MoE router IDs and weights remain device-resident; the
    /// output-head argmax is the normal end-of-token synchronization point.
    ///
    /// Returns `Ok(None)` when any layer lacks a device path or a layer has no
    /// resident paged past KV; the caller falls back to the eager route.
    pub(crate) fn forward_decode_device<B, F, S, I>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        backend: &B,
        mut past_kv_for_layer: F,
        mut selected_kv_for_tokens: S,
        mut index_keys_for_layer: I,
    ) -> Result<Option<LayerStackDecodeDeviceTensors>>
    where
        B: Backend,
        F: FnMut(usize) -> Result<Option<backend::DevicePagedKvView>>,
        S: FnMut(usize, &[u32]) -> Result<Option<backend::DeviceSelectedKvView>>,
        I: FnMut(usize) -> Result<Option<backend::DeviceValue>>,
    {
        validate_hidden_states_f32(self.hidden_size, hidden_states)?;

        let mut current = crate::try_device!(backend.device_upload_f32_tensor(hidden_states));
        let mut layer_kv_cache = Vec::with_capacity(self.layers.len());
        let mut last_dsa_selection: Option<Vec<u32>> = None;

        for layer in &self.layers {
            match layer {
                RuntimeLayer::Dense(block) => {
                    let layer_index = block.load_report().layer_index;
                    let Some(past_kv) = past_kv_for_layer(layer_index)? else {
                        return Ok(None);
                    };
                    let output = match profile::run_layer_stage(layer_index, "dense", || {
                        block.forward_decode_device(config, &current, backend, &past_kv)
                    })? {
                        Some(output) => output,
                        None => {
                            warn!(
                                layer_index,
                                layer_kind = "dense",
                                "device-batched layer path unavailable"
                            );
                            return Err(Error::backend(format!(
                                "device-batched dense layer {layer_index} has no complete native path"
                            )));
                        }
                    };
                    debug!(
                        layer_index,
                        layer_kind = "dense",
                        "GLM-5.2 Q2 device-batched layer completed"
                    );
                    layer_kv_cache.push(LayerDeviceKvCacheTensors {
                        layer_index,
                        layer_kind: LayerKind::Dense,
                        cache_k: output.cache_k,
                        cache_v: output.cache_v,
                        index_key: output.index_key,
                    });
                    current = output.hidden_states;
                }
                RuntimeLayer::Sparse(block) => {
                    let layer_index = block.load_report().layer_index;
                    let Some(past_kv) = past_kv_for_layer(layer_index)? else {
                        return Ok(None);
                    };
                    let (output, next_shared_selection) = match profile::run_layer_stage(
                        layer_index,
                        "sparse_moe",
                        || {
                            block.forward_sparse_decode_device(
                                config,
                                &current,
                                backend,
                                &past_kv,
                                &mut selected_kv_for_tokens,
                                &mut index_keys_for_layer,
                                last_dsa_selection.as_deref(),
                                None,
                                true,
                            )
                        },
                    )? {
                        Some(output) => output,
                        None => {
                            warn!(
                                layer_index,
                                layer_kind = "sparse_moe",
                                "device-batched layer path unavailable"
                            );
                            return Err(Error::backend(format!(
                                    "device-batched sparse layer {layer_index} has no complete native path"
                                )));
                        }
                    };
                    if let Some(selection) = next_shared_selection {
                        last_dsa_selection = Some(selection);
                    }
                    debug!(
                        layer_index,
                        layer_kind = "sparse_moe",
                        "GLM-5.2 Q2 device-batched layer completed"
                    );
                    layer_kv_cache.push(LayerDeviceKvCacheTensors {
                        layer_index,
                        layer_kind: LayerKind::SparseMoe,
                        cache_k: output.cache_k,
                        cache_v: output.cache_v,
                        index_key: output.index_key,
                    });
                    current = output.hidden_states;
                }
            }
        }

        Ok(Some(LayerStackDecodeDeviceTensors {
            hidden_states: current,
            layer_kv_cache,
        }))
    }

    pub(crate) fn forward_seed_device<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        backend: &B,
    ) -> Result<Option<LayerStackDecodeDeviceTensors>> {
        validate_hidden_states_f32(self.hidden_size, hidden_states)?;

        let mut current = crate::try_device!(backend.device_upload_f32_tensor(hidden_states));
        let mut layer_kv_cache = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            match layer {
                RuntimeLayer::Dense(block) => {
                    let layer_index = block.load_report().layer_index;
                    let output = match profile::run_layer_stage(
                        layer_index,
                        "dense.seed_device",
                        || block.forward_seed_device(config, &current, backend),
                    )? {
                        Some(output) => output,
                        None => {
                            return Err(Error::backend(format!(
                                    "device-batched dense seed layer {layer_index} has no complete native path"
                                )));
                        }
                    };
                    layer_kv_cache.push(LayerDeviceKvCacheTensors {
                        layer_index,
                        layer_kind: LayerKind::Dense,
                        cache_k: output.cache_k,
                        cache_v: output.cache_v,
                        index_key: output.index_key,
                    });
                    current = output.hidden_states;
                }
                RuntimeLayer::Sparse(block) => {
                    let layer_index = block.load_report().layer_index;
                    let output = match profile::run_layer_stage(
                        layer_index,
                        "sparse_moe.seed_device",
                        || block.forward_seed_device(config, &current, backend),
                    )? {
                        Some(output) => output,
                        None => {
                            return Err(Error::backend(format!(
                                    "device-batched sparse seed layer {layer_index} has no complete native path"
                                )));
                        }
                    };
                    layer_kv_cache.push(LayerDeviceKvCacheTensors {
                        layer_index,
                        layer_kind: LayerKind::SparseMoe,
                        cache_k: output.cache_k,
                        cache_v: output.cache_v,
                        index_key: output.index_key,
                    });
                    current = output.hidden_states;
                }
            }
        }

        Ok(Some(LayerStackDecodeDeviceTensors {
            hidden_states: current,
            layer_kv_cache,
        }))
    }
}

/// Result of the device-resident decode driver: final hidden states still on
/// the GPU plus the per-layer K/V tensors for the host cache appends.
pub(crate) struct LayerStackDecodeDeviceTensors {
    pub(crate) hidden_states: backend::DeviceValue,
    pub(crate) layer_kv_cache: Vec<LayerDeviceKvCacheTensors>,
}

fn past_kv_to_host<B: Backend>(
    past_kv: Option<(F32Tensor, F32Tensor)>,
    backend: &B,
) -> Result<Option<(Tensor, Tensor)>> {
    match past_kv {
        Some((cache_k, cache_v)) => Ok(Some((
            cache_tensor_to_host(&cache_k, backend.device())?,
            cache_tensor_to_host(&cache_v, backend.device())?,
        ))),
        None => Ok(None),
    }
}

fn capture_first_kv_shapes(
    k_shape: &mut Option<Shape>,
    v_shape: &mut Option<Shape>,
    cache_k: &Tensor,
    cache_v: &Tensor,
) {
    if k_shape.is_none() {
        *k_shape = Some(Shape::new(cache_k.dims().to_vec()));
    }
    if v_shape.is_none() {
        *v_shape = Some(Shape::new(cache_v.dims().to_vec()));
    }
}

fn validate_hidden_states(hidden_size: usize, hidden_states: &Tensor) -> Result<()> {
    let dims = hidden_states.dims();
    if dims.len() != 3 {
        return Err(Error::model(format!(
            "GLM-5.2 GGUF layer stack input must be rank 3 [B,T,H], got {dims:?}"
        )));
    }
    validate_exact_shape("gguf_layer_stack_hidden_size", &[dims[2]], &[hidden_size])
}

fn validate_hidden_states_f32(hidden_size: usize, hidden_states: &F32Tensor) -> Result<()> {
    let dims = hidden_states.dims();
    if dims.len() != 3 {
        return Err(Error::model(format!(
            "GLM-5.2 GGUF native layer stack input must be rank 3 [B,T,H], got {dims:?}"
        )));
    }
    validate_exact_shape(
        "gguf_native_layer_stack_hidden_size",
        &[dims[2]],
        &[hidden_size],
    )
}

fn tensor_to_f32_tensor(tensor: &Tensor) -> Result<F32Tensor> {
    let dims = tensor.dims().to_vec();
    let values = tensor
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    F32Tensor::new(values, dims)
}

fn tensor_from_f32_tensor(tensor: F32Tensor, device: &common::Device) -> Result<Tensor> {
    let (shape, values) = tensor.into_parts();
    Ok(Tensor::from_vec(values, shape.dims(), device)?)
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
    fn q2_layer_stack_runs_dense_prefix_then_sparse_layer() {
        let path = write_layer_stack_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let index = Index::from_gguf(&gguf, &config).unwrap();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let stack = LayerStack::open(&gguf, &config, &index, &backend, 256).unwrap();
        let hidden_states =
            Tensor::from_vec(vec![1.0_f32; 2 * 256], (1, 2, 256), &Device::Cpu).unwrap();

        let output = stack.forward(&config, &hidden_states, &backend).unwrap();
        let tensors = stack
            .forward_tensors(&config, &hidden_states, &backend)
            .unwrap();

        assert_eq!(stack.load_report().total_layers, 2);
        assert_eq!(stack.load_report().loaded_dense_layers, 1);
        assert_eq!(stack.load_report().loaded_sparse_layers, 1);
        assert_eq!(
            stack.load_report().layers[0].runtime_kind,
            LayerRuntimeKind::DenseBlock
        );
        assert_eq!(
            stack.load_report().layers[1].runtime_kind,
            LayerRuntimeKind::SparseBlock
        );
        assert_eq!(output.hidden_states.dims(), &[1, 2, 256]);
        assert_eq!(tensors.hidden_states.dims(), &[1, 2, 256]);
        assert_eq!(output.report.executed_layer_count, 2);
        assert_eq!(output.report.dense_block_count, 1);
        assert_eq!(output.report.sparse_block_count, 1);
        assert_eq!(output.report.kv_cache_layer_count, 2);
        assert_eq!(
            output.report.layer_k_cache_shape.as_ref().unwrap().dims(),
            &[1, 1, 2, 256]
        );
        assert_eq!(
            output.report.layer_v_cache_shape.as_ref().unwrap().dims(),
            &[1, 1, 2, 128]
        );
        assert_eq!(output.layer_kv_cache.len(), 2);
        assert_eq!(tensors.layer_kv_cache.len(), 2);
        assert_eq!(output.layer_kv_cache[0].layer_index, 0);
        assert_eq!(output.layer_kv_cache[1].layer_index, 1);
        assert_eq!(tensors.layer_kv_cache[0].layer_index, 0);
        assert_eq!(tensors.layer_kv_cache[1].layer_index, 1);
        assert_eq!(output.layer_kv_cache[0].cache_k.dims(), &[1, 1, 2, 256]);
        assert_eq!(output.layer_kv_cache[1].cache_k.dims(), &[1, 1, 2, 256]);
        assert_eq!(tensors.layer_kv_cache[0].cache_k.dims(), &[1, 1, 2, 256]);
        assert_eq!(tensors.layer_kv_cache[1].cache_k.dims(), &[1, 1, 2, 256]);
        assert_eq!(output.report.loaded_expert_requests, 2);
        assert!(output.report.routed_source_payload_bytes_read > 0);
        assert!(output.report.routed_peak_decoded_f32_bytes > 0);
    }

    #[test]
    fn q2_layer_stack_decode_reads_past_kv_for_each_layer() {
        let path = write_layer_stack_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let index = Index::from_gguf(&gguf, &config).unwrap();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let stack = LayerStack::open(&gguf, &config, &index, &backend, 256).unwrap();
        let prefill_hidden = Tensor::zeros((1, 2, 256)).unwrap();
        let prefill = stack.forward(&config, &prefill_hidden, &backend).unwrap();
        let decode_hidden = Tensor::zeros((1, 1, 256)).unwrap();
        let mut provider_calls = 0_usize;

        let decode = stack
            .forward_with_past_kv_provider(&config, &decode_hidden, &backend, |layer_index| {
                provider_calls += 1;
                Ok(prefill
                    .layer_kv_cache
                    .iter()
                    .find(|entry| entry.layer_index == layer_index)
                    .map(|entry| (entry.cache_k.clone(), entry.cache_v.clone())))
            })
            .unwrap();
        let mut tensor_provider_calls = 0_usize;
        let decode_tensors = stack
            .forward_tensors_with_past_kv_provider(
                &config,
                &decode_hidden,
                &backend,
                |layer_index| {
                    tensor_provider_calls += 1;
                    Ok(prefill
                        .layer_kv_cache
                        .iter()
                        .find(|entry| entry.layer_index == layer_index)
                        .map(|entry| (entry.cache_k.clone(), entry.cache_v.clone())))
                },
            )
            .unwrap();

        assert_eq!(provider_calls, 2);
        assert_eq!(tensor_provider_calls, 2);
        assert_eq!(decode.hidden_states.dims(), &[1, 1, 256]);
        assert_eq!(decode_tensors.hidden_states.dims(), &[1, 1, 256]);
        assert_eq!(decode.report.executed_layer_count, 2);
        assert_eq!(decode.layer_kv_cache.len(), 2);
        assert_eq!(decode_tensors.layer_kv_cache.len(), 2);
        assert_eq!(decode.layer_kv_cache[0].cache_k.dims(), &[1, 1, 1, 256]);
        assert_eq!(decode.layer_kv_cache[1].cache_k.dims(), &[1, 1, 1, 256]);
        assert_eq!(
            decode_tensors.layer_kv_cache[0].cache_k.dims(),
            &[1, 1, 1, 256]
        );
        assert_eq!(
            decode_tensors.layer_kv_cache[1].cache_k.dims(),
            &[1, 1, 1, 256]
        );
        assert_eq!(decode.report.max_attention_past_tokens, 2);
    }

    fn tiny_config() -> Config {
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
            kv_lora_rank: 256,
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
            index_head_dim: 128,
            index_n_heads: 32,
            index_topk_freq: 4,
            index_skip_topk_offset: 3,
            index_share_for_mtp_iteration: true,
            indexer_rope_interleave: true,
            indexer_types: Vec::new(),
            num_nextn_predict_layers: 0,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000_000.0,
        }
    }

    fn write_layer_stack_fixture(ty: GgmlType) -> PathBuf {
        let path = unique_temp_file("layer-stack");
        let mut specs = vec![
            TensorSpec::quant("token_embd.weight", vec![256, 8], ty, None),
            TensorSpec::f32("output_norm.weight", vec![256], None),
            TensorSpec::quant("output.weight", vec![256, 8], ty, None),
        ];
        insert_dense_layer(&mut specs, 0, ty);
        insert_sparse_layer(&mut specs, 1, ty);
        write_gguf(path, &specs)
    }

    fn insert_dense_layer(specs: &mut Vec<TensorSpec>, layer: usize, ty: GgmlType) {
        insert_attention(specs, layer, ty);
        specs.push(TensorSpec::quant(
            "blk.0.ffn_gate.weight",
            vec![256, 512],
            ty,
            None,
        ));
        specs.push(TensorSpec::quant(
            "blk.0.ffn_up.weight",
            vec![256, 512],
            ty,
            None,
        ));
        specs.push(TensorSpec::quant(
            "blk.0.ffn_down.weight",
            vec![512, 256],
            ty,
            None,
        ));
    }

    fn insert_sparse_layer(specs: &mut Vec<TensorSpec>, layer: usize, ty: GgmlType) {
        insert_attention(specs, layer, ty);
        specs.push(TensorSpec::f32(
            "blk.1.exp_probs_b.bias",
            vec![4],
            Some(vec![0.0, 0.4, 0.2, 0.8]),
        ));
        specs.push(TensorSpec::quant(
            "blk.1.ffn_gate_inp.weight",
            vec![256, 4],
            ty,
            None,
        ));
        specs.push(TensorSpec::quant(
            "blk.1.ffn_gate_shexp.weight",
            vec![256, 256],
            ty,
            None,
        ));
        specs.push(TensorSpec::quant(
            "blk.1.ffn_up_shexp.weight",
            vec![256, 256],
            ty,
            None,
        ));
        specs.push(TensorSpec::quant(
            "blk.1.ffn_down_shexp.weight",
            vec![256, 256],
            ty,
            None,
        ));
        specs.push(TensorSpec::quant(
            "blk.1.ffn_gate_exps.weight",
            vec![256, 256, 4],
            ty,
            None,
        ));
        specs.push(TensorSpec::quant(
            "blk.1.ffn_up_exps.weight",
            vec![256, 256, 4],
            ty,
            None,
        ));
        specs.push(TensorSpec::quant(
            "blk.1.ffn_down_exps.weight",
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

    fn write_gguf(path: PathBuf, specs: &[TensorSpec]) -> PathBuf {
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
                GgmlType::Q2K => {
                    for _ in 0..spec.payload_len() / GGML_Q2_K_BLOCK_BYTES {
                        writer.bytes(&vec![0_u8; GGML_Q2_K_BLOCK_BYTES as usize]);
                    }
                }
                GgmlType::Q8_0 => {
                    for _ in 0..spec.payload_len() / GGML_Q8_0_BLOCK_BYTES {
                        writer.bytes(&vec![0_u8; GGML_Q8_0_BLOCK_BYTES as usize]);
                    }
                }
                other => panic!("unsupported fixture tensor type {other}"),
            }
        }
        writer.pad_to_absolute_data_offset(offset);
        writer.finish_to(path)
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
        std::env::temp_dir().join(format!("inferno-{label}-{}-{id}", std::process::id()))
    }
}
