use backend::{Backend, BackendCapabilities, DevicePagedKvView, DeviceSelectedKvView, DeviceValue};
use common::Tensor;
use common::{validate_exact_shape, DeviceKind, Error, F32Tensor, PagedKvView, Result, Shape};
use config::Config;
use gguf::{
    matmul_q2_k_payload_f32, GgmlType, GgufFile, GgufQuantBlockKind, GGML_Q2_K_BLOCK_BYTES,
    GGML_Q8_0_BLOCK_BYTES,
};

use crate::{
    attention_math, profile, AttentionIndex, DsaIndexer, DsaIndexerLoadReport, LayerIndex,
    QuantizedLinear, RmsNorm, RmsNormLoadReport, TensorRef,
};

#[derive(Debug)]
pub struct Attention<'a> {
    layer_index: usize,
    input_norm: RmsNorm,
    q_a_norm: RmsNorm,
    kv_a_norm: RmsNorm,
    q_a: QuantizedLinear<'a>,
    q_b: QuantizedLinear<'a>,
    kv_a_mqa: QuantizedLinear<'a>,
    k_b: HeadProjection<'a>,
    v_b: HeadProjection<'a>,
    output: QuantizedLinear<'a>,
    indexer: Option<DsaIndexer<'a>>,
    kv_lora_rank: usize,
    load_report: AttentionLoadReport,
}

#[derive(Debug)]
pub struct AttentionOutput {
    pub hidden_states: Tensor,
    pub cache_k: Tensor,
    pub cache_v: Tensor,
    pub report: AttentionForwardReport,
}

#[derive(Debug)]
pub struct AttentionTensors {
    pub hidden_states: Tensor,
    pub cache_k: F32Tensor,
    pub cache_v: F32Tensor,
    pub index_key: Option<F32Tensor>,
}

#[derive(Debug)]
pub struct AttentionF32Tensors {
    pub hidden_states: F32Tensor,
    pub cache_k: F32Tensor,
    pub cache_v: F32Tensor,
    pub index_key: Option<F32Tensor>,
}

#[derive(Debug)]
pub struct SparseAttentionF32Tensors {
    pub tensors: AttentionF32Tensors,
    pub next_shared_selection: Option<Vec<u32>>,
}

/// Output of the batched device-resident decode attention path. The hidden
/// states and current token K/V stay on the GPU.
#[derive(Debug)]
pub(crate) struct AttentionDeviceTensors {
    pub(crate) hidden_states: DeviceValue,
    pub(crate) cache_k: DeviceValue,
    pub(crate) cache_v: DeviceValue,
    pub(crate) index_key: Option<DeviceValue>,
}

#[derive(Debug)]
pub(crate) struct SparseAttentionDeviceTensors {
    pub(crate) tensors: AttentionDeviceTensors,
    pub(crate) next_shared_selection: Option<Vec<u32>>,
}

#[derive(Debug)]
pub enum AttentionPastKv<'a> {
    Contiguous {
        cache_k: &'a F32Tensor,
        cache_v: &'a F32Tensor,
    },
}

#[derive(Debug, Clone, Copy)]
struct SparseAttentionContext<'a> {
    cached_index_keys: Option<&'a F32Tensor>,
    shared_selection: Option<&'a [u32]>,
}

#[derive(Debug)]
struct SparsePastSelection {
    selected_past_tokens: Option<Vec<u32>>,
    next_shared_selection: Option<Vec<u32>>,
    current_index_key: Option<F32Tensor>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionLoadReport {
    pub backend: BackendCapabilities,
    pub layer_index: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub input_norm: RmsNormLoadReport,
    pub q_a_norm: RmsNormLoadReport,
    pub kv_a_norm: RmsNormLoadReport,
    pub q_a_weight_shape: Shape,
    pub q_b_weight_shape: Shape,
    pub kv_a_mqa_weight_shape: Shape,
    pub k_b_weight_shape: Shape,
    pub v_b_weight_shape: Shape,
    pub output_weight_shape: Shape,
    pub indexer: Option<DsaIndexerLoadReport>,
    pub projection_tensor_type: GgmlType,
    pub output_chunk_rows: usize,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionForwardReport {
    pub layer_index: usize,
    pub past_tokens: usize,
    pub input_hidden_states_shape: Shape,
    pub normed_hidden_states_shape: Shape,
    pub q_a_shape: Shape,
    pub q_a_norm_shape: Shape,
    pub q_b_shape: Shape,
    pub q_b_chunk_count: usize,
    pub q_heads_shape: Shape,
    pub q_no_rope_shape: Shape,
    pub q_rope_shape: Shape,
    pub q_rope_after_rope_shape: Shape,
    pub kv_a_mqa_shape: Shape,
    pub kv_a_mqa_chunk_count: usize,
    pub kv_latent_shape: Shape,
    pub k_rope_mqa_shape: Shape,
    pub kv_a_norm_shape: Shape,
    pub k_b_shape: Shape,
    pub k_b_chunk_count: usize,
    pub v_b_shape: Shape,
    pub v_b_chunk_count: usize,
    pub k_no_rope_shape: Shape,
    pub k_rope_after_rope_shape: Shape,
    pub k_heads_shape: Shape,
    pub v_heads_shape: Shape,
    pub cache_k_shape: Shape,
    pub cache_v_shape: Shape,
    pub attention_k_shape: Shape,
    pub attention_v_shape: Shape,
    pub raw_attention_scores_shape: Shape,
    pub attention_scores_shape: Shape,
    pub attention_probs_shape: Shape,
    pub context_heads_shape: Shape,
    pub merged_attention_output_shape: Shape,
    pub output_projection_shape: Shape,
    pub output_projection_chunk_count: usize,
    pub output_hidden_states_shape: Shape,
}

#[derive(Debug)]
struct HeadProjection<'a> {
    gguf: &'a GgufFile,
    tensor_ref: TensorRef,
    layout: HeadProjectionLayout,
    input_features: usize,
    output_features: usize,
    heads: usize,
    blocks_per_head: u64,
    q2_payload_bytes: Option<&'a [u8]>,
    q8_payload_bytes: Option<&'a [u8]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HeadProjectionLayout {
    OutputInputHeads,
    InputOutputHeads,
}

#[derive(Debug)]
struct HeadProjectionOutput {
    output_heads: Tensor,
    chunk_count: usize,
}

struct HeadProjectionF32Output {
    output_heads: F32Tensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DevicePastKvLayout {
    ExpandedHeads,
    MlaLatent,
}

impl<'a> HeadProjection<'a> {
    fn q8_payload(&self, context: &str) -> Result<&'a [u8]> {
        self.q8_payload_bytes.ok_or_else(|| {
            Error::gguf(format!(
                "{context} requires Q8_0 tensor {}, got {}",
                self.tensor_ref.name, self.tensor_ref.ty
            ))
        })
    }

    fn open_output_input_heads(
        gguf: &'a GgufFile,
        tensor_ref: &TensorRef,
        input_features: usize,
        output_features: usize,
        heads: usize,
    ) -> Result<Self> {
        Self::open(
            gguf,
            tensor_ref,
            HeadProjectionLayout::OutputInputHeads,
            input_features,
            output_features,
            heads,
        )
    }

    fn open_input_output_heads(
        gguf: &'a GgufFile,
        tensor_ref: &TensorRef,
        input_features: usize,
        output_features: usize,
        heads: usize,
    ) -> Result<Self> {
        Self::open(
            gguf,
            tensor_ref,
            HeadProjectionLayout::InputOutputHeads,
            input_features,
            output_features,
            heads,
        )
    }

    fn open(
        gguf: &'a GgufFile,
        tensor_ref: &TensorRef,
        layout: HeadProjectionLayout,
        input_features: usize,
        output_features: usize,
        heads: usize,
    ) -> Result<Self> {
        let info = gguf.tensor(&tensor_ref.name).ok_or_else(|| {
            Error::gguf(format!("missing GLM-5.2 GGUF tensor {}", tensor_ref.name))
        })?;
        validate_tensor_ref(tensor_ref, info)?;
        validate_quantized_projection_type("gguf_attention_head_projection", tensor_ref)?;
        let storage = gguf.tensor_quantized_storage(&tensor_ref.name)?;
        let values_per_block = usize::try_from(storage.block.values_per_block()).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} quant block size does not fit usize",
                tensor_ref.name
            ))
        })?;
        let first_dim = usize::try_from(tensor_ref.dims[0]).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} first dimension does not fit usize",
                tensor_ref.name
            ))
        })?;
        if first_dim % values_per_block != 0 {
            return Err(Error::gguf(format!(
                "GGUF tensor {} first dimension {first_dim} must be divisible by {} block size {values_per_block}",
                tensor_ref.name, storage.block
            )));
        }
        let blocks_per_row = u64::try_from(first_dim / values_per_block).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} blocks_per_row does not fit u64",
                tensor_ref.name
            ))
        })?;
        let rows_per_head = match layout {
            HeadProjectionLayout::OutputInputHeads => input_features,
            HeadProjectionLayout::InputOutputHeads => output_features,
        };
        let blocks_per_head = blocks_per_row
            .checked_mul(u64::try_from(rows_per_head).map_err(|_| {
                Error::gguf(format!(
                    "GGUF tensor {} rows_per_head does not fit u64",
                    tensor_ref.name
                ))
            })?)
            .ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} blocks_per_head overflow",
                    tensor_ref.name
                ))
            })?;
        let expected_blocks = blocks_per_head
            .checked_mul(u64::try_from(heads).map_err(|_| {
                Error::gguf(format!(
                    "GGUF tensor {} head count does not fit u64",
                    tensor_ref.name
                ))
            })?)
            .ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} expected block count overflow",
                    tensor_ref.name
                ))
            })?;
        if storage.block_count != expected_blocks {
            return Err(Error::gguf(format!(
                "GGUF tensor {} has {} quant blocks but per-head projection shape requires {expected_blocks}",
                tensor_ref.name, storage.block_count
            )));
        }
        let q2_payload_bytes = if storage.block == GgufQuantBlockKind::Q2K {
            Some(storage.bytes)
        } else {
            None
        };
        let q8_payload_bytes = if storage.block == GgufQuantBlockKind::Q8_0 {
            Some(storage.bytes)
        } else {
            None
        };
        Ok(Self {
            gguf,
            tensor_ref: tensor_ref.clone(),
            layout,
            input_features,
            output_features,
            heads,
            blocks_per_head,
            q2_payload_bytes,
            q8_payload_bytes,
        })
    }

    fn forward_heads<B: Backend>(
        &self,
        input: &Tensor,
        backend: &B,
    ) -> Result<HeadProjectionOutput> {
        let dims = input.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF head projection input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            format!(
                "gguf_attention_head_projection_input:{}",
                self.tensor_ref.name
            ),
            dims,
            &[batch, tokens, self.input_features],
        )?;
        if self.layout == HeadProjectionLayout::InputOutputHeads
            && self.tensor_ref.ty == GgmlType::Q2K
        {
            return self.forward_input_output_heads_q2(input, backend, batch, tokens);
        }
        if self.layout == HeadProjectionLayout::OutputInputHeads
            && self.tensor_ref.ty == GgmlType::Q2K
        {
            return self.forward_output_input_heads_q2(input, backend, batch, tokens);
        }

        let flat_tokens = batch
            .checked_mul(tokens)
            .ok_or_else(|| Error::model("GLM head projection batch*tokens overflow"))?;
        let flat_input = input
            .contiguous()?
            .reshape((flat_tokens, self.input_features))?;
        let storage = self.gguf.tensor_quantized_storage(&self.tensor_ref.name)?;
        let mut head_outputs = Vec::with_capacity(self.heads);

        for head_index in 0..self.heads {
            let block_start = u64::try_from(head_index)
                .ok()
                .and_then(|head| head.checked_mul(self.blocks_per_head))
                .ok_or_else(|| {
                    Error::gguf(format!(
                        "GGUF tensor {} head block offset overflow",
                        self.tensor_ref.name
                    ))
                })?;
            let head_values =
                storage.dequantize_block_range_as_f32(block_start, self.blocks_per_head)?;
            let weight = match self.layout {
                HeadProjectionLayout::OutputInputHeads => Tensor::from_vec(
                    head_values,
                    (self.input_features, self.output_features),
                    backend.device(),
                )?,
                HeadProjectionLayout::InputOutputHeads => Tensor::from_vec(
                    head_values,
                    (self.output_features, self.input_features),
                    backend.device(),
                )?
                .t()?
                .contiguous()?,
            };
            let head_output = backend.matmul(&flat_input, &weight)?.unsqueeze(1)?;
            head_outputs.push(head_output);
        }

        let head_refs = head_outputs.iter().collect::<Vec<_>>();
        let output_heads = Tensor::cat(&head_refs, 1)?.reshape((
            batch,
            tokens,
            self.heads,
            self.output_features,
        ))?;

        Ok(HeadProjectionOutput {
            output_heads,
            chunk_count: self.heads,
        })
    }

    fn forward_heads_f32<B: Backend>(
        &self,
        input: &F32Tensor,
        backend: &B,
    ) -> Result<HeadProjectionF32Output> {
        let dims = input.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF native head projection input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            format!(
                "gguf_native_attention_head_projection_input:{}",
                self.tensor_ref.name
            ),
            dims,
            &[batch, tokens, self.input_features],
        )?;
        match (self.tensor_ref.ty, self.layout) {
            (GgmlType::Q2K, HeadProjectionLayout::InputOutputHeads) => {
                self.forward_input_output_heads_q2_f32(input, backend, batch, tokens)
            }
            (GgmlType::Q2K, HeadProjectionLayout::OutputInputHeads) => {
                self.forward_output_input_heads_q2_f32(input, backend, batch, tokens)
            }
            (GgmlType::Q8_0, HeadProjectionLayout::InputOutputHeads) => {
                self.forward_input_output_heads_q8_f32(input, backend, batch, tokens)
            }
            (GgmlType::Q8_0, HeadProjectionLayout::OutputInputHeads) => {
                self.forward_output_input_heads_q8_f32(input, backend, batch, tokens)
            }
            (other, _) => Err(Error::gguf(format!(
                "GGUF tensor {} must be Q2_K or Q8_0 for native head projection, got {}",
                self.tensor_ref.name, other
            ))),
        }
    }

    /// Batched device-resident variant of `forward_heads_f32`.
    fn forward_heads_device<B: Backend>(
        &self,
        input: &DeviceValue,
        backend: &B,
    ) -> Result<Option<DeviceValue>> {
        let dims = input.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF device head projection input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            format!(
                "gguf_device_attention_head_projection_input:{}",
                self.tensor_ref.name
            ),
            dims,
            &[batch, tokens, self.input_features],
        )?;
        let flat_tokens = batch
            .checked_mul(tokens)
            .ok_or_else(|| Error::model("GLM device head projection batch*tokens overflow"))?;
        let flat_input = input.reshape(vec![flat_tokens, self.input_features])?;
        let merged_output_features = self
            .heads
            .checked_mul(self.output_features)
            .ok_or_else(|| Error::model("GLM device head projection merged width overflow"))?;

        match (self.tensor_ref.ty, self.layout) {
            (GgmlType::Q2K, HeadProjectionLayout::InputOutputHeads) => {
                let raw_data = self.q2_payload_bytes.ok_or_else(|| {
                    Error::gguf(format!(
                        "GGUF tensor {} must be Q2_K for the device Q2 head projection path, got {}",
                        self.tensor_ref.name, self.tensor_ref.ty
                    ))
                })?;
                let flat_output = crate::try_device!(backend.q2_k_matvec_device(
                    raw_data,
                    &flat_input,
                    flat_tokens,
                    self.input_features,
                    merged_output_features,
                ));
                Ok(Some(flat_output.reshape(vec![
                    batch,
                    tokens,
                    self.heads,
                    self.output_features,
                ])?))
            }
            (GgmlType::Q8_0, HeadProjectionLayout::InputOutputHeads) => {
                let raw_data = self.q8_payload_bytes.ok_or_else(|| {
                    Error::gguf(format!(
                        "GGUF tensor {} must be Q8_0 for the device Q8 head projection path, got {}",
                        self.tensor_ref.name, self.tensor_ref.ty
                    ))
                })?;
                let flat_output = crate::try_device!(backend.q8_0_matvec_device(
                    raw_data,
                    &flat_input,
                    flat_tokens,
                    self.input_features,
                    merged_output_features,
                ));
                Ok(Some(flat_output.reshape(vec![
                    batch,
                    tokens,
                    self.heads,
                    self.output_features,
                ])?))
            }
            (GgmlType::Q2K, HeadProjectionLayout::OutputInputHeads)
            | (GgmlType::Q8_0, HeadProjectionLayout::OutputInputHeads) => {
                self.forward_transposed_heads_device(backend, &flat_input, batch, tokens)
            }
            (other, _) => Err(Error::gguf(format!(
                "GGUF tensor {} must be Q2_K or Q8_0 for device head projection, got {}",
                self.tensor_ref.name, other
            ))),
        }
    }

    /// Transposed head projection over the complete packed head tensor. One
    /// dispatch writes `[batch, tokens, heads, output_features]` directly.
    fn forward_transposed_heads_device<B: Backend>(
        &self,
        backend: &B,
        flat_input: &DeviceValue,
        batch: usize,
        tokens: usize,
    ) -> Result<Option<DeviceValue>> {
        let flat_tokens = batch.checked_mul(tokens).ok_or_else(|| {
            Error::model("GLM device transposed head projection row count overflow")
        })?;
        let packed = match self.tensor_ref.ty {
            GgmlType::Q2K => {
                let raw_data = self.q2_payload_bytes.ok_or_else(|| {
                    Error::gguf(format!(
                        "GGUF tensor {} has no Q2_K packed head payload",
                        self.tensor_ref.name
                    ))
                })?;
                crate::try_device!(backend.q2_k_packed_heads_transposed_matvec_device(
                    raw_data,
                    flat_input,
                    flat_tokens,
                    self.heads,
                    self.input_features,
                    self.output_features,
                ))
            }
            GgmlType::Q8_0 => {
                let raw_data = self.q8_payload_bytes.ok_or_else(|| {
                    Error::gguf(format!(
                        "GGUF tensor {} has no Q8_0 packed head payload",
                        self.tensor_ref.name
                    ))
                })?;
                crate::try_device!(backend.q8_0_packed_heads_transposed_matvec_device(
                    raw_data,
                    flat_input,
                    flat_tokens,
                    self.heads,
                    self.input_features,
                    self.output_features,
                ))
            }
            other => {
                return Err(Error::gguf(format!(
                    "GGUF tensor {} must be Q2_K or Q8_0 for packed head projection, got {other}",
                    self.tensor_ref.name
                )))
            }
        };
        Ok(Some(packed.reshape(vec![
            batch,
            tokens,
            self.heads,
            self.output_features,
        ])?))
    }

    fn forward_input_output_heads_q2<B: Backend>(
        &self,
        input: &Tensor,
        backend: &B,
        batch: usize,
        tokens: usize,
    ) -> Result<HeadProjectionOutput> {
        let flat_tokens = batch
            .checked_mul(tokens)
            .ok_or_else(|| Error::model("GLM Q2 head projection batch*tokens overflow"))?;
        let flat_input = input
            .contiguous()?
            .reshape((flat_tokens, self.input_features))?;
        let merged_output_features = self
            .heads
            .checked_mul(self.output_features)
            .ok_or_else(|| Error::model("GLM Q2 head projection merged output width overflow"))?;
        let raw_data = self.q2_payload_bytes.ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} must be Q2_K for the native Q2 head projection path, got {}",
                self.tensor_ref.name, self.tensor_ref.ty
            ))
        })?;

        let flat_output = if let Some(output) = backend.q2_k_matvec_f32(
            raw_data,
            &flat_input,
            flat_tokens,
            self.input_features,
            merged_output_features,
        )? {
            output
        } else {
            reject_missing_native_q2_kernel_on_metal(backend, &self.tensor_ref.name)?;
            let input_values = flat_input
                .to_vec2::<f32>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            let output_values = matmul_q2_k_payload_f32(
                &self.tensor_ref.name,
                raw_data,
                &input_values,
                flat_tokens,
                self.input_features,
                merged_output_features,
            )?;
            Tensor::from_vec(
                output_values,
                (flat_tokens, merged_output_features),
                backend.device(),
            )?
        };
        validate_exact_shape(
            format!(
                "gguf_attention_q2_head_projection_output:{}",
                self.tensor_ref.name
            ),
            flat_output.dims(),
            &[flat_tokens, merged_output_features],
        )?;
        let output_heads =
            flat_output.reshape((batch, tokens, self.heads, self.output_features))?;

        Ok(HeadProjectionOutput {
            output_heads,
            chunk_count: 1,
        })
    }

    fn forward_input_output_heads_q2_f32<B: Backend>(
        &self,
        input: &F32Tensor,
        backend: &B,
        batch: usize,
        tokens: usize,
    ) -> Result<HeadProjectionF32Output> {
        let flat_tokens = batch
            .checked_mul(tokens)
            .ok_or_else(|| Error::model("GLM native Q2 head projection batch*tokens overflow"))?;
        let flat_input = input.clone().reshape([flat_tokens, self.input_features])?;
        let merged_output_features =
            self.heads
                .checked_mul(self.output_features)
                .ok_or_else(|| {
                    Error::model("GLM native Q2 head projection merged output width overflow")
                })?;
        let raw_data = self.q2_payload_bytes.ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} must be Q2_K for the native Q2 head projection path, got {}",
                self.tensor_ref.name, self.tensor_ref.ty
            ))
        })?;

        let flat_output = backend
            .q2_k_matvec_f32_tensor(
                raw_data,
                &flat_input,
                flat_tokens,
                self.input_features,
                merged_output_features,
            )?
            .ok_or_else(|| {
                Error::backend(format!(
                    "native Metal Q2 head projection is required for {}",
                    self.tensor_ref.name
                ))
            })?;
        validate_exact_shape(
            format!(
                "gguf_native_attention_q2_head_projection_output:{}",
                self.tensor_ref.name
            ),
            flat_output.dims(),
            &[flat_tokens, merged_output_features],
        )?;
        let output_heads =
            flat_output.reshape([batch, tokens, self.heads, self.output_features])?;

        Ok(HeadProjectionF32Output { output_heads })
    }

    fn forward_input_output_heads_q8_f32<B: Backend>(
        &self,
        input: &F32Tensor,
        backend: &B,
        batch: usize,
        tokens: usize,
    ) -> Result<HeadProjectionF32Output> {
        let flat_tokens = batch
            .checked_mul(tokens)
            .ok_or_else(|| Error::model("GLM native Q8 head projection batch*tokens overflow"))?;
        let flat_input = input.clone().reshape([flat_tokens, self.input_features])?;
        let merged_output_features =
            self.heads
                .checked_mul(self.output_features)
                .ok_or_else(|| {
                    Error::model("GLM native Q8 head projection merged output width overflow")
                })?;
        let raw_data = self.q8_payload_bytes.ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} must be Q8_0 for the native Q8 head projection path, got {}",
                self.tensor_ref.name, self.tensor_ref.ty
            ))
        })?;

        let flat_output = backend
            .q8_0_matvec_f32_tensor(
                raw_data,
                &flat_input,
                flat_tokens,
                self.input_features,
                merged_output_features,
            )?
            .ok_or_else(|| {
                Error::backend(format!(
                    "native Metal Q8_0 head projection is required for {}",
                    self.tensor_ref.name
                ))
            })?;
        validate_exact_shape(
            format!(
                "gguf_native_attention_q8_head_projection_output:{}",
                self.tensor_ref.name
            ),
            flat_output.dims(),
            &[flat_tokens, merged_output_features],
        )?;
        let output_heads =
            flat_output.reshape([batch, tokens, self.heads, self.output_features])?;

        Ok(HeadProjectionF32Output { output_heads })
    }

    fn forward_output_input_heads_q2<B: Backend>(
        &self,
        input: &Tensor,
        backend: &B,
        batch: usize,
        tokens: usize,
    ) -> Result<HeadProjectionOutput> {
        let flat_tokens = batch.checked_mul(tokens).ok_or_else(|| {
            Error::model("GLM transposed Q2 head projection batch*tokens overflow")
        })?;
        let flat_input = input
            .contiguous()?
            .reshape((flat_tokens, self.input_features))?;
        let raw_data = self.q2_payload_bytes.ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} must be Q2_K for the native transposed Q2 head projection path, got {}",
                self.tensor_ref.name, self.tensor_ref.ty
            ))
        })?;
        let head_byte_len = usize::try_from(self.blocks_per_head)
            .ok()
            .and_then(|blocks| blocks.checked_mul(GGML_Q2_K_BLOCK_BYTES as usize))
            .ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} transposed Q2 head byte length overflow",
                    self.tensor_ref.name
                ))
            })?;
        let expected_payload_len = head_byte_len.checked_mul(self.heads).ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} transposed Q2 payload length overflow",
                self.tensor_ref.name
            ))
        })?;
        validate_exact_shape(
            format!(
                "gguf_attention_transposed_q2_head_projection_payload:{}",
                self.tensor_ref.name
            ),
            &[raw_data.len()],
            &[expected_payload_len],
        )?;

        let mut head_outputs = Vec::with_capacity(self.heads);
        for head_index in 0..self.heads {
            let byte_start = head_index.checked_mul(head_byte_len).ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} transposed Q2 head byte offset overflow",
                    self.tensor_ref.name
                ))
            })?;
            let byte_end = byte_start.checked_add(head_byte_len).ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} transposed Q2 head byte end overflow",
                    self.tensor_ref.name
                ))
            })?;
            let head_bytes = &raw_data[byte_start..byte_end];

            let flat_output = if let Some(output) = backend.q2_k_transposed_matvec_f32(
                head_bytes,
                &flat_input,
                flat_tokens,
                self.input_features,
                self.output_features,
            )? {
                output
            } else {
                reject_missing_native_q2_kernel_on_metal(backend, &self.tensor_ref.name)?;
                self.forward_output_input_head_reference(
                    head_index,
                    &flat_input,
                    backend,
                    flat_tokens,
                )?
            };
            validate_exact_shape(
                format!(
                    "gguf_attention_transposed_q2_head_projection_output:{}",
                    self.tensor_ref.name
                ),
                flat_output.dims(),
                &[flat_tokens, self.output_features],
            )?;
            head_outputs.push(flat_output.unsqueeze(1)?);
        }

        let head_refs = head_outputs.iter().collect::<Vec<_>>();
        let output_heads = Tensor::cat(&head_refs, 1)?.reshape((
            batch,
            tokens,
            self.heads,
            self.output_features,
        ))?;

        Ok(HeadProjectionOutput {
            output_heads,
            chunk_count: self.heads,
        })
    }

    fn forward_output_input_heads_q2_f32<B: Backend>(
        &self,
        input: &F32Tensor,
        backend: &B,
        batch: usize,
        tokens: usize,
    ) -> Result<HeadProjectionF32Output> {
        let flat_tokens = batch.checked_mul(tokens).ok_or_else(|| {
            Error::model("GLM native transposed Q2 head projection batch*tokens overflow")
        })?;
        let flat_input = input.clone().reshape([flat_tokens, self.input_features])?;
        let raw_data = self.q2_payload_bytes.ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} must be Q2_K for the native transposed Q2 head projection path, got {}",
                self.tensor_ref.name, self.tensor_ref.ty
            ))
        })?;
        let head_byte_len = usize::try_from(self.blocks_per_head)
            .ok()
            .and_then(|blocks| blocks.checked_mul(GGML_Q2_K_BLOCK_BYTES as usize))
            .ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} native transposed Q2 head byte length overflow",
                    self.tensor_ref.name
                ))
            })?;
        let expected_payload_len = head_byte_len.checked_mul(self.heads).ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} native transposed Q2 payload length overflow",
                self.tensor_ref.name
            ))
        })?;
        validate_exact_shape(
            format!(
                "gguf_native_attention_transposed_q2_head_projection_payload:{}",
                self.tensor_ref.name
            ),
            &[raw_data.len()],
            &[expected_payload_len],
        )?;

        let mut output_values = vec![0.0_f32; flat_tokens * self.heads * self.output_features];
        for head_index in 0..self.heads {
            let byte_start = head_index.checked_mul(head_byte_len).ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} native transposed Q2 head byte offset overflow",
                    self.tensor_ref.name
                ))
            })?;
            let byte_end = byte_start.checked_add(head_byte_len).ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} native transposed Q2 head byte end overflow",
                    self.tensor_ref.name
                ))
            })?;
            let head_bytes = &raw_data[byte_start..byte_end];
            let flat_output = backend
                .q2_k_transposed_matvec_f32_tensor(
                    head_bytes,
                    &flat_input,
                    flat_tokens,
                    self.input_features,
                    self.output_features,
                )?
                .ok_or_else(|| {
                    Error::backend(format!(
                        "native Metal transposed Q2 head projection is required for {}",
                        self.tensor_ref.name
                    ))
                })?;
            validate_exact_shape(
                format!(
                    "gguf_native_attention_transposed_q2_head_projection_output:{}",
                    self.tensor_ref.name
                ),
                flat_output.dims(),
                &[flat_tokens, self.output_features],
            )?;
            for token_index in 0..flat_tokens {
                for dim_index in 0..self.output_features {
                    let source = token_index * self.output_features + dim_index;
                    let target =
                        (token_index * self.heads + head_index) * self.output_features + dim_index;
                    output_values[target] = flat_output.values()[source];
                }
            }
        }

        Ok(HeadProjectionF32Output {
            output_heads: F32Tensor::new(
                output_values,
                [batch, tokens, self.heads, self.output_features],
            )?,
        })
    }

    fn forward_output_input_heads_q8_f32<B: Backend>(
        &self,
        input: &F32Tensor,
        backend: &B,
        batch: usize,
        tokens: usize,
    ) -> Result<HeadProjectionF32Output> {
        let flat_tokens = batch.checked_mul(tokens).ok_or_else(|| {
            Error::model("GLM native transposed Q8 head projection batch*tokens overflow")
        })?;
        let flat_input = input.clone().reshape([flat_tokens, self.input_features])?;
        let raw_data = self.q8_payload_bytes.ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} must be Q8_0 for the native transposed Q8 head projection path, got {}",
                self.tensor_ref.name, self.tensor_ref.ty
            ))
        })?;
        let head_byte_len = usize::try_from(self.blocks_per_head)
            .ok()
            .and_then(|blocks| blocks.checked_mul(GGML_Q8_0_BLOCK_BYTES as usize))
            .ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} native transposed Q8 head byte length overflow",
                    self.tensor_ref.name
                ))
            })?;
        let expected_payload_len = head_byte_len.checked_mul(self.heads).ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} native transposed Q8 payload length overflow",
                self.tensor_ref.name
            ))
        })?;
        validate_exact_shape(
            format!(
                "gguf_native_attention_transposed_q8_head_projection_payload:{}",
                self.tensor_ref.name
            ),
            &[raw_data.len()],
            &[expected_payload_len],
        )?;

        let mut output_values = vec![0.0_f32; flat_tokens * self.heads * self.output_features];
        for head_index in 0..self.heads {
            let byte_start = head_index.checked_mul(head_byte_len).ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} native transposed Q8 head byte offset overflow",
                    self.tensor_ref.name
                ))
            })?;
            let byte_end = byte_start.checked_add(head_byte_len).ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} native transposed Q8 head byte end overflow",
                    self.tensor_ref.name
                ))
            })?;
            let head_bytes = &raw_data[byte_start..byte_end];
            let flat_output = backend
                .q8_0_transposed_matvec_f32_tensor(
                    head_bytes,
                    &flat_input,
                    flat_tokens,
                    self.input_features,
                    self.output_features,
                )?
                .ok_or_else(|| {
                    Error::backend(format!(
                        "native Metal transposed Q8_0 head projection is required for {}",
                        self.tensor_ref.name
                    ))
                })?;
            validate_exact_shape(
                format!(
                    "gguf_native_attention_transposed_q8_head_projection_output:{}",
                    self.tensor_ref.name
                ),
                flat_output.dims(),
                &[flat_tokens, self.output_features],
            )?;
            for token_index in 0..flat_tokens {
                for dim_index in 0..self.output_features {
                    let source = token_index * self.output_features + dim_index;
                    let target =
                        (token_index * self.heads + head_index) * self.output_features + dim_index;
                    output_values[target] = flat_output.values()[source];
                }
            }
        }

        Ok(HeadProjectionF32Output {
            output_heads: F32Tensor::new(
                output_values,
                [batch, tokens, self.heads, self.output_features],
            )?,
        })
    }

    fn forward_output_input_head_reference<B: Backend>(
        &self,
        head_index: usize,
        flat_input: &Tensor,
        backend: &B,
        flat_tokens: usize,
    ) -> Result<Tensor> {
        let storage = self.gguf.tensor_quantized_storage(&self.tensor_ref.name)?;
        let block_start = u64::try_from(head_index)
            .ok()
            .and_then(|head| head.checked_mul(self.blocks_per_head))
            .ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} head block offset overflow",
                    self.tensor_ref.name
                ))
            })?;
        let head_values =
            storage.dequantize_block_range_as_f32(block_start, self.blocks_per_head)?;
        let weight = Tensor::from_vec(
            head_values,
            (self.input_features, self.output_features),
            backend.device(),
        )?;
        let output = backend.matmul(flat_input, &weight)?;
        validate_exact_shape(
            format!(
                "gguf_attention_transposed_q2_reference_head_projection_output:{}",
                self.tensor_ref.name
            ),
            output.dims(),
            &[flat_tokens, self.output_features],
        )?;
        Ok(output)
    }
}

impl<'a> Attention<'a> {
    pub fn open<B: Backend>(
        gguf: &'a GgufFile,
        config: &Config,
        layer: &LayerIndex,
        backend: &B,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        Self::open_from_parts(
            gguf,
            config,
            layer.layer_index,
            &layer.input_norm,
            &layer.attention,
            backend,
            output_chunk_rows,
        )
    }

    pub fn open_from_parts<B: Backend>(
        gguf: &'a GgufFile,
        config: &Config,
        layer_index: usize,
        input_norm_ref: &TensorRef,
        attention: &AttentionIndex,
        backend: &B,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        let is_single_mtp_layer =
            layer_index == config.num_layers && config.num_nextn_predict_layers == 1;
        if layer_index >= config.num_layers && !is_single_mtp_layer {
            return Err(Error::gguf(format!(
                "GLM-5.2 GGUF attention layer {layer_index} exceeds num_layers {}",
                config.num_layers
            )));
        }
        attention_math::validate_attention_config(config)?;

        let q_lora_rank = validate_q_a(config, &attention.q_a)?;
        validate_linear_ref(
            "gguf_attention_q_b",
            &attention.q_b,
            q_lora_rank,
            config.attention_heads * config.qk_head_dim,
        )?;
        let kv_lora_rank = validate_kv_a(config, &attention.kv_a_mqa)?;
        validate_k_b_ref(config, kv_lora_rank, &attention.k_b)?;
        validate_v_b_ref(config, kv_lora_rank, &attention.v_b)?;
        validate_linear_ref(
            "gguf_attention_output",
            &attention.output,
            config.merged_attention_width(),
            config.hidden_size,
        )?;

        let input_norm = RmsNorm::open(gguf, config, input_norm_ref, backend)?;
        let q_a_norm = RmsNorm::open_with_expected_size(
            gguf,
            &attention.q_a_norm,
            q_lora_rank,
            config.rms_norm_eps as f32,
            backend,
        )?;
        let kv_a_norm = RmsNorm::open_with_expected_size(
            gguf,
            &attention.kv_a_norm,
            kv_lora_rank,
            config.rms_norm_eps as f32,
            backend,
        )?;
        let q_a = QuantizedLinear::open(
            gguf,
            &attention.q_a,
            config.hidden_size,
            q_lora_rank,
            output_chunk_rows,
        )?;
        let q_b = QuantizedLinear::open(
            gguf,
            &attention.q_b,
            q_lora_rank,
            config.attention_heads * config.qk_head_dim,
            output_chunk_rows,
        )?;
        let kv_a_mqa = QuantizedLinear::open(
            gguf,
            &attention.kv_a_mqa,
            config.hidden_size,
            kv_lora_rank + config.qk_rope_dim,
            output_chunk_rows,
        )?;
        let k_b = HeadProjection::open_output_input_heads(
            gguf,
            &attention.k_b,
            kv_lora_rank,
            config.qk_no_rope_dim,
            config.attention_heads,
        )?;
        let v_b = HeadProjection::open_input_output_heads(
            gguf,
            &attention.v_b,
            kv_lora_rank,
            config.v_head_dim(),
            config.attention_heads,
        )?;
        let output = QuantizedLinear::open(
            gguf,
            &attention.output,
            config.merged_attention_width(),
            config.hidden_size,
            output_chunk_rows,
        )?;
        let indexer = match attention.indexer.as_ref() {
            Some(indexer) => DsaIndexer::open(
                gguf,
                config,
                layer_index,
                indexer,
                q_lora_rank,
                backend,
                output_chunk_rows,
            )?,
            None => None,
        };

        let load_report = AttentionLoadReport {
            backend: backend.capabilities(),
            layer_index,
            q_lora_rank,
            kv_lora_rank,
            input_norm: input_norm.load_report().clone(),
            q_a_norm: q_a_norm.load_report().clone(),
            kv_a_norm: kv_a_norm.load_report().clone(),
            q_a_weight_shape: logical_weight_shape(&attention.q_a)?,
            q_b_weight_shape: logical_weight_shape(&attention.q_b)?,
            kv_a_mqa_weight_shape: logical_weight_shape(&attention.kv_a_mqa)?,
            k_b_weight_shape: head_projection_logical_weight_shape(
                config.attention_heads,
                config.qk_no_rope_dim,
                kv_lora_rank,
            ),
            v_b_weight_shape: head_projection_logical_weight_shape(
                config.attention_heads,
                config.v_head_dim(),
                kv_lora_rank,
            ),
            output_weight_shape: logical_weight_shape(&attention.output)?,
            indexer: indexer.as_ref().map(|indexer| indexer.load_report().clone()),
            projection_tensor_type: attention.q_a.ty,
            output_chunk_rows,
            limitations: vec![
                "GGUF attention uses split k_b and v_b tensors as stored by Antirez GLM-5.2 GGUF"
                    .to_string(),
                "non-Q2 projection tensors still use chunked reference decoding".to_string(),
                "DSA indexer top-k currently uses the host reference path before selected Metal attention"
                    .to_string(),
            ],
        };

        Ok(Self {
            layer_index,
            input_norm,
            q_a_norm,
            kv_a_norm,
            q_a,
            q_b,
            kv_a_mqa,
            k_b,
            v_b,
            output,
            indexer,
            kv_lora_rank,
            load_report,
        })
    }

    pub fn load_report(&self) -> &AttentionLoadReport {
        &self.load_report
    }

    pub fn has_dsa_indexer(&self) -> bool {
        self.indexer.is_some()
    }

    #[cfg(test)]
    pub fn forward<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<AttentionOutput> {
        self.forward_with_past_kv(config, hidden_states, backend, None)
    }

    pub fn forward_with_past_kv<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
        past_kv: Option<(&Tensor, &Tensor)>,
    ) -> Result<AttentionOutput> {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF attention input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            "gguf_attention_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;
        let past_tokens = match past_kv {
            Some((past_k, past_v)) => validate_mla_past_kv(
                "gguf_attention",
                past_k,
                past_v,
                batch,
                self.kv_lora_rank,
                config.qk_rope_dim,
            )?,
            None => 0,
        };

        let input_norm = self.input_norm.forward(hidden_states, backend)?;
        let q_a = self.q_a.forward(&input_norm.hidden_states, backend)?;
        let q_a_norm = self.q_a_norm.forward(&q_a.output, backend)?;
        let q_b = self.q_b.forward(&q_a_norm.hidden_states, backend)?;
        let q_b_shape = Shape::new(q_b.output.dims().to_vec());
        let q_heads =
            q_b.output
                .reshape((batch, tokens, config.attention_heads, config.qk_head_dim))?;
        let (q_no_rope, q_rope) =
            backend.split_rope_tail(&q_heads, config.qk_no_rope_dim, config.qk_rope_dim)?;
        let q_rope_after_rope = backend.rope_slice(
            &q_rope,
            config.qk_rope_dim,
            past_tokens,
            config.rope_theta as f32,
        )?;
        let q_recombined = backend.combine_rope_tail(&q_no_rope, &q_rope_after_rope)?;

        let kv_a_mqa = self.kv_a_mqa.forward(&input_norm.hidden_states, backend)?;
        let (kv_latent, k_rope_mqa) =
            backend.split_kv_mqa(&kv_a_mqa.output, self.kv_lora_rank, config.qk_rope_dim)?;
        let k_rope_after_rope = backend.rope_slice(
            &k_rope_mqa,
            config.qk_rope_dim,
            past_tokens,
            config.rope_theta as f32,
        )?;
        let current_cache_k = mla_latent_cache(&kv_latent)?;
        let current_cache_v = backend.heads_to_attention_layout(&k_rope_after_rope)?;
        let (attention_latent, attention_rope) = mla_attention_sequences(
            past_kv,
            &kv_latent,
            &k_rope_after_rope,
            batch,
            past_tokens,
            self.kv_lora_rank,
            config.qk_rope_dim,
        )?;
        let kv_a_norm = self.kv_a_norm.forward(&attention_latent, backend)?;
        let k_b = self.k_b.forward_heads(&kv_a_norm.hidden_states, backend)?;
        let v_b = self.v_b.forward_heads(&kv_a_norm.hidden_states, backend)?;
        let k_no_rope = k_b.output_heads;
        let v_heads = v_b.output_heads;
        let k_heads = backend.combine_rope_tail(&k_no_rope, &attention_rope)?;

        let q_for_attention = backend.heads_to_attention_layout(&q_recombined)?;
        let k_for_attention = backend.heads_to_attention_layout(&k_heads)?;
        let v_for_attention = backend.heads_to_attention_layout(&v_heads)?;
        let attention_k = k_for_attention;
        let attention_v = v_for_attention;

        let raw_attention_scores =
            backend.attention_scores(&q_for_attention, &attention_k, config.qk_head_dim)?;
        let attention_probs =
            backend.attention_causal_softmax(&raw_attention_scores, past_tokens)?;
        let context_heads = backend.attention_values(&attention_probs, &attention_v)?;
        let merged_attention_output = backend.merge_attention_heads(&context_heads)?;
        let output_projection = self.output.forward(&merged_attention_output, backend)?;
        let output_hidden_states = backend.add(hidden_states, &output_projection.output)?;
        let report = AttentionForwardReport {
            layer_index: self.layer_index,
            past_tokens,
            input_hidden_states_shape: Shape::new(hidden_states.dims().to_vec()),
            normed_hidden_states_shape: Shape::new(input_norm.hidden_states.dims().to_vec()),
            q_a_shape: Shape::new(q_a.output.dims().to_vec()),
            q_a_norm_shape: Shape::new(q_a_norm.hidden_states.dims().to_vec()),
            q_b_shape,
            q_b_chunk_count: q_b.report.chunk_count,
            q_heads_shape: Shape::new(q_heads.dims().to_vec()),
            q_no_rope_shape: Shape::new(q_no_rope.dims().to_vec()),
            q_rope_shape: Shape::new(q_rope.dims().to_vec()),
            q_rope_after_rope_shape: Shape::new(q_rope_after_rope.dims().to_vec()),
            kv_a_mqa_shape: Shape::new(kv_a_mqa.output.dims().to_vec()),
            kv_a_mqa_chunk_count: kv_a_mqa.report.chunk_count,
            kv_latent_shape: Shape::new(kv_latent.dims().to_vec()),
            k_rope_mqa_shape: Shape::new(k_rope_mqa.dims().to_vec()),
            kv_a_norm_shape: Shape::new(kv_a_norm.hidden_states.dims().to_vec()),
            k_b_shape: Shape::new(k_no_rope.dims().to_vec()),
            k_b_chunk_count: k_b.chunk_count,
            v_b_shape: Shape::new(v_heads.dims().to_vec()),
            v_b_chunk_count: v_b.chunk_count,
            k_no_rope_shape: Shape::new(k_no_rope.dims().to_vec()),
            k_rope_after_rope_shape: Shape::new(k_rope_after_rope.dims().to_vec()),
            k_heads_shape: Shape::new(k_heads.dims().to_vec()),
            v_heads_shape: Shape::new(v_heads.dims().to_vec()),
            cache_k_shape: Shape::new(current_cache_k.dims().to_vec()),
            cache_v_shape: Shape::new(current_cache_v.dims().to_vec()),
            attention_k_shape: Shape::new(attention_k.dims().to_vec()),
            attention_v_shape: Shape::new(attention_v.dims().to_vec()),
            raw_attention_scores_shape: Shape::new(raw_attention_scores.dims().to_vec()),
            attention_scores_shape: Shape::new(raw_attention_scores.dims().to_vec()),
            attention_probs_shape: Shape::new(attention_probs.dims().to_vec()),
            context_heads_shape: Shape::new(context_heads.dims().to_vec()),
            merged_attention_output_shape: Shape::new(merged_attention_output.dims().to_vec()),
            output_projection_shape: Shape::new(output_projection.output.dims().to_vec()),
            output_projection_chunk_count: output_projection.report.chunk_count,
            output_hidden_states_shape: Shape::new(output_hidden_states.dims().to_vec()),
        };

        Ok(AttentionOutput {
            hidden_states: output_hidden_states,
            cache_k: current_cache_k,
            cache_v: current_cache_v,
            report,
        })
    }

    pub fn forward_tensors_with_past_kv<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
        past_kv: Option<(&F32Tensor, &F32Tensor)>,
    ) -> Result<AttentionTensors> {
        if backend.capabilities().custom_kernels {
            return self.forward_tensors_native_with_past_kv(
                config,
                hidden_states,
                backend,
                past_kv,
            );
        }

        let past_kv_tensors = match past_kv {
            Some((past_k, past_v)) => Some((
                tensor_from_f32_tensor(past_k.clone(), backend.device())?,
                tensor_from_f32_tensor(past_v.clone(), backend.device())?,
            )),
            None => None,
        };
        let past_kv = past_kv_tensors
            .as_ref()
            .map(|(cache_k, cache_v)| (cache_k, cache_v));
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF attention input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            "gguf_attention_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;
        let past_tokens = match past_kv {
            Some((past_k, past_v)) => validate_mla_past_kv(
                "gguf_attention",
                past_k,
                past_v,
                batch,
                self.kv_lora_rank,
                config.qk_rope_dim,
            )?,
            None => 0,
        };

        let input_norm = self.input_norm.forward_tensor(hidden_states, backend)?;
        let q_a = self.q_a.forward_tensor(&input_norm, backend)?;
        let q_a_norm = self.q_a_norm.forward_tensor(&q_a, backend)?;
        let q_b = self.q_b.forward_tensor(&q_a_norm, backend)?;
        let q_heads = q_b.reshape((batch, tokens, config.attention_heads, config.qk_head_dim))?;
        let (q_no_rope, q_rope) =
            backend.split_rope_tail(&q_heads, config.qk_no_rope_dim, config.qk_rope_dim)?;
        let q_rope_after_rope = backend.rope_slice(
            &q_rope,
            config.qk_rope_dim,
            past_tokens,
            config.rope_theta as f32,
        )?;
        let q_recombined = backend.combine_rope_tail(&q_no_rope, &q_rope_after_rope)?;

        let kv_a_mqa = self.kv_a_mqa.forward_tensor(&input_norm, backend)?;
        let (kv_latent, k_rope_mqa) =
            backend.split_kv_mqa(&kv_a_mqa, self.kv_lora_rank, config.qk_rope_dim)?;
        let k_rope_after_rope = backend.rope_slice(
            &k_rope_mqa,
            config.qk_rope_dim,
            past_tokens,
            config.rope_theta as f32,
        )?;
        let current_cache_k = mla_latent_cache(&kv_latent)?;
        let current_cache_v = backend.heads_to_attention_layout(&k_rope_after_rope)?;
        let (attention_latent, attention_rope) = mla_attention_sequences(
            past_kv,
            &kv_latent,
            &k_rope_after_rope,
            batch,
            past_tokens,
            self.kv_lora_rank,
            config.qk_rope_dim,
        )?;
        let kv_a_norm = self.kv_a_norm.forward_tensor(&attention_latent, backend)?;
        let k_no_rope = self.k_b.forward_heads(&kv_a_norm, backend)?.output_heads;
        let v_heads = self.v_b.forward_heads(&kv_a_norm, backend)?.output_heads;
        let k_heads = backend.combine_rope_tail(&k_no_rope, &attention_rope)?;

        let q_for_attention = backend.heads_to_attention_layout(&q_recombined)?;
        let k_for_attention = backend.heads_to_attention_layout(&k_heads)?;
        let v_for_attention = backend.heads_to_attention_layout(&v_heads)?;
        let attention_k = k_for_attention;
        let attention_v = v_for_attention;

        let raw_attention_scores =
            backend.attention_scores(&q_for_attention, &attention_k, config.qk_head_dim)?;
        let attention_probs =
            backend.attention_causal_softmax(&raw_attention_scores, past_tokens)?;
        let context_heads = backend.attention_values(&attention_probs, &attention_v)?;
        let merged_attention_output = backend.merge_attention_heads(&context_heads)?;
        let output_projection = self
            .output
            .forward_tensor(&merged_attention_output, backend)?;
        let output_hidden_states = backend.add(hidden_states, &output_projection)?;

        Ok(AttentionTensors {
            hidden_states: output_hidden_states,
            cache_k: tensor_to_f32_tensor(&current_cache_k)?,
            cache_v: tensor_to_f32_tensor(&current_cache_v)?,
            index_key: None,
        })
    }

    fn forward_tensors_native_with_past_kv<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
        past_kv: Option<(&F32Tensor, &F32Tensor)>,
    ) -> Result<AttentionTensors> {
        let hidden_states = tensor_to_f32_tensor(hidden_states)?;
        let output =
            self.forward_f32_tensors_with_past_kv(config, &hidden_states, backend, past_kv)?;

        Ok(AttentionTensors {
            hidden_states: tensor_from_f32_tensor(output.hidden_states, backend.device())?,
            cache_k: output.cache_k,
            cache_v: output.cache_v,
            index_key: output.index_key,
        })
    }

    pub fn forward_f32_tensors_with_past_kv<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        backend: &B,
        past_kv: Option<(&F32Tensor, &F32Tensor)>,
    ) -> Result<AttentionF32Tensors> {
        self.forward_f32_tensors_with_attention_past_kv(
            config,
            hidden_states,
            backend,
            past_kv.map(|(cache_k, cache_v)| AttentionPastKv::Contiguous { cache_k, cache_v }),
        )
    }

    pub fn forward_sparse_f32_tensors_with_past_kv<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        backend: &B,
        past_kv: Option<(&F32Tensor, &F32Tensor)>,
        cached_index_keys: Option<&F32Tensor>,
        shared_selection: Option<&[u32]>,
    ) -> Result<SparseAttentionF32Tensors> {
        self.forward_f32_tensors_with_sparse_context(
            config,
            hidden_states,
            backend,
            past_kv.map(|(cache_k, cache_v)| AttentionPastKv::Contiguous { cache_k, cache_v }),
            Some(SparseAttentionContext {
                cached_index_keys,
                shared_selection,
            }),
        )
    }

    pub fn forward_f32_tensors_with_paged_past_kv<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        backend: &B,
        past_kv: Option<PagedKvView<'_>>,
    ) -> Result<AttentionF32Tensors> {
        if past_kv.is_some() {
            return Err(Error::backend(
                "paged native attention still expects expanded K/V; MLA latent paged attention requires an absorbed Metal kernel",
            ));
        }
        self.forward_f32_tensors_with_attention_past_kv(config, hidden_states, backend, None)
    }

    fn forward_f32_tensors_with_attention_past_kv<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        backend: &B,
        past_kv: Option<AttentionPastKv<'_>>,
    ) -> Result<AttentionF32Tensors> {
        Ok(self
            .forward_f32_tensors_with_sparse_context(config, hidden_states, backend, past_kv, None)?
            .tensors)
    }

    fn forward_f32_tensors_with_sparse_context<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        backend: &B,
        past_kv: Option<AttentionPastKv<'_>>,
        sparse_context: Option<SparseAttentionContext<'_>>,
    ) -> Result<SparseAttentionF32Tensors> {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF native attention input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            "gguf_native_attention_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;
        let past_tokens = match past_kv.as_ref() {
            Some(AttentionPastKv::Contiguous { cache_k, cache_v }) => validate_mla_past_kv(
                "gguf_native_attention",
                cache_k,
                cache_v,
                batch,
                self.kv_lora_rank,
                config.qk_rope_dim,
            )?,
            None => 0,
        };

        let input_norm = self
            .input_norm
            .forward_f32(&hidden_states, backend)?
            .hidden_states;
        let q_a = self.q_a.forward_f32_tensor(&input_norm, backend)?;
        let q_a_norm = self.q_a_norm.forward_f32(&q_a, backend)?.hidden_states;
        let q_b = self.q_b.forward_f32_tensor(&q_a_norm, backend)?;
        let q_heads = q_b.reshape([batch, tokens, config.attention_heads, config.qk_head_dim])?;
        let (q_no_rope, q_rope) = require_native(
            "split_rope_tail",
            backend.split_rope_tail_f32_tensor(
                &q_heads,
                config.qk_no_rope_dim,
                config.qk_rope_dim,
            )?,
        )?;
        let q_rope_after_rope = require_native(
            "rope_slice",
            backend.rope_slice_f32_tensor(
                &q_rope,
                config.qk_rope_dim,
                past_tokens,
                config.rope_theta as f32,
            )?,
        )?;
        let q_recombined = require_native(
            "combine_rope_tail",
            backend.combine_rope_tail_f32_tensor(&q_no_rope, &q_rope_after_rope)?,
        )?;

        let kv_a_mqa = self.kv_a_mqa.forward_f32_tensor(&input_norm, backend)?;
        let (kv_latent, k_rope_mqa) = require_native(
            "split_kv_mqa",
            backend.split_kv_mqa_f32_tensor(&kv_a_mqa, self.kv_lora_rank, config.qk_rope_dim)?,
        )?;
        let k_rope_after_rope = require_native(
            "rope_slice",
            backend.rope_slice_f32_tensor(
                &k_rope_mqa,
                config.qk_rope_dim,
                past_tokens,
                config.rope_theta as f32,
            )?,
        )?;
        let current_cache_k = mla_latent_cache(&kv_latent)?;
        let current_cache_v = require_native(
            "heads_to_attention_layout",
            backend.heads_to_attention_layout_f32_tensor(&k_rope_after_rope)?,
        )?;
        let contiguous_past = match past_kv.as_ref() {
            Some(AttentionPastKv::Contiguous { cache_k, cache_v }) => Some((*cache_k, *cache_v)),
            None => None,
        };
        let sparse_selection = self.select_sparse_past_tokens(
            config,
            hidden_states,
            &q_a_norm,
            contiguous_past,
            sparse_context,
            past_tokens,
            backend,
        )?;
        let mut index_key = sparse_selection.current_index_key.clone();
        if index_key.is_none() {
            index_key = match self.indexer.as_ref() {
                Some(indexer) => {
                    Some(indexer.key_f32(config, hidden_states, past_tokens, backend)?)
                }
                None => None,
            };
        }
        let (attention_latent, attention_rope, attention_past_tokens) =
            match sparse_selection.selected_past_tokens.as_ref() {
                Some(selected_past_tokens) => mla_selected_attention_sequences(
                    contiguous_past.ok_or_else(|| {
                        Error::model("DSA sparse attention requires past MLA cache")
                    })?,
                    selected_past_tokens,
                    &kv_latent,
                    &k_rope_after_rope,
                    batch,
                    past_tokens,
                    self.kv_lora_rank,
                    config.qk_rope_dim,
                )?,
                None => {
                    let (attention_latent, attention_rope) = mla_attention_sequences(
                        contiguous_past,
                        &kv_latent,
                        &k_rope_after_rope,
                        batch,
                        past_tokens,
                        self.kv_lora_rank,
                        config.qk_rope_dim,
                    )?;
                    (attention_latent, attention_rope, past_tokens)
                }
            };
        let kv_a_norm = self
            .kv_a_norm
            .forward_f32(&attention_latent, backend)?
            .hidden_states;
        let k_no_rope = self
            .k_b
            .forward_heads_f32(&kv_a_norm, backend)?
            .output_heads;
        let v_heads = self
            .v_b
            .forward_heads_f32(&kv_a_norm, backend)?
            .output_heads;
        let k_heads = require_native(
            "combine_rope_tail",
            backend.combine_rope_tail_f32_tensor(&k_no_rope, &attention_rope)?,
        )?;

        let q_for_attention = require_native(
            "heads_to_attention_layout",
            backend.heads_to_attention_layout_f32_tensor(&q_recombined)?,
        )?;
        let attention_k = require_native(
            "heads_to_attention_layout",
            backend.heads_to_attention_layout_f32_tensor(&k_heads)?,
        )?;
        let attention_v = require_native(
            "heads_to_attention_layout",
            backend.heads_to_attention_layout_f32_tensor(&v_heads)?,
        )?;
        let context_heads = if tokens == 1 {
            require_native(
                "decode_attention",
                backend.decode_attention_f32_tensor(
                    &q_for_attention,
                    &attention_k,
                    &attention_v,
                    config.qk_head_dim,
                    attention_past_tokens,
                )?,
            )?
        } else {
            let raw_attention_scores = require_native(
                "attention_scores",
                backend.attention_scores_f32_tensor(
                    &q_for_attention,
                    &attention_k,
                    config.qk_head_dim,
                )?,
            )?;
            let attention_probs = require_native(
                "attention_causal_softmax",
                backend.attention_causal_softmax_f32_tensor(
                    &raw_attention_scores,
                    attention_past_tokens,
                )?,
            )?;
            require_native(
                "attention_values",
                backend.attention_values_f32_tensor(&attention_probs, &attention_v)?,
            )?
        };
        let merged_attention_output = require_native(
            "merge_attention_heads",
            backend.merge_attention_heads_f32_tensor(&context_heads)?,
        )?;
        let output_hidden_states = self.output.forward_f32_tensor_add_residual(
            &merged_attention_output,
            hidden_states,
            backend,
        )?;

        Ok(SparseAttentionF32Tensors {
            tensors: AttentionF32Tensors {
                hidden_states: output_hidden_states,
                cache_k: current_cache_k,
                cache_v: current_cache_v,
                index_key,
            },
            next_shared_selection: sparse_selection.next_shared_selection,
        })
    }

    fn select_sparse_past_tokens<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        q_resid: &F32Tensor,
        past_kv: Option<(&F32Tensor, &F32Tensor)>,
        sparse_context: Option<SparseAttentionContext<'_>>,
        past_tokens: usize,
        backend: &B,
    ) -> Result<SparsePastSelection> {
        let Some(context) = sparse_context else {
            return Ok(SparsePastSelection {
                selected_past_tokens: None,
                next_shared_selection: None,
                current_index_key: None,
            });
        };
        if hidden_states.dims()[1] != 1 || past_tokens == 0 || past_kv.is_none() {
            return Ok(SparsePastSelection {
                selected_past_tokens: None,
                next_shared_selection: None,
                current_index_key: None,
            });
        }

        if let Some(indexer) = self.indexer.as_ref() {
            let cached_index_keys = context.cached_index_keys.ok_or_else(|| {
                Error::model(format!(
                    "DSA index keys missing for sparse attention layer {}",
                    self.layer_index
                ))
            })?;
            let selection = indexer.select_decode_topk(
                config,
                hidden_states,
                q_resid,
                cached_index_keys,
                past_tokens,
                backend,
            )?;
            let selected_past_tokens =
                past_only_selected_indices(&selection.token_indices, past_tokens)?;
            return Ok(SparsePastSelection {
                selected_past_tokens: Some(selected_past_tokens),
                next_shared_selection: Some(selection.token_indices),
                current_index_key: Some(selection.current_key),
            });
        }

        if let Some(shared_selection) = context.shared_selection {
            return Ok(SparsePastSelection {
                selected_past_tokens: Some(past_only_selected_indices(
                    shared_selection,
                    past_tokens,
                )?),
                next_shared_selection: None,
                current_index_key: None,
            });
        }

        Ok(SparsePastSelection {
            selected_past_tokens: None,
            next_shared_selection: None,
            current_index_key: None,
        })
    }

    /// Batched device-resident decode attention: every kernel is encoded into
    /// the backend's open batch and intermediate tensors never leave the GPU.
    /// Past K/V is read from the resident Metal cache, and the current
    /// token's K/V is returned as device handles for append.
    ///
    /// Returns `Ok(None)` when any component has no device path (the caller
    /// falls back to the eager route); requires a single decode token.
    fn mla_context_device<B: Backend>(
        &self,
        config: &Config,
        q_no_rope: &DeviceValue,
        q_rope: &DeviceValue,
        current_latent_norm: &DeviceValue,
        current_rope: &DeviceValue,
        past_kv: &DevicePagedKvView,
        backend: &B,
    ) -> Result<Option<DeviceValue>> {
        backend.q8_0_absorbed_mla_decode_device(
            self.k_b.q8_payload("absorbed MLA K_b")?,
            self.v_b.q8_payload("absorbed MLA V_b")?,
            q_no_rope,
            q_rope,
            current_latent_norm,
            current_rope,
            past_kv,
            config.qk_head_dim,
            config.v_head_dim(),
        )
    }

    fn mla_sequence_context_device<B: Backend>(
        &self,
        config: &Config,
        batch: usize,
        q_for_attention: &DeviceValue,
        current_k_for_attention: &DeviceValue,
        current_v_for_attention: &DeviceValue,
        past_kv: &DevicePagedKvView,
        backend: &B,
    ) -> Result<Option<DeviceValue>> {
        let (past_k_for_attention, past_v_for_attention) =
            crate::try_device!(self.mla_past_attention_device(config, batch, past_kv, backend));
        backend.selected_sequence_attention_device(
            q_for_attention,
            &past_k_for_attention,
            &past_v_for_attention,
            current_k_for_attention,
            current_v_for_attention,
        )
    }

    fn merge_context_heads_device<B: Backend>(
        &self,
        config: &Config,
        context_heads: &DeviceValue,
        batch: usize,
        tokens: usize,
        backend: &B,
    ) -> Result<Option<DeviceValue>> {
        validate_exact_shape(
            "attention context heads before output projection",
            context_heads.dims(),
            &[batch, config.attention_heads, tokens, config.v_head_dim()],
        )?;
        let merged_width = config
            .attention_heads
            .checked_mul(config.v_head_dim())
            .ok_or_else(|| Error::model("attention merged head width overflow"))?;
        if tokens == 1 {
            return Ok(Some(context_heads.reshape(vec![
                batch,
                tokens,
                merged_width,
            ])?));
        }
        backend.merge_attention_heads_device(context_heads)
    }

    fn mla_past_attention_device<B: Backend>(
        &self,
        config: &Config,
        batch: usize,
        past_kv: &DevicePagedKvView,
        backend: &B,
    ) -> Result<Option<(DeviceValue, DeviceValue)>> {
        let past = crate::try_device!(backend.paged_kv_contiguous_device(past_kv));
        validate_exact_shape(
            "dense_mla_past_device_kv_shape",
            &[
                past.batch,
                past.attention_heads,
                past.key_head_dim,
                past.value_head_dim,
            ],
            &[batch, 1, self.kv_lora_rank, config.qk_rope_dim],
        )?;
        let past_latent = past
            .k
            .reshape(vec![batch, past.selected_tokens, self.kv_lora_rank])?;
        let past_rope = past
            .v
            .reshape(vec![batch, past.selected_tokens, 1, config.qk_rope_dim])?;
        let past_k_no_rope =
            crate::try_device!(self.k_b.forward_heads_device(&past_latent, backend));
        let past_v_heads = crate::try_device!(self.v_b.forward_heads_device(&past_latent, backend));
        let past_k_heads =
            crate::try_device!(backend.combine_rope_tail_device(&past_k_no_rope, &past_rope));
        let past_k_for_attention =
            crate::try_device!(backend.heads_to_attention_layout_device(&past_k_heads));
        let past_v_for_attention =
            crate::try_device!(backend.heads_to_attention_layout_device(&past_v_heads));
        Ok(Some((past_k_for_attention, past_v_for_attention)))
    }

    pub(crate) fn forward_decode_device<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &DeviceValue,
        backend: &B,
        past_kv: &DevicePagedKvView,
    ) -> Result<Option<AttentionDeviceTensors>> {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF device attention input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        if tokens == 0 || tokens > 3 {
            return Ok(None);
        }
        validate_exact_shape(
            "gguf_device_attention_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;
        past_kv.validate()?;
        let past_layout =
            validate_device_past_kv_layout(past_kv, batch, config, self.kv_lora_rank)?;
        let past_tokens = past_kv.cached_tokens;
        let input_norm = crate::try_device!(profile::run_layer_stage(
            self.layer_index,
            "attention.input_norm",
            || self.input_norm.forward_device(hidden_states, backend),
        ));
        let (q_no_rope, q_rope_after_rope, q_recombined) = crate::try_device!(
            profile::run_layer_stage(self.layer_index, "attention.q_projection", || {
                let q_a = crate::try_device!(self.q_a.forward_device(&input_norm, backend));
                let q_a_norm = crate::try_device!(self.q_a_norm.forward_device(&q_a, backend));
                let q_b = crate::try_device!(self.q_b.forward_device(&q_a_norm, backend));
                let q_heads = q_b.reshape(vec![
                    batch,
                    tokens,
                    config.attention_heads,
                    config.qk_head_dim,
                ])?;
                let (q_no_rope, q_rope) = crate::try_device!(backend.split_rope_tail_device(
                    &q_heads,
                    config.qk_no_rope_dim,
                    config.qk_rope_dim,
                ));
                let q_rope_after_rope = crate::try_device!(backend.rope_slice_device(
                    &q_rope,
                    config.qk_rope_dim,
                    past_tokens,
                    config.rope_theta as f32,
                ));
                let q_recombined = crate::try_device!(
                    backend.combine_rope_tail_device(&q_no_rope, &q_rope_after_rope)
                );
                Ok(Some((q_no_rope, q_rope_after_rope, q_recombined)))
            },)
        );

        let (kv_latent_norm, k_rope_after_rope, k_heads, v_heads) = crate::try_device!(
            profile::run_layer_stage(self.layer_index, "attention.kv_projection", || {
                let kv_a_mqa =
                    crate::try_device!(self.kv_a_mqa.forward_device(&input_norm, backend));
                let (kv_latent, k_rope_mqa) = crate::try_device!(backend.split_kv_mqa_device(
                    &kv_a_mqa,
                    self.kv_lora_rank,
                    config.qk_rope_dim,
                ));
                let kv_a_norm =
                    crate::try_device!(self.kv_a_norm.forward_device(&kv_latent, backend));
                let k_no_rope =
                    crate::try_device!(self.k_b.forward_heads_device(&kv_a_norm, backend));
                let v_heads =
                    crate::try_device!(self.v_b.forward_heads_device(&kv_a_norm, backend));
                let k_rope_after_rope = crate::try_device!(backend.rope_slice_device(
                    &k_rope_mqa,
                    config.qk_rope_dim,
                    past_tokens,
                    config.rope_theta as f32,
                ));
                let k_heads = crate::try_device!(
                    backend.combine_rope_tail_device(&k_no_rope, &k_rope_after_rope)
                );
                Ok(Some((kv_a_norm, k_rope_after_rope, k_heads, v_heads)))
            },)
        );

        let (q_for_attention, current_k_for_attention, current_v_for_attention) = crate::try_device!(
            profile::run_layer_stage(self.layer_index, "attention.cache_layout", || {
                let q_for_attention =
                    crate::try_device!(backend.heads_to_attention_layout_device(&q_recombined));
                let current_k_for_attention =
                    crate::try_device!(backend.heads_to_attention_layout_device(&k_heads));
                let current_v_for_attention =
                    crate::try_device!(backend.heads_to_attention_layout_device(&v_heads));
                Ok(Some((
                    q_for_attention,
                    current_k_for_attention,
                    current_v_for_attention,
                )))
            },)
        );

        let (k_for_cache, v_for_cache) = match past_layout {
            DevicePastKvLayout::ExpandedHeads => (
                current_k_for_attention.clone(),
                current_v_for_attention.clone(),
            ),
            DevicePastKvLayout::MlaLatent => {
                let cache_k = kv_latent_norm.reshape(vec![batch, 1, tokens, self.kv_lora_rank])?;
                let cache_v = crate::try_device!(
                    backend.heads_to_attention_layout_device(&k_rope_after_rope)
                );
                (cache_k, cache_v)
            }
        };

        let context_heads = match (past_layout, tokens) {
            (DevicePastKvLayout::ExpandedHeads, 1) => {
                crate::try_device!(profile::run_layer_stage(
                    self.layer_index,
                    "attention.dense_paged_decode",
                    || backend.paged_decode_attention_resident_device(
                        &q_for_attention,
                        &current_k_for_attention,
                        &current_v_for_attention,
                        past_kv,
                    ),
                ))
            }
            (DevicePastKvLayout::MlaLatent, 1) => crate::try_device!(profile::run_layer_stage(
                self.layer_index,
                "attention.dense_mla_decode",
                || {
                    self.mla_context_device(
                        config,
                        &q_no_rope,
                        &q_rope_after_rope,
                        &kv_latent_norm,
                        &k_rope_after_rope,
                        past_kv,
                        backend,
                    )
                },
            )),
            (DevicePastKvLayout::ExpandedHeads, _) => {
                let past = crate::try_device!(backend.paged_kv_contiguous_device(past_kv));
                crate::try_device!(profile::run_layer_stage(
                    self.layer_index,
                    "attention.dense_sequence",
                    || backend.selected_sequence_attention_device(
                        &q_for_attention,
                        &past.k,
                        &past.v,
                        &current_k_for_attention,
                        &current_v_for_attention,
                    ),
                ))
            }
            (DevicePastKvLayout::MlaLatent, _) => crate::try_device!(profile::run_layer_stage(
                self.layer_index,
                "attention.dense_mla_sequence",
                || {
                    self.mla_sequence_context_device(
                        config,
                        batch,
                        &q_for_attention,
                        &current_k_for_attention,
                        &current_v_for_attention,
                        past_kv,
                        backend,
                    )
                },
            )),
        };
        let output_hidden_states = crate::try_device!(profile::run_layer_stage(
            self.layer_index,
            "attention.output_projection",
            || {
                let merged_attention_output = crate::try_device!(self.merge_context_heads_device(
                    config,
                    &context_heads,
                    batch,
                    tokens,
                    backend,
                ));
                self.output.forward_device_add_residual(
                    &merged_attention_output,
                    hidden_states,
                    backend,
                )
            },
        ));

        Ok(Some(AttentionDeviceTensors {
            hidden_states: output_hidden_states,
            cache_k: k_for_cache,
            cache_v: v_for_cache,
            index_key: None,
        }))
    }

    pub(crate) fn forward_seed_device<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &DeviceValue,
        backend: &B,
    ) -> Result<Option<AttentionDeviceTensors>> {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF seed device attention input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        if tokens == 0 || tokens > 3 {
            return Ok(None);
        }
        validate_exact_shape(
            "gguf_seed_device_attention_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;
        let sparse_layer = self.layer_index >= config.dense_layers;

        let input_norm = crate::try_device!(profile::run_sparse_token_device_stage(
            sparse_layer,
            profile::TokenProfileStage::SparseInputNorm,
            backend,
            || profile::run_layer_stage(
                self.layer_index,
                "attention.input_norm.seed_device",
                || self.input_norm.forward_device(hidden_states, backend),
            ),
        ));
        let (kv_latent_norm, k_rope_after_rope, v_heads) =
            crate::try_device!(profile::run_sparse_token_device_stage(
                sparse_layer,
                profile::TokenProfileStage::SparseKvProjection,
                backend,
                || profile::run_layer_stage(
                    self.layer_index,
                    "attention.kv_projection.seed_device",
                    || {
                        let kv_a_mqa =
                            crate::try_device!(self.kv_a_mqa.forward_device(&input_norm, backend));
                        let (kv_latent, k_rope_mqa) =
                            crate::try_device!(backend.split_kv_mqa_device(
                                &kv_a_mqa,
                                self.kv_lora_rank,
                                config.qk_rope_dim,
                            ));
                        let kv_a_norm =
                            crate::try_device!(self.kv_a_norm.forward_device(&kv_latent, backend));
                        let v_heads =
                            crate::try_device!(self.v_b.forward_heads_device(&kv_a_norm, backend));
                        let k_rope_after_rope = crate::try_device!(backend.rope_slice_device(
                            &k_rope_mqa,
                            config.qk_rope_dim,
                            0,
                            config.rope_theta as f32,
                        ));
                        Ok(Some((kv_a_norm, k_rope_after_rope, v_heads)))
                    },
                ),
            ));

        let (context_heads, cache_v) = crate::try_device!(profile::run_sparse_token_device_stage(
            sparse_layer,
            profile::TokenProfileStage::SparseCacheLayout,
            backend,
            || {
                let context_heads = crate::try_device!(profile::run_layer_stage(
                    self.layer_index,
                    "attention.single_token_context.seed_device",
                    || backend.heads_to_attention_layout_device(&v_heads),
                ));
                let cache_v = crate::try_device!(
                    backend.heads_to_attention_layout_device(&k_rope_after_rope)
                );
                Ok(Some((context_heads, cache_v)))
            },
        ));
        let cache_k = kv_latent_norm.reshape(vec![batch, 1, tokens, self.kv_lora_rank])?;

        let output_hidden_states = crate::try_device!(profile::run_sparse_token_device_stage(
            sparse_layer,
            profile::TokenProfileStage::SparseOutputProjection,
            backend,
            || profile::run_layer_stage(
                self.layer_index,
                "attention.output_projection.seed_device",
                || {
                    let merged_attention_output = crate::try_device!(self
                        .merge_context_heads_device(
                            config,
                            &context_heads,
                            batch,
                            tokens,
                            backend,
                        ));
                    self.output.forward_device_add_residual(
                        &merged_attention_output,
                        hidden_states,
                        backend,
                    )
                },
            ),
        ));

        let index_key = if let Some(indexer) = self.indexer.as_ref() {
            Some(crate::try_device!(profile::run_sparse_token_device_stage(
                sparse_layer,
                profile::TokenProfileStage::SparseDsaIndexer,
                backend,
                || indexer.key_device(config, hidden_states, 0, backend),
            )))
        } else {
            None
        };

        Ok(Some(AttentionDeviceTensors {
            hidden_states: output_hidden_states,
            cache_k,
            cache_v,
            index_key,
        }))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_sparse_decode_device<B, S, I>(
        &self,
        config: &Config,
        hidden_states: &DeviceValue,
        backend: &B,
        past_kv: &DevicePagedKvView,
        selected_kv_for_tokens: &mut S,
        index_keys_for_layer: &mut I,
        shared_selection: Option<&[u32]>,
    ) -> Result<Option<SparseAttentionDeviceTensors>>
    where
        B: Backend,
        S: FnMut(usize, &[u32]) -> Result<Option<DeviceSelectedKvView>>,
        I: FnMut(usize) -> Result<Option<DeviceValue>>,
    {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF sparse device attention input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        if tokens == 0 || tokens > 3 {
            return Ok(None);
        }
        validate_exact_shape(
            "gguf_sparse_device_attention_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;
        past_kv.validate()?;
        let past_layout =
            validate_device_past_kv_layout(past_kv, batch, config, self.kv_lora_rank)?;
        let past_tokens = past_kv.cached_tokens;
        let full_context = past_tokens
            .checked_add(tokens)
            .ok_or_else(|| Error::model("DSA decode key token count overflow"))?
            <= config.dsa_index_topk;
        let use_absorbed_mla =
            past_layout == DevicePastKvLayout::MlaLatent && tokens == 1 && full_context;

        let input_norm = crate::try_device!(profile::run_token_device_stage(
            profile::TokenProfileStage::SparseInputNorm,
            backend,
            || profile::run_layer_stage(self.layer_index, "attention.input_norm", || {
                self.input_norm.forward_device(hidden_states, backend)
            }),
        ));
        let (q_no_rope, q_rope_after_rope, q_recombined, q_resid) =
            crate::try_device!(profile::run_token_device_stage(
                profile::TokenProfileStage::SparseQProjection,
                backend,
                || profile::run_layer_stage(self.layer_index, "attention.q_projection", || {
                    let q_a = crate::try_device!(self.q_a.forward_device(&input_norm, backend));
                    let q_a_norm = crate::try_device!(self.q_a_norm.forward_device(&q_a, backend));
                    let q_b = crate::try_device!(self.q_b.forward_device(&q_a_norm, backend));
                    let q_heads = q_b.reshape(vec![
                        batch,
                        tokens,
                        config.attention_heads,
                        config.qk_head_dim,
                    ])?;
                    let (q_no_rope, q_rope) = crate::try_device!(backend.split_rope_tail_device(
                        &q_heads,
                        config.qk_no_rope_dim,
                        config.qk_rope_dim,
                    ));
                    let q_rope_after_rope = crate::try_device!(backend.rope_slice_device(
                        &q_rope,
                        config.qk_rope_dim,
                        past_tokens,
                        config.rope_theta as f32,
                    ));
                    let q_recombined = crate::try_device!(
                        backend.combine_rope_tail_device(&q_no_rope, &q_rope_after_rope)
                    );
                    Ok(Some((q_no_rope, q_rope_after_rope, q_recombined, q_a_norm)))
                }),
            ));

        let (kv_latent_norm, k_rope_after_rope, k_heads, v_heads) =
            crate::try_device!(profile::run_token_device_stage(
                profile::TokenProfileStage::SparseKvProjection,
                backend,
                || profile::run_layer_stage(self.layer_index, "attention.kv_projection", || {
                    let kv_a_mqa =
                        crate::try_device!(self.kv_a_mqa.forward_device(&input_norm, backend));
                    let (kv_latent, k_rope_mqa) = crate::try_device!(backend.split_kv_mqa_device(
                        &kv_a_mqa,
                        self.kv_lora_rank,
                        config.qk_rope_dim,
                    ));
                    let kv_a_norm =
                        crate::try_device!(self.kv_a_norm.forward_device(&kv_latent, backend));
                    let k_rope_after_rope = crate::try_device!(backend.rope_slice_device(
                        &k_rope_mqa,
                        config.qk_rope_dim,
                        past_tokens,
                        config.rope_theta as f32,
                    ));
                    let (k_heads, v_heads) = if use_absorbed_mla {
                        (None, None)
                    } else {
                        let k_no_rope =
                            crate::try_device!(self.k_b.forward_heads_device(&kv_a_norm, backend));
                        let v_heads =
                            crate::try_device!(self.v_b.forward_heads_device(&kv_a_norm, backend));
                        let k_heads = crate::try_device!(
                            backend.combine_rope_tail_device(&k_no_rope, &k_rope_after_rope)
                        );
                        (Some(k_heads), Some(v_heads))
                    };
                    Ok(Some((kv_a_norm, k_rope_after_rope, k_heads, v_heads)))
                }),
            ));

        let (q_for_attention, current_k_for_attention, current_v_for_attention) =
            crate::try_device!(profile::run_token_device_stage(
                profile::TokenProfileStage::SparseCacheLayout,
                backend,
                || profile::run_layer_stage(self.layer_index, "attention.cache_layout", || {
                    if use_absorbed_mla {
                        return Ok(Some((None, None, None)));
                    }
                    let q_for_attention = Some(crate::try_device!(
                        backend.heads_to_attention_layout_device(&q_recombined)
                    ));
                    let current_k_for_attention = Some(crate::try_device!(backend
                        .heads_to_attention_layout_device(k_heads.as_ref().ok_or_else(|| {
                            Error::model("expanded attention is missing current K heads")
                        },)?)));
                    let current_v_for_attention = Some(crate::try_device!(backend
                        .heads_to_attention_layout_device(v_heads.as_ref().ok_or_else(|| {
                            Error::model("expanded attention is missing current V heads")
                        },)?)));
                    Ok(Some((
                        q_for_attention,
                        current_k_for_attention,
                        current_v_for_attention,
                    )))
                }),
            ));

        let (k_for_cache, v_for_cache) = match past_layout {
            DevicePastKvLayout::ExpandedHeads => (
                current_k_for_attention
                    .clone()
                    .ok_or_else(|| Error::model("expanded attention cache is missing current K"))?,
                current_v_for_attention
                    .clone()
                    .ok_or_else(|| Error::model("expanded attention cache is missing current V"))?,
            ),
            DevicePastKvLayout::MlaLatent => {
                let cache_k = kv_latent_norm.reshape(vec![batch, 1, tokens, self.kv_lora_rank])?;
                let cache_v = crate::try_device!(
                    backend.heads_to_attention_layout_device(&k_rope_after_rope)
                );
                (cache_k, cache_v)
            }
        };

        let mut index_key = None;
        let mut next_shared_selection = None;
        if tokens > 1 && !full_context {
            return Ok(None);
        }
        let selected_past_tokens = crate::try_device!(profile::run_token_device_stage(
            profile::TokenProfileStage::SparseDsaIndexer,
            backend,
            || {
                let selected_past_tokens = if let Some(indexer) = self.indexer.as_ref() {
                    if full_context {
                        index_key = Some(crate::try_device!(profile::run_layer_stage(
                            self.layer_index,
                            "dsa_indexer.current_key_device",
                            || indexer.key_device(config, hidden_states, past_tokens, backend),
                        )));
                        Vec::new()
                    } else {
                        let cached_index_keys = index_keys_for_layer(self.layer_index)?
                            .ok_or_else(|| {
                                Error::model(format!(
                                    "DSA index keys missing for full sparse attention layer {}",
                                    self.layer_index
                                ))
                            })?;
                        let selection = crate::try_device!(profile::run_layer_stage(
                            self.layer_index,
                            "dsa_indexer.topk_device",
                            || indexer.select_decode_topk_device(
                                config,
                                hidden_states,
                                &q_resid,
                                &cached_index_keys,
                                past_tokens,
                                backend,
                            ),
                        ));
                        index_key = Some(selection.current_key_device);
                        next_shared_selection = Some(selection.token_indices.clone());
                        past_only_selected_indices(&selection.token_indices, past_tokens)?
                    }
                } else if full_context {
                    Vec::new()
                } else if let Some(shared_selection) = shared_selection {
                    past_only_selected_indices(shared_selection, past_tokens)?
                } else {
                    return Err(Error::model(format!(
                        "DSA shared selection missing for sparse attention layer {}",
                        self.layer_index
                    )));
                };
                Ok(Some(selected_past_tokens))
            },
        ));

        let context_heads = crate::try_device!(profile::run_token_device_stage(
            profile::TokenProfileStage::SparseContextAttention,
            backend,
            || {
                let context_heads = if past_tokens == 0 && tokens == 1 {
                    let (_, _, current_v) = require_expanded_attention_components(
                        &q_for_attention,
                        &current_k_for_attention,
                        &current_v_for_attention,
                    )?;
                    current_v.clone()
                } else if full_context {
                    match (past_layout, tokens) {
                        (DevicePastKvLayout::ExpandedHeads, 1) => {
                            let (q, current_k, current_v) = require_expanded_attention_components(
                                &q_for_attention,
                                &current_k_for_attention,
                                &current_v_for_attention,
                            )?;
                            crate::try_device!(profile::run_layer_stage(
                                self.layer_index,
                                "attention.dense_paged_decode",
                                || backend.paged_decode_attention_resident_device(
                                    q, current_k, current_v, past_kv,
                                ),
                            ))
                        }
                        (DevicePastKvLayout::MlaLatent, 1) => {
                            crate::try_device!(profile::run_layer_stage(
                                self.layer_index,
                                "attention.full_context_mla_decode",
                                || self.mla_context_device(
                                    config,
                                    &q_no_rope,
                                    &q_rope_after_rope,
                                    &kv_latent_norm,
                                    &k_rope_after_rope,
                                    past_kv,
                                    backend,
                                ),
                            ))
                        }
                        (DevicePastKvLayout::ExpandedHeads, _) => {
                            let (q, current_k, current_v) = require_expanded_attention_components(
                                &q_for_attention,
                                &current_k_for_attention,
                                &current_v_for_attention,
                            )?;
                            let past =
                                crate::try_device!(backend.paged_kv_contiguous_device(past_kv));
                            crate::try_device!(profile::run_layer_stage(
                                self.layer_index,
                                "attention.full_context_sequence",
                                || backend.selected_sequence_attention_device(
                                    q, &past.k, &past.v, current_k, current_v,
                                ),
                            ))
                        }
                        (DevicePastKvLayout::MlaLatent, _) => {
                            let (q, current_k, current_v) = require_expanded_attention_components(
                                &q_for_attention,
                                &current_k_for_attention,
                                &current_v_for_attention,
                            )?;
                            crate::try_device!(profile::run_layer_stage(
                                self.layer_index,
                                "attention.full_context_mla_sequence",
                                || self.mla_sequence_context_device(
                                    config, batch, q, current_k, current_v, past_kv, backend,
                                ),
                            ))
                        }
                    }
                } else {
                    let selected_kv =
                        selected_kv_for_tokens(self.layer_index, &selected_past_tokens)?
                            .ok_or_else(|| {
                                Error::cache(format!(
                                    "selected device KV missing for sparse attention layer {}",
                                    self.layer_index
                                ))
                            })?;
                    selected_kv.validate()?;
                    match past_layout {
                        DevicePastKvLayout::ExpandedHeads => {
                            let (q, current_k, current_v) = require_expanded_attention_components(
                                &q_for_attention,
                                &current_k_for_attention,
                                &current_v_for_attention,
                            )?;
                            validate_exact_shape(
                                "sparse_selected_device_kv_shape",
                                &[
                                    selected_kv.batch,
                                    selected_kv.attention_heads,
                                    selected_kv.key_head_dim,
                                    selected_kv.value_head_dim,
                                ],
                                &[
                                    batch,
                                    config.attention_heads,
                                    config.qk_head_dim,
                                    config.v_head_dim(),
                                ],
                            )?;
                            crate::try_device!(profile::run_layer_stage(
                                self.layer_index,
                                "attention.selected_decode",
                                || backend.selected_decode_attention_device(
                                    q,
                                    &selected_kv.k,
                                    &selected_kv.v,
                                    current_k,
                                    current_v,
                                ),
                            ))
                        }
                        DevicePastKvLayout::MlaLatent => {
                            let (q, current_k, current_v) = require_expanded_attention_components(
                                &q_for_attention,
                                &current_k_for_attention,
                                &current_v_for_attention,
                            )?;
                            validate_exact_shape(
                                "sparse_selected_mla_device_kv_shape",
                                &[
                                    selected_kv.batch,
                                    selected_kv.attention_heads,
                                    selected_kv.key_head_dim,
                                    selected_kv.value_head_dim,
                                ],
                                &[batch, 1, self.kv_lora_rank, config.qk_rope_dim],
                            )?;
                            let selected_tokens = selected_kv.selected_tokens;
                            let context =
                                crate::try_device!(profile::run_layer_stage(
                                    self.layer_index,
                                    "attention.selected_mla_decode",
                                    || {
                                        let selected_latent = selected_kv.k.reshape(vec![
                                            batch,
                                            selected_tokens,
                                            self.kv_lora_rank,
                                        ])?;
                                        let selected_rope = selected_kv.v.reshape(vec![
                                            batch,
                                            selected_tokens,
                                            1,
                                            config.qk_rope_dim,
                                        ])?;
                                        let selected_k_no_rope = crate::try_device!(self
                                            .k_b
                                            .forward_heads_device(&selected_latent, backend));
                                        let selected_v_heads = crate::try_device!(self
                                            .v_b
                                            .forward_heads_device(&selected_latent, backend));
                                        let selected_k_heads = crate::try_device!(backend
                                            .combine_rope_tail_device(
                                                &selected_k_no_rope,
                                                &selected_rope,
                                            ));
                                        let selected_k_for_attention = crate::try_device!(backend
                                            .heads_to_attention_layout_device(&selected_k_heads));
                                        let selected_v_for_attention = crate::try_device!(backend
                                            .heads_to_attention_layout_device(&selected_v_heads));
                                        backend.selected_decode_attention_device(
                                            q,
                                            &selected_k_for_attention,
                                            &selected_v_for_attention,
                                            current_k,
                                            current_v,
                                        )
                                    },
                                ));
                            context
                        }
                    }
                };
                Ok(Some(context_heads))
            },
        ));

        let output_hidden_states = crate::try_device!(profile::run_token_device_stage(
            profile::TokenProfileStage::SparseOutputProjection,
            backend,
            || profile::run_layer_stage(self.layer_index, "attention.output_projection", || {
                let merged_attention_output = crate::try_device!(self.merge_context_heads_device(
                    config,
                    &context_heads,
                    batch,
                    tokens,
                    backend,
                ));
                self.output.forward_device_add_residual(
                    &merged_attention_output,
                    hidden_states,
                    backend,
                )
            },),
        ));

        Ok(Some(SparseAttentionDeviceTensors {
            tensors: AttentionDeviceTensors {
                hidden_states: output_hidden_states,
                cache_k: k_for_cache,
                cache_v: v_for_cache,
                index_key,
            },
            next_shared_selection,
        }))
    }
}

fn require_expanded_attention_components<'a>(
    q: &'a Option<DeviceValue>,
    current_k: &'a Option<DeviceValue>,
    current_v: &'a Option<DeviceValue>,
) -> Result<(&'a DeviceValue, &'a DeviceValue, &'a DeviceValue)> {
    Ok((
        q.as_ref()
            .ok_or_else(|| Error::model("expanded attention is missing Q"))?,
        current_k
            .as_ref()
            .ok_or_else(|| Error::model("expanded attention is missing current K"))?,
        current_v
            .as_ref()
            .ok_or_else(|| Error::model("expanded attention is missing current V"))?,
    ))
}

fn past_only_selected_indices(token_indices: &[u32], past_tokens: usize) -> Result<Vec<u32>> {
    let mut selected = Vec::with_capacity(token_indices.len());
    for token in token_indices {
        let token_usize = usize::try_from(*token)
            .map_err(|_| Error::model("DSA selected token index does not fit usize"))?;
        if token_usize < past_tokens {
            selected.push(*token);
        } else if token_usize > past_tokens {
            return Err(Error::model(format!(
                "DSA selected future token {token_usize}; current decode position is {past_tokens}"
            )));
        }
    }
    Ok(selected)
}

fn validate_q_a(config: &Config, tensor_ref: &TensorRef) -> Result<usize> {
    let (_, q_lora_rank) = validate_linear_ref(
        "gguf_attention_q_a",
        tensor_ref,
        config.hidden_size,
        usize::try_from(tensor_ref.dims[1])
            .map_err(|_| Error::gguf("GGUF q_a rank does not fit usize"))?,
    )?;
    if q_lora_rank == 0 {
        return Err(Error::gguf("GGUF q_lora_rank must be positive"));
    }
    Ok(q_lora_rank)
}

fn validate_kv_a(config: &Config, tensor_ref: &TensorRef) -> Result<usize> {
    let (_, output_features) = linear_dims(tensor_ref)?;
    let kv_lora_rank = output_features
        .checked_sub(config.qk_rope_dim)
        .ok_or_else(|| {
            Error::gguf(format!(
                "GGUF kv_a_mqa output width {output_features} must exceed qk_rope_dim {}",
                config.qk_rope_dim
            ))
        })?;
    if kv_lora_rank == 0 {
        return Err(Error::gguf("GGUF kv_lora_rank must be positive"));
    }
    if kv_lora_rank != config.kv_lora_rank {
        return Err(Error::gguf(format!(
            "GGUF kv_lora_rank {kv_lora_rank} does not match config kv_lora_rank {}",
            config.kv_lora_rank
        )));
    }
    validate_linear_ref(
        "gguf_attention_kv_a_mqa",
        tensor_ref,
        config.hidden_size,
        kv_lora_rank + config.qk_rope_dim,
    )?;
    Ok(kv_lora_rank)
}

fn validate_k_b_ref(config: &Config, kv_lora_rank: usize, tensor_ref: &TensorRef) -> Result<()> {
    validate_quantized_projection_type("gguf_attention_k_b", tensor_ref)?;
    validate_exact_shape(
        "gguf_attention_k_b",
        &tensor_ref.dims_as_usize()?,
        &[config.qk_no_rope_dim, kv_lora_rank, config.attention_heads],
    )
}

fn validate_v_b_ref(config: &Config, kv_lora_rank: usize, tensor_ref: &TensorRef) -> Result<()> {
    validate_quantized_projection_type("gguf_attention_v_b", tensor_ref)?;
    validate_exact_shape(
        "gguf_attention_v_b",
        &tensor_ref.dims_as_usize()?,
        &[kv_lora_rank, config.v_head_dim(), config.attention_heads],
    )
}

fn validate_linear_ref(
    context: &str,
    tensor_ref: &TensorRef,
    expected_input_features: usize,
    expected_output_features: usize,
) -> Result<(usize, usize)> {
    validate_quantized_projection_type(context, tensor_ref)?;
    let (input_features, output_features) = linear_dims(tensor_ref)?;
    validate_exact_shape(
        context,
        &[input_features, output_features],
        &[expected_input_features, expected_output_features],
    )?;
    Ok((input_features, output_features))
}

fn validate_quantized_projection_type(context: &str, tensor_ref: &TensorRef) -> Result<()> {
    match tensor_ref.ty {
        GgmlType::Q2K | GgmlType::Q8_0 => Ok(()),
        other => Err(Error::gguf(format!(
            "{context} tensor {} must be Q2_K or Q8_0 for the GLM-5.2 Q2 runtime, got {other}",
            tensor_ref.name
        ))),
    }
}

fn linear_dims(tensor_ref: &TensorRef) -> Result<(usize, usize)> {
    validate_exact_shape(
        format!("gguf_attention_linear_rank:{}", tensor_ref.name),
        &[tensor_ref.dims.len()],
        &[2],
    )?;
    let input_features = usize::try_from(tensor_ref.dims[0]).map_err(|_| {
        Error::gguf(format!(
            "GGUF tensor {} input dimension does not fit usize",
            tensor_ref.name
        ))
    })?;
    let output_features = usize::try_from(tensor_ref.dims[1]).map_err(|_| {
        Error::gguf(format!(
            "GGUF tensor {} output dimension does not fit usize",
            tensor_ref.name
        ))
    })?;
    Ok((input_features, output_features))
}

fn logical_weight_shape(tensor_ref: &TensorRef) -> Result<Shape> {
    let (input_features, output_features) = linear_dims(tensor_ref)?;
    Ok(Shape::new(vec![output_features, input_features]))
}

fn head_projection_logical_weight_shape(
    heads: usize,
    output_features: usize,
    input_features: usize,
) -> Shape {
    Shape::new(vec![heads, output_features, input_features])
}

trait TensorRefDims {
    fn dims_as_usize(&self) -> Result<Vec<usize>>;
}

impl TensorRefDims for TensorRef {
    fn dims_as_usize(&self) -> Result<Vec<usize>> {
        self.dims
            .iter()
            .map(|dim| {
                usize::try_from(*dim).map_err(|_| {
                    Error::gguf(format!(
                        "GGUF tensor {} dimension {dim} does not fit usize",
                        self.name
                    ))
                })
            })
            .collect()
    }
}

fn validate_tensor_ref(tensor_ref: &TensorRef, info: &gguf::GgufTensorInfo) -> Result<()> {
    if tensor_ref.dims != info.dims {
        return Err(Error::gguf(format!(
            "GGUF tensor {} dims changed from {:?} to {:?}",
            tensor_ref.name, tensor_ref.dims, info.dims
        )));
    }
    if tensor_ref.ty != info.ty {
        return Err(Error::gguf(format!(
            "GGUF tensor {} type changed from {} to {}",
            tensor_ref.name, tensor_ref.ty, info.ty
        )));
    }
    if tensor_ref.absolute_offset != info.absolute_offset {
        return Err(Error::gguf(format!(
            "GGUF tensor {} offset changed from {} to {}",
            tensor_ref.name, tensor_ref.absolute_offset, info.absolute_offset
        )));
    }
    if tensor_ref.storage_byte_len != info.storage_byte_len {
        return Err(Error::gguf(format!(
            "GGUF tensor {} storage length changed from {} to {}",
            tensor_ref.name, tensor_ref.storage_byte_len, info.storage_byte_len
        )));
    }
    Ok(())
}

fn validate_mla_past_kv(
    context: &str,
    past_k: &Tensor,
    past_v: &Tensor,
    batch: usize,
    kv_lora_rank: usize,
    qk_rope_dim: usize,
) -> Result<usize> {
    let k_dims = past_k.dims();
    let v_dims = past_v.dims();
    if k_dims.len() != 4 || v_dims.len() != 4 {
        return Err(Error::model(format!(
            "{context} MLA past cache must be rank 4; latent={k_dims:?} rope={v_dims:?}"
        )));
    }
    let past_tokens = k_dims[2];
    validate_exact_shape(
        format!("{context}_mla_past_latent"),
        k_dims,
        &[batch, 1, past_tokens, kv_lora_rank],
    )?;
    validate_exact_shape(
        format!("{context}_mla_past_rope"),
        v_dims,
        &[batch, 1, past_tokens, qk_rope_dim],
    )?;
    Ok(past_tokens)
}

fn validate_device_past_kv_layout(
    past_kv: &DevicePagedKvView,
    batch: usize,
    config: &Config,
    kv_lora_rank: usize,
) -> Result<DevicePastKvLayout> {
    let layout = [
        past_kv.batch,
        past_kv.attention_heads,
        past_kv.key_head_dim,
        past_kv.value_head_dim,
    ];
    let expanded = [
        batch,
        config.attention_heads,
        config.qk_head_dim,
        config.v_head_dim(),
    ];
    if layout == expanded {
        return Ok(DevicePastKvLayout::ExpandedHeads);
    }

    let latent = [batch, 1, kv_lora_rank, config.qk_rope_dim];
    if layout == latent {
        return Ok(DevicePastKvLayout::MlaLatent);
    }

    Err(Error::model(format!(
        "GLM sparse device attention requires expanded K/V {:?} or MLA latent K/V {:?}, got {:?}",
        expanded, latent, layout
    )))
}

fn mla_latent_cache(kv_latent: &Tensor) -> Result<Tensor> {
    let dims = kv_latent.dims();
    if dims.len() != 3 {
        return Err(Error::model(format!(
            "GLM MLA latent cache source must be rank 3 [B,T,R], got {dims:?}"
        )));
    }
    kv_latent.unsqueeze(1)
}

fn mla_attention_sequences(
    past_kv: Option<(&Tensor, &Tensor)>,
    current_latent: &Tensor,
    current_rope: &Tensor,
    batch: usize,
    past_tokens: usize,
    kv_lora_rank: usize,
    qk_rope_dim: usize,
) -> Result<(Tensor, Tensor)> {
    let current_latent_dims = current_latent.dims();
    if current_latent_dims.len() != 3 {
        return Err(Error::model(format!(
            "GLM MLA current latent must be rank 3 [B,T,R], got {current_latent_dims:?}"
        )));
    }
    let current_tokens = current_latent_dims[1];
    validate_exact_shape(
        "glm_mla_current_latent",
        current_latent_dims,
        &[batch, current_tokens, kv_lora_rank],
    )?;
    validate_exact_shape(
        "glm_mla_current_rope",
        current_rope.dims(),
        &[batch, current_tokens, 1, qk_rope_dim],
    )?;

    match past_kv {
        Some((past_k, past_v)) => {
            let past_latent = past_k.clone().reshape((batch, past_tokens, kv_lora_rank))?;
            let past_rope = past_v
                .clone()
                .reshape((batch, past_tokens, 1, qk_rope_dim))?;
            Ok((
                Tensor::cat(&[&past_latent, current_latent], 1)?,
                Tensor::cat(&[&past_rope, current_rope], 1)?,
            ))
        }
        None => Ok((current_latent.clone(), current_rope.clone())),
    }
}

#[allow(clippy::too_many_arguments)]
fn mla_selected_attention_sequences(
    past_kv: (&Tensor, &Tensor),
    selected_past_tokens: &[u32],
    current_latent: &Tensor,
    current_rope: &Tensor,
    batch: usize,
    past_tokens: usize,
    kv_lora_rank: usize,
    qk_rope_dim: usize,
) -> Result<(Tensor, Tensor, usize)> {
    validate_exact_shape(
        "glm_mla_selected_current_latent",
        current_latent.dims(),
        &[batch, 1, kv_lora_rank],
    )?;
    validate_exact_shape(
        "glm_mla_selected_current_rope",
        current_rope.dims(),
        &[batch, 1, 1, qk_rope_dim],
    )?;
    validate_mla_past_kv(
        "glm_mla_selected",
        past_kv.0,
        past_kv.1,
        batch,
        kv_lora_rank,
        qk_rope_dim,
    )?;

    let selected_tokens = selected_past_tokens.len();
    let total_tokens = selected_tokens
        .checked_add(1)
        .ok_or_else(|| Error::model("GLM MLA selected attention token count overflow"))?;
    let mut latent = vec![0.0_f32; batch * total_tokens * kv_lora_rank];
    let mut rope = vec![0.0_f32; batch * total_tokens * qk_rope_dim];

    for (selected_index, token) in selected_past_tokens.iter().copied().enumerate() {
        let token = usize::try_from(token)
            .map_err(|_| Error::model("DSA selected token index does not fit usize"))?;
        if token >= past_tokens {
            return Err(Error::model(format!(
                "DSA selected past token {token} exceeds cached past token count {past_tokens}"
            )));
        }
        for batch_index in 0..batch {
            let latent_source = ((batch_index * past_tokens + token) * kv_lora_rank) as usize;
            let latent_target =
                ((batch_index * total_tokens + selected_index) * kv_lora_rank) as usize;
            latent[latent_target..latent_target + kv_lora_rank]
                .copy_from_slice(&past_kv.0.values()[latent_source..latent_source + kv_lora_rank]);

            let rope_source = ((batch_index * past_tokens + token) * qk_rope_dim) as usize;
            let rope_target =
                ((batch_index * total_tokens + selected_index) * qk_rope_dim) as usize;
            rope[rope_target..rope_target + qk_rope_dim]
                .copy_from_slice(&past_kv.1.values()[rope_source..rope_source + qk_rope_dim]);
        }
    }

    for batch_index in 0..batch {
        let latent_source = batch_index * kv_lora_rank;
        let latent_target =
            ((batch_index * total_tokens + selected_tokens) * kv_lora_rank) as usize;
        latent[latent_target..latent_target + kv_lora_rank]
            .copy_from_slice(&current_latent.values()[latent_source..latent_source + kv_lora_rank]);

        let rope_source = batch_index * qk_rope_dim;
        let rope_target = ((batch_index * total_tokens + selected_tokens) * qk_rope_dim) as usize;
        rope[rope_target..rope_target + qk_rope_dim]
            .copy_from_slice(&current_rope.values()[rope_source..rope_source + qk_rope_dim]);
    }

    Ok((
        Tensor::new(latent, [batch, total_tokens, kv_lora_rank])?,
        Tensor::new(rope, [batch, total_tokens, 1, qk_rope_dim])?,
        selected_tokens,
    ))
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

fn require_native<T>(operation: &str, value: Option<T>) -> Result<T> {
    value.ok_or_else(|| {
        Error::backend(format!(
            "native Metal {operation} is required for the GLM-5.2 Q2 attention path"
        ))
    })
}

fn reject_missing_native_q2_kernel_on_metal<B: Backend>(
    backend: &B,
    tensor_name: &str,
) -> Result<()> {
    if backend.capabilities().device == DeviceKind::Metal {
        return Err(Error::backend(format!(
            "native Metal Q2_K matvec is required for {tensor_name}; CPU reference fallback is disabled on Metal"
        )));
    }
    Ok(())
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

    use crate::IndexerIndex;

    use super::*;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn q2_attention_runs_decode_with_past_kv() {
        let path = write_attention_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let attention_index = attention_index(&gguf);
        let input_norm = tensor_ref(&gguf, "blk.0.attn_norm.weight");
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let attention = Attention::open_from_parts(
            &gguf,
            &config,
            0,
            &input_norm,
            &attention_index,
            &backend,
            256,
        )
        .unwrap();
        let prefill_hidden =
            Tensor::from_vec(vec![1.0_f32; 3 * 256], (1, 3, 256), &Device::Cpu).unwrap();
        let prefill = attention
            .forward(&config, &prefill_hidden, &backend)
            .unwrap();
        let decode_hidden =
            Tensor::from_vec(vec![1.0_f32; 256], (1, 1, 256), &Device::Cpu).unwrap();

        let decode = attention
            .forward_with_past_kv(
                &config,
                &decode_hidden,
                &backend,
                Some((&prefill.cache_k, &prefill.cache_v)),
            )
            .unwrap();

        assert_eq!(
            attention.load_report().projection_tensor_type,
            GgmlType::Q2K
        );
        assert_eq!(decode.hidden_states.dims(), &[1, 1, 256]);
        assert_eq!(decode.cache_k.dims(), &[1, 1, 1, 256]);
        assert_eq!(decode.cache_v.dims(), &[1, 1, 1, 128]);
        assert_eq!(decode.report.past_tokens, 3);
        assert_eq!(decode.report.attention_scores_shape.dims(), &[1, 2, 1, 4]);
        assert_eq!(decode.report.attention_k_shape.dims(), &[1, 2, 4, 256]);
        assert_eq!(decode.report.attention_v_shape.dims(), &[1, 2, 4, 256]);
    }

    #[test]
    fn rejects_wrong_split_k_shape() {
        let path = write_attention_fixture_with_bad_k_b();
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let attention_index = attention_index(&gguf);
        let input_norm = tensor_ref(&gguf, "blk.0.attn_norm.weight");
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();

        let err = Attention::open_from_parts(
            &gguf,
            &config,
            0,
            &input_norm,
            &attention_index,
            &backend,
            128,
        )
        .expect_err("wrong k_b output width should fail");

        assert!(err.to_string().contains("gguf_attention_k_b"));
    }

    #[test]
    fn loads_full_dsa_indexer_when_tensor_refs_are_present() {
        let path = write_attention_fixture_with_indexer(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let attention_index = attention_index_with_indexer(&gguf);
        let input_norm = tensor_ref(&gguf, "blk.0.attn_norm.weight");
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();

        let attention = Attention::open_from_parts(
            &gguf,
            &config,
            0,
            &input_norm,
            &attention_index,
            &backend,
            256,
        )
        .unwrap();

        let indexer = attention
            .load_report()
            .indexer
            .as_ref()
            .expect("full indexer layer should load DSA indexer");
        assert!(attention.has_dsa_indexer());
        assert_eq!(indexer.n_heads, 32);
        assert_eq!(indexer.head_dim, 128);
        assert_eq!(indexer.wq_b_shape.dims(), &[4096, 256]);
        assert_eq!(indexer.wk_shape.dims(), &[128, 256]);
    }

    #[test]
    fn mla_selected_attention_sequences_preserve_sparse_order() {
        let past_k = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 1, 3, 2]).unwrap();
        let past_v = Tensor::new(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [1, 1, 3, 2]).unwrap();
        let current_k = Tensor::new(vec![9.0, 10.0], [1, 1, 2]).unwrap();
        let current_v = Tensor::new(vec![90.0, 100.0], [1, 1, 1, 2]).unwrap();

        let (latent, rope, selected_tokens) = mla_selected_attention_sequences(
            (&past_k, &past_v),
            &[2, 0, 2],
            &current_k,
            &current_v,
            1,
            3,
            2,
            2,
        )
        .unwrap();

        assert_eq!(selected_tokens, 3);
        assert_eq!(latent.dims(), &[1, 4, 2]);
        assert_eq!(rope.dims(), &[1, 4, 1, 2]);
        assert_eq!(latent.values(), &[5.0, 6.0, 1.0, 2.0, 5.0, 6.0, 9.0, 10.0]);
        assert_eq!(
            rope.values(),
            &[50.0, 60.0, 10.0, 20.0, 50.0, 60.0, 90.0, 100.0]
        );
    }

    #[test]
    fn dsa_selected_indices_drop_current_token_only() {
        assert_eq!(
            past_only_selected_indices(&[0, 3, 2], 3).unwrap(),
            vec![0, 2]
        );

        let err = past_only_selected_indices(&[4], 3).unwrap_err();
        assert!(err.to_string().contains("future token"));
    }

    fn attention_index(gguf: &GgufFile) -> AttentionIndex {
        AttentionIndex {
            q_a: tensor_ref(gguf, "blk.0.attn_q_a.weight"),
            q_a_norm: tensor_ref(gguf, "blk.0.attn_q_a_norm.weight"),
            q_b: tensor_ref(gguf, "blk.0.attn_q_b.weight"),
            kv_a_mqa: tensor_ref(gguf, "blk.0.attn_kv_a_mqa.weight"),
            kv_a_norm: tensor_ref(gguf, "blk.0.attn_kv_a_norm.weight"),
            k_b: tensor_ref(gguf, "blk.0.attn_k_b.weight"),
            v_b: tensor_ref(gguf, "blk.0.attn_v_b.weight"),
            output: tensor_ref(gguf, "blk.0.attn_output.weight"),
            indexer: None,
        }
    }

    fn attention_index_with_indexer(gguf: &GgufFile) -> AttentionIndex {
        let mut index = attention_index(gguf);
        index.indexer = Some(IndexerIndex {
            k_norm_bias: tensor_ref(gguf, "blk.0.indexer.k_norm.bias"),
            k_norm_weight: tensor_ref(gguf, "blk.0.indexer.k_norm.weight"),
            proj: tensor_ref(gguf, "blk.0.indexer.proj.weight"),
            attn_k: tensor_ref(gguf, "blk.0.indexer.attn_k.weight"),
            attn_q_b: tensor_ref(gguf, "blk.0.indexer.attn_q_b.weight"),
        });
        index
    }

    fn tensor_ref(gguf: &GgufFile, name: &str) -> TensorRef {
        let info = gguf.tensor(name).unwrap();
        TensorRef {
            name: info.name.clone(),
            dims: info.dims.clone(),
            ty: info.ty,
            absolute_offset: info.absolute_offset,
            storage_byte_len: info.storage_byte_len,
        }
    }

    fn tiny_config() -> Config {
        Config {
            model_type: "glm_moe_dsa".to_string(),
            hidden_size: 256,
            num_layers: 1,
            dense_layers: 1,
            sparse_moe_layers: Some(0),
            vocab_size: 8,
            attention_heads: 2,
            qk_head_dim: 256,
            qk_no_rope_dim: 128,
            qk_rope_dim: 128,
            kv_lora_rank: 256,
            v_head_dim: Some(256),
            num_routed_experts: 1,
            experts_per_token: 1,
            moe_intermediate_size: 256,
            num_shared_experts: 1,
            moe_groups: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
            scoring_func: "softmax".to_string(),
            topk_method: "greedy".to_string(),
            max_context: 32,
            dsa_index_topk: 1,
            index_head_dim: 128,
            index_n_heads: 32,
            index_topk_freq: 4,
            indexer_rope_interleave: true,
            indexer_types: Vec::new(),
            num_nextn_predict_layers: 0,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000_000.0,
        }
    }

    fn write_attention_fixture(ty: GgmlType) -> PathBuf {
        write_attention_fixture_with_k_b_dims(ty, &[128, 256, 2])
    }

    fn write_attention_fixture_with_bad_k_b() -> PathBuf {
        write_attention_fixture_with_k_b_dims(GgmlType::Q2K, &[256, 256, 2])
    }

    fn write_attention_fixture_with_indexer(ty: GgmlType) -> PathBuf {
        let path = unique_temp_file("attention-indexer");
        let specs = vec![
            TensorSpec::f32("blk.0.attn_norm.weight", vec![256]),
            TensorSpec::quant("blk.0.attn_q_a.weight", vec![256, 256], ty),
            TensorSpec::f32("blk.0.attn_q_a_norm.weight", vec![256]),
            TensorSpec::quant("blk.0.attn_q_b.weight", vec![256, 512], ty),
            TensorSpec::quant("blk.0.attn_kv_a_mqa.weight", vec![256, 384], ty),
            TensorSpec::f32("blk.0.attn_kv_a_norm.weight", vec![256]),
            TensorSpec::quant("blk.0.attn_k_b.weight", vec![128, 256, 2], GgmlType::Q8_0),
            TensorSpec::quant("blk.0.attn_v_b.weight", vec![256, 256, 2], ty),
            TensorSpec::quant("blk.0.attn_output.weight", vec![512, 256], ty),
            TensorSpec::f32("blk.0.indexer.k_norm.bias", vec![128]),
            TensorSpec::f32("blk.0.indexer.k_norm.weight", vec![128]),
            TensorSpec::f32("blk.0.indexer.proj.weight", vec![256, 32]),
            TensorSpec::quant(
                "blk.0.indexer.attn_k.weight",
                vec![256, 128],
                GgmlType::Q8_0,
            ),
            TensorSpec::quant(
                "blk.0.indexer.attn_q_b.weight",
                vec![256, 4096],
                GgmlType::Q8_0,
            ),
        ];
        write_gguf(path, &specs)
    }

    fn write_attention_fixture_with_k_b_dims(ty: GgmlType, k_b_dims: &[u64]) -> PathBuf {
        let path = unique_temp_file("attention");
        let specs = vec![
            TensorSpec::f32("blk.0.attn_norm.weight", vec![256]),
            TensorSpec::quant("blk.0.attn_q_a.weight", vec![256, 256], ty),
            TensorSpec::f32("blk.0.attn_q_a_norm.weight", vec![256]),
            TensorSpec::quant("blk.0.attn_q_b.weight", vec![256, 512], ty),
            TensorSpec::quant("blk.0.attn_kv_a_mqa.weight", vec![256, 384], ty),
            TensorSpec::f32("blk.0.attn_kv_a_norm.weight", vec![256]),
            TensorSpec::quant("blk.0.attn_k_b.weight", k_b_dims.to_vec(), GgmlType::Q8_0),
            TensorSpec::quant("blk.0.attn_v_b.weight", vec![256, 256, 2], ty),
            TensorSpec::quant("blk.0.attn_output.weight", vec![512, 256], ty),
        ];
        write_gguf(path, &specs)
    }

    #[derive(Debug)]
    struct TensorSpec {
        name: &'static str,
        dims: Vec<u64>,
        ty: GgmlType,
    }

    impl TensorSpec {
        fn f32(name: &'static str, dims: Vec<u64>) -> Self {
            Self {
                name,
                dims,
                ty: GgmlType::F32,
            }
        }

        fn quant(name: &'static str, dims: Vec<u64>, ty: GgmlType) -> Self {
            Self { name, dims, ty }
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
            writer.tensor_info(spec.name, &spec.dims, spec.ty, offset);
        }
        writer.pad_to(32);

        for (spec, offset) in specs.iter().zip(offsets.iter().copied()) {
            writer.pad_to_absolute_data_offset(offset);
            match spec.ty {
                GgmlType::F32 => {
                    for _ in 0..spec.dims.iter().product::<u64>() {
                        writer.bytes(&1.0_f32.to_le_bytes());
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
        std::env::temp_dir().join(format!("attention-{label}-{}-{id}", std::process::id()))
    }
}
