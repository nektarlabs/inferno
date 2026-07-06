use backend::{Backend, BackendCapabilities, DeviceValue};
use common::Tensor;
use common::{validate_exact_shape, DeviceKind, Error, F32Tensor, PagedKvView, Result, Shape};
use config::Config;
use gguf::{
    matmul_q2_k_payload_f32, GgmlType, GgufFile, GgufQuantBlockKind, GGML_Q2_K_BLOCK_BYTES,
    GGML_Q8_0_BLOCK_BYTES,
};

use crate::{
    attention_math, AttentionIndex, LayerIndex, QuantizedLinear, RmsNorm, RmsNormLoadReport,
    TensorRef,
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
}

#[derive(Debug)]
pub struct AttentionF32Tensors {
    pub hidden_states: F32Tensor,
    pub cache_k: F32Tensor,
    pub cache_v: F32Tensor,
}

/// Output of the batched device-resident decode attention path. The hidden
/// states stay on the GPU for the next op; the current token's K/V are
/// downloaded because the paged KV cache lives on the host.
#[derive(Debug)]
pub(crate) struct AttentionDeviceTensors {
    pub(crate) hidden_states: DeviceValue,
    pub(crate) cache_k: F32Tensor,
    pub(crate) cache_v: F32Tensor,
}

#[derive(Debug)]
pub enum AttentionPastKv<'a> {
    Contiguous {
        cache_k: &'a F32Tensor,
        cache_v: &'a F32Tensor,
    },
    Paged(PagedKvView<'a>),
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

impl<'a> HeadProjection<'a> {
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

    /// Batched device-resident variant of `forward_heads_f32`. For the
    /// transposed (`OutputInputHeads`) layout the per-head outputs are stacked
    /// with GPU copies, which reproduces the eager path's host scatter only
    /// when there is a single flat token — so that layout bails out with
    /// `Ok(None)` outside single-token decode.
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
                if flat_tokens != 1 {
                    return Ok(None);
                }
                self.forward_transposed_heads_device(backend, &flat_input, batch, tokens)
            }
            (other, _) => Err(Error::gguf(format!(
                "GGUF tensor {} must be Q2_K or Q8_0 for device head projection, got {}",
                self.tensor_ref.name, other
            ))),
        }
    }

    /// Single-token transposed head projection: one encoded transposed matvec
    /// per head against that head's payload slice, stacked into `[1, 1, heads,
    /// output_features]` with GPU copies. No synchronization happens here.
    fn forward_transposed_heads_device<B: Backend>(
        &self,
        backend: &B,
        flat_input: &DeviceValue,
        batch: usize,
        tokens: usize,
    ) -> Result<Option<DeviceValue>> {
        let (raw_data, block_bytes) = match self.tensor_ref.ty {
            GgmlType::Q2K => (
                self.q2_payload_bytes.ok_or_else(|| {
                    Error::gguf(format!(
                        "GGUF tensor {} must be Q2_K for the device transposed Q2 head projection path, got {}",
                        self.tensor_ref.name, self.tensor_ref.ty
                    ))
                })?,
                GGML_Q2_K_BLOCK_BYTES as usize,
            ),
            GgmlType::Q8_0 => (
                self.q8_payload_bytes.ok_or_else(|| {
                    Error::gguf(format!(
                        "GGUF tensor {} must be Q8_0 for the device transposed Q8 head projection path, got {}",
                        self.tensor_ref.name, self.tensor_ref.ty
                    ))
                })?,
                GGML_Q8_0_BLOCK_BYTES as usize,
            ),
            other => {
                return Err(Error::gguf(format!(
                    "GGUF tensor {} must be Q2_K or Q8_0 for device transposed head projection, got {other}",
                    self.tensor_ref.name
                )))
            }
        };
        let head_byte_len = usize::try_from(self.blocks_per_head)
            .ok()
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} device transposed head byte length overflow",
                    self.tensor_ref.name
                ))
            })?;
        let expected_payload_len = head_byte_len.checked_mul(self.heads).ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} device transposed payload length overflow",
                self.tensor_ref.name
            ))
        })?;
        validate_exact_shape(
            format!(
                "gguf_device_attention_transposed_head_projection_payload:{}",
                self.tensor_ref.name
            ),
            &[raw_data.len()],
            &[expected_payload_len],
        )?;

        let mut head_outputs = Vec::with_capacity(self.heads);
        for head_index in 0..self.heads {
            let byte_start = head_index * head_byte_len;
            let head_bytes = &raw_data[byte_start..byte_start + head_byte_len];
            let head_output = match self.tensor_ref.ty {
                GgmlType::Q2K => crate::try_device!(backend.q2_k_transposed_matvec_device(
                    head_bytes,
                    flat_input,
                    1,
                    self.input_features,
                    self.output_features,
                )),
                _ => crate::try_device!(backend.q8_0_transposed_matvec_device(
                    head_bytes,
                    flat_input,
                    1,
                    self.input_features,
                    self.output_features,
                )),
            };
            head_outputs.push(head_output);
        }
        let stacked = crate::try_device!(backend.moe_stack_rows_device(&head_outputs));
        Ok(Some(stacked.reshape(vec![
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
        if layer_index >= config.num_layers {
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
            projection_tensor_type: attention.q_a.ty,
            output_chunk_rows,
            limitations: vec![
                "GGUF attention uses split k_b and v_b tensors as stored by Antirez GLM-5.2 GGUF"
                    .to_string(),
                "non-Q2 projection tensors still use chunked reference decoding".to_string(),
                "paged attention and DSA sparse attention are not active in this component yet"
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
            kv_lora_rank,
            load_report,
        })
    }

    pub fn load_report(&self) -> &AttentionLoadReport {
        &self.load_report
    }

    pub fn forward<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<AttentionOutput> {
        self.forward_with_past_kv(config, hidden_states, backend, None)
    }

    pub fn forward_tensors<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<AttentionTensors> {
        self.forward_tensors_with_past_kv(config, hidden_states, backend, None)
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
            Some((past_k, past_v)) => validate_past_kv(
                past_k,
                past_v,
                batch,
                config.attention_heads,
                config.qk_head_dim,
                config.v_head_dim(),
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
        let kv_a_norm = self.kv_a_norm.forward(&kv_latent, backend)?;
        let k_b = self.k_b.forward_heads(&kv_a_norm.hidden_states, backend)?;
        let v_b = self.v_b.forward_heads(&kv_a_norm.hidden_states, backend)?;
        let k_no_rope = k_b.output_heads;
        let v_heads = v_b.output_heads;
        let k_rope_after_rope = backend.rope_slice(
            &k_rope_mqa,
            config.qk_rope_dim,
            past_tokens,
            config.rope_theta as f32,
        )?;
        let k_heads = backend.combine_rope_tail(&k_no_rope, &k_rope_after_rope)?;

        let q_for_attention = backend.heads_to_attention_layout(&q_recombined)?;
        let k_for_attention = backend.heads_to_attention_layout(&k_heads)?;
        let v_for_attention = backend.heads_to_attention_layout(&v_heads)?;
        let (attention_k, attention_v) = match past_kv {
            Some((past_k, past_v)) => (
                Tensor::cat(&[past_k, &k_for_attention], 2)?,
                Tensor::cat(&[past_v, &v_for_attention], 2)?,
            ),
            None => (k_for_attention.clone(), v_for_attention.clone()),
        };

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
            cache_k_shape: Shape::new(k_for_attention.dims().to_vec()),
            cache_v_shape: Shape::new(v_for_attention.dims().to_vec()),
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
            cache_k: k_for_attention,
            cache_v: v_for_attention,
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
            Some((past_k, past_v)) => validate_past_kv(
                past_k,
                past_v,
                batch,
                config.attention_heads,
                config.qk_head_dim,
                config.v_head_dim(),
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
        let kv_a_norm = self.kv_a_norm.forward_tensor(&kv_latent, backend)?;
        let k_no_rope = self.k_b.forward_heads(&kv_a_norm, backend)?.output_heads;
        let v_heads = self.v_b.forward_heads(&kv_a_norm, backend)?.output_heads;
        let k_rope_after_rope = backend.rope_slice(
            &k_rope_mqa,
            config.qk_rope_dim,
            past_tokens,
            config.rope_theta as f32,
        )?;
        let k_heads = backend.combine_rope_tail(&k_no_rope, &k_rope_after_rope)?;

        let q_for_attention = backend.heads_to_attention_layout(&q_recombined)?;
        let k_for_attention = backend.heads_to_attention_layout(&k_heads)?;
        let v_for_attention = backend.heads_to_attention_layout(&v_heads)?;
        let (attention_k, attention_v) = match past_kv {
            Some((past_k, past_v)) => (
                Tensor::cat(&[past_k, &k_for_attention], 2)?,
                Tensor::cat(&[past_v, &v_for_attention], 2)?,
            ),
            None => (k_for_attention.clone(), v_for_attention.clone()),
        };

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
            cache_k: tensor_to_f32_tensor(&k_for_attention)?,
            cache_v: tensor_to_f32_tensor(&v_for_attention)?,
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

    pub fn forward_f32_tensors_with_paged_past_kv<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        backend: &B,
        past_kv: Option<PagedKvView<'_>>,
    ) -> Result<AttentionF32Tensors> {
        self.forward_f32_tensors_with_attention_past_kv(
            config,
            hidden_states,
            backend,
            past_kv.map(AttentionPastKv::Paged),
        )
    }

    fn forward_f32_tensors_with_attention_past_kv<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        backend: &B,
        past_kv: Option<AttentionPastKv<'_>>,
    ) -> Result<AttentionF32Tensors> {
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
            Some(AttentionPastKv::Contiguous { cache_k, cache_v }) => validate_past_kv_f32(
                cache_k,
                cache_v,
                batch,
                config.attention_heads,
                config.qk_head_dim,
                config.v_head_dim(),
            )?,
            Some(AttentionPastKv::Paged(paged)) => {
                paged.validate()?;
                validate_exact_shape(
                    "gguf_native_attention_paged_past_shape",
                    &[
                        paged.batch,
                        paged.attention_heads,
                        paged.key_head_dim,
                        paged.value_head_dim,
                    ],
                    &[
                        batch,
                        config.attention_heads,
                        config.qk_head_dim,
                        config.v_head_dim(),
                    ],
                )?;
                paged.cached_tokens
            }
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
        let kv_a_norm = self
            .kv_a_norm
            .forward_f32(&kv_latent, backend)?
            .hidden_states;
        let k_no_rope = self
            .k_b
            .forward_heads_f32(&kv_a_norm, backend)?
            .output_heads;
        let v_for_cache = self
            .v_b
            .forward_heads_f32(&kv_a_norm, backend)?
            .output_heads;
        let k_rope_after_rope = require_native(
            "rope_slice",
            backend.rope_slice_f32_tensor(
                &k_rope_mqa,
                config.qk_rope_dim,
                past_tokens,
                config.rope_theta as f32,
            )?,
        )?;
        let k_heads = require_native(
            "combine_rope_tail",
            backend.combine_rope_tail_f32_tensor(&k_no_rope, &k_rope_after_rope)?,
        )?;

        let q_for_attention = require_native(
            "heads_to_attention_layout",
            backend.heads_to_attention_layout_f32_tensor(&q_recombined)?,
        )?;
        let k_for_cache = require_native(
            "heads_to_attention_layout",
            backend.heads_to_attention_layout_f32_tensor(&k_heads)?,
        )?;
        let v_for_cache = require_native(
            "heads_to_attention_layout",
            backend.heads_to_attention_layout_f32_tensor(&v_for_cache)?,
        )?;
        let context_heads = if let Some(AttentionPastKv::Paged(paged)) = past_kv.as_ref() {
            validate_exact_shape("paged_decode_attention_tokens", &[tokens], &[1])?;
            require_native(
                "paged_decode_attention",
                backend.paged_decode_attention_f32_tensor(
                    &q_for_attention,
                    &k_for_cache,
                    &v_for_cache,
                    paged,
                )?,
            )?
        } else {
            let (attention_k, attention_v) = match past_kv.as_ref() {
                Some(AttentionPastKv::Contiguous { cache_k, cache_v }) => (
                    concat_attention_kv_f32("K", cache_k, &k_for_cache)?,
                    concat_attention_kv_f32("V", cache_v, &v_for_cache)?,
                ),
                Some(AttentionPastKv::Paged(_)) => unreachable!("paged attention handled above"),
                None => (k_for_cache.clone(), v_for_cache.clone()),
            };

            if tokens == 1 {
                require_native(
                    "decode_attention",
                    backend.decode_attention_f32_tensor(
                        &q_for_attention,
                        &attention_k,
                        &attention_v,
                        config.qk_head_dim,
                        past_tokens,
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
                    backend
                        .attention_causal_softmax_f32_tensor(&raw_attention_scores, past_tokens)?,
                )?;
                require_native(
                    "attention_values",
                    backend.attention_values_f32_tensor(&attention_probs, &attention_v)?,
                )?
            }
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

        Ok(AttentionF32Tensors {
            hidden_states: output_hidden_states,
            cache_k: k_for_cache,
            cache_v: v_for_cache,
        })
    }

    /// Batched device-resident decode attention: mirrors the native paged
    /// path above, but every kernel is encoded into the backend's open batch
    /// and intermediate tensors never leave the GPU. The single
    /// synchronization is the download of the current token's K/V at the end,
    /// which the host paged cache needs for its append.
    ///
    /// Returns `Ok(None)` when any component has no device path (the caller
    /// falls back to the eager route); requires a single decode token.
    pub(crate) fn forward_decode_device<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &DeviceValue,
        backend: &B,
        past_kv: &PagedKvView<'_>,
    ) -> Result<Option<AttentionDeviceTensors>> {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF device attention input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        if tokens != 1 {
            return Ok(None);
        }
        validate_exact_shape(
            "gguf_device_attention_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;
        past_kv.validate()?;
        validate_exact_shape(
            "gguf_device_attention_paged_past_shape",
            &[
                past_kv.batch,
                past_kv.attention_heads,
                past_kv.key_head_dim,
                past_kv.value_head_dim,
            ],
            &[
                batch,
                config.attention_heads,
                config.qk_head_dim,
                config.v_head_dim(),
            ],
        )?;
        let past_tokens = past_kv.cached_tokens;

        let input_norm = crate::try_device!(self.input_norm.forward_device(hidden_states, backend));
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
        let q_recombined =
            crate::try_device!(backend.combine_rope_tail_device(&q_no_rope, &q_rope_after_rope));

        let kv_a_mqa = crate::try_device!(self.kv_a_mqa.forward_device(&input_norm, backend));
        let (kv_latent, k_rope_mqa) = crate::try_device!(backend.split_kv_mqa_device(
            &kv_a_mqa,
            self.kv_lora_rank,
            config.qk_rope_dim,
        ));
        let kv_a_norm = crate::try_device!(self.kv_a_norm.forward_device(&kv_latent, backend));
        let k_no_rope = crate::try_device!(self.k_b.forward_heads_device(&kv_a_norm, backend));
        let v_heads = crate::try_device!(self.v_b.forward_heads_device(&kv_a_norm, backend));
        let k_rope_after_rope = crate::try_device!(backend.rope_slice_device(
            &k_rope_mqa,
            config.qk_rope_dim,
            past_tokens,
            config.rope_theta as f32,
        ));
        let k_heads =
            crate::try_device!(backend.combine_rope_tail_device(&k_no_rope, &k_rope_after_rope));

        let q_for_attention =
            crate::try_device!(backend.heads_to_attention_layout_device(&q_recombined));
        let k_for_cache = crate::try_device!(backend.heads_to_attention_layout_device(&k_heads));
        let v_for_cache = crate::try_device!(backend.heads_to_attention_layout_device(&v_heads));

        let context_heads = crate::try_device!(backend.paged_decode_attention_device(
            &q_for_attention,
            &k_for_cache,
            &v_for_cache,
            past_kv,
        ));
        let merged_attention_output =
            crate::try_device!(backend.merge_attention_heads_device(&context_heads));
        let output_hidden_states = crate::try_device!(self.output.forward_device_add_residual(
            &merged_attention_output,
            hidden_states,
            backend,
        ));

        // The paged KV cache lives on the host, so the current token's K/V
        // must come back. This download flushes the batch — the one GPU sync
        // for everything encoded above.
        let cache_k = backend.device_download_f32_tensor(&k_for_cache)?;
        let cache_v = backend.device_download_f32_tensor(&v_for_cache)?;

        Ok(Some(AttentionDeviceTensors {
            hidden_states: output_hidden_states,
            cache_k,
            cache_v,
        }))
    }
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

fn validate_past_kv(
    past_k: &Tensor,
    past_v: &Tensor,
    batch: usize,
    attention_heads: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
) -> Result<usize> {
    let k_dims = past_k.dims();
    let v_dims = past_v.dims();
    if k_dims.len() != 4 || v_dims.len() != 4 {
        return Err(Error::model(format!(
            "GGUF attention past K/V must be rank 4 [B,H,T,D], got k={k_dims:?} v={v_dims:?}"
        )));
    }
    let past_tokens = k_dims[2];
    validate_exact_shape(
        "gguf_attention_past_k",
        k_dims,
        &[batch, attention_heads, past_tokens, qk_head_dim],
    )?;
    validate_exact_shape(
        "gguf_attention_past_v",
        v_dims,
        &[batch, attention_heads, past_tokens, v_head_dim],
    )?;
    Ok(past_tokens)
}

fn validate_past_kv_f32(
    past_k: &F32Tensor,
    past_v: &F32Tensor,
    batch: usize,
    attention_heads: usize,
    qk_head_dim: usize,
    v_head_dim: usize,
) -> Result<usize> {
    let k_dims = past_k.dims();
    let v_dims = past_v.dims();
    if k_dims.len() != 4 || v_dims.len() != 4 {
        return Err(Error::model(format!(
            "GGUF native attention past K/V must be rank 4 [B,H,T,D], got k={k_dims:?} v={v_dims:?}"
        )));
    }
    let past_tokens = k_dims[2];
    validate_exact_shape(
        "gguf_native_attention_past_k",
        k_dims,
        &[batch, attention_heads, past_tokens, qk_head_dim],
    )?;
    validate_exact_shape(
        "gguf_native_attention_past_v",
        v_dims,
        &[batch, attention_heads, past_tokens, v_head_dim],
    )?;
    Ok(past_tokens)
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

fn concat_attention_kv_f32(name: &str, past: &F32Tensor, current: &F32Tensor) -> Result<F32Tensor> {
    let past_dims = past.dims();
    let current_dims = current.dims();
    if past_dims.len() != 4 || current_dims.len() != 4 {
        return Err(Error::model(format!(
            "native attention {name} concat expects rank 4 [B,H,T,D], got past={past_dims:?} current={current_dims:?}"
        )));
    }
    validate_exact_shape(
        format!("native_attention_{name}_concat_batch_heads_dim"),
        &[current_dims[0], current_dims[1], current_dims[3]],
        &[past_dims[0], past_dims[1], past_dims[3]],
    )?;
    let batch = past_dims[0];
    let heads = past_dims[1];
    let past_tokens = past_dims[2];
    let current_tokens = current_dims[2];
    let dim = past_dims[3];
    let total_tokens = past_tokens.checked_add(current_tokens).ok_or_else(|| {
        Error::model(format!(
            "native attention {name} concat token count overflow"
        ))
    })?;

    let mut output = vec![0.0_f32; batch * heads * total_tokens * dim];
    for batch_index in 0..batch {
        for head_index in 0..heads {
            for token_index in 0..past_tokens {
                let source = (((batch_index * heads + head_index) * past_tokens + token_index)
                    * dim) as usize;
                let target = (((batch_index * heads + head_index) * total_tokens + token_index)
                    * dim) as usize;
                output[target..target + dim].copy_from_slice(&past.values()[source..source + dim]);
            }
            for token_index in 0..current_tokens {
                let source = (((batch_index * heads + head_index) * current_tokens + token_index)
                    * dim) as usize;
                let target = (((batch_index * heads + head_index) * total_tokens
                    + past_tokens
                    + token_index)
                    * dim) as usize;
                output[target..target + dim]
                    .copy_from_slice(&current.values()[source..source + dim]);
            }
        }
    }

    F32Tensor::new(output, [batch, heads, total_tokens, dim])
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
        assert_eq!(decode.cache_k.dims(), &[1, 2, 1, 256]);
        assert_eq!(decode.cache_v.dims(), &[1, 2, 1, 256]);
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
