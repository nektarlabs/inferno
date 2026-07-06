use std::cmp::Ordering;

use backend::{Backend, BackendCapabilities};
use common::Tensor;
use common::{validate_exact_shape, Error, F32Tensor, Result, Shape};
use config::Config;
use gguf::{GgmlType, GgufFile};
use moe::{build_dispatch_plan, ExpertDispatch, TopKSelection};

use crate::{
    FfnIndex, LayerIndex, QuantizedLinear, RmsNorm, RmsNormLoadReport, TensorLoadReport, TensorRef,
    WeightLoader,
};

#[derive(Debug)]
pub struct MoeRouter<'a> {
    layer_index: usize,
    post_attention_norm: RmsNorm,
    router: RouterProjection<'a>,
    correction_bias: Vec<f32>,
    load_report: MoeRouterLoadReport,
}

#[derive(Debug)]
enum RouterProjection<'a> {
    Quantized(QuantizedLinear<'a>),
    F32 {
        weight: F32Tensor,
        input_features: usize,
        output_features: usize,
    },
}

#[derive(Debug)]
struct RouterProjectionOutput {
    output: Tensor,
    chunk_count: usize,
}

#[derive(Debug)]
pub struct MoeRoutingOutput {
    pub normed_hidden_states: Tensor,
    pub dispatch_plan: Vec<ExpertDispatch>,
    pub report: MoeRoutingReport,
}

#[derive(Debug)]
pub struct MoeRoutingTensors {
    pub normed_hidden_states: Tensor,
    pub dispatch_plan: Vec<ExpertDispatch>,
}

#[derive(Debug)]
pub(crate) struct MoeRoutingDeviceTensors {
    pub(crate) normed_hidden_states: backend::DeviceValue,
    pub(crate) dispatch_plan: Vec<ExpertDispatch>,
}

pub struct MoeRoutingF32Tensors {
    pub normed_hidden_states: F32Tensor,
    pub dispatch_plan: Vec<ExpertDispatch>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoeRouterLoadReport {
    pub backend: BackendCapabilities,
    pub layer_index: usize,
    pub post_attention_norm: RmsNormLoadReport,
    pub router_weight_shape: Shape,
    pub correction_bias: TensorLoadReport,
    pub router_tensor_type: GgmlType,
    pub output_chunk_rows: usize,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoeRoutingReport {
    pub layer_index: usize,
    pub input_hidden_states_shape: Shape,
    pub normed_hidden_states_shape: Shape,
    pub flat_tokens_shape: Shape,
    pub router_logits_shape: Shape,
    pub router_logits_chunk_count: usize,
    pub topk_expert_ids_shape: Shape,
    pub topk_weights_shape: Shape,
    pub dispatch_expert_count: usize,
    pub dispatch_assignment_count: usize,
}

impl<'a> MoeRouter<'a> {
    pub fn open<B: Backend>(
        gguf: &'a GgufFile,
        config: &Config,
        layer: &LayerIndex,
        backend: &B,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        let FfnIndex::SparseMoe {
            router,
            router_correction_bias,
            ..
        } = &layer.ffn
        else {
            return Err(Error::gguf(format!(
                "GLM-5.2 GGUF layer {} is dense, not sparse MoE",
                layer.layer_index
            )));
        };
        Self::open_from_parts(
            gguf,
            config,
            layer.layer_index,
            &layer.post_attention_norm,
            router,
            router_correction_bias,
            backend,
            output_chunk_rows,
        )
    }

    pub fn open_from_parts<B: Backend>(
        gguf: &'a GgufFile,
        config: &Config,
        layer_index: usize,
        post_attention_norm_ref: &TensorRef,
        router_ref: &TensorRef,
        correction_bias_ref: &TensorRef,
        backend: &B,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        validate_sparse_layer(config, layer_index)?;
        validate_router_weight(config, router_ref)?;
        if correction_bias_ref.ty != GgmlType::F32 {
            return Err(Error::gguf(format!(
                "GLM-5.2 GGUF router correction bias {} must be F32, got {}",
                correction_bias_ref.name, correction_bias_ref.ty
            )));
        }

        let post_attention_norm = RmsNorm::open(gguf, config, post_attention_norm_ref, backend)?;
        let router = RouterProjection::open(
            gguf,
            router_ref,
            config.hidden_size,
            config.num_routed_experts,
            backend,
            output_chunk_rows,
        )?;
        let loaded_correction_bias =
            WeightLoader::new(gguf).load_tensor_as_f32_tensor(correction_bias_ref)?;
        validate_exact_shape(
            "gguf_moe_router_correction_bias",
            loaded_correction_bias.report.shape.dims(),
            &[config.num_routed_experts],
        )?;
        let correction_bias = loaded_correction_bias.tensor.values().to_vec();
        validate_exact_shape(
            "gguf_moe_router_correction_bias_host",
            &[correction_bias.len()],
            &[config.num_routed_experts],
        )?;

        let load_report = MoeRouterLoadReport {
            backend: backend.capabilities(),
            layer_index,
            post_attention_norm: post_attention_norm.load_report().clone(),
            router_weight_shape: logical_weight_shape(router_ref)?,
            correction_bias: loaded_correction_bias.report,
            router_tensor_type: router_ref.ty,
            output_chunk_rows,
            limitations: vec![
                "GGUF MoE router loads only norm, router projection, and correction bias"
                    .to_string(),
                "shared experts and routed expert payloads are intentionally not touched here"
                    .to_string(),
                "grouped routing is rejected until the GLM-5.2 runtime needs it".to_string(),
            ],
        };

        Ok(Self {
            layer_index,
            post_attention_norm,
            router,
            correction_bias,
            load_report,
        })
    }

    pub fn load_report(&self) -> &MoeRouterLoadReport {
        &self.load_report
    }

    pub fn route<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<MoeRoutingOutput> {
        validate_supported_router_config(config)?;
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::moe(format!(
                "GLM-5.2 GGUF MoE router input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            "gguf_moe_router_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;

        let flat_token_count = batch
            .checked_mul(tokens)
            .ok_or_else(|| Error::moe("GGUF MoE flat token count overflow"))?;
        let normed = self.post_attention_norm.forward(hidden_states, backend)?;
        let flat_tokens = normed
            .hidden_states
            .contiguous()?
            .reshape((flat_token_count, config.hidden_size))?;
        let router_logits = self.router.forward(&flat_tokens, backend)?;
        validate_exact_shape(
            "gguf_moe_router_logits",
            router_logits.output.dims(),
            &[flat_token_count, config.num_routed_experts],
        )?;

        let logits_host = router_logits.output.to_vec2::<f32>()?;
        let topk_selections = select_topk(config, &logits_host, &self.correction_bias)?;
        let dispatch_plan = build_dispatch_plan(&topk_selections, config.num_routed_experts)?;
        let dispatch_expert_count = dispatch_plan.len();
        let dispatch_assignment_count = dispatch_plan
            .iter()
            .map(|dispatch| dispatch.assignments.len())
            .sum::<usize>();
        let report = MoeRoutingReport {
            layer_index: self.layer_index,
            input_hidden_states_shape: Shape::new(hidden_states.dims().to_vec()),
            normed_hidden_states_shape: Shape::new(normed.hidden_states.dims().to_vec()),
            flat_tokens_shape: Shape::new(flat_tokens.dims().to_vec()),
            router_logits_shape: Shape::new(router_logits.output.dims().to_vec()),
            router_logits_chunk_count: router_logits.chunk_count,
            topk_expert_ids_shape: Shape::new(vec![flat_token_count, config.experts_per_token]),
            topk_weights_shape: Shape::new(vec![flat_token_count, config.experts_per_token]),
            dispatch_expert_count,
            dispatch_assignment_count,
        };

        Ok(MoeRoutingOutput {
            normed_hidden_states: normed.hidden_states,
            dispatch_plan,
            report,
        })
    }

    pub fn route_tensors<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<MoeRoutingTensors> {
        if backend.capabilities().custom_kernels {
            let hidden_states = tensor_to_f32_tensor(hidden_states)?;
            let routed = self.route_f32_tensors(config, &hidden_states, backend)?;
            return Ok(MoeRoutingTensors {
                normed_hidden_states: tensor_from_f32_tensor(
                    routed.normed_hidden_states,
                    backend.device(),
                )?,
                dispatch_plan: routed.dispatch_plan,
            });
        }

        validate_supported_router_config(config)?;
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::moe(format!(
                "GLM-5.2 GGUF MoE router input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            "gguf_moe_router_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;

        let flat_token_count = batch
            .checked_mul(tokens)
            .ok_or_else(|| Error::moe("GGUF MoE flat token count overflow"))?;
        let normed_hidden_states = self
            .post_attention_norm
            .forward_tensor(hidden_states, backend)?;
        let flat_tokens = normed_hidden_states
            .contiguous()?
            .reshape((flat_token_count, config.hidden_size))?;
        let router_logits = self.router.forward_tensor(&flat_tokens, backend)?;
        validate_exact_shape(
            "gguf_moe_router_logits",
            router_logits.dims(),
            &[flat_token_count, config.num_routed_experts],
        )?;

        let logits_host = router_logits.to_vec2::<f32>()?;
        let topk_selections = select_topk(config, &logits_host, &self.correction_bias)?;
        let dispatch_plan = build_dispatch_plan(&topk_selections, config.num_routed_experts)?;

        Ok(MoeRoutingTensors {
            normed_hidden_states,
            dispatch_plan,
        })
    }

    pub fn route_f32_tensors<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        backend: &B,
    ) -> Result<MoeRoutingF32Tensors> {
        validate_supported_router_config(config)?;
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::moe(format!(
                "GLM-5.2 GGUF native MoE router input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            "gguf_native_moe_router_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;

        let flat_token_count = batch
            .checked_mul(tokens)
            .ok_or_else(|| Error::moe("GGUF native MoE flat token count overflow"))?;
        let normed_hidden_states = self
            .post_attention_norm
            .forward_f32(hidden_states, backend)?
            .hidden_states;
        let flat_tokens = normed_hidden_states
            .clone()
            .reshape([flat_token_count, config.hidden_size])?;
        let router_logits = self.router.forward_f32_tensor(&flat_tokens, backend)?;
        validate_exact_shape(
            "gguf_native_moe_router_logits",
            router_logits.dims(),
            &[flat_token_count, config.num_routed_experts],
        )?;

        let logits_host = matrix_rows(
            router_logits.values(),
            flat_token_count,
            config.num_routed_experts,
        )?;
        let topk_selections = select_topk(config, &logits_host, &self.correction_bias)?;
        let dispatch_plan = build_dispatch_plan(&topk_selections, config.num_routed_experts)?;

        Ok(MoeRoutingF32Tensors {
            normed_hidden_states,
            dispatch_plan,
        })
    }

    /// Batched device-resident routing: the post-attention norm is encoded on
    /// the GPU, but the top-k expert selection is CPU logic, so the normed
    /// activations are downloaded here. That download flushes the batch and is
    /// the MoE block's synchronization point.
    pub(crate) fn route_device<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &backend::DeviceValue,
        backend: &B,
    ) -> Result<Option<MoeRoutingDeviceTensors>> {
        validate_supported_router_config(config)?;
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::moe(format!(
                "GLM-5.2 GGUF device MoE router input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            "gguf_device_moe_router_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;
        let flat_token_count = batch
            .checked_mul(tokens)
            .ok_or_else(|| Error::moe("GGUF device MoE flat token count overflow"))?;

        let normed_hidden_states = crate::try_device!(self
            .post_attention_norm
            .forward_device(hidden_states, backend));
        let flat_tokens =
            normed_hidden_states.reshape(vec![flat_token_count, config.hidden_size])?;
        let flat_tokens_host = backend.device_download_f32_tensor(&flat_tokens)?;
        let router_logits = self.router.forward_f32_tensor(&flat_tokens_host, backend)?;
        validate_exact_shape(
            "gguf_device_moe_router_logits",
            router_logits.dims(),
            &[flat_token_count, config.num_routed_experts],
        )?;

        let logits_host = matrix_rows(
            router_logits.values(),
            flat_token_count,
            config.num_routed_experts,
        )?;
        let topk_selections = select_topk(config, &logits_host, &self.correction_bias)?;
        let dispatch_plan = build_dispatch_plan(&topk_selections, config.num_routed_experts)?;

        Ok(Some(MoeRoutingDeviceTensors {
            normed_hidden_states,
            dispatch_plan,
        }))
    }
}

impl<'a> RouterProjection<'a> {
    fn open<B: Backend>(
        gguf: &'a GgufFile,
        tensor_ref: &TensorRef,
        input_features: usize,
        output_features: usize,
        _backend: &B,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        match tensor_ref.ty {
            GgmlType::F32 => {
                let info = gguf.tensor(&tensor_ref.name).ok_or_else(|| {
                    Error::gguf(format!("missing GLM-5.2 GGUF tensor {}", tensor_ref.name))
                })?;
                validate_tensor_ref(tensor_ref, info)?;
                let values = gguf.tensor_f32_values(&tensor_ref.name)?;
                let expected_values = input_features
                    .checked_mul(output_features)
                    .ok_or_else(|| Error::gguf("GGUF router F32 value count overflow"))?;
                validate_exact_shape(
                    "gguf_moe_router_f32_values",
                    &[values.len()],
                    &[expected_values],
                )?;
                let weight = F32Tensor::new(values, [output_features, input_features])?;
                Ok(Self::F32 {
                    weight,
                    input_features,
                    output_features,
                })
            }
            GgmlType::Q2K | GgmlType::Q8_0 => {
                Ok(Self::Quantized(QuantizedLinear::open(
                    gguf,
                    tensor_ref,
                    input_features,
                    output_features,
                    output_chunk_rows,
                )?))
            }
            other => Err(Error::gguf(format!(
                "gguf_moe_router_weight tensor {} must be F32, Q2_K, or Q8_0 for the GLM-5.2 Q2 runtime, got {other}",
                tensor_ref.name
            ))),
        }
    }

    fn forward<B: Backend>(&self, input: &Tensor, backend: &B) -> Result<RouterProjectionOutput> {
        match self {
            Self::Quantized(router) => {
                let output = router.forward(input, backend)?;
                Ok(RouterProjectionOutput {
                    output: output.output,
                    chunk_count: output.report.chunk_count,
                })
            }
            Self::F32 {
                weight,
                input_features,
                output_features,
            } => {
                let dims = input.dims();
                if dims.len() != 2 {
                    return Err(Error::model(format!(
                        "GGUF MoE router F32 input must be rank 2 [N,H], got {dims:?}"
                    )));
                }
                validate_exact_shape(
                    "gguf_moe_router_f32_input",
                    dims,
                    &[dims[0], *input_features],
                )?;
                let weight = tensor_from_f32_tensor(weight.clone(), backend.device())?;
                let output = backend.linear(input, &weight)?;
                validate_exact_shape(
                    "gguf_moe_router_f32_output",
                    output.dims(),
                    &[dims[0], *output_features],
                )?;
                Ok(RouterProjectionOutput {
                    output,
                    chunk_count: 1,
                })
            }
        }
    }

    fn forward_tensor<B: Backend>(&self, input: &Tensor, backend: &B) -> Result<Tensor> {
        match self {
            Self::Quantized(router) => router.forward_tensor(input, backend),
            Self::F32 {
                weight: _,
                input_features,
                output_features,
            } => {
                let dims = input.dims();
                if dims.len() != 2 {
                    return Err(Error::model(format!(
                        "GGUF MoE router F32 input must be rank 2 [N,H], got {dims:?}"
                    )));
                }
                validate_exact_shape(
                    "gguf_moe_router_f32_input",
                    dims,
                    &[dims[0], *input_features],
                )?;
                let input_f32 = tensor_to_f32_tensor(input)?;
                let output_f32 = self.forward_f32_tensor(&input_f32, backend)?;
                let output = tensor_from_f32_tensor(output_f32, backend.device())?;
                validate_exact_shape(
                    "gguf_moe_router_f32_output",
                    output.dims(),
                    &[dims[0], *output_features],
                )?;
                Ok(output)
            }
        }
    }

    fn forward_f32_tensor<B: Backend>(&self, input: &F32Tensor, backend: &B) -> Result<F32Tensor> {
        match self {
            Self::Quantized(router) => router.forward_f32_tensor(input, backend),
            Self::F32 {
                weight,
                input_features,
                output_features,
            } => {
                let dims = input.dims();
                if dims.len() != 2 {
                    return Err(Error::model(format!(
                        "GGUF native MoE router F32 input must be rank 2 [N,H], got {dims:?}"
                    )));
                }
                validate_exact_shape(
                    "gguf_native_moe_router_f32_input",
                    dims,
                    &[dims[0], *input_features],
                )?;
                if let Some(output) = backend.linear_f32_tensor(input, weight)? {
                    validate_exact_shape(
                        "gguf_native_moe_router_f32_output",
                        output.dims(),
                        &[dims[0], *output_features],
                    )?;
                    return Ok(output);
                }
                reference_linear_f32(input, weight, dims[0], *input_features, *output_features)
            }
        }
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

fn tensor_from_f32_tensor(tensor: F32Tensor, device: &common::Device) -> Result<Tensor> {
    let (shape, values) = tensor.into_parts();
    Ok(Tensor::from_vec(values, shape.dims(), device)?)
}

fn matrix_rows(values: &[f32], rows: usize, cols: usize) -> Result<Vec<Vec<f32>>> {
    let expected = rows
        .checked_mul(cols)
        .ok_or_else(|| Error::moe("router logits matrix value count overflow"))?;
    validate_exact_shape("router_logits_matrix_values", &[values.len()], &[expected])?;
    Ok(values
        .chunks_exact(cols)
        .map(|row| row.to_vec())
        .collect::<Vec<_>>())
}

fn reference_linear_f32(
    input: &F32Tensor,
    weight: &F32Tensor,
    rows: usize,
    in_features: usize,
    out_features: usize,
) -> Result<F32Tensor> {
    let mut output = vec![0.0_f32; rows * out_features];
    for row in 0..rows {
        for out in 0..out_features {
            let mut acc = 0.0_f32;
            for input_col in 0..in_features {
                acc += input.values()[row * in_features + input_col]
                    * weight.values()[out * in_features + input_col];
            }
            output[row * out_features + out] = acc;
        }
    }
    F32Tensor::new(output, [rows, out_features])
}

fn validate_sparse_layer(config: &Config, layer_index: usize) -> Result<()> {
    if layer_index >= config.num_layers {
        return Err(Error::moe(format!(
            "GLM-5.2 GGUF MoE router layer {layer_index} exceeds num_layers {}",
            config.num_layers
        )));
    }
    if layer_index < config.dense_layers {
        return Err(Error::moe(format!(
            "GLM-5.2 GGUF MoE router layer {layer_index} is dense; first sparse layer is {}",
            config.dense_layers
        )));
    }
    Ok(())
}

fn validate_supported_router_config(config: &Config) -> Result<()> {
    if config.moe_groups != 1 || config.topk_group != 1 {
        return Err(Error::moe(format!(
            "GLM MoE grouped routing is not implemented yet; got n_group={} topk_group={}",
            config.moe_groups, config.topk_group
        )));
    }
    if config.topk_method != "noaux_tc" {
        return Err(Error::moe(format!(
            "unsupported GLM-5.2 Q2 MoE topk_method {}; expected noaux_tc",
            config.topk_method
        )));
    }
    if config.scoring_func != "sigmoid" {
        return Err(Error::moe(format!(
            "unsupported GLM-5.2 Q2 MoE scoring_func {}; expected sigmoid",
            config.scoring_func
        )));
    }
    Ok(())
}

fn validate_router_weight(config: &Config, tensor_ref: &TensorRef) -> Result<()> {
    validate_linear_ref(
        "gguf_moe_router_weight",
        tensor_ref,
        config.hidden_size,
        config.num_routed_experts,
    )
}

fn validate_linear_ref(
    context: &str,
    tensor_ref: &TensorRef,
    expected_input_features: usize,
    expected_output_features: usize,
) -> Result<()> {
    match tensor_ref.ty {
        GgmlType::F32 | GgmlType::Q2K | GgmlType::Q8_0 => {}
        other => {
            return Err(Error::gguf(format!(
                "{context} tensor {} must be F32, Q2_K, or Q8_0 for the GLM-5.2 Q2 runtime, got {other}",
                tensor_ref.name
            )));
        }
    }
    let (input_features, output_features) = linear_dims(tensor_ref)?;
    validate_exact_shape(
        context,
        &[input_features, output_features],
        &[expected_input_features, expected_output_features],
    )
}

fn linear_dims(tensor_ref: &TensorRef) -> Result<(usize, usize)> {
    validate_exact_shape(
        "gguf_moe_router_linear_rank",
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

fn select_topk(
    config: &Config,
    logits: &[Vec<f32>],
    correction_bias: &[f32],
) -> Result<Vec<TopKSelection>> {
    if logits.is_empty() {
        return Err(Error::moe("GLM MoE router logits must not be empty"));
    }
    validate_exact_shape(
        "gguf_moe_router_correction_bias_host",
        &[correction_bias.len()],
        &[config.num_routed_experts],
    )?;
    let expert_count = config.num_routed_experts;
    let top_k = config.experts_per_token;
    if top_k > expert_count {
        return Err(Error::moe(format!(
            "experts_per_token {top_k} exceeds num_routed_experts {expert_count}"
        )));
    }

    let mut selections = Vec::with_capacity(logits.len());
    for (token_index, logit_row) in logits.iter().enumerate() {
        selections.push(select_topk_row(
            config,
            token_index,
            logit_row,
            correction_bias,
        )?);
    }

    Ok(selections)
}

fn select_topk_row(
    config: &Config,
    token_index: usize,
    logit_row: &[f32],
    correction_bias: &[f32],
) -> Result<TopKSelection> {
    let expert_count = config.num_routed_experts;
    let top_k = config.experts_per_token;
    validate_exact_shape(
        format!("gguf_moe_router_logit_row_{token_index}"),
        &[logit_row.len()],
        &[expert_count],
    )?;

    let mut top = Vec::<RouterTopKCandidate>::with_capacity(top_k);
    for (expert_id, logit) in logit_row.iter().copied().enumerate() {
        let score = sigmoid(logit);
        let corrected = score + correction_bias[expert_id];
        insert_topk_candidate(
            &mut top,
            top_k,
            RouterTopKCandidate {
                expert_id,
                corrected,
                score,
            },
        );
    }

    let expert_ids = top
        .iter()
        .map(|candidate| candidate.expert_id)
        .collect::<Vec<_>>();
    let mut weights = top
        .iter()
        .map(|candidate| candidate.score)
        .collect::<Vec<_>>();
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(Error::moe(format!(
            "GLM MoE selected score row {token_index} contains invalid weights"
        )));
    }
    if config.norm_topk_prob {
        let sum = weights.iter().sum::<f32>();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(Error::moe(format!(
                "GLM MoE selected score sum for token {token_index} must be positive"
            )));
        }
        for weight in &mut weights {
            *weight /= sum;
        }
    }
    for weight in &mut weights {
        *weight *= config.routed_scaling_factor as f32;
    }

    Ok(TopKSelection {
        token_index,
        expert_ids,
        weights,
    })
}

#[derive(Debug, Clone, Copy)]
struct RouterTopKCandidate {
    expert_id: usize,
    corrected: f32,
    score: f32,
}

fn insert_topk_candidate(
    top: &mut Vec<RouterTopKCandidate>,
    top_k: usize,
    candidate: RouterTopKCandidate,
) {
    if top_k == 0 {
        return;
    }

    let insert_at = top
        .iter()
        .position(|existing| is_better_topk_candidate(candidate, *existing))
        .unwrap_or(top.len());
    if insert_at < top_k {
        top.insert(insert_at, candidate);
        if top.len() > top_k {
            top.pop();
        }
    }
}

fn is_better_topk_candidate(left: RouterTopKCandidate, right: RouterTopKCandidate) -> bool {
    match left.corrected.total_cmp(&right.corrected) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => left.expert_id < right.expert_id,
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
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
        GgmlType, GgufFile, GgufMetadataValueType, GGML_Q2_K_BLOCK_BYTES, GGUF_MAGIC,
        GGUF_VERSION_V3,
    };

    use super::*;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn topk_scan_prefers_lower_expert_id_on_equal_scores() {
        let config = tiny_config();

        let selection =
            select_topk_row(&config, 0, &[0.5, 0.5, 0.1, 0.2], &[0.0, 0.0, 0.0, 0.0]).unwrap();

        assert_eq!(selection.expert_ids, vec![0, 1]);
        assert!((selection.weights[0] - 1.25).abs() <= 1e-6);
        assert!((selection.weights[1] - 1.25).abs() <= 1e-6);
    }

    #[test]
    fn q2_router_selects_topk_from_correction_bias_without_loading_experts() {
        let path = write_router_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let router = MoeRouter::open_from_parts(
            &gguf,
            &config,
            1,
            &tensor_ref(&gguf, "blk.1.ffn_norm.weight"),
            &tensor_ref(&gguf, "blk.1.ffn_gate_inp.weight"),
            &tensor_ref(&gguf, "blk.1.exp_probs_b.bias"),
            &backend,
            3,
        )
        .unwrap();
        let hidden_states =
            Tensor::from_vec(vec![0.0_f32; 256], (1, 1, 256), &Device::Cpu).unwrap();

        let output = router.route(&config, &hidden_states, &backend).unwrap();

        assert_eq!(router.load_report().router_tensor_type, GgmlType::Q2K);
        assert_eq!(output.report.router_logits_shape.dims(), &[1, 4]);
        assert_eq!(output.report.router_logits_chunk_count, 2);
        assert_eq!(output.report.dispatch_expert_count, 2);
        assert!(output
            .dispatch_plan
            .iter()
            .any(|dispatch| dispatch.expert_id == 3));
        assert!(output
            .dispatch_plan
            .iter()
            .any(|dispatch| dispatch.expert_id == 1));
    }

    #[test]
    fn f32_router_selects_topk_from_real_router_weight_layout() {
        let path = write_router_fixture(GgmlType::F32);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let router = MoeRouter::open_from_parts(
            &gguf,
            &config,
            1,
            &tensor_ref(&gguf, "blk.1.ffn_norm.weight"),
            &tensor_ref(&gguf, "blk.1.ffn_gate_inp.weight"),
            &tensor_ref(&gguf, "blk.1.exp_probs_b.bias"),
            &backend,
            2,
        )
        .unwrap();
        let hidden_states =
            Tensor::from_vec(vec![0.0_f32; 256], (1, 1, 256), &Device::Cpu).unwrap();

        let output = router.route(&config, &hidden_states, &backend).unwrap();

        assert_eq!(router.load_report().router_tensor_type, GgmlType::F32);
        assert_eq!(router.load_report().router_weight_shape.dims(), &[4, 256]);
        assert_eq!(output.report.router_logits_shape.dims(), &[1, 4]);
        assert_eq!(output.report.router_logits_chunk_count, 1);
        assert_eq!(output.report.dispatch_expert_count, 2);
        assert!(output
            .dispatch_plan
            .iter()
            .any(|dispatch| dispatch.expert_id == 3));
        assert!(output
            .dispatch_plan
            .iter()
            .any(|dispatch| dispatch.expert_id == 1));
    }

    #[test]
    fn rejects_dense_layer_index() {
        let path = write_router_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();

        let err = MoeRouter::open_from_parts(
            &gguf,
            &config,
            0,
            &tensor_ref(&gguf, "blk.1.ffn_norm.weight"),
            &tensor_ref(&gguf, "blk.1.ffn_gate_inp.weight"),
            &tensor_ref(&gguf, "blk.1.exp_probs_b.bias"),
            &backend,
            2,
        )
        .expect_err("dense layer index should fail");

        assert!(err.to_string().contains("is dense"));
    }

    #[test]
    fn rejects_wrong_correction_bias_shape() {
        let path = write_router_fixture_with_bias(GgmlType::Q2K, &[0.0, 0.1, 0.2]);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();

        let err = MoeRouter::open_from_parts(
            &gguf,
            &config,
            1,
            &tensor_ref(&gguf, "blk.1.ffn_norm.weight"),
            &tensor_ref(&gguf, "blk.1.ffn_gate_inp.weight"),
            &tensor_ref(&gguf, "blk.1.exp_probs_b.bias"),
            &backend,
            2,
        )
        .expect_err("wrong correction bias shape should fail");

        assert!(err.to_string().contains("gguf_moe_router_correction_bias"));
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
            num_layers: 2,
            dense_layers: 1,
            sparse_moe_layers: Some(1),
            vocab_size: 8,
            attention_heads: 1,
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

    fn write_router_fixture(ty: GgmlType) -> PathBuf {
        write_router_fixture_with_bias(ty, &[0.0, 0.4, 0.2, 0.8])
    }

    fn write_router_fixture_with_bias(ty: GgmlType, bias: &[f32]) -> PathBuf {
        let path = unique_temp_file("moe-router");
        let specs = vec![
            TensorSpec::f32("blk.1.ffn_norm.weight", vec![256], None),
            TensorSpec::quant("blk.1.ffn_gate_inp.weight", vec![256, 4], ty),
            TensorSpec::f32(
                "blk.1.exp_probs_b.bias",
                vec![bias.len() as u64],
                Some(bias.to_vec()),
            ),
        ];
        write_gguf(path, &specs)
    }

    #[derive(Debug)]
    struct TensorSpec {
        name: &'static str,
        dims: Vec<u64>,
        ty: GgmlType,
        values: Option<Vec<f32>>,
    }

    impl TensorSpec {
        fn f32(name: &'static str, dims: Vec<u64>, values: Option<Vec<f32>>) -> Self {
            Self {
                name,
                dims,
                ty: GgmlType::F32,
                values,
            }
        }

        fn quant(name: &'static str, dims: Vec<u64>, ty: GgmlType) -> Self {
            Self {
                name,
                dims,
                ty,
                values: None,
            }
        }

        fn payload_len(&self) -> u64 {
            match self.ty {
                GgmlType::F32 => self.dims.iter().product::<u64>() * 4,
                GgmlType::Q2K => self.dims.iter().product::<u64>() / 256 * GGML_Q2_K_BLOCK_BYTES,
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
