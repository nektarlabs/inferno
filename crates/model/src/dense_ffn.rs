use backend::{Backend, BackendCapabilities};
use common::Tensor;
use common::{validate_exact_shape, Error, F32Tensor, Result, Shape};
use config::Config;
use gguf::{GgmlType, GgufFile};

use crate::{
    DenseFfnIndex, FfnIndex, LayerIndex, QuantizedLinear, RmsNorm, RmsNormLoadReport, TensorRef,
};

#[derive(Debug)]
pub struct DenseFfn<'a> {
    layer_index: usize,
    post_attention_norm: RmsNorm,
    gate: QuantizedLinear<'a>,
    up: QuantizedLinear<'a>,
    down: QuantizedLinear<'a>,
    intermediate_size: usize,
    load_report: DenseFfnLoadReport,
}

#[derive(Debug)]
pub struct DenseFfnOutput {
    pub hidden_states: Tensor,
    pub report: DenseFfnForwardReport,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseFfnLoadReport {
    pub backend: BackendCapabilities,
    pub layer_index: usize,
    pub post_attention_norm: RmsNormLoadReport,
    pub gate_weight_shape: Shape,
    pub up_weight_shape: Shape,
    pub down_weight_shape: Shape,
    pub intermediate_size: usize,
    pub projection_tensor_type: GgmlType,
    pub output_chunk_rows: usize,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseFfnForwardReport {
    pub layer_index: usize,
    pub input_hidden_states_shape: Shape,
    pub normed_hidden_states_shape: Shape,
    pub gate_shape: Shape,
    pub gate_chunk_count: usize,
    pub up_shape: Shape,
    pub up_chunk_count: usize,
    pub activated_gate_shape: Shape,
    pub gated_shape: Shape,
    pub down_shape: Shape,
    pub down_chunk_count: usize,
    pub output_hidden_states_shape: Shape,
}

impl<'a> DenseFfn<'a> {
    pub fn open<B: Backend>(
        gguf: &'a GgufFile,
        config: &Config,
        layer: &LayerIndex,
        backend: &B,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        let FfnIndex::Dense(ffn) = &layer.ffn else {
            return Err(Error::gguf(format!(
                "GLM-5.2 GGUF layer {} is sparse MoE, not dense FFN",
                layer.layer_index
            )));
        };
        Self::open_from_parts(
            gguf,
            config,
            layer.layer_index,
            &layer.post_attention_norm,
            ffn,
            backend,
            output_chunk_rows,
        )
    }

    pub fn open_from_parts<B: Backend>(
        gguf: &'a GgufFile,
        config: &Config,
        layer_index: usize,
        post_attention_norm_ref: &TensorRef,
        ffn: &DenseFfnIndex,
        backend: &B,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        validate_dense_layer(config, layer_index)?;
        let intermediate_size = validate_gate_or_up(config, "gguf_dense_ffn_gate", &ffn.gate)?;
        let up_intermediate_size = validate_gate_or_up(config, "gguf_dense_ffn_up", &ffn.up)?;
        validate_exact_shape(
            "gguf_dense_ffn_gate_up_match",
            &[up_intermediate_size],
            &[intermediate_size],
        )?;
        validate_down(config, intermediate_size, &ffn.down)?;

        let post_attention_norm = RmsNorm::open(gguf, config, post_attention_norm_ref, backend)?;
        let gate = QuantizedLinear::open(
            gguf,
            &ffn.gate,
            config.hidden_size,
            intermediate_size,
            output_chunk_rows,
        )?;
        let up = QuantizedLinear::open(
            gguf,
            &ffn.up,
            config.hidden_size,
            intermediate_size,
            output_chunk_rows,
        )?;
        let down = QuantizedLinear::open(
            gguf,
            &ffn.down,
            intermediate_size,
            config.hidden_size,
            output_chunk_rows,
        )?;

        let load_report = DenseFfnLoadReport {
            backend: backend.capabilities(),
            layer_index,
            post_attention_norm: post_attention_norm.load_report().clone(),
            gate_weight_shape: logical_weight_shape(&ffn.gate)?,
            up_weight_shape: logical_weight_shape(&ffn.up)?,
            down_weight_shape: logical_weight_shape(&ffn.down)?,
            intermediate_size,
            projection_tensor_type: ffn.gate.ty,
            output_chunk_rows,
            limitations: vec![
                "GGUF dense FFN executes native backend SwiGLU with chunked Q2_K/Q8_0 projection decoding"
                    .to_string(),
            ],
        };

        Ok(Self {
            layer_index,
            post_attention_norm,
            gate,
            up,
            down,
            intermediate_size,
            load_report,
        })
    }

    pub fn load_report(&self) -> &DenseFfnLoadReport {
        &self.load_report
    }

    pub fn forward<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<DenseFfnOutput> {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF dense FFN input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            "gguf_dense_ffn_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;

        let normed = self.post_attention_norm.forward(hidden_states, backend)?;
        let gate = self.gate.forward(&normed.hidden_states, backend)?;
        let up = self.up.forward(&normed.hidden_states, backend)?;
        let activated_gate_shape = Shape::new(gate.output.dims().to_vec());
        let gated = backend.swiglu(&gate.output, &up.output)?;
        validate_exact_shape(
            "gguf_dense_ffn_gated",
            gated.dims(),
            &[batch, tokens, self.intermediate_size],
        )?;
        let down = self.down.forward(&gated, backend)?;
        let output_hidden_states = backend.add(hidden_states, &down.output)?;
        let output_hidden_states_shape = Shape::new(output_hidden_states.dims().to_vec());

        Ok(DenseFfnOutput {
            hidden_states: output_hidden_states,
            report: DenseFfnForwardReport {
                layer_index: self.layer_index,
                input_hidden_states_shape: Shape::new(hidden_states.dims().to_vec()),
                normed_hidden_states_shape: Shape::new(normed.hidden_states.dims().to_vec()),
                gate_shape: Shape::new(gate.output.dims().to_vec()),
                gate_chunk_count: gate.report.chunk_count,
                up_shape: Shape::new(up.output.dims().to_vec()),
                up_chunk_count: up.report.chunk_count,
                activated_gate_shape,
                gated_shape: Shape::new(gated.dims().to_vec()),
                down_shape: Shape::new(down.output.dims().to_vec()),
                down_chunk_count: down.report.chunk_count,
                output_hidden_states_shape,
            },
        })
    }

    pub fn forward_tensors<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<Tensor> {
        if backend.capabilities().custom_kernels {
            return self.forward_tensors_native(config, hidden_states, backend);
        }

        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF dense FFN input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            "gguf_dense_ffn_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;

        let normed = self
            .post_attention_norm
            .forward_tensor(hidden_states, backend)?;
        let gate = self.gate.forward_tensor(&normed, backend)?;
        let up = self.up.forward_tensor(&normed, backend)?;
        let gated = backend.swiglu(&gate, &up)?;
        validate_exact_shape(
            "gguf_dense_ffn_gated",
            gated.dims(),
            &[batch, tokens, self.intermediate_size],
        )?;
        let down = self.down.forward_tensor(&gated, backend)?;
        backend.add(hidden_states, &down)
    }

    fn forward_tensors_native<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<Tensor> {
        let hidden_states = tensor_to_f32_tensor(hidden_states)?;
        let output = self.forward_f32_tensor(config, &hidden_states, backend)?;
        tensor_from_f32_tensor(output, backend.device())
    }

    pub fn forward_f32_tensor<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        backend: &B,
    ) -> Result<F32Tensor> {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF native dense FFN input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            "gguf_native_dense_ffn_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;

        let normed = self
            .post_attention_norm
            .forward_f32(&hidden_states, backend)?
            .hidden_states;
        let gate = self.gate.forward_f32_tensor(&normed, backend)?;
        let up = self.up.forward_f32_tensor(&normed, backend)?;
        let gated = require_native("SwiGLU", backend.swiglu_f32_tensor(&gate, &up)?)?;
        validate_exact_shape(
            "gguf_native_dense_ffn_gated",
            gated.dims(),
            &[batch, tokens, self.intermediate_size],
        )?;
        let down = self.down.forward_f32_tensor(&gated, backend)?;
        require_native("add", backend.add_f32_tensor(hidden_states, &down)?)
    }

    /// Batched device-resident variant of `forward_f32_tensor`: the whole
    /// dense FFN (norm, gate/up, SwiGLU, down, residual) is encoded into the
    /// backend's open batch with no synchronization. `Ok(None)` falls back to
    /// the eager path.
    pub(crate) fn forward_device<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &backend::DeviceValue,
        backend: &B,
    ) -> Result<Option<backend::DeviceValue>> {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF device dense FFN input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            "gguf_device_dense_ffn_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;

        let normed = require_dense_ffn_device_stage(
            "dense_ffn.post_attention_norm",
            self.post_attention_norm
                .forward_device(hidden_states, backend),
        )?;
        let gate = require_dense_ffn_device_stage(
            "dense_ffn.gate",
            self.gate.forward_device(&normed, backend),
        )?;
        let up = require_dense_ffn_device_stage(
            "dense_ffn.up",
            self.up.forward_device(&normed, backend),
        )?;
        let gated =
            require_dense_ffn_device_stage("dense_ffn.swiglu", backend.swiglu_device(&gate, &up))?;
        validate_exact_shape(
            "gguf_device_dense_ffn_gated",
            gated.dims(),
            &[batch, tokens, self.intermediate_size],
        )?;
        let down = require_dense_ffn_device_stage(
            "dense_ffn.down",
            self.down.forward_device(&gated, backend),
        )?;
        let output = require_dense_ffn_device_stage(
            "dense_ffn.residual_add",
            backend.add_device(hidden_states, &down),
        )?;
        Ok(Some(output))
    }
}

fn require_dense_ffn_device_stage<T>(stage: &str, result: Result<Option<T>>) -> Result<T> {
    result?.ok_or_else(|| Error::backend(format!("{stage} has no native device path")))
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
            "native Metal {operation} is required for the GLM-5.2 Q2 dense FFN path"
        ))
    })
}

fn validate_dense_layer(config: &Config, layer_index: usize) -> Result<()> {
    if layer_index >= config.num_layers {
        return Err(Error::gguf(format!(
            "GLM-5.2 GGUF dense FFN layer {layer_index} exceeds num_layers {}",
            config.num_layers
        )));
    }
    if layer_index >= config.dense_layers {
        return Err(Error::gguf(format!(
            "GLM-5.2 GGUF layer {layer_index} is sparse; dense layers end before {}",
            config.dense_layers
        )));
    }
    Ok(())
}

fn validate_gate_or_up(config: &Config, context: &str, tensor_ref: &TensorRef) -> Result<usize> {
    let (input_features, output_features) = linear_dims(tensor_ref)?;
    validate_exact_shape(context, &[input_features], &[config.hidden_size])?;
    if output_features == 0 {
        return Err(Error::gguf(format!(
            "{context} intermediate size must be positive"
        )));
    }
    Ok(output_features)
}

fn validate_down(config: &Config, intermediate_size: usize, tensor_ref: &TensorRef) -> Result<()> {
    let (input_features, output_features) = linear_dims(tensor_ref)?;
    validate_exact_shape(
        "gguf_dense_ffn_down",
        &[input_features, output_features],
        &[intermediate_size, config.hidden_size],
    )
}

fn linear_dims(tensor_ref: &TensorRef) -> Result<(usize, usize)> {
    match tensor_ref.ty {
        GgmlType::Q2K | GgmlType::Q8_0 => {}
        other => {
            return Err(Error::gguf(format!(
                "GLM-5.2 Q2 GGUF dense FFN tensor {} must be Q2_K or Q8_0, got {other}",
                tensor_ref.name
            )));
        }
    }
    validate_exact_shape("gguf_dense_ffn_linear_rank", &[tensor_ref.dims.len()], &[2])?;
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
    fn q2_dense_ffn_runs_swiglu_in_chunks() {
        let path = write_dense_ffn_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let post_norm = tensor_ref(&gguf, "blk.0.ffn_norm.weight");
        let ffn_index = dense_ffn_index(&gguf);
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let ffn =
            DenseFfn::open_from_parts(&gguf, &config, 0, &post_norm, &ffn_index, &backend, 128)
                .unwrap();
        let hidden_states =
            Tensor::from_vec(vec![1.0_f32; 256], (1, 1, 256), &Device::Cpu).unwrap();

        let output = ffn.forward(&config, &hidden_states, &backend).unwrap();

        assert_eq!(ffn.load_report().projection_tensor_type, GgmlType::Q2K);
        assert_eq!(output.hidden_states.dims(), &[1, 1, 256]);
        assert_eq!(output.report.gate_chunk_count, 4);
        assert_eq!(output.report.up_chunk_count, 4);
        assert_eq!(output.report.down_chunk_count, 2);
    }

    #[test]
    fn q2_dense_ffn_tensor_path_matches_reference() {
        let path = write_dense_ffn_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let post_norm = tensor_ref(&gguf, "blk.0.ffn_norm.weight");
        let ffn_index = dense_ffn_index(&gguf);
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let ffn =
            DenseFfn::open_from_parts(&gguf, &config, 0, &post_norm, &ffn_index, &backend, 128)
                .unwrap();
        let hidden_states =
            Tensor::from_vec(vec![1.0_f32; 256], (1, 1, 256), &Device::Cpu).unwrap();

        let reference = ffn
            .forward(&config, &hidden_states, &backend)
            .unwrap()
            .hidden_states;
        let optimized = ffn
            .forward_tensors(&config, &hidden_states, &backend)
            .unwrap();

        assert_eq!(optimized.dims(), &[1, 1, 256]);
        let reference = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let optimized = optimized.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (reference_value, optimized_value) in reference.iter().zip(optimized.iter()) {
            assert!((reference_value - optimized_value).abs() <= 1e-4);
        }
    }

    #[test]
    fn rejects_sparse_layer_index() {
        let path = write_dense_ffn_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let mut config = tiny_config();
        config.dense_layers = 0;
        let post_norm = tensor_ref(&gguf, "blk.0.ffn_norm.weight");
        let ffn_index = dense_ffn_index(&gguf);
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();

        let err =
            DenseFfn::open_from_parts(&gguf, &config, 0, &post_norm, &ffn_index, &backend, 256)
                .expect_err("sparse layer index should reject dense FFN");

        assert!(err.to_string().contains("is sparse"));
    }

    #[test]
    fn rejects_wrong_down_shape() {
        let path = write_dense_ffn_fixture_with_down_dims(GgmlType::Q2K, &[256, 256]);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let post_norm = tensor_ref(&gguf, "blk.0.ffn_norm.weight");
        let ffn_index = dense_ffn_index(&gguf);
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();

        let err =
            DenseFfn::open_from_parts(&gguf, &config, 0, &post_norm, &ffn_index, &backend, 256)
                .expect_err("wrong down shape should fail");

        assert!(err.to_string().contains("gguf_dense_ffn_down"));
    }

    fn dense_ffn_index(gguf: &GgufFile) -> DenseFfnIndex {
        DenseFfnIndex {
            gate: tensor_ref(gguf, "blk.0.ffn_gate.weight"),
            up: tensor_ref(gguf, "blk.0.ffn_up.weight"),
            down: tensor_ref(gguf, "blk.0.ffn_down.weight"),
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
            attention_heads: 1,
            qk_head_dim: 256,
            qk_no_rope_dim: 128,
            qk_rope_dim: 128,
            kv_lora_rank: 256,
            v_head_dim: Some(256),
            num_routed_experts: 1,
            experts_per_token: 1,
            moe_intermediate_size: 512,
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
            index_skip_topk_offset: 3,
            index_share_for_mtp_iteration: true,
            indexer_rope_interleave: true,
            indexer_types: Vec::new(),
            num_nextn_predict_layers: 0,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000_000.0,
        }
    }

    fn write_dense_ffn_fixture(ty: GgmlType) -> PathBuf {
        write_dense_ffn_fixture_with_down_dims(ty, &[512, 256])
    }

    fn write_dense_ffn_fixture_with_down_dims(ty: GgmlType, down_dims: &[u64]) -> PathBuf {
        let path = unique_temp_file("dense-ffn");
        let specs = vec![
            TensorSpec::f32("blk.0.ffn_norm.weight", vec![256]),
            TensorSpec::quant("blk.0.ffn_gate.weight", vec![256, 512], ty),
            TensorSpec::quant("blk.0.ffn_up.weight", vec![256, 512], ty),
            TensorSpec::quant("blk.0.ffn_down.weight", down_dims.to_vec(), ty),
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
