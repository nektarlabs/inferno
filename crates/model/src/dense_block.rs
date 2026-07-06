use backend::{Backend, BackendCapabilities};
use common::Tensor;
use common::{validate_exact_shape, Error, F32Tensor, PagedKvView, Result, Shape};
use config::Config;
use gguf::GgufFile;

use crate::{
    profile, Attention, AttentionLoadReport, AttentionOutput, DenseFfn, DenseFfnLoadReport,
    DenseFfnOutput, FfnIndex, LayerIndex, LayerKind,
};

#[derive(Debug)]
pub struct DenseBlock<'a> {
    attention: Attention<'a>,
    ffn: DenseFfn<'a>,
    load_report: DenseBlockLoadReport,
}

#[derive(Debug)]
pub struct DenseBlockOutput {
    pub hidden_states: Tensor,
    pub cache_k: Tensor,
    pub cache_v: Tensor,
    pub report: DenseBlockForwardReport,
}

#[derive(Debug)]
pub struct DenseBlockTensors {
    pub hidden_states: Tensor,
    pub cache_k: F32Tensor,
    pub cache_v: F32Tensor,
}

#[derive(Debug)]
pub struct DenseBlockF32Tensors {
    pub hidden_states: F32Tensor,
    pub cache_k: F32Tensor,
    pub cache_v: F32Tensor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseBlockLoadReport {
    pub backend: BackendCapabilities,
    pub layer_index: usize,
    pub attention: AttentionLoadReport,
    pub ffn: DenseFfnLoadReport,
    pub output_chunk_rows: usize,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DenseBlockForwardReport {
    pub layer_index: usize,
    pub input_hidden_states_shape: Shape,
    pub attention_output_hidden_states_shape: Shape,
    pub cache_k_shape: Shape,
    pub cache_v_shape: Shape,
    pub attention_scores_shape: Shape,
    pub attention_past_tokens: usize,
    pub ffn_output_hidden_states_shape: Shape,
    pub ffn_gate_output_shape: Shape,
    pub output_hidden_states_shape: Shape,
}

impl<'a> DenseBlock<'a> {
    pub fn open<B: Backend>(
        gguf: &'a GgufFile,
        config: &Config,
        layer: &LayerIndex,
        backend: &B,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        validate_dense_layer(config, layer)?;
        let attention = Attention::open(gguf, config, layer, backend, output_chunk_rows)?;
        let ffn = DenseFfn::open(gguf, config, layer, backend, output_chunk_rows)?;
        let load_report = DenseBlockLoadReport {
            backend: backend.capabilities(),
            layer_index: layer.layer_index,
            attention: attention.load_report().clone(),
            ffn: ffn.load_report().clone(),
            output_chunk_rows,
            limitations: vec![
                "GGUF dense block composes GLM attention and dense SwiGLU FFN".to_string(),
                "K/V cache data is expanded per head for the current reference path".to_string(),
                "generation uses native Metal Q2_K kernels for attention and dense FFN projections"
                    .to_string(),
                "paged attention and DSA sparse attention are not fused in this block yet"
                    .to_string(),
            ],
        };

        Ok(Self {
            attention,
            ffn,
            load_report,
        })
    }

    pub fn load_report(&self) -> &DenseBlockLoadReport {
        &self.load_report
    }

    pub fn forward<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<DenseBlockOutput> {
        self.forward_with_past_kv(config, hidden_states, backend, None)
    }

    pub fn forward_with_past_kv<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
        past_kv: Option<(&Tensor, &Tensor)>,
    ) -> Result<DenseBlockOutput> {
        validate_hidden_states(config, hidden_states)?;
        let attention_output =
            self.attention
                .forward_with_past_kv(config, hidden_states, backend, past_kv)?;
        self.forward_ffn(config, hidden_states, attention_output, backend)
    }

    pub fn forward_tensors<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<DenseBlockTensors> {
        self.forward_tensors_with_past_kv(config, hidden_states, backend, None)
    }

    pub fn forward_tensors_with_past_kv<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
        past_kv: Option<(&F32Tensor, &F32Tensor)>,
    ) -> Result<DenseBlockTensors> {
        if backend.capabilities().custom_kernels {
            return self.forward_tensors_native_with_past_kv(
                config,
                hidden_states,
                backend,
                past_kv,
            );
        }

        validate_hidden_states(config, hidden_states)?;
        let attention_output =
            profile::run_layer_stage(self.load_report.layer_index, "dense.attention", || {
                self.attention
                    .forward_tensors_with_past_kv(config, hidden_states, backend, past_kv)
            })?;
        let hidden_states =
            profile::run_layer_stage(self.load_report.layer_index, "dense.ffn", || {
                self.ffn
                    .forward_tensors(config, &attention_output.hidden_states, backend)
            })?;

        Ok(DenseBlockTensors {
            hidden_states,
            cache_k: attention_output.cache_k,
            cache_v: attention_output.cache_v,
        })
    }

    fn forward_tensors_native_with_past_kv<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
        past_kv: Option<(&F32Tensor, &F32Tensor)>,
    ) -> Result<DenseBlockTensors> {
        validate_hidden_states(config, hidden_states)?;
        let hidden_states = tensor_to_f32_tensor(hidden_states)?;
        let output =
            self.forward_f32_tensors_with_past_kv(config, &hidden_states, backend, past_kv)?;

        Ok(DenseBlockTensors {
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
    ) -> Result<DenseBlockF32Tensors> {
        validate_hidden_states_f32(config, hidden_states)?;
        let attention_output =
            profile::run_layer_stage(self.load_report.layer_index, "dense.attention", || {
                self.attention.forward_f32_tensors_with_past_kv(
                    config,
                    hidden_states,
                    backend,
                    past_kv,
                )
            })?;
        let output_hidden_states =
            profile::run_layer_stage(self.load_report.layer_index, "dense.ffn", || {
                self.ffn
                    .forward_f32_tensor(config, &attention_output.hidden_states, backend)
            })?;

        Ok(DenseBlockF32Tensors {
            hidden_states: output_hidden_states,
            cache_k: attention_output.cache_k,
            cache_v: attention_output.cache_v,
        })
    }

    pub fn forward_f32_tensors_with_paged_past_kv<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        backend: &B,
        past_kv: Option<PagedKvView<'_>>,
    ) -> Result<DenseBlockF32Tensors> {
        validate_hidden_states_f32(config, hidden_states)?;
        let attention_output =
            profile::run_layer_stage(self.load_report.layer_index, "dense.attention", || {
                self.attention.forward_f32_tensors_with_paged_past_kv(
                    config,
                    hidden_states,
                    backend,
                    past_kv,
                )
            })?;
        let output_hidden_states =
            profile::run_layer_stage(self.load_report.layer_index, "dense.ffn", || {
                self.ffn
                    .forward_f32_tensor(config, &attention_output.hidden_states, backend)
            })?;

        Ok(DenseBlockF32Tensors {
            hidden_states: output_hidden_states,
            cache_k: attention_output.cache_k,
            cache_v: attention_output.cache_v,
        })
    }

    /// Batched device-resident decode layer: attention and dense FFN encoded
    /// into the backend's open batch, hidden states never leaving the GPU.
    pub(crate) fn forward_decode_device<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &backend::DeviceValue,
        backend: &B,
        past_kv: &PagedKvView<'_>,
    ) -> Result<Option<crate::kv_types::BlockDeviceTensors>> {
        let attention_output = crate::try_device!(profile::run_layer_stage(
            self.load_report.layer_index,
            "dense.attention",
            || self
                .attention
                .forward_decode_device(config, hidden_states, backend, past_kv),
        ));
        let output_hidden_states = crate::try_device!(profile::run_layer_stage(
            self.load_report.layer_index,
            "dense.ffn",
            || self
                .ffn
                .forward_device(config, &attention_output.hidden_states, backend),
        ));

        Ok(Some(crate::kv_types::BlockDeviceTensors {
            hidden_states: output_hidden_states,
            cache_k: attention_output.cache_k,
            cache_v: attention_output.cache_v,
        }))
    }

    fn forward_ffn<B: Backend>(
        &self,
        config: &Config,
        input_hidden_states: &Tensor,
        attention_output: AttentionOutput,
        backend: &B,
    ) -> Result<DenseBlockOutput> {
        let ffn_output = self
            .ffn
            .forward(config, &attention_output.hidden_states, backend)?;
        block_output(input_hidden_states, attention_output, ffn_output)
    }
}

fn block_output(
    input_hidden_states: &Tensor,
    attention_output: AttentionOutput,
    ffn_output: DenseFfnOutput,
) -> Result<DenseBlockOutput> {
    let attention_report = attention_output.report;
    let ffn_report = ffn_output.report;
    let report = DenseBlockForwardReport {
        layer_index: attention_report.layer_index,
        input_hidden_states_shape: Shape::new(input_hidden_states.dims().to_vec()),
        attention_output_hidden_states_shape: Shape::new(
            attention_output.hidden_states.dims().to_vec(),
        ),
        cache_k_shape: Shape::new(attention_output.cache_k.dims().to_vec()),
        cache_v_shape: Shape::new(attention_output.cache_v.dims().to_vec()),
        attention_scores_shape: attention_report.attention_scores_shape,
        attention_past_tokens: attention_report.past_tokens,
        ffn_output_hidden_states_shape: Shape::new(ffn_output.hidden_states.dims().to_vec()),
        ffn_gate_output_shape: ffn_report.gate_shape,
        output_hidden_states_shape: Shape::new(ffn_output.hidden_states.dims().to_vec()),
    };

    Ok(DenseBlockOutput {
        hidden_states: ffn_output.hidden_states,
        cache_k: attention_output.cache_k,
        cache_v: attention_output.cache_v,
        report,
    })
}

fn validate_dense_layer(config: &Config, layer: &LayerIndex) -> Result<()> {
    if layer.layer_index >= config.num_layers {
        return Err(Error::gguf(format!(
            "GLM-5.2 GGUF dense block layer {} exceeds num_layers {}",
            layer.layer_index, config.num_layers
        )));
    }
    if layer.kind != LayerKind::Dense {
        return Err(Error::gguf(format!(
            "GLM-5.2 GGUF layer {} is {:?}, not dense",
            layer.layer_index, layer.kind
        )));
    }
    if !matches!(layer.ffn, FfnIndex::Dense(_)) {
        return Err(Error::gguf(format!(
            "GLM-5.2 GGUF layer {} has non-dense FFN index",
            layer.layer_index
        )));
    }
    Ok(())
}

fn validate_hidden_states(config: &Config, hidden_states: &Tensor) -> Result<()> {
    let dims = hidden_states.dims();
    if dims.len() != 3 {
        return Err(Error::model(format!(
            "GLM-5.2 GGUF dense block input must be rank 3 [B,T,H], got {dims:?}"
        )));
    }
    validate_exact_shape(
        "gguf_dense_block_hidden_states",
        dims,
        &[dims[0], dims[1], config.hidden_size],
    )
}

fn validate_hidden_states_f32(config: &Config, hidden_states: &F32Tensor) -> Result<()> {
    let dims = hidden_states.dims();
    if dims.len() != 3 {
        return Err(Error::model(format!(
            "GLM-5.2 GGUF native dense block input must be rank 3 [B,T,H], got {dims:?}"
        )));
    }
    validate_exact_shape(
        "gguf_native_dense_block_hidden_states",
        dims,
        &[dims[0], dims[1], config.hidden_size],
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
    use crate::Index;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn q2_dense_block_runs_prefill_attention_then_ffn() {
        let path = write_dense_block_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let index = Index::from_gguf(&gguf, &config).unwrap();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let block = DenseBlock::open(&gguf, &config, &index.layers[0], &backend, 256).unwrap();
        let hidden_states =
            Tensor::from_vec(vec![1.0_f32; 2 * 3 * 256], (2, 3, 256), &Device::Cpu).unwrap();

        let output = block.forward(&config, &hidden_states, &backend).unwrap();
        let tensors = block
            .forward_tensors(&config, &hidden_states, &backend)
            .unwrap();

        assert_eq!(block.load_report().layer_index, 0);
        assert_eq!(block.load_report().attention.q_lora_rank, 256);
        assert_eq!(block.load_report().ffn.intermediate_size, 512);
        assert_eq!(output.hidden_states.dims(), &[2, 3, 256]);
        assert_eq!(tensors.hidden_states.dims(), &[2, 3, 256]);
        assert_eq!(output.cache_k.dims(), &[2, 2, 3, 256]);
        assert_eq!(output.cache_v.dims(), &[2, 2, 3, 256]);
        assert_eq!(tensors.cache_k.dims(), &[2, 2, 3, 256]);
        assert_eq!(tensors.cache_v.dims(), &[2, 2, 3, 256]);
        assert_eq!(
            output.report.attention_output_hidden_states_shape.dims(),
            &[2, 3, 256]
        );
        assert_eq!(
            output.report.ffn_output_hidden_states_shape.dims(),
            &[2, 3, 256]
        );
        assert_eq!(output.report.attention_scores_shape.dims(), &[2, 2, 3, 3]);
        assert_eq!(output.report.ffn_gate_output_shape.dims(), &[2, 3, 512]);
    }

    #[test]
    fn q2_dense_block_runs_decode_with_past_kv() {
        let path = write_dense_block_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let index = Index::from_gguf(&gguf, &config).unwrap();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let block = DenseBlock::open(&gguf, &config, &index.layers[0], &backend, 256).unwrap();
        let prefill_hidden =
            Tensor::from_vec(vec![1.0_f32; 3 * 256], (1, 3, 256), &Device::Cpu).unwrap();
        let prefill = block.forward(&config, &prefill_hidden, &backend).unwrap();
        let prefill_tensors = block
            .forward_tensors(&config, &prefill_hidden, &backend)
            .unwrap();
        let decode_hidden =
            Tensor::from_vec(vec![1.0_f32; 256], (1, 1, 256), &Device::Cpu).unwrap();

        let decode = block
            .forward_with_past_kv(
                &config,
                &decode_hidden,
                &backend,
                Some((&prefill.cache_k, &prefill.cache_v)),
            )
            .unwrap();
        let decode_tensors = block
            .forward_tensors_with_past_kv(
                &config,
                &decode_hidden,
                &backend,
                Some((&prefill_tensors.cache_k, &prefill_tensors.cache_v)),
            )
            .unwrap();

        assert_eq!(
            block.load_report().attention.projection_tensor_type,
            GgmlType::Q2K
        );
        assert_eq!(decode.hidden_states.dims(), &[1, 1, 256]);
        assert_eq!(decode_tensors.hidden_states.dims(), &[1, 1, 256]);
        assert_eq!(decode.cache_k.dims(), &[1, 2, 1, 256]);
        assert_eq!(decode.cache_v.dims(), &[1, 2, 1, 256]);
        assert_eq!(decode_tensors.cache_k.dims(), &[1, 2, 1, 256]);
        assert_eq!(decode_tensors.cache_v.dims(), &[1, 2, 1, 256]);
        assert_eq!(decode.report.attention_past_tokens, 3);
        assert_eq!(decode.report.attention_scores_shape.dims(), &[1, 2, 1, 4]);
    }

    #[test]
    fn rejects_sparse_layer_kind() {
        let path = write_dense_block_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let mut index = Index::from_gguf(&gguf, &config).unwrap();
        index.layers[0].kind = LayerKind::SparseMoe;
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();

        let err = DenseBlock::open(&gguf, &config, &index.layers[0], &backend, 256)
            .expect_err("sparse layer kind should reject dense block");

        assert!(err.to_string().contains("not dense"));
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
            rms_norm_eps: 1e-5,
            rope_theta: 10_000_000.0,
        }
    }

    fn write_dense_block_fixture(ty: GgmlType) -> PathBuf {
        let path = unique_temp_file("dense-block");
        let specs = vec![
            TensorSpec::quant("token_embd.weight", vec![256, 8], ty),
            TensorSpec::f32("output_norm.weight", vec![256]),
            TensorSpec::quant("output.weight", vec![256, 8], ty),
            TensorSpec::f32("blk.0.attn_norm.weight", vec![256]),
            TensorSpec::f32("blk.0.ffn_norm.weight", vec![256]),
            TensorSpec::quant("blk.0.attn_q_a.weight", vec![256, 256], ty),
            TensorSpec::f32("blk.0.attn_q_a_norm.weight", vec![256]),
            TensorSpec::quant("blk.0.attn_q_b.weight", vec![256, 512], ty),
            TensorSpec::quant("blk.0.attn_kv_a_mqa.weight", vec![256, 384], ty),
            TensorSpec::f32("blk.0.attn_kv_a_norm.weight", vec![256]),
            TensorSpec::quant("blk.0.attn_k_b.weight", vec![128, 256, 2], GgmlType::Q8_0),
            TensorSpec::quant("blk.0.attn_v_b.weight", vec![256, 256, 2], GgmlType::Q8_0),
            TensorSpec::quant("blk.0.attn_output.weight", vec![512, 256], ty),
            TensorSpec::quant("blk.0.ffn_gate.weight", vec![256, 512], ty),
            TensorSpec::quant("blk.0.ffn_up.weight", vec![256, 512], ty),
            TensorSpec::quant("blk.0.ffn_down.weight", vec![512, 256], ty),
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
        std::env::temp_dir().join(format!("inferno-{label}-{}-{id}", std::process::id()))
    }
}
