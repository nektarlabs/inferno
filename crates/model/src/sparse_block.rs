use backend::{Backend, BackendCapabilities};
use common::Tensor;
use common::{validate_exact_shape, Error, F32Tensor, PagedKvView, Result, Shape};
use config::Config;
use gguf::GgufFile;

use crate::{
    profile, Attention, AttentionLoadReport, AttentionOutput, FfnIndex, LayerIndex, LayerKind,
    MoeFfn, MoeFfnLoadReport, MoeFfnOutput,
};

#[derive(Debug)]
pub struct SparseBlock<'a> {
    attention: Attention<'a>,
    ffn: MoeFfn<'a>,
    load_report: SparseBlockLoadReport,
}

#[derive(Debug)]
pub struct SparseBlockOutput {
    pub hidden_states: Tensor,
    pub cache_k: Tensor,
    pub cache_v: Tensor,
    pub report: SparseBlockForwardReport,
}

#[derive(Debug)]
pub struct SparseBlockTensors {
    pub hidden_states: Tensor,
    pub cache_k: F32Tensor,
    pub cache_v: F32Tensor,
}

#[derive(Debug)]
pub struct SparseBlockF32Tensors {
    pub hidden_states: F32Tensor,
    pub cache_k: F32Tensor,
    pub cache_v: F32Tensor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseBlockLoadReport {
    pub backend: BackendCapabilities,
    pub layer_index: usize,
    pub attention: AttentionLoadReport,
    pub ffn: MoeFfnLoadReport,
    pub output_chunk_rows: usize,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseBlockForwardReport {
    pub layer_index: usize,
    pub input_hidden_states_shape: Shape,
    pub attention_output_hidden_states_shape: Shape,
    pub cache_k_shape: Shape,
    pub cache_v_shape: Shape,
    pub attention_scores_shape: Shape,
    pub attention_past_tokens: usize,
    pub ffn_output_hidden_states_shape: Shape,
    pub ffn_flat_tokens_shape: Shape,
    pub routed_expert_outputs_shape: Shape,
    pub loaded_expert_count: usize,
    pub routed_source_payload_bytes_read: u64,
    pub routed_peak_decoded_f32_bytes: u64,
    pub output_hidden_states_shape: Shape,
}

impl<'a> SparseBlock<'a> {
    pub fn open<B: Backend>(
        gguf: &'a GgufFile,
        config: &Config,
        layer: &LayerIndex,
        backend: &B,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        validate_sparse_layer(config, layer)?;
        let attention = Attention::open(gguf, config, layer, backend, output_chunk_rows)?;
        let ffn = MoeFfn::open(gguf, config, layer, backend, output_chunk_rows)?;
        let load_report = SparseBlockLoadReport {
            backend: backend.capabilities(),
            layer_index: layer.layer_index,
            attention: attention.load_report().clone(),
            ffn: ffn.load_report().clone(),
            output_chunk_rows,
            limitations: vec![
                "GGUF sparse block composes GLM attention and sparse MoE FFN".to_string(),
                "routed experts are decoded from packed GGUF tensors only when selected"
                    .to_string(),
                "generation uses native Metal Q2_K kernels for attention and selected expert projections"
                    .to_string(),
                "paged attention and fused multi-expert dispatch are not active yet".to_string(),
            ],
        };

        Ok(Self {
            attention,
            ffn,
            load_report,
        })
    }

    pub fn load_report(&self) -> &SparseBlockLoadReport {
        &self.load_report
    }

    pub fn forward<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<SparseBlockOutput> {
        self.forward_with_past_kv(config, hidden_states, backend, None)
    }

    pub fn forward_with_past_kv<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
        past_kv: Option<(&Tensor, &Tensor)>,
    ) -> Result<SparseBlockOutput> {
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
    ) -> Result<SparseBlockTensors> {
        self.forward_tensors_with_past_kv(config, hidden_states, backend, None)
    }

    pub fn forward_tensors_with_past_kv<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
        past_kv: Option<(&F32Tensor, &F32Tensor)>,
    ) -> Result<SparseBlockTensors> {
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
            profile::run_layer_stage(self.load_report.layer_index, "sparse_moe.attention", || {
                self.attention
                    .forward_tensors_with_past_kv(config, hidden_states, backend, past_kv)
            })?;
        let hidden_states =
            profile::run_layer_stage(self.load_report.layer_index, "sparse_moe.ffn", || {
                self.ffn
                    .forward_tensors(config, &attention_output.hidden_states, backend)
            })?;

        Ok(SparseBlockTensors {
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
    ) -> Result<SparseBlockTensors> {
        validate_hidden_states(config, hidden_states)?;
        let hidden_states = tensor_to_f32_tensor(hidden_states)?;
        let output =
            self.forward_f32_tensors_with_past_kv(config, &hidden_states, backend, past_kv)?;

        Ok(SparseBlockTensors {
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
    ) -> Result<SparseBlockF32Tensors> {
        validate_hidden_states_f32(config, hidden_states)?;
        let attention_output =
            profile::run_layer_stage(self.load_report.layer_index, "sparse_moe.attention", || {
                self.attention.forward_f32_tensors_with_past_kv(
                    config,
                    hidden_states,
                    backend,
                    past_kv,
                )
            })?;
        let output_hidden_states =
            profile::run_layer_stage(self.load_report.layer_index, "sparse_moe.ffn", || {
                self.ffn
                    .forward_f32_tensor(config, &attention_output.hidden_states, backend)
            })?;

        Ok(SparseBlockF32Tensors {
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
    ) -> Result<SparseBlockF32Tensors> {
        validate_hidden_states_f32(config, hidden_states)?;
        let attention_output =
            profile::run_layer_stage(self.load_report.layer_index, "sparse_moe.attention", || {
                self.attention.forward_f32_tensors_with_paged_past_kv(
                    config,
                    hidden_states,
                    backend,
                    past_kv,
                )
            })?;
        let output_hidden_states =
            profile::run_layer_stage(self.load_report.layer_index, "sparse_moe.ffn", || {
                self.ffn
                    .forward_f32_tensor(config, &attention_output.hidden_states, backend)
            })?;

        Ok(SparseBlockF32Tensors {
            hidden_states: output_hidden_states,
            cache_k: attention_output.cache_k,
            cache_v: attention_output.cache_v,
        })
    }

    /// Batched device-resident decode layer: attention and MoE FFN encoded
    /// into the backend's open batch, hidden states never leaving the GPU
    /// (the router's top-k download is the layer's second sync point).
    pub(crate) fn forward_decode_device<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &backend::DeviceValue,
        backend: &B,
        past_kv: &PagedKvView<'_>,
    ) -> Result<Option<crate::kv_types::BlockDeviceTensors>> {
        let attention_output = crate::try_device!(profile::run_layer_stage(
            self.load_report.layer_index,
            "sparse_moe.attention",
            || self
                .attention
                .forward_decode_device(config, hidden_states, backend, past_kv),
        ));
        let output_hidden_states = crate::try_device!(profile::run_layer_stage(
            self.load_report.layer_index,
            "sparse_moe.ffn",
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
    ) -> Result<SparseBlockOutput> {
        let ffn_output = self
            .ffn
            .forward(config, &attention_output.hidden_states, backend)?;
        block_output(input_hidden_states, attention_output, ffn_output)
    }
}

fn block_output(
    input_hidden_states: &Tensor,
    attention_output: AttentionOutput,
    ffn_output: MoeFfnOutput,
) -> Result<SparseBlockOutput> {
    let attention_report = attention_output.report;
    let ffn_report = ffn_output.report;
    let report = SparseBlockForwardReport {
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
        ffn_flat_tokens_shape: ffn_report.flat_tokens_shape,
        routed_expert_outputs_shape: ffn_report.routed_expert_outputs_shape,
        loaded_expert_count: ffn_report.routed_expert_count,
        routed_source_payload_bytes_read: ffn_report.routed_source_payload_bytes_read,
        routed_peak_decoded_f32_bytes: ffn_report.routed_peak_decoded_f32_bytes,
        output_hidden_states_shape: Shape::new(ffn_output.hidden_states.dims().to_vec()),
    };

    Ok(SparseBlockOutput {
        hidden_states: ffn_output.hidden_states,
        cache_k: attention_output.cache_k,
        cache_v: attention_output.cache_v,
        report,
    })
}

fn validate_sparse_layer(config: &Config, layer: &LayerIndex) -> Result<()> {
    if layer.layer_index >= config.num_layers {
        return Err(Error::gguf(format!(
            "GLM-5.2 GGUF sparse block layer {} exceeds num_layers {}",
            layer.layer_index, config.num_layers
        )));
    }
    if layer.kind != LayerKind::SparseMoe {
        return Err(Error::gguf(format!(
            "GLM-5.2 GGUF layer {} is {:?}, not sparse_moe",
            layer.layer_index, layer.kind
        )));
    }
    if !matches!(layer.ffn, FfnIndex::SparseMoe { .. }) {
        return Err(Error::gguf(format!(
            "GLM-5.2 GGUF layer {} has non-sparse FFN index",
            layer.layer_index
        )));
    }
    Ok(())
}

fn validate_hidden_states(config: &Config, hidden_states: &Tensor) -> Result<()> {
    let dims = hidden_states.dims();
    if dims.len() != 3 {
        return Err(Error::model(format!(
            "GLM-5.2 GGUF sparse block input must be rank 3 [B,T,H], got {dims:?}"
        )));
    }
    validate_exact_shape(
        "gguf_sparse_block_hidden_states",
        dims,
        &[dims[0], dims[1], config.hidden_size],
    )
}

fn validate_hidden_states_f32(config: &Config, hidden_states: &F32Tensor) -> Result<()> {
    let dims = hidden_states.dims();
    if dims.len() != 3 {
        return Err(Error::model(format!(
            "GLM-5.2 GGUF native sparse block input must be rank 3 [B,T,H], got {dims:?}"
        )));
    }
    validate_exact_shape(
        "gguf_native_sparse_block_hidden_states",
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
    fn q2_sparse_block_runs_prefill_attention_then_moe() {
        let path = write_sparse_block_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let index = Index::from_gguf(&gguf, &config).unwrap();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let block = SparseBlock::open(&gguf, &config, &index.layers[0], &backend, 256).unwrap();
        let hidden_states =
            Tensor::from_vec(vec![1.0_f32; 2 * 3 * 256], (2, 3, 256), &Device::Cpu).unwrap();

        let output = block.forward(&config, &hidden_states, &backend).unwrap();
        let tensors = block
            .forward_tensors(&config, &hidden_states, &backend)
            .unwrap();

        assert_eq!(block.load_report().layer_index, 0);
        assert_eq!(block.load_report().attention.q_lora_rank, 256);
        assert_eq!(block.load_report().ffn.routed_expert_count, 4);
        assert_eq!(output.hidden_states.dims(), &[2, 3, 256]);
        assert_eq!(tensors.hidden_states.dims(), &[2, 3, 256]);
        assert_eq!(output.cache_k.dims(), &[2, 2, 3, 256]);
        assert_eq!(output.cache_v.dims(), &[2, 2, 3, 256]);
        assert_eq!(tensors.cache_k.dims(), &[2, 2, 3, 256]);
        assert_eq!(tensors.cache_v.dims(), &[2, 2, 3, 256]);
        assert_eq!(output.report.attention_scores_shape.dims(), &[2, 2, 3, 3]);
        assert_eq!(output.report.ffn_flat_tokens_shape.dims(), &[6, 256]);
        assert_eq!(output.report.routed_expert_outputs_shape.dims(), &[12, 256]);
        assert_eq!(output.report.loaded_expert_count, 2);
    }

    #[test]
    fn q2_sparse_block_runs_decode_with_past_kv() {
        let path = write_sparse_block_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let index = Index::from_gguf(&gguf, &config).unwrap();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let block = SparseBlock::open(&gguf, &config, &index.layers[0], &backend, 256).unwrap();
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
        assert_eq!(decode.report.routed_expert_outputs_shape.dims(), &[2, 256]);
    }

    #[test]
    fn rejects_dense_layer_kind() {
        let path = write_sparse_block_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let mut index = Index::from_gguf(&gguf, &config).unwrap();
        index.layers[0].kind = LayerKind::Dense;
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();

        let err = SparseBlock::open(&gguf, &config, &index.layers[0], &backend, 256)
            .expect_err("dense layer kind should reject sparse block");

        assert!(err.to_string().contains("not sparse_moe"));
    }

    fn tiny_config() -> Config {
        Config {
            model_type: "glm_moe_dsa".to_string(),
            hidden_size: 256,
            num_layers: 1,
            dense_layers: 0,
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

    fn write_sparse_block_fixture(ty: GgmlType) -> PathBuf {
        let path = unique_temp_file("sparse-block");
        let specs = vec![
            TensorSpec::quant("token_embd.weight", vec![256, 8], ty, None),
            TensorSpec::f32("output_norm.weight", vec![256], None),
            TensorSpec::quant("output.weight", vec![256, 8], ty, None),
            TensorSpec::f32("blk.0.attn_norm.weight", vec![256], None),
            TensorSpec::f32("blk.0.ffn_norm.weight", vec![256], None),
            TensorSpec::quant("blk.0.attn_q_a.weight", vec![256, 256], ty, None),
            TensorSpec::f32("blk.0.attn_q_a_norm.weight", vec![256], None),
            TensorSpec::quant("blk.0.attn_q_b.weight", vec![256, 512], ty, None),
            TensorSpec::quant("blk.0.attn_kv_a_mqa.weight", vec![256, 384], ty, None),
            TensorSpec::f32("blk.0.attn_kv_a_norm.weight", vec![256], None),
            TensorSpec::quant(
                "blk.0.attn_k_b.weight",
                vec![128, 256, 2],
                GgmlType::Q8_0,
                None,
            ),
            TensorSpec::quant(
                "blk.0.attn_v_b.weight",
                vec![256, 256, 2],
                GgmlType::Q8_0,
                None,
            ),
            TensorSpec::quant("blk.0.attn_output.weight", vec![512, 256], ty, None),
            TensorSpec::f32(
                "blk.0.exp_probs_b.bias",
                vec![4],
                Some(vec![0.0, 0.4, 0.2, 0.8]),
            ),
            TensorSpec::quant("blk.0.ffn_gate_inp.weight", vec![256, 4], ty, None),
            TensorSpec::quant("blk.0.ffn_gate_shexp.weight", vec![256, 256], ty, None),
            TensorSpec::quant("blk.0.ffn_up_shexp.weight", vec![256, 256], ty, None),
            TensorSpec::quant("blk.0.ffn_down_shexp.weight", vec![256, 256], ty, None),
            TensorSpec::quant("blk.0.ffn_gate_exps.weight", vec![256, 256, 4], ty, None),
            TensorSpec::quant("blk.0.ffn_up_exps.weight", vec![256, 256, 4], ty, None),
            TensorSpec::quant("blk.0.ffn_down_exps.weight", vec![256, 256, 4], ty, None),
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

        fn quant(
            name: &'static str,
            dims: Vec<u64>,
            ty: GgmlType,
            values: Option<Vec<f32>>,
        ) -> Self {
            Self {
                name,
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
