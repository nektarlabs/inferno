use backend::{Backend, BackendCapabilities};
use common::Tensor;
use common::{validate_exact_shape, Error, F32Tensor, Result, Shape};
use config::Config;
use gguf::{GgmlType, GgufFile};
use tracing::debug;

use crate::{QuantizedLinear, RmsNorm, RmsNormLoadReport, RootIndex};

#[derive(Debug)]
pub struct OutputHead<'a> {
    final_norm: RmsNorm,
    output_projection: QuantizedLinear<'a>,
    hidden_size: usize,
    vocab_rows: usize,
    load_report: OutputHeadLoadReport,
}

#[derive(Debug)]
pub struct LogitsOutput {
    pub logits: Tensor,
    pub report: LogitsReport,
}

#[derive(Debug)]
pub struct GreedyOutput {
    pub token_id: u32,
    pub token_score: f32,
    pub report: GreedyReport,
}

#[derive(Debug)]
pub struct TokenOutput {
    pub token_id: u32,
    pub token_score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OutputHeadLoadReport {
    pub backend: BackendCapabilities,
    pub final_norm: RmsNormLoadReport,
    pub output_tensor_name: String,
    pub output_tensor_type: GgmlType,
    pub logical_output_weight_shape: Shape,
    pub output_chunk_rows: usize,
    pub full_output_source_payload_bytes: u64,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogitsReport {
    pub input_hidden_states_shape: Shape,
    pub normalized_hidden_states_shape: Shape,
    pub logits_shape: Shape,
    pub output_projection_shape: Shape,
    pub output_projection_chunk_count: usize,
    pub output_projection_source_payload_bytes_read: u64,
    pub output_projection_full_source_payload_bytes: u64,
    pub output_projection_peak_decoded_f32_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GreedyReport {
    pub input_hidden_states_shape: Shape,
    pub normalized_hidden_states_shape: Shape,
    pub logits_shape: Shape,
    pub output_projection_chunk_count: usize,
    pub output_projection_source_payload_bytes_read: u64,
    pub output_projection_full_source_payload_bytes: u64,
    pub output_projection_peak_decoded_f32_bytes: u64,
    pub materialized_full_logits: bool,
}

impl<'a> OutputHead<'a> {
    pub fn open<B: Backend>(
        gguf: &'a GgufFile,
        config: &Config,
        root: &RootIndex,
        backend: &B,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        let (input_features, vocab_rows) = validate_output_weight_shape(config, &root.output.dims)?;
        let final_norm = RmsNorm::open(gguf, config, &root.final_norm, backend)?;
        let output_projection = QuantizedLinear::open(
            gguf,
            &root.output,
            input_features,
            config.vocab_size,
            output_chunk_rows,
        )?;

        let load_report = OutputHeadLoadReport {
            backend: backend.capabilities(),
            final_norm: final_norm.load_report().clone(),
            output_tensor_name: root.output.name.clone(),
            output_tensor_type: root.output.ty,
            logical_output_weight_shape: Shape::new(vec![vocab_rows, input_features]),
            output_chunk_rows,
            full_output_source_payload_bytes: root.output.storage_byte_len,
            limitations: vec![
                "GGUF output projection is decoded one output chunk at a time".to_string(),
                "token generation uses native Metal Q2_K greedy argmax for the output projection"
                    .to_string(),
            ],
        };

        debug!(
            output_tensor = %load_report.output_tensor_name,
            output_tensor_type = %load_report.output_tensor_type,
            output_chunk_rows,
            full_output_source_payload_bytes = load_report.full_output_source_payload_bytes,
            "opened GLM-5.2 GGUF output head"
        );

        Ok(Self {
            final_norm,
            output_projection,
            hidden_size: config.hidden_size,
            vocab_rows,
            load_report,
        })
    }

    pub fn load_report(&self) -> &OutputHeadLoadReport {
        &self.load_report
    }

    pub fn decode_logits<B: Backend>(
        &self,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<LogitsOutput> {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF decode logits expects hidden states [B, 1, H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        validate_exact_shape(
            "gguf_output_head_decode_hidden_states",
            dims,
            &[batch, 1, self.hidden_size],
        )?;

        let normalized = self.final_norm.forward(hidden_states, backend)?;
        let projection = self
            .output_projection
            .forward(&normalized.hidden_states, backend)?;
        let logits = projection.output.reshape((batch, self.vocab_rows))?;
        let projection_report = projection.report;
        let report = LogitsReport {
            input_hidden_states_shape: Shape::new(hidden_states.dims().to_vec()),
            normalized_hidden_states_shape: Shape::new(normalized.hidden_states.dims().to_vec()),
            logits_shape: Shape::new(logits.dims().to_vec()),
            output_projection_shape: projection_report.output_shape,
            output_projection_chunk_count: projection_report.chunk_count,
            output_projection_source_payload_bytes_read: projection_report
                .source_payload_bytes_read,
            output_projection_full_source_payload_bytes: projection_report
                .full_source_payload_bytes,
            output_projection_peak_decoded_f32_bytes: projection_report.peak_decoded_f32_bytes,
        };

        debug!(
            batch,
            vocab_rows = self.vocab_rows,
            output_chunk_rows = self.load_report.output_chunk_rows,
            chunk_count = report.output_projection_chunk_count,
            "computed GLM-5.2 GGUF decode logits"
        );

        Ok(LogitsOutput { logits, report })
    }

    pub fn decode_greedy<B: Backend>(
        &self,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<GreedyOutput> {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF greedy decode expects hidden states [B, 1, H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        validate_exact_shape(
            "gguf_output_head_greedy_hidden_states",
            dims,
            &[batch, 1, self.hidden_size],
        )?;

        let normalized = self.final_norm.forward(hidden_states, backend)?;
        let greedy = self
            .output_projection
            .greedy_argmax(&normalized.hidden_states, backend)?;
        let greedy_report = greedy.report;
        let report = GreedyReport {
            input_hidden_states_shape: Shape::new(hidden_states.dims().to_vec()),
            normalized_hidden_states_shape: Shape::new(normalized.hidden_states.dims().to_vec()),
            logits_shape: greedy_report.logits_shape,
            output_projection_chunk_count: greedy_report.chunk_count,
            output_projection_source_payload_bytes_read: greedy_report.source_payload_bytes_read,
            output_projection_full_source_payload_bytes: greedy_report.full_source_payload_bytes,
            output_projection_peak_decoded_f32_bytes: greedy_report.peak_decoded_f32_bytes,
            materialized_full_logits: false,
        };

        debug!(
            batch,
            vocab_rows = self.vocab_rows,
            output_chunk_rows = self.load_report.output_chunk_rows,
            chunk_count = report.output_projection_chunk_count,
            "computed GLM-5.2 GGUF greedy decode"
        );

        Ok(GreedyOutput {
            token_id: greedy.token_id,
            token_score: greedy.token_score,
            report,
        })
    }

    pub fn decode_token<B: Backend>(
        &self,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<TokenOutput> {
        if backend.capabilities().custom_kernels {
            return self.decode_token_native(hidden_states, backend);
        }

        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF token decode expects hidden states [B, 1, H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        validate_exact_shape(
            "gguf_output_head_token_hidden_states",
            dims,
            &[batch, 1, self.hidden_size],
        )?;

        let normalized = self.final_norm.forward_tensor(hidden_states, backend)?;
        let token = self.output_projection.greedy_token(&normalized, backend)?;

        Ok(TokenOutput {
            token_id: token.token_id,
            token_score: token.token_score,
        })
    }

    fn decode_token_native<B: Backend>(
        &self,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<TokenOutput> {
        let hidden_states = tensor_to_f32_tensor(hidden_states)?;
        self.decode_token_f32(&hidden_states, backend)
    }

    /// Batched device-resident greedy decode: select-last-token, final norm
    /// and the fused output matvec + argmax are encoded into the backend's
    /// open batch; the argmax flushes it and returns the winning token. This
    /// is the end-of-token synchronization point of the device decode path.
    pub(crate) fn decode_token_device<B: Backend>(
        &self,
        hidden_states: &backend::DeviceValue,
        backend: &B,
    ) -> Result<Option<TokenOutput>> {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF device token decode expects hidden states [B, T, H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        validate_exact_shape(
            "gguf_device_output_head_hidden_size",
            &[dims[2]],
            &[self.hidden_size],
        )?;

        let selected = crate::try_device!(backend.select_last_token_device(hidden_states));
        let normalized = crate::try_device!(self.final_norm.forward_device(&selected, backend));
        let flat = normalized.reshape(vec![batch, self.hidden_size])?;
        let token = crate::try_device!(self.output_projection.greedy_token_device(&flat, backend));

        Ok(Some(TokenOutput {
            token_id: token.token_id,
            token_score: token.token_score,
        }))
    }

    pub fn decode_token_f32<B: Backend>(
        &self,
        hidden_states: &F32Tensor,
        backend: &B,
    ) -> Result<TokenOutput> {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF native token decode expects hidden states [B, 1, H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        validate_exact_shape(
            "gguf_native_output_head_token_hidden_states",
            dims,
            &[batch, 1, self.hidden_size],
        )?;

        if let Some(token) = self.output_projection.greedy_token_after_rms_norm_f32(
            hidden_states,
            self.final_norm.weight(),
            self.final_norm.eps(),
            backend,
        )? {
            return Ok(TokenOutput {
                token_id: token.token_id,
                token_score: token.token_score,
            });
        }

        let normalized = self
            .final_norm
            .forward_f32(&hidden_states, backend)?
            .hidden_states;
        let token = self
            .output_projection
            .greedy_token_f32(&normalized, backend)?;

        Ok(TokenOutput {
            token_id: token.token_id,
            token_score: token.token_score,
        })
    }

    pub(crate) fn decode_token_from_normalized_f32<B: Backend>(
        &self,
        normalized_hidden_states: &F32Tensor,
        backend: &B,
    ) -> Result<TokenOutput> {
        let dims = normalized_hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF normalized token decode expects hidden states [B, 1, H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        validate_exact_shape(
            "gguf_output_head_normalized_token_hidden_states",
            dims,
            &[batch, 1, self.hidden_size],
        )?;
        let token = self
            .output_projection
            .greedy_token_f32(normalized_hidden_states, backend)?;

        Ok(TokenOutput {
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

fn validate_output_weight_shape(config: &Config, dims: &[u64]) -> Result<(usize, usize)> {
    if dims.len() != 2 {
        return Err(Error::gguf(format!(
            "GLM-5.2 GGUF output.weight must be rank 2 [hidden_size, vocab_rows], got {dims:?}"
        )));
    }
    let input_features = usize::try_from(dims[0])
        .map_err(|_| Error::gguf("GGUF output input dimension does not fit usize"))?;
    let vocab_rows = usize::try_from(dims[1])
        .map_err(|_| Error::gguf("GGUF output vocab dimension does not fit usize"))?;

    validate_exact_shape(
        "gguf_output_head_input_features",
        &[input_features],
        &[config.hidden_size],
    )?;
    if vocab_rows < config.vocab_size {
        return Err(Error::gguf(format!(
            "GLM-5.2 GGUF output rows {vocab_rows} are smaller than config vocab_size {}",
            config.vocab_size
        )));
    }

    Ok((input_features, vocab_rows))
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
    use crate::TensorRef;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn q2_output_head_runs_decode_logits_in_chunks() {
        let path = write_output_head_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let root = root_index(&gguf);
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let head = OutputHead::open(&gguf, &tiny_config(256, 4), &root, &backend, 3).unwrap();
        let hidden_states =
            Tensor::from_vec(vec![1.0_f32; 2 * 256], (2, 1, 256), &Device::Cpu).unwrap();

        let output = head.decode_logits(&hidden_states, &backend).unwrap();

        assert_eq!(
            head.load_report().full_output_source_payload_bytes,
            GGML_Q2_K_BLOCK_BYTES * 4
        );
        assert_eq!(output.logits.dims(), &[2, 4]);
        assert_eq!(output.report.output_projection_chunk_count, 2);
        assert_eq!(
            output.report.output_projection_peak_decoded_f32_bytes,
            3 * 256 * 4
        );
    }

    #[test]
    fn rejects_prefill_shape_for_decode_logits() {
        let path = write_output_head_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let root = root_index(&gguf);
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let head = OutputHead::open(&gguf, &tiny_config(256, 4), &root, &backend, 2).unwrap();
        let hidden_states =
            Tensor::from_vec(vec![1.0_f32; 2 * 256], (1, 2, 256), &Device::Cpu).unwrap();

        let err = head
            .decode_logits(&hidden_states, &backend)
            .expect_err("decode logits should only accept one token");

        assert!(err
            .to_string()
            .contains("gguf_output_head_decode_hidden_states"));
    }

    #[test]
    fn rejects_output_vocab_smaller_than_config() {
        let path = write_output_head_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let root = root_index(&gguf);
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();

        let err = OutputHead::open(&gguf, &tiny_config(256, 5), &root, &backend, 2)
            .expect_err("output rows smaller than vocab should fail");

        assert!(err.to_string().contains("smaller than config vocab_size"));
    }

    fn root_index(gguf: &GgufFile) -> RootIndex {
        RootIndex {
            token_embedding: tensor_ref(gguf, "token_embd.weight"),
            final_norm: tensor_ref(gguf, "output_norm.weight"),
            output: tensor_ref(gguf, "output.weight"),
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

    fn tiny_config(hidden_size: usize, vocab_size: usize) -> Config {
        Config {
            model_type: "glm_moe_dsa".to_string(),
            hidden_size,
            num_layers: 1,
            dense_layers: 1,
            sparse_moe_layers: Some(0),
            vocab_size,
            attention_heads: 1,
            qk_head_dim: hidden_size,
            qk_no_rope_dim: hidden_size,
            qk_rope_dim: 0,
            kv_lora_rank: hidden_size,
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
            max_context: 16,
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

    fn write_output_head_fixture(output_ty: GgmlType) -> PathBuf {
        let path = unique_temp_file("output-head");
        let mut writer = GgufWriter::new();
        writer.header(3, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);

        let embedding_offset = 0_u64;
        let embedding_bytes = quantized_payload_bytes(output_ty, 4);
        let norm_offset = align_u64(embedding_offset + embedding_bytes, 32);
        let norm_bytes = 256_u64 * 4;
        let output_offset = align_u64(norm_offset + norm_bytes, 32);
        let output_bytes = quantized_payload_bytes(output_ty, 4);

        writer.tensor_info("token_embd.weight", &[256, 4], output_ty, embedding_offset);
        writer.tensor_info("output_norm.weight", &[256], GgmlType::F32, norm_offset);
        writer.tensor_info("output.weight", &[256, 4], output_ty, output_offset);
        writer.pad_to(32);

        write_quantized_payload(&mut writer, output_ty);
        writer.pad_to_absolute_data_offset(norm_offset);
        for _ in 0..256 {
            writer.bytes(&1.0_f32.to_le_bytes());
        }
        writer.pad_to_absolute_data_offset(output_offset);
        write_quantized_payload(&mut writer, output_ty);
        writer.pad_to_absolute_data_offset(output_offset + output_bytes);
        writer.finish_to(path)
    }

    fn write_quantized_payload(writer: &mut GgufWriter, ty: GgmlType) {
        match ty {
            GgmlType::Q2K => {
                writer.bytes(&q2_k_block(0xe4, 1));
                writer.bytes(&q2_k_block(0xe4, 2));
                writer.bytes(&q2_k_block(0xe4, 3));
                writer.bytes(&q2_k_block(0xe4, 4));
            }
            other => panic!("unsupported fixture tensor type {other}"),
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
        std::env::temp_dir().join(format!("inferno-{label}-{}-{id}", std::process::id()))
    }
}
