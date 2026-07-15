use std::collections::{BTreeMap, BTreeSet};

use backend::{Backend, BackendCapabilities};
use common::Tensor;
use common::{validate_exact_shape, Error, F32Tensor, Result, Shape};
use config::Config;
use gguf::{GgmlType, GgufFile};
use tracing::debug;

use crate::TensorRef;

#[derive(Debug)]
pub struct EmbeddingLookupOutput {
    pub hidden_states: Tensor,
    pub report: EmbeddingLookupReport,
}

#[derive(Debug)]
pub struct EmbeddingLookupF32Output {
    pub hidden_states: F32Tensor,
    pub report: EmbeddingLookupF32Report,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingLookupReport {
    pub source_tensor_name: String,
    pub tensor_type: GgmlType,
    pub backend: BackendCapabilities,
    pub input_ids_shape: Shape,
    pub gguf_embedding_shape: Shape,
    pub logical_embedding_shape: Shape,
    pub hidden_states_shape: Shape,
    pub token_count: usize,
    pub unique_token_count: usize,
    pub row_ranges_read: usize,
    pub blocks_per_row: u64,
    pub source_payload_bytes_read: u64,
    pub decoded_f32_bytes: u64,
    pub full_source_payload_bytes: u64,
    pub avoided_full_tensor_decode: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingLookupF32Report {
    pub source_tensor_name: String,
    pub tensor_type: GgmlType,
    pub input_ids_shape: Shape,
    pub gguf_embedding_shape: Shape,
    pub logical_embedding_shape: Shape,
    pub hidden_states_shape: Shape,
    pub token_count: usize,
    pub unique_token_count: usize,
    pub row_ranges_read: usize,
    pub blocks_per_row: u64,
    pub source_payload_bytes_read: u64,
    pub decoded_f32_bytes: u64,
    pub full_source_payload_bytes: u64,
    pub avoided_full_tensor_decode: bool,
}

impl EmbeddingLookupF32Report {
    fn with_backend(&self, backend: BackendCapabilities) -> EmbeddingLookupReport {
        EmbeddingLookupReport {
            source_tensor_name: self.source_tensor_name.clone(),
            tensor_type: self.tensor_type,
            backend,
            input_ids_shape: self.input_ids_shape.clone(),
            gguf_embedding_shape: self.gguf_embedding_shape.clone(),
            logical_embedding_shape: self.logical_embedding_shape.clone(),
            hidden_states_shape: self.hidden_states_shape.clone(),
            token_count: self.token_count,
            unique_token_count: self.unique_token_count,
            row_ranges_read: self.row_ranges_read,
            blocks_per_row: self.blocks_per_row,
            source_payload_bytes_read: self.source_payload_bytes_read,
            decoded_f32_bytes: self.decoded_f32_bytes,
            full_source_payload_bytes: self.full_source_payload_bytes,
            avoided_full_tensor_decode: self.avoided_full_tensor_decode,
        }
    }
}

#[derive(Debug)]
pub struct EmbeddingTable<'a> {
    gguf: &'a GgufFile,
    tensor_ref: TensorRef,
    hidden_size: usize,
    vocab_rows: usize,
    blocks_per_row: u64,
    source_row_bytes: u64,
    full_source_payload_bytes: u64,
}

impl<'a> EmbeddingTable<'a> {
    pub fn open(gguf: &'a GgufFile, config: &Config, tensor_ref: &TensorRef) -> Result<Self> {
        let info = gguf.tensor(&tensor_ref.name).ok_or_else(|| {
            Error::gguf(format!("missing GLM-5.2 GGUF tensor {}", tensor_ref.name))
        })?;
        validate_tensor_ref(tensor_ref, info)?;
        validate_embedding_type(tensor_ref)?;
        validate_exact_shape("gguf_token_embedding_rank", &[tensor_ref.dims.len()], &[2])?;

        let hidden_size = usize::try_from(tensor_ref.dims[0]).map_err(|_| {
            Error::gguf(format!(
                "GGUF token embedding hidden dimension {} does not fit usize",
                tensor_ref.dims[0]
            ))
        })?;
        let vocab_rows = usize::try_from(tensor_ref.dims[1]).map_err(|_| {
            Error::gguf(format!(
                "GGUF token embedding vocab dimension {} does not fit usize",
                tensor_ref.dims[1]
            ))
        })?;
        validate_exact_shape(
            "gguf_token_embedding_hidden_size",
            &[hidden_size],
            &[config.hidden_size],
        )?;
        if vocab_rows < config.vocab_size {
            return Err(Error::gguf(format!(
                "GGUF token embedding vocab rows {vocab_rows} are smaller than config vocab_size {}",
                config.vocab_size
            )));
        }

        let storage = gguf.tensor_quantized_storage(&tensor_ref.name)?;
        let block_size = usize::try_from(storage.block.values_per_block())
            .map_err(|_| Error::gguf("GGUF token embedding quant block size does not fit usize"))?;
        if hidden_size % block_size != 0 {
            return Err(Error::gguf(format!(
                "GGUF token embedding hidden_size {hidden_size} must be divisible by {block_size}"
            )));
        }
        let blocks_per_row = u64::try_from(hidden_size / block_size)
            .map_err(|_| Error::gguf("GGUF token embedding blocks_per_row does not fit u64"))?;
        let expected_blocks = blocks_per_row
            .checked_mul(u64::try_from(vocab_rows).map_err(|_| {
                Error::gguf("GGUF token embedding vocab row count does not fit u64")
            })?)
            .ok_or_else(|| Error::gguf("GGUF token embedding block count overflow"))?;
        validate_exact_shape(
            "gguf_token_embedding_block_count",
            &[usize::try_from(storage.block_count)
                .map_err(|_| Error::gguf("GGUF token embedding block count does not fit usize"))?],
            &[usize::try_from(expected_blocks).map_err(|_| {
                Error::gguf("GGUF token embedding expected block count does not fit usize")
            })?],
        )?;
        let source_row_bytes = blocks_per_row
            .checked_mul(storage.block.block_byte_len())
            .ok_or_else(|| Error::gguf("GGUF token embedding source row byte overflow"))?;

        Ok(Self {
            gguf,
            tensor_ref: tensor_ref.clone(),
            hidden_size,
            vocab_rows,
            blocks_per_row,
            source_row_bytes,
            full_source_payload_bytes: storage.payload_byte_len,
        })
    }

    pub fn lookup_with_backend<B: Backend>(
        &self,
        input_ids: &[u32],
        backend: &B,
    ) -> Result<EmbeddingLookupOutput> {
        let native = self.lookup_f32(input_ids)?;
        let dims = native.hidden_states.dims().to_vec();
        validate_exact_shape("gguf_embedding_native_rank", &[dims.len()], &[3])?;
        let (_, values) = native.hidden_states.into_parts();
        let hidden_states =
            Tensor::from_vec(values, (dims[0], dims[1], dims[2]), backend.device())?;

        Ok(EmbeddingLookupOutput {
            hidden_states,
            report: native.report.with_backend(backend.capabilities()),
        })
    }

    pub fn lookup_f32(&self, input_ids: &[u32]) -> Result<EmbeddingLookupF32Output> {
        let tokens = validate_input_ids(input_ids)?;
        let batch = 1_usize;
        for token_id in input_ids {
            if *token_id as usize >= self.vocab_rows {
                return Err(Error::model(format!(
                    "token id {token_id} is outside GGUF embedding rows {}",
                    self.vocab_rows
                )));
            }
        }

        let unique_ids = unique_sorted_token_ids(input_ids);
        let ranges = coalesce_contiguous_ranges(&unique_ids);
        let storage = self.gguf.tensor_quantized_storage(&self.tensor_ref.name)?;
        let mut row_cache = BTreeMap::<u32, Vec<f32>>::new();

        for range in &ranges {
            let block_start = u64::from(range.start_token_id)
                .checked_mul(self.blocks_per_row)
                .ok_or_else(|| Error::gguf("GGUF embedding row range block offset overflow"))?;
            let range_row_count = u64::try_from(range.len)
                .map_err(|_| Error::gguf("GGUF embedding row range length does not fit u64"))?;
            let range_block_count = range_row_count
                .checked_mul(self.blocks_per_row)
                .ok_or_else(|| Error::gguf("GGUF embedding row range block count overflow"))?;
            let range_values =
                storage.dequantize_block_range_as_f32(block_start, range_block_count)?;
            let expected_values = range.len.checked_mul(self.hidden_size).ok_or_else(|| {
                Error::gguf("GGUF embedding decoded row range value count overflow")
            })?;
            validate_exact_shape(
                format!(
                    "gguf_embedding_row_range_{}_{}",
                    range.start_token_id, range.len
                ),
                &[range_values.len()],
                &[expected_values],
            )?;
            for row_offset in 0..range.len {
                let token_id =
                    range
                        .start_token_id
                        .checked_add(u32::try_from(row_offset).map_err(|_| {
                            Error::gguf("GGUF embedding row offset does not fit u32")
                        })?)
                        .ok_or_else(|| Error::gguf("GGUF embedding token id overflow"))?;
                let row_start = row_offset
                    .checked_mul(self.hidden_size)
                    .ok_or_else(|| Error::gguf("GGUF embedding decoded row start overflow"))?;
                let row_end = row_start
                    .checked_add(self.hidden_size)
                    .ok_or_else(|| Error::gguf("GGUF embedding decoded row end overflow"))?;
                row_cache.insert(token_id, range_values[row_start..row_end].to_vec());
            }
        }

        let flat_value_count = tokens
            .checked_mul(self.hidden_size)
            .ok_or_else(|| Error::gguf("GGUF embedding flat hidden value count overflow"))?;
        let mut flat_values = Vec::with_capacity(flat_value_count);
        for token_id in input_ids {
            let row = row_cache.get(token_id).ok_or_else(|| {
                Error::model(format!(
                    "GGUF embedding row cache missing token id {token_id}"
                ))
            })?;
            flat_values.extend_from_slice(row);
        }
        let hidden_states = F32Tensor::new(flat_values, [batch, tokens, self.hidden_size])?;
        let source_payload_bytes_read = u64::try_from(unique_ids.len())
            .ok()
            .and_then(|unique| unique.checked_mul(self.source_row_bytes))
            .ok_or_else(|| Error::gguf("GGUF embedding bytes-read overflow"))?;
        let decoded_f32_bytes = u64::try_from(tokens)
            .ok()
            .and_then(|tokens| tokens.checked_mul(self.hidden_size as u64))
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| Error::gguf("GGUF embedding decoded byte count overflow"))?;

        debug!(
            tensor = %self.tensor_ref.name,
            token_count = tokens,
            unique_token_count = unique_ids.len(),
            source_payload_bytes_read,
            full_source_payload_bytes = self.full_source_payload_bytes,
            "loaded GLM-5.2 GGUF embedding rows"
        );

        Ok(EmbeddingLookupF32Output {
            hidden_states,
            report: EmbeddingLookupF32Report {
                source_tensor_name: self.tensor_ref.name.clone(),
                tensor_type: self.tensor_ref.ty,
                input_ids_shape: Shape::new(vec![batch, tokens]),
                gguf_embedding_shape: Shape::new(vec![self.hidden_size, self.vocab_rows]),
                logical_embedding_shape: Shape::new(vec![self.vocab_rows, self.hidden_size]),
                hidden_states_shape: Shape::new([batch, tokens, self.hidden_size]),
                token_count: tokens,
                unique_token_count: unique_ids.len(),
                row_ranges_read: ranges.len(),
                blocks_per_row: self.blocks_per_row,
                source_payload_bytes_read,
                decoded_f32_bytes,
                full_source_payload_bytes: self.full_source_payload_bytes,
                avoided_full_tensor_decode: source_payload_bytes_read
                    < self.full_source_payload_bytes,
            },
        })
    }
}

fn validate_embedding_type(tensor_ref: &TensorRef) -> Result<()> {
    match tensor_ref.ty {
        GgmlType::Q2K | GgmlType::Q8_0 => Ok(()),
        other => Err(Error::gguf(format!(
            "GLM-5.2 Q2 GGUF token embedding must be Q2_K or Q8_0, got {other}"
        ))),
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

fn validate_input_ids(input_ids: &[u32]) -> Result<usize> {
    if input_ids.is_empty() {
        return Err(Error::model("GGUF embedding token count must be positive"));
    }
    Ok(input_ids.len())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TokenRowRange {
    start_token_id: u32,
    len: usize,
}

impl TokenRowRange {
    fn next_token_id(&self) -> Option<u32> {
        self.start_token_id
            .checked_add(u32::try_from(self.len).ok()?)
    }
}

fn unique_sorted_token_ids(flat_ids: &[u32]) -> Vec<u32> {
    flat_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn coalesce_contiguous_ranges(sorted_unique_ids: &[u32]) -> Vec<TokenRowRange> {
    let mut ranges: Vec<TokenRowRange> = Vec::new();
    for token_id in sorted_unique_ids {
        match ranges.last_mut() {
            Some(range) if range.next_token_id() == Some(*token_id) => range.len += 1,
            _ => ranges.push(TokenRowRange {
                start_token_id: *token_id,
                len: 1,
            }),
        }
    }
    ranges
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use backend::MetalBackend;
    use common::Device;
    use config::Config;
    use gguf::{
        GgmlType, GgufFile, GgufMetadataValueType, GGML_Q2_K_BLOCK_BYTES, GGUF_MAGIC,
        GGUF_VERSION_V3,
    };

    use super::*;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn q2_lookup_decodes_only_unique_token_rows() {
        let path = write_embedding_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let tensor_ref = tensor_ref(&gguf, "token_embd.weight");
        let table = EmbeddingTable::open(&gguf, &tiny_config(), &tensor_ref).unwrap();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();

        let native = table.lookup_f32(&[3, 1, 3]).unwrap();
        assert_eq!(native.hidden_states.dims(), &[1, 3, 256]);
        assert_eq!(native.report.hidden_states_shape.dims(), &[1, 3, 256]);
        assert_eq!(native.hidden_states.values()[0], -2.0);

        let output = table.lookup_with_backend(&[3, 1, 3], &backend).unwrap();

        assert_eq!(output.report.hidden_states_shape.dims(), &[1, 3, 256]);
        assert_eq!(output.report.unique_token_count, 2);
        assert_eq!(output.report.row_ranges_read, 2);
        assert_eq!(
            output.report.source_payload_bytes_read,
            GGML_Q2_K_BLOCK_BYTES * 2
        );
        assert_eq!(
            output.report.full_source_payload_bytes,
            GGML_Q2_K_BLOCK_BYTES * 4
        );

        let values = output.hidden_states.to_vec3::<f32>().unwrap();
        assert_eq!(values[0][0][0], -2.0);
        assert_eq!(values[0][1][0], -1.0);
        assert_eq!(values[0][2][0], -2.0);
        assert_eq!(values[0][0][32], 1.0);
        assert_eq!(values[0][1][32], 2.0);
    }

    #[test]
    fn rejects_out_of_range_token_id() {
        let path = write_embedding_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let tensor_ref = tensor_ref(&gguf, "token_embd.weight");
        let table = EmbeddingTable::open(&gguf, &tiny_config(), &tensor_ref).unwrap();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();

        let err = table
            .lookup_with_backend(&[4], &backend)
            .expect_err("token id should exceed vocab rows");

        assert!(err.to_string().contains("outside GGUF embedding rows"));
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

    fn write_embedding_fixture(ty: GgmlType) -> PathBuf {
        let path = unique_temp_file("embedding");
        fs::write(&path, tiny_embedding_gguf(ty)).unwrap();
        path
    }

    fn tiny_embedding_gguf(ty: GgmlType) -> Vec<u8> {
        let mut writer = GgufWriter::new();
        writer.header(1, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);
        writer.tensor_info("token_embd.weight", &[256, 4], ty, 0);
        writer.pad_to(32);
        match ty {
            GgmlType::Q2K => {
                writer.bytes(&q2_k_block(0xe4, 1));
                writer.bytes(&q2_k_block(0xe4, 2));
                writer.bytes(&q2_k_block(0xe4, 3));
                writer.bytes(&q2_k_block(0xe4, 4));
            }
            other => panic!("unsupported fixture tensor type {other}"),
        }
        writer.bytes(&[0_u8; 16]);
        writer.finish()
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
    }

    impl GgufWriter {
        fn new() -> Self {
            Self { bytes: Vec::new() }
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
        }

        fn bytes(&mut self, bytes: &[u8]) {
            self.bytes.extend_from_slice(bytes);
        }

        fn finish(self) -> Vec<u8> {
            self.bytes
        }
    }

    fn tiny_config() -> Config {
        Config {
            model_type: "glm_moe_dsa".to_string(),
            hidden_size: 256,
            num_layers: 1,
            dense_layers: 1,
            sparse_moe_layers: Some(0),
            vocab_size: 4,
            attention_heads: 2,
            qk_head_dim: 4,
            qk_no_rope_dim: 2,
            qk_rope_dim: 2,
            kv_lora_rank: 2,
            v_head_dim: Some(4),
            num_routed_experts: 2,
            experts_per_token: 1,
            max_context: 16,
            dsa_index_topk: 4,
            index_head_dim: 128,
            index_n_heads: 32,
            index_topk_freq: 4,
            index_skip_topk_offset: 3,
            index_share_for_mtp_iteration: true,
            indexer_rope_interleave: true,
            indexer_types: Vec::new(),
            moe_intermediate_size: 3,
            num_shared_experts: 1,
            moe_groups: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 2.5,
            scoring_func: "sigmoid".to_string(),
            topk_method: "noaux_tc".to_string(),
            num_nextn_predict_layers: 0,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
        .validated()
        .unwrap()
    }

    fn unique_temp_file(label: &str) -> PathBuf {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("embedding-{label}-{}-{id}", std::process::id()))
    }
}
