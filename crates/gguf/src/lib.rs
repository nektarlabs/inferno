#![deny(unsafe_code)]

//! Read-only GGUF container adapter.
//!
//! This crate only parses the GGUF header, metadata block, and tensor directory.
//! Tensor dequantization and execution stay behind the model/backend crates.

use std::{
    collections::BTreeMap,
    fmt,
    path::{Path, PathBuf},
};

use common::{Error, Result};
use io::{MappedBytes, MappedFile, MappedFileAdvice};
use tracing::debug;

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";
pub const GGUF_VERSION_V3: u32 = 3;
pub const DEFAULT_GGUF_ALIGNMENT: u64 = 32;
pub const GGML_K_QUANT_BLOCK_SIZE: u64 = 256;
pub const GGML_Q2_K_BLOCK_BYTES: u64 = 84;
pub const GGML_Q3_K_BLOCK_BYTES: u64 = 110;
pub const GGML_Q8_0_BLOCK_SIZE: u64 = 32;
pub const GGML_Q8_0_BLOCK_BYTES: u64 = 34;
const GGML_Q2_K_SCALE_BYTES: usize = 16;
const GGML_Q2_K_QUANT_BYTES: usize = 64;
const GGML_Q3_K_HIGH_MASK_BYTES: usize = 32;
const GGML_Q3_K_QUANT_BYTES: usize = 64;
const GGML_Q3_K_SCALE_BYTES: usize = 12;
const MAX_TENSOR_DIMS: usize = 4;
const MAX_METADATA_KEY_BYTES: usize = u16::MAX as usize;
const MAX_TENSOR_NAME_BYTES: usize = 64;
const MAX_STORED_ARRAY_VALUES: u64 = 1024;
const MAX_METADATA_DEPTH: usize = 8;

#[derive(Debug)]
pub struct GgufFile {
    path: PathBuf,
    mapped: MappedFile,
    file_size: u64,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    alignment: u64,
    tensor_data_offset: u64,
    metadata: BTreeMap<String, GgufMetadataValue>,
    tensors: Vec<GgufTensorInfo>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GgufMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array {
        element_type: GgufMetadataValueType,
        len: u64,
        values: Option<Vec<GgufMetadataValue>>,
    },
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GgufMetadataValueType {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Bool,
    String,
    Array,
    Uint64,
    Int64,
    Float64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    pub relative_offset: u64,
    pub absolute_offset: u64,
    pub storage_byte_len: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct GgufTensorStorage<'a> {
    pub info: &'a GgufTensorInfo,
    pub bytes: &'a [u8],
}

#[derive(Debug, Clone, Copy)]
pub struct GgufQuantizedTensorStorage<'a> {
    pub info: &'a GgufTensorInfo,
    pub block: GgufQuantBlockKind,
    pub block_count: u64,
    pub payload_byte_len: u64,
    pub bytes: &'a [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufQuantBlockKind {
    Q2K,
    Q3K,
    Q8_0,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GgmlType {
    F32,
    Q8_0,
    Q2K,
    Q3K,
    Unsupported(u32),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufSummary {
    pub path: PathBuf,
    pub file_size: u64,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub alignment: u64,
    pub tensor_data_offset: u64,
    pub architecture: Option<String>,
    pub quantization_version: Option<u64>,
    pub file_type: Option<u64>,
    pub tensor_type_counts: Vec<GgufTensorTypeCount>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorTypeCount {
    pub ty: GgmlType,
    pub count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufTensorAdvice {
    Random,
    WillNeed,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mapped = MappedFile::open(path)?;
        let file_size = mapped
            .len()
            .try_into()
            .map_err(|_| Error::gguf("GGUF file size does not fit u64"))?;
        let (
            version,
            tensor_count,
            metadata_kv_count,
            alignment,
            tensor_data_offset,
            metadata,
            mut tensors,
        ) = {
            let mut reader = GgufReader::new(mapped.slice(0, mapped.len())?);

            reader.expect_magic()?;
            let version = reader.read_u32()?;
            if version != GGUF_VERSION_V3 {
                return Err(Error::gguf(format!(
                    "unsupported GGUF version {version}; expected {GGUF_VERSION_V3}"
                )));
            }
            let tensor_count = reader.read_u64()?;
            let metadata_kv_count = reader.read_u64()?;

            let mut metadata = BTreeMap::new();
            for _ in 0..metadata_kv_count {
                let key = reader.read_limited_string(MAX_METADATA_KEY_BYTES)?;
                validate_metadata_key(&key)?;
                let value_type = reader.read_metadata_type()?;
                let value = reader.read_metadata_value(value_type, true, 0)?;
                if metadata.insert(key.clone(), value).is_some() {
                    return Err(Error::gguf(format!("duplicate GGUF metadata key {key}")));
                }
            }

            let alignment = metadata_alignment(&metadata)?;
            let tensor_count_usize = usize::try_from(tensor_count)
                .map_err(|_| Error::gguf("GGUF tensor_count does not fit usize"))?;
            let mut tensors = Vec::with_capacity(tensor_count_usize);
            for _ in 0..tensor_count {
                let name = reader.read_limited_string(MAX_TENSOR_NAME_BYTES)?;
                let n_dims = reader.read_u32()?;
                if n_dims == 0 {
                    return Err(Error::gguf(format!(
                        "GGUF tensor {name} must have at least one dimension"
                    )));
                }
                if n_dims as usize > MAX_TENSOR_DIMS {
                    return Err(Error::gguf(format!(
                        "GGUF tensor {name} has {n_dims} dimensions; max supported is {MAX_TENSOR_DIMS}"
                    )));
                }

                let mut dims = Vec::with_capacity(n_dims as usize);
                for _ in 0..n_dims {
                    let dim = reader.read_u64()?;
                    if dim == 0 {
                        return Err(Error::gguf(format!(
                            "GGUF tensor {name} contains a zero dimension"
                        )));
                    }
                    dims.push(dim);
                }

                let ty = GgmlType::from_code(reader.read_u32()?);
                let relative_offset = reader.read_u64()?;
                if relative_offset % alignment != 0 {
                    return Err(Error::gguf(format!(
                        "GGUF tensor {name} relative offset {relative_offset} is not aligned to {alignment}"
                    )));
                }
                tensors.push(GgufTensorInfo {
                    name,
                    dims,
                    ty,
                    relative_offset,
                    absolute_offset: 0,
                    storage_byte_len: 0,
                });
            }

            let tensor_data_offset = align_offset(reader.position_u64()?, alignment)?;
            (
                version,
                tensor_count,
                metadata_kv_count,
                alignment,
                tensor_data_offset,
                metadata,
                tensors,
            )
        };

        for tensor in &mut tensors {
            tensor.absolute_offset = tensor_data_offset
                .checked_add(tensor.relative_offset)
                .ok_or_else(|| Error::gguf("GGUF tensor absolute offset overflow"))?;
            if tensor.absolute_offset > file_size {
                return Err(Error::gguf(format!(
                    "GGUF tensor {} absolute offset {} exceeds file size {}",
                    tensor.name, tensor.absolute_offset, file_size
                )));
            }
        }
        assign_storage_byte_lengths(&mut tensors, file_size)?;

        debug!(
            path = %path.display(),
            version,
            tensor_count,
            metadata_kv_count,
            alignment,
            tensor_data_offset,
            "parsed GGUF header and tensor directory"
        );

        Ok(Self {
            path: path.to_path_buf(),
            mapped,
            file_size,
            version,
            tensor_count,
            metadata_kv_count,
            alignment,
            tensor_data_offset,
            metadata,
            tensors,
        })
    }

    pub fn summary(&self) -> GgufSummary {
        let mut counts = BTreeMap::<GgmlType, usize>::new();
        for tensor in &self.tensors {
            *counts.entry(tensor.ty).or_default() += 1;
        }

        GgufSummary {
            path: self.path.clone(),
            file_size: self.file_size,
            version: self.version,
            tensor_count: self.tensor_count,
            metadata_kv_count: self.metadata_kv_count,
            alignment: self.alignment,
            tensor_data_offset: self.tensor_data_offset,
            architecture: self
                .metadata_string("general.architecture")
                .map(str::to_string),
            quantization_version: self.metadata_unsigned("general.quantization_version"),
            file_type: self.metadata_unsigned("general.file_type"),
            tensor_type_counts: counts
                .into_iter()
                .map(|(ty, count)| GgufTensorTypeCount { ty, count })
                .collect(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn mapped_bytes(&self) -> MappedBytes {
        self.mapped.shared_bytes()
    }

    pub fn tensor_data_offset(&self) -> u64 {
        self.tensor_data_offset
    }

    pub fn max_tensor_storage_byte_len(&self) -> u64 {
        self.tensors
            .iter()
            .map(|tensor| tensor.storage_byte_len)
            .max()
            .unwrap_or(0)
    }

    pub fn metadata(&self) -> &BTreeMap<String, GgufMetadataValue> {
        &self.metadata
    }

    pub fn tensors(&self) -> &[GgufTensorInfo] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    pub fn tensor_storage(&self, name: &str) -> Result<GgufTensorStorage<'_>> {
        let info = self
            .tensor(name)
            .ok_or_else(|| Error::gguf(format!("missing GGUF tensor {name}")))?;
        self.tensor_storage_by_info(info)
    }

    pub fn tensor_storage_by_info<'a>(
        &'a self,
        info: &'a GgufTensorInfo,
    ) -> Result<GgufTensorStorage<'a>> {
        let byte_len = usize::try_from(info.storage_byte_len).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} storage length {} does not fit usize",
                info.name, info.storage_byte_len
            ))
        })?;
        let bytes = self.mapped.slice(info.absolute_offset, byte_len)?;
        Ok(GgufTensorStorage { info, bytes })
    }

    pub fn tensor_f32_values(&self, name: &str) -> Result<Vec<f32>> {
        let storage = self.tensor_storage(name)?;
        if storage.info.ty != GgmlType::F32 {
            return Err(Error::gguf(format!(
                "GGUF tensor {name} must be F32, got {}",
                storage.info.ty
            )));
        }
        let element_count = usize::try_from(storage.info.element_count()?).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {name} element count does not fit usize"
            ))
        })?;
        let byte_len = element_count
            .checked_mul(4)
            .ok_or_else(|| Error::gguf(format!("GGUF tensor {name} F32 byte count overflow")))?;
        if storage.bytes.len() < byte_len {
            return Err(Error::gguf(format!(
                "GGUF tensor {name} storage has {} bytes but F32 payload requires {byte_len}",
                storage.bytes.len()
            )));
        }

        Ok(storage.bytes[..byte_len]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect())
    }

    pub fn tensor_quantized_storage(&self, name: &str) -> Result<GgufQuantizedTensorStorage<'_>> {
        let storage = self.tensor_storage(name)?;
        storage.quantized_payload()
    }

    pub fn advise_tensor(&self, name: &str, advice: GgufTensorAdvice) -> Result<()> {
        let tensor = self
            .tensor(name)
            .ok_or_else(|| Error::gguf(format!("missing GGUF tensor {name}")))?;
        self.advise_tensor_by_info(tensor, advice)
    }

    pub fn advise_tensor_by_info(
        &self,
        info: &GgufTensorInfo,
        advice: GgufTensorAdvice,
    ) -> Result<()> {
        self.mapped
            .advise_range(advice.into(), info.absolute_offset, info.storage_byte_len)
    }

    pub fn prefetch_range(&self, absolute_offset: u64, byte_len: u64) -> Result<()> {
        self.mapped.prefetch_range(absolute_offset, byte_len)
    }

    pub fn prefetch_ranges(&self, ranges: &[(u64, u64)]) -> Result<()> {
        self.mapped.prefetch_ranges(ranges)
    }

    pub fn cache_identity(&self) -> usize {
        self.mapped.cache_identity()
    }

    pub fn tensor_q2_k_f32_values(&self, name: &str) -> Result<Vec<f32>> {
        self.tensor_quantized_storage(name)?.dequantize_q2_k()
    }

    pub fn tensor_q3_k_f32_values(&self, name: &str) -> Result<Vec<f32>> {
        self.tensor_quantized_storage(name)?.dequantize_q3_k()
    }

    pub fn tensor_q8_0_f32_values(&self, name: &str) -> Result<Vec<f32>> {
        self.tensor_quantized_storage(name)?.dequantize_q8_0()
    }

    pub fn metadata_string(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(GgufMetadataValue::String(value)) => Some(value),
            _ => None,
        }
    }

    pub fn metadata_unsigned(&self, key: &str) -> Option<u64> {
        metadata_value_as_unsigned(self.metadata.get(key)?)
    }
}

impl From<GgufTensorAdvice> for MappedFileAdvice {
    fn from(value: GgufTensorAdvice) -> Self {
        match value {
            GgufTensorAdvice::Random => Self::Random,
            GgufTensorAdvice::WillNeed => Self::WillNeed,
        }
    }
}

impl GgufMetadataValue {
    pub fn array_len(&self) -> Option<u64> {
        match self {
            Self::Array { len, .. } => Some(*len),
            _ => None,
        }
    }

    pub fn is_array_omitted(&self) -> bool {
        matches!(self, Self::Array { values: None, .. })
    }
}

impl<'a> GgufTensorStorage<'a> {
    pub fn quantized_payload(self) -> Result<GgufQuantizedTensorStorage<'a>> {
        let block = GgufQuantBlockKind::from_ggml_type(self.info.ty)?;
        let element_count = self.info.element_count()?;
        let block_count = block_count_for_quant_block(self.info, element_count, block)?;
        let block_bytes = block.block_byte_len();
        let payload_byte_len = block_count.checked_mul(block_bytes).ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} quantized payload byte count overflow",
                self.info.name
            ))
        })?;
        let payload_len = usize::try_from(payload_byte_len).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} quantized payload byte count does not fit usize",
                self.info.name
            ))
        })?;
        if self.bytes.len() < payload_len {
            return Err(Error::gguf(format!(
                "GGUF tensor {} storage has {} bytes but {} payload requires {payload_byte_len}",
                self.info.name,
                self.bytes.len(),
                block
            )));
        }

        Ok(GgufQuantizedTensorStorage {
            info: self.info,
            block,
            block_count,
            payload_byte_len,
            bytes: &self.bytes[..payload_len],
        })
    }
}

impl GgufQuantizedTensorStorage<'_> {
    pub fn matmul_rows_q2_k_f32(
        &self,
        input_rows: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Vec<f32>> {
        if self.block != GgufQuantBlockKind::Q2K {
            return Err(Error::gguf(format!(
                "GGUF tensor {} must be Q2_K for Q2 direct matmul, got {}",
                self.info.name, self.block
            )));
        }
        matmul_q2_k_payload_f32(
            &self.info.name,
            self.bytes,
            input_rows,
            row_count,
            in_features,
            out_features,
        )
    }

    pub fn matmul_rows_f32(
        &self,
        input_rows: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Vec<f32>> {
        let expected_input_values = row_count.checked_mul(in_features).ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} direct quantized matmul input value count overflow",
                self.info.name
            ))
        })?;
        if input_rows.len() != expected_input_values {
            return Err(Error::gguf(format!(
                "GGUF tensor {} direct quantized matmul expected {expected_input_values} input values, got {}",
                self.info.name,
                input_rows.len()
            )));
        }

        let values_per_block = usize::try_from(self.block.values_per_block()).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} quant block value count does not fit usize",
                self.info.name
            ))
        })?;
        if !in_features.is_multiple_of(values_per_block) {
            return Err(Error::gguf(format!(
                "GGUF tensor {} direct quantized matmul input width {in_features} must be divisible by {values_per_block}",
                self.info.name
            )));
        }
        let blocks_per_row = in_features / values_per_block;
        let expected_blocks = out_features.checked_mul(blocks_per_row).ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} direct quantized matmul block count overflow",
                self.info.name
            ))
        })?;
        if self.block_count
            != u64::try_from(expected_blocks).map_err(|_| {
                Error::gguf(format!(
                    "GGUF tensor {} direct quantized matmul block count does not fit u64",
                    self.info.name
                ))
            })?
        {
            return Err(Error::gguf(format!(
                "GGUF tensor {} has {} quant blocks but direct matmul shape requires {expected_blocks}",
                self.info.name, self.block_count
            )));
        }

        let output_len = row_count.checked_mul(out_features).ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} direct quantized matmul output value count overflow",
                self.info.name
            ))
        })?;
        let block_bytes = usize::try_from(self.block.block_byte_len()).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} block byte count does not fit usize",
                self.info.name
            ))
        })?;
        let mut output = vec![0.0_f32; output_len];
        let mut decoded_block = vec![0.0_f32; values_per_block];

        for output_feature in 0..out_features {
            let row_block_offset = output_feature.checked_mul(blocks_per_row).ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} direct quantized matmul row block offset overflow",
                    self.info.name
                ))
            })?;
            for block_in_row in 0..blocks_per_row {
                let block_index = row_block_offset.checked_add(block_in_row).ok_or_else(|| {
                    Error::gguf(format!(
                        "GGUF tensor {} direct quantized matmul block index overflow",
                        self.info.name
                    ))
                })?;
                let block_start = block_index.checked_mul(block_bytes).ok_or_else(|| {
                    Error::gguf(format!(
                        "GGUF tensor {} direct quantized matmul byte offset overflow",
                        self.info.name
                    ))
                })?;
                let block_end = block_start.checked_add(block_bytes).ok_or_else(|| {
                    Error::gguf(format!(
                        "GGUF tensor {} direct quantized matmul byte end overflow",
                        self.info.name
                    ))
                })?;
                if block_end > self.bytes.len() {
                    return Err(Error::gguf(format!(
                        "GGUF tensor {} direct quantized matmul block bytes [{block_start}..{block_end}) exceed payload length {}",
                        self.info.name,
                        self.bytes.len()
                    )));
                }

                dequantize_block(
                    self.block,
                    &self.bytes[block_start..block_end],
                    &mut decoded_block,
                )?;
                let input_feature_offset =
                    block_in_row.checked_mul(values_per_block).ok_or_else(|| {
                        Error::gguf(format!(
                            "GGUF tensor {} direct quantized matmul input offset overflow",
                            self.info.name
                        ))
                    })?;
                for row in 0..row_count {
                    let input_offset = row
                        .checked_mul(in_features)
                        .and_then(|offset| offset.checked_add(input_feature_offset))
                        .ok_or_else(|| {
                            Error::gguf(format!(
                                "GGUF tensor {} direct quantized matmul input row offset overflow",
                                self.info.name
                            ))
                        })?;
                    let output_offset = row
                        .checked_mul(out_features)
                        .and_then(|offset| offset.checked_add(output_feature))
                        .ok_or_else(|| {
                            Error::gguf(format!(
                                "GGUF tensor {} direct quantized matmul output row offset overflow",
                                self.info.name
                            ))
                        })?;
                    let input = &input_rows[input_offset..input_offset + values_per_block];
                    output[output_offset] += dot_f32(input, &decoded_block);
                }
            }
        }

        if output.iter().any(|value| !value.is_finite()) {
            return Err(Error::gguf(format!(
                "GGUF tensor {} direct quantized matmul produced non-finite values",
                self.info.name
            )));
        }
        Ok(output)
    }

    pub fn dequantize_block_range_as_f32(
        &self,
        block_start: u64,
        block_count: u64,
    ) -> Result<Vec<f32>> {
        let block_end = block_start.checked_add(block_count).ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} quantized block range overflow",
                self.info.name
            ))
        })?;
        if block_end > self.block_count {
            return Err(Error::gguf(format!(
                "GGUF tensor {} block range [{block_start}..{block_end}) exceeds block count {}",
                self.info.name, self.block_count
            )));
        }

        let block_bytes = usize::try_from(self.block.block_byte_len()).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} block byte count does not fit usize",
                self.info.name
            ))
        })?;
        let start = usize::try_from(block_start)
            .ok()
            .and_then(|block| block.checked_mul(block_bytes))
            .ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} block range start byte overflow",
                    self.info.name
                ))
            })?;
        let len = usize::try_from(block_count)
            .ok()
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} block range byte length overflow",
                    self.info.name
                ))
            })?;
        let end = start.checked_add(len).ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} block range end byte overflow",
                self.info.name
            ))
        })?;
        if end > self.bytes.len() {
            return Err(Error::gguf(format!(
                "GGUF tensor {} block range bytes [{start}..{end}) exceed payload length {}",
                self.info.name,
                self.bytes.len()
            )));
        }

        let values_per_block = usize::try_from(self.block.values_per_block()).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} quant block value count does not fit usize",
                self.info.name
            ))
        })?;
        let value_count = usize::try_from(block_count)
            .ok()
            .and_then(|blocks| blocks.checked_mul(values_per_block))
            .ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} decoded block range value count overflow",
                    self.info.name
                ))
            })?;
        let mut values = vec![0.0_f32; value_count];
        for (block, output) in self.bytes[start..end]
            .chunks_exact(block_bytes)
            .zip(values.chunks_exact_mut(values_per_block))
        {
            dequantize_block(self.block, block, output)?;
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::gguf(format!(
                "GGUF tensor {} dequantized block range to non-finite values",
                self.info.name
            )));
        }
        Ok(values)
    }

    pub fn dequantize_q2_k(&self) -> Result<Vec<f32>> {
        if self.block != GgufQuantBlockKind::Q2K {
            return Err(Error::gguf(format!(
                "GGUF tensor {} must be Q2_K for Q2_K dequantization, got {}",
                self.info.name, self.block
            )));
        }

        let element_count = usize::try_from(self.info.element_count()?).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} element count does not fit usize",
                self.info.name
            ))
        })?;
        let expected_len = usize::try_from(
            self.block_count
                .checked_mul(GGML_Q2_K_BLOCK_BYTES)
                .ok_or_else(|| {
                    Error::gguf(format!(
                        "GGUF tensor {} Q2_K byte count overflow",
                        self.info.name
                    ))
                })?,
        )
        .map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} Q2_K byte count does not fit usize",
                self.info.name
            ))
        })?;
        if self.bytes.len() != expected_len {
            return Err(Error::gguf(format!(
                "GGUF tensor {} has {} Q2_K payload bytes; expected {expected_len}",
                self.info.name,
                self.bytes.len()
            )));
        }

        let mut values = vec![0.0_f32; element_count];
        for (block, output) in self
            .bytes
            .chunks_exact(GGML_Q2_K_BLOCK_BYTES as usize)
            .zip(values.chunks_exact_mut(GGML_K_QUANT_BLOCK_SIZE as usize))
        {
            dequantize_q2_k_block(block, output)?;
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::gguf(format!(
                "GGUF tensor {} dequantized to non-finite Q2_K values",
                self.info.name
            )));
        }
        Ok(values)
    }

    pub fn dequantize_q3_k(&self) -> Result<Vec<f32>> {
        if self.block != GgufQuantBlockKind::Q3K {
            return Err(Error::gguf(format!(
                "GGUF tensor {} must be Q3_K for Q3_K dequantization, got {}",
                self.info.name, self.block
            )));
        }

        let element_count = usize::try_from(self.info.element_count()?).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} element count does not fit usize",
                self.info.name
            ))
        })?;
        let expected_len = usize::try_from(
            self.block_count
                .checked_mul(GGML_Q3_K_BLOCK_BYTES)
                .ok_or_else(|| {
                    Error::gguf(format!(
                        "GGUF tensor {} Q3_K byte count overflow",
                        self.info.name
                    ))
                })?,
        )
        .map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} Q3_K byte count does not fit usize",
                self.info.name
            ))
        })?;
        if self.bytes.len() != expected_len {
            return Err(Error::gguf(format!(
                "GGUF tensor {} has {} Q3_K payload bytes; expected {expected_len}",
                self.info.name,
                self.bytes.len()
            )));
        }

        let mut values = vec![0.0_f32; element_count];
        for (block, output) in self
            .bytes
            .chunks_exact(GGML_Q3_K_BLOCK_BYTES as usize)
            .zip(values.chunks_exact_mut(GGML_K_QUANT_BLOCK_SIZE as usize))
        {
            dequantize_q3_k_block(block, output)?;
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::gguf(format!(
                "GGUF tensor {} dequantized to non-finite Q3_K values",
                self.info.name
            )));
        }
        Ok(values)
    }

    pub fn dequantize_q8_0(&self) -> Result<Vec<f32>> {
        if self.block != GgufQuantBlockKind::Q8_0 {
            return Err(Error::gguf(format!(
                "GGUF tensor {} must be Q8_0 for Q8_0 dequantization, got {}",
                self.info.name, self.block
            )));
        }

        let element_count = usize::try_from(self.info.element_count()?).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} element count does not fit usize",
                self.info.name
            ))
        })?;
        let expected_len = usize::try_from(
            self.block_count
                .checked_mul(GGML_Q8_0_BLOCK_BYTES)
                .ok_or_else(|| {
                    Error::gguf(format!(
                        "GGUF tensor {} Q8_0 byte count overflow",
                        self.info.name
                    ))
                })?,
        )
        .map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} Q8_0 byte count does not fit usize",
                self.info.name
            ))
        })?;
        if self.bytes.len() != expected_len {
            return Err(Error::gguf(format!(
                "GGUF tensor {} has {} Q8_0 payload bytes; expected {expected_len}",
                self.info.name,
                self.bytes.len()
            )));
        }

        let mut values = vec![0.0_f32; element_count];
        for (block, output) in self
            .bytes
            .chunks_exact(GGML_Q8_0_BLOCK_BYTES as usize)
            .zip(values.chunks_exact_mut(GGML_Q8_0_BLOCK_SIZE as usize))
        {
            dequantize_q8_0_block(block, output)?;
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::gguf(format!(
                "GGUF tensor {} dequantized to non-finite Q8_0 values",
                self.info.name
            )));
        }
        Ok(values)
    }
}

pub fn matmul_q2_k_payload_f32(
    context: &str,
    bytes: &[u8],
    input_rows: &[f32],
    row_count: usize,
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f32>> {
    matmul_quantized_payload_rows_f32(
        context,
        GgufQuantBlockKind::Q2K,
        bytes,
        input_rows,
        row_count,
        in_features,
        out_features,
    )
}

fn matmul_quantized_payload_rows_f32(
    context: &str,
    block: GgufQuantBlockKind,
    bytes: &[u8],
    input_rows: &[f32],
    row_count: usize,
    in_features: usize,
    out_features: usize,
) -> Result<Vec<f32>> {
    let expected_input_values = row_count.checked_mul(in_features).ok_or_else(|| {
        Error::gguf(format!(
            "GGUF tensor {context} direct quantized matmul input value count overflow"
        ))
    })?;
    if input_rows.len() != expected_input_values {
        return Err(Error::gguf(format!(
            "GGUF tensor {context} direct quantized matmul expected {expected_input_values} input values, got {}",
            input_rows.len()
        )));
    }

    let values_per_block = usize::try_from(block.values_per_block()).map_err(|_| {
        Error::gguf(format!(
            "GGUF tensor {context} quant block value count does not fit usize"
        ))
    })?;
    if !in_features.is_multiple_of(values_per_block) {
        return Err(Error::gguf(format!(
            "GGUF tensor {context} direct quantized matmul input width {in_features} must be divisible by {values_per_block}"
        )));
    }
    let blocks_per_row = in_features / values_per_block;
    let expected_blocks = out_features.checked_mul(blocks_per_row).ok_or_else(|| {
        Error::gguf(format!(
            "GGUF tensor {context} direct quantized matmul block count overflow"
        ))
    })?;
    let block_bytes = usize::try_from(block.block_byte_len()).map_err(|_| {
        Error::gguf(format!(
            "GGUF tensor {context} block byte count does not fit usize"
        ))
    })?;
    let expected_bytes = expected_blocks.checked_mul(block_bytes).ok_or_else(|| {
        Error::gguf(format!(
            "GGUF tensor {context} direct quantized matmul byte count overflow"
        ))
    })?;
    if bytes.len() != expected_bytes {
        return Err(Error::gguf(format!(
            "GGUF tensor {context} direct quantized matmul expected {expected_bytes} payload bytes, got {}",
            bytes.len()
        )));
    }

    let output_len = row_count.checked_mul(out_features).ok_or_else(|| {
        Error::gguf(format!(
            "GGUF tensor {context} direct quantized matmul output value count overflow"
        ))
    })?;
    let mut output = vec![0.0_f32; output_len];
    let mut decoded_block = vec![0.0_f32; values_per_block];

    for output_feature in 0..out_features {
        let row_block_offset = output_feature.checked_mul(blocks_per_row).ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {context} direct quantized matmul row block offset overflow"
            ))
        })?;
        for block_in_row in 0..blocks_per_row {
            let block_index = row_block_offset.checked_add(block_in_row).ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {context} direct quantized matmul block index overflow"
                ))
            })?;
            let block_start = block_index.checked_mul(block_bytes).ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {context} direct quantized matmul byte offset overflow"
                ))
            })?;
            let block_end = block_start.checked_add(block_bytes).ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {context} direct quantized matmul byte end overflow"
                ))
            })?;

            dequantize_block(block, &bytes[block_start..block_end], &mut decoded_block)?;
            let input_feature_offset =
                block_in_row.checked_mul(values_per_block).ok_or_else(|| {
                    Error::gguf(format!(
                        "GGUF tensor {context} direct quantized matmul input offset overflow"
                    ))
                })?;
            for row in 0..row_count {
                let input_offset = row
                    .checked_mul(in_features)
                    .and_then(|offset| offset.checked_add(input_feature_offset))
                    .ok_or_else(|| {
                        Error::gguf(format!(
                            "GGUF tensor {context} direct quantized matmul input row offset overflow"
                        ))
                    })?;
                let output_offset = row
                    .checked_mul(out_features)
                    .and_then(|offset| offset.checked_add(output_feature))
                    .ok_or_else(|| {
                        Error::gguf(format!(
                            "GGUF tensor {context} direct quantized matmul output row offset overflow"
                        ))
                    })?;
                let mut sum = 0.0_f32;
                for value_index in 0..values_per_block {
                    sum += input_rows[input_offset + value_index] * decoded_block[value_index];
                }
                output[output_offset] += sum;
            }
        }
    }

    Ok(output)
}

fn dequantize_block(kind: GgufQuantBlockKind, block: &[u8], output: &mut [f32]) -> Result<()> {
    match kind {
        GgufQuantBlockKind::Q2K => dequantize_q2_k_block(block, output),
        GgufQuantBlockKind::Q3K => dequantize_q3_k_block(block, output),
        GgufQuantBlockKind::Q8_0 => dequantize_q8_0_block(block, output),
    }
}

fn dot_f32(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

impl GgufQuantBlockKind {
    pub fn from_ggml_type(ty: GgmlType) -> Result<Self> {
        match ty {
            GgmlType::Q2K => Ok(Self::Q2K),
            GgmlType::Q3K => Ok(Self::Q3K),
            GgmlType::Q8_0 => Ok(Self::Q8_0),
            other => Err(Error::gguf(format!(
                "GGUF quant block reader supports Q2_K, Q3_K, and Q8_0 only, got {other}"
            ))),
        }
    }

    pub const fn block_byte_len(self) -> u64 {
        match self {
            Self::Q2K => GGML_Q2_K_BLOCK_BYTES,
            Self::Q3K => GGML_Q3_K_BLOCK_BYTES,
            Self::Q8_0 => GGML_Q8_0_BLOCK_BYTES,
        }
    }

    pub const fn values_per_block(self) -> u64 {
        match self {
            Self::Q2K | Self::Q3K => GGML_K_QUANT_BLOCK_SIZE,
            Self::Q8_0 => GGML_Q8_0_BLOCK_SIZE,
        }
    }
}

fn dequantize_q2_k_block(block: &[u8], output: &mut [f32]) -> Result<()> {
    let block_len = usize::try_from(GGML_Q2_K_BLOCK_BYTES)
        .map_err(|_| Error::gguf("Q2_K block byte size does not fit usize"))?;
    let value_count = usize::try_from(GGML_K_QUANT_BLOCK_SIZE)
        .map_err(|_| Error::gguf("K-quant block size does not fit usize"))?;
    if block.len() != block_len {
        return Err(Error::gguf(format!(
            "Q2_K block must contain {block_len} bytes, got {}",
            block.len()
        )));
    }
    if output.len() != value_count {
        return Err(Error::gguf(format!(
            "Q2_K output block must contain {value_count} values, got {}",
            output.len()
        )));
    }

    let scales = &block[..GGML_Q2_K_SCALE_BYTES];
    let quants = &block[GGML_Q2_K_SCALE_BYTES..GGML_Q2_K_SCALE_BYTES + GGML_Q2_K_QUANT_BYTES];
    let scale_offset = GGML_Q2_K_SCALE_BYTES + GGML_Q2_K_QUANT_BYTES;
    let d = f16_to_f32(u16::from_le_bytes([
        block[scale_offset],
        block[scale_offset + 1],
    ]));
    let min = f16_to_f32(u16::from_le_bytes([
        block[scale_offset + 2],
        block[scale_offset + 3],
    ]));
    if !d.is_finite() || !min.is_finite() {
        return Err(Error::gguf("Q2_K block contains non-finite scale values"));
    }

    let mut scale_index = 0;
    let mut quant_offset = 0;
    let mut output_offset = 0;
    while output_offset < value_count {
        let mut shift = 0;
        for _ in 0..4 {
            let scale_min = scales[scale_index];
            scale_index += 1;
            let scale = d * (scale_min & 0x0f) as f32;
            let min_offset = min * (scale_min >> 4) as f32;
            for value_index in 0..16 {
                output[output_offset + value_index] = scale
                    * ((quants[quant_offset + value_index] >> shift) & 0x03) as f32
                    - min_offset;
            }
            output_offset += 16;

            let scale_min = scales[scale_index];
            scale_index += 1;
            let scale = d * (scale_min & 0x0f) as f32;
            let min_offset = min * (scale_min >> 4) as f32;
            for value_index in 0..16 {
                output[output_offset + value_index] = scale
                    * ((quants[quant_offset + 16 + value_index] >> shift) & 0x03) as f32
                    - min_offset;
            }
            output_offset += 16;

            shift += 2;
        }
        quant_offset += 32;
    }

    Ok(())
}

fn dequantize_q3_k_block(block: &[u8], output: &mut [f32]) -> Result<()> {
    let block_len = usize::try_from(GGML_Q3_K_BLOCK_BYTES)
        .map_err(|_| Error::gguf("Q3_K block byte size does not fit usize"))?;
    let value_count = usize::try_from(GGML_K_QUANT_BLOCK_SIZE)
        .map_err(|_| Error::gguf("K-quant block size does not fit usize"))?;
    if block.len() != block_len {
        return Err(Error::gguf(format!(
            "Q3_K block must contain {block_len} bytes, got {}",
            block.len()
        )));
    }
    if output.len() != value_count {
        return Err(Error::gguf(format!(
            "Q3_K output block must contain {value_count} values, got {}",
            output.len()
        )));
    }

    let high_masks = &block[..GGML_Q3_K_HIGH_MASK_BYTES];
    let quant_start = GGML_Q3_K_HIGH_MASK_BYTES;
    let quants = &block[quant_start..quant_start + GGML_Q3_K_QUANT_BYTES];
    let scale_start = quant_start + GGML_Q3_K_QUANT_BYTES;
    let scales = &block[scale_start..scale_start + GGML_Q3_K_SCALE_BYTES];
    let d_offset = scale_start + GGML_Q3_K_SCALE_BYTES;
    let d = f16_to_f32(u16::from_le_bytes([block[d_offset], block[d_offset + 1]]));
    if !d.is_finite() {
        return Err(Error::gguf("Q3_K block contains a non-finite scale"));
    }

    for group in 0..16_usize {
        let quant_offset = 32 * (group / 8) + 16 * (group & 1);
        let high_offset = 16 * (group & 1);
        let high_bit = 1_u8 << (group / 2);

        let scale_low_mask = match group / 4 {
            0 => 0x03_u16,
            1 => 0x0c_u16,
            2 => 0x30_u16,
            _ => 0xc0_u16,
        };
        let scale_nibble_mask = if group < 8 { 0x0f_u16 } else { 0xf0_u16 };
        let scale_low = scales[group % 8] as u16;
        let scale_high = scales[8 + group % 4] as u16;
        let packed_scale = if (group / 4) & 1 == 1 {
            (scale_low & scale_nibble_mask) | ((scale_high & scale_low_mask) << 2)
        } else {
            (scale_low & scale_nibble_mask) | ((scale_high & scale_low_mask) << 4)
        };
        let group_scale = if group < 8 {
            d * (packed_scale as f32 - 32.0)
        } else {
            d * (packed_scale as f32 / 16.0 - 32.0)
        };
        let negative_offset = 4.0 * group_scale;

        let quant_lane = (group / 2) & 3;
        let shift = 2 * quant_lane;
        let quant_mask = 0x03_u8 << shift;
        let quant_scale = 1.0 / (1_u32 << shift) as f32;
        let output_offset = group * 16;
        for index in 0..16 {
            let low = (quants[quant_offset + index] & quant_mask) as f32 * quant_scale;
            let subtract = if high_masks[high_offset + index] & high_bit == 0 {
                negative_offset
            } else {
                0.0
            };
            output[output_offset + index] = group_scale * low - subtract;
        }
    }

    Ok(())
}

fn dequantize_q8_0_block(block: &[u8], output: &mut [f32]) -> Result<()> {
    let block_len = usize::try_from(GGML_Q8_0_BLOCK_BYTES)
        .map_err(|_| Error::gguf("Q8_0 block byte size does not fit usize"))?;
    let value_count = usize::try_from(GGML_Q8_0_BLOCK_SIZE)
        .map_err(|_| Error::gguf("Q8_0 block size does not fit usize"))?;
    if block.len() != block_len {
        return Err(Error::gguf(format!(
            "Q8_0 block must contain {block_len} bytes, got {}",
            block.len()
        )));
    }
    if output.len() != value_count {
        return Err(Error::gguf(format!(
            "Q8_0 output block must contain {value_count} values, got {}",
            output.len()
        )));
    }

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    if !d.is_finite() {
        return Err(Error::gguf("Q8_0 block contains non-finite scale"));
    }
    for (output, quantized) in output.iter_mut().zip(&block[2..]) {
        *output = d * (*quantized as i8) as f32;
    }
    Ok(())
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let fraction = bits & 0x03ff;

    let out = match exponent {
        0 => {
            if fraction == 0 {
                sign
            } else {
                let mut mantissa = fraction as u32;
                let mut exp = -14_i32;
                while mantissa & 0x0400 == 0 {
                    mantissa <<= 1;
                    exp -= 1;
                }
                mantissa &= 0x03ff;
                sign | (((exp + 127) as u32) << 23) | (mantissa << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | ((fraction as u32) << 13),
        _ => {
            let exp = (exponent as u32) + 112;
            sign | (exp << 23) | ((fraction as u32) << 13)
        }
    };
    f32::from_bits(out)
}

impl fmt::Display for GgufQuantBlockKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Q2K => f.write_str("Q2_K"),
            Self::Q3K => f.write_str("Q3_K"),
            Self::Q8_0 => f.write_str("Q8_0"),
        }
    }
}

impl GgufTensorInfo {
    pub fn element_count(&self) -> Result<u64> {
        self.dims.iter().try_fold(1_u64, |acc, dim| {
            acc.checked_mul(*dim).ok_or_else(|| {
                Error::gguf(format!("GGUF tensor {} element count overflow", self.name))
            })
        })
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

impl GgufMetadataValueType {
    fn from_code(code: u32) -> Result<Self> {
        match code {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            other => Err(Error::gguf(format!(
                "unsupported GGUF metadata value type {other}"
            ))),
        }
    }
}

impl GgmlType {
    pub fn from_code(code: u32) -> Self {
        match code {
            0 => Self::F32,
            8 => Self::Q8_0,
            10 => Self::Q2K,
            11 => Self::Q3K,
            other => Self::Unsupported(other),
        }
    }

    pub const fn code(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::Q8_0 => 8,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Unsupported(code) => code,
        }
    }

    pub const fn is_quantized(self) -> bool {
        matches!(self, Self::Q2K | Self::Q3K | Self::Q8_0)
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::F32 => "F32",
            Self::Q8_0 => "Q8_0",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Unsupported(code) => return write!(f, "UNSUPPORTED_GGML_TYPE_{code}"),
        };
        f.write_str(name)
    }
}

struct GgufReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> GgufReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn position_u64(&self) -> Result<u64> {
        self.position
            .try_into()
            .map_err(|_| Error::gguf("GGUF reader position does not fit u64"))
    }

    fn expect_magic(&mut self) -> Result<()> {
        let magic = self.read_exact(4)?;
        if magic != GGUF_MAGIC {
            return Err(Error::gguf("invalid GGUF magic"));
        }
        Ok(())
    }

    fn read_metadata_type(&mut self) -> Result<GgufMetadataValueType> {
        GgufMetadataValueType::from_code(self.read_u32()?)
    }

    fn read_metadata_value(
        &mut self,
        ty: GgufMetadataValueType,
        store: bool,
        depth: usize,
    ) -> Result<GgufMetadataValue> {
        if depth > MAX_METADATA_DEPTH {
            return Err(Error::gguf("GGUF metadata array nesting is too deep"));
        }

        match ty {
            GgufMetadataValueType::Uint8 => Ok(GgufMetadataValue::Uint8(self.read_u8()?)),
            GgufMetadataValueType::Int8 => Ok(GgufMetadataValue::Int8(self.read_i8()?)),
            GgufMetadataValueType::Uint16 => Ok(GgufMetadataValue::Uint16(self.read_u16()?)),
            GgufMetadataValueType::Int16 => Ok(GgufMetadataValue::Int16(self.read_i16()?)),
            GgufMetadataValueType::Uint32 => Ok(GgufMetadataValue::Uint32(self.read_u32()?)),
            GgufMetadataValueType::Int32 => Ok(GgufMetadataValue::Int32(self.read_i32()?)),
            GgufMetadataValueType::Float32 => Ok(GgufMetadataValue::Float32(self.read_f32()?)),
            GgufMetadataValueType::Bool => {
                let value = self.read_u8()?;
                match value {
                    0 => Ok(GgufMetadataValue::Bool(false)),
                    1 => Ok(GgufMetadataValue::Bool(true)),
                    _ => Err(Error::gguf(format!(
                        "invalid GGUF bool value {value}; expected 0 or 1"
                    ))),
                }
            }
            GgufMetadataValueType::String => {
                if store {
                    Ok(GgufMetadataValue::String(self.read_string()?))
                } else {
                    self.skip_string()?;
                    Ok(GgufMetadataValue::String(String::new()))
                }
            }
            GgufMetadataValueType::Array => {
                let element_type = self.read_metadata_type()?;
                let len = self.read_u64()?;
                let store_values = store && len <= MAX_STORED_ARRAY_VALUES;
                let mut values = if store_values {
                    Some(Vec::with_capacity(len as usize))
                } else {
                    None
                };

                for _ in 0..len {
                    let value = self.read_metadata_value(element_type, store_values, depth + 1)?;
                    if let Some(values) = &mut values {
                        values.push(value);
                    }
                }

                Ok(GgufMetadataValue::Array {
                    element_type,
                    len,
                    values,
                })
            }
            GgufMetadataValueType::Uint64 => Ok(GgufMetadataValue::Uint64(self.read_u64()?)),
            GgufMetadataValueType::Int64 => Ok(GgufMetadataValue::Int64(self.read_i64()?)),
            GgufMetadataValueType::Float64 => Ok(GgufMetadataValue::Float64(self.read_f64()?)),
        }
    }

    fn read_limited_string(&mut self, max_bytes: usize) -> Result<String> {
        let start = self.position;
        let value = self.read_string()?;
        if value.len() > max_bytes {
            return Err(Error::gguf(format!(
                "GGUF string at byte {start} has {} bytes; max is {max_bytes}",
                value.len()
            )));
        }
        Ok(value)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()?;
        let len = usize::try_from(len)
            .map_err(|_| Error::gguf("GGUF string length does not fit usize"))?;
        let bytes = self.read_exact(len)?;
        std::str::from_utf8(bytes)
            .map(str::to_string)
            .map_err(|source| Error::gguf(format!("GGUF string is not UTF-8: {source}")))
    }

    fn skip_string(&mut self) -> Result<()> {
        let len = self.read_u64()?;
        let len = usize::try_from(len)
            .map_err(|_| Error::gguf("GGUF string length does not fit usize"))?;
        self.skip(len)
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| Error::gguf("GGUF reader position overflow"))?;
        if end > self.bytes.len() {
            return Err(Error::gguf(format!(
                "GGUF file ended at byte {}; needed bytes [{}..{})",
                self.bytes.len(),
                self.position,
                end
            )));
        }
        let slice = &self.bytes[self.position..end];
        self.position = end;
        Ok(slice)
    }

    fn skip(&mut self, len: usize) -> Result<()> {
        self.read_exact(len).map(|_| ())
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_exact(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_i16(&mut self) -> Result<i16> {
        let bytes = self.read_exact(2)?;
        Ok(i16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_exact(4)?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_i32(&mut self) -> Result<i32> {
        let bytes = self.read_exact(4)?;
        Ok(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.read_u32()?))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_exact(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_i64(&mut self) -> Result<i64> {
        let bytes = self.read_exact(8)?;
        Ok(i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.read_u64()?))
    }
}

fn metadata_alignment(metadata: &BTreeMap<String, GgufMetadataValue>) -> Result<u64> {
    let alignment = metadata
        .get("general.alignment")
        .and_then(metadata_value_as_unsigned)
        .unwrap_or(DEFAULT_GGUF_ALIGNMENT);
    if alignment < 8 || !alignment.is_multiple_of(8) {
        return Err(Error::gguf(format!(
            "GGUF general.alignment must be a multiple of 8 and at least 8, got {alignment}"
        )));
    }
    Ok(alignment)
}

fn metadata_value_as_unsigned(value: &GgufMetadataValue) -> Option<u64> {
    match value {
        GgufMetadataValue::Uint8(value) => Some(*value as u64),
        GgufMetadataValue::Uint16(value) => Some(*value as u64),
        GgufMetadataValue::Uint32(value) => Some(*value as u64),
        GgufMetadataValue::Uint64(value) => Some(*value),
        _ => None,
    }
}

fn align_offset(offset: u64, alignment: u64) -> Result<u64> {
    let remainder = offset % alignment;
    if remainder == 0 {
        return Ok(offset);
    }
    offset
        .checked_add(alignment - remainder)
        .ok_or_else(|| Error::gguf("GGUF aligned offset overflow"))
}

fn assign_storage_byte_lengths(tensors: &mut [GgufTensorInfo], file_size: u64) -> Result<()> {
    let mut ordered_indices = (0..tensors.len()).collect::<Vec<_>>();
    ordered_indices.sort_by_key(|index| tensors[*index].absolute_offset);

    for window_index in 0..ordered_indices.len() {
        let tensor_index = ordered_indices[window_index];
        let start = tensors[tensor_index].absolute_offset;
        let end = ordered_indices
            .get(window_index + 1)
            .map(|next_index| tensors[*next_index].absolute_offset)
            .unwrap_or(file_size);
        if end <= start {
            return Err(Error::gguf(format!(
                "GGUF tensor {} has invalid storage range [{}..{})",
                tensors[tensor_index].name, start, end
            )));
        }
        tensors[tensor_index].storage_byte_len = end - start;
    }

    Ok(())
}

fn block_count_for_quant_block(
    info: &GgufTensorInfo,
    element_count: u64,
    block: GgufQuantBlockKind,
) -> Result<u64> {
    let values_per_block = block.values_per_block();
    let row_width = info
        .dims
        .first()
        .copied()
        .ok_or_else(|| Error::gguf(format!("GGUF tensor {} has no dimensions", info.name)))?;
    if row_width % values_per_block != 0 {
        return Err(Error::gguf(format!(
            "GGUF tensor {} first dimension {row_width} is not divisible by {block} block size {values_per_block}",
            info.name
        )));
    }
    if !element_count.is_multiple_of(values_per_block) {
        return Err(Error::gguf(format!(
            "GGUF tensor {} element count {element_count} is not divisible by {block} block size {values_per_block}",
            info.name
        )));
    }
    Ok(element_count / values_per_block)
}

fn validate_metadata_key(key: &str) -> Result<()> {
    if !key.is_ascii() {
        return Err(Error::gguf(format!(
            "GGUF metadata key {key:?} must be ASCII"
        )));
    }
    if key.is_empty() || key.starts_with('.') || key.ends_with('.') || key.contains("..") {
        return Err(Error::gguf(format!(
            "GGUF metadata key {key:?} must be hierarchical"
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

    use super::*;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn parses_header_metadata_and_tensor_directory() {
        let path = unique_temp_file("valid");
        fs::write(&path, tiny_gguf()).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let summary = gguf.summary();

        assert_eq!(summary.version, GGUF_VERSION_V3);
        assert_eq!(summary.tensor_count, 2);
        assert_eq!(summary.metadata_kv_count, 5);
        assert_eq!(summary.alignment, 32);
        assert_eq!(summary.architecture.as_deref(), Some("glm-dsa"));
        assert_eq!(summary.quantization_version, Some(2));
        assert_eq!(summary.file_type, Some(12));
        assert_eq!(summary.tensor_type_counts.len(), 2);

        let q = gguf.tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(q.dims, vec![32]);
        assert_eq!(q.ty, GgmlType::Q8_0);
        assert_eq!(q.relative_offset, 0);
        assert_eq!(q.absolute_offset % 32, 0);
        assert_eq!(q.storage_byte_len, 64);
        assert_eq!(q.element_count().unwrap(), 32);
        assert_eq!(q.rank(), 1);

        let tokens = gguf.metadata().get("tokenizer.ggml.tokens").unwrap();
        assert_eq!(tokens.array_len(), Some(1025));
        assert!(tokens.is_array_omitted());
    }

    #[test]
    fn tensor_storage_returns_checked_mapped_payload_range() {
        let path = unique_temp_file("storage");
        fs::write(&path, tiny_gguf()).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let q = gguf.tensor_storage("blk.0.attn_q.weight").unwrap();
        let embedding = gguf.tensor_storage("token_embd.weight").unwrap();

        assert_eq!(q.info.ty, GgmlType::Q8_0);
        assert_eq!(q.bytes.len(), 64);
        assert_eq!(q.bytes[0], 0);
        assert_eq!(q.bytes[63], 63);
        assert_eq!(embedding.info.ty, GgmlType::Q2K);
        assert_eq!(embedding.bytes.len(), 96);
        assert_eq!(embedding.bytes[0], 64);
        assert_eq!(embedding.bytes[95], 159);
    }

    #[test]
    fn tensor_storage_rejects_missing_tensor() {
        let path = unique_temp_file("missing-storage");
        fs::write(&path, tiny_gguf()).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let err = gguf
            .tensor_storage("missing.weight")
            .expect_err("missing tensor storage should fail");

        assert!(err.to_string().contains("missing GGUF tensor"));
    }

    #[test]
    fn tensor_f32_values_decodes_raw_payload_without_padding() {
        let path = unique_temp_file("f32");
        fs::write(&path, tiny_gguf_with_f32_tensor()).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let values = gguf.tensor_f32_values("output_norm.weight").unwrap();

        assert_eq!(values, vec![1.0, -2.0, 3.5, 0.25]);
    }

    #[test]
    fn tensor_f32_values_rejects_quantized_tensor() {
        let path = unique_temp_file("f32-wrong-type");
        fs::write(&path, tiny_gguf()).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let err = gguf
            .tensor_f32_values("token_embd.weight")
            .expect_err("quantized tensor should not decode as raw F32");

        assert!(err.to_string().contains("must be F32"));
    }

    #[test]
    fn tensor_quantized_storage_validates_supported_payload_sizes() {
        let path = unique_temp_file("k-quant");
        fs::write(&path, tiny_supported_quant_gguf()).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let q2 = gguf.tensor_quantized_storage("q2.weight").unwrap();
        let q8 = gguf.tensor_quantized_storage("q8.weight").unwrap();

        assert_eq!(q2.block, GgufQuantBlockKind::Q2K);
        assert_eq!(q2.block_count, 1);
        assert_eq!(q2.payload_byte_len, GGML_Q2_K_BLOCK_BYTES);
        assert_eq!(q2.bytes.len(), GGML_Q2_K_BLOCK_BYTES as usize);
        assert_eq!(q2.bytes[0], 0x22);
        assert_eq!(q8.block, GgufQuantBlockKind::Q8_0);
        assert_eq!(q8.block_count, 1);
        assert_eq!(q8.payload_byte_len, GGML_Q8_0_BLOCK_BYTES);
        assert_eq!(q8.bytes.len(), GGML_Q8_0_BLOCK_BYTES as usize);
        assert_eq!(q8.bytes[0], 0x88);
    }

    #[test]
    fn tensor_q8_0_f32_values_decodes_standard_payload() {
        let path = unique_temp_file("q8-values");
        fs::write(&path, tiny_q8_0_values_gguf()).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let storage = gguf.tensor_quantized_storage("q8.weight").unwrap();
        let values = gguf.tensor_q8_0_f32_values("q8.weight").unwrap();

        assert_eq!(storage.block, GgufQuantBlockKind::Q8_0);
        assert_eq!(storage.block_count, 1);
        assert_eq!(storage.payload_byte_len, GGML_Q8_0_BLOCK_BYTES);
        assert_eq!(values.len(), 32);
        assert_eq!(values[0], -4.0);
        assert_eq!(values[1], -3.5);
        assert_eq!(values[16], 4.0);
        assert_eq!(values[31], 11.5);
    }

    #[test]
    fn tensor_q2_k_f32_values_decodes_standard_payload() {
        let path = unique_temp_file("q2-values");
        fs::write(&path, tiny_q2_k_values_gguf()).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let values = gguf.tensor_q2_k_f32_values("q2.weight").unwrap();

        assert_eq!(values.len(), 256);
        assert_eq!(values[0], 0.0);
        assert_eq!(values[16], -0.5);
        assert_eq!(values[32], 2.0);
        assert_eq!(values[48], 2.5);
        assert_eq!(values[64], 8.0);
        assert_eq!(values[80], 9.5);
        assert_eq!(values[96], 18.0);
        assert_eq!(values[112], 20.5);
        assert_eq!(values[128], -4.0);
        assert_eq!(values[144], -4.5);
        assert_eq!(values[160], 6.0);
        assert_eq!(values[176], 6.5);
        assert_eq!(values[192], 20.0);
        assert_eq!(values[208], 21.5);
        assert_eq!(values[224], 38.0);
        assert_eq!(values[240], -7.5);
    }

    #[test]
    fn tensor_q3_k_f32_values_decodes_standard_payload() {
        let path = unique_temp_file("q3-values");
        fs::write(&path, tiny_q3_k_values_gguf()).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let storage = gguf.tensor_quantized_storage("q3.weight").unwrap();
        let values = gguf.tensor_q3_k_f32_values("q3.weight").unwrap();

        assert_eq!(storage.block, GgufQuantBlockKind::Q3K);
        assert_eq!(storage.block_count, 1);
        assert_eq!(storage.payload_byte_len, GGML_Q3_K_BLOCK_BYTES);
        assert_eq!(values.len(), 256);
        for (group, expected) in [
            0.0_f32, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(values[group * 16], expected);
            assert_eq!(values[group * 16 + 15], expected);
        }
    }

    #[test]
    fn quantized_storage_direct_matmul_matches_dequantized_reference() {
        for (name, bytes) in [
            ("q2.weight", tiny_q2_k_values_gguf()),
            ("q3.weight", tiny_q3_k_values_gguf()),
            ("q8.weight", tiny_q8_0_values_gguf()),
        ] {
            let path = unique_temp_file("direct-matmul");
            fs::write(&path, bytes).unwrap();
            let gguf = GgufFile::open(&path).unwrap();
            let storage = gguf.tensor_quantized_storage(name).unwrap();
            let in_features = usize::try_from(storage.info.dims[0]).unwrap();
            let out_features =
                usize::try_from(storage.info.element_count().unwrap()).unwrap() / in_features;
            let input_rows = [
                (0..in_features)
                    .map(|index| (index % 7) as f32 - 3.0)
                    .collect::<Vec<_>>(),
                (0..in_features)
                    .map(|index| (index % 5) as f32 * 0.25)
                    .collect::<Vec<_>>(),
            ]
            .concat();

            let direct = storage
                .matmul_rows_f32(&input_rows, 2, in_features, out_features)
                .unwrap();
            let decoded = storage
                .dequantize_block_range_as_f32(0, storage.block_count)
                .unwrap();
            let mut reference = vec![0.0_f32; 2 * out_features];
            for row in 0..2 {
                for output_feature in 0..out_features {
                    let weight_offset = output_feature * in_features;
                    let input_offset = row * in_features;
                    reference[row * out_features + output_feature] = dot_f32(
                        &input_rows[input_offset..input_offset + in_features],
                        &decoded[weight_offset..weight_offset + in_features],
                    );
                }
            }

            assert_eq!(direct.len(), reference.len());
            for (actual, expected) in direct.iter().zip(reference) {
                assert!((actual - expected).abs() <= 1e-4);
            }
        }
    }

    #[test]
    fn q2_direct_matmul_rejects_non_q2_payload() {
        let path = unique_temp_file("q2-direct-from-q8");
        fs::write(&path, tiny_q8_0_values_gguf()).unwrap();
        let gguf = GgufFile::open(&path).unwrap();
        let storage = gguf.tensor_quantized_storage("q8.weight").unwrap();
        let input = vec![1.0_f32; 32];

        let err = storage
            .matmul_rows_q2_k_f32(&input, 1, 32, 1)
            .expect_err("Q2 direct matmul must reject non-Q2 tensors");

        assert!(err.to_string().contains("must be Q2_K"));
    }

    #[test]
    fn quantized_storage_decodes_selected_block_range() {
        let path = unique_temp_file("q2-block-range");
        fs::write(&path, tiny_q2_k_multi_block_values_gguf()).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let storage = gguf.tensor_quantized_storage("q2.weight").unwrap();
        let values = storage.dequantize_block_range_as_f32(1, 1).unwrap();

        assert_eq!(values.len(), 256);
        assert_eq!(values[0], 3.0);
        assert_eq!(values[255], 3.0);
    }

    #[test]
    fn quantized_storage_rejects_out_of_range_block_range() {
        let path = unique_temp_file("q2-block-range-bad");
        fs::write(&path, tiny_q2_k_multi_block_values_gguf()).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let storage = gguf.tensor_quantized_storage("q2.weight").unwrap();
        let err = storage
            .dequantize_block_range_as_f32(1, 2)
            .expect_err("block range should exceed available blocks");

        assert!(err.to_string().contains("exceeds block count"));
    }

    #[test]
    fn tensor_quantized_storage_rejects_unsupported_type() {
        let path = unique_temp_file("k-quant-f32");
        fs::write(&path, tiny_gguf_with_f32_tensor()).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let err = gguf
            .tensor_quantized_storage("output_norm.weight")
            .expect_err("F32 tensor should not decode as K-quant");

        assert!(err.to_string().contains("Q2_K, Q3_K, and Q8_0 only"));
    }

    #[test]
    fn tensor_quantized_storage_rejects_bad_k_block_shape() {
        let path = unique_temp_file("bad-k-shape");
        fs::write(&path, tiny_bad_k_quant_shape_gguf()).unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let err = gguf
            .tensor_quantized_storage("bad_q2.weight")
            .expect_err("bad K-quant shape should fail");

        assert!(err.to_string().contains("first dimension"));
    }

    #[test]
    fn rejects_invalid_magic() {
        let path = unique_temp_file("bad-magic");
        let mut bytes = tiny_gguf();
        bytes[0] = b'X';
        fs::write(&path, bytes).unwrap();

        let err = GgufFile::open(&path).expect_err("bad magic should fail");

        assert!(err.to_string().contains("invalid GGUF magic"));
    }

    #[test]
    fn rejects_unaligned_tensor_offset() {
        let path = unique_temp_file("unaligned");
        fs::write(&path, tiny_gguf_with_second_offset(33)).unwrap();

        let err = GgufFile::open(&path).expect_err("unaligned tensor should fail");

        assert!(err.to_string().contains("not aligned"));
    }

    #[test]
    fn rejects_bad_alignment_metadata() {
        let path = unique_temp_file("bad-alignment");
        let mut writer = GgufWriter::new();
        writer.header(0, 1);
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(7);
        fs::write(&path, writer.finish()).unwrap();

        let err = GgufFile::open(&path).expect_err("bad alignment should fail");

        assert!(err.to_string().contains("general.alignment"));
    }

    fn tiny_gguf() -> Vec<u8> {
        tiny_gguf_with_second_offset(64)
    }

    fn tiny_gguf_with_second_offset(second_offset: u64) -> Vec<u8> {
        let mut writer = GgufWriter::new();
        writer.header(2, 5);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);
        writer.metadata_key("general.quantization_version");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(2);
        writer.metadata_key("general.file_type");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(12);
        writer.metadata_key("tokenizer.ggml.tokens");
        writer.u32(GgufMetadataValueType::Array as u32);
        writer.u32(GgufMetadataValueType::String as u32);
        writer.u64(1025);
        for _ in 0..1025 {
            writer.string("");
        }

        writer.tensor_info("blk.0.attn_q.weight", &[32], GgmlType::Q8_0, 0);
        writer.tensor_info("token_embd.weight", &[256], GgmlType::Q2K, second_offset);
        writer.pad_to(32);
        writer.bytes(&(0_u8..160).collect::<Vec<_>>());
        writer.finish()
    }

    fn tiny_gguf_with_f32_tensor() -> Vec<u8> {
        let mut writer = GgufWriter::new();
        writer.header(1, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);
        writer.tensor_info("output_norm.weight", &[4], GgmlType::F32, 0);
        writer.pad_to(32);
        for value in [1.0_f32, -2.0, 3.5, 0.25] {
            writer.bytes(&value.to_le_bytes());
        }
        writer.bytes(&[0_u8; 16]);
        writer.finish()
    }

    fn tiny_supported_quant_gguf() -> Vec<u8> {
        let mut writer = GgufWriter::new();
        writer.header(2, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);
        writer.tensor_info("q2.weight", &[256], GgmlType::Q2K, 0);
        writer.tensor_info("q8.weight", &[32], GgmlType::Q8_0, 96);
        writer.pad_to(32);
        writer.bytes(&vec![0x22; GGML_Q2_K_BLOCK_BYTES as usize]);
        writer.bytes(&vec![0; (96 - GGML_Q2_K_BLOCK_BYTES) as usize]);
        writer.bytes(&vec![0x88; GGML_Q8_0_BLOCK_BYTES as usize]);
        writer.bytes(&[0_u8; 16]);
        writer.finish()
    }

    fn tiny_q8_0_values_gguf() -> Vec<u8> {
        let mut writer = GgufWriter::new();
        writer.header(1, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);
        writer.tensor_info("q8.weight", &[32], GgmlType::Q8_0, 0);
        writer.pad_to(32);
        writer.bytes(&q8_0_block(0x3800, -8));
        writer.bytes(&[0_u8; 16]);
        writer.finish()
    }

    fn tiny_q2_k_values_gguf() -> Vec<u8> {
        let mut writer = GgufWriter::new();
        writer.header(1, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);
        writer.tensor_info("q2.weight", &[256], GgmlType::Q2K, 0);
        writer.pad_to(32);
        writer.bytes(&q2_k_block(
            0x3c00,
            0x3800,
            [
                (1, 0),
                (2, 1),
                (3, 2),
                (4, 3),
                (5, 4),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 8),
                (10, 9),
                (11, 10),
                (12, 11),
                (13, 12),
                (14, 13),
                (15, 14),
                (0, 15),
            ],
            0xe4,
        ));
        writer.bytes(&[0_u8; 16]);
        writer.finish()
    }

    fn tiny_q3_k_values_gguf() -> Vec<u8> {
        let mut writer = GgufWriter::new();
        writer.header(1, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("laguna");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);
        writer.tensor_info("q3.weight", &[256], GgmlType::Q3K, 0);
        writer.pad_to(32);
        writer.bytes(&q3_k_block(0x3c00, [33; 16], 0xe4, 0xff));
        writer.bytes(&[0_u8; 16]);
        writer.finish()
    }

    fn tiny_q2_k_multi_block_values_gguf() -> Vec<u8> {
        let mut writer = GgufWriter::new();
        writer.header(1, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);
        writer.tensor_info("q2.weight", &[256, 2], GgmlType::Q2K, 0);
        writer.pad_to(32);
        writer.bytes(&q2_k_block(
            0x3c00,
            0x3800,
            [
                (1, 0),
                (2, 1),
                (3, 2),
                (4, 3),
                (5, 4),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 8),
                (10, 9),
                (11, 10),
                (12, 11),
                (13, 12),
                (14, 13),
                (15, 14),
                (0, 15),
            ],
            0xe4,
        ));
        writer.bytes(&q2_k_block(
            0x3c00,
            0x3800,
            [
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
                (1, 0),
            ],
            0xff,
        ));
        writer.bytes(&[0_u8; 16]);
        writer.finish()
    }

    fn q2_k_block(d: u16, dmin: u16, scale_mins: [(u8, u8); 16], quant_byte: u8) -> Vec<u8> {
        let mut block = Vec::with_capacity(GGML_Q2_K_BLOCK_BYTES as usize);
        for (scale, min) in scale_mins {
            block.push((scale & 0x0f) | ((min & 0x0f) << 4));
        }
        block.extend(std::iter::repeat_n(quant_byte, GGML_Q2_K_QUANT_BYTES));
        block.extend_from_slice(&d.to_le_bytes());
        block.extend_from_slice(&dmin.to_le_bytes());
        block
    }

    fn q8_0_block(d: u16, start: i8) -> Vec<u8> {
        let mut block = Vec::with_capacity(GGML_Q8_0_BLOCK_BYTES as usize);
        block.extend_from_slice(&d.to_le_bytes());
        for offset in 0..GGML_Q8_0_BLOCK_SIZE {
            block.push(start.wrapping_add(offset as i8) as u8);
        }
        block
    }

    fn q3_k_block(d: u16, decoded_scales: [u8; 16], quant_byte: u8, high_mask_byte: u8) -> Vec<u8> {
        let mut packed_scales = [0_u8; GGML_Q3_K_SCALE_BYTES];
        for (group, scale) in decoded_scales.into_iter().enumerate() {
            assert!(scale < 64);
            if group < 8 {
                packed_scales[group] |= scale & 0x0f;
            } else {
                packed_scales[group - 8] |= (scale & 0x0f) << 4;
            }
            packed_scales[8 + group % 4] |= ((scale >> 4) & 0x03) << (2 * (group / 4));
        }

        let mut block = Vec::with_capacity(GGML_Q3_K_BLOCK_BYTES as usize);
        block.extend(std::iter::repeat_n(
            high_mask_byte,
            GGML_Q3_K_HIGH_MASK_BYTES,
        ));
        block.extend(std::iter::repeat_n(quant_byte, GGML_Q3_K_QUANT_BYTES));
        block.extend_from_slice(&packed_scales);
        block.extend_from_slice(&d.to_le_bytes());
        block
    }

    fn tiny_bad_k_quant_shape_gguf() -> Vec<u8> {
        let mut writer = GgufWriter::new();
        writer.header(1, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);
        writer.tensor_info("bad_q2.weight", &[128, 2], GgmlType::Q2K, 0);
        writer.pad_to(32);
        writer.bytes(&vec![0x22; GGML_Q2_K_BLOCK_BYTES as usize]);
        writer.finish()
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

    fn unique_temp_file(label: &str) -> PathBuf {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("gguf-{label}-{}-{id}", std::process::id()))
    }
}
