use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::{Component, Path, PathBuf},
    sync::{Arc, Mutex},
};

use ::safetensors::{Dtype, SafeTensors};
use common::{Error, Result};
use serde::Deserialize;

use crate::{MappedFile, MappedFileAdvice};

pub const SAFETENSORS_INDEX_FILE: &str = "model.safetensors.index.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeTensorDtype {
    Bf16,
    F32,
    I32,
}

impl SafeTensorDtype {
    pub const fn byte_width(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 | Self::I32 => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeTensorInfo {
    pub name: String,
    pub dtype: SafeTensorDtype,
    pub shape: Vec<usize>,
    pub file_offset: u64,
    pub byte_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafeTensorIndex {
    model_dir: PathBuf,
    total_size: u64,
    weight_map: BTreeMap<String, PathBuf>,
    shard_paths: Vec<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct IndexDocument {
    metadata: IndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct IndexMetadata {
    total_size: u64,
}

impl SafeTensorIndex {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let index_path = model_dir.join(SAFETENSORS_INDEX_FILE);
        let json = fs::read_to_string(&index_path).map_err(|source| Error::Io {
            path: index_path.clone(),
            source,
        })?;
        let document: IndexDocument = serde_json::from_str(&json).map_err(|source| {
            Error::weights(format!(
                "failed to parse safetensors index at {}: {source}",
                index_path.display()
            ))
        })?;
        if document.metadata.total_size == 0 {
            return Err(Error::weights(
                "safetensors index metadata.total_size must be positive",
            ));
        }
        if document.weight_map.is_empty() {
            return Err(Error::weights("safetensors index weight_map is empty"));
        }

        let mut weight_map = BTreeMap::new();
        let mut shard_paths = BTreeSet::new();
        for (tensor_name, shard_name) in document.weight_map {
            validate_tensor_name(&tensor_name)?;
            validate_shard_name(&shard_name)?;
            let shard_path = model_dir.join(shard_name);
            shard_paths.insert(shard_path.clone());
            weight_map.insert(tensor_name, shard_path);
        }

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            total_size: document.metadata.total_size,
            weight_map,
            shard_paths: shard_paths.into_iter().collect(),
        })
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    pub fn total_size(&self) -> u64 {
        self.total_size
    }

    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    pub fn shard_count(&self) -> usize {
        self.shard_paths.len()
    }

    pub fn shard_paths(&self) -> &[PathBuf] {
        &self.shard_paths
    }

    pub fn contains_tensor(&self, name: &str) -> bool {
        self.weight_map.contains_key(name)
    }

    pub fn shard_for(&self, name: &str) -> Result<&Path> {
        self.weight_map
            .get(name)
            .map(PathBuf::as_path)
            .ok_or_else(|| Error::weights(format!("safetensors index is missing tensor {name}")))
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.weight_map.keys().map(String::as_str)
    }
}

#[derive(Debug)]
pub struct SafeTensorShard {
    path: PathBuf,
    mapped: MappedFile,
    tensors: BTreeMap<String, SafeTensorInfo>,
}

impl SafeTensorShard {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mapped = MappedFile::open(path)?;
        let bytes = mapped.slice(0, mapped.len())?;
        let (header_len, metadata) = SafeTensors::read_metadata(bytes).map_err(|source| {
            Error::weights(format!(
                "invalid safetensors shard {}: {source}",
                path.display()
            ))
        })?;
        let data_offset = 8_usize
            .checked_add(header_len)
            .ok_or_else(|| Error::weights("safetensors data offset overflow"))?;
        let data_offset_u64 = u64::try_from(data_offset)
            .map_err(|_| Error::weights("safetensors data offset does not fit u64"))?;
        let mut tensors = BTreeMap::new();
        for (name, info) in metadata.tensors() {
            let dtype = supported_dtype(&name, info.dtype)?;
            let relative_start = u64::try_from(info.data_offsets.0)
                .map_err(|_| Error::weights("safetensors tensor offset does not fit u64"))?;
            let relative_end = u64::try_from(info.data_offsets.1)
                .map_err(|_| Error::weights("safetensors tensor offset does not fit u64"))?;
            let file_offset = data_offset_u64
                .checked_add(relative_start)
                .ok_or_else(|| Error::weights("safetensors tensor file offset overflow"))?;
            let byte_len = relative_end
                .checked_sub(relative_start)
                .ok_or_else(|| Error::weights("safetensors tensor offset order is invalid"))?;
            tensors.insert(
                name.clone(),
                SafeTensorInfo {
                    name,
                    dtype,
                    shape: info.shape.clone(),
                    file_offset,
                    byte_len,
                },
            );
        }

        Ok(Self {
            path: path.to_path_buf(),
            mapped,
            tensors,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn info(&self, name: &str) -> Result<&SafeTensorInfo> {
        self.tensors.get(name).ok_or_else(|| {
            Error::weights(format!(
                "safetensors shard {} is missing tensor {name}",
                self.path.display()
            ))
        })
    }

    fn bytes(&self, info: &SafeTensorInfo) -> Result<&[u8]> {
        let byte_len = usize::try_from(info.byte_len)
            .map_err(|_| Error::weights("safetensors tensor length does not fit usize"))?;
        self.mapped.slice(info.file_offset, byte_len)
    }

    fn advise(&self, info: &SafeTensorInfo, advice: MappedFileAdvice) -> Result<()> {
        self.mapped
            .advise_range(advice, info.file_offset, info.byte_len)
    }

    fn prefetch(&self, info: &SafeTensorInfo) -> Result<()> {
        self.mapped.prefetch_range(info.file_offset, info.byte_len)
    }

    fn prefetch_ranges_serial(&self, ranges: &[(u64, u64)]) -> Result<()> {
        self.mapped.prefetch_ranges_serial(ranges)
    }
}

#[derive(Debug, Clone)]
pub struct SafeTensorHandle {
    shard: Arc<SafeTensorShard>,
    info: SafeTensorInfo,
}

impl SafeTensorHandle {
    pub fn shard_path(&self) -> &Path {
        self.shard.path()
    }

    pub fn info(&self) -> &SafeTensorInfo {
        &self.info
    }

    pub fn bytes(&self) -> Result<&[u8]> {
        self.shard.bytes(&self.info)
    }

    /// Materializes a BF16 tensor as F32. Intended for small vectors such as
    /// normalization weights and scalar quantization scales, not matrices.
    pub fn bf16_values_f32(&self) -> Result<Vec<f32>> {
        if self.info.dtype != SafeTensorDtype::Bf16 {
            return Err(Error::weights(format!(
                "safetensors tensor {} must be BF16 to decode as BF16, got {:?}",
                self.info.name, self.info.dtype
            )));
        }
        let bytes = self.bytes()?;
        if bytes.len() % 2 != 0 {
            return Err(Error::weights(format!(
                "BF16 tensor {} has odd byte length {}",
                self.info.name,
                bytes.len()
            )));
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|bytes| {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                f32::from_bits(u32::from(bits) << 16)
            })
            .collect())
    }

    /// Materializes an F32 tensor using the little-endian safetensors format.
    /// This is used for small metadata vectors such as router correction bias.
    pub fn f32_values(&self) -> Result<Vec<f32>> {
        if self.info.dtype != SafeTensorDtype::F32 {
            return Err(Error::weights(format!(
                "safetensors tensor {} must be F32 to decode as F32, got {:?}",
                self.info.name, self.info.dtype
            )));
        }
        let bytes = self.bytes()?;
        if bytes.len() % 4 != 0 {
            return Err(Error::weights(format!(
                "F32 tensor {} has non-divisible byte length {}",
                self.info.name,
                bytes.len()
            )));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect())
    }

    pub fn advise_random(&self) -> Result<()> {
        self.shard.advise(&self.info, MappedFileAdvice::Random)
    }

    pub fn advise_will_need(&self) -> Result<()> {
        self.shard.advise(&self.info, MappedFileAdvice::WillNeed)
    }

    pub fn prefetch(&self) -> Result<()> {
        self.shard.prefetch(&self.info)
    }

    /// Prefetches related tensors as coalesced shard ranges on the caller's
    /// worker thread. This is the SSD primitive used by one routed expert.
    pub fn prefetch_together(handles: &[&Self]) -> Result<()> {
        let mut by_shard = BTreeMap::<PathBuf, (Arc<SafeTensorShard>, Vec<(u64, u64)>)>::new();
        for handle in handles {
            let path = handle.shard.path().to_path_buf();
            let entry = by_shard
                .entry(path.clone())
                .or_insert_with(|| (handle.shard.clone(), Vec::new()));
            if !Arc::ptr_eq(&entry.0, &handle.shard) {
                return Err(Error::weights(format!(
                    "safetensors shard {} was opened more than once while grouping prefetch ranges",
                    path.display()
                )));
            }
            entry
                .1
                .push((handle.info.file_offset, handle.info.byte_len));
        }

        for (_, (shard, mut ranges)) in by_shard {
            ranges.sort_unstable_by_key(|(offset, _)| *offset);
            let ranges = coalesce_ranges(&ranges)?;
            shard.prefetch_ranges_serial(&ranges)?;
        }
        Ok(())
    }
}

fn coalesce_ranges(ranges: &[(u64, u64)]) -> Result<Vec<(u64, u64)>> {
    let mut coalesced = Vec::<(u64, u64)>::with_capacity(ranges.len());
    for &(offset, byte_len) in ranges {
        if byte_len == 0 {
            continue;
        }
        let end = offset
            .checked_add(byte_len)
            .ok_or_else(|| Error::weights("safetensors prefetch range overflow"))?;
        if let Some((previous_offset, previous_len)) = coalesced.last_mut() {
            let previous_end = previous_offset
                .checked_add(*previous_len)
                .ok_or_else(|| Error::weights("safetensors prefetch range overflow"))?;
            if offset <= previous_end {
                *previous_len = previous_end.max(end) - *previous_offset;
                continue;
            }
        }
        coalesced.push((offset, byte_len));
    }
    Ok(coalesced)
}

#[derive(Debug)]
pub struct SafeTensorModel {
    index: SafeTensorIndex,
    shards: Mutex<HashMap<PathBuf, Arc<SafeTensorShard>>>,
}

impl SafeTensorModel {
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            index: SafeTensorIndex::open(model_dir)?,
            shards: Mutex::new(HashMap::new()),
        })
    }

    pub fn index(&self) -> &SafeTensorIndex {
        &self.index
    }

    pub fn loaded_shard_count(&self) -> Result<usize> {
        Ok(self.lock_shards()?.len())
    }

    pub fn tensor(&self, name: &str) -> Result<SafeTensorHandle> {
        let shard_path = self.index.shard_for(name)?.to_path_buf();
        let shard = self.open_shard(&shard_path)?;
        let info = shard.info(name)?.clone();
        Ok(SafeTensorHandle { shard, info })
    }

    pub fn open_all_shards(&self) -> Result<Vec<Arc<SafeTensorShard>>> {
        self.index
            .shard_paths()
            .iter()
            .map(|path| self.open_shard(path))
            .collect()
    }

    fn open_shard(&self, path: &Path) -> Result<Arc<SafeTensorShard>> {
        {
            let shards = self.lock_shards()?;
            if let Some(shard) = shards.get(path) {
                return Ok(Arc::clone(shard));
            }
        }

        let opened = Arc::new(SafeTensorShard::open(path)?);
        let mut shards = self.lock_shards()?;
        Ok(Arc::clone(
            shards.entry(path.to_path_buf()).or_insert_with(|| opened),
        ))
    }

    fn lock_shards(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, HashMap<PathBuf, Arc<SafeTensorShard>>>> {
        self.shards
            .lock()
            .map_err(|_| Error::weights("safetensors shard cache lock is poisoned"))
    }
}

fn supported_dtype(name: &str, dtype: Dtype) -> Result<SafeTensorDtype> {
    match dtype {
        Dtype::BF16 => Ok(SafeTensorDtype::Bf16),
        Dtype::F32 => Ok(SafeTensorDtype::F32),
        Dtype::I32 => Ok(SafeTensorDtype::I32),
        other => Err(Error::weights(format!(
            "safetensors tensor {name} uses unsupported dtype {other:?}; Laguna INT4 supports BF16, F32, and I32 storage"
        ))),
    }
}

fn validate_tensor_name(name: &str) -> Result<()> {
    if name.is_empty() || name.bytes().any(|byte| byte == 0) {
        return Err(Error::weights(
            "safetensors tensor names must be non-empty and contain no NUL bytes",
        ));
    }
    Ok(())
}

fn validate_shard_name(name: &str) -> Result<()> {
    let path = Path::new(name);
    let mut components = path.components();
    let valid_component = matches!(components.next(), Some(Component::Normal(_)));
    if !valid_component
        || components.next().is_some()
        || path.extension().and_then(|extension| extension.to_str()) != Some("safetensors")
    {
        return Err(Error::weights(format!(
            "safetensors shard {name:?} must be one local .safetensors file name"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use serde_json::json;

    use super::*;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn index_opens_shards_only_when_a_tensor_is_requested() {
        let model_dir = write_tiny_model();
        let model = SafeTensorModel::open(&model_dir).unwrap();

        assert_eq!(model.index().total_size(), 8);
        assert_eq!(model.index().tensor_count(), 1);
        assert_eq!(model.index().shard_count(), 1);
        assert_eq!(model.loaded_shard_count().unwrap(), 0);

        let tensor = model.tensor("weight").unwrap();
        assert_eq!(tensor.info().dtype, SafeTensorDtype::I32);
        assert_eq!(tensor.info().shape, vec![1, 2]);
        assert_eq!(tensor.bytes().unwrap(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(model.loaded_shard_count().unwrap(), 1);
    }

    #[test]
    fn index_rejects_shard_path_traversal() {
        let model_dir = test_dir();
        fs::write(
            model_dir.join(SAFETENSORS_INDEX_FILE),
            json!({
                "metadata": {"total_size": 8},
                "weight_map": {"weight": "../outside.safetensors"}
            })
            .to_string(),
        )
        .unwrap();

        let error = SafeTensorIndex::open(model_dir).unwrap_err();
        assert!(error
            .to_string()
            .contains("one local .safetensors file name"));
    }

    #[test]
    fn decodes_small_bf16_and_f32_tensors_without_touching_other_payloads() {
        let model_dir = test_dir();
        let shard_name = "model-00001-of-00001.safetensors";
        let bf16 = [1.0_f32, -2.5]
            .into_iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        let f32_values = [0.25_f32, -4.0];
        let f32_bytes = f32_values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let header = format!(
            r#"{{"bias":{{"dtype":"F32","shape":[2],"data_offsets":[{},{}]}},"norm":{{"dtype":"BF16","shape":[2],"data_offsets":[0,{}]}}}}"#,
            bf16.len(),
            bf16.len() + f32_bytes.len(),
            bf16.len()
        );
        let mut shard = Vec::new();
        shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
        shard.extend_from_slice(header.as_bytes());
        shard.extend_from_slice(&bf16);
        shard.extend_from_slice(&f32_bytes);
        fs::write(model_dir.join(shard_name), shard).unwrap();
        fs::write(
            model_dir.join(SAFETENSORS_INDEX_FILE),
            json!({
                "metadata": {"total_size": bf16.len() + f32_bytes.len()},
                "weight_map": {"bias": shard_name, "norm": shard_name}
            })
            .to_string(),
        )
        .unwrap();

        let model = SafeTensorModel::open(model_dir).unwrap();
        let norm = model.tensor("norm").unwrap();
        let bias = model.tensor("bias").unwrap();
        SafeTensorHandle::prefetch_together(&[&norm, &bias]).unwrap();
        assert_eq!(norm.bf16_values_f32().unwrap(), vec![1.0, -2.5]);
        assert_eq!(bias.f32_values().unwrap(), f32_values);
    }

    #[test]
    fn coalesces_adjacent_expert_tensor_ranges_without_reading_gaps() {
        assert_eq!(
            coalesce_ranges(&[(10, 5), (15, 7), (30, 2), (31, 4)]).unwrap(),
            vec![(10, 12), (30, 5)]
        );
    }

    fn write_tiny_model() -> PathBuf {
        let model_dir = test_dir();
        let shard_name = "model-00001-of-00001.safetensors";
        let header = r#"{"weight":{"dtype":"I32","shape":[1,2],"data_offsets":[0,8]}}"#;
        let mut shard = Vec::new();
        shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
        shard.extend_from_slice(header.as_bytes());
        shard.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        fs::write(model_dir.join(shard_name), shard).unwrap();
        fs::write(
            model_dir.join(SAFETENSORS_INDEX_FILE),
            json!({
                "metadata": {"total_size": 8},
                "weight_map": {"weight": shard_name}
            })
            .to_string(),
        )
        .unwrap();
        model_dir
    }

    fn test_dir() -> PathBuf {
        let test_id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "inferno-safetensors-{}-{test_id}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }
}
