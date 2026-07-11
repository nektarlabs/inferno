use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use common::{validate_exact_shape, Error, F32Tensor, Result};

use crate::LayerKvCacheAppend;

const RECORD_MAGIC: [u8; 4] = *b"IKV1";
const RECORD_VERSION: u16 = 1;
const RECORD_HEADER_LEN: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ColdKvTensorKind {
    Key,
    Value,
}

impl ColdKvTensorKind {
    fn to_u8(self) -> u8 {
        match self {
            Self::Key => 1,
            Self::Value => 2,
        }
    }

    fn from_u8(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::Key),
            2 => Ok(Self::Value),
            other => Err(Error::cache(format!(
                "unknown cold KV tensor kind tag {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColdKvCodec {
    Q8Row,
}

impl ColdKvCodec {
    fn to_u8(self) -> u8 {
        match self {
            Self::Q8Row => 1,
        }
    }

    fn from_u8(value: u8) -> Result<Self> {
        match value {
            1 => Ok(Self::Q8Row),
            other => Err(Error::cache(format!("unknown cold KV codec tag {other}"))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColdKvStoreSpec {
    pub batch: usize,
    pub attention_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub block_tokens: usize,
    pub codec: ColdKvCodec,
}

impl ColdKvStoreSpec {
    pub fn validate(&self) -> Result<()> {
        if self.batch == 0 {
            return Err(Error::cache("cold KV batch must be positive"));
        }
        if self.attention_heads == 0 {
            return Err(Error::cache("cold KV attention_heads must be positive"));
        }
        if self.key_head_dim == 0 {
            return Err(Error::cache("cold KV key_head_dim must be positive"));
        }
        if self.value_head_dim == 0 {
            return Err(Error::cache("cold KV value_head_dim must be positive"));
        }
        if self.block_tokens == 0 {
            return Err(Error::cache("cold KV block_tokens must be positive"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColdKvBlockMeta {
    pub layer_index: usize,
    pub tensor_kind: ColdKvTensorKind,
    pub token_start: usize,
    pub token_count: usize,
    pub batch: usize,
    pub attention_heads: usize,
    pub head_dim: usize,
    pub codec: ColdKvCodec,
    pub file_offset: u64,
    pub payload_len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ColdKvBlockKey {
    pub layer_index: usize,
    pub tensor_kind: ColdKvTensorKind,
    pub token_start: usize,
    pub token_count: usize,
}

impl ColdKvBlockKey {
    fn from_meta(meta: &ColdKvBlockMeta) -> Self {
        Self {
            layer_index: meta.layer_index,
            tensor_kind: meta.tensor_kind,
            token_start: meta.token_start,
            token_count: meta.token_count,
        }
    }

    fn token_end(self) -> Result<usize> {
        self.token_start
            .checked_add(self.token_count)
            .ok_or_else(|| Error::cache("cold KV block token range overflow"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColdKvWriteReport {
    pub layer_index: usize,
    pub token_start: usize,
    pub token_count: usize,
    pub key_payload_bytes: u64,
    pub value_payload_bytes: u64,
    pub total_file_bytes_written: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayeredColdKvWriteReport {
    pub layer_count: usize,
    pub cached_tokens: usize,
    pub logical_block_count: usize,
    pub tensor_record_count: usize,
    pub stored_bytes: u64,
    pub raw_f32_bytes: u64,
    pub compression_ratio: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColdKvSelectedQ8TensorRows {
    pub batch: usize,
    pub attention_heads: usize,
    pub selected_tokens: usize,
    pub head_dim: usize,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColdKvSelectedQ8LayerRows {
    pub key: ColdKvSelectedQ8TensorRows,
    pub value: ColdKvSelectedQ8TensorRows,
}

#[derive(Debug, Clone)]
struct RecordHeader {
    layer_index: usize,
    tensor_kind: ColdKvTensorKind,
    token_start: usize,
    token_count: usize,
    batch: usize,
    attention_heads: usize,
    head_dim: usize,
    codec: ColdKvCodec,
    payload_len: u64,
}

/// Cold KV tier backed by a single append-only block file.
///
/// This is deliberately not the old disk spill design. Every API works with a
/// whole compressed K or V block, so callers can stream large chunks and then
/// keep the hot block resident in Metal.
#[derive(Debug)]
pub struct ColdKvBlockStore {
    path: PathBuf,
    file: File,
    spec: ColdKvStoreSpec,
    blocks: Vec<ColdKvBlockMeta>,
    index: BTreeMap<ColdKvBlockKey, usize>,
}

#[derive(Debug)]
pub struct LayeredColdKvBlockStore {
    store: ColdKvBlockStore,
    cached_tokens: usize,
    layer_indices: Vec<usize>,
}

impl LayeredColdKvBlockStore {
    pub fn create(path: impl AsRef<Path>, spec: ColdKvStoreSpec) -> Result<Self> {
        Ok(Self {
            store: ColdKvBlockStore::create(path, spec)?,
            cached_tokens: 0,
            layer_indices: Vec::new(),
        })
    }

    pub fn open(path: impl AsRef<Path>, spec: ColdKvStoreSpec) -> Result<Self> {
        let store = ColdKvBlockStore::open(path, spec)?;
        let (cached_tokens, layer_indices) = derive_layered_state(store.blocks())?;
        Ok(Self {
            store,
            cached_tokens,
            layer_indices,
        })
    }

    pub fn store(&self) -> &ColdKvBlockStore {
        &self.store
    }

    pub fn store_mut(&mut self) -> &mut ColdKvBlockStore {
        &mut self.store
    }

    pub fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    pub fn layer_indices(&self) -> &[usize] {
        &self.layer_indices
    }

    pub fn write_prefill(
        &mut self,
        layers: &[LayerKvCacheAppend<'_>],
    ) -> Result<LayeredColdKvWriteReport> {
        if self.cached_tokens != 0 || !self.layer_indices.is_empty() {
            return Err(Error::cache(
                "layered cold KV prefill requires an empty store",
            ));
        }
        if layers.is_empty() {
            return Err(Error::cache(
                "layered cold KV prefill requires at least one layer",
            ));
        }
        let token_count = validate_layered_cold_appends(self.store.spec(), layers)?;
        let mut sorted_indices = layers
            .iter()
            .map(|layer| layer.layer_index)
            .collect::<Vec<_>>();
        sorted_indices.sort_unstable();

        let mut logical_block_count = 0_usize;
        let mut token_start = 0_usize;
        while token_start < token_count {
            let block_tokens = self.store.spec.block_tokens.min(token_count - token_start);
            for layer in layers {
                let k_block = slice_token_range(layer.k, token_start, block_tokens)?;
                let v_block = slice_token_range(layer.v, token_start, block_tokens)?;
                self.store
                    .write_layer_block(layer.layer_index, token_start, &k_block, &v_block)?;
            }
            logical_block_count = logical_block_count
                .checked_add(1)
                .ok_or_else(|| Error::cache("layered cold KV logical block count overflow"))?;
            token_start = token_start
                .checked_add(block_tokens)
                .ok_or_else(|| Error::cache("layered cold KV token_start overflow"))?;
        }

        self.cached_tokens = token_count;
        self.layer_indices = sorted_indices;
        let stored_bytes = self.store.stored_bytes()?;
        let raw_f32_bytes = layered_raw_f32_bytes(self.store.spec(), layers.len(), token_count)?;
        let compression_ratio = if raw_f32_bytes == 0 {
            1.0
        } else {
            stored_bytes as f32 / raw_f32_bytes as f32
        };

        Ok(LayeredColdKvWriteReport {
            layer_count: layers.len(),
            cached_tokens: token_count,
            logical_block_count,
            tensor_record_count: self.store.block_count(),
            stored_bytes,
            raw_f32_bytes,
            compression_ratio,
        })
    }

    pub fn read_layer_block(
        &mut self,
        layer_index: usize,
        token_start: usize,
    ) -> Result<(F32Tensor, F32Tensor)> {
        if !self.layer_indices.contains(&layer_index) {
            return Err(Error::cache(format!(
                "layer {layer_index} is not present in cold KV tier"
            )));
        }
        self.store.read_layer_block(layer_index, token_start)
    }

    pub fn read_layer_selected_tokens(
        &mut self,
        layer_index: usize,
        token_indices: &[u32],
    ) -> Result<(F32Tensor, F32Tensor)> {
        if !self.layer_indices.contains(&layer_index) {
            return Err(Error::cache(format!(
                "layer {layer_index} is not present in cold KV tier"
            )));
        }
        if token_indices
            .iter()
            .any(|token| *token as usize >= self.cached_tokens)
        {
            return Err(Error::cache(format!(
                "cold KV selected token exceeds cached token count {}",
                self.cached_tokens
            )));
        }
        self.store
            .read_layer_selected_tokens(layer_index, token_indices)
    }

    pub fn read_layer_selected_q8_rows(
        &mut self,
        layer_index: usize,
        token_indices: &[u32],
    ) -> Result<ColdKvSelectedQ8LayerRows> {
        if !self.layer_indices.contains(&layer_index) {
            return Err(Error::cache(format!(
                "layer {layer_index} is not present in cold KV tier"
            )));
        }
        if token_indices
            .iter()
            .any(|token| *token as usize >= self.cached_tokens)
        {
            return Err(Error::cache(format!(
                "cold KV selected token exceeds cached token count {}",
                self.cached_tokens
            )));
        }
        self.store
            .read_layer_selected_q8_rows(layer_index, token_indices)
    }
}

impl ColdKvBlockStore {
    pub fn create(path: impl AsRef<Path>, spec: ColdKvStoreSpec) -> Result<Self> {
        spec.validate()?;
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(&path)
            .map_err(|error| {
                Error::cache(format!(
                    "failed to create cold KV block store {}: {error}",
                    path.display()
                ))
            })?;
        Ok(Self {
            path,
            file,
            spec,
            blocks: Vec::new(),
            index: BTreeMap::new(),
        })
    }

    pub fn open(path: impl AsRef<Path>, spec: ColdKvStoreSpec) -> Result<Self> {
        spec.validate()?;
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new()
            .write(true)
            .read(true)
            .open(&path)
            .map_err(|error| {
                Error::cache(format!(
                    "failed to open cold KV block store {}: {error}",
                    path.display()
                ))
            })?;
        let (blocks, index) = rebuild_block_index(&mut file, &spec, &path)?;
        file.seek(SeekFrom::End(0)).map_err(|error| {
            Error::cache(format!(
                "failed to seek cold KV block store {} after index rebuild: {error}",
                path.display()
            ))
        })?;
        Ok(Self {
            path,
            file,
            spec,
            blocks,
            index,
        })
    }

    pub fn clone_reader(&self) -> Result<Self> {
        // `File::try_clone` duplicates the descriptor but may share the same
        // kernel file offset. Prefetch threads seek while the writer appends,
        // so a shared offset can make a reader start in the middle of another
        // record. Reopening the path gives every reader an independent cursor.
        let mut file = OpenOptions::new()
            .read(true)
            .open(&self.path)
            .map_err(|error| {
                Error::cache(format!(
                    "failed to open cold KV block store reader {}: {error}",
                    self.path.display()
                ))
            })?;
        file.seek(SeekFrom::Start(0)).map_err(|error| {
            Error::cache(format!(
                "failed to seek cloned cold KV block store reader {}: {error}",
                self.path.display()
            ))
        })?;
        Ok(Self {
            path: self.path.clone(),
            file,
            spec: self.spec.clone(),
            blocks: self.blocks.clone(),
            index: self.index.clone(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn spec(&self) -> &ColdKvStoreSpec {
        &self.spec
    }

    pub fn blocks(&self) -> &[ColdKvBlockMeta] {
        &self.blocks
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn lookup_block(
        &self,
        layer_index: usize,
        tensor_kind: ColdKvTensorKind,
        token_start: usize,
        token_count: usize,
    ) -> Option<&ColdKvBlockMeta> {
        let key = ColdKvBlockKey {
            layer_index,
            tensor_kind,
            token_start,
            token_count,
        };
        self.index
            .get(&key)
            .map(|block_index| &self.blocks[*block_index])
    }

    pub fn stored_bytes(&self) -> Result<u64> {
        self.file
            .metadata()
            .map(|metadata| metadata.len())
            .map_err(|error| {
                Error::cache(format!(
                    "failed to read cold KV block store metadata {}: {error}",
                    self.path.display()
                ))
            })
    }

    pub fn flush(&mut self) -> Result<()> {
        self.file.flush().map_err(|error| {
            Error::cache(format!(
                "failed to flush cold KV block store {}: {error}",
                self.path.display()
            ))
        })
    }

    pub fn sync_all(&mut self) -> Result<()> {
        self.file.sync_all().map_err(|error| {
            Error::cache(format!(
                "failed to sync cold KV block store {}: {error}",
                self.path.display()
            ))
        })
    }

    pub fn write_layer_block(
        &mut self,
        layer_index: usize,
        token_start: usize,
        k: &F32Tensor,
        v: &F32Tensor,
    ) -> Result<ColdKvWriteReport> {
        let token_count = validate_block_tensor(
            "cold KV key block",
            k,
            self.spec.batch,
            self.spec.attention_heads,
            self.spec.key_head_dim,
            self.spec.block_tokens,
        )?;
        validate_exact_shape(
            "cold_kv_value_block_token_count",
            &[validate_block_tensor(
                "cold KV value block",
                v,
                self.spec.batch,
                self.spec.attention_heads,
                self.spec.value_head_dim,
                self.spec.block_tokens,
            )?],
            &[token_count],
        )?;

        let key_payload = encode_q8_rows(k)?;
        let value_payload = encode_q8_rows(v)?;
        let key_payload_bytes = u64::try_from(key_payload.len())
            .map_err(|_| Error::cache("cold KV key payload length does not fit u64"))?;
        let value_payload_bytes = u64::try_from(value_payload.len())
            .map_err(|_| Error::cache("cold KV value payload length does not fit u64"))?;

        let key_written = self.write_record(
            RecordHeader {
                layer_index,
                tensor_kind: ColdKvTensorKind::Key,
                token_start,
                token_count,
                batch: self.spec.batch,
                attention_heads: self.spec.attention_heads,
                head_dim: self.spec.key_head_dim,
                codec: self.spec.codec,
                payload_len: key_payload_bytes,
            },
            &key_payload,
        )?;
        let value_written = self.write_record(
            RecordHeader {
                layer_index,
                tensor_kind: ColdKvTensorKind::Value,
                token_start,
                token_count,
                batch: self.spec.batch,
                attention_heads: self.spec.attention_heads,
                head_dim: self.spec.value_head_dim,
                codec: self.spec.codec,
                payload_len: value_payload_bytes,
            },
            &value_payload,
        )?;

        Ok(ColdKvWriteReport {
            layer_index,
            token_start,
            token_count,
            key_payload_bytes,
            value_payload_bytes,
            total_file_bytes_written: key_written
                .checked_add(value_written)
                .ok_or_else(|| Error::cache("cold KV written byte count overflow"))?,
        })
    }

    pub fn read_tensor_block(
        &mut self,
        layer_index: usize,
        tensor_kind: ColdKvTensorKind,
        token_start: usize,
    ) -> Result<F32Tensor> {
        let meta = self.find_block_by_start(layer_index, tensor_kind, token_start)?;
        self.read_meta_block(&meta)
    }

    pub fn read_tensor_range(
        &mut self,
        layer_index: usize,
        tensor_kind: ColdKvTensorKind,
        token_start: usize,
        token_count: usize,
    ) -> Result<F32Tensor> {
        let meta = self
            .lookup_block(layer_index, tensor_kind, token_start, token_count)
            .cloned()
            .ok_or_else(|| {
                Error::cache(format!(
                    "cold KV block not found for layer={layer_index} kind={tensor_kind:?} token_range=[{token_start},{})",
                    token_start.saturating_add(token_count)
                ))
            })?;
        self.read_meta_block(&meta)
    }

    pub fn read_tensor_range_contiguous(
        &mut self,
        layer_index: usize,
        tensor_kind: ColdKvTensorKind,
        token_start: usize,
        token_count: usize,
    ) -> Result<F32Tensor> {
        if token_count == 0 {
            return Err(Error::cache(
                "cold KV contiguous read token_count must be positive",
            ));
        }
        if let Some(meta) = self
            .lookup_block(layer_index, tensor_kind, token_start, token_count)
            .cloned()
        {
            return self.read_meta_block(&meta);
        }

        let token_end = token_start
            .checked_add(token_count)
            .ok_or_else(|| Error::cache("cold KV contiguous read token range overflow"))?;
        let mut cursor = token_start;
        let mut metas = Vec::new();
        while cursor < token_end {
            let meta = self.find_block_by_start(layer_index, tensor_kind, cursor)?;
            let meta_end = meta
                .token_start
                .checked_add(meta.token_count)
                .ok_or_else(|| Error::cache("cold KV block token range overflow"))?;
            if meta_end > token_end {
                return Err(Error::cache(format!(
                    "cold KV block layer={layer_index} kind={tensor_kind:?} range=[{}, {}) exceeds requested range=[{token_start}, {token_end})",
                    meta.token_start, meta_end
                )));
            }
            cursor = meta_end;
            metas.push(meta);
        }

        let mut tensors = Vec::with_capacity(metas.len());
        for meta in metas {
            tensors.push(self.read_meta_block(&meta)?);
        }
        concat_token_blocks(&tensors)
    }

    pub fn read_layer_block(
        &mut self,
        layer_index: usize,
        token_start: usize,
    ) -> Result<(F32Tensor, F32Tensor)> {
        Ok((
            self.read_tensor_block(layer_index, ColdKvTensorKind::Key, token_start)?,
            self.read_tensor_block(layer_index, ColdKvTensorKind::Value, token_start)?,
        ))
    }

    pub fn read_layer_range(
        &mut self,
        layer_index: usize,
        token_start: usize,
        token_count: usize,
    ) -> Result<(F32Tensor, F32Tensor)> {
        Ok((
            self.read_tensor_range(layer_index, ColdKvTensorKind::Key, token_start, token_count)?,
            self.read_tensor_range(
                layer_index,
                ColdKvTensorKind::Value,
                token_start,
                token_count,
            )?,
        ))
    }

    pub fn read_layer_range_contiguous(
        &mut self,
        layer_index: usize,
        token_start: usize,
        token_count: usize,
    ) -> Result<(F32Tensor, F32Tensor)> {
        Ok((
            self.read_tensor_range_contiguous(
                layer_index,
                ColdKvTensorKind::Key,
                token_start,
                token_count,
            )?,
            self.read_tensor_range_contiguous(
                layer_index,
                ColdKvTensorKind::Value,
                token_start,
                token_count,
            )?,
        ))
    }

    pub fn read_layer_selected_tokens(
        &mut self,
        layer_index: usize,
        token_indices: &[u32],
    ) -> Result<(F32Tensor, F32Tensor)> {
        if token_indices.is_empty() {
            return Err(Error::cache(
                "cold KV selected read requires at least one token",
            ));
        }
        Ok((
            self.read_tensor_selected_tokens(layer_index, ColdKvTensorKind::Key, token_indices)?,
            self.read_tensor_selected_tokens(layer_index, ColdKvTensorKind::Value, token_indices)?,
        ))
    }

    pub fn read_layer_selected_q8_rows(
        &mut self,
        layer_index: usize,
        token_indices: &[u32],
    ) -> Result<ColdKvSelectedQ8LayerRows> {
        if token_indices.is_empty() {
            return Err(Error::cache(
                "cold KV selected Q8 row read requires at least one token",
            ));
        }
        Ok(ColdKvSelectedQ8LayerRows {
            key: self.read_tensor_selected_q8_rows(
                layer_index,
                ColdKvTensorKind::Key,
                token_indices,
            )?,
            value: self.read_tensor_selected_q8_rows(
                layer_index,
                ColdKvTensorKind::Value,
                token_indices,
            )?,
        })
    }

    pub fn read_tensor_selected_tokens(
        &mut self,
        layer_index: usize,
        tensor_kind: ColdKvTensorKind,
        token_indices: &[u32],
    ) -> Result<F32Tensor> {
        if token_indices.is_empty() {
            return Err(Error::cache(
                "cold KV selected tensor read requires at least one token",
            ));
        }

        let mut block_cache = BTreeMap::<usize, (ColdKvBlockMeta, F32Tensor)>::new();
        for token in token_indices {
            let token = usize::try_from(*token)
                .map_err(|_| Error::cache("cold KV selected token does not fit usize"))?;
            let meta = self.find_block_containing(layer_index, tensor_kind, token)?;
            if !block_cache.contains_key(&meta.token_start) {
                let tensor = self.read_meta_block(&meta)?;
                block_cache.insert(meta.token_start, (meta, tensor));
            }
        }

        gather_selected_tokens_from_blocks(token_indices, &block_cache)
    }

    pub fn read_tensor_selected_q8_rows(
        &mut self,
        layer_index: usize,
        tensor_kind: ColdKvTensorKind,
        token_indices: &[u32],
    ) -> Result<ColdKvSelectedQ8TensorRows> {
        if token_indices.is_empty() {
            return Err(Error::cache(
                "cold KV selected Q8 tensor read requires at least one token",
            ));
        }

        let mut block_cache = BTreeMap::<usize, (ColdKvBlockMeta, Vec<u8>)>::new();
        for token in token_indices {
            let token = usize::try_from(*token)
                .map_err(|_| Error::cache("cold KV selected token does not fit usize"))?;
            let meta = self.find_block_containing(layer_index, tensor_kind, token)?;
            if meta.codec != ColdKvCodec::Q8Row {
                return Err(Error::cache(format!(
                    "cold KV selected compressed read requires Q8Row blocks, got {:?}",
                    meta.codec
                )));
            }
            if !block_cache.contains_key(&meta.token_start) {
                let payload = self.read_meta_payload(&meta)?;
                block_cache.insert(meta.token_start, (meta, payload));
            }
        }

        gather_selected_q8_rows_from_blocks(token_indices, &block_cache)
    }

    fn write_record(&mut self, header: RecordHeader, payload: &[u8]) -> Result<u64> {
        validate_record_header_against_spec(&header, &self.spec)?;
        validate_payload_len(&header, payload.len())?;

        let file_offset = self.file.seek(SeekFrom::End(0)).map_err(|error| {
            Error::cache(format!(
                "failed to seek cold KV block store {}: {error}",
                self.path.display()
            ))
        })?;
        let meta = ColdKvBlockMeta {
            layer_index: header.layer_index,
            tensor_kind: header.tensor_kind,
            token_start: header.token_start,
            token_count: header.token_count,
            batch: header.batch,
            attention_heads: header.attention_heads,
            head_dim: header.head_dim,
            codec: header.codec,
            file_offset,
            payload_len: header.payload_len,
        };
        validate_no_overlap(&self.index, &meta)?;
        let mut record = encode_record_header(&header)?;
        record.extend_from_slice(payload);
        self.file.write_all(&record).map_err(|error| {
            Error::cache(format!(
                "failed to write cold KV block store {}: {error}",
                self.path.display()
            ))
        })?;
        insert_indexed_meta(&mut self.blocks, &mut self.index, meta)?;
        u64::try_from(record.len())
            .map_err(|_| Error::cache("cold KV record length does not fit u64"))
    }

    fn find_block_by_start(
        &self,
        layer_index: usize,
        tensor_kind: ColdKvTensorKind,
        token_start: usize,
    ) -> Result<ColdKvBlockMeta> {
        let lower = ColdKvBlockKey {
            layer_index,
            tensor_kind,
            token_start,
            token_count: 0,
        };
        let upper = ColdKvBlockKey {
            layer_index,
            tensor_kind,
            token_start,
            token_count: usize::MAX,
        };
        self.index
            .range(lower..=upper)
            .next()
            .map(|(_, block_index)| self.blocks[*block_index].clone())
            .ok_or_else(|| {
                Error::cache(format!(
                    "cold KV block not found for layer={layer_index} kind={tensor_kind:?} token_start={token_start}"
                ))
            })
    }

    fn find_block_containing(
        &self,
        layer_index: usize,
        tensor_kind: ColdKvTensorKind,
        token: usize,
    ) -> Result<ColdKvBlockMeta> {
        self.blocks
            .iter()
            .find(|meta| {
                meta.layer_index == layer_index
                    && meta.tensor_kind == tensor_kind
                    && token >= meta.token_start
                    && token < meta.token_start + meta.token_count
            })
            .cloned()
            .ok_or_else(|| {
                Error::cache(format!(
                    "cold KV block containing token {token} not found for layer={layer_index} kind={tensor_kind:?}"
                ))
            })
    }

    fn read_meta_block(&mut self, meta: &ColdKvBlockMeta) -> Result<F32Tensor> {
        let payload = self.read_meta_payload(meta)?;
        decode_q8_rows(
            &payload,
            meta.batch,
            meta.attention_heads,
            meta.token_count,
            meta.head_dim,
        )
    }

    fn read_meta_payload(&mut self, meta: &ColdKvBlockMeta) -> Result<Vec<u8>> {
        self.file
            .seek(SeekFrom::Start(meta.file_offset))
            .map_err(|error| {
                Error::cache(format!(
                    "failed to seek cold KV block store {}: {error}",
                    self.path.display()
                ))
            })?;
        let record_len = RECORD_HEADER_LEN
            .checked_add(
                usize::try_from(meta.payload_len)
                    .map_err(|_| Error::cache("cold KV payload length does not fit usize"))?,
            )
            .ok_or_else(|| Error::cache("cold KV record length overflow"))?;
        let mut record = vec![0_u8; record_len];
        self.file.read_exact(&mut record).map_err(|error| {
            Error::cache(format!(
                "failed to read cold KV block store {}: {error}",
                self.path.display()
            ))
        })?;
        let header = decode_record_header(&record[..RECORD_HEADER_LEN])?;
        validate_record_matches_meta(&header, meta)?;
        Ok(record[RECORD_HEADER_LEN..].to_vec())
    }
}

fn rebuild_block_index(
    file: &mut File,
    spec: &ColdKvStoreSpec,
    path: &Path,
) -> Result<(Vec<ColdKvBlockMeta>, BTreeMap<ColdKvBlockKey, usize>)> {
    let file_len = file
        .metadata()
        .map_err(|error| {
            Error::cache(format!(
                "failed to read cold KV block store metadata {}: {error}",
                path.display()
            ))
        })?
        .len();
    let mut blocks = Vec::new();
    let mut index = BTreeMap::new();
    let mut file_offset = 0_u64;

    while file_offset < file_len {
        let remaining = file_len
            .checked_sub(file_offset)
            .ok_or_else(|| Error::cache("cold KV index rebuild offset overflow"))?;
        if remaining < RECORD_HEADER_LEN as u64 {
            return Err(Error::cache(format!(
                "truncated cold KV record header at offset {file_offset} in {}",
                path.display()
            )));
        }

        file.seek(SeekFrom::Start(file_offset)).map_err(|error| {
            Error::cache(format!(
                "failed to seek cold KV block store {} while rebuilding index: {error}",
                path.display()
            ))
        })?;
        let mut header_bytes = [0_u8; RECORD_HEADER_LEN];
        file.read_exact(&mut header_bytes).map_err(|error| {
            Error::cache(format!(
                "failed to read cold KV record header at offset {file_offset} in {}: {error}",
                path.display()
            ))
        })?;
        let header = decode_record_header(&header_bytes)?;
        validate_record_header_against_spec(&header, spec)?;
        let record_len = record_len(&header)?;
        let record_end = file_offset
            .checked_add(record_len)
            .ok_or_else(|| Error::cache("cold KV record end offset overflow"))?;
        if record_end > file_len {
            return Err(Error::cache(format!(
                "truncated cold KV record payload at offset {file_offset} in {}",
                path.display()
            )));
        }

        insert_indexed_meta(
            &mut blocks,
            &mut index,
            ColdKvBlockMeta {
                layer_index: header.layer_index,
                tensor_kind: header.tensor_kind,
                token_start: header.token_start,
                token_count: header.token_count,
                batch: header.batch,
                attention_heads: header.attention_heads,
                head_dim: header.head_dim,
                codec: header.codec,
                file_offset,
                payload_len: header.payload_len,
            },
        )?;
        file_offset = record_end;
    }

    Ok((blocks, index))
}

fn insert_indexed_meta(
    blocks: &mut Vec<ColdKvBlockMeta>,
    index: &mut BTreeMap<ColdKvBlockKey, usize>,
    meta: ColdKvBlockMeta,
) -> Result<()> {
    validate_no_overlap(index, &meta)?;
    let key = ColdKvBlockKey::from_meta(&meta);
    let block_index = blocks.len();
    let previous = index.insert(key, block_index);
    if previous.is_some() {
        return Err(Error::cache(format!(
            "duplicate cold KV block for layer={} kind={:?} token_range=[{},{})",
            key.layer_index,
            key.tensor_kind,
            key.token_start,
            key.token_end()?
        )));
    }
    blocks.push(meta);
    Ok(())
}

fn validate_no_overlap(
    index: &BTreeMap<ColdKvBlockKey, usize>,
    meta: &ColdKvBlockMeta,
) -> Result<()> {
    let candidate = ColdKvBlockKey::from_meta(meta);
    let candidate_end = candidate.token_end()?;
    let lower = ColdKvBlockKey {
        layer_index: candidate.layer_index,
        tensor_kind: candidate.tensor_kind,
        token_start: 0,
        token_count: 0,
    };
    let upper = ColdKvBlockKey {
        layer_index: candidate.layer_index,
        tensor_kind: candidate.tensor_kind,
        token_start: usize::MAX,
        token_count: usize::MAX,
    };

    for (existing, _) in index.range(lower..=upper) {
        let existing_end = existing.token_end()?;
        if candidate.token_start == existing.token_start
            && candidate.token_count == existing.token_count
        {
            return Err(Error::cache(format!(
                "duplicate cold KV block for layer={} kind={:?} token_range=[{},{})",
                candidate.layer_index, candidate.tensor_kind, candidate.token_start, candidate_end
            )));
        }
        if candidate.token_start < existing_end && existing.token_start < candidate_end {
            return Err(Error::cache(format!(
                "overlapping cold KV block for layer={} kind={:?}: candidate=[{},{}) existing=[{},{})",
                candidate.layer_index,
                candidate.tensor_kind,
                candidate.token_start,
                candidate_end,
                existing.token_start,
                existing_end
            )));
        }
    }
    Ok(())
}

fn validate_record_header_against_spec(
    header: &RecordHeader,
    spec: &ColdKvStoreSpec,
) -> Result<()> {
    if header.token_count == 0 || header.token_count > spec.block_tokens {
        return Err(Error::cache(format!(
            "cold KV record token_count {} must be in 1..={}",
            header.token_count, spec.block_tokens
        )));
    }
    validate_exact_shape("cold_kv_record_batch", &[header.batch], &[spec.batch])?;
    validate_exact_shape(
        "cold_kv_record_attention_heads",
        &[header.attention_heads],
        &[spec.attention_heads],
    )?;
    let expected_head_dim = match header.tensor_kind {
        ColdKvTensorKind::Key => spec.key_head_dim,
        ColdKvTensorKind::Value => spec.value_head_dim,
    };
    validate_exact_shape(
        "cold_kv_record_head_dim",
        &[header.head_dim],
        &[expected_head_dim],
    )?;
    if header.codec != spec.codec {
        return Err(Error::cache(format!(
            "cold KV record codec {:?} does not match store codec {:?}",
            header.codec, spec.codec
        )));
    }
    let expected_payload_len = q8_row_payload_len(
        header.batch,
        header.attention_heads,
        header.token_count,
        header.head_dim,
    )?;
    if header.payload_len != expected_payload_len {
        return Err(Error::cache(format!(
            "cold KV record payload_len {} does not match expected {}",
            header.payload_len, expected_payload_len
        )));
    }
    Ok(())
}

fn validate_payload_len(header: &RecordHeader, payload_len: usize) -> Result<()> {
    let payload_len = checked_u64("cold KV write payload length", payload_len)?;
    if payload_len != header.payload_len {
        return Err(Error::cache(format!(
            "cold KV write payload length {payload_len} does not match header payload_len {}",
            header.payload_len
        )));
    }
    Ok(())
}

fn record_len(header: &RecordHeader) -> Result<u64> {
    (RECORD_HEADER_LEN as u64)
        .checked_add(header.payload_len)
        .ok_or_else(|| Error::cache("cold KV record length overflow"))
}

fn q8_row_payload_len(
    batch: usize,
    attention_heads: usize,
    token_count: usize,
    head_dim: usize,
) -> Result<u64> {
    let rows = batch
        .checked_mul(attention_heads)
        .and_then(|value| value.checked_mul(token_count))
        .ok_or_else(|| Error::cache("Q8 row cold KV payload row count overflow"))?;
    let row_bytes = q8_row_byte_len(head_dim)?;
    rows.checked_mul(row_bytes)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| Error::cache("Q8 row cold KV payload length does not fit u64"))
}

fn q8_payload_byte_len(
    batch: usize,
    attention_heads: usize,
    token_count: usize,
    head_dim: usize,
) -> Result<usize> {
    usize::try_from(q8_row_payload_len(
        batch,
        attention_heads,
        token_count,
        head_dim,
    )?)
    .map_err(|_| Error::cache("Q8 row cold KV payload length does not fit usize"))
}

fn q8_row_byte_len(head_dim: usize) -> Result<usize> {
    4_usize
        .checked_add(head_dim)
        .ok_or_else(|| Error::cache("Q8 row cold KV payload row size overflow"))
}

fn derive_layered_state(blocks: &[ColdKvBlockMeta]) -> Result<(usize, Vec<usize>)> {
    let mut ranges_by_layer: BTreeMap<usize, BTreeMap<(usize, usize), (bool, bool)>> =
        BTreeMap::new();
    for block in blocks {
        let flags = ranges_by_layer
            .entry(block.layer_index)
            .or_default()
            .entry((block.token_start, block.token_count))
            .or_default();
        match block.tensor_kind {
            ColdKvTensorKind::Key => flags.0 = true,
            ColdKvTensorKind::Value => flags.1 = true,
        }
    }

    let mut cached_tokens = None;
    let mut layer_indices = Vec::with_capacity(ranges_by_layer.len());
    for (layer_index, ranges) in ranges_by_layer {
        let mut cursor = 0_usize;
        for ((token_start, token_count), (has_key, has_value)) in ranges {
            if !has_key || !has_value {
                return Err(Error::cache(format!(
                    "layered cold KV layer {layer_index} range [{},{}) is missing {}",
                    token_start,
                    token_start.saturating_add(token_count),
                    if !has_key { "key block" } else { "value block" }
                )));
            }
            if token_start != cursor {
                return Err(Error::cache(format!(
                    "layered cold KV layer {layer_index} has non-contiguous token ranges: expected start {cursor}, got {token_start}"
                )));
            }
            cursor = cursor
                .checked_add(token_count)
                .ok_or_else(|| Error::cache("layered cold KV cached token count overflow"))?;
        }

        match cached_tokens {
            Some(expected) if expected != cursor => {
                return Err(Error::cache(format!(
                    "layered cold KV layer {layer_index} has {cursor} cached tokens, expected {expected}"
                )));
            }
            None => cached_tokens = Some(cursor),
            _ => {}
        }
        layer_indices.push(layer_index);
    }

    Ok((cached_tokens.unwrap_or(0), layer_indices))
}

fn concat_token_blocks(blocks: &[F32Tensor]) -> Result<F32Tensor> {
    let first = blocks
        .first()
        .ok_or_else(|| Error::cache("cold KV contiguous read produced no blocks"))?;
    let first_dims = first.dims();
    if first_dims.len() != 4 {
        return Err(Error::cache(format!(
            "cold KV concat requires rank 4 [B,H,T,D], got {first_dims:?}"
        )));
    }
    let batch = first_dims[0];
    let heads = first_dims[1];
    let dim = first_dims[3];
    let total_tokens = blocks.iter().try_fold(0_usize, |total, block| {
        let dims = block.dims();
        validate_exact_shape(
            "cold_kv_concat_batch_heads_dim",
            &[dims[0], dims[1], dims[3]],
            &[batch, heads, dim],
        )?;
        total
            .checked_add(dims[2])
            .ok_or_else(|| Error::cache("cold KV concat token count overflow"))
    })?;

    let mut values = vec![0.0_f32; batch * heads * total_tokens * dim];
    let mut token_cursor = 0_usize;
    for block in blocks {
        let block_tokens = block.dims()[2];
        for batch_index in 0..batch {
            for head_index in 0..heads {
                for token_index in 0..block_tokens {
                    let source =
                        ((batch_index * heads + head_index) * block_tokens + token_index) * dim;
                    let target = ((batch_index * heads + head_index) * total_tokens
                        + token_cursor
                        + token_index)
                        * dim;
                    values[target..target + dim]
                        .copy_from_slice(&block.values()[source..source + dim]);
                }
            }
        }
        token_cursor = token_cursor
            .checked_add(block_tokens)
            .ok_or_else(|| Error::cache("cold KV concat token cursor overflow"))?;
    }
    F32Tensor::new(values, [batch, heads, total_tokens, dim])
}

fn gather_selected_tokens_from_blocks(
    token_indices: &[u32],
    blocks: &BTreeMap<usize, (ColdKvBlockMeta, F32Tensor)>,
) -> Result<F32Tensor> {
    let (_, first_tensor) = blocks
        .values()
        .next()
        .ok_or_else(|| Error::cache("cold KV selected read touched no blocks"))?;
    let first_dims = first_tensor.dims();
    if first_dims.len() != 4 {
        return Err(Error::cache(format!(
            "cold KV selected gather requires rank 4 [B,H,T,D], got {first_dims:?}"
        )));
    }
    let batch = first_dims[0];
    let heads = first_dims[1];
    let dim = first_dims[3];
    let selected_tokens = token_indices.len();
    let mut values = vec![0.0_f32; batch * heads * selected_tokens * dim];

    for (selected_index, token) in token_indices.iter().enumerate() {
        let token = usize::try_from(*token)
            .map_err(|_| Error::cache("cold KV selected token does not fit usize"))?;
        let (meta, tensor) = blocks
            .range(..=token)
            .next_back()
            .map(|(_, value)| value)
            .ok_or_else(|| Error::cache(format!("cold KV selected token {token} has no block")))?;
        let meta_end = meta
            .token_start
            .checked_add(meta.token_count)
            .ok_or_else(|| Error::cache("cold KV selected block token range overflow"))?;
        if token < meta.token_start || token >= meta_end {
            return Err(Error::cache(format!(
                "cold KV selected token {token} is not inside cached block [{},{meta_end})",
                meta.token_start
            )));
        }
        let dims = tensor.dims();
        validate_exact_shape(
            "cold_kv_selected_block_batch_heads_dim",
            &[dims[0], dims[1], dims[3]],
            &[batch, heads, dim],
        )?;
        let local_token = token - meta.token_start;
        if local_token >= dims[2] {
            return Err(Error::cache(format!(
                "cold KV selected local token {local_token} exceeds block token count {}",
                dims[2]
            )));
        }

        for batch_index in 0..batch {
            for head_index in 0..heads {
                let source = ((batch_index * heads + head_index) * dims[2] + local_token) * dim;
                let target =
                    ((batch_index * heads + head_index) * selected_tokens + selected_index) * dim;
                values[target..target + dim]
                    .copy_from_slice(&tensor.values()[source..source + dim]);
            }
        }
    }

    F32Tensor::new(values, [batch, heads, selected_tokens, dim])
}

fn gather_selected_q8_rows_from_blocks(
    token_indices: &[u32],
    blocks: &BTreeMap<usize, (ColdKvBlockMeta, Vec<u8>)>,
) -> Result<ColdKvSelectedQ8TensorRows> {
    let (first_meta, _) = blocks
        .values()
        .next()
        .ok_or_else(|| Error::cache("cold KV selected Q8 read touched no blocks"))?;
    let batch = first_meta.batch;
    let heads = first_meta.attention_heads;
    let dim = first_meta.head_dim;
    let selected_tokens = token_indices.len();
    let row_bytes = q8_row_byte_len(dim)?;

    let mut values =
        vec![
            0_u8;
            batch
                .checked_mul(heads)
                .and_then(|value| value.checked_mul(selected_tokens))
                .and_then(|value| value.checked_mul(row_bytes))
                .ok_or_else(|| Error::cache("cold KV selected Q8 payload length overflow"))?
        ];

    for (selected_index, token) in token_indices.iter().enumerate() {
        let token = usize::try_from(*token)
            .map_err(|_| Error::cache("cold KV selected token does not fit usize"))?;
        let (meta, payload) = blocks
            .range(..=token)
            .next_back()
            .map(|(_, value)| value)
            .ok_or_else(|| Error::cache(format!("cold KV selected token {token} has no block")))?;
        let meta_end = meta
            .token_start
            .checked_add(meta.token_count)
            .ok_or_else(|| Error::cache("cold KV selected Q8 block token range overflow"))?;
        if token < meta.token_start || token >= meta_end {
            return Err(Error::cache(format!(
                "cold KV selected token {token} is not inside cached block [{},{meta_end})",
                meta.token_start
            )));
        }
        validate_exact_shape(
            "cold_kv_selected_q8_block_batch_heads_dim",
            &[meta.batch, meta.attention_heads, meta.head_dim],
            &[batch, heads, dim],
        )?;
        validate_exact_shape(
            "cold_kv_selected_q8_payload_len",
            &[payload.len()],
            &[q8_payload_byte_len(batch, heads, meta.token_count, dim)?],
        )?;

        let local_token = token - meta.token_start;
        for batch_index in 0..batch {
            for head_index in 0..heads {
                let source_row =
                    ((batch_index * heads + head_index) * meta.token_count) + local_token;
                let target_row =
                    ((batch_index * heads + head_index) * selected_tokens) + selected_index;
                let source = source_row
                    .checked_mul(row_bytes)
                    .ok_or_else(|| Error::cache("cold KV selected Q8 source offset overflow"))?;
                let target = target_row
                    .checked_mul(row_bytes)
                    .ok_or_else(|| Error::cache("cold KV selected Q8 target offset overflow"))?;
                values[target..target + row_bytes]
                    .copy_from_slice(&payload[source..source + row_bytes]);
            }
        }
    }

    Ok(ColdKvSelectedQ8TensorRows {
        batch,
        attention_heads: heads,
        selected_tokens,
        head_dim: dim,
        payload: values,
    })
}

fn validate_block_tensor(
    context: &str,
    tensor: &F32Tensor,
    batch: usize,
    attention_heads: usize,
    head_dim: usize,
    max_block_tokens: usize,
) -> Result<usize> {
    let dims = tensor.dims();
    if dims.len() != 4 {
        return Err(Error::cache(format!(
            "{context} must be rank 4 [B,H,T,D], got {dims:?}"
        )));
    }
    validate_exact_shape(
        format!("{context}_batch_heads_dim"),
        &[dims[0], dims[1], dims[3]],
        &[batch, attention_heads, head_dim],
    )?;
    if dims[2] == 0 || dims[2] > max_block_tokens {
        return Err(Error::cache(format!(
            "{context} token count {} must be in 1..={max_block_tokens}",
            dims[2]
        )));
    }
    Ok(dims[2])
}

fn validate_layered_cold_appends(
    spec: &ColdKvStoreSpec,
    layers: &[LayerKvCacheAppend<'_>],
) -> Result<usize> {
    let mut sorted_layer_indices = layers
        .iter()
        .map(|layer| layer.layer_index)
        .collect::<Vec<_>>();
    sorted_layer_indices.sort_unstable();
    for pair in sorted_layer_indices.windows(2) {
        if pair[0] == pair[1] {
            return Err(Error::cache(format!(
                "layered cold KV prefill has duplicate layer index {}",
                pair[0]
            )));
        }
    }

    let first_tokens = validate_block_tensor(
        "layered cold KV key prefill",
        layers[0].k,
        spec.batch,
        spec.attention_heads,
        spec.key_head_dim,
        usize::MAX,
    )?;
    validate_exact_shape(
        "layered_cold_kv_first_value_tokens",
        &[validate_block_tensor(
            "layered cold KV value prefill",
            layers[0].v,
            spec.batch,
            spec.attention_heads,
            spec.value_head_dim,
            usize::MAX,
        )?],
        &[first_tokens],
    )?;

    for layer in &layers[1..] {
        let k_tokens = validate_block_tensor(
            "layered cold KV key prefill",
            layer.k,
            spec.batch,
            spec.attention_heads,
            spec.key_head_dim,
            usize::MAX,
        )?;
        let v_tokens = validate_block_tensor(
            "layered cold KV value prefill",
            layer.v,
            spec.batch,
            spec.attention_heads,
            spec.value_head_dim,
            usize::MAX,
        )?;
        validate_exact_shape(
            format!("layered_cold_kv_layer_{}_k_tokens", layer.layer_index),
            &[k_tokens],
            &[first_tokens],
        )?;
        validate_exact_shape(
            format!("layered_cold_kv_layer_{}_v_tokens", layer.layer_index),
            &[v_tokens],
            &[first_tokens],
        )?;
    }

    Ok(first_tokens)
}

fn slice_token_range(
    tensor: &F32Tensor,
    token_start: usize,
    token_count: usize,
) -> Result<F32Tensor> {
    let dims = tensor.dims();
    if dims.len() != 4 {
        return Err(Error::cache(format!(
            "cold KV token slice requires rank 4 [B,H,T,D], got {dims:?}"
        )));
    }
    let batch = dims[0];
    let heads = dims[1];
    let tokens = dims[2];
    let dim = dims[3];
    let token_end = token_start
        .checked_add(token_count)
        .ok_or_else(|| Error::cache("cold KV token slice end overflow"))?;
    if token_count == 0 || token_end > tokens {
        return Err(Error::cache(format!(
            "cold KV token slice [{token_start},{token_end}) is invalid for {tokens} tokens"
        )));
    }

    let mut values = vec![0.0_f32; batch * heads * token_count * dim];
    for b in 0..batch {
        for h in 0..heads {
            for local_t in 0..token_count {
                let source_t = token_start + local_t;
                let source_base = ((b * heads + h) * tokens + source_t) * dim;
                let target_base = ((b * heads + h) * token_count + local_t) * dim;
                values[target_base..target_base + dim]
                    .copy_from_slice(&tensor.values()[source_base..source_base + dim]);
            }
        }
    }
    F32Tensor::new(values, [batch, heads, token_count, dim])
}

fn layered_raw_f32_bytes(
    spec: &ColdKvStoreSpec,
    layer_count: usize,
    token_count: usize,
) -> Result<u64> {
    let values = layer_count
        .checked_mul(spec.batch)
        .and_then(|value| value.checked_mul(spec.attention_heads))
        .and_then(|value| value.checked_mul(token_count))
        .and_then(|value| value.checked_mul(spec.key_head_dim + spec.value_head_dim))
        .ok_or_else(|| Error::cache("layered cold KV raw f32 byte count overflow"))?;
    values
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| Error::cache("layered cold KV raw f32 byte count does not fit u64"))
}

fn encode_q8_rows(tensor: &F32Tensor) -> Result<Vec<u8>> {
    let dims = tensor.dims();
    if dims.len() != 4 {
        return Err(Error::cache(format!(
            "Q8 row cold KV encode requires rank 4 [B,H,T,D], got {dims:?}"
        )));
    }
    let batch = dims[0];
    let heads = dims[1];
    let tokens = dims[2];
    let dim = dims[3];
    let rows = batch
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(tokens))
        .ok_or_else(|| Error::cache("Q8 row cold KV row count overflow"))?;
    let mut payload = Vec::with_capacity(rows * (4 + dim));
    for row in 0..rows {
        let start = row
            .checked_mul(dim)
            .ok_or_else(|| Error::cache("Q8 row cold KV row offset overflow"))?;
        let values = &tensor.values()[start..start + dim];
        let max_abs = values
            .iter()
            .fold(0.0_f32, |max_abs, value| max_abs.max(value.abs()));
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        payload.extend_from_slice(&scale.to_le_bytes());
        for value in values {
            let quantized = (value / scale).round().clamp(-127.0, 127.0) as i8;
            payload.push(quantized as u8);
        }
    }
    Ok(payload)
}

fn decode_q8_rows(
    payload: &[u8],
    batch: usize,
    attention_heads: usize,
    token_count: usize,
    head_dim: usize,
) -> Result<F32Tensor> {
    let rows = batch
        .checked_mul(attention_heads)
        .and_then(|value| value.checked_mul(token_count))
        .ok_or_else(|| Error::cache("Q8 row cold KV decode row count overflow"))?;
    let row_bytes = 4_usize
        .checked_add(head_dim)
        .ok_or_else(|| Error::cache("Q8 row cold KV row byte count overflow"))?;
    let expected_payload_len = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::cache("Q8 row cold KV payload length overflow"))?;
    validate_exact_shape(
        "q8_row_cold_kv_payload_len",
        &[payload.len()],
        &[expected_payload_len],
    )?;

    let mut values = Vec::with_capacity(rows * head_dim);
    for row in 0..rows {
        let row_offset = row
            .checked_mul(row_bytes)
            .ok_or_else(|| Error::cache("Q8 row cold KV payload row offset overflow"))?;
        let scale = f32::from_le_bytes([
            payload[row_offset],
            payload[row_offset + 1],
            payload[row_offset + 2],
            payload[row_offset + 3],
        ]);
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::cache(format!(
                "Q8 row cold KV scale must be positive and finite, got {scale}"
            )));
        }
        for byte in &payload[row_offset + 4..row_offset + 4 + head_dim] {
            values.push((*byte as i8) as f32 * scale);
        }
    }

    F32Tensor::new(values, [batch, attention_heads, token_count, head_dim])
}

fn encode_record_header(header: &RecordHeader) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(RECORD_HEADER_LEN);
    bytes.extend_from_slice(&RECORD_MAGIC);
    push_u16(&mut bytes, RECORD_VERSION);
    bytes.push(header.tensor_kind.to_u8());
    bytes.push(header.codec.to_u8());
    push_u64(
        &mut bytes,
        checked_u64("cold KV layer_index", header.layer_index)?,
    );
    push_u64(
        &mut bytes,
        checked_u64("cold KV token_start", header.token_start)?,
    );
    push_u64(
        &mut bytes,
        checked_u64("cold KV token_count", header.token_count)?,
    );
    push_u64(&mut bytes, checked_u64("cold KV batch", header.batch)?);
    push_u64(
        &mut bytes,
        checked_u64("cold KV attention_heads", header.attention_heads)?,
    );
    push_u64(
        &mut bytes,
        checked_u64("cold KV head_dim", header.head_dim)?,
    );
    push_u64(&mut bytes, header.payload_len);
    validate_exact_shape(
        "cold_kv_record_header_final_len",
        &[bytes.len()],
        &[RECORD_HEADER_LEN],
    )?;
    Ok(bytes)
}

fn decode_record_header(bytes: &[u8]) -> Result<RecordHeader> {
    validate_exact_shape(
        "cold_kv_record_header_len",
        &[bytes.len()],
        &[RECORD_HEADER_LEN],
    )?;
    if bytes[0..4] != RECORD_MAGIC {
        return Err(Error::cache("cold KV record has invalid magic"));
    }
    let version = read_u16(bytes, 4)?;
    if version != RECORD_VERSION {
        return Err(Error::cache(format!(
            "unsupported cold KV record version {version}"
        )));
    }
    Ok(RecordHeader {
        tensor_kind: ColdKvTensorKind::from_u8(bytes[6])?,
        codec: ColdKvCodec::from_u8(bytes[7])?,
        layer_index: checked_usize("cold KV layer_index", read_u64(bytes, 8)?)?,
        token_start: checked_usize("cold KV token_start", read_u64(bytes, 16)?)?,
        token_count: checked_usize("cold KV token_count", read_u64(bytes, 24)?)?,
        batch: checked_usize("cold KV batch", read_u64(bytes, 32)?)?,
        attention_heads: checked_usize("cold KV attention_heads", read_u64(bytes, 40)?)?,
        head_dim: checked_usize("cold KV head_dim", read_u64(bytes, 48)?)?,
        payload_len: read_u64(bytes, 56)?,
    })
}

fn validate_record_matches_meta(header: &RecordHeader, meta: &ColdKvBlockMeta) -> Result<()> {
    if header.layer_index != meta.layer_index
        || header.tensor_kind != meta.tensor_kind
        || header.token_start != meta.token_start
        || header.token_count != meta.token_count
        || header.batch != meta.batch
        || header.attention_heads != meta.attention_heads
        || header.head_dim != meta.head_dim
        || header.codec != meta.codec
        || header.payload_len != meta.payload_len
    {
        return Err(Error::cache(format!(
            "cold KV record header does not match index metadata: header={header:?} meta={meta:?}"
        )));
    }
    Ok(())
}

fn push_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let end = offset
        .checked_add(2)
        .ok_or_else(|| Error::cache("cold KV u16 read offset overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| Error::cache("cold KV u16 read out of bounds"))?;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| Error::cache("cold KV u64 read offset overflow"))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| Error::cache("cold KV u64 read out of bounds"))?;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn checked_u64(context: &str, value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::cache(format!("{context} does not fit u64")))
}

fn checked_usize(context: &str, value: u64) -> Result<usize> {
    usize::try_from(value).map_err(|_| Error::cache(format!("{context} does not fit usize")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> ColdKvStoreSpec {
        ColdKvStoreSpec {
            batch: 1,
            attention_heads: 2,
            key_head_dim: 32,
            value_head_dim: 32,
            block_tokens: 3,
            codec: ColdKvCodec::Q8Row,
        }
    }

    fn tensor(tokens: usize, offset: f32) -> F32Tensor {
        let shape = [1, 2, tokens, 32];
        let count = shape.iter().product::<usize>();
        let values = (0..count)
            .map(|index| offset + index as f32 / 10.0)
            .collect::<Vec<_>>();
        F32Tensor::new(values, shape).unwrap()
    }

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "inferno-{name}-{}-{}.kv",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }

    fn assert_close(actual: &F32Tensor, expected: &F32Tensor, tolerance: f32) {
        assert_eq!(actual.dims(), expected.dims());
        for (index, (actual, expected)) in actual.values().iter().zip(expected.values()).enumerate()
        {
            let delta = (actual - expected).abs();
            assert!(
                delta <= tolerance,
                "value {index} differs: actual={actual} expected={expected} delta={delta}"
            );
        }
    }

    fn gather_expected_tokens(blocks: &[(&F32Tensor, usize)], token_indices: &[u32]) -> F32Tensor {
        let first = blocks[0].0;
        let batch = first.dims()[0];
        let heads = first.dims()[1];
        let dim = first.dims()[3];
        let mut values = vec![0.0_f32; batch * heads * token_indices.len() * dim];
        for (selected_index, token) in token_indices.iter().enumerate() {
            let token = *token as usize;
            let (block, block_start) = blocks
                .iter()
                .find(|(block, start)| token >= *start && token < *start + block.dims()[2])
                .copied()
                .unwrap();
            let local_token = token - block_start;
            for batch_index in 0..batch {
                for head_index in 0..heads {
                    let source =
                        ((batch_index * heads + head_index) * block.dims()[2] + local_token) * dim;
                    let target = ((batch_index * heads + head_index) * token_indices.len()
                        + selected_index)
                        * dim;
                    values[target..target + dim]
                        .copy_from_slice(&block.values()[source..source + dim]);
                }
            }
        }
        F32Tensor::new(values, [batch, heads, token_indices.len(), dim]).unwrap()
    }

    #[test]
    fn q8_row_codec_round_trips_with_bounded_error() {
        let source = tensor(2, -1.0);
        let payload = encode_q8_rows(&source).unwrap();
        let restored = decode_q8_rows(&payload, 1, 2, 2, 32).unwrap();

        assert_close(&restored, &source, 0.08);
        assert!(payload.len() < source.values().len() * std::mem::size_of::<f32>());
    }

    #[test]
    fn writes_and_reads_large_compressed_kv_blocks() {
        let path = temp_path("cold-kv-roundtrip");
        let mut store = ColdKvBlockStore::create(&path, spec()).unwrap();
        let k = tensor(3, 0.0);
        let v = tensor(3, 10.0);

        let report = store.write_layer_block(7, 0, &k, &v).unwrap();
        let stored_bytes = store.stored_bytes().unwrap();
        let (read_k, read_v) = store.read_layer_block(7, 0).unwrap();

        assert_eq!(store.block_count(), 2);
        assert_eq!(report.layer_index, 7);
        assert_eq!(report.token_start, 0);
        assert_eq!(report.token_count, 3);
        assert_eq!(
            stored_bytes,
            report.key_payload_bytes + report.value_payload_bytes + (RECORD_HEADER_LEN as u64 * 2)
        );
        assert!(stored_bytes < ((k.values().len() + v.values().len()) * 4) as u64);
        assert_close(&read_k, &k, 0.08);
        assert_close(&read_v, &v, 0.12);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn indexed_lookup_reads_exact_token_range() {
        let path = temp_path("cold-kv-indexed-range");
        let mut store = ColdKvBlockStore::create(&path, spec()).unwrap();
        let k = tensor(3, 0.0);
        let v = tensor(3, 10.0);

        store.write_layer_block(7, 9, &k, &v).unwrap();
        let key_meta = store
            .lookup_block(7, ColdKvTensorKind::Key, 9, 3)
            .cloned()
            .expect("key range should be indexed");
        let value_meta = store
            .lookup_block(7, ColdKvTensorKind::Value, 9, 3)
            .cloned()
            .expect("value range should be indexed");
        let read_k = store
            .read_tensor_range(7, ColdKvTensorKind::Key, 9, 3)
            .unwrap();
        let (read_layer_k, read_layer_v) = store.read_layer_range(7, 9, 3).unwrap();

        assert_eq!(key_meta.token_start, 9);
        assert_eq!(key_meta.token_count, 3);
        assert_eq!(value_meta.token_start, 9);
        assert_eq!(value_meta.token_count, 3);
        assert_close(&read_k, &k, 0.08);
        assert_close(&read_layer_k, &k, 0.08);
        assert_close(&read_layer_v, &v, 0.12);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn open_rebuilds_index_from_append_only_records() {
        let path = temp_path("cold-kv-reopen");
        let k0 = tensor(3, 0.0);
        let v0 = tensor(3, 10.0);
        let k1 = tensor(2, 20.0);
        let v1 = tensor(2, 30.0);
        {
            let mut store = ColdKvBlockStore::create(&path, spec()).unwrap();
            store.write_layer_block(2, 0, &k0, &v0).unwrap();
            store.write_layer_block(2, 3, &k1, &v1).unwrap();
            store.flush().unwrap();
        }

        let mut reopened = ColdKvBlockStore::open(&path, spec()).unwrap();
        let (read_k0, read_v0) = reopened.read_layer_range(2, 0, 3).unwrap();
        let (read_k1, read_v1) = reopened.read_layer_range(2, 3, 2).unwrap();

        assert_eq!(reopened.block_count(), 4);
        assert!(reopened
            .lookup_block(2, ColdKvTensorKind::Key, 3, 2)
            .is_some());
        assert_close(&read_k0, &k0, 0.08);
        assert_close(&read_v0, &v0, 0.12);
        assert_close(&read_k1, &k1, 0.25);
        assert_close(&read_v1, &v1, 0.25);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn reads_contiguous_range_across_multiple_append_only_blocks() {
        let path = temp_path("cold-kv-contiguous-range");
        let mut store = ColdKvBlockStore::create(&path, spec()).unwrap();
        let k0 = tensor(3, 0.0);
        let v0 = tensor(3, 10.0);
        let k1 = tensor(2, 20.0);
        let v1 = tensor(2, 30.0);

        store.write_layer_block(2, 0, &k0, &v0).unwrap();
        store.write_layer_block(2, 3, &k1, &v1).unwrap();
        let (read_k, read_v) = store.read_layer_range_contiguous(2, 0, 5).unwrap();
        let expected_k = concat_token_blocks(&[k0, k1]).unwrap();
        let expected_v = concat_token_blocks(&[v0, v1]).unwrap();

        assert_eq!(read_k.dims(), &[1, 2, 5, 32]);
        assert_eq!(read_v.dims(), &[1, 2, 5, 32]);
        assert_close(&read_k, &expected_k, 0.25);
        assert_close(&read_v, &expected_v, 0.25);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn selected_read_packs_only_requested_tokens_in_requested_order() {
        let path = temp_path("cold-kv-selected");
        let mut store = ColdKvBlockStore::create(&path, spec()).unwrap();
        let k0 = tensor(3, 0.0);
        let v0 = tensor(3, 10.0);
        let k1 = tensor(3, 100.0);
        let v1 = tensor(3, 110.0);

        store.write_layer_block(2, 0, &k0, &v0).unwrap();
        store.write_layer_block(2, 3, &k1, &v1).unwrap();
        let (read_k, read_v) = store.read_layer_selected_tokens(2, &[4, 1, 4]).unwrap();
        let expected_k = gather_expected_tokens(&[(&k0, 0), (&k1, 3)], &[4, 1, 4]);
        let expected_v = gather_expected_tokens(&[(&v0, 0), (&v1, 3)], &[4, 1, 4]);

        assert_eq!(read_k.dims(), &[1, 2, 3, 32]);
        assert_eq!(read_v.dims(), &[1, 2, 3, 32]);
        assert_close(&read_k, &expected_k, 0.60);
        assert_close(&read_v, &expected_v, 0.60);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn selected_q8_rows_preserve_requested_token_order() {
        let path = temp_path("cold-kv-selected-q8");
        let mut store = ColdKvBlockStore::create(&path, spec()).unwrap();
        let k0 = tensor(3, 0.0);
        let v0 = tensor(3, 10.0);
        let k1 = tensor(3, 100.0);
        let v1 = tensor(3, 110.0);

        store.write_layer_block(2, 0, &k0, &v0).unwrap();
        store.write_layer_block(2, 3, &k1, &v1).unwrap();
        let selected = store.read_layer_selected_q8_rows(2, &[4, 1, 4]).unwrap();
        let decoded_k = decode_q8_rows(
            &selected.key.payload,
            selected.key.batch,
            selected.key.attention_heads,
            selected.key.selected_tokens,
            selected.key.head_dim,
        )
        .unwrap();
        let decoded_v = decode_q8_rows(
            &selected.value.payload,
            selected.value.batch,
            selected.value.attention_heads,
            selected.value.selected_tokens,
            selected.value.head_dim,
        )
        .unwrap();
        let expected_k = gather_expected_tokens(&[(&k0, 0), (&k1, 3)], &[4, 1, 4]);
        let expected_v = gather_expected_tokens(&[(&v0, 0), (&v1, 3)], &[4, 1, 4]);

        assert_eq!(decoded_k.dims(), &[1, 2, 3, 32]);
        assert_eq!(decoded_v.dims(), &[1, 2, 3, 32]);
        assert_close(&decoded_k, &expected_k, 0.60);
        assert_close(&decoded_v, &expected_v, 0.60);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn cloned_reader_reads_stable_index_without_blocking_writer() {
        let path = temp_path("cold-kv-cloned-reader");
        let mut writer = ColdKvBlockStore::create(&path, spec()).unwrap();
        let k0 = tensor(3, 0.0);
        let v0 = tensor(3, 10.0);
        let k1 = tensor(1, 20.0);
        let v1 = tensor(1, 30.0);

        writer.write_layer_block(2, 0, &k0, &v0).unwrap();
        let mut reader = writer.clone_reader().unwrap();
        writer.write_layer_block(2, 3, &k1, &v1).unwrap();

        assert!(reader.read_layer_range_contiguous(2, 0, 3).is_ok());
        assert!(reader.read_layer_range_contiguous(2, 0, 4).is_err());
        assert!(writer.read_layer_range_contiguous(2, 0, 4).is_ok());

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn cloned_reader_has_an_independent_file_cursor() {
        let path = temp_path("cold-kv-independent-reader-cursor");
        let mut writer = ColdKvBlockStore::create(&path, spec()).unwrap();
        writer
            .write_layer_block(2, 0, &tensor(1, 0.0), &tensor(1, 10.0))
            .unwrap();
        let writer_offset = writer.file.stream_position().unwrap();
        let mut reader = writer.clone_reader().unwrap();

        reader.file.seek(SeekFrom::Start(7)).unwrap();

        assert_eq!(writer.file.stream_position().unwrap(), writer_offset);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn rejects_duplicate_layer_block() {
        let path = temp_path("cold-kv-duplicate");
        let mut store = ColdKvBlockStore::create(&path, spec()).unwrap();
        let k = tensor(2, 0.0);
        let v = tensor(2, 1.0);

        store.write_layer_block(0, 0, &k, &v).unwrap();
        let err = store.write_layer_block(0, 0, &k, &v).unwrap_err();

        assert!(err.to_string().contains("duplicate cold KV block"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn rejects_overlapping_token_ranges() {
        let path = temp_path("cold-kv-overlap");
        let mut store = ColdKvBlockStore::create(&path, spec()).unwrap();
        let k0 = tensor(3, 0.0);
        let v0 = tensor(3, 1.0);
        let k1 = tensor(2, 2.0);
        let v1 = tensor(2, 3.0);

        store.write_layer_block(0, 0, &k0, &v0).unwrap();
        let err = store.write_layer_block(0, 2, &k1, &v1).unwrap_err();

        assert!(err.to_string().contains("overlapping cold KV block"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn rejects_blocks_larger_than_configured_stream_block() {
        let path = temp_path("cold-kv-too-large");
        let mut store = ColdKvBlockStore::create(&path, spec()).unwrap();
        let k = tensor(4, 0.0);
        let v = tensor(4, 1.0);

        let err = store.write_layer_block(0, 0, &k, &v).unwrap_err();

        assert!(err.to_string().contains("token count 4"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn open_rejects_truncated_append_only_record() {
        let path = temp_path("cold-kv-truncated");
        let stored_bytes = {
            let mut store = ColdKvBlockStore::create(&path, spec()).unwrap();
            let k = tensor(3, 0.0);
            let v = tensor(3, 1.0);
            store.write_layer_block(0, 0, &k, &v).unwrap();
            store.flush().unwrap();
            store.stored_bytes().unwrap()
        };
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(stored_bytes - 1)
            .unwrap();

        let err = ColdKvBlockStore::open(&path, spec()).unwrap_err();

        assert!(err.to_string().contains("truncated cold KV record"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn layered_store_splits_prefill_into_large_blocks() {
        let path = temp_path("layered-cold-kv");
        let mut spec = spec();
        spec.block_tokens = 3;
        let mut tier = LayeredColdKvBlockStore::create(&path, spec).unwrap();
        let layer0_k = tensor(5, 0.0);
        let layer0_v = tensor(5, 10.0);
        let layer1_k = tensor(5, 20.0);
        let layer1_v = tensor(5, 30.0);

        let report = tier
            .write_prefill(&[
                LayerKvCacheAppend {
                    layer_index: 0,
                    k: &layer0_k,
                    v: &layer0_v,
                },
                LayerKvCacheAppend {
                    layer_index: 1,
                    k: &layer1_k,
                    v: &layer1_v,
                },
            ])
            .unwrap();
        let (tail_k, tail_v) = tier.read_layer_block(1, 3).unwrap();
        let expected_tail_k = slice_token_range(&layer1_k, 3, 2).unwrap();
        let expected_tail_v = slice_token_range(&layer1_v, 3, 2).unwrap();

        assert_eq!(tier.cached_tokens(), 5);
        assert_eq!(tier.layer_indices(), &[0, 1]);
        assert_eq!(report.layer_count, 2);
        assert_eq!(report.cached_tokens, 5);
        assert_eq!(report.logical_block_count, 2);
        assert_eq!(report.tensor_record_count, 8);
        assert!(report.compression_ratio < 1.0);
        assert_close(&tail_k, &expected_tail_k, 0.25);
        assert_close(&tail_v, &expected_tail_v, 0.25);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn layered_store_reopens_and_derives_layer_state() {
        let path = temp_path("layered-cold-kv-reopen");
        let mut spec = spec();
        spec.block_tokens = 3;
        let layer0_k = tensor(5, 0.0);
        let layer0_v = tensor(5, 10.0);
        let layer1_k = tensor(5, 20.0);
        let layer1_v = tensor(5, 30.0);
        {
            let mut tier = LayeredColdKvBlockStore::create(&path, spec.clone()).unwrap();
            tier.write_prefill(&[
                LayerKvCacheAppend {
                    layer_index: 0,
                    k: &layer0_k,
                    v: &layer0_v,
                },
                LayerKvCacheAppend {
                    layer_index: 1,
                    k: &layer1_k,
                    v: &layer1_v,
                },
            ])
            .unwrap();
            tier.store_mut().flush().unwrap();
        }

        let mut reopened = LayeredColdKvBlockStore::open(&path, spec).unwrap();
        let (tail_k, tail_v) = reopened.read_layer_block(1, 3).unwrap();
        let expected_tail_k = slice_token_range(&layer1_k, 3, 2).unwrap();
        let expected_tail_v = slice_token_range(&layer1_v, 3, 2).unwrap();

        assert_eq!(reopened.cached_tokens(), 5);
        assert_eq!(reopened.layer_indices(), &[0, 1]);
        assert_close(&tail_k, &expected_tail_k, 0.25);
        assert_close(&tail_v, &expected_tail_v, 0.25);

        std::fs::remove_file(path).ok();
    }
}
