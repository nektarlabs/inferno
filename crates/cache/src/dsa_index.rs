use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use common::{validate_exact_shape, Error, F32Tensor, Result};

const RECORD_MAGIC: [u8; 4] = *b"IDX1";
const RECORD_VERSION: u16 = 1;
const RECORD_HEADER_LEN: usize = 56;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsaIndexStoreSpec {
    pub batch: usize,
    pub dim: usize,
    pub block_tokens: usize,
}

impl DsaIndexStoreSpec {
    pub fn validate(&self) -> Result<()> {
        if self.batch == 0 {
            return Err(Error::cache("DSA index store batch must be positive"));
        }
        if self.dim == 0 {
            return Err(Error::cache("DSA index store dim must be positive"));
        }
        if self.block_tokens == 0 {
            return Err(Error::cache(
                "DSA index store block_tokens must be positive",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct DsaIndexLayerAppend<'a> {
    pub layer_index: usize,
    pub index_key: &'a F32Tensor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsaIndexBlockMeta {
    pub layer_index: usize,
    pub token_start: usize,
    pub token_count: usize,
    pub batch: usize,
    pub dim: usize,
    pub file_offset: u64,
    pub payload_len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DsaIndexBlockKey {
    layer_index: usize,
    token_start: usize,
    token_count: usize,
}

#[derive(Debug)]
pub struct LayeredDsaIndexBlockStore {
    path: PathBuf,
    file: File,
    spec: DsaIndexStoreSpec,
    blocks: Vec<DsaIndexBlockMeta>,
    index: BTreeMap<DsaIndexBlockKey, usize>,
    layer_indices: Vec<usize>,
    cached_tokens: usize,
    read_bytes: Arc<AtomicU64>,
}

#[derive(Debug, Clone)]
struct RecordHeader {
    layer_index: usize,
    token_start: usize,
    token_count: usize,
    batch: usize,
    dim: usize,
    payload_len: u64,
}

impl LayeredDsaIndexBlockStore {
    pub fn create(path: impl AsRef<Path>, spec: DsaIndexStoreSpec) -> Result<Self> {
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
                    "failed to create DSA index block store {}: {error}",
                    path.display()
                ))
            })?;
        Ok(Self {
            path,
            file,
            spec,
            blocks: Vec::new(),
            index: BTreeMap::new(),
            layer_indices: Vec::new(),
            cached_tokens: 0,
            read_bytes: Arc::new(AtomicU64::new(0)),
        })
    }

    pub fn clone_reader(&self) -> Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .open(&self.path)
            .map_err(|error| {
                Error::cache(format!(
                    "failed to open DSA index block store reader {}: {error}",
                    self.path.display()
                ))
            })?;
        file.seek(SeekFrom::Start(0)).map_err(|error| {
            Error::cache(format!(
                "failed to seek cloned DSA index block store {}: {error}",
                self.path.display()
            ))
        })?;
        Ok(Self {
            path: self.path.clone(),
            file,
            spec: self.spec.clone(),
            blocks: self.blocks.clone(),
            index: self.index.clone(),
            layer_indices: self.layer_indices.clone(),
            cached_tokens: self.cached_tokens,
            read_bytes: Arc::clone(&self.read_bytes),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    pub fn stored_bytes(&self) -> Result<u64> {
        self.file
            .metadata()
            .map(|metadata| metadata.len())
            .map_err(|error| {
                Error::cache(format!(
                    "failed to read DSA index block store metadata {}: {error}",
                    self.path.display()
                ))
            })
    }

    /// Exact bytes read from the append-only file by this store and all
    /// readers cloned from it.
    pub fn read_bytes(&self) -> u64 {
        self.read_bytes.load(Ordering::Relaxed)
    }

    pub fn flush(&mut self) -> Result<()> {
        self.file.flush().map_err(|error| {
            Error::cache(format!(
                "failed to flush DSA index block store {}: {error}",
                self.path.display()
            ))
        })
    }

    pub fn write_prefill(&mut self, layers: &[DsaIndexLayerAppend<'_>]) -> Result<()> {
        if self.cached_tokens != 0 || !self.layer_indices.is_empty() {
            return Err(Error::cache(
                "DSA index prefill requires an empty block store",
            ));
        }
        if layers.is_empty() {
            return Ok(());
        }
        let token_count = validate_layer_appends(&self.spec, layers)?;
        let mut layer_indices = layers
            .iter()
            .map(|layer| layer.layer_index)
            .collect::<Vec<_>>();
        layer_indices.sort_unstable();

        let mut token_start = 0_usize;
        while token_start < token_count {
            let block_tokens = self.spec.block_tokens.min(token_count - token_start);
            for layer in layers {
                let block = slice_index_token_range(layer.index_key, token_start, block_tokens)?;
                self.write_layer_block(layer.layer_index, token_start, &block)?;
            }
            token_start = token_start
                .checked_add(block_tokens)
                .ok_or_else(|| Error::cache("DSA index prefill token_start overflow"))?;
        }

        self.layer_indices = layer_indices;
        self.cached_tokens = token_count;
        Ok(())
    }

    pub fn append_decode_layer(
        &mut self,
        layer_index: usize,
        token_start: usize,
        index_key: &F32Tensor,
    ) -> Result<()> {
        validate_index_tensor("DSA decode index key", index_key, &self.spec, Some(1))?;
        if !self.layer_indices.contains(&layer_index) {
            self.layer_indices.push(layer_index);
            self.layer_indices.sort_unstable();
        }
        self.write_layer_block(layer_index, token_start, index_key)?;
        self.cached_tokens = self.cached_tokens.max(token_start.saturating_add(1));
        Ok(())
    }

    pub fn read_layer_contiguous(
        &mut self,
        layer_index: usize,
        token_start: usize,
        token_count: usize,
    ) -> Result<F32Tensor> {
        if token_count == 0 {
            return Err(Error::cache(
                "DSA index contiguous read token_count must be positive",
            ));
        }
        if !self.layer_indices.contains(&layer_index) {
            return Err(Error::cache(format!(
                "DSA index layer {layer_index} is not present in the block store"
            )));
        }
        let token_end = token_start
            .checked_add(token_count)
            .ok_or_else(|| Error::cache("DSA index contiguous read range overflow"))?;
        if token_end > self.cached_tokens {
            return Err(Error::cache(format!(
                "DSA index read range [{token_start},{token_end}) exceeds cached tokens {}",
                self.cached_tokens
            )));
        }

        let mut cursor = token_start;
        let mut blocks = Vec::new();
        while cursor < token_end {
            let meta = self.find_block_containing(layer_index, cursor)?;
            let meta_end = meta
                .token_start
                .checked_add(meta.token_count)
                .ok_or_else(|| Error::cache("DSA index block range overflow"))?;
            if meta.token_start != cursor || meta_end > token_end {
                return Err(Error::cache(format!(
                    "DSA index block layer={layer_index} range=[{}, {}) does not exactly cover requested cursor {cursor} within [{token_start}, {token_end})",
                    meta.token_start, meta_end
                )));
            }
            cursor = meta_end;
            blocks.push(self.read_meta_block(&meta)?);
        }
        concat_index_blocks(&blocks)
    }

    fn write_layer_block(
        &mut self,
        layer_index: usize,
        token_start: usize,
        index_key: &F32Tensor,
    ) -> Result<()> {
        let token_count = validate_index_tensor(
            "DSA index block",
            index_key,
            &self.spec,
            Some(self.spec.block_tokens),
        )?;
        let payload = encode_f32_payload(index_key.values())?;
        let payload_len = u64::try_from(payload.len())
            .map_err(|_| Error::cache("DSA index payload length does not fit u64"))?;
        let header = RecordHeader {
            layer_index,
            token_start,
            token_count,
            batch: self.spec.batch,
            dim: self.spec.dim,
            payload_len,
        };
        self.write_record(header, &payload)
    }

    fn write_record(&mut self, header: RecordHeader, payload: &[u8]) -> Result<()> {
        validate_record_header(&header, &self.spec)?;
        validate_payload_len(&header, payload.len())?;
        let file_offset = self.file.seek(SeekFrom::End(0)).map_err(|error| {
            Error::cache(format!(
                "failed to seek DSA index block store {}: {error}",
                self.path.display()
            ))
        })?;
        let meta = DsaIndexBlockMeta {
            layer_index: header.layer_index,
            token_start: header.token_start,
            token_count: header.token_count,
            batch: header.batch,
            dim: header.dim,
            file_offset,
            payload_len: header.payload_len,
        };
        validate_no_overlap(&self.index, &meta)?;
        let mut record = encode_record_header(&header)?;
        record.extend_from_slice(payload);
        self.file.write_all(&record).map_err(|error| {
            Error::cache(format!(
                "failed to write DSA index block store {}: {error}",
                self.path.display()
            ))
        })?;
        insert_indexed_meta(&mut self.blocks, &mut self.index, meta)
    }

    fn find_block_containing(&self, layer_index: usize, token: usize) -> Result<DsaIndexBlockMeta> {
        self.blocks
            .iter()
            .find(|meta| {
                meta.layer_index == layer_index
                    && token >= meta.token_start
                    && token < meta.token_start + meta.token_count
            })
            .cloned()
            .ok_or_else(|| {
                Error::cache(format!(
                    "DSA index block containing token {token} not found for layer={layer_index}"
                ))
            })
    }

    fn read_meta_block(&mut self, meta: &DsaIndexBlockMeta) -> Result<F32Tensor> {
        self.file
            .seek(SeekFrom::Start(meta.file_offset))
            .map_err(|error| {
                Error::cache(format!(
                    "failed to seek DSA index block store {}: {error}",
                    self.path.display()
                ))
            })?;
        let record_len = RECORD_HEADER_LEN
            .checked_add(
                usize::try_from(meta.payload_len)
                    .map_err(|_| Error::cache("DSA index payload length does not fit usize"))?,
            )
            .ok_or_else(|| Error::cache("DSA index record length overflow"))?;
        let mut record = vec![0_u8; record_len];
        self.file.read_exact(&mut record).map_err(|error| {
            Error::cache(format!(
                "failed to read DSA index block store {}: {error}",
                self.path.display()
            ))
        })?;
        self.read_bytes.fetch_add(
            u64::try_from(record_len)
                .map_err(|_| Error::cache("DSA index record length does not fit u64"))?,
            Ordering::Relaxed,
        );
        let header = decode_record_header(&record[..RECORD_HEADER_LEN])?;
        validate_record_matches_meta(&header, meta)?;
        decode_f32_payload(
            &record[RECORD_HEADER_LEN..],
            header.batch,
            header.token_count,
            header.dim,
        )
    }
}

fn validate_layer_appends(
    spec: &DsaIndexStoreSpec,
    layers: &[DsaIndexLayerAppend<'_>],
) -> Result<usize> {
    let first = layers
        .first()
        .ok_or_else(|| Error::cache("DSA index prefill requires layers"))?;
    let token_count =
        validate_index_tensor("DSA index prefill first", first.index_key, spec, None)?;
    for layer in layers.iter().skip(1) {
        validate_exact_shape(
            format!("DSA index layer {} token_count", layer.layer_index),
            &[validate_index_tensor(
                "DSA index prefill layer",
                layer.index_key,
                spec,
                None,
            )?],
            &[token_count],
        )?;
    }
    Ok(token_count)
}

fn validate_index_tensor(
    context: &str,
    tensor: &F32Tensor,
    spec: &DsaIndexStoreSpec,
    max_tokens: Option<usize>,
) -> Result<usize> {
    let dims = tensor.dims();
    if dims.len() != 3 {
        return Err(Error::cache(format!(
            "{context} must be rank 3 [B,T,D], got {dims:?}"
        )));
    }
    validate_exact_shape(context, &[dims[0], dims[2]], &[spec.batch, spec.dim])?;
    if dims[1] == 0 {
        return Err(Error::cache(format!(
            "{context} token_count must be positive"
        )));
    }
    if let Some(max_tokens) = max_tokens {
        if dims[1] > max_tokens {
            return Err(Error::cache(format!(
                "{context} token_count {} exceeds block_tokens {max_tokens}",
                dims[1]
            )));
        }
    }
    Ok(dims[1])
}

fn slice_index_token_range(
    tensor: &F32Tensor,
    token_start: usize,
    token_count: usize,
) -> Result<F32Tensor> {
    let dims = tensor.dims();
    let (batch, tokens, dim) = (dims[0], dims[1], dims[2]);
    let token_end = token_start
        .checked_add(token_count)
        .ok_or_else(|| Error::cache("DSA index slice range overflow"))?;
    if token_end > tokens {
        return Err(Error::cache(format!(
            "DSA index slice [{token_start},{token_end}) exceeds token_count {tokens}"
        )));
    }
    let mut values = Vec::with_capacity(batch * token_count * dim);
    for batch_index in 0..batch {
        let source = (batch_index * tokens + token_start) * dim;
        values.extend_from_slice(&tensor.values()[source..source + token_count * dim]);
    }
    F32Tensor::new(values, [batch, token_count, dim])
}

fn concat_index_blocks(blocks: &[F32Tensor]) -> Result<F32Tensor> {
    let first = blocks
        .first()
        .ok_or_else(|| Error::cache("DSA index concat requires at least one block"))?;
    let dims = first.dims();
    let (batch, dim) = (dims[0], dims[2]);
    let total_tokens = blocks.iter().try_fold(0_usize, |sum, block| {
        let block_dims = block.dims();
        validate_exact_shape(
            "DSA index concat block shape",
            &[block_dims[0], block_dims[2]],
            &[batch, dim],
        )?;
        sum.checked_add(block_dims[1])
            .ok_or_else(|| Error::cache("DSA index concat token count overflow"))
    })?;
    let mut values = Vec::with_capacity(batch * total_tokens * dim);
    for batch_index in 0..batch {
        for block in blocks {
            let block_dims = block.dims();
            let block_tokens = block_dims[1];
            let source = batch_index * block_tokens * dim;
            values.extend_from_slice(&block.values()[source..source + block_tokens * dim]);
        }
    }
    F32Tensor::new(values, [batch, total_tokens, dim])
}

fn encode_f32_payload(values: &[f32]) -> Result<Vec<u8>> {
    let bytes_len = values
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| Error::cache("DSA index payload byte length overflow"))?;
    let mut bytes = Vec::with_capacity(bytes_len);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn decode_f32_payload(
    payload: &[u8],
    batch: usize,
    token_count: usize,
    dim: usize,
) -> Result<F32Tensor> {
    if payload.len() % std::mem::size_of::<f32>() != 0 {
        return Err(Error::cache(
            "DSA index payload byte length is not f32-aligned",
        ));
    }
    let expected = batch
        .checked_mul(token_count)
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| Error::cache("DSA index decode element count overflow"))?;
    validate_exact_shape(
        "DSA index payload element count",
        &[payload.len() / std::mem::size_of::<f32>()],
        &[expected],
    )?;
    let mut values = Vec::with_capacity(expected);
    for chunk in payload.chunks_exact(std::mem::size_of::<f32>()) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    F32Tensor::new(values, [batch, token_count, dim])
}

fn encode_record_header(header: &RecordHeader) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(RECORD_HEADER_LEN);
    bytes.extend_from_slice(&RECORD_MAGIC);
    bytes.extend_from_slice(&RECORD_VERSION.to_le_bytes());
    bytes.extend_from_slice(&0_u16.to_le_bytes());
    push_u64(&mut bytes, header.layer_index)?;
    push_u64(&mut bytes, header.token_start)?;
    push_u64(&mut bytes, header.token_count)?;
    push_u64(&mut bytes, header.batch)?;
    push_u64(&mut bytes, header.dim)?;
    bytes.extend_from_slice(&header.payload_len.to_le_bytes());
    validate_exact_shape(
        "DSA index record header length",
        &[bytes.len()],
        &[RECORD_HEADER_LEN],
    )?;
    Ok(bytes)
}

fn decode_record_header(bytes: &[u8]) -> Result<RecordHeader> {
    validate_exact_shape(
        "DSA index record header length",
        &[bytes.len()],
        &[RECORD_HEADER_LEN],
    )?;
    if bytes[0..4] != RECORD_MAGIC {
        return Err(Error::cache("invalid DSA index record magic"));
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != RECORD_VERSION {
        return Err(Error::cache(format!(
            "unsupported DSA index record version {version}"
        )));
    }
    Ok(RecordHeader {
        layer_index: read_u64(bytes, 8)?,
        token_start: read_u64(bytes, 16)?,
        token_count: read_u64(bytes, 24)?,
        batch: read_u64(bytes, 32)?,
        dim: read_u64(bytes, 40)?,
        payload_len: u64::from_le_bytes([
            bytes[48], bytes[49], bytes[50], bytes[51], bytes[52], bytes[53], bytes[54], bytes[55],
        ]),
    })
}

fn push_u64(bytes: &mut Vec<u8>, value: usize) -> Result<()> {
    let value = u64::try_from(value)
        .map_err(|_| Error::cache("DSA index header value does not fit u64"))?;
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<usize> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| Error::cache("DSA index header offset overflow"))?;
    let raw = u64::from_le_bytes(
        bytes[offset..end]
            .try_into()
            .map_err(|_| Error::cache("DSA index header field is truncated"))?,
    );
    usize::try_from(raw).map_err(|_| Error::cache("DSA index header value does not fit usize"))
}

fn validate_record_header(header: &RecordHeader, spec: &DsaIndexStoreSpec) -> Result<()> {
    if header.token_count == 0 || header.token_count > spec.block_tokens {
        return Err(Error::cache(format!(
            "DSA index record token_count {} must be in 1..={}",
            header.token_count, spec.block_tokens
        )));
    }
    validate_exact_shape("DSA index record batch", &[header.batch], &[spec.batch])?;
    validate_exact_shape("DSA index record dim", &[header.dim], &[spec.dim])
}

fn validate_payload_len(header: &RecordHeader, payload_len: usize) -> Result<()> {
    let expected = header
        .batch
        .checked_mul(header.token_count)
        .and_then(|value| value.checked_mul(header.dim))
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| Error::cache("DSA index payload length overflow"))?;
    validate_exact_shape("DSA index payload byte length", &[payload_len], &[expected])?;
    validate_exact_shape(
        "DSA index payload header length",
        &[usize::try_from(header.payload_len)
            .map_err(|_| Error::cache("DSA index payload header length does not fit usize"))?],
        &[payload_len],
    )
}

fn validate_record_matches_meta(header: &RecordHeader, meta: &DsaIndexBlockMeta) -> Result<()> {
    validate_exact_shape(
        "DSA index meta record",
        &[
            header.layer_index,
            header.token_start,
            header.token_count,
            header.batch,
            header.dim,
        ],
        &[
            meta.layer_index,
            meta.token_start,
            meta.token_count,
            meta.batch,
            meta.dim,
        ],
    )?;
    validate_exact_shape(
        "DSA index meta payload length",
        &[usize::try_from(header.payload_len)
            .map_err(|_| Error::cache("DSA index header payload length does not fit usize"))?],
        &[usize::try_from(meta.payload_len)
            .map_err(|_| Error::cache("DSA index meta payload length does not fit usize"))?],
    )
}

fn insert_indexed_meta(
    blocks: &mut Vec<DsaIndexBlockMeta>,
    index: &mut BTreeMap<DsaIndexBlockKey, usize>,
    meta: DsaIndexBlockMeta,
) -> Result<()> {
    let key = DsaIndexBlockKey {
        layer_index: meta.layer_index,
        token_start: meta.token_start,
        token_count: meta.token_count,
    };
    let block_index = blocks.len();
    let previous = index.insert(key, block_index);
    if previous.is_some() {
        return Err(Error::cache(format!(
            "duplicate DSA index block for layer={} token_range=[{},{})",
            key.layer_index,
            key.token_start,
            key.token_start.saturating_add(key.token_count)
        )));
    }
    blocks.push(meta);
    Ok(())
}

fn validate_no_overlap(
    index: &BTreeMap<DsaIndexBlockKey, usize>,
    meta: &DsaIndexBlockMeta,
) -> Result<()> {
    let candidate_end = meta
        .token_start
        .checked_add(meta.token_count)
        .ok_or_else(|| Error::cache("DSA index block range overflow"))?;
    for existing in index
        .keys()
        .filter(|key| key.layer_index == meta.layer_index)
    {
        let existing_end = existing
            .token_start
            .checked_add(existing.token_count)
            .ok_or_else(|| Error::cache("DSA index existing block range overflow"))?;
        if meta.token_start < existing_end && existing.token_start < candidate_end {
            return Err(Error::cache(format!(
                "overlapping DSA index block for layer={}: candidate=[{},{}) existing=[{},{})",
                meta.layer_index,
                meta.token_start,
                candidate_end,
                existing.token_start,
                existing_end
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dsa_index_store_round_trips_prefill_and_decode_append() {
        let path =
            std::env::temp_dir().join(format!("inferno-dsa-index-test-{}.idx", std::process::id()));
        let mut store = LayeredDsaIndexBlockStore::create(
            &path,
            DsaIndexStoreSpec {
                batch: 1,
                dim: 2,
                block_tokens: 2,
            },
        )
        .unwrap();
        let prefill = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]).unwrap();
        store
            .write_prefill(&[DsaIndexLayerAppend {
                layer_index: 7,
                index_key: &prefill,
            }])
            .unwrap();
        let current = F32Tensor::new(vec![5.0, 6.0], [1, 1, 2]).unwrap();
        store.append_decode_layer(7, 2, &current).unwrap();
        store.flush().unwrap();

        let read = store.read_layer_contiguous(7, 0, 3).unwrap();

        assert_eq!(read.dims(), &[1, 3, 2]);
        assert_eq!(read.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(store.read_bytes(), 2 * RECORD_HEADER_LEN as u64 + 6 * 4);
        std::fs::remove_file(path).ok();
    }
}
