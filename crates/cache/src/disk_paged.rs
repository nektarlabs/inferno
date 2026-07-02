use std::{
    borrow::Cow,
    fs::{self, File, OpenOptions},
    io::{self, ErrorKind},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

use common::{validate_exact_shape, Error, F32Tensor, PagedKvPageView, PagedKvView, Result, Shape};

use crate::{
    layered_paged::{
        LayerKvCacheAppend, LayerPagedCacheAppendReport, LayeredPagedCacheAppendReport,
        LayeredPagedKvCacheSpec,
    },
    paged::{validate_kv_append, PageStats, PagedCacheAppendReport, PagedKvCacheSpec},
};

#[derive(Debug)]
pub struct LayeredDiskPagedKvCache {
    spec: LayeredPagedKvCacheSpec,
    root_dir: PathBuf,
    owns_root_dir: bool,
    layers: Vec<LayeredDiskPagedKvCacheEntry>,
}

#[derive(Debug)]
struct LayeredDiskPagedKvCacheEntry {
    layer_index: usize,
    cache: DiskPagedKvCache,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerDiskPageStats {
    pub layer_index: usize,
    pub page_stats: Vec<PageStats>,
}

#[derive(Debug, Clone)]
pub struct LayerDiskPagedKvView {
    pub layer_index: usize,
    pub view: PagedKvView<'static>,
}

#[derive(Debug)]
struct DiskPagedKvCache {
    spec: PagedKvCacheSpec,
    cached_tokens: usize,
    pages: Vec<PhysicalDiskPage>,
    k_file: File,
    v_file: File,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PhysicalDiskPage {
    physical_page_id: usize,
    start_token: usize,
    token_count: usize,
}

impl PhysicalDiskPage {
    fn is_full(&self, page_size: usize) -> bool {
        self.token_count == page_size
    }

    fn free_slots(&self, page_size: usize) -> usize {
        page_size.saturating_sub(self.token_count)
    }
}

impl LayeredDiskPagedKvCache {
    pub fn new_in_temp(spec: LayeredPagedKvCacheSpec) -> Result<Self> {
        let root_dir = unique_temp_dir()?;
        fs::create_dir_all(&root_dir).map_err(|error| {
            Error::cache(format!(
                "failed to create SSD KV cache directory {}: {error}",
                root_dir.display()
            ))
        })?;
        Self::new_with_dir(spec, root_dir, true)
    }

    pub fn new_with_dir(
        spec: LayeredPagedKvCacheSpec,
        root_dir: PathBuf,
        owns_root_dir: bool,
    ) -> Result<Self> {
        spec.validate()?;
        fs::create_dir_all(&root_dir).map_err(|error| {
            Error::cache(format!(
                "failed to create SSD KV cache directory {}: {error}",
                root_dir.display()
            ))
        })?;
        Ok(Self {
            spec,
            root_dir,
            owns_root_dir,
            layers: Vec::new(),
        })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn cached_tokens(&self) -> usize {
        self.layers
            .first()
            .map(|entry| entry.cache.cached_tokens())
            .unwrap_or(0)
    }

    pub fn next_position(&self) -> usize {
        self.cached_tokens()
    }

    pub fn page_count(&self) -> usize {
        self.layers
            .iter()
            .map(|entry| entry.cache.page_count())
            .sum()
    }

    pub fn root_dir(&self) -> &Path {
        &self.root_dir
    }

    pub fn spec(&self) -> &LayeredPagedKvCacheSpec {
        &self.spec
    }

    pub fn stored_bytes(&self) -> Result<u64> {
        self.layers.iter().try_fold(0_u64, |sum, layer| {
            let layer_bytes = layer.cache.stored_bytes()?;
            sum.checked_add(layer_bytes)
                .ok_or_else(|| Error::cache("SSD KV cache stored byte count overflow"))
        })
    }

    pub fn append_prefill(
        &mut self,
        layers: &[LayerKvCacheAppend<'_>],
    ) -> Result<LayeredPagedCacheAppendReport> {
        if self.cached_tokens() != 0 || !self.layers.is_empty() {
            return Err(Error::cache(
                "SSD paged KV cache prefill requires an empty cache",
            ));
        }
        validate_layer_appends("prefill", &self.spec, layers, None)?;

        let mut next_layers = Vec::with_capacity(layers.len());
        let mut reports = Vec::with_capacity(layers.len());
        for layer in layers {
            let layer_dir = self.root_dir.join(format!("layer-{}", layer.layer_index));
            fs::create_dir_all(&layer_dir).map_err(|error| {
                Error::cache(format!(
                    "failed to create SSD KV cache layer directory {}: {error}",
                    layer_dir.display()
                ))
            })?;
            let mut cache = DiskPagedKvCache::new(self.spec.paged(), &layer_dir)?;
            let cache_append = cache.append_prefill(layer.k, layer.v)?;
            reports.push(LayerPagedCacheAppendReport {
                layer_index: layer.layer_index,
                cache_append,
            });
            next_layers.push(LayeredDiskPagedKvCacheEntry {
                layer_index: layer.layer_index,
                cache,
            });
        }

        let report = summarize_layer_appends(&self.spec, reports)?;
        self.layers = next_layers;
        Ok(report)
    }

    pub fn append_decode(
        &mut self,
        layers: &[LayerKvCacheAppend<'_>],
    ) -> Result<LayeredPagedCacheAppendReport> {
        if self.layers.is_empty() {
            return Err(Error::cache(
                "SSD paged KV cache decode requires a completed prefill",
            ));
        }
        validate_layer_appends("decode", &self.spec, layers, Some(&self.layers))?;

        let mut reports = Vec::with_capacity(layers.len());
        for layer in layers {
            let entry = self
                .layers
                .iter_mut()
                .find(|entry| entry.layer_index == layer.layer_index)
                .ok_or_else(|| {
                    Error::cache(format!(
                        "decode layer {} was not initialized during SSD prefill",
                        layer.layer_index
                    ))
                })?;
            let cache_append = entry.cache.append_decode(layer.k, layer.v)?;
            reports.push(LayerPagedCacheAppendReport {
                layer_index: layer.layer_index,
                cache_append,
            });
        }

        summarize_layer_appends(&self.spec, reports)
    }

    pub fn layer_view_owned(&self, layer_index: usize) -> Result<LayerDiskPagedKvView> {
        let entry = self.layer(layer_index)?;
        Ok(LayerDiskPagedKvView {
            layer_index: entry.layer_index,
            view: entry.cache.view_owned()?,
        })
    }

    pub fn layer_page_stats(&self) -> Vec<LayerDiskPageStats> {
        self.layers
            .iter()
            .map(|entry| LayerDiskPageStats {
                layer_index: entry.layer_index,
                page_stats: entry.cache.page_stats(),
            })
            .collect()
    }

    fn layer(&self, layer_index: usize) -> Result<&LayeredDiskPagedKvCacheEntry> {
        self.layers
            .iter()
            .find(|entry| entry.layer_index == layer_index)
            .ok_or_else(|| Error::cache(format!("SSD cache layer {layer_index} is not cached")))
    }
}

impl Drop for LayeredDiskPagedKvCache {
    fn drop(&mut self) {
        if self.owns_root_dir {
            let _ = fs::remove_dir_all(&self.root_dir);
        }
    }
}

impl DiskPagedKvCache {
    fn new(spec: PagedKvCacheSpec, layer_dir: &Path) -> Result<Self> {
        spec.validate()?;
        let k_file = create_cache_file(&layer_dir.join("k.f32"))?;
        let v_file = create_cache_file(&layer_dir.join("v.f32"))?;
        Ok(Self {
            spec,
            cached_tokens: 0,
            pages: Vec::new(),
            k_file,
            v_file,
        })
    }

    fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    fn page_count(&self) -> usize {
        self.pages.len()
    }

    fn append_prefill(&mut self, k: &F32Tensor, v: &F32Tensor) -> Result<PagedCacheAppendReport> {
        let append_tokens = validate_kv_append("prefill", &self.spec, k, v)?;
        self.append("prefill", k, v, append_tokens)
    }

    fn append_decode(&mut self, k: &F32Tensor, v: &F32Tensor) -> Result<PagedCacheAppendReport> {
        let append_tokens = validate_kv_append("decode", &self.spec, k, v)?;
        validate_exact_shape("ssd_paged_decode_token_count", &[append_tokens], &[1])?;
        self.append("decode", k, v, append_tokens)
    }

    fn view_owned(&self) -> Result<PagedKvView<'static>> {
        if self.pages.is_empty() {
            return Err(Error::cache("SSD paged K/V cache is empty"));
        }
        let mut pages = Vec::with_capacity(self.pages.len());
        for page in &self.pages {
            let k = read_page_tensor(
                "K",
                &self.k_file,
                &self.spec,
                page.physical_page_id,
                page.token_count,
                self.spec.key_head_dim,
            )?;
            let v = read_page_tensor(
                "V",
                &self.v_file,
                &self.spec,
                page.physical_page_id,
                page.token_count,
                self.spec.value_head_dim,
            )?;
            pages.push(PagedKvPageView {
                physical_page_id: page.physical_page_id,
                start_token: page.start_token,
                token_count: page.token_count,
                k: Cow::Owned(k),
                v: Cow::Owned(v),
            });
        }
        let view = PagedKvView {
            batch: self.spec.batch,
            attention_heads: self.spec.attention_heads,
            key_head_dim: self.spec.key_head_dim,
            value_head_dim: self.spec.value_head_dim,
            page_size: self.spec.page_size,
            cached_tokens: self.cached_tokens,
            pages,
        };
        view.validate()?;
        Ok(view)
    }

    fn page_stats(&self) -> Vec<PageStats> {
        self.pages
            .iter()
            .map(|page| PageStats {
                physical_page_id: page.physical_page_id,
                start_token: page.start_token,
                token_count: page.token_count,
                free_slots: page.free_slots(self.spec.page_size),
            })
            .collect()
    }

    fn stored_bytes(&self) -> Result<u64> {
        let k_bytes = file_len_for_pages(self.pages.len(), &self.spec, self.spec.key_head_dim)?;
        let v_bytes = file_len_for_pages(self.pages.len(), &self.spec, self.spec.value_head_dim)?;
        k_bytes
            .checked_add(v_bytes)
            .ok_or_else(|| Error::cache("SSD KV cache layer stored byte count overflow"))
    }

    fn append(
        &mut self,
        context: &str,
        k: &F32Tensor,
        v: &F32Tensor,
        append_tokens: usize,
    ) -> Result<PagedCacheAppendReport> {
        let start_position = self.cached_tokens;
        let end_position_exclusive = start_position
            .checked_add(append_tokens)
            .ok_or_else(|| Error::cache(format!("{context} SSD paged append position overflow")))?;
        if end_position_exclusive > self.spec.max_context {
            return Err(Error::cache(format!(
                "{context} SSD paged append would exceed max_context {}: current={} append={append_tokens}",
                self.spec.max_context, self.cached_tokens
            )));
        }

        let page_count_before = self.pages.len();
        let mut local_token_index = 0_usize;
        while local_token_index < append_tokens {
            if self.needs_new_page() {
                self.allocate_page()?;
            }

            let page_index = self.pages.len() - 1;
            let physical_page_id = self.pages[page_index].physical_page_id;
            let page_offset_start = self.pages[page_index].token_count;
            let free_slots = self.pages[page_index].free_slots(self.spec.page_size);
            let remaining_tokens = append_tokens - local_token_index;
            let chunk_tokens = free_slots.min(remaining_tokens);
            if chunk_tokens == 0 {
                return Err(Error::cache(format!(
                    "{context} SSD paged append found a full page after allocation"
                )));
            }

            write_token_chunk_into_file(
                "K",
                &self.k_file,
                &self.spec,
                physical_page_id,
                k,
                local_token_index,
                page_offset_start,
                chunk_tokens,
                self.spec.key_head_dim,
            )?;
            write_token_chunk_into_file(
                "V",
                &self.v_file,
                &self.spec,
                physical_page_id,
                v,
                local_token_index,
                page_offset_start,
                chunk_tokens,
                self.spec.value_head_dim,
            )?;

            self.pages[page_index].token_count += chunk_tokens;
            self.cached_tokens += chunk_tokens;
            local_token_index += chunk_tokens;
        }

        Ok(PagedCacheAppendReport {
            start_position,
            appended_tokens: append_tokens,
            end_position_exclusive,
            allocated_pages: self.pages.len() - page_count_before,
            page_count: self.pages.len(),
            cached_tokens: self.cached_tokens,
        })
    }

    fn needs_new_page(&self) -> bool {
        match self.pages.last() {
            Some(page) => page.is_full(self.spec.page_size),
            None => true,
        }
    }

    fn allocate_page(&mut self) -> Result<()> {
        let physical_page_id = self.pages.len();
        self.pages.push(PhysicalDiskPage {
            physical_page_id,
            start_token: self.cached_tokens,
            token_count: 0,
        });
        let k_len = file_len_for_pages(self.pages.len(), &self.spec, self.spec.key_head_dim)?;
        let v_len = file_len_for_pages(self.pages.len(), &self.spec, self.spec.value_head_dim)?;
        self.k_file
            .set_len(k_len)
            .map_err(|error| Error::cache(format!("failed to grow SSD K cache file: {error}")))?;
        self.v_file
            .set_len(v_len)
            .map_err(|error| Error::cache(format!("failed to grow SSD V cache file: {error}")))?;
        Ok(())
    }
}

fn validate_layer_appends(
    context: &str,
    spec: &LayeredPagedKvCacheSpec,
    layers: &[LayerKvCacheAppend<'_>],
    existing_layers: Option<&[LayeredDiskPagedKvCacheEntry]>,
) -> Result<()> {
    if layers.is_empty() {
        return Err(Error::cache(format!(
            "{context} SSD layered paged KV append requires at least one layer"
        )));
    }

    let paged_spec = spec.paged();
    let mut sorted_layer_indices = layers
        .iter()
        .map(|layer| layer.layer_index)
        .collect::<Vec<_>>();
    sorted_layer_indices.sort_unstable();
    for pair in sorted_layer_indices.windows(2) {
        if pair[0] == pair[1] {
            return Err(Error::cache(format!(
                "{context} SSD layered paged KV append has duplicate layer index {}",
                pair[0]
            )));
        }
    }

    let first_tokens = validate_kv_append(context, &paged_spec, layers[0].k, layers[0].v)?;
    for layer in &layers[1..] {
        let tokens = validate_kv_append(context, &paged_spec, layer.k, layer.v)?;
        validate_exact_shape(
            format!("{context}_ssd_layer_{}_token_count", layer.layer_index),
            &[tokens],
            &[first_tokens],
        )?;
    }

    if let Some(existing_layers) = existing_layers {
        validate_exact_shape(
            format!("{context}_ssd_layer_count"),
            &[layers.len()],
            &[existing_layers.len()],
        )?;

        let mut existing_indices = existing_layers
            .iter()
            .map(|entry| entry.layer_index)
            .collect::<Vec<_>>();
        existing_indices.sort_unstable();
        validate_exact_shape(
            format!("{context}_ssd_layer_index_count"),
            &[sorted_layer_indices.len()],
            &[existing_indices.len()],
        )?;
        for (actual, expected) in sorted_layer_indices.iter().zip(existing_indices.iter()) {
            if actual != expected {
                return Err(Error::cache(format!(
                    "{context} SSD layer set mismatch: got {sorted_layer_indices:?}, expected {existing_indices:?}"
                )));
            }
        }
    }

    Ok(())
}

fn summarize_layer_appends(
    spec: &LayeredPagedKvCacheSpec,
    reports: Vec<LayerPagedCacheAppendReport>,
) -> Result<LayeredPagedCacheAppendReport> {
    let first = reports
        .first()
        .ok_or_else(|| Error::cache("cannot summarize empty SSD layered paged KV append"))?;
    let start_position = first.cache_append.start_position;
    let appended_tokens = first.cache_append.appended_tokens;
    let end_position_exclusive = first.cache_append.end_position_exclusive;
    let mut allocated_pages = 0_usize;
    let mut page_count = 0_usize;

    for report in &reports {
        validate_exact_shape(
            format!(
                "ssd_layer_{}_paged_cache_start_position",
                report.layer_index
            ),
            &[report.cache_append.start_position],
            &[start_position],
        )?;
        validate_exact_shape(
            format!(
                "ssd_layer_{}_paged_cache_appended_tokens",
                report.layer_index
            ),
            &[report.cache_append.appended_tokens],
            &[appended_tokens],
        )?;
        validate_exact_shape(
            format!("ssd_layer_{}_paged_cache_end_position", report.layer_index),
            &[report.cache_append.end_position_exclusive],
            &[end_position_exclusive],
        )?;
        allocated_pages = allocated_pages
            .checked_add(report.cache_append.allocated_pages)
            .ok_or_else(|| Error::cache("SSD layered paged allocated page count overflow"))?;
        page_count = page_count
            .checked_add(report.cache_append.page_count)
            .ok_or_else(|| Error::cache("SSD layered paged page count overflow"))?;
    }

    let next_decode_key_tokens = end_position_exclusive
        .checked_add(1)
        .ok_or_else(|| Error::cache("next SSD paged decode attention key-token count overflow"))?;

    Ok(LayeredPagedCacheAppendReport {
        layer_count: reports.len(),
        start_position,
        appended_tokens,
        end_position_exclusive,
        cached_tokens: end_position_exclusive,
        next_position: end_position_exclusive,
        allocated_pages,
        page_count,
        next_decode_attention_scores_shape: Shape::new(vec![
            spec.batch,
            spec.attention_heads,
            1,
            next_decode_key_tokens,
        ]),
        layers: reports,
    })
}

fn create_cache_file(path: &Path) -> Result<File> {
    match OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)
    {
        Ok(file) => Ok(file),
        Err(error) if error.kind() == ErrorKind::AlreadyExists => Err(Error::cache(format!(
            "SSD KV cache file already exists: {}",
            path.display()
        ))),
        Err(error) => Err(Error::cache(format!(
            "failed to create SSD KV cache file {}: {error}",
            path.display()
        ))),
    }
}

fn unique_temp_dir() -> Result<PathBuf> {
    let mut root = std::env::temp_dir();
    root.push(format!(
        "inferno-kv-cache-{}-{}",
        std::process::id(),
        unique_temp_suffix()
    ));
    Ok(root)
}

fn unique_temp_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

fn write_token_chunk_into_file(
    name: &str,
    file: &File,
    spec: &PagedKvCacheSpec,
    page_id: usize,
    source: &F32Tensor,
    source_start_token: usize,
    page_start_token: usize,
    token_count: usize,
    dim: usize,
) -> Result<()> {
    let source_dims = source.dims();
    if source_dims.len() != 4 {
        return Err(Error::cache(format!(
            "SSD {name} write requires rank 4 [B,H,T,D], got {source_dims:?}"
        )));
    }
    validate_exact_shape(
        format!("ssd_{name}_write_batch_heads_dim"),
        &[source_dims[0], source_dims[1], source_dims[3]],
        &[spec.batch, spec.attention_heads, dim],
    )?;
    let source_end = source_start_token
        .checked_add(token_count)
        .ok_or_else(|| Error::cache(format!("SSD {name} source token range overflow")))?;
    let page_end = page_start_token
        .checked_add(token_count)
        .ok_or_else(|| Error::cache(format!("SSD {name} page token range overflow")))?;
    if token_count == 0 || source_end > source_dims[2] || page_end > spec.page_size {
        return Err(Error::cache(format!(
            "SSD {name} write token range is invalid: source=[{source_start_token},{source_end})/{}, page=[{page_start_token},{page_end})/{}",
            source_dims[2], spec.page_size
        )));
    }

    let source_values = source.values();
    let mut row_bytes = Vec::with_capacity(dim * std::mem::size_of::<f32>());
    for batch in 0..spec.batch {
        for head in 0..spec.attention_heads {
            for local_t in 0..token_count {
                let source_t = source_start_token + local_t;
                let page_t = page_start_token + local_t;
                let source_base = (((batch * spec.attention_heads + head) * source_dims[2]
                    + source_t)
                    * dim) as usize;
                row_bytes.clear();
                encode_f32_slice(
                    &source_values[source_base..source_base + dim],
                    &mut row_bytes,
                );
                let offset = page_row_byte_offset(spec, page_id, batch, head, page_t, dim)?;
                write_all_at(file, &row_bytes, offset)?;
            }
        }
    }
    Ok(())
}

fn read_page_tensor(
    name: &str,
    file: &File,
    spec: &PagedKvCacheSpec,
    page_id: usize,
    token_count: usize,
    dim: usize,
) -> Result<F32Tensor> {
    if token_count == 0 || token_count > spec.page_size {
        return Err(Error::cache(format!(
            "SSD {name} read token_count {token_count} is invalid for page_size {}",
            spec.page_size
        )));
    }
    let mut values = vec![0.0_f32; spec.batch * spec.attention_heads * token_count * dim];
    let mut row_bytes = vec![0_u8; dim * std::mem::size_of::<f32>()];
    for batch in 0..spec.batch {
        for head in 0..spec.attention_heads {
            for token in 0..token_count {
                let offset = page_row_byte_offset(spec, page_id, batch, head, token, dim)?;
                read_exact_at(file, &mut row_bytes, offset)?;
                let target_base =
                    (((batch * spec.attention_heads + head) * token_count + token) * dim) as usize;
                decode_f32_slice(&row_bytes, &mut values[target_base..target_base + dim])?;
            }
        }
    }
    F32Tensor::new(values, [spec.batch, spec.attention_heads, token_count, dim])
}

fn file_len_for_pages(page_count: usize, spec: &PagedKvCacheSpec, dim: usize) -> Result<u64> {
    let values = page_count
        .checked_mul(spec.batch)
        .and_then(|value| value.checked_mul(spec.attention_heads))
        .and_then(|value| value.checked_mul(spec.page_size))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| Error::cache("SSD KV cache file element count overflow"))?;
    values
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| Error::cache("SSD KV cache file byte length overflow"))
}

fn page_row_byte_offset(
    spec: &PagedKvCacheSpec,
    page_id: usize,
    batch: usize,
    head: usize,
    token: usize,
    dim: usize,
) -> Result<u64> {
    let element_offset = page_id
        .checked_mul(spec.batch)
        .and_then(|value| value.checked_add(batch))
        .and_then(|value| value.checked_mul(spec.attention_heads))
        .and_then(|value| value.checked_add(head))
        .and_then(|value| value.checked_mul(spec.page_size))
        .and_then(|value| value.checked_add(token))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| Error::cache("SSD KV cache byte offset overflow"))?;
    element_offset
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| Error::cache("SSD KV cache byte offset overflow"))
}

fn encode_f32_slice(values: &[f32], output: &mut Vec<u8>) {
    output.reserve(values.len() * std::mem::size_of::<f32>());
    for value in values {
        output.extend_from_slice(&value.to_le_bytes());
    }
}

fn decode_f32_slice(bytes: &[u8], output: &mut [f32]) -> Result<()> {
    validate_exact_shape(
        "ssd_f32_decode_byte_count",
        &[bytes.len()],
        &[output.len() * std::mem::size_of::<f32>()],
    )?;
    for (index, chunk) in bytes.chunks_exact(std::mem::size_of::<f32>()).enumerate() {
        output[index] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(())
}

fn write_all_at(file: &File, mut bytes: &[u8], mut offset: u64) -> Result<()> {
    while !bytes.is_empty() {
        let written = file
            .write_at(bytes, offset)
            .map_err(|error| Error::cache(format!("failed to write SSD KV cache: {error}")))?;
        if written == 0 {
            return Err(Error::cache(
                "failed to write SSD KV cache: wrote zero bytes",
            ));
        }
        bytes = &bytes[written..];
        offset = offset
            .checked_add(written as u64)
            .ok_or_else(|| Error::cache("SSD KV cache write offset overflow"))?;
    }
    Ok(())
}

fn read_exact_at(file: &File, mut bytes: &mut [u8], mut offset: u64) -> Result<()> {
    while !bytes.is_empty() {
        let read = file.read_at(bytes, offset).map_err(|error| {
            Error::cache(format!(
                "failed to read SSD KV cache at byte {offset}: {error}"
            ))
        })?;
        if read == 0 {
            return Err(Error::cache(
                io::Error::from(ErrorKind::UnexpectedEof).to_string(),
            ));
        }
        let remaining = bytes;
        bytes = &mut remaining[read..];
        offset = offset
            .checked_add(read as u64)
            .ok_or_else(|| Error::cache("SSD KV cache read offset overflow"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_prefill_and_decode_to_ssd_pages() {
        let root =
            std::env::temp_dir().join(format!("inferno-test-ssd-kv-{}", unique_temp_suffix()));
        let spec = LayeredPagedKvCacheSpec {
            batch: 1,
            attention_heads: 1,
            key_head_dim: 2,
            value_head_dim: 1,
            max_context: 8,
            page_size: 2,
        };
        let mut cache = LayeredDiskPagedKvCache::new_with_dir(spec, root.clone(), true).unwrap();
        let prefill_k = F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 1, 3, 2]).unwrap();
        let prefill_v = F32Tensor::new(vec![10.0, 11.0, 12.0], [1, 1, 3, 1]).unwrap();
        cache
            .append_prefill(&[LayerKvCacheAppend {
                layer_index: 0,
                k: &prefill_k,
                v: &prefill_v,
            }])
            .unwrap();

        let decode_k = F32Tensor::new(vec![7.0, 8.0], [1, 1, 1, 2]).unwrap();
        let decode_v = F32Tensor::new(vec![13.0], [1, 1, 1, 1]).unwrap();
        cache
            .append_decode(&[LayerKvCacheAppend {
                layer_index: 0,
                k: &decode_k,
                v: &decode_v,
            }])
            .unwrap();

        let view = cache.layer_view_owned(0).unwrap().view;
        assert_eq!(view.cached_tokens, 4);
        assert_eq!(view.pages.len(), 2);
        assert_eq!(view.pages[0].k.values(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(view.pages[0].v.values(), &[10.0, 11.0]);
        assert_eq!(view.pages[1].k.values(), &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(view.pages[1].v.values(), &[12.0, 13.0]);

        drop(cache);
        assert!(!root.exists());
    }
}
