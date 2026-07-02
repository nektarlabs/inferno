use std::borrow::Cow;

use common::{validate_exact_shape, Error, F32Tensor, PagedKvPageView, PagedKvView, Result, Shape};
use config::Config;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvCacheSpec {
    pub batch: usize,
    pub attention_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub max_context: usize,
    pub page_size: usize,
}

impl PagedKvCacheSpec {
    pub fn from_config(config: &Config, batch: usize, page_size: usize) -> Result<Self> {
        let spec = Self {
            batch,
            attention_heads: config.attention_heads,
            key_head_dim: config.qk_head_dim,
            value_head_dim: config.v_head_dim(),
            max_context: config.max_context,
            page_size,
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn validate(&self) -> Result<()> {
        if self.batch == 0 {
            return Err(Error::cache("batch must be positive"));
        }
        if self.attention_heads == 0 {
            return Err(Error::cache("attention_heads must be positive"));
        }
        if self.key_head_dim == 0 {
            return Err(Error::cache("key_head_dim must be positive"));
        }
        if self.value_head_dim == 0 {
            return Err(Error::cache("value_head_dim must be positive"));
        }
        if self.max_context == 0 {
            return Err(Error::cache("max_context must be positive"));
        }
        if self.page_size == 0 {
            return Err(Error::cache("page_size must be positive"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalTokenLocation {
    pub logical_token_index: usize,
    pub physical_page_id: usize,
    pub page_offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageStats {
    pub physical_page_id: usize,
    pub start_token: usize,
    pub token_count: usize,
    pub free_slots: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedCacheAppendReport {
    pub start_position: usize,
    pub appended_tokens: usize,
    pub end_position_exclusive: usize,
    pub allocated_pages: usize,
    pub page_count: usize,
    pub cached_tokens: usize,
}

#[derive(Debug, Clone)]
pub struct PagedKvCache {
    spec: PagedKvCacheSpec,
    cached_tokens: usize,
    pages: Vec<PhysicalPage>,
    token_locations: Vec<LogicalTokenLocation>,
    k_pages: Vec<F32Tensor>,
    v_pages: Vec<F32Tensor>,
}

#[derive(Debug, Clone)]
struct PhysicalPage {
    physical_page_id: usize,
    start_token: usize,
    token_count: usize,
}

impl PhysicalPage {
    fn is_full(&self, page_size: usize) -> bool {
        self.token_count == page_size
    }

    fn free_slots(&self, page_size: usize) -> usize {
        page_size.saturating_sub(self.token_count)
    }
}

impl PagedKvCache {
    pub fn new(spec: PagedKvCacheSpec) -> Result<Self> {
        spec.validate()?;
        Ok(Self {
            spec,
            cached_tokens: 0,
            pages: Vec::new(),
            token_locations: Vec::new(),
            k_pages: Vec::new(),
            v_pages: Vec::new(),
        })
    }

    pub fn spec(&self) -> &PagedKvCacheSpec {
        &self.spec
    }

    pub fn cached_tokens(&self) -> usize {
        self.cached_tokens
    }

    pub fn next_position(&self) -> usize {
        self.cached_tokens
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    #[cfg(test)]
    pub(crate) fn k_page_ptr(&self, page_index: usize) -> Result<*const f32> {
        self.k_pages
            .get(page_index)
            .map(|page| page.values().as_ptr())
            .ok_or_else(|| Error::cache(format!("K page {page_index} does not exist")))
    }

    pub fn token_location(&self, logical_token_index: usize) -> Result<&LogicalTokenLocation> {
        self.token_locations
            .get(logical_token_index)
            .ok_or_else(|| {
                Error::cache(format!(
                    "logical token index {logical_token_index} is outside cached token count {}",
                    self.cached_tokens
                ))
            })
    }

    pub fn token_locations(&self) -> &[LogicalTokenLocation] {
        &self.token_locations
    }

    pub fn page_stats(&self) -> Vec<PageStats> {
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

    pub fn view(&self) -> Result<PagedKvView<'_>> {
        validate_exact_shape(
            "paged_view_page_count",
            &[self.pages.len(), self.k_pages.len(), self.v_pages.len()],
            &[self.pages.len(), self.pages.len(), self.pages.len()],
        )?;
        let mut pages = Vec::with_capacity(self.pages.len());
        for (index, page) in self.pages.iter().enumerate() {
            pages.push(PagedKvPageView {
                physical_page_id: page.physical_page_id,
                start_token: page.start_token,
                token_count: page.token_count,
                k: Cow::Borrowed(&self.k_pages[index]),
                v: Cow::Borrowed(&self.v_pages[index]),
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

    pub fn append_prefill(
        &mut self,
        k: &F32Tensor,
        v: &F32Tensor,
    ) -> Result<PagedCacheAppendReport> {
        let append_tokens = validate_kv_append("prefill", &self.spec, k, v)?;
        self.append("prefill", k, v, append_tokens)
    }

    pub fn append_decode(
        &mut self,
        k: &F32Tensor,
        v: &F32Tensor,
    ) -> Result<PagedCacheAppendReport> {
        let append_tokens = validate_kv_append("decode", &self.spec, k, v)?;
        validate_exact_shape("paged_decode_token_count", &[append_tokens], &[1])?;
        self.append("decode", k, v, append_tokens)
    }

    pub fn reconstruct_keys(&self) -> Result<F32Tensor> {
        reconstruct_pages("K", &self.pages, &self.k_pages)
    }

    pub fn reconstruct_values(&self) -> Result<F32Tensor> {
        reconstruct_pages("V", &self.pages, &self.v_pages)
    }

    pub fn reconstruct_kv(&self) -> Result<(F32Tensor, F32Tensor)> {
        if self.pages.is_empty() {
            return Err(Error::cache("paged K/V cache is empty"));
        }
        validate_exact_shape(
            "paged_kv_page_count",
            &[self.k_pages.len(), self.v_pages.len()],
            &[self.pages.len(), self.pages.len()],
        )?;

        Ok((
            reconstruct_pages("K", &self.pages, &self.k_pages)?,
            reconstruct_pages("V", &self.pages, &self.v_pages)?,
        ))
    }

    pub fn decode_attention_scores_shape(&self, decode_tokens: usize) -> Result<Shape> {
        if decode_tokens == 0 {
            return Err(Error::cache("decode_tokens must be positive"));
        }
        if self.cached_tokens == 0 {
            return Err(Error::cache(
                "cannot build decode attention shape for an empty paged cache",
            ));
        }

        Ok(Shape::new(vec![
            self.spec.batch,
            self.spec.attention_heads,
            decode_tokens,
            self.cached_tokens,
        ]))
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
            .ok_or_else(|| Error::cache(format!("{context} paged append position overflow")))?;
        if end_position_exclusive > self.spec.max_context {
            return Err(Error::cache(format!(
                "{context} paged append would exceed max_context {}: current={} append={append_tokens}",
                self.spec.max_context, self.cached_tokens
            )));
        }

        let page_count_before = self.pages.len();
        let mut local_token_index = 0_usize;
        while local_token_index < append_tokens {
            if self.needs_new_page() {
                self.allocate_page();
            }

            let page_index = self.pages.len() - 1;
            let physical_page_id = self.pages[page_index].physical_page_id;
            let page_offset_start = self.pages[page_index].token_count;
            let free_slots = self.pages[page_index].free_slots(self.spec.page_size);
            let remaining_tokens = append_tokens - local_token_index;
            let chunk_tokens = free_slots.min(remaining_tokens);
            if chunk_tokens == 0 {
                return Err(Error::cache(format!(
                    "{context} paged append found a full page after allocation"
                )));
            }

            if page_offset_start == 0 {
                self.k_pages
                    .push(empty_page_tensor(&self.spec, self.spec.key_head_dim)?);
                self.v_pages
                    .push(empty_page_tensor(&self.spec, self.spec.value_head_dim)?);
            }
            write_token_chunk_into_page(
                "K page",
                &mut self.k_pages[page_index],
                k,
                local_token_index,
                page_offset_start,
                chunk_tokens,
            )?;
            write_token_chunk_into_page(
                "V page",
                &mut self.v_pages[page_index],
                v,
                local_token_index,
                page_offset_start,
                chunk_tokens,
            )?;

            for token_offset in 0..chunk_tokens {
                self.token_locations.push(LogicalTokenLocation {
                    logical_token_index: self.cached_tokens + token_offset,
                    physical_page_id,
                    page_offset: page_offset_start + token_offset,
                });
            }

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

    fn allocate_page(&mut self) {
        self.pages.push(PhysicalPage {
            physical_page_id: self.pages.len(),
            start_token: self.cached_tokens,
            token_count: 0,
        });
    }
}

fn reconstruct_pages(
    name: &str,
    physical_pages: &[PhysicalPage],
    tensor_pages: &[F32Tensor],
) -> Result<F32Tensor> {
    if tensor_pages.is_empty() {
        return Err(Error::cache(format!("{name} paged cache is empty")));
    }
    validate_exact_shape(
        format!("{name}_reconstruct_page_count"),
        &[physical_pages.len()],
        &[tensor_pages.len()],
    )?;
    concat_valid_token_pages(name, physical_pages, tensor_pages)
}

pub(crate) fn validate_kv_append(
    context: &str,
    spec: &PagedKvCacheSpec,
    k: &F32Tensor,
    v: &F32Tensor,
) -> Result<usize> {
    let k_dims = k.dims();
    let v_dims = v.dims();
    if k_dims.len() != 4 || v_dims.len() != 4 {
        return Err(Error::cache(format!(
            "{context} paged K/V tensors must be rank 4 [B,H,T,D], got k={k_dims:?} v={v_dims:?}"
        )));
    }

    validate_exact_shape(
        format!("{context}_paged_k_batch_heads_dim"),
        &[k_dims[0], k_dims[1], k_dims[3]],
        &[spec.batch, spec.attention_heads, spec.key_head_dim],
    )?;
    validate_exact_shape(
        format!("{context}_paged_v_batch_heads_dim"),
        &[v_dims[0], v_dims[1], v_dims[3]],
        &[spec.batch, spec.attention_heads, spec.value_head_dim],
    )?;
    validate_exact_shape(
        format!("{context}_paged_kv_tokens"),
        &[k_dims[2]],
        &[v_dims[2]],
    )?;
    if k_dims[2] == 0 {
        return Err(Error::cache(format!(
            "{context} paged append token count must be positive"
        )));
    }

    Ok(k_dims[2])
}

fn empty_page_tensor(spec: &PagedKvCacheSpec, dim: usize) -> Result<F32Tensor> {
    F32Tensor::zeros([spec.batch, spec.attention_heads, spec.page_size, dim])
}

fn write_token_chunk_into_page(
    name: &str,
    page: &mut F32Tensor,
    source: &F32Tensor,
    source_start_token: usize,
    page_start_token: usize,
    token_count: usize,
) -> Result<()> {
    let page_dims = page.dims();
    let source_dims = source.dims();
    if page_dims.len() != 4 || source_dims.len() != 4 {
        return Err(Error::cache(format!(
            "{name} write requires rank 4 [B,H,T,D], got page={page_dims:?} source={source_dims:?}"
        )));
    }
    validate_exact_shape(
        format!("{name}_write_batch_heads_dim"),
        &[source_dims[0], source_dims[1], source_dims[3]],
        &[page_dims[0], page_dims[1], page_dims[3]],
    )?;

    let source_end = source_start_token
        .checked_add(token_count)
        .ok_or_else(|| Error::cache(format!("{name} source token range overflow")))?;
    let page_end = page_start_token
        .checked_add(token_count)
        .ok_or_else(|| Error::cache(format!("{name} page token range overflow")))?;
    if token_count == 0 || source_end > source_dims[2] || page_end > page_dims[2] {
        return Err(Error::cache(format!(
            "{name} write token range is invalid: source=[{source_start_token},{source_end})/{}, page=[{page_start_token},{page_end})/{}",
            source_dims[2], page_dims[2]
        )));
    }

    let batch = page_dims[0];
    let heads = page_dims[1];
    let page_tokens = page_dims[2];
    let source_tokens = source_dims[2];
    let dim = page_dims[3];
    let source_values = source.values();
    let page_values = page.values_mut();
    for b in 0..batch {
        for h in 0..heads {
            for local_t in 0..token_count {
                let source_t = source_start_token + local_t;
                let page_t = page_start_token + local_t;
                let source_base = (((b * heads + h) * source_tokens + source_t) * dim) as usize;
                let page_base = (((b * heads + h) * page_tokens + page_t) * dim) as usize;
                page_values[page_base..page_base + dim]
                    .copy_from_slice(&source_values[source_base..source_base + dim]);
            }
        }
    }

    Ok(())
}

fn concat_valid_token_pages(
    name: &str,
    physical_pages: &[PhysicalPage],
    pages: &[F32Tensor],
) -> Result<F32Tensor> {
    let first = pages
        .first()
        .ok_or_else(|| Error::cache(format!("{name} concat requires at least one page")))?;
    let first_dims = first.dims();
    if first_dims.len() != 4 {
        return Err(Error::cache(format!(
            "{name} concat requires rank 4 [B,H,T,D], got {first_dims:?}"
        )));
    }

    let batch = first_dims[0];
    let heads = first_dims[1];
    let dim = first_dims[3];
    let mut total_tokens = 0_usize;
    for (physical_page, page) in physical_pages.iter().zip(pages) {
        let dims = page.dims();
        if dims.len() != 4 {
            return Err(Error::cache(format!(
                "{name} concat page must be rank 4 [B,H,T,D], got {dims:?}"
            )));
        }
        validate_exact_shape(
            format!("{name}_concat_batch_heads_dim"),
            &[dims[0], dims[1], dims[3]],
            &[batch, heads, dim],
        )?;
        if physical_page.token_count > dims[2] {
            return Err(Error::cache(format!(
                "{name} physical page {} token_count {} exceeds tensor capacity {}",
                physical_page.physical_page_id, physical_page.token_count, dims[2]
            )));
        }
        total_tokens = total_tokens
            .checked_add(physical_page.token_count)
            .ok_or_else(|| Error::cache(format!("{name} concat token count overflow")))?;
    }

    let mut values = vec![0.0_f32; batch * heads * total_tokens * dim];
    let mut target_token_offset = 0_usize;
    for (physical_page, page) in physical_pages.iter().zip(pages) {
        let page_capacity_tokens = page.dims()[2];
        let valid_tokens = physical_page.token_count;
        for b in 0..batch {
            for h in 0..heads {
                for t in 0..valid_tokens {
                    let source_base = (((b * heads + h) * page_capacity_tokens + t) * dim) as usize;
                    let target_base =
                        (((b * heads + h) * total_tokens + target_token_offset + t) * dim) as usize;
                    values[target_base..target_base + dim]
                        .copy_from_slice(&page.values()[source_base..source_base + dim]);
                }
            }
        }
        target_token_offset += valid_tokens;
    }

    F32Tensor::new(values, [batch, heads, total_tokens, dim])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_spec(page_size: usize, max_context: usize) -> PagedKvCacheSpec {
        PagedKvCacheSpec {
            batch: 1,
            attention_heads: 1,
            key_head_dim: 1,
            value_head_dim: 1,
            max_context,
            page_size,
        }
    }

    fn marker_tensor(values: &[f32]) -> F32Tensor {
        F32Tensor::new(values.to_vec(), [1, 1, values.len(), 1]).unwrap()
    }

    fn token_value(tensor: &F32Tensor, token_index: usize) -> f32 {
        tensor.values()[token_index]
    }

    #[test]
    fn page_boundaries_map_logical_tokens_to_page_offsets() {
        let mut cache = PagedKvCache::new(tiny_spec(4, 16)).unwrap();
        cache
            .append_prefill(
                &marker_tensor(&[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
                &marker_tensor(&[20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0]),
            )
            .unwrap();
        cache
            .append_decode(&marker_tensor(&[17.0]), &marker_tensor(&[27.0]))
            .unwrap();
        cache
            .append_decode(&marker_tensor(&[18.0]), &marker_tensor(&[28.0]))
            .unwrap();

        assert_eq!(cache.cached_tokens(), 9);
        assert_eq!(cache.page_count(), 3);
        assert_eq!(cache.token_location(0).unwrap().physical_page_id, 0);
        assert_eq!(cache.token_location(0).unwrap().page_offset, 0);
        assert_eq!(cache.token_location(3).unwrap().physical_page_id, 0);
        assert_eq!(cache.token_location(3).unwrap().page_offset, 3);
        assert_eq!(cache.token_location(4).unwrap().physical_page_id, 1);
        assert_eq!(cache.token_location(4).unwrap().page_offset, 0);
        assert_eq!(cache.token_location(8).unwrap().physical_page_id, 2);
        assert_eq!(cache.token_location(8).unwrap().page_offset, 0);

        let stats = cache.page_stats();
        assert_eq!(stats[0].token_count, 4);
        assert_eq!(stats[0].free_slots, 0);
        assert_eq!(stats[1].token_count, 4);
        assert_eq!(stats[1].free_slots, 0);
        assert_eq!(stats[2].token_count, 1);
        assert_eq!(stats[2].free_slots, 3);
    }

    #[test]
    fn reconstruct_contiguous_keys_preserves_token_order() {
        let mut cache = PagedKvCache::new(tiny_spec(2, 8)).unwrap();
        cache
            .append_prefill(
                &marker_tensor(&[1.0, 2.0, 3.0]),
                &marker_tensor(&[11.0, 12.0, 13.0]),
            )
            .unwrap();
        cache
            .append_decode(&marker_tensor(&[4.0]), &marker_tensor(&[14.0]))
            .unwrap();

        let reconstructed = cache.reconstruct_keys().unwrap();
        assert_eq!(reconstructed.dims(), &[1, 1, 4, 1]);
        assert_eq!(token_value(&reconstructed, 0), 1.0);
        assert_eq!(token_value(&reconstructed, 1), 2.0);
        assert_eq!(token_value(&reconstructed, 2), 3.0);
        assert_eq!(token_value(&reconstructed, 3), 4.0);
    }

    #[test]
    fn decode_appends_into_existing_page_storage() {
        let mut cache = PagedKvCache::new(tiny_spec(4, 8)).unwrap();
        cache
            .append_prefill(
                &marker_tensor(&[1.0, 2.0, 3.0]),
                &marker_tensor(&[11.0, 12.0, 13.0]),
            )
            .unwrap();
        let k_page_ptr = cache.k_pages[0].values().as_ptr();
        let v_page_ptr = cache.v_pages[0].values().as_ptr();

        let report = cache
            .append_decode(&marker_tensor(&[4.0]), &marker_tensor(&[14.0]))
            .unwrap();

        assert_eq!(report.allocated_pages, 0);
        assert_eq!(cache.k_pages[0].values().as_ptr(), k_page_ptr);
        assert_eq!(cache.v_pages[0].values().as_ptr(), v_page_ptr);
        assert_eq!(cache.k_pages[0].dims(), &[1, 1, 4, 1]);
        assert_eq!(cache.v_pages[0].dims(), &[1, 1, 4, 1]);
        assert_eq!(cache.k_pages[0].values(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.v_pages[0].values(), &[11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn view_exposes_pages_without_reconstructing_contiguous_cache() {
        let mut cache = PagedKvCache::new(tiny_spec(2, 8)).unwrap();
        cache
            .append_prefill(
                &marker_tensor(&[1.0, 2.0, 3.0]),
                &marker_tensor(&[11.0, 12.0, 13.0]),
            )
            .unwrap();
        cache
            .append_decode(&marker_tensor(&[4.0]), &marker_tensor(&[14.0]))
            .unwrap();

        let view = cache.view().unwrap();

        assert_eq!(view.cached_tokens, 4);
        assert_eq!(view.pages.len(), 2);
        assert_eq!(view.pages[0].physical_page_id, 0);
        assert_eq!(view.pages[0].start_token, 0);
        assert_eq!(view.pages[0].token_count, 2);
        assert_eq!(view.pages[0].k.values(), &[1.0, 2.0]);
        assert_eq!(view.pages[0].v.values(), &[11.0, 12.0]);
        assert_eq!(view.pages[1].physical_page_id, 1);
        assert_eq!(view.pages[1].start_token, 2);
        assert_eq!(view.pages[1].token_count, 2);
        assert_eq!(view.pages[1].k.values(), &[3.0, 4.0]);
        assert_eq!(view.pages[1].v.values(), &[13.0, 14.0]);
        assert_eq!(view.page_size, 2);
    }

    #[test]
    fn reconstruct_kv_returns_keys_and_values_together() {
        let mut cache = PagedKvCache::new(tiny_spec(2, 8)).unwrap();
        cache
            .append_prefill(
                &marker_tensor(&[1.0, 2.0, 3.0]),
                &marker_tensor(&[11.0, 12.0, 13.0]),
            )
            .unwrap();
        cache
            .append_decode(&marker_tensor(&[4.0]), &marker_tensor(&[14.0]))
            .unwrap();

        let (keys, values) = cache.reconstruct_kv().unwrap();

        assert_eq!(keys.dims(), &[1, 1, 4, 1]);
        assert_eq!(values.dims(), &[1, 1, 4, 1]);
        assert_eq!(token_value(&keys, 0), 1.0);
        assert_eq!(token_value(&keys, 3), 4.0);
        assert_eq!(token_value(&values, 0), 11.0);
        assert_eq!(token_value(&values, 3), 14.0);
    }

    #[test]
    fn decode_allocates_new_page_when_current_page_is_full() {
        let mut cache = PagedKvCache::new(tiny_spec(2, 8)).unwrap();
        cache
            .append_prefill(&marker_tensor(&[1.0, 2.0]), &marker_tensor(&[3.0, 4.0]))
            .unwrap();

        let report = cache
            .append_decode(&marker_tensor(&[5.0]), &marker_tensor(&[6.0]))
            .unwrap();

        assert_eq!(report.allocated_pages, 1);
        assert_eq!(report.page_count, 2);
        assert_eq!(cache.token_location(2).unwrap().physical_page_id, 1);
        assert_eq!(cache.token_location(2).unwrap().page_offset, 0);
    }

    #[test]
    fn token_location_rejects_out_of_range_index() {
        let mut cache = PagedKvCache::new(tiny_spec(2, 8)).unwrap();
        cache
            .append_prefill(&marker_tensor(&[1.0]), &marker_tensor(&[2.0]))
            .unwrap();

        let err = cache
            .token_location(1)
            .expect_err("token index 1 should be out of range");

        assert!(err.to_string().contains("outside cached token count"));
    }
}
