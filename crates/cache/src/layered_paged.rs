use common::{validate_exact_shape, Error, F32Tensor, PagedKvView, Result, Shape};
use config::Config;

use crate::paged::{
    validate_kv_append, PageStats, PagedCacheAppendReport, PagedKvCache, PagedKvCacheSpec,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayeredPagedKvCacheSpec {
    pub batch: usize,
    pub attention_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub max_context: usize,
    pub page_size: usize,
}

impl LayeredPagedKvCacheSpec {
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

    pub(crate) fn paged(&self) -> PagedKvCacheSpec {
        PagedKvCacheSpec {
            batch: self.batch,
            attention_heads: self.attention_heads,
            key_head_dim: self.key_head_dim,
            value_head_dim: self.value_head_dim,
            max_context: self.max_context,
            page_size: self.page_size,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LayerKvCacheAppend<'a> {
    pub layer_index: usize,
    pub k: &'a F32Tensor,
    pub v: &'a F32Tensor,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerPagedCacheAppendReport {
    pub layer_index: usize,
    pub cache_append: PagedCacheAppendReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayeredPagedCacheAppendReport {
    pub layer_count: usize,
    pub start_position: usize,
    pub appended_tokens: usize,
    pub end_position_exclusive: usize,
    pub cached_tokens: usize,
    pub next_position: usize,
    pub allocated_pages: usize,
    pub page_count: usize,
    pub next_decode_attention_scores_shape: Shape,
    pub layers: Vec<LayerPagedCacheAppendReport>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerPageStats {
    pub layer_index: usize,
    pub page_stats: Vec<PageStats>,
}

#[derive(Debug, Clone)]
pub struct LayerPagedKvView<'a> {
    pub layer_index: usize,
    pub view: PagedKvView<'a>,
}

#[derive(Debug, Clone)]
pub struct LayeredPagedKvCache {
    spec: LayeredPagedKvCacheSpec,
    layers: Vec<LayeredPagedKvCacheEntry>,
}

#[derive(Debug, Clone)]
struct LayeredPagedKvCacheEntry {
    layer_index: usize,
    cache: PagedKvCache,
}

impl LayeredPagedKvCache {
    pub fn new(spec: LayeredPagedKvCacheSpec) -> Result<Self> {
        spec.validate()?;
        Ok(Self {
            spec,
            layers: Vec::new(),
        })
    }

    pub fn spec(&self) -> &LayeredPagedKvCacheSpec {
        &self.spec
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

    pub fn append_prefill(
        &mut self,
        layers: &[LayerKvCacheAppend<'_>],
    ) -> Result<LayeredPagedCacheAppendReport> {
        if self.cached_tokens() != 0 || !self.layers.is_empty() {
            return Err(Error::cache(
                "layered paged KV cache prefill requires an empty cache",
            ));
        }
        validate_layer_appends("prefill", &self.spec, layers, None)?;

        let mut next_layers = Vec::with_capacity(layers.len());
        let mut reports = Vec::with_capacity(layers.len());
        for layer in layers {
            let mut cache = PagedKvCache::new(self.spec.paged())?;
            let cache_append = cache.append_prefill(layer.k, layer.v)?;
            reports.push(LayerPagedCacheAppendReport {
                layer_index: layer.layer_index,
                cache_append,
            });
            next_layers.push(LayeredPagedKvCacheEntry {
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
                "layered paged KV cache decode requires a completed prefill",
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
                        "decode layer {} was not initialized during prefill",
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

    pub fn reconstruct_layer_keys(&self, layer_index: usize) -> Result<F32Tensor> {
        self.layer(layer_index)?.cache.reconstruct_keys()
    }

    pub fn reconstruct_layer_values(&self, layer_index: usize) -> Result<F32Tensor> {
        self.layer(layer_index)?.cache.reconstruct_values()
    }

    pub fn reconstruct_layer_kv(&self, layer_index: usize) -> Result<(F32Tensor, F32Tensor)> {
        self.layer(layer_index)?.cache.reconstruct_kv()
    }

    pub fn layer_view(&self, layer_index: usize) -> Result<LayerPagedKvView<'_>> {
        let entry = self.layer(layer_index)?;
        Ok(LayerPagedKvView {
            layer_index: entry.layer_index,
            view: entry.cache.view()?,
        })
    }

    pub fn layer_page_stats(&self) -> Vec<LayerPageStats> {
        self.layers
            .iter()
            .map(|entry| LayerPageStats {
                layer_index: entry.layer_index,
                page_stats: entry.cache.page_stats(),
            })
            .collect()
    }

    fn layer(&self, layer_index: usize) -> Result<&LayeredPagedKvCacheEntry> {
        self.layers
            .iter()
            .find(|entry| entry.layer_index == layer_index)
            .ok_or_else(|| Error::cache(format!("layer {layer_index} is not cached")))
    }
}

fn validate_layer_appends(
    context: &str,
    spec: &LayeredPagedKvCacheSpec,
    layers: &[LayerKvCacheAppend<'_>],
    existing_layers: Option<&[LayeredPagedKvCacheEntry]>,
) -> Result<()> {
    if layers.is_empty() {
        return Err(Error::cache(format!(
            "{context} layered paged KV append requires at least one layer"
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
                "{context} layered paged KV append has duplicate layer index {}",
                pair[0]
            )));
        }
    }

    let first_tokens = validate_kv_append(context, &paged_spec, layers[0].k, layers[0].v)?;
    for layer in &layers[1..] {
        let tokens = validate_kv_append(context, &paged_spec, layer.k, layer.v)?;
        validate_exact_shape(
            format!("{context}_layer_{}_token_count", layer.layer_index),
            &[tokens],
            &[first_tokens],
        )?;
    }

    if let Some(existing_layers) = existing_layers {
        validate_exact_shape(
            format!("{context}_layer_count"),
            &[layers.len()],
            &[existing_layers.len()],
        )?;

        let mut existing_indices = existing_layers
            .iter()
            .map(|entry| entry.layer_index)
            .collect::<Vec<_>>();
        existing_indices.sort_unstable();
        validate_exact_shape(
            format!("{context}_layer_index_count"),
            &[sorted_layer_indices.len()],
            &[existing_indices.len()],
        )?;
        for (actual, expected) in sorted_layer_indices.iter().zip(existing_indices.iter()) {
            if actual != expected {
                return Err(Error::cache(format!(
                    "{context} layer set mismatch: got {sorted_layer_indices:?}, expected {existing_indices:?}"
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
        .ok_or_else(|| Error::cache("cannot summarize empty layered paged KV append"))?;
    let start_position = first.cache_append.start_position;
    let appended_tokens = first.cache_append.appended_tokens;
    let end_position_exclusive = first.cache_append.end_position_exclusive;
    let mut allocated_pages = 0_usize;
    let mut page_count = 0_usize;

    for report in &reports {
        validate_exact_shape(
            format!("layer_{}_paged_cache_start_position", report.layer_index),
            &[report.cache_append.start_position],
            &[start_position],
        )?;
        validate_exact_shape(
            format!("layer_{}_paged_cache_appended_tokens", report.layer_index),
            &[report.cache_append.appended_tokens],
            &[appended_tokens],
        )?;
        validate_exact_shape(
            format!("layer_{}_paged_cache_end_position", report.layer_index),
            &[report.cache_append.end_position_exclusive],
            &[end_position_exclusive],
        )?;
        allocated_pages = allocated_pages
            .checked_add(report.cache_append.allocated_pages)
            .ok_or_else(|| Error::cache("layered paged allocated page count overflow"))?;
        page_count = page_count
            .checked_add(report.cache_append.page_count)
            .ok_or_else(|| Error::cache("layered paged page count overflow"))?;
    }

    let next_decode_key_tokens = end_position_exclusive
        .checked_add(1)
        .ok_or_else(|| Error::cache("next paged decode attention key-token count overflow"))?;

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

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(page_size: usize, max_context: usize) -> LayeredPagedKvCacheSpec {
        LayeredPagedKvCacheSpec {
            batch: 1,
            attention_heads: 2,
            key_head_dim: 4,
            value_head_dim: 4,
            max_context,
            page_size,
        }
    }

    fn tensor(tokens: usize, offset: f32) -> F32Tensor {
        let shape = [1, 2, tokens, 4];
        let count = shape.iter().product::<usize>();
        let values = (0..count)
            .map(|index| offset + index as f32 / 100.0)
            .collect::<Vec<_>>();
        F32Tensor::new(values, shape).unwrap()
    }

    #[test]
    fn appends_prefill_and_decode_by_layer_across_pages() {
        let mut cache = LayeredPagedKvCache::new(spec(2, 8)).unwrap();
        let layer0_k = tensor(3, 0.0);
        let layer0_v = tensor(3, 1.0);
        let layer1_k = tensor(3, 2.0);
        let layer1_v = tensor(3, 3.0);

        let prefill = cache
            .append_prefill(&[
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

        assert_eq!(prefill.layer_count, 2);
        assert_eq!(prefill.appended_tokens, 3);
        assert_eq!(prefill.allocated_pages, 4);
        assert_eq!(prefill.page_count, 4);
        assert_eq!(cache.cached_tokens(), 3);
        assert_eq!(cache.page_count(), 4);
        assert_eq!(
            prefill.next_decode_attention_scores_shape.dims(),
            &[1, 2, 1, 4]
        );
        assert_eq!(
            cache.reconstruct_layer_keys(0).unwrap().dims(),
            &[1, 2, 3, 4]
        );
        assert_eq!(
            cache.reconstruct_layer_values(1).unwrap().dims(),
            &[1, 2, 3, 4]
        );
        let (layer0_keys, layer0_values) = cache.reconstruct_layer_kv(0).unwrap();
        assert_eq!(layer0_keys.dims(), &[1, 2, 3, 4]);
        assert_eq!(layer0_values.dims(), &[1, 2, 3, 4]);

        let layer0_decode_k = tensor(1, 4.0);
        let layer0_decode_v = tensor(1, 5.0);
        let layer1_decode_k = tensor(1, 6.0);
        let layer1_decode_v = tensor(1, 7.0);
        let layer0_open_page_k_ptr = cache.layers[0].cache.k_page_ptr(1).unwrap();
        let layer1_open_page_k_ptr = cache.layers[1].cache.k_page_ptr(1).unwrap();
        let decode = cache
            .append_decode(&[
                LayerKvCacheAppend {
                    layer_index: 0,
                    k: &layer0_decode_k,
                    v: &layer0_decode_v,
                },
                LayerKvCacheAppend {
                    layer_index: 1,
                    k: &layer1_decode_k,
                    v: &layer1_decode_v,
                },
            ])
            .unwrap();

        assert_eq!(decode.start_position, 3);
        assert_eq!(decode.appended_tokens, 1);
        assert_eq!(decode.allocated_pages, 0);
        assert_eq!(decode.page_count, 4);
        assert_eq!(cache.cached_tokens(), 4);
        assert_eq!(cache.page_count(), 4);
        assert_eq!(
            cache.layers[0].cache.k_page_ptr(1).unwrap(),
            layer0_open_page_k_ptr
        );
        assert_eq!(
            cache.layers[1].cache.k_page_ptr(1).unwrap(),
            layer1_open_page_k_ptr
        );
        assert_eq!(
            decode.next_decode_attention_scores_shape.dims(),
            &[1, 2, 1, 5]
        );
        assert_eq!(
            cache.reconstruct_layer_keys(0).unwrap().dims(),
            &[1, 2, 4, 4]
        );
        assert_eq!(
            cache.reconstruct_layer_values(1).unwrap().dims(),
            &[1, 2, 4, 4]
        );
        let layer_stats = cache.layer_page_stats();
        assert_eq!(layer_stats.len(), 2);
        assert_eq!(layer_stats[0].page_stats.len(), 2);
        assert_eq!(layer_stats[0].page_stats[0].token_count, 2);
        assert_eq!(layer_stats[0].page_stats[1].token_count, 2);

        let layer_view = cache.layer_view(1).unwrap();
        assert_eq!(layer_view.layer_index, 1);
        assert_eq!(layer_view.view.cached_tokens, 4);
        assert_eq!(layer_view.view.pages.len(), 2);
        assert_eq!(layer_view.view.pages[0].token_count, 2);
        assert_eq!(layer_view.view.pages[1].token_count, 2);
        assert_eq!(layer_view.view.pages[0].k.dims(), &[1, 2, 2, 4]);
        assert_eq!(layer_view.view.pages[0].v.dims(), &[1, 2, 2, 4]);
    }

    #[test]
    fn decode_rejects_layer_set_mismatch_without_mutating_cache() {
        let mut cache = LayeredPagedKvCache::new(spec(2, 8)).unwrap();
        let layer0_k = tensor(2, 0.0);
        let layer0_v = tensor(2, 1.0);
        cache
            .append_prefill(&[LayerKvCacheAppend {
                layer_index: 0,
                k: &layer0_k,
                v: &layer0_v,
            }])
            .unwrap();

        let layer1_k = tensor(1, 2.0);
        let layer1_v = tensor(1, 3.0);
        let err = cache
            .append_decode(&[LayerKvCacheAppend {
                layer_index: 1,
                k: &layer1_k,
                v: &layer1_v,
            }])
            .expect_err("wrong layer index must fail");

        assert!(err.to_string().contains("layer set mismatch"));
        assert_eq!(cache.cached_tokens(), 2);
        assert_eq!(cache.page_count(), 1);
    }

    #[test]
    fn decode_rejects_bad_shape_without_mutating_cache() {
        let mut cache = LayeredPagedKvCache::new(spec(2, 8)).unwrap();
        let prefill_k = tensor(2, 0.0);
        let prefill_v = tensor(2, 1.0);
        cache
            .append_prefill(&[LayerKvCacheAppend {
                layer_index: 0,
                k: &prefill_k,
                v: &prefill_v,
            }])
            .unwrap();

        let bad_k = F32Tensor::new(vec![1.0_f32; 1 * 2 * 1 * 3], [1, 2, 1, 3]).unwrap();
        let good_v = tensor(1, 2.0);
        let err = cache
            .append_decode(&[LayerKvCacheAppend {
                layer_index: 0,
                k: &bad_k,
                v: &good_v,
            }])
            .expect_err("bad decode shape must fail");

        assert!(err.to_string().contains("decode_paged_k_batch_heads_dim"));
        assert_eq!(cache.cached_tokens(), 2);
        assert_eq!(
            cache.reconstruct_layer_keys(0).unwrap().dims(),
            &[1, 2, 2, 4]
        );
    }

    #[test]
    fn append_rejects_duplicate_layer_index() {
        let mut cache = LayeredPagedKvCache::new(spec(2, 8)).unwrap();
        let k = tensor(2, 0.0);
        let v = tensor(2, 1.0);

        let err = cache
            .append_prefill(&[
                LayerKvCacheAppend {
                    layer_index: 0,
                    k: &k,
                    v: &v,
                },
                LayerKvCacheAppend {
                    layer_index: 0,
                    k: &k,
                    v: &v,
                },
            ])
            .expect_err("duplicate layer index must fail");

        assert!(err.to_string().contains("duplicate layer index"));
    }
}
