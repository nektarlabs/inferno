use std::borrow::Cow;

use crate::{validate_exact_shape, Error, F32Tensor, Result};

#[derive(Debug, Clone)]
pub struct PagedKvPageView<'a> {
    pub physical_page_id: usize,
    pub start_token: usize,
    pub token_count: usize,
    pub k: Cow<'a, F32Tensor>,
    pub v: Cow<'a, F32Tensor>,
}

#[derive(Debug, Clone)]
pub struct PagedKvView<'a> {
    pub batch: usize,
    pub attention_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub page_size: usize,
    pub cached_tokens: usize,
    pub pages: Vec<PagedKvPageView<'a>>,
}

impl PagedKvView<'_> {
    pub fn validate(&self) -> Result<()> {
        if self.batch == 0
            || self.attention_heads == 0
            || self.key_head_dim == 0
            || self.value_head_dim == 0
            || self.page_size == 0
            || self.cached_tokens == 0
        {
            return Err(Error::cache(
                "paged KV view dimensions must be positive for decode attention",
            ));
        }
        if self.pages.is_empty() {
            return Err(Error::cache("paged KV view requires at least one page"));
        }

        let mut token_total = 0_usize;
        for page in &self.pages {
            if page.token_count == 0 || page.token_count > self.page_size {
                return Err(Error::cache(format!(
                    "paged KV page {} has invalid token_count {} for page_size {}",
                    page.physical_page_id, page.token_count, self.page_size
                )));
            }
            let k_dims = page.k.dims();
            let v_dims = page.v.dims();
            if k_dims.len() != 4 || v_dims.len() != 4 {
                return Err(Error::cache(format!(
                    "paged KV page {} tensors must be rank 4 [B,H,T,D], got k={k_dims:?} v={v_dims:?}",
                    page.physical_page_id
                )));
            }
            let k_page_tokens = k_dims[2];
            let v_page_tokens = v_dims[2];
            if k_page_tokens < page.token_count
                || v_page_tokens < page.token_count
                || k_page_tokens > self.page_size
                || v_page_tokens > self.page_size
            {
                return Err(Error::cache(format!(
                    "paged KV page {} capacity must contain token_count {} and fit page_size {}, got k_tokens={k_page_tokens} v_tokens={v_page_tokens}",
                    page.physical_page_id, page.token_count, self.page_size
                )));
            }
            validate_exact_shape(
                "paged_kv_page_k_shape",
                &[k_dims[0], k_dims[1], k_dims[3]],
                &[self.batch, self.attention_heads, self.key_head_dim],
            )?;
            validate_exact_shape(
                "paged_kv_page_v_shape",
                &[v_dims[0], v_dims[1], v_dims[3]],
                &[self.batch, self.attention_heads, self.value_head_dim],
            )?;
            token_total = token_total
                .checked_add(page.token_count)
                .ok_or_else(|| Error::cache("paged KV token count overflow"))?;
        }
        validate_exact_shape(
            "paged_kv_cached_tokens",
            &[token_total],
            &[self.cached_tokens],
        )?;
        Ok(())
    }
}
