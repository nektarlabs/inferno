use std::{path::Path, sync::Arc};

use crate::device_value::DeviceValue;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::Metal;
#[cfg(all(target_os = "macos", feature = "metal"))]
use crate::metal::QuantMatvecKind;
#[cfg(test)]
use common::Shape;
use common::{
    validate_exact_shape, BackendKind, DType, DeviceKind, DeviceReport, Error, F32Tensor,
    PagedKvView, Result,
};
use common::{Device, Tensor};
use inferno_io::{ExpertPackHeader, MappedBytes};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCapabilities {
    pub backend: BackendKind,
    pub device: DeviceKind,
    pub custom_kernels: bool,
    pub supports_f32: bool,
    pub supports_f16: bool,
    pub supports_bf16: bool,
    pub operations: Vec<&'static str>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BackendMemoryReport {
    pub total_physical_bytes: Option<u64>,
    pub process_rss_bytes: Option<u64>,
    pub process_virtual_bytes: Option<u64>,
    pub system_free_bytes: Option<u64>,
    pub system_active_bytes: Option<u64>,
    pub system_inactive_bytes: Option<u64>,
    pub system_wired_bytes: Option<u64>,
    pub system_compressed_bytes: Option<u64>,
    pub system_purgeable_bytes: Option<u64>,
    pub system_speculative_bytes: Option<u64>,
    pub swap_used_bytes: Option<u64>,
    pub metal_current_allocated_bytes: Option<u64>,
    pub metal_recommended_max_working_set_bytes: Option<u64>,
}

/// Stable Metal view layout for one exact Laguna Q2/Q3 GGUF mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LagunaModelViewReport {
    pub view_count: usize,
    pub model_bytes: usize,
    pub view_bytes: usize,
    pub max_view_bytes: usize,
    pub warmup_samples: usize,
}

/// Cumulative production-path metrics for the routed-expert cache.
///
/// A lookup represents one unique expert selected by one sparse layer. If
/// several token assignments select the same expert in that layer, the weight
/// triplet is looked up once and the assignments share it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExpertCacheMetrics {
    pub configured_slots_per_layer: u64,
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub ssd_read_bytes: u64,
    pub prefetch_lookups: u64,
    pub prefetch_hits: u64,
    pub prefetch_misses: u64,
    pub prefetch_ssd_read_bytes: u64,
    pub prefetch_nanoseconds: u64,
    pub transient_experts: u64,
    pub ready_waves: u64,
    pub resident_experts: u64,
    pub allocated_slots: u64,
    pub capacity_slots: u64,
    pub bytes_per_expert: u64,
    pub allocated_bytes: u64,
    pub capacity_bytes: u64,
    pub lookup_nanoseconds: u64,
    pub ssd_load_nanoseconds: u64,
    pub q2_matmul_gpu_nanoseconds: u64,
}

impl ExpertCacheMetrics {
    pub fn hit_rate(self) -> f64 {
        if self.lookups == 0 {
            return 0.0;
        }
        self.hits as f64 / self.lookups as f64
    }

    pub fn prefetch_hit_rate(self) -> f64 {
        if self.prefetch_lookups == 0 {
            return 0.0;
        }
        self.prefetch_hits as f64 / self.prefetch_lookups as f64
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RouterTopK {
    pub token_count: usize,
    pub expert_count: usize,
    pub top_k: usize,
    pub expert_ids: Vec<u32>,
    pub weights: Vec<f32>,
}

/// Opaque device-resident MoE routing result.
///
/// Expert IDs and weights remain in Metal buffers and are consumed directly
/// by the routed expert kernels. SSD streaming reads only the selected IDs at
/// the explicit sparse-layer synchronization point.
#[derive(Debug, Clone)]
pub struct DeviceRouterTopK {
    pub(crate) token_count: usize,
    pub(crate) expert_count: usize,
    pub(crate) top_k: usize,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) token_indices: ::metal::Buffer,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) expert_ids: ::metal::Buffer,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) expert_weights: ::metal::Buffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufExpertQuant {
    Q2K,
    Q3K,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufKQuant {
    Q4K,
    Q6K,
}

/// One row-major BF16 matrix prepared for native device execution.
#[derive(Debug, Clone)]
pub struct DeviceBf16Matrix {
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) storage_bytes: usize,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) buffer: ::metal::Buffer,
}

impl DeviceBf16Matrix {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn storage_bytes(&self) -> usize {
        self.storage_bytes
    }
}

/// Device-resident outputs of Laguna's shared Q/K/V/gate projection pass.
///
/// Q has shape `[B,T,48|72,128]`, K/V `[B,T,8,128]`, and gate
/// `[B,T,48|72]`. One native dispatch produces all four buffers from the same
/// normalized hidden states.
#[derive(Debug)]
pub struct LagunaAttentionProjections {
    pub query: DeviceValue,
    pub key: DeviceValue,
    pub value: DeviceValue,
    pub gate: DeviceValue,
}

/// Owns immutable packed INT4 data used by a zero-copy device weight.
pub trait W4WeightSource: std::fmt::Debug + Send + Sync {
    fn packed_bytes(&self) -> Result<&[u8]>;
    fn scale_bytes(&self) -> Result<&[u8]>;
}

/// One groupwise signed-INT4 matrix prepared for native device execution.
///
/// Packed values use the compressed-tensors convention: eight values per
/// little-endian I32 word, with stored nibbles in `0..=15` representing
/// signed values in `-8..=7`. `scales` contains one BF16 scale per group.
#[derive(Debug, Clone)]
pub struct DeviceW4Weight {
    pub(crate) in_features: usize,
    pub(crate) out_features: usize,
    pub(crate) group_size: usize,
    pub(crate) packed_bytes: usize,
    pub(crate) scale_bytes: usize,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) packed: ::metal::Buffer,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) scales: ::metal::Buffer,
    // Declared after the Metal buffers so those views are dropped first.
    pub(crate) _source_owner: Option<Arc<dyn W4WeightSource>>,
}

/// One routed INT4 expert and the router assignments that use it in a ready
/// execution wave.
///
/// Assignment indices refer to rows in the token-major `[tokens * top_k, ...]`
/// routed output. Keeping this metadata together lets the native backend batch
/// every expert that is currently resident or has finished loading from SSD.
#[derive(Debug, Clone, Copy)]
pub struct W4ExpertGroup<'a> {
    pub(crate) gate: &'a DeviceW4Weight,
    pub(crate) up: &'a DeviceW4Weight,
    pub(crate) down: &'a DeviceW4Weight,
    pub(crate) assignment_indices: &'a [u32],
}

impl<'a> W4ExpertGroup<'a> {
    pub fn new(
        gate: &'a DeviceW4Weight,
        up: &'a DeviceW4Weight,
        down: &'a DeviceW4Weight,
        assignment_indices: &'a [u32],
    ) -> Self {
        Self {
            gate,
            up,
            down,
            assignment_indices,
        }
    }
}

/// Immutable inverse-frequency table used by half-split RoPE on Metal.
///
/// Laguna owns two of these tables: a 64-wide YaRN table for global layers and
/// a 128-wide default table for sliding-window layers. Preparing them once
/// avoids uploading the same coefficients in every layer and token.
#[derive(Debug, Clone)]
pub struct DeviceRopeTable {
    pub(crate) rotary_dim: usize,
    pub(crate) attention_factor: f32,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) inverse_frequency: ::metal::Buffer,
}

/// Retention policy for Laguna's native FP8 KV cache.
///
/// Full-attention layers retain every token up to the configured capacity.
/// Sliding-attention layers retain only the checkpoint's 512-token window in
/// a ring buffer. Both policies use the same `[B, capacity, 8, 128]` byte
/// layout, so the attention kernel reads cache rows without a transpose or a
/// host-side reconstruction step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LagunaKvRetention {
    Full,
    Sliding,
}

/// Device-resident E4M3FN K/V cache for one Laguna attention layer.
///
/// The buffers contain one byte per K/V value. `total_tokens` is the absolute
/// sequence length, while `stored_tokens` is the number of rows currently
/// addressable in the buffer. For sliding attention, physical slot
/// `absolute_position % capacity_tokens` holds the corresponding logical row.
#[derive(Debug)]
pub struct LagunaFp8KvCache {
    pub(crate) batch: usize,
    pub(crate) capacity_tokens: usize,
    pub(crate) stored_tokens: usize,
    pub(crate) total_tokens: usize,
    pub(crate) retention: LagunaKvRetention,
    pub(crate) key_scale: f32,
    pub(crate) value_scale: f32,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) key: ::metal::Buffer,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) value: ::metal::Buffer,
}

/// Device-resident F16 K/V cache used by the Antirez Laguna GGUF path.
///
/// Layout and retention match `LagunaFp8KvCache`, but each cached value uses
/// IEEE F16 and therefore needs no checkpoint-specific quantization scale.
#[derive(Debug)]
pub struct LagunaF16KvCache {
    pub(crate) batch: usize,
    pub(crate) capacity_tokens: usize,
    pub(crate) stored_tokens: usize,
    pub(crate) total_tokens: usize,
    pub(crate) retention: LagunaKvRetention,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) key: ::metal::Buffer,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) value: ::metal::Buffer,
}

impl LagunaF16KvCache {
    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn capacity_tokens(&self) -> usize {
        self.capacity_tokens
    }

    pub fn stored_tokens(&self) -> usize {
        self.stored_tokens
    }

    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    pub fn retention(&self) -> LagunaKvRetention {
        self.retention
    }

    pub fn storage_bytes(&self) -> Result<usize> {
        self.batch
            .checked_mul(self.capacity_tokens)
            .and_then(|values| values.checked_mul(8))
            .and_then(|values| values.checked_mul(128))
            .and_then(|values| values.checked_mul(2))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| Error::cache("Laguna F16 KV storage byte count overflow"))
    }

    pub fn reset(&mut self) {
        self.stored_tokens = 0;
        self.total_tokens = 0;
    }

    fn commit_append(&mut self, token_count: usize) -> Result<()> {
        let total_tokens = self
            .total_tokens
            .checked_add(token_count)
            .ok_or_else(|| Error::cache("Laguna F16 KV token count overflow"))?;
        if self.retention == LagunaKvRetention::Full && total_tokens > self.capacity_tokens {
            return Err(Error::cache(format!(
                "Laguna full-attention F16 KV capacity {} is smaller than sequence length {total_tokens}",
                self.capacity_tokens
            )));
        }
        self.total_tokens = total_tokens;
        self.stored_tokens = total_tokens.min(self.capacity_tokens);
        Ok(())
    }
}

impl LagunaFp8KvCache {
    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn capacity_tokens(&self) -> usize {
        self.capacity_tokens
    }

    pub fn stored_tokens(&self) -> usize {
        self.stored_tokens
    }

    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    pub fn retention(&self) -> LagunaKvRetention {
        self.retention
    }

    pub fn storage_bytes(&self) -> Result<usize> {
        self.batch
            .checked_mul(self.capacity_tokens)
            .and_then(|values| values.checked_mul(8))
            .and_then(|values| values.checked_mul(128))
            .and_then(|values| values.checked_mul(2))
            .ok_or_else(|| Error::cache("Laguna FP8 KV storage byte count overflow"))
    }

    pub fn reset(&mut self) {
        self.stored_tokens = 0;
        self.total_tokens = 0;
    }

    fn commit_append(&mut self, token_count: usize) -> Result<()> {
        let total_tokens = self
            .total_tokens
            .checked_add(token_count)
            .ok_or_else(|| Error::cache("Laguna FP8 KV token count overflow"))?;
        if self.retention == LagunaKvRetention::Full && total_tokens > self.capacity_tokens {
            return Err(Error::cache(format!(
                "Laguna full-attention FP8 KV capacity {} is smaller than sequence length {total_tokens}",
                self.capacity_tokens
            )));
        }
        self.total_tokens = total_tokens;
        self.stored_tokens = total_tokens.min(self.capacity_tokens);
        Ok(())
    }
}

impl DeviceRopeTable {
    pub fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }

    pub fn attention_factor(&self) -> f32 {
        self.attention_factor
    }
}

impl DeviceW4Weight {
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn group_size(&self) -> usize {
        self.group_size
    }

    pub fn storage_bytes(&self) -> Result<usize> {
        self.packed_bytes
            .checked_add(self.scale_bytes)
            .ok_or_else(|| Error::backend("device W4 storage byte count overflow"))
    }
}

impl DeviceRouterTopK {
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn expert_count(&self) -> usize {
        self.expert_count
    }

    pub fn top_k(&self) -> usize {
        self.top_k
    }

    pub fn assignment_count(&self) -> Result<usize> {
        self.token_count
            .checked_mul(self.top_k)
            .ok_or_else(|| Error::backend("device router assignment count overflow"))
    }
}

/// Routed expert rows produced as selected Q2 weights become available.
///
/// `output` preserves router assignment order as
/// `[token_count * top_k, hidden_size]`, so the normal weighted combine kernel
/// can consume it without changing model semantics. On the asynchronous Metal
/// path, the backend must encode this result's completion dependency before a
/// consumer reads `output`.
#[derive(Debug, Clone)]
pub struct DeviceRoutedExperts {
    output: DeviceValue,
    completion_value: u64,
    selected_experts: usize,
    cache_hits: usize,
    cache_misses: usize,
    transient_experts: usize,
    read_bytes: u64,
    ready_waves: usize,
}

impl DeviceRoutedExperts {
    pub fn output(&self) -> &DeviceValue {
        &self.output
    }

    pub fn selected_experts(&self) -> usize {
        self.selected_experts
    }

    pub fn cache_hits(&self) -> usize {
        self.cache_hits
    }

    pub fn cache_misses(&self) -> usize {
        self.cache_misses
    }

    pub fn transient_experts(&self) -> usize {
        self.transient_experts
    }

    pub fn read_bytes(&self) -> u64 {
        self.read_bytes
    }

    pub fn ready_waves(&self) -> usize {
        self.ready_waves
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Q2ExpertSource<'a> {
    pub expert_id: u32,
    pub bytes: &'a [u8],
    pub absolute_offset: u64,
}

/// Device-resident paged KV view for the Metal decode path.
///
/// The K/V buffers use an append-only layout:
///
/// `[page_count, batch, attention_heads, page_size, head_dim]`
///
/// `cached_tokens` tells attention how much of that capacity is valid. This
/// avoids re-uploading the full KV history every decode step; decode attention
/// reads the existing page buffer and the current token's K/V buffer directly.
#[derive(Debug, Clone)]
pub struct DevicePagedKvView {
    pub batch: usize,
    pub attention_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub page_size: usize,
    pub cached_tokens: usize,
    pub capacity_tokens: usize,
    pub k: DeviceValue,
    pub v: DeviceValue,
}

impl DevicePagedKvView {
    pub fn validate(&self) -> Result<()> {
        if self.batch == 0
            || self.attention_heads == 0
            || self.key_head_dim == 0
            || self.value_head_dim == 0
            || self.page_size == 0
            || self.cached_tokens == 0
            || self.capacity_tokens == 0
        {
            return Err(Error::cache(
                "device paged KV view dimensions must be positive for decode attention",
            ));
        }
        if self.cached_tokens > self.capacity_tokens {
            return Err(Error::cache(format!(
                "device paged KV cached_tokens {} exceeds capacity_tokens {}",
                self.cached_tokens, self.capacity_tokens
            )));
        }
        if self.capacity_tokens % self.page_size != 0 {
            return Err(Error::cache(format!(
                "device paged KV capacity_tokens {} must be a multiple of page_size {}",
                self.capacity_tokens, self.page_size
            )));
        }
        let page_count = self.capacity_tokens / self.page_size;
        validate_exact_shape(
            "device_paged_kv_k_shape",
            self.k.dims(),
            &[
                page_count,
                self.batch,
                self.attention_heads,
                self.page_size,
                self.key_head_dim,
            ],
        )?;
        validate_exact_shape(
            "device_paged_kv_v_shape",
            self.v.dims(),
            &[
                page_count,
                self.batch,
                self.attention_heads,
                self.page_size,
                self.value_head_dim,
            ],
        )?;
        if self.k.dtype() != self.v.dtype() {
            return Err(Error::cache(format!(
                "device paged KV dtype mismatch: k={:?}, v={:?}",
                self.k.dtype(),
                self.v.dtype()
            )));
        }
        match self.k.dtype() {
            DType::F32 | DType::F16 => Ok(()),
            DType::BF16 => Err(Error::cache(
                "device paged KV BF16 is not supported by the native Metal attention path",
            )),
        }
    }
}

/// Device-resident compact KV rows selected by the DSA indexer.
///
/// Layout:
///
/// `[batch, attention_heads, selected_tokens, head_dim]`
///
/// This is intentionally separate from `DevicePagedKvView`: sparse layers do
/// not need page metadata once the SSD/block-store layer has packed exactly
/// the selected rows for the current decode token.
#[derive(Debug, Clone)]
pub struct DeviceSelectedKvView {
    pub batch: usize,
    pub attention_heads: usize,
    pub selected_tokens: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub k: DeviceValue,
    pub v: DeviceValue,
}

impl DeviceSelectedKvView {
    pub fn validate(&self) -> Result<()> {
        if self.batch == 0
            || self.attention_heads == 0
            || self.selected_tokens == 0
            || self.key_head_dim == 0
            || self.value_head_dim == 0
        {
            return Err(Error::cache(
                "device selected KV dimensions must be positive for sparse decode attention",
            ));
        }
        validate_exact_shape(
            "device_selected_kv_k_shape",
            self.k.dims(),
            &[
                self.batch,
                self.attention_heads,
                self.selected_tokens,
                self.key_head_dim,
            ],
        )?;
        validate_exact_shape(
            "device_selected_kv_v_shape",
            self.v.dims(),
            &[
                self.batch,
                self.attention_heads,
                self.selected_tokens,
                self.value_head_dim,
            ],
        )?;
        if self.k.dtype() != self.v.dtype() {
            return Err(Error::cache(format!(
                "device selected KV dtype mismatch: k={:?}, v={:?}",
                self.k.dtype(),
                self.v.dtype()
            )));
        }
        match self.k.dtype() {
            DType::F32 | DType::F16 => Ok(()),
            DType::BF16 => Err(Error::cache(
                "device selected KV BF16 is not supported by the native Metal attention path",
            )),
        }
    }
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct NamedShape {
    pub name: String,
    pub shape: Shape,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct BackendOperationReport {
    pub name: String,
    pub inputs: Vec<NamedShape>,
    pub output: Shape,
    pub checksum: f32,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct BackendCheckReport {
    pub capabilities: BackendCapabilities,
    pub operations: Vec<BackendOperationReport>,
}

pub trait Backend: Sync {
    fn capabilities(&self) -> BackendCapabilities;
    fn device(&self) -> &Device;
    fn memory_report(&self) -> BackendMemoryReport {
        BackendMemoryReport::default()
    }
    fn expert_cache_metrics(&self) -> Result<ExpertCacheMetrics> {
        Ok(ExpertCacheMetrics::default())
    }
    fn configure_expert_cache_slots_per_layer(&self, _slots_per_layer: usize) -> Result<()> {
        Ok(())
    }
    fn resize_expert_cache_slots_per_layer(&self, slots_per_layer: usize) -> Result<()> {
        self.configure_expert_cache_slots_per_layer(slots_per_layer)
    }
    fn release_prefill_resources(&self) -> Result<()> {
        Ok(())
    }
    fn configure_expert_pack(&self, _path: &Path, _header: ExpertPackHeader) -> Result<()> {
        Err(Error::backend(
            "Q2 expert packs require the native Metal backend",
        ))
    }
    fn prepare_laguna_gguf_views(
        &self,
        _mapping: MappedBytes,
        _tensor_data_offset: usize,
        _max_tensor_bytes: usize,
    ) -> Result<Option<LagunaModelViewReport>> {
        Ok(None)
    }

    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor>;
    fn linear(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor>;
    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor>;
    fn select_last_token(&self, hidden_states: &Tensor) -> Result<Tensor>;
    fn heads_to_attention_layout(&self, heads: &Tensor) -> Result<Tensor>;
    fn merge_attention_heads(&self, context_heads: &Tensor) -> Result<Tensor>;
    fn split_rope_tail(
        &self,
        heads: &Tensor,
        no_rope_dim: usize,
        rope_dim: usize,
    ) -> Result<(Tensor, Tensor)>;
    fn split_kv_mqa(
        &self,
        kv_mqa: &Tensor,
        kv_lora_rank: usize,
        rope_dim: usize,
    ) -> Result<(Tensor, Tensor)>;
    fn combine_rope_tail(&self, no_rope: &Tensor, rope: &Tensor) -> Result<Tensor>;
    fn swiglu(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor>;
    fn swiglu_f32_tensor(&self, _gate: &F32Tensor, _up: &F32Tensor) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn attention_scores(&self, q: &Tensor, k: &Tensor, head_dim: usize) -> Result<Tensor>;
    fn attention_values(&self, probs: &Tensor, values: &Tensor) -> Result<Tensor>;
    fn attention_causal_softmax(&self, scores: &Tensor, past_tokens: usize) -> Result<Tensor>;
    fn rope_slice(
        &self,
        input: &Tensor,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
    ) -> Result<Tensor>;
    fn add_f32_tensor(&self, _lhs: &F32Tensor, _rhs: &F32Tensor) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn linear_f32_tensor(
        &self,
        _input: &F32Tensor,
        _weight: &F32Tensor,
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn linear_f32_device(
        &self,
        _input: &DeviceValue,
        _weight: &F32Tensor,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }
    fn select_last_token_f32_tensor(
        &self,
        _hidden_states: &F32Tensor,
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn heads_to_attention_layout_f32_tensor(
        &self,
        _heads: &F32Tensor,
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn merge_attention_heads_f32_tensor(
        &self,
        _context_heads: &F32Tensor,
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn split_rope_tail_f32_tensor(
        &self,
        _heads: &F32Tensor,
        _no_rope_dim: usize,
        _rope_dim: usize,
    ) -> Result<Option<(F32Tensor, F32Tensor)>> {
        Ok(None)
    }
    fn split_kv_mqa_f32_tensor(
        &self,
        _kv_mqa: &F32Tensor,
        _kv_lora_rank: usize,
        _rope_dim: usize,
    ) -> Result<Option<(F32Tensor, F32Tensor)>> {
        Ok(None)
    }
    fn combine_rope_tail_f32_tensor(
        &self,
        _no_rope: &F32Tensor,
        _rope: &F32Tensor,
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn attention_scores_f32_tensor(
        &self,
        _q: &F32Tensor,
        _k: &F32Tensor,
        _head_dim: usize,
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn attention_values_f32_tensor(
        &self,
        _probs: &F32Tensor,
        _values: &F32Tensor,
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn attention_causal_softmax_f32_tensor(
        &self,
        _scores: &F32Tensor,
        _past_tokens: usize,
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn decode_attention_f32_tensor(
        &self,
        _q: &F32Tensor,
        _k: &F32Tensor,
        _v: &F32Tensor,
        _head_dim: usize,
        _past_tokens: usize,
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn paged_decode_attention_f32_tensor(
        &self,
        _q: &F32Tensor,
        _current_k: &F32Tensor,
        _current_v: &F32Tensor,
        _past_kv: &PagedKvView<'_>,
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn rope_slice_f32_tensor(
        &self,
        _input: &F32Tensor,
        _rope_dim: usize,
        _position_offset: usize,
        _theta: f32,
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn rms_norm_f32(
        &self,
        hidden_states: &F32Tensor,
        weight: &F32Tensor,
        eps: f32,
    ) -> Result<F32Tensor>;
    fn rms_norm(&self, hidden_states: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;
    fn moe_gather_tokens(&self, flat_tokens: &Tensor, token_indices: &[u32]) -> Result<Tensor>;
    fn moe_gather_tokens_f32_tensor(
        &self,
        _flat_tokens: &F32Tensor,
        _token_indices: &[u32],
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn moe_weighted_index_add_combine(
        &self,
        accumulator: &Tensor,
        token_indices: &Tensor,
        expert_outputs: &Tensor,
        expert_weights: &Tensor,
    ) -> Result<Tensor>;
    fn moe_weighted_index_add_combine_f32_tensor(
        &self,
        _accumulator: &F32Tensor,
        _token_indices: &[u32],
        _expert_outputs: &F32Tensor,
        _expert_weights: &[f32],
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn q2_k_matvec_f32_tensor(
        &self,
        _weights: &[u8],
        _input: &F32Tensor,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn q2_k_matvec_add_f32_tensor(
        &self,
        _weights: &[u8],
        _input: &F32Tensor,
        _residual: &F32Tensor,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn q2_k_matvec_f32(
        &self,
        _weights: &[u8],
        _input: &Tensor,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }
    fn q2_k_gate_up_swiglu_f32_tensor(
        &self,
        _gate_weights: &[u8],
        _up_weights: &[u8],
        _input: &F32Tensor,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn q2_k_gate_up_swiglu_f32(
        &self,
        _gate_weights: &[u8],
        _up_weights: &[u8],
        _input: &Tensor,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }
    fn q2_k_matvec_argmax_f32_tensor(
        &self,
        _weights: &[u8],
        _input: &F32Tensor,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<(u32, f32)>> {
        Ok(None)
    }
    fn q2_k_rms_norm_argmax_f32_tensor(
        &self,
        _weights: &[u8],
        _input: &F32Tensor,
        _rms_weight: &F32Tensor,
        _eps: f32,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<(u32, f32)>> {
        Ok(None)
    }
    fn q2_k_matvec_argmax_f32(
        &self,
        _weights: &[u8],
        _input: &Tensor,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<(u32, f32)>> {
        Ok(None)
    }
    fn q2_k_transposed_matvec_f32_tensor(
        &self,
        _weights: &[u8],
        _input: &F32Tensor,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn q2_k_transposed_matvec_f32(
        &self,
        _weights: &[u8],
        _input: &Tensor,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }
    fn q8_0_matvec_f32_tensor(
        &self,
        _weights: &[u8],
        _input: &F32Tensor,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }
    fn q8_0_transposed_matvec_f32_tensor(
        &self,
        _weights: &[u8],
        _input: &F32Tensor,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<F32Tensor>> {
        Ok(None)
    }

    // ------------------------------------------------------------------
    // Batched device-resident ops.
    //
    // The *_device methods mirror the native *_f32_tensor ops above but keep
    // tensors on the GPU as `DeviceValue` handles and defer execution: each
    // call encodes its kernel into a shared command buffer instead of
    // committing one command buffer per op and blocking on it. The GPU runs
    // the accumulated work when `device_submit` starts it or when the host
    // actually needs values through `device_download_f32_tensor`,
    // `device_flush`, or a fused sink such as `q2_k_matvec_argmax_device`.
    // Chaining `DeviceValue`s through these ops removes the per-kernel
    // synchronization stall from the decode hot path.
    //
    // Backends without device-resident execution keep the `Ok(None)` defaults
    // and callers fall back to the eager paths.
    // ------------------------------------------------------------------

    /// Whether this backend executes the `*_device` ops. When false, every
    /// `*_device` method returns `Ok(None)`.
    fn device_values_supported(&self) -> bool {
        false
    }

    /// Commits and waits for any batched GPU work encoded so far. A no-op
    /// when nothing is pending or the backend has no device-resident path.
    fn device_flush(&self) -> Result<()> {
        Ok(())
    }

    /// Submits pending device work without waiting for completion. Backends
    /// without asynchronous queue support preserve correctness by flushing.
    fn device_submit(&self) -> Result<()> {
        self.device_flush()
    }

    /// Ends a named GPU profiling segment without waiting for completion.
    ///
    /// Production callers should invoke this only behind a tracing check.
    /// Backends without native asynchronous execution keep it as a no-op.
    fn device_profile_boundary(&self, _label: &str) -> Result<()> {
        Ok(())
    }

    /// Copies a host tensor into GPU memory, returning a handle usable with
    /// the other `*_device` ops.
    fn device_upload_f32_tensor(&self, _tensor: &F32Tensor) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Copies a host f32 tensor into a GPU f16 buffer. Kernels that consume it
    /// must accumulate in f32 if they need full accumulation precision.
    fn device_upload_f32_tensor_as_f16(&self, _tensor: &F32Tensor) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Allocates an uninitialized GPU tensor. The caller must fill every
    /// element it will later read.
    fn device_alloc_f32_tensor(&self, _dims: &[usize]) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Allocates an uninitialized GPU f16 tensor. Shape is still expressed in
    /// logical tensor elements, not bytes.
    fn device_alloc_f16_tensor(&self, _dims: &[usize]) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Encodes a GPU-side copy between tensors with the same dtype. Offsets
    /// and length are in logical tensor elements.
    fn device_copy_same_dtype(
        &self,
        _source: &DeviceValue,
        _source_offset: usize,
        _destination: &DeviceValue,
        _destination_offset: usize,
        _len: usize,
    ) -> Result<Option<()>> {
        Ok(None)
    }

    /// Encodes a GPU-side f32 copy between two device tensors. Offsets and
    /// length are in f32 elements.
    fn device_copy_f32(
        &self,
        _source: &DeviceValue,
        _source_offset: usize,
        _destination: &DeviceValue,
        _destination_offset: usize,
        _len: usize,
    ) -> Result<Option<()>> {
        Ok(None)
    }

    /// Synchronizes pending batched work, then copies a device value back to
    /// the host. This is the only way to observe `*_device` results.
    fn device_download_f32_tensor(&self, _value: &DeviceValue) -> Result<F32Tensor> {
        Err(Error::backend(
            "device-resident values are not supported by this backend",
        ))
    }

    fn rms_norm_device(
        &self,
        _input: &DeviceValue,
        _weight: &F32Tensor,
        _eps: f32,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Applies RMSNorm to one Laguna XS decode hidden row `[1,2048]`.
    fn laguna_xs_rms_norm_device(
        &self,
        _input: &DeviceValue,
        _weight: &F32Tensor,
        _eps: f32,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Fuses Laguna XS decode RMSNorm with its following F32 router GEMV.
    fn laguna_xs_rms_norm_router_device(
        &self,
        _input: &DeviceValue,
        _norm_weight: &F32Tensor,
        _router_weight: &F32Tensor,
        _eps: f32,
    ) -> Result<Option<(DeviceValue, DeviceValue)>> {
        Ok(None)
    }

    fn prepare_bf16_matrix(
        &self,
        _bytes: &[u8],
        _rows: usize,
        _columns: usize,
    ) -> Result<Option<DeviceBf16Matrix>> {
        Ok(None)
    }

    fn bf16_linear_device(
        &self,
        _matrix: &DeviceBf16Matrix,
        _input: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Computes BF16 gate and up projections and applies SwiGLU in one native
    /// dispatch. The output shape equals the input prefix plus the projection
    /// row count.
    fn bf16_gate_up_swiglu_device(
        &self,
        _gate: &DeviceBf16Matrix,
        _up: &DeviceBf16Matrix,
        _input: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Computes Laguna Q, K, V, and per-head gate projections in one native
    /// Metal dispatch. The input remains device resident and has shape
    /// `[B,T,3072]`.
    fn laguna_attention_projections_device(
        &self,
        _query: &DeviceBf16Matrix,
        _key: &DeviceBf16Matrix,
        _value: &DeviceBf16Matrix,
        _gate: &DeviceBf16Matrix,
        _input: &DeviceValue,
    ) -> Result<Option<LagunaAttentionProjections>> {
        Ok(None)
    }

    /// Gathers BF16 embedding rows into an F32 device tensor. `token_shape`
    /// describes the logical token ID dimensions; the hidden width is appended.
    fn bf16_embedding_device(
        &self,
        _embedding: &DeviceBf16Matrix,
        _token_ids: &[u32],
        _token_shape: &[usize],
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Prepares one immutable half-split RoPE frequency table on the device.
    fn prepare_rope_table(
        &self,
        _inverse_frequency: &[f32],
        _rotary_dim: usize,
        _attention_factor: f32,
    ) -> Result<Option<DeviceRopeTable>> {
        Ok(None)
    }

    /// Applies per-head RMSNorm and half-split RoPE while keeping Q/K resident
    /// on the device. Input and output shapes are `[B,T,H,128]`.
    fn qk_rms_norm_rope_device(
        &self,
        _input: &DeviceValue,
        _norm_weight: &F32Tensor,
        _eps: f32,
        _position_offset: usize,
        _table: &DeviceRopeTable,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Applies Laguna Q/K per-head RMSNorm and RoPE in one native dispatch.
    /// Q and K retain their `[B,T,H,128]` shapes in separate device buffers.
    #[allow(clippy::too_many_arguments)]
    fn laguna_qk_rms_norm_rope_pair_device(
        &self,
        _query: &DeviceValue,
        _key: &DeviceValue,
        _query_norm_weight: &F32Tensor,
        _key_norm_weight: &F32Tensor,
        _eps: f32,
        _position_offset: usize,
        _table: &DeviceRopeTable,
    ) -> Result<Option<(DeviceValue, DeviceValue)>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn laguna_xs_qk_rms_norm_rope_pair_device(
        &self,
        _query: &DeviceValue,
        _key: &DeviceValue,
        _query_norm_weight: &F32Tensor,
        _key_norm_weight: &F32Tensor,
        _eps: f32,
        _position_offset: usize,
        _table: &DeviceRopeTable,
    ) -> Result<Option<(DeviceValue, DeviceValue)>> {
        Ok(None)
    }

    /// Allocates one exact Laguna FP8 K/V cache on Metal.
    ///
    /// Sliding caches always use the checkpoint's 512-token window. Full
    /// caches use `capacity_tokens` as the maximum supported sequence length.
    fn prepare_laguna_fp8_kv_cache(
        &self,
        _batch: usize,
        _capacity_tokens: usize,
        _retention: LagunaKvRetention,
        _key_scale: f32,
        _value_scale: f32,
    ) -> Result<Option<LagunaFp8KvCache>> {
        Ok(None)
    }

    /// Grows one full-attention Laguna FP8 K/V cache on the device while
    /// preserving every stored K/V row. Sliding caches never need growth.
    fn grow_laguna_fp8_kv_cache(
        &self,
        _cache: &mut LagunaFp8KvCache,
        _capacity_tokens: usize,
    ) -> Result<bool> {
        Ok(false)
    }

    /// Runs causal grouped-query attention directly against Laguna's FP8 KV
    /// cache and appends the current K/V rows before returning.
    ///
    /// Shapes are Q `[B,T,48|72,128]`, K/V `[B,T,8,128]`, gate
    /// `[B,T,48|72]`, and output `[B,T,48|72,128]`. The native path encodes
    /// attention and cache append into the existing open Metal batch and does
    /// not synchronize the host.
    fn laguna_gated_gqa_attention_device(
        &self,
        _query: &DeviceValue,
        _current_key: &DeviceValue,
        _current_value: &DeviceValue,
        _gate: &DeviceValue,
        _cache: &mut LagunaFp8KvCache,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn prepare_laguna_f16_kv_cache(
        &self,
        _batch: usize,
        _capacity_tokens: usize,
        _retention: LagunaKvRetention,
    ) -> Result<Option<LagunaF16KvCache>> {
        Ok(None)
    }

    fn grow_laguna_f16_kv_cache(
        &self,
        _cache: &mut LagunaF16KvCache,
        _capacity_tokens: usize,
    ) -> Result<bool> {
        Ok(false)
    }

    fn laguna_gated_gqa_f16_attention_device(
        &self,
        _query: &DeviceValue,
        _current_key: &DeviceValue,
        _current_value: &DeviceValue,
        _gate: &DeviceValue,
        _cache: &mut LagunaF16KvCache,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Copies one immutable packed INT4 matrix and its BF16 group scales into
    /// persistent device buffers. The returned object is reused by subsequent
    /// projections; inference must not upload the same weight for every token.
    fn prepare_w4_groupwise_weight(
        &self,
        _packed: &[u8],
        _scales: &[u8],
        _in_features: usize,
        _out_features: usize,
        _group_size: usize,
    ) -> Result<Option<DeviceW4Weight>> {
        Ok(None)
    }

    /// Creates a W4 weight view without copying immutable model-owned bytes.
    fn prepare_w4_groupwise_weight_no_copy(
        &self,
        _source: Arc<dyn W4WeightSource>,
        _in_features: usize,
        _out_features: usize,
        _group_size: usize,
    ) -> Result<Option<DeviceW4Weight>> {
        Ok(None)
    }

    fn w4_groupwise_matvec_device(
        &self,
        _weight: &DeviceW4Weight,
        _input: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn w4_groupwise_gate_up_swiglu_device(
        &self,
        _gate: &DeviceW4Weight,
        _up: &DeviceW4Weight,
        _input: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Executes every currently ready routed INT4 expert as one Metal wave.
    ///
    /// Input is `[tokens, hidden]`; destination is
    /// `[tokens * top_k, hidden]`. The native implementation folds token gather
    /// and assignment scatter into two dispatches: gate/up/SwiGLU, then down.
    #[allow(clippy::too_many_arguments)]
    fn w4_groupwise_expert_wave_device(
        &self,
        _groups: &[W4ExpertGroup<'_>],
        _input: &DeviceValue,
        _token_count: usize,
        _top_k: usize,
        _destination: &DeviceValue,
    ) -> Result<Option<()>> {
        Ok(None)
    }

    fn q2_k_matvec_device(
        &self,
        _weights: &[u8],
        _input: &DeviceValue,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn q2_k_transposed_matvec_device(
        &self,
        _weights: &[u8],
        _input: &DeviceValue,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn q8_0_matvec_device(
        &self,
        _weights: &[u8],
        _input: &DeviceValue,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn gguf_k_matvec_device(
        &self,
        _quant: GgufKQuant,
        _weights: &[u8],
        _input: &DeviceValue,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn prepare_laguna_xs_mps_prefill_weight(
        &self,
        _quant: GgufKQuant,
        _weights: &[u8],
        _in_features: usize,
        _out_features: usize,
    ) -> Result<bool> {
        Ok(false)
    }

    #[allow(clippy::too_many_arguments)]
    fn laguna_xs_k_matvec_add_device(
        &self,
        _quant: GgufKQuant,
        _weights: &[u8],
        _input: &DeviceValue,
        _residual: &DeviceValue,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn laguna_xs_k_matvec_add2_device(
        &self,
        _quant: GgufKQuant,
        _weights: &[u8],
        _input: &DeviceValue,
        _residual_a: &DeviceValue,
        _residual_b: &DeviceValue,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn q4_k_embedding_device(
        &self,
        _weights: &[u8],
        _token_ids: &[u32],
        _token_shape: &[usize],
        _vocab_size: usize,
        _hidden_size: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn q8_0_embedding_device(
        &self,
        _weights: &[u8],
        _token_ids: &[u32],
        _token_shape: &[usize],
        _vocab_size: usize,
        _hidden_size: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn q8_0_matvec_pair_device(
        &self,
        _weights_a: &[u8],
        _weights_b: &[u8],
        _input: &DeviceValue,
        _row_count: usize,
        _in_features: usize,
        _out_features_a: usize,
        _out_features_b: usize,
    ) -> Result<Option<(DeviceValue, DeviceValue)>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn laguna_q8_0_attention_projections_device(
        &self,
        _query_weights: &[u8],
        _key_weights: &[u8],
        _value_weights: &[u8],
        _gate_weights: &[u8],
        _input: &DeviceValue,
        _row_count: usize,
        _in_features: usize,
        _query_features: usize,
        _key_features: usize,
        _value_features: usize,
        _gate_features: usize,
    ) -> Result<Option<[DeviceValue; 4]>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn laguna_xs_attention_projections_device(
        &self,
        _query_weights: &[u8],
        _key_weights: &[u8],
        _value_weights: &[u8],
        _value_quant: GgufKQuant,
        _gate_weights: &[u8],
        _input: &DeviceValue,
        _row_count: usize,
        _in_features: usize,
        _query_features: usize,
        _key_features: usize,
        _value_features: usize,
        _gate_features: usize,
    ) -> Result<Option<[DeviceValue; 4]>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn q8_0_gate_up_swiglu_device(
        &self,
        _gate_weights: &[u8],
        _up_weights: &[u8],
        _input: &DeviceValue,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn laguna_xs_q4_gate_up_swiglu_device(
        &self,
        _gate_weights: &[u8],
        _up_weights: &[u8],
        _input: &DeviceValue,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn laguna_xs_router_topk_device(
        &self,
        _router_logits: &DeviceValue,
        _correction_bias: &[f32],
        _top_k: usize,
        _norm_topk_prob: bool,
        _routed_scaling_factor: f32,
    ) -> Result<Option<DeviceRouterTopK>> {
        Ok(None)
    }

    fn q8_0_transposed_matvec_device(
        &self,
        _weights: &[u8],
        _input: &DeviceValue,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn q2_k_packed_heads_transposed_matvec_device(
        &self,
        _weights: &[u8],
        _input: &DeviceValue,
        _row_count: usize,
        _head_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn q8_0_packed_heads_transposed_matvec_device(
        &self,
        _weights: &[u8],
        _input: &DeviceValue,
        _row_count: usize,
        _head_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Q2_K matvec fused with a residual add; the output takes the residual's
    /// shape.
    fn q2_k_matvec_add_device(
        &self,
        _weights: &[u8],
        _input: &DeviceValue,
        _residual: &DeviceValue,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Q8_0 matvec fused with a residual add; the output takes the residual's
    /// shape.
    fn q8_0_matvec_add_device(
        &self,
        _weights: &[u8],
        _input: &DeviceValue,
        _residual: &DeviceValue,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Laguna Q8_0 matvec fused with two residual adds. This is intentionally
    /// model-specific because it binds weights through the Laguna GGUF view.
    #[allow(clippy::too_many_arguments)]
    fn laguna_q8_0_matvec_add2_device(
        &self,
        _weights: &[u8],
        _input: &DeviceValue,
        _residual_a: &DeviceValue,
        _residual_b: &DeviceValue,
        _row_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn q2_k_multi_expert_gate_up_swiglu_device(
        &self,
        _gate_weights: &[u8],
        _up_weights: &[u8],
        _input: &DeviceValue,
        _token_indices: &[u32],
        _expert_ids: &[u32],
        _token_count: usize,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn q2_k_multi_expert_matvec_device(
        &self,
        _weights: &[u8],
        _input: &DeviceValue,
        _expert_ids: &[u32],
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn laguna_gguf_moe_device(
        &self,
        _gate_weights: &[u8],
        _up_weights: &[u8],
        _down_weights: &[u8],
        _quant: GgufExpertQuant,
        _input: &DeviceValue,
        _routing: &DeviceRouterTopK,
        _in_features: usize,
        _intermediate_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn laguna_xs_gguf_moe_device(
        &self,
        _gate_weights: &[u8],
        _up_weights: &[u8],
        _down_weights: &[u8],
        _down_quant: GgufKQuant,
        _input: &DeviceValue,
        _routing: &DeviceRouterTopK,
        _in_features: usize,
        _intermediate_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Streams selected Q2 experts through a per-layer cache and executes
    /// ready groups without waiting for every SSD miss to complete.
    #[allow(clippy::too_many_arguments)]
    fn ready_routed_experts_device(
        &self,
        _layer_index: usize,
        _model_path: &Path,
        _gate_payloads: &[Q2ExpertSource<'_>],
        _up_payloads: &[Q2ExpertSource<'_>],
        _down_payloads: &[Q2ExpertSource<'_>],
        _input: &DeviceValue,
        _routing: &DeviceRouterTopK,
        _in_features: usize,
        _intermediate_features: usize,
        _out_features: usize,
    ) -> Result<Option<DeviceRoutedExperts>> {
        Ok(None)
    }

    /// Loads predicted Q2 experts into the resident cache before exact routing.
    /// Implementations must keep predictive reads separate from demand misses
    /// in their observability counters.
    fn prefetch_routed_experts_device(
        &self,
        _layer_index: usize,
        _model_path: &Path,
        _gate_payloads: &[Q2ExpertSource<'_>],
        _up_payloads: &[Q2ExpertSource<'_>],
        _down_payloads: &[Q2ExpertSource<'_>],
    ) -> Result<()> {
        Ok(())
    }

    /// Inserts a device-side dependency before consuming routed-expert rows.
    /// The default backend has no asynchronous routed-expert path.
    fn wait_for_routed_experts_device(&self, _routed: &DeviceRoutedExperts) -> Result<()> {
        Ok(())
    }

    /// Output-head matvec + greedy argmax over a device-resident hidden state.
    /// Flushes the batch (this is the end-of-token sink) and returns the
    /// winning token id and score.
    fn q2_k_matvec_argmax_device(
        &self,
        _weights: &[u8],
        _input: &DeviceValue,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<(u32, f32)>> {
        Ok(None)
    }

    /// Laguna Q8_0 output projection plus greedy argmax. The native path keeps
    /// only one candidate per four vocabulary rows instead of full logits.
    fn laguna_q8_0_matvec_argmax_device(
        &self,
        _weights: &[u8],
        _input: &DeviceValue,
        _in_features: usize,
        _out_features: usize,
    ) -> Result<Option<(u32, f32)>> {
        Ok(None)
    }

    /// Greedy argmax over a device-resident f32 score vector. Flushes the
    /// batch and returns the winning token id and score.
    fn argmax_f32_device(&self, _scores: &DeviceValue) -> Result<Option<(u32, f32)>> {
        Ok(None)
    }

    /// Greedy argmax independently over each row of `[rows, row_width]`.
    /// The returned vectors contain one token id and score per row.
    fn argmax_rows_f32_device(
        &self,
        _scores: &DeviceValue,
        _row_width: usize,
    ) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
        Ok(None)
    }

    fn rope_slice_device(
        &self,
        _input: &DeviceValue,
        _rope_dim: usize,
        _position_offset: usize,
        _theta: f32,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn split_rope_tail_device(
        &self,
        _heads: &DeviceValue,
        _no_rope_dim: usize,
        _rope_dim: usize,
    ) -> Result<Option<(DeviceValue, DeviceValue)>> {
        Ok(None)
    }

    fn split_kv_mqa_device(
        &self,
        _kv_mqa: &DeviceValue,
        _kv_lora_rank: usize,
        _rope_dim: usize,
    ) -> Result<Option<(DeviceValue, DeviceValue)>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn mla_kv_postprocess_device(
        &self,
        _kv_mqa: &DeviceValue,
        _norm_weight: &F32Tensor,
        _norm_eps: f32,
        _kv_lora_rank: usize,
        _rope_dim: usize,
        _position_offset: usize,
        _theta: f32,
    ) -> Result<Option<(DeviceValue, DeviceValue)>> {
        Ok(None)
    }

    fn combine_rope_tail_device(
        &self,
        _no_rope: &DeviceValue,
        _rope: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn heads_to_attention_layout_device(
        &self,
        _heads: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn merge_attention_heads_device(
        &self,
        _context_heads: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Stack per-head projection outputs from `[head][row, head_dim]` into
    /// one token-major device tensor `[row, head, head_dim]`.
    fn stack_head_outputs_device(
        &self,
        _head_outputs: &[DeviceValue],
        _row_count: usize,
        _head_dim: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn select_last_token_device(
        &self,
        _hidden_states: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn add_device(&self, _lhs: &DeviceValue, _rhs: &DeviceValue) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn swiglu_device(&self, _gate: &DeviceValue, _up: &DeviceValue) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn paged_decode_attention_device(
        &self,
        _q: &DeviceValue,
        _current_k: &DeviceValue,
        _current_v: &DeviceValue,
        _past_kv: &PagedKvView<'_>,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn paged_decode_attention_resident_device(
        &self,
        _q: &DeviceValue,
        _current_k: &DeviceValue,
        _current_v: &DeviceValue,
        _past_kv: &DevicePagedKvView,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// GLM MLA attention without materializing historical per-head K/V.
    ///
    /// Inputs are a short causal query sequence, its normalized latent/RoPE
    /// rows, and the normalized latent/RoPE paged cache. The result is
    /// `[B,T,H,V]`; callers convert it to attention layout `[B,H,T,V]`.
    #[allow(clippy::too_many_arguments)]
    fn q8_0_absorbed_mla_device(
        &self,
        _k_b_weights: &[u8],
        _v_b_weights: &[u8],
        _q_no_rope: &DeviceValue,
        _q_rope: &DeviceValue,
        _current_latent: &DeviceValue,
        _current_rope: &DeviceValue,
        _past_kv: &DevicePagedKvView,
        _qk_head_dim: usize,
        _value_dim: usize,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Sparse decode attention over a compact selected KV set plus the
    /// current token. This is the DSA/indexer hot primitive for GLM sparse
    /// layers: callers pass only the K/V rows chosen by the indexer, so the
    /// kernel does not scan the full context.
    fn selected_decode_attention_device(
        &self,
        _q: &DeviceValue,
        _selected_k: &DeviceValue,
        _selected_v: &DeviceValue,
        _current_k: &DeviceValue,
        _current_v: &DeviceValue,
        _include_current_kv: bool,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Causal attention for a short speculative token sequence over an
    /// already-selected or full contiguous past KV set.
    fn selected_sequence_attention_device(
        &self,
        _q: &DeviceValue,
        _past_k: &DeviceValue,
        _past_v: &DeviceValue,
        _current_k: &DeviceValue,
        _current_v: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Causal attention for the first prompt chunk, where no past KV exists.
    /// Query row `t` can attend to current K/V rows `0..=t`.
    fn causal_sequence_attention_device(
        &self,
        _q: &DeviceValue,
        _current_k: &DeviceValue,
        _current_v: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Linearizes resident paged KV into contiguous `[B,H,T,D]` device tensors.
    /// Dense MLA uses this to expand latent cache on GPU without CPU
    /// reconstruction.
    fn paged_kv_contiguous_device(
        &self,
        _past_kv: &DevicePagedKvView,
    ) -> Result<Option<DeviceSelectedKvView>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn q8_row_selected_kv_device(
        &self,
        _key_payload: &[u8],
        _value_payload: &[u8],
        _batch: usize,
        _attention_heads: usize,
        _selected_tokens: usize,
        _key_head_dim: usize,
        _value_head_dim: usize,
    ) -> Result<Option<DeviceSelectedKvView>> {
        Ok(None)
    }

    fn dsa_index_key_device(
        &self,
        _raw_key: &DeviceValue,
        _weight: &F32Tensor,
        _bias: &F32Tensor,
        _rope_dim: usize,
        _position_offset: usize,
        _theta: f32,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn dsa_decode_topk_device(
        &self,
        _hidden_states: &DeviceValue,
        _q_raw: &DeviceValue,
        _past_index_keys: &DeviceValue,
        _current_index_key: &DeviceValue,
        _weights_proj: &F32Tensor,
        _heads: usize,
        _head_dim: usize,
        _rope_dim: usize,
        _position_offset: usize,
        _theta: f32,
        _top_k: usize,
    ) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }

    /// Stacks equal-length device rows into one `[rows.len(), row_len]` value
    /// with GPU-side copies. Used to assemble per-expert outputs for the MoE
    /// combine without a host round-trip.
    fn moe_stack_rows_device(&self, _rows: &[DeviceValue]) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Gathers selected rows from `[tokens, hidden]` into
    /// `[indices.len(), hidden]` without downloading the source tensor.
    fn moe_gather_rows_device(
        &self,
        _input: &DeviceValue,
        _token_indices: &[u32],
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    /// Writes source rows into distinct rows of a preallocated destination.
    /// This is used to restore expert-grouped work to token-major assignment
    /// order before the fused top-k combine.
    fn moe_scatter_rows_device(
        &self,
        _rows: &DeviceValue,
        _destination_rows: &[u32],
        _destination: &DeviceValue,
    ) -> Result<Option<()>> {
        Ok(None)
    }

    fn moe_weighted_index_add_combine_device(
        &self,
        _accumulator: &DeviceValue,
        _token_indices: &[u32],
        _expert_outputs: &DeviceValue,
        _expert_weights: &[f32],
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn moe_topk_combine_residual_device(
        &self,
        _shared: &DeviceValue,
        _residual: &DeviceValue,
        _expert_outputs: &DeviceValue,
        _routing: &DeviceRouterTopK,
    ) -> Result<Option<DeviceValue>> {
        Ok(None)
    }

    fn moe_router_topk_device(
        &self,
        _router_logits: &DeviceValue,
        _correction_bias: &[f32],
        _top_k: usize,
        _norm_topk_prob: bool,
        _routed_scaling_factor: f32,
    ) -> Result<Option<RouterTopK>> {
        Ok(None)
    }

    fn moe_router_topk_resident_device(
        &self,
        _router_logits: &DeviceValue,
        _correction_bias: &[f32],
        _top_k: usize,
        _norm_topk_prob: bool,
        _routed_scaling_factor: f32,
    ) -> Result<Option<DeviceRouterTopK>> {
        Ok(None)
    }

    /// Completes pending work and reads only the selected expert IDs from a
    /// resident router result. Inferno uses this single sparse-layer
    /// synchronization point to prefetch exact Q2 expert ranges from SSD
    /// before their Metal kernels start touching mmap pages.
    fn moe_router_expert_ids_device(
        &self,
        _routing: &DeviceRouterTopK,
    ) -> Result<Option<Vec<u32>>> {
        Ok(None)
    }
}

#[derive(Clone)]
pub struct MetalBackend {
    device: Device,
    device_kind: DeviceKind,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    native_metal: Option<Arc<Metal>>,
}

impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("MetalBackend");
        debug
            .field("device", &self.device)
            .field("device_kind", &self.device_kind);
        #[cfg(all(target_os = "macos", feature = "metal"))]
        debug.field("native_metal", &self.native_metal.is_some());
        debug.finish()
    }
}

impl MetalBackend {
    pub fn new() -> Result<Self> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let native_metal = Metal::new()?;
            return Ok(Self {
                device: Device::Cpu,
                device_kind: DeviceKind::Metal,
                native_metal: Some(Arc::new(native_metal)),
            });
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            Err(Error::backend(
                "Inferno Q2 production backend requires Apple Metal and native Metal kernels",
            ))
        }
    }

    pub fn reference() -> Result<Self> {
        Ok(Self {
            device: Device::Cpu,
            device_kind: DeviceKind::Cpu,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            native_metal: None,
        })
    }

    pub fn device_report(&self) -> DeviceReport {
        if self.has_native_metal() {
            DeviceReport::metal()
        } else {
            DeviceReport::reference(self.device_kind)
        }
    }

    pub fn device_debug(&self) -> String {
        format!("{:?}", self.device)
    }

    pub fn from_device(device: Device) -> Result<Self> {
        let device_kind = device_kind(&device);
        Ok(Self {
            device,
            device_kind,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            native_metal: native_metal_for(device_kind),
        })
    }

    #[cfg(test)]
    fn operation_report(
        name: &str,
        inputs: Vec<NamedShape>,
        output: &Tensor,
    ) -> Result<BackendOperationReport> {
        Ok(BackendOperationReport {
            name: name.to_string(),
            inputs,
            output: Shape::new(output.dims().to_vec()),
            checksum: tensor_checksum(output)?,
        })
    }
}

impl Backend for MetalBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            backend: if self.has_native_metal() {
                BackendKind::Metal
            } else {
                BackendKind::Reference
            },
            device: self.device_kind,
            custom_kernels: self.has_native_metal(),
            supports_f32: true,
            supports_f16: self.device_kind == DeviceKind::Metal,
            supports_bf16: self.device_kind == DeviceKind::Metal,
            operations: backend_operations(self.has_native_metal()),
        }
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn memory_report(&self) -> BackendMemoryReport {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                return native_metal.memory_report();
            }
        }

        BackendMemoryReport::default()
    }

    fn expert_cache_metrics(&self) -> Result<ExpertCacheMetrics> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                return native_metal.expert_cache_metrics();
            }
        }

        Ok(ExpertCacheMetrics::default())
    }

    fn configure_expert_cache_slots_per_layer(&self, slots_per_layer: usize) -> Result<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                return native_metal.configure_expert_cache_slots_per_layer(slots_per_layer);
            }
        }

        let _ = slots_per_layer;
        Ok(())
    }

    fn resize_expert_cache_slots_per_layer(&self, slots_per_layer: usize) -> Result<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                return native_metal.resize_expert_cache_slots_per_layer(slots_per_layer);
            }
        }

        let _ = slots_per_layer;
        Ok(())
    }

    fn release_prefill_resources(&self) -> Result<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                return native_metal.release_prefill_resources();
            }
        }

        Ok(())
    }

    fn configure_expert_pack(&self, path: &Path, header: ExpertPackHeader) -> Result<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                return native_metal.configure_expert_pack(path, header);
            }
        }

        let _ = (path, header);
        Err(Error::backend(
            "Q2 expert packs require the native Metal backend",
        ))
    }

    fn prepare_laguna_gguf_views(
        &self,
        mapping: MappedBytes,
        tensor_data_offset: usize,
        max_tensor_bytes: usize,
    ) -> Result<Option<LagunaModelViewReport>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            return native_metal
                .prepare_laguna_gguf_views(mapping, tensor_data_offset, max_tensor_bytes)
                .map(Some);
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (mapping, tensor_data_offset, max_tensor_bytes);
            Ok(None)
        }
    }

    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        validate_matmul_shapes(lhs, rhs)?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let lhs_dims = lhs.dims();
                let rhs_dims = rhs.dims();
                let rows = lhs_dims[0];
                let inner = lhs_dims[1];
                let cols = rhs_dims[1];
                let lhs_values = lhs
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let rhs_values = rhs
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let output_values =
                    native_metal.matmul_f32(&lhs_values, &rhs_values, rows, inner, cols)?;
                return Tensor::from_vec(output_values, (rows, cols), self.device())
                    .map_err(Into::into);
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal F32 matmul is required on Metal",
                ));
            }
        }

        reference_matmul(lhs, rhs, self.device())
    }

    fn linear(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        validate_linear_shapes(input, weight)?;

        let (rows, mut output_shape, input_values) = flatten_linear_input(input)?;
        let in_features = *input
            .dims()
            .last()
            .ok_or_else(|| Error::backend("linear input has empty shape"))?;
        let out_features = weight.dims()[0];
        *output_shape
            .last_mut()
            .ok_or_else(|| Error::backend("linear output shape is empty"))? = out_features;
        let weight_values = weight
            .to_dtype(common::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output_values = native_metal.linear_f32(
                    &input_values,
                    &weight_values,
                    rows,
                    in_features,
                    out_features,
                )?;
                return tensor_from_matvec_output(output_values, &output_shape, self.device());
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal F32 linear is required on Metal",
                ));
            }
        }

        reference_linear_from_values(
            &input_values,
            &weight_values,
            rows,
            in_features,
            out_features,
            &output_shape,
            self.device(),
        )
    }

    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        validate_add_shapes(lhs, rhs)?;
        let output_shape = lhs.dims().to_vec();
        let lhs_values = lhs
            .to_dtype(common::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let rhs_values = rhs
            .to_dtype(common::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output_values = native_metal.add_f32(&lhs_values, &rhs_values)?;
                return tensor_from_native_values(output_values, &output_shape, self.device());
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend("native Metal add is required on Metal"));
            }
        }

        reference_add_from_values(&lhs_values, &rhs_values, &output_shape, self.device())
    }

    fn select_last_token(&self, hidden_states: &Tensor) -> Result<Tensor> {
        validate_select_last_token_shapes(hidden_states)?;
        let dims = hidden_states.dims();
        let batch = dims[0];
        let tokens = dims[1];
        let hidden_size = dims[2];

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let values = hidden_states
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let output_values =
                    native_metal.select_last_token_f32(&values, batch, tokens, hidden_size)?;
                return Tensor::from_vec(output_values, (batch, 1, hidden_size), self.device())
                    .map_err(Into::into);
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal select_last_token is required on Metal",
                ));
            }
        }

        reference_select_last_token(hidden_states, self.device())
    }

    fn heads_to_attention_layout(&self, heads: &Tensor) -> Result<Tensor> {
        validate_heads_to_attention_layout_shapes(heads)?;
        let dims = heads.dims();
        let batch = dims[0];
        let tokens = dims[1];
        let head_count = dims[2];
        let head_dim = dims[3];

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let values = heads
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let output_values = native_metal
                    .heads_to_attention_layout_f32(&values, batch, tokens, head_count, head_dim)?;
                return Tensor::from_vec(
                    output_values,
                    (batch, head_count, tokens, head_dim),
                    self.device(),
                )
                .map_err(Into::into);
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal heads_to_attention_layout is required on Metal",
                ));
            }
        }

        reference_heads_to_attention_layout(heads, self.device())
    }

    fn merge_attention_heads(&self, context_heads: &Tensor) -> Result<Tensor> {
        validate_merge_attention_heads_shapes(context_heads)?;
        let dims = context_heads.dims();
        let batch = dims[0];
        let head_count = dims[1];
        let tokens = dims[2];
        let head_dim = dims[3];
        let merged_width = head_count
            .checked_mul(head_dim)
            .ok_or_else(|| Error::backend("merge_attention_heads merged width overflow"))?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let values = context_heads
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let output_values = native_metal
                    .merge_attention_heads_f32(&values, batch, head_count, tokens, head_dim)?;
                return Tensor::from_vec(
                    output_values,
                    (batch, tokens, merged_width),
                    self.device(),
                )
                .map_err(Into::into);
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal merge_attention_heads is required on Metal",
                ));
            }
        }

        reference_merge_attention_heads(context_heads, self.device())
    }

    fn split_rope_tail(
        &self,
        heads: &Tensor,
        no_rope_dim: usize,
        rope_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        validate_split_rope_tail_shapes(heads, no_rope_dim, rope_dim)?;
        let dims = heads.dims();
        let batch = dims[0];
        let tokens = dims[1];
        let head_count = dims[2];

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let values = heads
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let (no_rope_values, rope_values) = native_metal.split_rope_tail_f32(
                    &values,
                    batch,
                    tokens,
                    head_count,
                    no_rope_dim,
                    rope_dim,
                )?;
                let no_rope = Tensor::from_vec(
                    no_rope_values,
                    (batch, tokens, head_count, no_rope_dim),
                    self.device(),
                )?;
                let rope = Tensor::from_vec(
                    rope_values,
                    (batch, tokens, head_count, rope_dim),
                    self.device(),
                )?;
                return Ok((no_rope, rope));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal split_rope_tail is required on Metal",
                ));
            }
        }

        reference_split_rope_tail(heads, no_rope_dim, rope_dim, self.device())
    }

    fn split_kv_mqa(
        &self,
        kv_mqa: &Tensor,
        kv_lora_rank: usize,
        rope_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        validate_split_kv_mqa_shapes(kv_mqa, kv_lora_rank, rope_dim)?;
        let dims = kv_mqa.dims();
        let batch = dims[0];
        let tokens = dims[1];

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let values = kv_mqa
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let (kv_latent_values, k_rope_values) = native_metal.split_kv_mqa_f32(
                    &values,
                    batch,
                    tokens,
                    kv_lora_rank,
                    rope_dim,
                )?;
                let kv_latent = Tensor::from_vec(
                    kv_latent_values,
                    (batch, tokens, kv_lora_rank),
                    self.device(),
                )?;
                let k_rope =
                    Tensor::from_vec(k_rope_values, (batch, tokens, 1, rope_dim), self.device())?;
                return Ok((kv_latent, k_rope));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal split_kv_mqa is required on Metal",
                ));
            }
        }

        reference_split_kv_mqa(kv_mqa, kv_lora_rank, rope_dim, self.device())
    }

    fn combine_rope_tail(&self, no_rope: &Tensor, rope: &Tensor) -> Result<Tensor> {
        validate_combine_rope_tail_shapes(no_rope, rope)?;
        let no_rope_dims = no_rope.dims();
        let rope_dims = rope.dims();
        let batch = no_rope_dims[0];
        let tokens = no_rope_dims[1];
        let head_count = no_rope_dims[2];
        let no_rope_dim = no_rope_dims[3];
        let rope_head_count = rope_dims[2];
        let rope_dim = rope_dims[3];
        let total_dim = no_rope_dim
            .checked_add(rope_dim)
            .ok_or_else(|| Error::backend("combine_rope_tail total dim overflow"))?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let no_rope_values = no_rope
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let rope_values = rope
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let output_values = native_metal.combine_rope_tail_f32(
                    &no_rope_values,
                    &rope_values,
                    batch,
                    tokens,
                    head_count,
                    rope_head_count,
                    no_rope_dim,
                    rope_dim,
                )?;
                return Tensor::from_vec(
                    output_values,
                    (batch, tokens, head_count, total_dim),
                    self.device(),
                )
                .map_err(Into::into);
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal combine_rope_tail is required on Metal",
                ));
            }
        }

        reference_combine_rope_tail(no_rope, rope, self.device())
    }

    fn swiglu(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        validate_swiglu_shapes(gate, up)?;
        let output_shape = gate.dims().to_vec();
        let gate_values = gate
            .to_dtype(common::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let up_values = up
            .to_dtype(common::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output_values = native_metal.swiglu_f32(&gate_values, &up_values)?;
                return tensor_from_native_values(output_values, &output_shape, self.device());
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend("native Metal SwiGLU is required on Metal"));
            }
        }

        reference_swiglu_from_values(&gate_values, &up_values, &output_shape, self.device())
    }

    fn swiglu_f32_tensor(&self, gate: &F32Tensor, up: &F32Tensor) -> Result<Option<F32Tensor>> {
        validate_exact_shape("native_swiglu_shape", gate.dims(), up.dims())?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output = native_metal.swiglu_f32(gate.values(), up.values())?;
                return Ok(Some(F32Tensor::new(output, gate.dims().to_vec())?));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend("native Metal SwiGLU is required on Metal"));
            }
        }

        Ok(None)
    }

    fn attention_scores(&self, q: &Tensor, k: &Tensor, head_dim: usize) -> Result<Tensor> {
        validate_attention_shapes(q, k, head_dim)?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let q_dims = q.dims();
                let k_dims = k.dims();
                let batch = q_dims[0];
                let heads = q_dims[1];
                let query_tokens = q_dims[2];
                let key_tokens = k_dims[2];
                let q_values = q
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let k_values = k
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let output_values = native_metal.attention_scores_f32(
                    &q_values,
                    &k_values,
                    batch,
                    heads,
                    query_tokens,
                    key_tokens,
                    head_dim,
                )?;
                return Tensor::from_vec(
                    output_values,
                    (batch, heads, query_tokens, key_tokens),
                    self.device(),
                )
                .map_err(Into::into);
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal attention scores are required on Metal",
                ));
            }
        }

        reference_attention_scores(q, k, head_dim, self.device())
    }

    fn attention_values(&self, probs: &Tensor, values: &Tensor) -> Result<Tensor> {
        validate_attention_value_shapes(probs, values)?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let probs_dims = probs.dims();
                let values_dims = values.dims();
                let batch = probs_dims[0];
                let heads = probs_dims[1];
                let query_tokens = probs_dims[2];
                let key_tokens = probs_dims[3];
                let value_dim = values_dims[3];
                let probs_values = probs
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let value_values = values
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let output_values = native_metal.attention_values_f32(
                    &probs_values,
                    &value_values,
                    batch,
                    heads,
                    query_tokens,
                    key_tokens,
                    value_dim,
                )?;
                return Tensor::from_vec(
                    output_values,
                    (batch, heads, query_tokens, value_dim),
                    self.device(),
                )
                .map_err(Into::into);
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal attention value aggregation is required on Metal",
                ));
            }
        }

        reference_attention_values(probs, values, self.device())
    }

    fn attention_causal_softmax(&self, scores: &Tensor, past_tokens: usize) -> Result<Tensor> {
        validate_attention_causal_softmax_shapes(scores, past_tokens)?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let dims = scores.dims();
                let batch = dims[0];
                let heads = dims[1];
                let query_tokens = dims[2];
                let key_tokens = dims[3];
                let input_values = scores
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let output_values = native_metal.attention_causal_softmax_f32(
                    &input_values,
                    batch,
                    heads,
                    query_tokens,
                    key_tokens,
                    past_tokens,
                )?;
                return Tensor::from_vec(
                    output_values,
                    (batch, heads, query_tokens, key_tokens),
                    self.device(),
                )
                .map_err(Into::into);
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal attention causal softmax is required on Metal",
                ));
            }
        }

        reference_attention_causal_softmax(scores, past_tokens, self.device())
    }

    fn rope_slice(
        &self,
        input: &Tensor,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
    ) -> Result<Tensor> {
        validate_rope_slice_shapes(input, rope_dim, position_offset, theta)?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let dims = input.dims();
                let batch = dims[0];
                let tokens = dims[1];
                let heads = dims[2];
                let input_values = input
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let output_values = native_metal.rope_slice_f32(
                    &input_values,
                    batch,
                    tokens,
                    heads,
                    rope_dim,
                    position_offset,
                    theta,
                )?;
                return Tensor::from_vec(
                    output_values,
                    (batch, tokens, heads, rope_dim),
                    self.device(),
                )
                .map_err(Into::into);
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend("native Metal RoPE is required on Metal"));
            }
        }

        reference_rope_slice(input, rope_dim, position_offset, theta, self.device())
    }

    fn add_f32_tensor(&self, lhs: &F32Tensor, rhs: &F32Tensor) -> Result<Option<F32Tensor>> {
        validate_exact_shape("native_add_f32_shape", lhs.dims(), rhs.dims())?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output = native_metal.add_f32(lhs.values(), rhs.values())?;
                return Ok(Some(F32Tensor::new(output, lhs.dims().to_vec())?));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend("native Metal add is required on Metal"));
            }
        }

        Ok(None)
    }

    fn linear_f32_tensor(
        &self,
        input: &F32Tensor,
        weight: &F32Tensor,
    ) -> Result<Option<F32Tensor>> {
        let input_dims = require_f32_rank("native_linear_input", input, 2)?;
        let weight_dims = require_f32_rank("native_linear_weight", weight, 2)?;
        let rows = input_dims[0];
        let in_features = input_dims[1];
        let out_features = weight_dims[0];
        validate_exact_shape(
            "native_linear_input_weight_features",
            &[weight_dims[1]],
            &[in_features],
        )?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output = native_metal.linear_f32(
                    input.values(),
                    weight.values(),
                    rows,
                    in_features,
                    out_features,
                )?;
                return Ok(Some(F32Tensor::new(output, [rows, out_features])?));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend("native Metal linear is required on Metal"));
            }
        }

        Ok(None)
    }

    fn linear_f32_device(
        &self,
        input: &DeviceValue,
        weight: &F32Tensor,
    ) -> Result<Option<DeviceValue>> {
        let input_dims = input.dims();
        if input_dims.len() != 2 {
            return Err(Error::backend(format!(
                "native_linear_device input must be rank 2 [rows,in_features], got {input_dims:?}"
            )));
        }
        if input.dtype() != DType::F32 {
            return Err(Error::backend(format!(
                "native_linear_device input must be F32, got {:?}",
                input.dtype()
            )));
        }
        let weight_dims = require_f32_rank("native_linear_device_weight", weight, 2)?;
        let rows = input_dims[0];
        let in_features = input_dims[1];
        let out_features = weight_dims[0];
        validate_exact_shape(
            "native_linear_device_input_weight_features",
            &[weight_dims[1]],
            &[in_features],
        )?;
        let input_len = input.element_count()?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let buffer = native_metal.batched_linear_f32(
                &input.buffer,
                input_len,
                weight.values(),
                rows,
                in_features,
                out_features,
            )?;
            return Ok(Some(DeviceValue::new(vec![rows, out_features], buffer)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (input_len, rows, in_features, out_features);
            Ok(None)
        }
    }

    fn select_last_token_f32_tensor(&self, hidden_states: &F32Tensor) -> Result<Option<F32Tensor>> {
        let dims = require_f32_rank("native_select_last_token", hidden_states, 3)?;
        let batch = dims[0];
        let tokens = dims[1];
        let hidden_size = dims[2];

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output = native_metal.select_last_token_f32(
                    hidden_states.values(),
                    batch,
                    tokens,
                    hidden_size,
                )?;
                return Ok(Some(F32Tensor::new(output, [batch, 1, hidden_size])?));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal select_last_token is required on Metal",
                ));
            }
        }

        Ok(None)
    }

    fn heads_to_attention_layout_f32_tensor(&self, heads: &F32Tensor) -> Result<Option<F32Tensor>> {
        let dims = require_f32_rank("native_heads_to_attention_layout", heads, 4)?;
        let batch = dims[0];
        let tokens = dims[1];
        let head_count = dims[2];
        let head_dim = dims[3];

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output = native_metal.heads_to_attention_layout_f32(
                    heads.values(),
                    batch,
                    tokens,
                    head_count,
                    head_dim,
                )?;
                return Ok(Some(F32Tensor::new(
                    output,
                    [batch, head_count, tokens, head_dim],
                )?));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal heads_to_attention_layout is required on Metal",
                ));
            }
        }

        Ok(None)
    }

    fn merge_attention_heads_f32_tensor(
        &self,
        context_heads: &F32Tensor,
    ) -> Result<Option<F32Tensor>> {
        let dims = require_f32_rank("native_merge_attention_heads", context_heads, 4)?;
        let batch = dims[0];
        let head_count = dims[1];
        let tokens = dims[2];
        let head_dim = dims[3];
        let merged_width = head_count
            .checked_mul(head_dim)
            .ok_or_else(|| Error::backend("native merge_attention_heads width overflow"))?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output = native_metal.merge_attention_heads_f32(
                    context_heads.values(),
                    batch,
                    head_count,
                    tokens,
                    head_dim,
                )?;
                return Ok(Some(F32Tensor::new(output, [batch, tokens, merged_width])?));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal merge_attention_heads is required on Metal",
                ));
            }
        }

        Ok(None)
    }

    fn split_rope_tail_f32_tensor(
        &self,
        heads: &F32Tensor,
        no_rope_dim: usize,
        rope_dim: usize,
    ) -> Result<Option<(F32Tensor, F32Tensor)>> {
        let dims = require_f32_rank("native_split_rope_tail", heads, 4)?;
        let batch = dims[0];
        let tokens = dims[1];
        let head_count = dims[2];
        let total_dim = no_rope_dim
            .checked_add(rope_dim)
            .ok_or_else(|| Error::backend("native split_rope_tail total dim overflow"))?;
        validate_exact_shape("native_split_rope_tail_last_dim", &[dims[3]], &[total_dim])?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let (no_rope, rope) = native_metal.split_rope_tail_f32(
                    heads.values(),
                    batch,
                    tokens,
                    head_count,
                    no_rope_dim,
                    rope_dim,
                )?;
                return Ok(Some((
                    F32Tensor::new(no_rope, [batch, tokens, head_count, no_rope_dim])?,
                    F32Tensor::new(rope, [batch, tokens, head_count, rope_dim])?,
                )));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal split_rope_tail is required on Metal",
                ));
            }
        }

        Ok(None)
    }

    fn split_kv_mqa_f32_tensor(
        &self,
        kv_mqa: &F32Tensor,
        kv_lora_rank: usize,
        rope_dim: usize,
    ) -> Result<Option<(F32Tensor, F32Tensor)>> {
        let dims = require_f32_rank("native_split_kv_mqa", kv_mqa, 3)?;
        let batch = dims[0];
        let tokens = dims[1];
        let total_dim = kv_lora_rank
            .checked_add(rope_dim)
            .ok_or_else(|| Error::backend("native split_kv_mqa total dim overflow"))?;
        validate_exact_shape("native_split_kv_mqa_last_dim", &[dims[2]], &[total_dim])?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let (kv_latent, k_rope) = native_metal.split_kv_mqa_f32(
                    kv_mqa.values(),
                    batch,
                    tokens,
                    kv_lora_rank,
                    rope_dim,
                )?;
                return Ok(Some((
                    F32Tensor::new(kv_latent, [batch, tokens, kv_lora_rank])?,
                    F32Tensor::new(k_rope, [batch, tokens, 1, rope_dim])?,
                )));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal split_kv_mqa is required on Metal",
                ));
            }
        }

        Ok(None)
    }

    fn combine_rope_tail_f32_tensor(
        &self,
        no_rope: &F32Tensor,
        rope: &F32Tensor,
    ) -> Result<Option<F32Tensor>> {
        let no_rope_dims = require_f32_rank("native_combine_rope_tail_no_rope", no_rope, 4)?;
        let rope_dims = require_f32_rank("native_combine_rope_tail_rope", rope, 4)?;
        let batch = no_rope_dims[0];
        let tokens = no_rope_dims[1];
        let head_count = no_rope_dims[2];
        let no_rope_dim = no_rope_dims[3];
        let rope_head_count = rope_dims[2];
        let rope_dim = rope_dims[3];
        validate_exact_shape(
            "native_combine_rope_tail_batch_tokens",
            &[rope_dims[0], rope_dims[1]],
            &[batch, tokens],
        )?;
        let total_dim = no_rope_dim
            .checked_add(rope_dim)
            .ok_or_else(|| Error::backend("native combine_rope_tail total dim overflow"))?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output = native_metal.combine_rope_tail_f32(
                    no_rope.values(),
                    rope.values(),
                    batch,
                    tokens,
                    head_count,
                    rope_head_count,
                    no_rope_dim,
                    rope_dim,
                )?;
                return Ok(Some(F32Tensor::new(
                    output,
                    [batch, tokens, head_count, total_dim],
                )?));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal combine_rope_tail is required on Metal",
                ));
            }
        }

        Ok(None)
    }

    fn attention_scores_f32_tensor(
        &self,
        q: &F32Tensor,
        k: &F32Tensor,
        head_dim: usize,
    ) -> Result<Option<F32Tensor>> {
        let q_dims = require_f32_rank("native_attention_scores_q", q, 4)?;
        let k_dims = require_f32_rank("native_attention_scores_k", k, 4)?;
        let batch = q_dims[0];
        let heads = q_dims[1];
        let query_tokens = q_dims[2];
        let key_tokens = k_dims[2];
        validate_exact_shape(
            "native_attention_scores_batch_heads_dim",
            &[k_dims[0], k_dims[1], q_dims[3], k_dims[3]],
            &[batch, heads, head_dim, head_dim],
        )?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output = native_metal.attention_scores_f32(
                    q.values(),
                    k.values(),
                    batch,
                    heads,
                    query_tokens,
                    key_tokens,
                    head_dim,
                )?;
                return Ok(Some(F32Tensor::new(
                    output,
                    [batch, heads, query_tokens, key_tokens],
                )?));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal attention scores are required on Metal",
                ));
            }
        }

        Ok(None)
    }

    fn attention_values_f32_tensor(
        &self,
        probs: &F32Tensor,
        values: &F32Tensor,
    ) -> Result<Option<F32Tensor>> {
        let probs_dims = require_f32_rank("native_attention_values_probs", probs, 4)?;
        let values_dims = require_f32_rank("native_attention_values_values", values, 4)?;
        let batch = probs_dims[0];
        let heads = probs_dims[1];
        let query_tokens = probs_dims[2];
        let key_tokens = probs_dims[3];
        let value_dim = values_dims[3];
        validate_exact_shape(
            "native_attention_values_batch_heads_tokens",
            &[values_dims[0], values_dims[1], values_dims[2]],
            &[batch, heads, key_tokens],
        )?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output = native_metal.attention_values_f32(
                    probs.values(),
                    values.values(),
                    batch,
                    heads,
                    query_tokens,
                    key_tokens,
                    value_dim,
                )?;
                return Ok(Some(F32Tensor::new(
                    output,
                    [batch, heads, query_tokens, value_dim],
                )?));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal attention value aggregation is required on Metal",
                ));
            }
        }

        Ok(None)
    }

    fn attention_causal_softmax_f32_tensor(
        &self,
        scores: &F32Tensor,
        past_tokens: usize,
    ) -> Result<Option<F32Tensor>> {
        let dims = require_f32_rank("native_attention_causal_softmax", scores, 4)?;
        let batch = dims[0];
        let heads = dims[1];
        let query_tokens = dims[2];
        let key_tokens = dims[3];

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output = native_metal.attention_causal_softmax_f32(
                    scores.values(),
                    batch,
                    heads,
                    query_tokens,
                    key_tokens,
                    past_tokens,
                )?;
                return Ok(Some(F32Tensor::new(
                    output,
                    [batch, heads, query_tokens, key_tokens],
                )?));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal attention causal softmax is required on Metal",
                ));
            }
        }

        Ok(None)
    }

    fn decode_attention_f32_tensor(
        &self,
        q: &F32Tensor,
        k: &F32Tensor,
        v: &F32Tensor,
        head_dim: usize,
        past_tokens: usize,
    ) -> Result<Option<F32Tensor>> {
        let q_dims = require_f32_rank("native_decode_attention_q", q, 4)?;
        let k_dims = require_f32_rank("native_decode_attention_k", k, 4)?;
        let v_dims = require_f32_rank("native_decode_attention_v", v, 4)?;
        let batch = q_dims[0];
        let heads = q_dims[1];
        let query_tokens = q_dims[2];
        let key_tokens = k_dims[2];
        let value_dim = v_dims[3];
        validate_exact_shape(
            "native_decode_attention_query_tokens",
            &[query_tokens],
            &[1],
        )?;
        validate_exact_shape(
            "native_decode_attention_k_shape",
            &[k_dims[0], k_dims[1], k_dims[3]],
            &[batch, heads, head_dim],
        )?;
        validate_exact_shape(
            "native_decode_attention_v_shape",
            &[v_dims[0], v_dims[1], v_dims[2]],
            &[batch, heads, key_tokens],
        )?;
        past_tokens
            .checked_add(1)
            .filter(|expected| *expected == key_tokens)
            .ok_or_else(|| {
                Error::backend(format!(
                    "native decode attention expects key_tokens == past_tokens + 1, got key_tokens={key_tokens}, past_tokens={past_tokens}"
                ))
            })?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output = native_metal.decode_attention_f32(
                    q.values(),
                    k.values(),
                    v.values(),
                    batch,
                    heads,
                    key_tokens,
                    head_dim,
                    value_dim,
                    past_tokens,
                )?;
                return Ok(Some(F32Tensor::new(
                    output,
                    [batch, heads, query_tokens, value_dim],
                )?));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal fused decode attention is required on Metal",
                ));
            }
        }

        Ok(None)
    }

    fn paged_decode_attention_f32_tensor(
        &self,
        q: &F32Tensor,
        current_k: &F32Tensor,
        current_v: &F32Tensor,
        past_kv: &PagedKvView<'_>,
    ) -> Result<Option<F32Tensor>> {
        past_kv.validate()?;
        let q_dims = require_f32_rank("native_paged_decode_attention_q", q, 4)?;
        let current_k_dims =
            require_f32_rank("native_paged_decode_attention_current_k", current_k, 4)?;
        let current_v_dims =
            require_f32_rank("native_paged_decode_attention_current_v", current_v, 4)?;
        validate_exact_shape(
            "native_paged_decode_attention_q_shape",
            q_dims,
            &[
                past_kv.batch,
                past_kv.attention_heads,
                1,
                past_kv.key_head_dim,
            ],
        )?;
        validate_exact_shape(
            "native_paged_decode_attention_current_k_shape",
            current_k_dims,
            &[
                past_kv.batch,
                past_kv.attention_heads,
                1,
                past_kv.key_head_dim,
            ],
        )?;
        validate_exact_shape(
            "native_paged_decode_attention_current_v_shape",
            current_v_dims,
            &[
                past_kv.batch,
                past_kv.attention_heads,
                1,
                past_kv.value_head_dim,
            ],
        )?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output = native_metal.paged_decode_attention_f32(
                    q.values(),
                    current_k.values(),
                    current_v.values(),
                    past_kv,
                )?;
                return Ok(Some(F32Tensor::new(
                    output,
                    [
                        past_kv.batch,
                        past_kv.attention_heads,
                        1,
                        past_kv.value_head_dim,
                    ],
                )?));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal paged decode attention is required on Metal",
                ));
            }
        }

        Ok(None)
    }

    fn rope_slice_f32_tensor(
        &self,
        input: &F32Tensor,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
    ) -> Result<Option<F32Tensor>> {
        let dims = require_f32_rank("native_rope_slice", input, 4)?;
        let batch = dims[0];
        let tokens = dims[1];
        let heads = dims[2];
        validate_exact_shape("native_rope_slice_dim", &[dims[3]], &[rope_dim])?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output = native_metal.rope_slice_f32(
                    input.values(),
                    batch,
                    tokens,
                    heads,
                    rope_dim,
                    position_offset,
                    theta,
                )?;
                return Ok(Some(F32Tensor::new(
                    output,
                    [batch, tokens, heads, rope_dim],
                )?));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend("native Metal RoPE is required on Metal"));
            }
        }

        Ok(None)
    }

    fn rms_norm(&self, hidden_states: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        validate_rms_norm_shapes(hidden_states, weight)?;

        let input = tensor_to_f32_tensor(hidden_states)?;
        let weight = tensor_to_f32_tensor(weight)?;
        let output = self.rms_norm_f32(&input, &weight, eps)?;
        let (shape, values) = output.into_parts();
        Tensor::from_vec(values, shape.dims(), self.device()).map_err(Into::into)
    }

    fn rms_norm_f32(
        &self,
        hidden_states: &F32Tensor,
        weight: &F32Tensor,
        eps: f32,
    ) -> Result<F32Tensor> {
        validate_rms_norm_f32_shapes(hidden_states, weight, eps)?;
        let dims = hidden_states.dims();
        let hidden_size = *dims
            .last()
            .ok_or_else(|| Error::backend("rms_norm input has empty shape"))?;
        let rows = hidden_states
            .values()
            .len()
            .checked_div(hidden_size)
            .ok_or_else(|| Error::backend("rms_norm hidden_size division overflow"))?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output_values = native_metal.rms_norm_f32(
                    hidden_states.values(),
                    weight.values(),
                    rows,
                    hidden_size,
                    eps,
                )?;
                return F32Tensor::new(output_values, dims.to_vec());
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend("native Metal RMSNorm is required on Metal"));
            }
        }

        reference_rms_norm_f32(hidden_states, weight, eps)
    }

    fn moe_gather_tokens(&self, flat_tokens: &Tensor, token_indices: &[u32]) -> Result<Tensor> {
        validate_moe_gather_shapes(flat_tokens, token_indices)?;
        let token_count = flat_tokens.dims()[0];
        let hidden_size = flat_tokens.dims()[1];
        let assignment_count = token_indices.len();

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let flat_token_values = flat_tokens
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let output_values = native_metal.moe_gather_tokens_f32(
                    &flat_token_values,
                    token_indices,
                    token_count,
                    hidden_size,
                    assignment_count,
                )?;
                return Tensor::from_vec(
                    output_values,
                    (assignment_count, hidden_size),
                    self.device(),
                )
                .map_err(Into::into);
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal MoE token gather is required on Metal",
                ));
            }
        }

        reference_moe_gather_tokens(flat_tokens, token_indices, self.device())
    }

    fn moe_gather_tokens_f32_tensor(
        &self,
        flat_tokens: &F32Tensor,
        token_indices: &[u32],
    ) -> Result<Option<F32Tensor>> {
        let dims = require_f32_rank("native_moe_gather_tokens", flat_tokens, 2)?;
        let token_count = dims[0];
        let hidden_size = dims[1];
        let assignment_count = token_indices.len();

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output_values = native_metal.moe_gather_tokens_f32(
                    flat_tokens.values(),
                    token_indices,
                    token_count,
                    hidden_size,
                    assignment_count,
                )?;
                return Ok(Some(F32Tensor::new(
                    output_values,
                    [assignment_count, hidden_size],
                )?));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal MoE token gather is required on Metal",
                ));
            }
        }

        Ok(None)
    }

    fn moe_weighted_index_add_combine(
        &self,
        accumulator: &Tensor,
        token_indices: &Tensor,
        expert_outputs: &Tensor,
        expert_weights: &Tensor,
    ) -> Result<Tensor> {
        validate_moe_weighted_index_add_shapes(
            accumulator,
            token_indices,
            expert_outputs,
            expert_weights,
        )?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let token_count = accumulator.dims()[0];
                let hidden_size = accumulator.dims()[1];
                let assignment_count = token_indices.dims()[0];
                let accumulator_values = accumulator
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let token_index_values = token_indices.to_vec1::<u32>()?;
                let expert_output_values = expert_outputs
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let expert_weight_values = expert_weights
                    .to_dtype(common::DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let output_values = native_metal.moe_weighted_index_add_combine_f32(
                    &accumulator_values,
                    &token_index_values,
                    &expert_output_values,
                    &expert_weight_values,
                    token_count,
                    hidden_size,
                    assignment_count,
                )?;
                return Tensor::from_vec(output_values, (token_count, hidden_size), self.device())
                    .map_err(Into::into);
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal weighted MoE combine is required on Metal",
                ));
            }
        }

        reference_moe_weighted_index_add_combine(
            accumulator,
            token_indices,
            expert_outputs,
            expert_weights,
            self.device(),
        )
    }

    fn moe_weighted_index_add_combine_f32_tensor(
        &self,
        accumulator: &F32Tensor,
        token_indices: &[u32],
        expert_outputs: &F32Tensor,
        expert_weights: &[f32],
    ) -> Result<Option<F32Tensor>> {
        let accumulator_dims = require_f32_rank("native_moe_combine_accumulator", accumulator, 2)?;
        let expert_dims = require_f32_rank("native_moe_combine_expert_outputs", expert_outputs, 2)?;
        let token_count = accumulator_dims[0];
        let hidden_size = accumulator_dims[1];
        let assignment_count = token_indices.len();
        validate_exact_shape(
            "native_moe_combine_expert_shape",
            expert_dims,
            &[assignment_count, hidden_size],
        )?;
        validate_exact_shape(
            "native_moe_combine_weight_count",
            &[expert_weights.len()],
            &[assignment_count],
        )?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let output_values = native_metal.moe_weighted_index_add_combine_f32(
                    accumulator.values(),
                    token_indices,
                    expert_outputs.values(),
                    expert_weights,
                    token_count,
                    hidden_size,
                    assignment_count,
                )?;
                return Ok(Some(F32Tensor::new(
                    output_values,
                    [token_count, hidden_size],
                )?));
            }
            if self.device_kind == DeviceKind::Metal {
                return Err(Error::backend(
                    "native Metal weighted MoE combine is required on Metal",
                ));
            }
        }

        Ok(None)
    }

    fn q2_k_matvec_f32(
        &self,
        weights: &[u8],
        input: &Tensor,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<Tensor>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let input = tensor_to_f32_tensor(input)?;
            self.q2_k_matvec_f32_tensor(weights, &input, row_count, in_features, out_features)?
                .map(|output| tensor_from_f32_tensor(output, self.device()))
                .transpose()
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weights, input, row_count, in_features, out_features);
            Ok(None)
        }
    }

    fn q2_k_matvec_f32_tensor(
        &self,
        weights: &[u8],
        input: &F32Tensor,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<F32Tensor>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };

            let (actual_rows, output_shape) = matvec_input_shape(input, in_features, out_features)?;
            validate_exact_shape("native_q2_k_matvec_rows", &[actual_rows], &[row_count])?;

            let output_values = native_metal.q2_k_matvec_f32(
                weights,
                input.values(),
                row_count,
                in_features,
                out_features,
            )?;
            Ok(Some(F32Tensor::new(output_values, output_shape)?))
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weights, input, row_count, in_features, out_features);
            Ok(None)
        }
    }

    fn q2_k_matvec_add_f32_tensor(
        &self,
        weights: &[u8],
        input: &F32Tensor,
        residual: &F32Tensor,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<F32Tensor>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };

            let (actual_rows, output_shape) = matvec_input_shape(input, in_features, out_features)?;
            validate_exact_shape("native_q2_k_matvec_add_rows", &[actual_rows], &[row_count])?;
            validate_exact_shape(
                "native_q2_k_matvec_add_residual",
                residual.dims(),
                &output_shape,
            )?;

            let output_values = native_metal.q2_k_matvec_add_f32(
                weights,
                input.values(),
                residual.values(),
                row_count,
                in_features,
                out_features,
            )?;
            Ok(Some(F32Tensor::new(output_values, output_shape)?))
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                weights,
                input,
                residual,
                row_count,
                in_features,
                out_features,
            );
            Ok(None)
        }
    }

    fn q2_k_matvec_argmax_f32(
        &self,
        weights: &[u8],
        input: &Tensor,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<(u32, f32)>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let input = tensor_to_f32_tensor(input)?;
            self.q2_k_matvec_argmax_f32_tensor(
                weights,
                &input,
                row_count,
                in_features,
                out_features,
            )
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weights, input, row_count, in_features, out_features);
            Ok(None)
        }
    }

    fn q2_k_matvec_argmax_f32_tensor(
        &self,
        weights: &[u8],
        input: &F32Tensor,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<(u32, f32)>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };

            let (actual_rows, _output_shape) =
                matvec_input_shape(input, in_features, out_features)?;
            validate_exact_shape(
                "native_q2_k_matvec_argmax_rows",
                &[actual_rows],
                &[row_count],
            )?;
            validate_exact_shape("native_q2_k_matvec_argmax_row_count", &[row_count], &[1])?;

            let (token_id, token_score) = native_metal.q2_k_matvec_argmax_f32(
                weights,
                input.values(),
                row_count,
                in_features,
                out_features,
            )?;
            Ok(Some((token_id, token_score)))
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weights, input, row_count, in_features, out_features);
            Ok(None)
        }
    }

    fn q2_k_rms_norm_argmax_f32_tensor(
        &self,
        weights: &[u8],
        input: &F32Tensor,
        rms_weight: &F32Tensor,
        eps: f32,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<(u32, f32)>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };

            let (actual_rows, _output_shape) =
                matvec_input_shape(input, in_features, out_features)?;
            validate_exact_shape(
                "native_q2_k_rms_norm_argmax_rows",
                &[actual_rows],
                &[row_count],
            )?;
            validate_exact_shape("native_q2_k_rms_norm_argmax_row_count", &[row_count], &[1])?;
            validate_exact_shape(
                "native_q2_k_rms_norm_argmax_weight",
                rms_weight.dims(),
                &[in_features],
            )?;
            validate_rms_norm_f32_shapes(input, rms_weight, eps)?;

            let (token_id, token_score) = native_metal.q2_k_rms_norm_argmax_f32(
                weights,
                input.values(),
                rms_weight.values(),
                row_count,
                in_features,
                out_features,
                eps,
            )?;
            Ok(Some((token_id, token_score)))
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                weights,
                input,
                rms_weight,
                eps,
                row_count,
                in_features,
                out_features,
            );
            Ok(None)
        }
    }

    fn q2_k_gate_up_swiglu_f32(
        &self,
        gate_weights: &[u8],
        up_weights: &[u8],
        input: &Tensor,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<Tensor>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let input = tensor_to_f32_tensor(input)?;
            self.q2_k_gate_up_swiglu_f32_tensor(
                gate_weights,
                up_weights,
                &input,
                row_count,
                in_features,
                out_features,
            )?
            .map(|output| tensor_from_f32_tensor(output, self.device()))
            .transpose()
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                gate_weights,
                up_weights,
                input,
                row_count,
                in_features,
                out_features,
            );
            Ok(None)
        }
    }

    fn q2_k_gate_up_swiglu_f32_tensor(
        &self,
        gate_weights: &[u8],
        up_weights: &[u8],
        input: &F32Tensor,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<F32Tensor>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };

            let (actual_rows, output_shape) = matvec_input_shape(input, in_features, out_features)?;
            validate_exact_shape(
                "native_q2_k_gate_up_swiglu_rows",
                &[actual_rows],
                &[row_count],
            )?;

            let output_values = native_metal.q2_k_gate_up_swiglu_f32(
                gate_weights,
                up_weights,
                input.values(),
                row_count,
                in_features,
                out_features,
            )?;
            Ok(Some(F32Tensor::new(output_values, output_shape)?))
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                gate_weights,
                up_weights,
                input,
                row_count,
                in_features,
                out_features,
            );
            Ok(None)
        }
    }

    fn q2_k_transposed_matvec_f32(
        &self,
        weights: &[u8],
        input: &Tensor,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<Tensor>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let input = tensor_to_f32_tensor(input)?;
            self.q2_k_transposed_matvec_f32_tensor(
                weights,
                &input,
                row_count,
                in_features,
                out_features,
            )?
            .map(|output| tensor_from_f32_tensor(output, self.device()))
            .transpose()
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weights, input, row_count, in_features, out_features);
            Ok(None)
        }
    }

    fn q2_k_transposed_matvec_f32_tensor(
        &self,
        weights: &[u8],
        input: &F32Tensor,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<F32Tensor>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };

            let (actual_rows, output_shape) = matvec_input_shape(input, in_features, out_features)?;
            validate_exact_shape(
                "native_q2_k_transposed_matvec_rows",
                &[actual_rows],
                &[row_count],
            )?;

            let output_values = native_metal.q2_k_transposed_matvec_f32(
                weights,
                input.values(),
                row_count,
                in_features,
                out_features,
            )?;
            Ok(Some(F32Tensor::new(output_values, output_shape)?))
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weights, input, row_count, in_features, out_features);
            Ok(None)
        }
    }

    fn q8_0_matvec_f32_tensor(
        &self,
        weights: &[u8],
        input: &F32Tensor,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<F32Tensor>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };

            let (actual_rows, output_shape) = matvec_input_shape(input, in_features, out_features)?;
            validate_exact_shape("native_q8_0_matvec_rows", &[actual_rows], &[row_count])?;

            let output_values = native_metal.q8_0_matvec_f32(
                weights,
                input.values(),
                row_count,
                in_features,
                out_features,
            )?;
            Ok(Some(F32Tensor::new(output_values, output_shape)?))
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weights, input, row_count, in_features, out_features);
            Ok(None)
        }
    }

    fn q8_0_transposed_matvec_f32_tensor(
        &self,
        weights: &[u8],
        input: &F32Tensor,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<F32Tensor>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };

            let (actual_rows, output_shape) = matvec_input_shape(input, in_features, out_features)?;
            validate_exact_shape(
                "native_q8_0_transposed_matvec_rows",
                &[actual_rows],
                &[row_count],
            )?;

            let output_values = native_metal.q8_0_transposed_matvec_f32(
                weights,
                input.values(),
                row_count,
                in_features,
                out_features,
            )?;
            Ok(Some(F32Tensor::new(output_values, output_shape)?))
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weights, input, row_count, in_features, out_features);
            Ok(None)
        }
    }

    fn device_values_supported(&self) -> bool {
        self.has_native_metal()
    }

    fn device_flush(&self) -> Result<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                return native_metal.batch_flush();
            }
        }
        Ok(())
    }

    fn device_submit(&self) -> Result<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                return native_metal.batch_submit();
            }
        }
        Ok(())
    }

    fn device_profile_boundary(&self, label: &str) -> Result<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                return native_metal.batch_submit_profile_segment(label);
            }
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let _ = label;
        Ok(())
    }

    fn device_upload_f32_tensor(&self, tensor: &F32Tensor) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let buffer = native_metal.batch_upload_f32(tensor.values())?;
            return Ok(Some(DeviceValue::new(tensor.dims().to_vec(), buffer)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = tensor;
            Ok(None)
        }
    }

    fn device_upload_f32_tensor_as_f16(&self, tensor: &F32Tensor) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let buffer = native_metal.batch_upload_f32_as_f16(tensor.values())?;
            return Ok(Some(DeviceValue::new_with_dtype(
                tensor.dims().to_vec(),
                DType::F16,
                buffer,
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = tensor;
            Ok(None)
        }
    }

    fn device_alloc_f32_tensor(&self, dims: &[usize]) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let len = dims.iter().try_fold(1_usize, |count, dim| {
                count
                    .checked_mul(*dim)
                    .ok_or_else(|| Error::backend("device allocation element count overflow"))
            })?;
            let buffer = native_metal.batched_alloc_f32(len)?;
            return Ok(Some(DeviceValue::new(dims.to_vec(), buffer)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = dims;
            Ok(None)
        }
    }

    fn device_alloc_f16_tensor(&self, dims: &[usize]) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let len = dims.iter().try_fold(1_usize, |count, dim| {
                count
                    .checked_mul(*dim)
                    .ok_or_else(|| Error::backend("device f16 allocation element count overflow"))
            })?;
            let buffer = native_metal.batched_alloc_f16(len)?;
            return Ok(Some(DeviceValue::new_with_dtype(
                dims.to_vec(),
                DType::F16,
                buffer,
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = dims;
            Ok(None)
        }
    }

    fn device_copy_same_dtype(
        &self,
        source: &DeviceValue,
        source_offset: usize,
        destination: &DeviceValue,
        destination_offset: usize,
        len: usize,
    ) -> Result<Option<()>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            if source.dtype() != destination.dtype() {
                return Err(Error::backend(format!(
                    "device copy dtype mismatch: source={:?}, destination={:?}",
                    source.dtype(),
                    destination.dtype()
                )));
            }
            let source_count = source.element_count()?;
            let destination_count = destination.element_count()?;
            source_offset
                .checked_add(len)
                .filter(|end| *end <= source_count)
                .ok_or_else(|| {
                    Error::backend(format!(
                        "device copy source range {source_offset}..{} exceeds element count {source_count}",
                        source_offset.saturating_add(len)
                    ))
                })?;
            destination_offset
                .checked_add(len)
                .filter(|end| *end <= destination_count)
                .ok_or_else(|| {
                    Error::backend(format!(
                        "device copy destination range {destination_offset}..{} exceeds element count {destination_count}",
                        destination_offset.saturating_add(len)
                    ))
                })?;
            native_metal.batched_element_copy(
                &source.buffer,
                source_offset,
                &destination.buffer,
                destination_offset,
                len,
                source.dtype().byte_size(),
            )?;
            return Ok(Some(()));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (source, source_offset, destination, destination_offset, len);
            Ok(None)
        }
    }

    fn device_copy_f32(
        &self,
        source: &DeviceValue,
        source_offset: usize,
        destination: &DeviceValue,
        destination_offset: usize,
        len: usize,
    ) -> Result<Option<()>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            if source.dtype() != DType::F32 || destination.dtype() != DType::F32 {
                return Err(Error::backend(format!(
                    "device f32 copy requires f32 tensors, got source={:?}, destination={:?}",
                    source.dtype(),
                    destination.dtype()
                )));
            }
            let source_count = source.element_count()?;
            let destination_count = destination.element_count()?;
            source_offset
                .checked_add(len)
                .filter(|end| *end <= source_count)
                .ok_or_else(|| {
                    Error::backend(format!(
                        "device copy source range {source_offset}..{} exceeds element count {source_count}",
                        source_offset.saturating_add(len)
                    ))
                })?;
            destination_offset
                .checked_add(len)
                .filter(|end| *end <= destination_count)
                .ok_or_else(|| {
                    Error::backend(format!(
                        "device copy destination range {destination_offset}..{} exceeds element count {destination_count}",
                        destination_offset.saturating_add(len)
                    ))
                })?;
            native_metal.batched_f32_copy(
                &source.buffer,
                source_offset,
                &destination.buffer,
                destination_offset,
                len,
            )?;
            return Ok(Some(()));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (source, source_offset, destination, destination_offset, len);
            Ok(None)
        }
    }

    fn device_download_f32_tensor(&self, value: &DeviceValue) -> Result<F32Tensor> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(native_metal) = self.native_metal() {
                let values =
                    match value.dtype() {
                        DType::F32 => {
                            native_metal.batch_read_f32(&value.buffer, value.element_count()?)?
                        }
                        DType::F16 => native_metal
                            .batch_read_f16_as_f32(&value.buffer, value.element_count()?)?,
                        DType::BF16 => return Err(Error::backend(
                            "BF16 device download is not implemented for the native Metal backend",
                        )),
                    };
                return F32Tensor::new(values, value.dims().to_vec());
            }
        }

        let _ = value;
        Err(Error::backend(
            "device-resident values are not supported by this backend",
        ))
    }

    fn rms_norm_device(
        &self,
        input: &DeviceValue,
        weight: &F32Tensor,
        eps: f32,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let dims = input.dims();
            let hidden_size = *dims
                .last()
                .ok_or_else(|| Error::backend("rms_norm device input has empty shape"))?;
            let input_len = input.element_count()?;
            let rows = if hidden_size == 0 {
                return Err(Error::backend(
                    "rms_norm device hidden_size must be non-zero",
                ));
            } else {
                input_len / hidden_size
            };
            let buffer = native_metal.batched_rms_norm(
                &input.buffer,
                input_len,
                weight.values(),
                rows,
                hidden_size,
                eps,
            )?;
            return Ok(Some(DeviceValue::new(dims.to_vec(), buffer)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (input, weight, eps);
            Ok(None)
        }
    }

    fn laguna_xs_rms_norm_device(
        &self,
        input: &DeviceValue,
        weight: &F32Tensor,
        eps: f32,
    ) -> Result<Option<DeviceValue>> {
        const HIDDEN_SIZE: usize = 2_048;
        if input.dtype() != DType::F32 {
            return Err(Error::backend(format!(
                "Laguna XS RMSNorm input must be F32, got {:?}",
                input.dtype()
            )));
        }
        if input.dims().last().copied() != Some(HIDDEN_SIZE)
            || input.element_count()? != HIDDEN_SIZE
            || weight.dims() != [HIDDEN_SIZE]
        {
            return Ok(None);
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let output = native_metal.batched_laguna_xs_rms_norm(
                &input.buffer,
                HIDDEN_SIZE,
                weight.values(),
                eps,
            )?;
            return Ok(Some(DeviceValue::new(input.dims().to_vec(), output)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (input, weight, eps);
            Ok(None)
        }
    }

    fn laguna_xs_rms_norm_router_device(
        &self,
        input: &DeviceValue,
        norm_weight: &F32Tensor,
        router_weight: &F32Tensor,
        eps: f32,
    ) -> Result<Option<(DeviceValue, DeviceValue)>> {
        const HIDDEN_SIZE: usize = 2_048;
        const EXPERT_COUNT: usize = 256;
        if input.dtype() != DType::F32 {
            return Err(Error::backend(format!(
                "Laguna XS fused norm+router input must be F32, got {:?}",
                input.dtype()
            )));
        }
        if input.dims().last().copied() != Some(HIDDEN_SIZE)
            || input.element_count()? != HIDDEN_SIZE
            || norm_weight.dims() != [HIDDEN_SIZE]
            || router_weight.dims() != [EXPERT_COUNT, HIDDEN_SIZE]
        {
            return Ok(None);
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let (normalized, router_logits) = native_metal.batched_laguna_xs_rms_norm_router(
                &input.buffer,
                HIDDEN_SIZE,
                norm_weight.values(),
                router_weight.values(),
                eps,
            )?;
            return Ok(Some((
                DeviceValue::new(input.dims().to_vec(), normalized),
                DeviceValue::new(vec![1, EXPERT_COUNT], router_logits),
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (input, norm_weight, router_weight, eps);
            Ok(None)
        }
    }

    fn prepare_bf16_matrix(
        &self,
        bytes: &[u8],
        rows: usize,
        columns: usize,
    ) -> Result<Option<DeviceBf16Matrix>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            return native_metal
                .prepare_bf16_matrix(bytes, rows, columns)
                .map(Some);
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (bytes, rows, columns);
            Ok(None)
        }
    }

    fn bf16_linear_device(
        &self,
        matrix: &DeviceBf16Matrix,
        input: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            if input.dtype() != DType::F32 {
                return Err(Error::backend(format!(
                    "BF16 linear input must be F32, got {:?}",
                    input.dtype()
                )));
            }
            let (row_count, output_shape) =
                matvec_dims_shape(input.dims(), matrix.columns, matrix.rows)?;
            let output = native_metal.batched_bf16_linear(
                matrix,
                &input.buffer,
                input.element_count()?,
                row_count,
            )?;
            return Ok(Some(DeviceValue::new(output_shape, output)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (matrix, input);
            Ok(None)
        }
    }

    fn bf16_gate_up_swiglu_device(
        &self,
        gate: &DeviceBf16Matrix,
        up: &DeviceBf16Matrix,
        input: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        if input.dtype() != DType::F32 {
            return Err(Error::backend(format!(
                "BF16 gate/up SwiGLU input must be F32, got {:?}",
                input.dtype()
            )));
        }
        if gate.rows != up.rows || gate.columns != up.columns {
            return Err(Error::backend(format!(
                "BF16 gate/up matrices must match, got [{},{}] and [{},{}]",
                gate.rows, gate.columns, up.rows, up.columns
            )));
        }
        let (row_count, output_shape) = matvec_dims_shape(input.dims(), gate.columns, gate.rows)?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let output = native_metal.batched_bf16_gate_up_swiglu(
                gate,
                up,
                &input.buffer,
                input.element_count()?,
                row_count,
            )?;
            return Ok(Some(DeviceValue::new(output_shape, output)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (gate, up, input, row_count, output_shape);
            Ok(None)
        }
    }

    fn laguna_attention_projections_device(
        &self,
        query: &DeviceBf16Matrix,
        key: &DeviceBf16Matrix,
        value: &DeviceBf16Matrix,
        gate: &DeviceBf16Matrix,
        input: &DeviceValue,
    ) -> Result<Option<LagunaAttentionProjections>> {
        const HEAD_DIM: usize = 128;
        const KV_HEADS: usize = 8;

        let [batch, tokens, hidden_size] = input.dims() else {
            return Err(Error::backend(format!(
                "Laguna attention projection input must be [B,T,H], got {:?}",
                input.dims()
            )));
        };
        if input.dtype() != DType::F32 || *batch == 0 || *tokens == 0 {
            return Err(Error::backend(format!(
                "Laguna attention projection input must be non-empty F32 [B,T,H], got {:?} {:?}",
                input.dtype(),
                input.dims()
            )));
        }
        if query.columns != *hidden_size
            || key.columns != *hidden_size
            || value.columns != *hidden_size
            || gate.columns != *hidden_size
        {
            return Err(Error::backend(format!(
                "Laguna attention projection matrices must consume hidden width {hidden_size}"
            )));
        }
        if query.rows % HEAD_DIM != 0
            || key.rows != KV_HEADS * HEAD_DIM
            || value.rows != KV_HEADS * HEAD_DIM
            || gate.rows != query.rows / HEAD_DIM
        {
            return Err(Error::backend(format!(
                "Laguna attention projection widths must be Q=[heads*{HEAD_DIM}], K/V=[{}], gate=[heads]; got Q={}, K={}, V={}, gate={}",
                KV_HEADS * HEAD_DIM,
                query.rows,
                key.rows,
                value.rows,
                gate.rows
            )));
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let row_count = batch
                .checked_mul(*tokens)
                .ok_or_else(|| Error::backend("Laguna attention projection row count overflow"))?;
            let (query_buffer, key_buffer, value_buffer, gate_buffer) = native_metal
                .batched_laguna_attention_projections(
                    query,
                    key,
                    value,
                    gate,
                    &input.buffer,
                    input.element_count()?,
                    row_count,
                )?;
            let query_heads = query.rows / HEAD_DIM;
            return Ok(Some(LagunaAttentionProjections {
                query: DeviceValue::new(vec![*batch, *tokens, query_heads, HEAD_DIM], query_buffer),
                key: DeviceValue::new(vec![*batch, *tokens, KV_HEADS, HEAD_DIM], key_buffer),
                value: DeviceValue::new(vec![*batch, *tokens, KV_HEADS, HEAD_DIM], value_buffer),
                gate: DeviceValue::new(vec![*batch, *tokens, query_heads], gate_buffer),
            }));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (query, key, value, gate, input);
            Ok(None)
        }
    }

    fn bf16_embedding_device(
        &self,
        embedding: &DeviceBf16Matrix,
        token_ids: &[u32],
        token_shape: &[usize],
    ) -> Result<Option<DeviceValue>> {
        let token_count = token_shape.iter().try_fold(1_usize, |count, dim| {
            count
                .checked_mul(*dim)
                .ok_or_else(|| Error::backend("BF16 embedding token count overflow"))
        })?;
        if token_count != token_ids.len() {
            return Err(Error::backend(format!(
                "BF16 embedding token shape {token_shape:?} contains {token_count} IDs, got {}",
                token_ids.len()
            )));
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let output = native_metal.batched_bf16_embedding(embedding, token_ids)?;
            let mut output_shape = token_shape.to_vec();
            output_shape.push(embedding.columns);
            return Ok(Some(DeviceValue::new(output_shape, output)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (embedding, token_ids, token_count);
            Ok(None)
        }
    }

    fn prepare_rope_table(
        &self,
        inverse_frequency: &[f32],
        rotary_dim: usize,
        attention_factor: f32,
    ) -> Result<Option<DeviceRopeTable>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            return native_metal
                .prepare_rope_table(inverse_frequency, rotary_dim, attention_factor)
                .map(Some);
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (inverse_frequency, rotary_dim, attention_factor);
            Ok(None)
        }
    }

    fn qk_rms_norm_rope_device(
        &self,
        input: &DeviceValue,
        norm_weight: &F32Tensor,
        eps: f32,
        position_offset: usize,
        table: &DeviceRopeTable,
    ) -> Result<Option<DeviceValue>> {
        if input.dtype() != DType::F32 {
            return Err(Error::backend(format!(
                "Laguna Q/K RMSNorm RoPE input must be F32, got {:?}",
                input.dtype()
            )));
        }
        let dims = require_device_rank("Laguna Q/K RMSNorm RoPE", input, 4)?;
        let batch_count = dims[0];
        let token_count = dims[1];
        let head_count = dims[2];
        let head_dim = dims[3];
        if head_dim != 128 {
            return Err(Error::backend(format!(
                "Laguna Q/K head dimension must be 128, got {head_dim}"
            )));
        }
        validate_exact_shape("Laguna Q/K norm weight", norm_weight.dims(), &[head_dim])?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let output = native_metal.batched_qk_rms_norm_rope(
                &input.buffer,
                input.element_count()?,
                norm_weight.values(),
                batch_count,
                token_count,
                head_count,
                position_offset,
                eps,
                table,
            )?;
            return Ok(Some(DeviceValue::new(dims.to_vec(), output)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (eps, position_offset, table);
            Ok(None)
        }
    }

    fn laguna_qk_rms_norm_rope_pair_device(
        &self,
        query: &DeviceValue,
        key: &DeviceValue,
        query_norm_weight: &F32Tensor,
        key_norm_weight: &F32Tensor,
        eps: f32,
        position_offset: usize,
        table: &DeviceRopeTable,
    ) -> Result<Option<(DeviceValue, DeviceValue)>> {
        if query.dtype() != DType::F32 || key.dtype() != DType::F32 {
            return Err(Error::backend(format!(
                "Laguna Q/K pair must be F32, got {:?}/{:?}",
                query.dtype(),
                key.dtype()
            )));
        }
        let query_dims = require_device_rank("Laguna query RMSNorm RoPE", query, 4)?;
        let key_dims = require_device_rank("Laguna key RMSNorm RoPE", key, 4)?;
        let query_batch = query_dims[0];
        let query_tokens = query_dims[1];
        let query_heads = query_dims[2];
        let query_head_dim = query_dims[3];
        let key_batch = key_dims[0];
        let key_tokens = key_dims[1];
        let key_heads = key_dims[2];
        let key_head_dim = key_dims[3];
        if query_batch != key_batch
            || query_tokens != key_tokens
            || query_head_dim != 128
            || key_head_dim != 128
            || key_heads != 8
        {
            return Err(Error::backend(format!(
                "Laguna Q/K pair shapes must be [B,T,H,128]/[B,T,8,128], got {query_dims:?}/{key_dims:?}"
            )));
        }
        validate_exact_shape(
            "Laguna query norm weight",
            query_norm_weight.dims(),
            &[query_head_dim],
        )?;
        validate_exact_shape(
            "Laguna key norm weight",
            key_norm_weight.dims(),
            &[key_head_dim],
        )?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let (query_output, key_output) = native_metal.batched_laguna_qk_rms_norm_rope_pair(
                &query.buffer,
                query.element_count()?,
                &key.buffer,
                key.element_count()?,
                query_norm_weight.values(),
                key_norm_weight.values(),
                query_batch,
                query_tokens,
                query_heads,
                key_heads,
                position_offset,
                eps,
                table,
            )?;
            return Ok(Some((
                DeviceValue::new(query_dims.to_vec(), query_output),
                DeviceValue::new(key_dims.to_vec(), key_output),
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                query,
                key,
                query_norm_weight,
                key_norm_weight,
                eps,
                position_offset,
                table,
            );
            Ok(None)
        }
    }

    fn laguna_xs_qk_rms_norm_rope_pair_device(
        &self,
        query: &DeviceValue,
        key: &DeviceValue,
        query_norm_weight: &F32Tensor,
        key_norm_weight: &F32Tensor,
        eps: f32,
        position_offset: usize,
        table: &DeviceRopeTable,
    ) -> Result<Option<(DeviceValue, DeviceValue)>> {
        if query.dtype() != DType::F32 || key.dtype() != DType::F32 {
            return Err(Error::backend(format!(
                "Laguna XS Q/K pair must be F32, got {:?}/{:?}",
                query.dtype(),
                key.dtype()
            )));
        }
        let query_dims = require_device_rank("Laguna XS query RMSNorm RoPE", query, 4)?;
        let key_dims = require_device_rank("Laguna XS key RMSNorm RoPE", key, 4)?;
        let [query_batch, query_tokens, query_heads, query_head_dim] = query_dims else {
            unreachable!("rank validated above")
        };
        let [key_batch, key_tokens, key_heads, key_head_dim] = key_dims else {
            unreachable!("rank validated above")
        };
        validate_exact_shape(
            "Laguna XS Q/K batch and token shape",
            &[*key_batch, *key_tokens],
            &[*query_batch, *query_tokens],
        )?;
        validate_exact_shape(
            "Laguna XS query norm weight",
            query_norm_weight.dims(),
            &[*query_head_dim],
        )?;
        validate_exact_shape(
            "Laguna XS key norm weight",
            key_norm_weight.dims(),
            &[*key_head_dim],
        )?;
        if *query_batch != 1
            || *query_tokens != 1
            || !matches!(*query_heads, 48 | 64)
            || *query_head_dim != 128
            || *key_heads != 8
            || *key_head_dim != 128
            || !matches!(table.rotary_dim, 64 | 128)
        {
            return Ok(None);
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let (query_output, key_output) = native_metal.batched_laguna_xs_qk_rms_norm_rope_pair(
                &query.buffer,
                query.element_count()?,
                &key.buffer,
                key.element_count()?,
                query_norm_weight.values(),
                key_norm_weight.values(),
                *query_heads,
                position_offset,
                eps,
                table,
            )?;
            return Ok(Some((
                DeviceValue::new(query_dims.to_vec(), query_output),
                DeviceValue::new(key_dims.to_vec(), key_output),
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (eps, position_offset, table);
            Ok(None)
        }
    }

    fn prepare_laguna_fp8_kv_cache(
        &self,
        batch: usize,
        capacity_tokens: usize,
        retention: LagunaKvRetention,
        key_scale: f32,
        value_scale: f32,
    ) -> Result<Option<LagunaFp8KvCache>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            return native_metal
                .prepare_laguna_fp8_kv_cache(
                    batch,
                    capacity_tokens,
                    retention,
                    key_scale,
                    value_scale,
                )
                .map(Some);
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (batch, capacity_tokens, retention, key_scale, value_scale);
            Ok(None)
        }
    }

    fn grow_laguna_fp8_kv_cache(
        &self,
        cache: &mut LagunaFp8KvCache,
        capacity_tokens: usize,
    ) -> Result<bool> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(false);
            };
            native_metal.grow_laguna_fp8_kv_cache(cache, capacity_tokens)?;
            return Ok(true);
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (cache, capacity_tokens);
            Ok(false)
        }
    }

    fn laguna_gated_gqa_attention_device(
        &self,
        query: &DeviceValue,
        current_key: &DeviceValue,
        current_value: &DeviceValue,
        gate: &DeviceValue,
        cache: &mut LagunaFp8KvCache,
    ) -> Result<Option<DeviceValue>> {
        for (label, value) in [
            ("query", query),
            ("current key", current_key),
            ("current value", current_value),
            ("gate", gate),
        ] {
            if value.dtype() != DType::F32 {
                return Err(Error::backend(format!(
                    "Laguna {label} must be F32, got {:?}",
                    value.dtype()
                )));
            }
        }
        let query_dims = require_device_rank("Laguna query", query, 4)?;
        let key_dims = require_device_rank("Laguna current key", current_key, 4)?;
        let value_dims = require_device_rank("Laguna current value", current_value, 4)?;
        let gate_dims = require_device_rank("Laguna attention gate", gate, 3)?;
        let batch = query_dims[0];
        let query_tokens = query_dims[1];
        let query_heads = query_dims[2];
        let head_dim = query_dims[3];
        validate_exact_shape(
            "Laguna current key shape",
            key_dims,
            &[batch, query_tokens, 8, 128],
        )?;
        validate_exact_shape(
            "Laguna current value shape",
            value_dims,
            &[batch, query_tokens, 8, 128],
        )?;
        validate_exact_shape(
            "Laguna attention gate shape",
            gate_dims,
            &[batch, query_tokens, query_heads],
        )?;
        if head_dim != 128 || !matches!(query_heads, 48 | 72) {
            return Err(Error::backend(format!(
                "Laguna query shape must end in [48|72,128], got {query_dims:?}"
            )));
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let output = native_metal.batched_laguna_gated_gqa_attention(
                &query.buffer,
                query.element_count()?,
                &current_key.buffer,
                current_key.element_count()?,
                &current_value.buffer,
                current_value.element_count()?,
                &gate.buffer,
                gate.element_count()?,
                batch,
                query_tokens,
                query_heads,
                cache,
            )?;
            cache.commit_append(query_tokens)?;
            return Ok(Some(DeviceValue::new(query_dims.to_vec(), output)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = cache;
            Ok(None)
        }
    }

    fn prepare_laguna_f16_kv_cache(
        &self,
        batch: usize,
        capacity_tokens: usize,
        retention: LagunaKvRetention,
    ) -> Result<Option<LagunaF16KvCache>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            return native_metal
                .prepare_laguna_f16_kv_cache(batch, capacity_tokens, retention)
                .map(Some);
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (batch, capacity_tokens, retention);
            Ok(None)
        }
    }

    fn grow_laguna_f16_kv_cache(
        &self,
        cache: &mut LagunaF16KvCache,
        capacity_tokens: usize,
    ) -> Result<bool> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(false);
            };
            native_metal.grow_laguna_f16_kv_cache(cache, capacity_tokens)?;
            return Ok(true);
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (cache, capacity_tokens);
            Ok(false)
        }
    }

    fn laguna_gated_gqa_f16_attention_device(
        &self,
        query: &DeviceValue,
        current_key: &DeviceValue,
        current_value: &DeviceValue,
        gate: &DeviceValue,
        cache: &mut LagunaF16KvCache,
    ) -> Result<Option<DeviceValue>> {
        for (label, value) in [
            ("query", query),
            ("current key", current_key),
            ("current value", current_value),
            ("gate", gate),
        ] {
            if value.dtype() != DType::F32 {
                return Err(Error::backend(format!(
                    "Laguna F16 {label} must be F32, got {:?}",
                    value.dtype()
                )));
            }
        }
        let query_dims = require_device_rank("Laguna F16 query", query, 4)?;
        let key_dims = require_device_rank("Laguna F16 current key", current_key, 4)?;
        let value_dims = require_device_rank("Laguna F16 current value", current_value, 4)?;
        let gate_dims = require_device_rank("Laguna F16 attention gate", gate, 3)?;
        let batch = query_dims[0];
        let query_tokens = query_dims[1];
        let query_heads = query_dims[2];
        let head_dim = query_dims[3];
        validate_exact_shape(
            "Laguna F16 current key shape",
            key_dims,
            &[batch, query_tokens, 8, 128],
        )?;
        validate_exact_shape(
            "Laguna F16 current value shape",
            value_dims,
            &[batch, query_tokens, 8, 128],
        )?;
        validate_exact_shape(
            "Laguna F16 attention gate shape",
            gate_dims,
            &[batch, query_tokens, query_heads],
        )?;
        if head_dim != 128 || query_heads == 0 || query_heads > 72 || !query_heads.is_multiple_of(8)
        {
            return Err(Error::backend(format!(
                "Laguna F16 query shape must end in [QH,128], where QH is a positive multiple of 8 up to 72; got {query_dims:?}"
            )));
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let output = native_metal.batched_laguna_gated_gqa_f16_attention(
                &query.buffer,
                query.element_count()?,
                &current_key.buffer,
                current_key.element_count()?,
                &current_value.buffer,
                current_value.element_count()?,
                &gate.buffer,
                gate.element_count()?,
                batch,
                query_tokens,
                query_heads,
                cache,
            )?;
            cache.commit_append(query_tokens)?;
            return Ok(Some(DeviceValue::new(query_dims.to_vec(), output)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = cache;
            Ok(None)
        }
    }

    fn prepare_w4_groupwise_weight(
        &self,
        packed: &[u8],
        scales: &[u8],
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> Result<Option<DeviceW4Weight>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            return native_metal
                .prepare_w4_groupwise_weight(packed, scales, in_features, out_features, group_size)
                .map(Some);
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (packed, scales, in_features, out_features, group_size);
            Ok(None)
        }
    }

    fn prepare_w4_groupwise_weight_no_copy(
        &self,
        source: Arc<dyn W4WeightSource>,
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> Result<Option<DeviceW4Weight>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            return native_metal
                .prepare_w4_groupwise_weight_no_copy(source, in_features, out_features, group_size)
                .map(Some);
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (source, in_features, out_features, group_size);
            Ok(None)
        }
    }

    fn w4_groupwise_matvec_device(
        &self,
        weight: &DeviceW4Weight,
        input: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            if input.dtype() != DType::F32 {
                return Err(Error::backend(format!(
                    "W4 matvec input must be F32, got {:?}",
                    input.dtype()
                )));
            }
            let (row_count, output_shape) =
                matvec_dims_shape(input.dims(), weight.in_features, weight.out_features)?;
            let output = native_metal.batched_w4_groupwise_matvec(
                weight,
                &input.buffer,
                input.element_count()?,
                row_count,
            )?;
            return Ok(Some(DeviceValue::new(output_shape, output)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weight, input);
            Ok(None)
        }
    }

    fn w4_groupwise_gate_up_swiglu_device(
        &self,
        gate: &DeviceW4Weight,
        up: &DeviceW4Weight,
        input: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            if input.dtype() != DType::F32 {
                return Err(Error::backend(format!(
                    "W4 gate/up input must be F32, got {:?}",
                    input.dtype()
                )));
            }
            if gate.in_features != up.in_features
                || gate.out_features != up.out_features
                || gate.group_size != up.group_size
            {
                return Err(Error::backend(
                    "W4 gate/up device weights do not share one layout",
                ));
            }
            let (row_count, output_shape) =
                matvec_dims_shape(input.dims(), gate.in_features, gate.out_features)?;
            let output = native_metal.batched_w4_groupwise_gate_up_swiglu(
                gate,
                up,
                &input.buffer,
                input.element_count()?,
                row_count,
            )?;
            return Ok(Some(DeviceValue::new(output_shape, output)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (gate, up, input);
            Ok(None)
        }
    }

    fn w4_groupwise_expert_wave_device(
        &self,
        groups: &[W4ExpertGroup<'_>],
        input: &DeviceValue,
        token_count: usize,
        top_k: usize,
        destination: &DeviceValue,
    ) -> Result<Option<()>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            if input.dtype() != DType::F32 || destination.dtype() != DType::F32 {
                return Err(Error::backend(
                    "W4 expert wave input and destination must be F32",
                ));
            }
            let input_dims = require_device_rank("W4 expert wave input", input, 2)?;
            let destination_dims =
                require_device_rank("W4 expert wave destination", destination, 2)?;
            let assignment_count = token_count
                .checked_mul(top_k)
                .ok_or_else(|| Error::backend("W4 expert wave assignment count overflow"))?;
            let hidden_size = groups
                .first()
                .map(|group| group.gate.in_features)
                .ok_or_else(|| Error::backend("W4 expert wave requires at least one expert"))?;
            validate_exact_shape(
                "W4 expert wave input",
                input_dims,
                &[token_count, hidden_size],
            )?;
            validate_exact_shape(
                "W4 expert wave destination",
                destination_dims,
                &[assignment_count, hidden_size],
            )?;
            native_metal.batched_w4_expert_wave(
                groups,
                &input.buffer,
                input.element_count()?,
                token_count,
                top_k,
                &destination.buffer,
                destination.element_count()?,
            )?;
            return Ok(Some(()));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (groups, input, token_count, top_k, destination);
            Ok(None)
        }
    }

    fn q2_k_matvec_device(
        &self,
        weights: &[u8],
        input: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            return self.quant_matvec_device(
                QuantMatvecKind::Q2K,
                weights,
                input,
                row_count,
                in_features,
                out_features,
            );
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weights, input, row_count, in_features, out_features);
            Ok(None)
        }
    }

    fn q2_k_transposed_matvec_device(
        &self,
        weights: &[u8],
        input: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            return self.quant_matvec_device(
                QuantMatvecKind::Q2KTransposed,
                weights,
                input,
                row_count,
                in_features,
                out_features,
            );
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weights, input, row_count, in_features, out_features);
            Ok(None)
        }
    }

    fn q8_0_matvec_device(
        &self,
        weights: &[u8],
        input: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            return self.quant_matvec_device(
                QuantMatvecKind::Q80,
                weights,
                input,
                row_count,
                in_features,
                out_features,
            );
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weights, input, row_count, in_features, out_features);
            Ok(None)
        }
    }

    fn gguf_k_matvec_device(
        &self,
        quant: GgufKQuant,
        weights: &[u8],
        input: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        let (actual_rows, output_shape) =
            matvec_dims_shape(input.dims(), in_features, out_features)?;
        validate_exact_shape(
            "Laguna XS K-quant matvec rows",
            &[actual_rows],
            &[row_count],
        )?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let output = native_metal.batched_laguna_xs_k_matvec(
                quant,
                weights,
                &input.buffer,
                input.element_count()?,
                row_count,
                in_features,
                out_features,
            )?;
            return Ok(Some(DeviceValue::new(output_shape, output)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (quant, weights);
            Ok(None)
        }
    }

    fn prepare_laguna_xs_mps_prefill_weight(
        &self,
        quant: GgufKQuant,
        weights: &[u8],
        in_features: usize,
        out_features: usize,
    ) -> Result<bool> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(false);
            };
            native_metal.prepare_laguna_xs_mps_prefill_weight(
                quant,
                weights,
                in_features,
                out_features,
            )?;
            return Ok(true);
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (quant, weights, in_features, out_features);
            Ok(false)
        }
    }

    fn laguna_xs_k_matvec_add_device(
        &self,
        quant: GgufKQuant,
        weights: &[u8],
        input: &DeviceValue,
        residual: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        if !(row_count == 1 && out_features.is_multiple_of(2))
            && !(row_count >= 4 && out_features.is_multiple_of(32))
        {
            return Ok(None);
        }
        let (actual_rows, output_shape) =
            matvec_dims_shape(input.dims(), in_features, out_features)?;
        validate_exact_shape("Laguna XS fused matvec rows", &[actual_rows], &[row_count])?;
        validate_exact_shape(
            "Laguna XS fused matvec residual",
            residual.dims(),
            &output_shape,
        )?;
        if input.dtype() != DType::F32 || residual.dtype() != DType::F32 {
            return Err(Error::backend(format!(
                "Laguna XS fused matvec requires F32 input and residual, got {:?}/{:?}",
                input.dtype(),
                residual.dtype()
            )));
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let output = native_metal.batched_laguna_xs_k_matvec_residuals(
                quant,
                weights,
                &input.buffer,
                input.element_count()?,
                &residual.buffer,
                None,
                row_count,
                in_features,
                out_features,
            )?;
            return Ok(Some(DeviceValue::new(output_shape, output)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (quant, weights);
            Ok(None)
        }
    }

    fn laguna_xs_k_matvec_add2_device(
        &self,
        quant: GgufKQuant,
        weights: &[u8],
        input: &DeviceValue,
        residual_a: &DeviceValue,
        residual_b: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        if !(row_count == 1 && out_features.is_multiple_of(2))
            && !(row_count >= 4 && out_features.is_multiple_of(32))
        {
            return Ok(None);
        }
        let (actual_rows, output_shape) =
            matvec_dims_shape(input.dims(), in_features, out_features)?;
        validate_exact_shape("Laguna XS fused matvec rows", &[actual_rows], &[row_count])?;
        validate_exact_shape(
            "Laguna XS fused matvec first residual",
            residual_a.dims(),
            &output_shape,
        )?;
        validate_exact_shape(
            "Laguna XS fused matvec second residual",
            residual_b.dims(),
            &output_shape,
        )?;
        if [input.dtype(), residual_a.dtype(), residual_b.dtype()]
            .into_iter()
            .any(|dtype| dtype != DType::F32)
        {
            return Err(Error::backend(format!(
                "Laguna XS fused matvec requires F32 values, got input={:?}, residuals={:?}/{:?}",
                input.dtype(),
                residual_a.dtype(),
                residual_b.dtype()
            )));
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let output = native_metal.batched_laguna_xs_k_matvec_residuals(
                quant,
                weights,
                &input.buffer,
                input.element_count()?,
                &residual_a.buffer,
                Some(&residual_b.buffer),
                row_count,
                in_features,
                out_features,
            )?;
            return Ok(Some(DeviceValue::new(output_shape, output)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (quant, weights);
            Ok(None)
        }
    }

    fn q4_k_embedding_device(
        &self,
        weights: &[u8],
        token_ids: &[u32],
        token_shape: &[usize],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Option<DeviceValue>> {
        let expected_tokens = token_shape.iter().try_fold(1_usize, |count, dim| {
            count
                .checked_mul(*dim)
                .ok_or_else(|| Error::backend("Q4_K embedding token shape overflow"))
        })?;
        if expected_tokens != token_ids.len() {
            return Err(Error::backend(format!(
                "Q4_K embedding token shape {token_shape:?} contains {expected_tokens} IDs, got {}",
                token_ids.len()
            )));
        }
        let mut output_shape = token_shape.to_vec();
        output_shape.push(hidden_size);

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let output = native_metal.batched_laguna_xs_q4_embedding(
                weights,
                token_ids,
                vocab_size,
                hidden_size,
            )?;
            return Ok(Some(DeviceValue::new(output_shape, output)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weights, token_ids, vocab_size);
            Ok(None)
        }
    }

    fn q8_0_embedding_device(
        &self,
        weights: &[u8],
        token_ids: &[u32],
        token_shape: &[usize],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Option<DeviceValue>> {
        let expected_tokens = token_shape.iter().try_fold(1_usize, |count, dim| {
            count
                .checked_mul(*dim)
                .ok_or_else(|| Error::backend("Q8_0 embedding token shape overflow"))
        })?;
        if expected_tokens != token_ids.len() {
            return Err(Error::backend(format!(
                "Q8_0 embedding token shape {token_shape:?} contains {expected_tokens} IDs, got {}",
                token_ids.len()
            )));
        }
        let mut output_shape = token_shape.to_vec();
        output_shape.push(hidden_size);

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let output =
                native_metal.batched_q8_0_embedding(weights, token_ids, vocab_size, hidden_size)?;
            return Ok(Some(DeviceValue::new(output_shape, output)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weights, token_ids, vocab_size);
            Ok(None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn q8_0_matvec_pair_device(
        &self,
        weights_a: &[u8],
        weights_b: &[u8],
        input: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features_a: usize,
        out_features_b: usize,
    ) -> Result<Option<(DeviceValue, DeviceValue)>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let (actual_rows_a, output_shape_a) =
                matvec_dims_shape(input.dims(), in_features, out_features_a)?;
            let (actual_rows_b, output_shape_b) =
                matvec_dims_shape(input.dims(), in_features, out_features_b)?;
            validate_exact_shape(
                "device_paired_q8_matvec_rows_a",
                &[actual_rows_a],
                &[row_count],
            )?;
            validate_exact_shape(
                "device_paired_q8_matvec_rows_b",
                &[actual_rows_b],
                &[row_count],
            )?;
            let (buffer_a, buffer_b) = native_metal.batched_q8_0_matvec_pair(
                weights_a,
                weights_b,
                &input.buffer,
                input.element_count()?,
                row_count,
                in_features,
                out_features_a,
                out_features_b,
            )?;
            Ok(Some((
                DeviceValue::new(output_shape_a, buffer_a),
                DeviceValue::new(output_shape_b, buffer_b),
            )))
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                weights_a,
                weights_b,
                input,
                row_count,
                in_features,
                out_features_a,
                out_features_b,
            );
            Ok(None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn laguna_q8_0_attention_projections_device(
        &self,
        query_weights: &[u8],
        key_weights: &[u8],
        value_weights: &[u8],
        gate_weights: &[u8],
        input: &DeviceValue,
        row_count: usize,
        in_features: usize,
        query_features: usize,
        key_features: usize,
        value_features: usize,
        gate_features: usize,
    ) -> Result<Option<[DeviceValue; 4]>> {
        if row_count != 1
            || [query_features, key_features, value_features, gate_features]
                .into_iter()
                .any(|features| !features.is_multiple_of(2))
        {
            return Ok(None);
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let (query_rows, query_shape) =
                matvec_dims_shape(input.dims(), in_features, query_features)?;
            let (key_rows, key_shape) = matvec_dims_shape(input.dims(), in_features, key_features)?;
            let (value_rows, value_shape) =
                matvec_dims_shape(input.dims(), in_features, value_features)?;
            let (gate_rows, gate_shape) =
                matvec_dims_shape(input.dims(), in_features, gate_features)?;
            for (component, actual_rows) in [
                ("query", query_rows),
                ("key", key_rows),
                ("value", value_rows),
                ("gate", gate_rows),
            ] {
                validate_exact_shape(
                    &format!("device_laguna_{component}_projection_rows"),
                    &[actual_rows],
                    &[row_count],
                )?;
            }
            let [query, key, value, gate] = native_metal
                .batched_laguna_q8_0_attention_projections(
                    query_weights,
                    key_weights,
                    value_weights,
                    gate_weights,
                    &input.buffer,
                    input.element_count()?,
                    row_count,
                    in_features,
                    query_features,
                    key_features,
                    value_features,
                    gate_features,
                )?;
            Ok(Some([
                DeviceValue::new(query_shape, query),
                DeviceValue::new(key_shape, key),
                DeviceValue::new(value_shape, value),
                DeviceValue::new(gate_shape, gate),
            ]))
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                query_weights,
                key_weights,
                value_weights,
                gate_weights,
                input,
                in_features,
                query_features,
                key_features,
                value_features,
                gate_features,
            );
            Ok(None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn laguna_xs_attention_projections_device(
        &self,
        query_weights: &[u8],
        key_weights: &[u8],
        value_weights: &[u8],
        value_quant: GgufKQuant,
        gate_weights: &[u8],
        input: &DeviceValue,
        row_count: usize,
        in_features: usize,
        query_features: usize,
        key_features: usize,
        value_features: usize,
        gate_features: usize,
    ) -> Result<Option<[DeviceValue; 4]>> {
        if row_count != 1 {
            return Ok(None);
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let (query_rows, query_shape) =
                matvec_dims_shape(input.dims(), in_features, query_features)?;
            let (key_rows, key_shape) = matvec_dims_shape(input.dims(), in_features, key_features)?;
            let (value_rows, value_shape) =
                matvec_dims_shape(input.dims(), in_features, value_features)?;
            let (gate_rows, gate_shape) =
                matvec_dims_shape(input.dims(), in_features, gate_features)?;
            for (component, actual_rows) in [
                ("query", query_rows),
                ("key", key_rows),
                ("value", value_rows),
                ("gate", gate_rows),
            ] {
                validate_exact_shape(
                    &format!("device_laguna_xs_{component}_projection_rows"),
                    &[actual_rows],
                    &[row_count],
                )?;
            }
            let [query, key, value, gate] = native_metal.batched_laguna_xs_attention_projections(
                query_weights,
                key_weights,
                value_weights,
                value_quant,
                gate_weights,
                &input.buffer,
                input.element_count()?,
                in_features,
                query_features,
                key_features,
                value_features,
                gate_features,
            )?;
            Ok(Some([
                DeviceValue::new(query_shape, query),
                DeviceValue::new(key_shape, key),
                DeviceValue::new(value_shape, value),
                DeviceValue::new(gate_shape, gate),
            ]))
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                query_weights,
                key_weights,
                value_weights,
                value_quant,
                gate_weights,
                input,
                in_features,
                query_features,
                key_features,
                value_features,
                gate_features,
            );
            Ok(None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn q8_0_gate_up_swiglu_device(
        &self,
        gate_weights: &[u8],
        up_weights: &[u8],
        input: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let (actual_rows, output_shape) =
                matvec_dims_shape(input.dims(), in_features, out_features)?;
            validate_exact_shape("device_fused_q8_gate_up_rows", &[actual_rows], &[row_count])?;
            let output = native_metal.batched_q8_0_gate_up_swiglu(
                gate_weights,
                up_weights,
                &input.buffer,
                input.element_count()?,
                row_count,
                in_features,
                out_features,
            )?;
            Ok(Some(DeviceValue::new(output_shape, output)))
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                gate_weights,
                up_weights,
                input,
                row_count,
                in_features,
                out_features,
            );
            Ok(None)
        }
    }

    fn laguna_xs_q4_gate_up_swiglu_device(
        &self,
        gate_weights: &[u8],
        up_weights: &[u8],
        input: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        if row_count != 1 {
            return Ok(None);
        }
        let (actual_rows, output_shape) =
            matvec_dims_shape(input.dims(), in_features, out_features)?;
        validate_exact_shape("Laguna XS fused gate/up rows", &[actual_rows], &[row_count])?;
        if input.dtype() != DType::F32 {
            return Err(Error::backend(format!(
                "Laguna XS fused gate/up requires F32 input, got {:?}",
                input.dtype()
            )));
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let output = native_metal.batched_laguna_xs_q4_gate_up_swiglu(
                gate_weights,
                up_weights,
                &input.buffer,
                input.element_count()?,
                row_count,
                in_features,
                out_features,
            )?;
            return Ok(Some(DeviceValue::new(output_shape, output)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (gate_weights, up_weights);
            Ok(None)
        }
    }

    fn laguna_xs_router_topk_device(
        &self,
        router_logits: &DeviceValue,
        correction_bias: &[f32],
        top_k: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
    ) -> Result<Option<DeviceRouterTopK>> {
        let dims = require_device_rank("Laguna XS router logits", router_logits, 2)?;
        let (token_count, expert_count) = (dims[0], dims[1]);
        validate_exact_shape(
            "Laguna XS router correction bias",
            &[correction_bias.len()],
            &[expert_count],
        )?;
        if router_logits.dtype() != DType::F32 {
            return Err(Error::backend(format!(
                "Laguna XS router logits must be F32, got {:?}",
                router_logits.dtype()
            )));
        }
        if token_count != 1 || expert_count != 256 || top_k != 8 || !norm_topk_prob {
            return Ok(None);
        }

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            return native_metal
                .batched_laguna_xs_router_topk(
                    &router_logits.buffer,
                    router_logits.element_count()?,
                    correction_bias,
                    routed_scaling_factor,
                )
                .map(Some);
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = routed_scaling_factor;
            Ok(None)
        }
    }

    fn q8_0_transposed_matvec_device(
        &self,
        weights: &[u8],
        input: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            return self.quant_matvec_device(
                QuantMatvecKind::Q80Transposed,
                weights,
                input,
                row_count,
                in_features,
                out_features,
            );
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weights, input, row_count, in_features, out_features);
            Ok(None)
        }
    }

    fn q2_k_packed_heads_transposed_matvec_device(
        &self,
        weights: &[u8],
        input: &DeviceValue,
        row_count: usize,
        head_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            return self.packed_heads_transposed_matvec_device(
                QuantMatvecKind::Q2KTransposed,
                weights,
                input,
                row_count,
                head_count,
                in_features,
                out_features,
            );
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                weights,
                input,
                row_count,
                head_count,
                in_features,
                out_features,
            );
            Ok(None)
        }
    }

    fn q8_0_packed_heads_transposed_matvec_device(
        &self,
        weights: &[u8],
        input: &DeviceValue,
        row_count: usize,
        head_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            return self.packed_heads_transposed_matvec_device(
                QuantMatvecKind::Q80Transposed,
                weights,
                input,
                row_count,
                head_count,
                in_features,
                out_features,
            );
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                weights,
                input,
                row_count,
                head_count,
                in_features,
                out_features,
            );
            Ok(None)
        }
    }

    fn q2_k_matvec_add_device(
        &self,
        weights: &[u8],
        input: &DeviceValue,
        residual: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let (actual_rows, _) = matvec_dims_shape(input.dims(), in_features, out_features)?;
            validate_exact_shape("device_q2_k_matvec_add_rows", &[actual_rows], &[row_count])?;
            let buffer = native_metal.batched_q2_k_matvec_add(
                weights,
                &input.buffer,
                input.element_count()?,
                &residual.buffer,
                residual.element_count()?,
                row_count,
                in_features,
                out_features,
            )?;
            return Ok(Some(DeviceValue::new(residual.dims().to_vec(), buffer)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                weights,
                input,
                residual,
                row_count,
                in_features,
                out_features,
            );
            Ok(None)
        }
    }

    fn q8_0_matvec_add_device(
        &self,
        weights: &[u8],
        input: &DeviceValue,
        residual: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let (actual_rows, _) = matvec_dims_shape(input.dims(), in_features, out_features)?;
            validate_exact_shape("device_q8_0_matvec_add_rows", &[actual_rows], &[row_count])?;
            let buffer = native_metal.batched_q8_0_matvec_add(
                weights,
                &input.buffer,
                input.element_count()?,
                &residual.buffer,
                residual.element_count()?,
                row_count,
                in_features,
                out_features,
            )?;
            return Ok(Some(DeviceValue::new(residual.dims().to_vec(), buffer)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                weights,
                input,
                residual,
                row_count,
                in_features,
                out_features,
            );
            Ok(None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn laguna_q8_0_matvec_add2_device(
        &self,
        weights: &[u8],
        input: &DeviceValue,
        residual_a: &DeviceValue,
        residual_b: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let (actual_rows, _) = matvec_dims_shape(input.dims(), in_features, out_features)?;
            validate_exact_shape(
                "device_laguna_q8_0_matvec_add2_rows",
                &[actual_rows],
                &[row_count],
            )?;
            let buffer = native_metal.batched_laguna_q8_0_matvec_add2(
                weights,
                &input.buffer,
                input.element_count()?,
                &residual_a.buffer,
                residual_a.element_count()?,
                &residual_b.buffer,
                residual_b.element_count()?,
                row_count,
                in_features,
                out_features,
            )?;
            return Ok(Some(DeviceValue::new(residual_b.dims().to_vec(), buffer)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                weights,
                input,
                residual_a,
                residual_b,
                row_count,
                in_features,
                out_features,
            );
            Ok(None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn q2_k_multi_expert_gate_up_swiglu_device(
        &self,
        gate_weights: &[u8],
        up_weights: &[u8],
        input: &DeviceValue,
        token_indices: &[u32],
        expert_ids: &[u32],
        token_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let dims = require_device_rank("device_q2_multi_expert_gate_up_input", input, 2)?;
            validate_exact_shape(
                "device_q2_multi_expert_gate_up_input_shape",
                dims,
                &[token_count, in_features],
            )?;
            if token_indices.len() != expert_ids.len() {
                return Err(Error::backend(format!(
                    "device Q2 multi-expert gate/up routing mismatch: {} token indices and {} expert ids",
                    token_indices.len(),
                    expert_ids.len()
                )));
            }
            let assignment_count = token_indices.len();
            let buffer = native_metal.batched_q2_k_multi_expert_gate_up_swiglu(
                gate_weights,
                up_weights,
                &input.buffer,
                input.element_count()?,
                token_indices,
                expert_ids,
                token_count,
                in_features,
                out_features,
            )?;
            return Ok(Some(DeviceValue::new(
                vec![assignment_count, out_features],
                buffer,
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                gate_weights,
                up_weights,
                input,
                token_indices,
                expert_ids,
                token_count,
                in_features,
                out_features,
            );
            Ok(None)
        }
    }

    fn q2_k_multi_expert_matvec_device(
        &self,
        weights: &[u8],
        input: &DeviceValue,
        expert_ids: &[u32],
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let dims = require_device_rank("device_q2_multi_expert_matvec_input", input, 2)?;
            let assignment_count = expert_ids.len();
            validate_exact_shape(
                "device_q2_multi_expert_matvec_input_shape",
                dims,
                &[assignment_count, in_features],
            )?;
            let buffer = native_metal.batched_q2_k_multi_expert_matvec(
                weights,
                &input.buffer,
                input.element_count()?,
                expert_ids,
                in_features,
                out_features,
            )?;
            return Ok(Some(DeviceValue::new(
                vec![assignment_count, out_features],
                buffer,
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weights, input, expert_ids, in_features, out_features);
            Ok(None)
        }
    }

    fn laguna_gguf_moe_device(
        &self,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        quant: GgufExpertQuant,
        input: &DeviceValue,
        routing: &DeviceRouterTopK,
        in_features: usize,
        intermediate_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        if input.dtype() != DType::F32 {
            return Err(Error::backend(format!(
                "Laguna GGUF MoE input must be F32, got {:?}",
                input.dtype()
            )));
        }
        let (row_count, output_shape) = matvec_dims_shape(input.dims(), in_features, out_features)?;
        validate_exact_shape(
            "Laguna GGUF MoE routing token count",
            &[routing.token_count],
            &[row_count],
        )?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let output = native_metal.batched_laguna_gguf_moe(
                gate_weights,
                up_weights,
                down_weights,
                quant,
                &input.buffer,
                input.element_count()?,
                routing,
                in_features,
                intermediate_features,
                out_features,
            )?;
            return Ok(Some(DeviceValue::new(output_shape, output)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                gate_weights,
                up_weights,
                down_weights,
                quant,
                routing,
                intermediate_features,
            );
            Ok(None)
        }
    }

    fn laguna_xs_gguf_moe_device(
        &self,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: GgufKQuant,
        input: &DeviceValue,
        routing: &DeviceRouterTopK,
        in_features: usize,
        intermediate_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        if input.dtype() != DType::F32 {
            return Err(Error::backend(format!(
                "Laguna XS GGUF MoE input must be F32, got {:?}",
                input.dtype()
            )));
        }
        let (row_count, output_shape) = matvec_dims_shape(input.dims(), in_features, out_features)?;
        validate_exact_shape(
            "Laguna XS GGUF MoE routing token count",
            &[routing.token_count],
            &[row_count],
        )?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let output = native_metal.batched_laguna_xs_gguf_moe(
                gate_weights,
                up_weights,
                down_weights,
                down_quant,
                &input.buffer,
                input.element_count()?,
                routing,
                in_features,
                intermediate_features,
                out_features,
            )?;
            return Ok(Some(DeviceValue::new(output_shape, output)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                gate_weights,
                up_weights,
                down_weights,
                down_quant,
                routing,
                intermediate_features,
            );
            Ok(None)
        }
    }

    fn ready_routed_experts_device(
        &self,
        layer_index: usize,
        model_path: &Path,
        gate_payloads: &[Q2ExpertSource<'_>],
        up_payloads: &[Q2ExpertSource<'_>],
        down_payloads: &[Q2ExpertSource<'_>],
        input: &DeviceValue,
        routing: &DeviceRouterTopK,
        in_features: usize,
        intermediate_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceRoutedExperts>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            validate_exact_shape(
                "ready_routed_experts_input",
                input.dims(),
                &[routing.token_count, in_features],
            )?;
            let result = native_metal.ready_routed_experts(
                layer_index,
                model_path,
                gate_payloads,
                up_payloads,
                down_payloads,
                &input.buffer,
                input.element_count()?,
                routing,
                in_features,
                intermediate_features,
                out_features,
            )?;
            return Ok(Some(DeviceRoutedExperts {
                output: DeviceValue::new(
                    vec![routing.assignment_count()?, out_features],
                    result.output,
                ),
                completion_value: result.completion_value,
                selected_experts: result.selected_experts,
                cache_hits: result.cache_hits,
                cache_misses: result.cache_misses,
                transient_experts: result.transient_experts,
                read_bytes: result.read_bytes,
                ready_waves: result.ready_waves,
            }));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                layer_index,
                model_path,
                gate_payloads,
                up_payloads,
                down_payloads,
                input,
                routing,
                in_features,
                intermediate_features,
                out_features,
            );
            Ok(None)
        }
    }

    fn prefetch_routed_experts_device(
        &self,
        layer_index: usize,
        model_path: &Path,
        gate_payloads: &[Q2ExpertSource<'_>],
        up_payloads: &[Q2ExpertSource<'_>],
        down_payloads: &[Q2ExpertSource<'_>],
    ) -> Result<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(());
            };
            return native_metal.prefetch_routed_experts(
                layer_index,
                model_path,
                gate_payloads,
                up_payloads,
                down_payloads,
            );
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                layer_index,
                model_path,
                gate_payloads,
                up_payloads,
                down_payloads,
            );
            Ok(())
        }
    }

    fn wait_for_routed_experts_device(&self, routed: &DeviceRoutedExperts) -> Result<()> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(());
            };
            return native_metal.batched_wait_for_ready_routed_experts(routed.completion_value);
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = routed;
            Ok(())
        }
    }

    fn q2_k_matvec_argmax_device(
        &self,
        weights: &[u8],
        input: &DeviceValue,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<(u32, f32)>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let (actual_rows, _) = matvec_dims_shape(input.dims(), in_features, out_features)?;
            validate_exact_shape("device_q2_k_matvec_argmax_rows", &[actual_rows], &[1])?;
            let (token_id, token_score) = native_metal.batched_q2_k_matvec_argmax(
                weights,
                &input.buffer,
                input.element_count()?,
                in_features,
                out_features,
            )?;
            return Ok(Some((token_id, token_score)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weights, input, in_features, out_features);
            Ok(None)
        }
    }

    fn laguna_q8_0_matvec_argmax_device(
        &self,
        weights: &[u8],
        input: &DeviceValue,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<(u32, f32)>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let (actual_rows, _) = matvec_dims_shape(input.dims(), in_features, out_features)?;
            validate_exact_shape(
                "device_laguna_q8_0_matvec_argmax_rows",
                &[actual_rows],
                &[1],
            )?;
            let result = native_metal.batched_laguna_q8_0_matvec_argmax(
                weights,
                &input.buffer,
                input.element_count()?,
                in_features,
                out_features,
            )?;
            return Ok(Some(result));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (weights, input, in_features, out_features);
            Ok(None)
        }
    }

    fn argmax_f32_device(&self, scores: &DeviceValue) -> Result<Option<(u32, f32)>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            validate_exact_shape("device_argmax_f32_rank", &[scores.dims().len()], &[1])?;
            let value_count = scores.element_count()?;
            let (token_id, token_score) =
                native_metal.batched_f32_argmax(&scores.buffer, value_count)?;
            return Ok(Some((token_id, token_score)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = scores;
            Ok(None)
        }
    }

    fn argmax_rows_f32_device(
        &self,
        scores: &DeviceValue,
        row_width: usize,
    ) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            validate_exact_shape("device_argmax_rows_f32_rank", &[scores.dims().len()], &[2])?;
            validate_exact_shape(
                "device_argmax_rows_f32_width",
                &[scores.dims()[1]],
                &[row_width],
            )?;
            let row_count = scores.dims()[0];
            let result =
                native_metal.batched_f32_argmax_rows(&scores.buffer, row_count, row_width)?;
            return Ok(Some(result));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (scores, row_width);
            Ok(None)
        }
    }

    fn rope_slice_device(
        &self,
        input: &DeviceValue,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let dims = require_device_rank("device_rope_slice", input, 4)?;
            validate_exact_shape("device_rope_slice_dim", &[dims[3]], &[rope_dim])?;
            let buffer = native_metal.batched_rope_slice(
                &input.buffer,
                input.element_count()?,
                dims[0],
                dims[1],
                dims[2],
                rope_dim,
                position_offset,
                theta,
            )?;
            return Ok(Some(DeviceValue::new(dims.to_vec(), buffer)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (input, rope_dim, position_offset, theta);
            Ok(None)
        }
    }

    fn split_rope_tail_device(
        &self,
        heads: &DeviceValue,
        no_rope_dim: usize,
        rope_dim: usize,
    ) -> Result<Option<(DeviceValue, DeviceValue)>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let dims = require_device_rank("device_split_rope_tail", heads, 4)?;
            let total_dim = no_rope_dim
                .checked_add(rope_dim)
                .ok_or_else(|| Error::backend("device split_rope_tail total dim overflow"))?;
            validate_exact_shape("device_split_rope_tail_last_dim", &[dims[3]], &[total_dim])?;
            let (batch, tokens, head_count) = (dims[0], dims[1], dims[2]);
            let (no_rope_buffer, rope_buffer) = native_metal.batched_split_rope_tail(
                &heads.buffer,
                heads.element_count()?,
                batch,
                tokens,
                head_count,
                no_rope_dim,
                rope_dim,
            )?;
            return Ok(Some((
                DeviceValue::new(vec![batch, tokens, head_count, no_rope_dim], no_rope_buffer),
                DeviceValue::new(vec![batch, tokens, head_count, rope_dim], rope_buffer),
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (heads, no_rope_dim, rope_dim);
            Ok(None)
        }
    }

    fn split_kv_mqa_device(
        &self,
        kv_mqa: &DeviceValue,
        kv_lora_rank: usize,
        rope_dim: usize,
    ) -> Result<Option<(DeviceValue, DeviceValue)>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let dims = require_device_rank("device_split_kv_mqa", kv_mqa, 3)?;
            let total_dim = kv_lora_rank
                .checked_add(rope_dim)
                .ok_or_else(|| Error::backend("device split_kv_mqa total dim overflow"))?;
            validate_exact_shape("device_split_kv_mqa_last_dim", &[dims[2]], &[total_dim])?;
            let (batch, tokens) = (dims[0], dims[1]);
            let (latent_buffer, rope_buffer) = native_metal.batched_split_kv_mqa(
                &kv_mqa.buffer,
                kv_mqa.element_count()?,
                batch,
                tokens,
                kv_lora_rank,
                rope_dim,
            )?;
            return Ok(Some((
                DeviceValue::new(vec![batch, tokens, kv_lora_rank], latent_buffer),
                DeviceValue::new(vec![batch, tokens, 1, rope_dim], rope_buffer),
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (kv_mqa, kv_lora_rank, rope_dim);
            Ok(None)
        }
    }

    fn mla_kv_postprocess_device(
        &self,
        kv_mqa: &DeviceValue,
        norm_weight: &F32Tensor,
        norm_eps: f32,
        kv_lora_rank: usize,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
    ) -> Result<Option<(DeviceValue, DeviceValue)>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let dims = require_device_rank("device_mla_kv_postprocess", kv_mqa, 3)?;
            let total_dim = kv_lora_rank
                .checked_add(rope_dim)
                .ok_or_else(|| Error::backend("device MLA KV total dim overflow"))?;
            validate_exact_shape(
                "device_mla_kv_postprocess_input",
                dims,
                &[dims[0], dims[1], total_dim],
            )?;
            validate_exact_shape(
                "device_mla_kv_postprocess_norm_weight",
                norm_weight.dims(),
                &[kv_lora_rank],
            )?;
            let (latent, rope) = native_metal.batched_mla_kv_postprocess(
                &kv_mqa.buffer,
                kv_mqa.element_count()?,
                norm_weight.values(),
                dims[0],
                dims[1],
                kv_lora_rank,
                rope_dim,
                position_offset,
                theta,
                norm_eps,
            )?;
            return Ok(Some((
                DeviceValue::new(vec![dims[0], dims[1], kv_lora_rank], latent),
                DeviceValue::new(vec![dims[0], dims[1], 1, rope_dim], rope),
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                kv_mqa,
                norm_weight,
                norm_eps,
                kv_lora_rank,
                rope_dim,
                position_offset,
                theta,
            );
            Ok(None)
        }
    }

    fn combine_rope_tail_device(
        &self,
        no_rope: &DeviceValue,
        rope: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let no_rope_dims = require_device_rank("device_combine_rope_tail_no_rope", no_rope, 4)?;
            let rope_dims = require_device_rank("device_combine_rope_tail_rope", rope, 4)?;
            let (batch, tokens, head_count, no_rope_dim) = (
                no_rope_dims[0],
                no_rope_dims[1],
                no_rope_dims[2],
                no_rope_dims[3],
            );
            let (rope_head_count, rope_dim) = (rope_dims[2], rope_dims[3]);
            validate_exact_shape(
                "device_combine_rope_tail_batch_tokens",
                &[rope_dims[0], rope_dims[1]],
                &[batch, tokens],
            )?;
            let total_dim = no_rope_dim
                .checked_add(rope_dim)
                .ok_or_else(|| Error::backend("device combine_rope_tail total dim overflow"))?;
            let buffer = native_metal.batched_combine_rope_tail(
                &no_rope.buffer,
                no_rope.element_count()?,
                &rope.buffer,
                rope.element_count()?,
                batch,
                tokens,
                head_count,
                rope_head_count,
                no_rope_dim,
                rope_dim,
            )?;
            return Ok(Some(DeviceValue::new(
                vec![batch, tokens, head_count, total_dim],
                buffer,
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (no_rope, rope);
            Ok(None)
        }
    }

    fn heads_to_attention_layout_device(&self, heads: &DeviceValue) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let dims = require_device_rank("device_heads_to_attention_layout", heads, 4)?;
            let (batch, tokens, head_count, head_dim) = (dims[0], dims[1], dims[2], dims[3]);
            let buffer = native_metal.batched_heads_to_attention_layout(
                &heads.buffer,
                heads.element_count()?,
                batch,
                tokens,
                head_count,
                head_dim,
            )?;
            return Ok(Some(DeviceValue::new(
                vec![batch, head_count, tokens, head_dim],
                buffer,
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = heads;
            Ok(None)
        }
    }

    fn merge_attention_heads_device(
        &self,
        context_heads: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let dims = require_device_rank("device_merge_attention_heads", context_heads, 4)?;
            let (batch, head_count, tokens, head_dim) = (dims[0], dims[1], dims[2], dims[3]);
            let merged_width = head_count
                .checked_mul(head_dim)
                .ok_or_else(|| Error::backend("device merge_attention_heads width overflow"))?;
            let buffer = native_metal.batched_merge_attention_heads(
                &context_heads.buffer,
                context_heads.element_count()?,
                batch,
                head_count,
                tokens,
                head_dim,
            )?;
            return Ok(Some(DeviceValue::new(
                vec![batch, tokens, merged_width],
                buffer,
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = context_heads;
            Ok(None)
        }
    }

    fn stack_head_outputs_device(
        &self,
        head_outputs: &[DeviceValue],
        row_count: usize,
        head_dim: usize,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            if head_outputs.is_empty() {
                return Err(Error::backend(
                    "device stack_head_outputs requires at least one head",
                ));
            }
            let expected_dims = [row_count, head_dim];
            for head_output in head_outputs {
                validate_exact_shape(
                    "device_stack_head_outputs_head",
                    head_output.dims(),
                    &expected_dims,
                )?;
            }
            let buffers = head_outputs
                .iter()
                .map(|output| &output.buffer)
                .collect::<Vec<_>>();
            let buffer = native_metal.batched_stack_head_outputs(&buffers, row_count, head_dim)?;
            return Ok(Some(DeviceValue::new(
                vec![row_count, head_outputs.len(), head_dim],
                buffer,
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (head_outputs, row_count, head_dim);
            Ok(None)
        }
    }

    fn select_last_token_device(&self, hidden_states: &DeviceValue) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let dims = require_device_rank("device_select_last_token", hidden_states, 3)?;
            let (batch, tokens, hidden_size) = (dims[0], dims[1], dims[2]);
            let buffer = native_metal.batched_select_last_token(
                &hidden_states.buffer,
                hidden_states.element_count()?,
                batch,
                tokens,
                hidden_size,
            )?;
            return Ok(Some(DeviceValue::new(vec![batch, 1, hidden_size], buffer)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = hidden_states;
            Ok(None)
        }
    }

    fn add_device(&self, lhs: &DeviceValue, rhs: &DeviceValue) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            validate_exact_shape("device_add_shapes", lhs.dims(), rhs.dims())?;
            let buffer = native_metal.batched_add(
                &lhs.buffer,
                lhs.element_count()?,
                &rhs.buffer,
                rhs.element_count()?,
            )?;
            return Ok(Some(DeviceValue::new(lhs.dims().to_vec(), buffer)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (lhs, rhs);
            Ok(None)
        }
    }

    fn swiglu_device(&self, gate: &DeviceValue, up: &DeviceValue) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            validate_exact_shape("device_swiglu_shapes", gate.dims(), up.dims())?;
            let buffer = native_metal.batched_swiglu(
                &gate.buffer,
                gate.element_count()?,
                &up.buffer,
                up.element_count()?,
            )?;
            return Ok(Some(DeviceValue::new(gate.dims().to_vec(), buffer)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (gate, up);
            Ok(None)
        }
    }

    fn paged_decode_attention_device(
        &self,
        q: &DeviceValue,
        current_k: &DeviceValue,
        current_v: &DeviceValue,
        past_kv: &PagedKvView<'_>,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            past_kv.validate()?;
            let q_dims = require_device_rank("device_paged_decode_attention_q", q, 4)?;
            let k_dims =
                require_device_rank("device_paged_decode_attention_current_k", current_k, 4)?;
            let v_dims =
                require_device_rank("device_paged_decode_attention_current_v", current_v, 4)?;
            let expected_qk = [
                past_kv.batch,
                past_kv.attention_heads,
                1,
                past_kv.key_head_dim,
            ];
            validate_exact_shape(
                "device_paged_decode_attention_q_shape",
                q_dims,
                &expected_qk,
            )?;
            validate_exact_shape(
                "device_paged_decode_attention_current_k_shape",
                k_dims,
                &expected_qk,
            )?;
            validate_exact_shape(
                "device_paged_decode_attention_current_v_shape",
                v_dims,
                &[
                    past_kv.batch,
                    past_kv.attention_heads,
                    1,
                    past_kv.value_head_dim,
                ],
            )?;
            let (buffer, _output_len) = native_metal.batched_paged_decode_attention(
                &q.buffer,
                q.element_count()?,
                &current_k.buffer,
                current_k.element_count()?,
                &current_v.buffer,
                current_v.element_count()?,
                past_kv,
            )?;
            return Ok(Some(DeviceValue::new(
                vec![
                    past_kv.batch,
                    past_kv.attention_heads,
                    1,
                    past_kv.value_head_dim,
                ],
                buffer,
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (q, current_k, current_v, past_kv);
            Ok(None)
        }
    }

    fn paged_decode_attention_resident_device(
        &self,
        q: &DeviceValue,
        current_k: &DeviceValue,
        current_v: &DeviceValue,
        past_kv: &DevicePagedKvView,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            past_kv.validate()?;
            let q_dims = require_device_rank("device_resident_paged_decode_attention_q", q, 4)?;
            let k_dims = require_device_rank(
                "device_resident_paged_decode_attention_current_k",
                current_k,
                4,
            )?;
            let v_dims = require_device_rank(
                "device_resident_paged_decode_attention_current_v",
                current_v,
                4,
            )?;
            let expected_qk = [
                past_kv.batch,
                past_kv.attention_heads,
                1,
                past_kv.key_head_dim,
            ];
            if q.dtype() != DType::F32
                || current_k.dtype() != DType::F32
                || current_v.dtype() != DType::F32
            {
                return Err(Error::backend(format!(
                    "device resident paged decode attention requires f32 q/current K/current V, got q={:?}, current_k={:?}, current_v={:?}",
                    q.dtype(),
                    current_k.dtype(),
                    current_v.dtype()
                )));
            }
            validate_exact_shape(
                "device_resident_paged_decode_attention_q_shape",
                q_dims,
                &expected_qk,
            )?;
            validate_exact_shape(
                "device_resident_paged_decode_attention_current_k_shape",
                k_dims,
                &expected_qk,
            )?;
            validate_exact_shape(
                "device_resident_paged_decode_attention_current_v_shape",
                v_dims,
                &[
                    past_kv.batch,
                    past_kv.attention_heads,
                    1,
                    past_kv.value_head_dim,
                ],
            )?;
            let (buffer, _output_len) = native_metal.batched_paged_decode_attention_resident(
                &q.buffer,
                q.element_count()?,
                &current_k.buffer,
                current_k.element_count()?,
                &current_v.buffer,
                current_v.element_count()?,
                past_kv,
            )?;
            return Ok(Some(DeviceValue::new(
                vec![
                    past_kv.batch,
                    past_kv.attention_heads,
                    1,
                    past_kv.value_head_dim,
                ],
                buffer,
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (q, current_k, current_v, past_kv);
            Ok(None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn q8_0_absorbed_mla_device(
        &self,
        k_b_weights: &[u8],
        v_b_weights: &[u8],
        q_no_rope: &DeviceValue,
        q_rope: &DeviceValue,
        current_latent: &DeviceValue,
        current_rope: &DeviceValue,
        past_kv: &DevicePagedKvView,
        qk_head_dim: usize,
        value_dim: usize,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            past_kv.validate()?;
            let q_no_dims = require_device_rank("device_absorbed_mla_q_no_rope", q_no_rope, 4)?;
            let q_rope_dims = require_device_rank("device_absorbed_mla_q_rope", q_rope, 4)?;
            let current_latent_dims =
                require_device_rank("device_absorbed_mla_current_latent", current_latent, 3)?;
            let current_rope_dims =
                require_device_rank("device_absorbed_mla_current_rope", current_rope, 4)?;
            if q_no_rope.dtype() != DType::F32
                || q_rope.dtype() != DType::F32
                || current_latent.dtype() != DType::F32
                || current_rope.dtype() != DType::F32
            {
                return Err(Error::backend(format!(
                    "absorbed MLA requires F32 query/current tensors, got q_no={:?}, q_rope={:?}, latent={:?}, current_rope={:?}",
                    q_no_rope.dtype(),
                    q_rope.dtype(),
                    current_latent.dtype(),
                    current_rope.dtype()
                )));
            }

            let batch = q_no_dims[0];
            let tokens = q_no_dims[1];
            let head_count = q_no_dims[2];
            let q_no_rope_dim = q_no_dims[3];
            let rope_dim = q_rope_dims[3];
            let latent_dim = current_latent_dims[2];
            validate_exact_shape(
                "device_absorbed_mla_q_rope_shape",
                q_rope_dims,
                &[batch, tokens, head_count, rope_dim],
            )?;
            validate_exact_shape(
                "device_absorbed_mla_current_latent_shape",
                current_latent_dims,
                &[batch, tokens, latent_dim],
            )?;
            validate_exact_shape(
                "device_absorbed_mla_current_rope_shape",
                current_rope_dims,
                &[batch, tokens, 1, rope_dim],
            )?;
            validate_exact_shape(
                "device_absorbed_mla_cache_shape",
                &[
                    past_kv.batch,
                    past_kv.attention_heads,
                    past_kv.key_head_dim,
                    past_kv.value_head_dim,
                ],
                &[batch, 1, latent_dim, rope_dim],
            )?;
            if qk_head_dim != q_no_rope_dim + rope_dim {
                return Err(Error::backend(format!(
                    "absorbed MLA qk_head_dim {qk_head_dim} must equal no-RoPE {q_no_rope_dim} + RoPE {rope_dim}"
                )));
            }

            let (buffer, _) = native_metal.batched_q8_0_absorbed_mla(
                k_b_weights,
                v_b_weights,
                &q_no_rope.buffer,
                q_no_rope.element_count()?,
                &q_rope.buffer,
                q_rope.element_count()?,
                &current_latent.buffer,
                current_latent.element_count()?,
                &current_rope.buffer,
                current_rope.element_count()?,
                past_kv,
                batch,
                tokens,
                head_count,
                q_no_rope_dim,
                rope_dim,
                latent_dim,
                value_dim,
                qk_head_dim,
            )?;
            return Ok(Some(DeviceValue::new(
                vec![batch, tokens, head_count, value_dim],
                buffer,
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                k_b_weights,
                v_b_weights,
                q_no_rope,
                q_rope,
                current_latent,
                current_rope,
                past_kv,
                qk_head_dim,
                value_dim,
            );
            Ok(None)
        }
    }

    fn selected_decode_attention_device(
        &self,
        q: &DeviceValue,
        selected_k: &DeviceValue,
        selected_v: &DeviceValue,
        current_k: &DeviceValue,
        current_v: &DeviceValue,
        include_current_kv: bool,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let q_dims = require_device_rank("device_selected_decode_attention_q", q, 4)?;
            let selected_k_dims =
                require_device_rank("device_selected_decode_attention_selected_k", selected_k, 4)?;
            let selected_v_dims =
                require_device_rank("device_selected_decode_attention_selected_v", selected_v, 4)?;
            let current_k_dims =
                require_device_rank("device_selected_decode_attention_current_k", current_k, 4)?;
            let current_v_dims =
                require_device_rank("device_selected_decode_attention_current_v", current_v, 4)?;

            let (batch, heads, query_tokens, head_dim) =
                (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
            if q.dtype() != DType::F32
                || current_k.dtype() != DType::F32
                || current_v.dtype() != DType::F32
            {
                return Err(Error::backend(format!(
                    "device selected decode attention requires f32 q/current K/current V, got q={:?}, current_k={:?}, current_v={:?}",
                    q.dtype(),
                    current_k.dtype(),
                    current_v.dtype()
                )));
            }
            if selected_k.dtype() != selected_v.dtype() {
                return Err(Error::backend(format!(
                    "device selected decode attention selected KV dtype mismatch: k={:?}, v={:?}",
                    selected_k.dtype(),
                    selected_v.dtype()
                )));
            }
            validate_exact_shape(
                "device_selected_decode_attention_query_tokens",
                &[query_tokens],
                &[1],
            )?;
            let selected_tokens = selected_k_dims[2];
            if selected_tokens == 0 {
                return Err(Error::backend(
                    "device selected decode attention requires at least one selected token",
                ));
            }
            validate_exact_shape(
                "device_selected_decode_attention_selected_k_shape",
                selected_k_dims,
                &[batch, heads, selected_tokens, head_dim],
            )?;
            let value_dim = selected_v_dims[3];
            validate_exact_shape(
                "device_selected_decode_attention_selected_v_shape",
                selected_v_dims,
                &[batch, heads, selected_tokens, value_dim],
            )?;
            validate_exact_shape(
                "device_selected_decode_attention_current_k_shape",
                current_k_dims,
                &[batch, heads, 1, head_dim],
            )?;
            validate_exact_shape(
                "device_selected_decode_attention_current_v_shape",
                current_v_dims,
                &[batch, heads, 1, value_dim],
            )?;

            let (buffer, _output_len) = native_metal.batched_selected_decode_attention(
                &q.buffer,
                q.element_count()?,
                &selected_k.buffer,
                selected_k.element_count()?,
                &selected_v.buffer,
                selected_v.element_count()?,
                selected_k.dtype(),
                &current_k.buffer,
                current_k.element_count()?,
                &current_v.buffer,
                current_v.element_count()?,
                batch,
                heads,
                selected_tokens,
                head_dim,
                value_dim,
                include_current_kv,
            )?;
            return Ok(Some(DeviceValue::new(
                vec![batch, heads, 1, value_dim],
                buffer,
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                q,
                selected_k,
                selected_v,
                current_k,
                current_v,
                include_current_kv,
            );
            Ok(None)
        }
    }

    fn selected_sequence_attention_device(
        &self,
        q: &DeviceValue,
        past_k: &DeviceValue,
        past_v: &DeviceValue,
        current_k: &DeviceValue,
        current_v: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let q_dims = require_device_rank("device_selected_sequence_q", q, 4)?;
            let past_k_dims = require_device_rank("device_selected_sequence_past_k", past_k, 4)?;
            let past_v_dims = require_device_rank("device_selected_sequence_past_v", past_v, 4)?;
            let current_k_dims =
                require_device_rank("device_selected_sequence_current_k", current_k, 4)?;
            let current_v_dims =
                require_device_rank("device_selected_sequence_current_v", current_v, 4)?;
            if [q, past_k, past_v, current_k, current_v]
                .iter()
                .any(|value| value.dtype() != DType::F32)
            {
                return Err(Error::backend(
                    "selected sequence attention currently requires f32 tensors",
                ));
            }
            let (batch, heads, query_tokens, head_dim) =
                (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
            let past_tokens = past_k_dims[2];
            let value_dim = past_v_dims[3];
            validate_exact_shape(
                "device_selected_sequence_past_k_shape",
                past_k_dims,
                &[batch, heads, past_tokens, head_dim],
            )?;
            validate_exact_shape(
                "device_selected_sequence_past_v_shape",
                past_v_dims,
                &[batch, heads, past_tokens, value_dim],
            )?;
            validate_exact_shape(
                "device_selected_sequence_current_k_shape",
                current_k_dims,
                &[batch, heads, query_tokens, head_dim],
            )?;
            validate_exact_shape(
                "device_selected_sequence_current_v_shape",
                current_v_dims,
                &[batch, heads, query_tokens, value_dim],
            )?;
            let (buffer, _) = native_metal.batched_selected_sequence_attention(
                &q.buffer,
                q.element_count()?,
                &past_k.buffer,
                past_k.element_count()?,
                &past_v.buffer,
                past_v.element_count()?,
                &current_k.buffer,
                current_k.element_count()?,
                &current_v.buffer,
                current_v.element_count()?,
                batch,
                heads,
                past_tokens,
                query_tokens,
                head_dim,
                value_dim,
            )?;
            return Ok(Some(DeviceValue::new(
                vec![batch, heads, query_tokens, value_dim],
                buffer,
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (q, past_k, past_v, current_k, current_v);
            Ok(None)
        }
    }

    fn causal_sequence_attention_device(
        &self,
        q: &DeviceValue,
        current_k: &DeviceValue,
        current_v: &DeviceValue,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let q_dims = require_device_rank("device_causal_sequence_q", q, 4)?;
            let current_k_dims =
                require_device_rank("device_causal_sequence_current_k", current_k, 4)?;
            let current_v_dims =
                require_device_rank("device_causal_sequence_current_v", current_v, 4)?;
            if [q, current_k, current_v]
                .iter()
                .any(|value| value.dtype() != DType::F32)
            {
                return Err(Error::backend(
                    "causal sequence attention currently requires f32 tensors",
                ));
            }
            let (batch, heads, query_tokens, head_dim) =
                (q_dims[0], q_dims[1], q_dims[2], q_dims[3]);
            let value_dim = current_v_dims[3];
            validate_exact_shape(
                "device_causal_sequence_current_k_shape",
                current_k_dims,
                &[batch, heads, query_tokens, head_dim],
            )?;
            validate_exact_shape(
                "device_causal_sequence_current_v_shape",
                current_v_dims,
                &[batch, heads, query_tokens, value_dim],
            )?;
            let (buffer, _) = native_metal.batched_selected_sequence_attention(
                &q.buffer,
                q.element_count()?,
                &current_k.buffer,
                0,
                &current_v.buffer,
                0,
                &current_k.buffer,
                current_k.element_count()?,
                &current_v.buffer,
                current_v.element_count()?,
                batch,
                heads,
                0,
                query_tokens,
                head_dim,
                value_dim,
            )?;
            return Ok(Some(DeviceValue::new(
                vec![batch, heads, query_tokens, value_dim],
                buffer,
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (q, current_k, current_v);
            Ok(None)
        }
    }

    fn paged_kv_contiguous_device(
        &self,
        past_kv: &DevicePagedKvView,
    ) -> Result<Option<DeviceSelectedKvView>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            past_kv.validate()?;
            if past_kv.k.dtype() != DType::F32 || past_kv.v.dtype() != DType::F32 {
                return Err(Error::backend(format!(
                    "paged KV linearization requires f32 cache buffers, got k={:?}, v={:?}",
                    past_kv.k.dtype(),
                    past_kv.v.dtype()
                )));
            }
            let k = native_metal.batched_linearize_paged_cache(
                &past_kv.k.buffer,
                past_kv.k.element_count()?,
                past_kv.batch,
                past_kv.attention_heads,
                past_kv.cached_tokens,
                past_kv.capacity_tokens,
                past_kv.page_size,
                past_kv.key_head_dim,
            )?;
            let v = native_metal.batched_linearize_paged_cache(
                &past_kv.v.buffer,
                past_kv.v.element_count()?,
                past_kv.batch,
                past_kv.attention_heads,
                past_kv.cached_tokens,
                past_kv.capacity_tokens,
                past_kv.page_size,
                past_kv.value_head_dim,
            )?;
            let view = DeviceSelectedKvView {
                batch: past_kv.batch,
                attention_heads: past_kv.attention_heads,
                selected_tokens: past_kv.cached_tokens,
                key_head_dim: past_kv.key_head_dim,
                value_head_dim: past_kv.value_head_dim,
                k: DeviceValue::new(
                    vec![
                        past_kv.batch,
                        past_kv.attention_heads,
                        past_kv.cached_tokens,
                        past_kv.key_head_dim,
                    ],
                    k,
                ),
                v: DeviceValue::new(
                    vec![
                        past_kv.batch,
                        past_kv.attention_heads,
                        past_kv.cached_tokens,
                        past_kv.value_head_dim,
                    ],
                    v,
                ),
            };
            view.validate()?;
            return Ok(Some(view));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = past_kv;
            Ok(None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn q8_row_selected_kv_device(
        &self,
        key_payload: &[u8],
        value_payload: &[u8],
        batch: usize,
        attention_heads: usize,
        selected_tokens: usize,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Option<DeviceSelectedKvView>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            if batch == 0 || attention_heads == 0 || selected_tokens == 0 {
                return Err(Error::backend(
                    "Q8 selected KV device upload requires positive batch, heads and selected_tokens",
                ));
            }
            if key_head_dim == 0 || value_head_dim == 0 {
                return Err(Error::backend(
                    "Q8 selected KV device upload requires positive head dimensions",
                ));
            }
            let rows = batch
                .checked_mul(attention_heads)
                .and_then(|value| value.checked_mul(selected_tokens))
                .ok_or_else(|| Error::backend("Q8 selected KV row count overflow"))?;
            let key_payload_len = q8_row_payload_len(rows, key_head_dim)?;
            let value_payload_len = q8_row_payload_len(rows, value_head_dim)?;
            validate_exact_shape(
                "q8_selected_kv_key_payload_len",
                &[key_payload.len()],
                &[key_payload_len],
            )?;
            validate_exact_shape(
                "q8_selected_kv_value_payload_len",
                &[value_payload.len()],
                &[value_payload_len],
            )?;

            let k = native_metal.batch_upload_q8_rows_as_f32(key_payload, rows, key_head_dim)?;
            let v =
                native_metal.batch_upload_q8_rows_as_f32(value_payload, rows, value_head_dim)?;
            let view = DeviceSelectedKvView {
                batch,
                attention_heads,
                selected_tokens,
                key_head_dim,
                value_head_dim,
                k: DeviceValue::new_with_dtype(
                    vec![batch, attention_heads, selected_tokens, key_head_dim],
                    DType::F32,
                    k,
                ),
                v: DeviceValue::new_with_dtype(
                    vec![batch, attention_heads, selected_tokens, value_head_dim],
                    DType::F32,
                    v,
                ),
            };
            view.validate()?;
            return Ok(Some(view));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                key_payload,
                value_payload,
                batch,
                attention_heads,
                selected_tokens,
                key_head_dim,
                value_head_dim,
            );
            Ok(None)
        }
    }

    fn dsa_index_key_device(
        &self,
        raw_key: &DeviceValue,
        weight: &F32Tensor,
        bias: &F32Tensor,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let dims = require_device_rank("device_dsa_index_key_raw", raw_key, 3)?;
            let (batch, tokens, head_dim) = (dims[0], dims[1], dims[2]);
            validate_exact_shape("device_dsa_index_key_weight", weight.dims(), &[head_dim])?;
            validate_exact_shape("device_dsa_index_key_bias", bias.dims(), &[head_dim])?;
            let buffer = native_metal.batched_dsa_index_key(
                &raw_key.buffer,
                raw_key.element_count()?,
                weight.values(),
                bias.values(),
                batch,
                tokens,
                head_dim,
                rope_dim,
                position_offset,
                theta,
            )?;
            return Ok(Some(DeviceValue::new(dims.to_vec(), buffer)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (raw_key, weight, bias, rope_dim, position_offset, theta);
            Ok(None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn dsa_decode_topk_device(
        &self,
        hidden_states: &DeviceValue,
        q_raw: &DeviceValue,
        past_index_keys: &DeviceValue,
        current_index_key: &DeviceValue,
        weights_proj: &F32Tensor,
        heads: usize,
        head_dim: usize,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
        top_k: usize,
    ) -> Result<Option<Vec<u32>>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let hidden_dims = require_device_rank("device_dsa_topk_hidden", hidden_states, 3)?;
            let q_dims = require_device_rank("device_dsa_topk_q_raw", q_raw, 3)?;
            let past_dims = require_device_rank("device_dsa_topk_past_keys", past_index_keys, 3)?;
            let current_dims =
                require_device_rank("device_dsa_topk_current_key", current_index_key, 3)?;
            let (batch, query_tokens, hidden_size) =
                (hidden_dims[0], hidden_dims[1], hidden_dims[2]);
            validate_exact_shape("device_dsa_topk_query_tokens", &[query_tokens], &[1])?;
            validate_exact_shape(
                "device_dsa_topk_q_shape",
                q_dims,
                &[batch, 1, heads * head_dim],
            )?;
            let past_tokens = past_dims[1];
            validate_exact_shape(
                "device_dsa_topk_past_key_shape",
                past_dims,
                &[batch, past_tokens, head_dim],
            )?;
            validate_exact_shape(
                "device_dsa_topk_current_key_shape",
                current_dims,
                &[batch, 1, head_dim],
            )?;
            validate_exact_shape(
                "device_dsa_topk_weights_proj",
                weights_proj.dims(),
                &[hidden_size, heads],
            )?;

            let token_ids = native_metal.batched_dsa_decode_topk(
                &hidden_states.buffer,
                hidden_states.element_count()?,
                &q_raw.buffer,
                q_raw.element_count()?,
                &past_index_keys.buffer,
                past_index_keys.element_count()?,
                &current_index_key.buffer,
                current_index_key.element_count()?,
                weights_proj.values(),
                batch,
                hidden_size,
                past_tokens,
                heads,
                head_dim,
                rope_dim,
                position_offset,
                theta,
                top_k,
            )?;
            return Ok(Some(token_ids));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                hidden_states,
                q_raw,
                past_index_keys,
                current_index_key,
                weights_proj,
                heads,
                head_dim,
                rope_dim,
                position_offset,
                theta,
                top_k,
            );
            Ok(None)
        }
    }

    fn moe_stack_rows_device(&self, rows: &[DeviceValue]) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let first = rows
                .first()
                .ok_or_else(|| Error::backend("device MoE stack requires at least one row"))?;
            let row_len = first.element_count()?;
            if row_len == 0 {
                return Err(Error::backend("device MoE stack rows must be non-empty"));
            }
            let total_len = rows
                .len()
                .checked_mul(row_len)
                .ok_or_else(|| Error::backend("device MoE stack length overflow"))?;
            let stacked = native_metal.batched_alloc_f32(total_len)?;
            for (index, row) in rows.iter().enumerate() {
                let count = row.element_count()?;
                validate_exact_shape("device_moe_stack_row_len", &[count], &[row_len])?;
                native_metal.batched_f32_copy(
                    &row.buffer,
                    0,
                    &stacked,
                    index * row_len,
                    row_len,
                )?;
            }
            return Ok(Some(DeviceValue::new(vec![rows.len(), row_len], stacked)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = rows;
            Ok(None)
        }
    }

    fn moe_gather_rows_device(
        &self,
        input: &DeviceValue,
        token_indices: &[u32],
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            if input.dtype() != DType::F32 {
                return Err(Error::backend(format!(
                    "device MoE gather input must be F32, got {:?}",
                    input.dtype()
                )));
            }
            let dims = require_device_rank("device MoE gather input", input, 2)?;
            let token_count = dims[0];
            let hidden_size = dims[1];
            let output = native_metal.batched_moe_gather_rows(
                &input.buffer,
                input.element_count()?,
                token_indices,
                token_count,
                hidden_size,
            )?;
            return Ok(Some(DeviceValue::new(
                vec![token_indices.len(), hidden_size],
                output,
            )));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (input, token_indices);
            Ok(None)
        }
    }

    fn moe_scatter_rows_device(
        &self,
        rows: &DeviceValue,
        destination_rows: &[u32],
        destination: &DeviceValue,
    ) -> Result<Option<()>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            if rows.dtype() != DType::F32 || destination.dtype() != DType::F32 {
                return Err(Error::backend(format!(
                    "device MoE scatter source/destination must be F32, got {:?}/{:?}",
                    rows.dtype(),
                    destination.dtype()
                )));
            }
            let row_dims = require_device_rank("device MoE scatter source", rows, 2)?;
            let destination_dims =
                require_device_rank("device MoE scatter destination", destination, 2)?;
            validate_exact_shape(
                "device MoE scatter source row count",
                &[row_dims[0]],
                &[destination_rows.len()],
            )?;
            validate_exact_shape(
                "device MoE scatter hidden size",
                &[row_dims[1]],
                &[destination_dims[1]],
            )?;
            native_metal.batched_moe_scatter_rows(
                &rows.buffer,
                rows.element_count()?,
                destination_rows,
                &destination.buffer,
                destination.element_count()?,
                destination_dims[0],
                destination_dims[1],
            )?;
            return Ok(Some(()));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (rows, destination_rows, destination);
            Ok(None)
        }
    }

    fn moe_weighted_index_add_combine_device(
        &self,
        accumulator: &DeviceValue,
        token_indices: &[u32],
        expert_outputs: &DeviceValue,
        expert_weights: &[f32],
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let accumulator_dims =
                require_device_rank("device_moe_combine_accumulator", accumulator, 2)?;
            let expert_dims =
                require_device_rank("device_moe_combine_expert_outputs", expert_outputs, 2)?;
            let (token_count, hidden_size) = (accumulator_dims[0], accumulator_dims[1]);
            let assignment_count = expert_dims[0];
            validate_exact_shape(
                "device_moe_combine_hidden_size",
                &[expert_dims[1]],
                &[hidden_size],
            )?;
            let buffer = native_metal.batched_moe_weighted_index_add_combine(
                &accumulator.buffer,
                accumulator.element_count()?,
                token_indices,
                &expert_outputs.buffer,
                expert_outputs.element_count()?,
                expert_weights,
                token_count,
                hidden_size,
                assignment_count,
            )?;
            return Ok(Some(DeviceValue::new(accumulator.dims().to_vec(), buffer)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (accumulator, token_indices, expert_outputs, expert_weights);
            Ok(None)
        }
    }

    fn moe_topk_combine_residual_device(
        &self,
        shared: &DeviceValue,
        residual: &DeviceValue,
        expert_outputs: &DeviceValue,
        routing: &DeviceRouterTopK,
    ) -> Result<Option<DeviceValue>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let shared_dims = require_device_rank("device_moe_topk_shared", shared, 2)?;
            let expert_dims =
                require_device_rank("device_moe_topk_expert_outputs", expert_outputs, 2)?;
            validate_exact_shape(
                "device_moe_topk_residual_values",
                &[residual.element_count()?],
                &[shared.element_count()?],
            )?;
            validate_exact_shape(
                "device_moe_topk_token_count",
                &[shared_dims[0]],
                &[routing.token_count],
            )?;
            validate_exact_shape(
                "device_moe_topk_expert_shape",
                expert_dims,
                &[routing.assignment_count()?, shared_dims[1]],
            )?;
            let buffer = native_metal.batched_moe_topk_combine_residual(
                &shared.buffer,
                shared.element_count()?,
                &residual.buffer,
                residual.element_count()?,
                &expert_outputs.buffer,
                expert_outputs.element_count()?,
                routing,
                shared_dims[1],
            )?;
            return Ok(Some(DeviceValue::new(residual.dims().to_vec(), buffer)));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (shared, residual, expert_outputs, routing);
            Ok(None)
        }
    }

    fn moe_router_topk_device(
        &self,
        router_logits: &DeviceValue,
        correction_bias: &[f32],
        top_k: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
    ) -> Result<Option<RouterTopK>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let dims = require_device_rank("device_moe_router_topk_logits", router_logits, 2)?;
            let (token_count, expert_count) = (dims[0], dims[1]);
            validate_exact_shape(
                "device_moe_router_topk_correction_bias",
                &[correction_bias.len()],
                &[expert_count],
            )?;
            let (expert_ids, weights) = native_metal.batched_moe_router_topk(
                &router_logits.buffer,
                router_logits.element_count()?,
                correction_bias,
                token_count,
                expert_count,
                top_k,
                norm_topk_prob,
                routed_scaling_factor,
            )?;
            return Ok(Some(RouterTopK {
                token_count,
                expert_count,
                top_k,
                expert_ids,
                weights,
            }));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                router_logits,
                correction_bias,
                top_k,
                norm_topk_prob,
                routed_scaling_factor,
            );
            Ok(None)
        }
    }

    fn moe_router_topk_resident_device(
        &self,
        router_logits: &DeviceValue,
        correction_bias: &[f32],
        top_k: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
    ) -> Result<Option<DeviceRouterTopK>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            let dims = require_device_rank("device_resident_moe_router_logits", router_logits, 2)?;
            let (token_count, expert_count) = (dims[0], dims[1]);
            validate_exact_shape(
                "device_resident_moe_router_correction_bias",
                &[correction_bias.len()],
                &[expert_count],
            )?;
            let routing = native_metal.batched_moe_router_topk_resident(
                &router_logits.buffer,
                router_logits.element_count()?,
                correction_bias,
                token_count,
                expert_count,
                top_k,
                norm_topk_prob,
                routed_scaling_factor,
            )?;
            return Ok(Some(routing));
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (
                router_logits,
                correction_bias,
                top_k,
                norm_topk_prob,
                routed_scaling_factor,
            );
            Ok(None)
        }
    }

    fn moe_router_expert_ids_device(&self, routing: &DeviceRouterTopK) -> Result<Option<Vec<u32>>> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let Some(native_metal) = self.native_metal() else {
                return Ok(None);
            };
            return native_metal
                .batched_moe_router_expert_ids(routing)
                .map(Some);
        }

        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = routing;
            Ok(None)
        }
    }
}

impl MetalBackend {
    fn has_native_metal(&self) -> bool {
        self.native_metal().is_some()
    }

    /// Shared body for the four quantized matvec `*_device` trait methods.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn quant_matvec_device(
        &self,
        kind: QuantMatvecKind,
        weights: &[u8],
        input: &DeviceValue,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        let Some(native_metal) = self.native_metal() else {
            return Ok(None);
        };
        let (actual_rows, output_shape) =
            matvec_dims_shape(input.dims(), in_features, out_features)?;
        validate_exact_shape("device_quant_matvec_rows", &[actual_rows], &[row_count])?;
        let buffer = native_metal.batched_quant_matvec(
            kind,
            weights,
            &input.buffer,
            input.element_count()?,
            row_count,
            in_features,
            out_features,
        )?;
        Ok(Some(DeviceValue::new(output_shape, buffer)))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[allow(clippy::too_many_arguments)]
    fn packed_heads_transposed_matvec_device(
        &self,
        kind: QuantMatvecKind,
        weights: &[u8],
        input: &DeviceValue,
        row_count: usize,
        head_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Option<DeviceValue>> {
        let Some(native_metal) = self.native_metal() else {
            return Ok(None);
        };
        let dims = require_device_rank("device_packed_heads_transposed_input", input, 2)?;
        validate_exact_shape(
            "device_packed_heads_transposed_input_shape",
            dims,
            &[row_count, in_features],
        )?;
        let buffer = native_metal.batched_packed_heads_transposed_matvec(
            kind,
            weights,
            &input.buffer,
            input.element_count()?,
            row_count,
            head_count,
            in_features,
            out_features,
        )?;
        Ok(Some(DeviceValue::new(
            vec![row_count, head_count, out_features],
            buffer,
        )))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn native_metal(&self) -> Option<&Metal> {
        self.native_metal.as_deref()
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn native_metal(&self) -> Option<&()> {
        None
    }
}

fn backend_operations(has_native_metal: bool) -> Vec<&'static str> {
    let mut operations = vec![
        "linear",
        "matmul",
        "add",
        "select_last_token",
        "heads_to_attention_layout",
        "merge_attention_heads",
        "split_rope_tail",
        "split_kv_mqa",
        "combine_rope_tail",
        "swiglu",
        "attention_scores",
        "attention_values",
        "attention_causal_softmax",
        "rope_slice",
        "rms_norm",
        "moe_gather_tokens",
        "moe_weighted_index_add_combine",
    ];
    if has_native_metal {
        operations.push("add_f32_tensor");
        operations.push("linear_f32_tensor");
        operations.push("linear_f32_device");
        operations.push("select_last_token_f32_tensor");
        operations.push("heads_to_attention_layout_f32_tensor");
        operations.push("merge_attention_heads_f32_tensor");
        operations.push("stack_head_outputs_device");
        operations.push("paged_kv_contiguous_device");
        operations.push("split_rope_tail_f32_tensor");
        operations.push("split_kv_mqa_f32_tensor");
        operations.push("combine_rope_tail_f32_tensor");
        operations.push("swiglu_f32_tensor");
        operations.push("attention_scores_f32_tensor");
        operations.push("attention_values_f32_tensor");
        operations.push("attention_causal_softmax_f32_tensor");
        operations.push("decode_attention_f32_tensor");
        operations.push("paged_decode_attention_f32_tensor");
        operations.push("selected_decode_attention_device");
        operations.push("dsa_index_key_device");
        operations.push("dsa_decode_topk_device");
        operations.push("laguna_attention_projections_device");
        operations.push("laguna_qk_rms_norm_rope_pair_device");
        operations.push("bf16_gate_up_swiglu_device");
        operations.push("w4_groupwise_expert_wave_device");
        operations.push("laguna_fp8_kv_cache");
        operations.push("laguna_gated_gqa_attention_device");
        operations.push("rope_slice_f32_tensor");
        operations.push("moe_gather_tokens_f32_tensor");
        operations.push("moe_weighted_index_add_combine_f32_tensor");
        operations.push("moe_router_topk_device");
        operations.push("q2_k_matvec_f32_tensor");
        operations.push("q2_k_matvec_add_f32_tensor");
        operations.push("q2_k_matvec_f32");
        operations.push("q2_k_gate_up_swiglu_f32_tensor");
        operations.push("q2_k_gate_up_swiglu_f32");
        operations.push("q2_k_matvec_argmax_f32_tensor");
        operations.push("q2_k_rms_norm_argmax_f32_tensor");
        operations.push("q2_k_matvec_argmax_f32");
        operations.push("argmax_f32_device");
        operations.push("q2_k_transposed_matvec_f32_tensor");
        operations.push("q2_k_transposed_matvec_f32");
        operations.push("q8_0_matvec_f32_tensor");
        operations.push("q8_0_transposed_matvec_f32_tensor");
    }
    operations
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn native_metal_for(device_kind: DeviceKind) -> Option<Arc<Metal>> {
    if device_kind != DeviceKind::Metal {
        return None;
    }

    match Metal::new() {
        Ok(native_metal) => Some(Arc::new(native_metal)),
        Err(error) => {
            tracing::debug!(
                error = %error,
                "native Metal kernels unavailable"
            );
            None
        }
    }
}

fn matvec_input_shape(
    input: &F32Tensor,
    in_features: usize,
    out_features: usize,
) -> Result<(usize, Vec<usize>)> {
    matvec_dims_shape(input.dims(), in_features, out_features)
}

/// Derives the matvec row count and output shape from an input's dims,
/// accepting the `[rows, features]` and `[batch, tokens, features]` layouts
/// the quantized matvec ops support.
fn matvec_dims_shape(
    dims: &[usize],
    in_features: usize,
    out_features: usize,
) -> Result<(usize, Vec<usize>)> {
    match dims {
        [rows, features] => {
            validate_exact_shape("native_q2_k_matvec_input", &[*features], &[in_features])?;
            Ok((*rows, vec![*rows, out_features]))
        }
        [batch, tokens, features] => {
            validate_exact_shape("native_q2_k_matvec_input", &[*features], &[in_features])?;
            let row_count = batch
                .checked_mul(*tokens)
                .ok_or_else(|| Error::backend("native Q2_K matvec input row count overflow"))?;
            Ok((row_count, vec![*batch, *tokens, out_features]))
        }
        dims => Err(Error::backend(format!(
            "native Q2_K matvec input rank must be 2 or 3, got {dims:?}"
        ))),
    }
}

/// `require_f32_rank` for device-resident values.
fn require_device_rank<'a>(
    context: &str,
    value: &'a DeviceValue,
    rank: usize,
) -> Result<&'a [usize]> {
    let dims = value.dims();
    if dims.len() != rank {
        return Err(Error::backend(format!(
            "{context} expects rank {rank}, got {dims:?}"
        )));
    }
    Ok(dims)
}

fn q8_row_payload_len(row_count: usize, dim: usize) -> Result<usize> {
    let row_bytes = dim
        .checked_add(4)
        .ok_or_else(|| Error::backend("Q8 row byte length overflow"))?;
    row_count
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::backend("Q8 row payload length overflow"))
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn tensor_from_matvec_output(
    values: Vec<f32>,
    output_shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    match output_shape {
        [rows, features] => Ok(Tensor::from_vec(values, (*rows, *features), device)?),
        [batch, tokens, features] => Ok(Tensor::from_vec(
            values,
            (*batch, *tokens, *features),
            device,
        )?),
        other => Err(Error::backend(format!(
            "native Q2_K matvec output shape must be rank 2 or 3, got {other:?}"
        ))),
    }
}

fn tensor_from_f32_tensor(tensor: F32Tensor, device: &Device) -> Result<Tensor> {
    let (shape, values) = tensor.into_parts();
    Ok(Tensor::from_vec(values, shape.dims(), device)?)
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn tensor_from_native_values(
    values: Vec<f32>,
    output_shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    if output_shape.is_empty() {
        return Err(Error::backend("native tensor output shape is empty"));
    }
    Ok(Tensor::from_vec(values, output_shape, device)?)
}

#[cfg(test)]
fn run_backend_check_with<B: Backend>(backend: &B) -> Result<BackendCheckReport> {
    let device = backend.device();
    let mut operations = Vec::new();

    let linear_input = Tensor::from_vec(
        vec![
            0.10_f32, 0.20, -0.10, 0.30, //
            1.00, -0.50, 0.25, 0.75, //
            -0.40, 0.60, 0.80, -0.20,
        ],
        (3, 4),
        device,
    )?;
    let linear_weight = Tensor::from_vec(
        vec![
            0.25_f32, -0.50, 0.75, -1.00, //
            1.10, 0.20, -0.30, 0.60, //
            -0.40, 0.90, 0.10, -0.80, //
            0.05, -0.15, 0.35, 0.70, //
            -1.25, 0.45, 0.25, -0.05,
        ],
        (5, 4),
        device,
    )?;
    let linear_output = backend.linear(&linear_input, &linear_weight)?;
    operations.push(MetalBackend::operation_report(
        "linear",
        vec![
            shape("input", &linear_input),
            shape("weight", &linear_weight),
        ],
        &linear_output,
    )?);

    let matmul_lhs = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), device)?;
    let matmul_rhs = Tensor::from_vec(vec![0.5_f32, 1.0, -1.0, 1.5, -0.5, 0.25], (3, 2), device)?;
    let matmul_output = backend.matmul(&matmul_lhs, &matmul_rhs)?;
    operations.push(MetalBackend::operation_report(
        "matmul",
        vec![shape("lhs", &matmul_lhs), shape("rhs", &matmul_rhs)],
        &matmul_output,
    )?);

    let add_lhs = Tensor::from_vec(vec![1.0_f32, -2.0, 0.5, 4.0, 3.0, -0.25], (2, 3), device)?;
    let add_rhs = Tensor::from_vec(vec![0.25_f32, 2.5, -1.5, -0.75, 1.0, 0.5], (2, 3), device)?;
    let add_output = backend.add(&add_lhs, &add_rhs)?;
    operations.push(MetalBackend::operation_report(
        "add",
        vec![shape("lhs", &add_lhs), shape("rhs", &add_rhs)],
        &add_output,
    )?);

    let hidden_for_last = Tensor::from_vec(
        (0..2 * 3 * 4)
            .map(|idx| (idx as f32 + 1.0) / 10.0)
            .collect::<Vec<_>>(),
        (2, 3, 4),
        device,
    )?;
    let last_token = backend.select_last_token(&hidden_for_last)?;
    operations.push(MetalBackend::operation_report(
        "select_last_token",
        vec![shape("hidden_states", &hidden_for_last)],
        &last_token,
    )?);

    let heads_bthd = Tensor::from_vec(
        (0..1 * 3 * 2 * 4)
            .map(|idx| (idx as f32 + 1.0) / 10.0)
            .collect::<Vec<_>>(),
        (1, 3, 2, 4),
        device,
    )?;
    let heads_bhtd = backend.heads_to_attention_layout(&heads_bthd)?;
    operations.push(MetalBackend::operation_report(
        "heads_to_attention_layout",
        vec![shape("heads", &heads_bthd)],
        &heads_bhtd,
    )?);

    let merged_heads = backend.merge_attention_heads(&heads_bhtd)?;
    operations.push(MetalBackend::operation_report(
        "merge_attention_heads",
        vec![shape("context_heads", &heads_bhtd)],
        &merged_heads,
    )?);

    let split_input = Tensor::from_vec(
        (0..1 * 2 * 2 * 5)
            .map(|idx| (idx as f32 + 1.0) / 10.0)
            .collect::<Vec<_>>(),
        (1, 2, 2, 5),
        device,
    )?;
    let (no_rope, rope) = backend.split_rope_tail(&split_input, 3, 2)?;
    operations.push(MetalBackend::operation_report(
        "split_rope_tail",
        vec![shape("heads", &split_input)],
        &rope,
    )?);

    let kv_mqa_input = Tensor::from_vec(
        (0..1 * 2 * 6)
            .map(|idx| (idx as f32 + 1.0) / 10.0)
            .collect::<Vec<_>>(),
        (1, 2, 6),
        device,
    )?;
    let (_kv_latent, k_rope) = backend.split_kv_mqa(&kv_mqa_input, 4, 2)?;
    operations.push(MetalBackend::operation_report(
        "split_kv_mqa",
        vec![shape("kv_mqa", &kv_mqa_input)],
        &k_rope,
    )?);

    let combined_rope = backend.combine_rope_tail(&no_rope, &rope)?;
    operations.push(MetalBackend::operation_report(
        "combine_rope_tail",
        vec![shape("no_rope", &no_rope), shape("rope", &rope)],
        &combined_rope,
    )?);

    let swiglu_gate = Tensor::from_vec(vec![-1.0_f32, 0.0, 0.5, 2.0, 1.0, -0.25], (2, 3), device)?;
    let swiglu_up = Tensor::from_vec(vec![0.25_f32, 0.5, -1.0, 1.5, -0.75, 2.0], (2, 3), device)?;
    let swiglu_output = backend.swiglu(&swiglu_gate, &swiglu_up)?;
    operations.push(MetalBackend::operation_report(
        "swiglu",
        vec![shape("gate", &swiglu_gate), shape("up", &swiglu_up)],
        &swiglu_output,
    )?);

    let q = Tensor::from_vec(
        (0..1 * 2 * 3 * 4)
            .map(|idx| (idx as f32 + 1.0) / 20.0)
            .collect::<Vec<_>>(),
        (1, 2, 3, 4),
        device,
    )?;
    let k = Tensor::from_vec(
        (0..1 * 2 * 5 * 4)
            .map(|idx| (idx as f32 + 1.0) / 25.0)
            .collect::<Vec<_>>(),
        (1, 2, 5, 4),
        device,
    )?;
    let scores = backend.attention_scores(&q, &k, 4)?;
    operations.push(MetalBackend::operation_report(
        "attention_scores",
        vec![shape("q", &q), shape("k", &k)],
        &scores,
    )?);

    let causal_probabilities = backend.attention_causal_softmax(&scores, 2)?;
    operations.push(MetalBackend::operation_report(
        "attention_causal_softmax",
        vec![shape("scores", &scores)],
        &causal_probabilities,
    )?);

    let attention_values = Tensor::from_vec(
        (0..1 * 2 * 5 * 4)
            .map(|idx| (idx as f32 + 1.0) / 30.0)
            .collect::<Vec<_>>(),
        (1, 2, 5, 4),
        device,
    )?;
    let attention_context = backend.attention_values(&causal_probabilities, &attention_values)?;
    operations.push(MetalBackend::operation_report(
        "attention_values",
        vec![
            shape("probabilities", &causal_probabilities),
            shape("values", &attention_values),
        ],
        &attention_context,
    )?);

    let rope_input = Tensor::from_vec(
        (0..1 * 2 * 2 * 4)
            .map(|idx| (idx as f32 + 1.0) / 10.0)
            .collect::<Vec<_>>(),
        (1, 2, 2, 4),
        device,
    )?;
    let rope_output = backend.rope_slice(&rope_input, 4, 3, 10_000.0)?;
    operations.push(MetalBackend::operation_report(
        "rope_slice",
        vec![shape("input", &rope_input)],
        &rope_output,
    )?);

    let hidden_states = Tensor::from_vec(
        (0..2 * 3 * 4)
            .map(|idx| (idx as f32 + 1.0) / 10.0)
            .collect::<Vec<_>>(),
        (2, 3, 4),
        device,
    )?;
    let rms_weight = Tensor::from_vec(vec![1.0_f32, 1.1, 0.9, 1.2], 4, device)?;
    let normed = backend.rms_norm(&hidden_states, &rms_weight, 1e-5)?;
    operations.push(MetalBackend::operation_report(
        "rms_norm",
        vec![
            shape("hidden_states", &hidden_states),
            shape("weight", &rms_weight),
        ],
        &normed,
    )?);

    let flat_tokens = Tensor::from_vec(
        vec![
            1.0_f32, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0,
        ],
        (3, 4),
        device,
    )?;
    let gather_indices = vec![2_u32, 0, 2];
    let gathered = backend.moe_gather_tokens(&flat_tokens, &gather_indices)?;
    operations.push(MetalBackend::operation_report(
        "moe_gather_tokens",
        vec![shape("flat_tokens", &flat_tokens)],
        &gathered,
    )?);

    let accumulator = Tensor::zeros((6, 4))?;
    let token_indices = Tensor::from_vec(vec![0_u32, 2, 2, 5], 4, device)?;
    let expert_outputs = Tensor::from_vec(
        (0..4 * 4)
            .map(|idx| (idx as f32 + 1.0) / 30.0)
            .collect::<Vec<_>>(),
        (4, 4),
        device,
    )?;
    let expert_weights = Tensor::from_vec(vec![1.0_f32, 0.5, 2.0, 0.25], 4, device)?;
    let combined = backend.moe_weighted_index_add_combine(
        &accumulator,
        &token_indices,
        &expert_outputs,
        &expert_weights,
    )?;
    operations.push(MetalBackend::operation_report(
        "moe_weighted_index_add_combine",
        vec![
            shape("accumulator", &accumulator),
            shape("token_indices", &token_indices),
            shape("expert_outputs", &expert_outputs),
            shape("expert_weights", &expert_weights),
        ],
        &combined,
    )?);

    Ok(BackendCheckReport {
        capabilities: backend.capabilities(),
        operations,
    })
}

fn validate_matmul_shapes(lhs: &Tensor, rhs: &Tensor) -> Result<()> {
    let lhs_dims = lhs.dims();
    let rhs_dims = rhs.dims();
    if lhs_dims.len() != 2 || rhs_dims.len() != 2 {
        return Err(Error::backend(format!(
            "matmul inputs must be rank 2 [rows, inner] x [inner, cols], got lhs={lhs_dims:?} rhs={rhs_dims:?}"
        )));
    }

    validate_exact_shape("matmul_contract_dim", &[lhs_dims[1]], &[rhs_dims[0]])
}

fn validate_linear_shapes(input: &Tensor, weight: &Tensor) -> Result<()> {
    let input_dims = input.dims();
    let weight_dims = weight.dims();
    if weight_dims.len() != 2 {
        return Err(Error::backend(format!(
            "linear weight rank must be 2 [out_features, in_features], got {weight_dims:?}"
        )));
    }
    if !(input_dims.len() == 2 || input_dims.len() == 3) {
        return Err(Error::backend(format!(
            "linear input rank must be 2 or 3, got {input_dims:?}"
        )));
    }

    let input_in_features = *input_dims
        .last()
        .ok_or_else(|| Error::backend("linear input has empty shape"))?;
    let weight_in_features = weight_dims[1];
    validate_exact_shape(
        "linear_in_features",
        &[input_in_features],
        &[weight_in_features],
    )
}

fn validate_add_shapes(lhs: &Tensor, rhs: &Tensor) -> Result<()> {
    let lhs_dims = lhs.dims();
    let rhs_dims = rhs.dims();
    if !(lhs_dims.len() == 2 || lhs_dims.len() == 3) {
        return Err(Error::backend(format!(
            "add lhs rank must be 2 or 3, got {lhs_dims:?}"
        )));
    }
    if lhs_dims != rhs_dims {
        return Err(Error::backend(format!(
            "add lhs/rhs shapes must match, got lhs={lhs_dims:?} rhs={rhs_dims:?}"
        )));
    }
    if lhs_dims.iter().any(|dim| *dim == 0) {
        return Err(Error::backend(format!(
            "add dimensions must be positive, got {lhs_dims:?}"
        )));
    }
    Ok(())
}

fn validate_select_last_token_shapes(hidden_states: &Tensor) -> Result<()> {
    let dims = hidden_states.dims();
    if dims.len() != 3 {
        return Err(Error::backend(format!(
            "select_last_token input must be rank 3 [B,T,H], got {dims:?}"
        )));
    }
    if dims[0] == 0 || dims[1] == 0 || dims[2] == 0 {
        return Err(Error::backend(format!(
            "select_last_token dimensions must be positive, got {dims:?}"
        )));
    }
    Ok(())
}

fn validate_heads_to_attention_layout_shapes(heads: &Tensor) -> Result<()> {
    let dims = heads.dims();
    if dims.len() != 4 {
        return Err(Error::backend(format!(
            "heads_to_attention_layout input must be rank 4 [B,T,H,D], got {dims:?}"
        )));
    }
    if dims.iter().any(|dim| *dim == 0) {
        return Err(Error::backend(format!(
            "heads_to_attention_layout dimensions must be positive, got {dims:?}"
        )));
    }
    Ok(())
}

fn validate_merge_attention_heads_shapes(context_heads: &Tensor) -> Result<()> {
    let dims = context_heads.dims();
    if dims.len() != 4 {
        return Err(Error::backend(format!(
            "merge_attention_heads input must be rank 4 [B,H,T,D], got {dims:?}"
        )));
    }
    if dims.iter().any(|dim| *dim == 0) {
        return Err(Error::backend(format!(
            "merge_attention_heads dimensions must be positive, got {dims:?}"
        )));
    }
    Ok(())
}

fn validate_split_rope_tail_shapes(
    heads: &Tensor,
    no_rope_dim: usize,
    rope_dim: usize,
) -> Result<()> {
    let dims = heads.dims();
    if dims.len() != 4 {
        return Err(Error::backend(format!(
            "split_rope_tail input must be rank 4 [B,T,H,D], got {dims:?}"
        )));
    }
    if dims.iter().any(|dim| *dim == 0) || no_rope_dim == 0 || rope_dim == 0 {
        return Err(Error::backend(format!(
            "split_rope_tail dimensions must be positive, got input={dims:?} no_rope_dim={no_rope_dim} rope_dim={rope_dim}"
        )));
    }
    let total_dim = no_rope_dim
        .checked_add(rope_dim)
        .ok_or_else(|| Error::backend("split_rope_tail total dim overflow"))?;
    validate_exact_shape("split_rope_tail_last_dim", &[dims[3]], &[total_dim])
}

fn validate_split_kv_mqa_shapes(
    kv_mqa: &Tensor,
    kv_lora_rank: usize,
    rope_dim: usize,
) -> Result<()> {
    let dims = kv_mqa.dims();
    if dims.len() != 3 {
        return Err(Error::backend(format!(
            "split_kv_mqa input must be rank 3 [B,T,kv_lora+rope], got {dims:?}"
        )));
    }
    if dims.iter().any(|dim| *dim == 0) || kv_lora_rank == 0 || rope_dim == 0 {
        return Err(Error::backend(format!(
            "split_kv_mqa dimensions must be positive, got input={dims:?} kv_lora_rank={kv_lora_rank} rope_dim={rope_dim}"
        )));
    }
    let total_dim = kv_lora_rank
        .checked_add(rope_dim)
        .ok_or_else(|| Error::backend("split_kv_mqa total dim overflow"))?;
    validate_exact_shape("split_kv_mqa_last_dim", &[dims[2]], &[total_dim])
}

fn validate_combine_rope_tail_shapes(no_rope: &Tensor, rope: &Tensor) -> Result<()> {
    let no_rope_dims = no_rope.dims();
    let rope_dims = rope.dims();
    if no_rope_dims.len() != 4 || rope_dims.len() != 4 {
        return Err(Error::backend(format!(
            "combine_rope_tail inputs must be rank 4 [B,T,H,D], got no_rope={no_rope_dims:?} rope={rope_dims:?}"
        )));
    }
    if no_rope_dims.iter().any(|dim| *dim == 0) || rope_dims.iter().any(|dim| *dim == 0) {
        return Err(Error::backend(format!(
            "combine_rope_tail dimensions must be positive, got no_rope={no_rope_dims:?} rope={rope_dims:?}"
        )));
    }
    validate_exact_shape(
        "combine_rope_tail_batch",
        &[no_rope_dims[0]],
        &[rope_dims[0]],
    )?;
    validate_exact_shape(
        "combine_rope_tail_tokens",
        &[no_rope_dims[1]],
        &[rope_dims[1]],
    )?;
    if rope_dims[2] != 1 && rope_dims[2] != no_rope_dims[2] {
        return Err(Error::backend(format!(
            "combine_rope_tail rope head count must be 1 or {}, got {}",
            no_rope_dims[2], rope_dims[2]
        )));
    }
    Ok(())
}

fn validate_swiglu_shapes(gate: &Tensor, up: &Tensor) -> Result<()> {
    let gate_dims = gate.dims();
    let up_dims = up.dims();
    if !(gate_dims.len() == 2 || gate_dims.len() == 3) {
        return Err(Error::backend(format!(
            "SwiGLU gate rank must be 2 or 3, got {gate_dims:?}"
        )));
    }
    if gate_dims != up_dims {
        return Err(Error::backend(format!(
            "SwiGLU gate/up shapes must match, got gate={gate_dims:?} up={up_dims:?}"
        )));
    }
    if gate_dims.iter().any(|dim| *dim == 0) {
        return Err(Error::backend(format!(
            "SwiGLU dimensions must be positive, got {gate_dims:?}"
        )));
    }
    Ok(())
}

fn validate_attention_shapes(q: &Tensor, k: &Tensor, head_dim: usize) -> Result<()> {
    let q_dims = q.dims();
    let k_dims = k.dims();
    if q_dims.len() != 4 || k_dims.len() != 4 {
        return Err(Error::backend(format!(
            "attention q/k rank must be 4, got q={q_dims:?} k={k_dims:?}"
        )));
    }

    validate_exact_shape("attention_batch", &[q_dims[0]], &[k_dims[0]])?;
    validate_exact_shape("attention_heads", &[q_dims[1]], &[k_dims[1]])?;
    validate_exact_shape("attention_q_head_dim", &[q_dims[3]], &[head_dim])?;
    validate_exact_shape("attention_k_head_dim", &[k_dims[3]], &[head_dim])
}

fn validate_attention_value_shapes(probs: &Tensor, values: &Tensor) -> Result<()> {
    let probs_dims = probs.dims();
    let value_dims = values.dims();
    if probs_dims.len() != 4 || value_dims.len() != 4 {
        return Err(Error::backend(format!(
            "attention value aggregation inputs must be rank 4, got probs={probs_dims:?} values={value_dims:?}"
        )));
    }

    validate_exact_shape("attention_values_batch", &[probs_dims[0]], &[value_dims[0]])?;
    validate_exact_shape("attention_values_heads", &[probs_dims[1]], &[value_dims[1]])?;
    validate_exact_shape(
        "attention_values_key_tokens",
        &[probs_dims[3]],
        &[value_dims[2]],
    )?;
    Ok(())
}

fn validate_attention_causal_softmax_shapes(scores: &Tensor, past_tokens: usize) -> Result<()> {
    let dims = scores.dims();
    if dims.len() != 4 {
        return Err(Error::backend(format!(
            "attention causal softmax input must be rank 4 [B,H,Q,K], got {dims:?}"
        )));
    }

    let query_tokens = dims[2];
    let key_tokens = dims[3];
    if past_tokens
        .checked_add(query_tokens)
        .filter(|expected_key_tokens| *expected_key_tokens == key_tokens)
        .is_none()
    {
        return Err(Error::backend(format!(
            "attention causal softmax expects key_tokens == past_tokens + query_tokens, got key_tokens={key_tokens}, past_tokens={past_tokens}, query_tokens={query_tokens}"
        )));
    }
    if query_tokens == 0 || key_tokens == 0 {
        return Err(Error::backend(
            "attention causal softmax query_tokens and key_tokens must be positive",
        ));
    }

    Ok(())
}

fn validate_rope_slice_shapes(
    input: &Tensor,
    rope_dim: usize,
    position_offset: usize,
    theta: f32,
) -> Result<()> {
    let dims = input.dims();
    if dims.len() != 4 {
        return Err(Error::backend(format!(
            "RoPE input must be rank 4 [B,T,H,D], got {dims:?}"
        )));
    }
    if rope_dim == 0 {
        return Err(Error::backend("RoPE rope_dim must be positive"));
    }
    if rope_dim % 2 != 0 {
        return Err(Error::backend(format!(
            "RoPE rope_dim must be even, got {rope_dim}"
        )));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(Error::backend(
            "RoPE theta must be finite and greater than zero",
        ));
    }
    if dims[1] == 0 {
        return Err(Error::backend("RoPE token dimension must be positive"));
    }
    position_offset
        .checked_add(dims[1] - 1)
        .ok_or_else(|| Error::backend("RoPE position range overflow"))?;
    validate_exact_shape("rope_dim", &[dims[3]], &[rope_dim])
}

fn validate_rms_norm_shapes(hidden_states: &Tensor, weight: &Tensor) -> Result<()> {
    let hidden_dims = hidden_states.dims();
    let weight_dims = weight.dims();
    if hidden_dims.is_empty() {
        return Err(Error::backend("rms_norm hidden_states must have rank >= 1"));
    }
    if weight_dims.len() != 1 {
        return Err(Error::backend(format!(
            "rms_norm weight rank must be 1 [hidden_size], got {weight_dims:?}"
        )));
    }

    let hidden_size = hidden_dims
        .last()
        .copied()
        .ok_or_else(|| Error::backend("rms_norm hidden_states must have rank >= 1"))?;
    validate_exact_shape("rms_norm_hidden_size", &[hidden_size], &[weight_dims[0]])
}

fn validate_rms_norm_f32_shapes(
    hidden_states: &F32Tensor,
    weight: &F32Tensor,
    eps: f32,
) -> Result<()> {
    let hidden_dims = hidden_states.dims();
    let weight_dims = weight.dims();
    if hidden_dims.is_empty() {
        return Err(Error::backend("rms_norm hidden_states must have rank >= 1"));
    }
    if hidden_dims.iter().any(|dim| *dim == 0) {
        return Err(Error::backend(format!(
            "rms_norm hidden_states dimensions must be positive, got {hidden_dims:?}"
        )));
    }
    if weight_dims.len() != 1 {
        return Err(Error::backend(format!(
            "rms_norm weight rank must be 1 [hidden_size], got {weight_dims:?}"
        )));
    }
    if eps <= 0.0 || !eps.is_finite() {
        return Err(Error::backend(
            "rms_norm eps must be finite and greater than zero",
        ));
    }

    let hidden_size = hidden_dims
        .last()
        .copied()
        .ok_or_else(|| Error::backend("rms_norm hidden_states must have rank >= 1"))?;
    validate_exact_shape("rms_norm_hidden_size", &[hidden_size], &[weight_dims[0]])
}

fn require_f32_rank<'a>(context: &str, tensor: &'a F32Tensor, rank: usize) -> Result<&'a [usize]> {
    let dims = tensor.dims();
    if dims.len() != rank {
        return Err(Error::backend(format!(
            "{context} tensor must have rank {rank}, got {dims:?}"
        )));
    }
    if dims.iter().any(|dim| *dim == 0) {
        return Err(Error::backend(format!(
            "{context} tensor dimensions must be positive, got {dims:?}"
        )));
    }
    Ok(dims)
}

fn validate_moe_gather_shapes(flat_tokens: &Tensor, token_indices: &[u32]) -> Result<()> {
    let dims = flat_tokens.dims();
    if dims.len() != 2 {
        return Err(Error::backend(format!(
            "MoE gather flat_tokens must be rank 2 [token_count, hidden_size], got {dims:?}"
        )));
    }
    if dims[0] == 0 || dims[1] == 0 {
        return Err(Error::backend(format!(
            "MoE gather dimensions must be positive, got {dims:?}"
        )));
    }
    if token_indices.is_empty() {
        return Err(Error::backend("MoE gather token_indices must not be empty"));
    }
    if let Some(token_index) = token_indices
        .iter()
        .copied()
        .find(|token_index| *token_index as usize >= dims[0])
    {
        return Err(Error::backend(format!(
            "MoE gather token index {token_index} is outside token_count {}",
            dims[0]
        )));
    }
    Ok(())
}

fn validate_moe_weighted_index_add_shapes(
    accumulator: &Tensor,
    token_indices: &Tensor,
    expert_outputs: &Tensor,
    expert_weights: &Tensor,
) -> Result<()> {
    let accumulator_dims = accumulator.dims();
    let token_index_dims = token_indices.dims();
    let expert_output_dims = expert_outputs.dims();
    let expert_weight_dims = expert_weights.dims();
    if accumulator_dims.len() != 2 || expert_output_dims.len() != 2 {
        return Err(Error::backend(format!(
            "moe combine accumulator and expert_outputs must be rank 2, got accumulator={accumulator_dims:?} expert_outputs={expert_output_dims:?}"
        )));
    }
    if token_index_dims.len() != 1 {
        return Err(Error::backend(format!(
            "moe combine token_indices must be rank 1, got {token_index_dims:?}"
        )));
    }
    if expert_weight_dims.len() != 1 {
        return Err(Error::backend(format!(
            "moe combine expert_weights must be rank 1, got {expert_weight_dims:?}"
        )));
    }

    validate_exact_shape(
        "moe_combine_hidden_size",
        &[accumulator_dims[1]],
        &[expert_output_dims[1]],
    )?;
    validate_exact_shape(
        "moe_combine_assignment_count",
        &[token_index_dims[0]],
        &[expert_output_dims[0]],
    )?;
    validate_exact_shape(
        "moe_combine_weight_count",
        &[expert_weight_dims[0]],
        &[expert_output_dims[0]],
    )
}

fn flatten_linear_input(input: &Tensor) -> Result<(usize, Vec<usize>, Vec<f32>)> {
    let input = input.to_dtype(common::DType::F32)?;
    match input.dims() {
        [rows, features] => {
            let values = input
                .to_vec2::<f32>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            Ok((*rows, vec![*rows, *features], values))
        }
        [batch, tokens, features] => {
            let rows = batch
                .checked_mul(*tokens)
                .ok_or_else(|| Error::backend("linear input row count overflow"))?;
            let values = input
                .to_vec3::<f32>()?
                .into_iter()
                .flat_map(|batch_rows| batch_rows.into_iter().flatten())
                .collect::<Vec<_>>();
            Ok((rows, vec![*batch, *tokens, *features], values))
        }
        dims => Err(Error::backend(format!(
            "linear input rank must be 2 or 3, got {dims:?}"
        ))),
    }
}

fn reference_matmul(lhs: &Tensor, rhs: &Tensor, device: &Device) -> Result<Tensor> {
    let lhs_dims = lhs.dims();
    let rhs_dims = rhs.dims();
    let rows = lhs_dims[0];
    let inner = lhs_dims[1];
    let cols = rhs_dims[1];
    let lhs_values = lhs
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let rhs_values = rhs
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    debug_assert_finite_values("matmul lhs", &lhs_values);
    debug_assert_finite_values("matmul rhs", &rhs_values);

    let mut output = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0_f32;
            for index in 0..inner {
                sum += lhs_values[row * inner + index] * rhs_values[index * cols + col];
            }
            output[row * cols + col] = sum;
        }
    }

    Ok(Tensor::from_vec(output, (rows, cols), device)?)
}

fn reference_linear_from_values(
    input_values: &[f32],
    weight_values: &[f32],
    rows: usize,
    in_features: usize,
    out_features: usize,
    output_shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    debug_assert_finite_values("linear input", input_values);
    debug_assert_finite_values("linear weight", weight_values);

    let expected_input_len = rows
        .checked_mul(in_features)
        .ok_or_else(|| Error::backend("linear input length overflow"))?;
    validate_exact_shape(
        "linear_input_values",
        &[input_values.len()],
        &[expected_input_len],
    )?;
    let expected_weight_len = out_features
        .checked_mul(in_features)
        .ok_or_else(|| Error::backend("linear weight length overflow"))?;
    validate_exact_shape(
        "linear_weight_values",
        &[weight_values.len()],
        &[expected_weight_len],
    )?;

    let output_len = rows
        .checked_mul(out_features)
        .ok_or_else(|| Error::backend("linear output length overflow"))?;
    let mut output = vec![0.0_f32; output_len];
    for row in 0..rows {
        for output_feature in 0..out_features {
            let mut sum = 0.0_f32;
            for input_feature in 0..in_features {
                sum += input_values[row * in_features + input_feature]
                    * weight_values[output_feature * in_features + input_feature];
            }
            output[row * out_features + output_feature] = sum;
        }
    }

    tensor_from_values(output, output_shape, device)
}

fn reference_add_from_values(
    lhs_values: &[f32],
    rhs_values: &[f32],
    output_shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    validate_exact_shape("add_value_count", &[lhs_values.len()], &[rhs_values.len()])?;
    debug_assert_finite_values("add lhs", lhs_values);
    debug_assert_finite_values("add rhs", rhs_values);

    let output = lhs_values
        .iter()
        .zip(rhs_values)
        .map(|(lhs, rhs)| lhs + rhs)
        .collect::<Vec<_>>();

    tensor_from_values(output, output_shape, device)
}

fn reference_select_last_token(hidden_states: &Tensor, device: &Device) -> Result<Tensor> {
    let dims = hidden_states.dims();
    let batch = dims[0];
    let tokens = dims[1];
    let hidden_size = dims[2];
    let input = hidden_states
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    validate_reference_select_last_token(&input, batch, tokens, hidden_size)?;

    let mut output = Vec::with_capacity(batch * hidden_size);
    for batch_index in 0..batch {
        let start = ((batch_index * tokens) + (tokens - 1))
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("select_last_token source offset overflow"))?;
        let end = start
            .checked_add(hidden_size)
            .ok_or_else(|| Error::backend("select_last_token source end overflow"))?;
        output.extend_from_slice(&input[start..end]);
    }

    Ok(Tensor::from_vec(output, (batch, 1, hidden_size), device)?)
}

fn reference_heads_to_attention_layout(heads: &Tensor, device: &Device) -> Result<Tensor> {
    let dims = heads.dims();
    let batch = dims[0];
    let tokens = dims[1];
    let head_count = dims[2];
    let head_dim = dims[3];
    let input = heads
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    validate_reference_layout_values(
        "heads_to_attention_layout",
        &input,
        &[batch, tokens, head_count, head_dim],
    )?;

    let mut output = vec![0.0_f32; input.len()];
    for batch_index in 0..batch {
        for head_index in 0..head_count {
            for token_index in 0..tokens {
                for dim_index in 0..head_dim {
                    let source = (((batch_index * tokens + token_index) * head_count + head_index)
                        * head_dim)
                        + dim_index;
                    let target = (((batch_index * head_count + head_index) * tokens + token_index)
                        * head_dim)
                        + dim_index;
                    output[target] = input[source];
                }
            }
        }
    }

    Ok(Tensor::from_vec(
        output,
        (batch, head_count, tokens, head_dim),
        device,
    )?)
}

fn reference_merge_attention_heads(context_heads: &Tensor, device: &Device) -> Result<Tensor> {
    let dims = context_heads.dims();
    let batch = dims[0];
    let head_count = dims[1];
    let tokens = dims[2];
    let head_dim = dims[3];
    let merged_width = head_count
        .checked_mul(head_dim)
        .ok_or_else(|| Error::backend("merge_attention_heads merged width overflow"))?;
    let input = context_heads
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    validate_reference_layout_values(
        "merge_attention_heads",
        &input,
        &[batch, head_count, tokens, head_dim],
    )?;

    let mut output = vec![0.0_f32; input.len()];
    for batch_index in 0..batch {
        for token_index in 0..tokens {
            for head_index in 0..head_count {
                for dim_index in 0..head_dim {
                    let source = (((batch_index * head_count + head_index) * tokens + token_index)
                        * head_dim)
                        + dim_index;
                    let target = (((batch_index * tokens + token_index) * head_count + head_index)
                        * head_dim)
                        + dim_index;
                    output[target] = input[source];
                }
            }
        }
    }

    Ok(Tensor::from_vec(
        output,
        (batch, tokens, merged_width),
        device,
    )?)
}

fn reference_split_rope_tail(
    heads: &Tensor,
    no_rope_dim: usize,
    rope_dim: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let dims = heads.dims();
    let batch = dims[0];
    let tokens = dims[1];
    let head_count = dims[2];
    let total_dim = dims[3];
    let input = heads
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    validate_reference_layout_values(
        "split_rope_tail",
        &input,
        &[batch, tokens, head_count, total_dim],
    )?;

    let mut no_rope = vec![0.0_f32; batch * tokens * head_count * no_rope_dim];
    let mut rope = vec![0.0_f32; batch * tokens * head_count * rope_dim];
    for batch_index in 0..batch {
        for token_index in 0..tokens {
            for head_index in 0..head_count {
                for dim_index in 0..no_rope_dim {
                    let source = (((batch_index * tokens + token_index) * head_count + head_index)
                        * total_dim)
                        + dim_index;
                    let target = (((batch_index * tokens + token_index) * head_count + head_index)
                        * no_rope_dim)
                        + dim_index;
                    no_rope[target] = input[source];
                }
                for dim_index in 0..rope_dim {
                    let source = (((batch_index * tokens + token_index) * head_count + head_index)
                        * total_dim)
                        + no_rope_dim
                        + dim_index;
                    let target = (((batch_index * tokens + token_index) * head_count + head_index)
                        * rope_dim)
                        + dim_index;
                    rope[target] = input[source];
                }
            }
        }
    }

    let no_rope = Tensor::from_vec(no_rope, (batch, tokens, head_count, no_rope_dim), device)?;
    let rope = Tensor::from_vec(rope, (batch, tokens, head_count, rope_dim), device)?;
    Ok((no_rope, rope))
}

fn reference_split_kv_mqa(
    kv_mqa: &Tensor,
    kv_lora_rank: usize,
    rope_dim: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let dims = kv_mqa.dims();
    let batch = dims[0];
    let tokens = dims[1];
    let total_dim = dims[2];
    let input = kv_mqa
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    validate_reference_layout_values("split_kv_mqa", &input, &[batch, tokens, total_dim])?;

    let mut kv_latent = vec![0.0_f32; batch * tokens * kv_lora_rank];
    let mut k_rope = vec![0.0_f32; batch * tokens * rope_dim];
    for batch_index in 0..batch {
        for token_index in 0..tokens {
            for dim_index in 0..kv_lora_rank {
                let source = ((batch_index * tokens + token_index) * total_dim) + dim_index;
                let target = ((batch_index * tokens + token_index) * kv_lora_rank) + dim_index;
                kv_latent[target] = input[source];
            }
            for dim_index in 0..rope_dim {
                let source =
                    ((batch_index * tokens + token_index) * total_dim) + kv_lora_rank + dim_index;
                let target = ((batch_index * tokens + token_index) * rope_dim) + dim_index;
                k_rope[target] = input[source];
            }
        }
    }

    let kv_latent = Tensor::from_vec(kv_latent, (batch, tokens, kv_lora_rank), device)?;
    let k_rope = Tensor::from_vec(k_rope, (batch, tokens, 1, rope_dim), device)?;
    Ok((kv_latent, k_rope))
}

fn reference_combine_rope_tail(no_rope: &Tensor, rope: &Tensor, device: &Device) -> Result<Tensor> {
    let no_rope_dims = no_rope.dims();
    let rope_dims = rope.dims();
    let batch = no_rope_dims[0];
    let tokens = no_rope_dims[1];
    let head_count = no_rope_dims[2];
    let no_rope_dim = no_rope_dims[3];
    let rope_head_count = rope_dims[2];
    let rope_dim = rope_dims[3];
    let total_dim = no_rope_dim
        .checked_add(rope_dim)
        .ok_or_else(|| Error::backend("combine_rope_tail total dim overflow"))?;
    let no_rope_values = no_rope
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let rope_values = rope
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    validate_reference_layout_values(
        "combine_rope_tail_no_rope",
        &no_rope_values,
        &[batch, tokens, head_count, no_rope_dim],
    )?;
    validate_reference_layout_values(
        "combine_rope_tail_rope",
        &rope_values,
        &[batch, tokens, rope_head_count, rope_dim],
    )?;

    let mut output = vec![0.0_f32; batch * tokens * head_count * total_dim];
    for batch_index in 0..batch {
        for token_index in 0..tokens {
            for head_index in 0..head_count {
                for dim_index in 0..no_rope_dim {
                    let source = (((batch_index * tokens + token_index) * head_count + head_index)
                        * no_rope_dim)
                        + dim_index;
                    let target = (((batch_index * tokens + token_index) * head_count + head_index)
                        * total_dim)
                        + dim_index;
                    output[target] = no_rope_values[source];
                }
                for dim_index in 0..rope_dim {
                    let rope_head_index = if rope_head_count == 1 { 0 } else { head_index };
                    let source = (((batch_index * tokens + token_index) * rope_head_count
                        + rope_head_index)
                        * rope_dim)
                        + dim_index;
                    let target = (((batch_index * tokens + token_index) * head_count + head_index)
                        * total_dim)
                        + no_rope_dim
                        + dim_index;
                    output[target] = rope_values[source];
                }
            }
        }
    }

    Ok(Tensor::from_vec(
        output,
        (batch, tokens, head_count, total_dim),
        device,
    )?)
}

fn reference_swiglu_from_values(
    gate_values: &[f32],
    up_values: &[f32],
    output_shape: &[usize],
    device: &Device,
) -> Result<Tensor> {
    validate_exact_shape(
        "SwiGLU_value_count",
        &[gate_values.len()],
        &[up_values.len()],
    )?;
    debug_assert_finite_values("SwiGLU gate", gate_values);
    debug_assert_finite_values("SwiGLU up", up_values);

    let mut output = Vec::with_capacity(gate_values.len());
    for (gate, up) in gate_values.iter().zip(up_values) {
        let silu = *gate / (1.0 + (-*gate).exp());
        output.push(silu * *up);
    }

    tensor_from_values(output, output_shape, device)
}

fn reference_attention_causal_softmax(
    scores: &Tensor,
    past_tokens: usize,
    device: &Device,
) -> Result<Tensor> {
    let dims = scores.dims().to_vec();
    let batch = dims[0];
    let heads = dims[1];
    let query_tokens = dims[2];
    let key_tokens = dims[3];
    let input = scores
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    validate_reference_attention_causal_softmax(
        &input,
        batch,
        heads,
        query_tokens,
        key_tokens,
        past_tokens,
    )?;

    let mut output = vec![0.0_f32; input.len()];

    for batch_index in 0..batch {
        for head_index in 0..heads {
            for query_index in 0..query_tokens {
                let base =
                    ((batch_index * heads + head_index) * query_tokens + query_index) * key_tokens;
                let max_visible_key = past_tokens
                    .checked_add(query_index)
                    .ok_or_else(|| Error::backend("attention causal softmax key index overflow"))?;
                let row_values = &input[base..base + max_visible_key + 1];
                let max_value = row_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0_f32;
                for key_index in 0..=max_visible_key {
                    let value = (input[base + key_index] - max_value).exp();
                    output[base + key_index] = value;
                    sum += value;
                }
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(Error::backend(
                        "attention causal softmax normalization sum is invalid",
                    ));
                }
                for key_index in 0..=max_visible_key {
                    output[base + key_index] /= sum;
                }
            }
        }
    }

    Ok(Tensor::from_vec(
        output,
        (batch, heads, query_tokens, key_tokens),
        device,
    )?)
}

fn tensor_to_f32_tensor(tensor: &Tensor) -> Result<F32Tensor> {
    let dims = tensor.dims().to_vec();
    let values = tensor
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    F32Tensor::new(values, dims)
}

fn reference_rope_slice(
    input: &Tensor,
    rope_dim: usize,
    position_offset: usize,
    theta: f32,
    device: &Device,
) -> Result<Tensor> {
    let dims = input.dims().to_vec();
    let batch = dims[0];
    let tokens = dims[1];
    let heads = dims[2];
    let input_values = input
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    validate_reference_rope_slice(
        &input_values,
        batch,
        tokens,
        heads,
        rope_dim,
        position_offset,
        theta,
    )?;

    let mut output = vec![0.0_f32; input_values.len()];
    let pair_count = rope_dim / 2;

    for batch_index in 0..batch {
        for token_index in 0..tokens {
            let position = position_offset
                .checked_add(token_index)
                .ok_or_else(|| Error::backend("RoPE position overflow"))?;
            for head_index in 0..heads {
                let base = ((batch_index * tokens + token_index) * heads + head_index) * rope_dim;
                for dim_index in 0..rope_dim {
                    let pair_index = dim_index / 2;
                    let partner_dim = if dim_index % 2 == 0 {
                        dim_index + 1
                    } else {
                        dim_index - 1
                    };
                    let inv_freq = 1.0_f32 / theta.powf(pair_index as f32 / pair_count as f32);
                    let angle = position as f32 * inv_freq;
                    let rotated = if dim_index % 2 == 0 {
                        -input_values[base + partner_dim]
                    } else {
                        input_values[base + partner_dim]
                    };
                    output[base + dim_index] =
                        input_values[base + dim_index] * angle.cos() + rotated * angle.sin();
                }
            }
        }
    }

    Ok(Tensor::from_vec(
        output,
        (batch, tokens, heads, rope_dim),
        device,
    )?)
}

fn reference_rms_norm_f32(
    hidden_states: &F32Tensor,
    weight: &F32Tensor,
    eps: f32,
) -> Result<F32Tensor> {
    validate_rms_norm_f32_shapes(hidden_states, weight, eps)?;
    let dims = hidden_states.dims();
    let hidden_size = *dims
        .last()
        .ok_or_else(|| Error::backend("rms_norm input has empty shape"))?;
    let rows = hidden_states
        .values()
        .len()
        .checked_div(hidden_size)
        .ok_or_else(|| Error::backend("rms_norm row count division overflow"))?;

    let input = hidden_states.values();
    let weight = weight.values();
    let mut output = vec![0.0_f32; input.len()];
    for row in 0..rows {
        let start = row
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("rms_norm row offset overflow"))?;
        let end = start
            .checked_add(hidden_size)
            .ok_or_else(|| Error::backend("rms_norm row end overflow"))?;
        let row_values = &input[start..end];
        let mut squared_sum = 0.0_f32;
        for value in row_values {
            let squared = *value * *value;
            if !squared.is_finite() {
                return Err(Error::backend("rms_norm square is non-finite"));
            }
            squared_sum += squared;
        }
        let mean_square = squared_sum / hidden_size as f32;
        let scale = (mean_square + eps).sqrt();
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::backend("rms_norm scale is invalid"));
        }
        for index in 0..hidden_size {
            output[start + index] = row_values[index] / scale * weight[index];
        }
    }

    F32Tensor::new(output, dims.to_vec())
}

fn reference_moe_gather_tokens(
    flat_tokens: &Tensor,
    token_indices: &[u32],
    device: &Device,
) -> Result<Tensor> {
    let token_count = flat_tokens.dims()[0];
    let hidden_size = flat_tokens.dims()[1];
    let flat_values = flat_tokens
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    validate_reference_moe_gather_tokens(
        &flat_values,
        token_indices,
        token_count,
        hidden_size,
        token_indices.len(),
    )?;

    let mut output = vec![0.0_f32; token_indices.len() * hidden_size];
    for (assignment, token_index) in token_indices.iter().copied().enumerate() {
        let token_index = token_index as usize;
        let source_start = token_index
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("MoE gather source offset overflow"))?;
        let output_start = assignment
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("MoE gather output offset overflow"))?;
        output[output_start..output_start + hidden_size]
            .copy_from_slice(&flat_values[source_start..source_start + hidden_size]);
    }

    Ok(Tensor::from_vec(
        output,
        (token_indices.len(), hidden_size),
        device,
    )?)
}

fn reference_moe_weighted_index_add_combine(
    accumulator: &Tensor,
    token_indices: &Tensor,
    expert_outputs: &Tensor,
    expert_weights: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    let token_count = accumulator.dims()[0];
    let hidden_size = accumulator.dims()[1];
    let assignment_count = token_indices.dims()[0];
    let mut output = accumulator
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let token_indices = token_indices.to_vec1::<u32>()?;
    let expert_outputs = expert_outputs
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expert_weights = expert_weights
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    validate_reference_moe_weighted_index_add_combine(
        &output,
        &token_indices,
        &expert_outputs,
        &expert_weights,
        token_count,
        hidden_size,
        assignment_count,
    )?;

    for (assignment, token_index) in token_indices.iter().copied().enumerate() {
        let token_index = token_index as usize;
        for hidden in 0..hidden_size {
            let output_index = token_index
                .checked_mul(hidden_size)
                .and_then(|offset| offset.checked_add(hidden))
                .ok_or_else(|| Error::backend("MoE combine output index overflow"))?;
            let expert_index = assignment
                .checked_mul(hidden_size)
                .and_then(|offset| offset.checked_add(hidden))
                .ok_or_else(|| Error::backend("MoE combine expert output index overflow"))?;
            output[output_index] += expert_outputs[expert_index] * expert_weights[assignment];
        }
    }

    Ok(Tensor::from_vec(
        output,
        (token_count, hidden_size),
        device,
    )?)
}

fn reference_attention_scores(
    q: &Tensor,
    k: &Tensor,
    head_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    let q_dims = q.dims();
    let k_dims = k.dims();
    let batch = q_dims[0];
    let heads = q_dims[1];
    let query_tokens = q_dims[2];
    let key_tokens = k_dims[2];
    let q_values = q
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let k_values = k
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    validate_reference_attention_scores(
        &q_values,
        &k_values,
        batch,
        heads,
        query_tokens,
        key_tokens,
        head_dim,
    )?;

    let output_len = batch
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(query_tokens))
        .and_then(|value| value.checked_mul(key_tokens))
        .ok_or_else(|| Error::backend("attention score output length overflow"))?;
    let mut output = vec![0.0_f32; output_len];
    let scale = (head_dim as f32).sqrt();

    for batch_index in 0..batch {
        for head_index in 0..heads {
            for query_index in 0..query_tokens {
                for key_index in 0..key_tokens {
                    let mut sum = 0.0_f32;
                    for dim in 0..head_dim {
                        let q_index = (((batch_index * heads + head_index) * query_tokens
                            + query_index)
                            * head_dim)
                            + dim;
                        let k_index = (((batch_index * heads + head_index) * key_tokens
                            + key_index)
                            * head_dim)
                            + dim;
                        sum += q_values[q_index] * k_values[k_index];
                    }
                    let output_index = (((batch_index * heads + head_index) * query_tokens
                        + query_index)
                        * key_tokens)
                        + key_index;
                    output[output_index] = sum / scale;
                }
            }
        }
    }

    Ok(Tensor::from_vec(
        output,
        (batch, heads, query_tokens, key_tokens),
        device,
    )?)
}

fn validate_reference_attention_scores(
    q: &[f32],
    k: &[f32],
    batch: usize,
    heads: usize,
    query_tokens: usize,
    key_tokens: usize,
    head_dim: usize,
) -> Result<()> {
    if batch == 0 || heads == 0 || query_tokens == 0 || key_tokens == 0 || head_dim == 0 {
        return Err(Error::backend(
            "attention batch, heads, tokens, and head_dim must be positive",
        ));
    }
    let expected_q_len = batch
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(query_tokens))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| Error::backend("attention q value count overflow"))?;
    validate_exact_shape("attention_q_values", &[q.len()], &[expected_q_len])?;
    let expected_k_len = batch
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(key_tokens))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| Error::backend("attention k value count overflow"))?;
    validate_exact_shape("attention_k_values", &[k.len()], &[expected_k_len])?;
    debug_assert_finite_values("attention q", q);
    debug_assert_finite_values("attention k", k);
    Ok(())
}

fn reference_attention_values(probs: &Tensor, values: &Tensor, device: &Device) -> Result<Tensor> {
    let probs_dims = probs.dims();
    let value_dims = values.dims();
    let batch = probs_dims[0];
    let heads = probs_dims[1];
    let query_tokens = probs_dims[2];
    let key_tokens = probs_dims[3];
    let value_dim = value_dims[3];
    let probs_values = probs
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let value_values = values
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    validate_reference_attention_values(
        &probs_values,
        &value_values,
        batch,
        heads,
        query_tokens,
        key_tokens,
        value_dim,
    )?;

    let output_len = batch
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(query_tokens))
        .and_then(|value| value.checked_mul(value_dim))
        .ok_or_else(|| Error::backend("attention values output length overflow"))?;
    let mut output = vec![0.0_f32; output_len];

    for batch_index in 0..batch {
        for head_index in 0..heads {
            for query_index in 0..query_tokens {
                for value_index in 0..value_dim {
                    let mut sum = 0.0_f32;
                    for key_index in 0..key_tokens {
                        let probs_index = (((batch_index * heads + head_index) * query_tokens
                            + query_index)
                            * key_tokens)
                            + key_index;
                        let value_tensor_index =
                            (((batch_index * heads + head_index) * key_tokens + key_index)
                                * value_dim)
                                + value_index;
                        sum += probs_values[probs_index] * value_values[value_tensor_index];
                    }
                    let output_index = (((batch_index * heads + head_index) * query_tokens
                        + query_index)
                        * value_dim)
                        + value_index;
                    output[output_index] = sum;
                }
            }
        }
    }

    Ok(Tensor::from_vec(
        output,
        (batch, heads, query_tokens, value_dim),
        device,
    )?)
}

fn validate_reference_attention_values(
    probs: &[f32],
    values: &[f32],
    batch: usize,
    heads: usize,
    query_tokens: usize,
    key_tokens: usize,
    value_dim: usize,
) -> Result<()> {
    if batch == 0 || heads == 0 || query_tokens == 0 || key_tokens == 0 || value_dim == 0 {
        return Err(Error::backend(
            "attention value aggregation dimensions must be positive",
        ));
    }
    let expected_probs_len = batch
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(query_tokens))
        .and_then(|value| value.checked_mul(key_tokens))
        .ok_or_else(|| Error::backend("attention probabilities value count overflow"))?;
    validate_exact_shape(
        "attention_probabilities_values",
        &[probs.len()],
        &[expected_probs_len],
    )?;
    let expected_values_len = batch
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(key_tokens))
        .and_then(|value| value.checked_mul(value_dim))
        .ok_or_else(|| Error::backend("attention value tensor value count overflow"))?;
    validate_exact_shape(
        "attention_value_tensor_values",
        &[values.len()],
        &[expected_values_len],
    )?;
    debug_assert_finite_values("attention probabilities", probs);
    debug_assert_finite_values("attention values", values);
    Ok(())
}

fn validate_reference_attention_causal_softmax(
    scores: &[f32],
    batch: usize,
    heads: usize,
    query_tokens: usize,
    key_tokens: usize,
    past_tokens: usize,
) -> Result<()> {
    if batch == 0 || heads == 0 || query_tokens == 0 || key_tokens == 0 {
        return Err(Error::backend(
            "attention causal softmax dimensions must be positive",
        ));
    }
    if past_tokens
        .checked_add(query_tokens)
        .filter(|expected_key_tokens| *expected_key_tokens == key_tokens)
        .is_none()
    {
        return Err(Error::backend(format!(
            "attention causal softmax expects key_tokens == past_tokens + query_tokens, got key_tokens={key_tokens}, past_tokens={past_tokens}, query_tokens={query_tokens}"
        )));
    }
    let expected_scores_len = batch
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(query_tokens))
        .and_then(|value| value.checked_mul(key_tokens))
        .ok_or_else(|| Error::backend("attention causal softmax score value count overflow"))?;
    validate_exact_shape(
        "attention_causal_softmax_scores",
        &[scores.len()],
        &[expected_scores_len],
    )?;
    debug_assert_finite_values("attention causal softmax scores", scores);
    Ok(())
}

fn validate_reference_rope_slice(
    input: &[f32],
    batch: usize,
    tokens: usize,
    heads: usize,
    rope_dim: usize,
    position_offset: usize,
    theta: f32,
) -> Result<()> {
    if batch == 0 || tokens == 0 || heads == 0 || rope_dim == 0 {
        return Err(Error::backend("RoPE dimensions must be positive"));
    }
    if rope_dim % 2 != 0 {
        return Err(Error::backend(format!(
            "RoPE rope_dim must be even, got {rope_dim}"
        )));
    }
    if !theta.is_finite() || theta <= 0.0 {
        return Err(Error::backend(
            "RoPE theta must be finite and greater than zero",
        ));
    }
    position_offset
        .checked_add(tokens - 1)
        .ok_or_else(|| Error::backend("RoPE position range overflow"))?;
    let expected_input_len = batch
        .checked_mul(tokens)
        .and_then(|value| value.checked_mul(heads))
        .and_then(|value| value.checked_mul(rope_dim))
        .ok_or_else(|| Error::backend("RoPE input value count overflow"))?;
    validate_exact_shape("rope_input_values", &[input.len()], &[expected_input_len])?;
    debug_assert_finite_values("RoPE input", input);
    Ok(())
}

fn validate_reference_select_last_token(
    hidden_states: &[f32],
    batch: usize,
    tokens: usize,
    hidden_size: usize,
) -> Result<()> {
    if batch == 0 || tokens == 0 || hidden_size == 0 {
        return Err(Error::backend(
            "select_last_token batch, tokens, and hidden_size must be positive",
        ));
    }
    let expected_len = batch
        .checked_mul(tokens)
        .and_then(|value| value.checked_mul(hidden_size))
        .ok_or_else(|| Error::backend("select_last_token input value count overflow"))?;
    validate_exact_shape(
        "select_last_token_values",
        &[hidden_states.len()],
        &[expected_len],
    )?;
    debug_assert_finite_values("select_last_token input", hidden_states);
    Ok(())
}

fn validate_reference_layout_values(name: &str, values: &[f32], dims: &[usize]) -> Result<()> {
    if dims.iter().any(|dim| *dim == 0) {
        return Err(Error::backend(format!(
            "{name} dimensions must be positive, got {dims:?}"
        )));
    }
    let expected_len = dims.iter().try_fold(1_usize, |accumulator, dim| {
        accumulator
            .checked_mul(*dim)
            .ok_or_else(|| Error::backend(format!("{name} input value count overflow")))
    })?;
    validate_exact_shape(name, &[values.len()], &[expected_len])?;
    debug_assert_finite_values(name, values);
    Ok(())
}

fn validate_reference_moe_gather_tokens(
    flat_tokens: &[f32],
    token_indices: &[u32],
    token_count: usize,
    hidden_size: usize,
    assignment_count: usize,
) -> Result<()> {
    if token_count == 0 || hidden_size == 0 || assignment_count == 0 {
        return Err(Error::backend(
            "MoE gather token_count, hidden_size, and assignment_count must be positive",
        ));
    }
    let expected_flat_len = token_count
        .checked_mul(hidden_size)
        .ok_or_else(|| Error::backend("MoE gather flat token value count overflow"))?;
    validate_exact_shape(
        "moe_gather_flat_token_values",
        &[flat_tokens.len()],
        &[expected_flat_len],
    )?;
    validate_exact_shape(
        "moe_gather_token_indices",
        &[token_indices.len()],
        &[assignment_count],
    )?;
    debug_assert_finite_values("MoE gather flat_tokens", flat_tokens);
    if let Some(token_index) = token_indices
        .iter()
        .copied()
        .find(|token_index| *token_index as usize >= token_count)
    {
        return Err(Error::backend(format!(
            "MoE gather token index {token_index} is outside token_count {token_count}"
        )));
    }
    Ok(())
}

fn validate_reference_moe_weighted_index_add_combine(
    accumulator: &[f32],
    token_indices: &[u32],
    expert_outputs: &[f32],
    expert_weights: &[f32],
    token_count: usize,
    hidden_size: usize,
    assignment_count: usize,
) -> Result<()> {
    if token_count == 0 || hidden_size == 0 || assignment_count == 0 {
        return Err(Error::backend(
            "MoE combine token_count, hidden_size, and assignment_count must be positive",
        ));
    }
    let expected_accumulator_len = token_count
        .checked_mul(hidden_size)
        .ok_or_else(|| Error::backend("MoE combine accumulator value count overflow"))?;
    validate_exact_shape(
        "moe_combine_accumulator_values",
        &[accumulator.len()],
        &[expected_accumulator_len],
    )?;
    validate_exact_shape(
        "moe_combine_token_indices_values",
        &[token_indices.len()],
        &[assignment_count],
    )?;
    validate_exact_shape(
        "moe_combine_expert_weight_values",
        &[expert_weights.len()],
        &[assignment_count],
    )?;
    let expected_expert_output_len = assignment_count
        .checked_mul(hidden_size)
        .ok_or_else(|| Error::backend("MoE combine expert output value count overflow"))?;
    validate_exact_shape(
        "moe_combine_expert_output_values",
        &[expert_outputs.len()],
        &[expected_expert_output_len],
    )?;
    debug_assert_finite_values("MoE combine accumulator", accumulator);
    debug_assert_finite_values("MoE combine expert_outputs", expert_outputs);
    debug_assert_finite_values("MoE combine expert_weights", expert_weights);
    if let Some(token_index) = token_indices
        .iter()
        .copied()
        .find(|token_index| *token_index as usize >= token_count)
    {
        return Err(Error::backend(format!(
            "MoE combine token index {token_index} is outside token_count {token_count}"
        )));
    }

    Ok(())
}

fn tensor_from_values(values: Vec<f32>, output_shape: &[usize], device: &Device) -> Result<Tensor> {
    if output_shape.is_empty() {
        return Err(Error::backend("tensor output shape is empty"));
    }
    Ok(Tensor::from_vec(values, output_shape, device)?)
}

fn debug_assert_finite_values(name: &str, values: &[f32]) {
    debug_assert!(
        values.iter().all(|value| value.is_finite()),
        "{name} contains non-finite values"
    );
}

#[cfg(test)]
fn shape(name: &str, tensor: &Tensor) -> NamedShape {
    NamedShape {
        name: name.to_string(),
        shape: Shape::new(tensor.dims().to_vec()),
    }
}

#[cfg(test)]
fn tensor_checksum(tensor: &Tensor) -> Result<f32> {
    Ok(tensor
        .to_dtype(common::DType::F32)?
        .sum_all()?
        .to_vec0::<f32>()?)
}

pub(crate) fn device_kind(device: &Device) -> DeviceKind {
    match device {
        Device::Cpu => DeviceKind::Cpu,
        Device::Metal => DeviceKind::Metal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_backend() -> MetalBackend {
        MetalBackend::from_device(Device::Cpu).unwrap()
    }

    fn q8_rows_payload(rows: &[(f32, &[i8])]) -> Vec<u8> {
        let value_count = rows.first().map(|(_, values)| values.len()).unwrap_or(0);
        let mut payload = Vec::with_capacity(rows.len() * (4 + value_count));
        for (scale, values) in rows {
            payload.extend(scale.to_le_bytes());
            for value in *values {
                payload.push(*value as u8);
            }
        }
        payload
    }

    #[test]
    fn capabilities_report_reference_without_custom_kernels() {
        let backend = cpu_backend();
        let capabilities = backend.capabilities();

        assert_eq!(capabilities.backend, BackendKind::Reference);
        assert_eq!(capabilities.device, DeviceKind::Cpu);
        assert!(!capabilities.custom_kernels);
        assert!(capabilities.operations.contains(&"attention_scores"));
    }

    #[test]
    fn dsa_decode_topk_device_matches_expected_ranking_when_metal_available() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let hidden = F32Tensor::new(vec![2.0_f32.sqrt(), 0.0, 0.0, 0.0], [1, 1, 4]).unwrap();
        let q_raw = F32Tensor::new(vec![1.0, 0.0, 0.0, 1.0], [1, 1, 4]).unwrap();
        let past_keys = F32Tensor::new(
            vec![
                1.0, 0.0, //
                0.0, 1.0, //
                2.0, 0.0,
            ],
            [1, 3, 2],
        )
        .unwrap();
        let current_key = F32Tensor::new(vec![0.0, 2.0], [1, 1, 2]).unwrap();
        let weights_proj = F32Tensor::new(
            vec![
                1.0, 1.0, //
                0.0, 0.0, //
                0.0, 0.0, //
                0.0, 0.0,
            ],
            [4, 2],
        )
        .unwrap();

        let hidden = backend
            .device_upload_f32_tensor(&hidden)
            .unwrap()
            .expect("Metal device value");
        let q_raw = backend
            .device_upload_f32_tensor(&q_raw)
            .unwrap()
            .expect("Metal device value");
        let past_keys = backend
            .device_upload_f32_tensor(&past_keys)
            .unwrap()
            .expect("Metal device value");
        let current_key = backend
            .device_upload_f32_tensor(&current_key)
            .unwrap()
            .expect("Metal device value");

        let topk = backend
            .dsa_decode_topk_device(
                &hidden,
                &q_raw,
                &past_keys,
                &current_key,
                &weights_proj,
                2,
                2,
                2,
                0,
                10_000.0,
                2,
            )
            .unwrap()
            .expect("Metal DSA top-k");

        assert_eq!(topk, vec![2, 3]);
    }

    #[test]
    fn dsa_decode_topk_device_merges_multiple_score_blocks() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let batch = 2;
        let past_tokens = 300;
        let top_k = 260;
        let hidden = F32Tensor::new(vec![1.0; batch], [batch, 1, 1]).unwrap();
        let q_raw = F32Tensor::new(vec![1.0, 0.0, 1.0, 0.0], [batch, 1, 2]).unwrap();
        let mut past_values = Vec::with_capacity(batch * past_tokens * 2);
        for _ in 0..past_tokens {
            past_values.push(1.0);
            past_values.push(0.0);
        }
        for token in 0..past_tokens {
            past_values.push((token + 1) as f32);
            past_values.push(0.0);
        }
        let past_keys = F32Tensor::new(past_values, [batch, past_tokens, 2]).unwrap();
        let current_key = F32Tensor::new(vec![1.0, 0.0, 1_000.0, 0.0], [batch, 1, 2]).unwrap();
        let weights_proj = F32Tensor::new(vec![1.0], [1, 1]).unwrap();

        let hidden = backend
            .device_upload_f32_tensor(&hidden)
            .unwrap()
            .expect("Metal device value");
        let q_raw = backend
            .device_upload_f32_tensor(&q_raw)
            .unwrap()
            .expect("Metal device value");
        let past_keys = backend
            .device_upload_f32_tensor(&past_keys)
            .unwrap()
            .expect("Metal device value");
        let current_key = backend
            .device_upload_f32_tensor(&current_key)
            .unwrap()
            .expect("Metal device value");

        let actual = backend
            .dsa_decode_topk_device(
                &hidden,
                &q_raw,
                &past_keys,
                &current_key,
                &weights_proj,
                1,
                2,
                2,
                0,
                10_000.0,
                top_k,
            )
            .unwrap()
            .expect("Metal DSA top-k");
        let expected_tied = (0..top_k as u32).collect::<Vec<_>>();
        let expected_ranked = std::iter::once(past_tokens as u32)
            .chain((41_u32..past_tokens as u32).rev())
            .collect::<Vec<_>>();

        assert_eq!(&actual[..top_k], expected_tied);
        assert_eq!(&actual[top_k..], expected_ranked);
    }

    #[test]
    fn q8_selected_kv_device_decodes_to_f32_when_metal_available() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let key_payload = q8_rows_payload(&[(0.5, &[2_i8, -2]), (1.0, &[3, -3])]);
        let value_payload = q8_rows_payload(&[(0.25, &[4_i8, -4]), (2.0, &[1, -1])]);

        let Some(view) = backend
            .q8_row_selected_kv_device(&key_payload, &value_payload, 1, 1, 2, 2, 2)
            .unwrap()
        else {
            return;
        };

        assert_eq!(view.k.dtype(), DType::F32);
        assert_eq!(view.v.dtype(), DType::F32);
        assert_eq!(view.k.dims(), &[1, 1, 2, 2]);
        assert_eq!(view.v.dims(), &[1, 1, 2, 2]);
        assert_eq!(
            backend
                .device_download_f32_tensor(&view.k)
                .unwrap()
                .values(),
            vec![1.0, -1.0, 3.0, -3.0]
        );
        assert_eq!(
            backend
                .device_download_f32_tensor(&view.v)
                .unwrap()
                .values(),
            vec![1.0, -1.0, 2.0, -2.0]
        );
    }

    #[test]
    fn matmul_reference_validates_and_runs() {
        let backend = cpu_backend();
        let lhs = Tensor::from_vec(
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (2, 3),
            backend.device(),
        )
        .unwrap();
        let rhs = Tensor::from_vec(
            vec![0.5_f32, 1.0, -1.0, 1.5, -0.5, 0.25],
            (3, 2),
            backend.device(),
        )
        .unwrap();

        let output = backend.matmul(&lhs, &rhs).unwrap();

        assert_eq!(output.dims(), &[2, 2]);
    }

    #[test]
    fn linear_reference_validates_and_runs() {
        let backend = cpu_backend();
        let input = Tensor::from_vec(
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (2, 3),
            backend.device(),
        )
        .unwrap();
        let weight = Tensor::from_vec(
            vec![0.5_f32, 1.0, -1.0, 1.5, -0.5, 0.25],
            (2, 3),
            backend.device(),
        )
        .unwrap();

        let output = backend.linear(&input, &weight).unwrap();

        assert_eq!(output.dims(), &[2, 2]);
        let values = output.to_vec2::<f32>().unwrap();
        assert!((values[0][0] - -0.5).abs() < 1e-6);
        assert!((values[0][1] - 1.25).abs() < 1e-6);
    }

    #[test]
    fn add_reference_validates_and_runs() {
        let backend = cpu_backend();
        let lhs =
            Tensor::from_vec(vec![1.0_f32, -2.0, 0.5, 4.0], (2, 2), backend.device()).unwrap();
        let rhs =
            Tensor::from_vec(vec![0.25_f32, 2.5, -1.5, -0.75], (2, 2), backend.device()).unwrap();

        let output = backend.add(&lhs, &rhs).unwrap();

        assert_eq!(output.dims(), &[2, 2]);
        let values = output.to_vec2::<f32>().unwrap();
        assert_eq!(values[0], vec![1.25, 0.5]);
        assert_eq!(values[1], vec![-1.0, 3.25]);
    }

    #[test]
    fn select_last_token_reference_validates_and_runs() {
        let backend = cpu_backend();
        let hidden_states = Tensor::from_vec(
            vec![
                1.0_f32, 2.0, 3.0, 4.0, //
                5.0, 6.0, 7.0, 8.0, //
                9.0, 10.0, 11.0, 12.0,
            ],
            (1, 3, 4),
            backend.device(),
        )
        .unwrap();

        let output = backend.select_last_token(&hidden_states).unwrap();

        assert_eq!(output.dims(), &[1, 1, 4]);
        let values = output.to_vec3::<f32>().unwrap();
        assert_eq!(values[0][0], vec![9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn heads_to_attention_layout_reference_validates_and_runs() {
        let backend = cpu_backend();
        let heads = Tensor::from_vec(
            (0..1 * 3 * 2 * 2).map(|idx| idx as f32).collect::<Vec<_>>(),
            (1, 3, 2, 2),
            backend.device(),
        )
        .unwrap();

        let output = backend.heads_to_attention_layout(&heads).unwrap();

        assert_eq!(output.dims(), &[1, 2, 3, 2]);
        let values = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            values,
            vec![0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 2.0, 3.0, 6.0, 7.0, 10.0, 11.0]
        );
    }

    #[test]
    fn merge_attention_heads_reference_validates_and_runs() {
        let backend = cpu_backend();
        let context_heads = Tensor::from_vec(
            vec![
                0.0_f32, 1.0, 4.0, 5.0, 8.0, 9.0, //
                2.0, 3.0, 6.0, 7.0, 10.0, 11.0,
            ],
            (1, 2, 3, 2),
            backend.device(),
        )
        .unwrap();

        let output = backend.merge_attention_heads(&context_heads).unwrap();

        assert_eq!(output.dims(), &[1, 3, 4]);
        let values = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            values,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        );
    }

    #[test]
    fn split_rope_tail_reference_validates_and_runs() {
        let backend = cpu_backend();
        let heads = Tensor::from_vec(
            (0..1 * 2 * 2 * 5).map(|idx| idx as f32).collect::<Vec<_>>(),
            (1, 2, 2, 5),
            backend.device(),
        )
        .unwrap();

        let (no_rope, rope) = backend.split_rope_tail(&heads, 3, 2).unwrap();

        assert_eq!(no_rope.dims(), &[1, 2, 2, 3]);
        assert_eq!(rope.dims(), &[1, 2, 2, 2]);
        let no_rope_values = no_rope.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rope_values = rope.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            no_rope_values,
            vec![0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 10.0, 11.0, 12.0, 15.0, 16.0, 17.0]
        );
        assert_eq!(
            rope_values,
            vec![3.0, 4.0, 8.0, 9.0, 13.0, 14.0, 18.0, 19.0]
        );
    }

    #[test]
    fn split_kv_mqa_reference_validates_and_runs() {
        let backend = cpu_backend();
        let kv_mqa = Tensor::from_vec(
            (0..1 * 3 * 6).map(|idx| idx as f32).collect::<Vec<_>>(),
            (1, 3, 6),
            backend.device(),
        )
        .unwrap();

        let (kv_latent, k_rope) = backend.split_kv_mqa(&kv_mqa, 4, 2).unwrap();

        assert_eq!(kv_latent.dims(), &[1, 3, 4]);
        assert_eq!(k_rope.dims(), &[1, 3, 1, 2]);
        let latent_values = kv_latent.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let rope_values = k_rope.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            latent_values,
            vec![0.0, 1.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 12.0, 13.0, 14.0, 15.0]
        );
        assert_eq!(rope_values, vec![4.0, 5.0, 10.0, 11.0, 16.0, 17.0]);
    }

    #[test]
    fn combine_rope_tail_reference_broadcasts_shared_rope_head() {
        let backend = cpu_backend();
        let no_rope = Tensor::from_vec(
            vec![
                0.0_f32, 1.0, //
                2.0, 3.0, //
                4.0, 5.0,
            ],
            (1, 1, 3, 2),
            backend.device(),
        )
        .unwrap();
        let rope =
            Tensor::from_vec(vec![100.0_f32, 101.0], (1, 1, 1, 2), backend.device()).unwrap();

        let combined = backend.combine_rope_tail(&no_rope, &rope).unwrap();

        assert_eq!(combined.dims(), &[1, 1, 3, 4]);
        let values = combined.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            values,
            vec![0.0, 1.0, 100.0, 101.0, 2.0, 3.0, 100.0, 101.0, 4.0, 5.0, 100.0, 101.0]
        );
    }

    #[test]
    fn swiglu_reference_validates_and_runs() {
        let backend = cpu_backend();
        let gate =
            Tensor::from_vec(vec![-1.0_f32, 0.0, 2.0, 4.0], (2, 2), backend.device()).unwrap();
        let up =
            Tensor::from_vec(vec![0.25_f32, 0.5, 1.5, -0.75], (2, 2), backend.device()).unwrap();

        let output = backend.swiglu(&gate, &up).unwrap();

        assert_eq!(output.dims(), &[2, 2]);
        let values = output.to_vec2::<f32>().unwrap();
        let expected = 2.0_f32 / (1.0 + (-2.0_f32).exp()) * 1.5;
        assert!((values[1][0] - expected).abs() < 1e-6);
    }

    #[test]
    fn attention_scores_reference_shape_is_bhqk() {
        let backend = cpu_backend();
        let q =
            Tensor::from_vec(vec![1.0_f32; 1 * 2 * 3 * 4], (1, 2, 3, 4), backend.device()).unwrap();
        let k =
            Tensor::from_vec(vec![0.5_f32; 1 * 2 * 5 * 4], (1, 2, 5, 4), backend.device()).unwrap();

        let scores = backend.attention_scores(&q, &k, 4).unwrap();

        assert_eq!(scores.dims(), &[1, 2, 3, 5]);
    }

    #[test]
    fn attention_values_reference_shape_is_bhqv() {
        let backend = cpu_backend();
        let probs =
            Tensor::from_vec(vec![0.2_f32; 1 * 2 * 3 * 5], (1, 2, 3, 5), backend.device()).unwrap();
        let values =
            Tensor::from_vec(vec![0.5_f32; 1 * 2 * 5 * 4], (1, 2, 5, 4), backend.device()).unwrap();

        let context = backend.attention_values(&probs, &values).unwrap();

        assert_eq!(context.dims(), &[1, 2, 3, 4]);
    }

    #[test]
    fn attention_causal_softmax_masks_future_keys() {
        let backend = cpu_backend();
        let scores = Tensor::from_vec(vec![0.0_f32; 9], (1, 1, 3, 3), backend.device()).unwrap();

        let probabilities = backend.attention_causal_softmax(&scores, 0).unwrap();
        let values = probabilities
            .reshape((3, 3))
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        assert!(values[0][0] > 0.999);
        assert!(values[0][1] < 0.000001);
        assert!(values[0][2] < 0.000001);
        assert!((values[1][0] - 0.5).abs() < 0.000001);
        assert!((values[1][1] - 0.5).abs() < 0.000001);
        assert!(values[1][2] < 0.000001);
        assert!((values[2][0] - (1.0 / 3.0)).abs() < 0.000001);
        assert!((values[2][1] - (1.0 / 3.0)).abs() < 0.000001);
        assert!((values[2][2] - (1.0 / 3.0)).abs() < 0.000001);
    }

    #[test]
    fn rope_slice_reference_uses_position_offset_and_theta() {
        let backend = cpu_backend();
        let input = Tensor::new(vec![1.0_f32; 4], (1, 1, 1, 4)).unwrap();

        let at_zero = backend.rope_slice(&input, 4, 0, 10_000.0).unwrap();
        let at_four = backend.rope_slice(&input, 4, 4, 10_000.0).unwrap();
        let different_theta = backend.rope_slice(&input, 4, 4, 10_000_000.0).unwrap();

        let zero_values = at_zero.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let four_values = at_four.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let different_theta_values = different_theta
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_ne!(zero_values, four_values);
        assert_ne!(four_values, different_theta_values);
    }

    #[test]
    fn rope_slice_reference_rotates_adjacent_glm_pairs() {
        let backend = cpu_backend();
        let input = Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], (1, 1, 1, 4)).unwrap();

        let output = backend.rope_slice(&input, 4, 1, 10_000.0).unwrap();
        let actual = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let slow_angle = 1.0_f32 / 10_000.0_f32.sqrt();
        let expected = [
            1.0 * 1.0_f32.cos() - 2.0 * 1.0_f32.sin(),
            2.0 * 1.0_f32.cos() + 1.0 * 1.0_f32.sin(),
            3.0 * slow_angle.cos() - 4.0 * slow_angle.sin(),
            4.0 * slow_angle.cos() + 3.0 * slow_angle.sin(),
        ];

        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn rms_norm_reference_preserves_shape() {
        let backend = cpu_backend();
        let hidden_states = Tensor::from_vec(
            vec![1.0_f32, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
            (2, 4),
            backend.device(),
        )
        .unwrap();
        let weight = Tensor::from_vec(vec![1.0_f32, 1.1, 0.9, 1.2], 4, backend.device()).unwrap();

        let output = backend.rms_norm(&hidden_states, &weight, 1e-5).unwrap();

        assert_eq!(output.dims(), &[2, 4]);
    }

    #[test]
    fn rms_norm_f32_reference_preserves_shape_and_values() {
        let backend = cpu_backend();
        let hidden_states =
            F32Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0], [2, 4]).unwrap();
        let weight = F32Tensor::new(vec![1.0_f32, 1.1, 0.9, 1.2], [4]).unwrap();

        let output = backend.rms_norm_f32(&hidden_states, &weight, 1e-5).unwrap();

        assert_eq!(output.dims(), &[2, 4]);
        assert_eq!(output.values().len(), 8);
        assert!(output.values().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn moe_gather_tokens_selects_rows_with_repeats() {
        let backend = cpu_backend();
        let flat_tokens = Tensor::from_vec(
            vec![
                1.0_f32, 2.0, //
                3.0, 4.0, //
                5.0, 6.0,
            ],
            (3, 2),
            backend.device(),
        )
        .unwrap();

        let gathered = backend.moe_gather_tokens(&flat_tokens, &[2, 0, 2]).unwrap();

        assert_eq!(gathered.dims(), &[3, 2]);
        let values = gathered.to_vec2::<f32>().unwrap();
        assert_eq!(values[0], vec![5.0, 6.0]);
        assert_eq!(values[1], vec![1.0, 2.0]);
        assert_eq!(values[2], vec![5.0, 6.0]);
    }

    #[test]
    fn moe_weighted_index_add_combine_accumulates_repeated_token_indices() {
        let backend = cpu_backend();
        let accumulator = Tensor::zeros((4, 2)).unwrap();
        let token_indices = Tensor::from_vec(vec![0_u32, 2, 2], 3, backend.device()).unwrap();
        let expert_outputs = Tensor::from_vec(
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (3, 2),
            backend.device(),
        )
        .unwrap();
        let expert_weights =
            Tensor::from_vec(vec![1.0_f32, 0.5, 2.0], 3, backend.device()).unwrap();

        let combined = backend
            .moe_weighted_index_add_combine(
                &accumulator,
                &token_indices,
                &expert_outputs,
                &expert_weights,
            )
            .unwrap();

        assert_eq!(combined.dims(), &[4, 2]);
        let values = combined.to_vec2::<f32>().unwrap();
        assert_eq!(values[0], vec![1.0, 2.0]);
        assert_eq!(values[2], vec![11.5, 14.0]);
    }

    #[test]
    fn backend_check_runs_all_reference_ops() {
        let backend = cpu_backend();
        let report = run_backend_check_with(&backend).unwrap();

        assert_eq!(report.operations.len(), 17);
        assert_eq!(report.operations[0].name, "linear");
        assert_eq!(report.operations[1].name, "matmul");
        assert_eq!(report.operations[2].name, "add");
        assert_eq!(report.operations[2].output.dims(), &[2, 3]);
        assert_eq!(report.operations[3].name, "select_last_token");
        assert_eq!(report.operations[3].output.dims(), &[2, 1, 4]);
        assert_eq!(report.operations[4].name, "heads_to_attention_layout");
        assert_eq!(report.operations[4].output.dims(), &[1, 2, 3, 4]);
        assert_eq!(report.operations[5].name, "merge_attention_heads");
        assert_eq!(report.operations[5].output.dims(), &[1, 3, 8]);
        assert_eq!(report.operations[6].name, "split_rope_tail");
        assert_eq!(report.operations[6].output.dims(), &[1, 2, 2, 2]);
        assert_eq!(report.operations[7].name, "split_kv_mqa");
        assert_eq!(report.operations[7].output.dims(), &[1, 2, 1, 2]);
        assert_eq!(report.operations[8].name, "combine_rope_tail");
        assert_eq!(report.operations[8].output.dims(), &[1, 2, 2, 5]);
        assert_eq!(report.operations[9].name, "swiglu");
        assert_eq!(report.operations[9].output.dims(), &[2, 3]);
        assert_eq!(report.operations[10].output.dims(), &[1, 2, 3, 5]);
        assert_eq!(report.operations[11].name, "attention_causal_softmax");
        assert_eq!(report.operations[11].output.dims(), &[1, 2, 3, 5]);
        assert_eq!(report.operations[12].name, "attention_values");
        assert_eq!(report.operations[12].output.dims(), &[1, 2, 3, 4]);
        assert_eq!(report.operations[13].name, "rope_slice");
        assert_eq!(report.operations[13].output.dims(), &[1, 2, 2, 4]);
        assert_eq!(report.operations[15].name, "moe_gather_tokens");
        assert_eq!(report.operations[15].output.dims(), &[3, 4]);
    }

    #[test]
    fn expert_cache_hit_rate_uses_unique_weight_lookups() {
        let metrics = ExpertCacheMetrics {
            lookups: 10,
            hits: 7,
            misses: 3,
            ..ExpertCacheMetrics::default()
        };

        assert_eq!(metrics.hit_rate(), 0.7);
    }
}
