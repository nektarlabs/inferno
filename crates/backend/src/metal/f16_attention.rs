use ::metal::{Buffer, CommandBufferRef, ComputePipelineState, Device};
use common::{Error, Result};
use tracing::trace;

use crate::{LagunaF16DecodeStrategy, LagunaF16KvCache, LagunaKvRetention};

use super::{
    arena::MetalArena,
    buffers::{empty_f16_buffer, require_f16_capacity, require_f32_capacity},
    command::{
        encode_1d, encode_1d_threadgroups, encode_1d_threadgroups_args, encode_element_copy,
        KernelArg,
    },
    library::MetalLibrary,
    pipeline::compute_pipeline,
};

const ATTENTION_KERNEL: &str = "laguna_gated_gqa_f16_attention_f32_kernel";
const PREFILL_ATTENTION_KERNEL: &str = "laguna_prefill_gated_gqa_f16_attention_f32_kernel";
const SEGMENTED_DECODE_PARTIAL_KERNEL: &str =
    "laguna_segmented_grouped_gqa_f16_decode_partial_f32_kernel";
const SEGMENTED_DECODE_MERGE_KERNEL: &str =
    "laguna_segmented_grouped_gqa_f16_decode_merge_f32_kernel";
const APPEND_KERNEL: &str = "laguna_f16_kv_append_f32_kernel";
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const SEGMENTED_DECODE_QUERY_HEADS: usize = 48;
const SLIDING_QUERY_HEADS: usize = 72;
const MAX_QUERY_HEADS: usize = SLIDING_QUERY_HEADS;
const SLIDING_WINDOW: usize = 512;
const ATTENTION_THREADS: usize = 256;
const PREFILL_MIN_TOKENS: usize = 8;

pub(crate) struct MetalF16Attention {
    attention_pipeline: ComputePipelineState,
    prefill_attention_pipeline: ComputePipelineState,
    segmented_decode_partial_pipeline: ComputePipelineState,
    segmented_decode_merge_pipeline: ComputePipelineState,
    append_pipeline: ComputePipelineState,
    arena: MetalArena,
}

impl MetalF16Attention {
    pub(crate) fn new(device: &Device, library: &MetalLibrary, arena: MetalArena) -> Result<Self> {
        let attention_pipeline = compute_pipeline(device, library, ATTENTION_KERNEL)?;
        let simd_width = attention_pipeline.thread_execution_width() as usize;
        if simd_width != 32
            || (attention_pipeline.max_total_threads_per_threadgroup() as usize) < ATTENTION_THREADS
        {
            return Err(Error::backend(
                "Laguna F16 attention requires 32-lane SIMD groups and 256-thread groups",
            ));
        }
        let prefill_attention_pipeline =
            compute_pipeline(device, library, PREFILL_ATTENTION_KERNEL)?;
        let prefill_max_threads =
            prefill_attention_pipeline.max_total_threads_per_threadgroup() as usize;
        if prefill_attention_pipeline.thread_execution_width() as usize != 32
            || prefill_max_threads < SLIDING_QUERY_HEADS / KV_HEADS * 32
        {
            return Err(Error::backend(
                "Laguna F16 prefill attention requires 32-lane SIMD groups and at least 288 threads per group",
            ));
        }
        let segmented_decode_partial_pipeline =
            compute_pipeline(device, library, SEGMENTED_DECODE_PARTIAL_KERNEL)?;
        if segmented_decode_partial_pipeline.thread_execution_width() as usize != 32
            || (segmented_decode_partial_pipeline.max_total_threads_per_threadgroup() as usize)
                < 6 * 32
        {
            return Err(Error::backend(
                "Laguna segmented F16 decode requires 32-lane SIMD groups and at least 192 threads per group",
            ));
        }
        let segmented_decode_merge_pipeline =
            compute_pipeline(device, library, SEGMENTED_DECODE_MERGE_KERNEL)?;
        if segmented_decode_merge_pipeline.thread_execution_width() as usize != 32
            || (segmented_decode_merge_pipeline.max_total_threads_per_threadgroup() as usize) < 32
        {
            return Err(Error::backend(
                "Laguna segmented F16 decode merge requires one 32-lane SIMD group",
            ));
        }
        let append_pipeline = compute_pipeline(device, library, APPEND_KERNEL)?;
        Ok(Self {
            attention_pipeline,
            prefill_attention_pipeline,
            segmented_decode_partial_pipeline,
            segmented_decode_merge_pipeline,
            append_pipeline,
            arena,
        })
    }

    pub(crate) fn prepare_cache(
        &self,
        device: &Device,
        batch: usize,
        capacity_tokens: usize,
        retention: LagunaKvRetention,
    ) -> Result<LagunaF16KvCache> {
        validate_cache_configuration(batch, capacity_tokens, retention)?;
        let values = cache_value_count(batch, capacity_tokens)?;
        Ok(LagunaF16KvCache {
            batch,
            capacity_tokens,
            stored_tokens: 0,
            total_tokens: 0,
            retention,
            decode_strategy: LagunaF16DecodeStrategy::QueryParallel,
            checkpoint_stored_tokens: None,
            checkpoint_total_tokens: None,
            key: empty_f16_buffer(device, values)?,
            value: empty_f16_buffer(device, values)?,
            checkpoint_key: None,
            checkpoint_value: None,
        })
    }

    pub(crate) fn checkpoint_cache(
        &self,
        device: &Device,
        command_buffer: &CommandBufferRef,
        cache: &mut LagunaF16KvCache,
    ) -> Result<()> {
        validate_cache_configuration(cache.batch, cache.capacity_tokens, cache.retention)?;
        if cache.retention == LagunaKvRetention::Sliding {
            let values = cache_value_count(cache.batch, cache.capacity_tokens)?;
            if cache.checkpoint_key.is_none() || cache.checkpoint_value.is_none() {
                let checkpoint_key = empty_f16_buffer(device, values)?;
                let checkpoint_value = empty_f16_buffer(device, values)?;
                cache.checkpoint_key = Some(checkpoint_key);
                cache.checkpoint_value = Some(checkpoint_value);
            }
            let checkpoint_key = cache
                .checkpoint_key
                .as_ref()
                .ok_or_else(|| Error::cache("Laguna F16 KV checkpoint key is missing"))?;
            let checkpoint_value = cache
                .checkpoint_value
                .as_ref()
                .ok_or_else(|| Error::cache("Laguna F16 KV checkpoint value is missing"))?;
            encode_element_copy(
                command_buffer,
                &cache.key,
                0,
                checkpoint_key,
                0,
                values,
                std::mem::size_of::<u16>(),
            )?;
            encode_element_copy(
                command_buffer,
                &cache.value,
                0,
                checkpoint_value,
                0,
                values,
                std::mem::size_of::<u16>(),
            )?;
        }
        cache.checkpoint_stored_tokens = Some(cache.stored_tokens);
        cache.checkpoint_total_tokens = Some(cache.total_tokens);
        Ok(())
    }

    pub(crate) fn restore_cache_checkpoint(
        &self,
        command_buffer: &CommandBufferRef,
        cache: &mut LagunaF16KvCache,
    ) -> Result<()> {
        let checkpoint_stored_tokens = cache
            .checkpoint_stored_tokens
            .ok_or_else(|| Error::cache("Laguna F16 KV cache has no checkpoint"))?;
        let checkpoint_total_tokens = cache
            .checkpoint_total_tokens
            .ok_or_else(|| Error::cache("Laguna F16 KV cache has no checkpoint"))?;
        if cache.retention == LagunaKvRetention::Sliding {
            let values = cache_value_count(cache.batch, cache.capacity_tokens)?;
            let checkpoint_key = cache
                .checkpoint_key
                .as_ref()
                .ok_or_else(|| Error::cache("Laguna F16 KV checkpoint key is missing"))?;
            let checkpoint_value = cache
                .checkpoint_value
                .as_ref()
                .ok_or_else(|| Error::cache("Laguna F16 KV checkpoint value is missing"))?;
            encode_element_copy(
                command_buffer,
                checkpoint_key,
                0,
                &cache.key,
                0,
                values,
                std::mem::size_of::<u16>(),
            )?;
            encode_element_copy(
                command_buffer,
                checkpoint_value,
                0,
                &cache.value,
                0,
                values,
                std::mem::size_of::<u16>(),
            )?;
        }
        cache.stored_tokens = checkpoint_stored_tokens;
        cache.total_tokens = checkpoint_total_tokens;
        Ok(())
    }

    pub(crate) fn grow_cache(
        &self,
        device: &Device,
        command_buffer: &CommandBufferRef,
        cache: &mut LagunaF16KvCache,
        capacity_tokens: usize,
    ) -> Result<()> {
        if cache.retention != LagunaKvRetention::Full {
            return Err(Error::cache(
                "Laguna sliding F16 KV caches have fixed capacity and cannot grow",
            ));
        }
        if capacity_tokens <= cache.capacity_tokens {
            return Err(Error::cache(format!(
                "Laguna F16 KV growth requires capacity above {}, got {capacity_tokens}",
                cache.capacity_tokens
            )));
        }

        let mut grown =
            self.prepare_cache(device, cache.batch, capacity_tokens, cache.retention)?;
        let stored_values = cache_value_count(cache.batch, cache.stored_tokens)?;
        if stored_values > 0 {
            encode_element_copy(
                command_buffer,
                &cache.key,
                0,
                &grown.key,
                0,
                stored_values,
                std::mem::size_of::<u16>(),
            )?;
            encode_element_copy(
                command_buffer,
                &cache.value,
                0,
                &grown.value,
                0,
                stored_values,
                std::mem::size_of::<u16>(),
            )?;
        }
        grown.stored_tokens = cache.stored_tokens;
        grown.total_tokens = cache.total_tokens;
        grown.decode_strategy = cache.decode_strategy;
        grown.checkpoint_stored_tokens = cache.checkpoint_stored_tokens;
        grown.checkpoint_total_tokens = cache.checkpoint_total_tokens;
        *cache = grown;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_attention_and_append(
        &self,
        command_buffer: &CommandBufferRef,
        query: &Buffer,
        query_len: usize,
        current_key: &Buffer,
        current_key_len: usize,
        current_value: &Buffer,
        current_value_len: usize,
        gate: &Buffer,
        gate_len: usize,
        batch: usize,
        query_tokens: usize,
        query_heads: usize,
        cache: &LagunaF16KvCache,
    ) -> Result<Buffer> {
        validate_execution(
            query,
            query_len,
            current_key,
            current_key_len,
            current_value,
            current_value_len,
            gate,
            gate_len,
            batch,
            query_tokens,
            query_heads,
            cache,
        )?;

        let output = self.arena.empty_f32(query_len)?;
        let fuse_append = cache.stored_tokens == cache.total_tokens
            && cache
                .total_tokens
                .checked_add(query_tokens)
                .is_some_and(|total| total <= cache.capacity_tokens);
        let segmented_decode = match cache.decode_strategy {
            LagunaF16DecodeStrategy::SegmentedGroupedKv {
                min_context_tokens,
                segment_tokens,
            } if query_tokens == 1
                && cache.retention == LagunaKvRetention::Full
                && cache.total_tokens >= min_context_tokens =>
            {
                Some(segment_tokens)
            }
            _ => None,
        };
        if let Some(segment_tokens) = segmented_decode {
            if query_heads != SEGMENTED_DECODE_QUERY_HEADS {
                return Err(Error::backend(format!(
                    "Laguna segmented F16 decode requires {SEGMENTED_DECODE_QUERY_HEADS} query heads, got {query_heads}"
                )));
            }
            if !fuse_append {
                return Err(Error::cache(
                    "Laguna segmented full-attention decode requires fused KV append",
                ));
            }
            self.encode_segmented_grouped_decode(
                command_buffer,
                query,
                current_key,
                current_value,
                gate,
                &output,
                batch,
                query_heads,
                cache,
                segment_tokens,
            )?;
            return Ok(output);
        }

        let batch_buffer = self.arena.u32(as_u32(batch, "batch")?)?;
        let query_heads_buffer = self.arena.u32(as_u32(query_heads, "query head count")?)?;
        let query_tokens_buffer = self.arena.u32(as_u32(query_tokens, "query token count")?)?;
        let past_tokens_buffer = self
            .arena
            .u32(as_u32(cache.total_tokens, "past token count")?)?;
        let stored_tokens_buffer = self
            .arena
            .u32(as_u32(cache.stored_tokens, "stored token count")?)?;
        let capacity_buffer = self
            .arena
            .u32(as_u32(cache.capacity_tokens, "cache capacity")?)?;
        let sliding_window = match cache.retention {
            LagunaKvRetention::Full => 0,
            LagunaKvRetention::Sliding => SLIDING_WINDOW,
        };
        let sliding_window_buffer = self.arena.u32(sliding_window as u32)?;
        let fuse_append_buffer = self.arena.u32(u32::from(fuse_append))?;
        let row_count = batch
            .checked_mul(query_tokens)
            .and_then(|rows| rows.checked_mul(query_heads))
            .ok_or_else(|| Error::backend("Laguna F16 attention row count overflow"))?;

        trace!(
            target: "inferno::metal",
            batch,
            query_tokens,
            query_heads,
            past_tokens = cache.total_tokens,
            stored_tokens = cache.stored_tokens,
            cache_capacity = cache.capacity_tokens,
            fuse_append,
            ?cache.retention,
            "encoding fused Laguna F16 grouped-query attention"
        );

        if query_tokens >= PREFILL_MIN_TOKENS {
            let prefill_rows = batch
                .checked_mul(query_tokens)
                .and_then(|rows| rows.checked_mul(KV_HEADS))
                .ok_or_else(|| Error::backend("Laguna F16 prefill row count overflow"))?;
            let prefill_threads = query_heads / KV_HEADS * 32;
            let args = [
                KernelArg::Buffer(query),
                KernelArg::Buffer(current_key),
                KernelArg::Buffer(current_value),
                KernelArg::Buffer(gate),
                KernelArg::Buffer(&cache.key),
                KernelArg::Buffer(&cache.value),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(batch, "batch")?),
                KernelArg::U32(as_u32(query_heads, "query head count")?),
                KernelArg::U32(as_u32(query_tokens, "query token count")?),
                KernelArg::U32(as_u32(cache.total_tokens, "past token count")?),
                KernelArg::U32(as_u32(cache.stored_tokens, "stored token count")?),
                KernelArg::U32(as_u32(cache.capacity_tokens, "cache capacity")?),
                KernelArg::U32(sliding_window as u32),
                KernelArg::U32(u32::from(fuse_append)),
            ];
            encode_1d_threadgroups_args(
                command_buffer,
                &self.prefill_attention_pipeline,
                &args,
                prefill_rows,
                prefill_threads,
            )?;
        } else {
            encode_1d_threadgroups(
                command_buffer,
                &self.attention_pipeline,
                &[
                    query,
                    current_key,
                    current_value,
                    gate,
                    &cache.key,
                    &cache.value,
                    &output,
                    &batch_buffer,
                    &query_heads_buffer,
                    &query_tokens_buffer,
                    &past_tokens_buffer,
                    &stored_tokens_buffer,
                    &capacity_buffer,
                    &sliding_window_buffer,
                    &fuse_append_buffer,
                ],
                row_count,
                ATTENTION_THREADS,
            )?;
        }

        if !fuse_append {
            let retained_current_tokens = query_tokens.min(cache.capacity_tokens);
            let retained_current_buffer = self.arena.u32(as_u32(
                retained_current_tokens,
                "retained current token count",
            )?)?;
            let append_threads = batch
                .checked_mul(retained_current_tokens)
                .and_then(|values| values.checked_mul(KV_HEADS))
                .and_then(|values| values.checked_mul(HEAD_DIM))
                .ok_or_else(|| Error::backend("Laguna F16 KV append thread count overflow"))?;
            encode_1d(
                command_buffer,
                &self.append_pipeline,
                &[
                    current_key,
                    current_value,
                    &cache.key,
                    &cache.value,
                    &batch_buffer,
                    &query_tokens_buffer,
                    &past_tokens_buffer,
                    &capacity_buffer,
                    &retained_current_buffer,
                ],
                append_threads,
            )?;
        }
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_segmented_grouped_decode(
        &self,
        command_buffer: &CommandBufferRef,
        query: &Buffer,
        current_key: &Buffer,
        current_value: &Buffer,
        gate: &Buffer,
        output: &Buffer,
        batch: usize,
        query_heads: usize,
        cache: &LagunaF16KvCache,
        segment_tokens: usize,
    ) -> Result<()> {
        if segment_tokens == 0 {
            return Err(Error::backend(
                "Laguna segmented F16 decode segment size must be positive",
            ));
        }
        let key_count = cache
            .total_tokens
            .checked_add(1)
            .ok_or_else(|| Error::backend("Laguna segmented decode key count overflow"))?;
        let segment_count = key_count.div_ceil(segment_tokens);
        let partial_rows = batch
            .checked_mul(segment_count)
            .and_then(|rows| rows.checked_mul(query_heads))
            .ok_or_else(|| Error::backend("Laguna segmented decode partial row overflow"))?;
        let partial_max = self.arena.empty_f32(partial_rows)?;
        let partial_sum = self.arena.empty_f32(partial_rows)?;
        let partial_accumulator =
            self.arena
                .empty_f32(partial_rows.checked_mul(HEAD_DIM).ok_or_else(|| {
                    Error::backend("Laguna segmented decode partial size overflow")
                })?)?;
        let partial_threadgroups = batch
            .checked_mul(segment_count)
            .and_then(|groups| groups.checked_mul(KV_HEADS))
            .ok_or_else(|| Error::backend("Laguna segmented decode grid overflow"))?;
        let partial_threads = query_heads / KV_HEADS * 32;

        trace!(
            target: "inferno::metal",
            batch,
            query_heads,
            past_tokens = cache.total_tokens,
            segment_tokens,
            segment_count,
            "encoding segmented grouped-KV Laguna F16 decode"
        );

        let partial_args = [
            KernelArg::Buffer(query),
            KernelArg::Buffer(current_key),
            KernelArg::Buffer(current_value),
            KernelArg::Buffer(&cache.key),
            KernelArg::Buffer(&cache.value),
            KernelArg::Buffer(&partial_max),
            KernelArg::Buffer(&partial_sum),
            KernelArg::Buffer(&partial_accumulator),
            KernelArg::U32(as_u32(batch, "batch")?),
            KernelArg::U32(as_u32(query_heads, "query head count")?),
            KernelArg::U32(as_u32(cache.total_tokens, "past token count")?),
            KernelArg::U32(as_u32(cache.capacity_tokens, "cache capacity")?),
            KernelArg::U32(as_u32(segment_tokens, "segment token count")?),
            KernelArg::U32(as_u32(segment_count, "segment count")?),
        ];
        encode_1d_threadgroups_args(
            command_buffer,
            &self.segmented_decode_partial_pipeline,
            &partial_args,
            partial_threadgroups,
            partial_threads,
        )?;

        let merge_args = [
            KernelArg::Buffer(&partial_max),
            KernelArg::Buffer(&partial_sum),
            KernelArg::Buffer(&partial_accumulator),
            KernelArg::Buffer(gate),
            KernelArg::Buffer(current_key),
            KernelArg::Buffer(current_value),
            KernelArg::Buffer(&cache.key),
            KernelArg::Buffer(&cache.value),
            KernelArg::Buffer(output),
            KernelArg::U32(as_u32(batch, "batch")?),
            KernelArg::U32(as_u32(query_heads, "query head count")?),
            KernelArg::U32(as_u32(cache.total_tokens, "past token count")?),
            KernelArg::U32(as_u32(cache.capacity_tokens, "cache capacity")?),
            KernelArg::U32(as_u32(segment_count, "segment count")?),
        ];
        encode_1d_threadgroups_args(
            command_buffer,
            &self.segmented_decode_merge_pipeline,
            &merge_args,
            batch
                .checked_mul(query_heads)
                .ok_or_else(|| Error::backend("Laguna segmented decode merge grid overflow"))?,
            32,
        )
    }
}

fn validate_cache_configuration(
    batch: usize,
    capacity_tokens: usize,
    retention: LagunaKvRetention,
) -> Result<()> {
    if batch == 0 || capacity_tokens == 0 {
        return Err(Error::cache(
            "Laguna F16 KV batch and capacity must be positive",
        ));
    }
    if retention == LagunaKvRetention::Sliding && capacity_tokens != SLIDING_WINDOW {
        return Err(Error::cache(format!(
            "Laguna sliding F16 KV capacity must be {SLIDING_WINDOW}, got {capacity_tokens}"
        )));
    }
    cache_value_count(batch, capacity_tokens)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_execution(
    query: &Buffer,
    query_len: usize,
    current_key: &Buffer,
    current_key_len: usize,
    current_value: &Buffer,
    current_value_len: usize,
    gate: &Buffer,
    gate_len: usize,
    batch: usize,
    query_tokens: usize,
    query_heads: usize,
    cache: &LagunaF16KvCache,
) -> Result<()> {
    validate_cache_configuration(cache.batch, cache.capacity_tokens, cache.retention)?;
    if batch != cache.batch || query_tokens == 0 {
        return Err(Error::cache(format!(
            "Laguna F16 attention/cache batch mismatch or empty query: attention batch={batch}, cache batch={}, tokens={query_tokens}",
            cache.batch
        )));
    }
    if query_heads == 0 || query_heads > MAX_QUERY_HEADS || !query_heads.is_multiple_of(KV_HEADS) {
        return Err(Error::backend(format!(
            "Laguna F16 attention query-head count must be a positive multiple of {KV_HEADS} up to {MAX_QUERY_HEADS}, got {query_heads}"
        )));
    }
    if cache.stored_tokens > cache.capacity_tokens || cache.stored_tokens > cache.total_tokens {
        return Err(Error::cache("Laguna F16 KV cache metadata is inconsistent"));
    }
    if cache.retention == LagunaKvRetention::Full {
        if cache.stored_tokens != cache.total_tokens {
            return Err(Error::cache(
                "Laguna full-attention F16 KV must retain every past token",
            ));
        }
        cache
            .total_tokens
            .checked_add(query_tokens)
            .filter(|total| *total <= cache.capacity_tokens)
            .ok_or_else(|| {
                Error::cache(format!(
                    "Laguna full-attention F16 KV capacity {} cannot append {} tokens after {}",
                    cache.capacity_tokens, query_tokens, cache.total_tokens
                ))
            })?;
    }

    let expected_query_len = batch
        .checked_mul(query_tokens)
        .and_then(|values| values.checked_mul(query_heads))
        .and_then(|values| values.checked_mul(HEAD_DIM))
        .ok_or_else(|| Error::backend("Laguna F16 query element count overflow"))?;
    let expected_kv_len = batch
        .checked_mul(query_tokens)
        .and_then(|values| values.checked_mul(KV_HEADS))
        .and_then(|values| values.checked_mul(HEAD_DIM))
        .ok_or_else(|| Error::backend("Laguna F16 K/V element count overflow"))?;
    let expected_gate_len = batch
        .checked_mul(query_tokens)
        .and_then(|values| values.checked_mul(query_heads))
        .ok_or_else(|| Error::backend("Laguna F16 gate element count overflow"))?;
    if query_len != expected_query_len
        || current_key_len != expected_kv_len
        || current_value_len != expected_kv_len
        || gate_len != expected_gate_len
    {
        return Err(Error::backend(format!(
            "Laguna F16 attention buffer lengths mismatch: expected q={expected_query_len}, k/v={expected_kv_len}, gate={expected_gate_len}; got q={query_len}, k={current_key_len}, v={current_value_len}, gate={gate_len}"
        )));
    }
    require_f32_capacity(query, query_len, "Laguna F16 query")?;
    require_f32_capacity(current_key, current_key_len, "Laguna F16 current key")?;
    require_f32_capacity(current_value, current_value_len, "Laguna F16 current value")?;
    require_f32_capacity(gate, gate_len, "Laguna F16 attention gate")?;
    let cache_values = cache_value_count(cache.batch, cache.capacity_tokens)?;
    require_f16_capacity(&cache.key, cache_values, "Laguna F16 key cache")?;
    require_f16_capacity(&cache.value, cache_values, "Laguna F16 value cache")?;
    Ok(())
}

fn cache_value_count(batch: usize, capacity_tokens: usize) -> Result<usize> {
    batch
        .checked_mul(capacity_tokens)
        .and_then(|values| values.checked_mul(KV_HEADS))
        .and_then(|values| values.checked_mul(HEAD_DIM))
        .ok_or_else(|| Error::cache("Laguna F16 KV value count overflow"))
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::backend(format!("Laguna F16 attention {label} exceeds u32")))
}

#[cfg(test)]
mod tests {
    use crate::{Backend, LagunaF16DecodeStrategy, LagunaKvRetention, MetalBackend};
    use common::F32Tensor;

    use super::{HEAD_DIM, KV_HEADS, SLIDING_WINDOW};

    const QUERY_HEADS: usize = 48;
    const GLOBAL_QUERY_HEADS: usize = 48;
    const LAGUNA_XS_SLIDING_QUERY_HEADS: usize = 64;
    const SLIDING_QUERY_HEADS: usize = 72;

    #[test]
    fn f16_cache_preserves_prefill_values_for_decode() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let mut cache = backend
            .prepare_laguna_f16_kv_cache(1, 4, LagunaKvRetention::Full)
            .unwrap()
            .unwrap();

        let query = upload(
            &backend,
            vec![0.0; 2 * QUERY_HEADS * HEAD_DIM],
            [1, 2, QUERY_HEADS, HEAD_DIM],
        );
        let key = upload(
            &backend,
            vec![0.0; 2 * KV_HEADS * HEAD_DIM],
            [1, 2, KV_HEADS, HEAD_DIM],
        );
        let mut values = vec![1.0; 2 * KV_HEADS * HEAD_DIM];
        values[KV_HEADS * HEAD_DIM..].fill(3.0);
        let value = upload(&backend, values, [1, 2, KV_HEADS, HEAD_DIM]);
        let gate = upload(&backend, vec![0.0; 2 * QUERY_HEADS], [1, 2, QUERY_HEADS]);
        let output = backend
            .laguna_gated_gqa_f16_attention_device(&query, &key, &value, &gate, &mut cache)
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();
        let softplus_zero = 2.0_f32.ln();
        assert!((output.values()[0] - softplus_zero).abs() <= 1e-4);
        assert!((output.values()[QUERY_HEADS * HEAD_DIM] - 2.0 * softplus_zero).abs() <= 1e-4);

        let query = upload(
            &backend,
            vec![0.0; QUERY_HEADS * HEAD_DIM],
            [1, 1, QUERY_HEADS, HEAD_DIM],
        );
        let key = upload(
            &backend,
            vec![0.0; KV_HEADS * HEAD_DIM],
            [1, 1, KV_HEADS, HEAD_DIM],
        );
        let value = upload(
            &backend,
            vec![5.0; KV_HEADS * HEAD_DIM],
            [1, 1, KV_HEADS, HEAD_DIM],
        );
        let gate = upload(&backend, vec![0.0; QUERY_HEADS], [1, 1, QUERY_HEADS]);
        let output = backend
            .laguna_gated_gqa_f16_attention_device(&query, &key, &value, &gate, &mut cache)
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();
        assert!((output.values()[0] - 3.0 * softplus_zero).abs() <= 1e-4);
        assert_eq!(cache.total_tokens(), 3);
        assert_eq!(
            cache.storage_bytes().unwrap(),
            2 * 4 * KV_HEADS * HEAD_DIM * 2
        );
    }

    #[test]
    fn full_cache_growth_preserves_segmented_decode_strategy() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let strategy = LagunaF16DecodeStrategy::SegmentedGroupedKv {
            min_context_tokens: 6_144,
            segment_tokens: 512,
        };
        let mut cache = backend
            .prepare_laguna_f16_kv_cache(1, 8, LagunaKvRetention::Full)
            .unwrap()
            .unwrap();
        cache.set_decode_strategy(strategy);

        assert!(backend.grow_laguna_f16_kv_cache(&mut cache, 16).unwrap());
        assert_eq!(cache.capacity_tokens(), 16);
        assert_eq!(cache.decode_strategy, strategy);
    }

    #[test]
    fn f16_prefill_matches_nonuniform_cpu_reference() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        const TOKENS: usize = 8;
        let query_values = patterned_values(TOKENS * QUERY_HEADS * HEAD_DIM, 17, 8, 16.0);
        let key_values = patterned_values(TOKENS * KV_HEADS * HEAD_DIM, 13, 6, 16.0);
        let value_values = patterned_values(TOKENS * KV_HEADS * HEAD_DIM, 11, 5, 8.0);
        let gate_values = patterned_values(TOKENS * QUERY_HEADS, 7, 3, 10.0);
        let query = upload(
            &backend,
            query_values.clone(),
            [1, TOKENS, QUERY_HEADS, HEAD_DIM],
        );
        let key = upload(
            &backend,
            key_values.clone(),
            [1, TOKENS, KV_HEADS, HEAD_DIM],
        );
        let value = upload(
            &backend,
            value_values.clone(),
            [1, TOKENS, KV_HEADS, HEAD_DIM],
        );
        let gate = upload(&backend, gate_values.clone(), [1, TOKENS, QUERY_HEADS]);
        let mut cache = backend
            .prepare_laguna_f16_kv_cache(1, TOKENS, LagunaKvRetention::Full)
            .unwrap()
            .unwrap();

        let actual = backend
            .laguna_gated_gqa_f16_attention_device(&query, &key, &value, &gate, &mut cache)
            .unwrap()
            .unwrap();
        let actual = backend.device_download_f32_tensor(&actual).unwrap();
        let expected = cpu_causal_gqa(
            &query_values,
            &key_values,
            &value_values,
            &gate_values,
            TOKENS,
        );

        for (index, (actual, expected)) in actual.values().iter().zip(expected.iter()).enumerate() {
            let tolerance = 5e-4_f32.max(expected.abs() * 5e-4);
            assert!(
                (actual - expected).abs() <= tolerance,
                "Laguna F16 attention mismatch at {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn sliding_f16_cache_checkpoint_restores_overwritten_rows() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let mut cache = backend
            .prepare_laguna_f16_kv_cache(1, SLIDING_WINDOW, LagunaKvRetention::Sliding)
            .unwrap()
            .unwrap();
        let zeros_query = upload(
            &backend,
            vec![0.0; SLIDING_WINDOW * LAGUNA_XS_SLIDING_QUERY_HEADS * HEAD_DIM],
            [1, SLIDING_WINDOW, LAGUNA_XS_SLIDING_QUERY_HEADS, HEAD_DIM],
        );
        let zeros_key = upload(
            &backend,
            vec![0.0; SLIDING_WINDOW * KV_HEADS * HEAD_DIM],
            [1, SLIDING_WINDOW, KV_HEADS, HEAD_DIM],
        );
        let ones_value = upload(
            &backend,
            vec![1.0; SLIDING_WINDOW * KV_HEADS * HEAD_DIM],
            [1, SLIDING_WINDOW, KV_HEADS, HEAD_DIM],
        );
        let zeros_gate = upload(
            &backend,
            vec![0.0; SLIDING_WINDOW * LAGUNA_XS_SLIDING_QUERY_HEADS],
            [1, SLIDING_WINDOW, LAGUNA_XS_SLIDING_QUERY_HEADS],
        );
        drop(
            backend
                .laguna_gated_gqa_f16_attention_device(
                    &zeros_query,
                    &zeros_key,
                    &ones_value,
                    &zeros_gate,
                    &mut cache,
                )
                .unwrap()
                .unwrap(),
        );
        assert!(backend.checkpoint_laguna_f16_kv_cache(&mut cache).unwrap());

        drop(
            backend
                .laguna_gated_gqa_f16_attention_device(
                    &upload(
                        &backend,
                        vec![0.0; LAGUNA_XS_SLIDING_QUERY_HEADS * HEAD_DIM],
                        [1, 1, LAGUNA_XS_SLIDING_QUERY_HEADS, HEAD_DIM],
                    ),
                    &upload(
                        &backend,
                        vec![0.0; KV_HEADS * HEAD_DIM],
                        [1, 1, KV_HEADS, HEAD_DIM],
                    ),
                    &upload(
                        &backend,
                        vec![100.0; KV_HEADS * HEAD_DIM],
                        [1, 1, KV_HEADS, HEAD_DIM],
                    ),
                    &upload(
                        &backend,
                        vec![0.0; LAGUNA_XS_SLIDING_QUERY_HEADS],
                        [1, 1, LAGUNA_XS_SLIDING_QUERY_HEADS],
                    ),
                    &mut cache,
                )
                .unwrap()
                .unwrap(),
        );
        assert!(backend
            .restore_laguna_f16_kv_cache_checkpoint(&mut cache)
            .unwrap());
        assert_eq!(cache.total_tokens(), SLIDING_WINDOW);
        assert_eq!(cache.stored_tokens(), SLIDING_WINDOW);

        let output = backend
            .laguna_gated_gqa_f16_attention_device(
                &upload(
                    &backend,
                    vec![0.0; LAGUNA_XS_SLIDING_QUERY_HEADS * HEAD_DIM],
                    [1, 1, LAGUNA_XS_SLIDING_QUERY_HEADS, HEAD_DIM],
                ),
                &upload(
                    &backend,
                    vec![0.0; KV_HEADS * HEAD_DIM],
                    [1, 1, KV_HEADS, HEAD_DIM],
                ),
                &upload(
                    &backend,
                    vec![1.0; KV_HEADS * HEAD_DIM],
                    [1, 1, KV_HEADS, HEAD_DIM],
                ),
                &upload(
                    &backend,
                    vec![0.0; LAGUNA_XS_SLIDING_QUERY_HEADS],
                    [1, 1, LAGUNA_XS_SLIDING_QUERY_HEADS],
                ),
                &mut cache,
            )
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();
        assert!((output.values()[0] - 2.0_f32.ln()).abs() <= 1e-4);
    }

    /// The sliding case runs past the 512-token window on purpose: that is the
    /// only configuration where a decode step both clamps the visible range and
    /// reads a ring buffer that has wrapped.
    #[test]
    fn f16_decode_step_matches_nonuniform_cpu_reference() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        for (retention, query_heads, tokens, grouped_decode) in [
            (LagunaKvRetention::Full, GLOBAL_QUERY_HEADS, 37, false),
            (LagunaKvRetention::Full, GLOBAL_QUERY_HEADS, 37, true),
            (
                LagunaKvRetention::Sliding,
                LAGUNA_XS_SLIDING_QUERY_HEADS,
                37,
                false,
            ),
            (LagunaKvRetention::Sliding, SLIDING_QUERY_HEADS, 600, false),
        ] {
            let query_values = patterned_values(tokens * query_heads * HEAD_DIM, 17, 8, 16.0);
            let key_values = patterned_values(tokens * KV_HEADS * HEAD_DIM, 13, 6, 16.0);
            let value_values = patterned_values(tokens * KV_HEADS * HEAD_DIM, 11, 5, 8.0);
            let gate_values = patterned_values(tokens * query_heads, 7, 3, 10.0);
            let sliding_window = match retention {
                LagunaKvRetention::Full => 0,
                LagunaKvRetention::Sliding => SLIDING_WINDOW,
            };
            let capacity = match retention {
                LagunaKvRetention::Full => tokens,
                LagunaKvRetention::Sliding => SLIDING_WINDOW,
            };
            let mut cache = backend
                .prepare_laguna_f16_kv_cache(1, capacity, retention)
                .unwrap()
                .unwrap();
            if grouped_decode {
                cache.set_decode_strategy(LagunaF16DecodeStrategy::SegmentedGroupedKv {
                    min_context_tokens: 0,
                    segment_tokens: 16,
                });
            }

            let prefill = tokens - 1;
            drop(
                backend
                    .laguna_gated_gqa_f16_attention_device(
                        &upload(
                            &backend,
                            query_values[..prefill * query_heads * HEAD_DIM].to_vec(),
                            [1, prefill, query_heads, HEAD_DIM],
                        ),
                        &upload(
                            &backend,
                            key_values[..prefill * KV_HEADS * HEAD_DIM].to_vec(),
                            [1, prefill, KV_HEADS, HEAD_DIM],
                        ),
                        &upload(
                            &backend,
                            value_values[..prefill * KV_HEADS * HEAD_DIM].to_vec(),
                            [1, prefill, KV_HEADS, HEAD_DIM],
                        ),
                        &upload(
                            &backend,
                            gate_values[..prefill * query_heads].to_vec(),
                            [1, prefill, query_heads],
                        ),
                        &mut cache,
                    )
                    .unwrap()
                    .unwrap(),
            );

            let decode = backend
                .laguna_gated_gqa_f16_attention_device(
                    &upload(
                        &backend,
                        query_values[prefill * query_heads * HEAD_DIM..].to_vec(),
                        [1, 1, query_heads, HEAD_DIM],
                    ),
                    &upload(
                        &backend,
                        key_values[prefill * KV_HEADS * HEAD_DIM..].to_vec(),
                        [1, 1, KV_HEADS, HEAD_DIM],
                    ),
                    &upload(
                        &backend,
                        value_values[prefill * KV_HEADS * HEAD_DIM..].to_vec(),
                        [1, 1, KV_HEADS, HEAD_DIM],
                    ),
                    &upload(
                        &backend,
                        gate_values[prefill * query_heads..].to_vec(),
                        [1, 1, query_heads],
                    ),
                    &mut cache,
                )
                .unwrap()
                .unwrap();
            let decode = backend.device_download_f32_tensor(&decode).unwrap();

            let expected = cpu_decode_gqa(
                &query_values,
                &key_values,
                &value_values,
                &gate_values,
                tokens,
                query_heads,
                sliding_window,
            );
            // The past keys and values came back through the f16 cache, so the
            // decoded row carries more error than the all-f32 prefill rows.
            for (index, (actual, expected)) in
                decode.values().iter().zip(expected.iter()).enumerate()
            {
                let tolerance = 2e-3_f32.max(expected.abs() * 2e-3);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "Laguna F16 {retention:?} decode mismatch at {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
                );
            }
            assert_eq!(cache.total_tokens(), tokens);
        }
    }

    /// Reference output for the final token only. Computing every row would be
    /// needlessly slow at the 600-token sliding shape.
    fn cpu_decode_gqa(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        gate: &[f32],
        tokens: usize,
        query_heads: usize,
        sliding_window: usize,
    ) -> Vec<f32> {
        let token = tokens - 1;
        let heads_per_kv = query_heads / KV_HEADS;
        let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
        let first_key = if sliding_window == 0 || token + 1 <= sliding_window {
            0
        } else {
            token + 1 - sliding_window
        };
        let mut output = vec![0.0_f32; query_heads * HEAD_DIM];

        for query_head in 0..query_heads {
            let kv_head = query_head / heads_per_kv;
            let query_base = (token * query_heads + query_head) * HEAD_DIM;
            let scores = (first_key..=token)
                .map(|key_token| {
                    let key_base = (key_token * KV_HEADS + kv_head) * HEAD_DIM;
                    query[query_base..query_base + HEAD_DIM]
                        .iter()
                        .zip(&key[key_base..key_base + HEAD_DIM])
                        .map(|(query, key)| query * key)
                        .sum::<f32>()
                        * scale
                })
                .collect::<Vec<_>>();
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores
                .iter()
                .map(|score| (score - maximum).exp())
                .collect::<Vec<_>>();
            let denominator = weights.iter().sum::<f32>();
            let gain = (1.0_f32 + gate[token * query_heads + query_head].exp()).ln();
            for dim in 0..HEAD_DIM {
                let weighted = weights
                    .iter()
                    .enumerate()
                    .map(|(offset, weight)| {
                        let value_index =
                            ((first_key + offset) * KV_HEADS + kv_head) * HEAD_DIM + dim;
                        weight * value[value_index]
                    })
                    .sum::<f32>();
                output[query_head * HEAD_DIM + dim] = weighted / denominator * gain;
            }
        }
        output
    }

    fn patterned_values(len: usize, period: usize, center: usize, divisor: f32) -> Vec<f32> {
        (0..len)
            .map(|index| ((index % period) as f32 - center as f32) / divisor)
            .collect()
    }

    fn cpu_causal_gqa(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        gate: &[f32],
        tokens: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; query.len()];
        let heads_per_kv = QUERY_HEADS / KV_HEADS;
        let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
        for token in 0..tokens {
            for query_head in 0..QUERY_HEADS {
                let kv_head = query_head / heads_per_kv;
                let query_base = (token * QUERY_HEADS + query_head) * HEAD_DIM;
                let scores = (0..=token)
                    .map(|key_token| {
                        let key_base = (key_token * KV_HEADS + kv_head) * HEAD_DIM;
                        query[query_base..query_base + HEAD_DIM]
                            .iter()
                            .zip(&key[key_base..key_base + HEAD_DIM])
                            .map(|(query, key)| query * key)
                            .sum::<f32>()
                            * scale
                    })
                    .collect::<Vec<_>>();
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights = scores
                    .iter()
                    .map(|score| (score - maximum).exp())
                    .collect::<Vec<_>>();
                let denominator = weights.iter().sum::<f32>();
                let gate = (1.0_f32 + gate[token * QUERY_HEADS + query_head].exp()).ln();
                for dim in 0..HEAD_DIM {
                    let weighted = weights
                        .iter()
                        .enumerate()
                        .map(|(key_token, weight)| {
                            let value_index = (key_token * KV_HEADS + kv_head) * HEAD_DIM + dim;
                            weight * value[value_index]
                        })
                        .sum::<f32>();
                    output[query_base + dim] = weighted / denominator * gate;
                }
            }
        }
        output
    }

    fn upload<const N: usize>(
        backend: &MetalBackend,
        values: Vec<f32>,
        dims: [usize; N],
    ) -> crate::DeviceValue {
        backend
            .device_upload_f32_tensor(&F32Tensor::new(values, dims).unwrap())
            .unwrap()
            .unwrap()
    }
}
