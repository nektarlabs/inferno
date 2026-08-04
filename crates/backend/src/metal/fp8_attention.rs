use ::metal::{Buffer, CommandBufferRef, ComputePipelineState, Device};
use common::{Error, Result};
use tracing::trace;

use crate::{LagunaFp8KvCache, LagunaKvRetention};

use super::{
    arena::MetalArena,
    buffers::{empty_u8_buffer, require_byte_capacity, require_f32_capacity},
    command::{encode_1d, encode_1d_threadgroups, encode_element_copy},
    library::MetalLibrary,
    pipeline::compute_pipeline,
};

const ATTENTION_KERNEL: &str = "laguna_gated_gqa_fp8_attention_f32_kernel";
const APPEND_KERNEL: &str = "laguna_fp8_kv_append_f32_kernel";
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const GLOBAL_QUERY_HEADS: usize = 48;
const SLIDING_QUERY_HEADS: usize = 72;
const SLIDING_WINDOW: usize = 512;
const ATTENTION_THREADS: usize = 256;

pub(crate) struct MetalFp8Attention {
    attention_pipeline: ComputePipelineState,
    append_pipeline: ComputePipelineState,
    arena: MetalArena,
}

impl MetalFp8Attention {
    pub(crate) fn new(device: &Device, library: &MetalLibrary, arena: MetalArena) -> Result<Self> {
        let attention_pipeline = compute_pipeline(device, library, ATTENTION_KERNEL)?;
        let append_pipeline = compute_pipeline(device, library, APPEND_KERNEL)?;
        let simd_width = attention_pipeline.thread_execution_width() as usize;
        if simd_width != 32 {
            return Err(Error::backend(format!(
                "Laguna FP8 attention requires a 32-lane SIMD group, device reports {simd_width}"
            )));
        }
        if (attention_pipeline.max_total_threads_per_threadgroup() as usize) < ATTENTION_THREADS {
            return Err(Error::backend(format!(
                "Laguna FP8 attention requires {ATTENTION_THREADS} threads per group"
            )));
        }
        Ok(Self {
            attention_pipeline,
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
        key_scale: f32,
        value_scale: f32,
    ) -> Result<LagunaFp8KvCache> {
        validate_cache_configuration(batch, capacity_tokens, retention, key_scale, value_scale)?;
        let values = cache_value_count(batch, capacity_tokens)?;
        Ok(LagunaFp8KvCache {
            batch,
            capacity_tokens,
            stored_tokens: 0,
            total_tokens: 0,
            retention,
            key_scale,
            value_scale,
            key: empty_u8_buffer(device, values)?,
            value: empty_u8_buffer(device, values)?,
        })
    }

    pub(crate) fn grow_cache(
        &self,
        device: &Device,
        command_buffer: &CommandBufferRef,
        cache: &mut LagunaFp8KvCache,
        capacity_tokens: usize,
    ) -> Result<()> {
        if cache.retention != LagunaKvRetention::Full {
            return Err(Error::cache(
                "Laguna sliding FP8 KV caches have fixed capacity and cannot grow",
            ));
        }
        if capacity_tokens <= cache.capacity_tokens {
            return Err(Error::cache(format!(
                "Laguna FP8 KV growth requires capacity above {}, got {capacity_tokens}",
                cache.capacity_tokens
            )));
        }

        let mut grown = self.prepare_cache(
            device,
            cache.batch,
            capacity_tokens,
            cache.retention,
            cache.key_scale,
            cache.value_scale,
        )?;
        let stored_values = cache_value_count(cache.batch, cache.stored_tokens)?;
        if stored_values > 0 {
            encode_element_copy(
                command_buffer,
                &cache.key,
                0,
                &grown.key,
                0,
                stored_values,
                std::mem::size_of::<u8>(),
            )?;
            encode_element_copy(
                command_buffer,
                &cache.value,
                0,
                &grown.value,
                0,
                stored_values,
                std::mem::size_of::<u8>(),
            )?;
        }
        grown.stored_tokens = cache.stored_tokens;
        grown.total_tokens = cache.total_tokens;

        trace!(
            target: "inferno::metal",
            previous_capacity = cache.capacity_tokens,
            capacity_tokens,
            stored_tokens = cache.stored_tokens,
            "growing Laguna full-attention FP8 KV cache"
        );
        *cache = grown;
        Ok(())
    }

    /// Encodes attention first and cache append second into the same command
    /// buffer. The ordering is required for a full sliding ring: attention
    /// must read the oldest retained rows before append overwrites their slots.
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
        cache: &LagunaFp8KvCache,
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
        let key_scale_buffer = self.arena.f32(cache.key_scale)?;
        let value_scale_buffer = self.arena.f32(cache.value_scale)?;
        let row_count = batch
            .checked_mul(query_tokens)
            .and_then(|rows| rows.checked_mul(query_heads))
            .ok_or_else(|| Error::backend("Laguna attention row count overflow"))?;

        trace!(
            target: "inferno::metal",
            batch,
            query_tokens,
            query_heads,
            past_tokens = cache.total_tokens,
            stored_tokens = cache.stored_tokens,
            cache_capacity = cache.capacity_tokens,
            ?cache.retention,
            "encoding fused Laguna FP8 grouped-query attention"
        );

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
                &key_scale_buffer,
                &value_scale_buffer,
            ],
            row_count,
            ATTENTION_THREADS,
        )?;

        let retained_current_tokens = query_tokens.min(cache.capacity_tokens);
        let retained_current_buffer = self.arena.u32(as_u32(
            retained_current_tokens,
            "retained current token count",
        )?)?;
        let append_threads = batch
            .checked_mul(retained_current_tokens)
            .and_then(|values| values.checked_mul(KV_HEADS))
            .and_then(|values| values.checked_mul(HEAD_DIM))
            .ok_or_else(|| Error::backend("Laguna FP8 KV append thread count overflow"))?;
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
                &key_scale_buffer,
                &value_scale_buffer,
            ],
            append_threads,
        )?;
        Ok(output)
    }
}

fn validate_cache_configuration(
    batch: usize,
    capacity_tokens: usize,
    retention: LagunaKvRetention,
    key_scale: f32,
    value_scale: f32,
) -> Result<()> {
    if batch == 0 || capacity_tokens == 0 {
        return Err(Error::cache(
            "Laguna FP8 KV batch and capacity must be positive",
        ));
    }
    if retention == LagunaKvRetention::Sliding && capacity_tokens != SLIDING_WINDOW {
        return Err(Error::cache(format!(
            "Laguna sliding FP8 KV capacity must be {SLIDING_WINDOW}, got {capacity_tokens}"
        )));
    }
    for (label, scale) in [("key", key_scale), ("value", value_scale)] {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::cache(format!(
                "Laguna FP8 KV {label} scale must be positive and finite, got {scale}"
            )));
        }
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
    cache: &LagunaFp8KvCache,
) -> Result<()> {
    validate_cache_configuration(
        cache.batch,
        cache.capacity_tokens,
        cache.retention,
        cache.key_scale,
        cache.value_scale,
    )?;
    if batch != cache.batch || query_tokens == 0 {
        return Err(Error::cache(format!(
            "Laguna attention/cache batch mismatch or empty query: attention batch={batch}, cache batch={}, tokens={query_tokens}",
            cache.batch
        )));
    }
    let expected_query_heads = match cache.retention {
        LagunaKvRetention::Full => GLOBAL_QUERY_HEADS,
        LagunaKvRetention::Sliding => SLIDING_QUERY_HEADS,
    };
    if query_heads != expected_query_heads {
        return Err(Error::backend(format!(
            "Laguna {:?} attention requires {expected_query_heads} query heads, got {query_heads}",
            cache.retention
        )));
    }
    if cache.stored_tokens > cache.capacity_tokens || cache.stored_tokens > cache.total_tokens {
        return Err(Error::cache("Laguna FP8 KV cache metadata is inconsistent"));
    }
    if cache.retention == LagunaKvRetention::Full {
        if cache.stored_tokens != cache.total_tokens {
            return Err(Error::cache(
                "Laguna full-attention FP8 KV must retain every past token",
            ));
        }
        cache
            .total_tokens
            .checked_add(query_tokens)
            .filter(|total| *total <= cache.capacity_tokens)
            .ok_or_else(|| {
                Error::cache(format!(
                    "Laguna full-attention FP8 KV capacity {} cannot append {} tokens after {}",
                    cache.capacity_tokens, query_tokens, cache.total_tokens
                ))
            })?;
    }

    let expected_query_len = batch
        .checked_mul(query_tokens)
        .and_then(|values| values.checked_mul(query_heads))
        .and_then(|values| values.checked_mul(HEAD_DIM))
        .ok_or_else(|| Error::backend("Laguna query element count overflow"))?;
    let expected_kv_len = batch
        .checked_mul(query_tokens)
        .and_then(|values| values.checked_mul(KV_HEADS))
        .and_then(|values| values.checked_mul(HEAD_DIM))
        .ok_or_else(|| Error::backend("Laguna K/V element count overflow"))?;
    let expected_gate_len = batch
        .checked_mul(query_tokens)
        .and_then(|values| values.checked_mul(query_heads))
        .ok_or_else(|| Error::backend("Laguna gate element count overflow"))?;
    if query_len != expected_query_len
        || current_key_len != expected_kv_len
        || current_value_len != expected_kv_len
        || gate_len != expected_gate_len
    {
        return Err(Error::backend(format!(
            "Laguna attention buffer lengths mismatch: expected q={expected_query_len}, k/v={expected_kv_len}, gate={expected_gate_len}; got q={query_len}, k={current_key_len}, v={current_value_len}, gate={gate_len}"
        )));
    }
    require_f32_capacity(query, query_len, "Laguna query")?;
    require_f32_capacity(current_key, current_key_len, "Laguna current key")?;
    require_f32_capacity(current_value, current_value_len, "Laguna current value")?;
    require_f32_capacity(gate, gate_len, "Laguna attention gate")?;
    let cache_values = cache_value_count(cache.batch, cache.capacity_tokens)?;
    require_byte_capacity(&cache.key, cache_values, "Laguna FP8 key cache")?;
    require_byte_capacity(&cache.value, cache_values, "Laguna FP8 value cache")?;
    Ok(())
}

fn cache_value_count(batch: usize, capacity_tokens: usize) -> Result<usize> {
    batch
        .checked_mul(capacity_tokens)
        .and_then(|values| values.checked_mul(KV_HEADS))
        .and_then(|values| values.checked_mul(HEAD_DIM))
        .ok_or_else(|| Error::cache("Laguna FP8 KV value count overflow"))
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::backend(format!("Laguna FP8 attention {label} exceeds u32")))
}

#[cfg(test)]
mod tests {
    use super::SLIDING_WINDOW;
    use crate::metal::buffers::write_u8_buffer;
    use crate::{Backend, LagunaKvRetention, MetalBackend};
    use common::F32Tensor;

    const HEAD_DIM: usize = 128;
    const KV_HEADS: usize = 8;
    const GLOBAL_HEADS: usize = 48;
    const SLIDING_HEADS: usize = 72;

    #[test]
    fn sliding_cache_requires_exact_checkpoint_window() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let error = backend
            .prepare_laguna_fp8_kv_cache(
                1,
                SLIDING_WINDOW - 1,
                LagunaKvRetention::Sliding,
                0.5,
                0.5,
            )
            .unwrap_err();
        assert!(error.to_string().contains("must be 512"));
    }

    #[test]
    fn sliding_prefill_larger_than_window_keeps_exact_causal_ring() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let prefill_tokens = SLIDING_WINDOW + 1;
        let mut cache = backend
            .prepare_laguna_fp8_kv_cache(1, SLIDING_WINDOW, LagunaKvRetention::Sliding, 1.0, 1.0)
            .unwrap()
            .unwrap();

        let query = upload_zeros(&backend, [1, prefill_tokens, SLIDING_HEADS, HEAD_DIM]);
        let key = upload_zeros(&backend, [1, prefill_tokens, KV_HEADS, HEAD_DIM]);
        let mut values = vec![2.0_f32; prefill_tokens * KV_HEADS * HEAD_DIM];
        values[..KV_HEADS * HEAD_DIM].fill(100.0);
        let value = upload(&backend, values, [1, prefill_tokens, KV_HEADS, HEAD_DIM]);
        let gate = upload_zeros(&backend, [1, prefill_tokens, SLIDING_HEADS]);

        let output = backend
            .laguna_gated_gqa_attention_device(&query, &key, &value, &gate, &mut cache)
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();
        let last_token_offset = (prefill_tokens - 1) * SLIDING_HEADS * HEAD_DIM;
        let expected_prefill = 2.0 * (2.0_f32).ln();
        let actual_prefill = output.values()[last_token_offset];
        assert!(
            (actual_prefill - expected_prefill).abs() < 1e-3,
            "sliding prefill mismatch: actual={actual_prefill}, expected={expected_prefill}, token0={}, token1={}, token511={}",
            output.values()[0],
            output.values()[SLIDING_HEADS * HEAD_DIM],
            output.values()[(SLIDING_WINDOW - 1) * SLIDING_HEADS * HEAD_DIM],
        );
        assert_eq!(cache.total_tokens(), prefill_tokens);
        assert_eq!(cache.stored_tokens(), SLIDING_WINDOW);

        let query = upload_zeros(&backend, [1, 1, SLIDING_HEADS, HEAD_DIM]);
        let key = upload_zeros(&backend, [1, 1, KV_HEADS, HEAD_DIM]);
        let value = upload(
            &backend,
            vec![4.0_f32; KV_HEADS * HEAD_DIM],
            [1, 1, KV_HEADS, HEAD_DIM],
        );
        let gate = upload_zeros(&backend, [1, 1, SLIDING_HEADS]);
        let output = backend
            .laguna_gated_gqa_attention_device(&query, &key, &value, &gate, &mut cache)
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();
        let expected_decode =
            ((SLIDING_WINDOW - 1) as f32 * 2.0 + 4.0) / SLIDING_WINDOW as f32 * (2.0_f32).ln();
        let actual_decode = output.values()[0];
        assert!(
            (actual_decode - expected_decode).abs() < 1e-3,
            "sliding decode mismatch: actual={actual_decode}, expected={expected_decode}"
        );
        assert_eq!(cache.total_tokens(), prefill_tokens + 1);
        assert_eq!(cache.stored_tokens(), SLIDING_WINDOW);
    }

    #[test]
    fn global_attention_batches_causal_prefill_fp8_append_and_decode() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let mut cache = backend
            .prepare_laguna_fp8_kv_cache(1, 4, LagunaKvRetention::Full, 1.0, 1.0)
            .unwrap()
            .unwrap();

        let query = upload_zeros(&backend, [1, 2, GLOBAL_HEADS, HEAD_DIM]);
        let key = upload_zeros(&backend, [1, 2, KV_HEADS, HEAD_DIM]);
        let mut value_data = vec![0.0_f32; 2 * KV_HEADS * HEAD_DIM];
        for token in 0..2 {
            let value = if token == 0 { 1.1 } else { 3.0 };
            let start = token * KV_HEADS * HEAD_DIM;
            value_data[start..start + KV_HEADS * HEAD_DIM].fill(value);
        }
        let value = upload(&backend, value_data, [1, 2, KV_HEADS, HEAD_DIM]);
        let gate = upload_zeros(&backend, [1, 2, GLOBAL_HEADS]);

        let prefill = backend
            .laguna_gated_gqa_attention_device(&query, &key, &value, &gate, &mut cache)
            .unwrap()
            .unwrap();
        let prefill = backend.device_download_f32_tensor(&prefill).unwrap();
        let softplus_zero = 2.0_f32.ln();
        let first_token_values = GLOBAL_HEADS * HEAD_DIM;
        for (index, actual) in prefill.values().iter().enumerate() {
            let expected_value = if index < first_token_values {
                1.1
            } else {
                2.05
            };
            assert_close(*actual, expected_value * softplus_zero, index);
        }
        assert_eq!(cache.total_tokens(), 2);
        assert_eq!(cache.stored_tokens(), 2);

        backend.grow_laguna_fp8_kv_cache(&mut cache, 8).unwrap();
        assert_eq!(cache.capacity_tokens(), 8);
        assert_eq!(cache.total_tokens(), 2);
        assert_eq!(cache.stored_tokens(), 2);

        // Zero queries make all three decode keys equally likely. The two
        // historical values must come from the byte-sized FP8 cache, while
        // the current value is consumed directly as F32. E4M3FN rounds 1.1
        // to 1.125, so this also validates the quantizer rather than only the
        // cache's indexing.
        let query = upload_zeros(&backend, [1, 1, GLOBAL_HEADS, HEAD_DIM]);
        let key = upload_zeros(&backend, [1, 1, KV_HEADS, HEAD_DIM]);
        let value = upload(
            &backend,
            vec![5.0_f32; KV_HEADS * HEAD_DIM],
            [1, 1, KV_HEADS, HEAD_DIM],
        );
        let gate = upload_zeros(&backend, [1, 1, GLOBAL_HEADS]);
        let decode = backend
            .laguna_gated_gqa_attention_device(&query, &key, &value, &gate, &mut cache)
            .unwrap()
            .unwrap();
        let decode = backend.device_download_f32_tensor(&decode).unwrap();
        for (index, actual) in decode.values().iter().enumerate() {
            assert_close(*actual, ((1.125 + 3.0 + 5.0) / 3.0) * softplus_zero, index);
        }
        assert_eq!(cache.total_tokens(), 3);
        assert_eq!(cache.stored_tokens(), 3);
        assert_eq!(cache.storage_bytes().unwrap(), 2 * 8 * KV_HEADS * HEAD_DIM);

        cache.reset();
        assert_eq!(cache.total_tokens(), 0);
        assert_eq!(cache.stored_tokens(), 0);
    }

    #[test]
    fn sliding_attention_reads_old_slots_before_batched_ring_append() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let mut cache = backend
            .prepare_laguna_fp8_kv_cache(1, SLIDING_WINDOW, LagunaKvRetention::Sliding, 1.0, 1.0)
            .unwrap()
            .unwrap();
        let cache_values = SLIDING_WINDOW * KV_HEADS * HEAD_DIM;
        write_u8_buffer(&cache.key, &vec![0_u8; cache_values]).unwrap();
        write_u8_buffer(&cache.value, &vec![0x38_u8; cache_values]).unwrap(); // 1.0
        cache.total_tokens = SLIDING_WINDOW - 1;
        cache.stored_tokens = SLIDING_WINDOW - 1;

        let query = upload_zeros(&backend, [1, 2, 72, HEAD_DIM]);
        let key = upload_zeros(&backend, [1, 2, KV_HEADS, HEAD_DIM]);
        let mut current_values = vec![0.0_f32; 2 * KV_HEADS * HEAD_DIM];
        current_values[..KV_HEADS * HEAD_DIM].fill(3.0);
        current_values[KV_HEADS * HEAD_DIM..].fill(5.0);
        let value = upload(&backend, current_values, [1, 2, KV_HEADS, HEAD_DIM]);
        let gate = upload_zeros(&backend, [1, 2, 72]);
        let output = backend
            .laguna_gated_gqa_attention_device(&query, &key, &value, &gate, &mut cache)
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();
        let token_width = 72 * HEAD_DIM;
        let softplus_zero = 2.0_f32.ln();
        for (index, actual) in output.values().iter().enumerate() {
            let expected = if index < token_width {
                (514.0 / 512.0) * softplus_zero
            } else {
                (518.0 / 512.0) * softplus_zero
            };
            assert_close(*actual, expected, index);
        }
        assert_eq!(cache.total_tokens(), 513);
        assert_eq!(cache.stored_tokens(), 512);

        let query = upload_zeros(&backend, [1, 1, 72, HEAD_DIM]);
        let key = upload_zeros(&backend, [1, 1, KV_HEADS, HEAD_DIM]);
        let value = upload(
            &backend,
            vec![7.0; KV_HEADS * HEAD_DIM],
            [1, 1, KV_HEADS, HEAD_DIM],
        );
        let gate = upload_zeros(&backend, [1, 1, 72]);
        let output = backend
            .laguna_gated_gqa_attention_device(&query, &key, &value, &gate, &mut cache)
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();
        for (index, actual) in output.values().iter().enumerate() {
            assert_close(*actual, (524.0 / 512.0) * softplus_zero, index);
        }
    }

    fn upload_zeros<const N: usize>(
        backend: &MetalBackend,
        dims: [usize; N],
    ) -> crate::DeviceValue {
        let len = dims.iter().product();
        upload(backend, vec![0.0; len], dims)
    }

    fn upload<const N: usize>(
        backend: &MetalBackend,
        values: Vec<f32>,
        dims: [usize; N],
    ) -> crate::DeviceValue {
        let tensor = F32Tensor::new(values, dims).unwrap();
        backend.device_upload_f32_tensor(&tensor).unwrap().unwrap()
    }

    fn assert_close(actual: f32, expected: f32, index: usize) {
        assert!(
            (actual - expected).abs() <= 1e-4,
            "value {index} differs: actual={actual}, expected={expected}"
        );
    }
}
