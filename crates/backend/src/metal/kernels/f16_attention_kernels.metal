// Flash-style grouped-query attention for Laguna's F16 KV cache.
//
// One threadgroup owns one [batch, query-token, query-head] row. Its eight
// SIMD groups walk disjoint key ranges and maintain independent online-softmax
// states. One barrier is needed before the eight partial states are merged.
// Lane `l` owns head dimensions [4l, 4l+4), so all Q/K/V traffic uses float4
// or half4 vector loads.
kernel void laguna_gated_gqa_f16_attention_f32_kernel(
    const device float* query [[buffer(0)]],
    const device float* current_key [[buffer(1)]],
    const device float* current_value [[buffer(2)]],
    const device float* gate [[buffer(3)]],
    device half* cached_key [[buffer(4)]],
    device half* cached_value [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& batch_count [[buffer(7)]],
    constant uint& query_heads [[buffer(8)]],
    constant uint& query_tokens [[buffer(9)]],
    constant uint& past_tokens [[buffer(10)]],
    constant uint& stored_tokens [[buffer(11)]],
    constant uint& cache_capacity [[buffer(12)]],
    constant uint& sliding_window [[buffer(13)]],
    constant uint& fuse_append [[buffer(14)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row_count = batch_count * query_tokens * query_heads;
    if (row >= row_count) {
        return;
    }

    uint query_head = row % query_heads;
    uint query_token = (row / query_heads) % query_tokens;
    uint batch = row / (query_heads * query_tokens);
    uint heads_per_kv = query_heads / LAGUNA_KV_HEADS;
    uint kv_head = query_head / heads_per_kv;
    uint absolute_query = past_tokens + query_token;

    uint visible_start = sliding_window == 0u || absolute_query + 1u <= sliding_window
        ? 0u
        : absolute_query + 1u - sliding_window;
    uint cached_start = past_tokens - stored_tokens;
    uint past_start = min(past_tokens, max(visible_start, cached_start));
    uint past_count = past_tokens - past_start;
    uint current_start = visible_start > past_tokens ? visible_start - past_tokens : 0u;
    uint current_count = query_token - current_start + 1u;
    uint key_count = past_count + current_count;

    uint query_base = (((batch * query_tokens + query_token) * query_heads
        + query_head) * LAGUNA_HEAD_DIM);
    uint dim_base = simd_lane * 4u;
    float4 query_vector =
        *reinterpret_cast<const device float4*>(query + query_base + dim_base);
    float score_scale = rsqrt(float(LAGUNA_HEAD_DIM));

    float running_max = LAGUNA_NEGATIVE_INFINITY;
    float running_sum = 0.0f;
    float4 accumulator = float4(0.0f);

    for (uint logical_key = simd_group; logical_key < key_count;
         logical_key += LAGUNA_ATTENTION_SIMDGROUPS) {
        float4 key_vector;
        float4 value_vector;
        if (logical_key < past_count) {
            uint cached_slot = (past_start + logical_key) % cache_capacity;
            uint cache_index = (((batch * cache_capacity + cached_slot)
                * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM) + dim_base;
            key_vector = float4(
                *reinterpret_cast<const device half4*>(cached_key + cache_index));
            value_vector = float4(
                *reinterpret_cast<const device half4*>(cached_value + cache_index));
        } else {
            uint current_token = current_start + logical_key - past_count;
            uint current_index = (((batch * query_tokens + current_token)
                * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM) + dim_base;
            key_vector = *reinterpret_cast<const device float4*>(
                current_key + current_index);
            value_vector = *reinterpret_cast<const device float4*>(
                current_value + current_index);
        }

        float score = simd_sum(dot(query_vector, key_vector)) * score_scale;
        float next_max = max(running_max, score);
        float rescale = running_sum == 0.0f ? 0.0f : exp(running_max - next_max);
        float weight = exp(score - next_max);
        running_sum = running_sum * rescale + weight;
        running_max = next_max;
        accumulator = accumulator * rescale + weight * value_vector;
    }

    threadgroup float partial_max[LAGUNA_ATTENTION_SIMDGROUPS];
    threadgroup float partial_sum[LAGUNA_ATTENTION_SIMDGROUPS];
    threadgroup float4 partial_accumulator[LAGUNA_ATTENTION_SIMDGROUPS * 32u];
    if (simd_lane == 0u) {
        partial_max[simd_group] = running_max;
        partial_sum[simd_group] = running_sum;
    }
    partial_accumulator[simd_group * 32u + simd_lane] = accumulator;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 32u) {
        float merged_max = LAGUNA_NEGATIVE_INFINITY;
        for (uint slice = 0u; slice < LAGUNA_ATTENTION_SIMDGROUPS; slice++) {
            if (partial_sum[slice] > 0.0f) {
                merged_max = max(merged_max, partial_max[slice]);
            }
        }

        float merged_sum = 0.0f;
        float4 merged = float4(0.0f);
        for (uint slice = 0u; slice < LAGUNA_ATTENTION_SIMDGROUPS; slice++) {
            if (partial_sum[slice] <= 0.0f) {
                continue;
            }
            float rescale = exp(partial_max[slice] - merged_max);
            merged_sum += partial_sum[slice] * rescale;
            merged += partial_accumulator[slice * 32u + tid] * rescale;
        }

        uint gate_index =
            ((batch * query_tokens + query_token) * query_heads) + query_head;
        float gain = laguna_softplus(gate[gate_index]) / merged_sum;
        *reinterpret_cast<device float4*>(output + query_base + tid * 4u) =
            merged * gain;

        // Current-chunk values are always read from the projection buffers,
        // never through the cache. Appending them here therefore races with no
        // attention read. One grouped-query head owns each K/V head.
        if (fuse_append != 0u && query_head % heads_per_kv == 0u) {
            uint cache_slot = absolute_query % cache_capacity;
            uint append_source = (((batch * query_tokens + query_token)
                * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM) + tid * 4u;
            uint append_target = (((batch * cache_capacity + cache_slot)
                * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM) + tid * 4u;
            *reinterpret_cast<device half4*>(cached_key + append_target) =
                half4(*reinterpret_cast<const device float4*>(
                    current_key + append_source));
            *reinterpret_cast<device half4*>(cached_value + append_target) =
                half4(*reinterpret_cast<const device float4*>(
                    current_value + append_source));
        }
    }
}

kernel void laguna_f16_kv_append_f32_kernel(
    const device float* current_key [[buffer(0)]],
    const device float* current_value [[buffer(1)]],
    device half* cached_key [[buffer(2)]],
    device half* cached_value [[buffer(3)]],
    constant uint& batch_count [[buffer(4)]],
    constant uint& query_tokens [[buffer(5)]],
    constant uint& past_tokens [[buffer(6)]],
    constant uint& cache_capacity [[buffer(7)]],
    constant uint& retained_current_tokens [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_count = batch_count * retained_current_tokens
        * LAGUNA_KV_HEADS * LAGUNA_HEAD_DIM;
    if (gid >= output_count) {
        return;
    }

    uint dim = gid % LAGUNA_HEAD_DIM;
    uint kv_head = (gid / LAGUNA_HEAD_DIM) % LAGUNA_KV_HEADS;
    uint retained_token = (gid / (LAGUNA_HEAD_DIM * LAGUNA_KV_HEADS))
        % retained_current_tokens;
    uint batch = gid / (LAGUNA_HEAD_DIM * LAGUNA_KV_HEADS * retained_current_tokens);
    uint current_start = query_tokens - retained_current_tokens;
    uint current_token = current_start + retained_token;
    uint cache_slot = (past_tokens + current_token) % cache_capacity;
    uint source_index = (((batch * query_tokens + current_token)
        * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM) + dim;
    uint target_index = (((batch * cache_capacity + cache_slot)
        * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM) + dim;
    cached_key[target_index] = half(current_key[source_index]);
    cached_value[target_index] = half(current_value[source_index]);
}
