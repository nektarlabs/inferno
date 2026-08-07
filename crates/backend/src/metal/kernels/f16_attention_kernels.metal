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

// Prompt-time GQA groups every query head that shares one K/V head into the
// same threadgroup. K/V vectors are loaded once into an eight-key tile and
// reused by all six full-attention heads or all nine sliding-attention heads.
// Decode keeps using the key-parallel kernel above.
constant uint LAGUNA_PREFILL_KV_TILE = 8u;

kernel void laguna_prefill_gated_gqa_f16_attention_f32_kernel(
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
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    uint row_count = batch_count * query_tokens * LAGUNA_KV_HEADS;
    if (row >= row_count) {
        return;
    }

    uint kv_head = row % LAGUNA_KV_HEADS;
    uint query_token = (row / LAGUNA_KV_HEADS) % query_tokens;
    uint batch = row / (LAGUNA_KV_HEADS * query_tokens);
    uint heads_per_kv = query_heads / LAGUNA_KV_HEADS;
    if (simd_group >= heads_per_kv) {
        return;
    }
    uint query_head = kv_head * heads_per_kv + simd_group;
    uint absolute_query = past_tokens + query_token;

    uint visible_start =
        sliding_window == 0u || absolute_query + 1u <= sliding_window
            ? 0u
            : absolute_query + 1u - sliding_window;
    uint cached_start = past_tokens - stored_tokens;
    uint past_start = min(
        past_tokens,
        max(visible_start, cached_start));
    uint past_count = past_tokens - past_start;
    uint current_start =
        visible_start > past_tokens
            ? visible_start - past_tokens
            : 0u;
    uint current_count = query_token - current_start + 1u;
    uint key_count = past_count + current_count;

    uint query_base = (((batch * query_tokens + query_token)
        * query_heads + query_head) * LAGUNA_HEAD_DIM);
    uint lane_dim = simd_lane * 4u;
    float4 query_vector =
        *reinterpret_cast<const device float4*>(
            query + query_base + lane_dim);
    float score_scale = rsqrt(float(LAGUNA_HEAD_DIM));
    float running_max = LAGUNA_NEGATIVE_INFINITY;
    float running_sum = 0.0f;
    float4 accumulator = float4(0.0f);

    threadgroup float4 key_tile[
        LAGUNA_PREFILL_KV_TILE * 32u
    ];
    threadgroup float4 value_tile[
        LAGUNA_PREFILL_KV_TILE * 32u
    ];

    for (uint tile_start = 0u;
         tile_start < key_count;
         tile_start += LAGUNA_PREFILL_KV_TILE) {
        uint tile_count = min(
            LAGUNA_PREFILL_KV_TILE,
            key_count - tile_start);
        uint tile_values = tile_count * 32u;
        for (uint index = tid;
             index < tile_values;
             index += threads_per_group) {
            uint local_key = index / 32u;
            uint lane = index - local_key * 32u;
            uint logical_key = tile_start + local_key;
            float4 key_vector;
            float4 value_vector;
            if (logical_key < past_count) {
                uint cached_slot =
                    (past_start + logical_key) % cache_capacity;
                uint cache_index = (((batch * cache_capacity
                    + cached_slot) * LAGUNA_KV_HEADS + kv_head)
                    * LAGUNA_HEAD_DIM) + lane * 4u;
                key_vector = float4(
                    *reinterpret_cast<const device half4*>(
                        cached_key + cache_index));
                value_vector = float4(
                    *reinterpret_cast<const device half4*>(
                        cached_value + cache_index));
            } else {
                uint current_token = current_start
                    + logical_key
                    - past_count;
                uint current_index = (((batch * query_tokens
                    + current_token) * LAGUNA_KV_HEADS + kv_head)
                    * LAGUNA_HEAD_DIM) + lane * 4u;
                key_vector =
                    *reinterpret_cast<const device float4*>(
                        current_key + current_index);
                value_vector =
                    *reinterpret_cast<const device float4*>(
                        current_value + current_index);
            }
            key_tile[index] = key_vector;
            value_tile[index] = value_vector;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint local_key = 0u;
             local_key < tile_count;
             local_key++) {
            uint tile_index = local_key * 32u + simd_lane;
            float score = simd_sum(dot(
                query_vector,
                key_tile[tile_index])) * score_scale;
            float next_max = max(running_max, score);
            float rescale = running_sum == 0.0f
                ? 0.0f
                : exp(running_max - next_max);
            float weight = exp(score - next_max);
            running_sum = running_sum * rescale + weight;
            running_max = next_max;
            accumulator =
                accumulator * rescale
                + weight * value_tile[tile_index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint gate_index =
        ((batch * query_tokens + query_token) * query_heads)
        + query_head;
    float gain = laguna_softplus(gate[gate_index]) / running_sum;
    *reinterpret_cast<device float4*>(
        output + query_base + lane_dim) = accumulator * gain;

    // One SIMD group owns the append for this token/KV-head pair.
    if (fuse_append != 0u && simd_group == 0u) {
        uint cache_slot = absolute_query % cache_capacity;
        uint append_source = (((batch * query_tokens + query_token)
            * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM)
            + lane_dim;
        uint append_target = (((batch * cache_capacity + cache_slot)
            * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM)
            + lane_dim;
        *reinterpret_cast<device half4*>(
            cached_key + append_target) = half4(
                *reinterpret_cast<const device float4*>(
                    current_key + append_source));
        *reinterpret_cast<device half4*>(
            cached_value + append_target) = half4(
                *reinterpret_cast<const device float4*>(
                    current_value + append_source));
    }
}

// Long-context Laguna XS decode. One threadgroup owns one
// [batch, K/V-head, context-segment] tile. Its SIMD groups represent every
// query head sharing that K/V head, so each cached K/V vector is loaded once
// and reused across six query heads. A second kernel merges segment-local
// online-softmax states without materializing attention scores.
constant uint LAGUNA_SEGMENTED_DECODE_KV_TILE = 8u;

kernel void laguna_segmented_grouped_gqa_f16_decode_partial_f32_kernel(
    const device float* query [[buffer(0)]],
    const device float* current_key [[buffer(1)]],
    const device float* current_value [[buffer(2)]],
    const device half* cached_key [[buffer(3)]],
    const device half* cached_value [[buffer(4)]],
    device float* partial_max [[buffer(5)]],
    device float* partial_sum [[buffer(6)]],
    device float* partial_accumulator [[buffer(7)]],
    constant uint& batch_count [[buffer(8)]],
    constant uint& query_heads [[buffer(9)]],
    constant uint& cache_capacity [[buffer(10)]],
    constant uint& cached_start [[buffer(11)]],
    constant uint& cached_count [[buffer(12)]],
    constant uint& segment_tokens [[buffer(13)]],
    constant uint& segment_count [[buffer(14)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    uint row_count = batch_count * segment_count * LAGUNA_KV_HEADS;
    if (row >= row_count) {
        return;
    }

    uint kv_head = row % LAGUNA_KV_HEADS;
    uint segment = (row / LAGUNA_KV_HEADS) % segment_count;
    uint batch = row / (LAGUNA_KV_HEADS * segment_count);
    uint heads_per_kv = query_heads / LAGUNA_KV_HEADS;
    if (simd_group >= heads_per_kv) {
        return;
    }
    uint query_head = kv_head * heads_per_kv + simd_group;
    uint key_count = cached_count + 1u;
    uint segment_start = segment * segment_tokens;
    uint segment_end = min(key_count, segment_start + segment_tokens);
    uint query_base = ((batch * query_heads + query_head) * LAGUNA_HEAD_DIM);
    uint lane_dim = simd_lane * 4u;
    float4 query_vector = *reinterpret_cast<const device float4*>(
        query + query_base + lane_dim);
    float score_scale = rsqrt(float(LAGUNA_HEAD_DIM));
    float running_max = LAGUNA_NEGATIVE_INFINITY;
    float running_sum = 0.0f;
    float4 accumulator = float4(0.0f);

    threadgroup float4 key_tile[LAGUNA_SEGMENTED_DECODE_KV_TILE * 32u];
    threadgroup float4 value_tile[LAGUNA_SEGMENTED_DECODE_KV_TILE * 32u];

    for (uint tile_start = segment_start;
         tile_start < segment_end;
         tile_start += LAGUNA_SEGMENTED_DECODE_KV_TILE) {
        uint tile_count = min(
            LAGUNA_SEGMENTED_DECODE_KV_TILE,
            segment_end - tile_start);
        uint tile_values = tile_count * 32u;
        for (uint index = tid;
             index < tile_values;
             index += threads_per_group) {
            uint local_key = index / 32u;
            uint lane = index - local_key * 32u;
            uint logical_key = tile_start + local_key;
            float4 key_vector;
            float4 value_vector;
            if (logical_key < cached_count) {
                uint absolute_key = cached_start + logical_key;
                uint cache_slot = absolute_key % cache_capacity;
                uint cache_index = (((batch * cache_capacity + cache_slot)
                    * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM)
                    + lane * 4u;
                key_vector = float4(
                    *reinterpret_cast<const device half4*>(
                        cached_key + cache_index));
                value_vector = float4(
                    *reinterpret_cast<const device half4*>(
                        cached_value + cache_index));
            } else {
                uint current_index = (((batch * LAGUNA_KV_HEADS + kv_head)
                    * LAGUNA_HEAD_DIM) + lane * 4u);
                key_vector = *reinterpret_cast<const device float4*>(
                    current_key + current_index);
                value_vector = *reinterpret_cast<const device float4*>(
                    current_value + current_index);
            }
            key_tile[index] = key_vector;
            value_tile[index] = value_vector;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint local_key = 0u; local_key < tile_count; local_key++) {
            uint tile_index = local_key * 32u + simd_lane;
            float score = simd_sum(dot(
                query_vector,
                key_tile[tile_index])) * score_scale;
            float next_max = max(running_max, score);
            float rescale = running_sum == 0.0f
                ? 0.0f
                : exp(running_max - next_max);
            float weight = exp(score - next_max);
            running_sum = running_sum * rescale + weight;
            running_max = next_max;
            accumulator = accumulator * rescale
                + weight * value_tile[tile_index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint partial_row = ((batch * segment_count + segment) * query_heads)
        + query_head;
    if (simd_lane == 0u) {
        partial_max[partial_row] = running_max;
        partial_sum[partial_row] = running_sum;
    }
    *reinterpret_cast<device float4*>(
        partial_accumulator + partial_row * LAGUNA_HEAD_DIM + lane_dim) =
        accumulator;
}

kernel void laguna_segmented_grouped_gqa_f16_decode_merge_f32_kernel(
    const device float* partial_max [[buffer(0)]],
    const device float* partial_sum [[buffer(1)]],
    const device float* partial_accumulator [[buffer(2)]],
    const device float* gate [[buffer(3)]],
    const device float* current_key [[buffer(4)]],
    const device float* current_value [[buffer(5)]],
    device half* cached_key [[buffer(6)]],
    device half* cached_value [[buffer(7)]],
    device float* output [[buffer(8)]],
    constant uint& batch_count [[buffer(9)]],
    constant uint& query_heads [[buffer(10)]],
    constant uint& past_tokens [[buffer(11)]],
    constant uint& cache_capacity [[buffer(12)]],
    constant uint& segment_count [[buffer(13)]],
    uint row [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint row_count = batch_count * query_heads;
    if (row >= row_count) {
        return;
    }

    uint query_head = row % query_heads;
    uint batch = row / query_heads;
    uint lane_dim = simd_lane * 4u;
    float merged_max = LAGUNA_NEGATIVE_INFINITY;
    float merged_sum = 0.0f;
    float4 merged = float4(0.0f);

    for (uint segment = 0u; segment < segment_count; segment++) {
        uint partial_row = ((batch * segment_count + segment) * query_heads)
            + query_head;
        float segment_max = partial_max[partial_row];
        float segment_sum = partial_sum[partial_row];
        float4 segment_accumulator = *reinterpret_cast<const device float4*>(
            partial_accumulator + partial_row * LAGUNA_HEAD_DIM + lane_dim);
        float next_max = max(merged_max, segment_max);
        float merged_scale = merged_sum == 0.0f
            ? 0.0f
            : exp(merged_max - next_max);
        float segment_scale = exp(segment_max - next_max);
        merged_sum = merged_sum * merged_scale + segment_sum * segment_scale;
        merged = merged * merged_scale + segment_accumulator * segment_scale;
        merged_max = next_max;
    }

    float gain = laguna_softplus(gate[batch * query_heads + query_head])
        / merged_sum;
    uint output_base = ((batch * query_heads + query_head) * LAGUNA_HEAD_DIM)
        + lane_dim;
    *reinterpret_cast<device float4*>(output + output_base) = merged * gain;

    uint heads_per_kv = query_heads / LAGUNA_KV_HEADS;
    if (query_head % heads_per_kv == 0u) {
        uint kv_head = query_head / heads_per_kv;
        uint source = ((batch * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM)
            + lane_dim;
        uint target_slot = past_tokens % cache_capacity;
        uint target = (((batch * cache_capacity + target_slot)
            * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM) + lane_dim;
        *reinterpret_cast<device half4*>(cached_key + target) = half4(
            *reinterpret_cast<const device float4*>(current_key + source));
        *reinterpret_cast<device half4*>(cached_value + target) = half4(
            *reinterpret_cast<const device float4*>(current_value + source));
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
