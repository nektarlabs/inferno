// F16 variant of Laguna's flash-style grouped-query attention. The shared
// constants and `laguna_softplus` helper are declared by the FP8 kernel unit,
// which is concatenated immediately before this source at library compile.
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
    uint query_base = row * LAGUNA_HEAD_DIM;
    float score_scale = rsqrt(float(LAGUNA_HEAD_DIM));

    threadgroup float tile_scores[LAGUNA_ATTENTION_SIMDGROUPS];
    threadgroup float tile_weights[LAGUNA_ATTENTION_SIMDGROUPS];
    threadgroup float running_max_shared;
    threadgroup float running_sum_shared;
    threadgroup float previous_rescale_shared;

    if (tid == 0u) {
        running_max_shared = LAGUNA_NEGATIVE_INFINITY;
        running_sum_shared = 0.0f;
        previous_rescale_shared = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float accumulator = 0.0f;
    for (uint tile_start = 0u; tile_start < key_count;
         tile_start += LAGUNA_ATTENTION_SIMDGROUPS) {
        uint logical_key = tile_start + simd_group;
        float local_dot = 0.0f;
        if (logical_key < key_count) {
            bool from_cache = logical_key < past_count;
            uint cached_slot = 0u;
            uint current_token = 0u;
            if (from_cache) {
                uint absolute_key = past_start + logical_key;
                cached_slot = absolute_key % cache_capacity;
            } else {
                current_token = current_start + logical_key - past_count;
            }

            for (uint dim = simd_lane; dim < LAGUNA_HEAD_DIM; dim += 32u) {
                float key_value;
                if (from_cache) {
                    uint cache_index = (((batch * cache_capacity + cached_slot)
                        * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM) + dim;
                    key_value = float(cached_key[cache_index]);
                } else {
                    uint current_index = (((batch * query_tokens + current_token)
                        * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM) + dim;
                    key_value = current_key[current_index];
                }
                local_dot += query[query_base + dim] * key_value;
            }
        }

        float dot = simd_sum(local_dot);
        if (simd_lane == 0u) {
            tile_scores[simd_group] = logical_key < key_count
                ? dot * score_scale
                : LAGUNA_NEGATIVE_INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            uint tile_count = min(LAGUNA_ATTENTION_SIMDGROUPS, key_count - tile_start);
            float tile_max = tile_scores[0];
            for (uint index = 1u; index < tile_count; index++) {
                tile_max = max(tile_max, tile_scores[index]);
            }
            float next_max = max(running_max_shared, tile_max);
            float previous_rescale = running_sum_shared == 0.0f
                ? 0.0f
                : exp(running_max_shared - next_max);
            float tile_sum = 0.0f;
            for (uint index = 0u; index < tile_count; index++) {
                float weight = exp(tile_scores[index] - next_max);
                tile_weights[index] = weight;
                tile_sum += weight;
            }
            previous_rescale_shared = previous_rescale;
            running_sum_shared = running_sum_shared * previous_rescale + tile_sum;
            running_max_shared = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < LAGUNA_HEAD_DIM) {
            accumulator *= previous_rescale_shared;
            uint tile_count = min(LAGUNA_ATTENTION_SIMDGROUPS, key_count - tile_start);
            for (uint index = 0u; index < tile_count; index++) {
                uint logical_value = tile_start + index;
                bool from_cache = logical_value < past_count;
                float value;
                if (from_cache) {
                    uint absolute_key = past_start + logical_value;
                    uint cached_slot = absolute_key % cache_capacity;
                    uint cache_index = (((batch * cache_capacity + cached_slot)
                        * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM) + tid;
                    value = float(cached_value[cache_index]);
                } else {
                    uint current_token = current_start + logical_value - past_count;
                    uint current_index = (((batch * query_tokens + current_token)
                        * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM) + tid;
                    value = current_value[current_index];
                }
                accumulator += tile_weights[index] * value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < LAGUNA_HEAD_DIM) {
        uint gate_index = ((batch * query_tokens + query_token) * query_heads) + query_head;
        output[query_base + tid] =
            accumulator / running_sum_shared * laguna_softplus(gate[gate_index]);

        // One query head owns the append for each grouped K/V head. This is
        // enabled only while the destination slots are unused, so no other
        // attention threadgroup can still be reading data that we overwrite.
        if (fuse_append != 0u && query_head % heads_per_kv == 0u) {
            uint absolute_token = past_tokens + query_token;
            uint cache_slot = absolute_token % cache_capacity;
            uint current_index = (((batch * query_tokens + query_token)
                * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM) + tid;
            uint cache_index = (((batch * cache_capacity + cache_slot)
                * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM) + tid;
            cached_key[cache_index] = half(current_key[current_index]);
            cached_value[cache_index] = half(current_value[current_index]);
        }
    }
}

// Decode-time variant of the kernel above, for a single query token.
//
// The general kernel walks the key range in tiles only as wide as the
// threadgroup's simdgroup count, which costs three threadgroup barriers and a
// single-threaded softmax update per eight keys. Prefill supplies thousands of
// query rows to hide that, but decode issues one query token per layer, so the
// serial sections dominate instead.
//
// Here each simdgroup takes a strided slice of the key range and runs its own
// online softmax to completion; the threadgroup synchronises once at the end to
// merge the partial softmaxes. Lane `l` owns head dimensions `[4l, 4l+4)`
// throughout, so the query, keys and values are read four wide.
kernel void laguna_gated_gqa_f16_decode_attention_f32_kernel(
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
    if (row >= batch_count * query_heads) {
        return;
    }

    uint query_head = row % query_heads;
    uint batch = row / query_heads;
    uint heads_per_kv = query_heads / LAGUNA_KV_HEADS;
    uint kv_head = query_head / heads_per_kv;

    // The lone query token sits at absolute position `past_tokens` and always
    // attends to itself, so the visible range is the retained past plus one.
    uint visible_start = sliding_window == 0u || past_tokens + 1u <= sliding_window
        ? 0u
        : past_tokens + 1u - sliding_window;
    uint cached_start = past_tokens - stored_tokens;
    uint past_start = min(past_tokens, max(visible_start, cached_start));
    uint past_count = past_tokens - past_start;
    uint key_count = past_count + 1u;

    uint query_base = row * LAGUNA_HEAD_DIM;
    uint dim_base = simd_lane * 4u;
    float4 query_vector =
        *reinterpret_cast<const device float4*>(query + query_base + dim_base);
    float score_scale = rsqrt(float(LAGUNA_HEAD_DIM));
    uint current_index =
        ((batch * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM) + dim_base;

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
            key_vector = *reinterpret_cast<const device float4*>(
                current_key + current_index);
            value_vector = *reinterpret_cast<const device float4*>(
                current_value + current_index);
        }

        float score = simd_sum(dot(query_vector, key_vector)) * score_scale;
        float next_max = max(running_max, score);
        // The first key of a slice has nothing to rescale, and `running_sum`
        // stays exactly zero until then.
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

    // One lane per output float4 merges the per-simdgroup softmaxes. Slices
    // that saw no keys carry a zero sum and are skipped, which also keeps their
    // negative-infinity maximum out of the merged maximum.
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

        uint gate_index = (batch * query_heads) + query_head;
        float gain = laguna_softplus(gate[gate_index]) / merged_sum;
        *reinterpret_cast<device float4*>(output + query_base + tid * 4u) =
            merged * gain;

        // One query head owns the append for each grouped K/V head, matching
        // the general kernel. The current token is read straight from
        // `current_key`, never from the cache, so this write races with no read.
        if (fuse_append != 0u && query_head % heads_per_kv == 0u) {
            uint cache_slot = past_tokens % cache_capacity;
            uint append_source =
                ((batch * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM) + tid * 4u;
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
    uint absolute_token = past_tokens + current_token;
    uint cache_slot = absolute_token % cache_capacity;
    uint current_index = (((batch * query_tokens + current_token)
        * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM) + dim;
    uint cache_index = (((batch * cache_capacity + cache_slot)
        * LAGUNA_KV_HEADS + kv_head) * LAGUNA_HEAD_DIM) + dim;

    cached_key[cache_index] = half(current_key[current_index]);
    cached_value[cache_index] = half(current_value[current_index]);
}
