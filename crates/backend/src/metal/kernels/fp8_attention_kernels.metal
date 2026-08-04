#include <metal_stdlib>

using namespace metal;

constant uint LAGUNA_KV_HEADS = 8u;
constant uint LAGUNA_HEAD_DIM = 128u;
constant uint LAGUNA_ATTENTION_THREADS = 256u;
constant uint LAGUNA_ATTENTION_SIMDGROUPS = 8u;
constant float LAGUNA_NEGATIVE_INFINITY = -3.4028234663852886e+38f;

static inline float laguna_fp8_e4m3fn_decode(uchar bits) {
    uint sign = uint(bits) >> 7u;
    uint exponent = (uint(bits) >> 3u) & 0x0fu;
    uint mantissa = uint(bits) & 0x07u;
    if (exponent == 15u && mantissa == 7u) {
        return 0.0f;
    }

    float magnitude;
    if (exponent == 0u) {
        magnitude = float(mantissa) * 0.001953125f; // 2^-9
    } else {
        magnitude = (1.0f + float(mantissa) * 0.125f)
            * exp2(float(int(exponent) - 7));
    }
    return sign == 0u ? magnitude : -magnitude;
}

static inline uchar laguna_fp8_e4m3fn_encode(float value) {
    if (isnan(value) || value == 0.0f) {
        return uchar(0u);
    }

    uint sign = value < 0.0f ? 0x80u : 0u;
    float magnitude = min(fabs(value), 448.0f);
    if (magnitude < 0.015625f) { // 2^-6, smallest normal value
        uint mantissa = uint(rint(magnitude * 512.0f));
        return uchar(sign | min(mantissa, 8u));
    }

    int unbiased_exponent = int(floor(log2(magnitude)));
    uint exponent = uint(unbiased_exponent + 7);
    float normalized = magnitude * exp2(float(-unbiased_exponent));
    uint mantissa = uint(rint((normalized - 1.0f) * 8.0f));
    if (mantissa == 8u) {
        mantissa = 0u;
        exponent += 1u;
    }
    if (exponent > 15u || (exponent == 15u && mantissa > 6u)) {
        exponent = 15u;
        mantissa = 6u;
    }
    return uchar(sign | (exponent << 3u) | mantissa);
}

static inline float laguna_softplus(float value) {
    if (value > 20.0f) {
        return value;
    }
    if (value < -20.0f) {
        return exp(value);
    }
    return log(1.0f + exp(value));
}

// Flash-style grouped-query attention for Laguna. One 256-thread group owns
// one [query token, query head] row. Eight SIMD groups score eight keys at a
// time; 128 threads accumulate the output vector. Scores and probabilities
// never leave threadgroup registers/memory.
kernel void laguna_gated_gqa_fp8_attention_f32_kernel(
    const device float* query [[buffer(0)]],
    const device float* current_key [[buffer(1)]],
    const device float* current_value [[buffer(2)]],
    const device float* gate [[buffer(3)]],
    const device uchar* cached_key [[buffer(4)]],
    const device uchar* cached_value [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& batch_count [[buffer(7)]],
    constant uint& query_heads [[buffer(8)]],
    constant uint& query_tokens [[buffer(9)]],
    constant uint& past_tokens [[buffer(10)]],
    constant uint& stored_tokens [[buffer(11)]],
    constant uint& cache_capacity [[buffer(12)]],
    constant uint& sliding_window [[buffer(13)]],
    constant float& key_scale [[buffer(14)]],
    constant float& value_scale [[buffer(15)]],
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
    // A large sliding prefill chunk can move visible_start into the current
    // chunk. Clamp the historical range before subtraction so it becomes
    // empty instead of wrapping uint past_count.
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
                    key_value = laguna_fp8_e4m3fn_decode(cached_key[cache_index]) * key_scale;
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
                    value = laguna_fp8_e4m3fn_decode(cached_value[cache_index]) * value_scale;
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
        float gated = accumulator / running_sum_shared * laguna_softplus(gate[gate_index]);
        output[query_base + tid] = gated;
    }
}

// Runs after attention in the same command buffer. Only the suffix that can
// survive in the cache is written, preventing duplicate writes when a prefill
// chunk is larger than the sliding window.
kernel void laguna_fp8_kv_append_f32_kernel(
    const device float* current_key [[buffer(0)]],
    const device float* current_value [[buffer(1)]],
    device uchar* cached_key [[buffer(2)]],
    device uchar* cached_value [[buffer(3)]],
    constant uint& batch_count [[buffer(4)]],
    constant uint& query_tokens [[buffer(5)]],
    constant uint& past_tokens [[buffer(6)]],
    constant uint& cache_capacity [[buffer(7)]],
    constant uint& retained_current_tokens [[buffer(8)]],
    constant float& key_scale [[buffer(9)]],
    constant float& value_scale [[buffer(10)]],
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

    cached_key[cache_index] = laguna_fp8_e4m3fn_encode(current_key[current_index] / key_scale);
    cached_value[cache_index] = laguna_fp8_e4m3fn_encode(current_value[current_index] / value_scale);
}
