#include <metal_stdlib>
using namespace metal;

static inline float grouped_bf16(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

static inline ushort grouped_round_bf16(float value) {
    uint bits = as_type<uint>(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) return ushort(bits >> 16);
    return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

// Eight SIMD groups per query head; paired heads reuse the same K/V tile.
// Softmax and value accumulation retain the reference's chronological order.
kernel void qwen_decode_grouped_attention_kernel(
    const device ushort* query [[buffer(0)]],
    const device ushort* gate [[buffer(1)]],
    const device ushort* key_cache [[buffer(2)]],
    const device ushort* value_cache [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& sequence_length [[buffer(6)]],
    constant uint& query_heads [[buffer(7)]],
    constant uint& key_value_heads [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& capacity_tokens [[buffer(10)]],
    constant uint& position_start [[buffer(11)]],
    constant uint& heads_per_group [[buffer(12)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint head_groups = query_heads / heads_per_group;
    if (group >= row_count * head_groups) return;
    uint row = group / head_groups;
    uint first_head = (group % head_groups) * heads_per_group;
    uint local_head = simd_group / 8u;
    uint key_in_tile = simd_group % 8u;
    uint batch = row / sequence_length;
    uint kv_head = first_head / (query_heads / key_value_heads);
    uint query_start = (row * query_heads + first_head + local_head) * head_dim;
    uint key_limit = position_start + row % sequence_length + 1u;
    uint dim = key_in_tile * 32u + lane;
    uint thread_index = simd_group * 32u + lane;
    uint threads = heads_per_group * 256u;
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float accumulator = 0.0f;
    float scale = rsqrt(float(head_dim));
    threadgroup ushort keys[8 * 256];
    threadgroup ushort values[8 * 256];
    threadgroup float scores[3 * 8 * 32];

    for (uint tile = 0u; tile < key_limit; tile += 8u) {
        uint count = min(8u, key_limit - tile);
        for (uint index = thread_index; index < count * 256u; index += threads) {
            uint cache_index = ((batch * capacity_tokens + tile + index / 256u) * key_value_heads + kv_head) * head_dim + index % 256u;
            keys[index] = key_cache[cache_index];
            values[index] = value_cache[cache_index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (key_in_tile < count) {
            float partial_score = 0.0f;
            for (uint item = 0u; item < 8u; item++) {
                partial_score += grouped_bf16(query[query_start + lane + item * 32u])
                    * grouped_bf16(keys[key_in_tile * 256u + lane + item * 32u]);
            }
            scores[simd_group * 32u + lane] = simd_sum(partial_score) * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint key = 0u; key < count; key++) {
            float score = scores[(local_head * 8u + key) * 32u + lane];
            float next_max = max(running_max, score);
            float previous_weight = running_max == -INFINITY ? 0.0f : exp(running_max - next_max);
            float current_weight = exp(score - next_max);
            running_sum = running_sum * previous_weight + current_weight;
            // Preserve the reference's contraction, not fma(acc, prev, weight*value).
            accumulator = fma(current_weight, grouped_bf16(values[key * 256u + dim]),
                accumulator * previous_weight);
            running_max = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float gate_value = grouped_bf16(gate[query_start + dim]);
    float gated = (accumulator / running_sum) / (1.0f + exp(-gate_value));
    output[query_start + dim] = grouped_round_bf16(gated);
}
