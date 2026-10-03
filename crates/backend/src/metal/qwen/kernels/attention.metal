#include <metal_stdlib>

using namespace metal;

static inline float qwen_attention_bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

static inline ushort qwen_attention_f32_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    uint exponent = bits & 0x7f800000u;
    if (exponent == 0x7f800000u) {
        return ushort(bits >> 16);
    }
    uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return ushort(rounded >> 16);
}

static inline float qwen_attention_round_bf16(float value) {
    return qwen_attention_bf16_to_f32(qwen_attention_f32_to_bf16(value));
}

static inline float qwen_rope_value(
    const device ushort* source,
    const device ushort* norm_weight,
    uint source_start,
    uint dim,
    uint head_dim,
    uint rope_dim,
    uint position,
    float theta,
    float inverse_rms,
    uint norm_weight_has_unit_offset
) {
    float unit_offset = norm_weight_has_unit_offset != 0u ? 1.0f : 0.0f;
    float normalized = qwen_attention_round_bf16(
        qwen_attention_bf16_to_f32(source[source_start + dim]) * inverse_rms
    );
    float value = qwen_attention_round_bf16(
        normalized * (unit_offset + qwen_attention_bf16_to_f32(norm_weight[dim]))
    );
    if (dim >= rope_dim) {
        return value;
    }

    uint half_dim = rope_dim / 2u;
    uint pair_dim = dim < half_dim ? dim + half_dim : dim - half_dim;
    float normalized_pair = qwen_attention_round_bf16(
        qwen_attention_bf16_to_f32(source[source_start + pair_dim]) * inverse_rms
    );
    float pair = qwen_attention_round_bf16(
        normalized_pair
            * (unit_offset + qwen_attention_bf16_to_f32(norm_weight[pair_dim]))
    );
    uint frequency_index = dim % half_dim;
    float exponent = -2.0f * float(frequency_index) / float(rope_dim);
    float angle = float(position) * pow(theta, exponent);
    float cosine = metal::fast::cos(angle);
    float sine = metal::fast::sin(angle);
    float rotated = dim < half_dim ? -pair : pair;
    return value * cosine + rotated * sine;
}

kernel void qwen_query_norm_rope_gate_kernel(
    const device ushort* query_gate [[buffer(0)]],
    const device ushort* norm_weight [[buffer(1)]],
    device ushort* query [[buffer(2)]],
    device ushort* gate [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& query_heads [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& rope_dim [[buffer(7)]],
    constant uint& position_start [[buffer(8)]],
    constant uint& sequence_length [[buffer(9)]],
    constant float& theta [[buffer(10)]],
    constant uint& norm_weight_has_unit_offset [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint group_count = row_count * query_heads;
    if (group >= group_count) {
        return;
    }
    uint row = group / query_heads;
    uint head = group - row * query_heads;
    uint projected_head_width = head_dim * 2u;
    uint source_start = row * query_heads * projected_head_width + head * projected_head_width;
    uint output_start = (row * query_heads + head) * head_dim;

    float square_sum = 0.0f;
    for (uint dim = lane; dim < head_dim; dim += 32u) {
        float value = qwen_attention_bf16_to_f32(query_gate[source_start + dim]);
        square_sum += value * value;
        gate[output_start + dim] = query_gate[source_start + head_dim + dim];
    }
    float inverse_rms = rsqrt(simd_sum(square_sum) / float(head_dim) + 1.0e-6f);
    inverse_rms = simd_broadcast_first(inverse_rms);
    uint position = position_start + (row % sequence_length);
    for (uint dim = lane; dim < head_dim; dim += 32u) {
        float value = qwen_rope_value(
            query_gate,
            norm_weight,
            source_start,
            dim,
            head_dim,
            rope_dim,
            position,
            theta,
            inverse_rms,
            norm_weight_has_unit_offset
        );
        query[output_start + dim] = qwen_attention_f32_to_bf16(value);
    }
}

kernel void qwen_key_norm_rope_append_kernel(
    const device ushort* key_projection [[buffer(0)]],
    const device ushort* norm_weight [[buffer(1)]],
    device ushort* key_cache [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& key_heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& rope_dim [[buffer(6)]],
    constant uint& capacity_tokens [[buffer(7)]],
    constant uint& position_start [[buffer(8)]],
    constant uint& sequence_length [[buffer(9)]],
    constant float& theta [[buffer(10)]],
    constant uint& norm_weight_has_unit_offset [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint group_count = row_count * key_heads;
    if (group >= group_count) {
        return;
    }
    uint row = group / key_heads;
    uint head = group - row * key_heads;
    uint batch = row / sequence_length;
    uint token = row - batch * sequence_length;
    uint source_start = (row * key_heads + head) * head_dim;
    uint cache_start = ((batch * capacity_tokens + position_start + token) * key_heads + head) * head_dim;

    float square_sum = 0.0f;
    for (uint dim = lane; dim < head_dim; dim += 32u) {
        float value = qwen_attention_bf16_to_f32(key_projection[source_start + dim]);
        square_sum += value * value;
    }
    float inverse_rms = rsqrt(simd_sum(square_sum) / float(head_dim) + 1.0e-6f);
    inverse_rms = simd_broadcast_first(inverse_rms);
    uint position = position_start + token;
    for (uint dim = lane; dim < head_dim; dim += 32u) {
        float value = qwen_rope_value(
            key_projection,
            norm_weight,
            source_start,
            dim,
            head_dim,
            rope_dim,
            position,
            theta,
            inverse_rms,
            norm_weight_has_unit_offset
        );
        key_cache[cache_start + dim] = qwen_attention_f32_to_bf16(value);
    }
}

kernel void qwen_value_append_kernel(
    const device ushort* value_projection [[buffer(0)]],
    device ushort* value_cache [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    constant uint& key_value_width [[buffer(3)]],
    constant uint& capacity_tokens [[buffer(4)]],
    constant uint& position_start [[buffer(5)]],
    constant uint& sequence_length [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint value_count = row_count * key_value_width;
    if (gid >= value_count) {
        return;
    }
    uint row = gid / key_value_width;
    uint within_row = gid - row * key_value_width;
    uint batch = row / sequence_length;
    uint token = row - batch * sequence_length;
    uint cache_index = (batch * capacity_tokens + position_start + token) * key_value_width + within_row;
    value_cache[cache_index] = value_projection[gid];
}

kernel void qwen_online_causal_gqa_kernel(
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
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint group_count = row_count * query_heads;
    if (group >= group_count) {
        return;
    }
    uint row = group / query_heads;
    uint query_head = group - row * query_heads;
    uint batch = row / sequence_length;
    uint token = row - batch * sequence_length;
    uint key_value_head = query_head / (query_heads / key_value_heads);
    uint query_start = (row * query_heads + query_head) * head_dim;
    uint key_limit = position_start + token + 1u;
    float scale = rsqrt(float(head_dim));
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float accumulator[8];
    for (uint item = 0u; item < 8u; item++) {
        accumulator[item] = 0.0f;
    }

    for (uint key_token = 0u; key_token < key_limit; key_token++) {
        uint cache_start = ((batch * capacity_tokens + key_token) * key_value_heads + key_value_head) * head_dim;
        float partial_score = 0.0f;
        for (uint item = 0u; item < 8u; item++) {
            uint dim = lane + item * 32u;
            partial_score += qwen_attention_bf16_to_f32(query[query_start + dim])
                * qwen_attention_bf16_to_f32(key_cache[cache_start + dim]);
        }
        float score = simd_sum(partial_score) * scale;
        float next_max = max(running_max, score);
        float previous_weight = running_max == -INFINITY ? 0.0f : exp(running_max - next_max);
        float current_weight = exp(score - next_max);
        running_sum = running_sum * previous_weight + current_weight;
        for (uint item = 0u; item < 8u; item++) {
            uint dim = lane + item * 32u;
            accumulator[item] = accumulator[item] * previous_weight
                + current_weight * qwen_attention_bf16_to_f32(value_cache[cache_start + dim]);
        }
        running_max = next_max;
    }

    for (uint item = 0u; item < 8u; item++) {
        uint dim = lane + item * 32u;
        uint output_index = query_start + dim;
        float gate_value = qwen_attention_bf16_to_f32(gate[output_index]);
        float gated = (accumulator[item] / running_sum) / (1.0f + exp(-gate_value));
        output[output_index] = qwen_attention_f32_to_bf16(gated);
    }
}
