#include <metal_stdlib>

using namespace metal;

kernel void rope_slice_f32_pair_kernel(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_count [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    constant uint& head_count [[buffer(4)]],
    constant uint& rope_dim [[buffer(5)]],
    constant uint& position_offset [[buffer(6)]],
    constant float& theta [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint pair_count = rope_dim / 2;
    uint pair_total = batch_count * token_count * head_count * pair_count;
    if (gid >= pair_total) {
        return;
    }

    uint pair_index = gid % pair_count;
    uint head = (gid / pair_count) % head_count;
    uint token = (gid / (pair_count * head_count)) % token_count;
    uint batch = gid / (pair_count * head_count * token_count);
    uint position = position_offset + token;
    uint base = (((batch * token_count + token) * head_count + head) * rope_dim);
    uint even_dim = pair_index * 2u;
    uint odd_dim = even_dim + 1u;

    float inv_freq = 1.0f / pow(theta, float(pair_index) / float(pair_count));
    float angle = float(position) * inv_freq;
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);
    float even = input[base + even_dim];
    float odd = input[base + odd_dim];

    output[base + even_dim] = even * cos_angle - odd * sin_angle;
    output[base + odd_dim] = even * sin_angle + odd * cos_angle;
}

// GLM-5.2 applies the same 32 rotary coefficients to every query head at a
// token position. During long prefill, four heads share one coefficient table
// so only one cohort evaluates pow/sin/cos. The fixed 64-value rotary slice is
// the exact production model shape.
kernel void rope_slice_f32_shared4_kernel(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_count [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    constant uint& head_count [[buffer(4)]],
    constant uint& rope_dim [[buffer(5)]],
    constant uint& position_offset [[buffer(6)]],
    constant float& theta [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_position [[threadgroup_position_in_grid]]
) {
    if (rope_dim != 64u) {
        return;
    }

    constexpr uint heads_per_group = 4u;
    constexpr uint pair_count = 32u;
    constexpr uint threads_per_head = 64u;
    uint head_group_count = (head_count + heads_per_group - 1u) / heads_per_group;
    uint batch_token = group_position.x / head_group_count;
    uint head_group = group_position.x - (batch_token * head_group_count);
    uint token = batch_token % token_count;
    uint batch = batch_token / token_count;
    if (batch >= batch_count) {
        return;
    }

    uint head_in_group = tid / threads_per_head;
    uint rotary_dim = tid - (head_in_group * threads_per_head);
    threadgroup float cos_shared[pair_count];
    threadgroup float sin_shared[pair_count];

    if (head_in_group == 0u && (rotary_dim & 1u) == 0u) {
        uint pair_index = rotary_dim / 2u;
        uint position = position_offset + token;
        float inv_freq = 1.0f / pow(theta, float(pair_index) / float(pair_count));
        float angle = float(position) * inv_freq;
        cos_shared[pair_index] = cos(angle);
        sin_shared[pair_index] = sin(angle);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint head = head_group * heads_per_group + head_in_group;
    if (head >= head_count || (rotary_dim & 1u) != 0u) {
        return;
    }

    uint pair_index = rotary_dim / 2u;
    uint base = (((batch * token_count + token) * head_count + head) * rope_dim);
    uint even_dim = pair_index * 2u;
    uint odd_dim = even_dim + 1u;
    float even = input[base + even_dim];
    float odd = input[base + odd_dim];
    float cos_angle = cos_shared[pair_index];
    float sin_angle = sin_shared[pair_index];

    output[base + even_dim] = even * cos_angle - odd * sin_angle;
    output[base + odd_dim] = even * sin_angle + odd * cos_angle;
}

// Laguna normalizes every Q/K head independently before applying RoPE. The
// checkpoint uses the non-interleaved (half-split) rotation convention:
// [x0..xN, y0..yN] -> [-y0..-yN, x0..xN]. Global layers rotate 64 of the 128
// head channels with YaRN; sliding layers rotate all 128 channels with the
// default frequency table. One SIMD group owns one complete head row.
kernel void qk_rms_norm_half_split_rope_f32_kernel(
    const device float* input [[buffer(0)]],
    const device float* norm_weight [[buffer(1)]],
    const device float* inverse_frequency [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& token_count [[buffer(5)]],
    constant uint& head_count [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& rotary_dim [[buffer(8)]],
    constant uint& position_offset [[buffer(9)]],
    constant float& epsilon [[buffer(10)]],
    constant float& attention_factor [[buffer(11)]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint3 group_position [[threadgroup_position_in_grid]]
) {
    uint row = group_position.x;
    if (row >= row_count) {
        return;
    }

    uint base = row * head_dim;
    float local_sumsq = 0.0f;
    for (uint dim = simd_lane; dim < head_dim; dim += 32u) {
        float value = input[base + dim];
        local_sumsq += value * value;
    }
    float inverse_rms = rsqrt(simd_sum(local_sumsq) / float(head_dim) + epsilon);

    uint token = (row / head_count) % token_count;
    float position = float(position_offset + token);
    uint rotary_half = rotary_dim / 2u;

    for (uint dim = simd_lane; dim < head_dim; dim += 32u) {
        float normalized = input[base + dim] * norm_weight[dim] * inverse_rms;
        if (dim >= rotary_dim) {
            output[base + dim] = normalized;
            continue;
        }

        uint frequency_index = dim % rotary_half;
        uint paired_dim = dim < rotary_half ? dim + rotary_half : dim - rotary_half;
        float paired = input[base + paired_dim] * norm_weight[paired_dim] * inverse_rms;
        float rotated = dim < rotary_half ? -paired : paired;
        float angle = position * inverse_frequency[frequency_index];
        float cos_angle = cos(angle) * attention_factor;
        float sin_angle = sin(angle) * attention_factor;
        output[base + dim] = normalized * cos_angle + rotated * sin_angle;
    }
}

// Q and K use the same position table but different head counts and learned
// RMSNorm weights. A single dispatch walks both row sets, avoiding a second
// encoder boundary while keeping their output buffers separate for GQA.
kernel void laguna_qk_rms_norm_half_split_rope_pair_f32_kernel(
    const device float* query_input [[buffer(0)]],
    const device float* key_input [[buffer(1)]],
    const device float* query_norm_weight [[buffer(2)]],
    const device float* key_norm_weight [[buffer(3)]],
    const device float* inverse_frequency [[buffer(4)]],
    device float* query_output [[buffer(5)]],
    device float* key_output [[buffer(6)]],
    constant uint& query_row_count [[buffer(7)]],
    constant uint& key_row_count [[buffer(8)]],
    constant uint& token_count [[buffer(9)]],
    constant uint& query_head_count [[buffer(10)]],
    constant uint& key_head_count [[buffer(11)]],
    constant uint& head_dim [[buffer(12)]],
    constant uint& rotary_dim [[buffer(13)]],
    constant uint& position_offset [[buffer(14)]],
    constant float& epsilon [[buffer(15)]],
    constant float& attention_factor [[buffer(16)]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint3 group_position [[threadgroup_position_in_grid]]
) {
    uint combined_row = group_position.x;
    if (combined_row >= query_row_count + key_row_count) {
        return;
    }

    bool is_query = combined_row < query_row_count;
    uint row = is_query ? combined_row : combined_row - query_row_count;
    uint head_count = is_query ? query_head_count : key_head_count;
    const device float* input = is_query ? query_input : key_input;
    const device float* norm_weight = is_query
        ? query_norm_weight
        : key_norm_weight;
    device float* output = is_query ? query_output : key_output;
    uint base = row * head_dim;

    float local_sumsq = 0.0f;
    for (uint dim = simd_lane; dim < head_dim; dim += 32u) {
        float value = input[base + dim];
        local_sumsq += value * value;
    }
    float inverse_rms = rsqrt(simd_sum(local_sumsq) / float(head_dim) + epsilon);

    uint token = (row / head_count) % token_count;
    float position = float(position_offset + token);
    uint rotary_half = rotary_dim / 2u;
    for (uint dim = simd_lane; dim < head_dim; dim += 32u) {
        float normalized = input[base + dim] * norm_weight[dim] * inverse_rms;
        if (dim >= rotary_dim) {
            output[base + dim] = normalized;
            continue;
        }

        uint frequency_index = dim % rotary_half;
        uint paired_dim = dim < rotary_half ? dim + rotary_half : dim - rotary_half;
        float paired = input[base + paired_dim] * norm_weight[paired_dim] * inverse_rms;
        float rotated = dim < rotary_half ? -paired : paired;
        float angle = position * inverse_frequency[frequency_index];
        float cos_angle = cos(angle) * attention_factor;
        float sin_angle = sin(angle) * attention_factor;
        output[base + dim] = normalized * cos_angle + rotated * sin_angle;
    }
}
