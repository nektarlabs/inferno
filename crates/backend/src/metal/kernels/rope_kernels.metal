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
