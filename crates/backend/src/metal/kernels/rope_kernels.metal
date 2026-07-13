#include <metal_stdlib>

using namespace metal;

kernel void rope_slice_f32_kernel(
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
    uint value_count = batch_count * token_count * head_count * rope_dim;
    if (gid >= value_count) {
        return;
    }

    uint dim = gid % rope_dim;
    uint head = (gid / rope_dim) % head_count;
    uint token = (gid / (rope_dim * head_count)) % token_count;
    uint batch = gid / (rope_dim * head_count * token_count);
    uint pair_count = rope_dim / 2;
    uint pair_index = dim / 2;
    bool is_even = (dim & 1u) == 0u;
    uint partner_dim = is_even ? dim + 1u : dim - 1u;
    uint position = position_offset + token;
    uint base = (((batch * token_count + token) * head_count + head) * rope_dim);

    // GLM-DSA uses normal RoPE: adjacent values form each rotary pair.
    float inv_freq = 1.0f / pow(theta, float(pair_index) / float(pair_count));
    float angle = float(position) * inv_freq;
    float rotated = is_even ? -input[base + partner_dim] : input[base + partner_dim];

    output[gid] = input[gid] * cos(angle) + rotated * sin(angle);
}
