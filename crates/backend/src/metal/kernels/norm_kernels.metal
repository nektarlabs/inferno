#include <metal_stdlib>

using namespace metal;

constant uint RMS_NORM_THREADS = 256;
constant uint RMS_NORM_SIMDGROUPS = 8;

kernel void rms_norm_f32_kernel(
    const device float* input [[buffer(0)]],
    const device float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (row >= rows) {
        return;
    }

    uint base = row * hidden_size;
    float local_sumsq = 0.0f;

    for (uint i = tid; i < hidden_size; i += RMS_NORM_THREADS) {
        float value = input[base + i];
        local_sumsq += value * value;
    }

    threadgroup float simd_sums[RMS_NORM_SIMDGROUPS];

    float simd_sumsq = simd_sum(local_sumsq);
    if (simd_lane == 0 && simd_group < RMS_NORM_SIMDGROUPS) {
        simd_sums[simd_group] = simd_sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float partial = 0.0f;
    if (simd_lane < RMS_NORM_SIMDGROUPS) {
        partial = simd_sums[simd_lane];
    }
    float sumsq = simd_sum(partial);
    float scale = rsqrt((sumsq / float(hidden_size)) + eps);

    for (uint i = tid; i < hidden_size; i += RMS_NORM_THREADS) {
        output[base + i] = input[base + i] * scale * weight[i];
    }
}

kernel void mla_kv_postprocess_f32_kernel(
    const device float* input [[buffer(0)]],
    const device float* norm_weight [[buffer(1)]],
    device float* latent_output [[buffer(2)]],
    device float* rope_output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& token_count [[buffer(5)]],
    constant uint& latent_dim [[buffer(6)]],
    constant uint& rope_dim [[buffer(7)]],
    constant uint& position_offset [[buffer(8)]],
    constant float& theta [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (row >= rows) {
        return;
    }

    uint input_base = row * (latent_dim + rope_dim);
    float local_sumsq = 0.0f;
    for (uint dim = tid; dim < latent_dim; dim += RMS_NORM_THREADS) {
        float value = input[input_base + dim];
        local_sumsq += value * value;
    }

    threadgroup float simd_sums[RMS_NORM_SIMDGROUPS];
    float simd_sumsq = simd_sum(local_sumsq);
    if (simd_lane == 0 && simd_group < RMS_NORM_SIMDGROUPS) {
        simd_sums[simd_group] = simd_sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float partial = 0.0f;
    if (simd_lane < RMS_NORM_SIMDGROUPS) {
        partial = simd_sums[simd_lane];
    }
    float sumsq = simd_sum(partial);
    float norm_scale = rsqrt((sumsq / float(latent_dim)) + eps);

    uint latent_base = row * latent_dim;
    for (uint dim = tid; dim < latent_dim; dim += RMS_NORM_THREADS) {
        latent_output[latent_base + dim] =
            input[input_base + dim] * norm_scale * norm_weight[dim];
    }

    uint rope_base = row * rope_dim;
    uint token = row % token_count;
    uint position = position_offset + token;
    uint pair_count = rope_dim / 2;
    for (uint dim = tid; dim < rope_dim; dim += RMS_NORM_THREADS) {
        uint pair_index = dim / 2;
        bool is_even = (dim & 1u) == 0u;
        uint partner_dim = is_even ? dim + 1u : dim - 1u;
        float inv_freq = 1.0f / pow(theta, float(pair_index) / float(pair_count));
        float angle = float(position) * inv_freq;
        float value = input[input_base + latent_dim + dim];
        float partner = input[input_base + latent_dim + partner_dim];
        float rotated = is_even ? -partner : partner;
        rope_output[rope_base + dim] = value * cos(angle) + rotated * sin(angle);
    }
}
