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
    threadgroup float scale;

    float simd_sumsq = simd_sum(local_sumsq);
    if (simd_lane == 0 && simd_group < RMS_NORM_SIMDGROUPS) {
        simd_sums[simd_group] = simd_sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float sumsq = 0.0f;
        for (uint i = 0; i < RMS_NORM_SIMDGROUPS; i++) {
            sumsq += simd_sums[i];
        }
        scale = rsqrt((sumsq / float(hidden_size)) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < hidden_size; i += RMS_NORM_THREADS) {
        output[base + i] = input[base + i] * scale * weight[i];
    }
}
