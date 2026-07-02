#include <metal_stdlib>

using namespace metal;

kernel void rms_norm_f32_kernel(
    const device float* input [[buffer(0)]],
    const device float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= rows) {
        return;
    }

    uint base = row * hidden_size;
    float sumsq = 0.0f;

    for (uint i = 0; i < hidden_size; i++) {
        float value = input[base + i];
        sumsq += value * value;
    }

    float scale = rsqrt((sumsq / float(hidden_size)) + eps);

    for (uint i = 0; i < hidden_size; i++) {
        output[base + i] = input[base + i] * scale * weight[i];
    }
}
