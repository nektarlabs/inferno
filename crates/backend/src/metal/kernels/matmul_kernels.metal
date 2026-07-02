#include <metal_stdlib>

using namespace metal;

kernel void matmul_f32_kernel(
    const device float* lhs [[buffer(0)]],
    const device float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = rows * cols;
    if (gid >= output_values) {
        return;
    }

    uint row = gid / cols;
    uint col = gid - (row * cols);
    float sum = 0.0f;

    for (uint index = 0; index < inner; index++) {
        sum += lhs[(row * inner) + index] * rhs[(index * cols) + col];
    }

    output[gid] = sum;
}

kernel void linear_f32_kernel(
    const device float* input [[buffer(0)]],
    const device float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = rows * out_features;
    if (gid >= output_values) {
        return;
    }

    uint row = gid / out_features;
    uint output_feature = gid - (row * out_features);
    float sum = 0.0f;

    for (uint input_feature = 0; input_feature < in_features; input_feature++) {
        sum += input[(row * in_features) + input_feature]
            * weight[(output_feature * in_features) + input_feature];
    }

    output[gid] = sum;
}
