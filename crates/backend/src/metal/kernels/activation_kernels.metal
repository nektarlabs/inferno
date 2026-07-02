#include <metal_stdlib>

using namespace metal;

kernel void swiglu_f32_kernel(
    const device float* gate [[buffer(0)]],
    const device float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& value_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= value_count) {
        return;
    }

    float gate_value = gate[gid];
    float silu = gate_value / (1.0f + exp(-gate_value));
    output[gid] = silu * up[gid];
}

kernel void add_f32_kernel(
    const device float* lhs [[buffer(0)]],
    const device float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& value_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= value_count) {
        return;
    }

    output[gid] = lhs[gid] + rhs[gid];
}
