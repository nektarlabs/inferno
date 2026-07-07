#include <metal_stdlib>

using namespace metal;

kernel void f32_to_f16_kernel(
    const device float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= len) {
        return;
    }
    output[gid] = half(input[gid]);
}

kernel void q8_rows_to_f32_kernel(
    const device uchar* payload [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = row_count * dim;
    if (gid >= total) {
        return;
    }

    uint row = gid / dim;
    uint value_index = gid - (row * dim);
    uint row_bytes = dim + 4u;
    uint row_offset = row * row_bytes;
    uint scale_bits =
        uint(payload[row_offset])
        | (uint(payload[row_offset + 1u]) << 8)
        | (uint(payload[row_offset + 2u]) << 16)
        | (uint(payload[row_offset + 3u]) << 24);
    float scale = as_type<float>(scale_bits);
    uchar raw = payload[row_offset + 4u + value_index];
    int quantized = raw > 127u ? int(raw) - 256 : int(raw);
    output[gid] = float(quantized) * scale;
}
