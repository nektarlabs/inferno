#include <metal_stdlib>
using namespace metal;

kernel void laguna_touch_model_view_kernel(
    device const uchar *source [[buffer(0)]],
    device uchar *samples [[buffer(1)]],
    constant ulong &stride [[buffer(2)]],
    constant ulong &source_bytes [[buffer(3)]],
    constant ulong &sample_offset [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    ulong source_offset = ulong(index) * stride;
    if (source_offset < source_bytes) {
        samples[sample_offset + ulong(index)] = source[source_offset];
    }
}
