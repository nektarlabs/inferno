#include <metal_stdlib>

using namespace metal;

constant uint TILE_M = 16;
constant uint TILE_N = 16;
constant uint TILE_K = 32;

kernel void matmul_f32_kernel(
    const device float* lhs [[buffer(0)]],
    const device float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& inner [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float lhs_tile[TILE_M][TILE_K];
    threadgroup float rhs_tile[TILE_K][TILE_N];

    uint row = gid.y;
    uint col = gid.x;
    uint local_row = lid.y;
    uint local_col = lid.x;
    uint local_index = (local_row * TILE_N) + local_col;
    float sum = 0.0f;

    for (uint tile_start = 0; tile_start < inner; tile_start += TILE_K) {
        for (uint load_index = local_index; load_index < (TILE_M * TILE_K); load_index += (TILE_M * TILE_N)) {
            uint tile_row = load_index / TILE_K;
            uint tile_k = load_index - (tile_row * TILE_K);
            uint global_row = ((gid.y / TILE_M) * TILE_M) + tile_row;
            uint global_k = tile_start + tile_k;
            lhs_tile[tile_row][tile_k] = (global_row < rows && global_k < inner)
                ? lhs[(global_row * inner) + global_k]
                : 0.0f;
        }

        for (uint load_index = local_index; load_index < (TILE_K * TILE_N); load_index += (TILE_M * TILE_N)) {
            uint tile_k = load_index / TILE_N;
            uint tile_col = load_index - (tile_k * TILE_N);
            uint global_k = tile_start + tile_k;
            uint global_col = ((gid.x / TILE_N) * TILE_N) + tile_col;
            rhs_tile[tile_k][tile_col] = (global_k < inner && global_col < cols)
                ? rhs[(global_k * cols) + global_col]
                : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tile_k = 0; tile_k < TILE_K; tile_k++) {
            sum += lhs_tile[local_row][tile_k] * rhs_tile[tile_k][local_col];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < rows && col < cols) {
        output[(row * cols) + col] = sum;
    }
}

kernel void linear_f32_kernel(
    const device float* input [[buffer(0)]],
    const device float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    threadgroup float input_tile[TILE_M][TILE_K];
    threadgroup float weight_tile[TILE_K][TILE_N];

    uint row = gid.y;
    uint output_feature = gid.x;
    uint local_row = lid.y;
    uint local_col = lid.x;
    uint local_index = (local_row * TILE_N) + local_col;
    float sum = 0.0f;

    for (uint tile_start = 0; tile_start < in_features; tile_start += TILE_K) {
        for (uint load_index = local_index; load_index < (TILE_M * TILE_K); load_index += (TILE_M * TILE_N)) {
            uint tile_row = load_index / TILE_K;
            uint tile_k = load_index - (tile_row * TILE_K);
            uint global_row = ((gid.y / TILE_M) * TILE_M) + tile_row;
            uint global_k = tile_start + tile_k;
            input_tile[tile_row][tile_k] = (global_row < rows && global_k < in_features)
                ? input[(global_row * in_features) + global_k]
                : 0.0f;
        }

        for (uint load_index = local_index; load_index < (TILE_K * TILE_N); load_index += (TILE_M * TILE_N)) {
            uint tile_k = load_index / TILE_N;
            uint tile_col = load_index - (tile_k * TILE_N);
            uint global_k = tile_start + tile_k;
            uint global_output_feature = ((gid.x / TILE_N) * TILE_N) + tile_col;
            weight_tile[tile_k][tile_col] = (global_output_feature < out_features && global_k < in_features)
                ? weight[(global_output_feature * in_features) + global_k]
                : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint tile_k = 0; tile_k < TILE_K; tile_k++) {
            sum += input_tile[local_row][tile_k] * weight_tile[tile_k][local_col];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < rows && output_feature < out_features) {
        output[(row * out_features) + output_feature] = sum;
    }
}

kernel void linear_f32_gemv_kernel(
    const device float* input [[buffer(0)]],
    const device float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& in_features_vec4 [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    constexpr uint simd_lanes = 32;
    uint output_feature = gid / simd_lanes;
    if (output_feature >= out_features) {
        return;
    }

    const device packed_float4* input4 = reinterpret_cast<const device packed_float4*>(input);
    const device packed_float4* weight4 = reinterpret_cast<const device packed_float4*>(
        weight + (output_feature * in_features_vec4 * 4)
    );

    float sum = 0.0f;
    for (uint index = simd_lane; index < in_features_vec4; index += simd_lanes) {
        sum += dot(float4(input4[index]), float4(weight4[index]));
    }
    sum = simd_sum(sum);

    if (simd_lane == 0) {
        output[output_feature] = sum;
    }
}
