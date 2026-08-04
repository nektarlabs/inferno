#include <metal_stdlib>

using namespace metal;

constant uint LAGUNA_XS_MPS_BLOCK_VALUES = 256u;

struct LagunaXsMpsQ4KBlock {
    half d;
    half dmin;
    uchar scales[12];
    uchar quants[128];
};

struct LagunaXsMpsQ6KBlock {
    uchar low_quants[128];
    uchar high_quants[64];
    char scales[16];
    half d;
};

static inline uchar laguna_xs_mps_q4_scale(
    const device LagunaXsMpsQ4KBlock* block,
    uint group
) {
    if (group < 4u) {
        return block->scales[group] & 63u;
    }
    return (block->scales[group + 4u] & 15u)
        | ((block->scales[group - 4u] & 192u) >> 2u);
}

static inline uchar laguna_xs_mps_q4_minimum(
    const device LagunaXsMpsQ4KBlock* block,
    uint group
) {
    if (group < 4u) {
        return block->scales[group + 4u] & 63u;
    }
    return (block->scales[group + 4u] >> 4u)
        | ((block->scales[group] & 192u) >> 2u);
}

kernel void laguna_xs_mps_dequant_q4_f16_kernel(
    const device LagunaXsMpsQ4KBlock* weights [[buffer(0)]],
    device half4* output [[buffer(1)]],
    constant uint& in_features [[buffer(2)]],
    constant uint& value_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint value_start = gid * 4u;
    if (value_start >= value_count) {
        return;
    }
    uint row = value_start / in_features;
    uint column = value_start - row * in_features;
    uint blocks_per_row = in_features / LAGUNA_XS_MPS_BLOCK_VALUES;
    uint block_in_row = column / LAGUNA_XS_MPS_BLOCK_VALUES;
    uint within_block = column - block_in_row * LAGUNA_XS_MPS_BLOCK_VALUES;
    uint group = within_block / 32u;
    uint lane = within_block - group * 32u;
    uint pair = group / 2u;
    const device LagunaXsMpsQ4KBlock* block =
        weights + row * blocks_per_row + block_in_row;
    uchar4 packed = *reinterpret_cast<const device uchar4*>(
        block->quants + pair * 32u + lane);
    uchar4 quantized = (group & 1u) == 0u
        ? packed & uchar4(15u)
        : packed >> uchar4(4u);
    float4 value =
        float(block->d)
            * float(laguna_xs_mps_q4_scale(block, group))
            * float4(quantized)
        - float(block->dmin)
            * float(laguna_xs_mps_q4_minimum(block, group));
    output[gid] = half4(value);
}

kernel void laguna_xs_mps_dequant_q6_f16_kernel(
    const device LagunaXsMpsQ6KBlock* weights [[buffer(0)]],
    device half4* output [[buffer(1)]],
    constant uint& in_features [[buffer(2)]],
    constant uint& value_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint value_start = gid * 4u;
    if (value_start >= value_count) {
        return;
    }
    uint row = value_start / in_features;
    uint column = value_start - row * in_features;
    uint blocks_per_row = in_features / LAGUNA_XS_MPS_BLOCK_VALUES;
    uint block_in_row = column / LAGUNA_XS_MPS_BLOCK_VALUES;
    uint within_block = column - block_in_row * LAGUNA_XS_MPS_BLOCK_VALUES;
    uint group = within_block / 32u;
    uint lane = within_block - group * 32u;
    uint half_index = group / 4u;
    uint quarter = group - half_index * 4u;
    const device LagunaXsMpsQ6KBlock* block =
        weights + row * blocks_per_row + block_in_row;
    uint low_base = half_index * 64u;
    uint high_base = half_index * 32u;
    uchar4 low_even = *reinterpret_cast<const device uchar4*>(
        block->low_quants + low_base + lane);
    uchar4 low_odd = *reinterpret_cast<const device uchar4*>(
        block->low_quants + low_base + 32u + lane);
    uchar4 high = *reinterpret_cast<const device uchar4*>(
        block->high_quants + high_base + lane);
    uchar4 quantized;
    switch (quarter) {
        case 0u:
            quantized =
                (low_even & uchar4(15u))
                | ((high & uchar4(3u)) << uchar4(4u));
            break;
        case 1u:
            quantized =
                (low_odd & uchar4(15u))
                | (((high >> uchar4(2u)) & uchar4(3u)) << uchar4(4u));
            break;
        case 2u:
            quantized =
                (low_even >> uchar4(4u))
                | (((high >> uchar4(4u)) & uchar4(3u)) << uchar4(4u));
            break;
        default:
            quantized =
                (low_odd >> uchar4(4u))
                | (((high >> uchar4(6u)) & uchar4(3u)) << uchar4(4u));
            break;
    }
    uint scale_index =
        half_index * 8u + quarter * 2u + lane / 16u;
    float4 value =
        float(block->d)
            * float(block->scales[scale_index])
            * (float4(quantized) - 32.0f);
    output[gid] = half4(value);
}

kernel void laguna_xs_mps_cast_f32_f16_kernel(
    const device float4* input [[buffer(0)]],
    device half4* output [[buffer(1)]],
    constant uint& value_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid * 4u < value_count) {
        output[gid] = half4(input[gid]);
    }
}

template <ushort ResidualCount>
kernel void laguna_xs_mps_cast_f16_f32_impl(
    const device half4* input [[buffer(0)]],
    const device float4* residual_a [[buffer(1)]],
    const device float4* residual_b [[buffer(2)]],
    device float4* output [[buffer(3)]],
    constant uint& value_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid * 4u >= value_count) {
        return;
    }
    float4 value = float4(input[gid]);
    if (ResidualCount >= 1) {
        value += residual_a[gid];
    }
    if (ResidualCount >= 2) {
        value += residual_b[gid];
    }
    output[gid] = value;
}

typedef decltype(laguna_xs_mps_cast_f16_f32_impl<0>)
    LagunaXsMpsCast;
typedef decltype(laguna_xs_mps_cast_f16_f32_impl<1>)
    LagunaXsMpsCastAdd;
typedef decltype(laguna_xs_mps_cast_f16_f32_impl<2>)
    LagunaXsMpsCastAdd2;

template [[host_name("laguna_xs_mps_cast_f16_f32_kernel")]]
kernel LagunaXsMpsCast laguna_xs_mps_cast_f16_f32_impl<0>;

template [[host_name("laguna_xs_mps_cast_f16_f32_add_kernel")]]
kernel LagunaXsMpsCastAdd laguna_xs_mps_cast_f16_f32_impl<1>;

template [[host_name("laguna_xs_mps_cast_f16_f32_add2_kernel")]]
kernel LagunaXsMpsCastAdd2 laguna_xs_mps_cast_f16_f32_impl<2>;
