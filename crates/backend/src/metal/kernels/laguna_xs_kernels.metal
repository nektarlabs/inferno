#include <metal_stdlib>

using namespace metal;

/*
 * Laguna XS uses the standard GGUF Q4_K and Q6_K byte layouts. These kernels
 * decode one 256-value quantization block cooperatively across a SIMD group.
 * Each lane consumes the same position from eight contiguous 32-value groups,
 * which keeps weight and activation reads coalesced.
 */

constant uint LAGUNA_XS_K_BLOCK_VALUES = 256u;
constant uint LAGUNA_XS_HIDDEN_BLOCKS = 8u;
constant uint LAGUNA_XS_SIMDGROUPS_PER_THREADGROUP = 4u;
constant uint LAGUNA_XS_EXPERT_GATE_UP_ROWS_PER_SIMDGROUP = 1u;
constant uint LAGUNA_XS_EXPERT_DOWN_ROWS_PER_SIMDGROUP = 1u;
constant uint LAGUNA_XS_EXPERT_DOWN_PARALLEL_SIMDGROUPS = 4u;

struct LagunaXsQ4KBlock {
    half d;
    half dmin;
    uchar scales[12];
    uchar quants[128];
};

struct LagunaXsQ6KBlock {
    uchar low_quants[128];
    uchar high_quants[64];
    char scales[16];
    half d;
};

static inline uchar laguna_xs_q4_scale(
    const device LagunaXsQ4KBlock* block,
    uint group
) {
    if (group < 4u) {
        return block->scales[group] & 63u;
    }
    return (block->scales[group + 4u] & 15u)
        | ((block->scales[group - 4u] & 192u) >> 2u);
}

static inline uchar laguna_xs_q4_minimum(
    const device LagunaXsQ4KBlock* block,
    uint group
) {
    if (group < 4u) {
        return block->scales[group + 4u] & 63u;
    }
    return (block->scales[group + 4u] >> 4u)
        | ((block->scales[group] & 192u) >> 2u);
}

static inline float laguna_xs_q4_lane_dot(
    const device LagunaXsQ4KBlock* block,
    const device float* input,
    uint lane
) {
    float dot = 0.0f;
    float d = float(block->d);
    float dmin = float(block->dmin);

    for (uint pair = 0u; pair < 4u; pair++) {
        uint low_group = pair * 2u;
        uint high_group = low_group + 1u;
        uint packed = uint(block->quants[pair * 32u + lane]);
        float low_weight =
            d * float(laguna_xs_q4_scale(block, low_group))
                * float(packed & 15u)
            - dmin * float(laguna_xs_q4_minimum(block, low_group));
        float high_weight =
            d * float(laguna_xs_q4_scale(block, high_group))
                * float(packed >> 4u)
            - dmin * float(laguna_xs_q4_minimum(block, high_group));
        dot = fma(
            low_weight,
            input[low_group * 32u + lane],
            dot);
        dot = fma(
            high_weight,
            input[high_group * 32u + lane],
            dot);
    }
    return dot;
}

static inline float laguna_xs_q6_lane_dot(
    const device LagunaXsQ6KBlock* block,
    const device float* input,
    uint lane
) {
    float dot = 0.0f;
    float d = float(block->d);

    for (uint half_index = 0u; half_index < 2u; half_index++) {
        uint low_base = half_index * 64u;
        uint high_base = half_index * 32u;
        uint low_even = uint(block->low_quants[low_base + lane]);
        uint low_odd = uint(block->low_quants[low_base + 32u + lane]);
        uint high = uint(block->high_quants[high_base + lane]);
        uint quantized[4] = {
            (low_even & 15u) | ((high & 3u) << 4u),
            (low_odd & 15u) | (((high >> 2u) & 3u) << 4u),
            (low_even >> 4u) | (((high >> 4u) & 3u) << 4u),
            (low_odd >> 4u) | (((high >> 6u) & 3u) << 4u),
        };
        for (uint quarter = 0u; quarter < 4u; quarter++) {
            uint group = half_index * 4u + quarter;
            uint scale_index = half_index * 8u
                + quarter * 2u
                + lane / 16u;
            float weight = d * float(block->scales[scale_index])
                * float(int(quantized[quarter]) - 32);
            dot = fma(weight, input[group * 32u + lane], dot);
        }
    }
    return dot;
}

struct LagunaXsQ4Operations {
    static inline float lane_dot(
        const device LagunaXsQ4KBlock* block,
        const device float* input,
        uint lane
    ) {
        return laguna_xs_q4_lane_dot(block, input, lane);
    }
};

struct LagunaXsQ6Operations {
    static inline float lane_dot(
        const device LagunaXsQ6KBlock* block,
        const device float* input,
        uint lane
    ) {
        return laguna_xs_q6_lane_dot(block, input, lane);
    }
};

template <typename Block, typename Operations, uint RowsPerSimdgroup>
static inline void laguna_xs_k_matvec(
    const device Block* weights,
    const device float* input,
    device float* output,
    uint row_count,
    uint in_features,
    uint out_features,
    uint3 threadgroup_position,
    ushort simd_lane,
    ushort simdgroup_index
) {
    uint simdgroup = threadgroup_position.x
        * LAGUNA_XS_SIMDGROUPS_PER_THREADGROUP
        + uint(simdgroup_index);
    uint output_row_groups = (out_features
        + RowsPerSimdgroup - 1u)
        / RowsPerSimdgroup;
    uint input_row = simdgroup / output_row_groups;
    if (input_row >= row_count) {
        return;
    }

    uint first_output_row = (simdgroup
        - input_row * output_row_groups)
        * RowsPerSimdgroup;
    uint blocks_per_row = in_features / LAGUNA_XS_K_BLOCK_VALUES;
    const device float* input_values = input + input_row * in_features;
    float sums[RowsPerSimdgroup] = {};

    for (uint block_index = 0u;
         block_index < blocks_per_row;
         block_index++) {
        const device float* input_block =
            input_values + block_index * LAGUNA_XS_K_BLOCK_VALUES;
        for (uint local_row = 0u;
             local_row < RowsPerSimdgroup;
             local_row++) {
            uint output_row = first_output_row + local_row;
            if (output_row < out_features) {
                const device Block* weight_block =
                    weights + output_row * blocks_per_row + block_index;
                sums[local_row] += Operations::lane_dot(
                    weight_block,
                    input_block,
                    uint(simd_lane));
            }
        }
    }

    for (uint local_row = 0u;
         local_row < RowsPerSimdgroup;
         local_row++) {
        uint output_row = first_output_row + local_row;
        float sum = simd_sum(sums[local_row]);
        if (simd_lane == 0u && output_row < out_features) {
            output[input_row * out_features + output_row] = sum;
        }
    }
}

kernel void laguna_xs_q4_k_matvec_f32_kernel(
    const device LagunaXsQ4KBlock* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    laguna_xs_k_matvec<
        LagunaXsQ4KBlock,
        LagunaXsQ4Operations,
        1u
    >(
        weights,
        input,
        output,
        row_count,
        in_features,
        out_features,
        threadgroup_position,
        simd_lane,
        simdgroup_index);
}

kernel void laguna_xs_q6_k_matvec_f32_kernel(
    const device LagunaXsQ6KBlock* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    laguna_xs_k_matvec<
        LagunaXsQ6KBlock,
        LagunaXsQ6Operations,
        1u
    >(
        weights,
        input,
        output,
        row_count,
        in_features,
        out_features,
        threadgroup_position,
        simd_lane,
        simdgroup_index);
}

/*
 * Decode uses four groups of eight lanes to process four Q4_K blocks
 * concurrently. Packed ushort operations expose four nibbles per instruction,
 * while two adjacent output rows reuse the activation fragments already held
 * by each lane.
 */
static inline float2 laguna_xs_q4_decode_two_rows_block_dot(
    const device LagunaXsQ4KBlock* first_row,
    const device LagunaXsQ4KBlock* second_row,
    const device float* input,
    uint block_index,
    ushort simd_lane
) {
    constexpr ushort scale_mask = 0x3f3f;
    constexpr ushort low_nibble_mask = 0x0f0f;
    constexpr ushort high_bits_mask = 0xc0c0;

    uint lane_in_block = uint(simd_lane) % 8u;
    uint quant_half = lane_in_block / 4u;
    uint quant_lane = lane_in_block % 4u;
    float2 row_sums = float2(0.0f);

    uint activation_base = block_index * LAGUNA_XS_K_BLOCK_VALUES
        + 64u * quant_half
        + 8u * quant_lane;
    float low_values[16];
    float high_values[16];
    float4 activation_sums = float4(0.0f);
    #pragma unroll
    for (uint index = 0u; index < 8u; index++) {
        low_values[index] = input[activation_base + index];
        low_values[index + 8u] =
            input[activation_base + index + 32u];
        high_values[index] =
            input[activation_base + index + 128u];
        high_values[index + 8u] =
            input[activation_base + index + 160u];
        activation_sums[0] += low_values[index];
        activation_sums[1] += low_values[index + 8u];
        activation_sums[2] += high_values[index];
        activation_sums[3] += high_values[index + 8u];
    }

    #pragma unroll
    for (uint local_row = 0u; local_row < 2u; local_row++) {
        const device LagunaXsQ4KBlock* block =
            (local_row == 0u ? first_row : second_row) + block_index;
        const device ushort* packed_scales =
            reinterpret_cast<const device ushort*>(block->scales)
            + quant_half;
        ushort decoded_scales[4];
        decoded_scales[0] = packed_scales[0] & scale_mask;
        decoded_scales[1] = packed_scales[2] & scale_mask;
        decoded_scales[2] =
            (packed_scales[4] & low_nibble_mask)
            | ((packed_scales[0] & high_bits_mask) >> 2u);
        decoded_scales[3] =
            ((packed_scales[4] >> 4u) & low_nibble_mask)
            | ((packed_scales[2] & high_bits_mask) >> 2u);
        const thread uchar* scales =
            reinterpret_cast<const thread uchar*>(decoded_scales);

        const device ushort* low_quants =
            reinterpret_cast<const device ushort*>(block->quants)
            + 16u * quant_half
            + 4u * quant_lane;
        const device ushort* high_quants = low_quants + 32u;
        float4 low_dot = float4(0.0f);
        float4 high_dot = float4(0.0f);
        #pragma unroll
        for (uint index = 0u; index < 4u; index++) {
            ushort low = low_quants[index];
            ushort high = high_quants[index];
            low_dot[0] +=
                low_values[2u * index] * float(low & 0x000f);
            low_dot[1] +=
                low_values[2u * index + 1u] * float(low & 0x0f00);
            low_dot[2] +=
                low_values[2u * index + 8u] * float(low & 0x00f0);
            low_dot[3] +=
                low_values[2u * index + 9u] * float(low & 0xf000);
            high_dot[0] +=
                high_values[2u * index] * float(high & 0x000f);
            high_dot[1] +=
                high_values[2u * index + 1u] * float(high & 0x0f00);
            high_dot[2] +=
                high_values[2u * index + 8u] * float(high & 0x00f0);
            high_dot[3] +=
                high_values[2u * index + 9u] * float(high & 0xf000);
        }

        float scaled_quant_dot =
            (low_dot[0] + low_dot[1] / 256.0f)
                * float(scales[0])
            + (low_dot[2] + low_dot[3] / 256.0f)
                * float(scales[1]) / 16.0f
            + (high_dot[0] + high_dot[1] / 256.0f)
                * float(scales[4])
            + (high_dot[2] + high_dot[3] / 256.0f)
                * float(scales[5]) / 16.0f;
        float scaled_minimum_dot =
            activation_sums[0] * float(scales[2])
            + activation_sums[1] * float(scales[3])
            + activation_sums[2] * float(scales[6])
            + activation_sums[3] * float(scales[7]);
        row_sums[local_row] =
            float(block->d) * scaled_quant_dot
            - float(block->dmin) * scaled_minimum_dot;
    }
    return row_sums;
}

static inline float2 laguna_xs_q4_decode_two_rows_dot(
    const device LagunaXsQ4KBlock* first_row,
    const device LagunaXsQ4KBlock* second_row,
    const device float* input,
    uint blocks_per_row,
    ushort simd_lane
) {
    uint block_lane = uint(simd_lane) / 8u;
    float2 row_sums = float2(0.0f);
    for (uint block_index = block_lane;
         block_index < blocks_per_row;
         block_index += 4u) {
        row_sums += laguna_xs_q4_decode_two_rows_block_dot(
            first_row,
            second_row,
            input,
            block_index,
            simd_lane);
    }
    return row_sums;
}

static inline float2 laguna_xs_q4_decode_hidden_two_rows_dot(
    const device LagunaXsQ4KBlock* first_row,
    const device LagunaXsQ4KBlock* second_row,
    const device float* input,
    ushort simd_lane
) {
    uint block_lane = uint(simd_lane) / 8u;
    return laguna_xs_q4_decode_two_rows_block_dot(
        first_row,
        second_row,
        input,
        block_lane,
        simd_lane)
        + laguna_xs_q4_decode_two_rows_block_dot(
            first_row,
            second_row,
            input,
            block_lane + 4u,
            simd_lane);
}

static inline float2 laguna_xs_q4_decode_row_pair_dot(
    const device LagunaXsQ4KBlock* weights,
    const device float* input,
    uint blocks_per_row,
    uint first_output_row,
    ushort simd_lane
) {
    const device LagunaXsQ4KBlock* first_row =
        weights + first_output_row * blocks_per_row;
    return laguna_xs_q4_decode_two_rows_dot(
        first_row,
        first_row + blocks_per_row,
        input,
        blocks_per_row,
        simd_lane);
}

kernel void laguna_xs_q4_k_decode_matvec_f32_kernel(
    const device LagunaXsQ4KBlock* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    if (row_count != 1u) {
        return;
    }
    uint output_pair = threadgroup_position.x
        * LAGUNA_XS_SIMDGROUPS_PER_THREADGROUP
        + uint(simdgroup_index);
    uint first_output_row = output_pair * 2u;
    if (first_output_row + 1u >= out_features) {
        return;
    }
    uint blocks_per_row = in_features / LAGUNA_XS_K_BLOCK_VALUES;
    float2 lane_dot = laguna_xs_q4_decode_row_pair_dot(
        weights,
        input,
        blocks_per_row,
        first_output_row,
        simd_lane);
    float first_sum = simd_sum(lane_dot.x);
    float second_sum = simd_sum(lane_dot.y);
    if (simd_lane == 0u) {
        output[first_output_row] = first_sum;
        output[first_output_row + 1u] = second_sum;
    }
}

static inline float2 laguna_xs_q4_decode_row_pair(
    const device LagunaXsQ4KBlock* weights,
    const device float* input,
    uint blocks_per_row,
    uint first_output_row,
    ushort simd_lane
) {
    return laguna_xs_q4_decode_row_pair_dot(
        weights,
        input,
        blocks_per_row,
        first_output_row,
        simd_lane);
}

static inline float2 laguna_xs_q4_decode_hidden_row_pair(
    const device LagunaXsQ4KBlock* weights,
    const device float* input,
    uint first_output_row,
    ushort simd_lane
) {
    const device LagunaXsQ4KBlock* first_row =
        weights + first_output_row * LAGUNA_XS_HIDDEN_BLOCKS;
    return laguna_xs_q4_decode_hidden_two_rows_dot(
        first_row,
        first_row + LAGUNA_XS_HIDDEN_BLOCKS,
        input,
        simd_lane);
}

static inline float2 laguna_xs_q4_decode_hidden_or_dynamic_row_pair(
    const device LagunaXsQ4KBlock* weights,
    const device float* input,
    uint blocks_per_row,
    uint first_output_row,
    ushort simd_lane
) {
    if (blocks_per_row == LAGUNA_XS_HIDDEN_BLOCKS) {
        return laguna_xs_q4_decode_hidden_row_pair(
            weights,
            input,
            first_output_row,
            simd_lane);
    }
    return laguna_xs_q4_decode_row_pair(
        weights,
        input,
        blocks_per_row,
        first_output_row,
        simd_lane);
}

static inline float2 laguna_xs_q6_decode_two_rows_block_dot(
    const device LagunaXsQ6KBlock* first_row,
    const device LagunaXsQ6KBlock* second_row,
    const device float* input,
    uint block_index,
    ushort simd_lane
) {
    uint lane_pair = uint(simd_lane) / 2u;
    uint half_index = lane_pair / 8u;
    uint lane_in_half = lane_pair % 8u;
    uint value_offset = 4u * lane_in_half;
    uint scale_offset =
        8u * half_index + value_offset / 16u;
    float2 row_sums = float2(0.0f);

    const device LagunaXsQ6KBlock* first =
        first_row + block_index;
    const device LagunaXsQ6KBlock* second =
        second_row + block_index;
    const device float* activation = input
        + block_index * LAGUNA_XS_K_BLOCK_VALUES
        + 128u * half_index
        + value_offset;
    float activation_values[16];
    #pragma unroll
    for (uint index = 0u; index < 4u; index++) {
        activation_values[4u * index] = activation[index];
        activation_values[4u * index + 1u] =
            activation[index + 32u];
        activation_values[4u * index + 2u] =
            activation[index + 64u];
        activation_values[4u * index + 3u] =
            activation[index + 96u];
    }

    #pragma unroll
    for (uint local_row = 0u; local_row < 2u; local_row++) {
        const device LagunaXsQ6KBlock* block =
            local_row == 0u ? first : second;
        const device uchar* low = block->low_quants
            + 64u * half_index
            + value_offset;
        const device uchar* low_second = low + 32u;
        const device uchar* high = block->high_quants
            + 32u * half_index
            + value_offset;
        const device char* scales =
            block->scales + scale_offset;
        float4 quant_sums = float4(0.0f);

        #pragma unroll
        for (uint index = 0u; index < 4u; index++) {
            int first_quant = int(
                (low[index] & 15u)
                | ((high[index] & 3u) << 4u)) - 32;
            int second_quant = int(
                (low_second[index] & 15u)
                | ((high[index] & 12u) << 2u)) - 32;
            int third_quant = int(
                (low[index] >> 4u)
                | (high[index] & 48u)) - 32;
            int fourth_quant = int(
                (low_second[index] >> 4u)
                | ((high[index] & 192u) >> 2u)) - 32;
            quant_sums[0] = fma(
                activation_values[4u * index],
                float(first_quant),
                quant_sums[0]);
            quant_sums[1] = fma(
                activation_values[4u * index + 1u],
                float(second_quant),
                quant_sums[1]);
            quant_sums[2] = fma(
                activation_values[4u * index + 2u],
                float(third_quant),
                quant_sums[2]);
            quant_sums[3] = fma(
                activation_values[4u * index + 3u],
                float(fourth_quant),
                quant_sums[3]);
        }
        row_sums[local_row] = float(block->d)
            * (quant_sums[0] * float(scales[0])
                + quant_sums[1] * float(scales[2])
                + quant_sums[2] * float(scales[4])
                + quant_sums[3] * float(scales[6]));
    }
    return row_sums;
}

static inline float2 laguna_xs_q6_decode_row_pair(
    const device LagunaXsQ6KBlock* weights,
    const device float* input,
    uint blocks_per_row,
    uint first_output_row,
    ushort simd_lane
) {
    uint block_lane = uint(simd_lane) & 1u;
    const device LagunaXsQ6KBlock* first_row =
        weights + first_output_row * blocks_per_row;
    const device LagunaXsQ6KBlock* second_row =
        first_row + blocks_per_row;
    float2 row_sums = float2(0.0f);
    for (uint block_index = block_lane;
         block_index < blocks_per_row;
         block_index += 2u) {
        row_sums += laguna_xs_q6_decode_two_rows_block_dot(
            first_row,
            second_row,
            input,
            block_index,
            simd_lane);
    }
    return row_sums;
}

static inline float2 laguna_xs_q6_decode_hidden_row_pair(
    const device LagunaXsQ6KBlock* weights,
    const device float* input,
    uint first_output_row,
    ushort simd_lane
) {
    const device LagunaXsQ6KBlock* first_row =
        weights + first_output_row * LAGUNA_XS_HIDDEN_BLOCKS;
    const device LagunaXsQ6KBlock* second_row =
        first_row + LAGUNA_XS_HIDDEN_BLOCKS;
    uint block_lane = uint(simd_lane) & 1u;
    return laguna_xs_q6_decode_two_rows_block_dot(
        first_row,
        second_row,
        input,
        block_lane,
        simd_lane)
        + laguna_xs_q6_decode_two_rows_block_dot(
            first_row,
            second_row,
            input,
            block_lane + 2u,
            simd_lane)
        + laguna_xs_q6_decode_two_rows_block_dot(
            first_row,
            second_row,
            input,
            block_lane + 4u,
            simd_lane)
        + laguna_xs_q6_decode_two_rows_block_dot(
            first_row,
            second_row,
            input,
            block_lane + 6u,
            simd_lane);
}

kernel void laguna_xs_q6_k_decode_matvec_f32_kernel(
    const device LagunaXsQ6KBlock* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    if (row_count != 1u) {
        return;
    }
    uint output_pair = threadgroup_position.x
        * LAGUNA_XS_SIMDGROUPS_PER_THREADGROUP
        + uint(simdgroup_index);
    uint first_output_row = output_pair * 2u;
    if (first_output_row + 1u >= out_features) {
        return;
    }
    uint blocks_per_row = in_features / LAGUNA_XS_K_BLOCK_VALUES;
    float2 lane_dot = laguna_xs_q6_decode_row_pair(
        weights,
        input,
        blocks_per_row,
        first_output_row,
        simd_lane);
    float first_sum = simd_sum(lane_dot.x);
    float second_sum = simd_sum(lane_dot.y);
    if (simd_lane == 0u) {
        output[first_output_row] = first_sum;
        output[first_output_row + 1u] = second_sum;
    }
}

static inline float2 laguna_xs_decode_value_row_pair(
    const device LagunaXsQ4KBlock* weights,
    const device float* input,
    uint blocks_per_row,
    uint first_output_row,
    ushort simd_lane
) {
    return laguna_xs_q4_decode_row_pair(
        weights,
        input,
        blocks_per_row,
        first_output_row,
        simd_lane);
}

static inline float2 laguna_xs_decode_hidden_value_row_pair(
    const device LagunaXsQ4KBlock* weights,
    const device float* input,
    uint blocks_per_row,
    uint first_output_row,
    ushort simd_lane
) {
    return laguna_xs_q4_decode_hidden_or_dynamic_row_pair(
        weights,
        input,
        blocks_per_row,
        first_output_row,
        simd_lane);
}

static inline float2 laguna_xs_decode_hidden_value_row_pair(
    const device LagunaXsQ6KBlock* weights,
    const device float* input,
    uint blocks_per_row,
    uint first_output_row,
    ushort simd_lane
) {
    if (blocks_per_row == LAGUNA_XS_HIDDEN_BLOCKS) {
        return laguna_xs_q6_decode_hidden_row_pair(
            weights,
            input,
            first_output_row,
            simd_lane);
    }
    return laguna_xs_q6_decode_row_pair(
        weights,
        input,
        blocks_per_row,
        first_output_row,
        simd_lane);
}

static inline float2 laguna_xs_decode_value_row_pair(
    const device LagunaXsQ6KBlock* weights,
    const device float* input,
    uint blocks_per_row,
    uint first_output_row,
    ushort simd_lane
) {
    return laguna_xs_q6_decode_row_pair(
        weights,
        input,
        blocks_per_row,
        first_output_row,
        simd_lane);
}

template <typename Block>
static inline void laguna_xs_decode_matvec_residuals(
    const device Block* weights,
    const device float* input,
    const device float* residual_a,
    const device float* residual_b,
    device float* output,
    uint row_count,
    uint in_features,
    uint out_features,
    uint residual_count,
    uint3 threadgroup_position,
    ushort simd_lane,
    ushort simdgroup_index
) {
    if (row_count != 1u) {
        return;
    }
    uint output_pair = threadgroup_position.x
        * LAGUNA_XS_SIMDGROUPS_PER_THREADGROUP
        + uint(simdgroup_index);
    uint first_output_row = output_pair * 2u;
    if (first_output_row + 1u >= out_features) {
        return;
    }
    uint blocks_per_row = in_features / LAGUNA_XS_K_BLOCK_VALUES;
    float2 lane_dot = laguna_xs_decode_value_row_pair(
        weights,
        input,
        blocks_per_row,
        first_output_row,
        simd_lane);
    float first_sum = simd_sum(lane_dot.x);
    float second_sum = simd_sum(lane_dot.y);
    if (simd_lane == 0u) {
        float2 result = float2(first_sum, second_sum)
            + float2(
                residual_a[first_output_row],
                residual_a[first_output_row + 1u]);
        if (residual_count == 2u) {
            result += float2(
                residual_b[first_output_row],
                residual_b[first_output_row + 1u]);
        }
        output[first_output_row] = result.x;
        output[first_output_row + 1u] = result.y;
    }
}

kernel void laguna_xs_q4_k_decode_matvec_residuals_f32_kernel(
    const device LagunaXsQ4KBlock* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    const device float* residual_a [[buffer(2)]],
    const device float* residual_b [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& in_features [[buffer(6)]],
    constant uint& out_features [[buffer(7)]],
    constant uint& residual_count [[buffer(8)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    laguna_xs_decode_matvec_residuals<LagunaXsQ4KBlock>(
        weights,
        input,
        residual_a,
        residual_b,
        output,
        row_count,
        in_features,
        out_features,
        residual_count,
        threadgroup_position,
        simd_lane,
        simdgroup_index);
}

kernel void laguna_xs_q6_k_decode_matvec_residuals_f32_kernel(
    const device LagunaXsQ6KBlock* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    const device float* residual_a [[buffer(2)]],
    const device float* residual_b [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& in_features [[buffer(6)]],
    constant uint& out_features [[buffer(7)]],
    constant uint& residual_count [[buffer(8)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    laguna_xs_decode_matvec_residuals<LagunaXsQ6KBlock>(
        weights,
        input,
        residual_a,
        residual_b,
        output,
        row_count,
        in_features,
        out_features,
        residual_count,
        threadgroup_position,
        simd_lane,
        simdgroup_index);
}

kernel void laguna_xs_q4_gate_up_swiglu_decode_f32_kernel(
    const device LagunaXsQ4KBlock* gate_weights [[buffer(0)]],
    const device LagunaXsQ4KBlock* up_weights [[buffer(1)]],
    const device float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    if (row_count != 1u) {
        return;
    }
    uint output_row = threadgroup_position.x
        * LAGUNA_XS_SIMDGROUPS_PER_THREADGROUP
        + uint(simdgroup_index);
    if (output_row >= out_features) {
        return;
    }
    uint blocks_per_row = in_features / LAGUNA_XS_K_BLOCK_VALUES;
    float2 gate_up = blocks_per_row == LAGUNA_XS_HIDDEN_BLOCKS
        ? laguna_xs_q4_decode_hidden_two_rows_dot(
            gate_weights + output_row * blocks_per_row,
            up_weights + output_row * blocks_per_row,
            input,
            simd_lane)
        : laguna_xs_q4_decode_two_rows_dot(
            gate_weights + output_row * blocks_per_row,
            up_weights + output_row * blocks_per_row,
            input,
            blocks_per_row,
            simd_lane);
    float gate = simd_sum(gate_up.x);
    float up = simd_sum(gate_up.y);
    if (simd_lane == 0u) {
        output[output_row] = gate / (1.0f + exp(-gate)) * up;
    }
}

/*
 * Decode-only top-8 selection for Laguna XS's fixed 256-expert router.
 * Lane i owns experts i, i+32, ..., i+224. It keeps those eight values in
 * registers; SIMD reductions merge the 32 lane-local winners for each rank.
 */
kernel void laguna_xs_router_top8_decode_f32_kernel(
    const device float* router_logits [[buffer(0)]],
    const device float* correction_bias [[buffer(1)]],
    device uint* expert_ids [[buffer(2)]],
    device float* expert_weights [[buffer(3)]],
    device uint* token_indices [[buffer(4)]],
    constant float& routed_scaling_factor [[buffer(5)]],
    ushort simd_lane [[thread_index_in_simdgroup]]
) {
    constexpr uint simd_lanes = 32;
    constexpr uint experts_per_lane = 8;
    constexpr uint top_k = 8;

    float local_corrected[experts_per_lane];
    float local_scores[experts_per_lane];
    uint local_ids[experts_per_lane];
    bool local_selected[experts_per_lane];
    #pragma unroll
    for (uint local = 0; local < experts_per_lane; local++) {
        uint expert = uint(simd_lane) + (local * simd_lanes);
        float score = 1.0f / (1.0f + exp(-router_logits[expert]));
        local_corrected[local] = score + correction_bias[expert];
        local_scores[local] = score;
        local_ids[local] = expert;
        local_selected[local] = false;
    }

    float selected_scores[top_k];
    uint selected_ids[top_k];
    #pragma unroll
    for (uint rank = 0; rank < top_k; rank++) {
        float lane_corrected = -3.402823466e+38F;
        float lane_score = 0.0f;
        uint lane_id = 0xffffffffu;
        uint lane_local_index = experts_per_lane;
        #pragma unroll
        for (uint local = 0; local < experts_per_lane; local++) {
            if (local_selected[local]) {
                continue;
            }
            float corrected = local_corrected[local];
            uint expert = local_ids[local];
            bool better = corrected > lane_corrected
                || (corrected == lane_corrected && expert < lane_id);
            if (better) {
                lane_corrected = corrected;
                lane_score = local_scores[local];
                lane_id = expert;
                lane_local_index = local;
            }
        }

        float best_corrected = simd_max(lane_corrected);
        uint candidate_id = lane_corrected == best_corrected
            ? lane_id
            : 0xffffffffu;
        uint best_id = simd_min(candidate_id);
        uint winning_lane = best_id % simd_lanes;
        float best_score = simd_broadcast(lane_score, winning_lane);
        if (lane_local_index < experts_per_lane && lane_id == best_id) {
            local_selected[lane_local_index] = true;
        }
        if (simd_lane == 0u) {
            selected_scores[rank] = best_score;
            selected_ids[rank] = best_id;
        }
    }

    if (simd_lane == 0u) {
        float weight_sum = 0.0f;
        #pragma unroll
        for (uint rank = 0; rank < top_k; rank++) {
            weight_sum += selected_scores[rank];
        }
        #pragma unroll
        for (uint rank = 0; rank < top_k; rank++) {
            expert_ids[rank] = selected_ids[rank];
            expert_weights[rank] =
                (selected_scores[rank] / weight_sum) * routed_scaling_factor;
            token_indices[rank] = 0u;
        }
    }
}

/*
 * Decode-only Q/K RMSNorm and half-split RoPE for Laguna XS.
 * One lane owns one contiguous float4, so every 128-wide head is processed by
 * exactly one SIMD group with vector loads, vector transcendental operations,
 * and one float4 store.
 */
kernel void laguna_xs_qk_norm_rope_pair_decode_f32_kernel(
    const device float* query_input [[buffer(0)]],
    const device float* key_input [[buffer(1)]],
    const device float* query_norm_weight [[buffer(2)]],
    const device float* key_norm_weight [[buffer(3)]],
    const device float* inverse_frequency [[buffer(4)]],
    device float* query_output [[buffer(5)]],
    device float* key_output [[buffer(6)]],
    constant uint& query_head_count [[buffer(7)]],
    constant uint& rotary_dim [[buffer(8)]],
    constant uint& position_offset [[buffer(9)]],
    constant float& epsilon [[buffer(10)]],
    constant float& attention_factor [[buffer(11)]],
    uint row [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]]
) {
    constexpr uint head_dim = 128;
    constexpr uint key_head_count = 8;
    bool is_query = row < query_head_count;
    uint local_row = is_query ? row : row - query_head_count;
    if (!is_query && local_row >= key_head_count) {
        return;
    }

    const device float* input = is_query ? query_input : key_input;
    const device float* norm_weight =
        is_query ? query_norm_weight : key_norm_weight;
    device float* output = is_query ? query_output : key_output;
    uint dim_base = uint(simd_lane) * 4u;
    uint row_base = local_row * head_dim;
    float4 values = *reinterpret_cast<const device float4*>(
        input + row_base + dim_base);
    float inverse_rms = rsqrt(
        simd_sum(dot(values, values)) / float(head_dim) + epsilon);
    float4 normalized = values
        * (*reinterpret_cast<const device float4*>(norm_weight + dim_base))
        * inverse_rms;

    if (dim_base < rotary_dim) {
        uint rotary_half = rotary_dim / 2u;
        bool lower_half = dim_base < rotary_half;
        uint paired_dim = lower_half
            ? dim_base + rotary_half
            : dim_base - rotary_half;
        float4 paired = *reinterpret_cast<const device float4*>(
            input + row_base + paired_dim);
        paired *= *reinterpret_cast<const device float4*>(
            norm_weight + paired_dim);
        paired *= inverse_rms;
        float4 rotated = lower_half ? -paired : paired;
        uint frequency_base = dim_base % rotary_half;
        float4 angle = float(position_offset)
            * (*reinterpret_cast<const device float4*>(
                inverse_frequency + frequency_base));
        normalized = normalized * (cos(angle) * attention_factor)
            + rotated * (sin(angle) * attention_factor);
    }

    *reinterpret_cast<device float4*>(output + row_base + dim_base) = normalized;
}

/*
 * Decode-only RMSNorm for Laguna XS hidden rows [1,2048]. Each of the 256
 * threads owns two contiguous float4 values. Accumulation remains F32.
 */
kernel void laguna_xs_rms_norm_decode_f32_kernel(
    const device float* input [[buffer(0)]],
    const device float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint hidden_size = 2048u;
    constexpr uint values_per_thread = 8u;
    constexpr uint simdgroup_count = 8u;
    uint value_base = uint(thread_index) * values_per_thread;

    float4 first = *reinterpret_cast<const device float4*>(
        input + value_base);
    float4 second = *reinterpret_cast<const device float4*>(
        input + value_base + 4u);
    float local_sum = dot(first, first) + dot(second, second);
    float simd_sum_value = simd_sum(local_sum);

    threadgroup float partial_sums[simdgroup_count];
    if (simd_lane == 0u) {
        partial_sums[simdgroup_index] = simd_sum_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float partial = simd_lane < simdgroup_count
        ? partial_sums[simd_lane]
        : 0.0f;
    float inverse_rms = rsqrt(
        simd_sum(partial) / float(hidden_size) + epsilon);
    float4 first_weight = *reinterpret_cast<const device float4*>(
        weight + value_base);
    float4 second_weight = *reinterpret_cast<const device float4*>(
        weight + value_base + 4u);
    *reinterpret_cast<device float4*>(output + value_base) =
        first * first_weight * inverse_rms;
    *reinterpret_cast<device float4*>(output + value_base + 4u) =
        second * second_weight * inverse_rms;
}

/*
 * Decode projects Q, K, V, and the attention gate from the same normalized
 * hidden row. Each SIMD group processes two adjacent rows within one physical
 * tensor, so packed Q4 activation fragments are reused without crossing GGUF
 * tensor boundaries. Q, K, and gate are Q4_K; V is Q4_K or Q6_K by layer.
 */
template <typename ValueBlock>
static inline void laguna_xs_attention_projections(
    const device LagunaXsQ4KBlock* query_weights,
    const device LagunaXsQ4KBlock* key_weights,
    const device ValueBlock* value_weights,
    const device LagunaXsQ4KBlock* gate_weights,
    const device float* input,
    device float* query_output,
    device float* key_output,
    device float* value_output,
    device float* gate_output,
    uint in_features,
    uint query_features,
    uint key_features,
    uint value_features,
    uint gate_features,
    uint3 threadgroup_position,
    ushort simd_lane,
    ushort simdgroup_index
) {
    uint output_pair = threadgroup_position.x
        * LAGUNA_XS_SIMDGROUPS_PER_THREADGROUP
        + uint(simdgroup_index);
    uint query_pairs = query_features / 2u;
    uint key_pairs = key_features / 2u;
    uint value_pairs = value_features / 2u;
    uint gate_pairs = gate_features / 2u;
    uint total_pairs =
        query_pairs + key_pairs + value_pairs + gate_pairs;
    if (output_pair >= total_pairs) {
        return;
    }

    uint blocks_per_row = in_features / LAGUNA_XS_K_BLOCK_VALUES;
    float2 lane_dot;
    device float* destination;
    uint first_output_row;

    if (output_pair < query_pairs) {
        first_output_row = output_pair * 2u;
        lane_dot = laguna_xs_q4_decode_hidden_or_dynamic_row_pair(
            query_weights,
            input,
            blocks_per_row,
            first_output_row,
            simd_lane);
        destination = query_output;
    } else if (output_pair < query_pairs + key_pairs) {
        uint local_pair = output_pair - query_pairs;
        first_output_row = local_pair * 2u;
        lane_dot = laguna_xs_q4_decode_hidden_or_dynamic_row_pair(
            key_weights,
            input,
            blocks_per_row,
            first_output_row,
            simd_lane);
        destination = key_output;
    } else if (output_pair < query_pairs + key_pairs + value_pairs) {
        uint local_pair = output_pair - query_pairs - key_pairs;
        first_output_row = local_pair * 2u;
        lane_dot = laguna_xs_decode_hidden_value_row_pair(
            value_weights,
            input,
            blocks_per_row,
            first_output_row,
            simd_lane);
        destination = value_output;
    } else {
        uint local_pair =
            output_pair - query_pairs - key_pairs - value_pairs;
        first_output_row = local_pair * 2u;
        lane_dot = laguna_xs_q4_decode_hidden_or_dynamic_row_pair(
            gate_weights,
            input,
            blocks_per_row,
            first_output_row,
            simd_lane);
        destination = gate_output;
    }

    float first_sum = simd_sum(lane_dot.x);
    float second_sum = simd_sum(lane_dot.y);
    if (simd_lane == 0u) {
        destination[first_output_row] = first_sum;
        destination[first_output_row + 1u] = second_sum;
    }
}

kernel void laguna_xs_q4_attention_projections_f32_kernel(
    const device LagunaXsQ4KBlock* query_weights [[buffer(0)]],
    const device LagunaXsQ4KBlock* key_weights [[buffer(1)]],
    const device LagunaXsQ4KBlock* value_weights [[buffer(2)]],
    const device LagunaXsQ4KBlock* gate_weights [[buffer(3)]],
    const device float* input [[buffer(4)]],
    device float* query_output [[buffer(5)]],
    device float* key_output [[buffer(6)]],
    device float* value_output [[buffer(7)]],
    device float* gate_output [[buffer(8)]],
    constant uint& in_features [[buffer(9)]],
    constant uint& query_features [[buffer(10)]],
    constant uint& key_features [[buffer(11)]],
    constant uint& value_features [[buffer(12)]],
    constant uint& gate_features [[buffer(13)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    laguna_xs_attention_projections<LagunaXsQ4KBlock>(
        query_weights,
        key_weights,
        value_weights,
        gate_weights,
        input,
        query_output,
        key_output,
        value_output,
        gate_output,
        in_features,
        query_features,
        key_features,
        value_features,
        gate_features,
        threadgroup_position,
        simd_lane,
        simdgroup_index);
}

kernel void laguna_xs_q6_value_attention_projections_f32_kernel(
    const device LagunaXsQ4KBlock* query_weights [[buffer(0)]],
    const device LagunaXsQ4KBlock* key_weights [[buffer(1)]],
    const device LagunaXsQ6KBlock* value_weights [[buffer(2)]],
    const device LagunaXsQ4KBlock* gate_weights [[buffer(3)]],
    const device float* input [[buffer(4)]],
    device float* query_output [[buffer(5)]],
    device float* key_output [[buffer(6)]],
    device float* value_output [[buffer(7)]],
    device float* gate_output [[buffer(8)]],
    constant uint& in_features [[buffer(9)]],
    constant uint& query_features [[buffer(10)]],
    constant uint& key_features [[buffer(11)]],
    constant uint& value_features [[buffer(12)]],
    constant uint& gate_features [[buffer(13)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    laguna_xs_attention_projections<LagunaXsQ6KBlock>(
        query_weights,
        key_weights,
        value_weights,
        gate_weights,
        input,
        query_output,
        key_output,
        value_output,
        gate_output,
        in_features,
        query_features,
        key_features,
        value_features,
        gate_features,
        threadgroup_position,
        simd_lane,
        simdgroup_index);
}

static inline float laguna_xs_q4_element(
    const device LagunaXsQ4KBlock* row,
    uint value_index
) {
    uint block_index = value_index / LAGUNA_XS_K_BLOCK_VALUES;
    uint position = value_index
        - block_index * LAGUNA_XS_K_BLOCK_VALUES;
    uint group = position / 32u;
    uint lane = position - group * 32u;
    const device LagunaXsQ4KBlock* block = row + block_index;
    uint packed_index = (group >> 1u) * 32u + lane;
    uint shift = (group & 1u) * 4u;
    uint quant = (uint(block->quants[packed_index]) >> shift) & 15u;
    return float(block->d) * float(laguna_xs_q4_scale(block, group))
        * float(quant)
        - float(block->dmin)
            * float(laguna_xs_q4_minimum(block, group));
}

kernel void laguna_xs_q4_k_embedding_f32_kernel(
    const device LagunaXsQ4KBlock* weights [[buffer(0)]],
    const device uint* token_ids [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    constant uint& vocab_size [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint value_count = token_count * hidden_size;
    if (gid >= value_count) {
        return;
    }
    uint token_index = gid / hidden_size;
    uint hidden_index = gid - token_index * hidden_size;
    uint token_id = token_ids[token_index];
    if (token_id >= vocab_size) {
        output[gid] = 0.0f;
        return;
    }

    uint blocks_per_row = hidden_size / LAGUNA_XS_K_BLOCK_VALUES;
    const device LagunaXsQ4KBlock* row =
        weights + token_id * blocks_per_row;
    output[gid] = laguna_xs_q4_element(row, hidden_index);
}

kernel void laguna_xs_q4_expert_gate_up_f32_kernel(
    const device LagunaXsQ4KBlock* gate_weights [[buffer(0)]],
    const device LagunaXsQ4KBlock* up_weights [[buffer(1)]],
    const device float* input [[buffer(2)]],
    const device uint* token_indices [[buffer(3)]],
    const device uint* expert_ids [[buffer(4)]],
    const device float* expert_weights [[buffer(5)]],
    device float* intermediate [[buffer(6)]],
    constant uint& assignment_count [[buffer(7)]],
    constant uint& expert_count [[buffer(8)]],
    constant uint& in_features [[buffer(9)]],
    constant uint& intermediate_features [[buffer(10)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint simdgroup = threadgroup_position.x
        * LAGUNA_XS_SIMDGROUPS_PER_THREADGROUP
        + uint(simdgroup_index);
    uint row_groups = (intermediate_features
        + LAGUNA_XS_EXPERT_GATE_UP_ROWS_PER_SIMDGROUP - 1u)
        / LAGUNA_XS_EXPERT_GATE_UP_ROWS_PER_SIMDGROUP;
    uint assignment = simdgroup / row_groups;
    if (assignment >= assignment_count) {
        return;
    }

    uint expert = expert_ids[assignment];
    if (expert >= expert_count) {
        return;
    }
    uint first_row = (simdgroup - assignment * row_groups)
        * LAGUNA_XS_EXPERT_GATE_UP_ROWS_PER_SIMDGROUP;
    uint blocks_per_row = in_features / LAGUNA_XS_K_BLOCK_VALUES;
    uint blocks_per_expert = intermediate_features * blocks_per_row;
    const device LagunaXsQ4KBlock* expert_gate =
        gate_weights + expert * blocks_per_expert;
    const device LagunaXsQ4KBlock* expert_up =
        up_weights + expert * blocks_per_expert;
    const device float* token_input =
        input + token_indices[assignment] * in_features;
    if (first_row >= intermediate_features) {
        return;
    }
    float2 gate_up = blocks_per_row == LAGUNA_XS_HIDDEN_BLOCKS
        ? laguna_xs_q4_decode_hidden_two_rows_dot(
            expert_gate + first_row * blocks_per_row,
            expert_up + first_row * blocks_per_row,
            token_input,
            simd_lane)
        : laguna_xs_q4_decode_two_rows_dot(
            expert_gate + first_row * blocks_per_row,
            expert_up + first_row * blocks_per_row,
            token_input,
            blocks_per_row,
            simd_lane);
    float gate = simd_sum(gate_up.x);
    float up = simd_sum(gate_up.y);
    if (simd_lane == 0u) {
        float activated_gate = gate / (1.0f + exp(-gate));
        intermediate[assignment * intermediate_features + first_row] =
            activated_gate * up * expert_weights[assignment];
    }
}

template <typename Block, typename Operations>
static inline void laguna_xs_expert_down_sum(
    const device Block* down_weights,
    const device uint* expert_ids,
    const device float* intermediate,
    device float* output,
    uint token_count,
    uint top_k,
    uint expert_count,
    uint intermediate_features,
    uint out_features,
    uint3 threadgroup_position,
    ushort simd_lane,
    ushort simdgroup_index
) {
    uint simdgroup = threadgroup_position.x
        * LAGUNA_XS_SIMDGROUPS_PER_THREADGROUP
        + uint(simdgroup_index);
    uint row_groups = (out_features
        + LAGUNA_XS_EXPERT_DOWN_ROWS_PER_SIMDGROUP - 1u)
        / LAGUNA_XS_EXPERT_DOWN_ROWS_PER_SIMDGROUP;
    uint token = simdgroup / row_groups;
    if (token >= token_count) {
        return;
    }

    uint first_row = (simdgroup - token * row_groups)
        * LAGUNA_XS_EXPERT_DOWN_ROWS_PER_SIMDGROUP;
    uint blocks_per_row = intermediate_features
        / LAGUNA_XS_K_BLOCK_VALUES;
    uint blocks_per_expert = out_features * blocks_per_row;
    float sums[LAGUNA_XS_EXPERT_DOWN_ROWS_PER_SIMDGROUP] = {};

    for (uint slot = 0u; slot < top_k; slot++) {
        uint assignment = token * top_k + slot;
        uint expert = expert_ids[assignment];
        if (expert >= expert_count) {
            continue;
        }

        const device Block* expert_weights =
            down_weights + expert * blocks_per_expert;
        const device float* assignment_input =
            intermediate + assignment * intermediate_features;
        for (uint block_index = 0u;
             block_index < blocks_per_row;
             block_index++) {
            const device float* input_block = assignment_input
                + block_index * LAGUNA_XS_K_BLOCK_VALUES;
            for (uint local_row = 0u;
                 local_row < LAGUNA_XS_EXPERT_DOWN_ROWS_PER_SIMDGROUP;
                 local_row++) {
                uint row = first_row + local_row;
                if (row < out_features) {
                    sums[local_row] += Operations::lane_dot(
                        expert_weights + row * blocks_per_row + block_index,
                        input_block,
                        uint(simd_lane));
                }
            }
        }
    }

    for (uint local_row = 0u;
         local_row < LAGUNA_XS_EXPERT_DOWN_ROWS_PER_SIMDGROUP;
         local_row++) {
        uint row = first_row + local_row;
        float sum = simd_sum(sums[local_row]);
        if (simd_lane == 0u && row < out_features) {
            output[token * out_features + row] = sum;
        }
    }
}

kernel void laguna_xs_q4_expert_down_sum_f32_kernel(
    const device LagunaXsQ4KBlock* down_weights [[buffer(0)]],
    const device uint* expert_ids [[buffer(1)]],
    const device float* intermediate [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& token_count [[buffer(4)]],
    constant uint& top_k [[buffer(5)]],
    constant uint& expert_count [[buffer(6)]],
    constant uint& intermediate_features [[buffer(7)]],
    constant uint& out_features [[buffer(8)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    laguna_xs_expert_down_sum<
        LagunaXsQ4KBlock,
        LagunaXsQ4Operations
    >(
        down_weights,
        expert_ids,
        intermediate,
        output,
        token_count,
        top_k,
        expert_count,
        intermediate_features,
        out_features,
        threadgroup_position,
        simd_lane,
        simdgroup_index);
}

kernel void laguna_xs_q6_expert_down_sum_f32_kernel(
    const device LagunaXsQ6KBlock* down_weights [[buffer(0)]],
    const device uint* expert_ids [[buffer(1)]],
    const device float* intermediate [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& token_count [[buffer(4)]],
    constant uint& top_k [[buffer(5)]],
    constant uint& expert_count [[buffer(6)]],
    constant uint& intermediate_features [[buffer(7)]],
    constant uint& out_features [[buffer(8)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    laguna_xs_expert_down_sum<
        LagunaXsQ6KBlock,
        LagunaXsQ6Operations
    >(
        down_weights,
        expert_ids,
        intermediate,
        output,
        token_count,
        top_k,
        expert_count,
        intermediate_features,
        out_features,
        threadgroup_position,
        simd_lane,
        simdgroup_index);
}

template <typename Block, typename Operations>
static inline void laguna_xs_expert_down_parallel(
    const device Block* down_weights,
    const device uint* expert_ids,
    const device float* intermediate,
    device float* output,
    uint token_count,
    uint top_k,
    uint expert_count,
    uint intermediate_features,
    uint out_features,
    threadgroup float* simdgroup_partials,
    uint3 threadgroup_position,
    ushort simd_lane,
    ushort simdgroup_index
) {
    uint work_index = threadgroup_position.x;
    uint token = work_index / out_features;
    uint row = work_index - token * out_features;
    if (token >= token_count
        || simdgroup_index >= LAGUNA_XS_EXPERT_DOWN_PARALLEL_SIMDGROUPS) {
        return;
    }

    uint blocks_per_row = intermediate_features
        / LAGUNA_XS_K_BLOCK_VALUES;
    uint blocks_per_expert = out_features * blocks_per_row;
    float partial = 0.0f;
    for (uint slot = uint(simdgroup_index);
         slot < top_k;
         slot += LAGUNA_XS_EXPERT_DOWN_PARALLEL_SIMDGROUPS) {
        uint assignment = token * top_k + slot;
        uint expert = expert_ids[assignment];
        if (expert >= expert_count) {
            continue;
        }
        const device Block* expert_row =
            down_weights + expert * blocks_per_expert + row * blocks_per_row;
        const device float* assignment_input =
            intermediate + assignment * intermediate_features;
        for (uint block_index = 0u;
             block_index < blocks_per_row;
             block_index++) {
            partial += Operations::lane_dot(
                expert_row + block_index,
                assignment_input
                    + block_index * LAGUNA_XS_K_BLOCK_VALUES,
                uint(simd_lane));
        }
    }

    float simdgroup_sum = simd_sum(partial);
    if (simd_lane == 0u) {
        simdgroup_partials[simdgroup_index] = simdgroup_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0u) {
        float lane_value =
            simd_lane < LAGUNA_XS_EXPERT_DOWN_PARALLEL_SIMDGROUPS
            ? simdgroup_partials[simd_lane]
            : 0.0f;
        float sum = simd_sum(lane_value);
        if (simd_lane == 0u) {
            output[token * out_features + row] = sum;
        }
    }
}

kernel void laguna_xs_q4_expert_down_parallel_f32_kernel(
    const device LagunaXsQ4KBlock* down_weights [[buffer(0)]],
    const device uint* expert_ids [[buffer(1)]],
    const device float* intermediate [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& token_count [[buffer(4)]],
    constant uint& top_k [[buffer(5)]],
    constant uint& expert_count [[buffer(6)]],
    constant uint& intermediate_features [[buffer(7)]],
    constant uint& out_features [[buffer(8)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float simdgroup_partials[
        LAGUNA_XS_EXPERT_DOWN_PARALLEL_SIMDGROUPS
    ];
    laguna_xs_expert_down_parallel<
        LagunaXsQ4KBlock,
        LagunaXsQ4Operations
    >(
        down_weights,
        expert_ids,
        intermediate,
        output,
        token_count,
        top_k,
        expert_count,
        intermediate_features,
        out_features,
        simdgroup_partials,
        threadgroup_position,
        simd_lane,
        simdgroup_index);
}

kernel void laguna_xs_q6_expert_down_parallel_f32_kernel(
    const device LagunaXsQ6KBlock* down_weights [[buffer(0)]],
    const device uint* expert_ids [[buffer(1)]],
    const device float* intermediate [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& token_count [[buffer(4)]],
    constant uint& top_k [[buffer(5)]],
    constant uint& expert_count [[buffer(6)]],
    constant uint& intermediate_features [[buffer(7)]],
    constant uint& out_features [[buffer(8)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float simdgroup_partials[
        LAGUNA_XS_EXPERT_DOWN_PARALLEL_SIMDGROUPS
    ];
    laguna_xs_expert_down_parallel<
        LagunaXsQ6KBlock,
        LagunaXsQ6Operations
    >(
        down_weights,
        expert_ids,
        intermediate,
        output,
        token_count,
        top_k,
        expert_count,
        intermediate_features,
        out_features,
        simdgroup_partials,
        threadgroup_position,
        simd_lane,
        simdgroup_index);
}
