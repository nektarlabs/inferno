#include <metal_stdlib>

using namespace metal;

constant uint LAGUNA_XS_PREFILL_BLOCK_VALUES = 256u;
constant uint LAGUNA_XS_PREFILL_GROUP_VALUES = 32u;
constant uint LAGUNA_XS_PREFILL_GROUPS_PER_BLOCK = 8u;
constant uint LAGUNA_XS_PREFILL_TOKEN_TILE = 64u;
constant uint LAGUNA_XS_PREFILL_OUTPUT_TILE = 32u;
constant uint LAGUNA_XS_PREFILL_K_TILE = 32u;
constant uint LAGUNA_XS_PREFILL_THREAD_COUNT = 128u;
constant uint LAGUNA_XS_PREFILL_TOKEN_GROUPS = 8u;

struct LagunaXsPrefillQ4KBlock {
    half d;
    half dmin;
    uchar scales[12];
    uchar quants[128];
};

struct LagunaXsPrefillQ6KBlock {
    uchar low_quants[128];
    uchar high_quants[64];
    char scales[16];
    half d;
};

static inline uchar laguna_xs_prefill_q4_scale(
    const device LagunaXsPrefillQ4KBlock* block,
    uint group
) {
    if (group < 4u) {
        return block->scales[group] & 63u;
    }
    return (block->scales[group + 4u] & 15u)
        | ((block->scales[group - 4u] & 192u) >> 2u);
}

static inline uchar laguna_xs_prefill_q4_minimum(
    const device LagunaXsPrefillQ4KBlock* block,
    uint group
) {
    if (group < 4u) {
        return block->scales[group + 4u] & 63u;
    }
    return (block->scales[group + 4u] >> 4u)
        | ((block->scales[group] & 192u) >> 2u);
}

struct LagunaXsPrefillQ4Operations {
    static inline half4 dequantize4(
        const device LagunaXsPrefillQ4KBlock* block,
        uint group,
        uint lane
    ) {
        uint pair = group / 2u;
        uchar4 packed =
            *reinterpret_cast<const device uchar4*>(
                block->quants + pair * 32u + lane);
        uchar4 quantized = (group & 1u) == 0u
            ? packed & uchar4(15u)
            : packed >> uchar4(4u);
        float4 value =
            float(block->d)
                * float(laguna_xs_prefill_q4_scale(block, group))
                * float4(quantized)
            - float(block->dmin)
                * float(laguna_xs_prefill_q4_minimum(block, group));
        return half4(value);
    }
};

struct LagunaXsPrefillQ6Operations {
    static inline half4 dequantize4(
        const device LagunaXsPrefillQ6KBlock* block,
        uint group,
        uint lane
    ) {
        uint half_index = group / 4u;
        uint quarter = group - half_index * 4u;
        uint low_base = half_index * 64u;
        uint high_base = half_index * 32u;
        uchar4 low_even =
            *reinterpret_cast<const device uchar4*>(
                block->low_quants + low_base + lane);
        uchar4 low_odd =
            *reinterpret_cast<const device uchar4*>(
                block->low_quants + low_base + 32u + lane);
        uchar4 high =
            *reinterpret_cast<const device uchar4*>(
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
                    | (
                        ((high >> uchar4(2u)) & uchar4(3u))
                        << uchar4(4u)
                    );
                break;
            case 2u:
                quantized =
                    (low_even >> uchar4(4u))
                    | (
                        ((high >> uchar4(4u)) & uchar4(3u))
                        << uchar4(4u)
                    );
                break;
            default:
                quantized =
                    (low_odd >> uchar4(4u))
                    | (
                        ((high >> uchar4(6u)) & uchar4(3u))
                        << uchar4(4u)
                    );
                break;
        }
        uint scale_index =
            half_index * 8u + quarter * 2u + lane / 16u;
        float4 value =
            float(block->d)
                * float(block->scales[scale_index])
                * (float4(quantized) - 32.0f);
        return half4(value);
    }

};

template <typename Block, typename Operations>
static inline void laguna_xs_stage_prefill_tile(
    const device Block* weights,
    const device float* input,
    threadgroup half* input_tile,
    threadgroup half* weight_tile,
    uint token_start,
    uint output_start,
    uint active_tokens,
    uint in_features,
    uint blocks_per_row,
    uint k_tile_index,
    uint thread_index
) {
    constexpr uint vectors_per_token = LAGUNA_XS_PREFILL_K_TILE / 4u;
    for (uint vector_index = thread_index;
         vector_index
            < LAGUNA_XS_PREFILL_TOKEN_TILE * vectors_per_token;
         vector_index += LAGUNA_XS_PREFILL_THREAD_COUNT) {
        uint token = vector_index / vectors_per_token;
        uint input_vector = vector_index - token * vectors_per_token;
        uint input_column = input_vector * 4u;
        threadgroup half4* destination =
            reinterpret_cast<threadgroup half4*>(
                input_tile
                    + token * LAGUNA_XS_PREFILL_K_TILE
                    + input_column);
        if (token < active_tokens) {
            const device float4* source =
                reinterpret_cast<const device float4*>(
                    input
                        + (token_start + token) * in_features
                        + k_tile_index * LAGUNA_XS_PREFILL_K_TILE
                        + input_column);
            *destination = half4(*source);
        } else {
            *destination = half4(0.0h);
        }
    }

    constexpr uint vectors_per_output = LAGUNA_XS_PREFILL_K_TILE / 4u;
    for (uint vector_index = thread_index;
         vector_index
            < LAGUNA_XS_PREFILL_OUTPUT_TILE * vectors_per_output;
         vector_index += LAGUNA_XS_PREFILL_THREAD_COUNT) {
        uint output_feature = vector_index / vectors_per_output;
        uint vector =
            vector_index - output_feature * vectors_per_output;
        uint lane = vector * 4u;
        uint block_in_row =
            k_tile_index / LAGUNA_XS_PREFILL_GROUPS_PER_BLOCK;
        uint group =
            k_tile_index
                - block_in_row * LAGUNA_XS_PREFILL_GROUPS_PER_BLOCK;
        const device Block* block =
            weights
                + (output_start + output_feature) * blocks_per_row
                + block_in_row;
        *reinterpret_cast<threadgroup half4*>(
            weight_tile
                + output_feature * LAGUNA_XS_PREFILL_K_TILE
                + lane
        ) = Operations::dequantize4(block, group, lane);
    }
}

template <typename Block, typename Operations, ushort ResidualCount>
kernel void laguna_xs_prefill_mma_impl(
    const device Block* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    const device float* residual_a [[buffer(2)]],
    const device float* residual_b [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& in_features [[buffer(6)]],
    constant uint& out_features [[buffer(7)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint output_tiles = out_features / LAGUNA_XS_PREFILL_OUTPUT_TILE;
    uint token_tile = threadgroup_position.x / output_tiles;
    uint output_tile = threadgroup_position.x - token_tile * output_tiles;
    uint token_start = token_tile * LAGUNA_XS_PREFILL_TOKEN_TILE;
    uint output_start = output_tile * LAGUNA_XS_PREFILL_OUTPUT_TILE;
    uint active_tokens =
        min(LAGUNA_XS_PREFILL_TOKEN_TILE, row_count - token_start);
    uint blocks_per_row =
        in_features / LAGUNA_XS_PREFILL_BLOCK_VALUES;
    uint k_tile_count = in_features / LAGUNA_XS_PREFILL_K_TILE;

    threadgroup half input_tile[
        LAGUNA_XS_PREFILL_TOKEN_TILE * LAGUNA_XS_PREFILL_K_TILE
    ];
    threadgroup half weight_tile[
        LAGUNA_XS_PREFILL_OUTPUT_TILE * LAGUNA_XS_PREFILL_K_TILE
    ];
    threadgroup float output_tile_values[
        LAGUNA_XS_PREFILL_TOKEN_TILE * LAGUNA_XS_PREFILL_OUTPUT_TILE
    ];
    simdgroup_float8x8 accumulators[LAGUNA_XS_PREFILL_TOKEN_GROUPS];
    for (uint group = 0u;
         group < LAGUNA_XS_PREFILL_TOKEN_GROUPS;
         group++) {
        accumulators[group] =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    laguna_xs_stage_prefill_tile<Block, Operations>(
        weights,
        input,
        input_tile,
        weight_tile,
        token_start,
        output_start,
        active_tokens,
        in_features,
        blocks_per_row,
        0u,
        thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k_tile_index = 0u;
         k_tile_index < k_tile_count;
         k_tile_index++) {
        for (uint k_offset = 0u;
             k_offset < LAGUNA_XS_PREFILL_K_TILE;
             k_offset += 8u) {
            simdgroup_half8x8 weight_matrix;
            simdgroup_load(
                weight_matrix,
                weight_tile
                    + simdgroup_index
                        * 8u
                        * LAGUNA_XS_PREFILL_K_TILE
                    + k_offset,
                LAGUNA_XS_PREFILL_K_TILE,
                0,
                true);
            for (uint group = 0u;
                 group < LAGUNA_XS_PREFILL_TOKEN_GROUPS;
                 group++) {
                simdgroup_half8x8 input_matrix;
                simdgroup_load(
                    input_matrix,
                    input_tile
                        + group
                            * 8u
                            * LAGUNA_XS_PREFILL_K_TILE
                        + k_offset,
                    LAGUNA_XS_PREFILL_K_TILE,
                    0,
                    false);
                simdgroup_multiply_accumulate(
                    accumulators[group],
                    input_matrix,
                    weight_matrix,
                    accumulators[group]);
            }
        }

        uint next_k_tile = k_tile_index + 1u;
        if (next_k_tile < k_tile_count) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            laguna_xs_stage_prefill_tile<Block, Operations>(
                weights,
                input,
                input_tile,
                weight_tile,
                token_start,
                output_start,
                active_tokens,
                in_features,
                blocks_per_row,
                next_k_tile,
                thread_index);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint group = 0u;
         group < LAGUNA_XS_PREFILL_TOKEN_GROUPS;
         group++) {
        simdgroup_store(
            accumulators[group],
            output_tile_values
                + group * 8u * LAGUNA_XS_PREFILL_OUTPUT_TILE
                + simdgroup_index * 8u,
            LAGUNA_XS_PREFILL_OUTPUT_TILE,
            0,
            false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = thread_index;
         index < active_tokens * LAGUNA_XS_PREFILL_OUTPUT_TILE;
         index += LAGUNA_XS_PREFILL_THREAD_COUNT) {
        uint token = index / LAGUNA_XS_PREFILL_OUTPUT_TILE;
        uint output_feature =
            index - token * LAGUNA_XS_PREFILL_OUTPUT_TILE;
        uint output_index =
            (token_start + token) * out_features
                + output_start
                + output_feature;
        float value = output_tile_values[index];
        if (ResidualCount >= 1) {
            value += residual_a[output_index];
        }
        if (ResidualCount >= 2) {
            value += residual_b[output_index];
        }
        output[output_index] = value;
    }
}

typedef decltype(
    laguna_xs_prefill_mma_impl<
        LagunaXsPrefillQ4KBlock,
        LagunaXsPrefillQ4Operations,
        0>)
    LagunaXsQ4PrefillMma;
typedef decltype(
    laguna_xs_prefill_mma_impl<
        LagunaXsPrefillQ6KBlock,
        LagunaXsPrefillQ6Operations,
        0>)
    LagunaXsQ6PrefillMma;
typedef decltype(
    laguna_xs_prefill_mma_impl<
        LagunaXsPrefillQ4KBlock,
        LagunaXsPrefillQ4Operations,
        1>)
    LagunaXsQ4PrefillMmaAdd;
typedef decltype(
    laguna_xs_prefill_mma_impl<
        LagunaXsPrefillQ6KBlock,
        LagunaXsPrefillQ6Operations,
        1>)
    LagunaXsQ6PrefillMmaAdd;
typedef decltype(
    laguna_xs_prefill_mma_impl<
        LagunaXsPrefillQ4KBlock,
        LagunaXsPrefillQ4Operations,
        2>)
    LagunaXsQ4PrefillMmaAdd2;
typedef decltype(
    laguna_xs_prefill_mma_impl<
        LagunaXsPrefillQ6KBlock,
        LagunaXsPrefillQ6Operations,
        2>)
    LagunaXsQ6PrefillMmaAdd2;

template [[host_name("laguna_xs_q4_prefill_mma_f32_kernel")]]
kernel LagunaXsQ4PrefillMma
laguna_xs_prefill_mma_impl<
    LagunaXsPrefillQ4KBlock,
    LagunaXsPrefillQ4Operations,
    0>;

template [[host_name("laguna_xs_q6_prefill_mma_f32_kernel")]]
kernel LagunaXsQ6PrefillMma
laguna_xs_prefill_mma_impl<
    LagunaXsPrefillQ6KBlock,
    LagunaXsPrefillQ6Operations,
    0>;

template [[host_name("laguna_xs_q4_prefill_mma_add_f32_kernel")]]
kernel LagunaXsQ4PrefillMmaAdd
laguna_xs_prefill_mma_impl<
    LagunaXsPrefillQ4KBlock,
    LagunaXsPrefillQ4Operations,
    1>;

template [[host_name("laguna_xs_q6_prefill_mma_add_f32_kernel")]]
kernel LagunaXsQ6PrefillMmaAdd
laguna_xs_prefill_mma_impl<
    LagunaXsPrefillQ6KBlock,
    LagunaXsPrefillQ6Operations,
    1>;

template [[host_name("laguna_xs_q4_prefill_mma_add2_f32_kernel")]]
kernel LagunaXsQ4PrefillMmaAdd2
laguna_xs_prefill_mma_impl<
    LagunaXsPrefillQ4KBlock,
    LagunaXsPrefillQ4Operations,
    2>;

template [[host_name("laguna_xs_q6_prefill_mma_add2_f32_kernel")]]
kernel LagunaXsQ6PrefillMmaAdd2
laguna_xs_prefill_mma_impl<
    LagunaXsPrefillQ6KBlock,
    LagunaXsPrefillQ6Operations,
    2>;
