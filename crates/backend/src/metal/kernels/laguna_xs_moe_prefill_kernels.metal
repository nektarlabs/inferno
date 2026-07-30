#include <metal_stdlib>

using namespace metal;

constant uint LAGUNA_XS_MOE_TOP_K = 8u;
constant uint LAGUNA_XS_MOE_EXPERTS = 256u;
constant uint LAGUNA_XS_MOE_BLOCK_VALUES = 256u;
constant uint LAGUNA_XS_MOE_GROUPS_PER_BLOCK = 8u;
constant uint LAGUNA_XS_MOE_OUTPUT_TILE = 32u;
constant uint LAGUNA_XS_MOE_SMALL_ASSIGNMENT_TILE = 8u;
constant uint LAGUNA_XS_MOE_LARGE_ASSIGNMENT_TILE = 32u;
constant uint LAGUNA_XS_MOE_K_TILE = 32u;
constant uint LAGUNA_XS_MOE_THREADS = 128u;
constant uint LAGUNA_XS_MOE_SMALL_TOKEN_GROUPS = 1u;
constant uint LAGUNA_XS_MOE_LARGE_TOKEN_GROUPS = 4u;
constant uint LAGUNA_XS_MOE_STAGE_VALUES =
    LAGUNA_XS_MOE_OUTPUT_TILE * LAGUNA_XS_MOE_K_TILE;

struct LagunaXsMoeQ4KBlock {
    half d;
    half dmin;
    uchar scales[12];
    uchar quants[128];
};

struct LagunaXsMoeQ6KBlock {
    uchar low_quants[128];
    uchar high_quants[64];
    char scales[16];
    half d;
};

static inline uchar laguna_xs_moe_q4_scale(
    const device LagunaXsMoeQ4KBlock* block,
    uint group
) {
    if (group < 4u) {
        return block->scales[group] & 63u;
    }
    return (block->scales[group + 4u] & 15u)
        | ((block->scales[group - 4u] & 192u) >> 2u);
}

static inline uchar laguna_xs_moe_q4_minimum(
    const device LagunaXsMoeQ4KBlock* block,
    uint group
) {
    if (group < 4u) {
        return block->scales[group + 4u] & 63u;
    }
    return (block->scales[group + 4u] >> 4u)
        | ((block->scales[group] & 192u) >> 2u);
}

struct LagunaXsMoeQ4Operations {
    static inline half4 dequantize4(
        const device LagunaXsMoeQ4KBlock* block,
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
                * float(laguna_xs_moe_q4_scale(block, group))
                * float4(quantized)
            - float(block->dmin)
                * float(laguna_xs_moe_q4_minimum(block, group));
        return half4(value);
    }
};

struct LagunaXsMoeQ6Operations {
    static inline half4 dequantize4(
        const device LagunaXsMoeQ6KBlock* block,
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

kernel void laguna_xs_prefill_build_expert_map_kernel(
    const device uint* expert_ids [[buffer(0)]],
    device atomic_uint* expert_counts [[buffer(1)]],
    device uint* assignment_map [[buffer(2)]],
    device uint4* small_work_tiles [[buffer(3)]],
    device uint4* large_work_tiles [[buffer(4)]],
    device atomic_uint* indirect_arguments [[buffer(5)]],
    constant uint& small_assignment_tile [[buffer(6)]],
    constant uint& token_count [[buffer(7)]],
    constant uint& gate_output_tiles [[buffer(8)]],
    constant uint& down_output_tiles [[buffer(9)]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort threads_per_group [[threads_per_threadgroup]]
) {
    uint thread_id = uint(thread_index);
    if (thread_id < LAGUNA_XS_MOE_EXPERTS) {
        atomic_store_explicit(
            expert_counts + thread_id,
            0u,
            memory_order_relaxed);
    }
    if (thread_id < 12u) {
        atomic_store_explicit(
            indirect_arguments + thread_id,
            0u,
            memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);

    uint assignment_count = token_count * LAGUNA_XS_MOE_TOP_K;
    for (uint assignment = thread_id;
         assignment < assignment_count;
         assignment += uint(threads_per_group)) {
        uint expert = expert_ids[assignment];
        uint position = atomic_fetch_add_explicit(
            expert_counts + expert,
            1u,
            memory_order_relaxed);
        assignment_map[expert * token_count + position] = assignment;
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (thread_id < LAGUNA_XS_MOE_EXPERTS) {
        uint count = atomic_load_explicit(
            expert_counts + thread_id,
            memory_order_relaxed);
        if (count > 0u && count <= small_assignment_tile) {
            uint tile = atomic_fetch_add_explicit(
                indirect_arguments,
                1u,
                memory_order_relaxed);
            small_work_tiles[tile] = uint4(
                thread_id,
                0u,
                count,
                0u);
        } else if (count > 0u) {
            uint tile_count =
                (
                    count
                    + LAGUNA_XS_MOE_LARGE_ASSIGNMENT_TILE
                    - 1u
                ) / LAGUNA_XS_MOE_LARGE_ASSIGNMENT_TILE;
            uint first_tile = atomic_fetch_add_explicit(
                indirect_arguments + 6u,
                tile_count,
                memory_order_relaxed);
            for (uint tile = 0u; tile < tile_count; tile++) {
                large_work_tiles[first_tile + tile] = uint4(
                    thread_id,
                    tile * LAGUNA_XS_MOE_LARGE_ASSIGNMENT_TILE,
                    count,
                    0u);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (thread_id == 0u) {
        uint work_tile_count = atomic_load_explicit(
            indirect_arguments,
            memory_order_relaxed);
        uint large_work_tile_count = atomic_load_explicit(
            indirect_arguments + 6u,
            memory_order_relaxed);
        device uint* arguments =
            reinterpret_cast<device uint*>(indirect_arguments);
        arguments[0] = work_tile_count;
        arguments[1] = gate_output_tiles;
        arguments[2] = 1u;
        arguments[3] = work_tile_count;
        arguments[4] = down_output_tiles;
        arguments[5] = 1u;
        arguments[6] = large_work_tile_count;
        arguments[7] = gate_output_tiles;
        arguments[8] = 1u;
        arguments[9] = large_work_tile_count;
        arguments[10] = down_output_tiles;
        arguments[11] = 1u;
    }
}

template <typename Block, typename Operations, uint AssignmentTile>
static inline void laguna_xs_moe_stage_gate_up(
    const device uchar* gate_weights,
    const device uchar* up_weights,
    const device float* input,
    const device uint* assignment_map,
    threadgroup half* input_tile,
    threadgroup half* gate_tile,
    threadgroup half* up_tile,
    uint expert,
    uint assignment_start,
    uint valid_assignments,
    uint output_start,
    uint input_features,
    uint token_count,
    uint row_bytes,
    uint expert_stride_bytes,
    uint k_tile_index,
    uint thread_index
) {
    constexpr uint vectors_per_assignment = LAGUNA_XS_MOE_K_TILE / 4u;
    for (uint vector_index = thread_index;
         vector_index
            < AssignmentTile * vectors_per_assignment;
         vector_index += LAGUNA_XS_MOE_THREADS) {
        uint local_assignment =
            vector_index / vectors_per_assignment;
        uint vector = vector_index
            - local_assignment * vectors_per_assignment;
        uint column = vector * 4u;
        threadgroup half4* destination =
            reinterpret_cast<threadgroup half4*>(
                input_tile
                    + local_assignment * LAGUNA_XS_MOE_K_TILE
                    + column);
        if (local_assignment < valid_assignments) {
            uint assignment = assignment_map[
                expert * token_count
                    + assignment_start
                    + local_assignment
            ];
            uint token = assignment / LAGUNA_XS_MOE_TOP_K;
            const device float4* source =
                reinterpret_cast<const device float4*>(
                    input
                        + token * input_features
                        + k_tile_index * LAGUNA_XS_MOE_K_TILE
                        + column);
            *destination = half4(*source);
        } else {
            *destination = half4(0.0h);
        }
    }

    uint block_in_row =
        k_tile_index / LAGUNA_XS_MOE_GROUPS_PER_BLOCK;
    uint group =
        k_tile_index - block_in_row * LAGUNA_XS_MOE_GROUPS_PER_BLOCK;
    constexpr uint vectors_per_output = LAGUNA_XS_MOE_K_TILE / 4u;
    for (uint vector_index = thread_index;
         vector_index
            < LAGUNA_XS_MOE_OUTPUT_TILE * vectors_per_output;
         vector_index += LAGUNA_XS_MOE_THREADS) {
        uint output = vector_index / vectors_per_output;
        uint vector = vector_index - output * vectors_per_output;
        uint lane = vector * 4u;
        const device Block* gate_row =
            reinterpret_cast<const device Block*>(
                gate_weights
                    + expert * expert_stride_bytes
                    + (output_start + output) * row_bytes);
        const device Block* up_row =
            reinterpret_cast<const device Block*>(
                up_weights
                    + expert * expert_stride_bytes
                    + (output_start + output) * row_bytes);
        *reinterpret_cast<threadgroup half4*>(
            gate_tile + output * LAGUNA_XS_MOE_K_TILE + lane
        ) = Operations::dequantize4(
            gate_row + block_in_row,
            group,
            lane);
        *reinterpret_cast<threadgroup half4*>(
            up_tile + output * LAGUNA_XS_MOE_K_TILE + lane
        ) = Operations::dequantize4(
            up_row + block_in_row,
            group,
            lane);
    }
}

template <
    typename Block,
    typename Operations,
    uint AssignmentTile,
    uint TokenGroups
>
static inline void laguna_xs_moe_gate_up(
    const device uchar* gate_weights,
    const device uchar* up_weights,
    const device float* input,
    const device uint* assignment_map,
    const device uint4* work_tiles,
    const device float* routing_weights,
    device half* intermediate,
    constant uint& input_features,
    constant uint& intermediate_features,
    constant uint& token_count,
    constant uint& row_bytes,
    constant uint& expert_stride_bytes,
    threadgroup uchar* shared_bytes,
    uint3 group_position,
    uint thread_index,
    uint simdgroup_index
) {
    uint4 work_tile = work_tiles[group_position.x];
    uint expert = work_tile.x;
    uint assignment_start = work_tile.y;
    uint valid_assignments = min(
        AssignmentTile,
        work_tile.z - assignment_start);
    uint output_start =
        group_position.y * LAGUNA_XS_MOE_OUTPUT_TILE;
    uint k_tile_count = input_features / LAGUNA_XS_MOE_K_TILE;

    threadgroup half* staging =
        reinterpret_cast<threadgroup half*>(shared_bytes);
    threadgroup half* input_tile = staging;
    threadgroup half* gate_tile =
        input_tile + LAGUNA_XS_MOE_STAGE_VALUES;
    threadgroup half* up_tile =
        gate_tile + LAGUNA_XS_MOE_STAGE_VALUES;
    simdgroup_float8x8 gate_accumulators[TokenGroups];
    simdgroup_float8x8 up_accumulators[TokenGroups];
    for (uint token_group = 0u;
         token_group < TokenGroups;
         token_group++) {
        gate_accumulators[token_group] =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
        up_accumulators[token_group] =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    laguna_xs_moe_stage_gate_up<Block, Operations, AssignmentTile>(
        gate_weights,
        up_weights,
        input,
        assignment_map,
        input_tile,
        gate_tile,
        up_tile,
        expert,
        assignment_start,
        valid_assignments,
        output_start,
        input_features,
        token_count,
        row_bytes,
        expert_stride_bytes,
        0u,
        thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k_tile_index = 0u;
         k_tile_index < k_tile_count;
         k_tile_index++) {
        for (uint k_offset = 0u;
             k_offset < LAGUNA_XS_MOE_K_TILE;
             k_offset += 8u) {
            simdgroup_half8x8 gate_matrix;
            simdgroup_half8x8 up_matrix;
            simdgroup_load(
                gate_matrix,
                gate_tile
                    + simdgroup_index
                        * 8u
                        * LAGUNA_XS_MOE_K_TILE
                    + k_offset,
                LAGUNA_XS_MOE_K_TILE,
                0,
                true);
            simdgroup_load(
                up_matrix,
                up_tile
                    + simdgroup_index
                        * 8u
                        * LAGUNA_XS_MOE_K_TILE
                    + k_offset,
                LAGUNA_XS_MOE_K_TILE,
                0,
                true);
            for (uint token_group = 0u;
                 token_group < TokenGroups;
                 token_group++) {
                simdgroup_half8x8 input_matrix;
                simdgroup_load(
                    input_matrix,
                    input_tile
                        + token_group
                            * 8u
                            * LAGUNA_XS_MOE_K_TILE
                        + k_offset,
                    LAGUNA_XS_MOE_K_TILE,
                    0,
                    false);
                simdgroup_multiply_accumulate(
                    gate_accumulators[token_group],
                    input_matrix,
                    gate_matrix,
                    gate_accumulators[token_group]);
                simdgroup_multiply_accumulate(
                    up_accumulators[token_group],
                    input_matrix,
                    up_matrix,
                    up_accumulators[token_group]);
            }
        }

        uint next_k_tile = k_tile_index + 1u;
        if (next_k_tile < k_tile_count) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            laguna_xs_moe_stage_gate_up<
                Block,
                Operations,
                AssignmentTile
            >(
                gate_weights,
                up_weights,
                input,
                assignment_map,
                input_tile,
                gate_tile,
                up_tile,
                expert,
                assignment_start,
                valid_assignments,
                output_start,
                input_features,
                token_count,
                row_bytes,
                expert_stride_bytes,
                next_k_tile,
                thread_index);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* gate_results =
        reinterpret_cast<threadgroup float*>(shared_bytes);
    threadgroup float* up_results =
        gate_results
            + AssignmentTile * LAGUNA_XS_MOE_OUTPUT_TILE;
    for (uint token_group = 0u;
         token_group < TokenGroups;
         token_group++) {
        simdgroup_store(
            gate_accumulators[token_group],
            gate_results
                + token_group
                    * 8u
                    * LAGUNA_XS_MOE_OUTPUT_TILE
                + simdgroup_index * 8u,
            LAGUNA_XS_MOE_OUTPUT_TILE,
            0,
            false);
        simdgroup_store(
            up_accumulators[token_group],
            up_results
                + token_group
                    * 8u
                    * LAGUNA_XS_MOE_OUTPUT_TILE
                + simdgroup_index * 8u,
            LAGUNA_XS_MOE_OUTPUT_TILE,
            0,
            false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = thread_index;
         index < valid_assignments * LAGUNA_XS_MOE_OUTPUT_TILE;
         index += LAGUNA_XS_MOE_THREADS) {
        uint local_assignment =
            index / LAGUNA_XS_MOE_OUTPUT_TILE;
        uint output = index
            - local_assignment * LAGUNA_XS_MOE_OUTPUT_TILE;
        uint assignment = assignment_map[
            expert * token_count
                + assignment_start
                + local_assignment
        ];
        float gate = gate_results[index];
        float up = up_results[index];
        float silu = gate / (1.0f + exp(-gate));
        float activated = silu * up * routing_weights[assignment];
        activated = isnan(activated)
            ? 0.0f
            : clamp(activated, -65504.0f, 65504.0f);
        intermediate[
            assignment * intermediate_features
                + output_start
                + output
        ] = half(activated);
    }
}

template <typename Block, typename Operations, uint AssignmentTile>
static inline void laguna_xs_moe_stage_down(
    const device uchar* down_weights,
    const device half* intermediate,
    const device uint* assignment_map,
    threadgroup half* input_tile,
    threadgroup half* weight_tile,
    uint expert,
    uint assignment_start,
    uint valid_assignments,
    uint output_start,
    uint input_features,
    uint token_count,
    uint row_bytes,
    uint expert_stride_bytes,
    uint k_tile_index,
    uint thread_index
) {
    constexpr uint vectors_per_assignment = LAGUNA_XS_MOE_K_TILE / 4u;
    for (uint vector_index = thread_index;
         vector_index
            < AssignmentTile * vectors_per_assignment;
         vector_index += LAGUNA_XS_MOE_THREADS) {
        uint local_assignment =
            vector_index / vectors_per_assignment;
        uint vector = vector_index
            - local_assignment * vectors_per_assignment;
        uint column = vector * 4u;
        threadgroup half4* destination =
            reinterpret_cast<threadgroup half4*>(
                input_tile
                    + local_assignment * LAGUNA_XS_MOE_K_TILE
                    + column);
        if (local_assignment < valid_assignments) {
            uint assignment = assignment_map[
                expert * token_count
                    + assignment_start
                    + local_assignment
            ];
            const device half4* source =
                reinterpret_cast<const device half4*>(
                    intermediate
                        + assignment * input_features
                        + k_tile_index * LAGUNA_XS_MOE_K_TILE
                        + column);
            *destination = *source;
        } else {
            *destination = half4(0.0h);
        }
    }

    uint block_in_row =
        k_tile_index / LAGUNA_XS_MOE_GROUPS_PER_BLOCK;
    uint group =
        k_tile_index - block_in_row * LAGUNA_XS_MOE_GROUPS_PER_BLOCK;
    constexpr uint vectors_per_output = LAGUNA_XS_MOE_K_TILE / 4u;
    for (uint vector_index = thread_index;
         vector_index
            < LAGUNA_XS_MOE_OUTPUT_TILE * vectors_per_output;
         vector_index += LAGUNA_XS_MOE_THREADS) {
        uint output = vector_index / vectors_per_output;
        uint vector = vector_index - output * vectors_per_output;
        uint lane = vector * 4u;
        const device Block* row =
            reinterpret_cast<const device Block*>(
                down_weights
                    + expert * expert_stride_bytes
                    + (output_start + output) * row_bytes);
        *reinterpret_cast<threadgroup half4*>(
            weight_tile + output * LAGUNA_XS_MOE_K_TILE + lane
        ) = Operations::dequantize4(
            row + block_in_row,
            group,
            lane);
    }
}

template <
    typename Block,
    typename Operations,
    uint AssignmentTile,
    uint TokenGroups
>
static inline void laguna_xs_moe_down(
    const device uchar* down_weights,
    const device half* intermediate,
    const device uint* assignment_map,
    const device uint4* work_tiles,
    device float* assignment_output,
    constant uint& input_features,
    constant uint& output_features,
    constant uint& token_count,
    constant uint& row_bytes,
    constant uint& expert_stride_bytes,
    threadgroup uchar* shared_bytes,
    uint3 group_position,
    uint thread_index,
    uint simdgroup_index
) {
    uint4 work_tile = work_tiles[group_position.x];
    uint expert = work_tile.x;
    uint assignment_start = work_tile.y;
    uint valid_assignments = min(
        AssignmentTile,
        work_tile.z - assignment_start);
    uint output_start =
        group_position.y * LAGUNA_XS_MOE_OUTPUT_TILE;
    uint k_tile_count = input_features / LAGUNA_XS_MOE_K_TILE;

    threadgroup half* staging =
        reinterpret_cast<threadgroup half*>(shared_bytes);
    threadgroup half* input_tile = staging;
    threadgroup half* weight_tile =
        input_tile + LAGUNA_XS_MOE_STAGE_VALUES;
    simdgroup_float8x8 accumulators[TokenGroups];
    for (uint token_group = 0u;
         token_group < TokenGroups;
         token_group++) {
        accumulators[token_group] =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    laguna_xs_moe_stage_down<Block, Operations, AssignmentTile>(
        down_weights,
        intermediate,
        assignment_map,
        input_tile,
        weight_tile,
        expert,
        assignment_start,
        valid_assignments,
        output_start,
        input_features,
        token_count,
        row_bytes,
        expert_stride_bytes,
        0u,
        thread_index);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k_tile_index = 0u;
        k_tile_index < k_tile_count;
        k_tile_index++) {
        for (uint k_offset = 0u;
             k_offset < LAGUNA_XS_MOE_K_TILE;
             k_offset += 8u) {
            simdgroup_half8x8 weight_matrix;
            simdgroup_load(
                weight_matrix,
                weight_tile
                    + simdgroup_index
                        * 8u
                        * LAGUNA_XS_MOE_K_TILE
                    + k_offset,
                LAGUNA_XS_MOE_K_TILE,
                0,
                true);
            for (uint token_group = 0u;
                 token_group < TokenGroups;
                 token_group++) {
                simdgroup_half8x8 input_matrix;
                simdgroup_load(
                    input_matrix,
                    input_tile
                        + token_group
                            * 8u
                            * LAGUNA_XS_MOE_K_TILE
                        + k_offset,
                    LAGUNA_XS_MOE_K_TILE,
                    0,
                    false);
                simdgroup_multiply_accumulate(
                    accumulators[token_group],
                    input_matrix,
                    weight_matrix,
                    accumulators[token_group]);
            }
        }

        uint next_k_tile = k_tile_index + 1u;
        if (next_k_tile < k_tile_count) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            laguna_xs_moe_stage_down<
                Block,
                Operations,
                AssignmentTile
            >(
                down_weights,
                intermediate,
                assignment_map,
                input_tile,
                weight_tile,
                expert,
                assignment_start,
                valid_assignments,
                output_start,
                input_features,
                token_count,
                row_bytes,
                expert_stride_bytes,
                next_k_tile,
                thread_index);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* results =
        reinterpret_cast<threadgroup float*>(shared_bytes);
    for (uint token_group = 0u;
         token_group < TokenGroups;
         token_group++) {
        simdgroup_store(
            accumulators[token_group],
            results
                + token_group
                    * 8u
                    * LAGUNA_XS_MOE_OUTPUT_TILE
                + simdgroup_index * 8u,
            LAGUNA_XS_MOE_OUTPUT_TILE,
            0,
            false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = thread_index;
         index < valid_assignments * LAGUNA_XS_MOE_OUTPUT_TILE;
         index += LAGUNA_XS_MOE_THREADS) {
        uint local_assignment =
            index / LAGUNA_XS_MOE_OUTPUT_TILE;
        uint output = index
            - local_assignment * LAGUNA_XS_MOE_OUTPUT_TILE;
        uint assignment = assignment_map[
            expert * token_count
                + assignment_start
                + local_assignment
        ];
        assignment_output[
            assignment * output_features
                + output_start
                + output
        ] = results[index];
    }
}

template <uint AssignmentTile, uint TokenGroups>
kernel void laguna_xs_prefill_q4_gate_up_mma_impl(
    const device uchar* gate_weights [[buffer(0)]],
    const device uchar* up_weights [[buffer(1)]],
    const device float* input [[buffer(2)]],
    const device uint* assignment_map [[buffer(3)]],
    const device uint4* work_tiles [[buffer(4)]],
    const device float* routing_weights [[buffer(5)]],
    device half* intermediate [[buffer(6)]],
    constant uint& input_features [[buffer(7)]],
    constant uint& intermediate_features [[buffer(8)]],
    constant uint& token_count [[buffer(9)]],
    constant uint& row_bytes [[buffer(10)]],
    constant uint& expert_stride_bytes [[buffer(11)]],
    threadgroup uchar* shared_bytes [[threadgroup(0)]],
    uint3 group_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    laguna_xs_moe_gate_up<
        LagunaXsMoeQ4KBlock,
        LagunaXsMoeQ4Operations,
        AssignmentTile,
        TokenGroups
    >(
        gate_weights,
        up_weights,
        input,
        assignment_map,
        work_tiles,
        routing_weights,
        intermediate,
        input_features,
        intermediate_features,
        token_count,
        row_bytes,
        expert_stride_bytes,
        shared_bytes,
        group_position,
        thread_index,
        simdgroup_index);
}

template <uint AssignmentTile, uint TokenGroups>
kernel void laguna_xs_prefill_q4_down_mma_impl(
    const device uchar* down_weights [[buffer(0)]],
    const device half* intermediate [[buffer(1)]],
    const device uint* assignment_map [[buffer(2)]],
    const device uint4* work_tiles [[buffer(3)]],
    device float* assignment_output [[buffer(4)]],
    constant uint& input_features [[buffer(5)]],
    constant uint& output_features [[buffer(6)]],
    constant uint& token_count [[buffer(7)]],
    constant uint& row_bytes [[buffer(8)]],
    constant uint& expert_stride_bytes [[buffer(9)]],
    threadgroup uchar* shared_bytes [[threadgroup(0)]],
    uint3 group_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    laguna_xs_moe_down<
        LagunaXsMoeQ4KBlock,
        LagunaXsMoeQ4Operations,
        AssignmentTile,
        TokenGroups
    >(
        down_weights,
        intermediate,
        assignment_map,
        work_tiles,
        assignment_output,
        input_features,
        output_features,
        token_count,
        row_bytes,
        expert_stride_bytes,
        shared_bytes,
        group_position,
        thread_index,
        simdgroup_index);
}

template <uint AssignmentTile, uint TokenGroups>
kernel void laguna_xs_prefill_q6_down_mma_impl(
    const device uchar* down_weights [[buffer(0)]],
    const device half* intermediate [[buffer(1)]],
    const device uint* assignment_map [[buffer(2)]],
    const device uint4* work_tiles [[buffer(3)]],
    device float* assignment_output [[buffer(4)]],
    constant uint& input_features [[buffer(5)]],
    constant uint& output_features [[buffer(6)]],
    constant uint& token_count [[buffer(7)]],
    constant uint& row_bytes [[buffer(8)]],
    constant uint& expert_stride_bytes [[buffer(9)]],
    threadgroup uchar* shared_bytes [[threadgroup(0)]],
    uint3 group_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    laguna_xs_moe_down<
        LagunaXsMoeQ6KBlock,
        LagunaXsMoeQ6Operations,
        AssignmentTile,
        TokenGroups
    >(
        down_weights,
        intermediate,
        assignment_map,
        work_tiles,
        assignment_output,
        input_features,
        output_features,
        token_count,
        row_bytes,
        expert_stride_bytes,
        shared_bytes,
        group_position,
        thread_index,
        simdgroup_index);
}

typedef decltype(laguna_xs_prefill_q4_gate_up_mma_impl<
    LAGUNA_XS_MOE_SMALL_ASSIGNMENT_TILE,
    LAGUNA_XS_MOE_SMALL_TOKEN_GROUPS>)
    LagunaXsQ4GateUpSmallMma;
typedef decltype(laguna_xs_prefill_q4_gate_up_mma_impl<
    LAGUNA_XS_MOE_LARGE_ASSIGNMENT_TILE,
    LAGUNA_XS_MOE_LARGE_TOKEN_GROUPS>)
    LagunaXsQ4GateUpLargeMma;
typedef decltype(laguna_xs_prefill_q4_down_mma_impl<
    LAGUNA_XS_MOE_SMALL_ASSIGNMENT_TILE,
    LAGUNA_XS_MOE_SMALL_TOKEN_GROUPS>)
    LagunaXsQ4DownSmallMma;
typedef decltype(laguna_xs_prefill_q4_down_mma_impl<
    LAGUNA_XS_MOE_LARGE_ASSIGNMENT_TILE,
    LAGUNA_XS_MOE_LARGE_TOKEN_GROUPS>)
    LagunaXsQ4DownLargeMma;
typedef decltype(laguna_xs_prefill_q6_down_mma_impl<
    LAGUNA_XS_MOE_SMALL_ASSIGNMENT_TILE,
    LAGUNA_XS_MOE_SMALL_TOKEN_GROUPS>)
    LagunaXsQ6DownSmallMma;
typedef decltype(laguna_xs_prefill_q6_down_mma_impl<
    LAGUNA_XS_MOE_LARGE_ASSIGNMENT_TILE,
    LAGUNA_XS_MOE_LARGE_TOKEN_GROUPS>)
    LagunaXsQ6DownLargeMma;

template [[host_name("laguna_xs_prefill_q4_gate_up_small_mma_kernel")]]
kernel LagunaXsQ4GateUpSmallMma
laguna_xs_prefill_q4_gate_up_mma_impl<
    LAGUNA_XS_MOE_SMALL_ASSIGNMENT_TILE,
    LAGUNA_XS_MOE_SMALL_TOKEN_GROUPS>;

template [[host_name("laguna_xs_prefill_q4_gate_up_large_mma_kernel")]]
kernel LagunaXsQ4GateUpLargeMma
laguna_xs_prefill_q4_gate_up_mma_impl<
    LAGUNA_XS_MOE_LARGE_ASSIGNMENT_TILE,
    LAGUNA_XS_MOE_LARGE_TOKEN_GROUPS>;

template [[host_name("laguna_xs_prefill_q4_down_small_mma_kernel")]]
kernel LagunaXsQ4DownSmallMma
laguna_xs_prefill_q4_down_mma_impl<
    LAGUNA_XS_MOE_SMALL_ASSIGNMENT_TILE,
    LAGUNA_XS_MOE_SMALL_TOKEN_GROUPS>;

template [[host_name("laguna_xs_prefill_q4_down_large_mma_kernel")]]
kernel LagunaXsQ4DownLargeMma
laguna_xs_prefill_q4_down_mma_impl<
    LAGUNA_XS_MOE_LARGE_ASSIGNMENT_TILE,
    LAGUNA_XS_MOE_LARGE_TOKEN_GROUPS>;

template [[host_name("laguna_xs_prefill_q6_down_small_mma_kernel")]]
kernel LagunaXsQ6DownSmallMma
laguna_xs_prefill_q6_down_mma_impl<
    LAGUNA_XS_MOE_SMALL_ASSIGNMENT_TILE,
    LAGUNA_XS_MOE_SMALL_TOKEN_GROUPS>;

template [[host_name("laguna_xs_prefill_q6_down_large_mma_kernel")]]
kernel LagunaXsQ6DownLargeMma
laguna_xs_prefill_q6_down_mma_impl<
    LAGUNA_XS_MOE_LARGE_ASSIGNMENT_TILE,
    LAGUNA_XS_MOE_LARGE_TOKEN_GROUPS>;

kernel void laguna_xs_prefill_combine_f32_kernel(
    const device float* assignment_output [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& token_count [[buffer(2)]],
    constant uint& output_features [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_value_count = token_count * output_features;
    if (gid >= output_value_count) {
        return;
    }
    uint token = gid / output_features;
    uint output_feature = gid - token * output_features;
    uint first_assignment = token * LAGUNA_XS_MOE_TOP_K;
    float sum = 0.0f;
    #pragma unroll
    for (uint slot = 0u; slot < LAGUNA_XS_MOE_TOP_K; slot++) {
        sum += assignment_output[
            (first_assignment + slot) * output_features
                + output_feature
        ];
    }
    output[gid] = sum;
}
