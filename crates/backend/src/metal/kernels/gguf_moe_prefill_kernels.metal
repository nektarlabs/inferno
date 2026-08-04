#include <metal_stdlib>

using namespace metal;

/*
 * The expert-major routing map and 64x32 quantized GEMM layout below are
 * adapted from ds4/metal/moe.metal.
 *
 * MIT License
 * Copyright (c) 2026 The ds4.c authors
 * Copyright (c) 2023-2026 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

constant uint LAGUNA_PREFILL_TOP_K = 10u;
constant uint LAGUNA_PREFILL_OUTPUT_TILE = 64u;
constant uint LAGUNA_PREFILL_ASSIGNMENT_TILE = 32u;
constant uint LAGUNA_PREFILL_K_TILE = 32u;
constant uint LAGUNA_PREFILL_THREADS = 128u;
constant uint LAGUNA_PREFILL_QK = 256u;
constant uint LAGUNA_PREFILL_WEIGHT_TILE_HALFS =
    LAGUNA_PREFILL_OUTPUT_TILE * LAGUNA_PREFILL_K_TILE;
constant uint LAGUNA_PREFILL_INPUT_TILE_HALFS =
    LAGUNA_PREFILL_ASSIGNMENT_TILE * LAGUNA_PREFILL_K_TILE;
constant uint LAGUNA_PREFILL_GATE_UP_STAGE_HALFS =
    2u * LAGUNA_PREFILL_WEIGHT_TILE_HALFS
    + LAGUNA_PREFILL_INPUT_TILE_HALFS;
constant uint LAGUNA_PREFILL_DOWN_STAGE_HALFS =
    LAGUNA_PREFILL_WEIGHT_TILE_HALFS
    + LAGUNA_PREFILL_INPUT_TILE_HALFS;

kernel void laguna_prefill_cast_f32_f16_kernel(
    const device float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& value_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint vector_index = gid * 4u;
    if (vector_index < value_count) {
        float4 values = *reinterpret_cast<const device float4*>(
            input + vector_index);
        *reinterpret_cast<device half4*>(
            output + vector_index) = half4(values);
    }
}

struct LagunaPrefillQ2Block {
    uchar scales[16];
    uchar quants[64];
    half d;
    half dmin;
};

struct LagunaPrefillQ3Block {
    uchar high_mask[32];
    uchar quants[64];
    uchar scales[12];
    half d;
};

template <typename Matrix>
static inline void laguna_prefill_dequantize_q2(
    const device LagunaPrefillQ2Block* block,
    short index,
    thread Matrix& output
) {
    half d = block->d;
    half minimum = block->dmin;
    const device uchar* quants = block->quants;
    uchar scale = block->scales[index];

    quants += 32 * (index / 8) + 16 * (index & 1);
    index = (index / 2) % 4;

    half coefficient = index > 1
        ? (index > 2 ? 1.0h / 64.0h : 1.0h / 16.0h)
        : (index > 0 ? 1.0h / 4.0h : 1.0h);
    uchar mask = index > 1
        ? (index > 2 ? 192 : 48)
        : (index > 0 ? 12 : 3);
    half scaled_d = d * half(scale & 0x0f) * coefficient;
    half scaled_minimum = minimum * half(scale >> 4);
    #pragma unroll
    for (short row = 0; row < 4; row++) {
        uchar4 packed = *reinterpret_cast<const device uchar4*>(
            quants + 4 * row);
        output[row] =
            scaled_d * half4(packed & uchar4(mask))
            - half4(scaled_minimum);
    }
}

template <typename Matrix>
static inline void laguna_prefill_dequantize_q3(
    const device LagunaPrefillQ3Block* block,
    short index,
    thread Matrix& output
) {
    half d = block->d;
    const device uchar* quants = block->quants;
    const device uchar* high = block->high_mask;
    const device char* scales =
        reinterpret_cast<const device char*>(block->scales);

    quants += 32 * (index / 8) + 16 * (index & 1);
    high += 16 * (index & 1);
    uchar high_bit = uchar(1u << (index / 2));
    ushort scale_low_mask = (index / 4) > 1
        ? ((index / 4) > 2 ? 192 : 48)
        : ((index / 4) > 0 ? 12 : 3);
    ushort scale_nibble_mask = index / 8 ? 0xf0 : 0x0f;
    ushort scale_low = uchar(scales[index % 8]);
    ushort scale_high = uchar(scales[8 + index % 4]);
    short packed_scale = (index / 4) & 1
        ? short(
            (scale_low & scale_nibble_mask)
            | ((scale_high & scale_low_mask) << 2))
        : short(
            (scale_low & scale_nibble_mask)
            | ((scale_high & scale_low_mask) << 4));
    half scaled_d = index < 8
        ? d * (half(packed_scale) - 32.0h)
        : d * (half(packed_scale) / 16.0h - 32.0h);
    half minimum = 4.0h * scaled_d;

    index = (index / 2) & 3;
    half coefficient = index > 1
        ? (index > 2 ? 1.0h / 64.0h : 1.0h / 16.0h)
        : (index > 0 ? 1.0h / 4.0h : 1.0h);
    uchar quant_mask = index > 1
        ? (index > 2 ? 192 : 48)
        : (index > 0 ? 12 : 3);
    scaled_d *= coefficient;

    #pragma unroll
    for (short row = 0; row < 4; row++) {
        uchar4 packed = *reinterpret_cast<const device uchar4*>(
            quants + 4 * row);
        uchar4 high_values = *reinterpret_cast<const device uchar4*>(
            high + 4 * row);
        bool4 has_high_bit =
            (high_values & uchar4(high_bit)) != uchar4(0);
        half4 minimums = select(
            half4(minimum),
            half4(0.0h),
            has_high_bit);
        output[row] =
            scaled_d * half4(packed & uchar4(quant_mask)) - minimums;
    }
}

// Converts token-major top-10 expert ids into compact expert-major assignment
// lists. One threadgroup scatters each real assignment exactly once, avoiding
// a full token scan for every expert.
kernel void laguna_prefill_build_expert_map_kernel(
    const device uint* expert_ids [[buffer(0)]],
    device atomic_uint* expert_counts [[buffer(1)]],
    device uint* assignment_map [[buffer(2)]],
    device uint4* work_tiles [[buffer(3)]],
    device atomic_uint* work_tile_count [[buffer(4)]],
    constant uint& token_count [[buffer(5)]],
    constant uint& expert_count [[buffer(6)]],
    constant uint& gate_output_tiles [[buffer(7)]],
    constant uint& down_output_tiles [[buffer(8)]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort threads_per_group [[threads_per_threadgroup]]
) {
    uint thread_id = uint(thread_index);
    if (thread_id < expert_count) {
        atomic_store_explicit(
            expert_counts + thread_id,
            0u,
            memory_order_relaxed);
    }
    if (thread_id == 0u) {
        atomic_store_explicit(
            work_tile_count,
            0u,
            memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);

    uint assignment_count = token_count * LAGUNA_PREFILL_TOP_K;
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

    if (thread_id < expert_count) {
        uint count = atomic_load_explicit(
            expert_counts + thread_id,
            memory_order_relaxed);
        uint tile_count =
            (count + LAGUNA_PREFILL_ASSIGNMENT_TILE - 1u)
            / LAGUNA_PREFILL_ASSIGNMENT_TILE;
        uint first_tile = atomic_fetch_add_explicit(
            work_tile_count,
            tile_count,
            memory_order_relaxed);
        for (uint tile = 0u; tile < tile_count; tile++) {
            work_tiles[first_tile + tile] = uint4(
                thread_id,
                tile * LAGUNA_PREFILL_ASSIGNMENT_TILE,
                count,
                0u);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (thread_id == 0u) {
        uint compact_tile_count = atomic_load_explicit(
            work_tile_count,
            memory_order_relaxed);
        device uint* indirect_arguments =
            reinterpret_cast<device uint*>(work_tile_count);
        indirect_arguments[0] = compact_tile_count;
        indirect_arguments[1] = gate_output_tiles;
        indirect_arguments[2] = 1u;
        indirect_arguments[3] = compact_tile_count;
        indirect_arguments[4] = down_output_tiles;
        indirect_arguments[5] = 1u;
    }
}

template <
    typename QuantBlock,
    void (*Dequantize)(
        const device QuantBlock*,
        short,
        thread half4x4&)
>
static inline void laguna_prefill_gate_up_mma(
    const device uchar* gate_weights,
    const device uchar* up_weights,
    const device half* input,
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
    uint3 group,
    ushort thread_index,
    ushort simd_lane,
    ushort simdgroup_index
) {
    uint4 work_tile = work_tiles[group.x];
    uint expert = work_tile.x;
    uint output_start = group.y * LAGUNA_PREFILL_OUTPUT_TILE;
    uint assignment_start = work_tile.y;

    short valid_outputs = short(min(
        LAGUNA_PREFILL_OUTPUT_TILE,
        intermediate_features - output_start));
    short valid_assignments = short(min(
        LAGUNA_PREFILL_ASSIGNMENT_TILE,
        work_tile.z - assignment_start));
    constexpr short k_lane_groups = LAGUNA_PREFILL_K_TILE / 16;
    constexpr short input_lane_groups = LAGUNA_PREFILL_K_TILE / 8;

    threadgroup half* staging =
        reinterpret_cast<threadgroup half*>(shared_bytes);

    short local_output = min(
        short(thread_index / k_lane_groups),
        short(valid_outputs - 1));
    short local_assignment = min(
        short(thread_index / input_lane_groups),
        short(valid_assignments - 1));
    short dequant_index = short(thread_index % k_lane_groups);
    short input_offset = short(
        8 * (thread_index % input_lane_groups));
    uint assignment = assignment_map[
        expert * token_count + assignment_start + local_assignment
    ];
    uint input_token = assignment / LAGUNA_PREFILL_TOP_K;

    const device QuantBlock* gate_row =
        reinterpret_cast<const device QuantBlock*>(
            gate_weights
                + expert * expert_stride_bytes
                + (output_start + uint(local_output)) * row_bytes);
    const device QuantBlock* up_row =
        reinterpret_cast<const device QuantBlock*>(
            up_weights
                + expert * expert_stride_bytes
                + (output_start + uint(local_output)) * row_bytes);
    const device QuantBlock* gate_block = gate_row;
    const device QuantBlock* up_block = up_row;
    simdgroup_half8x8 gate_matrices[4];
    simdgroup_half8x8 up_matrices[4];
    simdgroup_half8x8 input_matrices[2];
    simdgroup_float8x8 gate_accumulators[8];
    simdgroup_float8x8 up_accumulators[8];
    for (short index = 0; index < 8; index++) {
        gate_accumulators[index] =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
        up_accumulators[index] =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    uint staging_index = 0u;
    uint quant_block_count = input_features / LAGUNA_PREFILL_QK;
    for (uint quant_block = 0u;
         quant_block < quant_block_count;
         quant_block++) {
        #pragma unroll
        for (short subblock = 0; subblock < 8; subblock++) {
            short block_subindex = dequant_index + 2 * subblock;
            threadgroup half* stage =
                staging + staging_index * LAGUNA_PREFILL_GATE_UP_STAGE_HALFS;
            threadgroup half* staged_gate = stage;
            threadgroup half* staged_up =
                staged_gate + LAGUNA_PREFILL_WEIGHT_TILE_HALFS;
            threadgroup half* staged_input =
                staged_up + LAGUNA_PREFILL_WEIGHT_TILE_HALFS;

            uint input_column =
                quant_block * LAGUNA_PREFILL_QK
                + uint(subblock) * LAGUNA_PREFILL_K_TILE
                + uint(input_offset);
            const device half* input_values =
                input + input_token * input_features + input_column;
            *reinterpret_cast<threadgroup half2x4*>(
                staged_input
                    + uint(local_assignment) * LAGUNA_PREFILL_K_TILE
                    + uint(input_offset)
            ) = *reinterpret_cast<const device half2x4*>(input_values);

            half4x4 gate_values;
            half4x4 up_values;
            Dequantize(gate_block, block_subindex, gate_values);
            Dequantize(up_block, block_subindex, up_values);
            *reinterpret_cast<threadgroup half4x4*>(
                staged_gate
                    + uint(local_output) * LAGUNA_PREFILL_K_TILE
                    + uint(dequant_index) * 16u
            ) = gate_values;
            *reinterpret_cast<threadgroup half4x4*>(
                staged_up
                    + uint(local_output) * LAGUNA_PREFILL_K_TILE
                    + uint(dequant_index) * 16u
            ) = up_values;

            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup const half* gate_tile =
                staged_gate
                + 32u * LAGUNA_PREFILL_K_TILE * (simdgroup_index % 2);
            threadgroup const half* up_tile =
                staged_up
                + 32u * LAGUNA_PREFILL_K_TILE * (simdgroup_index % 2);
            threadgroup const half* input_tile =
                staged_input
                + 16u * LAGUNA_PREFILL_K_TILE * (simdgroup_index / 2);
            for (short k_block = 0;
                 k_block < short(LAGUNA_PREFILL_K_TILE / 8);
                 k_block++) {
                simdgroup_barrier(mem_flags::mem_none);
                for (short matrix = 0; matrix < 4; matrix++) {
                    simdgroup_load(
                        gate_matrices[matrix],
                        gate_tile
                            + 8u * LAGUNA_PREFILL_K_TILE * uint(matrix),
                        LAGUNA_PREFILL_K_TILE,
                        0,
                        true);
                    simdgroup_load(
                        up_matrices[matrix],
                        up_tile
                            + 8u * LAGUNA_PREFILL_K_TILE * uint(matrix),
                        LAGUNA_PREFILL_K_TILE,
                        0,
                        true);
                }
                for (short matrix = 0; matrix < 2; matrix++) {
                    simdgroup_load(
                        input_matrices[matrix],
                        input_tile
                            + 8u * LAGUNA_PREFILL_K_TILE * uint(matrix),
                        LAGUNA_PREFILL_K_TILE,
                        0,
                        false);
                }
                simdgroup_barrier(mem_flags::mem_none);
                for (short matrix = 0; matrix < 8; matrix++) {
                    simdgroup_multiply_accumulate(
                        gate_accumulators[matrix],
                        input_matrices[matrix / 4],
                        gate_matrices[matrix % 4],
                        gate_accumulators[matrix]);
                    simdgroup_multiply_accumulate(
                        up_accumulators[matrix],
                        input_matrices[matrix / 4],
                        up_matrices[matrix % 4],
                        up_accumulators[matrix]);
                }
                gate_tile += 8u;
                up_tile += 8u;
                input_tile += 8u;
            }

            staging_index ^= 1u;
        }
        gate_block++;
        up_block++;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* gate_results =
        reinterpret_cast<threadgroup float*>(shared_bytes);
    threadgroup float* up_results =
        gate_results
        + LAGUNA_PREFILL_OUTPUT_TILE
            * LAGUNA_PREFILL_ASSIGNMENT_TILE;
    threadgroup float* gate_store =
        gate_results
        + 32 * (simdgroup_index & 1)
        + 16 * (simdgroup_index >> 1)
            * LAGUNA_PREFILL_OUTPUT_TILE;
    threadgroup float* up_store =
        up_results
        + 32 * (simdgroup_index & 1)
        + 16 * (simdgroup_index >> 1)
            * LAGUNA_PREFILL_OUTPUT_TILE;
    for (short matrix = 0; matrix < 8; matrix++) {
        simdgroup_store(
            gate_accumulators[matrix],
            gate_store
                + 8 * (matrix % 4)
                + 8 * LAGUNA_PREFILL_OUTPUT_TILE * (matrix / 4),
            LAGUNA_PREFILL_OUTPUT_TILE,
            0,
            false);
        simdgroup_store(
            up_accumulators[matrix],
            up_store
                + 8 * (matrix % 4)
                + 8 * LAGUNA_PREFILL_OUTPUT_TILE * (matrix / 4),
            LAGUNA_PREFILL_OUTPUT_TILE,
            0,
            false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short local = simdgroup_index;
         local < valid_assignments;
         local += 4) {
        uint routed_assignment = assignment_map[
            expert * token_count + assignment_start + uint(local)
        ];
        device half* destination =
            intermediate
            + routed_assignment * intermediate_features
            + output_start;
        float route_weight = routing_weights[routed_assignment];
        threadgroup float* gate_row_result =
            gate_results
            + local * LAGUNA_PREFILL_OUTPUT_TILE;
        threadgroup float* up_row_result =
            up_results
            + local * LAGUNA_PREFILL_OUTPUT_TILE;
        for (short row = simd_lane;
             row < valid_outputs;
             row += 32) {
            float gate = gate_row_result[row];
            float up = up_row_result[row];
            float silu = gate / (1.0f + exp(-gate));
            destination[row] = half(silu * up * route_weight);
        }
    }
}

template <
    typename QuantBlock,
    void (*Dequantize)(
        const device QuantBlock*,
        short,
        thread half4x4&)
>
static inline void laguna_prefill_down_mma(
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
    uint3 group,
    ushort thread_index,
    ushort simd_lane,
    ushort simdgroup_index
) {
    uint4 work_tile = work_tiles[group.x];
    uint expert = work_tile.x;
    uint output_start = group.y * LAGUNA_PREFILL_OUTPUT_TILE;
    uint assignment_start = work_tile.y;

    short valid_outputs = short(min(
        LAGUNA_PREFILL_OUTPUT_TILE,
        output_features - output_start));
    short valid_assignments = short(min(
        LAGUNA_PREFILL_ASSIGNMENT_TILE,
        work_tile.z - assignment_start));
    constexpr short k_lane_groups = LAGUNA_PREFILL_K_TILE / 16;
    constexpr short input_lane_groups = LAGUNA_PREFILL_K_TILE / 8;

    threadgroup half* staging =
        reinterpret_cast<threadgroup half*>(shared_bytes);

    short local_output = min(
        short(thread_index / k_lane_groups),
        short(valid_outputs - 1));
    short local_assignment = min(
        short(thread_index / input_lane_groups),
        short(valid_assignments - 1));
    short dequant_index = short(thread_index % k_lane_groups);
    short input_offset = short(
        8 * (thread_index % input_lane_groups));
    uint assignment = assignment_map[
        expert * token_count + assignment_start + local_assignment
    ];

    const device QuantBlock* weight_row =
        reinterpret_cast<const device QuantBlock*>(
            down_weights
                + expert * expert_stride_bytes
                + (output_start + uint(local_output)) * row_bytes);
    const device QuantBlock* weight_block = weight_row;
    const device half* input_values =
        intermediate
        + assignment * input_features
        + uint(input_offset);

    simdgroup_half8x8 weight_matrices[4];
    simdgroup_half8x8 input_matrices[2];
    simdgroup_float8x8 accumulators[8];
    for (short index = 0; index < 8; index++) {
        accumulators[index] =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    uint staging_index = 0u;
    uint quant_block_count = input_features / LAGUNA_PREFILL_QK;
    for (uint quant_block = 0u;
         quant_block < quant_block_count;
         quant_block++) {
        #pragma unroll
        for (short subblock = 0; subblock < 8; subblock++) {
            short block_subindex = dequant_index + 2 * subblock;
            threadgroup half* stage =
                staging + staging_index * LAGUNA_PREFILL_DOWN_STAGE_HALFS;
            threadgroup half* staged_weights = stage;
            threadgroup half* staged_input =
                staged_weights + LAGUNA_PREFILL_WEIGHT_TILE_HALFS;

            *reinterpret_cast<threadgroup half2x4*>(
                staged_input
                    + uint(local_assignment) * LAGUNA_PREFILL_K_TILE
                    + uint(input_offset)
            ) = *reinterpret_cast<const device half2x4*>(input_values);

            half4x4 weight_values;
            Dequantize(weight_block, block_subindex, weight_values);
            *reinterpret_cast<threadgroup half4x4*>(
                staged_weights
                    + uint(local_output) * LAGUNA_PREFILL_K_TILE
                    + uint(dequant_index) * 16u
            ) = weight_values;

            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup const half* weight_tile =
                staged_weights
                + 32u * LAGUNA_PREFILL_K_TILE * (simdgroup_index % 2);
            threadgroup const half* input_tile =
                staged_input
                + 16u * LAGUNA_PREFILL_K_TILE * (simdgroup_index / 2);
            for (short k_block = 0;
                 k_block < short(LAGUNA_PREFILL_K_TILE / 8);
                 k_block++) {
                simdgroup_barrier(mem_flags::mem_none);
                for (short matrix = 0; matrix < 4; matrix++) {
                    simdgroup_load(
                        weight_matrices[matrix],
                        weight_tile
                            + 8u * LAGUNA_PREFILL_K_TILE * uint(matrix),
                        LAGUNA_PREFILL_K_TILE,
                        0,
                        true);
                }
                for (short matrix = 0; matrix < 2; matrix++) {
                    simdgroup_load(
                        input_matrices[matrix],
                        input_tile
                            + 8u * LAGUNA_PREFILL_K_TILE * uint(matrix),
                        LAGUNA_PREFILL_K_TILE,
                        0,
                        false);
                }
                simdgroup_barrier(mem_flags::mem_none);
                for (short matrix = 0; matrix < 8; matrix++) {
                    simdgroup_multiply_accumulate(
                        accumulators[matrix],
                        input_matrices[matrix / 4],
                        weight_matrices[matrix % 4],
                        accumulators[matrix]);
                }
                weight_tile += 8u;
                input_tile += 8u;
            }

            input_values += LAGUNA_PREFILL_K_TILE;
            staging_index ^= 1u;
        }
        weight_block++;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float* results =
        reinterpret_cast<threadgroup float*>(shared_bytes);
    threadgroup float* result_store =
        results
        + 32 * (simdgroup_index & 1)
        + 16 * (simdgroup_index >> 1)
            * LAGUNA_PREFILL_OUTPUT_TILE;
    for (short matrix = 0; matrix < 8; matrix++) {
        simdgroup_store(
            accumulators[matrix],
            result_store
                + 8 * (matrix % 4)
                + 8 * LAGUNA_PREFILL_OUTPUT_TILE * (matrix / 4),
            LAGUNA_PREFILL_OUTPUT_TILE,
            0,
            false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short local = simdgroup_index;
         local < valid_assignments;
         local += 4) {
        uint routed_assignment = assignment_map[
            expert * token_count + assignment_start + uint(local)
        ];
        device float* destination =
            assignment_output
            + routed_assignment * output_features
            + output_start;
        threadgroup float* source =
            results + local * LAGUNA_PREFILL_OUTPUT_TILE;
        for (short row = simd_lane;
             row < valid_outputs;
             row += 32) {
            destination[row] = source[row];
        }
    }
}

kernel void laguna_prefill_combine_f32_kernel(
    const device float* assignment_output [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& token_count [[buffer(2)]],
    constant uint& output_features [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = token_count * output_features;
    if (gid >= output_values) {
        return;
    }
    uint token = gid / output_features;
    uint output_index = gid - token * output_features;
    uint assignment = token * LAGUNA_PREFILL_TOP_K;
    float value = 0.0f;
    for (uint slot = 0u; slot < LAGUNA_PREFILL_TOP_K; slot++) {
        value += assignment_output[
            (assignment + slot) * output_features + output_index
        ];
    }
    output[gid] = value;
}

kernel void laguna_prefill_q2_gate_up_mma_kernel(
    const device uchar* gate_weights [[buffer(0)]],
    const device uchar* up_weights [[buffer(1)]],
    const device half* input [[buffer(2)]],
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
    uint3 group [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    laguna_prefill_gate_up_mma<
        LagunaPrefillQ2Block,
        laguna_prefill_dequantize_q2
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
        group,
        thread_index,
        simd_lane,
        simdgroup_index);
}

kernel void laguna_prefill_q3_gate_up_mma_kernel(
    const device uchar* gate_weights [[buffer(0)]],
    const device uchar* up_weights [[buffer(1)]],
    const device half* input [[buffer(2)]],
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
    uint3 group [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    laguna_prefill_gate_up_mma<
        LagunaPrefillQ3Block,
        laguna_prefill_dequantize_q3
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
        group,
        thread_index,
        simd_lane,
        simdgroup_index);
}

kernel void laguna_prefill_q2_down_mma_kernel(
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
    uint3 group [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    laguna_prefill_down_mma<
        LagunaPrefillQ2Block,
        laguna_prefill_dequantize_q2
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
        group,
        thread_index,
        simd_lane,
        simdgroup_index);
}

kernel void laguna_prefill_q3_down_mma_kernel(
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
    uint3 group [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    laguna_prefill_down_mma<
        LagunaPrefillQ3Block,
        laguna_prefill_dequantize_q3
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
        group,
        thread_index,
        simd_lane,
        simdgroup_index);
}
