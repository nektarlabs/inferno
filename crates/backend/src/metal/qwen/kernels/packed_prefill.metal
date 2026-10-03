#include <metal_stdlib>

using namespace metal;

constant uint QWEN_PACKED_GROUP_SIZE = 64;
constant uint QWEN_PACKED_VALUES_PER_WORD = 8;
constant uint QWEN_PACKED_TOKEN_ROWS = 8;
constant uint QWEN_PACKED_K_TILE = 32;
constant uint QWEN_PACKED_K_SUB_TILE = 8;
constant uint QWEN_PACKED_K_PARTS = 8;

static inline float qwen_packed_bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

static inline ushort qwen_packed_f32_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    uint exponent = bits & 0x7f800000u;
    if (exponent == 0x7f800000u) {
        return ushort(bits >> 16);
    }
    uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return ushort(rounded >> 16);
}

static inline float qwen_packed_round_bf16(float value) {
    return qwen_packed_bf16_to_f32(qwen_packed_f32_to_bf16(value));
}

kernel void qwen_packed_prefill_repack_kernel(
    const device uint* source_packed [[buffer(0)]],
    const device ushort* source_scales [[buffer(1)]],
    const device ushort* source_biases [[buffer(2)]],
    device uint* packed [[buffer(3)]],
    device ushort* scales [[buffer(4)]],
    device ushort* biases [[buffer(5)]],
    constant uint& input_width [[buffer(6)]],
    constant uint& output_width [[buffer(7)]],
    constant uint& output_tile [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    constexpr uint WORD_TILE = 32u;
    constexpr uint OUTPUT_TILE = 32u;
    // Transpose in shared memory so both source reads and destination writes coalesce.
    threadgroup uint tile[OUTPUT_TILE][WORD_TILE + 1u];
    uint words_per_row = input_width / QWEN_PACKED_VALUES_PER_WORD;
    uint word_tiles = words_per_row / WORD_TILE;
    uint output_group = group / word_tiles;
    uint first_word = (group % word_tiles) * WORD_TILE;
    for (uint item = tid; item < OUTPUT_TILE * WORD_TILE; item += 256u) {
        uint row = item / WORD_TILE;
        uint word = item % WORD_TILE;
        uint source = (output_group * OUTPUT_TILE + row) * words_per_row + first_word + word;
        tile[row][word] = source_packed[source];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint item = tid; item < OUTPUT_TILE * WORD_TILE; item += 256u) {
        uint word = first_word + item / OUTPUT_TILE;
        uint row = item % OUTPUT_TILE;
        packed[(output_group * words_per_row + word) * OUTPUT_TILE + row] =
            tile[row][item / OUTPUT_TILE];

        uint words_per_group = QWEN_PACKED_GROUP_SIZE / QWEN_PACKED_VALUES_PER_WORD;
        if (word % words_per_group == 0u) {
            uint groups_per_row = input_width / QWEN_PACKED_GROUP_SIZE;
            uint column = word / words_per_group;
            uint source = (output_group * OUTPUT_TILE + row) * groups_per_row + column;
            uint destination = (output_group * groups_per_row + column) * OUTPUT_TILE + row;
            scales[destination] = source_scales[source];
            biases[destination] = source_biases[source];
        }
    }
}

kernel void qwen_packed_prefill_linear_bf16_kernel(
    const device uint* packed [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device ushort* biases [[buffer(2)]],
    const device bfloat* input [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& input_width [[buffer(6)]],
    constant uint& output_width [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint k_part [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint OUTPUT_TILE = 16u;
    uint output_tiles = output_width / OUTPUT_TILE;
    uint output_tile = group % output_tiles;
    uint input_tile = group / output_tiles;
    uint first_output = output_tile * OUTPUT_TILE;
    uint first_input = input_tile * QWEN_PACKED_TOKEN_ROWS;
    uint words_per_row = input_width / QWEN_PACKED_VALUES_PER_WORD;
    uint groups_per_row = input_width / QWEN_PACKED_GROUP_SIZE;
    uint k_per_part = input_width / QWEN_PACKED_K_PARTS;
    uint first_k = k_part * k_per_part;
    uint last_k = first_k + k_per_part;

    threadgroup bfloat activation_tile[QWEN_PACKED_K_PARTS]
        [QWEN_PACKED_TOKEN_ROWS * QWEN_PACKED_K_TILE];
    threadgroup bfloat weight_tile[QWEN_PACKED_K_PARTS]
        [QWEN_PACKED_K_TILE * OUTPUT_TILE];
    threadgroup float partials[QWEN_PACKED_K_PARTS]
        [QWEN_PACKED_TOKEN_ROWS * OUTPUT_TILE];
    simdgroup_matrix<bfloat, 8, 8> activation;
    simdgroup_matrix<bfloat, 8, 8> weight_left;
    simdgroup_matrix<bfloat, 8, 8> weight_right;
    simdgroup_matrix<float, 8, 8> output_left =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> output_right =
        simdgroup_matrix<float, 8, 8>(0.0f);

    uint output_offset = lane % OUTPUT_TILE;
    uint k_lane = lane / OUTPUT_TILE;
    for (uint k_start = first_k; k_start < last_k; k_start += QWEN_PACKED_K_TILE) {
        for (uint index = lane;
             index < QWEN_PACKED_TOKEN_ROWS * QWEN_PACKED_K_TILE;
             index += 32u) {
            uint row = index / QWEN_PACKED_K_TILE;
            uint column = index - row * QWEN_PACKED_K_TILE;
            uint input_row = first_input + row;
            activation_tile[k_part][index] = input_row < row_count
                ? input[input_row * input_width + k_start + column]
                : bfloat(0.0f);
        }
        #pragma unroll
        for (uint pack_index = 0u; pack_index < 2u; ++pack_index) {
            uint packed_k = pack_index * 2u + k_lane;
            uint word_column = (k_start / QWEN_PACKED_VALUES_PER_WORD) + packed_k;
            uint word_index = (output_tile * words_per_row + word_column) * OUTPUT_TILE
                + output_offset;
            uint parameter_column = k_start / QWEN_PACKED_GROUP_SIZE;
            uint parameter = (output_tile * groups_per_row + parameter_column)
                * OUTPUT_TILE + output_offset;
            uint word = packed[word_index];
            float scale = qwen_packed_bf16_to_f32(scales[parameter]);
            float bias = qwen_packed_bf16_to_f32(biases[parameter]);
            #pragma unroll
            for (uint nibble = 0u; nibble < QWEN_PACKED_VALUES_PER_WORD; ++nibble) {
                weight_tile[k_part][
                    (packed_k * QWEN_PACKED_VALUES_PER_WORD + nibble) * OUTPUT_TILE
                        + output_offset
                ] = bfloat(
                    float((word >> (4u * nibble)) & 0x0fu) * scale + bias
                );
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint sub_tile = 0u;
             sub_tile < QWEN_PACKED_K_TILE / QWEN_PACKED_K_SUB_TILE;
             ++sub_tile) {
            simdgroup_load(
                activation,
                activation_tile[k_part] + sub_tile * QWEN_PACKED_K_SUB_TILE,
                QWEN_PACKED_K_TILE
            );
            simdgroup_load(
                weight_left,
                weight_tile[k_part] + sub_tile * QWEN_PACKED_K_SUB_TILE * OUTPUT_TILE,
                OUTPUT_TILE
            );
            simdgroup_load(
                weight_right,
                weight_tile[k_part]
                    + sub_tile * QWEN_PACKED_K_SUB_TILE * OUTPUT_TILE + 8u,
                OUTPUT_TILE
            );
            simdgroup_multiply_accumulate(output_left, activation, weight_left, output_left);
            simdgroup_multiply_accumulate(output_right, activation, weight_right, output_right);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(output_left, partials[k_part], OUTPUT_TILE);
    simdgroup_store(output_right, partials[k_part] + 8u, OUTPUT_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < QWEN_PACKED_TOKEN_ROWS * OUTPUT_TILE) {
        float sum = 0.0f;
        #pragma unroll
        for (uint part = 0u; part < QWEN_PACKED_K_PARTS; ++part) {
            sum += partials[part][tid];
        }
        uint tile_row = tid / OUTPUT_TILE;
        uint row = first_input + tile_row;
        uint local_output = tid - tile_row * OUTPUT_TILE;
        if (row < row_count) {
            output[row * output_width + first_output + local_output] =
                qwen_packed_f32_to_bf16(sum);
        }
    }
}

kernel void qwen_local_prefill_linear_bf16_kernel(
    const device uint* packed [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device ushort* biases [[buffer(2)]],
    const device bfloat* input [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& input_width [[buffer(6)]],
    constant uint& output_width [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TOKEN_ROWS = 16u;
    constexpr uint OUTPUT_TILE = 16u;
    constexpr uint ACTIVE_K_PARTS = 4u;
    uint output_tiles = output_width / OUTPUT_TILE;
    uint output_tile = group % output_tiles;
    uint input_tile = group / output_tiles;
    uint first_output = output_tile * OUTPUT_TILE;
    uint first_input = input_tile * TOKEN_ROWS;
    uint words_per_row = input_width / QWEN_PACKED_VALUES_PER_WORD;
    uint groups_per_row = input_width / QWEN_PACKED_GROUP_SIZE;
    uint k_per_part = input_width / QWEN_PACKED_K_PARTS;

    threadgroup bfloat activation_tile[ACTIVE_K_PARTS]
        [TOKEN_ROWS * QWEN_PACKED_K_TILE];
    threadgroup bfloat weight_tile[ACTIVE_K_PARTS]
        [QWEN_PACKED_K_TILE * OUTPUT_TILE];
    threadgroup float partials[QWEN_PACKED_K_PARTS]
        [TOKEN_ROWS * OUTPUT_TILE];

    #pragma unroll
    for (uint wave = 0u; wave < 2u; ++wave) {
        uint k_part = wave * ACTIVE_K_PARTS + simd_index;
        uint first_k = k_part * k_per_part;
        uint last_k = first_k + k_per_part;
        simdgroup_matrix<bfloat, 8, 8> activation_top;
        simdgroup_matrix<bfloat, 8, 8> activation_bottom;
        simdgroup_matrix<bfloat, 8, 8> weight_left;
        simdgroup_matrix<bfloat, 8, 8> weight_right;
        simdgroup_matrix<float, 8, 8> output_top_left =
            simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> output_top_right =
            simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> output_bottom_left =
            simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> output_bottom_right =
            simdgroup_matrix<float, 8, 8>(0.0f);

        uint output_offset = lane % OUTPUT_TILE;
        uint k_lane = lane / OUTPUT_TILE;
        for (uint k_start = first_k;
             k_start < last_k;
             k_start += QWEN_PACKED_K_TILE) {
            for (uint index = lane;
                 index < TOKEN_ROWS * QWEN_PACKED_K_TILE;
                 index += 32u) {
                uint row = index / QWEN_PACKED_K_TILE;
                uint column = index - row * QWEN_PACKED_K_TILE;
                uint input_row = first_input + row;
                activation_tile[simd_index][index] = input_row < row_count
                    ? input[input_row * input_width + k_start + column]
                    : bfloat(0.0f);
            }
            #pragma unroll
            for (uint pack_index = 0u; pack_index < 2u; ++pack_index) {
                uint packed_k = pack_index * 2u + k_lane;
                uint word_column =
                    (k_start / QWEN_PACKED_VALUES_PER_WORD) + packed_k;
                uint word_index =
                    (output_tile * words_per_row + word_column) * OUTPUT_TILE
                        + output_offset;
                uint parameter_column = k_start / QWEN_PACKED_GROUP_SIZE;
                uint parameter =
                    (output_tile * groups_per_row + parameter_column) * OUTPUT_TILE
                        + output_offset;
                uint word = packed[word_index];
                float scale = qwen_packed_bf16_to_f32(scales[parameter]);
                float bias = qwen_packed_bf16_to_f32(biases[parameter]);
                #pragma unroll
                for (uint nibble = 0u;
                     nibble < QWEN_PACKED_VALUES_PER_WORD;
                     ++nibble) {
                    weight_tile[simd_index][
                        (packed_k * QWEN_PACKED_VALUES_PER_WORD + nibble)
                                * OUTPUT_TILE
                            + output_offset
                    ] = bfloat(
                        float((word >> (4u * nibble)) & 0x0fu) * scale + bias
                    );
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll
            for (uint sub_tile = 0u;
                 sub_tile < QWEN_PACKED_K_TILE / QWEN_PACKED_K_SUB_TILE;
                 ++sub_tile) {
                uint k_offset = sub_tile * QWEN_PACKED_K_SUB_TILE;
                simdgroup_load(
                    activation_top,
                    activation_tile[simd_index] + k_offset,
                    QWEN_PACKED_K_TILE
                );
                simdgroup_load(
                    activation_bottom,
                    activation_tile[simd_index]
                        + 8u * QWEN_PACKED_K_TILE + k_offset,
                    QWEN_PACKED_K_TILE
                );
                simdgroup_load(
                    weight_left,
                    weight_tile[simd_index] + k_offset * OUTPUT_TILE,
                    OUTPUT_TILE
                );
                simdgroup_load(
                    weight_right,
                    weight_tile[simd_index] + k_offset * OUTPUT_TILE + 8u,
                    OUTPUT_TILE
                );
                simdgroup_multiply_accumulate(
                    output_top_left,
                    activation_top,
                    weight_left,
                    output_top_left
                );
                simdgroup_multiply_accumulate(
                    output_top_right,
                    activation_top,
                    weight_right,
                    output_top_right
                );
                simdgroup_multiply_accumulate(
                    output_bottom_left,
                    activation_bottom,
                    weight_left,
                    output_bottom_left
                );
                simdgroup_multiply_accumulate(
                    output_bottom_right,
                    activation_bottom,
                    weight_right,
                    output_bottom_right
                );
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_store(output_top_left, partials[k_part], OUTPUT_TILE);
        simdgroup_store(output_top_right, partials[k_part] + 8u, OUTPUT_TILE);
        simdgroup_store(
            output_bottom_left,
            partials[k_part] + 8u * OUTPUT_TILE,
            OUTPUT_TILE
        );
        simdgroup_store(
            output_bottom_right,
            partials[k_part] + 8u * OUTPUT_TILE + 8u,
            OUTPUT_TILE
        );
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < TOKEN_ROWS * OUTPUT_TILE; index += 128u) {
        float sum = 0.0f;
        #pragma unroll
        for (uint part = 0u; part < QWEN_PACKED_K_PARTS; ++part) {
            sum += partials[part][index];
        }
        uint row_offset = index / OUTPUT_TILE;
        uint output_offset = index - row_offset * OUTPUT_TILE;
        uint row = first_input + row_offset;
        if (row < row_count) {
            output[row * output_width + first_output + output_offset] =
                qwen_packed_f32_to_bf16(sum);
        }
    }
}

kernel void qwen_packed_prefill_gate_up_swiglu_bf16_kernel(
    const device uint* gate_packed [[buffer(0)]],
    const device ushort* gate_scales [[buffer(1)]],
    const device ushort* gate_biases [[buffer(2)]],
    const device uint* up_packed [[buffer(3)]],
    const device ushort* up_scales [[buffer(4)]],
    const device ushort* up_biases [[buffer(5)]],
    const device bfloat* input [[buffer(6)]],
    device ushort* output [[buffer(7)]],
    constant uint& row_count [[buffer(8)]],
    constant uint& input_width [[buffer(9)]],
    constant uint& output_width [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint k_part [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint OUTPUT_TILE = 8u;
    uint output_tiles = output_width / OUTPUT_TILE;
    uint output_tile = group % output_tiles;
    uint input_tile = group / output_tiles;
    uint first_output = output_tile * OUTPUT_TILE;
    uint first_input = input_tile * QWEN_PACKED_TOKEN_ROWS;
    uint words_per_row = input_width / QWEN_PACKED_VALUES_PER_WORD;
    uint groups_per_row = input_width / QWEN_PACKED_GROUP_SIZE;
    uint k_per_part = input_width / QWEN_PACKED_K_PARTS;
    uint first_k = k_part * k_per_part;
    uint last_k = first_k + k_per_part;

    threadgroup bfloat activation_tile[QWEN_PACKED_K_PARTS]
        [QWEN_PACKED_TOKEN_ROWS * QWEN_PACKED_K_TILE];
    threadgroup bfloat gate_weight_tile[QWEN_PACKED_K_PARTS]
        [QWEN_PACKED_K_TILE * OUTPUT_TILE];
    threadgroup bfloat up_weight_tile[QWEN_PACKED_K_PARTS]
        [QWEN_PACKED_K_TILE * OUTPUT_TILE];
    threadgroup float gate_partials[QWEN_PACKED_K_PARTS]
        [QWEN_PACKED_TOKEN_ROWS * OUTPUT_TILE];
    threadgroup float up_partials[QWEN_PACKED_K_PARTS]
        [QWEN_PACKED_TOKEN_ROWS * OUTPUT_TILE];
    simdgroup_matrix<bfloat, 8, 8> activation;
    simdgroup_matrix<bfloat, 8, 8> gate_weight;
    simdgroup_matrix<bfloat, 8, 8> up_weight;
    simdgroup_matrix<float, 8, 8> gate_output =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> up_output =
        simdgroup_matrix<float, 8, 8>(0.0f);

    uint output_offset = lane % OUTPUT_TILE;
    uint packed_k = lane / OUTPUT_TILE;
    for (uint k_start = first_k; k_start < last_k; k_start += QWEN_PACKED_K_TILE) {
        for (uint index = lane;
             index < QWEN_PACKED_TOKEN_ROWS * QWEN_PACKED_K_TILE;
             index += 32u) {
            uint row = index / QWEN_PACKED_K_TILE;
            uint column = index - row * QWEN_PACKED_K_TILE;
            uint input_row = first_input + row;
            activation_tile[k_part][index] = input_row < row_count
                ? input[input_row * input_width + k_start + column]
                : bfloat(0.0f);
        }
        uint word_column = (k_start / QWEN_PACKED_VALUES_PER_WORD) + packed_k;
        uint word_index = (output_tile * words_per_row + word_column) * OUTPUT_TILE
            + output_offset;
        uint parameter_column = k_start / QWEN_PACKED_GROUP_SIZE;
        uint parameter = (output_tile * groups_per_row + parameter_column) * OUTPUT_TILE
            + output_offset;
        uint gate_word = gate_packed[word_index];
        uint up_word = up_packed[word_index];
        float gate_scale = qwen_packed_bf16_to_f32(gate_scales[parameter]);
        float gate_bias = qwen_packed_bf16_to_f32(gate_biases[parameter]);
        float up_scale = qwen_packed_bf16_to_f32(up_scales[parameter]);
        float up_bias = qwen_packed_bf16_to_f32(up_biases[parameter]);
        #pragma unroll
        for (uint nibble = 0u; nibble < QWEN_PACKED_VALUES_PER_WORD; ++nibble) {
            uint destination =
                (packed_k * QWEN_PACKED_VALUES_PER_WORD + nibble) * OUTPUT_TILE
                    + output_offset;
            gate_weight_tile[k_part][destination] = bfloat(
                float((gate_word >> (4u * nibble)) & 0x0fu) * gate_scale + gate_bias
            );
            up_weight_tile[k_part][destination] = bfloat(
                float((up_word >> (4u * nibble)) & 0x0fu) * up_scale + up_bias
            );
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint sub_tile = 0u;
             sub_tile < QWEN_PACKED_K_TILE / QWEN_PACKED_K_SUB_TILE;
             ++sub_tile) {
            uint weight_offset = sub_tile * QWEN_PACKED_K_SUB_TILE * OUTPUT_TILE;
            simdgroup_load(
                activation,
                activation_tile[k_part] + sub_tile * QWEN_PACKED_K_SUB_TILE,
                QWEN_PACKED_K_TILE
            );
            simdgroup_load(gate_weight, gate_weight_tile[k_part] + weight_offset, OUTPUT_TILE);
            simdgroup_load(up_weight, up_weight_tile[k_part] + weight_offset, OUTPUT_TILE);
            simdgroup_multiply_accumulate(
                gate_output,
                activation,
                gate_weight,
                gate_output
            );
            simdgroup_multiply_accumulate(up_output, activation, up_weight, up_output);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(gate_output, gate_partials[k_part], OUTPUT_TILE);
    simdgroup_store(up_output, up_partials[k_part], OUTPUT_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < QWEN_PACKED_TOKEN_ROWS * OUTPUT_TILE) {
        float gate = 0.0f;
        float up = 0.0f;
        #pragma unroll
        for (uint part = 0u; part < QWEN_PACKED_K_PARTS; ++part) {
            gate += gate_partials[part][tid];
            up += up_partials[part][tid];
        }
        uint tile_row = tid / OUTPUT_TILE;
        uint row = first_input + tile_row;
        uint local_output = tid - tile_row * OUTPUT_TILE;
        if (row < row_count) {
            gate = qwen_packed_round_bf16(gate);
            up = qwen_packed_round_bf16(up);
            float activated = qwen_packed_round_bf16(gate / (1.0f + exp(-gate)));
            output[row * output_width + first_output + local_output] =
                qwen_packed_f32_to_bf16(qwen_packed_round_bf16(activated * up));
        }
    }
}

kernel void qwen_packed16_prefill_gate_up_swiglu_bf16_kernel(
    const device uint* gate_packed [[buffer(0)]],
    const device ushort* gate_scales [[buffer(1)]],
    const device ushort* gate_biases [[buffer(2)]],
    const device uint* up_packed [[buffer(3)]],
    const device ushort* up_scales [[buffer(4)]],
    const device ushort* up_biases [[buffer(5)]],
    const device bfloat* input [[buffer(6)]],
    device ushort* output [[buffer(7)]],
    constant uint& row_count [[buffer(8)]],
    constant uint& input_width [[buffer(9)]],
    constant uint& output_width [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint k_part [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TOKEN_ROWS = 16u;
    constexpr uint OUTPUT_TILE = 8u;
    uint output_tiles = output_width / OUTPUT_TILE;
    uint output_tile = group % output_tiles;
    uint input_tile = group / output_tiles;
    uint first_output = output_tile * OUTPUT_TILE;
    uint first_input = input_tile * TOKEN_ROWS;
    uint words_per_row = input_width / QWEN_PACKED_VALUES_PER_WORD;
    uint groups_per_row = input_width / QWEN_PACKED_GROUP_SIZE;
    uint k_per_part = input_width / QWEN_PACKED_K_PARTS;
    uint first_k = k_part * k_per_part;
    uint last_k = first_k + k_per_part;

    threadgroup bfloat activation_tile[QWEN_PACKED_K_PARTS]
        [TOKEN_ROWS * QWEN_PACKED_K_TILE];
    threadgroup bfloat gate_weight_tile[QWEN_PACKED_K_PARTS]
        [QWEN_PACKED_K_TILE * OUTPUT_TILE];
    threadgroup bfloat up_weight_tile[QWEN_PACKED_K_PARTS]
        [QWEN_PACKED_K_TILE * OUTPUT_TILE];
    threadgroup float gate_partials[QWEN_PACKED_K_PARTS]
        [TOKEN_ROWS * OUTPUT_TILE];
    threadgroup float up_partials[QWEN_PACKED_K_PARTS]
        [TOKEN_ROWS * OUTPUT_TILE];
    simdgroup_matrix<bfloat, 8, 8> activation_top;
    simdgroup_matrix<bfloat, 8, 8> activation_bottom;
    simdgroup_matrix<bfloat, 8, 8> gate_weight;
    simdgroup_matrix<bfloat, 8, 8> up_weight;
    simdgroup_matrix<float, 8, 8> gate_top =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> gate_bottom =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> up_top =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> up_bottom =
        simdgroup_matrix<float, 8, 8>(0.0f);

    uint output_offset = lane % OUTPUT_TILE;
    uint packed_k = lane / OUTPUT_TILE;
    for (uint k_start = first_k; k_start < last_k; k_start += QWEN_PACKED_K_TILE) {
        for (uint index = lane; index < TOKEN_ROWS * QWEN_PACKED_K_TILE; index += 32u) {
            uint row = index / QWEN_PACKED_K_TILE;
            uint column = index - row * QWEN_PACKED_K_TILE;
            uint input_row = first_input + row;
            activation_tile[k_part][index] = input_row < row_count
                ? input[input_row * input_width + k_start + column]
                : bfloat(0.0f);
        }
        uint word_column = (k_start / QWEN_PACKED_VALUES_PER_WORD) + packed_k;
        uint word_index = (output_tile * words_per_row + word_column) * OUTPUT_TILE
            + output_offset;
        uint parameter_column = k_start / QWEN_PACKED_GROUP_SIZE;
        uint parameter = (output_tile * groups_per_row + parameter_column) * OUTPUT_TILE
            + output_offset;
        uint gate_word = gate_packed[word_index];
        uint up_word = up_packed[word_index];
        float gate_scale = qwen_packed_bf16_to_f32(gate_scales[parameter]);
        float gate_bias = qwen_packed_bf16_to_f32(gate_biases[parameter]);
        float up_scale = qwen_packed_bf16_to_f32(up_scales[parameter]);
        float up_bias = qwen_packed_bf16_to_f32(up_biases[parameter]);
        #pragma unroll
        for (uint nibble = 0u; nibble < QWEN_PACKED_VALUES_PER_WORD; ++nibble) {
            uint destination =
                (packed_k * QWEN_PACKED_VALUES_PER_WORD + nibble) * OUTPUT_TILE
                    + output_offset;
            gate_weight_tile[k_part][destination] = bfloat(
                float((gate_word >> (4u * nibble)) & 0x0fu) * gate_scale + gate_bias
            );
            up_weight_tile[k_part][destination] = bfloat(
                float((up_word >> (4u * nibble)) & 0x0fu) * up_scale + up_bias
            );
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint sub_tile = 0u;
             sub_tile < QWEN_PACKED_K_TILE / QWEN_PACKED_K_SUB_TILE;
             ++sub_tile) {
            uint k_offset = sub_tile * QWEN_PACKED_K_SUB_TILE;
            uint weight_offset = k_offset * OUTPUT_TILE;
            simdgroup_load(
                activation_top,
                activation_tile[k_part] + k_offset,
                QWEN_PACKED_K_TILE
            );
            simdgroup_load(
                activation_bottom,
                activation_tile[k_part] + 8u * QWEN_PACKED_K_TILE + k_offset,
                QWEN_PACKED_K_TILE
            );
            simdgroup_load(gate_weight, gate_weight_tile[k_part] + weight_offset, OUTPUT_TILE);
            simdgroup_load(up_weight, up_weight_tile[k_part] + weight_offset, OUTPUT_TILE);
            simdgroup_multiply_accumulate(gate_top, activation_top, gate_weight, gate_top);
            simdgroup_multiply_accumulate(
                gate_bottom,
                activation_bottom,
                gate_weight,
                gate_bottom
            );
            simdgroup_multiply_accumulate(up_top, activation_top, up_weight, up_top);
            simdgroup_multiply_accumulate(
                up_bottom,
                activation_bottom,
                up_weight,
                up_bottom
            );
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(gate_top, gate_partials[k_part], OUTPUT_TILE);
    simdgroup_store(gate_bottom, gate_partials[k_part] + 8u * OUTPUT_TILE, OUTPUT_TILE);
    simdgroup_store(up_top, up_partials[k_part], OUTPUT_TILE);
    simdgroup_store(up_bottom, up_partials[k_part] + 8u * OUTPUT_TILE, OUTPUT_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < TOKEN_ROWS * OUTPUT_TILE) {
        float gate = 0.0f;
        float up = 0.0f;
        #pragma unroll
        for (uint part = 0u; part < QWEN_PACKED_K_PARTS; ++part) {
            gate += gate_partials[part][tid];
            up += up_partials[part][tid];
        }
        uint tile_row = tid / OUTPUT_TILE;
        uint row = first_input + tile_row;
        uint local_output = tid - tile_row * OUTPUT_TILE;
        if (row < row_count) {
            gate = qwen_packed_round_bf16(gate);
            up = qwen_packed_round_bf16(up);
            float activated = qwen_packed_round_bf16(gate / (1.0f + exp(-gate)));
            output[row * output_width + first_output + local_output] =
                qwen_packed_f32_to_bf16(qwen_packed_round_bf16(activated * up));
        }
    }
}

kernel void qwen_tiled_prefill_linear_bf16_kernel(
    const device uint* packed [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device ushort* biases [[buffer(2)]],
    const device bfloat* input [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& input_width [[buffer(6)]],
    constant uint& output_width [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint row_simd [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint M_TILE = 32u;
    constexpr uint N_TILE = 16u;
    constexpr uint SIMD_M_TILE = 8u;
    constexpr uint K_TILE = 64u;
    uint output_tiles = output_width / N_TILE;
    uint output_tile = group % output_tiles;
    uint input_tile = group / output_tiles;
    uint first_output = output_tile * N_TILE;
    uint first_input = input_tile * M_TILE;
    uint first_simd_input = first_input + row_simd * SIMD_M_TILE;
    uint words_per_row = input_width / QWEN_PACKED_VALUES_PER_WORD;
    uint groups_per_row = input_width / QWEN_PACKED_GROUP_SIZE;
    uint k_per_part = input_width / QWEN_PACKED_K_PARTS;

    threadgroup bfloat activation_tile[M_TILE * K_TILE];
    threadgroup bfloat weight_tile[K_TILE * N_TILE];
    threadgroup float partials[QWEN_PACKED_K_PARTS][M_TILE * N_TILE];
    simdgroup_matrix<bfloat, 8, 8> activation;
    simdgroup_matrix<bfloat, 8, 8> weight_left;
    simdgroup_matrix<bfloat, 8, 8> weight_right;

    #pragma unroll
    for (uint k_part = 0u; k_part < QWEN_PACKED_K_PARTS; ++k_part) {
        simdgroup_matrix<float, 8, 8> output_left =
            simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> output_right =
            simdgroup_matrix<float, 8, 8>(0.0f);
        uint first_k = k_part * k_per_part;
        uint last_k = first_k + k_per_part;
        for (uint k_start = first_k;
             k_start < last_k;
             k_start += K_TILE) {
            for (uint index = tid; index < M_TILE * K_TILE; index += 128u) {
                uint row = index / K_TILE;
                uint column = index - row * K_TILE;
                uint input_row = first_input + row;
                activation_tile[index] = input_row < row_count
                    ? input[input_row * input_width + k_start + column]
                    : bfloat(0.0f);
            }
            if (tid < (K_TILE / QWEN_PACKED_VALUES_PER_WORD) * N_TILE) {
                uint word_in_tile = tid / N_TILE;
                uint output_offset = tid - word_in_tile * N_TILE;
                uint word_column =
                    (k_start / QWEN_PACKED_VALUES_PER_WORD) + word_in_tile;
                uint word_index = (output_tile * words_per_row + word_column) * N_TILE
                    + output_offset;
                uint parameter_column = k_start / QWEN_PACKED_GROUP_SIZE;
                uint parameter = (output_tile * groups_per_row + parameter_column)
                    * N_TILE + output_offset;
                uint word = packed[word_index];
                float scale = qwen_packed_bf16_to_f32(scales[parameter]);
                float bias = qwen_packed_bf16_to_f32(biases[parameter]);
                #pragma unroll
                for (uint nibble = 0u;
                     nibble < QWEN_PACKED_VALUES_PER_WORD;
                     ++nibble) {
                    weight_tile[
                        (word_in_tile * QWEN_PACKED_VALUES_PER_WORD + nibble) * N_TILE
                            + output_offset
                    ] = bfloat(
                        float((word >> (4u * nibble)) & 0x0fu) * scale + bias
                    );
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll
            for (uint sub_tile = 0u;
                 sub_tile < K_TILE / QWEN_PACKED_K_SUB_TILE;
                 ++sub_tile) {
                uint k_offset = sub_tile * QWEN_PACKED_K_SUB_TILE;
                simdgroup_load(
                    activation,
                    activation_tile + row_simd * SIMD_M_TILE * K_TILE
                        + k_offset,
                    K_TILE
                );
                simdgroup_load(
                    weight_left,
                    weight_tile + k_offset * N_TILE,
                    N_TILE
                );
                simdgroup_load(
                    weight_right,
                    weight_tile + k_offset * N_TILE + 8u,
                    N_TILE
                );
                simdgroup_multiply_accumulate(
                    output_left,
                    activation,
                    weight_left,
                    output_left
                );
                simdgroup_multiply_accumulate(
                    output_right,
                    activation,
                    weight_right,
                    output_right
                );
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint partial_offset = row_simd * SIMD_M_TILE * N_TILE;
        simdgroup_store(output_left, partials[k_part] + partial_offset, N_TILE);
        simdgroup_store(output_right, partials[k_part] + partial_offset + 8u, N_TILE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint index = tid; index < M_TILE * N_TILE; index += 128u) {
        float sum = 0.0f;
        #pragma unroll
        for (uint part = 0u; part < QWEN_PACKED_K_PARTS; ++part) {
            sum += partials[part][index];
        }
        uint row_offset = index / N_TILE;
        uint output_offset = index - row_offset * N_TILE;
        uint row = first_input + row_offset;
        if (row < row_count) {
            output[row * output_width + first_output + output_offset] =
                qwen_packed_f32_to_bf16(sum);
        }
    }
}

kernel void qwen_tiled_prefill_gate_up_swiglu_bf16_kernel(
    const device uint* gate_packed [[buffer(0)]],
    const device ushort* gate_scales [[buffer(1)]],
    const device ushort* gate_biases [[buffer(2)]],
    const device uint* up_packed [[buffer(3)]],
    const device ushort* up_scales [[buffer(4)]],
    const device ushort* up_biases [[buffer(5)]],
    const device bfloat* input [[buffer(6)]],
    device ushort* output [[buffer(7)]],
    constant uint& row_count [[buffer(8)]],
    constant uint& input_width [[buffer(9)]],
    constant uint& output_width [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint M_TILE = 16u;
    constexpr uint N_TILE = 16u;
    constexpr uint SIMD_M_TILE = 8u;
    constexpr uint SIMD_N_TILE = 8u;
    uint output_tiles = output_width / N_TILE;
    uint output_tile = group % output_tiles;
    uint input_tile = group / output_tiles;
    uint simd_row = simd_index / 2u;
    uint simd_column = simd_index - simd_row * 2u;
    uint first_output = output_tile * N_TILE;
    uint first_input = input_tile * M_TILE;
    uint words_per_row = input_width / QWEN_PACKED_VALUES_PER_WORD;
    uint groups_per_row = input_width / QWEN_PACKED_GROUP_SIZE;
    uint k_per_part = input_width / QWEN_PACKED_K_PARTS;

    threadgroup bfloat activation_tile[M_TILE * QWEN_PACKED_K_TILE];
    threadgroup bfloat gate_weight_tile[QWEN_PACKED_K_TILE * N_TILE];
    threadgroup bfloat up_weight_tile[QWEN_PACKED_K_TILE * N_TILE];
    threadgroup float gate_partials[QWEN_PACKED_K_PARTS][M_TILE * N_TILE];
    threadgroup float up_partials[QWEN_PACKED_K_PARTS][M_TILE * N_TILE];
    simdgroup_matrix<bfloat, 8, 8> activation;
    simdgroup_matrix<bfloat, 8, 8> gate_weight;
    simdgroup_matrix<bfloat, 8, 8> up_weight;

    for (uint k_part = 0u; k_part < QWEN_PACKED_K_PARTS; ++k_part) {
        simdgroup_matrix<float, 8, 8> gate_output =
            simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_matrix<float, 8, 8> up_output =
            simdgroup_matrix<float, 8, 8>(0.0f);
        uint first_k = k_part * k_per_part;
        uint last_k = first_k + k_per_part;
        for (uint k_start = first_k;
             k_start < last_k;
             k_start += QWEN_PACKED_K_TILE) {
            for (uint index = tid; index < M_TILE * QWEN_PACKED_K_TILE; index += 128u) {
                uint row = index / QWEN_PACKED_K_TILE;
                uint column = index - row * QWEN_PACKED_K_TILE;
                uint input_row = first_input + row;
                activation_tile[index] = input_row < row_count
                    ? input[input_row * input_width + k_start + column]
                    : bfloat(0.0f);
            }
            if (tid < (QWEN_PACKED_K_TILE / QWEN_PACKED_VALUES_PER_WORD) * N_TILE) {
                uint word_in_tile = tid / N_TILE;
                uint output_offset = tid - word_in_tile * N_TILE;
                uint word_column =
                    (k_start / QWEN_PACKED_VALUES_PER_WORD) + word_in_tile;
                uint word_index = (output_tile * words_per_row + word_column) * N_TILE
                    + output_offset;
                uint parameter_column = k_start / QWEN_PACKED_GROUP_SIZE;
                uint parameter = (output_tile * groups_per_row + parameter_column)
                    * N_TILE + output_offset;
                uint gate_word = gate_packed[word_index];
                uint up_word = up_packed[word_index];
                float gate_scale = qwen_packed_bf16_to_f32(gate_scales[parameter]);
                float gate_bias = qwen_packed_bf16_to_f32(gate_biases[parameter]);
                float up_scale = qwen_packed_bf16_to_f32(up_scales[parameter]);
                float up_bias = qwen_packed_bf16_to_f32(up_biases[parameter]);
                #pragma unroll
                for (uint nibble = 0u;
                     nibble < QWEN_PACKED_VALUES_PER_WORD;
                     ++nibble) {
                    uint destination =
                        (word_in_tile * QWEN_PACKED_VALUES_PER_WORD + nibble) * N_TILE
                            + output_offset;
                    gate_weight_tile[destination] = bfloat(
                        float((gate_word >> (4u * nibble)) & 0x0fu) * gate_scale
                            + gate_bias
                    );
                    up_weight_tile[destination] = bfloat(
                        float((up_word >> (4u * nibble)) & 0x0fu) * up_scale
                            + up_bias
                    );
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            #pragma unroll
            for (uint sub_tile = 0u;
                 sub_tile < QWEN_PACKED_K_TILE / QWEN_PACKED_K_SUB_TILE;
                 ++sub_tile) {
                uint k_offset = sub_tile * QWEN_PACKED_K_SUB_TILE;
                simdgroup_load(
                    activation,
                    activation_tile + simd_row * SIMD_M_TILE * QWEN_PACKED_K_TILE
                        + k_offset,
                    QWEN_PACKED_K_TILE
                );
                uint weight_offset = k_offset * N_TILE + simd_column * SIMD_N_TILE;
                simdgroup_load(gate_weight, gate_weight_tile + weight_offset, N_TILE);
                simdgroup_load(up_weight, up_weight_tile + weight_offset, N_TILE);
                simdgroup_multiply_accumulate(
                    gate_output,
                    activation,
                    gate_weight,
                    gate_output
                );
                simdgroup_multiply_accumulate(up_output, activation, up_weight, up_output);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        uint partial_offset = simd_row * SIMD_M_TILE * N_TILE
            + simd_column * SIMD_N_TILE;
        simdgroup_store(gate_output, gate_partials[k_part] + partial_offset, N_TILE);
        simdgroup_store(up_output, up_partials[k_part] + partial_offset, N_TILE);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint index = tid; index < M_TILE * N_TILE; index += 128u) {
        float gate = 0.0f;
        float up = 0.0f;
        #pragma unroll
        for (uint part = 0u; part < QWEN_PACKED_K_PARTS; ++part) {
            gate += gate_partials[part][index];
            up += up_partials[part][index];
        }
        uint row_offset = index / N_TILE;
        uint output_offset = index - row_offset * N_TILE;
        uint row = first_input + row_offset;
        if (row < row_count) {
            gate = qwen_packed_round_bf16(gate);
            up = qwen_packed_round_bf16(up);
            float activated = qwen_packed_round_bf16(gate / (1.0f + exp(-gate)));
            output[row * output_width + first_output + output_offset] =
                qwen_packed_f32_to_bf16(qwen_packed_round_bf16(activated * up));
        }
    }
}

kernel void qwen_splitk_prefill_linear_bf16_kernel(
    const device uint* packed [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device ushort* biases [[buffer(2)]],
    const device bfloat* input [[buffer(3)]],
    device float* partials [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& input_width [[buffer(6)]],
    constant uint& output_width [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint M_TILE = 32u;
    constexpr uint N_TILE = 32u;
    constexpr uint SIMD_TILE = 16u;
    uint k_part = group % QWEN_PACKED_K_PARTS;
    uint matrix_group = group / QWEN_PACKED_K_PARTS;
    uint output_tiles = output_width / N_TILE;
    uint output_tile = matrix_group % output_tiles;
    uint input_tile = matrix_group / output_tiles;
    uint simd_row = simd_index / 2u;
    uint simd_column = simd_index - simd_row * 2u;
    uint first_output = output_tile * N_TILE;
    uint first_input = input_tile * M_TILE;
    uint words_per_row = input_width / QWEN_PACKED_VALUES_PER_WORD;
    uint groups_per_row = input_width / QWEN_PACKED_GROUP_SIZE;
    uint k_per_part = input_width / QWEN_PACKED_K_PARTS;
    uint first_k = k_part * k_per_part;
    uint last_k = first_k + k_per_part;

    alignas(16) threadgroup bfloat activation_tile[M_TILE * QWEN_PACKED_K_TILE];
    threadgroup bfloat weight_tile[QWEN_PACKED_K_TILE * N_TILE];
    threadgroup float result_tile[M_TILE * N_TILE];
    simdgroup_matrix<bfloat, 8, 8> activation_top;
    simdgroup_matrix<bfloat, 8, 8> activation_bottom;
    simdgroup_matrix<bfloat, 8, 8> weight_left;
    simdgroup_matrix<bfloat, 8, 8> weight_right;
    simdgroup_matrix<float, 8, 8> output_top_left =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> output_top_right =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> output_bottom_left =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> output_bottom_right =
        simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k_start = first_k; k_start < last_k; k_start += QWEN_PACKED_K_TILE) {
        for (uint index = tid * 8u; index < M_TILE * QWEN_PACKED_K_TILE; index += 1024u) {
            uint row = index / QWEN_PACKED_K_TILE;
            uint column = index - row * QWEN_PACKED_K_TILE;
            uint input_row = first_input + row;
            uint4 values = input_row < row_count
                ? *reinterpret_cast<const device uint4*>(input + input_row * input_width + k_start + column)
                : uint4(0u);
            *reinterpret_cast<threadgroup uint4*>(activation_tile + index) = values;
        }
        if (tid < (QWEN_PACKED_K_TILE / QWEN_PACKED_VALUES_PER_WORD) * N_TILE) {
            uint word_in_tile = tid / N_TILE;
            uint output_offset = tid - word_in_tile * N_TILE;
            uint word_column = (k_start / QWEN_PACKED_VALUES_PER_WORD) + word_in_tile;
            uint word_index = (output_tile * words_per_row + word_column) * N_TILE
                + output_offset;
            uint parameter_column = k_start / QWEN_PACKED_GROUP_SIZE;
            uint parameter = (output_tile * groups_per_row + parameter_column) * N_TILE
                + output_offset;
            uint word = packed[word_index];
            float scale = qwen_packed_bf16_to_f32(scales[parameter]);
            float bias = qwen_packed_bf16_to_f32(biases[parameter]);
            #pragma unroll
            for (uint nibble = 0u; nibble < QWEN_PACKED_VALUES_PER_WORD; ++nibble) {
                weight_tile[
                    (word_in_tile * QWEN_PACKED_VALUES_PER_WORD + nibble) * N_TILE
                        + output_offset
                ] = bfloat(
                    float((word >> (4u * nibble)) & 0x0fu) * scale + bias
                );
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint sub_tile = 0u;
             sub_tile < QWEN_PACKED_K_TILE / QWEN_PACKED_K_SUB_TILE;
             ++sub_tile) {
            uint k_offset = sub_tile * QWEN_PACKED_K_SUB_TILE;
            uint first_simd_row = simd_row * SIMD_TILE;
            uint first_simd_column = simd_column * SIMD_TILE;
            simdgroup_load(
                activation_top,
                activation_tile + first_simd_row * QWEN_PACKED_K_TILE + k_offset,
                QWEN_PACKED_K_TILE
            );
            simdgroup_load(
                activation_bottom,
                activation_tile
                    + (first_simd_row + 8u) * QWEN_PACKED_K_TILE + k_offset,
                QWEN_PACKED_K_TILE
            );
            simdgroup_load(
                weight_left,
                weight_tile + k_offset * N_TILE + first_simd_column,
                N_TILE
            );
            simdgroup_load(
                weight_right,
                weight_tile + k_offset * N_TILE + first_simd_column + 8u,
                N_TILE
            );
            simdgroup_multiply_accumulate(
                output_top_left,
                activation_top,
                weight_left,
                output_top_left
            );
            simdgroup_multiply_accumulate(
                output_top_right,
                activation_top,
                weight_right,
                output_top_right
            );
            simdgroup_multiply_accumulate(
                output_bottom_left,
                activation_bottom,
                weight_left,
                output_bottom_left
            );
            simdgroup_multiply_accumulate(
                output_bottom_right,
                activation_bottom,
                weight_right,
                output_bottom_right
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint first_simd_row = simd_row * SIMD_TILE;
    uint first_simd_column = simd_column * SIMD_TILE;
    uint result_offset = first_simd_row * N_TILE + first_simd_column;
    simdgroup_store(output_top_left, result_tile + result_offset, N_TILE);
    simdgroup_store(output_top_right, result_tile + result_offset + 8u, N_TILE);
    simdgroup_store(
        output_bottom_left,
        result_tile + result_offset + 8u * N_TILE,
        N_TILE
    );
    simdgroup_store(
        output_bottom_right,
        result_tile + result_offset + 8u * N_TILE + 8u,
        N_TILE
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint partial_base = k_part * row_count * output_width;
    for (uint index = tid; index < M_TILE * N_TILE; index += 128u) {
        uint row_offset = index / N_TILE;
        uint output_offset = index - row_offset * N_TILE;
        uint row = first_input + row_offset;
        if (row < row_count) {
            partials[partial_base + row * output_width + first_output + output_offset] =
                result_tile[index];
        }
    }
}

kernel void qwen_splitk_prefill_gate_up_bf16_kernel(
    const device uint* gate_packed [[buffer(0)]],
    const device ushort* gate_scales [[buffer(1)]],
    const device ushort* gate_biases [[buffer(2)]],
    const device uint* up_packed [[buffer(3)]],
    const device ushort* up_scales [[buffer(4)]],
    const device ushort* up_biases [[buffer(5)]],
    const device bfloat* input [[buffer(6)]],
    device float* gate_partials [[buffer(7)]],
    device float* up_partials [[buffer(8)]],
    constant uint& row_count [[buffer(9)]],
    constant uint& input_width [[buffer(10)]],
    constant uint& output_width [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_index [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint M_TILE = 32u;
    constexpr uint N_TILE = 32u;
    constexpr uint SIMD_TILE = 16u;
    uint k_part = group % QWEN_PACKED_K_PARTS;
    uint matrix_group = group / QWEN_PACKED_K_PARTS;
    uint output_tiles = output_width / N_TILE;
    uint output_tile = matrix_group % output_tiles;
    uint input_tile = matrix_group / output_tiles;
    uint simd_row = simd_index / 2u;
    uint simd_column = simd_index - simd_row * 2u;
    uint first_output = output_tile * N_TILE;
    uint first_input = input_tile * M_TILE;
    uint words_per_row = input_width / QWEN_PACKED_VALUES_PER_WORD;
    uint groups_per_row = input_width / QWEN_PACKED_GROUP_SIZE;
    uint k_per_part = input_width / QWEN_PACKED_K_PARTS;
    uint first_k = k_part * k_per_part;
    uint last_k = first_k + k_per_part;

    alignas(16) threadgroup bfloat activation_tile[M_TILE * QWEN_PACKED_K_TILE];
    threadgroup bfloat gate_weight_tile[QWEN_PACKED_K_TILE * N_TILE];
    threadgroup bfloat up_weight_tile[QWEN_PACKED_K_TILE * N_TILE];
    threadgroup float result_tile[M_TILE * N_TILE];
    simdgroup_matrix<bfloat, 8, 8> activation_top;
    simdgroup_matrix<bfloat, 8, 8> activation_bottom;
    simdgroup_matrix<bfloat, 8, 8> gate_weight_left;
    simdgroup_matrix<bfloat, 8, 8> gate_weight_right;
    simdgroup_matrix<float, 8, 8> gate_output_top_left =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> gate_output_top_right =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> gate_output_bottom_left =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> gate_output_bottom_right =
        simdgroup_matrix<float, 8, 8>(0.0f);

    simdgroup_matrix<bfloat, 8, 8> up_weight_left;
    simdgroup_matrix<bfloat, 8, 8> up_weight_right;
    simdgroup_matrix<float, 8, 8> up_output_top_left =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> up_output_top_right =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> up_output_bottom_left =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> up_output_bottom_right =
        simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k_start = first_k; k_start < last_k; k_start += QWEN_PACKED_K_TILE) {
        for (uint index = tid * 8u; index < M_TILE * QWEN_PACKED_K_TILE; index += 1024u) {
            uint row = index / QWEN_PACKED_K_TILE;
            uint column = index - row * QWEN_PACKED_K_TILE;
            uint input_row = first_input + row;
            uint4 values = input_row < row_count
                ? *reinterpret_cast<const device uint4*>(input + input_row * input_width + k_start + column)
                : uint4(0u);
            *reinterpret_cast<threadgroup uint4*>(activation_tile + index) = values;
        }
        if (tid < (QWEN_PACKED_K_TILE / QWEN_PACKED_VALUES_PER_WORD) * N_TILE) {
            uint word_in_tile = tid / N_TILE;
            uint output_offset = tid - word_in_tile * N_TILE;
            uint word_column = (k_start / QWEN_PACKED_VALUES_PER_WORD) + word_in_tile;
            uint word_index = (output_tile * words_per_row + word_column) * N_TILE
                + output_offset;
            uint parameter_column = k_start / QWEN_PACKED_GROUP_SIZE;
            uint parameter = (output_tile * groups_per_row + parameter_column) * N_TILE
                + output_offset;
            uint word = gate_packed[word_index];
            float scale = qwen_packed_bf16_to_f32(gate_scales[parameter]);
            float bias = qwen_packed_bf16_to_f32(gate_biases[parameter]);
            #pragma unroll
            for (uint nibble = 0u; nibble < QWEN_PACKED_VALUES_PER_WORD; ++nibble) {
                gate_weight_tile[
                    (word_in_tile * QWEN_PACKED_VALUES_PER_WORD + nibble) * N_TILE
                        + output_offset
                ] = bfloat(
                    float((word >> (4u * nibble)) & 0x0fu) * scale + bias
                );
            }
            uint up_word = up_packed[word_index];
            float up_scale = qwen_packed_bf16_to_f32(up_scales[parameter]);
            float up_bias = qwen_packed_bf16_to_f32(up_biases[parameter]);
            #pragma unroll
            for (uint nibble = 0u; nibble < QWEN_PACKED_VALUES_PER_WORD; ++nibble) {
                up_weight_tile[
                    (word_in_tile * QWEN_PACKED_VALUES_PER_WORD + nibble) * N_TILE
                        + output_offset
                ] = bfloat(
                    float((up_word >> (4u * nibble)) & 0x0fu) * up_scale + up_bias
                );
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint sub_tile = 0u;
             sub_tile < QWEN_PACKED_K_TILE / QWEN_PACKED_K_SUB_TILE;
             ++sub_tile) {
            uint k_offset = sub_tile * QWEN_PACKED_K_SUB_TILE;
            uint first_simd_row = simd_row * SIMD_TILE;
            uint first_simd_column = simd_column * SIMD_TILE;
            simdgroup_load(
                activation_top,
                activation_tile + first_simd_row * QWEN_PACKED_K_TILE + k_offset,
                QWEN_PACKED_K_TILE
            );
            simdgroup_load(
                activation_bottom,
                activation_tile
                    + (first_simd_row + 8u) * QWEN_PACKED_K_TILE + k_offset,
                QWEN_PACKED_K_TILE
            );
            simdgroup_load(
                gate_weight_left,
                gate_weight_tile + k_offset * N_TILE + first_simd_column,
                N_TILE
            );
            simdgroup_load(
                gate_weight_right,
                gate_weight_tile + k_offset * N_TILE + first_simd_column + 8u,
                N_TILE
            );
            simdgroup_multiply_accumulate(
                gate_output_top_left,
                activation_top,
                gate_weight_left,
                gate_output_top_left
            );
            simdgroup_multiply_accumulate(
                gate_output_top_right,
                activation_top,
                gate_weight_right,
                gate_output_top_right
            );
            simdgroup_multiply_accumulate(
                gate_output_bottom_left,
                activation_bottom,
                gate_weight_left,
                gate_output_bottom_left
            );
            simdgroup_multiply_accumulate(
                gate_output_bottom_right,
                activation_bottom,
                gate_weight_right,
                gate_output_bottom_right
            );
            simdgroup_load(
                up_weight_left,
                up_weight_tile + k_offset * N_TILE + first_simd_column,
                N_TILE
            );
            simdgroup_load(
                up_weight_right,
                up_weight_tile + k_offset * N_TILE + first_simd_column + 8u,
                N_TILE
            );
            simdgroup_multiply_accumulate(
                up_output_top_left,
                activation_top,
                up_weight_left,
                up_output_top_left
            );
            simdgroup_multiply_accumulate(
                up_output_top_right,
                activation_top,
                up_weight_right,
                up_output_top_right
            );
            simdgroup_multiply_accumulate(
                up_output_bottom_left,
                activation_bottom,
                up_weight_left,
                up_output_bottom_left
            );
            simdgroup_multiply_accumulate(
                up_output_bottom_right,
                activation_bottom,
                up_weight_right,
                up_output_bottom_right
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint first_simd_row = simd_row * SIMD_TILE;
    uint first_simd_column = simd_column * SIMD_TILE;
    uint result_offset = first_simd_row * N_TILE + first_simd_column;
    simdgroup_store(gate_output_top_left, result_tile + result_offset, N_TILE);
    simdgroup_store(gate_output_top_right, result_tile + result_offset + 8u, N_TILE);
    simdgroup_store(
        gate_output_bottom_left,
        result_tile + result_offset + 8u * N_TILE,
        N_TILE
    );
    simdgroup_store(
        gate_output_bottom_right,
        result_tile + result_offset + 8u * N_TILE + 8u,
        N_TILE
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint gate_partial_base = k_part * row_count * output_width;
    for (uint index = tid; index < M_TILE * N_TILE; index += 128u) {
        uint row_offset = index / N_TILE;
        uint output_offset = index - row_offset * N_TILE;
        uint row = first_input + row_offset;
        if (row < row_count) {
            gate_partials[gate_partial_base + row * output_width + first_output + output_offset] =
                result_tile[index];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(up_output_top_left, result_tile + result_offset, N_TILE);
    simdgroup_store(up_output_top_right, result_tile + result_offset + 8u, N_TILE);
    simdgroup_store(
        up_output_bottom_left,
        result_tile + result_offset + 8u * N_TILE,
        N_TILE
    );
    simdgroup_store(
        up_output_bottom_right,
        result_tile + result_offset + 8u * N_TILE + 8u,
        N_TILE
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint up_partial_base = k_part * row_count * output_width;
    for (uint index = tid; index < M_TILE * N_TILE; index += 128u) {
        uint row_offset = index / N_TILE;
        uint output_offset = index - row_offset * N_TILE;
        uint row = first_input + row_offset;
        if (row < row_count) {
            up_partials[up_partial_base + row * output_width + first_output + output_offset] =
                result_tile[index];
        }
    }
}

kernel void qwen_splitk_prefill_reduce_bf16_kernel(
    const device float* partials [[buffer(0)]],
    device ushort* output [[buffer(1)]],
    constant uint& output_len [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= output_len) {
        return;
    }
    float sum = 0.0f;
    #pragma unroll
    for (uint part = 0u; part < QWEN_PACKED_K_PARTS; ++part) {
        sum += partials[part * output_len + gid];
    }
    output[gid] = qwen_packed_f32_to_bf16(sum);
}

kernel void qwen_splitk_prefill_reduce_add_bf16_kernel(
    const device float* partials [[buffer(0)]],
    const device ushort* residual [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& output_len [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= output_len) {
        return;
    }
    float sum = 0.0f;
    #pragma unroll
    for (uint part = 0u; part < QWEN_PACKED_K_PARTS; ++part) {
        sum += partials[part * output_len + gid];
    }
    float projected = qwen_packed_bf16_to_f32(qwen_packed_f32_to_bf16(sum));
    output[gid] = qwen_packed_f32_to_bf16(
        qwen_packed_bf16_to_f32(residual[gid]) + projected
    );
}

kernel void qwen_splitk_prefill_reduce_swiglu_bf16_kernel(
    const device float* gate_partials [[buffer(0)]],
    const device float* up_partials [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& output_vector_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= output_vector_count) {
        return;
    }
    uint output_len = output_vector_count * 4u;
    uint first_element = gid * 4u;
    float4 gate = float4(0.0f);
    float4 up = float4(0.0f);
    #pragma unroll
    for (uint part = 0u; part < QWEN_PACKED_K_PARTS; ++part) {
        uint partial_offset = part * output_len + first_element;
        gate += *reinterpret_cast<const device float4*>(gate_partials + partial_offset);
        up += *reinterpret_cast<const device float4*>(up_partials + partial_offset);
    }
    ushort4 result;
    #pragma unroll
    for (uint component = 0u; component < 4u; ++component) {
        float gate_value = qwen_packed_round_bf16(gate[component]);
        float up_value = qwen_packed_round_bf16(up[component]);
        float activated = qwen_packed_round_bf16(
            gate_value / (1.0f + exp(-gate_value))
        );
        result[component] = qwen_packed_f32_to_bf16(
            qwen_packed_round_bf16(activated * up_value)
        );
    }
    *reinterpret_cast<device ushort4*>(output + first_element) = result;
}
