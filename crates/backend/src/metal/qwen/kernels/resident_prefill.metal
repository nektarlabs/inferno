kernel void qwen_resident_prefill_gate_up_kernel(
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
    constexpr uint M_TILE = 32u;
    constexpr uint N_TILE = 32u;
    constexpr uint SIMD_TILE = 16u;
    uint matrix_group = group;
    uint input_tiles = (row_count + M_TILE - 1u) / M_TILE;
    uint output_tile = matrix_group / input_tiles;
    uint input_tile = matrix_group % input_tiles;
    uint simd_row = simd_index / 2u;
    uint simd_column = simd_index - simd_row * 2u;
    uint first_output = output_tile * N_TILE;
    uint first_input = input_tile * M_TILE;
    uint words_per_row = input_width / QWEN_PACKED_VALUES_PER_WORD;
    uint groups_per_row = input_width / QWEN_PACKED_GROUP_SIZE;
    uint k_per_part = input_width / QWEN_PACKED_K_PARTS;

    alignas(16) threadgroup bfloat activation_tile[M_TILE * QWEN_PACKED_K_TILE];
    threadgroup bfloat gate_weight_tile[QWEN_PACKED_K_TILE * N_TILE];
    threadgroup bfloat up_weight_tile[QWEN_PACKED_K_TILE * N_TILE];
    threadgroup float result_tile[M_TILE * N_TILE];
    constexpr uint ITEMS_PER_THREAD = M_TILE * N_TILE / 128u;
    float gate_sums[ITEMS_PER_THREAD] = {0.0f};
    float up_sums[ITEMS_PER_THREAD] = {0.0f};
    // Keep the reference's eight-part summation order, without global partial buffers.
    for (uint k_part = 0u; k_part < QWEN_PACKED_K_PARTS; ++k_part) {
        uint first_k = k_part * k_per_part;
        uint last_k = first_k + k_per_part;
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

        #pragma unroll
        for (uint item = 0u; item < ITEMS_PER_THREAD; ++item) {
            gate_sums[item] += result_tile[tid + item * 128u];
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

        #pragma unroll
        for (uint item = 0u; item < ITEMS_PER_THREAD; ++item) {
            up_sums[item] += result_tile[tid + item * 128u];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    #pragma unroll
    for (uint item = 0u; item < ITEMS_PER_THREAD; ++item) {
        uint index = tid + item * 128u;
        uint row = first_input + index / N_TILE;
        if (row < row_count) {
            float gate = qwen_packed_round_bf16(gate_sums[item]);
            float up = qwen_packed_round_bf16(up_sums[item]);
            float activated = qwen_packed_round_bf16(gate / (1.0f + exp(-gate)));
            output[row * output_width + first_output + index % N_TILE] =
                qwen_packed_f32_to_bf16(activated * up);
        }
    }
}
