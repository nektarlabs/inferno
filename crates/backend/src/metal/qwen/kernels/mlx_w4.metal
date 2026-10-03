#include <metal_stdlib>

using namespace metal;

constant uint W4_GROUP_SIZE = 64;
constant uint W4_VALUES_PER_WORD = 8;
constant uint W4_ROW_TILE = 4;
constant uint W4_VERIFY_ROWS = 5;
constant uint W4_SIMDGROUPS = 4;
constant uint W4_TOP_K = 16;
constant uint W4_TOP_K_THREADS = 128;
constant uint W4_TOP_K_VOCAB_TILE = 256;

static inline float bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

static inline ushort f32_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    uint exponent = bits & 0x7f800000u;
    if (exponent == 0x7f800000u) {
        return ushort(bits >> 16);
    }
    uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return ushort(rounded >> 16);
}

static inline float round_bf16(float value) {
    return bf16_to_f32(f32_to_bf16(value));
}

static inline bool w4_candidate_is_better(
    float candidate_value,
    uint candidate_index,
    float current_value,
    uint current_index
) {
    return candidate_value > current_value
        || (candidate_value == current_value && candidate_index < current_index);
}

kernel void qwen_mlx_w4_linear_bf16_kernel(
    const device uint* packed [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device ushort* biases [[buffer(2)]],
    const device ushort* input [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& input_width [[buffer(6)]],
    constant uint& output_width [[buffer(7)]],
    constant uint& first_input_row_offset [[buffer(8)]],
    constant uint& input_row_stride [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint output_row = group % output_width;
    uint input_tile = group / output_width;
    uint first_input_row = input_tile * W4_ROW_TILE;
    if (output_row >= output_width || first_input_row >= row_count) {
        return;
    }

    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    uint weight_start = output_row * words_per_row;
    uint parameter_start = output_row * groups_per_row;
    float sums[W4_ROW_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint word_index = lid; word_index < words_per_row; word_index += 128u) {
        uint word = packed[weight_start + word_index];
        uint parameter = parameter_start + word_index / (W4_GROUP_SIZE / W4_VALUES_PER_WORD);
        float scale = bf16_to_f32(scales[parameter]);
        float bias = bf16_to_f32(biases[parameter]);
        uint input_column = word_index * W4_VALUES_PER_WORD;

        for (uint nibble = 0; nibble < W4_VALUES_PER_WORD; ++nibble) {
            float weight = float((word >> (4u * nibble)) & 0x0fu) * scale + bias;
            for (uint tile_row = 0; tile_row < W4_ROW_TILE; ++tile_row) {
                uint logical_input_row = first_input_row + tile_row;
                if (logical_input_row < row_count) {
                    uint input_row = first_input_row_offset
                        + logical_input_row * input_row_stride;
                    sums[tile_row] +=
                        bf16_to_f32(input[input_row * input_width + input_column + nibble]) * weight;
                }
            }
        }
    }

    threadgroup float partials[W4_SIMDGROUPS * W4_ROW_TILE];
    for (uint tile_row = 0; tile_row < W4_ROW_TILE; ++tile_row) {
        float reduced = simd_sum(sums[tile_row]);
        if (lane == 0u) {
            partials[simd_group * W4_ROW_TILE + tile_row] = reduced;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0u && lane < W4_ROW_TILE) {
        uint input_row = first_input_row + lane;
        if (input_row < row_count) {
            float total = 0.0f;
            for (uint subgroup = 0; subgroup < W4_SIMDGROUPS; ++subgroup) {
                total += partials[subgroup * W4_ROW_TILE + lane];
            }
            output[input_row * output_width + output_row] = f32_to_bf16(total);
        }
    }
}

// Five-row target verification with the exact accumulation and reduction
// order used by the autoregressive row kernel.
kernel void qwen_mlx_w4_verify5_exact_linear_bf16_kernel(
    const device uint* packed [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device ushort* biases [[buffer(2)]],
    const device ushort* input [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    constant uint& input_width [[buffer(5)]],
    constant uint& output_width [[buffer(6)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint VERIFY_ROWS = 5u;
    constexpr uint OUTPUT_TILE = 4u;
    constexpr uint ACCUMULATORS = VERIFY_ROWS * OUTPUT_TILE;
    uint first_output = output_tile * OUTPUT_TILE;
    if (first_output >= output_width) {
        return;
    }

    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    float sums[ACCUMULATORS];
    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        sums[index] = 0.0f;
    }

    for (uint word_index = lid; word_index < words_per_row; word_index += 128u) {
        uint parameter_column = word_index / (W4_GROUP_SIZE / W4_VALUES_PER_WORD);
        uint words[OUTPUT_TILE];
        float weight_scales[OUTPUT_TILE];
        float weight_biases[OUTPUT_TILE];
        #pragma unroll
        for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
            uint output_row = min(first_output + output_offset, output_width - 1u);
            words[output_offset] = packed[output_row * words_per_row + word_index];
            uint parameter = output_row * groups_per_row + parameter_column;
            weight_scales[output_offset] = bf16_to_f32(scales[parameter]);
            weight_biases[output_offset] = bf16_to_f32(biases[parameter]);
        }
        uint input_column = word_index * W4_VALUES_PER_WORD;

        // Pair input loads without changing the scalar accumulation order.
        #pragma unroll
        for (uint first_nibble = 0u; first_nibble < W4_VALUES_PER_WORD; first_nibble += 2u) {
            ushort2 values[VERIFY_ROWS];
            #pragma unroll
            for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                values[row] = *reinterpret_cast<const device ushort2*>(
                    input + row * input_width + input_column + first_nibble
                );
            }
            #pragma unroll
            for (uint component = 0u; component < 2u; ++component) {
                uint nibble = first_nibble + component;
                #pragma unroll
                for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
                    float weight = float(
                        (words[output_offset] >> (4u * nibble)) & 0x0fu
                    ) * weight_scales[output_offset] + weight_biases[output_offset];
                    #pragma unroll
                    for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                        sums[output_offset * VERIFY_ROWS + row] +=
                            bf16_to_f32(values[row][component]) * weight;
                    }
                }
            }
        }
    }

    threadgroup float partials[W4_SIMDGROUPS * ACCUMULATORS];
    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        float reduced = simd_sum(sums[index]);
        if (lane == 0u) {
            partials[simd_group * ACCUMULATORS + index] = reduced;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint result_index = simd_group * 32u + lane;
    if (result_index < ACCUMULATORS) {
        uint output_offset = result_index / VERIFY_ROWS;
        uint row = result_index - output_offset * VERIFY_ROWS;
        uint output_row = first_output + output_offset;
        float total = 0.0f;
        #pragma unroll
        for (uint subgroup = 0u; subgroup < W4_SIMDGROUPS; ++subgroup) {
            total += partials[subgroup * ACCUMULATORS + result_index];
        }
        if (output_row < output_width) {
            output[row * output_width + output_row] = f32_to_bf16(total);
        }
    }
}

// Exact five-row projection followed by the residual add. The projection is
// rounded to BF16 before the add so this is bit-identical to the two-kernel
// path while avoiding its intermediate activation.
kernel void qwen_mlx_w4_verify5_exact_linear_add_bf16_kernel(
    const device uint* packed [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device ushort* biases [[buffer(2)]],
    const device ushort* input [[buffer(3)]],
    const device ushort* residual [[buffer(4)]],
    device ushort* output [[buffer(5)]],
    constant uint& input_width [[buffer(6)]],
    constant uint& output_width [[buffer(7)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint VERIFY_ROWS = 5u;
    constexpr uint OUTPUT_TILE = 4u;
    constexpr uint ACCUMULATORS = VERIFY_ROWS * OUTPUT_TILE;
    uint first_output = output_tile * OUTPUT_TILE;
    if (first_output >= output_width) {
        return;
    }

    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    float sums[ACCUMULATORS];
    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        sums[index] = 0.0f;
    }

    for (uint word_index = lid; word_index < words_per_row; word_index += 128u) {
        uint parameter_column = word_index / (W4_GROUP_SIZE / W4_VALUES_PER_WORD);
        uint words[OUTPUT_TILE];
        float weight_scales[OUTPUT_TILE];
        float weight_biases[OUTPUT_TILE];
        #pragma unroll
        for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
            uint output_row = min(first_output + output_offset, output_width - 1u);
            words[output_offset] = packed[output_row * words_per_row + word_index];
            uint parameter = output_row * groups_per_row + parameter_column;
            weight_scales[output_offset] = bf16_to_f32(scales[parameter]);
            weight_biases[output_offset] = bf16_to_f32(biases[parameter]);
        }
        uint input_column = word_index * W4_VALUES_PER_WORD;

        // Pair input loads without changing the scalar accumulation order.
        #pragma unroll
        for (uint first_nibble = 0u; first_nibble < W4_VALUES_PER_WORD; first_nibble += 2u) {
            ushort2 values[VERIFY_ROWS];
            #pragma unroll
            for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                values[row] = *reinterpret_cast<const device ushort2*>(
                    input + row * input_width + input_column + first_nibble
                );
            }
            #pragma unroll
            for (uint component = 0u; component < 2u; ++component) {
                uint nibble = first_nibble + component;
                #pragma unroll
                for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
                    float weight = float(
                        (words[output_offset] >> (4u * nibble)) & 0x0fu
                    ) * weight_scales[output_offset] + weight_biases[output_offset];
                    #pragma unroll
                    for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                        sums[output_offset * VERIFY_ROWS + row] +=
                            bf16_to_f32(values[row][component]) * weight;
                    }
                }
            }
        }
    }

    threadgroup float partials[W4_SIMDGROUPS * ACCUMULATORS];
    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        float reduced = simd_sum(sums[index]);
        if (lane == 0u) {
            partials[simd_group * ACCUMULATORS + index] = reduced;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint result_index = simd_group * 32u + lane;
    if (result_index < ACCUMULATORS) {
        uint output_offset = result_index / VERIFY_ROWS;
        uint row = result_index - output_offset * VERIFY_ROWS;
        uint output_row = first_output + output_offset;
        float total = 0.0f;
        #pragma unroll
        for (uint subgroup = 0u; subgroup < W4_SIMDGROUPS; ++subgroup) {
            total += partials[subgroup * ACCUMULATORS + result_index];
        }
        if (output_row < output_width) {
            uint output_index = row * output_width + output_row;
            float projected = bf16_to_f32(f32_to_bf16(total));
            output[output_index] = f32_to_bf16(
                bf16_to_f32(residual[output_index]) + projected
            );
        }
    }
}

// Eight-row target verification with the exact accumulation and reduction
// order used by the autoregressive row kernel. Four output neurons share each
// activation load while every decoded W4 weight is reused across all rows.
kernel void qwen_mlx_w4_verify8_exact_linear_bf16_kernel(
    const device uint* packed [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device ushort* biases [[buffer(2)]],
    const device ushort* input [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    constant uint& input_width [[buffer(5)]],
    constant uint& output_width [[buffer(6)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint VERIFY_ROWS = 8u;
    constexpr uint OUTPUT_TILE = 4u;
    constexpr uint ACCUMULATORS = VERIFY_ROWS * OUTPUT_TILE;
    uint first_output = output_tile * OUTPUT_TILE;
    if (first_output >= output_width) {
        return;
    }

    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    float sums[ACCUMULATORS];
    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        sums[index] = 0.0f;
    }

    for (uint word_index = lid; word_index < words_per_row; word_index += 128u) {
        uint parameter_column = word_index / (W4_GROUP_SIZE / W4_VALUES_PER_WORD);
        uint words[OUTPUT_TILE];
        float weight_scales[OUTPUT_TILE];
        float weight_biases[OUTPUT_TILE];
        #pragma unroll
        for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
            uint output_row = min(first_output + output_offset, output_width - 1u);
            words[output_offset] = packed[output_row * words_per_row + word_index];
            uint parameter = output_row * groups_per_row + parameter_column;
            weight_scales[output_offset] = bf16_to_f32(scales[parameter]);
            weight_biases[output_offset] = bf16_to_f32(biases[parameter]);
        }
        uint input_column = word_index * W4_VALUES_PER_WORD;

        // Pair input loads without changing the scalar accumulation order.
        #pragma unroll
        for (uint first_nibble = 0u; first_nibble < W4_VALUES_PER_WORD; first_nibble += 2u) {
            ushort2 values[VERIFY_ROWS];
            #pragma unroll
            for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                values[row] = *reinterpret_cast<const device ushort2*>(
                    input + row * input_width + input_column + first_nibble
                );
            }
            #pragma unroll
            for (uint component = 0u; component < 2u; ++component) {
                uint nibble = first_nibble + component;
                #pragma unroll
                for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
                    float weight = float(
                        (words[output_offset] >> (4u * nibble)) & 0x0fu
                    ) * weight_scales[output_offset] + weight_biases[output_offset];
                    #pragma unroll
                    for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                        sums[output_offset * VERIFY_ROWS + row] +=
                            bf16_to_f32(values[row][component]) * weight;
                    }
                }
            }
        }
    }

    threadgroup float partials[W4_SIMDGROUPS * ACCUMULATORS];
    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        float reduced = simd_sum(sums[index]);
        if (lane == 0u) {
            partials[simd_group * ACCUMULATORS + index] = reduced;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint result_index = simd_group * 32u + lane;
    if (result_index < ACCUMULATORS) {
        uint output_offset = result_index / VERIFY_ROWS;
        uint row = result_index - output_offset * VERIFY_ROWS;
        uint output_row = first_output + output_offset;
        float total = 0.0f;
        #pragma unroll
        for (uint subgroup = 0u; subgroup < W4_SIMDGROUPS; ++subgroup) {
            total += partials[subgroup * ACCUMULATORS + result_index];
        }
        if (output_row < output_width) {
            output[row * output_width + output_row] = f32_to_bf16(total);
        }
    }
}

// Exact eight-row projection followed by the residual add. See the five-row
// variant above for the required BF16 rounding order.
kernel void qwen_mlx_w4_verify8_exact_linear_add_bf16_kernel(
    const device uint* packed [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device ushort* biases [[buffer(2)]],
    const device ushort* input [[buffer(3)]],
    const device ushort* residual [[buffer(4)]],
    device ushort* output [[buffer(5)]],
    constant uint& input_width [[buffer(6)]],
    constant uint& output_width [[buffer(7)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint VERIFY_ROWS = 8u;
    constexpr uint OUTPUT_TILE = 4u;
    constexpr uint ACCUMULATORS = VERIFY_ROWS * OUTPUT_TILE;
    uint first_output = output_tile * OUTPUT_TILE;
    if (first_output >= output_width) {
        return;
    }

    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    float sums[ACCUMULATORS];
    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        sums[index] = 0.0f;
    }

    for (uint word_index = lid; word_index < words_per_row; word_index += 128u) {
        uint parameter_column = word_index / (W4_GROUP_SIZE / W4_VALUES_PER_WORD);
        uint words[OUTPUT_TILE];
        float weight_scales[OUTPUT_TILE];
        float weight_biases[OUTPUT_TILE];
        #pragma unroll
        for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
            uint output_row = min(first_output + output_offset, output_width - 1u);
            words[output_offset] = packed[output_row * words_per_row + word_index];
            uint parameter = output_row * groups_per_row + parameter_column;
            weight_scales[output_offset] = bf16_to_f32(scales[parameter]);
            weight_biases[output_offset] = bf16_to_f32(biases[parameter]);
        }
        uint input_column = word_index * W4_VALUES_PER_WORD;

        // Pair input loads without changing the scalar accumulation order.
        #pragma unroll
        for (uint first_nibble = 0u; first_nibble < W4_VALUES_PER_WORD; first_nibble += 2u) {
            ushort2 values[VERIFY_ROWS];
            #pragma unroll
            for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                values[row] = *reinterpret_cast<const device ushort2*>(
                    input + row * input_width + input_column + first_nibble
                );
            }
            #pragma unroll
            for (uint component = 0u; component < 2u; ++component) {
                uint nibble = first_nibble + component;
                #pragma unroll
                for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
                    float weight = float(
                        (words[output_offset] >> (4u * nibble)) & 0x0fu
                    ) * weight_scales[output_offset] + weight_biases[output_offset];
                    #pragma unroll
                    for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                        sums[output_offset * VERIFY_ROWS + row] +=
                            bf16_to_f32(values[row][component]) * weight;
                    }
                }
            }
        }
    }

    threadgroup float partials[W4_SIMDGROUPS * ACCUMULATORS];
    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        float reduced = simd_sum(sums[index]);
        if (lane == 0u) {
            partials[simd_group * ACCUMULATORS + index] = reduced;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint result_index = simd_group * 32u + lane;
    if (result_index < ACCUMULATORS) {
        uint output_offset = result_index / VERIFY_ROWS;
        uint row = result_index - output_offset * VERIFY_ROWS;
        uint output_row = first_output + output_offset;
        float total = 0.0f;
        #pragma unroll
        for (uint subgroup = 0u; subgroup < W4_SIMDGROUPS; ++subgroup) {
            total += partials[subgroup * ACCUMULATORS + result_index];
        }
        if (output_row < output_width) {
            uint output_index = row * output_width + output_row;
            float projected = bf16_to_f32(f32_to_bf16(total));
            output[output_index] = f32_to_bf16(
                bf16_to_f32(residual[output_index]) + projected
            );
        }
    }
}

// Exact five-row output projection combined with a tile-local top-16. Each
// group scans 256 vocabulary rows and emits only 16 candidates per proposal
// row, avoiding a full logits buffer while preserving BF16 logit rounding.
kernel void qwen_mlx_w4_verify5_top_k_local_kernel(
    const device uint* packed [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device ushort* biases [[buffer(2)]],
    const device ushort* input [[buffer(3)]],
    device uint* local_ids [[buffer(4)]],
    device float* local_values [[buffer(5)]],
    constant uint& input_width [[buffer(6)]],
    constant uint& output_width [[buffer(7)]],
    constant uint& tile_count [[buffer(8)]],
    uint vocab_tile [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint VERIFY_ROWS = 5u;
    constexpr uint OUTPUT_TILE = 2u;
    constexpr uint ACCUMULATORS = VERIFY_ROWS * OUTPUT_TILE;
    uint vocab_start = vocab_tile * W4_TOP_K_VOCAB_TILE;
    if (vocab_start >= output_width) {
        return;
    }
    uint tile_width = min(W4_TOP_K_VOCAB_TILE, output_width - vocab_start);
    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    threadgroup float partials[W4_SIMDGROUPS * ACCUMULATORS];
    threadgroup ushort tile_logits[VERIFY_ROWS * W4_TOP_K_VOCAB_TILE];

    for (uint tile_offset = 0u; tile_offset < tile_width; tile_offset += OUTPUT_TILE) {
        uint first_output = vocab_start + tile_offset;
        float sums[ACCUMULATORS];
        #pragma unroll
        for (uint index = 0u; index < ACCUMULATORS; ++index) {
            sums[index] = 0.0f;
        }

        for (uint word_index = lid; word_index < words_per_row; word_index += 128u) {
            uint parameter_column = word_index / (W4_GROUP_SIZE / W4_VALUES_PER_WORD);
            uint words[OUTPUT_TILE];
            float weight_scales[OUTPUT_TILE];
            float weight_biases[OUTPUT_TILE];
            #pragma unroll
            for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
                uint output_row = min(first_output + output_offset, output_width - 1u);
                words[output_offset] = packed[output_row * words_per_row + word_index];
                uint parameter = output_row * groups_per_row + parameter_column;
                weight_scales[output_offset] = bf16_to_f32(scales[parameter]);
                weight_biases[output_offset] = bf16_to_f32(biases[parameter]);
            }
            uint input_column = word_index * W4_VALUES_PER_WORD;
            for (uint nibble = 0u; nibble < W4_VALUES_PER_WORD; ++nibble) {
                float values[VERIFY_ROWS];
                #pragma unroll
                for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                    values[row] = bf16_to_f32(
                        input[row * input_width + input_column + nibble]
                    );
                }
                #pragma unroll
                for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
                    float weight = float(
                        (words[output_offset] >> (4u * nibble)) & 0x0fu
                    ) * weight_scales[output_offset] + weight_biases[output_offset];
                    #pragma unroll
                    for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                        sums[output_offset * VERIFY_ROWS + row] += values[row] * weight;
                    }
                }
            }
        }

        #pragma unroll
        for (uint index = 0u; index < ACCUMULATORS; ++index) {
            float reduced = simd_sum(sums[index]);
            if (lane == 0u) {
                partials[simd_group * ACCUMULATORS + index] = reduced;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint result_index = simd_group * 32u + lane;
        if (result_index < ACCUMULATORS) {
            uint output_offset = result_index / VERIFY_ROWS;
            uint row = result_index - output_offset * VERIFY_ROWS;
            uint output_row = first_output + output_offset;
            if (output_row < output_width) {
                float total = 0.0f;
                #pragma unroll
                for (uint subgroup = 0u; subgroup < W4_SIMDGROUPS; ++subgroup) {
                    total += partials[subgroup * ACCUMULATORS + result_index];
                }
                tile_logits[row * W4_TOP_K_VOCAB_TILE + tile_offset + output_offset] =
                    f32_to_bf16(total);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float ordered_values[VERIFY_ROWS * W4_TOP_K_VOCAB_TILE];
    threadgroup ushort ordered_offsets[VERIFY_ROWS * W4_TOP_K_VOCAB_TILE];
    threadgroup ushort cursors[W4_TOP_K_THREADS];
    #pragma unroll
    for (uint row = 0u; row < VERIFY_ROWS; ++row) {
        uint first_offset = lid;
        uint second_offset = lid + W4_TOP_K_THREADS;
        float first_value = first_offset < tile_width
            ? bf16_to_f32(tile_logits[row * W4_TOP_K_VOCAB_TILE + first_offset])
            : -INFINITY;
        float second_value = second_offset < tile_width
            ? bf16_to_f32(tile_logits[row * W4_TOP_K_VOCAB_TILE + second_offset])
            : -INFINITY;
        uint first_id = first_offset < tile_width ? vocab_start + first_offset : 0xffffffffu;
        uint second_id = second_offset < tile_width ? vocab_start + second_offset : 0xffffffffu;
        if (w4_candidate_is_better(second_value, second_id, first_value, first_id)) {
            float swapped_value = first_value;
            uint swapped_id = first_id;
            first_value = second_value;
            first_id = second_id;
            second_value = swapped_value;
            second_id = swapped_id;
        }
        uint ordered_start = row * W4_TOP_K_VOCAB_TILE + lid * 2u;
        ordered_values[ordered_start] = first_value;
        ordered_values[ordered_start + 1u] = second_value;
        ordered_offsets[ordered_start] = first_id == 0xffffffffu
            ? ushort(0xffffu)
            : ushort(first_id - vocab_start);
        ordered_offsets[ordered_start + 1u] = second_id == 0xffffffffu
            ? ushort(0xffffu)
            : ushort(second_id - vocab_start);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0u) {
        #pragma unroll
        for (uint row = 0u; row < VERIFY_ROWS; ++row) {
            for (uint source = 0u; source < W4_TOP_K_THREADS; ++source) {
                cursors[source] = 0u;
            }
            uint destination = (row * tile_count + vocab_tile) * W4_TOP_K;
            for (uint rank = 0u; rank < W4_TOP_K; ++rank) {
                float best_value = -INFINITY;
                uint best_id = 0xffffffffu;
                uint best_source = 0xffffffffu;
                for (uint source = 0u; source < W4_TOP_K_THREADS; ++source) {
                    uint cursor = uint(cursors[source]);
                    if (cursor >= 2u) {
                        continue;
                    }
                    uint position = row * W4_TOP_K_VOCAB_TILE + source * 2u + cursor;
                    ushort candidate_offset = ordered_offsets[position];
                    uint candidate_id = candidate_offset == ushort(0xffffu)
                        ? 0xffffffffu
                        : vocab_start + uint(candidate_offset);
                    float candidate_value = ordered_values[position];
                    if (w4_candidate_is_better(
                            candidate_value,
                            candidate_id,
                            best_value,
                            best_id)) {
                        best_value = candidate_value;
                        best_id = candidate_id;
                        best_source = source;
                    }
                }
                local_ids[destination + rank] = best_id;
                local_values[destination + rank] = best_value;
                if (best_source != 0xffffffffu) {
                    cursors[best_source]++;
                }
            }
        }
    }
}

// Eight-row variant. Four vocabulary rows share each decoded W4 word, matching
// the exact M8 verification kernel's accumulation and BF16 rounding order.
kernel void qwen_mlx_w4_verify8_top_k_local_kernel(
    const device uint* packed [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device ushort* biases [[buffer(2)]],
    const device ushort* input [[buffer(3)]],
    device uint* local_ids [[buffer(4)]],
    device float* local_values [[buffer(5)]],
    constant uint& input_width [[buffer(6)]],
    constant uint& output_width [[buffer(7)]],
    constant uint& tile_count [[buffer(8)]],
    uint vocab_tile [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint VERIFY_ROWS = 8u;
    constexpr uint OUTPUT_TILE = 4u;
    constexpr uint ACCUMULATORS = VERIFY_ROWS * OUTPUT_TILE;
    uint vocab_start = vocab_tile * W4_TOP_K_VOCAB_TILE;
    if (vocab_start >= output_width) {
        return;
    }
    uint tile_width = min(W4_TOP_K_VOCAB_TILE, output_width - vocab_start);
    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    threadgroup float partials[W4_SIMDGROUPS * ACCUMULATORS];
    threadgroup ushort tile_logits[VERIFY_ROWS * W4_TOP_K_VOCAB_TILE];

    for (uint tile_offset = 0u; tile_offset < tile_width; tile_offset += OUTPUT_TILE) {
        uint first_output = vocab_start + tile_offset;
        float sums[ACCUMULATORS];
        #pragma unroll
        for (uint index = 0u; index < ACCUMULATORS; ++index) {
            sums[index] = 0.0f;
        }

        for (uint word_index = lid; word_index < words_per_row; word_index += 128u) {
            uint parameter_column = word_index / (W4_GROUP_SIZE / W4_VALUES_PER_WORD);
            uint words[OUTPUT_TILE];
            float weight_scales[OUTPUT_TILE];
            float weight_biases[OUTPUT_TILE];
            #pragma unroll
            for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
                uint output_row = min(first_output + output_offset, output_width - 1u);
                words[output_offset] = packed[output_row * words_per_row + word_index];
                uint parameter = output_row * groups_per_row + parameter_column;
                weight_scales[output_offset] = bf16_to_f32(scales[parameter]);
                weight_biases[output_offset] = bf16_to_f32(biases[parameter]);
            }
            uint input_column = word_index * W4_VALUES_PER_WORD;
            for (uint nibble = 0u; nibble < W4_VALUES_PER_WORD; ++nibble) {
                float values[VERIFY_ROWS];
                #pragma unroll
                for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                    values[row] = bf16_to_f32(
                        input[row * input_width + input_column + nibble]
                    );
                }
                #pragma unroll
                for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
                    float weight = float(
                        (words[output_offset] >> (4u * nibble)) & 0x0fu
                    ) * weight_scales[output_offset] + weight_biases[output_offset];
                    #pragma unroll
                    for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                        sums[output_offset * VERIFY_ROWS + row] += values[row] * weight;
                    }
                }
            }
        }

        #pragma unroll
        for (uint index = 0u; index < ACCUMULATORS; ++index) {
            float reduced = simd_sum(sums[index]);
            if (lane == 0u) {
                partials[simd_group * ACCUMULATORS + index] = reduced;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint result_index = simd_group * 32u + lane;
        if (result_index < ACCUMULATORS) {
            uint output_offset = result_index / VERIFY_ROWS;
            uint row = result_index - output_offset * VERIFY_ROWS;
            uint output_row = first_output + output_offset;
            if (output_row < output_width) {
                float total = 0.0f;
                #pragma unroll
                for (uint subgroup = 0u; subgroup < W4_SIMDGROUPS; ++subgroup) {
                    total += partials[subgroup * ACCUMULATORS + result_index];
                }
                tile_logits[row * W4_TOP_K_VOCAB_TILE + tile_offset + output_offset] =
                    f32_to_bf16(total);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float ordered_values[VERIFY_ROWS * W4_TOP_K_VOCAB_TILE];
    threadgroup ushort ordered_offsets[VERIFY_ROWS * W4_TOP_K_VOCAB_TILE];
    threadgroup ushort cursors[W4_TOP_K_THREADS];
    #pragma unroll
    for (uint row = 0u; row < VERIFY_ROWS; ++row) {
        uint first_offset = lid;
        uint second_offset = lid + W4_TOP_K_THREADS;
        float first_value = first_offset < tile_width
            ? bf16_to_f32(tile_logits[row * W4_TOP_K_VOCAB_TILE + first_offset])
            : -INFINITY;
        float second_value = second_offset < tile_width
            ? bf16_to_f32(tile_logits[row * W4_TOP_K_VOCAB_TILE + second_offset])
            : -INFINITY;
        uint first_id = first_offset < tile_width ? vocab_start + first_offset : 0xffffffffu;
        uint second_id = second_offset < tile_width ? vocab_start + second_offset : 0xffffffffu;
        if (w4_candidate_is_better(second_value, second_id, first_value, first_id)) {
            float swapped_value = first_value;
            uint swapped_id = first_id;
            first_value = second_value;
            first_id = second_id;
            second_value = swapped_value;
            second_id = swapped_id;
        }
        uint ordered_start = row * W4_TOP_K_VOCAB_TILE + lid * 2u;
        ordered_values[ordered_start] = first_value;
        ordered_values[ordered_start + 1u] = second_value;
        ordered_offsets[ordered_start] = first_id == 0xffffffffu
            ? ushort(0xffffu)
            : ushort(first_id - vocab_start);
        ordered_offsets[ordered_start + 1u] = second_id == 0xffffffffu
            ? ushort(0xffffu)
            : ushort(second_id - vocab_start);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0u) {
        #pragma unroll
        for (uint row = 0u; row < VERIFY_ROWS; ++row) {
            for (uint source = 0u; source < W4_TOP_K_THREADS; ++source) {
                cursors[source] = 0u;
            }
            uint destination = (row * tile_count + vocab_tile) * W4_TOP_K;
            for (uint rank = 0u; rank < W4_TOP_K; ++rank) {
                float best_value = -INFINITY;
                uint best_id = 0xffffffffu;
                uint best_source = 0xffffffffu;
                for (uint source = 0u; source < W4_TOP_K_THREADS; ++source) {
                    uint cursor = uint(cursors[source]);
                    if (cursor >= 2u) {
                        continue;
                    }
                    uint position = row * W4_TOP_K_VOCAB_TILE + source * 2u + cursor;
                    ushort candidate_offset = ordered_offsets[position];
                    uint candidate_id = candidate_offset == ushort(0xffffu)
                        ? 0xffffffffu
                        : vocab_start + uint(candidate_offset);
                    float candidate_value = ordered_values[position];
                    if (w4_candidate_is_better(
                            candidate_value,
                            candidate_id,
                            best_value,
                            best_id)) {
                        best_value = candidate_value;
                        best_id = candidate_id;
                        best_source = source;
                    }
                }
                local_ids[destination + rank] = best_id;
                local_values[destination + rank] = best_value;
                if (best_source != 0xffffffffu) {
                    cursors[best_source]++;
                }
            }
        }
    }
}

// DFlash produces four or seven draft rows. Each group handles one four-row
// input tile and 256 vocabulary entries, matching the generic W4 projection's
// accumulation order while retaining only a tile-local top-16.
kernel void qwen_mlx_w4_dflash_top_k_local_kernel(
    const device uint* packed [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device ushort* biases [[buffer(2)]],
    const device ushort* input [[buffer(3)]],
    device uint* local_ids [[buffer(4)]],
    device float* local_values [[buffer(5)]],
    constant uint& row_count [[buffer(6)]],
    constant uint& input_width [[buffer(7)]],
    constant uint& output_width [[buffer(8)]],
    constant uint& tile_count [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint ROW_TILE = 4u;
    constexpr uint OUTPUT_TILE = 8u;
    constexpr uint ACCUMULATORS = ROW_TILE * OUTPUT_TILE;
    uint vocab_tile = group % tile_count;
    uint input_tile = group / tile_count;
    uint first_input_row = input_tile * ROW_TILE;
    uint vocab_start = vocab_tile * W4_TOP_K_VOCAB_TILE;
    if (first_input_row >= row_count || vocab_start >= output_width) {
        return;
    }
    uint tile_width = min(W4_TOP_K_VOCAB_TILE, output_width - vocab_start);
    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    threadgroup float partials[W4_SIMDGROUPS * ACCUMULATORS];
    threadgroup ushort tile_logits[ROW_TILE * W4_TOP_K_VOCAB_TILE];

    for (uint tile_offset = 0u; tile_offset < tile_width; tile_offset += OUTPUT_TILE) {
        uint first_output = vocab_start + tile_offset;
        float sums[ACCUMULATORS];
        #pragma unroll
        for (uint index = 0u; index < ACCUMULATORS; ++index) {
            sums[index] = 0.0f;
        }

        for (uint word_index = lid; word_index < words_per_row; word_index += 128u) {
            uint parameter_column = word_index / (W4_GROUP_SIZE / W4_VALUES_PER_WORD);
            uint words[OUTPUT_TILE];
            float weight_scales[OUTPUT_TILE];
            float weight_biases[OUTPUT_TILE];
            #pragma unroll
            for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
                uint output_row = min(first_output + output_offset, output_width - 1u);
                words[output_offset] = packed[output_row * words_per_row + word_index];
                uint parameter = output_row * groups_per_row + parameter_column;
                weight_scales[output_offset] = bf16_to_f32(scales[parameter]);
                weight_biases[output_offset] = bf16_to_f32(biases[parameter]);
            }
            uint input_column = word_index * W4_VALUES_PER_WORD;
            for (uint nibble = 0u; nibble < W4_VALUES_PER_WORD; ++nibble) {
                float values[ROW_TILE];
                #pragma unroll
                for (uint tile_row = 0u; tile_row < ROW_TILE; ++tile_row) {
                    uint input_row = first_input_row + tile_row;
                    values[tile_row] = input_row < row_count
                        ? bf16_to_f32(input[input_row * input_width + input_column + nibble])
                        : 0.0f;
                }
                #pragma unroll
                for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
                    float weight = float(
                        (words[output_offset] >> (4u * nibble)) & 0x0fu
                    ) * weight_scales[output_offset] + weight_biases[output_offset];
                    #pragma unroll
                    for (uint tile_row = 0u; tile_row < ROW_TILE; ++tile_row) {
                        sums[output_offset * ROW_TILE + tile_row] +=
                            values[tile_row] * weight;
                    }
                }
            }
        }

        #pragma unroll
        for (uint index = 0u; index < ACCUMULATORS; ++index) {
            float reduced = simd_sum(sums[index]);
            if (lane == 0u) {
                partials[simd_group * ACCUMULATORS + index] = reduced;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint result_index = simd_group * 32u + lane;
        if (result_index < ACCUMULATORS) {
            uint output_offset = result_index / ROW_TILE;
            uint tile_row = result_index - output_offset * ROW_TILE;
            uint input_row = first_input_row + tile_row;
            uint output_row = first_output + output_offset;
            if (input_row < row_count && output_row < output_width) {
                float total = 0.0f;
                #pragma unroll
                for (uint subgroup = 0u; subgroup < W4_SIMDGROUPS; ++subgroup) {
                    total += partials[subgroup * ACCUMULATORS + result_index];
                }
                tile_logits[tile_row * W4_TOP_K_VOCAB_TILE + tile_offset + output_offset] =
                    f32_to_bf16(total);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float ordered_values[ROW_TILE * W4_TOP_K_VOCAB_TILE];
    threadgroup ushort ordered_offsets[ROW_TILE * W4_TOP_K_VOCAB_TILE];
    threadgroup ushort cursors[W4_TOP_K_THREADS];
    threadgroup float group_values[W4_SIMDGROUPS * W4_TOP_K];
    threadgroup uint group_ids[W4_SIMDGROUPS * W4_TOP_K];
    #pragma unroll
    for (uint tile_row = 0u; tile_row < ROW_TILE; ++tile_row) {
        uint input_row = first_input_row + tile_row;
        if (input_row >= row_count) {
            continue;
        }
        uint first_offset = lid;
        uint second_offset = lid + W4_TOP_K_THREADS;
        float first_value = first_offset < tile_width
            ? bf16_to_f32(tile_logits[tile_row * W4_TOP_K_VOCAB_TILE + first_offset])
            : -INFINITY;
        float second_value = second_offset < tile_width
            ? bf16_to_f32(tile_logits[tile_row * W4_TOP_K_VOCAB_TILE + second_offset])
            : -INFINITY;
        uint first_id = first_offset < tile_width ? vocab_start + first_offset : 0xffffffffu;
        uint second_id = second_offset < tile_width ? vocab_start + second_offset : 0xffffffffu;
        if (w4_candidate_is_better(second_value, second_id, first_value, first_id)) {
            float swapped_value = first_value;
            uint swapped_id = first_id;
            first_value = second_value;
            first_id = second_id;
            second_value = swapped_value;
            second_id = swapped_id;
        }
        uint ordered_start = tile_row * W4_TOP_K_VOCAB_TILE + lid * 2u;
        ordered_values[ordered_start] = first_value;
        ordered_values[ordered_start + 1u] = second_value;
        ordered_offsets[ordered_start] = first_id == 0xffffffffu
            ? ushort(0xffffu)
            : ushort(first_id - vocab_start);
        ordered_offsets[ordered_start + 1u] = second_id == 0xffffffffu
            ? ushort(0xffffu)
            : ushort(second_id - vocab_start);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for (uint tile_row = 0u; tile_row < ROW_TILE; ++tile_row) {
        uint input_row = first_input_row + tile_row;
        if (input_row >= row_count) {
            continue;
        }
        cursors[lid] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0u) {
            uint first_source = simd_group * 32u;
            uint last_source = first_source + 32u;
            uint group_start = simd_group * W4_TOP_K;
            for (uint rank = 0u; rank < W4_TOP_K; ++rank) {
                float best_value = -INFINITY;
                uint best_id = 0xffffffffu;
                uint best_source = 0xffffffffu;
                for (uint source = first_source; source < last_source; ++source) {
                    uint cursor = uint(cursors[source]);
                    if (cursor >= 2u) {
                        continue;
                    }
                    uint position = tile_row * W4_TOP_K_VOCAB_TILE + source * 2u + cursor;
                    ushort candidate_offset = ordered_offsets[position];
                    uint candidate_id = candidate_offset == ushort(0xffffu)
                        ? 0xffffffffu
                        : vocab_start + uint(candidate_offset);
                    float candidate_value = ordered_values[position];
                    if (w4_candidate_is_better(
                            candidate_value,
                            candidate_id,
                            best_value,
                            best_id)) {
                        best_value = candidate_value;
                        best_id = candidate_id;
                        best_source = source;
                    }
                }
                group_values[group_start + rank] = best_value;
                group_ids[group_start + rank] = best_id;
                if (best_source != 0xffffffffu) {
                    cursors[best_source]++;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0u) {
            uint group_cursors[W4_SIMDGROUPS] = {0u, 0u, 0u, 0u};
            uint destination = (input_row * tile_count + vocab_tile) * W4_TOP_K;
            for (uint rank = 0u; rank < W4_TOP_K; ++rank) {
                float best_value = -INFINITY;
                uint best_id = 0xffffffffu;
                uint best_group = 0xffffffffu;
                #pragma unroll
                for (uint group_index = 0u; group_index < W4_SIMDGROUPS; ++group_index) {
                    uint position = group_index * W4_TOP_K + group_cursors[group_index];
                    float candidate_value = group_values[position];
                    uint candidate_id = group_ids[position];
                    if (w4_candidate_is_better(
                            candidate_value,
                            candidate_id,
                            best_value,
                            best_id)) {
                        best_value = candidate_value;
                        best_id = candidate_id;
                        best_group = group_index;
                    }
                }
                local_ids[destination + rank] = best_id;
                local_values[destination + rank] = best_value;
                group_cursors[best_group]++;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Merges the exact top-16 emitted by every vocabulary tile. A global top-16
// can only contain values present in a tile's local top-16, so this reduction
// is lossless and deterministic.
kernel void qwen_mlx_w4_top_k_merge_kernel(
    const device uint* local_ids [[buffer(0)]],
    const device float* local_values [[buffer(1)]],
    device uint* candidate_ids [[buffer(2)]],
    device float* candidate_values [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& tile_count [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]]
) {
    if (row >= row_count) {
        return;
    }
    float best_values[W4_TOP_K];
    uint best_ids[W4_TOP_K];
    #pragma unroll
    for (uint rank = 0u; rank < W4_TOP_K; ++rank) {
        best_values[rank] = -INFINITY;
        best_ids[rank] = 0xffffffffu;
    }
    uint row_start = row * tile_count * W4_TOP_K;
    uint candidate_count = tile_count * W4_TOP_K;
    for (uint index = lid; index < candidate_count; index += W4_TOP_K_THREADS) {
        float value = local_values[row_start + index];
        uint id = local_ids[row_start + index];
        uint last = W4_TOP_K - 1u;
        if (!w4_candidate_is_better(value, id, best_values[last], best_ids[last])) {
            continue;
        }
        uint insert = last;
        while (insert > 0u && w4_candidate_is_better(
                value,
                id,
                best_values[insert - 1u],
                best_ids[insert - 1u])) {
            best_values[insert] = best_values[insert - 1u];
            best_ids[insert] = best_ids[insert - 1u];
            insert--;
        }
        best_values[insert] = value;
        best_ids[insert] = id;
    }

    threadgroup float values[W4_TOP_K_THREADS * W4_TOP_K];
    threadgroup uint ids[W4_TOP_K_THREADS * W4_TOP_K];
    threadgroup uint cursors[W4_TOP_K_THREADS];
    uint local_start = lid * W4_TOP_K;
    #pragma unroll
    for (uint rank = 0u; rank < W4_TOP_K; ++rank) {
        values[local_start + rank] = best_values[rank];
        ids[local_start + rank] = best_ids[rank];
    }
    cursors[lid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0u) {
        for (uint rank = 0u; rank < W4_TOP_K; ++rank) {
            float best_value = -INFINITY;
            uint best_id = 0xffffffffu;
            uint best_source = 0xffffffffu;
            for (uint source = 0u; source < W4_TOP_K_THREADS; ++source) {
                uint cursor = cursors[source];
                if (cursor >= W4_TOP_K) {
                    continue;
                }
                uint position = source * W4_TOP_K + cursor;
                float value = values[position];
                uint id = ids[position];
                if (w4_candidate_is_better(value, id, best_value, best_id)) {
                    best_value = value;
                    best_id = id;
                    best_source = source;
                }
            }
            candidate_ids[row * W4_TOP_K + rank] = best_id;
            candidate_values[row * W4_TOP_K + rank] = best_value;
            if (best_source != 0xffffffffu) {
                cursors[best_source]++;
            }
        }
    }
}

// DFlash target verification is a small matrix multiplication, not a matvec.
// Two SIMD-groups split K while four output rows share each loaded activation.
kernel void qwen_mlx_w4_verify5_ksplit_linear_bf16_kernel(
    const device uint* packed [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device ushort* biases [[buffer(2)]],
    const device ushort* input [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    constant uint& input_width [[buffer(5)]],
    constant uint& output_width [[buffer(6)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint k_part [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint OUTPUT_TILE = 4u;
    constexpr uint K_PARTS = 2u;
    constexpr uint ACCUMULATORS = OUTPUT_TILE * W4_VERIFY_ROWS;

    uint first_output = output_tile * OUTPUT_TILE;
    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    uint words_per_part = words_per_row / K_PARTS;
    uint first_word = k_part * words_per_part;
    uint last_word = k_part + 1u == K_PARTS ? words_per_row : first_word + words_per_part;
    float sums[ACCUMULATORS];
    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        sums[index] = 0.0f;
    }

    for (uint word_index = first_word + lane; word_index < last_word; word_index += 32u) {
        uint input_column = word_index * W4_VALUES_PER_WORD;
        uint parameter_column = word_index / (W4_GROUP_SIZE / W4_VALUES_PER_WORD);
        uint words[OUTPUT_TILE];
        float weight_scales[OUTPUT_TILE];
        float weight_biases[OUTPUT_TILE];
        #pragma unroll
        for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
            uint output_row = min(first_output + output_offset, output_width - 1u);
            words[output_offset] = packed[output_row * words_per_row + word_index];
            uint parameter = output_row * groups_per_row + parameter_column;
            weight_scales[output_offset] = bf16_to_f32(scales[parameter]);
            weight_biases[output_offset] = bf16_to_f32(biases[parameter]);
        }

        #pragma unroll
        for (uint nibble = 0u; nibble < W4_VALUES_PER_WORD; ++nibble) {
            float values[W4_VERIFY_ROWS];
            #pragma unroll
            for (uint row = 0u; row < W4_VERIFY_ROWS; ++row) {
                values[row] = bf16_to_f32(input[row * input_width + input_column + nibble]);
            }
            #pragma unroll
            for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
                float weight = float((words[output_offset] >> (4u * nibble)) & 0x0fu)
                    * weight_scales[output_offset] + weight_biases[output_offset];
                #pragma unroll
                for (uint row = 0u; row < W4_VERIFY_ROWS; ++row) {
                    sums[output_offset * W4_VERIFY_ROWS + row] += values[row] * weight;
                }
            }
        }
    }

    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        sums[index] = simd_sum(sums[index]);
    }
    threadgroup float partials[K_PARTS * ACCUMULATORS];
    if (lane == 0u) {
        #pragma unroll
        for (uint index = 0u; index < ACCUMULATORS; ++index) {
            partials[k_part * ACCUMULATORS + index] = sums[index];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (k_part == 0u && lane < ACCUMULATORS) {
        uint output_offset = lane / W4_VERIFY_ROWS;
        uint row = lane - output_offset * W4_VERIFY_ROWS;
        uint output_row = first_output + output_offset;
        if (output_row < output_width) {
            float total = partials[lane] + partials[ACCUMULATORS + lane];
            output[row * output_width + output_row] = f32_to_bf16(total);
        }
    }
}

// Eight-token DFlash verification. Four output neurons share each weight pass,
// while two SIMD-groups split K.
kernel void qwen_mlx_w4_verify8_ksplit_linear_bf16_kernel(
    const device uint* packed [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device ushort* biases [[buffer(2)]],
    const device ushort* input [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    constant uint& input_width [[buffer(5)]],
    constant uint& output_width [[buffer(6)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint k_part [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint VERIFY_ROWS = 8u;
    constexpr uint OUTPUT_TILE = 4u;
    constexpr uint K_PARTS = 2u;
    constexpr uint ACCUMULATORS = VERIFY_ROWS * OUTPUT_TILE;

    uint first_output = output_tile * OUTPUT_TILE;
    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    uint words_per_part = words_per_row / K_PARTS;
    uint first_word = k_part * words_per_part;
    uint last_word = k_part + 1u == K_PARTS ? words_per_row : first_word + words_per_part;
    float sums[ACCUMULATORS];
    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        sums[index] = 0.0f;
    }

    for (uint word_index = first_word + lane; word_index < last_word; word_index += 32u) {
        uint input_column = word_index * W4_VALUES_PER_WORD;
        uint parameter_column = word_index / (W4_GROUP_SIZE / W4_VALUES_PER_WORD);
        uint words[OUTPUT_TILE];
        float weight_scales[OUTPUT_TILE];
        float weight_biases[OUTPUT_TILE];
        #pragma unroll
        for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
            uint output_row = min(first_output + output_offset, output_width - 1u);
            words[output_offset] = packed[output_row * words_per_row + word_index];
            uint parameter = output_row * groups_per_row + parameter_column;
            weight_scales[output_offset] = bf16_to_f32(scales[parameter]);
            weight_biases[output_offset] = bf16_to_f32(biases[parameter]);
        }

        #pragma unroll
        for (uint nibble = 0u; nibble < W4_VALUES_PER_WORD; ++nibble) {
            float values[VERIFY_ROWS];
            #pragma unroll
            for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                values[row] = bf16_to_f32(input[row * input_width + input_column + nibble]);
            }
            #pragma unroll
            for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
                float weight = float((words[output_offset] >> (4u * nibble)) & 0x0fu)
                    * weight_scales[output_offset] + weight_biases[output_offset];
                #pragma unroll
                for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                    sums[output_offset * VERIFY_ROWS + row] += values[row] * weight;
                }
            }
        }
    }

    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        sums[index] = simd_sum(sums[index]);
    }
    threadgroup float partials[K_PARTS * ACCUMULATORS];
    if (lane == 0u) {
        #pragma unroll
        for (uint index = 0u; index < ACCUMULATORS; ++index) {
            partials[k_part * ACCUMULATORS + index] = sums[index];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (k_part == 0u && lane < ACCUMULATORS) {
        uint output_offset = lane / VERIFY_ROWS;
        uint row = lane - output_offset * VERIFY_ROWS;
        uint output_row = first_output + output_offset;
        if (output_row < output_width) {
            float total = partials[lane] + partials[ACCUMULATORS + lane];
            output[row * output_width + output_row] = f32_to_bf16(total);
        }
    }
}

// Prompt GEMM for Apple SIMD-group matrix units. Each threadgroup computes an
// [8 token, 16 output] tile while eight SIMD-groups split K.
kernel void qwen_mlx_w4_prefill_matrix_linear_bf16_kernel(
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
    constexpr uint TOKEN_ROWS = 8u;
    constexpr uint OUTPUT_TILE = 16u;
    constexpr uint K_TILE = 32u;
    constexpr uint K_SUB_TILE = 8u;
    constexpr uint K_PARTS = 8u;

    uint output_tiles = (output_width + OUTPUT_TILE - 1u) / OUTPUT_TILE;
    uint output_tile = group % output_tiles;
    uint input_tile = group / output_tiles;
    uint first_output = output_tile * OUTPUT_TILE;
    uint first_input = input_tile * TOKEN_ROWS;
    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    uint k_per_part = input_width / K_PARTS;
    uint first_k = k_part * k_per_part;
    uint last_k = first_k + k_per_part;

    threadgroup bfloat activation_tile[K_PARTS][TOKEN_ROWS * K_TILE];
    threadgroup bfloat weight_tile[K_PARTS][K_TILE * OUTPUT_TILE];
    threadgroup float partials[K_PARTS][TOKEN_ROWS * OUTPUT_TILE];
    simdgroup_matrix<bfloat, 8, 8> activation;
    simdgroup_matrix<bfloat, 8, 8> weight_left;
    simdgroup_matrix<bfloat, 8, 8> weight_right;
    simdgroup_matrix<float, 8, 8> output_left =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> output_right =
        simdgroup_matrix<float, 8, 8>(0.0f);

    uint dequant_output = lane % OUTPUT_TILE;
    uint dequant_k_lane = lane / OUTPUT_TILE;
    for (uint k_start = first_k; k_start < last_k; k_start += K_TILE) {
        for (uint index = lane; index < TOKEN_ROWS * K_TILE; index += 32u) {
            uint row = index / K_TILE;
            uint column = index - row * K_TILE;
            uint input_row = first_input + row;
            activation_tile[k_part][index] = input_row < row_count
                ? input[input_row * input_width + k_start + column]
                : bfloat(0.0f);
        }
        #pragma unroll
        for (uint pack_index = 0u; pack_index < 2u; ++pack_index) {
            uint packed_k = pack_index * 2u + dequant_k_lane;
            uint output_row = min(first_output + dequant_output, output_width - 1u);
            uint k_base = k_start + packed_k * W4_VALUES_PER_WORD;
            uint word = packed[
                output_row * words_per_row + k_base / W4_VALUES_PER_WORD
            ];
            uint parameter = output_row * groups_per_row + k_base / W4_GROUP_SIZE;
            float scale = bf16_to_f32(scales[parameter]);
            float bias = bf16_to_f32(biases[parameter]);
            #pragma unroll
            for (uint nibble = 0u; nibble < W4_VALUES_PER_WORD; ++nibble) {
                weight_tile[k_part][
                    (packed_k * W4_VALUES_PER_WORD + nibble) * OUTPUT_TILE + dequant_output
                ] = bfloat(
                    float((word >> (4u * nibble)) & 0x0fu) * scale + bias
                );
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint sub_tile = 0u; sub_tile < K_TILE / K_SUB_TILE; ++sub_tile) {
            simdgroup_load(
                activation,
                activation_tile[k_part] + sub_tile * K_SUB_TILE,
                K_TILE
            );
            simdgroup_load(
                weight_left,
                weight_tile[k_part] + sub_tile * K_SUB_TILE * OUTPUT_TILE,
                OUTPUT_TILE
            );
            simdgroup_load(
                weight_right,
                weight_tile[k_part] + sub_tile * K_SUB_TILE * OUTPUT_TILE + 8u,
                OUTPUT_TILE
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
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(output_left, partials[k_part], OUTPUT_TILE);
    simdgroup_store(output_right, partials[k_part] + 8u, OUTPUT_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < TOKEN_ROWS * OUTPUT_TILE) {
        float sum = 0.0f;
        #pragma unroll
        for (uint part = 0u; part < K_PARTS; ++part) {
            sum += partials[part][tid];
        }
        uint tile_row = tid / OUTPUT_TILE;
        uint row = first_input + tile_row;
        uint output_offset = tid - tile_row * OUTPUT_TILE;
        uint output_row = first_output + output_offset;
        if (row < row_count && output_row < output_width) {
            output[row * output_width + output_row] = f32_to_bf16(sum);
        }
    }
}

kernel void qwen_mlx_w4_gate_up_swiglu_bf16_kernel(
    const device uint* gate_packed [[buffer(0)]],
    const device ushort* gate_scales [[buffer(1)]],
    const device ushort* gate_biases [[buffer(2)]],
    const device uint* up_packed [[buffer(3)]],
    const device ushort* up_scales [[buffer(4)]],
    const device ushort* up_biases [[buffer(5)]],
    const device ushort* input [[buffer(6)]],
    device ushort* output [[buffer(7)]],
    constant uint& row_count [[buffer(8)]],
    constant uint& input_width [[buffer(9)]],
    constant uint& output_width [[buffer(10)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint output_row = group % output_width;
    uint input_tile = group / output_width;
    uint first_input_row = input_tile * W4_ROW_TILE;
    if (output_row >= output_width || first_input_row >= row_count) {
        return;
    }

    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    uint weight_start = output_row * words_per_row;
    uint parameter_start = output_row * groups_per_row;
    float gate_sum[W4_ROW_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};
    float up_sum[W4_ROW_TILE] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (uint word_index = lid; word_index < words_per_row; word_index += 128u) {
        uint gate_word = gate_packed[weight_start + word_index];
        uint up_word = up_packed[weight_start + word_index];
        uint parameter = parameter_start + word_index / (W4_GROUP_SIZE / W4_VALUES_PER_WORD);
        float gate_scale = bf16_to_f32(gate_scales[parameter]);
        float gate_bias = bf16_to_f32(gate_biases[parameter]);
        float up_scale = bf16_to_f32(up_scales[parameter]);
        float up_bias = bf16_to_f32(up_biases[parameter]);
        uint input_column = word_index * W4_VALUES_PER_WORD;

        for (uint nibble = 0; nibble < W4_VALUES_PER_WORD; ++nibble) {
            float gate_weight =
                float((gate_word >> (4u * nibble)) & 0x0fu) * gate_scale + gate_bias;
            float up_weight =
                float((up_word >> (4u * nibble)) & 0x0fu) * up_scale + up_bias;
            for (uint tile_row = 0; tile_row < W4_ROW_TILE; ++tile_row) {
                uint input_row = first_input_row + tile_row;
                if (input_row < row_count) {
                    float value = bf16_to_f32(
                        input[input_row * input_width + input_column + nibble]);
                    gate_sum[tile_row] += value * gate_weight;
                    up_sum[tile_row] += value * up_weight;
                }
            }
        }
    }

    threadgroup float gate_partials[W4_SIMDGROUPS * W4_ROW_TILE];
    threadgroup float up_partials[W4_SIMDGROUPS * W4_ROW_TILE];
    for (uint tile_row = 0; tile_row < W4_ROW_TILE; ++tile_row) {
        float gate_reduced = simd_sum(gate_sum[tile_row]);
        float up_reduced = simd_sum(up_sum[tile_row]);
        if (lane == 0u) {
            gate_partials[simd_group * W4_ROW_TILE + tile_row] = gate_reduced;
            up_partials[simd_group * W4_ROW_TILE + tile_row] = up_reduced;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0u && lane < W4_ROW_TILE) {
        uint input_row = first_input_row + lane;
        if (input_row < row_count) {
            float gate = 0.0f;
            float up = 0.0f;
            for (uint subgroup = 0; subgroup < W4_SIMDGROUPS; ++subgroup) {
                gate += gate_partials[subgroup * W4_ROW_TILE + lane];
                up += up_partials[subgroup * W4_ROW_TILE + lane];
            }
            gate = round_bf16(gate);
            up = round_bf16(up);
            float activated = round_bf16(gate / (1.0f + exp(-gate)));
            output[input_row * output_width + output_row] =
                f32_to_bf16(round_bf16(activated * up));
        }
    }
}

// Five-row target verification with the exact accumulation, reduction, and
// BF16 rounding order used by the autoregressive gate/up kernel. Two output
// neurons share each activation load while every decoded W4 weight is reused
// across all candidate rows.
kernel void qwen_mlx_w4_verify5_exact_gate_up_swiglu_bf16_kernel(
    const device uint* gate_packed [[buffer(0)]],
    const device ushort* gate_scales [[buffer(1)]],
    const device ushort* gate_biases [[buffer(2)]],
    const device uint* up_packed [[buffer(3)]],
    const device ushort* up_scales [[buffer(4)]],
    const device ushort* up_biases [[buffer(5)]],
    const device ushort* input [[buffer(6)]],
    device ushort* output [[buffer(7)]],
    constant uint& input_width [[buffer(8)]],
    constant uint& output_width [[buffer(9)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint VERIFY_ROWS = 5u;
    constexpr uint OUTPUT_TILE = 2u;
    constexpr uint ACCUMULATORS = VERIFY_ROWS * OUTPUT_TILE;
    uint first_output = output_tile * OUTPUT_TILE;
    if (first_output >= output_width) {
        return;
    }

    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    float gate_sums[ACCUMULATORS];
    float up_sums[ACCUMULATORS];
    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        gate_sums[index] = 0.0f;
        up_sums[index] = 0.0f;
    }

    for (uint word_index = lid; word_index < words_per_row; word_index += 128u) {
        uint parameter_column = word_index / (W4_GROUP_SIZE / W4_VALUES_PER_WORD);
        uint gate_words[OUTPUT_TILE];
        uint up_words[OUTPUT_TILE];
        float gate_weight_scales[OUTPUT_TILE];
        float gate_weight_biases[OUTPUT_TILE];
        float up_weight_scales[OUTPUT_TILE];
        float up_weight_biases[OUTPUT_TILE];
        #pragma unroll
        for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
            uint output_row = min(first_output + output_offset, output_width - 1u);
            uint weight_index = output_row * words_per_row + word_index;
            uint parameter = output_row * groups_per_row + parameter_column;
            gate_words[output_offset] = gate_packed[weight_index];
            up_words[output_offset] = up_packed[weight_index];
            gate_weight_scales[output_offset] = bf16_to_f32(gate_scales[parameter]);
            gate_weight_biases[output_offset] = bf16_to_f32(gate_biases[parameter]);
            up_weight_scales[output_offset] = bf16_to_f32(up_scales[parameter]);
            up_weight_biases[output_offset] = bf16_to_f32(up_biases[parameter]);
        }
        uint input_column = word_index * W4_VALUES_PER_WORD;

        #pragma unroll
        for (uint nibble = 0u; nibble < W4_VALUES_PER_WORD; ++nibble) {
            float values[VERIFY_ROWS];
            #pragma unroll
            for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                values[row] = bf16_to_f32(
                    input[row * input_width + input_column + nibble]
                );
            }
            #pragma unroll
            for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
                float gate_weight = float(
                    (gate_words[output_offset] >> (4u * nibble)) & 0x0fu
                ) * gate_weight_scales[output_offset] + gate_weight_biases[output_offset];
                float up_weight = float(
                    (up_words[output_offset] >> (4u * nibble)) & 0x0fu
                ) * up_weight_scales[output_offset] + up_weight_biases[output_offset];
                #pragma unroll
                for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                    uint index = output_offset * VERIFY_ROWS + row;
                    gate_sums[index] += values[row] * gate_weight;
                    up_sums[index] += values[row] * up_weight;
                }
            }
        }
    }

    threadgroup float gate_partials[W4_SIMDGROUPS * ACCUMULATORS];
    threadgroup float up_partials[W4_SIMDGROUPS * ACCUMULATORS];
    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        float gate_reduced = simd_sum(gate_sums[index]);
        float up_reduced = simd_sum(up_sums[index]);
        if (lane == 0u) {
            gate_partials[simd_group * ACCUMULATORS + index] = gate_reduced;
            up_partials[simd_group * ACCUMULATORS + index] = up_reduced;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint result_index = simd_group * 32u + lane;
    if (result_index < ACCUMULATORS) {
        uint output_offset = result_index / VERIFY_ROWS;
        uint row = result_index - output_offset * VERIFY_ROWS;
        uint output_row = first_output + output_offset;
        float gate = 0.0f;
        float up = 0.0f;
        #pragma unroll
        for (uint subgroup = 0u; subgroup < W4_SIMDGROUPS; ++subgroup) {
            gate += gate_partials[subgroup * ACCUMULATORS + result_index];
            up += up_partials[subgroup * ACCUMULATORS + result_index];
        }
        if (output_row < output_width) {
            gate = round_bf16(gate);
            up = round_bf16(up);
            float activated = round_bf16(gate / (1.0f + exp(-gate)));
            output[row * output_width + output_row] =
                f32_to_bf16(round_bf16(activated * up));
        }
    }
}

// Eight-row target verification with the exact accumulation, reduction, and
// BF16 rounding order used by the autoregressive gate/up kernel.
kernel void qwen_mlx_w4_verify8_exact_gate_up_swiglu_bf16_kernel(
    const device uint* gate_packed [[buffer(0)]],
    const device ushort* gate_scales [[buffer(1)]],
    const device ushort* gate_biases [[buffer(2)]],
    const device uint* up_packed [[buffer(3)]],
    const device ushort* up_scales [[buffer(4)]],
    const device ushort* up_biases [[buffer(5)]],
    const device ushort* input [[buffer(6)]],
    device ushort* output [[buffer(7)]],
    constant uint& input_width [[buffer(8)]],
    constant uint& output_width [[buffer(9)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint VERIFY_ROWS = 8u;
    constexpr uint OUTPUT_TILE = 2u;
    constexpr uint ACCUMULATORS = VERIFY_ROWS * OUTPUT_TILE;
    uint first_output = output_tile * OUTPUT_TILE;
    if (first_output >= output_width) {
        return;
    }

    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    float gate_sums[ACCUMULATORS];
    float up_sums[ACCUMULATORS];
    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        gate_sums[index] = 0.0f;
        up_sums[index] = 0.0f;
    }

    for (uint word_index = lid; word_index < words_per_row; word_index += 128u) {
        uint parameter_column = word_index / (W4_GROUP_SIZE / W4_VALUES_PER_WORD);
        uint gate_words[OUTPUT_TILE];
        uint up_words[OUTPUT_TILE];
        float gate_weight_scales[OUTPUT_TILE];
        float gate_weight_biases[OUTPUT_TILE];
        float up_weight_scales[OUTPUT_TILE];
        float up_weight_biases[OUTPUT_TILE];
        #pragma unroll
        for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
            uint output_row = min(first_output + output_offset, output_width - 1u);
            uint weight_index = output_row * words_per_row + word_index;
            uint parameter = output_row * groups_per_row + parameter_column;
            gate_words[output_offset] = gate_packed[weight_index];
            up_words[output_offset] = up_packed[weight_index];
            gate_weight_scales[output_offset] = bf16_to_f32(gate_scales[parameter]);
            gate_weight_biases[output_offset] = bf16_to_f32(gate_biases[parameter]);
            up_weight_scales[output_offset] = bf16_to_f32(up_scales[parameter]);
            up_weight_biases[output_offset] = bf16_to_f32(up_biases[parameter]);
        }
        uint input_column = word_index * W4_VALUES_PER_WORD;

        // Load four BF16 inputs together while preserving scalar accumulation order.
        #pragma unroll
        for (uint first_nibble = 0u; first_nibble < W4_VALUES_PER_WORD; first_nibble += 4u) {
            ushort4 values[VERIFY_ROWS];
            #pragma unroll
            for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                values[row] = *reinterpret_cast<const device ushort4*>(
                    input + row * input_width + input_column + first_nibble
                );
            }
            #pragma unroll
            for (uint component = 0u; component < 4u; ++component) {
                uint nibble = first_nibble + component;
                #pragma unroll
                for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
                    float gate_weight = float(
                        (gate_words[output_offset] >> (4u * nibble)) & 0x0fu
                    ) * gate_weight_scales[output_offset] + gate_weight_biases[output_offset];
                    float up_weight = float(
                        (up_words[output_offset] >> (4u * nibble)) & 0x0fu
                    ) * up_weight_scales[output_offset] + up_weight_biases[output_offset];
                    #pragma unroll
                    for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                        uint index = output_offset * VERIFY_ROWS + row;
                        float value = bf16_to_f32(values[row][component]);
                        gate_sums[index] += value * gate_weight;
                        up_sums[index] += value * up_weight;
                    }
                }
            }
        }
    }

    threadgroup float gate_partials[W4_SIMDGROUPS * ACCUMULATORS];
    threadgroup float up_partials[W4_SIMDGROUPS * ACCUMULATORS];
    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        float gate_reduced = simd_sum(gate_sums[index]);
        float up_reduced = simd_sum(up_sums[index]);
        if (lane == 0u) {
            gate_partials[simd_group * ACCUMULATORS + index] = gate_reduced;
            up_partials[simd_group * ACCUMULATORS + index] = up_reduced;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint result_index = simd_group * 32u + lane;
    if (result_index < ACCUMULATORS) {
        uint output_offset = result_index / VERIFY_ROWS;
        uint row = result_index - output_offset * VERIFY_ROWS;
        uint output_row = first_output + output_offset;
        float gate = 0.0f;
        float up = 0.0f;
        #pragma unroll
        for (uint subgroup = 0u; subgroup < W4_SIMDGROUPS; ++subgroup) {
            gate += gate_partials[subgroup * ACCUMULATORS + result_index];
            up += up_partials[subgroup * ACCUMULATORS + result_index];
        }
        if (output_row < output_width) {
            gate = round_bf16(gate);
            up = round_bf16(up);
            float activated = round_bf16(gate / (1.0f + exp(-gate)));
            output[row * output_width + output_row] =
                f32_to_bf16(round_bf16(activated * up));
        }
    }
}

kernel void qwen_mlx_w4_verify5_gate_up_swiglu_bf16_kernel(
    const device uint* gate_packed [[buffer(0)]],
    const device ushort* gate_scales [[buffer(1)]],
    const device ushort* gate_biases [[buffer(2)]],
    const device uint* up_packed [[buffer(3)]],
    const device ushort* up_scales [[buffer(4)]],
    const device ushort* up_biases [[buffer(5)]],
    const device ushort* input [[buffer(6)]],
    device ushort* output [[buffer(7)]],
    constant uint& input_width [[buffer(8)]],
    constant uint& output_width [[buffer(9)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint k_lane = lane % 8u;
    uint row_in_simd = lane / 8u;
    uint output_row = output_tile * 8u + simd_group * 4u + row_in_simd;
    uint matrix_row = min(output_row, output_width - 1u);
    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    uint weight_start = matrix_row * words_per_row;
    uint parameter_start = matrix_row * groups_per_row;
    float gate_sums[W4_VERIFY_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up_sums[W4_VERIFY_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    for (uint group = k_lane; group < groups_per_row; group += 8u) {
        float gate_scale = bf16_to_f32(gate_scales[parameter_start + group]);
        float gate_bias = bf16_to_f32(gate_biases[parameter_start + group]);
        float up_scale = bf16_to_f32(up_scales[parameter_start + group]);
        float up_bias = bf16_to_f32(up_biases[parameter_start + group]);
        #pragma unroll
        for (uint subchunk = 0u; subchunk < 8u; ++subchunk) {
            uint gate_word = gate_packed[weight_start + group * 8u + subchunk];
            uint up_word = up_packed[weight_start + group * 8u + subchunk];
            uint input_column = group * W4_GROUP_SIZE + subchunk * W4_VALUES_PER_WORD;
            #pragma unroll
            for (uint row = 0; row < W4_VERIFY_ROWS; ++row) {
                float gate_partial = 0.0f;
                float up_partial = 0.0f;
                #pragma unroll
                for (uint nibble = 0u; nibble < W4_VALUES_PER_WORD; ++nibble) {
                    float value = bf16_to_f32(input[row * input_width + input_column + nibble]);
                    gate_partial += value
                        * (float((gate_word >> (4u * nibble)) & 0x0fu) * gate_scale + gate_bias);
                    up_partial += value
                        * (float((up_word >> (4u * nibble)) & 0x0fu) * up_scale + up_bias);
                }
                gate_sums[row] += gate_partial;
                up_sums[row] += up_partial;
            }
        }
    }

    #pragma unroll
    for (uint row = 0; row < W4_VERIFY_ROWS; ++row) {
        gate_sums[row] += simd_shuffle_down(gate_sums[row], 4u);
        gate_sums[row] += simd_shuffle_down(gate_sums[row], 2u);
        gate_sums[row] += simd_shuffle_down(gate_sums[row], 1u);
        up_sums[row] += simd_shuffle_down(up_sums[row], 4u);
        up_sums[row] += simd_shuffle_down(up_sums[row], 2u);
        up_sums[row] += simd_shuffle_down(up_sums[row], 1u);
    }
    if (k_lane == 0u && output_row < output_width) {
        #pragma unroll
        for (uint row = 0u; row < W4_VERIFY_ROWS; ++row) {
            float gate = round_bf16(gate_sums[row]);
            float up = round_bf16(up_sums[row]);
            float activated = round_bf16(gate / (1.0f + exp(-gate)));
            output[row * output_width + output_row] =
                f32_to_bf16(round_bf16(activated * up));
        }
    }
}

// Eight-token gate/up projection. Two output neurons share each activation
// load while two SIMD-groups split K.
kernel void qwen_mlx_w4_verify8_ksplit_gate_up_swiglu_bf16_kernel(
    const device uint* gate_packed [[buffer(0)]],
    const device ushort* gate_scales [[buffer(1)]],
    const device ushort* gate_biases [[buffer(2)]],
    const device uint* up_packed [[buffer(3)]],
    const device ushort* up_scales [[buffer(4)]],
    const device ushort* up_biases [[buffer(5)]],
    const device ushort* input [[buffer(6)]],
    device ushort* output [[buffer(7)]],
    constant uint& input_width [[buffer(8)]],
    constant uint& output_width [[buffer(9)]],
    uint output_tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint k_part [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint VERIFY_ROWS = 8u;
    constexpr uint OUTPUT_TILE = 2u;
    constexpr uint K_PARTS = 2u;
    constexpr uint ACCUMULATORS = VERIFY_ROWS * OUTPUT_TILE;

    uint first_output = output_tile * OUTPUT_TILE;
    if (first_output >= output_width) {
        return;
    }
    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    uint words_per_part = words_per_row / K_PARTS;
    uint first_word = k_part * words_per_part;
    uint last_word = k_part + 1u == K_PARTS ? words_per_row : first_word + words_per_part;
    float gate_sums[ACCUMULATORS];
    float up_sums[ACCUMULATORS];
    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        gate_sums[index] = 0.0f;
        up_sums[index] = 0.0f;
    }

    for (uint word_index = first_word + lane; word_index < last_word; word_index += 32u) {
        uint parameter_column = word_index / (W4_GROUP_SIZE / W4_VALUES_PER_WORD);
        uint gate_words[OUTPUT_TILE];
        uint up_words[OUTPUT_TILE];
        float gate_weight_scales[OUTPUT_TILE];
        float gate_weight_biases[OUTPUT_TILE];
        float up_weight_scales[OUTPUT_TILE];
        float up_weight_biases[OUTPUT_TILE];
        #pragma unroll
        for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
            uint output_row = min(first_output + output_offset, output_width - 1u);
            uint weight_index = output_row * words_per_row + word_index;
            uint parameter = output_row * groups_per_row + parameter_column;
            gate_words[output_offset] = gate_packed[weight_index];
            up_words[output_offset] = up_packed[weight_index];
            gate_weight_scales[output_offset] = bf16_to_f32(gate_scales[parameter]);
            gate_weight_biases[output_offset] = bf16_to_f32(gate_biases[parameter]);
            up_weight_scales[output_offset] = bf16_to_f32(up_scales[parameter]);
            up_weight_biases[output_offset] = bf16_to_f32(up_biases[parameter]);
        }
        uint input_column = word_index * W4_VALUES_PER_WORD;

        #pragma unroll
        for (uint nibble = 0u; nibble < W4_VALUES_PER_WORD; ++nibble) {
            float values[VERIFY_ROWS];
            #pragma unroll
            for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                values[row] = bf16_to_f32(input[row * input_width + input_column + nibble]);
            }
            #pragma unroll
            for (uint output_offset = 0u; output_offset < OUTPUT_TILE; ++output_offset) {
                float gate_weight =
                    float((gate_words[output_offset] >> (4u * nibble)) & 0x0fu)
                        * gate_weight_scales[output_offset]
                        + gate_weight_biases[output_offset];
                float up_weight =
                    float((up_words[output_offset] >> (4u * nibble)) & 0x0fu)
                        * up_weight_scales[output_offset]
                        + up_weight_biases[output_offset];
                #pragma unroll
                for (uint row = 0u; row < VERIFY_ROWS; ++row) {
                    uint index = output_offset * VERIFY_ROWS + row;
                    gate_sums[index] += values[row] * gate_weight;
                    up_sums[index] += values[row] * up_weight;
                }
            }
        }
    }

    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; ++index) {
        gate_sums[index] = simd_sum(gate_sums[index]);
        up_sums[index] = simd_sum(up_sums[index]);
    }
    threadgroup float gate_partials[K_PARTS * ACCUMULATORS];
    threadgroup float up_partials[K_PARTS * ACCUMULATORS];
    if (lane == 0u) {
        #pragma unroll
        for (uint index = 0u; index < ACCUMULATORS; ++index) {
            gate_partials[k_part * ACCUMULATORS + index] = gate_sums[index];
            up_partials[k_part * ACCUMULATORS + index] = up_sums[index];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (k_part == 0u && lane < ACCUMULATORS) {
        uint output_offset = lane / VERIFY_ROWS;
        uint row = lane - output_offset * VERIFY_ROWS;
        uint output_row = first_output + output_offset;
        if (output_row < output_width) {
            float gate = round_bf16(
                gate_partials[lane] + gate_partials[ACCUMULATORS + lane]);
            float up = round_bf16(
                up_partials[lane] + up_partials[ACCUMULATORS + lane]);
            float activated = round_bf16(gate / (1.0f + exp(-gate)));
            output[row * output_width + output_row] =
                f32_to_bf16(round_bf16(activated * up));
        }
    }
}

// Prompt-only fused gate/up GEMM. The activation tile is loaded once, both W4
// projections use the SIMD-group matrix units, and only SwiGLU is written.
kernel void qwen_mlx_w4_prefill_matrix_gate_up_swiglu_bf16_kernel(
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
    constexpr uint TOKEN_ROWS = 8u;
    constexpr uint OUTPUT_TILE = 8u;
    constexpr uint K_TILE = 32u;
    constexpr uint K_SUB_TILE = 8u;
    constexpr uint K_PARTS = 8u;

    uint output_tiles = (output_width + OUTPUT_TILE - 1u) / OUTPUT_TILE;
    uint output_tile = group % output_tiles;
    uint input_tile = group / output_tiles;
    uint first_output = output_tile * OUTPUT_TILE;
    uint first_input = input_tile * TOKEN_ROWS;
    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    uint k_per_part = input_width / K_PARTS;
    uint first_k = k_part * k_per_part;
    uint last_k = first_k + k_per_part;

    threadgroup bfloat activation_tile[K_PARTS][TOKEN_ROWS * K_TILE];
    threadgroup bfloat gate_weight_tile[K_PARTS][K_TILE * OUTPUT_TILE];
    threadgroup bfloat up_weight_tile[K_PARTS][K_TILE * OUTPUT_TILE];
    threadgroup float gate_partials[K_PARTS][TOKEN_ROWS * OUTPUT_TILE];
    threadgroup float up_partials[K_PARTS][TOKEN_ROWS * OUTPUT_TILE];
    simdgroup_matrix<bfloat, 8, 8> activation;
    simdgroup_matrix<bfloat, 8, 8> gate_left;
    simdgroup_matrix<bfloat, 8, 8> up_left;
    simdgroup_matrix<float, 8, 8> gate_output =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> up_output =
        simdgroup_matrix<float, 8, 8>(0.0f);

    uint dequant_output = lane % OUTPUT_TILE;
    uint dequant_k_lane = lane / OUTPUT_TILE;
    for (uint k_start = first_k; k_start < last_k; k_start += K_TILE) {
        for (uint index = lane; index < TOKEN_ROWS * K_TILE; index += 32u) {
            uint row = index / K_TILE;
            uint column = index - row * K_TILE;
            uint input_row = first_input + row;
            activation_tile[k_part][index] = input_row < row_count
                ? input[input_row * input_width + k_start + column]
                : bfloat(0.0f);
        }
        #pragma unroll
        for (uint pack_index = 0u; pack_index < 1u; ++pack_index) {
            uint packed_k = dequant_k_lane;
            uint output_row = min(first_output + dequant_output, output_width - 1u);
            uint k_base = k_start + packed_k * W4_VALUES_PER_WORD;
            uint word_index = output_row * words_per_row
                + k_base / W4_VALUES_PER_WORD;
            uint parameter = output_row * groups_per_row + k_base / W4_GROUP_SIZE;
            uint gate_word = gate_packed[word_index];
            uint up_word = up_packed[word_index];
            float gate_scale = bf16_to_f32(gate_scales[parameter]);
            float gate_bias = bf16_to_f32(gate_biases[parameter]);
            float up_scale = bf16_to_f32(up_scales[parameter]);
            float up_bias = bf16_to_f32(up_biases[parameter]);
            #pragma unroll
            for (uint nibble = 0u; nibble < W4_VALUES_PER_WORD; ++nibble) {
                uint destination =
                    (packed_k * W4_VALUES_PER_WORD + nibble) * OUTPUT_TILE
                    + dequant_output;
                gate_weight_tile[k_part][destination] = bfloat(
                    float((gate_word >> (4u * nibble)) & 0x0fu) * gate_scale
                        + gate_bias
                );
                up_weight_tile[k_part][destination] = bfloat(
                    float((up_word >> (4u * nibble)) & 0x0fu) * up_scale
                        + up_bias
                );
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint sub_tile = 0u; sub_tile < K_TILE / K_SUB_TILE; ++sub_tile) {
            uint weight_offset = sub_tile * K_SUB_TILE * OUTPUT_TILE;
            simdgroup_load(
                activation,
                activation_tile[k_part] + sub_tile * K_SUB_TILE,
                K_TILE
            );
            simdgroup_load(
                gate_left,
                gate_weight_tile[k_part] + weight_offset,
                OUTPUT_TILE
            );
            simdgroup_load(
                up_left,
                up_weight_tile[k_part] + weight_offset,
                OUTPUT_TILE
            );
            simdgroup_multiply_accumulate(
                gate_output,
                activation,
                gate_left,
                gate_output
            );
            simdgroup_multiply_accumulate(
                up_output,
                activation,
                up_left,
                up_output
            );
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(gate_output, gate_partials[k_part], OUTPUT_TILE);
    simdgroup_store(up_output, up_partials[k_part], OUTPUT_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < TOKEN_ROWS * OUTPUT_TILE) {
        float gate = 0.0f;
        float up = 0.0f;
        #pragma unroll
        for (uint part = 0u; part < K_PARTS; ++part) {
            gate += gate_partials[part][tid];
            up += up_partials[part][tid];
        }
        uint tile_row = tid / OUTPUT_TILE;
        uint row = first_input + tile_row;
        uint output_offset = tid - tile_row * OUTPUT_TILE;
        uint output_row = first_output + output_offset;
        if (row < row_count && output_row < output_width) {
            gate = round_bf16(gate);
            up = round_bf16(up);
            float activated = round_bf16(gate / (1.0f + exp(-gate)));
            output[row * output_width + output_row] =
                f32_to_bf16(round_bf16(activated * up));
        }
    }
}

// Long-prompt linear projection. Two eight-token SIMD-group sets share one
// dequantized Q4 tile while preserving the eight K partitions and reduction
// order used by the exact eight-token prefill kernel.
kernel void qwen_mlx_w4_prefill_wide_linear_bf16_kernel(
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
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint TOKEN_ROWS = 16u;
    constexpr uint TOKEN_HALVES = 2u;
    constexpr uint OUTPUT_TILE = 16u;
    constexpr uint K_TILE = 32u;
    constexpr uint K_SUB_TILE = 8u;
    constexpr uint K_PARTS = 8u;

    uint token_half = simd_group / K_PARTS;
    uint k_part = simd_group - token_half * K_PARTS;
    uint output_tiles = (output_width + OUTPUT_TILE - 1u) / OUTPUT_TILE;
    uint output_tile = group % output_tiles;
    uint input_tile = group / output_tiles;
    uint first_output = output_tile * OUTPUT_TILE;
    uint first_input = input_tile * TOKEN_ROWS;
    uint words_per_row = input_width / W4_VALUES_PER_WORD;
    uint groups_per_row = input_width / W4_GROUP_SIZE;
    uint k_per_part = input_width / K_PARTS;
    uint first_k = k_part * k_per_part;
    uint last_k = first_k + k_per_part;

    threadgroup bfloat activation_tile[K_PARTS][TOKEN_ROWS * K_TILE];
    threadgroup bfloat weight_tile[K_PARTS][K_TILE * OUTPUT_TILE];
    threadgroup float partials[K_PARTS][TOKEN_ROWS * OUTPUT_TILE];
    simdgroup_matrix<bfloat, 8, 8> activation;
    simdgroup_matrix<bfloat, 8, 8> weight_left;
    simdgroup_matrix<bfloat, 8, 8> weight_right;
    simdgroup_matrix<float, 8, 8> output_left =
        simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> output_right =
        simdgroup_matrix<float, 8, 8>(0.0f);

    uint dequant_output = lane % OUTPUT_TILE;
    uint dequant_k_lane = lane / OUTPUT_TILE;
    for (uint k_start = first_k; k_start < last_k; k_start += K_TILE) {
        for (uint index = lane; index < 8u * K_TILE; index += 32u) {
            uint row = index / K_TILE;
            uint column = index - row * K_TILE;
            uint input_row = first_input + token_half * 8u + row;
            activation_tile[k_part][token_half * 8u * K_TILE + index] =
                input_row < row_count
                    ? input[input_row * input_width + k_start + column]
                    : bfloat(0.0f);
        }
        if (token_half == 0u) {
            #pragma unroll
            for (uint pack_index = 0u; pack_index < 2u; ++pack_index) {
                uint packed_k = pack_index * 2u + dequant_k_lane;
                uint output_row = min(first_output + dequant_output, output_width - 1u);
                uint k_base = k_start + packed_k * W4_VALUES_PER_WORD;
                uint word = packed[
                    output_row * words_per_row + k_base / W4_VALUES_PER_WORD
                ];
                uint parameter = output_row * groups_per_row + k_base / W4_GROUP_SIZE;
                float scale = bf16_to_f32(scales[parameter]);
                float bias = bf16_to_f32(biases[parameter]);
                #pragma unroll
                for (uint nibble = 0u; nibble < W4_VALUES_PER_WORD; ++nibble) {
                    weight_tile[k_part][
                        (packed_k * W4_VALUES_PER_WORD + nibble) * OUTPUT_TILE
                            + dequant_output
                    ] = bfloat(
                        float((word >> (4u * nibble)) & 0x0fu) * scale + bias
                    );
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint sub_tile = 0u; sub_tile < K_TILE / K_SUB_TILE; ++sub_tile) {
            simdgroup_load(
                activation,
                activation_tile[k_part]
                    + token_half * 8u * K_TILE
                    + sub_tile * K_SUB_TILE,
                K_TILE
            );
            simdgroup_load(
                weight_left,
                weight_tile[k_part] + sub_tile * K_SUB_TILE * OUTPUT_TILE,
                OUTPUT_TILE
            );
            simdgroup_load(
                weight_right,
                weight_tile[k_part] + sub_tile * K_SUB_TILE * OUTPUT_TILE + 8u,
                OUTPUT_TILE
            );
            simdgroup_multiply_accumulate(output_left, activation, weight_left, output_left);
            simdgroup_multiply_accumulate(output_right, activation, weight_right, output_right);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint partial_offset = token_half * 8u * OUTPUT_TILE;
    simdgroup_store(output_left, partials[k_part] + partial_offset, OUTPUT_TILE);
    simdgroup_store(output_right, partials[k_part] + partial_offset + 8u, OUTPUT_TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < TOKEN_ROWS * OUTPUT_TILE) {
        float sum = 0.0f;
        #pragma unroll
        for (uint part = 0u; part < K_PARTS; ++part) {
            sum += partials[part][tid];
        }
        uint tile_row = tid / OUTPUT_TILE;
        uint row = first_input + tile_row;
        uint output_offset = tid - tile_row * OUTPUT_TILE;
        uint output_row = first_output + output_offset;
        if (row < row_count && output_row < output_width) {
            output[row * output_width + output_row] = f32_to_bf16(sum);
        }
    }
}


kernel void qwen_mlx_w4_embedding_bf16_kernel(
    const device uint* packed [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device ushort* biases [[buffer(2)]],
    const device uint* token_ids [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    constant uint& token_count [[buffer(5)]],
    constant uint& hidden_size [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_count = token_count * hidden_size;
    if (gid >= output_count) {
        return;
    }
    uint token_index = gid / hidden_size;
    uint column = gid - token_index * hidden_size;
    uint row = token_ids[token_index];
    uint words_per_row = hidden_size / W4_VALUES_PER_WORD;
    uint groups_per_row = hidden_size / W4_GROUP_SIZE;
    uint word = packed[row * words_per_row + column / W4_VALUES_PER_WORD];
    uint quantized = (word >> (4u * (column % W4_VALUES_PER_WORD))) & 0x0fu;
    uint parameter = row * groups_per_row + column / W4_GROUP_SIZE;
    float value = float(quantized) * bf16_to_f32(scales[parameter])
        + bf16_to_f32(biases[parameter]);
    output[gid] = f32_to_bf16(value);
}

kernel void qwen_mlx_w4_argmax_bf16_kernel(
    const device ushort* logits [[buffer(0)]],
    device uint* token_ids [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    if (row >= row_count) {
        return;
    }
    float best_value = -INFINITY;
    uint best_id = 0u;
    for (uint token = lid; token < vocab_size; token += 256u) {
        float value = bf16_to_f32(logits[row * vocab_size + token]);
        if (value > best_value || (value == best_value && token < best_id)) {
            best_value = value;
            best_id = token;
        }
    }
    float group_max = simd_max(best_value);
    uint candidate = best_value == group_max ? best_id : 0xffffffffu;
    uint group_id = simd_min(candidate);
    threadgroup float maxima[8];
    threadgroup uint ids[8];
    if (lane == 0u) {
        maxima[simd_group] = group_max;
        ids[simd_group] = group_id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0u) {
        float value = lane < 8u ? maxima[lane] : -INFINITY;
        uint id = lane < 8u ? ids[lane] : 0xffffffffu;
        float maximum = simd_max(value);
        uint selected = value == maximum ? id : 0xffffffffu;
        selected = simd_min(selected);
        if (lane == 0u) {
            token_ids[row] = selected;
        }
    }
}
