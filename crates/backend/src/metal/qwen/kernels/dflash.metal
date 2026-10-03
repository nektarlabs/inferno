#include <metal_stdlib>

using namespace metal;

constant uint DFLASH_HIDDEN = 5120u;
constant uint DFLASH_QUERY_HEADS = 32u;
constant uint DFLASH_KV_HEADS = 8u;
constant uint DFLASH_HEAD_DIM = 128u;
constant uint DFLASH_VOCAB = 248320u;
constant uint DFLASH_SELECTOR_RANK = 256u;
constant uint DFLASH_TOP_K = 16u;
constant uint DFLASH_TOP_K_THREADS = 128u;
constant uint DFLASH_MAX_ROWS = 7u;
constant uint DFLASH_LINEAR_ROW_TILE = 8u;
constant uint DFLASH_W4_GROUP_SIZE = 64u;

static inline float dflash_bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

static inline ushort dflash_f32_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    uint exponent = bits & 0x7f800000u;
    if (exponent == 0x7f800000u) {
        return ushort(bits >> 16);
    }
    uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return ushort(rounded >> 16);
}

static inline float dflash_round_bf16(float value) {
    return dflash_bf16_to_f32(dflash_f32_to_bf16(value));
}

static inline bool dflash_candidate_is_better(
    float candidate_value,
    uint candidate_index,
    float current_value,
    uint current_index
) {
    return candidate_value > current_value
        || (candidate_value == current_value && candidate_index < current_index);
}

static inline uint dflash_w4_value(uchar packed, bool upper) {
    return upper ? uint(packed >> 4) : uint(packed & 0x0fu);
}

kernel void dflash_pack_features_kernel(
    const device ushort* feature_0 [[buffer(0)]],
    const device ushort* feature_1 [[buffer(1)]],
    const device ushort* feature_2 [[buffer(2)]],
    const device ushort* feature_3 [[buffer(3)]],
    const device ushort* feature_4 [[buffer(4)]],
    device ushort* output [[buffer(5)]],
    constant uint& row_count [[buffer(6)]],
    constant uint& hidden_size [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_width = hidden_size * 5u;
    uint output_count = row_count * output_width;
    if (gid >= output_count) {
        return;
    }
    uint row = gid / output_width;
    uint column = gid - row * output_width;
    uint feature = column / hidden_size;
    uint hidden = column - feature * hidden_size;
    uint source = row * hidden_size + hidden;
    switch (feature) {
        case 0u: output[gid] = feature_0[source]; break;
        case 1u: output[gid] = feature_1[source]; break;
        case 2u: output[gid] = feature_2[source]; break;
        case 3u: output[gid] = feature_3[source]; break;
        default: output[gid] = feature_4[source]; break;
    }
}

kernel void dflash_copy_rows_kernel(
    const device ushort* input [[buffer(0)]],
    device ushort* output [[buffer(1)]],
    constant uint& row_start [[buffer(2)]],
    constant uint& row_width [[buffer(3)]],
    constant uint& output_count [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < output_count) {
        output[gid] = input[row_start * row_width + gid];
    }
}

kernel void dflash_rms_norm_kernel(
    const device ushort* input [[buffer(0)]],
    const device ushort* weight [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    constant uint& input_row_start [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (row >= row_count) {
        return;
    }
    uint input_start = (input_row_start + row) * hidden_size;
    uint output_start = row * hidden_size;
    float square_sum = 0.0f;
    for (uint hidden = lane; hidden < hidden_size; hidden += 32u) {
        float value = dflash_bf16_to_f32(input[input_start + hidden]);
        square_sum += value * value;
    }
    float inverse_rms = rsqrt(simd_sum(square_sum) / float(hidden_size) + eps);
    inverse_rms = simd_broadcast_first(inverse_rms);
    for (uint hidden = lane; hidden < hidden_size; hidden += 32u) {
        float value = dflash_bf16_to_f32(input[input_start + hidden]);
        float scale = dflash_bf16_to_f32(weight[hidden]);
        float normalized = dflash_round_bf16(value * inverse_rms);
        output[output_start + hidden] = dflash_f32_to_bf16(normalized * scale);
    }
}

// DFlash evaluates several proposal rows together. Sharing each weight value
// across eight rows prevents rereading the complete matrix for every draft
// position while preserving the per-row dot-product reduction order.
kernel void dflash_bf16_linear_rows8_kernel(
    const device ushort* weight [[buffer(0)]],
    const device ushort* input [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row_tiles = (row_count + DFLASH_LINEAR_ROW_TILE - 1u) / DFLASH_LINEAR_ROW_TILE;
    uint output_feature = group % out_features;
    uint row_tile = group / out_features;
    if (row_tile >= row_tiles) {
        return;
    }
    uint row_start = row_tile * DFLASH_LINEAR_ROW_TILE;
    uint weight_start = output_feature * in_features;
    float partials[DFLASH_LINEAR_ROW_TILE] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };

    for (uint column = lane; column < in_features; column += 32u) {
        float weight_value = dflash_bf16_to_f32(weight[weight_start + column]);
        #pragma unroll
        for (uint item = 0u; item < DFLASH_LINEAR_ROW_TILE; item++) {
            uint row = row_start + item;
            if (row < row_count) {
                partials[item] += dflash_bf16_to_f32(input[row * in_features + column])
                    * weight_value;
            }
        }
    }

    #pragma unroll
    for (uint item = 0u; item < DFLASH_LINEAR_ROW_TILE; item++) {
        float sum = simd_sum(partials[item]);
        uint row = row_start + item;
        if (lane == 0u && row < row_count) {
            output[row * out_features + output_feature] = dflash_f32_to_bf16(sum);
        }
    }
}

kernel void dflash_bf16_gate_up_swiglu_rows8_kernel(
    const device ushort* gate_weight [[buffer(0)]],
    const device ushort* up_weight [[buffer(1)]],
    const device ushort* input [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint row_tiles = (row_count + DFLASH_LINEAR_ROW_TILE - 1u) / DFLASH_LINEAR_ROW_TILE;
    uint output_feature = group % out_features;
    uint row_tile = group / out_features;
    if (row_tile >= row_tiles) {
        return;
    }
    uint row_start = row_tile * DFLASH_LINEAR_ROW_TILE;
    uint weight_start = output_feature * in_features;
    float gate_partials[DFLASH_LINEAR_ROW_TILE] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    float up_partials[DFLASH_LINEAR_ROW_TILE] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };

    for (uint column = lane; column < in_features; column += 32u) {
        float gate_value = dflash_bf16_to_f32(gate_weight[weight_start + column]);
        float up_value = dflash_bf16_to_f32(up_weight[weight_start + column]);
        #pragma unroll
        for (uint item = 0u; item < DFLASH_LINEAR_ROW_TILE; item++) {
            uint row = row_start + item;
            if (row < row_count) {
                float input_value = dflash_bf16_to_f32(input[row * in_features + column]);
                gate_partials[item] += input_value * gate_value;
                up_partials[item] += input_value * up_value;
            }
        }
    }

    #pragma unroll
    for (uint item = 0u; item < DFLASH_LINEAR_ROW_TILE; item++) {
        float gate_sum = simd_sum(gate_partials[item]);
        float up_sum = simd_sum(up_partials[item]);
        uint row = row_start + item;
        if (lane == 0u && row < row_count) {
            float gate_bf16 = dflash_round_bf16(gate_sum);
            float up_bf16 = dflash_round_bf16(up_sum);
            float activated = gate_bf16 / (1.0f + exp(-gate_bf16));
            output[row * out_features + output_feature] =
                dflash_f32_to_bf16(activated * up_bf16);
        }
    }
}

kernel void dflash_quantize_bf16_w4_group64_kernel(
    const device ushort* input [[buffer(0)]],
    device uchar* packed [[buffer(1)]],
    device ushort* scales [[buffer(2)]],
    device ushort* biases [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    uint group_index [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint groups_per_row = columns / DFLASH_W4_GROUP_SIZE;
    uint group_count = rows * groups_per_row;
    if (group_index >= group_count) {
        return;
    }
    uint row = group_index / groups_per_row;
    uint group = group_index - row * groups_per_row;
    uint pair_column = group * DFLASH_W4_GROUP_SIZE + lane * 2u;
    uint source = row * columns + pair_column;
    float first = dflash_bf16_to_f32(input[source]);
    float second = dflash_bf16_to_f32(input[source + 1u]);

    float minimum = simd_min(min(first, second));
    float maximum = simd_max(max(0.0f, max(first, second)));
    float scale = max((maximum - minimum) / 15.0f, 1.0e-7f);
    bool negative_edge = fabs(minimum) > fabs(maximum);
    scale = negative_edge ? scale : -scale;
    float edge = negative_edge ? minimum : maximum;
    float q0 = rint(edge / scale);
    bool edge_quantizes_to_zero = q0 == 0.0f;
    scale = edge_quantizes_to_zero ? scale : edge / q0;
    float bias = edge_quantizes_to_zero ? 0.0f : edge;
    scale = simd_broadcast_first(scale);
    bias = simd_broadcast_first(bias);
    uint first_q = uint(clamp(rint((first - bias) / scale), 0.0f, 15.0f));
    uint second_q = uint(clamp(rint((second - bias) / scale), 0.0f, 15.0f));
    uint packed_row_start = row * (columns / 2u);
    uint packed_group_start = packed_row_start + group * (DFLASH_W4_GROUP_SIZE / 2u);
    packed[packed_group_start + lane] = uchar(first_q | (second_q << 4));
    if (lane == 0u) {
        scales[group_index] = dflash_f32_to_bf16(scale);
        biases[group_index] = dflash_f32_to_bf16(bias);
    }
}

kernel void dflash_w4_linear_rows8_kernel(
    const device uchar* packed_weight [[buffer(0)]],
    const device ushort* weight_scales [[buffer(1)]],
    const device ushort* weight_biases [[buffer(2)]],
    const device ushort* input [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& in_features [[buffer(6)]],
    constant uint& out_features [[buffer(7)]],
    uint grid_group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint k_part [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint K_PARTS = 2u;
    constexpr uint OUTPUT_TILE = 2u;
    constexpr uint ACCUMULATORS = OUTPUT_TILE * DFLASH_LINEAR_ROW_TILE;
    uint row_tiles = (row_count + DFLASH_LINEAR_ROW_TILE - 1u) / DFLASH_LINEAR_ROW_TILE;
    uint output_tiles = (out_features + OUTPUT_TILE - 1u) / OUTPUT_TILE;
    uint output_tile = grid_group % output_tiles;
    uint first_output = output_tile * OUTPUT_TILE;
    uint row_tile = grid_group / output_tiles;
    if (row_tile >= row_tiles) {
        return;
    }
    uint row_start = row_tile * DFLASH_LINEAR_ROW_TILE;
    uint groups_per_row = in_features / DFLASH_W4_GROUP_SIZE;
    uint groups_per_part = (groups_per_row + K_PARTS - 1u) / K_PARTS;
    uint first_group = k_part * groups_per_part;
    uint last_group = min(groups_per_row, first_group + groups_per_part);
    float partials[ACCUMULATORS];
    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; index++) {
        partials[index] = 0.0f;
    }

    for (uint group = first_group; group < last_group; group++) {
        uchar packed[OUTPUT_TILE];
        float first_weight[OUTPUT_TILE];
        float second_weight[OUTPUT_TILE];
        #pragma unroll
        for (uint output_offset = 0u; output_offset < OUTPUT_TILE; output_offset++) {
            uint output_feature = min(first_output + output_offset, out_features - 1u);
            uint packed_row_start = output_feature * (in_features / 2u);
            uint scale_row_start = output_feature * groups_per_row;
            packed[output_offset] = packed_weight[
                packed_row_start + group * (DFLASH_W4_GROUP_SIZE / 2u) + lane
            ];
            float scale = dflash_bf16_to_f32(weight_scales[scale_row_start + group]);
            float bias = dflash_bf16_to_f32(weight_biases[scale_row_start + group]);
            first_weight[output_offset] =
                float(dflash_w4_value(packed[output_offset], false)) * scale + bias;
            second_weight[output_offset] =
                float(dflash_w4_value(packed[output_offset], true)) * scale + bias;
        }
        uint input_column = group * DFLASH_W4_GROUP_SIZE + lane * 2u;
        float input_first[DFLASH_LINEAR_ROW_TILE];
        float input_second[DFLASH_LINEAR_ROW_TILE];
        #pragma unroll
        for (uint item = 0u; item < DFLASH_LINEAR_ROW_TILE; item++) {
            uint row = row_start + item;
            if (row < row_count) {
                uint input_start = row * in_features + input_column;
                input_first[item] = dflash_bf16_to_f32(input[input_start]);
                input_second[item] = dflash_bf16_to_f32(input[input_start + 1u]);
            } else {
                input_first[item] = 0.0f;
                input_second[item] = 0.0f;
            }
        }
        #pragma unroll
        for (uint output_offset = 0u; output_offset < OUTPUT_TILE; output_offset++) {
            #pragma unroll
            for (uint item = 0u; item < DFLASH_LINEAR_ROW_TILE; item++) {
                uint index = output_offset * DFLASH_LINEAR_ROW_TILE + item;
                partials[index] += input_first[item] * first_weight[output_offset]
                    + input_second[item] * second_weight[output_offset];
            }
        }
    }

    threadgroup float reduced[K_PARTS * ACCUMULATORS];
    #pragma unroll
    for (uint index = 0u; index < ACCUMULATORS; index++) {
        float sum = simd_sum(partials[index]);
        if (lane == 0u) {
            reduced[k_part * ACCUMULATORS + index] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k_part == 0u && lane < ACCUMULATORS) {
        uint output_offset = lane / DFLASH_LINEAR_ROW_TILE;
        uint item = lane - output_offset * DFLASH_LINEAR_ROW_TILE;
        uint output_feature = first_output + output_offset;
        uint row = row_start + item;
        if (output_feature < out_features && row < row_count) {
            float sum = reduced[lane] + reduced[ACCUMULATORS + lane];
            output[row * out_features + output_feature] = dflash_f32_to_bf16(sum);
        }
    }
}

kernel void dflash_w4_gate_up_swiglu_rows8_kernel(
    const device uchar* gate_packed [[buffer(0)]],
    const device ushort* gate_scales [[buffer(1)]],
    const device ushort* gate_biases [[buffer(2)]],
    const device uchar* up_packed [[buffer(3)]],
    const device ushort* up_scales [[buffer(4)]],
    const device ushort* up_biases [[buffer(5)]],
    const device ushort* input [[buffer(6)]],
    device ushort* output [[buffer(7)]],
    constant uint& row_count [[buffer(8)]],
    constant uint& in_features [[buffer(9)]],
    constant uint& out_features [[buffer(10)]],
    uint grid_group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint k_part [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint K_PARTS = 2u;
    uint row_tiles = (row_count + DFLASH_LINEAR_ROW_TILE - 1u) / DFLASH_LINEAR_ROW_TILE;
    uint output_feature = grid_group % out_features;
    uint row_tile = grid_group / out_features;
    if (row_tile >= row_tiles) {
        return;
    }
    uint row_start = row_tile * DFLASH_LINEAR_ROW_TILE;
    uint groups_per_row = in_features / DFLASH_W4_GROUP_SIZE;
    uint groups_per_part = (groups_per_row + K_PARTS - 1u) / K_PARTS;
    uint first_group = k_part * groups_per_part;
    uint last_group = min(groups_per_row, first_group + groups_per_part);
    uint packed_row_start = output_feature * (in_features / 2u);
    uint scale_row_start = output_feature * groups_per_row;
    float gate_partials[DFLASH_LINEAR_ROW_TILE] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    float up_partials[DFLASH_LINEAR_ROW_TILE] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };

    for (uint group = first_group; group < last_group; group++) {
        uint packed_index = packed_row_start
            + group * (DFLASH_W4_GROUP_SIZE / 2u)
            + lane;
        uchar gate_byte = gate_packed[packed_index];
        uchar up_byte = up_packed[packed_index];
        float gate_scale = dflash_bf16_to_f32(gate_scales[scale_row_start + group]);
        float up_scale = dflash_bf16_to_f32(up_scales[scale_row_start + group]);
        float gate_bias = dflash_bf16_to_f32(gate_biases[scale_row_start + group]);
        float up_bias = dflash_bf16_to_f32(up_biases[scale_row_start + group]);
        float gate_first = float(dflash_w4_value(gate_byte, false)) * gate_scale + gate_bias;
        float gate_second = float(dflash_w4_value(gate_byte, true)) * gate_scale + gate_bias;
        float up_first = float(dflash_w4_value(up_byte, false)) * up_scale + up_bias;
        float up_second = float(dflash_w4_value(up_byte, true)) * up_scale + up_bias;
        uint input_column = group * DFLASH_W4_GROUP_SIZE + lane * 2u;
        #pragma unroll
        for (uint item = 0u; item < DFLASH_LINEAR_ROW_TILE; item++) {
            uint row = row_start + item;
            if (row < row_count) {
                uint input_start = row * in_features + input_column;
                float first_input = dflash_bf16_to_f32(input[input_start]);
                float second_input = dflash_bf16_to_f32(input[input_start + 1u]);
                gate_partials[item] += first_input * gate_first + second_input * gate_second;
                up_partials[item] += first_input * up_first + second_input * up_second;
            }
        }
    }

    threadgroup float gate_reduced[K_PARTS * DFLASH_LINEAR_ROW_TILE];
    threadgroup float up_reduced[K_PARTS * DFLASH_LINEAR_ROW_TILE];
    #pragma unroll
    for (uint item = 0u; item < DFLASH_LINEAR_ROW_TILE; item++) {
        float gate_sum = simd_sum(gate_partials[item]);
        float up_sum = simd_sum(up_partials[item]);
        if (lane == 0u) {
            gate_reduced[k_part * DFLASH_LINEAR_ROW_TILE + item] = gate_sum;
            up_reduced[k_part * DFLASH_LINEAR_ROW_TILE + item] = up_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k_part == 0u && lane < DFLASH_LINEAR_ROW_TILE) {
        uint row = row_start + lane;
        if (row < row_count) {
            float gate_bf16 = dflash_round_bf16(
                gate_reduced[lane] + gate_reduced[DFLASH_LINEAR_ROW_TILE + lane]);
            float up_bf16 = dflash_round_bf16(
                up_reduced[lane] + up_reduced[DFLASH_LINEAR_ROW_TILE + lane]);
            float activated = gate_bf16 / (1.0f + exp(-gate_bf16));
            output[row * out_features + output_feature] =
                dflash_f32_to_bf16(activated * up_bf16);
        }
    }
}

kernel void dflash_dynamic_conv_kernel(
    const device ushort* input [[buffer(0)]],
    const device ushort* dynamic [[buffer(1)]],
    const device ushort* base_kernel [[buffer(2)]],
    device ushort* output [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& sequence_length [[buffer(5)]],
    constant uint& hidden_size [[buffer(6)]],
    constant uint& stage [[buffer(7)]],
    constant uint& kernel_size [[buffer(8)]],
    constant uint& group_size [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint value_count = row_count * hidden_size;
    if (gid >= value_count) {
        return;
    }
    uint row = gid / hidden_size;
    uint hidden = gid - row * hidden_size;
    uint token = row % sequence_length;
    uint groups = hidden_size / group_size;
    uint group = hidden / group_size;
    uint dynamic_width = 2u * kernel_size * groups;
    float sum = 0.0f;
    for (uint offset = 0u; offset < kernel_size; offset++) {
        if (token < offset) {
            continue;
        }
        uint source = (row - offset) * hidden_size + hidden;
        float value = dflash_bf16_to_f32(input[source]);
        float base = dflash_bf16_to_f32(
            base_kernel[(stage * kernel_size + offset) * hidden_size + hidden]
        );
        float delta = dflash_bf16_to_f32(
            dynamic[row * dynamic_width + (stage * kernel_size + offset) * groups + group]
        );
        sum += (base + delta) * value;
    }
    output[gid] = dflash_f32_to_bf16(sum);
}

// The second dynamic-convolution stage is always followed by a residual add.
// Keep the convolved value in a register and preserve the same two BF16
// rounding points as the separate convolution and add kernels.
kernel void dflash_dynamic_conv_residual_kernel(
    const device ushort* input [[buffer(0)]],
    const device ushort* dynamic [[buffer(1)]],
    const device ushort* base_kernel [[buffer(2)]],
    const device ushort* residual [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& sequence_length [[buffer(6)]],
    constant uint& hidden_size [[buffer(7)]],
    constant uint& kernel_size [[buffer(8)]],
    constant uint& group_size [[buffer(9)]],
    uint gid [[thread_position_in_grid]]
) {
    uint value_count = row_count * hidden_size;
    if (gid >= value_count) {
        return;
    }
    uint row = gid / hidden_size;
    uint hidden = gid - row * hidden_size;
    uint token = row % sequence_length;
    uint groups = hidden_size / group_size;
    uint group = hidden / group_size;
    uint dynamic_width = 2u * kernel_size * groups;
    float sum = 0.0f;
    for (uint offset = 0u; offset < kernel_size; offset++) {
        if (token < offset) {
            continue;
        }
        uint source = (row - offset) * hidden_size + hidden;
        float value = dflash_bf16_to_f32(input[source]);
        float base = dflash_bf16_to_f32(
            base_kernel[(kernel_size + offset) * hidden_size + hidden]
        );
        float delta = dflash_bf16_to_f32(
            dynamic[row * dynamic_width + (kernel_size + offset) * groups + group]
        );
        sum += (base + delta) * value;
    }
    float convolved = dflash_round_bf16(sum);
    output[gid] = dflash_f32_to_bf16(
        dflash_bf16_to_f32(residual[gid]) + convolved
    );
}

static inline float dflash_norm_rope_value(
    const device ushort* source,
    const device ushort* norm_weight,
    uint source_start,
    uint dim,
    uint position,
    float theta,
    float inverse_rms
) {
    float normalized = dflash_round_bf16(
        dflash_bf16_to_f32(source[source_start + dim]) * inverse_rms
    );
    float value = dflash_round_bf16(
        normalized * dflash_bf16_to_f32(norm_weight[dim])
    );
    uint half_dim = DFLASH_HEAD_DIM / 2u;
    uint pair_dim = dim < half_dim ? dim + half_dim : dim - half_dim;
    float normalized_pair = dflash_round_bf16(
        dflash_bf16_to_f32(source[source_start + pair_dim]) * inverse_rms
    );
    float pair = dflash_round_bf16(
        normalized_pair * dflash_bf16_to_f32(norm_weight[pair_dim])
    );
    uint frequency_index = dim % half_dim;
    float exponent = -2.0f * float(frequency_index) / float(DFLASH_HEAD_DIM);
    float angle = float(position) * pow(theta, exponent);
    float cosine = metal::fast::cos(angle);
    float sine = metal::fast::sin(angle);
    float rotated = dim < half_dim ? -pair : pair;
    return value * cosine + rotated * sine;
}

kernel void dflash_query_norm_rope_kernel(
    const device ushort* query_projection [[buffer(0)]],
    const device ushort* norm_weight [[buffer(1)]],
    device ushort* query [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& sequence_length [[buffer(4)]],
    constant uint& position_start [[buffer(5)]],
    constant float& theta [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint group_count = row_count * DFLASH_QUERY_HEADS;
    if (group >= group_count) {
        return;
    }
    uint row = group / DFLASH_QUERY_HEADS;
    uint source_start = group * DFLASH_HEAD_DIM;
    float square_sum = 0.0f;
    for (uint dim = lane; dim < DFLASH_HEAD_DIM; dim += 32u) {
        float value = dflash_bf16_to_f32(query_projection[source_start + dim]);
        square_sum += value * value;
    }
    float inverse_rms = rsqrt(simd_sum(square_sum) / float(DFLASH_HEAD_DIM) + 1.0e-6f);
    inverse_rms = simd_broadcast_first(inverse_rms);
    uint position = position_start + (row % sequence_length);
    for (uint dim = lane; dim < DFLASH_HEAD_DIM; dim += 32u) {
        query[source_start + dim] = dflash_f32_to_bf16(dflash_norm_rope_value(
            query_projection, norm_weight, source_start, dim, position, theta, inverse_rms
        ));
    }
}

kernel void dflash_context_kv_append_kernel(
    const device ushort* key_projection [[buffer(0)]],
    const device ushort* value_projection [[buffer(1)]],
    const device ushort* norm_weight [[buffer(2)]],
    device ushort* key_cache [[buffer(3)]],
    device ushort* value_cache [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& sequence_length [[buffer(6)]],
    constant uint& capacity_tokens [[buffer(7)]],
    constant uint& position_start [[buffer(8)]],
    constant float& theta [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint group_count = row_count * DFLASH_KV_HEADS;
    if (group >= group_count) {
        return;
    }
    uint row = group / DFLASH_KV_HEADS;
    uint head = group - row * DFLASH_KV_HEADS;
    uint batch = row / sequence_length;
    uint token = row - batch * sequence_length;
    uint position = position_start + token;
    uint source_start = group * DFLASH_HEAD_DIM;
    uint cache_start = ((batch * capacity_tokens + position % capacity_tokens)
        * DFLASH_KV_HEADS + head) * DFLASH_HEAD_DIM;
    float square_sum = 0.0f;
    for (uint dim = lane; dim < DFLASH_HEAD_DIM; dim += 32u) {
        float value = dflash_bf16_to_f32(key_projection[source_start + dim]);
        square_sum += value * value;
    }
    float inverse_rms = rsqrt(simd_sum(square_sum) / float(DFLASH_HEAD_DIM) + 1.0e-6f);
    inverse_rms = simd_broadcast_first(inverse_rms);
    for (uint dim = lane; dim < DFLASH_HEAD_DIM; dim += 32u) {
        key_cache[cache_start + dim] = dflash_f32_to_bf16(dflash_norm_rope_value(
            key_projection, norm_weight, source_start, dim, position, theta, inverse_rms
        ));
        value_cache[cache_start + dim] = value_projection[source_start + dim];
    }
}

kernel void dflash_proposal_key_norm_rope_kernel(
    const device ushort* key_projection [[buffer(0)]],
    const device ushort* norm_weight [[buffer(1)]],
    device ushort* key [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& sequence_length [[buffer(4)]],
    constant uint& position_start [[buffer(5)]],
    constant float& theta [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint group_count = row_count * DFLASH_KV_HEADS;
    if (group >= group_count) {
        return;
    }
    uint row = group / DFLASH_KV_HEADS;
    uint source_start = group * DFLASH_HEAD_DIM;
    float square_sum = 0.0f;
    for (uint dim = lane; dim < DFLASH_HEAD_DIM; dim += 32u) {
        float value = dflash_bf16_to_f32(key_projection[source_start + dim]);
        square_sum += value * value;
    }
    float inverse_rms = rsqrt(simd_sum(square_sum) / float(DFLASH_HEAD_DIM) + 1.0e-6f);
    inverse_rms = simd_broadcast_first(inverse_rms);
    uint position = position_start + (row % sequence_length);
    for (uint dim = lane; dim < DFLASH_HEAD_DIM; dim += 32u) {
        key[source_start + dim] = dflash_f32_to_bf16(dflash_norm_rope_value(
            key_projection, norm_weight, source_start, dim, position, theta, inverse_rms
        ));
    }
}

kernel void dflash_online_sliding_gqa_kernel(
    const device ushort* query [[buffer(0)]],
    const device ushort* proposal_key [[buffer(1)]],
    const device ushort* proposal_value [[buffer(2)]],
    const device ushort* key_cache [[buffer(3)]],
    const device ushort* value_cache [[buffer(4)]],
    device ushort* output [[buffer(5)]],
    constant uint& row_count [[buffer(6)]],
    constant uint& sequence_length [[buffer(7)]],
    constant uint& capacity_tokens [[buffer(8)]],
    constant uint& cache_length [[buffer(9)]],
    constant uint& next_position [[buffer(10)]],
    constant uint& sliding_window [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint group_count = row_count * DFLASH_QUERY_HEADS;
    if (group >= group_count) {
        return;
    }
    uint row = group / DFLASH_QUERY_HEADS;
    uint query_head = group - row * DFLASH_QUERY_HEADS;
    uint batch = row / sequence_length;
    uint token = row - batch * sequence_length;
    uint kv_head = query_head / (DFLASH_QUERY_HEADS / DFLASH_KV_HEADS);
    uint query_start = group * DFLASH_HEAD_DIM;
    uint query_position = next_position + token;
    uint cache_start_position = next_position - cache_length;
    uint window_start = query_position + 1u > sliding_window
        ? query_position + 1u - sliding_window
        : 0u;
    uint first_context = max(cache_start_position, window_start);
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float scale = rsqrt(float(DFLASH_HEAD_DIM));

    for (uint position = first_context; position < next_position; position++) {
        uint kv_start = ((batch * capacity_tokens + position % capacity_tokens)
            * DFLASH_KV_HEADS + kv_head) * DFLASH_HEAD_DIM;
        float partial = 0.0f;
        #pragma unroll
        for (uint item = 0u; item < 4u; item++) {
            uint dim = lane + item * 32u;
            partial += dflash_bf16_to_f32(query[query_start + dim])
                * dflash_bf16_to_f32(key_cache[kv_start + dim]);
        }
        float score = simd_sum(partial) * scale;
        float next_maximum = max(running_max, score);
        float previous_scale = running_max == -INFINITY ? 0.0f : exp(running_max - next_maximum);
        float current_scale = exp(score - next_maximum);
        running_sum = running_sum * previous_scale + current_scale;
        #pragma unroll
        for (uint item = 0u; item < 4u; item++) {
            uint dim = lane + item * 32u;
            accumulator[item] = accumulator[item] * previous_scale
                + current_scale * dflash_bf16_to_f32(value_cache[kv_start + dim]);
        }
        running_max = next_maximum;
    }

    for (uint proposal = 0u; proposal < sequence_length; proposal++) {
        uint kv_start = ((batch * sequence_length + proposal) * DFLASH_KV_HEADS + kv_head)
            * DFLASH_HEAD_DIM;
        float partial = 0.0f;
        #pragma unroll
        for (uint item = 0u; item < 4u; item++) {
            uint dim = lane + item * 32u;
            partial += dflash_bf16_to_f32(query[query_start + dim])
                * dflash_bf16_to_f32(proposal_key[kv_start + dim]);
        }
        float score = simd_sum(partial) * scale;
        float next_maximum = max(running_max, score);
        float previous_scale = running_max == -INFINITY ? 0.0f : exp(running_max - next_maximum);
        float current_scale = exp(score - next_maximum);
        running_sum = running_sum * previous_scale + current_scale;
        #pragma unroll
        for (uint item = 0u; item < 4u; item++) {
            uint dim = lane + item * 32u;
            accumulator[item] = accumulator[item] * previous_scale
                + current_scale * dflash_bf16_to_f32(proposal_value[kv_start + dim]);
        }
        running_max = next_maximum;
    }

    #pragma unroll
    for (uint item = 0u; item < 4u; item++) {
        uint dim = lane + item * 32u;
        output[query_start + dim] = dflash_f32_to_bf16(accumulator[item] / running_sum);
    }
}

kernel void dflash_all_row_logits_kernel(
    const device ushort* weight [[buffer(0)]],
    const device ushort* hidden [[buffer(1)]],
    device ushort* logits [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    uint vocab [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (vocab >= DFLASH_VOCAB || row_count == 0u || row_count > DFLASH_MAX_ROWS) {
        return;
    }
    uint weight_start = vocab * DFLASH_HIDDEN;
    float partials[DFLASH_MAX_ROWS] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint hidden_index = lane; hidden_index < DFLASH_HIDDEN; hidden_index += 32u) {
        float weight_value = dflash_bf16_to_f32(weight[weight_start + hidden_index]);
        for (uint row = 0u; row < row_count; row++) {
            partials[row] += dflash_bf16_to_f32(hidden[row * DFLASH_HIDDEN + hidden_index])
                * weight_value;
        }
    }
    for (uint row = 0u; row < row_count; row++) {
        float value = simd_sum(partials[row]);
        if (lane == 0u) {
            logits[row * DFLASH_VOCAB + vocab] = dflash_f32_to_bf16(value);
        }
    }
}

kernel void dflash_top_k_kernel(
    const device ushort* logits [[buffer(0)]],
    device uint* candidate_ids [[buffer(1)]],
    device float* candidate_logits [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]
) {
    threadgroup float values[DFLASH_TOP_K_THREADS * DFLASH_TOP_K];
    threadgroup uint indices[DFLASH_TOP_K_THREADS * DFLASH_TOP_K];
    threadgroup uint cursors[DFLASH_TOP_K_THREADS];
    if (row >= row_count || threads != DFLASH_TOP_K_THREADS) {
        return;
    }

    // Each thread scans a disjoint vocabulary stripe once and keeps its local
    // top-16 ordered from best to worst. Most logits are rejected by the
    // comparison against the current local minimum, so insertion is uncommon
    // after the first few values.
    float local_values[DFLASH_TOP_K];
    uint local_indices[DFLASH_TOP_K];
    for (uint rank = 0u; rank < DFLASH_TOP_K; rank++) {
        local_values[rank] = -INFINITY;
        local_indices[rank] = 0xffffffffu;
    }
    uint row_start = row * DFLASH_VOCAB;
    for (uint vocab = tid; vocab < DFLASH_VOCAB; vocab += threads) {
        float value = dflash_bf16_to_f32(logits[row_start + vocab]);
        uint last = DFLASH_TOP_K - 1u;
        if (!dflash_candidate_is_better(
                value,
                vocab,
                local_values[last],
                local_indices[last])) {
            continue;
        }

        uint insert = last;
        while (insert > 0u && dflash_candidate_is_better(
                value,
                vocab,
                local_values[insert - 1u],
                local_indices[insert - 1u])) {
            local_values[insert] = local_values[insert - 1u];
            local_indices[insert] = local_indices[insert - 1u];
            insert--;
        }
        local_values[insert] = value;
        local_indices[insert] = vocab;
    }

    uint local_start = tid * DFLASH_TOP_K;
    for (uint rank = 0u; rank < DFLASH_TOP_K; rank++) {
        values[local_start + rank] = local_values[rank];
        indices[local_start + rank] = local_indices[rank];
    }
    cursors[tid] = 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The local lists are already sorted. A 16-step k-way merge therefore
    // produces the exact global top-16 with only 128 comparisons per step.
    if (tid == 0u) {
        for (uint rank = 0u; rank < DFLASH_TOP_K; rank++) {
            float best_value = -INFINITY;
            uint best_index = 0xffffffffu;
            uint best_thread = 0xffffffffu;
            for (uint source = 0u; source < DFLASH_TOP_K_THREADS; source++) {
                uint cursor = cursors[source];
                if (cursor >= DFLASH_TOP_K) {
                    continue;
                }
                uint position = source * DFLASH_TOP_K + cursor;
                float candidate_value = values[position];
                uint candidate_index = indices[position];
                if (dflash_candidate_is_better(
                        candidate_value,
                        candidate_index,
                        best_value,
                        best_index)) {
                    best_value = candidate_value;
                    best_index = candidate_index;
                    best_thread = source;
                }
            }
            candidate_ids[row * DFLASH_TOP_K + rank] = best_index;
            candidate_logits[row * DFLASH_TOP_K + rank] = best_value;
            if (best_thread != 0xffffffffu) {
                cursors[best_thread]++;
            }
        }
    }
}

kernel void dflash_path_selector_kernel(
    const device uint* candidate_ids [[buffer(0)]],
    const device float* unary_logits [[buffer(1)]],
    const device ushort* hidden [[buffer(2)]],
    const device ushort* predecessor_codebook [[buffer(3)]],
    const device ushort* successor_codebook [[buffer(4)]],
    device uint* output_ids [[buffer(5)]],
    constant uint& row_count [[buffer(6)]],
    constant uint& anchor_token [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    threadgroup float partials[256];
    threadgroup float scores[DFLASH_TOP_K];
    threadgroup uint predecessor;
    if (tid == 0u) {
        predecessor = anchor_token;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint candidate = tid / 16u;
    uint lane = tid % 16u;
    for (uint row = 0u; row < row_count; row++) {
        uint token = candidate_ids[row * DFLASH_TOP_K + candidate];
        float partial = 0.0f;
        for (uint rank = lane; rank < DFLASH_SELECTOR_RANK; rank += 16u) {
            partial += dflash_bf16_to_f32(
                predecessor_codebook[predecessor * DFLASH_SELECTOR_RANK + rank]
            ) * dflash_bf16_to_f32(hidden[row * DFLASH_SELECTOR_RANK + rank])
              * dflash_bf16_to_f32(
                  successor_codebook[token * DFLASH_SELECTOR_RANK + rank]
              );
        }
        partials[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            float score = unary_logits[row * DFLASH_TOP_K + candidate];
            for (uint item = 0u; item < 16u; item++) {
                score += partials[candidate * 16u + item];
            }
            scores[candidate] = score;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint best = 0u;
            for (uint item = 1u; item < DFLASH_TOP_K; item++) {
                uint best_token = candidate_ids[row * DFLASH_TOP_K + best];
                uint candidate_token = candidate_ids[row * DFLASH_TOP_K + item];
                if (scores[item] > scores[best]
                    || (scores[item] == scores[best] && candidate_token < best_token)) {
                    best = item;
                }
            }
            predecessor = candidate_ids[row * DFLASH_TOP_K + best];
            output_ids[row] = predecessor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
