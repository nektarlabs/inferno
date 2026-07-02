#include <metal_stdlib>

using namespace metal;

constant uint Q2_K_BLOCK_VALUES = 256;
constant uint Q2_K_BLOCK_BYTES = 84;
constant uint Q2_K_SCALE_BYTES = 16;
constant uint Q2_K_QUANT_BYTES = 64;
constant uint Q8_0_BLOCK_VALUES = 32;
constant uint Q8_0_BLOCK_BYTES = 34;

static inline float f16_bits_to_f32(ushort bits) {
    return float(as_type<half>(bits));
}

static inline ushort read_le_u16(const device uchar* bytes, uint offset) {
    return ushort(bytes[offset]) | (ushort(bytes[offset + 1]) << 8);
}

static inline float q2_k_block_value(const device uchar* weights, uint block_offset, uint value_index) {
    uint half_index = value_index / 128;
    uint within_half = value_index - (half_index * 128);
    uint pair = within_half / 32;
    uint within_pair = within_half - (pair * 32);
    bool upper_half_of_pair = within_pair >= 16;
    uint scale_index = block_offset + (half_index * 8) + (pair * 2) + (upper_half_of_pair ? 1 : 0);
    uint quant_index = block_offset
        + Q2_K_SCALE_BYTES
        + (half_index * 32)
        + (upper_half_of_pair ? 16 : 0)
        + (within_pair % 16);
    uint shift = pair * 2;
    uint scale_offset = block_offset + Q2_K_SCALE_BYTES + Q2_K_QUANT_BYTES;

    float d = f16_bits_to_f32(read_le_u16(weights, scale_offset));
    float min_scale = f16_bits_to_f32(read_le_u16(weights, scale_offset + 2));
    uchar scale_min = weights[scale_index];
    float scale = d * float(scale_min & 0x0f);
    float min_offset = min_scale * float(scale_min >> 4);
    float quant = float((weights[quant_index] >> shift) & 0x03);

    return (scale * quant) - min_offset;
}

static inline float q8_0_block_value(const device uchar* weights, uint block_offset, uint value_index) {
    float d = f16_bits_to_f32(read_le_u16(weights, block_offset));
    uchar raw = weights[block_offset + 2 + value_index];
    int quant = raw < 128 ? int(raw) : int(raw) - 256;
    return d * float(quant);
}

kernel void q2_k_matvec_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    constant uint& blocks_per_row [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = row_count * out_features;
    if (gid >= output_values) {
        return;
    }

    uint input_row = gid / out_features;
    uint output_feature = gid - (input_row * out_features);
    uint input_row_offset = input_row * in_features;
    float sum = 0.0f;

    for (uint block_in_row = 0; block_in_row < blocks_per_row; block_in_row++) {
        uint block_index = (output_feature * blocks_per_row) + block_in_row;
        uint block_offset = block_index * Q2_K_BLOCK_BYTES;
        uint scale_offset = block_offset + Q2_K_SCALE_BYTES + Q2_K_QUANT_BYTES;

        float d = f16_bits_to_f32(read_le_u16(weights, scale_offset));
        float min_scale = f16_bits_to_f32(read_le_u16(weights, scale_offset + 2));

        uint scale_index = block_offset;
        uint quant_offset = block_offset + Q2_K_SCALE_BYTES;
        uint local_value = 0;

        while (local_value < Q2_K_BLOCK_VALUES) {
            uint shift = 0;
            for (uint group = 0; group < 4; group++) {
                uchar scale_min = weights[scale_index];
                scale_index += 1;
                float scale = d * float(scale_min & 0x0f);
                float min_offset = min_scale * float(scale_min >> 4);

                for (uint value_index = 0; value_index < 16; value_index++) {
                    float quant = float((weights[quant_offset + value_index] >> shift) & 0x03);
                    float weight_value = (scale * quant) - min_offset;
                    uint input_index = input_row_offset
                        + (block_in_row * Q2_K_BLOCK_VALUES)
                        + local_value
                        + value_index;
                    sum += input[input_index] * weight_value;
                }
                local_value += 16;

                scale_min = weights[scale_index];
                scale_index += 1;
                scale = d * float(scale_min & 0x0f);
                min_offset = min_scale * float(scale_min >> 4);

                for (uint value_index = 0; value_index < 16; value_index++) {
                    float quant = float((weights[quant_offset + 16 + value_index] >> shift) & 0x03);
                    float weight_value = (scale * quant) - min_offset;
                    uint input_index = input_row_offset
                        + (block_in_row * Q2_K_BLOCK_VALUES)
                        + local_value
                        + value_index;
                    sum += input[input_index] * weight_value;
                }
                local_value += 16;

                shift += 2;
            }
            quant_offset += 32;
        }
    }

    output[gid] = sum;
}

kernel void q2_k_matvec_add_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    const device float* residual [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& blocks_per_row [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = row_count * out_features;
    if (gid >= output_values) {
        return;
    }

    uint input_row = gid / out_features;
    uint output_feature = gid - (input_row * out_features);
    uint input_row_offset = input_row * in_features;
    float sum = 0.0f;

    for (uint block_in_row = 0; block_in_row < blocks_per_row; block_in_row++) {
        uint block_index = (output_feature * blocks_per_row) + block_in_row;
        uint block_offset = block_index * Q2_K_BLOCK_BYTES;
        uint scale_offset = block_offset + Q2_K_SCALE_BYTES + Q2_K_QUANT_BYTES;

        float d = f16_bits_to_f32(read_le_u16(weights, scale_offset));
        float min_scale = f16_bits_to_f32(read_le_u16(weights, scale_offset + 2));

        uint scale_index = block_offset;
        uint quant_offset = block_offset + Q2_K_SCALE_BYTES;
        uint local_value = 0;

        while (local_value < Q2_K_BLOCK_VALUES) {
            uint shift = 0;
            for (uint group = 0; group < 4; group++) {
                uchar scale_min = weights[scale_index];
                scale_index += 1;
                float scale = d * float(scale_min & 0x0f);
                float min_offset = min_scale * float(scale_min >> 4);

                for (uint value_index = 0; value_index < 16; value_index++) {
                    float quant = float((weights[quant_offset + value_index] >> shift) & 0x03);
                    float weight_value = (scale * quant) - min_offset;
                    uint input_index = input_row_offset
                        + (block_in_row * Q2_K_BLOCK_VALUES)
                        + local_value
                        + value_index;
                    sum += input[input_index] * weight_value;
                }
                local_value += 16;

                scale_min = weights[scale_index];
                scale_index += 1;
                scale = d * float(scale_min & 0x0f);
                min_offset = min_scale * float(scale_min >> 4);

                for (uint value_index = 0; value_index < 16; value_index++) {
                    float quant = float((weights[quant_offset + 16 + value_index] >> shift) & 0x03);
                    float weight_value = (scale * quant) - min_offset;
                    uint input_index = input_row_offset
                        + (block_in_row * Q2_K_BLOCK_VALUES)
                        + local_value
                        + value_index;
                    sum += input[input_index] * weight_value;
                }
                local_value += 16;

                shift += 2;
            }
            quant_offset += 32;
        }
    }

    output[gid] = sum + residual[gid];
}

static inline float q2_k_row_dot(
    const device uchar* weights,
    const device float* input,
    uint input_row_offset,
    uint output_feature,
    uint blocks_per_row
) {
    float sum = 0.0f;

    for (uint block_in_row = 0; block_in_row < blocks_per_row; block_in_row++) {
        uint block_index = (output_feature * blocks_per_row) + block_in_row;
        uint block_offset = block_index * Q2_K_BLOCK_BYTES;
        uint scale_offset = block_offset + Q2_K_SCALE_BYTES + Q2_K_QUANT_BYTES;

        float d = f16_bits_to_f32(read_le_u16(weights, scale_offset));
        float min_scale = f16_bits_to_f32(read_le_u16(weights, scale_offset + 2));

        uint scale_index = block_offset;
        uint quant_offset = block_offset + Q2_K_SCALE_BYTES;
        uint local_value = 0;

        while (local_value < Q2_K_BLOCK_VALUES) {
            uint shift = 0;
            for (uint group = 0; group < 4; group++) {
                uchar scale_min = weights[scale_index];
                scale_index += 1;
                float scale = d * float(scale_min & 0x0f);
                float min_offset = min_scale * float(scale_min >> 4);

                for (uint value_index = 0; value_index < 16; value_index++) {
                    float quant = float((weights[quant_offset + value_index] >> shift) & 0x03);
                    float weight_value = (scale * quant) - min_offset;
                    uint input_index = input_row_offset
                        + (block_in_row * Q2_K_BLOCK_VALUES)
                        + local_value
                        + value_index;
                    sum += input[input_index] * weight_value;
                }
                local_value += 16;

                scale_min = weights[scale_index];
                scale_index += 1;
                scale = d * float(scale_min & 0x0f);
                min_offset = min_scale * float(scale_min >> 4);

                for (uint value_index = 0; value_index < 16; value_index++) {
                    float quant = float((weights[quant_offset + 16 + value_index] >> shift) & 0x03);
                    float weight_value = (scale * quant) - min_offset;
                    uint input_index = input_row_offset
                        + (block_in_row * Q2_K_BLOCK_VALUES)
                        + local_value
                        + value_index;
                    sum += input[input_index] * weight_value;
                }
                local_value += 16;

                shift += 2;
            }
            quant_offset += 32;
        }
    }

    return sum;
}

kernel void q2_k_gate_up_swiglu_f32_kernel(
    const device uchar* gate_weights [[buffer(0)]],
    const device uchar* up_weights [[buffer(1)]],
    const device float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& blocks_per_row [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = row_count * out_features;
    if (gid >= output_values) {
        return;
    }

    uint input_row = gid / out_features;
    uint output_feature = gid - (input_row * out_features);
    uint input_row_offset = input_row * in_features;

    float gate = q2_k_row_dot(gate_weights, input, input_row_offset, output_feature, blocks_per_row);
    float up = q2_k_row_dot(up_weights, input, input_row_offset, output_feature, blocks_per_row);
    float silu_gate = gate / (1.0f + exp(-gate));

    output[gid] = silu_gate * up;
}

kernel void q2_k_transposed_matvec_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    constant uint& blocks_per_input_row [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = row_count * out_features;
    if (gid >= output_values) {
        return;
    }

    uint input_row = gid / out_features;
    uint output_feature = gid - (input_row * out_features);
    uint output_block = output_feature / Q2_K_BLOCK_VALUES;
    uint output_value_in_block = output_feature - (output_block * Q2_K_BLOCK_VALUES);
    uint input_row_offset = input_row * in_features;
    float sum = 0.0f;

    for (uint input_feature = 0; input_feature < in_features; input_feature++) {
        uint block_index = (input_feature * blocks_per_input_row) + output_block;
        uint block_offset = block_index * Q2_K_BLOCK_BYTES;
        float weight_value = q2_k_block_value(weights, block_offset, output_value_in_block);
        sum += input[input_row_offset + input_feature] * weight_value;
    }

    output[gid] = sum;
}

kernel void q8_0_matvec_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    constant uint& blocks_per_row [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = row_count * out_features;
    if (gid >= output_values) {
        return;
    }

    uint input_row = gid / out_features;
    uint output_feature = gid - (input_row * out_features);
    uint input_row_offset = input_row * in_features;
    float sum = 0.0f;

    for (uint block_in_row = 0; block_in_row < blocks_per_row; block_in_row++) {
        uint block_index = (output_feature * blocks_per_row) + block_in_row;
        uint block_offset = block_index * Q8_0_BLOCK_BYTES;

        for (uint value_index = 0; value_index < Q8_0_BLOCK_VALUES; value_index++) {
            uint input_index = input_row_offset
                + (block_in_row * Q8_0_BLOCK_VALUES)
                + value_index;
            sum += input[input_index] * q8_0_block_value(weights, block_offset, value_index);
        }
    }

    output[gid] = sum;
}

kernel void q8_0_transposed_matvec_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    constant uint& blocks_per_input_row [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = row_count * out_features;
    if (gid >= output_values) {
        return;
    }

    uint input_row = gid / out_features;
    uint output_feature = gid - (input_row * out_features);
    uint output_block = output_feature / Q8_0_BLOCK_VALUES;
    uint output_value_in_block = output_feature - (output_block * Q8_0_BLOCK_VALUES);
    uint input_row_offset = input_row * in_features;
    float sum = 0.0f;

    for (uint input_feature = 0; input_feature < in_features; input_feature++) {
        uint block_index = (input_feature * blocks_per_input_row) + output_block;
        uint block_offset = block_index * Q8_0_BLOCK_BYTES;
        float weight_value = q8_0_block_value(weights, block_offset, output_value_in_block);
        sum += input[input_row_offset + input_feature] * weight_value;
    }

    output[gid] = sum;
}

kernel void argmax_f32_kernel(
    const device float* scores [[buffer(0)]],
    device uint* token_id [[buffer(1)]],
    device float* token_score [[buffer(2)]],
    constant uint& value_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid > 0 || value_count == 0) {
        return;
    }

    uint best_id = 0;
    float best_score = scores[0];

    for (uint index = 1; index < value_count; index++) {
        float score = scores[index];
        if (score > best_score) {
            best_id = index;
            best_score = score;
        }
    }

    token_id[0] = best_id;
    token_score[0] = best_score;
}
