#include <metal_stdlib>

using namespace metal;

constant uint Q2_K_BLOCK_VALUES = 256;
constant uint Q2_K_BLOCK_BYTES = 84;
constant uint Q2_K_SCALE_BYTES = 16;
constant uint Q2_K_QUANT_BYTES = 64;
constant uint Q2_K_SIMD_LANES = 32;
constant uint Q2_K_READY_MAX_ASSIGNMENTS = 8;
constant uint Q2_K_READY_OUTPUT_ROWS = 4;
constant uint ARGMAX_THREADS = 256;
constant uint Q8_0_BLOCK_VALUES = 32;
constant uint Q8_0_BLOCK_BYTES = 34;
constant uint Q8_0_MAX_SIMDGROUPS_PER_OUTPUT = 8;
constant uint Q8_0_BATCH_ROW_TILE = 4;
constant uint Q8_0_OUTPUT_FEATURE_TILE = 4;
constant uint Q8_0_LAGUNA_ARGMAX_ROWS = 4;
constant uint Q8_0_LAGUNA_OUTPUT_ROWS = 2;
constant uint Q8_0_LAGUNA_VALUES_PER_LANE = 8;
constant uint Q8_0_MMA_TOKEN_TILE = 32;
constant uint Q8_0_MMA_TOKEN_GROUPS = 4;
constant uint Q8_0_MMA_OUTPUT_TILE = 32;
constant uint Q8_0_MMA_K_TILE = 32;
constant uint Q8_0_MMA_THREAD_COUNT = 128;

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

static inline float q2_k_block_dot_partial(
    const device uchar* weights,
    const device float* input,
    uint input_block_offset,
    uint block_offset,
    uint simd_lane
) {
    uint scale_offset = block_offset + Q2_K_SCALE_BYTES + Q2_K_QUANT_BYTES;
    float d = f16_bits_to_f32(read_le_u16(weights, scale_offset));
    float min_scale = f16_bits_to_f32(read_le_u16(weights, scale_offset + 2));
    float sum = 0.0f;

    for (uint quant_byte_index = simd_lane; quant_byte_index < Q2_K_QUANT_BYTES; quant_byte_index += Q2_K_SIMD_LANES) {
        uchar packed = weights[block_offset + Q2_K_SCALE_BYTES + quant_byte_index];
        uint half_index = quant_byte_index / 32;
        uint within_half = quant_byte_index - (half_index * 32);
        bool upper_half_of_pair = within_half >= 16;
        uint byte_in_pair = within_half % 16;
        uint value_base = (half_index * 128)
            + (upper_half_of_pair ? 16 : 0)
            + byte_in_pair;
        uint scale_base = block_offset
            + (half_index * 8)
            + (upper_half_of_pair ? 1 : 0);

        for (uint pair = 0; pair < 4; pair++) {
            uchar scale_min = weights[scale_base + (pair * 2)];
            float scale = d * float(scale_min & 0x0f);
            float min_offset = min_scale * float(scale_min >> 4);
            float quant = float((packed >> (pair * 2)) & 0x03);
            uint value_index = value_base + (pair * 32);
            sum += input[input_block_offset + value_index] * ((scale * quant) - min_offset);
        }
    }

    return sum;
}

static inline float q8_0_block_value(const device uchar* weights, uint block_offset, uint value_index) {
    float d = f16_bits_to_f32(read_le_u16(weights, block_offset));
    uchar raw = weights[block_offset + 2 + value_index];
    int quant = raw < 128 ? int(raw) : int(raw) - 256;
    return d * float(quant);
}

// A Q8_0 block is an f16 scale followed by 32 signed quants, so block starts
// sit on 34-byte boundaries. That keeps the scale 2-byte aligned and readable
// as a `half`, which costs one load instead of two byte loads plus a shift.
static inline float q8_0_block_scale(const device uchar* weights, uint block_offset) {
    return float(*reinterpret_cast<const device half*>(weights + block_offset));
}

// The quant payload starts at an odd offset inside the block, so its alignment
// alternates between 2 and 4 bytes from block to block. `packed_char4` is
// 1-byte aligned, which makes it the widest vector load that is always legal
// here, and it turns Laguna's eight-quant lane stride into two loads instead
// of eight.
static inline float q8_0_lane_dot8(
    const device uchar* weights,
    uint quant_offset,
    thread const float* values
) {
    const device packed_char4* quants =
        reinterpret_cast<const device packed_char4*>(weights + quant_offset);
    char4 low = char4(quants[0]);
    char4 high = char4(quants[1]);
    return values[0] * float(low.x)
        + values[1] * float(low.y)
        + values[2] * float(low.z)
        + values[3] * float(low.w)
        + values[4] * float(high.x)
        + values[5] * float(high.y)
        + values[6] * float(high.z)
        + values[7] * float(high.w);
}

// Laguna reads eight consecutive activations per lane per block, and every
// such offset is a multiple of eight floats, so the 16-byte alignment `float4`
// needs always holds.
static inline void q8_0_lane_load8(
    const device float* input,
    uint value_offset,
    thread float* values
) {
    const device float4* packed =
        reinterpret_cast<const device float4*>(input + value_offset);
    float4 low = packed[0];
    float4 high = packed[1];
    values[0] = low.x;
    values[1] = low.y;
    values[2] = low.z;
    values[3] = low.w;
    values[4] = high.x;
    values[5] = high.y;
    values[6] = high.z;
    values[7] = high.w;
}

kernel void q8_0_embedding_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device uint* token_ids [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    constant uint& vocab_size [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    constant uint& blocks_per_row [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_count = token_count * hidden_size;
    if (gid >= output_count) {
        return;
    }
    uint token = gid / hidden_size;
    uint hidden = gid - token * hidden_size;
    uint token_id = token_ids[token];
    if (token_id >= vocab_size) {
        output[gid] = 0.0f;
        return;
    }
    uint block = hidden / Q8_0_BLOCK_VALUES;
    uint value = hidden - block * Q8_0_BLOCK_VALUES;
    uint block_offset = (token_id * blocks_per_row + block) * Q8_0_BLOCK_BYTES;
    output[gid] = q8_0_block_value(weights, block_offset, value);
}

kernel void q2_k_matvec_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    constant uint& blocks_per_row [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_values = row_count * out_features;
    uint output_index = gid / Q2_K_SIMD_LANES;
    if (output_index >= output_values) {
        return;
    }

    uint input_row = output_index / out_features;
    uint output_feature = output_index - (input_row * out_features);
    uint input_row_offset = input_row * in_features;
    float sum = 0.0f;

    for (uint block_in_row = 0; block_in_row < blocks_per_row; block_in_row++) {
        uint block_index = (output_feature * blocks_per_row) + block_in_row;
        uint block_offset = block_index * Q2_K_BLOCK_BYTES;
        uint input_block_offset = input_row_offset + (block_in_row * Q2_K_BLOCK_VALUES);
        sum += q2_k_block_dot_partial(weights, input, input_block_offset, block_offset, simd_lane);
    }

    float reduced_sum = simd_sum(sum);
    if (simd_lane == 0) {
        output[output_index] = reduced_sum;
    }
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
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_values = row_count * out_features;
    uint output_index = gid / Q2_K_SIMD_LANES;
    if (output_index >= output_values) {
        return;
    }

    uint input_row = output_index / out_features;
    uint output_feature = output_index - (input_row * out_features);
    uint input_row_offset = input_row * in_features;
    float sum = 0.0f;

    for (uint block_in_row = 0; block_in_row < blocks_per_row; block_in_row++) {
        uint block_index = (output_feature * blocks_per_row) + block_in_row;
        uint block_offset = block_index * Q2_K_BLOCK_BYTES;
        uint input_block_offset = input_row_offset + (block_in_row * Q2_K_BLOCK_VALUES);
        sum += q2_k_block_dot_partial(weights, input, input_block_offset, block_offset, simd_lane);
    }

    float reduced_sum = simd_sum(sum);
    if (simd_lane == 0) {
        output[output_index] = reduced_sum + residual[output_index];
    }
}

static inline float q2_k_row_dot_partial(
    const device uchar* weights,
    const device float* input,
    uint input_row_offset,
    uint output_feature,
    uint blocks_per_row,
    uint simd_lane
) {
    float sum = 0.0f;

    for (uint block_in_row = 0; block_in_row < blocks_per_row; block_in_row++) {
        uint block_index = (output_feature * blocks_per_row) + block_in_row;
        uint block_offset = block_index * Q2_K_BLOCK_BYTES;
        uint input_block_offset = input_row_offset + (block_in_row * Q2_K_BLOCK_VALUES);
        sum += q2_k_block_dot_partial(weights, input, input_block_offset, block_offset, simd_lane);
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
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_values = row_count * out_features;
    uint output_index = gid / Q2_K_SIMD_LANES;
    if (output_index >= output_values) {
        return;
    }

    uint input_row = output_index / out_features;
    uint output_feature = output_index - (input_row * out_features);
    uint input_row_offset = input_row * in_features;

    float gate_partial = q2_k_row_dot_partial(
        gate_weights,
        input,
        input_row_offset,
        output_feature,
        blocks_per_row,
        simd_lane
    );
    float up_partial = q2_k_row_dot_partial(
        up_weights,
        input,
        input_row_offset,
        output_feature,
        blocks_per_row,
        simd_lane
    );
    float gate = simd_sum(gate_partial);
    float up = simd_sum(up_partial);

    if (simd_lane == 0) {
        float silu_gate = gate / (1.0f + exp(-gate));
        output[output_index] = silu_gate * up;
    }
}

kernel void q2_k_multi_expert_gate_up_swiglu_f32_kernel(
    const device uchar* gate_weights [[buffer(0)]],
    const device uchar* up_weights [[buffer(1)]],
    const device float* input [[buffer(2)]],
    const device uint* token_indices [[buffer(3)]],
    const device uint* expert_ids [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& token_count [[buffer(6)]],
    constant uint& assignment_count [[buffer(7)]],
    constant uint& in_features [[buffer(8)]],
    constant uint& out_features [[buffer(9)]],
    constant uint& blocks_per_row [[buffer(10)]],
    constant uint& expert_stride_bytes [[buffer(11)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_values = assignment_count * out_features;
    uint output_index = gid / Q2_K_SIMD_LANES;
    if (output_index >= output_values) {
        return;
    }

    uint assignment = output_index / out_features;
    uint output_feature = output_index - (assignment * out_features);
    uint token = token_indices[assignment];
    if (token >= token_count) {
        if (simd_lane == 0) {
            output[output_index] = 0.0f;
        }
        return;
    }

    uint expert = expert_ids[assignment];
    uint input_row_offset = token * in_features;
    uint expert_base = expert * expert_stride_bytes;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    for (uint block_in_row = 0; block_in_row < blocks_per_row; block_in_row++) {
        uint block_offset = expert_base
            + ((output_feature * blocks_per_row) + block_in_row) * Q2_K_BLOCK_BYTES;
        uint input_block_offset = input_row_offset + (block_in_row * Q2_K_BLOCK_VALUES);
        gate_sum += q2_k_block_dot_partial(
            gate_weights,
            input,
            input_block_offset,
            block_offset,
            simd_lane
        );
        up_sum += q2_k_block_dot_partial(
            up_weights,
            input,
            input_block_offset,
            block_offset,
            simd_lane
        );
    }

    float gate = simd_sum(gate_sum);
    float up = simd_sum(up_sum);
    if (simd_lane == 0) {
        float silu_gate = gate / (1.0f + exp(-gate));
        output[output_index] = silu_gate * up;
    }
}

kernel void q2_k_multi_expert_matvec_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    const device uint* expert_ids [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& assignment_count [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& blocks_per_row [[buffer(7)]],
    constant uint& expert_stride_bytes [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_values = assignment_count * out_features;
    uint output_index = gid / Q2_K_SIMD_LANES;
    if (output_index >= output_values) {
        return;
    }

    uint assignment = output_index / out_features;
    uint output_feature = output_index - (assignment * out_features);
    uint expert = expert_ids[assignment];
    uint input_row_offset = assignment * in_features;
    uint expert_base = expert * expert_stride_bytes;
    float sum = 0.0f;

    for (uint block_in_row = 0; block_in_row < blocks_per_row; block_in_row++) {
        uint block_offset = expert_base
            + ((output_feature * blocks_per_row) + block_in_row) * Q2_K_BLOCK_BYTES;
        uint input_block_offset = input_row_offset + (block_in_row * Q2_K_BLOCK_VALUES);
        sum += q2_k_block_dot_partial(
            weights,
            input,
            input_block_offset,
            block_offset,
            simd_lane
        );
    }

    float reduced_sum = simd_sum(sum);
    if (simd_lane == 0) {
        output[output_index] = reduced_sum;
    }
}

static inline void q2_k_ready_group_gate_up_swiglu(
    const device uchar* gate_weights,
    const device uchar* up_weights,
    const device float* input,
    const device uint* token_indices,
    const device uint* assignment_indices,
    device float* output,
    uint assignment_start,
    uint assignment_end,
    uint token_count,
    uint assignment_count,
    uint in_features,
    uint out_features,
    uint blocks_per_row,
    uint first_output_feature,
    uint simd_lane
) {
    uint group_size = assignment_end - assignment_start;
    if (group_size == 0 || group_size > Q2_K_READY_MAX_ASSIGNMENTS) {
        return;
    }

    uint assignments[Q2_K_READY_MAX_ASSIGNMENTS];
    uint input_row_offsets[Q2_K_READY_MAX_ASSIGNMENTS];
    float gate_sums[Q2_K_READY_MAX_ASSIGNMENTS * Q2_K_READY_OUTPUT_ROWS];
    float up_sums[Q2_K_READY_MAX_ASSIGNMENTS * Q2_K_READY_OUTPUT_ROWS];
    for (uint local = 0; local < Q2_K_READY_MAX_ASSIGNMENTS; local++) {
        assignments[local] = 0;
        input_row_offsets[local] = 0;
        for (uint row = 0; row < Q2_K_READY_OUTPUT_ROWS; row++) {
            gate_sums[(local * Q2_K_READY_OUTPUT_ROWS) + row] = 0.0f;
            up_sums[(local * Q2_K_READY_OUTPUT_ROWS) + row] = 0.0f;
        }
    }
    for (uint local = 0; local < group_size; local++) {
        uint assignment = assignment_indices[assignment_start + local];
        if (assignment >= assignment_count) {
            return;
        }
        uint token = token_indices[assignment];
        if (token >= token_count) {
            return;
        }
        assignments[local] = assignment;
        input_row_offsets[local] = token * in_features;
    }

    for (uint block_in_row = 0; block_in_row < blocks_per_row; block_in_row++) {
        uint block_offsets[Q2_K_READY_OUTPUT_ROWS];
        float gate_ds[Q2_K_READY_OUTPUT_ROWS];
        float gate_min_scales[Q2_K_READY_OUTPUT_ROWS];
        float up_ds[Q2_K_READY_OUTPUT_ROWS];
        float up_min_scales[Q2_K_READY_OUTPUT_ROWS];
        for (uint row = 0; row < Q2_K_READY_OUTPUT_ROWS; row++) {
            uint output_feature = first_output_feature + row;
            if (output_feature >= out_features) {
                block_offsets[row] = 0;
                gate_ds[row] = 0.0f;
                gate_min_scales[row] = 0.0f;
                up_ds[row] = 0.0f;
                up_min_scales[row] = 0.0f;
                continue;
            }
            uint block_offset =
                ((output_feature * blocks_per_row) + block_in_row) * Q2_K_BLOCK_BYTES;
            uint scale_offset = block_offset + Q2_K_SCALE_BYTES + Q2_K_QUANT_BYTES;
            block_offsets[row] = block_offset;
            gate_ds[row] = f16_bits_to_f32(read_le_u16(gate_weights, scale_offset));
            gate_min_scales[row] =
                f16_bits_to_f32(read_le_u16(gate_weights, scale_offset + 2));
            up_ds[row] = f16_bits_to_f32(read_le_u16(up_weights, scale_offset));
            up_min_scales[row] =
                f16_bits_to_f32(read_le_u16(up_weights, scale_offset + 2));
        }

        for (uint quant_byte_index = simd_lane; quant_byte_index < Q2_K_QUANT_BYTES; quant_byte_index += Q2_K_SIMD_LANES) {
            uint half_index = quant_byte_index / 32;
            uint within_half = quant_byte_index - (half_index * 32);
            bool upper_half_of_pair = within_half >= 16;
            uint byte_in_pair = within_half % 16;
            uint value_base = (half_index * 128)
                + (upper_half_of_pair ? 16 : 0)
                + byte_in_pair;

            for (uint pair = 0; pair < 4; pair++) {
                uint value_index = value_base + (pair * 32);
                uint block_value_offset = (block_in_row * Q2_K_BLOCK_VALUES) + value_index;
                float input_values[Q2_K_READY_MAX_ASSIGNMENTS];
                for (uint local = 0; local < group_size; local++) {
                    input_values[local] = input[input_row_offsets[local] + block_value_offset];
                }
                for (uint row = 0; row < Q2_K_READY_OUTPUT_ROWS; row++) {
                    if (first_output_feature + row >= out_features) {
                        continue;
                    }
                    uint block_offset = block_offsets[row];
                    uint scale_base = block_offset
                        + (half_index * 8)
                        + (upper_half_of_pair ? 1 : 0);
                    uchar gate_packed =
                        gate_weights[block_offset + Q2_K_SCALE_BYTES + quant_byte_index];
                    uchar up_packed =
                        up_weights[block_offset + Q2_K_SCALE_BYTES + quant_byte_index];
                    uchar gate_scale_min = gate_weights[scale_base + (pair * 2)];
                    uchar up_scale_min = up_weights[scale_base + (pair * 2)];
                    float gate_weight = (gate_ds[row] * float(gate_scale_min & 0x0f)
                        * float((gate_packed >> (pair * 2)) & 0x03))
                        - (gate_min_scales[row] * float(gate_scale_min >> 4));
                    float up_weight = (up_ds[row] * float(up_scale_min & 0x0f)
                        * float((up_packed >> (pair * 2)) & 0x03))
                        - (up_min_scales[row] * float(up_scale_min >> 4));
                    for (uint local = 0; local < group_size; local++) {
                        uint sum_index = (local * Q2_K_READY_OUTPUT_ROWS) + row;
                        gate_sums[sum_index] += input_values[local] * gate_weight;
                        up_sums[sum_index] += input_values[local] * up_weight;
                    }
                }
            }
        }
    }

    for (uint local = 0; local < group_size; local++) {
        for (uint row = 0; row < Q2_K_READY_OUTPUT_ROWS; row++) {
            uint output_feature = first_output_feature + row;
            if (output_feature >= out_features) {
                continue;
            }
            uint sum_index = (local * Q2_K_READY_OUTPUT_ROWS) + row;
            float gate = simd_sum(gate_sums[sum_index]);
            float up = simd_sum(up_sums[sum_index]);
            if (simd_lane == 0) {
                float silu_gate = gate / (1.0f + exp(-gate));
                output[(assignments[local] * out_features) + output_feature] = silu_gate * up;
            }
        }
    }
}

static inline void q2_k_ready_group_matvec(
    const device uchar* weights,
    const device float* input,
    const device uint* assignment_indices,
    device float* output,
    uint assignment_start,
    uint assignment_end,
    uint assignment_count,
    uint in_features,
    uint out_features,
    uint blocks_per_row,
    uint first_output_feature,
    uint simd_lane
) {
    uint group_size = assignment_end - assignment_start;
    if (group_size == 0 || group_size > Q2_K_READY_MAX_ASSIGNMENTS) {
        return;
    }

    uint assignments[Q2_K_READY_MAX_ASSIGNMENTS];
    uint input_row_offsets[Q2_K_READY_MAX_ASSIGNMENTS];
    float sums[Q2_K_READY_MAX_ASSIGNMENTS * Q2_K_READY_OUTPUT_ROWS];
    for (uint local = 0; local < Q2_K_READY_MAX_ASSIGNMENTS; local++) {
        assignments[local] = 0;
        input_row_offsets[local] = 0;
        for (uint row = 0; row < Q2_K_READY_OUTPUT_ROWS; row++) {
            sums[(local * Q2_K_READY_OUTPUT_ROWS) + row] = 0.0f;
        }
    }
    for (uint local = 0; local < group_size; local++) {
        uint assignment = assignment_indices[assignment_start + local];
        if (assignment >= assignment_count) {
            return;
        }
        assignments[local] = assignment;
        input_row_offsets[local] = assignment * in_features;
    }

    for (uint block_in_row = 0; block_in_row < blocks_per_row; block_in_row++) {
        uint block_offsets[Q2_K_READY_OUTPUT_ROWS];
        float ds[Q2_K_READY_OUTPUT_ROWS];
        float min_scales[Q2_K_READY_OUTPUT_ROWS];
        for (uint row = 0; row < Q2_K_READY_OUTPUT_ROWS; row++) {
            uint output_feature = first_output_feature + row;
            if (output_feature >= out_features) {
                block_offsets[row] = 0;
                ds[row] = 0.0f;
                min_scales[row] = 0.0f;
                continue;
            }
            uint block_offset =
                ((output_feature * blocks_per_row) + block_in_row) * Q2_K_BLOCK_BYTES;
            uint scale_offset = block_offset + Q2_K_SCALE_BYTES + Q2_K_QUANT_BYTES;
            block_offsets[row] = block_offset;
            ds[row] = f16_bits_to_f32(read_le_u16(weights, scale_offset));
            min_scales[row] = f16_bits_to_f32(read_le_u16(weights, scale_offset + 2));
        }

        for (uint quant_byte_index = simd_lane; quant_byte_index < Q2_K_QUANT_BYTES; quant_byte_index += Q2_K_SIMD_LANES) {
            uint half_index = quant_byte_index / 32;
            uint within_half = quant_byte_index - (half_index * 32);
            bool upper_half_of_pair = within_half >= 16;
            uint byte_in_pair = within_half % 16;
            uint value_base = (half_index * 128)
                + (upper_half_of_pair ? 16 : 0)
                + byte_in_pair;

            for (uint pair = 0; pair < 4; pair++) {
                uint value_index = value_base + (pair * 32);
                uint block_value_offset = (block_in_row * Q2_K_BLOCK_VALUES) + value_index;
                float input_values[Q2_K_READY_MAX_ASSIGNMENTS];
                for (uint local = 0; local < group_size; local++) {
                    input_values[local] = input[input_row_offsets[local] + block_value_offset];
                }
                for (uint row = 0; row < Q2_K_READY_OUTPUT_ROWS; row++) {
                    if (first_output_feature + row >= out_features) {
                        continue;
                    }
                    uint block_offset = block_offsets[row];
                    uint scale_base = block_offset
                        + (half_index * 8)
                        + (upper_half_of_pair ? 1 : 0);
                    uchar packed = weights[block_offset + Q2_K_SCALE_BYTES + quant_byte_index];
                    uchar scale_min = weights[scale_base + (pair * 2)];
                    float weight = (ds[row] * float(scale_min & 0x0f)
                        * float((packed >> (pair * 2)) & 0x03))
                        - (min_scales[row] * float(scale_min >> 4));
                    for (uint local = 0; local < group_size; local++) {
                        sums[(local * Q2_K_READY_OUTPUT_ROWS) + row] +=
                            input_values[local] * weight;
                    }
                }
            }
        }
    }

    for (uint local = 0; local < group_size; local++) {
        for (uint row = 0; row < Q2_K_READY_OUTPUT_ROWS; row++) {
            uint output_feature = first_output_feature + row;
            if (output_feature >= out_features) {
                continue;
            }
            float sum = simd_sum(sums[(local * Q2_K_READY_OUTPUT_ROWS) + row]);
            if (simd_lane == 0) {
                output[(assignments[local] * out_features) + output_feature] = sum;
            }
        }
    }
}

static inline void q2_k_tiled_four_row_dot(
    const device uchar* weights,
    const device float* input,
    uint input_row_offset,
    uint first_output_feature,
    uint out_features,
    uint blocks_per_row,
    uint simd_lane,
    thread float sums[Q2_K_READY_OUTPUT_ROWS]
) {
    for (uint row = 0; row < Q2_K_READY_OUTPUT_ROWS; row++) {
        sums[row] = 0.0f;
    }

    uint block_lane = simd_lane / 8;
    uint lane_in_block = simd_lane % 8;
    uint quant_half = lane_in_block / 4;
    uint quant_quarter = lane_in_block % 4;
    uint scale_shift = quant_quarter / 2;

    for (uint block = block_lane; block < blocks_per_row; block += 4) {
        uint input_base = input_row_offset
            + (block * Q2_K_BLOCK_VALUES)
            + (128 * quant_half)
            + (8 * quant_quarter);
        float values[32];
        float4 value_sums = float4(0.0f);
        for (uint index = 0; index < 8; index++) {
            values[index] = input[input_base + index];
            values[index + 8] = input[input_base + index + 32];
            values[index + 16] = input[input_base + index + 64];
            values[index + 24] = input[input_base + index + 96];
            value_sums[0] += values[index];
            value_sums[1] += values[index + 8];
            value_sums[2] += values[index + 16];
            value_sums[3] += values[index + 24];
        }

        for (uint row = 0; row < Q2_K_READY_OUTPUT_ROWS; row++) {
            uint output_feature = first_output_feature + row;
            if (output_feature >= out_features) {
                continue;
            }
            uint block_offset = ((output_feature * blocks_per_row) + block) * Q2_K_BLOCK_BYTES;
            uint scale_base = block_offset + (8 * quant_half) + scale_shift;
            uint quant_base = block_offset
                + Q2_K_SCALE_BYTES
                + (32 * quant_half)
                + (8 * quant_quarter);
            float4 low = float4(0.0f);
            float4 high = float4(0.0f);
            for (uint pair = 0; pair < 4; pair++) {
                uint value_index = pair * 2;
                ushort packed = read_le_u16(weights, quant_base + (pair * 2));
                low[0] += values[value_index] * float(packed & 0x0003);
                high[0] += values[value_index + 1] * float(packed & 0x0300);
                low[1] += values[value_index + 8] * float(packed & 0x000c);
                high[1] += values[value_index + 9] * float(packed & 0x0c00);
                low[2] += values[value_index + 16] * float(packed & 0x0030);
                high[2] += values[value_index + 17] * float(packed & 0x3000);
                low[3] += values[value_index + 24] * float(packed & 0x00c0);
                high[3] += values[value_index + 25] * float(packed & 0xc000);
            }

            float d = f16_bits_to_f32(read_le_u16(
                weights,
                block_offset + Q2_K_SCALE_BYTES + Q2_K_QUANT_BYTES
            ));
            float dmin = f16_bits_to_f32(read_le_u16(
                weights,
                block_offset + Q2_K_SCALE_BYTES + Q2_K_QUANT_BYTES + 2
            )) / 16.0f;
            float4 scaled = low + (high / 256.0f);
            uchar4 scales = uchar4(
                weights[scale_base],
                weights[scale_base + 2],
                weights[scale_base + 4],
                weights[scale_base + 6]
            );
            sums[row] += d * (
                scaled[0] * float(scales[0] & 0x0f)
                + scaled[1] * float(scales[1] & 0x0f) / 4.0f
                + scaled[2] * float(scales[2] & 0x0f) / 16.0f
                + scaled[3] * float(scales[3] & 0x0f) / 64.0f
            ) - dmin * (
                value_sums[0] * float(scales[0] & 0xf0)
                + value_sums[1] * float(scales[1] & 0xf0)
                + value_sums[2] * float(scales[2] & 0xf0)
                + value_sums[3] * float(scales[3] & 0xf0)
            );
        }
    }
}

static inline void q2_k_ready_tiled_gate_up_swiglu(
    const device uchar* gate_weights,
    const device uchar* up_weights,
    const device float* input,
    const device uint* token_indices,
    const device uint* assignment_indices,
    device float* output,
    uint assignment_start,
    uint assignment_end,
    uint token_count,
    uint assignment_count,
    uint in_features,
    uint out_features,
    uint blocks_per_row,
    uint first_output_feature,
    uint simd_lane
) {
    if (assignment_end - assignment_start != 1) {
        return;
    }
    uint assignment = assignment_indices[assignment_start];
    if (assignment >= assignment_count) {
        return;
    }
    uint token = token_indices[assignment];
    if (token >= token_count) {
        return;
    }

    float gate_sums[Q2_K_READY_OUTPUT_ROWS];
    float up_sums[Q2_K_READY_OUTPUT_ROWS];
    q2_k_tiled_four_row_dot(
        gate_weights,
        input,
        token * in_features,
        first_output_feature,
        out_features,
        blocks_per_row,
        simd_lane,
        gate_sums
    );
    q2_k_tiled_four_row_dot(
        up_weights,
        input,
        token * in_features,
        first_output_feature,
        out_features,
        blocks_per_row,
        simd_lane,
        up_sums
    );
    for (uint row = 0; row < Q2_K_READY_OUTPUT_ROWS; row++) {
        uint output_feature = first_output_feature + row;
        if (output_feature >= out_features) {
            continue;
        }
        float gate = simd_sum(gate_sums[row]);
        float up = simd_sum(up_sums[row]);
        if (simd_lane == 0) {
            float silu_gate = gate / (1.0f + exp(-gate));
            output[(assignment * out_features) + output_feature] = silu_gate * up;
        }
    }
}

static inline void q2_k_ready_tiled_matvec(
    const device uchar* weights,
    const device float* input,
    const device uint* assignment_indices,
    device float* output,
    uint assignment_start,
    uint assignment_end,
    uint assignment_count,
    uint in_features,
    uint out_features,
    uint blocks_per_row,
    uint first_output_feature,
    uint simd_lane
) {
    if (assignment_end - assignment_start != 1) {
        return;
    }
    uint assignment = assignment_indices[assignment_start];
    if (assignment >= assignment_count) {
        return;
    }
    float sums[Q2_K_READY_OUTPUT_ROWS];
    q2_k_tiled_four_row_dot(
        weights,
        input,
        assignment * in_features,
        first_output_feature,
        out_features,
        blocks_per_row,
        simd_lane,
        sums
    );
    for (uint row = 0; row < Q2_K_READY_OUTPUT_ROWS; row++) {
        uint output_feature = first_output_feature + row;
        if (output_feature >= out_features) {
            continue;
        }
        float sum = simd_sum(sums[row]);
        if (simd_lane == 0) {
            output[(assignment * out_features) + output_feature] = sum;
        }
    }
}

kernel void q2_k_ready_gate_up_swiglu_f32_kernel(
    const device ulong* gate_addresses [[buffer(0)]],
    const device ulong* up_addresses [[buffer(1)]],
    const device float* input [[buffer(2)]],
    const device uint* token_indices [[buffer(3)]],
    const device uint* assignment_indices [[buffer(4)]],
    const device uint* group_offsets [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& token_count [[buffer(7)]],
    constant uint& assignment_count [[buffer(8)]],
    constant uint& ready_group_count [[buffer(9)]],
    constant uint& in_features [[buffer(10)]],
    constant uint& out_features [[buffer(11)]],
    constant uint& blocks_per_row [[buffer(12)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_tiles = (out_features + Q2_K_READY_OUTPUT_ROWS - 1) / Q2_K_READY_OUTPUT_ROWS;
    uint group_output_index = gid / Q2_K_SIMD_LANES;
    if (group_output_index >= ready_group_count * output_tiles) {
        return;
    }
    uint group = group_output_index / output_tiles;
    uint first_output_feature =
        (group_output_index - (group * output_tiles)) * Q2_K_READY_OUTPUT_ROWS;
    if (gate_addresses[group] == 0 || up_addresses[group] == 0) {
        return;
    }
    const device uchar* gate_weights =
        reinterpret_cast<device const uchar*>(gate_addresses[group]);
    const device uchar* up_weights =
        reinterpret_cast<device const uchar*>(up_addresses[group]);
    uint assignment_start = group_offsets[group];
    uint assignment_end = group_offsets[group + 1];
    if (assignment_end - assignment_start == 1) {
        q2_k_ready_tiled_gate_up_swiglu(
            gate_weights,
            up_weights,
            input,
            token_indices,
            assignment_indices,
            output,
            assignment_start,
            assignment_end,
            token_count,
            assignment_count,
            in_features,
            out_features,
            blocks_per_row,
            first_output_feature,
            simd_lane
        );
    } else {
        q2_k_ready_group_gate_up_swiglu(
            gate_weights,
            up_weights,
            input,
            token_indices,
            assignment_indices,
            output,
            assignment_start,
            assignment_end,
            token_count,
            assignment_count,
            in_features,
            out_features,
            blocks_per_row,
            first_output_feature,
            simd_lane
        );
    }
}

kernel void q2_k_ready_matvec_f32_kernel(
    const device ulong* weight_addresses [[buffer(0)]],
    const device float* input [[buffer(1)]],
    const device uint* assignment_indices [[buffer(2)]],
    const device uint* group_offsets [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& assignment_count [[buffer(5)]],
    constant uint& ready_group_count [[buffer(6)]],
    constant uint& in_features [[buffer(7)]],
    constant uint& out_features [[buffer(8)]],
    constant uint& blocks_per_row [[buffer(9)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_tiles = (out_features + Q2_K_READY_OUTPUT_ROWS - 1) / Q2_K_READY_OUTPUT_ROWS;
    uint group_output_index = gid / Q2_K_SIMD_LANES;
    if (group_output_index >= ready_group_count * output_tiles) {
        return;
    }
    uint group = group_output_index / output_tiles;
    uint first_output_feature =
        (group_output_index - (group * output_tiles)) * Q2_K_READY_OUTPUT_ROWS;
    if (weight_addresses[group] == 0) {
        return;
    }
    const device uchar* weights =
        reinterpret_cast<device const uchar*>(weight_addresses[group]);
    uint assignment_start = group_offsets[group];
    uint assignment_end = group_offsets[group + 1];
    if (assignment_end - assignment_start == 1) {
        q2_k_ready_tiled_matvec(
            weights,
            input,
            assignment_indices,
            output,
            assignment_start,
            assignment_end,
            assignment_count,
            in_features,
            out_features,
            blocks_per_row,
            first_output_feature,
            simd_lane
        );
    } else {
        q2_k_ready_group_matvec(
            weights,
            input,
            assignment_indices,
            output,
            assignment_start,
            assignment_end,
            assignment_count,
            in_features,
            out_features,
            blocks_per_row,
            first_output_feature,
            simd_lane
        );
    }
}

kernel void q2_k_ready_slot_gate_up_swiglu_f32_kernel(
    const device uchar* gate_weights [[buffer(0)]],
    const device uchar* up_weights [[buffer(1)]],
    const device float* input [[buffer(2)]],
    const device uint* token_indices [[buffer(3)]],
    const device uint* assignment_indices [[buffer(4)]],
    const device uint* group_offsets [[buffer(5)]],
    const device uint* slot_indices [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant uint& token_count [[buffer(8)]],
    constant uint& assignment_count [[buffer(9)]],
    constant uint& ready_group_count [[buffer(10)]],
    constant uint& in_features [[buffer(11)]],
    constant uint& out_features [[buffer(12)]],
    constant uint& blocks_per_row [[buffer(13)]],
    constant uint& expert_stride_bytes [[buffer(14)]],
    constant uint& slot_count [[buffer(15)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_tiles = (out_features + Q2_K_READY_OUTPUT_ROWS - 1) / Q2_K_READY_OUTPUT_ROWS;
    uint group_output_index = gid / Q2_K_SIMD_LANES;
    if (group_output_index >= ready_group_count * output_tiles) {
        return;
    }
    uint group = group_output_index / output_tiles;
    uint first_output_feature =
        (group_output_index - (group * output_tiles)) * Q2_K_READY_OUTPUT_ROWS;
    uint slot = slot_indices[group];
    if (slot >= slot_count) {
        return;
    }
    uint expert_base = slot * expert_stride_bytes;
    uint assignment_start = group_offsets[group];
    uint assignment_end = group_offsets[group + 1];
    if (assignment_end - assignment_start == 1) {
        q2_k_ready_tiled_gate_up_swiglu(
            gate_weights + expert_base,
            up_weights + expert_base,
            input,
            token_indices,
            assignment_indices,
            output,
            assignment_start,
            assignment_end,
            token_count,
            assignment_count,
            in_features,
            out_features,
            blocks_per_row,
            first_output_feature,
            simd_lane
        );
    } else {
        q2_k_ready_group_gate_up_swiglu(
            gate_weights + expert_base,
            up_weights + expert_base,
            input,
            token_indices,
            assignment_indices,
            output,
            assignment_start,
            assignment_end,
            token_count,
            assignment_count,
            in_features,
            out_features,
            blocks_per_row,
            first_output_feature,
            simd_lane
        );
    }
}

kernel void q2_k_ready_slot_matvec_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    const device uint* assignment_indices [[buffer(2)]],
    const device uint* group_offsets [[buffer(3)]],
    const device uint* slot_indices [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& assignment_count [[buffer(6)]],
    constant uint& ready_group_count [[buffer(7)]],
    constant uint& in_features [[buffer(8)]],
    constant uint& out_features [[buffer(9)]],
    constant uint& blocks_per_row [[buffer(10)]],
    constant uint& expert_stride_bytes [[buffer(11)]],
    constant uint& slot_count [[buffer(12)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_tiles = (out_features + Q2_K_READY_OUTPUT_ROWS - 1) / Q2_K_READY_OUTPUT_ROWS;
    uint group_output_index = gid / Q2_K_SIMD_LANES;
    if (group_output_index >= ready_group_count * output_tiles) {
        return;
    }
    uint group = group_output_index / output_tiles;
    uint first_output_feature =
        (group_output_index - (group * output_tiles)) * Q2_K_READY_OUTPUT_ROWS;
    uint slot = slot_indices[group];
    if (slot >= slot_count) {
        return;
    }
    uint assignment_start = group_offsets[group];
    uint assignment_end = group_offsets[group + 1];
    if (assignment_end - assignment_start == 1) {
        q2_k_ready_tiled_matvec(
            weights + (slot * expert_stride_bytes),
            input,
            assignment_indices,
            output,
            assignment_start,
            assignment_end,
            assignment_count,
            in_features,
            out_features,
            blocks_per_row,
            first_output_feature,
            simd_lane
        );
    } else {
        q2_k_ready_group_matvec(
            weights + (slot * expert_stride_bytes),
            input,
            assignment_indices,
            output,
            assignment_start,
            assignment_end,
            assignment_count,
            in_features,
            out_features,
            blocks_per_row,
            first_output_feature,
            simd_lane
        );
    }
}

kernel void q2_k_transposed_matvec_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    constant uint& blocks_per_input_row [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_values = row_count * out_features;
    uint output_index = gid / Q2_K_SIMD_LANES;
    if (output_index >= output_values) {
        return;
    }

    uint input_row = output_index / out_features;
    uint output_feature = output_index - (input_row * out_features);
    uint output_block = output_feature / Q2_K_BLOCK_VALUES;
    uint output_value_in_block = output_feature - (output_block * Q2_K_BLOCK_VALUES);
    uint input_row_offset = input_row * in_features;
    float sum = 0.0f;

    for (uint input_feature = simd_lane; input_feature < in_features; input_feature += Q2_K_SIMD_LANES) {
        uint block_index = (input_feature * blocks_per_input_row) + output_block;
        uint block_offset = block_index * Q2_K_BLOCK_BYTES;
        float weight_value = q2_k_block_value(weights, block_offset, output_value_in_block);
        sum += input[input_row_offset + input_feature] * weight_value;
    }

    float reduced_sum = simd_sum(sum);
    if (simd_lane == 0) {
        output[output_index] = reduced_sum;
    }
}

kernel void q8_0_matvec_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    constant uint& blocks_per_row [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_values = row_count * out_features;
    uint output_index = gid / Q2_K_SIMD_LANES;
    if (output_index >= output_values) {
        return;
    }

    uint input_row = output_index / out_features;
    uint output_feature = output_index - (input_row * out_features);
    uint input_row_offset = input_row * in_features;
    float sum = 0.0f;

    for (uint block_in_row = 0; block_in_row < blocks_per_row; block_in_row++) {
        uint block_index = (output_feature * blocks_per_row) + block_in_row;
        uint block_offset = block_index * Q8_0_BLOCK_BYTES;
        uint input_index = input_row_offset
            + (block_in_row * Q8_0_BLOCK_VALUES)
            + simd_lane;
        sum += input[input_index] * q8_0_block_value(weights, block_offset, simd_lane);
    }

    float reduced_sum = simd_sum(sum);
    if (simd_lane == 0) {
        output[output_index] = reduced_sum;
    }
}

kernel void q8_0_matvec_tiled_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    constant uint& blocks_per_row [[buffer(6)]],
    constant uint& simdgroups_per_output [[buffer(7)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint output_values = row_count * out_features;
    uint output_index = threadgroup_position.x;
    if (output_index >= output_values || simdgroup_index >= simdgroups_per_output) {
        return;
    }

    uint input_row = output_index / out_features;
    uint output_feature = output_index - (input_row * out_features);
    uint input_row_offset = input_row * in_features;
    float sum = 0.0f;

    for (uint block_in_row = simdgroup_index;
         block_in_row < blocks_per_row;
         block_in_row += simdgroups_per_output) {
        uint block_index = (output_feature * blocks_per_row) + block_in_row;
        uint block_offset = block_index * Q8_0_BLOCK_BYTES;
        uint input_index = input_row_offset
            + (block_in_row * Q8_0_BLOCK_VALUES)
            + simd_lane;
        sum += input[input_index] * q8_0_block_value(weights, block_offset, simd_lane);
    }

    threadgroup float partial_sums[Q8_0_MAX_SIMDGROUPS_PER_OUTPUT];
    float simdgroup_sum = simd_sum(sum);
    if (simd_lane == 0) {
        partial_sums[simdgroup_index] = simdgroup_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0) {
        float partial = simd_lane < simdgroups_per_output ? partial_sums[simd_lane] : 0.0f;
        float reduced_sum = simd_sum(partial);
        if (simd_lane == 0) {
            output[output_index] = reduced_sum;
        }
    }
}

// Decode-time output projection. Four adjacent vocabulary rows consume the
// same hidden-state vector, so load each activation once and apply it to four
// independent Q8_0 rows. Each row keeps the same two-stage SIMD reduction as
// q8_0_matvec_tiled_f32_kernel, preserving its arithmetic order.
kernel void q8_0_matvec_output4_tiled_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    constant uint& blocks_per_row [[buffer(6)]],
    constant uint& simdgroups_per_output [[buffer(7)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint tiles_per_input_row = (out_features + Q8_0_OUTPUT_FEATURE_TILE - 1)
        / Q8_0_OUTPUT_FEATURE_TILE;
    uint input_row = threadgroup_position.x / tiles_per_input_row;
    uint output_tile = threadgroup_position.x - (input_row * tiles_per_input_row);
    uint output_feature_start = output_tile * Q8_0_OUTPUT_FEATURE_TILE;
    if (input_row >= row_count
        || output_feature_start >= out_features
        || simdgroup_index >= simdgroups_per_output) {
        return;
    }
    uint active_features = min(Q8_0_OUTPUT_FEATURE_TILE, out_features - output_feature_start);
    uint input_row_offset = input_row * in_features;
    float sums[Q8_0_OUTPUT_FEATURE_TILE];
    for (uint feature = 0; feature < Q8_0_OUTPUT_FEATURE_TILE; feature++) {
        sums[feature] = 0.0f;
    }

    for (uint block_in_row = simdgroup_index;
         block_in_row < blocks_per_row;
         block_in_row += simdgroups_per_output) {
        uint input_index = input_row_offset
            + (block_in_row * Q8_0_BLOCK_VALUES)
            + simd_lane;
        float input_value = input[input_index];
        for (uint feature = 0; feature < active_features; feature++) {
            uint output_feature = output_feature_start + feature;
            uint block_index = (output_feature * blocks_per_row) + block_in_row;
            uint block_offset = block_index * Q8_0_BLOCK_BYTES;
            sums[feature] += input_value
                * q8_0_block_value(weights, block_offset, simd_lane);
        }
    }

    threadgroup float partial_sums[
        Q8_0_OUTPUT_FEATURE_TILE * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT
    ];
    for (uint feature = 0; feature < active_features; feature++) {
        float simdgroup_sum = simd_sum(sums[feature]);
        if (simd_lane == 0) {
            partial_sums[
                (feature * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT) + simdgroup_index
            ] = simdgroup_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0) {
        for (uint feature = 0; feature < active_features; feature++) {
            float partial = simd_lane < simdgroups_per_output
                ? partial_sums[
                    (feature * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT) + simd_lane
                ]
                : 0.0f;
            float reduced_sum = simd_sum(partial);
            if (simd_lane == 0) {
                output[(input_row * out_features) + output_feature_start + feature]
                    = reduced_sum;
            }
        }
    }
}

kernel void laguna_q8_0_matvec_output4_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    constant uint& blocks_per_row [[buffer(6)]],
    constant uint& simdgroups_per_output [[buffer(7)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint row_groups_per_input =
        (out_features + Q8_0_OUTPUT_FEATURE_TILE - 1u)
        / Q8_0_OUTPUT_FEATURE_TILE;
    uint input_row = threadgroup_position.x / row_groups_per_input;
    uint row_group = threadgroup_position.x
        - input_row * row_groups_per_input;
    uint output_row = row_group * Q8_0_OUTPUT_FEATURE_TILE;
    if (input_row >= row_count
        || output_row >= out_features
        || simdgroup_index >= simdgroups_per_output) {
        return;
    }

    ushort block_lane = simd_lane / 4u;
    ushort value_lane = simd_lane % 4u;
    uint first_block = uint(simdgroup_index) * Q8_0_LAGUNA_VALUES_PER_LANE
        + uint(block_lane);
    uint block_step =
        simdgroups_per_output * Q8_0_LAGUNA_VALUES_PER_LANE;
    uint input_row_offset = input_row * in_features;
    float4 sums = float4(0.0f);

    for (uint block = first_block; block < blocks_per_row; block += block_step) {
        uint input_offset = input_row_offset
            + block * Q8_0_BLOCK_VALUES
            + uint(value_lane) * Q8_0_LAGUNA_VALUES_PER_LANE;
        float values[Q8_0_LAGUNA_VALUES_PER_LANE];
        q8_0_lane_load8(input, input_offset, values);
        uint lane_quant_offset =
            2u + uint(value_lane) * Q8_0_LAGUNA_VALUES_PER_LANE;

        for (uint row = 0u; row < Q8_0_OUTPUT_FEATURE_TILE; row++) {
            uint feature = output_row + row;
            if (feature >= out_features) {
                continue;
            }
            uint block_offset =
                (feature * blocks_per_row + block) * Q8_0_BLOCK_BYTES;
            sums[row] +=
                q8_0_lane_dot8(weights, block_offset + lane_quant_offset, values)
                * q8_0_block_scale(weights, block_offset);
        }
    }

    threadgroup float partial[
        Q8_0_OUTPUT_FEATURE_TILE * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT
    ];
    for (uint row = 0u; row < Q8_0_OUTPUT_FEATURE_TILE; row++) {
        float value = simd_sum(sums[row]);
        if (simd_lane == 0u) {
            partial[
                row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT + uint(simdgroup_index)
            ] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0u) {
        for (uint row = 0u; row < Q8_0_OUTPUT_FEATURE_TILE; row++) {
            uint feature = output_row + row;
            float lane = simd_lane < simdgroups_per_output
                ? partial[row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT + simd_lane]
                : 0.0f;
            float value = simd_sum(lane);
            if (simd_lane == 0u && feature < out_features) {
                output[input_row * out_features + feature] = value;
            }
        }
    }
}

kernel void laguna_q8_0_matvec_argmax_candidates_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device uint* candidate_ids [[buffer(2)]],
    device float* candidate_scores [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    constant uint& blocks_per_row [[buffer(6)]],
    constant uint& simdgroups_per_output [[buffer(7)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint output_row = threadgroup_position.x * Q8_0_LAGUNA_ARGMAX_ROWS;
    if (output_row >= out_features
        || simdgroup_index >= simdgroups_per_output) {
        return;
    }

    ushort block_lane = simd_lane / 4u;
    ushort value_lane = simd_lane % 4u;
    uint first_block = uint(simdgroup_index) * Q8_0_LAGUNA_VALUES_PER_LANE
        + uint(block_lane);
    uint block_step =
        simdgroups_per_output * Q8_0_LAGUNA_VALUES_PER_LANE;
    float sums[Q8_0_LAGUNA_ARGMAX_ROWS];
    for (uint row = 0u; row < Q8_0_LAGUNA_ARGMAX_ROWS; row++) {
        sums[row] = 0.0f;
    }

    for (uint block = first_block; block < blocks_per_row; block += block_step) {
        uint input_offset =
            block * Q8_0_BLOCK_VALUES
            + uint(value_lane) * Q8_0_LAGUNA_VALUES_PER_LANE;
        float values[Q8_0_LAGUNA_VALUES_PER_LANE];
        q8_0_lane_load8(input, input_offset, values);
        uint lane_quant_offset =
            2u + uint(value_lane) * Q8_0_LAGUNA_VALUES_PER_LANE;

        for (uint row = 0u; row < Q8_0_LAGUNA_ARGMAX_ROWS; row++) {
            uint feature = output_row + row;
            if (feature >= out_features) {
                continue;
            }
            uint block_offset =
                (feature * blocks_per_row + block) * Q8_0_BLOCK_BYTES;
            sums[row] +=
                q8_0_lane_dot8(weights, block_offset + lane_quant_offset, values)
                * q8_0_block_scale(weights, block_offset);
        }
    }

    threadgroup float partial[
        Q8_0_LAGUNA_ARGMAX_ROWS * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT
    ];
    for (uint row = 0u; row < Q8_0_LAGUNA_ARGMAX_ROWS; row++) {
        float value = simd_sum(sums[row]);
        if (simd_lane == 0u) {
            partial[
                row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT + uint(simdgroup_index)
            ] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0u) {
        float best_score = -3.402823466e+38F;
        uint best_id = 0xffffffffu;
        for (uint row = 0u; row < Q8_0_LAGUNA_ARGMAX_ROWS; row++) {
            uint feature = output_row + row;
            float lane = simd_lane < simdgroups_per_output
                ? partial[row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT + simd_lane]
                : 0.0f;
            float value = simd_sum(lane);
            if (simd_lane == 0u
                && feature < out_features
                && (value > best_score
                    || (value == best_score && feature < best_id))) {
                best_score = value;
                best_id = feature;
            }
        }
        if (simd_lane == 0u) {
            candidate_ids[threadgroup_position.x] = best_id;
            candidate_scores[threadgroup_position.x] = best_score;
        }
    }
}

// Decode Q-A and KV-A together. Both matrices consume the same activation
// row, so each lane loads one input value and applies it to both independent
// Q8_0 weight streams while both output rows are active.
kernel void q8_0_matvec_pair_tiled_f32_kernel(
    const device uchar* weights_a [[buffer(0)]],
    const device uchar* weights_b [[buffer(1)]],
    const device float* input [[buffer(2)]],
    device float* output_a [[buffer(3)]],
    device float* output_b [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& in_features [[buffer(6)]],
    constant uint& out_features_a [[buffer(7)]],
    constant uint& out_features_b [[buffer(8)]],
    constant uint& blocks_per_row [[buffer(9)]],
    constant uint& simdgroups_per_output [[buffer(10)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint max_out_features = max(out_features_a, out_features_b);
    uint input_row = threadgroup_position.x / max_out_features;
    uint output_feature = threadgroup_position.x - (input_row * max_out_features);
    if (input_row >= row_count
        || output_feature >= max_out_features
        || simdgroup_index >= simdgroups_per_output) {
        return;
    }

    uint input_row_offset = input_row * in_features;
    float sum_a = 0.0f;
    float sum_b = 0.0f;
    for (uint block_in_row = simdgroup_index;
         block_in_row < blocks_per_row;
         block_in_row += simdgroups_per_output) {
        uint input_index = input_row_offset
            + (block_in_row * Q8_0_BLOCK_VALUES)
            + simd_lane;
        float input_value = input[input_index];
        uint block_index = (output_feature * blocks_per_row) + block_in_row;
        uint block_offset = block_index * Q8_0_BLOCK_BYTES;
        if (output_feature < out_features_a) {
            sum_a += input_value
                * q8_0_block_value(weights_a, block_offset, simd_lane);
        }
        if (output_feature < out_features_b) {
            sum_b += input_value
                * q8_0_block_value(weights_b, block_offset, simd_lane);
        }
    }

    threadgroup float partial_a[Q8_0_MAX_SIMDGROUPS_PER_OUTPUT];
    threadgroup float partial_b[Q8_0_MAX_SIMDGROUPS_PER_OUTPUT];
    float simd_sum_a = simd_sum(sum_a);
    float simd_sum_b = simd_sum(sum_b);
    if (simd_lane == 0) {
        partial_a[simdgroup_index] = simd_sum_a;
        partial_b[simdgroup_index] = simd_sum_b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0) {
        float lane_a = simd_lane < simdgroups_per_output ? partial_a[simd_lane] : 0.0f;
        float lane_b = simd_lane < simdgroups_per_output ? partial_b[simd_lane] : 0.0f;
        float reduced_a = simd_sum(lane_a);
        float reduced_b = simd_sum(lane_b);
        if (simd_lane == 0) {
            if (output_feature < out_features_a) {
                output_a[(input_row * out_features_a) + output_feature] = reduced_a;
            }
            if (output_feature < out_features_b) {
                output_b[(input_row * out_features_b) + output_feature] = reduced_b;
            }
        }
    }
}

// Laguna decode uses four SIMD groups to compute two adjacent Q8_0 rows.
// Each lane loads eight consecutive activation values and reuses them for
// both projections and both rows.
kernel void laguna_q8_0_matvec_pair_rows2_f32_kernel(
    const device uchar* weights_a [[buffer(0)]],
    const device uchar* weights_b [[buffer(1)]],
    const device float* input [[buffer(2)]],
    device float* output_a [[buffer(3)]],
    device float* output_b [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& in_features [[buffer(6)]],
    constant uint& out_features_a [[buffer(7)]],
    constant uint& out_features_b [[buffer(8)]],
    constant uint& blocks_per_row [[buffer(9)]],
    constant uint& simdgroups_per_output [[buffer(10)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint max_out_features = max(out_features_a, out_features_b);
    uint row_groups_per_input =
        (max_out_features + Q8_0_LAGUNA_OUTPUT_ROWS - 1u)
        / Q8_0_LAGUNA_OUTPUT_ROWS;
    uint input_row = threadgroup_position.x / row_groups_per_input;
    uint row_group = threadgroup_position.x
        - input_row * row_groups_per_input;
    uint output_row = row_group * Q8_0_LAGUNA_OUTPUT_ROWS;
    if (input_row >= row_count
        || output_row >= max_out_features
        || simdgroup_index >= simdgroups_per_output) {
        return;
    }

    ushort block_lane = simd_lane / 4u;
    ushort value_lane = simd_lane % 4u;
    uint first_block = uint(simdgroup_index) * Q8_0_LAGUNA_VALUES_PER_LANE
        + uint(block_lane);
    uint block_step = simdgroups_per_output * Q8_0_LAGUNA_VALUES_PER_LANE;
    uint input_row_offset = input_row * in_features;
    float sums_a[Q8_0_LAGUNA_OUTPUT_ROWS] = {0.0f};
    float sums_b[Q8_0_LAGUNA_OUTPUT_ROWS] = {0.0f};

    for (uint block = first_block; block < blocks_per_row; block += block_step) {
        uint input_offset = input_row_offset
            + block * Q8_0_BLOCK_VALUES
            + uint(value_lane) * Q8_0_LAGUNA_VALUES_PER_LANE;
        float values[Q8_0_LAGUNA_VALUES_PER_LANE];
        q8_0_lane_load8(input, input_offset, values);
        uint lane_quant_offset =
            2u + uint(value_lane) * Q8_0_LAGUNA_VALUES_PER_LANE;

        for (uint row = 0u; row < Q8_0_LAGUNA_OUTPUT_ROWS; row++) {
            uint feature = output_row + row;
            uint block_offset =
                (feature * blocks_per_row + block) * Q8_0_BLOCK_BYTES;
            uint quant_offset = block_offset + lane_quant_offset;
            if (feature < out_features_a) {
                sums_a[row] +=
                    q8_0_lane_dot8(weights_a, quant_offset, values)
                    * q8_0_block_scale(weights_a, block_offset);
            }
            if (feature < out_features_b) {
                sums_b[row] +=
                    q8_0_lane_dot8(weights_b, quant_offset, values)
                    * q8_0_block_scale(weights_b, block_offset);
            }
        }
    }

    threadgroup float partial_a[
        Q8_0_LAGUNA_OUTPUT_ROWS * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT
    ];
    threadgroup float partial_b[
        Q8_0_LAGUNA_OUTPUT_ROWS * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT
    ];
    for (uint row = 0u; row < Q8_0_LAGUNA_OUTPUT_ROWS; row++) {
        float sum_a = simd_sum(sums_a[row]);
        float sum_b = simd_sum(sums_b[row]);
        if (simd_lane == 0u) {
            uint partial_index =
                row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT + uint(simdgroup_index);
            partial_a[partial_index] = sum_a;
            partial_b[partial_index] = sum_b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0u) {
        for (uint row = 0u; row < Q8_0_LAGUNA_OUTPUT_ROWS; row++) {
            uint feature = output_row + row;
            float lane_a = simd_lane < simdgroups_per_output
                ? partial_a[row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT + simd_lane]
                : 0.0f;
            float lane_b = simd_lane < simdgroups_per_output
                ? partial_b[row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT + simd_lane]
                : 0.0f;
            float value_a = simd_sum(lane_a);
            float value_b = simd_sum(lane_b);
            if (simd_lane == 0u) {
                if (feature < out_features_a) {
                    output_a[input_row * out_features_a + feature] = value_a;
                }
                if (feature < out_features_b) {
                    output_b[input_row * out_features_b + feature] = value_b;
                }
            }
        }
    }
}

// Laguna decode projects Q, K, V, and the attention gate from the same
// normalized hidden row. Loading that row once avoids a second full activation
// traversal for the V/gate pair.
kernel void laguna_q8_0_attention_projections_rows2_f32_kernel(
    const device uchar* query_weights [[buffer(0)]],
    const device uchar* key_weights [[buffer(1)]],
    const device uchar* value_weights [[buffer(2)]],
    const device uchar* gate_weights [[buffer(3)]],
    const device float* input [[buffer(4)]],
    device float* query_output [[buffer(5)]],
    device float* key_output [[buffer(6)]],
    device float* value_output [[buffer(7)]],
    device float* gate_output [[buffer(8)]],
    constant uint& row_count [[buffer(9)]],
    constant uint& in_features [[buffer(10)]],
    constant uint& query_features [[buffer(11)]],
    constant uint& key_features [[buffer(12)]],
    constant uint& value_features [[buffer(13)]],
    constant uint& gate_features [[buffer(14)]],
    constant uint& blocks_per_row [[buffer(15)]],
    constant uint& simdgroups_per_output [[buffer(16)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint max_out_features = max(
        max(query_features, key_features),
        max(value_features, gate_features));
    uint row_groups_per_input =
        (max_out_features + Q8_0_LAGUNA_OUTPUT_ROWS - 1u)
        / Q8_0_LAGUNA_OUTPUT_ROWS;
    uint input_row = threadgroup_position.x / row_groups_per_input;
    uint row_group = threadgroup_position.x
        - input_row * row_groups_per_input;
    uint output_row = row_group * Q8_0_LAGUNA_OUTPUT_ROWS;
    if (input_row >= row_count
        || output_row >= max_out_features
        || simdgroup_index >= simdgroups_per_output) {
        return;
    }

    ushort block_lane = simd_lane / 4u;
    ushort value_lane = simd_lane % 4u;
    uint first_block = uint(simdgroup_index) * Q8_0_LAGUNA_VALUES_PER_LANE
        + uint(block_lane);
    uint block_step = simdgroups_per_output * Q8_0_LAGUNA_VALUES_PER_LANE;
    uint input_row_offset = input_row * in_features;
    float2 query_sums = float2(0.0f);
    float2 key_sums = float2(0.0f);
    float2 value_sums = float2(0.0f);
    float2 gate_sums = float2(0.0f);

    for (uint block = first_block; block < blocks_per_row; block += block_step) {
        uint input_offset = input_row_offset
            + block * Q8_0_BLOCK_VALUES
            + uint(value_lane) * Q8_0_LAGUNA_VALUES_PER_LANE;
        float values[Q8_0_LAGUNA_VALUES_PER_LANE];
        q8_0_lane_load8(input, input_offset, values);
        uint lane_quant_offset =
            2u + uint(value_lane) * Q8_0_LAGUNA_VALUES_PER_LANE;

        for (uint row = 0u; row < Q8_0_LAGUNA_OUTPUT_ROWS; row++) {
            uint feature = output_row + row;
            uint weight_offset =
                (feature * blocks_per_row + block) * Q8_0_BLOCK_BYTES;
            uint quant_offset = weight_offset + lane_quant_offset;
            if (feature < query_features) {
                query_sums[row] +=
                    q8_0_lane_dot8(query_weights, quant_offset, values)
                    * q8_0_block_scale(query_weights, weight_offset);
            }
            if (feature < key_features) {
                key_sums[row] +=
                    q8_0_lane_dot8(key_weights, quant_offset, values)
                    * q8_0_block_scale(key_weights, weight_offset);
            }
            if (feature < value_features) {
                value_sums[row] +=
                    q8_0_lane_dot8(value_weights, quant_offset, values)
                    * q8_0_block_scale(value_weights, weight_offset);
            }
            if (feature < gate_features) {
                gate_sums[row] +=
                    q8_0_lane_dot8(gate_weights, quant_offset, values)
                    * q8_0_block_scale(gate_weights, weight_offset);
            }
        }
    }

    threadgroup float query_partials[
        Q8_0_LAGUNA_OUTPUT_ROWS * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT
    ];
    threadgroup float key_partials[
        Q8_0_LAGUNA_OUTPUT_ROWS * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT
    ];
    threadgroup float value_partials[
        Q8_0_LAGUNA_OUTPUT_ROWS * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT
    ];
    threadgroup float gate_partials[
        Q8_0_LAGUNA_OUTPUT_ROWS * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT
    ];
    // Laguna's key, value and gate projections are far narrower than the query
    // projection, so most threadgroups own query rows only. Reducing the three
    // idle projections there would cost three simdgroup reductions per row to
    // produce values nothing reads. `output_row` is threadgroup-uniform, so
    // these guards never diverge within a simdgroup.
    for (uint row = 0u; row < Q8_0_LAGUNA_OUTPUT_ROWS; row++) {
        uint feature = output_row + row;
        uint partial_index =
            row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT + uint(simdgroup_index);
        if (feature < query_features) {
            float partial = simd_sum(query_sums[row]);
            if (simd_lane == 0u) {
                query_partials[partial_index] = partial;
            }
        }
        if (feature < key_features) {
            float partial = simd_sum(key_sums[row]);
            if (simd_lane == 0u) {
                key_partials[partial_index] = partial;
            }
        }
        if (feature < value_features) {
            float partial = simd_sum(value_sums[row]);
            if (simd_lane == 0u) {
                value_partials[partial_index] = partial;
            }
        }
        if (feature < gate_features) {
            float partial = simd_sum(gate_sums[row]);
            if (simd_lane == 0u) {
                gate_partials[partial_index] = partial;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0u) {
        for (uint row = 0u; row < Q8_0_LAGUNA_OUTPUT_ROWS; row++) {
            uint feature = output_row + row;
            uint partial_base = row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT;
            bool contributes = simd_lane < simdgroups_per_output;
            if (feature < query_features) {
                float value = simd_sum(
                    contributes ? query_partials[partial_base + simd_lane] : 0.0f);
                if (simd_lane == 0u) {
                    query_output[input_row * query_features + feature] = value;
                }
            }
            if (feature < key_features) {
                float value = simd_sum(
                    contributes ? key_partials[partial_base + simd_lane] : 0.0f);
                if (simd_lane == 0u) {
                    key_output[input_row * key_features + feature] = value;
                }
            }
            if (feature < value_features) {
                float value = simd_sum(
                    contributes ? value_partials[partial_base + simd_lane] : 0.0f);
                if (simd_lane == 0u) {
                    value_output[input_row * value_features + feature] = value;
                }
            }
            if (feature < gate_features) {
                float value = simd_sum(
                    contributes ? gate_partials[partial_base + simd_lane] : 0.0f);
                if (simd_lane == 0u) {
                    gate_output[input_row * gate_features + feature] = value;
                }
            }
        }
    }
}

static inline float2 laguna_q8_0_rows2_partial(
    const device uchar* weights,
    const device float* input,
    uint input_row_offset,
    uint output_row,
    uint out_features,
    uint blocks_per_row,
    uint simdgroups_per_output,
    ushort simd_lane,
    ushort simdgroup_index
) {
    ushort block_lane = simd_lane / 4u;
    ushort value_lane = simd_lane % 4u;
    uint first_block = uint(simdgroup_index) * Q8_0_LAGUNA_VALUES_PER_LANE
        + uint(block_lane);
    uint block_step =
        simdgroups_per_output * Q8_0_LAGUNA_VALUES_PER_LANE;
    float2 sums = float2(0.0f);

    for (uint block = first_block; block < blocks_per_row; block += block_step) {
        uint input_offset = input_row_offset
            + block * Q8_0_BLOCK_VALUES
            + uint(value_lane) * Q8_0_LAGUNA_VALUES_PER_LANE;
        float values[Q8_0_LAGUNA_VALUES_PER_LANE];
        q8_0_lane_load8(input, input_offset, values);
        uint lane_quant_offset =
            2u + uint(value_lane) * Q8_0_LAGUNA_VALUES_PER_LANE;

        for (uint row = 0u; row < Q8_0_LAGUNA_OUTPUT_ROWS; row++) {
            uint feature = output_row + row;
            if (feature >= out_features) {
                continue;
            }
            uint block_offset =
                (feature * blocks_per_row + block) * Q8_0_BLOCK_BYTES;
            sums[row] +=
                q8_0_lane_dot8(weights, block_offset + lane_quant_offset, values)
                * q8_0_block_scale(weights, block_offset);
        }
    }
    return sums;
}

kernel void laguna_q8_0_matvec_rows2_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    constant uint& blocks_per_row [[buffer(6)]],
    constant uint& simdgroups_per_output [[buffer(7)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint row_groups_per_input =
        (out_features + Q8_0_LAGUNA_OUTPUT_ROWS - 1u)
        / Q8_0_LAGUNA_OUTPUT_ROWS;
    uint input_row = threadgroup_position.x / row_groups_per_input;
    uint row_group = threadgroup_position.x
        - input_row * row_groups_per_input;
    uint output_row = row_group * Q8_0_LAGUNA_OUTPUT_ROWS;
    if (input_row >= row_count
        || output_row >= out_features
        || simdgroup_index >= simdgroups_per_output) {
        return;
    }

    float2 sums = laguna_q8_0_rows2_partial(
        weights,
        input,
        input_row * in_features,
        output_row,
        out_features,
        blocks_per_row,
        simdgroups_per_output,
        simd_lane,
        simdgroup_index);
    threadgroup float partial[
        Q8_0_LAGUNA_OUTPUT_ROWS * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT
    ];
    for (uint row = 0u; row < Q8_0_LAGUNA_OUTPUT_ROWS; row++) {
        float value = simd_sum(sums[row]);
        if (simd_lane == 0u) {
            partial[
                row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT + uint(simdgroup_index)
            ] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0u) {
        for (uint row = 0u; row < Q8_0_LAGUNA_OUTPUT_ROWS; row++) {
            uint feature = output_row + row;
            float lane = simd_lane < simdgroups_per_output
                ? partial[row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT + simd_lane]
                : 0.0f;
            float value = simd_sum(lane);
            if (simd_lane == 0u && feature < out_features) {
                output[input_row * out_features + feature] = value;
            }
        }
    }
}

kernel void laguna_q8_0_matvec_add_rows2_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    const device float* residual [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& blocks_per_row [[buffer(7)]],
    constant uint& simdgroups_per_output [[buffer(8)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint row_groups_per_input =
        (out_features + Q8_0_LAGUNA_OUTPUT_ROWS - 1u)
        / Q8_0_LAGUNA_OUTPUT_ROWS;
    uint input_row = threadgroup_position.x / row_groups_per_input;
    uint row_group = threadgroup_position.x
        - input_row * row_groups_per_input;
    uint output_row = row_group * Q8_0_LAGUNA_OUTPUT_ROWS;
    if (input_row >= row_count
        || output_row >= out_features
        || simdgroup_index >= simdgroups_per_output) {
        return;
    }

    float2 sums = laguna_q8_0_rows2_partial(
        weights,
        input,
        input_row * in_features,
        output_row,
        out_features,
        blocks_per_row,
        simdgroups_per_output,
        simd_lane,
        simdgroup_index);
    threadgroup float partial[
        Q8_0_LAGUNA_OUTPUT_ROWS * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT
    ];
    for (uint row = 0u; row < Q8_0_LAGUNA_OUTPUT_ROWS; row++) {
        float value = simd_sum(sums[row]);
        if (simd_lane == 0u) {
            partial[
                row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT + uint(simdgroup_index)
            ] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0u) {
        for (uint row = 0u; row < Q8_0_LAGUNA_OUTPUT_ROWS; row++) {
            uint feature = output_row + row;
            float lane = simd_lane < simdgroups_per_output
                ? partial[row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT + simd_lane]
                : 0.0f;
            float value = simd_sum(lane);
            if (simd_lane == 0u && feature < out_features) {
                uint output_index = input_row * out_features + feature;
                output[output_index] = residual[output_index] + value;
            }
        }
    }
}

kernel void laguna_q8_0_matvec_add2_rows2_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    const device float* residual_a [[buffer(2)]],
    const device float* residual_b [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& in_features [[buffer(6)]],
    constant uint& out_features [[buffer(7)]],
    constant uint& blocks_per_row [[buffer(8)]],
    constant uint& simdgroups_per_output [[buffer(9)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint row_groups_per_input =
        (out_features + Q8_0_LAGUNA_OUTPUT_ROWS - 1u)
        / Q8_0_LAGUNA_OUTPUT_ROWS;
    uint input_row = threadgroup_position.x / row_groups_per_input;
    uint row_group = threadgroup_position.x
        - input_row * row_groups_per_input;
    uint output_row = row_group * Q8_0_LAGUNA_OUTPUT_ROWS;
    if (input_row >= row_count
        || output_row >= out_features
        || simdgroup_index >= simdgroups_per_output) {
        return;
    }

    float2 sums = laguna_q8_0_rows2_partial(
        weights,
        input,
        input_row * in_features,
        output_row,
        out_features,
        blocks_per_row,
        simdgroups_per_output,
        simd_lane,
        simdgroup_index);
    threadgroup float partial[
        Q8_0_LAGUNA_OUTPUT_ROWS * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT
    ];
    for (uint row = 0u; row < Q8_0_LAGUNA_OUTPUT_ROWS; row++) {
        float value = simd_sum(sums[row]);
        if (simd_lane == 0u) {
            partial[
                row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT + uint(simdgroup_index)
            ] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0u) {
        for (uint row = 0u; row < Q8_0_LAGUNA_OUTPUT_ROWS; row++) {
            uint feature = output_row + row;
            float lane = simd_lane < simdgroups_per_output
                ? partial[row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT + simd_lane]
                : 0.0f;
            float value = simd_sum(lane);
            if (simd_lane == 0u && feature < out_features) {
                uint output_index = input_row * out_features + feature;
                output[output_index] =
                    residual_a[output_index] + residual_b[output_index] + value;
            }
        }
    }
}

kernel void laguna_q8_0_gate_up_swiglu_rows2_f32_kernel(
    const device uchar* gate_weights [[buffer(0)]],
    const device uchar* up_weights [[buffer(1)]],
    const device float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& blocks_per_row [[buffer(7)]],
    constant uint& simdgroups_per_output [[buffer(8)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint row_groups_per_input =
        (out_features + Q8_0_LAGUNA_OUTPUT_ROWS - 1u)
        / Q8_0_LAGUNA_OUTPUT_ROWS;
    uint input_row = threadgroup_position.x / row_groups_per_input;
    uint row_group = threadgroup_position.x
        - input_row * row_groups_per_input;
    uint output_row = row_group * Q8_0_LAGUNA_OUTPUT_ROWS;
    if (input_row >= row_count
        || output_row >= out_features
        || simdgroup_index >= simdgroups_per_output) {
        return;
    }

    ushort block_lane = simd_lane / 4u;
    ushort value_lane = simd_lane % 4u;
    uint first_block = uint(simdgroup_index) * Q8_0_LAGUNA_VALUES_PER_LANE
        + uint(block_lane);
    uint block_step =
        simdgroups_per_output * Q8_0_LAGUNA_VALUES_PER_LANE;
    uint input_row_offset = input_row * in_features;
    float2 gate_sums = float2(0.0f);
    float2 up_sums = float2(0.0f);

    for (uint block = first_block; block < blocks_per_row; block += block_step) {
        uint input_offset = input_row_offset
            + block * Q8_0_BLOCK_VALUES
            + uint(value_lane) * Q8_0_LAGUNA_VALUES_PER_LANE;
        float values[Q8_0_LAGUNA_VALUES_PER_LANE];
        q8_0_lane_load8(input, input_offset, values);
        uint lane_quant_offset =
            2u + uint(value_lane) * Q8_0_LAGUNA_VALUES_PER_LANE;

        for (uint row = 0u; row < Q8_0_LAGUNA_OUTPUT_ROWS; row++) {
            uint feature = output_row + row;
            if (feature >= out_features) {
                continue;
            }
            uint block_offset =
                (feature * blocks_per_row + block) * Q8_0_BLOCK_BYTES;
            uint quant_offset = block_offset + lane_quant_offset;
            gate_sums[row] +=
                q8_0_lane_dot8(gate_weights, quant_offset, values)
                * q8_0_block_scale(gate_weights, block_offset);
            up_sums[row] +=
                q8_0_lane_dot8(up_weights, quant_offset, values)
                * q8_0_block_scale(up_weights, block_offset);
        }
    }

    threadgroup float partial_gate[
        Q8_0_LAGUNA_OUTPUT_ROWS * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT
    ];
    threadgroup float partial_up[
        Q8_0_LAGUNA_OUTPUT_ROWS * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT
    ];
    for (uint row = 0u; row < Q8_0_LAGUNA_OUTPUT_ROWS; row++) {
        float gate_value = simd_sum(gate_sums[row]);
        float up_value = simd_sum(up_sums[row]);
        if (simd_lane == 0u) {
            uint partial_index =
                row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT + uint(simdgroup_index);
            partial_gate[partial_index] = gate_value;
            partial_up[partial_index] = up_value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0u) {
        for (uint row = 0u; row < Q8_0_LAGUNA_OUTPUT_ROWS; row++) {
            uint feature = output_row + row;
            float gate_lane = simd_lane < simdgroups_per_output
                ? partial_gate[
                    row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT + simd_lane
                ]
                : 0.0f;
            float up_lane = simd_lane < simdgroups_per_output
                ? partial_up[
                    row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT + simd_lane
                ]
                : 0.0f;
            float gate = simd_sum(gate_lane);
            float up = simd_sum(up_lane);
            if (simd_lane == 0u && feature < out_features) {
                output[input_row * out_features + feature] =
                    (gate / (1.0f + exp(-gate))) * up;
            }
        }
    }
}

kernel void q8_0_gate_up_swiglu_tiled_f32_kernel(
    const device uchar* gate_weights [[buffer(0)]],
    const device uchar* up_weights [[buffer(1)]],
    const device float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& blocks_per_row [[buffer(7)]],
    constant uint& simdgroups_per_output [[buffer(8)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint input_row = threadgroup_position.x / out_features;
    uint output_feature = threadgroup_position.x - (input_row * out_features);
    if (input_row >= row_count
        || output_feature >= out_features
        || simdgroup_index >= simdgroups_per_output) {
        return;
    }

    uint input_row_offset = input_row * in_features;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint block_in_row = simdgroup_index;
         block_in_row < blocks_per_row;
         block_in_row += simdgroups_per_output) {
        uint input_index = input_row_offset
            + (block_in_row * Q8_0_BLOCK_VALUES)
            + simd_lane;
        float input_value = input[input_index];
        uint block_index = (output_feature * blocks_per_row) + block_in_row;
        uint block_offset = block_index * Q8_0_BLOCK_BYTES;
        gate_sum += input_value
            * q8_0_block_value(gate_weights, block_offset, simd_lane);
        up_sum += input_value
            * q8_0_block_value(up_weights, block_offset, simd_lane);
    }

    threadgroup float partial_gate[Q8_0_MAX_SIMDGROUPS_PER_OUTPUT];
    threadgroup float partial_up[Q8_0_MAX_SIMDGROUPS_PER_OUTPUT];
    float simd_gate = simd_sum(gate_sum);
    float simd_up = simd_sum(up_sum);
    if (simd_lane == 0) {
        partial_gate[simdgroup_index] = simd_gate;
        partial_up[simdgroup_index] = simd_up;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0) {
        float gate_partial = simd_lane < simdgroups_per_output
            ? partial_gate[simd_lane]
            : 0.0f;
        float up_partial = simd_lane < simdgroups_per_output
            ? partial_up[simd_lane]
            : 0.0f;
        float gate = simd_sum(gate_partial);
        float up = simd_sum(up_partial);
        if (simd_lane == 0) {
            output[(input_row * out_features) + output_feature]
                = (gate / (1.0f + exp(-gate))) * up;
        }
    }
}

// Dense Q8_0 projections share the same weight matrix across MTP and prefill
// rows. Decode each weight value once per row tile before moving to the next
// block; row_count may contain any number of four-row tiles.
kernel void q8_0_batched_matvec_tiled_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    constant uint& blocks_per_row [[buffer(6)]],
    constant uint& simdgroups_per_output [[buffer(7)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint output_tile = threadgroup_position.x;
    uint row_tile = output_tile / out_features;
    uint output_feature = output_tile - (row_tile * out_features);
    uint row_start = row_tile * Q8_0_BATCH_ROW_TILE;
    if (output_feature >= out_features
        || row_start >= row_count
        || simdgroup_index >= simdgroups_per_output) {
        return;
    }
    uint active_rows = min(Q8_0_BATCH_ROW_TILE, row_count - row_start);

    float sums[Q8_0_BATCH_ROW_TILE];
    for (uint row = 0; row < Q8_0_BATCH_ROW_TILE; row++) {
        sums[row] = 0.0f;
    }

    for (uint block_in_row = simdgroup_index;
         block_in_row < blocks_per_row;
         block_in_row += simdgroups_per_output) {
        uint block_index = (output_feature * blocks_per_row) + block_in_row;
        uint block_offset = block_index * Q8_0_BLOCK_BYTES;
        float weight = q8_0_block_value(weights, block_offset, simd_lane);
        uint input_column = (block_in_row * Q8_0_BLOCK_VALUES) + simd_lane;
        for (uint row = 0; row < active_rows; row++) {
            sums[row] += input[((row_start + row) * in_features) + input_column] * weight;
        }
    }

    threadgroup float partial_sums[Q8_0_BATCH_ROW_TILE * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT];
    for (uint row = 0; row < active_rows; row++) {
        float simdgroup_sum = simd_sum(sums[row]);
        if (simd_lane == 0) {
            partial_sums[(row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT) + simdgroup_index] = simdgroup_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0) {
        for (uint row = 0; row < active_rows; row++) {
            float partial = simd_lane < simdgroups_per_output
                ? partial_sums[(row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT) + simd_lane]
                : 0.0f;
            float reduced_sum = simd_sum(partial);
            if (simd_lane == 0) {
                output[((row_start + row) * out_features) + output_feature] = reduced_sum;
            }
        }
    }
}

template <bool TailSafe>
static inline void q8_0_stage_prefill_mma_tile(
    const device uchar* weights,
    const device float* input,
    threadgroup half* input_tile,
    threadgroup half* weight_tile,
    uint token_start,
    uint output_start,
    uint active_tokens,
    uint active_outputs,
    uint in_features,
    uint blocks_per_row,
    uint block_in_row,
    uint thread_index
) {
    constexpr uint input_vectors_per_token = Q8_0_MMA_K_TILE / 4u;
    for (uint vector_index = thread_index;
         vector_index < Q8_0_MMA_TOKEN_TILE * input_vectors_per_token;
         vector_index += Q8_0_MMA_THREAD_COUNT) {
        uint token = vector_index / input_vectors_per_token;
        uint input_vector =
            vector_index - (token * input_vectors_per_token);
        uint input_column = input_vector * 4u;
        threadgroup half4* destination =
            reinterpret_cast<threadgroup half4*>(
                input_tile
                    + token * Q8_0_MMA_K_TILE
                    + input_column);
        if (token < active_tokens) {
            const device float4* source =
                reinterpret_cast<const device float4*>(
                    input
                        + (token_start + token) * in_features
                        + block_in_row * Q8_0_MMA_K_TILE
                        + input_column);
            *destination = half4(*source);
        } else {
            *destination = half4(0.0h);
        }
    }

    constexpr uint quant_groups_per_block = Q8_0_MMA_K_TILE / 8u;
    for (uint quant_group_index = thread_index;
         quant_group_index
            < Q8_0_MMA_OUTPUT_TILE * quant_groups_per_block;
         quant_group_index += Q8_0_MMA_THREAD_COUNT) {
        uint output_feature =
            quant_group_index / quant_groups_per_block;
        uint quant_group =
            quant_group_index
            - output_feature * quant_groups_per_block;
        threadgroup half4* destination =
            reinterpret_cast<threadgroup half4*>(
                weight_tile
                    + output_feature * Q8_0_MMA_K_TILE
                    + quant_group * 8u);
        if (TailSafe && output_feature >= active_outputs) {
            destination[0] = half4(0.0h);
            destination[1] = half4(0.0h);
            continue;
        }
        uint block_index = ((output_start + output_feature) * blocks_per_row)
            + block_in_row;
        uint block_offset = block_index * Q8_0_BLOCK_BYTES;
        float scale = q8_0_block_scale(weights, block_offset);
        const device packed_char4* quants =
            reinterpret_cast<const device packed_char4*>(
                weights
                    + block_offset
                    + 2u
                    + quant_group * 8u);
        char4 low = char4(quants[0]);
        char4 high = char4(quants[1]);
        destination[0] = half4(float4(low) * scale);
        destination[1] = half4(float4(high) * scale);
    }
}

// Prompt-time Q8_0 matrix multiplication. Four SIMD groups share 32
// activation rows and one dequantized weight tile. Each group computes eight
// adjacent output features for all four 8-token subtiles. Alternating the K
// tile storage removes the overwrite barrier between tiles, reducing weight
// decoding and synchronization without materializing F16 tensors.
template <bool TailSafe, ushort ResidualCount>
kernel void q8_0_prefill_mma_impl(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    const device float* residual_a [[buffer(2)]],
    const device float* residual_b [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& row_count [[buffer(5)]],
    constant uint& in_features [[buffer(6)]],
    constant uint& out_features [[buffer(7)]],
    constant uint& blocks_per_row [[buffer(8)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint output_tiles = TailSafe
        ? (out_features + Q8_0_MMA_OUTPUT_TILE - 1u)
            / Q8_0_MMA_OUTPUT_TILE
        : out_features / Q8_0_MMA_OUTPUT_TILE;
    uint token_tile = threadgroup_position.x / output_tiles;
    uint output_tile = threadgroup_position.x - (token_tile * output_tiles);
    uint token_start = token_tile * Q8_0_MMA_TOKEN_TILE;
    uint output_start = output_tile * Q8_0_MMA_OUTPUT_TILE;
    uint active_tokens = min(Q8_0_MMA_TOKEN_TILE, row_count - token_start);
    uint active_outputs = TailSafe
        ? min(Q8_0_MMA_OUTPUT_TILE, out_features - output_start)
        : Q8_0_MMA_OUTPUT_TILE;

    threadgroup half input_tiles[2 * Q8_0_MMA_TOKEN_TILE * Q8_0_MMA_K_TILE];
    threadgroup half weight_tiles[2 * Q8_0_MMA_OUTPUT_TILE * Q8_0_MMA_K_TILE];
    threadgroup float output_tile_values[
        Q8_0_MMA_TOKEN_TILE * Q8_0_MMA_OUTPUT_TILE
    ];
    simdgroup_float8x8 accumulators[Q8_0_MMA_TOKEN_GROUPS];
    for (uint group = 0; group < Q8_0_MMA_TOKEN_GROUPS; group++) {
        accumulators[group] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    q8_0_stage_prefill_mma_tile<TailSafe>(
        weights,
        input,
        input_tiles,
        weight_tiles,
        token_start,
        output_start,
        active_tokens,
        active_outputs,
        in_features,
        blocks_per_row,
        0,
        thread_index
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tile_index = 0;
    for (uint block_in_row = 0; block_in_row < blocks_per_row; block_in_row++) {
        threadgroup half* input_tile = input_tiles
            + (tile_index * Q8_0_MMA_TOKEN_TILE * Q8_0_MMA_K_TILE);
        threadgroup half* weight_tile = weight_tiles
            + (tile_index * Q8_0_MMA_OUTPUT_TILE * Q8_0_MMA_K_TILE);

        for (uint k_offset = 0; k_offset < Q8_0_MMA_K_TILE; k_offset += 8u) {
            simdgroup_half8x8 weight_matrix;
            simdgroup_load(
                weight_matrix,
                weight_tile
                    + (simdgroup_index * 8u * Q8_0_MMA_K_TILE)
                    + k_offset,
                Q8_0_MMA_K_TILE,
                0,
                true
            );
            for (uint group = 0; group < Q8_0_MMA_TOKEN_GROUPS; group++) {
                simdgroup_half8x8 input_matrix;
                simdgroup_load(
                    input_matrix,
                    input_tile
                        + (group * 8u * Q8_0_MMA_K_TILE)
                        + k_offset,
                    Q8_0_MMA_K_TILE,
                    0,
                    false
                );
                simdgroup_multiply_accumulate(
                    accumulators[group],
                    input_matrix,
                    weight_matrix,
                    accumulators[group]
                );
            }
        }

        uint next_block = block_in_row + 1u;
        if (next_block < blocks_per_row) {
            tile_index ^= 1u;
            q8_0_stage_prefill_mma_tile<TailSafe>(
                weights,
                input,
                input_tiles
                    + (tile_index * Q8_0_MMA_TOKEN_TILE * Q8_0_MMA_K_TILE),
                weight_tiles
                    + (tile_index * Q8_0_MMA_OUTPUT_TILE * Q8_0_MMA_K_TILE),
                token_start,
                output_start,
                active_tokens,
                active_outputs,
                in_features,
                blocks_per_row,
                next_block,
                thread_index
            );
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint group = 0; group < Q8_0_MMA_TOKEN_GROUPS; group++) {
        simdgroup_store(
            accumulators[group],
            output_tile_values
                + (group * 8u * Q8_0_MMA_OUTPUT_TILE)
                + (simdgroup_index * 8u),
            Q8_0_MMA_OUTPUT_TILE,
            0,
            false
        );
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = thread_index;
         index < active_tokens * Q8_0_MMA_OUTPUT_TILE;
         index += Q8_0_MMA_OUTPUT_TILE * 4u) {
        uint token = index / Q8_0_MMA_OUTPUT_TILE;
        uint output_feature = index - (token * Q8_0_MMA_OUTPUT_TILE);
        if (!TailSafe || output_feature < active_outputs) {
            uint output_index =
                ((token_start + token) * out_features)
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
}

typedef decltype(q8_0_prefill_mma_impl<false, 0>)
    Q8_0PrefillMmaAligned;
typedef decltype(q8_0_prefill_mma_impl<true, 0>)
    Q8_0PrefillMmaTail;
typedef decltype(q8_0_prefill_mma_impl<false, 1>)
    Q8_0PrefillMmaAdd;
typedef decltype(q8_0_prefill_mma_impl<false, 2>)
    Q8_0PrefillMmaAdd2;

template [[host_name("q8_0_prefill_mma_f32_kernel")]]
kernel Q8_0PrefillMmaAligned
q8_0_prefill_mma_impl<false, 0>;

template [[host_name("q8_0_prefill_mma_tail_f32_kernel")]]
kernel Q8_0PrefillMmaTail
q8_0_prefill_mma_impl<true, 0>;

template [[host_name("q8_0_prefill_mma_add_f32_kernel")]]
kernel Q8_0PrefillMmaAdd
q8_0_prefill_mma_impl<false, 1>;

template [[host_name("q8_0_prefill_mma_add2_f32_kernel")]]
kernel Q8_0PrefillMmaAdd2
q8_0_prefill_mma_impl<false, 2>;

static inline void q8_0_stage_prefill_mma_gate_up_tile(
    const device uchar* gate_weights,
    const device uchar* up_weights,
    const device float* input,
    threadgroup half* input_tile,
    threadgroup half* gate_tile,
    threadgroup half* up_tile,
    uint token_start,
    uint output_start,
    uint active_tokens,
    uint in_features,
    uint blocks_per_row,
    uint block_in_row,
    uint thread_index
) {
    constexpr uint input_vectors_per_token = Q8_0_MMA_K_TILE / 4u;
    for (uint vector_index = thread_index;
         vector_index < Q8_0_MMA_TOKEN_TILE * input_vectors_per_token;
         vector_index += Q8_0_MMA_THREAD_COUNT) {
        uint token = vector_index / input_vectors_per_token;
        uint input_vector = vector_index - token * input_vectors_per_token;
        uint input_column = input_vector * 4u;
        threadgroup half4* destination =
            reinterpret_cast<threadgroup half4*>(
                input_tile + token * Q8_0_MMA_K_TILE + input_column);
        if (token < active_tokens) {
            const device float4* source =
                reinterpret_cast<const device float4*>(
                    input
                        + (token_start + token) * in_features
                        + block_in_row * Q8_0_MMA_K_TILE
                        + input_column);
            *destination = half4(*source);
        } else {
            *destination = half4(0.0h);
        }
    }

    constexpr uint quant_groups_per_block = Q8_0_MMA_K_TILE / 8u;
    for (uint quant_group_index = thread_index;
         quant_group_index
            < Q8_0_MMA_OUTPUT_TILE * quant_groups_per_block;
         quant_group_index += Q8_0_MMA_THREAD_COUNT) {
        uint output_feature = quant_group_index / quant_groups_per_block;
        uint quant_group =
            quant_group_index - output_feature * quant_groups_per_block;
        uint block_index =
            (output_start + output_feature) * blocks_per_row + block_in_row;
        uint block_offset = block_index * Q8_0_BLOCK_BYTES;
        uint quant_offset = block_offset + 2u + quant_group * 8u;

        float gate_scale = q8_0_block_scale(gate_weights, block_offset);
        const device packed_char4* gate_quants =
            reinterpret_cast<const device packed_char4*>(
                gate_weights + quant_offset);
        threadgroup half4* gate_destination =
            reinterpret_cast<threadgroup half4*>(
                gate_tile
                    + output_feature * Q8_0_MMA_K_TILE
                    + quant_group * 8u);
        gate_destination[0] =
            half4(float4(char4(gate_quants[0])) * gate_scale);
        gate_destination[1] =
            half4(float4(char4(gate_quants[1])) * gate_scale);

        float up_scale = q8_0_block_scale(up_weights, block_offset);
        const device packed_char4* up_quants =
            reinterpret_cast<const device packed_char4*>(
                up_weights + quant_offset);
        threadgroup half4* up_destination =
            reinterpret_cast<threadgroup half4*>(
                up_tile
                    + output_feature * Q8_0_MMA_K_TILE
                    + quant_group * 8u);
        up_destination[0] =
            half4(float4(char4(up_quants[0])) * up_scale);
        up_destination[1] =
            half4(float4(char4(up_quants[1])) * up_scale);
    }
}

// Multi-token gate/up projection. One threadgroup reuses each staged
// activation tile for both matrices and writes only the final SwiGLU result.
kernel void q8_0_prefill_mma_gate_up_swiglu_f32_kernel(
    const device uchar* gate_weights [[buffer(0)]],
    const device uchar* up_weights [[buffer(1)]],
    const device float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& blocks_per_row [[buffer(7)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint output_tiles = out_features / Q8_0_MMA_OUTPUT_TILE;
    uint token_tile = threadgroup_position.x / output_tiles;
    uint output_tile = threadgroup_position.x - token_tile * output_tiles;
    uint token_start = token_tile * Q8_0_MMA_TOKEN_TILE;
    uint output_start = output_tile * Q8_0_MMA_OUTPUT_TILE;
    uint active_tokens = min(Q8_0_MMA_TOKEN_TILE, row_count - token_start);

    threadgroup half input_tiles[
        2 * Q8_0_MMA_TOKEN_TILE * Q8_0_MMA_K_TILE
    ];
    threadgroup half gate_tiles[
        2 * Q8_0_MMA_OUTPUT_TILE * Q8_0_MMA_K_TILE
    ];
    threadgroup half up_tiles[
        2 * Q8_0_MMA_OUTPUT_TILE * Q8_0_MMA_K_TILE
    ];
    threadgroup float gate_output[
        Q8_0_MMA_TOKEN_TILE * Q8_0_MMA_OUTPUT_TILE
    ];
    threadgroup float up_output[
        Q8_0_MMA_TOKEN_TILE * Q8_0_MMA_OUTPUT_TILE
    ];
    simdgroup_float8x8 gate_accumulators[Q8_0_MMA_TOKEN_GROUPS];
    simdgroup_float8x8 up_accumulators[Q8_0_MMA_TOKEN_GROUPS];
    for (uint group = 0; group < Q8_0_MMA_TOKEN_GROUPS; group++) {
        gate_accumulators[group] =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
        up_accumulators[group] =
            make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    q8_0_stage_prefill_mma_gate_up_tile(
        gate_weights,
        up_weights,
        input,
        input_tiles,
        gate_tiles,
        up_tiles,
        token_start,
        output_start,
        active_tokens,
        in_features,
        blocks_per_row,
        0u,
        thread_index
    );
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint tile_index = 0u;
    for (uint block_in_row = 0u;
         block_in_row < blocks_per_row;
         block_in_row++) {
        threadgroup half* input_tile = input_tiles
            + tile_index * Q8_0_MMA_TOKEN_TILE * Q8_0_MMA_K_TILE;
        threadgroup half* gate_tile = gate_tiles
            + tile_index * Q8_0_MMA_OUTPUT_TILE * Q8_0_MMA_K_TILE;
        threadgroup half* up_tile = up_tiles
            + tile_index * Q8_0_MMA_OUTPUT_TILE * Q8_0_MMA_K_TILE;

        for (uint k_offset = 0u;
             k_offset < Q8_0_MMA_K_TILE;
             k_offset += 8u) {
            simdgroup_half8x8 gate_matrix;
            simdgroup_half8x8 up_matrix;
            simdgroup_load(
                gate_matrix,
                gate_tile
                    + simdgroup_index * 8u * Q8_0_MMA_K_TILE
                    + k_offset,
                Q8_0_MMA_K_TILE,
                0,
                true);
            simdgroup_load(
                up_matrix,
                up_tile
                    + simdgroup_index * 8u * Q8_0_MMA_K_TILE
                    + k_offset,
                Q8_0_MMA_K_TILE,
                0,
                true);
            for (uint group = 0u;
                 group < Q8_0_MMA_TOKEN_GROUPS;
                 group++) {
                simdgroup_half8x8 input_matrix;
                simdgroup_load(
                    input_matrix,
                    input_tile
                        + group * 8u * Q8_0_MMA_K_TILE
                        + k_offset,
                    Q8_0_MMA_K_TILE,
                    0,
                    false);
                simdgroup_multiply_accumulate(
                    gate_accumulators[group],
                    input_matrix,
                    gate_matrix,
                    gate_accumulators[group]);
                simdgroup_multiply_accumulate(
                    up_accumulators[group],
                    input_matrix,
                    up_matrix,
                    up_accumulators[group]);
            }
        }

        uint next_block = block_in_row + 1u;
        if (next_block < blocks_per_row) {
            tile_index ^= 1u;
            q8_0_stage_prefill_mma_gate_up_tile(
                gate_weights,
                up_weights,
                input,
                input_tiles
                    + tile_index
                        * Q8_0_MMA_TOKEN_TILE
                        * Q8_0_MMA_K_TILE,
                gate_tiles
                    + tile_index
                        * Q8_0_MMA_OUTPUT_TILE
                        * Q8_0_MMA_K_TILE,
                up_tiles
                    + tile_index
                        * Q8_0_MMA_OUTPUT_TILE
                        * Q8_0_MMA_K_TILE,
                token_start,
                output_start,
                active_tokens,
                in_features,
                blocks_per_row,
                next_block,
                thread_index
            );
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint group = 0u; group < Q8_0_MMA_TOKEN_GROUPS; group++) {
        uint store_offset =
            group * 8u * Q8_0_MMA_OUTPUT_TILE
            + simdgroup_index * 8u;
        simdgroup_store(
            gate_accumulators[group],
            gate_output + store_offset,
            Q8_0_MMA_OUTPUT_TILE,
            0,
            false);
        simdgroup_store(
            up_accumulators[group],
            up_output + store_offset,
            Q8_0_MMA_OUTPUT_TILE,
            0,
            false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = thread_index;
         index < active_tokens * Q8_0_MMA_OUTPUT_TILE;
         index += Q8_0_MMA_THREAD_COUNT) {
        uint token = index / Q8_0_MMA_OUTPUT_TILE;
        uint output_feature = index - token * Q8_0_MMA_OUTPUT_TILE;
        float gate = gate_output[index];
        float up = up_output[index];
        uint output_index =
            (token_start + token) * out_features
            + output_start
            + output_feature;
        output[output_index] = (gate / (1.0f + exp(-gate))) * up;
    }
}

kernel void q8_0_matvec_add_tiled_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    const device float* residual [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& blocks_per_row [[buffer(7)]],
    constant uint& simdgroups_per_output [[buffer(8)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint output_values = row_count * out_features;
    uint output_index = threadgroup_position.x;
    if (output_index >= output_values || simdgroup_index >= simdgroups_per_output) {
        return;
    }

    uint input_row = output_index / out_features;
    uint output_feature = output_index - (input_row * out_features);
    uint input_row_offset = input_row * in_features;
    float sum = 0.0f;

    for (uint block_in_row = simdgroup_index;
         block_in_row < blocks_per_row;
         block_in_row += simdgroups_per_output) {
        uint block_index = (output_feature * blocks_per_row) + block_in_row;
        uint block_offset = block_index * Q8_0_BLOCK_BYTES;
        uint input_index = input_row_offset
            + (block_in_row * Q8_0_BLOCK_VALUES)
            + simd_lane;
        sum += input[input_index] * q8_0_block_value(weights, block_offset, simd_lane);
    }

    threadgroup float partial_sums[Q8_0_MAX_SIMDGROUPS_PER_OUTPUT];
    float simdgroup_sum = simd_sum(sum);
    if (simd_lane == 0) {
        partial_sums[simdgroup_index] = simdgroup_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0) {
        float partial = simd_lane < simdgroups_per_output ? partial_sums[simd_lane] : 0.0f;
        float reduced_sum = simd_sum(partial);
        if (simd_lane == 0) {
            output[output_index] = reduced_sum + residual[output_index];
        }
    }
}

kernel void q8_0_batched_matvec_add_tiled_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    const device float* residual [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& blocks_per_row [[buffer(7)]],
    constant uint& simdgroups_per_output [[buffer(8)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint output_tile = threadgroup_position.x;
    uint row_tile = output_tile / out_features;
    uint output_feature = output_tile - (row_tile * out_features);
    uint row_start = row_tile * Q8_0_BATCH_ROW_TILE;
    if (output_feature >= out_features
        || row_start >= row_count
        || simdgroup_index >= simdgroups_per_output) {
        return;
    }
    uint active_rows = min(Q8_0_BATCH_ROW_TILE, row_count - row_start);

    float sums[Q8_0_BATCH_ROW_TILE];
    for (uint row = 0; row < Q8_0_BATCH_ROW_TILE; row++) {
        sums[row] = 0.0f;
    }

    for (uint block_in_row = simdgroup_index;
         block_in_row < blocks_per_row;
         block_in_row += simdgroups_per_output) {
        uint block_index = (output_feature * blocks_per_row) + block_in_row;
        uint block_offset = block_index * Q8_0_BLOCK_BYTES;
        float weight = q8_0_block_value(weights, block_offset, simd_lane);
        uint input_column = (block_in_row * Q8_0_BLOCK_VALUES) + simd_lane;
        for (uint row = 0; row < active_rows; row++) {
            sums[row] += input[((row_start + row) * in_features) + input_column] * weight;
        }
    }

    threadgroup float partial_sums[Q8_0_BATCH_ROW_TILE * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT];
    for (uint row = 0; row < active_rows; row++) {
        float simdgroup_sum = simd_sum(sums[row]);
        if (simd_lane == 0) {
            partial_sums[(row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT) + simdgroup_index] = simdgroup_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_index == 0) {
        for (uint row = 0; row < active_rows; row++) {
            float partial = simd_lane < simdgroups_per_output
                ? partial_sums[(row * Q8_0_MAX_SIMDGROUPS_PER_OUTPUT) + simd_lane]
                : 0.0f;
            float reduced_sum = simd_sum(partial);
            if (simd_lane == 0) {
                uint output_index = ((row_start + row) * out_features) + output_feature;
                output[output_index] = reduced_sum + residual[output_index];
            }
        }
    }
}

kernel void q8_0_transposed_matvec_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    constant uint& blocks_per_input_row [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_values = row_count * out_features;
    uint output_index = gid / Q2_K_SIMD_LANES;
    if (output_index >= output_values) {
        return;
    }

    uint input_row = output_index / out_features;
    uint output_feature = output_index - (input_row * out_features);
    uint output_block = output_feature / Q8_0_BLOCK_VALUES;
    uint output_value_in_block = output_feature - (output_block * Q8_0_BLOCK_VALUES);
    uint input_row_offset = input_row * in_features;
    float sum = 0.0f;

    for (uint input_feature = simd_lane; input_feature < in_features; input_feature += Q2_K_SIMD_LANES) {
        uint block_index = (input_feature * blocks_per_input_row) + output_block;
        uint block_offset = block_index * Q8_0_BLOCK_BYTES;
        float weight_value = q8_0_block_value(weights, block_offset, output_value_in_block);
        sum += input[input_row_offset + input_feature] * weight_value;
    }

    float reduced_sum = simd_sum(sum);
    if (simd_lane == 0) {
        output[output_index] = reduced_sum;
    }
}

kernel void q2_k_packed_heads_transposed_matvec_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& head_count [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& blocks_per_input_row [[buffer(7)]],
    constant uint& blocks_per_head [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint outputs_per_row = head_count * out_features;
    uint output_values = row_count * outputs_per_row;
    uint output_index = gid / Q2_K_SIMD_LANES;
    if (output_index >= output_values) {
        return;
    }

    uint input_row = output_index / outputs_per_row;
    uint output_in_row = output_index - (input_row * outputs_per_row);
    uint head = output_in_row / out_features;
    uint output_feature = output_in_row - (head * out_features);
    uint output_block = output_feature / Q2_K_BLOCK_VALUES;
    uint output_value_in_block = output_feature - (output_block * Q2_K_BLOCK_VALUES);
    uint input_row_offset = input_row * in_features;
    uint head_block_offset = head * blocks_per_head;
    float sum = 0.0f;

    for (uint input_feature = simd_lane; input_feature < in_features; input_feature += Q2_K_SIMD_LANES) {
        uint block_index = head_block_offset
            + (input_feature * blocks_per_input_row)
            + output_block;
        uint block_offset = block_index * Q2_K_BLOCK_BYTES;
        float weight_value = q2_k_block_value(weights, block_offset, output_value_in_block);
        sum += input[input_row_offset + input_feature] * weight_value;
    }

    float reduced_sum = simd_sum(sum);
    if (simd_lane == 0) {
        output[output_index] = reduced_sum;
    }
}

kernel void q8_0_packed_heads_transposed_matvec_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& head_count [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& blocks_per_input_row [[buffer(7)]],
    constant uint& blocks_per_head [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint outputs_per_row = head_count * out_features;
    uint output_values = row_count * outputs_per_row;
    uint output_index = gid / Q2_K_SIMD_LANES;
    if (output_index >= output_values) {
        return;
    }

    uint input_row = output_index / outputs_per_row;
    uint output_in_row = output_index - (input_row * outputs_per_row);
    uint head = output_in_row / out_features;
    uint output_feature = output_in_row - (head * out_features);
    uint output_block = output_feature / Q8_0_BLOCK_VALUES;
    uint output_value_in_block = output_feature - (output_block * Q8_0_BLOCK_VALUES);
    uint input_row_offset = input_row * in_features;
    uint head_block_offset = head * blocks_per_head;
    float sum = 0.0f;

    for (uint input_feature = simd_lane; input_feature < in_features; input_feature += Q2_K_SIMD_LANES) {
        uint block_index = head_block_offset
            + (input_feature * blocks_per_input_row)
            + output_block;
        uint block_offset = block_index * Q8_0_BLOCK_BYTES;
        float weight_value = q8_0_block_value(weights, block_offset, output_value_in_block);
        sum += input[input_row_offset + input_feature] * weight_value;
    }

    float reduced_sum = simd_sum(sum);
    if (simd_lane == 0) {
        output[output_index] = reduced_sum;
    }
}

// Applies one independently packed Q8_0 matrix per attention head.
//
// Input:   [row_count, head_count, in_features]
// Weight:  [head_count, out_features, in_features] in Q8_0 row blocks
// Output:  [row_count, head_count, out_features]
//
// This is the native building block for absorbed MLA: K_b projects each
// head's no-RoPE query into latent space and V_b projects the attended latent
// back into that head's value space.
kernel void q8_0_packed_heads_matvec_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& head_count [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& blocks_per_row [[buffer(7)]],
    constant uint& blocks_per_head [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint outputs_per_row = head_count * out_features;
    uint output_values = row_count * outputs_per_row;
    uint output_index = gid / Q2_K_SIMD_LANES;
    if (output_index >= output_values) {
        return;
    }

    uint input_row = output_index / outputs_per_row;
    uint output_in_row = output_index - (input_row * outputs_per_row);
    uint head = output_in_row / out_features;
    uint output_feature = output_in_row - (head * out_features);
    uint input_row_offset = ((input_row * head_count + head) * in_features);
    uint weight_block_offset = (head * blocks_per_head)
        + (output_feature * blocks_per_row);
    float sum = 0.0f;

    for (uint block_in_row = 0; block_in_row < blocks_per_row; block_in_row++) {
        uint block_offset = (weight_block_offset + block_in_row) * Q8_0_BLOCK_BYTES;
        uint input_index = input_row_offset
            + (block_in_row * Q8_0_BLOCK_VALUES)
            + simd_lane;
        sum += input[input_index] * q8_0_block_value(weights, block_offset, simd_lane);
    }

    float reduced_sum = simd_sum(sum);
    if (simd_lane == 0) {
        output[output_index] = reduced_sum;
    }
}

kernel void argmax_f32_kernel(
    const device float* scores [[buffer(0)]],
    device uint* token_id [[buffer(1)]],
    device float* token_score [[buffer(2)]],
    constant uint& value_count [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (value_count == 0) {
        return;
    }

    float best_score = -3.402823466e+38F;
    uint best_id = 0xffffffff;

    for (uint index = tid; index < value_count; index += ARGMAX_THREADS) {
        float score = scores[index];
        if (score > best_score || (score == best_score && index < best_id)) {
            best_id = index;
            best_score = score;
        }
    }

    threadgroup float partial_scores[ARGMAX_THREADS];
    threadgroup uint partial_ids[ARGMAX_THREADS];
    partial_scores[tid] = best_score;
    partial_ids[tid] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float group_best_score = partial_scores[0];
        uint group_best_id = partial_ids[0];
        for (uint index = 1; index < ARGMAX_THREADS; index++) {
            float score = partial_scores[index];
            uint id = partial_ids[index];
            if (score > group_best_score || (score == group_best_score && id < group_best_id)) {
                group_best_score = score;
                group_best_id = id;
            }
        }
        token_id[0] = group_best_id;
        token_score[0] = group_best_score;
    }
}

kernel void argmax_rows_f32_kernel(
    const device float* scores [[buffer(0)]],
    device uint* token_ids [[buffer(1)]],
    device float* token_scores [[buffer(2)]],
    constant uint& row_width [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (row_width == 0) {
        return;
    }

    uint row_offset = row * row_width;
    float best_score = -3.402823466e+38F;
    uint best_id = 0xffffffff;

    for (uint index = tid; index < row_width; index += ARGMAX_THREADS) {
        float score = scores[row_offset + index];
        if (score > best_score || (score == best_score && index < best_id)) {
            best_id = index;
            best_score = score;
        }
    }

    threadgroup float partial_scores[ARGMAX_THREADS];
    threadgroup uint partial_ids[ARGMAX_THREADS];
    partial_scores[tid] = best_score;
    partial_ids[tid] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float group_best_score = partial_scores[0];
        uint group_best_id = partial_ids[0];
        for (uint index = 1; index < ARGMAX_THREADS; index++) {
            float score = partial_scores[index];
            uint id = partial_ids[index];
            if (score > group_best_score || (score == group_best_score && id < group_best_id)) {
                group_best_score = score;
                group_best_id = id;
            }
        }
        token_ids[row] = group_best_id;
        token_scores[row] = group_best_score;
    }
}

kernel void argmax_candidates_f32_kernel(
    const device uint* candidate_ids [[buffer(0)]],
    const device float* candidate_scores [[buffer(1)]],
    device uint* token_id [[buffer(2)]],
    device float* token_score [[buffer(3)]],
    constant uint& candidate_count [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]]
) {
    if (candidate_count == 0u) {
        return;
    }

    float best_score = -3.402823466e+38F;
    uint best_id = 0xffffffffu;
    for (uint index = tid; index < candidate_count; index += ARGMAX_THREADS) {
        float score = candidate_scores[index];
        uint id = candidate_ids[index];
        if (score > best_score || (score == best_score && id < best_id)) {
            best_score = score;
            best_id = id;
        }
    }

    threadgroup float partial_scores[ARGMAX_THREADS];
    threadgroup uint partial_ids[ARGMAX_THREADS];
    partial_scores[tid] = best_score;
    partial_ids[tid] = best_id;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0u) {
        float group_best_score = partial_scores[0];
        uint group_best_id = partial_ids[0];
        for (uint index = 1u; index < ARGMAX_THREADS; index++) {
            float score = partial_scores[index];
            uint id = partial_ids[index];
            if (score > group_best_score
                || (score == group_best_score && id < group_best_id)) {
                group_best_score = score;
                group_best_id = id;
            }
        }
        token_id[0] = group_best_id;
        token_score[0] = group_best_score;
    }
}
