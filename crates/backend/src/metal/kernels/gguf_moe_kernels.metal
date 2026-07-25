constant uint Q3_K_BLOCK_BYTES = 110u;
constant uint LAGUNA_SIMDGROUPS_PER_THREADGROUP = 2u;
constant uint LAGUNA_Q3_ROWS_PER_SIMDGROUP = 2u;

struct LagunaQ3KBlock {
    uchar high_mask[32];
    uchar quants[64];
    uchar scales[12];
    half d;
};

// Two adjacent Q3_K rows reuse each activation load. This layout-specific
// implementation follows the GGML block organization used by Laguna.
static inline float2 laguna_q3_k_dot2(
    const device uchar* rows,
    uint row_bytes,
    uint in_features,
    const device float* input,
    ushort simd_lane
) {
    int block_count = int(in_features / Q2_K_BLOCK_VALUES);
    short lane_group = short(simd_lane / 4u);
    short block_lane = short(simd_lane % 4u);
    short input_part = short(lane_group / 4);
    short quant_part = short(2 * ((lane_group % 4) / 2));
    short pair = short(lane_group % 2);
    short lane_offset = short(8 * pair);

    const ushort4 high_masks[4] = {
        {0x0001, 0x0100, 0x0002, 0x0200},
        {0x0004, 0x0400, 0x0008, 0x0800},
        {0x0010, 0x1000, 0x0020, 0x2000},
        {0x0040, 0x4000, 0x0080, 0x8000},
    };
    const int4 low_masks[2] = {
        {0x0003, 0x0300, 0x000c, 0x0c00},
        {0x0030, 0x3000, 0x00c0, 0xc000},
    };
    ushort4 high_mask = high_masks[2 * input_part + quant_part / 2];
    short shift = short(2 * quant_part);
    float high_base_1 = quant_part == 0 ? 4.0f : 64.0f;
    float high_base_2 = 4.0f * high_base_1;
    ushort scale_shift_1 = ushort(4 * input_part);
    ushort scale_shift_2 = ushort(scale_shift_1 + quant_part);
    short quant_offset = short(32 * input_part + lane_offset);
    short input_offset = short(128 * input_part + 32 * quant_part + lane_offset);

    const device float* input_part_values =
        input + block_lane * Q2_K_BLOCK_VALUES + input_offset;
    float2 sum_1 = float2(0.0f);
    float2 sum_2 = float2(0.0f);

    for (int block = block_lane; block < block_count; block += 4) {
        float values[32];
        for (short index = 0; index < 8; index++) {
            values[index +  0] = input_part_values[index +  0];
            values[index +  8] = input_part_values[index + 16];
            values[index + 16] = input_part_values[index + 32];
            values[index + 24] = input_part_values[index + 48];
        }

        for (short row = 0; row < short(LAGUNA_Q3_ROWS_PER_SIMDGROUP); row++) {
            const device LagunaQ3KBlock* row_blocks =
                reinterpret_cast<const device LagunaQ3KBlock*>(
                    rows + uint(row) * row_bytes);
            const device ushort* quants =
                reinterpret_cast<const device ushort*>(
                    row_blocks[block].quants + quant_offset);
            const device ushort* high =
                reinterpret_cast<const device ushort*>(
                    row_blocks[block].high_mask + lane_offset);
            const device ushort* packed_scales =
                reinterpret_cast<const device ushort*>(
                    row_blocks[block].scales);

            uint packed = 0u;
            thread ushort* packed_16 = reinterpret_cast<thread ushort*>(&packed);
            thread const char* scales =
                reinterpret_cast<thread const char*>(&packed);
            packed_16[0] = packed_scales[4];
            packed_16[1] = packed_scales[5];
            uint scale_high =
                ((packed >> scale_shift_2) << 4u) & 0x30303030u;
            packed_16[0] = packed_scales[quant_part + 0];
            packed_16[1] = packed_scales[quant_part + 1];
            packed = ((packed >> scale_shift_1) & 0x0f0f0f0fu)
                | scale_high;

            float s1 = 0.0f;
            float s2 = 0.0f;
            float s3 = 0.0f;
            float s4 = 0.0f;
            float s5 = 0.0f;
            float s6 = 0.0f;
            for (short index = 0; index < 8; index += 2) {
                int quant = quants[index / 2];
                s1 += values[index + 0] * float(quant & low_masks[quant_part / 2][0]);
                s2 += values[index + 1] * float(quant & low_masks[quant_part / 2][1]);
                s3 += ((high[index / 2] & high_mask[0]) ? 0.0f : values[index + 0])
                    + ((high[index / 2] & high_mask[1]) ? 0.0f : values[index + 1]);
                s4 += values[index + 16] * float(quant & low_masks[quant_part / 2][2]);
                s5 += values[index + 17] * float(quant & low_masks[quant_part / 2][3]);
                s6 += ((high[index / 2] & high_mask[2]) ? 0.0f : values[index + 16])
                    + ((high[index / 2] & high_mask[3]) ? 0.0f : values[index + 17]);
            }

            float d = float(row_blocks[block].d);
            float d1 = d * (s1 + (1.0f / 256.0f) * s2 - s3 * high_base_1);
            float d2 = d * (s4 + (1.0f / 256.0f) * s5 - s6 * high_base_2);
            sum_1[row] += d1 * (float(scales[0]) - 32.0f);
            sum_2[row] += d2 * (float(scales[2]) - 32.0f);

            s1 = 0.0f;
            s2 = 0.0f;
            s3 = 0.0f;
            s4 = 0.0f;
            s5 = 0.0f;
            s6 = 0.0f;
            for (short index = 0; index < 8; index += 2) {
                int quant = quants[index / 2 + 8];
                s1 += values[index + 8] * float(quant & low_masks[quant_part / 2][0]);
                s2 += values[index + 9] * float(quant & low_masks[quant_part / 2][1]);
                s3 += ((high[index / 2 + 8] & high_mask[0]) ? 0.0f : values[index + 8])
                    + ((high[index / 2 + 8] & high_mask[1]) ? 0.0f : values[index + 9]);
                s4 += values[index + 24] * float(quant & low_masks[quant_part / 2][2]);
                s5 += values[index + 25] * float(quant & low_masks[quant_part / 2][3]);
                s6 += ((high[index / 2 + 8] & high_mask[2]) ? 0.0f : values[index + 24])
                    + ((high[index / 2 + 8] & high_mask[3]) ? 0.0f : values[index + 25]);
            }

            float e1 = d * (s1 + (1.0f / 256.0f) * s2 - s3 * high_base_1);
            float e2 = d * (s4 + (1.0f / 256.0f) * s5 - s6 * high_base_2);
            sum_1[row] += e1 * (float(scales[1]) - 32.0f);
            sum_2[row] += e2 * (float(scales[3]) - 32.0f);
        }

        input_part_values += 4 * Q2_K_BLOCK_VALUES;
    }

    return (sum_1 + 0.25f * sum_2) / float(1 << shift);
}

kernel void laguna_q2_expert_gate_up_f32_kernel(
    const device uchar* gate_weights [[buffer(0)]],
    const device uchar* up_weights [[buffer(1)]],
    const device float* input [[buffer(2)]],
    const device uint* token_indices [[buffer(3)]],
    const device uint* expert_ids [[buffer(4)]],
    const device float* expert_weights [[buffer(5)]],
    device float* intermediate [[buffer(6)]],
    constant uint& assignment_count [[buffer(7)]],
    constant uint& in_features [[buffer(8)]],
    constant uint& intermediate_features [[buffer(9)]],
    constant uint& blocks_per_row [[buffer(10)]],
    constant uint& expert_stride_bytes [[buffer(11)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_index = gid / 32u;
    if (output_index >= assignment_count * intermediate_features) {
        return;
    }
    uint assignment = output_index / intermediate_features;
    uint row = output_index - assignment * intermediate_features;
    uint input_offset = token_indices[assignment] * in_features;
    uint expert_offset = expert_ids[assignment] * expert_stride_bytes;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint block = 0u; block < blocks_per_row; block++) {
        uint weight_offset = expert_offset
            + (row * blocks_per_row + block) * Q2_K_BLOCK_BYTES;
        uint input_block = input_offset + block * Q2_K_BLOCK_VALUES;
        gate += q2_k_block_dot_partial(
            gate_weights, input, input_block, weight_offset, simd_lane);
        up += q2_k_block_dot_partial(
            up_weights, input, input_block, weight_offset, simd_lane);
    }
    gate = simd_sum(gate);
    up = simd_sum(up);
    if (simd_lane == 0u) {
        float silu = gate / (1.0f + exp(-gate));
        intermediate[output_index] = silu * up * expert_weights[assignment];
    }
}

kernel void laguna_q3_expert_gate_up_f32_kernel(
    const device uchar* gate_weights [[buffer(0)]],
    const device uchar* up_weights [[buffer(1)]],
    const device float* input [[buffer(2)]],
    const device uint* token_indices [[buffer(3)]],
    const device uint* expert_ids [[buffer(4)]],
    const device float* expert_weights [[buffer(5)]],
    device float* intermediate [[buffer(6)]],
    constant uint& assignment_count [[buffer(7)]],
    constant uint& in_features [[buffer(8)]],
    constant uint& intermediate_features [[buffer(9)]],
    constant uint& blocks_per_row [[buffer(10)]],
    constant uint& expert_stride_bytes [[buffer(11)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint simdgroup = threadgroup_position.x
        * LAGUNA_SIMDGROUPS_PER_THREADGROUP
        + uint(simdgroup_index);
    uint row_groups_per_assignment =
        (intermediate_features + LAGUNA_Q3_ROWS_PER_SIMDGROUP - 1u)
        / LAGUNA_Q3_ROWS_PER_SIMDGROUP;
    uint assignment = simdgroup / row_groups_per_assignment;
    if (assignment >= assignment_count) {
        return;
    }
    uint row_group = simdgroup - assignment * row_groups_per_assignment;
    uint row = row_group * LAGUNA_Q3_ROWS_PER_SIMDGROUP;
    uint row_bytes = blocks_per_row * Q3_K_BLOCK_BYTES;
    uint expert_offset = expert_ids[assignment] * expert_stride_bytes
        + row * row_bytes;
    const device float* token_input =
        input + token_indices[assignment] * in_features;
    float2 gate = laguna_q3_k_dot2(
        gate_weights + expert_offset,
        row_bytes,
        in_features,
        token_input,
        simd_lane);
    float2 up = laguna_q3_k_dot2(
        up_weights + expert_offset,
        row_bytes,
        in_features,
        token_input,
        simd_lane);

    for (uint local_row = 0u;
         local_row < LAGUNA_Q3_ROWS_PER_SIMDGROUP
             && row + local_row < intermediate_features;
         local_row++) {
        float gate_sum = simd_sum(gate[local_row]);
        float up_sum = simd_sum(up[local_row]);
        if (simd_lane == 0u) {
            float silu = gate_sum / (1.0f + exp(-gate_sum));
            intermediate[assignment * intermediate_features + row + local_row] =
                silu * up_sum * expert_weights[assignment];
        }
    }
}

kernel void laguna_q2_expert_down_sum_f32_kernel(
    const device uchar* down_weights [[buffer(0)]],
    const device uint* expert_ids [[buffer(1)]],
    const device float* intermediate [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& token_count [[buffer(4)]],
    constant uint& top_k [[buffer(5)]],
    constant uint& intermediate_features [[buffer(6)]],
    constant uint& out_features [[buffer(7)]],
    constant uint& blocks_per_row [[buffer(8)]],
    constant uint& expert_stride_bytes [[buffer(9)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_index = gid / 32u;
    if (output_index >= token_count * out_features) {
        return;
    }
    uint token = output_index / out_features;
    uint row = output_index - token * out_features;
    float total = 0.0f;
    for (uint slot = 0u; slot < top_k; slot++) {
        uint assignment = token * top_k + slot;
        uint expert_offset = expert_ids[assignment] * expert_stride_bytes;
        uint input_offset = assignment * intermediate_features;
        float partial = 0.0f;
        for (uint block = 0u; block < blocks_per_row; block++) {
            uint weight_offset = expert_offset
                + (row * blocks_per_row + block) * Q2_K_BLOCK_BYTES;
            partial += q2_k_block_dot_partial(
                down_weights,
                intermediate,
                input_offset + block * Q2_K_BLOCK_VALUES,
                weight_offset,
                simd_lane
            );
        }
        total += simd_sum(partial);
    }
    if (simd_lane == 0u) {
        output[output_index] = total;
    }
}

kernel void laguna_q3_expert_down_sum_f32_kernel(
    const device uchar* down_weights [[buffer(0)]],
    const device uint* expert_ids [[buffer(1)]],
    const device float* intermediate [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& token_count [[buffer(4)]],
    constant uint& top_k [[buffer(5)]],
    constant uint& intermediate_features [[buffer(6)]],
    constant uint& out_features [[buffer(7)]],
    constant uint& blocks_per_row [[buffer(8)]],
    constant uint& expert_stride_bytes [[buffer(9)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint simdgroup = threadgroup_position.x
        * LAGUNA_SIMDGROUPS_PER_THREADGROUP
        + uint(simdgroup_index);
    uint row_groups_per_token =
        (out_features + LAGUNA_Q3_ROWS_PER_SIMDGROUP - 1u)
        / LAGUNA_Q3_ROWS_PER_SIMDGROUP;
    uint token = simdgroup / row_groups_per_token;
    if (token >= token_count) {
        return;
    }
    uint row_group = simdgroup - token * row_groups_per_token;
    uint row = row_group * LAGUNA_Q3_ROWS_PER_SIMDGROUP;
    uint row_bytes = blocks_per_row * Q3_K_BLOCK_BYTES;
    float2 total = float2(0.0f);
    for (uint slot = 0u; slot < top_k; slot++) {
        uint assignment = token * top_k + slot;
        uint expert_offset = expert_ids[assignment] * expert_stride_bytes
            + row * row_bytes;
        total += laguna_q3_k_dot2(
            down_weights + expert_offset,
            row_bytes,
            intermediate_features,
            intermediate + assignment * intermediate_features,
            simd_lane);
    }

    for (uint local_row = 0u;
         local_row < LAGUNA_Q3_ROWS_PER_SIMDGROUP
             && row + local_row < out_features;
         local_row++) {
        float value = simd_sum(total[local_row]);
        if (simd_lane == 0u) {
            output[token * out_features + row + local_row] = value;
        }
    }
}
