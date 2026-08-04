constant uint Q3_K_BLOCK_BYTES = 110u;
constant uint LAGUNA_SIMDGROUPS_PER_THREADGROUP = 2u;
constant uint LAGUNA_Q3_ROWS_PER_SIMDGROUP = 4u;
constant uint LAGUNA_Q2_ROWS_PER_SIMDGROUP = 4u;

// Decodes one 256-value Q2_K block of one row against a lane's activation
// window.
//
// The four 2-bit fields of a quant byte are left in place rather than shifted
// down one at a time: each field keeps its 1x/4x/16x/64x magnitude and the
// shift is folded into the group scale at the end. `value_sums` carries the
// plain activation sums that the block minimum needs, so the minimum costs one
// multiply per group instead of one per value.
//
// Every offset here is even (blocks are 84 bytes and GGUF aligns tensor data
// to at least 8), so the quant pair and the two f16 block constants can be
// read with 2-byte-aligned vector loads instead of byte at a time.
static inline float laguna_q2_k_block_dot(
    const device uchar* weights,
    uint block_offset,
    uint lane_scale_offset,
    uint lane_quant_offset,
    thread const float* values,
    float4 value_sums
) {
    ushort4 packed = ushort4(*reinterpret_cast<const device packed_ushort4*>(
        weights + block_offset + lane_quant_offset));
    float4 low = float4(0.0f);
    float4 high = float4(0.0f);
    for (uint pair = 0u; pair < 4u; pair++) {
        uint index = pair * 2u;
        ushort bits = packed[pair];
        low[0] += values[index] * float(bits & 0x0003);
        high[0] += values[index + 1] * float(bits & 0x0300);
        low[1] += values[index + 8] * float(bits & 0x000c);
        high[1] += values[index + 9] * float(bits & 0x0c00);
        low[2] += values[index + 16] * float(bits & 0x0030);
        high[2] += values[index + 17] * float(bits & 0x3000);
        low[3] += values[index + 24] * float(bits & 0x00c0);
        high[3] += values[index + 25] * float(bits & 0xc000);
    }

    // The scale nibbles this lane needs sit at +0, +2, +4 and +6, so two
    // 4-byte reads cover them.
    const device packed_uchar4* scale_bytes =
        reinterpret_cast<const device packed_uchar4*>(
            weights + block_offset + lane_scale_offset);
    uchar4 first = uchar4(scale_bytes[0]);
    uchar4 second = uchar4(scale_bytes[1]);
    uchar4 scales = uchar4(first[0], first[2], second[0], second[2]);

    half2 constants = half2(*reinterpret_cast<const device packed_half2*>(
        weights + block_offset + Q2_K_SCALE_BYTES + Q2_K_QUANT_BYTES));
    float d = float(constants[0]);
    float dmin = float(constants[1]) / 16.0f;
    float4 scaled = low + (high / 256.0f);

    return d * (
        scaled[0] * float(scales[0] & 0x0f)
        + scaled[1] * float(scales[1] & 0x0f) * (1.0f / 4.0f)
        + scaled[2] * float(scales[2] & 0x0f) * (1.0f / 16.0f)
        + scaled[3] * float(scales[3] & 0x0f) * (1.0f / 64.0f)
    ) - dmin * (
        value_sums[0] * float(scales[0] & 0xf0)
        + value_sums[1] * float(scales[1] & 0xf0)
        + value_sums[2] * float(scales[2] & 0xf0)
        + value_sums[3] * float(scales[3] & 0xf0)
    );
}

// Splits a simdgroup eight ways across a Q2_K block and four ways across the
// blocks of a row, which is the partition `laguna_q2_k_block_dot` expects.
struct LagunaQ2LanePartition {
    ushort block_lane;
    uint scale_offset;
    uint quant_offset;
    uint input_offset;
};

static inline LagunaQ2LanePartition laguna_q2_k_lane_partition(ushort simd_lane) {
    ushort lane_in_block = simd_lane % 8u;
    ushort quant_half = lane_in_block / 4u;
    ushort quant_quarter = lane_in_block % 4u;
    return LagunaQ2LanePartition{
        ushort(simd_lane / 8u),
        8u * uint(quant_half) + uint(quant_quarter / 2u),
        Q2_K_SCALE_BYTES + 32u * uint(quant_half) + 8u * uint(quant_quarter),
        128u * uint(quant_half) + 8u * uint(quant_quarter),
    };
}

// Four consecutive Q2_K rows of two matrices against one activation window.
//
// Laguna's routed gate and up projections address the same expert rows, so a
// single activation load drives eight dot products instead of the one the
// untiled kernel managed.
static inline void laguna_q2_k_dot4x2(
    const device uchar* rows_a,
    const device uchar* rows_b,
    uint row_bytes,
    uint valid_rows,
    uint blocks_per_row,
    const device float* input,
    ushort simd_lane,
    thread float4& sums_a,
    thread float4& sums_b
) {
    sums_a = float4(0.0f);
    sums_b = float4(0.0f);
    LagunaQ2LanePartition lane = laguna_q2_k_lane_partition(simd_lane);

    for (uint block = lane.block_lane; block < blocks_per_row; block += 4u) {
        const device float* block_input =
            input + block * Q2_K_BLOCK_VALUES + lane.input_offset;
        float values[32];
        float4 value_sums = float4(0.0f);
        for (uint index = 0u; index < 8u; index++) {
            values[index] = block_input[index];
            values[index + 8] = block_input[index + 32];
            values[index + 16] = block_input[index + 64];
            values[index + 24] = block_input[index + 96];
            value_sums[0] += values[index];
            value_sums[1] += values[index + 8];
            value_sums[2] += values[index + 16];
            value_sums[3] += values[index + 24];
        }

        uint block_offset = block * Q2_K_BLOCK_BYTES;
        for (uint row = 0u; row < LAGUNA_Q2_ROWS_PER_SIMDGROUP; row++) {
            if (row >= valid_rows) {
                break;
            }
            uint row_offset = block_offset + row * row_bytes;
            sums_a[row] += laguna_q2_k_block_dot(
                rows_a, row_offset, lane.scale_offset, lane.quant_offset,
                values, value_sums);
            sums_b[row] += laguna_q2_k_block_dot(
                rows_b, row_offset, lane.scale_offset, lane.quant_offset,
                values, value_sums);
        }
    }
}

// Four consecutive Q2_K rows of a single matrix. The routed down projection
// has no second matrix to pair with.
static inline float4 laguna_q2_k_dot4(
    const device uchar* rows,
    uint row_bytes,
    uint valid_rows,
    uint blocks_per_row,
    const device float* input,
    ushort simd_lane
) {
    float4 sums = float4(0.0f);
    LagunaQ2LanePartition lane = laguna_q2_k_lane_partition(simd_lane);

    for (uint block = lane.block_lane; block < blocks_per_row; block += 4u) {
        const device float* block_input =
            input + block * Q2_K_BLOCK_VALUES + lane.input_offset;
        float values[32];
        float4 value_sums = float4(0.0f);
        for (uint index = 0u; index < 8u; index++) {
            values[index] = block_input[index];
            values[index + 8] = block_input[index + 32];
            values[index + 16] = block_input[index + 64];
            values[index + 24] = block_input[index + 96];
            value_sums[0] += values[index];
            value_sums[1] += values[index + 8];
            value_sums[2] += values[index + 16];
            value_sums[3] += values[index + 24];
        }

        uint block_offset = block * Q2_K_BLOCK_BYTES;
        for (uint row = 0u; row < LAGUNA_Q2_ROWS_PER_SIMDGROUP; row++) {
            if (row >= valid_rows) {
                break;
            }
            sums[row] += laguna_q2_k_block_dot(
                rows, block_offset + row * row_bytes,
                lane.scale_offset, lane.quant_offset, values, value_sums);
        }
    }
    return sums;
}

struct LagunaQ3KBlock {
    uchar high_mask[32];
    uchar quants[64];
    uchar scales[12];
    half d;
};

// Four adjacent Q3_K rows reuse each activation load. This layout-specific
// implementation follows the GGML block organization used by Laguna.
static inline float4 laguna_q3_k_dot4(
    const device uchar* rows,
    uint row_bytes,
    uint valid_rows,
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
    float4 sum_1 = float4(0.0f);
    float4 sum_2 = float4(0.0f);

    for (int block = block_lane; block < block_count; block += 4) {
        float values[32];
        for (short index = 0; index < 8; index++) {
            values[index +  0] = input_part_values[index +  0];
            values[index +  8] = input_part_values[index + 16];
            values[index + 16] = input_part_values[index + 32];
            values[index + 24] = input_part_values[index + 48];
        }

        for (short row = 0; row < short(LAGUNA_Q3_ROWS_PER_SIMDGROUP); row++) {
            if (uint(row) >= valid_rows) {
                break;
            }
            const device LagunaQ3KBlock* row_blocks =
                reinterpret_cast<const device LagunaQ3KBlock*>(
                    rows + uint(row) * row_bytes);
            // Each of these spans is eight contiguous bytes that the loop below
            // walks two bytes at a time. Block starts are only 2-byte aligned,
            // so `packed_ushort4` is the widest legal load; it replaces four
            // scalar loads per span.
            ushort4 quants = ushort4(
                *reinterpret_cast<const device packed_ushort4*>(
                    row_blocks[block].quants + quant_offset));
            ushort4 quants_upper = ushort4(
                *reinterpret_cast<const device packed_ushort4*>(
                    row_blocks[block].quants + quant_offset + 16));
            ushort4 high = ushort4(
                *reinterpret_cast<const device packed_ushort4*>(
                    row_blocks[block].high_mask + lane_offset));
            ushort4 high_upper = ushort4(
                *reinterpret_cast<const device packed_ushort4*>(
                    row_blocks[block].high_mask + lane_offset + 16));
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
                ushort high_bits = high[index / 2];
                s1 += values[index + 0] * float(quant & low_masks[quant_part / 2][0]);
                s2 += values[index + 1] * float(quant & low_masks[quant_part / 2][1]);
                s3 += ((high_bits & high_mask[0]) ? 0.0f : values[index + 0])
                    + ((high_bits & high_mask[1]) ? 0.0f : values[index + 1]);
                s4 += values[index + 16] * float(quant & low_masks[quant_part / 2][2]);
                s5 += values[index + 17] * float(quant & low_masks[quant_part / 2][3]);
                s6 += ((high_bits & high_mask[2]) ? 0.0f : values[index + 16])
                    + ((high_bits & high_mask[3]) ? 0.0f : values[index + 17]);
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
                int quant = quants_upper[index / 2];
                ushort high_bits = high_upper[index / 2];
                s1 += values[index + 8] * float(quant & low_masks[quant_part / 2][0]);
                s2 += values[index + 9] * float(quant & low_masks[quant_part / 2][1]);
                s3 += ((high_bits & high_mask[0]) ? 0.0f : values[index + 8])
                    + ((high_bits & high_mask[1]) ? 0.0f : values[index + 9]);
                s4 += values[index + 24] * float(quant & low_masks[quant_part / 2][2]);
                s5 += values[index + 25] * float(quant & low_masks[quant_part / 2][3]);
                s6 += ((high_bits & high_mask[2]) ? 0.0f : values[index + 24])
                    + ((high_bits & high_mask[3]) ? 0.0f : values[index + 25]);
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
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint simdgroup = threadgroup_position.x
        * LAGUNA_SIMDGROUPS_PER_THREADGROUP
        + uint(simdgroup_index);
    uint row_groups_per_assignment =
        (intermediate_features + LAGUNA_Q2_ROWS_PER_SIMDGROUP - 1u)
        / LAGUNA_Q2_ROWS_PER_SIMDGROUP;
    uint assignment = simdgroup / row_groups_per_assignment;
    if (assignment >= assignment_count) {
        return;
    }
    uint row_group = simdgroup - assignment * row_groups_per_assignment;
    uint row = row_group * LAGUNA_Q2_ROWS_PER_SIMDGROUP;
    uint row_bytes = blocks_per_row * Q2_K_BLOCK_BYTES;
    uint expert_offset = expert_ids[assignment] * expert_stride_bytes
        + row * row_bytes;
    const device float* token_input =
        input + token_indices[assignment] * in_features;
    uint valid_rows = min(
        LAGUNA_Q2_ROWS_PER_SIMDGROUP, intermediate_features - row);

    float4 gate;
    float4 up;
    laguna_q2_k_dot4x2(
        gate_weights + expert_offset,
        up_weights + expert_offset,
        row_bytes,
        valid_rows,
        blocks_per_row,
        token_input,
        simd_lane,
        gate,
        up);

    float routing_weight = expert_weights[assignment];
    for (uint local_row = 0u; local_row < valid_rows; local_row++) {
        float gate_sum = simd_sum(gate[local_row]);
        float up_sum = simd_sum(up[local_row]);
        if (simd_lane == 0u) {
            float silu = gate_sum / (1.0f + exp(-gate_sum));
            intermediate[assignment * intermediate_features + row + local_row] =
                silu * up_sum * routing_weight;
        }
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
    uint valid_rows = min(
        LAGUNA_Q3_ROWS_PER_SIMDGROUP, intermediate_features - row);
    float4 gate = laguna_q3_k_dot4(
        gate_weights + expert_offset, row_bytes, valid_rows, in_features,
        token_input, simd_lane);
    float4 up = laguna_q3_k_dot4(
        up_weights + expert_offset, row_bytes, valid_rows, in_features,
        token_input, simd_lane);

    float routing_weight = expert_weights[assignment];
    for (uint local_row = 0u; local_row < valid_rows; local_row++) {
        float gate_sum = simd_sum(gate[local_row]);
        float up_sum = simd_sum(up[local_row]);
        if (simd_lane == 0u) {
            float silu = gate_sum / (1.0f + exp(-gate_sum));
            intermediate[assignment * intermediate_features + row + local_row] =
                silu * up_sum * routing_weight;
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
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    ushort simd_lane [[thread_index_in_simdgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]]
) {
    uint simdgroup = threadgroup_position.x
        * LAGUNA_SIMDGROUPS_PER_THREADGROUP
        + uint(simdgroup_index);
    uint row_groups_per_token =
        (out_features + LAGUNA_Q2_ROWS_PER_SIMDGROUP - 1u)
        / LAGUNA_Q2_ROWS_PER_SIMDGROUP;
    uint token = simdgroup / row_groups_per_token;
    if (token >= token_count) {
        return;
    }
    uint row_group = simdgroup - token * row_groups_per_token;
    uint row = row_group * LAGUNA_Q2_ROWS_PER_SIMDGROUP;
    uint row_bytes = blocks_per_row * Q2_K_BLOCK_BYTES;
    uint valid_rows = min(LAGUNA_Q2_ROWS_PER_SIMDGROUP, out_features - row);

    // Every selected expert contributes to the same output rows with the same
    // lane partition, so the partial sums add up before the reduction and one
    // simdgroup reduction covers all of top-k.
    float4 total = float4(0.0f);
    for (uint slot = 0u; slot < top_k; slot++) {
        uint assignment = token * top_k + slot;
        uint expert_offset = expert_ids[assignment] * expert_stride_bytes
            + row * row_bytes;
        total += laguna_q2_k_dot4(
            down_weights + expert_offset,
            row_bytes,
            valid_rows,
            blocks_per_row,
            intermediate + assignment * intermediate_features,
            simd_lane);
    }

    for (uint local_row = 0u; local_row < valid_rows; local_row++) {
        float value = simd_sum(total[local_row]);
        if (simd_lane == 0u) {
            output[token * out_features + row + local_row] = value;
        }
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
    uint valid_rows = min(LAGUNA_Q3_ROWS_PER_SIMDGROUP, out_features - row);
    float4 total = float4(0.0f);
    for (uint slot = 0u; slot < top_k; slot++) {
        uint assignment = token * top_k + slot;
        uint expert_offset = expert_ids[assignment] * expert_stride_bytes
            + row * row_bytes;
        total += laguna_q3_k_dot4(
            down_weights + expert_offset,
            row_bytes,
            valid_rows,
            intermediate_features,
            intermediate + assignment * intermediate_features,
            simd_lane);
    }

    for (uint local_row = 0u; local_row < valid_rows; local_row++) {
        float value = simd_sum(total[local_row]);
        if (simd_lane == 0u) {
            output[token * out_features + row + local_row] = value;
        }
    }
}
