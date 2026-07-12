#include <metal_stdlib>

using namespace metal;

constant uint Q2_K_BLOCK_VALUES = 256;
constant uint Q2_K_BLOCK_BYTES = 84;
constant uint Q2_K_SCALE_BYTES = 16;
constant uint Q2_K_QUANT_BYTES = 64;
constant uint Q2_K_SIMD_LANES = 32;
constant uint ARGMAX_THREADS = 256;
constant uint Q8_0_BLOCK_VALUES = 32;
constant uint Q8_0_BLOCK_BYTES = 34;
constant uint Q8_0_MAX_SIMDGROUPS_PER_OUTPUT = 8;

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

kernel void q2_k_ready_gate_up_swiglu_f32_kernel(
    const device ulong* gate_addresses [[buffer(0)]],
    const device ulong* up_addresses [[buffer(1)]],
    const device float* input [[buffer(2)]],
    const device uint* token_indices [[buffer(3)]],
    const device uint* assignment_indices [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& token_count [[buffer(6)]],
    constant uint& assignment_count [[buffer(7)]],
    constant uint& ready_count [[buffer(8)]],
    constant uint& in_features [[buffer(9)]],
    constant uint& out_features [[buffer(10)]],
    constant uint& blocks_per_row [[buffer(11)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint ready_output_values = ready_count * out_features;
    uint ready_output_index = gid / Q2_K_SIMD_LANES;
    if (ready_output_index >= ready_output_values) {
        return;
    }

    uint ready_assignment = ready_output_index / out_features;
    uint output_feature = ready_output_index - (ready_assignment * out_features);
    uint assignment = assignment_indices[ready_assignment];
    if (assignment >= assignment_count) {
        return;
    }
    uint token = token_indices[assignment];
    if (token >= token_count || gate_addresses[ready_assignment] == 0 || up_addresses[ready_assignment] == 0) {
        return;
    }

    const device uchar* gate_weights =
        reinterpret_cast<device const uchar*>(gate_addresses[ready_assignment]);
    const device uchar* up_weights =
        reinterpret_cast<device const uchar*>(up_addresses[ready_assignment]);
    uint input_row_offset = token * in_features;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint block_in_row = 0; block_in_row < blocks_per_row; block_in_row++) {
        uint block_offset = ((output_feature * blocks_per_row) + block_in_row) * Q2_K_BLOCK_BYTES;
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
        output[(assignment * out_features) + output_feature] = silu_gate * up;
    }
}

kernel void q2_k_ready_matvec_f32_kernel(
    const device ulong* weight_addresses [[buffer(0)]],
    const device float* input [[buffer(1)]],
    const device uint* assignment_indices [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& assignment_count [[buffer(4)]],
    constant uint& ready_count [[buffer(5)]],
    constant uint& in_features [[buffer(6)]],
    constant uint& out_features [[buffer(7)]],
    constant uint& blocks_per_row [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint ready_output_values = ready_count * out_features;
    uint ready_output_index = gid / Q2_K_SIMD_LANES;
    if (ready_output_index >= ready_output_values) {
        return;
    }

    uint ready_assignment = ready_output_index / out_features;
    uint output_feature = ready_output_index - (ready_assignment * out_features);
    uint assignment = assignment_indices[ready_assignment];
    if (assignment >= assignment_count || weight_addresses[ready_assignment] == 0) {
        return;
    }

    const device uchar* weights =
        reinterpret_cast<device const uchar*>(weight_addresses[ready_assignment]);
    uint input_row_offset = assignment * in_features;
    float sum = 0.0f;
    for (uint block_in_row = 0; block_in_row < blocks_per_row; block_in_row++) {
        uint block_offset = ((output_feature * blocks_per_row) + block_in_row) * Q2_K_BLOCK_BYTES;
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
        output[(assignment * out_features) + output_feature] = reduced_sum;
    }
}

kernel void q2_k_ready_slot_gate_up_swiglu_f32_kernel(
    const device uchar* gate_weights [[buffer(0)]],
    const device uchar* up_weights [[buffer(1)]],
    const device float* input [[buffer(2)]],
    const device uint* token_indices [[buffer(3)]],
    const device uint* assignment_indices [[buffer(4)]],
    const device uint* slot_indices [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& token_count [[buffer(7)]],
    constant uint& assignment_count [[buffer(8)]],
    constant uint& ready_count [[buffer(9)]],
    constant uint& in_features [[buffer(10)]],
    constant uint& out_features [[buffer(11)]],
    constant uint& blocks_per_row [[buffer(12)]],
    constant uint& expert_stride_bytes [[buffer(13)]],
    constant uint& slot_count [[buffer(14)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint ready_output_values = ready_count * out_features;
    uint ready_output_index = gid / Q2_K_SIMD_LANES;
    if (ready_output_index >= ready_output_values) {
        return;
    }

    uint ready_assignment = ready_output_index / out_features;
    uint output_feature = ready_output_index - (ready_assignment * out_features);
    uint assignment = assignment_indices[ready_assignment];
    uint slot = slot_indices[ready_assignment];
    if (assignment >= assignment_count || slot >= slot_count) {
        return;
    }
    uint token = token_indices[assignment];
    if (token >= token_count) {
        return;
    }

    uint expert_base = slot * expert_stride_bytes;
    uint input_row_offset = token * in_features;
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
        output[(assignment * out_features) + output_feature] = silu_gate * up;
    }
}

kernel void q2_k_ready_slot_matvec_f32_kernel(
    const device uchar* weights [[buffer(0)]],
    const device float* input [[buffer(1)]],
    const device uint* assignment_indices [[buffer(2)]],
    const device uint* slot_indices [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& assignment_count [[buffer(5)]],
    constant uint& ready_count [[buffer(6)]],
    constant uint& in_features [[buffer(7)]],
    constant uint& out_features [[buffer(8)]],
    constant uint& blocks_per_row [[buffer(9)]],
    constant uint& expert_stride_bytes [[buffer(10)]],
    constant uint& slot_count [[buffer(11)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint ready_output_values = ready_count * out_features;
    uint ready_output_index = gid / Q2_K_SIMD_LANES;
    if (ready_output_index >= ready_output_values) {
        return;
    }

    uint ready_assignment = ready_output_index / out_features;
    uint output_feature = ready_output_index - (ready_assignment * out_features);
    uint assignment = assignment_indices[ready_assignment];
    uint slot = slot_indices[ready_assignment];
    if (assignment >= assignment_count || slot >= slot_count) {
        return;
    }

    uint expert_base = slot * expert_stride_bytes;
    uint input_row_offset = assignment * in_features;
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
        output[(assignment * out_features) + output_feature] = reduced_sum;
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
