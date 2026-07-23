#include <metal_stdlib>

using namespace metal;

constant uint W4_SIMD_LANES = 32;
constant uint W4_VALUES_PER_WORD = 8;

static inline float w4_bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

static inline int w4_signed_value(uint packed_word, uint value_index) {
    uint shift = (value_index % W4_VALUES_PER_WORD) * 4;
    // compressed-tensors stores q + 8, not a two's-complement nibble.
    return int((packed_word >> shift) & 0x0f) - 8;
}

static inline float w4_row_dot_partial(
    const device uint* packed,
    const device ushort* scales,
    const device float* input,
    uint input_row_offset,
    uint output_feature,
    uint packed_words_per_row,
    uint groups_per_row,
    uint group_size,
    uint simd_lane
) {
    float sum = 0.0f;
    uint packed_row_offset = output_feature * packed_words_per_row;
    uint scale_row_offset = output_feature * groups_per_row;

    for (uint group_index = 0; group_index < groups_per_row; group_index++) {
        uint value_index = (group_index * group_size) + simd_lane;
        uint packed_word = packed[packed_row_offset + (value_index / W4_VALUES_PER_WORD)];
        int quantized = w4_signed_value(packed_word, value_index);
        float scale = w4_bf16_to_f32(scales[scale_row_offset + group_index]);
        sum += input[input_row_offset + value_index] * (float(quantized) * scale);
    }
    return sum;
}

kernel void w4_groupwise_matvec_f32_kernel(
    const device uint* packed [[buffer(0)]],
    const device ushort* scales [[buffer(1)]],
    const device float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    constant uint& packed_words_per_row [[buffer(7)]],
    constant uint& groups_per_row [[buffer(8)]],
    constant uint& group_size [[buffer(9)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_index = gid / W4_SIMD_LANES;
    uint output_count = row_count * out_features;
    if (output_index >= output_count) {
        return;
    }

    uint input_row = output_index / out_features;
    uint output_feature = output_index - (input_row * out_features);
    float partial = w4_row_dot_partial(
        packed,
        scales,
        input,
        input_row * in_features,
        output_feature,
        packed_words_per_row,
        groups_per_row,
        group_size,
        simd_lane
    );
    float sum = simd_sum(partial);
    if (simd_lane == 0) {
        output[output_index] = sum;
    }
}

kernel void w4_groupwise_gate_up_swiglu_f32_kernel(
    const device uint* gate_packed [[buffer(0)]],
    const device ushort* gate_scales [[buffer(1)]],
    const device uint* up_packed [[buffer(2)]],
    const device ushort* up_scales [[buffer(3)]],
    const device float* input [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& row_count [[buffer(6)]],
    constant uint& in_features [[buffer(7)]],
    constant uint& out_features [[buffer(8)]],
    constant uint& packed_words_per_row [[buffer(9)]],
    constant uint& groups_per_row [[buffer(10)]],
    constant uint& group_size [[buffer(11)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_index = gid / W4_SIMD_LANES;
    uint output_count = row_count * out_features;
    if (output_index >= output_count) {
        return;
    }

    uint input_row = output_index / out_features;
    uint output_feature = output_index - (input_row * out_features);
    uint input_row_offset = input_row * in_features;
    float gate = w4_row_dot_partial(
        gate_packed,
        gate_scales,
        input,
        input_row_offset,
        output_feature,
        packed_words_per_row,
        groups_per_row,
        group_size,
        simd_lane
    );
    float up = w4_row_dot_partial(
        up_packed,
        up_scales,
        input,
        input_row_offset,
        output_feature,
        packed_words_per_row,
        groups_per_row,
        group_size,
        simd_lane
    );
    gate = simd_sum(gate);
    up = simd_sum(up);
    if (simd_lane == 0) {
        output[output_index] = (gate / (1.0f + exp(-gate))) * up;
    }
}

kernel void w4_groupwise_expert_gate_up_swiglu_f32_kernel(
    const device ulong* gate_packed_addresses [[buffer(0)]],
    const device ulong* gate_scale_addresses [[buffer(1)]],
    const device ulong* up_packed_addresses [[buffer(2)]],
    const device ulong* up_scale_addresses [[buffer(3)]],
    const device float* input [[buffer(4)]],
    const device uint* assignment_indices [[buffer(5)]],
    const device uint* assignment_groups [[buffer(6)]],
    device float* gated [[buffer(7)]],
    constant uint& token_count [[buffer(8)]],
    constant uint& top_k [[buffer(9)]],
    constant uint& ready_assignment_count [[buffer(10)]],
    constant uint& in_features [[buffer(11)]],
    constant uint& out_features [[buffer(12)]],
    constant uint& packed_words_per_row [[buffer(13)]],
    constant uint& groups_per_row [[buffer(14)]],
    constant uint& group_size [[buffer(15)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint assignment_output_index = gid / W4_SIMD_LANES;
    uint output_count = ready_assignment_count * out_features;
    if (assignment_output_index >= output_count) {
        return;
    }

    uint local_assignment = assignment_output_index / out_features;
    uint output_feature = assignment_output_index
        - (local_assignment * out_features);
    uint group = assignment_groups[local_assignment];
    if (gate_packed_addresses[group] == 0 || gate_scale_addresses[group] == 0
        || up_packed_addresses[group] == 0 || up_scale_addresses[group] == 0) {
        return;
    }
    const device uint* gate_packed =
        reinterpret_cast<device const uint*>(gate_packed_addresses[group]);
    const device ushort* gate_scales =
        reinterpret_cast<device const ushort*>(gate_scale_addresses[group]);
    const device uint* up_packed =
        reinterpret_cast<device const uint*>(up_packed_addresses[group]);
    const device ushort* up_scales =
        reinterpret_cast<device const ushort*>(up_scale_addresses[group]);

    uint assignment = assignment_indices[local_assignment];
    uint token = assignment / top_k;
    if (token >= token_count) {
        return;
    }
    uint input_row_offset = token * in_features;
    float gate = w4_row_dot_partial(
        gate_packed,
        gate_scales,
        input,
        input_row_offset,
        output_feature,
        packed_words_per_row,
        groups_per_row,
        group_size,
        simd_lane
    );
    float up = w4_row_dot_partial(
        up_packed,
        up_scales,
        input,
        input_row_offset,
        output_feature,
        packed_words_per_row,
        groups_per_row,
        group_size,
        simd_lane
    );
    gate = simd_sum(gate);
    up = simd_sum(up);
    if (simd_lane == 0) {
        gated[(local_assignment * out_features) + output_feature] =
            (gate / (1.0f + exp(-gate))) * up;
    }
}

kernel void w4_groupwise_expert_down_f32_kernel(
    const device ulong* packed_addresses [[buffer(0)]],
    const device ulong* scale_addresses [[buffer(1)]],
    const device float* gated [[buffer(2)]],
    const device uint* assignment_indices [[buffer(3)]],
    const device uint* assignment_groups [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& assignment_count [[buffer(6)]],
    constant uint& ready_assignment_count [[buffer(7)]],
    constant uint& in_features [[buffer(8)]],
    constant uint& out_features [[buffer(9)]],
    constant uint& packed_words_per_row [[buffer(10)]],
    constant uint& groups_per_row [[buffer(11)]],
    constant uint& group_size [[buffer(12)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint assignment_output_index = gid / W4_SIMD_LANES;
    uint output_count = ready_assignment_count * out_features;
    if (assignment_output_index >= output_count) {
        return;
    }

    uint local_assignment = assignment_output_index / out_features;
    uint output_feature = assignment_output_index
        - (local_assignment * out_features);
    uint group = assignment_groups[local_assignment];
    if (packed_addresses[group] == 0 || scale_addresses[group] == 0) {
        return;
    }
    const device uint* packed =
        reinterpret_cast<device const uint*>(packed_addresses[group]);
    const device ushort* scales =
        reinterpret_cast<device const ushort*>(scale_addresses[group]);

    uint assignment = assignment_indices[local_assignment];
    if (assignment >= assignment_count) {
        return;
    }
    float value = w4_row_dot_partial(
        packed,
        scales,
        gated,
        local_assignment * in_features,
        output_feature,
        packed_words_per_row,
        groups_per_row,
        group_size,
        simd_lane
    );
    value = simd_sum(value);
    if (simd_lane == 0) {
        output[(assignment * out_features) + output_feature] = value;
    }
}
