#include <metal_stdlib>

using namespace metal;

constant uint BF16_SIMD_LANES = 32;

static inline float bf16_value(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

kernel void bf16_linear_f32_kernel(
    const device ushort* weight [[buffer(0)]],
    const device float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_index = gid / BF16_SIMD_LANES;
    uint output_count = row_count * out_features;
    if (output_index >= output_count) {
        return;
    }

    uint input_row = output_index / out_features;
    uint output_feature = output_index - (input_row * out_features);
    uint input_offset = input_row * in_features;
    uint weight_offset = output_feature * in_features;
    float sum = 0.0f;
    for (uint feature = simd_lane; feature < in_features; feature += BF16_SIMD_LANES) {
        sum += input[input_offset + feature] * bf16_value(weight[weight_offset + feature]);
    }
    sum = simd_sum(sum);
    if (simd_lane == 0) {
        output[output_index] = sum;
    }
}

kernel void bf16_gate_up_swiglu_f32_kernel(
    const device ushort* gate_weight [[buffer(0)]],
    const device ushort* up_weight [[buffer(1)]],
    const device float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint output_index = gid / BF16_SIMD_LANES;
    uint output_count = row_count * out_features;
    if (output_index >= output_count) {
        return;
    }

    uint input_row = output_index / out_features;
    uint output_feature = output_index - (input_row * out_features);
    uint input_offset = input_row * in_features;
    uint weight_offset = output_feature * in_features;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint feature = simd_lane; feature < in_features; feature += BF16_SIMD_LANES) {
        float input_value = input[input_offset + feature];
        gate += input_value * bf16_value(gate_weight[weight_offset + feature]);
        up += input_value * bf16_value(up_weight[weight_offset + feature]);
    }
    gate = simd_sum(gate);
    up = simd_sum(up);
    if (simd_lane == 0) {
        output[output_index] = (gate / (1.0f + exp(-gate))) * up;
    }
}

// Laguna uses four projections from the same normalized hidden row before
// attention: Q, K, V, and one scalar gate per query head. Encoding them as a
// single grid removes three dispatch boundaries while preserving separate
// output buffers for the following Q/K normalization and FP8 attention path.
kernel void laguna_bf16_attention_projections_f32_kernel(
    const device ushort* query_weight [[buffer(0)]],
    const device ushort* key_weight [[buffer(1)]],
    const device ushort* value_weight [[buffer(2)]],
    const device ushort* gate_weight [[buffer(3)]],
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
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint combined_feature_count = query_features + key_features
        + value_features + gate_features;
    uint combined_output_index = gid / BF16_SIMD_LANES;
    uint output_count = row_count * combined_feature_count;
    if (combined_output_index >= output_count) {
        return;
    }

    uint input_row = combined_output_index / combined_feature_count;
    uint combined_feature = combined_output_index
        - (input_row * combined_feature_count);
    uint feature = combined_feature;
    const device ushort* weight = query_weight;
    device float* output = query_output;
    uint output_features = query_features;

    if (combined_feature >= query_features
        && combined_feature < query_features + key_features) {
        feature = combined_feature - query_features;
        weight = key_weight;
        output = key_output;
        output_features = key_features;
    } else if (combined_feature >= query_features + key_features
        && combined_feature < query_features + key_features + value_features) {
        feature = combined_feature - query_features - key_features;
        weight = value_weight;
        output = value_output;
        output_features = value_features;
    } else if (combined_feature >= query_features + key_features + value_features) {
        feature = combined_feature - query_features - key_features - value_features;
        weight = gate_weight;
        output = gate_output;
        output_features = gate_features;
    }

    uint input_offset = input_row * in_features;
    uint weight_offset = feature * in_features;
    float sum = 0.0f;
    for (uint input_feature = simd_lane; input_feature < in_features;
         input_feature += BF16_SIMD_LANES) {
        sum += input[input_offset + input_feature]
            * bf16_value(weight[weight_offset + input_feature]);
    }
    sum = simd_sum(sum);
    if (simd_lane == 0) {
        output[(input_row * output_features) + feature] = sum;
    }
}

kernel void bf16_embedding_f32_kernel(
    const device ushort* embedding [[buffer(0)]],
    const device uint* token_ids [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    constant uint& vocab_size [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_count = token_count * hidden_size;
    if (gid >= output_count) {
        return;
    }
    uint token_index = gid / hidden_size;
    uint hidden_index = gid - (token_index * hidden_size);
    uint token_id = token_ids[token_index];
    if (token_id >= vocab_size) {
        output[gid] = 0.0f;
        return;
    }
    output[gid] = bf16_value(embedding[(token_id * hidden_size) + hidden_index]);
}
