#include <metal_stdlib>

using namespace metal;

constant uint QWEN_MAX_VERIFY_ROWS = 9;
constant uint QWEN_COMPACT_DRAFT_PREFIX = 98304;
constant uint QWEN_COMPACT_DRAFT_CONTROL_START = 248044;
constant uint QWEN_COMPACT_DRAFT_CONTROL_END = 248070;
constant uint QWEN_COMPACT_DRAFT_VOCAB =
    QWEN_COMPACT_DRAFT_PREFIX + QWEN_COMPACT_DRAFT_CONTROL_END
    - QWEN_COMPACT_DRAFT_CONTROL_START;

static inline float qwen_bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

static inline ushort qwen_f32_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    uint exponent = bits & 0x7f800000u;
    if (exponent == 0x7f800000u) {
        return ushort(bits >> 16);
    }
    uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return ushort(rounded >> 16);
}

kernel void qwen_bf16_embedding_kernel(
    const device ushort* embedding [[buffer(0)]],
    const device uint* token_ids [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& hidden_size [[buffer(3)]],
    constant uint& output_len [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= output_len) {
        return;
    }
    uint token = gid / hidden_size;
    uint hidden = gid - token * hidden_size;
    output[gid] = embedding[token_ids[token] * hidden_size + hidden];
}

kernel void qwen_bf16_rms_norm_kernel(
    const device ushort* input [[buffer(0)]],
    const device ushort* weight [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (row >= row_count) {
        return;
    }
    uint row_start = row * hidden_size;
    float sum = 0.0f;
    for (uint hidden = lane; hidden < hidden_size; hidden += 32u) {
        float value = qwen_bf16_to_f32(input[row_start + hidden]);
        sum += value * value;
    }
    sum = simd_sum(sum);
    float inverse_rms = rsqrt(sum / float(hidden_size) + eps);
    inverse_rms = simd_broadcast_first(inverse_rms);
    for (uint hidden = lane; hidden < hidden_size; hidden += 32u) {
        float value = qwen_bf16_to_f32(input[row_start + hidden]);
        float scale = 1.0f + qwen_bf16_to_f32(weight[hidden]);
        float normalized = qwen_bf16_to_f32(qwen_f32_to_bf16(value * inverse_rms));
        output[row_start + hidden] = qwen_f32_to_bf16(normalized * scale);
    }
}

kernel void qwen_bf16_rms_norm_standard_kernel(
    const device ushort* input [[buffer(0)]],
    const device ushort* weight [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (row >= row_count) {
        return;
    }
    uint row_start = row * hidden_size;
    float sum = 0.0f;
    for (uint hidden = lane; hidden < hidden_size; hidden += 32u) {
        float value = qwen_bf16_to_f32(input[row_start + hidden]);
        sum += value * value;
    }
    sum = simd_sum(sum);
    float inverse_rms = rsqrt(sum / float(hidden_size) + eps);
    inverse_rms = simd_broadcast_first(inverse_rms);
    for (uint hidden = lane; hidden < hidden_size; hidden += 32u) {
        float value = qwen_bf16_to_f32(input[row_start + hidden]);
        float scale = qwen_bf16_to_f32(weight[hidden]);
        float normalized = qwen_bf16_to_f32(qwen_f32_to_bf16(value * inverse_rms));
        output[row_start + hidden] = qwen_f32_to_bf16(normalized * scale);
    }
}

kernel void qwen_bf16_add_kernel(
    const device ushort* left [[buffer(0)]],
    const device ushort* right [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& element_count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= element_count) {
        return;
    }
    output[gid] = qwen_f32_to_bf16(
        qwen_bf16_to_f32(left[gid]) + qwen_bf16_to_f32(right[gid])
    );
}

kernel void qwen_bf16_concat_last_kernel(
    const device ushort* left [[buffer(0)]],
    const device ushort* right [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& left_width [[buffer(4)]],
    constant uint& right_width [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_width = left_width + right_width;
    uint output_count = row_count * output_width;
    if (gid >= output_count) {
        return;
    }
    uint row = gid / output_width;
    uint column = gid - row * output_width;
    output[gid] = column < left_width
        ? left[row * left_width + column]
        : right[row * right_width + column - left_width];
}

kernel void qwen_bf16_copy_row_kernel(
    const device ushort* input [[buffer(0)]],
    device ushort* output [[buffer(1)]],
    constant uint& row_index [[buffer(2)]],
    constant uint& row_width [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < row_width) {
        output[gid] = input[row_index * row_width + gid];
    }
}

kernel void qwen_bf16_linear_kernel(
    const device ushort* weight [[buffer(0)]],
    const device ushort* input [[buffer(1)]],
    device ushort* output [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    uint output_index [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint output_count = row_count * out_features;
    if (output_index >= output_count) {
        return;
    }
    uint row = output_index / out_features;
    uint output_feature = output_index - row * out_features;
    uint input_start = row * in_features;
    uint weight_start = output_feature * in_features;
    float partial = 0.0f;
    for (uint column = lane; column < in_features; column += 32u) {
        partial += qwen_bf16_to_f32(input[input_start + column])
            * qwen_bf16_to_f32(weight[weight_start + column]);
    }
    float sum = simd_sum(partial);
    if (lane == 0u) {
        output[output_index] = qwen_f32_to_bf16(sum);
    }
}

kernel void qwen_bf16_last_token_logits_kernel(
    const device ushort* weight [[buffer(0)]],
    const device ushort* hidden_states [[buffer(1)]],
    device ushort* logits [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& sequence_length [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    constant uint& vocab_size [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint output_count = batch_size * vocab_size;
    if (group >= output_count) {
        return;
    }
    uint batch = group / vocab_size;
    uint token = batch * sequence_length + sequence_length - 1u;
    uint vocab = group - batch * vocab_size;
    uint input_start = token * hidden_size;
    uint weight_start = vocab * hidden_size;
    float partial = 0.0f;
    for (uint hidden = lane; hidden < hidden_size; hidden += 32u) {
        partial += qwen_bf16_to_f32(hidden_states[input_start + hidden])
            * qwen_bf16_to_f32(weight[weight_start + hidden]);
    }
    float sum = simd_sum(partial);
    if (lane == 0u) {
        logits[group] = qwen_f32_to_bf16(sum);
    }
}

kernel void qwen_bf16_all_token_logits_kernel(
    const device ushort* weight [[buffer(0)]],
    const device ushort* hidden_states [[buffer(1)]],
    device ushort* logits [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant uint& vocab_size [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint output_count = row_count * vocab_size;
    if (group >= output_count) {
        return;
    }
    uint row = group / vocab_size;
    uint vocab = group - row * vocab_size;
    uint input_start = row * hidden_size;
    uint weight_start = vocab * hidden_size;
    float partial = 0.0f;
    for (uint hidden = lane; hidden < hidden_size; hidden += 32u) {
        partial += qwen_bf16_to_f32(hidden_states[input_start + hidden])
            * qwen_bf16_to_f32(weight[weight_start + hidden]);
    }
    float sum = simd_sum(partial);
    if (lane == 0u) {
        logits[group] = qwen_f32_to_bf16(sum);
    }
}

kernel void qwen_bf16_verify_rows_logits_kernel(
    const device ushort* weight [[buffer(0)]],
    const device ushort* hidden_states [[buffer(1)]],
    device ushort* logits [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant uint& vocab_size [[buffer(5)]],
    uint vocab [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (row_count < 2u || row_count > QWEN_MAX_VERIFY_ROWS || vocab >= vocab_size) {
        return;
    }
    uint weight_start = vocab * hidden_size;
    float partials[QWEN_MAX_VERIFY_ROWS] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    for (uint hidden = lane; hidden < hidden_size; hidden += 32u) {
        float weight_value = qwen_bf16_to_f32(weight[weight_start + hidden]);
        for (uint row = 0u; row < row_count; row++) {
            partials[row] += qwen_bf16_to_f32(hidden_states[row * hidden_size + hidden])
                * weight_value;
        }
    }
    for (uint row = 0u; row < row_count; row++) {
        float sum = simd_sum(partials[row]);
        if (lane == 0u) {
            logits[row * vocab_size + vocab] = qwen_f32_to_bf16(sum);
        }
    }
}

kernel void qwen_bf16_verify3_logits_kernel(
    const device ushort* weight [[buffer(0)]],
    const device ushort* hidden_states [[buffer(1)]],
    device ushort* logits [[buffer(2)]],
    constant uint& row_count [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant uint& vocab_size [[buffer(5)]],
    uint vocab [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (row_count != 3u || vocab >= vocab_size) {
        return;
    }
    uint weight_start = vocab * hidden_size;
    float partial_0 = 0.0f;
    float partial_1 = 0.0f;
    float partial_2 = 0.0f;
    for (uint hidden = lane; hidden < hidden_size; hidden += 32u) {
        float weight_value = qwen_bf16_to_f32(weight[weight_start + hidden]);
        partial_0 += qwen_bf16_to_f32(hidden_states[hidden]) * weight_value;
        partial_1 += qwen_bf16_to_f32(hidden_states[hidden_size + hidden]) * weight_value;
        partial_2 += qwen_bf16_to_f32(hidden_states[hidden_size * 2u + hidden]) * weight_value;
    }
    float sum_0 = simd_sum(partial_0);
    float sum_1 = simd_sum(partial_1);
    float sum_2 = simd_sum(partial_2);
    if (lane == 0u) {
        logits[vocab] = qwen_f32_to_bf16(sum_0);
        logits[vocab_size + vocab] = qwen_f32_to_bf16(sum_1);
        logits[vocab_size * 2u + vocab] = qwen_f32_to_bf16(sum_2);
    }
}

kernel void qwen_bf16_compact_draft_logits_kernel(
    const device ushort* weight [[buffer(0)]],
    const device ushort* hidden_states [[buffer(1)]],
    device ushort* logits [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& sequence_length [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    uint output_count = batch_size * QWEN_COMPACT_DRAFT_VOCAB;
    if (group >= output_count) {
        return;
    }
    uint batch = group / QWEN_COMPACT_DRAFT_VOCAB;
    uint compact_vocab = group - batch * QWEN_COMPACT_DRAFT_VOCAB;
    uint vocab = compact_vocab < QWEN_COMPACT_DRAFT_PREFIX
        ? compact_vocab
        : QWEN_COMPACT_DRAFT_CONTROL_START
            + compact_vocab - QWEN_COMPACT_DRAFT_PREFIX;
    uint token = batch * sequence_length + sequence_length - 1u;
    uint input_start = token * hidden_size;
    uint weight_start = vocab * hidden_size;
    float partial = 0.0f;
    for (uint hidden = lane; hidden < hidden_size; hidden += 32u) {
        partial += qwen_bf16_to_f32(hidden_states[input_start + hidden])
            * qwen_bf16_to_f32(weight[weight_start + hidden]);
    }
    float sum = simd_sum(partial);
    if (lane == 0u) {
        logits[group] = qwen_f32_to_bf16(sum);
    }
}

kernel void qwen_bf16_argmax_kernel(
    const device ushort* logits [[buffer(0)]],
    device uint* token_ids [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    uint batch [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float shared_values[256];
    threadgroup uint shared_indices[256];
    if (batch >= batch_size) {
        return;
    }
    uint row_start = batch * vocab_size;
    float best_value = -INFINITY;
    uint best_index = 0u;
    for (uint vocab = thread_index; vocab < vocab_size; vocab += threads_per_group) {
        float value = qwen_bf16_to_f32(logits[row_start + vocab]);
        if (value > best_value || (value == best_value && vocab < best_index)) {
            best_value = value;
            best_index = vocab;
        }
    }
    shared_values[thread_index] = best_value;
    shared_indices[thread_index] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2u; stride > 0u; stride /= 2u) {
        if (thread_index < stride) {
            float candidate_value = shared_values[thread_index + stride];
            uint candidate_index = shared_indices[thread_index + stride];
            if (candidate_value > shared_values[thread_index]
                || (candidate_value == shared_values[thread_index]
                    && candidate_index < shared_indices[thread_index])) {
                shared_values[thread_index] = candidate_value;
                shared_indices[thread_index] = candidate_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index == 0u) {
        token_ids[batch] = shared_indices[0];
    }
}

kernel void qwen_bf16_compact_draft_argmax_kernel(
    const device ushort* logits [[buffer(0)]],
    device uint* token_ids [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    uint batch [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float shared_values[256];
    threadgroup uint shared_indices[256];
    if (batch >= batch_size) {
        return;
    }
    uint row_start = batch * QWEN_COMPACT_DRAFT_VOCAB;
    float best_value = -INFINITY;
    uint best_index = 0u;
    for (uint vocab = thread_index; vocab < QWEN_COMPACT_DRAFT_VOCAB;
         vocab += threads_per_group) {
        float value = qwen_bf16_to_f32(logits[row_start + vocab]);
        if (value > best_value || (value == best_value && vocab < best_index)) {
            best_value = value;
            best_index = vocab;
        }
    }
    shared_values[thread_index] = best_value;
    shared_indices[thread_index] = best_index;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2u; stride > 0u; stride /= 2u) {
        if (thread_index < stride) {
            float candidate_value = shared_values[thread_index + stride];
            uint candidate_index = shared_indices[thread_index + stride];
            if (candidate_value > shared_values[thread_index]
                || (candidate_value == shared_values[thread_index]
                    && candidate_index < shared_indices[thread_index])) {
                shared_values[thread_index] = candidate_value;
                shared_indices[thread_index] = candidate_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (thread_index == 0u) {
        uint compact_vocab = shared_indices[0];
        token_ids[batch] = compact_vocab < QWEN_COMPACT_DRAFT_PREFIX
            ? compact_vocab
            : QWEN_COMPACT_DRAFT_CONTROL_START
                + compact_vocab - QWEN_COMPACT_DRAFT_PREFIX;
    }
}
