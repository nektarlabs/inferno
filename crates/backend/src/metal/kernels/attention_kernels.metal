#include <metal_stdlib>

using namespace metal;

kernel void attention_scores_f32_kernel(
    const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch_count [[buffer(3)]],
    constant uint& head_count [[buffer(4)]],
    constant uint& query_tokens [[buffer(5)]],
    constant uint& key_tokens [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = batch_count * head_count * query_tokens * key_tokens;
    if (gid >= output_values) {
        return;
    }

    uint key = gid % key_tokens;
    uint query = (gid / key_tokens) % query_tokens;
    uint head = (gid / (key_tokens * query_tokens)) % head_count;
    uint batch = gid / (key_tokens * query_tokens * head_count);

    uint q_base = (((batch * head_count + head) * query_tokens + query) * head_dim);
    uint k_base = (((batch * head_count + head) * key_tokens + key) * head_dim);

    float sum = 0.0f;
    for (uint dim = 0; dim < head_dim; dim++) {
        sum += q[q_base + dim] * k[k_base + dim];
    }

    output[gid] = sum / sqrt(float(head_dim));
}

kernel void attention_values_f32_kernel(
    const device float* probs [[buffer(0)]],
    const device float* values [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch_count [[buffer(3)]],
    constant uint& head_count [[buffer(4)]],
    constant uint& query_tokens [[buffer(5)]],
    constant uint& key_tokens [[buffer(6)]],
    constant uint& value_dim [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = batch_count * head_count * query_tokens * value_dim;
    if (gid >= output_values) {
        return;
    }

    uint value_index = gid % value_dim;
    uint query = (gid / value_dim) % query_tokens;
    uint head = (gid / (value_dim * query_tokens)) % head_count;
    uint batch = gid / (value_dim * query_tokens * head_count);

    uint probs_base = (((batch * head_count + head) * query_tokens + query) * key_tokens);
    uint values_base = ((batch * head_count + head) * key_tokens * value_dim) + value_index;

    float sum = 0.0f;
    for (uint key = 0; key < key_tokens; key++) {
        sum += probs[probs_base + key] * values[values_base + (key * value_dim)];
    }

    output[gid] = sum;
}

kernel void attention_causal_softmax_f32_kernel(
    const device float* scores [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_count [[buffer(2)]],
    constant uint& head_count [[buffer(3)]],
    constant uint& query_tokens [[buffer(4)]],
    constant uint& key_tokens [[buffer(5)]],
    constant uint& past_tokens [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    uint row_count = batch_count * head_count * query_tokens;
    if (row >= row_count) {
        return;
    }

    uint query = row % query_tokens;
    uint head = (row / query_tokens) % head_count;
    uint batch = row / (query_tokens * head_count);
    uint base = (((batch * head_count + head) * query_tokens + query) * key_tokens);
    uint max_visible_key = past_tokens + query;

    float max_value = scores[base];
    for (uint key = 1; key <= max_visible_key; key++) {
        max_value = max(max_value, scores[base + key]);
    }

    float sum = 0.0f;
    for (uint key = 0; key <= max_visible_key; key++) {
        float value = exp(scores[base + key] - max_value);
        output[base + key] = value;
        sum += value;
    }

    float inv_sum = 1.0f / sum;
    for (uint key = 0; key <= max_visible_key; key++) {
        output[base + key] *= inv_sum;
    }

    for (uint key = max_visible_key + 1; key < key_tokens; key++) {
        output[base + key] = 0.0f;
    }
}

kernel void paged_decode_attention_scores_f32_kernel(
    const device float* q [[buffer(0)]],
    const device float* paged_k [[buffer(1)]],
    const device float* current_k [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch_count [[buffer(4)]],
    constant uint& head_count [[buffer(5)]],
    constant uint& past_tokens [[buffer(6)]],
    constant uint& page_size [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint key_tokens = past_tokens + 1;
    uint output_values = batch_count * head_count * key_tokens;
    if (gid >= output_values) {
        return;
    }

    uint key = gid % key_tokens;
    uint head = (gid / key_tokens) % head_count;
    uint batch = gid / (key_tokens * head_count);

    uint q_base = ((batch * head_count + head) * head_dim);
    uint current_k_base = ((batch * head_count + head) * head_dim);

    float sum = 0.0f;
    if (key < past_tokens) {
        uint page_index = key / page_size;
        uint page_offset = key - (page_index * page_size);
        uint k_base = (((page_index * batch_count + batch) * head_count + head) * page_size + page_offset) * head_dim;
        for (uint dim = 0; dim < head_dim; dim++) {
            sum += q[q_base + dim] * paged_k[k_base + dim];
        }
    } else {
        for (uint dim = 0; dim < head_dim; dim++) {
            sum += q[q_base + dim] * current_k[current_k_base + dim];
        }
    }

    output[gid] = sum / sqrt(float(head_dim));
}

kernel void paged_decode_attention_values_f32_kernel(
    const device float* probs [[buffer(0)]],
    const device float* paged_v [[buffer(1)]],
    const device float* current_v [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch_count [[buffer(4)]],
    constant uint& head_count [[buffer(5)]],
    constant uint& past_tokens [[buffer(6)]],
    constant uint& page_size [[buffer(7)]],
    constant uint& value_dim [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = batch_count * head_count * value_dim;
    if (gid >= output_values) {
        return;
    }

    uint value_index = gid % value_dim;
    uint head = (gid / value_dim) % head_count;
    uint batch = gid / (value_dim * head_count);
    uint key_tokens = past_tokens + 1;
    uint probs_base = (batch * head_count + head) * key_tokens;
    uint current_v_base = ((batch * head_count + head) * value_dim);

    float sum = 0.0f;
    for (uint key = 0; key < past_tokens; key++) {
        uint page_index = key / page_size;
        uint page_offset = key - (page_index * page_size);
        uint v_index = (((page_index * batch_count + batch) * head_count + head) * page_size + page_offset) * value_dim + value_index;
        sum += probs[probs_base + key] * paged_v[v_index];
    }
    sum += probs[probs_base + past_tokens] * current_v[current_v_base + value_index];

    output[gid] = sum;
}
