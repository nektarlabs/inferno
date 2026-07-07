#include <metal_stdlib>

using namespace metal;

static inline void dsa_rotate_pair(
    float x0,
    float x1,
    uint pair_index,
    uint rope_dim,
    uint position,
    float theta,
    thread float& out0,
    thread float& out1
) {
    float freq = 1.0f / pow(theta, float(pair_index * 2u) / float(rope_dim));
    float angle = float(position) * freq;
    float c = cos(angle);
    float s = sin(angle);
    out0 = (x0 * c) - (x1 * s);
    out1 = (x1 * c) + (x0 * s);
}

kernel void dsa_key_norm_rope_f32_kernel(
    const device float* raw_key [[buffer(0)]],
    const device float* weight [[buffer(1)]],
    const device float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& row_count [[buffer(4)]],
    constant uint& token_count [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    constant uint& rope_dim [[buffer(7)]],
    constant uint& position_offset [[buffer(8)]],
    constant float& theta [[buffer(9)]],
    constant float& eps [[buffer(10)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= row_count) {
        return;
    }

    uint base = row * head_dim;
    float mean = 0.0f;
    for (uint dim = 0; dim < head_dim; dim++) {
        mean += raw_key[base + dim];
    }
    mean /= float(head_dim);

    float variance = 0.0f;
    for (uint dim = 0; dim < head_dim; dim++) {
        float centered = raw_key[base + dim] - mean;
        variance += centered * centered;
    }
    float inv_std = rsqrt((variance / float(head_dim)) + eps);

    uint token = row - ((row / token_count) * token_count);
    uint pair_count = rope_dim / 2u;
    for (uint dim = 0; dim < head_dim; dim++) {
        output[base + dim] = ((raw_key[base + dim] - mean) * inv_std * weight[dim]) + bias[dim];
    }

    for (uint pair = 0; pair < pair_count; pair++) {
        uint even = pair * 2u;
        uint odd = even + 1u;
        float rotated_even = 0.0f;
        float rotated_odd = 0.0f;
        dsa_rotate_pair(
            output[base + even],
            output[base + odd],
            pair,
            rope_dim,
            position_offset + token,
            theta,
            rotated_even,
            rotated_odd
        );
        output[base + pair] = rotated_even;
        output[base + pair_count + pair] = rotated_odd;
    }
}

kernel void dsa_query_weights_f32_kernel(
    const device float* hidden_states [[buffer(0)]],
    const device float* q_raw [[buffer(1)]],
    const device float* weights_proj [[buffer(2)]],
    device float* q_output [[buffer(3)]],
    device float* weight_output [[buffer(4)]],
    constant uint& batch_count [[buffer(5)]],
    constant uint& hidden_size [[buffer(6)]],
    constant uint& head_count [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant uint& rope_dim [[buffer(9)]],
    constant uint& position_offset [[buffer(10)]],
    constant float& theta [[buffer(11)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = batch_count * head_count;
    if (gid >= total) {
        return;
    }

    uint batch = gid / head_count;
    uint head = gid - (batch * head_count);
    uint hidden_base = batch * hidden_size;

    float projected_weight = 0.0f;
    for (uint dim = 0; dim < hidden_size; dim++) {
        projected_weight += hidden_states[hidden_base + dim] * weights_proj[(dim * head_count) + head];
    }
    weight_output[(batch * head_count) + head] = projected_weight * rsqrt(float(head_count));

    uint q_raw_base = (batch * head_count * head_dim) + (head * head_dim);
    uint q_out_base = q_raw_base;
    uint pair_count = rope_dim / 2u;
    for (uint pair = 0; pair < pair_count; pair++) {
        uint even = pair * 2u;
        uint odd = even + 1u;
        float rotated_even = 0.0f;
        float rotated_odd = 0.0f;
        dsa_rotate_pair(
            q_raw[q_raw_base + even],
            q_raw[q_raw_base + odd],
            pair,
            rope_dim,
            position_offset,
            theta,
            rotated_even,
            rotated_odd
        );
        q_output[q_out_base + pair] = rotated_even;
        q_output[q_out_base + pair_count + pair] = rotated_odd;
    }
    for (uint dim = rope_dim; dim < head_dim; dim++) {
        q_output[q_out_base + dim] = q_raw[q_raw_base + dim];
    }
}

kernel void dsa_scores_f32_kernel(
    const device float* q [[buffer(0)]],
    const device float* weights [[buffer(1)]],
    const device float* past_keys [[buffer(2)]],
    const device float* current_key [[buffer(3)]],
    device float* scores [[buffer(4)]],
    constant uint& batch_count [[buffer(5)]],
    constant uint& past_tokens [[buffer(6)]],
    constant uint& head_count [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint key_tokens = past_tokens + 1u;
    uint total = batch_count * key_tokens;
    if (gid >= total) {
        return;
    }

    uint batch = gid / key_tokens;
    uint token = gid - (batch * key_tokens);
    float score = 0.0f;
    float scale = rsqrt(float(head_dim));

    for (uint head = 0; head < head_count; head++) {
        uint q_base = ((batch * head_count + head) * head_dim);
        float dot = 0.0f;
        for (uint dim = 0; dim < head_dim; dim++) {
            float key_value;
            if (token < past_tokens) {
                key_value = past_keys[((batch * past_tokens + token) * head_dim) + dim];
            } else {
                key_value = current_key[(batch * head_dim) + dim];
            }
            dot += q[q_base + dim] * key_value;
        }
        float relu_score = max(dot * scale, 0.0f);
        score += weights[(batch * head_count) + head] * relu_score;
    }

    scores[gid] = score;
}

kernel void dsa_topk_scores_u32_kernel(
    const device float* scores [[buffer(0)]],
    device uint* token_ids [[buffer(1)]],
    constant uint& batch_count [[buffer(2)]],
    constant uint& key_tokens [[buffer(3)]],
    constant uint& top_k [[buffer(4)]],
    uint batch [[thread_position_in_grid]]
) {
    if (batch >= batch_count) {
        return;
    }

    constexpr uint max_top_k = 2048;
    float top_scores[max_top_k];
    uint top_ids[max_top_k];

    for (uint rank = 0; rank < max_top_k; rank++) {
        top_scores[rank] = -3.402823466e+38F;
        top_ids[rank] = 0;
    }

    for (uint token = 0; token < key_tokens; token++) {
        float score = scores[(batch * key_tokens) + token];
        uint insert_at = top_k;
        for (uint rank = 0; rank < top_k; rank++) {
            bool better = score > top_scores[rank]
                || (score == top_scores[rank] && token < top_ids[rank]);
            if (better) {
                insert_at = rank;
                break;
            }
        }

        if (insert_at < top_k) {
            for (uint rank = top_k - 1u; rank > insert_at; rank--) {
                top_scores[rank] = top_scores[rank - 1u];
                top_ids[rank] = top_ids[rank - 1u];
            }
            top_scores[insert_at] = score;
            top_ids[insert_at] = token;
        }
    }

    for (uint rank = 0; rank < top_k; rank++) {
        token_ids[(batch * top_k) + rank] = top_ids[rank];
    }
}
