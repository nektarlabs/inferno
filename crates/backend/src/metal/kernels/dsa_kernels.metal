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

constant uint DSA_TOPK_BLOCK_THREADS = 256;
constant uint DSA_INVALID_TOKEN = 0xffffffffu;
constant float DSA_LOWEST_SCORE = -3.402823466e+38F;

static inline bool dsa_precedes(
    float left_score,
    uint left_id,
    float right_score,
    uint right_id
) {
    if (left_id == DSA_INVALID_TOKEN) {
        return false;
    }
    if (right_id == DSA_INVALID_TOKEN) {
        return true;
    }
    return left_score > right_score
        || (left_score == right_score && left_id < right_id);
}

static inline float dsa_token_score(
    const device float* scores,
    uint score_row_offset,
    uint token_id
) {
    return token_id == DSA_INVALID_TOKEN
        ? DSA_LOWEST_SCORE
        : scores[score_row_offset + token_id];
}

// Sort one block of scores entirely in threadgroup memory. This avoids a
// device-memory gather for every comparator in the bitonic network.
kernel void dsa_topk_block_sort_u32_kernel(
    const device float* scores [[buffer(0)]],
    device uint* output_ids [[buffer(1)]],
    constant uint& batch_count [[buffer(2)]],
    constant uint& key_tokens [[buffer(3)]],
    constant uint& blocks_per_batch [[buffer(4)]],
    constant uint& block_top_k [[buffer(5)]],
    constant uint& output_stride [[buffer(6)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    uint group = threadgroup_position.x;
    uint batch = group / blocks_per_batch;
    uint block = group - (batch * blocks_per_batch);
    if (batch >= batch_count || lane >= DSA_TOPK_BLOCK_THREADS) {
        return;
    }

    uint token_id = (block * DSA_TOPK_BLOCK_THREADS) + lane;
    bool valid = token_id < key_tokens;
    threadgroup float block_scores[DSA_TOPK_BLOCK_THREADS];
    threadgroup uint block_ids[DSA_TOPK_BLOCK_THREADS];
    block_scores[lane] = valid
        ? scores[(batch * key_tokens) + token_id]
        : DSA_LOWEST_SCORE;
    block_ids[lane] = valid ? token_id : DSA_INVALID_TOKEN;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint sequence = 2; sequence <= DSA_TOPK_BLOCK_THREADS; sequence <<= 1) {
        for (uint distance = sequence >> 1; distance > 0; distance >>= 1) {
            uint partner = lane ^ distance;
            if (partner > lane) {
                bool left_must_precede = (lane & sequence) == 0;
                float left_score = block_scores[lane];
                uint left_id = block_ids[lane];
                float right_score = block_scores[partner];
                uint right_id = block_ids[partner];
                bool right_precedes_left = dsa_precedes(
                    right_score,
                    right_id,
                    left_score,
                    left_id
                );
                bool left_precedes_right = dsa_precedes(
                    left_score,
                    left_id,
                    right_score,
                    right_id
                );
                bool swap = left_must_precede
                    ? right_precedes_left
                    : left_precedes_right;
                if (swap) {
                    block_scores[lane] = right_score;
                    block_ids[lane] = right_id;
                    block_scores[partner] = left_score;
                    block_ids[partner] = left_id;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (lane < block_top_k) {
        uint output_base = (batch * output_stride) + (block * block_top_k);
        output_ids[output_base + lane] = block_ids[lane];
    }
}

// Merge two descending candidate runs. Threads divide the output into short
// contiguous ranges, find each range's merge partition by binary search, and
// then merge that range sequentially from registers.
kernel void dsa_topk_merge_u32_kernel(
    const device float* scores [[buffer(0)]],
    const device uint* input_ids [[buffer(1)]],
    device uint* output_ids [[buffer(2)]],
    constant uint& batch_count [[buffer(3)]],
    constant uint& key_tokens [[buffer(4)]],
    constant uint& input_run_count [[buffer(5)]],
    constant uint& input_run_length [[buffer(6)]],
    constant uint& output_run_length [[buffer(7)]],
    constant uint& input_stride [[buffer(8)]],
    constant uint& output_stride [[buffer(9)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]]
) {
    uint output_run_count = (input_run_count + 1u) / 2u;
    uint group = threadgroup_position.x;
    uint batch = group / output_run_count;
    uint output_run = group - (batch * output_run_count);
    if (batch >= batch_count) {
        return;
    }

    uint left_run = output_run * 2u;
    uint right_run = left_run + 1u;
    uint left_length = left_run < input_run_count ? input_run_length : 0u;
    uint right_length = right_run < input_run_count ? input_run_length : 0u;
    uint total_length = left_length + right_length;
    uint input_base = batch * input_stride;
    const device uint* left = input_ids + input_base + (left_run * input_run_length);
    const device uint* right = right_run < input_run_count
        ? input_ids + input_base + (right_run * input_run_length)
        : left;
    uint output_base = (batch * output_stride) + (output_run * output_run_length);
    uint score_row_offset = batch * key_tokens;
    uint chunk = (output_run_length + threads_per_group.x - 1u) / threads_per_group.x;
    uint output_start = lane * chunk;
    uint output_end = min(output_start + chunk, output_run_length);
    if (output_start >= output_run_length) {
        return;
    }

    uint merge_end = min(output_end, total_length);
    if (output_start < total_length) {
        uint low = output_start > right_length
            ? output_start - right_length
            : 0u;
        uint high = min(output_start, left_length);
        while (low < high) {
            uint left_index = (low + high) >> 1;
            uint right_index = output_start - left_index - 1u;
            uint left_id = left[left_index];
            uint right_id = right[right_index];
            float left_score = dsa_token_score(
                scores,
                score_row_offset,
                left_id
            );
            float right_score = dsa_token_score(
                scores,
                score_row_offset,
                right_id
            );
            if (dsa_precedes(left_score, left_id, right_score, right_id)) {
                low = left_index + 1u;
            } else {
                high = left_index;
            }
        }

        uint left_index = low;
        uint right_index = output_start - left_index;
        for (uint output_index = output_start;
             output_index < merge_end;
             output_index++) {
            uint selected_id;
            if (left_index >= left_length) {
                selected_id = right[right_index++];
            } else if (right_index >= right_length) {
                selected_id = left[left_index++];
            } else {
                uint left_id = left[left_index];
                uint right_id = right[right_index];
                float left_score = dsa_token_score(
                    scores,
                    score_row_offset,
                    left_id
                );
                float right_score = dsa_token_score(
                    scores,
                    score_row_offset,
                    right_id
                );
                if (dsa_precedes(left_score, left_id, right_score, right_id)) {
                    selected_id = left_id;
                    left_index++;
                } else {
                    selected_id = right_id;
                    right_index++;
                }
            }
            output_ids[output_base + output_index] = selected_id;
        }
    }

    for (uint output_index = max(output_start, total_length);
         output_index < output_end;
         output_index++) {
        output_ids[output_base + output_index] = DSA_INVALID_TOKEN;
    }
}
