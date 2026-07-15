#include <metal_stdlib>

using namespace metal;

constant uint FUSED_DECODE_THREADS = 256;
constant uint FUSED_DECODE_SIMDGROUPS = 8;
constant uint FUSED_DECODE_MAX_KEYS = 4096;
constant float FUSED_NEG_INF = -3.4028234663852886e+38f;

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

kernel void fused_decode_attention_f32_kernel(
    const device float* q [[buffer(0)]],
    const device float* k [[buffer(1)]],
    const device float* v [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& batch_count [[buffer(4)]],
    constant uint& head_count [[buffer(5)]],
    constant uint& key_tokens [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],
    constant uint& value_dim [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row_count = batch_count * head_count;
    if (row >= row_count || key_tokens > FUSED_DECODE_MAX_KEYS) {
        return;
    }

    threadgroup float scores[FUSED_DECODE_MAX_KEYS];
    threadgroup float partials[FUSED_DECODE_SIMDGROUPS];
    threadgroup float row_max_shared;
    threadgroup float denom_shared;

    uint head = row % head_count;
    uint batch = row / head_count;
    uint q_base = ((batch * head_count + head) * head_dim);
    float scale = 1.0f / sqrt(float(head_dim));

    for (uint key = 0; key < key_tokens; key++) {
        uint k_base = ((batch * head_count + head) * key_tokens + key) * head_dim;
        float local_dot = 0.0f;
        for (uint dim = tid; dim < head_dim; dim += FUSED_DECODE_THREADS) {
            local_dot += q[q_base + dim] * k[k_base + dim];
        }

        float simd_dot = simd_sum(local_dot);
        if (simd_lane == 0) {
            partials[simd_group] = simd_dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            float group_dot = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : 0.0f;
            float total_dot = simd_sum(group_dot);
            if (tid == 0) {
                scores[key] = total_dot * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float local_max = FUSED_NEG_INF;
    for (uint key = tid; key < key_tokens; key += FUSED_DECODE_THREADS) {
        local_max = max(local_max, scores[key]);
    }
    float simd_max_value = simd_max(local_max);
    if (simd_lane == 0) {
        partials[simd_group] = simd_max_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float group_max = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : FUSED_NEG_INF;
        float total_max = simd_max(group_max);
        if (tid == 0) {
            row_max_shared = total_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_denom = 0.0f;
    for (uint key = tid; key < key_tokens; key += FUSED_DECODE_THREADS) {
        local_denom += exp(scores[key] - row_max_shared);
    }
    float simd_denom = simd_sum(local_denom);
    if (simd_lane == 0) {
        partials[simd_group] = simd_denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float group_denom = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : 0.0f;
        float total_denom = simd_sum(group_denom);
        if (tid == 0) {
            denom_shared = total_denom;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint value_index = tid; value_index < value_dim; value_index += FUSED_DECODE_THREADS) {
        float sum = 0.0f;
        for (uint key = 0; key < key_tokens; key++) {
            float probability = exp(scores[key] - row_max_shared) / denom_shared;
            uint value_base = ((batch * head_count + head) * key_tokens + key) * value_dim;
            sum += probability * v[value_base + value_index];
        }
        output[(row * value_dim) + value_index] = sum;
    }
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

kernel void fused_paged_decode_attention_f32_kernel(
    const device float* q [[buffer(0)]],
    const device float* paged_k [[buffer(1)]],
    const device float* current_k [[buffer(2)]],
    const device float* paged_v [[buffer(3)]],
    const device float* current_v [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch_count [[buffer(6)]],
    constant uint& head_count [[buffer(7)]],
    constant uint& past_tokens [[buffer(8)]],
    constant uint& page_size [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant uint& value_dim [[buffer(11)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row_count = batch_count * head_count;
    uint key_tokens = past_tokens + 1;
    if (row >= row_count || key_tokens > FUSED_DECODE_MAX_KEYS) {
        return;
    }

    threadgroup float scores[FUSED_DECODE_MAX_KEYS];
    threadgroup float partials[FUSED_DECODE_SIMDGROUPS];
    threadgroup float row_max_shared;
    threadgroup float denom_shared;

    uint head = row % head_count;
    uint batch = row / head_count;
    uint q_base = ((batch * head_count + head) * head_dim);
    uint current_k_base = ((batch * head_count + head) * head_dim);
    float scale = 1.0f / sqrt(float(head_dim));

    for (uint key = 0; key < key_tokens; key++) {
        float local_dot = 0.0f;
        if (key < past_tokens) {
            uint page_index = key / page_size;
            uint page_offset = key - (page_index * page_size);
            uint k_base = (((page_index * batch_count + batch) * head_count + head) * page_size + page_offset) * head_dim;
            for (uint dim = tid; dim < head_dim; dim += FUSED_DECODE_THREADS) {
                local_dot += q[q_base + dim] * paged_k[k_base + dim];
            }
        } else {
            for (uint dim = tid; dim < head_dim; dim += FUSED_DECODE_THREADS) {
                local_dot += q[q_base + dim] * current_k[current_k_base + dim];
            }
        }

        float simd_dot = simd_sum(local_dot);
        if (simd_lane == 0) {
            partials[simd_group] = simd_dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            float group_dot = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : 0.0f;
            float total_dot = simd_sum(group_dot);
            if (tid == 0) {
                scores[key] = total_dot * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float local_max = FUSED_NEG_INF;
    for (uint key = tid; key < key_tokens; key += FUSED_DECODE_THREADS) {
        local_max = max(local_max, scores[key]);
    }
    float simd_max_value = simd_max(local_max);
    if (simd_lane == 0) {
        partials[simd_group] = simd_max_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float group_max = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : FUSED_NEG_INF;
        float total_max = simd_max(group_max);
        if (tid == 0) {
            row_max_shared = total_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_denom = 0.0f;
    for (uint key = tid; key < key_tokens; key += FUSED_DECODE_THREADS) {
        local_denom += exp(scores[key] - row_max_shared);
    }
    float simd_denom = simd_sum(local_denom);
    if (simd_lane == 0) {
        partials[simd_group] = simd_denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float group_denom = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : 0.0f;
        float total_denom = simd_sum(group_denom);
        if (tid == 0) {
            denom_shared = total_denom;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint current_v_base = ((batch * head_count + head) * value_dim);
    for (uint value_index = tid; value_index < value_dim; value_index += FUSED_DECODE_THREADS) {
        float sum = 0.0f;
        for (uint key = 0; key < past_tokens; key++) {
            float probability = exp(scores[key] - row_max_shared) / denom_shared;
            uint page_index = key / page_size;
            uint page_offset = key - (page_index * page_size);
            uint v_index = (((page_index * batch_count + batch) * head_count + head) * page_size + page_offset) * value_dim + value_index;
            sum += probability * paged_v[v_index];
        }
        float current_probability = exp(scores[past_tokens] - row_max_shared) / denom_shared;
        sum += current_probability * current_v[current_v_base + value_index];
        output[(row * value_dim) + value_index] = sum;
    }
}

kernel void fused_paged_decode_attention_f16_kv_kernel(
    const device float* q [[buffer(0)]],
    const device half* paged_k [[buffer(1)]],
    const device float* current_k [[buffer(2)]],
    const device half* paged_v [[buffer(3)]],
    const device float* current_v [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch_count [[buffer(6)]],
    constant uint& head_count [[buffer(7)]],
    constant uint& past_tokens [[buffer(8)]],
    constant uint& page_size [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant uint& value_dim [[buffer(11)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row_count = batch_count * head_count;
    uint key_tokens = past_tokens + 1;
    if (row >= row_count || key_tokens > FUSED_DECODE_MAX_KEYS) {
        return;
    }

    threadgroup float scores[FUSED_DECODE_MAX_KEYS];
    threadgroup float partials[FUSED_DECODE_SIMDGROUPS];
    threadgroup float row_max_shared;
    threadgroup float denom_shared;

    uint head = row % head_count;
    uint batch = row / head_count;
    uint q_base = ((batch * head_count + head) * head_dim);
    uint current_k_base = ((batch * head_count + head) * head_dim);
    float scale = 1.0f / sqrt(float(head_dim));

    for (uint key = 0; key < key_tokens; key++) {
        float local_dot = 0.0f;
        if (key < past_tokens) {
            uint page_index = key / page_size;
            uint page_offset = key - (page_index * page_size);
            uint k_base = (((page_index * batch_count + batch) * head_count + head) * page_size + page_offset) * head_dim;
            for (uint dim = tid; dim < head_dim; dim += FUSED_DECODE_THREADS) {
                local_dot += q[q_base + dim] * float(paged_k[k_base + dim]);
            }
        } else {
            for (uint dim = tid; dim < head_dim; dim += FUSED_DECODE_THREADS) {
                local_dot += q[q_base + dim] * current_k[current_k_base + dim];
            }
        }

        float simd_dot = simd_sum(local_dot);
        if (simd_lane == 0) {
            partials[simd_group] = simd_dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            float group_dot = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : 0.0f;
            float total_dot = simd_sum(group_dot);
            if (tid == 0) {
                scores[key] = total_dot * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float local_max = FUSED_NEG_INF;
    for (uint key = tid; key < key_tokens; key += FUSED_DECODE_THREADS) {
        local_max = max(local_max, scores[key]);
    }
    float simd_max_value = simd_max(local_max);
    if (simd_lane == 0) {
        partials[simd_group] = simd_max_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float group_max = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : FUSED_NEG_INF;
        float total_max = simd_max(group_max);
        if (tid == 0) {
            row_max_shared = total_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_denom = 0.0f;
    for (uint key = tid; key < key_tokens; key += FUSED_DECODE_THREADS) {
        local_denom += exp(scores[key] - row_max_shared);
    }
    float simd_denom = simd_sum(local_denom);
    if (simd_lane == 0) {
        partials[simd_group] = simd_denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float group_denom = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : 0.0f;
        float total_denom = simd_sum(group_denom);
        if (tid == 0) {
            denom_shared = total_denom;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint current_v_base = ((batch * head_count + head) * value_dim);
    for (uint value_index = tid; value_index < value_dim; value_index += FUSED_DECODE_THREADS) {
        float sum = 0.0f;
        for (uint key = 0; key < past_tokens; key++) {
            float probability = exp(scores[key] - row_max_shared) / denom_shared;
            uint page_index = key / page_size;
            uint page_offset = key - (page_index * page_size);
            uint v_index = (((page_index * batch_count + batch) * head_count + head) * page_size + page_offset) * value_dim + value_index;
            sum += probability * float(paged_v[v_index]);
        }
        float current_probability = exp(scores[past_tokens] - row_max_shared) / denom_shared;
        sum += current_probability * current_v[current_v_base + value_index];
        output[(row * value_dim) + value_index] = sum;
    }
}

// Flash-style causal attention over normalized MLA latent cache rows.
//
// q_latent is K_b * q_no_rope, so the no-RoPE score can be computed directly
// against the cached 512-wide latent. The kernel aggregates one latent vector
// per query head; V_b is applied once after this kernel instead of once for
// every cached token.
kernel void fused_paged_absorbed_mla_sequence_f32_kernel(
    const device float* q_latent [[buffer(0)]],
    const device float* q_rope [[buffer(1)]],
    const device float* paged_latent [[buffer(2)]],
    const device float* current_latent [[buffer(3)]],
    const device float* paged_rope [[buffer(4)]],
    const device float* current_rope [[buffer(5)]],
    device float* output_latent [[buffer(6)]],
    constant uint& batch_count [[buffer(7)]],
    constant uint& head_count [[buffer(8)]],
    constant uint& query_tokens [[buffer(9)]],
    constant uint& past_tokens [[buffer(10)]],
    constant uint& page_size [[buffer(11)]],
    constant uint& latent_dim [[buffer(12)]],
    constant uint& rope_dim [[buffer(13)]],
    constant uint& scale_dim [[buffer(14)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row_count = batch_count * query_tokens * head_count;
    uint query = (row / head_count) % query_tokens;
    uint key_tokens = past_tokens + query + 1;
    if (row >= row_count || key_tokens > FUSED_DECODE_MAX_KEYS) {
        return;
    }

    threadgroup float scores[FUSED_DECODE_MAX_KEYS];
    threadgroup float partials[FUSED_DECODE_SIMDGROUPS];
    threadgroup float row_max_shared;
    threadgroup float denom_shared;

    uint batch = row / (query_tokens * head_count);
    uint q_latent_base = row * latent_dim;
    uint q_rope_base = row * rope_dim;
    float scale = 1.0f / sqrt(float(scale_dim));

    for (uint key = 0; key < key_tokens; key++) {
        float local_dot = 0.0f;
        if (key < past_tokens) {
            uint page_index = key / page_size;
            uint page_offset = key - (page_index * page_size);
            uint token_base = ((page_index * batch_count + batch) * page_size + page_offset);
            uint latent_base = token_base * latent_dim;
            uint rope_base = token_base * rope_dim;
            for (uint dim = tid; dim < latent_dim; dim += FUSED_DECODE_THREADS) {
                local_dot += q_latent[q_latent_base + dim]
                    * paged_latent[latent_base + dim];
            }
            for (uint dim = tid; dim < rope_dim; dim += FUSED_DECODE_THREADS) {
                local_dot += q_rope[q_rope_base + dim] * paged_rope[rope_base + dim];
            }
        } else {
            uint current_token = key - past_tokens;
            uint current_base = batch * query_tokens + current_token;
            uint latent_base = current_base * latent_dim;
            uint rope_base = current_base * rope_dim;
            for (uint dim = tid; dim < latent_dim; dim += FUSED_DECODE_THREADS) {
                local_dot += q_latent[q_latent_base + dim]
                    * current_latent[latent_base + dim];
            }
            for (uint dim = tid; dim < rope_dim; dim += FUSED_DECODE_THREADS) {
                local_dot += q_rope[q_rope_base + dim] * current_rope[rope_base + dim];
            }
        }

        float simd_dot = simd_sum(local_dot);
        if (simd_lane == 0) {
            partials[simd_group] = simd_dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            float group_dot = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : 0.0f;
            float total_dot = simd_sum(group_dot);
            if (tid == 0) {
                scores[key] = total_dot * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float local_max = FUSED_NEG_INF;
    for (uint key = tid; key < key_tokens; key += FUSED_DECODE_THREADS) {
        local_max = max(local_max, scores[key]);
    }
    float simd_max_value = simd_max(local_max);
    if (simd_lane == 0) {
        partials[simd_group] = simd_max_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float group_max = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : FUSED_NEG_INF;
        float total_max = simd_max(group_max);
        if (tid == 0) {
            row_max_shared = total_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_denom = 0.0f;
    for (uint key = tid; key < key_tokens; key += FUSED_DECODE_THREADS) {
        local_denom += exp(scores[key] - row_max_shared);
    }
    float simd_denom = simd_sum(local_denom);
    if (simd_lane == 0) {
        partials[simd_group] = simd_denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float group_denom = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : 0.0f;
        float total_denom = simd_sum(group_denom);
        if (tid == 0) {
            denom_shared = total_denom;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint dim = tid; dim < latent_dim; dim += FUSED_DECODE_THREADS) {
        float sum = 0.0f;
        for (uint key = 0; key < past_tokens; key++) {
            float probability = exp(scores[key] - row_max_shared) / denom_shared;
            uint page_index = key / page_size;
            uint page_offset = key - (page_index * page_size);
            uint token_base = ((page_index * batch_count + batch) * page_size + page_offset);
            sum += probability * paged_latent[token_base * latent_dim + dim];
        }
        for (uint current_token = 0; current_token <= query; current_token++) {
            uint key = past_tokens + current_token;
            float probability = exp(scores[key] - row_max_shared) / denom_shared;
            uint current_base = batch * query_tokens + current_token;
            sum += probability * current_latent[current_base * latent_dim + dim];
        }
        output_latent[row * latent_dim + dim] = sum;
    }
}

kernel void fused_selected_decode_attention_f32_kernel(
    const device float* q [[buffer(0)]],
    const device float* selected_k [[buffer(1)]],
    const device float* current_k [[buffer(2)]],
    const device float* selected_v [[buffer(3)]],
    const device float* current_v [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch_count [[buffer(6)]],
    constant uint& head_count [[buffer(7)]],
    constant uint& selected_tokens [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& value_dim [[buffer(10)]],
    constant uint& include_current_kv [[buffer(11)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row_count = batch_count * head_count;
    uint key_tokens = selected_tokens + include_current_kv;
    if (row >= row_count || key_tokens > FUSED_DECODE_MAX_KEYS) {
        return;
    }

    threadgroup float scores[FUSED_DECODE_MAX_KEYS];
    threadgroup float partials[FUSED_DECODE_SIMDGROUPS];
    threadgroup float row_max_shared;
    threadgroup float denom_shared;

    uint head = row % head_count;
    uint batch = row / head_count;
    uint q_base = ((batch * head_count + head) * head_dim);
    uint current_k_base = ((batch * head_count + head) * head_dim);
    float scale = 1.0f / sqrt(float(head_dim));

    for (uint key = 0; key < key_tokens; key++) {
        float local_dot = 0.0f;
        if (key < selected_tokens) {
            uint k_base = ((batch * head_count + head) * selected_tokens + key) * head_dim;
            for (uint dim = tid; dim < head_dim; dim += FUSED_DECODE_THREADS) {
                local_dot += q[q_base + dim] * selected_k[k_base + dim];
            }
        } else if (include_current_kv != 0) {
            for (uint dim = tid; dim < head_dim; dim += FUSED_DECODE_THREADS) {
                local_dot += q[q_base + dim] * current_k[current_k_base + dim];
            }
        }

        float simd_dot = simd_sum(local_dot);
        if (simd_lane == 0) {
            partials[simd_group] = simd_dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            float group_dot = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : 0.0f;
            float total_dot = simd_sum(group_dot);
            if (tid == 0) {
                scores[key] = total_dot * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float local_max = FUSED_NEG_INF;
    for (uint key = tid; key < key_tokens; key += FUSED_DECODE_THREADS) {
        local_max = max(local_max, scores[key]);
    }
    float simd_max_value = simd_max(local_max);
    if (simd_lane == 0) {
        partials[simd_group] = simd_max_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float group_max = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : FUSED_NEG_INF;
        float total_max = simd_max(group_max);
        if (tid == 0) {
            row_max_shared = total_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_denom = 0.0f;
    for (uint key = tid; key < key_tokens; key += FUSED_DECODE_THREADS) {
        local_denom += exp(scores[key] - row_max_shared);
    }
    float simd_denom = simd_sum(local_denom);
    if (simd_lane == 0) {
        partials[simd_group] = simd_denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float group_denom = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : 0.0f;
        float total_denom = simd_sum(group_denom);
        if (tid == 0) {
            denom_shared = total_denom;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint current_v_base = ((batch * head_count + head) * value_dim);
    for (uint value_index = tid; value_index < value_dim; value_index += FUSED_DECODE_THREADS) {
        float sum = 0.0f;
        for (uint key = 0; key < selected_tokens; key++) {
            float probability = exp(scores[key] - row_max_shared) / denom_shared;
            uint v_index = ((batch * head_count + head) * selected_tokens + key) * value_dim + value_index;
            sum += probability * selected_v[v_index];
        }
        if (include_current_kv != 0) {
            float current_probability = exp(scores[selected_tokens] - row_max_shared) / denom_shared;
            sum += current_probability * current_v[current_v_base + value_index];
        }
        output[(row * value_dim) + value_index] = sum;
    }
}

kernel void fused_selected_sequence_attention_f32_kernel(
    const device float* q [[buffer(0)]],
    const device float* selected_k [[buffer(1)]],
    const device float* current_k [[buffer(2)]],
    const device float* selected_v [[buffer(3)]],
    const device float* current_v [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch_count [[buffer(6)]],
    constant uint& head_count [[buffer(7)]],
    constant uint& selected_tokens [[buffer(8)]],
    constant uint& query_tokens [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant uint& value_dim [[buffer(11)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row_count = batch_count * head_count * query_tokens;
    if (row >= row_count || query_tokens == 0) {
        return;
    }

    uint query = row % query_tokens;
    uint head = (row / query_tokens) % head_count;
    uint batch = row / (head_count * query_tokens);
    uint visible_keys = selected_tokens + query + 1;
    if (visible_keys > FUSED_DECODE_MAX_KEYS) {
        return;
    }

    threadgroup float scores[FUSED_DECODE_MAX_KEYS];
    threadgroup float partials[FUSED_DECODE_SIMDGROUPS];
    threadgroup float row_max_shared;
    threadgroup float denom_shared;

    uint q_base = (((batch * head_count + head) * query_tokens + query) * head_dim);
    float scale = 1.0f / sqrt(float(head_dim));

    for (uint key = 0; key < visible_keys; key++) {
        float local_dot = 0.0f;
        if (key < selected_tokens) {
            uint k_base = (((batch * head_count + head) * selected_tokens + key) * head_dim);
            for (uint dim = tid; dim < head_dim; dim += FUSED_DECODE_THREADS) {
                local_dot += q[q_base + dim] * selected_k[k_base + dim];
            }
        } else {
            uint current = key - selected_tokens;
            uint k_base = (((batch * head_count + head) * query_tokens + current) * head_dim);
            for (uint dim = tid; dim < head_dim; dim += FUSED_DECODE_THREADS) {
                local_dot += q[q_base + dim] * current_k[k_base + dim];
            }
        }

        float simd_dot = simd_sum(local_dot);
        if (simd_lane == 0) {
            partials[simd_group] = simd_dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            float group_dot = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : 0.0f;
            float total_dot = simd_sum(group_dot);
            if (tid == 0) {
                scores[key] = total_dot * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float local_max = FUSED_NEG_INF;
    for (uint key = tid; key < visible_keys; key += FUSED_DECODE_THREADS) {
        local_max = max(local_max, scores[key]);
    }
    float simd_max_value = simd_max(local_max);
    if (simd_lane == 0) {
        partials[simd_group] = simd_max_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float group_max = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : FUSED_NEG_INF;
        float total_max = simd_max(group_max);
        if (tid == 0) {
            row_max_shared = total_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_denom = 0.0f;
    for (uint key = tid; key < visible_keys; key += FUSED_DECODE_THREADS) {
        local_denom += exp(scores[key] - row_max_shared);
    }
    float simd_denom = simd_sum(local_denom);
    if (simd_lane == 0) {
        partials[simd_group] = simd_denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float group_denom = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : 0.0f;
        float total_denom = simd_sum(group_denom);
        if (tid == 0) {
            denom_shared = total_denom;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint value_index = tid; value_index < value_dim; value_index += FUSED_DECODE_THREADS) {
        float sum = 0.0f;
        for (uint key = 0; key < visible_keys; key++) {
            float probability = exp(scores[key] - row_max_shared) / denom_shared;
            if (key < selected_tokens) {
                uint v_index = (((batch * head_count + head) * selected_tokens + key) * value_dim)
                    + value_index;
                sum += probability * selected_v[v_index];
            } else {
                uint current = key - selected_tokens;
                uint v_index = (((batch * head_count + head) * query_tokens + current) * value_dim)
                    + value_index;
                sum += probability * current_v[v_index];
            }
        }
        uint output_index = (((batch * head_count + head) * query_tokens + query) * value_dim)
            + value_index;
        output[output_index] = sum;
    }
}

kernel void fused_selected_decode_attention_f16_kv_kernel(
    const device float* q [[buffer(0)]],
    const device half* selected_k [[buffer(1)]],
    const device float* current_k [[buffer(2)]],
    const device half* selected_v [[buffer(3)]],
    const device float* current_v [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& batch_count [[buffer(6)]],
    constant uint& head_count [[buffer(7)]],
    constant uint& selected_tokens [[buffer(8)]],
    constant uint& head_dim [[buffer(9)]],
    constant uint& value_dim [[buffer(10)]],
    constant uint& include_current_kv [[buffer(11)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row_count = batch_count * head_count;
    uint key_tokens = selected_tokens + include_current_kv;
    if (row >= row_count || key_tokens > FUSED_DECODE_MAX_KEYS) {
        return;
    }

    threadgroup float scores[FUSED_DECODE_MAX_KEYS];
    threadgroup float partials[FUSED_DECODE_SIMDGROUPS];
    threadgroup float row_max_shared;
    threadgroup float denom_shared;

    uint head = row % head_count;
    uint batch = row / head_count;
    uint q_base = ((batch * head_count + head) * head_dim);
    uint current_k_base = ((batch * head_count + head) * head_dim);
    float scale = 1.0f / sqrt(float(head_dim));

    for (uint key = 0; key < key_tokens; key++) {
        float local_dot = 0.0f;
        if (key < selected_tokens) {
            uint k_base = ((batch * head_count + head) * selected_tokens + key) * head_dim;
            for (uint dim = tid; dim < head_dim; dim += FUSED_DECODE_THREADS) {
                local_dot += q[q_base + dim] * float(selected_k[k_base + dim]);
            }
        } else if (include_current_kv != 0) {
            for (uint dim = tid; dim < head_dim; dim += FUSED_DECODE_THREADS) {
                local_dot += q[q_base + dim] * current_k[current_k_base + dim];
            }
        }

        float simd_dot = simd_sum(local_dot);
        if (simd_lane == 0) {
            partials[simd_group] = simd_dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0) {
            float group_dot = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : 0.0f;
            float total_dot = simd_sum(group_dot);
            if (tid == 0) {
                scores[key] = total_dot * scale;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float local_max = FUSED_NEG_INF;
    for (uint key = tid; key < key_tokens; key += FUSED_DECODE_THREADS) {
        local_max = max(local_max, scores[key]);
    }
    float simd_max_value = simd_max(local_max);
    if (simd_lane == 0) {
        partials[simd_group] = simd_max_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float group_max = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : FUSED_NEG_INF;
        float total_max = simd_max(group_max);
        if (tid == 0) {
            row_max_shared = total_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_denom = 0.0f;
    for (uint key = tid; key < key_tokens; key += FUSED_DECODE_THREADS) {
        local_denom += exp(scores[key] - row_max_shared);
    }
    float simd_denom = simd_sum(local_denom);
    if (simd_lane == 0) {
        partials[simd_group] = simd_denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group == 0) {
        float group_denom = tid < FUSED_DECODE_SIMDGROUPS ? partials[tid] : 0.0f;
        float total_denom = simd_sum(group_denom);
        if (tid == 0) {
            denom_shared = total_denom;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint current_v_base = ((batch * head_count + head) * value_dim);
    for (uint value_index = tid; value_index < value_dim; value_index += FUSED_DECODE_THREADS) {
        float sum = 0.0f;
        for (uint key = 0; key < selected_tokens; key++) {
            float probability = exp(scores[key] - row_max_shared) / denom_shared;
            uint v_index = ((batch * head_count + head) * selected_tokens + key) * value_dim + value_index;
            sum += probability * float(selected_v[v_index]);
        }
        if (include_current_kv != 0) {
            float current_probability = exp(scores[selected_tokens] - row_max_shared) / denom_shared;
            sum += current_probability * current_v[current_v_base + value_index];
        }
        output[(row * value_dim) + value_index] = sum;
    }
}
