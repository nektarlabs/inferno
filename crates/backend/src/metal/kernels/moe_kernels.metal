#include <metal_stdlib>

using namespace metal;

kernel void moe_gather_tokens_f32_kernel(
    const device float* flat_tokens [[buffer(0)]],
    const device uint* token_indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    constant uint& assignment_count [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = assignment_count * hidden_size;
    if (gid >= output_values) {
        return;
    }

    uint assignment = gid / hidden_size;
    uint hidden = gid - (assignment * hidden_size);
    uint token = token_indices[assignment];
    if (token >= token_count) {
        output[gid] = 0.0f;
        return;
    }

    output[gid] = flat_tokens[(token * hidden_size) + hidden];
}

kernel void moe_weighted_index_add_combine_f32_kernel(
    const device float* accumulator [[buffer(0)]],
    const device uint* token_indices [[buffer(1)]],
    const device float* expert_outputs [[buffer(2)]],
    const device float* expert_weights [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& token_count [[buffer(5)]],
    constant uint& hidden_size [[buffer(6)]],
    constant uint& assignment_count [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = token_count * hidden_size;
    if (gid >= output_values) {
        return;
    }

    uint token = gid / hidden_size;
    uint hidden = gid - (token * hidden_size);
    float value = accumulator[gid];

    for (uint assignment = 0; assignment < assignment_count; assignment++) {
        if (token_indices[assignment] == token) {
            uint expert_offset = (assignment * hidden_size) + hidden;
            value += expert_outputs[expert_offset] * expert_weights[assignment];
        }
    }

    output[gid] = value;
}

kernel void moe_weighted_token_major_combine_f32_kernel(
    const device float* accumulator [[buffer(0)]],
    const device uint* token_indices [[buffer(1)]],
    const device float* expert_outputs [[buffer(2)]],
    const device float* expert_weights [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& token_count [[buffer(5)]],
    constant uint& hidden_size [[buffer(6)]],
    constant uint& assignments_per_token [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = token_count * hidden_size;
    if (gid >= output_values) {
        return;
    }

    uint token = gid / hidden_size;
    uint hidden = gid - (token * hidden_size);
    uint assignment_base = token * assignments_per_token;
    float value = accumulator[gid];

    for (uint rank = 0; rank < assignments_per_token; rank++) {
        uint assignment = assignment_base + rank;
        if (token_indices[assignment] == token) {
            uint expert_offset = (assignment * hidden_size) + hidden;
            value += expert_outputs[expert_offset] * expert_weights[assignment];
        }
    }

    output[gid] = value;
}

kernel void moe_router_topk_f32_kernel(
    const device float* router_logits [[buffer(0)]],
    const device float* correction_bias [[buffer(1)]],
    device uint* expert_ids [[buffer(2)]],
    device float* expert_weights [[buffer(3)]],
    constant uint& token_count [[buffer(4)]],
    constant uint& expert_count [[buffer(5)]],
    constant uint& top_k [[buffer(6)]],
    constant uint& norm_topk_prob [[buffer(7)]],
    constant float& routed_scaling_factor [[buffer(8)]],
    uint token [[thread_position_in_grid]]
) {
    if (token >= token_count) {
        return;
    }

    constexpr uint max_top_k = 8;
    float top_corrected[max_top_k];
    float top_scores[max_top_k];
    uint top_ids[max_top_k];

    for (uint rank = 0; rank < max_top_k; rank++) {
        top_corrected[rank] = -3.402823466e+38F;
        top_scores[rank] = 0.0f;
        top_ids[rank] = 0;
    }

    for (uint expert = 0; expert < expert_count; expert++) {
        float logit = router_logits[(token * expert_count) + expert];
        float score = 1.0f / (1.0f + exp(-logit));
        float corrected = score + correction_bias[expert];

        uint insert_at = top_k;
        for (uint rank = 0; rank < top_k; rank++) {
            bool better = corrected > top_corrected[rank]
                || (corrected == top_corrected[rank] && expert < top_ids[rank]);
            if (better) {
                insert_at = rank;
                break;
            }
        }

        if (insert_at < top_k) {
            for (uint rank = top_k - 1; rank > insert_at; rank--) {
                top_corrected[rank] = top_corrected[rank - 1];
                top_scores[rank] = top_scores[rank - 1];
                top_ids[rank] = top_ids[rank - 1];
            }
            top_corrected[insert_at] = corrected;
            top_scores[insert_at] = score;
            top_ids[insert_at] = expert;
        }
    }

    float weight_sum = 0.0f;
    if (norm_topk_prob != 0) {
        for (uint rank = 0; rank < top_k; rank++) {
            weight_sum += top_scores[rank];
        }
    }

    for (uint rank = 0; rank < top_k; rank++) {
        float weight = top_scores[rank];
        if (norm_topk_prob != 0 && weight_sum > 0.0f && isfinite(weight_sum)) {
            weight /= weight_sum;
        }
        expert_ids[(token * top_k) + rank] = top_ids[rank];
        expert_weights[(token * top_k) + rank] = weight * routed_scaling_factor;
    }
}
