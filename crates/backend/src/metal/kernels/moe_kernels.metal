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

kernel void moe_scatter_rows_f32_kernel(
    const device float* rows [[buffer(0)]],
    const device uint* destination_rows [[buffer(1)]],
    device float* destination [[buffer(2)]],
    constant uint& source_row_count [[buffer(3)]],
    constant uint& destination_row_count [[buffer(4)]],
    constant uint& hidden_size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint value_count = source_row_count * hidden_size;
    if (gid >= value_count) {
        return;
    }

    uint source_row = gid / hidden_size;
    uint hidden = gid - (source_row * hidden_size);
    uint destination_row = destination_rows[source_row];
    if (destination_row >= destination_row_count) {
        return;
    }
    destination[(destination_row * hidden_size) + hidden] = rows[gid];
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
    device uint* token_indices [[buffer(4)]],
    constant uint& token_count [[buffer(5)]],
    constant uint& expert_count [[buffer(6)]],
    constant uint& top_k [[buffer(7)]],
    constant uint& norm_topk_prob [[buffer(8)]],
    constant float& routed_scaling_factor [[buffer(9)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    constexpr uint simd_lanes = 32;
    constexpr uint max_local_experts = 8;
    constexpr uint max_top_k = 16;
    uint token = gid / simd_lanes;
    if (token >= token_count) {
        return;
    }

    float local_corrected[max_local_experts];
    float local_scores[max_local_experts];
    uint local_ids[max_local_experts];
    bool local_selected[max_local_experts];
    uint local_count = 0;
    for (uint expert = simd_lane; expert < expert_count; expert += simd_lanes) {
        float logit = router_logits[(token * expert_count) + expert];
        float score = 1.0f / (1.0f + exp(-logit));
        local_corrected[local_count] = score + correction_bias[expert];
        local_scores[local_count] = score;
        local_ids[local_count] = expert;
        local_selected[local_count] = false;
        local_count++;
    }

    float selected_scores[max_top_k];
    uint selected_ids[max_top_k];
    for (uint rank = 0; rank < top_k; rank++) {
        float lane_corrected = -3.402823466e+38F;
        float lane_score = 0.0f;
        uint lane_id = 0xffffffffu;
        uint lane_local_index = max_local_experts;
        for (uint local = 0; local < local_count; local++) {
            if (local_selected[local]) {
                continue;
            }
            float corrected = local_corrected[local];
            uint expert = local_ids[local];
            bool better = corrected > lane_corrected
                || (corrected == lane_corrected && expert < lane_id);
            if (better) {
                lane_corrected = corrected;
                lane_score = local_scores[local];
                lane_id = expert;
                lane_local_index = local;
            }
        }

        float best_corrected = simd_max(lane_corrected);
        uint candidate_id = lane_corrected == best_corrected ? lane_id : 0xffffffffu;
        uint best_id = simd_min(candidate_id);
        uint winning_lane = best_id % simd_lanes;
        float best_score = simd_broadcast(lane_score, winning_lane);
        if (lane_local_index < local_count && lane_id == best_id) {
            local_selected[lane_local_index] = true;
        }
        if (simd_lane == 0) {
            selected_scores[rank] = best_score;
            selected_ids[rank] = best_id;
        }
    }

    if (simd_lane == 0) {
        float weight_sum = 0.0f;
        if (norm_topk_prob != 0) {
            for (uint rank = 0; rank < top_k; rank++) {
                weight_sum += selected_scores[rank];
            }
        }

        for (uint rank = 0; rank < top_k; rank++) {
            float weight = selected_scores[rank];
            if (norm_topk_prob != 0 && weight_sum > 0.0f && isfinite(weight_sum)) {
                weight /= weight_sum;
            }
            uint assignment = (token * top_k) + rank;
            expert_ids[assignment] = selected_ids[rank];
            expert_weights[assignment] = weight * routed_scaling_factor;
            token_indices[assignment] = token;
        }
    }
}

kernel void moe_topk_combine_residual_f32_kernel(
    const device float* shared [[buffer(0)]],
    const device float* residual [[buffer(1)]],
    const device float* expert_outputs [[buffer(2)]],
    const device float* expert_weights [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant uint& token_count [[buffer(5)]],
    constant uint& hidden_size [[buffer(6)]],
    constant uint& top_k [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = token_count * hidden_size;
    if (gid >= output_values) {
        return;
    }

    uint token = gid / hidden_size;
    uint hidden = gid - (token * hidden_size);
    uint assignment_base = token * top_k;
    float value = shared[gid] + residual[gid];

    for (uint rank = 0; rank < top_k; rank++) {
        uint assignment = assignment_base + rank;
        value += expert_outputs[(assignment * hidden_size) + hidden]
            * expert_weights[assignment];
    }

    output[gid] = value;
}
