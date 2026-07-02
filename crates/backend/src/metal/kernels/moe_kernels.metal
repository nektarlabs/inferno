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
