#include <metal_stdlib>

using namespace metal;

static inline float qwen_delta_bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

static inline ushort qwen_delta_f32_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    uint exponent = bits & 0x7f800000u;
    if (exponent == 0x7f800000u) {
        return ushort(bits >> 16);
    }
    uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return ushort(rounded >> 16);
}

static inline float qwen_delta_silu(float value) {
    return value / (1.0f + exp(-value));
}

static inline float qwen_delta_softplus(float value) {
    return max(value, 0.0f) + log(1.0f + exp(-abs(value)));
}

kernel void qwen_delta_causal_conv_kernel(
    const device ushort* input [[buffer(0)]],
    const device ushort* weight [[buffer(1)]],
    device ushort* state [[buffer(2)]],
    device ushort* checkpoint_state [[buffer(3)]],
    device ushort* output [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    constant uint& sequence_length [[buffer(6)]],
    constant uint& channels [[buffer(7)]],
    constant uint& checkpoint_capacity [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint channel_count = batch_size * channels;
    if (gid >= channel_count) {
        return;
    }
    uint batch = gid / channels;
    uint channel = gid - batch * channels;
    uint state_start = gid * 3u;
    float previous_0 = qwen_delta_bf16_to_f32(state[state_start]);
    float previous_1 = qwen_delta_bf16_to_f32(state[state_start + 1u]);
    float previous_2 = qwen_delta_bf16_to_f32(state[state_start + 2u]);
    float weight_0 = qwen_delta_bf16_to_f32(weight[channel * 4u]);
    float weight_1 = qwen_delta_bf16_to_f32(weight[channel * 4u + 1u]);
    float weight_2 = qwen_delta_bf16_to_f32(weight[channel * 4u + 2u]);
    float weight_3 = qwen_delta_bf16_to_f32(weight[channel * 4u + 3u]);

    for (uint token = 0u; token < sequence_length; token++) {
        uint index = (batch * sequence_length + token) * channels + channel;
        float current = qwen_delta_bf16_to_f32(input[index]);
        float convolved = previous_0 * weight_0
            + previous_1 * weight_1
            + previous_2 * weight_2
            + current * weight_3;
        output[index] = qwen_delta_f32_to_bf16(qwen_delta_silu(convolved));
        previous_0 = previous_1;
        previous_1 = previous_2;
        previous_2 = current;
        if (token < checkpoint_capacity) {
            uint checkpoint_start = token * channel_count * 3u + state_start;
            checkpoint_state[checkpoint_start] = qwen_delta_f32_to_bf16(previous_0);
            checkpoint_state[checkpoint_start + 1u] = qwen_delta_f32_to_bf16(previous_1);
            checkpoint_state[checkpoint_start + 2u] = qwen_delta_f32_to_bf16(previous_2);
        }
    }

    state[state_start] = qwen_delta_f32_to_bf16(previous_0);
    state[state_start + 1u] = qwen_delta_f32_to_bf16(previous_1);
    state[state_start + 2u] = qwen_delta_f32_to_bf16(previous_2);
}

kernel void qwen_delta_recurrent_kernel(
    const device ushort* mixed_qkv [[buffer(0)]],
    const device ushort* gate [[buffer(1)]],
    const device ushort* input_a [[buffer(2)]],
    const device ushort* input_b [[buffer(3)]],
    const device ushort* a_log [[buffer(4)]],
    const device ushort* dt_bias [[buffer(5)]],
    const device ushort* norm_weight [[buffer(6)]],
    device float* recurrent_state [[buffer(7)]],
    device float* checkpoint_state [[buffer(8)]],
    device ushort* output [[buffer(9)]],
    constant uint& batch_size [[buffer(10)]],
    constant uint& sequence_length [[buffer(11)]],
    constant uint& key_heads [[buffer(12)]],
    constant uint& value_heads [[buffer(13)]],
    constant uint& head_dim [[buffer(14)]],
    constant float& epsilon [[buffer(15)]],
    constant uint& checkpoint_capacity [[buffer(16)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_query[128];
    threadgroup float shared_key[128];
    threadgroup float shared_inverse_query_norm;
    threadgroup float shared_inverse_key_norm;
    threadgroup float shared_attended[128];
    threadgroup float shared_inverse_rms;
    uint group_count = batch_size * value_heads;
    if (group >= group_count) {
        return;
    }
    uint batch = group / value_heads;
    uint value_head = group - batch * value_heads;
    uint key_head = value_head / (value_heads / key_heads);
    uint key_width = key_heads * head_dim;
    uint value_width = value_heads * head_dim;
    uint mixed_width = key_width * 2u + value_width;
    uint state_start = group * head_dim * head_dim;
    float inverse_scale = rsqrt(float(head_dim));
    float a_scale = -exp(qwen_delta_bf16_to_f32(a_log[value_head]));
    float time_bias = qwen_delta_bf16_to_f32(dt_bias[value_head]);

    for (uint token = 0u; token < sequence_length; token++) {
        uint row = batch * sequence_length + token;
        uint query_start = row * mixed_width + key_head * head_dim;
        uint key_start = row * mixed_width + key_width + key_head * head_dim;
        uint value_start = row * mixed_width + key_width * 2u + value_head * head_dim;
        uint gate_start = row * value_width + value_head * head_dim;

        if (simd_group == 0u) {
            float query_square_sum = 0.0f;
            float key_square_sum = 0.0f;
            for (uint dim = lane; dim < head_dim; dim += 32u) {
                float query_value = qwen_delta_bf16_to_f32(mixed_qkv[query_start + dim]);
                float key_value = qwen_delta_bf16_to_f32(mixed_qkv[key_start + dim]);
                query_square_sum += query_value * query_value;
                key_square_sum += key_value * key_value;
            }
            // MLX RMSNorm uses mean(x^2) + eps. These kernels operate on the
            // unscaled sum, so the equivalent denominator is sum(x^2) + D*eps.
            float norm_epsilon = float(head_dim) * 1.0e-6f;
            float inverse_query_norm = rsqrt(simd_sum(query_square_sum) + norm_epsilon);
            float inverse_key_norm = rsqrt(simd_sum(key_square_sum) + norm_epsilon);
            if (lane == 0u) {
                shared_inverse_query_norm = inverse_query_norm;
                shared_inverse_key_norm = inverse_key_norm;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inverse_query_norm = shared_inverse_query_norm;
        float inverse_key_norm = shared_inverse_key_norm;
        uint value_dim = lane + simd_group * 32u;

        // MLX materializes the normalized and scaled Q/K tensors as BF16
        // before the recurrent update. Preserve both BF16 boundaries here
        // instead of folding the scale into the F32 state loops.
        float sqrt_head_dim = sqrt(float(head_dim));
        float query_rms = qwen_delta_bf16_to_f32(qwen_delta_f32_to_bf16(
            qwen_delta_bf16_to_f32(mixed_qkv[query_start + value_dim])
                * inverse_query_norm
                * sqrt_head_dim
        ));
        float key_rms = qwen_delta_bf16_to_f32(qwen_delta_f32_to_bf16(
            qwen_delta_bf16_to_f32(mixed_qkv[key_start + value_dim])
                * inverse_key_norm
                * sqrt_head_dim
        ));
        shared_query[value_dim] = qwen_delta_bf16_to_f32(qwen_delta_f32_to_bf16(
            query_rms / float(head_dim)
        ));
        shared_key[value_dim] = qwen_delta_bf16_to_f32(qwen_delta_f32_to_bf16(
            key_rms * inverse_scale
        ));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float a = qwen_delta_bf16_to_f32(input_a[row * value_heads + value_head]);
        float beta_input = qwen_delta_bf16_to_f32(input_b[row * value_heads + value_head]);
        float beta = 1.0f / (1.0f + exp(-beta_input));
        float decay = exp(a_scale * qwen_delta_softplus(a + time_bias));

        float memory = 0.0f;
        for (uint key_dim = 0u; key_dim < head_dim; key_dim++) {
            uint state_index = state_start + key_dim * head_dim + value_dim;
            float state_value = recurrent_state[state_index] * decay;
            recurrent_state[state_index] = state_value;
            memory += state_value * shared_key[key_dim];
        }
        float value = qwen_delta_bf16_to_f32(mixed_qkv[value_start + value_dim]);
        float delta = (value - memory) * beta;
        float attended = 0.0f;
        for (uint key_dim = 0u; key_dim < head_dim; key_dim++) {
            uint state_index = state_start + key_dim * head_dim + value_dim;
            float state_value = recurrent_state[state_index]
                + shared_key[key_dim] * delta;
            recurrent_state[state_index] = state_value;
            if (token < checkpoint_capacity) {
                uint state_values = group_count * head_dim * head_dim;
                checkpoint_state[token * state_values + state_index] = state_value;
            }
            attended += state_value * shared_query[key_dim];
        }
        float rounded_attended = qwen_delta_bf16_to_f32(qwen_delta_f32_to_bf16(attended));
        shared_attended[value_dim] = rounded_attended;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (simd_group == 0u) {
            float attended_square_sum = 0.0f;
            for (uint item = 0u; item < 4u; item++) {
                float item_value = shared_attended[lane + item * 32u];
                attended_square_sum += item_value * item_value;
            }
            float inverse_rms = rsqrt(
                simd_sum(attended_square_sum) / float(head_dim) + epsilon
            );
            if (lane == 0u) {
                shared_inverse_rms = inverse_rms;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float normalized = qwen_delta_bf16_to_f32(
            qwen_delta_f32_to_bf16(rounded_attended * shared_inverse_rms)
        ) * qwen_delta_bf16_to_f32(norm_weight[value_dim]);
        normalized = qwen_delta_bf16_to_f32(qwen_delta_f32_to_bf16(normalized));
        float gate_value = qwen_delta_bf16_to_f32(gate[gate_start + value_dim]);
        uint output_index = row * value_width + value_head * head_dim + value_dim;
        output[output_index] = qwen_delta_f32_to_bf16(
            normalized * qwen_delta_silu(gate_value)
        );
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void qwen_delta_restore_checkpoint_kernel(
    const device uint* checkpoint [[buffer(0)]],
    device uint* state [[buffer(1)]],
    constant uint& word_count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < word_count) {
        state[gid] = checkpoint[gid];
    }
}
