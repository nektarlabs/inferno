#include <metal_stdlib>

using namespace metal;

static inline float qwen_delta_prefill_bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

static inline ushort qwen_delta_prefill_f32_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    uint exponent = bits & 0x7f800000u;
    if (exponent == 0x7f800000u) {
        return ushort(bits >> 16);
    }
    uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return ushort(rounded >> 16);
}

static inline float qwen_delta_prefill_silu(float value) {
    return value / (1.0f + exp(-value));
}

static inline float qwen_delta_prefill_softplus(float value) {
    return max(value, 0.0f) + log(1.0f + exp(-abs(value)));
}

kernel void qwen_delta_prefill_prepare_kernel(
    const device ushort* mixed_qkv [[buffer(0)]],
    device ushort* prepared_qk [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]
) {
    constexpr uint HEAD_DIM = 128u;
    constexpr uint KEY_HEADS = 16u;
    constexpr uint KEY_WIDTH = 2048u;
    constexpr uint MIXED_WIDTH = 10240u;
    uint row = group / KEY_HEADS;
    // Three value heads share each Q/K head; prepare it only once.
    uint key_head = group % KEY_HEADS;
    uint source = row * MIXED_WIDTH + key_head * HEAD_DIM;
    uint destination = row * KEY_WIDTH * 2u + key_head * HEAD_DIM;
    float query[4];
    float key[4];
    float query_sum = 0.0f;
    float key_sum = 0.0f;
    for (uint dim = lane; dim < head_dim; dim += 32u) {
        uint item = dim / 32u;
        query[item] = qwen_delta_prefill_bf16_to_f32(mixed_qkv[source + dim]);
        key[item] = qwen_delta_prefill_bf16_to_f32(mixed_qkv[source + KEY_WIDTH + dim]);
        query_sum += query[item] * query[item];
        key_sum += key[item] * key[item];
    }
    float query_inverse = rsqrt(simd_sum(query_sum) + float(head_dim) * 1.0e-6f);
    float key_inverse = rsqrt(simd_sum(key_sum) + float(head_dim) * 1.0e-6f);
    for (uint dim = lane; dim < head_dim; dim += 32u) {
        uint item = dim / 32u;
        float query_rms = qwen_delta_prefill_bf16_to_f32(
            qwen_delta_prefill_f32_to_bf16(query[item] * query_inverse * sqrt(float(head_dim)))
        );
        float key_rms = qwen_delta_prefill_bf16_to_f32(
            qwen_delta_prefill_f32_to_bf16(key[item] * key_inverse * sqrt(float(head_dim)))
        );
        prepared_qk[destination + dim] = qwen_delta_prefill_f32_to_bf16(query_rms / float(head_dim));
        prepared_qk[destination + KEY_WIDTH + dim] = qwen_delta_prefill_f32_to_bf16(key_rms * rsqrt(float(head_dim)));
    }
}

kernel void qwen_delta_prefill_recurrent_kernel(
    const device ushort* mixed_qkv [[buffer(0)]],
    const device ushort* gate [[buffer(1)]],
    const device ushort* input_a [[buffer(2)]],
    const device ushort* input_b [[buffer(3)]],
    const device ushort* a_log [[buffer(4)]],
    const device ushort* dt_bias [[buffer(5)]],
    const device ushort* prepared_qk [[buffer(6)]],
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
    float a_scale = -exp(qwen_delta_prefill_bf16_to_f32(a_log[value_head]));
    float time_bias = qwen_delta_prefill_bf16_to_f32(dt_bias[value_head]);
    uint value_dim = lane + simd_group * 32u;

    // Each lane owns one state column throughout prefill; only persist it at the end.
    float state_column[128];
    #pragma unroll
    for (uint key_dim = 0u; key_dim < 128u; key_dim++) {
        state_column[key_dim] = recurrent_state[state_start + key_dim * 128u + value_dim];
    }

    for (uint token = 0u; token < sequence_length; token++) {
        uint row = batch * sequence_length + token;
        uint value_start = row * mixed_width + key_width * 2u + value_head * head_dim;

        uint qk_row = row * key_width * 2u;
        // Each SIMD group owns its Q/K values and state columns; no cross-group barrier is needed.
        float query[4];
        float key[4];
        #pragma unroll
        for (uint item = 0u; item < 4u; ++item) {
            uint dim = lane + item * 32u;
            query[item] = qwen_delta_prefill_bf16_to_f32(
                prepared_qk[qk_row + key_head * head_dim + dim]
            );
            key[item] = qwen_delta_prefill_bf16_to_f32(
                prepared_qk[qk_row + key_width + key_head * head_dim + dim]
            );
        }
        float a = qwen_delta_prefill_bf16_to_f32(
            input_a[row * value_heads + value_head]
        );
        float beta_input = qwen_delta_prefill_bf16_to_f32(
            input_b[row * value_heads + value_head]
        );
        float beta = 1.0f / (1.0f + exp(-beta_input));
        float decay = exp(
            a_scale * qwen_delta_prefill_softplus(a + time_bias)
        );
        float memory = 0.0f;
        #pragma unroll
        for (uint key_dim = 0u; key_dim < 128u; key_dim++) {
            float state_value = state_column[key_dim] * decay;
            state_column[key_dim] = state_value;
            memory += state_value * simd_shuffle(key[key_dim / 32u], key_dim % 32u);
        }
        float value = qwen_delta_prefill_bf16_to_f32(
            mixed_qkv[value_start + value_dim]
        );
        float delta = (value - memory) * beta;
        float attended = 0.0f;
        #pragma unroll
        for (uint key_dim = 0u; key_dim < 128u; key_dim++) {
            float state_value = state_column[key_dim]
                + simd_shuffle(key[key_dim / 32u], key_dim % 32u) * delta;
            state_column[key_dim] = state_value;
            attended += state_value * simd_shuffle(query[key_dim / 32u], key_dim % 32u);
        }
        // Keep optional checkpoint stores out of the ordered attention reduction.
        if (token < checkpoint_capacity) {
            uint state_values = group_count * head_dim * head_dim;
            #pragma unroll
            for (uint key_dim = 0u; key_dim < 128u; key_dim++) {
                uint state_index = state_start + key_dim * head_dim + value_dim;
                checkpoint_state[token * state_values + state_index] = state_column[key_dim];
            }
        }
        uint output_index = row * value_width + value_head * head_dim + value_dim;
        output[output_index] = qwen_delta_prefill_f32_to_bf16(attended);
    }

    #pragma unroll
    for (uint key_dim = 0u; key_dim < 128u; key_dim++) {
        recurrent_state[state_start + key_dim * 128u + value_dim] = state_column[key_dim];
    }
}

kernel void qwen_delta_prefill_output_norm_kernel(
    device ushort* output [[buffer(0)]],
    const device ushort* gate [[buffer(1)]],
    const device ushort* norm_weight [[buffer(2)]],
    constant float& epsilon [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint HEAD_DIM = 128u;
    threadgroup float inverse_rms;
    uint value_dim = lane + simd_group * 32u;
    uint offset = group * HEAD_DIM;
    float attended = qwen_delta_prefill_bf16_to_f32(output[offset + value_dim]);
    if (simd_group == 0u) {
        float square_sum = 0.0f;
        for (uint item = 0u; item < 4u; item++) {
            float value = qwen_delta_prefill_bf16_to_f32(output[offset + lane + item * 32u]);
            square_sum += value * value;
        }
        float inverse = rsqrt(simd_sum(square_sum) / float(HEAD_DIM) + epsilon);
        if (lane == 0u) {
            inverse_rms = inverse;
        }
    }
    // All reads of the unnormalized row must complete before its in-place update.
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float normalized = qwen_delta_prefill_bf16_to_f32(
        qwen_delta_prefill_f32_to_bf16(attended * inverse_rms)
    ) * qwen_delta_prefill_bf16_to_f32(norm_weight[value_dim]);
    normalized = qwen_delta_prefill_bf16_to_f32(
        qwen_delta_prefill_f32_to_bf16(normalized)
    );
    float gate_value = qwen_delta_prefill_bf16_to_f32(gate[offset + value_dim]);
    output[offset + value_dim] = qwen_delta_prefill_f32_to_bf16(
        normalized * qwen_delta_prefill_silu(gate_value)
    );
}
