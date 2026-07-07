#include <metal_stdlib>

using namespace metal;

kernel void select_last_token_f32_kernel(
    const device float* hidden_states [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_count [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    constant uint& hidden_size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = batch_count * hidden_size;
    if (gid >= output_values) {
        return;
    }

    uint batch = gid / hidden_size;
    uint hidden = gid - (batch * hidden_size);
    uint source_index = ((batch * token_count + (token_count - 1)) * hidden_size) + hidden;

    output[gid] = hidden_states[source_index];
}

kernel void heads_to_attention_layout_f32_kernel(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_count [[buffer(2)]],
    constant uint& token_count [[buffer(3)]],
    constant uint& head_count [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = batch_count * head_count * token_count * head_dim;
    if (gid >= output_values) {
        return;
    }

    uint values_per_batch = head_count * token_count * head_dim;
    uint values_per_head = token_count * head_dim;
    uint batch = gid / values_per_batch;
    uint batch_rem = gid - (batch * values_per_batch);
    uint head = batch_rem / values_per_head;
    uint head_rem = batch_rem - (head * values_per_head);
    uint token = head_rem / head_dim;
    uint dim = head_rem - (token * head_dim);

    uint source_index = (((batch * token_count + token) * head_count + head) * head_dim) + dim;
    output[gid] = input[source_index];
}

kernel void merge_attention_heads_f32_kernel(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_count [[buffer(2)]],
    constant uint& head_count [[buffer(3)]],
    constant uint& token_count [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = batch_count * token_count * head_count * head_dim;
    if (gid >= output_values) {
        return;
    }

    uint values_per_batch = token_count * head_count * head_dim;
    uint values_per_token = head_count * head_dim;
    uint batch = gid / values_per_batch;
    uint batch_rem = gid - (batch * values_per_batch);
    uint token = batch_rem / values_per_token;
    uint token_rem = batch_rem - (token * values_per_token);
    uint head = token_rem / head_dim;
    uint dim = token_rem - (head * head_dim);

    uint source_index = (((batch * head_count + head) * token_count + token) * head_dim) + dim;
    output[gid] = input[source_index];
}

kernel void split_rope_tail_f32_kernel(
    const device float* input [[buffer(0)]],
    device float* no_rope [[buffer(1)]],
    device float* rope [[buffer(2)]],
    constant uint& batch_count [[buffer(3)]],
    constant uint& token_count [[buffer(4)]],
    constant uint& head_count [[buffer(5)]],
    constant uint& no_rope_dim [[buffer(6)]],
    constant uint& rope_dim [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    uint no_rope_values = batch_count * token_count * head_count * no_rope_dim;
    uint rope_values = batch_count * token_count * head_count * rope_dim;
    uint output_values = no_rope_values + rope_values;
    if (gid >= output_values) {
        return;
    }

    uint total_dim = no_rope_dim + rope_dim;

    if (gid < no_rope_values) {
        uint dim = gid % no_rope_dim;
        uint token_head_index = gid / no_rope_dim;
        uint source_index = (token_head_index * total_dim) + dim;
        no_rope[gid] = input[source_index];
        return;
    }

    uint rope_gid = gid - no_rope_values;
    uint dim = rope_gid % rope_dim;
    uint token_head_index = rope_gid / rope_dim;
    uint source_index = (token_head_index * total_dim) + no_rope_dim + dim;
    rope[rope_gid] = input[source_index];
}

kernel void split_kv_mqa_f32_kernel(
    const device float* input [[buffer(0)]],
    device float* kv_latent [[buffer(1)]],
    device float* k_rope [[buffer(2)]],
    constant uint& batch_count [[buffer(3)]],
    constant uint& token_count [[buffer(4)]],
    constant uint& kv_lora_rank [[buffer(5)]],
    constant uint& rope_dim [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint latent_values = batch_count * token_count * kv_lora_rank;
    uint rope_values = batch_count * token_count * rope_dim;
    uint output_values = latent_values + rope_values;
    if (gid >= output_values) {
        return;
    }

    uint total_dim = kv_lora_rank + rope_dim;

    if (gid < latent_values) {
        uint dim = gid % kv_lora_rank;
        uint token_index = gid / kv_lora_rank;
        uint source_index = (token_index * total_dim) + dim;
        kv_latent[gid] = input[source_index];
        return;
    }

    uint rope_gid = gid - latent_values;
    uint dim = rope_gid % rope_dim;
    uint token_index = rope_gid / rope_dim;
    uint source_index = (token_index * total_dim) + kv_lora_rank + dim;
    k_rope[rope_gid] = input[source_index];
}

kernel void combine_rope_tail_f32_kernel(
    const device float* no_rope [[buffer(0)]],
    const device float* rope [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch_count [[buffer(3)]],
    constant uint& token_count [[buffer(4)]],
    constant uint& head_count [[buffer(5)]],
    constant uint& rope_head_count [[buffer(6)]],
    constant uint& no_rope_dim [[buffer(7)]],
    constant uint& rope_dim [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total_dim = no_rope_dim + rope_dim;
    uint output_values = batch_count * token_count * head_count * total_dim;
    if (gid >= output_values) {
        return;
    }

    uint dim = gid % total_dim;
    uint token_head_index = gid / total_dim;
    uint head = token_head_index % head_count;
    uint token = (token_head_index / head_count) % token_count;
    uint batch = token_head_index / (head_count * token_count);

    if (dim < no_rope_dim) {
        uint source_index = (token_head_index * no_rope_dim) + dim;
        output[gid] = no_rope[source_index];
        return;
    }

    uint rope_head = rope_head_count == 1 ? 0 : head;
    uint rope_index =
        (((batch * token_count + token) * rope_head_count + rope_head) * rope_dim)
        + (dim - no_rope_dim);
    output[gid] = rope[rope_index];
}

kernel void stack_head_output_f32_kernel(
    const device float* head_input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    constant uint& head_count [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& output_head_index [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint input_values = row_count * head_dim;
    if (gid >= input_values) {
        return;
    }

    uint row = gid / head_dim;
    uint dim = gid - (row * head_dim);
    uint output_index = ((row * head_count + output_head_index) * head_dim) + dim;
    output[output_index] = head_input[gid];
}

kernel void linearize_paged_cache_f32_kernel(
    const device float* paged [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_count [[buffer(2)]],
    constant uint& head_count [[buffer(3)]],
    constant uint& cached_tokens [[buffer(4)]],
    constant uint& page_size [[buffer(5)]],
    constant uint& head_dim [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint output_values = batch_count * head_count * cached_tokens * head_dim;
    if (gid >= output_values) {
        return;
    }

    uint dim = gid % head_dim;
    uint token_index = (gid / head_dim) % cached_tokens;
    uint head = (gid / (head_dim * cached_tokens)) % head_count;
    uint batch = gid / (head_dim * cached_tokens * head_count);
    uint page = token_index / page_size;
    uint page_offset = token_index - (page * page_size);

    uint source_index =
        ((((page * batch_count + batch) * head_count + head) * page_size + page_offset)
        * head_dim)
        + dim;
    output[gid] = paged[source_index];
}
