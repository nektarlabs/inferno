use ::metal::{Buffer, CommandBufferRef, ComputePipelineState, Device};
use common::{validate_exact_shape, Error, Result};

use super::{
    arena::MetalArena,
    buffers::{read_u32_buffer, require_f32_capacity, ImmutableF32BufferCache},
    command::encode_1d,
    library::MetalLibrary,
    pipeline::compute_pipeline,
};

const DSA_KEY_NORM_ROPE_KERNEL: &str = "dsa_key_norm_rope_f32_kernel";
const DSA_QUERY_WEIGHTS_KERNEL: &str = "dsa_query_weights_f32_kernel";
const DSA_SCORES_KERNEL: &str = "dsa_scores_f32_kernel";
const DSA_TOPK_KERNEL: &str = "dsa_topk_scores_u32_kernel";
const INDEXER_LAYER_NORM_EPS: f32 = 1e-6;
const MAX_DSA_TOP_K: usize = 2048;

pub(crate) struct MetalDsa {
    arena: MetalArena,
    key_norm_rope_pipeline: ComputePipelineState,
    query_weights_pipeline: ComputePipelineState,
    scores_pipeline: ComputePipelineState,
    topk_pipeline: ComputePipelineState,
    weight_buffers: ImmutableF32BufferCache,
}

#[derive(Debug)]
pub(crate) struct MetalDsaTopKBuffers {
    pub(crate) token_ids: Buffer,
    pub(crate) output_len: usize,
}

impl MetalDsa {
    pub(crate) fn new(device: &Device, library: &MetalLibrary, arena: MetalArena) -> Result<Self> {
        Ok(Self {
            arena,
            key_norm_rope_pipeline: compute_pipeline(device, library, DSA_KEY_NORM_ROPE_KERNEL)?,
            query_weights_pipeline: compute_pipeline(device, library, DSA_QUERY_WEIGHTS_KERNEL)?,
            scores_pipeline: compute_pipeline(device, library, DSA_SCORES_KERNEL)?,
            topk_pipeline: compute_pipeline(device, library, DSA_TOPK_KERNEL)?,
            weight_buffers: ImmutableF32BufferCache::default(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_key_norm_rope(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        raw_key: &Buffer,
        raw_key_len: usize,
        weight: &[f32],
        bias: &[f32],
        batch: usize,
        tokens: usize,
        head_dim: usize,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
    ) -> Result<Buffer> {
        if batch == 0 || tokens == 0 || head_dim == 0 {
            return Err(Error::backend(
                "DSA key norm/RoPE requires non-zero batch, tokens and head_dim",
            ));
        }
        validate_exact_shape("DSA k norm weight", &[weight.len()], &[head_dim])?;
        validate_exact_shape("DSA k norm bias", &[bias.len()], &[head_dim])?;
        if rope_dim == 0 || rope_dim > head_dim || rope_dim % 2 != 0 {
            return Err(Error::backend(format!(
                "DSA key norm/RoPE invalid rope_dim {rope_dim} for head_dim {head_dim}"
            )));
        }
        let expected_len = batch
            .checked_mul(tokens)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| Error::backend("DSA key norm/RoPE length overflow"))?;
        validate_exact_shape("DSA raw key len", &[raw_key_len], &[expected_len])?;
        require_f32_capacity(raw_key, raw_key_len, "DSA raw key")?;

        let output = self.arena.empty_f32(expected_len)?;
        let weight = self.weight_buffers.get(device, weight)?;
        let bias = self.weight_buffers.get(device, bias)?;
        let row_count = batch
            .checked_mul(tokens)
            .ok_or_else(|| Error::backend("DSA key norm/RoPE row count overflow"))?;
        let row_count_buffer = self.arena.u32(to_u32(row_count, "DSA row_count")?)?;
        let token_count_buffer = self.arena.u32(to_u32(tokens, "DSA token_count")?)?;
        let head_dim_buffer = self.arena.u32(to_u32(head_dim, "DSA head_dim")?)?;
        let rope_dim_buffer = self.arena.u32(to_u32(rope_dim, "DSA rope_dim")?)?;
        let position_offset_buffer = self
            .arena
            .u32(to_u32(position_offset, "DSA position_offset")?)?;
        let theta_buffer = self.arena.f32(theta)?;
        let eps_buffer = self.arena.f32(INDEXER_LAYER_NORM_EPS)?;

        encode_1d(
            command_buffer,
            &self.key_norm_rope_pipeline,
            &[
                raw_key,
                &weight,
                &bias,
                &output,
                &row_count_buffer,
                &token_count_buffer,
                &head_dim_buffer,
                &rope_dim_buffer,
                &position_offset_buffer,
                &theta_buffer,
                &eps_buffer,
            ],
            row_count,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_decode_topk(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        hidden_states: &Buffer,
        hidden_states_len: usize,
        q_raw: &Buffer,
        q_raw_len: usize,
        past_index_keys: &Buffer,
        past_index_keys_len: usize,
        current_index_key: &Buffer,
        current_index_key_len: usize,
        weights_proj: &[f32],
        batch: usize,
        hidden_size: usize,
        past_tokens: usize,
        heads: usize,
        head_dim: usize,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
        top_k: usize,
    ) -> Result<MetalDsaTopKBuffers> {
        if batch == 0 || hidden_size == 0 || heads == 0 || head_dim == 0 {
            return Err(Error::backend(
                "DSA decode top-k requires non-zero batch, hidden_size, heads and head_dim",
            ));
        }
        if top_k == 0 || top_k > MAX_DSA_TOP_K {
            return Err(Error::backend(format!(
                "DSA decode top-k supports 1..={MAX_DSA_TOP_K}, got {top_k}"
            )));
        }
        let key_tokens = past_tokens
            .checked_add(1)
            .ok_or_else(|| Error::backend("DSA decode key token count overflow"))?;
        if top_k > key_tokens {
            return Err(Error::backend(format!(
                "DSA decode top_k {top_k} exceeds key token count {key_tokens}"
            )));
        }
        if rope_dim == 0 || rope_dim > head_dim || rope_dim % 2 != 0 {
            return Err(Error::backend(format!(
                "DSA decode top-k invalid rope_dim {rope_dim} for head_dim {head_dim}"
            )));
        }

        let expected_hidden = batch
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("DSA decode hidden length overflow"))?;
        let expected_q = batch
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| Error::backend("DSA decode q length overflow"))?;
        let expected_past = batch
            .checked_mul(past_tokens)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| Error::backend("DSA decode past key length overflow"))?;
        let expected_current = batch
            .checked_mul(head_dim)
            .ok_or_else(|| Error::backend("DSA decode current key length overflow"))?;
        let expected_weights = hidden_size
            .checked_mul(heads)
            .ok_or_else(|| Error::backend("DSA decode weights projection length overflow"))?;

        validate_exact_shape(
            "DSA decode hidden len",
            &[hidden_states_len],
            &[expected_hidden],
        )?;
        validate_exact_shape("DSA decode q raw len", &[q_raw_len], &[expected_q])?;
        validate_exact_shape(
            "DSA decode past index key len",
            &[past_index_keys_len],
            &[expected_past],
        )?;
        validate_exact_shape(
            "DSA decode current index key len",
            &[current_index_key_len],
            &[expected_current],
        )?;
        validate_exact_shape(
            "DSA decode weights projection len",
            &[weights_proj.len()],
            &[expected_weights],
        )?;

        require_f32_capacity(hidden_states, hidden_states_len, "DSA hidden states")?;
        require_f32_capacity(q_raw, q_raw_len, "DSA q raw")?;
        require_f32_capacity(past_index_keys, past_index_keys_len, "DSA past index keys")?;
        require_f32_capacity(
            current_index_key,
            current_index_key_len,
            "DSA current index key",
        )?;

        let weights_proj = self.weight_buffers.get(device, weights_proj)?;
        let q_output = self.arena.empty_f32(expected_q)?;
        let weights = self.arena.empty_f32(
            batch
                .checked_mul(heads)
                .ok_or_else(|| Error::backend("DSA decode weights length overflow"))?,
        )?;
        let scores = self.arena.empty_f32(
            batch
                .checked_mul(key_tokens)
                .ok_or_else(|| Error::backend("DSA decode scores length overflow"))?,
        )?;
        let output_len = batch
            .checked_mul(top_k)
            .ok_or_else(|| Error::backend("DSA top-k output length overflow"))?;
        let token_ids = self.arena.empty_u32(output_len)?;

        let batch_buffer = self.arena.u32(to_u32(batch, "DSA batch")?)?;
        let hidden_size_buffer = self.arena.u32(to_u32(hidden_size, "DSA hidden")?)?;
        let heads_buffer = self.arena.u32(to_u32(heads, "DSA heads")?)?;
        let head_dim_buffer = self.arena.u32(to_u32(head_dim, "DSA head_dim")?)?;
        let rope_dim_buffer = self.arena.u32(to_u32(rope_dim, "DSA rope_dim")?)?;
        let position_offset_buffer = self
            .arena
            .u32(to_u32(position_offset, "DSA position_offset")?)?;
        let theta_buffer = self.arena.f32(theta)?;

        encode_1d(
            command_buffer,
            &self.query_weights_pipeline,
            &[
                hidden_states,
                q_raw,
                &weights_proj,
                &q_output,
                &weights,
                &batch_buffer,
                &hidden_size_buffer,
                &heads_buffer,
                &head_dim_buffer,
                &rope_dim_buffer,
                &position_offset_buffer,
                &theta_buffer,
            ],
            batch
                .checked_mul(heads)
                .ok_or_else(|| Error::backend("DSA query thread count overflow"))?,
        )?;

        let past_tokens_buffer = self.arena.u32(to_u32(past_tokens, "DSA past_tokens")?)?;
        encode_1d(
            command_buffer,
            &self.scores_pipeline,
            &[
                &q_output,
                &weights,
                past_index_keys,
                current_index_key,
                &scores,
                &batch_buffer,
                &past_tokens_buffer,
                &heads_buffer,
                &head_dim_buffer,
            ],
            batch
                .checked_mul(key_tokens)
                .ok_or_else(|| Error::backend("DSA score thread count overflow"))?,
        )?;

        let key_tokens_buffer = self.arena.u32(to_u32(key_tokens, "DSA key_tokens")?)?;
        let top_k_buffer = self.arena.u32(to_u32(top_k, "DSA top_k")?)?;
        encode_1d(
            command_buffer,
            &self.topk_pipeline,
            &[
                &scores,
                &token_ids,
                &batch_buffer,
                &key_tokens_buffer,
                &top_k_buffer,
            ],
            batch,
        )?;

        Ok(MetalDsaTopKBuffers {
            token_ids,
            output_len,
        })
    }

    pub(crate) fn read_token_ids(buffer: &Buffer, len: usize) -> Result<Vec<u32>> {
        read_u32_buffer(buffer, len)
    }
}

fn to_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::backend(format!("{label} exceeds Metal u32 limit")))
}
