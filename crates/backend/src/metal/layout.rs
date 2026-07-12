use ::metal::{Buffer, CommandBufferRef, CommandQueue, ComputePipelineState, Device};
use common::{Error, Result};
use tracing::trace;

use super::{
    buffers::{
        empty_f32_buffer, f32_buffer, read_f32_buffer, require_f32_capacity, u32_scalar_buffer,
    },
    command::{dispatch_1d, encode_1d},
    library::MetalLibrary,
    pipeline::compute_pipeline,
    validation::{
        validate_combine_rope_tail_f32, validate_heads_to_attention_layout_f32,
        validate_merge_attention_heads_f32, validate_select_last_token_f32,
        validate_split_kv_mqa_f32, validate_split_rope_tail_f32,
    },
};

const SELECT_LAST_TOKEN_KERNEL: &str = "select_last_token_f32_kernel";
const HEADS_TO_ATTENTION_LAYOUT_KERNEL: &str = "heads_to_attention_layout_f32_kernel";
const MERGE_ATTENTION_HEADS_KERNEL: &str = "merge_attention_heads_f32_kernel";
const SPLIT_ROPE_TAIL_KERNEL: &str = "split_rope_tail_f32_kernel";
const SPLIT_KV_MQA_KERNEL: &str = "split_kv_mqa_f32_kernel";
const COMBINE_ROPE_TAIL_KERNEL: &str = "combine_rope_tail_f32_kernel";
const STACK_HEAD_OUTPUT_KERNEL: &str = "stack_head_output_f32_kernel";
const LINEARIZE_PAGED_CACHE_KERNEL: &str = "linearize_paged_cache_f32_kernel";

pub(crate) struct MetalLayout {
    select_last_token_pipeline: ComputePipelineState,
    heads_to_attention_layout_pipeline: ComputePipelineState,
    merge_attention_heads_pipeline: ComputePipelineState,
    split_rope_tail_pipeline: ComputePipelineState,
    split_kv_mqa_pipeline: ComputePipelineState,
    combine_rope_tail_pipeline: ComputePipelineState,
    stack_head_output_pipeline: ComputePipelineState,
    linearize_paged_cache_pipeline: ComputePipelineState,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalSelectLastTokenReport {
    pub values: Vec<f32>,
    pub batch_count: usize,
    pub token_count: usize,
    pub hidden_size: usize,
    pub thread_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalHeadsToAttentionLayoutReport {
    pub values: Vec<f32>,
    pub batch_count: usize,
    pub token_count: usize,
    pub head_count: usize,
    pub head_dim: usize,
    pub thread_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalMergeAttentionHeadsReport {
    pub values: Vec<f32>,
    pub batch_count: usize,
    pub token_count: usize,
    pub head_count: usize,
    pub head_dim: usize,
    pub thread_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalSplitRopeTailReport {
    pub no_rope_values: Vec<f32>,
    pub rope_values: Vec<f32>,
    pub batch_count: usize,
    pub token_count: usize,
    pub head_count: usize,
    pub no_rope_dim: usize,
    pub rope_dim: usize,
    pub thread_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalSplitKvMqaReport {
    pub kv_latent_values: Vec<f32>,
    pub k_rope_values: Vec<f32>,
    pub batch_count: usize,
    pub token_count: usize,
    pub kv_lora_rank: usize,
    pub rope_dim: usize,
    pub thread_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalCombineRopeTailReport {
    pub values: Vec<f32>,
    pub batch_count: usize,
    pub token_count: usize,
    pub head_count: usize,
    pub rope_head_count: usize,
    pub no_rope_dim: usize,
    pub rope_dim: usize,
    pub thread_count: usize,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub struct MetalStackHeadOutputsReport {
    pub values: Vec<f32>,
    pub row_count: usize,
    pub head_count: usize,
    pub head_dim: usize,
    pub thread_count: usize,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub struct MetalLinearizePagedCacheReport {
    pub values: Vec<f32>,
    pub batch_count: usize,
    pub head_count: usize,
    pub cached_tokens: usize,
    pub page_size: usize,
    pub head_dim: usize,
    pub thread_count: usize,
}

impl MetalLayout {
    pub(crate) fn new(device: &Device, library: &MetalLibrary) -> Result<Self> {
        Ok(Self {
            select_last_token_pipeline: compute_pipeline(
                device,
                library,
                SELECT_LAST_TOKEN_KERNEL,
            )?,
            heads_to_attention_layout_pipeline: compute_pipeline(
                device,
                library,
                HEADS_TO_ATTENTION_LAYOUT_KERNEL,
            )?,
            merge_attention_heads_pipeline: compute_pipeline(
                device,
                library,
                MERGE_ATTENTION_HEADS_KERNEL,
            )?,
            split_rope_tail_pipeline: compute_pipeline(device, library, SPLIT_ROPE_TAIL_KERNEL)?,
            split_kv_mqa_pipeline: compute_pipeline(device, library, SPLIT_KV_MQA_KERNEL)?,
            combine_rope_tail_pipeline: compute_pipeline(
                device,
                library,
                COMBINE_ROPE_TAIL_KERNEL,
            )?,
            stack_head_output_pipeline: compute_pipeline(
                device,
                library,
                STACK_HEAD_OUTPUT_KERNEL,
            )?,
            linearize_paged_cache_pipeline: compute_pipeline(
                device,
                library,
                LINEARIZE_PAGED_CACHE_KERNEL,
            )?,
        })
    }

    pub(crate) fn select_last_token(
        &self,
        device: &Device,
        queue: &CommandQueue,
        hidden_states: &[f32],
        batch_count: usize,
        token_count: usize,
        hidden_size: usize,
    ) -> Result<MetalSelectLastTokenReport> {
        validate_select_last_token_f32(hidden_states, batch_count, token_count, hidden_size)?;
        let output_len = batch_count
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("select_last_token output length overflow"))?;

        let batch_count_u32 = u32::try_from(batch_count)
            .map_err(|_| Error::backend("select_last_token batch_count exceeds Metal u32 limit"))?;
        let token_count_u32 = u32::try_from(token_count)
            .map_err(|_| Error::backend("select_last_token token_count exceeds Metal u32 limit"))?;
        let hidden_size_u32 = u32::try_from(hidden_size)
            .map_err(|_| Error::backend("select_last_token hidden_size exceeds Metal u32 limit"))?;

        let hidden_states_buffer = f32_buffer(device, hidden_states)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let batch_count_buffer = u32_scalar_buffer(device, batch_count_u32)?;
        let token_count_buffer = u32_scalar_buffer(device, token_count_u32)?;
        let hidden_size_buffer = u32_scalar_buffer(device, hidden_size_u32)?;

        trace!(
            target: "inferno::metal",
            batch_count,
            token_count,
            hidden_size,
            "running native Metal select_last_token"
        );

        dispatch_1d(
            queue,
            &self.select_last_token_pipeline,
            &[
                &hidden_states_buffer,
                &output_buffer,
                &batch_count_buffer,
                &token_count_buffer,
                &hidden_size_buffer,
            ],
            output_len,
        )?;

        let values = read_f32_buffer(&output_buffer, output_len)?;

        Ok(MetalSelectLastTokenReport {
            values,
            batch_count,
            token_count,
            hidden_size,
            thread_count: output_len,
        })
    }

    pub(crate) fn heads_to_attention_layout(
        &self,
        device: &Device,
        queue: &CommandQueue,
        input: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        head_dim: usize,
    ) -> Result<MetalHeadsToAttentionLayoutReport> {
        validate_heads_to_attention_layout_f32(
            input,
            batch_count,
            token_count,
            head_count,
            head_dim,
        )?;
        let output_len =
            attention_head_value_count(batch_count, token_count, head_count, head_dim)?;

        let batch_count_buffer = u32_scalar_buffer(
            device,
            u32::try_from(batch_count).map_err(|_| {
                Error::backend("heads_to_attention_layout batch_count exceeds Metal u32 limit")
            })?,
        )?;
        let token_count_buffer = u32_scalar_buffer(
            device,
            u32::try_from(token_count).map_err(|_| {
                Error::backend("heads_to_attention_layout token_count exceeds Metal u32 limit")
            })?,
        )?;
        let head_count_buffer = u32_scalar_buffer(
            device,
            u32::try_from(head_count).map_err(|_| {
                Error::backend("heads_to_attention_layout head_count exceeds Metal u32 limit")
            })?,
        )?;
        let head_dim_buffer = u32_scalar_buffer(
            device,
            u32::try_from(head_dim).map_err(|_| {
                Error::backend("heads_to_attention_layout head_dim exceeds Metal u32 limit")
            })?,
        )?;
        let input_buffer = f32_buffer(device, input)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;

        trace!(
            target: "inferno::metal",
            batch_count,
            token_count,
            head_count,
            head_dim,
            "running native Metal heads_to_attention_layout"
        );

        dispatch_1d(
            queue,
            &self.heads_to_attention_layout_pipeline,
            &[
                &input_buffer,
                &output_buffer,
                &batch_count_buffer,
                &token_count_buffer,
                &head_count_buffer,
                &head_dim_buffer,
            ],
            output_len,
        )?;

        let values = read_f32_buffer(&output_buffer, output_len)?;

        Ok(MetalHeadsToAttentionLayoutReport {
            values,
            batch_count,
            token_count,
            head_count,
            head_dim,
            thread_count: output_len,
        })
    }

    pub(crate) fn merge_attention_heads(
        &self,
        device: &Device,
        queue: &CommandQueue,
        input: &[f32],
        batch_count: usize,
        head_count: usize,
        token_count: usize,
        head_dim: usize,
    ) -> Result<MetalMergeAttentionHeadsReport> {
        validate_merge_attention_heads_f32(input, batch_count, head_count, token_count, head_dim)?;
        let output_len =
            attention_head_value_count(batch_count, token_count, head_count, head_dim)?;

        let batch_count_buffer = u32_scalar_buffer(
            device,
            u32::try_from(batch_count).map_err(|_| {
                Error::backend("merge_attention_heads batch_count exceeds Metal u32 limit")
            })?,
        )?;
        let head_count_buffer = u32_scalar_buffer(
            device,
            u32::try_from(head_count).map_err(|_| {
                Error::backend("merge_attention_heads head_count exceeds Metal u32 limit")
            })?,
        )?;
        let token_count_buffer = u32_scalar_buffer(
            device,
            u32::try_from(token_count).map_err(|_| {
                Error::backend("merge_attention_heads token_count exceeds Metal u32 limit")
            })?,
        )?;
        let head_dim_buffer = u32_scalar_buffer(
            device,
            u32::try_from(head_dim).map_err(|_| {
                Error::backend("merge_attention_heads head_dim exceeds Metal u32 limit")
            })?,
        )?;
        let input_buffer = f32_buffer(device, input)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;

        trace!(
            target: "inferno::metal",
            batch_count,
            token_count,
            head_count,
            head_dim,
            "running native Metal merge_attention_heads"
        );

        dispatch_1d(
            queue,
            &self.merge_attention_heads_pipeline,
            &[
                &input_buffer,
                &output_buffer,
                &batch_count_buffer,
                &head_count_buffer,
                &token_count_buffer,
                &head_dim_buffer,
            ],
            output_len,
        )?;

        let values = read_f32_buffer(&output_buffer, output_len)?;

        Ok(MetalMergeAttentionHeadsReport {
            values,
            batch_count,
            token_count,
            head_count,
            head_dim,
            thread_count: output_len,
        })
    }

    pub(crate) fn split_rope_tail(
        &self,
        device: &Device,
        queue: &CommandQueue,
        input: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        no_rope_dim: usize,
        rope_dim: usize,
    ) -> Result<MetalSplitRopeTailReport> {
        validate_split_rope_tail_f32(
            input,
            batch_count,
            token_count,
            head_count,
            no_rope_dim,
            rope_dim,
        )?;
        let no_rope_len =
            attention_head_value_count(batch_count, token_count, head_count, no_rope_dim)?;
        let rope_len = attention_head_value_count(batch_count, token_count, head_count, rope_dim)?;
        let thread_count = no_rope_len
            .checked_add(rope_len)
            .ok_or_else(|| Error::backend("split_rope_tail thread count overflow"))?;

        let input_buffer = f32_buffer(device, input)?;
        let no_rope_buffer = empty_f32_buffer(device, no_rope_len)?;
        let rope_buffer = empty_f32_buffer(device, rope_len)?;
        let batch_count_buffer = u32_scalar_buffer(
            device,
            u32::try_from(batch_count).map_err(|_| {
                Error::backend("split_rope_tail batch_count exceeds Metal u32 limit")
            })?,
        )?;
        let token_count_buffer = u32_scalar_buffer(
            device,
            u32::try_from(token_count).map_err(|_| {
                Error::backend("split_rope_tail token_count exceeds Metal u32 limit")
            })?,
        )?;
        let head_count_buffer = u32_scalar_buffer(
            device,
            u32::try_from(head_count).map_err(|_| {
                Error::backend("split_rope_tail head_count exceeds Metal u32 limit")
            })?,
        )?;
        let no_rope_dim_buffer = u32_scalar_buffer(
            device,
            u32::try_from(no_rope_dim).map_err(|_| {
                Error::backend("split_rope_tail no_rope_dim exceeds Metal u32 limit")
            })?,
        )?;
        let rope_dim_buffer = u32_scalar_buffer(
            device,
            u32::try_from(rope_dim)
                .map_err(|_| Error::backend("split_rope_tail rope_dim exceeds Metal u32 limit"))?,
        )?;

        trace!(
            target: "inferno::metal",
            batch_count,
            token_count,
            head_count,
            no_rope_dim,
            rope_dim,
            "running native Metal split_rope_tail"
        );

        dispatch_1d(
            queue,
            &self.split_rope_tail_pipeline,
            &[
                &input_buffer,
                &no_rope_buffer,
                &rope_buffer,
                &batch_count_buffer,
                &token_count_buffer,
                &head_count_buffer,
                &no_rope_dim_buffer,
                &rope_dim_buffer,
            ],
            thread_count,
        )?;

        let no_rope_values = read_f32_buffer(&no_rope_buffer, no_rope_len)?;
        let rope_values = read_f32_buffer(&rope_buffer, rope_len)?;

        Ok(MetalSplitRopeTailReport {
            no_rope_values,
            rope_values,
            batch_count,
            token_count,
            head_count,
            no_rope_dim,
            rope_dim,
            thread_count,
        })
    }

    pub(crate) fn split_kv_mqa(
        &self,
        device: &Device,
        queue: &CommandQueue,
        input: &[f32],
        batch_count: usize,
        token_count: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
    ) -> Result<MetalSplitKvMqaReport> {
        validate_split_kv_mqa_f32(input, batch_count, token_count, kv_lora_rank, rope_dim)?;
        let latent_len = batch_count
            .checked_mul(token_count)
            .and_then(|value| value.checked_mul(kv_lora_rank))
            .ok_or_else(|| Error::backend("split_kv_mqa latent length overflow"))?;
        let rope_len = batch_count
            .checked_mul(token_count)
            .and_then(|value| value.checked_mul(rope_dim))
            .ok_or_else(|| Error::backend("split_kv_mqa rope length overflow"))?;
        let thread_count = latent_len
            .checked_add(rope_len)
            .ok_or_else(|| Error::backend("split_kv_mqa thread count overflow"))?;

        let input_buffer = f32_buffer(device, input)?;
        let latent_buffer = empty_f32_buffer(device, latent_len)?;
        let rope_buffer = empty_f32_buffer(device, rope_len)?;
        let batch_count_buffer = u32_scalar_buffer(
            device,
            u32::try_from(batch_count)
                .map_err(|_| Error::backend("split_kv_mqa batch_count exceeds Metal u32 limit"))?,
        )?;
        let token_count_buffer = u32_scalar_buffer(
            device,
            u32::try_from(token_count)
                .map_err(|_| Error::backend("split_kv_mqa token_count exceeds Metal u32 limit"))?,
        )?;
        let kv_lora_rank_buffer = u32_scalar_buffer(
            device,
            u32::try_from(kv_lora_rank)
                .map_err(|_| Error::backend("split_kv_mqa kv_lora_rank exceeds Metal u32 limit"))?,
        )?;
        let rope_dim_buffer = u32_scalar_buffer(
            device,
            u32::try_from(rope_dim)
                .map_err(|_| Error::backend("split_kv_mqa rope_dim exceeds Metal u32 limit"))?,
        )?;

        trace!(
            target: "inferno::metal",
            batch_count,
            token_count,
            kv_lora_rank,
            rope_dim,
            "running native Metal split_kv_mqa"
        );

        dispatch_1d(
            queue,
            &self.split_kv_mqa_pipeline,
            &[
                &input_buffer,
                &latent_buffer,
                &rope_buffer,
                &batch_count_buffer,
                &token_count_buffer,
                &kv_lora_rank_buffer,
                &rope_dim_buffer,
            ],
            thread_count,
        )?;

        let kv_latent_values = read_f32_buffer(&latent_buffer, latent_len)?;
        let k_rope_values = read_f32_buffer(&rope_buffer, rope_len)?;

        Ok(MetalSplitKvMqaReport {
            kv_latent_values,
            k_rope_values,
            batch_count,
            token_count,
            kv_lora_rank,
            rope_dim,
            thread_count,
        })
    }

    pub(crate) fn combine_rope_tail(
        &self,
        device: &Device,
        queue: &CommandQueue,
        no_rope: &[f32],
        rope: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        rope_head_count: usize,
        no_rope_dim: usize,
        rope_dim: usize,
    ) -> Result<MetalCombineRopeTailReport> {
        validate_combine_rope_tail_f32(
            no_rope,
            rope,
            batch_count,
            token_count,
            head_count,
            rope_head_count,
            no_rope_dim,
            rope_dim,
        )?;
        let total_dim = no_rope_dim
            .checked_add(rope_dim)
            .ok_or_else(|| Error::backend("combine_rope_tail total dim overflow"))?;
        let output_len =
            attention_head_value_count(batch_count, token_count, head_count, total_dim)?;

        let no_rope_buffer = f32_buffer(device, no_rope)?;
        let rope_buffer = f32_buffer(device, rope)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let batch_count_buffer = u32_scalar_buffer(
            device,
            u32::try_from(batch_count).map_err(|_| {
                Error::backend("combine_rope_tail batch_count exceeds Metal u32 limit")
            })?,
        )?;
        let token_count_buffer = u32_scalar_buffer(
            device,
            u32::try_from(token_count).map_err(|_| {
                Error::backend("combine_rope_tail token_count exceeds Metal u32 limit")
            })?,
        )?;
        let head_count_buffer = u32_scalar_buffer(
            device,
            u32::try_from(head_count).map_err(|_| {
                Error::backend("combine_rope_tail head_count exceeds Metal u32 limit")
            })?,
        )?;
        let rope_head_count_buffer = u32_scalar_buffer(
            device,
            u32::try_from(rope_head_count).map_err(|_| {
                Error::backend("combine_rope_tail rope_head_count exceeds Metal u32 limit")
            })?,
        )?;
        let no_rope_dim_buffer = u32_scalar_buffer(
            device,
            u32::try_from(no_rope_dim).map_err(|_| {
                Error::backend("combine_rope_tail no_rope_dim exceeds Metal u32 limit")
            })?,
        )?;
        let rope_dim_buffer = u32_scalar_buffer(
            device,
            u32::try_from(rope_dim).map_err(|_| {
                Error::backend("combine_rope_tail rope_dim exceeds Metal u32 limit")
            })?,
        )?;

        trace!(
            target: "inferno::metal",
            batch_count,
            token_count,
            head_count,
            rope_head_count,
            no_rope_dim,
            rope_dim,
            "running native Metal combine_rope_tail"
        );

        dispatch_1d(
            queue,
            &self.combine_rope_tail_pipeline,
            &[
                &no_rope_buffer,
                &rope_buffer,
                &output_buffer,
                &batch_count_buffer,
                &token_count_buffer,
                &head_count_buffer,
                &rope_head_count_buffer,
                &no_rope_dim_buffer,
                &rope_dim_buffer,
            ],
            output_len,
        )?;

        let values = read_f32_buffer(&output_buffer, output_len)?;

        Ok(MetalCombineRopeTailReport {
            values,
            batch_count,
            token_count,
            head_count,
            rope_head_count,
            no_rope_dim,
            rope_dim,
            thread_count: output_len,
        })
    }

    #[cfg(test)]
    pub(crate) fn stack_head_outputs(
        &self,
        device: &Device,
        queue: &CommandQueue,
        head_outputs: &[Vec<f32>],
        row_count: usize,
        head_dim: usize,
    ) -> Result<MetalStackHeadOutputsReport> {
        let head_count = head_outputs.len();
        validate_stack_head_outputs(head_count, row_count, head_dim)?;
        let input_len = row_count
            .checked_mul(head_dim)
            .ok_or_else(|| Error::backend("stack_head_outputs input length overflow"))?;
        let output_len = input_len
            .checked_mul(head_count)
            .ok_or_else(|| Error::backend("stack_head_outputs output length overflow"))?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let row_count_buffer = layout_u32_buffer(device, row_count, "row_count")?;
        let head_count_buffer = layout_u32_buffer(device, head_count, "head_count")?;
        let head_dim_buffer = layout_u32_buffer(device, head_dim, "head_dim")?;

        for (head_index, values) in head_outputs.iter().enumerate() {
            require_layout_input_len("stack_head_outputs head", values.len(), input_len)?;
            let input_buffer = f32_buffer(device, values)?;
            let head_index_buffer = layout_u32_buffer(device, head_index, "head_index")?;
            dispatch_1d(
                queue,
                &self.stack_head_output_pipeline,
                &[
                    &input_buffer,
                    &output_buffer,
                    &row_count_buffer,
                    &head_count_buffer,
                    &head_dim_buffer,
                    &head_index_buffer,
                ],
                input_len,
            )?;
        }

        let values = read_f32_buffer(&output_buffer, output_len)?;
        Ok(MetalStackHeadOutputsReport {
            values,
            row_count,
            head_count,
            head_dim,
            thread_count: output_len,
        })
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linearize_paged_cache(
        &self,
        device: &Device,
        queue: &CommandQueue,
        paged: &[f32],
        batch_count: usize,
        head_count: usize,
        cached_tokens: usize,
        capacity_tokens: usize,
        page_size: usize,
        head_dim: usize,
    ) -> Result<MetalLinearizePagedCacheReport> {
        validate_paged_cache_shape(
            paged.len(),
            batch_count,
            head_count,
            capacity_tokens,
            page_size,
            head_dim,
        )?;
        let output_len = paged_cache_output_len(batch_count, head_count, cached_tokens, head_dim)?;
        require_nonzero_output("linearize_paged_cache", output_len)?;
        let input = f32_buffer(device, paged)?;
        let output = empty_f32_buffer(device, output_len)?;
        let batch_count_buffer = layout_u32_buffer(device, batch_count, "batch_count")?;
        let head_count_buffer = layout_u32_buffer(device, head_count, "head_count")?;
        let cached_tokens_buffer = layout_u32_buffer(device, cached_tokens, "cached_tokens")?;
        let page_size_buffer = layout_u32_buffer(device, page_size, "page_size")?;
        let head_dim_buffer = layout_u32_buffer(device, head_dim, "head_dim")?;

        dispatch_1d(
            queue,
            &self.linearize_paged_cache_pipeline,
            &[
                &input,
                &output,
                &batch_count_buffer,
                &head_count_buffer,
                &cached_tokens_buffer,
                &page_size_buffer,
                &head_dim_buffer,
            ],
            output_len,
        )?;
        let values = read_f32_buffer(&output, output_len)?;
        Ok(MetalLinearizePagedCacheReport {
            values,
            batch_count,
            head_count,
            cached_tokens,
            page_size,
            head_dim,
            thread_count: output_len,
        })
    }

    // ------------------------------------------------------------------
    // Batched encode variants.
    //
    // These mirror the run_* entry points above but read their inputs from
    // device-resident buffers and encode into an open batched command buffer
    // instead of dispatching immediately. Inputs are validated by length only
    // (the values may not have been computed yet); outputs are fresh buffers.
    // See `BatchSlot` for the batching rules.
    // ------------------------------------------------------------------

    pub(crate) fn encode_select_last_token(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        hidden_states: &Buffer,
        input_len: usize,
        batch_count: usize,
        token_count: usize,
        hidden_size: usize,
    ) -> Result<Buffer> {
        let expected_input_len = batch_count
            .checked_mul(token_count)
            .and_then(|value| value.checked_mul(hidden_size))
            .ok_or_else(|| Error::backend("select_last_token input length overflow"))?;
        require_layout_input_len("select_last_token", input_len, expected_input_len)?;
        require_f32_capacity(hidden_states, input_len, "select_last_token input")?;
        let output_len = batch_count
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("select_last_token output length overflow"))?;
        require_nonzero_output("select_last_token", output_len)?;

        let output_buffer = empty_f32_buffer(device, output_len)?;
        let batch_count_buffer = layout_u32_buffer(device, batch_count, "batch_count")?;
        let token_count_buffer = layout_u32_buffer(device, token_count, "token_count")?;
        let hidden_size_buffer = layout_u32_buffer(device, hidden_size, "hidden_size")?;

        encode_1d(
            command_buffer,
            &self.select_last_token_pipeline,
            &[
                hidden_states,
                &output_buffer,
                &batch_count_buffer,
                &token_count_buffer,
                &hidden_size_buffer,
            ],
            output_len,
        )?;
        Ok(output_buffer)
    }

    pub(crate) fn encode_heads_to_attention_layout(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        input: &Buffer,
        input_len: usize,
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        head_dim: usize,
    ) -> Result<Buffer> {
        let output_len =
            attention_head_value_count(batch_count, token_count, head_count, head_dim)?;
        require_layout_input_len("heads_to_attention_layout", input_len, output_len)?;
        require_f32_capacity(input, input_len, "heads_to_attention_layout input")?;
        require_nonzero_output("heads_to_attention_layout", output_len)?;

        let output_buffer = empty_f32_buffer(device, output_len)?;
        let batch_count_buffer = layout_u32_buffer(device, batch_count, "batch_count")?;
        let token_count_buffer = layout_u32_buffer(device, token_count, "token_count")?;
        let head_count_buffer = layout_u32_buffer(device, head_count, "head_count")?;
        let head_dim_buffer = layout_u32_buffer(device, head_dim, "head_dim")?;

        encode_1d(
            command_buffer,
            &self.heads_to_attention_layout_pipeline,
            &[
                input,
                &output_buffer,
                &batch_count_buffer,
                &token_count_buffer,
                &head_count_buffer,
                &head_dim_buffer,
            ],
            output_len,
        )?;
        Ok(output_buffer)
    }

    pub(crate) fn encode_merge_attention_heads(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        input: &Buffer,
        input_len: usize,
        batch_count: usize,
        head_count: usize,
        token_count: usize,
        head_dim: usize,
    ) -> Result<Buffer> {
        let output_len =
            attention_head_value_count(batch_count, token_count, head_count, head_dim)?;
        require_layout_input_len("merge_attention_heads", input_len, output_len)?;
        require_f32_capacity(input, input_len, "merge_attention_heads input")?;
        require_nonzero_output("merge_attention_heads", output_len)?;

        let output_buffer = empty_f32_buffer(device, output_len)?;
        let batch_count_buffer = layout_u32_buffer(device, batch_count, "batch_count")?;
        let head_count_buffer = layout_u32_buffer(device, head_count, "head_count")?;
        let token_count_buffer = layout_u32_buffer(device, token_count, "token_count")?;
        let head_dim_buffer = layout_u32_buffer(device, head_dim, "head_dim")?;

        encode_1d(
            command_buffer,
            &self.merge_attention_heads_pipeline,
            &[
                input,
                &output_buffer,
                &batch_count_buffer,
                &head_count_buffer,
                &token_count_buffer,
                &head_dim_buffer,
            ],
            output_len,
        )?;
        Ok(output_buffer)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_split_rope_tail(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        input: &Buffer,
        input_len: usize,
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        no_rope_dim: usize,
        rope_dim: usize,
    ) -> Result<(Buffer, Buffer)> {
        let no_rope_len =
            attention_head_value_count(batch_count, token_count, head_count, no_rope_dim)?;
        let rope_len = attention_head_value_count(batch_count, token_count, head_count, rope_dim)?;
        let thread_count = no_rope_len
            .checked_add(rope_len)
            .ok_or_else(|| Error::backend("split_rope_tail thread count overflow"))?;
        require_layout_input_len("split_rope_tail", input_len, thread_count)?;
        require_f32_capacity(input, input_len, "split_rope_tail input")?;
        require_nonzero_output("split_rope_tail", thread_count)?;

        let no_rope_buffer = empty_f32_buffer(device, no_rope_len)?;
        let rope_buffer = empty_f32_buffer(device, rope_len)?;
        let batch_count_buffer = layout_u32_buffer(device, batch_count, "batch_count")?;
        let token_count_buffer = layout_u32_buffer(device, token_count, "token_count")?;
        let head_count_buffer = layout_u32_buffer(device, head_count, "head_count")?;
        let no_rope_dim_buffer = layout_u32_buffer(device, no_rope_dim, "no_rope_dim")?;
        let rope_dim_buffer = layout_u32_buffer(device, rope_dim, "rope_dim")?;

        encode_1d(
            command_buffer,
            &self.split_rope_tail_pipeline,
            &[
                input,
                &no_rope_buffer,
                &rope_buffer,
                &batch_count_buffer,
                &token_count_buffer,
                &head_count_buffer,
                &no_rope_dim_buffer,
                &rope_dim_buffer,
            ],
            thread_count,
        )?;
        Ok((no_rope_buffer, rope_buffer))
    }

    pub(crate) fn encode_split_kv_mqa(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        input: &Buffer,
        input_len: usize,
        batch_count: usize,
        token_count: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
    ) -> Result<(Buffer, Buffer)> {
        let latent_len = batch_count
            .checked_mul(token_count)
            .and_then(|value| value.checked_mul(kv_lora_rank))
            .ok_or_else(|| Error::backend("split_kv_mqa latent length overflow"))?;
        let rope_len = batch_count
            .checked_mul(token_count)
            .and_then(|value| value.checked_mul(rope_dim))
            .ok_or_else(|| Error::backend("split_kv_mqa rope length overflow"))?;
        let thread_count = latent_len
            .checked_add(rope_len)
            .ok_or_else(|| Error::backend("split_kv_mqa thread count overflow"))?;
        require_layout_input_len("split_kv_mqa", input_len, thread_count)?;
        require_f32_capacity(input, input_len, "split_kv_mqa input")?;
        require_nonzero_output("split_kv_mqa", thread_count)?;

        let latent_buffer = empty_f32_buffer(device, latent_len)?;
        let rope_buffer = empty_f32_buffer(device, rope_len)?;
        let batch_count_buffer = layout_u32_buffer(device, batch_count, "batch_count")?;
        let token_count_buffer = layout_u32_buffer(device, token_count, "token_count")?;
        let kv_lora_rank_buffer = layout_u32_buffer(device, kv_lora_rank, "kv_lora_rank")?;
        let rope_dim_buffer = layout_u32_buffer(device, rope_dim, "rope_dim")?;

        encode_1d(
            command_buffer,
            &self.split_kv_mqa_pipeline,
            &[
                input,
                &latent_buffer,
                &rope_buffer,
                &batch_count_buffer,
                &token_count_buffer,
                &kv_lora_rank_buffer,
                &rope_dim_buffer,
            ],
            thread_count,
        )?;
        Ok((latent_buffer, rope_buffer))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_combine_rope_tail(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        no_rope: &Buffer,
        no_rope_len: usize,
        rope: &Buffer,
        rope_len: usize,
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        rope_head_count: usize,
        no_rope_dim: usize,
        rope_dim: usize,
    ) -> Result<Buffer> {
        if rope_head_count != 1 && rope_head_count != head_count {
            return Err(Error::backend(format!(
                "combine_rope_tail rope_head_count must be 1 or {head_count}, got {rope_head_count}"
            )));
        }
        let expected_no_rope_len =
            attention_head_value_count(batch_count, token_count, head_count, no_rope_dim)?;
        require_layout_input_len(
            "combine_rope_tail no_rope",
            no_rope_len,
            expected_no_rope_len,
        )?;
        let expected_rope_len =
            attention_head_value_count(batch_count, token_count, rope_head_count, rope_dim)?;
        require_layout_input_len("combine_rope_tail rope", rope_len, expected_rope_len)?;
        require_f32_capacity(no_rope, no_rope_len, "combine_rope_tail no_rope input")?;
        require_f32_capacity(rope, rope_len, "combine_rope_tail rope input")?;

        let total_dim = no_rope_dim
            .checked_add(rope_dim)
            .ok_or_else(|| Error::backend("combine_rope_tail total dim overflow"))?;
        let output_len =
            attention_head_value_count(batch_count, token_count, head_count, total_dim)?;
        require_nonzero_output("combine_rope_tail", output_len)?;

        let output_buffer = empty_f32_buffer(device, output_len)?;
        let batch_count_buffer = layout_u32_buffer(device, batch_count, "batch_count")?;
        let token_count_buffer = layout_u32_buffer(device, token_count, "token_count")?;
        let head_count_buffer = layout_u32_buffer(device, head_count, "head_count")?;
        let rope_head_count_buffer = layout_u32_buffer(device, rope_head_count, "rope_head_count")?;
        let no_rope_dim_buffer = layout_u32_buffer(device, no_rope_dim, "no_rope_dim")?;
        let rope_dim_buffer = layout_u32_buffer(device, rope_dim, "rope_dim")?;

        encode_1d(
            command_buffer,
            &self.combine_rope_tail_pipeline,
            &[
                no_rope,
                rope,
                &output_buffer,
                &batch_count_buffer,
                &token_count_buffer,
                &head_count_buffer,
                &rope_head_count_buffer,
                &no_rope_dim_buffer,
                &rope_dim_buffer,
            ],
            output_len,
        )?;
        Ok(output_buffer)
    }

    pub(crate) fn encode_stack_head_outputs(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        head_outputs: &[&Buffer],
        row_count: usize,
        head_dim: usize,
    ) -> Result<Buffer> {
        let head_count = head_outputs.len();
        validate_stack_head_outputs(head_count, row_count, head_dim)?;
        let input_len = row_count
            .checked_mul(head_dim)
            .ok_or_else(|| Error::backend("stack_head_outputs input length overflow"))?;
        let output_len = input_len
            .checked_mul(head_count)
            .ok_or_else(|| Error::backend("stack_head_outputs output length overflow"))?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let row_count_buffer = layout_u32_buffer(device, row_count, "row_count")?;
        let head_count_buffer = layout_u32_buffer(device, head_count, "head_count")?;
        let head_dim_buffer = layout_u32_buffer(device, head_dim, "head_dim")?;

        for (head_index, head_output) in head_outputs.iter().enumerate() {
            require_f32_capacity(head_output, input_len, "stack_head_outputs head input")?;
            let head_index_buffer = layout_u32_buffer(device, head_index, "head_index")?;
            encode_1d(
                command_buffer,
                &self.stack_head_output_pipeline,
                &[
                    *head_output,
                    &output_buffer,
                    &row_count_buffer,
                    &head_count_buffer,
                    &head_dim_buffer,
                    &head_index_buffer,
                ],
                input_len,
            )?;
        }

        Ok(output_buffer)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_linearize_paged_cache(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        paged: &Buffer,
        paged_len: usize,
        batch_count: usize,
        head_count: usize,
        cached_tokens: usize,
        capacity_tokens: usize,
        page_size: usize,
        head_dim: usize,
    ) -> Result<Buffer> {
        validate_paged_cache_shape(
            paged_len,
            batch_count,
            head_count,
            capacity_tokens,
            page_size,
            head_dim,
        )?;
        require_f32_capacity(paged, paged_len, "linearize_paged_cache input")?;
        let output_len = paged_cache_output_len(batch_count, head_count, cached_tokens, head_dim)?;
        require_nonzero_output("linearize_paged_cache", output_len)?;
        let output = empty_f32_buffer(device, output_len)?;
        let batch_count_buffer = layout_u32_buffer(device, batch_count, "batch_count")?;
        let head_count_buffer = layout_u32_buffer(device, head_count, "head_count")?;
        let cached_tokens_buffer = layout_u32_buffer(device, cached_tokens, "cached_tokens")?;
        let page_size_buffer = layout_u32_buffer(device, page_size, "page_size")?;
        let head_dim_buffer = layout_u32_buffer(device, head_dim, "head_dim")?;

        encode_1d(
            command_buffer,
            &self.linearize_paged_cache_pipeline,
            &[
                paged,
                &output,
                &batch_count_buffer,
                &head_count_buffer,
                &cached_tokens_buffer,
                &page_size_buffer,
                &head_dim_buffer,
            ],
            output_len,
        )?;
        Ok(output)
    }
}

fn layout_u32_buffer(device: &Device, value: usize, label: &str) -> Result<Buffer> {
    let value = u32::try_from(value)
        .map_err(|_| Error::backend(format!("layout {label} exceeds Metal u32 limit")))?;
    u32_scalar_buffer(device, value)
}

fn require_layout_input_len(name: &str, input_len: usize, expected: usize) -> Result<()> {
    if input_len != expected {
        return Err(Error::backend(format!(
            "{name} input length mismatch: expected {expected} values, got {input_len}"
        )));
    }
    Ok(())
}

fn require_nonzero_output(name: &str, output_len: usize) -> Result<()> {
    if output_len == 0 {
        return Err(Error::backend(format!(
            "{name} requires a non-empty output (a shape dimension is zero)"
        )));
    }
    Ok(())
}

fn validate_stack_head_outputs(head_count: usize, row_count: usize, head_dim: usize) -> Result<()> {
    if head_count == 0 || row_count == 0 || head_dim == 0 {
        return Err(Error::backend(format!(
            "stack_head_outputs dimensions must be positive: rows={row_count}, heads={head_count}, head_dim={head_dim}"
        )));
    }
    Ok(())
}

fn validate_paged_cache_shape(
    len: usize,
    batch_count: usize,
    head_count: usize,
    capacity_tokens: usize,
    page_size: usize,
    head_dim: usize,
) -> Result<()> {
    if batch_count == 0
        || head_count == 0
        || capacity_tokens == 0
        || page_size == 0
        || head_dim == 0
    {
        return Err(Error::backend(format!(
            "linearize_paged_cache dimensions must be positive: batch={batch_count}, heads={head_count}, capacity={capacity_tokens}, page_size={page_size}, head_dim={head_dim}"
        )));
    }
    if capacity_tokens % page_size != 0 {
        return Err(Error::backend(format!(
            "linearize_paged_cache capacity_tokens {capacity_tokens} must be divisible by page_size {page_size}"
        )));
    }
    let expected_len = paged_cache_output_len(batch_count, head_count, capacity_tokens, head_dim)?;
    require_layout_input_len("linearize_paged_cache", len, expected_len)
}

fn paged_cache_output_len(
    batch_count: usize,
    head_count: usize,
    token_count: usize,
    head_dim: usize,
) -> Result<usize> {
    batch_count
        .checked_mul(head_count)
        .and_then(|value| value.checked_mul(token_count))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| Error::backend("linearize_paged_cache output length overflow"))
}

fn attention_head_value_count(
    batch_count: usize,
    token_count: usize,
    head_count: usize,
    head_dim: usize,
) -> Result<usize> {
    batch_count
        .checked_mul(token_count)
        .and_then(|value| value.checked_mul(head_count))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| Error::backend("attention layout output length overflow"))
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use crate::metal::Metal;

    #[test]
    fn select_last_token_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch_count = 2;
        let token_count = 3;
        let hidden_size = 4;
        let hidden_states = (0..batch_count * token_count * hidden_size)
            .map(|index| index as f32)
            .collect::<Vec<_>>();

        let report = metal
            .select_last_token_f32_report(&hidden_states, batch_count, token_count, hidden_size)
            .unwrap();
        let expected = cpu_select_last_token(&hidden_states, batch_count, token_count, hidden_size);

        assert_eq!(report.batch_count, batch_count);
        assert_eq!(report.token_count, token_count);
        assert_eq!(report.hidden_size, hidden_size);
        assert_eq!(report.thread_count, batch_count * hidden_size);
        assert_eq!(report.values, expected);
    }

    #[test]
    fn heads_to_attention_layout_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch_count = 2;
        let token_count = 3;
        let head_count = 2;
        let head_dim = 4;
        let input = (0..batch_count * token_count * head_count * head_dim)
            .map(|index| index as f32)
            .collect::<Vec<_>>();

        let report = metal
            .heads_to_attention_layout_f32_report(
                &input,
                batch_count,
                token_count,
                head_count,
                head_dim,
            )
            .unwrap();
        let expected =
            cpu_heads_to_attention_layout(&input, batch_count, token_count, head_count, head_dim);

        assert_eq!(report.batch_count, batch_count);
        assert_eq!(report.token_count, token_count);
        assert_eq!(report.head_count, head_count);
        assert_eq!(report.head_dim, head_dim);
        assert_eq!(
            report.thread_count,
            batch_count * token_count * head_count * head_dim
        );
        assert_eq!(report.values, expected);
    }

    #[test]
    fn merge_attention_heads_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch_count = 2;
        let head_count = 2;
        let token_count = 3;
        let head_dim = 4;
        let input = (0..batch_count * head_count * token_count * head_dim)
            .map(|index| index as f32)
            .collect::<Vec<_>>();

        let report = metal
            .merge_attention_heads_f32_report(
                &input,
                batch_count,
                head_count,
                token_count,
                head_dim,
            )
            .unwrap();
        let expected =
            cpu_merge_attention_heads(&input, batch_count, head_count, token_count, head_dim);

        assert_eq!(report.batch_count, batch_count);
        assert_eq!(report.token_count, token_count);
        assert_eq!(report.head_count, head_count);
        assert_eq!(report.head_dim, head_dim);
        assert_eq!(
            report.thread_count,
            batch_count * token_count * head_count * head_dim
        );
        assert_eq!(report.values, expected);
    }

    #[test]
    fn split_rope_tail_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch_count = 1;
        let token_count = 3;
        let head_count = 2;
        let no_rope_dim = 3;
        let rope_dim = 2;
        let total_dim = no_rope_dim + rope_dim;
        let input = (0..batch_count * token_count * head_count * total_dim)
            .map(|index| index as f32)
            .collect::<Vec<_>>();

        let report = metal
            .split_rope_tail_f32_report(
                &input,
                batch_count,
                token_count,
                head_count,
                no_rope_dim,
                rope_dim,
            )
            .unwrap();
        let (expected_no_rope, expected_rope) = cpu_split_rope_tail(
            &input,
            batch_count,
            token_count,
            head_count,
            no_rope_dim,
            rope_dim,
        );

        assert_eq!(report.batch_count, batch_count);
        assert_eq!(report.token_count, token_count);
        assert_eq!(report.head_count, head_count);
        assert_eq!(report.no_rope_dim, no_rope_dim);
        assert_eq!(report.rope_dim, rope_dim);
        assert_eq!(
            report.thread_count,
            batch_count * token_count * head_count * total_dim
        );
        assert_eq!(report.no_rope_values, expected_no_rope);
        assert_eq!(report.rope_values, expected_rope);
    }

    #[test]
    fn split_kv_mqa_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch_count = 1;
        let token_count = 3;
        let kv_lora_rank = 4;
        let rope_dim = 2;
        let total_dim = kv_lora_rank + rope_dim;
        let input = (0..batch_count * token_count * total_dim)
            .map(|index| index as f32)
            .collect::<Vec<_>>();

        let report = metal
            .split_kv_mqa_f32_report(&input, batch_count, token_count, kv_lora_rank, rope_dim)
            .unwrap();
        let (expected_latent, expected_rope) =
            cpu_split_kv_mqa(&input, batch_count, token_count, kv_lora_rank, rope_dim);

        assert_eq!(report.batch_count, batch_count);
        assert_eq!(report.token_count, token_count);
        assert_eq!(report.kv_lora_rank, kv_lora_rank);
        assert_eq!(report.rope_dim, rope_dim);
        assert_eq!(report.thread_count, batch_count * token_count * total_dim);
        assert_eq!(report.kv_latent_values, expected_latent);
        assert_eq!(report.k_rope_values, expected_rope);
    }

    #[test]
    fn combine_rope_tail_broadcasts_shared_rope_head() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch_count = 1;
        let token_count = 2;
        let head_count = 3;
        let rope_head_count = 1;
        let no_rope_dim = 2;
        let rope_dim = 2;
        let no_rope = (0..batch_count * token_count * head_count * no_rope_dim)
            .map(|index| index as f32)
            .collect::<Vec<_>>();
        let rope = (0..batch_count * token_count * rope_head_count * rope_dim)
            .map(|index| 100.0 + index as f32)
            .collect::<Vec<_>>();

        let report = metal
            .combine_rope_tail_f32_report(
                &no_rope,
                &rope,
                batch_count,
                token_count,
                head_count,
                rope_head_count,
                no_rope_dim,
                rope_dim,
            )
            .unwrap();
        let expected = cpu_combine_rope_tail(
            &no_rope,
            &rope,
            batch_count,
            token_count,
            head_count,
            rope_head_count,
            no_rope_dim,
            rope_dim,
        );

        assert_eq!(report.batch_count, batch_count);
        assert_eq!(report.token_count, token_count);
        assert_eq!(report.head_count, head_count);
        assert_eq!(report.rope_head_count, rope_head_count);
        assert_eq!(report.no_rope_dim, no_rope_dim);
        assert_eq!(report.rope_dim, rope_dim);
        assert_eq!(report.values, expected);
    }

    #[test]
    fn stack_head_outputs_matches_token_major_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let row_count = 3;
        let head_count = 2;
        let head_dim = 4;
        let head_outputs = (0..head_count)
            .map(|head| {
                (0..row_count * head_dim)
                    .map(|index| (head * 100 + index) as f32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let report = metal
            .stack_head_outputs_f32_report(&head_outputs, row_count, head_dim)
            .unwrap();
        let expected = cpu_stack_head_outputs(&head_outputs, row_count, head_dim);

        assert_eq!(report.row_count, row_count);
        assert_eq!(report.head_count, head_count);
        assert_eq!(report.head_dim, head_dim);
        assert_eq!(report.thread_count, row_count * head_count * head_dim);
        assert_eq!(report.values, expected);
    }

    #[test]
    fn linearize_paged_cache_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch_count = 2;
        let head_count = 2;
        let cached_tokens = 3;
        let capacity_tokens = 4;
        let page_size = 2;
        let head_dim = 2;
        let paged = (0..batch_count * head_count * capacity_tokens * head_dim)
            .map(|index| index as f32)
            .collect::<Vec<_>>();

        let report = metal
            .linearize_paged_cache_f32_report(
                &paged,
                batch_count,
                head_count,
                cached_tokens,
                capacity_tokens,
                page_size,
                head_dim,
            )
            .unwrap();
        let expected = cpu_linearize_paged_cache(
            &paged,
            batch_count,
            head_count,
            cached_tokens,
            page_size,
            head_dim,
        );

        assert_eq!(report.batch_count, batch_count);
        assert_eq!(report.head_count, head_count);
        assert_eq!(report.cached_tokens, cached_tokens);
        assert_eq!(report.page_size, page_size);
        assert_eq!(report.head_dim, head_dim);
        assert_eq!(
            report.thread_count,
            batch_count * head_count * cached_tokens * head_dim
        );
        assert_eq!(report.values, expected);
    }

    fn native_metal_or_skip() -> Option<Metal> {
        Metal::new().ok()
    }

    fn cpu_select_last_token(
        hidden_states: &[f32],
        batch_count: usize,
        token_count: usize,
        hidden_size: usize,
    ) -> Vec<f32> {
        let mut output = Vec::with_capacity(batch_count * hidden_size);
        for batch in 0..batch_count {
            let start = ((batch * token_count) + (token_count - 1)) * hidden_size;
            output.extend_from_slice(&hidden_states[start..start + hidden_size]);
        }
        output
    }

    fn cpu_heads_to_attention_layout(
        input: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; batch_count * head_count * token_count * head_dim];
        for batch in 0..batch_count {
            for head in 0..head_count {
                for token in 0..token_count {
                    for dim in 0..head_dim {
                        let source =
                            (((batch * token_count + token) * head_count + head) * head_dim) + dim;
                        let target =
                            (((batch * head_count + head) * token_count + token) * head_dim) + dim;
                        output[target] = input[source];
                    }
                }
            }
        }
        output
    }

    fn cpu_merge_attention_heads(
        input: &[f32],
        batch_count: usize,
        head_count: usize,
        token_count: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; batch_count * token_count * head_count * head_dim];
        for batch in 0..batch_count {
            for token in 0..token_count {
                for head in 0..head_count {
                    for dim in 0..head_dim {
                        let source =
                            (((batch * head_count + head) * token_count + token) * head_dim) + dim;
                        let target =
                            (((batch * token_count + token) * head_count + head) * head_dim) + dim;
                        output[target] = input[source];
                    }
                }
            }
        }
        output
    }

    fn cpu_split_rope_tail(
        input: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        no_rope_dim: usize,
        rope_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let total_dim = no_rope_dim + rope_dim;
        let mut no_rope = vec![0.0; batch_count * token_count * head_count * no_rope_dim];
        let mut rope = vec![0.0; batch_count * token_count * head_count * rope_dim];
        for batch in 0..batch_count {
            for token in 0..token_count {
                for head in 0..head_count {
                    for dim in 0..no_rope_dim {
                        let source =
                            (((batch * token_count + token) * head_count + head) * total_dim) + dim;
                        let target = (((batch * token_count + token) * head_count + head)
                            * no_rope_dim)
                            + dim;
                        no_rope[target] = input[source];
                    }
                    for dim in 0..rope_dim {
                        let source = (((batch * token_count + token) * head_count + head)
                            * total_dim)
                            + no_rope_dim
                            + dim;
                        let target =
                            (((batch * token_count + token) * head_count + head) * rope_dim) + dim;
                        rope[target] = input[source];
                    }
                }
            }
        }
        (no_rope, rope)
    }

    fn cpu_split_kv_mqa(
        input: &[f32],
        batch_count: usize,
        token_count: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let total_dim = kv_lora_rank + rope_dim;
        let mut kv_latent = vec![0.0; batch_count * token_count * kv_lora_rank];
        let mut k_rope = vec![0.0; batch_count * token_count * rope_dim];
        for batch in 0..batch_count {
            for token in 0..token_count {
                for dim in 0..kv_lora_rank {
                    let source = ((batch * token_count + token) * total_dim) + dim;
                    let target = ((batch * token_count + token) * kv_lora_rank) + dim;
                    kv_latent[target] = input[source];
                }
                for dim in 0..rope_dim {
                    let source = ((batch * token_count + token) * total_dim) + kv_lora_rank + dim;
                    let target = ((batch * token_count + token) * rope_dim) + dim;
                    k_rope[target] = input[source];
                }
            }
        }
        (kv_latent, k_rope)
    }

    fn cpu_combine_rope_tail(
        no_rope: &[f32],
        rope: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        rope_head_count: usize,
        no_rope_dim: usize,
        rope_dim: usize,
    ) -> Vec<f32> {
        let total_dim = no_rope_dim + rope_dim;
        let mut output = vec![0.0; batch_count * token_count * head_count * total_dim];
        for batch in 0..batch_count {
            for token in 0..token_count {
                for head in 0..head_count {
                    for dim in 0..no_rope_dim {
                        let source = (((batch * token_count + token) * head_count + head)
                            * no_rope_dim)
                            + dim;
                        let target =
                            (((batch * token_count + token) * head_count + head) * total_dim) + dim;
                        output[target] = no_rope[source];
                    }
                    for dim in 0..rope_dim {
                        let rope_head = if rope_head_count == 1 { 0 } else { head };
                        let source = (((batch * token_count + token) * rope_head_count
                            + rope_head)
                            * rope_dim)
                            + dim;
                        let target = (((batch * token_count + token) * head_count + head)
                            * total_dim)
                            + no_rope_dim
                            + dim;
                        output[target] = rope[source];
                    }
                }
            }
        }
        output
    }

    fn cpu_stack_head_outputs(
        head_outputs: &[Vec<f32>],
        row_count: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let head_count = head_outputs.len();
        let mut output = vec![0.0; row_count * head_count * head_dim];
        for (head, values) in head_outputs.iter().enumerate() {
            for row in 0..row_count {
                for dim in 0..head_dim {
                    let source = row * head_dim + dim;
                    let target = (row * head_count + head) * head_dim + dim;
                    output[target] = values[source];
                }
            }
        }
        output
    }

    fn cpu_linearize_paged_cache(
        paged: &[f32],
        batch_count: usize,
        head_count: usize,
        cached_tokens: usize,
        page_size: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; batch_count * head_count * cached_tokens * head_dim];
        for batch in 0..batch_count {
            for head in 0..head_count {
                for token in 0..cached_tokens {
                    let page = token / page_size;
                    let page_offset = token % page_size;
                    for dim in 0..head_dim {
                        let source = ((((page * batch_count + batch) * head_count + head)
                            * page_size
                            + page_offset)
                            * head_dim)
                            + dim;
                        let target = (((batch * head_count + head) * cached_tokens + token)
                            * head_dim)
                            + dim;
                        output[target] = paged[source];
                    }
                }
            }
        }
        output
    }
}
