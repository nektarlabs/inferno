use ::metal::{CommandQueue, ComputePipelineState, Device};
use common::{Error, Result};
use tracing::trace;

use super::{
    buffers::{empty_f32_buffer, f32_buffer, read_f32_buffer, u32_scalar_buffer},
    command::dispatch_1d,
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

pub(crate) struct MetalLayout {
    select_last_token_pipeline: ComputePipelineState,
    heads_to_attention_layout_pipeline: ComputePipelineState,
    merge_attention_heads_pipeline: ComputePipelineState,
    split_rope_tail_pipeline: ComputePipelineState,
    split_kv_mqa_pipeline: ComputePipelineState,
    combine_rope_tail_pipeline: ComputePipelineState,
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
}
