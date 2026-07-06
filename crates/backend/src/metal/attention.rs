use std::sync::Mutex;

use ::metal::{Buffer, CommandBufferRef, CommandQueue, ComputePipelineState, Device};
use common::{validate_exact_shape, Error, PagedKvView, Result};
use tracing::trace;

use super::{
    buffers::{
        empty_f32_buffer, empty_u32_buffer, f32_buffer, read_f32_buffer, require_f32_capacity,
        u32_scalar_buffer, write_f32_buffer, write_f32_buffer_at, write_u32_buffer,
    },
    command::{dispatch_1d, dispatch_1d_many, encode_1d, Dispatch1d},
    library::MetalLibrary,
    pipeline::compute_pipeline,
    validation::{
        validate_attention_causal_softmax_f32, validate_attention_scores_f32,
        validate_attention_values_f32,
    },
};

const ATTENTION_SCORES_KERNEL: &str = "attention_scores_f32_kernel";
const ATTENTION_VALUES_KERNEL: &str = "attention_values_f32_kernel";
const ATTENTION_CAUSAL_SOFTMAX_KERNEL: &str = "attention_causal_softmax_f32_kernel";
const PAGED_DECODE_ATTENTION_SCORES_KERNEL: &str = "paged_decode_attention_scores_f32_kernel";
const PAGED_DECODE_ATTENTION_VALUES_KERNEL: &str = "paged_decode_attention_values_f32_kernel";

pub(crate) struct MetalAttentionScores {
    pipeline: ComputePipelineState,
}

pub(crate) struct MetalAttentionValues {
    pipeline: ComputePipelineState,
}

pub(crate) struct MetalAttentionCausalSoftmax {
    pipeline: ComputePipelineState,
}

pub(crate) struct MetalDecodeAttention {
    scores_pipeline: ComputePipelineState,
    paged_scores_pipeline: ComputePipelineState,
    softmax_pipeline: ComputePipelineState,
    values_pipeline: ComputePipelineState,
    paged_values_pipeline: ComputePipelineState,
    workspace: Mutex<DecodeAttentionWorkspace>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalAttentionScoresReport {
    pub values: Vec<f32>,
    pub batch_count: usize,
    pub head_count: usize,
    pub query_tokens: usize,
    pub key_tokens: usize,
    pub head_dim: usize,
    pub thread_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalAttentionValuesReport {
    pub values: Vec<f32>,
    pub batch_count: usize,
    pub head_count: usize,
    pub query_tokens: usize,
    pub key_tokens: usize,
    pub value_dim: usize,
    pub thread_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalAttentionCausalSoftmaxReport {
    pub values: Vec<f32>,
    pub batch_count: usize,
    pub head_count: usize,
    pub query_tokens: usize,
    pub key_tokens: usize,
    pub past_tokens: usize,
    pub thread_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalDecodeAttentionReport {
    pub values: Vec<f32>,
    pub batch_count: usize,
    pub head_count: usize,
    pub query_tokens: usize,
    pub key_tokens: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    pub past_tokens: usize,
    pub scores_len: usize,
    pub probs_len: usize,
    pub output_len: usize,
    pub reused_scores_buffer: bool,
    pub reused_probs_buffer: bool,
    pub reused_output_buffer: bool,
    pub reused_q_buffer: bool,
    pub reused_k_buffer: bool,
    pub reused_v_buffer: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalPagedDecodeAttentionReport {
    pub values: Vec<f32>,
    pub batch_count: usize,
    pub head_count: usize,
    pub past_tokens: usize,
    pub key_tokens: usize,
    pub page_count: usize,
    pub page_size: usize,
    pub head_dim: usize,
    pub value_dim: usize,
    pub scores_len: usize,
    pub probs_len: usize,
    pub output_len: usize,
    pub reused_scores_buffer: bool,
    pub reused_probs_buffer: bool,
    pub reused_output_buffer: bool,
    pub reused_q_buffer: bool,
    pub transient_kv_buffers: bool,
    pub uploaded_page_count: usize,
    pub uploaded_page_tokens: usize,
}

#[derive(Debug, Default)]
struct DecodeAttentionWorkspace {
    scores: Option<WorkspaceBuffer>,
    probs: Option<WorkspaceBuffer>,
    output: Option<WorkspaceBuffer>,
    q: Option<WorkspaceBuffer>,
    k: Option<WorkspaceBuffer>,
    v: Option<WorkspaceBuffer>,
    batch_count: Option<WorkspaceBuffer>,
    head_count: Option<WorkspaceBuffer>,
    query_tokens: Option<WorkspaceBuffer>,
    key_tokens: Option<WorkspaceBuffer>,
    page_size: Option<WorkspaceBuffer>,
    head_dim: Option<WorkspaceBuffer>,
    value_dim: Option<WorkspaceBuffer>,
    past_tokens: Option<WorkspaceBuffer>,
}

#[derive(Debug)]
struct WorkspaceBuffer {
    buffer: Buffer,
    len: usize,
}

impl MetalAttentionScores {
    pub(crate) fn new(device: &Device, library: &MetalLibrary) -> Result<Self> {
        Ok(Self {
            pipeline: compute_pipeline(device, library, ATTENTION_SCORES_KERNEL)?,
        })
    }

    pub(crate) fn run(
        &self,
        device: &Device,
        queue: &CommandQueue,
        q: &[f32],
        k: &[f32],
        batch_count: usize,
        head_count: usize,
        query_tokens: usize,
        key_tokens: usize,
        head_dim: usize,
    ) -> Result<MetalAttentionScoresReport> {
        validate_attention_scores_f32(
            q,
            k,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            head_dim,
        )?;

        let batch_count_u32 = u32::try_from(batch_count)
            .map_err(|_| Error::backend("attention batch_count exceeds Metal u32 limit"))?;
        let head_count_u32 = u32::try_from(head_count)
            .map_err(|_| Error::backend("attention head_count exceeds Metal u32 limit"))?;
        let query_tokens_u32 = u32::try_from(query_tokens)
            .map_err(|_| Error::backend("attention query_tokens exceeds Metal u32 limit"))?;
        let key_tokens_u32 = u32::try_from(key_tokens)
            .map_err(|_| Error::backend("attention key_tokens exceeds Metal u32 limit"))?;
        let head_dim_u32 = u32::try_from(head_dim)
            .map_err(|_| Error::backend("attention head_dim exceeds Metal u32 limit"))?;
        let output_len = batch_count
            .checked_mul(head_count)
            .and_then(|value| value.checked_mul(query_tokens))
            .and_then(|value| value.checked_mul(key_tokens))
            .ok_or_else(|| Error::backend("attention score output length overflow"))?;

        let q_buffer = f32_buffer(device, q)?;
        let k_buffer = f32_buffer(device, k)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let batch_count_buffer = u32_scalar_buffer(device, batch_count_u32)?;
        let head_count_buffer = u32_scalar_buffer(device, head_count_u32)?;
        let query_tokens_buffer = u32_scalar_buffer(device, query_tokens_u32)?;
        let key_tokens_buffer = u32_scalar_buffer(device, key_tokens_u32)?;
        let head_dim_buffer = u32_scalar_buffer(device, head_dim_u32)?;

        trace!(
            target: "inferno::metal",
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            head_dim,
            "running native Metal attention scores"
        );

        dispatch_1d(
            queue,
            &self.pipeline,
            &[
                &q_buffer,
                &k_buffer,
                &output_buffer,
                &batch_count_buffer,
                &head_count_buffer,
                &query_tokens_buffer,
                &key_tokens_buffer,
                &head_dim_buffer,
            ],
            output_len,
        )?;

        let values = read_f32_buffer(&output_buffer, output_len)?;

        Ok(MetalAttentionScoresReport {
            values,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            head_dim,
            thread_count: output_len,
        })
    }
}

impl MetalAttentionValues {
    pub(crate) fn new(device: &Device, library: &MetalLibrary) -> Result<Self> {
        Ok(Self {
            pipeline: compute_pipeline(device, library, ATTENTION_VALUES_KERNEL)?,
        })
    }

    pub(crate) fn run(
        &self,
        device: &Device,
        queue: &CommandQueue,
        probs: &[f32],
        values: &[f32],
        batch_count: usize,
        head_count: usize,
        query_tokens: usize,
        key_tokens: usize,
        value_dim: usize,
    ) -> Result<MetalAttentionValuesReport> {
        validate_attention_values_f32(
            probs,
            values,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            value_dim,
        )?;

        let batch_count_u32 = u32::try_from(batch_count)
            .map_err(|_| Error::backend("attention values batch_count exceeds Metal u32 limit"))?;
        let head_count_u32 = u32::try_from(head_count)
            .map_err(|_| Error::backend("attention values head_count exceeds Metal u32 limit"))?;
        let query_tokens_u32 = u32::try_from(query_tokens)
            .map_err(|_| Error::backend("attention values query_tokens exceeds Metal u32 limit"))?;
        let key_tokens_u32 = u32::try_from(key_tokens)
            .map_err(|_| Error::backend("attention values key_tokens exceeds Metal u32 limit"))?;
        let value_dim_u32 = u32::try_from(value_dim)
            .map_err(|_| Error::backend("attention values value_dim exceeds Metal u32 limit"))?;
        let output_len = batch_count
            .checked_mul(head_count)
            .and_then(|value| value.checked_mul(query_tokens))
            .and_then(|value| value.checked_mul(value_dim))
            .ok_or_else(|| Error::backend("attention values output length overflow"))?;

        let probs_buffer = f32_buffer(device, probs)?;
        let values_buffer = f32_buffer(device, values)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let batch_count_buffer = u32_scalar_buffer(device, batch_count_u32)?;
        let head_count_buffer = u32_scalar_buffer(device, head_count_u32)?;
        let query_tokens_buffer = u32_scalar_buffer(device, query_tokens_u32)?;
        let key_tokens_buffer = u32_scalar_buffer(device, key_tokens_u32)?;
        let value_dim_buffer = u32_scalar_buffer(device, value_dim_u32)?;

        trace!(
            target: "inferno::metal",
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            value_dim,
            "running native Metal attention value aggregation"
        );

        dispatch_1d(
            queue,
            &self.pipeline,
            &[
                &probs_buffer,
                &values_buffer,
                &output_buffer,
                &batch_count_buffer,
                &head_count_buffer,
                &query_tokens_buffer,
                &key_tokens_buffer,
                &value_dim_buffer,
            ],
            output_len,
        )?;

        let output = read_f32_buffer(&output_buffer, output_len)?;

        Ok(MetalAttentionValuesReport {
            values: output,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            value_dim,
            thread_count: output_len,
        })
    }
}

impl MetalAttentionCausalSoftmax {
    pub(crate) fn new(device: &Device, library: &MetalLibrary) -> Result<Self> {
        Ok(Self {
            pipeline: compute_pipeline(device, library, ATTENTION_CAUSAL_SOFTMAX_KERNEL)?,
        })
    }

    pub(crate) fn run(
        &self,
        device: &Device,
        queue: &CommandQueue,
        scores: &[f32],
        batch_count: usize,
        head_count: usize,
        query_tokens: usize,
        key_tokens: usize,
        past_tokens: usize,
    ) -> Result<MetalAttentionCausalSoftmaxReport> {
        validate_attention_causal_softmax_f32(
            scores,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            past_tokens,
        )?;

        let batch_count_u32 = u32::try_from(batch_count).map_err(|_| {
            Error::backend("attention causal softmax batch_count exceeds Metal u32 limit")
        })?;
        let head_count_u32 = u32::try_from(head_count).map_err(|_| {
            Error::backend("attention causal softmax head_count exceeds Metal u32 limit")
        })?;
        let query_tokens_u32 = u32::try_from(query_tokens).map_err(|_| {
            Error::backend("attention causal softmax query_tokens exceeds Metal u32 limit")
        })?;
        let key_tokens_u32 = u32::try_from(key_tokens).map_err(|_| {
            Error::backend("attention causal softmax key_tokens exceeds Metal u32 limit")
        })?;
        let past_tokens_u32 = u32::try_from(past_tokens).map_err(|_| {
            Error::backend("attention causal softmax past_tokens exceeds Metal u32 limit")
        })?;
        let row_count = batch_count
            .checked_mul(head_count)
            .and_then(|value| value.checked_mul(query_tokens))
            .ok_or_else(|| Error::backend("attention causal softmax row count overflow"))?;
        let output_len = row_count
            .checked_mul(key_tokens)
            .ok_or_else(|| Error::backend("attention causal softmax output length overflow"))?;

        let scores_buffer = f32_buffer(device, scores)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let batch_count_buffer = u32_scalar_buffer(device, batch_count_u32)?;
        let head_count_buffer = u32_scalar_buffer(device, head_count_u32)?;
        let query_tokens_buffer = u32_scalar_buffer(device, query_tokens_u32)?;
        let key_tokens_buffer = u32_scalar_buffer(device, key_tokens_u32)?;
        let past_tokens_buffer = u32_scalar_buffer(device, past_tokens_u32)?;

        trace!(
            target: "inferno::metal",
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            past_tokens,
            "running native Metal attention causal softmax"
        );

        dispatch_1d(
            queue,
            &self.pipeline,
            &[
                &scores_buffer,
                &output_buffer,
                &batch_count_buffer,
                &head_count_buffer,
                &query_tokens_buffer,
                &key_tokens_buffer,
                &past_tokens_buffer,
            ],
            row_count,
        )?;

        let values = read_f32_buffer(&output_buffer, output_len)?;

        Ok(MetalAttentionCausalSoftmaxReport {
            values,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            past_tokens,
            thread_count: row_count,
        })
    }
}

impl MetalDecodeAttention {
    pub(crate) fn new(device: &Device, library: &MetalLibrary) -> Result<Self> {
        Ok(Self {
            scores_pipeline: compute_pipeline(device, library, ATTENTION_SCORES_KERNEL)?,
            paged_scores_pipeline: compute_pipeline(
                device,
                library,
                PAGED_DECODE_ATTENTION_SCORES_KERNEL,
            )?,
            softmax_pipeline: compute_pipeline(device, library, ATTENTION_CAUSAL_SOFTMAX_KERNEL)?,
            values_pipeline: compute_pipeline(device, library, ATTENTION_VALUES_KERNEL)?,
            paged_values_pipeline: compute_pipeline(
                device,
                library,
                PAGED_DECODE_ATTENTION_VALUES_KERNEL,
            )?,
            workspace: Mutex::new(DecodeAttentionWorkspace::default()),
        })
    }

    pub(crate) fn run(
        &self,
        device: &Device,
        queue: &CommandQueue,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch_count: usize,
        head_count: usize,
        query_tokens: usize,
        key_tokens: usize,
        head_dim: usize,
        value_dim: usize,
        past_tokens: usize,
    ) -> Result<MetalDecodeAttentionReport> {
        if query_tokens != 1 {
            return Err(Error::backend(format!(
                "fused decode attention requires query_tokens=1, got {query_tokens}"
            )));
        }
        validate_decode_attention_f32(
            q,
            k,
            v,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            head_dim,
            value_dim,
            past_tokens,
        )?;

        let scores_len = batch_count
            .checked_mul(head_count)
            .and_then(|value| value.checked_mul(query_tokens))
            .and_then(|value| value.checked_mul(key_tokens))
            .ok_or_else(|| Error::backend("fused decode attention score length overflow"))?;
        let output_len = batch_count
            .checked_mul(head_count)
            .and_then(|value| value.checked_mul(query_tokens))
            .and_then(|value| value.checked_mul(value_dim))
            .ok_or_else(|| Error::backend("fused decode attention output length overflow"))?;
        let row_count = batch_count
            .checked_mul(head_count)
            .and_then(|value| value.checked_mul(query_tokens))
            .ok_or_else(|| Error::backend("fused decode attention row count overflow"))?;

        let batch_count_u32 = u32::try_from(batch_count).map_err(|_| {
            Error::backend("fused decode attention batch_count exceeds Metal u32 limit")
        })?;
        let head_count_u32 = u32::try_from(head_count).map_err(|_| {
            Error::backend("fused decode attention head_count exceeds Metal u32 limit")
        })?;
        let query_tokens_u32 = u32::try_from(query_tokens).map_err(|_| {
            Error::backend("fused decode attention query_tokens exceeds Metal u32 limit")
        })?;
        let key_tokens_u32 = u32::try_from(key_tokens).map_err(|_| {
            Error::backend("fused decode attention key_tokens exceeds Metal u32 limit")
        })?;
        let head_dim_u32 = u32::try_from(head_dim).map_err(|_| {
            Error::backend("fused decode attention head_dim exceeds Metal u32 limit")
        })?;
        let value_dim_u32 = u32::try_from(value_dim).map_err(|_| {
            Error::backend("fused decode attention value_dim exceeds Metal u32 limit")
        })?;
        let past_tokens_u32 = u32::try_from(past_tokens).map_err(|_| {
            Error::backend("fused decode attention past_tokens exceeds Metal u32 limit")
        })?;

        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| Error::backend("decode attention workspace lock poisoned"))?;
        let (batch_count_buffer, _) = workspace
            .batch_count_buffer(device, batch_count_u32)?
            .clone_for_dispatch();
        let (head_count_buffer, _) = workspace
            .head_count_buffer(device, head_count_u32)?
            .clone_for_dispatch();
        let (query_tokens_buffer, _) = workspace
            .query_tokens_buffer(device, query_tokens_u32)?
            .clone_for_dispatch();
        let (key_tokens_buffer, _) = workspace
            .key_tokens_buffer(device, key_tokens_u32)?
            .clone_for_dispatch();
        let (head_dim_buffer, _) = workspace
            .head_dim_buffer(device, head_dim_u32)?
            .clone_for_dispatch();
        let (value_dim_buffer, _) = workspace
            .value_dim_buffer(device, value_dim_u32)?
            .clone_for_dispatch();
        let (past_tokens_buffer, _) = workspace
            .past_tokens_buffer(device, past_tokens_u32)?
            .clone_for_dispatch();
        let (scores_buffer, reused_scores_buffer) = workspace
            .scores_buffer(device, scores_len)?
            .clone_for_dispatch();
        let (probs_buffer, reused_probs_buffer) = workspace
            .probs_buffer(device, scores_len)?
            .clone_for_dispatch();
        let (output_buffer, reused_output_buffer) = workspace
            .output_buffer(device, output_len)?
            .clone_for_dispatch();
        let (q_buffer, reused_q_buffer) = workspace.q_buffer(device, q)?.clone_for_dispatch();
        let (k_buffer, reused_k_buffer) = workspace.k_buffer(device, k)?.clone_for_dispatch();
        let (v_buffer, reused_v_buffer) = workspace.v_buffer(device, v)?.clone_for_dispatch();
        drop(workspace);

        trace!(
            target: "inferno::metal",
            batch_count,
            head_count,
            key_tokens,
            head_dim,
            value_dim,
            past_tokens,
            "running fused native Metal decode attention"
        );

        let scores_buffers = [
            &q_buffer,
            &k_buffer,
            &scores_buffer,
            &batch_count_buffer,
            &head_count_buffer,
            &query_tokens_buffer,
            &key_tokens_buffer,
            &head_dim_buffer,
        ];
        let softmax_buffers = [
            &scores_buffer,
            &probs_buffer,
            &batch_count_buffer,
            &head_count_buffer,
            &query_tokens_buffer,
            &key_tokens_buffer,
            &past_tokens_buffer,
        ];
        let values_buffers = [
            &probs_buffer,
            &v_buffer,
            &output_buffer,
            &batch_count_buffer,
            &head_count_buffer,
            &query_tokens_buffer,
            &key_tokens_buffer,
            &value_dim_buffer,
        ];
        dispatch_1d_many(
            queue,
            &[
                Dispatch1d {
                    pipeline: &self.scores_pipeline,
                    buffers: &scores_buffers,
                    threads: scores_len,
                },
                Dispatch1d {
                    pipeline: &self.softmax_pipeline,
                    buffers: &softmax_buffers,
                    threads: row_count,
                },
                Dispatch1d {
                    pipeline: &self.values_pipeline,
                    buffers: &values_buffers,
                    threads: output_len,
                },
            ],
        )?;

        let values = read_f32_buffer(&output_buffer, output_len)?;

        Ok(MetalDecodeAttentionReport {
            values,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            head_dim,
            value_dim,
            past_tokens,
            scores_len,
            probs_len: scores_len,
            output_len,
            reused_scores_buffer,
            reused_probs_buffer,
            reused_output_buffer,
            reused_q_buffer,
            reused_k_buffer,
            reused_v_buffer,
        })
    }

    pub(crate) fn run_paged(
        &self,
        device: &Device,
        queue: &CommandQueue,
        q: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        past_kv: &PagedKvView<'_>,
    ) -> Result<MetalPagedDecodeAttentionReport> {
        past_kv.validate()?;
        validate_append_only_paged_layout(past_kv)?;

        let batch_count = past_kv.batch;
        let head_count = past_kv.attention_heads;
        let past_tokens = past_kv.cached_tokens;
        let key_tokens = past_tokens
            .checked_add(1)
            .ok_or_else(|| Error::backend("paged decode attention key token count overflow"))?;
        let page_count = past_kv.pages.len();
        let page_size = past_kv.page_size;
        let head_dim = past_kv.key_head_dim;
        let value_dim = past_kv.value_head_dim;
        let query_tokens = 1_usize;
        let q_len = batch_count
            .checked_mul(head_count)
            .and_then(|value| value.checked_mul(query_tokens))
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| Error::backend("paged decode attention q length overflow"))?;
        validate_exact_shape("paged_decode_attention_q_values", &[q.len()], &[q_len])?;
        if q.iter().any(|value| !value.is_finite()) {
            return Err(Error::backend(
                "paged decode attention query contains non-finite values",
            ));
        }
        let current_k_len = batch_count
            .checked_mul(head_count)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| Error::backend("paged decode attention current K length overflow"))?;
        let current_v_len = batch_count
            .checked_mul(head_count)
            .and_then(|value| value.checked_mul(value_dim))
            .ok_or_else(|| Error::backend("paged decode attention current V length overflow"))?;
        validate_exact_shape(
            "paged_decode_attention_current_k_values",
            &[current_k.len()],
            &[current_k_len],
        )?;
        validate_exact_shape(
            "paged_decode_attention_current_v_values",
            &[current_v.len()],
            &[current_v_len],
        )?;
        if current_k.iter().any(|value| !value.is_finite())
            || current_v.iter().any(|value| !value.is_finite())
        {
            return Err(Error::backend(
                "paged decode attention current K/V contains non-finite values",
            ));
        }

        let scores_len = batch_count
            .checked_mul(head_count)
            .and_then(|value| value.checked_mul(key_tokens))
            .ok_or_else(|| Error::backend("paged decode attention score length overflow"))?;
        let output_len = batch_count
            .checked_mul(head_count)
            .and_then(|value| value.checked_mul(value_dim))
            .ok_or_else(|| Error::backend("paged decode attention output length overflow"))?;
        let page_k_len = page_buffer_len(page_count, batch_count, head_count, page_size, head_dim)?;
        let page_v_len =
            page_buffer_len(page_count, batch_count, head_count, page_size, value_dim)?;

        let batch_count_u32 = checked_u32("paged decode attention batch_count", batch_count)?;
        let head_count_u32 = checked_u32("paged decode attention head_count", head_count)?;
        let query_tokens_u32 = 1_u32;
        let key_tokens_u32 = checked_u32("paged decode attention key_tokens", key_tokens)?;
        let page_size_u32 = checked_u32("paged decode attention page_size", page_size)?;
        let head_dim_u32 = checked_u32("paged decode attention head_dim", head_dim)?;
        let value_dim_u32 = checked_u32("paged decode attention value_dim", value_dim)?;
        let past_tokens_u32 = checked_u32("paged decode attention past_tokens", past_tokens)?;

        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| Error::backend("decode attention workspace lock poisoned"))?;
        let (batch_count_buffer, _) = workspace
            .batch_count_buffer(device, batch_count_u32)?
            .clone_for_dispatch();
        let (head_count_buffer, _) = workspace
            .head_count_buffer(device, head_count_u32)?
            .clone_for_dispatch();
        let (query_tokens_buffer, _) = workspace
            .query_tokens_buffer(device, query_tokens_u32)?
            .clone_for_dispatch();
        let (key_tokens_buffer, _) = workspace
            .key_tokens_buffer(device, key_tokens_u32)?
            .clone_for_dispatch();
        let (page_size_buffer, _) = workspace
            .page_size_buffer(device, page_size_u32)?
            .clone_for_dispatch();
        let (head_dim_buffer, _) = workspace
            .head_dim_buffer(device, head_dim_u32)?
            .clone_for_dispatch();
        let (value_dim_buffer, _) = workspace
            .value_dim_buffer(device, value_dim_u32)?
            .clone_for_dispatch();
        let (past_tokens_buffer, _) = workspace
            .past_tokens_buffer(device, past_tokens_u32)?
            .clone_for_dispatch();
        let (scores_buffer, reused_scores_buffer) = workspace
            .scores_buffer(device, scores_len)?
            .clone_for_dispatch();
        let (probs_buffer, reused_probs_buffer) = workspace
            .probs_buffer(device, scores_len)?
            .clone_for_dispatch();
        let (output_buffer, reused_output_buffer) = workspace
            .output_buffer(device, output_len)?
            .clone_for_dispatch();
        let (q_buffer, reused_q_buffer) = workspace.q_buffer(device, q)?.clone_for_dispatch();
        drop(workspace);

        let page_k_buffer = empty_f32_buffer(device, page_k_len)?;
        let page_v_buffer = empty_f32_buffer(device, page_v_len)?;
        let current_k_buffer = f32_buffer(device, current_k)?;
        let current_v_buffer = f32_buffer(device, current_v)?;
        let page_upload = write_all_paged_kv_to_buffers(past_kv, &page_k_buffer, &page_v_buffer)?;

        trace!(
            target: "inferno::metal",
            batch_count,
            head_count,
            past_tokens,
            key_tokens,
            page_count,
            page_size,
            head_dim,
            value_dim,
            "running native Metal paged decode attention"
        );

        let scores_buffers = [
            &q_buffer,
            &page_k_buffer,
            &current_k_buffer,
            &scores_buffer,
            &batch_count_buffer,
            &head_count_buffer,
            &past_tokens_buffer,
            &page_size_buffer,
            &head_dim_buffer,
        ];
        let softmax_buffers = [
            &scores_buffer,
            &probs_buffer,
            &batch_count_buffer,
            &head_count_buffer,
            &query_tokens_buffer,
            &key_tokens_buffer,
            &past_tokens_buffer,
        ];
        let values_buffers = [
            &probs_buffer,
            &page_v_buffer,
            &current_v_buffer,
            &output_buffer,
            &batch_count_buffer,
            &head_count_buffer,
            &past_tokens_buffer,
            &page_size_buffer,
            &value_dim_buffer,
        ];
        dispatch_1d_many(
            queue,
            &[
                Dispatch1d {
                    pipeline: &self.paged_scores_pipeline,
                    buffers: &scores_buffers,
                    threads: scores_len,
                },
                Dispatch1d {
                    pipeline: &self.softmax_pipeline,
                    buffers: &softmax_buffers,
                    threads: batch_count * head_count,
                },
                Dispatch1d {
                    pipeline: &self.paged_values_pipeline,
                    buffers: &values_buffers,
                    threads: output_len,
                },
            ],
        )?;

        let values = read_f32_buffer(&output_buffer, output_len)?;

        Ok(MetalPagedDecodeAttentionReport {
            values,
            batch_count,
            head_count,
            past_tokens,
            key_tokens,
            page_count,
            page_size,
            head_dim,
            value_dim,
            scores_len,
            probs_len: scores_len,
            output_len,
            reused_scores_buffer,
            reused_probs_buffer,
            reused_output_buffer,
            reused_q_buffer,
            transient_kv_buffers: true,
            uploaded_page_count: page_upload.uploaded_page_count,
            uploaded_page_tokens: page_upload.uploaded_page_tokens,
        })
    }

    /// Encodes the fused paged decode attention (scores -> causal softmax ->
    /// values) into an open batched command buffer. The query and the current
    /// token's K/V are device-resident buffers produced by earlier encoded
    /// kernels; the past K/V pages are host data and are uploaded into fresh
    /// buffers here (fresh, not pooled: the workspace pool must not be
    /// CPU-written while earlier encoded kernels may still read it).
    ///
    /// Returns the attention output buffer and its element count.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_paged(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        q: &Buffer,
        q_len: usize,
        current_k: &Buffer,
        current_k_len: usize,
        current_v: &Buffer,
        current_v_len: usize,
        past_kv: &PagedKvView<'_>,
    ) -> Result<(Buffer, usize)> {
        past_kv.validate()?;
        validate_append_only_paged_layout(past_kv)?;

        let batch_count = past_kv.batch;
        let head_count = past_kv.attention_heads;
        let past_tokens = past_kv.cached_tokens;
        let key_tokens = past_tokens
            .checked_add(1)
            .ok_or_else(|| Error::backend("paged decode attention key token count overflow"))?;
        let page_count = past_kv.pages.len();
        let page_size = past_kv.page_size;
        let head_dim = past_kv.key_head_dim;
        let value_dim = past_kv.value_head_dim;
        let expected_q_len = batch_count
            .checked_mul(head_count)
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| Error::backend("paged decode attention q length overflow"))?;
        validate_exact_shape("paged_decode_attention_q_values", &[q_len], &[expected_q_len])?;
        let expected_k_len = expected_q_len;
        let expected_v_len = batch_count
            .checked_mul(head_count)
            .and_then(|value| value.checked_mul(value_dim))
            .ok_or_else(|| Error::backend("paged decode attention current V length overflow"))?;
        validate_exact_shape(
            "paged_decode_attention_current_k_values",
            &[current_k_len],
            &[expected_k_len],
        )?;
        validate_exact_shape(
            "paged_decode_attention_current_v_values",
            &[current_v_len],
            &[expected_v_len],
        )?;
        require_f32_capacity(q, q_len, "paged decode attention q")?;
        require_f32_capacity(current_k, current_k_len, "paged decode attention current K")?;
        require_f32_capacity(current_v, current_v_len, "paged decode attention current V")?;

        let scores_len = batch_count
            .checked_mul(head_count)
            .and_then(|value| value.checked_mul(key_tokens))
            .ok_or_else(|| Error::backend("paged decode attention score length overflow"))?;
        let output_len = batch_count
            .checked_mul(head_count)
            .and_then(|value| value.checked_mul(value_dim))
            .ok_or_else(|| Error::backend("paged decode attention output length overflow"))?;
        let page_k_len = page_buffer_len(page_count, batch_count, head_count, page_size, head_dim)?;
        let page_v_len =
            page_buffer_len(page_count, batch_count, head_count, page_size, value_dim)?;

        let batch_count_buffer = u32_scalar_buffer(
            device,
            checked_u32("paged decode attention batch_count", batch_count)?,
        )?;
        let head_count_buffer = u32_scalar_buffer(
            device,
            checked_u32("paged decode attention head_count", head_count)?,
        )?;
        let query_tokens_buffer = u32_scalar_buffer(device, 1_u32)?;
        let key_tokens_buffer = u32_scalar_buffer(
            device,
            checked_u32("paged decode attention key_tokens", key_tokens)?,
        )?;
        let page_size_buffer = u32_scalar_buffer(
            device,
            checked_u32("paged decode attention page_size", page_size)?,
        )?;
        let head_dim_buffer = u32_scalar_buffer(
            device,
            checked_u32("paged decode attention head_dim", head_dim)?,
        )?;
        let value_dim_buffer = u32_scalar_buffer(
            device,
            checked_u32("paged decode attention value_dim", value_dim)?,
        )?;
        let past_tokens_buffer = u32_scalar_buffer(
            device,
            checked_u32("paged decode attention past_tokens", past_tokens)?,
        )?;
        let scores_buffer = empty_f32_buffer(device, scores_len)?;
        let probs_buffer = empty_f32_buffer(device, scores_len)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let page_k_buffer = empty_f32_buffer(device, page_k_len)?;
        let page_v_buffer = empty_f32_buffer(device, page_v_len)?;
        write_all_paged_kv_to_buffers(past_kv, &page_k_buffer, &page_v_buffer)?;

        trace!(
            target: "inferno::metal",
            batch_count,
            head_count,
            past_tokens,
            key_tokens,
            page_count,
            page_size,
            head_dim,
            value_dim,
            "encoding batched paged decode attention"
        );

        encode_1d(
            command_buffer,
            &self.paged_scores_pipeline,
            &[
                q,
                &page_k_buffer,
                current_k,
                &scores_buffer,
                &batch_count_buffer,
                &head_count_buffer,
                &past_tokens_buffer,
                &page_size_buffer,
                &head_dim_buffer,
            ],
            scores_len,
        )?;
        encode_1d(
            command_buffer,
            &self.softmax_pipeline,
            &[
                &scores_buffer,
                &probs_buffer,
                &batch_count_buffer,
                &head_count_buffer,
                &query_tokens_buffer,
                &key_tokens_buffer,
                &past_tokens_buffer,
            ],
            batch_count
                .checked_mul(head_count)
                .ok_or_else(|| Error::backend("paged decode attention row count overflow"))?,
        )?;
        encode_1d(
            command_buffer,
            &self.paged_values_pipeline,
            &[
                &probs_buffer,
                &page_v_buffer,
                current_v,
                &output_buffer,
                &batch_count_buffer,
                &head_count_buffer,
                &past_tokens_buffer,
                &page_size_buffer,
                &value_dim_buffer,
            ],
            output_len,
        )?;
        Ok((output_buffer, output_len))
    }
}

fn validate_decode_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch_count: usize,
    head_count: usize,
    query_tokens: usize,
    key_tokens: usize,
    head_dim: usize,
    value_dim: usize,
    past_tokens: usize,
) -> Result<()> {
    if batch_count == 0
        || head_count == 0
        || query_tokens == 0
        || key_tokens == 0
        || head_dim == 0
        || value_dim == 0
    {
        return Err(Error::backend(
            "fused decode attention dimensions must be positive",
        ));
    }
    validate_exact_shape("fused_decode_attention_query_tokens", &[query_tokens], &[1])?;
    past_tokens
        .checked_add(query_tokens)
        .filter(|expected| *expected == key_tokens)
        .ok_or_else(|| {
            Error::backend(format!(
                "fused decode attention expects key_tokens == past_tokens + query_tokens, got key_tokens={key_tokens}, past_tokens={past_tokens}, query_tokens={query_tokens}"
            ))
        })?;
    let q_len = batch_count
        .checked_mul(head_count)
        .and_then(|value| value.checked_mul(query_tokens))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| Error::backend("fused decode attention q length overflow"))?;
    let k_len = batch_count
        .checked_mul(head_count)
        .and_then(|value| value.checked_mul(key_tokens))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| Error::backend("fused decode attention k length overflow"))?;
    let v_len = batch_count
        .checked_mul(head_count)
        .and_then(|value| value.checked_mul(key_tokens))
        .and_then(|value| value.checked_mul(value_dim))
        .ok_or_else(|| Error::backend("fused decode attention v length overflow"))?;
    validate_exact_shape("fused_decode_attention_q_values", &[q.len()], &[q_len])?;
    validate_exact_shape("fused_decode_attention_k_values", &[k.len()], &[k_len])?;
    validate_exact_shape("fused_decode_attention_v_values", &[v.len()], &[v_len])?;
    if q.iter().any(|value| !value.is_finite())
        || k.iter().any(|value| !value.is_finite())
        || v.iter().any(|value| !value.is_finite())
    {
        return Err(Error::backend(
            "fused decode attention q/k/v contains non-finite values",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PageUploadReport {
    uploaded_page_count: usize,
    uploaded_page_tokens: usize,
}

fn validate_append_only_paged_layout(kv: &PagedKvView<'_>) -> Result<()> {
    let mut expected_start = 0_usize;
    for (page_index, page) in kv.pages.iter().enumerate() {
        if page.physical_page_id != page_index {
            return Err(Error::backend(format!(
                "paged decode attention requires physical_page_id == page order for the lightweight append-only Metal path, got physical_page_id={} at page_index={page_index}",
                page.physical_page_id
            )));
        }
        if page.start_token != expected_start {
            return Err(Error::backend(format!(
                "paged decode attention requires contiguous append-only pages, expected start_token {expected_start}, got {}",
                page.start_token
            )));
        }
        if page_index + 1 < kv.pages.len() && page.token_count != kv.page_size {
            return Err(Error::backend(format!(
                "paged decode attention requires every non-final page to be full, page {page_index} has {} tokens for page_size {}",
                page.token_count, kv.page_size
            )));
        }
        expected_start = expected_start
            .checked_add(page.token_count)
            .ok_or_else(|| Error::backend("paged decode attention page token overflow"))?;
    }
    validate_exact_shape(
        "paged_decode_attention_append_only_cached_tokens",
        &[expected_start],
        &[kv.cached_tokens],
    )?;
    Ok(())
}

fn write_page_values_to_buffer(
    name: &str,
    source: &[f32],
    target: &Buffer,
    page_index: usize,
    batch_count: usize,
    head_count: usize,
    source_page_tokens: usize,
    source_token_start: usize,
    token_count: usize,
    page_size: usize,
    dim: usize,
) -> Result<()> {
    if source.iter().any(|value| !value.is_finite()) {
        return Err(Error::backend(format!(
            "paged decode attention {name} page contains non-finite values"
        )));
    }
    for batch in 0..batch_count {
        for head in 0..head_count {
            for token in 0..token_count {
                let source_token = source_token_start
                    .checked_add(token)
                    .ok_or_else(|| Error::backend("paged decode source token overflow"))?;
                let source_base =
                    (((batch * head_count + head) * source_page_tokens + source_token) * dim)
                        .checked_add(0)
                        .ok_or_else(|| Error::backend("paged decode source index overflow"))?;
                let target_base = (((((page_index * batch_count + batch) * head_count + head)
                    * page_size
                    + source_token)
                    * dim)
                    .checked_add(0))
                .ok_or_else(|| Error::backend("paged decode target index overflow"))?;
                write_f32_buffer_at(target, target_base, &source[source_base..source_base + dim])?;
            }
        }
    }
    Ok(())
}

fn page_buffer_len(
    page_count: usize,
    batch_count: usize,
    head_count: usize,
    page_size: usize,
    dim: usize,
) -> Result<usize> {
    page_count
        .checked_mul(batch_count)
        .and_then(|value| value.checked_mul(head_count))
        .and_then(|value| value.checked_mul(page_size))
        .and_then(|value| value.checked_mul(dim))
        .ok_or_else(|| Error::backend("paged decode attention page buffer length overflow"))
}

fn checked_u32(context: &str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::backend(format!("{context} exceeds Metal u32 limit")))
}

fn write_all_paged_kv_to_buffers(
    kv: &PagedKvView<'_>,
    page_k_buffer: &Buffer,
    page_v_buffer: &Buffer,
) -> Result<PageUploadReport> {
    let mut uploaded_page_count = 0_usize;
    let mut uploaded_page_tokens = 0_usize;
    for (page_index, page) in kv.pages.iter().enumerate() {
        let k_source_tokens = page.k.dims()[2];
        let v_source_tokens = page.v.dims()[2];
        write_page_values_to_buffer(
            "K",
            page.k.values(),
            page_k_buffer,
            page_index,
            kv.batch,
            kv.attention_heads,
            k_source_tokens,
            0,
            page.token_count,
            kv.page_size,
            kv.key_head_dim,
        )?;
        write_page_values_to_buffer(
            "V",
            page.v.values(),
            page_v_buffer,
            page_index,
            kv.batch,
            kv.attention_heads,
            v_source_tokens,
            0,
            page.token_count,
            kv.page_size,
            kv.value_head_dim,
        )?;
        uploaded_page_count += 1;
        uploaded_page_tokens = uploaded_page_tokens
            .checked_add(page.token_count)
            .ok_or_else(|| Error::backend("paged decode uploaded token count overflow"))?;
    }

    Ok(PageUploadReport {
        uploaded_page_count,
        uploaded_page_tokens,
    })
}

impl DecodeAttentionWorkspace {
    fn q_buffer(&mut self, device: &Device, values: &[f32]) -> Result<WorkspaceBufferRef> {
        let buffer = workspace_f32_buffer(device, &mut self.q, values.len())?;
        write_f32_buffer(&buffer.buffer, values)?;
        Ok(buffer)
    }

    fn k_buffer(&mut self, device: &Device, values: &[f32]) -> Result<WorkspaceBufferRef> {
        let buffer = workspace_f32_buffer(device, &mut self.k, values.len())?;
        write_f32_buffer(&buffer.buffer, values)?;
        Ok(buffer)
    }

    fn v_buffer(&mut self, device: &Device, values: &[f32]) -> Result<WorkspaceBufferRef> {
        let buffer = workspace_f32_buffer(device, &mut self.v, values.len())?;
        write_f32_buffer(&buffer.buffer, values)?;
        Ok(buffer)
    }

    fn scores_buffer(&mut self, device: &Device, len: usize) -> Result<WorkspaceBufferRef> {
        workspace_f32_buffer(device, &mut self.scores, len)
    }

    fn probs_buffer(&mut self, device: &Device, len: usize) -> Result<WorkspaceBufferRef> {
        workspace_f32_buffer(device, &mut self.probs, len)
    }

    fn output_buffer(&mut self, device: &Device, len: usize) -> Result<WorkspaceBufferRef> {
        workspace_f32_buffer(device, &mut self.output, len)
    }

    fn batch_count_buffer(&mut self, device: &Device, value: u32) -> Result<WorkspaceBufferRef> {
        workspace_u32_value_buffer(device, &mut self.batch_count, value)
    }

    fn head_count_buffer(&mut self, device: &Device, value: u32) -> Result<WorkspaceBufferRef> {
        workspace_u32_value_buffer(device, &mut self.head_count, value)
    }

    fn query_tokens_buffer(&mut self, device: &Device, value: u32) -> Result<WorkspaceBufferRef> {
        workspace_u32_value_buffer(device, &mut self.query_tokens, value)
    }

    fn key_tokens_buffer(&mut self, device: &Device, value: u32) -> Result<WorkspaceBufferRef> {
        workspace_u32_value_buffer(device, &mut self.key_tokens, value)
    }

    fn page_size_buffer(&mut self, device: &Device, value: u32) -> Result<WorkspaceBufferRef> {
        workspace_u32_value_buffer(device, &mut self.page_size, value)
    }

    fn head_dim_buffer(&mut self, device: &Device, value: u32) -> Result<WorkspaceBufferRef> {
        workspace_u32_value_buffer(device, &mut self.head_dim, value)
    }

    fn value_dim_buffer(&mut self, device: &Device, value: u32) -> Result<WorkspaceBufferRef> {
        workspace_u32_value_buffer(device, &mut self.value_dim, value)
    }

    fn past_tokens_buffer(&mut self, device: &Device, value: u32) -> Result<WorkspaceBufferRef> {
        workspace_u32_value_buffer(device, &mut self.past_tokens, value)
    }
}

struct WorkspaceBufferRef {
    buffer: Buffer,
    reused: bool,
}

impl WorkspaceBufferRef {
    fn clone_for_dispatch(&self) -> (Buffer, bool) {
        (self.buffer.clone(), self.reused)
    }
}

fn workspace_f32_buffer(
    device: &Device,
    slot: &mut Option<WorkspaceBuffer>,
    len: usize,
) -> Result<WorkspaceBufferRef> {
    workspace_buffer_with(device, slot, len, empty_f32_buffer)
}

fn workspace_u32_value_buffer(
    device: &Device,
    slot: &mut Option<WorkspaceBuffer>,
    value: u32,
) -> Result<WorkspaceBufferRef> {
    let buffer = workspace_buffer_with(device, slot, 1, empty_u32_buffer)?;
    write_u32_buffer(&buffer.buffer, &[value])?;
    Ok(buffer)
}

fn workspace_buffer_with(
    device: &Device,
    slot: &mut Option<WorkspaceBuffer>,
    len: usize,
    allocate: fn(&Device, usize) -> Result<Buffer>,
) -> Result<WorkspaceBufferRef> {
    if len == 0 {
        return Err(Error::backend("workspace buffer length must be positive"));
    }
    if let Some(existing) = slot.as_ref() {
        if existing.len >= len {
            return Ok(WorkspaceBufferRef {
                buffer: existing.buffer.clone(),
                reused: true,
            });
        }
    }

    let buffer = allocate(device, len)?;
    *slot = Some(WorkspaceBuffer {
        buffer: buffer.clone(),
        len,
    });
    Ok(WorkspaceBufferRef {
        buffer,
        reused: false,
    })
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use crate::metal::Metal;
    use std::borrow::Cow;

    use common::{F32Tensor, PagedKvPageView, PagedKvView};

    #[test]
    fn matches_cpu_reference_for_small_attention_scores() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch_count = 1;
        let head_count = 2;
        let query_tokens = 2;
        let key_tokens = 3;
        let head_dim = 4;
        let q = (0..batch_count * head_count * query_tokens * head_dim)
            .map(|index| (index as f32 + 1.0) / 10.0)
            .collect::<Vec<_>>();
        let k = (0..batch_count * head_count * key_tokens * head_dim)
            .map(|index| (index as f32 + 1.0) / 20.0)
            .collect::<Vec<_>>();

        let report = metal
            .attention_scores_f32_report(
                &q,
                &k,
                batch_count,
                head_count,
                query_tokens,
                key_tokens,
                head_dim,
            )
            .unwrap();
        let expected = cpu_attention_scores(
            &q,
            &k,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            head_dim,
        );

        assert_eq!(report.batch_count, batch_count);
        assert_eq!(report.head_count, head_count);
        assert_eq!(report.query_tokens, query_tokens);
        assert_eq!(report.key_tokens, key_tokens);
        assert_eq!(report.head_dim, head_dim);
        assert_eq!(
            report.thread_count,
            batch_count * head_count * query_tokens * key_tokens
        );
        assert_close(&report.values, &expected, 1e-5);
    }

    #[test]
    fn rejects_bad_shape_before_dispatch() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let err = metal
            .attention_scores_f32(&[0.5; 8], &[0.25; 10], 1, 2, 2, 3, 2)
            .expect_err("bad k length should fail before Metal dispatch");

        assert!(err.to_string().contains("k shape mismatch"));
    }

    #[test]
    fn value_aggregation_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch_count = 1;
        let head_count = 2;
        let query_tokens = 2;
        let key_tokens = 3;
        let value_dim = 4;
        let probs = vec![
            0.2_f32, 0.3, 0.5, //
            0.1, 0.4, 0.5, //
            0.7, 0.2, 0.1, //
            0.25, 0.25, 0.5,
        ];
        let values = (0..batch_count * head_count * key_tokens * value_dim)
            .map(|index| (index as f32 + 1.0) / 10.0)
            .collect::<Vec<_>>();

        let report = metal
            .attention_values_f32_report(
                &probs,
                &values,
                batch_count,
                head_count,
                query_tokens,
                key_tokens,
                value_dim,
            )
            .unwrap();
        let expected = cpu_attention_values(
            &probs,
            &values,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            value_dim,
        );

        assert_eq!(report.batch_count, batch_count);
        assert_eq!(report.head_count, head_count);
        assert_eq!(report.query_tokens, query_tokens);
        assert_eq!(report.key_tokens, key_tokens);
        assert_eq!(report.value_dim, value_dim);
        assert_eq!(
            report.thread_count,
            batch_count * head_count * query_tokens * value_dim
        );
        assert_close(&report.values, &expected, 1e-5);
    }

    #[test]
    fn value_aggregation_rejects_bad_shape_before_dispatch() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let err = metal
            .attention_values_f32(&[0.5; 12], &[0.25; 17], 1, 2, 2, 3, 3)
            .expect_err("bad value length should fail before Metal dispatch");

        assert!(err.to_string().contains("attention values shape mismatch"));
    }

    #[test]
    fn causal_softmax_matches_cpu_reference_for_prefill() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch_count = 1;
        let head_count = 2;
        let query_tokens = 3;
        let key_tokens = 3;
        let past_tokens = 0;
        let scores = (0..batch_count * head_count * query_tokens * key_tokens)
            .map(|index| (index as f32 - 3.0) / 10.0)
            .collect::<Vec<_>>();

        let report = metal
            .attention_causal_softmax_f32_report(
                &scores,
                batch_count,
                head_count,
                query_tokens,
                key_tokens,
                past_tokens,
            )
            .unwrap();
        let expected = cpu_attention_causal_softmax(
            &scores,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            past_tokens,
        );

        assert_eq!(report.batch_count, batch_count);
        assert_eq!(report.head_count, head_count);
        assert_eq!(report.query_tokens, query_tokens);
        assert_eq!(report.key_tokens, key_tokens);
        assert_eq!(report.past_tokens, past_tokens);
        assert_eq!(report.thread_count, batch_count * head_count * query_tokens);
        assert_close(&report.values, &expected, 1e-5);
    }

    #[test]
    fn causal_softmax_matches_cpu_reference_for_decode() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch_count = 1;
        let head_count = 1;
        let query_tokens = 1;
        let key_tokens = 5;
        let past_tokens = 4;
        let scores = vec![0.0_f32, 1.0, 2.0, 3.0, 4.0];

        let report = metal
            .attention_causal_softmax_f32_report(
                &scores,
                batch_count,
                head_count,
                query_tokens,
                key_tokens,
                past_tokens,
            )
            .unwrap();
        let expected = cpu_attention_causal_softmax(
            &scores,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            past_tokens,
        );

        assert_close(&report.values, &expected, 1e-5);
    }

    #[test]
    fn fused_decode_attention_matches_cpu_reference_and_reuses_workspace() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch_count = 1;
        let head_count = 2;
        let query_tokens = 1;
        let key_tokens = 5;
        let head_dim = 4;
        let value_dim = 3;
        let past_tokens = 4;
        let q = (0..batch_count * head_count * query_tokens * head_dim)
            .map(|index| (index as f32 + 1.0) / 10.0)
            .collect::<Vec<_>>();
        let k = (0..batch_count * head_count * key_tokens * head_dim)
            .map(|index| (index as f32 + 1.0) / 20.0)
            .collect::<Vec<_>>();
        let v = (0..batch_count * head_count * key_tokens * value_dim)
            .map(|index| (index as f32 + 1.0) / 30.0)
            .collect::<Vec<_>>();
        let scores = cpu_attention_scores(
            &q,
            &k,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            head_dim,
        );
        let probs = cpu_attention_causal_softmax(
            &scores,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            past_tokens,
        );
        let expected = cpu_attention_values(
            &probs,
            &v,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            value_dim,
        );

        let first = metal
            .decode_attention_f32_report(
                &q,
                &k,
                &v,
                batch_count,
                head_count,
                key_tokens,
                head_dim,
                value_dim,
                past_tokens,
            )
            .unwrap();
        let second = metal
            .decode_attention_f32_report(
                &q,
                &k,
                &v,
                batch_count,
                head_count,
                key_tokens,
                head_dim,
                value_dim,
                past_tokens,
            )
            .unwrap();

        assert_close(&first.values, &expected, 1e-5);
        assert_close(&second.values, &expected, 1e-5);
        assert!(!first.reused_scores_buffer);
        assert!(!first.reused_probs_buffer);
        assert!(!first.reused_output_buffer);
        assert!(second.reused_scores_buffer);
        assert!(second.reused_probs_buffer);
        assert!(second.reused_output_buffer);
    }

    #[test]
    fn paged_decode_attention_matches_contiguous_reference_and_reuses_workspace() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch_count = 1;
        let head_count = 2;
        let past_tokens = 5;
        let key_tokens = past_tokens + 1;
        let head_dim = 4;
        let value_dim = 3;
        let page_size = 3;
        let q = (0..batch_count * head_count * head_dim)
            .map(|index| (index as f32 + 1.0) / 10.0)
            .collect::<Vec<_>>();
        let past_k = (0..batch_count * head_count * past_tokens * head_dim)
            .map(|index| (index as f32 + 1.0) / 20.0)
            .collect::<Vec<_>>();
        let past_v = (0..batch_count * head_count * past_tokens * value_dim)
            .map(|index| (index as f32 + 1.0) / 30.0)
            .collect::<Vec<_>>();
        let current_k = (0..batch_count * head_count * head_dim)
            .map(|index| (index as f32 + 1.0) / 40.0)
            .collect::<Vec<_>>();
        let current_v = (0..batch_count * head_count * value_dim)
            .map(|index| (index as f32 + 1.0) / 50.0)
            .collect::<Vec<_>>();

        let k_page_0 = page_tensor(
            &past_k,
            batch_count,
            head_count,
            past_tokens,
            0,
            3,
            head_dim,
        );
        let k_page_1 = page_tensor(
            &past_k,
            batch_count,
            head_count,
            past_tokens,
            3,
            2,
            head_dim,
        );
        let v_page_0 = page_tensor(
            &past_v,
            batch_count,
            head_count,
            past_tokens,
            0,
            3,
            value_dim,
        );
        let v_page_1 = page_tensor(
            &past_v,
            batch_count,
            head_count,
            past_tokens,
            3,
            2,
            value_dim,
        );
        let past_kv = PagedKvView {
            batch: batch_count,
            attention_heads: head_count,
            key_head_dim: head_dim,
            value_head_dim: value_dim,
            page_size,
            cached_tokens: past_tokens,
            pages: vec![
                PagedKvPageView {
                    physical_page_id: 0,
                    start_token: 0,
                    token_count: 3,
                    k: Cow::Borrowed(&k_page_0),
                    v: Cow::Borrowed(&v_page_0),
                },
                PagedKvPageView {
                    physical_page_id: 1,
                    start_token: 3,
                    token_count: 2,
                    k: Cow::Borrowed(&k_page_1),
                    v: Cow::Borrowed(&v_page_1),
                },
            ],
        };
        let contiguous_k = append_current_token(
            &past_k,
            &current_k,
            batch_count,
            head_count,
            past_tokens,
            head_dim,
        );
        let contiguous_v = append_current_token(
            &past_v,
            &current_v,
            batch_count,
            head_count,
            past_tokens,
            value_dim,
        );
        let scores = cpu_attention_scores(
            &q,
            &contiguous_k,
            batch_count,
            head_count,
            1,
            key_tokens,
            head_dim,
        );
        let probs = cpu_attention_causal_softmax(
            &scores,
            batch_count,
            head_count,
            1,
            key_tokens,
            past_tokens,
        );
        let expected = cpu_attention_values(
            &probs,
            &contiguous_v,
            batch_count,
            head_count,
            1,
            key_tokens,
            value_dim,
        );

        let first = metal
            .paged_decode_attention_f32_report(&q, &current_k, &current_v, &past_kv)
            .unwrap();
        let second = metal
            .paged_decode_attention_f32_report(&q, &current_k, &current_v, &past_kv)
            .unwrap();

        assert_close(&first.values, &expected, 1e-5);
        assert_close(&second.values, &expected, 1e-5);
        assert_eq!(first.past_tokens, past_tokens);
        assert_eq!(first.key_tokens, key_tokens);
        assert_eq!(first.page_count, 2);
        assert_eq!(first.uploaded_page_count, 2);
        assert_eq!(first.uploaded_page_tokens, past_tokens);
        assert_eq!(second.uploaded_page_count, 2);
        assert_eq!(second.uploaded_page_tokens, past_tokens);
        assert!(first.transient_kv_buffers);
        assert!(second.transient_kv_buffers);
    }

    #[test]
    fn causal_softmax_rejects_bad_cache_offset_before_dispatch() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let err = metal
            .attention_causal_softmax_f32(&[0.0; 6], 1, 1, 2, 3, 0)
            .expect_err("bad cache offset should fail before Metal dispatch");

        assert!(err.to_string().contains("past_tokens"));
    }

    fn native_metal_or_skip() -> Option<Metal> {
        Metal::new().ok()
    }

    fn cpu_attention_scores(
        q: &[f32],
        k: &[f32],
        batch_count: usize,
        head_count: usize,
        query_tokens: usize,
        key_tokens: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; batch_count * head_count * query_tokens * key_tokens];
        let scale = (head_dim as f32).sqrt();

        for batch in 0..batch_count {
            for head in 0..head_count {
                for query in 0..query_tokens {
                    for key in 0..key_tokens {
                        let mut sum = 0.0;
                        for dim in 0..head_dim {
                            let q_index = (((batch * head_count + head) * query_tokens + query)
                                * head_dim)
                                + dim;
                            let k_index =
                                (((batch * head_count + head) * key_tokens + key) * head_dim) + dim;
                            sum += q[q_index] * k[k_index];
                        }
                        let output_index = (((batch * head_count + head) * query_tokens + query)
                            * key_tokens)
                            + key;
                        output[output_index] = sum / scale;
                    }
                }
            }
        }

        output
    }

    fn cpu_attention_values(
        probs: &[f32],
        values: &[f32],
        batch_count: usize,
        head_count: usize,
        query_tokens: usize,
        key_tokens: usize,
        value_dim: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; batch_count * head_count * query_tokens * value_dim];

        for batch in 0..batch_count {
            for head in 0..head_count {
                for query in 0..query_tokens {
                    for value_index in 0..value_dim {
                        let mut sum = 0.0;
                        for key in 0..key_tokens {
                            let probs_index =
                                (((batch * head_count + head) * query_tokens + query) * key_tokens)
                                    + key;
                            let values_index = (((batch * head_count + head) * key_tokens + key)
                                * value_dim)
                                + value_index;
                            sum += probs[probs_index] * values[values_index];
                        }
                        let output_index = (((batch * head_count + head) * query_tokens + query)
                            * value_dim)
                            + value_index;
                        output[output_index] = sum;
                    }
                }
            }
        }

        output
    }

    fn cpu_attention_causal_softmax(
        scores: &[f32],
        batch_count: usize,
        head_count: usize,
        query_tokens: usize,
        key_tokens: usize,
        past_tokens: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; scores.len()];

        for batch in 0..batch_count {
            for head in 0..head_count {
                for query in 0..query_tokens {
                    let base = ((batch * head_count + head) * query_tokens + query) * key_tokens;
                    let max_visible_key = past_tokens + query;
                    let row = &scores[base..base + max_visible_key + 1];
                    let max_value = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0;
                    for key in 0..=max_visible_key {
                        let value = (scores[base + key] - max_value).exp();
                        output[base + key] = value;
                        sum += value;
                    }
                    for key in 0..=max_visible_key {
                        output[base + key] /= sum;
                    }
                }
            }
        }

        output
    }

    fn page_tensor(
        source: &[f32],
        batch_count: usize,
        head_count: usize,
        source_tokens: usize,
        start_token: usize,
        token_count: usize,
        dim: usize,
    ) -> F32Tensor {
        let mut values = vec![0.0_f32; batch_count * head_count * token_count * dim];
        for batch in 0..batch_count {
            for head in 0..head_count {
                for token in 0..token_count {
                    let source_token = start_token + token;
                    let source_base = (((batch * head_count + head) * source_tokens + source_token)
                        * dim) as usize;
                    let target_base =
                        (((batch * head_count + head) * token_count + token) * dim) as usize;
                    values[target_base..target_base + dim]
                        .copy_from_slice(&source[source_base..source_base + dim]);
                }
            }
        }
        F32Tensor::new(values, [batch_count, head_count, token_count, dim]).unwrap()
    }

    fn append_current_token(
        past: &[f32],
        current: &[f32],
        batch_count: usize,
        head_count: usize,
        past_tokens: usize,
        dim: usize,
    ) -> Vec<f32> {
        let key_tokens = past_tokens + 1;
        let mut values = vec![0.0_f32; batch_count * head_count * key_tokens * dim];
        for batch in 0..batch_count {
            for head in 0..head_count {
                for token in 0..past_tokens {
                    let past_base =
                        (((batch * head_count + head) * past_tokens + token) * dim) as usize;
                    let target_base =
                        (((batch * head_count + head) * key_tokens + token) * dim) as usize;
                    values[target_base..target_base + dim]
                        .copy_from_slice(&past[past_base..past_base + dim]);
                }
                let current_base = ((batch * head_count + head) * dim) as usize;
                let target_base =
                    (((batch * head_count + head) * key_tokens + past_tokens) * dim) as usize;
                values[target_base..target_base + dim]
                    .copy_from_slice(&current[current_base..current_base + dim]);
            }
        }
        values
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let delta = (actual - expected).abs();
            assert!(
                delta <= tolerance,
                "value {index} differs: actual={actual}, expected={expected}, delta={delta}"
            );
        }
    }
}
