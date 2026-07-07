use std::sync::Mutex;

use ::metal::Buffer;
use ::metal::{CommandBufferRef, CommandQueue, ComputePipelineState, Device};
use common::{Error, Result};
use tracing::trace;

use super::{
    buffers::{
        empty_f32_buffer, empty_u32_buffer, read_f32_buffer, read_u32_buffer, require_f32_capacity,
        u32_buffer, u32_scalar_buffer, u8_buffer_no_copy, write_f32_buffer, write_u32_buffer,
    },
    command::{dispatch_1d, dispatch_1d_many, encode_1d, Dispatch1d},
    library::MetalLibrary,
    pipeline::compute_pipeline,
    validation::{
        debug_assert_finite_values, validate_q2_k_gate_up_swiglu_f32, validate_q2_k_matvec_buffer,
        validate_q2_k_matvec_f32, validate_q2_k_transposed_matvec_buffer,
        validate_q2_k_transposed_matvec_f32, validate_q8_0_matvec_buffer, validate_q8_0_matvec_f32,
        validate_q8_0_transposed_matvec_buffer, validate_q8_0_transposed_matvec_f32,
        Q2_K_BLOCK_BYTES, Q2_K_BLOCK_VALUES,
    },
};

/// Which quantized matvec kernel family a batched encode call targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuantMatvecKind {
    Q2K,
    Q2KTransposed,
    Q80,
    Q80Transposed,
}

const Q2_K_MATVEC_KERNEL: &str = "q2_k_matvec_f32_kernel";
const Q2_K_MATVEC_ADD_KERNEL: &str = "q2_k_matvec_add_f32_kernel";
const Q2_K_GATE_UP_SWIGLU_KERNEL: &str = "q2_k_gate_up_swiglu_f32_kernel";
const Q2_K_MULTI_EXPERT_GATE_UP_SWIGLU_KERNEL: &str = "q2_k_multi_expert_gate_up_swiglu_f32_kernel";
const Q2_K_MULTI_EXPERT_MATVEC_KERNEL: &str = "q2_k_multi_expert_matvec_f32_kernel";
const Q2_K_TRANSPOSED_MATVEC_KERNEL: &str = "q2_k_transposed_matvec_f32_kernel";
const Q8_0_MATVEC_KERNEL: &str = "q8_0_matvec_f32_kernel";
const Q8_0_TRANSPOSED_MATVEC_KERNEL: &str = "q8_0_transposed_matvec_f32_kernel";
const ARGMAX_F32_KERNEL: &str = "argmax_f32_kernel";
const Q2_K_SIMD_LANES: usize = 32;
const ARGMAX_THREADS_PER_VECTOR: usize = 256;

pub(crate) struct MetalQ2Matvec {
    pipeline: ComputePipelineState,
    add_pipeline: ComputePipelineState,
    gate_up_swiglu_pipeline: ComputePipelineState,
    multi_expert_gate_up_swiglu_pipeline: ComputePipelineState,
    multi_expert_matvec_pipeline: ComputePipelineState,
    transposed_pipeline: ComputePipelineState,
    q8_0_pipeline: ComputePipelineState,
    q8_0_transposed_pipeline: ComputePipelineState,
    argmax_pipeline: ComputePipelineState,
    scratch: Mutex<Q2ScratchBuffers>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalQ2MatvecReport {
    pub values: Vec<f32>,
    pub row_count: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub blocks_per_row: usize,
    pub input_len: usize,
    pub weight_bytes: usize,
    pub thread_count: usize,
    pub transposed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalQ2MatvecAddReport {
    pub values: Vec<f32>,
    pub row_count: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub blocks_per_row: usize,
    pub input_len: usize,
    pub residual_len: usize,
    pub weight_bytes: usize,
    pub thread_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalQ2MatvecArgmaxReport {
    pub token_id: u32,
    pub token_score: f32,
    pub row_count: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub blocks_per_row: usize,
    pub input_len: usize,
    pub weight_bytes: usize,
    pub matvec_thread_count: usize,
    pub argmax_thread_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalQ2GateUpSwiGluReport {
    pub values: Vec<f32>,
    pub row_count: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub blocks_per_row: usize,
    pub input_len: usize,
    pub gate_weight_bytes: usize,
    pub up_weight_bytes: usize,
    pub thread_count: usize,
}

#[derive(Debug)]
struct WeightBuffer {
    buffer: Buffer,
}

#[derive(Debug)]
struct ScratchBuffer {
    buffer: Buffer,
    len: usize,
}

#[derive(Debug, Default)]
struct Q2ScratchBuffers {
    input: Option<ScratchBuffer>,
    residual: Option<ScratchBuffer>,
    output: Option<ScratchBuffer>,
    logits: Option<ScratchBuffer>,
    token_id: Option<ScratchBuffer>,
    token_score: Option<ScratchBuffer>,
    row_count: Option<ScratchBuffer>,
    in_features: Option<ScratchBuffer>,
    out_features: Option<ScratchBuffer>,
    blocks_per_row: Option<ScratchBuffer>,
    output_len: Option<ScratchBuffer>,
}

#[derive(Debug)]
struct ScratchBufferRef {
    buffer: Buffer,
    reused: bool,
}

impl MetalQ2Matvec {
    pub(crate) fn new(device: &Device, library: &MetalLibrary) -> Result<Self> {
        Ok(Self {
            pipeline: compute_pipeline(device, library, Q2_K_MATVEC_KERNEL)?,
            add_pipeline: compute_pipeline(device, library, Q2_K_MATVEC_ADD_KERNEL)?,
            gate_up_swiglu_pipeline: compute_pipeline(device, library, Q2_K_GATE_UP_SWIGLU_KERNEL)?,
            multi_expert_gate_up_swiglu_pipeline: compute_pipeline(
                device,
                library,
                Q2_K_MULTI_EXPERT_GATE_UP_SWIGLU_KERNEL,
            )?,
            multi_expert_matvec_pipeline: compute_pipeline(
                device,
                library,
                Q2_K_MULTI_EXPERT_MATVEC_KERNEL,
            )?,
            transposed_pipeline: compute_pipeline(device, library, Q2_K_TRANSPOSED_MATVEC_KERNEL)?,
            q8_0_pipeline: compute_pipeline(device, library, Q8_0_MATVEC_KERNEL)?,
            q8_0_transposed_pipeline: compute_pipeline(
                device,
                library,
                Q8_0_TRANSPOSED_MATVEC_KERNEL,
            )?,
            argmax_pipeline: compute_pipeline(device, library, ARGMAX_F32_KERNEL)?,
            scratch: Mutex::new(Q2ScratchBuffers::default()),
        })
    }

    pub(crate) fn run(
        &self,
        device: &Device,
        queue: &CommandQueue,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<MetalQ2MatvecReport> {
        let blocks_per_row =
            validate_q2_k_matvec_f32(weights, input, row_count, in_features, out_features)?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K matvec output length overflow"))?;
        let physical_threads = q2_k_cooperative_threads(&self.pipeline, output_len, "Q2_K matvec")?;

        let row_count_u32 = u32::try_from(row_count)
            .map_err(|_| Error::backend("Q2_K matvec row_count exceeds Metal u32 limit"))?;
        let in_features_u32 = u32::try_from(in_features)
            .map_err(|_| Error::backend("Q2_K matvec in_features exceeds Metal u32 limit"))?;
        let out_features_u32 = u32::try_from(out_features)
            .map_err(|_| Error::backend("Q2_K matvec out_features exceeds Metal u32 limit"))?;
        let blocks_per_row_u32 = u32::try_from(blocks_per_row)
            .map_err(|_| Error::backend("Q2_K matvec blocks_per_row exceeds Metal u32 limit"))?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Q2 scratch buffer lock poisoned"))?;
        let input_buffer = scratch.input_buffer(device, input)?;
        let output_buffer = scratch.output_buffer(device, output_len)?;
        let row_count_buffer = scratch.row_count_buffer(device, row_count_u32)?;
        let in_features_buffer = scratch.in_features_buffer(device, in_features_u32)?;
        let out_features_buffer = scratch.out_features_buffer(device, out_features_u32)?;
        let blocks_per_row_buffer = scratch.blocks_per_row_buffer(device, blocks_per_row_u32)?;

        trace!(
            target: "inferno::metal",
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            weight_bytes = weights.len(),
            weight_zero_copy = true,
            input_buffer_reused = input_buffer.reused,
            output_buffer_reused = output_buffer.reused,
            "running native Metal Q2_K matvec"
        );

        dispatch_1d(
            queue,
            &self.pipeline,
            &[
                &weight_buffer.buffer,
                &input_buffer.buffer,
                &output_buffer.buffer,
                &row_count_buffer.buffer,
                &in_features_buffer.buffer,
                &out_features_buffer.buffer,
                &blocks_per_row_buffer.buffer,
            ],
            physical_threads,
        )?;

        let values = read_f32_buffer(&output_buffer.buffer, output_len)?;

        Ok(MetalQ2MatvecReport {
            values,
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            input_len: input.len(),
            weight_bytes: weights.len(),
            thread_count: physical_threads,
            transposed: false,
        })
    }

    pub(crate) fn run_add(
        &self,
        device: &Device,
        queue: &CommandQueue,
        weights: &[u8],
        input: &[f32],
        residual: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<MetalQ2MatvecAddReport> {
        let blocks_per_row =
            validate_q2_k_matvec_f32(weights, input, row_count, in_features, out_features)?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K matvec add output length overflow"))?;
        let physical_threads =
            q2_k_cooperative_threads(&self.add_pipeline, output_len, "Q2_K matvec add")?;
        if residual.len() != output_len {
            return Err(Error::backend(format!(
                "Q2_K matvec add residual length mismatch: expected {output_len}, got {}",
                residual.len()
            )));
        }
        debug_assert_finite_values("Q2_K matvec add residual", residual);

        let row_count_u32 = u32::try_from(row_count)
            .map_err(|_| Error::backend("Q2_K matvec add row_count exceeds Metal u32 limit"))?;
        let in_features_u32 = u32::try_from(in_features)
            .map_err(|_| Error::backend("Q2_K matvec add in_features exceeds Metal u32 limit"))?;
        let out_features_u32 = u32::try_from(out_features)
            .map_err(|_| Error::backend("Q2_K matvec add out_features exceeds Metal u32 limit"))?;
        let blocks_per_row_u32 = u32::try_from(blocks_per_row).map_err(|_| {
            Error::backend("Q2_K matvec add blocks_per_row exceeds Metal u32 limit")
        })?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Q2 scratch buffer lock poisoned"))?;
        let input_buffer = scratch.input_buffer(device, input)?;
        let residual_buffer = scratch.residual_buffer(device, residual)?;
        let output_buffer = scratch.output_buffer(device, output_len)?;
        let row_count_buffer = scratch.row_count_buffer(device, row_count_u32)?;
        let in_features_buffer = scratch.in_features_buffer(device, in_features_u32)?;
        let out_features_buffer = scratch.out_features_buffer(device, out_features_u32)?;
        let blocks_per_row_buffer = scratch.blocks_per_row_buffer(device, blocks_per_row_u32)?;

        trace!(
            target: "inferno::metal",
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            weight_bytes = weights.len(),
            weight_zero_copy = true,
            input_buffer_reused = input_buffer.reused,
            residual_buffer_reused = residual_buffer.reused,
            output_buffer_reused = output_buffer.reused,
            "running native Metal Q2_K matvec plus residual"
        );

        dispatch_1d(
            queue,
            &self.add_pipeline,
            &[
                &weight_buffer.buffer,
                &input_buffer.buffer,
                &residual_buffer.buffer,
                &output_buffer.buffer,
                &row_count_buffer.buffer,
                &in_features_buffer.buffer,
                &out_features_buffer.buffer,
                &blocks_per_row_buffer.buffer,
            ],
            physical_threads,
        )?;

        let values = read_f32_buffer(&output_buffer.buffer, output_len)?;

        Ok(MetalQ2MatvecAddReport {
            values,
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            input_len: input.len(),
            residual_len: residual.len(),
            weight_bytes: weights.len(),
            thread_count: physical_threads,
        })
    }

    pub(crate) fn run_gate_up_swiglu(
        &self,
        device: &Device,
        queue: &CommandQueue,
        gate_weights: &[u8],
        up_weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<MetalQ2GateUpSwiGluReport> {
        let blocks_per_row = validate_q2_k_gate_up_swiglu_f32(
            gate_weights,
            up_weights,
            input,
            row_count,
            in_features,
            out_features,
        )?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K gate/up SwiGLU output length overflow"))?;
        let physical_threads = q2_k_cooperative_threads(
            &self.gate_up_swiglu_pipeline,
            output_len,
            "Q2_K gate/up SwiGLU",
        )?;

        let row_count_u32 = u32::try_from(row_count)
            .map_err(|_| Error::backend("Q2_K gate/up SwiGLU row_count exceeds Metal u32 limit"))?;
        let in_features_u32 = u32::try_from(in_features).map_err(|_| {
            Error::backend("Q2_K gate/up SwiGLU in_features exceeds Metal u32 limit")
        })?;
        let out_features_u32 = u32::try_from(out_features).map_err(|_| {
            Error::backend("Q2_K gate/up SwiGLU out_features exceeds Metal u32 limit")
        })?;
        let blocks_per_row_u32 = u32::try_from(blocks_per_row).map_err(|_| {
            Error::backend("Q2_K gate/up SwiGLU blocks_per_row exceeds Metal u32 limit")
        })?;

        let gate_weight_buffer = self.weight_buffer(device, gate_weights)?;
        let up_weight_buffer = self.weight_buffer(device, up_weights)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Q2 scratch buffer lock poisoned"))?;
        let input_buffer = scratch.input_buffer(device, input)?;
        let output_buffer = scratch.output_buffer(device, output_len)?;
        let row_count_buffer = scratch.row_count_buffer(device, row_count_u32)?;
        let in_features_buffer = scratch.in_features_buffer(device, in_features_u32)?;
        let out_features_buffer = scratch.out_features_buffer(device, out_features_u32)?;
        let blocks_per_row_buffer = scratch.blocks_per_row_buffer(device, blocks_per_row_u32)?;

        trace!(
            target: "inferno::metal",
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            gate_weight_bytes = gate_weights.len(),
            up_weight_bytes = up_weights.len(),
            gate_weight_zero_copy = true,
            up_weight_zero_copy = true,
            input_buffer_reused = input_buffer.reused,
            output_buffer_reused = output_buffer.reused,
            "running native Metal Q2_K gate/up SwiGLU"
        );

        dispatch_1d(
            queue,
            &self.gate_up_swiglu_pipeline,
            &[
                &gate_weight_buffer.buffer,
                &up_weight_buffer.buffer,
                &input_buffer.buffer,
                &output_buffer.buffer,
                &row_count_buffer.buffer,
                &in_features_buffer.buffer,
                &out_features_buffer.buffer,
                &blocks_per_row_buffer.buffer,
            ],
            physical_threads,
        )?;

        let values = read_f32_buffer(&output_buffer.buffer, output_len)?;

        Ok(MetalQ2GateUpSwiGluReport {
            values,
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            input_len: input.len(),
            gate_weight_bytes: gate_weights.len(),
            up_weight_bytes: up_weights.len(),
            thread_count: physical_threads,
        })
    }

    pub(crate) fn run_argmax(
        &self,
        device: &Device,
        queue: &CommandQueue,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<MetalQ2MatvecArgmaxReport> {
        if row_count != 1 {
            return Err(Error::backend(format!(
                "Q2_K greedy argmax requires row_count 1, got {row_count}"
            )));
        }
        let blocks_per_row =
            validate_q2_k_matvec_f32(weights, input, row_count, in_features, out_features)?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K argmax matvec output length overflow"))?;
        let matvec_threads =
            q2_k_cooperative_threads(&self.pipeline, output_len, "Q2_K argmax matvec")?;

        let row_count_u32 = u32::try_from(row_count)
            .map_err(|_| Error::backend("Q2_K argmax row_count exceeds Metal u32 limit"))?;
        let in_features_u32 = u32::try_from(in_features)
            .map_err(|_| Error::backend("Q2_K argmax in_features exceeds Metal u32 limit"))?;
        let out_features_u32 = u32::try_from(out_features)
            .map_err(|_| Error::backend("Q2_K argmax out_features exceeds Metal u32 limit"))?;
        let blocks_per_row_u32 = u32::try_from(blocks_per_row)
            .map_err(|_| Error::backend("Q2_K argmax blocks_per_row exceeds Metal u32 limit"))?;
        let output_len_u32 = u32::try_from(output_len)
            .map_err(|_| Error::backend("Q2_K argmax output length exceeds Metal u32 limit"))?;
        let argmax_threads = argmax_threads(&self.argmax_pipeline, "Q2_K greedy argmax")?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Q2 scratch buffer lock poisoned"))?;
        let input_buffer = scratch.input_buffer(device, input)?;
        let logits_buffer = scratch.logits_buffer(device, output_len)?;
        let row_count_buffer = scratch.row_count_buffer(device, row_count_u32)?;
        let in_features_buffer = scratch.in_features_buffer(device, in_features_u32)?;
        let out_features_buffer = scratch.out_features_buffer(device, out_features_u32)?;
        let blocks_per_row_buffer = scratch.blocks_per_row_buffer(device, blocks_per_row_u32)?;

        trace!(
            target: "inferno::metal",
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            weight_bytes = weights.len(),
            weight_zero_copy = true,
            input_buffer_reused = input_buffer.reused,
            logits_buffer_reused = logits_buffer.reused,
            "running native Metal Q2_K matvec for greedy argmax"
        );

        let token_id_buffer = scratch.token_id_buffer(device)?;
        let token_score_buffer = scratch.token_score_buffer(device)?;
        let output_len_buffer = scratch.output_len_buffer(device, output_len_u32)?;

        trace!(
            target: "inferno::metal",
            value_count = output_len,
            token_id_buffer_reused = token_id_buffer.reused,
            token_score_buffer_reused = token_score_buffer.reused,
            "running native Metal greedy argmax over Q2 logits"
        );

        let matvec_buffers = [
            &weight_buffer.buffer,
            &input_buffer.buffer,
            &logits_buffer.buffer,
            &row_count_buffer.buffer,
            &in_features_buffer.buffer,
            &out_features_buffer.buffer,
            &blocks_per_row_buffer.buffer,
        ];
        let argmax_buffers = [
            &logits_buffer.buffer,
            &token_id_buffer.buffer,
            &token_score_buffer.buffer,
            &output_len_buffer.buffer,
        ];
        dispatch_1d_many(
            queue,
            &[
                Dispatch1d {
                    pipeline: &self.pipeline,
                    buffers: &matvec_buffers,
                    threads: matvec_threads,
                },
                Dispatch1d {
                    pipeline: &self.argmax_pipeline,
                    buffers: &argmax_buffers,
                    threads: argmax_threads,
                },
            ],
        )?;

        let token_id = read_u32_buffer(&token_id_buffer.buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::backend("Q2_K argmax produced no token id"))?;
        let token_score = read_f32_buffer(&token_score_buffer.buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::backend("Q2_K argmax produced no token score"))?;

        Ok(MetalQ2MatvecArgmaxReport {
            token_id,
            token_score,
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            input_len: input.len(),
            weight_bytes: weights.len(),
            matvec_thread_count: matvec_threads,
            argmax_thread_count: argmax_threads,
        })
    }

    pub(crate) fn run_argmax_with_input_buffer_after_dispatch(
        &self,
        device: &Device,
        queue: &CommandQueue,
        weights: &[u8],
        input_buffer: &Buffer,
        input_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
        prefix: Dispatch1d<'_>,
    ) -> Result<MetalQ2MatvecArgmaxReport> {
        if row_count != 1 {
            return Err(Error::backend(format!(
                "Q2_K greedy argmax requires row_count 1, got {row_count}"
            )));
        }
        let blocks_per_row =
            validate_q2_k_matvec_buffer(weights, input_len, row_count, in_features, out_features)?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K argmax matvec output length overflow"))?;
        let matvec_threads =
            q2_k_cooperative_threads(&self.pipeline, output_len, "Q2_K argmax matvec")?;

        let row_count_u32 = u32::try_from(row_count)
            .map_err(|_| Error::backend("Q2_K argmax row_count exceeds Metal u32 limit"))?;
        let in_features_u32 = u32::try_from(in_features)
            .map_err(|_| Error::backend("Q2_K argmax in_features exceeds Metal u32 limit"))?;
        let out_features_u32 = u32::try_from(out_features)
            .map_err(|_| Error::backend("Q2_K argmax out_features exceeds Metal u32 limit"))?;
        let blocks_per_row_u32 = u32::try_from(blocks_per_row)
            .map_err(|_| Error::backend("Q2_K argmax blocks_per_row exceeds Metal u32 limit"))?;
        let output_len_u32 = u32::try_from(output_len)
            .map_err(|_| Error::backend("Q2_K argmax output length exceeds Metal u32 limit"))?;
        let argmax_threads = argmax_threads(&self.argmax_pipeline, "Q2_K greedy argmax")?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Q2 scratch buffer lock poisoned"))?;
        let logits_buffer = scratch.logits_buffer(device, output_len)?;
        let row_count_buffer = scratch.row_count_buffer(device, row_count_u32)?;
        let in_features_buffer = scratch.in_features_buffer(device, in_features_u32)?;
        let out_features_buffer = scratch.out_features_buffer(device, out_features_u32)?;
        let blocks_per_row_buffer = scratch.blocks_per_row_buffer(device, blocks_per_row_u32)?;

        trace!(
            target: "inferno::metal",
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            weight_bytes = weights.len(),
            weight_zero_copy = true,
            logits_buffer_reused = logits_buffer.reused,
            "running native Metal Q2_K argmax from resident input buffer"
        );

        let token_id_buffer = scratch.token_id_buffer(device)?;
        let token_score_buffer = scratch.token_score_buffer(device)?;
        let output_len_buffer = scratch.output_len_buffer(device, output_len_u32)?;

        let matvec_buffers = [
            &weight_buffer.buffer,
            input_buffer,
            &logits_buffer.buffer,
            &row_count_buffer.buffer,
            &in_features_buffer.buffer,
            &out_features_buffer.buffer,
            &blocks_per_row_buffer.buffer,
        ];
        let argmax_buffers = [
            &logits_buffer.buffer,
            &token_id_buffer.buffer,
            &token_score_buffer.buffer,
            &output_len_buffer.buffer,
        ];
        dispatch_1d_many(
            queue,
            &[
                prefix,
                Dispatch1d {
                    pipeline: &self.pipeline,
                    buffers: &matvec_buffers,
                    threads: matvec_threads,
                },
                Dispatch1d {
                    pipeline: &self.argmax_pipeline,
                    buffers: &argmax_buffers,
                    threads: argmax_threads,
                },
            ],
        )?;

        let token_id = read_u32_buffer(&token_id_buffer.buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::backend("Q2_K argmax produced no token id"))?;
        let token_score = read_f32_buffer(&token_score_buffer.buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::backend("Q2_K argmax produced no token score"))?;

        Ok(MetalQ2MatvecArgmaxReport {
            token_id,
            token_score,
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            input_len,
            weight_bytes: weights.len(),
            matvec_thread_count: matvec_threads,
            argmax_thread_count: argmax_threads,
        })
    }

    pub(crate) fn run_q8_0(
        &self,
        device: &Device,
        queue: &CommandQueue,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<MetalQ2MatvecReport> {
        let blocks_per_row =
            validate_q8_0_matvec_f32(weights, input, row_count, in_features, out_features)?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q8_0 matvec output length overflow"))?;

        let row_count_u32 = u32::try_from(row_count)
            .map_err(|_| Error::backend("Q8_0 matvec row_count exceeds Metal u32 limit"))?;
        let in_features_u32 = u32::try_from(in_features)
            .map_err(|_| Error::backend("Q8_0 matvec in_features exceeds Metal u32 limit"))?;
        let out_features_u32 = u32::try_from(out_features)
            .map_err(|_| Error::backend("Q8_0 matvec out_features exceeds Metal u32 limit"))?;
        let blocks_per_row_u32 = u32::try_from(blocks_per_row)
            .map_err(|_| Error::backend("Q8_0 matvec blocks_per_row exceeds Metal u32 limit"))?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Q2 scratch buffer lock poisoned"))?;
        let input_buffer = scratch.input_buffer(device, input)?;
        let output_buffer = scratch.output_buffer(device, output_len)?;
        let row_count_buffer = scratch.row_count_buffer(device, row_count_u32)?;
        let in_features_buffer = scratch.in_features_buffer(device, in_features_u32)?;
        let out_features_buffer = scratch.out_features_buffer(device, out_features_u32)?;
        let blocks_per_row_buffer = scratch.blocks_per_row_buffer(device, blocks_per_row_u32)?;

        trace!(
            target: "inferno::metal",
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            weight_bytes = weights.len(),
            weight_zero_copy = true,
            input_buffer_reused = input_buffer.reused,
            output_buffer_reused = output_buffer.reused,
            "running native Metal Q8_0 matvec"
        );

        dispatch_1d(
            queue,
            &self.q8_0_pipeline,
            &[
                &weight_buffer.buffer,
                &input_buffer.buffer,
                &output_buffer.buffer,
                &row_count_buffer.buffer,
                &in_features_buffer.buffer,
                &out_features_buffer.buffer,
                &blocks_per_row_buffer.buffer,
            ],
            output_len,
        )?;

        let values = read_f32_buffer(&output_buffer.buffer, output_len)?;

        Ok(MetalQ2MatvecReport {
            values,
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            input_len: input.len(),
            weight_bytes: weights.len(),
            thread_count: output_len,
            transposed: false,
        })
    }

    pub(crate) fn run_q8_0_transposed(
        &self,
        device: &Device,
        queue: &CommandQueue,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<MetalQ2MatvecReport> {
        let blocks_per_input_row = validate_q8_0_transposed_matvec_f32(
            weights,
            input,
            row_count,
            in_features,
            out_features,
        )?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q8_0 transposed matvec output length overflow"))?;

        let row_count_u32 = u32::try_from(row_count).map_err(|_| {
            Error::backend("Q8_0 transposed matvec row_count exceeds Metal u32 limit")
        })?;
        let in_features_u32 = u32::try_from(in_features).map_err(|_| {
            Error::backend("Q8_0 transposed matvec in_features exceeds Metal u32 limit")
        })?;
        let out_features_u32 = u32::try_from(out_features).map_err(|_| {
            Error::backend("Q8_0 transposed matvec out_features exceeds Metal u32 limit")
        })?;
        let blocks_per_input_row_u32 = u32::try_from(blocks_per_input_row).map_err(|_| {
            Error::backend("Q8_0 transposed matvec blocks_per_input_row exceeds Metal u32 limit")
        })?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Q2 scratch buffer lock poisoned"))?;
        let input_buffer = scratch.input_buffer(device, input)?;
        let output_buffer = scratch.output_buffer(device, output_len)?;
        let row_count_buffer = scratch.row_count_buffer(device, row_count_u32)?;
        let in_features_buffer = scratch.in_features_buffer(device, in_features_u32)?;
        let out_features_buffer = scratch.out_features_buffer(device, out_features_u32)?;
        let blocks_per_input_row_buffer =
            scratch.blocks_per_row_buffer(device, blocks_per_input_row_u32)?;

        trace!(
            target: "inferno::metal",
            row_count,
            in_features,
            out_features,
            blocks_per_input_row,
            weight_bytes = weights.len(),
            weight_zero_copy = true,
            input_buffer_reused = input_buffer.reused,
            output_buffer_reused = output_buffer.reused,
            "running native Metal transposed Q8_0 matvec"
        );

        dispatch_1d(
            queue,
            &self.q8_0_transposed_pipeline,
            &[
                &weight_buffer.buffer,
                &input_buffer.buffer,
                &output_buffer.buffer,
                &row_count_buffer.buffer,
                &in_features_buffer.buffer,
                &out_features_buffer.buffer,
                &blocks_per_input_row_buffer.buffer,
            ],
            output_len,
        )?;

        let values = read_f32_buffer(&output_buffer.buffer, output_len)?;

        Ok(MetalQ2MatvecReport {
            values,
            row_count,
            in_features,
            out_features,
            blocks_per_row: blocks_per_input_row,
            input_len: input.len(),
            weight_bytes: weights.len(),
            thread_count: output_len,
            transposed: true,
        })
    }

    pub(crate) fn run_transposed(
        &self,
        device: &Device,
        queue: &CommandQueue,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<MetalQ2MatvecReport> {
        let blocks_per_input_row = validate_q2_k_transposed_matvec_f32(
            weights,
            input,
            row_count,
            in_features,
            out_features,
        )?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K transposed matvec output length overflow"))?;

        let row_count_u32 = u32::try_from(row_count).map_err(|_| {
            Error::backend("Q2_K transposed matvec row_count exceeds Metal u32 limit")
        })?;
        let in_features_u32 = u32::try_from(in_features).map_err(|_| {
            Error::backend("Q2_K transposed matvec in_features exceeds Metal u32 limit")
        })?;
        let out_features_u32 = u32::try_from(out_features).map_err(|_| {
            Error::backend("Q2_K transposed matvec out_features exceeds Metal u32 limit")
        })?;
        let blocks_per_input_row_u32 = u32::try_from(blocks_per_input_row).map_err(|_| {
            Error::backend("Q2_K transposed matvec blocks_per_input_row exceeds Metal u32 limit")
        })?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Q2 scratch buffer lock poisoned"))?;
        let input_buffer = scratch.input_buffer(device, input)?;
        let output_buffer = scratch.output_buffer(device, output_len)?;
        let row_count_buffer = scratch.row_count_buffer(device, row_count_u32)?;
        let in_features_buffer = scratch.in_features_buffer(device, in_features_u32)?;
        let out_features_buffer = scratch.out_features_buffer(device, out_features_u32)?;
        let blocks_per_input_row_buffer =
            scratch.blocks_per_row_buffer(device, blocks_per_input_row_u32)?;

        trace!(
            target: "inferno::metal",
            row_count,
            in_features,
            out_features,
            blocks_per_input_row,
            weight_bytes = weights.len(),
            weight_zero_copy = true,
            input_buffer_reused = input_buffer.reused,
            output_buffer_reused = output_buffer.reused,
            "running native Metal transposed Q2_K matvec"
        );

        dispatch_1d(
            queue,
            &self.transposed_pipeline,
            &[
                &weight_buffer.buffer,
                &input_buffer.buffer,
                &output_buffer.buffer,
                &row_count_buffer.buffer,
                &in_features_buffer.buffer,
                &out_features_buffer.buffer,
                &blocks_per_input_row_buffer.buffer,
            ],
            output_len,
        )?;

        let values = read_f32_buffer(&output_buffer.buffer, output_len)?;

        Ok(MetalQ2MatvecReport {
            values,
            row_count,
            in_features,
            out_features,
            blocks_per_row: blocks_per_input_row,
            input_len: input.len(),
            weight_bytes: weights.len(),
            thread_count: output_len,
            transposed: true,
        })
    }

    /// Encodes one quantized matvec kernel into an open batched command buffer
    /// without dispatching it. The input already lives in a GPU buffer (it is
    /// usually the not-yet-computed output of a previously encoded kernel), so
    /// unlike the `run_*` entry points this performs shape-only validation and
    /// allocates a fresh output buffer instead of recycling pooled scratch —
    /// pooled buffers must not be overwritten while earlier encoded kernels
    /// still reference them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_matvec(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        kind: QuantMatvecKind,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        let blocks_per_row = match kind {
            QuantMatvecKind::Q2K => validate_q2_k_matvec_buffer(
                weights,
                input_len,
                row_count,
                in_features,
                out_features,
            )?,
            QuantMatvecKind::Q2KTransposed => validate_q2_k_transposed_matvec_buffer(
                weights,
                input_len,
                row_count,
                in_features,
                out_features,
            )?,
            QuantMatvecKind::Q80 => validate_q8_0_matvec_buffer(
                weights,
                input_len,
                row_count,
                in_features,
                out_features,
            )?,
            QuantMatvecKind::Q80Transposed => validate_q8_0_transposed_matvec_buffer(
                weights,
                input_len,
                row_count,
                in_features,
                out_features,
            )?,
        };
        require_f32_capacity(input, input_len, "quantized matvec input")?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("quantized matvec output length overflow"))?;

        let pipeline = match kind {
            QuantMatvecKind::Q2K => &self.pipeline,
            QuantMatvecKind::Q2KTransposed => &self.transposed_pipeline,
            QuantMatvecKind::Q80 => &self.q8_0_pipeline,
            QuantMatvecKind::Q80Transposed => &self.q8_0_transposed_pipeline,
        };
        let dispatch_threads = match kind {
            QuantMatvecKind::Q2K => {
                q2_k_cooperative_threads(pipeline, output_len, "batched Q2_K matvec")?
            }
            QuantMatvecKind::Q2KTransposed
            | QuantMatvecKind::Q80
            | QuantMatvecKind::Q80Transposed => output_len,
        };
        let weight_buffer = self.weight_buffer(device, weights)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let row_count_buffer = u32_scalar_buffer(device, matvec_u32(row_count, "row_count")?)?;
        let in_features_buffer =
            u32_scalar_buffer(device, matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer =
            u32_scalar_buffer(device, matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer =
            u32_scalar_buffer(device, matvec_u32(blocks_per_row, "blocks_per_row")?)?;

        trace!(
            target: "inferno::metal",
            ?kind,
            row_count,
            in_features,
            out_features,
            weight_zero_copy = true,
            "encoding batched quantized matvec"
        );

        encode_1d(
            command_buffer,
            pipeline,
            &[
                &weight_buffer.buffer,
                input,
                &output_buffer,
                &row_count_buffer,
                &in_features_buffer,
                &out_features_buffer,
                &blocks_per_row_buffer,
            ],
            dispatch_threads,
        )?;
        Ok(output_buffer)
    }

    /// Encodes a Q2_K matvec fused with a residual add into an open batched
    /// command buffer. See `encode_matvec` for the batching rules.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_matvec_add(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        residual: &Buffer,
        residual_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        let blocks_per_row =
            validate_q2_k_matvec_buffer(weights, input_len, row_count, in_features, out_features)?;
        require_f32_capacity(input, input_len, "Q2_K matvec add input")?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K matvec add output length overflow"))?;
        let physical_threads =
            q2_k_cooperative_threads(&self.add_pipeline, output_len, "batched Q2_K matvec add")?;
        if residual_len != output_len {
            return Err(Error::backend(format!(
                "Q2_K matvec add residual length mismatch: expected {output_len}, got {residual_len}"
            )));
        }
        require_f32_capacity(residual, residual_len, "Q2_K matvec add residual")?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let row_count_buffer = u32_scalar_buffer(device, matvec_u32(row_count, "row_count")?)?;
        let in_features_buffer =
            u32_scalar_buffer(device, matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer =
            u32_scalar_buffer(device, matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer =
            u32_scalar_buffer(device, matvec_u32(blocks_per_row, "blocks_per_row")?)?;

        trace!(
            target: "inferno::metal",
            row_count,
            in_features,
            out_features,
            weight_zero_copy = true,
            "encoding batched Q2_K matvec plus residual"
        );

        encode_1d(
            command_buffer,
            &self.add_pipeline,
            &[
                &weight_buffer.buffer,
                input,
                residual,
                &output_buffer,
                &row_count_buffer,
                &in_features_buffer,
                &out_features_buffer,
                &blocks_per_row_buffer,
            ],
            physical_threads,
        )?;
        Ok(output_buffer)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_multi_expert_gate_up_swiglu(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        gate_weights: &[u8],
        up_weights: &[u8],
        input: &Buffer,
        input_len: usize,
        token_indices: &[u32],
        expert_ids: &[u32],
        token_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        let assignment_count = validate_multi_expert_routing(
            token_indices,
            expert_ids,
            token_count,
            "Q2_K multi-expert gate/up SwiGLU",
        )?;
        let (gate_blocks, gate_experts, expert_stride_bytes) = validate_q2_k_packed_experts(
            gate_weights,
            in_features,
            out_features,
            "Q2_K multi-expert gate",
        )?;
        let (up_blocks, up_experts, up_expert_stride_bytes) = validate_q2_k_packed_experts(
            up_weights,
            in_features,
            out_features,
            "Q2_K multi-expert up",
        )?;
        if gate_blocks != up_blocks {
            return Err(Error::backend(format!(
                "Q2_K multi-expert gate/up block mismatch: gate has {gate_blocks}, up has {up_blocks}"
            )));
        }
        if gate_experts != up_experts || expert_stride_bytes != up_expert_stride_bytes {
            return Err(Error::backend(format!(
                "Q2_K multi-expert gate/up packed layout mismatch: gate experts={gate_experts} stride={expert_stride_bytes}, up experts={up_experts} stride={up_expert_stride_bytes}"
            )));
        }
        validate_expert_ids(expert_ids, gate_experts, "Q2_K multi-expert gate/up SwiGLU")?;

        let expected_input_len = token_count
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("Q2_K multi-expert gate/up input length overflow"))?;
        if input_len != expected_input_len {
            return Err(Error::backend(format!(
                "Q2_K multi-expert gate/up input length mismatch: expected {expected_input_len}, got {input_len}"
            )));
        }
        require_f32_capacity(input, input_len, "Q2_K multi-expert gate/up input")?;

        let output_len = assignment_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K multi-expert gate/up output length overflow"))?;
        let physical_threads = q2_k_cooperative_threads(
            &self.multi_expert_gate_up_swiglu_pipeline,
            output_len,
            "batched Q2_K multi-expert gate/up SwiGLU",
        )?;

        let gate_weight_buffer = self.weight_buffer(device, gate_weights)?;
        let up_weight_buffer = self.weight_buffer(device, up_weights)?;
        let token_indices_buffer = u32_buffer(device, token_indices)?;
        let expert_ids_buffer = u32_buffer(device, expert_ids)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let token_count_buffer =
            u32_scalar_buffer(device, matvec_u32(token_count, "token_count")?)?;
        let assignment_count_buffer =
            u32_scalar_buffer(device, matvec_u32(assignment_count, "assignment_count")?)?;
        let in_features_buffer =
            u32_scalar_buffer(device, matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer =
            u32_scalar_buffer(device, matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer =
            u32_scalar_buffer(device, matvec_u32(gate_blocks, "blocks_per_row")?)?;
        let expert_stride_buffer = u32_scalar_buffer(
            device,
            matvec_u32(expert_stride_bytes, "expert_stride_bytes")?,
        )?;

        trace!(
            target: "inferno::metal",
            token_count,
            assignment_count,
            in_features,
            out_features,
            expert_count = gate_experts,
            "encoding batched Q2_K multi-expert gate/up SwiGLU"
        );

        encode_1d(
            command_buffer,
            &self.multi_expert_gate_up_swiglu_pipeline,
            &[
                &gate_weight_buffer.buffer,
                &up_weight_buffer.buffer,
                input,
                &token_indices_buffer,
                &expert_ids_buffer,
                &output_buffer,
                &token_count_buffer,
                &assignment_count_buffer,
                &in_features_buffer,
                &out_features_buffer,
                &blocks_per_row_buffer,
                &expert_stride_buffer,
            ],
            physical_threads,
        )?;
        Ok(output_buffer)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_multi_expert_matvec(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        expert_ids: &[u32],
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        if expert_ids.is_empty() {
            return Err(Error::backend(
                "Q2_K multi-expert matvec requires at least one assignment",
            ));
        }
        let assignment_count = expert_ids.len();
        let (blocks_per_row, expert_count, expert_stride_bytes) = validate_q2_k_packed_experts(
            weights,
            in_features,
            out_features,
            "Q2_K multi-expert matvec",
        )?;
        validate_expert_ids(expert_ids, expert_count, "Q2_K multi-expert matvec")?;
        let expected_input_len = assignment_count
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("Q2_K multi-expert matvec input length overflow"))?;
        if input_len != expected_input_len {
            return Err(Error::backend(format!(
                "Q2_K multi-expert matvec input length mismatch: expected {expected_input_len}, got {input_len}"
            )));
        }
        require_f32_capacity(input, input_len, "Q2_K multi-expert matvec input")?;

        let output_len = assignment_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K multi-expert matvec output length overflow"))?;
        let physical_threads = q2_k_cooperative_threads(
            &self.multi_expert_matvec_pipeline,
            output_len,
            "batched Q2_K multi-expert matvec",
        )?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let expert_ids_buffer = u32_buffer(device, expert_ids)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let assignment_count_buffer =
            u32_scalar_buffer(device, matvec_u32(assignment_count, "assignment_count")?)?;
        let in_features_buffer =
            u32_scalar_buffer(device, matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer =
            u32_scalar_buffer(device, matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer =
            u32_scalar_buffer(device, matvec_u32(blocks_per_row, "blocks_per_row")?)?;
        let expert_stride_buffer = u32_scalar_buffer(
            device,
            matvec_u32(expert_stride_bytes, "expert_stride_bytes")?,
        )?;

        trace!(
            target: "inferno::metal",
            assignment_count,
            in_features,
            out_features,
            expert_count,
            "encoding batched Q2_K multi-expert matvec"
        );

        encode_1d(
            command_buffer,
            &self.multi_expert_matvec_pipeline,
            &[
                &weight_buffer.buffer,
                input,
                &expert_ids_buffer,
                &output_buffer,
                &assignment_count_buffer,
                &in_features_buffer,
                &out_features_buffer,
                &blocks_per_row_buffer,
                &expert_stride_buffer,
            ],
            physical_threads,
        )?;
        Ok(output_buffer)
    }

    /// Encodes the Q2_K output-head matvec plus greedy argmax into an open
    /// batched command buffer. Returns the one-element token id (u32) and token
    /// score (f32) buffers; they hold valid data only after the batch flushes.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_argmax(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<(Buffer, Buffer)> {
        if row_count != 1 {
            return Err(Error::backend(format!(
                "Q2_K greedy argmax requires row_count 1, got {row_count}"
            )));
        }
        let blocks_per_row =
            validate_q2_k_matvec_buffer(weights, input_len, row_count, in_features, out_features)?;
        require_f32_capacity(input, input_len, "Q2_K argmax input")?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K argmax matvec output length overflow"))?;
        let matvec_threads =
            q2_k_cooperative_threads(&self.pipeline, output_len, "batched Q2_K argmax matvec")?;
        let argmax_threads = argmax_threads(&self.argmax_pipeline, "batched Q2_K greedy argmax")?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let logits_buffer = empty_f32_buffer(device, output_len)?;
        let token_id_buffer = empty_u32_buffer(device, 1)?;
        let token_score_buffer = empty_f32_buffer(device, 1)?;
        let row_count_buffer = u32_scalar_buffer(device, matvec_u32(row_count, "row_count")?)?;
        let in_features_buffer =
            u32_scalar_buffer(device, matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer =
            u32_scalar_buffer(device, matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer =
            u32_scalar_buffer(device, matvec_u32(blocks_per_row, "blocks_per_row")?)?;
        let output_len_buffer = u32_scalar_buffer(device, matvec_u32(output_len, "output_len")?)?;

        trace!(
            target: "inferno::metal",
            in_features,
            out_features,
            weight_zero_copy = true,
            "encoding batched Q2_K matvec plus greedy argmax"
        );

        encode_1d(
            command_buffer,
            &self.pipeline,
            &[
                &weight_buffer.buffer,
                input,
                &logits_buffer,
                &row_count_buffer,
                &in_features_buffer,
                &out_features_buffer,
                &blocks_per_row_buffer,
            ],
            matvec_threads,
        )?;
        encode_1d(
            command_buffer,
            &self.argmax_pipeline,
            &[
                &logits_buffer,
                &token_id_buffer,
                &token_score_buffer,
                &output_len_buffer,
            ],
            argmax_threads,
        )?;
        Ok((token_id_buffer, token_score_buffer))
    }

    fn weight_buffer(&self, device: &Device, weights: &[u8]) -> Result<WeightBuffer> {
        Ok(WeightBuffer {
            buffer: u8_buffer_no_copy(device, weights)?,
        })
    }
}

fn matvec_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::backend(format!("quantized matvec {label} exceeds Metal u32 limit")))
}

fn validate_q2_k_packed_experts(
    weights: &[u8],
    in_features: usize,
    out_features: usize,
    context: &str,
) -> Result<(usize, usize, usize)> {
    if in_features == 0 || out_features == 0 {
        return Err(Error::backend(format!(
            "{context} requires non-zero in_features and out_features"
        )));
    }
    if in_features % Q2_K_BLOCK_VALUES != 0 {
        return Err(Error::backend(format!(
            "{context} in_features {in_features} must be divisible by {Q2_K_BLOCK_VALUES}"
        )));
    }
    let blocks_per_row = in_features / Q2_K_BLOCK_VALUES;
    let expert_stride_bytes = out_features
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(Q2_K_BLOCK_BYTES))
        .ok_or_else(|| Error::backend(format!("{context} expert byte stride overflow")))?;
    if expert_stride_bytes == 0 {
        return Err(Error::backend(format!(
            "{context} expert byte stride must be non-zero"
        )));
    }
    if weights.len() % expert_stride_bytes != 0 {
        return Err(Error::backend(format!(
            "{context} packed weight byte length {} is not divisible by expert stride {expert_stride_bytes}",
            weights.len()
        )));
    }
    let expert_count = weights.len() / expert_stride_bytes;
    if expert_count == 0 {
        return Err(Error::backend(format!(
            "{context} requires at least one packed expert"
        )));
    }
    matvec_u32(expert_stride_bytes, "expert_stride_bytes")?;
    Ok((blocks_per_row, expert_count, expert_stride_bytes))
}

fn validate_multi_expert_routing(
    token_indices: &[u32],
    expert_ids: &[u32],
    token_count: usize,
    context: &str,
) -> Result<usize> {
    if token_count == 0 {
        return Err(Error::backend(format!(
            "{context} requires non-zero token_count"
        )));
    }
    if token_indices.is_empty() {
        return Err(Error::backend(format!(
            "{context} requires at least one routed assignment"
        )));
    }
    if token_indices.len() != expert_ids.len() {
        return Err(Error::backend(format!(
            "{context} routing length mismatch: {} token indices and {} expert ids",
            token_indices.len(),
            expert_ids.len()
        )));
    }
    for token_index in token_indices {
        if *token_index as usize >= token_count {
            return Err(Error::backend(format!(
                "{context} token index {token_index} is outside token_count {token_count}"
            )));
        }
    }
    Ok(token_indices.len())
}

fn validate_expert_ids(expert_ids: &[u32], expert_count: usize, context: &str) -> Result<()> {
    for expert_id in expert_ids {
        if *expert_id as usize >= expert_count {
            return Err(Error::backend(format!(
                "{context} expert id {expert_id} is outside expert_count {expert_count}"
            )));
        }
    }
    Ok(())
}

fn q2_k_cooperative_threads(
    pipeline: &ComputePipelineState,
    output_len: usize,
    context: &str,
) -> Result<usize> {
    let thread_execution_width = pipeline.thread_execution_width() as usize;
    if thread_execution_width != Q2_K_SIMD_LANES {
        return Err(Error::backend(format!(
            "{context} requires {Q2_K_SIMD_LANES}-lane Apple Metal SIMD groups, got {thread_execution_width}"
        )));
    }
    output_len
        .checked_mul(Q2_K_SIMD_LANES)
        .ok_or_else(|| Error::backend(format!("{context} physical thread count overflow")))
}

fn argmax_threads(pipeline: &ComputePipelineState, context: &str) -> Result<usize> {
    let thread_execution_width = pipeline.thread_execution_width() as usize;
    let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
    if thread_execution_width == 0
        || thread_execution_width > ARGMAX_THREADS_PER_VECTOR
        || ARGMAX_THREADS_PER_VECTOR % thread_execution_width != 0
    {
        return Err(Error::backend(format!(
            "{context} requires a thread execution width that divides {ARGMAX_THREADS_PER_VECTOR}, got {thread_execution_width}"
        )));
    }
    if max_threads < ARGMAX_THREADS_PER_VECTOR {
        return Err(Error::backend(format!(
            "{context} requires {ARGMAX_THREADS_PER_VECTOR} threads per threadgroup, pipeline allows {max_threads}"
        )));
    }
    Ok(ARGMAX_THREADS_PER_VECTOR)
}

impl Q2ScratchBuffers {
    fn input_buffer(&mut self, device: &Device, values: &[f32]) -> Result<ScratchBufferRef> {
        let scratch = scratch_f32_buffer(device, &mut self.input, values.len())?;
        write_f32_buffer(&scratch.buffer, values)?;
        Ok(scratch)
    }

    fn residual_buffer(&mut self, device: &Device, values: &[f32]) -> Result<ScratchBufferRef> {
        let scratch = scratch_f32_buffer(device, &mut self.residual, values.len())?;
        write_f32_buffer(&scratch.buffer, values)?;
        Ok(scratch)
    }

    fn output_buffer(&mut self, device: &Device, len: usize) -> Result<ScratchBufferRef> {
        scratch_f32_buffer(device, &mut self.output, len)
    }

    fn logits_buffer(&mut self, device: &Device, len: usize) -> Result<ScratchBufferRef> {
        scratch_f32_buffer(device, &mut self.logits, len)
    }

    fn token_id_buffer(&mut self, device: &Device) -> Result<ScratchBufferRef> {
        scratch_u32_buffer(device, &mut self.token_id, 1)
    }

    fn token_score_buffer(&mut self, device: &Device) -> Result<ScratchBufferRef> {
        scratch_f32_buffer(device, &mut self.token_score, 1)
    }

    fn row_count_buffer(&mut self, device: &Device, value: u32) -> Result<ScratchBufferRef> {
        scratch_u32_value_buffer(device, &mut self.row_count, value)
    }

    fn in_features_buffer(&mut self, device: &Device, value: u32) -> Result<ScratchBufferRef> {
        scratch_u32_value_buffer(device, &mut self.in_features, value)
    }

    fn out_features_buffer(&mut self, device: &Device, value: u32) -> Result<ScratchBufferRef> {
        scratch_u32_value_buffer(device, &mut self.out_features, value)
    }

    fn blocks_per_row_buffer(&mut self, device: &Device, value: u32) -> Result<ScratchBufferRef> {
        scratch_u32_value_buffer(device, &mut self.blocks_per_row, value)
    }

    fn output_len_buffer(&mut self, device: &Device, value: u32) -> Result<ScratchBufferRef> {
        scratch_u32_value_buffer(device, &mut self.output_len, value)
    }
}

fn scratch_f32_buffer(
    device: &Device,
    slot: &mut Option<ScratchBuffer>,
    len: usize,
) -> Result<ScratchBufferRef> {
    scratch_buffer_with(device, slot, len, empty_f32_buffer)
}

fn scratch_u32_buffer(
    device: &Device,
    slot: &mut Option<ScratchBuffer>,
    len: usize,
) -> Result<ScratchBufferRef> {
    scratch_buffer_with(device, slot, len, empty_u32_buffer)
}

fn scratch_u32_value_buffer(
    device: &Device,
    slot: &mut Option<ScratchBuffer>,
    value: u32,
) -> Result<ScratchBufferRef> {
    let scratch = scratch_u32_buffer(device, slot, 1)?;
    write_u32_buffer(&scratch.buffer, &[value])?;
    Ok(scratch)
}

fn scratch_buffer_with(
    device: &Device,
    slot: &mut Option<ScratchBuffer>,
    len: usize,
    allocate: fn(&Device, usize) -> Result<Buffer>,
) -> Result<ScratchBufferRef> {
    if len == 0 {
        return Err(Error::backend("Q2 scratch buffer length must be positive"));
    }
    if let Some(existing) = slot.as_ref() {
        if existing.len >= len {
            return Ok(ScratchBufferRef {
                buffer: existing.buffer.clone(),
                reused: true,
            });
        }
    }

    let buffer = allocate(device, len)?;
    *slot = Some(ScratchBuffer {
        buffer: buffer.clone(),
        len,
    });
    Ok(ScratchBufferRef {
        buffer,
        reused: false,
    })
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use crate::metal::Metal;

    use super::super::validation::{Q2_K_BLOCK_BYTES, Q2_K_BLOCK_VALUES};

    #[test]
    fn matches_cpu_reference_for_q2_k_matvec() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let weights = [
            q2_k_block(0x3c00, 0x0000, 0x01, 0xe4),
            q2_k_block(0x4000, 0x0000, 0x01, 0x1b),
        ]
        .concat();
        let input = (0..Q2_K_BLOCK_VALUES)
            .map(|index| (index as f32 % 17.0) * 0.25)
            .collect::<Vec<_>>();

        let report = metal
            .q2_k_matvec_f32_report(&weights, &input, 1, Q2_K_BLOCK_VALUES, 2)
            .unwrap();
        let expected = cpu_q2_k_matvec(&weights, &input, 1, Q2_K_BLOCK_VALUES, 2);

        assert_eq!(report.row_count, 1);
        assert_eq!(report.in_features, Q2_K_BLOCK_VALUES);
        assert_eq!(report.out_features, 2);
        assert_eq!(report.blocks_per_row, 1);
        assert_eq!(report.weight_bytes, Q2_K_BLOCK_BYTES * 2);
        assert_close(&report.values, &expected, 1e-4);
    }

    #[test]
    fn q2_k_matvec_argmax_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let weights = [
            q2_k_block(0x3c00, 0x0000, 0x01, 0x1b),
            q2_k_block(0x4000, 0x0000, 0x01, 0xe4),
            q2_k_block(0x3c00, 0x0000, 0x01, 0x00),
        ]
        .concat();
        let input = (0..Q2_K_BLOCK_VALUES)
            .map(|index| (index as f32 % 11.0) * 0.125)
            .collect::<Vec<_>>();

        let report = metal
            .q2_k_matvec_argmax_f32_report(&weights, &input, 1, Q2_K_BLOCK_VALUES, 3)
            .unwrap();
        let scores = cpu_q2_k_matvec(&weights, &input, 1, Q2_K_BLOCK_VALUES, 3);
        let (expected_id, expected_score) = cpu_argmax(&scores);

        assert_eq!(report.row_count, 1);
        assert_eq!(report.out_features, 3);
        assert_eq!(report.token_id, expected_id);
        assert!((report.token_score - expected_score).abs() <= 1e-4);
    }

    #[test]
    fn q2_k_rms_norm_argmax_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let weights = [
            q2_k_block(0x3c00, 0x0000, 0x01, 0x1b),
            q2_k_block(0x4000, 0x0000, 0x01, 0xe4),
            q2_k_block(0x3c00, 0x0000, 0x01, 0x00),
        ]
        .concat();
        let input = (0..Q2_K_BLOCK_VALUES)
            .map(|index| (index as f32 % 11.0) * 0.125)
            .collect::<Vec<_>>();
        let rms_weight = (0..Q2_K_BLOCK_VALUES)
            .map(|index| 1.0 + (index as f32 % 7.0) * 0.01)
            .collect::<Vec<_>>();
        let eps = 1e-5;

        let (token_id, token_score) = metal
            .q2_k_rms_norm_argmax_f32(&weights, &input, &rms_weight, 1, Q2_K_BLOCK_VALUES, 3, eps)
            .unwrap();
        let normed = cpu_rms_norm(&input, &rms_weight, 1, Q2_K_BLOCK_VALUES, eps);
        let scores = cpu_q2_k_matvec(&weights, &normed, 1, Q2_K_BLOCK_VALUES, 3);
        let (expected_id, expected_score) = cpu_argmax(&scores);

        assert_eq!(token_id, expected_id);
        assert!((token_score - expected_score).abs() <= 1e-4);
    }

    #[test]
    fn q2_k_matvec_add_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let weights = [
            q2_k_block(0x3c00, 0x0000, 0x01, 0xe4),
            q2_k_block(0x4000, 0x0000, 0x01, 0x1b),
        ]
        .concat();
        let input = (0..Q2_K_BLOCK_VALUES)
            .map(|index| (index as f32 % 17.0) * 0.25)
            .collect::<Vec<_>>();
        let residual = vec![0.25_f32, -0.5];

        let report = metal
            .q2_k_matvec_add_f32_report(&weights, &input, &residual, 1, Q2_K_BLOCK_VALUES, 2)
            .unwrap();
        let mut expected = cpu_q2_k_matvec(&weights, &input, 1, Q2_K_BLOCK_VALUES, 2);
        for (value, residual) in expected.iter_mut().zip(&residual) {
            *value += residual;
        }

        assert_eq!(report.row_count, 1);
        assert_eq!(report.in_features, Q2_K_BLOCK_VALUES);
        assert_eq!(report.out_features, 2);
        assert_eq!(report.residual_len, 2);
        assert_close(&report.values, &expected, 1e-4);
    }

    #[test]
    fn gate_up_swiglu_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let gate_weights = [
            q2_k_block(0x3c00, 0x0000, 0x01, 0xe4),
            q2_k_block(0x4000, 0x0000, 0x01, 0x1b),
        ]
        .concat();
        let up_weights = [
            q2_k_block(0x4000, 0x0000, 0x01, 0xe4),
            q2_k_block(0x3c00, 0x0000, 0x01, 0x1b),
        ]
        .concat();
        let input = (0..Q2_K_BLOCK_VALUES)
            .map(|index| (index as f32 % 13.0) * 0.1 - 0.35)
            .collect::<Vec<_>>();

        let report = metal
            .q2_k_gate_up_swiglu_f32_report(
                &gate_weights,
                &up_weights,
                &input,
                1,
                Q2_K_BLOCK_VALUES,
                2,
            )
            .unwrap();
        let gate = cpu_q2_k_matvec(&gate_weights, &input, 1, Q2_K_BLOCK_VALUES, 2);
        let up = cpu_q2_k_matvec(&up_weights, &input, 1, Q2_K_BLOCK_VALUES, 2);
        let expected = cpu_swiglu(&gate, &up);

        assert_eq!(report.row_count, 1);
        assert_eq!(report.in_features, Q2_K_BLOCK_VALUES);
        assert_eq!(report.out_features, 2);
        assert_eq!(report.blocks_per_row, 1);
        assert_eq!(report.gate_weight_bytes, Q2_K_BLOCK_BYTES * 2);
        assert_eq!(report.up_weight_bytes, Q2_K_BLOCK_BYTES * 2);
        assert_close(&report.values, &expected, 1e-4);
    }

    #[test]
    fn multi_expert_gate_up_and_matvec_match_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let hidden_features = 2;
        let intermediate_features = Q2_K_BLOCK_VALUES;
        let expert_count = 2;
        let gate_expert_stride = intermediate_features * Q2_K_BLOCK_BYTES;
        let down_expert_stride = hidden_features * Q2_K_BLOCK_BYTES;
        let mut gate_weights = Vec::new();
        let mut up_weights = Vec::new();
        for expert in 0..expert_count {
            for row in 0..intermediate_features {
                let quant = if (expert + row) % 2 == 0 { 0xe4 } else { 0x1b };
                gate_weights.extend(q2_k_block(
                    if expert == 0 { 0x3c00 } else { 0x4000 },
                    0x0000,
                    0x01 + expert as u8,
                    quant,
                ));
                up_weights.extend(q2_k_block(
                    if expert == 0 { 0x4000 } else { 0x3c00 },
                    0x0000,
                    0x02 + expert as u8,
                    quant ^ 0xff,
                ));
            }
        }
        let mut down_weights = Vec::new();
        for expert in 0..expert_count {
            for row in 0..hidden_features {
                down_weights.extend(q2_k_block(
                    if expert == 0 { 0x3c00 } else { 0x4000 },
                    0x0000,
                    0x03 + row as u8,
                    if expert == row { 0xe4 } else { 0x1b },
                ));
            }
        }
        let token_count = 2;
        let input = (0..token_count * Q2_K_BLOCK_VALUES)
            .map(|index| (index as f32 % 19.0) * 0.05 - 0.20)
            .collect::<Vec<_>>();
        let token_indices = vec![1_u32, 0, 1];
        let expert_ids = vec![1_u32, 0, 1];

        let input_buffer = metal.batch_upload_f32(&input).unwrap();
        let gated_buffer = metal
            .batched_q2_k_multi_expert_gate_up_swiglu(
                &gate_weights,
                &up_weights,
                &input_buffer,
                input.len(),
                &token_indices,
                &expert_ids,
                token_count,
                Q2_K_BLOCK_VALUES,
                intermediate_features,
            )
            .unwrap();
        let down_buffer = metal
            .batched_q2_k_multi_expert_matvec(
                &down_weights,
                &gated_buffer,
                token_indices.len() * intermediate_features,
                &expert_ids,
                intermediate_features,
                hidden_features,
            )
            .unwrap();
        let gated = metal
            .batch_read_f32(&gated_buffer, token_indices.len() * intermediate_features)
            .unwrap();
        let down = metal
            .batch_read_f32(&down_buffer, token_indices.len() * hidden_features)
            .unwrap();

        let mut expected_gated = Vec::new();
        for (&token, &expert) in token_indices.iter().zip(&expert_ids) {
            let token = token as usize;
            let expert = expert as usize;
            let token_input = &input[token * Q2_K_BLOCK_VALUES..(token + 1) * Q2_K_BLOCK_VALUES];
            let expert_start = expert * gate_expert_stride;
            let expert_end = expert_start + gate_expert_stride;
            let gate = cpu_q2_k_matvec(
                &gate_weights[expert_start..expert_end],
                token_input,
                1,
                Q2_K_BLOCK_VALUES,
                intermediate_features,
            );
            let up = cpu_q2_k_matvec(
                &up_weights[expert_start..expert_end],
                token_input,
                1,
                Q2_K_BLOCK_VALUES,
                intermediate_features,
            );
            expected_gated.extend(cpu_swiglu(&gate, &up));
        }
        let mut expected_down = Vec::new();
        for (assignment, expert) in expert_ids.iter().copied().enumerate() {
            let expert = expert as usize;
            let expert_start = expert * down_expert_stride;
            let expert_end = expert_start + down_expert_stride;
            expected_down.extend(cpu_q2_k_matvec(
                &down_weights[expert_start..expert_end],
                &expected_gated
                    [assignment * intermediate_features..(assignment + 1) * intermediate_features],
                1,
                intermediate_features,
                hidden_features,
            ));
        }

        assert_close(&gated, &expected_gated, 1e-4);
        assert_close(&down, &expected_down, 1e-4);
    }

    #[test]
    fn rejects_bad_q2_k_shape_before_dispatch() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let err = metal
            .q2_k_matvec_f32(
                &[0_u8; 16],
                &[0.0_f32; Q2_K_BLOCK_VALUES],
                1,
                Q2_K_BLOCK_VALUES,
                1,
            )
            .expect_err("bad Q2 byte count should fail before Metal dispatch");

        assert!(err.to_string().contains("weight byte mismatch"));
    }

    #[test]
    fn transposed_matches_cpu_reference_for_q2_k_matvec() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let in_features = 2;
        let out_features = Q2_K_BLOCK_VALUES;
        let weights = [
            q2_k_block(0x3c00, 0x0000, 0x01, 0xe4),
            q2_k_block(0x4000, 0x0000, 0x01, 0x1b),
        ]
        .concat();
        let input = vec![1.25_f32, -0.5];

        let report = metal
            .q2_k_transposed_matvec_f32_report(&weights, &input, 1, in_features, out_features)
            .unwrap();
        let expected = cpu_q2_k_transposed_matvec(&weights, &input, 1, in_features, out_features);

        assert_eq!(report.row_count, 1);
        assert_eq!(report.in_features, in_features);
        assert_eq!(report.out_features, out_features);
        assert_eq!(report.blocks_per_row, 1);
        assert!(report.transposed);
        assert_close(&report.values, &expected, 1e-4);
    }

    fn native_metal_or_skip() -> Option<Metal> {
        Metal::new().ok()
    }

    fn q2_k_block(d: u16, dmin: u16, scale_min: u8, quant: u8) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q2_K_BLOCK_BYTES);
        block.extend(std::iter::repeat_n(scale_min, 16));
        block.extend(std::iter::repeat_n(quant, 64));
        block.extend(d.to_le_bytes());
        block.extend(dmin.to_le_bytes());
        block
    }

    fn cpu_q2_k_matvec(
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Vec<f32> {
        let blocks_per_row = in_features / Q2_K_BLOCK_VALUES;
        let mut output = vec![0.0; row_count * out_features];

        for input_row in 0..row_count {
            for output_feature in 0..out_features {
                let mut sum = 0.0;
                for block_in_row in 0..blocks_per_row {
                    let block_index = output_feature * blocks_per_row + block_in_row;
                    let block_offset = block_index * Q2_K_BLOCK_BYTES;
                    let block = &weights[block_offset..block_offset + Q2_K_BLOCK_BYTES];
                    sum += cpu_q2_k_block_dot(
                        block,
                        &input[input_row * in_features + block_in_row * Q2_K_BLOCK_VALUES..],
                    );
                }
                output[input_row * out_features + output_feature] = sum;
            }
        }

        output
    }

    fn cpu_q2_k_block_dot(block: &[u8], input: &[f32]) -> f32 {
        let scales = &block[..16];
        let quants = &block[16..80];
        let d = f16_fixture_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let min = f16_fixture_to_f32(u16::from_le_bytes([block[82], block[83]]));
        let mut scale_index = 0;
        let mut quant_offset = 0;
        let mut input_offset = 0;
        let mut sum = 0.0;

        while input_offset < Q2_K_BLOCK_VALUES {
            let mut shift = 0;
            for _ in 0..4 {
                let scale_min = scales[scale_index];
                scale_index += 1;
                let scale = d * (scale_min & 0x0f) as f32;
                let min_offset = min * (scale_min >> 4) as f32;
                for value_index in 0..16 {
                    let weight = scale
                        * ((quants[quant_offset + value_index] >> shift) & 0x03) as f32
                        - min_offset;
                    sum += input[input_offset + value_index] * weight;
                }
                input_offset += 16;

                let scale_min = scales[scale_index];
                scale_index += 1;
                let scale = d * (scale_min & 0x0f) as f32;
                let min_offset = min * (scale_min >> 4) as f32;
                for value_index in 0..16 {
                    let weight = scale
                        * ((quants[quant_offset + 16 + value_index] >> shift) & 0x03) as f32
                        - min_offset;
                    sum += input[input_offset + value_index] * weight;
                }
                input_offset += 16;

                shift += 2;
            }
            quant_offset += 32;
        }

        sum
    }

    fn cpu_q2_k_transposed_matvec(
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Vec<f32> {
        let blocks_per_input_row = out_features / Q2_K_BLOCK_VALUES;
        let mut output = vec![0.0; row_count * out_features];

        for row in 0..row_count {
            for output_feature in 0..out_features {
                let block_in_input_row = output_feature / Q2_K_BLOCK_VALUES;
                let value_in_block = output_feature % Q2_K_BLOCK_VALUES;
                let mut sum = 0.0;
                for input_feature in 0..in_features {
                    let block_index = input_feature * blocks_per_input_row + block_in_input_row;
                    let block_offset = block_index * Q2_K_BLOCK_BYTES;
                    let block = &weights[block_offset..block_offset + Q2_K_BLOCK_BYTES];
                    sum += input[row * in_features + input_feature]
                        * cpu_q2_k_block_value(block, value_in_block);
                }
                output[row * out_features + output_feature] = sum;
            }
        }

        output
    }

    fn cpu_argmax(scores: &[f32]) -> (u32, f32) {
        let mut best_id = 0_u32;
        let mut best_score = scores[0];
        for (index, score) in scores.iter().copied().enumerate().skip(1) {
            if score > best_score {
                best_id = index as u32;
                best_score = score;
            }
        }
        (best_id, best_score)
    }

    fn cpu_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .copied()
            .zip(up.iter().copied())
            .map(|(gate, up)| {
                let silu = gate / (1.0 + (-gate).exp());
                silu * up
            })
            .collect()
    }

    fn cpu_rms_norm(
        input: &[f32],
        weight: &[f32],
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        for row in 0..rows {
            let base = row * hidden_size;
            let mut sumsq = 0.0;
            for col in 0..hidden_size {
                let value = input[base + col];
                sumsq += value * value;
            }
            let scale = ((sumsq / hidden_size as f32) + eps).sqrt().recip();
            for col in 0..hidden_size {
                output[base + col] = input[base + col] * scale * weight[col];
            }
        }
        output
    }

    fn cpu_q2_k_block_value(block: &[u8], value_index: usize) -> f32 {
        let half = value_index / 128;
        let within_half = value_index % 128;
        let pair = within_half / 32;
        let within_pair = within_half % 32;
        let upper_half_of_pair = within_pair >= 16;
        let scale_index = half * 8 + pair * 2 + usize::from(upper_half_of_pair);
        let quant_index =
            16 + half * 32 + if upper_half_of_pair { 16 } else { 0 } + (within_pair % 16);
        let shift = pair * 2;
        let d = f16_fixture_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let min = f16_fixture_to_f32(u16::from_le_bytes([block[82], block[83]]));
        let scale_min = block[scale_index];
        let scale = d * (scale_min & 0x0f) as f32;
        let min_offset = min * (scale_min >> 4) as f32;
        let quant = ((block[quant_index] >> shift) & 0x03) as f32;

        scale * quant - min_offset
    }

    fn f16_fixture_to_f32(bits: u16) -> f32 {
        match bits {
            0x0000 => 0.0,
            0x3c00 => 1.0,
            0x4000 => 2.0,
            other => panic!("unsupported test f16 bits {other:#06x}"),
        }
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
