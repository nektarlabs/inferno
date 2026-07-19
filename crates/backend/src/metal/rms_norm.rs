use ::metal::{Buffer, CommandBufferRef, CommandQueue, ComputePipelineState, Device};
use common::{Error, Result};
use tracing::trace;

use super::{
    arena::MetalArena,
    buffers::{f32_buffer, read_f32_buffer, require_f32_capacity, ImmutableF32BufferCache},
    command::{dispatch_1d, encode_1d},
    library::MetalLibrary,
    pipeline::compute_pipeline,
    validation::{validate_rms_norm_buffer, validate_rms_norm_f32},
};

const RMS_NORM_KERNEL: &str = "rms_norm_f32_kernel";
const MLA_KV_POSTPROCESS_KERNEL: &str = "mla_kv_postprocess_f32_kernel";
const RMS_NORM_THREADS_PER_ROW: usize = 256;
const RMS_NORM_SIMD_LANES: usize = 32;

pub(crate) struct MetalRmsNorm {
    pipeline: ComputePipelineState,
    mla_kv_postprocess_pipeline: ComputePipelineState,
    weight_buffers: ImmutableF32BufferCache,
    arena: MetalArena,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalRmsNormReport {
    pub values: Vec<f32>,
    pub rows: usize,
    pub hidden_size: usize,
    pub input_len: usize,
    pub thread_count: usize,
}

#[derive(Debug)]
pub(crate) struct MetalRmsNormBufferReport {
    pub buffer: Buffer,
}

pub(crate) struct MetalRmsNormPrepared {
    pub(crate) input_buffer: Buffer,
    pub(crate) weight_buffer: Buffer,
    pub(crate) output_buffer: Buffer,
    pub(crate) rows_buffer: Buffer,
    pub(crate) hidden_size_buffer: Buffer,
    pub(crate) eps_buffer: Buffer,
    pub(crate) input_len: usize,
}

impl MetalRmsNorm {
    pub(crate) fn new(device: &Device, library: &MetalLibrary, arena: MetalArena) -> Result<Self> {
        Ok(Self {
            pipeline: compute_pipeline(device, library, RMS_NORM_KERNEL)?,
            mla_kv_postprocess_pipeline: compute_pipeline(
                device,
                library,
                MLA_KV_POSTPROCESS_KERNEL,
            )?,
            weight_buffers: ImmutableF32BufferCache::default(),
            arena,
        })
    }

    pub(crate) fn run(
        &self,
        device: &Device,
        queue: &CommandQueue,
        input: &[f32],
        weight: &[f32],
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<MetalRmsNormReport> {
        let output = self.run_to_buffer(device, queue, input, weight, rows, hidden_size, eps)?;
        let values = read_f32_buffer(&output.buffer, input.len())?;
        let physical_threads = rms_norm_threads(&self.pipeline, rows)?;

        Ok(MetalRmsNormReport {
            values,
            rows,
            hidden_size,
            input_len: input.len(),
            thread_count: physical_threads,
        })
    }

    pub(crate) fn run_to_buffer(
        &self,
        device: &Device,
        queue: &CommandQueue,
        input: &[f32],
        weight: &[f32],
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<MetalRmsNormBufferReport> {
        let prepared = self.prepare_to_buffer(device, input, weight, rows, hidden_size, eps)?;
        let physical_threads = rms_norm_threads(&self.pipeline, rows)?;

        trace!(
            target: "inferno::metal",
            rows,
            hidden_size,
            input_len = input.len(),
            physical_threads,
            "running native Metal RMSNorm"
        );

        dispatch_1d(
            queue,
            &self.pipeline,
            &[
                &prepared.input_buffer,
                &prepared.weight_buffer,
                &prepared.output_buffer,
                &prepared.rows_buffer,
                &prepared.hidden_size_buffer,
                &prepared.eps_buffer,
            ],
            physical_threads,
        )?;

        Ok(MetalRmsNormBufferReport {
            buffer: prepared.output_buffer,
        })
    }

    pub(crate) fn prepare_to_buffer(
        &self,
        device: &Device,
        input: &[f32],
        weight: &[f32],
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<MetalRmsNormPrepared> {
        validate_rms_norm_f32(input, weight, rows, hidden_size, eps)?;

        let rows_u32 = u32::try_from(rows)
            .map_err(|_| Error::backend("RMSNorm rows exceed Metal u32 limit"))?;
        let hidden_size_u32 = u32::try_from(hidden_size)
            .map_err(|_| Error::backend("RMSNorm hidden_size exceeds Metal u32 limit"))?;

        Ok(MetalRmsNormPrepared {
            input_buffer: f32_buffer(device, input)?,
            weight_buffer: f32_buffer(device, weight)?,
            output_buffer: self.arena.empty_f32(input.len())?,
            rows_buffer: self.arena.u32(rows_u32)?,
            hidden_size_buffer: self.arena.u32(hidden_size_u32)?,
            eps_buffer: self.arena.f32(eps)?,
            input_len: input.len(),
        })
    }

    /// Encodes an RMSNorm kernel into an open batched command buffer, reading
    /// its input from a device-resident buffer and returning a fresh output
    /// buffer. See `BatchSlot` for the batching rules.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        input: &Buffer,
        input_len: usize,
        weight: &[f32],
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<Buffer> {
        validate_rms_norm_buffer(input_len, weight, rows, hidden_size, eps)?;
        require_f32_capacity(input, input_len, "RMSNorm input")?;

        let rows_u32 = u32::try_from(rows)
            .map_err(|_| Error::backend("RMSNorm rows exceed Metal u32 limit"))?;
        let hidden_size_u32 = u32::try_from(hidden_size)
            .map_err(|_| Error::backend("RMSNorm hidden_size exceeds Metal u32 limit"))?;

        // Device batches only receive model-owned immutable weights. Keep one
        // no-copy Metal view per slice instead of copying 24 KiB for every
        // norm in every token. The eager reference path still copies its
        // potentially short-lived test inputs.
        let weight_buffer = self.weight_buffers.get(device, weight)?;
        let output_buffer = self.arena.empty_f32(input_len)?;
        let rows_buffer = self.arena.u32(rows_u32)?;
        let hidden_size_buffer = self.arena.u32(hidden_size_u32)?;
        let eps_buffer = self.arena.f32(eps)?;
        let physical_threads = rms_norm_threads(&self.pipeline, rows)?;

        trace!(
            target: "inferno::metal",
            rows,
            hidden_size,
            input_len,
            physical_threads,
            "encoding batched RMSNorm"
        );

        encode_1d(
            command_buffer,
            &self.pipeline,
            &[
                input,
                &weight_buffer,
                &output_buffer,
                &rows_buffer,
                &hidden_size_buffer,
                &eps_buffer,
            ],
            physical_threads,
        )?;
        Ok(output_buffer)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_mla_kv_postprocess(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        input: &Buffer,
        input_len: usize,
        norm_weight: &[f32],
        batch_count: usize,
        token_count: usize,
        latent_dim: usize,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
        eps: f32,
    ) -> Result<(Buffer, Buffer)> {
        if batch_count == 0 || token_count == 0 || latent_dim == 0 || rope_dim == 0 {
            return Err(Error::backend(
                "MLA KV postprocess dimensions must be positive",
            ));
        }
        if !rope_dim.is_multiple_of(2) {
            return Err(Error::backend(format!(
                "MLA KV postprocess rope_dim must be even, got {rope_dim}"
            )));
        }
        if norm_weight.len() != latent_dim {
            return Err(Error::backend(format!(
                "MLA KV postprocess norm weight mismatch: expected {latent_dim}, got {}",
                norm_weight.len()
            )));
        }
        if !theta.is_finite() || theta <= 0.0 {
            return Err(Error::backend(format!(
                "MLA KV postprocess theta must be finite and positive, got {theta}"
            )));
        }
        if !eps.is_finite() || eps <= 0.0 {
            return Err(Error::backend(format!(
                "MLA KV postprocess epsilon must be finite and positive, got {eps}"
            )));
        }

        let row_count = batch_count
            .checked_mul(token_count)
            .ok_or_else(|| Error::backend("MLA KV postprocess row count overflow"))?;
        let total_dim = latent_dim
            .checked_add(rope_dim)
            .ok_or_else(|| Error::backend("MLA KV postprocess input width overflow"))?;
        let expected_input_len = row_count
            .checked_mul(total_dim)
            .ok_or_else(|| Error::backend("MLA KV postprocess input length overflow"))?;
        if input_len != expected_input_len {
            return Err(Error::backend(format!(
                "MLA KV postprocess input mismatch: expected {expected_input_len}, got {input_len}"
            )));
        }
        require_f32_capacity(input, input_len, "MLA KV postprocess input")?;

        let latent_len = row_count
            .checked_mul(latent_dim)
            .ok_or_else(|| Error::backend("MLA KV postprocess latent length overflow"))?;
        let rope_len = row_count
            .checked_mul(rope_dim)
            .ok_or_else(|| Error::backend("MLA KV postprocess RoPE length overflow"))?;
        let latent_output = self.arena.empty_f32(latent_len)?;
        let rope_output = self.arena.empty_f32(rope_len)?;
        let norm_weight = self.weight_buffers.get(device, norm_weight)?;
        let rows_buffer = self.arena.u32(
            u32::try_from(row_count)
                .map_err(|_| Error::backend("MLA KV postprocess rows exceed Metal u32"))?,
        )?;
        let token_count_buffer = self.arena.u32(
            u32::try_from(token_count)
                .map_err(|_| Error::backend("MLA KV token count exceeds Metal u32"))?,
        )?;
        let latent_dim_buffer = self.arena.u32(
            u32::try_from(latent_dim)
                .map_err(|_| Error::backend("MLA KV latent_dim exceeds Metal u32"))?,
        )?;
        let rope_dim_buffer = self.arena.u32(
            u32::try_from(rope_dim)
                .map_err(|_| Error::backend("MLA KV rope_dim exceeds Metal u32"))?,
        )?;
        let position_offset_buffer = self.arena.u32(
            u32::try_from(position_offset)
                .map_err(|_| Error::backend("MLA KV position exceeds Metal u32"))?,
        )?;
        let theta_buffer = self.arena.f32(theta)?;
        let eps_buffer = self.arena.f32(eps)?;
        let physical_threads = rms_norm_threads(&self.mla_kv_postprocess_pipeline, row_count)?;
        encode_1d(
            command_buffer,
            &self.mla_kv_postprocess_pipeline,
            &[
                input,
                &norm_weight,
                &latent_output,
                &rope_output,
                &rows_buffer,
                &token_count_buffer,
                &latent_dim_buffer,
                &rope_dim_buffer,
                &position_offset_buffer,
                &theta_buffer,
                &eps_buffer,
            ],
            physical_threads,
        )?;
        Ok((latent_output, rope_output))
    }

    pub(crate) fn pipeline(&self) -> &ComputePipelineState {
        &self.pipeline
    }

    pub(crate) fn thread_count(&self, rows: usize) -> Result<usize> {
        rms_norm_threads(&self.pipeline, rows)
    }
}

fn rms_norm_threads(pipeline: &ComputePipelineState, rows: usize) -> Result<usize> {
    let thread_execution_width = pipeline.thread_execution_width() as usize;
    if thread_execution_width != RMS_NORM_SIMD_LANES {
        return Err(Error::backend(format!(
            "RMSNorm requires {RMS_NORM_SIMD_LANES}-lane Apple Metal SIMD groups, got {thread_execution_width}"
        )));
    }
    let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
    if max_threads < RMS_NORM_THREADS_PER_ROW {
        return Err(Error::backend(format!(
            "RMSNorm requires {RMS_NORM_THREADS_PER_ROW} threads per row, pipeline allows {max_threads}"
        )));
    }
    rows.checked_mul(RMS_NORM_THREADS_PER_ROW)
        .ok_or_else(|| Error::backend("RMSNorm physical thread count overflow"))
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::RMS_NORM_THREADS_PER_ROW;
    use crate::metal::Metal;

    #[test]
    fn matches_cpu_reference_for_small_rows() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let rows = 2;
        let hidden_size = 4;
        let input = vec![
            0.25_f32, -0.50, 0.75, 1.00, //
            -1.25, 0.50, 0.10, 2.00,
        ];
        let weight = vec![1.0_f32, 1.25, 0.75, 1.50];
        let eps = 1e-5;

        let report = metal
            .rms_norm_f32_report(&input, &weight, rows, hidden_size, eps)
            .unwrap();
        let expected = cpu_rms_norm(&input, &weight, rows, hidden_size, eps);

        assert_eq!(report.rows, rows);
        assert_eq!(report.hidden_size, hidden_size);
        assert_eq!(report.input_len, input.len());
        assert_eq!(report.thread_count, rows * RMS_NORM_THREADS_PER_ROW);
        assert_close(&report.values, &expected, 1e-5);
    }

    #[test]
    fn fused_mla_kv_postprocess_matches_separate_cpu_operations() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch_count = 1;
        let token_count = 2;
        let latent_dim = 4;
        let rope_dim = 4;
        let position_offset = 3;
        let theta = 10_000.0;
        let eps = 1e-5;
        let input = vec![
            0.25_f32, -0.50, 0.75, 1.00, 0.10, 0.20, 0.30, 0.40, //
            -1.25, 0.50, 0.10, 2.00, -0.50, 0.75, 1.25, -1.50,
        ];
        let weight = vec![1.0_f32, 1.25, 0.75, 1.50];
        let input_buffer = metal.batch_upload_f32(&input).unwrap();
        let (latent_buffer, rope_buffer) = metal
            .batched_mla_kv_postprocess(
                &input_buffer,
                input.len(),
                &weight,
                batch_count,
                token_count,
                latent_dim,
                rope_dim,
                position_offset,
                theta,
                eps,
            )
            .unwrap();
        let latent = metal
            .batch_read_f32(&latent_buffer, batch_count * token_count * latent_dim)
            .unwrap();
        let rope = metal
            .batch_read_f32(&rope_buffer, batch_count * token_count * rope_dim)
            .unwrap();

        let (expected_latent, expected_rope) = cpu_mla_kv_postprocess(
            &input,
            &weight,
            token_count,
            latent_dim,
            rope_dim,
            position_offset,
            theta,
            eps,
        );
        assert_close(&latent, &expected_latent, 1e-5);
        assert_close(&rope, &expected_rope, 1e-5);
    }

    #[test]
    fn rejects_wrong_shape_before_dispatch() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let err = metal
            .rms_norm_f32(&[1.0, 2.0, 3.0], &[1.0, 1.0], 2, 2, 1e-5)
            .expect_err("shape mismatch should fail before Metal dispatch");

        assert!(err.to_string().contains("input shape mismatch"));
    }

    fn native_metal_or_skip() -> Option<Metal> {
        Metal::new().ok()
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

    #[allow(clippy::too_many_arguments)]
    fn cpu_mla_kv_postprocess(
        input: &[f32],
        weight: &[f32],
        token_count: usize,
        latent_dim: usize,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let row_count = input.len() / (latent_dim + rope_dim);
        let mut latent = vec![0.0; row_count * latent_dim];
        let mut rope = vec![0.0; row_count * rope_dim];
        for row in 0..row_count {
            let input_base = row * (latent_dim + rope_dim);
            let sumsq = input[input_base..input_base + latent_dim]
                .iter()
                .map(|value| value * value)
                .sum::<f32>();
            let scale = ((sumsq / latent_dim as f32) + eps).sqrt().recip();
            for dim in 0..latent_dim {
                latent[row * latent_dim + dim] = input[input_base + dim] * scale * weight[dim];
            }

            let position = position_offset + (row % token_count);
            let pair_count = rope_dim / 2;
            for dim in 0..rope_dim {
                let pair_index = dim / 2;
                let partner_dim = if dim.is_multiple_of(2) {
                    dim + 1
                } else {
                    dim - 1
                };
                let inv_freq = theta.powf(-(pair_index as f32 / pair_count as f32));
                let angle = position as f32 * inv_freq;
                let partner = input[input_base + latent_dim + partner_dim];
                let rotated = if dim.is_multiple_of(2) {
                    -partner
                } else {
                    partner
                };
                rope[row * rope_dim + dim] =
                    input[input_base + latent_dim + dim] * angle.cos() + rotated * angle.sin();
            }
        }
        (latent, rope)
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
