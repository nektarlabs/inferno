use ::metal::{Buffer, CommandBufferRef, CommandQueue, ComputePipelineState, Device};
use common::{Error, Result};
use tracing::trace;

use super::{
    buffers::{
        empty_f32_buffer, f32_buffer, f32_scalar_buffer, read_f32_buffer, require_f32_capacity,
        u32_scalar_buffer,
    },
    command::{dispatch_1d, encode_1d},
    library::MetalLibrary,
    pipeline::compute_pipeline,
    validation::{validate_rms_norm_buffer, validate_rms_norm_f32},
};

const RMS_NORM_KERNEL: &str = "rms_norm_f32_kernel";
const RMS_NORM_THREADS_PER_ROW: usize = 256;
const RMS_NORM_SIMD_LANES: usize = 32;

pub(crate) struct MetalRmsNorm {
    pipeline: ComputePipelineState,
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
    pub(crate) fn new(device: &Device, library: &MetalLibrary) -> Result<Self> {
        Ok(Self {
            pipeline: compute_pipeline(device, library, RMS_NORM_KERNEL)?,
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
            output_buffer: empty_f32_buffer(device, input.len())?,
            rows_buffer: u32_scalar_buffer(device, rows_u32)?,
            hidden_size_buffer: u32_scalar_buffer(device, hidden_size_u32)?,
            eps_buffer: f32_scalar_buffer(device, eps)?,
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

        let weight_buffer = f32_buffer(device, weight)?;
        let output_buffer = empty_f32_buffer(device, input_len)?;
        let rows_buffer = u32_scalar_buffer(device, rows_u32)?;
        let hidden_size_buffer = u32_scalar_buffer(device, hidden_size_u32)?;
        let eps_buffer = f32_scalar_buffer(device, eps)?;
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

    pub(crate) fn pipeline(&self) -> &ComputePipelineState {
        &self.pipeline
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
