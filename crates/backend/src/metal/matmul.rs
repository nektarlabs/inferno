use ::metal::{Buffer, CommandBufferRef, CommandQueue, ComputePipelineState, Device};
use common::{Error, Result};
use tracing::trace;

use super::{
    buffers::{
        empty_f32_buffer, f32_buffer, read_f32_buffer, require_f32_capacity, u32_scalar_buffer,
        ImmutableF32BufferCache,
    },
    command::{dispatch_1d, dispatch_2d, encode_1d, encode_2d},
    library::MetalLibrary,
    pipeline::compute_pipeline,
    validation::{validate_linear_f32, validate_matmul_f32},
};

const MATMUL_KERNEL: &str = "matmul_f32_kernel";
const LINEAR_KERNEL: &str = "linear_f32_kernel";
const LINEAR_GEMV_KERNEL: &str = "linear_f32_gemv_kernel";
const TILE_M: usize = 16;
const TILE_N: usize = 16;

pub(crate) struct MetalMatmul {
    matmul_pipeline: ComputePipelineState,
    linear_pipeline: ComputePipelineState,
    linear_gemv_pipeline: ComputePipelineState,
    weight_buffers: ImmutableF32BufferCache,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalMatmulReport {
    pub values: Vec<f32>,
    pub rows: usize,
    pub inner: usize,
    pub cols: usize,
    pub lhs_len: usize,
    pub rhs_len: usize,
    pub thread_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalLinearReport {
    pub values: Vec<f32>,
    pub rows: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub input_len: usize,
    pub weight_len: usize,
    pub thread_count: usize,
}

impl MetalMatmul {
    pub(crate) fn new(device: &Device, library: &MetalLibrary) -> Result<Self> {
        Ok(Self {
            matmul_pipeline: compute_pipeline(device, library, MATMUL_KERNEL)?,
            linear_pipeline: compute_pipeline(device, library, LINEAR_KERNEL)?,
            linear_gemv_pipeline: compute_pipeline(device, library, LINEAR_GEMV_KERNEL)?,
            weight_buffers: ImmutableF32BufferCache::default(),
        })
    }

    pub(crate) fn matmul(
        &self,
        device: &Device,
        queue: &CommandQueue,
        lhs: &[f32],
        rhs: &[f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<MetalMatmulReport> {
        validate_matmul_f32(lhs, rhs, rows, inner, cols)?;
        let output_len = rows
            .checked_mul(cols)
            .ok_or_else(|| Error::backend("matmul output length overflow"))?;

        let rows_u32 = u32::try_from(rows)
            .map_err(|_| Error::backend("matmul rows exceed Metal u32 limit"))?;
        let inner_u32 = u32::try_from(inner)
            .map_err(|_| Error::backend("matmul inner exceeds Metal u32 limit"))?;
        let cols_u32 = u32::try_from(cols)
            .map_err(|_| Error::backend("matmul cols exceed Metal u32 limit"))?;

        let lhs_buffer = f32_buffer(device, lhs)?;
        let rhs_buffer = f32_buffer(device, rhs)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let rows_buffer = u32_scalar_buffer(device, rows_u32)?;
        let inner_buffer = u32_scalar_buffer(device, inner_u32)?;
        let cols_buffer = u32_scalar_buffer(device, cols_u32)?;

        trace!(
            target: "inferno::metal",
            rows,
            inner,
            cols,
            "running native Metal F32 matmul"
        );

        dispatch_2d(
            queue,
            &self.matmul_pipeline,
            &[
                &lhs_buffer,
                &rhs_buffer,
                &output_buffer,
                &rows_buffer,
                &inner_buffer,
                &cols_buffer,
            ],
            cols,
            rows,
            TILE_N,
            TILE_M,
        )?;

        let values = read_f32_buffer(&output_buffer, output_len)?;

        Ok(MetalMatmulReport {
            values,
            rows,
            inner,
            cols,
            lhs_len: lhs.len(),
            rhs_len: rhs.len(),
            thread_count: output_len,
        })
    }

    pub(crate) fn linear(
        &self,
        device: &Device,
        queue: &CommandQueue,
        input: &[f32],
        weight: &[f32],
        rows: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<MetalLinearReport> {
        validate_linear_f32(input, weight, rows, in_features, out_features)?;
        let output_len = rows
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("linear output length overflow"))?;

        let rows_u32 = u32::try_from(rows)
            .map_err(|_| Error::backend("linear rows exceed Metal u32 limit"))?;
        let in_features_u32 = u32::try_from(in_features)
            .map_err(|_| Error::backend("linear in_features exceed Metal u32 limit"))?;
        let out_features_u32 = u32::try_from(out_features)
            .map_err(|_| Error::backend("linear out_features exceed Metal u32 limit"))?;

        let input_buffer = f32_buffer(device, input)?;
        let weight_buffer = f32_buffer(device, weight)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        trace!(
            target: "inferno::metal",
            rows,
            in_features,
            out_features,
            "running native Metal F32 linear"
        );

        if rows == 1 && in_features % 4 == 0 {
            let in_features_vec4 = u32::try_from(in_features / 4)
                .map_err(|_| Error::backend("linear in_features/4 exceed Metal u32 limit"))?;
            let in_features_vec4_buffer = u32_scalar_buffer(device, in_features_vec4)?;
            let out_features_buffer = u32_scalar_buffer(device, out_features_u32)?;

            dispatch_1d(
                queue,
                &self.linear_gemv_pipeline,
                &[
                    &input_buffer,
                    &weight_buffer,
                    &output_buffer,
                    &in_features_vec4_buffer,
                    &out_features_buffer,
                ],
                out_features,
            )?;
        } else {
            let rows_buffer = u32_scalar_buffer(device, rows_u32)?;
            let in_features_buffer = u32_scalar_buffer(device, in_features_u32)?;
            let out_features_buffer = u32_scalar_buffer(device, out_features_u32)?;

            dispatch_2d(
                queue,
                &self.linear_pipeline,
                &[
                    &input_buffer,
                    &weight_buffer,
                    &output_buffer,
                    &rows_buffer,
                    &in_features_buffer,
                    &out_features_buffer,
                ],
                out_features,
                rows,
                TILE_N,
                TILE_M,
            )?;
        }

        let values = read_f32_buffer(&output_buffer, output_len)?;

        Ok(MetalLinearReport {
            values,
            rows,
            in_features,
            out_features,
            input_len: input.len(),
            weight_len: weight.len(),
            thread_count: output_len,
        })
    }

    pub(crate) fn encode_linear(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        input: &Buffer,
        input_len: usize,
        weight: &[f32],
        rows: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        let expected_input_len = rows
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("batched F32 linear input length overflow"))?;
        if input_len != expected_input_len {
            return Err(Error::backend(format!(
                "batched F32 linear input length mismatch: expected {expected_input_len}, got {input_len}"
            )));
        }
        let expected_weight_len = out_features
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("batched F32 linear weight length overflow"))?;
        if weight.len() != expected_weight_len {
            return Err(Error::backend(format!(
                "batched F32 linear weight length mismatch: expected {expected_weight_len}, got {}",
                weight.len()
            )));
        }
        require_f32_capacity(input, input_len, "batched F32 linear input")?;
        let output_len = rows
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("batched F32 linear output length overflow"))?;
        let weight_buffer = self.weight_buffers.get(device, weight)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;

        trace!(
            target: "inferno::metal",
            rows,
            in_features,
            out_features,
            "encoding batched native Metal F32 linear"
        );

        if rows == 1 && in_features % 4 == 0 {
            let in_features_vec4 = u32::try_from(in_features / 4).map_err(|_| {
                Error::backend("batched F32 linear in_features/4 exceeds Metal u32 limit")
            })?;
            let out_features_u32 = u32::try_from(out_features).map_err(|_| {
                Error::backend("batched F32 linear out_features exceeds Metal u32 limit")
            })?;
            let in_features_vec4_buffer = u32_scalar_buffer(device, in_features_vec4)?;
            let out_features_buffer = u32_scalar_buffer(device, out_features_u32)?;
            encode_1d(
                command_buffer,
                &self.linear_gemv_pipeline,
                &[
                    input,
                    &weight_buffer,
                    &output_buffer,
                    &in_features_vec4_buffer,
                    &out_features_buffer,
                ],
                out_features,
            )?;
        } else {
            let rows_u32 = u32::try_from(rows)
                .map_err(|_| Error::backend("batched F32 linear rows exceed Metal u32 limit"))?;
            let in_features_u32 = u32::try_from(in_features).map_err(|_| {
                Error::backend("batched F32 linear in_features exceeds Metal u32 limit")
            })?;
            let out_features_u32 = u32::try_from(out_features).map_err(|_| {
                Error::backend("batched F32 linear out_features exceeds Metal u32 limit")
            })?;
            let rows_buffer = u32_scalar_buffer(device, rows_u32)?;
            let in_features_buffer = u32_scalar_buffer(device, in_features_u32)?;
            let out_features_buffer = u32_scalar_buffer(device, out_features_u32)?;
            encode_2d(
                command_buffer,
                &self.linear_pipeline,
                &[
                    input,
                    &weight_buffer,
                    &output_buffer,
                    &rows_buffer,
                    &in_features_buffer,
                    &out_features_buffer,
                ],
                out_features,
                rows,
                TILE_N,
                TILE_M,
            )?;
        }

        Ok(output_buffer)
    }
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use crate::metal::Metal;

    #[test]
    fn matmul_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let rows = 2;
        let inner = 3;
        let cols = 2;
        let lhs = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs = vec![0.5_f32, 1.0, -1.0, 1.5, -0.5, 0.25];

        let report = metal
            .matmul_f32_report(&lhs, &rhs, rows, inner, cols)
            .unwrap();
        let expected = cpu_matmul(&lhs, &rhs, rows, inner, cols);

        assert_eq!(report.rows, rows);
        assert_eq!(report.inner, inner);
        assert_eq!(report.cols, cols);
        assert_eq!(report.thread_count, rows * cols);
        assert_close(&report.values, &expected, 1e-5);
    }

    #[test]
    fn linear_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let rows = 2;
        let in_features = 3;
        let out_features = 2;
        let input = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = vec![0.5_f32, -1.0, 0.25, 1.5, -0.5, 2.0];

        let report = metal
            .linear_f32_report(&input, &weight, rows, in_features, out_features)
            .unwrap();
        let expected = cpu_linear(&input, &weight, rows, in_features, out_features);

        assert_eq!(report.rows, rows);
        assert_eq!(report.in_features, in_features);
        assert_eq!(report.out_features, out_features);
        assert_eq!(report.thread_count, rows * out_features);
        assert_close(&report.values, &expected, 1e-5);
    }

    #[test]
    fn linear_gemv_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let rows = 1;
        let in_features = 4;
        let out_features = 3;
        let input = vec![1.0_f32, -2.0, 0.5, 3.0];
        let weight = vec![
            0.5_f32, -1.0, 0.25, 1.5, -0.5, 2.0, 1.25, -0.75, 0.125, 0.25, -0.5, 1.0,
        ];

        let report = metal
            .linear_f32_report(&input, &weight, rows, in_features, out_features)
            .unwrap();
        let expected = cpu_linear(&input, &weight, rows, in_features, out_features);

        assert_eq!(report.rows, rows);
        assert_eq!(report.in_features, in_features);
        assert_eq!(report.out_features, out_features);
        assert_eq!(report.thread_count, rows * out_features);
        assert_close(&report.values, &expected, 1e-5);
    }

    fn native_metal_or_skip() -> Option<Metal> {
        Metal::new().ok()
    }

    fn cpu_matmul(lhs: &[f32], rhs: &[f32], rows: usize, inner: usize, cols: usize) -> Vec<f32> {
        let mut output = vec![0.0; rows * cols];
        for row in 0..rows {
            for col in 0..cols {
                let mut sum = 0.0;
                for index in 0..inner {
                    sum += lhs[row * inner + index] * rhs[index * cols + col];
                }
                output[row * cols + col] = sum;
            }
        }
        output
    }

    fn cpu_linear(
        input: &[f32],
        weight: &[f32],
        rows: usize,
        in_features: usize,
        out_features: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; rows * out_features];
        for row in 0..rows {
            for output_feature in 0..out_features {
                let mut sum = 0.0;
                for input_feature in 0..in_features {
                    sum += input[row * in_features + input_feature]
                        * weight[output_feature * in_features + input_feature];
                }
                output[row * out_features + output_feature] = sum;
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
