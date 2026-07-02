use ::metal::{CommandQueue, ComputePipelineState, Device};
use common::{Error, Result};
use tracing::trace;

use super::{
    buffers::{empty_f32_buffer, f32_buffer, read_f32_buffer, u32_scalar_buffer},
    command::dispatch_1d,
    library::MetalLibrary,
    pipeline::compute_pipeline,
    validation::{validate_add_f32, validate_swiglu_f32},
};

const SWIGLU_KERNEL: &str = "swiglu_f32_kernel";
const ADD_KERNEL: &str = "add_f32_kernel";

pub(crate) struct MetalActivation {
    swiglu_pipeline: ComputePipelineState,
    add_pipeline: ComputePipelineState,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalSwiGluReport {
    pub values: Vec<f32>,
    pub value_count: usize,
    pub thread_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalAddReport {
    pub values: Vec<f32>,
    pub value_count: usize,
    pub thread_count: usize,
}

impl MetalActivation {
    pub(crate) fn new(device: &Device, library: &MetalLibrary) -> Result<Self> {
        Ok(Self {
            swiglu_pipeline: compute_pipeline(device, library, SWIGLU_KERNEL)?,
            add_pipeline: compute_pipeline(device, library, ADD_KERNEL)?,
        })
    }

    pub(crate) fn swiglu(
        &self,
        device: &Device,
        queue: &CommandQueue,
        gate: &[f32],
        up: &[f32],
    ) -> Result<MetalSwiGluReport> {
        let value_count = gate.len();
        validate_swiglu_f32(gate, up, value_count)?;

        let value_count_u32 = u32::try_from(value_count)
            .map_err(|_| Error::backend("SwiGLU value_count exceeds Metal u32 limit"))?;

        let gate_buffer = f32_buffer(device, gate)?;
        let up_buffer = f32_buffer(device, up)?;
        let output_buffer = empty_f32_buffer(device, value_count)?;
        let value_count_buffer = u32_scalar_buffer(device, value_count_u32)?;

        trace!(
            target: "inferno::metal",
            value_count,
            "running native Metal F32 SwiGLU"
        );

        dispatch_1d(
            queue,
            &self.swiglu_pipeline,
            &[
                &gate_buffer,
                &up_buffer,
                &output_buffer,
                &value_count_buffer,
            ],
            value_count,
        )?;

        let values = read_f32_buffer(&output_buffer, value_count)?;

        Ok(MetalSwiGluReport {
            values,
            value_count,
            thread_count: value_count,
        })
    }

    pub(crate) fn add(
        &self,
        device: &Device,
        queue: &CommandQueue,
        lhs: &[f32],
        rhs: &[f32],
    ) -> Result<MetalAddReport> {
        let value_count = lhs.len();
        validate_add_f32(lhs, rhs, value_count)?;

        let value_count_u32 = u32::try_from(value_count)
            .map_err(|_| Error::backend("add value_count exceeds Metal u32 limit"))?;

        let lhs_buffer = f32_buffer(device, lhs)?;
        let rhs_buffer = f32_buffer(device, rhs)?;
        let output_buffer = empty_f32_buffer(device, value_count)?;
        let value_count_buffer = u32_scalar_buffer(device, value_count_u32)?;

        trace!(
            target: "inferno::metal",
            value_count,
            "running native Metal F32 add"
        );

        dispatch_1d(
            queue,
            &self.add_pipeline,
            &[
                &lhs_buffer,
                &rhs_buffer,
                &output_buffer,
                &value_count_buffer,
            ],
            value_count,
        )?;

        let values = read_f32_buffer(&output_buffer, value_count)?;

        Ok(MetalAddReport {
            values,
            value_count,
            thread_count: value_count,
        })
    }
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use crate::metal::Metal;

    #[test]
    fn swiglu_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let gate = vec![-1.0_f32, 0.0, 0.5, 2.0, 4.0];
        let up = vec![0.25_f32, 0.5, -1.0, 1.5, -0.75];

        let report = metal.swiglu_f32_report(&gate, &up).unwrap();
        let expected = cpu_swiglu(&gate, &up);

        assert_eq!(report.value_count, gate.len());
        assert_eq!(report.thread_count, gate.len());
        assert_close(&report.values, &expected, 1e-5);
    }

    #[test]
    fn add_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let lhs = vec![1.0_f32, -2.0, 0.5, 4.0];
        let rhs = vec![0.25_f32, 2.5, -1.5, -0.75];

        let report = metal.add_f32_report(&lhs, &rhs).unwrap();
        let expected = lhs
            .iter()
            .zip(&rhs)
            .map(|(lhs, rhs)| lhs + rhs)
            .collect::<Vec<_>>();

        assert_eq!(report.value_count, lhs.len());
        assert_eq!(report.thread_count, lhs.len());
        assert_close(&report.values, &expected, 1e-6);
    }

    fn native_metal_or_skip() -> Option<Metal> {
        Metal::new().ok()
    }

    fn cpu_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .zip(up)
            .map(|(gate_value, up_value)| {
                let silu = *gate_value / (1.0 + (-*gate_value).exp());
                silu * *up_value
            })
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= tolerance,
                "actual={actual} expected={expected}"
            );
        }
    }
}
