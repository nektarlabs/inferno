use ::metal::{ComputePipelineState, Device};
use common::{Error, Result};

use super::super::{library::MetalLibrary, pipeline::compute_pipeline};

const KERNEL_SOURCE: &str = include_str!("kernels/delta_prefill.metal");
const RECURRENT_KERNEL: &str = "qwen_delta_prefill_recurrent_kernel";
const MIN_SEQUENCE_LENGTH: usize = 32;
const SIMD_LANES: usize = 32;

pub(super) struct MetalQwenDeltaPrefill {
    prepare: ComputePipelineState,
    recurrent: ComputePipelineState,
    output_norm: ComputePipelineState,
}

impl MetalQwenDeltaPrefill {
    pub(super) fn new(device: &Device) -> Result<Self> {
        let library = MetalLibrary::compile_source(device, KERNEL_SOURCE)?;
        let recurrent = compute_pipeline(device, &library, RECURRENT_KERNEL)?;
        let width = recurrent.thread_execution_width() as usize;
        if width != SIMD_LANES {
            return Err(Error::backend(format!(
                "{RECURRENT_KERNEL} requires a {SIMD_LANES}-lane SIMD group, got {width}"
            )));
        }
        let output_norm =
            compute_pipeline(device, &library, "qwen_delta_prefill_output_norm_kernel")?;
        let prepare = compute_pipeline(device, &library, "qwen_delta_prefill_prepare_kernel")?;
        Ok(Self {
            prepare,
            recurrent,
            output_norm,
        })
    }

    pub(super) const fn supports(sequence_length: usize) -> bool {
        sequence_length >= MIN_SEQUENCE_LENGTH
    }

    pub(super) fn pipeline(&self) -> &ComputePipelineState {
        &self.recurrent
    }

    pub(super) fn output_norm_pipeline(&self) -> &ComputePipelineState {
        &self.output_norm
    }

    pub(super) fn prepare_pipeline(&self) -> &ComputePipelineState {
        &self.prepare
    }
}
