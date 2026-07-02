use ::metal::{CommandQueue, ComputePipelineState, Device};
use common::{Error, Result};
use tracing::trace;

use super::{
    buffers::{empty_f32_buffer, f32_buffer, read_f32_buffer, u32_buffer, u32_scalar_buffer},
    command::dispatch_1d,
    library::MetalLibrary,
    pipeline::compute_pipeline,
    validation::{validate_moe_gather_tokens_f32, validate_moe_weighted_index_add_combine_f32},
};

const MOE_GATHER_TOKENS_KERNEL: &str = "moe_gather_tokens_f32_kernel";
const MOE_WEIGHTED_INDEX_ADD_COMBINE_KERNEL: &str = "moe_weighted_index_add_combine_f32_kernel";

pub(crate) struct MetalMoe {
    gather_pipeline: ComputePipelineState,
    combine_pipeline: ComputePipelineState,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalMoeGatherReport {
    pub values: Vec<f32>,
    pub token_count: usize,
    pub hidden_size: usize,
    pub assignment_count: usize,
    pub thread_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalMoeCombineReport {
    pub values: Vec<f32>,
    pub token_count: usize,
    pub hidden_size: usize,
    pub assignment_count: usize,
    pub thread_count: usize,
}

impl MetalMoe {
    pub(crate) fn new(device: &Device, library: &MetalLibrary) -> Result<Self> {
        Ok(Self {
            gather_pipeline: compute_pipeline(device, library, MOE_GATHER_TOKENS_KERNEL)?,
            combine_pipeline: compute_pipeline(
                device,
                library,
                MOE_WEIGHTED_INDEX_ADD_COMBINE_KERNEL,
            )?,
        })
    }

    pub(crate) fn gather_tokens(
        &self,
        device: &Device,
        queue: &CommandQueue,
        flat_tokens: &[f32],
        token_indices: &[u32],
        token_count: usize,
        hidden_size: usize,
        assignment_count: usize,
    ) -> Result<MetalMoeGatherReport> {
        validate_moe_gather_tokens_f32(
            flat_tokens,
            token_indices,
            token_count,
            hidden_size,
            assignment_count,
        )?;

        let token_count_u32 = u32::try_from(token_count)
            .map_err(|_| Error::backend("MoE gather token_count exceeds Metal u32 limit"))?;
        let hidden_size_u32 = u32::try_from(hidden_size)
            .map_err(|_| Error::backend("MoE gather hidden_size exceeds Metal u32 limit"))?;
        let assignment_count_u32 = u32::try_from(assignment_count)
            .map_err(|_| Error::backend("MoE gather assignment_count exceeds Metal u32 limit"))?;
        let output_len = assignment_count
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("MoE gather output length overflow"))?;

        let flat_tokens_buffer = f32_buffer(device, flat_tokens)?;
        let token_indices_buffer = u32_buffer(device, token_indices)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let token_count_buffer = u32_scalar_buffer(device, token_count_u32)?;
        let hidden_size_buffer = u32_scalar_buffer(device, hidden_size_u32)?;
        let assignment_count_buffer = u32_scalar_buffer(device, assignment_count_u32)?;

        trace!(
            target: "inferno::metal",
            token_count,
            hidden_size,
            assignment_count,
            "running native Metal MoE token gather"
        );

        dispatch_1d(
            queue,
            &self.gather_pipeline,
            &[
                &flat_tokens_buffer,
                &token_indices_buffer,
                &output_buffer,
                &token_count_buffer,
                &hidden_size_buffer,
                &assignment_count_buffer,
            ],
            output_len,
        )?;

        let values = read_f32_buffer(&output_buffer, output_len)?;

        Ok(MetalMoeGatherReport {
            values,
            token_count,
            hidden_size,
            assignment_count,
            thread_count: output_len,
        })
    }

    pub(crate) fn weighted_index_add_combine(
        &self,
        device: &Device,
        queue: &CommandQueue,
        accumulator: &[f32],
        token_indices: &[u32],
        expert_outputs: &[f32],
        expert_weights: &[f32],
        token_count: usize,
        hidden_size: usize,
        assignment_count: usize,
    ) -> Result<MetalMoeCombineReport> {
        validate_moe_weighted_index_add_combine_f32(
            accumulator,
            token_indices,
            expert_outputs,
            expert_weights,
            token_count,
            hidden_size,
            assignment_count,
        )?;

        let token_count_u32 = u32::try_from(token_count)
            .map_err(|_| Error::backend("MoE combine token_count exceeds Metal u32 limit"))?;
        let hidden_size_u32 = u32::try_from(hidden_size)
            .map_err(|_| Error::backend("MoE combine hidden_size exceeds Metal u32 limit"))?;
        let assignment_count_u32 = u32::try_from(assignment_count)
            .map_err(|_| Error::backend("MoE combine assignment_count exceeds Metal u32 limit"))?;
        let output_len = token_count
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("MoE combine output length overflow"))?;

        let accumulator_buffer = f32_buffer(device, accumulator)?;
        let token_indices_buffer = u32_buffer(device, token_indices)?;
        let expert_outputs_buffer = f32_buffer(device, expert_outputs)?;
        let expert_weights_buffer = f32_buffer(device, expert_weights)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let token_count_buffer = u32_scalar_buffer(device, token_count_u32)?;
        let hidden_size_buffer = u32_scalar_buffer(device, hidden_size_u32)?;
        let assignment_count_buffer = u32_scalar_buffer(device, assignment_count_u32)?;

        trace!(
            target: "inferno::metal",
            token_count,
            hidden_size,
            assignment_count,
            "running native Metal weighted MoE combine"
        );

        dispatch_1d(
            queue,
            &self.combine_pipeline,
            &[
                &accumulator_buffer,
                &token_indices_buffer,
                &expert_outputs_buffer,
                &expert_weights_buffer,
                &output_buffer,
                &token_count_buffer,
                &hidden_size_buffer,
                &assignment_count_buffer,
            ],
            output_len,
        )?;

        let values = read_f32_buffer(&output_buffer, output_len)?;

        Ok(MetalMoeCombineReport {
            values,
            token_count,
            hidden_size,
            assignment_count,
            thread_count: output_len,
        })
    }
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use crate::metal::Metal;

    #[test]
    fn gathers_selected_token_rows() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let token_count = 4;
        let hidden_size = 3;
        let assignment_count = 3;
        let flat_tokens = vec![
            1.0_f32, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
            10.0, 11.0, 12.0,
        ];
        let token_indices = vec![2_u32, 0, 2];

        let report = metal
            .moe_gather_tokens_f32_report(
                &flat_tokens,
                &token_indices,
                token_count,
                hidden_size,
                assignment_count,
            )
            .unwrap();

        assert_eq!(report.token_count, token_count);
        assert_eq!(report.hidden_size, hidden_size);
        assert_eq!(report.assignment_count, assignment_count);
        assert_eq!(report.thread_count, assignment_count * hidden_size);
        assert_eq!(
            report.values,
            vec![
                7.0_f32, 8.0, 9.0, //
                1.0, 2.0, 3.0, //
                7.0, 8.0, 9.0,
            ]
        );
    }

    #[test]
    fn matches_cpu_reference_for_repeated_token_indices() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let token_count = 4;
        let hidden_size = 3;
        let assignment_count = 5;
        let accumulator = vec![
            0.5_f32, 1.0, 1.5, //
            2.0, 2.5, 3.0, //
            3.5, 4.0, 4.5, //
            5.0, 5.5, 6.0,
        ];
        let token_indices = vec![0_u32, 2, 2, 3, 0];
        let expert_outputs = vec![
            1.0_f32, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
            10.0, 11.0, 12.0, //
            13.0, 14.0, 15.0,
        ];
        let expert_weights = vec![1.0_f32, 0.5, 2.0, -1.0, 0.25];

        let report = metal
            .moe_weighted_index_add_combine_f32_report(
                &accumulator,
                &token_indices,
                &expert_outputs,
                &expert_weights,
                token_count,
                hidden_size,
                assignment_count,
            )
            .unwrap();
        let expected = cpu_combine(
            &accumulator,
            &token_indices,
            &expert_outputs,
            &expert_weights,
            token_count,
            hidden_size,
        );

        assert_eq!(report.token_count, token_count);
        assert_eq!(report.hidden_size, hidden_size);
        assert_eq!(report.assignment_count, assignment_count);
        assert_eq!(report.thread_count, token_count * hidden_size);
        assert_eq!(report.values, expected);
    }

    #[test]
    fn rejects_bad_token_index_before_dispatch() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let err = metal
            .moe_weighted_index_add_combine_f32(&[0.0; 4], &[2], &[1.0, 2.0], &[1.0], 2, 2, 1)
            .expect_err("bad token index should fail before Metal dispatch");

        assert!(err.to_string().contains("outside token_count"));
    }

    fn native_metal_or_skip() -> Option<Metal> {
        Metal::new().ok()
    }

    fn cpu_combine(
        accumulator: &[f32],
        token_indices: &[u32],
        expert_outputs: &[f32],
        expert_weights: &[f32],
        token_count: usize,
        hidden_size: usize,
    ) -> Vec<f32> {
        let mut output = accumulator.to_vec();
        for (assignment, token_index) in token_indices.iter().copied().enumerate() {
            let token_index = token_index as usize;
            for hidden in 0..hidden_size {
                output[token_index * hidden_size + hidden] +=
                    expert_outputs[assignment * hidden_size + hidden] * expert_weights[assignment];
            }
        }
        assert_eq!(output.len(), token_count * hidden_size);
        output
    }
}
