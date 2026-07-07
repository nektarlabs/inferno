use ::metal::{Buffer, CommandBufferRef, CommandQueue, ComputePipelineState, Device};
use common::{Error, Result};
use tracing::trace;

use super::{
    buffers::{
        empty_f32_buffer, empty_u32_buffer, f32_buffer, f32_scalar_buffer, read_f32_buffer,
        require_f32_capacity, u32_buffer, u32_scalar_buffer,
    },
    command::{dispatch_1d, encode_1d},
    library::MetalLibrary,
    pipeline::compute_pipeline,
    validation::{
        debug_assert_finite_values, validate_moe_gather_tokens_f32,
        validate_moe_weighted_index_add_combine_f32,
    },
};

const MOE_GATHER_TOKENS_KERNEL: &str = "moe_gather_tokens_f32_kernel";
const MOE_WEIGHTED_INDEX_ADD_COMBINE_KERNEL: &str = "moe_weighted_index_add_combine_f32_kernel";
const MOE_WEIGHTED_TOKEN_MAJOR_COMBINE_KERNEL: &str = "moe_weighted_token_major_combine_f32_kernel";
const MOE_ROUTER_TOPK_KERNEL: &str = "moe_router_topk_f32_kernel";

pub(crate) struct MetalMoe {
    gather_pipeline: ComputePipelineState,
    combine_pipeline: ComputePipelineState,
    token_major_combine_pipeline: ComputePipelineState,
    router_topk_pipeline: ComputePipelineState,
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

#[derive(Debug)]
pub(crate) struct MetalRouterTopKBuffers {
    pub(crate) expert_ids: Buffer,
    pub(crate) expert_weights: Buffer,
    pub(crate) output_len: usize,
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
            token_major_combine_pipeline: compute_pipeline(
                device,
                library,
                MOE_WEIGHTED_TOKEN_MAJOR_COMBINE_KERNEL,
            )?,
            router_topk_pipeline: compute_pipeline(device, library, MOE_ROUTER_TOPK_KERNEL)?,
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
        let token_major_assignments_per_token =
            token_major_assignments_per_token(token_indices, token_count, assignment_count);
        let combine_count_u32 = match token_major_assignments_per_token {
            Some(assignments_per_token) => u32::try_from(assignments_per_token).map_err(|_| {
                Error::backend("MoE combine assignments_per_token exceeds Metal u32 limit")
            })?,
            None => assignment_count_u32,
        };
        let combine_count_buffer = u32_scalar_buffer(device, combine_count_u32)?;
        let combine_pipeline = if token_major_assignments_per_token.is_some() {
            &self.token_major_combine_pipeline
        } else {
            &self.combine_pipeline
        };

        trace!(
            target: "inferno::metal",
            token_count,
            hidden_size,
            assignment_count,
            assignments_per_token = token_major_assignments_per_token.unwrap_or(0),
            token_major = token_major_assignments_per_token.is_some(),
            "running native Metal weighted MoE combine"
        );

        dispatch_1d(
            queue,
            combine_pipeline,
            &[
                &accumulator_buffer,
                &token_indices_buffer,
                &expert_outputs_buffer,
                &expert_weights_buffer,
                &output_buffer,
                &token_count_buffer,
                &hidden_size_buffer,
                &combine_count_buffer,
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

    /// Encodes the weighted MoE combine into an open batched command buffer.
    /// The accumulator and stacked expert outputs are device-resident; the
    /// routing metadata (token indices and expert weights) is host data chosen
    /// by the CPU router, so it is validated fully and uploaded here.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_weighted_index_add_combine(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        accumulator: &Buffer,
        accumulator_len: usize,
        token_indices: &[u32],
        expert_outputs: &Buffer,
        expert_outputs_len: usize,
        expert_weights: &[f32],
        token_count: usize,
        hidden_size: usize,
        assignment_count: usize,
    ) -> Result<Buffer> {
        if token_count == 0 || hidden_size == 0 || assignment_count == 0 {
            return Err(Error::backend(
                "MoE combine requires non-zero token_count, hidden_size and assignment_count",
            ));
        }
        let output_len = token_count
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("MoE combine output length overflow"))?;
        if accumulator_len != output_len {
            return Err(Error::backend(format!(
                "MoE combine accumulator length mismatch: expected {output_len}, got {accumulator_len}"
            )));
        }
        let expected_expert_len = assignment_count
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("MoE combine expert output length overflow"))?;
        if expert_outputs_len != expected_expert_len {
            return Err(Error::backend(format!(
                "MoE combine expert outputs length mismatch: expected {expected_expert_len}, got {expert_outputs_len}"
            )));
        }
        if token_indices.len() != assignment_count || expert_weights.len() != assignment_count {
            return Err(Error::backend(format!(
                "MoE combine routing metadata mismatch: {} indices and {} weights for {assignment_count} assignments",
                token_indices.len(),
                expert_weights.len()
            )));
        }
        for token_index in token_indices {
            if *token_index as usize >= token_count {
                return Err(Error::backend(format!(
                    "MoE combine token index {token_index} is outside token_count {token_count}"
                )));
            }
        }
        debug_assert_finite_values("MoE combine expert weights", expert_weights);
        require_f32_capacity(accumulator, accumulator_len, "MoE combine accumulator")?;
        require_f32_capacity(
            expert_outputs,
            expert_outputs_len,
            "MoE combine expert outputs",
        )?;

        let token_count_u32 = u32::try_from(token_count)
            .map_err(|_| Error::backend("MoE combine token_count exceeds Metal u32 limit"))?;
        let hidden_size_u32 = u32::try_from(hidden_size)
            .map_err(|_| Error::backend("MoE combine hidden_size exceeds Metal u32 limit"))?;
        let assignment_count_u32 = u32::try_from(assignment_count)
            .map_err(|_| Error::backend("MoE combine assignment_count exceeds Metal u32 limit"))?;

        let token_indices_buffer = u32_buffer(device, token_indices)?;
        let expert_weights_buffer = f32_buffer(device, expert_weights)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let token_count_buffer = u32_scalar_buffer(device, token_count_u32)?;
        let hidden_size_buffer = u32_scalar_buffer(device, hidden_size_u32)?;
        let token_major_assignments_per_token =
            token_major_assignments_per_token(token_indices, token_count, assignment_count);
        let combine_count_u32 = match token_major_assignments_per_token {
            Some(assignments_per_token) => u32::try_from(assignments_per_token).map_err(|_| {
                Error::backend("MoE combine assignments_per_token exceeds Metal u32 limit")
            })?,
            None => assignment_count_u32,
        };
        let combine_count_buffer = u32_scalar_buffer(device, combine_count_u32)?;
        let combine_pipeline = if token_major_assignments_per_token.is_some() {
            &self.token_major_combine_pipeline
        } else {
            &self.combine_pipeline
        };

        encode_1d(
            command_buffer,
            combine_pipeline,
            &[
                accumulator,
                &token_indices_buffer,
                expert_outputs,
                &expert_weights_buffer,
                &output_buffer,
                &token_count_buffer,
                &hidden_size_buffer,
                &combine_count_buffer,
            ],
            output_len,
        )?;
        Ok(output_buffer)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_router_topk(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        router_logits: &Buffer,
        router_logits_len: usize,
        correction_bias: &[f32],
        token_count: usize,
        expert_count: usize,
        top_k: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
    ) -> Result<MetalRouterTopKBuffers> {
        if token_count == 0 || expert_count == 0 {
            return Err(Error::backend(
                "MoE router top-k requires non-zero token_count and expert_count",
            ));
        }
        if top_k == 0 || top_k > 8 {
            return Err(Error::backend(format!(
                "MoE router top-k supports 1..=8 selected experts, got {top_k}"
            )));
        }
        if top_k > expert_count {
            return Err(Error::backend(format!(
                "MoE router top-k {top_k} exceeds expert_count {expert_count}"
            )));
        }
        let expected_logits_len = token_count
            .checked_mul(expert_count)
            .ok_or_else(|| Error::backend("MoE router logits length overflow"))?;
        if router_logits_len != expected_logits_len {
            return Err(Error::backend(format!(
                "MoE router logits length mismatch: expected {expected_logits_len}, got {router_logits_len}"
            )));
        }
        if correction_bias.len() != expert_count {
            return Err(Error::backend(format!(
                "MoE router correction bias length mismatch: expected {expert_count}, got {}",
                correction_bias.len()
            )));
        }
        debug_assert_finite_values("MoE router correction bias", correction_bias);
        if !routed_scaling_factor.is_finite() {
            return Err(Error::backend(
                "MoE router routed scaling factor must be finite",
            ));
        }
        require_f32_capacity(router_logits, router_logits_len, "MoE router logits")?;

        let output_len = token_count
            .checked_mul(top_k)
            .ok_or_else(|| Error::backend("MoE router top-k output length overflow"))?;
        let expert_ids_buffer = empty_u32_buffer(device, output_len)?;
        let expert_weights_buffer = empty_f32_buffer(device, output_len)?;
        let correction_bias_buffer = f32_buffer(device, correction_bias)?;
        let token_count_buffer = u32_scalar_buffer(
            device,
            u32::try_from(token_count)
                .map_err(|_| Error::backend("MoE router token_count exceeds Metal u32 limit"))?,
        )?;
        let expert_count_buffer = u32_scalar_buffer(
            device,
            u32::try_from(expert_count)
                .map_err(|_| Error::backend("MoE router expert_count exceeds Metal u32 limit"))?,
        )?;
        let top_k_buffer = u32_scalar_buffer(
            device,
            u32::try_from(top_k)
                .map_err(|_| Error::backend("MoE router top_k exceeds Metal u32 limit"))?,
        )?;
        let norm_topk_prob_buffer = u32_scalar_buffer(device, u32::from(norm_topk_prob))?;
        let routed_scaling_factor_buffer = f32_scalar_buffer(device, routed_scaling_factor)?;

        encode_1d(
            command_buffer,
            &self.router_topk_pipeline,
            &[
                router_logits,
                &correction_bias_buffer,
                &expert_ids_buffer,
                &expert_weights_buffer,
                &token_count_buffer,
                &expert_count_buffer,
                &top_k_buffer,
                &norm_topk_prob_buffer,
                &routed_scaling_factor_buffer,
            ],
            token_count,
        )?;

        Ok(MetalRouterTopKBuffers {
            expert_ids: expert_ids_buffer,
            expert_weights: expert_weights_buffer,
            output_len,
        })
    }
}

fn token_major_assignments_per_token(
    token_indices: &[u32],
    token_count: usize,
    assignment_count: usize,
) -> Option<usize> {
    if token_count == 0 || assignment_count == 0 || assignment_count % token_count != 0 {
        return None;
    }
    let assignments_per_token = assignment_count / token_count;
    for token in 0..token_count {
        let expected = u32::try_from(token).ok()?;
        let base = token.checked_mul(assignments_per_token)?;
        for rank in 0..assignments_per_token {
            if token_indices.get(base + rank).copied()? != expected {
                return None;
            }
        }
    }
    Some(assignments_per_token)
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::token_major_assignments_per_token;
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
    fn matches_cpu_reference_for_token_major_topk_assignments() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let token_count = 3;
        let hidden_size = 2;
        let assignment_count = 6;
        let accumulator = vec![
            0.5_f32, 1.0, //
            1.5, 2.0, //
            2.5, 3.0,
        ];
        let token_indices = vec![0_u32, 0, 1, 1, 2, 2];
        let expert_outputs = vec![
            1.0_f32, 2.0, //
            3.0, 4.0, //
            5.0, 6.0, //
            7.0, 8.0, //
            9.0, 10.0, //
            11.0, 12.0,
        ];
        let expert_weights = vec![0.25_f32, 0.75, 0.5, 0.5, 1.0, -0.25];

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

        assert_eq!(
            token_major_assignments_per_token(&token_indices, token_count, assignment_count),
            Some(2)
        );
        assert_eq!(report.values, expected);
    }

    #[test]
    fn rejects_token_major_fast_path_for_expert_grouped_assignments() {
        let token_indices = vec![0_u32, 2, 2, 3, 0];
        assert_eq!(
            token_major_assignments_per_token(&token_indices, 4, token_indices.len()),
            None
        );
    }

    #[test]
    fn router_topk_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let token_count = 2;
        let expert_count = 5;
        let top_k = 2;
        let routed_scaling_factor = 2.0_f32;
        let logits = vec![
            0.0_f32, 1.0, -1.0, 2.0, 0.5, //
            2.0, 0.0, -2.0, 1.0, 1.0,
        ];
        let correction_bias = vec![0.0_f32, 0.10, 1.00, -0.50, 0.00];
        let logits_buffer = metal.batch_upload_f32(&logits).unwrap();

        let (expert_ids, expert_weights) = metal
            .batched_moe_router_topk(
                &logits_buffer,
                logits.len(),
                &correction_bias,
                token_count,
                expert_count,
                top_k,
                true,
                routed_scaling_factor,
            )
            .unwrap();
        let (expected_ids, expected_weights) = cpu_router_topk(
            &logits,
            &correction_bias,
            token_count,
            expert_count,
            top_k,
            true,
            routed_scaling_factor,
        );

        assert_eq!(expert_ids, expected_ids);
        assert_eq!(expert_weights.len(), expected_weights.len());
        for (index, (actual, expected)) in expert_weights.iter().zip(&expected_weights).enumerate()
        {
            let delta = (actual - expected).abs();
            assert!(
                delta <= 1e-5,
                "router weight {index} differs: actual={actual}, expected={expected}"
            );
        }
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

    fn cpu_router_topk(
        logits: &[f32],
        correction_bias: &[f32],
        token_count: usize,
        expert_count: usize,
        top_k: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
    ) -> (Vec<u32>, Vec<f32>) {
        let mut expert_ids = Vec::with_capacity(token_count * top_k);
        let mut weights = Vec::with_capacity(token_count * top_k);
        for token in 0..token_count {
            let mut top = Vec::<(usize, f32, f32)>::with_capacity(top_k);
            for expert in 0..expert_count {
                let score = 1.0 / (1.0 + (-logits[token * expert_count + expert]).exp());
                let corrected = score + correction_bias[expert];
                let insert_at = top
                    .iter()
                    .position(|(existing_expert, existing_corrected, _)| {
                        corrected > *existing_corrected
                            || (corrected == *existing_corrected && expert < *existing_expert)
                    })
                    .unwrap_or(top.len());
                if insert_at < top_k {
                    top.insert(insert_at, (expert, corrected, score));
                    if top.len() > top_k {
                        top.pop();
                    }
                }
            }
            let sum = top.iter().map(|(_, _, score)| *score).sum::<f32>();
            for (expert, _, score) in top {
                let weight = if norm_topk_prob { score / sum } else { score };
                expert_ids.push(expert as u32);
                weights.push(weight * routed_scaling_factor);
            }
        }
        (expert_ids, weights)
    }
}
