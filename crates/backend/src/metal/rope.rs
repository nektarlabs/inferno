use ::metal::{Buffer, CommandBufferRef, CommandQueue, ComputePipelineState, Device};
use common::{Error, Result};
use tracing::trace;

use super::{
    arena::MetalArena,
    buffers::{f32_buffer, read_f32_buffer, require_f32_capacity},
    command::{dispatch_1d, dispatch_1d_threadgroups, encode_1d, encode_1d_threadgroups},
    library::MetalLibrary,
    pipeline::compute_pipeline,
    validation::{validate_rope_slice_buffer, validate_rope_slice_f32},
};

const ROPE_PAIR_KERNEL: &str = "rope_slice_f32_pair_kernel";
const ROPE_SHARED4_KERNEL: &str = "rope_slice_f32_shared4_kernel";
const SHARED_HEADS_PER_GROUP: usize = 4;
const SHARED_THREADS_PER_GROUP: usize = 256;
const SHARED_ROPE_DIM: usize = 64;
const SHARED_MIN_TOKEN_COUNT: usize = 32;

pub(crate) struct MetalRope {
    pair_pipeline: ComputePipelineState,
    shared4_pipeline: ComputePipelineState,
    arena: MetalArena,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalRopeReport {
    pub values: Vec<f32>,
    pub batch_count: usize,
    pub token_count: usize,
    pub head_count: usize,
    pub rope_dim: usize,
    pub position_offset: usize,
    pub theta: f32,
    pub thread_count: usize,
    pub shared_coefficients: bool,
}

impl MetalRope {
    pub(crate) fn new(device: &Device, library: &MetalLibrary, arena: MetalArena) -> Result<Self> {
        Ok(Self {
            pair_pipeline: compute_pipeline(device, library, ROPE_PAIR_KERNEL)?,
            shared4_pipeline: compute_pipeline(device, library, ROPE_SHARED4_KERNEL)?,
            arena,
        })
    }

    pub(crate) fn run(
        &self,
        device: &Device,
        queue: &CommandQueue,
        input: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
    ) -> Result<MetalRopeReport> {
        validate_rope_slice_f32(
            input,
            batch_count,
            token_count,
            head_count,
            rope_dim,
            position_offset,
            theta,
        )?;

        let batch_count_u32 = u32::try_from(batch_count)
            .map_err(|_| Error::backend("RoPE batch_count exceeds Metal u32 limit"))?;
        let token_count_u32 = u32::try_from(token_count)
            .map_err(|_| Error::backend("RoPE token_count exceeds Metal u32 limit"))?;
        let head_count_u32 = u32::try_from(head_count)
            .map_err(|_| Error::backend("RoPE head_count exceeds Metal u32 limit"))?;
        let rope_dim_u32 = u32::try_from(rope_dim)
            .map_err(|_| Error::backend("RoPE rope_dim exceeds Metal u32 limit"))?;
        let position_offset_u32 = u32::try_from(position_offset)
            .map_err(|_| Error::backend("RoPE position_offset exceeds Metal u32 limit"))?;

        let input_buffer = f32_buffer(device, input)?;
        let output_buffer = self.arena.empty_f32(input.len())?;
        let batch_count_buffer = self.arena.u32(batch_count_u32)?;
        let token_count_buffer = self.arena.u32(token_count_u32)?;
        let head_count_buffer = self.arena.u32(head_count_u32)?;
        let rope_dim_buffer = self.arena.u32(rope_dim_u32)?;
        let position_offset_buffer = self.arena.u32(position_offset_u32)?;
        let theta_buffer = self.arena.f32(theta)?;

        trace!(
            target: "inferno::metal",
            batch_count,
            token_count,
            head_count,
            rope_dim,
            position_offset,
            theta,
            "running native Metal RoPE"
        );

        let buffers = [
            &input_buffer,
            &output_buffer,
            &batch_count_buffer,
            &token_count_buffer,
            &head_count_buffer,
            &rope_dim_buffer,
            &position_offset_buffer,
            &theta_buffer,
        ];
        let shared_coefficients = uses_shared_coefficients(token_count, head_count, rope_dim);
        let thread_count = if shared_coefficients {
            let group_count = shared_group_count(batch_count, token_count, head_count)?;
            dispatch_1d_threadgroups(
                queue,
                &self.shared4_pipeline,
                &buffers,
                group_count,
                SHARED_THREADS_PER_GROUP,
            )?;
            group_count
                .checked_mul(SHARED_THREADS_PER_GROUP)
                .ok_or_else(|| Error::backend("shared RoPE thread count overflow"))?
        } else {
            let pair_threads = input
                .len()
                .checked_div(2)
                .ok_or_else(|| Error::backend("RoPE pair thread count overflow"))?;
            dispatch_1d(queue, &self.pair_pipeline, &buffers, pair_threads)?;
            pair_threads
        };

        let values = read_f32_buffer(&output_buffer, input.len())?;

        Ok(MetalRopeReport {
            values,
            batch_count,
            token_count,
            head_count,
            rope_dim,
            position_offset,
            theta,
            thread_count,
            shared_coefficients,
        })
    }

    /// Encodes a RoPE kernel into an open batched command buffer, reading its
    /// input from a device-resident buffer. See `BatchSlot` for the batching
    /// rules.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        _device: &Device,
        input: &Buffer,
        input_len: usize,
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
    ) -> Result<Buffer> {
        validate_rope_slice_buffer(
            input_len,
            batch_count,
            token_count,
            head_count,
            rope_dim,
            position_offset,
            theta,
        )?;
        require_f32_capacity(input, input_len, "RoPE input")?;

        let batch_count_u32 = u32::try_from(batch_count)
            .map_err(|_| Error::backend("RoPE batch_count exceeds Metal u32 limit"))?;
        let token_count_u32 = u32::try_from(token_count)
            .map_err(|_| Error::backend("RoPE token_count exceeds Metal u32 limit"))?;
        let head_count_u32 = u32::try_from(head_count)
            .map_err(|_| Error::backend("RoPE head_count exceeds Metal u32 limit"))?;
        let rope_dim_u32 = u32::try_from(rope_dim)
            .map_err(|_| Error::backend("RoPE rope_dim exceeds Metal u32 limit"))?;
        let position_offset_u32 = u32::try_from(position_offset)
            .map_err(|_| Error::backend("RoPE position_offset exceeds Metal u32 limit"))?;

        let output_buffer = self.arena.empty_f32(input_len)?;
        let batch_count_buffer = self.arena.u32(batch_count_u32)?;
        let token_count_buffer = self.arena.u32(token_count_u32)?;
        let head_count_buffer = self.arena.u32(head_count_u32)?;
        let rope_dim_buffer = self.arena.u32(rope_dim_u32)?;
        let position_offset_buffer = self.arena.u32(position_offset_u32)?;
        let theta_buffer = self.arena.f32(theta)?;

        trace!(
            target: "inferno::metal",
            batch_count,
            token_count,
            head_count,
            rope_dim,
            position_offset,
            theta,
            "encoding batched RoPE"
        );

        let buffers = [
            input,
            &output_buffer,
            &batch_count_buffer,
            &token_count_buffer,
            &head_count_buffer,
            &rope_dim_buffer,
            &position_offset_buffer,
            &theta_buffer,
        ];
        if uses_shared_coefficients(token_count, head_count, rope_dim) {
            encode_1d_threadgroups(
                command_buffer,
                &self.shared4_pipeline,
                &buffers,
                shared_group_count(batch_count, token_count, head_count)?,
                SHARED_THREADS_PER_GROUP,
            )?;
        } else {
            encode_1d(command_buffer, &self.pair_pipeline, &buffers, input_len / 2)?;
        }
        Ok(output_buffer)
    }
}

fn uses_shared_coefficients(token_count: usize, head_count: usize, rope_dim: usize) -> bool {
    token_count >= SHARED_MIN_TOKEN_COUNT
        && head_count >= SHARED_HEADS_PER_GROUP
        && rope_dim == SHARED_ROPE_DIM
}

fn shared_group_count(batch_count: usize, token_count: usize, head_count: usize) -> Result<usize> {
    batch_count
        .checked_mul(token_count)
        .and_then(|count| count.checked_mul(head_count.div_ceil(SHARED_HEADS_PER_GROUP)))
        .ok_or_else(|| Error::backend("shared RoPE threadgroup count overflow"))
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use crate::metal::Metal;

    #[test]
    fn matches_cpu_reference_for_small_slice() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch_count = 1;
        let token_count = 2;
        let head_count = 2;
        let rope_dim = 4;
        let position_offset = 3;
        let theta = 10_000.0;
        let input = (0..batch_count * token_count * head_count * rope_dim)
            .map(|index| (index as f32 + 1.0) / 10.0)
            .collect::<Vec<_>>();

        let report = metal
            .rope_slice_f32_report(
                &input,
                batch_count,
                token_count,
                head_count,
                rope_dim,
                position_offset,
                theta,
            )
            .unwrap();
        let expected = cpu_rope_slice(
            &input,
            batch_count,
            token_count,
            head_count,
            rope_dim,
            position_offset,
            theta,
        );

        assert_eq!(report.batch_count, batch_count);
        assert_eq!(report.token_count, token_count);
        assert_eq!(report.head_count, head_count);
        assert_eq!(report.rope_dim, rope_dim);
        assert_eq!(report.position_offset, position_offset);
        assert_eq!(report.thread_count, input.len() / 2);
        assert!(!report.shared_coefficients);
        assert_close(&report.values, &expected, 1e-5);
    }

    #[test]
    fn shared_head_coefficients_match_cpu_for_glm_prefill_shape() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch_count = 1;
        let token_count = 32;
        let head_count = 64;
        let rope_dim = 64;
        let position_offset = 7;
        let theta = 10_000.0;
        let input = (0..batch_count * token_count * head_count * rope_dim)
            .map(|index| ((index % 251) as f32 - 125.0) / 64.0)
            .collect::<Vec<_>>();

        let report = metal
            .rope_slice_f32_report(
                &input,
                batch_count,
                token_count,
                head_count,
                rope_dim,
                position_offset,
                theta,
            )
            .unwrap();
        let expected = cpu_rope_slice(
            &input,
            batch_count,
            token_count,
            head_count,
            rope_dim,
            position_offset,
            theta,
        );

        assert!(report.shared_coefficients);
        assert_eq!(report.thread_count, token_count * 16 * 256);
        assert_close(&report.values, &expected, 1e-5);
    }

    #[test]
    fn rejects_odd_rope_dim_before_dispatch() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let err = metal
            .rope_slice_f32(&[1.0; 3], 1, 1, 1, 3, 0, 10_000.0)
            .expect_err("odd rope_dim should fail before Metal dispatch");

        assert!(err.to_string().contains("even"));
    }

    fn native_metal_or_skip() -> Option<Metal> {
        Metal::new().ok()
    }

    fn cpu_rope_slice(
        input: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        let pair_count = rope_dim / 2;

        for batch in 0..batch_count {
            for token in 0..token_count {
                let position = position_offset + token;
                for head in 0..head_count {
                    let base = ((batch * token_count + token) * head_count + head) * rope_dim;
                    for dim in 0..rope_dim {
                        let pair_index = dim / 2;
                        let partner_dim = if dim % 2 == 0 { dim + 1 } else { dim - 1 };
                        let inv_freq = 1.0 / theta.powf(pair_index as f32 / pair_count as f32);
                        let angle = position as f32 * inv_freq;
                        let rotated = if dim % 2 == 0 {
                            -input[base + partner_dim]
                        } else {
                            input[base + partner_dim]
                        };
                        output[base + dim] =
                            input[base + dim] * angle.cos() + rotated * angle.sin();
                    }
                }
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
