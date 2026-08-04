use ::metal::{Buffer, CommandBufferRef, CommandQueue, ComputePipelineState, Device};
use common::{Error, Result};
use tracing::trace;

use crate::DeviceRopeTable;

use super::{
    arena::MetalArena,
    buffers::{f32_buffer, read_f32_buffer, require_f32_capacity, ImmutableF32BufferCache},
    command::{dispatch_1d, dispatch_1d_threadgroups, encode_1d, encode_1d_threadgroups},
    library::MetalLibrary,
    pipeline::compute_pipeline,
    validation::{validate_rope_slice_buffer, validate_rope_slice_f32},
};

const ROPE_PAIR_KERNEL: &str = "rope_slice_f32_pair_kernel";
const ROPE_SHARED4_KERNEL: &str = "rope_slice_f32_shared4_kernel";
const QK_NORM_ROPE_KERNEL: &str = "qk_rms_norm_half_split_rope_f32_kernel";
const QK_NORM_ROPE_PAIR_KERNEL: &str = "laguna_qk_rms_norm_half_split_rope_pair_f32_kernel";
const SHARED_HEADS_PER_GROUP: usize = 4;
const SHARED_THREADS_PER_GROUP: usize = 256;
const SHARED_ROPE_DIM: usize = 64;
const SHARED_MIN_TOKEN_COUNT: usize = 32;
const LAGUNA_HEAD_DIM: usize = 128;
const LAGUNA_GLOBAL_ROTARY_DIM: usize = 64;
const LAGUNA_SLIDING_ROTARY_DIM: usize = 128;
const QK_NORM_ROPE_THREADS: usize = 32;

pub(crate) struct MetalRope {
    pair_pipeline: ComputePipelineState,
    shared4_pipeline: ComputePipelineState,
    qk_norm_rope_pipeline: ComputePipelineState,
    qk_norm_rope_pair_pipeline: ComputePipelineState,
    norm_weights: ImmutableF32BufferCache,
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
            qk_norm_rope_pipeline: compute_pipeline(device, library, QK_NORM_ROPE_KERNEL)?,
            qk_norm_rope_pair_pipeline: compute_pipeline(
                device,
                library,
                QK_NORM_ROPE_PAIR_KERNEL,
            )?,
            norm_weights: ImmutableF32BufferCache::default(),
            arena,
        })
    }

    pub(crate) fn prepare_table(
        &self,
        device: &Device,
        inverse_frequency: &[f32],
        rotary_dim: usize,
        attention_factor: f32,
    ) -> Result<DeviceRopeTable> {
        validate_laguna_rope_table(inverse_frequency, rotary_dim, attention_factor)?;
        Ok(DeviceRopeTable {
            rotary_dim,
            attention_factor,
            inverse_frequency: f32_buffer(device, inverse_frequency)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_qk_norm_rope(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        input: &Buffer,
        input_len: usize,
        norm_weight: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        position_offset: usize,
        eps: f32,
        table: &DeviceRopeTable,
    ) -> Result<Buffer> {
        let row_count = validate_laguna_qk_norm_rope(
            input_len,
            norm_weight,
            batch_count,
            token_count,
            head_count,
            position_offset,
            eps,
            table,
        )?;
        require_f32_capacity(input, input_len, "Q/K RMSNorm RoPE input")?;

        let norm_weight = self.norm_weights.get(device, norm_weight)?;
        let output = self.arena.empty_f32(input_len)?;
        let row_count_buffer = self.arena.u32(as_u32(row_count, "row count")?)?;
        let token_count_buffer = self.arena.u32(as_u32(token_count, "token count")?)?;
        let head_count_buffer = self.arena.u32(as_u32(head_count, "head count")?)?;
        let head_dim_buffer = self.arena.u32(LAGUNA_HEAD_DIM as u32)?;
        let rotary_dim_buffer = self
            .arena
            .u32(as_u32(table.rotary_dim, "rotary dimension")?)?;
        let position_offset_buffer = self
            .arena
            .u32(as_u32(position_offset, "position offset")?)?;
        let eps_buffer = self.arena.f32(eps)?;
        let attention_factor_buffer = self.arena.f32(table.attention_factor)?;

        trace!(
            target: "inferno::metal",
            batch_count,
            token_count,
            head_count,
            rotary_dim = table.rotary_dim,
            position_offset,
            attention_factor = table.attention_factor,
            "encoding Laguna Q/K RMSNorm and RoPE"
        );

        encode_1d_threadgroups(
            command_buffer,
            &self.qk_norm_rope_pipeline,
            &[
                input,
                &norm_weight,
                &table.inverse_frequency,
                &output,
                &row_count_buffer,
                &token_count_buffer,
                &head_count_buffer,
                &head_dim_buffer,
                &rotary_dim_buffer,
                &position_offset_buffer,
                &eps_buffer,
                &attention_factor_buffer,
            ],
            row_count,
            QK_NORM_ROPE_THREADS,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_qk_norm_rope_pair(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        query: &Buffer,
        query_len: usize,
        key: &Buffer,
        key_len: usize,
        query_norm_weight: &[f32],
        key_norm_weight: &[f32],
        batch_count: usize,
        token_count: usize,
        query_head_count: usize,
        key_head_count: usize,
        position_offset: usize,
        eps: f32,
        table: &DeviceRopeTable,
    ) -> Result<(Buffer, Buffer)> {
        let query_rows = validate_laguna_qk_norm_rope(
            query_len,
            query_norm_weight,
            batch_count,
            token_count,
            query_head_count,
            position_offset,
            eps,
            table,
        )?;
        let key_rows = validate_laguna_qk_norm_rope(
            key_len,
            key_norm_weight,
            batch_count,
            token_count,
            key_head_count,
            position_offset,
            eps,
            table,
        )?;
        require_f32_capacity(query, query_len, "Laguna query RMSNorm RoPE input")?;
        require_f32_capacity(key, key_len, "Laguna key RMSNorm RoPE input")?;

        let query_norm_weight = self.norm_weights.get(device, query_norm_weight)?;
        let key_norm_weight = self.norm_weights.get(device, key_norm_weight)?;
        let query_output = self.arena.empty_f32(query_len)?;
        let key_output = self.arena.empty_f32(key_len)?;
        let query_rows_buffer = self.arena.u32(as_u32(query_rows, "query row count")?)?;
        let key_rows_buffer = self.arena.u32(as_u32(key_rows, "key row count")?)?;
        let token_count_buffer = self.arena.u32(as_u32(token_count, "token count")?)?;
        let query_head_count_buffer = self
            .arena
            .u32(as_u32(query_head_count, "query head count")?)?;
        let key_head_count_buffer = self.arena.u32(as_u32(key_head_count, "key head count")?)?;
        let head_dim_buffer = self.arena.u32(LAGUNA_HEAD_DIM as u32)?;
        let rotary_dim_buffer = self
            .arena
            .u32(as_u32(table.rotary_dim, "rotary dimension")?)?;
        let position_offset_buffer = self
            .arena
            .u32(as_u32(position_offset, "position offset")?)?;
        let eps_buffer = self.arena.f32(eps)?;
        let attention_factor_buffer = self.arena.f32(table.attention_factor)?;
        let combined_rows = query_rows
            .checked_add(key_rows)
            .ok_or_else(|| Error::backend("Laguna Q/K row count overflow"))?;

        encode_1d_threadgroups(
            command_buffer,
            &self.qk_norm_rope_pair_pipeline,
            &[
                query,
                key,
                &query_norm_weight,
                &key_norm_weight,
                &table.inverse_frequency,
                &query_output,
                &key_output,
                &query_rows_buffer,
                &key_rows_buffer,
                &token_count_buffer,
                &query_head_count_buffer,
                &key_head_count_buffer,
                &head_dim_buffer,
                &rotary_dim_buffer,
                &position_offset_buffer,
                &eps_buffer,
                &attention_factor_buffer,
            ],
            combined_rows,
            QK_NORM_ROPE_THREADS,
        )?;
        Ok((query_output, key_output))
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

fn validate_laguna_rope_table(
    inverse_frequency: &[f32],
    rotary_dim: usize,
    attention_factor: f32,
) -> Result<()> {
    validate_laguna_rope_metadata(rotary_dim, attention_factor)?;
    if inverse_frequency.len() != rotary_dim / 2 {
        return Err(Error::backend(format!(
            "Laguna inverse-frequency length must be {}, got {}",
            rotary_dim / 2,
            inverse_frequency.len()
        )));
    }
    if inverse_frequency
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(Error::backend(
            "Laguna inverse frequencies must be positive and finite",
        ));
    }
    Ok(())
}

fn validate_laguna_rope_metadata(rotary_dim: usize, attention_factor: f32) -> Result<()> {
    if !matches!(
        rotary_dim,
        LAGUNA_GLOBAL_ROTARY_DIM | LAGUNA_SLIDING_ROTARY_DIM
    ) {
        return Err(Error::backend(format!(
            "Laguna rotary dimension must be 64 or 128, got {rotary_dim}"
        )));
    }
    if !attention_factor.is_finite() || attention_factor <= 0.0 {
        return Err(Error::backend(format!(
            "Laguna RoPE attention factor must be positive and finite, got {attention_factor}"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_laguna_qk_norm_rope(
    input_len: usize,
    norm_weight: &[f32],
    batch_count: usize,
    token_count: usize,
    head_count: usize,
    position_offset: usize,
    eps: f32,
    table: &DeviceRopeTable,
) -> Result<usize> {
    validate_laguna_rope_metadata(table.rotary_dim, table.attention_factor)?;
    require_f32_capacity(
        &table.inverse_frequency,
        table.rotary_dim / 2,
        "Laguna inverse-frequency table",
    )?;
    if batch_count == 0 || token_count == 0 || head_count == 0 {
        return Err(Error::backend(
            "Laguna Q/K RMSNorm RoPE dimensions must be positive",
        ));
    }
    if norm_weight.len() != LAGUNA_HEAD_DIM {
        return Err(Error::backend(format!(
            "Laguna Q/K norm weight must contain {LAGUNA_HEAD_DIM} values, got {}",
            norm_weight.len()
        )));
    }
    if norm_weight.iter().any(|value| !value.is_finite()) {
        return Err(Error::backend(
            "Laguna Q/K norm weight contains a non-finite value",
        ));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(Error::backend(format!(
            "Laguna Q/K RMSNorm epsilon must be positive and finite, got {eps}"
        )));
    }
    let final_position = position_offset
        .checked_add(token_count - 1)
        .ok_or_else(|| Error::backend("Laguna RoPE position overflow"))?;
    as_u32(final_position, "final position")?;
    let row_count = batch_count
        .checked_mul(token_count)
        .and_then(|rows| rows.checked_mul(head_count))
        .ok_or_else(|| Error::backend("Laguna Q/K head row count overflow"))?;
    let expected_len = row_count
        .checked_mul(LAGUNA_HEAD_DIM)
        .ok_or_else(|| Error::backend("Laguna Q/K tensor length overflow"))?;
    if input_len != expected_len {
        return Err(Error::backend(format!(
            "Laguna Q/K input length mismatch: expected {expected_len} for [{batch_count},{token_count},{head_count},{LAGUNA_HEAD_DIM}], got {input_len}"
        )));
    }
    Ok(row_count)
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::backend(format!("Laguna RoPE {label} exceeds Metal u32 limit")))
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use common::{Device, F32Tensor};

    use crate::{metal::Metal, Backend, MetalBackend};

    use super::validate_laguna_rope_table;

    #[test]
    fn fused_laguna_qk_norm_rope_matches_global_and_sliding_references() {
        let Some(backend) = native_backend_or_skip() else {
            return;
        };
        let batch_count = 1;
        let token_count = 2;
        let head_count = 3;
        let head_dim = 128;
        let position_offset = 11;
        let eps = 1e-6;
        let input_values = (0..batch_count * token_count * head_count * head_dim)
            .map(|index| ((index % 97) as f32 - 48.0) / 31.0)
            .collect::<Vec<_>>();
        let norm_values = (0..head_dim)
            .map(|index| 0.75 + (index % 13) as f32 / 40.0)
            .collect::<Vec<_>>();
        let input = F32Tensor::new(
            input_values.clone(),
            [batch_count, token_count, head_count, head_dim],
        )
        .unwrap();
        let norm = F32Tensor::new(norm_values.clone(), [head_dim]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();

        for (rotary_dim, theta, attention_factor) in [
            (64_usize, 500_000.0_f32, 1.346_573_6_f32),
            (128_usize, 10_000.0_f32, 1.0_f32),
        ] {
            let inverse_frequency = (0..rotary_dim / 2)
                .map(|index| 1.0 / theta.powf((2 * index) as f32 / rotary_dim as f32))
                .collect::<Vec<_>>();
            let table = backend
                .prepare_rope_table(&inverse_frequency, rotary_dim, attention_factor)
                .unwrap()
                .unwrap();
            let output = backend
                .qk_rms_norm_rope_device(&input, &norm, eps, position_offset, &table)
                .unwrap()
                .unwrap();
            let actual = backend.device_download_f32_tensor(&output).unwrap();
            let expected = cpu_qk_norm_rope(
                &input_values,
                &norm_values,
                batch_count,
                token_count,
                head_count,
                position_offset,
                eps,
                &inverse_frequency,
                rotary_dim,
                attention_factor,
            );

            assert_eq!(
                actual.dims(),
                &[batch_count, token_count, head_count, head_dim]
            );
            assert_close(actual.values(), &expected, 3e-5);
        }
    }

    #[test]
    fn laguna_qk_pair_matches_two_independent_references() {
        let Some(backend) = native_backend_or_skip() else {
            return;
        };
        let batch_count = 1;
        let token_count = 2;
        let query_heads = 3;
        let key_heads = 8;
        let head_dim = 128;
        let position_offset = 7;
        let eps = 1e-6;
        let query_values = (0..batch_count * token_count * query_heads * head_dim)
            .map(|index| ((index % 43) as f32 - 21.0) / 16.0)
            .collect::<Vec<_>>();
        let key_values = (0..batch_count * token_count * key_heads * head_dim)
            .map(|index| ((index % 59) as f32 - 29.0) / 24.0)
            .collect::<Vec<_>>();
        let query_norm_values = (0..head_dim)
            .map(|index| 0.75 + (index % 5) as f32 * 0.0625)
            .collect::<Vec<_>>();
        let key_norm_values = (0..head_dim)
            .map(|index| 0.875 + (index % 7) as f32 * 0.03125)
            .collect::<Vec<_>>();
        let query = F32Tensor::new(
            query_values.clone(),
            [batch_count, token_count, query_heads, head_dim],
        )
        .unwrap();
        let key = F32Tensor::new(
            key_values.clone(),
            [batch_count, token_count, key_heads, head_dim],
        )
        .unwrap();
        let query_norm = F32Tensor::new(query_norm_values.clone(), [head_dim]).unwrap();
        let key_norm = F32Tensor::new(key_norm_values.clone(), [head_dim]).unwrap();
        let query = backend.device_upload_f32_tensor(&query).unwrap().unwrap();
        let key = backend.device_upload_f32_tensor(&key).unwrap().unwrap();
        let rotary_dim = 64;
        let theta = 500_000.0_f32;
        let attention_factor = 1.346_573_6_f32;
        let inverse_frequency = (0..rotary_dim / 2)
            .map(|index| 1.0 / theta.powf((2 * index) as f32 / rotary_dim as f32))
            .collect::<Vec<_>>();
        let table = backend
            .prepare_rope_table(&inverse_frequency, rotary_dim, attention_factor)
            .unwrap()
            .unwrap();

        let (query_output, key_output) = backend
            .laguna_qk_rms_norm_rope_pair_device(
                &query,
                &key,
                &query_norm,
                &key_norm,
                eps,
                position_offset,
                &table,
            )
            .unwrap()
            .unwrap();
        let query_output = backend.device_download_f32_tensor(&query_output).unwrap();
        let key_output = backend.device_download_f32_tensor(&key_output).unwrap();
        let expected_query = cpu_qk_norm_rope(
            &query_values,
            &query_norm_values,
            batch_count,
            token_count,
            query_heads,
            position_offset,
            eps,
            &inverse_frequency,
            rotary_dim,
            attention_factor,
        );
        let expected_key = cpu_qk_norm_rope(
            &key_values,
            &key_norm_values,
            batch_count,
            token_count,
            key_heads,
            position_offset,
            eps,
            &inverse_frequency,
            rotary_dim,
            attention_factor,
        );
        assert_close(query_output.values(), &expected_query, 3e-5);
        assert_close(key_output.values(), &expected_key, 3e-5);
    }

    #[test]
    fn laguna_rope_table_rejects_non_checkpoint_layouts() {
        assert!(validate_laguna_rope_table(&[1.0; 31], 64, 1.0).is_err());
        assert!(validate_laguna_rope_table(&[1.0; 48], 96, 1.0).is_err());
        assert!(validate_laguna_rope_table(&[1.0; 32], 64, 0.0).is_err());
    }

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

    fn native_backend_or_skip() -> Option<MetalBackend> {
        let backend = MetalBackend::from_device(Device::Metal).ok()?;
        backend.device_values_supported().then_some(backend)
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_qk_norm_rope(
        input: &[f32],
        norm_weight: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        position_offset: usize,
        eps: f32,
        inverse_frequency: &[f32],
        rotary_dim: usize,
        attention_factor: f32,
    ) -> Vec<f32> {
        let head_dim = norm_weight.len();
        let mut output = vec![0.0; input.len()];
        let row_count = batch_count * token_count * head_count;
        let rotary_half = rotary_dim / 2;
        for row in 0..row_count {
            let base = row * head_dim;
            let token = (row / head_count) % token_count;
            let position = (position_offset + token) as f32;
            let inverse_rms = 1.0
                / (input[base..base + head_dim]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    / head_dim as f32
                    + eps)
                    .sqrt();
            for dim in 0..head_dim {
                let normalized = input[base + dim] * norm_weight[dim] * inverse_rms;
                if dim >= rotary_dim {
                    output[base + dim] = normalized;
                    continue;
                }
                let pair_dim = if dim < rotary_half {
                    dim + rotary_half
                } else {
                    dim - rotary_half
                };
                let pair = input[base + pair_dim] * norm_weight[pair_dim] * inverse_rms;
                let rotated = if dim < rotary_half { -pair } else { pair };
                let angle = position * inverse_frequency[dim % rotary_half];
                output[base + dim] =
                    attention_factor * (normalized * angle.cos() + rotated * angle.sin());
            }
        }
        output
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
