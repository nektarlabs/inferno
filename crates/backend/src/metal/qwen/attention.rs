use ::metal::{Buffer, ComputePipelineState, Device, MTLResourceOptions};
use common::{Error, Result};

use crate::{DeviceQwenBf16Tensor, QwenFullAttentionCache};

use super::super::{
    arena::MetalArena,
    buffers::{require_byte_capacity, require_f16_capacity},
    command::{encode_1d_threadgroups_args, KernelArg},
    library::MetalLibrary,
    pipeline::compute_pipeline,
};

const KERNEL_SOURCE: &str = include_str!("kernels/attention.metal");
const QUERY_KERNEL: &str = "qwen_query_norm_rope_gate_kernel";
const KEY_KERNEL: &str = "qwen_key_norm_rope_append_kernel";
const VALUE_KERNEL: &str = "qwen_value_append_kernel";
const ATTENTION_KERNEL: &str = "qwen_online_causal_gqa_kernel";
const QUERY_HEADS: usize = 24;
const KEY_VALUE_HEADS: usize = 4;
const HEAD_DIM: usize = 256;
const QUERY_GATE_WIDTH: usize = QUERY_HEADS * HEAD_DIM * 2;
const KEY_VALUE_WIDTH: usize = KEY_VALUE_HEADS * HEAD_DIM;
const SIMD_LANES: usize = 32;

#[derive(Clone, Copy)]
struct Bf16Binding<'a> {
    buffer: &'a Buffer,
    byte_offset: usize,
    len: usize,
}

pub(super) struct MetalQwenAttention {
    query: ComputePipelineState,
    key: ComputePipelineState,
    value: ComputePipelineState,
    attention: ComputePipelineState,
    arena: MetalArena,
}

impl MetalQwenAttention {
    pub(super) fn new(device: &Device, arena: MetalArena) -> Result<Self> {
        let library = MetalLibrary::compile_source(device, KERNEL_SOURCE)?;
        let query = compute_pipeline(device, &library, QUERY_KERNEL)?;
        let key = compute_pipeline(device, &library, KEY_KERNEL)?;
        let value = compute_pipeline(device, &library, VALUE_KERNEL)?;
        let attention = compute_pipeline(device, &library, ATTENTION_KERNEL)?;
        for (pipeline, label) in [
            (&query, QUERY_KERNEL),
            (&key, KEY_KERNEL),
            (&attention, ATTENTION_KERNEL),
        ] {
            if pipeline.thread_execution_width() as usize != SIMD_LANES {
                return Err(Error::backend(format!(
                    "{label} requires a {SIMD_LANES}-lane SIMD group"
                )));
            }
        }
        Ok(Self {
            query,
            key,
            value,
            attention,
            arena,
        })
    }

    pub(super) fn create_cache(
        &self,
        device: &Device,
        batch: usize,
        capacity_tokens: usize,
    ) -> Result<QwenFullAttentionCache> {
        if batch == 0 || capacity_tokens == 0 {
            return Err(Error::cache(format!(
                "Qwen KV cache dimensions must be positive, got [{batch},{capacity_tokens}]"
            )));
        }
        let values = batch
            .checked_mul(capacity_tokens)
            .and_then(|count| count.checked_mul(KEY_VALUE_WIDTH))
            .ok_or_else(|| Error::cache("Qwen KV cache element count overflow"))?;
        let bytes = values
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| Error::cache("Qwen KV cache byte count overflow"))?;
        let options = MTLResourceOptions::StorageModeShared
            .union(MTLResourceOptions::HazardTrackingModeTracked);
        let key = device.new_buffer(bytes as u64, options);
        key.set_label("qwen full-attention key cache");
        let value = device.new_buffer(bytes as u64, options);
        value.set_label("qwen full-attention value cache");
        Ok(QwenFullAttentionCache {
            batch,
            capacity_tokens,
            length: 0,
            kv_heads: KEY_VALUE_HEADS,
            head_dim: HEAD_DIM,
            key,
            value,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode(
        &self,
        command_buffer: &::metal::CommandBufferRef,
        query_gate: &Buffer,
        key: &Buffer,
        value: &Buffer,
        query_norm: &DeviceQwenBf16Tensor,
        key_norm: &DeviceQwenBf16Tensor,
        cache: &QwenFullAttentionCache,
        row_count: usize,
        sequence_length: usize,
        rope_theta: f32,
        rotary_dim: usize,
        norm_weight_has_unit_offset: bool,
    ) -> Result<Buffer> {
        self.encode_raw(
            command_buffer,
            query_gate,
            key,
            value,
            Bf16Binding {
                buffer: &query_norm.buffer,
                byte_offset: query_norm.byte_offset,
                len: query_norm.element_count()?,
            },
            Bf16Binding {
                buffer: &key_norm.buffer,
                byte_offset: key_norm.byte_offset,
                len: key_norm.element_count()?,
            },
            cache,
            row_count,
            sequence_length,
            rope_theta,
            rotary_dim,
            norm_weight_has_unit_offset,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_raw(
        &self,
        command_buffer: &::metal::CommandBufferRef,
        query_gate_projection: &Buffer,
        key_projection: &Buffer,
        value_projection: &Buffer,
        query_norm: Bf16Binding<'_>,
        key_norm: Bf16Binding<'_>,
        cache: &QwenFullAttentionCache,
        row_count: usize,
        sequence_length: usize,
        rope_theta: f32,
        rotary_dim: usize,
        norm_weight_has_unit_offset: bool,
    ) -> Result<Buffer> {
        validate_execution(cache, row_count, sequence_length, rope_theta, rotary_dim)?;
        require_f16_capacity(
            query_gate_projection,
            row_count * QUERY_GATE_WIDTH,
            "Qwen query/gate projection",
        )?;
        require_f16_capacity(
            key_projection,
            row_count * KEY_VALUE_WIDTH,
            "Qwen key projection",
        )?;
        require_f16_capacity(
            value_projection,
            row_count * KEY_VALUE_WIDTH,
            "Qwen value projection",
        )?;
        require_norm_binding(query_norm, "Qwen query norm")?;
        require_norm_binding(key_norm, "Qwen key norm")?;

        let query_values = row_count
            .checked_mul(QUERY_HEADS * HEAD_DIM)
            .ok_or_else(|| Error::backend("Qwen query value count overflow"))?;
        let query = self.arena.empty_f16(query_values)?;
        let gate = self.arena.empty_f16(query_values)?;
        let output = self.arena.empty_f16(query_values)?;
        let row_count = as_u32(row_count, "row count")?;
        let sequence_length = as_u32(sequence_length, "sequence length")?;
        let capacity_tokens = as_u32(cache.capacity_tokens, "KV capacity")?;
        let position_start = as_u32(cache.length, "KV position")?;

        encode_1d_threadgroups_args(
            command_buffer,
            &self.query,
            &[
                KernelArg::Buffer(query_gate_projection),
                KernelArg::BufferOffset(query_norm.buffer, query_norm.byte_offset),
                KernelArg::Buffer(&query),
                KernelArg::Buffer(&gate),
                KernelArg::U32(row_count),
                KernelArg::U32(QUERY_HEADS as u32),
                KernelArg::U32(HEAD_DIM as u32),
                KernelArg::U32(rotary_dim as u32),
                KernelArg::U32(position_start),
                KernelArg::U32(sequence_length),
                KernelArg::F32(rope_theta),
                KernelArg::U32(u32::from(norm_weight_has_unit_offset)),
            ],
            row_count as usize * QUERY_HEADS,
            SIMD_LANES,
        )?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.key,
            &[
                KernelArg::Buffer(key_projection),
                KernelArg::BufferOffset(key_norm.buffer, key_norm.byte_offset),
                KernelArg::Buffer(&cache.key),
                KernelArg::U32(row_count),
                KernelArg::U32(KEY_VALUE_HEADS as u32),
                KernelArg::U32(HEAD_DIM as u32),
                KernelArg::U32(rotary_dim as u32),
                KernelArg::U32(capacity_tokens),
                KernelArg::U32(position_start),
                KernelArg::U32(sequence_length),
                KernelArg::F32(rope_theta),
                KernelArg::U32(u32::from(norm_weight_has_unit_offset)),
            ],
            row_count as usize * KEY_VALUE_HEADS,
            SIMD_LANES,
        )?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.value,
            &[
                KernelArg::Buffer(value_projection),
                KernelArg::Buffer(&cache.value),
                KernelArg::U32(row_count),
                KernelArg::U32(KEY_VALUE_WIDTH as u32),
                KernelArg::U32(capacity_tokens),
                KernelArg::U32(position_start),
                KernelArg::U32(sequence_length),
            ],
            (row_count as usize * KEY_VALUE_WIDTH).div_ceil(256),
            256,
        )?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.attention,
            &[
                KernelArg::Buffer(&query),
                KernelArg::Buffer(&gate),
                KernelArg::Buffer(&cache.key),
                KernelArg::Buffer(&cache.value),
                KernelArg::Buffer(&output),
                KernelArg::U32(row_count),
                KernelArg::U32(sequence_length),
                KernelArg::U32(QUERY_HEADS as u32),
                KernelArg::U32(KEY_VALUE_HEADS as u32),
                KernelArg::U32(HEAD_DIM as u32),
                KernelArg::U32(capacity_tokens),
                KernelArg::U32(position_start),
            ],
            row_count as usize * QUERY_HEADS,
            SIMD_LANES,
        )?;
        Ok(output)
    }
}

fn validate_execution(
    cache: &QwenFullAttentionCache,
    row_count: usize,
    sequence_length: usize,
    rope_theta: f32,
    rotary_dim: usize,
) -> Result<()> {
    if row_count == 0
        || sequence_length == 0
        || row_count != cache.batch * sequence_length
        || !rope_theta.is_finite()
        || rope_theta <= 0.0
        || rotary_dim == 0
        || rotary_dim > HEAD_DIM
        || !rotary_dim.is_multiple_of(2)
    {
        return Err(Error::backend(format!(
            "invalid Qwen attention rows={row_count}, sequence={sequence_length}, batch={}, theta={rope_theta}, rotary_dim={rotary_dim}",
            cache.batch
        )));
    }
    if cache.kv_heads != KEY_VALUE_HEADS || cache.head_dim != HEAD_DIM {
        return Err(Error::cache(
            "Qwen KV cache has incompatible head dimensions",
        ));
    }
    if cache.length + sequence_length > cache.capacity_tokens {
        return Err(Error::cache(format!(
            "Qwen KV append {} + {sequence_length} exceeds capacity {}",
            cache.length, cache.capacity_tokens
        )));
    }
    Ok(())
}

fn require_norm_binding(binding: Bf16Binding<'_>, label: &str) -> Result<()> {
    if binding.len != HEAD_DIM {
        return Err(Error::backend(format!(
            "{label} must contain {HEAD_DIM} values, got {}",
            binding.len
        )));
    }
    let end = binding
        .byte_offset
        .checked_add(HEAD_DIM * std::mem::size_of::<u16>())
        .ok_or_else(|| Error::backend(format!("{label} range overflow")))?;
    require_byte_capacity(binding.buffer, end, label)
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::backend(format!("Qwen {label} exceeds u32")))
}

#[cfg(test)]
mod tests {
    use std::slice;

    use ::metal::{MTLCommandBufferStatus, MTLResourceOptions};

    use super::*;

    #[test]
    fn one_token_full_attention_matches_exact_gated_value() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let Ok(arena) = MetalArena::new(&device) else {
            return;
        };
        let Ok(attention) = MetalQwenAttention::new(&device, arena) else {
            return;
        };
        let cache = attention.create_cache(&device, 1, 2).unwrap();
        let mut query_gate_values = vec![0.0_f32; QUERY_GATE_WIDTH];
        for head in 0..QUERY_HEADS {
            let start = head * HEAD_DIM * 2;
            query_gate_values[start..start + HEAD_DIM].fill(1.0);
        }
        let key_values = vec![1.0_f32; KEY_VALUE_WIDTH];
        let value_values = vec![3.0_f32; KEY_VALUE_WIDTH];
        let norm_values = vec![0.0_f32; HEAD_DIM];
        let options = MTLResourceOptions::StorageModeShared;
        let query_gate_bytes = bf16_bytes(&query_gate_values);
        let key_bytes = bf16_bytes(&key_values);
        let value_bytes = bf16_bytes(&value_values);
        let norm_bytes = bf16_bytes(&norm_values);
        let query_gate = device.new_buffer_with_data(
            query_gate_bytes.as_ptr().cast(),
            query_gate_bytes.len() as u64,
            options,
        );
        let key =
            device.new_buffer_with_data(key_bytes.as_ptr().cast(), key_bytes.len() as u64, options);
        let value = device.new_buffer_with_data(
            value_bytes.as_ptr().cast(),
            value_bytes.len() as u64,
            options,
        );
        let norm = device.new_buffer_with_data(
            norm_bytes.as_ptr().cast(),
            norm_bytes.len() as u64,
            options,
        );
        let norm = Bf16Binding {
            buffer: &norm,
            byte_offset: 0,
            len: HEAD_DIM,
        };

        let queue = device.new_command_queue();
        let command_buffer = queue.new_command_buffer();
        let output = attention
            .encode_raw(
                command_buffer,
                &query_gate,
                &key,
                &value,
                norm,
                norm,
                &cache,
                1,
                1,
                10_000_000.0,
                64,
                true,
            )
            .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);

        let actual = read_bf16(&output, QUERY_HEADS * HEAD_DIM);
        assert!(actual
            .iter()
            .all(|value| value.to_bits() == 1.5_f32.to_bits()));
    }

    #[test]
    fn prefill_then_decode_matches_cpu_reference() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let Ok(arena) = MetalArena::new(&device) else {
            return;
        };
        let Ok(attention) = MetalQwenAttention::new(&device, arena) else {
            return;
        };
        let mut cache = attention.create_cache(&device, 1, 3).unwrap();
        let norm_values = (0..HEAD_DIM)
            .map(|dim| (dim as f32 % 11.0 - 5.0) * 0.002)
            .collect::<Vec<_>>();
        let norm_bytes = bf16_bytes(&norm_values);
        let options = MTLResourceOptions::StorageModeShared;
        let norm = device.new_buffer_with_data(
            norm_bytes.as_ptr().cast(),
            norm_bytes.len() as u64,
            options,
        );
        let norm = Bf16Binding {
            buffer: &norm,
            byte_offset: 0,
            len: HEAD_DIM,
        };
        let queue = device.new_command_queue();
        let mut cpu_key_cache = vec![0.0_f32; 3 * KEY_VALUE_WIDTH];
        let mut cpu_value_cache = vec![0.0_f32; 3 * KEY_VALUE_WIDTH];

        let prefill_query_gate = projection_values(2, QUERY_GATE_WIDTH, 0.013, -0.4);
        let prefill_key = projection_values(2, KEY_VALUE_WIDTH, 0.017, -0.2);
        let prefill_value = projection_values(2, KEY_VALUE_WIDTH, 0.019, 0.3);
        let prefill_output = run_attention(
            &device,
            &queue,
            &attention,
            &cache,
            norm,
            &prefill_query_gate,
            &prefill_key,
            &prefill_value,
            2,
        );
        let expected_prefill = cpu_attention(
            &prefill_query_gate,
            &prefill_key,
            &prefill_value,
            &norm_values,
            &mut cpu_key_cache,
            &mut cpu_value_cache,
            0,
            2,
        );
        assert_close_bf16(&prefill_output, &expected_prefill);
        cache.length = 2;

        let decode_query_gate = projection_values(1, QUERY_GATE_WIDTH, 0.023, -0.1);
        let decode_key = projection_values(1, KEY_VALUE_WIDTH, 0.029, 0.2);
        let decode_value = projection_values(1, KEY_VALUE_WIDTH, 0.031, -0.3);
        let decode_output = run_attention(
            &device,
            &queue,
            &attention,
            &cache,
            norm,
            &decode_query_gate,
            &decode_key,
            &decode_value,
            1,
        );
        let expected_decode = cpu_attention(
            &decode_query_gate,
            &decode_key,
            &decode_value,
            &norm_values,
            &mut cpu_key_cache,
            &mut cpu_value_cache,
            2,
            1,
        );
        assert_close_bf16(&decode_output, &expected_decode);
    }

    #[allow(clippy::too_many_arguments)]
    fn run_attention(
        device: &Device,
        queue: &::metal::CommandQueueRef,
        attention: &MetalQwenAttention,
        cache: &QwenFullAttentionCache,
        norm: Bf16Binding<'_>,
        query_gate_values: &[f32],
        key_values: &[f32],
        value_values: &[f32],
        sequence_length: usize,
    ) -> Vec<f32> {
        let options = MTLResourceOptions::StorageModeShared;
        let query_gate_bytes = bf16_bytes(query_gate_values);
        let key_bytes = bf16_bytes(key_values);
        let value_bytes = bf16_bytes(value_values);
        let query_gate = device.new_buffer_with_data(
            query_gate_bytes.as_ptr().cast(),
            query_gate_bytes.len() as u64,
            options,
        );
        let key =
            device.new_buffer_with_data(key_bytes.as_ptr().cast(), key_bytes.len() as u64, options);
        let value = device.new_buffer_with_data(
            value_bytes.as_ptr().cast(),
            value_bytes.len() as u64,
            options,
        );
        let command_buffer = queue.new_command_buffer();
        let output = attention
            .encode_raw(
                command_buffer,
                &query_gate,
                &key,
                &value,
                norm,
                norm,
                cache,
                sequence_length,
                sequence_length,
                10_000_000.0,
                64,
                true,
            )
            .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        read_bf16(&output, sequence_length * QUERY_HEADS * HEAD_DIM)
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu_attention(
        query_gate_values: &[f32],
        key_values: &[f32],
        value_values: &[f32],
        norm_values: &[f32],
        key_cache: &mut [f32],
        value_cache: &mut [f32],
        position_start: usize,
        sequence_length: usize,
    ) -> Vec<f32> {
        let query_gate_values = round_bf16_values(query_gate_values);
        let key_values = round_bf16_values(key_values);
        let value_values = round_bf16_values(value_values);
        let norm_values = round_bf16_values(norm_values);
        let mut query = vec![0.0_f32; sequence_length * QUERY_HEADS * HEAD_DIM];

        for token in 0..sequence_length {
            for head in 0..QUERY_HEADS {
                let source_start = (token * QUERY_HEADS + head) * HEAD_DIM * 2;
                let output_start = (token * QUERY_HEADS + head) * HEAD_DIM;
                let normalized = normalized_head(
                    &query_gate_values[source_start..source_start + HEAD_DIM],
                    &norm_values,
                );
                let rotated = rope_head(&normalized, position_start + token);
                query[output_start..output_start + HEAD_DIM].copy_from_slice(&rotated);
            }
            for head in 0..KEY_VALUE_HEADS {
                let source_start = (token * KEY_VALUE_HEADS + head) * HEAD_DIM;
                let cache_start = ((position_start + token) * KEY_VALUE_HEADS + head) * HEAD_DIM;
                let normalized = normalized_head(
                    &key_values[source_start..source_start + HEAD_DIM],
                    &norm_values,
                );
                let rotated = rope_head(&normalized, position_start + token);
                key_cache[cache_start..cache_start + HEAD_DIM].copy_from_slice(&rotated);
                value_cache[cache_start..cache_start + HEAD_DIM]
                    .copy_from_slice(&value_values[source_start..source_start + HEAD_DIM]);
            }
        }

        let mut output = vec![0.0_f32; query.len()];
        for token in 0..sequence_length {
            let key_limit = position_start + token + 1;
            for query_head in 0..QUERY_HEADS {
                let kv_head = query_head / (QUERY_HEADS / KEY_VALUE_HEADS);
                let query_start = (token * QUERY_HEADS + query_head) * HEAD_DIM;
                let source_start = (token * QUERY_HEADS + query_head) * HEAD_DIM * 2;
                let scores = (0..key_limit)
                    .map(|key_token| {
                        let cache_start = (key_token * KEY_VALUE_HEADS + kv_head) * HEAD_DIM;
                        query[query_start..query_start + HEAD_DIM]
                            .iter()
                            .zip(&key_cache[cache_start..cache_start + HEAD_DIM])
                            .map(|(query, key)| query * key)
                            .sum::<f32>()
                            / (HEAD_DIM as f32).sqrt()
                    })
                    .collect::<Vec<_>>();
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator = scores
                    .iter()
                    .map(|score| (score - maximum).exp())
                    .sum::<f32>();
                for dim in 0..HEAD_DIM {
                    let attended = scores
                        .iter()
                        .enumerate()
                        .map(|(key_token, score)| {
                            let cache_index =
                                (key_token * KEY_VALUE_HEADS + kv_head) * HEAD_DIM + dim;
                            (score - maximum).exp() / denominator * value_cache[cache_index]
                        })
                        .sum::<f32>();
                    let gate = query_gate_values[source_start + HEAD_DIM + dim];
                    output[query_start + dim] = round_bf16(attended / (1.0 + (-gate).exp()));
                }
            }
        }
        output
    }

    fn normalized_head(values: &[f32], weights: &[f32]) -> Vec<f32> {
        let inverse_rms = (values.iter().map(|value| value * value).sum::<f32>() / HEAD_DIM as f32
            + 1.0e-6)
            .sqrt()
            .recip();
        values
            .iter()
            .zip(weights)
            .map(|(value, weight)| {
                let normalized = round_bf16(value * inverse_rms);
                round_bf16(normalized * (1.0 + weight))
            })
            .collect()
    }

    fn rope_head(values: &[f32], position: usize) -> Vec<f32> {
        let mut output = values.to_vec();
        for dim in 0..64 {
            let half = 32;
            let pair_dim = if dim < half { dim + half } else { dim - half };
            let frequency_index = dim % half;
            let exponent = -2.0 * frequency_index as f32 / 64.0;
            let angle = position as f32 * 10_000_000.0_f32.powf(exponent);
            let cosine = angle.cos();
            let sine = angle.sin();
            let rotated = if dim < half {
                -values[pair_dim]
            } else {
                values[pair_dim]
            };
            output[dim] = round_bf16(values[dim] * cosine + rotated * sine);
        }
        output
    }

    fn projection_values(rows: usize, width: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..rows * width)
            .map(|index| ((index % 37) as f32 - 18.0) * scale + offset)
            .collect()
    }

    fn round_bf16_values(values: &[f32]) -> Vec<f32> {
        values.iter().copied().map(round_bf16).collect()
    }

    fn round_bf16(value: f32) -> f32 {
        f32::from_bits(u32::from(round_bf16_bits(value)) << 16)
    }

    fn assert_close_bf16(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 0.015625,
                "BF16 mismatch at {index}: actual={actual}, expected={expected}"
            );
        }
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| round_bf16_bits(*value).to_le_bytes())
            .collect()
    }

    fn read_bf16(buffer: &Buffer, len: usize) -> Vec<f32> {
        let values = unsafe { slice::from_raw_parts(buffer.contents().cast::<u16>(), len) };
        values
            .iter()
            .map(|bits| f32::from_bits(u32::from(*bits) << 16))
            .collect()
    }

    fn round_bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }
}
