use ::metal::{Buffer, ComputePipelineState, Device, MTLResourceOptions};
use common::{Error, Result};

use crate::{DeviceQwenBf16Tensor, QwenLinearAttentionCache};

use super::delta_prefill::MetalQwenDeltaPrefill;

use super::super::{
    arena::MetalArena,
    buffers::{require_byte_capacity, require_f16_capacity},
    command::{encode_1d_threadgroups_args, KernelArg},
    library::MetalLibrary,
    pipeline::compute_pipeline,
};

const KERNEL_SOURCE: &str = include_str!("kernels/delta.metal");
const CONV_KERNEL: &str = "qwen_delta_causal_conv_kernel";
const RECURRENT_KERNEL: &str = "qwen_delta_recurrent_kernel";
const RESTORE_KERNEL: &str = "qwen_delta_restore_checkpoint_kernel";
const CONV_CHANNELS: usize = 10_240;
const CONV_HISTORY: usize = 3;
const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const HEAD_DIM: usize = 128;
const VALUE_WIDTH: usize = VALUE_HEADS * HEAD_DIM;
const SIMD_LANES: usize = 32;
const RECURRENT_THREADS: usize = 128;
const SPECULATIVE_CHECKPOINTS: usize = 8;

#[derive(Clone, Copy)]
struct Bf16Binding<'a> {
    buffer: &'a Buffer,
    byte_offset: usize,
    shape: &'a [usize],
}

impl<'a> From<&'a DeviceQwenBf16Tensor> for Bf16Binding<'a> {
    fn from(tensor: &'a DeviceQwenBf16Tensor) -> Self {
        Self {
            buffer: &tensor.buffer,
            byte_offset: tensor.byte_offset,
            shape: &tensor.shape,
        }
    }
}

pub(super) struct MetalQwenDelta {
    conv: ComputePipelineState,
    recurrent: ComputePipelineState,
    prefill_recurrent: MetalQwenDeltaPrefill,
    restore: ComputePipelineState,
    arena: MetalArena,
}

impl MetalQwenDelta {
    pub(super) fn new(device: &Device, arena: MetalArena) -> Result<Self> {
        let library = MetalLibrary::compile_source(device, KERNEL_SOURCE)?;
        let conv = compute_pipeline(device, &library, CONV_KERNEL)?;
        let recurrent = compute_pipeline(device, &library, RECURRENT_KERNEL)?;
        let prefill_recurrent = MetalQwenDeltaPrefill::new(device)?;
        let restore = compute_pipeline(device, &library, RESTORE_KERNEL)?;
        if recurrent.thread_execution_width() as usize != SIMD_LANES {
            return Err(Error::backend(format!(
                "{RECURRENT_KERNEL} requires a {SIMD_LANES}-lane SIMD group"
            )));
        }
        Ok(Self {
            conv,
            recurrent,
            prefill_recurrent,
            restore,
            arena,
        })
    }

    pub(super) fn create_cache(
        &self,
        device: &Device,
        batch: usize,
        with_checkpoint: bool,
    ) -> Result<QwenLinearAttentionCache> {
        if batch == 0 {
            return Err(Error::cache("Qwen linear-attention batch must be positive"));
        }
        let conv_values = batch
            .checked_mul(CONV_CHANNELS * CONV_HISTORY)
            .ok_or_else(|| Error::cache("Qwen convolution-state size overflow"))?;
        let recurrent_values = batch
            .checked_mul(VALUE_HEADS * HEAD_DIM * HEAD_DIM)
            .ok_or_else(|| Error::cache("Qwen recurrent-state size overflow"))?;
        let conv_bytes = conv_values * std::mem::size_of::<u16>();
        let recurrent_bytes = recurrent_values * std::mem::size_of::<f32>();
        let options = MTLResourceOptions::StorageModeShared
            .union(MTLResourceOptions::HazardTrackingModeTracked);
        let conv_state = device.new_buffer(conv_bytes as u64, options);
        conv_state.set_label("qwen delta convolution state");
        let recurrent_state = device.new_buffer(recurrent_bytes as u64, options);
        recurrent_state.set_label("qwen delta recurrent state");
        let checkpoint_capacity = if with_checkpoint {
            SPECULATIVE_CHECKPOINTS
        } else {
            0
        };
        let checkpoint_conv_state = with_checkpoint.then(|| {
            let buffer = device.new_buffer((conv_bytes * checkpoint_capacity) as u64, options);
            buffer.set_label("qwen delta convolution checkpoint");
            buffer
        });
        let checkpoint_recurrent_state = with_checkpoint.then(|| {
            let buffer = device.new_buffer((recurrent_bytes * checkpoint_capacity) as u64, options);
            buffer.set_label("qwen delta recurrent checkpoint");
            buffer
        });
        unsafe {
            std::ptr::write_bytes(
                conv_state.contents().cast::<u8>(),
                0,
                conv_state.length() as usize,
            );
            std::ptr::write_bytes(
                recurrent_state.contents().cast::<u8>(),
                0,
                recurrent_state.length() as usize,
            );
        }
        Ok(QwenLinearAttentionCache {
            batch,
            processed_tokens: 0,
            conv_channels: CONV_CHANNELS,
            conv_history: CONV_HISTORY,
            value_heads: VALUE_HEADS,
            key_head_dim: HEAD_DIM,
            value_head_dim: HEAD_DIM,
            conv_state,
            recurrent_state,
            checkpoint_conv_state,
            checkpoint_recurrent_state,
            checkpoint_capacity,
            checkpoint_valid_rows: 0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode(
        &self,
        command_buffer: &::metal::CommandBufferRef,
        mixed_qkv: &Buffer,
        gate: &Buffer,
        input_a: &Buffer,
        input_b: &Buffer,
        conv1d: &DeviceQwenBf16Tensor,
        a_log: &DeviceQwenBf16Tensor,
        dt_bias: &DeviceQwenBf16Tensor,
        norm: &DeviceQwenBf16Tensor,
        cache: &QwenLinearAttentionCache,
        sequence_length: usize,
        eps: f32,
    ) -> Result<Buffer> {
        self.encode_raw(
            command_buffer,
            mixed_qkv,
            gate,
            input_a,
            input_b,
            Bf16Binding::from(conv1d),
            Bf16Binding::from(a_log),
            Bf16Binding::from(dt_bias),
            Bf16Binding::from(norm),
            cache,
            sequence_length,
            eps,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_raw(
        &self,
        command_buffer: &::metal::CommandBufferRef,
        mixed_qkv: &Buffer,
        gate: &Buffer,
        input_a: &Buffer,
        input_b: &Buffer,
        conv1d: Bf16Binding<'_>,
        a_log: Bf16Binding<'_>,
        dt_bias: Bf16Binding<'_>,
        norm: Bf16Binding<'_>,
        cache: &QwenLinearAttentionCache,
        sequence_length: usize,
        eps: f32,
    ) -> Result<Buffer> {
        validate_execution(cache, sequence_length, eps)?;
        let rows = cache.batch * sequence_length;
        require_f16_capacity(mixed_qkv, rows * CONV_CHANNELS, "Qwen mixed QKV")?;
        require_f16_capacity(gate, rows * VALUE_WIDTH, "Qwen delta gate")?;
        require_f16_capacity(input_a, rows * VALUE_HEADS, "Qwen delta input A")?;
        require_f16_capacity(input_b, rows * VALUE_HEADS, "Qwen delta input B")?;
        require_conv1d_binding(conv1d)?;
        require_binding(a_log, &[VALUE_HEADS], "Qwen A_log")?;
        require_binding(dt_bias, &[VALUE_HEADS], "Qwen dt_bias")?;
        require_binding(norm, &[HEAD_DIM], "Qwen delta norm")?;

        let convolved = self.arena.empty_f16(rows * CONV_CHANNELS)?;
        let output = self.arena.empty_f16(rows * VALUE_WIDTH)?;
        let checkpoint_conv_state = cache
            .checkpoint_conv_state
            .as_ref()
            .unwrap_or(&cache.conv_state);
        let checkpoint_recurrent_state = cache
            .checkpoint_recurrent_state
            .as_ref()
            .unwrap_or(&cache.recurrent_state);
        let checkpoint_capacity = cache
            .checkpoint_capacity
            .min(sequence_length.saturating_sub(1));
        encode_1d_threadgroups_args(
            command_buffer,
            &self.conv,
            &[
                KernelArg::Buffer(mixed_qkv),
                KernelArg::BufferOffset(conv1d.buffer, conv1d.byte_offset),
                KernelArg::Buffer(&cache.conv_state),
                KernelArg::Buffer(checkpoint_conv_state),
                KernelArg::Buffer(&convolved),
                KernelArg::U32(as_u32(cache.batch, "batch")?),
                KernelArg::U32(as_u32(sequence_length, "sequence length")?),
                KernelArg::U32(CONV_CHANNELS as u32),
                KernelArg::U32(as_u32(checkpoint_capacity, "checkpoint capacity")?),
            ],
            (cache.batch * CONV_CHANNELS).div_ceil(256),
            256,
        )?;
        let prepared = if MetalQwenDeltaPrefill::supports(sequence_length) {
            Some(self.prepare_prefill_qk(command_buffer, &convolved, rows)?)
        } else {
            None
        };
        let recurrent_pipeline = if MetalQwenDeltaPrefill::supports(sequence_length) {
            self.prefill_recurrent.pipeline()
        } else {
            &self.recurrent
        };
        encode_1d_threadgroups_args(
            command_buffer,
            recurrent_pipeline,
            &[
                KernelArg::Buffer(&convolved),
                KernelArg::Buffer(gate),
                KernelArg::Buffer(input_a),
                KernelArg::Buffer(input_b),
                KernelArg::BufferOffset(a_log.buffer, a_log.byte_offset),
                KernelArg::BufferOffset(dt_bias.buffer, dt_bias.byte_offset),
                match prepared.as_ref() {
                    Some(qk) => KernelArg::Buffer(qk),
                    None => KernelArg::BufferOffset(norm.buffer, norm.byte_offset),
                },
                KernelArg::Buffer(&cache.recurrent_state),
                KernelArg::Buffer(checkpoint_recurrent_state),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(cache.batch, "batch")?),
                KernelArg::U32(as_u32(sequence_length, "sequence length")?),
                KernelArg::U32(KEY_HEADS as u32),
                KernelArg::U32(VALUE_HEADS as u32),
                KernelArg::U32(HEAD_DIM as u32),
                KernelArg::F32(eps),
                KernelArg::U32(as_u32(checkpoint_capacity, "checkpoint capacity")?),
            ],
            cache.batch * VALUE_HEADS,
            RECURRENT_THREADS,
        )?;
        if MetalQwenDeltaPrefill::supports(sequence_length) {
            encode_1d_threadgroups_args(
                command_buffer,
                self.prefill_recurrent.output_norm_pipeline(),
                &[
                    KernelArg::Buffer(&output),
                    KernelArg::Buffer(gate),
                    KernelArg::BufferOffset(norm.buffer, norm.byte_offset),
                    KernelArg::F32(eps),
                ],
                rows * VALUE_HEADS,
                RECURRENT_THREADS,
            )?;
        }
        Ok(output)
    }

    fn prepare_prefill_qk(
        &self,
        command_buffer: &::metal::CommandBufferRef,
        mixed_qkv: &Buffer,
        rows: usize,
    ) -> Result<Buffer> {
        let qk = self.arena.empty_f16(rows * KEY_HEADS * HEAD_DIM * 2)?;
        encode_1d_threadgroups_args(
            command_buffer,
            self.prefill_recurrent.prepare_pipeline(),
            &[
                KernelArg::Buffer(mixed_qkv),
                KernelArg::Buffer(&qk),
                KernelArg::U32(HEAD_DIM as u32),
            ],
            rows * KEY_HEADS,
            SIMD_LANES,
        )?;
        Ok(qk)
    }

    pub(super) fn encode_restore(
        &self,
        command_buffer: &::metal::CommandBufferRef,
        cache: &QwenLinearAttentionCache,
        checkpoint_index: usize,
    ) -> Result<()> {
        let checkpoint_conv = cache
            .checkpoint_conv_state
            .as_ref()
            .ok_or_else(|| Error::cache("Qwen convolution checkpoint is unavailable"))?;
        let checkpoint_recurrent = cache
            .checkpoint_recurrent_state
            .as_ref()
            .ok_or_else(|| Error::cache("Qwen recurrent checkpoint is unavailable"))?;
        for (checkpoint, state, label) in [
            (checkpoint_conv, &cache.conv_state, "convolution"),
            (checkpoint_recurrent, &cache.recurrent_state, "recurrent"),
        ] {
            let expected_checkpoint_bytes = state
                .length()
                .checked_mul(cache.checkpoint_capacity as u64)
                .ok_or_else(|| Error::cache("Qwen checkpoint byte length overflow"))?;
            if checkpoint.length() != expected_checkpoint_bytes
                || !state.length().is_multiple_of(4)
                || checkpoint_index >= cache.checkpoint_valid_rows
            {
                return Err(Error::cache(format!(
                    "Qwen {label} checkpoint byte length is invalid"
                )));
            }
            let words = usize::try_from(state.length() / 4)
                .map_err(|_| Error::cache("Qwen checkpoint length does not fit usize"))?;
            let checkpoint_byte_offset = usize::try_from(state.length())
                .ok()
                .and_then(|bytes| bytes.checked_mul(checkpoint_index))
                .ok_or_else(|| Error::cache("Qwen checkpoint offset overflow"))?;
            encode_1d_threadgroups_args(
                command_buffer,
                &self.restore,
                &[
                    KernelArg::BufferOffset(checkpoint, checkpoint_byte_offset),
                    KernelArg::Buffer(state),
                    KernelArg::U32(as_u32(words, "checkpoint words")?),
                ],
                words.div_ceil(256),
                256,
            )?;
        }
        Ok(())
    }
}

fn validate_execution(
    cache: &QwenLinearAttentionCache,
    sequence_length: usize,
    eps: f32,
) -> Result<()> {
    if sequence_length == 0 || !eps.is_finite() || eps <= 0.0 {
        return Err(Error::backend(format!(
            "invalid Qwen delta sequence={sequence_length}, eps={eps}"
        )));
    }
    if cache.conv_channels != CONV_CHANNELS
        || cache.conv_history != CONV_HISTORY
        || cache.value_heads != VALUE_HEADS
        || cache.key_head_dim != HEAD_DIM
        || cache.value_head_dim != HEAD_DIM
    {
        return Err(Error::cache("Qwen linear-attention state shape mismatch"));
    }
    Ok(())
}

fn require_binding(binding: Bf16Binding<'_>, shape: &[usize], label: &str) -> Result<()> {
    if binding.shape != shape {
        return Err(Error::backend(format!(
            "{label} must have shape {shape:?}, got {:?}",
            binding.shape
        )));
    }
    let values = shape.iter().try_fold(1_usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| Error::backend(format!("{label} element count overflow")))
    })?;
    let end = binding
        .byte_offset
        .checked_add(values * std::mem::size_of::<u16>())
        .ok_or_else(|| Error::backend(format!("{label} range overflow")))?;
    require_byte_capacity(binding.buffer, end, label)
}

fn require_conv1d_binding(binding: Bf16Binding<'_>) -> Result<()> {
    let expected_shape = [CONV_CHANNELS, 4, 1];
    if binding.shape != expected_shape {
        return Err(Error::backend(format!(
            "Qwen MLX W4 conv1d must have shape {expected_shape:?}, got {:?}",
            binding.shape
        )));
    }
    let bytes = CONV_CHANNELS
        .checked_mul(4 * std::mem::size_of::<u16>())
        .ok_or_else(|| Error::backend("Qwen conv1d size overflow"))?;
    let end = binding
        .byte_offset
        .checked_add(bytes)
        .ok_or_else(|| Error::backend("Qwen conv1d range overflow"))?;
    require_byte_capacity(binding.buffer, end, "Qwen conv1d")
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
    fn recurrent_state_and_convolution_history_survive_decode_steps() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let Ok(arena) = MetalArena::new(&device) else {
            return;
        };
        let Ok(delta) = MetalQwenDelta::new(&device, arena) else {
            return;
        };
        let cache = delta.create_cache(&device, 1, false).unwrap();
        assert_eq!(cache.storage_bytes().unwrap(), 3_207_168);

        let mut conv_values = vec![0.0_f32; CONV_CHANNELS * 4];
        for channel in 0..CONV_CHANNELS {
            conv_values[channel * 4 + 2] = 1.0;
            conv_values[channel * 4 + 3] = 1.0;
        }
        let conv = bf16_buffer(&device, &conv_values);
        let a_log = bf16_buffer(&device, &vec![0.0; VALUE_HEADS]);
        let dt_bias = bf16_buffer(&device, &vec![0.0; VALUE_HEADS]);
        let norm = bf16_buffer(&device, &vec![1.0; HEAD_DIM]);
        let conv_shape = [CONV_CHANNELS, 4, 1];
        let head_shape = [VALUE_HEADS];
        let norm_shape = [HEAD_DIM];
        let conv = Bf16Binding {
            buffer: &conv,
            byte_offset: 0,
            shape: &conv_shape,
        };
        let a_log = Bf16Binding {
            buffer: &a_log,
            byte_offset: 0,
            shape: &head_shape,
        };
        let dt_bias = Bf16Binding {
            buffer: &dt_bias,
            byte_offset: 0,
            shape: &head_shape,
        };
        let norm = Bf16Binding {
            buffer: &norm,
            byte_offset: 0,
            shape: &norm_shape,
        };
        let gate = vec![1.0_f32; VALUE_WIDTH];
        let control = vec![0.0_f32; VALUE_HEADS];
        let queue = device.new_command_queue();

        let first = run_step(
            &device,
            &queue,
            &delta,
            &cache,
            &vec![1.0; CONV_CHANNELS],
            &gate,
            &control,
            conv,
            a_log,
            dt_bias,
            norm,
        );
        let convolved = round_bf16(silu(1.0));
        let mut state = 0.0_f32;
        let expected_first = symmetric_delta_step(convolved, &mut state);
        assert_constant_close(&first, expected_first);
        assert!((read_first_f32(&cache.recurrent_state) - state).abs() <= 1.0e-5);
        assert_eq!(read_bf16(&cache.conv_state, 3), vec![0.0, 0.0, 1.0]);

        let second = run_step(
            &device,
            &queue,
            &delta,
            &cache,
            &vec![0.0; CONV_CHANNELS],
            &gate,
            &control,
            conv,
            a_log,
            dt_bias,
            norm,
        );
        let expected_second = symmetric_delta_step(convolved, &mut state);
        assert_constant_close(&second, expected_second);
        assert!((read_first_f32(&cache.recurrent_state) - state).abs() <= 1.0e-5);
        assert_eq!(read_bf16(&cache.conv_state, 3), vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn prefill_recurrent_kernel_matches_decode_reference_bitwise() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let arena = MetalArena::new(&device).unwrap();
        let delta = MetalQwenDelta::new(&device, arena).unwrap();
        for sequence_length in [32, 65, 511] {
            let mixed_qkv = (0..sequence_length * CONV_CHANNELS)
                .map(|index| ((index * 13 % 31) as f32 - 15.0) / 32.0)
                .collect::<Vec<_>>();
            let gate = (0..sequence_length * VALUE_WIDTH)
                .map(|index| ((index * 7 % 23) as f32 - 11.0) / 16.0)
                .collect::<Vec<_>>();
            let input_a = (0..sequence_length * VALUE_HEADS)
                .map(|index| ((index * 5 % 17) as f32 - 8.0) / 16.0)
                .collect::<Vec<_>>();
            let input_b = (0..sequence_length * VALUE_HEADS)
                .map(|index| ((index * 3 % 13) as f32 - 6.0) / 16.0)
                .collect::<Vec<_>>();
            let a_log = vec![-1.0_f32; VALUE_HEADS];
            let dt_bias = vec![0.25_f32; VALUE_HEADS];
            let norm = vec![1.0_f32; HEAD_DIM];
            let mixed_qkv = bf16_buffer(&device, &mixed_qkv);
            let gate = bf16_buffer(&device, &gate);
            let input_a = bf16_buffer(&device, &input_a);
            let input_b = bf16_buffer(&device, &input_b);
            let a_log = bf16_buffer(&device, &a_log);
            let dt_bias = bf16_buffer(&device, &dt_bias);
            let norm = bf16_buffer(&device, &norm);
            let reference_cache = delta.create_cache(&device, 1, true).unwrap();
            let candidate_cache = delta.create_cache(&device, 1, true).unwrap();
            let reference_output = delta
                .arena
                .empty_f16(sequence_length * VALUE_WIDTH)
                .unwrap();
            let candidate_output = delta
                .arena
                .empty_f16(sequence_length * VALUE_WIDTH)
                .unwrap();
            let queue = device.new_command_queue();

            // The second invocation starts from a nonzero recurrent state.
            for _ in 0..2 {
                run_recurrent_kernel(
                    queue.new_command_buffer(),
                    &delta.recurrent,
                    &mixed_qkv,
                    &gate,
                    &input_a,
                    &input_b,
                    &a_log,
                    &dt_bias,
                    &norm,
                    &reference_cache,
                    &reference_output,
                    sequence_length,
                    None,
                    None,
                );
                let command = queue.new_command_buffer();
                let prepared = delta
                    .prepare_prefill_qk(command, &mixed_qkv, sequence_length)
                    .unwrap();
                run_recurrent_kernel(
                    command,
                    delta.prefill_recurrent.pipeline(),
                    &mixed_qkv,
                    &gate,
                    &input_a,
                    &input_b,
                    &a_log,
                    &dt_bias,
                    &norm,
                    &candidate_cache,
                    &candidate_output,
                    sequence_length,
                    Some(delta.prefill_recurrent.output_norm_pipeline()),
                    Some(&prepared),
                );

                assert_f32_slices_bitwise_equal(
                    &read_bf16(&candidate_output, sequence_length * VALUE_WIDTH),
                    &read_bf16(&reference_output, sequence_length * VALUE_WIDTH),
                    "Delta prefill output",
                );
                assert_u32_slices_equal(
                    &read_f32_bits(&candidate_cache.recurrent_state),
                    &read_f32_bits(&reference_cache.recurrent_state),
                    "Delta prefill recurrent state",
                );
                assert_u32_slices_equal(
                    &read_f32_bits(candidate_cache.checkpoint_recurrent_state.as_ref().unwrap()),
                    &read_f32_bits(reference_cache.checkpoint_recurrent_state.as_ref().unwrap()),
                    "Delta prefill checkpoints",
                );
            }
        }
    }

    #[test]
    fn speculative_checkpoint_restores_the_exact_accepted_prefix() {
        let Some(device) = Device::system_default() else {
            return;
        };
        let arena = MetalArena::new(&device).unwrap();
        let delta = MetalQwenDelta::new(&device, arena).unwrap();
        let reference_cache = delta.create_cache(&device, 1, false).unwrap();
        let mut speculative_cache = delta.create_cache(&device, 1, true).unwrap();
        let queue = device.new_command_queue();

        let conv_values = (0..CONV_CHANNELS * 4)
            .map(|index| ((index * 3 % 17) as f32 - 8.0) / 16.0)
            .collect::<Vec<_>>();
        let a_log_values = vec![-1.0_f32; VALUE_HEADS];
        let dt_bias_values = vec![0.25_f32; VALUE_HEADS];
        let norm_values = vec![1.0_f32; HEAD_DIM];
        let conv_buffer = bf16_buffer(&device, &conv_values);
        let a_log_buffer = bf16_buffer(&device, &a_log_values);
        let dt_bias_buffer = bf16_buffer(&device, &dt_bias_values);
        let norm_buffer = bf16_buffer(&device, &norm_values);
        let conv_shape = [CONV_CHANNELS, 4, 1];
        let head_shape = [VALUE_HEADS];
        let norm_shape = [HEAD_DIM];
        let conv = Bf16Binding {
            buffer: &conv_buffer,
            byte_offset: 0,
            shape: &conv_shape,
        };
        let a_log = Bf16Binding {
            buffer: &a_log_buffer,
            byte_offset: 0,
            shape: &head_shape,
        };
        let dt_bias = Bf16Binding {
            buffer: &dt_bias_buffer,
            byte_offset: 0,
            shape: &head_shape,
        };
        let norm = Bf16Binding {
            buffer: &norm_buffer,
            byte_offset: 0,
            shape: &norm_shape,
        };

        const BLOCK_ROWS: usize = 8;
        const KEPT_ROWS: usize = 2;
        for round in 0..2 {
            let mixed_qkv = deterministic_values(round, BLOCK_ROWS * CONV_CHANNELS, 31, 64.0);
            let gate = deterministic_values(round + 3, BLOCK_ROWS * VALUE_WIDTH, 29, 32.0);
            let control = deterministic_values(round + 7, BLOCK_ROWS * VALUE_HEADS, 23, 32.0);

            run_block(
                &device,
                &queue,
                &delta,
                &speculative_cache,
                &mixed_qkv,
                &gate,
                &control,
                conv,
                a_log,
                dt_bias,
                norm,
                BLOCK_ROWS,
            );
            speculative_cache.checkpoint_valid_rows = BLOCK_ROWS - 1;
            let restore = queue.new_command_buffer();
            delta
                .encode_restore(restore, &speculative_cache, KEPT_ROWS - 1)
                .unwrap();
            restore.commit();
            restore.wait_until_completed();
            assert_eq!(restore.status(), MTLCommandBufferStatus::Completed);

            run_block(
                &device,
                &queue,
                &delta,
                &reference_cache,
                &mixed_qkv[..KEPT_ROWS * CONV_CHANNELS],
                &gate[..KEPT_ROWS * VALUE_WIDTH],
                &control[..KEPT_ROWS * VALUE_HEADS],
                conv,
                a_log,
                dt_bias,
                norm,
                KEPT_ROWS,
            );

            assert_u16_slices_equal(
                &read_u16_bits(&speculative_cache.conv_state),
                &read_u16_bits(&reference_cache.conv_state),
                "restored Delta convolution state",
            );
            assert_u32_slices_equal(
                &read_f32_bits(&speculative_cache.recurrent_state),
                &read_f32_bits(&reference_cache.recurrent_state),
                "restored Delta recurrent state",
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_step(
        device: &Device,
        queue: &::metal::CommandQueueRef,
        delta: &MetalQwenDelta,
        cache: &QwenLinearAttentionCache,
        mixed_qkv: &[f32],
        gate: &[f32],
        control: &[f32],
        conv: Bf16Binding<'_>,
        a_log: Bf16Binding<'_>,
        dt_bias: Bf16Binding<'_>,
        norm: Bf16Binding<'_>,
    ) -> Vec<f32> {
        let mixed_qkv = bf16_buffer(device, mixed_qkv);
        let gate = bf16_buffer(device, gate);
        let input_a = bf16_buffer(device, control);
        let input_b = bf16_buffer(device, control);
        let command_buffer = queue.new_command_buffer();
        let output = delta
            .encode_raw(
                command_buffer,
                &mixed_qkv,
                &gate,
                &input_a,
                &input_b,
                conv,
                a_log,
                dt_bias,
                norm,
                cache,
                1,
                1.0e-6,
            )
            .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        read_bf16(&output, VALUE_WIDTH)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_block(
        device: &Device,
        queue: &::metal::CommandQueueRef,
        delta: &MetalQwenDelta,
        cache: &QwenLinearAttentionCache,
        mixed_qkv: &[f32],
        gate: &[f32],
        control: &[f32],
        conv: Bf16Binding<'_>,
        a_log: Bf16Binding<'_>,
        dt_bias: Bf16Binding<'_>,
        norm: Bf16Binding<'_>,
        sequence_length: usize,
    ) {
        let mixed_qkv = bf16_buffer(device, mixed_qkv);
        let gate = bf16_buffer(device, gate);
        let input_a = bf16_buffer(device, control);
        let input_b = bf16_buffer(device, control);
        let command_buffer = queue.new_command_buffer();
        let _output = delta
            .encode_raw(
                command_buffer,
                &mixed_qkv,
                &gate,
                &input_a,
                &input_b,
                conv,
                a_log,
                dt_bias,
                norm,
                cache,
                sequence_length,
                1.0e-6,
            )
            .unwrap();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
    }

    fn deterministic_values(seed: usize, len: usize, modulus: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                (((index * 13 + seed * 7) % modulus) as f32 - modulus as f32 / 2.0) / scale
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn run_recurrent_kernel(
        command_buffer: &::metal::CommandBufferRef,
        pipeline: &ComputePipelineState,
        mixed_qkv: &Buffer,
        gate: &Buffer,
        input_a: &Buffer,
        input_b: &Buffer,
        a_log: &Buffer,
        dt_bias: &Buffer,
        norm: &Buffer,
        cache: &QwenLinearAttentionCache,
        output: &Buffer,
        sequence_length: usize,
        output_norm: Option<&ComputePipelineState>,
        prepared: Option<&Buffer>,
    ) {
        encode_1d_threadgroups_args(
            command_buffer,
            pipeline,
            &[
                KernelArg::Buffer(mixed_qkv),
                KernelArg::Buffer(gate),
                KernelArg::Buffer(input_a),
                KernelArg::Buffer(input_b),
                KernelArg::Buffer(a_log),
                KernelArg::Buffer(dt_bias),
                KernelArg::Buffer(prepared.unwrap_or(norm)),
                KernelArg::Buffer(&cache.recurrent_state),
                KernelArg::Buffer(
                    cache
                        .checkpoint_recurrent_state
                        .as_ref()
                        .unwrap_or(&cache.recurrent_state),
                ),
                KernelArg::Buffer(output),
                KernelArg::U32(1),
                KernelArg::U32(sequence_length as u32),
                KernelArg::U32(KEY_HEADS as u32),
                KernelArg::U32(VALUE_HEADS as u32),
                KernelArg::U32(HEAD_DIM as u32),
                KernelArg::F32(1.0e-6),
                KernelArg::U32(cache.checkpoint_capacity as u32),
            ],
            VALUE_HEADS,
            RECURRENT_THREADS,
        )
        .unwrap();
        if let Some(pipeline) = output_norm {
            encode_1d_threadgroups_args(
                command_buffer,
                pipeline,
                &[
                    KernelArg::Buffer(output),
                    KernelArg::Buffer(gate),
                    KernelArg::Buffer(norm),
                    KernelArg::F32(1.0e-6),
                ],
                sequence_length * VALUE_HEADS,
                RECURRENT_THREADS,
            )
            .unwrap();
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
    }

    fn symmetric_delta_step(value: f32, state: &mut f32) -> f32 {
        let inverse_norm = (HEAD_DIM as f32 * (value * value + 1.0e-6)).sqrt().recip();
        let sqrt_head_dim = (HEAD_DIM as f32).sqrt();
        let query_rms = round_bf16(value * inverse_norm * sqrt_head_dim);
        let key_rms = round_bf16(value * inverse_norm * sqrt_head_dim);
        let query = round_bf16(query_rms / HEAD_DIM as f32);
        let key = round_bf16(key_rms / sqrt_head_dim);
        *state *= 0.5;
        let memory = HEAD_DIM as f32 * *state * key;
        let delta = (value - memory) * 0.5;
        *state += key * delta;
        let attended = round_bf16(HEAD_DIM as f32 * *state * query);
        let normalized = round_bf16(attended / (attended * attended + 1.0e-6).sqrt());
        let weighted = round_bf16(normalized);
        round_bf16(weighted * silu(1.0))
    }

    fn bf16_buffer(device: &Device, values: &[f32]) -> Buffer {
        let bytes = values
            .iter()
            .flat_map(|value| round_bf16_bits(*value).to_le_bytes())
            .collect::<Vec<_>>();
        device.new_buffer_with_data(
            bytes.as_ptr().cast(),
            bytes.len() as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    fn read_bf16(buffer: &Buffer, len: usize) -> Vec<f32> {
        let values = unsafe { slice::from_raw_parts(buffer.contents().cast::<u16>(), len) };
        values
            .iter()
            .map(|bits| f32::from_bits(u32::from(*bits) << 16))
            .collect()
    }

    fn read_first_f32(buffer: &Buffer) -> f32 {
        unsafe { *buffer.contents().cast::<f32>() }
    }

    fn read_f32_bits(buffer: &Buffer) -> Vec<u32> {
        let len = buffer.length() as usize / std::mem::size_of::<u32>();
        unsafe { slice::from_raw_parts(buffer.contents().cast::<u32>(), len) }.to_vec()
    }

    fn read_u16_bits(buffer: &Buffer) -> Vec<u16> {
        let len = buffer.length() as usize / std::mem::size_of::<u16>();
        unsafe { slice::from_raw_parts(buffer.contents().cast::<u16>(), len) }.to_vec()
    }

    fn silu(value: f32) -> f32 {
        value / (1.0 + (-value).exp())
    }

    fn round_bf16(value: f32) -> f32 {
        f32::from_bits(u32::from(round_bf16_bits(value)) << 16)
    }

    fn round_bf16_bits(value: f32) -> u16 {
        let bits = value.to_bits();
        ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    }

    fn assert_constant_close(actual: &[f32], expected: f32) {
        assert_eq!(actual.len(), VALUE_WIDTH);
        for (index, actual) in actual.iter().enumerate() {
            assert!(
                (actual - expected).abs() <= 0.015625,
                "BF16 mismatch at {index}: actual={actual}, expected={expected}"
            );
        }
    }

    fn assert_f32_slices_bitwise_equal(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len());
        let first = actual
            .iter()
            .zip(expected)
            .enumerate()
            .find(|(_, (actual, expected))| actual.to_bits() != expected.to_bits());
        if let Some((index, (first_actual, first_expected))) = first {
            let max_absolute_error = actual
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f32, f32::max);
            panic!(
                "{label} first differs at {index}: actual={first_actual}, expected={first_expected}, maximum absolute error={max_absolute_error}"
            );
        }
    }

    fn assert_u32_slices_equal(actual: &[u32], expected: &[u32], label: &str) {
        assert_eq!(actual.len(), expected.len());
        if let Some((index, (actual, expected))) = actual
            .iter()
            .zip(expected)
            .enumerate()
            .find(|(_, (actual, expected))| actual != expected)
        {
            panic!("{label} first differs at {index}: actual={actual:#010x}, expected={expected:#010x}");
        }
    }

    fn assert_u16_slices_equal(actual: &[u16], expected: &[u16], label: &str) {
        assert_eq!(actual.len(), expected.len());
        let mismatch = actual
            .iter()
            .zip(expected)
            .position(|(actual, expected)| actual != expected);
        assert_eq!(mismatch, None, "{label} differs at {mismatch:?}");
    }
}
