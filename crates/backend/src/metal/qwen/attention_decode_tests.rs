use std::{slice, time::Instant};

use ::metal::{CommandQueueRef, MTLCommandBufferStatus};

use super::*;

struct Fixture {
    query: Buffer,
    gate: Buffer,
    key: Buffer,
    value: Buffer,
    batch: usize,
    rows: usize,
    context: usize,
    capacity: usize,
}

impl Fixture {
    fn new(device: &Device, batch: usize, rows: usize, context: usize) -> Self {
        let capacity = context + rows + 17;
        let values = batch * rows * QUERY_HEADS * HEAD_DIM;
        let make = |count, salt: u32| {
            let mut state = salt;
            let data = (0..count)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    let value = (state as f64 / u32::MAX as f64 * 4.0 - 2.0) as f32;
                    let bits = value.to_bits();
                    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
                })
                .collect::<Vec<_>>();
            device.new_buffer_with_data(
                data.as_ptr().cast(),
                (data.len() * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        Self {
            query: make(values, 123),
            gate: make(values, 456),
            key: make(batch * capacity * KEY_VALUE_WIDTH, 789),
            value: make(batch * capacity * KEY_VALUE_WIDTH, 981),
            batch,
            rows,
            context,
            capacity,
        }
    }

    fn run(
        &self,
        device: &Device,
        queue: &CommandQueueRef,
        pipeline: &ComputePipelineState,
        shared_heads: Option<usize>,
        repeats: usize,
    ) -> (Vec<u16>, f64) {
        let count = self.batch * self.rows * QUERY_HEADS * HEAD_DIM;
        let output = device.new_buffer((count * 2) as u64, MTLResourceOptions::StorageModeShared);
        let command = queue.new_command_buffer();
        let mut args = vec![
            KernelArg::Buffer(&self.query),
            KernelArg::Buffer(&self.gate),
            KernelArg::Buffer(&self.key),
            KernelArg::Buffer(&self.value),
            KernelArg::Buffer(&output),
            KernelArg::U32((self.batch * self.rows) as u32),
            KernelArg::U32(self.rows as u32),
            KernelArg::U32(QUERY_HEADS as u32),
            KernelArg::U32(KEY_VALUE_HEADS as u32),
            KernelArg::U32(HEAD_DIM as u32),
            KernelArg::U32(self.capacity as u32),
            KernelArg::U32(self.context as u32),
        ];
        let heads_per_group = shared_heads.unwrap_or(1);
        if shared_heads.is_some() {
            args.push(KernelArg::U32(heads_per_group as u32));
        }
        for _ in 0..repeats {
            encode_1d_threadgroups_args(
                command,
                pipeline,
                &args,
                self.batch * self.rows * QUERY_HEADS / heads_per_group,
                if shared_heads.is_some() {
                    8 * SIMD_LANES * heads_per_group
                } else {
                    SIMD_LANES
                },
            )
            .unwrap();
        }
        // Wall time of an already encoded batch, including one submit/wait.
        let started = Instant::now();
        command.commit();
        command.wait_until_completed();
        let ms = started.elapsed().as_secs_f64() * 1000.0 / repeats as f64;
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        let values = unsafe { slice::from_raw_parts(output.contents().cast::<u16>(), count) };
        (values.to_vec(), ms)
    }
}

#[test]
#[ignore = "requires exclusive native Metal access; benchmarks decode-only candidates"]
fn benchmark_decode_attention_tiles() {
    let device = Device::system_default().expect("native Metal required");
    let queue = device.new_command_queue();
    let reference = compute_pipeline(
        &device,
        &MetalLibrary::compile_source(&device, KERNEL_SOURCE).unwrap(),
        ATTENTION_KERNEL,
    )
    .unwrap();
    let candidate = compute_pipeline(
        &device,
        &MetalLibrary::compile_source(&device, include_str!("kernels/decode_attention.metal"))
            .unwrap(),
        DECODE_KERNEL,
    )
    .unwrap();

    for (batch, rows, context) in [
        (1, 1, 0),
        (1, 5, 16),
        (1, 1, 128),
        (2, 5, 33),
        (1, 1, 512),
        (1, 5, 512),
        (1, 8, 2048),
        (1, 1, 8192),
        (1, 5, 8192),
        (1, 1, 32768),
        (1, 5, 32768),
    ] {
        let fixture = Fixture::new(&device, batch, rows, context);
        let expected = fixture.run(&device, &queue, &reference, None, 1).0;
        let mut times = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut mismatches = [0; 4];
        for repetition in 0..4 {
            let mut methods = [None, Some(1), Some(2), Some(3)];
            if repetition % 2 != 0 {
                methods.reverse();
            }
            for method in methods {
                let pipeline = if method.is_none() {
                    &reference
                } else {
                    &candidate
                };
                let (actual, ms) = fixture.run(&device, &queue, pipeline, method, 3);
                let index = method.unwrap_or(0);
                mismatches[index] = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
                assert_eq!(
                    mismatches[index], 0,
                    "batch={batch} rows={rows} context={context} shared_heads={method:?}"
                );
                if repetition > 0 {
                    times[index].push(ms);
                }
            }
        }
        let medians = times.map(|mut samples| {
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        });
        eprintln!("qwen_attention_tiles batch={batch} rows={rows} context={context} reference_ms={:.4} staged1_ms={:.4} shared2_ms={:.4} shared3_ms={:.4} mismatches={mismatches:?}", medians[0], medians[1], medians[2], medians[3]);
    }
}

#[test]
fn tiled_decode_matches_reference_bits_at_causal_and_tile_boundaries() {
    let device = Device::system_default().expect("native Metal required");
    let queue = device.new_command_queue();
    let reference = compute_pipeline(
        &device,
        &MetalLibrary::compile_source(&device, KERNEL_SOURCE).unwrap(),
        ATTENTION_KERNEL,
    )
    .unwrap();
    let candidate = compute_pipeline(
        &device,
        &MetalLibrary::compile_source(&device, include_str!("kernels/decode_attention.metal"))
            .unwrap(),
        DECODE_KERNEL,
    )
    .unwrap();
    for (batch, rows, context) in [
        (1, 1, 0),
        (2, 3, 1),
        (2, 5, 7),
        (2, 8, 8),
        (1, 3, 127),
        (2, 5, 128),
        (1, 8, 511),
        (1, 1, 512),
        (2, 3, 513),
        (1, 5, 2047),
        (1, 8, 8191),
    ] {
        let fixture = Fixture::new(&device, batch, rows, context);
        // Unallocated future tokens must never enter the causal attention.
        for buffer in [&fixture.key, &fixture.value] {
            let data = unsafe {
                slice::from_raw_parts_mut(
                    buffer.contents().cast::<u16>(),
                    batch * fixture.capacity * KEY_VALUE_WIDTH,
                )
            };
            for batch_index in 0..batch {
                let start = (batch_index * fixture.capacity + context + rows) * KEY_VALUE_WIDTH;
                let end = (batch_index + 1) * fixture.capacity * KEY_VALUE_WIDTH;
                data[start..end].fill(0x7fc0);
            }
        }
        let expected = fixture.run(&device, &queue, &reference, None, 1).0;
        let actual = fixture.run(&device, &queue, &candidate, Some(2), 1).0;
        assert!(actual
            .iter()
            .all(|bits| f32::from_bits(u32::from(*bits) << 16).is_finite()));
        let mismatches = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
        assert_eq!(mismatches, 0, "batch={batch} rows={rows} context={context}");
    }

    for scale in [0.0_f32, 0.125, 4.0] {
        let fixture = Fixture::new(&device, 2, 5, 257);
        for buffer in [&fixture.query, &fixture.key] {
            let values = unsafe {
                slice::from_raw_parts_mut(
                    buffer.contents().cast::<u16>(),
                    buffer.length() as usize / 2,
                )
            };
            for bits in values {
                let value = f32::from_bits(u32::from(*bits) << 16) * scale;
                *bits = (value.to_bits() >> 16) as u16;
            }
        }
        let gates = unsafe {
            slice::from_raw_parts_mut(
                fixture.gate.contents().cast::<u16>(),
                fixture.gate.length() as usize / 2,
            )
        };
        for (index, bits) in gates.iter_mut().enumerate() {
            *bits = ([0.0_f32, -80.0, 80.0][index % 3].to_bits() >> 16) as u16;
        }
        let expected = fixture.run(&device, &queue, &reference, None, 1).0;
        let actual = fixture.run(&device, &queue, &candidate, Some(2), 1).0;
        let mismatches = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
        assert_eq!(mismatches, 0, "uniform/sharp scores scale={scale}");
    }
}

#[test]
#[ignore = "requires exclusive native Metal access; measures allocation plus GPU copy"]
fn benchmark_decode_kv_growth() {
    let device = Device::system_default().expect("native Metal required");
    let queue = device.new_command_queue();
    let attention = MetalQwenAttention::new(&device, MetalArena::new(&device).unwrap()).unwrap();
    for allocated in [512, 8192, 32768] {
        let mut times = Vec::new();
        for repetition in 0..4 {
            let mut cache = attention.create_cache(&device, 1, allocated * 2).unwrap();
            let prepare = queue.new_command_buffer();
            attention
                .reserve_cache(&device, prepare, &mut cache, allocated)
                .unwrap();
            let fill = prepare.new_blit_command_encoder();
            for buffer in [&cache.key, &cache.value] {
                fill.fill_buffer(buffer, ::metal::NSRange::new(0, buffer.length()), 0);
            }
            fill.end_encoding();
            prepare.commit();
            prepare.wait_until_completed();
            assert_eq!(prepare.status(), MTLCommandBufferStatus::Completed);
            cache.length = allocated - 1;
            let command = queue.new_command_buffer();
            let started = Instant::now();
            attention
                .reserve_cache(&device, command, &mut cache, 2)
                .unwrap();
            command.commit();
            command.wait_until_completed();
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            if repetition > 0 {
                times.push(ms);
            }
        }
        times.sort_by(f64::total_cmp);
        eprintln!("qwen_kv_growth layer_count=1 old_capacity={allocated} new_capacity={} copied_mib={:.3} allocate_copy_wait_ms={:.3}",
            allocated * 2, (allocated - 1) as f64 * KEY_VALUE_WIDTH as f64 * 4.0 / 1048576.0, times[1]);
    }
}
