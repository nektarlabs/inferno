use ::metal::{CommandQueue, MTLCommandBufferStatus, MTLResourceOptions};
use objc::{msg_send, sel, sel_impl};

use super::*;

fn upload<T>(device: &Device, values: &[T]) -> Buffer {
    device.new_buffer_with_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

#[allow(unexpected_cfgs)]
fn run(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    buffers: &[Buffer],
    output: &Buffer,
    shape: [usize; 3],
    token_tile: usize,
) -> f64 {
    let [rows, width, outputs] = shape;
    let command = queue.new_command_buffer();
    encode_1d_threadgroups_args(
        command,
        pipeline,
        &[
            KernelArg::Buffer(&buffers[0]),
            KernelArg::Buffer(&buffers[1]),
            KernelArg::Buffer(&buffers[2]),
            KernelArg::Buffer(&buffers[3]),
            KernelArg::Buffer(output),
            KernelArg::U32(rows as u32),
            KernelArg::U32(width as u32),
            KernelArg::U32(outputs as u32),
        ],
        rows.div_ceil(token_tile) * (outputs / LINEAR_OUTPUTS) * K_PARTS,
        token_tile * 4,
    )
    .unwrap();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    let start: f64 = unsafe { msg_send![command, GPUStartTime] };
    let end: f64 = unsafe { msg_send![command, GPUEndTime] };
    assert!(end > start);
    (end - start) * 1000.0
}

#[allow(unexpected_cfgs)]
fn run_fused(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    buffers: &[Buffer],
    gate: &Buffer,
    up: &Buffer,
    shape: [usize; 3],
) -> f64 {
    let [rows, width, outputs] = shape;
    let command = queue.new_command_buffer();
    encode_1d_threadgroups_args(
        command,
        pipeline,
        &[
            KernelArg::Buffer(&buffers[0]),
            KernelArg::Buffer(&buffers[1]),
            KernelArg::Buffer(&buffers[2]),
            KernelArg::Buffer(&buffers[4]),
            KernelArg::Buffer(&buffers[1]),
            KernelArg::Buffer(&buffers[2]),
            KernelArg::Buffer(&buffers[3]),
            KernelArg::Buffer(gate),
            KernelArg::Buffer(up),
            KernelArg::U32(rows as u32),
            KernelArg::U32(width as u32),
            KernelArg::U32(outputs as u32),
        ],
        rows.div_ceil(LINEAR_TOKEN_ROWS) * (outputs / LINEAR_OUTPUTS) * K_PARTS,
        LINEAR_THREADS,
    )
    .unwrap();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    let start: f64 = unsafe { msg_send![command, GPUStartTime] };
    let end: f64 = unsafe { msg_send![command, GPUEndTime] };
    assert!(end > start);
    (end - start) * 1000.0
}

#[allow(unexpected_cfgs)]
fn run_resident(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    buffers: &[Buffer],
    output: &Buffer,
    shape: [usize; 3],
) -> f64 {
    let [rows, width, outputs] = shape;
    let command = queue.new_command_buffer();
    encode_1d_threadgroups_args(
        command,
        pipeline,
        &[
            KernelArg::Buffer(&buffers[0]),
            KernelArg::Buffer(&buffers[1]),
            KernelArg::Buffer(&buffers[2]),
            KernelArg::Buffer(&buffers[4]),
            KernelArg::Buffer(&buffers[1]),
            KernelArg::Buffer(&buffers[2]),
            KernelArg::Buffer(&buffers[3]),
            KernelArg::Buffer(output),
            KernelArg::U32(rows as u32),
            KernelArg::U32(width as u32),
            KernelArg::U32(outputs as u32),
        ],
        rows.div_ceil(LINEAR_TOKEN_ROWS) * (outputs / LINEAR_OUTPUTS),
        LINEAR_THREADS,
    )
    .unwrap();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    let start: f64 = unsafe { msg_send![command, GPUStartTime] };
    let end: f64 = unsafe { msg_send![command, GPUEndTime] };
    assert!(end > start);
    (end - start) * 1000.0
}

#[allow(unexpected_cfgs)]
fn run_reduce(
    operations: &MetalQwenPackedPrefill,
    queue: &CommandQueue,
    gate: &Buffer,
    up: &Buffer,
    output: &Buffer,
    len: usize,
) -> f64 {
    let command = queue.new_command_buffer();
    operations
        .encode_reduce_swiglu(command, gate, up, output, len)
        .unwrap();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    let start: f64 = unsafe { msg_send![command, GPUStartTime] };
    let end: f64 = unsafe { msg_send![command, GPUEndTime] };
    assert!(end > start);
    (end - start) * 1000.0
}

#[test]
fn prefill_repack_preserves_weights_and_parameters_at_model_widths() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let operations =
        MetalQwenPackedPrefill::new(&device, MetalArena::new(&device).unwrap()).unwrap();
    let queue = device.new_command_queue();
    let outputs = 96;
    for width in [256, 5120, 17408] {
        let words = width / VALUES_PER_WORD;
        let groups = width / GROUP_SIZE;
        let packed: Vec<u32> = (0..outputs * words)
            .map(|i| (i as u32).wrapping_mul(0x9e3779b9))
            .collect();
        let scales: Vec<u16> = (0..outputs * groups).map(|i| i as u16).collect();
        let biases: Vec<u16> = scales.iter().map(|v| v ^ 0x5a5a).collect();
        let source = upload(&device, &packed);
        let scale_source = upload(&device, &scales);
        let bias_source = upload(&device, &biases);
        let command = queue.new_command_buffer();
        let result = operations
            .repack_raw(
                command,
                &source,
                0,
                &scale_source,
                0,
                &bias_source,
                0,
                width,
                outputs,
                0,
                LINEAR_OUTPUTS,
            )
            .unwrap();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        // The command is complete; the mapped shared buffers are safe to inspect.
        let actual = unsafe {
            std::slice::from_raw_parts(result.packed.contents().cast::<u32>(), packed.len())
        };
        let actual_scales = unsafe {
            std::slice::from_raw_parts(result.scales.contents().cast::<u16>(), scales.len())
        };
        let actual_biases = unsafe {
            std::slice::from_raw_parts(result.biases.contents().cast::<u16>(), biases.len())
        };
        for row in 0..outputs {
            let tile = row / LINEAR_OUTPUTS;
            let offset = row % LINEAR_OUTPUTS;
            for word in 0..words {
                assert_eq!(
                    actual[(tile * words + word) * LINEAR_OUTPUTS + offset],
                    packed[row * words + word]
                );
            }
            for group in 0..groups {
                let destination = (tile * groups + group) * LINEAR_OUTPUTS + offset;
                assert_eq!(actual_scales[destination], scales[row * groups + group]);
                assert_eq!(actual_biases[destination], biases[row * groups + group]);
            }
        }
    }
}

#[test]
fn prefill_scratch_reuse_is_ordered_across_queued_commands() {
    let Some(device) = Device::system_default() else {
        return;
    };
    let arena = MetalArena::new(&device).unwrap();
    let operations = MetalQwenPackedPrefill::new(&device, arena.clone()).unwrap();
    let queue = device.new_command_queue();
    let width = 256;
    let outputs = 64;
    let mut pending = Vec::new();

    // Grow the scratch once, then reuse it across commands before waiting for the GPU.
    for (job, rows) in [65, 96, 65, 96].into_iter().enumerate() {
        let packed = (0..outputs * width / 8)
            .map(|index| (((index / (width / 8) + job * 3) % 16) as u32) * 0x1111_1111)
            .collect::<Vec<_>>();
        let parameters = outputs * width / GROUP_SIZE;
        let input = (0..rows * width)
            .map(|index| ((((index / width + job) % 5 + 1) as f32 / 16.0).to_bits() >> 16) as u16)
            .collect::<Vec<_>>();
        let packed = upload(&device, &packed);
        let scales = upload(&device, &vec![0x3c80_u16; parameters]); // 1/64
        let biases = upload(&device, &vec![0xbe00_u16; parameters]); // -1/8
        let input = upload(&device, &input);
        let output = arena.empty_f16(rows * outputs).unwrap();
        let command = queue.new_command_buffer().to_owned();
        operations
            .encode_linear_raw(
                &command, &packed, &scales, &biases, &input, &output, rows, width, outputs,
            )
            .unwrap();
        pending.push((command, output, rows, job));
    }
    for (command, _, _, _) in &pending {
        command.commit();
    }
    pending.last().unwrap().0.wait_until_completed();
    for (command, output, rows, job) in &pending {
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        // All queued writes have completed; the reference values are exactly representable in BF16.
        let actual =
            unsafe { std::slice::from_raw_parts(output.contents().cast::<u16>(), rows * outputs) };
        for (index, &bits) in actual.iter().enumerate() {
            let input = ((index / outputs + job) % 5 + 1) as f32 / 16.0;
            let weight = ((index % outputs + job * 3) % 16) as f32 - 8.0;
            let expected = ((input * weight * 4.0).to_bits() >> 16) as u16;
            assert_eq!(bits, expected, "queued prefill job={job} index={index}");
        }
    }
}

#[test]
#[ignore = "requires native Metal; compares fused prefill gate/up at production dimensions"]
fn prefill_fused_gate_up_matches_bitwise_and_benchmark() {
    let device = Device::system_default().expect("native Metal is required");
    let queue = device.new_command_queue();
    let library = MetalLibrary::compile_source(&device, SOURCE).unwrap();
    let linear = compute_pipeline(&device, &library, LINEAR).unwrap();
    let fused = compute_pipeline(&device, &library, GATE_UP).unwrap();
    let operations =
        MetalQwenPackedPrefill::new(&device, MetalArena::new(&device).unwrap()).unwrap();
    for shape in [
        [65, 256, 64],
        [65, 5120, 17408],
        [257, 5120, 17408],
        [511, 5120, 17408],
    ] {
        let [rows, width, outputs] = shape;
        let mut state = 0x9e37_79b9_u32;
        let mut random = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let packed = (0..width * outputs / 8)
            .map(|_| random())
            .collect::<Vec<_>>();
        let up = packed
            .iter()
            .map(|word| word ^ 0xa35d790c)
            .collect::<Vec<_>>();
        let scales = (0..width * outputs / GROUP_SIZE)
            .map(|_| (((1 + random() % 31) as f32 / 1024.0).to_bits() >> 16) as u16)
            .collect::<Vec<_>>();
        let biases = scales
            .iter()
            .map(|&bits| ((-7.5 * f32::from_bits(u32::from(bits) << 16)).to_bits() >> 16) as u16)
            .collect::<Vec<_>>();
        let input = (0..rows * width)
            .map(|_| ((((random() % 2049) as f32 / 1024.0) - 1.0).to_bits() >> 16) as u16)
            .collect::<Vec<_>>();
        let buffers = [
            upload(&device, &packed),
            upload(&device, &scales),
            upload(&device, &biases),
            upload(&device, &input),
            upload(&device, &up),
        ];
        let up_buffers = [
            buffers[4].clone(),
            buffers[1].clone(),
            buffers[2].clone(),
            buffers[3].clone(),
        ];
        let len = rows * outputs * K_PARTS;
        let reference = device.new_buffer((len * 4) as u64, MTLResourceOptions::StorageModeShared);
        let up_reference =
            device.new_buffer((len * 4) as u64, MTLResourceOptions::StorageModeShared);
        let output = device.new_buffer((len * 4) as u64, MTLResourceOptions::StorageModeShared);
        let up_output = device.new_buffer((len * 4) as u64, MTLResourceOptions::StorageModeShared);
        run(&queue, &linear, &buffers, &reference, shape, 32);
        run(&queue, &linear, &up_buffers, &up_reference, shape, 32);
        let final_output = empty_f16_buffer(&device, rows * outputs).unwrap();
        let final_reference = empty_f16_buffer(&device, rows * outputs).unwrap();
        run_reduce(
            &operations,
            &queue,
            &reference,
            &up_reference,
            &final_reference,
            rows * outputs,
        );
        let mut samples = [Vec::new(), Vec::new(), Vec::new()];
        for iteration in 0..10 {
            for index in if iteration % 2 == 0 {
                [0, 1, 2]
            } else {
                [2, 1, 0]
            } {
                let mut elapsed = if index == 0 {
                    run(&queue, &linear, &buffers, &output, shape, 32)
                        + run(&queue, &linear, &up_buffers, &up_output, shape, 32)
                } else if index == 1 {
                    run_fused(&queue, &fused, &buffers, &output, &up_output, shape)
                } else {
                    run_resident(
                        &queue,
                        &operations.resident_gate_up,
                        &buffers,
                        &final_output,
                        shape,
                    )
                };
                if index != 2 {
                    elapsed += run_reduce(
                        &operations,
                        &queue,
                        &output,
                        &up_output,
                        &final_output,
                        rows * outputs,
                    );
                }
                // Both commands have completed before accessing their shared output memory.
                let actual_final = unsafe {
                    std::slice::from_raw_parts(
                        final_output.contents().cast::<u16>(),
                        rows * outputs,
                    )
                };
                let expected_final = unsafe {
                    std::slice::from_raw_parts(
                        final_reference.contents().cast::<u16>(),
                        rows * outputs,
                    )
                };
                if let Some(position) = actual_final
                    .iter()
                    .zip(expected_final)
                    .position(|(a, b)| a != b)
                {
                    panic!(
                        "final output mismatch shape={shape:?} variant={index} index={position}"
                    );
                }
                if iteration >= 2 {
                    samples[index].push(elapsed);
                }
                if index == 2 {
                    continue;
                }
                for (actual, expected) in [(&output, &reference), (&up_output, &up_reference)] {
                    // Shared buffers are fully written and their commands have completed.
                    let expected = unsafe {
                        std::slice::from_raw_parts(expected.contents().cast::<u32>(), len)
                    };
                    let actual =
                        unsafe { std::slice::from_raw_parts(actual.contents().cast::<u32>(), len) };
                    if let Some(position) = actual.iter().zip(expected).position(|(a, b)| a != b) {
                        panic!(
                            "prefill mismatch shape={shape:?} fused={} index={position}: {} != {}",
                            index == 1,
                            actual[position],
                            expected[position]
                        );
                    }
                }
            }
        }
        for (kind, times) in ["separate", "fused", "resident"]
            .into_iter()
            .zip(&mut samples)
        {
            times.sort_by(f64::total_cmp);
            eprintln!(
                "shape={shape:?} kind={kind} median_ms={:.6}",
                times[times.len() / 2]
            );
        }
    }
}
