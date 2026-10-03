use ::metal::{CommandQueue, MTLCommandBufferStatus, MTLResourceOptions};
use objc::{msg_send, sel, sel_impl};

use super::super::super::buffers::read_bf16_buffer_as_f32;
use super::*;

struct GateFixture {
    buffers: Vec<Buffer>,
    reference: Buffer,
    output: Buffer,
    rows: usize,
    width: usize,
    outputs: usize,
}

impl GateFixture {
    fn new(device: &Device, rows: usize, width: usize, outputs: usize) -> Self {
        let mut state = 0x9e37_79b9_u32;
        let mut random = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut buffers = Vec::new();
        for _ in 0..2 {
            let packed = (0..outputs * width / VALUES_PER_WORD)
                .map(|_| random())
                .collect::<Vec<_>>();
            buffers.push(upload(device, &packed));
            let parameters = outputs * width / GROUP_SIZE;
            let scales = (0..parameters)
                .map(|_| bf16((1 + random() % 113) as f32 / 8192.0))
                .collect::<Vec<_>>();
            let biases = scales
                .iter()
                .map(|&scale| bf16(-7.5 * f32::from_bits(u32::from(scale) << 16)))
                .collect::<Vec<_>>();
            buffers.push(upload(device, &scales));
            buffers.push(upload(device, &biases));
        }
        let input = (0..rows * width)
            .map(|_| bf16((random() % 2049) as f32 / 1024.0 - 1.0))
            .collect::<Vec<_>>();
        buffers.push(upload(device, &input));
        let blank = vec![0_u16; rows * outputs];
        Self {
            buffers,
            reference: upload(device, &blank),
            output: upload(device, &blank),
            rows,
            width,
            outputs,
        }
    }

    fn run(&self, queue: &CommandQueue, pipeline: &ComputePipelineState, generic: bool) -> f64 {
        let command = queue.new_command_buffer();
        let output = if generic {
            &self.reference
        } else {
            &self.output
        };
        let mut args = self
            .buffers
            .iter()
            .map(KernelArg::Buffer)
            .collect::<Vec<_>>();
        args.push(KernelArg::Buffer(output));
        if generic {
            args.push(KernelArg::U32(self.rows as u32));
        }
        args.push(KernelArg::U32(self.width as u32));
        args.push(KernelArg::U32(self.outputs as u32));
        let groups = if generic {
            self.outputs * self.rows.div_ceil(ROW_TILE)
        } else {
            self.outputs.div_ceil(2)
        };
        encode_1d_threadgroups_args(command, pipeline, &args, groups, LINEAR_THREADS).unwrap();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        gpu_milliseconds(command)
    }

    fn assert_exact(&self) {
        let count = self.rows * self.outputs;
        let reference = read_bf16_buffer_as_f32(&self.reference, count).unwrap();
        let output = read_bf16_buffer_as_f32(&self.output, count).unwrap();
        for (index, (actual, expected)) in output.iter().zip(&reference).enumerate() {
            assert!(actual.is_finite() && expected.is_finite());
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "BF16 mismatch at {index}"
            );
        }
    }

    fn run_linear(
        &self,
        queue: &CommandQueue,
        pipeline: &ComputePipelineState,
        generic: bool,
    ) -> f64 {
        let command = queue.new_command_buffer();
        let output = if generic {
            &self.reference
        } else {
            &self.output
        };
        let mut args = vec![
            KernelArg::Buffer(&self.buffers[0]),
            KernelArg::Buffer(&self.buffers[1]),
            KernelArg::Buffer(&self.buffers[2]),
            KernelArg::Buffer(&self.buffers[6]),
            KernelArg::Buffer(output),
        ];
        if generic {
            args.push(KernelArg::U32(self.rows as u32));
        }
        args.push(KernelArg::U32(self.width as u32));
        args.push(KernelArg::U32(self.outputs as u32));
        if generic {
            args.extend([KernelArg::U32(0), KernelArg::U32(1)]);
        }
        let groups = if generic {
            self.outputs * self.rows.div_ceil(ROW_TILE)
        } else {
            self.outputs.div_ceil(4)
        };
        encode_1d_threadgroups_args(command, pipeline, &args, groups, LINEAR_THREADS).unwrap();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        gpu_milliseconds(command)
    }

    fn check_linear_add(
        &self,
        device: &Device,
        queue: &CommandQueue,
        pipeline: &ComputePipelineState,
    ) {
        let count = self.rows * self.outputs;
        let residual = (0..count)
            .map(|index| bf16((index % 257) as f32 / 128.0 - 1.0))
            .collect::<Vec<_>>();
        let residual_buffer = upload(device, &residual);
        let command = queue.new_command_buffer();
        encode_1d_threadgroups_args(
            command,
            pipeline,
            &[
                KernelArg::Buffer(&self.buffers[0]),
                KernelArg::Buffer(&self.buffers[1]),
                KernelArg::Buffer(&self.buffers[2]),
                KernelArg::Buffer(&self.buffers[6]),
                KernelArg::Buffer(&residual_buffer),
                KernelArg::Buffer(&self.output),
                KernelArg::U32(self.width as u32),
                KernelArg::U32(self.outputs as u32),
            ],
            self.outputs.div_ceil(4),
            LINEAR_THREADS,
        )
        .unwrap();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

        let reference = read_bf16_buffer_as_f32(&self.reference, count).unwrap();
        let output = read_bf16_buffer_as_f32(&self.output, count).unwrap();
        for (index, ((projected, residual), actual)) in
            reference.iter().zip(residual).zip(output).enumerate()
        {
            let expected = bf16(projected + f32::from_bits(u32::from(residual) << 16));
            assert!(actual.is_finite());
            assert_eq!(
                actual.to_bits(),
                u32::from(expected) << 16,
                "residual mismatch at {index}"
            );
        }
    }
}

fn upload<T>(device: &Device, values: &[T]) -> Buffer {
    device.new_buffer_with_data(
        values.as_ptr().cast(),
        std::mem::size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[allow(unexpected_cfgs)]
fn gpu_milliseconds(command: &CommandBufferRef) -> f64 {
    // Metal publishes GPU timestamps only after command completion.
    let start: f64 = unsafe { msg_send![command, GPUStartTime] };
    let end: f64 = unsafe { msg_send![command, GPUEndTime] };
    assert!(end > start, "Metal GPU timestamps are unavailable");
    (end - start) * 1000.0
}

#[test]
#[ignore = "requires native Metal; validates full Qwen verification matrices"]
fn verification_gate_full_shape_is_bit_exact() {
    let device = Device::system_default().expect("native Metal is required for this validation");
    let library = MetalLibrary::compile_source(&device, SOURCE).unwrap();
    let generic = compute_pipeline(&device, &library, GATE_UP).unwrap();
    let queue = device.new_command_queue();
    for (rows, name) in [
        (VERIFY_ROWS, VERIFY5_EXACT_GATE_UP),
        (VERIFY8_ROWS, VERIFY8_EXACT_GATE_UP),
    ] {
        let pipeline = compute_pipeline(&device, &library, name).unwrap();
        for outputs in [17, QWEN_INTERMEDIATE_SIZE] {
            let fixture = GateFixture::new(&device, rows, QWEN_HIDDEN_SIZE, outputs);
            fixture.run(&queue, &generic, true);
            fixture.run(&queue, &pipeline, false);
            fixture.assert_exact();
        }
    }
}

#[test]
#[ignore = "requires native Metal; validates production-size Qwen verification projections"]
fn verification_linear_full_shape_is_bit_exact() {
    let device = Device::system_default().expect("native Metal is required for this validation");
    let library = MetalLibrary::compile_source(&device, SOURCE).unwrap();
    let generic = compute_pipeline(&device, &library, LINEAR).unwrap();
    let queue = device.new_command_queue();
    for (rows, name, add_name) in [
        (VERIFY_ROWS, VERIFY5_EXACT_LINEAR, VERIFY5_EXACT_LINEAR_ADD),
        (VERIFY8_ROWS, VERIFY8_EXACT_LINEAR, VERIFY8_EXACT_LINEAR_ADD),
    ] {
        let pipeline = compute_pipeline(&device, &library, name).unwrap();
        let add_pipeline = compute_pipeline(&device, &library, add_name).unwrap();
        for (width, outputs) in [(5120, 17), (5120, 10240), (17408, 5120)] {
            let fixture = GateFixture::new(&device, rows, width, outputs);
            fixture.run_linear(&queue, &generic, true);
            fixture.run_linear(&queue, &pipeline, false);
            fixture.assert_exact();
            fixture.check_linear_add(&device, &queue, &add_pipeline);
        }
    }
}

#[test]
#[ignore = "requires native Metal; reports current full-size verification gate latency"]
fn benchmark_verification_gate() {
    let device = Device::system_default().expect("native Metal is required for this benchmark");
    let queue = device.new_command_queue();
    let library = MetalLibrary::compile_source(&device, SOURCE).unwrap();
    let generic = compute_pipeline(&device, &library, GATE_UP).unwrap();
    for (rows, name) in [
        (VERIFY_ROWS, VERIFY5_EXACT_GATE_UP),
        (VERIFY8_ROWS, VERIFY8_EXACT_GATE_UP),
    ] {
        let fixture = GateFixture::new(&device, rows, QWEN_HIDDEN_SIZE, QWEN_INTERMEDIATE_SIZE);
        let pipeline = compute_pipeline(&device, &library, name).unwrap();
        fixture.run(&queue, &generic, true);
        let mut samples = Vec::new();
        for iteration in 0..24 {
            let elapsed = fixture.run(&queue, &pipeline, false);
            if iteration >= 4 {
                samples.push(elapsed);
            }
        }
        fixture.assert_exact();
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "rows={rows} median_ms={:.6} min_ms={:.6} max_ms={:.6}",
            samples[samples.len() / 2],
            samples[0],
            samples[samples.len() - 1]
        );
    }
}
