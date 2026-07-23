use std::sync::Arc;

use ::metal::{Buffer, CommandBufferRef, ComputePipelineState, Device};
use common::{Error, Result};

use crate::{DeviceW4Weight, W4ExpertGroup, W4WeightSource};

use super::{
    arena::MetalArena,
    buffers::{require_f32_capacity, u32_buffer, u64_buffer, u8_buffer, u8_buffer_no_copy},
    command::{encode_1d, encode_1d_with_indirect_reads},
    library::MetalLibrary,
    pipeline::compute_pipeline,
};

const GROUPWISE_MATVEC_KERNEL: &str = "w4_groupwise_matvec_f32_kernel";
const GROUPWISE_GATE_UP_SWIGLU_KERNEL: &str = "w4_groupwise_gate_up_swiglu_f32_kernel";
const EXPERT_GATE_UP_SWIGLU_KERNEL: &str = "w4_groupwise_expert_gate_up_swiglu_f32_kernel";
const EXPERT_DOWN_KERNEL: &str = "w4_groupwise_expert_down_f32_kernel";
const SIMD_LANES: usize = 32;
const PACKED_VALUES_PER_I32: usize = 8;
const SUPPORTED_GROUP_SIZE: usize = 32;

pub(crate) struct MetalW4 {
    matvec_pipeline: ComputePipelineState,
    gate_up_swiglu_pipeline: ComputePipelineState,
    expert_gate_up_swiglu_pipeline: ComputePipelineState,
    expert_down_pipeline: ComputePipelineState,
    arena: MetalArena,
}

impl MetalW4 {
    pub(crate) fn new(device: &Device, library: &MetalLibrary, arena: MetalArena) -> Result<Self> {
        let matvec_pipeline = compute_pipeline(device, library, GROUPWISE_MATVEC_KERNEL)?;
        let gate_up_swiglu_pipeline =
            compute_pipeline(device, library, GROUPWISE_GATE_UP_SWIGLU_KERNEL)?;
        let expert_gate_up_swiglu_pipeline =
            compute_pipeline(device, library, EXPERT_GATE_UP_SWIGLU_KERNEL)?;
        let expert_down_pipeline = compute_pipeline(device, library, EXPERT_DOWN_KERNEL)?;
        require_simd_width(&matvec_pipeline)?;
        require_simd_width(&gate_up_swiglu_pipeline)?;
        require_simd_width(&expert_gate_up_swiglu_pipeline)?;
        require_simd_width(&expert_down_pipeline)?;
        Ok(Self {
            matvec_pipeline,
            gate_up_swiglu_pipeline,
            expert_gate_up_swiglu_pipeline,
            expert_down_pipeline,
            arena,
        })
    }

    pub(crate) fn prepare(
        &self,
        device: &Device,
        packed: &[u8],
        scales: &[u8],
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> Result<DeviceW4Weight> {
        validate_weight_layout(
            packed.len(),
            scales.len(),
            in_features,
            out_features,
            group_size,
        )?;

        Ok(DeviceW4Weight {
            in_features,
            out_features,
            group_size,
            packed_bytes: packed.len(),
            scale_bytes: scales.len(),
            packed: u8_buffer(device, packed)?,
            scales: u8_buffer(device, scales)?,
            _source_owner: None,
        })
    }

    /// Creates immutable Metal views over model-owned mmap bytes.
    ///
    /// The caller must retain both source slices until every clone of the
    /// returned weight has been dropped.
    pub(crate) fn prepare_no_copy(
        &self,
        device: &Device,
        source: Arc<dyn W4WeightSource>,
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> Result<DeviceW4Weight> {
        let (packed_bytes, scale_bytes, packed, scales) = {
            let packed = source.packed_bytes()?;
            let scales = source.scale_bytes()?;
            validate_weight_layout(
                packed.len(),
                scales.len(),
                in_features,
                out_features,
                group_size,
            )?;
            (
                packed.len(),
                scales.len(),
                u8_buffer_no_copy(device, packed)?,
                u8_buffer_no_copy(device, scales)?,
            )
        };

        Ok(DeviceW4Weight {
            in_features,
            out_features,
            group_size,
            packed_bytes,
            scale_bytes,
            packed,
            scales,
            _source_owner: Some(source),
        })
    }

    pub(crate) fn encode_matvec(
        &self,
        command_buffer: &CommandBufferRef,
        weight: &DeviceW4Weight,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Buffer> {
        self.encode(
            command_buffer,
            &self.matvec_pipeline,
            weight,
            None,
            input,
            input_len,
            row_count,
        )
    }

    pub(crate) fn encode_gate_up_swiglu(
        &self,
        command_buffer: &CommandBufferRef,
        gate: &DeviceW4Weight,
        up: &DeviceW4Weight,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Buffer> {
        if gate.in_features != up.in_features
            || gate.out_features != up.out_features
            || gate.group_size != up.group_size
        {
            return Err(Error::backend(format!(
                "W4 gate/up layout mismatch: gate=[{}, {}, group {}], up=[{}, {}, group {}]",
                gate.out_features,
                gate.in_features,
                gate.group_size,
                up.out_features,
                up.in_features,
                up.group_size
            )));
        }
        self.encode(
            command_buffer,
            &self.gate_up_swiglu_pipeline,
            gate,
            Some(up),
            input,
            input_len,
            row_count,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_expert_wave(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        groups: &[W4ExpertGroup<'_>],
        input: &Buffer,
        input_len: usize,
        token_count: usize,
        top_k: usize,
        destination: &Buffer,
        destination_len: usize,
    ) -> Result<()> {
        let first = groups
            .first()
            .ok_or_else(|| Error::backend("W4 expert wave requires at least one group"))?;
        if token_count == 0 || top_k == 0 {
            return Err(Error::backend(
                "W4 expert wave token_count and top_k must be positive",
            ));
        }

        let hidden_size = first.gate.in_features;
        let intermediate_size = first.gate.out_features;
        let assignment_count = token_count
            .checked_mul(top_k)
            .ok_or_else(|| Error::backend("W4 expert wave assignment count overflow"))?;
        let expected_input_len = token_count
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("W4 expert wave input length overflow"))?;
        let expected_destination_len = assignment_count
            .checked_mul(hidden_size)
            .ok_or_else(|| Error::backend("W4 expert wave destination length overflow"))?;
        if input_len != expected_input_len || destination_len != expected_destination_len {
            return Err(Error::backend(format!(
                "W4 expert wave length mismatch: input {input_len}/{expected_input_len}, destination {destination_len}/{expected_destination_len}"
            )));
        }
        require_f32_capacity(input, input_len, "W4 expert wave input")?;
        require_f32_capacity(destination, destination_len, "W4 expert wave destination")?;

        let mut assignment_indices = Vec::<u32>::new();
        let mut assignment_groups = Vec::<u32>::new();
        let mut seen_assignments = vec![false; assignment_count];
        let mut gate_packed_addresses = Vec::<u64>::with_capacity(groups.len());
        let mut gate_scale_addresses = Vec::<u64>::with_capacity(groups.len());
        let mut up_packed_addresses = Vec::<u64>::with_capacity(groups.len());
        let mut up_scale_addresses = Vec::<u64>::with_capacity(groups.len());
        let mut down_packed_addresses = Vec::<u64>::with_capacity(groups.len());
        let mut down_scale_addresses = Vec::<u64>::with_capacity(groups.len());
        let mut gate_up_resources = Vec::<&Buffer>::with_capacity(groups.len() * 4);
        let mut down_resources = Vec::<&Buffer>::with_capacity(groups.len() * 2);
        for (group_index, group) in groups.iter().enumerate() {
            validate_expert_layout(group, hidden_size, intermediate_size, first.gate.group_size)?;
            if group.assignment_indices.is_empty() {
                return Err(Error::backend(
                    "W4 expert wave groups must contain at least one assignment",
                ));
            }
            for &assignment in group.assignment_indices {
                let assignment_usize = assignment as usize;
                if assignment_usize >= assignment_count {
                    return Err(Error::backend(format!(
                        "W4 expert wave assignment {assignment} exceeds {assignment_count}"
                    )));
                }
                if std::mem::replace(&mut seen_assignments[assignment_usize], true) {
                    return Err(Error::backend(format!(
                        "W4 expert wave assignment {assignment} appears more than once"
                    )));
                }
                assignment_indices.push(assignment);
                assignment_groups.push(as_u32(group_index, "ready expert group index")?);
            }

            gate_packed_addresses.push(buffer_address(&group.gate.packed, "gate packed")?);
            gate_scale_addresses.push(buffer_address(&group.gate.scales, "gate scales")?);
            up_packed_addresses.push(buffer_address(&group.up.packed, "up packed")?);
            up_scale_addresses.push(buffer_address(&group.up.scales, "up scales")?);
            down_packed_addresses.push(buffer_address(&group.down.packed, "down packed")?);
            down_scale_addresses.push(buffer_address(&group.down.scales, "down scales")?);
            gate_up_resources.extend([
                &group.gate.packed,
                &group.gate.scales,
                &group.up.packed,
                &group.up.scales,
            ]);
            down_resources.extend([&group.down.packed, &group.down.scales]);
        }

        let ready_assignment_count = assignment_indices.len();
        let gated_len = ready_assignment_count
            .checked_mul(intermediate_size)
            .ok_or_else(|| Error::backend("W4 expert gated length overflow"))?;
        let gated = self.arena.empty_f32(gated_len)?;
        let gate_packed_addresses = u64_buffer(device, &gate_packed_addresses)?;
        let gate_scale_addresses = u64_buffer(device, &gate_scale_addresses)?;
        let up_packed_addresses = u64_buffer(device, &up_packed_addresses)?;
        let up_scale_addresses = u64_buffer(device, &up_scale_addresses)?;
        let down_packed_addresses = u64_buffer(device, &down_packed_addresses)?;
        let down_scale_addresses = u64_buffer(device, &down_scale_addresses)?;
        let assignment_indices = u32_buffer(device, &assignment_indices)?;
        let assignment_groups = u32_buffer(device, &assignment_groups)?;
        let token_count = self.arena.u32(as_u32(token_count, "token_count")?)?;
        let top_k = self.arena.u32(as_u32(top_k, "top_k")?)?;
        let assignment_count = self
            .arena
            .u32(as_u32(assignment_count, "assignment_count")?)?;
        let ready_assignment_count_buffer = self
            .arena
            .u32(as_u32(ready_assignment_count, "ready_assignment_count")?)?;
        let hidden_size_buffer = self.arena.u32(as_u32(hidden_size, "hidden_size")?)?;
        let intermediate_size_buffer = self
            .arena
            .u32(as_u32(intermediate_size, "intermediate_size")?)?;
        let group_size = first.gate.group_size;
        let group_size_buffer = self.arena.u32(as_u32(group_size, "group_size")?)?;
        let gate_packed_words = self.arena.u32(as_u32(
            hidden_size / PACKED_VALUES_PER_I32,
            "gate packed words per row",
        )?)?;
        let gate_groups = self
            .arena
            .u32(as_u32(hidden_size / group_size, "gate groups per row")?)?;
        let down_packed_words = self.arena.u32(as_u32(
            intermediate_size / PACKED_VALUES_PER_I32,
            "down packed words per row",
        )?)?;
        let down_groups = self.arena.u32(as_u32(
            intermediate_size / group_size,
            "down groups per row",
        )?)?;

        let gate_threads = ready_assignment_count
            .checked_mul(intermediate_size)
            .and_then(|count| count.checked_mul(SIMD_LANES))
            .ok_or_else(|| Error::backend("W4 expert gate/up thread count overflow"))?;
        encode_1d_with_indirect_reads(
            command_buffer,
            &self.expert_gate_up_swiglu_pipeline,
            &[
                &gate_packed_addresses,
                &gate_scale_addresses,
                &up_packed_addresses,
                &up_scale_addresses,
                input,
                &assignment_indices,
                &assignment_groups,
                &gated,
                &token_count,
                &top_k,
                &ready_assignment_count_buffer,
                &hidden_size_buffer,
                &intermediate_size_buffer,
                &gate_packed_words,
                &gate_groups,
                &group_size_buffer,
            ],
            &gate_up_resources,
            gate_threads,
        )?;

        let down_threads = ready_assignment_count
            .checked_mul(hidden_size)
            .and_then(|count| count.checked_mul(SIMD_LANES))
            .ok_or_else(|| Error::backend("W4 expert down thread count overflow"))?;
        encode_1d_with_indirect_reads(
            command_buffer,
            &self.expert_down_pipeline,
            &[
                &down_packed_addresses,
                &down_scale_addresses,
                &gated,
                &assignment_indices,
                &assignment_groups,
                destination,
                &assignment_count,
                &ready_assignment_count_buffer,
                &intermediate_size_buffer,
                &hidden_size_buffer,
                &down_packed_words,
                &down_groups,
                &group_size_buffer,
            ],
            &down_resources,
            down_threads,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        pipeline: &ComputePipelineState,
        first: &DeviceW4Weight,
        second: Option<&DeviceW4Weight>,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Buffer> {
        if row_count == 0 {
            return Err(Error::backend("W4 row_count must be positive"));
        }
        let expected_input_len = row_count
            .checked_mul(first.in_features)
            .ok_or_else(|| Error::backend("W4 input element count overflow"))?;
        if input_len != expected_input_len {
            return Err(Error::backend(format!(
                "W4 input length mismatch: expected {expected_input_len}, got {input_len}"
            )));
        }
        require_f32_capacity(input, input_len, "W4 input")?;

        let output_len = row_count
            .checked_mul(first.out_features)
            .ok_or_else(|| Error::backend("W4 output element count overflow"))?;
        let output = self.arena.empty_f32(output_len)?;
        let row_count = self.arena.u32(as_u32(row_count, "row_count")?)?;
        let in_features = self.arena.u32(as_u32(first.in_features, "in_features")?)?;
        let out_features = self
            .arena
            .u32(as_u32(first.out_features, "out_features")?)?;
        let packed_words_per_row = self.arena.u32(as_u32(
            first.in_features / PACKED_VALUES_PER_I32,
            "packed_words_per_row",
        )?)?;
        let groups_per_row = self.arena.u32(as_u32(
            first.in_features / first.group_size,
            "groups_per_row",
        )?)?;
        let group_size = self.arena.u32(as_u32(first.group_size, "group_size")?)?;
        let thread_count = output_len
            .checked_mul(SIMD_LANES)
            .ok_or_else(|| Error::backend("W4 thread count overflow"))?;

        match second {
            None => encode_1d(
                command_buffer,
                pipeline,
                &[
                    &first.packed,
                    &first.scales,
                    input,
                    &output,
                    &row_count,
                    &in_features,
                    &out_features,
                    &packed_words_per_row,
                    &groups_per_row,
                    &group_size,
                ],
                thread_count,
            )?,
            Some(second) => encode_1d(
                command_buffer,
                pipeline,
                &[
                    &first.packed,
                    &first.scales,
                    &second.packed,
                    &second.scales,
                    input,
                    &output,
                    &row_count,
                    &in_features,
                    &out_features,
                    &packed_words_per_row,
                    &groups_per_row,
                    &group_size,
                ],
                thread_count,
            )?,
        }
        Ok(output)
    }
}

fn validate_expert_layout(
    group: &W4ExpertGroup<'_>,
    hidden_size: usize,
    intermediate_size: usize,
    group_size: usize,
) -> Result<()> {
    for (label, weight, in_features, out_features) in [
        ("gate", group.gate, hidden_size, intermediate_size),
        ("up", group.up, hidden_size, intermediate_size),
        ("down", group.down, intermediate_size, hidden_size),
    ] {
        if weight.in_features != in_features
            || weight.out_features != out_features
            || weight.group_size != group_size
        {
            return Err(Error::backend(format!(
                "W4 expert {label} layout must be [{out_features},{in_features}] group {group_size}, got [{},{}] group {}",
                weight.out_features, weight.in_features, weight.group_size
            )));
        }
    }
    Ok(())
}

fn buffer_address(buffer: &Buffer, label: &str) -> Result<u64> {
    let address = buffer.gpu_address();
    if address == 0 {
        return Err(Error::backend(format!(
            "W4 expert {label} buffer has no GPU address"
        )));
    }
    Ok(address)
}

fn validate_weight_layout(
    packed_bytes: usize,
    scale_bytes: usize,
    in_features: usize,
    out_features: usize,
    group_size: usize,
) -> Result<()> {
    if in_features == 0 || out_features == 0 {
        return Err(Error::backend("W4 matrix dimensions must be positive"));
    }
    if group_size != SUPPORTED_GROUP_SIZE {
        return Err(Error::backend(format!(
            "W4 group size must be {SUPPORTED_GROUP_SIZE}, got {group_size}"
        )));
    }
    if !in_features.is_multiple_of(group_size) {
        return Err(Error::backend(format!(
            "W4 in_features {in_features} must be divisible by group size {group_size}"
        )));
    }
    let expected_packed = out_features
        .checked_mul(in_features / PACKED_VALUES_PER_I32)
        .and_then(|words| words.checked_mul(std::mem::size_of::<u32>()))
        .ok_or_else(|| Error::backend("W4 packed byte count overflow"))?;
    let expected_scales = out_features
        .checked_mul(in_features / group_size)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .ok_or_else(|| Error::backend("W4 scale byte count overflow"))?;
    if packed_bytes != expected_packed {
        return Err(Error::backend(format!(
            "W4 packed byte length mismatch: expected {expected_packed}, got {packed_bytes}"
        )));
    }
    if scale_bytes != expected_scales {
        return Err(Error::backend(format!(
            "W4 scale byte length mismatch: expected {expected_scales}, got {scale_bytes}"
        )));
    }
    Ok(())
}

fn require_simd_width(pipeline: &ComputePipelineState) -> Result<()> {
    let width = pipeline.thread_execution_width() as usize;
    if width != SIMD_LANES {
        return Err(Error::backend(format!(
            "W4 kernels require a {SIMD_LANES}-lane SIMD group, device reports {width}"
        )));
    }
    Ok(())
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::backend(format!("W4 {label} exceeds Metal u32 limit")))
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use std::sync::Arc;

    use common::{Device, F32Tensor, Result};

    use crate::{Backend, MetalBackend, W4ExpertGroup, W4WeightSource};

    const GROUP_SIZE: usize = 32;

    #[derive(Debug)]
    struct OwnedW4Source {
        packed: Vec<u8>,
        scales: Vec<u8>,
    }

    impl W4WeightSource for OwnedW4Source {
        fn packed_bytes(&self) -> Result<&[u8]> {
            Ok(&self.packed)
        }

        fn scale_bytes(&self) -> Result<&[u8]> {
            Ok(&self.scales)
        }
    }

    #[test]
    fn groupwise_matvec_matches_cpu_reference() {
        let Some(backend) = native_backend_or_skip() else {
            return;
        };
        let quantized = [
            (0..GROUP_SIZE)
                .map(|index| (index % 16) as i8 - 8)
                .collect::<Vec<_>>(),
            (0..GROUP_SIZE)
                .map(|index| 7 - (index % 16) as i8)
                .collect::<Vec<_>>(),
        ];
        let packed = pack_rows(&quantized);
        let scales = bf16_bytes(&[0.5, 0.25]);
        let input_values = (0..GROUP_SIZE)
            .map(|index| (index as f32 - 12.0) / 8.0)
            .collect::<Vec<_>>();
        let input = F32Tensor::new(input_values.clone(), [1, GROUP_SIZE]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let weight = backend
            .prepare_w4_groupwise_weight(&packed, &scales, GROUP_SIZE, quantized.len(), GROUP_SIZE)
            .unwrap()
            .unwrap();
        let output = backend
            .w4_groupwise_matvec_device(&weight, &input)
            .unwrap()
            .unwrap();
        let actual = backend.device_download_f32_tensor(&output).unwrap();
        let expected = quantized
            .iter()
            .zip([0.5_f32, 0.25])
            .map(|(row, scale)| {
                row.iter()
                    .zip(&input_values)
                    .map(|(weight, input)| f32::from(*weight) * scale * input)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        assert_close(actual.values(), &expected, 1e-5);
    }

    #[test]
    fn no_copy_groupwise_matvec_reads_model_owned_bytes() {
        let Some(backend) = native_backend_or_skip() else {
            return;
        };
        let source = Arc::new(OwnedW4Source {
            packed: pack_rows(&[vec![2_i8; GROUP_SIZE]]),
            scales: bf16_bytes(&[0.25]),
        });
        let weight = backend
            .prepare_w4_groupwise_weight_no_copy(source, GROUP_SIZE, 1, GROUP_SIZE)
            .unwrap()
            .unwrap();
        let input = F32Tensor::new(vec![0.5_f32; GROUP_SIZE], [1, GROUP_SIZE]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let output = backend
            .w4_groupwise_matvec_device(&weight, &input)
            .unwrap()
            .unwrap();
        let actual = backend.device_download_f32_tensor(&output).unwrap();

        assert_close(actual.values(), &[8.0], 1e-5);
    }

    #[test]
    fn fused_gate_up_swiglu_matches_cpu_reference() {
        let Some(backend) = native_backend_or_skip() else {
            return;
        };
        let gate_rows = [vec![1_i8; GROUP_SIZE], vec![-2_i8; GROUP_SIZE]];
        let up_rows = [vec![3_i8; GROUP_SIZE], vec![2_i8; GROUP_SIZE]];
        let gate = backend
            .prepare_w4_groupwise_weight(
                &pack_rows(&gate_rows),
                &bf16_bytes(&[0.25, 0.5]),
                GROUP_SIZE,
                2,
                GROUP_SIZE,
            )
            .unwrap()
            .unwrap();
        let up = backend
            .prepare_w4_groupwise_weight(
                &pack_rows(&up_rows),
                &bf16_bytes(&[0.5, 0.25]),
                GROUP_SIZE,
                2,
                GROUP_SIZE,
            )
            .unwrap()
            .unwrap();
        let input_values = vec![0.125_f32; GROUP_SIZE];
        let input = F32Tensor::new(input_values.clone(), [1, GROUP_SIZE]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let output = backend
            .w4_groupwise_gate_up_swiglu_device(&gate, &up, &input)
            .unwrap()
            .unwrap();
        let actual = backend.device_download_f32_tensor(&output).unwrap();
        let expected = [
            (1.0_f32, 0.25_f32, 3.0_f32, 0.5_f32),
            (-2.0, 0.5, 2.0, 0.25),
        ]
        .into_iter()
        .map(|(gate_q, gate_scale, up_q, up_scale)| {
            let gate = input_values.iter().sum::<f32>() * gate_q * gate_scale;
            let up = input_values.iter().sum::<f32>() * up_q * up_scale;
            (gate / (1.0 + (-gate).exp())) * up
        })
        .collect::<Vec<_>>();
        assert_close(actual.values(), &expected, 1e-5);
    }

    #[test]
    fn ready_expert_wave_batches_experts_and_preserves_assignment_rows() {
        let Some(backend) = native_backend_or_skip() else {
            return;
        };
        let expert_zero = prepare_uniform_expert(&backend, 1, 0.25, 2, 0.5, 1, 0.125);
        let expert_one = prepare_uniform_expert(&backend, -1, 0.5, 3, 0.25, -2, 0.125);
        let input_values = [vec![0.125_f32; GROUP_SIZE], vec![0.25_f32; GROUP_SIZE]].concat();
        let input = F32Tensor::new(input_values.clone(), [2, GROUP_SIZE]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let output = backend
            .device_alloc_f32_tensor(&[4, GROUP_SIZE])
            .unwrap()
            .unwrap();
        let zero_assignments = [0_u32, 3];
        let one_assignments = [1_u32, 2];
        let groups = [
            W4ExpertGroup::new(
                &expert_zero.0,
                &expert_zero.1,
                &expert_zero.2,
                &zero_assignments,
            ),
            W4ExpertGroup::new(
                &expert_one.0,
                &expert_one.1,
                &expert_one.2,
                &one_assignments,
            ),
        ];
        backend
            .w4_groupwise_expert_wave_device(&groups, &input, 2, 2, &output)
            .unwrap()
            .unwrap();
        let actual = backend.device_download_f32_tensor(&output).unwrap();

        let token_sums = [
            input_values[..GROUP_SIZE].iter().sum::<f32>(),
            input_values[GROUP_SIZE..].iter().sum::<f32>(),
        ];
        let expert_value = |token_sum: f32,
                            gate_q: f32,
                            gate_scale: f32,
                            up_q: f32,
                            up_scale: f32,
                            down_q: f32,
                            down_scale: f32| {
            let gate = token_sum * gate_q * gate_scale;
            let up = token_sum * up_q * up_scale;
            let activated = (gate / (1.0 + (-gate).exp())) * up;
            activated * GROUP_SIZE as f32 * down_q * down_scale
        };
        let expected_rows = [
            expert_value(token_sums[0], 1.0, 0.25, 2.0, 0.5, 1.0, 0.125),
            expert_value(token_sums[0], -1.0, 0.5, 3.0, 0.25, -2.0, 0.125),
            expert_value(token_sums[1], -1.0, 0.5, 3.0, 0.25, -2.0, 0.125),
            expert_value(token_sums[1], 1.0, 0.25, 2.0, 0.5, 1.0, 0.125),
        ];
        let expected = expected_rows
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value, GROUP_SIZE))
            .collect::<Vec<_>>();
        assert_close(actual.values(), &expected, 1e-4);
    }

    fn prepare_uniform_expert(
        backend: &MetalBackend,
        gate_value: i8,
        gate_scale: f32,
        up_value: i8,
        up_scale: f32,
        down_value: i8,
        down_scale: f32,
    ) -> (
        crate::DeviceW4Weight,
        crate::DeviceW4Weight,
        crate::DeviceW4Weight,
    ) {
        let prepare = |value: i8, scale: f32| {
            let rows = vec![vec![value; GROUP_SIZE]; GROUP_SIZE];
            backend
                .prepare_w4_groupwise_weight(
                    &pack_rows(&rows),
                    &bf16_bytes(&[scale; GROUP_SIZE]),
                    GROUP_SIZE,
                    GROUP_SIZE,
                    GROUP_SIZE,
                )
                .unwrap()
                .unwrap()
        };
        (
            prepare(gate_value, gate_scale),
            prepare(up_value, up_scale),
            prepare(down_value, down_scale),
        )
    }

    fn native_backend_or_skip() -> Option<MetalBackend> {
        let backend = MetalBackend::from_device(Device::Metal).ok()?;
        backend.device_values_supported().then_some(backend)
    }

    fn pack_rows(rows: &[Vec<i8>]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(rows.iter().map(Vec::len).sum::<usize>() / 2);
        for row in rows {
            for values in row.chunks_exact(8) {
                let mut word = 0_u32;
                for (index, value) in values.iter().enumerate() {
                    let encoded = u32::from((*value + 8) as u8);
                    word |= encoded << (index * 4);
                }
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        bytes
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value {index} differs: actual={actual}, expected={expected}"
            );
        }
    }
}
