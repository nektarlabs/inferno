use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use ::metal::{Buffer, CommandBufferRef, ComputePipelineState, Device};
use common::{Error, Result};

use crate::{DeviceRouterTopK, GgufExpertQuant};

use super::{
    arena::MetalArena,
    buffers::{require_f32_capacity, u8_buffer_no_copy},
    command::{encode_1d_threadgroups_args, KernelArg},
    laguna_views::MetalLagunaViews,
    library::MetalLibrary,
    pipeline::compute_pipeline,
};

const Q2_BLOCK_BYTES: usize = 84;
const Q3_BLOCK_BYTES: usize = 110;
const BLOCK_VALUES: usize = 256;
const SIMD_LANES: usize = 32;
const SIMDGROUPS_PER_THREADGROUP: usize = 2;
const Q3_ROWS_PER_SIMDGROUP: usize = 4;
const Q2_ROWS_PER_SIMDGROUP: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct WeightKey {
    address: usize,
    byte_len: usize,
}

pub(crate) struct MetalGgufMoe {
    q2_gate_up: ComputePipelineState,
    q2_down: ComputePipelineState,
    q3_gate_up: ComputePipelineState,
    q3_down: ComputePipelineState,
    arena: MetalArena,
    weights: Mutex<HashMap<WeightKey, Buffer>>,
    laguna_views: Arc<MetalLagunaViews>,
}

struct WeightBuffer {
    storage: Buffer,
    byte_offset: usize,
}

impl MetalGgufMoe {
    pub(crate) fn new(
        device: &Device,
        library: &MetalLibrary,
        arena: MetalArena,
        laguna_views: Arc<MetalLagunaViews>,
    ) -> Result<Self> {
        Ok(Self {
            q2_gate_up: compute_pipeline(device, library, "laguna_q2_expert_gate_up_f32_kernel")?,
            q2_down: compute_pipeline(device, library, "laguna_q2_expert_down_sum_f32_kernel")?,
            q3_gate_up: compute_pipeline(device, library, "laguna_q3_expert_gate_up_f32_kernel")?,
            q3_down: compute_pipeline(device, library, "laguna_q3_expert_down_sum_f32_kernel")?,
            arena,
            weights: Mutex::new(HashMap::new()),
            laguna_views,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        quant: GgufExpertQuant,
        input: &Buffer,
        input_len: usize,
        routing: &DeviceRouterTopK,
        in_features: usize,
        intermediate_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        let token_count = routing.token_count;
        let top_k = routing.top_k;
        let expert_count = routing.expert_count;
        if token_count == 0 || top_k == 0 || top_k > expert_count {
            return Err(Error::backend(format!(
                "Laguna GGUF routing dimensions are invalid: tokens={token_count}, top_k={top_k}, experts={expert_count}"
            )));
        }
        if input_len
            != token_count
                .checked_mul(in_features)
                .ok_or_else(|| Error::backend("Laguna GGUF MoE input length overflow"))?
        {
            return Err(Error::backend(format!(
                "Laguna GGUF MoE expected {} input values, got {input_len}",
                token_count * in_features
            )));
        }
        require_f32_capacity(input, input_len, "Laguna GGUF MoE input")?;
        if !in_features.is_multiple_of(BLOCK_VALUES)
            || !intermediate_features.is_multiple_of(BLOCK_VALUES)
            || out_features == 0
        {
            return Err(Error::backend(format!(
                "Laguna GGUF MoE dimensions must use 256-wide quant blocks, got in={in_features}, intermediate={intermediate_features}, out={out_features}"
            )));
        }

        let block_bytes = match quant {
            GgufExpertQuant::Q2K => Q2_BLOCK_BYTES,
            GgufExpertQuant::Q3K => Q3_BLOCK_BYTES,
        };
        let gate_blocks_per_row = in_features / BLOCK_VALUES;
        let down_blocks_per_row = intermediate_features / BLOCK_VALUES;
        let gate_expert_stride =
            quantized_expert_stride(intermediate_features, gate_blocks_per_row, block_bytes)?;
        let down_expert_stride =
            quantized_expert_stride(out_features, down_blocks_per_row, block_bytes)?;
        validate_weight_len("gate", gate_weights, expert_count, gate_expert_stride)?;
        validate_weight_len("up", up_weights, expert_count, gate_expert_stride)?;
        validate_weight_len("down", down_weights, expert_count, down_expert_stride)?;

        let assignment_count = routing.assignment_count()?;
        let intermediate_len = assignment_count
            .checked_mul(intermediate_features)
            .ok_or_else(|| Error::backend("Laguna GGUF MoE intermediate length overflow"))?;
        let output_len = token_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Laguna GGUF MoE output length overflow"))?;
        let intermediate = self.arena.empty_f32(intermediate_len)?;
        let output = self.arena.empty_f32(output_len)?;
        let gate = self.weight_buffer(device, gate_weights)?;
        let up = self.weight_buffer(device, up_weights)?;
        let down = self.weight_buffer(device, down_weights)?;

        let gate_buffers = [
            KernelArg::BufferOffset(&gate.storage, gate.byte_offset),
            KernelArg::BufferOffset(&up.storage, up.byte_offset),
            KernelArg::Buffer(input),
            KernelArg::Buffer(&routing.token_indices),
            KernelArg::Buffer(&routing.expert_ids),
            KernelArg::Buffer(&routing.expert_weights),
            KernelArg::Buffer(&intermediate),
            KernelArg::U32(as_u32(assignment_count, "assignments")?),
            KernelArg::U32(as_u32(in_features, "input width")?),
            KernelArg::U32(as_u32(intermediate_features, "intermediate width")?),
            KernelArg::U32(as_u32(gate_blocks_per_row, "gate blocks per row")?),
            KernelArg::U32(as_u32(gate_expert_stride, "gate expert stride")?),
        ];
        let down_buffers = [
            KernelArg::BufferOffset(&down.storage, down.byte_offset),
            KernelArg::Buffer(&routing.expert_ids),
            KernelArg::Buffer(&intermediate),
            KernelArg::Buffer(&output),
            KernelArg::U32(as_u32(token_count, "tokens")?),
            KernelArg::U32(as_u32(top_k, "top-k")?),
            KernelArg::U32(as_u32(intermediate_features, "intermediate width")?),
            KernelArg::U32(as_u32(out_features, "output width")?),
            KernelArg::U32(as_u32(down_blocks_per_row, "down blocks per row")?),
            KernelArg::U32(as_u32(down_expert_stride, "down expert stride")?),
        ];

        match quant {
            GgufExpertQuant::Q2K => {
                encode_1d_threadgroups_args(
                    command_buffer,
                    &self.q2_gate_up,
                    &gate_buffers,
                    tiled_threadgroups(
                        assignment_count,
                        intermediate_features,
                        Q2_ROWS_PER_SIMDGROUP,
                    )?,
                    SIMDGROUPS_PER_THREADGROUP * SIMD_LANES,
                )?;
                encode_1d_threadgroups_args(
                    command_buffer,
                    &self.q2_down,
                    &down_buffers,
                    tiled_threadgroups(token_count, out_features, Q2_ROWS_PER_SIMDGROUP)?,
                    SIMDGROUPS_PER_THREADGROUP * SIMD_LANES,
                )?;
            }
            GgufExpertQuant::Q3K => {
                encode_1d_threadgroups_args(
                    command_buffer,
                    &self.q3_gate_up,
                    &gate_buffers,
                    tiled_threadgroups(
                        assignment_count,
                        intermediate_features,
                        Q3_ROWS_PER_SIMDGROUP,
                    )?,
                    SIMDGROUPS_PER_THREADGROUP * SIMD_LANES,
                )?;
                encode_1d_threadgroups_args(
                    command_buffer,
                    &self.q3_down,
                    &down_buffers,
                    tiled_threadgroups(token_count, out_features, Q3_ROWS_PER_SIMDGROUP)?,
                    SIMDGROUPS_PER_THREADGROUP * SIMD_LANES,
                )?;
            }
        }
        Ok(output)
    }

    fn weight_buffer(&self, device: &Device, bytes: &[u8]) -> Result<WeightBuffer> {
        if let Some(binding) = self.laguna_views.binding(bytes)? {
            return Ok(WeightBuffer {
                storage: binding.buffer,
                byte_offset: binding.byte_offset,
            });
        }

        let key = WeightKey {
            address: bytes.as_ptr() as usize,
            byte_len: bytes.len(),
        };
        let mut weights = self
            .weights
            .lock()
            .map_err(|_| Error::backend("Laguna GGUF Metal weight cache lock poisoned"))?;
        if let Some(buffer) = weights.get(&key) {
            return Ok(WeightBuffer {
                storage: buffer.clone(),
                byte_offset: 0,
            });
        }
        let buffer = u8_buffer_no_copy(device, bytes)?;
        weights.insert(key, buffer.clone());
        Ok(WeightBuffer {
            storage: buffer,
            byte_offset: 0,
        })
    }
}

fn quantized_expert_stride(
    rows: usize,
    blocks_per_row: usize,
    block_bytes: usize,
) -> Result<usize> {
    rows.checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(|| Error::backend("Laguna GGUF expert byte stride overflow"))
}

fn validate_weight_len(
    label: &str,
    bytes: &[u8],
    expert_count: usize,
    expert_stride: usize,
) -> Result<()> {
    let expected = expert_count
        .checked_mul(expert_stride)
        .ok_or_else(|| Error::backend("Laguna GGUF expert payload size overflow"))?;
    if bytes.len() != expected {
        return Err(Error::backend(format!(
            "Laguna GGUF {label} expert payload must contain {expected} bytes, got {}",
            bytes.len()
        )));
    }
    Ok(())
}

fn tiled_threadgroups(
    outer_count: usize,
    row_count: usize,
    rows_per_simdgroup: usize,
) -> Result<usize> {
    let simdgroups_per_outer = row_count.div_ceil(rows_per_simdgroup);
    let simdgroup_count = outer_count
        .checked_mul(simdgroups_per_outer)
        .ok_or_else(|| Error::backend("Laguna GGUF tiled SIMD-group count overflow"))?;
    Ok(simdgroup_count.div_ceil(SIMDGROUPS_PER_THREADGROUP))
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::backend(format!("Laguna GGUF MoE {label} exceeds Metal u32")))
}

#[cfg(test)]
mod tests {
    use crate::{Backend, GgufExpertQuant, MetalBackend};
    use common::F32Tensor;

    use super::{BLOCK_VALUES, Q2_BLOCK_BYTES, Q3_BLOCK_BYTES};

    #[test]
    fn q2_selected_expert_matches_patterned_cpu_reference() {
        assert_q2_case(BLOCK_VALUES, BLOCK_VALUES, BLOCK_VALUES);
    }

    #[test]
    fn q2_selected_expert_matches_laguna_widths() {
        assert_q2_case(3072, 1024, 3072);
    }

    #[test]
    fn q2_prefill_matches_laguna_top10_cpu_reference() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let token_count = 4;
        let expert_count = 256;
        let active_experts = 10;
        let in_features = BLOCK_VALUES;
        let intermediate_features = BLOCK_VALUES;
        let out_features = BLOCK_VALUES;
        let input_values = (0..token_count * in_features)
            .map(|index| {
                let token = index / in_features;
                let column = index - token * in_features;
                ((column % 17) as f32 - 8.0 + token as f32) / 32.0
            })
            .collect::<Vec<_>>();
        let input = F32Tensor::new(input_values.clone(), [token_count, in_features]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let logits = F32Tensor::new(
            (0..token_count)
                .flat_map(|_| {
                    (0..expert_count).map(|expert| {
                        if expert < active_experts {
                            10.0_f32 - expert as f32 * 0.1
                        } else {
                            -10.0
                        }
                    })
                })
                .collect::<Vec<_>>(),
            [token_count, expert_count],
        )
        .unwrap();
        let logits = backend.device_upload_f32_tensor(&logits).unwrap().unwrap();
        let routing = backend
            .moe_router_topk_resident_device(
                &logits,
                &vec![0.0; expert_count],
                active_experts,
                true,
                1.0,
            )
            .unwrap()
            .unwrap();

        let gate_pair = patterned_q2_expert_matrix(11, intermediate_features, in_features);
        let up_pair = patterned_q2_expert_matrix(37, intermediate_features, in_features);
        let down_pair = patterned_q2_expert_matrix(83, out_features, intermediate_features);
        let gate_stride = intermediate_features * Q2_BLOCK_BYTES;
        let down_stride = out_features * Q2_BLOCK_BYTES;
        let gate = repeat_first_expert(
            &gate_pair[..gate_stride],
            gate_stride,
            expert_count,
            active_experts,
        );
        let up = repeat_first_expert(
            &up_pair[..gate_stride],
            gate_stride,
            expert_count,
            active_experts,
        );
        let down = repeat_first_expert(
            &down_pair[..down_stride],
            down_stride,
            expert_count,
            active_experts,
        );
        let output = backend
            .laguna_gguf_moe_device(
                &gate,
                &up,
                &down,
                GgufExpertQuant::Q2K,
                &input,
                &routing,
                in_features,
                intermediate_features,
                out_features,
            )
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();

        for token in 0..token_count {
            let input_row = &input_values[token * in_features..(token + 1) * in_features];
            let gate_values =
                cpu_q2_matrix_vector(&gate[..gate_stride], input_row, intermediate_features);
            let up_values =
                cpu_q2_matrix_vector(&up[..gate_stride], input_row, intermediate_features);
            let intermediate = gate_values
                .iter()
                .copied()
                .zip(up_values)
                .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
                .collect::<Vec<_>>();
            let expected = cpu_q2_matrix_vector(&down[..down_stride], &intermediate, out_features);
            let actual = &output.values()[token * out_features..(token + 1) * out_features];
            for (row, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
                let tolerance = 2e-2_f32.max(expected.abs() * 2e-2);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "Q2 prefill GEMM mismatch at token {token}, row {row}: actual={actual}, expected={expected}, tolerance={tolerance}"
                );
            }
        }
    }

    fn assert_q2_case(in_features: usize, intermediate_features: usize, out_features: usize) {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let input_values = (0..in_features)
            .map(|index| ((index % 17) as f32 - 8.0) / 32.0)
            .collect::<Vec<_>>();
        let input = F32Tensor::new(input_values.clone(), [1, in_features]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let logits = F32Tensor::new(vec![10.0, -10.0], [1, 2]).unwrap();
        let logits = backend.device_upload_f32_tensor(&logits).unwrap().unwrap();
        let routing = backend
            .moe_router_topk_resident_device(&logits, &[0.0, 0.0], 1, true, 1.0)
            .unwrap()
            .unwrap();

        let gate = patterned_q2_expert_matrix(11, intermediate_features, in_features);
        let up = patterned_q2_expert_matrix(37, intermediate_features, in_features);
        let down = patterned_q2_expert_matrix(83, out_features, intermediate_features);
        let output = backend
            .laguna_gguf_moe_device(
                &gate,
                &up,
                &down,
                GgufExpertQuant::Q2K,
                &input,
                &routing,
                in_features,
                intermediate_features,
                out_features,
            )
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();

        let gate_values = cpu_q2_matrix_vector(&gate, &input_values, intermediate_features);
        let up_values = cpu_q2_matrix_vector(&up, &input_values, intermediate_features);
        let intermediate = gate_values
            .iter()
            .copied()
            .zip(up_values)
            .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
            .collect::<Vec<_>>();
        let expected = cpu_q2_matrix_vector(&down, &intermediate, out_features);

        for (index, (actual, expected)) in output.values().iter().zip(expected.iter()).enumerate() {
            let tolerance = 2e-4_f32.max(expected.abs() * 2e-4);
            assert!(
                (actual - expected).abs() <= tolerance,
                "patterned Q2 Laguna expert mismatch at {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn q3_selected_expert_matches_known_block_values() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let input = F32Tensor::new(vec![1.0; BLOCK_VALUES], [1, BLOCK_VALUES]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let logits = F32Tensor::new(vec![10.0, -10.0], [1, 2]).unwrap();
        let logits = backend.device_upload_f32_tensor(&logits).unwrap().unwrap();
        let routing = backend
            .moe_router_topk_resident_device(&logits, &[0.0, 0.0], 1, true, 1.0)
            .unwrap()
            .unwrap();

        let selected_block = q3_block(0x1400, [33; 16], 0xe4, 0xff);
        let unused_block = q3_block(0, [33; 16], 0xe4, 0xff);
        let mut weights = Vec::with_capacity(2 * BLOCK_VALUES * Q3_BLOCK_BYTES);
        for block in [&selected_block, &unused_block] {
            for _ in 0..BLOCK_VALUES {
                weights.extend_from_slice(block);
            }
        }
        let output = backend
            .laguna_gguf_moe_device(
                &weights,
                &weights,
                &weights,
                GgufExpertQuant::Q3K,
                &input,
                &routing,
                BLOCK_VALUES,
                BLOCK_VALUES,
                BLOCK_VALUES,
            )
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();

        let dot = 384.0_f32 / 1024.0;
        let expected = (dot / (1.0 + (-dot).exp()) * dot) * dot;
        for actual in output.values() {
            assert!(
                (*actual - expected).abs() <= 1e-4,
                "Q3 Laguna expert mismatch: actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    fn q3_selected_expert_matches_patterned_cpu_reference() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let input_values = (0..BLOCK_VALUES)
            .map(|index| ((index % 13) as f32 - 6.0) / 32.0)
            .collect::<Vec<_>>();
        let input = F32Tensor::new(input_values.clone(), [1, BLOCK_VALUES]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let logits = F32Tensor::new(vec![10.0, -10.0], [1, 2]).unwrap();
        let logits = backend.device_upload_f32_tensor(&logits).unwrap().unwrap();
        let routing = backend
            .moe_router_topk_resident_device(&logits, &[0.0, 0.0], 1, true, 1.0)
            .unwrap()
            .unwrap();

        let gate = patterned_q3_expert_matrix(11);
        let up = patterned_q3_expert_matrix(37);
        let down = patterned_q3_expert_matrix(83);
        let output = backend
            .laguna_gguf_moe_device(
                &gate,
                &up,
                &down,
                GgufExpertQuant::Q3K,
                &input,
                &routing,
                BLOCK_VALUES,
                BLOCK_VALUES,
                BLOCK_VALUES,
            )
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();

        let gate_values = cpu_q3_matrix_vector(&gate, &input_values);
        let up_values = cpu_q3_matrix_vector(&up, &input_values);
        let intermediate = gate_values
            .iter()
            .copied()
            .zip(up_values)
            .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
            .collect::<Vec<_>>();
        let expected = cpu_q3_matrix_vector(&down, &intermediate);

        for (index, (actual, expected)) in output.values().iter().zip(expected.iter()).enumerate() {
            let tolerance = 2e-4_f32.max(expected.abs() * 2e-4);
            assert!(
                (actual - expected).abs() <= tolerance,
                "patterned Q3 Laguna expert mismatch at {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn q3_prefill_matches_laguna_top10_cpu_reference() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let token_count = 4;
        let expert_count = 256;
        let active_experts = 10;
        let input_values = (0..token_count * BLOCK_VALUES)
            .map(|index| {
                let token = index / BLOCK_VALUES;
                let column = index - token * BLOCK_VALUES;
                ((column % 13) as f32 - 6.0 + token as f32) / 32.0
            })
            .collect::<Vec<_>>();
        let input = F32Tensor::new(input_values.clone(), [token_count, BLOCK_VALUES]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let logits = F32Tensor::new(
            (0..token_count)
                .flat_map(|_| {
                    (0..expert_count).map(|expert| {
                        if expert < active_experts {
                            10.0_f32 - expert as f32 * 0.1
                        } else {
                            -10.0
                        }
                    })
                })
                .collect::<Vec<_>>(),
            [token_count, expert_count],
        )
        .unwrap();
        let logits = backend.device_upload_f32_tensor(&logits).unwrap().unwrap();
        let routing = backend
            .moe_router_topk_resident_device(
                &logits,
                &vec![0.0; expert_count],
                active_experts,
                true,
                1.0,
            )
            .unwrap()
            .unwrap();

        let expert_stride = BLOCK_VALUES * Q3_BLOCK_BYTES;
        let gate_pair = patterned_q3_expert_matrix(11);
        let up_pair = patterned_q3_expert_matrix(37);
        let down_pair = patterned_q3_expert_matrix(83);
        let gate = repeat_first_expert(
            &gate_pair[..expert_stride],
            expert_stride,
            expert_count,
            active_experts,
        );
        let up = repeat_first_expert(
            &up_pair[..expert_stride],
            expert_stride,
            expert_count,
            active_experts,
        );
        let down = repeat_first_expert(
            &down_pair[..expert_stride],
            expert_stride,
            expert_count,
            active_experts,
        );
        let output = backend
            .laguna_gguf_moe_device(
                &gate,
                &up,
                &down,
                GgufExpertQuant::Q3K,
                &input,
                &routing,
                BLOCK_VALUES,
                BLOCK_VALUES,
                BLOCK_VALUES,
            )
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();

        for token in 0..token_count {
            let input_row = &input_values[token * BLOCK_VALUES..(token + 1) * BLOCK_VALUES];
            let gate_values = cpu_q3_matrix_vector(&gate[..expert_stride], input_row);
            let up_values = cpu_q3_matrix_vector(&up[..expert_stride], input_row);
            let intermediate = gate_values
                .iter()
                .copied()
                .zip(up_values)
                .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
                .collect::<Vec<_>>();
            let expected = cpu_q3_matrix_vector(&down[..expert_stride], &intermediate);
            let actual = &output.values()[token * BLOCK_VALUES..(token + 1) * BLOCK_VALUES];
            for (row, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
                let tolerance = 2e-2_f32.max(expected.abs() * 2e-2);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "Q3 prefill GEMM mismatch at token {token}, row {row}: actual={actual}, expected={expected}, tolerance={tolerance}"
                );
            }
        }
    }

    fn patterned_q3_expert_matrix(seed: u8) -> Vec<u8> {
        let mut weights = Vec::with_capacity(2 * BLOCK_VALUES * Q3_BLOCK_BYTES);
        for expert in 0..2 {
            for row in 0..BLOCK_VALUES {
                let row_seed = seed
                    .wrapping_add((row as u8).wrapping_mul(17))
                    .wrapping_add((expert as u8).wrapping_mul(29));
                let scales = std::array::from_fn(|group| {
                    row_seed.wrapping_add((group as u8).wrapping_mul(7)) & 0x3f
                });
                let quant = std::array::from_fn(|index| {
                    row_seed
                        .wrapping_mul(13)
                        .wrapping_add((index as u8).wrapping_mul(19))
                });
                let high_mask = std::array::from_fn(|index| {
                    row_seed.rotate_left((index % 8) as u32) ^ (index as u8).wrapping_mul(11)
                });
                let d = if expert == 0 { 0x1000 } else { 0 };
                weights.extend(q3_patterned_block(d, scales, quant, high_mask));
            }
        }
        weights
    }

    fn patterned_q2_expert_matrix(seed: u8, rows: usize, in_features: usize) -> Vec<u8> {
        let blocks_per_row = in_features / BLOCK_VALUES;
        let mut weights = Vec::with_capacity(2 * rows * blocks_per_row * Q2_BLOCK_BYTES);
        for expert in 0..2 {
            for row in 0..rows {
                for block in 0..blocks_per_row {
                    let block_seed = seed
                        .wrapping_add((row as u8).wrapping_mul(17))
                        .wrapping_add((block as u8).wrapping_mul(7))
                        .wrapping_add((expert as u8).wrapping_mul(29));
                    weights.extend(q2_patterned_block(
                        if expert == 0 { 0x1000 } else { 0 },
                        if expert == 0 { 0x0c00 } else { 0 },
                        block_seed,
                    ));
                }
            }
        }
        weights
    }

    fn repeat_first_expert(
        active_payload: &[u8],
        expert_stride: usize,
        expert_count: usize,
        active_experts: usize,
    ) -> Vec<u8> {
        assert_eq!(active_payload.len(), expert_stride);
        let mut weights = Vec::with_capacity(expert_count * expert_stride);
        for expert in 0..expert_count {
            if expert < active_experts {
                weights.extend_from_slice(active_payload);
            } else {
                weights.resize(weights.len() + expert_stride, 0);
            }
        }
        weights
    }

    fn q2_patterned_block(d: u16, dmin: u16, seed: u8) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q2_BLOCK_BYTES);
        for index in 0..16_u8 {
            block.push(seed.wrapping_add(index.wrapping_mul(13)));
        }
        for index in 0..64_u8 {
            block.push(seed.wrapping_mul(29).wrapping_add(index.wrapping_mul(17)));
        }
        block.extend_from_slice(&d.to_le_bytes());
        block.extend_from_slice(&dmin.to_le_bytes());
        block
    }

    fn cpu_q2_matrix_vector(weights: &[u8], input: &[f32], rows: usize) -> Vec<f32> {
        let blocks_per_row = input.len() / BLOCK_VALUES;
        (0..rows)
            .map(|row| {
                (0..blocks_per_row)
                    .map(|block| {
                        let weight_offset = (row * blocks_per_row + block) * Q2_BLOCK_BYTES;
                        let input_offset = block * BLOCK_VALUES;
                        cpu_q2_block_dot(
                            &weights[weight_offset..weight_offset + Q2_BLOCK_BYTES],
                            &input[input_offset..input_offset + BLOCK_VALUES],
                        )
                    })
                    .sum()
            })
            .collect()
    }

    fn cpu_q2_block_dot(block: &[u8], input: &[f32]) -> f32 {
        let scales = &block[..16];
        let quants = &block[16..80];
        let d = crate::metal::buffers::f16_bits_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let min =
            crate::metal::buffers::f16_bits_to_f32(u16::from_le_bytes([block[82], block[83]]));
        let mut scale_index = 0;
        let mut quant_offset = 0;
        let mut input_offset = 0;
        let mut sum = 0.0;

        while input_offset < BLOCK_VALUES {
            let mut shift = 0;
            for _ in 0..4 {
                for quant_half in 0..2 {
                    let scale_min = scales[scale_index];
                    scale_index += 1;
                    let scale = d * (scale_min & 0x0f) as f32;
                    let min_offset = min * (scale_min >> 4) as f32;
                    let quant_base = quant_offset + quant_half * 16;
                    for value_index in 0..16 {
                        let weight = scale
                            * ((quants[quant_base + value_index] >> shift) & 0x03) as f32
                            - min_offset;
                        sum += input[input_offset + value_index] * weight;
                    }
                    input_offset += 16;
                }
                shift += 2;
            }
            quant_offset += 32;
        }

        sum
    }

    fn cpu_q3_matrix_vector(weights: &[u8], input: &[f32]) -> Vec<f32> {
        (0..BLOCK_VALUES)
            .map(|row| {
                let offset = row * Q3_BLOCK_BYTES;
                cpu_q3_block_dot(&weights[offset..offset + Q3_BLOCK_BYTES], input)
            })
            .collect()
    }

    fn cpu_q3_block_dot(block: &[u8], input: &[f32]) -> f32 {
        input
            .iter()
            .copied()
            .enumerate()
            .map(|(index, value)| value * cpu_q3_block_value(block, index))
            .sum()
    }

    fn cpu_q3_block_value(block: &[u8], value_index: usize) -> f32 {
        let group = value_index / 16;
        let index = value_index % 16;
        let quant_offset = 32 + 32 * (group / 8) + 16 * (group & 1);
        let high_offset = 16 * (group & 1);
        let high_bit = 1_u8 << (group / 2);
        let scales_offset = 96;

        let scale_low_mask = match group / 4 {
            0 => 0x03_u16,
            1 => 0x0c_u16,
            2 => 0x30_u16,
            _ => 0xc0_u16,
        };
        let scale_nibble_mask = if group < 8 { 0x0f_u16 } else { 0xf0_u16 };
        let scale_low = block[scales_offset + group % 8] as u16;
        let scale_high = block[scales_offset + 8 + group % 4] as u16;
        let packed_scale = if (group / 4) & 1 == 1 {
            (scale_low & scale_nibble_mask) | ((scale_high & scale_low_mask) << 2)
        } else {
            (scale_low & scale_nibble_mask) | ((scale_high & scale_low_mask) << 4)
        };
        let d =
            crate::metal::buffers::f16_bits_to_f32(u16::from_le_bytes([block[108], block[109]]));
        let group_scale = if group < 8 {
            d * (packed_scale as f32 - 32.0)
        } else {
            d * (packed_scale as f32 / 16.0 - 32.0)
        };
        let shift = 2 * ((group / 2) & 3);
        let quant = (block[quant_offset + index] >> shift) & 0x03;
        let high = block[high_offset + index] & high_bit != 0;
        group_scale * (quant as f32 - if high { 0.0 } else { 4.0 })
    }

    fn q3_block(d: u16, decoded_scales: [u8; 16], quant_byte: u8, high_mask_byte: u8) -> Vec<u8> {
        q3_patterned_block(d, decoded_scales, [quant_byte; 64], [high_mask_byte; 32])
    }

    fn q3_patterned_block(
        d: u16,
        decoded_scales: [u8; 16],
        quant: [u8; 64],
        high_mask: [u8; 32],
    ) -> Vec<u8> {
        let mut packed_scales = [0_u8; 12];
        for (group, scale) in decoded_scales.into_iter().enumerate() {
            if group < 8 {
                packed_scales[group] |= scale & 0x0f;
            } else {
                packed_scales[group - 8] |= (scale & 0x0f) << 4;
            }
            packed_scales[8 + group % 4] |= ((scale >> 4) & 0x03) << (2 * (group / 4));
        }
        let mut block = Vec::with_capacity(Q3_BLOCK_BYTES);
        block.extend(high_mask);
        block.extend(quant);
        block.extend_from_slice(&packed_scales);
        block.extend_from_slice(&d.to_le_bytes());
        block
    }
}
