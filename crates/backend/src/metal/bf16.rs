use ::metal::{Buffer, CommandBufferRef, ComputePipelineState, Device};
use common::{Error, Result};

use crate::DeviceBf16Matrix;

use super::{
    arena::MetalArena,
    buffers::{require_f32_capacity, u32_buffer, u8_buffer},
    command::encode_1d,
    library::MetalLibrary,
    pipeline::compute_pipeline,
};

const LINEAR_KERNEL: &str = "bf16_linear_f32_kernel";
const GATE_UP_SWIGLU_KERNEL: &str = "bf16_gate_up_swiglu_f32_kernel";
const LAGUNA_ATTENTION_PROJECTIONS_KERNEL: &str = "laguna_bf16_attention_projections_f32_kernel";
const EMBEDDING_KERNEL: &str = "bf16_embedding_f32_kernel";
const SIMD_LANES: usize = 32;

pub(crate) struct MetalBf16 {
    linear_pipeline: ComputePipelineState,
    gate_up_swiglu_pipeline: ComputePipelineState,
    laguna_attention_projections_pipeline: ComputePipelineState,
    embedding_pipeline: ComputePipelineState,
    arena: MetalArena,
}

impl MetalBf16 {
    pub(crate) fn new(device: &Device, library: &MetalLibrary, arena: MetalArena) -> Result<Self> {
        let linear_pipeline = compute_pipeline(device, library, LINEAR_KERNEL)?;
        let gate_up_swiglu_pipeline = compute_pipeline(device, library, GATE_UP_SWIGLU_KERNEL)?;
        let laguna_attention_projections_pipeline =
            compute_pipeline(device, library, LAGUNA_ATTENTION_PROJECTIONS_KERNEL)?;
        let embedding_pipeline = compute_pipeline(device, library, EMBEDDING_KERNEL)?;
        for (label, pipeline) in [
            ("linear", &linear_pipeline),
            ("gate/up SwiGLU", &gate_up_swiglu_pipeline),
            (
                "Laguna attention projections",
                &laguna_attention_projections_pipeline,
            ),
        ] {
            let simd_width = pipeline.thread_execution_width() as usize;
            if simd_width != SIMD_LANES {
                return Err(Error::backend(format!(
                    "BF16 {label} requires a {SIMD_LANES}-lane SIMD group, device reports {simd_width}"
                )));
            }
        }
        Ok(Self {
            linear_pipeline,
            gate_up_swiglu_pipeline,
            laguna_attention_projections_pipeline,
            embedding_pipeline,
            arena,
        })
    }

    pub(crate) fn prepare(
        &self,
        device: &Device,
        bytes: &[u8],
        rows: usize,
        columns: usize,
    ) -> Result<DeviceBf16Matrix> {
        if rows == 0 || columns == 0 {
            return Err(Error::backend("BF16 matrix dimensions must be positive"));
        }
        let expected_bytes = rows
            .checked_mul(columns)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| Error::backend("BF16 matrix byte count overflow"))?;
        if bytes.len() != expected_bytes {
            return Err(Error::backend(format!(
                "BF16 matrix byte length mismatch: expected {expected_bytes}, got {}",
                bytes.len()
            )));
        }
        Ok(DeviceBf16Matrix {
            rows,
            columns,
            storage_bytes: bytes.len(),
            buffer: u8_buffer(device, bytes)?,
        })
    }

    pub(crate) fn encode_linear(
        &self,
        command_buffer: &CommandBufferRef,
        matrix: &DeviceBf16Matrix,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Buffer> {
        if row_count == 0 {
            return Err(Error::backend("BF16 linear row_count must be positive"));
        }
        let expected_input_len = row_count
            .checked_mul(matrix.columns)
            .ok_or_else(|| Error::backend("BF16 linear input element count overflow"))?;
        if input_len != expected_input_len {
            return Err(Error::backend(format!(
                "BF16 linear input length mismatch: expected {expected_input_len}, got {input_len}"
            )));
        }
        require_f32_capacity(input, input_len, "BF16 linear input")?;
        let output_len = row_count
            .checked_mul(matrix.rows)
            .ok_or_else(|| Error::backend("BF16 linear output element count overflow"))?;
        let output = self.arena.empty_f32(output_len)?;
        let row_count = self.arena.u32(as_u32(row_count, "row_count")?)?;
        let in_features = self.arena.u32(as_u32(matrix.columns, "in_features")?)?;
        let out_features = self.arena.u32(as_u32(matrix.rows, "out_features")?)?;
        let thread_count = output_len
            .checked_mul(SIMD_LANES)
            .ok_or_else(|| Error::backend("BF16 linear thread count overflow"))?;
        encode_1d(
            command_buffer,
            &self.linear_pipeline,
            &[
                &matrix.buffer,
                input,
                &output,
                &row_count,
                &in_features,
                &out_features,
            ],
            thread_count,
        )?;
        Ok(output)
    }

    pub(crate) fn encode_gate_up_swiglu(
        &self,
        command_buffer: &CommandBufferRef,
        gate: &DeviceBf16Matrix,
        up: &DeviceBf16Matrix,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Buffer> {
        if gate.rows != up.rows || gate.columns != up.columns {
            return Err(Error::backend(format!(
                "BF16 gate/up matrices must have the same shape, got [{},{}] and [{},{}]",
                gate.rows, gate.columns, up.rows, up.columns
            )));
        }
        if row_count == 0 {
            return Err(Error::backend(
                "BF16 gate/up SwiGLU row_count must be positive",
            ));
        }
        let expected_input_len = row_count
            .checked_mul(gate.columns)
            .ok_or_else(|| Error::backend("BF16 gate/up SwiGLU input length overflow"))?;
        if input_len != expected_input_len {
            return Err(Error::backend(format!(
                "BF16 gate/up SwiGLU input length mismatch: expected {expected_input_len}, got {input_len}"
            )));
        }
        require_f32_capacity(input, input_len, "BF16 gate/up SwiGLU input")?;
        let output_len = output_len(row_count, gate.rows)?;
        let output = self.arena.empty_f32(output_len)?;
        let row_count = self.arena.u32(as_u32(row_count, "row_count")?)?;
        let in_features = self
            .arena
            .u32(as_u32(gate.columns, "gate/up in_features")?)?;
        let out_features = self.arena.u32(as_u32(gate.rows, "gate/up out_features")?)?;
        let thread_count = output_len
            .checked_mul(SIMD_LANES)
            .ok_or_else(|| Error::backend("BF16 gate/up SwiGLU thread count overflow"))?;
        encode_1d(
            command_buffer,
            &self.gate_up_swiglu_pipeline,
            &[
                &gate.buffer,
                &up.buffer,
                input,
                &output,
                &row_count,
                &in_features,
                &out_features,
            ],
            thread_count,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_laguna_attention_projections(
        &self,
        command_buffer: &CommandBufferRef,
        query: &DeviceBf16Matrix,
        key: &DeviceBf16Matrix,
        value: &DeviceBf16Matrix,
        gate: &DeviceBf16Matrix,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<(Buffer, Buffer, Buffer, Buffer)> {
        if row_count == 0 {
            return Err(Error::backend(
                "Laguna attention projection row_count must be positive",
            ));
        }
        let in_features = query.columns;
        for (label, matrix) in [("key", key), ("value", value), ("gate", gate)] {
            if matrix.columns != in_features {
                return Err(Error::backend(format!(
                    "Laguna attention {label} input width must be {in_features}, got {}",
                    matrix.columns
                )));
            }
        }
        let expected_input_len = row_count
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("Laguna attention projection input length overflow"))?;
        if input_len != expected_input_len {
            return Err(Error::backend(format!(
                "Laguna attention projection input length mismatch: expected {expected_input_len}, got {input_len}"
            )));
        }
        require_f32_capacity(input, input_len, "Laguna attention projection input")?;

        let query_output = self.arena.empty_f32(output_len(row_count, query.rows)?)?;
        let key_output = self.arena.empty_f32(output_len(row_count, key.rows)?)?;
        let value_output = self.arena.empty_f32(output_len(row_count, value.rows)?)?;
        let gate_output = self.arena.empty_f32(output_len(row_count, gate.rows)?)?;
        let combined_features = query
            .rows
            .checked_add(key.rows)
            .and_then(|count| count.checked_add(value.rows))
            .and_then(|count| count.checked_add(gate.rows))
            .ok_or_else(|| Error::backend("Laguna attention projection output width overflow"))?;
        let thread_count = row_count
            .checked_mul(combined_features)
            .and_then(|count| count.checked_mul(SIMD_LANES))
            .ok_or_else(|| Error::backend("Laguna attention projection thread count overflow"))?;
        let row_count = self.arena.u32(as_u32(row_count, "row_count")?)?;
        let in_features = self
            .arena
            .u32(as_u32(in_features, "attention in_features")?)?;
        let query_features = self
            .arena
            .u32(as_u32(query.rows, "query output features")?)?;
        let key_features = self.arena.u32(as_u32(key.rows, "key output features")?)?;
        let value_features = self
            .arena
            .u32(as_u32(value.rows, "value output features")?)?;
        let gate_features = self.arena.u32(as_u32(gate.rows, "gate output features")?)?;

        encode_1d(
            command_buffer,
            &self.laguna_attention_projections_pipeline,
            &[
                &query.buffer,
                &key.buffer,
                &value.buffer,
                &gate.buffer,
                input,
                &query_output,
                &key_output,
                &value_output,
                &gate_output,
                &row_count,
                &in_features,
                &query_features,
                &key_features,
                &value_features,
                &gate_features,
            ],
            thread_count,
        )?;
        Ok((query_output, key_output, value_output, gate_output))
    }

    pub(crate) fn encode_embedding(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        matrix: &DeviceBf16Matrix,
        token_ids: &[u32],
    ) -> Result<Buffer> {
        if token_ids.is_empty() {
            return Err(Error::backend("BF16 embedding token IDs must not be empty"));
        }
        if let Some(token_id) = token_ids
            .iter()
            .copied()
            .find(|token_id| *token_id as usize >= matrix.rows)
        {
            return Err(Error::backend(format!(
                "BF16 embedding token ID {token_id} exceeds vocabulary {}",
                matrix.rows
            )));
        }
        let output_len = token_ids
            .len()
            .checked_mul(matrix.columns)
            .ok_or_else(|| Error::backend("BF16 embedding output element count overflow"))?;
        let token_buffer = u32_buffer(device, token_ids)?;
        let output = self.arena.empty_f32(output_len)?;
        let token_count = self
            .arena
            .u32(as_u32(token_ids.len(), "embedding token_count")?)?;
        let vocab_size = self
            .arena
            .u32(as_u32(matrix.rows, "embedding vocab_size")?)?;
        let hidden_size = self
            .arena
            .u32(as_u32(matrix.columns, "embedding hidden_size")?)?;
        encode_1d(
            command_buffer,
            &self.embedding_pipeline,
            &[
                &matrix.buffer,
                &token_buffer,
                &output,
                &token_count,
                &vocab_size,
                &hidden_size,
            ],
            output_len,
        )?;
        Ok(output)
    }
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::backend(format!("BF16 {label} exceeds Metal u32 limit")))
}

fn output_len(row_count: usize, features: usize) -> Result<usize> {
    row_count
        .checked_mul(features)
        .ok_or_else(|| Error::backend("BF16 output element count overflow"))
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use common::{Device, F32Tensor};

    use crate::{Backend, MetalBackend};

    #[test]
    fn linear_matches_f32_reference() {
        let Some(backend) = native_backend_or_skip() else {
            return;
        };
        let rows = 2;
        let columns = 32;
        let weights = (0..rows * columns)
            .map(|index| (index as f32 - 17.0) / 16.0)
            .collect::<Vec<_>>();
        let matrix = backend
            .prepare_bf16_matrix(&bf16_bytes(&weights), rows, columns)
            .unwrap()
            .unwrap();
        let input_values = (0..columns)
            .map(|index| (index as f32 - 9.0) / 8.0)
            .collect::<Vec<_>>();
        let input = F32Tensor::new(input_values.clone(), [1, columns]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let output = backend
            .bf16_linear_device(&matrix, &input)
            .unwrap()
            .unwrap();
        let actual = backend.device_download_f32_tensor(&output).unwrap();
        let expected = weights
            .chunks_exact(columns)
            .map(|row| row.iter().zip(&input_values).map(|(a, b)| a * b).sum())
            .collect::<Vec<f32>>();
        assert_close(actual.values(), &expected, 0.03);
    }

    #[test]
    fn fused_gate_up_swiglu_matches_cpu_reference() {
        let Some(backend) = native_backend_or_skip() else {
            return;
        };
        let rows = 5;
        let columns = 32;
        let row_count = 2;
        let gate_weights = patterned_weights(rows, columns, 1);
        let up_weights = patterned_weights(rows, columns, 3);
        let gate = prepared(&backend, &gate_weights, rows, columns);
        let up = prepared(&backend, &up_weights, rows, columns);
        let input_values = (0..row_count * columns)
            .map(|index| ((index % 11) as f32 - 5.0) * 0.125)
            .collect::<Vec<_>>();
        let input = F32Tensor::new(input_values.clone(), [1, row_count, columns]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let output = backend
            .bf16_gate_up_swiglu_device(&gate, &up, &input)
            .unwrap()
            .unwrap();
        let actual = backend.device_download_f32_tensor(&output).unwrap();
        let gate_values = cpu_linear(&input_values, &gate_weights, row_count, rows, columns);
        let up_values = cpu_linear(&input_values, &up_weights, row_count, rows, columns);
        let expected = gate_values
            .iter()
            .zip(up_values)
            .map(|(gate, up)| (gate / (1.0 + (-gate).exp())) * up)
            .collect::<Vec<_>>();
        assert_eq!(actual.dims(), &[1, row_count, rows]);
        assert_close(actual.values(), &expected, 0.001);
    }

    #[test]
    fn embedding_gathers_requested_rows() {
        let Some(backend) = native_backend_or_skip() else {
            return;
        };
        let rows = 4;
        let columns = 8;
        let values = (0..rows * columns)
            .map(|value| value as f32 * 0.25)
            .collect::<Vec<_>>();
        let matrix = backend
            .prepare_bf16_matrix(&bf16_bytes(&values), rows, columns)
            .unwrap()
            .unwrap();
        let output = backend
            .bf16_embedding_device(&matrix, &[3, 1], &[1, 2])
            .unwrap()
            .unwrap();
        let actual = backend.device_download_f32_tensor(&output).unwrap();
        let expected = values[3 * columns..4 * columns]
            .iter()
            .chain(&values[columns..2 * columns])
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(actual.dims(), &[1, 2, columns]);
        assert_close(actual.values(), &expected, 0.001);
    }

    #[test]
    fn laguna_attention_projections_match_independent_cpu_linears() {
        let Some(backend) = native_backend_or_skip() else {
            return;
        };
        let columns = 32;
        let row_count = 2;
        let query_rows = 128;
        let key_value_rows = 8 * 128;
        let gate_rows = 1;
        let query_weights = patterned_weights(query_rows, columns, 1);
        let key_weights = patterned_weights(key_value_rows, columns, 2);
        let value_weights = patterned_weights(key_value_rows, columns, 3);
        let gate_weights = patterned_weights(gate_rows, columns, 4);
        let query = prepared(&backend, &query_weights, query_rows, columns);
        let key = prepared(&backend, &key_weights, key_value_rows, columns);
        let value = prepared(&backend, &value_weights, key_value_rows, columns);
        let gate = prepared(&backend, &gate_weights, gate_rows, columns);
        let input_values = (0..row_count * columns)
            .map(|index| ((index % 9) as f32 - 4.0) * 0.125)
            .collect::<Vec<_>>();
        let input = F32Tensor::new(input_values.clone(), [1, row_count, columns]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();

        let projections = backend
            .laguna_attention_projections_device(&query, &key, &value, &gate, &input)
            .unwrap()
            .unwrap();
        assert_eq!(projections.query.dims(), &[1, row_count, 1, 128]);
        assert_eq!(projections.key.dims(), &[1, row_count, 8, 128]);
        assert_eq!(projections.value.dims(), &[1, row_count, 8, 128]);
        assert_eq!(projections.gate.dims(), &[1, row_count, 1]);

        for (actual, weights, rows) in [
            (projections.query, &query_weights, query_rows),
            (projections.key, &key_weights, key_value_rows),
            (projections.value, &value_weights, key_value_rows),
            (projections.gate, &gate_weights, gate_rows),
        ] {
            let actual = backend.device_download_f32_tensor(&actual).unwrap();
            let expected = cpu_linear(&input_values, weights, row_count, rows, columns);
            assert_close(actual.values(), &expected, 0.001);
        }
    }

    fn native_backend_or_skip() -> Option<MetalBackend> {
        let backend = MetalBackend::from_device(Device::Metal).ok()?;
        backend.device_values_supported().then_some(backend)
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn patterned_weights(rows: usize, columns: usize, phase: usize) -> Vec<f32> {
        (0..rows * columns)
            .map(|index| (((index + phase) % 7) as f32 - 3.0) * 0.125)
            .collect()
    }

    fn prepared(
        backend: &MetalBackend,
        weights: &[f32],
        rows: usize,
        columns: usize,
    ) -> crate::DeviceBf16Matrix {
        backend
            .prepare_bf16_matrix(&bf16_bytes(weights), rows, columns)
            .unwrap()
            .unwrap()
    }

    fn cpu_linear(
        input: &[f32],
        weights: &[f32],
        row_count: usize,
        out_features: usize,
        in_features: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; row_count * out_features];
        for row in 0..row_count {
            for out_feature in 0..out_features {
                output[row * out_features + out_feature] = (0..in_features)
                    .map(|feature| {
                        input[row * in_features + feature]
                            * weights[out_feature * in_features + feature]
                    })
                    .sum();
            }
        }
        output
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
