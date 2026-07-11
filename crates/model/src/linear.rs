use std::cmp::Ordering;

use backend::{Backend, BackendCapabilities};
use common::Tensor;
use common::{validate_exact_shape, DeviceKind, Error, F32Tensor, Result, Shape};
use gguf::{GgmlType, GgufFile, GgufQuantBlockKind};
use tracing::debug;

use crate::TensorRef;

#[derive(Debug)]
pub struct QuantizedLinear<'a> {
    gguf: &'a GgufFile,
    tensor_ref: TensorRef,
    in_features: usize,
    out_features: usize,
    blocks_per_row: u64,
    output_chunk_rows: usize,
    full_source_payload_bytes: u64,
    q2_payload_bytes: Option<&'a [u8]>,
}

#[derive(Debug)]
pub struct LinearOutput {
    pub output: Tensor,
    pub report: LinearForwardReport,
}

#[derive(Debug)]
pub struct LinearGreedyOutput {
    pub token_id: u32,
    pub token_score: f32,
    pub report: LinearGreedyReport,
}

#[derive(Debug)]
pub struct LinearTokenOutput {
    pub token_id: u32,
    pub token_score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearForwardReport {
    pub source_tensor_name: String,
    pub tensor_type: GgmlType,
    pub backend: BackendCapabilities,
    pub input_shape: Shape,
    pub logical_weight_shape: Shape,
    pub output_shape: Shape,
    pub output_chunk_rows: usize,
    pub chunk_count: usize,
    pub source_payload_bytes_read: u64,
    pub full_source_payload_bytes: u64,
    pub peak_decoded_f32_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearGreedyReport {
    pub source_tensor_name: String,
    pub tensor_type: GgmlType,
    pub backend: BackendCapabilities,
    pub input_shape: Shape,
    pub logical_weight_shape: Shape,
    pub logits_shape: Shape,
    pub output_chunk_rows: usize,
    pub chunk_count: usize,
    pub source_payload_bytes_read: u64,
    pub full_source_payload_bytes: u64,
    pub peak_decoded_f32_bytes: u64,
    pub materialized_full_logits: bool,
}

impl<'a> QuantizedLinear<'a> {
    pub fn open(
        gguf: &'a GgufFile,
        tensor_ref: &TensorRef,
        expected_in_features: usize,
        min_out_features: usize,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        if output_chunk_rows == 0 {
            return Err(Error::gguf(
                "GGUF quantized linear output_chunk_rows must be positive",
            ));
        }
        let info = gguf.tensor(&tensor_ref.name).ok_or_else(|| {
            Error::gguf(format!("missing GLM-5.2 GGUF tensor {}", tensor_ref.name))
        })?;
        validate_tensor_ref(tensor_ref, info)?;
        validate_quantized_type(tensor_ref)?;
        validate_exact_shape("gguf_quantized_linear_rank", &[tensor_ref.dims.len()], &[2])?;

        let in_features = usize::try_from(tensor_ref.dims[0]).map_err(|_| {
            Error::gguf(format!(
                "GGUF quantized linear input dimension {} does not fit usize",
                tensor_ref.dims[0]
            ))
        })?;
        let out_features = usize::try_from(tensor_ref.dims[1]).map_err(|_| {
            Error::gguf(format!(
                "GGUF quantized linear output dimension {} does not fit usize",
                tensor_ref.dims[1]
            ))
        })?;
        validate_exact_shape(
            "gguf_quantized_linear_input_features",
            &[in_features],
            &[expected_in_features],
        )?;
        if out_features < min_out_features {
            return Err(Error::gguf(format!(
                "GGUF quantized linear output rows {out_features} are smaller than required {min_out_features}"
            )));
        }

        let storage = gguf.tensor_quantized_storage(&tensor_ref.name)?;
        let block_size = usize::try_from(storage.block.values_per_block())
            .map_err(|_| Error::gguf("GGUF quantized linear block size does not fit usize"))?;
        if in_features % block_size != 0 {
            return Err(Error::gguf(format!(
                "GGUF quantized linear input dimension {in_features} must be divisible by {block_size}"
            )));
        }
        let blocks_per_row = u64::try_from(in_features / block_size)
            .map_err(|_| Error::gguf("GGUF quantized linear blocks_per_row does not fit u64"))?;
        let expected_blocks = blocks_per_row
            .checked_mul(u64::try_from(out_features).map_err(|_| {
                Error::gguf("GGUF quantized linear output row count does not fit u64")
            })?)
            .ok_or_else(|| Error::gguf("GGUF quantized linear block count overflow"))?;
        if storage.block_count != expected_blocks {
            return Err(Error::gguf(format!(
                "GGUF quantized linear tensor {} has {} quant blocks but shape requires {expected_blocks}",
                tensor_ref.name, storage.block_count
            )));
        }
        let q2_payload_bytes = if storage.block == GgufQuantBlockKind::Q2K {
            Some(storage.bytes)
        } else {
            None
        };

        Ok(Self {
            gguf,
            tensor_ref: tensor_ref.clone(),
            in_features,
            out_features,
            blocks_per_row,
            output_chunk_rows,
            full_source_payload_bytes: storage.payload_byte_len,
            q2_payload_bytes,
        })
    }

    pub fn forward<B: Backend>(&self, input: &Tensor, backend: &B) -> Result<LinearOutput> {
        let run = self.run_chunked_linear(input, backend, true)?;
        let report = LinearForwardReport {
            source_tensor_name: self.tensor_ref.name.clone(),
            tensor_type: self.tensor_ref.ty,
            backend: backend.capabilities(),
            input_shape: Shape::new(input.dims().to_vec()),
            logical_weight_shape: Shape::new(vec![self.out_features, self.in_features]),
            output_shape: Shape::new(run.output.dims().to_vec()),
            output_chunk_rows: self.output_chunk_rows,
            chunk_count: run.chunk_count,
            source_payload_bytes_read: self.full_source_payload_bytes,
            full_source_payload_bytes: self.full_source_payload_bytes,
            peak_decoded_f32_bytes: run.peak_decoded_f32_bytes,
        };

        debug!(
            tensor = %self.tensor_ref.name,
            output_chunk_rows = self.output_chunk_rows,
            chunk_count = report.chunk_count,
            peak_decoded_f32_bytes = run.peak_decoded_f32_bytes,
            full_source_payload_bytes = self.full_source_payload_bytes,
            "ran chunked GLM-5.2 GGUF quantized linear"
        );

        Ok(LinearOutput {
            output: run.output,
            report,
        })
    }

    pub fn forward_tensor<B: Backend>(&self, input: &Tensor, backend: &B) -> Result<Tensor> {
        let input = tensor_to_f32_tensor(input)?;
        let output = self.forward_f32_tensor(&input, backend)?;
        tensor_from_f32_output(output, backend.device())
    }

    pub fn forward_f32_tensor<B: Backend>(
        &self,
        input: &F32Tensor,
        backend: &B,
    ) -> Result<F32Tensor> {
        Ok(self.run_direct_quantized_linear_f32(input, backend)?.output)
    }

    /// Batched device-resident variant of `forward_f32_tensor`: encodes the
    /// quantized matvec into the backend's open batch. Returns `Ok(None)` when
    /// the backend has no device path or this weight's quantization has no
    /// batched kernel.
    pub(crate) fn forward_device<B: Backend>(
        &self,
        input: &backend::DeviceValue,
        backend: &B,
    ) -> Result<Option<backend::DeviceValue>> {
        let (row_count, expected_output_shape) = validate_input_dims(
            "GGUF quantized linear device",
            input.dims(),
            self.in_features,
            self.out_features,
        )?;

        let output = match self.tensor_ref.ty {
            GgmlType::Q2K => {
                let raw_data = self.q2_payload_bytes.ok_or_else(|| {
                    Error::gguf(format!(
                        "GGUF tensor {} must be Q2_K for the device Q2 matvec path, got {}",
                        self.tensor_ref.name, self.tensor_ref.ty
                    ))
                })?;
                crate::try_device!(backend.q2_k_matvec_device(
                    raw_data,
                    input,
                    row_count,
                    self.in_features,
                    self.out_features,
                ))
            }
            GgmlType::Q8_0 => {
                let storage = self.gguf.tensor_quantized_storage(&self.tensor_ref.name)?;
                crate::try_device!(backend.q8_0_matvec_device(
                    storage.bytes,
                    input,
                    row_count,
                    self.in_features,
                    self.out_features,
                ))
            }
            _ => return Ok(None),
        };
        validate_exact_shape(
            "gguf_device_quantized_linear_output",
            output.dims(),
            &expected_output_shape,
        )?;
        Ok(Some(output))
    }

    /// Batched device-resident variant of `forward_f32_tensor_add_residual`.
    /// Q2_K uses the fused matvec+add kernel; Q8_0 uses native matvec followed
    /// by a native add so the path remains GPU-resident.
    pub(crate) fn forward_device_add_residual<B: Backend>(
        &self,
        input: &backend::DeviceValue,
        residual: &backend::DeviceValue,
        backend: &B,
    ) -> Result<Option<backend::DeviceValue>> {
        let (row_count, expected_output_shape) = validate_input_dims(
            "GGUF quantized linear device residual",
            input.dims(),
            self.in_features,
            self.out_features,
        )?;
        validate_exact_shape(
            "gguf_q2_quantized_linear_device_residual",
            residual.dims(),
            &expected_output_shape,
        )?;
        let output = match self.tensor_ref.ty {
            GgmlType::Q2K => {
                let raw_data = self.q2_payload_bytes.ok_or_else(|| {
                    Error::gguf(format!(
                        "GGUF tensor {} must be Q2_K for the device Q2 matvec add path, got {}",
                        self.tensor_ref.name, self.tensor_ref.ty
                    ))
                })?;
                crate::try_device!(backend.q2_k_matvec_add_device(
                    raw_data,
                    input,
                    residual,
                    row_count,
                    self.in_features,
                    self.out_features,
                ))
            }
            GgmlType::Q8_0 => {
                let projected = crate::try_device!(self.forward_device(input, backend));
                crate::try_device!(backend.add_device(residual, &projected))
            }
            _ => return Ok(None),
        };
        validate_exact_shape(
            "gguf_device_q2_quantized_linear_add_output",
            output.dims(),
            &expected_output_shape,
        )?;
        Ok(Some(output))
    }

    /// Batched device-resident greedy decode: encodes the output-head matvec
    /// plus argmax, then flushes the whole batch. `Ok(None)` when unsupported.
    pub(crate) fn greedy_token_device<B: Backend>(
        &self,
        input: &backend::DeviceValue,
        backend: &B,
    ) -> Result<Option<LinearTokenOutput>> {
        let (token_id, token_score) = match self.tensor_ref.ty {
            GgmlType::Q2K => {
                let raw_data = self.q2_payload_bytes.ok_or_else(|| {
                    Error::gguf(format!(
                        "GGUF tensor {} must be Q2_K for the device Q2 greedy path, got {}",
                        self.tensor_ref.name, self.tensor_ref.ty
                    ))
                })?;
                crate::try_device!(backend.q2_k_matvec_argmax_device(
                    raw_data,
                    input,
                    self.in_features,
                    self.out_features,
                ))
            }
            GgmlType::Q8_0 => {
                let logits = crate::try_device!(self.forward_device(input, backend));
                let logits = logits.reshape(vec![self.out_features])?;
                crate::try_device!(backend.argmax_f32_device(&logits))
            }
            _ => return Ok(None),
        };
        Ok(Some(LinearTokenOutput {
            token_id,
            token_score,
        }))
    }

    pub fn forward_f32_tensor_add_residual<B: Backend>(
        &self,
        input: &F32Tensor,
        residual: &F32Tensor,
        backend: &B,
    ) -> Result<F32Tensor> {
        let (row_count, expected_output_shape) = validate_input_shape_f32(
            "GGUF Q2 quantized linear residual",
            input,
            self.in_features,
            self.out_features,
        )?;
        validate_exact_shape(
            "gguf_q2_quantized_linear_residual",
            residual.dims(),
            &expected_output_shape,
        )?;

        if self.tensor_ref.ty == GgmlType::Q2K {
            let raw_data = self.q2_payload_bytes.ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} must be Q2_K for the native Q2 matvec add path, got {}",
                    self.tensor_ref.name, self.tensor_ref.ty
                ))
            })?;
            if let Some(output) = backend.q2_k_matvec_add_f32_tensor(
                raw_data,
                input,
                residual,
                row_count,
                self.in_features,
                self.out_features,
            )? {
                validate_exact_shape(
                    "gguf_native_q2_quantized_linear_add_output",
                    output.dims(),
                    &expected_output_shape,
                )?;
                return Ok(output);
            }
            reject_missing_native_q2_kernel_on_metal(backend, &self.tensor_ref.name)?;
        }

        let projected = self.forward_f32_tensor(input, backend)?;
        add_f32_tensors("gguf_quantized_linear_residual_add", &projected, residual)
    }

    pub fn greedy_argmax<B: Backend>(
        &self,
        input: &Tensor,
        backend: &B,
    ) -> Result<LinearGreedyOutput> {
        let scan = self.scan_greedy_argmax(input, backend, true)?;

        Ok(LinearGreedyOutput {
            token_id: scan.token_id,
            token_score: scan.token_score,
            report: LinearGreedyReport {
                source_tensor_name: self.tensor_ref.name.clone(),
                tensor_type: self.tensor_ref.ty,
                backend: backend.capabilities(),
                input_shape: Shape::new(input.dims().to_vec()),
                logical_weight_shape: Shape::new(vec![self.out_features, self.in_features]),
                logits_shape: Shape::new(vec![scan.batch, self.out_features]),
                output_chunk_rows: self.output_chunk_rows,
                chunk_count: scan.chunk_count,
                source_payload_bytes_read: scan.source_payload_bytes_read,
                full_source_payload_bytes: self.full_source_payload_bytes,
                peak_decoded_f32_bytes: scan.peak_decoded_f32_bytes,
                materialized_full_logits: false,
            },
        })
    }

    pub fn greedy_token<B: Backend>(
        &self,
        input: &Tensor,
        backend: &B,
    ) -> Result<LinearTokenOutput> {
        let scan = self.scan_greedy_argmax(input, backend, false)?;

        Ok(LinearTokenOutput {
            token_id: scan.token_id,
            token_score: scan.token_score,
        })
    }

    pub fn greedy_token_f32<B: Backend>(
        &self,
        input: &F32Tensor,
        backend: &B,
    ) -> Result<LinearTokenOutput> {
        let scan = if self.tensor_ref.ty == GgmlType::Q2K {
            self.scan_q2_qmatmul_greedy_argmax_f32(input, backend)?
        } else {
            let tensor = tensor_from_f32_output(input.clone(), backend.device())?;
            self.scan_direct_greedy_argmax(&tensor)?
        };

        Ok(LinearTokenOutput {
            token_id: scan.token_id,
            token_score: scan.token_score,
        })
    }

    pub fn greedy_token_after_rms_norm_f32<B: Backend>(
        &self,
        input: &F32Tensor,
        rms_weight: &F32Tensor,
        rms_eps: f32,
        backend: &B,
    ) -> Result<Option<LinearTokenOutput>> {
        if self.tensor_ref.ty != GgmlType::Q2K {
            return Ok(None);
        }

        let batch = validate_greedy_input_shape_f32(input, self.in_features)?;
        let raw_data = self.q2_payload_bytes.ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} must be Q2_K for the native fused RMSNorm/Q2 greedy path, got {}",
                self.tensor_ref.name, self.tensor_ref.ty
            ))
        })?;
        let Some((token_id, token_score)) = backend.q2_k_rms_norm_argmax_f32_tensor(
            raw_data,
            input,
            rms_weight,
            rms_eps,
            batch,
            self.in_features,
            self.out_features,
        )?
        else {
            return Ok(None);
        };

        Ok(Some(LinearTokenOutput {
            token_id,
            token_score,
        }))
    }

    fn scan_greedy_argmax<B: Backend>(
        &self,
        input: &Tensor,
        backend: &B,
        collect_report_stats: bool,
    ) -> Result<LinearGreedyScan> {
        if !collect_report_stats {
            if self.tensor_ref.ty == GgmlType::Q2K {
                return self.scan_q2_qmatmul_greedy_argmax(input, backend);
            }
            return self.scan_direct_greedy_argmax(input);
        }

        let batch = validate_greedy_input_shape(input, self.in_features)?;
        validate_exact_shape("gguf_quantized_linear_greedy_batch", &[batch], &[1])?;
        let storage = self.gguf.tensor_quantized_storage(&self.tensor_ref.name)?;
        let mut rows_loaded = 0_usize;
        let mut chunk_count = 0_usize;
        let mut peak_decoded_f32_bytes = 0_u64;
        let mut source_payload_bytes_read = 0_u64;
        let mut best_token_id = 0_u32;
        let mut best_score = 0.0_f32;
        let mut initialized = false;

        while rows_loaded < self.out_features {
            let rows_this_chunk = self
                .output_chunk_rows
                .min(self.out_features.saturating_sub(rows_loaded));
            let block_start = u64::try_from(rows_loaded)
                .ok()
                .and_then(|row| row.checked_mul(self.blocks_per_row))
                .ok_or_else(|| Error::gguf("GGUF quantized linear chunk block offset overflow"))?;
            let block_count = u64::try_from(rows_this_chunk)
                .ok()
                .and_then(|rows| rows.checked_mul(self.blocks_per_row))
                .ok_or_else(|| Error::gguf("GGUF quantized linear chunk block count overflow"))?;
            if collect_report_stats {
                source_payload_bytes_read = source_payload_bytes_read
                    .checked_add(
                        block_count
                            .checked_mul(storage.block.block_byte_len())
                            .ok_or_else(|| {
                                Error::gguf("GGUF quantized linear source byte count overflow")
                            })?,
                    )
                    .ok_or_else(|| {
                        Error::gguf("GGUF quantized linear source byte count overflow")
                    })?;
            }
            let chunk_values = storage.dequantize_block_range_as_f32(block_start, block_count)?;
            if collect_report_stats {
                let decoded_bytes = u64::try_from(chunk_values.len())
                    .ok()
                    .and_then(|values| values.checked_mul(4))
                    .ok_or_else(|| Error::gguf("GGUF quantized linear decoded byte overflow"))?;
                peak_decoded_f32_bytes = peak_decoded_f32_bytes.max(decoded_bytes);
            }
            let chunk_weight = Tensor::from_vec(
                chunk_values,
                (rows_this_chunk, self.in_features),
                backend.device(),
            )?;
            let raw_chunk_logits = backend.linear(input, &chunk_weight)?;
            let chunk_logits = match raw_chunk_logits.dims() {
                [chunk_batch, chunk_rows] => {
                    validate_exact_shape(
                        "gguf_quantized_linear_greedy_chunk_logits",
                        &[*chunk_batch, *chunk_rows],
                        &[batch, rows_this_chunk],
                    )?;
                    raw_chunk_logits
                }
                [chunk_batch, 1, chunk_rows] => {
                    validate_exact_shape(
                        "gguf_quantized_linear_greedy_chunk_logits",
                        &[*chunk_batch, *chunk_rows],
                        &[batch, rows_this_chunk],
                    )?;
                    raw_chunk_logits.reshape((batch, rows_this_chunk))?
                }
                dims => {
                    return Err(Error::model(format!(
                        "GGUF quantized linear greedy chunk logits must be rank 2 [B,C] or rank 3 [B,1,C], got {dims:?}"
                    )));
                }
            };
            let rows = chunk_logits.to_vec2::<f32>()?;
            let row = rows.first().ok_or_else(|| {
                Error::model("GGUF quantized linear greedy produced no logits row")
            })?;
            for (local_token_id, score) in row.iter().copied().enumerate() {
                let ordering = score.partial_cmp(&best_score).unwrap_or(Ordering::Less);
                if !initialized || ordering == Ordering::Greater {
                    let token_id = rows_loaded
                        .checked_add(local_token_id)
                        .ok_or_else(|| Error::model("GGUF greedy token id overflow"))?;
                    best_token_id = u32::try_from(token_id).map_err(|_| {
                        Error::model(format!("GGUF greedy token id {token_id} does not fit u32"))
                    })?;
                    best_score = score;
                    initialized = true;
                }
            }

            chunk_count = chunk_count.saturating_add(1);
            rows_loaded += rows_this_chunk;
        }

        Ok(LinearGreedyScan {
            token_id: best_token_id,
            token_score: best_score,
            batch,
            chunk_count,
            source_payload_bytes_read,
            peak_decoded_f32_bytes,
        })
    }

    fn scan_direct_greedy_argmax(&self, input: &Tensor) -> Result<LinearGreedyScan> {
        let batch = validate_greedy_input_shape(input, self.in_features)?;
        validate_exact_shape("gguf_quantized_linear_greedy_batch", &[batch], &[1])?;
        let input_values = match input.dims() {
            [_, _features] => input
                .to_vec2::<f32>()?
                .into_iter()
                .next()
                .ok_or_else(|| Error::model("GGUF greedy input has no batch row"))?,
            [_, _, _features] => input
                .to_vec3::<f32>()?
                .into_iter()
                .next()
                .and_then(|batch_rows| batch_rows.into_iter().next())
                .ok_or_else(|| Error::model("GGUF greedy input has no token row"))?,
            dims => {
                return Err(Error::model(format!(
                    "GGUF quantized linear greedy input rank must be 2 [B,H] or 3 [B,1,H], got {dims:?}"
                )));
            }
        };
        let storage = self.gguf.tensor_quantized_storage(&self.tensor_ref.name)?;
        let scores = match storage.block {
            GgufQuantBlockKind::Q2K => storage.matmul_rows_q2_k_f32(
                &input_values,
                1,
                self.in_features,
                self.out_features,
            )?,
            GgufQuantBlockKind::Q8_0 => {
                storage.matmul_rows_f32(&input_values, 1, self.in_features, self.out_features)?
            }
        };
        let (token_id, token_score) = greedy_argmax_from_scores(&scores)?;

        Ok(LinearGreedyScan {
            token_id,
            token_score,
            batch,
            chunk_count: 1,
            source_payload_bytes_read: 0,
            peak_decoded_f32_bytes: 0,
        })
    }

    fn run_chunked_linear<B: Backend>(
        &self,
        input: &Tensor,
        backend: &B,
        collect_report_stats: bool,
    ) -> Result<LinearRun> {
        if !collect_report_stats {
            return self.run_direct_quantized_linear(input, backend);
        }

        validate_input_shape(input, self.in_features)?;
        let cat_dim = match input.dims().len() {
            2 => 1,
            3 => 2,
            rank => {
                return Err(Error::model(format!(
                    "GGUF quantized linear input rank must be 2 or 3, got {rank}"
                )));
            }
        };

        let storage = self.gguf.tensor_quantized_storage(&self.tensor_ref.name)?;
        let mut outputs = Vec::new();
        let mut rows_loaded = 0_usize;
        let mut peak_decoded_f32_bytes = 0_u64;

        while rows_loaded < self.out_features {
            let rows_this_chunk = self
                .output_chunk_rows
                .min(self.out_features.saturating_sub(rows_loaded));
            let block_start = u64::try_from(rows_loaded)
                .ok()
                .and_then(|row| row.checked_mul(self.blocks_per_row))
                .ok_or_else(|| Error::gguf("GGUF quantized linear chunk block offset overflow"))?;
            let block_count = u64::try_from(rows_this_chunk)
                .ok()
                .and_then(|rows| rows.checked_mul(self.blocks_per_row))
                .ok_or_else(|| Error::gguf("GGUF quantized linear chunk block count overflow"))?;
            let chunk_values = storage.dequantize_block_range_as_f32(block_start, block_count)?;
            if collect_report_stats {
                let decoded_bytes = u64::try_from(chunk_values.len())
                    .ok()
                    .and_then(|values| values.checked_mul(4))
                    .ok_or_else(|| Error::gguf("GGUF quantized linear decoded byte overflow"))?;
                peak_decoded_f32_bytes = peak_decoded_f32_bytes.max(decoded_bytes);
            }
            let chunk_weight = Tensor::from_vec(
                chunk_values,
                (rows_this_chunk, self.in_features),
                backend.device(),
            )?;
            outputs.push(backend.linear(input, &chunk_weight)?);
            rows_loaded += rows_this_chunk;
        }

        let chunk_count = outputs.len();
        let output_refs = outputs.iter().collect::<Vec<_>>();
        let output = Tensor::cat(&output_refs, cat_dim)?;

        Ok(LinearRun {
            output,
            chunk_count,
            peak_decoded_f32_bytes,
        })
    }

    fn run_direct_quantized_linear<B: Backend>(
        &self,
        input: &Tensor,
        backend: &B,
    ) -> Result<LinearRun> {
        let input = tensor_to_f32_tensor(input)?;
        let run = self.run_direct_quantized_linear_f32(&input, backend)?;
        let output = tensor_from_f32_output(run.output, backend.device())?;

        Ok(LinearRun {
            output,
            chunk_count: run.chunk_count,
            peak_decoded_f32_bytes: run.peak_decoded_f32_bytes,
        })
    }

    fn run_direct_quantized_linear_f32<B: Backend>(
        &self,
        input: &F32Tensor,
        backend: &B,
    ) -> Result<LinearF32Run> {
        if self.tensor_ref.ty == GgmlType::Q2K {
            return self.run_q2_qmatmul_linear_f32(input, backend);
        }

        let (input_rows, output_shape) = validate_input_shape_f32(
            "GGUF quantized linear",
            input,
            self.in_features,
            self.out_features,
        )?;
        let storage = self.gguf.tensor_quantized_storage(&self.tensor_ref.name)?;
        let output_values = match storage.block {
            GgufQuantBlockKind::Q2K => storage.matmul_rows_q2_k_f32(
                input.values(),
                input_rows,
                self.in_features,
                self.out_features,
            )?,
            GgufQuantBlockKind::Q8_0 => storage.matmul_rows_f32(
                input.values(),
                input_rows,
                self.in_features,
                self.out_features,
            )?,
        };

        Ok(LinearF32Run {
            output: F32Tensor::new(output_values, output_shape)?,
            chunk_count: 1,
            peak_decoded_f32_bytes: 0,
        })
    }

    fn run_q2_qmatmul_linear_f32<B: Backend>(
        &self,
        input: &F32Tensor,
        backend: &B,
    ) -> Result<LinearF32Run> {
        let (row_count, expected_output_shape) = validate_input_shape_f32(
            "GGUF Q2 quantized linear",
            input,
            self.in_features,
            self.out_features,
        )?;
        let raw_data = self.q2_payload_bytes.ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} must be Q2_K for the native Q2 matvec path, got {}",
                self.tensor_ref.name, self.tensor_ref.ty
            ))
        })?;

        if let Some(output) = backend.q2_k_matvec_f32_tensor(
            raw_data,
            input,
            row_count,
            self.in_features,
            self.out_features,
        )? {
            validate_exact_shape(
                "gguf_native_q2_quantized_linear_output",
                output.dims(),
                &expected_output_shape,
            )?;
            return Ok(LinearF32Run {
                output,
                chunk_count: 1,
                peak_decoded_f32_bytes: 0,
            });
        }
        reject_missing_native_q2_kernel_on_metal(backend, &self.tensor_ref.name)?;

        let storage = self.gguf.tensor_quantized_storage(&self.tensor_ref.name)?;
        let output_values = storage.matmul_rows_q2_k_f32(
            input.values(),
            row_count,
            self.in_features,
            self.out_features,
        )?;
        let output = F32Tensor::new(output_values, expected_output_shape.clone())?;
        validate_exact_shape(
            "gguf_q2_quantized_linear_output",
            output.dims(),
            &expected_output_shape,
        )?;

        Ok(LinearF32Run {
            output,
            chunk_count: 1,
            peak_decoded_f32_bytes: 0,
        })
    }

    fn scan_q2_qmatmul_greedy_argmax<B: Backend>(
        &self,
        input: &Tensor,
        backend: &B,
    ) -> Result<LinearGreedyScan> {
        let input = tensor_to_f32_tensor(input)?;
        self.scan_q2_qmatmul_greedy_argmax_f32(&input, backend)
    }

    fn scan_q2_qmatmul_greedy_argmax_f32<B: Backend>(
        &self,
        input: &F32Tensor,
        backend: &B,
    ) -> Result<LinearGreedyScan> {
        let batch = validate_greedy_input_shape_f32(input, self.in_features)?;
        validate_exact_shape("gguf_quantized_linear_greedy_batch", &[batch], &[1])?;
        let raw_data = self.q2_payload_bytes.ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} must be Q2_K for the native Q2 greedy path, got {}",
                self.tensor_ref.name, self.tensor_ref.ty
            ))
        })?;
        if let Some((token_id, token_score)) = backend.q2_k_matvec_argmax_f32_tensor(
            raw_data,
            input,
            batch,
            self.in_features,
            self.out_features,
        )? {
            return Ok(LinearGreedyScan {
                token_id,
                token_score,
                batch,
                chunk_count: 1,
                source_payload_bytes_read: 0,
                peak_decoded_f32_bytes: 0,
            });
        }
        reject_missing_native_q2_kernel_on_metal(backend, &self.tensor_ref.name)?;

        let storage = self.gguf.tensor_quantized_storage(&self.tensor_ref.name)?;
        let scores = storage.matmul_rows_q2_k_f32(
            input.values(),
            batch,
            self.in_features,
            self.out_features,
        )?;
        let (token_id, token_score) = greedy_argmax_from_scores(&scores)?;

        Ok(LinearGreedyScan {
            token_id,
            token_score,
            batch,
            chunk_count: 1,
            source_payload_bytes_read: 0,
            peak_decoded_f32_bytes: 0,
        })
    }
}

#[derive(Debug)]
struct LinearF32Run {
    output: F32Tensor,
    chunk_count: usize,
    peak_decoded_f32_bytes: u64,
}

fn greedy_argmax_from_scores(scores: &[f32]) -> Result<(u32, f32)> {
    let mut best_token_id = 0_u32;
    let mut best_score = 0.0_f32;
    let mut initialized = false;
    for (token_id, score) in scores.iter().copied().enumerate() {
        let ordering = score.partial_cmp(&best_score).unwrap_or(Ordering::Less);
        if !initialized || ordering == Ordering::Greater {
            best_token_id = u32::try_from(token_id).map_err(|_| {
                Error::model(format!("GGUF greedy token id {token_id} does not fit u32"))
            })?;
            best_score = score;
            initialized = true;
        }
    }
    if !initialized {
        return Err(Error::model(
            "GGUF greedy direct quantized scoring produced no logits",
        ));
    }
    Ok((best_token_id, best_score))
}

fn tensor_to_f32_tensor(tensor: &Tensor) -> Result<F32Tensor> {
    let dims = tensor.dims().to_vec();
    let values = tensor
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    F32Tensor::new(values, dims)
}

fn tensor_from_f32_output(tensor: F32Tensor, device: &common::Device) -> Result<Tensor> {
    let (shape, values) = tensor.into_parts();
    tensor_from_f32_values(values, shape.dims(), device)
}

fn tensor_from_f32_values(
    values: Vec<f32>,
    shape: &[usize],
    device: &common::Device,
) -> Result<Tensor> {
    match shape {
        [rows, features] => Ok(Tensor::from_vec(values, (*rows, *features), device)?),
        [batch, tokens, features] => Ok(Tensor::from_vec(
            values,
            (*batch, *tokens, *features),
            device,
        )?),
        dims => Err(Error::model(format!(
            "GGUF quantized linear output rank must be 2 or 3, got {dims:?}"
        ))),
    }
}

fn add_f32_tensors(context: &str, lhs: &F32Tensor, rhs: &F32Tensor) -> Result<F32Tensor> {
    validate_exact_shape(context, lhs.dims(), rhs.dims())?;
    let values = lhs
        .values()
        .iter()
        .zip(rhs.values())
        .map(|(left, right)| left + right)
        .collect::<Vec<_>>();
    F32Tensor::new(values, lhs.dims().to_vec())
}

fn validate_input_shape_f32(
    context: &str,
    input: &F32Tensor,
    in_features: usize,
    out_features: usize,
) -> Result<(usize, Vec<usize>)> {
    validate_input_dims(context, input.dims(), in_features, out_features)
}

fn validate_input_dims(
    context: &str,
    dims: &[usize],
    in_features: usize,
    out_features: usize,
) -> Result<(usize, Vec<usize>)> {
    match dims {
        [rows, features] => {
            validate_exact_shape(
                "gguf_quantized_linear_input_features",
                &[*features],
                &[in_features],
            )?;
            Ok((*rows, vec![*rows, out_features]))
        }
        [batch, tokens, features] => {
            validate_exact_shape(
                "gguf_quantized_linear_input_features",
                &[*features],
                &[in_features],
            )?;
            let row_count = batch
                .checked_mul(*tokens)
                .ok_or_else(|| Error::model(format!("{context} input row count overflow")))?;
            Ok((row_count, vec![*batch, *tokens, out_features]))
        }
        dims => Err(Error::model(format!(
            "{context} input rank must be 2 or 3, got {dims:?}"
        ))),
    }
}

fn validate_greedy_input_shape_f32(input: &F32Tensor, in_features: usize) -> Result<usize> {
    let dims = input.dims();
    match dims {
        [batch, features] => {
            validate_exact_shape(
                "gguf_quantized_linear_input_features",
                &[*features],
                &[in_features],
            )?;
            Ok(*batch)
        }
        [batch, tokens, features] => {
            validate_exact_shape(
                "gguf_quantized_linear_greedy_input",
                &[*tokens, *features],
                &[1, in_features],
            )?;
            Ok(*batch)
        }
        _ => Err(Error::model(format!(
            "GGUF quantized linear greedy input rank must be 2 [B,H] or 3 [B,1,H], got {dims:?}"
        ))),
    }
}

fn reject_missing_native_q2_kernel_on_metal<B: Backend>(
    backend: &B,
    tensor_name: &str,
) -> Result<()> {
    if backend.capabilities().device == DeviceKind::Metal {
        return Err(Error::backend(format!(
            "native Metal Q2_K matvec is required for {tensor_name}; CPU reference fallback is disabled on Metal"
        )));
    }
    Ok(())
}

#[derive(Debug)]
struct LinearRun {
    output: Tensor,
    chunk_count: usize,
    peak_decoded_f32_bytes: u64,
}

#[derive(Debug)]
struct LinearGreedyScan {
    token_id: u32,
    token_score: f32,
    batch: usize,
    chunk_count: usize,
    source_payload_bytes_read: u64,
    peak_decoded_f32_bytes: u64,
}

fn validate_quantized_type(tensor_ref: &TensorRef) -> Result<()> {
    match tensor_ref.ty {
        GgmlType::Q2K | GgmlType::Q8_0 => Ok(()),
        other => Err(Error::gguf(format!(
            "GLM-5.2 Q2 GGUF quantized linear tensor {} must be Q2_K or Q8_0, got {other}",
            tensor_ref.name
        ))),
    }
}

fn validate_tensor_ref(tensor_ref: &TensorRef, info: &gguf::GgufTensorInfo) -> Result<()> {
    if tensor_ref.dims != info.dims {
        return Err(Error::gguf(format!(
            "GGUF tensor {} dims changed from {:?} to {:?}",
            tensor_ref.name, tensor_ref.dims, info.dims
        )));
    }
    if tensor_ref.ty != info.ty {
        return Err(Error::gguf(format!(
            "GGUF tensor {} type changed from {} to {}",
            tensor_ref.name, tensor_ref.ty, info.ty
        )));
    }
    if tensor_ref.absolute_offset != info.absolute_offset {
        return Err(Error::gguf(format!(
            "GGUF tensor {} offset changed from {} to {}",
            tensor_ref.name, tensor_ref.absolute_offset, info.absolute_offset
        )));
    }
    if tensor_ref.storage_byte_len != info.storage_byte_len {
        return Err(Error::gguf(format!(
            "GGUF tensor {} storage length changed from {} to {}",
            tensor_ref.name, tensor_ref.storage_byte_len, info.storage_byte_len
        )));
    }
    Ok(())
}

fn validate_input_shape(input: &Tensor, in_features: usize) -> Result<()> {
    let dims = input.dims();
    match dims {
        [_, features] => validate_exact_shape(
            "gguf_quantized_linear_input_features",
            &[*features],
            &[in_features],
        ),
        [_, _, features] => validate_exact_shape(
            "gguf_quantized_linear_input_features",
            &[*features],
            &[in_features],
        ),
        _ => Err(Error::model(format!(
            "GGUF quantized linear input rank must be 2 or 3, got {dims:?}"
        ))),
    }
}

fn validate_greedy_input_shape(input: &Tensor, in_features: usize) -> Result<usize> {
    let dims = input.dims();
    match dims {
        [batch, features] => {
            validate_exact_shape(
                "gguf_quantized_linear_input_features",
                &[*features],
                &[in_features],
            )?;
            Ok(*batch)
        }
        [batch, tokens, features] => {
            validate_exact_shape(
                "gguf_quantized_linear_greedy_input",
                &[*tokens, *features],
                &[1, in_features],
            )?;
            Ok(*batch)
        }
        _ => Err(Error::model(format!(
            "GGUF quantized linear greedy input rank must be 2 [B,H] or 3 [B,1,H], got {dims:?}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use backend::MetalBackend;
    use common::{Device, Tensor};
    use gguf::{
        GgmlType, GgufFile, GgufMetadataValueType, GGML_Q2_K_BLOCK_BYTES, GGUF_MAGIC,
        GGUF_VERSION_V3,
    };

    use super::*;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn q2_linear_runs_in_output_row_chunks() {
        let path = write_linear_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let tensor_ref = tensor_ref(&gguf, "output.weight");
        let linear = QuantizedLinear::open(&gguf, &tensor_ref, 256, 4, 3).unwrap();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let input = Tensor::from_vec(vec![1.0_f32; 2 * 256], (2, 256), &Device::Cpu).unwrap();

        let output = linear.forward(&input, &backend).unwrap();

        assert_eq!(output.output.dims(), &[2, 4]);
        assert_eq!(output.report.input_shape.dims(), &[2, 256]);
        assert_eq!(output.report.output_chunk_rows, 3);
        assert_eq!(output.report.chunk_count, 2);
        assert_eq!(
            output.report.source_payload_bytes_read,
            GGML_Q2_K_BLOCK_BYTES * 4
        );
        assert_eq!(output.report.peak_decoded_f32_bytes, 3 * 256 * 4);
    }

    #[test]
    fn q2_tensor_path_matches_chunked_reference() {
        let path = write_linear_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let tensor_ref = tensor_ref(&gguf, "output.weight");
        let linear = QuantizedLinear::open(&gguf, &tensor_ref, 256, 4, 2).unwrap();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let input = Tensor::from_vec(vec![1.0_f32; 2 * 256], (2, 256), &Device::Cpu).unwrap();
        let native_input = F32Tensor::new(vec![1.0_f32; 2 * 256], [2, 256]).unwrap();

        let reference = linear.forward(&input, &backend).unwrap().output;
        let optimized = linear.forward_tensor(&input, &backend).unwrap();
        let native = linear.forward_f32_tensor(&native_input, &backend).unwrap();

        assert_eq!(optimized.dims(), &[2, 4]);
        assert_eq!(native.dims(), &[2, 4]);
        let reference = reference.to_vec2::<f32>().unwrap();
        let optimized = optimized.to_vec2::<f32>().unwrap();
        for ((reference_row, optimized_row), native_row) in reference
            .iter()
            .zip(optimized.iter())
            .zip(native.values().chunks(4))
        {
            for ((reference_value, optimized_value), native_value) in reference_row
                .iter()
                .zip(optimized_row.iter())
                .zip(native_row.iter())
            {
                assert!((reference_value - optimized_value).abs() <= 1e-4);
                assert!((reference_value - native_value).abs() <= 1e-4);
            }
        }
    }

    #[test]
    fn rejects_non_quantized_linear_weight() {
        let path = write_f32_linear_fixture();
        let gguf = GgufFile::open(&path).unwrap();
        let tensor_ref = tensor_ref(&gguf, "output.weight");

        let err = QuantizedLinear::open(&gguf, &tensor_ref, 256, 4, 2)
            .expect_err("F32 output weight should not load through quantized path");

        assert!(err.to_string().contains("must be Q2_K or Q8_0"));
    }

    #[test]
    fn rejects_zero_chunk_rows() {
        let path = write_linear_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let tensor_ref = tensor_ref(&gguf, "output.weight");

        let err = QuantizedLinear::open(&gguf, &tensor_ref, 256, 4, 0)
            .expect_err("zero chunk rows should fail");

        assert!(err.to_string().contains("output_chunk_rows"));
    }

    fn tensor_ref(gguf: &GgufFile, name: &str) -> TensorRef {
        let info = gguf.tensor(name).unwrap();
        TensorRef {
            name: info.name.clone(),
            dims: info.dims.clone(),
            ty: info.ty,
            absolute_offset: info.absolute_offset,
            storage_byte_len: info.storage_byte_len,
        }
    }

    fn write_linear_fixture(ty: GgmlType) -> PathBuf {
        let path = unique_temp_file("linear");
        fs::write(&path, tiny_linear_gguf(ty)).unwrap();
        path
    }

    fn write_f32_linear_fixture() -> PathBuf {
        let path = unique_temp_file("linear-f32");
        let mut writer = GgufWriter::new();
        writer.header(1, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);
        writer.tensor_info("output.weight", &[256, 4], GgmlType::F32, 0);
        writer.pad_to(32);
        writer.bytes(&vec![0_u8; 256 * 4 * 4]);
        writer.finish_to(path)
    }

    fn tiny_linear_gguf(ty: GgmlType) -> Vec<u8> {
        let mut writer = GgufWriter::new();
        writer.header(1, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);
        writer.tensor_info("output.weight", &[256, 4], ty, 0);
        writer.pad_to(32);
        match ty {
            GgmlType::Q2K => {
                writer.bytes(&q2_k_block(0xe4, 1));
                writer.bytes(&q2_k_block(0xe4, 2));
                writer.bytes(&q2_k_block(0xe4, 3));
                writer.bytes(&q2_k_block(0xe4, 4));
            }
            other => panic!("unsupported fixture tensor type {other}"),
        }
        writer.bytes(&[0_u8; 16]);
        writer.finish()
    }

    fn q2_k_block(quant_byte: u8, min_shift: u8) -> Vec<u8> {
        let mut block = Vec::with_capacity(GGML_Q2_K_BLOCK_BYTES as usize);
        for scale in 1_u8..=16 {
            block.push((scale & 0x0f) | ((min_shift & 0x0f) << 4));
        }
        block.extend(std::iter::repeat_n(quant_byte, 64));
        block.extend_from_slice(&0x3c00_u16.to_le_bytes());
        block.extend_from_slice(&0x3800_u16.to_le_bytes());
        block
    }

    struct GgufWriter {
        bytes: Vec<u8>,
    }

    impl GgufWriter {
        fn new() -> Self {
            Self { bytes: Vec::new() }
        }

        fn header(&mut self, tensor_count: u64, metadata_kv_count: u64) {
            self.bytes.extend_from_slice(GGUF_MAGIC);
            self.u32(GGUF_VERSION_V3);
            self.u64(tensor_count);
            self.u64(metadata_kv_count);
        }

        fn metadata_key(&mut self, key: &str) {
            self.string(key);
        }

        fn tensor_info(&mut self, name: &str, dims: &[u64], ty: GgmlType, offset: u64) {
            self.string(name);
            self.u32(dims.len() as u32);
            for dim in dims {
                self.u64(*dim);
            }
            self.u32(ty.code());
            self.u64(offset);
        }

        fn string(&mut self, value: &str) {
            self.u64(value.len() as u64);
            self.bytes.extend_from_slice(value.as_bytes());
        }

        fn u32(&mut self, value: u32) {
            self.bytes.extend_from_slice(&value.to_le_bytes());
        }

        fn u64(&mut self, value: u64) {
            self.bytes.extend_from_slice(&value.to_le_bytes());
        }

        fn pad_to(&mut self, alignment: usize) {
            let remainder = self.bytes.len() % alignment;
            if remainder != 0 {
                self.bytes
                    .resize(self.bytes.len() + alignment - remainder, 0);
            }
        }

        fn bytes(&mut self, bytes: &[u8]) {
            self.bytes.extend_from_slice(bytes);
        }

        fn finish(self) -> Vec<u8> {
            self.bytes
        }

        fn finish_to(self, path: PathBuf) -> PathBuf {
            fs::write(&path, self.finish()).unwrap();
            path
        }
    }

    fn unique_temp_file(label: &str) -> PathBuf {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("linear-{label}-{}-{id}", std::process::id()))
    }
}
