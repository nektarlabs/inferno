use ::metal::{Buffer, CommandBufferRef, ComputePipelineState, Device, MTLResourceOptions};
use common::{Error, Result};

use crate::{DFlashAttentionCache, DeviceDFlashW4Matrix, DeviceQwenBf16Tensor};

use super::super::{
    arena::MetalArena,
    buffers::{require_byte_capacity, require_f16_capacity},
    command::{encode_1d_threadgroups_args, KernelArg},
    library::MetalLibrary,
    pipeline::compute_pipeline,
};

const KERNEL_SOURCE: &str = include_str!("kernels/dflash.metal");
const PACK_FEATURES_KERNEL: &str = "dflash_pack_features_kernel";
const COPY_ROWS_KERNEL: &str = "dflash_copy_rows_kernel";
const RMS_NORM_KERNEL: &str = "dflash_rms_norm_kernel";
const LINEAR_ROWS8_KERNEL: &str = "dflash_bf16_linear_rows8_kernel";
const GATE_UP_SWIGLU_ROWS8_KERNEL: &str = "dflash_bf16_gate_up_swiglu_rows8_kernel";
const QUANTIZE_W4_KERNEL: &str = "dflash_quantize_bf16_w4_group64_kernel";
const W4_LINEAR_ROWS8_KERNEL: &str = "dflash_w4_linear_rows8_kernel";
const W4_GATE_UP_SWIGLU_ROWS8_KERNEL: &str = "dflash_w4_gate_up_swiglu_rows8_kernel";
const DYNAMIC_CONV_KERNEL: &str = "dflash_dynamic_conv_kernel";
const DYNAMIC_CONV_RESIDUAL_KERNEL: &str = "dflash_dynamic_conv_residual_kernel";
const QUERY_KERNEL: &str = "dflash_query_norm_rope_kernel";
const CONTEXT_APPEND_KERNEL: &str = "dflash_context_kv_append_kernel";
const PROPOSAL_KEY_KERNEL: &str = "dflash_proposal_key_norm_rope_kernel";
const ATTENTION_KERNEL: &str = "dflash_online_sliding_gqa_kernel";
const LOGITS_KERNEL: &str = "dflash_all_row_logits_kernel";
const TOP_K_KERNEL: &str = "dflash_top_k_kernel";
const SELECTOR_KERNEL: &str = "dflash_path_selector_kernel";
const SIMD_LANES: usize = 32;
const HIDDEN_SIZE: usize = 5_120;
const QUERY_HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const VOCAB_SIZE: usize = 248_320;
const SELECTOR_RANK: usize = 256;
const MAX_PROPOSAL_ROWS: usize = 7;
const SELECTOR_TOP_K: usize = 16;
const LINEAR_ROW_TILE: usize = 8;
const W4_GROUP_SIZE: usize = 64;

pub(super) struct MetalDFlash {
    pack_features: ComputePipelineState,
    copy_rows: ComputePipelineState,
    rms_norm: ComputePipelineState,
    linear_rows8: ComputePipelineState,
    gate_up_swiglu_rows8: ComputePipelineState,
    quantize_w4: ComputePipelineState,
    w4_linear_rows8: ComputePipelineState,
    w4_gate_up_swiglu_rows8: ComputePipelineState,
    dynamic_conv: ComputePipelineState,
    dynamic_conv_residual: ComputePipelineState,
    query: ComputePipelineState,
    context_append: ComputePipelineState,
    proposal_key: ComputePipelineState,
    attention: ComputePipelineState,
    logits: ComputePipelineState,
    top_k: ComputePipelineState,
    selector: ComputePipelineState,
    arena: MetalArena,
}

impl MetalDFlash {
    pub(super) fn new(device: &Device, arena: MetalArena) -> Result<Self> {
        let library = MetalLibrary::compile_source(device, KERNEL_SOURCE)?;
        let instance = Self {
            pack_features: compute_pipeline(device, &library, PACK_FEATURES_KERNEL)?,
            copy_rows: compute_pipeline(device, &library, COPY_ROWS_KERNEL)?,
            rms_norm: compute_pipeline(device, &library, RMS_NORM_KERNEL)?,
            linear_rows8: compute_pipeline(device, &library, LINEAR_ROWS8_KERNEL)?,
            gate_up_swiglu_rows8: compute_pipeline(device, &library, GATE_UP_SWIGLU_ROWS8_KERNEL)?,
            quantize_w4: compute_pipeline(device, &library, QUANTIZE_W4_KERNEL)?,
            w4_linear_rows8: compute_pipeline(device, &library, W4_LINEAR_ROWS8_KERNEL)?,
            w4_gate_up_swiglu_rows8: compute_pipeline(
                device,
                &library,
                W4_GATE_UP_SWIGLU_ROWS8_KERNEL,
            )?,
            dynamic_conv: compute_pipeline(device, &library, DYNAMIC_CONV_KERNEL)?,
            dynamic_conv_residual: compute_pipeline(
                device,
                &library,
                DYNAMIC_CONV_RESIDUAL_KERNEL,
            )?,
            query: compute_pipeline(device, &library, QUERY_KERNEL)?,
            context_append: compute_pipeline(device, &library, CONTEXT_APPEND_KERNEL)?,
            proposal_key: compute_pipeline(device, &library, PROPOSAL_KEY_KERNEL)?,
            attention: compute_pipeline(device, &library, ATTENTION_KERNEL)?,
            logits: compute_pipeline(device, &library, LOGITS_KERNEL)?,
            top_k: compute_pipeline(device, &library, TOP_K_KERNEL)?,
            selector: compute_pipeline(device, &library, SELECTOR_KERNEL)?,
            arena,
        };
        for (pipeline, name) in [
            (&instance.rms_norm, RMS_NORM_KERNEL),
            (&instance.linear_rows8, LINEAR_ROWS8_KERNEL),
            (&instance.gate_up_swiglu_rows8, GATE_UP_SWIGLU_ROWS8_KERNEL),
            (&instance.quantize_w4, QUANTIZE_W4_KERNEL),
            (&instance.w4_linear_rows8, W4_LINEAR_ROWS8_KERNEL),
            (
                &instance.w4_gate_up_swiglu_rows8,
                W4_GATE_UP_SWIGLU_ROWS8_KERNEL,
            ),
            (&instance.query, QUERY_KERNEL),
            (&instance.context_append, CONTEXT_APPEND_KERNEL),
            (&instance.proposal_key, PROPOSAL_KEY_KERNEL),
            (&instance.attention, ATTENTION_KERNEL),
            (&instance.logits, LOGITS_KERNEL),
        ] {
            require_simd_width(pipeline, name)?;
        }
        Ok(instance)
    }

    pub(super) fn encode_pack_features(
        &self,
        command_buffer: &CommandBufferRef,
        features: [&Buffer; 5],
        row_count: usize,
    ) -> Result<Buffer> {
        if row_count == 0 {
            return Err(Error::backend("DFlash2 feature packing requires rows"));
        }
        for feature in features {
            require_f16_capacity(feature, row_count * HIDDEN_SIZE, "DFlash2 target feature")?;
        }
        let output_len = row_count
            .checked_mul(5 * HIDDEN_SIZE)
            .ok_or_else(|| Error::backend("DFlash2 feature output size overflow"))?;
        let output = self.arena.empty_f16(output_len)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.pack_features,
            &[
                KernelArg::Buffer(features[0]),
                KernelArg::Buffer(features[1]),
                KernelArg::Buffer(features[2]),
                KernelArg::Buffer(features[3]),
                KernelArg::Buffer(features[4]),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(row_count, "feature rows")?),
                KernelArg::U32(HIDDEN_SIZE as u32),
            ],
            output_len.div_ceil(256),
            256,
        )?;
        Ok(output)
    }

    pub(super) fn encode_copy_rows(
        &self,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        row_start: usize,
        row_count: usize,
        row_width: usize,
    ) -> Result<Buffer> {
        if row_count == 0 || row_width == 0 {
            return Err(Error::backend(
                "DFlash2 row copy requires positive dimensions",
            ));
        }
        let source_rows = row_start
            .checked_add(row_count)
            .ok_or_else(|| Error::backend("DFlash2 row range overflow"))?;
        require_f16_capacity(input, source_rows * row_width, "DFlash2 row source")?;
        let output_len = row_count * row_width;
        let output = self.arena.empty_f16(output_len)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.copy_rows,
            &[
                KernelArg::Buffer(input),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(row_start, "row start")?),
                KernelArg::U32(as_u32(row_width, "row width")?),
                KernelArg::U32(as_u32(output_len, "row output size")?),
            ],
            output_len.div_ceil(256),
            256,
        )?;
        Ok(output)
    }

    pub(super) fn encode_linear(
        &self,
        command_buffer: &CommandBufferRef,
        matrix: &DeviceQwenBf16Tensor,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Buffer> {
        let [out_features, in_features]: [usize; 2] = matrix
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| Error::backend("DFlash2 BF16 matrix must be rank 2"))?;
        let expected = row_count
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("DFlash2 BF16 linear input size overflow"))?;
        if row_count == 0 || input_len != expected {
            return Err(Error::backend(format!(
                "DFlash2 BF16 linear [{row_count},{in_features}] requires {expected} values, got {input_len}"
            )));
        }
        require_f16_capacity(input, input_len, "DFlash2 BF16 linear input")?;
        require_tensor_range(matrix, "DFlash2 BF16 linear weight")?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("DFlash2 BF16 linear output size overflow"))?;
        let row_tiles = row_count.div_ceil(LINEAR_ROW_TILE);
        let threadgroups = row_tiles
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("DFlash2 BF16 linear grid overflow"))?;
        let output = self.arena.empty_f16(output_len)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.linear_rows8,
            &[
                KernelArg::BufferOffset(&matrix.buffer, matrix.byte_offset),
                KernelArg::Buffer(input),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(row_count, "linear rows")?),
                KernelArg::U32(as_u32(in_features, "linear input width")?),
                KernelArg::U32(as_u32(out_features, "linear output width")?),
            ],
            threadgroups,
            SIMD_LANES,
        )?;
        Ok(output)
    }

    pub(super) fn encode_gate_up_swiglu(
        &self,
        command_buffer: &CommandBufferRef,
        gate: &DeviceQwenBf16Tensor,
        up: &DeviceQwenBf16Tensor,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Buffer> {
        if gate.shape != up.shape {
            return Err(Error::backend(
                "DFlash2 gate/up matrices must have matching shapes",
            ));
        }
        let [out_features, in_features]: [usize; 2] = gate
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| Error::backend("DFlash2 gate/up matrices must be rank 2"))?;
        let expected = row_count
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("DFlash2 gate/up input size overflow"))?;
        if row_count == 0 || input_len != expected {
            return Err(Error::backend(format!(
                "DFlash2 gate/up [{row_count},{in_features}] requires {expected} values, got {input_len}"
            )));
        }
        require_f16_capacity(input, input_len, "DFlash2 gate/up input")?;
        require_tensor_range(gate, "DFlash2 gate weight")?;
        require_tensor_range(up, "DFlash2 up weight")?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("DFlash2 gate/up output size overflow"))?;
        let row_tiles = row_count.div_ceil(LINEAR_ROW_TILE);
        let threadgroups = row_tiles
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("DFlash2 gate/up grid overflow"))?;
        let output = self.arena.empty_f16(output_len)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.gate_up_swiglu_rows8,
            &[
                KernelArg::BufferOffset(&gate.buffer, gate.byte_offset),
                KernelArg::BufferOffset(&up.buffer, up.byte_offset),
                KernelArg::Buffer(input),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(row_count, "gate/up rows")?),
                KernelArg::U32(as_u32(in_features, "gate/up input width")?),
                KernelArg::U32(as_u32(out_features, "gate/up output width")?),
            ],
            threadgroups,
            SIMD_LANES,
        )?;
        Ok(output)
    }

    pub(super) fn encode_quantize_w4(
        &self,
        device: &Device,
        command_buffer: &CommandBufferRef,
        source: &DeviceQwenBf16Tensor,
        rows: usize,
        columns: usize,
        group_size: usize,
    ) -> Result<DeviceDFlashW4Matrix> {
        if source.shape != [rows, columns]
            || rows == 0
            || columns == 0
            || group_size != W4_GROUP_SIZE
            || !columns.is_multiple_of(group_size)
        {
            return Err(Error::backend(format!(
                "DFlash2 W4 quantization requires BF16 [{rows},{columns}] with group {W4_GROUP_SIZE}, got {:?} and group {group_size}",
                source.shape
            )));
        }
        require_tensor_range(source, "DFlash2 W4 source")?;
        let value_count = rows
            .checked_mul(columns)
            .ok_or_else(|| Error::backend("DFlash2 W4 value count overflow"))?;
        let packed_bytes = value_count / 2;
        let scale_count = rows
            .checked_mul(columns / group_size)
            .ok_or_else(|| Error::backend("DFlash2 W4 scale count overflow"))?;
        let scale_bytes = scale_count
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| Error::backend("DFlash2 W4 scale byte count overflow"))?;
        let bias_bytes = scale_bytes;
        let options = MTLResourceOptions::StorageModeShared
            .union(MTLResourceOptions::HazardTrackingModeTracked);
        let packed = device.new_buffer(packed_bytes as u64, options);
        packed.set_label("DFlash2 persistent W4 values");
        let scales = device.new_buffer(scale_bytes as u64, options);
        scales.set_label("DFlash2 persistent W4 scales");
        let biases = device.new_buffer(bias_bytes as u64, options);
        biases.set_label("DFlash2 persistent W4 biases");
        encode_1d_threadgroups_args(
            command_buffer,
            &self.quantize_w4,
            &[
                KernelArg::BufferOffset(&source.buffer, source.byte_offset),
                KernelArg::Buffer(&packed),
                KernelArg::Buffer(&scales),
                KernelArg::Buffer(&biases),
                KernelArg::U32(as_u32(rows, "W4 rows")?),
                KernelArg::U32(as_u32(columns, "W4 columns")?),
            ],
            scale_count,
            SIMD_LANES,
        )?;
        Ok(DeviceDFlashW4Matrix {
            rows,
            columns,
            group_size,
            packed_bytes,
            scale_bytes,
            bias_bytes,
            packed,
            scales,
            biases,
        })
    }

    pub(super) fn encode_w4_linear(
        &self,
        command_buffer: &CommandBufferRef,
        matrix: &DeviceDFlashW4Matrix,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Buffer> {
        validate_w4_matrix(matrix)?;
        let expected = row_count
            .checked_mul(matrix.columns)
            .ok_or_else(|| Error::backend("DFlash2 W4 linear input size overflow"))?;
        if row_count == 0 || input_len != expected {
            return Err(Error::backend(format!(
                "DFlash2 W4 linear [{row_count},{}] requires {expected} values, got {input_len}",
                matrix.columns
            )));
        }
        require_f16_capacity(input, input_len, "DFlash2 W4 linear input")?;
        let output_len = row_count
            .checked_mul(matrix.rows)
            .ok_or_else(|| Error::backend("DFlash2 W4 linear output size overflow"))?;
        let threadgroups = row_count
            .div_ceil(LINEAR_ROW_TILE)
            .checked_mul(matrix.rows.div_ceil(2))
            .ok_or_else(|| Error::backend("DFlash2 W4 linear grid overflow"))?;
        let output = self.arena.empty_f16(output_len)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.w4_linear_rows8,
            &[
                KernelArg::Buffer(&matrix.packed),
                KernelArg::Buffer(&matrix.scales),
                KernelArg::Buffer(&matrix.biases),
                KernelArg::Buffer(input),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(row_count, "W4 linear rows")?),
                KernelArg::U32(as_u32(matrix.columns, "W4 linear input width")?),
                KernelArg::U32(as_u32(matrix.rows, "W4 linear output width")?),
            ],
            threadgroups,
            SIMD_LANES * 2,
        )?;
        Ok(output)
    }

    pub(super) fn encode_w4_gate_up_swiglu(
        &self,
        command_buffer: &CommandBufferRef,
        gate: &DeviceDFlashW4Matrix,
        up: &DeviceDFlashW4Matrix,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Buffer> {
        validate_w4_matrix(gate)?;
        validate_w4_matrix(up)?;
        if gate.rows != up.rows || gate.columns != up.columns || gate.group_size != up.group_size {
            return Err(Error::backend(
                "DFlash2 W4 gate/up matrices must have matching layouts",
            ));
        }
        let expected = row_count
            .checked_mul(gate.columns)
            .ok_or_else(|| Error::backend("DFlash2 W4 gate/up input size overflow"))?;
        if row_count == 0 || input_len != expected {
            return Err(Error::backend(format!(
                "DFlash2 W4 gate/up [{row_count},{}] requires {expected} values, got {input_len}",
                gate.columns
            )));
        }
        require_f16_capacity(input, input_len, "DFlash2 W4 gate/up input")?;
        let output_len = row_count
            .checked_mul(gate.rows)
            .ok_or_else(|| Error::backend("DFlash2 W4 gate/up output size overflow"))?;
        let threadgroups = row_count
            .div_ceil(LINEAR_ROW_TILE)
            .checked_mul(gate.rows)
            .ok_or_else(|| Error::backend("DFlash2 W4 gate/up grid overflow"))?;
        let output = self.arena.empty_f16(output_len)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.w4_gate_up_swiglu_rows8,
            &[
                KernelArg::Buffer(&gate.packed),
                KernelArg::Buffer(&gate.scales),
                KernelArg::Buffer(&gate.biases),
                KernelArg::Buffer(&up.packed),
                KernelArg::Buffer(&up.scales),
                KernelArg::Buffer(&up.biases),
                KernelArg::Buffer(input),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(row_count, "W4 gate/up rows")?),
                KernelArg::U32(as_u32(gate.columns, "W4 gate/up input width")?),
                KernelArg::U32(as_u32(gate.rows, "W4 gate/up output width")?),
            ],
            threadgroups,
            SIMD_LANES * 2,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_rms_norm(
        &self,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        input_len: usize,
        weight: &DeviceQwenBf16Tensor,
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<Buffer> {
        if weight.shape != [hidden_size]
            || rows == 0
            || input_len != rows * hidden_size
            || !eps.is_finite()
            || eps <= 0.0
        {
            return Err(Error::backend(format!(
                "invalid DFlash2 RMSNorm input={input_len}, rows={rows}, hidden={hidden_size}, weight={:?}, eps={eps}",
                weight.shape
            )));
        }
        require_f16_capacity(input, input_len, "DFlash2 RMSNorm input")?;
        require_tensor_range(weight, "DFlash2 RMSNorm weight")?;
        let output = self.arena.empty_f16(input_len)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.rms_norm,
            &[
                KernelArg::Buffer(input),
                KernelArg::BufferOffset(&weight.buffer, weight.byte_offset),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(rows, "RMSNorm rows")?),
                KernelArg::U32(as_u32(hidden_size, "RMSNorm width")?),
                KernelArg::F32(eps),
                KernelArg::U32(0),
            ],
            rows,
            SIMD_LANES,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_rms_norm_suffix(
        &self,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        input_len: usize,
        weight: &DeviceQwenBf16Tensor,
        total_rows: usize,
        suffix_rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<Buffer> {
        if weight.shape != [hidden_size]
            || total_rows == 0
            || suffix_rows == 0
            || suffix_rows > total_rows
            || input_len != total_rows * hidden_size
            || !eps.is_finite()
            || eps <= 0.0
        {
            return Err(Error::backend(format!(
                "invalid DFlash2 suffix RMSNorm input={input_len}, total_rows={total_rows}, suffix_rows={suffix_rows}, hidden={hidden_size}, weight={:?}, eps={eps}",
                weight.shape
            )));
        }
        require_f16_capacity(input, input_len, "DFlash2 suffix RMSNorm input")?;
        require_tensor_range(weight, "DFlash2 suffix RMSNorm weight")?;
        let output_len = suffix_rows * hidden_size;
        let output = self.arena.empty_f16(output_len)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.rms_norm,
            &[
                KernelArg::Buffer(input),
                KernelArg::BufferOffset(&weight.buffer, weight.byte_offset),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(suffix_rows, "suffix RMSNorm rows")?),
                KernelArg::U32(as_u32(hidden_size, "suffix RMSNorm width")?),
                KernelArg::F32(eps),
                KernelArg::U32(as_u32(
                    total_rows - suffix_rows,
                    "suffix RMSNorm row start",
                )?),
            ],
            suffix_rows,
            SIMD_LANES,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_dynamic_conv(
        &self,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        dynamic: &Buffer,
        base_kernel: &DeviceQwenBf16Tensor,
        rows: usize,
        sequence_length: usize,
        hidden_size: usize,
        stage: usize,
        kernel_size: usize,
        group_size: usize,
    ) -> Result<Buffer> {
        if rows == 0
            || sequence_length == 0
            || !rows.is_multiple_of(sequence_length)
            || hidden_size == 0
            || stage >= 2
            || kernel_size == 0
            || group_size == 0
            || !hidden_size.is_multiple_of(group_size)
            || base_kernel.shape != [2, kernel_size, hidden_size]
        {
            return Err(Error::backend("invalid DFlash2 dynamic convolution shape"));
        }
        let groups = hidden_size / group_size;
        let dynamic_width = 2 * kernel_size * groups;
        require_f16_capacity(input, rows * hidden_size, "DFlash2 convolution input")?;
        require_f16_capacity(dynamic, rows * dynamic_width, "DFlash2 dynamic kernel")?;
        require_tensor_range(base_kernel, "DFlash2 base convolution kernel")?;
        let output = self.arena.empty_f16(rows * hidden_size)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.dynamic_conv,
            &[
                KernelArg::Buffer(input),
                KernelArg::Buffer(dynamic),
                KernelArg::BufferOffset(&base_kernel.buffer, base_kernel.byte_offset),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(rows, "convolution rows")?),
                KernelArg::U32(as_u32(sequence_length, "convolution sequence")?),
                KernelArg::U32(as_u32(hidden_size, "convolution hidden")?),
                KernelArg::U32(as_u32(stage, "convolution stage")?),
                KernelArg::U32(as_u32(kernel_size, "convolution kernel")?),
                KernelArg::U32(as_u32(group_size, "convolution group")?),
            ],
            (rows * hidden_size).div_ceil(256),
            256,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_dynamic_conv_residual(
        &self,
        command_buffer: &CommandBufferRef,
        input: &Buffer,
        dynamic: &Buffer,
        base_kernel: &DeviceQwenBf16Tensor,
        residual: &Buffer,
        rows: usize,
        sequence_length: usize,
        hidden_size: usize,
        kernel_size: usize,
        group_size: usize,
    ) -> Result<Buffer> {
        if rows == 0
            || sequence_length == 0
            || !rows.is_multiple_of(sequence_length)
            || hidden_size == 0
            || kernel_size == 0
            || group_size == 0
            || !hidden_size.is_multiple_of(group_size)
            || base_kernel.shape != [2, kernel_size, hidden_size]
        {
            return Err(Error::backend(
                "invalid DFlash2 dynamic convolution residual shape",
            ));
        }
        let groups = hidden_size / group_size;
        let dynamic_width = 2 * kernel_size * groups;
        let value_count = rows * hidden_size;
        require_f16_capacity(input, value_count, "DFlash2 convolution input")?;
        require_f16_capacity(dynamic, rows * dynamic_width, "DFlash2 dynamic kernel")?;
        require_f16_capacity(residual, value_count, "DFlash2 residual")?;
        require_tensor_range(base_kernel, "DFlash2 base convolution kernel")?;
        let output = self.arena.empty_f16(value_count)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.dynamic_conv_residual,
            &[
                KernelArg::Buffer(input),
                KernelArg::Buffer(dynamic),
                KernelArg::BufferOffset(&base_kernel.buffer, base_kernel.byte_offset),
                KernelArg::Buffer(residual),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(rows, "convolution rows")?),
                KernelArg::U32(as_u32(sequence_length, "convolution sequence")?),
                KernelArg::U32(as_u32(hidden_size, "convolution hidden")?),
                KernelArg::U32(as_u32(kernel_size, "convolution kernel")?),
                KernelArg::U32(as_u32(group_size, "convolution group")?),
            ],
            value_count.div_ceil(256),
            256,
        )?;
        Ok(output)
    }

    pub(super) fn create_attention_cache(
        &self,
        device: &Device,
        batch: usize,
        capacity_tokens: usize,
    ) -> Result<DFlashAttentionCache> {
        if batch == 0 || capacity_tokens == 0 {
            return Err(Error::cache("DFlash2 cache dimensions must be positive"));
        }
        let values = batch
            .checked_mul(capacity_tokens)
            .and_then(|count| count.checked_mul(KV_HEADS * HEAD_DIM))
            .ok_or_else(|| Error::cache("DFlash2 cache size overflow"))?;
        let bytes = values
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| Error::cache("DFlash2 cache byte count overflow"))?;
        let options = MTLResourceOptions::StorageModeShared
            .union(MTLResourceOptions::HazardTrackingModeTracked);
        let key = device.new_buffer(bytes as u64, options);
        key.set_label("DFlash2 key cache");
        let value = device.new_buffer(bytes as u64, options);
        value.set_label("DFlash2 value cache");
        Ok(DFlashAttentionCache {
            batch,
            capacity_tokens,
            length: 0,
            next_position: 0,
            kv_heads: KV_HEADS,
            head_dim: HEAD_DIM,
            key,
            value,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_attention(
        &self,
        command_buffer: &CommandBufferRef,
        query_projection: &Buffer,
        context_key: &Buffer,
        context_value: &Buffer,
        proposal_key: &Buffer,
        proposal_value: &Buffer,
        query_norm: &DeviceQwenBf16Tensor,
        key_norm: &DeviceQwenBf16Tensor,
        cache: &DFlashAttentionCache,
        batch: usize,
        proposal_length: usize,
        context_length: usize,
        context_position_start: usize,
        rope_theta: f32,
        sliding_window: usize,
    ) -> Result<Buffer> {
        validate_attention(
            cache,
            batch,
            proposal_length,
            context_length,
            context_position_start,
            rope_theta,
            sliding_window,
        )?;
        let proposal_rows = batch * proposal_length;
        let context_rows = batch * context_length;
        require_f16_capacity(
            query_projection,
            proposal_rows * QUERY_HEADS * HEAD_DIM,
            "DFlash2 query projection",
        )?;
        require_f16_capacity(
            context_key,
            context_rows * KV_HEADS * HEAD_DIM,
            "DFlash2 context key",
        )?;
        require_f16_capacity(
            context_value,
            context_rows * KV_HEADS * HEAD_DIM,
            "DFlash2 context value",
        )?;
        require_f16_capacity(
            proposal_key,
            proposal_rows * KV_HEADS * HEAD_DIM,
            "DFlash2 proposal key",
        )?;
        require_f16_capacity(
            proposal_value,
            proposal_rows * KV_HEADS * HEAD_DIM,
            "DFlash2 proposal value",
        )?;
        for (norm, label) in [
            (query_norm, "DFlash2 query norm"),
            (key_norm, "DFlash2 key norm"),
        ] {
            if norm.shape != [HEAD_DIM] {
                return Err(Error::backend(format!(
                    "{label} must have shape [{HEAD_DIM}], got {:?}",
                    norm.shape
                )));
            }
            require_tensor_range(norm, label)?;
        }

        let query = self
            .arena
            .empty_f16(proposal_rows * QUERY_HEADS * HEAD_DIM)?;
        let normalized_proposal_key = self.arena.empty_f16(proposal_rows * KV_HEADS * HEAD_DIM)?;
        let output = self
            .arena
            .empty_f16(proposal_rows * QUERY_HEADS * HEAD_DIM)?;
        let next_position = context_position_start + context_length;

        encode_1d_threadgroups_args(
            command_buffer,
            &self.query,
            &[
                KernelArg::Buffer(query_projection),
                KernelArg::BufferOffset(&query_norm.buffer, query_norm.byte_offset),
                KernelArg::Buffer(&query),
                KernelArg::U32(as_u32(proposal_rows, "query rows")?),
                KernelArg::U32(as_u32(proposal_length, "proposal length")?),
                KernelArg::U32(as_u32(next_position, "proposal position")?),
                KernelArg::F32(rope_theta),
            ],
            proposal_rows * QUERY_HEADS,
            SIMD_LANES,
        )?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.context_append,
            &[
                KernelArg::Buffer(context_key),
                KernelArg::Buffer(context_value),
                KernelArg::BufferOffset(&key_norm.buffer, key_norm.byte_offset),
                KernelArg::Buffer(&cache.key),
                KernelArg::Buffer(&cache.value),
                KernelArg::U32(as_u32(context_rows, "context rows")?),
                KernelArg::U32(as_u32(context_length, "context length")?),
                KernelArg::U32(as_u32(cache.capacity_tokens, "cache capacity")?),
                KernelArg::U32(as_u32(context_position_start, "context position")?),
                KernelArg::F32(rope_theta),
            ],
            context_rows * KV_HEADS,
            SIMD_LANES,
        )?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.proposal_key,
            &[
                KernelArg::Buffer(proposal_key),
                KernelArg::BufferOffset(&key_norm.buffer, key_norm.byte_offset),
                KernelArg::Buffer(&normalized_proposal_key),
                KernelArg::U32(as_u32(proposal_rows, "proposal key rows")?),
                KernelArg::U32(as_u32(proposal_length, "proposal key length")?),
                KernelArg::U32(as_u32(next_position, "proposal key position")?),
                KernelArg::F32(rope_theta),
            ],
            proposal_rows * KV_HEADS,
            SIMD_LANES,
        )?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.attention,
            &[
                KernelArg::Buffer(&query),
                KernelArg::Buffer(&normalized_proposal_key),
                KernelArg::Buffer(proposal_value),
                KernelArg::Buffer(&cache.key),
                KernelArg::Buffer(&cache.value),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(proposal_rows, "attention rows")?),
                KernelArg::U32(as_u32(proposal_length, "attention sequence")?),
                KernelArg::U32(as_u32(cache.capacity_tokens, "attention cache capacity")?),
                KernelArg::U32(as_u32(
                    cache
                        .length
                        .saturating_add(context_length)
                        .min(cache.capacity_tokens),
                    "attention cache length",
                )?),
                KernelArg::U32(as_u32(next_position, "attention next position")?),
                KernelArg::U32(as_u32(sliding_window, "attention window")?),
            ],
            proposal_rows * QUERY_HEADS,
            SIMD_LANES,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_select_candidates(
        &self,
        command_buffer: &CommandBufferRef,
        output_weight: &DeviceQwenBf16Tensor,
        hidden_states: &Buffer,
        projected_hidden: &Buffer,
        predecessor_codebook: &DeviceQwenBf16Tensor,
        successor_codebook: &DeviceQwenBf16Tensor,
        row_count: usize,
        anchor_token: u32,
        top_k: usize,
    ) -> Result<Buffer> {
        if row_count == 0
            || row_count > MAX_PROPOSAL_ROWS
            || top_k != SELECTOR_TOP_K
            || anchor_token as usize >= VOCAB_SIZE
            || output_weight.shape != [VOCAB_SIZE, HIDDEN_SIZE]
            || predecessor_codebook.shape != [VOCAB_SIZE, SELECTOR_RANK]
            || successor_codebook.shape != [VOCAB_SIZE, SELECTOR_RANK]
        {
            return Err(Error::backend("invalid DFlash2 candidate selector shape"));
        }
        require_f16_capacity(
            hidden_states,
            row_count * HIDDEN_SIZE,
            "DFlash2 selector hidden states",
        )?;
        require_f16_capacity(
            projected_hidden,
            row_count * SELECTOR_RANK,
            "DFlash2 projected selector hidden states",
        )?;
        for (tensor, label) in [
            (output_weight, "DFlash2 output head"),
            (predecessor_codebook, "DFlash2 predecessor codebook"),
            (successor_codebook, "DFlash2 successor codebook"),
        ] {
            require_tensor_range(tensor, label)?;
        }
        let logits = self.arena.empty_f16(row_count * VOCAB_SIZE)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.logits,
            &[
                KernelArg::BufferOffset(&output_weight.buffer, output_weight.byte_offset),
                KernelArg::Buffer(hidden_states),
                KernelArg::Buffer(&logits),
                KernelArg::U32(as_u32(row_count, "logit rows")?),
            ],
            VOCAB_SIZE,
            SIMD_LANES,
        )?;
        self.encode_select_candidates_from_logits(
            command_buffer,
            &logits,
            projected_hidden,
            predecessor_codebook,
            successor_codebook,
            row_count,
            anchor_token,
            top_k,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_select_candidates_from_logits(
        &self,
        command_buffer: &CommandBufferRef,
        logits: &Buffer,
        projected_hidden: &Buffer,
        predecessor_codebook: &DeviceQwenBf16Tensor,
        successor_codebook: &DeviceQwenBf16Tensor,
        row_count: usize,
        anchor_token: u32,
        top_k: usize,
    ) -> Result<Buffer> {
        if row_count == 0
            || row_count > MAX_PROPOSAL_ROWS
            || top_k != SELECTOR_TOP_K
            || anchor_token as usize >= VOCAB_SIZE
            || predecessor_codebook.shape != [VOCAB_SIZE, SELECTOR_RANK]
            || successor_codebook.shape != [VOCAB_SIZE, SELECTOR_RANK]
        {
            return Err(Error::backend("invalid DFlash2 candidate selector shape"));
        }
        require_f16_capacity(logits, row_count * VOCAB_SIZE, "DFlash2 selector logits")?;
        require_f16_capacity(
            projected_hidden,
            row_count * SELECTOR_RANK,
            "DFlash2 projected selector hidden states",
        )?;
        require_tensor_range(predecessor_codebook, "DFlash2 predecessor codebook")?;
        require_tensor_range(successor_codebook, "DFlash2 successor codebook")?;
        let candidate_ids = self.arena.empty_u32(row_count * top_k)?;
        let candidate_logits = self.arena.empty_f32(row_count * top_k)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.top_k,
            &[
                KernelArg::Buffer(&logits),
                KernelArg::Buffer(&candidate_ids),
                KernelArg::Buffer(&candidate_logits),
                KernelArg::U32(as_u32(row_count, "top-k rows")?),
            ],
            row_count,
            128,
        )?;
        self.encode_path_selector(
            command_buffer,
            &candidate_ids,
            &candidate_logits,
            projected_hidden,
            predecessor_codebook,
            successor_codebook,
            row_count,
            anchor_token,
            top_k,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_path_selector(
        &self,
        command_buffer: &CommandBufferRef,
        candidate_ids: &Buffer,
        candidate_logits: &Buffer,
        projected_hidden: &Buffer,
        predecessor_codebook: &DeviceQwenBf16Tensor,
        successor_codebook: &DeviceQwenBf16Tensor,
        row_count: usize,
        anchor_token: u32,
        top_k: usize,
    ) -> Result<Buffer> {
        if row_count == 0
            || row_count > MAX_PROPOSAL_ROWS
            || top_k != SELECTOR_TOP_K
            || anchor_token as usize >= VOCAB_SIZE
            || predecessor_codebook.shape != [VOCAB_SIZE, SELECTOR_RANK]
            || successor_codebook.shape != [VOCAB_SIZE, SELECTOR_RANK]
        {
            return Err(Error::backend("invalid DFlash2 path selector shape"));
        }
        let candidate_count = row_count
            .checked_mul(top_k)
            .ok_or_else(|| Error::backend("DFlash2 candidate count overflow"))?;
        require_byte_capacity(
            candidate_ids,
            candidate_count * std::mem::size_of::<u32>(),
            "DFlash2 candidate ids",
        )?;
        require_byte_capacity(
            candidate_logits,
            candidate_count * std::mem::size_of::<f32>(),
            "DFlash2 candidate logits",
        )?;
        require_f16_capacity(
            projected_hidden,
            row_count * SELECTOR_RANK,
            "DFlash2 projected selector hidden states",
        )?;
        require_tensor_range(predecessor_codebook, "DFlash2 predecessor codebook")?;
        require_tensor_range(successor_codebook, "DFlash2 successor codebook")?;
        let selected_ids = self.arena.empty_u32(row_count)?;
        encode_1d_threadgroups_args(
            command_buffer,
            &self.selector,
            &[
                KernelArg::Buffer(candidate_ids),
                KernelArg::Buffer(candidate_logits),
                KernelArg::Buffer(projected_hidden),
                KernelArg::BufferOffset(
                    &predecessor_codebook.buffer,
                    predecessor_codebook.byte_offset,
                ),
                KernelArg::BufferOffset(&successor_codebook.buffer, successor_codebook.byte_offset),
                KernelArg::Buffer(&selected_ids),
                KernelArg::U32(as_u32(row_count, "selector rows")?),
                KernelArg::U32(anchor_token),
            ],
            1,
            256,
        )?;
        Ok(selected_ids)
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_attention(
    cache: &DFlashAttentionCache,
    batch: usize,
    proposal_length: usize,
    context_length: usize,
    context_position_start: usize,
    rope_theta: f32,
    sliding_window: usize,
) -> Result<()> {
    if batch == 0
        || proposal_length == 0
        || proposal_length > MAX_PROPOSAL_ROWS + 1
        || context_length == 0
        || cache.batch != batch
        || cache.kv_heads != KV_HEADS
        || cache.head_dim != HEAD_DIM
        || sliding_window != cache.capacity_tokens + 1
        || !rope_theta.is_finite()
        || rope_theta <= 0.0
    {
        return Err(Error::backend("invalid DFlash2 attention dimensions"));
    }
    if cache.length > 0 && context_position_start != cache.next_position {
        return Err(Error::cache(format!(
            "DFlash2 context starts at {context_position_start}, expected {}",
            cache.next_position
        )));
    }
    Ok(())
}

fn require_tensor_range(tensor: &DeviceQwenBf16Tensor, label: &str) -> Result<()> {
    let byte_len = tensor
        .element_count()?
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| Error::backend(format!("{label} byte size overflow")))?;
    let end = tensor
        .byte_offset
        .checked_add(byte_len)
        .ok_or_else(|| Error::backend(format!("{label} range overflow")))?;
    require_byte_capacity(&tensor.buffer, end, label)
}

fn validate_w4_matrix(matrix: &DeviceDFlashW4Matrix) -> Result<()> {
    if matrix.rows == 0
        || matrix.columns == 0
        || matrix.group_size != W4_GROUP_SIZE
        || !matrix.columns.is_multiple_of(matrix.group_size)
    {
        return Err(Error::backend("invalid DFlash2 W4 matrix dimensions"));
    }
    let expected_packed = matrix
        .rows
        .checked_mul(matrix.columns / 2)
        .ok_or_else(|| Error::backend("DFlash2 W4 packed size overflow"))?;
    let expected_scales = matrix
        .rows
        .checked_mul(matrix.columns / matrix.group_size)
        .and_then(|count| count.checked_mul(std::mem::size_of::<u16>()))
        .ok_or_else(|| Error::backend("DFlash2 W4 scale size overflow"))?;
    if matrix.packed_bytes != expected_packed
        || matrix.scale_bytes != expected_scales
        || matrix.bias_bytes != expected_scales
        || matrix.packed.length() < expected_packed as u64
        || matrix.scales.length() < expected_scales as u64
        || matrix.biases.length() < expected_scales as u64
    {
        return Err(Error::backend("invalid DFlash2 W4 buffer layout"));
    }
    Ok(())
}

fn require_simd_width(pipeline: &ComputePipelineState, label: &str) -> Result<()> {
    if pipeline.thread_execution_width() as usize != SIMD_LANES {
        return Err(Error::backend(format!(
            "{label} requires a {SIMD_LANES}-lane SIMD group"
        )));
    }
    Ok(())
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::backend(format!("DFlash2 {label} exceeds u32")))
}
