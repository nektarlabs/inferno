use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
};

use ::metal::{
    foreign_types::{ForeignType, ForeignTypeRef},
    Buffer, CommandBufferRef, ComputePipelineState, Device, NSUInteger,
};
use common::{Error, Result};
use objc::{
    msg_send,
    rc::StrongPtr,
    runtime::{Class, Object, NO, YES},
    sel, sel_impl,
};

use crate::GgufKQuant;

use super::{
    arena::MetalArena,
    buffers::{require_f32_capacity, u8_buffer_no_copy},
    command::{encode_1d_threadgroups_args, KernelArg},
    laguna_views::MetalLagunaViews,
    library::MetalLibrary,
    pipeline::compute_pipeline,
};

const BLOCK_VALUES: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q6_K_BLOCK_BYTES: usize = 210;
const VECTOR_WIDTH: usize = 4;
const THREAD_COUNT: usize = 256;
const MIN_ROWS: usize = 64;
const MPS_DATA_TYPE_F16: u32 = 0x1000_0010;
const KERNEL_SOURCE: &str = include_str!("kernels/laguna_xs_mps_prefill_kernels.metal");

#[link(name = "MetalPerformanceShaders", kind = "framework")]
unsafe extern "C" {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct WeightKey {
    address: usize,
    byte_len: usize,
    quant_code: u8,
    in_features: usize,
    out_features: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct GemmKey {
    rows: usize,
    inner: usize,
    columns: usize,
}

struct WeightBuffer {
    storage: Buffer,
    byte_offset: usize,
}

struct MpsKernel(StrongPtr);

// MPS kernels are immutable after initialization and Apple documents them as
// reusable across command buffers. StrongPtr does not encode that contract.
unsafe impl Send for MpsKernel {}
unsafe impl Sync for MpsKernel {}

struct Pipelines {
    dequant_q4: ComputePipelineState,
    dequant_q6: ComputePipelineState,
    cast_input: ComputePipelineState,
    cast_output: ComputePipelineState,
    cast_output_add: ComputePipelineState,
    cast_output_add2: ComputePipelineState,
}

/// Dense-only Laguna XS prompt GEMM using pre-expanded F16 weights and MPS.
///
/// Routed experts remain quantized because expanding every expert would exceed
/// the target machine's unified-memory budget. Decode also remains on the
/// dedicated low-latency quantized kernels.
pub(crate) struct MetalLagunaXsMpsPrefill {
    pipelines: Mutex<Option<Pipelines>>,
    arena: MetalArena,
    weights: Mutex<HashMap<WeightKey, Buffer>>,
    gemms: Mutex<HashMap<GemmKey, MpsKernel>>,
    laguna_views: Arc<MetalLagunaViews>,
}

impl MetalLagunaXsMpsPrefill {
    pub(crate) fn new(arena: MetalArena, laguna_views: Arc<MetalLagunaViews>) -> Self {
        Self {
            pipelines: Mutex::new(None),
            arena,
            weights: Mutex::new(HashMap::new()),
            gemms: Mutex::new(HashMap::new()),
            laguna_views,
        }
    }

    pub(crate) const fn supports(row_count: usize) -> bool {
        row_count >= MIN_ROWS
    }

    pub(crate) fn is_prepared(
        &self,
        quant: GgufKQuant,
        weights: &[u8],
        in_features: usize,
        out_features: usize,
    ) -> Result<bool> {
        let key = weight_key(quant, weights, in_features, out_features);
        self.weights
            .lock()
            .map(|weights| weights.contains_key(&key))
            .map_err(|_| Error::backend("Laguna XS MPS weight cache lock poisoned"))
    }

    pub(crate) fn encode_prepare_weight(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        quant: GgufKQuant,
        weights: &[u8],
        in_features: usize,
        out_features: usize,
    ) -> Result<()> {
        validate_dimensions(quant, weights, in_features, out_features)?;
        let key = weight_key(quant, weights, in_features, out_features);
        let mut cache = self
            .weights
            .lock()
            .map_err(|_| Error::backend("Laguna XS MPS weight cache lock poisoned"))?;
        if cache.contains_key(&key) {
            return Ok(());
        }

        let value_count = in_features
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Laguna XS MPS weight value count overflow"))?;
        let output = self.arena.empty_f16(value_count)?;
        let source = self.weight_buffer(device, weights)?;
        let pipelines = self.pipelines(device)?;
        let pipelines = pipelines
            .as_ref()
            .ok_or_else(|| Error::backend("Laguna XS MPS pipelines were not initialized"))?;
        let pipeline = match quant {
            GgufKQuant::Q4K => &pipelines.dequant_q4,
            GgufKQuant::Q6K => &pipelines.dequant_q6,
        };
        encode_1d_threadgroups_args(
            command_buffer,
            pipeline,
            &[
                KernelArg::BufferOffset(&source.storage, source.byte_offset),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(in_features, "input width")?),
                KernelArg::U32(as_u32(value_count, "weight value count")?),
            ],
            (value_count / VECTOR_WIDTH).div_ceil(THREAD_COUNT),
            THREAD_COUNT,
        )?;
        cache.insert(key, output);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        quant: GgufKQuant,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        residual_a: Option<&Buffer>,
        residual_b: Option<&Buffer>,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        if !Self::supports(row_count) {
            return Err(Error::backend(format!(
                "Laguna XS MPS prefill requires at least {MIN_ROWS} rows, got {row_count}"
            )));
        }
        validate_dimensions(quant, weights, in_features, out_features)?;
        let expected_input_len = row_count
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("Laguna XS MPS input length overflow"))?;
        if input_len != expected_input_len {
            return Err(Error::backend(format!(
                "Laguna XS MPS expected {expected_input_len} input values, got {input_len}"
            )));
        }
        require_f32_capacity(input, input_len, "Laguna XS MPS input")?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Laguna XS MPS output length overflow"))?;
        if let Some(residual) = residual_a {
            require_f32_capacity(residual, output_len, "Laguna XS MPS first residual")?;
        }
        if let Some(residual) = residual_b {
            if residual_a.is_none() {
                return Err(Error::backend(
                    "Laguna XS MPS second residual requires the first residual",
                ));
            }
            require_f32_capacity(residual, output_len, "Laguna XS MPS second residual")?;
        }

        let key = weight_key(quant, weights, in_features, out_features);
        let prepared_weight = self
            .weights
            .lock()
            .map_err(|_| Error::backend("Laguna XS MPS weight cache lock poisoned"))?
            .get(&key)
            .cloned()
            .ok_or_else(|| {
                Error::backend("Laguna XS MPS dense weight was not prepared during model loading")
            })?;
        let input_f16 = self.arena.empty_f16(input_len)?;
        let output_f16 = self.arena.empty_f16(output_len)?;
        let output = self.arena.empty_f32(output_len)?;
        let pipelines = self.pipelines(device)?;
        let pipelines = pipelines
            .as_ref()
            .ok_or_else(|| Error::backend("Laguna XS MPS pipelines were not initialized"))?;

        encode_1d_threadgroups_args(
            command_buffer,
            &pipelines.cast_input,
            &[
                KernelArg::Buffer(input),
                KernelArg::Buffer(&input_f16),
                KernelArg::U32(as_u32(input_len, "input value count")?),
            ],
            (input_len / VECTOR_WIDTH).div_ceil(THREAD_COUNT),
            THREAD_COUNT,
        )?;
        self.encode_mps_gemm(
            command_buffer,
            device,
            &input_f16,
            &prepared_weight,
            &output_f16,
            row_count,
            in_features,
            out_features,
        )?;

        let residual_count = usize::from(residual_a.is_some()) + usize::from(residual_b.is_some());
        let cast_pipeline = match residual_count {
            0 => &pipelines.cast_output,
            1 => &pipelines.cast_output_add,
            2 => &pipelines.cast_output_add2,
            count => {
                return Err(Error::backend(format!(
                    "Laguna XS MPS supports at most two residuals, got {count}"
                )))
            }
        };
        let residual_a = residual_a.unwrap_or(&output);
        let residual_b = residual_b.unwrap_or(residual_a);
        encode_1d_threadgroups_args(
            command_buffer,
            cast_pipeline,
            &[
                KernelArg::Buffer(&output_f16),
                KernelArg::Buffer(residual_a),
                KernelArg::Buffer(residual_b),
                KernelArg::Buffer(&output),
                KernelArg::U32(as_u32(output_len, "output value count")?),
            ],
            (output_len / VECTOR_WIDTH).div_ceil(THREAD_COUNT),
            THREAD_COUNT,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_mps_gemm(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        input: &Buffer,
        weight: &Buffer,
        output: &Buffer,
        rows: usize,
        inner: usize,
        columns: usize,
    ) -> Result<()> {
        let key = GemmKey {
            rows,
            inner,
            columns,
        };
        let mut gemms = self
            .gemms
            .lock()
            .map_err(|_| Error::backend("Laguna XS MPS GEMM cache lock poisoned"))?;
        if !gemms.contains_key(&key) {
            gemms.insert(key, create_gemm(device, key)?);
        }
        let gemm = gemms
            .get(&key)
            .ok_or_else(|| Error::backend("Laguna XS MPS GEMM cache insertion failed"))?;
        let left = create_matrix(input, 0, rows, inner)?;
        let right = create_matrix(weight, 0, columns, inner)?;
        let result = create_matrix(output, 0, rows, columns)?;
        let command_buffer = command_buffer.as_ptr().cast::<Object>();
        // SAFETY: all objects wrap live Metal buffers on the same device. Matrix
        // dimensions match the immutable GEMM descriptor stored in `gemm`.
        unsafe {
            let _: () = msg_send![
                *gemm.0,
                encodeToCommandBuffer: command_buffer
                leftMatrix: *left
                rightMatrix: *right
                resultMatrix: *result
            ];
        }
        Ok(())
    }

    fn pipelines(&self, device: &Device) -> Result<MutexGuard<'_, Option<Pipelines>>> {
        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| Error::backend("Laguna XS MPS pipeline lock poisoned"))?;
        if pipelines.is_none() {
            let library = MetalLibrary::compile_source(device, KERNEL_SOURCE)?;
            *pipelines = Some(Pipelines {
                dequant_q4: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_mps_dequant_q4_f16_kernel",
                )?,
                dequant_q6: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_mps_dequant_q6_f16_kernel",
                )?,
                cast_input: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_mps_cast_f32_f16_kernel",
                )?,
                cast_output: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_mps_cast_f16_f32_kernel",
                )?,
                cast_output_add: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_mps_cast_f16_f32_add_kernel",
                )?,
                cast_output_add2: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_mps_cast_f16_f32_add2_kernel",
                )?,
            });
        }
        Ok(pipelines)
    }

    fn weight_buffer(&self, device: &Device, bytes: &[u8]) -> Result<WeightBuffer> {
        if let Some(binding) = self.laguna_views.binding(bytes)? {
            return Ok(WeightBuffer {
                storage: binding.buffer,
                byte_offset: binding.byte_offset,
            });
        }
        Ok(WeightBuffer {
            storage: u8_buffer_no_copy(device, bytes)?,
            byte_offset: 0,
        })
    }
}

fn create_gemm(device: &Device, key: GemmKey) -> Result<MpsKernel> {
    let class = Class::get("MPSMatrixMultiplication")
        .ok_or_else(|| Error::backend("MPSMatrixMultiplication is unavailable"))?;
    let allocation: *mut Object = unsafe { msg_send![class, alloc] };
    if allocation.is_null() {
        return Err(Error::backend("failed to allocate MPSMatrixMultiplication"));
    }
    let device = device.as_ptr().cast::<Object>();
    // SAFETY: `allocation` is an MPSMatrixMultiplication allocation and all
    // dimensions are validated nonzero before this function is called.
    let object: *mut Object = unsafe {
        msg_send![
            allocation,
            initWithDevice: device
            transposeLeft: NO
            transposeRight: YES
            resultRows: key.rows as NSUInteger
            resultColumns: key.columns as NSUInteger
            interiorColumns: key.inner as NSUInteger
            alpha: 1.0_f64
            beta: 0.0_f64
        ]
    };
    if object.is_null() {
        return Err(Error::backend(
            "failed to initialize MPSMatrixMultiplication",
        ));
    }
    Ok(MpsKernel(unsafe { StrongPtr::new(object) }))
}

fn create_matrix(
    buffer: &Buffer,
    byte_offset: usize,
    rows: usize,
    columns: usize,
) -> Result<StrongPtr> {
    let descriptor_class = Class::get("MPSMatrixDescriptor")
        .ok_or_else(|| Error::backend("MPSMatrixDescriptor is unavailable"))?;
    let row_bytes = columns
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| Error::backend("Laguna XS MPS row stride overflow"))?;
    // SAFETY: this class method returns an autoreleased descriptor. Retaining it
    // keeps it alive through matrix initialization.
    let descriptor_pointer: *mut Object = unsafe {
        msg_send![
            descriptor_class,
            matrixDescriptorWithRows: rows as NSUInteger
            columns: columns as NSUInteger
            rowBytes: row_bytes as NSUInteger
            dataType: MPS_DATA_TYPE_F16
        ]
    };
    if descriptor_pointer.is_null() {
        return Err(Error::backend("failed to create MPSMatrixDescriptor"));
    }
    let descriptor = unsafe { StrongPtr::retain(descriptor_pointer) };
    let matrix_class =
        Class::get("MPSMatrix").ok_or_else(|| Error::backend("MPSMatrix is unavailable"))?;
    let allocation: *mut Object = unsafe { msg_send![matrix_class, alloc] };
    if allocation.is_null() {
        return Err(Error::backend("failed to allocate MPSMatrix"));
    }
    let buffer = buffer.as_ptr().cast::<Object>();
    let matrix: *mut Object = unsafe {
        msg_send![
            allocation,
            initWithBuffer: buffer
            offset: byte_offset as NSUInteger
            descriptor: *descriptor
        ]
    };
    if matrix.is_null() {
        return Err(Error::backend("failed to initialize MPSMatrix"));
    }
    Ok(unsafe { StrongPtr::new(matrix) })
}

fn validate_dimensions(
    quant: GgufKQuant,
    weights: &[u8],
    in_features: usize,
    out_features: usize,
) -> Result<()> {
    if in_features == 0
        || out_features == 0
        || !in_features.is_multiple_of(BLOCK_VALUES)
        || !in_features.is_multiple_of(VECTOR_WIDTH)
        || !out_features.is_multiple_of(VECTOR_WIDTH)
    {
        return Err(Error::backend(format!(
            "Laguna XS MPS dimensions must be nonzero and vector/block aligned, got in={in_features}, out={out_features}"
        )));
    }
    let block_bytes = match quant {
        GgufKQuant::Q4K => Q4_K_BLOCK_BYTES,
        GgufKQuant::Q6K => Q6_K_BLOCK_BYTES,
    };
    let expected = out_features
        .checked_mul(in_features / BLOCK_VALUES)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(|| Error::backend("Laguna XS MPS weight size overflow"))?;
    if weights.len() != expected {
        return Err(Error::backend(format!(
            "Laguna XS MPS {quant:?} weights require {expected} bytes, got {}",
            weights.len()
        )));
    }
    Ok(())
}

fn weight_key(
    quant: GgufKQuant,
    weights: &[u8],
    in_features: usize,
    out_features: usize,
) -> WeightKey {
    WeightKey {
        address: weights.as_ptr() as usize,
        byte_len: weights.len(),
        quant_code: match quant {
            GgufKQuant::Q4K => 4,
            GgufKQuant::Q6K => 6,
        },
        in_features,
        out_features,
    }
}

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::backend(format!("Laguna XS MPS {label} exceeds Metal u32")))
}
