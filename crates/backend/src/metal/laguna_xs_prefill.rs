use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
};

use ::metal::{Buffer, CommandBufferRef, ComputePipelineState, Device};
use common::{Error, Result};

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
const TOKEN_TILE: usize = 64;
const OUTPUT_TILE: usize = 32;
const THREAD_COUNT: usize = 128;
const MIN_ROWS: usize = 4;
const KERNEL_SOURCE: &str = include_str!("kernels/laguna_xs_prefill_kernels.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct WeightKey {
    address: usize,
    byte_len: usize,
}

struct WeightBuffer {
    storage: Buffer,
    byte_offset: usize,
}

struct Pipelines {
    q4: ComputePipelineState,
    q6: ComputePipelineState,
    q4_add: ComputePipelineState,
    q6_add: ComputePipelineState,
    q4_add2: ComputePipelineState,
    q6_add2: ComputePipelineState,
}

/// Prompt-only Q4_K/Q6_K matrix multiplication for Laguna XS.
///
/// Decode continues to use `MetalLagunaXs`. This component exists separately
/// so prompt batching can evolve without changing the latency-sensitive
/// single-token kernels.
pub(crate) struct MetalLagunaXsPrefill {
    pipelines: Mutex<Option<Pipelines>>,
    arena: MetalArena,
    weights: Mutex<HashMap<WeightKey, Buffer>>,
    laguna_views: Arc<MetalLagunaViews>,
}

impl MetalLagunaXsPrefill {
    pub(crate) fn new(arena: MetalArena, laguna_views: Arc<MetalLagunaViews>) -> Self {
        Self {
            pipelines: Mutex::new(None),
            arena,
            weights: Mutex::new(HashMap::new()),
            laguna_views,
        }
    }

    pub(crate) fn supports(row_count: usize, out_features: usize) -> bool {
        row_count >= MIN_ROWS && out_features.is_multiple_of(OUTPUT_TILE)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_matvec(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        quant: GgufKQuant,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.encode(
            command_buffer,
            device,
            quant,
            weights,
            input,
            input_len,
            None,
            None,
            row_count,
            in_features,
            out_features,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_matvec_residuals(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        quant: GgufKQuant,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        residual_a: &Buffer,
        residual_b: Option<&Buffer>,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.encode(
            command_buffer,
            device,
            quant,
            weights,
            input,
            input_len,
            Some(residual_a),
            residual_b,
            row_count,
            in_features,
            out_features,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode(
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
        self.validate(
            quant,
            weights,
            input,
            input_len,
            residual_a,
            residual_b,
            row_count,
            in_features,
            out_features,
        )?;

        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Laguna XS prefill output length overflow"))?;
        let output = self.arena.empty_f32(output_len)?;
        let weight = self.weight_buffer(device, weights)?;
        let pipelines = self.pipelines(device)?;
        let pipelines = pipelines.as_ref().ok_or_else(|| {
            Error::backend("Laguna XS prefill Metal pipelines were not initialized")
        })?;
        let residual_count = usize::from(residual_a.is_some()) + usize::from(residual_b.is_some());
        let residual_a = residual_a.unwrap_or(input);
        let residual_b = residual_b.unwrap_or(residual_a);
        let pipeline = match (quant, residual_count) {
            (GgufKQuant::Q4K, 0) => &pipelines.q4,
            (GgufKQuant::Q6K, 0) => &pipelines.q6,
            (GgufKQuant::Q4K, 1) => &pipelines.q4_add,
            (GgufKQuant::Q6K, 1) => &pipelines.q6_add,
            (GgufKQuant::Q4K, 2) => &pipelines.q4_add2,
            (GgufKQuant::Q6K, 2) => &pipelines.q6_add2,
            (_, count) => {
                return Err(Error::backend(format!(
                    "Laguna XS prefill supports at most two residuals, got {count}"
                )))
            }
        };
        let args = [
            KernelArg::BufferOffset(&weight.storage, weight.byte_offset),
            KernelArg::Buffer(input),
            KernelArg::Buffer(residual_a),
            KernelArg::Buffer(residual_b),
            KernelArg::Buffer(&output),
            KernelArg::U32(as_u32(row_count, "row count")?),
            KernelArg::U32(as_u32(in_features, "input width")?),
            KernelArg::U32(as_u32(out_features, "output width")?),
        ];
        let threadgroup_count = row_count
            .div_ceil(TOKEN_TILE)
            .checked_mul(out_features / OUTPUT_TILE)
            .ok_or_else(|| Error::backend("Laguna XS prefill threadgroup count overflow"))?;
        encode_1d_threadgroups_args(
            command_buffer,
            pipeline,
            &args,
            threadgroup_count,
            THREAD_COUNT,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn validate(
        &self,
        quant: GgufKQuant,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        residual_a: Option<&Buffer>,
        residual_b: Option<&Buffer>,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<()> {
        if !Self::supports(row_count, out_features)
            || in_features == 0
            || !in_features.is_multiple_of(BLOCK_VALUES)
        {
            return Err(Error::backend(format!(
                "Laguna XS prefill requires rows >= {MIN_ROWS}, input width divisible by {BLOCK_VALUES}, and output width divisible by {OUTPUT_TILE}; got rows={row_count}, in={in_features}, out={out_features}"
            )));
        }
        let expected_input_len = row_count
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("Laguna XS prefill input length overflow"))?;
        if input_len != expected_input_len {
            return Err(Error::backend(format!(
                "Laguna XS prefill expected {expected_input_len} input values, got {input_len}"
            )));
        }
        require_f32_capacity(input, input_len, "Laguna XS prefill input")?;

        let block_bytes = match quant {
            GgufKQuant::Q4K => Q4_K_BLOCK_BYTES,
            GgufKQuant::Q6K => Q6_K_BLOCK_BYTES,
        };
        let expected_weight_len = out_features
            .checked_mul(in_features / BLOCK_VALUES)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or_else(|| Error::backend("Laguna XS prefill weight length overflow"))?;
        if weights.len() != expected_weight_len {
            return Err(Error::backend(format!(
                "Laguna XS {quant:?} prefill weights must contain {expected_weight_len} bytes, got {}",
                weights.len()
            )));
        }

        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Laguna XS prefill residual length overflow"))?;
        if let Some(residual) = residual_a {
            require_f32_capacity(residual, output_len, "Laguna XS prefill first residual")?;
        }
        if let Some(residual) = residual_b {
            if residual_a.is_none() {
                return Err(Error::backend(
                    "Laguna XS prefill second residual requires a first residual",
                ));
            }
            require_f32_capacity(residual, output_len, "Laguna XS prefill second residual")?;
        }
        Ok(())
    }

    fn pipelines(&self, device: &Device) -> Result<MutexGuard<'_, Option<Pipelines>>> {
        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| Error::backend("Laguna XS prefill pipeline lock poisoned"))?;
        if pipelines.is_none() {
            let library = MetalLibrary::compile_source(device, KERNEL_SOURCE)?;
            *pipelines = Some(Pipelines {
                q4: compute_pipeline(device, &library, "laguna_xs_q4_prefill_mma_f32_kernel")?,
                q6: compute_pipeline(device, &library, "laguna_xs_q6_prefill_mma_f32_kernel")?,
                q4_add: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_q4_prefill_mma_add_f32_kernel",
                )?,
                q6_add: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_q6_prefill_mma_add_f32_kernel",
                )?,
                q4_add2: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_q4_prefill_mma_add2_f32_kernel",
                )?,
                q6_add2: compute_pipeline(
                    device,
                    &library,
                    "laguna_xs_q6_prefill_mma_add2_f32_kernel",
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

        let key = WeightKey {
            address: bytes.as_ptr() as usize,
            byte_len: bytes.len(),
        };
        let mut weights = self
            .weights
            .lock()
            .map_err(|_| Error::backend("Laguna XS prefill weight cache lock poisoned"))?;
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

fn as_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::backend(format!("Laguna XS prefill {label} exceeds Metal u32")))
}
