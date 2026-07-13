use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs::File,
    io::{self, ErrorKind},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::Instant,
};

use ::metal::{
    Buffer, CommandBuffer, CommandBufferRef, CommandQueue, ComputePipelineState, Device,
    MTLCommandBufferStatus,
};
use common::{Error, Result};
use objc::{msg_send, sel, sel_impl};
use tracing::{debug, trace, warn};

use crate::{ExpertCacheMetrics, Q2ExpertSource};

use super::{
    buffers::{
        empty_f32_buffer, empty_u32_buffer, empty_u8_buffer, read_f32_buffer, read_u32_buffer,
        require_byte_capacity, require_f32_capacity, u32_buffer, u32_scalar_buffer, u64_buffer,
        u8_buffer_no_copy, write_f32_buffer, write_u32_buffer,
    },
    command::{
        dispatch_1d, dispatch_1d_many, encode_1d, encode_1d_threadgroups,
        encode_1d_with_indirect_reads, Dispatch1d,
    },
    library::MetalLibrary,
    pipeline::compute_pipeline,
    validation::{
        debug_assert_finite_values, validate_q2_k_gate_up_swiglu_f32, validate_q2_k_matvec_buffer,
        validate_q2_k_matvec_f32, validate_q2_k_transposed_matvec_buffer,
        validate_q2_k_transposed_matvec_f32, validate_q8_0_matvec_buffer, validate_q8_0_matvec_f32,
        validate_q8_0_transposed_matvec_buffer, validate_q8_0_transposed_matvec_f32,
        Q2_K_BLOCK_BYTES, Q2_K_BLOCK_VALUES, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_VALUES,
    },
};

/// Which quantized matvec kernel family a batched encode call targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QuantMatvecKind {
    Q2K,
    Q2KTransposed,
    Q80,
    Q80Transposed,
}

const Q2_K_MATVEC_KERNEL: &str = "q2_k_matvec_f32_kernel";
const Q2_K_MATVEC_ADD_KERNEL: &str = "q2_k_matvec_add_f32_kernel";
const Q2_K_GATE_UP_SWIGLU_KERNEL: &str = "q2_k_gate_up_swiglu_f32_kernel";
const Q2_K_MULTI_EXPERT_GATE_UP_SWIGLU_KERNEL: &str = "q2_k_multi_expert_gate_up_swiglu_f32_kernel";
const Q2_K_MULTI_EXPERT_MATVEC_KERNEL: &str = "q2_k_multi_expert_matvec_f32_kernel";
const Q2_K_READY_GATE_UP_SWIGLU_KERNEL: &str = "q2_k_ready_gate_up_swiglu_f32_kernel";
const Q2_K_READY_MATVEC_KERNEL: &str = "q2_k_ready_matvec_f32_kernel";
const Q2_K_READY_SLOT_GATE_UP_SWIGLU_KERNEL: &str = "q2_k_ready_slot_gate_up_swiglu_f32_kernel";
const Q2_K_READY_SLOT_MATVEC_KERNEL: &str = "q2_k_ready_slot_matvec_f32_kernel";
const Q2_K_TRANSPOSED_MATVEC_KERNEL: &str = "q2_k_transposed_matvec_f32_kernel";
const Q2_K_PACKED_HEADS_TRANSPOSED_MATVEC_KERNEL: &str =
    "q2_k_packed_heads_transposed_matvec_f32_kernel";
const Q8_0_MATVEC_KERNEL: &str = "q8_0_matvec_f32_kernel";
const Q8_0_MATVEC_TILED_KERNEL: &str = "q8_0_matvec_tiled_f32_kernel";
const Q8_0_MATVEC_ADD_TILED_KERNEL: &str = "q8_0_matvec_add_tiled_f32_kernel";
const Q8_0_TRANSPOSED_MATVEC_KERNEL: &str = "q8_0_transposed_matvec_f32_kernel";
const Q8_0_PACKED_HEADS_TRANSPOSED_MATVEC_KERNEL: &str =
    "q8_0_packed_heads_transposed_matvec_f32_kernel";
const Q8_0_PACKED_HEADS_MATVEC_KERNEL: &str = "q8_0_packed_heads_matvec_f32_kernel";
const ARGMAX_F32_KERNEL: &str = "argmax_f32_kernel";
const ARGMAX_ROWS_F32_KERNEL: &str = "argmax_rows_f32_kernel";
const Q2_K_SIMD_LANES: usize = 32;
const Q8_0_MAX_SIMDGROUPS_PER_OUTPUT: usize = 8;
const ARGMAX_THREADS_PER_VECTOR: usize = 256;
// Sixteen Q2 expert triplets per routed layer use about 14.9 GB for the main
// 75-layer model. On the 64 GB target this measured faster than 10 slots;
// 21 slots increased macOS memory pressure enough to reduce throughput.
const ROUTED_EXPERT_CACHE_SLOTS_PER_LAYER: usize = 16;
const ROUTED_EXPERT_READ_WORKERS: usize = 8;
static EXPERT_CACHE_LOCK_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);

pub(crate) struct MetalQ2Matvec {
    pipeline: ComputePipelineState,
    add_pipeline: ComputePipelineState,
    gate_up_swiglu_pipeline: ComputePipelineState,
    multi_expert_gate_up_swiglu_pipeline: ComputePipelineState,
    multi_expert_matvec_pipeline: ComputePipelineState,
    ready_gate_up_swiglu_pipeline: ComputePipelineState,
    ready_matvec_pipeline: ComputePipelineState,
    ready_slot_gate_up_swiglu_pipeline: ComputePipelineState,
    ready_slot_matvec_pipeline: ComputePipelineState,
    transposed_pipeline: ComputePipelineState,
    packed_heads_transposed_pipeline: ComputePipelineState,
    q8_0_pipeline: ComputePipelineState,
    q8_0_tiled_pipeline: ComputePipelineState,
    q8_0_tiled_add_pipeline: ComputePipelineState,
    q8_0_transposed_pipeline: ComputePipelineState,
    q8_0_packed_heads_transposed_pipeline: ComputePipelineState,
    q8_0_packed_heads_pipeline: ComputePipelineState,
    argmax_pipeline: ComputePipelineState,
    argmax_rows_pipeline: ComputePipelineState,
    scratch: Mutex<Q2ScratchBuffers>,
    ready_expert_cache: Mutex<Q2PerLayerExpertCache>,
    expert_cache_counters: Q2ExpertCacheCounters,
    expert_queue: CommandQueue,
    weight_buffers: Mutex<HashMap<WeightBufferKey, Buffer>>,
}

#[derive(Debug, Default)]
struct Q2ExpertCacheCounters {
    lookups: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    ssd_read_bytes: AtomicU64,
    transient_experts: AtomicU64,
    ready_waves: AtomicU64,
    lookup_nanoseconds: AtomicU64,
    ssd_load_nanoseconds: AtomicU64,
    q2_matmul_gpu_nanoseconds: AtomicU64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalQ2MatvecReport {
    pub values: Vec<f32>,
    pub row_count: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub blocks_per_row: usize,
    pub input_len: usize,
    pub weight_bytes: usize,
    pub thread_count: usize,
    pub transposed: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalQ2MatvecAddReport {
    pub values: Vec<f32>,
    pub row_count: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub blocks_per_row: usize,
    pub input_len: usize,
    pub residual_len: usize,
    pub weight_bytes: usize,
    pub thread_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalQ2MatvecArgmaxReport {
    pub token_id: u32,
    pub token_score: f32,
    pub row_count: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub blocks_per_row: usize,
    pub input_len: usize,
    pub weight_bytes: usize,
    pub matvec_thread_count: usize,
    pub argmax_thread_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetalQ2GateUpSwiGluReport {
    pub values: Vec<f32>,
    pub row_count: usize,
    pub in_features: usize,
    pub out_features: usize,
    pub blocks_per_row: usize,
    pub input_len: usize,
    pub gate_weight_bytes: usize,
    pub up_weight_bytes: usize,
    pub thread_count: usize,
}

#[derive(Debug)]
struct WeightBuffer {
    buffer: Buffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct WeightBufferKey {
    address: usize,
    byte_len: usize,
}

#[derive(Debug)]
struct ScratchBuffer {
    buffer: Buffer,
    len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ExpertCacheKey {
    gate_address: usize,
    gate_bytes: usize,
    up_address: usize,
    up_bytes: usize,
    down_address: usize,
    down_bytes: usize,
}

#[derive(Debug)]
struct Q2ExpertCacheLayout {
    gate_stride: usize,
    up_stride: usize,
    down_stride: usize,
}

#[derive(Debug)]
struct Q2ExpertSlotBuffers {
    gate: LockedMetalBuffer,
    up: LockedMetalBuffer,
    down: LockedMetalBuffer,
}

#[derive(Debug)]
struct Q2ExpertLayerBuffers {
    gate: LockedMetalSlab,
    up: LockedMetalSlab,
    down: LockedMetalSlab,
}

#[derive(Debug)]
struct LockedMetalBuffer {
    buffer: Buffer,
    locked: bool,
}

#[derive(Debug)]
struct LockedMetalSlab {
    buffer: Buffer,
    locked_ranges: Vec<(usize, usize)>,
}

#[derive(Debug)]
struct ExpertModelFile {
    path: PathBuf,
    file: File,
}

#[derive(Debug)]
struct ExpertReadTask {
    buffer: Buffer,
    destination_offset: usize,
    absolute_offset: u64,
    byte_len: usize,
}

#[derive(Debug, Clone)]
struct ReadyExpertBuffers {
    gate: Buffer,
    up: Buffer,
    down: Buffer,
    gate_offset: usize,
    up_offset: usize,
    down_offset: usize,
    slot_index: Option<usize>,
}

#[derive(Debug)]
struct ReadyExpertGroup {
    assignment_indices: Vec<usize>,
    buffers: ReadyExpertBuffers,
    read_tasks: Vec<ExpertReadTask>,
    cache_hit: bool,
    cached_miss_key: Option<ExpertCacheKey>,
    transient: bool,
    _transient_owner: Option<Q2ExpertSlotBuffers>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReadyExpertPhase {
    GateUp,
    Down,
}

#[derive(Debug)]
struct ReadyExpertSeed<'a> {
    key: ExpertCacheKey,
    assignment_indices: Vec<usize>,
    gate: Q2ExpertSource<'a>,
    up: Q2ExpertSource<'a>,
    down: Q2ExpertSource<'a>,
}

#[derive(Debug)]
struct Q2ExpertLayerCache {
    slots: Vec<bool>,
    storage: Option<Q2ExpertLayerBuffers>,
    entries: HashMap<ExpertCacheKey, usize>,
    probation: VecDeque<ExpertCacheKey>,
    protected: VecDeque<ExpertCacheKey>,
    protected_capacity: usize,
    free_slots: Vec<usize>,
}

#[derive(Debug)]
struct Q2PerLayerExpertCache {
    layout: Option<Q2ExpertCacheLayout>,
    layers: HashMap<usize, Arc<Mutex<Q2ExpertLayerCache>>>,
    model_file: Option<ExpertModelFile>,
    slots_per_layer: usize,
}

impl Default for Q2PerLayerExpertCache {
    fn default() -> Self {
        Self {
            layout: None,
            layers: HashMap::new(),
            model_file: None,
            slots_per_layer: ROUTED_EXPERT_CACHE_SLOTS_PER_LAYER,
        }
    }
}

#[derive(Debug)]
enum LayerSlotResolution {
    Hit(usize),
    Miss(usize),
    Transient,
}

#[derive(Debug)]
struct SubmittedReadyWave {
    command_buffer: CommandBuffer,
    assignment_count: usize,
}

#[derive(Debug)]
pub(crate) struct ReadyRoutedExperts {
    pub(crate) output: Buffer,
    pub(crate) selected_experts: usize,
    pub(crate) cache_hits: usize,
    pub(crate) cache_misses: usize,
    pub(crate) transient_experts: usize,
    pub(crate) read_bytes: u64,
    pub(crate) ready_waves: usize,
}

#[derive(Debug, Default)]
struct Q2ScratchBuffers {
    input: Option<ScratchBuffer>,
    residual: Option<ScratchBuffer>,
    output: Option<ScratchBuffer>,
    logits: Option<ScratchBuffer>,
    token_id: Option<ScratchBuffer>,
    token_score: Option<ScratchBuffer>,
    row_count: Option<ScratchBuffer>,
    in_features: Option<ScratchBuffer>,
    out_features: Option<ScratchBuffer>,
    blocks_per_row: Option<ScratchBuffer>,
    output_len: Option<ScratchBuffer>,
}

#[derive(Debug)]
struct ScratchBufferRef {
    buffer: Buffer,
    reused: bool,
}

impl MetalQ2Matvec {
    pub(crate) fn new(device: &Device, library: &MetalLibrary) -> Result<Self> {
        Ok(Self {
            pipeline: compute_pipeline(device, library, Q2_K_MATVEC_KERNEL)?,
            add_pipeline: compute_pipeline(device, library, Q2_K_MATVEC_ADD_KERNEL)?,
            gate_up_swiglu_pipeline: compute_pipeline(device, library, Q2_K_GATE_UP_SWIGLU_KERNEL)?,
            multi_expert_gate_up_swiglu_pipeline: compute_pipeline(
                device,
                library,
                Q2_K_MULTI_EXPERT_GATE_UP_SWIGLU_KERNEL,
            )?,
            multi_expert_matvec_pipeline: compute_pipeline(
                device,
                library,
                Q2_K_MULTI_EXPERT_MATVEC_KERNEL,
            )?,
            ready_gate_up_swiglu_pipeline: compute_pipeline(
                device,
                library,
                Q2_K_READY_GATE_UP_SWIGLU_KERNEL,
            )?,
            ready_matvec_pipeline: compute_pipeline(device, library, Q2_K_READY_MATVEC_KERNEL)?,
            ready_slot_gate_up_swiglu_pipeline: compute_pipeline(
                device,
                library,
                Q2_K_READY_SLOT_GATE_UP_SWIGLU_KERNEL,
            )?,
            ready_slot_matvec_pipeline: compute_pipeline(
                device,
                library,
                Q2_K_READY_SLOT_MATVEC_KERNEL,
            )?,
            transposed_pipeline: compute_pipeline(device, library, Q2_K_TRANSPOSED_MATVEC_KERNEL)?,
            packed_heads_transposed_pipeline: compute_pipeline(
                device,
                library,
                Q2_K_PACKED_HEADS_TRANSPOSED_MATVEC_KERNEL,
            )?,
            q8_0_pipeline: compute_pipeline(device, library, Q8_0_MATVEC_KERNEL)?,
            q8_0_tiled_pipeline: compute_pipeline(device, library, Q8_0_MATVEC_TILED_KERNEL)?,
            q8_0_tiled_add_pipeline: compute_pipeline(
                device,
                library,
                Q8_0_MATVEC_ADD_TILED_KERNEL,
            )?,
            q8_0_transposed_pipeline: compute_pipeline(
                device,
                library,
                Q8_0_TRANSPOSED_MATVEC_KERNEL,
            )?,
            q8_0_packed_heads_transposed_pipeline: compute_pipeline(
                device,
                library,
                Q8_0_PACKED_HEADS_TRANSPOSED_MATVEC_KERNEL,
            )?,
            q8_0_packed_heads_pipeline: compute_pipeline(
                device,
                library,
                Q8_0_PACKED_HEADS_MATVEC_KERNEL,
            )?,
            argmax_pipeline: compute_pipeline(device, library, ARGMAX_F32_KERNEL)?,
            argmax_rows_pipeline: compute_pipeline(device, library, ARGMAX_ROWS_F32_KERNEL)?,
            scratch: Mutex::new(Q2ScratchBuffers::default()),
            ready_expert_cache: Mutex::new(Q2PerLayerExpertCache::default()),
            expert_cache_counters: Q2ExpertCacheCounters::default(),
            expert_queue: device.new_command_queue(),
            weight_buffers: Mutex::new(HashMap::new()),
        })
    }

    pub(crate) fn expert_cache_metrics(&self) -> Result<ExpertCacheMetrics> {
        let cache = self
            .ready_expert_cache
            .lock()
            .map_err(|_| Error::backend("Q2 per-layer expert cache lock poisoned"))?;
        let bytes_per_expert = cache
            .layout
            .as_ref()
            .map(|layout| {
                layout
                    .gate_stride
                    .checked_add(layout.up_stride)
                    .and_then(|bytes| bytes.checked_add(layout.down_stride))
                    .ok_or_else(|| Error::backend("Q2 expert cache byte size overflow"))
            })
            .transpose()?
            .unwrap_or(0);

        let mut resident_experts = 0_usize;
        let mut allocated_slots = 0_usize;
        let mut capacity_slots = 0_usize;
        for layer in cache.layers.values() {
            let layer = layer
                .lock()
                .map_err(|_| Error::backend("Q2 layer expert cache lock poisoned"))?;
            resident_experts = resident_experts
                .checked_add(layer.entries.len())
                .ok_or_else(|| Error::backend("Q2 resident expert count overflow"))?;
            allocated_slots = allocated_slots
                .checked_add(layer.slots.iter().filter(|allocated| **allocated).count())
                .ok_or_else(|| Error::backend("Q2 allocated expert slot count overflow"))?;
            capacity_slots = capacity_slots
                .checked_add(layer.slots.len())
                .ok_or_else(|| Error::backend("Q2 expert cache capacity count overflow"))?;
        }

        let allocated_bytes = allocated_slots
            .checked_mul(bytes_per_expert)
            .ok_or_else(|| Error::backend("Q2 allocated expert cache bytes overflow"))?;
        let capacity_bytes = capacity_slots
            .checked_mul(bytes_per_expert)
            .ok_or_else(|| Error::backend("Q2 expert cache capacity bytes overflow"))?;

        Ok(ExpertCacheMetrics {
            lookups: self.expert_cache_counters.lookups.load(Ordering::Relaxed),
            hits: self.expert_cache_counters.hits.load(Ordering::Relaxed),
            misses: self.expert_cache_counters.misses.load(Ordering::Relaxed),
            ssd_read_bytes: self
                .expert_cache_counters
                .ssd_read_bytes
                .load(Ordering::Relaxed),
            transient_experts: self
                .expert_cache_counters
                .transient_experts
                .load(Ordering::Relaxed),
            ready_waves: self
                .expert_cache_counters
                .ready_waves
                .load(Ordering::Relaxed),
            resident_experts: u64::try_from(resident_experts)
                .map_err(|_| Error::backend("Q2 resident expert count does not fit u64"))?,
            allocated_slots: u64::try_from(allocated_slots)
                .map_err(|_| Error::backend("Q2 allocated expert slots do not fit u64"))?,
            capacity_slots: u64::try_from(capacity_slots)
                .map_err(|_| Error::backend("Q2 expert cache capacity does not fit u64"))?,
            bytes_per_expert: u64::try_from(bytes_per_expert)
                .map_err(|_| Error::backend("Q2 expert byte size does not fit u64"))?,
            allocated_bytes: u64::try_from(allocated_bytes)
                .map_err(|_| Error::backend("Q2 allocated expert bytes do not fit u64"))?,
            capacity_bytes: u64::try_from(capacity_bytes)
                .map_err(|_| Error::backend("Q2 expert cache bytes do not fit u64"))?,
            lookup_nanoseconds: self
                .expert_cache_counters
                .lookup_nanoseconds
                .load(Ordering::Relaxed),
            ssd_load_nanoseconds: self
                .expert_cache_counters
                .ssd_load_nanoseconds
                .load(Ordering::Relaxed),
            q2_matmul_gpu_nanoseconds: self
                .expert_cache_counters
                .q2_matmul_gpu_nanoseconds
                .load(Ordering::Relaxed),
        })
    }

    pub(crate) fn configure_expert_cache_slots_per_layer(
        &self,
        slots_per_layer: usize,
    ) -> Result<()> {
        if slots_per_layer == 0 {
            return Err(Error::backend(
                "Q2 expert cache requires at least one slot per routed layer",
            ));
        }
        let mut cache = self
            .ready_expert_cache
            .lock()
            .map_err(|_| Error::backend("Q2 per-layer expert cache lock poisoned"))?;
        if !cache.layers.is_empty() {
            return Err(Error::backend(
                "Q2 expert cache capacity must be configured before generation starts",
            ));
        }
        cache.slots_per_layer = slots_per_layer;
        Ok(())
    }

    pub(crate) fn resize_expert_cache_slots_per_layer(&self, slots_per_layer: usize) -> Result<()> {
        if slots_per_layer == 0 {
            return Err(Error::backend(
                "Q2 expert cache requires at least one slot per routed layer",
            ));
        }
        let mut cache = self
            .ready_expert_cache
            .lock()
            .map_err(|_| Error::backend("Q2 per-layer expert cache lock poisoned"))?;
        cache.resize(slots_per_layer)
    }

    pub(crate) fn run(
        &self,
        device: &Device,
        queue: &CommandQueue,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<MetalQ2MatvecReport> {
        let blocks_per_row =
            validate_q2_k_matvec_f32(weights, input, row_count, in_features, out_features)?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K matvec output length overflow"))?;
        let physical_threads = q2_k_cooperative_threads(&self.pipeline, output_len, "Q2_K matvec")?;

        let row_count_u32 = u32::try_from(row_count)
            .map_err(|_| Error::backend("Q2_K matvec row_count exceeds Metal u32 limit"))?;
        let in_features_u32 = u32::try_from(in_features)
            .map_err(|_| Error::backend("Q2_K matvec in_features exceeds Metal u32 limit"))?;
        let out_features_u32 = u32::try_from(out_features)
            .map_err(|_| Error::backend("Q2_K matvec out_features exceeds Metal u32 limit"))?;
        let blocks_per_row_u32 = u32::try_from(blocks_per_row)
            .map_err(|_| Error::backend("Q2_K matvec blocks_per_row exceeds Metal u32 limit"))?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Q2 scratch buffer lock poisoned"))?;
        let input_buffer = scratch.input_buffer(device, input)?;
        let output_buffer = scratch.output_buffer(device, output_len)?;
        let row_count_buffer = scratch.row_count_buffer(device, row_count_u32)?;
        let in_features_buffer = scratch.in_features_buffer(device, in_features_u32)?;
        let out_features_buffer = scratch.out_features_buffer(device, out_features_u32)?;
        let blocks_per_row_buffer = scratch.blocks_per_row_buffer(device, blocks_per_row_u32)?;

        debug!(
            target: "inferno::expert_cache",
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            weight_bytes = weights.len(),
            weight_zero_copy = true,
            input_buffer_reused = input_buffer.reused,
            output_buffer_reused = output_buffer.reused,
            "running native Metal Q2_K matvec"
        );

        dispatch_1d(
            queue,
            &self.pipeline,
            &[
                &weight_buffer.buffer,
                &input_buffer.buffer,
                &output_buffer.buffer,
                &row_count_buffer.buffer,
                &in_features_buffer.buffer,
                &out_features_buffer.buffer,
                &blocks_per_row_buffer.buffer,
            ],
            physical_threads,
        )?;

        let values = read_f32_buffer(&output_buffer.buffer, output_len)?;

        Ok(MetalQ2MatvecReport {
            values,
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            input_len: input.len(),
            weight_bytes: weights.len(),
            thread_count: physical_threads,
            transposed: false,
        })
    }

    pub(crate) fn run_add(
        &self,
        device: &Device,
        queue: &CommandQueue,
        weights: &[u8],
        input: &[f32],
        residual: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<MetalQ2MatvecAddReport> {
        let blocks_per_row =
            validate_q2_k_matvec_f32(weights, input, row_count, in_features, out_features)?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K matvec add output length overflow"))?;
        let physical_threads =
            q2_k_cooperative_threads(&self.add_pipeline, output_len, "Q2_K matvec add")?;
        if residual.len() != output_len {
            return Err(Error::backend(format!(
                "Q2_K matvec add residual length mismatch: expected {output_len}, got {}",
                residual.len()
            )));
        }
        debug_assert_finite_values("Q2_K matvec add residual", residual);

        let row_count_u32 = u32::try_from(row_count)
            .map_err(|_| Error::backend("Q2_K matvec add row_count exceeds Metal u32 limit"))?;
        let in_features_u32 = u32::try_from(in_features)
            .map_err(|_| Error::backend("Q2_K matvec add in_features exceeds Metal u32 limit"))?;
        let out_features_u32 = u32::try_from(out_features)
            .map_err(|_| Error::backend("Q2_K matvec add out_features exceeds Metal u32 limit"))?;
        let blocks_per_row_u32 = u32::try_from(blocks_per_row).map_err(|_| {
            Error::backend("Q2_K matvec add blocks_per_row exceeds Metal u32 limit")
        })?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Q2 scratch buffer lock poisoned"))?;
        let input_buffer = scratch.input_buffer(device, input)?;
        let residual_buffer = scratch.residual_buffer(device, residual)?;
        let output_buffer = scratch.output_buffer(device, output_len)?;
        let row_count_buffer = scratch.row_count_buffer(device, row_count_u32)?;
        let in_features_buffer = scratch.in_features_buffer(device, in_features_u32)?;
        let out_features_buffer = scratch.out_features_buffer(device, out_features_u32)?;
        let blocks_per_row_buffer = scratch.blocks_per_row_buffer(device, blocks_per_row_u32)?;

        trace!(
            target: "inferno::metal",
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            weight_bytes = weights.len(),
            weight_zero_copy = true,
            input_buffer_reused = input_buffer.reused,
            residual_buffer_reused = residual_buffer.reused,
            output_buffer_reused = output_buffer.reused,
            "running native Metal Q2_K matvec plus residual"
        );

        dispatch_1d(
            queue,
            &self.add_pipeline,
            &[
                &weight_buffer.buffer,
                &input_buffer.buffer,
                &residual_buffer.buffer,
                &output_buffer.buffer,
                &row_count_buffer.buffer,
                &in_features_buffer.buffer,
                &out_features_buffer.buffer,
                &blocks_per_row_buffer.buffer,
            ],
            physical_threads,
        )?;

        let values = read_f32_buffer(&output_buffer.buffer, output_len)?;

        Ok(MetalQ2MatvecAddReport {
            values,
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            input_len: input.len(),
            residual_len: residual.len(),
            weight_bytes: weights.len(),
            thread_count: physical_threads,
        })
    }

    pub(crate) fn run_gate_up_swiglu(
        &self,
        device: &Device,
        queue: &CommandQueue,
        gate_weights: &[u8],
        up_weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<MetalQ2GateUpSwiGluReport> {
        let blocks_per_row = validate_q2_k_gate_up_swiglu_f32(
            gate_weights,
            up_weights,
            input,
            row_count,
            in_features,
            out_features,
        )?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K gate/up SwiGLU output length overflow"))?;
        let physical_threads = q2_k_cooperative_threads(
            &self.gate_up_swiglu_pipeline,
            output_len,
            "Q2_K gate/up SwiGLU",
        )?;

        let row_count_u32 = u32::try_from(row_count)
            .map_err(|_| Error::backend("Q2_K gate/up SwiGLU row_count exceeds Metal u32 limit"))?;
        let in_features_u32 = u32::try_from(in_features).map_err(|_| {
            Error::backend("Q2_K gate/up SwiGLU in_features exceeds Metal u32 limit")
        })?;
        let out_features_u32 = u32::try_from(out_features).map_err(|_| {
            Error::backend("Q2_K gate/up SwiGLU out_features exceeds Metal u32 limit")
        })?;
        let blocks_per_row_u32 = u32::try_from(blocks_per_row).map_err(|_| {
            Error::backend("Q2_K gate/up SwiGLU blocks_per_row exceeds Metal u32 limit")
        })?;

        let gate_weight_buffer = self.weight_buffer(device, gate_weights)?;
        let up_weight_buffer = self.weight_buffer(device, up_weights)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Q2 scratch buffer lock poisoned"))?;
        let input_buffer = scratch.input_buffer(device, input)?;
        let output_buffer = scratch.output_buffer(device, output_len)?;
        let row_count_buffer = scratch.row_count_buffer(device, row_count_u32)?;
        let in_features_buffer = scratch.in_features_buffer(device, in_features_u32)?;
        let out_features_buffer = scratch.out_features_buffer(device, out_features_u32)?;
        let blocks_per_row_buffer = scratch.blocks_per_row_buffer(device, blocks_per_row_u32)?;

        trace!(
            target: "inferno::metal",
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            gate_weight_bytes = gate_weights.len(),
            up_weight_bytes = up_weights.len(),
            gate_weight_zero_copy = true,
            up_weight_zero_copy = true,
            input_buffer_reused = input_buffer.reused,
            output_buffer_reused = output_buffer.reused,
            "running native Metal Q2_K gate/up SwiGLU"
        );

        dispatch_1d(
            queue,
            &self.gate_up_swiglu_pipeline,
            &[
                &gate_weight_buffer.buffer,
                &up_weight_buffer.buffer,
                &input_buffer.buffer,
                &output_buffer.buffer,
                &row_count_buffer.buffer,
                &in_features_buffer.buffer,
                &out_features_buffer.buffer,
                &blocks_per_row_buffer.buffer,
            ],
            physical_threads,
        )?;

        let values = read_f32_buffer(&output_buffer.buffer, output_len)?;

        Ok(MetalQ2GateUpSwiGluReport {
            values,
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            input_len: input.len(),
            gate_weight_bytes: gate_weights.len(),
            up_weight_bytes: up_weights.len(),
            thread_count: physical_threads,
        })
    }

    pub(crate) fn run_argmax(
        &self,
        device: &Device,
        queue: &CommandQueue,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<MetalQ2MatvecArgmaxReport> {
        if row_count != 1 {
            return Err(Error::backend(format!(
                "Q2_K greedy argmax requires row_count 1, got {row_count}"
            )));
        }
        let blocks_per_row =
            validate_q2_k_matvec_f32(weights, input, row_count, in_features, out_features)?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K argmax matvec output length overflow"))?;
        let matvec_threads =
            q2_k_cooperative_threads(&self.pipeline, output_len, "Q2_K argmax matvec")?;

        let row_count_u32 = u32::try_from(row_count)
            .map_err(|_| Error::backend("Q2_K argmax row_count exceeds Metal u32 limit"))?;
        let in_features_u32 = u32::try_from(in_features)
            .map_err(|_| Error::backend("Q2_K argmax in_features exceeds Metal u32 limit"))?;
        let out_features_u32 = u32::try_from(out_features)
            .map_err(|_| Error::backend("Q2_K argmax out_features exceeds Metal u32 limit"))?;
        let blocks_per_row_u32 = u32::try_from(blocks_per_row)
            .map_err(|_| Error::backend("Q2_K argmax blocks_per_row exceeds Metal u32 limit"))?;
        let output_len_u32 = u32::try_from(output_len)
            .map_err(|_| Error::backend("Q2_K argmax output length exceeds Metal u32 limit"))?;
        let argmax_threads = argmax_threads(&self.argmax_pipeline, "Q2_K greedy argmax")?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Q2 scratch buffer lock poisoned"))?;
        let input_buffer = scratch.input_buffer(device, input)?;
        let logits_buffer = scratch.logits_buffer(device, output_len)?;
        let row_count_buffer = scratch.row_count_buffer(device, row_count_u32)?;
        let in_features_buffer = scratch.in_features_buffer(device, in_features_u32)?;
        let out_features_buffer = scratch.out_features_buffer(device, out_features_u32)?;
        let blocks_per_row_buffer = scratch.blocks_per_row_buffer(device, blocks_per_row_u32)?;

        trace!(
            target: "inferno::metal",
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            weight_bytes = weights.len(),
            weight_zero_copy = true,
            input_buffer_reused = input_buffer.reused,
            logits_buffer_reused = logits_buffer.reused,
            "running native Metal Q2_K matvec for greedy argmax"
        );

        let token_id_buffer = scratch.token_id_buffer(device)?;
        let token_score_buffer = scratch.token_score_buffer(device)?;
        let output_len_buffer = scratch.output_len_buffer(device, output_len_u32)?;

        trace!(
            target: "inferno::metal",
            value_count = output_len,
            token_id_buffer_reused = token_id_buffer.reused,
            token_score_buffer_reused = token_score_buffer.reused,
            "running native Metal greedy argmax over Q2 logits"
        );

        let matvec_buffers = [
            &weight_buffer.buffer,
            &input_buffer.buffer,
            &logits_buffer.buffer,
            &row_count_buffer.buffer,
            &in_features_buffer.buffer,
            &out_features_buffer.buffer,
            &blocks_per_row_buffer.buffer,
        ];
        let argmax_buffers = [
            &logits_buffer.buffer,
            &token_id_buffer.buffer,
            &token_score_buffer.buffer,
            &output_len_buffer.buffer,
        ];
        dispatch_1d_many(
            queue,
            &[
                Dispatch1d {
                    pipeline: &self.pipeline,
                    buffers: &matvec_buffers,
                    threads: matvec_threads,
                },
                Dispatch1d {
                    pipeline: &self.argmax_pipeline,
                    buffers: &argmax_buffers,
                    threads: argmax_threads,
                },
            ],
        )?;

        let token_id = read_u32_buffer(&token_id_buffer.buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::backend("Q2_K argmax produced no token id"))?;
        let token_score = read_f32_buffer(&token_score_buffer.buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::backend("Q2_K argmax produced no token score"))?;

        Ok(MetalQ2MatvecArgmaxReport {
            token_id,
            token_score,
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            input_len: input.len(),
            weight_bytes: weights.len(),
            matvec_thread_count: matvec_threads,
            argmax_thread_count: argmax_threads,
        })
    }

    pub(crate) fn run_argmax_with_input_buffer_after_dispatch(
        &self,
        device: &Device,
        queue: &CommandQueue,
        weights: &[u8],
        input_buffer: &Buffer,
        input_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
        prefix: Dispatch1d<'_>,
    ) -> Result<MetalQ2MatvecArgmaxReport> {
        if row_count != 1 {
            return Err(Error::backend(format!(
                "Q2_K greedy argmax requires row_count 1, got {row_count}"
            )));
        }
        let blocks_per_row =
            validate_q2_k_matvec_buffer(weights, input_len, row_count, in_features, out_features)?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K argmax matvec output length overflow"))?;
        let matvec_threads =
            q2_k_cooperative_threads(&self.pipeline, output_len, "Q2_K argmax matvec")?;

        let row_count_u32 = u32::try_from(row_count)
            .map_err(|_| Error::backend("Q2_K argmax row_count exceeds Metal u32 limit"))?;
        let in_features_u32 = u32::try_from(in_features)
            .map_err(|_| Error::backend("Q2_K argmax in_features exceeds Metal u32 limit"))?;
        let out_features_u32 = u32::try_from(out_features)
            .map_err(|_| Error::backend("Q2_K argmax out_features exceeds Metal u32 limit"))?;
        let blocks_per_row_u32 = u32::try_from(blocks_per_row)
            .map_err(|_| Error::backend("Q2_K argmax blocks_per_row exceeds Metal u32 limit"))?;
        let output_len_u32 = u32::try_from(output_len)
            .map_err(|_| Error::backend("Q2_K argmax output length exceeds Metal u32 limit"))?;
        let argmax_threads = argmax_threads(&self.argmax_pipeline, "Q2_K greedy argmax")?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Q2 scratch buffer lock poisoned"))?;
        let logits_buffer = scratch.logits_buffer(device, output_len)?;
        let row_count_buffer = scratch.row_count_buffer(device, row_count_u32)?;
        let in_features_buffer = scratch.in_features_buffer(device, in_features_u32)?;
        let out_features_buffer = scratch.out_features_buffer(device, out_features_u32)?;
        let blocks_per_row_buffer = scratch.blocks_per_row_buffer(device, blocks_per_row_u32)?;

        trace!(
            target: "inferno::metal",
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            weight_bytes = weights.len(),
            weight_zero_copy = true,
            logits_buffer_reused = logits_buffer.reused,
            "running native Metal Q2_K argmax from resident input buffer"
        );

        let token_id_buffer = scratch.token_id_buffer(device)?;
        let token_score_buffer = scratch.token_score_buffer(device)?;
        let output_len_buffer = scratch.output_len_buffer(device, output_len_u32)?;

        let matvec_buffers = [
            &weight_buffer.buffer,
            input_buffer,
            &logits_buffer.buffer,
            &row_count_buffer.buffer,
            &in_features_buffer.buffer,
            &out_features_buffer.buffer,
            &blocks_per_row_buffer.buffer,
        ];
        let argmax_buffers = [
            &logits_buffer.buffer,
            &token_id_buffer.buffer,
            &token_score_buffer.buffer,
            &output_len_buffer.buffer,
        ];
        dispatch_1d_many(
            queue,
            &[
                prefix,
                Dispatch1d {
                    pipeline: &self.pipeline,
                    buffers: &matvec_buffers,
                    threads: matvec_threads,
                },
                Dispatch1d {
                    pipeline: &self.argmax_pipeline,
                    buffers: &argmax_buffers,
                    threads: argmax_threads,
                },
            ],
        )?;

        let token_id = read_u32_buffer(&token_id_buffer.buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::backend("Q2_K argmax produced no token id"))?;
        let token_score = read_f32_buffer(&token_score_buffer.buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::backend("Q2_K argmax produced no token score"))?;

        Ok(MetalQ2MatvecArgmaxReport {
            token_id,
            token_score,
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            input_len,
            weight_bytes: weights.len(),
            matvec_thread_count: matvec_threads,
            argmax_thread_count: argmax_threads,
        })
    }

    pub(crate) fn run_q8_0(
        &self,
        device: &Device,
        queue: &CommandQueue,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<MetalQ2MatvecReport> {
        let blocks_per_row =
            validate_q8_0_matvec_f32(weights, input, row_count, in_features, out_features)?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q8_0 matvec output length overflow"))?;
        let physical_threads =
            q2_k_cooperative_threads(&self.q8_0_pipeline, output_len, "Q8_0 matvec")?;

        let row_count_u32 = u32::try_from(row_count)
            .map_err(|_| Error::backend("Q8_0 matvec row_count exceeds Metal u32 limit"))?;
        let in_features_u32 = u32::try_from(in_features)
            .map_err(|_| Error::backend("Q8_0 matvec in_features exceeds Metal u32 limit"))?;
        let out_features_u32 = u32::try_from(out_features)
            .map_err(|_| Error::backend("Q8_0 matvec out_features exceeds Metal u32 limit"))?;
        let blocks_per_row_u32 = u32::try_from(blocks_per_row)
            .map_err(|_| Error::backend("Q8_0 matvec blocks_per_row exceeds Metal u32 limit"))?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Q2 scratch buffer lock poisoned"))?;
        let input_buffer = scratch.input_buffer(device, input)?;
        let output_buffer = scratch.output_buffer(device, output_len)?;
        let row_count_buffer = scratch.row_count_buffer(device, row_count_u32)?;
        let in_features_buffer = scratch.in_features_buffer(device, in_features_u32)?;
        let out_features_buffer = scratch.out_features_buffer(device, out_features_u32)?;
        let blocks_per_row_buffer = scratch.blocks_per_row_buffer(device, blocks_per_row_u32)?;

        trace!(
            target: "inferno::metal",
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            weight_bytes = weights.len(),
            weight_zero_copy = true,
            input_buffer_reused = input_buffer.reused,
            output_buffer_reused = output_buffer.reused,
            "running native Metal Q8_0 matvec"
        );

        dispatch_1d(
            queue,
            &self.q8_0_pipeline,
            &[
                &weight_buffer.buffer,
                &input_buffer.buffer,
                &output_buffer.buffer,
                &row_count_buffer.buffer,
                &in_features_buffer.buffer,
                &out_features_buffer.buffer,
                &blocks_per_row_buffer.buffer,
            ],
            physical_threads,
        )?;

        let values = read_f32_buffer(&output_buffer.buffer, output_len)?;

        Ok(MetalQ2MatvecReport {
            values,
            row_count,
            in_features,
            out_features,
            blocks_per_row,
            input_len: input.len(),
            weight_bytes: weights.len(),
            thread_count: physical_threads,
            transposed: false,
        })
    }

    pub(crate) fn run_q8_0_transposed(
        &self,
        device: &Device,
        queue: &CommandQueue,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<MetalQ2MatvecReport> {
        let blocks_per_input_row = validate_q8_0_transposed_matvec_f32(
            weights,
            input,
            row_count,
            in_features,
            out_features,
        )?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q8_0 transposed matvec output length overflow"))?;
        let physical_threads = q2_k_cooperative_threads(
            &self.q8_0_transposed_pipeline,
            output_len,
            "Q8_0 transposed matvec",
        )?;

        let row_count_u32 = u32::try_from(row_count).map_err(|_| {
            Error::backend("Q8_0 transposed matvec row_count exceeds Metal u32 limit")
        })?;
        let in_features_u32 = u32::try_from(in_features).map_err(|_| {
            Error::backend("Q8_0 transposed matvec in_features exceeds Metal u32 limit")
        })?;
        let out_features_u32 = u32::try_from(out_features).map_err(|_| {
            Error::backend("Q8_0 transposed matvec out_features exceeds Metal u32 limit")
        })?;
        let blocks_per_input_row_u32 = u32::try_from(blocks_per_input_row).map_err(|_| {
            Error::backend("Q8_0 transposed matvec blocks_per_input_row exceeds Metal u32 limit")
        })?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Q2 scratch buffer lock poisoned"))?;
        let input_buffer = scratch.input_buffer(device, input)?;
        let output_buffer = scratch.output_buffer(device, output_len)?;
        let row_count_buffer = scratch.row_count_buffer(device, row_count_u32)?;
        let in_features_buffer = scratch.in_features_buffer(device, in_features_u32)?;
        let out_features_buffer = scratch.out_features_buffer(device, out_features_u32)?;
        let blocks_per_input_row_buffer =
            scratch.blocks_per_row_buffer(device, blocks_per_input_row_u32)?;

        trace!(
            target: "inferno::metal",
            row_count,
            in_features,
            out_features,
            blocks_per_input_row,
            weight_bytes = weights.len(),
            weight_zero_copy = true,
            input_buffer_reused = input_buffer.reused,
            output_buffer_reused = output_buffer.reused,
            "running native Metal transposed Q8_0 matvec"
        );

        dispatch_1d(
            queue,
            &self.q8_0_transposed_pipeline,
            &[
                &weight_buffer.buffer,
                &input_buffer.buffer,
                &output_buffer.buffer,
                &row_count_buffer.buffer,
                &in_features_buffer.buffer,
                &out_features_buffer.buffer,
                &blocks_per_input_row_buffer.buffer,
            ],
            physical_threads,
        )?;

        let values = read_f32_buffer(&output_buffer.buffer, output_len)?;

        Ok(MetalQ2MatvecReport {
            values,
            row_count,
            in_features,
            out_features,
            blocks_per_row: blocks_per_input_row,
            input_len: input.len(),
            weight_bytes: weights.len(),
            thread_count: physical_threads,
            transposed: true,
        })
    }

    pub(crate) fn run_transposed(
        &self,
        device: &Device,
        queue: &CommandQueue,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<MetalQ2MatvecReport> {
        let blocks_per_input_row = validate_q2_k_transposed_matvec_f32(
            weights,
            input,
            row_count,
            in_features,
            out_features,
        )?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K transposed matvec output length overflow"))?;
        let physical_threads = q2_k_cooperative_threads(
            &self.transposed_pipeline,
            output_len,
            "Q2_K transposed matvec",
        )?;

        let row_count_u32 = u32::try_from(row_count).map_err(|_| {
            Error::backend("Q2_K transposed matvec row_count exceeds Metal u32 limit")
        })?;
        let in_features_u32 = u32::try_from(in_features).map_err(|_| {
            Error::backend("Q2_K transposed matvec in_features exceeds Metal u32 limit")
        })?;
        let out_features_u32 = u32::try_from(out_features).map_err(|_| {
            Error::backend("Q2_K transposed matvec out_features exceeds Metal u32 limit")
        })?;
        let blocks_per_input_row_u32 = u32::try_from(blocks_per_input_row).map_err(|_| {
            Error::backend("Q2_K transposed matvec blocks_per_input_row exceeds Metal u32 limit")
        })?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| Error::backend("Q2 scratch buffer lock poisoned"))?;
        let input_buffer = scratch.input_buffer(device, input)?;
        let output_buffer = scratch.output_buffer(device, output_len)?;
        let row_count_buffer = scratch.row_count_buffer(device, row_count_u32)?;
        let in_features_buffer = scratch.in_features_buffer(device, in_features_u32)?;
        let out_features_buffer = scratch.out_features_buffer(device, out_features_u32)?;
        let blocks_per_input_row_buffer =
            scratch.blocks_per_row_buffer(device, blocks_per_input_row_u32)?;

        trace!(
            target: "inferno::metal",
            row_count,
            in_features,
            out_features,
            blocks_per_input_row,
            weight_bytes = weights.len(),
            weight_zero_copy = true,
            input_buffer_reused = input_buffer.reused,
            output_buffer_reused = output_buffer.reused,
            "running native Metal transposed Q2_K matvec"
        );

        dispatch_1d(
            queue,
            &self.transposed_pipeline,
            &[
                &weight_buffer.buffer,
                &input_buffer.buffer,
                &output_buffer.buffer,
                &row_count_buffer.buffer,
                &in_features_buffer.buffer,
                &out_features_buffer.buffer,
                &blocks_per_input_row_buffer.buffer,
            ],
            physical_threads,
        )?;

        let values = read_f32_buffer(&output_buffer.buffer, output_len)?;

        Ok(MetalQ2MatvecReport {
            values,
            row_count,
            in_features,
            out_features,
            blocks_per_row: blocks_per_input_row,
            input_len: input.len(),
            weight_bytes: weights.len(),
            thread_count: physical_threads,
            transposed: true,
        })
    }

    /// Encodes one quantized matvec kernel into an open batched command buffer
    /// without dispatching it. The input already lives in a GPU buffer (it is
    /// usually the not-yet-computed output of a previously encoded kernel), so
    /// unlike the `run_*` entry points this performs shape-only validation and
    /// allocates a fresh output buffer instead of recycling pooled scratch —
    /// pooled buffers must not be overwritten while earlier encoded kernels
    /// still reference them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_matvec(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        kind: QuantMatvecKind,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        let blocks_per_row = match kind {
            QuantMatvecKind::Q2K => validate_q2_k_matvec_buffer(
                weights,
                input_len,
                row_count,
                in_features,
                out_features,
            )?,
            QuantMatvecKind::Q2KTransposed => validate_q2_k_transposed_matvec_buffer(
                weights,
                input_len,
                row_count,
                in_features,
                out_features,
            )?,
            QuantMatvecKind::Q80 => validate_q8_0_matvec_buffer(
                weights,
                input_len,
                row_count,
                in_features,
                out_features,
            )?,
            QuantMatvecKind::Q80Transposed => validate_q8_0_transposed_matvec_buffer(
                weights,
                input_len,
                row_count,
                in_features,
                out_features,
            )?,
        };
        require_f32_capacity(input, input_len, "quantized matvec input")?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("quantized matvec output length overflow"))?;

        let pipeline = match kind {
            QuantMatvecKind::Q2K => &self.pipeline,
            QuantMatvecKind::Q2KTransposed => &self.transposed_pipeline,
            QuantMatvecKind::Q80 => &self.q8_0_tiled_pipeline,
            QuantMatvecKind::Q80Transposed => &self.q8_0_transposed_pipeline,
        };
        let weight_buffer = self.weight_buffer(device, weights)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let row_count_buffer = u32_scalar_buffer(device, matvec_u32(row_count, "row_count")?)?;
        let in_features_buffer =
            u32_scalar_buffer(device, matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer =
            u32_scalar_buffer(device, matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer =
            u32_scalar_buffer(device, matvec_u32(blocks_per_row, "blocks_per_row")?)?;

        trace!(
            target: "inferno::metal",
            ?kind,
            row_count,
            in_features,
            out_features,
            weight_zero_copy = true,
            "encoding batched quantized matvec"
        );

        let buffers = [
            &weight_buffer.buffer,
            input,
            &output_buffer,
            &row_count_buffer,
            &in_features_buffer,
            &out_features_buffer,
            &blocks_per_row_buffer,
        ];
        if kind == QuantMatvecKind::Q80 {
            let simdgroups_per_output = q8_0_simdgroups_per_output(blocks_per_row);
            let simdgroups_per_output_buffer = u32_scalar_buffer(
                device,
                matvec_u32(simdgroups_per_output, "simdgroups_per_output")?,
            )?;
            encode_1d_threadgroups(
                command_buffer,
                pipeline,
                &[
                    buffers[0],
                    buffers[1],
                    buffers[2],
                    buffers[3],
                    buffers[4],
                    buffers[5],
                    buffers[6],
                    &simdgroups_per_output_buffer,
                ],
                output_len,
                simdgroups_per_output * Q2_K_SIMD_LANES,
            )?;
        } else {
            let dispatch_threads = q2_k_cooperative_threads(
                pipeline,
                output_len,
                "batched cooperative quantized matvec",
            )?;
            encode_1d(command_buffer, pipeline, &buffers, dispatch_threads)?;
        }
        Ok(output_buffer)
    }

    /// Encodes all transposed per-head projections in one dispatch.
    ///
    /// GGUF stores `attn_k_b` and `attn_v_b` as `head_count` contiguous
    /// matrices. Encoding one kernel per head creates 128 compute encoders per
    /// layer for K and V. This entry point addresses the packed head dimension
    /// directly and writes `[row_count, head_count, out_features]` in one pass.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_packed_heads_transposed_matvec(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        kind: QuantMatvecKind,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        row_count: usize,
        head_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        let (pipeline, block_values, block_bytes) = match kind {
            QuantMatvecKind::Q2KTransposed => (
                &self.packed_heads_transposed_pipeline,
                Q2_K_BLOCK_VALUES,
                Q2_K_BLOCK_BYTES,
            ),
            QuantMatvecKind::Q80Transposed => (
                &self.q8_0_packed_heads_transposed_pipeline,
                Q8_0_BLOCK_VALUES,
                Q8_0_BLOCK_BYTES,
            ),
            other => {
                return Err(Error::backend(format!(
                    "packed-head transposed matvec does not support {other:?}"
                )))
            }
        };
        if row_count == 0 || head_count == 0 || in_features == 0 || out_features == 0 {
            return Err(Error::backend(
                "packed-head transposed matvec dimensions must be positive",
            ));
        }
        if out_features % block_values != 0 {
            return Err(Error::backend(format!(
                "packed-head transposed output width {out_features} must be divisible by quant block size {block_values}"
            )));
        }
        let expected_input_len = row_count
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("packed-head transposed input length overflow"))?;
        validate_exact_len(
            "packed-head transposed input",
            input_len,
            expected_input_len,
        )?;
        require_f32_capacity(input, input_len, "packed-head transposed input")?;

        let blocks_per_input_row = out_features / block_values;
        let blocks_per_head = in_features
            .checked_mul(blocks_per_input_row)
            .ok_or_else(|| Error::backend("packed-head transposed blocks per head overflow"))?;
        let expected_weight_bytes = head_count
            .checked_mul(blocks_per_head)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or_else(|| Error::backend("packed-head transposed weight length overflow"))?;
        validate_exact_len(
            "packed-head transposed weights",
            weights.len(),
            expected_weight_bytes,
        )?;

        let output_len = row_count
            .checked_mul(head_count)
            .and_then(|rows| rows.checked_mul(out_features))
            .ok_or_else(|| Error::backend("packed-head transposed output length overflow"))?;
        let physical_threads =
            q2_k_cooperative_threads(pipeline, output_len, "packed-head transposed matvec")?;
        let weight_buffer = self.weight_buffer(device, weights)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let row_count_buffer = u32_scalar_buffer(device, matvec_u32(row_count, "row_count")?)?;
        let head_count_buffer = u32_scalar_buffer(device, matvec_u32(head_count, "head_count")?)?;
        let in_features_buffer =
            u32_scalar_buffer(device, matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer =
            u32_scalar_buffer(device, matvec_u32(out_features, "out_features")?)?;
        let blocks_per_input_row_buffer = u32_scalar_buffer(
            device,
            matvec_u32(blocks_per_input_row, "blocks_per_input_row")?,
        )?;
        let blocks_per_head_buffer =
            u32_scalar_buffer(device, matvec_u32(blocks_per_head, "blocks_per_head")?)?;

        trace!(
            target: "inferno::metal",
            ?kind,
            row_count,
            head_count,
            in_features,
            out_features,
            "encoding packed-head transposed quantized matvec"
        );

        encode_1d(
            command_buffer,
            pipeline,
            &[
                &weight_buffer.buffer,
                input,
                &output_buffer,
                &row_count_buffer,
                &head_count_buffer,
                &in_features_buffer,
                &out_features_buffer,
                &blocks_per_input_row_buffer,
                &blocks_per_head_buffer,
            ],
            physical_threads,
        )?;
        Ok(output_buffer)
    }

    /// Applies one Q8_0 matrix per head to one input vector per head.
    ///
    /// The packed weight layout is `[head_count, out_features, in_features]`.
    /// Both input and output keep the head dimension explicit, which is what
    /// absorbed MLA needs for its per-head K_b and V_b projections.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_q8_0_packed_heads_matvec(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        row_count: usize,
        head_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        if row_count == 0 || head_count == 0 || in_features == 0 || out_features == 0 {
            return Err(Error::backend(
                "packed-head Q8_0 matvec dimensions must be positive",
            ));
        }
        if in_features % Q8_0_BLOCK_VALUES != 0 {
            return Err(Error::backend(format!(
                "packed-head Q8_0 input width {in_features} must be divisible by {Q8_0_BLOCK_VALUES}"
            )));
        }

        let expected_input_len = row_count
            .checked_mul(head_count)
            .and_then(|rows| rows.checked_mul(in_features))
            .ok_or_else(|| Error::backend("packed-head Q8_0 input length overflow"))?;
        validate_exact_len("packed-head Q8_0 input", input_len, expected_input_len)?;
        require_f32_capacity(input, input_len, "packed-head Q8_0 input")?;

        let blocks_per_row = in_features / Q8_0_BLOCK_VALUES;
        let blocks_per_head = out_features
            .checked_mul(blocks_per_row)
            .ok_or_else(|| Error::backend("packed-head Q8_0 blocks per head overflow"))?;
        let expected_weight_bytes = head_count
            .checked_mul(blocks_per_head)
            .and_then(|blocks| blocks.checked_mul(Q8_0_BLOCK_BYTES))
            .ok_or_else(|| Error::backend("packed-head Q8_0 weight length overflow"))?;
        validate_exact_len(
            "packed-head Q8_0 weights",
            weights.len(),
            expected_weight_bytes,
        )?;

        let output_len = row_count
            .checked_mul(head_count)
            .and_then(|rows| rows.checked_mul(out_features))
            .ok_or_else(|| Error::backend("packed-head Q8_0 output length overflow"))?;
        let physical_threads = q2_k_cooperative_threads(
            &self.q8_0_packed_heads_pipeline,
            output_len,
            "packed-head Q8_0 matvec",
        )?;
        let weight_buffer = self.weight_buffer(device, weights)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let row_count_buffer = u32_scalar_buffer(device, matvec_u32(row_count, "row_count")?)?;
        let head_count_buffer = u32_scalar_buffer(device, matvec_u32(head_count, "head_count")?)?;
        let in_features_buffer =
            u32_scalar_buffer(device, matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer =
            u32_scalar_buffer(device, matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer =
            u32_scalar_buffer(device, matvec_u32(blocks_per_row, "blocks_per_row")?)?;
        let blocks_per_head_buffer =
            u32_scalar_buffer(device, matvec_u32(blocks_per_head, "blocks_per_head")?)?;

        encode_1d(
            command_buffer,
            &self.q8_0_packed_heads_pipeline,
            &[
                &weight_buffer.buffer,
                input,
                &output_buffer,
                &row_count_buffer,
                &head_count_buffer,
                &in_features_buffer,
                &out_features_buffer,
                &blocks_per_row_buffer,
                &blocks_per_head_buffer,
            ],
            physical_threads,
        )?;
        Ok(output_buffer)
    }

    /// Encodes a Q2_K matvec fused with a residual add into an open batched
    /// command buffer. See `encode_matvec` for the batching rules.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_matvec_add(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        residual: &Buffer,
        residual_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        let blocks_per_row =
            validate_q2_k_matvec_buffer(weights, input_len, row_count, in_features, out_features)?;
        require_f32_capacity(input, input_len, "Q2_K matvec add input")?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K matvec add output length overflow"))?;
        let physical_threads =
            q2_k_cooperative_threads(&self.add_pipeline, output_len, "batched Q2_K matvec add")?;
        if residual_len != output_len {
            return Err(Error::backend(format!(
                "Q2_K matvec add residual length mismatch: expected {output_len}, got {residual_len}"
            )));
        }
        require_f32_capacity(residual, residual_len, "Q2_K matvec add residual")?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let row_count_buffer = u32_scalar_buffer(device, matvec_u32(row_count, "row_count")?)?;
        let in_features_buffer =
            u32_scalar_buffer(device, matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer =
            u32_scalar_buffer(device, matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer =
            u32_scalar_buffer(device, matvec_u32(blocks_per_row, "blocks_per_row")?)?;

        trace!(
            target: "inferno::metal",
            row_count,
            in_features,
            out_features,
            weight_zero_copy = true,
            "encoding batched Q2_K matvec plus residual"
        );

        encode_1d(
            command_buffer,
            &self.add_pipeline,
            &[
                &weight_buffer.buffer,
                input,
                residual,
                &output_buffer,
                &row_count_buffer,
                &in_features_buffer,
                &out_features_buffer,
                &blocks_per_row_buffer,
            ],
            physical_threads,
        )?;
        Ok(output_buffer)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_q8_0_matvec_add(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        residual: &Buffer,
        residual_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        let blocks_per_row =
            validate_q8_0_matvec_buffer(weights, input_len, row_count, in_features, out_features)?;
        require_f32_capacity(input, input_len, "Q8_0 matvec add input")?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q8_0 matvec add output length overflow"))?;
        if residual_len != output_len {
            return Err(Error::backend(format!(
                "Q8_0 matvec add residual length mismatch: expected {output_len}, got {residual_len}"
            )));
        }
        require_f32_capacity(residual, residual_len, "Q8_0 matvec add residual")?;

        let simdgroups_per_output = q8_0_simdgroups_per_output(blocks_per_row);
        let weight_buffer = self.weight_buffer(device, weights)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let row_count_buffer = u32_scalar_buffer(device, matvec_u32(row_count, "row_count")?)?;
        let in_features_buffer =
            u32_scalar_buffer(device, matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer =
            u32_scalar_buffer(device, matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer =
            u32_scalar_buffer(device, matvec_u32(blocks_per_row, "blocks_per_row")?)?;
        let simdgroups_per_output_buffer = u32_scalar_buffer(
            device,
            matvec_u32(simdgroups_per_output, "simdgroups_per_output")?,
        )?;

        encode_1d_threadgroups(
            command_buffer,
            &self.q8_0_tiled_add_pipeline,
            &[
                &weight_buffer.buffer,
                input,
                residual,
                &output_buffer,
                &row_count_buffer,
                &in_features_buffer,
                &out_features_buffer,
                &blocks_per_row_buffer,
                &simdgroups_per_output_buffer,
            ],
            output_len,
            simdgroups_per_output * Q2_K_SIMD_LANES,
        )?;
        Ok(output_buffer)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_multi_expert_gate_up_swiglu(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        gate_weights: &[u8],
        up_weights: &[u8],
        input: &Buffer,
        input_len: usize,
        token_indices: &[u32],
        expert_ids: &[u32],
        token_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        let assignment_count = validate_multi_expert_routing(
            token_indices,
            expert_ids,
            token_count,
            "Q2_K multi-expert gate/up SwiGLU",
        )?;
        let (gate_blocks, gate_experts, expert_stride_bytes) = validate_q2_k_packed_experts(
            gate_weights,
            in_features,
            out_features,
            "Q2_K multi-expert gate",
        )?;
        let (up_blocks, up_experts, up_expert_stride_bytes) = validate_q2_k_packed_experts(
            up_weights,
            in_features,
            out_features,
            "Q2_K multi-expert up",
        )?;
        if gate_blocks != up_blocks {
            return Err(Error::backend(format!(
                "Q2_K multi-expert gate/up block mismatch: gate has {gate_blocks}, up has {up_blocks}"
            )));
        }
        if gate_experts != up_experts || expert_stride_bytes != up_expert_stride_bytes {
            return Err(Error::backend(format!(
                "Q2_K multi-expert gate/up packed layout mismatch: gate experts={gate_experts} stride={expert_stride_bytes}, up experts={up_experts} stride={up_expert_stride_bytes}"
            )));
        }
        validate_expert_ids(expert_ids, gate_experts, "Q2_K multi-expert gate/up SwiGLU")?;

        let expected_input_len = token_count
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("Q2_K multi-expert gate/up input length overflow"))?;
        if input_len != expected_input_len {
            return Err(Error::backend(format!(
                "Q2_K multi-expert gate/up input length mismatch: expected {expected_input_len}, got {input_len}"
            )));
        }
        require_f32_capacity(input, input_len, "Q2_K multi-expert gate/up input")?;

        let output_len = assignment_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K multi-expert gate/up output length overflow"))?;
        let physical_threads = q2_k_cooperative_threads(
            &self.multi_expert_gate_up_swiglu_pipeline,
            output_len,
            "batched Q2_K multi-expert gate/up SwiGLU",
        )?;

        let gate_weight_buffer = self.weight_buffer(device, gate_weights)?;
        let up_weight_buffer = self.weight_buffer(device, up_weights)?;
        let token_indices_buffer = u32_buffer(device, token_indices)?;
        let expert_ids_buffer = u32_buffer(device, expert_ids)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let token_count_buffer =
            u32_scalar_buffer(device, matvec_u32(token_count, "token_count")?)?;
        let assignment_count_buffer =
            u32_scalar_buffer(device, matvec_u32(assignment_count, "assignment_count")?)?;
        let in_features_buffer =
            u32_scalar_buffer(device, matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer =
            u32_scalar_buffer(device, matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer =
            u32_scalar_buffer(device, matvec_u32(gate_blocks, "blocks_per_row")?)?;
        let expert_stride_buffer = u32_scalar_buffer(
            device,
            matvec_u32(expert_stride_bytes, "expert_stride_bytes")?,
        )?;

        trace!(
            target: "inferno::metal",
            token_count,
            assignment_count,
            in_features,
            out_features,
            expert_count = gate_experts,
            "encoding batched Q2_K multi-expert gate/up SwiGLU"
        );

        encode_1d(
            command_buffer,
            &self.multi_expert_gate_up_swiglu_pipeline,
            &[
                &gate_weight_buffer.buffer,
                &up_weight_buffer.buffer,
                input,
                &token_indices_buffer,
                &expert_ids_buffer,
                &output_buffer,
                &token_count_buffer,
                &assignment_count_buffer,
                &in_features_buffer,
                &out_features_buffer,
                &blocks_per_row_buffer,
                &expert_stride_buffer,
            ],
            physical_threads,
        )?;
        Ok(output_buffer)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_multi_expert_matvec(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        expert_ids: &[u32],
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        if expert_ids.is_empty() {
            return Err(Error::backend(
                "Q2_K multi-expert matvec requires at least one assignment",
            ));
        }
        let assignment_count = expert_ids.len();
        let (blocks_per_row, expert_count, expert_stride_bytes) = validate_q2_k_packed_experts(
            weights,
            in_features,
            out_features,
            "Q2_K multi-expert matvec",
        )?;
        validate_expert_ids(expert_ids, expert_count, "Q2_K multi-expert matvec")?;
        let expected_input_len = assignment_count
            .checked_mul(in_features)
            .ok_or_else(|| Error::backend("Q2_K multi-expert matvec input length overflow"))?;
        if input_len != expected_input_len {
            return Err(Error::backend(format!(
                "Q2_K multi-expert matvec input length mismatch: expected {expected_input_len}, got {input_len}"
            )));
        }
        require_f32_capacity(input, input_len, "Q2_K multi-expert matvec input")?;

        let output_len = assignment_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K multi-expert matvec output length overflow"))?;
        let physical_threads = q2_k_cooperative_threads(
            &self.multi_expert_matvec_pipeline,
            output_len,
            "batched Q2_K multi-expert matvec",
        )?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let expert_ids_buffer = u32_buffer(device, expert_ids)?;
        let output_buffer = empty_f32_buffer(device, output_len)?;
        let assignment_count_buffer =
            u32_scalar_buffer(device, matvec_u32(assignment_count, "assignment_count")?)?;
        let in_features_buffer =
            u32_scalar_buffer(device, matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer =
            u32_scalar_buffer(device, matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer =
            u32_scalar_buffer(device, matvec_u32(blocks_per_row, "blocks_per_row")?)?;
        let expert_stride_buffer = u32_scalar_buffer(
            device,
            matvec_u32(expert_stride_bytes, "expert_stride_bytes")?,
        )?;

        trace!(
            target: "inferno::metal",
            assignment_count,
            in_features,
            out_features,
            expert_count,
            "encoding batched Q2_K multi-expert matvec"
        );

        encode_1d(
            command_buffer,
            &self.multi_expert_matvec_pipeline,
            &[
                &weight_buffer.buffer,
                input,
                &expert_ids_buffer,
                &output_buffer,
                &assignment_count_buffer,
                &in_features_buffer,
                &out_features_buffer,
                &blocks_per_row_buffer,
                &expert_stride_buffer,
            ],
            physical_threads,
        )?;
        Ok(output_buffer)
    }

    /// Encodes the Q2_K output-head matvec plus greedy argmax into an open
    /// batched command buffer. Returns the one-element token id (u32) and token
    /// score (f32) buffers; they hold valid data only after the batch flushes.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_argmax(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<(Buffer, Buffer)> {
        if row_count != 1 {
            return Err(Error::backend(format!(
                "Q2_K greedy argmax requires row_count 1, got {row_count}"
            )));
        }
        let blocks_per_row =
            validate_q2_k_matvec_buffer(weights, input_len, row_count, in_features, out_features)?;
        require_f32_capacity(input, input_len, "Q2_K argmax input")?;
        let output_len = row_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("Q2_K argmax matvec output length overflow"))?;
        let matvec_threads =
            q2_k_cooperative_threads(&self.pipeline, output_len, "batched Q2_K argmax matvec")?;
        let argmax_threads = argmax_threads(&self.argmax_pipeline, "batched Q2_K greedy argmax")?;

        let weight_buffer = self.weight_buffer(device, weights)?;
        let logits_buffer = empty_f32_buffer(device, output_len)?;
        let token_id_buffer = empty_u32_buffer(device, 1)?;
        let token_score_buffer = empty_f32_buffer(device, 1)?;
        let row_count_buffer = u32_scalar_buffer(device, matvec_u32(row_count, "row_count")?)?;
        let in_features_buffer =
            u32_scalar_buffer(device, matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer =
            u32_scalar_buffer(device, matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer =
            u32_scalar_buffer(device, matvec_u32(blocks_per_row, "blocks_per_row")?)?;
        let output_len_buffer = u32_scalar_buffer(device, matvec_u32(output_len, "output_len")?)?;

        trace!(
            target: "inferno::metal",
            in_features,
            out_features,
            weight_zero_copy = true,
            "encoding batched Q2_K matvec plus greedy argmax"
        );

        encode_1d(
            command_buffer,
            &self.pipeline,
            &[
                &weight_buffer.buffer,
                input,
                &logits_buffer,
                &row_count_buffer,
                &in_features_buffer,
                &out_features_buffer,
                &blocks_per_row_buffer,
            ],
            matvec_threads,
        )?;
        encode_1d(
            command_buffer,
            &self.argmax_pipeline,
            &[
                &logits_buffer,
                &token_id_buffer,
                &token_score_buffer,
                &output_len_buffer,
            ],
            argmax_threads,
        )?;
        Ok((token_id_buffer, token_score_buffer))
    }

    pub(crate) fn encode_f32_argmax(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        scores: &Buffer,
        value_count: usize,
    ) -> Result<(Buffer, Buffer)> {
        if value_count == 0 {
            return Err(Error::backend("f32 argmax requires at least one value"));
        }
        require_f32_capacity(scores, value_count, "f32 argmax scores")?;
        let argmax_threads = argmax_threads(&self.argmax_pipeline, "batched f32 greedy argmax")?;
        let token_id_buffer = empty_u32_buffer(device, 1)?;
        let token_score_buffer = empty_f32_buffer(device, 1)?;
        let value_count_buffer =
            u32_scalar_buffer(device, matvec_u32(value_count, "value_count")?)?;

        encode_1d(
            command_buffer,
            &self.argmax_pipeline,
            &[
                scores,
                &token_id_buffer,
                &token_score_buffer,
                &value_count_buffer,
            ],
            argmax_threads,
        )?;
        Ok((token_id_buffer, token_score_buffer))
    }

    pub(crate) fn encode_f32_argmax_rows(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        scores: &Buffer,
        row_count: usize,
        row_width: usize,
    ) -> Result<(Buffer, Buffer)> {
        if row_count == 0 || row_width == 0 {
            return Err(Error::backend(
                "row-wise f32 argmax requires non-zero row_count and row_width",
            ));
        }
        let value_count = row_count
            .checked_mul(row_width)
            .ok_or_else(|| Error::backend("row-wise f32 argmax value count overflow"))?;
        require_f32_capacity(scores, value_count, "row-wise f32 argmax scores")?;
        let threads_per_row = argmax_threads(
            &self.argmax_rows_pipeline,
            "batched row-wise f32 greedy argmax",
        )?;
        let physical_threads = row_count
            .checked_mul(threads_per_row)
            .ok_or_else(|| Error::backend("row-wise f32 argmax thread count overflow"))?;
        let token_id_buffer = empty_u32_buffer(device, row_count)?;
        let token_score_buffer = empty_f32_buffer(device, row_count)?;
        let row_width_buffer = u32_scalar_buffer(device, matvec_u32(row_width, "row_width")?)?;

        encode_1d(
            command_buffer,
            &self.argmax_rows_pipeline,
            &[
                scores,
                &token_id_buffer,
                &token_score_buffer,
                &row_width_buffer,
            ],
            physical_threads,
        )?;
        Ok((token_id_buffer, token_score_buffer))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_ready_routed_experts(
        &self,
        device: &Device,
        layer_index: usize,
        model_path: &Path,
        gate_payloads: &[Q2ExpertSource<'_>],
        up_payloads: &[Q2ExpertSource<'_>],
        down_payloads: &[Q2ExpertSource<'_>],
        input: &Buffer,
        input_len: usize,
        token_indices: &Buffer,
        token_count: usize,
        top_k: usize,
        in_features: usize,
        intermediate_features: usize,
        out_features: usize,
    ) -> Result<ReadyRoutedExperts> {
        let assignment_count = token_count
            .checked_mul(top_k)
            .ok_or_else(|| Error::backend("ready routed expert assignment count overflow"))?;
        if assignment_count == 0 {
            return Err(Error::backend(
                "ready routed expert execution requires at least one assignment",
            ));
        }
        validate_exact_len(
            "ready routed gate assignment count",
            gate_payloads.len(),
            assignment_count,
        )?;
        validate_exact_len(
            "ready routed up assignment count",
            up_payloads.len(),
            assignment_count,
        )?;
        validate_exact_len(
            "ready routed down assignment count",
            down_payloads.len(),
            assignment_count,
        )?;
        validate_exact_len(
            "ready routed input",
            input_len,
            token_count
                .checked_mul(in_features)
                .ok_or_else(|| Error::backend("ready routed input length overflow"))?,
        )?;
        require_f32_capacity(input, input_len, "ready routed input")?;
        require_u32_buffer_capacity(
            token_indices,
            assignment_count,
            "ready routed token indices",
        )?;

        let gate_stride = uniform_payload_stride(gate_payloads, "ready routed gate")?;
        let up_stride = uniform_payload_stride(up_payloads, "ready routed up")?;
        let down_stride = uniform_payload_stride(down_payloads, "ready routed down")?;
        validate_exact_len(
            "ready routed gate stride",
            gate_stride,
            q2_expert_stride(in_features, intermediate_features)?,
        )?;
        validate_exact_len(
            "ready routed up stride",
            up_stride,
            q2_expert_stride(in_features, intermediate_features)?,
        )?;
        validate_exact_len(
            "ready routed down stride",
            down_stride,
            q2_expert_stride(intermediate_features, out_features)?,
        )?;

        let lookup_started = Instant::now();
        let (layer_cache, model_file, model_path) = {
            let mut cache = self
                .ready_expert_cache
                .lock()
                .map_err(|_| Error::backend("Q2 per-layer expert cache lock poisoned"))?;
            cache.ensure_model_file(model_path)?;
            cache.ensure_layout(gate_stride, up_stride, down_stride);
            let layer_cache = cache.layer(layer_index);
            let model_file = cache
                .model_file
                .as_ref()
                .ok_or_else(|| Error::backend("Q2 expert model file is not initialized"))?
                .file
                .try_clone()
                .map_err(|source| Error::Io {
                    path: model_path.to_path_buf(),
                    source,
                })?;
            (layer_cache, model_file, model_path.to_path_buf())
        };

        let mut layer_cache = layer_cache
            .lock()
            .map_err(|_| Error::backend("Q2 layer expert cache lock poisoned"))?;
        let slots_per_layer = layer_cache.slots.len();
        let groups = prepare_ready_expert_groups(
            device,
            &mut layer_cache,
            gate_payloads,
            up_payloads,
            down_payloads,
            gate_stride,
            up_stride,
            down_stride,
        )?;
        let lookup_nanoseconds = elapsed_nanoseconds(lookup_started.elapsed());
        let cache_hits = groups.iter().filter(|group| group.cache_hit).count();
        let cache_misses = groups.len() - cache_hits;
        let transient_experts = groups.iter().filter(|group| group.transient).count();
        let read_bytes = groups.iter().try_fold(0_u64, |total, group| {
            group.read_tasks.iter().try_fold(total, |total, task| {
                total
                    .checked_add(u64::try_from(task.byte_len).map_err(|_| {
                        Error::backend("ready routed expert read size does not fit u64")
                    })?)
                    .ok_or_else(|| Error::backend("ready routed expert read bytes overflow"))
            })
        })?;
        let cached_miss_keys = groups
            .iter()
            .filter_map(|group| group.cached_miss_key)
            .collect::<Vec<_>>();

        let started = Instant::now();
        let execution = self.execute_ready_expert_groups(
            device,
            &model_file,
            &model_path,
            &groups,
            input,
            input_len,
            token_indices,
            token_count,
            assignment_count,
            in_features,
            intermediate_features,
            out_features,
        );
        let (output, ready_waves, first_ready_ms, ssd_load_nanoseconds, q2_matmul_gpu_nanoseconds) =
            match execution {
                Ok(output) => output,
                Err(error) => {
                    layer_cache.invalidate(&cached_miss_keys);
                    return Err(error);
                }
            };

        self.expert_cache_counters
            .lookups
            .fetch_add(groups.len() as u64, Ordering::Relaxed);
        self.expert_cache_counters
            .hits
            .fetch_add(cache_hits as u64, Ordering::Relaxed);
        self.expert_cache_counters
            .misses
            .fetch_add(cache_misses as u64, Ordering::Relaxed);
        self.expert_cache_counters
            .ssd_read_bytes
            .fetch_add(read_bytes, Ordering::Relaxed);
        self.expert_cache_counters
            .transient_experts
            .fetch_add(transient_experts as u64, Ordering::Relaxed);
        self.expert_cache_counters
            .ready_waves
            .fetch_add(ready_waves as u64, Ordering::Relaxed);
        self.expert_cache_counters
            .lookup_nanoseconds
            .fetch_add(lookup_nanoseconds, Ordering::Relaxed);
        self.expert_cache_counters
            .ssd_load_nanoseconds
            .fetch_add(ssd_load_nanoseconds, Ordering::Relaxed);
        self.expert_cache_counters
            .q2_matmul_gpu_nanoseconds
            .fetch_add(q2_matmul_gpu_nanoseconds, Ordering::Relaxed);

        debug!(
            target: "inferno::expert_cache",
            layer_index,
            selected_experts = groups.len(),
            assignment_count,
            cache_hits,
            cache_misses,
            transient_experts,
            read_bytes,
            ready_waves,
            lookup_ms = lookup_nanoseconds as f64 / 1_000_000.0,
            ssd_load_ms = ssd_load_nanoseconds as f64 / 1_000_000.0,
            q2_matmul_gpu_ms = q2_matmul_gpu_nanoseconds as f64 / 1_000_000.0,
            first_ready_ms,
            elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0,
            slots_per_layer,
            "completed ready-first Q2 routed experts"
        );

        Ok(ReadyRoutedExperts {
            output,
            selected_experts: groups.len(),
            cache_hits,
            cache_misses,
            transient_experts,
            read_bytes,
            ready_waves,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_ready_expert_groups(
        &self,
        device: &Device,
        model_file: &File,
        model_path: &Path,
        groups: &[ReadyExpertGroup],
        input: &Buffer,
        input_len: usize,
        token_indices: &Buffer,
        token_count: usize,
        assignment_count: usize,
        in_features: usize,
        intermediate_features: usize,
        out_features: usize,
    ) -> Result<(Buffer, usize, f64, u64, u64)> {
        let gated_len = assignment_count
            .checked_mul(intermediate_features)
            .ok_or_else(|| Error::backend("ready routed gated length overflow"))?;
        let output_len = assignment_count
            .checked_mul(out_features)
            .ok_or_else(|| Error::backend("ready routed output length overflow"))?;
        let gated = empty_f32_buffer(device, gated_len)?;
        let output = empty_f32_buffer(device, output_len)?;
        let cached = groups
            .iter()
            .enumerate()
            .filter_map(|(index, group)| group.cache_hit.then_some(index))
            .collect::<Vec<_>>();
        let misses = groups
            .iter()
            .enumerate()
            .filter_map(|(index, group)| (!group.cache_hit).then_some(index))
            .collect::<Vec<_>>();
        let started = Instant::now();
        let mut first_ready_ms = None;
        let mut waves = Vec::<SubmittedReadyWave>::new();

        let ssd_started = Instant::now();
        let ssd_finished_nanoseconds = AtomicU64::new(0);
        let load_result = thread::scope(|scope| -> Result<()> {
            let (sender, receiver) = mpsc::channel::<(usize, ReadyExpertPhase, Result<()>)>();
            let worker_count = misses.len().min(ROUTED_EXPERT_READ_WORKERS);
            let mut workers = Vec::with_capacity(worker_count);
            if worker_count > 0 {
                let jobs_per_worker = misses.len().div_ceil(worker_count);
                for worker_jobs in misses.chunks(jobs_per_worker) {
                    let sender = sender.clone();
                    let ssd_finished_nanoseconds = &ssd_finished_nanoseconds;
                    workers.push(scope.spawn(move || {
                        for &group_index in worker_jobs {
                            let gate_up_result = pread_ready_expert_gate_up(
                                model_file,
                                model_path,
                                &groups[group_index],
                            );
                            let gate_up_succeeded = gate_up_result.is_ok();
                            if sender
                                .send((group_index, ReadyExpertPhase::GateUp, gate_up_result))
                                .is_err()
                            {
                                return;
                            }
                            if !gate_up_succeeded {
                                continue;
                            }

                            let down_result = pread_ready_expert_down(
                                model_file,
                                model_path,
                                &groups[group_index],
                            );
                            ssd_finished_nanoseconds.fetch_max(
                                elapsed_nanoseconds(ssd_started.elapsed()),
                                Ordering::Relaxed,
                            );
                            if sender
                                .send((group_index, ReadyExpertPhase::Down, down_result))
                                .is_err()
                            {
                                return;
                            }
                        }
                    }));
                }
            }
            drop(sender);

            if !cached.is_empty() {
                first_ready_ms = Some(started.elapsed().as_secs_f64() * 1_000.0);
                waves.push(self.submit_ready_expert_wave(
                    device,
                    groups,
                    &cached,
                    input,
                    input_len,
                    token_indices,
                    token_count,
                    assignment_count,
                    in_features,
                    intermediate_features,
                    out_features,
                    &gated,
                    &output,
                    ReadyExpertPhase::GateUp,
                )?);
            }

            let mut first_error = None;
            let mut completed_gate_up = 0_usize;
            let mut completed_down = 0_usize;
            let mut pending_down = cached.clone();
            while let Ok(first) = receiver.recv() {
                let mut completed = vec![first];
                completed.extend(receiver.try_iter());
                let mut pending_gate_up = Vec::new();
                for (group_index, phase, result) in completed {
                    match phase {
                        ReadyExpertPhase::GateUp => {
                            completed_gate_up += 1;
                            match result {
                                Ok(()) => pending_gate_up.push(group_index),
                                Err(error) if first_error.is_none() => first_error = Some(error),
                                Err(_) => {}
                            }
                        }
                        ReadyExpertPhase::Down => {
                            completed_down += 1;
                            match result {
                                Ok(()) => pending_down.push(group_index),
                                Err(error) if first_error.is_none() => first_error = Some(error),
                                Err(_) => {}
                            }
                        }
                    }
                }
                if !pending_gate_up.is_empty() {
                    first_ready_ms.get_or_insert_with(|| started.elapsed().as_secs_f64() * 1_000.0);
                    waves.push(self.submit_ready_expert_wave(
                        device,
                        groups,
                        &pending_gate_up,
                        input,
                        input_len,
                        token_indices,
                        token_count,
                        assignment_count,
                        in_features,
                        intermediate_features,
                        out_features,
                        &gated,
                        &output,
                        ReadyExpertPhase::GateUp,
                    )?);
                }
                if completed_gate_up == misses.len() && !pending_down.is_empty() {
                    waves.push(self.submit_ready_expert_wave(
                        device,
                        groups,
                        &pending_down,
                        input,
                        input_len,
                        token_indices,
                        token_count,
                        assignment_count,
                        in_features,
                        intermediate_features,
                        out_features,
                        &gated,
                        &output,
                        ReadyExpertPhase::Down,
                    )?);
                    pending_down.clear();
                }
            }

            if !pending_down.is_empty() {
                waves.push(self.submit_ready_expert_wave(
                    device,
                    groups,
                    &pending_down,
                    input,
                    input_len,
                    token_indices,
                    token_count,
                    assignment_count,
                    in_features,
                    intermediate_features,
                    out_features,
                    &gated,
                    &output,
                    ReadyExpertPhase::Down,
                )?);
            }

            for worker in workers {
                worker
                    .join()
                    .map_err(|_| Error::backend("Q2 expert read worker thread panicked"))?;
            }
            validate_exact_len(
                "ready routed completed gate/up SSD reads",
                completed_gate_up,
                misses.len(),
            )?;
            if first_error.is_none() {
                validate_exact_len(
                    "ready routed completed down SSD reads",
                    completed_down,
                    misses.len(),
                )?;
            }
            if let Some(error) = first_error {
                return Err(error);
            }
            Ok(())
        });
        let ssd_load_nanoseconds = if misses.is_empty() {
            0
        } else {
            ssd_finished_nanoseconds.load(Ordering::Relaxed)
        };
        let wave_result = wait_ready_expert_waves(&waves);
        let q2_matmul_gpu_nanoseconds = match (load_result, wave_result) {
            (Ok(()), Ok(nanoseconds)) => nanoseconds,
            (Err(error), Ok(_)) => return Err(error),
            (Ok(()), Err(error)) => return Err(error),
            (Err(error), Err(wave_error)) => {
                debug!(
                    target: "inferno::expert_cache",
                    error = %wave_error,
                    "ready expert Metal wave also failed while handling an SSD load error"
                );
                return Err(error);
            }
        };
        Ok((
            output,
            waves.len(),
            first_ready_ms.unwrap_or_default(),
            ssd_load_nanoseconds,
            q2_matmul_gpu_nanoseconds,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_ready_expert_wave(
        &self,
        device: &Device,
        groups: &[ReadyExpertGroup],
        ready_groups: &[usize],
        input: &Buffer,
        input_len: usize,
        token_indices: &Buffer,
        token_count: usize,
        assignment_count: usize,
        in_features: usize,
        intermediate_features: usize,
        out_features: usize,
        gated: &Buffer,
        output: &Buffer,
        phase: ReadyExpertPhase,
    ) -> Result<SubmittedReadyWave> {
        if !ready_groups.is_empty()
            && ready_groups.iter().all(|&group_index| {
                groups
                    .get(group_index)
                    .and_then(|group| group.buffers.slot_index)
                    .is_some()
            })
        {
            return self.submit_ready_expert_slot_wave(
                device,
                groups,
                ready_groups,
                input,
                input_len,
                token_indices,
                token_count,
                assignment_count,
                in_features,
                intermediate_features,
                out_features,
                gated,
                output,
                phase,
            );
        }

        let mut assignment_indices = Vec::<u32>::new();
        let mut gate_addresses = Vec::<u64>::new();
        let mut up_addresses = Vec::<u64>::new();
        let mut down_addresses = Vec::<u64>::new();
        let mut gate_resources = Vec::<&Buffer>::new();
        let mut up_resources = Vec::<&Buffer>::new();
        let mut down_resources = Vec::<&Buffer>::new();
        for &group_index in ready_groups {
            let group = groups.get(group_index).ok_or_else(|| {
                Error::backend(format!(
                    "ready routed group index {group_index} is out of bounds"
                ))
            })?;
            for &assignment_index in &group.assignment_indices {
                assignment_indices.push(u32::try_from(assignment_index).map_err(|_| {
                    Error::backend("ready routed assignment index exceeds Metal u32 limit")
                })?);
                gate_addresses.push(group.buffers.gate.gpu_address());
                up_addresses.push(group.buffers.up.gpu_address());
                down_addresses.push(group.buffers.down.gpu_address());
                gate_resources.push(&group.buffers.gate);
                up_resources.push(&group.buffers.up);
                down_resources.push(&group.buffers.down);
            }
        }
        if assignment_indices.is_empty() {
            return Err(Error::backend(
                "ready routed wave requires at least one assignment",
            ));
        }
        if gate_addresses.iter().any(|address| *address == 0)
            || up_addresses.iter().any(|address| *address == 0)
            || down_addresses.iter().any(|address| *address == 0)
        {
            return Err(Error::backend(
                "ready routed expert buffers require non-zero GPU addresses",
            ));
        }

        let gate_addresses = u64_buffer(device, &gate_addresses)?;
        let up_addresses = u64_buffer(device, &up_addresses)?;
        let down_addresses = u64_buffer(device, &down_addresses)?;
        let assignment_indices_buffer = u32_buffer(device, &assignment_indices)?;
        let command_buffer = self.expert_queue.new_command_buffer().to_owned();
        match phase {
            ReadyExpertPhase::GateUp => self.encode_ready_gate_up_swiglu(
                &command_buffer,
                device,
                &gate_addresses,
                &up_addresses,
                &gate_resources,
                &up_resources,
                input,
                input_len,
                token_indices,
                &assignment_indices_buffer,
                assignment_indices.len(),
                token_count,
                assignment_count,
                in_features,
                intermediate_features,
                gated,
            )?,
            ReadyExpertPhase::Down => self.encode_ready_matvec(
                &command_buffer,
                device,
                &down_addresses,
                &down_resources,
                gated,
                &assignment_indices_buffer,
                assignment_indices.len(),
                assignment_count,
                intermediate_features,
                out_features,
                output,
            )?,
        }
        command_buffer.commit();

        Ok(SubmittedReadyWave {
            command_buffer,
            assignment_count: assignment_indices.len(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_ready_expert_slot_wave(
        &self,
        device: &Device,
        groups: &[ReadyExpertGroup],
        ready_groups: &[usize],
        input: &Buffer,
        input_len: usize,
        token_indices: &Buffer,
        token_count: usize,
        assignment_count: usize,
        in_features: usize,
        intermediate_features: usize,
        out_features: usize,
        gated: &Buffer,
        output: &Buffer,
        phase: ReadyExpertPhase,
    ) -> Result<SubmittedReadyWave> {
        let first = groups
            .get(
                *ready_groups
                    .first()
                    .ok_or_else(|| Error::backend("ready routed slot wave has no expert groups"))?,
            )
            .ok_or_else(|| Error::backend("ready routed slot wave group is out of bounds"))?;
        let mut assignment_indices = Vec::<u32>::new();
        let mut slot_indices = Vec::<u32>::new();
        for &group_index in ready_groups {
            let group = groups.get(group_index).ok_or_else(|| {
                Error::backend(format!(
                    "ready routed slot group index {group_index} is out of bounds"
                ))
            })?;
            if group.buffers.gate.gpu_address() != first.buffers.gate.gpu_address()
                || group.buffers.up.gpu_address() != first.buffers.up.gpu_address()
                || group.buffers.down.gpu_address() != first.buffers.down.gpu_address()
            {
                return Err(Error::backend(
                    "ready routed slot wave mixes different layer slabs",
                ));
            }
            let slot = u32::try_from(group.buffers.slot_index.ok_or_else(|| {
                Error::backend("ready routed slot wave contains a transient expert")
            })?)
            .map_err(|_| Error::backend("ready routed expert slot exceeds Metal u32 limit"))?;
            for &assignment_index in &group.assignment_indices {
                assignment_indices.push(u32::try_from(assignment_index).map_err(|_| {
                    Error::backend("ready routed assignment index exceeds Metal u32 limit")
                })?);
                slot_indices.push(slot);
            }
        }
        if assignment_indices.is_empty() {
            return Err(Error::backend(
                "ready routed slot wave requires at least one assignment",
            ));
        }

        let gate_stride = q2_expert_stride(in_features, intermediate_features)?;
        let up_stride = gate_stride;
        let down_stride = q2_expert_stride(intermediate_features, out_features)?;
        let slot_count = (first.buffers.gate.length() as usize) / gate_stride;
        validate_exact_len(
            "ready routed up slab byte length",
            first.buffers.up.length() as usize,
            slot_count
                .checked_mul(up_stride)
                .ok_or_else(|| Error::backend("ready routed up slab byte count overflow"))?,
        )?;
        validate_exact_len(
            "ready routed down slab byte length",
            first.buffers.down.length() as usize,
            slot_count
                .checked_mul(down_stride)
                .ok_or_else(|| Error::backend("ready routed down slab byte count overflow"))?,
        )?;
        if slot_indices.iter().any(|&slot| slot as usize >= slot_count) {
            return Err(Error::backend(
                "ready routed expert slot is outside the layer slab",
            ));
        }

        let assignment_indices_buffer = u32_buffer(device, &assignment_indices)?;
        let slot_indices_buffer = u32_buffer(device, &slot_indices)?;
        let command_buffer = self.expert_queue.new_command_buffer().to_owned();
        match phase {
            ReadyExpertPhase::GateUp => self.encode_ready_slot_gate_up_swiglu(
                &command_buffer,
                device,
                &first.buffers.gate,
                &first.buffers.up,
                input,
                input_len,
                token_indices,
                &assignment_indices_buffer,
                &slot_indices_buffer,
                assignment_indices.len(),
                token_count,
                assignment_count,
                slot_count,
                in_features,
                intermediate_features,
                gate_stride,
                gated,
            )?,
            ReadyExpertPhase::Down => self.encode_ready_slot_matvec(
                &command_buffer,
                device,
                &first.buffers.down,
                gated,
                &assignment_indices_buffer,
                &slot_indices_buffer,
                assignment_indices.len(),
                assignment_count,
                slot_count,
                intermediate_features,
                out_features,
                down_stride,
                output,
            )?,
        }
        command_buffer.commit();

        Ok(SubmittedReadyWave {
            command_buffer,
            assignment_count: assignment_indices.len(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_ready_gate_up_swiglu(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        gate_addresses: &Buffer,
        up_addresses: &Buffer,
        gate_resources: &[&Buffer],
        up_resources: &[&Buffer],
        input: &Buffer,
        input_len: usize,
        token_indices: &Buffer,
        assignment_indices: &Buffer,
        ready_count: usize,
        token_count: usize,
        assignment_count: usize,
        in_features: usize,
        out_features: usize,
        output: &Buffer,
    ) -> Result<()> {
        require_f32_capacity(input, input_len, "ready routed gate/up input")?;
        require_u32_buffer_capacity(
            token_indices,
            assignment_count,
            "ready routed gate/up token indices",
        )?;
        require_u32_buffer_capacity(
            assignment_indices,
            ready_count,
            "ready routed gate/up assignment indices",
        )?;
        require_f32_capacity(
            output,
            assignment_count
                .checked_mul(out_features)
                .ok_or_else(|| Error::backend("ready routed gate/up output overflow"))?,
            "ready routed gate/up output",
        )?;
        let physical_threads = q2_k_cooperative_threads(
            &self.ready_gate_up_swiglu_pipeline,
            ready_count
                .checked_mul(out_features)
                .ok_or_else(|| Error::backend("ready routed gate/up thread count overflow"))?,
            "ready Q2_K routed gate/up",
        )?;
        let blocks_per_row_value = in_features / Q2_K_BLOCK_VALUES;
        let token_count = u32_scalar_buffer(device, matvec_u32(token_count, "token_count")?)?;
        let assignment_count =
            u32_scalar_buffer(device, matvec_u32(assignment_count, "assignment_count")?)?;
        let ready_count = u32_scalar_buffer(device, matvec_u32(ready_count, "ready_count")?)?;
        let in_features = u32_scalar_buffer(device, matvec_u32(in_features, "in_features")?)?;
        let out_features = u32_scalar_buffer(device, matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row =
            u32_scalar_buffer(device, matvec_u32(blocks_per_row_value, "blocks_per_row")?)?;
        let indirect_reads = gate_resources
            .iter()
            .chain(up_resources)
            .copied()
            .collect::<Vec<_>>();
        encode_1d_with_indirect_reads(
            command_buffer,
            &self.ready_gate_up_swiglu_pipeline,
            &[
                gate_addresses,
                up_addresses,
                input,
                token_indices,
                assignment_indices,
                output,
                &token_count,
                &assignment_count,
                &ready_count,
                &in_features,
                &out_features,
                &blocks_per_row,
            ],
            &indirect_reads,
            physical_threads,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_ready_matvec(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        weight_addresses: &Buffer,
        weight_resources: &[&Buffer],
        input: &Buffer,
        assignment_indices: &Buffer,
        ready_count: usize,
        assignment_count: usize,
        in_features: usize,
        out_features: usize,
        output: &Buffer,
    ) -> Result<()> {
        require_u32_buffer_capacity(
            assignment_indices,
            ready_count,
            "ready routed down assignment indices",
        )?;
        require_f32_capacity(
            input,
            assignment_count
                .checked_mul(in_features)
                .ok_or_else(|| Error::backend("ready routed down input overflow"))?,
            "ready routed down input",
        )?;
        require_f32_capacity(
            output,
            assignment_count
                .checked_mul(out_features)
                .ok_or_else(|| Error::backend("ready routed down output overflow"))?,
            "ready routed down output",
        )?;
        let physical_threads = q2_k_cooperative_threads(
            &self.ready_matvec_pipeline,
            ready_count
                .checked_mul(out_features)
                .ok_or_else(|| Error::backend("ready routed down thread count overflow"))?,
            "ready Q2_K routed down",
        )?;
        let blocks_per_row_value = in_features / Q2_K_BLOCK_VALUES;
        let assignment_count =
            u32_scalar_buffer(device, matvec_u32(assignment_count, "assignment_count")?)?;
        let ready_count = u32_scalar_buffer(device, matvec_u32(ready_count, "ready_count")?)?;
        let in_features = u32_scalar_buffer(device, matvec_u32(in_features, "in_features")?)?;
        let out_features = u32_scalar_buffer(device, matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row =
            u32_scalar_buffer(device, matvec_u32(blocks_per_row_value, "blocks_per_row")?)?;
        encode_1d_with_indirect_reads(
            command_buffer,
            &self.ready_matvec_pipeline,
            &[
                weight_addresses,
                input,
                assignment_indices,
                output,
                &assignment_count,
                &ready_count,
                &in_features,
                &out_features,
                &blocks_per_row,
            ],
            weight_resources,
            physical_threads,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_ready_slot_gate_up_swiglu(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        gate_weights: &Buffer,
        up_weights: &Buffer,
        input: &Buffer,
        input_len: usize,
        token_indices: &Buffer,
        assignment_indices: &Buffer,
        slot_indices: &Buffer,
        ready_count: usize,
        token_count: usize,
        assignment_count: usize,
        slot_count: usize,
        in_features: usize,
        out_features: usize,
        expert_stride_bytes: usize,
        output: &Buffer,
    ) -> Result<()> {
        let slab_bytes = slot_count
            .checked_mul(expert_stride_bytes)
            .ok_or_else(|| Error::backend("ready routed gate/up slab byte count overflow"))?;
        require_byte_capacity(gate_weights, slab_bytes, "ready routed gate slab")?;
        require_byte_capacity(up_weights, slab_bytes, "ready routed up slab")?;
        require_f32_capacity(input, input_len, "ready routed slot gate/up input")?;
        require_u32_buffer_capacity(
            token_indices,
            assignment_count,
            "ready routed slot token indices",
        )?;
        require_u32_buffer_capacity(
            assignment_indices,
            ready_count,
            "ready routed slot assignment indices",
        )?;
        require_u32_buffer_capacity(
            slot_indices,
            ready_count,
            "ready routed expert slot indices",
        )?;
        require_f32_capacity(
            output,
            assignment_count
                .checked_mul(out_features)
                .ok_or_else(|| Error::backend("ready routed slot gate/up output overflow"))?,
            "ready routed slot gate/up output",
        )?;
        let physical_threads = q2_k_cooperative_threads(
            &self.ready_slot_gate_up_swiglu_pipeline,
            ready_count
                .checked_mul(out_features)
                .ok_or_else(|| Error::backend("ready routed slot gate/up threads overflow"))?,
            "ready Q2_K slot gate/up",
        )?;
        let blocks_per_row_value = in_features / Q2_K_BLOCK_VALUES;
        let token_count = u32_scalar_buffer(device, matvec_u32(token_count, "token_count")?)?;
        let assignment_count =
            u32_scalar_buffer(device, matvec_u32(assignment_count, "assignment_count")?)?;
        let ready_count = u32_scalar_buffer(device, matvec_u32(ready_count, "ready_count")?)?;
        let in_features = u32_scalar_buffer(device, matvec_u32(in_features, "in_features")?)?;
        let out_features = u32_scalar_buffer(device, matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row =
            u32_scalar_buffer(device, matvec_u32(blocks_per_row_value, "blocks_per_row")?)?;
        let expert_stride_bytes = u32_scalar_buffer(
            device,
            matvec_u32(expert_stride_bytes, "expert_stride_bytes")?,
        )?;
        let slot_count = u32_scalar_buffer(device, matvec_u32(slot_count, "slot_count")?)?;
        encode_1d(
            command_buffer,
            &self.ready_slot_gate_up_swiglu_pipeline,
            &[
                gate_weights,
                up_weights,
                input,
                token_indices,
                assignment_indices,
                slot_indices,
                output,
                &token_count,
                &assignment_count,
                &ready_count,
                &in_features,
                &out_features,
                &blocks_per_row,
                &expert_stride_bytes,
                &slot_count,
            ],
            physical_threads,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_ready_slot_matvec(
        &self,
        command_buffer: &CommandBufferRef,
        device: &Device,
        weights: &Buffer,
        input: &Buffer,
        assignment_indices: &Buffer,
        slot_indices: &Buffer,
        ready_count: usize,
        assignment_count: usize,
        slot_count: usize,
        in_features: usize,
        out_features: usize,
        expert_stride_bytes: usize,
        output: &Buffer,
    ) -> Result<()> {
        require_byte_capacity(
            weights,
            slot_count
                .checked_mul(expert_stride_bytes)
                .ok_or_else(|| Error::backend("ready routed down slab byte count overflow"))?,
            "ready routed down slab",
        )?;
        require_u32_buffer_capacity(
            assignment_indices,
            ready_count,
            "ready routed slot down assignment indices",
        )?;
        require_u32_buffer_capacity(
            slot_indices,
            ready_count,
            "ready routed slot down expert indices",
        )?;
        require_f32_capacity(
            input,
            assignment_count
                .checked_mul(in_features)
                .ok_or_else(|| Error::backend("ready routed slot down input overflow"))?,
            "ready routed slot down input",
        )?;
        require_f32_capacity(
            output,
            assignment_count
                .checked_mul(out_features)
                .ok_or_else(|| Error::backend("ready routed slot down output overflow"))?,
            "ready routed slot down output",
        )?;
        let physical_threads = q2_k_cooperative_threads(
            &self.ready_slot_matvec_pipeline,
            ready_count
                .checked_mul(out_features)
                .ok_or_else(|| Error::backend("ready routed slot down threads overflow"))?,
            "ready Q2_K slot down",
        )?;
        let blocks_per_row_value = in_features / Q2_K_BLOCK_VALUES;
        let assignment_count =
            u32_scalar_buffer(device, matvec_u32(assignment_count, "assignment_count")?)?;
        let ready_count = u32_scalar_buffer(device, matvec_u32(ready_count, "ready_count")?)?;
        let in_features = u32_scalar_buffer(device, matvec_u32(in_features, "in_features")?)?;
        let out_features = u32_scalar_buffer(device, matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row =
            u32_scalar_buffer(device, matvec_u32(blocks_per_row_value, "blocks_per_row")?)?;
        let expert_stride_bytes = u32_scalar_buffer(
            device,
            matvec_u32(expert_stride_bytes, "expert_stride_bytes")?,
        )?;
        let slot_count = u32_scalar_buffer(device, matvec_u32(slot_count, "slot_count")?)?;
        encode_1d(
            command_buffer,
            &self.ready_slot_matvec_pipeline,
            &[
                weights,
                input,
                assignment_indices,
                slot_indices,
                output,
                &assignment_count,
                &ready_count,
                &in_features,
                &out_features,
                &blocks_per_row,
                &expert_stride_bytes,
                &slot_count,
            ],
            physical_threads,
        )
    }

    fn weight_buffer(&self, device: &Device, weights: &[u8]) -> Result<WeightBuffer> {
        let key = WeightBufferKey {
            address: weights.as_ptr() as usize,
            byte_len: weights.len(),
        };
        let mut buffers = self
            .weight_buffers
            .lock()
            .map_err(|_| Error::backend("Metal weight buffer cache lock poisoned"))?;
        if let Some(buffer) = buffers.get(&key) {
            return Ok(WeightBuffer {
                buffer: buffer.clone(),
            });
        }

        let buffer = u8_buffer_no_copy(device, weights)?;
        buffers.insert(key, buffer.clone());
        Ok(WeightBuffer { buffer })
    }
}

impl LockedMetalBuffer {
    fn new(device: &Device, byte_len: usize, component: &str) -> Result<Self> {
        let buffer = empty_u8_buffer(device, byte_len)?;
        let pointer = buffer.contents();
        let lock_disabled = EXPERT_CACHE_LOCK_WARNING_EMITTED.load(Ordering::Relaxed);
        let locked = if pointer.is_null() || lock_disabled {
            false
        } else {
            // SAFETY: `pointer` identifies the shared Metal buffer's CPU-visible
            // allocation and remains valid for the lifetime of `buffer`.
            unsafe { libc::mlock(pointer.cast_const(), byte_len) == 0 }
        };
        if !locked
            && !lock_disabled
            && !EXPERT_CACHE_LOCK_WARNING_EMITTED.swap(true, Ordering::Relaxed)
        {
            warn!(
                target: "inferno::expert_cache",
                component,
                byte_len,
                error = %io::Error::last_os_error(),
                "could not lock Q2 expert cache buffer; entry remains pageable"
            );
        }
        Ok(Self { buffer, locked })
    }
}

impl Q2ExpertSlotBuffers {
    fn new(
        device: &Device,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
    ) -> Result<Self> {
        Ok(Self {
            gate: LockedMetalBuffer::new(device, gate_stride, "gate")?,
            up: LockedMetalBuffer::new(device, up_stride, "up")?,
            down: LockedMetalBuffer::new(device, down_stride, "down")?,
        })
    }

    fn ready_buffers(&self) -> ReadyExpertBuffers {
        ReadyExpertBuffers {
            gate: self.gate.buffer.clone(),
            up: self.up.buffer.clone(),
            down: self.down.buffer.clone(),
            gate_offset: 0,
            up_offset: 0,
            down_offset: 0,
            slot_index: None,
        }
    }
}

impl Q2ExpertLayerBuffers {
    fn new(
        device: &Device,
        capacity: usize,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
    ) -> Result<Self> {
        let slab_bytes = |stride: usize, component: &str| {
            stride.checked_mul(capacity).ok_or_else(|| {
                Error::backend(format!("Q2 {component} expert slab byte count overflow"))
            })
        };
        Ok(Self {
            gate: LockedMetalSlab::new(device, slab_bytes(gate_stride, "gate")?)?,
            up: LockedMetalSlab::new(device, slab_bytes(up_stride, "up")?)?,
            down: LockedMetalSlab::new(device, slab_bytes(down_stride, "down")?)?,
        })
    }

    fn ready_buffers(
        &mut self,
        slot: usize,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
    ) -> Result<ReadyExpertBuffers> {
        let gate_offset = slot
            .checked_mul(gate_stride)
            .ok_or_else(|| Error::backend("Q2 gate expert slab offset overflow"))?;
        let up_offset = slot
            .checked_mul(up_stride)
            .ok_or_else(|| Error::backend("Q2 up expert slab offset overflow"))?;
        let down_offset = slot
            .checked_mul(down_stride)
            .ok_or_else(|| Error::backend("Q2 down expert slab offset overflow"))?;
        self.gate.lock_range(gate_offset, gate_stride, "gate")?;
        self.up.lock_range(up_offset, up_stride, "up")?;
        self.down.lock_range(down_offset, down_stride, "down")?;
        Ok(ReadyExpertBuffers {
            gate: self.gate.buffer.clone(),
            up: self.up.buffer.clone(),
            down: self.down.buffer.clone(),
            gate_offset,
            up_offset,
            down_offset,
            slot_index: Some(slot),
        })
    }
}

impl LockedMetalSlab {
    fn new(device: &Device, byte_len: usize) -> Result<Self> {
        Ok(Self {
            buffer: empty_u8_buffer(device, byte_len)?,
            locked_ranges: Vec::new(),
        })
    }

    fn lock_range(&mut self, offset: usize, byte_len: usize, component: &str) -> Result<()> {
        let end = offset
            .checked_add(byte_len)
            .ok_or_else(|| Error::backend("Q2 expert slab lock range overflow"))?;
        if end > self.buffer.length() as usize {
            return Err(Error::backend(format!(
                "Q2 {component} expert slab lock ends at {end}, beyond {} bytes",
                self.buffer.length()
            )));
        }
        if self.locked_ranges.contains(&(offset, byte_len)) {
            return Ok(());
        }

        let pointer = self.buffer.contents().cast::<u8>();
        let lock_disabled = EXPERT_CACHE_LOCK_WARNING_EMITTED.load(Ordering::Relaxed);
        let locked = if pointer.is_null() || lock_disabled {
            false
        } else {
            // SAFETY: the validated range belongs to this live shared Metal
            // allocation and remains valid until the slab is dropped.
            unsafe { libc::mlock(pointer.add(offset).cast_const().cast(), byte_len) == 0 }
        };
        if locked {
            self.locked_ranges.push((offset, byte_len));
        } else if !lock_disabled && !EXPERT_CACHE_LOCK_WARNING_EMITTED.swap(true, Ordering::Relaxed)
        {
            warn!(
                target: "inferno::expert_cache",
                component,
                byte_len,
                error = %io::Error::last_os_error(),
                "could not lock Q2 expert slab range; entry remains pageable"
            );
        }
        Ok(())
    }
}

impl Drop for LockedMetalBuffer {
    fn drop(&mut self) {
        if !self.locked {
            return;
        }
        let pointer = self.buffer.contents();
        if !pointer.is_null() {
            // SAFETY: this unlocks the same live Metal allocation locked in
            // `new`; no CPU pointer is dereferenced.
            let _ = unsafe { libc::munlock(pointer.cast_const(), self.buffer.length() as usize) };
        }
    }
}

impl Drop for LockedMetalSlab {
    fn drop(&mut self) {
        let pointer = self.buffer.contents().cast::<u8>();
        if pointer.is_null() {
            return;
        }
        for &(offset, byte_len) in &self.locked_ranges {
            // SAFETY: each range was locked by `lock_range` against this live
            // allocation and no range is recorded twice.
            let _ = unsafe { libc::munlock(pointer.add(offset).cast_const().cast(), byte_len) };
        }
    }
}

fn uniform_payload_stride(payloads: &[Q2ExpertSource<'_>], label: &str) -> Result<usize> {
    let stride = payloads
        .first()
        .map(|payload| payload.bytes.len())
        .ok_or_else(|| Error::backend(format!("{label} has no payloads")))?;
    if stride == 0 {
        return Err(Error::backend(format!("{label} payloads cannot be empty")));
    }
    for (index, payload) in payloads.iter().enumerate() {
        if payload.bytes.len() != stride {
            return Err(Error::backend(format!(
                "{label} payload {index} has {} bytes, expected {stride}",
                payload.bytes.len()
            )));
        }
    }
    Ok(stride)
}

fn q2_expert_stride(in_features: usize, out_features: usize) -> Result<usize> {
    if in_features == 0 || out_features == 0 || in_features % Q2_K_BLOCK_VALUES != 0 {
        return Err(Error::backend(format!(
            "Q2 staged expert shape requires positive features and input divisible by {Q2_K_BLOCK_VALUES}, got in_features={in_features}, out_features={out_features}"
        )));
    }
    out_features
        .checked_mul(in_features / Q2_K_BLOCK_VALUES)
        .and_then(|blocks| blocks.checked_mul(Q2_K_BLOCK_BYTES))
        .ok_or_else(|| Error::backend("Q2 staged expert stride overflow"))
}

fn expert_cache_key(gate: &[u8], up: &[u8], down: &[u8]) -> ExpertCacheKey {
    ExpertCacheKey {
        gate_address: gate.as_ptr() as usize,
        gate_bytes: gate.len(),
        up_address: up.as_ptr() as usize,
        up_bytes: up.len(),
        down_address: down.as_ptr() as usize,
        down_bytes: down.len(),
    }
}

impl Q2PerLayerExpertCache {
    fn resize(&mut self, slots_per_layer: usize) -> Result<()> {
        for layer in self.layers.values() {
            layer
                .lock()
                .map_err(|_| Error::backend("Q2 layer expert cache lock poisoned"))?
                .resize(slots_per_layer)?;
        }
        self.slots_per_layer = slots_per_layer;
        Ok(())
    }

    fn ensure_model_file(&mut self, path: &Path) -> Result<()> {
        if self
            .model_file
            .as_ref()
            .is_some_and(|model_file| model_file.path == path)
        {
            return Ok(());
        }
        let file = File::open(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        self.model_file = Some(ExpertModelFile {
            path: path.to_path_buf(),
            file,
        });
        self.layers.clear();
        Ok(())
    }

    fn ensure_layout(&mut self, gate_stride: usize, up_stride: usize, down_stride: usize) {
        if self.layout.as_ref().is_some_and(|layout| {
            layout.gate_stride == gate_stride
                && layout.up_stride == up_stride
                && layout.down_stride == down_stride
        }) {
            return;
        }
        self.layout = Some(Q2ExpertCacheLayout {
            gate_stride,
            up_stride,
            down_stride,
        });
        self.layers.clear();
    }

    fn layer(&mut self, layer_index: usize) -> Arc<Mutex<Q2ExpertLayerCache>> {
        let slots_per_layer = self.slots_per_layer;
        Arc::clone(
            self.layers
                .entry(layer_index)
                .or_insert_with(|| Arc::new(Mutex::new(Q2ExpertLayerCache::new(slots_per_layer)))),
        )
    }
}

impl Q2ExpertLayerCache {
    fn new(capacity: usize) -> Self {
        Self {
            slots: vec![false; capacity],
            storage: None,
            entries: HashMap::new(),
            probation: VecDeque::new(),
            protected: VecDeque::new(),
            protected_capacity: capacity.div_ceil(2),
            free_slots: (0..capacity).rev().collect(),
        }
    }

    fn resize(&mut self, capacity: usize) -> Result<()> {
        if capacity == 0 {
            return Err(Error::backend(
                "Q2 expert layer cache capacity must be positive",
            ));
        }
        let old_capacity = self.slots.len();
        if capacity == old_capacity {
            return Ok(());
        }
        // Slab offsets depend on capacity. Rebalancing is infrequent, so a
        // safe cold reset is simpler than copying gigabytes between slabs.
        *self = Self::new(capacity);
        Ok(())
    }

    fn resolve_slot(
        &mut self,
        key: ExpertCacheKey,
        selected_keys: &HashSet<ExpertCacheKey>,
    ) -> Result<LayerSlotResolution> {
        if let Some(&slot) = self.entries.get(&key) {
            self.touch(key);
            return Ok(LayerSlotResolution::Hit(slot));
        }

        let slot = match self.free_slots.pop() {
            Some(slot) => slot,
            None => {
                let Some(evicted) = self.take_evictable(selected_keys) else {
                    return Ok(LayerSlotResolution::Transient);
                };
                self.entries.remove(&evicted).ok_or_else(|| {
                    Error::backend("Q2 layer expert LRU and entry map are inconsistent")
                })?
            }
        };
        self.entries.insert(key, slot);
        self.insert_new(key);
        Ok(LayerSlotResolution::Miss(slot))
    }

    fn buffers(
        &mut self,
        device: &Device,
        slot: usize,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
    ) -> Result<ReadyExpertBuffers> {
        let capacity = self.slots.len();
        let cache_slot = self.slots.get_mut(slot).ok_or_else(|| {
            Error::backend(format!("Q2 layer expert slot {slot} is out of bounds"))
        })?;
        if self.storage.is_none() {
            self.storage = Some(Q2ExpertLayerBuffers::new(
                device,
                capacity,
                gate_stride,
                up_stride,
                down_stride,
            )?);
        }
        let buffers = self
            .storage
            .as_mut()
            .ok_or_else(|| Error::backend("Q2 layer expert slab allocation failed"))?
            .ready_buffers(slot, gate_stride, up_stride, down_stride)?;
        *cache_slot = true;
        Ok(buffers)
    }

    fn touch(&mut self, key: ExpertCacheKey) {
        if remove_key(&mut self.probation, key) {
            self.protected.push_back(key);
            if self.protected.len() > self.protected_capacity {
                if let Some(demoted) = self.protected.pop_front() {
                    self.probation.push_back(demoted);
                }
            }
        } else if remove_key(&mut self.protected, key) {
            self.protected.push_back(key);
        }
    }

    fn insert_new(&mut self, key: ExpertCacheKey) {
        self.probation.push_back(key);
    }

    fn take_evictable(
        &mut self,
        selected_keys: &HashSet<ExpertCacheKey>,
    ) -> Option<ExpertCacheKey> {
        take_unselected(&mut self.probation, selected_keys)
            .or_else(|| take_unselected(&mut self.protected, selected_keys))
    }

    fn invalidate(&mut self, keys: &[ExpertCacheKey]) {
        for key in keys {
            let Some(slot) = self.entries.remove(key) else {
                continue;
            };
            remove_key(&mut self.probation, *key);
            remove_key(&mut self.protected, *key);
            if !self.free_slots.contains(&slot) {
                self.free_slots.push(slot);
            }
        }
    }
}

fn remove_key(queue: &mut VecDeque<ExpertCacheKey>, key: ExpertCacheKey) -> bool {
    let Some(index) = queue.iter().position(|candidate| *candidate == key) else {
        return false;
    };
    queue.remove(index);
    true
}

fn take_unselected(
    queue: &mut VecDeque<ExpertCacheKey>,
    selected_keys: &HashSet<ExpertCacheKey>,
) -> Option<ExpertCacheKey> {
    let index = queue
        .iter()
        .position(|candidate| !selected_keys.contains(candidate))?;
    queue.remove(index)
}

fn prepare_ready_expert_groups<'a>(
    device: &Device,
    cache: &mut Q2ExpertLayerCache,
    gate_payloads: &[Q2ExpertSource<'a>],
    up_payloads: &[Q2ExpertSource<'a>],
    down_payloads: &[Q2ExpertSource<'a>],
    gate_stride: usize,
    up_stride: usize,
    down_stride: usize,
) -> Result<Vec<ReadyExpertGroup>> {
    let mut seed_indices = HashMap::<ExpertCacheKey, usize>::new();
    let mut seeds = Vec::<ReadyExpertSeed<'a>>::new();
    for assignment_index in 0..gate_payloads.len() {
        let gate = gate_payloads[assignment_index];
        let up = up_payloads[assignment_index];
        let down = down_payloads[assignment_index];
        let key = expert_cache_key(gate.bytes, up.bytes, down.bytes);
        if let Some(&seed_index) = seed_indices.get(&key) {
            let seed = &mut seeds[seed_index];
            if seed.gate.absolute_offset != gate.absolute_offset
                || seed.up.absolute_offset != up.absolute_offset
                || seed.down.absolute_offset != down.absolute_offset
            {
                return Err(Error::backend(
                    "duplicate Q2 expert payload addresses use inconsistent file offsets",
                ));
            }
            seed.assignment_indices.push(assignment_index);
            continue;
        }
        seed_indices.insert(key, seeds.len());
        seeds.push(ReadyExpertSeed {
            key,
            assignment_indices: vec![assignment_index],
            gate,
            up,
            down,
        });
    }

    let selected_keys = seeds.iter().map(|seed| seed.key).collect::<HashSet<_>>();
    let mut groups = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let resolution = cache.resolve_slot(seed.key, &selected_keys)?;
        let (buffers, cache_hit, cached_miss_key, transient, transient_owner) = match resolution {
            LayerSlotResolution::Hit(slot) => (
                cache.buffers(device, slot, gate_stride, up_stride, down_stride)?,
                true,
                None,
                false,
                None,
            ),
            LayerSlotResolution::Miss(slot) => (
                cache.buffers(device, slot, gate_stride, up_stride, down_stride)?,
                false,
                Some(seed.key),
                false,
                None,
            ),
            LayerSlotResolution::Transient => {
                let slot = Q2ExpertSlotBuffers::new(device, gate_stride, up_stride, down_stride)?;
                (slot.ready_buffers(), false, None, true, Some(slot))
            }
        };
        let read_tasks = if cache_hit {
            Vec::new()
        } else {
            vec![
                ExpertReadTask {
                    buffer: buffers.gate.clone(),
                    destination_offset: buffers.gate_offset,
                    absolute_offset: seed.gate.absolute_offset,
                    byte_len: seed.gate.bytes.len(),
                },
                ExpertReadTask {
                    buffer: buffers.up.clone(),
                    destination_offset: buffers.up_offset,
                    absolute_offset: seed.up.absolute_offset,
                    byte_len: seed.up.bytes.len(),
                },
                ExpertReadTask {
                    buffer: buffers.down.clone(),
                    destination_offset: buffers.down_offset,
                    absolute_offset: seed.down.absolute_offset,
                    byte_len: seed.down.bytes.len(),
                },
            ]
        };
        groups.push(ReadyExpertGroup {
            assignment_indices: seed.assignment_indices,
            buffers,
            read_tasks,
            cache_hit,
            cached_miss_key,
            transient,
            _transient_owner: transient_owner,
        });
    }
    Ok(groups)
}

fn pread_ready_expert_gate_up(file: &File, path: &Path, group: &ReadyExpertGroup) -> Result<()> {
    validate_exact_len(
        "ready routed expert read task count",
        group.read_tasks.len(),
        3,
    )?;
    for task in &group.read_tasks[..2] {
        pread_expert_buffer(file, path, task)?;
    }
    Ok(())
}

fn pread_ready_expert_down(file: &File, path: &Path, group: &ReadyExpertGroup) -> Result<()> {
    let task = group
        .read_tasks
        .get(2)
        .ok_or_else(|| Error::backend("ready routed expert has no down read task"))?;
    pread_expert_buffer(file, path, task)
}

fn wait_ready_expert_waves(waves: &[SubmittedReadyWave]) -> Result<u64> {
    if let Some(last) = waves.last() {
        last.command_buffer.wait_until_completed();
    }
    for (index, wave) in waves.iter().enumerate() {
        if wave.command_buffer.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::backend(format!(
                "Q2 ready expert wave {index} with {} assignments did not complete: {:?}",
                wave.assignment_count,
                wave.command_buffer.status()
            )));
        }
    }
    waves.iter().try_fold(0_u64, |total, wave| {
        total
            .checked_add(command_buffer_gpu_nanoseconds(&wave.command_buffer))
            .ok_or_else(|| Error::backend("Q2 expert GPU time overflow"))
    })
}

#[allow(unexpected_cfgs)]
fn command_buffer_gpu_nanoseconds(command_buffer: &CommandBufferRef) -> u64 {
    // SAFETY: GPUStartTime and GPUEndTime are read-only MTLCommandBuffer
    // properties and this function is called only after completion.
    let start: f64 = unsafe { msg_send![command_buffer, GPUStartTime] };
    let end: f64 = unsafe { msg_send![command_buffer, GPUEndTime] };
    if !start.is_finite() || !end.is_finite() || end <= start {
        return 0;
    }
    ((end - start) * 1_000_000_000.0) as u64
}

fn elapsed_nanoseconds(elapsed: std::time::Duration) -> u64 {
    u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
}

fn pread_expert_buffer(file: &File, path: &Path, task: &ExpertReadTask) -> Result<()> {
    let destination_end = task
        .destination_offset
        .checked_add(task.byte_len)
        .ok_or_else(|| Error::backend("Q2 expert pread destination range overflow"))?;
    require_byte_capacity(&task.buffer, destination_end, "Q2 expert pread destination")?;
    let pointer = task.buffer.contents().cast::<u8>();
    if pointer.is_null() {
        return Err(Error::backend(
            "Q2 expert pread destination pointer is null",
        ));
    }
    // SAFETY: this shared Metal buffer is exclusively owned by the cache slot,
    // the previous command buffer was flushed before reuse, and capacity was
    // validated above.
    let mut destination = unsafe {
        std::slice::from_raw_parts_mut(pointer.add(task.destination_offset), task.byte_len)
    };
    let mut offset = task.absolute_offset;
    while !destination.is_empty() {
        let read = file
            .read_at(destination, offset)
            .map_err(|source| Error::Io {
                path: path.to_path_buf(),
                source,
            })?;
        if read == 0 {
            return Err(Error::Io {
                path: path.to_path_buf(),
                source: io::Error::new(
                    ErrorKind::UnexpectedEof,
                    "unexpected EOF while loading Q2 expert",
                ),
            });
        }
        offset = offset
            .checked_add(
                u64::try_from(read)
                    .map_err(|_| Error::backend("Q2 expert read size does not fit u64"))?,
            )
            .ok_or_else(|| Error::backend("Q2 expert read offset overflow"))?;
        destination = &mut destination[read..];
    }
    Ok(())
}

fn matvec_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::backend(format!("quantized matvec {label} exceeds Metal u32 limit")))
}

fn validate_exact_len(label: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(Error::backend(format!(
            "{label} length mismatch: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn require_u32_buffer_capacity(buffer: &Buffer, len: usize, label: &str) -> Result<()> {
    let required_bytes = len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| Error::backend(format!("{label} byte length overflow")))?;
    if buffer.length() < required_bytes as u64 {
        return Err(Error::backend(format!(
            "{label} buffer is too small: expected at least {required_bytes} bytes, got {}",
            buffer.length()
        )));
    }
    Ok(())
}

fn validate_q2_k_packed_experts(
    weights: &[u8],
    in_features: usize,
    out_features: usize,
    context: &str,
) -> Result<(usize, usize, usize)> {
    if in_features == 0 || out_features == 0 {
        return Err(Error::backend(format!(
            "{context} requires non-zero in_features and out_features"
        )));
    }
    if in_features % Q2_K_BLOCK_VALUES != 0 {
        return Err(Error::backend(format!(
            "{context} in_features {in_features} must be divisible by {Q2_K_BLOCK_VALUES}"
        )));
    }
    let blocks_per_row = in_features / Q2_K_BLOCK_VALUES;
    let expert_stride_bytes = out_features
        .checked_mul(blocks_per_row)
        .and_then(|blocks| blocks.checked_mul(Q2_K_BLOCK_BYTES))
        .ok_or_else(|| Error::backend(format!("{context} expert byte stride overflow")))?;
    if expert_stride_bytes == 0 {
        return Err(Error::backend(format!(
            "{context} expert byte stride must be non-zero"
        )));
    }
    if weights.len() % expert_stride_bytes != 0 {
        return Err(Error::backend(format!(
            "{context} packed weight byte length {} is not divisible by expert stride {expert_stride_bytes}",
            weights.len()
        )));
    }
    let expert_count = weights.len() / expert_stride_bytes;
    if expert_count == 0 {
        return Err(Error::backend(format!(
            "{context} requires at least one packed expert"
        )));
    }
    matvec_u32(expert_stride_bytes, "expert_stride_bytes")?;
    Ok((blocks_per_row, expert_count, expert_stride_bytes))
}

fn validate_multi_expert_routing(
    token_indices: &[u32],
    expert_ids: &[u32],
    token_count: usize,
    context: &str,
) -> Result<usize> {
    if token_count == 0 {
        return Err(Error::backend(format!(
            "{context} requires non-zero token_count"
        )));
    }
    if token_indices.is_empty() {
        return Err(Error::backend(format!(
            "{context} requires at least one routed assignment"
        )));
    }
    if token_indices.len() != expert_ids.len() {
        return Err(Error::backend(format!(
            "{context} routing length mismatch: {} token indices and {} expert ids",
            token_indices.len(),
            expert_ids.len()
        )));
    }
    for token_index in token_indices {
        if *token_index as usize >= token_count {
            return Err(Error::backend(format!(
                "{context} token index {token_index} is outside token_count {token_count}"
            )));
        }
    }
    Ok(token_indices.len())
}

fn validate_expert_ids(expert_ids: &[u32], expert_count: usize, context: &str) -> Result<()> {
    for expert_id in expert_ids {
        if *expert_id as usize >= expert_count {
            return Err(Error::backend(format!(
                "{context} expert id {expert_id} is outside expert_count {expert_count}"
            )));
        }
    }
    Ok(())
}

fn q2_k_cooperative_threads(
    pipeline: &ComputePipelineState,
    output_len: usize,
    context: &str,
) -> Result<usize> {
    let thread_execution_width = pipeline.thread_execution_width() as usize;
    if thread_execution_width != Q2_K_SIMD_LANES {
        return Err(Error::backend(format!(
            "{context} requires {Q2_K_SIMD_LANES}-lane Apple Metal SIMD groups, got {thread_execution_width}"
        )));
    }
    output_len
        .checked_mul(Q2_K_SIMD_LANES)
        .ok_or_else(|| Error::backend(format!("{context} physical thread count overflow")))
}

fn q8_0_simdgroups_per_output(blocks_per_row: usize) -> usize {
    blocks_per_row.min(Q8_0_MAX_SIMDGROUPS_PER_OUTPUT).max(1)
}

fn argmax_threads(pipeline: &ComputePipelineState, context: &str) -> Result<usize> {
    let thread_execution_width = pipeline.thread_execution_width() as usize;
    let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
    if thread_execution_width == 0
        || thread_execution_width > ARGMAX_THREADS_PER_VECTOR
        || ARGMAX_THREADS_PER_VECTOR % thread_execution_width != 0
    {
        return Err(Error::backend(format!(
            "{context} requires a thread execution width that divides {ARGMAX_THREADS_PER_VECTOR}, got {thread_execution_width}"
        )));
    }
    if max_threads < ARGMAX_THREADS_PER_VECTOR {
        return Err(Error::backend(format!(
            "{context} requires {ARGMAX_THREADS_PER_VECTOR} threads per threadgroup, pipeline allows {max_threads}"
        )));
    }
    Ok(ARGMAX_THREADS_PER_VECTOR)
}

impl Q2ScratchBuffers {
    fn input_buffer(&mut self, device: &Device, values: &[f32]) -> Result<ScratchBufferRef> {
        let scratch = scratch_f32_buffer(device, &mut self.input, values.len())?;
        write_f32_buffer(&scratch.buffer, values)?;
        Ok(scratch)
    }

    fn residual_buffer(&mut self, device: &Device, values: &[f32]) -> Result<ScratchBufferRef> {
        let scratch = scratch_f32_buffer(device, &mut self.residual, values.len())?;
        write_f32_buffer(&scratch.buffer, values)?;
        Ok(scratch)
    }

    fn output_buffer(&mut self, device: &Device, len: usize) -> Result<ScratchBufferRef> {
        scratch_f32_buffer(device, &mut self.output, len)
    }

    fn logits_buffer(&mut self, device: &Device, len: usize) -> Result<ScratchBufferRef> {
        scratch_f32_buffer(device, &mut self.logits, len)
    }

    fn token_id_buffer(&mut self, device: &Device) -> Result<ScratchBufferRef> {
        scratch_u32_buffer(device, &mut self.token_id, 1)
    }

    fn token_score_buffer(&mut self, device: &Device) -> Result<ScratchBufferRef> {
        scratch_f32_buffer(device, &mut self.token_score, 1)
    }

    fn row_count_buffer(&mut self, device: &Device, value: u32) -> Result<ScratchBufferRef> {
        scratch_u32_value_buffer(device, &mut self.row_count, value)
    }

    fn in_features_buffer(&mut self, device: &Device, value: u32) -> Result<ScratchBufferRef> {
        scratch_u32_value_buffer(device, &mut self.in_features, value)
    }

    fn out_features_buffer(&mut self, device: &Device, value: u32) -> Result<ScratchBufferRef> {
        scratch_u32_value_buffer(device, &mut self.out_features, value)
    }

    fn blocks_per_row_buffer(&mut self, device: &Device, value: u32) -> Result<ScratchBufferRef> {
        scratch_u32_value_buffer(device, &mut self.blocks_per_row, value)
    }

    fn output_len_buffer(&mut self, device: &Device, value: u32) -> Result<ScratchBufferRef> {
        scratch_u32_value_buffer(device, &mut self.output_len, value)
    }
}

fn scratch_f32_buffer(
    device: &Device,
    slot: &mut Option<ScratchBuffer>,
    len: usize,
) -> Result<ScratchBufferRef> {
    scratch_buffer_with(device, slot, len, empty_f32_buffer)
}

fn scratch_u32_buffer(
    device: &Device,
    slot: &mut Option<ScratchBuffer>,
    len: usize,
) -> Result<ScratchBufferRef> {
    scratch_buffer_with(device, slot, len, empty_u32_buffer)
}

fn scratch_u32_value_buffer(
    device: &Device,
    slot: &mut Option<ScratchBuffer>,
    value: u32,
) -> Result<ScratchBufferRef> {
    let scratch = scratch_u32_buffer(device, slot, 1)?;
    write_u32_buffer(&scratch.buffer, &[value])?;
    Ok(scratch)
}

fn scratch_buffer_with(
    device: &Device,
    slot: &mut Option<ScratchBuffer>,
    len: usize,
    allocate: fn(&Device, usize) -> Result<Buffer>,
) -> Result<ScratchBufferRef> {
    if len == 0 {
        return Err(Error::backend("Q2 scratch buffer length must be positive"));
    }
    if let Some(existing) = slot.as_ref() {
        if existing.len >= len {
            return Ok(ScratchBufferRef {
                buffer: existing.buffer.clone(),
                reused: true,
            });
        }
    }

    let buffer = allocate(device, len)?;
    *slot = Some(ScratchBuffer {
        buffer: buffer.clone(),
        len,
    });
    Ok(ScratchBufferRef {
        buffer,
        reused: false,
    })
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use std::{
        collections::HashSet,
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    use crate::{metal::Metal, DevicePagedKvView, DeviceValue, Q2ExpertSource};

    use super::{
        super::validation::{
            Q2_K_BLOCK_BYTES, Q2_K_BLOCK_VALUES, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_VALUES,
        },
        ExpertCacheKey, LayerSlotResolution, Q2ExpertLayerCache, Q2PerLayerExpertCache,
        QuantMatvecKind,
    };

    fn expert_key(seed: usize) -> ExpertCacheKey {
        ExpertCacheKey {
            gate_address: seed * 10 + 1,
            gate_bytes: 4,
            up_address: seed * 10 + 2,
            up_bytes: 4,
            down_address: seed * 10 + 3,
            down_bytes: 4,
        }
    }

    #[test]
    fn expert_lru_is_partitioned_by_layer() {
        let mut cache = Q2PerLayerExpertCache::default();
        let key = expert_key(1);
        let selected = [key].into_iter().collect();
        let first_layer = cache.layer(3);
        let second_layer = cache.layer(4);

        assert!(matches!(
            first_layer
                .lock()
                .unwrap()
                .resolve_slot(key, &selected)
                .unwrap(),
            LayerSlotResolution::Miss(_)
        ));
        assert!(matches!(
            second_layer
                .lock()
                .unwrap()
                .resolve_slot(key, &selected)
                .unwrap(),
            LayerSlotResolution::Miss(_)
        ));
    }

    #[test]
    fn expert_lru_uses_configured_capacity_for_new_layers() {
        let mut cache = Q2PerLayerExpertCache {
            slots_per_layer: 3,
            ..Q2PerLayerExpertCache::default()
        };

        let layer = cache.layer(7);

        assert_eq!(layer.lock().unwrap().slots.len(), 3);
    }

    #[test]
    fn expert_lru_evicts_only_unselected_entries() {
        let mut cache = Q2ExpertLayerCache::new(2);
        let first = expert_key(1);
        let second = expert_key(2);
        let third = expert_key(3);
        cache.resolve_slot(first, &HashSet::new()).unwrap();
        cache.resolve_slot(second, &HashSet::new()).unwrap();
        let selected = [first, third].into_iter().collect();

        assert!(matches!(
            cache.resolve_slot(third, &selected).unwrap(),
            LayerSlotResolution::Miss(_)
        ));
        assert!(cache.entries.contains_key(&first));
        assert!(!cache.entries.contains_key(&second));
        assert!(cache.entries.contains_key(&third));
    }

    #[test]
    fn expert_lru_uses_transient_slot_when_every_entry_is_selected() {
        let mut cache = Q2ExpertLayerCache::new(2);
        let first = expert_key(1);
        let second = expert_key(2);
        let third = expert_key(3);
        cache.resolve_slot(first, &HashSet::new()).unwrap();
        cache.resolve_slot(second, &HashSet::new()).unwrap();
        let selected = [first, second, third].into_iter().collect();

        assert!(matches!(
            cache.resolve_slot(third, &selected).unwrap(),
            LayerSlotResolution::Transient
        ));
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn segmented_lru_protects_reused_experts_from_one_time_scans() {
        let mut cache = Q2ExpertLayerCache::new(3);
        let reused = expert_key(1);
        let scan_a = expert_key(2);
        let scan_b = expert_key(3);
        let scan_c = expert_key(4);
        cache.resolve_slot(reused, &HashSet::new()).unwrap();
        cache.resolve_slot(scan_a, &HashSet::new()).unwrap();
        cache.resolve_slot(scan_b, &HashSet::new()).unwrap();
        cache.resolve_slot(reused, &HashSet::new()).unwrap();

        cache.resolve_slot(scan_c, &HashSet::new()).unwrap();

        assert!(cache.entries.contains_key(&reused));
        assert!(cache.protected.contains(&reused));
        assert!(!cache.entries.contains_key(&scan_a));
    }

    #[test]
    fn matches_cpu_reference_for_q2_k_matvec() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let weights = [
            q2_k_block(0x3c00, 0x0000, 0x01, 0xe4),
            q2_k_block(0x4000, 0x0000, 0x01, 0x1b),
        ]
        .concat();
        let input = (0..Q2_K_BLOCK_VALUES)
            .map(|index| (index as f32 % 17.0) * 0.25)
            .collect::<Vec<_>>();

        let report = metal
            .q2_k_matvec_f32_report(&weights, &input, 1, Q2_K_BLOCK_VALUES, 2)
            .unwrap();
        let expected = cpu_q2_k_matvec(&weights, &input, 1, Q2_K_BLOCK_VALUES, 2);

        assert_eq!(report.row_count, 1);
        assert_eq!(report.in_features, Q2_K_BLOCK_VALUES);
        assert_eq!(report.out_features, 2);
        assert_eq!(report.blocks_per_row, 1);
        assert_eq!(report.weight_bytes, Q2_K_BLOCK_BYTES * 2);
        assert_close(&report.values, &expected, 1e-4);
    }

    #[test]
    fn q2_k_matvec_argmax_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let weights = [
            q2_k_block(0x3c00, 0x0000, 0x01, 0x1b),
            q2_k_block(0x4000, 0x0000, 0x01, 0xe4),
            q2_k_block(0x3c00, 0x0000, 0x01, 0x00),
        ]
        .concat();
        let input = (0..Q2_K_BLOCK_VALUES)
            .map(|index| (index as f32 % 11.0) * 0.125)
            .collect::<Vec<_>>();

        let report = metal
            .q2_k_matvec_argmax_f32_report(&weights, &input, 1, Q2_K_BLOCK_VALUES, 3)
            .unwrap();
        let scores = cpu_q2_k_matvec(&weights, &input, 1, Q2_K_BLOCK_VALUES, 3);
        let (expected_id, expected_score) = cpu_argmax(&scores);

        assert_eq!(report.row_count, 1);
        assert_eq!(report.out_features, 3);
        assert_eq!(report.token_id, expected_id);
        assert!((report.token_score - expected_score).abs() <= 1e-4);
    }

    #[test]
    fn row_wise_argmax_returns_one_result_per_token() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let scores = [
            1.0_f32, 9.0, 3.0, 9.0, // tie resolves to token 1
            -4.0, -2.0, 7.5, 0.0, // token 2
        ];
        let scores_buffer = metal.batch_upload_f32(&scores).unwrap();

        let (token_ids, token_scores) =
            metal.batched_f32_argmax_rows(&scores_buffer, 2, 4).unwrap();

        assert_eq!(token_ids, vec![1, 2]);
        assert_eq!(token_scores, vec![9.0, 7.5]);
    }

    #[test]
    fn q2_k_rms_norm_argmax_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let weights = [
            q2_k_block(0x3c00, 0x0000, 0x01, 0x1b),
            q2_k_block(0x4000, 0x0000, 0x01, 0xe4),
            q2_k_block(0x3c00, 0x0000, 0x01, 0x00),
        ]
        .concat();
        let input = (0..Q2_K_BLOCK_VALUES)
            .map(|index| (index as f32 % 11.0) * 0.125)
            .collect::<Vec<_>>();
        let rms_weight = (0..Q2_K_BLOCK_VALUES)
            .map(|index| 1.0 + (index as f32 % 7.0) * 0.01)
            .collect::<Vec<_>>();
        let eps = 1e-5;

        let (token_id, token_score) = metal
            .q2_k_rms_norm_argmax_f32(&weights, &input, &rms_weight, 1, Q2_K_BLOCK_VALUES, 3, eps)
            .unwrap();
        let normed = cpu_rms_norm(&input, &rms_weight, 1, Q2_K_BLOCK_VALUES, eps);
        let scores = cpu_q2_k_matvec(&weights, &normed, 1, Q2_K_BLOCK_VALUES, 3);
        let (expected_id, expected_score) = cpu_argmax(&scores);

        assert_eq!(token_id, expected_id);
        assert!((token_score - expected_score).abs() <= 1e-4);
    }

    #[test]
    fn q2_k_matvec_add_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let weights = [
            q2_k_block(0x3c00, 0x0000, 0x01, 0xe4),
            q2_k_block(0x4000, 0x0000, 0x01, 0x1b),
        ]
        .concat();
        let input = (0..Q2_K_BLOCK_VALUES)
            .map(|index| (index as f32 % 17.0) * 0.25)
            .collect::<Vec<_>>();
        let residual = vec![0.25_f32, -0.5];

        let report = metal
            .q2_k_matvec_add_f32_report(&weights, &input, &residual, 1, Q2_K_BLOCK_VALUES, 2)
            .unwrap();
        let mut expected = cpu_q2_k_matvec(&weights, &input, 1, Q2_K_BLOCK_VALUES, 2);
        for (value, residual) in expected.iter_mut().zip(&residual) {
            *value += residual;
        }

        assert_eq!(report.row_count, 1);
        assert_eq!(report.in_features, Q2_K_BLOCK_VALUES);
        assert_eq!(report.out_features, 2);
        assert_eq!(report.residual_len, 2);
        assert_close(&report.values, &expected, 1e-4);
    }

    #[test]
    fn q8_0_tiled_matvec_add_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let in_features = 64;
        let out_features = 2;
        let weights = [
            q8_0_block(0x3c00, 1),
            q8_0_block(0x4000, -2),
            q8_0_block(0x3800, 3),
            q8_0_block(0x3c00, 1),
        ]
        .concat();
        let input = (0..in_features)
            .map(|index| (index as f32 - 17.0) * 0.03125)
            .collect::<Vec<_>>();
        let residual = vec![0.25_f32, -0.5];
        let input_buffer = metal.batch_upload_f32(&input).unwrap();
        let residual_buffer = metal.batch_upload_f32(&residual).unwrap();

        let output = metal
            .batched_q8_0_matvec_add(
                &weights,
                &input_buffer,
                input.len(),
                &residual_buffer,
                residual.len(),
                1,
                in_features,
                out_features,
            )
            .unwrap();
        let actual = metal.batch_read_f32(&output, out_features).unwrap();
        let mut expected = cpu_q8_0_matvec(&weights, &input, 1, in_features, out_features);
        for (value, residual) in expected.iter_mut().zip(&residual) {
            *value += residual;
        }

        assert_close(&actual, &expected, 1e-4);
    }

    #[test]
    fn gate_up_swiglu_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let gate_weights = [
            q2_k_block(0x3c00, 0x0000, 0x01, 0xe4),
            q2_k_block(0x4000, 0x0000, 0x01, 0x1b),
        ]
        .concat();
        let up_weights = [
            q2_k_block(0x4000, 0x0000, 0x01, 0xe4),
            q2_k_block(0x3c00, 0x0000, 0x01, 0x1b),
        ]
        .concat();
        let input = (0..Q2_K_BLOCK_VALUES)
            .map(|index| (index as f32 % 13.0) * 0.1 - 0.35)
            .collect::<Vec<_>>();

        let report = metal
            .q2_k_gate_up_swiglu_f32_report(
                &gate_weights,
                &up_weights,
                &input,
                1,
                Q2_K_BLOCK_VALUES,
                2,
            )
            .unwrap();
        let gate = cpu_q2_k_matvec(&gate_weights, &input, 1, Q2_K_BLOCK_VALUES, 2);
        let up = cpu_q2_k_matvec(&up_weights, &input, 1, Q2_K_BLOCK_VALUES, 2);
        let expected = cpu_swiglu(&gate, &up);

        assert_eq!(report.row_count, 1);
        assert_eq!(report.in_features, Q2_K_BLOCK_VALUES);
        assert_eq!(report.out_features, 2);
        assert_eq!(report.blocks_per_row, 1);
        assert_eq!(report.gate_weight_bytes, Q2_K_BLOCK_BYTES * 2);
        assert_eq!(report.up_weight_bytes, Q2_K_BLOCK_BYTES * 2);
        assert_close(&report.values, &expected, 1e-4);
    }

    #[test]
    fn multi_expert_gate_up_and_matvec_match_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let hidden_features = 2;
        let intermediate_features = Q2_K_BLOCK_VALUES;
        let expert_count = 2;
        let gate_expert_stride = intermediate_features * Q2_K_BLOCK_BYTES;
        let down_expert_stride = hidden_features * Q2_K_BLOCK_BYTES;
        let mut gate_weights = Vec::new();
        let mut up_weights = Vec::new();
        for expert in 0..expert_count {
            for row in 0..intermediate_features {
                let quant = if (expert + row) % 2 == 0 { 0xe4 } else { 0x1b };
                gate_weights.extend(q2_k_block(
                    if expert == 0 { 0x3c00 } else { 0x4000 },
                    0x0000,
                    0x01 + expert as u8,
                    quant,
                ));
                up_weights.extend(q2_k_block(
                    if expert == 0 { 0x4000 } else { 0x3c00 },
                    0x0000,
                    0x02 + expert as u8,
                    quant ^ 0xff,
                ));
            }
        }
        let mut down_weights = Vec::new();
        for expert in 0..expert_count {
            for row in 0..hidden_features {
                down_weights.extend(q2_k_block(
                    if expert == 0 { 0x3c00 } else { 0x4000 },
                    0x0000,
                    0x03 + row as u8,
                    if expert == row { 0xe4 } else { 0x1b },
                ));
            }
        }
        let token_count = 2;
        let input = (0..token_count * Q2_K_BLOCK_VALUES)
            .map(|index| (index as f32 % 19.0) * 0.0005 - 0.002)
            .collect::<Vec<_>>();
        let token_indices = vec![1_u32, 0, 1];
        let expert_ids = vec![1_u32, 0, 1];

        let input_buffer = metal.batch_upload_f32(&input).unwrap();
        let gated_buffer = metal
            .batched_q2_k_multi_expert_gate_up_swiglu(
                &gate_weights,
                &up_weights,
                &input_buffer,
                input.len(),
                &token_indices,
                &expert_ids,
                token_count,
                Q2_K_BLOCK_VALUES,
                intermediate_features,
            )
            .unwrap();
        let down_buffer = metal
            .batched_q2_k_multi_expert_matvec(
                &down_weights,
                &gated_buffer,
                token_indices.len() * intermediate_features,
                &expert_ids,
                intermediate_features,
                hidden_features,
            )
            .unwrap();
        let gated = metal
            .batch_read_f32(&gated_buffer, token_indices.len() * intermediate_features)
            .unwrap();
        let down = metal
            .batch_read_f32(&down_buffer, token_indices.len() * hidden_features)
            .unwrap();

        let mut expected_gated = Vec::new();
        for (&token, &expert) in token_indices.iter().zip(&expert_ids) {
            let token = token as usize;
            let expert = expert as usize;
            let token_input = &input[token * Q2_K_BLOCK_VALUES..(token + 1) * Q2_K_BLOCK_VALUES];
            let expert_start = expert * gate_expert_stride;
            let expert_end = expert_start + gate_expert_stride;
            let gate = cpu_q2_k_matvec(
                &gate_weights[expert_start..expert_end],
                token_input,
                1,
                Q2_K_BLOCK_VALUES,
                intermediate_features,
            );
            let up = cpu_q2_k_matvec(
                &up_weights[expert_start..expert_end],
                token_input,
                1,
                Q2_K_BLOCK_VALUES,
                intermediate_features,
            );
            expected_gated.extend(cpu_swiglu(&gate, &up));
        }
        let mut expected_down = Vec::new();
        for (assignment, expert) in expert_ids.iter().copied().enumerate() {
            let expert = expert as usize;
            let expert_start = expert * down_expert_stride;
            let expert_end = expert_start + down_expert_stride;
            expected_down.extend(cpu_q2_k_matvec(
                &down_weights[expert_start..expert_end],
                &expected_gated
                    [assignment * intermediate_features..(assignment + 1) * intermediate_features],
                1,
                intermediate_features,
                hidden_features,
            ));
        }

        assert_close_relative(&gated, &expected_gated, 1e-6);
        assert_close_relative(&down, &expected_down, 1e-6);
    }

    #[test]
    fn ready_router_experts_match_cpu_combine_and_reuse_layer_cache() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let hidden_features = Q2_K_BLOCK_VALUES;
        let intermediate_features = Q2_K_BLOCK_VALUES;
        let expert_count = 2;
        let mut gate_weights = Vec::new();
        let mut up_weights = Vec::new();
        let mut down_weights = Vec::new();
        for expert in 0..expert_count {
            for _row in 0..intermediate_features {
                gate_weights.extend(q2_k_block(
                    if expert == 0 { 0x3c00 } else { 0x4000 },
                    0x0000,
                    0x01,
                    if expert == 0 { 0xe4 } else { 0x1b },
                ));
                up_weights.extend(q2_k_block(
                    if expert == 0 { 0x4000 } else { 0x3c00 },
                    0x0000,
                    0x01,
                    if expert == 0 { 0x1b } else { 0xe4 },
                ));
            }
            for row in 0..hidden_features {
                down_weights.extend(q2_k_block(
                    0x3c00,
                    0x0000,
                    0x01,
                    if (expert + row) % 2 == 0 { 0xe4 } else { 0x1b },
                ));
            }
        }
        let input = (0..hidden_features)
            .map(|index| (index as f32 % 13.0) * 0.001 - 0.004)
            .collect::<Vec<_>>();
        let router_logits = vec![3.0_f32, 1.0];
        let correction_bias = vec![0.0_f32; expert_count];
        let shared = vec![0.25_f32; hidden_features];
        let residual = vec![-0.5_f32; hidden_features];
        let input_buffer = metal.batch_upload_f32(&input).unwrap();
        let logits_buffer = metal.batch_upload_f32(&router_logits).unwrap();
        let shared_buffer = metal.batch_upload_f32(&shared).unwrap();
        let residual_buffer = metal.batch_upload_f32(&residual).unwrap();
        let routing = metal
            .batched_moe_router_topk_resident(
                &logits_buffer,
                router_logits.len(),
                &correction_bias,
                1,
                expert_count,
                2,
                true,
                1.0,
            )
            .unwrap();
        let selected_ids = metal.batched_moe_router_expert_ids(&routing).unwrap();
        let gate_stride = intermediate_features * Q2_K_BLOCK_BYTES;
        let down_stride = hidden_features * Q2_K_BLOCK_BYTES;
        let gate_payloads = selected_ids
            .iter()
            .map(|&expert| {
                let start = expert as usize * gate_stride;
                &gate_weights[start..start + gate_stride]
            })
            .collect::<Vec<_>>();
        let up_payloads = selected_ids
            .iter()
            .map(|&expert| {
                let start = expert as usize * gate_stride;
                &up_weights[start..start + gate_stride]
            })
            .collect::<Vec<_>>();
        let down_payloads = selected_ids
            .iter()
            .map(|&expert| {
                let start = expert as usize * down_stride;
                &down_weights[start..start + down_stride]
            })
            .collect::<Vec<_>>();

        let up_base = gate_weights.len() as u64;
        let down_base = up_base + up_weights.len() as u64;
        let model_file = TemporaryModelFile::new(
            "staged-router-experts",
            [&gate_weights[..], &up_weights[..], &down_weights[..]].concat(),
        );
        let gate_sources = selected_ids
            .iter()
            .zip(&gate_payloads)
            .map(|(&expert, &bytes)| Q2ExpertSource {
                bytes,
                absolute_offset: expert as u64 * gate_stride as u64,
            })
            .collect::<Vec<_>>();
        let up_sources = selected_ids
            .iter()
            .zip(&up_payloads)
            .map(|(&expert, &bytes)| Q2ExpertSource {
                bytes,
                absolute_offset: up_base + expert as u64 * gate_stride as u64,
            })
            .collect::<Vec<_>>();
        let down_sources = selected_ids
            .iter()
            .zip(&down_payloads)
            .map(|(&expert, &bytes)| Q2ExpertSource {
                bytes,
                absolute_offset: down_base + expert as u64 * down_stride as u64,
            })
            .collect::<Vec<_>>();
        let first = metal
            .ready_routed_experts(
                1,
                model_file.path(),
                &gate_sources,
                &up_sources,
                &down_sources,
                &input_buffer,
                input.len(),
                &routing,
                hidden_features,
                intermediate_features,
                hidden_features,
            )
            .unwrap();
        assert_eq!(first.cache_hits, 0);
        assert_eq!(first.cache_misses, selected_ids.len());
        assert_eq!(first.selected_experts, selected_ids.len());
        assert!(first.ready_waves >= 1);
        let ready = metal
            .ready_routed_experts(
                1,
                model_file.path(),
                &gate_sources,
                &up_sources,
                &down_sources,
                &input_buffer,
                input.len(),
                &routing,
                hidden_features,
                intermediate_features,
                hidden_features,
            )
            .unwrap();
        assert_eq!(ready.cache_hits, selected_ids.len());
        assert_eq!(ready.cache_misses, 0);
        let other_layer = metal
            .ready_routed_experts(
                2,
                model_file.path(),
                &gate_sources,
                &up_sources,
                &down_sources,
                &input_buffer,
                input.len(),
                &routing,
                hidden_features,
                intermediate_features,
                hidden_features,
            )
            .unwrap();
        assert_eq!(other_layer.cache_hits, 0);
        assert_eq!(other_layer.cache_misses, selected_ids.len());
        let combined = metal
            .batched_moe_topk_combine_residual(
                &shared_buffer,
                shared.len(),
                &residual_buffer,
                residual.len(),
                &ready.output,
                2 * hidden_features,
                &routing,
                hidden_features,
            )
            .unwrap();
        let actual = metal.batch_read_f32(&combined, hidden_features).unwrap();

        let mut expected_expert_outputs = Vec::new();
        for expert in 0..expert_count {
            let gate_start = expert * gate_stride;
            let gate_end = gate_start + gate_stride;
            let gate = cpu_q2_k_matvec(
                &gate_weights[gate_start..gate_end],
                &input,
                1,
                hidden_features,
                intermediate_features,
            );
            let up = cpu_q2_k_matvec(
                &up_weights[gate_start..gate_end],
                &input,
                1,
                hidden_features,
                intermediate_features,
            );
            let gated = cpu_swiglu(&gate, &up);
            let down_start = expert * down_stride;
            let down_end = down_start + down_stride;
            expected_expert_outputs.push(cpu_q2_k_matvec(
                &down_weights[down_start..down_end],
                &gated,
                1,
                intermediate_features,
                hidden_features,
            ));
        }
        let score_0 = 1.0 / (1.0 + (-router_logits[0]).exp());
        let score_1 = 1.0 / (1.0 + (-router_logits[1]).exp());
        let weight_0 = score_0 / (score_0 + score_1);
        let weight_1 = score_1 / (score_0 + score_1);
        let expected = (0..hidden_features)
            .map(|index| {
                shared[index]
                    + residual[index]
                    + expected_expert_outputs[0][index] * weight_0
                    + expected_expert_outputs[1][index] * weight_1
            })
            .collect::<Vec<_>>();
        assert_close_relative(&actual, &expected, 2e-6);
    }

    #[test]
    fn rejects_bad_q2_k_shape_before_dispatch() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let err = metal
            .q2_k_matvec_f32(
                &[0_u8; 16],
                &[0.0_f32; Q2_K_BLOCK_VALUES],
                1,
                Q2_K_BLOCK_VALUES,
                1,
            )
            .expect_err("bad Q2 byte count should fail before Metal dispatch");

        assert!(err.to_string().contains("weight byte mismatch"));
    }

    #[test]
    fn transposed_matches_cpu_reference_for_q2_k_matvec() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let in_features = 2;
        let out_features = Q2_K_BLOCK_VALUES;
        let weights = [
            q2_k_block(0x3c00, 0x0000, 0x01, 0xe4),
            q2_k_block(0x4000, 0x0000, 0x01, 0x1b),
        ]
        .concat();
        let input = vec![1.25_f32, -0.5];

        let report = metal
            .q2_k_transposed_matvec_f32_report(&weights, &input, 1, in_features, out_features)
            .unwrap();
        let expected = cpu_q2_k_transposed_matvec(&weights, &input, 1, in_features, out_features);

        assert_eq!(report.row_count, 1);
        assert_eq!(report.in_features, in_features);
        assert_eq!(report.out_features, out_features);
        assert_eq!(report.blocks_per_row, 1);
        assert!(report.transposed);
        assert_close(&report.values, &expected, 1e-4);
    }

    #[test]
    fn packed_q2_heads_match_per_head_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let in_features = 2;
        let out_features = Q2_K_BLOCK_VALUES;
        let head_count = 2;
        let head_0 = [
            q2_k_block(0x3c00, 0x0000, 0x01, 0xe4),
            q2_k_block(0x4000, 0x0000, 0x01, 0x1b),
        ]
        .concat();
        let head_1 = [
            q2_k_block(0x4000, 0x0000, 0x02, 0x1b),
            q2_k_block(0x3c00, 0x0000, 0x03, 0xe4),
        ]
        .concat();
        let weights = [head_0.as_slice(), head_1.as_slice()].concat();
        let input = vec![1.25_f32, -0.5];
        let input_buffer = metal.batch_upload_f32(&input).unwrap();

        let output = metal
            .batched_packed_heads_transposed_matvec(
                QuantMatvecKind::Q2KTransposed,
                &weights,
                &input_buffer,
                input.len(),
                1,
                head_count,
                in_features,
                out_features,
            )
            .unwrap();
        let actual = metal
            .batch_read_f32(&output, head_count * out_features)
            .unwrap();
        let expected = [
            cpu_q2_k_transposed_matvec(&head_0, &input, 1, in_features, out_features),
            cpu_q2_k_transposed_matvec(&head_1, &input, 1, in_features, out_features),
        ]
        .concat();

        assert_close(&actual, &expected, 1e-4);
    }

    #[test]
    fn packed_q8_heads_match_per_head_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let in_features = 2;
        let out_features = 32;
        let head_count = 2;
        let head_0 = [q8_0_block(0x3c00, 1), q8_0_block(0x4000, -2)].concat();
        let head_1 = [q8_0_block(0x4000, 3), q8_0_block(0x3c00, 1)].concat();
        let weights = [head_0.as_slice(), head_1.as_slice()].concat();
        let input = vec![0.75_f32, -0.25];
        let input_buffer = metal.batch_upload_f32(&input).unwrap();

        let output = metal
            .batched_packed_heads_transposed_matvec(
                QuantMatvecKind::Q80Transposed,
                &weights,
                &input_buffer,
                input.len(),
                1,
                head_count,
                in_features,
                out_features,
            )
            .unwrap();
        let actual = metal
            .batch_read_f32(&output, head_count * out_features)
            .unwrap();
        let expected = [
            cpu_q8_0_transposed_matvec(&head_0, &input, in_features, out_features),
            cpu_q8_0_transposed_matvec(&head_1, &input, in_features, out_features),
        ]
        .concat();

        assert_close(&actual, &expected, 1e-4);
    }

    #[test]
    fn absorbed_mla_matches_expanded_q8_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch = 1;
        let heads = 2;
        let q_no_dim = 32;
        let rope_dim = 8;
        let latent_dim = 32;
        let value_dim = 32;
        let past_tokens = 2;
        let page_size = 4;

        let q_no = (0..heads * q_no_dim)
            .map(|index| (index as f32 - 17.0) * 0.01)
            .collect::<Vec<_>>();
        let q_rope = (0..heads * rope_dim)
            .map(|index| (index as f32 + 1.0) * -0.015)
            .collect::<Vec<_>>();
        let current_latent = (0..latent_dim)
            .map(|index| (index as f32 - 9.0) * 0.02)
            .collect::<Vec<_>>();
        let current_rope = (0..rope_dim)
            .map(|index| (index as f32 + 2.0) * 0.025)
            .collect::<Vec<_>>();
        let mut paged_latent = vec![0.0_f32; page_size * latent_dim];
        let mut paged_rope = vec![0.0_f32; page_size * rope_dim];
        for token in 0..past_tokens {
            for dim in 0..latent_dim {
                paged_latent[token * latent_dim + dim] =
                    (token * latent_dim + dim + 3) as f32 * 0.007;
            }
            for dim in 0..rope_dim {
                paged_rope[token * rope_dim + dim] = (token * rope_dim + dim + 1) as f32 * -0.011;
            }
        }

        let mut k_b = Vec::new();
        let mut v_b = Vec::new();
        for head in 0..heads {
            for output in 0..latent_dim {
                let quant = ((head + output) % 5) as i8 - 2;
                k_b.extend(q8_0_block(0x2c00, quant));
            }
            for output in 0..value_dim {
                let quant = ((head * 2 + output) % 7) as i8 - 3;
                v_b.extend(q8_0_block(0x2800, quant));
            }
        }

        let q_no_buffer = metal.batch_upload_f32(&q_no).unwrap();
        let q_rope_buffer = metal.batch_upload_f32(&q_rope).unwrap();
        let current_latent_buffer = metal.batch_upload_f32(&current_latent).unwrap();
        let current_rope_buffer = metal.batch_upload_f32(&current_rope).unwrap();
        let paged_latent_buffer = metal.batch_upload_f32(&paged_latent).unwrap();
        let paged_rope_buffer = metal.batch_upload_f32(&paged_rope).unwrap();
        let past_kv = DevicePagedKvView {
            batch,
            attention_heads: 1,
            key_head_dim: latent_dim,
            value_head_dim: rope_dim,
            page_size,
            cached_tokens: past_tokens,
            capacity_tokens: page_size,
            k: DeviceValue::new(
                vec![1, batch, 1, page_size, latent_dim],
                paged_latent_buffer,
            ),
            v: DeviceValue::new(vec![1, batch, 1, page_size, rope_dim], paged_rope_buffer),
        };

        let (output, output_len) = metal
            .batched_q8_0_absorbed_mla_decode(
                &k_b,
                &v_b,
                &q_no_buffer,
                q_no.len(),
                &q_rope_buffer,
                q_rope.len(),
                &current_latent_buffer,
                current_latent.len(),
                &current_rope_buffer,
                current_rope.len(),
                &past_kv,
                batch,
                heads,
                q_no_dim,
                rope_dim,
                latent_dim,
                value_dim,
                q_no_dim + rope_dim,
            )
            .unwrap();
        let actual = metal.batch_read_f32(&output, output_len).unwrap();

        let mut expected = Vec::with_capacity(heads * value_dim);
        let k_head_bytes = latent_dim * Q8_0_BLOCK_BYTES;
        let v_head_bytes = value_dim * Q8_0_BLOCK_BYTES;
        for head in 0..heads {
            let q_start = head * q_no_dim;
            let q_latent = cpu_q8_0_matvec(
                &k_b[head * k_head_bytes..(head + 1) * k_head_bytes],
                &q_no[q_start..q_start + q_no_dim],
                1,
                q_no_dim,
                latent_dim,
            );
            let mut scores = Vec::with_capacity(past_tokens + 1);
            for token in 0..past_tokens {
                let latent_start = token * latent_dim;
                let rope_start = token * rope_dim;
                let latent_score = q_latent
                    .iter()
                    .zip(&paged_latent[latent_start..latent_start + latent_dim])
                    .map(|(left, right)| left * right)
                    .sum::<f32>();
                let rope_score = q_rope[head * rope_dim..(head + 1) * rope_dim]
                    .iter()
                    .zip(&paged_rope[rope_start..rope_start + rope_dim])
                    .map(|(left, right)| left * right)
                    .sum::<f32>();
                scores.push((latent_score + rope_score) / ((q_no_dim + rope_dim) as f32).sqrt());
            }
            let current_score = q_latent
                .iter()
                .zip(&current_latent)
                .map(|(left, right)| left * right)
                .sum::<f32>()
                + q_rope[head * rope_dim..(head + 1) * rope_dim]
                    .iter()
                    .zip(&current_rope)
                    .map(|(left, right)| left * right)
                    .sum::<f32>();
            scores.push(current_score / ((q_no_dim + rope_dim) as f32).sqrt());
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = scores
                .iter()
                .map(|score| (score - max_score).exp())
                .sum::<f32>();
            let probabilities = scores
                .iter()
                .map(|score| (score - max_score).exp() / denominator)
                .collect::<Vec<_>>();
            let mut context_latent = vec![0.0_f32; latent_dim];
            for dim in 0..latent_dim {
                for token in 0..past_tokens {
                    context_latent[dim] +=
                        probabilities[token] * paged_latent[token * latent_dim + dim];
                }
                context_latent[dim] += probabilities[past_tokens] * current_latent[dim];
            }
            expected.extend(cpu_q8_0_matvec(
                &v_b[head * v_head_bytes..(head + 1) * v_head_bytes],
                &context_latent,
                1,
                latent_dim,
                value_dim,
            ));
        }

        assert_close(&actual, &expected, 2e-4);
    }

    fn native_metal_or_skip() -> Option<Metal> {
        Metal::new().ok()
    }

    struct TemporaryModelFile {
        path: PathBuf,
    }

    impl TemporaryModelFile {
        fn new(label: &str, bytes: Vec<u8>) -> Self {
            static NEXT_ID: AtomicU64 = AtomicU64::new(0);
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "inferno-backend-{label}-{}-{id}.gguf",
                std::process::id()
            ));
            fs::write(&path, bytes).expect("temporary model file should be writable");
            Self { path }
        }

        fn path(&self) -> &std::path::Path {
            &self.path
        }
    }

    impl Drop for TemporaryModelFile {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }

    fn q2_k_block(d: u16, dmin: u16, scale_min: u8, quant: u8) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q2_K_BLOCK_BYTES);
        block.extend(std::iter::repeat_n(scale_min, 16));
        block.extend(std::iter::repeat_n(quant, 64));
        block.extend(d.to_le_bytes());
        block.extend(dmin.to_le_bytes());
        block
    }

    fn q8_0_block(d: u16, quant: i8) -> Vec<u8> {
        let mut block = Vec::with_capacity(34);
        block.extend(d.to_le_bytes());
        block.extend(std::iter::repeat_n(quant as u8, 32));
        block
    }

    fn cpu_q2_k_matvec(
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Vec<f32> {
        let blocks_per_row = in_features / Q2_K_BLOCK_VALUES;
        let mut output = vec![0.0; row_count * out_features];

        for input_row in 0..row_count {
            for output_feature in 0..out_features {
                let mut sum = 0.0;
                for block_in_row in 0..blocks_per_row {
                    let block_index = output_feature * blocks_per_row + block_in_row;
                    let block_offset = block_index * Q2_K_BLOCK_BYTES;
                    let block = &weights[block_offset..block_offset + Q2_K_BLOCK_BYTES];
                    sum += cpu_q2_k_block_dot(
                        block,
                        &input[input_row * in_features + block_in_row * Q2_K_BLOCK_VALUES..],
                    );
                }
                output[input_row * out_features + output_feature] = sum;
            }
        }

        output
    }

    fn cpu_q8_0_matvec(
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Vec<f32> {
        let blocks_per_row = in_features / Q8_0_BLOCK_VALUES;
        let mut output = vec![0.0; row_count * out_features];
        for input_row in 0..row_count {
            for output_feature in 0..out_features {
                let mut sum = 0.0_f32;
                for block_in_row in 0..blocks_per_row {
                    let block_index = output_feature * blocks_per_row + block_in_row;
                    let block_offset = block_index * Q8_0_BLOCK_BYTES;
                    let block = &weights[block_offset..block_offset + Q8_0_BLOCK_BYTES];
                    let scale = f16_fixture_to_f32(u16::from_le_bytes([block[0], block[1]]));
                    let input_offset = input_row * in_features + block_in_row * Q8_0_BLOCK_VALUES;
                    for value_index in 0..Q8_0_BLOCK_VALUES {
                        let raw = block[2 + value_index];
                        let quant = if raw < 128 {
                            raw as i32
                        } else {
                            raw as i32 - 256
                        };
                        sum += input[input_offset + value_index] * scale * quant as f32;
                    }
                }
                output[input_row * out_features + output_feature] = sum;
            }
        }
        output
    }

    fn cpu_q2_k_block_dot(block: &[u8], input: &[f32]) -> f32 {
        let scales = &block[..16];
        let quants = &block[16..80];
        let d = f16_fixture_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let min = f16_fixture_to_f32(u16::from_le_bytes([block[82], block[83]]));
        let mut scale_index = 0;
        let mut quant_offset = 0;
        let mut input_offset = 0;
        let mut sum = 0.0;

        while input_offset < Q2_K_BLOCK_VALUES {
            let mut shift = 0;
            for _ in 0..4 {
                let scale_min = scales[scale_index];
                scale_index += 1;
                let scale = d * (scale_min & 0x0f) as f32;
                let min_offset = min * (scale_min >> 4) as f32;
                for value_index in 0..16 {
                    let weight = scale
                        * ((quants[quant_offset + value_index] >> shift) & 0x03) as f32
                        - min_offset;
                    sum += input[input_offset + value_index] * weight;
                }
                input_offset += 16;

                let scale_min = scales[scale_index];
                scale_index += 1;
                let scale = d * (scale_min & 0x0f) as f32;
                let min_offset = min * (scale_min >> 4) as f32;
                for value_index in 0..16 {
                    let weight = scale
                        * ((quants[quant_offset + 16 + value_index] >> shift) & 0x03) as f32
                        - min_offset;
                    sum += input[input_offset + value_index] * weight;
                }
                input_offset += 16;

                shift += 2;
            }
            quant_offset += 32;
        }

        sum
    }

    fn cpu_q2_k_transposed_matvec(
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Vec<f32> {
        let blocks_per_input_row = out_features / Q2_K_BLOCK_VALUES;
        let mut output = vec![0.0; row_count * out_features];

        for row in 0..row_count {
            for output_feature in 0..out_features {
                let block_in_input_row = output_feature / Q2_K_BLOCK_VALUES;
                let value_in_block = output_feature % Q2_K_BLOCK_VALUES;
                let mut sum = 0.0;
                for input_feature in 0..in_features {
                    let block_index = input_feature * blocks_per_input_row + block_in_input_row;
                    let block_offset = block_index * Q2_K_BLOCK_BYTES;
                    let block = &weights[block_offset..block_offset + Q2_K_BLOCK_BYTES];
                    sum += input[row * in_features + input_feature]
                        * cpu_q2_k_block_value(block, value_in_block);
                }
                output[row * out_features + output_feature] = sum;
            }
        }

        output
    }

    fn cpu_q8_0_transposed_matvec(
        weights: &[u8],
        input: &[f32],
        in_features: usize,
        out_features: usize,
    ) -> Vec<f32> {
        let blocks_per_input = out_features / 32;
        let mut output = vec![0.0_f32; out_features];
        for (output_feature, value) in output.iter_mut().enumerate() {
            let block_in_input = output_feature / 32;
            let value_in_block = output_feature % 32;
            for (input_feature, input_value) in input.iter().copied().enumerate() {
                let block = (input_feature * blocks_per_input + block_in_input) * 34;
                let d =
                    f16_fixture_to_f32(u16::from_le_bytes([weights[block], weights[block + 1]]));
                let quant = weights[block + 2 + value_in_block] as i8;
                *value += input_value * d * f32::from(quant);
            }
        }
        assert_eq!(input.len(), in_features);
        output
    }

    fn cpu_argmax(scores: &[f32]) -> (u32, f32) {
        let mut best_id = 0_u32;
        let mut best_score = scores[0];
        for (index, score) in scores.iter().copied().enumerate().skip(1) {
            if score > best_score {
                best_id = index as u32;
                best_score = score;
            }
        }
        (best_id, best_score)
    }

    fn cpu_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
        gate.iter()
            .copied()
            .zip(up.iter().copied())
            .map(|(gate, up)| {
                let silu = gate / (1.0 + (-gate).exp());
                silu * up
            })
            .collect()
    }

    fn cpu_rms_norm(
        input: &[f32],
        weight: &[f32],
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0; input.len()];
        for row in 0..rows {
            let base = row * hidden_size;
            let mut sumsq = 0.0;
            for col in 0..hidden_size {
                let value = input[base + col];
                sumsq += value * value;
            }
            let scale = ((sumsq / hidden_size as f32) + eps).sqrt().recip();
            for col in 0..hidden_size {
                output[base + col] = input[base + col] * scale * weight[col];
            }
        }
        output
    }

    fn cpu_q2_k_block_value(block: &[u8], value_index: usize) -> f32 {
        let half = value_index / 128;
        let within_half = value_index % 128;
        let pair = within_half / 32;
        let within_pair = within_half % 32;
        let upper_half_of_pair = within_pair >= 16;
        let scale_index = half * 8 + pair * 2 + usize::from(upper_half_of_pair);
        let quant_index =
            16 + half * 32 + if upper_half_of_pair { 16 } else { 0 } + (within_pair % 16);
        let shift = pair * 2;
        let d = f16_fixture_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let min = f16_fixture_to_f32(u16::from_le_bytes([block[82], block[83]]));
        let scale_min = block[scale_index];
        let scale = d * (scale_min & 0x0f) as f32;
        let min_offset = min * (scale_min >> 4) as f32;
        let quant = ((block[quant_index] >> shift) & 0x03) as f32;

        scale * quant - min_offset
    }

    fn f16_fixture_to_f32(bits: u16) -> f32 {
        match bits {
            0x0000 => 0.0,
            0x3c00 => 1.0,
            0x4000 => 2.0,
            other => panic!("unsupported test f16 bits {other:#06x}"),
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let delta = (actual - expected).abs();
            assert!(
                delta <= tolerance,
                "value {index} differs: actual={actual}, expected={expected}, delta={delta}"
            );
        }
    }

    fn assert_close_relative(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let delta = (actual - expected).abs();
            let allowed = tolerance * expected.abs().max(1.0);
            assert!(
                delta <= allowed,
                "value {index} differs: actual={actual}, expected={expected}, delta={delta}, allowed={allowed}"
            );
        }
    }
}
