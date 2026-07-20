use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{self, ErrorKind, Read},
    os::{fd::AsRawFd, unix::fs::FileExt},
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
    MTLCommandBufferStatus, SharedEvent,
};
use common::{Error, Result};
use inferno_io::{ExpertComponent, ExpertPackHeader, EXPERT_PACK_HEADER_BYTES};
use objc::{msg_send, sel, sel_impl};
use tracing::{debug, trace, warn};

use crate::{ExpertCacheMetrics, Q2ExpertSource};

use super::{
    arena::MetalArena,
    buffers::{
        empty_f32_buffer, empty_u32_buffer, empty_u8_buffer, read_f32_buffer, read_u32_buffer,
        require_byte_capacity, require_f32_capacity, u32_buffer, u64_buffer, u8_buffer_no_copy,
        write_f32_buffer, write_u32_buffer,
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
const Q8_0_BATCHED_MATVEC_TILED_KERNEL: &str = "q8_0_batched_matvec_tiled_f32_kernel";
const Q8_0_MATVEC_ADD_TILED_KERNEL: &str = "q8_0_matvec_add_tiled_f32_kernel";
const Q8_0_BATCHED_MATVEC_ADD_TILED_KERNEL: &str = "q8_0_batched_matvec_add_tiled_f32_kernel";
const Q8_0_TRANSPOSED_MATVEC_KERNEL: &str = "q8_0_transposed_matvec_f32_kernel";
const Q8_0_PACKED_HEADS_TRANSPOSED_MATVEC_KERNEL: &str =
    "q8_0_packed_heads_transposed_matvec_f32_kernel";
const Q8_0_PACKED_HEADS_MATVEC_KERNEL: &str = "q8_0_packed_heads_matvec_f32_kernel";
const ARGMAX_F32_KERNEL: &str = "argmax_f32_kernel";
const ARGMAX_ROWS_F32_KERNEL: &str = "argmax_rows_f32_kernel";
const Q2_K_SIMD_LANES: usize = 32;
const READY_EXPERT_ASSIGNMENTS_PER_KERNEL_GROUP: usize = 8;
const READY_EXPERT_OUTPUT_ROWS_PER_SIMDGROUP: usize = 4;
const Q8_0_BATCH_ROW_TILE: usize = 4;
const Q8_0_MAX_SIMDGROUPS_PER_OUTPUT: usize = 8;
const ARGMAX_THREADS_PER_VECTOR: usize = 256;
const ROUTED_EXPERT_COUNT: usize = 256;
// Thirty Q2 expert triplets per routed layer allocate about 27.9 GB on the
// 64 GB target. This is the measured decode optimum: thirty-two slots caused
// severe memory pressure, while a smaller adaptive cache produced more SSD
// misses. Long-context runtime rebalancing may still shrink this base cache.
const ROUTED_EXPERT_CACHE_SLOTS_PER_LAYER: usize = 30;
const ROUTED_EXPERT_PROTECTED_PERCENT: usize = 25;
// Chunking bounds thread creation while exposing enough independent large
// reads to keep the sidecar's no-cache SSD path busy.
const ROUTED_EXPERT_READ_WORKERS: usize = 32;
// Submit small ready-first groups so SSD reads remain overlapped with Metal
// execution without creating one command buffer per expert.
const ROUTED_EXPERT_WAVE_MIN_GROUPS: usize = 4;
static EXPERT_CACHE_LOCK_WARNING_EMITTED: AtomicBool = AtomicBool::new(false);

pub(crate) struct MetalQ2Matvec {
    arena: MetalArena,
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
    q8_0_batched_tiled_pipeline: ComputePipelineState,
    q8_0_tiled_add_pipeline: ComputePipelineState,
    q8_0_batched_tiled_add_pipeline: ComputePipelineState,
    q8_0_transposed_pipeline: ComputePipelineState,
    q8_0_packed_heads_transposed_pipeline: ComputePipelineState,
    q8_0_packed_heads_pipeline: ComputePipelineState,
    argmax_pipeline: ComputePipelineState,
    argmax_rows_pipeline: ComputePipelineState,
    scratch: Mutex<Q2ScratchBuffers>,
    ready_expert_cache: Mutex<Q2PerLayerExpertCache>,
    transient_expert_pool: Mutex<Q2TransientExpertPool>,
    expert_cache_counters: Q2ExpertCacheCounters,
    expert_queue: CommandQueue,
    expert_completion_event: SharedEvent,
    next_expert_completion_value: AtomicU64,
    pending_expert_submissions: Mutex<Vec<PendingExpertSubmission>>,
    weight_buffers: Mutex<HashMap<WeightBufferKey, Buffer>>,
}

#[derive(Debug, Default)]
struct Q2ExpertCacheCounters {
    lookups: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    ssd_read_bytes: AtomicU64,
    prefetch_lookups: AtomicU64,
    prefetch_hits: AtomicU64,
    prefetch_misses: AtomicU64,
    prefetch_ssd_read_bytes: AtomicU64,
    prefetch_nanoseconds: AtomicU64,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[derive(Debug, Default)]
struct Q2TransientExpertPool {
    layout: Option<Q2ExpertCacheLayout>,
    capacity: usize,
    storage: Option<Q2ExpertLayerBuffers>,
    initialized_slots: Vec<bool>,
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
struct ExpertPackFile {
    path: PathBuf,
    file: File,
    header: ExpertPackHeader,
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
    cached_miss_expert_id: Option<u32>,
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
    expert_id: u32,
    assignment_indices: Vec<usize>,
    gate: Q2ExpertSource<'a>,
    up: Q2ExpertSource<'a>,
    down: Q2ExpertSource<'a>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ExpertQueueKind {
    #[default]
    None,
    Probation,
    Protected,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ExpertDirectoryEntry {
    slot: Option<usize>,
    queue: ExpertQueueKind,
    previous: Option<u32>,
    next: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ExpertQueue {
    head: Option<u32>,
    tail: Option<u32>,
    len: usize,
}

#[derive(Debug)]
struct Q2ExpertLayerCache {
    initialized_slots: Vec<bool>,
    storage: Option<Q2ExpertLayerBuffers>,
    directory: Box<[ExpertDirectoryEntry; ROUTED_EXPERT_COUNT]>,
    slot_experts: Vec<Option<u32>>,
    probation: ExpertQueue,
    protected: ExpertQueue,
    protected_capacity: usize,
    free_slots: Vec<usize>,
    resident_count: usize,
}

#[derive(Debug)]
struct Q2PerLayerExpertCache {
    layout: Option<Q2ExpertCacheLayout>,
    layers: HashMap<usize, Arc<Mutex<Q2ExpertLayerCache>>>,
    model_file: Option<ExpertModelFile>,
    expert_pack: Option<ExpertPackFile>,
    slots_per_layer: usize,
}

impl Default for Q2PerLayerExpertCache {
    fn default() -> Self {
        Self {
            layout: None,
            layers: HashMap::new(),
            model_file: None,
            expert_pack: None,
            slots_per_layer: ROUTED_EXPERT_CACHE_SLOTS_PER_LAYER,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
struct PendingExpertSubmission {
    completion_value: u64,
    waves: Vec<SubmittedReadyWave>,
    marker: CommandBuffer,
}

#[derive(Debug)]
pub(crate) struct ReadyRoutedExperts {
    /// Expert rows are device-resident but may still be executing when this
    /// value returns. Consumers must wait on `completion_value` on the GPU.
    pub(crate) output: Buffer,
    pub(crate) completion_value: u64,
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
    pub(crate) fn new(device: &Device, library: &MetalLibrary, arena: MetalArena) -> Result<Self> {
        Ok(Self {
            arena,
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
            q8_0_batched_tiled_pipeline: compute_pipeline(
                device,
                library,
                Q8_0_BATCHED_MATVEC_TILED_KERNEL,
            )?,
            q8_0_tiled_add_pipeline: compute_pipeline(
                device,
                library,
                Q8_0_MATVEC_ADD_TILED_KERNEL,
            )?,
            q8_0_batched_tiled_add_pipeline: compute_pipeline(
                device,
                library,
                Q8_0_BATCHED_MATVEC_ADD_TILED_KERNEL,
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
            transient_expert_pool: Mutex::new(Q2TransientExpertPool::default()),
            expert_cache_counters: Q2ExpertCacheCounters::default(),
            expert_queue: device.new_command_queue(),
            expert_completion_event: device.new_shared_event(),
            next_expert_completion_value: AtomicU64::new(0),
            pending_expert_submissions: Mutex::new(Vec::new()),
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
                .checked_add(layer.resident_count)
                .ok_or_else(|| Error::backend("Q2 resident expert count overflow"))?;
            allocated_slots = allocated_slots
                .checked_add(
                    layer
                        .initialized_slots
                        .iter()
                        .filter(|allocated| **allocated)
                        .count(),
                )
                .ok_or_else(|| Error::backend("Q2 allocated expert slot count overflow"))?;
            capacity_slots = capacity_slots
                .checked_add(layer.initialized_slots.len())
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
            prefetch_lookups: self
                .expert_cache_counters
                .prefetch_lookups
                .load(Ordering::Relaxed),
            prefetch_hits: self
                .expert_cache_counters
                .prefetch_hits
                .load(Ordering::Relaxed),
            prefetch_misses: self
                .expert_cache_counters
                .prefetch_misses
                .load(Ordering::Relaxed),
            prefetch_ssd_read_bytes: self
                .expert_cache_counters
                .prefetch_ssd_read_bytes
                .load(Ordering::Relaxed),
            prefetch_nanoseconds: self
                .expert_cache_counters
                .prefetch_nanoseconds
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
        if slots_per_layer == 0 || slots_per_layer > ROUTED_EXPERT_COUNT {
            return Err(Error::backend(
                "Q2 expert cache slots per layer must be within 1..=256",
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
        if slots_per_layer == 0 || slots_per_layer > ROUTED_EXPERT_COUNT {
            return Err(Error::backend(
                "Q2 expert cache slots per layer must be within 1..=256",
            ));
        }
        let mut cache = self
            .ready_expert_cache
            .lock()
            .map_err(|_| Error::backend("Q2 per-layer expert cache lock poisoned"))?;
        cache.resize(slots_per_layer)
    }

    pub(crate) fn configure_expert_pack(
        &self,
        path: &Path,
        expected_header: ExpertPackHeader,
    ) -> Result<()> {
        let mut cache = self
            .ready_expert_cache
            .lock()
            .map_err(|_| Error::backend("Q2 per-layer expert cache lock poisoned"))?;
        if !cache.layers.is_empty() {
            return Err(Error::backend(
                "Q2 expert pack must be configured before generation starts",
            ));
        }
        if let Some(configured) = &cache.expert_pack {
            if configured.path == path && configured.header == expected_header {
                return Ok(());
            }
            return Err(Error::backend(
                "a different Q2 expert pack is already configured",
            ));
        }

        let mut file = File::open(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        // The explicit expert cache owns useful reuse. Avoid filling the much
        // smaller macOS page cache with one-time reads from the 241 GB pack.
        let no_cache = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
        if no_cache == -1 {
            return Err(Error::Io {
                path: path.to_path_buf(),
                source: io::Error::last_os_error(),
            });
        }
        let mut encoded = [0_u8; EXPERT_PACK_HEADER_BYTES];
        file.read_exact(&mut encoded).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let header = ExpertPackHeader::decode(&encoded)?;
        if header != expected_header {
            return Err(Error::backend(format!(
                "Q2 expert pack {} does not match the selected GGUF layout",
                path.display()
            )));
        }
        let file_bytes = file
            .metadata()
            .map_err(|source| Error::Io {
                path: path.to_path_buf(),
                source,
            })?
            .len();
        header.validate_file_bytes(file_bytes)?;
        cache.expert_pack = Some(ExpertPackFile {
            path: path.to_path_buf(),
            file,
            header,
        });
        Ok(())
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

        let use_q8_batch = kind == QuantMatvecKind::Q80 && row_count >= 2;
        let pipeline = match kind {
            QuantMatvecKind::Q2K => &self.pipeline,
            QuantMatvecKind::Q2KTransposed => &self.transposed_pipeline,
            QuantMatvecKind::Q80 if use_q8_batch => &self.q8_0_batched_tiled_pipeline,
            QuantMatvecKind::Q80 => &self.q8_0_tiled_pipeline,
            QuantMatvecKind::Q80Transposed => &self.q8_0_transposed_pipeline,
        };
        let weight_buffer = self.weight_buffer(device, weights)?;
        let output_buffer = self.arena.empty_f32(output_len)?;
        let row_count_buffer = self.arena.u32(matvec_u32(row_count, "row_count")?)?;
        let in_features_buffer = self.arena.u32(matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer = self.arena.u32(matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer = self
            .arena
            .u32(matvec_u32(blocks_per_row, "blocks_per_row")?)?;

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
            let simdgroups_per_output_buffer = self
                .arena
                .u32(matvec_u32(simdgroups_per_output, "simdgroups_per_output")?)?;
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
                if use_q8_batch {
                    row_count.div_ceil(Q8_0_BATCH_ROW_TILE) * out_features
                } else {
                    output_len
                },
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
        let output_buffer = self.arena.empty_f32(output_len)?;
        let row_count_buffer = self.arena.u32(matvec_u32(row_count, "row_count")?)?;
        let head_count_buffer = self.arena.u32(matvec_u32(head_count, "head_count")?)?;
        let in_features_buffer = self.arena.u32(matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer = self.arena.u32(matvec_u32(out_features, "out_features")?)?;
        let blocks_per_input_row_buffer = self
            .arena
            .u32(matvec_u32(blocks_per_input_row, "blocks_per_input_row")?)?;
        let blocks_per_head_buffer = self
            .arena
            .u32(matvec_u32(blocks_per_head, "blocks_per_head")?)?;

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
        let output_buffer = self.arena.empty_f32(output_len)?;
        let row_count_buffer = self.arena.u32(matvec_u32(row_count, "row_count")?)?;
        let head_count_buffer = self.arena.u32(matvec_u32(head_count, "head_count")?)?;
        let in_features_buffer = self.arena.u32(matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer = self.arena.u32(matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer = self
            .arena
            .u32(matvec_u32(blocks_per_row, "blocks_per_row")?)?;
        let blocks_per_head_buffer = self
            .arena
            .u32(matvec_u32(blocks_per_head, "blocks_per_head")?)?;

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
        let output_buffer = self.arena.empty_f32(output_len)?;
        let row_count_buffer = self.arena.u32(matvec_u32(row_count, "row_count")?)?;
        let in_features_buffer = self.arena.u32(matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer = self.arena.u32(matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer = self
            .arena
            .u32(matvec_u32(blocks_per_row, "blocks_per_row")?)?;

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
        let output_buffer = self.arena.empty_f32(output_len)?;
        let row_count_buffer = self.arena.u32(matvec_u32(row_count, "row_count")?)?;
        let in_features_buffer = self.arena.u32(matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer = self.arena.u32(matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer = self
            .arena
            .u32(matvec_u32(blocks_per_row, "blocks_per_row")?)?;
        let simdgroups_per_output_buffer = self
            .arena
            .u32(matvec_u32(simdgroups_per_output, "simdgroups_per_output")?)?;

        let use_q8_batch = row_count >= 2;
        encode_1d_threadgroups(
            command_buffer,
            if use_q8_batch {
                &self.q8_0_batched_tiled_add_pipeline
            } else {
                &self.q8_0_tiled_add_pipeline
            },
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
            if use_q8_batch {
                row_count.div_ceil(Q8_0_BATCH_ROW_TILE) * out_features
            } else {
                output_len
            },
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
        let output_buffer = self.arena.empty_f32(output_len)?;
        let token_count_buffer = self.arena.u32(matvec_u32(token_count, "token_count")?)?;
        let assignment_count_buffer = self
            .arena
            .u32(matvec_u32(assignment_count, "assignment_count")?)?;
        let in_features_buffer = self.arena.u32(matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer = self.arena.u32(matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer = self.arena.u32(matvec_u32(gate_blocks, "blocks_per_row")?)?;
        let expert_stride_buffer = self
            .arena
            .u32(matvec_u32(expert_stride_bytes, "expert_stride_bytes")?)?;

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
        let output_buffer = self.arena.empty_f32(output_len)?;
        let assignment_count_buffer = self
            .arena
            .u32(matvec_u32(assignment_count, "assignment_count")?)?;
        let in_features_buffer = self.arena.u32(matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer = self.arena.u32(matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer = self
            .arena
            .u32(matvec_u32(blocks_per_row, "blocks_per_row")?)?;
        let expert_stride_buffer = self
            .arena
            .u32(matvec_u32(expert_stride_bytes, "expert_stride_bytes")?)?;

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
        let logits_buffer = self.arena.empty_f32(output_len)?;
        let token_id_buffer = self.arena.empty_u32(1)?;
        let token_score_buffer = self.arena.empty_f32(1)?;
        let row_count_buffer = self.arena.u32(matvec_u32(row_count, "row_count")?)?;
        let in_features_buffer = self.arena.u32(matvec_u32(in_features, "in_features")?)?;
        let out_features_buffer = self.arena.u32(matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row_buffer = self
            .arena
            .u32(matvec_u32(blocks_per_row, "blocks_per_row")?)?;
        let output_len_buffer = self.arena.u32(matvec_u32(output_len, "output_len")?)?;

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
        _device: &Device,
        scores: &Buffer,
        value_count: usize,
    ) -> Result<(Buffer, Buffer)> {
        if value_count == 0 {
            return Err(Error::backend("f32 argmax requires at least one value"));
        }
        require_f32_capacity(scores, value_count, "f32 argmax scores")?;
        let argmax_threads = argmax_threads(&self.argmax_pipeline, "batched f32 greedy argmax")?;
        let token_id_buffer = self.arena.empty_u32(1)?;
        let token_score_buffer = self.arena.empty_f32(1)?;
        let value_count_buffer = self.arena.u32(matvec_u32(value_count, "value_count")?)?;

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
        _device: &Device,
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
        let token_id_buffer = self.arena.empty_u32(row_count)?;
        let token_score_buffer = self.arena.empty_f32(row_count)?;
        let row_width_buffer = self.arena.u32(matvec_u32(row_width, "row_width")?)?;

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
        let (layer_cache, read_file, read_path, expert_pack_header) = {
            let mut cache = self
                .ready_expert_cache
                .lock()
                .map_err(|_| Error::backend("Q2 per-layer expert cache lock poisoned"))?;
            cache.ensure_model_file(model_path)?;
            cache.ensure_layout(gate_stride, up_stride, down_stride);
            let layer_cache = cache.layer(layer_index);
            let (read_file, read_path, expert_pack_header) = match &cache.expert_pack {
                Some(pack) => (
                    pack.file.try_clone().map_err(|source| Error::Io {
                        path: pack.path.clone(),
                        source,
                    })?,
                    pack.path.clone(),
                    Some(pack.header),
                ),
                None => {
                    let model_file = cache
                        .model_file
                        .as_ref()
                        .ok_or_else(|| Error::backend("Q2 expert model file is not initialized"))?;
                    (
                        model_file.file.try_clone().map_err(|source| Error::Io {
                            path: model_path.to_path_buf(),
                            source,
                        })?,
                        model_path.to_path_buf(),
                        None,
                    )
                }
            };
            (layer_cache, read_file, read_path, expert_pack_header)
        };

        let mut layer_cache = layer_cache
            .lock()
            .map_err(|_| Error::backend("Q2 layer expert cache lock poisoned"))?;
        let slots_per_layer = layer_cache.initialized_slots.len();
        let can_reuse_transient_pool = self
            .pending_expert_submissions
            .lock()
            .map_err(|_| Error::backend("pending routed expert submission lock poisoned"))?
            .is_empty();
        let mut transient_pool = if can_reuse_transient_pool {
            Some(
                self.transient_expert_pool
                    .lock()
                    .map_err(|_| Error::backend("Q2 transient expert pool lock poisoned"))?,
            )
        } else {
            None
        };
        let groups = prepare_ready_expert_groups(
            device,
            &mut layer_cache,
            transient_pool.as_mut().map(|pool| &mut **pool),
            gate_payloads,
            up_payloads,
            down_payloads,
            gate_stride,
            up_stride,
            down_stride,
            layer_index,
            expert_pack_header,
            false,
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
        let cached_miss_expert_ids = groups
            .iter()
            .filter_map(|group| group.cached_miss_expert_id)
            .collect::<Vec<_>>();

        let started = Instant::now();
        let execution = self.execute_ready_expert_groups(
            device,
            &read_file,
            &read_path,
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
        let (output, ready_waves, first_ready_ms, ssd_load_nanoseconds, completion_value) =
            match execution {
                Ok(output) => output,
                Err(error) => {
                    layer_cache.invalidate(&cached_miss_expert_ids)?;
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
            completion_value,
            gpu_timing_deferred = true,
            first_ready_ms,
            elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0,
            slots_per_layer,
            "completed ready-first Q2 routed experts"
        );

        Ok(ReadyRoutedExperts {
            output,
            completion_value,
            selected_experts: groups.len(),
            cache_hits,
            cache_misses,
            transient_experts,
            read_bytes,
            ready_waves,
        })
    }

    pub(crate) fn prefetch_ready_routed_experts(
        &self,
        device: &Device,
        layer_index: usize,
        model_path: &Path,
        gate_payloads: &[Q2ExpertSource<'_>],
        up_payloads: &[Q2ExpertSource<'_>],
        down_payloads: &[Q2ExpertSource<'_>],
    ) -> Result<()> {
        if gate_payloads.is_empty() {
            return Ok(());
        }
        validate_exact_len(
            "predictive expert up assignment count",
            up_payloads.len(),
            gate_payloads.len(),
        )?;
        validate_exact_len(
            "predictive expert down assignment count",
            down_payloads.len(),
            gate_payloads.len(),
        )?;
        let gate_stride = uniform_payload_stride(gate_payloads, "predictive expert gate")?;
        let up_stride = uniform_payload_stride(up_payloads, "predictive expert up")?;
        let down_stride = uniform_payload_stride(down_payloads, "predictive expert down")?;
        let predicted_experts = gate_payloads
            .iter()
            .map(|source| source.expert_id)
            .collect::<HashSet<_>>()
            .len();

        let started = Instant::now();
        let (layer_cache, read_file, read_path, expert_pack_header) = {
            let mut cache = self
                .ready_expert_cache
                .lock()
                .map_err(|_| Error::backend("Q2 per-layer expert cache lock poisoned"))?;
            cache.ensure_model_file(model_path)?;
            cache.ensure_layout(gate_stride, up_stride, down_stride);
            let layer_cache = cache.layer(layer_index);
            let (read_file, read_path, expert_pack_header) = match &cache.expert_pack {
                Some(pack) => (
                    pack.file.try_clone().map_err(|source| Error::Io {
                        path: pack.path.clone(),
                        source,
                    })?,
                    pack.path.clone(),
                    Some(pack.header),
                ),
                None => {
                    let model_file = cache
                        .model_file
                        .as_ref()
                        .ok_or_else(|| Error::backend("Q2 expert model file is not initialized"))?;
                    (
                        model_file.file.try_clone().map_err(|source| Error::Io {
                            path: model_path.to_path_buf(),
                            source,
                        })?,
                        model_path.to_path_buf(),
                        None,
                    )
                }
            };
            (layer_cache, read_file, read_path, expert_pack_header)
        };

        // Hold the layer lock until every predicted slot is complete. Exact
        // routing can run concurrently, but it cannot observe a partially read
        // Metal buffer.
        let mut layer_cache = layer_cache
            .lock()
            .map_err(|_| Error::backend("Q2 predictive expert cache lock poisoned"))?;
        let groups = prepare_ready_expert_groups(
            device,
            &mut layer_cache,
            None,
            gate_payloads,
            up_payloads,
            down_payloads,
            gate_stride,
            up_stride,
            down_stride,
            layer_index,
            expert_pack_header,
            true,
        )?;
        let prefetch_hits = groups.iter().filter(|group| group.cache_hit).count();
        let prefetch_misses = groups.len().saturating_sub(prefetch_hits);
        let read_bytes = groups.iter().try_fold(0_u64, |total, group| {
            group.read_tasks.iter().try_fold(total, |total, task| {
                total
                    .checked_add(u64::try_from(task.byte_len).map_err(|_| {
                        Error::backend("predictive expert read size does not fit u64")
                    })?)
                    .ok_or_else(|| Error::backend("predictive expert read bytes overflow"))
            })
        })?;
        let cached_miss_expert_ids = groups
            .iter()
            .filter_map(|group| group.cached_miss_expert_id)
            .collect::<Vec<_>>();
        if let Err(error) = pread_ready_expert_groups(&read_file, &read_path, &groups) {
            layer_cache.invalidate(&cached_miss_expert_ids)?;
            return Err(error);
        }
        let elapsed_nanoseconds = elapsed_nanoseconds(started.elapsed());

        self.expert_cache_counters
            .prefetch_lookups
            .fetch_add(predicted_experts as u64, Ordering::Relaxed);
        self.expert_cache_counters
            .prefetch_hits
            .fetch_add(prefetch_hits as u64, Ordering::Relaxed);
        self.expert_cache_counters
            .prefetch_misses
            .fetch_add(prefetch_misses as u64, Ordering::Relaxed);
        self.expert_cache_counters
            .prefetch_ssd_read_bytes
            .fetch_add(read_bytes, Ordering::Relaxed);
        self.expert_cache_counters
            .prefetch_nanoseconds
            .fetch_add(elapsed_nanoseconds, Ordering::Relaxed);
        self.expert_cache_counters
            .ssd_read_bytes
            .fetch_add(read_bytes, Ordering::Relaxed);
        debug!(
            target: "inferno::expert_predictor",
            layer_index,
            predicted_experts,
            prefetch_hits,
            prefetch_misses,
            skipped_experts = predicted_experts.saturating_sub(groups.len()),
            read_bytes,
            elapsed_ms = elapsed_nanoseconds as f64 / 1_000_000.0,
            "prefetched predicted experts into the Metal cache"
        );
        Ok(())
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
        let gated = self.arena.empty_f32(gated_len)?;
        let output = self.arena.empty_f32(output_len)?;
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
                        let mut gate_up_ready = Vec::with_capacity(worker_jobs.len());
                        for &group_index in worker_jobs {
                            let group = &groups[group_index];
                            let gate_up_result =
                                pread_ready_expert_gate_up(model_file, model_path, group);
                            let gate_up_succeeded = gate_up_result.is_ok();
                            if sender
                                .send((group_index, ReadyExpertPhase::GateUp, gate_up_result))
                                .is_err()
                            {
                                return;
                            }
                            if gate_up_succeeded {
                                gate_up_ready.push(group_index);
                            }
                        }

                        for group_index in gate_up_ready {
                            let group = &groups[group_index];
                            let down_result =
                                pread_ready_expert_down(model_file, model_path, group);
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
            let mut pending_gate_up = Vec::with_capacity(ROUTED_EXPERT_WAVE_MIN_GROUPS);
            let mut pending_down = cached.clone();
            while let Ok(first) = receiver.recv() {
                let mut completed = vec![first];
                completed.extend(receiver.try_iter());
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

                let all_gate_up_ready = completed_gate_up == misses.len();
                if pending_gate_up.len() >= ROUTED_EXPERT_WAVE_MIN_GROUPS
                    || (all_gate_up_ready && !pending_gate_up.is_empty())
                {
                    first_ready_ms.get_or_insert_with(|| started.elapsed().as_secs_f64() * 1_000.0);
                    let ready = std::mem::take(&mut pending_gate_up);
                    waves.push(self.submit_ready_expert_wave(
                        device,
                        groups,
                        &ready,
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
                if all_gate_up_ready
                    && (pending_down.len() >= ROUTED_EXPERT_WAVE_MIN_GROUPS
                        || completed_down == misses.len())
                    && !pending_down.is_empty()
                {
                    let ready = std::mem::take(&mut pending_down);
                    waves.push(self.submit_ready_expert_wave(
                        device,
                        groups,
                        &ready,
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
        if let Err(error) = load_result {
            if let Err(wave_error) = wait_ready_expert_waves(&waves) {
                debug!(
                    target: "inferno::expert_cache",
                    error = %wave_error,
                    "ready expert Metal wave also failed while handling an SSD load error"
                );
            }
            return Err(error);
        }
        let ready_waves = waves.len();
        let completion_value = self.submit_ready_expert_completion(waves)?;
        Ok((
            output,
            ready_waves,
            first_ready_ms.unwrap_or_default(),
            ssd_load_nanoseconds,
            completion_value,
        ))
    }

    fn submit_ready_expert_completion(&self, waves: Vec<SubmittedReadyWave>) -> Result<u64> {
        if waves.is_empty() {
            return Err(Error::backend(
                "ready routed expert execution produced no Metal waves",
            ));
        }
        let previous = self
            .next_expert_completion_value
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| Error::backend("routed expert completion value overflow"))?;
        let completion_value = previous + 1;
        let marker = self.expert_queue.new_command_buffer().to_owned();
        marker.encode_signal_event(&self.expert_completion_event, completion_value);
        marker.commit();

        self.pending_expert_submissions
            .lock()
            .map_err(|_| Error::backend("pending routed expert submission lock poisoned"))?
            .push(PendingExpertSubmission {
                completion_value,
                waves,
                marker,
            });
        trace!(
            target: "inferno::metal",
            completion_value,
            "submitted asynchronous routed expert completion marker"
        );
        Ok(completion_value)
    }

    /// Encodes a GPU-side wait immediately before work that consumes routed
    /// expert output. This does not block the CPU or drain either queue.
    pub(crate) fn encode_ready_expert_wait(
        &self,
        command_buffer: &CommandBufferRef,
        completion_value: u64,
    ) -> Result<()> {
        if completion_value == 0
            || completion_value > self.next_expert_completion_value.load(Ordering::Acquire)
        {
            return Err(Error::backend(format!(
                "invalid routed expert completion value {completion_value}"
            )));
        }
        command_buffer.encode_wait_for_event(&self.expert_completion_event, completion_value);
        trace!(
            target: "inferno::metal",
            completion_value,
            "encoded routed expert GPU dependency"
        );
        Ok(())
    }

    /// Validates expert-queue work at an existing host synchronization point.
    /// The main queue normally already waited for each shared-event value, so
    /// waiting for the final marker here does not add a new pipeline barrier.
    pub(crate) fn finish_ready_expert_submissions(&self) -> Result<()> {
        let submissions = {
            let mut pending = self
                .pending_expert_submissions
                .lock()
                .map_err(|_| Error::backend("pending routed expert submission lock poisoned"))?;
            std::mem::take(&mut *pending)
        };
        let Some(last) = submissions.last() else {
            return Ok(());
        };

        last.marker.wait_until_completed();
        let mut gpu_nanoseconds = 0_u64;
        for submission in &submissions {
            for (wave_index, wave) in submission.waves.iter().enumerate() {
                if wave.command_buffer.status() != MTLCommandBufferStatus::Completed {
                    return Err(Error::backend(format!(
                        "Q2 ready expert completion {} wave {wave_index} with {} assignments did not complete: {:?}",
                        submission.completion_value,
                        wave.assignment_count,
                        wave.command_buffer.status()
                    )));
                }
                gpu_nanoseconds = gpu_nanoseconds
                    .checked_add(command_buffer_gpu_nanoseconds(&wave.command_buffer))
                    .ok_or_else(|| Error::backend("Q2 expert GPU time overflow"))?;
            }
            if submission.marker.status() != MTLCommandBufferStatus::Completed {
                return Err(Error::backend(format!(
                    "Q2 ready expert completion marker {} did not complete: {:?}",
                    submission.completion_value,
                    submission.marker.status()
                )));
            }
        }
        self.expert_cache_counters
            .q2_matmul_gpu_nanoseconds
            .fetch_add(gpu_nanoseconds, Ordering::Relaxed);
        trace!(
            target: "inferno::expert_cache",
            submission_count = submissions.len(),
            gpu_ms = gpu_nanoseconds as f64 / 1_000_000.0,
            "validated asynchronous routed expert submissions"
        );
        Ok(())
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
        if ready_groups_share_expert_slab(groups, ready_groups) {
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
        let mut group_offsets = vec![0_u32];
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
            validate_ready_expert_group_width(group.assignment_indices.len())?;
            // A mixed wave uses indirect addresses for both transient
            // buffers and cache-slab slots. Slab buffers share one base
            // address, so the selected slot offset must be included.
            let gate_address =
                ready_expert_gpu_address(&group.buffers.gate, group.buffers.gate_offset, "gate")?;
            let up_address =
                ready_expert_gpu_address(&group.buffers.up, group.buffers.up_offset, "up")?;
            let down_address =
                ready_expert_gpu_address(&group.buffers.down, group.buffers.down_offset, "down")?;
            for assignment_group in group
                .assignment_indices
                .chunks(READY_EXPERT_ASSIGNMENTS_PER_KERNEL_GROUP)
            {
                gate_addresses.push(gate_address);
                up_addresses.push(up_address);
                down_addresses.push(down_address);
                gate_resources.push(&group.buffers.gate);
                up_resources.push(&group.buffers.up);
                down_resources.push(&group.buffers.down);
                for &assignment_index in assignment_group {
                    assignment_indices.push(u32::try_from(assignment_index).map_err(|_| {
                        Error::backend("ready routed assignment index exceeds Metal u32 limit")
                    })?);
                }
                group_offsets.push(u32::try_from(assignment_indices.len()).map_err(|_| {
                    Error::backend("ready routed assignment count exceeds Metal u32 limit")
                })?);
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

        let kernel_group_count = gate_addresses.len();
        let gate_addresses = u64_buffer(device, &gate_addresses)?;
        let up_addresses = u64_buffer(device, &up_addresses)?;
        let down_addresses = u64_buffer(device, &down_addresses)?;
        let assignment_indices_buffer = u32_buffer(device, &assignment_indices)?;
        let group_offsets_buffer = u32_buffer(device, &group_offsets)?;
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
                &group_offsets_buffer,
                kernel_group_count,
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
                &group_offsets_buffer,
                kernel_group_count,
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
        let mut group_offsets = vec![0_u32];
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
            validate_ready_expert_group_width(group.assignment_indices.len())?;
            for assignment_group in group
                .assignment_indices
                .chunks(READY_EXPERT_ASSIGNMENTS_PER_KERNEL_GROUP)
            {
                slot_indices.push(slot);
                for &assignment_index in assignment_group {
                    assignment_indices.push(u32::try_from(assignment_index).map_err(|_| {
                        Error::backend("ready routed assignment index exceeds Metal u32 limit")
                    })?);
                }
                group_offsets.push(u32::try_from(assignment_indices.len()).map_err(|_| {
                    Error::backend("ready routed assignment count exceeds Metal u32 limit")
                })?);
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
        let group_offsets_buffer = u32_buffer(device, &group_offsets)?;
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
                &group_offsets_buffer,
                &slot_indices_buffer,
                slot_indices.len(),
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
                &group_offsets_buffer,
                &slot_indices_buffer,
                slot_indices.len(),
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
        _device: &Device,
        gate_addresses: &Buffer,
        up_addresses: &Buffer,
        gate_resources: &[&Buffer],
        up_resources: &[&Buffer],
        input: &Buffer,
        input_len: usize,
        token_indices: &Buffer,
        assignment_indices: &Buffer,
        group_offsets: &Buffer,
        ready_group_count: usize,
        ready_assignment_count: usize,
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
            ready_assignment_count,
            "ready routed gate/up assignment indices",
        )?;
        require_u32_buffer_capacity(
            group_offsets,
            ready_group_count + 1,
            "ready routed gate/up group offsets",
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
            ready_group_count
                .checked_mul(out_features.div_ceil(READY_EXPERT_OUTPUT_ROWS_PER_SIMDGROUP))
                .ok_or_else(|| Error::backend("ready routed gate/up thread count overflow"))?,
            "ready Q2_K routed gate/up",
        )?;
        let blocks_per_row_value = in_features / Q2_K_BLOCK_VALUES;
        let token_count = self.arena.u32(matvec_u32(token_count, "token_count")?)?;
        let assignment_count = self
            .arena
            .u32(matvec_u32(assignment_count, "assignment_count")?)?;
        let ready_group_count = self
            .arena
            .u32(matvec_u32(ready_group_count, "ready_group_count")?)?;
        let in_features = self.arena.u32(matvec_u32(in_features, "in_features")?)?;
        let out_features = self.arena.u32(matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row = self
            .arena
            .u32(matvec_u32(blocks_per_row_value, "blocks_per_row")?)?;
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
                group_offsets,
                output,
                &token_count,
                &assignment_count,
                &ready_group_count,
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
        _device: &Device,
        weight_addresses: &Buffer,
        weight_resources: &[&Buffer],
        input: &Buffer,
        assignment_indices: &Buffer,
        group_offsets: &Buffer,
        ready_group_count: usize,
        ready_assignment_count: usize,
        assignment_count: usize,
        in_features: usize,
        out_features: usize,
        output: &Buffer,
    ) -> Result<()> {
        require_u32_buffer_capacity(
            assignment_indices,
            ready_assignment_count,
            "ready routed down assignment indices",
        )?;
        require_u32_buffer_capacity(
            group_offsets,
            ready_group_count + 1,
            "ready routed down group offsets",
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
            ready_group_count
                .checked_mul(out_features.div_ceil(READY_EXPERT_OUTPUT_ROWS_PER_SIMDGROUP))
                .ok_or_else(|| Error::backend("ready routed down thread count overflow"))?,
            "ready Q2_K routed down",
        )?;
        let blocks_per_row_value = in_features / Q2_K_BLOCK_VALUES;
        let assignment_count = self
            .arena
            .u32(matvec_u32(assignment_count, "assignment_count")?)?;
        let ready_group_count = self
            .arena
            .u32(matvec_u32(ready_group_count, "ready_group_count")?)?;
        let in_features = self.arena.u32(matvec_u32(in_features, "in_features")?)?;
        let out_features = self.arena.u32(matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row = self
            .arena
            .u32(matvec_u32(blocks_per_row_value, "blocks_per_row")?)?;
        encode_1d_with_indirect_reads(
            command_buffer,
            &self.ready_matvec_pipeline,
            &[
                weight_addresses,
                input,
                assignment_indices,
                group_offsets,
                output,
                &assignment_count,
                &ready_group_count,
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
        _device: &Device,
        gate_weights: &Buffer,
        up_weights: &Buffer,
        input: &Buffer,
        input_len: usize,
        token_indices: &Buffer,
        assignment_indices: &Buffer,
        group_offsets: &Buffer,
        slot_indices: &Buffer,
        ready_group_count: usize,
        ready_assignment_count: usize,
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
            ready_assignment_count,
            "ready routed slot assignment indices",
        )?;
        require_u32_buffer_capacity(
            group_offsets,
            ready_group_count + 1,
            "ready routed slot group offsets",
        )?;
        require_u32_buffer_capacity(
            slot_indices,
            ready_group_count,
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
            ready_group_count
                .checked_mul(out_features.div_ceil(READY_EXPERT_OUTPUT_ROWS_PER_SIMDGROUP))
                .ok_or_else(|| Error::backend("ready routed slot gate/up threads overflow"))?,
            "ready Q2_K slot gate/up",
        )?;
        let blocks_per_row_value = in_features / Q2_K_BLOCK_VALUES;
        let token_count = self.arena.u32(matvec_u32(token_count, "token_count")?)?;
        let assignment_count = self
            .arena
            .u32(matvec_u32(assignment_count, "assignment_count")?)?;
        let ready_group_count = self
            .arena
            .u32(matvec_u32(ready_group_count, "ready_group_count")?)?;
        let in_features = self.arena.u32(matvec_u32(in_features, "in_features")?)?;
        let out_features = self.arena.u32(matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row = self
            .arena
            .u32(matvec_u32(blocks_per_row_value, "blocks_per_row")?)?;
        let expert_stride_bytes = self
            .arena
            .u32(matvec_u32(expert_stride_bytes, "expert_stride_bytes")?)?;
        let slot_count = self.arena.u32(matvec_u32(slot_count, "slot_count")?)?;
        encode_1d(
            command_buffer,
            &self.ready_slot_gate_up_swiglu_pipeline,
            &[
                gate_weights,
                up_weights,
                input,
                token_indices,
                assignment_indices,
                group_offsets,
                slot_indices,
                output,
                &token_count,
                &assignment_count,
                &ready_group_count,
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
        _device: &Device,
        weights: &Buffer,
        input: &Buffer,
        assignment_indices: &Buffer,
        group_offsets: &Buffer,
        slot_indices: &Buffer,
        ready_group_count: usize,
        ready_assignment_count: usize,
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
            ready_assignment_count,
            "ready routed slot down assignment indices",
        )?;
        require_u32_buffer_capacity(
            group_offsets,
            ready_group_count + 1,
            "ready routed slot down group offsets",
        )?;
        require_u32_buffer_capacity(
            slot_indices,
            ready_group_count,
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
            ready_group_count
                .checked_mul(out_features.div_ceil(READY_EXPERT_OUTPUT_ROWS_PER_SIMDGROUP))
                .ok_or_else(|| Error::backend("ready routed slot down threads overflow"))?,
            "ready Q2_K slot down",
        )?;
        let blocks_per_row_value = in_features / Q2_K_BLOCK_VALUES;
        let assignment_count = self
            .arena
            .u32(matvec_u32(assignment_count, "assignment_count")?)?;
        let ready_group_count = self
            .arena
            .u32(matvec_u32(ready_group_count, "ready_group_count")?)?;
        let in_features = self.arena.u32(matvec_u32(in_features, "in_features")?)?;
        let out_features = self.arena.u32(matvec_u32(out_features, "out_features")?)?;
        let blocks_per_row = self
            .arena
            .u32(matvec_u32(blocks_per_row_value, "blocks_per_row")?)?;
        let expert_stride_bytes = self
            .arena
            .u32(matvec_u32(expert_stride_bytes, "expert_stride_bytes")?)?;
        let slot_count = self.arena.u32(matvec_u32(slot_count, "slot_count")?)?;
        encode_1d(
            command_buffer,
            &self.ready_slot_matvec_pipeline,
            &[
                weights,
                input,
                assignment_indices,
                group_offsets,
                slot_indices,
                output,
                &assignment_count,
                &ready_group_count,
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
        lock_ranges: bool,
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
        if lock_ranges {
            self.gate.lock_range(gate_offset, gate_stride, "gate")?;
            self.up.lock_range(up_offset, up_stride, "up")?;
            self.down.lock_range(down_offset, down_stride, "down")?;
        }
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

impl Q2TransientExpertPool {
    fn ensure(
        &mut self,
        device: &Device,
        capacity: usize,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
    ) -> Result<()> {
        if capacity == 0 {
            return Err(Error::backend(
                "Q2 transient expert pool capacity must be positive",
            ));
        }
        let layout = Q2ExpertCacheLayout {
            gate_stride,
            up_stride,
            down_stride,
        };
        if self.layout == Some(layout) && self.capacity >= capacity {
            return Ok(());
        }

        self.storage = Some(Q2ExpertLayerBuffers::new(
            device,
            capacity,
            gate_stride,
            up_stride,
            down_stride,
        )?);
        self.layout = Some(layout);
        self.capacity = capacity;
        self.initialized_slots = vec![false; capacity];
        Ok(())
    }

    fn ready_buffers(
        &mut self,
        slot: usize,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
    ) -> Result<ReadyExpertBuffers> {
        if slot >= self.capacity {
            return Err(Error::backend(format!(
                "Q2 transient expert slot {slot} exceeds pool capacity {}",
                self.capacity
            )));
        }
        let storage = self
            .storage
            .as_mut()
            .ok_or_else(|| Error::backend("Q2 transient expert pool is not allocated"))?;
        let initialized = *self
            .initialized_slots
            .get(slot)
            .ok_or_else(|| Error::backend("Q2 transient expert initialization is missing"))?;
        let mut buffers =
            storage.ready_buffers(slot, gate_stride, up_stride, down_stride, !initialized)?;
        self.initialized_slots[slot] = true;
        // Transient and persistent experts can share one wave, so dispatch by
        // explicit GPU addresses instead of treating this as a layer-cache slot.
        buffers.slot_index = None;
        Ok(buffers)
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

fn ready_expert_gpu_address(buffer: &Buffer, byte_offset: usize, component: &str) -> Result<u64> {
    let base = buffer.gpu_address();
    if base == 0 {
        return Err(Error::backend(format!(
            "ready routed {component} expert buffer has no GPU address"
        )));
    }
    let byte_offset = u64::try_from(byte_offset).map_err(|_| {
        Error::backend(format!(
            "ready routed {component} expert offset exceeds Metal u64 limit"
        ))
    })?;
    base.checked_add(byte_offset).ok_or_else(|| {
        Error::backend(format!(
            "ready routed {component} expert GPU address overflow"
        ))
    })
}

fn validate_ready_expert_group_width(assignment_count: usize) -> Result<()> {
    if assignment_count == 0 {
        return Err(Error::backend(
            "ready routed expert group must contain at least one assignment",
        ));
    }
    Ok(())
}

fn ready_groups_share_expert_slab(groups: &[ReadyExpertGroup], ready_groups: &[usize]) -> bool {
    let Some(first) = ready_groups.first().and_then(|&index| groups.get(index)) else {
        return false;
    };
    if first.buffers.slot_index.is_none() {
        return false;
    }
    ready_groups.iter().all(|&index| {
        groups.get(index).is_some_and(|group| {
            group.buffers.slot_index.is_some()
                && group.buffers.gate.gpu_address() == first.buffers.gate.gpu_address()
                && group.buffers.up.gpu_address() == first.buffers.up.gpu_address()
                && group.buffers.down.gpu_address() == first.buffers.down.gpu_address()
        })
    })
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
        if let Some(pack) = &self.expert_pack {
            let source_bytes = file
                .metadata()
                .map_err(|source| Error::Io {
                    path: path.to_path_buf(),
                    source,
                })?
                .len();
            if source_bytes != pack.header.source_bytes {
                return Err(Error::backend(format!(
                    "Q2 expert pack expects a {}-byte GGUF, but {} has {source_bytes} bytes",
                    pack.header.source_bytes,
                    path.display()
                )));
            }
        }
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
        debug_assert!(capacity > 0 && capacity <= ROUTED_EXPERT_COUNT);
        Self {
            initialized_slots: vec![false; capacity],
            storage: None,
            directory: Box::new([ExpertDirectoryEntry::default(); ROUTED_EXPERT_COUNT]),
            slot_experts: vec![None; capacity],
            probation: ExpertQueue::default(),
            protected: ExpertQueue::default(),
            protected_capacity: expert_protected_capacity(capacity),
            free_slots: (0..capacity).rev().collect(),
            resident_count: 0,
        }
    }

    fn resize(&mut self, capacity: usize) -> Result<()> {
        if capacity == 0 || capacity > ROUTED_EXPERT_COUNT {
            return Err(Error::backend(
                "Q2 expert layer cache capacity must be within 1..=256",
            ));
        }
        if capacity == self.initialized_slots.len() {
            return Ok(());
        }
        *self = Self::new(capacity);
        Ok(())
    }

    /// Resolves one router selection as a batch. Selected residents are first
    /// detached from the intrusive queues, so every lookup, touch, insertion,
    /// and victim removal is O(1) without scanning around pinned experts.
    fn resolve_slots(
        &mut self,
        expert_ids: &[u32],
        predictive_prefetch: bool,
    ) -> Result<Vec<LayerSlotResolution>> {
        let mut seen = [false; ROUTED_EXPERT_COUNT];
        for &expert_id in expert_ids {
            let index = expert_directory_index(expert_id)?;
            if std::mem::replace(&mut seen[index], true) {
                return Err(Error::backend(format!(
                    "Q2 expert cache batch contains duplicate expert ID {expert_id}"
                )));
            }
        }

        let mut original_queues = Vec::with_capacity(expert_ids.len());
        for &expert_id in expert_ids {
            let entry = self.directory[expert_id as usize];
            if entry.slot.is_some() {
                if entry.queue == ExpertQueueKind::None {
                    return Err(Error::backend(format!(
                        "resident Q2 expert {expert_id} is missing from the SLRU queues"
                    )));
                }
                self.unlink(expert_id)?;
            } else if entry.queue != ExpertQueueKind::None {
                return Err(Error::backend(format!(
                    "non-resident Q2 expert {expert_id} remains linked in the SLRU queues"
                )));
            }
            original_queues.push(entry.queue);
        }

        let mut resolutions = Vec::with_capacity(expert_ids.len());
        for &expert_id in expert_ids {
            if let Some(slot) = self.directory[expert_id as usize].slot {
                resolutions.push(LayerSlotResolution::Hit(slot));
                continue;
            }
            let slot = match self.free_slots.pop() {
                Some(slot) => Some(slot),
                None => self.evict_oldest()?,
            };
            match slot {
                Some(slot) => {
                    self.claim_slot(expert_id, slot)?;
                    resolutions.push(LayerSlotResolution::Miss(slot));
                }
                None => resolutions.push(LayerSlotResolution::Transient),
            }
        }

        for ((&expert_id, &original_queue), &resolution) in
            expert_ids.iter().zip(&original_queues).zip(&resolutions)
        {
            let target_queue = match resolution {
                LayerSlotResolution::Hit(_) if predictive_prefetch => original_queue,
                LayerSlotResolution::Hit(_) => ExpertQueueKind::Protected,
                LayerSlotResolution::Miss(_) => ExpertQueueKind::Probation,
                LayerSlotResolution::Transient => continue,
            };
            if target_queue == ExpertQueueKind::None {
                return Err(Error::backend(format!(
                    "resolved Q2 expert {expert_id} has no SLRU destination"
                )));
            }
            self.link_back(expert_id, target_queue)?;
            self.enforce_protected_capacity()?;
        }
        Ok(resolutions)
    }

    fn buffers(
        &mut self,
        device: &Device,
        slot: usize,
        gate_stride: usize,
        up_stride: usize,
        down_stride: usize,
    ) -> Result<ReadyExpertBuffers> {
        let initialized = *self.initialized_slots.get(slot).ok_or_else(|| {
            Error::backend(format!("Q2 layer expert slot {slot} is out of bounds"))
        })?;
        if self.storage.is_none() {
            self.storage = Some(Q2ExpertLayerBuffers::new(
                device,
                self.initialized_slots.len(),
                gate_stride,
                up_stride,
                down_stride,
            )?);
        }
        let buffers = self
            .storage
            .as_mut()
            .ok_or_else(|| Error::backend("Q2 layer expert slab allocation failed"))?
            .ready_buffers(slot, gate_stride, up_stride, down_stride, !initialized)?;
        self.initialized_slots[slot] = true;
        Ok(buffers)
    }

    #[cfg(test)]
    fn contains(&self, expert_id: u32) -> bool {
        expert_directory_index(expert_id)
            .ok()
            .is_some_and(|index| self.directory[index].slot.is_some())
    }

    #[cfg(test)]
    fn queue_kind(&self, expert_id: u32) -> Option<ExpertQueueKind> {
        expert_directory_index(expert_id)
            .ok()
            .map(|index| self.directory[index].queue)
    }

    fn claim_slot(&mut self, expert_id: u32, slot: usize) -> Result<()> {
        let index = expert_directory_index(expert_id)?;
        let owner = self
            .slot_experts
            .get_mut(slot)
            .ok_or_else(|| Error::backend(format!("Q2 expert slot {slot} is out of bounds")))?;
        if owner.is_some() || self.directory[index].slot.is_some() {
            return Err(Error::backend(format!(
                "Q2 expert {expert_id} or slot {slot} is already resident"
            )));
        }
        *owner = Some(expert_id);
        self.directory[index].slot = Some(slot);
        self.resident_count = self.resident_count.saturating_add(1);
        Ok(())
    }

    fn evict_oldest(&mut self) -> Result<Option<usize>> {
        let victim = self.probation.head.or(self.protected.head);
        let Some(victim) = victim else {
            return Ok(None);
        };
        self.unlink(victim)?;
        let index = expert_directory_index(victim)?;
        let slot = self.directory[index].slot.take().ok_or_else(|| {
            Error::backend(format!("evicted Q2 expert {victim} has no resident slot"))
        })?;
        let owner = self
            .slot_experts
            .get_mut(slot)
            .ok_or_else(|| Error::backend(format!("Q2 expert slot {slot} is out of bounds")))?;
        if *owner != Some(victim) {
            return Err(Error::backend(format!(
                "Q2 expert slot {slot} is not owned by eviction victim {victim}"
            )));
        }
        *owner = None;
        self.resident_count = self.resident_count.saturating_sub(1);
        Ok(Some(slot))
    }

    fn link_back(&mut self, expert_id: u32, queue_kind: ExpertQueueKind) -> Result<()> {
        if queue_kind == ExpertQueueKind::None {
            return Err(Error::backend("cannot link an expert into the empty queue"));
        }
        let index = expert_directory_index(expert_id)?;
        let entry = self.directory[index];
        if entry.slot.is_none() || entry.queue != ExpertQueueKind::None {
            return Err(Error::backend(format!(
                "Q2 expert {expert_id} cannot be linked from its current state"
            )));
        }
        let tail = self.queue(queue_kind)?.tail;
        if let Some(tail) = tail {
            self.directory[tail as usize].next = Some(expert_id);
        } else {
            self.queue_mut(queue_kind)?.head = Some(expert_id);
        }
        self.directory[index].queue = queue_kind;
        self.directory[index].previous = tail;
        self.directory[index].next = None;
        let queue = self.queue_mut(queue_kind)?;
        queue.tail = Some(expert_id);
        queue.len = queue.len.saturating_add(1);
        Ok(())
    }

    fn unlink(&mut self, expert_id: u32) -> Result<()> {
        let index = expert_directory_index(expert_id)?;
        let entry = self.directory[index];
        if entry.queue == ExpertQueueKind::None {
            return Err(Error::backend(format!(
                "Q2 expert {expert_id} is not linked in an SLRU queue"
            )));
        }
        if let Some(previous) = entry.previous {
            self.directory[previous as usize].next = entry.next;
        } else {
            self.queue_mut(entry.queue)?.head = entry.next;
        }
        if let Some(next) = entry.next {
            self.directory[next as usize].previous = entry.previous;
        } else {
            self.queue_mut(entry.queue)?.tail = entry.previous;
        }
        let queue = self.queue_mut(entry.queue)?;
        queue.len = queue
            .len
            .checked_sub(1)
            .ok_or_else(|| Error::backend("Q2 expert SLRU queue length underflow"))?;
        self.directory[index].queue = ExpertQueueKind::None;
        self.directory[index].previous = None;
        self.directory[index].next = None;
        Ok(())
    }

    fn enforce_protected_capacity(&mut self) -> Result<()> {
        while self.protected.len > self.protected_capacity {
            let demoted = self.protected.head.ok_or_else(|| {
                Error::backend("Q2 protected expert queue has a length but no head")
            })?;
            self.unlink(demoted)?;
            self.link_back(demoted, ExpertQueueKind::Probation)?;
        }
        Ok(())
    }

    fn queue(&self, queue_kind: ExpertQueueKind) -> Result<&ExpertQueue> {
        match queue_kind {
            ExpertQueueKind::Probation => Ok(&self.probation),
            ExpertQueueKind::Protected => Ok(&self.protected),
            ExpertQueueKind::None => Err(Error::backend("empty expert queue has no metadata")),
        }
    }

    fn queue_mut(&mut self, queue_kind: ExpertQueueKind) -> Result<&mut ExpertQueue> {
        match queue_kind {
            ExpertQueueKind::Probation => Ok(&mut self.probation),
            ExpertQueueKind::Protected => Ok(&mut self.protected),
            ExpertQueueKind::None => Err(Error::backend("empty expert queue has no metadata")),
        }
    }

    fn invalidate(&mut self, expert_ids: &[u32]) -> Result<()> {
        for &expert_id in expert_ids {
            let index = expert_directory_index(expert_id)?;
            let Some(slot) = self.directory[index].slot else {
                continue;
            };
            if self.directory[index].queue != ExpertQueueKind::None {
                self.unlink(expert_id)?;
            }
            self.directory[index].slot = None;
            let owner = self
                .slot_experts
                .get_mut(slot)
                .ok_or_else(|| Error::backend(format!("Q2 expert slot {slot} is out of bounds")))?;
            if *owner != Some(expert_id) {
                return Err(Error::backend(format!(
                    "Q2 expert slot {slot} is not owned by invalidated expert {expert_id}"
                )));
            }
            *owner = None;
            self.resident_count = self.resident_count.saturating_sub(1);
            self.free_slots.push(slot);
        }
        Ok(())
    }
}

fn expert_directory_index(expert_id: u32) -> Result<usize> {
    let index = expert_id as usize;
    if index >= ROUTED_EXPERT_COUNT {
        return Err(Error::backend(format!(
            "Q2 expert ID {expert_id} is outside the Inferno range 0..{ROUTED_EXPERT_COUNT}"
        )));
    }
    Ok(index)
}

fn expert_protected_capacity(capacity: usize) -> usize {
    capacity
        .saturating_mul(ROUTED_EXPERT_PROTECTED_PERCENT)
        .div_ceil(100)
        .min(capacity)
}

fn prepare_ready_expert_groups<'a>(
    device: &Device,
    cache: &mut Q2ExpertLayerCache,
    mut transient_pool: Option<&mut Q2TransientExpertPool>,
    gate_payloads: &[Q2ExpertSource<'a>],
    up_payloads: &[Q2ExpertSource<'a>],
    down_payloads: &[Q2ExpertSource<'a>],
    gate_stride: usize,
    up_stride: usize,
    down_stride: usize,
    layer_index: usize,
    expert_pack_header: Option<ExpertPackHeader>,
    predictive_prefetch: bool,
) -> Result<Vec<ReadyExpertGroup>> {
    if let Some(header) = expert_pack_header {
        validate_exact_len(
            "Q2 expert-pack gate stride",
            usize::try_from(header.gate_bytes)
                .map_err(|_| Error::backend("Q2 expert-pack gate stride exceeds usize"))?,
            gate_stride,
        )?;
        validate_exact_len(
            "Q2 expert-pack up stride",
            usize::try_from(header.up_bytes)
                .map_err(|_| Error::backend("Q2 expert-pack up stride exceeds usize"))?,
            up_stride,
        )?;
        validate_exact_len(
            "Q2 expert-pack down stride",
            usize::try_from(header.down_bytes)
                .map_err(|_| Error::backend("Q2 expert-pack down stride exceeds usize"))?,
            down_stride,
        )?;
    }
    let mut seed_indices = [None::<usize>; ROUTED_EXPERT_COUNT];
    let mut seeds = Vec::<ReadyExpertSeed<'a>>::new();
    for assignment_index in 0..gate_payloads.len() {
        let gate = gate_payloads[assignment_index];
        let up = up_payloads[assignment_index];
        let down = down_payloads[assignment_index];
        if gate.expert_id != up.expert_id || gate.expert_id != down.expert_id {
            return Err(Error::backend(format!(
                "Q2 routed expert assignment {assignment_index} mixes expert IDs {}, {}, and {}",
                gate.expert_id, up.expert_id, down.expert_id
            )));
        }
        let expert_index = expert_directory_index(gate.expert_id)?;
        if let Some(seed_index) = seed_indices[expert_index] {
            let seed = &mut seeds[seed_index];
            if !same_expert_source(seed.gate, gate)
                || !same_expert_source(seed.up, up)
                || !same_expert_source(seed.down, down)
            {
                return Err(Error::backend(format!(
                    "duplicate Q2 expert ID {} uses inconsistent payloads",
                    gate.expert_id
                )));
            }
            seed.assignment_indices.push(assignment_index);
            continue;
        }
        seed_indices[expert_index] = Some(seeds.len());
        seeds.push(ReadyExpertSeed {
            expert_id: gate.expert_id,
            assignment_indices: vec![assignment_index],
            gate,
            up,
            down,
        });
    }

    prioritize_ready_expert_seeds(&mut seeds);
    let selected_expert_ids = seeds.iter().map(|seed| seed.expert_id).collect::<Vec<_>>();
    let resolutions = cache.resolve_slots(&selected_expert_ids, predictive_prefetch)?;
    let transient_capacity = seeds.len();
    let mut transient_slot = 0_usize;
    let mut groups = Vec::with_capacity(seeds.len());
    for (seed, resolution) in seeds.into_iter().zip(resolutions) {
        if predictive_prefetch && matches!(resolution, LayerSlotResolution::Transient) {
            continue;
        }
        let (buffers, cache_hit, cached_miss_expert_id, transient, transient_owner) =
            match resolution {
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
                    Some(seed.expert_id),
                    false,
                    None,
                ),
                LayerSlotResolution::Transient => match transient_pool.as_deref_mut() {
                    Some(pool) => {
                        if transient_slot == 0 {
                            pool.ensure(
                                device,
                                transient_capacity,
                                gate_stride,
                                up_stride,
                                down_stride,
                            )?;
                        }
                        let buffers = pool.ready_buffers(
                            transient_slot,
                            gate_stride,
                            up_stride,
                            down_stride,
                        )?;
                        transient_slot += 1;
                        (buffers, false, None, true, None)
                    }
                    None => {
                        let slot =
                            Q2ExpertSlotBuffers::new(device, gate_stride, up_stride, down_stride)?;
                        (slot.ready_buffers(), false, None, true, Some(slot))
                    }
                },
            };
        let (gate_absolute_offset, up_absolute_offset, down_absolute_offset) =
            expert_read_offsets(expert_pack_header, layer_index, &seed)?;
        let read_tasks = if cache_hit {
            Vec::new()
        } else {
            vec![
                ExpertReadTask {
                    buffer: buffers.gate.clone(),
                    destination_offset: buffers.gate_offset,
                    absolute_offset: gate_absolute_offset,
                    byte_len: seed.gate.bytes.len(),
                },
                ExpertReadTask {
                    buffer: buffers.up.clone(),
                    destination_offset: buffers.up_offset,
                    absolute_offset: up_absolute_offset,
                    byte_len: seed.up.bytes.len(),
                },
                ExpertReadTask {
                    buffer: buffers.down.clone(),
                    destination_offset: buffers.down_offset,
                    absolute_offset: down_absolute_offset,
                    byte_len: seed.down.bytes.len(),
                },
            ]
        };
        groups.push(ReadyExpertGroup {
            assignment_indices: seed.assignment_indices,
            buffers,
            read_tasks,
            cache_hit,
            cached_miss_expert_id,
            transient,
            _transient_owner: transient_owner,
        });
    }
    Ok(groups)
}

fn same_expert_source(left: Q2ExpertSource<'_>, right: Q2ExpertSource<'_>) -> bool {
    left.expert_id == right.expert_id
        && left.bytes.as_ptr() == right.bytes.as_ptr()
        && left.bytes.len() == right.bytes.len()
        && left.absolute_offset == right.absolute_offset
}

fn expert_read_offsets(
    header: Option<ExpertPackHeader>,
    layer_index: usize,
    seed: &ReadyExpertSeed<'_>,
) -> Result<(u64, u64, u64)> {
    let Some(header) = header else {
        return Ok((
            seed.gate.absolute_offset,
            seed.up.absolute_offset,
            seed.down.absolute_offset,
        ));
    };
    let layer_index = u32::try_from(layer_index)
        .map_err(|_| Error::backend("Q2 expert-pack layer index exceeds u32"))?;
    Ok((
        header.component_offset(layer_index, seed.expert_id, ExpertComponent::Gate)?,
        header.component_offset(layer_index, seed.expert_id, ExpertComponent::Up)?,
        header.component_offset(layer_index, seed.expert_id, ExpertComponent::Down)?,
    ))
}

fn prioritize_ready_expert_seeds(seeds: &mut [ReadyExpertSeed<'_>]) {
    // MTP verification is token-major. Keep experts used by the newest row in
    // persistent slots because that row is closest to the next decode step.
    // Assignment count breaks ties in favor of weights reused by more rows.
    seeds.sort_by(|left, right| {
        let left_recency = left.assignment_indices.last().copied().unwrap_or(0);
        let right_recency = right.assignment_indices.last().copied().unwrap_or(0);
        right_recency.cmp(&left_recency).then_with(|| {
            right
                .assignment_indices
                .len()
                .cmp(&left.assignment_indices.len())
        })
    });
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

fn pread_ready_expert_groups(file: &File, path: &Path, groups: &[ReadyExpertGroup]) -> Result<()> {
    let misses = groups
        .iter()
        .filter(|group| !group.cache_hit)
        .collect::<Vec<_>>();
    if misses.is_empty() {
        return Ok(());
    }
    let worker_count = misses.len().min(ROUTED_EXPERT_READ_WORKERS);
    let jobs_per_worker = misses.len().div_ceil(worker_count);
    thread::scope(|scope| -> Result<()> {
        let mut workers = Vec::with_capacity(worker_count);
        for jobs in misses.chunks(jobs_per_worker) {
            workers.push(scope.spawn(move || -> Result<()> {
                for group in jobs {
                    pread_ready_expert_gate_up(file, path, group)?;
                    pread_ready_expert_down(file, path, group)?;
                }
                Ok(())
            }));
        }
        for worker in workers {
            worker
                .join()
                .map_err(|_| Error::backend("predictive expert read worker thread panicked"))??;
        }
        Ok(())
    })
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
    // The cache slot is not used by Metal until this read completes, and the
    // destination range was validated above.
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
    use inferno_io::ExpertPackHeader;

    use super::{
        super::validation::{
            Q2_K_BLOCK_BYTES, Q2_K_BLOCK_VALUES, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_VALUES,
        },
        expert_protected_capacity, prioritize_ready_expert_seeds, ExpertQueueKind,
        LayerSlotResolution, Q2ExpertLayerCache, Q2PerLayerExpertCache, Q2TransientExpertPool,
        QuantMatvecKind, ReadyExpertSeed, ROUTED_EXPERT_COUNT, ROUTED_EXPERT_WAVE_MIN_GROUPS,
    };

    fn resolve_one(cache: &mut Q2ExpertLayerCache, expert_id: u32) -> LayerSlotResolution {
        cache.resolve_slots(&[expert_id], false).unwrap()[0]
    }

    fn assert_cache_consistent(cache: &Q2ExpertLayerCache) {
        let mut queued = [false; ROUTED_EXPERT_COUNT];
        let mut queued_count = 0_usize;
        for (kind, queue) in [
            (ExpertQueueKind::Probation, cache.probation),
            (ExpertQueueKind::Protected, cache.protected),
        ] {
            let mut previous = None;
            let mut current = queue.head;
            let mut queue_count = 0_usize;
            while let Some(expert_id) = current {
                let index = expert_id as usize;
                assert!(index < ROUTED_EXPERT_COUNT);
                assert!(!std::mem::replace(&mut queued[index], true));
                let entry = cache.directory[index];
                assert_eq!(entry.queue, kind);
                assert_eq!(entry.previous, previous);
                assert!(entry.slot.is_some());
                previous = current;
                current = entry.next;
                queue_count += 1;
                assert!(queue_count <= cache.resident_count);
            }
            assert_eq!(previous, queue.tail);
            assert_eq!(queue_count, queue.len);
            queued_count += queue_count;
        }

        let mut resident_count = 0_usize;
        for (expert_id, entry) in cache.directory.iter().copied().enumerate() {
            match entry.slot {
                Some(slot) => {
                    resident_count += 1;
                    assert!(queued[expert_id]);
                    assert_eq!(cache.slot_experts[slot], Some(expert_id as u32));
                }
                None => {
                    assert!(!queued[expert_id]);
                    assert_eq!(entry.queue, ExpertQueueKind::None);
                    assert_eq!(entry.previous, None);
                    assert_eq!(entry.next, None);
                }
            }
        }
        assert_eq!(resident_count, cache.resident_count);
        assert_eq!(queued_count, cache.resident_count);

        let mut free = vec![false; cache.slot_experts.len()];
        for &slot in &cache.free_slots {
            assert!(!std::mem::replace(&mut free[slot], true));
            assert_eq!(cache.slot_experts[slot], None);
        }
        assert_eq!(
            cache.resident_count + cache.free_slots.len(),
            cache.slot_experts.len()
        );
    }

    #[test]
    fn expert_lru_is_partitioned_by_layer() {
        let mut cache = Q2PerLayerExpertCache::default();
        let first_layer = cache.layer(3);
        let second_layer = cache.layer(4);

        assert!(matches!(
            resolve_one(&mut first_layer.lock().unwrap(), 1),
            LayerSlotResolution::Miss(_)
        ));
        assert!(matches!(
            resolve_one(&mut second_layer.lock().unwrap(), 1),
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

        assert_eq!(layer.lock().unwrap().initialized_slots.len(), 3);
    }

    #[test]
    fn expert_slru_reserves_one_quarter_for_protected_entries() {
        assert_eq!(expert_protected_capacity(30), 8);
        assert_eq!(expert_protected_capacity(4), 1);
        assert_eq!(expert_protected_capacity(1), 1);
    }

    #[test]
    fn expert_lru_evicts_only_unselected_entries() {
        let mut cache = Q2ExpertLayerCache::new(2);
        resolve_one(&mut cache, 1);
        resolve_one(&mut cache, 2);

        let resolutions = cache.resolve_slots(&[1, 3], false).unwrap();

        assert!(matches!(resolutions[0], LayerSlotResolution::Hit(_)));
        assert!(matches!(resolutions[1], LayerSlotResolution::Miss(_)));
        assert!(cache.contains(1));
        assert!(!cache.contains(2));
        assert!(cache.contains(3));
    }

    #[test]
    fn expert_lru_uses_transient_slot_when_every_entry_is_selected() {
        let mut cache = Q2ExpertLayerCache::new(2);
        resolve_one(&mut cache, 1);
        resolve_one(&mut cache, 2);

        let resolutions = cache.resolve_slots(&[1, 2, 3], false).unwrap();

        assert!(matches!(resolutions[0], LayerSlotResolution::Hit(_)));
        assert!(matches!(resolutions[1], LayerSlotResolution::Hit(_)));
        assert!(matches!(resolutions[2], LayerSlotResolution::Transient));
        assert_eq!(cache.resident_count, 2);
    }

    #[test]
    fn transient_expert_pool_reuses_its_backing_slab() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let mut pool = Q2TransientExpertPool::default();
        pool.ensure(metal.device(), 4, 64, 64, 32).unwrap();
        let first = pool.ready_buffers(1, 64, 64, 32).unwrap();
        pool.ensure(metal.device(), 2, 64, 64, 32).unwrap();
        let second = pool.ready_buffers(1, 64, 64, 32).unwrap();

        assert_eq!(first.gate.gpu_address(), second.gate.gpu_address());
        assert_eq!(first.up.gpu_address(), second.up.gpu_address());
        assert_eq!(first.down.gpu_address(), second.down.gpu_address());
        assert_eq!(first.gate_offset, second.gate_offset);
        assert_eq!(first.up_offset, second.up_offset);
        assert_eq!(first.down_offset, second.down_offset);
        assert_eq!(first.slot_index, None);
        assert_eq!(second.slot_index, None);
    }

    #[test]
    fn mtp_cache_admission_prioritizes_the_latest_token_row() {
        let weights = [0_u8; 3];
        let source = |index: usize| Q2ExpertSource {
            expert_id: index as u32,
            bytes: &weights[index..index + 1],
            absolute_offset: index as u64,
        };
        let mut seeds = vec![
            ReadyExpertSeed {
                expert_id: 0,
                assignment_indices: vec![0, 8],
                gate: source(0),
                up: source(0),
                down: source(0),
            },
            ReadyExpertSeed {
                expert_id: 1,
                assignment_indices: vec![1, 15],
                gate: source(1),
                up: source(1),
                down: source(1),
            },
            ReadyExpertSeed {
                expert_id: 2,
                assignment_indices: vec![2, 9],
                gate: source(2),
                up: source(2),
                down: source(2),
            },
        ];

        prioritize_ready_expert_seeds(&mut seeds);

        assert_eq!(
            seeds.iter().map(|seed| seed.expert_id).collect::<Vec<_>>(),
            vec![1, 2, 0]
        );
    }

    #[test]
    fn segmented_lru_protects_reused_experts_from_one_time_scans() {
        let mut cache = Q2ExpertLayerCache::new(3);
        resolve_one(&mut cache, 1);
        resolve_one(&mut cache, 2);
        resolve_one(&mut cache, 3);
        resolve_one(&mut cache, 1);

        resolve_one(&mut cache, 4);

        assert!(cache.contains(1));
        assert_eq!(cache.queue_kind(1), Some(ExpertQueueKind::Protected));
        assert!(!cache.contains(2));
        assert!(cache.contains(4));
    }

    #[test]
    fn direct_directory_rejects_out_of_range_expert_ids_before_mutation() {
        let mut cache = Q2ExpertLayerCache::new(2);

        let error = cache
            .resolve_slots(&[ROUTED_EXPERT_COUNT as u32], false)
            .unwrap_err();

        assert!(error.to_string().contains("outside the Inferno range"));
        assert_eq!(cache.resident_count, 0);
        assert_eq!(cache.free_slots.len(), 2);
    }

    #[test]
    fn predictive_hit_stays_in_its_existing_slru_segment() {
        let mut cache = Q2ExpertLayerCache::new(2);
        assert!(matches!(
            cache.resolve_slots(&[1], true).unwrap()[0],
            LayerSlotResolution::Miss(_)
        ));
        assert_eq!(cache.queue_kind(1), Some(ExpertQueueKind::Probation));

        assert!(matches!(
            cache.resolve_slots(&[1], true).unwrap()[0],
            LayerSlotResolution::Hit(_)
        ));
        assert_eq!(cache.queue_kind(1), Some(ExpertQueueKind::Probation));
    }

    #[test]
    fn direct_directory_remains_consistent_under_repeated_eviction() {
        let mut cache = Q2ExpertLayerCache::new(30);
        for step in 0..4_096_u32 {
            let expert_ids = (0..8_u32)
                .map(|rank| (step.wrapping_mul(17) + rank * 31) % ROUTED_EXPERT_COUNT as u32)
                .collect::<Vec<_>>();
            cache.resolve_slots(&expert_ids, false).unwrap();
            assert_cache_consistent(&cache);
        }

        assert_eq!(cache.resident_count, 30);
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
    fn q8_0_batched_matvec_reuses_weights_across_prefill_rows() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let row_count = 17;
        let in_features = 64;
        let out_features = 3;
        let weights = [
            q8_0_block(0x3c00, 1),
            q8_0_block(0x4000, -2),
            q8_0_block(0x3800, 3),
            q8_0_block(0x3c00, -1),
            q8_0_block(0x4000, 2),
            q8_0_block(0x3800, -3),
        ]
        .concat();
        let input = (0..row_count * in_features)
            .map(|index| {
                let row = index / in_features;
                let column = index % in_features;
                (row as f32 - 2.0) * 0.03125 + (column as f32 - 17.0) * 0.00390625
            })
            .collect::<Vec<_>>();
        let input_buffer = metal.batch_upload_f32(&input).unwrap();

        let output = metal
            .batched_quant_matvec(
                QuantMatvecKind::Q80,
                &weights,
                &input_buffer,
                input.len(),
                row_count,
                in_features,
                out_features,
            )
            .unwrap();
        let actual = metal
            .batch_read_f32(&output, row_count * out_features)
            .unwrap();
        let expected = cpu_q8_0_matvec(&weights, &input, row_count, in_features, out_features);

        assert_close(&actual, &expected, 1e-4);
    }

    #[test]
    fn q8_0_batched_matvec_add_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let row_count = 19;
        let in_features = 64;
        let out_features = 2;
        let weights = [
            q8_0_block(0x3c00, 2),
            q8_0_block(0x3800, -3),
            q8_0_block(0x4000, -1),
            q8_0_block(0x3c00, 4),
        ]
        .concat();
        let input = (0..row_count * in_features)
            .map(|index| (index as f32 % 37.0) * 0.0078125 - 0.125)
            .collect::<Vec<_>>();
        let residual = (0..row_count * out_features)
            .map(|index| index as f32 * 0.015625 - 0.25)
            .collect::<Vec<_>>();
        let input_buffer = metal.batch_upload_f32(&input).unwrap();
        let residual_buffer = metal.batch_upload_f32(&residual).unwrap();

        let output = metal
            .batched_q8_0_matvec_add(
                &weights,
                &input_buffer,
                input.len(),
                &residual_buffer,
                residual.len(),
                row_count,
                in_features,
                out_features,
            )
            .unwrap();
        let actual = metal
            .batch_read_f32(&output, row_count * out_features)
            .unwrap();
        let mut expected = cpu_q8_0_matvec(&weights, &input, row_count, in_features, out_features);
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
            for row in 0..intermediate_features {
                let seed = (expert * intermediate_features + row) as u8;
                gate_weights.extend(q2_k_patterned_block(
                    if expert == 0 { 0x3c00 } else { 0x4000 },
                    seed,
                ));
                up_weights.extend(q2_k_patterned_block(
                    if expert == 0 { 0x4000 } else { 0x3c00 },
                    seed.wrapping_mul(3).wrapping_add(1),
                ));
            }
            for row in 0..hidden_features {
                down_weights.extend(q2_k_patterned_block(
                    0x3c00,
                    (expert * hidden_features + row) as u8,
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
        let pack_header = ExpertPackHeader::new(
            (gate_weights.len() + up_weights.len() + down_weights.len()) as u64,
            1,
            1,
            2,
            expert_count as u32,
            gate_stride as u64,
            gate_stride as u64,
            down_stride as u64,
        )
        .unwrap();
        let mut pack_bytes = pack_header.encode().unwrap().to_vec();
        for _layer in 0..pack_header.layer_count {
            for expert in 0..expert_count {
                let gate_start = expert * gate_stride;
                let down_start = expert * down_stride;
                pack_bytes.extend_from_slice(&gate_weights[gate_start..gate_start + gate_stride]);
                pack_bytes.extend_from_slice(&up_weights[gate_start..gate_start + gate_stride]);
                pack_bytes.extend_from_slice(&down_weights[down_start..down_start + down_stride]);
            }
        }
        let expert_pack = TemporaryModelFile::new("staged-router-expert-pack", pack_bytes);
        metal
            .configure_expert_pack(expert_pack.path(), pack_header)
            .unwrap();
        let gate_sources = selected_ids
            .iter()
            .zip(&gate_payloads)
            .map(|(&expert, &bytes)| Q2ExpertSource {
                expert_id: expert,
                bytes,
                absolute_offset: expert as u64 * gate_stride as u64,
            })
            .collect::<Vec<_>>();
        let up_sources = selected_ids
            .iter()
            .zip(&up_payloads)
            .map(|(&expert, &bytes)| Q2ExpertSource {
                expert_id: expert,
                bytes,
                absolute_offset: up_base + expert as u64 * gate_stride as u64,
            })
            .collect::<Vec<_>>();
        let down_sources = selected_ids
            .iter()
            .zip(&down_payloads)
            .map(|(&expert, &bytes)| Q2ExpertSource {
                expert_id: expert,
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
        assert_eq!(first.ready_waves, 2);
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
        assert_eq!(ready.ready_waves, 2);
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
        assert_eq!(other_layer.ready_waves, 2);
        metal
            .batched_wait_for_ready_routed_experts(ready.completion_value)
            .unwrap();
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
    fn ready_router_experts_preserve_eight_token_rows_beyond_cache_capacity() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let token_count = 8;
        let top_k = 8;
        let expert_count = super::ROUTED_EXPERT_CACHE_SLOTS_PER_LAYER + top_k;
        let hidden_features = Q2_K_BLOCK_VALUES;
        let intermediate_features = Q2_K_BLOCK_VALUES;
        let expert_stride = hidden_features * Q2_K_BLOCK_BYTES;

        let mut gate_weights = Vec::new();
        let mut up_weights = Vec::new();
        let mut down_weights = Vec::new();
        for expert in 0..expert_count {
            for row in 0..hidden_features {
                let selector = (expert + row) % 4;
                gate_weights.extend(q2_k_block(
                    0x3c00 + (expert % 2) as u16 * 0x0400,
                    0,
                    1 + (expert % 7) as u8,
                    [0x00, 0x1b, 0xe4, 0xff][selector],
                ));
                up_weights.extend(q2_k_block(
                    0x3c00,
                    0,
                    1 + (row % 7) as u8,
                    [0xff, 0xe4, 0x1b, 0x00][selector],
                ));
                down_weights.extend(q2_k_block(
                    0x3c00,
                    0,
                    1 + ((expert + row) % 7) as u8,
                    [0x1b, 0xe4, 0xff, 0x00][selector],
                ));
            }
        }

        let input = (0..token_count * hidden_features)
            .map(|index| {
                let token = index / hidden_features;
                let hidden = index % hidden_features;
                (token as f32 + 1.0) * 0.0001 + (hidden as f32 % 17.0) * 0.00001
            })
            .collect::<Vec<_>>();
        let mut router_logits = vec![-20.0_f32; token_count * expert_count];
        for token in 0..token_count {
            let expert_base = (token * top_k) % expert_count;
            for rank in 0..top_k {
                let expert = (expert_base + rank) % expert_count;
                router_logits[token * expert_count + expert] = 10.0 - rank as f32;
            }
        }
        let correction_bias = vec![0.0_f32; expert_count];
        let zeros = vec![0.0_f32; token_count * hidden_features];
        let input_buffer = metal.batch_upload_f32(&input).unwrap();
        let logits_buffer = metal.batch_upload_f32(&router_logits).unwrap();
        let zero_buffer = metal.batch_upload_f32(&zeros).unwrap();
        let routing = metal
            .batched_moe_router_topk_resident(
                &logits_buffer,
                router_logits.len(),
                &correction_bias,
                token_count,
                expert_count,
                top_k,
                true,
                1.0,
            )
            .unwrap();
        let selected_ids = metal.batched_moe_router_expert_ids(&routing).unwrap();
        assert_eq!(selected_ids.len(), token_count * top_k);
        assert!(
            selected_ids.iter().copied().collect::<HashSet<_>>().len()
                > super::ROUTED_EXPERT_CACHE_SLOTS_PER_LAYER
        );

        let up_base = gate_weights.len() as u64;
        let down_base = up_base + up_weights.len() as u64;
        let model_file = TemporaryModelFile::new(
            "eight-token-ready-experts",
            [&gate_weights[..], &up_weights[..], &down_weights[..]].concat(),
        );
        let gate_sources = selected_ids
            .iter()
            .map(|&expert| {
                let start = expert as usize * expert_stride;
                Q2ExpertSource {
                    expert_id: expert,
                    bytes: &gate_weights[start..start + expert_stride],
                    absolute_offset: start as u64,
                }
            })
            .collect::<Vec<_>>();
        let up_sources = selected_ids
            .iter()
            .map(|&expert| {
                let start = expert as usize * expert_stride;
                Q2ExpertSource {
                    expert_id: expert,
                    bytes: &up_weights[start..start + expert_stride],
                    absolute_offset: up_base + start as u64,
                }
            })
            .collect::<Vec<_>>();
        let down_sources = selected_ids
            .iter()
            .map(|&expert| {
                let start = expert as usize * expert_stride;
                Q2ExpertSource {
                    expert_id: expert,
                    bytes: &down_weights[start..start + expert_stride],
                    absolute_offset: down_base + start as u64,
                }
            })
            .collect::<Vec<_>>();

        let ready = metal
            .ready_routed_experts(
                79,
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
        assert!(ready.transient_experts > 0);
        assert!(ready.ready_waves >= 2);
        assert!(
            ready.ready_waves
                <= 2 * ready
                    .selected_experts
                    .div_ceil(ROUTED_EXPERT_WAVE_MIN_GROUPS)
        );
        metal
            .batched_wait_for_ready_routed_experts(ready.completion_value)
            .unwrap();
        let combined = metal
            .batched_moe_topk_combine_residual(
                &zero_buffer,
                zeros.len(),
                &zero_buffer,
                zeros.len(),
                &ready.output,
                selected_ids.len() * hidden_features,
                &routing,
                hidden_features,
            )
            .unwrap();
        let actual = metal.batch_read_f32(&combined, zeros.len()).unwrap();

        let mut expected = vec![0.0_f32; zeros.len()];
        for token in 0..token_count {
            let selected = &selected_ids[token * top_k..(token + 1) * top_k];
            let score_sum = selected
                .iter()
                .map(|&expert| {
                    let logit = router_logits[token * expert_count + expert as usize];
                    1.0 / (1.0 + (-logit).exp())
                })
                .sum::<f32>();
            let token_input = &input[token * hidden_features..(token + 1) * hidden_features];
            for &expert in selected {
                let expert = expert as usize;
                let start = expert * expert_stride;
                let end = start + expert_stride;
                let gate = cpu_q2_k_matvec(
                    &gate_weights[start..end],
                    token_input,
                    1,
                    hidden_features,
                    intermediate_features,
                );
                let up = cpu_q2_k_matvec(
                    &up_weights[start..end],
                    token_input,
                    1,
                    hidden_features,
                    intermediate_features,
                );
                let gated = cpu_swiglu(&gate, &up);
                let down = cpu_q2_k_matvec(
                    &down_weights[start..end],
                    &gated,
                    1,
                    intermediate_features,
                    hidden_features,
                );
                let logit = router_logits[token * expert_count + expert];
                let weight = (1.0 / (1.0 + (-logit).exp())) / score_sum;
                for hidden in 0..hidden_features {
                    expected[token * hidden_features + hidden] += down[hidden] * weight;
                }
            }
        }

        assert_close_relative(&actual, &expected, 2e-5);
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
    fn absorbed_mla_sequence_matches_causal_q8_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let batch = 1;
        let query_tokens = 3;
        let heads = 2;
        let q_no_dim = 32;
        let rope_dim = 8;
        let latent_dim = 32;
        let value_dim = 32;
        let past_tokens = 2;
        let page_size = 4;

        let q_no = (0..query_tokens * heads * q_no_dim)
            .map(|index| (index as f32 - 17.0) * 0.01)
            .collect::<Vec<_>>();
        let q_rope = (0..query_tokens * heads * rope_dim)
            .map(|index| (index as f32 + 1.0) * -0.015)
            .collect::<Vec<_>>();
        let current_latent = (0..query_tokens * latent_dim)
            .map(|index| (index as f32 - 9.0) * 0.02)
            .collect::<Vec<_>>();
        let current_rope = (0..query_tokens * rope_dim)
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
            .batched_q8_0_absorbed_mla(
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
                query_tokens,
                heads,
                q_no_dim,
                rope_dim,
                latent_dim,
                value_dim,
                q_no_dim + rope_dim,
            )
            .unwrap();
        let actual = metal.batch_read_f32(&output, output_len).unwrap();

        let mut expected = Vec::with_capacity(query_tokens * heads * value_dim);
        let k_head_bytes = latent_dim * Q8_0_BLOCK_BYTES;
        let v_head_bytes = value_dim * Q8_0_BLOCK_BYTES;
        for query in 0..query_tokens {
            for head in 0..heads {
                let query_head = query * heads + head;
                let q_no_start = query_head * q_no_dim;
                let q_rope_start = query_head * rope_dim;
                let q_latent = cpu_q8_0_matvec(
                    &k_b[head * k_head_bytes..(head + 1) * k_head_bytes],
                    &q_no[q_no_start..q_no_start + q_no_dim],
                    1,
                    q_no_dim,
                    latent_dim,
                );
                let mut scores = Vec::with_capacity(past_tokens + query + 1);
                for token in 0..past_tokens {
                    let latent_start = token * latent_dim;
                    let rope_start = token * rope_dim;
                    let latent_score = q_latent
                        .iter()
                        .zip(&paged_latent[latent_start..latent_start + latent_dim])
                        .map(|(left, right)| left * right)
                        .sum::<f32>();
                    let rope_score = q_rope[q_rope_start..q_rope_start + rope_dim]
                        .iter()
                        .zip(&paged_rope[rope_start..rope_start + rope_dim])
                        .map(|(left, right)| left * right)
                        .sum::<f32>();
                    scores
                        .push((latent_score + rope_score) / ((q_no_dim + rope_dim) as f32).sqrt());
                }
                for current_token in 0..=query {
                    let latent_start = current_token * latent_dim;
                    let rope_start = current_token * rope_dim;
                    let latent_score = q_latent
                        .iter()
                        .zip(&current_latent[latent_start..latent_start + latent_dim])
                        .map(|(left, right)| left * right)
                        .sum::<f32>();
                    let rope_score = q_rope[q_rope_start..q_rope_start + rope_dim]
                        .iter()
                        .zip(&current_rope[rope_start..rope_start + rope_dim])
                        .map(|(left, right)| left * right)
                        .sum::<f32>();
                    scores
                        .push((latent_score + rope_score) / ((q_no_dim + rope_dim) as f32).sqrt());
                }
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
                    for current_token in 0..=query {
                        context_latent[dim] += probabilities[past_tokens + current_token]
                            * current_latent[current_token * latent_dim + dim];
                    }
                }
                expected.extend(cpu_q8_0_matvec(
                    &v_b[head * v_head_bytes..(head + 1) * v_head_bytes],
                    &context_latent,
                    1,
                    latent_dim,
                    value_dim,
                ));
            }
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

    fn q2_k_patterned_block(d: u16, seed: u8) -> Vec<u8> {
        let mut block = Vec::with_capacity(Q2_K_BLOCK_BYTES);
        for index in 0..16_u8 {
            block.push(seed.wrapping_add(index) & 0x0f);
        }
        for index in 0..64_u8 {
            block.push(seed.wrapping_mul(29).wrapping_add(index.wrapping_mul(17)));
        }
        block.extend(d.to_le_bytes());
        block.extend(0_u16.to_le_bytes());
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
        crate::metal::buffers::f16_bits_to_f32(bits)
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
