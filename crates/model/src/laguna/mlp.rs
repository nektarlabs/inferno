use std::{
    collections::HashMap,
    hash::Hash,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
};

use backend::{Backend, DeviceRouterTopK, DeviceValue, W4ExpertGroup};
use common::{DType, Error, Result};
use config::LagunaConfig;
use rayon::{Scope, ThreadPool, ThreadPoolBuilder};

use super::{
    LagunaDeviceDenseWeights, LagunaDeviceExpertWeights, LagunaDeviceMoeWeights,
    LagunaExpertWeights, LagunaWeightIndex,
};

// Decode selects exactly ten experts. Ten workers let every cold expert begin
// its independent shard read without waiting for another selected expert.
const MAX_EXPERT_IO_WORKERS: usize = 10;

pub struct LagunaExpertPrefetchPool {
    workers: ThreadPool,
}

impl std::fmt::Debug for LagunaExpertPrefetchPool {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LagunaExpertPrefetchPool")
            .field("workers", &self.workers.current_num_threads())
            .finish()
    }
}

impl LagunaExpertPrefetchPool {
    pub fn new() -> Result<Self> {
        let workers = ThreadPoolBuilder::new()
            .num_threads(MAX_EXPERT_IO_WORKERS)
            .thread_name(|index| format!("laguna-expert-io-{index}"))
            .build()
            .map_err(|error| {
                Error::runtime(format!(
                    "failed to create Laguna expert I/O worker pool: {error}"
                ))
            })?;
        Ok(Self { workers })
    }

    fn in_place_scope<'scope, Operation, Output>(&self, operation: Operation) -> Result<Output>
    where
        Operation: FnOnce(&Scope<'scope>) -> Result<Output>,
    {
        catch_unwind(AssertUnwindSafe(|| self.workers.in_place_scope(operation)))
            .map_err(|_| Error::weights("Laguna expert I/O worker panicked"))?
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LagunaExpertCacheMetrics {
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub resident_loads: u64,
    pub ready_waves: u64,
    pub ssd_read_bytes: u64,
    pub resident_load_bytes: u64,
    pub resident_bytes: u64,
    pub capacity_bytes: u64,
    pub resident_experts: usize,
    pub capacity_experts: usize,
}

impl LagunaExpertCacheMetrics {
    pub fn hit_rate(self) -> f64 {
        if self.lookups == 0 {
            0.0
        } else {
            self.hits as f64 / self.lookups as f64
        }
    }

    pub fn source_loads(self) -> u64 {
        self.resident_loads
    }

    /// Verifies that every successful miss performed exactly one source read
    /// and that resident hits performed none.
    pub fn validate(self) -> Result<()> {
        if self.hits.saturating_add(self.misses) != self.lookups {
            return Err(Error::cache(format!(
                "Laguna expert accounting mismatch: hits {} + misses {} != lookups {}",
                self.hits, self.misses, self.lookups
            )));
        }
        if self.source_loads() != self.misses {
            return Err(Error::cache(format!(
                "Laguna expert source-read mismatch: resident loads {} != misses {}",
                self.resident_loads, self.misses
            )));
        }
        if self.resident_load_bytes != self.ssd_read_bytes {
            return Err(Error::cache(format!(
                "Laguna expert byte accounting mismatch: resident {} != SSD {}",
                self.resident_load_bytes, self.ssd_read_bytes
            )));
        }
        if self.resident_experts > self.capacity_experts
            || self.resident_bytes > self.capacity_bytes
        {
            return Err(Error::cache(format!(
                "Laguna resident expert cache exceeds capacity: experts {}/{}, bytes {}/{}",
                self.resident_experts,
                self.capacity_experts,
                self.resident_bytes,
                self.capacity_bytes
            )));
        }
        Ok(())
    }
}

#[derive(Debug)]
struct LruNode<Key> {
    key: Key,
    previous: Option<usize>,
    next: Option<usize>,
}

#[derive(Debug)]
struct LruDirectory<Key> {
    capacity: usize,
    positions: HashMap<Key, usize>,
    nodes: Vec<Option<LruNode<Key>>>,
    least_recent: Option<usize>,
    most_recent: Option<usize>,
}

impl<Key> LruDirectory<Key>
where
    Key: Copy + Eq + Hash,
{
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            positions: HashMap::with_capacity(capacity),
            nodes: Vec::with_capacity(capacity),
            least_recent: None,
            most_recent: None,
        }
    }

    fn len(&self) -> usize {
        self.positions.len()
    }

    fn get(&mut self, key: Key) -> Result<Option<usize>> {
        let Some(slot) = self.positions.get(&key).copied() else {
            return Ok(None);
        };
        self.promote(slot)?;
        Ok(Some(slot))
    }

    /// Returns the storage slot and whether its previous key was evicted.
    fn insert(&mut self, key: Key) -> Result<(usize, bool)> {
        if let Some(slot) = self.positions.get(&key).copied() {
            self.promote(slot)?;
            return Ok((slot, false));
        }

        let (slot, evicted) = if self.positions.len() == self.capacity {
            let slot = self
                .least_recent
                .ok_or_else(|| Error::cache("full Laguna LRU has no least-recent slot"))?;
            let evicted_key = self
                .nodes
                .get(slot)
                .and_then(Option::as_ref)
                .map(|node| node.key)
                .ok_or_else(|| Error::cache("Laguna LRU eviction slot is empty"))?;
            self.detach(slot)?;
            self.nodes[slot] = None;
            self.positions.remove(&evicted_key);
            (slot, true)
        } else {
            let slot = self.nodes.len();
            self.nodes.push(None);
            (slot, false)
        };

        self.nodes[slot] = Some(LruNode {
            key,
            previous: self.most_recent,
            next: None,
        });
        if let Some(previous) = self.most_recent {
            self.node_mut(previous)?.next = Some(slot);
        } else {
            self.least_recent = Some(slot);
        }
        self.most_recent = Some(slot);
        self.positions.insert(key, slot);
        Ok((slot, evicted))
    }

    fn clear(&mut self) {
        self.positions.clear();
        self.nodes.clear();
        self.least_recent = None;
        self.most_recent = None;
    }

    fn promote(&mut self, slot: usize) -> Result<()> {
        if self.most_recent == Some(slot) {
            return Ok(());
        }
        self.detach(slot)?;
        let previous = self.most_recent;
        let node = self.node_mut(slot)?;
        node.previous = previous;
        node.next = None;
        if let Some(previous) = previous {
            self.node_mut(previous)?.next = Some(slot);
        } else {
            self.least_recent = Some(slot);
        }
        self.most_recent = Some(slot);
        Ok(())
    }

    fn detach(&mut self, slot: usize) -> Result<()> {
        let node = self
            .nodes
            .get(slot)
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::cache("Laguna LRU slot is empty"))?;
        let previous = node.previous;
        let next = node.next;
        if let Some(previous) = previous {
            self.node_mut(previous)?.next = next;
        } else {
            self.least_recent = next;
        }
        if let Some(next) = next {
            self.node_mut(next)?.previous = previous;
        } else {
            self.most_recent = previous;
        }
        Ok(())
    }

    fn node_mut(&mut self, slot: usize) -> Result<&mut LruNode<Key>> {
        self.nodes
            .get_mut(slot)
            .and_then(Option::as_mut)
            .ok_or_else(|| Error::cache("Laguna LRU slot is empty"))
    }
}

/// Laguna-specific global cache of Metal-prepared routed experts.
///
/// A hash directory and linked LRU slots keep lookup, promotion, and eviction
/// O(1). The slot budget is shared across all routed layers because Laguna's
/// measured routing locality is uneven: a fixed per-layer partition wastes
/// memory and creates transient expert loads. This policy is intentionally
/// independent from the GLM expert cache.
#[derive(Debug)]
pub struct LagunaExpertCache {
    directory: LruDirectory<(usize, usize)>,
    weights: Vec<Option<LagunaDeviceExpertWeights>>,
    bytes_per_expert: u64,
    routed_layer_count: usize,
    experts_per_layer: usize,
    metrics: LagunaExpertCacheMetrics,
}

impl LagunaExpertCache {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        capacity_experts: usize,
        top_k: usize,
        bytes_per_expert: u64,
        routed_layer_count: usize,
        experts_per_layer: usize,
    ) -> Result<Self> {
        if top_k == 0 {
            return Err(Error::cache("Laguna expert top-k must be positive"));
        }
        if routed_layer_count == 0 {
            return Err(Error::cache(
                "Laguna expert cache requires at least one routed layer",
            ));
        }
        if experts_per_layer < top_k {
            return Err(Error::cache(format!(
                "Laguna experts per layer {experts_per_layer} is smaller than top-k {top_k}"
            )));
        }
        if bytes_per_expert == 0 {
            return Err(Error::cache(
                "Laguna resident expert byte size must be positive",
            ));
        }
        let maximum_capacity = routed_layer_count
            .checked_mul(experts_per_layer)
            .ok_or_else(|| Error::cache("Laguna maximum expert cache capacity overflow"))?;
        if capacity_experts < top_k {
            return Err(Error::cache(format!(
                "Laguna expert cache capacity {capacity_experts} must hold at least one top-k set of {top_k} experts"
            )));
        }
        if capacity_experts > maximum_capacity {
            return Err(Error::cache(format!(
                "Laguna expert cache capacity {capacity_experts} exceeds all {maximum_capacity} routed experts"
            )));
        }

        let capacity_bytes = u64::try_from(capacity_experts)
            .ok()
            .and_then(|capacity| capacity.checked_mul(bytes_per_expert))
            .ok_or_else(|| Error::cache("Laguna resident expert cache byte budget overflow"))?;

        Ok(Self {
            directory: LruDirectory::new(capacity_experts),
            weights: Vec::with_capacity(capacity_experts),
            bytes_per_expert,
            routed_layer_count,
            experts_per_layer,
            metrics: LagunaExpertCacheMetrics {
                capacity_bytes,
                capacity_experts,
                ..LagunaExpertCacheMetrics::default()
            },
        })
    }

    pub fn metrics(&self) -> LagunaExpertCacheMetrics {
        let resident_experts = self.directory.len();
        LagunaExpertCacheMetrics {
            resident_experts,
            resident_bytes: u64::try_from(resident_experts)
                .unwrap_or(u64::MAX)
                .saturating_mul(self.bytes_per_expert),
            ..self.metrics
        }
    }

    pub fn clear(&mut self) {
        self.directory.clear();
        self.weights.clear();
    }

    fn lookup(
        &mut self,
        layer_index: usize,
        expert_id: usize,
    ) -> Result<Option<LagunaDeviceExpertWeights>> {
        self.validate_key(layer_index, expert_id)?;
        self.metrics.lookups = self.metrics.lookups.saturating_add(1);
        if let Some(slot) = self.directory.get((layer_index, expert_id))? {
            let weights = self
                .weights
                .get(slot)
                .and_then(Option::as_ref)
                .cloned()
                .ok_or_else(|| Error::cache("Laguna expert LRU points to an empty weight slot"))?;
            self.metrics.hits = self.metrics.hits.saturating_add(1);
            return Ok(Some(weights));
        }
        self.metrics.misses = self.metrics.misses.saturating_add(1);
        Ok(None)
    }

    fn insert_loaded(
        &mut self,
        weights: LagunaDeviceExpertWeights,
        source_bytes: u64,
    ) -> Result<LagunaDeviceExpertWeights> {
        let layer_index = weights.layer_index;
        let expert_id = weights.expert_id;
        self.validate_weight_identity_and_size(layer_index, expert_id, &weights, source_bytes)?;
        let (slot, evicted) = self.directory.insert((layer_index, expert_id))?;
        if evicted {
            self.metrics.evictions = self.metrics.evictions.saturating_add(1);
        }
        if slot == self.weights.len() {
            self.weights.push(Some(weights.clone()));
        } else {
            self.weights[slot] = Some(weights.clone());
        }
        self.metrics.resident_loads = self.metrics.resident_loads.saturating_add(1);
        self.metrics.resident_load_bytes = self
            .metrics
            .resident_load_bytes
            .saturating_add(source_bytes);
        self.metrics.ssd_read_bytes = self.metrics.ssd_read_bytes.saturating_add(source_bytes);
        Ok(weights)
    }

    fn record_ready_wave(&mut self) {
        self.metrics.ready_waves = self.metrics.ready_waves.saturating_add(1);
    }

    fn validate_key(&self, layer_index: usize, expert_id: usize) -> Result<()> {
        let routed_index = layer_index
            .checked_sub(1)
            .ok_or_else(|| Error::cache("Laguna expert cache does not contain dense layer zero"))?;
        if routed_index >= self.routed_layer_count {
            return Err(Error::cache(format!(
                "Laguna expert cache has no routed layer {layer_index}"
            )));
        }
        if expert_id >= self.experts_per_layer {
            return Err(Error::cache(format!(
                "Laguna expert {expert_id} is outside 0..{}",
                self.experts_per_layer
            )));
        }
        Ok(())
    }

    fn validate_weight_identity_and_size(
        &self,
        layer_index: usize,
        expert_id: usize,
        weights: &LagunaDeviceExpertWeights,
        source_bytes: u64,
    ) -> Result<()> {
        if weights.layer_index != layer_index || weights.expert_id != expert_id {
            return Err(Error::cache(format!(
                "Laguna loaded expert identity mismatch: expected layer {layer_index} expert {expert_id}, got layer {} expert {}",
                weights.layer_index, weights.expert_id
            )));
        }
        let prepared_bytes = u64::try_from(weights.storage_bytes()?)
            .map_err(|_| Error::cache("Laguna prepared expert size does not fit u64"))?;
        if source_bytes != self.bytes_per_expert || prepared_bytes != self.bytes_per_expert {
            return Err(Error::cache(format!(
                "Laguna resident expert byte size mismatch: expected {}, prepared {prepared_bytes}, source {source_bytes}",
                self.bytes_per_expert
            )));
        }
        Ok(())
    }
}

/// Runs Laguna's dense SwiGLU MLP and adds the transformer residual.
///
/// Shapes: normalized input `[B,T,3072]`, gate/up `[B,T,12288]`, activated
/// product `[B,T,12288]`, down projection and output `[B,T,3072]`.
pub fn forward_dense_mlp_residual<B: Backend>(
    config: &LagunaConfig,
    normalized: &DeviceValue,
    residual: &DeviceValue,
    weights: &LagunaDeviceDenseWeights,
    backend: &B,
) -> Result<DeviceValue> {
    validate_hidden_pair(config, normalized, residual)?;
    validate_dense_weights(
        weights,
        config.hidden_size,
        config.intermediate_size,
        "dense MLP",
    )?;
    let output = dense_output(normalized, weights, backend)?;
    required(
        "Laguna dense MLP residual add",
        backend.add_device(residual, &output)?,
    )
}

/// Runs one exact Laguna top-10 sparse MoE MLP and adds the residual.
///
/// Cache misses prefetch and prepare persistent Metal buffers on parallel I/O
/// workers while Metal executes the shared expert and resident cache hits.
/// Each completed group is submitted without waiting for slower experts.
#[allow(clippy::too_many_arguments)]
pub fn forward_sparse_mlp_residual<B: Backend>(
    config: &LagunaConfig,
    layer_index: usize,
    normalized: &DeviceValue,
    residual: &DeviceValue,
    weights: &LagunaDeviceMoeWeights,
    index: &LagunaWeightIndex,
    expert_cache: &mut LagunaExpertCache,
    expert_prefetch_pool: &LagunaExpertPrefetchPool,
    backend: &B,
) -> Result<DeviceValue> {
    validate_hidden_pair(config, normalized, residual)?;
    if layer_index == 0 || layer_index >= config.num_hidden_layers {
        return Err(Error::model(format!(
            "Laguna sparse MLP layer must be within 1..{}, got {layer_index}",
            config.num_hidden_layers - 1
        )));
    }
    validate_dense_weights(
        &weights.shared,
        config.hidden_size,
        config.shared_expert_intermediate_size,
        "shared expert",
    )?;
    if weights.router.rows() != config.num_experts
        || weights.router.columns() != config.hidden_size
        || weights.correction_bias.len() != config.num_experts
    {
        return Err(Error::weights(format!(
            "Laguna router must be [{},{}] with {} correction values",
            config.num_experts, config.hidden_size, config.num_experts
        )));
    }

    let original_shape = normalized.dims().to_vec();
    let token_count = normalized.element_count()? / config.hidden_size;
    let flat_normalized = normalized.reshape(vec![token_count, config.hidden_size])?;
    let flat_residual = residual.reshape(vec![token_count, config.hidden_size])?;

    // The CPU needs only the small selected-ID vector to determine which
    // Safetensors ranges belong to this exact router result.
    let router_logits = required(
        "Laguna router projection",
        backend.bf16_linear_device(&weights.router, &flat_normalized)?,
    )?;
    let routing = required_routing(backend.moe_router_topk_resident_device(
        &router_logits,
        &weights.correction_bias,
        config.num_experts_per_tok,
        config.norm_topk_prob,
        config.moe_routed_scaling_factor as f32,
    )?)?;
    let expert_ids = backend
        .moe_router_expert_ids_device(&routing)?
        .ok_or_else(|| Error::backend("Laguna router expert-ID read requires native Metal"))?;
    let groups = group_assignments(
        &expert_ids,
        token_count,
        config.num_experts_per_tok,
        config.num_experts,
    )?;

    let assignment_count = routing.assignment_count()?;
    let expert_outputs = backend
        .device_alloc_f32_tensor(&[assignment_count, config.hidden_size])?
        .ok_or_else(|| Error::backend("Laguna expert output allocation requires native Metal"))?;

    let mut cache_hits = Vec::new();
    let mut cache_misses = HashMap::with_capacity(groups.len());
    let mut load_jobs = Vec::new();
    for group in groups {
        match expert_cache.lookup(layer_index, group.expert_id)? {
            Some(expert) => cache_hits.push((group, expert)),
            None => {
                let source = index.expert(layer_index, group.expert_id)?;
                let source_bytes = source.storage_bytes()?;
                load_jobs.push(ExpertLoadJob {
                    expert_id: group.expert_id,
                    source,
                    source_bytes,
                });
                cache_misses.insert(group.expert_id, group);
            }
        }
    }

    let next_job = AtomicUsize::new(0);
    expert_prefetch_pool.in_place_scope(|scope| -> Result<DeviceValue> {
        let (ready_sender, ready_receiver) = mpsc::channel::<ExpertLoadResult>();
        let worker_count = load_jobs.len().min(MAX_EXPERT_IO_WORKERS);
        for _ in 0..worker_count {
            let ready_sender = ready_sender.clone();
            let jobs = &load_jobs;
            let next_job = &next_job;
            scope.spawn(move |_| loop {
                let job_index = next_job.fetch_add(1, Ordering::Relaxed);
                let Some(job) = jobs.get(job_index) else {
                    break;
                };
                let result = load_expert(job, config, backend);
                if ready_sender.send((job.expert_id, result)).is_err() {
                    return;
                }
            });
        }
        drop(ready_sender);

        // I/O workers start first. Metal can execute the shared expert and
        // resident hits while cold experts are copied into persistent buffers.
        let shared = dense_output(&flat_normalized, &weights.shared, backend)?;
        encode_expert_wave(
            &flat_normalized,
            &expert_outputs,
            &cache_hits,
            token_count,
            config.num_experts_per_tok,
            backend,
        )?;
        if !cache_hits.is_empty() {
            expert_cache.record_ready_wave();
        }
        backend.device_submit()?;

        let mut remaining = cache_misses.len();
        while remaining > 0 {
            let first = ready_receiver.recv().map_err(|_| {
                Error::weights(format!(
                    "Laguna expert workers stopped with {remaining} loads pending"
                ))
            })?;
            let mut completed = vec![first];
            completed.extend(ready_receiver.try_iter());

            let mut ready_experts = Vec::with_capacity(completed.len());
            for (expert_id, result) in completed {
                let ready = result?;
                let expert = expert_cache.insert_loaded(ready.weights, ready.source_bytes)?;
                let group = cache_misses.remove(&expert_id).ok_or_else(|| {
                    Error::model(format!(
                        "Laguna expert loader completed unknown expert {expert_id}"
                    ))
                })?;
                ready_experts.push((group, expert));
                remaining -= 1;
            }
            encode_expert_wave(
                &flat_normalized,
                &expert_outputs,
                &ready_experts,
                token_count,
                config.num_experts_per_tok,
                backend,
            )?;
            expert_cache.record_ready_wave();
            backend.device_submit()?;
        }

        required(
            "Laguna sparse MoE combine",
            backend.moe_topk_combine_residual_device(
                &shared,
                &flat_residual,
                &expert_outputs,
                &routing,
            )?,
        )?
        .reshape(original_shape)
    })
}

#[derive(Debug)]
struct ExpertLoadJob<'a> {
    expert_id: usize,
    source: &'a LagunaExpertWeights,
    source_bytes: u64,
}

type ExpertLoadResult = (usize, Result<ExpertLoadReady>);

#[derive(Debug)]
struct ExpertLoadReady {
    weights: LagunaDeviceExpertWeights,
    source_bytes: u64,
}

fn load_expert<B: Backend>(
    job: &ExpertLoadJob<'_>,
    config: &LagunaConfig,
    backend: &B,
) -> Result<ExpertLoadReady> {
    job.source.prefetch()?;
    let weights = LagunaDeviceExpertWeights::prepare_cached(job.source, config, backend)?;
    Ok(ExpertLoadReady {
        weights,
        source_bytes: job.source_bytes,
    })
}

fn encode_expert_wave<B: Backend>(
    flat_input: &DeviceValue,
    expert_outputs: &DeviceValue,
    groups: &[(ExpertAssignments, LagunaDeviceExpertWeights)],
    token_count: usize,
    top_k: usize,
    backend: &B,
) -> Result<()> {
    if groups.is_empty() {
        return Ok(());
    }
    let groups = groups
        .iter()
        .map(|(assignments, expert)| {
            W4ExpertGroup::new(
                &expert.gate,
                &expert.up,
                &expert.down,
                &assignments.assignment_indices,
            )
        })
        .collect::<Vec<_>>();
    backend
        .w4_groupwise_expert_wave_device(&groups, flat_input, token_count, top_k, expert_outputs)?
        .ok_or_else(|| Error::backend("Laguna routed expert wave requires native Metal"))
}

fn dense_output<B: Backend>(
    input: &DeviceValue,
    weights: &LagunaDeviceDenseWeights,
    backend: &B,
) -> Result<DeviceValue> {
    let activated = required(
        "Laguna fused gate/up SwiGLU",
        backend.bf16_gate_up_swiglu_device(&weights.gate, &weights.up, input)?,
    )?;
    required(
        "Laguna dense down projection",
        backend.bf16_linear_device(&weights.down, &activated)?,
    )
}

fn validate_hidden_pair(
    config: &LagunaConfig,
    normalized: &DeviceValue,
    residual: &DeviceValue,
) -> Result<()> {
    if normalized.dtype() != DType::F32 || residual.dtype() != DType::F32 {
        return Err(Error::model("Laguna MLP inputs must be F32 on Metal"));
    }
    if normalized.dims() != residual.dims()
        || normalized.dims().len() != 3
        || normalized.dims()[2] != config.hidden_size
        || normalized.dims()[0] == 0
        || normalized.dims()[1] == 0
    {
        return Err(Error::model(format!(
            "Laguna MLP normalized/residual shapes must match [B,T,{}], got {:?}/{:?}",
            config.hidden_size,
            normalized.dims(),
            residual.dims()
        )));
    }
    Ok(())
}

fn validate_dense_weights(
    weights: &LagunaDeviceDenseWeights,
    hidden_size: usize,
    intermediate_size: usize,
    label: &str,
) -> Result<()> {
    for (name, matrix, rows, columns) in [
        ("gate", &weights.gate, intermediate_size, hidden_size),
        ("up", &weights.up, intermediate_size, hidden_size),
        ("down", &weights.down, hidden_size, intermediate_size),
    ] {
        if matrix.rows() != rows || matrix.columns() != columns {
            return Err(Error::weights(format!(
                "Laguna {label} {name} matrix must be [{rows},{columns}], got [{},{}]",
                matrix.rows(),
                matrix.columns()
            )));
        }
    }
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct ExpertAssignments {
    expert_id: usize,
    assignment_indices: Vec<u32>,
}

fn group_assignments(
    expert_ids: &[u32],
    token_count: usize,
    top_k: usize,
    expert_count: usize,
) -> Result<Vec<ExpertAssignments>> {
    let expected = token_count
        .checked_mul(top_k)
        .ok_or_else(|| Error::model("Laguna router assignment count overflow"))?;
    if expert_ids.len() != expected {
        return Err(Error::model(format!(
            "Laguna router returned {} expert IDs, expected {expected}",
            expert_ids.len()
        )));
    }
    let mut assignments_by_expert = (0..expert_count)
        .map(|_| Vec::<u32>::new())
        .collect::<Vec<_>>();
    for (assignment_index, expert_id) in expert_ids.iter().copied().enumerate() {
        let expert_id = expert_id as usize;
        if expert_id >= expert_count {
            return Err(Error::model(format!(
                "Laguna router selected expert {expert_id}, but expert_count is {expert_count}"
            )));
        }
        assignments_by_expert[expert_id].push(
            u32::try_from(assignment_index)
                .map_err(|_| Error::model("Laguna assignment index exceeds u32"))?,
        );
    }
    Ok(assignments_by_expert
        .into_iter()
        .enumerate()
        .filter_map(|(expert_id, assignment_indices)| {
            (!assignment_indices.is_empty()).then_some(ExpertAssignments {
                expert_id,
                assignment_indices,
            })
        })
        .collect())
}

fn required(label: &str, value: Option<DeviceValue>) -> Result<DeviceValue> {
    value.ok_or_else(|| Error::backend(format!("{label} requires native Metal")))
}

fn required_routing(value: Option<DeviceRouterTopK>) -> Result<DeviceRouterTopK> {
    value.ok_or_else(|| Error::backend("Laguna top-k routing requires native Metal"))
}

#[cfg(test)]
mod tests {
    use super::{group_assignments, LagunaExpertCache, LruDirectory};

    #[test]
    fn global_lru_lookup_promotion_and_eviction_are_constant_time() {
        let mut lru = LruDirectory::new(2);
        let (first_slot, evicted) = lru.insert((1, 10)).unwrap();
        assert!(!evicted);
        let (_, evicted) = lru.insert((1, 20)).unwrap();
        assert!(!evicted);

        assert_eq!(lru.get((1, 10)).unwrap(), Some(first_slot));
        let (reused_slot, evicted) = lru.insert((2, 30)).unwrap();
        assert!(evicted);
        assert_ne!(reused_slot, first_slot);
        assert_eq!(lru.get((1, 20)).unwrap(), None);
        assert_eq!(lru.get((1, 10)).unwrap(), Some(first_slot));
        assert!(lru.get((2, 30)).unwrap().is_some());
        assert_eq!(lru.len(), 2);
    }

    #[test]
    fn groups_token_major_topk_assignments_by_expert() {
        let groups = group_assignments(&[3, 1, 3, 2, 1, 2], 2, 3, 4).unwrap();
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].expert_id, 1);
        assert_eq!(groups[0].assignment_indices, [1, 4]);
        assert_eq!(groups[1].expert_id, 2);
        assert_eq!(groups[1].assignment_indices, [3, 5]);
        assert_eq!(groups[2].expert_id, 3);
        assert_eq!(groups[2].assignment_indices, [0, 2]);
    }

    #[test]
    fn cache_budget_is_global_and_holds_at_least_one_topk_set() {
        let error = LagunaExpertCache::new(9, 10, 128, 2, 256).unwrap_err();
        assert!(error.to_string().contains("at least one top-k set"));

        let cache = LagunaExpertCache::new(10, 10, 128, 2, 256).unwrap();
        let metrics = cache.metrics();
        assert_eq!(metrics.capacity_experts, 10);
        assert_eq!(metrics.capacity_bytes, 1_280);
        assert_eq!(metrics.resident_bytes, 0);
    }

    #[test]
    fn cache_rejects_zero_sized_experts() {
        let error = LagunaExpertCache::new(20, 10, 0, 2, 256).unwrap_err();
        assert!(error.to_string().contains("must be positive"));
    }

    #[test]
    fn clear_preserves_global_budget() {
        let mut cache = LagunaExpertCache::new(20, 10, 128, 2, 256).unwrap();
        cache.clear();

        let metrics = cache.metrics();
        assert_eq!(metrics.capacity_experts, 20);
        assert_eq!(metrics.resident_experts, 0);
        assert_eq!(metrics.resident_bytes, 0);
    }

    #[test]
    fn metrics_prove_hits_do_not_trigger_source_reads() {
        let metrics = super::LagunaExpertCacheMetrics {
            lookups: 10,
            hits: 6,
            misses: 4,
            resident_loads: 4,
            resident_load_bytes: 400,
            ssd_read_bytes: 400,
            resident_experts: 3,
            capacity_experts: 4,
            resident_bytes: 300,
            capacity_bytes: 400,
            ..super::LagunaExpertCacheMetrics::default()
        };

        metrics.validate().unwrap();
        assert_eq!(metrics.source_loads(), metrics.misses);
        assert_ne!(metrics.source_loads(), metrics.lookups);
    }
}
