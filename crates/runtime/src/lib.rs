#![deny(unsafe_code)]

//! GLM-5.2 production runtime orchestration.

mod cache_budget;
mod telemetry;

pub use cache_budget::{
    CacheBudgetAdjustment, CacheBudgetPlan, CacheBudgetSignals, CacheBudgetSpec, ContextTier,
    DEFAULT_DYNAMIC_CACHE_BUDGET_BYTES,
};

use std::{
    cell::RefCell,
    collections::HashSet,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex, OnceLock,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

#[cfg(test)]
use backend::BackendCapabilities;
use backend::{Backend, DevicePagedKvView, DeviceSelectedKvView, ExpertCacheMetrics};
#[cfg(test)]
use cache::LayeredPagedCacheAppendReport;
use cache::{
    ColdKvBlockStore, ColdKvCodec, ColdKvSelectedQ8LayerRows, ColdKvStoreSpec, DsaIndexLayerAppend,
    DsaIndexStoreSpec, LayerKvCacheAppend, LayeredColdKvBlockStore, LayeredDsaIndexBlockStore,
    LayeredPagedKvCache, LayeredPagedKvCacheSpec,
};
use common::{validate_exact_shape, Error, F32Tensor, Result, Shape};
use config::Config;
#[cfg(test)]
use model::ModelGreedyOutput;
use model::{
    set_layer_profile_context, LayerDeviceKvCacheTensors, LayerKvCacheTensors, Model,
    ModelDeviceTokenSequenceOutput,
};
use telemetry::{
    capture_memory_snapshot, log_memory_snapshot, RuntimeKvMemoryBytes, RuntimeMemorySnapshot,
};

pub const DEFAULT_KV_PAGE_SIZE: usize = 128;
pub const MAX_REFERENCE_PREFILL_ATTENTION_SCORE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
#[cfg(test)]
const MAX_REFERENCE_MLA_KV_CACHE_BYTES: u64 = 24 * 1024 * 1024 * 1024;
const F32_BYTES: u64 = 4;
#[cfg(test)]
const F16_BYTES: u64 = 2;
const Q2_K_BLOCK_VALUES: usize = 256;
const Q2_K_BLOCK_BYTES: usize = 84;
const MTP_DRAFTS_PER_STEP: usize = 2;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GenerationOptions {
    pub hot_kv_cache_budget_bytes: Option<usize>,
    pub dynamic_cache_budget: Option<CacheBudgetSpec>,
    pub profile_token_costs: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvCacheMetrics {
    pub full_layer_lookups: u64,
    pub full_layer_hits: u64,
    pub full_layer_misses: u64,
    pub selected_row_lookups: u64,
    pub selected_row_hits: u64,
    pub selected_row_misses: u64,
    pub selected_rows: u64,
    pub page_lookups: u64,
    pub page_hits: u64,
    pub page_misses: u64,
    pub ssd_read_bytes: u64,
    pub hot_bytes: u64,
    pub cold_bytes: u64,
    pub cached_tokens: u64,
    pub read_nanoseconds: u64,
    pub write_nanoseconds: u64,
}

impl KvCacheMetrics {
    pub fn lookups(self) -> u64 {
        self.page_lookups
    }

    pub fn hits(self) -> u64 {
        self.page_hits
    }

    pub fn misses(self) -> u64 {
        self.page_misses
    }

    pub fn hit_rate(self) -> f64 {
        let lookups = self.lookups();
        if lookups == 0 {
            return 0.0;
        }
        self.hits() as f64 / lookups as f64
    }

    pub fn miss_rate(self) -> f64 {
        let lookups = self.lookups();
        if lookups == 0 {
            return 0.0;
        }
        self.misses() as f64 / lookups as f64
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamingGenerationReport {
    pub kv_cache: KvCacheMetrics,
    pub mtp: MtpMetrics,
    pub cache_budget: Option<CacheBudgetRuntimeReport>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MtpMetrics {
    pub enabled: bool,
    pub verification_passes: u64,
    pub target_tokens: u64,
    pub draft_tokens: u64,
    pub accepted_draft_tokens: u64,
}

impl MtpMetrics {
    pub fn acceptance_rate(self) -> f64 {
        if self.draft_tokens == 0 {
            return 0.0;
        }
        self.accepted_draft_tokens as f64 / self.draft_tokens as f64
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheBudgetRuntimeReport {
    pub total_bytes: usize,
    pub expert_slots_per_layer: usize,
    pub expert_bytes: usize,
    pub hot_kv_budget_bytes: usize,
    pub all_layer_hot_bytes: usize,
    pub all_layers_fit: bool,
    pub rebalances: usize,
    pub memory_pressure: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenCostReport {
    pub step_index: usize,
    pub prompt_prefill: bool,
    pub token_id: u32,
    pub dense_attention_nanoseconds: u64,
    pub sparse_attention_nanoseconds: u64,
    pub sparse_input_norm_nanoseconds: u64,
    pub sparse_q_projection_nanoseconds: u64,
    pub sparse_kv_projection_nanoseconds: u64,
    pub sparse_cache_layout_nanoseconds: u64,
    pub sparse_dsa_indexer_nanoseconds: u64,
    pub sparse_context_attention_nanoseconds: u64,
    pub sparse_output_projection_nanoseconds: u64,
    pub moe_routing_nanoseconds: u64,
    pub expert_cache_lookup_nanoseconds: u64,
    pub expert_ssd_load_nanoseconds: u64,
    pub q2_expert_matmul_gpu_nanoseconds: u64,
    pub kv_cache_read_nanoseconds: u64,
    pub kv_cache_write_nanoseconds: u64,
    pub output_projection_nanoseconds: u64,
    pub sampling_argmax_nanoseconds: u64,
    pub total_token_nanoseconds: u64,
    pub expert_lookups: u64,
    pub expert_cache_hits: u64,
    pub expert_cache_misses: u64,
    pub kv_page_lookups: u64,
    pub kv_page_hits: u64,
    pub kv_page_misses: u64,
    pub expert_ssd_read_bytes: u64,
    pub kv_ssd_read_bytes: u64,
}

impl TokenCostReport {
    pub fn expert_cache_hit_rate(self) -> f64 {
        rate_u64(self.expert_cache_hits, self.expert_lookups)
    }

    pub fn kv_page_hit_rate(self) -> f64 {
        rate_u64(self.kv_page_hits, self.kv_page_lookups)
    }

    pub fn ssd_read_bytes(self) -> u64 {
        self.expert_ssd_read_bytes
            .saturating_add(self.kv_ssd_read_bytes)
    }

    pub fn sparse_attention_accounted_nanoseconds(self) -> u64 {
        self.sparse_input_norm_nanoseconds
            .saturating_add(self.sparse_q_projection_nanoseconds)
            .saturating_add(self.sparse_kv_projection_nanoseconds)
            .saturating_add(self.sparse_cache_layout_nanoseconds)
            .saturating_add(self.sparse_dsa_indexer_nanoseconds)
            .saturating_add(self.sparse_context_attention_nanoseconds)
            .saturating_add(self.sparse_output_projection_nanoseconds)
    }

    pub fn sparse_attention_unattributed_nanoseconds(self) -> u64 {
        self.sparse_attention_nanoseconds
            .saturating_sub(self.sparse_attention_accounted_nanoseconds())
    }
}

struct TokenCostSnapshot {
    started_at: Instant,
    experts: ExpertCacheMetrics,
    kv: KvCacheMetrics,
}

struct TokenCostProfileScope;

impl Drop for TokenCostProfileScope {
    fn drop(&mut self) {
        model::disable_token_cost_profile();
    }
}

impl TokenCostSnapshot {
    fn capture<B: Backend>(
        backend: &B,
        device_cache: &Option<DevicePagedRuntimeCache>,
    ) -> Result<Self> {
        Ok(Self {
            started_at: Instant::now(),
            experts: backend.expert_cache_metrics()?,
            kv: device_cache
                .as_ref()
                .map(DevicePagedRuntimeCache::cache_metrics)
                .unwrap_or_default(),
        })
    }

    fn finish<B: Backend>(
        self,
        step_index: usize,
        prompt_prefill: bool,
        token_id: u32,
        backend: &B,
        device_cache: &Option<DevicePagedRuntimeCache>,
    ) -> Result<TokenCostReport> {
        let experts = backend.expert_cache_metrics()?;
        let kv = device_cache
            .as_ref()
            .map(DevicePagedRuntimeCache::cache_metrics)
            .unwrap_or_default();
        let model = model::take_token_cost_profile()?;
        Ok(TokenCostReport {
            step_index,
            prompt_prefill,
            token_id,
            dense_attention_nanoseconds: model.dense_attention_nanoseconds,
            sparse_attention_nanoseconds: model.sparse_attention_nanoseconds,
            sparse_input_norm_nanoseconds: model.sparse_input_norm_nanoseconds,
            sparse_q_projection_nanoseconds: model.sparse_q_projection_nanoseconds,
            sparse_kv_projection_nanoseconds: model.sparse_kv_projection_nanoseconds,
            sparse_cache_layout_nanoseconds: model.sparse_cache_layout_nanoseconds,
            sparse_dsa_indexer_nanoseconds: model.sparse_dsa_indexer_nanoseconds,
            sparse_context_attention_nanoseconds: model.sparse_context_attention_nanoseconds,
            sparse_output_projection_nanoseconds: model.sparse_output_projection_nanoseconds,
            moe_routing_nanoseconds: model.moe_routing_nanoseconds,
            expert_cache_lookup_nanoseconds: experts
                .lookup_nanoseconds
                .saturating_sub(self.experts.lookup_nanoseconds),
            expert_ssd_load_nanoseconds: experts
                .ssd_load_nanoseconds
                .saturating_sub(self.experts.ssd_load_nanoseconds),
            q2_expert_matmul_gpu_nanoseconds: experts
                .q2_matmul_gpu_nanoseconds
                .saturating_sub(self.experts.q2_matmul_gpu_nanoseconds),
            kv_cache_read_nanoseconds: kv.read_nanoseconds.saturating_sub(self.kv.read_nanoseconds),
            kv_cache_write_nanoseconds: kv
                .write_nanoseconds
                .saturating_sub(self.kv.write_nanoseconds),
            output_projection_nanoseconds: model.output_projection_nanoseconds,
            sampling_argmax_nanoseconds: model.sampling_argmax_nanoseconds,
            total_token_nanoseconds: elapsed_nanoseconds_u64(self.started_at.elapsed()),
            expert_lookups: experts.lookups.saturating_sub(self.experts.lookups),
            expert_cache_hits: experts.hits.saturating_sub(self.experts.hits),
            expert_cache_misses: experts.misses.saturating_sub(self.experts.misses),
            kv_page_lookups: kv.page_lookups.saturating_sub(self.kv.page_lookups),
            kv_page_hits: kv.page_hits.saturating_sub(self.kv.page_hits),
            kv_page_misses: kv.page_misses.saturating_sub(self.kv.page_misses),
            expert_ssd_read_bytes: experts
                .ssd_read_bytes
                .saturating_sub(self.experts.ssd_read_bytes),
            kv_ssd_read_bytes: kv.ssd_read_bytes.saturating_sub(self.kv.ssd_read_bytes),
        })
    }
}

fn log_token_cost(report: TokenCostReport) {
    tracing::info!(
        target: "inferno::token_cost",
        step_index = report.step_index,
        phase = if report.prompt_prefill { "prefill" } else { "decode" },
        token_id = report.token_id,
        dense_attention_ms = report.dense_attention_nanoseconds as f64 / 1_000_000.0,
        sparse_attention_ms = report.sparse_attention_nanoseconds as f64 / 1_000_000.0,
        sparse_input_norm_ms = report.sparse_input_norm_nanoseconds as f64 / 1_000_000.0,
        sparse_q_projection_ms = report.sparse_q_projection_nanoseconds as f64 / 1_000_000.0,
        sparse_kv_projection_ms = report.sparse_kv_projection_nanoseconds as f64 / 1_000_000.0,
        sparse_cache_layout_ms = report.sparse_cache_layout_nanoseconds as f64 / 1_000_000.0,
        sparse_dsa_indexer_ms = report.sparse_dsa_indexer_nanoseconds as f64 / 1_000_000.0,
        sparse_context_attention_ms = report.sparse_context_attention_nanoseconds as f64 / 1_000_000.0,
        sparse_output_projection_ms = report.sparse_output_projection_nanoseconds as f64 / 1_000_000.0,
        sparse_unattributed_ms = report.sparse_attention_unattributed_nanoseconds() as f64 / 1_000_000.0,
        moe_routing_ms = report.moe_routing_nanoseconds as f64 / 1_000_000.0,
        expert_cache_lookup_ms = report.expert_cache_lookup_nanoseconds as f64 / 1_000_000.0,
        expert_ssd_load_ms = report.expert_ssd_load_nanoseconds as f64 / 1_000_000.0,
        q2_expert_matmul_gpu_ms = report.q2_expert_matmul_gpu_nanoseconds as f64 / 1_000_000.0,
        kv_cache_read_ms = report.kv_cache_read_nanoseconds as f64 / 1_000_000.0,
        kv_cache_write_ms = report.kv_cache_write_nanoseconds as f64 / 1_000_000.0,
        output_projection_ms = report.output_projection_nanoseconds as f64 / 1_000_000.0,
        sampling_argmax_ms = report.sampling_argmax_nanoseconds as f64 / 1_000_000.0,
        total_token_ms = report.total_token_nanoseconds as f64 / 1_000_000.0,
        expert_cache_hit_rate = report.expert_cache_hit_rate(),
        kv_page_hit_rate = report.kv_page_hit_rate(),
        ssd_read_bytes = report.ssd_read_bytes(),
        ssd_read_mb = report.ssd_read_bytes() as f64 / 1_000_000.0,
        expert_ssd_read_bytes = report.expert_ssd_read_bytes,
        kv_ssd_read_bytes = report.kv_ssd_read_bytes,
        expert_loads = report.expert_cache_misses,
        expert_lookups = report.expert_lookups,
        kv_page_lookups = report.kv_page_lookups,
        profiling_sync = true,
        timings_overlap = true,
        "generated token cost breakdown"
    );
}

const CACHE_SIGNAL_INTERVAL_TOKENS: usize = 8;
const CACHE_MEMORY_INTERVAL_TOKENS: usize = 128;
const CACHE_PRESSURE_FREE_BYTES: u64 = 512 * 1024 * 1024;
const CACHE_PRESSURE_COMPRESSED_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const CACHE_PRESSURE_SWAP_BYTES: u64 = 1024 * 1024 * 1024;
const CACHE_PRESSURE_SWAP_GROWTH_BYTES: u64 = 64 * 1024 * 1024;

struct CacheBudgetController {
    spec: CacheBudgetSpec,
    current: CacheBudgetPlan,
    signals: CacheBudgetSignals,
    last_signal_context: usize,
    last_memory_context: Option<usize>,
    last_expert_lookups: u64,
    last_expert_hits: u64,
    last_kv_lookups: u64,
    last_kv_misses: u64,
    baseline_swap_bytes: Option<u64>,
    rebalances: usize,
}

impl CacheBudgetController {
    fn new(spec: CacheBudgetSpec, context_len: usize) -> Result<Self> {
        let current = spec.plan(context_len, CacheBudgetSignals::default())?;
        Ok(Self {
            spec,
            current,
            signals: CacheBudgetSignals::default(),
            last_signal_context: context_len,
            last_memory_context: None,
            last_expert_lookups: 0,
            last_expert_hits: 0,
            last_kv_lookups: 0,
            last_kv_misses: 0,
            baseline_swap_bytes: None,
            rebalances: 0,
        })
    }

    fn maybe_rebalance<B: Backend>(
        &mut self,
        context_len: usize,
        backend: &B,
        cache: &mut DevicePagedRuntimeCache,
    ) -> Result<()> {
        let context_plan = self.spec.plan(context_len, self.signals)?;
        let page_changed = context_plan.capacity_tokens != self.current.capacity_tokens;
        let signal_due = context_len
            >= self
                .last_signal_context
                .saturating_add(CACHE_SIGNAL_INTERVAL_TOKENS);
        if !page_changed && !signal_due {
            return Ok(());
        }

        if signal_due {
            let experts = backend.expert_cache_metrics()?;
            let kv = cache.cache_metrics();
            self.signals.expert_lookups = experts.lookups.saturating_sub(self.last_expert_lookups);
            self.signals.expert_hits = experts.hits.saturating_sub(self.last_expert_hits);
            self.signals.kv_lookups = kv.lookups().saturating_sub(self.last_kv_lookups);
            self.signals.kv_misses = kv.misses().saturating_sub(self.last_kv_misses);
            self.last_expert_lookups = experts.lookups;
            self.last_expert_hits = experts.hits;
            self.last_kv_lookups = kv.lookups();
            self.last_kv_misses = kv.misses();
            self.last_signal_context = context_len;

            let memory_due = self.last_memory_context.is_none_or(|last| {
                context_len >= last.saturating_add(CACHE_MEMORY_INTERVAL_TOKENS)
            });
            if memory_due {
                let memory = capture_memory_snapshot(backend, cache.runtime_kv_memory());
                let baseline_swap = *self
                    .baseline_swap_bytes
                    .get_or_insert(memory.swap_used_bytes.unwrap_or(0));
                self.signals.memory_pressure = memory_pressure(memory, baseline_swap);
                self.last_memory_context = Some(context_len);
            }
        }

        let next = self.spec.plan(context_len, self.signals)?;
        if next.expert_slots_per_layer != self.current.expert_slots_per_layer {
            backend.resize_expert_cache_slots_per_layer(next.expert_slots_per_layer)?;
        }
        if next.hot_kv_budget_bytes != self.current.hot_kv_budget_bytes {
            cache.set_hot_all_layers_budget(next.hot_kv_budget_bytes)?;
        }
        if next.expert_slots_per_layer != self.current.expert_slots_per_layer
            || next.hot_kv_budget_bytes != self.current.hot_kv_budget_bytes
        {
            self.rebalances = self.rebalances.saturating_add(1);
            tracing::info!(
                target: "inferno::cache_budget",
                context_len,
                tier = ?next.tier,
                adjustment = ?next.adjustment,
                expert_slots_per_layer = next.expert_slots_per_layer,
                expert_cache_gb = next.expert_bytes as f64 / 1_000_000_000.0,
                hot_kv_budget_gb = next.hot_kv_budget_bytes as f64 / 1_000_000_000.0,
                all_layers_fit = next.all_layers_fit,
                expert_hit_rate = rate_u64(self.signals.expert_hits, self.signals.expert_lookups),
                kv_miss_rate = rate_u64(self.signals.kv_misses, self.signals.kv_lookups),
                memory_pressure = self.signals.memory_pressure,
                "rebalanced shared expert and hot KV cache budget"
            );
        }
        self.current = next;
        Ok(())
    }

    fn report(&self) -> CacheBudgetRuntimeReport {
        CacheBudgetRuntimeReport {
            total_bytes: self.spec.total_bytes,
            expert_slots_per_layer: self.current.expert_slots_per_layer,
            expert_bytes: self.current.expert_bytes,
            hot_kv_budget_bytes: self.current.hot_kv_budget_bytes,
            all_layer_hot_bytes: self.current.all_layer_hot_bytes,
            all_layers_fit: self.current.all_layers_fit,
            rebalances: self.rebalances,
            memory_pressure: self.signals.memory_pressure,
        }
    }
}

fn memory_pressure(snapshot: RuntimeMemorySnapshot, baseline_swap_bytes: u64) -> bool {
    let swap_used = snapshot.swap_used_bytes.unwrap_or(0);
    let swap_growth = swap_used.saturating_sub(baseline_swap_bytes);
    let swap_pressure =
        swap_used >= CACHE_PRESSURE_SWAP_BYTES || swap_growth >= CACHE_PRESSURE_SWAP_GROWTH_BYTES;
    let compressed_pressure = snapshot
        .system_compressed_bytes
        .is_some_and(|bytes| bytes >= CACHE_PRESSURE_COMPRESSED_BYTES);
    let low_free_memory = snapshot
        .system_free_bytes
        .is_some_and(|bytes| bytes <= CACHE_PRESSURE_FREE_BYTES);
    swap_pressure || (compressed_pressure && low_free_memory)
}

fn rate_u64(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        return 0.0;
    }
    numerator as f64 / denominator as f64
}

fn elapsed_nanoseconds_u64(elapsed: Duration) -> u64 {
    u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsdKvBenchmarkConfig {
    pub layer_count: usize,
    pub batch: usize,
    pub attention_heads: usize,
    pub tokens: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub block_tokens: usize,
    pub page_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SsdKvBenchmarkReport {
    pub layer_count: usize,
    pub tokens: usize,
    pub stored_bytes: u64,
    pub raw_f32_bytes: u64,
    pub compression_ratio: f32,
    pub write_seconds: f64,
    pub read_seconds: f64,
    pub hot_load_seconds: f64,
    pub ssd_read_gb_per_second: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsaAttentionBenchmarkConfig {
    pub layer_count: usize,
    pub batch: usize,
    pub attention_heads: usize,
    pub tokens: usize,
    pub selected_tokens: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub page_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DsaAttentionBenchmarkReport {
    pub layer_count: usize,
    pub tokens: usize,
    pub selected_tokens: usize,
    pub dense_seconds: f64,
    pub selected_seconds: f64,
    pub dense_seconds_per_layer: f64,
    pub selected_seconds_per_layer: f64,
    pub selected_vs_dense_speedup: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeExpertBenchmarkConfig {
    pub iterations: usize,
    pub token_count: usize,
    pub assignment_count: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub expert_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoeExpertBenchmarkReport {
    pub iterations: usize,
    pub token_count: usize,
    pub assignment_count: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub expert_count: usize,
    pub old_single_assignment_seconds: f64,
    pub multi_expert_seconds: f64,
    pub old_single_assignment_seconds_per_iteration: f64,
    pub multi_expert_seconds_per_iteration: f64,
    pub scheduling_speedup: f64,
}

static Q2_RUNTIME_PROFILE_FILE: OnceLock<Mutex<Option<File>>> = OnceLock::new();
static Q2_RUNTIME_PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefillStrategy {
    Dense,
    Streaming,
}

pub fn enable_memory_telemetry() {
    telemetry::enable_memory_telemetry();
}

pub fn enable_memory_telemetry_file(path: &Path) -> Result<()> {
    telemetry::enable_memory_telemetry_file(path)
}

pub fn enable_q2_runtime_profile(path: &Path) -> Result<()> {
    let mut file = File::create(path).map_err(|error| {
        Error::runtime(format!(
            "failed to create Q2 runtime profile file {}: {error}",
            path.display()
        ))
    })?;
    writeln!(file, "step_index\tlayer_index\tmetric\tvalue").map_err(|error| {
        Error::runtime(format!(
            "failed to write Q2 runtime profile header to {}: {error}",
            path.display()
        ))
    })?;

    let profile = q2_runtime_profile_file();
    let mut profile = profile
        .lock()
        .map_err(|_| Error::runtime("Q2 runtime profile lock poisoned"))?;
    if profile.is_some() {
        return Err(Error::runtime("Q2 runtime profile is already enabled"));
    }
    *profile = Some(file);
    Q2_RUNTIME_PROFILE_ENABLED.store(true, Ordering::Release);
    Ok(())
}

pub fn benchmark_ssd_kv_decode_hot_load<B: Backend>(
    backend: &B,
    config: SsdKvBenchmarkConfig,
) -> Result<SsdKvBenchmarkReport> {
    validate_ssd_kv_benchmark_config(&config)?;
    let cold_spec = ColdKvStoreSpec {
        batch: config.batch,
        attention_heads: config.attention_heads,
        key_head_dim: config.key_head_dim,
        value_head_dim: config.value_head_dim,
        block_tokens: config.block_tokens,
        codec: ColdKvCodec::Q8Row,
    };
    let path = unique_cold_kv_path();
    let mut store = LayeredColdKvBlockStore::create(&path, cold_spec)?;
    let layers = synthetic_benchmark_kv_layers(&config)?;
    let appends = layers
        .iter()
        .map(|(layer_index, k, v)| LayerKvCacheAppend {
            layer_index: *layer_index,
            k,
            v,
        })
        .collect::<Vec<_>>();

    let write_started_at = Instant::now();
    let write_report = store.write_prefill(&appends)?;
    store.store_mut().flush()?;
    let write_seconds = write_started_at.elapsed().as_secs_f64();

    let mut reader = store.store().clone_reader()?;
    let read_started_at = Instant::now();
    let (keys, values) = reader.read_layer_range_contiguous(0, 0, config.tokens)?;
    let read_seconds = read_started_at.elapsed().as_secs_f64();

    let hot_load_started_at = Instant::now();
    let capacity_tokens = device_capacity_tokens(config.tokens, config.page_size, config.tokens)?;
    let page_count = capacity_tokens / config.page_size;
    let hot_k = require_device_value(
        "SSD KV benchmark hot K allocation",
        backend.device_alloc_f32_tensor(&device_cache_shape(
            page_count,
            config.batch,
            config.attention_heads,
            config.page_size,
            config.key_head_dim,
        ))?,
    )?;
    let hot_v = require_device_value(
        "SSD KV benchmark hot V allocation",
        backend.device_alloc_f32_tensor(&device_cache_shape(
            page_count,
            config.batch,
            config.attention_heads,
            config.page_size,
            config.value_head_dim,
        ))?,
    )?;
    let source_k = require_device_value(
        "SSD KV benchmark K upload",
        backend.device_upload_f32_tensor(&keys)?,
    )?;
    let source_v = require_device_value(
        "SSD KV benchmark V upload",
        backend.device_upload_f32_tensor(&values)?,
    )?;
    copy_logical_kv_tokens(
        backend,
        &source_k,
        config.tokens,
        &hot_k,
        capacity_tokens,
        0,
        config.tokens,
        config.batch,
        config.attention_heads,
        config.key_head_dim,
    )?;
    copy_logical_kv_tokens(
        backend,
        &source_v,
        config.tokens,
        &hot_v,
        capacity_tokens,
        0,
        config.tokens,
        config.batch,
        config.attention_heads,
        config.value_head_dim,
    )?;
    backend.device_flush()?;
    let hot_load_seconds = hot_load_started_at.elapsed().as_secs_f64();

    let read_bytes = raw_layer_f32_bytes(
        config.batch,
        config.attention_heads,
        config.tokens,
        config.key_head_dim,
        config.value_head_dim,
    )?;
    let ssd_read_gb_per_second = if read_seconds > 0.0 {
        read_bytes as f64 / 1_000_000_000.0 / read_seconds
    } else {
        0.0
    };
    let report = SsdKvBenchmarkReport {
        layer_count: config.layer_count,
        tokens: config.tokens,
        stored_bytes: write_report.stored_bytes,
        raw_f32_bytes: write_report.raw_f32_bytes,
        compression_ratio: write_report.compression_ratio,
        write_seconds,
        read_seconds,
        hot_load_seconds,
        ssd_read_gb_per_second,
    };
    let _ = std::fs::remove_file(path);
    Ok(report)
}

pub fn benchmark_dsa_selected_attention<B: Backend>(
    backend: &B,
    config: DsaAttentionBenchmarkConfig,
) -> Result<DsaAttentionBenchmarkReport> {
    validate_dsa_attention_benchmark_config(&config)?;
    let q = synthetic_kv_tensor(
        config.batch,
        config.attention_heads,
        1,
        config.key_head_dim,
        17.0,
    )?;
    let current_k = synthetic_kv_tensor(
        config.batch,
        config.attention_heads,
        1,
        config.key_head_dim,
        23.0,
    )?;
    let current_v = synthetic_kv_tensor(
        config.batch,
        config.attention_heads,
        1,
        config.value_head_dim,
        31.0,
    )?;
    let dense_k = synthetic_kv_tensor(
        config.batch,
        config.attention_heads,
        config.tokens,
        config.key_head_dim,
        41.0,
    )?;
    let dense_v = synthetic_kv_tensor(
        config.batch,
        config.attention_heads,
        config.tokens,
        config.value_head_dim,
        53.0,
    )?;
    let selected_k = synthetic_kv_tensor(
        config.batch,
        config.attention_heads,
        config.selected_tokens,
        config.key_head_dim,
        61.0,
    )?;
    let selected_v = synthetic_kv_tensor(
        config.batch,
        config.attention_heads,
        config.selected_tokens,
        config.value_head_dim,
        71.0,
    )?;

    let q = require_device_value(
        "DSA attention benchmark q upload",
        backend.device_upload_f32_tensor(&q)?,
    )?;
    let current_k = require_device_value(
        "DSA attention benchmark current K upload",
        backend.device_upload_f32_tensor(&current_k)?,
    )?;
    let current_v = require_device_value(
        "DSA attention benchmark current V upload",
        backend.device_upload_f32_tensor(&current_v)?,
    )?;
    let selected_k = require_device_value(
        "DSA attention benchmark selected K upload",
        backend.device_upload_f32_tensor(&selected_k)?,
    )?;
    let selected_v = require_device_value(
        "DSA attention benchmark selected V upload",
        backend.device_upload_f32_tensor(&selected_v)?,
    )?;

    let capacity_tokens = device_capacity_tokens(config.tokens, config.page_size, config.tokens)?;
    let page_count = capacity_tokens / config.page_size;
    let dense_hot_k = require_device_value(
        "DSA attention benchmark dense hot K allocation",
        backend.device_alloc_f32_tensor(&device_cache_shape(
            page_count,
            config.batch,
            config.attention_heads,
            config.page_size,
            config.key_head_dim,
        ))?,
    )?;
    let dense_hot_v = require_device_value(
        "DSA attention benchmark dense hot V allocation",
        backend.device_alloc_f32_tensor(&device_cache_shape(
            page_count,
            config.batch,
            config.attention_heads,
            config.page_size,
            config.value_head_dim,
        ))?,
    )?;
    let dense_source_k = require_device_value(
        "DSA attention benchmark dense K upload",
        backend.device_upload_f32_tensor(&dense_k)?,
    )?;
    let dense_source_v = require_device_value(
        "DSA attention benchmark dense V upload",
        backend.device_upload_f32_tensor(&dense_v)?,
    )?;
    copy_logical_kv_tokens(
        backend,
        &dense_source_k,
        config.tokens,
        &dense_hot_k,
        capacity_tokens,
        0,
        config.tokens,
        config.batch,
        config.attention_heads,
        config.key_head_dim,
    )?;
    copy_logical_kv_tokens(
        backend,
        &dense_source_v,
        config.tokens,
        &dense_hot_v,
        capacity_tokens,
        0,
        config.tokens,
        config.batch,
        config.attention_heads,
        config.value_head_dim,
    )?;
    backend.device_flush()?;

    let dense_view = DevicePagedKvView {
        batch: config.batch,
        attention_heads: config.attention_heads,
        key_head_dim: config.key_head_dim,
        value_head_dim: config.value_head_dim,
        page_size: config.page_size,
        cached_tokens: config.tokens,
        capacity_tokens,
        k: dense_hot_k,
        v: dense_hot_v,
    };
    dense_view.validate()?;

    let dense_started_at = Instant::now();
    for _ in 0..config.layer_count {
        let _ = require_device_value(
            "DSA attention benchmark dense paged attention",
            backend.paged_decode_attention_resident_device(
                &q,
                &current_k,
                &current_v,
                &dense_view,
            )?,
        )?;
    }
    backend.device_flush()?;
    let dense_seconds = dense_started_at.elapsed().as_secs_f64();

    let selected_started_at = Instant::now();
    for _ in 0..config.layer_count {
        let _ = require_device_value(
            "DSA attention benchmark selected attention",
            backend.selected_decode_attention_device(
                &q,
                &selected_k,
                &selected_v,
                &current_k,
                &current_v,
                true,
            )?,
        )?;
    }
    backend.device_flush()?;
    let selected_seconds = selected_started_at.elapsed().as_secs_f64();

    let layer_count = config.layer_count as f64;
    Ok(DsaAttentionBenchmarkReport {
        layer_count: config.layer_count,
        tokens: config.tokens,
        selected_tokens: config.selected_tokens,
        dense_seconds,
        selected_seconds,
        dense_seconds_per_layer: dense_seconds / layer_count,
        selected_seconds_per_layer: selected_seconds / layer_count,
        selected_vs_dense_speedup: if selected_seconds > 0.0 {
            dense_seconds / selected_seconds
        } else {
            f64::INFINITY
        },
    })
}

pub fn benchmark_moe_multi_expert_decode<B: Backend>(
    backend: &B,
    config: MoeExpertBenchmarkConfig,
) -> Result<MoeExpertBenchmarkReport> {
    validate_moe_expert_benchmark_config(&config)?;
    let gate_weights = synthetic_q2_k_expert_weights(
        config.expert_count,
        config.intermediate_size,
        config.hidden_size,
        11,
    )?;
    let up_weights = synthetic_q2_k_expert_weights(
        config.expert_count,
        config.intermediate_size,
        config.hidden_size,
        29,
    )?;
    let down_weights = synthetic_q2_k_expert_weights(
        config.expert_count,
        config.hidden_size,
        config.intermediate_size,
        47,
    )?;
    let input = synthetic_moe_input(config.token_count, config.hidden_size)?;
    let input = require_device_value(
        "MoE benchmark input upload",
        backend.device_upload_f32_tensor(&input)?,
    )?;
    let token_indices = synthetic_moe_token_indices(config.token_count, config.assignment_count)?;
    let expert_ids = synthetic_moe_expert_ids(config.expert_count, config.assignment_count)?;

    let old_started_at = Instant::now();
    for _ in 0..config.iterations {
        for assignment in 0..config.assignment_count {
            let single_token = [token_indices[assignment]];
            let single_expert = [expert_ids[assignment]];
            let gated = require_device_value(
                "MoE benchmark single-assignment gate/up",
                backend.q2_k_multi_expert_gate_up_swiglu_device(
                    &gate_weights,
                    &up_weights,
                    &input,
                    &single_token,
                    &single_expert,
                    config.token_count,
                    config.hidden_size,
                    config.intermediate_size,
                )?,
            )?;
            let _ = require_device_value(
                "MoE benchmark single-assignment down",
                backend.q2_k_multi_expert_matvec_device(
                    &down_weights,
                    &gated,
                    &single_expert,
                    config.intermediate_size,
                    config.hidden_size,
                )?,
            )?;
        }
    }
    backend.device_flush()?;
    let old_single_assignment_seconds = old_started_at.elapsed().as_secs_f64();

    let multi_started_at = Instant::now();
    for _ in 0..config.iterations {
        let gated = require_device_value(
            "MoE benchmark multi-expert gate/up",
            backend.q2_k_multi_expert_gate_up_swiglu_device(
                &gate_weights,
                &up_weights,
                &input,
                &token_indices,
                &expert_ids,
                config.token_count,
                config.hidden_size,
                config.intermediate_size,
            )?,
        )?;
        let _ = require_device_value(
            "MoE benchmark multi-expert down",
            backend.q2_k_multi_expert_matvec_device(
                &down_weights,
                &gated,
                &expert_ids,
                config.intermediate_size,
                config.hidden_size,
            )?,
        )?;
    }
    backend.device_flush()?;
    let multi_expert_seconds = multi_started_at.elapsed().as_secs_f64();

    let iterations = config.iterations as f64;
    Ok(MoeExpertBenchmarkReport {
        iterations: config.iterations,
        token_count: config.token_count,
        assignment_count: config.assignment_count,
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        expert_count: config.expert_count,
        old_single_assignment_seconds,
        multi_expert_seconds,
        old_single_assignment_seconds_per_iteration: old_single_assignment_seconds / iterations,
        multi_expert_seconds_per_iteration: multi_expert_seconds / iterations,
        scheduling_speedup: if multi_expert_seconds > 0.0 {
            old_single_assignment_seconds / multi_expert_seconds
        } else {
            f64::INFINITY
        },
    })
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct GenerationReport {
    pub capabilities: BackendCapabilities,
    pub page_size: usize,
    pub prompt_token_count: usize,
    pub generated_token_ids: Vec<u32>,
    pub total_token_count: usize,
    pub max_new_tokens: usize,
    pub memory: GenerationMemoryEstimate,
    pub prefill: GenerationStepReport,
    pub decode: DecodeLoopReport,
    pub cached_tokens: usize,
    pub next_position: usize,
    pub expert_loads: ExpertLoadReport,
    pub stop_reason: String,
    pub limitations: Vec<String>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct GenerationMemoryEstimate {
    pub batch: usize,
    pub prompt_tokens: usize,
    pub requested_context_tokens: usize,
    pub prefill_attention_scores_shape: Shape,
    pub prefill_attention_scores_bytes: u64,
    pub max_decode_attention_scores_shape: Shape,
    pub max_decode_attention_scores_bytes: u64,
    pub per_layer_k_cache_shape: Shape,
    pub per_layer_v_cache_shape: Shape,
    pub mla_kv_cache_bytes: u64,
    pub max_prefill_attention_scores_bytes: u64,
    pub max_mla_kv_cache_bytes: u64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct GenerationStepReport {
    pub step_index: usize,
    pub input_token_count: usize,
    pub hidden_states_shape: Shape,
    pub layer_kv_cache_count: usize,
    pub layer_k_cache_shape: Shape,
    pub layer_v_cache_shape: Shape,
    pub cache_append: PagedCacheAppendReport,
    pub cached_tokens: usize,
    pub next_position: usize,
    pub next_decode_attention_scores_shape: Shape,
    pub logits_shape: Shape,
    pub sampled_token_id: u32,
    pub sampled_token_score: f32,
    pub model: GenerationModelStepReport,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct GenerationModelStepReport {
    pub dense_layer_count: usize,
    pub sparse_layer_count: usize,
    pub max_attention_past_tokens: usize,
    pub materialized_full_logits: bool,
    pub output_projection_chunk_count: usize,
    pub output_projection_source_payload_bytes_read: u64,
    pub output_projection_peak_decoded_f32_bytes: u64,
    pub expert_loads: StepExpertLoadReport,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct StepExpertLoadReport {
    pub loaded_expert_requests: usize,
    pub materialized_expert_bytes_loaded: u64,
    pub source_expert_bytes_loaded: u64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct DecodeLoopReport {
    pub step_count: usize,
    pub total_appended_tokens: usize,
    pub total_allocated_pages: usize,
    pub final_cached_tokens: usize,
    pub final_next_position: usize,
    pub max_attention_past_tokens: usize,
    pub materialized_full_logits: bool,
    pub total_output_projection_source_payload_bytes_read: u64,
    pub peak_output_projection_decoded_f32_bytes: u64,
    pub last_hidden_states_shape: Option<Shape>,
    pub last_logits_shape: Option<Shape>,
    pub last_next_decode_attention_scores_shape: Option<Shape>,
    pub last_sampled_token_id: Option<u32>,
    pub last_sampled_token_score: Option<f32>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct PagedCacheAppendReport {
    pub page_size: usize,
    pub layer_count: usize,
    pub start_position: usize,
    pub appended_tokens: usize,
    pub end_position_exclusive: usize,
    pub cached_tokens: usize,
    pub next_position: usize,
    pub allocated_pages: usize,
    pub page_count: usize,
    pub next_decode_attention_scores_shape: Shape,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct ExpertLoadReport {
    pub loaded_expert_requests: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub hit_rate: f32,
    pub materialized_expert_bytes_loaded: u64,
    pub source_expert_bytes_loaded: u64,
    pub limitations: Vec<String>,
}

pub fn run_generate_token_ids_with_stop_tokens<B: Backend>(
    model: &Model<'_>,
    config: &Config,
    backend: &B,
    prompt_token_ids: &[u32],
    max_new_tokens: Option<usize>,
    page_size: usize,
    stop_token_ids: &[u32],
) -> Result<Vec<u32>> {
    let mut generated_token_ids = Vec::new();
    run_generate_streaming_with_stop_tokens(
        model,
        config,
        backend,
        prompt_token_ids,
        max_new_tokens,
        page_size,
        stop_token_ids,
        |token_id| {
            generated_token_ids.push(token_id);
            Ok(())
        },
    )?;
    Ok(generated_token_ids)
}

pub fn run_generate_streaming_with_stop_tokens<B, F>(
    model: &Model<'_>,
    config: &Config,
    backend: &B,
    prompt_token_ids: &[u32],
    max_new_tokens: Option<usize>,
    page_size: usize,
    stop_token_ids: &[u32],
    on_token: F,
) -> Result<()>
where
    B: Backend,
    F: FnMut(u32) -> Result<()>,
{
    run_generate_streaming_with_options(
        model,
        config,
        backend,
        prompt_token_ids,
        max_new_tokens,
        page_size,
        stop_token_ids,
        GenerationOptions::default(),
        on_token,
    )
    .map(|_| ())
}

#[allow(clippy::too_many_arguments)]
pub fn run_generate_streaming_with_options<B, F>(
    model: &Model<'_>,
    config: &Config,
    backend: &B,
    prompt_token_ids: &[u32],
    max_new_tokens: Option<usize>,
    page_size: usize,
    stop_token_ids: &[u32],
    options: GenerationOptions,
    mut on_token: F,
) -> Result<StreamingGenerationReport>
where
    B: Backend,
    F: FnMut(u32) -> Result<()>,
{
    let generate_started_at = Instant::now();
    let use_streaming_prefill = backend.capabilities().custom_kernels;
    let use_device_kv = use_streaming_prefill;
    let mtp_available =
        config.index_share_for_mtp_iteration && use_device_kv && model.has_mtp_head();
    let prefill_strategy = if use_streaming_prefill {
        PrefillStrategy::Streaming
    } else {
        PrefillStrategy::Dense
    };
    let effective_max_new_tokens = validate_generate_request_with_prefill_strategy(
        config,
        prompt_token_ids,
        max_new_tokens,
        page_size,
        prefill_strategy,
    )?;
    let use_mtp = should_enable_mtp(mtp_available, effective_max_new_tokens);
    let mut mtp_metrics = MtpMetrics {
        enabled: use_mtp,
        ..MtpMetrics::default()
    };
    if options.profile_token_costs && use_mtp {
        return Err(Error::runtime(
            "per-token cost profiling cannot be combined with speculative MTP because one verification step can emit multiple tokens",
        ));
    }
    let _token_cost_profile_scope = if options.profile_token_costs {
        model::enable_token_cost_profile()?;
        Some(TokenCostProfileScope)
    } else {
        None
    };
    let prefill_cost_snapshot = options
        .profile_token_costs
        .then(|| TokenCostSnapshot::capture(backend, &None))
        .transpose()?;
    let mut cache_budget_controller = options
        .dynamic_cache_budget
        .map(|spec| CacheBudgetController::new(spec, prompt_token_ids.len()))
        .transpose()?;
    if let Some(controller) = cache_budget_controller.as_ref() {
        backend
            .configure_expert_cache_slots_per_layer(controller.current.expert_slots_per_layer)?;
    }
    let initial_hot_kv_budget = cache_budget_controller
        .as_ref()
        .map(|controller| controller.current.hot_kv_budget_bytes)
        .or(options.hot_kv_cache_budget_bytes);
    log_memory_snapshot(
        "generate.start",
        Some(0),
        backend,
        RuntimeKvMemoryBytes::default(),
    );

    let prefill_started_at = Instant::now();
    set_layer_profile_context(0, "prefill.seed")?;
    let seed_input_ids = if use_streaming_prefill {
        &prompt_token_ids[..1]
    } else {
        prompt_token_ids
    };
    let mut last_main_hidden_device = None;
    let mut device_seed_layer_kv_cache = None;
    let (mut prefill_token_id, prefill_layer_kv_cache) = if use_device_kv {
        let output = model
            .prefill_seed_device(config, seed_input_ids, backend)?
            .ok_or_else(|| Error::backend("native Metal seed prefill requires device path"))?;
        last_main_hidden_device = Some(output.hidden_states);
        device_seed_layer_kv_cache = Some(output.layer_kv_cache);
        (output.token_id, Vec::new())
    } else {
        let output = model.prefill_next_token(config, seed_input_ids, backend)?;
        (output.token_id, output.layer_kv_cache)
    };
    record_q2_runtime_stage(0, None, "prefill.model", prefill_started_at.elapsed());
    log_memory_snapshot(
        "prefill.after_model",
        Some(0),
        backend,
        RuntimeKvMemoryBytes::default(),
    );
    let cache_started_at = Instant::now();
    let mut device_kv_cache = if use_device_kv {
        let seed_layer_kv_cache = device_seed_layer_kv_cache
            .as_ref()
            .ok_or_else(|| Error::runtime("device seed K/V cache was not initialized"))?;
        Some(DevicePagedRuntimeCache::new_from_device_seed(
            model.max_context(),
            page_size,
            seed_layer_kv_cache,
            backend,
            initial_hot_kv_budget,
        )?)
    } else {
        None
    };
    let mut host_kv_cache = if use_device_kv {
        None
    } else {
        let mut cache =
            PagedRuntimeCache::new(model.max_context(), page_size, &prefill_layer_kv_cache)?;
        cache.append_prefill(&prefill_layer_kv_cache)?;
        Some(cache)
    };
    drop(prefill_layer_kv_cache);
    record_q2_runtime_stage(0, None, "prefill.cache_init", cache_started_at.elapsed());
    log_memory_snapshot(
        "prefill.after_cache_append",
        Some(0),
        backend,
        runtime_kv_memory(&device_kv_cache, &host_kv_cache),
    );

    let mut mtp_device_cache = None;

    if use_streaming_prefill && prompt_token_ids.len() > 1 {
        for (prefill_index, prompt_token_id) in prompt_token_ids.iter().copied().enumerate().skip(1)
        {
            if let Some(cache) = device_kv_cache.as_mut() {
                if let Some(controller) = cache_budget_controller.as_mut() {
                    controller.maybe_rebalance(
                        cache.cached_tokens()?.saturating_add(1),
                        backend,
                        cache,
                    )?;
                }
                cache.set_profile_step_index(prefill_index);
            }
            log_memory_snapshot(
                "prefill.decode.before_model",
                Some(prefill_index),
                backend,
                runtime_kv_memory(&device_kv_cache, &host_kv_cache),
            );
            if use_mtp {
                if let Some(hidden) = last_main_hidden_device.as_ref() {
                    let _ = run_mtp_device_draft(
                        model,
                        config,
                        backend,
                        hidden,
                        &[prompt_token_id],
                        &mut mtp_device_cache,
                        page_size,
                        options.hot_kv_cache_budget_bytes,
                        None,
                        None,
                        true,
                        true,
                    )?;
                }
                let output = run_cached_token_step(
                    model,
                    config,
                    backend,
                    &[prompt_token_id],
                    &mut device_kv_cache,
                    &mut host_kv_cache,
                    prefill_index,
                    "prefill.decode",
                    "prefill.decode.model",
                    "prefill.decode.after_model",
                    "prefill.decode.cache_append",
                )?;
                prefill_token_id = output.token_id;
                last_main_hidden_device = output.device_hidden_states;
            } else {
                prefill_token_id = run_cached_token_step(
                    model,
                    config,
                    backend,
                    &[prompt_token_id],
                    &mut device_kv_cache,
                    &mut host_kv_cache,
                    prefill_index,
                    "prefill.decode",
                    "prefill.decode.model",
                    "prefill.decode.after_model",
                    "prefill.decode.cache_append",
                )?
                .token_id;
            }
            log_memory_snapshot(
                "prefill.decode.after_cache_append",
                Some(prefill_index),
                backend,
                runtime_kv_memory(&device_kv_cache, &host_kv_cache),
            );
        }
    }

    let mut generated_token_count = 1_usize;
    if let Some(snapshot) = prefill_cost_snapshot {
        log_token_cost(snapshot.finish(0, true, prefill_token_id, backend, &device_kv_cache)?);
    }
    on_token(prefill_token_id)?;
    if contains_stop_token(prefill_token_id, stop_token_ids) {
        log_memory_snapshot(
            "generate.stop_after_prefill",
            Some(0),
            backend,
            runtime_kv_memory(&device_kv_cache, &host_kv_cache),
        );
        record_q2_runtime_stage(0, None, "generate.total", generate_started_at.elapsed());
        return Ok(streaming_generation_report(
            &device_kv_cache,
            &host_kv_cache,
            cache_budget_controller.as_ref(),
            mtp_metrics,
        ));
    }

    let mut pending_mtp_drafts = if use_mtp {
        match last_main_hidden_device.as_ref() {
            Some(hidden) => build_pending_mtp_drafts(
                model,
                config,
                backend,
                hidden,
                &[prefill_token_id],
                &mut mtp_device_cache,
                page_size,
                options.hot_kv_cache_budget_bytes,
                effective_max_new_tokens.saturating_sub(generated_token_count),
            )?,
            None => None,
        }
    } else {
        None
    };

    let mut next_input_token_id = prefill_token_id;
    for step_index in 1..effective_max_new_tokens {
        let token_cost_snapshot = options
            .profile_token_costs
            .then(|| TokenCostSnapshot::capture(backend, &device_kv_cache))
            .transpose()?;
        if max_new_tokens.is_none() && prefill_strategy == PrefillStrategy::Dense {
            validate_reference_memory_bounds(config, prompt_token_ids.len())?;
        }
        if let Some(cache) = device_kv_cache.as_mut() {
            if let Some(controller) = cache_budget_controller.as_mut() {
                controller.maybe_rebalance(
                    cache.cached_tokens()?.saturating_add(1),
                    backend,
                    cache,
                )?;
            }
            cache.set_profile_step_index(step_index);
        }
        if let Some(cache) = host_kv_cache.as_mut() {
            cache.set_profile_step_index(step_index);
        }
        log_memory_snapshot(
            "decode.before_model",
            Some(step_index),
            backend,
            runtime_kv_memory(&device_kv_cache, &host_kv_cache),
        );
        if use_mtp {
            let cache = device_kv_cache
                .as_mut()
                .ok_or_else(|| Error::runtime("device KV cache is not initialized"))?;
            let remaining_tokens = effective_max_new_tokens.saturating_sub(generated_token_count);
            let mut input_ids = vec![next_input_token_id];
            if let Some(pending) = pending_mtp_drafts.take() {
                let draft_limit = mtp_draft_count(remaining_tokens);
                input_ids.extend(pending.token_ids.into_iter().take(draft_limit));
            }
            let output = run_cached_device_token_sequence_without_append(
                model,
                config,
                backend,
                &input_ids,
                cache,
                step_index,
                "decode",
                "decode.model",
            )?;
            validate_exact_shape(
                "MTP target output row count",
                &[output.token_ids.len(), output.token_scores.len()],
                &[input_ids.len(), input_ids.len()],
            )?;
            let draft_count = input_ids.len() - 1;
            mtp_metrics.verification_passes = mtp_metrics.verification_passes.saturating_add(1);
            mtp_metrics.target_tokens = mtp_metrics
                .target_tokens
                .saturating_add(u64::try_from(input_ids.len()).unwrap_or(u64::MAX));
            mtp_metrics.draft_tokens = mtp_metrics
                .draft_tokens
                .saturating_add(u64::try_from(draft_count).unwrap_or(u64::MAX));
            let mut accepted_drafts = 0_usize;
            let mut accepted_eos = false;
            for draft_index in 0..draft_count {
                let draft = input_ids[draft_index + 1];
                let verified = output.token_ids[draft_index];
                let accepted = verified == draft;
                tracing::debug!(
                    target: "inferno::mtp",
                    step_index,
                    draft_index,
                    draft_token_id = draft,
                    verified_token_id = verified,
                    accepted,
                    "verified speculative MTP token"
                );
                if !accepted {
                    break;
                }
                accepted_drafts += 1;
                if contains_stop_token(draft, stop_token_ids) {
                    accepted_eos = true;
                    break;
                }
            }
            mtp_metrics.accepted_draft_tokens = mtp_metrics
                .accepted_draft_tokens
                .saturating_add(u64::try_from(accepted_drafts).unwrap_or(u64::MAX));
            let accepted_input_rows = 1 + accepted_drafts;
            let mut emitted = input_ids[1..accepted_input_rows].to_vec();
            if !accepted_eos {
                emitted.push(output.token_ids[accepted_drafts]);
            }
            cache.append_decode_prefix(
                &output.layer_kv_cache,
                input_ids.len(),
                accepted_input_rows,
                backend,
            )?;

            let mut stopped = false;
            for token_id in emitted {
                generated_token_count = generated_token_count
                    .checked_add(1)
                    .ok_or_else(|| Error::runtime("generated token count overflow"))?;
                on_token(token_id)?;
                next_input_token_id = token_id;
                if contains_stop_token(token_id, stop_token_ids)
                    || generated_token_count >= effective_max_new_tokens
                {
                    stopped = true;
                    break;
                }
            }

            if stopped {
                log_memory_snapshot(
                    "decode.after_cache_append",
                    Some(step_index),
                    backend,
                    runtime_kv_memory(&device_kv_cache, &host_kv_cache),
                );
                break;
            }

            let accepted_hidden =
                slice_device_token_prefix(backend, &output.hidden_states, accepted_input_rows)?;
            let mut accepted_next_tokens = input_ids[1..accepted_input_rows].to_vec();
            accepted_next_tokens.push(next_input_token_id);
            pending_mtp_drafts = build_pending_mtp_drafts(
                model,
                config,
                backend,
                &accepted_hidden,
                &accepted_next_tokens,
                &mut mtp_device_cache,
                page_size,
                options.hot_kv_cache_budget_bytes,
                effective_max_new_tokens.saturating_sub(generated_token_count),
            )?;
            log_memory_snapshot(
                "decode.after_cache_append",
                Some(step_index),
                backend,
                runtime_kv_memory(&device_kv_cache, &host_kv_cache),
            );
            continue;
        }

        let token_id = run_cached_token_step(
            model,
            config,
            backend,
            &[next_input_token_id],
            &mut device_kv_cache,
            &mut host_kv_cache,
            step_index,
            "decode",
            "decode.model",
            "decode.after_model",
            "decode.cache_append",
        )?
        .token_id;
        log_memory_snapshot(
            "decode.after_cache_append",
            Some(step_index),
            backend,
            runtime_kv_memory(&device_kv_cache, &host_kv_cache),
        );
        if let Some(snapshot) = token_cost_snapshot {
            log_token_cost(snapshot.finish(
                step_index,
                false,
                token_id,
                backend,
                &device_kv_cache,
            )?);
        }

        generated_token_count = generated_token_count
            .checked_add(1)
            .ok_or_else(|| Error::runtime("generated token count overflow"))?;
        on_token(token_id)?;
        if contains_stop_token(token_id, stop_token_ids) {
            break;
        }
        next_input_token_id = token_id;
    }

    log_memory_snapshot(
        "generate.end",
        None,
        backend,
        runtime_kv_memory(&device_kv_cache, &host_kv_cache),
    );
    record_q2_runtime_stage(0, None, "generate.total", generate_started_at.elapsed());
    Ok(streaming_generation_report(
        &device_kv_cache,
        &host_kv_cache,
        cache_budget_controller.as_ref(),
        mtp_metrics,
    ))
}

#[allow(clippy::too_many_arguments)]
struct CachedTokenStepOutput {
    token_id: u32,
    device_hidden_states: Option<backend::DeviceValue>,
}

#[allow(clippy::too_many_arguments)]
fn run_cached_token_step<B: Backend>(
    model: &Model<'_>,
    config: &Config,
    backend: &B,
    input_token_ids: &[u32],
    device_kv_cache: &mut Option<DevicePagedRuntimeCache>,
    host_kv_cache: &mut Option<PagedRuntimeCache>,
    step_index: usize,
    profile_context: &'static str,
    model_stage: &'static str,
    after_model_memory_stage: &'static str,
    cache_stage: &'static str,
) -> Result<CachedTokenStepOutput> {
    let decode_started_at = Instant::now();
    set_layer_profile_context(step_index, profile_context)?;
    if let Some(cache) = device_kv_cache.as_mut() {
        let cache_cell = RefCell::new(cache);
        let decode_output = model
            .decode_next_token_with_device_paged_kv_provider(
                config,
                input_token_ids,
                backend,
                |layer_index| {
                    cache_cell
                        .borrow_mut()
                        .device_paged_kv_for_layer(layer_index, backend)
                },
                |layer_index, token_indices| {
                    cache_cell.borrow_mut().selected_device_kv_for_layer(
                        layer_index,
                        token_indices,
                        backend,
                    )
                },
                |layer_index| {
                    cache_cell
                        .borrow_mut()
                        .dsa_index_keys_for_layer_device(layer_index, backend)
                },
            )?
            .ok_or_else(|| {
                Error::backend(
                    "native Metal cached token step requires the resident device KV path; fallback to host KV is disabled",
                )
            })?;
        record_q2_runtime_stage(step_index, None, model_stage, decode_started_at.elapsed());
        log_memory_snapshot(
            after_model_memory_stage,
            Some(step_index),
            backend,
            cache_cell.borrow().runtime_kv_memory(),
        );
        let cache_started_at = Instant::now();
        cache_cell
            .borrow_mut()
            .append_decode(&decode_output.layer_kv_cache, backend)?;
        record_q2_runtime_stage(step_index, None, cache_stage, cache_started_at.elapsed());
        Ok(CachedTokenStepOutput {
            token_id: decode_output.token_id,
            device_hidden_states: Some(decode_output.hidden_states),
        })
    } else {
        let cache = host_kv_cache
            .as_mut()
            .ok_or_else(|| Error::runtime("host KV cache is not initialized"))?;
        let cache_cell = RefCell::new(cache);
        let decode_output = model.decode_next_token_with_sparse_past_kv_provider(
            config,
            input_token_ids,
            backend,
            |layer_index| cache_cell.borrow().past_kv_for_layer(layer_index),
            |layer_index| {
                cache_cell
                    .borrow_mut()
                    .dsa_index_keys_for_layer(layer_index)
            },
        )?;
        record_q2_runtime_stage(step_index, None, model_stage, decode_started_at.elapsed());
        log_memory_snapshot(
            after_model_memory_stage,
            Some(step_index),
            backend,
            cache_cell.borrow().runtime_kv_memory(),
        );
        let cache_started_at = Instant::now();
        cache_cell
            .borrow_mut()
            .append_decode(&decode_output.layer_kv_cache)?;
        record_q2_runtime_stage(step_index, None, cache_stage, cache_started_at.elapsed());
        Ok(CachedTokenStepOutput {
            token_id: decode_output.token_id,
            device_hidden_states: None,
        })
    }
}

fn run_cached_device_token_sequence_without_append<B: Backend>(
    model: &Model<'_>,
    config: &Config,
    backend: &B,
    input_token_ids: &[u32],
    device_kv_cache: &mut DevicePagedRuntimeCache,
    step_index: usize,
    profile_context: &'static str,
    model_stage: &'static str,
) -> Result<ModelDeviceTokenSequenceOutput> {
    let decode_started_at = Instant::now();
    set_layer_profile_context(step_index, profile_context)?;
    let cache_cell = RefCell::new(device_kv_cache);
    let output = model
        .decode_token_sequence_with_device_paged_kv_provider(
            config,
            input_token_ids,
            backend,
            |layer_index| {
                cache_cell
                    .borrow_mut()
                    .device_paged_kv_for_layer(layer_index, backend)
            },
            |layer_index, token_indices| {
                cache_cell.borrow_mut().selected_device_kv_for_layer(
                    layer_index,
                    token_indices,
                    backend,
                )
            },
            |layer_index| {
                cache_cell
                    .borrow_mut()
                    .dsa_index_keys_for_layer_device(layer_index, backend)
            },
        )?
        .ok_or_else(|| {
            Error::backend("native Metal speculative verification requires the two-row device path")
        })?;
    record_q2_runtime_stage(step_index, None, model_stage, decode_started_at.elapsed());
    Ok(output)
}

struct PendingMtpDrafts {
    token_ids: Vec<u32>,
}

struct MtpDeviceDraftState {
    token_id: u32,
    recycle_hidden_states: backend::DeviceValue,
    shared_selection: Option<Vec<u32>>,
}

#[allow(clippy::too_many_arguments)]
fn run_mtp_device_draft<B: Backend>(
    model: &Model<'_>,
    config: &Config,
    backend: &B,
    main_hidden_states: &backend::DeviceValue,
    next_token_ids: &[u32],
    mtp_cache: &mut Option<DevicePagedRuntimeCache>,
    page_size: usize,
    hot_kv_cache_budget_bytes: Option<usize>,
    shared_selection: Option<&[u32]>,
    query_position: Option<usize>,
    include_current_kv: bool,
    commit_cache: bool,
) -> Result<Option<MtpDeviceDraftState>> {
    if !model.has_mtp_head() || next_token_ids.is_empty() {
        return Ok(None);
    }
    let draft = match mtp_cache.as_mut() {
        Some(cache) => {
            let cache_cell = RefCell::new(cache);
            let past_kv = cache_cell
                .borrow_mut()
                .device_paged_kv_for_layer(config.num_layers, backend)?;
            model.draft_next_token_with_mtp_device(
                config,
                main_hidden_states,
                next_token_ids,
                backend,
                past_kv.as_ref(),
                |layer_index, token_indices| {
                    cache_cell.borrow_mut().selected_device_kv_for_layer(
                        layer_index,
                        token_indices,
                        backend,
                    )
                },
                |layer_index| {
                    cache_cell
                        .borrow_mut()
                        .dsa_index_keys_for_layer_device(layer_index, backend)
                },
                shared_selection,
                query_position,
                include_current_kv,
            )?
        }
        None => model.draft_next_token_with_mtp_device(
            config,
            main_hidden_states,
            next_token_ids,
            backend,
            None,
            |_layer_index, _token_indices| Ok(None),
            |_layer_index| Ok(None),
            shared_selection,
            query_position,
            include_current_kv,
        )?,
    };
    let Some(draft) = draft else {
        return Ok(None);
    };
    if commit_cache {
        match mtp_cache.as_mut() {
            Some(cache) => cache.append_decode_prefix(
                &[draft.layer_kv_cache],
                next_token_ids.len(),
                next_token_ids.len(),
                backend,
            )?,
            None => {
                validate_exact_shape(
                    "device MTP initial token count",
                    &[next_token_ids.len()],
                    &[1],
                )?;
                *mtp_cache = Some(DevicePagedRuntimeCache::new_from_device_seed(
                    model.max_context(),
                    page_size,
                    &[draft.layer_kv_cache],
                    backend,
                    hot_kv_cache_budget_bytes,
                )?);
            }
        }
    }
    Ok(Some(MtpDeviceDraftState {
        token_id: draft.token_id,
        recycle_hidden_states: draft.recycle_hidden_states,
        shared_selection: draft.shared_selection,
    }))
}

#[allow(clippy::too_many_arguments)]
fn build_pending_mtp_drafts<B: Backend>(
    model: &Model<'_>,
    config: &Config,
    backend: &B,
    main_hidden_states: &backend::DeviceValue,
    next_token_ids: &[u32],
    mtp_cache: &mut Option<DevicePagedRuntimeCache>,
    page_size: usize,
    hot_kv_cache_budget_bytes: Option<usize>,
    remaining_tokens: usize,
) -> Result<Option<PendingMtpDrafts>> {
    let draft_count = mtp_draft_count(remaining_tokens);
    if draft_count == 0 {
        return Ok(None);
    }
    let Some(first) = run_mtp_device_draft(
        model,
        config,
        backend,
        main_hidden_states,
        next_token_ids,
        mtp_cache,
        page_size,
        hot_kv_cache_budget_bytes,
        None,
        None,
        true,
        true,
    )?
    else {
        return Ok(None);
    };
    let mut token_ids = vec![first.token_id];
    if draft_count > 1 {
        let shared_selection = first.shared_selection.ok_or_else(|| {
            Error::runtime("MTP IndexShare step 0 did not produce reusable token indices")
        })?;
        let fixed_cache_tokens = mtp_cache
            .as_ref()
            .ok_or_else(|| Error::runtime("MTP KVShare cache was not initialized"))?
            .cached_tokens()?;
        let mut draft_hidden = require_device_value(
            "MTP next-step hidden state selection",
            backend.select_last_token_device(&first.recycle_hidden_states)?,
        )?;
        let mut draft_token_id = first.token_id;

        for draft_index in 1..draft_count {
            let query_position = fixed_cache_tokens
                .checked_add(draft_index - 1)
                .ok_or_else(|| Error::runtime("MTP KVShare query position overflow"))?;
            let next = run_mtp_device_draft(
                model,
                config,
                backend,
                &draft_hidden,
                &[draft_token_id],
                mtp_cache,
                page_size,
                hot_kv_cache_budget_bytes,
                Some(&shared_selection),
                Some(query_position),
                false,
                false,
            )?
            .ok_or_else(|| Error::runtime("MTP KVShare draft step produced no output"))?;
            token_ids.push(next.token_id);
            draft_token_id = next.token_id;
            draft_hidden = require_device_value(
                "MTP next-step hidden state selection",
                backend.select_last_token_device(&next.recycle_hidden_states)?,
            )?;
        }
    }

    Ok(Some(PendingMtpDrafts { token_ids }))
}

fn should_enable_mtp(mtp_available: bool, effective_max_new_tokens: usize) -> bool {
    // The prefill pass already emits the first generated token. MTP only helps
    // when decode has room to verify a draft and emit at least one more token.
    mtp_available && effective_max_new_tokens > 2
}

fn mtp_draft_count(remaining_tokens: usize) -> usize {
    remaining_tokens.saturating_sub(1).min(MTP_DRAFTS_PER_STEP)
}

#[cfg(test)]
fn run_generate_with_stop_tokens<B: Backend>(
    model: &Model<'_>,
    config: &Config,
    backend: &B,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    page_size: usize,
    stop_token_ids: &[u32],
) -> Result<GenerationReport> {
    let effective_max_new_tokens =
        validate_generate_request(config, prompt_token_ids, Some(max_new_tokens), page_size)?;
    let memory = estimate_generate_memory(config, prompt_token_ids.len(), max_new_tokens)?;

    let prefill_output = model.prefill_greedy(config, prompt_token_ids, backend)?;
    let mut kv_cache = PagedRuntimeCache::new(
        model.max_context(),
        page_size,
        &prefill_output.layer_kv_cache,
    )?;
    let prefill_cache_append = kv_cache.append_prefill_report(&prefill_output.layer_kv_cache)?;
    let prefill_step = generation_step(
        0,
        prompt_token_ids.len(),
        prefill_output,
        prefill_cache_append,
    )?;

    let mut generated_token_ids = vec![prefill_step.sampled_token_id];
    let mut next_input_ids = [prefill_step.sampled_token_id];
    let mut decode = empty_decode_loop_report();
    let mut expert_loads = empty_expert_load_report();
    accumulate_expert_load_step(&mut expert_loads, &prefill_step);
    let mut stop_reason = if contains_stop_token(prefill_step.sampled_token_id, stop_token_ids) {
        "stop token reached".to_string()
    } else {
        "max_new_tokens reached".to_string()
    };

    for step_index in 1..effective_max_new_tokens {
        if stop_reason == "stop token reached" {
            break;
        }

        let decode_output = model.decode_step_greedy_with_past_kv_provider(
            config,
            &next_input_ids,
            backend,
            |layer_index| kv_cache.past_kv_for_layer(layer_index),
        )?;
        let decode_cache_append = kv_cache.append_decode_report(&decode_output.layer_kv_cache)?;
        let decode_step = generation_step(
            step_index,
            next_input_ids.len(),
            decode_output,
            decode_cache_append,
        )?;

        next_input_ids = [decode_step.sampled_token_id];
        generated_token_ids.push(decode_step.sampled_token_id);
        if contains_stop_token(decode_step.sampled_token_id, stop_token_ids) {
            stop_reason = "stop token reached".to_string();
        }
        accumulate_expert_load_step(&mut expert_loads, &decode_step);
        accumulate_decode_loop_report(&mut decode, &decode_step);
    }
    finalize_expert_load_report(&mut expert_loads);

    let total_token_count = prompt_token_ids
        .len()
        .checked_add(generated_token_ids.len())
        .ok_or_else(|| Error::runtime("generated token count overflow"))?;

    Ok(GenerationReport {
        capabilities: backend.capabilities(),
        page_size,
        prompt_token_count: prompt_token_ids.len(),
        generated_token_ids,
        total_token_count,
        max_new_tokens,
        memory,
        prefill: prefill_step,
        decode,
        cached_tokens: kv_cache.cached_tokens(),
        next_position: kv_cache.next_position(),
        expert_loads,
        stop_reason,
        limitations: vec![
            "GLM-5.2 generation runs the Q2 GGUF embedding, dense prefix, sparse MoE blocks, final norm, and lm_head".to_string(),
            "paged K/V cache is the only runtime cache mode".to_string(),
            "native Metal streaming generation stores K/V in the SSD cold tier and streams one layer at a time into a reusable Metal hot window".to_string(),
            "native streaming prefill avoids dense prompt attention; sparse layers use DSA top-k decode selection when index keys are available".to_string(),
            "sampling is greedy with tokenizer-derived stop-token support".to_string(),
        ],
    })
}

#[cfg(test)]
fn validate_generate_request(
    config: &Config,
    prompt_token_ids: &[u32],
    max_new_tokens: Option<usize>,
    page_size: usize,
) -> Result<usize> {
    validate_generate_request_with_prefill_strategy(
        config,
        prompt_token_ids,
        max_new_tokens,
        page_size,
        PrefillStrategy::Dense,
    )
}

fn validate_generate_request_with_prefill_strategy(
    config: &Config,
    prompt_token_ids: &[u32],
    max_new_tokens: Option<usize>,
    page_size: usize,
    prefill_strategy: PrefillStrategy,
) -> Result<usize> {
    if prompt_token_ids.is_empty() {
        return Err(Error::runtime(
            "GLM generation requires at least one prompt token",
        ));
    }
    if max_new_tokens == Some(0) {
        return Err(Error::runtime(
            "max_new_tokens must be positive when provided",
        ));
    }
    if page_size == 0 {
        return Err(Error::cache("paged KV cache page_size must be positive"));
    }
    if page_size > config.max_context {
        return Err(Error::cache(format!(
            "paged KV cache page_size {page_size} exceeds max_context {}",
            config.max_context
        )));
    }
    let effective_max_new_tokens =
        effective_max_new_tokens(config, prompt_token_ids.len(), max_new_tokens)?;
    let requested_context_tokens = prompt_token_ids
        .len()
        .checked_add(effective_max_new_tokens)
        .ok_or_else(|| Error::runtime("requested context token count overflow"))?;
    if requested_context_tokens > config.max_context {
        return Err(Error::runtime(format!(
            "requested context tokens {requested_context_tokens} exceed max_context {}",
            config.max_context
        )));
    }
    if let Some(token_id) = prompt_token_ids
        .iter()
        .copied()
        .find(|token_id| *token_id as usize >= config.vocab_size)
    {
        return Err(Error::tokenizer(format!(
            "prompt token id {token_id} is outside config vocab_size {}",
            config.vocab_size
        )));
    }
    if prefill_strategy == PrefillStrategy::Dense {
        validate_reference_memory_bounds(config, prompt_token_ids.len())?;
    }
    Ok(effective_max_new_tokens)
}

fn effective_max_new_tokens(
    config: &Config,
    prompt_token_count: usize,
    max_new_tokens: Option<usize>,
) -> Result<usize> {
    let effective = match max_new_tokens {
        Some(max_new_tokens) => max_new_tokens,
        None => config
            .max_context
            .checked_sub(prompt_token_count)
            .ok_or_else(|| {
                Error::runtime(format!(
                    "prompt token count {prompt_token_count} exceeds max_context {}",
                    config.max_context
                ))
            })?,
    };

    if effective == 0 {
        return Err(Error::runtime(format!(
            "prompt token count {prompt_token_count} leaves no room for generation within max_context {}",
            config.max_context
        )));
    }

    Ok(effective)
}

#[cfg(test)]
fn estimate_generate_memory(
    config: &Config,
    prompt_tokens: usize,
    max_new_tokens: usize,
) -> Result<GenerationMemoryEstimate> {
    let requested_context_tokens = prompt_tokens
        .checked_add(max_new_tokens)
        .ok_or_else(|| Error::runtime("requested context token count overflow"))?;
    let batch = 1;
    let prefill_attention_scores_shape = Shape::new(vec![
        batch,
        config.attention_heads,
        prompt_tokens,
        prompt_tokens,
    ]);
    let prefill_attention_scores_bytes = checked_f32_bytes(
        "prefill attention scores",
        &[batch, config.attention_heads, prompt_tokens, prompt_tokens],
    )?;
    let max_decode_attention_scores_shape = Shape::new(vec![
        batch,
        config.attention_heads,
        1,
        requested_context_tokens,
    ]);
    let max_decode_attention_scores_bytes = checked_f32_bytes(
        "decode attention scores",
        &[batch, config.attention_heads, 1, requested_context_tokens],
    )?;
    let per_layer_k_cache_shape = Shape::new(vec![
        batch,
        1,
        requested_context_tokens,
        config.kv_lora_rank,
    ]);
    let per_layer_v_cache_shape =
        Shape::new(vec![batch, 1, requested_context_tokens, config.qk_rope_dim]);
    let mla_kv_cache_bytes = checked_element_bytes(
        "MLA latent KV cache",
        &[
            config.num_layers,
            batch,
            1,
            requested_context_tokens,
            config
                .kv_lora_rank
                .checked_add(config.qk_rope_dim)
                .ok_or_else(|| Error::runtime("MLA KV dimension overflow"))?,
        ],
        F16_BYTES,
    )?;

    Ok(GenerationMemoryEstimate {
        batch,
        prompt_tokens,
        requested_context_tokens,
        prefill_attention_scores_shape,
        prefill_attention_scores_bytes,
        max_decode_attention_scores_shape,
        max_decode_attention_scores_bytes,
        per_layer_k_cache_shape,
        per_layer_v_cache_shape,
        mla_kv_cache_bytes,
        max_prefill_attention_scores_bytes: MAX_REFERENCE_PREFILL_ATTENTION_SCORE_BYTES,
        max_mla_kv_cache_bytes: MAX_REFERENCE_MLA_KV_CACHE_BYTES,
    })
}

fn validate_reference_memory_bounds(config: &Config, prompt_tokens: usize) -> Result<()> {
    let batch = 1;

    let prefill_attention_scores_bytes = checked_f32_bytes(
        "prefill attention scores",
        &[batch, config.attention_heads, prompt_tokens, prompt_tokens],
    )?;
    if prefill_attention_scores_bytes > MAX_REFERENCE_PREFILL_ATTENTION_SCORE_BYTES {
        let shape = Shape::new(vec![
            batch,
            config.attention_heads,
            prompt_tokens,
            prompt_tokens,
        ]);
        return Err(Error::runtime(format!(
            "current dense prefill attention would allocate {prefill_attention_scores_bytes} bytes for attention scores {shape}; limit is {MAX_REFERENCE_PREFILL_ATTENTION_SCORE_BYTES} bytes; use the native streaming prefill path for long prompts"
        )));
    }

    Ok(())
}

fn checked_f32_bytes(context: &str, dims: &[usize]) -> Result<u64> {
    checked_element_bytes(context, dims, F32_BYTES)
}

fn checked_element_bytes(context: &str, dims: &[usize], element_bytes: u64) -> Result<u64> {
    let mut elements = 1_u128;
    for dim in dims {
        elements = elements
            .checked_mul(*dim as u128)
            .ok_or_else(|| Error::runtime(format!("{context} element count overflow")))?;
    }
    let bytes = elements
        .checked_mul(element_bytes as u128)
        .ok_or_else(|| Error::runtime(format!("{context} byte count overflow")))?;
    u64::try_from(bytes)
        .map_err(|_| Error::runtime(format!("{context} byte count does not fit u64")))
}

#[cfg(test)]
fn generation_step(
    step_index: usize,
    input_token_count: usize,
    model_output: ModelGreedyOutput,
    cache_append: PagedCacheAppendReport,
) -> Result<GenerationStepReport> {
    let logits_shape = model_output.report.greedy.logits_shape.clone();
    let model = compact_model_step_report(&model_output);
    let (layer_kv_cache_count, layer_k_cache_shape, layer_v_cache_shape) =
        compact_layer_kv_cache_shapes(&model_output.layer_kv_cache)?;

    Ok(GenerationStepReport {
        step_index,
        input_token_count,
        hidden_states_shape: Shape::new(model_output.hidden_states.dims().to_vec()),
        layer_kv_cache_count,
        layer_k_cache_shape,
        layer_v_cache_shape,
        cached_tokens: cache_append.cached_tokens,
        next_position: cache_append.next_position,
        next_decode_attention_scores_shape: cache_append.next_decode_attention_scores_shape.clone(),
        cache_append,
        logits_shape,
        sampled_token_id: model_output.token_id,
        sampled_token_score: model_output.token_score,
        model,
    })
}

#[cfg(test)]
fn compact_layer_kv_cache_shapes(
    layer_kv_cache: &[LayerKvCacheTensors],
) -> Result<(usize, Shape, Shape)> {
    let first = layer_kv_cache
        .first()
        .ok_or_else(|| Error::runtime("model step produced no layer K/V tensors"))?;
    let k_shape = Shape::new(first.cache_k.dims().to_vec());
    let v_shape = Shape::new(first.cache_v.dims().to_vec());

    for entry in layer_kv_cache.iter().skip(1) {
        validate_exact_shape(
            format!("step_layer_{}_k_cache_shape", entry.layer_index),
            entry.cache_k.dims(),
            k_shape.dims(),
        )?;
        validate_exact_shape(
            format!("step_layer_{}_v_cache_shape", entry.layer_index),
            entry.cache_v.dims(),
            v_shape.dims(),
        )?;
    }

    Ok((layer_kv_cache.len(), k_shape, v_shape))
}

#[cfg(test)]
fn compact_model_step_report(model_output: &ModelGreedyOutput) -> GenerationModelStepReport {
    let layer_stack = &model_output.report.hidden.layer_stack;

    GenerationModelStepReport {
        dense_layer_count: layer_stack.dense_block_count,
        sparse_layer_count: layer_stack.sparse_block_count,
        max_attention_past_tokens: layer_stack.max_attention_past_tokens,
        materialized_full_logits: model_output.report.greedy.materialized_full_logits,
        output_projection_chunk_count: model_output.report.greedy.output_projection_chunk_count,
        output_projection_source_payload_bytes_read: model_output
            .report
            .greedy
            .output_projection_source_payload_bytes_read,
        output_projection_peak_decoded_f32_bytes: model_output
            .report
            .greedy
            .output_projection_peak_decoded_f32_bytes,
        expert_loads: StepExpertLoadReport {
            loaded_expert_requests: layer_stack.loaded_expert_requests,
            materialized_expert_bytes_loaded: layer_stack.routed_peak_decoded_f32_bytes,
            source_expert_bytes_loaded: layer_stack.routed_source_payload_bytes_read,
        },
    }
}

fn runtime_kv_memory(
    device: &Option<DevicePagedRuntimeCache>,
    host: &Option<PagedRuntimeCache>,
) -> RuntimeKvMemoryBytes {
    if let Some(device) = device.as_ref() {
        return device.runtime_kv_memory();
    }
    if let Some(host) = host.as_ref() {
        return host.runtime_kv_memory();
    }
    RuntimeKvMemoryBytes::default()
}

fn streaming_generation_report(
    device: &Option<DevicePagedRuntimeCache>,
    host: &Option<PagedRuntimeCache>,
    budget: Option<&CacheBudgetController>,
    mtp: MtpMetrics,
) -> StreamingGenerationReport {
    if let Some(device) = device.as_ref() {
        return StreamingGenerationReport {
            kv_cache: device.cache_metrics(),
            mtp,
            cache_budget: budget.map(CacheBudgetController::report),
        };
    }
    let memory = runtime_kv_memory(device, host);
    StreamingGenerationReport {
        kv_cache: KvCacheMetrics {
            hot_bytes: memory.hot_bytes.unwrap_or(0),
            cold_bytes: memory.cold_bytes.unwrap_or(0),
            ..KvCacheMetrics::default()
        },
        mtp,
        cache_budget: budget.map(CacheBudgetController::report),
    }
}

struct PagedRuntimeCache {
    storage: LayeredPagedKvCache,
    dsa_index: Option<LayeredDsaIndexBlockStore>,
    profile_step_index: usize,
}

impl PagedRuntimeCache {
    fn new(
        max_context: usize,
        page_size: usize,
        layer_kv_cache: &[LayerKvCacheTensors],
    ) -> Result<Self> {
        let spec = cache_spec(max_context, page_size, layer_kv_cache)?;
        let dsa_spec = infer_dsa_index_spec(layer_kv_cache, page_size)?;
        let dsa_index = match dsa_spec {
            Some(spec) => Some(LayeredDsaIndexBlockStore::create(
                unique_dsa_index_path(),
                spec,
            )?),
            None => None,
        };
        Ok(Self {
            storage: LayeredPagedKvCache::new(spec)?,
            dsa_index,
            profile_step_index: 0,
        })
    }

    fn set_profile_step_index(&mut self, step_index: usize) {
        self.profile_step_index = step_index;
    }

    #[cfg(test)]
    fn cached_tokens(&self) -> usize {
        self.storage.cached_tokens()
    }

    #[cfg(test)]
    fn next_position(&self) -> usize {
        self.storage.next_position()
    }

    #[cfg(test)]
    fn storage_kind(&self) -> &'static str {
        "memory"
    }

    fn runtime_kv_memory(&self) -> RuntimeKvMemoryBytes {
        let spec = self.storage.spec();
        let hot_bytes = spec
            .batch
            .checked_mul(spec.attention_heads)
            .and_then(|value| value.checked_mul(spec.page_size))
            .and_then(|value| {
                spec.key_head_dim
                    .checked_add(spec.value_head_dim)
                    .and_then(|head_dim| value.checked_mul(head_dim))
            })
            .and_then(|values_per_page| values_per_page.checked_mul(self.storage.page_count()))
            .and_then(|values| values.checked_mul(F32_BYTES as usize))
            .and_then(|bytes| u64::try_from(bytes).ok());
        RuntimeKvMemoryBytes {
            hot_bytes,
            cold_bytes: None,
        }
    }

    fn append_prefill(&mut self, layer_kv_cache: &[LayerKvCacheTensors]) -> Result<()> {
        let appends = layer_appends(layer_kv_cache);
        self.storage.append_prefill(&appends)?;
        self.append_dsa_prefill(layer_kv_cache)?;
        Ok(())
    }

    fn append_decode(&mut self, layer_kv_cache: &[LayerKvCacheTensors]) -> Result<()> {
        let append_position = self.storage.cached_tokens();
        let appends = layer_appends(layer_kv_cache);
        self.storage.append_decode(&appends)?;
        self.append_dsa_decode(layer_kv_cache, append_position)?;
        Ok(())
    }

    #[cfg(test)]
    fn append_prefill_report(
        &mut self,
        layer_kv_cache: &[LayerKvCacheTensors],
    ) -> Result<PagedCacheAppendReport> {
        let appends = layer_appends(layer_kv_cache);
        let report = paged_append_report(
            self.storage.append_prefill(&appends)?,
            self.storage.spec().page_size,
        )?;
        self.append_dsa_prefill(layer_kv_cache)?;
        Ok(report)
    }

    #[cfg(test)]
    fn append_decode_report(
        &mut self,
        layer_kv_cache: &[LayerKvCacheTensors],
    ) -> Result<PagedCacheAppendReport> {
        let append_position = self.storage.cached_tokens();
        let appends = layer_appends(layer_kv_cache);
        let report = paged_append_report(
            self.storage.append_decode(&appends)?,
            self.storage.spec().page_size,
        )?;
        self.append_dsa_decode(layer_kv_cache, append_position)?;
        Ok(report)
    }

    fn past_kv_for_layer(&self, layer_index: usize) -> Result<Option<(F32Tensor, F32Tensor)>> {
        let started_at = Instant::now();
        let (keys, values) = self.storage.reconstruct_layer_kv(layer_index)?;
        record_q2_runtime_stage(
            self.profile_step_index,
            Some(layer_index),
            "decode.kv_reconstruct",
            started_at.elapsed(),
        );
        Ok(Some((keys, values)))
    }

    fn dsa_index_keys_for_layer(&mut self, layer_index: usize) -> Result<Option<F32Tensor>> {
        let Some(dsa_index) = self.dsa_index.as_mut() else {
            return Ok(None);
        };
        let started_at = Instant::now();
        let index_keys =
            dsa_index.read_layer_contiguous(layer_index, 0, self.storage.cached_tokens())?;
        record_q2_runtime_stage(
            self.profile_step_index,
            Some(layer_index),
            "decode.dsa_index_read",
            started_at.elapsed(),
        );
        Ok(Some(index_keys))
    }

    fn append_dsa_prefill(&mut self, layer_kv_cache: &[LayerKvCacheTensors]) -> Result<()> {
        let Some(store) = self.dsa_index.as_mut() else {
            return Ok(());
        };
        let dsa_appends = layer_kv_cache
            .iter()
            .filter_map(|entry| {
                entry
                    .index_key
                    .as_ref()
                    .map(|index_key| DsaIndexLayerAppend {
                        layer_index: entry.layer_index,
                        index_key,
                    })
            })
            .collect::<Vec<_>>();
        store.write_prefill(&dsa_appends)?;
        store.flush()
    }

    fn append_dsa_decode(
        &mut self,
        layer_kv_cache: &[LayerKvCacheTensors],
        append_position: usize,
    ) -> Result<()> {
        let Some(store) = self.dsa_index.as_mut() else {
            return Ok(());
        };
        for entry in layer_kv_cache {
            if let Some(index_key) = entry.index_key.as_ref() {
                validate_dsa_index_key_shape("decode", entry.layer_index, index_key)?;
                store.append_decode_layer(entry.layer_index, append_position, index_key)?;
            }
        }
        store.flush()
    }
}

impl Drop for PagedRuntimeCache {
    fn drop(&mut self) {
        if let Some(dsa_index) = self.dsa_index.as_ref() {
            let _ = std::fs::remove_file(dsa_index.path());
        }
    }
}

struct DevicePagedRuntimeCache {
    spec: LayeredPagedKvCacheSpec,
    cold: LayeredColdKvBlockStore,
    dsa_index: Option<LayeredDsaIndexBlockStore>,
    layers: Vec<ColdDeviceLayer>,
    hot_layers: Vec<HotDeviceLayerWindow>,
    streaming_hot: Option<HotDeviceLayerWindow>,
    prefetch: ColdKvPrefetcher,
    selected_prefetch: SelectedColdKvPrefetcher,
    cached_tokens: usize,
    profile_step_index: usize,
    policy: ColdKvRuntimePolicy,
    metrics: KvCacheMetrics,
}

struct ColdDeviceLayer {
    layer_index: usize,
    layer_kind: model::LayerKind,
}

struct HotDeviceLayerWindow {
    layer_index: Option<usize>,
    k: backend::DeviceValue,
    v: backend::DeviceValue,
    capacity_tokens: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ColdKvRuntimePolicy {
    block_tokens: usize,
    hot_all_layers_budget_bytes: usize,
}

const HOT_ALL_LAYERS_BUDGET_BYTES: usize = 512 * 1024 * 1024;

impl ColdKvRuntimePolicy {
    fn new(block_tokens: usize, hot_all_layers_budget_bytes: usize) -> Result<Self> {
        if block_tokens == 0 {
            return Err(Error::cache("cold KV block_tokens must be positive"));
        }
        if hot_all_layers_budget_bytes == 0 {
            return Err(Error::cache("hot KV cache budget must be positive"));
        }
        Ok(Self {
            block_tokens,
            hot_all_layers_budget_bytes,
        })
    }
}

struct ColdKvPrefetcher {
    pending: Option<ColdKvPrefetchJob>,
}

struct ColdKvPrefetchJob {
    layer_index: usize,
    token_start: usize,
    token_count: usize,
    handle: JoinHandle<Result<(F32Tensor, F32Tensor)>>,
}

struct SelectedColdKvPrefetcher {
    pending: Option<SelectedColdKvPrefetchJob>,
}

struct SelectedColdKvPrefetchJob {
    layer_index: usize,
    token_indices: Vec<u32>,
    handle: JoinHandle<Result<ColdKvSelectedQ8LayerRows>>,
}

impl DevicePagedRuntimeCache {
    fn new_from_device_seed<B: Backend>(
        max_context: usize,
        page_size: usize,
        layer_kv_cache: &[LayerDeviceKvCacheTensors],
        backend: &B,
        hot_kv_cache_budget_bytes: Option<usize>,
    ) -> Result<Self> {
        let spec = device_cache_spec(max_context, page_size, layer_kv_cache)?;
        let policy = ColdKvRuntimePolicy::new(
            page_size,
            hot_kv_cache_budget_bytes.unwrap_or(HOT_ALL_LAYERS_BUDGET_BYTES),
        )?;
        let cold_spec = ColdKvStoreSpec {
            batch: spec.batch,
            attention_heads: spec.attention_heads,
            key_head_dim: spec.key_head_dim,
            value_head_dim: spec.value_head_dim,
            block_tokens: policy.block_tokens,
            codec: ColdKvCodec::Q8Row,
        };
        let cold = LayeredColdKvBlockStore::create(unique_cold_kv_path(), cold_spec)?;
        let dsa_spec = infer_device_dsa_index_spec(layer_kv_cache, policy.block_tokens)?;
        let dsa_index = match dsa_spec {
            Some(spec) => Some(LayeredDsaIndexBlockStore::create(
                unique_dsa_index_path(),
                spec,
            )?),
            None => None,
        };
        let layers = layer_kv_cache
            .iter()
            .map(|entry| ColdDeviceLayer {
                layer_index: entry.layer_index,
                layer_kind: entry.layer_kind,
            })
            .collect::<Vec<_>>();
        let mut cache = Self {
            spec,
            cold,
            dsa_index,
            layers,
            hot_layers: Vec::new(),
            streaming_hot: None,
            prefetch: ColdKvPrefetcher::new(),
            selected_prefetch: SelectedColdKvPrefetcher::new(),
            cached_tokens: 0,
            profile_step_index: 0,
            policy,
            metrics: KvCacheMetrics::default(),
        };
        cache.initialize_hot_layers(backend, 1)?;
        cache.append_decode(layer_kv_cache, backend)?;
        Ok(cache)
    }

    fn set_profile_step_index(&mut self, step_index: usize) {
        self.profile_step_index = step_index;
    }

    fn runtime_kv_memory(&self) -> RuntimeKvMemoryBytes {
        let hot_bytes = self
            .hot_layers
            .iter()
            .chain(self.streaming_hot.iter())
            .try_fold(0_u64, |total, layer| {
                let layer_bytes = layer
                    .k
                    .byte_count()
                    .ok()?
                    .checked_add(layer.v.byte_count().ok()?)?;
                total.checked_add(u64::try_from(layer_bytes).ok()?)
            });
        let cold_bytes = self.cold.store().stored_bytes().ok().and_then(|kv_bytes| {
            let index_bytes = self
                .dsa_index
                .as_ref()
                .and_then(|store| store.stored_bytes().ok())
                .unwrap_or(0);
            kv_bytes.checked_add(index_bytes)
        });
        RuntimeKvMemoryBytes {
            hot_bytes,
            cold_bytes,
        }
    }

    fn cache_metrics(&self) -> KvCacheMetrics {
        let memory = self.runtime_kv_memory();
        let dsa_index_read_bytes = self
            .dsa_index
            .as_ref()
            .map(LayeredDsaIndexBlockStore::read_bytes)
            .unwrap_or(0);
        KvCacheMetrics {
            ssd_read_bytes: self
                .cold
                .store()
                .read_bytes()
                .saturating_add(dsa_index_read_bytes),
            hot_bytes: memory.hot_bytes.unwrap_or(0),
            cold_bytes: memory.cold_bytes.unwrap_or(0),
            cached_tokens: u64::try_from(self.cached_tokens).unwrap_or(u64::MAX),
            ..self.metrics
        }
    }

    fn set_hot_all_layers_budget(&mut self, budget_bytes: usize) -> Result<()> {
        if budget_bytes == 0 {
            return Err(Error::cache("hot KV cache budget must be positive"));
        }
        self.policy.hot_all_layers_budget_bytes = budget_bytes;
        if !self.hot_layers.is_empty() {
            let capacity_tokens = self
                .hot_layers
                .first()
                .map(|layer| layer.capacity_tokens)
                .unwrap_or(0);
            if self.hot_all_layers_bytes(capacity_tokens)? > budget_bytes {
                self.hot_layers.clear();
            }
        }
        Ok(())
    }

    fn append_decode<B: Backend>(
        &mut self,
        layer_kv_cache: &[LayerDeviceKvCacheTensors],
        backend: &B,
    ) -> Result<()> {
        self.append_decode_prefix(layer_kv_cache, 1, 1, backend)
    }

    fn append_decode_prefix<B: Backend>(
        &mut self,
        layer_kv_cache: &[LayerDeviceKvCacheTensors],
        source_tokens: usize,
        accepted_tokens: usize,
        backend: &B,
    ) -> Result<()> {
        let write_started = Instant::now();
        if source_tokens == 0 || accepted_tokens == 0 || accepted_tokens > source_tokens {
            return Err(Error::cache(format!(
                "device paged KV append requires 0 < accepted_tokens <= source_tokens, got accepted={accepted_tokens}, source={source_tokens}"
            )));
        }
        if layer_kv_cache.len() != self.layers.len() {
            return Err(Error::cache(format!(
                "device paged KV decode append expected {} layers, got {}",
                self.layers.len(),
                layer_kv_cache.len()
            )));
        }
        let append_position = self.cached_tokens()?;
        let needed_tokens = append_position
            .checked_add(accepted_tokens)
            .ok_or_else(|| Error::cache("device paged KV decode position overflow"))?;
        self.prepare_hot_layers_for_append(backend, needed_tokens)?;
        let batch = self.spec.batch;
        let attention_heads = self.spec.attention_heads;
        let key_head_dim = self.spec.key_head_dim;
        let value_head_dim = self.spec.value_head_dim;

        for (layer, append) in self.layers.iter().zip(layer_kv_cache) {
            if layer.layer_index != append.layer_index {
                return Err(Error::cache(format!(
                    "device paged KV layer order mismatch: expected layer {}, got {}",
                    layer.layer_index, append.layer_index
                )));
            }
            if layer.layer_kind != append.layer_kind {
                return Err(Error::cache(format!(
                    "device paged KV layer kind mismatch at layer {}",
                    layer.layer_index
                )));
            }
            validate_exact_shape(
                "device_decode_k_shape",
                append.cache_k.dims(),
                &[
                    self.spec.batch,
                    self.spec.attention_heads,
                    source_tokens,
                    self.spec.key_head_dim,
                ],
            )?;
            validate_exact_shape(
                "device_decode_v_shape",
                append.cache_v.dims(),
                &[
                    self.spec.batch,
                    self.spec.attention_heads,
                    source_tokens,
                    self.spec.value_head_dim,
                ],
            )?;
            let download_started_at = Instant::now();
            let downloaded_k = backend.device_download_f32_tensor(&append.cache_k)?;
            let downloaded_v = backend.device_download_f32_tensor(&append.cache_v)?;
            let host_k = if accepted_tokens == source_tokens {
                downloaded_k
            } else {
                slice_cache_token_range(
                    "device accepted K/V key prefix",
                    &downloaded_k,
                    0,
                    accepted_tokens,
                )?
            };
            let host_v = if accepted_tokens == source_tokens {
                downloaded_v
            } else {
                slice_cache_token_range(
                    "device accepted K/V value prefix",
                    &downloaded_v,
                    0,
                    accepted_tokens,
                )?
            };
            record_q2_runtime_stage(
                self.profile_step_index,
                Some(layer.layer_index),
                "decode.cold_kv_download_current",
                download_started_at.elapsed(),
            );
            let append_started_at = Instant::now();
            self.cold.store_mut().write_layer_block(
                layer.layer_index,
                append_position,
                &host_k,
                &host_v,
            )?;
            if let Some(index_key_device) = append.index_key.as_ref() {
                let index_key = backend.device_download_f32_tensor(index_key_device)?;
                validate_dsa_index_key_shape("decode", layer.layer_index, &index_key)?;
                let dsa_index = self.dsa_index.as_mut().ok_or_else(|| {
                    Error::cache(format!(
                        "DSA index store missing while appending layer {}",
                        layer.layer_index
                    ))
                })?;
                for token_offset in 0..accepted_tokens {
                    let token = slice_index_token_range(
                        "device accepted DSA index prefix",
                        &index_key,
                        token_offset,
                        1,
                    )?;
                    dsa_index.append_decode_layer(
                        layer.layer_index,
                        append_position + token_offset,
                        &token,
                    )?;
                }
            }
            record_q2_runtime_stage(
                self.profile_step_index,
                Some(layer.layer_index),
                "decode.cold_kv_append",
                append_started_at.elapsed(),
            );
        }
        for append in layer_kv_cache {
            if let Some(hot) = self
                .hot_layers
                .iter_mut()
                .find(|hot| hot.layer_index == Some(append.layer_index))
            {
                copy_logical_kv_tokens(
                    backend,
                    &append.cache_k,
                    source_tokens,
                    &hot.k,
                    hot.capacity_tokens,
                    append_position,
                    accepted_tokens,
                    batch,
                    attention_heads,
                    key_head_dim,
                )?;
                copy_logical_kv_tokens(
                    backend,
                    &append.cache_v,
                    source_tokens,
                    &hot.v,
                    hot.capacity_tokens,
                    append_position,
                    accepted_tokens,
                    batch,
                    attention_heads,
                    value_head_dim,
                )?;
            }
        }
        self.cold.store_mut().flush()?;
        if let Some(dsa_index) = self.dsa_index.as_mut() {
            dsa_index.flush()?;
        }
        self.cached_tokens = needed_tokens;
        self.prefetch_first_layer()?;
        self.metrics.write_nanoseconds = self
            .metrics
            .write_nanoseconds
            .saturating_add(elapsed_nanoseconds_u64(write_started.elapsed()));
        Ok(())
    }

    fn device_paged_kv_for_layer<B: Backend>(
        &mut self,
        layer_index: usize,
        backend: &B,
    ) -> Result<Option<DevicePagedKvView>> {
        let started_at = Instant::now();
        let layer_position = self.layer_position(layer_index)?;
        let token_count = self.cached_tokens()?;
        let page_count = token_count.div_ceil(self.spec.page_size) as u64;
        self.metrics.full_layer_lookups = self.metrics.full_layer_lookups.saturating_add(1);
        self.metrics.page_lookups = self.metrics.page_lookups.saturating_add(page_count);
        if let Some(hot) = self
            .hot_layers
            .iter()
            .find(|hot| hot.layer_index == Some(layer_index))
        {
            self.metrics.full_layer_hits = self.metrics.full_layer_hits.saturating_add(1);
            self.metrics.page_hits = self.metrics.page_hits.saturating_add(page_count);
            if hot.capacity_tokens < token_count {
                return Err(Error::cache(format!(
                    "resident device KV layer {layer_index} has capacity {} for {token_count} tokens",
                    hot.capacity_tokens
                )));
            }
            let view = DevicePagedKvView {
                batch: self.spec.batch,
                attention_heads: self.spec.attention_heads,
                key_head_dim: self.spec.key_head_dim,
                value_head_dim: self.spec.value_head_dim,
                page_size: self.spec.page_size,
                cached_tokens: token_count,
                capacity_tokens: hot.capacity_tokens,
                k: hot.k.clone(),
                v: hot.v.clone(),
            };
            view.validate()?;
            record_q2_runtime_stage(
                self.profile_step_index,
                Some(layer_index),
                "decode.resident_device_kv_view",
                started_at.elapsed(),
            );
            self.metrics.read_nanoseconds = self
                .metrics
                .read_nanoseconds
                .saturating_add(elapsed_nanoseconds_u64(started_at.elapsed()));
            return Ok(Some(view));
        }
        self.metrics.full_layer_misses = self.metrics.full_layer_misses.saturating_add(1);
        self.metrics.page_misses = self.metrics.page_misses.saturating_add(page_count);
        let (keys, values) = self
            .prefetch
            .take_or_read(&self.cold, layer_index, 0, token_count)?;
        record_q2_runtime_stage(
            self.profile_step_index,
            Some(layer_index),
            "decode.cold_kv_prefetch_wait",
            started_at.elapsed(),
        );
        self.prefetch_layer_after(layer_position)?;

        let upload_started_at = Instant::now();
        self.ensure_streaming_hot_capacity(backend, token_count)?;
        let hot = self
            .streaming_hot
            .as_mut()
            .ok_or_else(|| Error::cache("cold device KV hot window is not initialized"))?;
        let source_k = require_device_value(
            "cold device KV K hot upload",
            backend.device_upload_f32_tensor(&keys)?,
        )?;
        let source_v = require_device_value(
            "cold device KV V hot upload",
            backend.device_upload_f32_tensor(&values)?,
        )?;
        copy_logical_kv_tokens(
            backend,
            &source_k,
            token_count,
            &hot.k,
            hot.capacity_tokens,
            0,
            token_count,
            self.spec.batch,
            self.spec.attention_heads,
            self.spec.key_head_dim,
        )?;
        copy_logical_kv_tokens(
            backend,
            &source_v,
            token_count,
            &hot.v,
            hot.capacity_tokens,
            0,
            token_count,
            self.spec.batch,
            self.spec.attention_heads,
            self.spec.value_head_dim,
        )?;
        hot.layer_index = Some(layer_index);
        record_q2_runtime_stage(
            self.profile_step_index,
            Some(layer_index),
            "decode.cold_kv_hot_upload",
            upload_started_at.elapsed(),
        );

        let view = DevicePagedKvView {
            batch: self.spec.batch,
            attention_heads: self.spec.attention_heads,
            key_head_dim: self.spec.key_head_dim,
            value_head_dim: self.spec.value_head_dim,
            page_size: self.spec.page_size,
            cached_tokens: token_count,
            capacity_tokens: hot.capacity_tokens,
            k: hot.k.clone(),
            v: hot.v.clone(),
        };
        view.validate()?;
        record_q2_runtime_stage(
            self.profile_step_index,
            Some(layer_index),
            "decode.cold_device_kv_view",
            started_at.elapsed(),
        );
        self.metrics.read_nanoseconds = self
            .metrics
            .read_nanoseconds
            .saturating_add(elapsed_nanoseconds_u64(started_at.elapsed()));
        Ok(Some(view))
    }

    fn selected_device_kv_for_layer<B: Backend>(
        &mut self,
        layer_index: usize,
        token_indices: &[u32],
        backend: &B,
    ) -> Result<Option<DeviceSelectedKvView>> {
        let total_started = Instant::now();
        if token_indices.is_empty() {
            return Ok(None);
        }
        self.metrics.selected_row_lookups = self.metrics.selected_row_lookups.saturating_add(1);
        self.metrics.selected_row_misses = self.metrics.selected_row_misses.saturating_add(1);
        let selected_pages = token_indices
            .iter()
            .map(|token| *token as usize / self.spec.page_size)
            .collect::<HashSet<_>>()
            .len() as u64;
        self.metrics.page_lookups = self.metrics.page_lookups.saturating_add(selected_pages);
        self.metrics.page_misses = self.metrics.page_misses.saturating_add(selected_pages);
        self.metrics.selected_rows = self
            .metrics
            .selected_rows
            .saturating_add(u64::try_from(token_indices.len()).unwrap_or(u64::MAX));
        let layer_position = self.layer_position(layer_index)?;
        let read_started_at = Instant::now();
        let selected =
            self.selected_prefetch
                .take_or_read(&self.cold, layer_index, token_indices)?;
        record_q2_runtime_stage(
            self.profile_step_index,
            Some(layer_index),
            "decode.cold_kv_selected_q8_read",
            read_started_at.elapsed(),
        );
        self.prefetch_selected_layer_after(layer_position, token_indices)?;

        let upload_started_at = Instant::now();
        validate_exact_shape(
            "selected_cold_q8_key_shape",
            &[
                selected.key.batch,
                selected.key.attention_heads,
                selected.key.selected_tokens,
            ],
            &[
                self.spec.batch,
                self.spec.attention_heads,
                token_indices.len(),
            ],
        )?;
        validate_exact_shape(
            "selected_cold_q8_value_shape",
            &[
                selected.value.batch,
                selected.value.attention_heads,
                selected.value.selected_tokens,
            ],
            &[
                self.spec.batch,
                self.spec.attention_heads,
                token_indices.len(),
            ],
        )?;
        validate_exact_shape(
            "selected_cold_q8_key_dim",
            &[selected.key.head_dim],
            &[self.spec.key_head_dim],
        )?;
        validate_exact_shape(
            "selected_cold_q8_value_dim",
            &[selected.value.head_dim],
            &[self.spec.value_head_dim],
        )?;
        let view = backend.q8_row_selected_kv_device(
            &selected.key.payload,
            &selected.value.payload,
            selected.key.batch,
            selected.key.attention_heads,
            selected.key.selected_tokens,
            selected.key.head_dim,
            selected.value.head_dim,
        )?;
        let view = view.ok_or_else(|| {
            Error::backend("selected cold Q8 KV upload requires native Metal Q8 row decode")
        })?;
        record_q2_runtime_stage(
            self.profile_step_index,
            Some(layer_index),
            "decode.cold_kv_selected_q8_decode",
            upload_started_at.elapsed(),
        );

        view.validate()?;
        self.metrics.read_nanoseconds = self
            .metrics
            .read_nanoseconds
            .saturating_add(elapsed_nanoseconds_u64(total_started.elapsed()));
        Ok(Some(view))
    }

    fn dsa_index_keys_for_layer(&mut self, layer_index: usize) -> Result<Option<F32Tensor>> {
        let Some(dsa_index) = self.dsa_index.as_mut() else {
            return Ok(None);
        };
        let started_at = Instant::now();
        let index_keys = dsa_index.read_layer_contiguous(layer_index, 0, self.cached_tokens)?;
        record_q2_runtime_stage(
            self.profile_step_index,
            Some(layer_index),
            "decode.dsa_index_read",
            started_at.elapsed(),
        );
        Ok(Some(index_keys))
    }

    fn dsa_index_keys_for_layer_device<B: Backend>(
        &mut self,
        layer_index: usize,
        backend: &B,
    ) -> Result<Option<backend::DeviceValue>> {
        let total_started = Instant::now();
        let Some(index_keys) = self.dsa_index_keys_for_layer(layer_index)? else {
            return Ok(None);
        };
        let upload_started_at = Instant::now();
        let device = require_device_value(
            "DSA index key device upload",
            backend.device_upload_f32_tensor(&index_keys)?,
        )?;
        record_q2_runtime_stage(
            self.profile_step_index,
            Some(layer_index),
            "decode.dsa_index_device_upload",
            upload_started_at.elapsed(),
        );
        self.metrics.read_nanoseconds = self
            .metrics
            .read_nanoseconds
            .saturating_add(elapsed_nanoseconds_u64(total_started.elapsed()));
        Ok(Some(device))
    }

    fn cached_tokens(&self) -> Result<usize> {
        if self.layers.is_empty() {
            return Err(Error::cache("device paged KV cache has no layers"));
        }
        Ok(self.cached_tokens)
    }

    fn layer_position(&self, layer_index: usize) -> Result<usize> {
        self.layers
            .iter()
            .position(|entry| entry.layer_index == layer_index)
            .ok_or_else(|| Error::cache(format!("device KV layer {layer_index} is not cached")))
    }

    fn prefetch_first_layer(&mut self) -> Result<()> {
        if self.hot_layers.len() == self.layers.len() {
            return Ok(());
        }
        if let Some(layer) = self.layers.first() {
            self.prefetch.schedule(
                self.cold.store().clone_reader()?,
                layer.layer_index,
                0,
                self.cached_tokens,
            )?;
        }
        Ok(())
    }

    fn prefetch_layer_after(&mut self, layer_position: usize) -> Result<()> {
        if let Some(next_layer) = self.layers.get(layer_position + 1) {
            self.prefetch.schedule(
                self.cold.store().clone_reader()?,
                next_layer.layer_index,
                0,
                self.cached_tokens,
            )?;
        }
        Ok(())
    }

    fn prefetch_selected_layer_after(
        &mut self,
        layer_position: usize,
        token_indices: &[u32],
    ) -> Result<()> {
        if token_indices.is_empty() {
            return Ok(());
        }
        if let Some(layer) = self
            .layers
            .iter()
            .skip(layer_position + 1)
            .find(|layer| layer.layer_kind == model::LayerKind::SparseMoe)
        {
            self.selected_prefetch.schedule(
                self.cold.store().clone_reader()?,
                layer.layer_index,
                token_indices.to_vec(),
            )?;
        }
        Ok(())
    }

    fn initialize_hot_layers<B: Backend>(
        &mut self,
        backend: &B,
        needed_tokens: usize,
    ) -> Result<()> {
        let needed_capacity =
            device_capacity_tokens(needed_tokens, self.spec.page_size, self.spec.max_context)?;
        if self.hot_all_layers_bytes(needed_capacity)? > self.policy.hot_all_layers_budget_bytes {
            return Ok(());
        }
        let layer_indices = self
            .layers
            .iter()
            .map(|layer| layer.layer_index)
            .collect::<Vec<_>>();
        let mut hot_layers = Vec::with_capacity(layer_indices.len());
        for layer_index in layer_indices {
            hot_layers.push(self.allocate_hot_window(
                backend,
                needed_capacity,
                Some(layer_index),
            )?);
        }
        self.hot_layers = hot_layers;
        Ok(())
    }

    fn prepare_hot_layers_for_append<B: Backend>(
        &mut self,
        backend: &B,
        needed_tokens: usize,
    ) -> Result<()> {
        if self.hot_layers.is_empty() {
            return Ok(());
        }
        let needed_capacity =
            device_capacity_tokens(needed_tokens, self.spec.page_size, self.spec.max_context)?;
        if self.hot_all_layers_bytes(needed_capacity)? > self.policy.hot_all_layers_budget_bytes {
            self.hot_layers.clear();
            return Ok(());
        }
        if self
            .hot_layers
            .iter()
            .all(|hot| hot.capacity_tokens >= needed_capacity)
        {
            return Ok(());
        }

        let mut grown = Vec::with_capacity(self.hot_layers.len());
        for hot in &self.hot_layers {
            let next = self.allocate_hot_window(backend, needed_capacity, hot.layer_index)?;
            require_device_copy(
                "resident K/V K growth copy",
                backend.device_copy_same_dtype(&hot.k, 0, &next.k, 0, hot.k.element_count()?)?,
            )?;
            require_device_copy(
                "resident K/V V growth copy",
                backend.device_copy_same_dtype(&hot.v, 0, &next.v, 0, hot.v.element_count()?)?,
            )?;
            grown.push(next);
        }
        self.hot_layers = grown;
        Ok(())
    }

    fn ensure_streaming_hot_capacity<B: Backend>(
        &mut self,
        backend: &B,
        needed_tokens: usize,
    ) -> Result<()> {
        let needed_capacity =
            device_capacity_tokens(needed_tokens, self.spec.page_size, self.spec.max_context)?;
        if self
            .streaming_hot
            .as_ref()
            .is_some_and(|hot| hot.capacity_tokens >= needed_capacity)
        {
            return Ok(());
        }

        self.streaming_hot = Some(self.allocate_hot_window(backend, needed_capacity, None)?);
        Ok(())
    }

    fn allocate_hot_window<B: Backend>(
        &self,
        backend: &B,
        capacity_tokens: usize,
        layer_index: Option<usize>,
    ) -> Result<HotDeviceLayerWindow> {
        let page_count = capacity_tokens / self.spec.page_size;
        let k = require_device_value(
            "cold device KV K hot allocation",
            backend.device_alloc_f32_tensor(&device_cache_shape(
                page_count,
                self.spec.batch,
                self.spec.attention_heads,
                self.spec.page_size,
                self.spec.key_head_dim,
            ))?,
        )?;
        let v = require_device_value(
            "cold device KV V hot allocation",
            backend.device_alloc_f32_tensor(&device_cache_shape(
                page_count,
                self.spec.batch,
                self.spec.attention_heads,
                self.spec.page_size,
                self.spec.value_head_dim,
            ))?,
        )?;
        Ok(HotDeviceLayerWindow {
            layer_index,
            k,
            v,
            capacity_tokens,
        })
    }

    fn hot_all_layers_bytes(&self, capacity_tokens: usize) -> Result<usize> {
        self.spec
            .batch
            .checked_mul(self.spec.attention_heads)
            .and_then(|values| values.checked_mul(capacity_tokens))
            .and_then(|values| {
                values.checked_mul(
                    self.spec
                        .key_head_dim
                        .checked_add(self.spec.value_head_dim)?,
                )
            })
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .and_then(|bytes| bytes.checked_mul(self.layers.len()))
            .ok_or_else(|| Error::cache("resident K/V budget byte count overflow"))
    }
}

impl Drop for DevicePagedRuntimeCache {
    fn drop(&mut self) {
        self.prefetch.join_and_discard();
        self.selected_prefetch.join_and_discard();
        let _ = std::fs::remove_file(self.cold.store().path());
        if let Some(dsa_index) = self.dsa_index.as_ref() {
            let _ = std::fs::remove_file(dsa_index.path());
        }
    }
}

impl ColdKvPrefetcher {
    fn new() -> Self {
        Self { pending: None }
    }

    fn schedule(
        &mut self,
        mut reader: ColdKvBlockStore,
        layer_index: usize,
        token_start: usize,
        token_count: usize,
    ) -> Result<()> {
        if token_count == 0 {
            return Ok(());
        }
        if self.pending.as_ref().is_some_and(|job| {
            job.layer_index == layer_index
                && job.token_start == token_start
                && job.token_count == token_count
        }) {
            return Ok(());
        }
        self.join_and_discard();
        let handle = thread::spawn(move || {
            reader.read_layer_range_contiguous(layer_index, token_start, token_count)
        });
        self.pending = Some(ColdKvPrefetchJob {
            layer_index,
            token_start,
            token_count,
            handle,
        });
        Ok(())
    }

    fn take_or_read(
        &mut self,
        cold: &LayeredColdKvBlockStore,
        layer_index: usize,
        token_start: usize,
        token_count: usize,
    ) -> Result<(F32Tensor, F32Tensor)> {
        if let Some(job) = self.pending.take() {
            if job.layer_index == layer_index
                && job.token_start == token_start
                && job.token_count == token_count
            {
                return join_prefetch_job(job);
            }
            let _ = join_prefetch_job(job);
        }

        let mut reader = cold.store().clone_reader()?;
        reader.read_layer_range_contiguous(layer_index, token_start, token_count)
    }

    fn join_and_discard(&mut self) {
        if let Some(job) = self.pending.take() {
            let _ = join_prefetch_job(job);
        }
    }
}

fn join_prefetch_job(job: ColdKvPrefetchJob) -> Result<(F32Tensor, F32Tensor)> {
    let ColdKvPrefetchJob {
        layer_index,
        token_start,
        token_count,
        handle,
    } = job;
    handle.join().map_err(|_| {
        Error::cache(format!(
            "cold KV prefetch thread panicked for layer={layer_index} token_range=[{token_start},{})",
            token_start.saturating_add(token_count)
        ))
    })?
}

impl SelectedColdKvPrefetcher {
    fn new() -> Self {
        Self { pending: None }
    }

    fn schedule(
        &mut self,
        mut reader: ColdKvBlockStore,
        layer_index: usize,
        token_indices: Vec<u32>,
    ) -> Result<()> {
        if token_indices.is_empty() {
            return Ok(());
        }
        if self
            .pending
            .as_ref()
            .is_some_and(|job| job.layer_index == layer_index && job.token_indices == token_indices)
        {
            return Ok(());
        }
        self.join_and_discard();
        let job_indices = token_indices.clone();
        let handle =
            thread::spawn(move || reader.read_layer_selected_q8_rows(layer_index, &job_indices));
        self.pending = Some(SelectedColdKvPrefetchJob {
            layer_index,
            token_indices,
            handle,
        });
        Ok(())
    }

    fn take_or_read(
        &mut self,
        cold: &LayeredColdKvBlockStore,
        layer_index: usize,
        token_indices: &[u32],
    ) -> Result<ColdKvSelectedQ8LayerRows> {
        if let Some(job) = self.pending.take() {
            if job.layer_index == layer_index && job.token_indices == token_indices {
                return join_selected_prefetch_job(job);
            }
            let _ = join_selected_prefetch_job(job);
        }

        let mut reader = cold.store().clone_reader()?;
        reader.read_layer_selected_q8_rows(layer_index, token_indices)
    }

    fn join_and_discard(&mut self) {
        if let Some(job) = self.pending.take() {
            let _ = join_selected_prefetch_job(job);
        }
    }
}

fn join_selected_prefetch_job(job: SelectedColdKvPrefetchJob) -> Result<ColdKvSelectedQ8LayerRows> {
    let SelectedColdKvPrefetchJob {
        layer_index,
        token_indices,
        handle,
    } = job;
    handle.join().map_err(|_| {
        Error::cache(format!(
            "selected cold KV prefetch thread panicked for layer={layer_index} selected_tokens={}",
            token_indices.len()
        ))
    })?
}

fn unique_cold_kv_path() -> PathBuf {
    static NEXT_COLD_KV_ID: OnceLock<Mutex<u64>> = OnceLock::new();
    let counter = NEXT_COLD_KV_ID.get_or_init(|| Mutex::new(0));
    let mut id = counter.lock().expect("cold KV id lock poisoned");
    let current = *id;
    *id = id.saturating_add(1);
    std::env::temp_dir().join(format!(
        "inferno-cold-kv-{}-{current}.kv",
        std::process::id()
    ))
}

fn unique_dsa_index_path() -> PathBuf {
    static NEXT_DSA_INDEX_ID: OnceLock<Mutex<u64>> = OnceLock::new();
    let counter = NEXT_DSA_INDEX_ID.get_or_init(|| Mutex::new(0));
    let mut id = counter.lock().expect("DSA index id lock poisoned");
    let current = *id;
    *id = id.saturating_add(1);
    std::env::temp_dir().join(format!(
        "inferno-dsa-index-{}-{current}.idx",
        std::process::id()
    ))
}

fn validate_ssd_kv_benchmark_config(config: &SsdKvBenchmarkConfig) -> Result<()> {
    if config.layer_count == 0 {
        return Err(Error::runtime(
            "SSD KV benchmark layer_count must be positive",
        ));
    }
    if config.batch == 0 {
        return Err(Error::runtime("SSD KV benchmark batch must be positive"));
    }
    if config.attention_heads == 0 {
        return Err(Error::runtime(
            "SSD KV benchmark attention_heads must be positive",
        ));
    }
    if config.tokens == 0 {
        return Err(Error::runtime("SSD KV benchmark tokens must be positive"));
    }
    if config.key_head_dim == 0 || config.value_head_dim == 0 {
        return Err(Error::runtime(
            "SSD KV benchmark head dimensions must be positive",
        ));
    }
    if config.block_tokens == 0 || config.page_size == 0 {
        return Err(Error::runtime(
            "SSD KV benchmark block_tokens and page_size must be positive",
        ));
    }
    Ok(())
}

fn validate_dsa_attention_benchmark_config(config: &DsaAttentionBenchmarkConfig) -> Result<()> {
    if config.layer_count == 0 {
        return Err(Error::runtime(
            "DSA attention benchmark layer_count must be positive",
        ));
    }
    if config.batch == 0 {
        return Err(Error::runtime(
            "DSA attention benchmark batch must be positive",
        ));
    }
    if config.attention_heads == 0 {
        return Err(Error::runtime(
            "DSA attention benchmark attention_heads must be positive",
        ));
    }
    if config.tokens == 0 || config.selected_tokens == 0 {
        return Err(Error::runtime(
            "DSA attention benchmark token counts must be positive",
        ));
    }
    if config.selected_tokens > config.tokens {
        return Err(Error::runtime(format!(
            "DSA attention benchmark selected_tokens {} exceed tokens {}",
            config.selected_tokens, config.tokens
        )));
    }
    if config.key_head_dim == 0 || config.value_head_dim == 0 {
        return Err(Error::runtime(
            "DSA attention benchmark head dimensions must be positive",
        ));
    }
    if config.page_size == 0 {
        return Err(Error::runtime(
            "DSA attention benchmark page_size must be positive",
        ));
    }
    Ok(())
}

fn validate_moe_expert_benchmark_config(config: &MoeExpertBenchmarkConfig) -> Result<()> {
    if config.iterations == 0 {
        return Err(Error::runtime(
            "MoE expert benchmark iterations must be positive",
        ));
    }
    if config.token_count == 0 || config.assignment_count == 0 {
        return Err(Error::runtime(
            "MoE expert benchmark token_count and assignment_count must be positive",
        ));
    }
    if config.hidden_size == 0 || config.intermediate_size == 0 {
        return Err(Error::runtime(
            "MoE expert benchmark hidden_size and intermediate_size must be positive",
        ));
    }
    if config.expert_count == 0 {
        return Err(Error::runtime(
            "MoE expert benchmark expert_count must be positive",
        ));
    }
    if config.hidden_size % Q2_K_BLOCK_VALUES != 0 {
        return Err(Error::runtime(format!(
            "MoE expert benchmark hidden_size {} must be divisible by {Q2_K_BLOCK_VALUES}",
            config.hidden_size
        )));
    }
    if config.intermediate_size % Q2_K_BLOCK_VALUES != 0 {
        return Err(Error::runtime(format!(
            "MoE expert benchmark intermediate_size {} must be divisible by {Q2_K_BLOCK_VALUES}",
            config.intermediate_size
        )));
    }
    Ok(())
}

fn synthetic_benchmark_kv_layers(
    config: &SsdKvBenchmarkConfig,
) -> Result<Vec<(usize, F32Tensor, F32Tensor)>> {
    let mut layers = Vec::with_capacity(config.layer_count);
    for layer_index in 0..config.layer_count {
        layers.push((
            layer_index,
            synthetic_kv_tensor(
                config.batch,
                config.attention_heads,
                config.tokens,
                config.key_head_dim,
                layer_index as f32,
            )?,
            synthetic_kv_tensor(
                config.batch,
                config.attention_heads,
                config.tokens,
                config.value_head_dim,
                1000.0 + layer_index as f32,
            )?,
        ));
    }
    Ok(layers)
}

fn synthetic_kv_tensor(
    batch: usize,
    attention_heads: usize,
    tokens: usize,
    head_dim: usize,
    offset: f32,
) -> Result<F32Tensor> {
    let element_count = batch
        .checked_mul(attention_heads)
        .and_then(|value| value.checked_mul(tokens))
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| Error::runtime("SSD KV benchmark tensor element count overflow"))?;
    let values = (0..element_count)
        .map(|index| offset + (index % 127) as f32 / 127.0)
        .collect::<Vec<_>>();
    F32Tensor::new(values, [batch, attention_heads, tokens, head_dim])
}

fn synthetic_moe_input(token_count: usize, hidden_size: usize) -> Result<F32Tensor> {
    let element_count = token_count
        .checked_mul(hidden_size)
        .ok_or_else(|| Error::runtime("MoE benchmark input element count overflow"))?;
    let values = (0..element_count)
        .map(|index| (index % 257) as f32 / 257.0 - 0.5)
        .collect::<Vec<_>>();
    F32Tensor::new(values, [token_count, hidden_size])
}

fn synthetic_moe_token_indices(token_count: usize, assignment_count: usize) -> Result<Vec<u32>> {
    (0..assignment_count)
        .map(|assignment| {
            u32::try_from(assignment % token_count)
                .map_err(|_| Error::runtime("MoE benchmark token index does not fit Metal u32"))
        })
        .collect()
}

fn synthetic_moe_expert_ids(expert_count: usize, assignment_count: usize) -> Result<Vec<u32>> {
    (0..assignment_count)
        .map(|assignment| {
            let expert = (assignment * 17 + 3) % expert_count;
            u32::try_from(expert)
                .map_err(|_| Error::runtime("MoE benchmark expert id does not fit Metal u32"))
        })
        .collect()
}

fn synthetic_q2_k_expert_weights(
    expert_count: usize,
    out_features: usize,
    in_features: usize,
    seed: u8,
) -> Result<Vec<u8>> {
    if in_features % Q2_K_BLOCK_VALUES != 0 {
        return Err(Error::runtime(format!(
            "synthetic Q2_K expert input width {in_features} must be divisible by {Q2_K_BLOCK_VALUES}"
        )));
    }
    let blocks_per_row = in_features / Q2_K_BLOCK_VALUES;
    let block_count = expert_count
        .checked_mul(out_features)
        .and_then(|value| value.checked_mul(blocks_per_row))
        .ok_or_else(|| Error::runtime("synthetic Q2_K expert block count overflow"))?;
    let byte_count = block_count
        .checked_mul(Q2_K_BLOCK_BYTES)
        .ok_or_else(|| Error::runtime("synthetic Q2_K expert byte count overflow"))?;
    let mut weights = Vec::with_capacity(byte_count);
    for block_index in 0..block_count {
        let scale_min = seed.wrapping_add((block_index % 23) as u8).max(1);
        let quant = seed
            .wrapping_mul(13)
            .wrapping_add((block_index % 251) as u8);
        weights.extend(q2_k_block(0x3c00, 0x0000, scale_min, quant));
    }
    Ok(weights)
}

fn q2_k_block(d: u16, dmin: u16, scale_min: u8, quant: u8) -> [u8; Q2_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q2_K_BLOCK_BYTES];
    block[..16].fill(scale_min);
    block[16..80].fill(quant);
    block[80..82].copy_from_slice(&d.to_le_bytes());
    block[82..84].copy_from_slice(&dmin.to_le_bytes());
    block
}

fn raw_layer_f32_bytes(
    batch: usize,
    attention_heads: usize,
    tokens: usize,
    key_head_dim: usize,
    value_head_dim: usize,
) -> Result<u64> {
    checked_f32_bytes(
        "SSD KV benchmark raw layer bytes",
        &[
            batch,
            attention_heads,
            tokens,
            key_head_dim
                .checked_add(value_head_dim)
                .ok_or_else(|| Error::runtime("SSD KV benchmark head dimension overflow"))?,
        ],
    )
}

fn cache_spec(
    max_context: usize,
    page_size: usize,
    layer_kv_cache: &[LayerKvCacheTensors],
) -> Result<LayeredPagedKvCacheSpec> {
    let first_layer = layer_kv_cache
        .first()
        .ok_or_else(|| Error::runtime("prefill produced no layer K/V tensors"))?;
    let k_dims = first_layer.cache_k.dims();
    let v_dims = first_layer.cache_v.dims();
    if k_dims.len() != 4 || v_dims.len() != 4 {
        return Err(Error::runtime(format!(
            "K/V tensors must be rank 4 [B,H,T,D], got k={k_dims:?} v={v_dims:?}"
        )));
    }
    validate_exact_shape(
        "cache_batch_heads_tokens",
        &[v_dims[0], v_dims[1], v_dims[2]],
        &[k_dims[0], k_dims[1], k_dims[2]],
    )?;

    Ok(LayeredPagedKvCacheSpec {
        batch: k_dims[0],
        attention_heads: k_dims[1],
        key_head_dim: k_dims[3],
        value_head_dim: v_dims[3],
        max_context,
        page_size,
    })
}

fn device_cache_spec(
    max_context: usize,
    page_size: usize,
    layer_kv_cache: &[LayerDeviceKvCacheTensors],
) -> Result<LayeredPagedKvCacheSpec> {
    let first_layer = layer_kv_cache
        .first()
        .ok_or_else(|| Error::runtime("device seed produced no layer K/V tensors"))?;
    let k_dims = first_layer.cache_k.dims();
    let v_dims = first_layer.cache_v.dims();
    if k_dims.len() != 4 || v_dims.len() != 4 {
        return Err(Error::runtime(format!(
            "device seed K/V tensors must be rank 4 [B,H,T,D], got k={k_dims:?} v={v_dims:?}"
        )));
    }
    validate_exact_shape(
        "device_seed_cache_batch_heads_tokens",
        &[v_dims[0], v_dims[1], v_dims[2]],
        &[k_dims[0], k_dims[1], k_dims[2]],
    )?;
    validate_exact_shape("device_seed_cache_token_count", &[k_dims[2]], &[1])?;

    let spec = LayeredPagedKvCacheSpec {
        batch: k_dims[0],
        attention_heads: k_dims[1],
        key_head_dim: k_dims[3],
        value_head_dim: v_dims[3],
        max_context,
        page_size,
    };
    for entry in layer_kv_cache {
        validate_exact_shape(
            format!("device_seed_layer_{}_k_shape", entry.layer_index),
            entry.cache_k.dims(),
            &[spec.batch, spec.attention_heads, 1, spec.key_head_dim],
        )?;
        validate_exact_shape(
            format!("device_seed_layer_{}_v_shape", entry.layer_index),
            entry.cache_v.dims(),
            &[spec.batch, spec.attention_heads, 1, spec.value_head_dim],
        )?;
    }
    Ok(spec)
}

fn infer_dsa_index_spec(
    layer_kv_cache: &[LayerKvCacheTensors],
    block_tokens: usize,
) -> Result<Option<DsaIndexStoreSpec>> {
    let Some(first) = layer_kv_cache
        .iter()
        .find_map(|entry| entry.index_key.as_ref())
    else {
        return Ok(None);
    };
    validate_dsa_index_key_shape("prefill", 0, first)?;
    let dims = first.dims();
    let spec = DsaIndexStoreSpec {
        batch: dims[0],
        dim: dims[2],
        block_tokens,
    };
    spec.validate()?;
    for entry in layer_kv_cache {
        if let Some(index_key) = entry.index_key.as_ref() {
            validate_dsa_index_key_shape("prefill", entry.layer_index, index_key)?;
            validate_exact_shape(
                format!("prefill_dsa_index_layer_{}_spec", entry.layer_index),
                &[index_key.dims()[0], index_key.dims()[2]],
                &[spec.batch, spec.dim],
            )?;
        }
    }
    Ok(Some(spec))
}

fn infer_device_dsa_index_spec(
    layer_kv_cache: &[LayerDeviceKvCacheTensors],
    block_tokens: usize,
) -> Result<Option<DsaIndexStoreSpec>> {
    let Some(first) = layer_kv_cache
        .iter()
        .find_map(|entry| entry.index_key.as_ref())
    else {
        return Ok(None);
    };
    validate_dsa_index_key_dims("device_seed", 0, first.dims())?;
    let dims = first.dims();
    let spec = DsaIndexStoreSpec {
        batch: dims[0],
        dim: dims[2],
        block_tokens,
    };
    spec.validate()?;
    for entry in layer_kv_cache {
        if let Some(index_key) = entry.index_key.as_ref() {
            validate_dsa_index_key_dims("device_seed", entry.layer_index, index_key.dims())?;
            validate_exact_shape(
                format!("device_seed_dsa_index_layer_{}_spec", entry.layer_index),
                &[index_key.dims()[0], index_key.dims()[2]],
                &[spec.batch, spec.dim],
            )?;
        }
    }
    Ok(Some(spec))
}

fn validate_dsa_index_key_shape(
    phase: &str,
    layer_index: usize,
    index_key: &F32Tensor,
) -> Result<()> {
    validate_dsa_index_key_dims(phase, layer_index, index_key.dims())
}

fn validate_dsa_index_key_dims(phase: &str, layer_index: usize, dims: &[usize]) -> Result<()> {
    if dims.len() != 3 {
        return Err(Error::cache(format!(
            "DSA index key for {phase} layer {layer_index} must be rank 3 [B,T,D], got {dims:?}"
        )));
    }
    if dims[0] == 0 || dims[1] == 0 || dims[2] == 0 {
        return Err(Error::cache(format!(
            "DSA index key for {phase} layer {layer_index} must have positive dimensions, got {dims:?}"
        )));
    }
    Ok(())
}

fn device_capacity_tokens(tokens: usize, page_size: usize, max_context: usize) -> Result<usize> {
    if tokens == 0 {
        return Err(Error::cache(
            "device KV capacity requires at least one token",
        ));
    }
    if tokens > max_context {
        return Err(Error::cache(format!(
            "device KV token count {tokens} exceeds max_context {max_context}"
        )));
    }
    let pages = tokens
        .checked_add(page_size - 1)
        .ok_or_else(|| Error::cache("device KV capacity token count overflow"))?
        / page_size;
    pages
        .max(1)
        .checked_mul(page_size)
        .ok_or_else(|| Error::cache("device KV capacity overflow"))
}

fn device_cache_shape(
    page_count: usize,
    batch: usize,
    attention_heads: usize,
    page_size: usize,
    head_dim: usize,
) -> Vec<usize> {
    vec![page_count, batch, attention_heads, page_size, head_dim]
}

fn require_device_value(
    context: &'static str,
    value: Option<backend::DeviceValue>,
) -> Result<backend::DeviceValue> {
    value.ok_or_else(|| Error::backend(format!("{context} requires native Metal device values")))
}

fn require_device_copy(context: &'static str, copied: Option<()>) -> Result<()> {
    copied.ok_or_else(|| Error::backend(format!("{context} requires native Metal device copy")))
}

fn slice_device_token_prefix<B: Backend>(
    backend: &B,
    tensor: &backend::DeviceValue,
    token_count: usize,
) -> Result<backend::DeviceValue> {
    let dims = tensor.dims();
    validate_exact_shape("device token prefix rank", &[dims.len()], &[3])?;
    validate_exact_shape("device token prefix batch", &[dims[0]], &[1])?;
    if token_count == 0 || token_count > dims[1] {
        return Err(Error::runtime(format!(
            "device token prefix count {token_count} exceeds source token count {}",
            dims[1]
        )));
    }
    if token_count == dims[1] {
        return Ok(tensor.clone());
    }
    let output = require_device_value(
        "device token prefix allocation",
        backend.device_alloc_f32_tensor(&[1, token_count, dims[2]])?,
    )?;
    let element_count = token_count
        .checked_mul(dims[2])
        .ok_or_else(|| Error::runtime("device token prefix element count overflow"))?;
    require_device_copy(
        "device token prefix copy",
        backend.device_copy_f32(tensor, 0, &output, 0, element_count)?,
    )?;
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn copy_logical_kv_tokens<B: Backend>(
    backend: &B,
    source: &backend::DeviceValue,
    source_tokens: usize,
    destination: &backend::DeviceValue,
    destination_capacity_tokens: usize,
    destination_start_token: usize,
    token_count: usize,
    batch: usize,
    attention_heads: usize,
    head_dim: usize,
) -> Result<()> {
    let source_expected = [batch, attention_heads, source_tokens, head_dim];
    validate_exact_shape(
        "device_kv_copy_source_shape",
        source.dims(),
        &source_expected,
    )?;
    let page_size = destination
        .dims()
        .get(3)
        .copied()
        .ok_or_else(|| Error::cache("device K/V destination must be rank 5"))?;
    let page_count = destination
        .dims()
        .first()
        .copied()
        .ok_or_else(|| Error::cache("device K/V destination must be rank 5"))?;
    validate_exact_shape(
        "device_kv_copy_destination_shape",
        destination.dims(),
        &[page_count, batch, attention_heads, page_size, head_dim],
    )?;
    validate_exact_shape(
        "device_kv_copy_destination_capacity",
        &[page_count * page_size],
        &[destination_capacity_tokens],
    )?;
    destination_start_token
        .checked_add(token_count)
        .filter(|end| *end <= destination_capacity_tokens)
        .ok_or_else(|| {
            Error::cache(format!(
                "device K/V copy destination token range starts at {destination_start_token} with {token_count} tokens but capacity is {destination_capacity_tokens}"
            ))
        })?;
    if token_count > source_tokens {
        return Err(Error::cache(format!(
            "device K/V copy token_count {token_count} exceeds source_tokens {source_tokens}"
        )));
    }

    for token_offset in 0..token_count {
        let logical_token = destination_start_token
            .checked_add(token_offset)
            .ok_or_else(|| Error::cache("device K/V logical token overflow"))?;
        let page_index = logical_token / page_size;
        let page_offset = logical_token - (page_index * page_size);
        for batch_index in 0..batch {
            for head_index in 0..attention_heads {
                let source_offset = ((batch_index * attention_heads + head_index) * source_tokens
                    + token_offset)
                    * head_dim;
                let destination_offset = (((page_index * batch + batch_index) * attention_heads
                    + head_index)
                    * page_size
                    + page_offset)
                    * head_dim;
                require_device_copy(
                    "device K/V row copy",
                    backend.device_copy_same_dtype(
                        source,
                        source_offset,
                        destination,
                        destination_offset,
                        head_dim,
                    )?,
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn paged_append_report(
    report: LayeredPagedCacheAppendReport,
    page_size: usize,
) -> Result<PagedCacheAppendReport> {
    Ok(PagedCacheAppendReport {
        page_size,
        layer_count: report.layer_count,
        start_position: report.start_position,
        appended_tokens: report.appended_tokens,
        end_position_exclusive: report.end_position_exclusive,
        cached_tokens: report.cached_tokens,
        next_position: report.next_position,
        allocated_pages: report.allocated_pages,
        page_count: report.page_count,
        next_decode_attention_scores_shape: report.next_decode_attention_scores_shape,
    })
}

fn layer_appends(layer_kv_cache: &[LayerKvCacheTensors]) -> Vec<LayerKvCacheAppend<'_>> {
    layer_kv_cache
        .iter()
        .map(|entry| LayerKvCacheAppend {
            layer_index: entry.layer_index,
            k: &entry.cache_k,
            v: &entry.cache_v,
        })
        .collect()
}

fn slice_cache_token_range(
    context: &str,
    tensor: &F32Tensor,
    token_start: usize,
    token_count: usize,
) -> Result<F32Tensor> {
    let dims = tensor.dims();
    validate_exact_shape(format!("{context}_rank"), &[dims.len()], &[4])?;
    let batch = dims[0];
    let heads = dims[1];
    let tokens = dims[2];
    let dim = dims[3];
    let token_end = token_start
        .checked_add(token_count)
        .ok_or_else(|| Error::cache(format!("{context} token range overflow")))?;
    if token_count == 0 || token_end > tokens {
        return Err(Error::cache(format!(
            "{context} token range [{token_start},{token_end}) exceeds token count {tokens}"
        )));
    }
    let mut values = Vec::with_capacity(batch * heads * token_count * dim);
    for batch_index in 0..batch {
        for head in 0..heads {
            let source = (((batch_index * heads + head) * tokens) + token_start)
                .checked_mul(dim)
                .ok_or_else(|| Error::cache(format!("{context} source offset overflow")))?;
            let source_end = source
                .checked_add(token_count * dim)
                .ok_or_else(|| Error::cache(format!("{context} source end overflow")))?;
            values.extend_from_slice(&tensor.values()[source..source_end]);
        }
    }
    F32Tensor::new(values, [batch, heads, token_count, dim])
}

fn slice_index_token_range(
    context: &str,
    tensor: &F32Tensor,
    token_start: usize,
    token_count: usize,
) -> Result<F32Tensor> {
    let dims = tensor.dims();
    validate_exact_shape(format!("{context}_rank"), &[dims.len()], &[3])?;
    let batch = dims[0];
    let tokens = dims[1];
    let dim = dims[2];
    let token_end = token_start
        .checked_add(token_count)
        .ok_or_else(|| Error::cache(format!("{context} token range overflow")))?;
    if token_count == 0 || token_end > tokens {
        return Err(Error::cache(format!(
            "{context} token range [{token_start},{token_end}) exceeds token count {tokens}"
        )));
    }
    let mut values = Vec::with_capacity(batch * token_count * dim);
    for batch_index in 0..batch {
        let source = (batch_index * tokens + token_start)
            .checked_mul(dim)
            .ok_or_else(|| Error::cache(format!("{context} source offset overflow")))?;
        let source_end = source
            .checked_add(token_count * dim)
            .ok_or_else(|| Error::cache(format!("{context} source end overflow")))?;
        values.extend_from_slice(&tensor.values()[source..source_end]);
    }
    F32Tensor::new(values, [batch, token_count, dim])
}

#[cfg(test)]
fn empty_expert_load_report() -> ExpertLoadReport {
    ExpertLoadReport {
        loaded_expert_requests: 0,
        cache_hits: 0,
        cache_misses: 0,
        hit_rate: 0.0,
        materialized_expert_bytes_loaded: 0,
        source_expert_bytes_loaded: 0,
        limitations: vec![
            "GLM routed experts use direct Q2 payload dispatch; fused Metal expert dispatch is not active yet"
                .to_string(),
        ],
    }
}

#[cfg(test)]
fn finalize_expert_load_report(report: &mut ExpertLoadReport) {
    report.loaded_expert_requests = report.cache_hits.saturating_add(report.cache_misses);
    if report.loaded_expert_requests > 0 {
        report.hit_rate = report.cache_hits as f32 / report.loaded_expert_requests as f32;
    }
}

#[cfg(test)]
fn accumulate_expert_load_step(report: &mut ExpertLoadReport, step: &GenerationStepReport) {
    report.cache_misses = report
        .cache_misses
        .saturating_add(step.model.expert_loads.loaded_expert_requests);
    report.materialized_expert_bytes_loaded = report
        .materialized_expert_bytes_loaded
        .saturating_add(step.model.expert_loads.materialized_expert_bytes_loaded);
    report.source_expert_bytes_loaded = report
        .source_expert_bytes_loaded
        .saturating_add(step.model.expert_loads.source_expert_bytes_loaded);
}

#[cfg(test)]
fn empty_decode_loop_report() -> DecodeLoopReport {
    DecodeLoopReport {
        step_count: 0,
        total_appended_tokens: 0,
        total_allocated_pages: 0,
        final_cached_tokens: 0,
        final_next_position: 0,
        max_attention_past_tokens: 0,
        materialized_full_logits: false,
        total_output_projection_source_payload_bytes_read: 0,
        peak_output_projection_decoded_f32_bytes: 0,
        last_hidden_states_shape: None,
        last_logits_shape: None,
        last_next_decode_attention_scores_shape: None,
        last_sampled_token_id: None,
        last_sampled_token_score: None,
    }
}

#[cfg(test)]
fn accumulate_decode_loop_report(report: &mut DecodeLoopReport, step: &GenerationStepReport) {
    report.step_count = report.step_count.saturating_add(1);
    report.total_appended_tokens = report
        .total_appended_tokens
        .saturating_add(step.cache_append.appended_tokens);
    report.total_allocated_pages = report
        .total_allocated_pages
        .saturating_add(step.cache_append.allocated_pages);
    report.final_cached_tokens = step.cached_tokens;
    report.final_next_position = step.next_position;
    report.max_attention_past_tokens = report
        .max_attention_past_tokens
        .max(step.model.max_attention_past_tokens);
    report.materialized_full_logits =
        report.materialized_full_logits || step.model.materialized_full_logits;
    report.total_output_projection_source_payload_bytes_read = report
        .total_output_projection_source_payload_bytes_read
        .saturating_add(step.model.output_projection_source_payload_bytes_read);
    report.peak_output_projection_decoded_f32_bytes = report
        .peak_output_projection_decoded_f32_bytes
        .max(step.model.output_projection_peak_decoded_f32_bytes);
    report.last_hidden_states_shape = Some(step.hidden_states_shape.clone());
    report.last_logits_shape = Some(step.logits_shape.clone());
    report.last_next_decode_attention_scores_shape =
        Some(step.next_decode_attention_scores_shape.clone());
    report.last_sampled_token_id = Some(step.sampled_token_id);
    report.last_sampled_token_score = Some(step.sampled_token_score);
}

fn contains_stop_token(token_id: u32, stop_token_ids: &[u32]) -> bool {
    !stop_token_ids.is_empty() && stop_token_ids.contains(&token_id)
}

fn record_q2_runtime_stage(
    step_index: usize,
    layer_index: Option<usize>,
    stage: &str,
    elapsed: Duration,
) {
    let value = format!("{:.3}", elapsed.as_secs_f64() * 1000.0);
    record_q2_runtime_value(step_index, layer_index, stage, &value);
}

fn record_q2_runtime_value(
    step_index: usize,
    layer_index: Option<usize>,
    metric: &str,
    value: &str,
) {
    if !Q2_RUNTIME_PROFILE_ENABLED.load(Ordering::Acquire) {
        return;
    }
    let Ok(mut profile) = q2_runtime_profile_file().lock() else {
        return;
    };
    let Some(file) = profile.as_mut() else {
        return;
    };
    let layer_index = layer_index
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string());
    let _ = writeln!(file, "{step_index}\t{layer_index}\t{metric}\t{value}",);
    let _ = file.flush();
}

fn q2_runtime_profile_file() -> &'static Mutex<Option<File>> {
    Q2_RUNTIME_PROFILE_FILE.get_or_init(|| Mutex::new(None))
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use backend::MetalBackend;
    use gguf::{
        GgmlType, GgufFile, GgufMetadataValueType, GGML_Q2_K_BLOCK_BYTES, GGML_Q8_0_BLOCK_BYTES,
        GGUF_MAGIC, GGUF_VERSION_V3,
    };
    use model::{Model, DEFAULT_GGUF_OUTPUT_CHUNK_ROWS};

    use super::*;

    #[test]
    fn kv_cache_rates_include_full_and_selected_lookups() {
        let metrics = KvCacheMetrics {
            full_layer_lookups: 4,
            full_layer_hits: 3,
            full_layer_misses: 1,
            selected_row_lookups: 6,
            selected_row_hits: 2,
            selected_row_misses: 4,
            page_lookups: 10,
            page_hits: 5,
            page_misses: 5,
            ..KvCacheMetrics::default()
        };

        assert_eq!(metrics.lookups(), 10);
        assert_eq!(metrics.hits(), 5);
        assert_eq!(metrics.misses(), 5);
        assert_eq!(metrics.hit_rate(), 0.5);
        assert_eq!(metrics.miss_rate(), 0.5);
    }

    #[test]
    fn token_cost_report_uses_per_token_cache_deltas() {
        let report = TokenCostReport {
            sparse_attention_nanoseconds: 40,
            sparse_input_norm_nanoseconds: 2,
            sparse_q_projection_nanoseconds: 5,
            sparse_kv_projection_nanoseconds: 7,
            sparse_cache_layout_nanoseconds: 3,
            sparse_dsa_indexer_nanoseconds: 4,
            sparse_context_attention_nanoseconds: 8,
            sparse_output_projection_nanoseconds: 6,
            expert_lookups: 10,
            expert_cache_hits: 3,
            expert_cache_misses: 7,
            kv_page_lookups: 8,
            kv_page_hits: 6,
            kv_page_misses: 2,
            expert_ssd_read_bytes: 12,
            kv_ssd_read_bytes: 5,
            ..TokenCostReport::default()
        };

        assert_eq!(report.expert_cache_hit_rate(), 0.3);
        assert_eq!(report.kv_page_hit_rate(), 0.75);
        assert_eq!(report.ssd_read_bytes(), 17);
        assert_eq!(report.sparse_attention_accounted_nanoseconds(), 35);
        assert_eq!(report.sparse_attention_unattributed_nanoseconds(), 5);
    }

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn q2_runtime_profile_is_disabled_by_default() {
        assert!(!Q2_RUNTIME_PROFILE_ENABLED.load(Ordering::Acquire));
        assert!(q2_runtime_profile_file().lock().unwrap().is_none());
    }

    #[test]
    fn memory_telemetry_can_be_enabled_explicitly() {
        enable_memory_telemetry();

        assert!(telemetry::memory_telemetry_enabled());
    }

    #[test]
    fn runtime_cache_uses_memory_storage() {
        let layer_kv_cache = tiny_layer_kv_cache(1);
        let memory =
            PagedRuntimeCache::new(8, 2, &layer_kv_cache).expect("memory cache should build");

        assert_eq!(memory.storage_kind(), "memory");
        assert_eq!(
            memory.runtime_kv_memory(),
            RuntimeKvMemoryBytes {
                hot_bytes: Some(0),
                cold_bytes: None,
            }
        );
    }

    #[test]
    fn cold_kv_policy_bounds_the_all_layer_hot_tier() {
        let policy = ColdKvRuntimePolicy::new(128, 1_000_000_000).unwrap();
        let block_error = ColdKvRuntimePolicy::new(0, 1_000_000_000).unwrap_err();
        let budget_error = ColdKvRuntimePolicy::new(128, 0).unwrap_err();

        assert_eq!(policy.block_tokens, 128);
        assert_eq!(policy.hot_all_layers_budget_bytes, 1_000_000_000);
        assert!(block_error.to_string().contains("block_tokens"));
        assert!(budget_error.to_string().contains("budget"));
    }

    #[test]
    fn dsa_index_store_preserves_batch_token_order() {
        let path = unique_temp_file("runtime-dsa-index");
        let mut store = LayeredDsaIndexBlockStore::create(
            &path,
            DsaIndexStoreSpec {
                batch: 2,
                dim: 2,
                block_tokens: 2,
            },
        )
        .unwrap();
        let prefill = F32Tensor::new(vec![1.0, 2.0, 10.0, 20.0], [2, 1, 2]).unwrap();
        store
            .write_prefill(&[DsaIndexLayerAppend {
                layer_index: 3,
                index_key: &prefill,
            }])
            .unwrap();
        let current = F32Tensor::new(vec![3.0, 4.0, 30.0, 40.0], [2, 1, 2]).unwrap();
        store.append_decode_layer(3, 1, &current).unwrap();

        let appended = store.read_layer_contiguous(3, 0, 2).unwrap();
        assert_eq!(appended.dims(), &[2, 2, 2]);
        assert_eq!(
            appended.values(),
            &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
        );
        fs::remove_file(path).ok();
    }

    #[test]
    fn paged_runtime_cache_persists_dsa_index_keys() {
        let prefill_layers = vec![LayerKvCacheTensors {
            layer_index: 3,
            layer_kind: model::LayerKind::SparseMoe,
            cache_k: F32Tensor::zeros([1, 2, 2, 256]).unwrap(),
            cache_v: F32Tensor::zeros([1, 2, 2, 256]).unwrap(),
            index_key: Some(F32Tensor::new(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]).unwrap()),
        }];
        let decode_layers = vec![LayerKvCacheTensors {
            layer_index: 3,
            layer_kind: model::LayerKind::SparseMoe,
            cache_k: F32Tensor::zeros([1, 2, 1, 256]).unwrap(),
            cache_v: F32Tensor::zeros([1, 2, 1, 256]).unwrap(),
            index_key: Some(F32Tensor::new(vec![5.0, 6.0], [1, 1, 2]).unwrap()),
        }];
        let mut cache = PagedRuntimeCache::new(8, 2, &prefill_layers).unwrap();

        cache.append_prefill(&prefill_layers).unwrap();
        cache.append_decode(&decode_layers).unwrap();

        let index_keys = cache.dsa_index_keys_for_layer(3).unwrap().unwrap();
        assert_eq!(index_keys.dims(), &[1, 3, 2]);
        assert_eq!(index_keys.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn cold_kv_prefetcher_reads_contiguous_layer_range() {
        let path = unique_temp_file("cold-prefetch");
        let mut cold = LayeredColdKvBlockStore::create(
            &path,
            ColdKvStoreSpec {
                batch: 1,
                attention_heads: 2,
                key_head_dim: 256,
                value_head_dim: 256,
                block_tokens: 2,
                codec: ColdKvCodec::Q8Row,
            },
        )
        .unwrap();
        let layer_kv_cache = tiny_layer_kv_cache(4);
        cold.write_prefill(&layer_appends(&layer_kv_cache)).unwrap();
        cold.store_mut().flush().unwrap();

        let mut prefetcher = ColdKvPrefetcher::new();
        prefetcher
            .schedule(cold.store().clone_reader().unwrap(), 0, 0, 4)
            .unwrap();
        let (keys, values) = prefetcher.take_or_read(&cold, 0, 0, 4).unwrap();

        assert_eq!(keys.dims(), &[1, 2, 4, 256]);
        assert_eq!(values.dims(), &[1, 2, 4, 256]);
        fs::remove_file(path).ok();
    }

    #[test]
    fn ssd_kv_benchmark_rejects_invalid_config() {
        let err = validate_ssd_kv_benchmark_config(&SsdKvBenchmarkConfig {
            layer_count: 0,
            batch: 1,
            attention_heads: 2,
            tokens: 4,
            key_head_dim: 256,
            value_head_dim: 256,
            block_tokens: 2,
            page_size: 2,
        })
        .unwrap_err();

        assert!(err.to_string().contains("layer_count"));
    }

    #[test]
    fn dsa_attention_benchmark_rejects_invalid_selection() {
        let err = validate_dsa_attention_benchmark_config(&DsaAttentionBenchmarkConfig {
            layer_count: 1,
            batch: 1,
            attention_heads: 2,
            tokens: 4,
            selected_tokens: 8,
            key_head_dim: 256,
            value_head_dim: 256,
            page_size: 2,
        })
        .unwrap_err();

        assert!(err.to_string().contains("selected_tokens"));
    }

    #[test]
    fn moe_expert_benchmark_rejects_invalid_q2_width() {
        let err = validate_moe_expert_benchmark_config(&MoeExpertBenchmarkConfig {
            iterations: 1,
            token_count: 1,
            assignment_count: 1,
            hidden_size: 384,
            intermediate_size: 256,
            expert_count: 2,
        })
        .unwrap_err();

        assert!(err.to_string().contains("hidden_size"));
    }

    #[test]
    fn moe_expert_benchmark_runs_on_metal_when_available() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let report = benchmark_moe_multi_expert_decode(
            &backend,
            MoeExpertBenchmarkConfig {
                iterations: 1,
                token_count: 2,
                assignment_count: 3,
                hidden_size: 256,
                intermediate_size: 256,
                expert_count: 2,
            },
        )
        .unwrap();

        assert_eq!(report.iterations, 1);
        assert_eq!(report.assignment_count, 3);
        assert!(report.old_single_assignment_seconds >= 0.0);
        assert!(report.multi_expert_seconds >= 0.0);
    }

    #[test]
    fn ssd_kv_benchmark_runs_hot_load_on_metal_when_available() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };

        let report = benchmark_ssd_kv_decode_hot_load(
            &backend,
            SsdKvBenchmarkConfig {
                layer_count: 1,
                batch: 1,
                attention_heads: 2,
                tokens: 4,
                key_head_dim: 256,
                value_head_dim: 256,
                block_tokens: 2,
                page_size: 2,
            },
        )
        .unwrap();

        assert_eq!(report.layer_count, 1);
        assert_eq!(report.tokens, 4);
        assert!(report.stored_bytes > 0);
        assert!(report.raw_f32_bytes > report.stored_bytes);
    }

    /// Production Metal generation must never silently fall back to host KV.
    /// The explicit debug switch proves that runtime enforcement remains in
    /// place if the resident device path becomes unavailable.
    #[test]
    fn metal_generation_rejects_disabled_resident_device_decode() {
        let Ok(native_backend) = MetalBackend::new() else {
            return;
        };
        let path = write_gguf_model_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();

        let model = Model::open_from_gguf(
            &gguf,
            &config,
            &native_backend,
            DEFAULT_GGUF_OUTPUT_CHUNK_ROWS,
        )
        .unwrap();
        model.disable_device_decode();
        let error = run_generate_token_ids_with_stop_tokens(
            &model,
            &config,
            &native_backend,
            &[1, 2],
            Some(3),
            1,
            &[],
        )
        .expect_err("disabled resident decode must not fall back to host KV");

        assert!(error
            .to_string()
            .contains("fallback to host KV is disabled"));
    }

    #[test]
    fn eight_row_device_verifier_preserves_the_first_causal_row_on_metal() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let path = write_gguf_model_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let model = Model::open_from_gguf(&gguf, &config, &backend, DEFAULT_GGUF_OUTPUT_CHUNK_ROWS)
            .unwrap();

        let single_seed = model
            .prefill_seed_device(&config, &[1], &backend)
            .unwrap()
            .expect("single-row seed device path");
        let mut single_cache = DevicePagedRuntimeCache::new_from_device_seed(
            model.max_context(),
            1,
            &single_seed.layer_kv_cache,
            &backend,
            None,
        )
        .unwrap();
        let single = run_cached_device_token_sequence_without_append(
            &model,
            &config,
            &backend,
            &[2],
            &mut single_cache,
            1,
            "test.single",
            "test.single.model",
        )
        .unwrap();

        let sequence_seed = model
            .prefill_seed_device(&config, &[1], &backend)
            .unwrap()
            .expect("sequence seed device path");
        let mut sequence_cache = DevicePagedRuntimeCache::new_from_device_seed(
            model.max_context(),
            1,
            &sequence_seed.layer_kv_cache,
            &backend,
            None,
        )
        .unwrap();
        let sequence = run_cached_device_token_sequence_without_append(
            &model,
            &config,
            &backend,
            &[2, 3, 4, 5, 6, 7, 1, 2],
            &mut sequence_cache,
            1,
            "test.sequence",
            "test.sequence.model",
        )
        .unwrap();

        let single_hidden = backend
            .device_download_f32_tensor(&single.hidden_states)
            .unwrap();
        let sequence_hidden = backend
            .device_download_f32_tensor(&sequence.hidden_states)
            .unwrap();
        assert_eq!(single_hidden.dims(), &[1, 1, config.hidden_size]);
        assert_eq!(sequence_hidden.dims(), &[1, 8, config.hidden_size]);
        for (hidden_index, (&single_value, &sequence_value)) in single_hidden
            .values()
            .iter()
            .zip(&sequence_hidden.values()[..config.hidden_size])
            .enumerate()
        {
            let tolerance = 1e-4_f32 * single_value.abs().max(sequence_value.abs()).max(1.0);
            assert!(
                (single_value - sequence_value).abs() <= tolerance,
                "first verifier row differs at hidden {hidden_index}: single={single_value}, sequence={sequence_value}"
            );
        }
        assert_eq!(single.token_ids[0], sequence.token_ids[0]);
    }

    #[test]
    fn generate_uses_q2_gguf_model_and_paged_cache() {
        let path = write_gguf_model_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let backend = MetalBackend::reference().unwrap();
        let model = Model::open_from_gguf(&gguf, &config, &backend, DEFAULT_GGUF_OUTPUT_CHUNK_ROWS)
            .unwrap();

        let report =
            run_generate_with_stop_tokens(&model, &config, &backend, &[1, 2], 2, 1, &[]).unwrap();

        assert_eq!(report.page_size, 1);
        assert_eq!(report.prompt_token_count, 2);
        assert_eq!(report.generated_token_ids.len(), 2);
        assert_eq!(report.total_token_count, 4);
        assert_eq!(report.prefill.input_token_count, 2);
        assert_eq!(report.prefill.layer_kv_cache_count, 2);
        assert_eq!(report.prefill.layer_k_cache_shape.dims(), &[1, 1, 2, 256]);
        assert_eq!(report.prefill.layer_v_cache_shape.dims(), &[1, 1, 2, 128]);
        assert_eq!(report.prefill.cache_append.layer_count, 2);
        assert_eq!(report.prefill.cache_append.appended_tokens, 2);
        assert_eq!(report.prefill.cache_append.allocated_pages, 4);
        assert_eq!(report.prefill.hidden_states_shape.dims(), &[1, 2, 256]);
        assert_eq!(report.prefill.logits_shape.dims(), &[1, 8]);
        assert_eq!(report.prefill.model.dense_layer_count, 1);
        assert_eq!(report.prefill.model.sparse_layer_count, 1);
        assert_eq!(report.prefill.model.max_attention_past_tokens, 0);
        assert!(!report.prefill.model.materialized_full_logits);
        assert_eq!(report.prefill.model.output_projection_chunk_count, 1);
        assert!(
            report
                .prefill
                .model
                .output_projection_source_payload_bytes_read
                > 0
        );
        assert!(
            report
                .prefill
                .model
                .output_projection_peak_decoded_f32_bytes
                > 0
        );
        assert!(report.prefill.model.expert_loads.loaded_expert_requests > 0);
        assert_eq!(report.decode.step_count, 1);
        assert_eq!(report.decode.total_appended_tokens, 1);
        assert_eq!(report.decode.total_allocated_pages, 2);
        assert_eq!(report.decode.max_attention_past_tokens, 2);
        assert!(!report.decode.materialized_full_logits);
        assert_eq!(
            report
                .decode
                .last_hidden_states_shape
                .as_ref()
                .unwrap()
                .dims(),
            &[1, 1, 256]
        );
        assert_eq!(
            report.decode.last_logits_shape.as_ref().unwrap().dims(),
            &[1, 8]
        );
        assert!(
            report
                .decode
                .total_output_projection_source_payload_bytes_read
                > 0
        );
        assert!(report.decode.peak_output_projection_decoded_f32_bytes > 0);
        assert_eq!(report.cached_tokens, 3);
        assert_eq!(report.next_position, 3);
        assert_eq!(report.memory.prompt_tokens, 2);
        assert_eq!(report.memory.requested_context_tokens, 4);
        assert_eq!(
            report.memory.prefill_attention_scores_shape.dims(),
            &[1, 2, 2, 2]
        );
        assert_eq!(report.memory.prefill_attention_scores_bytes, 32);
        assert_eq!(
            report.memory.per_layer_k_cache_shape.dims(),
            &[1, 1, 4, 256]
        );
        assert_eq!(
            report.memory.per_layer_v_cache_shape.dims(),
            &[1, 1, 4, 128]
        );
        assert_eq!(report.memory.mla_kv_cache_bytes, 6_144);
        assert_eq!(report.expert_loads.cache_hits, 0);
        assert!(report.expert_loads.cache_misses > 0);
        assert!(report.expert_loads.materialized_expert_bytes_loaded > 0);
        assert!(report.expert_loads.source_expert_bytes_loaded > 0);
    }

    #[test]
    fn generate_token_ids_uses_lean_runtime_path() {
        let path = write_gguf_model_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let backend = MetalBackend::reference().unwrap();
        let model = Model::open_from_gguf(&gguf, &config, &backend, DEFAULT_GGUF_OUTPUT_CHUNK_ROWS)
            .unwrap();

        let report =
            run_generate_with_stop_tokens(&model, &config, &backend, &[1, 2], 2, 1, &[]).unwrap();
        let token_ids = run_generate_token_ids_with_stop_tokens(
            &model,
            &config,
            &backend,
            &[1, 2],
            Some(2),
            1,
            &[],
        )
        .unwrap();

        assert_eq!(token_ids, report.generated_token_ids);

        let mut streamed_token_ids = Vec::new();
        run_generate_streaming_with_stop_tokens(
            &model,
            &config,
            &backend,
            &[1, 2],
            Some(2),
            1,
            &[],
            |token_id| {
                streamed_token_ids.push(token_id);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(streamed_token_ids, token_ids);

        let stopped = run_generate_token_ids_with_stop_tokens(
            &model,
            &config,
            &backend,
            &[1, 2],
            Some(2),
            1,
            &[token_ids[0]],
        )
        .unwrap();

        assert_eq!(stopped, vec![token_ids[0]]);
    }

    #[test]
    fn mtp_requires_two_decode_slots_after_prefill() {
        assert!(!should_enable_mtp(false, 3));
        assert!(!should_enable_mtp(true, 1));
        assert!(!should_enable_mtp(true, 2));
        assert!(should_enable_mtp(true, 3));
    }

    #[test]
    fn mtp_caps_speculative_rows_for_streamed_moe() {
        assert_eq!(mtp_draft_count(0), 0);
        assert_eq!(mtp_draft_count(1), 0);
        assert_eq!(mtp_draft_count(2), 1);
        assert_eq!(mtp_draft_count(3), 2);
        assert_eq!(mtp_draft_count(8), 2);
        assert_eq!(mtp_draft_count(32), 2);
    }

    #[test]
    fn mtp_acceptance_rate_uses_only_verified_drafts() {
        let metrics = MtpMetrics {
            enabled: true,
            verification_passes: 2,
            target_tokens: 6,
            draft_tokens: 4,
            accepted_draft_tokens: 3,
        };

        assert_eq!(metrics.acceptance_rate(), 0.75);
        assert_eq!(MtpMetrics::default().acceptance_rate(), 0.0);
    }

    #[test]
    fn generate_validates_request_shape() {
        let config = tiny_config();
        let err = validate_generate_request(&config, &[1, 2], Some(0), 1)
            .expect_err("zero generated token count must fail");

        assert!(err.to_string().contains("max_new_tokens"));
    }

    #[test]
    fn generate_without_token_limit_uses_remaining_context() {
        let mut config = tiny_config();
        config.max_context = 5;

        let effective = validate_generate_request(&config, &[1, 2], None, 1).unwrap();

        assert_eq!(effective, 3);
    }

    #[test]
    fn generate_without_token_limit_requires_context_room() {
        let mut config = tiny_config();
        config.max_context = 2;

        let err = validate_generate_request(&config, &[1, 2], None, 1)
            .expect_err("full context should leave no room for generation");

        assert!(err.to_string().contains("leaves no room"));
    }

    #[test]
    fn generate_rejects_unsafe_dense_prefill_attention() {
        let mut config = tiny_config();
        config.max_context = 20_000;
        let prompt = vec![1_u32; 17_000];

        let err = validate_generate_request(&config, &prompt, Some(1), 128)
            .expect_err("large dense prefill attention should be rejected");

        assert!(err.to_string().contains("dense prefill attention"));
        assert!(err.to_string().contains("[1, 2, 17000, 17000]"));
    }

    #[test]
    fn generate_allows_large_prompt_with_streaming_prefill() {
        let mut config = tiny_config();
        config.max_context = 20_000;
        let prompt = vec![1_u32; 17_000];

        let effective = validate_generate_request_with_prefill_strategy(
            &config,
            &prompt,
            Some(1),
            128,
            PrefillStrategy::Streaming,
        )
        .unwrap();

        assert_eq!(effective, 1);
    }

    #[test]
    fn generate_allows_large_kv_when_ssd_cold_tier_is_available() {
        let mut config = tiny_config();
        config.num_layers = 78;
        config.attention_heads = 64;
        config.max_context = 5_000;
        let prompt = vec![1_u32; 16];

        let effective = validate_generate_request(&config, &prompt, Some(3_000), 128).unwrap();

        assert_eq!(effective, 3_000);
    }

    fn write_gguf_model_fixture(ty: GgmlType) -> PathBuf {
        let path = unique_temp_file("runtime-gguf-model");
        let mut specs = vec![
            TensorSpec::quant("token_embd.weight", vec![256, 8], ty),
            TensorSpec::f32("output_norm.weight", vec![256], None),
            TensorSpec::quant("output.weight", vec![256, 8], ty),
        ];
        insert_dense_layer(&mut specs, 0, ty);
        insert_sparse_layer(&mut specs, 1, ty);
        write_specs_gguf(path, &specs)
    }

    fn insert_dense_layer(specs: &mut Vec<TensorSpec>, layer: usize, ty: GgmlType) {
        insert_attention(specs, layer, ty);
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_gate.weight"),
            vec![256, 512],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_up.weight"),
            vec![256, 512],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_down.weight"),
            vec![512, 256],
            ty,
        ));
    }

    fn insert_sparse_layer(specs: &mut Vec<TensorSpec>, layer: usize, ty: GgmlType) {
        insert_attention(specs, layer, ty);
        specs.push(TensorSpec::f32(
            format!("blk.{layer}.exp_probs_b.bias"),
            vec![4],
            Some(vec![0.0, 0.4, 0.2, 0.8]),
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_gate_inp.weight"),
            vec![256, 4],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_gate_shexp.weight"),
            vec![256, 256],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_up_shexp.weight"),
            vec![256, 256],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_down_shexp.weight"),
            vec![256, 256],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_gate_exps.weight"),
            vec![256, 256, 4],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_up_exps.weight"),
            vec![256, 256, 4],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_down_exps.weight"),
            vec![256, 256, 4],
            ty,
        ));
    }

    fn insert_attention(specs: &mut Vec<TensorSpec>, layer: usize, ty: GgmlType) {
        specs.push(TensorSpec::f32(
            format!("blk.{layer}.attn_norm.weight"),
            vec![256],
            None,
        ));
        specs.push(TensorSpec::f32(
            format!("blk.{layer}.ffn_norm.weight"),
            vec![256],
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_q_a.weight"),
            vec![256, 256],
            ty,
        ));
        specs.push(TensorSpec::f32(
            format!("blk.{layer}.attn_q_a_norm.weight"),
            vec![256],
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_q_b.weight"),
            vec![256, 512],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_kv_a_mqa.weight"),
            vec![256, 384],
            ty,
        ));
        specs.push(TensorSpec::f32(
            format!("blk.{layer}.attn_kv_a_norm.weight"),
            vec![256],
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_k_b.weight"),
            vec![128, 256, 2],
            GgmlType::Q8_0,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_v_b.weight"),
            vec![256, 256, 2],
            GgmlType::Q8_0,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_output.weight"),
            vec![512, 256],
            ty,
        ));
    }

    #[derive(Debug)]
    struct TensorSpec {
        name: String,
        dims: Vec<u64>,
        ty: GgmlType,
        values: Option<Vec<f32>>,
    }

    impl TensorSpec {
        fn f32(name: impl Into<String>, dims: Vec<u64>, values: Option<Vec<f32>>) -> Self {
            Self {
                name: name.into(),
                dims,
                ty: GgmlType::F32,
                values,
            }
        }

        fn quant(name: impl Into<String>, dims: Vec<u64>, ty: GgmlType) -> Self {
            Self {
                name: name.into(),
                dims,
                ty,
                values: None,
            }
        }

        fn payload_len(&self) -> u64 {
            match self.ty {
                GgmlType::F32 => self.dims.iter().product::<u64>() * 4,
                GgmlType::Q2K => self.dims.iter().product::<u64>() / 256 * GGML_Q2_K_BLOCK_BYTES,
                GgmlType::Q8_0 => self.dims.iter().product::<u64>() / 32 * GGML_Q8_0_BLOCK_BYTES,
                other => panic!("unsupported fixture tensor type {other}"),
            }
        }
    }

    fn write_specs_gguf(path: PathBuf, specs: &[TensorSpec]) -> PathBuf {
        let mut writer = GgufWriter::new();
        writer.header(specs.len() as u64, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);

        let mut offset = 0_u64;
        let offsets = specs
            .iter()
            .map(|spec| {
                let current = offset;
                offset = align_u64(current + spec.payload_len(), 32);
                current
            })
            .collect::<Vec<_>>();
        for (spec, offset) in specs.iter().zip(offsets.iter().copied()) {
            writer.tensor_info(&spec.name, &spec.dims, spec.ty, offset);
        }
        writer.pad_to(32);

        for (spec, offset) in specs.iter().zip(offsets.iter().copied()) {
            writer.pad_to_absolute_data_offset(offset);
            match spec.ty {
                GgmlType::F32 => {
                    let element_count = spec.dims.iter().product::<u64>() as usize;
                    let values = spec
                        .values
                        .clone()
                        .unwrap_or_else(|| vec![1.0_f32; element_count]);
                    assert_eq!(values.len(), element_count);
                    for value in values {
                        writer.bytes(&value.to_le_bytes());
                    }
                }
                GgmlType::Q2K | GgmlType::Q8_0 => {
                    writer.bytes(&vec![0_u8; spec.payload_len() as usize]);
                }
                other => panic!("unsupported fixture tensor type {other}"),
            }
        }
        writer.pad_to_absolute_data_offset(offset);
        writer.finish_to(path)
    }

    struct GgufWriter {
        bytes: Vec<u8>,
        data_start: Option<usize>,
    }

    impl GgufWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                data_start: None,
            }
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
            self.data_start = Some(self.bytes.len());
        }

        fn pad_to_absolute_data_offset(&mut self, offset: u64) {
            let target = self.data_start.unwrap() + offset as usize;
            if self.bytes.len() < target {
                self.bytes.resize(target, 0);
            }
        }

        fn bytes(&mut self, bytes: &[u8]) {
            self.bytes.extend_from_slice(bytes);
        }

        fn finish_to(self, path: PathBuf) -> PathBuf {
            fs::write(&path, self.bytes).unwrap();
            path
        }
    }

    fn align_u64(value: u64, alignment: u64) -> u64 {
        let remainder = value % alignment;
        if remainder == 0 {
            value
        } else {
            value + alignment - remainder
        }
    }

    fn unique_temp_file(label: &str) -> PathBuf {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("runtime-{label}-{}-{id}", std::process::id()))
    }

    fn tiny_layer_kv_cache(tokens: usize) -> Vec<LayerKvCacheTensors> {
        vec![LayerKvCacheTensors {
            layer_index: 0,
            layer_kind: model::LayerKind::Dense,
            cache_k: F32Tensor::zeros([1, 2, tokens, 256]).unwrap(),
            cache_v: F32Tensor::zeros([1, 2, tokens, 256]).unwrap(),
            index_key: None,
        }]
    }

    fn tiny_config() -> Config {
        Config {
            model_type: "glm_moe_dsa".to_string(),
            hidden_size: 256,
            num_layers: 2,
            dense_layers: 1,
            sparse_moe_layers: Some(1),
            vocab_size: 8,
            attention_heads: 2,
            qk_head_dim: 256,
            qk_no_rope_dim: 128,
            qk_rope_dim: 128,
            kv_lora_rank: 256,
            v_head_dim: Some(256),
            num_routed_experts: 4,
            experts_per_token: 2,
            max_context: 128,
            dsa_index_topk: 16,
            index_head_dim: 128,
            index_n_heads: 32,
            index_topk_freq: 4,
            index_skip_topk_offset: 3,
            index_share_for_mtp_iteration: true,
            indexer_rope_interleave: true,
            indexer_types: Vec::new(),
            moe_intermediate_size: 256,
            num_shared_experts: 1,
            moe_groups: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 2.5,
            scoring_func: "sigmoid".to_string(),
            topk_method: "noaux_tc".to_string(),
            num_nextn_predict_layers: 0,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
        .validated()
        .unwrap()
    }
}
