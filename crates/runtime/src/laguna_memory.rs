use std::{
    fs::File,
    io::Write,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex, OnceLock,
    },
    time::Duration,
};

use common::{Error, Result};
use model::LagunaExpertCacheMetrics;

use crate::telemetry::RuntimeMemorySnapshot;

pub const DEFAULT_LAGUNA_MEMORY_DECISION_WINDOW_TOKENS: usize = 32;
pub const DEFAULT_LAGUNA_MEMORY_TRIAL_WARMUP_TOKENS: usize = 8;
pub const DEFAULT_LAGUNA_MEMORY_STABILIZATION_WINDOWS: usize = 1;
pub const DEFAULT_LAGUNA_MEMORY_TARGET_HEADROOM_BYTES: u64 = 4_000_000_000;
pub const DEFAULT_LAGUNA_MEMORY_HARD_HEADROOM_BYTES: u64 = 1_500_000_000;

const MIN_EXPERT_SSD_READ_BYTES_PER_TOKEN_FOR_GROWTH: u64 = 128 * 1024 * 1024;
const MIN_TRIAL_THROUGHPUT_IMPROVEMENT: f64 = 0.10;
// A completed A/B trial is expensive. After keeping a faster size, or after
// proving both adjacent sizes slower, hold the local optimum for 2,048 decode
// tokens before probing again. Memory-pressure shrink bypasses these cooldowns.
const LOCAL_OPTIMUM_HOLD_WINDOWS: usize = 64;
const PRESSURE_SWAP_GROWTH_BYTES: u64 = 16 * 1024 * 1024;
const PRESSURE_COMPRESSION_GROWTH_BYTES: u64 = 256 * 1024 * 1024;
const PRESSURE_METAL_HEADROOM_BYTES: u64 = 512 * 1024 * 1024;

static LAGUNA_MEMORY_CONTROLLER_FILE: OnceLock<Mutex<Option<File>>> = OnceLock::new();
static LAGUNA_MEMORY_CONTROLLER_FILE_ENABLED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LagunaMemoryControllerSpec {
    pub initial_expert_capacity: usize,
    pub minimum_expert_capacity: usize,
    pub maximum_expert_capacity: usize,
    pub expert_capacity_step: usize,
    pub bytes_per_expert: u64,
    pub decision_window_tokens: usize,
    pub trial_warmup_tokens: usize,
    pub stabilization_windows: usize,
    pub target_headroom_bytes: u64,
    pub hard_headroom_bytes: u64,
}

impl LagunaMemoryControllerSpec {
    pub fn validate(self) -> Result<()> {
        if self.minimum_expert_capacity == 0
            || self.minimum_expert_capacity > self.initial_expert_capacity
            || self.initial_expert_capacity > self.maximum_expert_capacity
        {
            return Err(Error::runtime(format!(
                "Laguna adaptive expert capacities must satisfy 0 < minimum <= initial <= maximum, got {}/{}/{}",
                self.minimum_expert_capacity,
                self.initial_expert_capacity,
                self.maximum_expert_capacity
            )));
        }
        if self.expert_capacity_step == 0 {
            return Err(Error::runtime(
                "Laguna adaptive expert-capacity step must be positive",
            ));
        }
        if self.bytes_per_expert == 0 {
            return Err(Error::runtime(
                "Laguna adaptive bytes per expert must be positive",
            ));
        }
        if self.decision_window_tokens == 0 || self.trial_warmup_tokens == 0 {
            return Err(Error::runtime(
                "Laguna adaptive decision and warmup windows must be positive",
            ));
        }
        if self.hard_headroom_bytes == 0 || self.hard_headroom_bytes >= self.target_headroom_bytes {
            return Err(Error::runtime(format!(
                "Laguna adaptive headroom must satisfy 0 < hard < target, got hard={} target={}",
                self.hard_headroom_bytes, self.target_headroom_bytes
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LagunaMemoryAction {
    Hold,
    GrowTrial,
    WarmupComplete,
    CandidateMeasured,
    ConfirmationWarmupComplete,
    KeepTrial,
    RollbackTrial,
    PressureShrink,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LagunaMemoryPressure {
    None,
    CriticalHeadroom,
    MetalWorkingSet,
    SwapGrowth,
    CompressionGrowth,
}

impl LagunaMemoryPressure {
    fn is_pressure(self) -> bool {
        self != Self::None
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LagunaMemoryDecision {
    pub sequence: usize,
    pub context_tokens: usize,
    pub action: LagunaMemoryAction,
    pub pressure: LagunaMemoryPressure,
    pub previous_expert_capacity: usize,
    pub next_expert_capacity: usize,
    pub window_tokens: usize,
    pub window_duration: Duration,
    pub measured_tokens_per_second: f64,
    pub comparison_tokens_per_second: Option<f64>,
    pub trial_candidate_tokens_per_second: Option<f64>,
    pub expert_lookups: u64,
    pub expert_hits: u64,
    pub expert_ssd_read_bytes: u64,
}

impl LagunaMemoryDecision {
    pub fn changes_capacity(self) -> bool {
        self.previous_expert_capacity != self.next_expert_capacity
    }

    pub fn expert_hit_rate(self) -> f64 {
        rate(self.expert_hits, self.expert_lookups)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LagunaMemoryControllerReport {
    pub current_expert_capacity: usize,
    pub decisions: usize,
    pub rebalances: usize,
    pub trials_started: usize,
    pub trials_kept: usize,
    pub trials_rolled_back: usize,
    pub pressure_events: usize,
    pub last_measured_tokens_per_second: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CapacityTrial {
    previous_capacity: usize,
    candidate_capacity: usize,
    baseline_tokens_per_second: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CapacityConfirmation {
    trial: CapacityTrial,
    candidate_tokens_per_second: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ControllerPhase {
    Baseline,
    CandidateWarmup(CapacityTrial),
    CandidateMeasure(CapacityTrial),
    ConfirmationWarmup(CapacityConfirmation),
    ConfirmationMeasure(CapacityConfirmation),
}

impl ControllerPhase {
    fn trial(self) -> Option<CapacityTrial> {
        match self {
            Self::Baseline => None,
            Self::CandidateWarmup(trial) | Self::CandidateMeasure(trial) => Some(trial),
            Self::ConfirmationWarmup(confirmation) | Self::ConfirmationMeasure(confirmation) => {
                Some(confirmation.trial)
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct LagunaMemoryController {
    spec: LagunaMemoryControllerSpec,
    current_capacity: usize,
    phase: ControllerPhase,
    window_tokens: usize,
    window_elapsed_nanoseconds: u64,
    window_expert_lookups: u64,
    window_expert_hits: u64,
    window_expert_ssd_read_bytes: u64,
    segment_experts: LagunaExpertCacheMetrics,
    decode_segment_active: bool,
    pressure_reference_swap_bytes: Option<u64>,
    pressure_reference_compressed_bytes: Option<u64>,
    rejected_growth_at: Option<usize>,
    rejected_retry_windows: usize,
    kept_capacity_hold_windows: usize,
    stabilization_windows_remaining: usize,
    report: LagunaMemoryControllerReport,
}

impl LagunaMemoryController {
    pub(crate) fn new(spec: LagunaMemoryControllerSpec) -> Result<Self> {
        spec.validate()?;
        Ok(Self {
            current_capacity: spec.initial_expert_capacity,
            spec,
            phase: ControllerPhase::Baseline,
            window_tokens: 0,
            window_elapsed_nanoseconds: 0,
            window_expert_lookups: 0,
            window_expert_hits: 0,
            window_expert_ssd_read_bytes: 0,
            segment_experts: LagunaExpertCacheMetrics::default(),
            decode_segment_active: false,
            pressure_reference_swap_bytes: None,
            pressure_reference_compressed_bytes: None,
            rejected_growth_at: None,
            rejected_retry_windows: 0,
            kept_capacity_hold_windows: 0,
            stabilization_windows_remaining: spec.stabilization_windows,
            report: LagunaMemoryControllerReport {
                current_expert_capacity: spec.initial_expert_capacity,
                ..LagunaMemoryControllerReport::default()
            },
        })
    }

    #[cfg(test)]
    fn current_capacity(&self) -> usize {
        self.current_capacity
    }

    pub(crate) fn begin_decode(
        &mut self,
        experts: LagunaExpertCacheMetrics,
        memory: RuntimeMemorySnapshot,
    ) {
        debug_assert!(!self.decode_segment_active);
        self.segment_experts = experts;
        self.decode_segment_active = true;
        if self.pressure_reference_swap_bytes.is_none() {
            self.pressure_reference_swap_bytes = memory.swap_used_bytes;
        }
        if self.pressure_reference_compressed_bytes.is_none() {
            self.pressure_reference_compressed_bytes = memory.system_compressed_bytes;
        }
    }

    pub(crate) fn record_decode_step(&mut self, elapsed: Duration) {
        debug_assert!(self.decode_segment_active);
        self.window_tokens = self.window_tokens.saturating_add(1);
        self.window_elapsed_nanoseconds = self
            .window_elapsed_nanoseconds
            .saturating_add(elapsed_nanoseconds_u64(elapsed));
    }

    /// Pauses a decode-only measurement window at a request boundary.
    ///
    /// Decode-only samples continue across requests. Prompt prefill is excluded
    /// because each new decode segment records its expert-counter baseline only
    /// after prefill has completed.
    pub(crate) fn pause_decode(&mut self, experts: LagunaExpertCacheMetrics) {
        if !self.decode_segment_active {
            return;
        }
        self.accumulate_segment_experts(experts);
        self.decode_segment_active = false;
    }

    pub(crate) fn observation_due(&self) -> bool {
        let required_tokens = match self.phase {
            ControllerPhase::CandidateWarmup(_) | ControllerPhase::ConfirmationWarmup(_) => {
                self.spec.trial_warmup_tokens
            }
            ControllerPhase::Baseline
            | ControllerPhase::CandidateMeasure(_)
            | ControllerPhase::ConfirmationMeasure(_) => self.spec.decision_window_tokens,
        };
        self.window_tokens >= required_tokens
    }

    pub(crate) fn observe(
        &mut self,
        context_tokens: usize,
        experts: LagunaExpertCacheMetrics,
        memory: RuntimeMemorySnapshot,
    ) -> Result<LagunaMemoryDecision> {
        debug_assert!(self.decode_segment_active);
        if !self.observation_due() {
            return Err(Error::runtime(
                "Laguna memory controller observed an incomplete measurement window",
            ));
        }

        let pressure = self.memory_pressure(memory);
        let measured_tps = tokens_per_second(self.window_tokens, self.window_elapsed_nanoseconds);
        self.accumulate_segment_experts(experts);
        let lookups = self.window_expert_lookups;
        let hits = self.window_expert_hits;
        let ssd_read_bytes = self.window_expert_ssd_read_bytes;
        let previous_capacity = self.current_capacity;

        let (action, next_capacity, comparison_tps, candidate_tps) = if pressure.is_pressure() {
            let active_trial = self.phase.trial();
            let pressure_base =
                active_trial.map_or(self.current_capacity, |trial| trial.previous_capacity);
            let capacity_step_target = pressure_base
                .saturating_sub(self.spec.expert_capacity_step)
                .max(self.spec.minimum_expert_capacity);
            let resident_step_target = experts
                .resident_experts
                .saturating_sub(self.spec.expert_capacity_step)
                .max(self.spec.minimum_expert_capacity);
            let next = capacity_step_target.min(resident_step_target);
            if next != self.current_capacity {
                self.current_capacity = next;
                self.report.rebalances = self.report.rebalances.saturating_add(1);
            }
            self.phase = ControllerPhase::Baseline;
            self.rejected_growth_at = Some(next);
            self.rejected_retry_windows = LOCAL_OPTIMUM_HOLD_WINDOWS;
            self.kept_capacity_hold_windows = 0;
            self.report.pressure_events = self.report.pressure_events.saturating_add(1);
            if active_trial.is_some() {
                self.report.trials_rolled_back = self.report.trials_rolled_back.saturating_add(1);
            }
            (
                if next == previous_capacity {
                    LagunaMemoryAction::Hold
                } else {
                    LagunaMemoryAction::PressureShrink
                },
                next,
                None,
                None,
            )
        } else {
            match self.phase {
                ControllerPhase::CandidateWarmup(trial) => {
                    self.phase = ControllerPhase::CandidateMeasure(trial);
                    (
                        LagunaMemoryAction::WarmupComplete,
                        self.current_capacity,
                        Some(trial.baseline_tokens_per_second),
                        None,
                    )
                }
                ControllerPhase::CandidateMeasure(trial) => {
                    let confirmation = CapacityConfirmation {
                        trial,
                        candidate_tokens_per_second: measured_tps,
                    };
                    self.current_capacity = trial.previous_capacity;
                    self.phase = ControllerPhase::ConfirmationWarmup(confirmation);
                    self.report.rebalances = self.report.rebalances.saturating_add(1);
                    (
                        LagunaMemoryAction::CandidateMeasured,
                        trial.previous_capacity,
                        Some(trial.baseline_tokens_per_second),
                        Some(measured_tps),
                    )
                }
                ControllerPhase::ConfirmationWarmup(confirmation) => {
                    self.phase = ControllerPhase::ConfirmationMeasure(confirmation);
                    (
                        LagunaMemoryAction::ConfirmationWarmupComplete,
                        self.current_capacity,
                        Some(confirmation.trial.baseline_tokens_per_second),
                        Some(confirmation.candidate_tokens_per_second),
                    )
                }
                ControllerPhase::ConfirmationMeasure(confirmation) => {
                    let trial = confirmation.trial;
                    let confirmed_baseline_tps = measured_tps;
                    let required_tps = confirmed_baseline_tps.max(trial.baseline_tokens_per_second)
                        * (1.0 + MIN_TRIAL_THROUGHPUT_IMPROVEMENT);
                    if confirmation.candidate_tokens_per_second >= required_tps {
                        self.current_capacity = trial.candidate_capacity;
                        self.report.rebalances = self.report.rebalances.saturating_add(1);
                        self.report.trials_kept = self.report.trials_kept.saturating_add(1);
                        self.kept_capacity_hold_windows = LOCAL_OPTIMUM_HOLD_WINDOWS;
                        self.rejected_growth_at = None;
                        self.phase = ControllerPhase::Baseline;
                        (
                            LagunaMemoryAction::KeepTrial,
                            trial.candidate_capacity,
                            Some(confirmed_baseline_tps),
                            Some(confirmation.candidate_tokens_per_second),
                        )
                    } else {
                        self.report.trials_rolled_back =
                            self.report.trials_rolled_back.saturating_add(1);
                        self.rejected_growth_at = Some(trial.previous_capacity);
                        self.rejected_retry_windows = LOCAL_OPTIMUM_HOLD_WINDOWS;
                        self.phase = ControllerPhase::Baseline;
                        (
                            LagunaMemoryAction::RollbackTrial,
                            trial.previous_capacity,
                            Some(confirmed_baseline_tps),
                            Some(confirmation.candidate_tokens_per_second),
                        )
                    }
                }
                ControllerPhase::Baseline => {
                    if self.stabilization_windows_remaining > 0 {
                        self.stabilization_windows_remaining -= 1;
                        (LagunaMemoryAction::Hold, self.current_capacity, None, None)
                    } else {
                        let should_grow = self.should_start_growth_trial(
                            bytes_per_token(ssd_read_bytes, self.window_tokens),
                            experts.resident_experts,
                            memory,
                        );
                        if should_grow {
                            let candidate_capacity = self
                                .current_capacity
                                .saturating_add(self.spec.expert_capacity_step)
                                .min(self.spec.maximum_expert_capacity);
                            let trial = CapacityTrial {
                                previous_capacity: self.current_capacity,
                                candidate_capacity,
                                baseline_tokens_per_second: measured_tps,
                            };
                            self.current_capacity = candidate_capacity;
                            self.phase = ControllerPhase::CandidateWarmup(trial);
                            self.report.rebalances = self.report.rebalances.saturating_add(1);
                            self.report.trials_started =
                                self.report.trials_started.saturating_add(1);
                            (
                                LagunaMemoryAction::GrowTrial,
                                candidate_capacity,
                                None,
                                None,
                            )
                        } else {
                            (LagunaMemoryAction::Hold, self.current_capacity, None, None)
                        }
                    }
                }
            }
        };

        self.report.decisions = self.report.decisions.saturating_add(1);
        self.report.current_expert_capacity = self.current_capacity;
        self.report.last_measured_tokens_per_second = Some(measured_tps);
        let decision = LagunaMemoryDecision {
            sequence: self.report.decisions,
            context_tokens,
            action,
            pressure,
            previous_expert_capacity: previous_capacity,
            next_expert_capacity: next_capacity,
            window_tokens: self.window_tokens,
            window_duration: Duration::from_nanos(self.window_elapsed_nanoseconds),
            measured_tokens_per_second: measured_tps,
            comparison_tokens_per_second: comparison_tps,
            trial_candidate_tokens_per_second: candidate_tps,
            expert_lookups: lookups,
            expert_hits: hits,
            expert_ssd_read_bytes: ssd_read_bytes,
        };
        self.reset_window(experts, memory);
        Ok(decision)
    }

    pub(crate) fn report(&self) -> LagunaMemoryControllerReport {
        self.report
    }

    pub(crate) fn record_decision(
        &self,
        decision: LagunaMemoryDecision,
        memory: RuntimeMemorySnapshot,
    ) -> Result<()> {
        tracing::info!(
            target: "inferno::laguna_memory_controller",
            decision = decision.sequence,
            context_tokens = decision.context_tokens,
            action = ?decision.action,
            pressure = ?decision.pressure,
            previous_expert_capacity = decision.previous_expert_capacity,
            next_expert_capacity = decision.next_expert_capacity,
            window_tokens = decision.window_tokens,
            measured_tokens_per_second = decision.measured_tokens_per_second,
            comparison_tokens_per_second = ?decision.comparison_tokens_per_second,
            trial_candidate_tokens_per_second = ?decision.trial_candidate_tokens_per_second,
            expert_hit_rate = decision.expert_hit_rate(),
            expert_ssd_read_gb = decision.expert_ssd_read_bytes as f64 / 1_000_000_000.0,
            effective_available_gb = memory.effective_available_bytes().map(|bytes| bytes as f64 / 1_000_000_000.0),
            metal_headroom_gb = memory.metal_headroom_bytes().map(|bytes| bytes as f64 / 1_000_000_000.0),
            swap_used_gb = memory.swap_used_bytes.map(|bytes| bytes as f64 / 1_000_000_000.0),
            compressed_gb = memory.system_compressed_bytes.map(|bytes| bytes as f64 / 1_000_000_000.0),
            "Laguna adaptive expert-cache decision"
        );

        if !LAGUNA_MEMORY_CONTROLLER_FILE_ENABLED.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut writer = laguna_memory_controller_file()
            .lock()
            .map_err(|_| Error::runtime("Laguna memory controller file lock poisoned"))?;
        let Some(writer) = writer.as_mut() else {
            return Ok(());
        };
        writeln!(
            writer,
            "{}\t{}\t{:?}\t{:?}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{}\t{}",
            decision.sequence,
            decision.context_tokens,
            decision.action,
            decision.pressure,
            decision.previous_expert_capacity,
            decision.next_expert_capacity,
            decision.window_tokens,
            decision.window_duration.as_secs_f64(),
            decision.measured_tokens_per_second,
            decision
                .comparison_tokens_per_second
                .map(|value| value.to_string())
                .unwrap_or_default(),
            decision
                .trial_candidate_tokens_per_second
                .map(|value| value.to_string())
                .unwrap_or_default(),
            decision.expert_lookups,
            decision.expert_hits,
            decision.expert_hit_rate(),
            decision.expert_ssd_read_bytes,
            memory.effective_available_bytes().unwrap_or(0),
            memory.metal_headroom_bytes().unwrap_or(0),
            memory.swap_used_bytes.unwrap_or(0),
        )
        .map_err(|source| {
            Error::runtime(format!(
                "Laguna memory controller log write failed: {source}"
            ))
        })?;
        writer.flush().map_err(|source| {
            Error::runtime(format!(
                "Laguna memory controller log flush failed: {source}"
            ))
        })
    }

    fn should_start_growth_trial(
        &mut self,
        expert_ssd_read_bytes_per_token: u64,
        resident_experts: usize,
        memory: RuntimeMemorySnapshot,
    ) -> bool {
        if self.kept_capacity_hold_windows > 0 {
            self.kept_capacity_hold_windows -= 1;
            return false;
        }
        if self.rejected_retry_windows > 0 {
            self.rejected_retry_windows -= 1;
            if self.rejected_retry_windows == 0 {
                self.rejected_growth_at = None;
            }
            return false;
        }

        let growth_rejected = self.rejected_growth_at == Some(self.current_capacity);
        let can_grow = self.current_capacity < self.spec.maximum_expert_capacity
            && resident_experts >= self.current_capacity
            && self.can_grow(memory)
            && !growth_rejected;
        expert_ssd_read_bytes_per_token >= MIN_EXPERT_SSD_READ_BYTES_PER_TOKEN_FOR_GROWTH
            && can_grow
    }

    fn can_grow(&self, memory: RuntimeMemorySnapshot) -> bool {
        let capacity_delta = u64::try_from(self.spec.expert_capacity_step)
            .unwrap_or(u64::MAX)
            .saturating_mul(self.spec.bytes_per_expert);
        memory.effective_available_bytes().is_some_and(|bytes| {
            bytes
                >= self
                    .spec
                    .target_headroom_bytes
                    .saturating_add(capacity_delta)
        }) && memory.metal_headroom_bytes().is_none_or(|bytes| {
            bytes >= PRESSURE_METAL_HEADROOM_BYTES.saturating_add(capacity_delta)
        })
    }

    fn memory_pressure(&self, memory: RuntimeMemorySnapshot) -> LagunaMemoryPressure {
        let swap_growth = memory.swap_used_bytes.unwrap_or(0).saturating_sub(
            self.pressure_reference_swap_bytes
                .unwrap_or(memory.swap_used_bytes.unwrap_or(0)),
        );
        if swap_growth >= PRESSURE_SWAP_GROWTH_BYTES {
            return LagunaMemoryPressure::SwapGrowth;
        }
        let compressed_growth = memory.system_compressed_bytes.unwrap_or(0).saturating_sub(
            self.pressure_reference_compressed_bytes
                .unwrap_or(memory.system_compressed_bytes.unwrap_or(0)),
        );
        if compressed_growth >= PRESSURE_COMPRESSION_GROWTH_BYTES
            && memory
                .effective_available_bytes()
                .is_some_and(|bytes| bytes < self.spec.target_headroom_bytes)
        {
            return LagunaMemoryPressure::CompressionGrowth;
        }
        if memory
            .effective_available_bytes()
            .is_some_and(|bytes| bytes < self.spec.hard_headroom_bytes)
        {
            return LagunaMemoryPressure::CriticalHeadroom;
        }
        if memory
            .metal_headroom_bytes()
            .is_some_and(|bytes| bytes < PRESSURE_METAL_HEADROOM_BYTES)
        {
            return LagunaMemoryPressure::MetalWorkingSet;
        }
        LagunaMemoryPressure::None
    }

    fn reset_window(&mut self, experts: LagunaExpertCacheMetrics, memory: RuntimeMemorySnapshot) {
        self.window_tokens = 0;
        self.window_elapsed_nanoseconds = 0;
        self.window_expert_lookups = 0;
        self.window_expert_hits = 0;
        self.window_expert_ssd_read_bytes = 0;
        self.segment_experts = experts;
        self.pressure_reference_swap_bytes = memory.swap_used_bytes;
        self.pressure_reference_compressed_bytes = memory.system_compressed_bytes;
    }

    fn accumulate_segment_experts(&mut self, experts: LagunaExpertCacheMetrics) {
        self.window_expert_lookups = self
            .window_expert_lookups
            .saturating_add(experts.lookups.saturating_sub(self.segment_experts.lookups));
        self.window_expert_hits = self
            .window_expert_hits
            .saturating_add(experts.hits.saturating_sub(self.segment_experts.hits));
        self.window_expert_ssd_read_bytes = self.window_expert_ssd_read_bytes.saturating_add(
            experts
                .ssd_read_bytes
                .saturating_sub(self.segment_experts.ssd_read_bytes),
        );
        self.segment_experts = experts;
    }
}

pub fn enable_laguna_memory_controller_log(path: &Path) -> Result<()> {
    let mut file = File::create(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    writeln!(
        file,
        "decision\tcontext_tokens\taction\tpressure\tprevious_expert_capacity\tnext_expert_capacity\twindow_tokens\twindow_seconds\tmeasured_tps\tcomparison_tps\ttrial_candidate_tps\texpert_lookups\texpert_hits\texpert_hit_rate\texpert_ssd_read_bytes\teffective_available_bytes\tmetal_headroom_bytes\tswap_used_bytes"
    )
    .map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = laguna_memory_controller_file()
        .lock()
        .map_err(|_| Error::runtime("Laguna memory controller file lock poisoned"))?;
    if writer.is_some() {
        return Err(Error::runtime(
            "Laguna memory controller decision logging is already enabled",
        ));
    }
    *writer = Some(file);
    LAGUNA_MEMORY_CONTROLLER_FILE_ENABLED.store(true, Ordering::Release);
    Ok(())
}

fn laguna_memory_controller_file() -> &'static Mutex<Option<File>> {
    LAGUNA_MEMORY_CONTROLLER_FILE.get_or_init(|| Mutex::new(None))
}

fn tokens_per_second(tokens: usize, elapsed_nanoseconds: u64) -> f64 {
    if tokens == 0 || elapsed_nanoseconds == 0 {
        0.0
    } else {
        tokens as f64 * 1_000_000_000.0 / elapsed_nanoseconds as f64
    }
}

fn elapsed_nanoseconds_u64(elapsed: Duration) -> u64 {
    u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX)
}

fn bytes_per_token(bytes: u64, tokens: usize) -> u64 {
    u64::try_from(tokens)
        .ok()
        .filter(|tokens| *tokens > 0)
        .map_or(0, |tokens| bytes / tokens)
}

fn rate(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> LagunaMemoryControllerSpec {
        LagunaMemoryControllerSpec {
            initial_expert_capacity: 24,
            minimum_expert_capacity: 16,
            maximum_expert_capacity: 32,
            expert_capacity_step: 1,
            bytes_per_expert: 1_000_000,
            decision_window_tokens: 2,
            trial_warmup_tokens: 1,
            stabilization_windows: 0,
            target_headroom_bytes: 4_000_000_000,
            hard_headroom_bytes: 1_000_000_000,
        }
    }

    fn memory(available: u64, swap: u64, compressed: u64) -> RuntimeMemorySnapshot {
        RuntimeMemorySnapshot {
            total_physical_bytes: Some(64_000_000_000),
            process_rss_bytes: None,
            process_virtual_bytes: None,
            system_free_bytes: Some(available),
            system_active_bytes: None,
            system_inactive_bytes: Some(0),
            system_wired_bytes: None,
            system_compressed_bytes: Some(compressed),
            system_purgeable_bytes: Some(0),
            system_speculative_bytes: Some(0),
            swap_used_bytes: Some(swap),
            metal_current_allocated_bytes: Some(10_000_000_000),
            metal_recommended_max_working_set_bytes: Some(60_000_000_000),
            runtime_kv_hot_bytes: None,
            runtime_kv_cold_bytes: None,
        }
    }

    fn metrics(lookups: u64, hits: u64, bytes: u64) -> LagunaExpertCacheMetrics {
        LagunaExpertCacheMetrics {
            lookups,
            hits,
            misses: lookups - hits,
            resident_loads: lookups - hits,
            ssd_read_bytes: bytes,
            resident_load_bytes: bytes,
            resident_experts: 24,
            resident_bytes: 24_000,
            capacity_experts: 32,
            capacity_bytes: 32_000,
            ..LagunaExpertCacheMetrics::default()
        }
    }

    fn add_steps(controller: &mut LagunaMemoryController, count: usize, milliseconds: u64) {
        for _ in 0..count {
            controller.record_decode_step(Duration::from_millis(milliseconds));
        }
    }

    #[test]
    fn retains_growth_only_when_measured_throughput_improves() {
        let mut controller = LagunaMemoryController::new(spec()).unwrap();
        let memory = memory(20_000_000_000, 0, 0);
        controller.begin_decode(metrics(0, 0, 0), memory);

        // Request 1 measures the baseline, resizes, and completes candidate
        // warmup with its remaining decode token.
        add_steps(&mut controller, 2, 100);
        let grow = controller
            .observe(10, metrics(20, 5, 300_000_000), memory)
            .unwrap();
        assert_eq!(grow.action, LagunaMemoryAction::GrowTrial);
        assert_eq!(grow.next_expert_capacity, 25);
        add_steps(&mut controller, 1, 100);
        let warmup = controller
            .observe(11, metrics(30, 10, 400_000_000), memory)
            .unwrap();
        assert_eq!(warmup.action, LagunaMemoryAction::WarmupComplete);
        controller.pause_decode(metrics(30, 10, 400_000_000));

        // Request 2 measures the candidate, restores the baseline, and warms
        // that baseline with its remaining token.
        controller.begin_decode(metrics(30, 10, 400_000_000), memory);
        add_steps(&mut controller, 2, 90);
        let candidate = controller
            .observe(13, metrics(50, 25, 600_000_000), memory)
            .unwrap();
        assert_eq!(candidate.action, LagunaMemoryAction::CandidateMeasured);
        assert_eq!(controller.current_capacity(), 24);
        add_steps(&mut controller, 1, 100);
        let confirmation_warmup = controller
            .observe(14, metrics(60, 30, 700_000_000), memory)
            .unwrap();
        assert_eq!(
            confirmation_warmup.action,
            LagunaMemoryAction::ConfirmationWarmupComplete
        );
        controller.pause_decode(metrics(60, 30, 700_000_000));

        // Request 3 confirms the baseline. Candidate throughput is more than
        // 10% above both baseline samples, so the growth is retained.
        controller.begin_decode(metrics(60, 30, 700_000_000), memory);
        add_steps(&mut controller, 2, 100);
        let keep = controller
            .observe(16, metrics(80, 45, 900_000_000), memory)
            .unwrap();
        assert_eq!(keep.action, LagunaMemoryAction::KeepTrial);
        assert_eq!(controller.current_capacity(), 25);
        assert_eq!(controller.report().trials_kept, 1);

        controller.pause_decode(metrics(80, 45, 900_000_000));
        controller.begin_decode(metrics(80, 45, 900_000_000), memory);
        add_steps(&mut controller, 2, 90);
        let settled = controller
            .observe(18, metrics(100, 55, 1_200_000_000), memory)
            .unwrap();
        assert_eq!(settled.action, LagunaMemoryAction::Hold);
        assert_eq!(settled.next_expert_capacity, 25);
        assert_eq!(controller.report().trials_started, 1);
    }

    #[test]
    fn rolls_back_a_growth_that_does_not_improve_throughput() {
        let mut controller = LagunaMemoryController::new(spec()).unwrap();
        let memory = memory(20_000_000_000, 0, 0);
        controller.begin_decode(metrics(0, 0, 0), memory);

        add_steps(&mut controller, 2, 100);
        controller
            .observe(10, metrics(20, 5, 300_000_000), memory)
            .unwrap();
        add_steps(&mut controller, 1, 100);
        let warmup = controller
            .observe(11, metrics(30, 10, 400_000_000), memory)
            .unwrap();
        assert_eq!(warmup.action, LagunaMemoryAction::WarmupComplete);
        controller.pause_decode(metrics(30, 10, 400_000_000));

        controller.begin_decode(metrics(30, 10, 400_000_000), memory);
        add_steps(&mut controller, 2, 110);
        let candidate = controller
            .observe(13, metrics(50, 20, 700_000_000), memory)
            .unwrap();
        assert_eq!(candidate.action, LagunaMemoryAction::CandidateMeasured);

        add_steps(&mut controller, 1, 100);
        let confirmation_warmup = controller
            .observe(14, metrics(60, 25, 800_000_000), memory)
            .unwrap();
        assert_eq!(
            confirmation_warmup.action,
            LagunaMemoryAction::ConfirmationWarmupComplete
        );
        controller.pause_decode(metrics(60, 25, 800_000_000));

        controller.begin_decode(metrics(60, 25, 800_000_000), memory);
        add_steps(&mut controller, 2, 100);
        let rollback = controller
            .observe(16, metrics(80, 35, 1_000_000_000), memory)
            .unwrap();

        assert_eq!(rollback.action, LagunaMemoryAction::RollbackTrial);
        assert_eq!(rollback.next_expert_capacity, 24);
        assert_eq!(controller.current_capacity(), 24);
        assert_eq!(controller.report().trials_rolled_back, 1);
    }

    #[test]
    fn a_trial_can_continue_across_short_requests() {
        let mut controller = LagunaMemoryController::new(spec()).unwrap();
        let memory = memory(20_000_000_000, 0, 0);
        controller.begin_decode(metrics(0, 0, 0), memory);
        add_steps(&mut controller, 2, 100);

        let decision = controller
            .observe(10, metrics(20, 18, 300_000_000), memory)
            .unwrap();
        assert_eq!(decision.action, LagunaMemoryAction::GrowTrial);
        assert_eq!(decision.next_expert_capacity, 25);

        controller.pause_decode(metrics(20, 18, 300_000_000));
        controller.begin_decode(metrics(30, 25, 400_000_000), memory);
        add_steps(&mut controller, 1, 100);
        let next_request = controller
            .observe(20, metrics(40, 30, 500_000_000), memory)
            .unwrap();
        assert_eq!(next_request.action, LagunaMemoryAction::WarmupComplete);
        assert_eq!(controller.current_capacity(), 25);
    }

    #[test]
    fn a_long_request_can_complete_an_entire_growth_trial() {
        let mut controller = LagunaMemoryController::new(spec()).unwrap();
        let memory = memory(20_000_000_000, 0, 0);
        controller.begin_decode(metrics(0, 0, 0), memory);

        add_steps(&mut controller, 2, 100);
        let grow = controller
            .observe(10, metrics(20, 5, 300_000_000), memory)
            .unwrap();
        assert_eq!(grow.action, LagunaMemoryAction::GrowTrial);

        add_steps(&mut controller, 1, 100);
        let warm = controller
            .observe(11, metrics(30, 10, 400_000_000), memory)
            .unwrap();
        assert_eq!(warm.action, LagunaMemoryAction::WarmupComplete);

        add_steps(&mut controller, 2, 80);
        let candidate = controller
            .observe(13, metrics(50, 25, 600_000_000), memory)
            .unwrap();
        assert_eq!(candidate.action, LagunaMemoryAction::CandidateMeasured);
        assert_eq!(controller.current_capacity(), 24);

        add_steps(&mut controller, 1, 100);
        let confirmation_warmup = controller
            .observe(14, metrics(60, 30, 700_000_000), memory)
            .unwrap();
        assert_eq!(
            confirmation_warmup.action,
            LagunaMemoryAction::ConfirmationWarmupComplete
        );

        add_steps(&mut controller, 2, 100);
        let keep = controller
            .observe(16, metrics(80, 45, 900_000_000), memory)
            .unwrap();
        assert_eq!(keep.action, LagunaMemoryAction::KeepTrial);
        assert_eq!(keep.next_expert_capacity, 25);
        assert_eq!(controller.report().decisions, 5);
        assert_eq!(controller.report().trials_kept, 1);
    }

    #[test]
    fn a_long_request_keeps_checking_memory_after_a_decision() {
        let mut controller_spec = spec();
        controller_spec.stabilization_windows = 1;
        let mut controller = LagunaMemoryController::new(controller_spec).unwrap();
        let baseline_memory = memory(20_000_000_000, 0, 0);
        controller.begin_decode(metrics(0, 0, 0), baseline_memory);

        add_steps(&mut controller, 2, 100);
        let stabilized = controller
            .observe(10, metrics(20, 18, 300_000_000), baseline_memory)
            .unwrap();
        assert_eq!(stabilized.action, LagunaMemoryAction::Hold);

        add_steps(&mut controller, 2, 100);
        let pressure = controller
            .observe(
                12,
                metrics(40, 30, 600_000_000),
                memory(20_000_000_000, PRESSURE_SWAP_GROWTH_BYTES, 0),
            )
            .unwrap();
        assert_eq!(pressure.action, LagunaMemoryAction::PressureShrink);
        assert_eq!(pressure.pressure, LagunaMemoryPressure::SwapGrowth);
        assert_eq!(pressure.next_expert_capacity, 23);
    }

    #[test]
    fn low_ssd_traffic_does_not_resize_without_memory_pressure() {
        let mut controller = LagunaMemoryController::new(spec()).unwrap();
        let memory = memory(20_000_000_000, 0, 0);
        controller.begin_decode(metrics(0, 0, 0), memory);
        add_steps(&mut controller, 2, 100);

        let decision = controller
            .observe(10, metrics(20, 18, 100_000_000), memory)
            .unwrap();

        assert_eq!(decision.action, LagunaMemoryAction::Hold);
        assert_eq!(decision.next_expert_capacity, 24);
    }

    #[test]
    fn a_cache_with_free_slots_does_not_start_a_growth_trial() {
        let mut controller = LagunaMemoryController::new(spec()).unwrap();
        let memory = memory(20_000_000_000, 0, 0);
        controller.begin_decode(metrics(0, 0, 0), memory);
        add_steps(&mut controller, 2, 100);
        let mut experts = metrics(20, 5, 300_000_000);
        experts.resident_experts = 12;
        experts.resident_bytes = 12_000;

        let decision = controller.observe(10, experts, memory).unwrap();

        assert_eq!(decision.action, LagunaMemoryAction::Hold);
        assert_eq!(decision.next_expert_capacity, 24);
        assert_eq!(controller.report().trials_started, 0);
    }

    #[test]
    fn persistent_runtime_pays_stabilization_only_once() {
        let mut controller_spec = spec();
        controller_spec.stabilization_windows = 1;
        let mut controller = LagunaMemoryController::new(controller_spec).unwrap();
        let memory = memory(20_000_000_000, 0, 0);

        controller.begin_decode(metrics(0, 0, 0), memory);
        add_steps(&mut controller, 2, 100);
        let first_request = controller
            .observe(10, metrics(20, 18, 300_000_000), memory)
            .unwrap();
        assert_eq!(first_request.action, LagunaMemoryAction::Hold);

        controller.pause_decode(metrics(20, 18, 300_000_000));
        controller.begin_decode(metrics(20, 18, 300_000_000), memory);
        add_steps(&mut controller, 2, 100);
        let second_request = controller
            .observe(20, metrics(40, 36, 600_000_000), memory)
            .unwrap();
        assert_eq!(second_request.action, LagunaMemoryAction::GrowTrial);
        assert_eq!(second_request.next_expert_capacity, 25);
    }

    #[test]
    fn cold_initial_baseline_cannot_hide_a_slower_candidate() {
        let mut controller = LagunaMemoryController::new(spec()).unwrap();
        let memory = memory(20_000_000_000, 0, 0);
        controller.begin_decode(metrics(0, 0, 0), memory);

        add_steps(&mut controller, 2, 200);
        controller
            .observe(10, metrics(20, 18, 300_000_000), memory)
            .unwrap();
        add_steps(&mut controller, 1, 100);
        controller
            .observe(11, metrics(30, 27, 400_000_000), memory)
            .unwrap();
        controller.pause_decode(metrics(30, 27, 400_000_000));

        controller.begin_decode(metrics(30, 27, 400_000_000), memory);
        add_steps(&mut controller, 2, 100);
        controller
            .observe(13, metrics(50, 46, 500_000_000), memory)
            .unwrap();
        add_steps(&mut controller, 1, 90);
        controller
            .observe(14, metrics(60, 55, 600_000_000), memory)
            .unwrap();
        controller.pause_decode(metrics(60, 55, 600_000_000));

        controller.begin_decode(metrics(60, 55, 600_000_000), memory);
        add_steps(&mut controller, 2, 90);
        let rollback = controller
            .observe(16, metrics(80, 73, 800_000_000), memory)
            .unwrap();

        assert_eq!(rollback.action, LagunaMemoryAction::RollbackTrial);
        assert_eq!(rollback.next_expert_capacity, 24);
        let comparison = rollback.comparison_tokens_per_second.unwrap();
        assert!((comparison - 1_000.0 / 90.0).abs() < 1e-9);
    }

    #[test]
    fn pressure_shrinks_without_running_a_trial() {
        let mut controller = LagunaMemoryController::new(spec()).unwrap();
        controller.begin_decode(metrics(0, 0, 0), memory(20_000_000_000, 0, 0));
        add_steps(&mut controller, 2, 100);
        let pressure = controller
            .observe(
                10,
                metrics(20, 5, 300_000_000),
                memory(20_000_000_000, PRESSURE_SWAP_GROWTH_BYTES, 0),
            )
            .unwrap();

        assert_eq!(pressure.action, LagunaMemoryAction::PressureShrink);
        assert_eq!(pressure.pressure, LagunaMemoryPressure::SwapGrowth);
        assert_eq!(pressure.next_expert_capacity, 23);
        assert_eq!(controller.report().pressure_events, 1);
    }

    #[test]
    fn compression_growth_is_not_pressure_while_headroom_is_safe() {
        let mut controller = LagunaMemoryController::new(spec()).unwrap();
        controller.begin_decode(metrics(0, 0, 0), memory(20_000_000_000, 0, 0));
        add_steps(&mut controller, 2, 100);
        let decision = controller
            .observe(
                10,
                metrics(20, 5, 300_000_000),
                memory(20_000_000_000, 0, PRESSURE_COMPRESSION_GROWTH_BYTES),
            )
            .unwrap();

        assert_eq!(decision.pressure, LagunaMemoryPressure::None);
        assert_eq!(decision.action, LagunaMemoryAction::GrowTrial);
    }

    #[test]
    fn partial_decode_windows_continue_while_prefill_metrics_are_excluded() {
        let mut controller = LagunaMemoryController::new(spec()).unwrap();
        let memory = memory(20_000_000_000, 0, 0);
        controller.begin_decode(metrics(0, 0, 0), memory);
        add_steps(&mut controller, 1, 100);
        controller.pause_decode(metrics(10, 5, 300_000_000));

        // The next prompt prefill adds metrics before decode resumes. Those
        // values must not enter the decode-only controller window. The one
        // decode token from the previous request must remain in the window.
        controller.begin_decode(metrics(20, 10, 600_000_000), memory);
        add_steps(&mut controller, 1, 100);
        assert!(controller.observation_due());
        let decision = controller
            .observe(20, metrics(30, 15, 900_000_000), memory)
            .unwrap();

        assert_eq!(decision.action, LagunaMemoryAction::GrowTrial);
        assert_eq!(decision.expert_lookups, 20);
        assert_eq!(decision.expert_hits, 10);
        assert_eq!(decision.expert_ssd_read_bytes, 600_000_000);
    }

    #[test]
    fn swap_growth_during_prefill_is_visible_to_the_next_decode_window() {
        let mut controller = LagunaMemoryController::new(spec()).unwrap();
        controller.begin_decode(metrics(0, 0, 0), memory(20_000_000_000, 0, 0));
        add_steps(&mut controller, 1, 100);
        controller.pause_decode(metrics(10, 5, 300_000_000));

        controller.begin_decode(
            metrics(20, 10, 600_000_000),
            memory(20_000_000_000, PRESSURE_SWAP_GROWTH_BYTES, 0),
        );
        add_steps(&mut controller, 1, 100);
        let decision = controller
            .observe(
                20,
                metrics(30, 15, 900_000_000),
                memory(20_000_000_000, PRESSURE_SWAP_GROWTH_BYTES, 0),
            )
            .unwrap();

        assert_eq!(decision.action, LagunaMemoryAction::PressureShrink);
        assert_eq!(decision.pressure, LagunaMemoryPressure::SwapGrowth);
    }
}
