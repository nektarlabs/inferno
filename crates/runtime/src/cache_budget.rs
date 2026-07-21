use common::{Error, Result};

pub const DEFAULT_EXPERT_CACHE_SLOTS_PER_LAYER: usize = 30;
pub const DEFAULT_HOT_KV_CACHE_BUDGET_BYTES: usize = 512 * 1024 * 1024;
pub const DEFAULT_MIN_EXPERT_CACHE_SLOTS_PER_LAYER: usize = 16;
pub const DEFAULT_MAX_EXPERT_CACHE_SLOTS_PER_LAYER: usize = 32;
pub const DEFAULT_TARGET_HEADROOM_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub const DEFAULT_HARD_HEADROOM_BYTES: u64 = 3 * 1024 * 1024 * 1024;
pub const DEFAULT_DECISION_WINDOW_TOKENS: usize = 8;

const HOT_KV_BUDGET_STEP_BYTES: usize = 512 * 1024 * 1024;
const MIN_TRIAL_THROUGHPUT_IMPROVEMENT: f64 = 0.01;
const SHORT_CONTEXT_TOKENS: usize = 32 * 1024;
const MEDIUM_CONTEXT_TOKENS: usize = 256 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheBudgetSpec {
    pub kv_bytes_per_token: usize,
    pub expert_bytes_per_layer_slot: usize,
    pub routed_layer_count: usize,
    pub page_size: usize,
    pub initial_expert_slots_per_layer: usize,
    pub min_expert_slots_per_layer: usize,
    pub max_expert_slots_per_layer: usize,
    pub initial_hot_kv_budget_bytes: usize,
    pub min_hot_kv_budget_bytes: usize,
    pub max_hot_kv_budget_bytes: usize,
    pub target_headroom_bytes: u64,
    pub hard_headroom_bytes: u64,
    pub decision_window_tokens: usize,
}

impl CacheBudgetSpec {
    pub fn validate(self) -> Result<()> {
        if self.kv_bytes_per_token == 0
            || self.expert_bytes_per_layer_slot == 0
            || self.routed_layer_count == 0
            || self.page_size == 0
            || self.decision_window_tokens == 0
        {
            return Err(Error::runtime(
                "adaptive cache dimensions and decision window must be positive",
            ));
        }
        if self.min_expert_slots_per_layer == 0
            || self.min_expert_slots_per_layer > self.initial_expert_slots_per_layer
            || self.initial_expert_slots_per_layer > self.max_expert_slots_per_layer
        {
            return Err(Error::runtime(format!(
                "adaptive expert slots must satisfy 0 < min <= initial <= max, got min={} initial={} max={}",
                self.min_expert_slots_per_layer,
                self.initial_expert_slots_per_layer,
                self.max_expert_slots_per_layer
            )));
        }
        if self.min_hot_kv_budget_bytes == 0
            || self.min_hot_kv_budget_bytes > self.initial_hot_kv_budget_bytes
            || self.initial_hot_kv_budget_bytes > self.max_hot_kv_budget_bytes
        {
            return Err(Error::runtime(format!(
                "adaptive hot KV bytes must satisfy 0 < min <= initial <= max, got min={} initial={} max={}",
                self.min_hot_kv_budget_bytes,
                self.initial_hot_kv_budget_bytes,
                self.max_hot_kv_budget_bytes
            )));
        }
        if self.hard_headroom_bytes == 0 || self.hard_headroom_bytes >= self.target_headroom_bytes {
            return Err(Error::runtime(format!(
                "adaptive memory headroom must satisfy 0 < hard < target, got hard={} target={}",
                self.hard_headroom_bytes, self.target_headroom_bytes
            )));
        }
        self.global_expert_slot_bytes()?;
        Ok(())
    }

    pub fn initial_plan(self, context_len: usize) -> Result<CacheBudgetPlan> {
        self.validate()?;
        self.plan(
            context_len,
            self.initial_expert_slots_per_layer,
            self.initial_hot_kv_budget_bytes,
            CacheBudgetAdjustment::Initial,
        )
    }

    pub fn global_expert_slot_bytes(self) -> Result<usize> {
        self.expert_bytes_per_layer_slot
            .checked_mul(self.routed_layer_count)
            .ok_or_else(|| Error::runtime("global expert slot byte count overflow"))
    }

    pub fn all_layer_hot_bytes(self, context_len: usize) -> Result<usize> {
        let capacity_tokens = self.capacity_tokens(context_len)?;
        capacity_tokens
            .checked_mul(self.kv_bytes_per_token)
            .ok_or_else(|| Error::runtime("all-layer hot KV byte count overflow"))
    }

    fn capacity_tokens(self, context_len: usize) -> Result<usize> {
        context_len
            .max(1)
            .div_ceil(self.page_size)
            .checked_mul(self.page_size)
            .ok_or_else(|| Error::runtime("hot KV capacity token count overflow"))
    }

    fn plan(
        self,
        context_len: usize,
        expert_slots_per_layer: usize,
        hot_kv_budget_bytes: usize,
        adjustment: CacheBudgetAdjustment,
    ) -> Result<CacheBudgetPlan> {
        let capacity_tokens = self.capacity_tokens(context_len)?;
        let all_layer_hot_bytes = self.all_layer_hot_bytes(context_len)?;
        let expert_bytes = self
            .global_expert_slot_bytes()?
            .checked_mul(expert_slots_per_layer)
            .ok_or_else(|| Error::runtime("expert cache plan byte count overflow"))?;
        Ok(CacheBudgetPlan {
            context_len,
            capacity_tokens,
            tier: ContextTier::for_context(context_len),
            adjustment,
            expert_slots_per_layer,
            expert_bytes,
            hot_kv_budget_bytes,
            all_layer_hot_bytes,
            all_layers_fit: all_layer_hot_bytes <= hot_kv_budget_bytes,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CacheBudgetSignals {
    pub tokens: usize,
    pub elapsed_nanoseconds: u64,
    pub expert_lookups: u64,
    pub expert_hits: u64,
    pub expert_ssd_read_bytes: u64,
    pub expert_ssd_load_nanoseconds: u64,
    pub kv_lookups: u64,
    pub kv_misses: u64,
    pub kv_ssd_read_bytes: u64,
    pub kv_read_nanoseconds: u64,
    pub effective_available_bytes: Option<u64>,
    pub metal_headroom_bytes: Option<u64>,
    pub pressure: MemoryPressure,
}

impl CacheBudgetSignals {
    pub fn tokens_per_second(self) -> f64 {
        if self.tokens == 0 || self.elapsed_nanoseconds == 0 {
            return 0.0;
        }
        self.tokens as f64 * 1_000_000_000.0 / self.elapsed_nanoseconds as f64
    }

    pub fn expert_hit_rate(self) -> f64 {
        rate(self.expert_hits, self.expert_lookups)
    }

    pub fn kv_miss_rate(self) -> f64 {
        rate(self.kv_misses, self.kv_lookups)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheBudgetPlan {
    pub context_len: usize,
    pub capacity_tokens: usize,
    pub tier: ContextTier,
    pub adjustment: CacheBudgetAdjustment,
    pub expert_slots_per_layer: usize,
    pub expert_bytes: usize,
    pub hot_kv_budget_bytes: usize,
    pub all_layer_hot_bytes: usize,
    pub all_layers_fit: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextTier {
    Short,
    Medium,
    Long,
}

impl ContextTier {
    fn for_context(context_len: usize) -> Self {
        if context_len < SHORT_CONTEXT_TOKENS {
            Self::Short
        } else if context_len < MEDIUM_CONTEXT_TOKENS {
            Self::Medium
        } else {
            Self::Long
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheBudgetAdjustment {
    Initial,
    Hold,
    ExpertTrial,
    HotKvTrial,
    KeepTrial,
    RollbackTrial,
    MemoryPressure,
    PhaseTransition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    None,
    Headroom,
    CriticalHeadroom,
    MetalWorkingSet,
    SwapGrowth,
    CompressionGrowth,
}

impl MemoryPressure {
    pub fn is_pressure(self) -> bool {
        self != Self::None
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheResource {
    Experts,
    HotKv,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CacheBudgetDecision {
    pub plan: CacheBudgetPlan,
    pub action: CacheBudgetAdjustment,
    pub resource: Option<CacheResource>,
    pub measured_tokens_per_second: f64,
    pub comparison_tokens_per_second: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct CacheTrial {
    resource: CacheResource,
    previous_expert_slots_per_layer: usize,
    previous_hot_kv_budget_bytes: usize,
    baseline_tokens_per_second: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveCachePolicy {
    spec: CacheBudgetSpec,
    current: CacheBudgetPlan,
    trial: Option<CacheTrial>,
    rejected_expert_growth_at: Option<usize>,
    rejected_hot_kv_growth_at: Option<usize>,
}

impl AdaptiveCachePolicy {
    pub fn new(spec: CacheBudgetSpec, context_len: usize) -> Result<Self> {
        Ok(Self {
            current: spec.initial_plan(context_len)?,
            spec,
            trial: None,
            rejected_expert_growth_at: None,
            rejected_hot_kv_growth_at: None,
        })
    }

    pub fn current(self) -> CacheBudgetPlan {
        self.current
    }

    pub fn observe(
        &mut self,
        context_len: usize,
        signals: CacheBudgetSignals,
    ) -> Result<CacheBudgetDecision> {
        let measured_tps = signals.tokens_per_second();
        if signals.pressure.is_pressure() {
            return self.pressure_decision(context_len, measured_tps, signals.pressure);
        }

        if let Some(trial) = self.trial.take() {
            let required =
                trial.baseline_tokens_per_second * (1.0 + MIN_TRIAL_THROUGHPUT_IMPROVEMENT);
            if measured_tps >= required {
                match trial.resource {
                    CacheResource::Experts => self.rejected_expert_growth_at = None,
                    CacheResource::HotKv => self.rejected_hot_kv_growth_at = None,
                }
                self.current = self.spec.plan(
                    context_len,
                    self.current.expert_slots_per_layer,
                    self.current.hot_kv_budget_bytes,
                    CacheBudgetAdjustment::KeepTrial,
                )?;
                return Ok(CacheBudgetDecision {
                    plan: self.current,
                    action: CacheBudgetAdjustment::KeepTrial,
                    resource: Some(trial.resource),
                    measured_tokens_per_second: measured_tps,
                    comparison_tokens_per_second: Some(trial.baseline_tokens_per_second),
                });
            }

            match trial.resource {
                CacheResource::Experts => {
                    self.rejected_expert_growth_at = Some(trial.previous_expert_slots_per_layer)
                }
                CacheResource::HotKv => {
                    self.rejected_hot_kv_growth_at = Some(trial.previous_hot_kv_budget_bytes)
                }
            }
            self.current = self.spec.plan(
                context_len,
                trial.previous_expert_slots_per_layer,
                trial.previous_hot_kv_budget_bytes,
                CacheBudgetAdjustment::RollbackTrial,
            )?;
            return Ok(CacheBudgetDecision {
                plan: self.current,
                action: CacheBudgetAdjustment::RollbackTrial,
                resource: Some(trial.resource),
                measured_tokens_per_second: measured_tps,
                comparison_tokens_per_second: Some(trial.baseline_tokens_per_second),
            });
        }

        if let Some(resource) = self.choose_trial(context_len, signals)? {
            let previous = self.current;
            let (expert_slots, hot_kv_bytes, action) = match resource {
                CacheResource::Experts => (
                    previous.expert_slots_per_layer + 1,
                    previous.hot_kv_budget_bytes,
                    CacheBudgetAdjustment::ExpertTrial,
                ),
                CacheResource::HotKv => (
                    previous.expert_slots_per_layer,
                    self.hot_kv_trial_bytes(context_len)?,
                    CacheBudgetAdjustment::HotKvTrial,
                ),
            };
            self.current = self
                .spec
                .plan(context_len, expert_slots, hot_kv_bytes, action)?;
            self.trial = Some(CacheTrial {
                resource,
                previous_expert_slots_per_layer: previous.expert_slots_per_layer,
                previous_hot_kv_budget_bytes: previous.hot_kv_budget_bytes,
                baseline_tokens_per_second: measured_tps,
            });
            return Ok(CacheBudgetDecision {
                plan: self.current,
                action,
                resource: Some(resource),
                measured_tokens_per_second: measured_tps,
                comparison_tokens_per_second: None,
            });
        }

        self.current = self.spec.plan(
            context_len,
            self.current.expert_slots_per_layer,
            self.current.hot_kv_budget_bytes,
            CacheBudgetAdjustment::Hold,
        )?;
        Ok(CacheBudgetDecision {
            plan: self.current,
            action: CacheBudgetAdjustment::Hold,
            resource: None,
            measured_tokens_per_second: measured_tps,
            comparison_tokens_per_second: None,
        })
    }

    pub fn phase_transition(&mut self, context_len: usize) -> Result<CacheBudgetDecision> {
        self.trial = None;
        self.current = self.spec.plan(
            context_len,
            self.current.expert_slots_per_layer,
            self.current.hot_kv_budget_bytes,
            CacheBudgetAdjustment::PhaseTransition,
        )?;
        Ok(CacheBudgetDecision {
            plan: self.current,
            action: CacheBudgetAdjustment::PhaseTransition,
            resource: None,
            measured_tokens_per_second: 0.0,
            comparison_tokens_per_second: None,
        })
    }

    fn choose_trial(
        self,
        context_len: usize,
        signals: CacheBudgetSignals,
    ) -> Result<Option<CacheResource>> {
        let expert_cost = signals.expert_ssd_load_nanoseconds;
        let kv_cost = signals.kv_read_nanoseconds;
        let can_grow_experts = self.current.expert_slots_per_layer
            < self.spec.max_expert_slots_per_layer
            && self.rejected_expert_growth_at != Some(self.current.expert_slots_per_layer)
            && signals.expert_lookups > 0
            && signals.expert_hits < signals.expert_lookups
            && signals.expert_ssd_read_bytes > 0
            && self.has_headroom(self.spec.global_expert_slot_bytes()?, signals);
        let hot_trial_bytes = self.hot_kv_trial_bytes(context_len)?;
        let hot_growth_bytes = hot_trial_bytes.saturating_sub(self.current.hot_kv_budget_bytes);
        let can_grow_hot_kv = hot_growth_bytes > 0
            && self.rejected_hot_kv_growth_at != Some(self.current.hot_kv_budget_bytes)
            && signals.kv_lookups > 0
            && signals.kv_misses > 0
            && signals.kv_ssd_read_bytes > 0
            && self.has_headroom(hot_growth_bytes, signals);

        match (can_grow_experts, can_grow_hot_kv) {
            (true, true) if kv_cost > expert_cost => Ok(Some(CacheResource::HotKv)),
            (true, _) => Ok(Some(CacheResource::Experts)),
            (_, true) => Ok(Some(CacheResource::HotKv)),
            _ => Ok(None),
        }
    }

    fn has_headroom(self, growth_bytes: usize, signals: CacheBudgetSignals) -> bool {
        let growth = u64::try_from(growth_bytes).unwrap_or(u64::MAX);
        let system_ok = signals.effective_available_bytes.is_none_or(|available| {
            available >= self.spec.target_headroom_bytes.saturating_add(growth)
        });
        let metal_ok = signals
            .metal_headroom_bytes
            .is_none_or(|available| available >= growth.saturating_add(512 * 1024 * 1024));
        system_ok && metal_ok
    }

    fn hot_kv_trial_bytes(self, context_len: usize) -> Result<usize> {
        let all_layer_bytes = self.spec.all_layer_hot_bytes(context_len)?;
        let stepped = self
            .current
            .hot_kv_budget_bytes
            .saturating_add(HOT_KV_BUDGET_STEP_BYTES);
        Ok(stepped
            .max(all_layer_bytes)
            .min(self.spec.max_hot_kv_budget_bytes))
    }

    fn pressure_decision(
        &mut self,
        context_len: usize,
        measured_tps: f64,
        pressure: MemoryPressure,
    ) -> Result<CacheBudgetDecision> {
        if let Some(trial) = self.trial.take() {
            self.current = self.spec.plan(
                context_len,
                trial.previous_expert_slots_per_layer,
                trial.previous_hot_kv_budget_bytes,
                CacheBudgetAdjustment::MemoryPressure,
            )?;
        }

        let mut resource = None;
        let mut expert_slots = self.current.expert_slots_per_layer;
        let mut hot_kv_bytes = self.current.hot_kv_budget_bytes;
        if expert_slots > self.spec.min_expert_slots_per_layer {
            let shrink_slots = match pressure {
                MemoryPressure::CriticalHeadroom
                | MemoryPressure::SwapGrowth
                | MemoryPressure::CompressionGrowth => 2,
                _ => 1,
            };
            expert_slots = expert_slots
                .saturating_sub(shrink_slots)
                .max(self.spec.min_expert_slots_per_layer);
            resource = Some(CacheResource::Experts);
        } else if hot_kv_bytes > self.spec.min_hot_kv_budget_bytes {
            hot_kv_bytes = hot_kv_bytes
                .saturating_sub(HOT_KV_BUDGET_STEP_BYTES)
                .max(self.spec.min_hot_kv_budget_bytes);
            resource = Some(CacheResource::HotKv);
        }
        self.current = self.spec.plan(
            context_len,
            expert_slots,
            hot_kv_bytes,
            CacheBudgetAdjustment::MemoryPressure,
        )?;
        Ok(CacheBudgetDecision {
            plan: self.current,
            action: CacheBudgetAdjustment::MemoryPressure,
            resource,
            measured_tokens_per_second: measured_tps,
            comparison_tokens_per_second: None,
        })
    }
}

fn rate(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        return 0.0;
    }
    numerator as f64 / denominator as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> CacheBudgetSpec {
        CacheBudgetSpec {
            kv_bytes_per_token: 78 * 576 * 4,
            expert_bytes_per_layer_slot: 12_386_304,
            routed_layer_count: 75,
            page_size: 128,
            initial_expert_slots_per_layer: 30,
            min_expert_slots_per_layer: 16,
            max_expert_slots_per_layer: 32,
            initial_hot_kv_budget_bytes: DEFAULT_HOT_KV_CACHE_BUDGET_BYTES,
            min_hot_kv_budget_bytes: 128 * 1024 * 1024,
            max_hot_kv_budget_bytes: 16 * 1024 * 1024 * 1024,
            target_headroom_bytes: DEFAULT_TARGET_HEADROOM_BYTES,
            hard_headroom_bytes: DEFAULT_HARD_HEADROOM_BYTES,
            decision_window_tokens: DEFAULT_DECISION_WINDOW_TOKENS,
        }
    }

    fn signals(tps: f64) -> CacheBudgetSignals {
        CacheBudgetSignals {
            tokens: 8,
            elapsed_nanoseconds: (8.0 / tps * 1_000_000_000.0) as u64,
            expert_lookups: 4_800,
            expert_hits: 2_400,
            expert_ssd_read_bytes: 10_000_000_000,
            expert_ssd_load_nanoseconds: 4_000_000_000,
            kv_lookups: 24,
            kv_misses: 0,
            kv_ssd_read_bytes: 0,
            kv_read_nanoseconds: 0,
            effective_available_bytes: Some(8 * 1024 * 1024 * 1024),
            metal_headroom_bytes: Some(8 * 1024 * 1024 * 1024),
            pressure: MemoryPressure::None,
        }
    }

    #[test]
    fn expert_trial_requires_headroom_and_real_misses() {
        let mut policy = AdaptiveCachePolicy::new(spec(), 1_024).unwrap();

        let decision = policy.observe(1_024, signals(1.5)).unwrap();

        assert_eq!(decision.action, CacheBudgetAdjustment::ExpertTrial);
        assert_eq!(decision.resource, Some(CacheResource::Experts));
        assert_eq!(decision.plan.expert_slots_per_layer, 31);
    }

    #[test]
    fn slower_trial_is_rolled_back() {
        let mut policy = AdaptiveCachePolicy::new(spec(), 1_024).unwrap();
        policy.observe(1_024, signals(1.5)).unwrap();

        let decision = policy.observe(1_032, signals(1.4)).unwrap();

        assert_eq!(decision.action, CacheBudgetAdjustment::RollbackTrial);
        assert_eq!(decision.plan.expert_slots_per_layer, 30);
    }

    #[test]
    fn faster_trial_is_retained() {
        let mut policy = AdaptiveCachePolicy::new(spec(), 1_024).unwrap();
        policy.observe(1_024, signals(1.5)).unwrap();

        let decision = policy.observe(1_032, signals(1.6)).unwrap();

        assert_eq!(decision.action, CacheBudgetAdjustment::KeepTrial);
        assert_eq!(decision.plan.expert_slots_per_layer, 31);
    }

    #[test]
    fn pressure_rolls_back_a_trial_and_releases_an_expert_slot() {
        let mut policy = AdaptiveCachePolicy::new(spec(), 1_024).unwrap();
        policy.observe(1_024, signals(1.5)).unwrap();
        let mut pressure = signals(1.4);
        pressure.pressure = MemoryPressure::SwapGrowth;

        let decision = policy.observe(1_025, pressure).unwrap();

        assert_eq!(decision.action, CacheBudgetAdjustment::MemoryPressure);
        assert_eq!(decision.plan.expert_slots_per_layer, 28);
    }

    #[test]
    fn rejected_growth_is_not_retried_at_the_same_cache_size() {
        let mut policy = AdaptiveCachePolicy::new(spec(), 1_024).unwrap();
        policy.observe(1_024, signals(1.5)).unwrap();
        policy.observe(1_032, signals(1.4)).unwrap();

        let decision = policy.observe(1_040, signals(1.5)).unwrap();

        assert_eq!(decision.action, CacheBudgetAdjustment::Hold);
        assert_eq!(decision.plan.expert_slots_per_layer, 30);
    }

    #[test]
    fn kv_trial_is_selected_when_kv_is_the_larger_measured_cost() {
        let mut policy = AdaptiveCachePolicy::new(spec(), 32 * 1024).unwrap();
        let mut input = signals(1.5);
        input.expert_ssd_load_nanoseconds = 1;
        input.kv_lookups = 24;
        input.kv_misses = 24;
        input.kv_ssd_read_bytes = 2_000_000_000;
        input.kv_read_nanoseconds = 5_000_000_000;
        input.effective_available_bytes = Some(16 * 1024 * 1024 * 1024);
        input.metal_headroom_bytes = Some(16 * 1024 * 1024 * 1024);

        let decision = policy.observe(32 * 1024, input).unwrap();

        assert_eq!(decision.action, CacheBudgetAdjustment::HotKvTrial);
        assert_eq!(decision.resource, Some(CacheResource::HotKv));
        assert!(decision.plan.hot_kv_budget_bytes >= decision.plan.all_layer_hot_bytes);
    }
}
