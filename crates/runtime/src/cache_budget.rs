use common::{Error, Result};

pub const DEFAULT_DYNAMIC_CACHE_BUDGET_BYTES: usize = 10_000_000_000;
const SHORT_CONTEXT_TOKENS: usize = 32 * 1024;
const MEDIUM_CONTEXT_TOKENS: usize = 256 * 1024;
const MIN_SIGNAL_LOOKUPS: u64 = 1_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheBudgetSpec {
    pub total_bytes: usize,
    pub kv_bytes_per_token: usize,
    pub expert_bytes_per_layer_slot: usize,
    pub routed_layer_count: usize,
    pub page_size: usize,
    pub min_expert_slots_per_layer: usize,
    pub max_expert_slots_per_layer: usize,
}

impl CacheBudgetSpec {
    pub fn validate(self) -> Result<()> {
        if self.total_bytes == 0
            || self.kv_bytes_per_token == 0
            || self.expert_bytes_per_layer_slot == 0
            || self.routed_layer_count == 0
            || self.page_size == 0
        {
            return Err(Error::runtime(
                "dynamic cache budget dimensions must all be positive",
            ));
        }
        if self.min_expert_slots_per_layer == 0
            || self.min_expert_slots_per_layer > self.max_expert_slots_per_layer
        {
            return Err(Error::runtime(format!(
                "dynamic cache expert slot range must satisfy 0 < min <= max, got min={} max={}",
                self.min_expert_slots_per_layer, self.max_expert_slots_per_layer
            )));
        }
        let minimum_expert_bytes = self
            .global_expert_slot_bytes()?
            .checked_mul(self.min_expert_slots_per_layer)
            .ok_or_else(|| Error::runtime("minimum expert cache byte count overflow"))?;
        if minimum_expert_bytes >= self.total_bytes {
            return Err(Error::runtime(format!(
                "dynamic cache budget {} bytes cannot fit minimum expert cache {} bytes and a hot KV tier",
                self.total_bytes, minimum_expert_bytes
            )));
        }
        Ok(())
    }

    pub fn plan(self, context_len: usize, signals: CacheBudgetSignals) -> Result<CacheBudgetPlan> {
        self.validate()?;
        let capacity_tokens = context_len
            .max(1)
            .div_ceil(self.page_size)
            .checked_mul(self.page_size)
            .ok_or_else(|| Error::runtime("hot KV capacity token count overflow"))?;
        let all_layer_hot_bytes = capacity_tokens
            .checked_mul(self.kv_bytes_per_token)
            .ok_or_else(|| Error::runtime("all-layer hot KV byte count overflow"))?;
        let tier = ContextTier::for_context(context_len);
        let hot_cap_bytes = self
            .total_bytes
            .checked_mul(tier.hot_budget_percent())
            .ok_or_else(|| Error::runtime("hot KV tier cap overflow"))?
            / 100;
        let desired_hot_bytes = all_layer_hot_bytes.min(hot_cap_bytes);
        let global_expert_slot_bytes = self.global_expert_slot_bytes()?;
        let available_for_experts = self.total_bytes.saturating_sub(desired_hot_bytes);
        let mut expert_slots_per_layer = (available_for_experts / global_expert_slot_bytes).clamp(
            self.min_expert_slots_per_layer,
            self.max_expert_slots_per_layer,
        );
        let mut adjustment = CacheBudgetAdjustment::Context;

        if signals.memory_pressure {
            expert_slots_per_layer = expert_slots_per_layer
                .saturating_sub(2)
                .max(self.min_expert_slots_per_layer);
            adjustment = CacheBudgetAdjustment::MemoryPressure;
        } else if signals.kv_lookups >= MIN_SIGNAL_LOOKUPS && signals.kv_miss_rate() > 0.10 {
            expert_slots_per_layer = expert_slots_per_layer
                .saturating_sub(1)
                .max(self.min_expert_slots_per_layer);
            adjustment = CacheBudgetAdjustment::KvMisses;
        } else if signals.expert_lookups >= MIN_SIGNAL_LOOKUPS && signals.expert_hit_rate() < 0.25 {
            expert_slots_per_layer = expert_slots_per_layer
                .saturating_sub(1)
                .max(self.min_expert_slots_per_layer);
            adjustment = CacheBudgetAdjustment::LowExpertReuse;
        } else if signals.expert_lookups >= MIN_SIGNAL_LOOKUPS
            && signals.expert_hit_rate() > 0.60
            && (signals.kv_lookups < MIN_SIGNAL_LOOKUPS || signals.kv_miss_rate() < 0.05)
        {
            let candidate = expert_slots_per_layer
                .saturating_add(1)
                .min(self.max_expert_slots_per_layer);
            let candidate_hot_bytes = self.total_bytes.saturating_sub(
                global_expert_slot_bytes
                    .checked_mul(candidate)
                    .ok_or_else(|| Error::runtime("candidate expert cache size overflow"))?,
            );
            if candidate_hot_bytes >= desired_hot_bytes {
                expert_slots_per_layer = candidate;
                adjustment = CacheBudgetAdjustment::HighExpertReuse;
            }
        }

        let expert_bytes = global_expert_slot_bytes
            .checked_mul(expert_slots_per_layer)
            .ok_or_else(|| Error::runtime("expert cache plan byte count overflow"))?;
        let hot_kv_budget_bytes = self.total_bytes.saturating_sub(expert_bytes);
        Ok(CacheBudgetPlan {
            context_len,
            capacity_tokens,
            tier,
            adjustment,
            expert_slots_per_layer,
            expert_bytes,
            hot_kv_budget_bytes,
            all_layer_hot_bytes,
            all_layers_fit: all_layer_hot_bytes <= hot_kv_budget_bytes,
        })
    }

    fn global_expert_slot_bytes(self) -> Result<usize> {
        self.expert_bytes_per_layer_slot
            .checked_mul(self.routed_layer_count)
            .ok_or_else(|| Error::runtime("global expert slot byte count overflow"))
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheBudgetSignals {
    pub expert_lookups: u64,
    pub expert_hits: u64,
    pub kv_lookups: u64,
    pub kv_misses: u64,
    pub memory_pressure: bool,
}

impl CacheBudgetSignals {
    fn expert_hit_rate(self) -> f64 {
        rate(self.expert_hits, self.expert_lookups)
    }

    fn kv_miss_rate(self) -> f64 {
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

    fn hot_budget_percent(self) -> usize {
        match self {
            Self::Short => 60,
            Self::Medium => 75,
            Self::Long => 85,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheBudgetAdjustment {
    Context,
    MemoryPressure,
    KvMisses,
    LowExpertReuse,
    HighExpertReuse,
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
            total_bytes: 10_000_000_000,
            kv_bytes_per_token: 78 * 576 * 4,
            expert_bytes_per_layer_slot: 12_386_304,
            routed_layer_count: 75,
            page_size: 128,
            min_expert_slots_per_layer: 2,
            max_expert_slots_per_layer: 10,
        }
    }

    #[test]
    fn short_context_prioritizes_expert_cache_after_fitting_kv() {
        let plan = spec().plan(13, CacheBudgetSignals::default()).unwrap();

        assert_eq!(plan.tier, ContextTier::Short);
        assert_eq!(plan.capacity_tokens, 128);
        assert_eq!(plan.expert_slots_per_layer, 10);
        assert!(plan.all_layers_fit);
        assert!(plan.expert_bytes > plan.all_layer_hot_bytes);
    }

    #[test]
    fn longer_context_moves_slots_from_experts_to_hot_kv() {
        let short = spec().plan(13, CacheBudgetSignals::default()).unwrap();
        let medium = spec()
            .plan(32 * 1024, CacheBudgetSignals::default())
            .unwrap();
        let long = spec()
            .plan(256 * 1024, CacheBudgetSignals::default())
            .unwrap();

        assert!(medium.expert_slots_per_layer < short.expert_slots_per_layer);
        assert!(long.expert_slots_per_layer <= medium.expert_slots_per_layer);
        assert!(medium.hot_kv_budget_bytes > short.hot_kv_budget_bytes);
        assert!(long.hot_kv_budget_bytes >= medium.hot_kv_budget_bytes);
    }

    #[test]
    fn memory_pressure_shrinks_experts_before_other_caches() {
        let baseline = spec().plan(13, CacheBudgetSignals::default()).unwrap();
        let pressured = spec()
            .plan(
                13,
                CacheBudgetSignals {
                    memory_pressure: true,
                    ..CacheBudgetSignals::default()
                },
            )
            .unwrap();

        assert_eq!(pressured.adjustment, CacheBudgetAdjustment::MemoryPressure);
        assert!(pressured.expert_slots_per_layer < baseline.expert_slots_per_layer);
        assert!(pressured.hot_kv_budget_bytes > baseline.hot_kv_budget_bytes);
    }

    #[test]
    fn low_expert_reuse_releases_one_slot_after_enough_samples() {
        let plan = spec()
            .plan(
                13,
                CacheBudgetSignals {
                    expert_lookups: 10_000,
                    expert_hits: 1_000,
                    ..CacheBudgetSignals::default()
                },
            )
            .unwrap();

        assert_eq!(plan.adjustment, CacheBudgetAdjustment::LowExpertReuse);
        assert_eq!(plan.expert_slots_per_layer, 9);
    }
}
