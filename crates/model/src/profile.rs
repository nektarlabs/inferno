use std::{
    fs::File,
    io::Write,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex, OnceLock,
    },
    time::{Duration, Instant},
};

use backend::Backend;
use common::{Error, Result};

static LAYER_PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);
static LAYER_PROFILE_FILE: OnceLock<Mutex<Option<File>>> = OnceLock::new();
static LAYER_PROFILE_CONTEXT: OnceLock<Mutex<LayerProfileContext>> = OnceLock::new();
static TOKEN_COST_PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);
static TOKEN_COST_PROFILE: OnceLock<Mutex<TokenModelProfile>> = OnceLock::new();

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenModelProfile {
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
    pub output_projection_nanoseconds: u64,
    pub sampling_argmax_nanoseconds: u64,
}

impl TokenModelProfile {
    fn record(&mut self, stage: TokenProfileStage, nanoseconds: u64) {
        let target = match stage {
            TokenProfileStage::DenseAttention => &mut self.dense_attention_nanoseconds,
            TokenProfileStage::SparseAttention => &mut self.sparse_attention_nanoseconds,
            TokenProfileStage::SparseInputNorm => &mut self.sparse_input_norm_nanoseconds,
            TokenProfileStage::SparseQProjection => &mut self.sparse_q_projection_nanoseconds,
            TokenProfileStage::SparseKvProjection => &mut self.sparse_kv_projection_nanoseconds,
            TokenProfileStage::SparseCacheLayout => &mut self.sparse_cache_layout_nanoseconds,
            TokenProfileStage::SparseDsaIndexer => &mut self.sparse_dsa_indexer_nanoseconds,
            TokenProfileStage::SparseContextAttention => {
                &mut self.sparse_context_attention_nanoseconds
            }
            TokenProfileStage::SparseOutputProjection => {
                &mut self.sparse_output_projection_nanoseconds
            }
            TokenProfileStage::MoeRouting => &mut self.moe_routing_nanoseconds,
            TokenProfileStage::OutputProjection => &mut self.output_projection_nanoseconds,
            TokenProfileStage::SamplingArgmax => &mut self.sampling_argmax_nanoseconds,
        };
        *target = target.saturating_add(nanoseconds);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TokenProfileStage {
    DenseAttention,
    SparseAttention,
    SparseInputNorm,
    SparseQProjection,
    SparseKvProjection,
    SparseCacheLayout,
    SparseDsaIndexer,
    SparseContextAttention,
    SparseOutputProjection,
    MoeRouting,
    OutputProjection,
    SamplingArgmax,
}

pub fn enable_token_cost_profile() -> Result<()> {
    TOKEN_COST_PROFILE_ENABLED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .map_err(|_| Error::model("token cost profile is already enabled"))?;
    let mut profile = token_cost_profile().lock().map_err(|_| {
        TOKEN_COST_PROFILE_ENABLED.store(false, Ordering::Release);
        Error::model("token cost profile lock poisoned")
    })?;
    *profile = TokenModelProfile::default();
    Ok(())
}

pub fn disable_token_cost_profile() {
    TOKEN_COST_PROFILE_ENABLED.store(false, Ordering::Release);
}

pub fn take_token_cost_profile() -> Result<TokenModelProfile> {
    let mut profile = token_cost_profile()
        .lock()
        .map_err(|_| Error::model("token cost profile lock poisoned"))?;
    Ok(std::mem::take(&mut *profile))
}

pub(crate) fn token_cost_profile_enabled() -> bool {
    TOKEN_COST_PROFILE_ENABLED.load(Ordering::Acquire)
}

pub(crate) fn run_token_device_stage<T, B, F>(
    stage: TokenProfileStage,
    backend: &B,
    operation: F,
) -> Result<T>
where
    B: Backend,
    F: FnOnce() -> Result<T>,
{
    if !token_cost_profile_enabled() {
        return operation();
    }

    backend.device_flush()?;
    let started = Instant::now();
    let output = operation();
    if output.is_ok() {
        backend.device_flush()?;
    }
    record_token_stage(stage, started.elapsed());
    output
}

pub(crate) fn run_sparse_token_device_stage<T, B, F>(
    sparse_layer: bool,
    stage: TokenProfileStage,
    backend: &B,
    operation: F,
) -> Result<T>
where
    B: Backend,
    F: FnOnce() -> Result<T>,
{
    if !sparse_layer {
        return operation();
    }
    run_token_device_stage(stage, backend, operation)
}

fn record_token_stage(stage: TokenProfileStage, elapsed: Duration) {
    if !token_cost_profile_enabled() {
        return;
    }
    let Ok(mut profile) = token_cost_profile().lock() else {
        return;
    };
    let nanoseconds = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
    profile.record(stage, nanoseconds);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LayerProfileContext {
    step_index: usize,
    phase: &'static str,
}

pub fn enable_layer_profile(path: &Path) -> Result<()> {
    let mut file = File::create(path).map_err(|error| {
        Error::model(format!(
            "failed to create GLM layer profile file {}: {error}",
            path.display()
        ))
    })?;
    writeln!(file, "step_index\tphase\tlayer_index\tstage\telapsed_ms").map_err(|error| {
        Error::model(format!(
            "failed to write GLM layer profile header to {}: {error}",
            path.display()
        ))
    })?;

    let mut profile = layer_profile_file()
        .lock()
        .map_err(|_| Error::model("GLM layer profile lock poisoned"))?;
    if profile.is_some() {
        return Err(Error::model("GLM layer profile is already enabled"));
    }
    *profile = Some(file);
    LAYER_PROFILE_ENABLED.store(true, Ordering::Release);
    Ok(())
}

pub fn set_layer_profile_context(step_index: usize, phase: &'static str) -> Result<()> {
    if !LAYER_PROFILE_ENABLED.load(Ordering::Acquire) {
        return Ok(());
    }
    let mut context = layer_profile_context()
        .lock()
        .map_err(|_| Error::model("GLM layer profile context lock poisoned"))?;
    *context = LayerProfileContext { step_index, phase };
    Ok(())
}

pub(crate) fn record_layer(layer_index: usize, layer_kind: &str, elapsed: Duration) {
    if !LAYER_PROFILE_ENABLED.load(Ordering::Acquire) {
        return;
    }
    let Ok(mut profile) = layer_profile_file().lock() else {
        return;
    };
    let Some(file) = profile.as_mut() else {
        return;
    };
    let context = layer_profile_context()
        .lock()
        .map(|context| *context)
        .unwrap_or(LayerProfileContext {
            step_index: 0,
            phase: "unknown",
        });
    let _ = writeln!(
        file,
        "{}\t{}\t{layer_index}\t{layer_kind}\t{:.3}",
        context.step_index,
        context.phase,
        elapsed.as_secs_f64() * 1000.0
    );
    let _ = file.flush();
}

pub(crate) fn run_layer_stage<T, F>(layer_index: usize, layer_kind: &str, operation: F) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    if !LAYER_PROFILE_ENABLED.load(Ordering::Acquire) {
        return operation();
    }

    let started_at = Instant::now();
    let output = operation();
    record_layer(layer_index, layer_kind, started_at.elapsed());
    output
}

fn layer_profile_file() -> &'static Mutex<Option<File>> {
    LAYER_PROFILE_FILE.get_or_init(|| Mutex::new(None))
}

fn layer_profile_context() -> &'static Mutex<LayerProfileContext> {
    LAYER_PROFILE_CONTEXT.get_or_init(|| {
        Mutex::new(LayerProfileContext {
            step_index: 0,
            phase: "unknown",
        })
    })
}

fn token_cost_profile() -> &'static Mutex<TokenModelProfile> {
    TOKEN_COST_PROFILE.get_or_init(|| Mutex::new(TokenModelProfile::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_profile_is_disabled_by_default() {
        assert!(!LAYER_PROFILE_ENABLED.load(Ordering::Acquire));
        assert!(layer_profile_file().lock().unwrap().is_none());
    }

    #[test]
    fn disabled_layer_profile_does_not_touch_context() {
        *layer_profile_context().lock().unwrap() = LayerProfileContext {
            step_index: 0,
            phase: "unknown",
        };
        set_layer_profile_context(7, "decode").unwrap();
        assert_eq!(
            *layer_profile_context().lock().unwrap(),
            LayerProfileContext {
                step_index: 0,
                phase: "unknown",
            }
        );
    }

    #[test]
    fn token_profile_records_sparse_attention_substages_independently() {
        let mut profile = TokenModelProfile::default();
        profile.record(TokenProfileStage::SparseQProjection, 7);
        profile.record(TokenProfileStage::SparseQProjection, 5);
        profile.record(TokenProfileStage::SparseDsaIndexer, 3);

        assert_eq!(profile.sparse_q_projection_nanoseconds, 12);
        assert_eq!(profile.sparse_dsa_indexer_nanoseconds, 3);
        assert_eq!(profile.sparse_context_attention_nanoseconds, 0);
    }
}
