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

use common::{Error, Result};

static LAYER_PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);
static LAYER_PROFILE_FILE: OnceLock<Mutex<Option<File>>> = OnceLock::new();
static LAYER_PROFILE_CONTEXT: OnceLock<Mutex<LayerProfileContext>> = OnceLock::new();

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
}
