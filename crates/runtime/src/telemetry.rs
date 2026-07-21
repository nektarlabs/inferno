use std::{
    fs::File,
    io::Write,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex, OnceLock,
    },
};

use backend::Backend;
use common::{Error, Result};
use tracing::info;

static MEMORY_TELEMETRY_ENABLED: AtomicBool = AtomicBool::new(false);
static MEMORY_TELEMETRY_FILE: OnceLock<Mutex<Option<File>>> = OnceLock::new();
const BYTES_PER_GB: f64 = 1_000_000_000.0;

pub(crate) fn enable_memory_telemetry() {
    MEMORY_TELEMETRY_ENABLED.store(true, Ordering::Release);
}

pub(crate) fn enable_memory_telemetry_file(path: &Path) -> Result<()> {
    let file = File::create(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = memory_telemetry_file()
        .lock()
        .map_err(|_| Error::runtime("memory telemetry file lock poisoned"))?;
    if writer.is_some() {
        return Err(Error::runtime("memory telemetry file is already enabled"));
    }
    *writer = Some(file);
    MEMORY_TELEMETRY_ENABLED.store(true, Ordering::Release);
    Ok(())
}

#[cfg(test)]
pub(crate) fn memory_telemetry_enabled() -> bool {
    MEMORY_TELEMETRY_ENABLED.load(Ordering::Acquire)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct RuntimeKvMemoryBytes {
    pub hot_bytes: Option<u64>,
    pub cold_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RuntimeMemorySnapshot {
    pub total_physical_bytes: Option<u64>,
    pub process_rss_bytes: Option<u64>,
    pub process_virtual_bytes: Option<u64>,
    pub system_free_bytes: Option<u64>,
    pub system_active_bytes: Option<u64>,
    pub system_inactive_bytes: Option<u64>,
    pub system_wired_bytes: Option<u64>,
    pub system_compressed_bytes: Option<u64>,
    pub system_purgeable_bytes: Option<u64>,
    pub system_speculative_bytes: Option<u64>,
    pub swap_used_bytes: Option<u64>,
    pub metal_current_allocated_bytes: Option<u64>,
    pub metal_recommended_max_working_set_bytes: Option<u64>,
    pub runtime_kv_hot_bytes: Option<u64>,
    pub runtime_kv_cold_bytes: Option<u64>,
}

impl RuntimeMemorySnapshot {
    pub(crate) fn effective_available_bytes(self) -> Option<u64> {
        let available = self
            .system_free_bytes?
            .saturating_add(self.system_inactive_bytes.unwrap_or(0))
            .saturating_add(self.system_purgeable_bytes.unwrap_or(0))
            .saturating_add(self.system_speculative_bytes.unwrap_or(0));
        Some(
            self.total_physical_bytes
                .map_or(available, |total| available.min(total)),
        )
    }

    pub(crate) fn metal_headroom_bytes(self) -> Option<u64> {
        Some(
            self.metal_recommended_max_working_set_bytes?
                .saturating_sub(self.metal_current_allocated_bytes?),
        )
    }
}

pub(crate) fn log_memory_snapshot<B: Backend>(
    stage: &'static str,
    step_index: Option<usize>,
    backend: &B,
    runtime_kv: RuntimeKvMemoryBytes,
) {
    if !MEMORY_TELEMETRY_ENABLED.load(Ordering::Acquire) {
        return;
    }

    let snapshot = capture_memory_snapshot(backend, runtime_kv);
    if write_memory_snapshot_to_file(stage, step_index, snapshot).is_some() {
        return;
    }

    info!(
        target: "inferno::memory",
        stage,
        step_index,
        total_physical_gb = %format_gb(snapshot.total_physical_bytes),
        process_rss_gb = %format_gb(snapshot.process_rss_bytes),
        process_virtual_gb = %format_gb(snapshot.process_virtual_bytes),
        system_free_gb = %format_gb(snapshot.system_free_bytes),
        system_active_gb = %format_gb(snapshot.system_active_bytes),
        system_inactive_gb = %format_gb(snapshot.system_inactive_bytes),
        system_wired_gb = %format_gb(snapshot.system_wired_bytes),
        system_compressed_gb = %format_gb(snapshot.system_compressed_bytes),
        system_purgeable_gb = %format_gb(snapshot.system_purgeable_bytes),
        system_speculative_gb = %format_gb(snapshot.system_speculative_bytes),
        effective_available_gb = %format_gb(snapshot.effective_available_bytes()),
        swap_used_gb = %format_gb(snapshot.swap_used_bytes),
        metal_current_allocated_gb = %format_gb(snapshot.metal_current_allocated_bytes),
        metal_recommended_max_working_set_gb = %format_gb(snapshot.metal_recommended_max_working_set_bytes),
        metal_headroom_gb = %format_gb(snapshot.metal_headroom_bytes()),
        runtime_kv_hot_gb = %format_gb(snapshot.runtime_kv_hot_bytes),
        runtime_kv_cold_gb = %format_gb(snapshot.runtime_kv_cold_bytes),
        "runtime memory snapshot"
    );
}

fn write_memory_snapshot_to_file(
    stage: &'static str,
    step_index: Option<usize>,
    snapshot: RuntimeMemorySnapshot,
) -> Option<()> {
    let mut writer = memory_telemetry_file().lock().ok()?;
    let writer = writer.as_mut()?;
    writeln!(
        writer,
        "runtime memory snapshot stage=\"{}\" step_index={} total_physical_gb={} process_rss_gb={} process_virtual_gb={} system_free_gb={} system_active_gb={} system_inactive_gb={} system_wired_gb={} system_compressed_gb={} system_purgeable_gb={} system_speculative_gb={} effective_available_gb={} swap_used_gb={} metal_current_allocated_gb={} metal_recommended_max_working_set_gb={} metal_headroom_gb={} runtime_kv_hot_gb={} runtime_kv_cold_gb={}",
        stage,
        step_index
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        format_gb(snapshot.total_physical_bytes),
        format_gb(snapshot.process_rss_bytes),
        format_gb(snapshot.process_virtual_bytes),
        format_gb(snapshot.system_free_bytes),
        format_gb(snapshot.system_active_bytes),
        format_gb(snapshot.system_inactive_bytes),
        format_gb(snapshot.system_wired_bytes),
        format_gb(snapshot.system_compressed_bytes),
        format_gb(snapshot.system_purgeable_bytes),
        format_gb(snapshot.system_speculative_bytes),
        format_gb(snapshot.effective_available_bytes()),
        format_gb(snapshot.swap_used_bytes),
        format_gb(snapshot.metal_current_allocated_bytes),
        format_gb(snapshot.metal_recommended_max_working_set_bytes),
        format_gb(snapshot.metal_headroom_bytes()),
        format_gb(snapshot.runtime_kv_hot_bytes),
        format_gb(snapshot.runtime_kv_cold_bytes),
    )
    .ok()?;
    writer.flush().ok()?;
    Some(())
}

fn format_gb(bytes: Option<u64>) -> String {
    match bytes {
        Some(bytes) => format!("{:.3}", bytes as f64 / BYTES_PER_GB),
        None => "unknown".to_string(),
    }
}

pub(crate) fn capture_memory_snapshot<B: Backend>(
    backend: &B,
    runtime_kv: RuntimeKvMemoryBytes,
) -> RuntimeMemorySnapshot {
    let backend = backend.memory_report();
    RuntimeMemorySnapshot {
        total_physical_bytes: backend.total_physical_bytes,
        process_rss_bytes: backend.process_rss_bytes,
        process_virtual_bytes: backend.process_virtual_bytes,
        system_free_bytes: backend.system_free_bytes,
        system_active_bytes: backend.system_active_bytes,
        system_inactive_bytes: backend.system_inactive_bytes,
        system_wired_bytes: backend.system_wired_bytes,
        system_compressed_bytes: backend.system_compressed_bytes,
        system_purgeable_bytes: backend.system_purgeable_bytes,
        system_speculative_bytes: backend.system_speculative_bytes,
        swap_used_bytes: backend.swap_used_bytes,
        metal_current_allocated_bytes: backend.metal_current_allocated_bytes,
        metal_recommended_max_working_set_bytes: backend.metal_recommended_max_working_set_bytes,
        runtime_kv_hot_bytes: runtime_kv.hot_bytes,
        runtime_kv_cold_bytes: runtime_kv.cold_bytes,
    }
}

fn memory_telemetry_file() -> &'static Mutex<Option<File>> {
    MEMORY_TELEMETRY_FILE.get_or_init(|| Mutex::new(None))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_bytes_as_decimal_gb() {
        assert_eq!(format_gb(Some(1_500_000_000)), "1.500");
        assert_eq!(format_gb(None), "unknown");
    }

    #[test]
    fn calculates_effective_available_memory_from_reclaimable_pages() {
        let snapshot = RuntimeMemorySnapshot {
            total_physical_bytes: Some(64_000),
            process_rss_bytes: None,
            process_virtual_bytes: None,
            system_free_bytes: Some(1_000),
            system_active_bytes: None,
            system_inactive_bytes: Some(2_000),
            system_wired_bytes: None,
            system_compressed_bytes: None,
            system_purgeable_bytes: Some(3_000),
            system_speculative_bytes: Some(4_000),
            swap_used_bytes: None,
            metal_current_allocated_bytes: Some(40_000),
            metal_recommended_max_working_set_bytes: Some(55_000),
            runtime_kv_hot_bytes: None,
            runtime_kv_cold_bytes: None,
        };

        assert_eq!(snapshot.effective_available_bytes(), Some(10_000));
        assert_eq!(snapshot.metal_headroom_bytes(), Some(15_000));
    }
}
