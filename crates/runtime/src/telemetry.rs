use std::{
    fs::File,
    io::Write,
    path::Path,
    process::Command,
    str::FromStr,
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
    pub process_rss_bytes: Option<u64>,
    pub process_virtual_bytes: Option<u64>,
    pub system_free_bytes: Option<u64>,
    pub system_active_bytes: Option<u64>,
    pub system_inactive_bytes: Option<u64>,
    pub system_wired_bytes: Option<u64>,
    pub system_compressed_bytes: Option<u64>,
    pub swap_used_bytes: Option<u64>,
    pub metal_current_allocated_bytes: Option<u64>,
    pub metal_recommended_max_working_set_bytes: Option<u64>,
    pub runtime_kv_hot_bytes: Option<u64>,
    pub runtime_kv_cold_bytes: Option<u64>,
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

    let snapshot = memory_snapshot(backend, runtime_kv);
    if write_memory_snapshot_to_file(stage, step_index, snapshot).is_some() {
        return;
    }

    info!(
        target: "inferno::memory",
        stage,
        step_index,
        process_rss_gb = %format_gb(snapshot.process_rss_bytes),
        process_virtual_gb = %format_gb(snapshot.process_virtual_bytes),
        system_free_gb = %format_gb(snapshot.system_free_bytes),
        system_active_gb = %format_gb(snapshot.system_active_bytes),
        system_inactive_gb = %format_gb(snapshot.system_inactive_bytes),
        system_wired_gb = %format_gb(snapshot.system_wired_bytes),
        system_compressed_gb = %format_gb(snapshot.system_compressed_bytes),
        swap_used_gb = %format_gb(snapshot.swap_used_bytes),
        metal_current_allocated_gb = %format_gb(snapshot.metal_current_allocated_bytes),
        metal_recommended_max_working_set_gb = %format_gb(snapshot.metal_recommended_max_working_set_bytes),
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
        "runtime memory snapshot stage=\"{}\" step_index={} process_rss_gb={} process_virtual_gb={} system_free_gb={} system_active_gb={} system_inactive_gb={} system_wired_gb={} system_compressed_gb={} swap_used_gb={} metal_current_allocated_gb={} metal_recommended_max_working_set_gb={} runtime_kv_hot_gb={} runtime_kv_cold_gb={}",
        stage,
        step_index
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        format_gb(snapshot.process_rss_bytes),
        format_gb(snapshot.process_virtual_bytes),
        format_gb(snapshot.system_free_bytes),
        format_gb(snapshot.system_active_bytes),
        format_gb(snapshot.system_inactive_bytes),
        format_gb(snapshot.system_wired_bytes),
        format_gb(snapshot.system_compressed_bytes),
        format_gb(snapshot.swap_used_bytes),
        format_gb(snapshot.metal_current_allocated_bytes),
        format_gb(snapshot.metal_recommended_max_working_set_bytes),
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

fn memory_snapshot<B: Backend>(
    backend: &B,
    runtime_kv: RuntimeKvMemoryBytes,
) -> RuntimeMemorySnapshot {
    let os = os_memory_snapshot();
    let backend = backend.memory_report();
    RuntimeMemorySnapshot {
        process_rss_bytes: os.process_rss_bytes,
        process_virtual_bytes: os.process_virtual_bytes,
        system_free_bytes: os.system_free_bytes,
        system_active_bytes: os.system_active_bytes,
        system_inactive_bytes: os.system_inactive_bytes,
        system_wired_bytes: os.system_wired_bytes,
        system_compressed_bytes: os.system_compressed_bytes,
        swap_used_bytes: os.swap_used_bytes,
        metal_current_allocated_bytes: backend.metal_current_allocated_bytes,
        metal_recommended_max_working_set_bytes: backend.metal_recommended_max_working_set_bytes,
        runtime_kv_hot_bytes: runtime_kv.hot_bytes,
        runtime_kv_cold_bytes: runtime_kv.cold_bytes,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct OsMemorySnapshot {
    process_rss_bytes: Option<u64>,
    process_virtual_bytes: Option<u64>,
    system_free_bytes: Option<u64>,
    system_active_bytes: Option<u64>,
    system_inactive_bytes: Option<u64>,
    system_wired_bytes: Option<u64>,
    system_compressed_bytes: Option<u64>,
    swap_used_bytes: Option<u64>,
}

fn os_memory_snapshot() -> OsMemorySnapshot {
    let process = process_memory_bytes();
    let mut snapshot = OsMemorySnapshot {
        process_rss_bytes: process.process_rss_bytes,
        process_virtual_bytes: process.process_virtual_bytes,
        ..OsMemorySnapshot::default()
    };

    if cfg!(target_os = "macos") {
        if let Some(vm_stat) = run_command("/usr/bin/vm_stat", &[]) {
            snapshot = merge_vm_stat(snapshot, &vm_stat);
        }
        if let Some(swap) = run_command("/usr/sbin/sysctl", &["-n", "vm.swapusage"]) {
            snapshot.swap_used_bytes = parse_swapusage_used_bytes(&swap);
        }
    }

    snapshot
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct ProcessMemoryBytes {
    process_rss_bytes: Option<u64>,
    process_virtual_bytes: Option<u64>,
}

fn process_memory_bytes() -> ProcessMemoryBytes {
    let pid = std::process::id().to_string();
    let Some(output) = run_command("/bin/ps", &["-o", "rss=,vsz=", "-p", &pid]) else {
        return ProcessMemoryBytes::default();
    };
    parse_ps_memory_bytes(&output)
}

fn run_command(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

fn memory_telemetry_file() -> &'static Mutex<Option<File>> {
    MEMORY_TELEMETRY_FILE.get_or_init(|| Mutex::new(None))
}

fn merge_vm_stat(mut snapshot: OsMemorySnapshot, vm_stat: &str) -> OsMemorySnapshot {
    let page_size = parse_vm_page_size(vm_stat).unwrap_or(4096);
    snapshot.system_free_bytes =
        parse_vm_page_count(vm_stat, "Pages free").and_then(|pages| pages.checked_mul(page_size));
    snapshot.system_active_bytes =
        parse_vm_page_count(vm_stat, "Pages active").and_then(|pages| pages.checked_mul(page_size));
    snapshot.system_inactive_bytes = parse_vm_page_count(vm_stat, "Pages inactive")
        .and_then(|pages| pages.checked_mul(page_size));
    snapshot.system_wired_bytes = parse_vm_page_count(vm_stat, "Pages wired down")
        .and_then(|pages| pages.checked_mul(page_size));
    snapshot.system_compressed_bytes = parse_vm_page_count(vm_stat, "Pages occupied by compressor")
        .and_then(|pages| pages.checked_mul(page_size));
    snapshot
}

fn parse_ps_memory_bytes(output: &str) -> ProcessMemoryBytes {
    let mut fields = output.split_whitespace();
    let process_rss_bytes = fields
        .next()
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|kib| kib.checked_mul(1024));
    let process_virtual_bytes = fields
        .next()
        .and_then(|value| value.parse::<u64>().ok())
        .and_then(|kib| kib.checked_mul(1024));
    ProcessMemoryBytes {
        process_rss_bytes,
        process_virtual_bytes,
    }
}

fn parse_vm_page_size(output: &str) -> Option<u64> {
    let first_line = output.lines().next()?;
    let start = first_line.find("page size of ")? + "page size of ".len();
    let suffix = &first_line[start..];
    let end = suffix.find(" bytes")?;
    suffix[..end].trim().parse::<u64>().ok()
}

fn parse_vm_page_count(output: &str, label: &str) -> Option<u64> {
    for line in output.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix(label) else {
            continue;
        };
        let Some(value) = rest.split(':').nth(1) else {
            continue;
        };
        let value = value.trim().trim_end_matches('.').replace('_', "");
        return u64::from_str(&value).ok();
    }
    None
}

fn parse_swapusage_used_bytes(output: &str) -> Option<u64> {
    let used = output.split_whitespace().collect::<Vec<_>>();
    let index = used.iter().position(|part| *part == "used")?;
    let value = *used.get(index + 2)?;
    parse_binary_size_bytes(value)
}

fn parse_binary_size_bytes(value: &str) -> Option<u64> {
    let value = value.trim().trim_end_matches(',');
    let unit_start = value
        .find(|character: char| !character.is_ascii_digit() && character != '.')
        .unwrap_or(value.len());
    let number = value[..unit_start].parse::<f64>().ok()?;
    let unit = value[unit_start..].to_ascii_uppercase();
    let multiplier = match unit.as_str() {
        "K" | "KB" => 1024.0,
        "M" | "MB" => 1024.0 * 1024.0,
        "G" | "GB" => 1024.0 * 1024.0 * 1024.0,
        "T" | "TB" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        "" => 1.0,
        _ => return None,
    };
    Some((number * multiplier) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ps_rss_as_bytes() {
        assert_eq!(
            parse_ps_memory_bytes(" 1234 5678\n"),
            ProcessMemoryBytes {
                process_rss_bytes: Some(1_263_616),
                process_virtual_bytes: Some(5_814_272),
            }
        );
    }

    #[test]
    fn formats_bytes_as_decimal_gb() {
        assert_eq!(format_gb(Some(1_500_000_000)), "1.500");
        assert_eq!(format_gb(None), "unknown");
    }

    #[test]
    fn parses_vm_stat_page_values() {
        let vm_stat = "\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100.
Pages active:                             200.
Pages inactive:                           300.
Pages wired down:                         400.
Pages occupied by compressor:             500.
";
        let snapshot = merge_vm_stat(OsMemorySnapshot::default(), vm_stat);
        assert_eq!(snapshot.system_free_bytes, Some(1_638_400));
        assert_eq!(snapshot.system_active_bytes, Some(3_276_800));
        assert_eq!(snapshot.system_inactive_bytes, Some(4_915_200));
        assert_eq!(snapshot.system_wired_bytes, Some(6_553_600));
        assert_eq!(snapshot.system_compressed_bytes, Some(8_192_000));
    }

    #[test]
    fn parses_swap_usage() {
        assert_eq!(
            parse_swapusage_used_bytes("total = 2048.00M  used = 512.50M  free = 1535.50M"),
            Some(537_395_200)
        );
    }
}
