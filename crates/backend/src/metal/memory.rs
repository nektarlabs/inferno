use std::{ffi::CString, mem::MaybeUninit, sync::OnceLock};

use crate::BackendMemoryReport;

pub(super) fn native_memory_report() -> BackendMemoryReport {
    let mut report = BackendMemoryReport::default();

    if let Some((rss, virtual_size)) = task_memory_bytes() {
        report.process_rss_bytes = Some(rss);
        report.process_virtual_bytes = Some(virtual_size);
    }
    if let Some(vm) = host_vm_bytes() {
        report.system_free_bytes = Some(vm.free);
        report.system_active_bytes = Some(vm.active);
        report.system_inactive_bytes = Some(vm.inactive);
        report.system_wired_bytes = Some(vm.wired);
        report.system_compressed_bytes = Some(vm.compressed);
        report.system_purgeable_bytes = Some(vm.purgeable);
        report.system_speculative_bytes = Some(vm.speculative);
    }
    report.total_physical_bytes = sysctl_value::<u64>("hw.memsize");
    report.swap_used_bytes =
        sysctl_value::<libc::xsw_usage>("vm.swapusage").map(|usage| usage.xsu_used);
    report
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct HostVmBytes {
    free: u64,
    active: u64,
    inactive: u64,
    wired: u64,
    compressed: u64,
    purgeable: u64,
    speculative: u64,
}

#[allow(deprecated)]
fn task_memory_bytes() -> Option<(u64, u64)> {
    let mut info = MaybeUninit::<libc::mach_task_basic_info_data_t>::zeroed();
    let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
    // SAFETY: Mach writes at most `count` integer words into the correctly
    // sized output structure. The current task port remains valid for the
    // process lifetime.
    let status = unsafe {
        let task = libc::mach_task_self_;
        libc::task_info(
            task,
            libc::MACH_TASK_BASIC_INFO,
            info.as_mut_ptr().cast::<libc::integer_t>(),
            &mut count,
        )
    };
    if status != libc::KERN_SUCCESS || count < libc::MACH_TASK_BASIC_INFO_COUNT {
        return None;
    }
    // SAFETY: `task_info` succeeded and initialized the complete structure.
    let info = unsafe { info.assume_init() };
    Some((info.resident_size, info.virtual_size))
}

#[allow(deprecated)]
fn host_vm_bytes() -> Option<HostVmBytes> {
    static HOST_PORT: OnceLock<libc::mach_port_t> = OnceLock::new();
    let host = *HOST_PORT.get_or_init(|| {
        // SAFETY: returns the current host's send right. It is retained for the
        // process lifetime instead of allocating one right per sample.
        unsafe { libc::mach_host_self() }
    });
    let mut stats = MaybeUninit::<libc::vm_statistics64_data_t>::zeroed();
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: Mach writes at most `count` integer words into the correctly
    // sized VM statistics structure.
    let status = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            stats.as_mut_ptr().cast::<libc::integer_t>(),
            &mut count,
        )
    };
    if status != libc::KERN_SUCCESS || count < libc::HOST_VM_INFO64_COUNT {
        return None;
    }
    // SAFETY: `host_statistics64` succeeded and initialized the structure.
    let stats = unsafe { stats.assume_init() };
    // SAFETY: `sysconf` reads a process-global constant and has no output
    // pointer or mutable state owned by Rust.
    let page_size = u64::try_from(unsafe { libc::sysconf(libc::_SC_PAGESIZE) }).ok()?;
    let bytes = |pages: libc::natural_t| u64::from(pages).checked_mul(page_size);
    Some(HostVmBytes {
        free: bytes(stats.free_count)?,
        active: bytes(stats.active_count)?,
        inactive: bytes(stats.inactive_count)?,
        wired: bytes(stats.wire_count)?,
        compressed: bytes(stats.compressor_page_count)?,
        purgeable: bytes(stats.purgeable_count)?,
        speculative: bytes(stats.speculative_count)?,
    })
}

fn sysctl_value<T>(name: &str) -> Option<T> {
    let name = CString::new(name).ok()?;
    let mut value = MaybeUninit::<T>::uninit();
    let mut size = std::mem::size_of::<T>();
    // SAFETY: `value` points to `size_of::<T>()` writable bytes and no new
    // value is supplied because this is a read-only sysctl call.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            value.as_mut_ptr().cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    if status != 0 || size != std::mem::size_of::<T>() {
        return None;
    }
    // SAFETY: a successful sysctl call initialized exactly one `T`.
    Some(unsafe { value.assume_init() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_current_process_and_host_memory_without_subprocesses() {
        let report = native_memory_report();

        assert!(report.total_physical_bytes.is_some_and(|bytes| bytes > 0));
        assert!(report.process_rss_bytes.is_some_and(|bytes| bytes > 0));
        assert!(report.process_virtual_bytes.is_some_and(|bytes| bytes > 0));
        assert!(report.system_free_bytes.is_some());
        assert!(report.system_compressed_bytes.is_some());
        assert!(report.swap_used_bytes.is_some());
    }
}
