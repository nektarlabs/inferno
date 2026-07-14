use std::sync::Arc;

use ::metal::{
    Buffer, Device, Heap, HeapDescriptor, MTLCPUCacheMode, MTLHazardTrackingMode,
    MTLResourceOptions, MTLStorageMode,
};
use common::{Error, Result};
use tracing::debug;

use super::buffers::{write_f32_buffer, write_u32_buffer};

const TRANSIENT_ARENA_BYTES: u64 = 128 * 1024 * 1024;
const BUFFER_OPTIONS: MTLResourceOptions = MTLResourceOptions::StorageModeShared
    .union(MTLResourceOptions::CPUCacheModeDefaultCache)
    .union(MTLResourceOptions::HazardTrackingModeTracked);

/// Persistent Metal heap used for short-lived activation and parameter buffers.
///
/// Every returned `MTLBuffer` remains an independent tracked resource. Metal
/// returns its range to the heap when the final buffer handle is dropped, so
/// normal Rust tensor lifetimes define when storage can be reused. If one
/// unusually large prefill exceeds the bounded heap, allocation falls back to
/// the device rather than failing inference.
#[derive(Clone)]
pub(crate) struct MetalArena {
    inner: Arc<MetalArenaInner>,
}

struct MetalArenaInner {
    device: Device,
    heap: Heap,
}

impl MetalArena {
    pub(crate) fn new(device: &Device) -> Result<Self> {
        let descriptor = HeapDescriptor::new();
        descriptor.set_size(TRANSIENT_ARENA_BYTES);
        descriptor.set_storage_mode(MTLStorageMode::Shared);
        descriptor.set_cpu_cache_mode(MTLCPUCacheMode::DefaultCache);
        descriptor.set_hazard_tracking_mode(MTLHazardTrackingMode::Tracked);

        let heap = device.new_heap(&descriptor);
        heap.set_label("inferno transient activation arena");
        if heap.size() < TRANSIENT_ARENA_BYTES {
            return Err(Error::backend(format!(
                "Metal activation arena requested {TRANSIENT_ARENA_BYTES} bytes but received {}",
                heap.size()
            )));
        }

        Ok(Self {
            inner: Arc::new(MetalArenaInner {
                device: device.to_owned(),
                heap,
            }),
        })
    }

    pub(crate) fn empty_f32(&self, len: usize) -> Result<Buffer> {
        self.allocate_elements::<f32>(len, "f32")
    }

    pub(crate) fn empty_f16(&self, len: usize) -> Result<Buffer> {
        self.allocate_elements::<u16>(len, "f16")
    }

    pub(crate) fn empty_u32(&self, len: usize) -> Result<Buffer> {
        self.allocate_elements::<u32>(len, "u32")
    }

    pub(crate) fn u32(&self, value: u32) -> Result<Buffer> {
        let buffer = self.empty_u32(1)?;
        write_u32_buffer(&buffer, &[value])?;
        Ok(buffer)
    }

    pub(crate) fn f32(&self, value: f32) -> Result<Buffer> {
        let buffer = self.empty_f32(1)?;
        write_f32_buffer(&buffer, &[value])?;
        Ok(buffer)
    }

    fn allocate_elements<T>(&self, len: usize, label: &str) -> Result<Buffer> {
        let byte_len = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::backend(format!("Metal arena {label} byte length overflow")))?;
        let allocation_len = byte_len.max(1) as u64;

        if let Some(buffer) = self.inner.heap.new_buffer(allocation_len, BUFFER_OPTIONS) {
            return Ok(buffer);
        }

        debug!(
            target: "inferno::metal",
            requested_bytes = allocation_len,
            arena_used_bytes = self.inner.heap.used_size(),
            arena_capacity_bytes = self.inner.heap.size(),
            "Metal activation arena exhausted; using a direct buffer allocation"
        );
        Ok(self.inner.device.new_buffer(allocation_len, BUFFER_OPTIONS))
    }
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::MetalArena;

    #[test]
    fn allocates_tracked_shared_buffers_from_the_heap() {
        let Some(device) = ::metal::Device::system_default() else {
            return;
        };
        let arena = MetalArena::new(&device).unwrap();
        let buffer = arena.empty_f32(1024).unwrap();

        assert!(buffer.length() >= 4096);
        assert!(arena.inner.heap.used_size() >= 4096);
    }
}
