use std::{collections::HashMap, ffi::c_void, sync::Mutex};

use ::metal::{Buffer, Device, MTLResourceOptions};
use common::{Error, Result};
use inferno_io::{MappedBytes, SafeTensorHandle};

const BUFFER_OPTIONS: MTLResourceOptions = MTLResourceOptions::StorageModeShared
    .union(MTLResourceOptions::CPUCacheModeDefaultCache)
    .union(MTLResourceOptions::HazardTrackingModeUntracked);
const VIEW_BYTES: usize = 1 << 30;

pub(super) struct QwenBufferBinding {
    pub(super) buffer: Buffer,
    pub(super) byte_offset: usize,
}

pub(super) struct QwenShardViews {
    views: Mutex<HashMap<QwenViewKey, QwenShardView>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct QwenViewKey {
    mapping_identity: usize,
    byte_start: usize,
    byte_len: usize,
}

struct QwenShardView {
    buffer: Buffer,
    _mapping: MappedBytes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ViewRange {
    byte_start: usize,
    byte_len: usize,
}

impl QwenShardViews {
    pub(super) fn new() -> Self {
        Self {
            views: Mutex::new(HashMap::new()),
        }
    }

    pub(super) fn binding(
        &self,
        device: &Device,
        tensor: &SafeTensorHandle,
    ) -> Result<QwenBufferBinding> {
        let mapping = tensor.shared_mapping();
        let tensor_start = usize::try_from(tensor.info().file_offset)
            .map_err(|_| Error::backend("Qwen tensor offset does not fit usize"))?;
        let tensor_len = usize::try_from(tensor.info().byte_len)
            .map_err(|_| Error::backend("Qwen tensor length does not fit usize"))?;
        let page_size = host_page_size()?;
        let range = select_view_range(mapping.len(), tensor_start, tensor_len, page_size)?;
        let key = QwenViewKey {
            mapping_identity: mapping.cache_identity(),
            byte_start: range.byte_start,
            byte_len: range.byte_len,
        };
        let mut views = self
            .views
            .lock()
            .map_err(|_| Error::backend("Qwen Metal shard-view lock poisoned"))?;
        if !views.contains_key(&key) {
            let view = create_view(
                device,
                mapping,
                range,
                tensor.shard_path().display().to_string(),
            )?;
            views.insert(key, view);
        }
        let view = views
            .get(&key)
            .ok_or_else(|| Error::backend("Qwen Metal shard view disappeared"))?;
        let byte_offset = tensor_start - range.byte_start;
        let end = byte_offset
            .checked_add(tensor_len)
            .ok_or_else(|| Error::backend("Qwen tensor Metal range overflow"))?;
        if end > range.byte_len {
            return Err(Error::backend(format!(
                "Qwen tensor {} local range [{byte_offset}..{end}] exceeds Metal view size {}",
                tensor.info().name,
                range.byte_len
            )));
        }

        Ok(QwenBufferBinding {
            buffer: view.buffer.clone(),
            byte_offset,
        })
    }
}

fn create_view(
    device: &Device,
    mapping: MappedBytes,
    range: ViewRange,
    label: String,
) -> Result<QwenShardView> {
    if range.byte_len > device.max_buffer_length() as usize {
        return Err(Error::backend(format!(
            "Qwen Safetensors view for {label} needs {} bytes, exceeding maxBufferLength {}",
            range.byte_len,
            device.max_buffer_length()
        )));
    }

    // The range begins at a VM-page boundary. POSIX maps the final partial file
    // page as a complete zero-filled page, so its rounded length is valid.
    let pointer = mapping.as_slice()[range.byte_start..].as_ptr() as *mut c_void;
    let buffer =
        device.new_buffer_with_bytes_no_copy(pointer, range.byte_len as u64, BUFFER_OPTIONS, None);
    buffer.set_label(&format!(
        "qwen_safetensors:{label}:{}+{}",
        range.byte_start, range.byte_len
    ));
    Ok(QwenShardView {
        buffer,
        _mapping: mapping,
    })
}

fn select_view_range(
    mapping_len: usize,
    tensor_start: usize,
    tensor_len: usize,
    page_size: usize,
) -> Result<ViewRange> {
    if tensor_len == 0 || page_size == 0 || !VIEW_BYTES.is_multiple_of(page_size) {
        return Err(Error::backend("invalid Qwen Metal view dimensions"));
    }
    let tensor_end = tensor_start
        .checked_add(tensor_len)
        .ok_or_else(|| Error::backend("Qwen tensor range overflow"))?;
    if tensor_end > mapping_len {
        return Err(Error::backend(format!(
            "Qwen tensor range [{tensor_start}..{tensor_end}] exceeds mapping size {mapping_len}"
        )));
    }

    let regular_start = tensor_start / VIEW_BYTES * VIEW_BYTES;
    let regular_end = regular_start
        .checked_add(VIEW_BYTES)
        .ok_or_else(|| Error::backend("Qwen regular view range overflow"))?;
    let byte_start = if tensor_end <= regular_end {
        regular_start
    } else {
        tensor_start / page_size * page_size
    };
    let required_len = tensor_end - byte_start;
    let available_len = mapping_len - byte_start;
    let requested_len = if byte_start == regular_start {
        VIEW_BYTES.min(available_len)
    } else {
        required_len
    };
    let byte_len = round_up(requested_len, page_size)?;

    Ok(ViewRange {
        byte_start,
        byte_len,
    })
}

fn host_page_size() -> Result<usize> {
    // SAFETY: `_SC_PAGESIZE` takes no pointer arguments and is available on
    // the macOS target required by this module.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return Err(Error::backend(
            "failed to determine host page size for Qwen Metal shard views",
        ));
    }
    usize::try_from(page_size).map_err(|_| Error::backend("host page size does not fit usize"))
}

fn round_up(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded / alignment * alignment)
        .ok_or_else(|| Error::backend("Qwen Metal shard-view alignment overflow"))
}

#[cfg(test)]
mod tests {
    use super::{round_up, select_view_range, ViewRange};

    #[test]
    fn rounds_shard_views_to_complete_vm_pages() {
        assert_eq!(round_up(16_385, 16_384).unwrap(), 32_768);
        assert_eq!(round_up(32_768, 16_384).unwrap(), 32_768);
    }

    #[test]
    fn keeps_offsets_local_above_the_four_gib_boundary() {
        let gib = 1_usize << 30;
        let range = select_view_range(6 * gib, 4 * gib + 16_384, 64 * 1024, 16_384).unwrap();

        assert_eq!(
            range,
            ViewRange {
                byte_start: 4 * gib,
                byte_len: gib,
            }
        );
        assert_eq!(4 * gib + 16_384 - range.byte_start, 16_384);
    }

    #[test]
    fn creates_a_page_aligned_view_for_a_boundary_crossing_tensor() {
        let gib = 1_usize << 30;
        let tensor_start = gib - 32 * 1024;
        let range = select_view_range(3 * gib, tensor_start, 96 * 1024, 16_384).unwrap();

        assert_eq!(range.byte_start, tensor_start);
        assert_eq!(range.byte_len, 96 * 1024);
    }

    #[test]
    fn creates_a_dedicated_view_for_a_tensor_larger_than_one_gibibyte() {
        let gib = 1_usize << 30;
        let page = 16_384;
        let tensor_start = 2 * gib + page;
        let tensor_len = 2 * gib + 384 * 1024 * 1024;
        let range = select_view_range(6 * gib, tensor_start, tensor_len, page).unwrap();

        assert_eq!(range.byte_start, tensor_start);
        assert_eq!(range.byte_len, tensor_len);
    }
}
