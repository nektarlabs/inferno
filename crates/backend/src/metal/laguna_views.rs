use std::{
    ffi::{c_void, CStr},
    os::raw::c_char,
    ptr,
    sync::Mutex,
};

use ::metal::{
    foreign_types::ForeignType, Buffer, CommandQueue, ComputePipelineState, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize, NSUInteger,
};
use common::{Error, Result};
use inferno_io::MappedBytes;
use objc::{
    msg_send,
    rc::StrongPtr,
    runtime::{Class, Object, BOOL, NO},
    sel, sel_impl,
};

use super::{buffers::empty_u8_buffer, library::MetalLibrary, pipeline::compute_pipeline};
use crate::LagunaModelViewReport;

const WARMUP_STRIDE_BYTES: usize = 1024 * 1024;
const WARMUP_THREADS_PER_GROUP: usize = 256;
const BUFFER_OPTIONS: MTLResourceOptions =
    MTLResourceOptions::StorageModeShared.union(MTLResourceOptions::CPUCacheModeDefaultCache);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ViewRange {
    offset: usize,
    byte_len: usize,
}

pub(crate) struct MetalLagunaViews {
    warmup_pipeline: ComputePipelineState,
    prepared: Mutex<Option<PreparedViews>>,
}

struct PreparedViews {
    _residency: ResidencySet,
    views: Vec<ModelView>,
    _mapping: MappedBytes,
    report: LagunaModelViewReport,
}

struct ModelView {
    range: ViewRange,
    buffer: Buffer,
}

pub(crate) struct LagunaBufferBinding {
    pub(crate) buffer: Buffer,
    pub(crate) byte_offset: usize,
}

struct ResidencySet {
    object: StrongPtr,
    queue: CommandQueue,
}

// Metal resources and command queues are explicitly designed for concurrent
// encoding. `StrongPtr` does not express that Objective-C contract in Rust.
unsafe impl Send for ResidencySet {}
unsafe impl Sync for ResidencySet {}

impl MetalLagunaViews {
    pub(crate) fn new(device: &Device, library: &MetalLibrary) -> Result<Self> {
        Ok(Self {
            warmup_pipeline: compute_pipeline(device, library, "laguna_touch_model_view_kernel")?,
            prepared: Mutex::new(None),
        })
    }

    pub(crate) fn prepare(
        &self,
        device: &Device,
        queue: &CommandQueue,
        mapping: MappedBytes,
        tensor_data_offset: usize,
        max_tensor_bytes: usize,
    ) -> Result<LagunaModelViewReport> {
        let mut prepared = self
            .prepared
            .lock()
            .map_err(|_| Error::backend("Laguna Metal model-view lock poisoned"))?;
        if let Some(existing) = prepared.as_ref() {
            if existing._mapping.cache_identity() != mapping.cache_identity() {
                return Err(Error::backend(
                    "one Metal backend cannot own two different Laguna GGUF mappings",
                ));
            }
            return Ok(existing.report);
        }

        let page_size = host_page_size()?;
        let ranges = plan_view_ranges(
            mapping.len(),
            tensor_data_offset,
            usize::try_from(device.max_buffer_length())
                .map_err(|_| Error::backend("Metal maxBufferLength does not fit usize"))?,
            max_tensor_bytes,
            page_size,
        )?;
        let mut views = Vec::with_capacity(ranges.len());
        for (index, range) in ranges.iter().enumerate() {
            let buffer = mapped_buffer(device, &mapping, *range, page_size)?;
            buffer.set_label(&format!("laguna_model_view_{index}"));
            views.push(ModelView {
                range: *range,
                buffer,
            });
        }

        let residency = ResidencySet::new(device, queue, &views)?;
        let warmup_samples = self.warm(device, queue, &views)?;
        let view_bytes = ranges.iter().try_fold(0_usize, |total, range| {
            total
                .checked_add(range.byte_len)
                .ok_or_else(|| Error::backend("Laguna Metal view-byte count overflow"))
        })?;
        let report = LagunaModelViewReport {
            view_count: views.len(),
            model_bytes: mapping.len().saturating_sub(tensor_data_offset),
            view_bytes,
            max_view_bytes: ranges.iter().map(|range| range.byte_len).max().unwrap_or(0),
            warmup_samples,
        };

        *prepared = Some(PreparedViews {
            _residency: residency,
            views,
            _mapping: mapping,
            report,
        });
        Ok(report)
    }

    pub(crate) fn binding(&self, bytes: &[u8]) -> Result<Option<LagunaBufferBinding>> {
        if bytes.is_empty() {
            return Ok(None);
        }
        let prepared = self
            .prepared
            .lock()
            .map_err(|_| Error::backend("Laguna Metal model-view lock poisoned"))?;
        let Some(prepared) = prepared.as_ref() else {
            return Ok(None);
        };
        let mapping_start = prepared._mapping.cache_identity();
        let mapping_end = mapping_start
            .checked_add(prepared._mapping.len())
            .ok_or_else(|| Error::backend("Laguna GGUF mapping address range overflow"))?;
        let bytes_start = bytes.as_ptr() as usize;
        let bytes_end = bytes_start
            .checked_add(bytes.len())
            .ok_or_else(|| Error::backend("Laguna tensor address range overflow"))?;
        if bytes_start < mapping_start || bytes_end > mapping_end {
            return Ok(None);
        }
        let tensor_offset = bytes_start - mapping_start;
        let view = prepared
            .views
            .iter()
            .find(|view| {
                tensor_offset >= view.range.offset
                    && bytes_end - mapping_start <= view.range.offset + view.range.byte_len
            })
            .ok_or_else(|| {
                Error::backend(format!(
                    "Laguna tensor range [{tensor_offset}..{}] is not contained in a persistent Metal model view",
                    tensor_offset + bytes.len()
                ))
            })?;
        Ok(Some(LagunaBufferBinding {
            buffer: view.buffer.clone(),
            byte_offset: tensor_offset - view.range.offset,
        }))
    }

    fn warm(&self, device: &Device, queue: &CommandQueue, views: &[ModelView]) -> Result<usize> {
        let sample_counts = views
            .iter()
            .map(|view| {
                usize::try_from(view.buffer.length())
                    .map_err(|_| Error::backend("Laguna Metal view length does not fit usize"))
                    .map(|bytes| bytes.div_ceil(WARMUP_STRIDE_BYTES))
            })
            .collect::<Result<Vec<_>>>()?;
        let total_samples = sample_counts.iter().try_fold(0_usize, |total, count| {
            total
                .checked_add(*count)
                .ok_or_else(|| Error::backend("Laguna Metal warm-up sample count overflow"))
        })?;
        if total_samples == 0 {
            return Err(Error::backend(
                "Laguna Metal model views produced no warm-up samples",
            ));
        }
        let samples = empty_u8_buffer(device, total_samples)?;
        samples.set_label("laguna_model_view_warmup");

        let command_buffer = queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.warmup_pipeline);
        let stride = WARMUP_STRIDE_BYTES as u64;
        let mut sample_offset = 0_u64;
        for (view, sample_count) in views.iter().zip(sample_counts) {
            let source_bytes = view.buffer.length();
            encoder.set_buffer(0, Some(&view.buffer), 0);
            encoder.set_buffer(1, Some(&samples), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of_val(&stride) as NSUInteger,
                (&stride as *const u64).cast::<c_void>(),
            );
            encoder.set_bytes(
                3,
                std::mem::size_of_val(&source_bytes) as NSUInteger,
                (&source_bytes as *const u64).cast::<c_void>(),
            );
            encoder.set_bytes(
                4,
                std::mem::size_of_val(&sample_offset) as NSUInteger,
                (&sample_offset as *const u64).cast::<c_void>(),
            );
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    sample_count.div_ceil(WARMUP_THREADS_PER_GROUP) as NSUInteger,
                    1,
                    1,
                ),
                MTLSize::new(WARMUP_THREADS_PER_GROUP as NSUInteger, 1, 1),
            );
            sample_offset = sample_offset
                .checked_add(sample_count as u64)
                .ok_or_else(|| Error::backend("Laguna Metal warm-up offset overflow"))?;
        }
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        match command_buffer.status() {
            MTLCommandBufferStatus::Completed => Ok(total_samples),
            status => Err(Error::backend(format!(
                "Laguna Metal model-view warm-up did not complete: {status:?}"
            ))),
        }
    }
}

impl ResidencySet {
    fn new(device: &Device, queue: &CommandQueue, views: &[ModelView]) -> Result<Self> {
        let device_object = device.as_ptr().cast::<Object>();
        let queue_object = queue.as_ptr().cast::<Object>();
        if !responds_to(device_object, sel!(newResidencySetWithDescriptor:error:))
            || !responds_to(queue_object, sel!(addResidencySet:))
        {
            return Err(Error::backend(
                "Laguna Q2/Q3 model views require the macOS 15 Metal residency-set API",
            ));
        }
        let descriptor_class = Class::get("MTLResidencySetDescriptor").ok_or_else(|| {
            Error::backend("MTLResidencySetDescriptor is unavailable on this macOS version")
        })?;

        // SAFETY: Objective-C `new` returns a retained descriptor object.
        let descriptor_pointer: *mut Object = unsafe { msg_send![descriptor_class, new] };
        if descriptor_pointer.is_null() {
            return Err(Error::backend(
                "Metal failed to allocate a Laguna residency-set descriptor",
            ));
        }
        // SAFETY: `descriptor_pointer` is non-null and carries a +1 retain count.
        let descriptor = unsafe { StrongPtr::new(descriptor_pointer) };
        let initial_capacity = views.len() as NSUInteger;
        // SAFETY: this selector and argument are part of MTLResidencySetDescriptor.
        unsafe {
            let _: () = msg_send![*descriptor, setInitialCapacity: initial_capacity];
        }

        let mut creation_error: *mut Object = ptr::null_mut();
        // SAFETY: selector availability was checked and both pointers reference
        // live Metal objects for the duration of this call.
        let residency_pointer: *mut Object = unsafe {
            msg_send![
                device_object,
                newResidencySetWithDescriptor: *descriptor
                error: &mut creation_error
            ]
        };
        if residency_pointer.is_null() {
            let detail = objective_c_error(creation_error)
                .unwrap_or_else(|| "unknown Metal error".to_string());
            return Err(Error::backend(format!(
                "failed to create Laguna Metal residency set: {detail}"
            )));
        }
        // SAFETY: `newResidencySetWithDescriptor:error:` returns a retained set.
        let object = unsafe { StrongPtr::new(residency_pointer) };
        for view in views {
            let allocation = view.buffer.as_ptr().cast::<Object>();
            // SAFETY: every view is a live MTLBuffer implementing MTLAllocation.
            unsafe {
                let _: () = msg_send![*object, addAllocation: allocation];
            }
        }
        // SAFETY: these are parameterless MTLResidencySet lifecycle methods.
        unsafe {
            let _: () = msg_send![*object, commit];
            let _: () = msg_send![*object, requestResidency];
            let _: () = msg_send![queue_object, addResidencySet: *object];
        }

        Ok(Self {
            object,
            queue: queue.clone(),
        })
    }
}

impl Drop for ResidencySet {
    fn drop(&mut self) {
        let queue_object = self.queue.as_ptr().cast::<Object>();
        // SAFETY: the retained residency set and queue remain live until this
        // destructor returns. Removing the set precedes releasing allocations.
        unsafe {
            let _: () = msg_send![queue_object, removeResidencySet: *self.object];
            let _: () = msg_send![*self.object, endResidency];
            let _: () = msg_send![*self.object, removeAllAllocations];
        }
    }
}

fn mapped_buffer(
    device: &Device,
    mapping: &MappedBytes,
    range: ViewRange,
    page_size: usize,
) -> Result<Buffer> {
    let rounded_mapping_len = round_up(mapping.len(), page_size)?;
    let end = range
        .offset
        .checked_add(range.byte_len)
        .ok_or_else(|| Error::backend("Laguna Metal model-view range overflow"))?;
    if range.offset >= mapping.len()
        || end > rounded_mapping_len
        || !range.offset.is_multiple_of(page_size)
        || !range.byte_len.is_multiple_of(page_size)
    {
        return Err(Error::backend(format!(
            "invalid page-aligned Laguna Metal model view [{}..{}] for {} mapped bytes",
            range.offset,
            end,
            mapping.len()
        )));
    }

    // SAFETY: the mmap base and offset are page-aligned. POSIX maps the final
    // partial file page as a complete zero-padded VM page, which is why the
    // validated view may end at `round_up(mapping.len(), page_size)`.
    let pointer = unsafe { mapping.as_ptr().add(range.offset) } as *mut c_void;
    Ok(device.new_buffer_with_bytes_no_copy(pointer, range.byte_len as u64, BUFFER_OPTIONS, None))
}

fn plan_view_ranges(
    mapping_len: usize,
    tensor_data_offset: usize,
    max_buffer_bytes: usize,
    max_tensor_bytes: usize,
    page_size: usize,
) -> Result<Vec<ViewRange>> {
    if page_size == 0 || !page_size.is_power_of_two() {
        return Err(Error::backend(format!(
            "host page size must be a non-zero power of two, got {page_size}"
        )));
    }
    if tensor_data_offset >= mapping_len || max_tensor_bytes == 0 {
        return Err(Error::backend(format!(
            "invalid Laguna GGUF view range: data_offset={tensor_data_offset}, mapping_len={mapping_len}, max_tensor={max_tensor_bytes}"
        )));
    }

    let aligned_start = tensor_data_offset / page_size * page_size;
    let leading_bytes = tensor_data_offset - aligned_start;
    let tensor_data_bytes = mapping_len - tensor_data_offset;
    let mapped_bytes = round_up(
        leading_bytes
            .checked_add(tensor_data_bytes)
            .ok_or_else(|| Error::backend("Laguna Metal mapped data length overflow"))?,
        page_size,
    )?;
    let max_buffer_bytes = max_buffer_bytes / page_size * page_size;
    let overlap = round_up(max_tensor_bytes, page_size)?
        .checked_add(page_size)
        .ok_or_else(|| Error::backend("Laguna Metal model-view overlap overflow"))?;
    if max_buffer_bytes <= overlap {
        return Err(Error::backend(format!(
            "Metal maxBufferLength {max_buffer_bytes} cannot contain Laguna tensor span {max_tensor_bytes} with overlap"
        )));
    }

    let step = max_buffer_bytes - overlap;
    let mut ranges = Vec::new();
    let mut relative_offset = 0_usize;
    while relative_offset < mapped_bytes {
        let byte_len = (mapped_bytes - relative_offset).min(max_buffer_bytes);
        ranges.push(ViewRange {
            offset: aligned_start
                .checked_add(relative_offset)
                .ok_or_else(|| Error::backend("Laguna Metal model-view offset overflow"))?,
            byte_len,
        });
        if relative_offset
            .checked_add(byte_len)
            .ok_or_else(|| Error::backend("Laguna Metal model-view end overflow"))?
            >= mapped_bytes
        {
            break;
        }
        relative_offset = relative_offset
            .checked_add(step)
            .ok_or_else(|| Error::backend("Laguna Metal model-view step overflow"))?;
    }
    Ok(ranges)
}

fn host_page_size() -> Result<usize> {
    // SAFETY: sysconf has no pointer arguments and `_SC_PAGESIZE` is supported
    // on the macOS target required by this module.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return Err(Error::backend(
            "failed to determine the host VM page size for Laguna model views",
        ));
    }
    usize::try_from(page_size).map_err(|_| Error::backend("host VM page size does not fit usize"))
}

fn round_up(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded / alignment * alignment)
        .ok_or_else(|| Error::backend("Laguna Metal page-alignment overflow"))
}

fn responds_to(object: *mut Object, selector: objc::runtime::Sel) -> bool {
    if object.is_null() {
        return false;
    }
    // SAFETY: `object` is a live Objective-C object and
    // `respondsToSelector:` is provided by NSObjectProtocol.
    let result: BOOL = unsafe { msg_send![object, respondsToSelector: selector] };
    result != NO
}

fn objective_c_error(error: *mut Object) -> Option<String> {
    if error.is_null() {
        return None;
    }
    // SAFETY: NSError implements localizedDescription and NSString implements
    // UTF8String; both returned objects remain alive for this conversion.
    let description: *mut Object = unsafe { msg_send![error, localizedDescription] };
    if description.is_null() {
        return None;
    }
    // SAFETY: `description` is a live NSString.
    let bytes: *const c_char = unsafe { msg_send![description, UTF8String] };
    if bytes.is_null() {
        return None;
    }
    // SAFETY: NSString guarantees a NUL-terminated UTF-8 representation.
    Some(
        unsafe { CStr::from_ptr(bytes) }
            .to_string_lossy()
            .into_owned(),
    )
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use inferno_io::MappedFile;

    use crate::{Backend, MetalBackend};
    use common::F32Tensor;

    use super::super::device::Metal;
    use super::{plan_view_ranges, ViewRange};

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn view_plan_overlaps_by_more_than_largest_tensor() {
        let page = 16;
        let ranges = plan_view_ranges(1_024, 20, 256, 80, page).unwrap();

        assert_eq!(
            ranges,
            vec![
                ViewRange {
                    offset: 16,
                    byte_len: 256,
                },
                ViewRange {
                    offset: 176,
                    byte_len: 256,
                },
                ViewRange {
                    offset: 336,
                    byte_len: 256,
                },
                ViewRange {
                    offset: 496,
                    byte_len: 256,
                },
                ViewRange {
                    offset: 656,
                    byte_len: 256,
                },
                ViewRange {
                    offset: 816,
                    byte_len: 208,
                },
            ]
        );
        for pair in ranges.windows(2) {
            let overlap = pair[0].offset + pair[0].byte_len - pair[1].offset;
            assert!(overlap > 80);
        }
    }

    #[test]
    fn view_plan_rejects_tensor_larger_than_usable_buffer() {
        let error = plan_view_ranges(1_024, 32, 128, 112, 16).unwrap_err();

        assert!(error.to_string().contains("cannot contain Laguna tensor"));
    }

    #[test]
    fn native_views_keep_small_mapping_resident() {
        let Ok(metal) = Metal::new() else {
            return;
        };
        let page_size = super::host_page_size().unwrap();
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "inferno-laguna-metal-views-{}-{id}",
            std::process::id()
        ));
        fs::write(&path, vec![0x5a_u8; page_size * 8]).unwrap();
        let mapped = MappedFile::open(&path).unwrap();

        let report = metal
            .prepare_laguna_gguf_views(mapped.shared_bytes(), page_size / 2, page_size)
            .unwrap();

        assert_eq!(report.view_count, 1);
        assert!(report.warmup_samples > 0);
        drop(metal);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn native_q8_matvec_reads_weight_at_view_offset() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let page_size = super::host_page_size().unwrap();
        let tensor_offset = page_size + 32;
        let mut file_bytes = vec![0_u8; page_size * 4];
        file_bytes[tensor_offset..tensor_offset + 2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        file_bytes[tensor_offset + 2..tensor_offset + 34].fill(1);
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "inferno-laguna-offset-binding-{}-{id}",
            std::process::id()
        ));
        fs::write(&path, file_bytes).unwrap();
        let mapped = MappedFile::open(&path).unwrap();
        let weights = mapped.slice(tensor_offset as u64, 34).unwrap();
        backend
            .prepare_laguna_gguf_views(mapped.shared_bytes(), tensor_offset, weights.len())
            .unwrap()
            .unwrap();
        let input = F32Tensor::new(vec![1.0; 32], [1, 32]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();

        let output = backend
            .q8_0_matvec_device(weights, &input, 1, 32, 1)
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();

        assert_eq!(output.dims(), &[1, 1]);
        assert!((output.values()[0] - 32.0).abs() < 1e-5);
        drop(backend);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn native_laguna_q8_matvec_adds_both_residuals() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let page_size = super::host_page_size().unwrap();
        let tensor_offset = page_size + 32;
        let tensor_bytes = 68;
        let mut file_bytes = vec![0_u8; page_size * 4];
        file_bytes[tensor_offset..tensor_offset + 2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        file_bytes[tensor_offset + 2..tensor_offset + 34].fill(1);
        file_bytes[tensor_offset + 34..tensor_offset + 36]
            .copy_from_slice(&0x3800_u16.to_le_bytes());
        file_bytes[tensor_offset + 36..tensor_offset + tensor_bytes].fill((-2_i8) as u8);
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "inferno-laguna-add2-binding-{}-{id}",
            std::process::id()
        ));
        fs::write(&path, file_bytes).unwrap();
        let mapped = MappedFile::open(&path).unwrap();
        let weights = mapped.slice(tensor_offset as u64, tensor_bytes).unwrap();
        backend
            .prepare_laguna_gguf_views(mapped.shared_bytes(), tensor_offset, weights.len())
            .unwrap()
            .unwrap();
        let input = F32Tensor::new(vec![1.0; 32], [1, 32]).unwrap();
        let residual_a = F32Tensor::new(vec![0.25, -0.5], [2]).unwrap();
        let residual_b = F32Tensor::new(vec![1.0, 2.0], [1, 2]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();
        let residual_a = backend
            .device_upload_f32_tensor(&residual_a)
            .unwrap()
            .unwrap();
        let residual_b = backend
            .device_upload_f32_tensor(&residual_b)
            .unwrap()
            .unwrap();

        let output = backend
            .laguna_q8_0_matvec_add2_device(weights, &input, &residual_a, &residual_b, 1, 32, 2)
            .unwrap()
            .unwrap();
        let output = backend.device_download_f32_tensor(&output).unwrap();

        assert_eq!(output.dims(), &[1, 2]);
        assert!((output.values()[0] - 33.25).abs() < 1e-5);
        assert!((output.values()[1] + 30.5).abs() < 1e-5);
        drop(backend);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn native_laguna_q8_attention_projects_four_outputs_from_one_input() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let page_size = super::host_page_size().unwrap();
        let tensor_offset = page_size + 32;
        let mut file_bytes = vec![0_u8; page_size * 4];
        let mut cursor = tensor_offset;
        let mut write_matrix = |quants: &[i8]| {
            let start = cursor;
            for quant in quants {
                file_bytes[cursor..cursor + 2].copy_from_slice(&0x3c00_u16.to_le_bytes());
                file_bytes[cursor + 2..cursor + 34].fill(*quant as u8);
                cursor += 34;
            }
            start..cursor
        };
        let query_range = write_matrix(&[1, 2, -1, 3]);
        let key_range = write_matrix(&[4, -2]);
        let value_range = write_matrix(&[1, -3]);
        let gate_range = write_matrix(&[2, 0]);
        let tensor_bytes = cursor - tensor_offset;
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "inferno-laguna-attention-projections-binding-{}-{id}",
            std::process::id()
        ));
        fs::write(&path, file_bytes).unwrap();
        let mapped = MappedFile::open(&path).unwrap();
        let query_weights = mapped
            .slice(query_range.start as u64, query_range.len())
            .unwrap();
        let key_weights = mapped
            .slice(key_range.start as u64, key_range.len())
            .unwrap();
        let value_weights = mapped
            .slice(value_range.start as u64, value_range.len())
            .unwrap();
        let gate_weights = mapped
            .slice(gate_range.start as u64, gate_range.len())
            .unwrap();
        backend
            .prepare_laguna_gguf_views(mapped.shared_bytes(), tensor_offset, tensor_bytes)
            .unwrap()
            .unwrap();
        let input = F32Tensor::new(vec![1.0; 32], [1, 32]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();

        let [query, key, value, gate] = backend
            .laguna_q8_0_attention_projections_device(
                query_weights,
                key_weights,
                value_weights,
                gate_weights,
                &input,
                1,
                32,
                4,
                2,
                2,
                2,
            )
            .unwrap()
            .unwrap();
        let query = backend.device_download_f32_tensor(&query).unwrap();
        let key = backend.device_download_f32_tensor(&key).unwrap();
        let value = backend.device_download_f32_tensor(&value).unwrap();
        let gate = backend.device_download_f32_tensor(&gate).unwrap();

        assert_eq!(query.dims(), &[1, 4]);
        assert_eq!(query.values(), &[32.0, 64.0, -32.0, 96.0]);
        assert_eq!(key.dims(), &[1, 2]);
        assert_eq!(key.values(), &[128.0, -64.0]);
        assert_eq!(value.dims(), &[1, 2]);
        assert_eq!(value.values(), &[32.0, -96.0]);
        assert_eq!(gate.dims(), &[1, 2]);
        assert_eq!(gate.values(), &[64.0, 0.0]);
        drop(backend);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn native_laguna_q8_output_head_argmax_preserves_lowest_id_ties() {
        let Ok(backend) = MetalBackend::new() else {
            return;
        };
        let page_size = super::host_page_size().unwrap();
        let tensor_offset = page_size + 32;
        let rows = [
            (0x3c00_u16, 1_i8),
            (0x3c00, 2),
            (0x3800, 4),
            (0x3c00, -1),
            (0x3c00, 0),
            (0x3800, 1),
            (0x3800, -2),
            (0x3c00, 1),
        ];
        let tensor_bytes = rows.len() * 34;
        let mut file_bytes = vec![0_u8; page_size * 4];
        for (row, (scale, quant)) in rows.iter().copied().enumerate() {
            let offset = tensor_offset + row * 34;
            file_bytes[offset..offset + 2].copy_from_slice(&scale.to_le_bytes());
            file_bytes[offset + 2..offset + 34].fill(quant as u8);
        }
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "inferno-laguna-output-argmax-binding-{}-{id}",
            std::process::id()
        ));
        fs::write(&path, file_bytes).unwrap();
        let mapped = MappedFile::open(&path).unwrap();
        let weights = mapped.slice(tensor_offset as u64, tensor_bytes).unwrap();
        backend
            .prepare_laguna_gguf_views(mapped.shared_bytes(), tensor_offset, weights.len())
            .unwrap()
            .unwrap();
        let input = F32Tensor::new(vec![1.0; 32], [1, 32]).unwrap();
        let input = backend.device_upload_f32_tensor(&input).unwrap().unwrap();

        let (token_id, token_score) = backend
            .laguna_q8_0_matvec_argmax_device(weights, &input, 32, rows.len())
            .unwrap()
            .unwrap();

        assert_eq!(token_id, 1);
        assert!((token_score - 64.0).abs() < 1e-5);
        drop(backend);
        fs::remove_file(path).unwrap();
    }
}
