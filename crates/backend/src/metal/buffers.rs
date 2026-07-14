use std::{collections::HashMap, ffi::c_void, mem::size_of, ptr, slice, sync::Mutex};

use ::metal::{Buffer, Device, MTLResourceOptions};
use common::{Error, Result};

const BUFFER_OPTIONS: MTLResourceOptions =
    MTLResourceOptions::StorageModeShared.union(MTLResourceOptions::CPUCacheModeDefaultCache);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BufferKey {
    address: usize,
    byte_len: usize,
}

#[derive(Debug, Default)]
pub(crate) struct ImmutableF32BufferCache {
    buffers: Mutex<HashMap<BufferKey, Buffer>>,
}

impl ImmutableF32BufferCache {
    pub(crate) fn get(&self, device: &Device, values: &[f32]) -> Result<Buffer> {
        let key = BufferKey {
            address: values.as_ptr() as usize,
            byte_len: byte_len::<f32>(values.len())?,
        };
        let mut buffers = self
            .buffers
            .lock()
            .map_err(|_| Error::backend("Metal immutable F32 buffer cache lock poisoned"))?;
        if let Some(buffer) = buffers.get(&key) {
            return Ok(buffer.clone());
        }
        let buffer = f32_buffer_no_copy(device, values)?;
        buffers.insert(key, buffer.clone());
        Ok(buffer)
    }
}

pub(crate) fn f32_buffer(device: &Device, values: &[f32]) -> Result<Buffer> {
    buffer_with_data(device, values)
}

/// References immutable model-owned F32 storage without copying it. The slice
/// must outlive every command buffer and cached `Buffer` that uses it.
pub(crate) fn f32_buffer_no_copy(device: &Device, values: &[f32]) -> Result<Buffer> {
    let bytes = byte_len::<f32>(values.len())?;
    if bytes == 0 {
        return Err(Error::backend("cannot create an empty Metal F32 buffer"));
    }
    Ok(device.new_buffer_with_bytes_no_copy(
        values.as_ptr().cast::<c_void>(),
        bytes as u64,
        BUFFER_OPTIONS,
        None,
    ))
}

pub(crate) fn u32_buffer(device: &Device, values: &[u32]) -> Result<Buffer> {
    buffer_with_data(device, values)
}

pub(crate) fn u64_buffer(device: &Device, values: &[u64]) -> Result<Buffer> {
    buffer_with_data(device, values)
}

pub(crate) fn u8_buffer(device: &Device, values: &[u8]) -> Result<Buffer> {
    buffer_with_data(device, values)
}

pub(crate) fn empty_u8_buffer(device: &Device, len: usize) -> Result<Buffer> {
    let bytes = byte_len::<u8>(len)?;
    if bytes == 0 {
        return Err(Error::backend("cannot create an empty Metal byte buffer"));
    }
    Ok(device.new_buffer(bytes as u64, BUFFER_OPTIONS))
}

pub(crate) fn empty_u32_buffer(device: &Device, len: usize) -> Result<Buffer> {
    let bytes = byte_len::<u32>(len)?;
    Ok(device.new_buffer(bytes as u64, BUFFER_OPTIONS))
}

pub(crate) fn empty_f32_buffer(device: &Device, len: usize) -> Result<Buffer> {
    let bytes = byte_len::<f32>(len)?;
    Ok(device.new_buffer(bytes as u64, BUFFER_OPTIONS))
}

pub(crate) fn empty_f16_buffer(device: &Device, len: usize) -> Result<Buffer> {
    let bytes = len
        .checked_mul(size_of::<u16>())
        .ok_or_else(|| Error::backend("Metal f16 buffer byte length overflow"))?;
    Ok(device.new_buffer(bytes as u64, BUFFER_OPTIONS))
}

/// Creates a Metal view over immutable model-owned bytes without copying the
/// GGUF mapping. Routed experts use a separate bounded cache; this helper is
/// for always-used quantized tensors whose mmap storage outlives the backend.
pub(crate) fn u8_buffer_no_copy(device: &Device, values: &[u8]) -> Result<Buffer> {
    let bytes = byte_len::<u8>(values.len())?;
    if bytes == 0 {
        return Err(Error::backend("cannot create an empty Metal buffer"));
    }

    Ok(device.new_buffer_with_bytes_no_copy(
        values.as_ptr().cast::<c_void>(),
        bytes as u64,
        BUFFER_OPTIONS,
        None,
    ))
}

/// Checks that `buffer` can hold at least `len` f32 elements. Used to validate
/// device-resident op inputs, whose lengths are tracked by the caller rather
/// than by a host slice.
pub(crate) fn require_f32_capacity(buffer: &Buffer, len: usize, label: &str) -> Result<()> {
    let bytes = byte_len::<f32>(len)?;
    require_byte_capacity(buffer, bytes, label)
}

pub(crate) fn require_f16_capacity(buffer: &Buffer, len: usize, label: &str) -> Result<()> {
    let bytes = len
        .checked_mul(size_of::<u16>())
        .ok_or_else(|| Error::backend("Metal f16 buffer byte length overflow"))?;
    require_byte_capacity(buffer, bytes, label)
}

pub(crate) fn require_byte_capacity(buffer: &Buffer, bytes: usize, label: &str) -> Result<()> {
    if buffer.length() < bytes as u64 {
        return Err(Error::backend(format!(
            "Metal {label} buffer is too small: expected at least {bytes} bytes, got {}",
            buffer.length()
        )));
    }
    Ok(())
}

pub(crate) fn read_f32_buffer(buffer: &Buffer, len: usize) -> Result<Vec<f32>> {
    let bytes = byte_len::<f32>(len)?;
    if buffer.length() < bytes as u64 {
        return Err(Error::backend(format!(
            "Metal output buffer is too small: expected at least {bytes} bytes, got {}",
            buffer.length()
        )));
    }

    let ptr = buffer.contents().cast::<f32>();
    if ptr.is_null() {
        return Err(Error::backend("Metal buffer contents pointer is null"));
    }

    // SAFETY: the buffer was allocated by this module with shared storage, is at least
    // len * sizeof(f32) bytes long, and the command buffer has completed before reads.
    let values = unsafe { slice::from_raw_parts(ptr, len) };
    Ok(values.to_vec())
}

pub(crate) fn read_f16_buffer_as_f32(buffer: &Buffer, len: usize) -> Result<Vec<f32>> {
    let bytes = len
        .checked_mul(size_of::<u16>())
        .ok_or_else(|| Error::backend("Metal f16 read byte length overflow"))?;
    if buffer.length() < bytes as u64 {
        return Err(Error::backend(format!(
            "Metal f16 output buffer is too small: expected at least {bytes} bytes, got {}",
            buffer.length()
        )));
    }

    let ptr = buffer.contents().cast::<u16>();
    if ptr.is_null() {
        return Err(Error::backend("Metal buffer contents pointer is null"));
    }

    // SAFETY: the buffer uses shared storage, is at least len * sizeof(u16)
    // bytes long, and the command buffer has completed before reads.
    let values = unsafe { slice::from_raw_parts(ptr, len) };
    Ok(values.iter().copied().map(f16_bits_to_f32).collect())
}

pub(crate) fn write_f32_buffer(buffer: &Buffer, values: &[f32]) -> Result<()> {
    write_buffer(buffer, 0, values)
}

pub(crate) fn write_f32_buffer_at(
    buffer: &Buffer,
    element_offset: usize,
    values: &[f32],
) -> Result<()> {
    write_buffer(buffer, element_offset, values)
}

pub(crate) fn write_u32_buffer(buffer: &Buffer, values: &[u32]) -> Result<()> {
    write_buffer(buffer, 0, values)
}

fn write_buffer<T>(buffer: &Buffer, element_offset: usize, values: &[T]) -> Result<()> {
    let offset_bytes = byte_len::<T>(element_offset)?;
    let value_bytes = byte_len::<T>(values.len())?;
    let required_bytes = offset_bytes
        .checked_add(value_bytes)
        .ok_or_else(|| Error::backend("Metal buffer write byte range overflow"))?;
    if buffer.length() < required_bytes as u64 {
        return Err(Error::backend(format!(
            "Metal input buffer is too small: expected at least {required_bytes} bytes, got {}",
            buffer.length()
        )));
    }

    let ptr = buffer.contents().cast::<T>();
    if ptr.is_null() {
        return Err(Error::backend("Metal buffer contents pointer is null"));
    }

    // SAFETY: the destination buffer uses shared storage, is at least
    // element_offset + values.len() elements long, and the source slice is valid.
    unsafe {
        ptr::copy_nonoverlapping(values.as_ptr(), ptr.add(element_offset), values.len());
    }
    Ok(())
}

pub(crate) fn read_u32_buffer(buffer: &Buffer, len: usize) -> Result<Vec<u32>> {
    let bytes = byte_len::<u32>(len)?;
    if buffer.length() < bytes as u64 {
        return Err(Error::backend(format!(
            "Metal output buffer is too small: expected at least {bytes} bytes, got {}",
            buffer.length()
        )));
    }

    let ptr = buffer.contents().cast::<u32>();
    if ptr.is_null() {
        return Err(Error::backend("Metal buffer contents pointer is null"));
    }

    // SAFETY: the buffer was allocated by this module with shared storage, is at least
    // len * sizeof(u32) bytes long, and the command buffer has completed before reads.
    let values = unsafe { slice::from_raw_parts(ptr, len) };
    Ok(values.to_vec())
}

fn buffer_with_data<T>(device: &Device, values: &[T]) -> Result<Buffer> {
    let bytes = byte_len::<T>(values.len())?;
    if bytes == 0 {
        return Err(Error::backend("cannot create an empty Metal buffer"));
    }

    Ok(device.new_buffer_with_data(values.as_ptr().cast(), bytes as u64, BUFFER_OPTIONS))
}

fn byte_len<T>(len: usize) -> Result<usize> {
    len.checked_mul(size_of::<T>())
        .ok_or_else(|| Error::backend("Metal buffer byte length overflow"))
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = (bits & 0x03ff) as u32;

    let f32_bits = match exponent {
        0 => {
            if mantissa == 0 {
                sign
            } else {
                let mut mantissa = mantissa;
                let mut exponent = -14_i32;
                while (mantissa & 0x0400) == 0 {
                    mantissa <<= 1;
                    exponent -= 1;
                }
                mantissa &= 0x03ff;
                let exponent_bits = ((exponent + 127) as u32) << 23;
                sign | exponent_bits | (mantissa << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => {
            let exponent_bits = ((exponent as u32) + (127 - 15)) << 23;
            sign | exponent_bits | (mantissa << 13)
        }
    };
    f32::from_bits(f32_bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_len_rejects_overflow() {
        let err = byte_len::<f32>(usize::MAX).expect_err("overflow should fail");
        assert!(err.to_string().contains("overflow"));
    }

    #[test]
    fn f16_bits_to_f32_decodes_basic_values() {
        assert_eq!(f16_bits_to_f32(0x0000), 0.0);
        assert_eq!(f16_bits_to_f32(0x3c00), 1.0);
        assert_eq!(f16_bits_to_f32(0xc000), -2.0);
    }
}
