use std::{mem::size_of, ptr, slice};

use ::metal::{Buffer, Device, MTLResourceOptions};
use common::{Error, Result};

const BUFFER_OPTIONS: MTLResourceOptions =
    MTLResourceOptions::StorageModeShared.union(MTLResourceOptions::CPUCacheModeDefaultCache);

pub(crate) fn f32_buffer(device: &Device, values: &[f32]) -> Result<Buffer> {
    buffer_with_data(device, values)
}

pub(crate) fn u8_buffer(device: &Device, values: &[u8]) -> Result<Buffer> {
    buffer_with_data(device, values)
}

pub(crate) fn u32_scalar_buffer(device: &Device, value: u32) -> Result<Buffer> {
    buffer_with_data(device, &[value])
}

pub(crate) fn u32_buffer(device: &Device, values: &[u32]) -> Result<Buffer> {
    buffer_with_data(device, values)
}

pub(crate) fn empty_u32_buffer(device: &Device, len: usize) -> Result<Buffer> {
    let bytes = byte_len::<u32>(len)?;
    Ok(device.new_buffer(bytes as u64, BUFFER_OPTIONS))
}

pub(crate) fn f32_scalar_buffer(device: &Device, value: f32) -> Result<Buffer> {
    buffer_with_data(device, &[value])
}

pub(crate) fn empty_f32_buffer(device: &Device, len: usize) -> Result<Buffer> {
    let bytes = byte_len::<f32>(len)?;
    Ok(device.new_buffer(bytes as u64, BUFFER_OPTIONS))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_len_rejects_overflow() {
        let err = byte_len::<f32>(usize::MAX).expect_err("overflow should fail");
        assert!(err.to_string().contains("overflow"));
    }
}
