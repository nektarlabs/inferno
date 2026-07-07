use common::{DType, Error, Result, Shape};

/// An opaque handle to a tensor that lives in GPU memory.
///
/// Values produced by the batched `*_device` backend ops stay resident on the
/// device; the kernels that computed them may not have executed yet. Passing a
/// `DeviceValue` into another `*_device` op chains work on the GPU without a
/// host round-trip. The only way to observe the numbers from the CPU is
/// `Backend::device_download_f32`, which synchronizes the pending batch first.
///
/// Cloning is cheap: it retains the underlying GPU buffer, it does not copy
/// the data.
#[derive(Debug, Clone)]
pub struct DeviceValue {
    shape: Shape,
    dtype: DType,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) buffer: ::metal::Buffer,
}

impl DeviceValue {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn new(dims: Vec<usize>, buffer: ::metal::Buffer) -> Self {
        Self::new_with_dtype(dims, DType::F32, buffer)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn new_with_dtype(dims: Vec<usize>, dtype: DType, buffer: ::metal::Buffer) -> Self {
        Self {
            shape: Shape::new(dims),
            dtype,
            buffer,
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Total number of logical tensor elements this value holds.
    pub fn element_count(&self) -> Result<usize> {
        self.dims().iter().try_fold(1_usize, |count, dim| {
            count
                .checked_mul(*dim)
                .ok_or_else(|| Error::backend("device value element count overflow"))
        })
    }

    pub fn byte_count(&self) -> Result<usize> {
        self.element_count()?
            .checked_mul(self.dtype.byte_size())
            .ok_or_else(|| Error::backend("device value byte count overflow"))
    }

    /// Reinterprets the value with a new shape holding the same element count.
    /// This is metadata-only: no GPU work and no synchronization.
    pub fn reshape(&self, dims: Vec<usize>) -> Result<Self> {
        let new_count = dims.iter().try_fold(1_usize, |count, dim| {
            count
                .checked_mul(*dim)
                .ok_or_else(|| Error::backend("device value reshape element count overflow"))
        })?;
        if new_count != self.element_count()? {
            return Err(Error::backend(format!(
                "device value reshape element count mismatch: {:?} -> {dims:?}",
                self.dims()
            )));
        }
        Ok(Self {
            shape: Shape::new(dims),
            dtype: self.dtype,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            buffer: self.buffer.clone(),
        })
    }
}
