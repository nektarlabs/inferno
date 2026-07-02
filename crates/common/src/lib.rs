#![deny(unsafe_code)]

//! Shared production primitives for the GLM inference workspace.

pub mod device;
pub mod error;
pub mod f32_tensor;
pub mod paged_kv;
pub mod shape;

pub use device::{BackendKind, DeviceKind, DeviceReport};
pub use error::{Error, Result};
pub use f32_tensor::{DType, Device, F32Tensor, Tensor};
pub use paged_kv::{PagedKvPageView, PagedKvView};
pub use shape::{validate_exact_shape, Shape};
