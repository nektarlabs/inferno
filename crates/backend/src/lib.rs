#![deny(unsafe_op_in_unsafe_fn)]

//! Backend selection and operation boundaries for GLM inference.

mod backend;
mod device_value;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal;

pub use backend::{
    Backend, BackendCapabilities, BackendMemoryReport, DevicePagedKvView, DeviceRoutedExperts,
    DeviceRouterTopK, DeviceSelectedKvView, ExpertCacheMetrics, MetalBackend, Q2ExpertSource,
    RouterTopK,
};
pub use device_value::DeviceValue;
