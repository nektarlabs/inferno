#![deny(unsafe_op_in_unsafe_fn)]

//! Backend selection and operation boundaries for native model inference.

mod backend;
mod device_value;
#[cfg(all(target_os = "macos", feature = "metal"))]
mod metal;

pub use backend::{
    Backend, BackendCapabilities, BackendMemoryReport, DeviceBf16Matrix, DevicePagedKvView,
    DeviceRopeTable, DeviceRoutedExperts, DeviceRouterTopK, DeviceSelectedKvView, DeviceW4Weight,
    ExpertCacheMetrics, GgufExpertQuant, GgufKQuant, LagunaAttentionProjections, LagunaF16KvCache,
    LagunaFp8KvCache, LagunaKvRetention, LagunaModelViewReport, MetalBackend, Q2ExpertSource,
    RouterTopK, W4ExpertGroup, W4WeightSource,
};
pub use device_value::DeviceValue;
