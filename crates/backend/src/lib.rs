#![deny(unsafe_op_in_unsafe_fn)]

//! Backend selection and operation boundaries for GLM inference.

mod backend;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal;

pub use backend::{Backend, BackendCapabilities, BackendMemoryReport, MetalBackend};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub use metal::{
    Metal, MetalAddReport, MetalAttentionCausalSoftmaxReport, MetalAttentionScoresReport,
    MetalAttentionValuesReport, MetalCombineRopeTailReport, MetalHeadsToAttentionLayoutReport,
    MetalLinearReport, MetalMatmulReport, MetalMergeAttentionHeadsReport, MetalMoeCombineReport,
    MetalMoeGatherReport, MetalQ2MatvecAddReport, MetalQ2MatvecArgmaxReport, MetalQ2MatvecReport,
    MetalRmsNormReport, MetalRopeReport, MetalSelectLastTokenReport, MetalSplitKvMqaReport,
    MetalSplitRopeTailReport, MetalSwiGluReport,
};
