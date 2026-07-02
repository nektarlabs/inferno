#![deny(unsafe_code)]

//! Paged KV cache primitives for the GLM runtime.

mod disk_paged;
mod layered_paged;
mod paged;

pub use disk_paged::{LayerDiskPageStats, LayerDiskPagedKvView, LayeredDiskPagedKvCache};
pub use layered_paged::{
    LayerKvCacheAppend, LayerPageStats, LayerPagedCacheAppendReport, LayerPagedKvView,
    LayeredPagedCacheAppendReport, LayeredPagedKvCache, LayeredPagedKvCacheSpec,
};
pub use paged::{
    LogicalTokenLocation, PageStats, PagedCacheAppendReport, PagedKvCache, PagedKvCacheSpec,
};
