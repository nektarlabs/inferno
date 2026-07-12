#![deny(unsafe_code)]

//! Paged KV cache primitives for the GLM runtime.

mod cold_kv;
mod dsa_index;
mod layered_paged;
mod paged;

pub use cold_kv::{
    ColdKvBlockKey, ColdKvBlockMeta, ColdKvBlockStore, ColdKvCodec, ColdKvSelectedQ8LayerRows,
    ColdKvSelectedQ8TensorRows, ColdKvStoreSpec, ColdKvTensorKind, ColdKvWriteReport,
    LayeredColdKvBlockStore, LayeredColdKvWriteReport,
};
pub use dsa_index::{
    DsaIndexBlockMeta, DsaIndexLayerAppend, DsaIndexStoreSpec, LayeredDsaIndexBlockStore,
};
pub use layered_paged::{
    LayerKvCacheAppend, LayerPageStats, LayerPagedCacheAppendReport, LayerPagedKvView,
    LayeredPagedCacheAppendReport, LayeredPagedKvCache, LayeredPagedKvCacheSpec,
};
pub use paged::{PageStats, PagedCacheAppendReport};
