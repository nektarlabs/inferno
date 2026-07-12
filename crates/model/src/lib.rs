#![deny(unsafe_code)]

//! GLM-5.2 model components for quantized inference.

/// Unwraps a `Result<Option<T>>` from a batched `*_device` backend op inside a
/// function that itself returns `Result<Option<_>>`. `None` means the backend
/// (or this weight's quantization) has no device-resident path, so the caller
/// bails out with `Ok(None)` and the eager per-op path runs instead.
macro_rules! try_device {
    ($expr:expr) => {
        match $expr? {
            Some(value) => value,
            None => return Ok(None),
        }
    };
}
pub(crate) use try_device;

mod artifact_source;
mod attention;
mod attention_math;
mod dense_block;
mod dense_ffn;
mod embedding;
mod index;
mod indexer;
mod kv_types;
mod layer_kind;
mod layer_stack;
mod linear;
mod model;
mod moe_ffn;
mod moe_router;
mod mtp;
mod output_head;
mod profile;
mod rms_norm;
mod sparse_block;
mod weights;

pub use artifact_source::{
    antirez_q2_artifact, Artifact, ArtifactFormat, ANTIREZ_Q2_GGUF_FILE, ANTIREZ_Q2_GGUF_REPO_ID,
    ANTIREZ_Q2_GGUF_REPO_URL,
};
pub(crate) use attention::{Attention, AttentionOutput};
pub use attention::{AttentionForwardReport, AttentionLoadReport};
pub(crate) use dense_block::DenseBlock;
pub use dense_block::{DenseBlockForwardReport, DenseBlockLoadReport};
pub(crate) use dense_ffn::{DenseFfn, DenseFfnOutput};
pub use dense_ffn::{DenseFfnForwardReport, DenseFfnLoadReport};
pub use embedding::EmbeddingLookupReport;
pub(crate) use embedding::{EmbeddingLookupOutput, EmbeddingTable};
pub use index::{
    AttentionIndex, DenseFfnIndex, FfnIndex, Index, IndexSummary, IndexerIndex, LayerIndex,
    MemoryAdviceReport, MtpIndex, PackedExpertsIndex, RootIndex, SharedExpertIndex, TensorRef,
};
pub(crate) use indexer::DsaIndexer;
pub use indexer::DsaIndexerLoadReport;
pub use kv_types::{LayerDeviceKvCacheTensors, LayerKvCacheReport, LayerKvCacheTensors};
pub use layer_kind::LayerKind;
pub(crate) use layer_stack::LayerStack;
pub use layer_stack::{
    LayerRuntimeKind, LayerRuntimeReport, LayerStackForwardReport, LayerStackLoadReport,
};
pub(crate) use linear::QuantizedLinear;
pub use linear::{LinearForwardReport, LinearGreedyReport};
pub use model::{
    Model, ModelDeviceTokenOutput, ModelDeviceTokenSequenceOutput, ModelGreedyOutput,
    ModelGreedyReport, ModelHiddenOutput, ModelHiddenReport, ModelLoadReport, ModelLogitsOutput,
    ModelLogitsReport, ModelTokenOutput, ModelTokenSequenceOutput, ModelTokenWithHiddenOutput,
    DEFAULT_GGUF_OUTPUT_CHUNK_ROWS,
};
pub(crate) use moe_ffn::{MoeFfn, MoeFfnOutput};
pub use moe_ffn::{MoeFfnForwardReport, MoeFfnLoadReport};
pub(crate) use moe_router::MoeRouter;
pub use moe_router::MoeRouterLoadReport;
pub(crate) use mtp::MtpHead;
pub use mtp::{MtpDeviceDraftOutput, MtpDraftOutput, MtpLoadReport};
pub use output_head::{GreedyReport, LogitsReport, OutputHeadLoadReport};
pub(crate) use output_head::{LogitsOutput, OutputHead};
pub use profile::{
    disable_token_cost_profile, enable_token_cost_profile, take_token_cost_profile,
    TokenModelProfile,
};
pub use profile::{enable_layer_profile, set_layer_profile_context};
pub(crate) use rms_norm::RmsNorm;
pub use rms_norm::RmsNormLoadReport;
pub(crate) use sparse_block::SparseBlock;
pub use sparse_block::{SparseBlockForwardReport, SparseBlockLoadReport};
pub use weights::TensorLoadReport;
pub(crate) use weights::WeightLoader;
