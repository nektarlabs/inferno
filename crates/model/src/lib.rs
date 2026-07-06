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
mod kv_types;
mod layer_kind;
mod layer_stack;
mod linear;
mod model;
mod moe_ffn;
mod moe_router;
mod output_head;
mod profile;
mod rms_norm;
mod sparse_block;
mod weights;

pub use artifact_source::{
    antirez_q2_artifact, Artifact, ArtifactFormat, ANTIREZ_Q2_GGUF_FILE, ANTIREZ_Q2_GGUF_REPO_ID,
    ANTIREZ_Q2_GGUF_REPO_URL,
};
pub use attention::{
    Attention, AttentionF32Tensors, AttentionForwardReport, AttentionLoadReport, AttentionOutput,
};
pub use dense_block::{
    DenseBlock, DenseBlockF32Tensors, DenseBlockForwardReport, DenseBlockLoadReport,
    DenseBlockOutput, DenseBlockTensors,
};
pub use dense_ffn::{DenseFfn, DenseFfnForwardReport, DenseFfnLoadReport, DenseFfnOutput};
pub use embedding::{
    EmbeddingLookupF32Output, EmbeddingLookupF32Report, EmbeddingLookupOutput,
    EmbeddingLookupReport, EmbeddingTable,
};
pub use index::{
    AttentionIndex, DenseFfnIndex, FfnIndex, Index, IndexSummary, IndexerIndex, LayerIndex,
    PackedExpertsIndex, RootIndex, SharedExpertIndex, TensorRef,
};
pub use kv_types::{LayerKvCacheReport, LayerKvCacheTensors};
pub use layer_kind::LayerKind;
pub use layer_stack::{
    LayerRuntimeKind, LayerRuntimeReport, LayerStack, LayerStackForwardF32Tensors,
    LayerStackForwardOutput, LayerStackForwardReport, LayerStackForwardTensors,
    LayerStackLoadReport,
};
pub use linear::{
    LinearForwardReport, LinearGreedyOutput, LinearGreedyReport, LinearOutput, LinearTokenOutput,
    QuantizedLinear,
};
pub use model::{
    Model, ModelGreedyOutput, ModelGreedyReport, ModelHiddenOutput, ModelHiddenReport,
    ModelLoadReport, ModelLogitsOutput, ModelLogitsReport, ModelTokenOutput,
    DEFAULT_GGUF_OUTPUT_CHUNK_ROWS,
};
pub use moe_ffn::{MoeFfn, MoeFfnForwardReport, MoeFfnLoadReport, MoeFfnOutput};
pub use moe_router::{MoeRouter, MoeRouterLoadReport, MoeRoutingOutput, MoeRoutingReport};
pub use output_head::{
    GreedyOutput, GreedyReport, LogitsOutput, LogitsReport, OutputHead, OutputHeadLoadReport,
    TokenOutput,
};
pub use profile::{enable_layer_profile, set_layer_profile_context};
pub use rms_norm::{
    RmsNorm, RmsNormF32Output, RmsNormForwardReport, RmsNormLoadReport, RmsNormOutput,
};
pub use sparse_block::{
    SparseBlock, SparseBlockF32Tensors, SparseBlockForwardReport, SparseBlockLoadReport,
    SparseBlockOutput, SparseBlockTensors,
};
pub use weights::{TensorLoadF32Output, TensorLoadOutput, TensorLoadReport, WeightLoader};
