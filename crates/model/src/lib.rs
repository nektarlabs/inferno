#![deny(unsafe_code)]

//! Model-specific components for quantized inference.

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

/// Maximum number of causal token rows handled by one decode or MTP
/// verification pass.
pub const MAX_DEVICE_SEQUENCE_TOKENS: usize = 8;

/// Maximum number of prompt rows handled by one native Metal prefill pass.
/// Larger prompt batches amortize routed-expert SSD loads across more tokens;
/// each expert is still dispatched in eight-assignment Metal kernel groups.
pub const MAX_DEVICE_PREFILL_TOKENS: usize = 512;

mod artifact_source;
mod attention;
mod attention_math;
mod dense_block;
mod dense_ffn;
mod embedding;
mod expert_pack;
mod index;
mod indexer;
mod kv_types;
mod laguna;
mod layer_kind;
mod layer_stack;
mod linear;
mod model;
mod moe_ffn;
mod moe_router;
mod mtp;
mod output_head;
mod policy;
mod profile;
mod qwen;
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
pub use expert_pack::{create_expert_pack, expected_expert_pack_header, ExpertPackReport};
pub use index::{
    AttentionIndex, DenseFfnIndex, FfnIndex, Index, IndexSummary, IndexerIndex, LayerIndex,
    MemoryAdviceReport, MtpIndex, PackedExpertsIndex, RootIndex, SharedExpertIndex, TensorRef,
};
pub(crate) use indexer::DsaIndexer;
pub use indexer::DsaIndexerLoadReport;
pub use kv_types::{LayerDeviceKvCacheTensors, LayerKvCacheReport, LayerKvCacheTensors};
pub use laguna::{
    forward_attention as forward_laguna_attention, forward_dense_mlp_residual,
    forward_layer as forward_laguna_layer, forward_sparse_mlp_residual, LagunaArtifactKind,
    LagunaAttentionCache, LagunaAttentionWeights, LagunaDenseWeights, LagunaDeviceAttentionWeights,
    LagunaDeviceDenseWeights, LagunaDeviceExpertWeights, LagunaDeviceLayerMlpWeights,
    LagunaDeviceLayerWeights, LagunaDeviceMoeWeights, LagunaDeviceRootWeights,
    LagunaDeviceRopeTables, LagunaDeviceWeights, LagunaExpertCache, LagunaExpertCacheMetrics,
    LagunaExpertWeights, LagunaGgufAttention, LagunaGgufDense, LagunaGgufFlavor, LagunaGgufIndex,
    LagunaGgufLayer, LagunaGgufMlp, LagunaGgufModel, LagunaGgufMoe, LagunaGgufRoot,
    LagunaGgufSession, LagunaLayerMlpWeights, LagunaLayerWeights, LagunaModel, LagunaMoeWeights,
    LagunaRootWeights, LagunaSession, LagunaTokenOutput, LagunaWeightIndex, LagunaWeightSummary,
    LAGUNA_GGUF_FILE_BYTES, LAGUNA_GGUF_FILE_NAME, LAGUNA_GGUF_REPO_ID, LAGUNA_GGUF_REPO_URL,
    LAGUNA_GGUF_SHA256, LAGUNA_INT4_REPO_ID, LAGUNA_INT4_REPO_URL, LAGUNA_INT4_TOTAL_BYTES,
    LAGUNA_XS_GGUF_FILE_BYTES, LAGUNA_XS_GGUF_FILE_NAME, LAGUNA_XS_GGUF_REPO_ID,
    LAGUNA_XS_GGUF_REPO_URL, LAGUNA_XS_GGUF_SHA256,
};
pub use layer_kind::LayerKind;
pub(crate) use layer_stack::LayerStack;
pub use layer_stack::{
    LayerRuntimeKind, LayerRuntimeReport, LayerStackForwardReport, LayerStackLoadReport,
};
pub(crate) use linear::QuantizedLinear;
pub use linear::{LinearForwardReport, LinearGreedyReport};
pub use model::{
    Model, ModelDevicePrefillChunkOutput, ModelDevicePrefillToken, ModelDeviceTokenOutput,
    ModelDeviceTokenSequenceOutput, ModelGreedyOutput, ModelGreedyReport, ModelHiddenOutput,
    ModelHiddenReport, ModelLoadReport, ModelLogitsOutput, ModelLogitsReport, ModelTokenOutput,
    ModelTokenSequenceOutput, ModelTokenWithHiddenOutput, DEFAULT_GGUF_OUTPUT_CHUNK_ROWS,
};
pub(crate) use moe_ffn::{MoeFfn, MoeFfnOutput};
pub use moe_ffn::{MoeFfnForwardReport, MoeFfnLoadReport};
pub(crate) use moe_router::MoeRouter;
pub use moe_router::MoeRouterLoadReport;
pub(crate) use mtp::MtpHead;
pub use mtp::{MtpDeviceDraftOutput, MtpDraftOutput, MtpLoadReport};
pub use output_head::{GreedyReport, LogitsReport, OutputHeadLoadReport};
pub(crate) use output_head::{LogitsOutput, OutputHead};
pub use policy::{validate_routing_policy, ROUTED_EXPERTS_PER_TOKEN};
pub use profile::{
    disable_token_cost_profile, enable_token_cost_profile, take_token_cost_profile,
    TokenModelProfile,
};
pub use profile::{enable_layer_profile, set_layer_profile_context};
pub use qwen::forward_dflash_proposals_device;
pub use qwen::{
    draft_next_token_device_handle as draft_qwen_next_token_device_handle,
    forward_full_attention_residual_device as forward_qwen_full_attention_residual_device,
    forward_greedy_device as forward_qwen_greedy_device,
    forward_hidden_device as forward_qwen_hidden_device,
    forward_hidden_with_dflash_features_device as forward_qwen_hidden_with_dflash_features_device,
    forward_layer_device as forward_qwen_layer_device,
    forward_linear_attention_residual_device as forward_qwen_linear_attention_residual_device,
    forward_mlp_device as forward_qwen_mlp_device,
    forward_mlp_residual_device as forward_qwen_mlp_residual_device,
    forward_mtp_hidden_device as forward_qwen_mtp_hidden_device,
    forward_mtp_hidden_from_device_tokens as forward_qwen_mtp_hidden_from_device_tokens,
    greedy_all_tokens_device as greedy_all_qwen_tokens_device,
    greedy_next_tokens_device as greedy_qwen_next_tokens_device, DFlashAttentionWeights,
    DFlashConvWeights, DFlashDeviceAttentionWeights, DFlashDeviceConvWeights,
    DFlashDeviceLayerWeights, DFlashDeviceMlpWeights, DFlashDeviceWeights, DFlashLayerWeights,
    DFlashMlpWeights, DFlashState, DFlashWeightIndex, DFlashWeights, QwenArtifactIndex,
    QwenArtifactSummary, QwenAttentionWeights, QwenDeviceAttentionWeights,
    QwenDeviceFullAttentionWeights, QwenDeviceLayerWeights, QwenDeviceLinearAttentionWeights,
    QwenDeviceMlpWeights, QwenDeviceMtpLayerWeights, QwenDeviceMtpWeights, QwenDeviceRootWeights,
    QwenDeviceWeightSummary, QwenDeviceWeights, QwenFullAttentionWeights, QwenLayerState,
    QwenLayerWeights, QwenLinearAttentionWeights, QwenMlpWeights, QwenModelState,
    QwenMtpLayerWeights, QwenMtpState, QwenMtpWeightIndex, QwenMtpWeights, QwenRootWeights,
    QwenWeightIndex, QwenWeightSummary, DFLASH_MODEL_FILE, DFLASH_TENSOR_COUNT,
    QWEN_MLX_W4_REPO_ID, QWEN_MLX_W4_REPO_URL, QWEN_MLX_W4_SHARD_COUNT, QWEN_MLX_W4_TENSOR_COUNT,
    QWEN_MLX_W4_TEXT_TENSOR_COUNT, QWEN_MLX_W4_TOTAL_SHARD_BYTES, QWEN_MTP_DIRECTORY,
    QWEN_MTP_W4_REPO_ID, QWEN_MTP_W4_REPO_URL, QWEN_MTP_W4_REVISION, QWEN_MTP_W4_TENSOR_COUNT,
    QWEN_VISION_TENSOR_COUNT,
};
pub(crate) use rms_norm::RmsNorm;
pub use rms_norm::RmsNormLoadReport;
pub(crate) use sparse_block::SparseBlock;
pub use sparse_block::{SparseBlockForwardReport, SparseBlockLoadReport};
pub use weights::TensorLoadReport;
pub(crate) use weights::WeightLoader;
