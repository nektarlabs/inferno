use std::{path::Path, sync::Arc};

use common::{Error, Result};
use config::QwenConfig;
use inferno_io::{SafeTensorDtype, SafeTensorHandle, SafeTensorModel};

use super::{
    weights::{required, required_mlx_w4},
    QwenFullAttentionWeights, QwenMatrix, QwenMlpWeights,
};

pub const QWEN_MTP_DIRECTORY: &str = "mtp";
pub const QWEN_MTP_W4_REPO_ID: &str = "mlx-community/Qwen3.8-27B-MTP-4bit";
pub const QWEN_MTP_W4_REPO_URL: &str = "https://huggingface.co/mlx-community/Qwen3.8-27B-MTP-4bit";
pub const QWEN_MTP_W4_REVISION: &str = "b643c01b6d3b094e325edb6ebd832e16c486c575";
pub const QWEN_MTP_W4_TENSOR_COUNT: usize = 31;

#[derive(Debug, Clone)]
pub struct QwenMtpLayerWeights {
    pub input_norm: SafeTensorHandle,
    pub post_attention_norm: SafeTensorHandle,
    pub attention: QwenFullAttentionWeights,
    pub mlp: QwenMlpWeights,
}

#[derive(Debug, Clone)]
pub struct QwenMtpWeights {
    pub pre_fc_hidden_norm: SafeTensorHandle,
    pub pre_fc_embedding_norm: SafeTensorHandle,
    pub fusion: QwenMatrix,
    pub layer: QwenMtpLayerWeights,
    pub final_norm: SafeTensorHandle,
}

#[derive(Debug)]
pub struct QwenMtpWeightIndex {
    _storage: Arc<SafeTensorModel>,
    pub weights: QwenMtpWeights,
}

impl QwenMtpWeightIndex {
    /// Opens only the official 4-bit MTP companion artifact.
    pub fn open(mtp_dir: impl AsRef<Path>, config: &QwenConfig) -> Result<Self> {
        let storage = Arc::new(SafeTensorModel::open(mtp_dir)?);
        if storage.index().tensor_count() != QWEN_MTP_W4_TENSOR_COUNT {
            return Err(Error::weights(format!(
                "Qwen Q4 MTP artifact must contain {QWEN_MTP_W4_TENSOR_COUNT} tensors, got {}",
                storage.index().tensor_count()
            )));
        }

        let hidden = config.text_config.hidden_size;
        let intermediate = config.text_config.intermediate_size;
        let query_width = 2 * config.full_query_width();
        let kv_width = config.full_kv_width();
        let attention_width = config.full_query_width();
        let head_dim = config.text_config.head_dim;
        let layer = "layers.0";

        let weights = QwenMtpWeights {
            pre_fc_hidden_norm: required(
                &storage,
                "pre_fc_norm_hidden.weight",
                SafeTensorDtype::Bf16,
                &[hidden],
            )?,
            pre_fc_embedding_norm: required(
                &storage,
                "pre_fc_norm_embedding.weight",
                SafeTensorDtype::Bf16,
                &[hidden],
            )?,
            fusion: required_mlx_w4(&storage, "fc", hidden, 2 * hidden)?,
            layer: QwenMtpLayerWeights {
                input_norm: required(
                    &storage,
                    &format!("{layer}.input_layernorm.weight"),
                    SafeTensorDtype::Bf16,
                    &[hidden],
                )?,
                post_attention_norm: required(
                    &storage,
                    &format!("{layer}.post_attention_layernorm.weight"),
                    SafeTensorDtype::Bf16,
                    &[hidden],
                )?,
                attention: QwenFullAttentionWeights {
                    query: required_mlx_w4(
                        &storage,
                        &format!("{layer}.self_attn.q_proj"),
                        query_width,
                        hidden,
                    )?,
                    key: required_mlx_w4(
                        &storage,
                        &format!("{layer}.self_attn.k_proj"),
                        kv_width,
                        hidden,
                    )?,
                    value: required_mlx_w4(
                        &storage,
                        &format!("{layer}.self_attn.v_proj"),
                        kv_width,
                        hidden,
                    )?,
                    output: required_mlx_w4(
                        &storage,
                        &format!("{layer}.self_attn.o_proj"),
                        hidden,
                        attention_width,
                    )?,
                    query_norm: required(
                        &storage,
                        &format!("{layer}.self_attn.q_norm.weight"),
                        SafeTensorDtype::Bf16,
                        &[head_dim],
                    )?,
                    key_norm: required(
                        &storage,
                        &format!("{layer}.self_attn.k_norm.weight"),
                        SafeTensorDtype::Bf16,
                        &[head_dim],
                    )?,
                },
                mlp: QwenMlpWeights {
                    gate: required_mlx_w4(
                        &storage,
                        &format!("{layer}.mlp.gate_proj"),
                        intermediate,
                        hidden,
                    )?,
                    up: required_mlx_w4(
                        &storage,
                        &format!("{layer}.mlp.up_proj"),
                        intermediate,
                        hidden,
                    )?,
                    down: required_mlx_w4(
                        &storage,
                        &format!("{layer}.mlp.down_proj"),
                        hidden,
                        intermediate,
                    )?,
                },
            },
            final_norm: required(&storage, "norm.weight", SafeTensorDtype::Bf16, &[hidden])?,
        };

        Ok(Self {
            _storage: storage,
            weights,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_mtp_inventory_has_eight_matrices_and_seven_norms() {
        let matrices = 8;
        let bf16_norms = 7;
        assert_eq!(matrices * 3 + bf16_norms, QWEN_MTP_W4_TENSOR_COUNT);
    }
}
