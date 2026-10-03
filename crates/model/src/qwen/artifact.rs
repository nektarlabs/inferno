use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use common::{Error, Result};
use config::{QwenConfig, QwenLayerKind};
use inferno_io::SafeTensorIndex;

pub const QWEN_MLX_W4_REPO_ID: &str = "mlx-community/Qwen3.8-27B-4bit";
pub const QWEN_MLX_W4_REPO_URL: &str = "https://huggingface.co/mlx-community/Qwen3.8-27B-4bit";
pub const QWEN_MLX_W4_TOTAL_SHARD_BYTES: u64 = 16_054_262_240;
pub const QWEN_MLX_W4_TENSOR_COUNT: usize = 2_180;
pub const QWEN_MLX_W4_SHARD_COUNT: usize = 3;
pub const QWEN_MLX_W4_TEXT_TENSOR_COUNT: usize = 1_847;
pub const QWEN_VISION_TENSOR_COUNT: usize = 333;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenArtifactSummary {
    pub total_shard_bytes: u64,
    pub shard_count: usize,
    pub tensor_count: usize,
    pub text_tensor_count: usize,
    pub vision_tensor_count: usize,
    pub base_text_shard_count: usize,
}

#[derive(Debug)]
pub struct QwenArtifactIndex {
    index: SafeTensorIndex,
    base_text_shards: Vec<PathBuf>,
    summary: QwenArtifactSummary,
}

impl QwenArtifactIndex {
    /// Validates the exact MLX 4-bit checkpoint without mapping tensor payloads.
    /// Vision tensors remain indexed but are never opened by the text runtime.
    pub fn open(model_dir: impl AsRef<Path>, config: &QwenConfig) -> Result<Self> {
        let index = SafeTensorIndex::open(model_dir)?;
        let text = expected_text_tensors(config)?;
        validate_index(&index, &text)?;
        let base_text_shards = mapped_text_shards(&index, &text)?;
        let summary = QwenArtifactSummary {
            total_shard_bytes: index.total_size(),
            shard_count: index.shard_count(),
            tensor_count: index.tensor_count(),
            text_tensor_count: QWEN_MLX_W4_TEXT_TENSOR_COUNT,
            vision_tensor_count: QWEN_VISION_TENSOR_COUNT,
            base_text_shard_count: base_text_shards.len(),
        };
        Ok(Self {
            index,
            base_text_shards,
            summary,
        })
    }

    pub fn index(&self) -> &SafeTensorIndex {
        &self.index
    }

    pub fn base_text_shards(&self) -> &[PathBuf] {
        &self.base_text_shards
    }

    pub const fn summary(&self) -> QwenArtifactSummary {
        self.summary
    }
}

fn expected_text_tensors(config: &QwenConfig) -> Result<BTreeSet<String>> {
    if config.text_config.layer_types.len() != config.text_config.num_hidden_layers {
        return Err(Error::weights(format!(
            "Qwen3.8 config contains {} layer descriptors for {} layers",
            config.text_config.layer_types.len(),
            config.text_config.num_hidden_layers
        )));
    }
    expected_text_tensors_for_layers(&config.text_config.layer_types)
}

fn expected_text_tensors_for_layers(layer_types: &[QwenLayerKind]) -> Result<BTreeSet<String>> {
    let mut names = BTreeSet::from(["language_model.model.norm.weight".to_string()]);
    insert_matrix(&mut names, "language_model.model.embed_tokens");
    insert_matrix(&mut names, "language_model.lm_head");

    for (layer_index, layer_kind) in layer_types.iter().copied().enumerate() {
        let prefix = format!("language_model.model.layers.{layer_index}");
        names.insert(format!("{prefix}.input_layernorm.weight"));
        names.insert(format!("{prefix}.post_attention_layernorm.weight"));
        for suffix in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
            insert_matrix(&mut names, &format!("{prefix}.{suffix}"));
        }
        match layer_kind {
            QwenLayerKind::LinearAttention => {
                for suffix in [
                    "linear_attn.A_log",
                    "linear_attn.conv1d.weight",
                    "linear_attn.dt_bias",
                    "linear_attn.norm.weight",
                ] {
                    names.insert(format!("{prefix}.{suffix}"));
                }
                for suffix in [
                    "linear_attn.in_proj_a",
                    "linear_attn.in_proj_b",
                    "linear_attn.in_proj_qkv",
                    "linear_attn.in_proj_z",
                    "linear_attn.out_proj",
                ] {
                    insert_matrix(&mut names, &format!("{prefix}.{suffix}"));
                }
            }
            QwenLayerKind::FullAttention => {
                names.insert(format!("{prefix}.self_attn.q_norm.weight"));
                names.insert(format!("{prefix}.self_attn.k_norm.weight"));
                for suffix in [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                ] {
                    insert_matrix(&mut names, &format!("{prefix}.{suffix}"));
                }
            }
        }
    }
    if names.len() != QWEN_MLX_W4_TEXT_TENSOR_COUNT {
        return Err(Error::weights(format!(
            "internal Qwen3.8 MLX W4 contract has {} names, expected {QWEN_MLX_W4_TEXT_TENSOR_COUNT}",
            names.len()
        )));
    }
    Ok(names)
}

fn insert_matrix(names: &mut BTreeSet<String>, prefix: &str) {
    for suffix in ["weight", "scales", "biases"] {
        names.insert(format!("{prefix}.{suffix}"));
    }
}

fn validate_index(index: &SafeTensorIndex, text: &BTreeSet<String>) -> Result<()> {
    if index.total_size() != QWEN_MLX_W4_TOTAL_SHARD_BYTES {
        return Err(Error::weights(format!(
            "Qwen3.8 MLX W4 shard bytes must be {QWEN_MLX_W4_TOTAL_SHARD_BYTES}, got {}",
            index.total_size()
        )));
    }
    if index.shard_count() != QWEN_MLX_W4_SHARD_COUNT {
        return Err(Error::weights(format!(
            "Qwen3.8 MLX W4 must contain {QWEN_MLX_W4_SHARD_COUNT} Safetensors shards, got {}",
            index.shard_count()
        )));
    }
    if index.tensor_count() != QWEN_MLX_W4_TENSOR_COUNT {
        return Err(Error::weights(format!(
            "Qwen3.8 MLX W4 index must contain {QWEN_MLX_W4_TENSOR_COUNT} tensors, got {}",
            index.tensor_count()
        )));
    }
    for name in text {
        index.shard_for(name)?;
    }
    let unexpected = index
        .tensor_names()
        .filter(|name| !text.contains(*name) && !name.starts_with("vision_tower."))
        .collect::<Vec<_>>();
    if !unexpected.is_empty() {
        return Err(Error::weights(format!(
            "Qwen3.8 MLX W4 contains unexpected tensors: {}",
            unexpected.join(", ")
        )));
    }
    let vision_count = index.tensor_count() - text.len();
    if vision_count != QWEN_VISION_TENSOR_COUNT {
        return Err(Error::weights(format!(
            "Qwen3.8 MLX W4 index must contain {QWEN_VISION_TENSOR_COUNT} vision tensors, got {vision_count}"
        )));
    }
    Ok(())
}

fn mapped_text_shards(index: &SafeTensorIndex, text: &BTreeSet<String>) -> Result<Vec<PathBuf>> {
    let shards = text
        .iter()
        .map(|name| index.shard_for(name).map(Path::to_path_buf))
        .collect::<Result<BTreeSet<_>>>()?
        .into_iter()
        .collect::<Vec<_>>();
    for path in &shards {
        if !path.is_file() {
            return Err(Error::weights(format!(
                "Qwen3.8 text shard is missing: {}",
                path.display()
            )));
        }
    }
    Ok(shards)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_w4_text_inventory_matches_the_published_artifact() {
        let layers = (0_usize..64)
            .map(|layer| {
                if (layer + 1).is_multiple_of(4) {
                    QwenLayerKind::FullAttention
                } else {
                    QwenLayerKind::LinearAttention
                }
            })
            .collect::<Vec<_>>();
        let names = expected_text_tensors_for_layers(&layers).unwrap();
        assert_eq!(names.len(), QWEN_MLX_W4_TEXT_TENSOR_COUNT);
        assert!(names.contains("language_model.model.embed_tokens.weight"));
        assert!(names.contains("language_model.model.layers.63.mlp.down_proj.biases"));
        assert!(names.contains("language_model.lm_head.scales"));
    }
}
