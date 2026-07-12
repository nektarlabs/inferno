use serde::{de::Error as DeError, Deserialize, Deserializer, Serialize};

use crate::validation::validate_config;
use common::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IndexerLayerKind {
    Full,
    Shared,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dense_layers: usize,
    pub sparse_moe_layers: Option<usize>,
    pub vocab_size: usize,
    pub attention_heads: usize,
    pub qk_head_dim: usize,
    pub qk_no_rope_dim: usize,
    pub qk_rope_dim: usize,
    pub kv_lora_rank: usize,
    pub v_head_dim: Option<usize>,
    pub num_routed_experts: usize,
    pub experts_per_token: usize,
    pub moe_intermediate_size: usize,
    pub num_shared_experts: usize,
    pub moe_groups: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f64,
    pub scoring_func: String,
    pub topk_method: String,
    pub max_context: usize,
    pub dsa_index_topk: usize,
    pub index_head_dim: usize,
    pub index_n_heads: usize,
    pub index_topk_freq: usize,
    pub indexer_rope_interleave: bool,
    pub indexer_types: Vec<IndexerLayerKind>,
    pub num_nextn_predict_layers: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    model_type: Option<String>,
    hidden_size: Option<usize>,
    num_layers: Option<usize>,
    num_hidden_layers: Option<usize>,
    dense_layers: Option<usize>,
    first_k_dense_replace: Option<usize>,
    sparse_moe_layers: Option<usize>,
    vocab_size: Option<usize>,
    attention_heads: Option<usize>,
    num_attention_heads: Option<usize>,
    head_dim: Option<usize>,
    qk_head_dim: Option<usize>,
    qk_no_rope_dim: Option<usize>,
    qk_nope_head_dim: Option<usize>,
    qk_rope_dim: Option<usize>,
    qk_rope_head_dim: Option<usize>,
    kv_lora_rank: Option<usize>,
    v_head_dim: Option<usize>,
    num_routed_experts: Option<usize>,
    n_routed_experts: Option<usize>,
    experts_per_token: Option<usize>,
    num_experts_per_tok: Option<usize>,
    intermediate_size: Option<usize>,
    moe_intermediate_size: Option<usize>,
    num_shared_experts: Option<usize>,
    n_shared_experts: Option<usize>,
    moe_groups: Option<usize>,
    n_group: Option<usize>,
    topk_group: Option<usize>,
    norm_topk_prob: Option<bool>,
    routed_scaling_factor: Option<f64>,
    scoring_func: Option<String>,
    topk_method: Option<String>,
    max_context: Option<usize>,
    max_position_embeddings: Option<usize>,
    dsa_index_topk: Option<usize>,
    index_topk: Option<usize>,
    index_head_dim: Option<usize>,
    index_n_heads: Option<usize>,
    index_topk_freq: Option<usize>,
    indexer_rope_interleave: Option<bool>,
    indexer_types: Option<Vec<IndexerLayerKind>>,
    num_nextn_predict_layers: Option<usize>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    rope_parameters: Option<RawRopeParameters>,
}

#[derive(Debug, Deserialize)]
struct RawRopeParameters {
    rope_theta: Option<f64>,
}

impl<'de> Deserialize<'de> for Config {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawConfig::deserialize(deserializer)?;
        Ok(Self {
            model_type: required(raw.model_type, "model_type")?,
            hidden_size: required(raw.hidden_size, "hidden_size")?,
            num_layers: required(
                prefer(raw.num_hidden_layers, raw.num_layers),
                "num_hidden_layers/num_layers",
            )?,
            dense_layers: required(
                prefer(raw.first_k_dense_replace, raw.dense_layers),
                "first_k_dense_replace/dense_layers",
            )?,
            sparse_moe_layers: raw.sparse_moe_layers,
            vocab_size: required(raw.vocab_size, "vocab_size")?,
            attention_heads: required(
                prefer(raw.num_attention_heads, raw.attention_heads),
                "num_attention_heads/attention_heads",
            )?,
            qk_head_dim: required(
                prefer(raw.qk_head_dim, raw.head_dim),
                "qk_head_dim/head_dim",
            )?,
            qk_no_rope_dim: required(
                prefer(raw.qk_nope_head_dim, raw.qk_no_rope_dim),
                "qk_nope_head_dim/qk_no_rope_dim",
            )?,
            qk_rope_dim: required(
                prefer(raw.qk_rope_head_dim, raw.qk_rope_dim),
                "qk_rope_head_dim/qk_rope_dim",
            )?,
            kv_lora_rank: raw.kv_lora_rank.unwrap_or_else(default_kv_lora_rank),
            v_head_dim: raw.v_head_dim,
            num_routed_experts: required(
                prefer(raw.n_routed_experts, raw.num_routed_experts),
                "n_routed_experts/num_routed_experts",
            )?,
            experts_per_token: required(
                prefer(raw.num_experts_per_tok, raw.experts_per_token),
                "num_experts_per_tok/experts_per_token",
            )?,
            moe_intermediate_size: prefer(raw.moe_intermediate_size, raw.intermediate_size)
                .unwrap_or_else(default_moe_intermediate_size),
            num_shared_experts: prefer(raw.n_shared_experts, raw.num_shared_experts)
                .unwrap_or_else(default_num_shared_experts),
            moe_groups: prefer(raw.n_group, raw.moe_groups).unwrap_or_else(default_moe_groups),
            topk_group: raw.topk_group.unwrap_or_else(default_topk_group),
            norm_topk_prob: raw.norm_topk_prob.unwrap_or_else(default_norm_topk_prob),
            routed_scaling_factor: raw
                .routed_scaling_factor
                .unwrap_or_else(default_routed_scaling_factor),
            scoring_func: raw.scoring_func.unwrap_or_else(default_scoring_func),
            topk_method: raw.topk_method.unwrap_or_else(default_topk_method),
            max_context: required(
                prefer(raw.max_position_embeddings, raw.max_context),
                "max_position_embeddings/max_context",
            )?,
            dsa_index_topk: required(
                prefer(raw.index_topk, raw.dsa_index_topk),
                "index_topk/dsa_index_topk",
            )?,
            index_head_dim: raw.index_head_dim.unwrap_or_else(default_index_head_dim),
            index_n_heads: raw.index_n_heads.unwrap_or_else(default_index_n_heads),
            index_topk_freq: raw.index_topk_freq.unwrap_or_else(default_index_topk_freq),
            indexer_rope_interleave: raw
                .indexer_rope_interleave
                .unwrap_or_else(default_indexer_rope_interleave),
            indexer_types: raw.indexer_types.unwrap_or_default(),
            num_nextn_predict_layers: raw.num_nextn_predict_layers.unwrap_or(0),
            rms_norm_eps: raw.rms_norm_eps.unwrap_or_else(default_rms_norm_eps),
            rope_theta: raw
                .rope_parameters
                .and_then(|parameters| parameters.rope_theta)
                .or(raw.rope_theta)
                .unwrap_or_else(default_rope_theta),
        })
    }
}

impl Config {
    pub fn from_json_str(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)?;
        config.validated()
    }

    pub fn validated(mut self) -> Result<Self> {
        if self.sparse_moe_layers.is_none() {
            self.sparse_moe_layers = Some(
                self.num_layers
                    .checked_sub(self.dense_layers)
                    .ok_or_else(|| common::Error::config("dense_layers exceeds num_layers"))?,
            );
        }

        if self.v_head_dim.is_none() {
            self.v_head_dim = Some(self.qk_head_dim);
        }

        if self.indexer_types.is_empty() {
            self.indexer_types =
                default_indexer_types_for_dense_layers(self.num_layers, self.dense_layers);
        }

        validate_config(&self)?;
        Ok(self)
    }

    pub fn sparse_moe_layers(&self) -> usize {
        self.sparse_moe_layers
            .expect("validated config has sparse_moe_layers")
    }

    pub fn v_head_dim(&self) -> usize {
        self.v_head_dim.expect("validated config has v_head_dim")
    }

    pub fn merged_attention_width(&self) -> usize {
        self.attention_heads * self.v_head_dim()
    }

    pub fn indexer_layer_kind(&self, layer_index: usize) -> Option<IndexerLayerKind> {
        if layer_index >= self.num_layers {
            return None;
        }
        self.indexer_types.get(layer_index).copied().or_else(|| {
            if layer_index < self.dense_layers
                || (layer_index - self.dense_layers) % self.index_topk_freq == 0
            {
                Some(IndexerLayerKind::Full)
            } else {
                Some(IndexerLayerKind::Shared)
            }
        })
    }
}

fn default_rms_norm_eps() -> f64 {
    1e-5
}

fn default_moe_intermediate_size() -> usize {
    2048
}

fn default_kv_lora_rank() -> usize {
    512
}

fn default_num_shared_experts() -> usize {
    1
}

fn default_moe_groups() -> usize {
    1
}

fn default_topk_group() -> usize {
    1
}

fn default_norm_topk_prob() -> bool {
    true
}

fn default_routed_scaling_factor() -> f64 {
    1.0
}

fn default_scoring_func() -> String {
    "softmax".to_string()
}

fn default_topk_method() -> String {
    "greedy".to_string()
}

fn default_index_head_dim() -> usize {
    128
}

fn default_index_n_heads() -> usize {
    32
}

fn default_index_topk_freq() -> usize {
    4
}

fn default_indexer_rope_interleave() -> bool {
    true
}

fn default_indexer_types_for_dense_layers(
    num_layers: usize,
    dense_layers: usize,
) -> Vec<IndexerLayerKind> {
    (0..num_layers)
        .map(|layer_index| {
            if layer_index < dense_layers
                || (layer_index - dense_layers) % default_index_topk_freq() == 0
            {
                IndexerLayerKind::Full
            } else {
                IndexerLayerKind::Shared
            }
        })
        .collect()
}

fn default_rope_theta() -> f64 {
    10_000_000.0
}

fn prefer<T>(preferred: Option<T>, fallback: Option<T>) -> Option<T> {
    preferred.or(fallback)
}

fn required<T, E>(value: Option<T>, field: &str) -> std::result::Result<T, E>
where
    E: DeError,
{
    value.ok_or_else(|| E::custom(format!("missing required config field {field}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::GLM52_LIKE_CONFIG_JSON;

    #[test]
    fn embedded_snapshot_has_expected_defaults() {
        let config = Config::from_json_str(GLM52_LIKE_CONFIG_JSON).unwrap();

        assert_eq!(config.hidden_size, 6144);
        assert_eq!(config.num_layers, 78);
        assert_eq!(config.dense_layers, 3);
        assert_eq!(config.sparse_moe_layers(), 75);
        assert_eq!(config.vocab_size, 154880);
        assert_eq!(config.attention_heads, 64);
        assert_eq!(config.qk_head_dim, 256);
        assert_eq!(config.qk_no_rope_dim, 192);
        assert_eq!(config.qk_rope_dim, 64);
        assert_eq!(config.num_routed_experts, 256);
        assert_eq!(config.experts_per_token, 8);
        assert_eq!(config.moe_intermediate_size, 2048);
        assert_eq!(config.num_shared_experts, 1);
        assert_eq!(config.moe_groups, 1);
        assert_eq!(config.topk_group, 1);
        assert!(config.norm_topk_prob);
        assert_eq!(config.routed_scaling_factor, 2.5);
        assert_eq!(config.scoring_func, "sigmoid");
        assert_eq!(config.topk_method, "noaux_tc");
        assert_eq!(config.max_context, 1_048_576);
        assert_eq!(config.dsa_index_topk, 2048);
        assert_eq!(config.index_head_dim, 128);
        assert_eq!(config.index_n_heads, 32);
        assert_eq!(config.index_topk_freq, 4);
        assert!(config.indexer_rope_interleave);
        assert_eq!(config.indexer_types.len(), 78);
        assert_eq!(config.indexer_types[3], IndexerLayerKind::Full);
        assert_eq!(config.indexer_types[4], IndexerLayerKind::Shared);
    }

    #[test]
    fn accepts_external_aliases_used_by_model_configs() {
        let json = r#"{
          "model_type": "glm_moe_dsa",
          "hidden_size": 6144,
          "num_hidden_layers": 78,
          "first_k_dense_replace": 3,
          "vocab_size": 154880,
          "num_attention_heads": 64,
          "qk_head_dim": 256,
          "qk_nope_head_dim": 192,
          "qk_rope_head_dim": 64,
          "v_head_dim": 256,
          "n_routed_experts": 256,
          "num_experts_per_tok": 8,
          "max_position_embeddings": 1048576,
          "index_topk": 2048
        }"#;

        let config = Config::from_json_str(json).unwrap();

        assert_eq!(config.num_layers, 78);
        assert_eq!(config.dense_layers, 3);
        assert_eq!(config.sparse_moe_layers(), 75);
        assert_eq!(config.index_head_dim, 128);
        assert_eq!(config.index_n_heads, 32);
    }

    #[test]
    fn real_config_duplicate_head_dim_prefers_qk_head_dim_and_nested_rope() {
        let json = r#"{
          "model_type": "glm_moe_dsa",
          "hidden_size": 6144,
          "num_hidden_layers": 78,
          "first_k_dense_replace": 3,
          "vocab_size": 154880,
          "num_attention_heads": 64,
          "head_dim": 192,
          "qk_head_dim": 256,
          "qk_nope_head_dim": 192,
          "qk_rope_head_dim": 64,
          "v_head_dim": 256,
          "n_routed_experts": 256,
          "num_experts_per_tok": 8,
          "max_position_embeddings": 1048576,
          "index_topk": 2048,
          "rope_parameters": {
            "rope_theta": 8000000
          }
        }"#;

        let config = Config::from_json_str(json).unwrap();

        assert_eq!(config.qk_head_dim, 256);
        assert_eq!(config.qk_no_rope_dim, 192);
        assert_eq!(config.qk_rope_dim, 64);
        assert_eq!(config.rope_theta, 8_000_000.0);
    }
}
