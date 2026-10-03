use std::{fs, path::Path};

use common::{Error, Result};
use serde::Deserialize;
use tracing::info;

const ARCHITECTURE: &str = "DFlash2DraftModel";
const MODEL_TYPE: &str = "qwen3";
const DTYPE: &str = "bfloat16";
const HIDDEN_SIZE: usize = 5_120;
const INTERMEDIATE_SIZE: usize = 17_408;
const LAYER_COUNT: usize = 5;
const TARGET_LAYER_COUNT: usize = 64;
const ATTENTION_HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const VOCAB_SIZE: usize = 248_320;
const MAX_CONTEXT: usize = 262_144;
const SLIDING_WINDOW: usize = 2_048;
const RMS_NORM_EPS: f64 = 1e-6;
const ROPE_THETA: f64 = 10_000_000.0;
const BLOCK_SIZE: usize = 8;
const CONV_KERNEL_SIZE: usize = 2;
const CONV_GROUP_SIZE: usize = 16;
const SELECTOR_RANK: usize = 256;
const SELECTOR_TOP_K: usize = 16;
const MASK_TOKEN_ID: u32 = 248_070;
const EOS_TOKEN_ID: u32 = 248_044;
const TARGET_LAYER_IDS: [usize; LAYER_COUNT] = [5, 19, 33, 47, 61];

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DFlashConfig {
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub is_causal: bool,
    pub dflash_config: DFlashOptions,
    pub dtype: String,
    pub eos_token_id: u32,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_target_layers: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: DFlashRopeParameters,
    pub sliding_window: usize,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct DFlashOptions {
    pub block_size: usize,
    pub conv_group_size: usize,
    pub conv_kernel_size: usize,
    pub mask_token_id: u32,
    pub selector_rank: usize,
    pub selector_top_k: usize,
    pub target_layer_ids: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DFlashRopeParameters {
    pub rope_theta: f64,
    pub rope_type: String,
}

impl DFlashConfig {
    pub fn from_json_str(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)?;
        config.validated()
    }

    pub fn validated(self) -> Result<Self> {
        validate_config(&self)?;
        Ok(self)
    }

    pub const fn query_width(&self) -> usize {
        ATTENTION_HEADS * HEAD_DIM
    }

    pub const fn kv_width(&self) -> usize {
        KV_HEADS * HEAD_DIM
    }

    pub fn target_feature_width(&self) -> Result<usize> {
        self.hidden_size
            .checked_mul(self.dflash_config.target_layer_ids.len())
            .ok_or_else(|| Error::config("DFlash target feature width overflow"))
    }
}

pub fn load_dflash_config(path: &Path) -> Result<DFlashConfig> {
    info!(path = %path.display(), "loading external DFlash2 config");
    let json = fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    DFlashConfig::from_json_str(&json)
}

fn validate_config(config: &DFlashConfig) -> Result<()> {
    require_exact(
        "architectures",
        config.architectures.as_slice(),
        &[ARCHITECTURE.to_string()],
    )?;
    require_exact("model_type", config.model_type.as_str(), MODEL_TYPE)?;
    require_exact("dtype", config.dtype.as_str(), DTYPE)?;
    require_exact("hidden_size", config.hidden_size, HIDDEN_SIZE)?;
    require_exact(
        "intermediate_size",
        config.intermediate_size,
        INTERMEDIATE_SIZE,
    )?;
    require_exact("num_hidden_layers", config.num_hidden_layers, LAYER_COUNT)?;
    require_exact(
        "num_target_layers",
        config.num_target_layers,
        TARGET_LAYER_COUNT,
    )?;
    require_exact(
        "num_attention_heads",
        config.num_attention_heads,
        ATTENTION_HEADS,
    )?;
    require_exact("num_key_value_heads", config.num_key_value_heads, KV_HEADS)?;
    require_exact("head_dim", config.head_dim, HEAD_DIM)?;
    require_exact("vocab_size", config.vocab_size, VOCAB_SIZE)?;
    require_exact(
        "max_position_embeddings",
        config.max_position_embeddings,
        MAX_CONTEXT,
    )?;
    require_exact("sliding_window", config.sliding_window, SLIDING_WINDOW)?;
    require_exact("rms_norm_eps", config.rms_norm_eps, RMS_NORM_EPS)?;
    require_exact("eos_token_id", config.eos_token_id, EOS_TOKEN_ID)?;
    require_exact("attention_bias", config.attention_bias, false)?;
    require_exact("attention_dropout", config.attention_dropout, 0.0)?;
    require_exact("is_causal", config.is_causal, false)?;
    require_exact("tie_word_embeddings", config.tie_word_embeddings, false)?;
    require_exact("use_cache", config.use_cache, true)?;
    require_exact("hidden_act", config.hidden_act.as_str(), "silu")?;
    if config.layer_types.len() != LAYER_COUNT
        || config
            .layer_types
            .iter()
            .any(|layer_type| layer_type != "sliding_attention")
    {
        return Err(Error::config(format!(
            "unsupported DFlash2 layer_types: expected {LAYER_COUNT} sliding_attention layers, got {:?}",
            config.layer_types
        )));
    }
    require_exact(
        "rope_parameters.rope_type",
        config.rope_parameters.rope_type.as_str(),
        "default",
    )?;
    require_exact(
        "rope_parameters.rope_theta",
        config.rope_parameters.rope_theta,
        ROPE_THETA,
    )?;

    let options = &config.dflash_config;
    require_exact("dflash.block_size", options.block_size, BLOCK_SIZE)?;
    require_exact(
        "dflash.conv_kernel_size",
        options.conv_kernel_size,
        CONV_KERNEL_SIZE,
    )?;
    require_exact(
        "dflash.conv_group_size",
        options.conv_group_size,
        CONV_GROUP_SIZE,
    )?;
    require_exact("dflash.selector_rank", options.selector_rank, SELECTOR_RANK)?;
    require_exact(
        "dflash.selector_top_k",
        options.selector_top_k,
        SELECTOR_TOP_K,
    )?;
    require_exact("dflash.mask_token_id", options.mask_token_id, MASK_TOKEN_ID)?;
    require_exact(
        "dflash.target_layer_ids",
        options.target_layer_ids.as_slice(),
        TARGET_LAYER_IDS.as_slice(),
    )?;
    if MASK_TOKEN_ID as usize >= VOCAB_SIZE {
        return Err(Error::config(
            "DFlash mask token must be inside the target vocabulary",
        ));
    }
    Ok(())
}

fn require_exact<T>(name: &str, actual: T, expected: T) -> Result<()>
where
    T: PartialEq + std::fmt::Debug,
{
    if actual != expected {
        return Err(Error::config(format!(
            "unsupported DFlash2 {name}: expected {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const OFFICIAL_CONFIG: &str = r#"{
      "architectures":["DFlash2DraftModel"],
      "attention_bias":false,
      "attention_dropout":0.0,
      "is_causal":false,
      "dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,"mask_token_id":248070,"selector_rank":256,"selector_top_k":16,"target_layer_ids":[5,19,33,47,61]},
      "dtype":"bfloat16",
      "eos_token_id":248044,
      "head_dim":128,
      "hidden_act":"silu",
      "hidden_size":5120,
      "intermediate_size":17408,
      "layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
      "max_position_embeddings":262144,
      "model_type":"qwen3",
      "num_attention_heads":32,
      "num_hidden_layers":5,
      "num_key_value_heads":8,
      "num_target_layers":64,
      "rms_norm_eps":0.000001,
      "rope_parameters":{"rope_theta":10000000.0,"rope_type":"default"},
      "sliding_window":2048,
      "tie_word_embeddings":false,
      "use_cache":true,
      "vocab_size":248320
    }"#;

    #[test]
    fn accepts_official_dflash2_contract() {
        let config = DFlashConfig::from_json_str(OFFICIAL_CONFIG).unwrap();
        assert_eq!(config.target_feature_width().unwrap(), 25_600);
        assert_eq!(config.query_width(), 4_096);
        assert_eq!(config.kv_width(), 1_024);
    }

    #[test]
    fn rejects_a_different_target_layer_schedule() {
        let json = OFFICIAL_CONFIG.replace("[5,19,33,47,61]", "[4,19,33,47,61]");
        let error = DFlashConfig::from_json_str(&json).unwrap_err();
        assert!(error.to_string().contains("target_layer_ids"));
    }
}
