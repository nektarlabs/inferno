use std::{fs, path::Path};

use common::{Error, Result};
use serde::Deserialize;
use tracing::info;

const ARCHITECTURE: &str = "Qwen3_5ForConditionalGeneration";
const MODEL_TYPE: &str = "qwen3_5";
const TEXT_MODEL_TYPE: &str = "qwen3_5_text";
const HIDDEN_SIZE: usize = 5_120;
const LAYER_COUNT: usize = 64;
const INTERMEDIATE_SIZE: usize = 17_408;
const VOCAB_SIZE: usize = 248_320;
const MAX_CONTEXT: usize = 262_144;
const FULL_ATTENTION_INTERVAL: usize = 4;
const ATTENTION_HEADS: usize = 24;
const KV_HEADS: usize = 4;
const ATTENTION_HEAD_DIM: usize = 256;
const LINEAR_QK_HEADS: usize = 16;
const LINEAR_VALUE_HEADS: usize = 48;
const LINEAR_HEAD_DIM: usize = 128;
const LINEAR_CONV_KERNEL: usize = 4;
const MTP_LAYER_COUNT: usize = 1;
const BOS_EOS_TOKEN_ID: u32 = 248_044;
const RMS_NORM_EPS: f64 = 1e-6;
const ROPE_THETA: f64 = 10_000_000.0;
const PARTIAL_ROTARY_FACTOR: f64 = 0.25;
const MLX_W4_GROUP_SIZE: usize = 64;
const MTP_MODEL_TYPE: &str = "qwen3_5_mtp";
const MTP_BLOCK_SIZE: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QwenLayerKind {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct QwenConfig {
    pub architectures: Vec<String>,
    pub language_model_only: bool,
    pub model_type: String,
    pub text_config: QwenTextConfig,
    pub tie_word_embeddings: bool,
    pub quantization_config: QwenQuantizationConfig,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct QwenTextConfig {
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub attn_output_gate: bool,
    pub bos_token_id: u32,
    pub dtype: String,
    pub eos_token_id: u32,
    pub full_attention_interval: usize,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_types: Vec<QwenLayerKind>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub mamba_ssm_dtype: String,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub output_gate_type: String,
    pub partial_rotary_factor: f64,
    pub rms_norm_eps: f64,
    pub rope_parameters: QwenRopeParameters,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct QwenRopeParameters {
    pub mrope_interleaved: bool,
    pub mrope_section: Vec<usize>,
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,
    pub rope_type: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct QwenQuantizationConfig {
    pub group_size: usize,
    pub bits: usize,
    pub mode: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct QwenMtpConfig {
    pub model_type: String,
    pub block_size: usize,
    pub text_config: QwenMtpTextConfig,
    pub quantization_config: QwenQuantizationConfig,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct QwenMtpTextConfig {
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
}

impl QwenConfig {
    pub fn from_json_str(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)?;
        config.validated()
    }

    pub fn validated(self) -> Result<Self> {
        validate_config(&self)?;
        Ok(self)
    }

    pub fn layer_kind(&self, layer_index: usize) -> Option<QwenLayerKind> {
        self.text_config.layer_types.get(layer_index).copied()
    }

    pub fn linear_attention_layer_count(&self) -> usize {
        self.text_config
            .layer_types
            .iter()
            .filter(|kind| **kind == QwenLayerKind::LinearAttention)
            .count()
    }

    pub fn full_attention_layer_count(&self) -> usize {
        self.text_config
            .layer_types
            .iter()
            .filter(|kind| **kind == QwenLayerKind::FullAttention)
            .count()
    }

    pub const fn full_query_width(&self) -> usize {
        ATTENTION_HEADS * ATTENTION_HEAD_DIM
    }

    pub const fn full_kv_width(&self) -> usize {
        KV_HEADS * ATTENTION_HEAD_DIM
    }

    pub const fn rotary_dim(&self) -> usize {
        ATTENTION_HEAD_DIM / 4
    }

    pub const fn linear_qk_width(&self) -> usize {
        LINEAR_QK_HEADS * LINEAR_HEAD_DIM
    }

    pub const fn linear_value_width(&self) -> usize {
        LINEAR_VALUE_HEADS * LINEAR_HEAD_DIM
    }

    pub const fn linear_qkv_width(&self) -> usize {
        (2 * LINEAR_QK_HEADS + LINEAR_VALUE_HEADS) * LINEAR_HEAD_DIM
    }
}

impl QwenMtpConfig {
    pub fn from_json_str(json: &str, target: &QwenConfig) -> Result<Self> {
        let config: Self = serde_json::from_str(json)?;
        validate_mtp_config(&config, target)?;
        Ok(config)
    }
}

pub fn load_qwen_config(path: &Path) -> Result<QwenConfig> {
    info!(path = %path.display(), "loading external Qwen3.8 config");
    let json = fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    QwenConfig::from_json_str(&json)
}

pub fn load_qwen_mtp_config(path: &Path, target: &QwenConfig) -> Result<QwenMtpConfig> {
    info!(path = %path.display(), "loading Qwen3.8 Q4 MTP config");
    let json = fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    QwenMtpConfig::from_json_str(&json, target)
}

fn validate_mtp_config(config: &QwenMtpConfig, target: &QwenConfig) -> Result<()> {
    require_exact("MTP model_type", config.model_type.as_str(), MTP_MODEL_TYPE)?;
    require_exact("MTP block_size", config.block_size, MTP_BLOCK_SIZE)?;
    require_exact(
        "MTP quantization_config.group_size",
        config.quantization_config.group_size,
        MLX_W4_GROUP_SIZE,
    )?;
    require_exact(
        "MTP quantization_config.bits",
        config.quantization_config.bits,
        4,
    )?;
    require_exact(
        "MTP quantization_config.mode",
        config.quantization_config.mode.as_str(),
        "affine",
    )?;

    let mtp = &config.text_config;
    let target = &target.text_config;
    require_exact("MTP hidden_size", mtp.hidden_size, target.hidden_size)?;
    require_exact(
        "MTP intermediate_size",
        mtp.intermediate_size,
        target.intermediate_size,
    )?;
    require_exact(
        "MTP num_attention_heads",
        mtp.num_attention_heads,
        target.num_attention_heads,
    )?;
    require_exact(
        "MTP num_key_value_heads",
        mtp.num_key_value_heads,
        target.num_key_value_heads,
    )?;
    require_exact("MTP head_dim", mtp.head_dim, target.head_dim)?;
    require_exact("MTP vocab_size", mtp.vocab_size, target.vocab_size)?;
    Ok(())
}

fn validate_config(config: &QwenConfig) -> Result<()> {
    require_exact(
        "architectures",
        config.architectures.as_slice(),
        &[ARCHITECTURE.to_string()],
    )?;
    require_exact("model_type", config.model_type.as_str(), MODEL_TYPE)?;
    require_exact("language_model_only", config.language_model_only, false)?;
    require_exact("tie_word_embeddings", config.tie_word_embeddings, false)?;

    let text = &config.text_config;
    require_exact(
        "text_config.model_type",
        text.model_type.as_str(),
        TEXT_MODEL_TYPE,
    )?;
    require_exact("text_config.hidden_size", text.hidden_size, HIDDEN_SIZE)?;
    require_exact(
        "text_config.num_hidden_layers",
        text.num_hidden_layers,
        LAYER_COUNT,
    )?;
    require_exact(
        "text_config.intermediate_size",
        text.intermediate_size,
        INTERMEDIATE_SIZE,
    )?;
    require_exact("text_config.vocab_size", text.vocab_size, VOCAB_SIZE)?;
    require_exact(
        "text_config.max_position_embeddings",
        text.max_position_embeddings,
        MAX_CONTEXT,
    )?;
    require_exact(
        "text_config.full_attention_interval",
        text.full_attention_interval,
        FULL_ATTENTION_INTERVAL,
    )?;
    require_exact(
        "text_config.num_attention_heads",
        text.num_attention_heads,
        ATTENTION_HEADS,
    )?;
    require_exact(
        "text_config.num_key_value_heads",
        text.num_key_value_heads,
        KV_HEADS,
    )?;
    require_exact("text_config.head_dim", text.head_dim, ATTENTION_HEAD_DIM)?;
    require_exact(
        "text_config.linear_num_key_heads",
        text.linear_num_key_heads,
        LINEAR_QK_HEADS,
    )?;
    require_exact(
        "text_config.linear_num_value_heads",
        text.linear_num_value_heads,
        LINEAR_VALUE_HEADS,
    )?;
    require_exact(
        "text_config.linear_key_head_dim",
        text.linear_key_head_dim,
        LINEAR_HEAD_DIM,
    )?;
    require_exact(
        "text_config.linear_value_head_dim",
        text.linear_value_head_dim,
        LINEAR_HEAD_DIM,
    )?;
    require_exact(
        "text_config.linear_conv_kernel_dim",
        text.linear_conv_kernel_dim,
        LINEAR_CONV_KERNEL,
    )?;
    require_exact(
        "text_config.mamba_ssm_dtype",
        text.mamba_ssm_dtype.as_str(),
        "float32",
    )?;
    require_exact(
        "text_config.mtp_num_hidden_layers",
        text.mtp_num_hidden_layers,
        MTP_LAYER_COUNT,
    )?;
    require_exact(
        "text_config.mtp_use_dedicated_embeddings",
        text.mtp_use_dedicated_embeddings,
        false,
    )?;
    require_exact(
        "text_config.bos_token_id",
        text.bos_token_id,
        BOS_EOS_TOKEN_ID,
    )?;
    require_exact(
        "text_config.eos_token_id",
        text.eos_token_id,
        BOS_EOS_TOKEN_ID,
    )?;
    require_exact("text_config.dtype", text.dtype.as_str(), "bfloat16")?;
    require_exact("text_config.hidden_act", text.hidden_act.as_str(), "silu")?;
    require_exact(
        "text_config.output_gate_type",
        text.output_gate_type.as_str(),
        "swish",
    )?;
    require_exact("text_config.attention_bias", text.attention_bias, false)?;
    require_exact("text_config.attention_dropout", text.attention_dropout, 0.0)?;
    require_exact("text_config.attn_output_gate", text.attn_output_gate, true)?;
    require_exact(
        "text_config.tie_word_embeddings",
        text.tie_word_embeddings,
        false,
    )?;
    require_exact("text_config.use_cache", text.use_cache, true)?;
    require_exact(
        "text_config.partial_rotary_factor",
        text.partial_rotary_factor,
        PARTIAL_ROTARY_FACTOR,
    )?;
    require_exact("text_config.rms_norm_eps", text.rms_norm_eps, RMS_NORM_EPS)?;
    validate_layer_schedule(text)?;
    validate_rope(&text.rope_parameters)?;
    validate_quantization(&config.quantization_config)?;
    Ok(())
}

fn validate_layer_schedule(config: &QwenTextConfig) -> Result<()> {
    if config.layer_types.len() != LAYER_COUNT {
        return Err(Error::config(format!(
            "Qwen3.8 layer_types must contain {LAYER_COUNT} entries, got {}",
            config.layer_types.len()
        )));
    }
    for (layer_index, actual) in config.layer_types.iter().copied().enumerate() {
        let expected = if (layer_index + 1).is_multiple_of(FULL_ATTENTION_INTERVAL) {
            QwenLayerKind::FullAttention
        } else {
            QwenLayerKind::LinearAttention
        };
        if actual != expected {
            return Err(Error::config(format!(
                "Qwen3.8 layer {layer_index} must be {expected:?}, got {actual:?}"
            )));
        }
    }
    Ok(())
}

fn validate_rope(rope: &QwenRopeParameters) -> Result<()> {
    require_exact(
        "rope_parameters.rope_type",
        rope.rope_type.as_str(),
        "default",
    )?;
    require_exact("rope_parameters.rope_theta", rope.rope_theta, ROPE_THETA)?;
    require_exact(
        "rope_parameters.partial_rotary_factor",
        rope.partial_rotary_factor,
        PARTIAL_ROTARY_FACTOR,
    )?;
    require_exact(
        "rope_parameters.mrope_interleaved",
        rope.mrope_interleaved,
        true,
    )?;
    require_exact(
        "rope_parameters.mrope_section",
        rope.mrope_section.as_slice(),
        &[11, 11, 10],
    )
}

fn validate_quantization(quantization: &QwenQuantizationConfig) -> Result<()> {
    require_exact(
        "quantization_config.group_size",
        quantization.group_size,
        MLX_W4_GROUP_SIZE,
    )?;
    require_exact("quantization_config.bits", quantization.bits, 4)?;
    require_exact(
        "quantization_config.mode",
        quantization.mode.as_str(),
        "affine",
    )
}

fn require_exact<T>(name: &str, actual: T, expected: T) -> Result<()>
where
    T: PartialEq + std::fmt::Debug,
{
    if actual != expected {
        return Err(Error::config(format!(
            "Qwen3.8 {name} must be {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn validates_official_text_and_mlx_w4_contract() {
        let config = QwenConfig::from_json_str(&supported_config_json()).unwrap();

        assert_eq!(config.linear_attention_layer_count(), 48);
        assert_eq!(config.full_attention_layer_count(), 16);
        assert_eq!(config.full_query_width(), 6_144);
        assert_eq!(config.full_kv_width(), 1_024);
        assert_eq!(config.rotary_dim(), 64);
        assert_eq!(config.linear_qk_width(), 2_048);
        assert_eq!(config.linear_value_width(), 6_144);
        assert_eq!(config.linear_qkv_width(), 10_240);
    }

    #[test]
    fn rejects_a_changed_layer_schedule() {
        let mut value: serde_json::Value = serde_json::from_str(&supported_config_json()).unwrap();
        value["text_config"]["layer_types"][3] = json!("linear_attention");

        let error = QwenConfig::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("layer 3"));
    }

    #[test]
    fn rejects_a_different_mlx_group_size() {
        let mut value: serde_json::Value = serde_json::from_str(&supported_config_json()).unwrap();
        value["quantization_config"]["group_size"] = json!(32);

        let error = QwenConfig::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("group_size"));
    }

    #[test]
    fn rejects_non_four_bit_mlx_quantization() {
        let mut value: serde_json::Value = serde_json::from_str(&supported_config_json()).unwrap();
        value["quantization_config"]["bits"] = json!(3);

        let error = QwenConfig::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("bits"));
    }

    #[test]
    fn validates_the_q4_mtp_companion_contract() {
        let target = QwenConfig::from_json_str(&supported_config_json()).unwrap();
        let config = QwenMtpConfig::from_json_str(&supported_mtp_config_json(), &target).unwrap();

        assert_eq!(config.block_size, 3);
        assert_eq!(config.quantization_config.bits, 4);
    }

    #[test]
    fn rejects_an_mtp_companion_for_a_different_target() {
        let target = QwenConfig::from_json_str(&supported_config_json()).unwrap();
        let mut value: serde_json::Value =
            serde_json::from_str(&supported_mtp_config_json()).unwrap();
        value["text_config"]["hidden_size"] = json!(4_096);

        let error = QwenMtpConfig::from_json_str(&value.to_string(), &target).unwrap_err();
        assert!(error.to_string().contains("MTP hidden_size"));
    }

    fn supported_config_json() -> String {
        let layer_types = (0..LAYER_COUNT)
            .map(|layer_index| {
                if (layer_index + 1).is_multiple_of(FULL_ATTENTION_INTERVAL) {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect::<Vec<_>>();
        json!({
            "architectures": [ARCHITECTURE],
            "language_model_only": false,
            "model_type": MODEL_TYPE,
            "tie_word_embeddings": false,
            "quantization_config": {
                "group_size": MLX_W4_GROUP_SIZE,
                "bits": 4,
                "mode": "affine",
            },
            "text_config": {
                "attention_bias": false,
                "attention_dropout": 0.0,
                "attn_output_gate": true,
                "bos_token_id": BOS_EOS_TOKEN_ID,
                "dtype": "bfloat16",
                "eos_token_id": BOS_EOS_TOKEN_ID,
                "full_attention_interval": FULL_ATTENTION_INTERVAL,
                "head_dim": ATTENTION_HEAD_DIM,
                "hidden_act": "silu",
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": INTERMEDIATE_SIZE,
                "layer_types": layer_types,
                "linear_conv_kernel_dim": LINEAR_CONV_KERNEL,
                "linear_key_head_dim": LINEAR_HEAD_DIM,
                "linear_num_key_heads": LINEAR_QK_HEADS,
                "linear_num_value_heads": LINEAR_VALUE_HEADS,
                "linear_value_head_dim": LINEAR_HEAD_DIM,
                "mamba_ssm_dtype": "float32",
                "max_position_embeddings": MAX_CONTEXT,
                "model_type": TEXT_MODEL_TYPE,
                "mtp_num_hidden_layers": MTP_LAYER_COUNT,
                "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": ATTENTION_HEADS,
                "num_hidden_layers": LAYER_COUNT,
                "num_key_value_heads": KV_HEADS,
                "output_gate_type": "swish",
                "partial_rotary_factor": PARTIAL_ROTARY_FACTOR,
                "rms_norm_eps": RMS_NORM_EPS,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10],
                    "partial_rotary_factor": PARTIAL_ROTARY_FACTOR,
                    "rope_theta": ROPE_THETA,
                    "rope_type": "default"
                },
                "tie_word_embeddings": false,
                "use_cache": true,
                "vocab_size": VOCAB_SIZE
            }
        })
        .to_string()
    }

    fn supported_mtp_config_json() -> String {
        json!({
            "model_type": MTP_MODEL_TYPE,
            "block_size": MTP_BLOCK_SIZE,
            "quantization_config": {
                "group_size": MLX_W4_GROUP_SIZE,
                "bits": 4,
                "mode": "affine",
            },
            "text_config": {
                "head_dim": ATTENTION_HEAD_DIM,
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": INTERMEDIATE_SIZE,
                "num_attention_heads": ATTENTION_HEADS,
                "num_key_value_heads": KV_HEADS,
                "vocab_size": VOCAB_SIZE,
            }
        })
        .to_string()
    }
}
