use std::{collections::BTreeMap, fs, path::Path};

use common::{Error, Result};
use serde::Deserialize;

const SUPPORTED_MODEL_TYPE: &str = "laguna";
const SUPPORTED_ARCHITECTURE: &str = "LagunaForCausalLM";
const SUPPORTED_VOCAB_SIZE: usize = 100_352;
const SUPPORTED_GLOBAL_QUERY_HEADS: usize = 48;
const SUPPORTED_KV_HEADS: usize = 8;
const SUPPORTED_HEAD_DIM: usize = 128;
const SUPPORTED_MAX_CONTEXT: usize = 262_144;
const SUPPORTED_EXPERTS: usize = 256;
const SUPPORTED_SLIDING_WINDOW: usize = 512;
const SUPPORTED_INT4_GROUP_SIZE: usize = 32;
const SUPPORTED_RMS_NORM_EPS: f64 = 1e-6;
const SUPPORTED_ROUTED_SCALING_FACTOR: f64 = 2.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LagunaProfile {
    S21,
    Xs21,
}

impl LagunaProfile {
    const fn contract(self) -> LagunaProfileContract {
        match self {
            Self::S21 => LagunaProfileContract {
                display_name: "Laguna S 2.1",
                hidden_size: 3_072,
                dense_intermediate_size: 12_288,
                moe_intermediate_size: 1_024,
                shared_intermediate_size: 1_024,
                layer_count: 48,
                top_k: 10,
                sliding_query_heads: 72,
                yarn_beta_fast: 32.0,
                requires_compressed_tensors_config: true,
            },
            Self::Xs21 => LagunaProfileContract {
                display_name: "Laguna XS 2.1",
                hidden_size: 2_048,
                dense_intermediate_size: 8_192,
                moe_intermediate_size: 512,
                shared_intermediate_size: 512,
                layer_count: 40,
                top_k: 8,
                sliding_query_heads: 64,
                yarn_beta_fast: 64.0,
                requires_compressed_tensors_config: false,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LagunaProfileContract {
    display_name: &'static str,
    hidden_size: usize,
    dense_intermediate_size: usize,
    moe_intermediate_size: usize,
    shared_intermediate_size: usize,
    layer_count: usize,
    top_k: usize,
    sliding_query_heads: usize,
    yarn_beta_fast: f64,
    requires_compressed_tensors_config: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LagunaAttentionKind {
    FullAttention,
    SlidingAttention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LagunaMlpKind {
    Dense,
    Sparse,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LagunaConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_attention_heads_per_layer: Vec<usize>,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub qkv_bias: bool,
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub rms_norm_eps: f64,
    #[serde(default = "default_hidden_activation")]
    pub hidden_act: String,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub moe_routed_scaling_factor: f64,
    pub moe_apply_router_weight_on_input: bool,
    #[serde(default)]
    pub moe_router_logit_softcapping: f64,
    pub decoder_sparse_step: usize,
    pub mlp_only_layers: Vec<usize>,
    pub sliding_window: usize,
    #[serde(default)]
    pub swa_attention_sink_enabled: bool,
    pub layer_types: Vec<LagunaAttentionKind>,
    pub mlp_layer_types: Vec<LagunaMlpKind>,
    pub gating: String,
    pub gating_types: Vec<String>,
    pub bos_token_id: u32,
    pub eos_token_id: Vec<u32>,
    pub pad_token_id: u32,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub torch_dtype: String,
    pub rope_parameters: LagunaRopeParameters,
    #[serde(default)]
    pub quantization_config: Option<LagunaQuantizationConfig>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LagunaRopeParameters {
    pub full_attention: LagunaRopeSettings,
    pub sliding_attention: LagunaRopeSettings,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LagunaRopeSettings {
    pub rope_type: String,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub factor: Option<f64>,
    pub original_max_position_embeddings: Option<usize>,
    pub beta_slow: Option<f64>,
    pub beta_fast: Option<f64>,
    pub attention_factor: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LagunaQuantizationConfig {
    pub config_groups: BTreeMap<String, LagunaQuantizationGroup>,
    pub format: String,
    pub quant_method: String,
    pub quantization_status: String,
    pub kv_cache_scheme: LagunaQuantizationArgs,
    pub transform_config: LagunaTransformConfig,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LagunaQuantizationGroup {
    pub format: String,
    pub targets: Vec<String>,
    pub weights: LagunaQuantizationArgs,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LagunaQuantizationArgs {
    pub dynamic: bool,
    pub group_size: Option<usize>,
    pub num_bits: usize,
    pub strategy: String,
    pub symmetric: bool,
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LagunaTransformConfig {
    pub config_groups: BTreeMap<String, LagunaTransformGroup>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LagunaTransformGroup {
    pub apply: Vec<LagunaTransformApplication>,
    pub head_dim: usize,
    pub randomize: bool,
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LagunaTransformApplication {
    pub inverse: bool,
    pub location: String,
    pub targets: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LagunaKvCacheBudget {
    pub context_tokens: usize,
    pub full_layer_count: usize,
    pub sliding_layer_count: usize,
    pub full_retained_tokens_per_layer: usize,
    pub sliding_retained_tokens_per_layer: usize,
    pub full_bytes: u64,
    pub sliding_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LagunaRopeFrequencies {
    pub rotary_dim: usize,
    pub inverse_frequencies: Vec<f32>,
    pub attention_factor: f32,
}

pub fn load_laguna_config(path: &Path) -> Result<LagunaConfig> {
    let json = fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let config: LagunaConfig = serde_json::from_str(&json).map_err(|source| {
        Error::config(format!(
            "failed to parse Laguna config at {}: {source}",
            path.display()
        ))
    })?;
    config.validated()
}

impl LagunaConfig {
    pub fn from_json_str(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)?;
        config.validated()
    }

    pub fn validated(self) -> Result<Self> {
        validate_supported_contract(&self)?;
        Ok(self)
    }

    pub fn profile(&self) -> Result<LagunaProfile> {
        detect_profile(self)
    }

    pub fn attention_kind(&self, layer_index: usize) -> Option<LagunaAttentionKind> {
        self.layer_types.get(layer_index).copied()
    }

    pub fn mlp_kind(&self, layer_index: usize) -> Option<LagunaMlpKind> {
        self.mlp_layer_types.get(layer_index).copied()
    }

    pub fn query_heads(&self, layer_index: usize) -> Option<usize> {
        self.num_attention_heads_per_layer.get(layer_index).copied()
    }

    pub fn query_width(&self, layer_index: usize) -> Option<usize> {
        self.query_heads(layer_index)?.checked_mul(self.head_dim)
    }

    pub fn key_value_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub fn rotary_dim(&self, layer_index: usize) -> Option<usize> {
        let factor = match self.attention_kind(layer_index)? {
            LagunaAttentionKind::FullAttention => {
                self.rope_parameters.full_attention.partial_rotary_factor
            }
            LagunaAttentionKind::SlidingAttention => {
                self.rope_parameters.sliding_attention.partial_rotary_factor
            }
        };
        Some((self.head_dim as f64 * factor) as usize)
    }

    pub fn rope_frequencies(&self, layer_index: usize) -> Result<LagunaRopeFrequencies> {
        let kind = self.attention_kind(layer_index).ok_or_else(|| {
            Error::config(format!("Laguna layer index {layer_index} is out of range"))
        })?;
        match kind {
            LagunaAttentionKind::FullAttention => {
                yarn_frequencies(self.head_dim, &self.rope_parameters.full_attention)
            }
            LagunaAttentionKind::SlidingAttention => {
                default_frequencies(self.head_dim, &self.rope_parameters.sliding_attention)
            }
        }
    }

    /// Returns the exact K/V storage needed when both K and V use one byte per
    /// value, as required by this checkpoint's FP8 cache contract.
    pub fn fp8_kv_cache_budget(
        &self,
        batch: usize,
        context_tokens: usize,
    ) -> Result<LagunaKvCacheBudget> {
        if batch == 0 {
            return Err(Error::config("Laguna KV cache batch must be positive"));
        }
        if context_tokens == 0 || context_tokens > self.max_position_embeddings {
            return Err(Error::config(format!(
                "Laguna KV context {context_tokens} must be within 1..={}",
                self.max_position_embeddings
            )));
        }

        let full_layer_count = self
            .layer_types
            .iter()
            .filter(|kind| **kind == LagunaAttentionKind::FullAttention)
            .count();
        let sliding_layer_count = self.num_hidden_layers - full_layer_count;
        let sliding_retained_tokens_per_layer = context_tokens.min(self.sliding_window);
        let row_bytes = checked_u64(batch)?
            .checked_mul(checked_u64(self.key_value_width())?)
            .and_then(|bytes| bytes.checked_mul(2))
            .ok_or_else(|| Error::config("Laguna KV row byte count overflow"))?;
        let full_bytes = row_bytes
            .checked_mul(checked_u64(context_tokens)?)
            .and_then(|bytes| bytes.checked_mul(checked_u64(full_layer_count).ok()?))
            .ok_or_else(|| Error::config("Laguna full-attention KV byte count overflow"))?;
        let sliding_bytes = row_bytes
            .checked_mul(checked_u64(sliding_retained_tokens_per_layer)?)
            .and_then(|bytes| bytes.checked_mul(checked_u64(sliding_layer_count).ok()?))
            .ok_or_else(|| Error::config("Laguna sliding-attention KV byte count overflow"))?;
        let total_bytes = full_bytes
            .checked_add(sliding_bytes)
            .ok_or_else(|| Error::config("Laguna total KV byte count overflow"))?;

        Ok(LagunaKvCacheBudget {
            context_tokens,
            full_layer_count,
            sliding_layer_count,
            full_retained_tokens_per_layer: context_tokens,
            sliding_retained_tokens_per_layer,
            full_bytes,
            sliding_bytes,
            total_bytes,
        })
    }
}

fn validate_supported_contract(config: &LagunaConfig) -> Result<()> {
    let profile = detect_profile(config)?;
    let contract = profile.contract();

    require_equal("model_type", &config.model_type, SUPPORTED_MODEL_TYPE)?;
    if config.architectures.as_slice() != [SUPPORTED_ARCHITECTURE] {
        return Err(Error::config(format!(
            "Laguna architectures must be [{SUPPORTED_ARCHITECTURE:?}], got {:?}",
            config.architectures
        )));
    }
    require_usize("vocab_size", config.vocab_size, SUPPORTED_VOCAB_SIZE)?;
    require_usize("hidden_size", config.hidden_size, contract.hidden_size)?;
    require_usize(
        "intermediate_size",
        config.intermediate_size,
        contract.dense_intermediate_size,
    )?;
    require_usize(
        "moe_intermediate_size",
        config.moe_intermediate_size,
        contract.moe_intermediate_size,
    )?;
    require_usize(
        "shared_expert_intermediate_size",
        config.shared_expert_intermediate_size,
        contract.shared_intermediate_size,
    )?;
    require_usize(
        "num_hidden_layers",
        config.num_hidden_layers,
        contract.layer_count,
    )?;
    require_usize(
        "num_attention_heads",
        config.num_attention_heads,
        SUPPORTED_GLOBAL_QUERY_HEADS,
    )?;
    require_usize(
        "num_key_value_heads",
        config.num_key_value_heads,
        SUPPORTED_KV_HEADS,
    )?;
    require_usize("head_dim", config.head_dim, SUPPORTED_HEAD_DIM)?;
    require_usize(
        "max_position_embeddings",
        config.max_position_embeddings,
        SUPPORTED_MAX_CONTEXT,
    )?;
    require_usize("num_experts", config.num_experts, SUPPORTED_EXPERTS)?;
    require_usize(
        "num_experts_per_tok",
        config.num_experts_per_tok,
        contract.top_k,
    )?;
    require_usize(
        "sliding_window",
        config.sliding_window,
        SUPPORTED_SLIDING_WINDOW,
    )?;
    if config.qkv_bias || config.attention_bias {
        return Err(Error::config(format!(
            "{} requires bias-free Q/K/V and attention output projections",
            contract.display_name
        )));
    }
    if config.hidden_act != "silu" {
        return Err(Error::config(format!(
            "Laguna hidden activation must be silu, got {:?}",
            config.hidden_act
        )));
    }
    if config.attention_dropout != 0.0 {
        return Err(Error::config(
            "Laguna inference requires attention_dropout=0",
        ));
    }
    if config.gating != "per-head" {
        return Err(Error::config(format!(
            "Laguna attention gating must be per-head, got {:?}",
            config.gating
        )));
    }
    if config.moe_apply_router_weight_on_input {
        return Err(Error::config(
            "Laguna router weights on expert inputs are not supported by this checkpoint",
        ));
    }
    if config.moe_router_logit_softcapping != 0.0 {
        return Err(Error::config(format!(
            "{} requires disabled router logit soft-capping",
            contract.display_name
        )));
    }
    if config.decoder_sparse_step != 1 || config.mlp_only_layers != [0] {
        return Err(Error::config(format!(
            "Laguna requires decoder_sparse_step=1 and mlp_only_layers=[0], got {}/{:?}",
            config.decoder_sparse_step, config.mlp_only_layers
        )));
    }
    if config.swa_attention_sink_enabled {
        return Err(Error::config(format!(
            "{} does not contain sliding-attention sink weights",
            contract.display_name
        )));
    }
    if config.gating_types.len() != config.num_hidden_layers
        || config
            .gating_types
            .iter()
            .any(|gating| gating != "per_head")
    {
        return Err(Error::config(format!(
            "Laguna gating_types must contain {} per_head entries",
            config.num_hidden_layers
        )));
    }
    if config.bos_token_id != 2 || config.eos_token_id != [2, 24] || config.pad_token_id != 9 {
        return Err(Error::config(format!(
            "Laguna special token IDs must be bos=2, eos=[2,24], pad=9; got bos={}, eos={:?}, pad={}",
            config.bos_token_id, config.eos_token_id, config.pad_token_id
        )));
    }
    if config.tie_word_embeddings || !config.use_cache || config.torch_dtype != "bfloat16" {
        return Err(Error::config(
            "Laguna requires untied embeddings, enabled KV cache, and bfloat16 fixed weights",
        ));
    }
    if !config.norm_topk_prob {
        return Err(Error::config(format!(
            "{} requires normalized top-k router weights",
            contract.display_name
        )));
    }
    if config.moe_routed_scaling_factor != SUPPORTED_ROUTED_SCALING_FACTOR
        || config.rms_norm_eps != SUPPORTED_RMS_NORM_EPS
    {
        return Err(Error::config(format!(
            "Laguna requires moe_routed_scaling_factor={SUPPORTED_ROUTED_SCALING_FACTOR} and rms_norm_eps={SUPPORTED_RMS_NORM_EPS}, got {}/{}",
            config.moe_routed_scaling_factor, config.rms_norm_eps
        )));
    }

    validate_layer_schedule(config, contract)?;
    validate_rope(config, contract)?;
    validate_quantization(config, contract)?;
    Ok(())
}

fn detect_profile(config: &LagunaConfig) -> Result<LagunaProfile> {
    match config.hidden_size {
        3_072 => Ok(LagunaProfile::S21),
        2_048 => Ok(LagunaProfile::Xs21),
        hidden_size => Err(Error::config(format!(
            "unsupported Laguna hidden_size {hidden_size}; supported profiles are Laguna S 2.1 (3072) and Laguna XS 2.1 (2048)"
        ))),
    }
}

fn validate_layer_schedule(config: &LagunaConfig, contract: LagunaProfileContract) -> Result<()> {
    if config.layer_types.len() != config.num_hidden_layers
        || config.mlp_layer_types.len() != config.num_hidden_layers
        || config.num_attention_heads_per_layer.len() != config.num_hidden_layers
    {
        return Err(Error::config(format!(
            "Laguna layer_types, mlp_layer_types, and per-layer head counts must each contain {} entries",
            config.num_hidden_layers
        )));
    }

    for layer_index in 0..config.num_hidden_layers {
        let expected_attention = if layer_index % 4 == 0 {
            LagunaAttentionKind::FullAttention
        } else {
            LagunaAttentionKind::SlidingAttention
        };
        if config.layer_types[layer_index] != expected_attention {
            return Err(Error::config(format!(
                "Laguna layer {layer_index} has {:?} attention; expected {:?}",
                config.layer_types[layer_index], expected_attention
            )));
        }
        let expected_heads = if expected_attention == LagunaAttentionKind::FullAttention {
            SUPPORTED_GLOBAL_QUERY_HEADS
        } else {
            contract.sliding_query_heads
        };
        if config.num_attention_heads_per_layer[layer_index] != expected_heads {
            return Err(Error::config(format!(
                "Laguna layer {layer_index} has {} query heads; expected {expected_heads}",
                config.num_attention_heads_per_layer[layer_index]
            )));
        }
        let expected_mlp = if layer_index == 0 {
            LagunaMlpKind::Dense
        } else {
            LagunaMlpKind::Sparse
        };
        if config.mlp_layer_types[layer_index] != expected_mlp {
            return Err(Error::config(format!(
                "Laguna layer {layer_index} has {:?} MLP; expected {:?}",
                config.mlp_layer_types[layer_index], expected_mlp
            )));
        }
        if expected_heads % config.num_key_value_heads != 0 {
            return Err(Error::config(format!(
                "Laguna layer {layer_index} query heads {expected_heads} are not divisible by {} KV heads",
                config.num_key_value_heads
            )));
        }
    }
    Ok(())
}

fn validate_rope(config: &LagunaConfig, contract: LagunaProfileContract) -> Result<()> {
    let full = &config.rope_parameters.full_attention;
    let sliding = &config.rope_parameters.sliding_attention;
    if full.rope_type != "yarn"
        || full.partial_rotary_factor != 0.5
        || full.rope_theta != 500_000.0
        || full.factor != Some(32.0)
        || full.original_max_position_embeddings != Some(8_192)
        || full.beta_slow != Some(1.0)
        || full.beta_fast != Some(contract.yarn_beta_fast)
        || full.attention_factor != Some(1.346_573_590_279_972_7)
        || sliding.rope_type != "default"
        || sliding.rope_theta != 10_000.0
        || sliding.partial_rotary_factor != 1.0
    {
        return Err(Error::config(
            "Laguna requires partial YaRN RoPE for full attention and full default RoPE for sliding attention",
        ));
    }
    for (name, theta) in [
        ("full_attention.rope_theta", full.rope_theta),
        ("sliding_attention.rope_theta", sliding.rope_theta),
    ] {
        if !theta.is_finite() || theta <= 0.0 {
            return Err(Error::config(format!("Laguna {name} must be positive")));
        }
    }
    if config.rotary_dim(0) != Some(64) || config.rotary_dim(1) != Some(128) {
        return Err(Error::config(
            "Laguna rotary dimensions must be 64 for full attention and 128 for sliding attention",
        ));
    }
    Ok(())
}

fn default_hidden_activation() -> String {
    "silu".to_string()
}

fn default_frequencies(
    head_dim: usize,
    settings: &LagunaRopeSettings,
) -> Result<LagunaRopeFrequencies> {
    let rotary_dim = rotary_dimension(head_dim, settings.partial_rotary_factor)?;
    let inverse_frequencies = (0..rotary_dim / 2)
        .map(|index| {
            1.0_f32 / (settings.rope_theta as f32).powf((2 * index) as f32 / rotary_dim as f32)
        })
        .collect();
    Ok(LagunaRopeFrequencies {
        rotary_dim,
        inverse_frequencies,
        attention_factor: 1.0,
    })
}

fn yarn_frequencies(
    head_dim: usize,
    settings: &LagunaRopeSettings,
) -> Result<LagunaRopeFrequencies> {
    let rotary_dim = rotary_dimension(head_dim, settings.partial_rotary_factor)?;
    let factor = settings
        .factor
        .ok_or_else(|| Error::config("Laguna YaRN factor is missing"))? as f32;
    let original_context = settings
        .original_max_position_embeddings
        .ok_or_else(|| Error::config("Laguna YaRN original context is missing"))?
        as f32;
    let beta_fast = settings
        .beta_fast
        .ok_or_else(|| Error::config("Laguna YaRN beta_fast is missing"))?
        as f32;
    let beta_slow = settings
        .beta_slow
        .ok_or_else(|| Error::config("Laguna YaRN beta_slow is missing"))?
        as f32;
    let base = settings.rope_theta as f32;
    let correction = |rotations: f32| {
        rotary_dim as f32 * (original_context / (rotations * 2.0 * std::f32::consts::PI)).ln()
            / (2.0 * base.ln())
    };
    let low = correction(beta_fast).floor().max(0.0);
    let high = correction(beta_slow).ceil().min((rotary_dim - 1) as f32);
    let ramp_denominator = if low == high { 0.001 } else { high - low };
    let inverse_frequencies = (0..rotary_dim / 2)
        .map(|index| {
            let positional_frequency = base.powf((2 * index) as f32 / rotary_dim as f32);
            let interpolated = 1.0 / (factor * positional_frequency);
            let extrapolated = 1.0 / positional_frequency;
            let ramp = ((index as f32 - low) / ramp_denominator).clamp(0.0, 1.0);
            let extrapolation_factor = 1.0 - ramp;
            interpolated * (1.0 - extrapolation_factor) + extrapolated * extrapolation_factor
        })
        .collect();
    let attention_factor = settings
        .attention_factor
        .ok_or_else(|| Error::config("Laguna YaRN attention_factor is missing"))?
        as f32;
    Ok(LagunaRopeFrequencies {
        rotary_dim,
        inverse_frequencies,
        attention_factor,
    })
}

fn rotary_dimension(head_dim: usize, factor: f64) -> Result<usize> {
    let rotary_dim = (head_dim as f64 * factor) as usize;
    if rotary_dim == 0 || rotary_dim > head_dim || !rotary_dim.is_multiple_of(2) {
        return Err(Error::config(format!(
            "Laguna rotary dimension {rotary_dim} must be positive, even, and at most head_dim {head_dim}"
        )));
    }
    Ok(rotary_dim)
}

fn validate_quantization(config: &LagunaConfig, contract: LagunaProfileContract) -> Result<()> {
    if !contract.requires_compressed_tensors_config {
        if config.quantization_config.is_some() {
            return Err(Error::config(format!(
                "{} config must not declare the Laguna S compressed-tensors INT4 contract; GGUF quantization is validated from the artifact",
                contract.display_name
            )));
        }
        return Ok(());
    }

    let quant = config.quantization_config.as_ref().ok_or_else(|| {
        Error::config(format!(
            "{} config is missing quantization_config",
            contract.display_name
        ))
    })?;
    if quant.format != "pack-quantized"
        || quant.quant_method != "compressed-tensors"
        || quant.quantization_status != "compressed"
    {
        return Err(Error::config(
            "Laguna requires compressed-tensors pack-quantized weights",
        ));
    }
    if quant.config_groups.len() != 1 {
        return Err(Error::config(
            "Laguna requires exactly one INT4 quantization group",
        ));
    }
    let group = quant.config_groups.get("group_0").ok_or_else(|| {
        Error::config("Laguna quantization config is missing config_groups.group_0")
    })?;
    if group.format != "pack-quantized"
        || group.targets != ["re:.*layers\\.\\d+\\..*(w[1-3]|gate_proj|up_proj|down_proj)$"]
        || group.weights.kind != "int"
        || group.weights.num_bits != 4
        || group.weights.group_size != Some(SUPPORTED_INT4_GROUP_SIZE)
        || group.weights.strategy != "group"
        || !group.weights.symmetric
        || group.weights.dynamic
    {
        return Err(Error::config(
            "Laguna routed experts must use symmetric static INT4 groups of 32",
        ));
    }
    let kv = &quant.kv_cache_scheme;
    if kv.kind != "float"
        || kv.num_bits != 8
        || kv.group_size.is_some()
        || kv.strategy != "tensor"
        || !kv.symmetric
        || kv.dynamic
    {
        return Err(Error::config(
            "Laguna KV cache must use static symmetric tensor-wise FP8",
        ));
    }
    if quant.transform_config.config_groups.len() != 1 {
        return Err(Error::config(
            "Laguna requires exactly one offline transform group",
        ));
    }
    let transform = quant
        .transform_config
        .config_groups
        .get("R1")
        .ok_or_else(|| Error::config("Laguna transform config is missing R1"))?;
    if transform.kind != "hadamard"
        || transform.head_dim != SUPPORTED_HEAD_DIM
        || transform.randomize
        || transform.apply.len() != 2
    {
        return Err(Error::config(
            "Laguna requires deterministic offline 128-wide Hadamard weight transforms",
        ));
    }
    let output = &transform.apply[0];
    let input = &transform.apply[1];
    if output.inverse
        || output.location != "weight_output"
        || output.targets != ["re:.*embed_tokens$", "re:.*o_proj$", "re:.*down_proj$"]
        || !input.inverse
        || input.location != "weight_input"
        || input.targets
            != [
                "re:.*q_proj$",
                "re:.*k_proj$",
                "re:.*v_proj$",
                "re:.*gate_proj$",
                "re:.*up_proj$",
                "re:.*mlp.gate$",
                "re:.*g_proj$",
                "re:.*lm_head$",
            ]
    {
        return Err(Error::config(
            "Laguna R1 must contain the checkpoint's exact offline output/input transform pairs",
        ));
    }
    Ok(())
}

fn require_equal(name: &str, actual: &str, expected: &str) -> Result<()> {
    if actual != expected {
        return Err(Error::config(format!(
            "Laguna {name} must be {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

fn require_usize(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(Error::config(format!(
            "Laguna {name} must be {expected}, got {actual}"
        )));
    }
    Ok(())
}

fn checked_u64(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| Error::config("Laguna dimension does not fit u64"))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn parses_exact_laguna_s_2_1_int4_contract() {
        let config = LagunaConfig::from_json_str(&supported_config_json()).unwrap();

        assert_eq!(config.profile().unwrap(), LagunaProfile::S21);
        assert_eq!(config.query_width(0), Some(6_144));
        assert_eq!(config.query_width(1), Some(9_216));
        assert_eq!(config.key_value_width(), 1_024);
        assert_eq!(config.rotary_dim(0), Some(64));
        assert_eq!(config.rotary_dim(1), Some(128));
    }

    #[test]
    fn parses_exact_laguna_xs_2_1_contract() {
        let config = LagunaConfig::from_json_str(&xs_config_json()).unwrap();

        assert_eq!(config.profile().unwrap(), LagunaProfile::Xs21);
        assert_eq!(config.hidden_size, 2_048);
        assert_eq!(config.num_hidden_layers, 40);
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.query_width(0), Some(6_144));
        assert_eq!(config.query_width(1), Some(8_192));
        assert_eq!(config.key_value_width(), 1_024);
        assert_eq!(config.rotary_dim(0), Some(64));
        assert_eq!(config.rotary_dim(1), Some(128));
        assert!(config.quantization_config.is_none());
    }

    #[test]
    fn laguna_xs_sliding_window_bounds_three_quarters_of_the_kv_layers() {
        let config = LagunaConfig::from_json_str(&xs_config_json()).unwrap();
        let budget = config.fp8_kv_cache_budget(1, 262_144).unwrap();

        assert_eq!(budget.full_layer_count, 10);
        assert_eq!(budget.sliding_layer_count, 30);
        assert_eq!(budget.full_retained_tokens_per_layer, 262_144);
        assert_eq!(budget.sliding_retained_tokens_per_layer, 512);
        assert_eq!(budget.full_bytes, 5_368_709_120);
        assert_eq!(budget.sliding_bytes, 31_457_280);
        assert_eq!(budget.total_bytes, 5_400_166_400);
    }

    #[test]
    fn rejects_a_hybrid_laguna_xs_contract() {
        let mut value: serde_json::Value = serde_json::from_str(&xs_config_json()).unwrap();
        value["num_experts_per_tok"] = json!(10);

        let error = LagunaConfig::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("num_experts_per_tok must be 8"));
    }

    #[test]
    fn sliding_window_bounds_three_quarters_of_the_kv_layers() {
        let config = LagunaConfig::from_json_str(&supported_config_json()).unwrap();
        let budget = config.fp8_kv_cache_budget(1, 262_144).unwrap();

        assert_eq!(budget.full_layer_count, 12);
        assert_eq!(budget.sliding_layer_count, 36);
        assert_eq!(budget.full_retained_tokens_per_layer, 262_144);
        assert_eq!(budget.sliding_retained_tokens_per_layer, 512);
        assert_eq!(budget.full_bytes, 6_442_450_944);
        assert_eq!(budget.sliding_bytes, 37_748_736);
        assert_eq!(budget.total_bytes, 6_480_199_680);
    }

    #[test]
    fn builds_distinct_global_yarn_and_sliding_default_frequencies() {
        let config = LagunaConfig::from_json_str(&supported_config_json()).unwrap();
        let global = config.rope_frequencies(0).unwrap();
        let sliding = config.rope_frequencies(1).unwrap();

        assert_eq!(global.rotary_dim, 64);
        assert_eq!(global.inverse_frequencies.len(), 32);
        assert_eq!(global.attention_factor, 1.346_573_6);
        assert_eq!(sliding.rotary_dim, 128);
        assert_eq!(sliding.inverse_frequencies.len(), 64);
        assert_eq!(sliding.attention_factor, 1.0);
        assert_eq!(global.inverse_frequencies[0], 1.0);
        assert_eq!(sliding.inverse_frequencies[0], 1.0);
        assert_ne!(
            global.inverse_frequencies[20],
            sliding.inverse_frequencies[20]
        );
        assert!(global
            .inverse_frequencies
            .windows(2)
            .all(|pair| pair[0] >= pair[1]));
        assert!(sliding
            .inverse_frequencies
            .windows(2)
            .all(|pair| pair[0] > pair[1]));
    }

    #[test]
    fn rejects_a_changed_top_k_instead_of_silently_changing_model_semantics() {
        let mut value: serde_json::Value = serde_json::from_str(&supported_config_json()).unwrap();
        value["num_experts_per_tok"] = json!(8);

        let error = LagunaConfig::from_json_str(&value.to_string()).unwrap_err();
        assert!(error.to_string().contains("num_experts_per_tok must be 10"));
    }

    #[test]
    fn rejects_changed_checkpoint_dimensions() {
        for (field, changed, expected) in [
            ("vocab_size", json!(100_000), "vocab_size must be 100352"),
            (
                "intermediate_size",
                json!(8_192),
                "intermediate_size must be 12288",
            ),
            (
                "max_position_embeddings",
                json!(131_072),
                "max_position_embeddings must be 262144",
            ),
        ] {
            let mut value: serde_json::Value =
                serde_json::from_str(&supported_config_json()).unwrap();
            value[field] = changed;

            let error = LagunaConfig::from_json_str(&value.to_string()).unwrap_err();
            assert!(error.to_string().contains(expected));
        }
    }

    fn supported_config_json() -> String {
        let layer_types = (0..48)
            .map(|layer| {
                if layer % 4 == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect::<Vec<_>>();
        let heads = (0..48)
            .map(|layer| if layer % 4 == 0 { 48 } else { 72 })
            .collect::<Vec<_>>();
        let mlp = (0..48)
            .map(|layer| if layer == 0 { "dense" } else { "sparse" })
            .collect::<Vec<_>>();
        let rope_parameters = json!({
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "partial_rotary_factor": 0.5,
                "factor": 32.0,
                "original_max_position_embeddings": 8192,
                "beta_slow": 1.0,
                "beta_fast": 32.0,
                "attention_factor": 1.3465735902799727
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0
            }
        });
        let weight_quantization = json!({
            "dynamic": false,
            "group_size": 32,
            "num_bits": 4,
            "strategy": "group",
            "symmetric": true,
            "type": "int"
        });
        let kv_quantization = json!({
            "dynamic": false,
            "group_size": null,
            "num_bits": 8,
            "strategy": "tensor",
            "symmetric": true,
            "type": "float"
        });
        let transform_config = json!({
            "config_groups": {
                "R1": {
                    "apply": [
                        {
                            "inverse": false,
                            "location": "weight_output",
                            "targets": ["re:.*embed_tokens$", "re:.*o_proj$", "re:.*down_proj$"]
                        },
                        {
                            "inverse": true,
                            "location": "weight_input",
                            "targets": [
                                "re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$",
                                "re:.*gate_proj$", "re:.*up_proj$", "re:.*mlp.gate$",
                                "re:.*g_proj$", "re:.*lm_head$"
                            ]
                        }
                    ],
                    "head_dim": 128,
                    "randomize": false,
                    "type": "hadamard"
                }
            }
        });
        let quantization_config = json!({
            "config_groups": {
                "group_0": {
                    "format": "pack-quantized",
                    "targets": ["re:.*layers\\.\\d+\\..*(w[1-3]|gate_proj|up_proj|down_proj)$"],
                    "weights": weight_quantization
                }
            },
            "format": "pack-quantized",
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "kv_cache_scheme": kv_quantization,
            "transform_config": transform_config
        });
        json!({
            "architectures": ["LagunaForCausalLM"],
            "model_type": "laguna",
            "vocab_size": 100352,
            "hidden_size": 3072,
            "intermediate_size": 12288,
            "num_hidden_layers": 48,
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "num_attention_heads_per_layer": heads,
            "head_dim": 128,
            "max_position_embeddings": 262144,
            "qkv_bias": false,
            "attention_bias": false,
            "attention_dropout": 0.0,
            "rms_norm_eps": 0.000001,
            "hidden_act": "silu",
            "num_experts": 256,
            "num_experts_per_tok": 10,
            "moe_intermediate_size": 1024,
            "shared_expert_intermediate_size": 1024,
            "norm_topk_prob": true,
            "moe_routed_scaling_factor": 2.5,
            "moe_apply_router_weight_on_input": false,
            "moe_router_logit_softcapping": 0.0,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [0],
            "sliding_window": 512,
            "swa_attention_sink_enabled": false,
            "layer_types": layer_types,
            "mlp_layer_types": mlp,
            "gating": "per-head",
            "gating_types": vec!["per_head"; 48],
            "bos_token_id": 2,
            "eos_token_id": [2, 24],
            "pad_token_id": 9,
            "tie_word_embeddings": false,
            "use_cache": true,
            "torch_dtype": "bfloat16",
            "rope_parameters": rope_parameters,
            "quantization_config": quantization_config
        })
        .to_string()
    }

    fn xs_config_json() -> String {
        let mut value: serde_json::Value = serde_json::from_str(&supported_config_json()).unwrap();
        let layer_types = (0..40)
            .map(|layer| {
                if layer % 4 == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect::<Vec<_>>();
        let heads = (0..40)
            .map(|layer| if layer % 4 == 0 { 48 } else { 64 })
            .collect::<Vec<_>>();
        let mlp = (0..40)
            .map(|layer| if layer == 0 { "dense" } else { "sparse" })
            .collect::<Vec<_>>();

        value["hidden_size"] = json!(2_048);
        value["intermediate_size"] = json!(8_192);
        value["moe_intermediate_size"] = json!(512);
        value["shared_expert_intermediate_size"] = json!(512);
        value["num_hidden_layers"] = json!(40);
        value["num_experts_per_tok"] = json!(8);
        value["layer_types"] = json!(layer_types);
        value["num_attention_heads_per_layer"] = json!(heads);
        value["mlp_layer_types"] = json!(mlp);
        value["gating_types"] = json!(vec!["per_head"; 40]);
        value["rope_parameters"]["full_attention"]["beta_fast"] = json!(64.0);
        value
            .as_object_mut()
            .expect("test config must be an object")
            .remove("quantization_config");
        value
            .as_object_mut()
            .expect("test config must be an object")
            .remove("moe_router_logit_softcapping");
        value.to_string()
    }
}
