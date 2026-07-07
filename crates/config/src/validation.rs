use common::{Error, Result};

use crate::Config;

pub fn validate_config(config: &Config) -> Result<()> {
    require_positive("hidden_size", config.hidden_size)?;
    require_positive("num_layers", config.num_layers)?;
    require_positive("vocab_size", config.vocab_size)?;
    require_positive("attention_heads", config.attention_heads)?;
    require_positive("qk_head_dim", config.qk_head_dim)?;
    require_positive("qk_no_rope_dim", config.qk_no_rope_dim)?;
    require_positive("qk_rope_dim", config.qk_rope_dim)?;
    require_positive("kv_lora_rank", config.kv_lora_rank)?;
    require_positive("v_head_dim", config.v_head_dim())?;
    require_positive("num_routed_experts", config.num_routed_experts)?;
    require_positive("experts_per_token", config.experts_per_token)?;
    require_positive("moe_intermediate_size", config.moe_intermediate_size)?;
    require_positive("num_shared_experts", config.num_shared_experts)?;
    require_positive("moe_groups", config.moe_groups)?;
    require_positive("topk_group", config.topk_group)?;
    require_positive("max_context", config.max_context)?;
    require_positive("dsa_index_topk", config.dsa_index_topk)?;
    require_positive("index_head_dim", config.index_head_dim)?;
    require_positive("index_n_heads", config.index_n_heads)?;
    require_positive("index_topk_freq", config.index_topk_freq)?;

    if config.qk_no_rope_dim + config.qk_rope_dim != config.qk_head_dim {
        return Err(Error::config(format!(
            "qk_no_rope_dim + qk_rope_dim must equal qk_head_dim: {} + {} != {}",
            config.qk_no_rope_dim, config.qk_rope_dim, config.qk_head_dim
        )));
    }

    if config.dense_layers + config.sparse_moe_layers() != config.num_layers {
        return Err(Error::config(format!(
            "dense_layers + sparse_moe_layers must equal num_layers: {} + {} != {}",
            config.dense_layers,
            config.sparse_moe_layers(),
            config.num_layers
        )));
    }

    if config.experts_per_token > config.num_routed_experts {
        return Err(Error::config(format!(
            "experts_per_token {} exceeds num_routed_experts {}",
            config.experts_per_token, config.num_routed_experts
        )));
    }

    if config.topk_group > config.moe_groups {
        return Err(Error::config(format!(
            "topk_group {} exceeds moe_groups {}",
            config.topk_group, config.moe_groups
        )));
    }

    if config.dsa_index_topk > config.max_context {
        return Err(Error::config(format!(
            "dsa_index_topk {} exceeds max_context {}",
            config.dsa_index_topk, config.max_context
        )));
    }

    if config.qk_rope_dim > config.index_head_dim {
        return Err(Error::config(format!(
            "qk_rope_dim {} exceeds index_head_dim {}",
            config.qk_rope_dim, config.index_head_dim
        )));
    }

    if !config.indexer_types.is_empty() && config.indexer_types.len() != config.num_layers {
        return Err(Error::config(format!(
            "indexer_types length {} must equal num_layers {}",
            config.indexer_types.len(),
            config.num_layers
        )));
    }

    if config.num_nextn_predict_layers > 1 {
        return Err(Error::config(format!(
            "num_nextn_predict_layers {} is unsupported; Inferno targets the GLM-5.2 Q2 single-MTP-head artifact",
            config.num_nextn_predict_layers
        )));
    }

    if config.rms_norm_eps <= 0.0 {
        return Err(Error::config("rms_norm_eps must be positive"));
    }

    if config.rope_theta <= 0.0 {
        return Err(Error::config("rope_theta must be positive"));
    }

    if !config.routed_scaling_factor.is_finite() || config.routed_scaling_factor <= 0.0 {
        return Err(Error::config(
            "routed_scaling_factor must be positive and finite",
        ));
    }

    if !matches!(config.scoring_func.as_str(), "softmax" | "sigmoid") {
        return Err(Error::config(format!(
            "unsupported scoring_func {}; expected softmax or sigmoid",
            config.scoring_func
        )));
    }

    if !matches!(config.topk_method.as_str(), "greedy" | "noaux_tc") {
        return Err(Error::config(format!(
            "unsupported topk_method {}; expected greedy or noaux_tc",
            config.topk_method
        )));
    }

    Ok(())
}

fn require_positive(name: &str, value: usize) -> Result<()> {
    if value == 0 {
        return Err(Error::config(format!("{name} must be positive")));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::GLM52_LIKE_CONFIG_JSON;

    #[test]
    fn rejects_invalid_qk_split() {
        let mut config = Config::from_json_str(GLM52_LIKE_CONFIG_JSON).unwrap();
        config.qk_rope_dim = 32;

        let err = validate_config(&config).expect_err("invalid split should fail");
        assert!(err.to_string().contains("qk_no_rope_dim"));
    }

    #[test]
    fn rejects_topk_larger_than_expert_count() {
        let mut config = Config::from_json_str(GLM52_LIKE_CONFIG_JSON).unwrap();
        config.experts_per_token = 257;

        let err = validate_config(&config).expect_err("invalid top-k should fail");
        assert!(err.to_string().contains("experts_per_token"));
    }

    #[test]
    fn rejects_unknown_moe_scoring_function() {
        let mut config = Config::from_json_str(GLM52_LIKE_CONFIG_JSON).unwrap();
        config.scoring_func = "mystery".to_string();

        let err = validate_config(&config).expect_err("invalid scoring function should fail");
        assert!(err.to_string().contains("scoring_func"));
    }

    #[test]
    fn rejects_multiple_nextn_predict_layers() {
        let mut config = Config::from_json_str(GLM52_LIKE_CONFIG_JSON).unwrap();
        config.num_nextn_predict_layers = 2;

        let err = validate_config(&config).expect_err("multiple MTP heads should fail");
        assert!(err.to_string().contains("num_nextn_predict_layers"));
    }
}
