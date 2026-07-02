use common::{validate_exact_shape, Result, Shape};

use crate::Config;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchitectureSummary {
    pub model_type: String,
    pub hidden_size: usize,
    pub layers: usize,
    pub dense_layers: usize,
    pub sparse_moe_layers: usize,
    pub vocab_size: usize,
    pub attention_heads: usize,
    pub qk_head_dim: usize,
    pub qk_no_rope_dim: usize,
    pub qk_rope_dim: usize,
    pub v_head_dim: usize,
    pub merged_attention_width: usize,
    pub routed_experts: usize,
    pub experts_per_token: usize,
    pub moe_intermediate_size: usize,
    pub shared_experts: usize,
    pub moe_groups: usize,
    pub topk_group: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: String,
    pub scoring_func: String,
    pub topk_method: String,
    pub max_context: usize,
    pub dsa_index_topk: usize,
}

impl ArchitectureSummary {
    pub fn from_config(config: &Config) -> Self {
        Self {
            model_type: config.model_type.clone(),
            hidden_size: config.hidden_size,
            layers: config.num_layers,
            dense_layers: config.dense_layers,
            sparse_moe_layers: config.sparse_moe_layers(),
            vocab_size: config.vocab_size,
            attention_heads: config.attention_heads,
            qk_head_dim: config.qk_head_dim,
            qk_no_rope_dim: config.qk_no_rope_dim,
            qk_rope_dim: config.qk_rope_dim,
            v_head_dim: config.v_head_dim(),
            merged_attention_width: config.merged_attention_width(),
            routed_experts: config.num_routed_experts,
            experts_per_token: config.experts_per_token,
            moe_intermediate_size: config.moe_intermediate_size,
            shared_experts: config.num_shared_experts,
            moe_groups: config.moe_groups,
            topk_group: config.topk_group,
            norm_topk_prob: config.norm_topk_prob,
            routed_scaling_factor: config.routed_scaling_factor.to_string(),
            scoring_func: config.scoring_func.clone(),
            topk_method: config.topk_method.clone(),
            max_context: config.max_context,
            dsa_index_topk: config.dsa_index_topk,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeSummary {
    pub batch: usize,
    pub tokens: usize,
    pub input_ids: Shape,
    pub hidden_states: Shape,
    pub q_heads: Shape,
    pub k_heads: Shape,
    pub v_heads: Shape,
    pub q_no_rope: Shape,
    pub q_rope: Shape,
    pub k_no_rope: Shape,
    pub k_rope: Shape,
    pub merged_attention_output: Shape,
    pub projected_output: Shape,
    pub flat_tokens: Shape,
    pub moe_router_logits: Shape,
    pub moe_topk_ids: Shape,
    pub moe_topk_weights: Shape,
    pub k_cache: Shape,
    pub v_cache: Shape,
    pub logits: Shape,
}

impl ShapeSummary {
    pub fn from_config(config: &Config, batch: usize, tokens: usize) -> Result<Self> {
        if batch == 0 {
            return Err(common::Error::config("batch must be positive"));
        }
        if tokens == 0 {
            return Err(common::Error::config("tokens must be positive"));
        }
        if tokens > config.max_context {
            return Err(common::Error::config(format!(
                "tokens {tokens} exceeds max_context {}",
                config.max_context
            )));
        }

        let flat_token_count = batch * tokens;
        let summary = Self {
            batch,
            tokens,
            input_ids: Shape::new(vec![batch, tokens]),
            hidden_states: Shape::new(vec![batch, tokens, config.hidden_size]),
            q_heads: Shape::new(vec![
                batch,
                tokens,
                config.attention_heads,
                config.qk_head_dim,
            ]),
            k_heads: Shape::new(vec![
                batch,
                tokens,
                config.attention_heads,
                config.qk_head_dim,
            ]),
            v_heads: Shape::new(vec![
                batch,
                tokens,
                config.attention_heads,
                config.v_head_dim(),
            ]),
            q_no_rope: Shape::new(vec![
                batch,
                tokens,
                config.attention_heads,
                config.qk_no_rope_dim,
            ]),
            q_rope: Shape::new(vec![
                batch,
                tokens,
                config.attention_heads,
                config.qk_rope_dim,
            ]),
            k_no_rope: Shape::new(vec![
                batch,
                tokens,
                config.attention_heads,
                config.qk_no_rope_dim,
            ]),
            k_rope: Shape::new(vec![
                batch,
                tokens,
                config.attention_heads,
                config.qk_rope_dim,
            ]),
            merged_attention_output: Shape::new(vec![
                batch,
                tokens,
                config.merged_attention_width(),
            ]),
            projected_output: Shape::new(vec![batch, tokens, config.hidden_size]),
            flat_tokens: Shape::new(vec![flat_token_count, config.hidden_size]),
            moe_router_logits: Shape::new(vec![flat_token_count, config.num_routed_experts]),
            moe_topk_ids: Shape::new(vec![flat_token_count, config.experts_per_token]),
            moe_topk_weights: Shape::new(vec![flat_token_count, config.experts_per_token]),
            k_cache: Shape::new(vec![
                batch,
                config.attention_heads,
                tokens,
                config.qk_head_dim,
            ]),
            v_cache: Shape::new(vec![
                batch,
                config.attention_heads,
                tokens,
                config.v_head_dim(),
            ]),
            logits: Shape::new(vec![batch, tokens, config.vocab_size]),
        };

        validate_exact_shape(
            "q_split",
            &[config.qk_no_rope_dim + config.qk_rope_dim],
            &[config.qk_head_dim],
        )?;

        Ok(summary)
    }
}

#[cfg(test)]
mod tests {
    use crate::load_embedded_config;

    #[test]
    fn shape_summary_matches_glm52_q2_shapes() {
        let config = load_embedded_config().unwrap();
        let shapes = config.shape_summary(1, 4).unwrap();

        assert_eq!(shapes.input_ids.dims(), &[1, 4]);
        assert_eq!(shapes.hidden_states.dims(), &[1, 4, 6144]);
        assert_eq!(shapes.q_heads.dims(), &[1, 4, 64, 256]);
        assert_eq!(shapes.q_no_rope.dims(), &[1, 4, 64, 192]);
        assert_eq!(shapes.q_rope.dims(), &[1, 4, 64, 64]);
        assert_eq!(shapes.k_cache.dims(), &[1, 64, 4, 256]);
        assert_eq!(shapes.v_cache.dims(), &[1, 64, 4, 256]);
        assert_eq!(shapes.moe_router_logits.dims(), &[4, 256]);
        assert_eq!(shapes.moe_topk_ids.dims(), &[4, 8]);
        assert_eq!(shapes.logits.dims(), &[1, 4, 154880]);
    }
}
