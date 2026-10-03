use std::{path::Path, sync::Arc};

use common::{Error, Result};
use config::DFlashConfig;
use inferno_io::{SafeTensorDtype, SafeTensorHandle, SafeTensorShard};

use super::weights::validate_tensor_info;

pub const DFLASH_MODEL_FILE: &str = "model.safetensors";
pub const DFLASH_TENSOR_COUNT: usize = 81;

#[derive(Debug, Clone)]
pub struct DFlashConvWeights {
    pub base_kernel: SafeTensorHandle,
    pub kernel_projection: SafeTensorHandle,
}

#[derive(Debug, Clone)]
pub struct DFlashAttentionWeights {
    pub query: SafeTensorHandle,
    pub key: SafeTensorHandle,
    pub value: SafeTensorHandle,
    pub output: SafeTensorHandle,
    pub query_norm: SafeTensorHandle,
    pub key_norm: SafeTensorHandle,
}

#[derive(Debug, Clone)]
pub struct DFlashMlpWeights {
    pub gate: SafeTensorHandle,
    pub up: SafeTensorHandle,
    pub down: SafeTensorHandle,
}

#[derive(Debug, Clone)]
pub struct DFlashLayerWeights {
    pub layer_index: usize,
    pub input_norm: SafeTensorHandle,
    pub attention_conv: DFlashConvWeights,
    pub attention: DFlashAttentionWeights,
    pub post_attention_norm: SafeTensorHandle,
    pub mlp_conv: DFlashConvWeights,
    pub mlp: DFlashMlpWeights,
}

#[derive(Debug, Clone)]
pub struct DFlashWeights {
    pub selector_hidden_projection: SafeTensorHandle,
    pub predecessor_codebook: SafeTensorHandle,
    pub successor_codebook: SafeTensorHandle,
    pub target_fusion: SafeTensorHandle,
    pub target_hidden_norm: SafeTensorHandle,
    pub layers: Vec<DFlashLayerWeights>,
    pub final_norm: SafeTensorHandle,
}

#[derive(Debug)]
pub struct DFlashWeightIndex {
    _storage: Arc<SafeTensorShard>,
    pub weights: DFlashWeights,
}

impl DFlashWeightIndex {
    pub fn open(model_dir: impl AsRef<Path>, config: &DFlashConfig) -> Result<Self> {
        let shard = Arc::new(SafeTensorShard::open(
            model_dir.as_ref().join(DFLASH_MODEL_FILE),
        )?);
        if shard.tensor_count() != DFLASH_TENSOR_COUNT {
            return Err(Error::weights(format!(
                "DFlash2 shard must contain {DFLASH_TENSOR_COUNT} tensors, got {}",
                shard.tensor_count()
            )));
        }

        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let query_width = config.query_width();
        let kv_width = config.kv_width();
        let conv_projection_width = 2
            * config.dflash_config.conv_kernel_size
            * (hidden / config.dflash_config.conv_group_size);
        let selector_rank = config.dflash_config.selector_rank;

        let layers = (0..config.num_hidden_layers)
            .map(|layer_index| {
                let prefix = format!("layers.{layer_index}");
                Ok(DFlashLayerWeights {
                    layer_index,
                    input_norm: required(
                        &shard,
                        &format!("{prefix}.input_layernorm.weight"),
                        &[hidden],
                    )?,
                    attention_conv: load_conv(
                        &shard,
                        &format!("{prefix}.attention_conv"),
                        hidden,
                        conv_projection_width,
                        config.dflash_config.conv_kernel_size,
                    )?,
                    attention: DFlashAttentionWeights {
                        query: required(
                            &shard,
                            &format!("{prefix}.self_attn.q_proj.weight"),
                            &[query_width, hidden],
                        )?,
                        key: required(
                            &shard,
                            &format!("{prefix}.self_attn.k_proj.weight"),
                            &[kv_width, hidden],
                        )?,
                        value: required(
                            &shard,
                            &format!("{prefix}.self_attn.v_proj.weight"),
                            &[kv_width, hidden],
                        )?,
                        output: required(
                            &shard,
                            &format!("{prefix}.self_attn.o_proj.weight"),
                            &[hidden, query_width],
                        )?,
                        query_norm: required(
                            &shard,
                            &format!("{prefix}.self_attn.q_norm.weight"),
                            &[config.head_dim],
                        )?,
                        key_norm: required(
                            &shard,
                            &format!("{prefix}.self_attn.k_norm.weight"),
                            &[config.head_dim],
                        )?,
                    },
                    post_attention_norm: required(
                        &shard,
                        &format!("{prefix}.post_attention_layernorm.weight"),
                        &[hidden],
                    )?,
                    mlp_conv: load_conv(
                        &shard,
                        &format!("{prefix}.mlp_conv"),
                        hidden,
                        conv_projection_width,
                        config.dflash_config.conv_kernel_size,
                    )?,
                    mlp: DFlashMlpWeights {
                        gate: required(
                            &shard,
                            &format!("{prefix}.mlp.gate_proj.weight"),
                            &[intermediate, hidden],
                        )?,
                        up: required(
                            &shard,
                            &format!("{prefix}.mlp.up_proj.weight"),
                            &[intermediate, hidden],
                        )?,
                        down: required(
                            &shard,
                            &format!("{prefix}.mlp.down_proj.weight"),
                            &[hidden, intermediate],
                        )?,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let weights = DFlashWeights {
            selector_hidden_projection: required(
                &shard,
                "candidate_selector.hidden_projection.weight",
                &[selector_rank, hidden],
            )?,
            predecessor_codebook: required(
                &shard,
                "candidate_selector.predecessor_codebook",
                &[config.vocab_size, selector_rank],
            )?,
            successor_codebook: required(
                &shard,
                "candidate_selector.successor_codebook",
                &[config.vocab_size, selector_rank],
            )?,
            target_fusion: required(
                &shard,
                "fc.weight",
                &[hidden, config.target_feature_width()?],
            )?,
            target_hidden_norm: required(&shard, "hidden_norm.weight", &[hidden])?,
            layers,
            final_norm: required(&shard, "norm.weight", &[hidden])?,
        };

        Ok(Self {
            _storage: shard,
            weights,
        })
    }
}

fn load_conv(
    shard: &Arc<SafeTensorShard>,
    prefix: &str,
    hidden: usize,
    projection_width: usize,
    kernel_size: usize,
) -> Result<DFlashConvWeights> {
    Ok(DFlashConvWeights {
        base_kernel: required(
            shard,
            &format!("{prefix}.base_kernel"),
            &[2, kernel_size, hidden],
        )?,
        kernel_projection: required(
            shard,
            &format!("{prefix}.kernel_projection.weight"),
            &[projection_width, hidden],
        )?,
    })
}

fn required(shard: &Arc<SafeTensorShard>, name: &str, shape: &[usize]) -> Result<SafeTensorHandle> {
    let tensor = shard.tensor(name)?;
    validate_tensor_info(tensor.info(), SafeTensorDtype::Bf16, shape)?;
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_inventory_count_matches_structured_layout() {
        let root_tensors = 6;
        let tensors_per_layer = 15;
        assert_eq!(root_tensors + 5 * tensors_per_layer, DFLASH_TENSOR_COUNT);
    }
}
