use std::{path::Path, sync::Arc};

use common::{Error, Result};
use config::{LagunaConfig, LagunaMlpKind};
use inferno_io::{SafeTensorDtype, SafeTensorHandle, SafeTensorIndex, SafeTensorModel};

pub const LAGUNA_INT4_REPO_ID: &str = "poolside/Laguna-S-2.1-INT4";
pub const LAGUNA_INT4_REPO_URL: &str = "https://huggingface.co/poolside/Laguna-S-2.1-INT4";
pub const LAGUNA_INT4_TOTAL_BYTES: u64 = 71_898_444_992;
const LAGUNA_INT4_TENSOR_COUNT: usize = 72_961;
const ROUTED_LAYER_COUNT: usize = 47;
const PACKED_VALUES_PER_I32: usize = 8;
const INT4_GROUP_SIZE: usize = 32;

#[derive(Debug, Clone)]
pub struct LagunaRootWeights {
    pub embedding: SafeTensorHandle,
    pub final_norm: SafeTensorHandle,
    pub output: SafeTensorHandle,
}

#[derive(Debug, Clone)]
pub struct LagunaAttentionWeights {
    pub query: SafeTensorHandle,
    pub key: SafeTensorHandle,
    pub value: SafeTensorHandle,
    pub output: SafeTensorHandle,
    pub gate: SafeTensorHandle,
    pub query_norm: SafeTensorHandle,
    pub key_norm: SafeTensorHandle,
    pub key_scale: SafeTensorHandle,
    pub value_scale: SafeTensorHandle,
}

#[derive(Debug, Clone)]
pub struct LagunaDenseWeights {
    pub gate: SafeTensorHandle,
    pub up: SafeTensorHandle,
    pub down: SafeTensorHandle,
}

#[derive(Debug, Clone)]
pub struct LagunaMoeWeights {
    pub router: SafeTensorHandle,
    pub correction_bias: SafeTensorHandle,
    pub shared: LagunaDenseWeights,
}

#[derive(Debug, Clone)]
pub enum LagunaLayerMlpWeights {
    Dense(LagunaDenseWeights),
    Moe(LagunaMoeWeights),
}

#[derive(Debug, Clone)]
pub struct LagunaLayerWeights {
    pub layer_index: usize,
    pub input_norm: SafeTensorHandle,
    pub post_attention_norm: SafeTensorHandle,
    pub attention: LagunaAttentionWeights,
    pub mlp: LagunaLayerMlpWeights,
}

#[derive(Debug, Clone)]
pub struct LagunaExpertWeights {
    pub layer_index: usize,
    pub expert_id: usize,
    pub gate_packed: SafeTensorHandle,
    pub gate_scales: SafeTensorHandle,
    pub up_packed: SafeTensorHandle,
    pub up_scales: SafeTensorHandle,
    pub down_packed: SafeTensorHandle,
    pub down_scales: SafeTensorHandle,
}

impl LagunaExpertWeights {
    pub fn storage_bytes(&self) -> Result<u64> {
        [
            &self.gate_packed,
            &self.gate_scales,
            &self.up_packed,
            &self.up_scales,
            &self.down_packed,
            &self.down_scales,
        ]
        .into_iter()
        .try_fold(0_u64, |total, tensor| {
            total
                .checked_add(tensor.info().byte_len)
                .ok_or_else(|| Error::weights("Laguna expert storage byte count overflow"))
        })
    }

    pub fn advise_random(&self) -> Result<()> {
        self.gate_packed.advise_random()?;
        self.gate_scales.advise_random()?;
        self.up_packed.advise_random()?;
        self.up_scales.advise_random()?;
        self.down_packed.advise_random()?;
        self.down_scales.advise_random()?;
        Ok(())
    }

    pub fn prefetch(&self) -> Result<()> {
        SafeTensorHandle::prefetch_together(&[
            &self.gate_packed,
            &self.gate_scales,
            &self.up_packed,
            &self.up_scales,
            &self.down_packed,
            &self.down_scales,
        ])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LagunaWeightSummary {
    pub tensor_count: usize,
    pub shard_count: usize,
    pub total_bytes: u64,
    pub fixed_weight_bytes: u64,
    pub routed_expert_bytes: u64,
    pub routed_layer_count: usize,
    pub experts_per_layer: usize,
    pub bytes_per_expert: u64,
}

#[derive(Debug)]
pub struct LagunaWeightIndex {
    experts: Vec<LagunaExpertWeights>,
    pub root: LagunaRootWeights,
    pub layers: Vec<LagunaLayerWeights>,
    pub summary: LagunaWeightSummary,
}

impl LagunaWeightIndex {
    /// Opens all shard headers and validates every tensor shape. Tensor payloads
    /// remain memory-mapped and are not paged into RAM by this operation.
    pub fn open(model_dir: impl AsRef<Path>, config: &LagunaConfig) -> Result<Self> {
        let storage = Arc::new(SafeTensorModel::open(model_dir)?);
        validate_index_contract(storage.index(), config)?;

        let root = LagunaRootWeights {
            embedding: required(
                &storage,
                "model.embed_tokens.weight",
                SafeTensorDtype::Bf16,
                &[config.vocab_size, config.hidden_size],
            )?,
            final_norm: required(
                &storage,
                "model.norm.weight",
                SafeTensorDtype::Bf16,
                &[config.hidden_size],
            )?,
            output: required(
                &storage,
                "lm_head.weight",
                SafeTensorDtype::Bf16,
                &[config.vocab_size, config.hidden_size],
            )?,
        };

        let layers = (0..config.num_hidden_layers)
            .map(|layer_index| load_layer(&storage, config, layer_index))
            .collect::<Result<Vec<_>>>()?;

        // Shape validation happens before inference. Opening all expert handles
        // reads only safetensors headers; it does not touch their 63.87 GB payload.
        let expert_count = ROUTED_LAYER_COUNT
            .checked_mul(config.num_experts)
            .ok_or_else(|| Error::weights("Laguna expert directory length overflow"))?;
        let mut experts = Vec::with_capacity(expert_count);
        let mut routed_expert_bytes = 0_u64;
        for layer_index in 1..config.num_hidden_layers {
            for expert_id in 0..config.num_experts {
                let expert = load_expert(&storage, config, layer_index, expert_id)?;
                routed_expert_bytes = routed_expert_bytes
                    .checked_add(expert.storage_bytes()?)
                    .ok_or_else(|| Error::weights("Laguna routed expert byte count overflow"))?;
                experts.push(expert);
            }
        }
        if experts.len() != expert_count {
            return Err(Error::weights(format!(
                "Laguna expert directory must contain {expert_count} entries, got {}",
                experts.len()
            )));
        }
        let fixed_weight_bytes = storage
            .index()
            .total_size()
            .checked_sub(routed_expert_bytes)
            .ok_or_else(|| Error::weights("Laguna routed weights exceed total model size"))?;
        let bytes_per_expert = expected_expert_bytes(config)?;

        Ok(Self {
            experts,
            root,
            layers,
            summary: LagunaWeightSummary {
                tensor_count: LAGUNA_INT4_TENSOR_COUNT,
                shard_count: 15,
                total_bytes: LAGUNA_INT4_TOTAL_BYTES,
                fixed_weight_bytes,
                routed_expert_bytes,
                routed_layer_count: ROUTED_LAYER_COUNT,
                experts_per_layer: config.num_experts,
                bytes_per_expert,
            },
        })
    }

    pub fn expert(&self, layer_index: usize, expert_id: usize) -> Result<&LagunaExpertWeights> {
        if layer_index == 0 || layer_index >= self.layers.len() {
            return Err(Error::weights(format!(
                "Laguna routed expert layer must be within 1..{}, got {layer_index}",
                self.layers.len() - 1
            )));
        }
        if expert_id >= self.summary.experts_per_layer {
            return Err(Error::weights(format!(
                "Laguna expert id {expert_id} exceeds expert count {}",
                self.summary.experts_per_layer
            )));
        }
        let directory_index = (layer_index - 1)
            .checked_mul(self.summary.experts_per_layer)
            .and_then(|offset| offset.checked_add(expert_id))
            .ok_or_else(|| Error::weights("Laguna expert directory index overflow"))?;
        let expert = self.experts.get(directory_index).ok_or_else(|| {
            Error::weights(format!(
                "Laguna expert directory is missing layer {layer_index} expert {expert_id}"
            ))
        })?;
        if expert.layer_index != layer_index || expert.expert_id != expert_id {
            return Err(Error::weights(format!(
                "Laguna expert directory mismatch at {directory_index}: expected layer {layer_index} expert {expert_id}, got layer {} expert {}",
                expert.layer_index, expert.expert_id
            )));
        }
        Ok(expert)
    }
}

fn load_layer(
    storage: &SafeTensorModel,
    config: &LagunaConfig,
    layer_index: usize,
) -> Result<LagunaLayerWeights> {
    let prefix = format!("model.layers.{layer_index}");
    let query_width = config
        .query_width(layer_index)
        .ok_or_else(|| Error::weights(format!("Laguna layer {layer_index} is out of range")))?;
    let kv_width = config.key_value_width();
    let query_heads = config
        .query_heads(layer_index)
        .ok_or_else(|| Error::weights(format!("Laguna layer {layer_index} is out of range")))?;
    let attention = LagunaAttentionWeights {
        query: required(
            storage,
            &format!("{prefix}.self_attn.q_proj.weight"),
            SafeTensorDtype::Bf16,
            &[query_width, config.hidden_size],
        )?,
        key: required(
            storage,
            &format!("{prefix}.self_attn.k_proj.weight"),
            SafeTensorDtype::Bf16,
            &[kv_width, config.hidden_size],
        )?,
        value: required(
            storage,
            &format!("{prefix}.self_attn.v_proj.weight"),
            SafeTensorDtype::Bf16,
            &[kv_width, config.hidden_size],
        )?,
        output: required(
            storage,
            &format!("{prefix}.self_attn.o_proj.weight"),
            SafeTensorDtype::Bf16,
            &[config.hidden_size, query_width],
        )?,
        gate: required(
            storage,
            &format!("{prefix}.self_attn.g_proj.weight"),
            SafeTensorDtype::Bf16,
            &[query_heads, config.hidden_size],
        )?,
        query_norm: required(
            storage,
            &format!("{prefix}.self_attn.q_norm.weight"),
            SafeTensorDtype::Bf16,
            &[config.head_dim],
        )?,
        key_norm: required(
            storage,
            &format!("{prefix}.self_attn.k_norm.weight"),
            SafeTensorDtype::Bf16,
            &[config.head_dim],
        )?,
        key_scale: required(
            storage,
            &format!("{prefix}.self_attn.k_scale"),
            SafeTensorDtype::Bf16,
            &[1],
        )?,
        value_scale: required(
            storage,
            &format!("{prefix}.self_attn.v_scale"),
            SafeTensorDtype::Bf16,
            &[1],
        )?,
    };
    let mlp = match config.mlp_kind(layer_index) {
        Some(LagunaMlpKind::Dense) => LagunaLayerMlpWeights::Dense(load_dense(
            storage,
            &format!("{prefix}.mlp"),
            config.hidden_size,
            config.intermediate_size,
        )?),
        Some(LagunaMlpKind::Sparse) => LagunaLayerMlpWeights::Moe(LagunaMoeWeights {
            router: required(
                storage,
                &format!("{prefix}.mlp.gate.weight"),
                SafeTensorDtype::Bf16,
                &[config.num_experts, config.hidden_size],
            )?,
            correction_bias: required(
                storage,
                &format!("{prefix}.mlp.experts.e_score_correction_bias"),
                SafeTensorDtype::F32,
                &[config.num_experts],
            )?,
            shared: load_dense(
                storage,
                &format!("{prefix}.mlp.shared_expert"),
                config.hidden_size,
                config.shared_expert_intermediate_size,
            )?,
        }),
        None => {
            return Err(Error::weights(format!(
                "Laguna layer {layer_index} has no MLP type"
            )))
        }
    };

    Ok(LagunaLayerWeights {
        layer_index,
        input_norm: required(
            storage,
            &format!("{prefix}.input_layernorm.weight"),
            SafeTensorDtype::Bf16,
            &[config.hidden_size],
        )?,
        post_attention_norm: required(
            storage,
            &format!("{prefix}.post_attention_layernorm.weight"),
            SafeTensorDtype::Bf16,
            &[config.hidden_size],
        )?,
        attention,
        mlp,
    })
}

fn load_dense(
    storage: &SafeTensorModel,
    prefix: &str,
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<LagunaDenseWeights> {
    Ok(LagunaDenseWeights {
        gate: required(
            storage,
            &format!("{prefix}.gate_proj.weight"),
            SafeTensorDtype::Bf16,
            &[intermediate_size, hidden_size],
        )?,
        up: required(
            storage,
            &format!("{prefix}.up_proj.weight"),
            SafeTensorDtype::Bf16,
            &[intermediate_size, hidden_size],
        )?,
        down: required(
            storage,
            &format!("{prefix}.down_proj.weight"),
            SafeTensorDtype::Bf16,
            &[hidden_size, intermediate_size],
        )?,
    })
}

fn load_expert(
    storage: &SafeTensorModel,
    config: &LagunaConfig,
    layer_index: usize,
    expert_id: usize,
) -> Result<LagunaExpertWeights> {
    if expert_id >= config.num_experts {
        return Err(Error::weights(format!(
            "Laguna expert id {expert_id} exceeds expert count {}",
            config.num_experts
        )));
    }
    let prefix = format!("model.layers.{layer_index}.mlp.experts.{expert_id}");
    let gate_up_packed_shape = [
        config.moe_intermediate_size,
        config.hidden_size / PACKED_VALUES_PER_I32,
    ];
    let gate_up_scale_shape = [
        config.moe_intermediate_size,
        config.hidden_size / INT4_GROUP_SIZE,
    ];
    let down_packed_shape = [
        config.hidden_size,
        config.moe_intermediate_size / PACKED_VALUES_PER_I32,
    ];
    let down_scale_shape = [
        config.hidden_size,
        config.moe_intermediate_size / INT4_GROUP_SIZE,
    ];

    Ok(LagunaExpertWeights {
        layer_index,
        expert_id,
        gate_packed: required(
            storage,
            &format!("{prefix}.gate_proj.weight_packed"),
            SafeTensorDtype::I32,
            &gate_up_packed_shape,
        )?,
        gate_scales: required(
            storage,
            &format!("{prefix}.gate_proj.weight_scale"),
            SafeTensorDtype::Bf16,
            &gate_up_scale_shape,
        )?,
        up_packed: required(
            storage,
            &format!("{prefix}.up_proj.weight_packed"),
            SafeTensorDtype::I32,
            &gate_up_packed_shape,
        )?,
        up_scales: required(
            storage,
            &format!("{prefix}.up_proj.weight_scale"),
            SafeTensorDtype::Bf16,
            &gate_up_scale_shape,
        )?,
        down_packed: required(
            storage,
            &format!("{prefix}.down_proj.weight_packed"),
            SafeTensorDtype::I32,
            &down_packed_shape,
        )?,
        down_scales: required(
            storage,
            &format!("{prefix}.down_proj.weight_scale"),
            SafeTensorDtype::Bf16,
            &down_scale_shape,
        )?,
    })
}

fn required(
    storage: &SafeTensorModel,
    name: &str,
    dtype: SafeTensorDtype,
    shape: &[usize],
) -> Result<SafeTensorHandle> {
    let tensor = storage.tensor(name)?;
    let info = tensor.info();
    if info.dtype != dtype {
        return Err(Error::weights(format!(
            "Laguna tensor {name} must be {dtype:?}, got {:?}",
            info.dtype
        )));
    }
    if info.shape != shape {
        return Err(Error::ShapeMismatch {
            context: name.to_string(),
            expected: shape.to_vec(),
            actual: info.shape.clone(),
        });
    }
    Ok(tensor)
}

fn validate_index_contract(index: &SafeTensorIndex, config: &LagunaConfig) -> Result<()> {
    if index.total_size() != LAGUNA_INT4_TOTAL_BYTES {
        return Err(Error::weights(format!(
            "Laguna INT4 safetensors total size must be {LAGUNA_INT4_TOTAL_BYTES}, got {}",
            index.total_size()
        )));
    }
    if index.tensor_count() != LAGUNA_INT4_TENSOR_COUNT {
        return Err(Error::weights(format!(
            "Laguna INT4 safetensors index must contain {LAGUNA_INT4_TENSOR_COUNT} tensors, got {}",
            index.tensor_count()
        )));
    }
    if index.shard_count() != 15 {
        return Err(Error::weights(format!(
            "Laguna INT4 safetensors index must contain 15 shards, got {}",
            index.shard_count()
        )));
    }

    for name in expected_tensor_names(config) {
        if !index.contains_tensor(&name) {
            return Err(Error::weights(format!(
                "Laguna INT4 safetensors index is missing tensor {name}"
            )));
        }
    }
    Ok(())
}

fn expected_tensor_names(config: &LagunaConfig) -> Vec<String> {
    let mut names = Vec::with_capacity(LAGUNA_INT4_TENSOR_COUNT);
    names.extend([
        "model.embed_tokens.weight".to_string(),
        "model.norm.weight".to_string(),
        "lm_head.weight".to_string(),
    ]);
    for layer_index in 0..config.num_hidden_layers {
        let prefix = format!("model.layers.{layer_index}");
        names.extend([
            format!("{prefix}.input_layernorm.weight"),
            format!("{prefix}.post_attention_layernorm.weight"),
            format!("{prefix}.self_attn.q_proj.weight"),
            format!("{prefix}.self_attn.k_proj.weight"),
            format!("{prefix}.self_attn.v_proj.weight"),
            format!("{prefix}.self_attn.o_proj.weight"),
            format!("{prefix}.self_attn.g_proj.weight"),
            format!("{prefix}.self_attn.q_norm.weight"),
            format!("{prefix}.self_attn.k_norm.weight"),
            format!("{prefix}.self_attn.k_scale"),
            format!("{prefix}.self_attn.v_scale"),
        ]);
        if layer_index == 0 {
            names.extend([
                format!("{prefix}.mlp.gate_proj.weight"),
                format!("{prefix}.mlp.up_proj.weight"),
                format!("{prefix}.mlp.down_proj.weight"),
            ]);
            continue;
        }
        names.extend([
            format!("{prefix}.mlp.gate.weight"),
            format!("{prefix}.mlp.experts.e_score_correction_bias"),
            format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
            format!("{prefix}.mlp.shared_expert.up_proj.weight"),
            format!("{prefix}.mlp.shared_expert.down_proj.weight"),
        ]);
        for expert_id in 0..config.num_experts {
            let expert = format!("{prefix}.mlp.experts.{expert_id}");
            names.extend([
                format!("{expert}.gate_proj.weight_packed"),
                format!("{expert}.gate_proj.weight_scale"),
                format!("{expert}.up_proj.weight_packed"),
                format!("{expert}.up_proj.weight_scale"),
                format!("{expert}.down_proj.weight_packed"),
                format!("{expert}.down_proj.weight_scale"),
            ]);
        }
    }
    names
}

fn expected_expert_bytes(config: &LagunaConfig) -> Result<u64> {
    let packed_values = config
        .moe_intermediate_size
        .checked_mul(config.hidden_size)
        .and_then(|values| values.checked_mul(3))
        .ok_or_else(|| Error::weights("Laguna expert packed value count overflow"))?;
    let packed_bytes = packed_values / 2;
    let scale_values = config
        .moe_intermediate_size
        .checked_mul(config.hidden_size / INT4_GROUP_SIZE)
        .and_then(|values| values.checked_mul(2))
        .and_then(|gate_up| {
            config
                .hidden_size
                .checked_mul(config.moe_intermediate_size / INT4_GROUP_SIZE)
                .and_then(|down| gate_up.checked_add(down))
        })
        .ok_or_else(|| Error::weights("Laguna expert scale count overflow"))?;
    let total = packed_bytes
        .checked_add(
            scale_values
                .checked_mul(2)
                .ok_or_else(|| Error::weights("Laguna expert BF16 scale byte count overflow"))?,
        )
        .ok_or_else(|| Error::weights("Laguna expert byte count overflow"))?;
    u64::try_from(total).map_err(|_| Error::weights("Laguna expert bytes do not fit u64"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_tensor_directory_has_published_tensor_count() {
        // 3 root + 48 * (2 norms + 9 attention) + 3 dense MLP +
        // 47 * (router + bias + 3 shared + 256 * 6 expert tensors).
        let count = 3 + 48 * 11 + 3 + 47 * (5 + 256 * 6);
        assert_eq!(count, LAGUNA_INT4_TENSOR_COUNT);
    }

    #[test]
    fn one_expert_uses_three_int4_matrices_and_three_bf16_scale_tables() {
        let packed_bytes = (1_024 * 3_072 * 2 + 3_072 * 1_024) / 2;
        let scale_bytes = (1_024 * 96 * 2 + 3_072 * 32) * 2;

        assert_eq!(packed_bytes + scale_bytes, 5_308_416);
        assert_eq!(
            (packed_bytes + scale_bytes) as u64 * 256 * 47,
            63_870_861_312
        );
        assert_eq!(LAGUNA_INT4_TOTAL_BYTES - 63_870_861_312, 8_027_583_680);
    }
}
