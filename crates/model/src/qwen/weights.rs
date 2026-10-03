use std::{path::Path, sync::Arc};

use common::{Error, Result};
use config::{QwenConfig, QwenLayerKind};
use inferno_io::{SafeTensorDtype, SafeTensorHandle, SafeTensorInfo, SafeTensorModel};

use super::{QwenArtifactIndex, QwenArtifactSummary, QWEN_MLX_W4_TEXT_TENSOR_COUNT};

#[derive(Debug, Clone)]
pub struct QwenMlxW4Matrix {
    pub weight: SafeTensorHandle,
    pub scales: SafeTensorHandle,
    pub biases: SafeTensorHandle,
    pub rows: usize,
    pub columns: usize,
}

pub type QwenMatrix = QwenMlxW4Matrix;

#[derive(Debug, Clone)]
pub struct QwenRootWeights {
    pub embedding: QwenMatrix,
    pub final_norm: SafeTensorHandle,
    pub output: QwenMatrix,
}

#[derive(Debug, Clone)]
pub struct QwenMlpWeights {
    pub gate: QwenMatrix,
    pub up: QwenMatrix,
    pub down: QwenMatrix,
}

#[derive(Debug, Clone)]
pub struct QwenLinearAttentionWeights {
    pub a_log: SafeTensorHandle,
    pub conv1d: SafeTensorHandle,
    pub dt_bias: SafeTensorHandle,
    pub input_a: QwenMatrix,
    pub input_b: QwenMatrix,
    pub qkv: QwenMatrix,
    pub gate: QwenMatrix,
    pub norm: SafeTensorHandle,
    pub output: QwenMatrix,
}

#[derive(Debug, Clone)]
pub struct QwenFullAttentionWeights {
    pub query: QwenMatrix,
    pub key: QwenMatrix,
    pub value: QwenMatrix,
    pub output: QwenMatrix,
    pub query_norm: SafeTensorHandle,
    pub key_norm: SafeTensorHandle,
}

#[derive(Debug, Clone)]
pub enum QwenAttentionWeights {
    Linear(QwenLinearAttentionWeights),
    Full(QwenFullAttentionWeights),
}

#[derive(Debug, Clone)]
pub struct QwenLayerWeights {
    pub layer_index: usize,
    pub input_norm: SafeTensorHandle,
    pub post_attention_norm: SafeTensorHandle,
    pub attention: QwenAttentionWeights,
    pub mlp: QwenMlpWeights,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenWeightSummary {
    pub tensor_count: usize,
    pub mapped_shard_count: usize,
    pub base_text_bytes: u64,
    pub matrix_count: usize,
    pub raw_tensor_count: usize,
}

#[derive(Debug)]
pub struct QwenWeightIndex {
    storage: Arc<SafeTensorModel>,
    pub root: QwenRootWeights,
    pub layers: Vec<QwenLayerWeights>,
    pub artifact: QwenArtifactSummary,
    pub summary: QwenWeightSummary,
}

impl QwenWeightIndex {
    /// Maps only the base language-model shards and validates every text tensor.
    ///
    /// Reading Safetensors headers does not fault the large payloads into RAM.
    /// Vision tensor handles are never created.
    pub fn open(model_dir: impl AsRef<Path>, config: &QwenConfig) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let artifact = QwenArtifactIndex::open(model_dir, config)?.summary();
        let storage = Arc::new(SafeTensorModel::open(model_dir)?);
        let root = load_root(&storage, config)?;
        let layers = (0..config.text_config.num_hidden_layers)
            .map(|layer_index| load_layer(&storage, config, layer_index))
            .collect::<Result<Vec<_>>>()?;

        let expected_tensor_count = QWEN_MLX_W4_TEXT_TENSOR_COUNT;
        let mut handles = Vec::with_capacity(expected_tensor_count);
        root.collect_handles(&mut handles);
        for layer in &layers {
            layer.collect_handles(&mut handles);
        }
        if handles.len() != expected_tensor_count {
            return Err(Error::weights(format!(
                "Qwen3.8 base text index must hold {expected_tensor_count} tensors, got {}",
                handles.len()
            )));
        }
        let base_text_bytes = handles.into_iter().try_fold(0_u64, |total, handle| {
            total
                .checked_add(handle.info().byte_len)
                .ok_or_else(|| Error::weights("Qwen3.8 text weight byte count overflow"))
        })?;
        let mapped_shard_count = storage.loaded_shard_count()?;
        if mapped_shard_count != artifact.base_text_shard_count {
            return Err(Error::weights(format!(
                "Qwen3.8 base text path must map {} shards, got {mapped_shard_count}",
                artifact.base_text_shard_count
            )));
        }

        let (matrix_count, raw_tensor_count) = expected_weight_counts(config);
        Ok(Self {
            storage,
            root,
            layers,
            artifact,
            summary: QwenWeightSummary {
                tensor_count: expected_tensor_count,
                mapped_shard_count,
                base_text_bytes,
                matrix_count,
                raw_tensor_count,
            },
        })
    }

    pub fn loaded_shard_count(&self) -> Result<usize> {
        self.storage.loaded_shard_count()
    }
}

impl QwenMlxW4Matrix {
    fn collect_handles<'a>(&'a self, handles: &mut Vec<&'a SafeTensorHandle>) {
        handles.extend([&self.weight, &self.scales, &self.biases]);
    }
}

impl QwenMlxW4Matrix {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }
}

impl QwenRootWeights {
    fn collect_handles<'a>(&'a self, handles: &mut Vec<&'a SafeTensorHandle>) {
        self.embedding.collect_handles(handles);
        handles.push(&self.final_norm);
        self.output.collect_handles(handles);
    }
}

impl QwenMlpWeights {
    fn collect_handles<'a>(&'a self, handles: &mut Vec<&'a SafeTensorHandle>) {
        self.gate.collect_handles(handles);
        self.up.collect_handles(handles);
        self.down.collect_handles(handles);
    }
}

impl QwenLayerWeights {
    fn collect_handles<'a>(&'a self, handles: &mut Vec<&'a SafeTensorHandle>) {
        handles.extend([&self.input_norm, &self.post_attention_norm]);
        match &self.attention {
            QwenAttentionWeights::Linear(weights) => weights.collect_handles(handles),
            QwenAttentionWeights::Full(weights) => weights.collect_handles(handles),
        }
        self.mlp.collect_handles(handles);
    }
}

impl QwenLinearAttentionWeights {
    fn collect_handles<'a>(&'a self, handles: &mut Vec<&'a SafeTensorHandle>) {
        handles.extend([&self.a_log, &self.conv1d, &self.dt_bias, &self.norm]);
        self.input_a.collect_handles(handles);
        self.input_b.collect_handles(handles);
        self.qkv.collect_handles(handles);
        self.gate.collect_handles(handles);
        self.output.collect_handles(handles);
    }
}

impl QwenFullAttentionWeights {
    fn collect_handles<'a>(&'a self, handles: &mut Vec<&'a SafeTensorHandle>) {
        handles.extend([&self.query_norm, &self.key_norm]);
        self.query.collect_handles(handles);
        self.key.collect_handles(handles);
        self.value.collect_handles(handles);
        self.output.collect_handles(handles);
    }
}

fn load_root(storage: &SafeTensorModel, config: &QwenConfig) -> Result<QwenRootWeights> {
    let hidden = config.text_config.hidden_size;
    let vocab = config.text_config.vocab_size;
    let prefix = "language_model.model";
    let output_prefix = "language_model.lm_head";
    Ok(QwenRootWeights {
        embedding: required_matrix(
            storage,
            config,
            &format!("{prefix}.embed_tokens"),
            vocab,
            hidden,
        )?,
        final_norm: required(
            storage,
            &format!("{prefix}.norm.weight"),
            SafeTensorDtype::Bf16,
            &[hidden],
        )?,
        output: required_matrix(storage, config, output_prefix, vocab, hidden)?,
    })
}

fn load_layer(
    storage: &SafeTensorModel,
    config: &QwenConfig,
    layer_index: usize,
) -> Result<QwenLayerWeights> {
    let prefix = format!("language_model.model.layers.{layer_index}");
    let hidden = config.text_config.hidden_size;
    let attention = match config.layer_kind(layer_index) {
        Some(QwenLayerKind::LinearAttention) => QwenAttentionWeights::Linear(
            load_linear_attention(storage, config, &format!("{prefix}.linear_attn"))?,
        ),
        Some(QwenLayerKind::FullAttention) => QwenAttentionWeights::Full(load_full_attention(
            storage,
            config,
            &format!("{prefix}.self_attn"),
        )?),
        None => {
            return Err(Error::weights(format!(
                "Qwen3.8 config is missing layer {layer_index}"
            )))
        }
    };
    Ok(QwenLayerWeights {
        layer_index,
        input_norm: required(
            storage,
            &format!("{prefix}.input_layernorm.weight"),
            SafeTensorDtype::Bf16,
            &[hidden],
        )?,
        post_attention_norm: required(
            storage,
            &format!("{prefix}.post_attention_layernorm.weight"),
            SafeTensorDtype::Bf16,
            &[hidden],
        )?,
        attention,
        mlp: load_mlp(storage, config, &format!("{prefix}.mlp"))?,
    })
}

fn load_linear_attention(
    storage: &SafeTensorModel,
    config: &QwenConfig,
    prefix: &str,
) -> Result<QwenLinearAttentionWeights> {
    let hidden = config.text_config.hidden_size;
    let value_heads = config.text_config.linear_num_value_heads;
    let conv_shape = [
        config.linear_qkv_width(),
        config.text_config.linear_conv_kernel_dim,
        1,
    ];
    Ok(QwenLinearAttentionWeights {
        a_log: required(
            storage,
            &format!("{prefix}.A_log"),
            SafeTensorDtype::Bf16,
            &[value_heads],
        )?,
        conv1d: required(
            storage,
            &format!("{prefix}.conv1d.weight"),
            SafeTensorDtype::Bf16,
            &conv_shape,
        )?,
        dt_bias: required(
            storage,
            &format!("{prefix}.dt_bias"),
            SafeTensorDtype::Bf16,
            &[value_heads],
        )?,
        input_a: required_matrix(
            storage,
            config,
            &format!("{prefix}.in_proj_a"),
            value_heads,
            hidden,
        )?,
        input_b: required_matrix(
            storage,
            config,
            &format!("{prefix}.in_proj_b"),
            value_heads,
            hidden,
        )?,
        qkv: required_matrix(
            storage,
            config,
            &format!("{prefix}.in_proj_qkv"),
            config.linear_qkv_width(),
            hidden,
        )?,
        gate: required_matrix(
            storage,
            config,
            &format!("{prefix}.in_proj_z"),
            config.linear_value_width(),
            hidden,
        )?,
        norm: required(
            storage,
            &format!("{prefix}.norm.weight"),
            SafeTensorDtype::Bf16,
            &[config.text_config.linear_value_head_dim],
        )?,
        output: required_matrix(
            storage,
            config,
            &format!("{prefix}.out_proj"),
            hidden,
            config.linear_value_width(),
        )?,
    })
}

fn load_full_attention(
    storage: &SafeTensorModel,
    config: &QwenConfig,
    prefix: &str,
) -> Result<QwenFullAttentionWeights> {
    let hidden = config.text_config.hidden_size;
    let head_dim = config.text_config.head_dim;
    Ok(QwenFullAttentionWeights {
        // Qwen stores query channels followed by an equally sized output gate.
        // Attention itself still has 24 heads with 256 values per head.
        query: required_matrix(
            storage,
            config,
            &format!("{prefix}.q_proj"),
            2 * config.full_query_width(),
            hidden,
        )?,
        key: required_matrix(
            storage,
            config,
            &format!("{prefix}.k_proj"),
            config.full_kv_width(),
            hidden,
        )?,
        value: required_matrix(
            storage,
            config,
            &format!("{prefix}.v_proj"),
            config.full_kv_width(),
            hidden,
        )?,
        output: required_matrix(
            storage,
            config,
            &format!("{prefix}.o_proj"),
            hidden,
            config.full_query_width(),
        )?,
        query_norm: required(
            storage,
            &format!("{prefix}.q_norm.weight"),
            SafeTensorDtype::Bf16,
            &[head_dim],
        )?,
        key_norm: required(
            storage,
            &format!("{prefix}.k_norm.weight"),
            SafeTensorDtype::Bf16,
            &[head_dim],
        )?,
    })
}

fn load_mlp(
    storage: &SafeTensorModel,
    config: &QwenConfig,
    prefix: &str,
) -> Result<QwenMlpWeights> {
    let hidden = config.text_config.hidden_size;
    let intermediate = config.text_config.intermediate_size;
    Ok(QwenMlpWeights {
        gate: required_matrix(
            storage,
            config,
            &format!("{prefix}.gate_proj"),
            intermediate,
            hidden,
        )?,
        up: required_matrix(
            storage,
            config,
            &format!("{prefix}.up_proj"),
            intermediate,
            hidden,
        )?,
        down: required_matrix(
            storage,
            config,
            &format!("{prefix}.down_proj"),
            hidden,
            intermediate,
        )?,
    })
}

fn required_matrix(
    storage: &SafeTensorModel,
    config: &QwenConfig,
    prefix: &str,
    rows: usize,
    columns: usize,
) -> Result<QwenMatrix> {
    let _ = config;
    required_mlx_w4(storage, prefix, rows, columns)
}

pub(super) fn required_mlx_w4(
    storage: &SafeTensorModel,
    prefix: &str,
    rows: usize,
    columns: usize,
) -> Result<QwenMlxW4Matrix> {
    const GROUP_SIZE: usize = 64;
    const VALUES_PER_WORD: usize = 8;
    if rows == 0 || columns == 0 || !columns.is_multiple_of(GROUP_SIZE) {
        return Err(Error::weights(format!(
            "Qwen3.8 MLX W4 matrix [{rows}, {columns}] must have a positive width divisible by {GROUP_SIZE}"
        )));
    }
    Ok(QwenMlxW4Matrix {
        weight: required(
            storage,
            &format!("{prefix}.weight"),
            SafeTensorDtype::U32,
            &[rows, columns / VALUES_PER_WORD],
        )?,
        scales: required(
            storage,
            &format!("{prefix}.scales"),
            SafeTensorDtype::Bf16,
            &[rows, columns / GROUP_SIZE],
        )?,
        biases: required(
            storage,
            &format!("{prefix}.biases"),
            SafeTensorDtype::Bf16,
            &[rows, columns / GROUP_SIZE],
        )?,
        rows,
        columns,
    })
}

fn expected_weight_counts(config: &QwenConfig) -> (usize, usize) {
    let _ = config;
    (498, 353)
}

pub(super) fn required(
    storage: &SafeTensorModel,
    name: &str,
    dtype: SafeTensorDtype,
    shape: &[usize],
) -> Result<SafeTensorHandle> {
    let tensor = storage.tensor(name)?;
    validate_tensor_info(tensor.info(), dtype, shape)?;
    Ok(tensor)
}

pub(crate) fn validate_tensor_info(
    info: &SafeTensorInfo,
    dtype: SafeTensorDtype,
    shape: &[usize],
) -> Result<()> {
    if info.dtype != dtype {
        return Err(Error::weights(format!(
            "Qwen3.8 tensor {} must be {dtype:?}, got {:?}",
            info.name, info.dtype
        )));
    }
    if info.shape != shape {
        return Err(Error::ShapeMismatch {
            context: info.name.clone(),
            expected: shape.to_vec(),
            actual: info.shape.clone(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_dtype_and_shape_before_execution() {
        let info = SafeTensorInfo {
            name: "q_proj.weight".to_string(),
            dtype: SafeTensorDtype::U32,
            shape: vec![12_288, 640],
            file_offset: 0,
            byte_len: (12_288 * 640 * 4) as u64,
        };
        validate_tensor_info(&info, SafeTensorDtype::U32, &[12_288, 640]).unwrap();

        let error = validate_tensor_info(&info, SafeTensorDtype::U32, &[6_144, 640]).unwrap_err();
        assert!(matches!(error, Error::ShapeMismatch { .. }));
    }
}
