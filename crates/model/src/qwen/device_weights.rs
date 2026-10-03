use backend::{Backend, DeviceQwenBf16Tensor, DeviceQwenMatrix};
use common::{Error, Result};
use config::{QwenConfig, QwenLayerKind};
use inferno_io::SafeTensorHandle;

use super::{
    QwenArtifactSummary, QwenAttentionWeights, QwenFullAttentionWeights,
    QwenLinearAttentionWeights, QwenMatrix, QwenMlpWeights, QwenWeightIndex, QwenWeightSummary,
};

#[derive(Debug, Clone)]
pub struct QwenDeviceRootWeights {
    pub embedding: DeviceQwenMatrix,
    pub final_norm: DeviceQwenBf16Tensor,
    pub output: DeviceQwenMatrix,
}

#[derive(Debug, Clone)]
pub struct QwenDeviceMlpWeights {
    pub gate: DeviceQwenMatrix,
    pub up: DeviceQwenMatrix,
    pub down: DeviceQwenMatrix,
}

#[derive(Debug, Clone)]
pub struct QwenDeviceLinearAttentionWeights {
    pub a_log: DeviceQwenBf16Tensor,
    pub conv1d: DeviceQwenBf16Tensor,
    pub dt_bias: DeviceQwenBf16Tensor,
    pub input_a: DeviceQwenMatrix,
    pub input_b: DeviceQwenMatrix,
    pub qkv: DeviceQwenMatrix,
    pub gate: DeviceQwenMatrix,
    pub norm: DeviceQwenBf16Tensor,
    pub output: DeviceQwenMatrix,
}

#[derive(Debug, Clone)]
pub struct QwenDeviceFullAttentionWeights {
    pub query: DeviceQwenMatrix,
    pub key: DeviceQwenMatrix,
    pub value: DeviceQwenMatrix,
    pub output: DeviceQwenMatrix,
    pub query_norm: DeviceQwenBf16Tensor,
    pub key_norm: DeviceQwenBf16Tensor,
}

#[derive(Debug, Clone)]
pub enum QwenDeviceAttentionWeights {
    Linear(QwenDeviceLinearAttentionWeights),
    Full(QwenDeviceFullAttentionWeights),
}

#[derive(Debug, Clone)]
pub struct QwenDeviceLayerWeights {
    pub layer_index: usize,
    pub input_norm: DeviceQwenBf16Tensor,
    pub post_attention_norm: DeviceQwenBf16Tensor,
    pub attention: QwenDeviceAttentionWeights,
    pub mlp: QwenDeviceMlpWeights,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenDeviceWeightSummary {
    pub source: QwenWeightSummary,
    pub matrix_count: usize,
    pub raw_tensor_count: usize,
}

#[derive(Debug)]
pub struct QwenDeviceWeights {
    pub root: QwenDeviceRootWeights,
    pub layers: Vec<QwenDeviceLayerWeights>,
    pub artifact: QwenArtifactSummary,
    pub summary: QwenDeviceWeightSummary,
}

impl QwenDeviceWeights {
    pub fn prepare_resident<B: Backend>(
        source: &QwenWeightIndex,
        config: &QwenConfig,
        backend: &B,
    ) -> Result<Self> {
        if source.layers.len() != config.text_config.num_hidden_layers {
            return Err(Error::weights(format!(
                "Qwen device preparation requires {} layers, got {}",
                config.text_config.num_hidden_layers,
                source.layers.len()
            )));
        }
        let (matrix_count, raw_tensor_count) = expected_device_counts(config);
        let tensors_per_matrix = 3;
        if matrix_count * tensors_per_matrix + raw_tensor_count != source.summary.tensor_count {
            return Err(Error::weights(
                "Qwen prepared weight count does not match the validated text artifact",
            ));
        }
        let root = QwenDeviceRootWeights {
            embedding: prepare_matrix_resident(&source.root.embedding, backend)?,
            final_norm: prepare_bf16_resident(&source.root.final_norm, backend)?,
            output: prepare_matrix_resident(&source.root.output, backend)?,
        };
        backend.device_submit()?;
        let mut layers = Vec::with_capacity(source.layers.len());
        for layer in &source.layers {
            let attention = match &layer.attention {
                QwenAttentionWeights::Linear(weights) => {
                    QwenDeviceAttentionWeights::Linear(prepare_linear_attention(weights, backend)?)
                }
                QwenAttentionWeights::Full(weights) => {
                    QwenDeviceAttentionWeights::Full(prepare_full_attention(weights, backend)?)
                }
            };
            layers.push(QwenDeviceLayerWeights {
                layer_index: layer.layer_index,
                input_norm: prepare_bf16_resident(&layer.input_norm, backend)?,
                post_attention_norm: prepare_bf16_resident(&layer.post_attention_norm, backend)?,
                attention,
                mlp: prepare_mlp(&layer.mlp, backend)?,
            });
            backend.device_submit()?;
        }
        backend.device_flush()?;

        Ok(Self {
            root,
            layers,
            artifact: source.artifact,
            summary: QwenDeviceWeightSummary {
                source: source.summary,
                matrix_count,
                raw_tensor_count,
            },
        })
    }
}

fn prepare_linear_attention<B: Backend>(
    weights: &QwenLinearAttentionWeights,
    backend: &B,
) -> Result<QwenDeviceLinearAttentionWeights> {
    Ok(QwenDeviceLinearAttentionWeights {
        a_log: prepare_bf16_resident(&weights.a_log, backend)?,
        conv1d: prepare_bf16_resident(&weights.conv1d, backend)?,
        dt_bias: prepare_bf16_resident(&weights.dt_bias, backend)?,
        input_a: prepare_matrix_resident(&weights.input_a, backend)?,
        input_b: prepare_matrix_resident(&weights.input_b, backend)?,
        qkv: prepare_matrix_resident(&weights.qkv, backend)?,
        gate: prepare_matrix_resident(&weights.gate, backend)?,
        norm: prepare_bf16_resident(&weights.norm, backend)?,
        output: prepare_matrix_resident(&weights.output, backend)?,
    })
}

pub(super) fn prepare_full_attention<B: Backend>(
    weights: &QwenFullAttentionWeights,
    backend: &B,
) -> Result<QwenDeviceFullAttentionWeights> {
    Ok(QwenDeviceFullAttentionWeights {
        query: prepare_matrix_resident(&weights.query, backend)?,
        key: prepare_matrix_resident(&weights.key, backend)?,
        value: prepare_matrix_resident(&weights.value, backend)?,
        output: prepare_matrix_resident(&weights.output, backend)?,
        query_norm: prepare_bf16_resident(&weights.query_norm, backend)?,
        key_norm: prepare_bf16_resident(&weights.key_norm, backend)?,
    })
}

pub(super) fn prepare_mlp<B: Backend>(
    weights: &QwenMlpWeights,
    backend: &B,
) -> Result<QwenDeviceMlpWeights> {
    Ok(QwenDeviceMlpWeights {
        gate: prepare_matrix_resident(&weights.gate, backend)?,
        up: prepare_matrix_resident(&weights.up, backend)?,
        down: prepare_matrix_resident(&weights.down, backend)?,
    })
}

pub(super) fn prepare_matrix_resident<B: Backend>(
    matrix: &QwenMatrix,
    backend: &B,
) -> Result<DeviceQwenMatrix> {
    let prepared = backend.prepare_qwen_mlx_w4_matrix_resident(
        matrix.weight.clone(),
        matrix.scales.clone(),
        matrix.biases.clone(),
        matrix.rows,
        matrix.columns,
    )?;
    prepared.ok_or_else(|| Error::backend("Qwen3.8 MLX W4 requires the native Metal backend"))
}

pub(crate) fn prepare_bf16<B: Backend>(
    tensor: &SafeTensorHandle,
    backend: &B,
) -> Result<DeviceQwenBf16Tensor> {
    let prepared = backend.prepare_qwen_bf16_tensor(tensor.clone())?;
    prepared.ok_or_else(|| Error::backend("Qwen3.8 BF16 weights require the native Metal backend"))
}

pub(super) fn prepare_bf16_resident<B: Backend>(
    tensor: &SafeTensorHandle,
    backend: &B,
) -> Result<DeviceQwenBf16Tensor> {
    let prepared = backend.prepare_qwen_bf16_tensor_resident(tensor.clone())?;
    prepared.ok_or_else(|| Error::backend("Qwen3.8 BF16 weights require the native Metal backend"))
}

fn expected_device_counts(config: &QwenConfig) -> (usize, usize) {
    device_counts_for_layer_kinds(config.text_config.layer_types.iter().copied())
}

fn device_counts_for_layer_kinds(
    layers: impl IntoIterator<Item = QwenLayerKind>,
) -> (usize, usize) {
    let (mut matrices, mut raw_tensors) = (2_usize, 1_usize);
    for kind in layers {
        match kind {
            QwenLayerKind::LinearAttention => {
                matrices += 8;
                raw_tensors += 6;
            }
            QwenLayerKind::FullAttention => {
                matrices += 7;
                raw_tensors += 4;
            }
        }
    }
    (matrices, raw_tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_counts_cover_every_base_text_tensor_once() {
        let schedule = (0..64).map(|layer_index| {
            if (layer_index + 1) % 4 == 0 {
                QwenLayerKind::FullAttention
            } else {
                QwenLayerKind::LinearAttention
            }
        });
        let (matrices, raw) = device_counts_for_layer_kinds(schedule);

        assert_eq!(matrices, 498);
        assert_eq!(raw, 353);
        assert_eq!(
            matrices * 3 + raw,
            super::super::QWEN_MLX_W4_TEXT_TENSOR_COUNT
        );
    }
}
