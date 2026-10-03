use backend::{Backend, DeviceQwenBf16Tensor, DeviceQwenMatrix};
use common::Result;

use super::{
    device_weights::{
        prepare_bf16_resident, prepare_full_attention, prepare_matrix_resident, prepare_mlp,
    },
    QwenDeviceFullAttentionWeights, QwenDeviceMlpWeights, QwenMtpWeightIndex,
};

#[derive(Debug, Clone)]
pub struct QwenDeviceMtpLayerWeights {
    pub input_norm: DeviceQwenBf16Tensor,
    pub post_attention_norm: DeviceQwenBf16Tensor,
    pub attention: QwenDeviceFullAttentionWeights,
    pub mlp: QwenDeviceMlpWeights,
}

#[derive(Debug, Clone)]
pub struct QwenDeviceMtpWeights {
    pub pre_fc_hidden_norm: DeviceQwenBf16Tensor,
    pub pre_fc_embedding_norm: DeviceQwenBf16Tensor,
    pub fusion: DeviceQwenMatrix,
    pub layer: QwenDeviceMtpLayerWeights,
    pub final_norm: DeviceQwenBf16Tensor,
}

impl QwenDeviceMtpWeights {
    pub fn prepare<B: Backend>(source: &QwenMtpWeightIndex, backend: &B) -> Result<Self> {
        let source = &source.weights;
        let weights = Self {
            pre_fc_hidden_norm: prepare_bf16_resident(&source.pre_fc_hidden_norm, backend)?,
            pre_fc_embedding_norm: prepare_bf16_resident(&source.pre_fc_embedding_norm, backend)?,
            fusion: prepare_matrix_resident(&source.fusion, backend)?,
            layer: QwenDeviceMtpLayerWeights {
                input_norm: prepare_bf16_resident(&source.layer.input_norm, backend)?,
                post_attention_norm: prepare_bf16_resident(
                    &source.layer.post_attention_norm,
                    backend,
                )?,
                attention: prepare_full_attention(&source.layer.attention, backend)?,
                mlp: prepare_mlp(&source.layer.mlp, backend)?,
            },
            final_norm: prepare_bf16_resident(&source.final_norm, backend)?,
        };
        backend.device_submit()?;
        backend.device_flush()?;
        Ok(weights)
    }
}
