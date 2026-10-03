use backend::{Backend, DeviceDFlashW4Matrix, DeviceQwenBf16Tensor};
use common::{Error, Result};
use inferno_io::SafeTensorHandle;

use super::{
    device_weights::prepare_bf16, DFlashAttentionWeights, DFlashConvWeights, DFlashLayerWeights,
    DFlashMlpWeights, DFlashWeightIndex,
};

#[derive(Debug, Clone)]
pub struct DFlashDeviceConvWeights {
    pub base_kernel: DeviceQwenBf16Tensor,
    pub kernel_projection: DeviceDFlashW4Matrix,
}

#[derive(Debug, Clone)]
pub struct DFlashDeviceAttentionWeights {
    pub query: DeviceDFlashW4Matrix,
    pub key: DeviceDFlashW4Matrix,
    pub value: DeviceDFlashW4Matrix,
    pub output: DeviceDFlashW4Matrix,
    pub query_norm: DeviceQwenBf16Tensor,
    pub key_norm: DeviceQwenBf16Tensor,
}

#[derive(Debug, Clone)]
pub struct DFlashDeviceMlpWeights {
    pub gate: DeviceDFlashW4Matrix,
    pub up: DeviceDFlashW4Matrix,
    pub down: DeviceDFlashW4Matrix,
}

#[derive(Debug, Clone)]
pub struct DFlashDeviceLayerWeights {
    pub layer_index: usize,
    pub input_norm: DeviceQwenBf16Tensor,
    pub attention_conv: DFlashDeviceConvWeights,
    pub attention: DFlashDeviceAttentionWeights,
    pub post_attention_norm: DeviceQwenBf16Tensor,
    pub mlp_conv: DFlashDeviceConvWeights,
    pub mlp: DFlashDeviceMlpWeights,
}

#[derive(Debug)]
pub struct DFlashDeviceWeights {
    pub selector_hidden_projection: DeviceDFlashW4Matrix,
    pub predecessor_codebook: DeviceQwenBf16Tensor,
    pub successor_codebook: DeviceQwenBf16Tensor,
    pub target_fusion: DeviceDFlashW4Matrix,
    pub target_hidden_norm: DeviceQwenBf16Tensor,
    pub layers: Vec<DFlashDeviceLayerWeights>,
    pub final_norm: DeviceQwenBf16Tensor,
}

impl DFlashDeviceWeights {
    pub fn prepare<B: Backend>(source: &DFlashWeightIndex, backend: &B) -> Result<Self> {
        let source = &source.weights;
        let weights = Self {
            selector_hidden_projection: prepare_w4(&source.selector_hidden_projection, backend)?,
            predecessor_codebook: prepare_bf16(&source.predecessor_codebook, backend)?,
            successor_codebook: prepare_bf16(&source.successor_codebook, backend)?,
            target_fusion: prepare_w4(&source.target_fusion, backend)?,
            target_hidden_norm: prepare_bf16(&source.target_hidden_norm, backend)?,
            layers: source
                .layers
                .iter()
                .map(|layer| prepare_layer(layer, backend))
                .collect::<Result<Vec<_>>>()?,
            final_norm: prepare_bf16(&source.final_norm, backend)?,
        };
        // W4 destinations are persistent, while their BF16 sources are mapped.
        // Finish quantization before the source mappings may be reclaimed.
        backend.device_flush()?;
        Ok(weights)
    }
}

fn prepare_layer<B: Backend>(
    source: &DFlashLayerWeights,
    backend: &B,
) -> Result<DFlashDeviceLayerWeights> {
    Ok(DFlashDeviceLayerWeights {
        layer_index: source.layer_index,
        input_norm: prepare_bf16(&source.input_norm, backend)?,
        attention_conv: prepare_conv(&source.attention_conv, backend)?,
        attention: prepare_attention(&source.attention, backend)?,
        post_attention_norm: prepare_bf16(&source.post_attention_norm, backend)?,
        mlp_conv: prepare_conv(&source.mlp_conv, backend)?,
        mlp: prepare_mlp(&source.mlp, backend)?,
    })
}

fn prepare_conv<B: Backend>(
    source: &DFlashConvWeights,
    backend: &B,
) -> Result<DFlashDeviceConvWeights> {
    Ok(DFlashDeviceConvWeights {
        base_kernel: prepare_bf16(&source.base_kernel, backend)?,
        kernel_projection: prepare_w4(&source.kernel_projection, backend)?,
    })
}

fn prepare_attention<B: Backend>(
    source: &DFlashAttentionWeights,
    backend: &B,
) -> Result<DFlashDeviceAttentionWeights> {
    Ok(DFlashDeviceAttentionWeights {
        query: prepare_w4(&source.query, backend)?,
        key: prepare_w4(&source.key, backend)?,
        value: prepare_w4(&source.value, backend)?,
        output: prepare_w4(&source.output, backend)?,
        query_norm: prepare_bf16(&source.query_norm, backend)?,
        key_norm: prepare_bf16(&source.key_norm, backend)?,
    })
}

fn prepare_mlp<B: Backend>(
    source: &DFlashMlpWeights,
    backend: &B,
) -> Result<DFlashDeviceMlpWeights> {
    Ok(DFlashDeviceMlpWeights {
        gate: prepare_w4(&source.gate, backend)?,
        up: prepare_w4(&source.up, backend)?,
        down: prepare_w4(&source.down, backend)?,
    })
}

fn prepare_w4<B: Backend>(source: &SafeTensorHandle, backend: &B) -> Result<DeviceDFlashW4Matrix> {
    const GROUP_SIZE: usize = 64;
    let [rows, columns]: [usize; 2] = source
        .info()
        .shape
        .as_slice()
        .try_into()
        .map_err(|_| Error::weights("DFlash2 quantized matrix must be rank 2"))?;
    backend
        .prepare_dflash_w4_matrix(source.clone(), rows, columns, GROUP_SIZE)?
        .ok_or_else(|| Error::backend("DFlash2 W4 weights require native Metal"))
}
