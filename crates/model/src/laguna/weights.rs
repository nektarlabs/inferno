use backend::{Backend, DeviceBf16Matrix, DeviceRopeTable, DeviceW4Weight};
use common::{Error, F32Tensor, Result};
use config::LagunaConfig;
use inferno_io::SafeTensorHandle;

use super::{
    LagunaAttentionWeights, LagunaDenseWeights, LagunaExpertWeights, LagunaLayerMlpWeights,
    LagunaMoeWeights, LagunaWeightIndex,
};

#[derive(Debug, Clone)]
pub struct LagunaDeviceRootWeights {
    pub embedding: DeviceBf16Matrix,
    pub final_norm: F32Tensor,
    pub output: DeviceBf16Matrix,
}

#[derive(Debug, Clone)]
pub struct LagunaDeviceRopeTables {
    pub full_attention: DeviceRopeTable,
    pub sliding_attention: DeviceRopeTable,
}

#[derive(Debug, Clone)]
pub struct LagunaDeviceAttentionWeights {
    pub query: DeviceBf16Matrix,
    pub key: DeviceBf16Matrix,
    pub value: DeviceBf16Matrix,
    pub output: DeviceBf16Matrix,
    pub gate: DeviceBf16Matrix,
    pub query_norm: F32Tensor,
    pub key_norm: F32Tensor,
    pub key_scale: f32,
    pub value_scale: f32,
}

#[derive(Debug, Clone)]
pub struct LagunaDeviceDenseWeights {
    pub gate: DeviceBf16Matrix,
    pub up: DeviceBf16Matrix,
    pub down: DeviceBf16Matrix,
}

#[derive(Debug, Clone)]
pub struct LagunaDeviceMoeWeights {
    pub router: DeviceBf16Matrix,
    pub correction_bias: Vec<f32>,
    pub shared: LagunaDeviceDenseWeights,
}

#[derive(Debug, Clone)]
pub enum LagunaDeviceLayerMlpWeights {
    Dense(LagunaDeviceDenseWeights),
    Moe(LagunaDeviceMoeWeights),
}

#[derive(Debug, Clone)]
pub struct LagunaDeviceLayerWeights {
    pub layer_index: usize,
    pub input_norm: F32Tensor,
    pub post_attention_norm: F32Tensor,
    pub attention: LagunaDeviceAttentionWeights,
    pub mlp: LagunaDeviceLayerMlpWeights,
}

#[derive(Debug, Clone)]
pub struct LagunaDeviceExpertWeights {
    pub layer_index: usize,
    pub expert_id: usize,
    pub gate: DeviceW4Weight,
    pub up: DeviceW4Weight,
    pub down: DeviceW4Weight,
}

#[derive(Debug)]
pub struct LagunaDeviceWeights {
    pub root: LagunaDeviceRootWeights,
    pub rope: LagunaDeviceRopeTables,
    pub layers: Vec<LagunaDeviceLayerWeights>,
    pub prepared_matrix_bytes: u64,
}

impl LagunaDeviceWeights {
    /// Prepares only the matrices used by every forward pass. Routed expert
    /// payloads remain cold until routing selects them.
    pub fn prepare<B: Backend>(
        index: &LagunaWeightIndex,
        config: &LagunaConfig,
        backend: &B,
    ) -> Result<Self> {
        let root = LagunaDeviceRootWeights {
            embedding: prepare_matrix(&index.root.embedding, backend)?,
            final_norm: prepare_hidden_norm(&index.root.final_norm)?,
            output: prepare_matrix(&index.root.output, backend)?,
        };
        let rope = LagunaDeviceRopeTables {
            full_attention: prepare_rope_table(config, 0, backend)?,
            sliding_attention: prepare_rope_table(config, 1, backend)?,
        };
        let layers = index
            .layers
            .iter()
            .map(|layer| {
                let attention = prepare_attention(&layer.attention, backend)?;
                let mlp = match &layer.mlp {
                    LagunaLayerMlpWeights::Dense(weights) => {
                        LagunaDeviceLayerMlpWeights::Dense(prepare_dense(weights, backend)?)
                    }
                    LagunaLayerMlpWeights::Moe(weights) => {
                        LagunaDeviceLayerMlpWeights::Moe(prepare_moe(weights, backend)?)
                    }
                };
                Ok(LagunaDeviceLayerWeights {
                    layer_index: layer.layer_index,
                    input_norm: prepare_hidden_norm(&layer.input_norm)?,
                    post_attention_norm: prepare_hidden_norm(&layer.post_attention_norm)?,
                    attention,
                    mlp,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if layers.len() != config.num_hidden_layers {
            return Err(Error::weights(format!(
                "Laguna prepared layer count mismatch: expected {}, got {}",
                config.num_hidden_layers,
                layers.len()
            )));
        }
        let prepared_matrix_bytes = matrix_bytes(&root, &layers)?;
        Ok(Self {
            root,
            rope,
            layers,
            prepared_matrix_bytes,
        })
    }
}

fn prepare_rope_table<B: Backend>(
    config: &LagunaConfig,
    layer_index: usize,
    backend: &B,
) -> Result<DeviceRopeTable> {
    let frequencies = config.rope_frequencies(layer_index)?;
    backend
        .prepare_rope_table(
            &frequencies.inverse_frequencies,
            frequencies.rotary_dim,
            frequencies.attention_factor,
        )?
        .ok_or_else(|| Error::backend("native backend cannot prepare Laguna RoPE table"))
}

impl LagunaDeviceExpertWeights {
    /// Copies one selected expert into Metal-owned buffers.
    ///
    /// Laguna's small resident cache reuses these prepared buffers across
    /// tokens. This avoids repeated page faults while a selected expert kernel
    /// is reading its gate, up, and down matrices.
    pub fn prepare_resident<B: Backend>(
        source: &LagunaExpertWeights,
        config: &LagunaConfig,
        backend: &B,
    ) -> Result<Self> {
        source.advise_random()?;
        Ok(Self {
            layer_index: source.layer_index,
            expert_id: source.expert_id,
            gate: prepare_resident_w4(
                &source.gate_packed,
                &source.gate_scales,
                config.hidden_size,
                config.moe_intermediate_size,
                backend,
            )?,
            up: prepare_resident_w4(
                &source.up_packed,
                &source.up_scales,
                config.hidden_size,
                config.moe_intermediate_size,
                backend,
            )?,
            down: prepare_resident_w4(
                &source.down_packed,
                &source.down_scales,
                config.moe_intermediate_size,
                config.hidden_size,
                backend,
            )?,
        })
    }

    pub fn storage_bytes(&self) -> Result<usize> {
        let gate_up = self
            .gate
            .storage_bytes()?
            .checked_add(self.up.storage_bytes()?)
            .ok_or_else(|| Error::weights("Laguna prepared expert gate/up bytes overflow"))?;
        gate_up
            .checked_add(self.down.storage_bytes()?)
            .ok_or_else(|| Error::weights("Laguna prepared expert byte count overflow"))
    }
}

fn prepare_attention<B: Backend>(
    weights: &LagunaAttentionWeights,
    backend: &B,
) -> Result<LagunaDeviceAttentionWeights> {
    Ok(LagunaDeviceAttentionWeights {
        query: prepare_matrix(&weights.query, backend)?,
        key: prepare_matrix(&weights.key, backend)?,
        value: prepare_matrix(&weights.value, backend)?,
        output: prepare_matrix(&weights.output, backend)?,
        gate: prepare_matrix(&weights.gate, backend)?,
        query_norm: prepare_vector(&weights.query_norm)?,
        key_norm: prepare_vector(&weights.key_norm)?,
        key_scale: prepare_scalar(&weights.key_scale)?,
        value_scale: prepare_scalar(&weights.value_scale)?,
    })
}

fn prepare_dense<B: Backend>(
    weights: &LagunaDenseWeights,
    backend: &B,
) -> Result<LagunaDeviceDenseWeights> {
    Ok(LagunaDeviceDenseWeights {
        gate: prepare_matrix(&weights.gate, backend)?,
        up: prepare_matrix(&weights.up, backend)?,
        down: prepare_matrix(&weights.down, backend)?,
    })
}

fn prepare_moe<B: Backend>(
    weights: &LagunaMoeWeights,
    backend: &B,
) -> Result<LagunaDeviceMoeWeights> {
    Ok(LagunaDeviceMoeWeights {
        router: prepare_matrix(&weights.router, backend)?,
        correction_bias: weights.correction_bias.f32_values()?,
        shared: prepare_dense(&weights.shared, backend)?,
    })
}

fn prepare_matrix<B: Backend>(tensor: &SafeTensorHandle, backend: &B) -> Result<DeviceBf16Matrix> {
    let [rows, columns] = tensor.info().shape.as_slice() else {
        return Err(Error::weights(format!(
            "Laguna BF16 matrix {} must be rank 2, got {:?}",
            tensor.info().name,
            tensor.info().shape
        )));
    };
    tensor.advise_will_need()?;
    backend
        .prepare_bf16_matrix(tensor.bytes()?, *rows, *columns)?
        .ok_or_else(|| {
            Error::backend(format!(
                "native backend cannot prepare Laguna BF16 matrix {}",
                tensor.info().name
            ))
        })
}

fn prepare_vector(tensor: &SafeTensorHandle) -> Result<F32Tensor> {
    let [length] = tensor.info().shape.as_slice() else {
        return Err(Error::weights(format!(
            "Laguna BF16 vector {} must be rank 1, got {:?}",
            tensor.info().name,
            tensor.info().shape
        )));
    };
    F32Tensor::new(tensor.bf16_values_f32()?, [*length])
}

fn prepare_hidden_norm(tensor: &SafeTensorHandle) -> Result<F32Tensor> {
    let vector = prepare_vector(tensor)?;
    if vector.values().iter().any(|value| *value != 1.0) {
        return Err(Error::weights(format!(
            "Laguna hidden RMSNorm {} must contain unit weights after the checkpoint's offline R1 transform",
            tensor.info().name
        )));
    }
    Ok(vector)
}

fn prepare_scalar(tensor: &SafeTensorHandle) -> Result<f32> {
    let values = tensor.bf16_values_f32()?;
    let [value] = values.as_slice() else {
        return Err(Error::weights(format!(
            "Laguna BF16 scalar {} must contain one value, got {}",
            tensor.info().name,
            values.len()
        )));
    };
    if !value.is_finite() || *value <= 0.0 {
        return Err(Error::weights(format!(
            "Laguna quantization scale {} must be positive and finite, got {value}",
            tensor.info().name
        )));
    }
    Ok(*value)
}

fn prepare_resident_w4<B: Backend>(
    packed: &SafeTensorHandle,
    scales: &SafeTensorHandle,
    in_features: usize,
    out_features: usize,
    backend: &B,
) -> Result<DeviceW4Weight> {
    backend
        .prepare_w4_groupwise_weight(
            packed.bytes()?,
            scales.bytes()?,
            in_features,
            out_features,
            32,
        )?
        .ok_or_else(|| {
            Error::backend(format!(
                "native backend cannot prepare resident Laguna INT4 matrix {}",
                packed.info().name
            ))
        })
}

fn matrix_bytes(
    root: &LagunaDeviceRootWeights,
    layers: &[LagunaDeviceLayerWeights],
) -> Result<u64> {
    let mut bytes = root
        .embedding
        .storage_bytes()
        .checked_add(root.output.storage_bytes())
        .ok_or_else(|| Error::weights("Laguna root matrix byte count overflow"))?;
    for layer in layers {
        let attention = &layer.attention;
        for matrix in [
            &attention.query,
            &attention.key,
            &attention.value,
            &attention.output,
            &attention.gate,
        ] {
            bytes = bytes
                .checked_add(matrix.storage_bytes())
                .ok_or_else(|| Error::weights("Laguna attention matrix byte count overflow"))?;
        }
        match &layer.mlp {
            LagunaDeviceLayerMlpWeights::Dense(dense) => {
                bytes = add_dense_bytes(bytes, dense)?;
            }
            LagunaDeviceLayerMlpWeights::Moe(moe) => {
                bytes = bytes
                    .checked_add(moe.router.storage_bytes())
                    .ok_or_else(|| Error::weights("Laguna router matrix byte count overflow"))?;
                bytes = add_dense_bytes(bytes, &moe.shared)?;
            }
        }
    }
    u64::try_from(bytes).map_err(|_| Error::weights("Laguna matrix bytes do not fit u64"))
}

fn add_dense_bytes(mut bytes: usize, dense: &LagunaDeviceDenseWeights) -> Result<usize> {
    for matrix in [&dense.gate, &dense.up, &dense.down] {
        bytes = bytes
            .checked_add(matrix.storage_bytes())
            .ok_or_else(|| Error::weights("Laguna dense matrix byte count overflow"))?;
    }
    Ok(bytes)
}
