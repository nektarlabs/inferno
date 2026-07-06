use common::{Device, Tensor};
use common::{F32Tensor, Result, Shape};

use crate::LayerKind;

#[derive(Debug)]
pub struct LayerKvCacheTensors {
    pub layer_index: usize,
    pub layer_kind: LayerKind,
    pub cache_k: F32Tensor,
    pub cache_v: F32Tensor,
}

/// Output of a batched device-resident decode layer: the hidden states stay
/// on the GPU for the next layer; the current token's K/V come back to the
/// host because the paged KV cache appends there.
#[derive(Debug)]
pub(crate) struct BlockDeviceTensors {
    pub(crate) hidden_states: backend::DeviceValue,
    pub(crate) cache_k: F32Tensor,
    pub(crate) cache_v: F32Tensor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerKvCacheReport {
    pub layer_index: usize,
    pub layer_kind: LayerKind,
    pub cache_k_shape: Shape,
    pub cache_v_shape: Shape,
}

pub(crate) fn cache_tensor_from_host(tensor: &Tensor) -> Result<F32Tensor> {
    let dims = tensor.dims().to_vec();
    let values = tensor
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    F32Tensor::new(values, dims)
}

pub(crate) fn cache_tensor_to_host(tensor: &F32Tensor, device: &Device) -> Result<Tensor> {
    let dims = tensor.dims();
    let values = tensor.values().to_vec();
    match dims {
        [d0] => Ok(Tensor::from_vec(values, *d0, device)?),
        [d0, d1] => Ok(Tensor::from_vec(values, (*d0, *d1), device)?),
        [d0, d1, d2] => Ok(Tensor::from_vec(values, (*d0, *d1, *d2), device)?),
        [d0, d1, d2, d3] => Ok(Tensor::from_vec(values, (*d0, *d1, *d2, *d3), device)?),
        dims => Err(common::Error::model(format!(
            "unsupported cache tensor rank {} for host tensor bridge: {dims:?}",
            dims.len()
        ))),
    }
}
