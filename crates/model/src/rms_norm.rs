use backend::{Backend, BackendCapabilities};
#[cfg(test)]
use common::Shape;
use common::Tensor;
use common::{validate_exact_shape, Error, F32Tensor, Result};
use config::Config;
use gguf::{GgmlType, GgufFile};

use crate::{TensorLoadReport, TensorRef, WeightLoader};

#[derive(Debug)]
pub struct RmsNorm {
    #[cfg(test)]
    tensor_name: String,
    hidden_size: usize,
    eps: f32,
    weight: F32Tensor,
    load_report: RmsNormLoadReport,
}

#[derive(Debug)]
pub struct RmsNormOutput {
    pub hidden_states: Tensor,
    #[cfg(test)]
    pub report: RmsNormForwardReport,
}

#[derive(Debug)]
pub struct RmsNormF32Output {
    pub hidden_states: F32Tensor,
    #[cfg(test)]
    pub report: RmsNormForwardReport,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RmsNormLoadReport {
    pub source_tensor_name: String,
    pub backend: BackendCapabilities,
    pub weight: TensorLoadReport,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub struct RmsNormForwardReport {
    pub source_tensor_name: String,
    pub input_shape: Shape,
    pub weight_shape: Shape,
    pub output_shape: Shape,
}

impl RmsNorm {
    pub fn open<B: Backend>(
        gguf: &GgufFile,
        config: &Config,
        tensor_ref: &TensorRef,
        backend: &B,
    ) -> Result<Self> {
        Self::open_with_expected_size(
            gguf,
            tensor_ref,
            config.hidden_size,
            config.rms_norm_eps as f32,
            backend,
        )
    }

    pub fn open_with_expected_size<B: Backend>(
        gguf: &GgufFile,
        tensor_ref: &TensorRef,
        hidden_size: usize,
        eps: f32,
        backend: &B,
    ) -> Result<Self> {
        if tensor_ref.ty != GgmlType::F32 {
            return Err(Error::gguf(format!(
                "GLM-5.2 GGUF RMSNorm tensor {} must be F32, got {}",
                tensor_ref.name, tensor_ref.ty
            )));
        }
        let loader = WeightLoader::new(gguf);
        let loaded = loader.load_tensor_as_f32_tensor(tensor_ref)?;
        validate_exact_shape(
            "gguf_rms_norm_weight_shape",
            loaded.report.shape.dims(),
            &[hidden_size],
        )?;

        let load_report = RmsNormLoadReport {
            source_tensor_name: tensor_ref.name.clone(),
            backend: backend.capabilities(),
            weight: loaded.report,
        };

        Ok(Self {
            #[cfg(test)]
            tensor_name: tensor_ref.name.clone(),
            hidden_size,
            eps,
            weight: loaded.tensor,
            load_report,
        })
    }

    pub fn forward<B: Backend>(
        &self,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<RmsNormOutput> {
        let output = self.forward_tensor(hidden_states, backend)?;
        Ok(RmsNormOutput {
            #[cfg(test)]
            report: RmsNormForwardReport {
                source_tensor_name: self.tensor_name.clone(),
                input_shape: Shape::new(hidden_states.dims().to_vec()),
                weight_shape: Shape::new(self.weight.dims().to_vec()),
                output_shape: Shape::new(output.dims().to_vec()),
            },
            hidden_states: output,
        })
    }

    pub fn forward_tensor<B: Backend>(
        &self,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<Tensor> {
        let input = tensor_to_f32_tensor(hidden_states)?;
        let output = self.forward_f32(&input, backend)?;
        let dims = output.hidden_states.dims().to_vec();
        validate_exact_shape("gguf_rms_norm_output_rank", &[dims.len()], &[3])?;
        let (_, values) = output.hidden_states.into_parts();

        Tensor::from_vec(values, (dims[0], dims[1], dims[2]), backend.device()).map_err(Into::into)
    }

    pub fn forward_f32<B: Backend>(
        &self,
        hidden_states: &F32Tensor,
        backend: &B,
    ) -> Result<RmsNormF32Output> {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF RMSNorm input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        validate_exact_shape("gguf_rms_norm_hidden_size", &[dims[2]], &[self.hidden_size])?;

        let output = backend.rms_norm_f32(hidden_states, &self.weight, self.eps)?;
        Ok(RmsNormF32Output {
            #[cfg(test)]
            report: RmsNormForwardReport {
                source_tensor_name: self.tensor_name.clone(),
                input_shape: Shape::new(dims.to_vec()),
                weight_shape: Shape::new(self.weight.dims().to_vec()),
                output_shape: Shape::new(output.dims().to_vec()),
            },
            hidden_states: output,
        })
    }

    /// Batched device-resident variant of `forward_f32`: encodes the RMSNorm
    /// kernel into the backend's open batch without synchronizing. Returns
    /// `Ok(None)` when the backend has no device-resident path.
    pub(crate) fn forward_device<B: Backend>(
        &self,
        hidden_states: &backend::DeviceValue,
        backend: &B,
    ) -> Result<Option<backend::DeviceValue>> {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::model(format!(
                "GLM-5.2 GGUF RMSNorm device input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        validate_exact_shape("gguf_rms_norm_hidden_size", &[dims[2]], &[self.hidden_size])?;

        backend.rms_norm_device(hidden_states, &self.weight, self.eps)
    }

    pub fn load_report(&self) -> &RmsNormLoadReport {
        &self.load_report
    }

    pub(crate) fn weight(&self) -> &F32Tensor {
        &self.weight
    }

    pub(crate) fn eps(&self) -> f32 {
        self.eps
    }
}

fn tensor_to_f32_tensor(tensor: &Tensor) -> Result<F32Tensor> {
    let dims = tensor.dims().to_vec();
    let values = tensor
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    F32Tensor::new(values, dims)
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use backend::MetalBackend;
    use common::{Device, Tensor};
    use config::Config;
    use gguf::{GgmlType, GgufFile, GgufMetadataValueType, GGUF_MAGIC, GGUF_VERSION_V3};

    use super::*;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn loads_and_runs_f32_rms_norm() {
        let path = write_norm_fixture(GgmlType::F32, &[4], &[1.0, 1.5, 0.5, 2.0]);
        let gguf = GgufFile::open(&path).unwrap();
        let tensor_ref = tensor_ref(&gguf, "output_norm.weight");
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let norm = RmsNorm::open(&gguf, &tiny_config(4), &tensor_ref, &backend).unwrap();
        let hidden_states =
            Tensor::from_vec(vec![0.0_f32; 1 * 2 * 4], (1, 2, 4), &Device::Cpu).unwrap();

        let output = norm.forward(&hidden_states, &backend).unwrap();
        let tensor_only = norm.forward_tensor(&hidden_states, &backend).unwrap();
        let native_input = F32Tensor::new(vec![0.0_f32; 1 * 2 * 4], [1, 2, 4]).unwrap();
        let native_output = norm.forward_f32(&native_input, &backend).unwrap();

        assert_eq!(norm.load_report().source_tensor_name, "output_norm.weight");
        assert_eq!(norm.load_report().weight.source_payload_bytes, 16);
        assert_eq!(output.report.input_shape.dims(), &[1, 2, 4]);
        assert_eq!(output.report.weight_shape.dims(), &[4]);
        assert_eq!(output.report.output_shape.dims(), &[1, 2, 4]);
        assert_eq!(native_output.report.output_shape.dims(), &[1, 2, 4]);
        assert_eq!(native_output.hidden_states.dims(), &[1, 2, 4]);
        assert_eq!(
            output.hidden_states.to_vec3::<f32>().unwrap()[0][0],
            vec![0.0; 4]
        );
        assert_eq!(
            tensor_only.to_vec3::<f32>().unwrap(),
            output.hidden_states.to_vec3::<f32>().unwrap()
        );
    }

    #[test]
    fn rejects_quantized_norm_tensor() {
        let path = write_norm_fixture(GgmlType::Q2K, &[256], &[]);
        let gguf = GgufFile::open(&path).unwrap();
        let tensor_ref = tensor_ref(&gguf, "output_norm.weight");
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();

        let err = RmsNorm::open(&gguf, &tiny_config(256), &tensor_ref, &backend)
            .expect_err("quantized norm should not load");

        assert!(err.to_string().contains("must be F32"));
    }

    #[test]
    fn rejects_wrong_hidden_size() {
        let path = write_norm_fixture(GgmlType::F32, &[3], &[1.0, 1.0, 1.0]);
        let gguf = GgufFile::open(&path).unwrap();
        let tensor_ref = tensor_ref(&gguf, "output_norm.weight");
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();

        let err = RmsNorm::open(&gguf, &tiny_config(4), &tensor_ref, &backend)
            .expect_err("wrong norm shape should fail");

        assert!(err.to_string().contains("gguf_rms_norm_weight_shape"));
    }

    #[test]
    fn rejects_wrong_input_rank() {
        let path = write_norm_fixture(GgmlType::F32, &[4], &[1.0, 1.0, 1.0, 1.0]);
        let gguf = GgufFile::open(&path).unwrap();
        let tensor_ref = tensor_ref(&gguf, "output_norm.weight");
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let norm = RmsNorm::open(&gguf, &tiny_config(4), &tensor_ref, &backend).unwrap();
        let hidden_states = Tensor::from_vec(vec![0.0_f32; 4], 4, &Device::Cpu).unwrap();

        let err = norm
            .forward(&hidden_states, &backend)
            .expect_err("rank-1 hidden states should fail");

        assert!(err.to_string().contains("rank 3"));
    }

    fn tensor_ref(gguf: &GgufFile, name: &str) -> TensorRef {
        let info = gguf.tensor(name).unwrap();
        TensorRef {
            name: info.name.clone(),
            dims: info.dims.clone(),
            ty: info.ty,
            absolute_offset: info.absolute_offset,
            storage_byte_len: info.storage_byte_len,
        }
    }

    fn write_norm_fixture(ty: GgmlType, dims: &[u64], f32_values: &[f32]) -> PathBuf {
        let path = unique_temp_file("rms-norm");
        fs::write(&path, tiny_norm_gguf(ty, dims, f32_values)).unwrap();
        path
    }

    fn tiny_norm_gguf(ty: GgmlType, dims: &[u64], f32_values: &[f32]) -> Vec<u8> {
        let mut writer = GgufWriter::new();
        writer.header(1, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);
        writer.tensor_info("output_norm.weight", dims, ty, 0);
        writer.pad_to(32);
        match ty {
            GgmlType::F32 => {
                for value in f32_values {
                    writer.bytes(&value.to_le_bytes());
                }
            }
            GgmlType::Q2K => writer.bytes(&vec![0_u8; 84]),
            other => panic!("unsupported fixture type {other}"),
        }
        writer.bytes(&[0_u8; 32]);
        writer.finish()
    }

    struct GgufWriter {
        bytes: Vec<u8>,
    }

    impl GgufWriter {
        fn new() -> Self {
            Self { bytes: Vec::new() }
        }

        fn header(&mut self, tensor_count: u64, metadata_kv_count: u64) {
            self.bytes.extend_from_slice(GGUF_MAGIC);
            self.u32(GGUF_VERSION_V3);
            self.u64(tensor_count);
            self.u64(metadata_kv_count);
        }

        fn metadata_key(&mut self, key: &str) {
            self.string(key);
        }

        fn tensor_info(&mut self, name: &str, dims: &[u64], ty: GgmlType, offset: u64) {
            self.string(name);
            self.u32(dims.len() as u32);
            for dim in dims {
                self.u64(*dim);
            }
            self.u32(ty.code());
            self.u64(offset);
        }

        fn string(&mut self, value: &str) {
            self.u64(value.len() as u64);
            self.bytes.extend_from_slice(value.as_bytes());
        }

        fn u32(&mut self, value: u32) {
            self.bytes.extend_from_slice(&value.to_le_bytes());
        }

        fn u64(&mut self, value: u64) {
            self.bytes.extend_from_slice(&value.to_le_bytes());
        }

        fn pad_to(&mut self, alignment: usize) {
            let remainder = self.bytes.len() % alignment;
            if remainder != 0 {
                self.bytes
                    .resize(self.bytes.len() + alignment - remainder, 0);
            }
        }

        fn bytes(&mut self, bytes: &[u8]) {
            self.bytes.extend_from_slice(bytes);
        }

        fn finish(self) -> Vec<u8> {
            self.bytes
        }
    }

    fn tiny_config(hidden_size: usize) -> Config {
        Config {
            model_type: "glm_moe_dsa".to_string(),
            hidden_size,
            num_layers: 1,
            dense_layers: 1,
            sparse_moe_layers: Some(0),
            vocab_size: 4,
            attention_heads: 2,
            qk_head_dim: 4,
            qk_no_rope_dim: 2,
            qk_rope_dim: 2,
            kv_lora_rank: 2,
            v_head_dim: Some(4),
            num_routed_experts: 2,
            experts_per_token: 1,
            max_context: 16,
            dsa_index_topk: 4,
            index_head_dim: 128,
            index_n_heads: 32,
            index_topk_freq: 4,
            indexer_rope_interleave: true,
            indexer_types: Vec::new(),
            moe_intermediate_size: 3,
            num_shared_experts: 1,
            moe_groups: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 2.5,
            scoring_func: "sigmoid".to_string(),
            topk_method: "noaux_tc".to_string(),
            num_nextn_predict_layers: 0,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
        .validated()
        .unwrap()
    }

    fn unique_temp_file(label: &str) -> PathBuf {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("inferno-{label}-{}-{id}", std::process::id()))
    }
}
