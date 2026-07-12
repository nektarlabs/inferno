#[cfg(test)]
use common::{Device, Tensor};
use common::{Error, F32Tensor, Result, Shape};
use gguf::{GgmlType, GgufFile, GgufQuantBlockKind};
use tracing::debug;

use crate::TensorRef;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorLoadReport {
    pub source_tensor_name: String,
    pub ty: GgmlType,
    pub shape: Shape,
    pub source_payload_bytes: u64,
    pub decoded_f32_bytes: u64,
}

#[cfg(test)]
#[derive(Debug)]
pub struct TensorLoadOutput {
    pub tensor: Tensor,
    pub report: TensorLoadReport,
}

#[derive(Debug)]
pub struct TensorLoadF32Output {
    pub tensor: F32Tensor,
    pub report: TensorLoadReport,
}

#[derive(Debug)]
pub struct WeightLoader<'a> {
    gguf: &'a GgufFile,
}

impl<'a> WeightLoader<'a> {
    pub fn new(gguf: &'a GgufFile) -> Self {
        Self { gguf }
    }

    #[cfg(test)]
    #[tracing::instrument(skip(self, tensor_ref, device), fields(tensor = %tensor_ref.name))]
    pub fn load_tensor_as_f32(
        &self,
        tensor_ref: &TensorRef,
        device: &Device,
    ) -> Result<TensorLoadOutput> {
        let loaded = self.load_tensor_as_f32_tensor(tensor_ref)?;
        let dims = loaded.tensor.dims().to_vec();
        let (_, values) = loaded.tensor.into_parts();
        let tensor = Tensor::from_vec(values, dims.as_slice(), device)?;

        Ok(TensorLoadOutput {
            tensor,
            report: loaded.report,
        })
    }

    #[tracing::instrument(skip(self, tensor_ref), fields(tensor = %tensor_ref.name))]
    pub fn load_tensor_as_f32_tensor(&self, tensor_ref: &TensorRef) -> Result<TensorLoadF32Output> {
        let info = self.gguf.tensor(&tensor_ref.name).ok_or_else(|| {
            Error::gguf(format!("missing GLM-5.2 GGUF tensor {}", tensor_ref.name))
        })?;
        validate_tensor_ref(tensor_ref, info)?;

        let shape = shape_from_gguf_dims(&tensor_ref.name, &tensor_ref.dims)?;
        let expected_values = shape.dims().iter().try_fold(1_usize, |acc, dim| {
            acc.checked_mul(*dim).ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} decoded element count overflow",
                    tensor_ref.name
                ))
            })
        })?;

        let (values, source_payload_bytes) = match tensor_ref.ty {
            GgmlType::F32 => {
                let values = self.gguf.tensor_f32_values(&tensor_ref.name)?;
                let bytes = u64::try_from(values.len())
                    .map_err(|_| {
                        Error::gguf(format!(
                            "GGUF tensor {} F32 element count does not fit u64",
                            tensor_ref.name
                        ))
                    })?
                    .checked_mul(4)
                    .ok_or_else(|| {
                        Error::gguf(format!(
                            "GGUF tensor {} F32 payload byte count overflow",
                            tensor_ref.name
                        ))
                    })?;
                (values, bytes)
            }
            GgmlType::Q2K => {
                let storage = self.gguf.tensor_quantized_storage(&tensor_ref.name)?;
                if storage.block != GgufQuantBlockKind::Q2K {
                    return Err(Error::gguf(format!(
                        "GGUF tensor {} expected Q2_K storage, got {}",
                        tensor_ref.name, storage.block
                    )));
                }
                (storage.dequantize_q2_k()?, storage.payload_byte_len)
            }
            GgmlType::Q8_0 => {
                let storage = self.gguf.tensor_quantized_storage(&tensor_ref.name)?;
                if storage.block != GgufQuantBlockKind::Q8_0 {
                    return Err(Error::gguf(format!(
                        "GGUF tensor {} expected Q8_0 storage, got {}",
                        tensor_ref.name, storage.block
                    )));
                }
                (storage.dequantize_q8_0()?, storage.payload_byte_len)
            }
            other => {
                return Err(Error::gguf(format!(
                    "GLM-5.2 Q2 GGUF runtime supports F32, Q2_K, and Q8_0 tensors only, got {other} for {}",
                    tensor_ref.name
                )));
            }
        };

        if values.len() != expected_values {
            return Err(Error::gguf(format!(
                "GGUF tensor {} decoded {} F32 values but shape {} requires {expected_values}",
                tensor_ref.name,
                values.len(),
                shape
            )));
        }
        let decoded_f32_bytes = u64::try_from(values.len())
            .map_err(|_| {
                Error::gguf(format!(
                    "GGUF tensor {} decoded element count does not fit u64",
                    tensor_ref.name
                ))
            })?
            .checked_mul(4)
            .ok_or_else(|| {
                Error::gguf(format!(
                    "GGUF tensor {} decoded F32 byte count overflow",
                    tensor_ref.name
                ))
            })?;

        let tensor = F32Tensor::new(values, shape.dims().to_vec())?;
        debug!(
            tensor = %tensor_ref.name,
            ty = %tensor_ref.ty,
            source_payload_bytes,
            decoded_f32_bytes,
            shape = %shape,
            "loaded GLM-5.2 GGUF tensor as F32"
        );

        Ok(TensorLoadF32Output {
            tensor,
            report: TensorLoadReport {
                source_tensor_name: tensor_ref.name.clone(),
                ty: tensor_ref.ty,
                shape,
                source_payload_bytes,
                decoded_f32_bytes,
            },
        })
    }
}

fn validate_tensor_ref(tensor_ref: &TensorRef, info: &gguf::GgufTensorInfo) -> Result<()> {
    if tensor_ref.dims != info.dims {
        return Err(Error::gguf(format!(
            "GGUF tensor {} dims changed from {:?} to {:?}",
            tensor_ref.name, tensor_ref.dims, info.dims
        )));
    }
    if tensor_ref.ty != info.ty {
        return Err(Error::gguf(format!(
            "GGUF tensor {} type changed from {} to {}",
            tensor_ref.name, tensor_ref.ty, info.ty
        )));
    }
    if tensor_ref.absolute_offset != info.absolute_offset {
        return Err(Error::gguf(format!(
            "GGUF tensor {} offset changed from {} to {}",
            tensor_ref.name, tensor_ref.absolute_offset, info.absolute_offset
        )));
    }
    if tensor_ref.storage_byte_len != info.storage_byte_len {
        return Err(Error::gguf(format!(
            "GGUF tensor {} storage length changed from {} to {}",
            tensor_ref.name, tensor_ref.storage_byte_len, info.storage_byte_len
        )));
    }
    Ok(())
}

fn shape_from_gguf_dims(name: &str, dims: &[u64]) -> Result<Shape> {
    let dims = dims
        .iter()
        .map(|dim| {
            usize::try_from(*dim).map_err(|_| {
                Error::gguf(format!(
                    "GGUF tensor {name} dimension {dim} does not fit usize"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Shape::new(dims))
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use common::Device;
    use gguf::{
        GgmlType, GgufFile, GgufMetadataValueType, GGML_Q2_K_BLOCK_BYTES, GGUF_MAGIC,
        GGUF_VERSION_V3,
    };

    use super::*;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn loads_raw_f32_tensor_without_touching_other_tensors() {
        let path = write_loader_fixture();
        let gguf = GgufFile::open(&path).unwrap();
        let loader = WeightLoader::new(&gguf);
        let tensor_ref = tensor_ref(&gguf, "output_norm.weight");

        let loaded = loader
            .load_tensor_as_f32(&tensor_ref, &Device::Cpu)
            .unwrap();
        let native = loader.load_tensor_as_f32_tensor(&tensor_ref).unwrap();

        assert_eq!(loaded.report.source_tensor_name, "output_norm.weight");
        assert_eq!(loaded.report.ty, GgmlType::F32);
        assert_eq!(loaded.report.shape.dims(), &[4]);
        assert_eq!(loaded.report.source_payload_bytes, 16);
        assert_eq!(loaded.report.decoded_f32_bytes, 16);
        assert_eq!(
            loaded.tensor.to_vec1::<f32>().unwrap(),
            vec![1.0, -2.0, 3.5, 0.25]
        );
        assert_eq!(native.tensor.dims(), &[4]);
        assert_eq!(native.tensor.values(), &[1.0, -2.0, 3.5, 0.25]);
    }

    #[test]
    fn loads_q2_k_tensor_as_f32_with_compressed_payload_report() {
        let path = write_loader_fixture();
        let gguf = GgufFile::open(&path).unwrap();
        let loader = WeightLoader::new(&gguf);
        let tensor_ref = tensor_ref(&gguf, "q2.weight");

        let loaded = loader
            .load_tensor_as_f32(&tensor_ref, &Device::Cpu)
            .unwrap();

        assert_eq!(loaded.report.ty, GgmlType::Q2K);
        assert_eq!(loaded.report.shape.dims(), &[256]);
        assert_eq!(loaded.report.source_payload_bytes, GGML_Q2_K_BLOCK_BYTES);
        assert_eq!(loaded.report.decoded_f32_bytes, 1024);
        let values = loaded.tensor.to_vec1::<f32>().unwrap();
        assert_eq!(values[0], 0.0);
        assert_eq!(values[16], -0.5);
        assert_eq!(values[240], -7.5);
    }

    #[test]
    fn rejects_stale_tensor_ref() {
        let path = write_loader_fixture();
        let gguf = GgufFile::open(&path).unwrap();
        let loader = WeightLoader::new(&gguf);
        let mut tensor_ref = tensor_ref(&gguf, "q2.weight");
        tensor_ref.dims = vec![128, 2];

        let err = loader
            .load_tensor_as_f32(&tensor_ref, &Device::Cpu)
            .expect_err("stale tensor ref should fail");

        assert!(err.to_string().contains("dims changed"));
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

    fn write_loader_fixture() -> PathBuf {
        let path = unique_temp_file("loader");
        fs::write(&path, tiny_loader_gguf()).unwrap();
        path
    }

    fn tiny_loader_gguf() -> Vec<u8> {
        let alignment = 32_usize;
        let mut writer = GgufWriter::new();
        writer.header(2, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(alignment as u32);
        writer.tensor_info("output_norm.weight", &[4], GgmlType::F32, 0);
        writer.tensor_info("q2.weight", &[256], GgmlType::Q2K, 32);
        writer.pad_to(alignment);

        for value in [1.0_f32, -2.0, 3.5, 0.25] {
            writer.bytes(&value.to_le_bytes());
        }
        writer.pad_to(alignment);
        writer.bytes(&q2_k_block(
            0x3c00,
            0x3800,
            [
                (1, 0),
                (2, 1),
                (3, 2),
                (4, 3),
                (5, 4),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 8),
                (10, 9),
                (11, 10),
                (12, 11),
                (13, 12),
                (14, 13),
                (15, 14),
                (0, 15),
            ],
            0xe4,
        ));
        writer.bytes(&[0_u8; 16]);
        writer.finish()
    }

    fn q2_k_block(d: u16, dmin: u16, scale_mins: [(u8, u8); 16], quant_byte: u8) -> Vec<u8> {
        let mut block = Vec::with_capacity(GGML_Q2_K_BLOCK_BYTES as usize);
        for (scale, min) in scale_mins {
            block.push((scale & 0x0f) | ((min & 0x0f) << 4));
        }
        block.extend(std::iter::repeat_n(quant_byte, 64));
        block.extend_from_slice(&d.to_le_bytes());
        block.extend_from_slice(&dmin.to_le_bytes());
        block
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

    fn unique_temp_file(label: &str) -> PathBuf {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("weight-loader-{label}-{}-{id}", std::process::id()))
    }
}
