use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use common::{Error, Result};
use config::Config;
use gguf::{GgmlType, GgufFile, GgufQuantBlockKind, GgufQuantizedTensorStorage};
use inferno_io::{ExpertPackHeader, EXPERT_PACK_HEADER_BYTES};

use crate::{FfnIndex, Index, PackedExpertsIndex, TensorRef};

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertPackReport {
    pub output_path: PathBuf,
    pub file_bytes: u64,
    pub record_count: u64,
    pub resumed_records: u64,
    pub already_complete: bool,
}

struct PackedLayerStorage<'a> {
    layer_index: usize,
    gate: GgufQuantizedTensorStorage<'a>,
    up: GgufQuantizedTensorStorage<'a>,
    down: GgufQuantizedTensorStorage<'a>,
}

pub fn expected_expert_pack_header(
    gguf: &GgufFile,
    config: &Config,
    index: &Index,
) -> Result<ExpertPackHeader> {
    let layers = packed_expert_layers(config, index)?;
    let first = layers
        .first()
        .ok_or_else(|| Error::weights("Q2 expert pack has no routed layers"))?;
    let first_layer = u32::try_from(first.0)
        .map_err(|_| Error::weights("Q2 expert-pack first layer exceeds u32"))?;
    let layer_count = u32::try_from(layers.len())
        .map_err(|_| Error::weights("Q2 expert-pack layer count exceeds u32"))?;
    let expert_count = u32::try_from(config.num_routed_experts)
        .map_err(|_| Error::weights("Q2 expert-pack expert count exceeds u32"))?;

    let mut expected_strides = None;
    for (_, packed) in &layers {
        let strides = packed_expert_strides(gguf, packed, config.num_routed_experts)?;
        match expected_strides {
            Some(expected) if expected != strides => {
                return Err(Error::weights(format!(
                    "Q2 expert-pack routed layers have inconsistent component strides: expected {expected:?}, got {strides:?}"
                )));
            }
            None => expected_strides = Some(strides),
            _ => {}
        }
    }
    let (gate_bytes, up_bytes, down_bytes) = expected_strides
        .ok_or_else(|| Error::weights("Q2 expert-pack has no component strides"))?;
    ExpertPackHeader::new(
        gguf.summary().file_size,
        expert_layout_fingerprint(gguf, config, &layers),
        first_layer,
        layer_count,
        expert_count,
        gate_bytes,
        up_bytes,
        down_bytes,
    )
}

pub fn create_expert_pack(
    gguf: &GgufFile,
    config: &Config,
    index: &Index,
    output_path: &Path,
) -> Result<ExpertPackReport> {
    let header = expected_expert_pack_header(gguf, config, index)?;
    let record_count = u64::from(header.layer_count)
        .checked_mul(u64::from(header.expert_count))
        .ok_or_else(|| Error::weights("Q2 expert-pack record count overflow"))?;

    if output_path.exists() {
        validate_existing_pack(output_path, header)?;
        return Ok(ExpertPackReport {
            output_path: output_path.to_path_buf(),
            file_bytes: header.expected_file_bytes()?,
            record_count,
            resumed_records: record_count,
            already_complete: true,
        });
    }

    let parent = output_path.parent().ok_or_else(|| {
        Error::weights(format!(
            "Q2 expert-pack output {} has no parent directory",
            output_path.display()
        ))
    })?;
    if !parent.is_dir() {
        return Err(Error::weights(format!(
            "Q2 expert-pack output directory {} does not exist",
            parent.display()
        )));
    }
    let partial_path = partial_path(output_path)?;
    let mut output = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&partial_path)
        .map_err(|source| Error::Io {
            path: partial_path.clone(),
            source,
        })?;
    let resumed_records = prepare_partial_file(&mut output, &partial_path, header)?;
    let layers = packed_layer_storage(gguf, config, index)?;
    output
        .seek(SeekFrom::Start(
            (EXPERT_PACK_HEADER_BYTES as u64)
                .checked_add(
                    resumed_records
                        .checked_mul(header.record_bytes()?)
                        .ok_or_else(|| Error::weights("Q2 expert-pack resume offset overflow"))?,
                )
                .ok_or_else(|| Error::weights("Q2 expert-pack resume offset overflow"))?,
        ))
        .map_err(|source| Error::Io {
            path: partial_path.clone(),
            source,
        })?;

    for record_index in resumed_records..record_count {
        let layer_slot = record_index / u64::from(header.expert_count);
        let expert_id = record_index % u64::from(header.expert_count);
        let layer = layers.get(layer_slot as usize).ok_or_else(|| {
            Error::weights(format!(
                "Q2 expert-pack layer slot {layer_slot} is out of bounds"
            ))
        })?;
        write_expert_component(
            &mut output,
            &partial_path,
            layer.gate.bytes,
            header.gate_bytes,
            expert_id,
        )?;
        write_expert_component(
            &mut output,
            &partial_path,
            layer.up.bytes,
            header.up_bytes,
            expert_id,
        )?;
        write_expert_component(
            &mut output,
            &partial_path,
            layer.down.bytes,
            header.down_bytes,
            expert_id,
        )?;

        if expert_id + 1 == u64::from(header.expert_count) {
            output.sync_data().map_err(|source| Error::Io {
                path: partial_path.clone(),
                source,
            })?;
            tracing::info!(
                target: "inferno::expert_pack",
                layer_index = layer.layer_index,
                completed_layers = layer_slot + 1,
                total_layers = header.layer_count,
                "packed routed Q2 expert layer"
            );
        }
    }

    output.sync_all().map_err(|source| Error::Io {
        path: partial_path.clone(),
        source,
    })?;
    let file_bytes = output
        .metadata()
        .map_err(|source| Error::Io {
            path: partial_path.clone(),
            source,
        })?
        .len();
    header.validate_file_bytes(file_bytes)?;
    drop(output);
    fs::rename(&partial_path, output_path).map_err(|source| Error::Io {
        path: output_path.to_path_buf(),
        source,
    })?;
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| Error::Io {
            path: parent.to_path_buf(),
            source,
        })?;

    Ok(ExpertPackReport {
        output_path: output_path.to_path_buf(),
        file_bytes,
        record_count,
        resumed_records,
        already_complete: false,
    })
}

fn packed_expert_layers<'a>(
    config: &Config,
    index: &'a Index,
) -> Result<Vec<(usize, &'a PackedExpertsIndex)>> {
    let mut layers = Vec::new();
    for layer in &index.layers {
        if let FfnIndex::SparseMoe { packed_experts, .. } = &layer.ffn {
            layers.push((layer.layer_index, packed_experts));
        }
    }
    if let Some(mtp) = &index.mtp {
        let FfnIndex::SparseMoe { packed_experts, .. } = &mtp.layer.ffn else {
            return Err(Error::weights(
                "Q2 expert-pack MTP layer must be sparse MoE",
            ));
        };
        layers.push((mtp.layer.layer_index, packed_experts));
    }

    let expected_count = config
        .num_layers
        .checked_sub(config.dense_layers)
        .and_then(|layers| layers.checked_add(config.num_nextn_predict_layers))
        .ok_or_else(|| Error::weights("Q2 expert-pack routed layer count overflow"))?;
    if layers.len() != expected_count {
        return Err(Error::weights(format!(
            "Q2 expert-pack expected {expected_count} routed layers, got {}",
            layers.len()
        )));
    }
    for (slot, (layer_index, _)) in layers.iter().enumerate() {
        let expected_layer = config
            .dense_layers
            .checked_add(slot)
            .ok_or_else(|| Error::weights("Q2 expert-pack layer index overflow"))?;
        if *layer_index != expected_layer {
            return Err(Error::weights(format!(
                "Q2 expert-pack routed layer {slot} has index {layer_index}; expected {expected_layer}"
            )));
        }
    }
    Ok(layers)
}

fn packed_layer_storage<'a>(
    gguf: &'a GgufFile,
    config: &Config,
    index: &'a Index,
) -> Result<Vec<PackedLayerStorage<'a>>> {
    packed_expert_layers(config, index)?
        .into_iter()
        .map(|(layer_index, packed)| {
            Ok(PackedLayerStorage {
                layer_index,
                gate: q2_storage(gguf, &packed.gate)?,
                up: q2_storage(gguf, &packed.up)?,
                down: q2_storage(gguf, &packed.down)?,
            })
        })
        .collect()
}

fn packed_expert_strides(
    gguf: &GgufFile,
    packed: &PackedExpertsIndex,
    expert_count: usize,
) -> Result<(u64, u64, u64)> {
    Ok((
        expert_stride(q2_storage(gguf, &packed.gate)?, expert_count)?,
        expert_stride(q2_storage(gguf, &packed.up)?, expert_count)?,
        expert_stride(q2_storage(gguf, &packed.down)?, expert_count)?,
    ))
}

fn q2_storage<'a>(
    gguf: &'a GgufFile,
    tensor: &TensorRef,
) -> Result<GgufQuantizedTensorStorage<'a>> {
    let storage = gguf.tensor_quantized_storage(&tensor.name)?;
    if storage.block != GgufQuantBlockKind::Q2K {
        return Err(Error::weights(format!(
            "Q2 expert-pack tensor {} must be Q2_K, got {}",
            tensor.name, tensor.ty
        )));
    }
    Ok(storage)
}

fn expert_stride(storage: GgufQuantizedTensorStorage<'_>, expert_count: usize) -> Result<u64> {
    let expert_count = u64::try_from(expert_count)
        .map_err(|_| Error::weights("Q2 expert-pack expert count exceeds u64"))?;
    if expert_count == 0 || storage.payload_byte_len % expert_count != 0 {
        return Err(Error::weights(format!(
            "Q2 expert-pack tensor {} payload {} is not divisible by expert count {expert_count}",
            storage.info.name, storage.payload_byte_len
        )));
    }
    Ok(storage.payload_byte_len / expert_count)
}

fn write_expert_component(
    output: &mut File,
    output_path: &Path,
    source: &[u8],
    stride: u64,
    expert_id: u64,
) -> Result<()> {
    let start = expert_id
        .checked_mul(stride)
        .ok_or_else(|| Error::weights("Q2 expert-pack source offset overflow"))?;
    let end = start
        .checked_add(stride)
        .ok_or_else(|| Error::weights("Q2 expert-pack source range overflow"))?;
    let start = usize::try_from(start)
        .map_err(|_| Error::weights("Q2 expert-pack source offset exceeds usize"))?;
    let end = usize::try_from(end)
        .map_err(|_| Error::weights("Q2 expert-pack source end exceeds usize"))?;
    let bytes = source.get(start..end).ok_or_else(|| {
        Error::weights(format!(
            "Q2 expert-pack source range {start}..{end} exceeds {} bytes",
            source.len()
        ))
    })?;
    output.write_all(bytes).map_err(|source| Error::Io {
        path: output_path.to_path_buf(),
        source,
    })
}

fn prepare_partial_file(
    file: &mut File,
    path: &Path,
    expected_header: ExpertPackHeader,
) -> Result<u64> {
    let file_bytes = file
        .metadata()
        .map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    if file_bytes == 0 {
        file.write_all(&expected_header.encode()?)
            .map_err(|source| Error::Io {
                path: path.to_path_buf(),
                source,
            })?;
        file.sync_data().map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        return Ok(0);
    }
    if file_bytes < EXPERT_PACK_HEADER_BYTES as u64 {
        return Err(Error::weights(format!(
            "partial Q2 expert-pack {} is shorter than its header",
            path.display()
        )));
    }
    file.seek(SeekFrom::Start(0)).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut encoded = [0_u8; EXPERT_PACK_HEADER_BYTES];
    file.read_exact(&mut encoded).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let actual_header = ExpertPackHeader::decode(&encoded)?;
    if actual_header != expected_header {
        return Err(Error::weights(format!(
            "partial Q2 expert-pack {} does not match the selected GGUF layout",
            path.display()
        )));
    }
    let data_bytes = file_bytes - EXPERT_PACK_HEADER_BYTES as u64;
    let record_bytes = expected_header.record_bytes()?;
    let complete_records = data_bytes / record_bytes;
    let complete_bytes = (EXPERT_PACK_HEADER_BYTES as u64)
        .checked_add(
            complete_records
                .checked_mul(record_bytes)
                .ok_or_else(|| Error::weights("Q2 expert-pack resume length overflow"))?,
        )
        .ok_or_else(|| Error::weights("Q2 expert-pack resume length overflow"))?;
    if complete_bytes > expected_header.expected_file_bytes()? {
        return Err(Error::weights(format!(
            "partial Q2 expert-pack {} exceeds the expected file size",
            path.display()
        )));
    }
    if complete_bytes != file_bytes {
        file.set_len(complete_bytes).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
    }
    Ok(complete_records)
}

fn validate_existing_pack(path: &Path, expected_header: ExpertPackHeader) -> Result<()> {
    let mut file = File::open(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut encoded = [0_u8; EXPERT_PACK_HEADER_BYTES];
    file.read_exact(&mut encoded).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let header = ExpertPackHeader::decode(&encoded)?;
    if header != expected_header {
        return Err(Error::weights(format!(
            "Q2 expert-pack {} does not match the selected GGUF layout",
            path.display()
        )));
    }
    let file_bytes = file
        .metadata()
        .map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?
        .len();
    header.validate_file_bytes(file_bytes)
}

fn partial_path(output_path: &Path) -> Result<PathBuf> {
    let file_name = output_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            Error::weights(format!(
                "Q2 expert-pack output {} has no UTF-8 file name",
                output_path.display()
            ))
        })?;
    Ok(output_path.with_file_name(format!("{file_name}.partial")))
}

fn expert_layout_fingerprint(
    gguf: &GgufFile,
    config: &Config,
    layers: &[(usize, &PackedExpertsIndex)],
) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    hash_u64(&mut hash, gguf.summary().file_size);
    hash_u64(&mut hash, config.hidden_size as u64);
    hash_u64(&mut hash, config.moe_intermediate_size as u64);
    hash_u64(&mut hash, config.num_routed_experts as u64);
    for (layer_index, packed) in layers {
        hash_u64(&mut hash, *layer_index as u64);
        for tensor in [&packed.gate, &packed.up, &packed.down] {
            hash_bytes(&mut hash, tensor.name.as_bytes());
            hash_u64(&mut hash, tensor_type_code(tensor.ty));
            hash_u64(&mut hash, tensor.absolute_offset);
            hash_u64(&mut hash, tensor.storage_byte_len);
            for &dim in &tensor.dims {
                hash_u64(&mut hash, dim);
            }
        }
    }
    hash
}

fn tensor_type_code(ty: GgmlType) -> u64 {
    match ty {
        GgmlType::F32 => 0,
        GgmlType::Q8_0 => 8,
        GgmlType::Q2K => 10,
        GgmlType::Q3K => 11,
        GgmlType::Q4K => 12,
        GgmlType::Q6K => 14,
        GgmlType::Unsupported(value) => u64::from(value),
    }
}

fn hash_u64(hash: &mut u64, value: u64) {
    hash_bytes(hash, &value.to_le_bytes());
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_path_keeps_the_final_name_visible() {
        assert_eq!(
            partial_path(Path::new(
                "/tmp/GLM-5.2-UD-Q2_K_RoutedQ2K-Inferno-ExpertPack-v1.q2pack",
            ))
            .unwrap(),
            PathBuf::from("/tmp/GLM-5.2-UD-Q2_K_RoutedQ2K-Inferno-ExpertPack-v1.q2pack.partial",)
        );
    }

    #[test]
    fn partial_file_truncates_an_incomplete_record_and_resumes() {
        let directory =
            std::env::temp_dir().join(format!("inferno-expert-pack-test-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("GLM-5.2-UD-Q2_K_RoutedQ2K-Inferno-ExpertPack-v1.q2pack.partial");
        let header = ExpertPackHeader::new(100, 1, 3, 1, 2, 4, 4, 4).unwrap();
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.write_all(&header.encode().unwrap()).unwrap();
        file.write_all(&[7_u8; 17]).unwrap();

        assert_eq!(prepare_partial_file(&mut file, &path, header).unwrap(), 1);
        assert_eq!(file.metadata().unwrap().len(), 4096 + 12);

        drop(file);
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }
}
