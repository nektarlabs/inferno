use common::{Error, Result};
use config::Config;
use gguf::{GgmlType, GgufFile, GgufTensorInfo};

use crate::LayerKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Index {
    pub architecture: String,
    pub root: RootIndex,
    pub layers: Vec<LayerIndex>,
    pub summary: IndexSummary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootIndex {
    pub token_embedding: TensorRef,
    pub final_norm: TensorRef,
    pub output: TensorRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerIndex {
    pub layer_index: usize,
    pub kind: LayerKind,
    pub input_norm: TensorRef,
    pub post_attention_norm: TensorRef,
    pub attention: AttentionIndex,
    pub ffn: FfnIndex,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttentionIndex {
    pub q_a: TensorRef,
    pub q_a_norm: TensorRef,
    pub q_b: TensorRef,
    pub kv_a_mqa: TensorRef,
    pub kv_a_norm: TensorRef,
    pub k_b: TensorRef,
    pub v_b: TensorRef,
    pub output: TensorRef,
    pub indexer: Option<IndexerIndex>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexerIndex {
    pub k_norm_bias: TensorRef,
    pub k_norm_weight: TensorRef,
    pub proj: TensorRef,
    pub attn_k: TensorRef,
    pub attn_q_b: TensorRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FfnIndex {
    Dense(DenseFfnIndex),
    SparseMoe {
        router: TensorRef,
        router_correction_bias: TensorRef,
        shared_experts: SharedExpertIndex,
        packed_experts: PackedExpertsIndex,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseFfnIndex {
    pub gate: TensorRef,
    pub up: TensorRef,
    pub down: TensorRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedExpertIndex {
    pub gate: TensorRef,
    pub up: TensorRef,
    pub down: TensorRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedExpertsIndex {
    pub gate: TensorRef,
    pub up: TensorRef,
    pub down: TensorRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorRef {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    pub absolute_offset: u64,
    pub storage_byte_len: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexSummary {
    pub tensor_count: usize,
    pub metadata_kv_count: u64,
    pub dense_layer_count: usize,
    pub sparse_layer_count: usize,
    pub dsa_indexer_layer_count: usize,
    pub split_kv_b_projection_count: usize,
    pub packed_expert_tensor_count: usize,
    pub quantized_tensor_count: usize,
    pub raw_tensor_count: usize,
}

impl Index {
    pub fn from_gguf(gguf: &GgufFile, config: &Config) -> Result<Self> {
        if config.model_type != "glm_moe_dsa" {
            return Err(Error::gguf(format!(
                "GLM-5.2 GGUF index requires model_type glm_moe_dsa, got {}",
                config.model_type
            )));
        }

        let summary = gguf.summary();
        let architecture = summary
            .architecture
            .as_deref()
            .ok_or_else(|| Error::gguf("GGUF metadata is missing general.architecture"))?;
        if architecture != "glm-dsa" {
            return Err(Error::gguf(format!(
                "GLM-5.2 GGUF architecture must be glm-dsa, got {architecture}"
            )));
        }
        validate_q2_artifact_tensor_types(gguf)?;

        let root = RootIndex {
            token_embedding: required(gguf, "token_embd.weight")?,
            final_norm: required(gguf, "output_norm.weight")?,
            output: required(gguf, "output.weight")?,
        };

        let layers = (0..config.num_layers)
            .map(|layer_index| build_layer(gguf, config, layer_index))
            .collect::<Result<Vec<_>>>()?;

        let dense_layer_count = layers
            .iter()
            .filter(|layer| layer.kind == LayerKind::Dense)
            .count();
        let sparse_layer_count = layers.len().saturating_sub(dense_layer_count);
        let dsa_indexer_layer_count = layers
            .iter()
            .filter(|layer| layer.attention.indexer.is_some())
            .count();
        let split_kv_b_projection_count = layers.len() * 2;
        let packed_expert_tensor_count = layers
            .iter()
            .filter(|layer| matches!(layer.ffn, FfnIndex::SparseMoe { .. }))
            .count()
            * 3;
        let quantized_tensor_count = gguf
            .tensors()
            .iter()
            .filter(|tensor| tensor.ty.is_quantized())
            .count();
        let raw_tensor_count = gguf.tensors().len().saturating_sub(quantized_tensor_count);

        Ok(Self {
            architecture: architecture.to_string(),
            root,
            layers,
            summary: IndexSummary {
                tensor_count: gguf.tensors().len(),
                metadata_kv_count: summary.metadata_kv_count,
                dense_layer_count,
                sparse_layer_count,
                dsa_indexer_layer_count,
                split_kv_b_projection_count,
                packed_expert_tensor_count,
                quantized_tensor_count,
                raw_tensor_count,
            },
        })
    }
}

fn validate_q2_artifact_tensor_types(gguf: &GgufFile) -> Result<()> {
    for tensor in gguf.tensors() {
        match tensor.ty {
            GgmlType::F32 | GgmlType::Q2K | GgmlType::Q8_0 => {}
            other => {
                return Err(Error::gguf(format!(
                    "GLM-5.2 Q2 GGUF supports F32, Q2_K, and Q8_0 tensors only; tensor {} has {other}",
                    tensor.name
                )));
            }
        }
    }
    Ok(())
}

impl LayerIndex {
    pub fn is_sparse_moe(&self) -> bool {
        self.kind == LayerKind::SparseMoe
    }
}

impl TensorRef {
    fn from_info(info: &GgufTensorInfo) -> Self {
        Self {
            name: info.name.clone(),
            dims: info.dims.clone(),
            ty: info.ty,
            absolute_offset: info.absolute_offset,
            storage_byte_len: info.storage_byte_len,
        }
    }
}

fn build_layer(gguf: &GgufFile, config: &Config, layer_index: usize) -> Result<LayerIndex> {
    let kind = if layer_index < config.dense_layers {
        LayerKind::Dense
    } else {
        LayerKind::SparseMoe
    };
    let prefix = format!("blk.{layer_index}");

    Ok(LayerIndex {
        layer_index,
        kind,
        input_norm: required(gguf, format!("{prefix}.attn_norm.weight"))?,
        post_attention_norm: required(gguf, format!("{prefix}.ffn_norm.weight"))?,
        attention: build_attention(gguf, &prefix)?,
        ffn: build_ffn(gguf, &prefix, kind)?,
    })
}

fn build_attention(gguf: &GgufFile, prefix: &str) -> Result<AttentionIndex> {
    Ok(AttentionIndex {
        q_a: required(gguf, format!("{prefix}.attn_q_a.weight"))?,
        q_a_norm: required(gguf, format!("{prefix}.attn_q_a_norm.weight"))?,
        q_b: required(gguf, format!("{prefix}.attn_q_b.weight"))?,
        kv_a_mqa: required(gguf, format!("{prefix}.attn_kv_a_mqa.weight"))?,
        kv_a_norm: required(gguf, format!("{prefix}.attn_kv_a_norm.weight"))?,
        k_b: required(gguf, format!("{prefix}.attn_k_b.weight"))?,
        v_b: required(gguf, format!("{prefix}.attn_v_b.weight"))?,
        output: required(gguf, format!("{prefix}.attn_output.weight"))?,
        indexer: build_indexer(gguf, prefix)?,
    })
}

fn build_indexer(gguf: &GgufFile, prefix: &str) -> Result<Option<IndexerIndex>> {
    let names = [
        format!("{prefix}.indexer.k_norm.bias"),
        format!("{prefix}.indexer.k_norm.weight"),
        format!("{prefix}.indexer.proj.weight"),
        format!("{prefix}.indexer.attn_k.weight"),
        format!("{prefix}.indexer.attn_q_b.weight"),
    ];
    let present_count = names
        .iter()
        .filter(|name| gguf.tensor(name.as_str()).is_some())
        .count();
    if present_count == 0 {
        return Ok(None);
    }
    if present_count != names.len() {
        return Err(Error::gguf(format!(
            "incomplete GLM-5.2 GGUF DSA indexer tensors for {prefix}; expected all {} tensors",
            names.len()
        )));
    }

    Ok(Some(IndexerIndex {
        k_norm_bias: required(gguf, &names[0])?,
        k_norm_weight: required(gguf, &names[1])?,
        proj: required(gguf, &names[2])?,
        attn_k: required(gguf, &names[3])?,
        attn_q_b: required(gguf, &names[4])?,
    }))
}

fn build_ffn(gguf: &GgufFile, prefix: &str, kind: LayerKind) -> Result<FfnIndex> {
    match kind {
        LayerKind::Dense => Ok(FfnIndex::Dense(DenseFfnIndex {
            gate: required(gguf, format!("{prefix}.ffn_gate.weight"))?,
            up: required(gguf, format!("{prefix}.ffn_up.weight"))?,
            down: required(gguf, format!("{prefix}.ffn_down.weight"))?,
        })),
        LayerKind::SparseMoe => Ok(FfnIndex::SparseMoe {
            router: required(gguf, format!("{prefix}.ffn_gate_inp.weight"))?,
            router_correction_bias: required(gguf, format!("{prefix}.exp_probs_b.bias"))?,
            shared_experts: SharedExpertIndex {
                gate: required(gguf, format!("{prefix}.ffn_gate_shexp.weight"))?,
                up: required(gguf, format!("{prefix}.ffn_up_shexp.weight"))?,
                down: required(gguf, format!("{prefix}.ffn_down_shexp.weight"))?,
            },
            packed_experts: PackedExpertsIndex {
                gate: required(gguf, format!("{prefix}.ffn_gate_exps.weight"))?,
                up: required(gguf, format!("{prefix}.ffn_up_exps.weight"))?,
                down: required(gguf, format!("{prefix}.ffn_down_exps.weight"))?,
            },
        }),
    }
}

fn required(gguf: &GgufFile, name: impl AsRef<str>) -> Result<TensorRef> {
    let name = name.as_ref();
    let tensor = gguf
        .tensor(name)
        .ok_or_else(|| Error::gguf(format!("missing GLM-5.2 GGUF tensor {name}")))?;
    Ok(TensorRef::from_info(tensor))
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use gguf::{GgufMetadataValueType, GGUF_MAGIC, GGUF_VERSION_V3};

    use super::*;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn builds_index_with_dense_and_sparse_layers() {
        let path = write_tiny_gguf(false);
        let gguf = GgufFile::open(&path).unwrap();

        let index = Index::from_gguf(&gguf, &tiny_config()).unwrap();

        assert_eq!(index.architecture, "glm-dsa");
        assert_eq!(index.root.token_embedding.name, "token_embd.weight");
        assert_eq!(index.root.output.name, "output.weight");
        assert_eq!(index.layers.len(), 2);
        assert_eq!(index.layers[0].kind, LayerKind::Dense);
        assert_eq!(index.layers[1].kind, LayerKind::SparseMoe);
        assert!(index.layers[0].attention.indexer.is_some());
        assert!(index.layers[1].attention.indexer.is_none());
        assert_eq!(index.layers[0].attention.k_b.name, "blk.0.attn_k_b.weight");
        assert_eq!(index.layers[0].attention.v_b.name, "blk.0.attn_v_b.weight");
        assert_eq!(index.layers[0].attention.q_a.storage_byte_len, 32);

        let FfnIndex::SparseMoe { packed_experts, .. } = &index.layers[1].ffn else {
            panic!("layer 1 should be sparse MoE");
        };
        assert_eq!(packed_experts.gate.name, "blk.1.ffn_gate_exps.weight");
        assert_eq!(index.summary.dense_layer_count, 1);
        assert_eq!(index.summary.sparse_layer_count, 1);
        assert_eq!(index.summary.dsa_indexer_layer_count, 1);
        assert_eq!(index.summary.split_kv_b_projection_count, 4);
        assert_eq!(index.summary.packed_expert_tensor_count, 3);
        assert!(index.summary.quantized_tensor_count > 0);
    }

    #[test]
    fn rejects_missing_split_v_b_projection() {
        let path = write_tiny_gguf(true);
        let gguf = GgufFile::open(&path).unwrap();

        let err = Index::from_gguf(&gguf, &tiny_config())
            .expect_err("missing v_b projection should fail");

        assert!(err.to_string().contains("blk.0.attn_v_b.weight"));
    }

    #[test]
    fn rejects_non_glm_dsa_architecture() {
        let path = write_tiny_non_dsa_gguf();
        let gguf = GgufFile::open(&path).unwrap();

        let err =
            Index::from_gguf(&gguf, &tiny_config()).expect_err("non GLM architecture should fail");

        assert!(err.to_string().contains("architecture"));
    }

    #[test]
    fn rejects_non_q2_artifact_tensor_type() {
        let path = write_tiny_unsupported_quant_tensor_gguf();
        let gguf = GgufFile::open(&path).unwrap();

        let err = Index::from_gguf(&gguf, &tiny_config()).expect_err(
            "unsupported quantized tensors are outside the GLM-5.2 Q2 runtime contract",
        );

        assert!(err.to_string().contains("Q2 GGUF"));
        assert!(err.to_string().contains("token_embd.weight"));
        assert!(err.to_string().contains("UNSUPPORTED_GGML_TYPE_12"));
    }

    fn write_tiny_gguf(omit_v_b: bool) -> PathBuf {
        let path = unique_temp_file("inferno");
        let mut writer = GgufWriter::new();
        writer.header(0, 2);
        writer.metadata_string("general.architecture", "glm-dsa");
        writer.metadata_u32("general.alignment", 32);

        writer.tensor("token_embd.weight", &[4, 4], GgmlType::Q2K);
        writer.tensor("output_norm.weight", &[4], GgmlType::F32);
        writer.tensor("output.weight", &[4, 4], GgmlType::Q2K);
        insert_dense_layer(&mut writer, 0, omit_v_b);
        insert_sparse_layer(&mut writer, 1);
        writer.finish_to(&path);
        path
    }

    fn write_tiny_unsupported_quant_tensor_gguf() -> PathBuf {
        let path = unique_temp_file("unsupported-quant");
        let mut writer = GgufWriter::new();
        writer.header(0, 2);
        writer.metadata_string("general.architecture", "glm-dsa");
        writer.metadata_u32("general.alignment", 32);
        writer.tensor("token_embd.weight", &[4, 4], GgmlType::Unsupported(12));
        writer.finish_to(&path);
        path
    }

    fn write_tiny_non_dsa_gguf() -> PathBuf {
        let path = unique_temp_file("not-glm");
        let mut writer = GgufWriter::new();
        writer.header(0, 2);
        writer.metadata_string("general.architecture", "llama");
        writer.metadata_u32("general.alignment", 32);
        writer.tensor("token_embd.weight", &[4, 4], GgmlType::Q2K);
        writer.finish_to(&path);
        path
    }

    fn insert_dense_layer(writer: &mut GgufWriter, layer: usize, omit_v_b: bool) {
        insert_attention(writer, layer, true, omit_v_b);
        writer.tensor(
            &format!("blk.{layer}.ffn_gate.weight"),
            &[4, 3],
            GgmlType::Q2K,
        );
        writer.tensor(
            &format!("blk.{layer}.ffn_up.weight"),
            &[4, 3],
            GgmlType::Q2K,
        );
        writer.tensor(
            &format!("blk.{layer}.ffn_down.weight"),
            &[3, 4],
            GgmlType::Q2K,
        );
    }

    fn insert_sparse_layer(writer: &mut GgufWriter, layer: usize) {
        insert_attention(writer, layer, false, false);
        writer.tensor(
            &format!("blk.{layer}.exp_probs_b.bias"),
            &[2],
            GgmlType::F32,
        );
        writer.tensor(
            &format!("blk.{layer}.ffn_gate_inp.weight"),
            &[4, 2],
            GgmlType::Q2K,
        );
        writer.tensor(
            &format!("blk.{layer}.ffn_gate_shexp.weight"),
            &[4, 3],
            GgmlType::Q2K,
        );
        writer.tensor(
            &format!("blk.{layer}.ffn_up_shexp.weight"),
            &[4, 3],
            GgmlType::Q2K,
        );
        writer.tensor(
            &format!("blk.{layer}.ffn_down_shexp.weight"),
            &[3, 4],
            GgmlType::Q2K,
        );
        writer.tensor(
            &format!("blk.{layer}.ffn_gate_exps.weight"),
            &[2, 4, 3],
            GgmlType::Q2K,
        );
        writer.tensor(
            &format!("blk.{layer}.ffn_up_exps.weight"),
            &[2, 4, 3],
            GgmlType::Q2K,
        );
        writer.tensor(
            &format!("blk.{layer}.ffn_down_exps.weight"),
            &[2, 3, 4],
            GgmlType::Q2K,
        );
    }

    fn insert_attention(writer: &mut GgufWriter, layer: usize, indexer: bool, omit_v_b: bool) {
        writer.tensor(
            &format!("blk.{layer}.attn_norm.weight"),
            &[4],
            GgmlType::F32,
        );
        writer.tensor(&format!("blk.{layer}.ffn_norm.weight"), &[4], GgmlType::F32);
        writer.tensor(
            &format!("blk.{layer}.attn_q_a.weight"),
            &[4, 4],
            GgmlType::Q2K,
        );
        writer.tensor(
            &format!("blk.{layer}.attn_q_a_norm.weight"),
            &[4],
            GgmlType::F32,
        );
        writer.tensor(
            &format!("blk.{layer}.attn_q_b.weight"),
            &[4, 4],
            GgmlType::Q2K,
        );
        writer.tensor(
            &format!("blk.{layer}.attn_kv_a_mqa.weight"),
            &[4, 4],
            GgmlType::Q2K,
        );
        writer.tensor(
            &format!("blk.{layer}.attn_kv_a_norm.weight"),
            &[4],
            GgmlType::F32,
        );
        writer.tensor(
            &format!("blk.{layer}.attn_k_b.weight"),
            &[4, 4],
            GgmlType::Q2K,
        );
        if !omit_v_b {
            writer.tensor(
                &format!("blk.{layer}.attn_v_b.weight"),
                &[4, 4],
                GgmlType::Q2K,
            );
        }
        writer.tensor(
            &format!("blk.{layer}.attn_output.weight"),
            &[4, 4],
            GgmlType::Q2K,
        );

        if indexer {
            writer.tensor(
                &format!("blk.{layer}.indexer.k_norm.bias"),
                &[4],
                GgmlType::F32,
            );
            writer.tensor(
                &format!("blk.{layer}.indexer.k_norm.weight"),
                &[4],
                GgmlType::F32,
            );
            writer.tensor(
                &format!("blk.{layer}.indexer.proj.weight"),
                &[4, 4],
                GgmlType::Q2K,
            );
            writer.tensor(
                &format!("blk.{layer}.indexer.attn_k.weight"),
                &[4, 4],
                GgmlType::Q2K,
            );
            writer.tensor(
                &format!("blk.{layer}.indexer.attn_q_b.weight"),
                &[4, 4],
                GgmlType::Q2K,
            );
        }
    }

    struct GgufWriter {
        metadata_count: u64,
        tensors: Vec<TestTensor>,
        bytes: Vec<u8>,
    }

    struct TestTensor {
        name: String,
        dims: Vec<u64>,
        ty: GgmlType,
    }

    impl GgufWriter {
        fn new() -> Self {
            Self {
                metadata_count: 0,
                tensors: Vec::new(),
                bytes: Vec::new(),
            }
        }

        fn header(&mut self, _tensor_count: u64, metadata_count: u64) {
            self.bytes.extend_from_slice(GGUF_MAGIC);
            self.u32(GGUF_VERSION_V3);
            self.u64(0);
            self.u64(metadata_count);
            self.metadata_count = metadata_count;
        }

        fn metadata_string(&mut self, key: &str, value: &str) {
            self.string(key);
            self.u32(GgufMetadataValueType::String as u32);
            self.string(value);
        }

        fn metadata_u32(&mut self, key: &str, value: u32) {
            self.string(key);
            self.u32(GgufMetadataValueType::Uint32 as u32);
            self.u32(value);
        }

        fn tensor(&mut self, name: &str, dims: &[u64], ty: GgmlType) {
            self.tensors.push(TestTensor {
                name: name.to_string(),
                dims: dims.to_vec(),
                ty,
            });
        }

        fn finish_to(mut self, path: &PathBuf) {
            let tensor_count = self.tensors.len() as u64;
            self.bytes[8..16].copy_from_slice(&tensor_count.to_le_bytes());

            let mut offset = 0_u64;
            let tensors = std::mem::take(&mut self.tensors);
            for tensor in tensors {
                self.string(&tensor.name);
                self.u32(tensor.dims.len() as u32);
                for dim in tensor.dims {
                    self.u64(dim);
                }
                self.u32(tensor.ty.code());
                self.u64(offset);
                offset += 32;
            }

            self.pad_to(32);
            self.bytes
                .resize(self.bytes.len() + offset as usize + 32, 0);
            fs::write(path, self.bytes).unwrap();
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
    }

    fn tiny_config() -> Config {
        Config {
            model_type: "glm_moe_dsa".to_string(),
            hidden_size: 4,
            num_layers: 2,
            dense_layers: 1,
            sparse_moe_layers: Some(1),
            vocab_size: 8,
            attention_heads: 2,
            qk_head_dim: 4,
            qk_no_rope_dim: 2,
            qk_rope_dim: 2,
            v_head_dim: Some(4),
            num_routed_experts: 2,
            experts_per_token: 1,
            max_context: 16,
            dsa_index_topk: 4,
            moe_intermediate_size: 3,
            num_shared_experts: 1,
            moe_groups: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 2.5,
            scoring_func: "sigmoid".to_string(),
            topk_method: "noaux_tc".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
        .validated()
        .unwrap()
    }

    fn unique_temp_file(label: &str) -> PathBuf {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("index-{label}-{}-{id}", std::process::id()))
    }
}
