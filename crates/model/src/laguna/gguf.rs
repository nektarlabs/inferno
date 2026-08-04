use std::{collections::HashSet, path::Path};

use ::gguf::{
    GgmlType, GgufFile, GgufMetadataValue, GgufQuantizedTensorStorage, GgufTensorAdvice,
    GgufTensorInfo, GgufTensorStorage,
};
use common::{Error, Result};
use config::{LagunaConfig, LagunaProfile};
use inferno_io::MappedBytes;

pub const LAGUNA_GGUF_REPO_ID: &str = "antirez/Laguna-S-2.1-GGUF";
pub const LAGUNA_GGUF_REPO_URL: &str = "https://huggingface.co/antirez/Laguna-S-2.1-GGUF";
pub const LAGUNA_GGUF_FILE_NAME: &str = "laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf";
pub const LAGUNA_GGUF_FILE_BYTES: u64 = 48_260_803_968;
pub const LAGUNA_GGUF_SHA256: &str =
    "61fc66596597985cb9408a8530de6322d9e0d5b1d2ad4ed6503938018e0ce903";

pub const LAGUNA_XS_GGUF_REPO_ID: &str = "poolside/Laguna-XS-2.1-GGUF";
pub const LAGUNA_XS_GGUF_REPO_URL: &str = "https://huggingface.co/poolside/Laguna-XS-2.1-GGUF";
pub const LAGUNA_XS_GGUF_FILE_NAME: &str = "Laguna-XS-2.1-Q4_K_M.gguf";
pub const LAGUNA_XS_GGUF_FILE_BYTES: u64 = 20_274_300_032;
pub const LAGUNA_XS_GGUF_SHA256: &str =
    "1ac7079101fca5a6df8c5a7523a3c30ea7d1c0e4b1258090e7d6d4039287f6cb";

const FIRST_Q3_LAYER: usize = 21;
const LAGUNA_XS_Q6_LAYERS: [usize; 20] = [
    0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 35, 36, 37, 38, 39,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LagunaGgufFlavor {
    S21Q2Q3,
    Xs21Q4KM,
}

impl LagunaGgufFlavor {
    pub const fn file_name(self) -> &'static str {
        match self {
            Self::S21Q2Q3 => LAGUNA_GGUF_FILE_NAME,
            Self::Xs21Q4KM => LAGUNA_XS_GGUF_FILE_NAME,
        }
    }

    const fn container(self) -> LagunaGgufContainer {
        match self {
            Self::S21Q2Q3 => LagunaGgufContainer {
                file_bytes: LAGUNA_GGUF_FILE_BYTES,
                tensor_count: 814,
                metadata_count: 59,
                tensor_data_offset: 3_733_888,
                file_type: 10,
            },
            Self::Xs21Q4KM => LagunaGgufContainer {
                file_bytes: LAGUNA_XS_GGUF_FILE_BYTES,
                tensor_count: 678,
                metadata_count: 56,
                tensor_data_offset: 3_725_440,
                file_type: 15,
            },
        }
    }

    const fn root_types(self) -> (GgmlType, GgmlType) {
        match self {
            Self::S21Q2Q3 => (GgmlType::Q8_0, GgmlType::Q8_0),
            Self::Xs21Q4KM => (GgmlType::Q4K, GgmlType::Q6K),
        }
    }

    const fn attention_type(self, layer_index: usize, value: bool) -> GgmlType {
        match self {
            Self::S21Q2Q3 => GgmlType::Q8_0,
            Self::Xs21Q4KM if value && laguna_xs_uses_q6(layer_index) => GgmlType::Q6K,
            Self::Xs21Q4KM => GgmlType::Q4K,
        }
    }

    const fn dense_types(self, layer_index: usize) -> (GgmlType, GgmlType) {
        match self {
            Self::S21Q2Q3 => (GgmlType::Q8_0, GgmlType::Q8_0),
            Self::Xs21Q4KM if laguna_xs_uses_q6(layer_index) => (GgmlType::Q4K, GgmlType::Q6K),
            Self::Xs21Q4KM => (GgmlType::Q4K, GgmlType::Q4K),
        }
    }

    const fn routed_types(self, layer_index: usize) -> (GgmlType, GgmlType) {
        match self {
            Self::S21Q2Q3 if layer_index < FIRST_Q3_LAYER => (GgmlType::Q2K, GgmlType::Q2K),
            Self::S21Q2Q3 => (GgmlType::Q3K, GgmlType::Q3K),
            Self::Xs21Q4KM if laguna_xs_uses_q6(layer_index) => (GgmlType::Q4K, GgmlType::Q6K),
            Self::Xs21Q4KM => (GgmlType::Q4K, GgmlType::Q4K),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LagunaGgufContainer {
    file_bytes: u64,
    tensor_count: u64,
    metadata_count: u64,
    tensor_data_offset: u64,
    file_type: u64,
}

const fn laguna_xs_uses_q6(layer_index: usize) -> bool {
    let mut index = 0;
    while index < LAGUNA_XS_Q6_LAYERS.len() {
        if LAGUNA_XS_Q6_LAYERS[index] == layer_index {
            return true;
        }
        index += 1;
    }
    false
}

fn flavor_for_config(config: &LagunaConfig) -> Result<LagunaGgufFlavor> {
    match config.profile()? {
        LagunaProfile::S21 => Ok(LagunaGgufFlavor::S21Q2Q3),
        LagunaProfile::Xs21 => Ok(LagunaGgufFlavor::Xs21Q4KM),
    }
}

#[derive(Debug, Clone)]
pub struct LagunaGgufRoot {
    pub embedding: GgufTensorInfo,
    pub final_norm: GgufTensorInfo,
    pub output: GgufTensorInfo,
}

#[derive(Debug, Clone)]
pub struct LagunaGgufAttention {
    pub input_norm: GgufTensorInfo,
    pub query: GgufTensorInfo,
    pub key: GgufTensorInfo,
    pub value: GgufTensorInfo,
    pub gate: GgufTensorInfo,
    pub query_norm: GgufTensorInfo,
    pub key_norm: GgufTensorInfo,
    pub output: GgufTensorInfo,
}

#[derive(Debug, Clone)]
pub struct LagunaGgufDense {
    pub gate: GgufTensorInfo,
    pub up: GgufTensorInfo,
    pub down: GgufTensorInfo,
}

#[derive(Debug, Clone)]
pub struct LagunaGgufMoe {
    pub router: GgufTensorInfo,
    pub correction_bias: GgufTensorInfo,
    pub routed_gate: GgufTensorInfo,
    pub routed_up: GgufTensorInfo,
    pub routed_down: GgufTensorInfo,
    pub shared: LagunaGgufDense,
}

#[derive(Debug, Clone)]
pub enum LagunaGgufMlp {
    Dense(LagunaGgufDense),
    Moe(Box<LagunaGgufMoe>),
}

#[derive(Debug, Clone)]
pub struct LagunaGgufLayer {
    pub layer_index: usize,
    pub attention: LagunaGgufAttention,
    pub post_attention_norm: GgufTensorInfo,
    pub mlp: LagunaGgufMlp,
}

/// Exact tensor directory for Antirez's mixed Q2_K/Q3_K Laguna artifact.
///
/// The directory is validated once at model open. Forward execution then uses
/// the stored offsets directly instead of searching tensor names per token.
#[derive(Debug)]
pub struct LagunaGgufIndex {
    file: GgufFile,
    flavor: LagunaGgufFlavor,
    pub root: LagunaGgufRoot,
    pub layers: Vec<LagunaGgufLayer>,
}

impl LagunaGgufIndex {
    pub fn open(path: impl AsRef<Path>, config: &LagunaConfig) -> Result<Self> {
        let flavor = flavor_for_config(config)?;
        let file = GgufFile::open(path)?;
        validate_container(&file, flavor)?;
        validate_metadata(&file, config, flavor)?;

        let mut used = HashSet::with_capacity(flavor.container().tensor_count as usize);
        let (embedding_type, output_type) = flavor.root_types();
        let root = LagunaGgufRoot {
            embedding: required(
                &file,
                &mut used,
                "token_embd.weight",
                &[config.hidden_size, config.vocab_size],
                embedding_type,
            )?,
            final_norm: required(
                &file,
                &mut used,
                "output_norm.weight",
                &[config.hidden_size],
                GgmlType::F32,
            )?,
            output: required(
                &file,
                &mut used,
                "output.weight",
                &[config.hidden_size, config.vocab_size],
                output_type,
            )?,
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_index in 0..config.num_hidden_layers {
            layers.push(index_layer(&file, &mut used, config, flavor, layer_index)?);
        }
        if used.len() != file.tensors().len() {
            let unexpected = file
                .tensors()
                .iter()
                .filter(|tensor| !used.contains(tensor.name.as_str()))
                .map(|tensor| tensor.name.as_str())
                .take(8)
                .collect::<Vec<_>>();
            return Err(Error::weights(format!(
                "Laguna GGUF contains {} tensors outside the exact runtime contract: {unexpected:?}",
                file.tensors().len().saturating_sub(used.len())
            )));
        }

        Ok(Self {
            file,
            flavor,
            root,
            layers,
        })
    }

    pub const fn flavor(&self) -> LagunaGgufFlavor {
        self.flavor
    }

    pub fn path(&self) -> &Path {
        self.file.path()
    }

    pub fn mapped_bytes(&self) -> MappedBytes {
        self.file.mapped_bytes()
    }

    pub fn tensor_data_offset(&self) -> Result<usize> {
        usize::try_from(self.file.tensor_data_offset())
            .map_err(|_| Error::weights("Laguna GGUF tensor-data offset does not fit usize"))
    }

    pub fn max_tensor_storage_byte_len(&self) -> Result<usize> {
        usize::try_from(self.file.max_tensor_storage_byte_len())
            .map_err(|_| Error::weights("Laguna GGUF maximum tensor span does not fit usize"))
    }

    pub fn storage<'a>(&'a self, tensor: &'a GgufTensorInfo) -> Result<GgufTensorStorage<'a>> {
        self.file.tensor_storage_by_info(tensor)
    }

    pub fn quantized_storage<'a>(
        &'a self,
        tensor: &'a GgufTensorInfo,
    ) -> Result<GgufQuantizedTensorStorage<'a>> {
        self.storage(tensor)?.quantized_payload()
    }

    pub fn f32_values(&self, tensor: &GgufTensorInfo) -> Result<Vec<f32>> {
        self.file.tensor_f32_values(&tensor.name)
    }

    pub fn advise(&self, tensor: &GgufTensorInfo, advice: GgufTensorAdvice) -> Result<()> {
        self.file.advise_tensor_by_info(tensor, advice)
    }
}

fn index_layer(
    file: &GgufFile,
    used: &mut HashSet<String>,
    config: &LagunaConfig,
    flavor: LagunaGgufFlavor,
    layer_index: usize,
) -> Result<LagunaGgufLayer> {
    let prefix = format!("blk.{layer_index}");
    let query_heads = config
        .query_heads(layer_index)
        .ok_or_else(|| Error::config(format!("Laguna layer {layer_index} has no head count")))?;
    let query_width = query_heads
        .checked_mul(config.head_dim)
        .ok_or_else(|| Error::weights("Laguna GGUF query width overflow"))?;
    let kv_width = config.key_value_width();

    let attention = LagunaGgufAttention {
        input_norm: required(
            file,
            used,
            &format!("{prefix}.attn_norm.weight"),
            &[config.hidden_size],
            GgmlType::F32,
        )?,
        query: required(
            file,
            used,
            &format!("{prefix}.attn_q.weight"),
            &[config.hidden_size, query_width],
            flavor.attention_type(layer_index, false),
        )?,
        key: required(
            file,
            used,
            &format!("{prefix}.attn_k.weight"),
            &[config.hidden_size, kv_width],
            flavor.attention_type(layer_index, false),
        )?,
        value: required(
            file,
            used,
            &format!("{prefix}.attn_v.weight"),
            &[config.hidden_size, kv_width],
            flavor.attention_type(layer_index, true),
        )?,
        gate: required(
            file,
            used,
            &format!("{prefix}.attn_gate.weight"),
            &[config.hidden_size, query_heads],
            flavor.attention_type(layer_index, false),
        )?,
        query_norm: required(
            file,
            used,
            &format!("{prefix}.attn_q_norm.weight"),
            &[config.head_dim],
            GgmlType::F32,
        )?,
        key_norm: required(
            file,
            used,
            &format!("{prefix}.attn_k_norm.weight"),
            &[config.head_dim],
            GgmlType::F32,
        )?,
        output: required(
            file,
            used,
            &format!("{prefix}.attn_output.weight"),
            &[query_width, config.hidden_size],
            flavor.attention_type(layer_index, false),
        )?,
    };
    let post_attention_norm = required(
        file,
        used,
        &format!("{prefix}.ffn_norm.weight"),
        &[config.hidden_size],
        GgmlType::F32,
    )?;
    let mlp = if layer_index == 0 {
        LagunaGgufMlp::Dense(index_dense(
            file,
            used,
            &prefix,
            "",
            config.hidden_size,
            config.intermediate_size,
            flavor,
            layer_index,
        )?)
    } else {
        let (expert_gate_up_type, expert_down_type) = flavor.routed_types(layer_index);
        LagunaGgufMlp::Moe(Box::new(LagunaGgufMoe {
            router: required(
                file,
                used,
                &format!("{prefix}.ffn_gate_inp.weight"),
                &[config.hidden_size, config.num_experts],
                GgmlType::F32,
            )?,
            correction_bias: required(
                file,
                used,
                &format!("{prefix}.exp_probs_b.bias"),
                &[config.num_experts],
                GgmlType::F32,
            )?,
            routed_gate: required(
                file,
                used,
                &format!("{prefix}.ffn_gate_exps.weight"),
                &[
                    config.hidden_size,
                    config.moe_intermediate_size,
                    config.num_experts,
                ],
                expert_gate_up_type,
            )?,
            routed_up: required(
                file,
                used,
                &format!("{prefix}.ffn_up_exps.weight"),
                &[
                    config.hidden_size,
                    config.moe_intermediate_size,
                    config.num_experts,
                ],
                expert_gate_up_type,
            )?,
            routed_down: required(
                file,
                used,
                &format!("{prefix}.ffn_down_exps.weight"),
                &[
                    config.moe_intermediate_size,
                    config.hidden_size,
                    config.num_experts,
                ],
                expert_down_type,
            )?,
            shared: index_dense(
                file,
                used,
                &prefix,
                "_shexp",
                config.hidden_size,
                config.shared_expert_intermediate_size,
                flavor,
                layer_index,
            )?,
        }))
    };

    Ok(LagunaGgufLayer {
        layer_index,
        attention,
        post_attention_norm,
        mlp,
    })
}

fn index_dense(
    file: &GgufFile,
    used: &mut HashSet<String>,
    prefix: &str,
    suffix: &str,
    hidden_size: usize,
    intermediate_size: usize,
    flavor: LagunaGgufFlavor,
    layer_index: usize,
) -> Result<LagunaGgufDense> {
    let (gate_up_type, down_type) = flavor.dense_types(layer_index);
    Ok(LagunaGgufDense {
        gate: required(
            file,
            used,
            &format!("{prefix}.ffn_gate{suffix}.weight"),
            &[hidden_size, intermediate_size],
            gate_up_type,
        )?,
        up: required(
            file,
            used,
            &format!("{prefix}.ffn_up{suffix}.weight"),
            &[hidden_size, intermediate_size],
            gate_up_type,
        )?,
        down: required(
            file,
            used,
            &format!("{prefix}.ffn_down{suffix}.weight"),
            &[intermediate_size, hidden_size],
            down_type,
        )?,
    })
}

fn required(
    file: &GgufFile,
    used: &mut HashSet<String>,
    name: &str,
    dims: &[usize],
    ty: GgmlType,
) -> Result<GgufTensorInfo> {
    let tensor = file
        .tensor(name)
        .ok_or_else(|| Error::weights(format!("Laguna GGUF is missing tensor {name}")))?;
    let expected_dims = dims
        .iter()
        .copied()
        .map(|dim| {
            u64::try_from(dim)
                .map_err(|_| Error::weights(format!("Laguna GGUF dimension {dim} exceeds u64")))
        })
        .collect::<Result<Vec<_>>>()?;
    if tensor.dims != expected_dims || tensor.ty != ty {
        return Err(Error::weights(format!(
            "Laguna GGUF tensor {name} must be {ty} {expected_dims:?}, got {} {:?}",
            tensor.ty, tensor.dims
        )));
    }
    if !used.insert(name.to_string()) {
        return Err(Error::weights(format!(
            "Laguna GGUF tensor {name} is used more than once"
        )));
    }
    match ty {
        GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q6K | GgmlType::Q8_0 => {
            file.tensor_quantized_storage(name)?;
        }
        GgmlType::F32 => {
            let expected_bytes = tensor
                .element_count()?
                .checked_mul(4)
                .ok_or_else(|| Error::weights(format!("{name} F32 byte count overflow")))?;
            if tensor.storage_byte_len < expected_bytes {
                return Err(Error::weights(format!(
                    "Laguna GGUF tensor {name} has {} storage bytes; expected at least {expected_bytes}",
                    tensor.storage_byte_len
                )));
            }
        }
        GgmlType::Unsupported(_) => unreachable!("required tensors never use unsupported types"),
    }
    Ok(tensor.clone())
}

fn validate_container(file: &GgufFile, flavor: LagunaGgufFlavor) -> Result<()> {
    let summary = file.summary();
    let expected = flavor.container();
    if summary.file_size != expected.file_bytes
        || summary.tensor_count != expected.tensor_count
        || summary.metadata_kv_count != expected.metadata_count
        || summary.tensor_data_offset != expected.tensor_data_offset
        || summary.architecture.as_deref() != Some("laguna")
        || summary.quantization_version != Some(2)
        || summary.file_type != Some(expected.file_type)
    {
        return Err(Error::weights(format!(
            "GGUF is not the exact {} artifact: size={}, tensors={}, metadata={}, data_offset={}, architecture={:?}, quantization={:?}, file_type={:?}",
            flavor.file_name(),
            summary.file_size,
            summary.tensor_count,
            summary.metadata_kv_count,
            summary.tensor_data_offset,
            summary.architecture,
            summary.quantization_version,
            summary.file_type
        )));
    }
    Ok(())
}

fn validate_metadata(
    file: &GgufFile,
    config: &LagunaConfig,
    flavor: LagunaGgufFlavor,
) -> Result<()> {
    let exact_unsigned = [
        ("laguna.block_count", config.num_hidden_layers),
        ("laguna.context_length", config.max_position_embeddings),
        ("laguna.embedding_length", config.hidden_size),
        ("laguna.feed_forward_length", config.intermediate_size),
        ("laguna.attention.head_count_kv", config.num_key_value_heads),
        ("laguna.attention.key_length", config.head_dim),
        ("laguna.attention.value_length", config.head_dim),
        ("laguna.attention.sliding_window", config.sliding_window),
        ("laguna.expert_count", config.num_experts),
        (
            "laguna.expert_feed_forward_length",
            config.moe_intermediate_size,
        ),
        (
            "laguna.expert_shared_feed_forward_length",
            config.shared_expert_intermediate_size,
        ),
        ("laguna.expert_used_count", config.num_experts_per_tok),
        // GGUF gating function 2 is sigmoid. Selection is
        // sigmoid(router_logits) + correction_bias; routed weights use the
        // unbiased sigmoid values.
        ("laguna.expert_gating_func", 2),
        ("laguna.leading_dense_block_count", 1),
        ("laguna.rope.dimension_count", 64),
        ("laguna.rope.dimension_count_swa", 128),
        ("laguna.rope.scaling.original_context_length", 8_192),
        ("laguna.vocab_size", config.vocab_size),
    ];
    for (key, expected) in exact_unsigned {
        if file.metadata_unsigned(key) != u64::try_from(expected).ok() {
            return Err(Error::weights(format!(
                "Laguna GGUF metadata {key} must be {expected}, got {:?}",
                file.metadata().get(key)
            )));
        }
    }
    expect_bool(file, "laguna.expert_weights_norm", true)?;
    expect_f32(
        file,
        "laguna.attention.layer_norm_rms_epsilon",
        config.rms_norm_eps as f32,
    )?;
    expect_f32(
        file,
        "laguna.expert_weights_scale",
        config.moe_routed_scaling_factor as f32,
    )?;
    expect_f32(file, "laguna.rope.freq_base", 500_000.0)?;
    expect_f32(file, "laguna.rope.freq_base_swa", 10_000.0)?;
    expect_f32(file, "laguna.rope.scaling.factor", 32.0)?;
    expect_string(file, "laguna.rope.scaling.type", "yarn")?;
    // The official GGUF stores 1.0 here because GGUF runtimes derive YaRN's
    // magnitude correction from the scaling factor. Inferno derives the same
    // correction from the external Laguna config when preparing its RoPE
    // table.
    expect_f32(file, "laguna.rope.scaling.yarn_attn_factor", 1.0)?;
    let yarn_beta_fast = match flavor {
        LagunaGgufFlavor::S21Q2Q3 => 32.0,
        LagunaGgufFlavor::Xs21Q4KM => 64.0,
    };
    expect_f32(file, "laguna.rope.scaling.yarn_beta_fast", yarn_beta_fast)?;
    expect_f32(file, "laguna.rope.scaling.yarn_beta_slow", 1.0)?;
    if metadata_i32_array(file, "laguna.attention.head_count")?
        != config
            .num_attention_heads_per_layer
            .iter()
            .map(|value| *value as i32)
            .collect::<Vec<_>>()
    {
        return Err(Error::weights(
            "Laguna GGUF per-layer attention head schedule does not match config.json",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{laguna_xs_uses_q6, LagunaGgufFlavor, LAGUNA_XS_Q6_LAYERS};
    use gguf::GgmlType;

    #[test]
    fn xs_q6_layer_schedule_matches_the_official_q4_k_m_directory() {
        let actual = (0..40)
            .filter(|layer_index| laguna_xs_uses_q6(*layer_index))
            .collect::<Vec<_>>();

        assert_eq!(actual, LAGUNA_XS_Q6_LAYERS);
    }

    #[test]
    fn s_and_xs_quantization_contracts_remain_separate() {
        assert_eq!(
            LagunaGgufFlavor::S21Q2Q3.routed_types(20),
            (GgmlType::Q2K, GgmlType::Q2K)
        );
        assert_eq!(
            LagunaGgufFlavor::S21Q2Q3.routed_types(21),
            (GgmlType::Q3K, GgmlType::Q3K)
        );
        assert_eq!(
            LagunaGgufFlavor::Xs21Q4KM.routed_types(1),
            (GgmlType::Q4K, GgmlType::Q6K)
        );
        assert_eq!(
            LagunaGgufFlavor::Xs21Q4KM.routed_types(5),
            (GgmlType::Q4K, GgmlType::Q4K)
        );
    }
}

fn expect_bool(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    match file.metadata().get(key) {
        Some(GgufMetadataValue::Bool(value)) if *value == expected => Ok(()),
        actual => Err(Error::weights(format!(
            "Laguna GGUF metadata {key} must be {expected}, got {actual:?}"
        ))),
    }
}

fn expect_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    match file.metadata().get(key) {
        Some(GgufMetadataValue::String(value)) if value == expected => Ok(()),
        actual => Err(Error::weights(format!(
            "Laguna GGUF metadata {key} must be {expected:?}, got {actual:?}"
        ))),
    }
}

fn expect_f32(file: &GgufFile, key: &str, expected: f32) -> Result<()> {
    match file.metadata().get(key) {
        Some(GgufMetadataValue::Float32(value)) if *value == expected => Ok(()),
        actual => Err(Error::weights(format!(
            "Laguna GGUF metadata {key} must be {expected}, got {actual:?}"
        ))),
    }
}

fn metadata_i32_array(file: &GgufFile, key: &str) -> Result<Vec<i32>> {
    match file.metadata().get(key) {
        Some(GgufMetadataValue::Array {
            values: Some(values),
            ..
        }) => values
            .iter()
            .map(|value| match value {
                GgufMetadataValue::Int32(value) => Ok(*value),
                other => Err(Error::weights(format!(
                    "Laguna GGUF metadata {key} contains non-I32 value {other:?}"
                ))),
            })
            .collect(),
        actual => Err(Error::weights(format!(
            "Laguna GGUF metadata {key} must be a stored I32 array, got {actual:?}"
        ))),
    }
}
