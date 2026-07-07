use backend::Backend;
use common::{validate_exact_shape, Error, F32Tensor, Result, Shape};
use config::Config;
use gguf::{GgmlType, GgufFile};

use crate::{
    EmbeddingTable, LayerKind, LayerKvCacheTensors, MtpIndex, OutputHead, QuantizedLinear, RmsNorm,
    RmsNormLoadReport, SparseBlock, SparseBlockLoadReport,
};

#[derive(Debug)]
pub struct MtpHead<'a> {
    hnorm: RmsNorm,
    enorm: RmsNorm,
    eh_proj: QuantizedLinear<'a>,
    block: SparseBlock<'a>,
    shared_head_norm: RmsNorm,
    load_report: MtpLoadReport,
}

#[derive(Debug)]
pub struct MtpDraftOutput {
    pub hidden_states: F32Tensor,
    pub layer_kv_cache: LayerKvCacheTensors,
    pub token_id: u32,
    pub token_score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MtpLoadReport {
    pub layer_index: usize,
    pub hnorm: RmsNormLoadReport,
    pub enorm: RmsNormLoadReport,
    pub eh_proj_tensor_name: String,
    pub eh_proj_tensor_type: GgmlType,
    pub eh_proj_shape: Shape,
    pub block: SparseBlockLoadReport,
    pub shared_head_norm: RmsNormLoadReport,
    pub limitations: Vec<String>,
}

impl<'a> MtpHead<'a> {
    pub fn open<B: Backend>(
        gguf: &'a GgufFile,
        config: &Config,
        index: &MtpIndex,
        backend: &B,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        validate_exact_shape(
            "gguf_mtp_layer_index",
            &[index.layer.layer_index],
            &[config.num_layers],
        )?;
        if index.layer.kind != LayerKind::SparseMoe {
            return Err(Error::gguf(format!(
                "GLM-5.2 MTP layer {} must be sparse_moe",
                index.layer.layer_index
            )));
        }

        let hnorm = RmsNorm::open(gguf, config, &index.hnorm, backend)?;
        let enorm = RmsNorm::open(gguf, config, &index.enorm, backend)?;
        let shared_head_norm = RmsNorm::open(gguf, config, &index.shared_head_norm, backend)?;
        let eh_proj = QuantizedLinear::open(
            gguf,
            &index.eh_proj,
            config.hidden_size * 2,
            config.hidden_size,
            output_chunk_rows,
        )?;
        let block = SparseBlock::open(gguf, config, &index.layer, backend, output_chunk_rows)?;

        let load_report = MtpLoadReport {
            layer_index: index.layer.layer_index,
            hnorm: hnorm.load_report().clone(),
            enorm: enorm.load_report().clone(),
            eh_proj_tensor_name: index.eh_proj.name.clone(),
            eh_proj_tensor_type: index.eh_proj.ty,
            eh_proj_shape: Shape::new(
                index
                    .eh_proj
                    .dims
                    .iter()
                    .map(|dim| *dim as usize)
                    .collect::<Vec<_>>(),
            ),
            block: block.load_report().clone(),
            shared_head_norm: shared_head_norm.load_report().clone(),
            limitations: vec![
                "MTP targets the single GLM-5.2 nextn head stored at blk.num_layers".to_string(),
                "MTP verification is greedy-only and uses the shared output projection".to_string(),
            ],
        };

        Ok(Self {
            hnorm,
            enorm,
            eh_proj,
            block,
            shared_head_norm,
            load_report,
        })
    }

    pub fn load_report(&self) -> &MtpLoadReport {
        &self.load_report
    }

    pub fn draft<B: Backend>(
        &self,
        config: &Config,
        main_hidden_states: &F32Tensor,
        next_token_ids: &[u32],
        embedding_table: &EmbeddingTable<'_>,
        output_head: &OutputHead<'_>,
        backend: &B,
        past_kv: Option<(&F32Tensor, &F32Tensor)>,
        cached_index_keys: Option<&F32Tensor>,
    ) -> Result<MtpDraftOutput> {
        let dims = main_hidden_states.dims();
        validate_exact_shape("gguf_mtp_hidden_rank", &[dims.len()], &[3])?;
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            "gguf_mtp_hidden_shape",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;
        validate_exact_shape("gguf_mtp_batch", &[batch], &[1])?;
        validate_exact_shape(
            "gguf_mtp_next_token_count",
            &[next_token_ids.len()],
            &[tokens],
        )?;

        let embeddings = embedding_table.lookup_f32(next_token_ids)?.hidden_states;
        let hnorm = self
            .hnorm
            .forward_f32(main_hidden_states, backend)?
            .hidden_states;
        let enorm = self.enorm.forward_f32(&embeddings, backend)?.hidden_states;
        let fused = F32Tensor::cat(&[&enorm, &hnorm], 2)?;
        let projected = self.eh_proj.forward_f32_tensor(&fused, backend)?;
        let (block_output, _) = self.block.forward_sparse_f32_tensors_with_past_kv(
            config,
            &projected,
            backend,
            past_kv,
            cached_index_keys,
            None,
        )?;
        let selected = require_native(
            "select_last_token",
            backend.select_last_token_f32_tensor(&block_output.hidden_states)?,
        )?;
        let normalized = self
            .shared_head_norm
            .forward_f32(&selected, backend)?
            .hidden_states;
        let token = output_head.decode_token_from_normalized_f32(&normalized, backend)?;

        Ok(MtpDraftOutput {
            hidden_states: block_output.hidden_states,
            layer_kv_cache: LayerKvCacheTensors {
                layer_index: self.load_report.layer_index,
                layer_kind: LayerKind::SparseMoe,
                cache_k: block_output.cache_k,
                cache_v: block_output.cache_v,
                index_key: block_output.index_key,
            },
            token_id: token.token_id,
            token_score: token.token_score,
        })
    }
}

fn require_native<T>(operation: &str, value: Option<T>) -> Result<T> {
    value.ok_or_else(|| {
        Error::backend(format!(
            "native Metal {operation} is required for the GLM-5.2 Q2 MTP path"
        ))
    })
}
