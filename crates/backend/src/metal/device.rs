use std::{path::Path, sync::Arc};

use crate::{
    BackendMemoryReport, DeviceBf16Matrix, DevicePagedKvView, DeviceRopeTable, DeviceRouterTopK,
    DeviceW4Weight, ExpertCacheMetrics, GgufExpertQuant, GgufKQuant, LagunaF16KvCache,
    LagunaFp8KvCache, LagunaKvRetention, LagunaModelViewReport, Q2ExpertSource, W4ExpertGroup,
    W4WeightSource,
};
use ::metal::{Buffer, CommandQueue, Device};
use common::{DType, Error, PagedKvView, Result};
use inferno_io::{ExpertPackHeader, MappedBytes};

use super::activation::MetalActivation;
use super::arena::MetalArena;
use super::attention::{
    MetalAttentionCausalSoftmax, MetalAttentionScores, MetalAttentionValues, MetalDecodeAttention,
};
use super::batch::BatchSlot;
use super::bf16::MetalBf16;
use super::buffers::{empty_f16_buffer, f32_buffer, read_f32_buffer, read_u32_buffer, u8_buffer};
use super::cast::MetalCast;
use super::command::{encode_element_copy, encode_f32_copy, Dispatch1d};
use super::dsa::MetalDsa;
use super::f16_attention::MetalF16Attention;
use super::fp8_attention::MetalFp8Attention;
use super::gguf_moe::MetalGgufMoe;
use super::gguf_moe_prefill::MetalGgufMoePrefill;
use super::laguna_views::MetalLagunaViews;
use super::laguna_xs::MetalLagunaXs;
use super::laguna_xs_moe_prefill::MetalLagunaXsMoePrefill;
use super::laguna_xs_mps_prefill::MetalLagunaXsMpsPrefill;
use super::laguna_xs_prefill::MetalLagunaXsPrefill;
use super::layout::MetalLayout;
use super::library::MetalLibrary;
use super::matmul::MetalMatmul;
use super::moe::MetalMoe;
use super::q2::{
    MetalQ2Matvec, QuantMatvecKind, ReadyRoutedExperts, Q8_0_MMA_MIN_PREFILL_ROWS,
    Q8_0_MMA_OUTPUT_TILE,
};
use super::rms_norm::MetalRmsNorm;
use super::rope::MetalRope;
use super::w4::MetalW4;

pub struct Metal {
    device: Device,
    queue: CommandQueue,
    attention_scores: MetalAttentionScores,
    attention_values: MetalAttentionValues,
    attention_causal_softmax: MetalAttentionCausalSoftmax,
    decode_attention: MetalDecodeAttention,
    activation: MetalActivation,
    bf16: MetalBf16,
    cast: MetalCast,
    layout: MetalLayout,
    matmul: MetalMatmul,
    q2_matvec: MetalQ2Matvec,
    w4: MetalW4,
    rms_norm: MetalRmsNorm,
    rope: MetalRope,
    moe: MetalMoe,
    dsa: MetalDsa,
    f16_attention: MetalF16Attention,
    fp8_attention: MetalFp8Attention,
    gguf_moe: MetalGgufMoe,
    gguf_moe_prefill: MetalGgufMoePrefill,
    laguna_xs: MetalLagunaXs,
    laguna_xs_mps_prefill: MetalLagunaXsMpsPrefill,
    laguna_xs_moe_prefill: MetalLagunaXsMoePrefill,
    laguna_xs_prefill: MetalLagunaXsPrefill,
    laguna_views: Arc<MetalLagunaViews>,
    batch: BatchSlot,
}

impl Metal {
    pub fn new() -> Result<Self> {
        let device = select_native_device()?;
        let queue = device.new_command_queue();
        let library = MetalLibrary::compile(&device)?;
        let arena = MetalArena::new(&device)?;
        let laguna_views = Arc::new(MetalLagunaViews::new(&device, &library)?);
        let attention_scores = MetalAttentionScores::new(&device, &library, arena.clone())?;
        let attention_values = MetalAttentionValues::new(&device, &library, arena.clone())?;
        let attention_causal_softmax =
            MetalAttentionCausalSoftmax::new(&device, &library, arena.clone())?;
        let decode_attention = MetalDecodeAttention::new(&device, &library, arena.clone())?;
        let activation = MetalActivation::new(&device, &library, arena.clone())?;
        let bf16 = MetalBf16::new(&device, &library, arena.clone())?;
        let cast = MetalCast::new(&device, &library, arena.clone())?;
        let layout = MetalLayout::new(&device, &library, arena.clone())?;
        let matmul = MetalMatmul::new(&device, &library, arena.clone())?;
        let q2_matvec =
            MetalQ2Matvec::new(&device, &library, arena.clone(), Arc::clone(&laguna_views))?;
        let w4 = MetalW4::new(&device, &library, arena.clone())?;
        let rms_norm = MetalRmsNorm::new(&device, &library, arena.clone())?;
        let rope = MetalRope::new(&device, &library, arena.clone())?;
        let moe = MetalMoe::new(&device, &library, arena.clone())?;
        let dsa = MetalDsa::new(&device, &library, arena.clone())?;
        let f16_attention = MetalF16Attention::new(&device, &library, arena.clone())?;
        let fp8_attention = MetalFp8Attention::new(&device, &library, arena.clone())?;
        let gguf_moe_prefill =
            MetalGgufMoePrefill::new(&device, &library, arena.clone(), Arc::clone(&laguna_views))?;
        let laguna_xs = MetalLagunaXs::new(arena.clone(), Arc::clone(&laguna_views));
        let laguna_xs_mps_prefill =
            MetalLagunaXsMpsPrefill::new(arena.clone(), Arc::clone(&laguna_views));
        let laguna_xs_moe_prefill =
            MetalLagunaXsMoePrefill::new(arena.clone(), Arc::clone(&laguna_views));
        let laguna_xs_prefill = MetalLagunaXsPrefill::new(arena.clone(), Arc::clone(&laguna_views));
        let gguf_moe = MetalGgufMoe::new(&device, &library, arena, Arc::clone(&laguna_views))?;

        Ok(Self {
            device,
            queue,
            attention_scores,
            attention_values,
            attention_causal_softmax,
            decode_attention,
            activation,
            bf16,
            cast,
            layout,
            matmul,
            q2_matvec,
            w4,
            rms_norm,
            rope,
            moe,
            dsa,
            f16_attention,
            fp8_attention,
            gguf_moe,
            gguf_moe_prefill,
            laguna_xs,
            laguna_xs_mps_prefill,
            laguna_xs_moe_prefill,
            laguna_xs_prefill,
            laguna_views,
            batch: BatchSlot::new(),
        })
    }

    pub fn current_allocated_bytes(&self) -> u64 {
        self.device.current_allocated_size() as u64
    }

    pub fn recommended_max_working_set_bytes(&self) -> u64 {
        self.device.recommended_max_working_set_size()
    }

    pub fn memory_report(&self) -> BackendMemoryReport {
        let mut report = super::memory::native_memory_report();
        report.metal_current_allocated_bytes = Some(self.current_allocated_bytes());
        report.metal_recommended_max_working_set_bytes =
            Some(self.recommended_max_working_set_bytes());
        report
    }

    pub fn expert_cache_metrics(&self) -> Result<ExpertCacheMetrics> {
        self.q2_matvec.expert_cache_metrics()
    }

    pub fn configure_expert_cache_slots_per_layer(&self, slots_per_layer: usize) -> Result<()> {
        self.q2_matvec
            .configure_expert_cache_slots_per_layer(slots_per_layer)
    }

    pub fn resize_expert_cache_slots_per_layer(&self, slots_per_layer: usize) -> Result<()> {
        self.q2_matvec
            .resize_expert_cache_slots_per_layer(slots_per_layer)
    }

    pub fn release_prefill_resources(&self) -> Result<()> {
        self.q2_matvec.release_prefill_resources()
    }

    pub fn configure_expert_pack(&self, path: &Path, header: ExpertPackHeader) -> Result<()> {
        self.q2_matvec.configure_expert_pack(path, header)
    }

    pub(crate) fn prepare_laguna_gguf_views(
        &self,
        mapping: MappedBytes,
        tensor_data_offset: usize,
        max_tensor_bytes: usize,
    ) -> Result<LagunaModelViewReport> {
        self.laguna_views.prepare(
            &self.device,
            &self.queue,
            mapping,
            tensor_data_offset,
            max_tensor_bytes,
        )
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    pub fn rms_norm_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        self.rms_norm
            .run(
                self.device(),
                self.queue(),
                input,
                weight,
                rows,
                hidden_size,
                eps,
            )
            .map(|output| output.values)
    }

    pub fn matmul_f32(
        &self,
        lhs: &[f32],
        rhs: &[f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<Vec<f32>> {
        self.matmul
            .matmul(self.device(), self.queue(), lhs, rhs, rows, inner, cols)
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn matmul_f32_report(
        &self,
        lhs: &[f32],
        rhs: &[f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<super::matmul::MetalMatmulReport> {
        self.matmul
            .matmul(self.device(), self.queue(), lhs, rhs, rows, inner, cols)
    }

    pub fn linear_f32(
        &self,
        input: &[f32],
        weight: &[f32],
        rows: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Vec<f32>> {
        self.matmul
            .linear(
                self.device(),
                self.queue(),
                input,
                weight,
                rows,
                in_features,
                out_features,
            )
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn linear_f32_report(
        &self,
        input: &[f32],
        weight: &[f32],
        rows: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<super::matmul::MetalLinearReport> {
        self.matmul.linear(
            self.device(),
            self.queue(),
            input,
            weight,
            rows,
            in_features,
            out_features,
        )
    }

    pub fn swiglu_f32(&self, gate: &[f32], up: &[f32]) -> Result<Vec<f32>> {
        self.activation
            .swiglu(self.device(), self.queue(), gate, up)
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn swiglu_f32_report(
        &self,
        gate: &[f32],
        up: &[f32],
    ) -> Result<super::activation::MetalSwiGluReport> {
        self.activation
            .swiglu(self.device(), self.queue(), gate, up)
    }

    pub fn add_f32(&self, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>> {
        self.activation
            .add(self.device(), self.queue(), lhs, rhs)
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn add_f32_report(
        &self,
        lhs: &[f32],
        rhs: &[f32],
    ) -> Result<super::activation::MetalAddReport> {
        self.activation.add(self.device(), self.queue(), lhs, rhs)
    }

    pub fn select_last_token_f32(
        &self,
        hidden_states: &[f32],
        batch_count: usize,
        token_count: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>> {
        self.layout
            .select_last_token(
                self.device(),
                self.queue(),
                hidden_states,
                batch_count,
                token_count,
                hidden_size,
            )
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn select_last_token_f32_report(
        &self,
        hidden_states: &[f32],
        batch_count: usize,
        token_count: usize,
        hidden_size: usize,
    ) -> Result<super::layout::MetalSelectLastTokenReport> {
        self.layout.select_last_token(
            self.device(),
            self.queue(),
            hidden_states,
            batch_count,
            token_count,
            hidden_size,
        )
    }

    pub fn heads_to_attention_layout_f32(
        &self,
        input: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        self.layout
            .heads_to_attention_layout(
                self.device(),
                self.queue(),
                input,
                batch_count,
                token_count,
                head_count,
                head_dim,
            )
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn heads_to_attention_layout_f32_report(
        &self,
        input: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        head_dim: usize,
    ) -> Result<super::layout::MetalHeadsToAttentionLayoutReport> {
        self.layout.heads_to_attention_layout(
            self.device(),
            self.queue(),
            input,
            batch_count,
            token_count,
            head_count,
            head_dim,
        )
    }

    pub fn merge_attention_heads_f32(
        &self,
        input: &[f32],
        batch_count: usize,
        head_count: usize,
        token_count: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        self.layout
            .merge_attention_heads(
                self.device(),
                self.queue(),
                input,
                batch_count,
                head_count,
                token_count,
                head_dim,
            )
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn merge_attention_heads_f32_report(
        &self,
        input: &[f32],
        batch_count: usize,
        head_count: usize,
        token_count: usize,
        head_dim: usize,
    ) -> Result<super::layout::MetalMergeAttentionHeadsReport> {
        self.layout.merge_attention_heads(
            self.device(),
            self.queue(),
            input,
            batch_count,
            head_count,
            token_count,
            head_dim,
        )
    }

    pub fn split_rope_tail_f32(
        &self,
        input: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        no_rope_dim: usize,
        rope_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.layout
            .split_rope_tail(
                self.device(),
                self.queue(),
                input,
                batch_count,
                token_count,
                head_count,
                no_rope_dim,
                rope_dim,
            )
            .map(|output| (output.no_rope_values, output.rope_values))
    }

    #[cfg(test)]
    pub fn split_rope_tail_f32_report(
        &self,
        input: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        no_rope_dim: usize,
        rope_dim: usize,
    ) -> Result<super::layout::MetalSplitRopeTailReport> {
        self.layout.split_rope_tail(
            self.device(),
            self.queue(),
            input,
            batch_count,
            token_count,
            head_count,
            no_rope_dim,
            rope_dim,
        )
    }

    pub fn split_kv_mqa_f32(
        &self,
        input: &[f32],
        batch_count: usize,
        token_count: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.layout
            .split_kv_mqa(
                self.device(),
                self.queue(),
                input,
                batch_count,
                token_count,
                kv_lora_rank,
                rope_dim,
            )
            .map(|output| (output.kv_latent_values, output.k_rope_values))
    }

    #[cfg(test)]
    pub fn split_kv_mqa_f32_report(
        &self,
        input: &[f32],
        batch_count: usize,
        token_count: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
    ) -> Result<super::layout::MetalSplitKvMqaReport> {
        self.layout.split_kv_mqa(
            self.device(),
            self.queue(),
            input,
            batch_count,
            token_count,
            kv_lora_rank,
            rope_dim,
        )
    }

    pub fn combine_rope_tail_f32(
        &self,
        no_rope: &[f32],
        rope: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        rope_head_count: usize,
        no_rope_dim: usize,
        rope_dim: usize,
    ) -> Result<Vec<f32>> {
        self.layout
            .combine_rope_tail(
                self.device(),
                self.queue(),
                no_rope,
                rope,
                batch_count,
                token_count,
                head_count,
                rope_head_count,
                no_rope_dim,
                rope_dim,
            )
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn combine_rope_tail_f32_report(
        &self,
        no_rope: &[f32],
        rope: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        rope_head_count: usize,
        no_rope_dim: usize,
        rope_dim: usize,
    ) -> Result<super::layout::MetalCombineRopeTailReport> {
        self.layout.combine_rope_tail(
            self.device(),
            self.queue(),
            no_rope,
            rope,
            batch_count,
            token_count,
            head_count,
            rope_head_count,
            no_rope_dim,
            rope_dim,
        )
    }

    #[cfg(test)]
    pub fn stack_head_outputs_f32_report(
        &self,
        head_outputs: &[Vec<f32>],
        row_count: usize,
        head_dim: usize,
    ) -> Result<super::layout::MetalStackHeadOutputsReport> {
        self.layout.stack_head_outputs(
            self.device(),
            self.queue(),
            head_outputs,
            row_count,
            head_dim,
        )
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub fn linearize_paged_cache_f32_report(
        &self,
        paged: &[f32],
        batch_count: usize,
        head_count: usize,
        cached_tokens: usize,
        capacity_tokens: usize,
        page_size: usize,
        head_dim: usize,
    ) -> Result<super::layout::MetalLinearizePagedCacheReport> {
        self.layout.linearize_paged_cache(
            self.device(),
            self.queue(),
            paged,
            batch_count,
            head_count,
            cached_tokens,
            capacity_tokens,
            page_size,
            head_dim,
        )
    }

    pub fn attention_scores_f32(
        &self,
        q: &[f32],
        k: &[f32],
        batch_count: usize,
        head_count: usize,
        query_tokens: usize,
        key_tokens: usize,
        head_dim: usize,
    ) -> Result<Vec<f32>> {
        self.attention_scores
            .run(
                self.device(),
                self.queue(),
                q,
                k,
                batch_count,
                head_count,
                query_tokens,
                key_tokens,
                head_dim,
            )
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn attention_scores_f32_report(
        &self,
        q: &[f32],
        k: &[f32],
        batch_count: usize,
        head_count: usize,
        query_tokens: usize,
        key_tokens: usize,
        head_dim: usize,
    ) -> Result<super::attention::MetalAttentionScoresReport> {
        self.attention_scores.run(
            self.device(),
            self.queue(),
            q,
            k,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            head_dim,
        )
    }

    pub fn attention_values_f32(
        &self,
        probs: &[f32],
        values: &[f32],
        batch_count: usize,
        head_count: usize,
        query_tokens: usize,
        key_tokens: usize,
        value_dim: usize,
    ) -> Result<Vec<f32>> {
        self.attention_values
            .run(
                self.device(),
                self.queue(),
                probs,
                values,
                batch_count,
                head_count,
                query_tokens,
                key_tokens,
                value_dim,
            )
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn attention_values_f32_report(
        &self,
        probs: &[f32],
        values: &[f32],
        batch_count: usize,
        head_count: usize,
        query_tokens: usize,
        key_tokens: usize,
        value_dim: usize,
    ) -> Result<super::attention::MetalAttentionValuesReport> {
        self.attention_values.run(
            self.device(),
            self.queue(),
            probs,
            values,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            value_dim,
        )
    }

    pub fn attention_causal_softmax_f32(
        &self,
        scores: &[f32],
        batch_count: usize,
        head_count: usize,
        query_tokens: usize,
        key_tokens: usize,
        past_tokens: usize,
    ) -> Result<Vec<f32>> {
        self.attention_causal_softmax
            .run(
                self.device(),
                self.queue(),
                scores,
                batch_count,
                head_count,
                query_tokens,
                key_tokens,
                past_tokens,
            )
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn attention_causal_softmax_f32_report(
        &self,
        scores: &[f32],
        batch_count: usize,
        head_count: usize,
        query_tokens: usize,
        key_tokens: usize,
        past_tokens: usize,
    ) -> Result<super::attention::MetalAttentionCausalSoftmaxReport> {
        self.attention_causal_softmax.run(
            self.device(),
            self.queue(),
            scores,
            batch_count,
            head_count,
            query_tokens,
            key_tokens,
            past_tokens,
        )
    }

    pub fn decode_attention_f32(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch_count: usize,
        head_count: usize,
        key_tokens: usize,
        head_dim: usize,
        value_dim: usize,
        past_tokens: usize,
    ) -> Result<Vec<f32>> {
        self.decode_attention
            .run(
                self.device(),
                self.queue(),
                q,
                k,
                v,
                batch_count,
                head_count,
                1,
                key_tokens,
                head_dim,
                value_dim,
                past_tokens,
            )
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn decode_attention_f32_report(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        batch_count: usize,
        head_count: usize,
        key_tokens: usize,
        head_dim: usize,
        value_dim: usize,
        past_tokens: usize,
    ) -> Result<super::attention::MetalDecodeAttentionReport> {
        self.decode_attention.run(
            self.device(),
            self.queue(),
            q,
            k,
            v,
            batch_count,
            head_count,
            1,
            key_tokens,
            head_dim,
            value_dim,
            past_tokens,
        )
    }

    pub fn paged_decode_attention_f32(
        &self,
        q: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        past_kv: &PagedKvView<'_>,
    ) -> Result<Vec<f32>> {
        self.decode_attention
            .run_paged(
                self.device(),
                self.queue(),
                q,
                current_k,
                current_v,
                past_kv,
            )
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn paged_decode_attention_f32_report(
        &self,
        q: &[f32],
        current_k: &[f32],
        current_v: &[f32],
        past_kv: &PagedKvView<'_>,
    ) -> Result<super::attention::MetalPagedDecodeAttentionReport> {
        self.decode_attention.run_paged(
            self.device(),
            self.queue(),
            q,
            current_k,
            current_v,
            past_kv,
        )
    }

    #[cfg(test)]
    pub fn rms_norm_f32_report(
        &self,
        input: &[f32],
        weight: &[f32],
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<super::rms_norm::MetalRmsNormReport> {
        self.rms_norm.run(
            self.device(),
            self.queue(),
            input,
            weight,
            rows,
            hidden_size,
            eps,
        )
    }

    pub fn q2_k_matvec_f32(
        &self,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Vec<f32>> {
        self.q2_matvec
            .run(
                self.device(),
                self.queue(),
                weights,
                input,
                row_count,
                in_features,
                out_features,
            )
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn q2_k_matvec_f32_report(
        &self,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<super::q2::MetalQ2MatvecReport> {
        self.q2_matvec.run(
            self.device(),
            self.queue(),
            weights,
            input,
            row_count,
            in_features,
            out_features,
        )
    }

    pub fn q2_k_matvec_add_f32(
        &self,
        weights: &[u8],
        input: &[f32],
        residual: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Vec<f32>> {
        self.q2_matvec
            .run_add(
                self.device(),
                self.queue(),
                weights,
                input,
                residual,
                row_count,
                in_features,
                out_features,
            )
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn q2_k_matvec_add_f32_report(
        &self,
        weights: &[u8],
        input: &[f32],
        residual: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<super::q2::MetalQ2MatvecAddReport> {
        self.q2_matvec.run_add(
            self.device(),
            self.queue(),
            weights,
            input,
            residual,
            row_count,
            in_features,
            out_features,
        )
    }

    pub fn q2_k_gate_up_swiglu_f32(
        &self,
        gate_weights: &[u8],
        up_weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Vec<f32>> {
        self.q2_matvec
            .run_gate_up_swiglu(
                self.device(),
                self.queue(),
                gate_weights,
                up_weights,
                input,
                row_count,
                in_features,
                out_features,
            )
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn q2_k_gate_up_swiglu_f32_report(
        &self,
        gate_weights: &[u8],
        up_weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<super::q2::MetalQ2GateUpSwiGluReport> {
        self.q2_matvec.run_gate_up_swiglu(
            self.device(),
            self.queue(),
            gate_weights,
            up_weights,
            input,
            row_count,
            in_features,
            out_features,
        )
    }

    pub fn q2_k_matvec_argmax_f32(
        &self,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<(u32, f32)> {
        self.q2_matvec
            .run_argmax(
                self.device(),
                self.queue(),
                weights,
                input,
                row_count,
                in_features,
                out_features,
            )
            .map(|output| (output.token_id, output.token_score))
    }

    pub fn q2_k_rms_norm_argmax_f32(
        &self,
        weights: &[u8],
        input: &[f32],
        rms_weight: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
        eps: f32,
    ) -> Result<(u32, f32)> {
        let normed = self.rms_norm.prepare_to_buffer(
            self.device(),
            input,
            rms_weight,
            row_count,
            in_features,
            eps,
        )?;
        let norm_buffers = [
            &normed.input_buffer,
            &normed.weight_buffer,
            &normed.output_buffer,
            &normed.rows_buffer,
            &normed.hidden_size_buffer,
            &normed.eps_buffer,
        ];
        let norm_dispatch = Dispatch1d {
            pipeline: self.rms_norm.pipeline(),
            buffers: &norm_buffers,
            threads: self.rms_norm.thread_count(row_count)?,
        };
        self.q2_matvec
            .run_argmax_with_input_buffer_after_dispatch(
                self.device(),
                self.queue(),
                weights,
                &normed.output_buffer,
                normed.input_len,
                row_count,
                in_features,
                out_features,
                norm_dispatch,
            )
            .map(|output| (output.token_id, output.token_score))
    }

    #[cfg(test)]
    pub fn q2_k_matvec_argmax_f32_report(
        &self,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<super::q2::MetalQ2MatvecArgmaxReport> {
        self.q2_matvec.run_argmax(
            self.device(),
            self.queue(),
            weights,
            input,
            row_count,
            in_features,
            out_features,
        )
    }

    pub fn q2_k_transposed_matvec_f32(
        &self,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Vec<f32>> {
        self.q2_matvec
            .run_transposed(
                self.device(),
                self.queue(),
                weights,
                input,
                row_count,
                in_features,
                out_features,
            )
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn q2_k_transposed_matvec_f32_report(
        &self,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<super::q2::MetalQ2MatvecReport> {
        self.q2_matvec.run_transposed(
            self.device(),
            self.queue(),
            weights,
            input,
            row_count,
            in_features,
            out_features,
        )
    }

    pub fn q8_0_matvec_f32(
        &self,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Vec<f32>> {
        self.q2_matvec
            .run_q8_0(
                self.device(),
                self.queue(),
                weights,
                input,
                row_count,
                in_features,
                out_features,
            )
            .map(|output| output.values)
    }

    pub fn q8_0_transposed_matvec_f32(
        &self,
        weights: &[u8],
        input: &[f32],
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Vec<f32>> {
        self.q2_matvec
            .run_q8_0_transposed(
                self.device(),
                self.queue(),
                weights,
                input,
                row_count,
                in_features,
                out_features,
            )
            .map(|output| output.values)
    }

    pub fn rope_slice_f32(
        &self,
        input: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
    ) -> Result<Vec<f32>> {
        self.rope
            .run(
                self.device(),
                self.queue(),
                input,
                batch_count,
                token_count,
                head_count,
                rope_dim,
                position_offset,
                theta,
            )
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn rope_slice_f32_report(
        &self,
        input: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
    ) -> Result<super::rope::MetalRopeReport> {
        self.rope.run(
            self.device(),
            self.queue(),
            input,
            batch_count,
            token_count,
            head_count,
            rope_dim,
            position_offset,
            theta,
        )
    }

    pub fn moe_gather_tokens_f32(
        &self,
        flat_tokens: &[f32],
        token_indices: &[u32],
        token_count: usize,
        hidden_size: usize,
        assignment_count: usize,
    ) -> Result<Vec<f32>> {
        self.moe
            .gather_tokens(
                self.device(),
                self.queue(),
                flat_tokens,
                token_indices,
                token_count,
                hidden_size,
                assignment_count,
            )
            .map(|output| output.values)
    }

    #[cfg(test)]
    pub fn moe_gather_tokens_f32_report(
        &self,
        flat_tokens: &[f32],
        token_indices: &[u32],
        token_count: usize,
        hidden_size: usize,
        assignment_count: usize,
    ) -> Result<super::moe::MetalMoeGatherReport> {
        self.moe.gather_tokens(
            self.device(),
            self.queue(),
            flat_tokens,
            token_indices,
            token_count,
            hidden_size,
            assignment_count,
        )
    }

    pub fn moe_weighted_index_add_combine_f32(
        &self,
        accumulator: &[f32],
        token_indices: &[u32],
        expert_outputs: &[f32],
        expert_weights: &[f32],
        token_count: usize,
        hidden_size: usize,
        assignment_count: usize,
    ) -> Result<Vec<f32>> {
        self.moe
            .weighted_index_add_combine(
                self.device(),
                self.queue(),
                accumulator,
                token_indices,
                expert_outputs,
                expert_weights,
                token_count,
                hidden_size,
                assignment_count,
            )
            .map(|output| output.values)
    }

    // ------------------------------------------------------------------
    // Batched (device-resident) execution.
    //
    // The batched_* methods encode kernels into a shared open command buffer
    // instead of committing one command buffer per op. Inputs and outputs are
    // GPU buffers; the GPU only runs the accumulated work when `batch_flush`
    // commits and waits. See `BatchSlot` for the safety rules.
    // ------------------------------------------------------------------

    /// Commits the open batch without waiting for the GPU.
    pub fn batch_submit(&self) -> Result<()> {
        self.batch.submit()
    }

    pub fn batch_submit_profile_segment(&self, label: &str) -> Result<()> {
        self.batch.submit_profile_segment(label)
    }

    /// Commits the open batch, if any, and waits for the GPU to finish it.
    /// Routed-expert submissions are validated after the main queue reaches
    /// the same synchronization point.
    pub fn batch_flush(&self) -> Result<()> {
        let batch_result = self.batch.flush();
        let expert_result = self.q2_matvec.finish_ready_expert_submissions();
        match (batch_result, expert_result) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
            (Err(error), Err(expert_error)) => {
                tracing::debug!(
                    error = %expert_error,
                    "routed expert submission also failed while flushing the main Metal queue"
                );
                Err(error)
            }
        }
    }

    /// Uploads host values into a fresh device buffer. Safe while a batch is
    /// open: the new buffer cannot be referenced by already-encoded kernels.
    pub(crate) fn batch_upload_f32(&self, values: &[f32]) -> Result<Buffer> {
        f32_buffer(&self.device, values)
    }

    /// Uploads host f32 values and converts them to a fresh f16 device buffer.
    /// The f32 staging buffer is short-lived; subsequent kernels consume only
    /// the half-precision output.
    pub(crate) fn batch_upload_f32_as_f16(&self, values: &[f32]) -> Result<Buffer> {
        let input = f32_buffer(&self.device, values)?;
        self.batch.encode(&self.queue, |command_buffer| {
            self.cast
                .encode_f32_to_f16(command_buffer, &self.device, &input, values.len())
        })
    }

    /// Uploads compressed Q8 cold-cache rows and decodes them into f32 on the
    /// GPU. MLA selected-row expansion uses this path because RMSNorm and Q2
    /// projections consume f32 device values today.
    pub(crate) fn batch_upload_q8_rows_as_f32(
        &self,
        payload: &[u8],
        row_count: usize,
        dim: usize,
    ) -> Result<Buffer> {
        let input = u8_buffer(&self.device, payload)?;
        self.batch.encode(&self.queue, |command_buffer| {
            self.cast.encode_q8_rows_to_f32(
                command_buffer,
                &self.device,
                &input,
                payload.len(),
                row_count,
                dim,
            )
        })
    }

    /// Reads values out of a device buffer, synchronizing the open batch
    /// first so every encoded kernel that may write the buffer has completed.
    pub(crate) fn batch_read_f32(&self, buffer: &Buffer, len: usize) -> Result<Vec<f32>> {
        self.batch_flush()?;
        self.cast.read_f32(buffer, len)
    }

    /// Reads f16 values out of a device buffer and expands them to f32 on the
    /// CPU side. This is for validation/cold-tier serialization, not the hot
    /// decode path.
    pub(crate) fn batch_read_f16_as_f32(&self, buffer: &Buffer, len: usize) -> Result<Vec<f32>> {
        self.batch_flush()?;
        self.cast.read_f16_as_f32(buffer, len)
    }

    pub(crate) fn batched_rms_norm(
        &self,
        input: &Buffer,
        input_len: usize,
        weight: &[f32],
        rows: usize,
        hidden_size: usize,
        eps: f32,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.rms_norm.encode(
                command_buffer,
                &self.device,
                input,
                input_len,
                weight,
                rows,
                hidden_size,
                eps,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_mla_kv_postprocess(
        &self,
        input: &Buffer,
        input_len: usize,
        norm_weight: &[f32],
        batch_count: usize,
        token_count: usize,
        latent_dim: usize,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
        eps: f32,
    ) -> Result<(Buffer, Buffer)> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.rms_norm.encode_mla_kv_postprocess(
                command_buffer,
                &self.device,
                input,
                input_len,
                norm_weight,
                batch_count,
                token_count,
                latent_dim,
                rope_dim,
                position_offset,
                theta,
                eps,
            )
        })
    }

    pub(crate) fn batched_linear_f32(
        &self,
        input: &Buffer,
        input_len: usize,
        weight: &[f32],
        rows: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.matmul.encode_linear(
                command_buffer,
                &self.device,
                input,
                input_len,
                weight,
                rows,
                in_features,
                out_features,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_quant_matvec(
        &self,
        kind: QuantMatvecKind,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.q2_matvec.encode_matvec(
                command_buffer,
                &self.device,
                kind,
                weights,
                input,
                input_len,
                row_count,
                in_features,
                out_features,
            )
        })
    }

    pub(crate) fn batched_q8_0_embedding(
        &self,
        weights: &[u8],
        token_ids: &[u32],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.q2_matvec.encode_q8_0_embedding(
                command_buffer,
                &self.device,
                weights,
                token_ids,
                vocab_size,
                hidden_size,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_laguna_xs_k_matvec(
        &self,
        quant: GgufKQuant,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            if MetalLagunaXsMpsPrefill::supports(row_count)
                && self.laguna_xs_mps_prefill.is_prepared(
                    quant,
                    weights,
                    in_features,
                    out_features,
                )?
            {
                return self.laguna_xs_mps_prefill.encode(
                    command_buffer,
                    &self.device,
                    quant,
                    weights,
                    input,
                    input_len,
                    None,
                    None,
                    row_count,
                    in_features,
                    out_features,
                );
            }
            if MetalLagunaXsPrefill::supports(row_count, out_features) {
                return self.laguna_xs_prefill.encode_matvec(
                    command_buffer,
                    &self.device,
                    quant,
                    weights,
                    input,
                    input_len,
                    row_count,
                    in_features,
                    out_features,
                );
            }
            self.laguna_xs.encode_matvec(
                command_buffer,
                &self.device,
                quant,
                weights,
                input,
                input_len,
                row_count,
                in_features,
                out_features,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_laguna_xs_k_matvec_residuals(
        &self,
        quant: GgufKQuant,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        residual_a: &Buffer,
        residual_b: Option<&Buffer>,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            if MetalLagunaXsMpsPrefill::supports(row_count)
                && self.laguna_xs_mps_prefill.is_prepared(
                    quant,
                    weights,
                    in_features,
                    out_features,
                )?
            {
                return self.laguna_xs_mps_prefill.encode(
                    command_buffer,
                    &self.device,
                    quant,
                    weights,
                    input,
                    input_len,
                    Some(residual_a),
                    residual_b,
                    row_count,
                    in_features,
                    out_features,
                );
            }
            if MetalLagunaXsPrefill::supports(row_count, out_features) {
                return self.laguna_xs_prefill.encode_matvec_residuals(
                    command_buffer,
                    &self.device,
                    quant,
                    weights,
                    input,
                    input_len,
                    residual_a,
                    residual_b,
                    row_count,
                    in_features,
                    out_features,
                );
            }
            self.laguna_xs.encode_matvec_residuals(
                command_buffer,
                &self.device,
                quant,
                weights,
                input,
                input_len,
                residual_a,
                residual_b,
                row_count,
                in_features,
                out_features,
            )
        })
    }

    pub(crate) fn prepare_laguna_xs_mps_prefill_weight(
        &self,
        quant: GgufKQuant,
        weights: &[u8],
        in_features: usize,
        out_features: usize,
    ) -> Result<()> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.laguna_xs_mps_prefill.encode_prepare_weight(
                command_buffer,
                &self.device,
                quant,
                weights,
                in_features,
                out_features,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_laguna_xs_q4_gate_up_swiglu(
        &self,
        gate_weights: &[u8],
        up_weights: &[u8],
        input: &Buffer,
        input_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.laguna_xs.encode_gate_up_swiglu(
                command_buffer,
                &self.device,
                gate_weights,
                up_weights,
                input,
                input_len,
                row_count,
                in_features,
                out_features,
            )
        })
    }

    pub(crate) fn batched_laguna_xs_router_topk(
        &self,
        router_logits: &Buffer,
        router_logits_len: usize,
        correction_bias: &[f32],
        routed_scaling_factor: f32,
    ) -> Result<DeviceRouterTopK> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.laguna_xs.encode_router_topk(
                command_buffer,
                &self.device,
                router_logits,
                router_logits_len,
                correction_bias,
                routed_scaling_factor,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_laguna_xs_qk_rms_norm_rope_pair(
        &self,
        query: &Buffer,
        query_len: usize,
        key: &Buffer,
        key_len: usize,
        query_norm_weight: &[f32],
        key_norm_weight: &[f32],
        query_head_count: usize,
        position_offset: usize,
        eps: f32,
        table: &DeviceRopeTable,
    ) -> Result<(Buffer, Buffer)> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.laguna_xs.encode_qk_norm_rope_pair(
                command_buffer,
                &self.device,
                query,
                query_len,
                key,
                key_len,
                query_norm_weight,
                key_norm_weight,
                query_head_count,
                position_offset,
                eps,
                table,
            )
        })
    }

    pub(crate) fn batched_laguna_xs_rms_norm(
        &self,
        input: &Buffer,
        input_len: usize,
        weight: &[f32],
        eps: f32,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.laguna_xs.encode_rms_norm(
                command_buffer,
                &self.device,
                input,
                input_len,
                weight,
                eps,
            )
        })
    }

    pub(crate) fn batched_laguna_xs_rms_norm_router(
        &self,
        input: &Buffer,
        input_len: usize,
        norm_weight: &[f32],
        router_weight: &[f32],
        eps: f32,
    ) -> Result<(Buffer, Buffer)> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.laguna_xs.encode_rms_norm_router(
                command_buffer,
                &self.device,
                input,
                input_len,
                norm_weight,
                router_weight,
                eps,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_laguna_xs_attention_projections(
        &self,
        query_weights: &[u8],
        key_weights: &[u8],
        value_weights: &[u8],
        value_quant: GgufKQuant,
        gate_weights: &[u8],
        input: &Buffer,
        input_len: usize,
        in_features: usize,
        query_features: usize,
        key_features: usize,
        value_features: usize,
        gate_features: usize,
    ) -> Result<[Buffer; 4]> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.laguna_xs.encode_attention_projections(
                command_buffer,
                &self.device,
                query_weights,
                key_weights,
                value_weights,
                value_quant,
                gate_weights,
                input,
                input_len,
                in_features,
                query_features,
                key_features,
                value_features,
                gate_features,
            )
        })
    }

    pub(crate) fn batched_laguna_xs_q4_embedding(
        &self,
        weights: &[u8],
        token_ids: &[u32],
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.laguna_xs.encode_q4_embedding(
                command_buffer,
                &self.device,
                weights,
                token_ids,
                vocab_size,
                hidden_size,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_laguna_xs_gguf_moe(
        &self,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        down_quant: GgufKQuant,
        input: &Buffer,
        input_len: usize,
        routing: &DeviceRouterTopK,
        in_features: usize,
        intermediate_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            if MetalLagunaXsMoePrefill::supports(routing) {
                return self.laguna_xs_moe_prefill.encode(
                    command_buffer,
                    &self.device,
                    gate_weights,
                    up_weights,
                    down_weights,
                    down_quant,
                    input,
                    input_len,
                    routing,
                    in_features,
                    intermediate_features,
                    out_features,
                );
            }
            self.laguna_xs.encode_moe(
                command_buffer,
                &self.device,
                gate_weights,
                up_weights,
                down_weights,
                down_quant,
                input,
                input_len,
                routing,
                in_features,
                intermediate_features,
                out_features,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_laguna_xs_gguf_moe_shared(
        &self,
        routed_gate_weights: &[u8],
        routed_up_weights: &[u8],
        routed_down_weights: &[u8],
        shared_gate_weights: &[u8],
        shared_up_weights: &[u8],
        shared_down_weights: &[u8],
        down_quant: GgufKQuant,
        input: &Buffer,
        input_len: usize,
        routing: &DeviceRouterTopK,
        residual: &Buffer,
        residual_len: usize,
        in_features: usize,
        intermediate_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.laguna_xs.encode_moe_shared(
                command_buffer,
                &self.device,
                routed_gate_weights,
                routed_up_weights,
                routed_down_weights,
                shared_gate_weights,
                shared_up_weights,
                shared_down_weights,
                down_quant,
                input,
                input_len,
                routing,
                residual,
                residual_len,
                in_features,
                intermediate_features,
                out_features,
            )
        })
    }

    pub(crate) fn prepare_w4_groupwise_weight(
        &self,
        packed: &[u8],
        scales: &[u8],
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> Result<DeviceW4Weight> {
        self.w4.prepare(
            &self.device,
            packed,
            scales,
            in_features,
            out_features,
            group_size,
        )
    }

    pub(crate) fn prepare_w4_groupwise_weight_no_copy(
        &self,
        source: Arc<dyn W4WeightSource>,
        in_features: usize,
        out_features: usize,
        group_size: usize,
    ) -> Result<DeviceW4Weight> {
        self.w4
            .prepare_no_copy(&self.device, source, in_features, out_features, group_size)
    }

    pub(crate) fn batched_w4_groupwise_matvec(
        &self,
        weight: &DeviceW4Weight,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.w4
                .encode_matvec(command_buffer, weight, input, input_len, row_count)
        })
    }

    pub(crate) fn batched_w4_groupwise_gate_up_swiglu(
        &self,
        gate: &DeviceW4Weight,
        up: &DeviceW4Weight,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.w4
                .encode_gate_up_swiglu(command_buffer, gate, up, input, input_len, row_count)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_w4_expert_wave(
        &self,
        groups: &[W4ExpertGroup<'_>],
        input: &Buffer,
        input_len: usize,
        token_count: usize,
        top_k: usize,
        destination: &Buffer,
        destination_len: usize,
    ) -> Result<()> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.w4.encode_expert_wave(
                command_buffer,
                &self.device,
                groups,
                input,
                input_len,
                token_count,
                top_k,
                destination,
                destination_len,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_laguna_gguf_moe(
        &self,
        gate_weights: &[u8],
        up_weights: &[u8],
        down_weights: &[u8],
        quant: GgufExpertQuant,
        input: &Buffer,
        input_len: usize,
        routing: &DeviceRouterTopK,
        in_features: usize,
        intermediate_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            if routing.token_count > 1 && routing.top_k == 10 && routing.expert_count == 256 {
                return self.gguf_moe_prefill.encode(
                    command_buffer,
                    &self.device,
                    gate_weights,
                    up_weights,
                    down_weights,
                    quant,
                    input,
                    input_len,
                    routing,
                    in_features,
                    intermediate_features,
                    out_features,
                );
            }
            self.gguf_moe.encode(
                command_buffer,
                &self.device,
                gate_weights,
                up_weights,
                down_weights,
                quant,
                input,
                input_len,
                routing,
                in_features,
                intermediate_features,
                out_features,
            )
        })
    }

    pub(crate) fn prepare_bf16_matrix(
        &self,
        bytes: &[u8],
        rows: usize,
        columns: usize,
    ) -> Result<DeviceBf16Matrix> {
        self.bf16.prepare(&self.device, bytes, rows, columns)
    }

    pub(crate) fn batched_bf16_linear(
        &self,
        matrix: &DeviceBf16Matrix,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.bf16
                .encode_linear(command_buffer, matrix, input, input_len, row_count)
        })
    }

    pub(crate) fn batched_bf16_gate_up_swiglu(
        &self,
        gate: &DeviceBf16Matrix,
        up: &DeviceBf16Matrix,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.bf16
                .encode_gate_up_swiglu(command_buffer, gate, up, input, input_len, row_count)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_laguna_attention_projections(
        &self,
        query: &DeviceBf16Matrix,
        key: &DeviceBf16Matrix,
        value: &DeviceBf16Matrix,
        gate: &DeviceBf16Matrix,
        input: &Buffer,
        input_len: usize,
        row_count: usize,
    ) -> Result<(Buffer, Buffer, Buffer, Buffer)> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.bf16.encode_laguna_attention_projections(
                command_buffer,
                query,
                key,
                value,
                gate,
                input,
                input_len,
                row_count,
            )
        })
    }

    pub(crate) fn batched_bf16_embedding(
        &self,
        matrix: &DeviceBf16Matrix,
        token_ids: &[u32],
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.bf16
                .encode_embedding(command_buffer, &self.device, matrix, token_ids)
        })
    }

    pub(crate) fn prepare_rope_table(
        &self,
        inverse_frequency: &[f32],
        rotary_dim: usize,
        attention_factor: f32,
    ) -> Result<DeviceRopeTable> {
        self.rope.prepare_table(
            &self.device,
            inverse_frequency,
            rotary_dim,
            attention_factor,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_qk_rms_norm_rope(
        &self,
        input: &Buffer,
        input_len: usize,
        norm_weight: &[f32],
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        position_offset: usize,
        eps: f32,
        table: &DeviceRopeTable,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.rope.encode_qk_norm_rope(
                command_buffer,
                &self.device,
                input,
                input_len,
                norm_weight,
                batch_count,
                token_count,
                head_count,
                position_offset,
                eps,
                table,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_laguna_qk_rms_norm_rope_pair(
        &self,
        query: &Buffer,
        query_len: usize,
        key: &Buffer,
        key_len: usize,
        query_norm_weight: &[f32],
        key_norm_weight: &[f32],
        batch_count: usize,
        token_count: usize,
        query_head_count: usize,
        key_head_count: usize,
        position_offset: usize,
        eps: f32,
        table: &DeviceRopeTable,
    ) -> Result<(Buffer, Buffer)> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.rope.encode_qk_norm_rope_pair(
                command_buffer,
                &self.device,
                query,
                query_len,
                key,
                key_len,
                query_norm_weight,
                key_norm_weight,
                batch_count,
                token_count,
                query_head_count,
                key_head_count,
                position_offset,
                eps,
                table,
            )
        })
    }

    pub(crate) fn prepare_laguna_fp8_kv_cache(
        &self,
        batch: usize,
        capacity_tokens: usize,
        retention: LagunaKvRetention,
        key_scale: f32,
        value_scale: f32,
    ) -> Result<LagunaFp8KvCache> {
        self.fp8_attention.prepare_cache(
            &self.device,
            batch,
            capacity_tokens,
            retention,
            key_scale,
            value_scale,
        )
    }

    pub(crate) fn grow_laguna_fp8_kv_cache(
        &self,
        cache: &mut LagunaFp8KvCache,
        capacity_tokens: usize,
    ) -> Result<()> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.fp8_attention
                .grow_cache(&self.device, command_buffer, cache, capacity_tokens)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_laguna_gated_gqa_attention(
        &self,
        query: &Buffer,
        query_len: usize,
        current_key: &Buffer,
        current_key_len: usize,
        current_value: &Buffer,
        current_value_len: usize,
        gate: &Buffer,
        gate_len: usize,
        batch: usize,
        query_tokens: usize,
        query_heads: usize,
        cache: &LagunaFp8KvCache,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.fp8_attention.encode_attention_and_append(
                command_buffer,
                query,
                query_len,
                current_key,
                current_key_len,
                current_value,
                current_value_len,
                gate,
                gate_len,
                batch,
                query_tokens,
                query_heads,
                cache,
            )
        })
    }

    pub(crate) fn prepare_laguna_f16_kv_cache(
        &self,
        batch: usize,
        capacity_tokens: usize,
        retention: LagunaKvRetention,
    ) -> Result<LagunaF16KvCache> {
        self.f16_attention
            .prepare_cache(&self.device, batch, capacity_tokens, retention)
    }

    pub(crate) fn grow_laguna_f16_kv_cache(
        &self,
        cache: &mut LagunaF16KvCache,
        capacity_tokens: usize,
    ) -> Result<()> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.f16_attention
                .grow_cache(&self.device, command_buffer, cache, capacity_tokens)
        })
    }

    pub(crate) fn checkpoint_laguna_f16_kv_cache(
        &self,
        cache: &mut LagunaF16KvCache,
    ) -> Result<()> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.f16_attention
                .checkpoint_cache(&self.device, command_buffer, cache)
        })
    }

    pub(crate) fn restore_laguna_f16_kv_cache_checkpoint(
        &self,
        cache: &mut LagunaF16KvCache,
    ) -> Result<()> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.f16_attention
                .restore_cache_checkpoint(command_buffer, cache)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_laguna_gated_gqa_f16_attention(
        &self,
        query: &Buffer,
        query_len: usize,
        current_key: &Buffer,
        current_key_len: usize,
        current_value: &Buffer,
        current_value_len: usize,
        gate: &Buffer,
        gate_len: usize,
        batch: usize,
        query_tokens: usize,
        query_heads: usize,
        cache: &LagunaF16KvCache,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.f16_attention.encode_attention_and_append(
                command_buffer,
                query,
                query_len,
                current_key,
                current_key_len,
                current_value,
                current_value_len,
                gate,
                gate_len,
                batch,
                query_tokens,
                query_heads,
                cache,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_q8_0_matvec_pair(
        &self,
        weights_a: &[u8],
        weights_b: &[u8],
        input: &Buffer,
        input_len: usize,
        row_count: usize,
        in_features: usize,
        out_features_a: usize,
        out_features_b: usize,
    ) -> Result<(Buffer, Buffer)> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.q2_matvec.encode_q8_0_matvec_pair(
                command_buffer,
                &self.device,
                weights_a,
                weights_b,
                input,
                input_len,
                row_count,
                in_features,
                out_features_a,
                out_features_b,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_laguna_q8_0_attention_projections(
        &self,
        query_weights: &[u8],
        key_weights: &[u8],
        value_weights: &[u8],
        gate_weights: &[u8],
        input: &Buffer,
        input_len: usize,
        row_count: usize,
        in_features: usize,
        query_features: usize,
        key_features: usize,
        value_features: usize,
        gate_features: usize,
    ) -> Result<[Buffer; 4]> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.q2_matvec.encode_laguna_q8_0_attention_projections(
                command_buffer,
                &self.device,
                query_weights,
                key_weights,
                value_weights,
                gate_weights,
                input,
                input_len,
                row_count,
                in_features,
                query_features,
                key_features,
                value_features,
                gate_features,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_q8_0_gate_up_swiglu(
        &self,
        gate_weights: &[u8],
        up_weights: &[u8],
        input: &Buffer,
        input_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            if row_count >= Q8_0_MMA_MIN_PREFILL_ROWS
                && out_features.is_multiple_of(Q8_0_MMA_OUTPUT_TILE)
            {
                return self.q2_matvec.encode_q8_0_prefill_mma_gate_up_swiglu(
                    command_buffer,
                    &self.device,
                    gate_weights,
                    up_weights,
                    input,
                    input_len,
                    row_count,
                    in_features,
                    out_features,
                );
            }

            self.q2_matvec.encode_q8_0_gate_up_swiglu(
                command_buffer,
                &self.device,
                gate_weights,
                up_weights,
                input,
                input_len,
                row_count,
                in_features,
                out_features,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_packed_heads_transposed_matvec(
        &self,
        kind: QuantMatvecKind,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        row_count: usize,
        head_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.q2_matvec.encode_packed_heads_transposed_matvec(
                command_buffer,
                &self.device,
                kind,
                weights,
                input,
                input_len,
                row_count,
                head_count,
                in_features,
                out_features,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_q2_k_matvec_add(
        &self,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        residual: &Buffer,
        residual_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.q2_matvec.encode_matvec_add(
                command_buffer,
                &self.device,
                weights,
                input,
                input_len,
                residual,
                residual_len,
                row_count,
                in_features,
                out_features,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_q8_0_matvec_add(
        &self,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        residual: &Buffer,
        residual_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            if row_count >= Q8_0_MMA_MIN_PREFILL_ROWS
                && out_features.is_multiple_of(Q8_0_MMA_OUTPUT_TILE)
            {
                return self.q2_matvec.encode_q8_0_prefill_mma_add(
                    command_buffer,
                    &self.device,
                    weights,
                    input,
                    input_len,
                    residual,
                    residual_len,
                    row_count,
                    in_features,
                    out_features,
                );
            }

            self.q2_matvec.encode_q8_0_matvec_add(
                command_buffer,
                &self.device,
                weights,
                input,
                input_len,
                residual,
                residual_len,
                row_count,
                in_features,
                out_features,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_laguna_q8_0_matvec_add2(
        &self,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        residual_a: &Buffer,
        residual_a_len: usize,
        residual_b: &Buffer,
        residual_b_len: usize,
        row_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            if row_count >= Q8_0_MMA_MIN_PREFILL_ROWS
                && out_features.is_multiple_of(Q8_0_MMA_OUTPUT_TILE)
            {
                return self.q2_matvec.encode_q8_0_prefill_mma_add2(
                    command_buffer,
                    &self.device,
                    weights,
                    input,
                    input_len,
                    residual_a,
                    residual_a_len,
                    residual_b,
                    residual_b_len,
                    row_count,
                    in_features,
                    out_features,
                );
            }

            self.q2_matvec.encode_laguna_q8_0_matvec_add2(
                command_buffer,
                &self.device,
                weights,
                input,
                input_len,
                residual_a,
                residual_a_len,
                residual_b,
                residual_b_len,
                row_count,
                in_features,
                out_features,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_q2_k_multi_expert_gate_up_swiglu(
        &self,
        gate_weights: &[u8],
        up_weights: &[u8],
        input: &Buffer,
        input_len: usize,
        token_indices: &[u32],
        expert_ids: &[u32],
        token_count: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.q2_matvec.encode_multi_expert_gate_up_swiglu(
                command_buffer,
                &self.device,
                gate_weights,
                up_weights,
                input,
                input_len,
                token_indices,
                expert_ids,
                token_count,
                in_features,
                out_features,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_q2_k_multi_expert_matvec(
        &self,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        expert_ids: &[u32],
        in_features: usize,
        out_features: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.q2_matvec.encode_multi_expert_matvec(
                command_buffer,
                &self.device,
                weights,
                input,
                input_len,
                expert_ids,
                in_features,
                out_features,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ready_routed_experts(
        &self,
        layer_index: usize,
        model_path: &Path,
        gate_payloads: &[Q2ExpertSource<'_>],
        up_payloads: &[Q2ExpertSource<'_>],
        down_payloads: &[Q2ExpertSource<'_>],
        input: &Buffer,
        input_len: usize,
        routing: &DeviceRouterTopK,
        in_features: usize,
        intermediate_features: usize,
        out_features: usize,
    ) -> Result<ReadyRoutedExperts> {
        self.q2_matvec.run_ready_routed_experts(
            &self.device,
            layer_index,
            model_path,
            gate_payloads,
            up_payloads,
            down_payloads,
            input,
            input_len,
            &routing.token_indices,
            routing.token_count,
            routing.top_k,
            in_features,
            intermediate_features,
            out_features,
        )
    }

    pub(crate) fn prefetch_routed_experts(
        &self,
        layer_index: usize,
        model_path: &Path,
        gate_payloads: &[Q2ExpertSource<'_>],
        up_payloads: &[Q2ExpertSource<'_>],
        down_payloads: &[Q2ExpertSource<'_>],
    ) -> Result<()> {
        self.q2_matvec.prefetch_ready_routed_experts(
            &self.device,
            layer_index,
            model_path,
            gate_payloads,
            up_payloads,
            down_payloads,
        )
    }

    /// Adds a GPU-side dependency immediately before a routed-expert consumer.
    /// The CPU does not wait here: Metal starts subsequent work as soon as the
    /// expert queue signals that every output row is complete.
    pub(crate) fn batched_wait_for_ready_routed_experts(
        &self,
        completion_value: u64,
    ) -> Result<()> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.q2_matvec
                .encode_ready_expert_wait(command_buffer, completion_value)
        })
    }

    /// Encodes the output-head matvec + greedy argmax against a device-resident
    /// hidden state, then flushes the whole batch and returns the chosen token.
    /// This is the natural end-of-token synchronization point.
    pub(crate) fn batched_q2_k_matvec_argmax(
        &self,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<(u32, f32)> {
        let (token_id_buffer, token_score_buffer) =
            self.batch.encode(&self.queue, |command_buffer| {
                self.q2_matvec.encode_argmax(
                    command_buffer,
                    &self.device,
                    weights,
                    input,
                    input_len,
                    1,
                    in_features,
                    out_features,
                )
            })?;
        self.batch_flush()?;
        let token_id = read_u32_buffer(&token_id_buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::backend("Q2_K argmax produced no token id"))?;
        let token_score = read_f32_buffer(&token_score_buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::backend("Q2_K argmax produced no token score"))?;
        Ok((token_id, token_score))
    }

    pub(crate) fn batched_laguna_q8_0_matvec_argmax(
        &self,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<(u32, f32)> {
        let (token_id_buffer, token_score_buffer) =
            self.batch.encode(&self.queue, |command_buffer| {
                self.q2_matvec.encode_laguna_q8_0_matvec_argmax(
                    command_buffer,
                    &self.device,
                    weights,
                    input,
                    input_len,
                    in_features,
                    out_features,
                )
            })?;
        self.batch_flush()?;
        let token_id = read_u32_buffer(&token_id_buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::backend("Laguna Q8_0 argmax produced no token id"))?;
        let token_score = read_f32_buffer(&token_score_buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::backend("Laguna Q8_0 argmax produced no token score"))?;
        Ok((token_id, token_score))
    }

    pub(crate) fn batched_laguna_xs_k_matvec_argmax(
        &self,
        quant: GgufKQuant,
        weights: &[u8],
        input: &Buffer,
        input_len: usize,
        in_features: usize,
        out_features: usize,
    ) -> Result<(u32, f32)> {
        let (token_id_buffer, token_score_buffer) =
            self.batch.encode(&self.queue, |command_buffer| {
                let (candidate_ids, candidate_scores, candidate_count) =
                    self.laguna_xs.encode_matvec_argmax_candidates(
                        command_buffer,
                        &self.device,
                        quant,
                        weights,
                        input,
                        input_len,
                        in_features,
                        out_features,
                    )?;
                self.q2_matvec.encode_candidate_argmax(
                    command_buffer,
                    &candidate_ids,
                    &candidate_scores,
                    candidate_count,
                )
            })?;
        self.batch_flush()?;
        let token_id = read_u32_buffer(&token_id_buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::backend("Laguna XS argmax produced no token id"))?;
        let token_score = read_f32_buffer(&token_score_buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::backend("Laguna XS argmax produced no token score"))?;
        Ok((token_id, token_score))
    }

    /// Encodes greedy argmax over an existing device-resident f32 score
    /// buffer, then flushes the batch and returns the chosen token.
    pub(crate) fn batched_f32_argmax(
        &self,
        scores: &Buffer,
        value_count: usize,
    ) -> Result<(u32, f32)> {
        let (token_id_buffer, token_score_buffer) =
            self.batch.encode(&self.queue, |command_buffer| {
                self.q2_matvec
                    .encode_f32_argmax(command_buffer, &self.device, scores, value_count)
            })?;
        self.batch_flush()?;
        let token_id = read_u32_buffer(&token_id_buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::backend("f32 argmax produced no token id"))?;
        let token_score = read_f32_buffer(&token_score_buffer, 1)?
            .into_iter()
            .next()
            .ok_or_else(|| Error::backend("f32 argmax produced no token score"))?;
        Ok((token_id, token_score))
    }

    pub(crate) fn batched_f32_argmax_rows(
        &self,
        scores: &Buffer,
        row_count: usize,
        row_width: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let (token_id_buffer, token_score_buffer) =
            self.batch.encode(&self.queue, |command_buffer| {
                self.q2_matvec.encode_f32_argmax_rows(
                    command_buffer,
                    &self.device,
                    scores,
                    row_count,
                    row_width,
                )
            })?;
        self.batch_flush()?;
        Ok((
            read_u32_buffer(&token_id_buffer, row_count)?,
            read_f32_buffer(&token_score_buffer, row_count)?,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_rope_slice(
        &self,
        input: &Buffer,
        input_len: usize,
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.rope.encode(
                command_buffer,
                &self.device,
                input,
                input_len,
                batch_count,
                token_count,
                head_count,
                rope_dim,
                position_offset,
                theta,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_split_rope_tail(
        &self,
        input: &Buffer,
        input_len: usize,
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        no_rope_dim: usize,
        rope_dim: usize,
    ) -> Result<(Buffer, Buffer)> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.layout.encode_split_rope_tail(
                command_buffer,
                &self.device,
                input,
                input_len,
                batch_count,
                token_count,
                head_count,
                no_rope_dim,
                rope_dim,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_split_kv_mqa(
        &self,
        input: &Buffer,
        input_len: usize,
        batch_count: usize,
        token_count: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
    ) -> Result<(Buffer, Buffer)> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.layout.encode_split_kv_mqa(
                command_buffer,
                &self.device,
                input,
                input_len,
                batch_count,
                token_count,
                kv_lora_rank,
                rope_dim,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_combine_rope_tail(
        &self,
        no_rope: &Buffer,
        no_rope_len: usize,
        rope: &Buffer,
        rope_len: usize,
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        rope_head_count: usize,
        no_rope_dim: usize,
        rope_dim: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.layout.encode_combine_rope_tail(
                command_buffer,
                &self.device,
                no_rope,
                no_rope_len,
                rope,
                rope_len,
                batch_count,
                token_count,
                head_count,
                rope_head_count,
                no_rope_dim,
                rope_dim,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_heads_to_attention_layout(
        &self,
        input: &Buffer,
        input_len: usize,
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        head_dim: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.layout.encode_heads_to_attention_layout(
                command_buffer,
                &self.device,
                input,
                input_len,
                batch_count,
                token_count,
                head_count,
                head_dim,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_merge_attention_heads(
        &self,
        input: &Buffer,
        input_len: usize,
        batch_count: usize,
        head_count: usize,
        token_count: usize,
        head_dim: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.layout.encode_merge_attention_heads(
                command_buffer,
                &self.device,
                input,
                input_len,
                batch_count,
                head_count,
                token_count,
                head_dim,
            )
        })
    }

    pub(crate) fn batched_stack_head_outputs(
        &self,
        head_outputs: &[&Buffer],
        row_count: usize,
        head_dim: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.layout.encode_stack_head_outputs(
                command_buffer,
                &self.device,
                head_outputs,
                row_count,
                head_dim,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_linearize_paged_cache(
        &self,
        paged: &Buffer,
        paged_len: usize,
        batch_count: usize,
        head_count: usize,
        cached_tokens: usize,
        capacity_tokens: usize,
        page_size: usize,
        head_dim: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.layout.encode_linearize_paged_cache(
                command_buffer,
                &self.device,
                paged,
                paged_len,
                batch_count,
                head_count,
                cached_tokens,
                capacity_tokens,
                page_size,
                head_dim,
            )
        })
    }

    pub(crate) fn batched_select_last_token(
        &self,
        input: &Buffer,
        input_len: usize,
        batch_count: usize,
        token_count: usize,
        hidden_size: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.layout.encode_select_last_token(
                command_buffer,
                &self.device,
                input,
                input_len,
                batch_count,
                token_count,
                hidden_size,
            )
        })
    }

    pub(crate) fn batched_add(
        &self,
        lhs: &Buffer,
        lhs_len: usize,
        rhs: &Buffer,
        rhs_len: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.activation
                .encode_add(command_buffer, &self.device, lhs, lhs_len, rhs, rhs_len)
        })
    }

    pub(crate) fn batched_swiglu(
        &self,
        gate: &Buffer,
        gate_len: usize,
        up: &Buffer,
        up_len: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.activation
                .encode_swiglu(command_buffer, &self.device, gate, gate_len, up, up_len)
        })
    }

    pub(crate) fn batched_moe_gather_rows(
        &self,
        input: &Buffer,
        input_len: usize,
        token_indices: &[u32],
        token_count: usize,
        hidden_size: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.moe.encode_gather_tokens(
                command_buffer,
                &self.device,
                input,
                input_len,
                token_indices,
                token_count,
                hidden_size,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_moe_scatter_rows(
        &self,
        rows: &Buffer,
        rows_len: usize,
        destination_rows: &[u32],
        destination: &Buffer,
        destination_len: usize,
        destination_row_count: usize,
        hidden_size: usize,
    ) -> Result<()> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.moe.encode_scatter_rows(
                command_buffer,
                &self.device,
                rows,
                rows_len,
                destination_rows,
                destination,
                destination_len,
                destination_row_count,
                hidden_size,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_moe_weighted_index_add_combine(
        &self,
        accumulator: &Buffer,
        accumulator_len: usize,
        token_indices: &[u32],
        expert_outputs: &Buffer,
        expert_outputs_len: usize,
        expert_weights: &[f32],
        token_count: usize,
        hidden_size: usize,
        assignment_count: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.moe.encode_weighted_index_add_combine(
                command_buffer,
                &self.device,
                accumulator,
                accumulator_len,
                token_indices,
                expert_outputs,
                expert_outputs_len,
                expert_weights,
                token_count,
                hidden_size,
                assignment_count,
            )
        })
    }

    pub(crate) fn batched_moe_topk_combine_residual(
        &self,
        shared: &Buffer,
        shared_len: usize,
        residual: &Buffer,
        residual_len: usize,
        expert_outputs: &Buffer,
        expert_outputs_len: usize,
        routing: &DeviceRouterTopK,
        hidden_size: usize,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.moe.encode_topk_combine_residual(
                command_buffer,
                &self.device,
                shared,
                shared_len,
                residual,
                residual_len,
                expert_outputs,
                expert_outputs_len,
                &routing.expert_weights,
                routing.token_count,
                hidden_size,
                routing.top_k,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_moe_router_topk_resident(
        &self,
        router_logits: &Buffer,
        router_logits_len: usize,
        correction_bias: &[f32],
        token_count: usize,
        expert_count: usize,
        top_k: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
    ) -> Result<DeviceRouterTopK> {
        let buffers = self.batch.encode(&self.queue, |command_buffer| {
            self.moe.encode_router_topk(
                command_buffer,
                &self.device,
                router_logits,
                router_logits_len,
                correction_bias,
                token_count,
                expert_count,
                top_k,
                norm_topk_prob,
                routed_scaling_factor,
            )
        })?;
        Ok(DeviceRouterTopK {
            token_count,
            expert_count,
            top_k,
            token_indices: buffers.token_indices,
            expert_ids: buffers.expert_ids,
            expert_weights: buffers.expert_weights,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_moe_router_topk(
        &self,
        router_logits: &Buffer,
        router_logits_len: usize,
        correction_bias: &[f32],
        token_count: usize,
        expert_count: usize,
        top_k: usize,
        norm_topk_prob: bool,
        routed_scaling_factor: f32,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let routing = self.batched_moe_router_topk_resident(
            router_logits,
            router_logits_len,
            correction_bias,
            token_count,
            expert_count,
            top_k,
            norm_topk_prob,
            routed_scaling_factor,
        )?;
        self.batch_flush()?;
        let output_len = routing.assignment_count()?;
        let expert_ids = read_u32_buffer(&routing.expert_ids, output_len)?;
        let expert_weights = read_f32_buffer(&routing.expert_weights, output_len)?;
        Ok((expert_ids, expert_weights))
    }

    pub(crate) fn batched_moe_router_expert_ids(
        &self,
        routing: &DeviceRouterTopK,
    ) -> Result<Vec<u32>> {
        self.batch_flush()?;
        read_u32_buffer(&routing.expert_ids, routing.assignment_count()?)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_dsa_index_key(
        &self,
        raw_key: &Buffer,
        raw_key_len: usize,
        weight: &[f32],
        bias: &[f32],
        batch: usize,
        tokens: usize,
        head_dim: usize,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
    ) -> Result<Buffer> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.dsa.encode_key_norm_rope(
                command_buffer,
                &self.device,
                raw_key,
                raw_key_len,
                weight,
                bias,
                batch,
                tokens,
                head_dim,
                rope_dim,
                position_offset,
                theta,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_dsa_decode_topk(
        &self,
        hidden_states: &Buffer,
        hidden_states_len: usize,
        q_raw: &Buffer,
        q_raw_len: usize,
        past_index_keys: &Buffer,
        past_index_keys_len: usize,
        current_index_key: &Buffer,
        current_index_key_len: usize,
        weights_proj: &[f32],
        batch: usize,
        hidden_size: usize,
        past_tokens: usize,
        heads: usize,
        head_dim: usize,
        rope_dim: usize,
        position_offset: usize,
        theta: f32,
        top_k: usize,
    ) -> Result<Vec<u32>> {
        let buffers = self.batch.encode(&self.queue, |command_buffer| {
            self.dsa.encode_decode_topk(
                command_buffer,
                &self.device,
                hidden_states,
                hidden_states_len,
                q_raw,
                q_raw_len,
                past_index_keys,
                past_index_keys_len,
                current_index_key,
                current_index_key_len,
                weights_proj,
                batch,
                hidden_size,
                past_tokens,
                heads,
                head_dim,
                rope_dim,
                position_offset,
                theta,
                top_k,
            )
        })?;
        self.batch_flush()?;
        MetalDsa::read_token_ids(&buffers.token_ids, buffers.output_len)
    }

    /// Encodes a GPU buffer-to-buffer copy (element offsets/length in f32).
    /// Used to stack per-expert outputs into one combine input without a host
    /// round-trip.
    pub(crate) fn batched_f32_copy(
        &self,
        source: &Buffer,
        source_offset: usize,
        destination: &Buffer,
        destination_offset: usize,
        len: usize,
    ) -> Result<()> {
        self.batch.encode(&self.queue, |command_buffer| {
            encode_f32_copy(
                command_buffer,
                source,
                source_offset,
                destination,
                destination_offset,
                len,
            )
        })
    }

    pub(crate) fn batched_element_copy(
        &self,
        source: &Buffer,
        source_offset: usize,
        destination: &Buffer,
        destination_offset: usize,
        len: usize,
        element_size: usize,
    ) -> Result<()> {
        self.batch.encode(&self.queue, |command_buffer| {
            encode_element_copy(
                command_buffer,
                source,
                source_offset,
                destination,
                destination_offset,
                len,
                element_size,
            )
        })
    }

    /// Allocates an uninitialized device buffer for `len` f32 elements, e.g.
    /// as the destination for `batched_f32_copy` stacking.
    pub(crate) fn batched_alloc_f32(&self, len: usize) -> Result<Buffer> {
        super::buffers::empty_f32_buffer(&self.device, len)
    }

    pub(crate) fn batched_alloc_f16(&self, len: usize) -> Result<Buffer> {
        empty_f16_buffer(&self.device, len)
    }

    /// Encodes fused paged decode attention. Returns the output buffer and
    /// its element count.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_paged_decode_attention(
        &self,
        q: &Buffer,
        q_len: usize,
        current_k: &Buffer,
        current_k_len: usize,
        current_v: &Buffer,
        current_v_len: usize,
        past_kv: &PagedKvView<'_>,
    ) -> Result<(Buffer, usize)> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.decode_attention.encode_paged(
                command_buffer,
                &self.device,
                q,
                q_len,
                current_k,
                current_k_len,
                current_v,
                current_v_len,
                past_kv,
            )
        })
    }

    /// Encodes fused paged decode attention using append-only resident K/V
    /// buffers. Returns the output buffer and its element count.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_paged_decode_attention_resident(
        &self,
        q: &Buffer,
        q_len: usize,
        current_k: &Buffer,
        current_k_len: usize,
        current_v: &Buffer,
        current_v_len: usize,
        past_kv: &DevicePagedKvView,
    ) -> Result<(Buffer, usize)> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.decode_attention.encode_paged_resident(
                command_buffer,
                &self.device,
                q,
                q_len,
                current_k,
                current_k_len,
                current_v,
                current_v_len,
                past_kv,
            )
        })
    }

    /// Encodes absorbed MLA for a short causal sequence into the current Metal
    /// batch: K_b query absorption, paged latent attention, then V_b output
    /// projection. No expanded historical or current K/V is materialized.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_q8_0_absorbed_mla(
        &self,
        k_b_weights: &[u8],
        v_b_weights: &[u8],
        q_no_rope: &Buffer,
        q_no_rope_len: usize,
        q_rope: &Buffer,
        q_rope_len: usize,
        current_latent: &Buffer,
        current_latent_len: usize,
        current_rope: &Buffer,
        current_rope_len: usize,
        past_kv: &DevicePagedKvView,
        batch_count: usize,
        token_count: usize,
        head_count: usize,
        q_no_rope_dim: usize,
        rope_dim: usize,
        latent_dim: usize,
        value_dim: usize,
        scale_dim: usize,
    ) -> Result<(Buffer, usize)> {
        if batch_count != past_kv.batch {
            return Err(Error::backend(format!(
                "absorbed MLA batch {batch_count} does not match cache batch {}",
                past_kv.batch
            )));
        }
        let row_count = batch_count
            .checked_mul(token_count)
            .ok_or_else(|| Error::backend("absorbed MLA row count overflow"))?;
        let absorbed_q_len = row_count
            .checked_mul(head_count)
            .and_then(|rows| rows.checked_mul(latent_dim))
            .ok_or_else(|| Error::backend("absorbed MLA query length overflow"))?;

        self.batch.encode(&self.queue, |command_buffer| {
            let absorbed_q = self.q2_matvec.encode_q8_0_packed_heads_matvec(
                command_buffer,
                &self.device,
                k_b_weights,
                q_no_rope,
                q_no_rope_len,
                row_count,
                head_count,
                q_no_rope_dim,
                latent_dim,
            )?;
            let (context_latent, context_latent_len) =
                self.decode_attention.encode_paged_absorbed_mla_f32(
                    command_buffer,
                    &self.device,
                    &absorbed_q,
                    absorbed_q_len,
                    q_rope,
                    q_rope_len,
                    current_latent,
                    current_latent_len,
                    current_rope,
                    current_rope_len,
                    past_kv,
                    token_count,
                    head_count,
                    latent_dim,
                    rope_dim,
                    scale_dim,
                )?;
            let output = self.q2_matvec.encode_q8_0_packed_heads_matvec(
                command_buffer,
                &self.device,
                v_b_weights,
                &context_latent,
                context_latent_len,
                row_count,
                head_count,
                latent_dim,
                value_dim,
            )?;
            let output_len = row_count
                .checked_mul(head_count)
                .and_then(|rows| rows.checked_mul(value_dim))
                .ok_or_else(|| Error::backend("absorbed MLA output length overflow"))?;
            Ok((output, output_len))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_selected_decode_attention(
        &self,
        q: &Buffer,
        q_len: usize,
        selected_k: &Buffer,
        selected_k_len: usize,
        selected_v: &Buffer,
        selected_v_len: usize,
        selected_kv_dtype: DType,
        current_k: &Buffer,
        current_k_len: usize,
        current_v: &Buffer,
        current_v_len: usize,
        batch_count: usize,
        head_count: usize,
        selected_tokens: usize,
        head_dim: usize,
        value_dim: usize,
        include_current_kv: bool,
    ) -> Result<(Buffer, usize)> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.decode_attention.encode_selected(
                command_buffer,
                &self.device,
                q,
                q_len,
                selected_k,
                selected_k_len,
                selected_v,
                selected_v_len,
                selected_kv_dtype,
                current_k,
                current_k_len,
                current_v,
                current_v_len,
                batch_count,
                head_count,
                selected_tokens,
                head_dim,
                value_dim,
                include_current_kv,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn batched_selected_sequence_attention(
        &self,
        q: &Buffer,
        q_len: usize,
        past_k: &Buffer,
        past_k_len: usize,
        past_v: &Buffer,
        past_v_len: usize,
        current_k: &Buffer,
        current_k_len: usize,
        current_v: &Buffer,
        current_v_len: usize,
        batch_count: usize,
        head_count: usize,
        past_tokens: usize,
        query_tokens: usize,
        head_dim: usize,
        value_dim: usize,
    ) -> Result<(Buffer, usize)> {
        self.batch.encode(&self.queue, |command_buffer| {
            self.decode_attention.encode_selected_sequence(
                command_buffer,
                &self.device,
                q,
                q_len,
                past_k,
                past_k_len,
                past_v,
                past_v_len,
                current_k,
                current_k_len,
                current_v,
                current_v_len,
                batch_count,
                head_count,
                past_tokens,
                query_tokens,
                head_dim,
                value_dim,
            )
        })
    }

    #[cfg(test)]
    pub fn moe_weighted_index_add_combine_f32_report(
        &self,
        accumulator: &[f32],
        token_indices: &[u32],
        expert_outputs: &[f32],
        expert_weights: &[f32],
        token_count: usize,
        hidden_size: usize,
        assignment_count: usize,
    ) -> Result<super::moe::MetalMoeCombineReport> {
        self.moe.weighted_index_add_combine(
            self.device(),
            self.queue(),
            accumulator,
            token_indices,
            expert_outputs,
            expert_weights,
            token_count,
            hidden_size,
            assignment_count,
        )
    }
}

fn select_native_device() -> Result<Device> {
    if let Some(device) = Device::system_default() {
        return Ok(device);
    }

    Device::all()
        .into_iter()
        .next()
        .ok_or_else(|| Error::backend("native Metal device is not available"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_rows_upload_decode_matches_cpu_reference() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let row_count = 3;
        let dim = 4;
        let payload = q8_payload(&[
            (0.5, [2_i8, -2, 4, -4]),
            (0.25, [1_i8, 2, 3, 4]),
            (1.0, [-1_i8, 0, 1, 2]),
        ]);
        let buffer = metal
            .batch_upload_q8_rows_as_f32(&payload, row_count, dim)
            .unwrap();
        let actual = metal.batch_read_f32(&buffer, row_count * dim).unwrap();
        let expected = vec![
            1.0, -1.0, 2.0, -2.0, 0.25, 0.5, 0.75, 1.0, -1.0, 0.0, 1.0, 2.0,
        ];

        assert_eq!(actual, expected);
    }

    fn native_metal_or_skip() -> Option<Metal> {
        Metal::new().ok()
    }

    fn q8_payload(rows: &[(f32, [i8; 4])]) -> Vec<u8> {
        let mut payload = Vec::with_capacity(rows.len() * 8);
        for (scale, values) in rows {
            payload.extend(scale.to_le_bytes());
            for value in values {
                payload.push(*value as u8);
            }
        }
        payload
    }
}
