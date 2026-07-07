use crate::DevicePagedKvView;
use ::metal::{Buffer, CommandQueue, Device};
use common::{DType, DeviceKind, Error, PagedKvView, Result};

use super::activation::MetalActivation;
use super::attention::{
    MetalAttentionCausalSoftmax, MetalAttentionScores, MetalAttentionValues, MetalDecodeAttention,
};
use super::batch::BatchSlot;
use super::buffers::{empty_f16_buffer, f32_buffer, read_f32_buffer, read_u32_buffer, u8_buffer};
use super::cast::MetalCast;
use super::command::{encode_element_copy, encode_f32_copy, Dispatch1d};
use super::dsa::MetalDsa;
use super::layout::MetalLayout;
use super::library::MetalLibrary;
use super::matmul::MetalMatmul;
use super::moe::MetalMoe;
use super::q2::{MetalQ2Matvec, QuantMatvecKind};
use super::rms_norm::MetalRmsNorm;
use super::rope::MetalRope;

pub struct Metal {
    device: Device,
    queue: CommandQueue,
    attention_scores: MetalAttentionScores,
    attention_values: MetalAttentionValues,
    attention_causal_softmax: MetalAttentionCausalSoftmax,
    decode_attention: MetalDecodeAttention,
    activation: MetalActivation,
    cast: MetalCast,
    layout: MetalLayout,
    matmul: MetalMatmul,
    q2_matvec: MetalQ2Matvec,
    rms_norm: MetalRmsNorm,
    rope: MetalRope,
    moe: MetalMoe,
    dsa: MetalDsa,
    batch: BatchSlot,
}

impl Metal {
    pub fn new() -> Result<Self> {
        let device = select_native_device()?;
        let queue = device.new_command_queue();
        let library = MetalLibrary::compile(&device)?;
        let attention_scores = MetalAttentionScores::new(&device, &library)?;
        let attention_values = MetalAttentionValues::new(&device, &library)?;
        let attention_causal_softmax = MetalAttentionCausalSoftmax::new(&device, &library)?;
        let decode_attention = MetalDecodeAttention::new(&device, &library)?;
        let activation = MetalActivation::new(&device, &library)?;
        let cast = MetalCast::new(&device, &library)?;
        let layout = MetalLayout::new(&device, &library)?;
        let matmul = MetalMatmul::new(&device, &library)?;
        let q2_matvec = MetalQ2Matvec::new(&device, &library)?;
        let rms_norm = MetalRmsNorm::new(&device, &library)?;
        let rope = MetalRope::new(&device, &library)?;
        let moe = MetalMoe::new(&device, &library)?;
        let dsa = MetalDsa::new(&device, &library)?;

        Ok(Self {
            device,
            queue,
            attention_scores,
            attention_values,
            attention_causal_softmax,
            decode_attention,
            activation,
            cast,
            layout,
            matmul,
            q2_matvec,
            rms_norm,
            rope,
            moe,
            dsa,
            batch: BatchSlot::new(),
        })
    }

    pub fn device_kind(&self) -> DeviceKind {
        DeviceKind::Metal
    }

    pub fn current_allocated_bytes(&self) -> u64 {
        self.device.current_allocated_size() as u64
    }

    pub fn recommended_max_working_set_bytes(&self) -> u64 {
        self.device.recommended_max_working_set_size()
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
            threads: row_count,
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

    /// Commits the open batch, if any, and waits for the GPU to finish it.
    pub fn batch_flush(&self) -> Result<()> {
        self.batch.flush()
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
        self.batch.flush()?;
        self.cast.read_f32(buffer, len)
    }

    /// Reads f16 values out of a device buffer and expands them to f32 on the
    /// CPU side. This is for validation/cold-tier serialization, not the hot
    /// decode path.
    pub(crate) fn batch_read_f16_as_f32(&self, buffer: &Buffer, len: usize) -> Result<Vec<f32>> {
        self.batch.flush()?;
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
        self.batch.flush()?;
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
        self.batch.flush()?;
        let expert_ids = read_u32_buffer(&buffers.expert_ids, buffers.output_len)?;
        let expert_weights = read_f32_buffer(&buffers.expert_weights, buffers.output_len)?;
        Ok((expert_ids, expert_weights))
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
        self.batch.flush()?;
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
            )
        })
    }

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
