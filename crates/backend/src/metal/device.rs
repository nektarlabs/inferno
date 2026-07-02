use ::metal::{CommandQueue, Device};
use common::{DeviceKind, Error, PagedKvView, Result};

use super::activation::MetalActivation;
use super::attention::{
    MetalAttentionCausalSoftmax, MetalAttentionScores, MetalAttentionValues, MetalDecodeAttention,
};
use super::layout::MetalLayout;
use super::library::MetalLibrary;
use super::matmul::MetalMatmul;
use super::moe::MetalMoe;
use super::q2::MetalQ2Matvec;
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
    layout: MetalLayout,
    matmul: MetalMatmul,
    q2_matvec: MetalQ2Matvec,
    rms_norm: MetalRmsNorm,
    rope: MetalRope,
    moe: MetalMoe,
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
        let layout = MetalLayout::new(&device, &library)?;
        let matmul = MetalMatmul::new(&device, &library)?;
        let q2_matvec = MetalQ2Matvec::new(&device, &library)?;
        let rms_norm = MetalRmsNorm::new(&device, &library)?;
        let rope = MetalRope::new(&device, &library)?;
        let moe = MetalMoe::new(&device, &library)?;

        Ok(Self {
            device,
            queue,
            attention_scores,
            attention_values,
            attention_causal_softmax,
            decode_attention,
            activation,
            layout,
            matmul,
            q2_matvec,
            rms_norm,
            rope,
            moe,
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
        let normed = self.rms_norm.run_to_buffer(
            self.device(),
            self.queue(),
            input,
            rms_weight,
            row_count,
            in_features,
            eps,
        )?;
        self.q2_matvec
            .run_argmax_with_input_buffer(
                self.device(),
                self.queue(),
                weights,
                &normed.buffer,
                normed.input_len,
                row_count,
                in_features,
                out_features,
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
