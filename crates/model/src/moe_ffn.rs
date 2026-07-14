use std::{
    collections::{BTreeSet, HashMap, VecDeque},
    fmt,
    sync::{Mutex, OnceLock},
    thread,
};

use backend::{Backend, BackendCapabilities, Q2ExpertSource};
use common::Tensor;
use common::{validate_exact_shape, DeviceKind, Error, F32Tensor, Result, Shape};
use config::Config;
use gguf::{
    matmul_q2_k_payload_f32, GgmlType, GgufFile, GgufQuantBlockKind, GGML_K_QUANT_BLOCK_SIZE,
};
use moe::ExpertDispatch;

use crate::{
    profile, FfnIndex, LayerIndex, MoeRouter, MoeRouterLoadReport, PackedExpertsIndex,
    QuantizedLinear, SharedExpertIndex, TensorRef,
};

const ROUTED_EXPERT_PREFETCH_CACHE_BYTES: u64 = 9 * 1024 * 1024 * 1024;
static ROUTED_EXPERT_PREFETCH_CACHE: OnceLock<Mutex<ExpertPayloadPrefetchCache>> = OnceLock::new();

#[derive(Debug)]
pub struct MoeFfn<'a> {
    layer_index: usize,
    router: MoeRouter<'a>,
    shared_gate: QuantizedLinear<'a>,
    shared_up: QuantizedLinear<'a>,
    shared_down: QuantizedLinear<'a>,
    routed_gate: PackedExpertLinear<'a>,
    routed_up: PackedExpertLinear<'a>,
    routed_gate_up: PackedExpertGateUp<'a>,
    routed_down: PackedExpertLinear<'a>,
    load_report: MoeFfnLoadReport,
}

#[derive(Debug)]
pub struct MoeFfnOutput {
    pub hidden_states: Tensor,
    pub report: MoeFfnForwardReport,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoeFfnLoadReport {
    pub backend: BackendCapabilities,
    pub layer_index: usize,
    pub router: MoeRouterLoadReport,
    pub shared_gate_weight_shape: Shape,
    pub shared_up_weight_shape: Shape,
    pub shared_down_weight_shape: Shape,
    pub packed_gate_weight_shape: Shape,
    pub packed_up_weight_shape: Shape,
    pub packed_down_weight_shape: Shape,
    pub shared_width: usize,
    pub routed_expert_count: usize,
    pub routed_intermediate_size: usize,
    pub projection_tensor_type: GgmlType,
    pub output_chunk_rows: usize,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoeFfnForwardReport {
    pub layer_index: usize,
    pub input_hidden_states_shape: Shape,
    pub normed_hidden_states_shape: Shape,
    pub flat_tokens_shape: Shape,
    pub shared_gate_shape: Shape,
    pub shared_gate_chunk_count: usize,
    pub shared_up_shape: Shape,
    pub shared_up_chunk_count: usize,
    pub shared_activated_gate_shape: Shape,
    pub shared_gated_shape: Shape,
    pub shared_down_shape: Shape,
    pub shared_down_chunk_count: usize,
    pub routed_expert_outputs_shape: Shape,
    pub routed_combined_shape: Shape,
    pub ffn_delta_shape: Shape,
    pub output_hidden_states_shape: Shape,
    pub routed_expert_count: usize,
    pub routed_assignment_count: usize,
    pub routed_source_payload_bytes_read: u64,
    pub routed_full_source_payload_bytes: u64,
    pub routed_peak_decoded_f32_bytes: u64,
}

fn require_moe_device_stage<T>(stage: &str, result: Result<Option<T>>) -> Result<T> {
    result?.ok_or_else(|| Error::backend(format!("{stage} has no native device path")))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ExpertPayloadPrefetchKey {
    file_id: usize,
    absolute_offset: u64,
    byte_len: u64,
}

#[derive(Debug)]
struct ExpertPayloadPrefetchCache {
    max_bytes: u64,
    current_bytes: u64,
    order: VecDeque<ExpertPayloadPrefetchKey>,
    entries: HashMap<ExpertPayloadPrefetchKey, u64>,
}

#[derive(Debug)]
struct RoutedExpertPrefetchWork<'a> {
    payloads: Vec<Q2ExpertPayload<'a>>,
}

#[derive(Debug)]
struct SelectedExpertSources<'a> {
    gate: Vec<Q2ExpertSource<'a>>,
    up: Vec<Q2ExpertSource<'a>>,
    down: Vec<Q2ExpertSource<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrefetchRange {
    offset: u64,
    byte_len: u64,
}

impl<'a> MoeFfn<'a> {
    pub fn open<B: Backend>(
        gguf: &'a GgufFile,
        config: &Config,
        layer: &LayerIndex,
        backend: &B,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        let FfnIndex::SparseMoe {
            shared_experts,
            packed_experts,
            ..
        } = &layer.ffn
        else {
            return Err(Error::gguf(format!(
                "GLM-5.2 GGUF layer {} is dense, not sparse MoE",
                layer.layer_index
            )));
        };
        Self::open_from_parts(
            gguf,
            config,
            layer,
            shared_experts,
            packed_experts,
            backend,
            output_chunk_rows,
        )
    }

    pub fn open_from_parts<B: Backend>(
        gguf: &'a GgufFile,
        config: &Config,
        layer: &LayerIndex,
        shared_experts: &SharedExpertIndex,
        packed_experts: &PackedExpertsIndex,
        backend: &B,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        validate_sparse_layer(config, layer.layer_index)?;
        let shared_width = config
            .moe_intermediate_size
            .checked_mul(config.num_shared_experts)
            .ok_or_else(|| Error::gguf("GGUF shared expert width overflow"))?;
        validate_shared_gate_or_up(
            config,
            shared_width,
            "gguf_moe_shared_gate",
            &shared_experts.gate,
        )?;
        validate_shared_gate_or_up(
            config,
            shared_width,
            "gguf_moe_shared_up",
            &shared_experts.up,
        )?;
        validate_shared_down(config, shared_width, &shared_experts.down)?;
        validate_packed_gate_or_up(config, "gguf_moe_packed_gate", &packed_experts.gate)?;
        validate_packed_gate_or_up(config, "gguf_moe_packed_up", &packed_experts.up)?;
        validate_packed_down(config, &packed_experts.down)?;

        let router = MoeRouter::open(gguf, config, layer, backend, output_chunk_rows)?;
        let shared_gate = QuantizedLinear::open(
            gguf,
            &shared_experts.gate,
            config.hidden_size,
            shared_width,
            output_chunk_rows,
        )?;
        let shared_up = QuantizedLinear::open(
            gguf,
            &shared_experts.up,
            config.hidden_size,
            shared_width,
            output_chunk_rows,
        )?;
        let shared_down = QuantizedLinear::open(
            gguf,
            &shared_experts.down,
            shared_width,
            config.hidden_size,
            output_chunk_rows,
        )?;
        let routed_gate = PackedExpertLinear::open(
            gguf,
            &packed_experts.gate,
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_routed_experts,
            output_chunk_rows,
        )?;
        let routed_up = PackedExpertLinear::open(
            gguf,
            &packed_experts.up,
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_routed_experts,
            output_chunk_rows,
        )?;
        let routed_gate_up = PackedExpertGateUp::open(&routed_gate, &routed_up)?;
        let routed_down = PackedExpertLinear::open(
            gguf,
            &packed_experts.down,
            config.moe_intermediate_size,
            config.hidden_size,
            config.num_routed_experts,
            output_chunk_rows,
        )?;

        let load_report = MoeFfnLoadReport {
            backend: backend.capabilities(),
            layer_index: layer.layer_index,
            router: router.load_report().clone(),
            shared_gate_weight_shape: logical_weight_shape_2d(&shared_experts.gate)?,
            shared_up_weight_shape: logical_weight_shape_2d(&shared_experts.up)?,
            shared_down_weight_shape: logical_weight_shape_2d(&shared_experts.down)?,
            packed_gate_weight_shape: logical_weight_shape_3d(&packed_experts.gate)?,
            packed_up_weight_shape: logical_weight_shape_3d(&packed_experts.up)?,
            packed_down_weight_shape: logical_weight_shape_3d(&packed_experts.down)?,
            shared_width,
            routed_expert_count: config.num_routed_experts,
            routed_intermediate_size: config.moe_intermediate_size,
            projection_tensor_type: packed_experts.gate.ty,
            output_chunk_rows,
            limitations: vec![
                "GGUF sparse FFN reads packed routed expert slices only for selected experts"
                    .to_string(),
                "native decode uses multi-expert Q2 dispatch; host-observed F32 prefill validation path still runs selected expert groups sequentially"
                    .to_string(),
            ],
        };

        Ok(Self {
            layer_index: layer.layer_index,
            router,
            shared_gate,
            shared_up,
            shared_down,
            routed_gate,
            routed_up,
            routed_gate_up,
            routed_down,
            load_report,
        })
    }

    pub fn load_report(&self) -> &MoeFfnLoadReport {
        &self.load_report
    }

    fn selected_routed_expert_prefetch_work(
        &self,
        dispatch_plan: &[ExpertDispatch],
    ) -> Result<RoutedExpertPrefetchWork<'a>> {
        if dispatch_plan.is_empty()
            || !self.routed_gate.is_q2()
            || !self.routed_up.is_q2()
            || !self.routed_down.is_q2()
        {
            return Ok(RoutedExpertPrefetchWork::default());
        }

        let mut expert_ids = BTreeSet::<usize>::new();
        for dispatch in dispatch_plan {
            expert_ids.insert(dispatch.expert_id);
        }

        let mut payloads = Vec::with_capacity(expert_ids.len() * 3);
        for expert_id in expert_ids {
            payloads.push(*self.routed_gate.q2_expert_payload(expert_id)?);
            payloads.push(*self.routed_up.q2_expert_payload(expert_id)?);
            payloads.push(*self.routed_down.q2_expert_payload(expert_id)?);
        }

        let misses = routed_expert_prefetch_misses(&payloads)?;
        Ok(RoutedExpertPrefetchWork { payloads: misses })
    }

    fn prefetch_selected_routed_experts(&self, dispatch_plan: &[ExpertDispatch]) -> Result<()> {
        let work = self.selected_routed_expert_prefetch_work(dispatch_plan)?;
        work.run(self.routed_gate.gguf, self.layer_index)
    }

    fn selected_routed_expert_sources(
        &self,
        expert_ids: &[u32],
    ) -> Result<SelectedExpertSources<'a>> {
        if expert_ids.is_empty() {
            return Err(Error::moe(
                "device Q2 expert execution requires selected expert IDs",
            ));
        }
        let mut gate_payloads = Vec::with_capacity(expert_ids.len());
        let mut up_payloads = Vec::with_capacity(expert_ids.len());
        let mut down_payloads = Vec::with_capacity(expert_ids.len());
        for &expert_id in expert_ids {
            let expert_id = usize::try_from(expert_id)
                .map_err(|_| Error::moe("device router expert id does not fit usize"))?;
            let gate = self.routed_gate.q2_expert_payload(expert_id)?;
            let up = self.routed_up.q2_expert_payload(expert_id)?;
            let down = self.routed_down.q2_expert_payload(expert_id)?;
            gate_payloads.push(Q2ExpertSource {
                bytes: gate.bytes,
                absolute_offset: gate.absolute_offset,
            });
            up_payloads.push(Q2ExpertSource {
                bytes: up.bytes,
                absolute_offset: up.absolute_offset,
            });
            down_payloads.push(Q2ExpertSource {
                bytes: down.bytes,
                absolute_offset: down.absolute_offset,
            });
        }
        Ok(SelectedExpertSources {
            gate: gate_payloads,
            up: up_payloads,
            down: down_payloads,
        })
    }

    fn run_with_async_selected_routed_expert_prefetch<T>(
        &self,
        dispatch_plan: &[ExpertDispatch],
        work: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let prefetch_work = self.selected_routed_expert_prefetch_work(dispatch_plan)?;
        if prefetch_work.is_empty() {
            return work();
        }

        let gguf = self.routed_gate.gguf;
        let layer_index = self.layer_index;
        thread::scope(|scope| {
            let prefetch = scope.spawn(move || prefetch_work.run(gguf, layer_index));
            let work_result = work();
            let prefetch_result = match prefetch.join() {
                Ok(result) => result,
                Err(_) => Err(Error::moe("routed expert prefetch thread panicked")),
            };

            match (work_result, prefetch_result) {
                (Ok(value), Ok(())) => Ok(value),
                (Err(error), Ok(())) => Err(error),
                (Ok(_), Err(error)) => Err(error),
                (Err(error), Err(prefetch_error)) => {
                    tracing::debug!(
                        layer_index,
                        error = %prefetch_error,
                        "routed expert async prefetch failed after foreground work already failed"
                    );
                    Err(error)
                }
            }
        })
    }

    pub fn forward<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<MoeFfnOutput> {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::moe(format!(
                "GLM-5.2 GGUF MoE FFN input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            "gguf_moe_ffn_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;
        let flat_token_count = batch
            .checked_mul(tokens)
            .ok_or_else(|| Error::moe("GGUF MoE FFN flat token count overflow"))?;

        let routing = self.router.route(config, hidden_states, backend)?;
        let flat_tokens = routing
            .normed_hidden_states
            .contiguous()?
            .reshape((flat_token_count, config.hidden_size))?;

        let (shared_gate, shared_up, shared_activated_gate_shape, shared_gated, shared_down) = self
            .run_with_async_selected_routed_expert_prefetch(&routing.dispatch_plan, || {
                let shared_gate = self.shared_gate.forward(&flat_tokens, backend)?;
                let shared_up = self.shared_up.forward(&flat_tokens, backend)?;
                let shared_activated_gate_shape = Shape::new(shared_gate.output.dims().to_vec());
                let shared_gated = backend.swiglu(&shared_gate.output, &shared_up.output)?;
                let shared_down = self.shared_down.forward(&shared_gated, backend)?;
                Ok((
                    shared_gate,
                    shared_up,
                    shared_activated_gate_shape,
                    shared_gated,
                    shared_down,
                ))
            })?;
        validate_exact_shape(
            "gguf_moe_ffn_shared_output",
            shared_down.output.dims(),
            &[flat_token_count, config.hidden_size],
        )?;

        let mut token_indices = Vec::<u32>::new();
        let mut expert_weights = Vec::<f32>::new();
        let mut expert_output_batches = Vec::<Tensor>::new();
        let routed_expert_count = routing.dispatch_plan.len();
        let routed_assignment_count = routing
            .dispatch_plan
            .iter()
            .map(|dispatch| dispatch.assignments.len())
            .sum::<usize>();
        let mut routed_source_payload_bytes_read = 0_u64;
        let mut routed_full_source_payload_bytes = 0_u64;
        let mut routed_peak_decoded_f32_bytes = 0_u64;

        self.prefetch_selected_routed_experts(&routing.dispatch_plan)?;

        for dispatch in &routing.dispatch_plan {
            let dispatch_token_indices = dispatch
                .assignments
                .iter()
                .map(|assignment| {
                    u32::try_from(assignment.token_index)
                        .map_err(|_| Error::moe("GGUF MoE token index does not fit u32"))
                })
                .collect::<Result<Vec<_>>>()?;
            token_indices.extend(dispatch_token_indices.iter().copied());
            expert_weights.extend(
                dispatch
                    .assignments
                    .iter()
                    .map(|assignment| assignment.weight),
            );

            let expert_input = backend.moe_gather_tokens(&flat_tokens, &dispatch_token_indices)?;
            let gate =
                self.routed_gate
                    .forward_expert(dispatch.expert_id, &expert_input, backend)?;
            let up = self
                .routed_up
                .forward_expert(dispatch.expert_id, &expert_input, backend)?;
            let gated = backend.swiglu(&gate.output, &up.output)?;
            let down = self
                .routed_down
                .forward_expert(dispatch.expert_id, &gated, backend)?;
            expert_output_batches.push(down.output);

            for report in [&gate.report, &up.report, &down.report] {
                routed_source_payload_bytes_read = routed_source_payload_bytes_read
                    .checked_add(report.source_payload_bytes_read)
                    .ok_or_else(|| Error::gguf("GGUF MoE routed source byte count overflow"))?;
                routed_full_source_payload_bytes = routed_full_source_payload_bytes
                    .checked_add(report.full_source_payload_bytes)
                    .ok_or_else(|| Error::gguf("GGUF MoE routed full byte count overflow"))?;
                routed_peak_decoded_f32_bytes =
                    routed_peak_decoded_f32_bytes.max(report.peak_decoded_f32_bytes);
            }
        }

        if expert_output_batches.is_empty() {
            return Err(Error::moe(
                "GLM-5.2 GGUF MoE FFN routing produced no expert assignments",
            ));
        }

        let expert_output_refs = expert_output_batches.iter().collect::<Vec<_>>();
        let routed_expert_outputs = Tensor::cat(&expert_output_refs, 0)?;
        let token_indices = Tensor::from_vec(
            token_indices,
            routed_expert_outputs.dims()[0],
            backend.device(),
        )?;
        let expert_weights = Tensor::from_vec(
            expert_weights,
            routed_expert_outputs.dims()[0],
            backend.device(),
        )?;
        let routed_accumulator = Tensor::zeros((flat_token_count, config.hidden_size))?;
        let routed_combined = backend.moe_weighted_index_add_combine(
            &routed_accumulator,
            &token_indices,
            &routed_expert_outputs,
            &expert_weights,
        )?;
        let ffn_delta_flat = backend.add(&routed_combined, &shared_down.output)?;
        let ffn_delta = ffn_delta_flat.reshape((batch, tokens, config.hidden_size))?;
        let output_hidden_states = backend.add(hidden_states, &ffn_delta)?;
        let output_hidden_states_shape = Shape::new(output_hidden_states.dims().to_vec());

        Ok(MoeFfnOutput {
            hidden_states: output_hidden_states,
            report: MoeFfnForwardReport {
                layer_index: self.layer_index,
                input_hidden_states_shape: Shape::new(hidden_states.dims().to_vec()),
                normed_hidden_states_shape: Shape::new(
                    routing.normed_hidden_states.dims().to_vec(),
                ),
                flat_tokens_shape: Shape::new(flat_tokens.dims().to_vec()),
                shared_gate_shape: Shape::new(shared_gate.output.dims().to_vec()),
                shared_gate_chunk_count: shared_gate.report.chunk_count,
                shared_up_shape: Shape::new(shared_up.output.dims().to_vec()),
                shared_up_chunk_count: shared_up.report.chunk_count,
                shared_activated_gate_shape,
                shared_gated_shape: Shape::new(shared_gated.dims().to_vec()),
                shared_down_shape: Shape::new(shared_down.output.dims().to_vec()),
                shared_down_chunk_count: shared_down.report.chunk_count,
                routed_expert_outputs_shape: Shape::new(routed_expert_outputs.dims().to_vec()),
                routed_combined_shape: Shape::new(routed_combined.dims().to_vec()),
                ffn_delta_shape: Shape::new(ffn_delta.dims().to_vec()),
                output_hidden_states_shape,
                routed_expert_count,
                routed_assignment_count,
                routed_source_payload_bytes_read,
                routed_full_source_payload_bytes,
                routed_peak_decoded_f32_bytes,
            },
        })
    }

    pub fn forward_tensors<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &Tensor,
        backend: &B,
    ) -> Result<Tensor> {
        if backend.capabilities().custom_kernels {
            let hidden_states = tensor_to_f32_tensor(hidden_states)?;
            let output = self.forward_f32_tensor(config, &hidden_states, backend)?;
            return tensor_from_f32_tensor(output, backend.device());
        }

        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::moe(format!(
                "GLM-5.2 GGUF MoE FFN input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            "gguf_moe_ffn_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;
        let flat_token_count = batch
            .checked_mul(tokens)
            .ok_or_else(|| Error::moe("GGUF MoE FFN flat token count overflow"))?;

        let routing = profile::run_layer_stage(self.layer_index, "sparse_moe.router", || {
            self.router.route_tensors(config, hidden_states, backend)
        })?;
        let flat_tokens = routing
            .normed_hidden_states
            .contiguous()?
            .reshape((flat_token_count, config.hidden_size))?;

        let shared_down =
            self.run_with_async_selected_routed_expert_prefetch(&routing.dispatch_plan, || {
                let shared_gate =
                    profile::run_layer_stage(self.layer_index, "sparse_moe.shared_gate", || {
                        self.shared_gate.forward_tensor(&flat_tokens, backend)
                    })?;
                let shared_up =
                    profile::run_layer_stage(self.layer_index, "sparse_moe.shared_up", || {
                        self.shared_up.forward_tensor(&flat_tokens, backend)
                    })?;
                let shared_gated = backend.swiglu(&shared_gate, &shared_up)?;
                profile::run_layer_stage(self.layer_index, "sparse_moe.shared_down", || {
                    self.shared_down.forward_tensor(&shared_gated, backend)
                })
            })?;
        validate_exact_shape(
            "gguf_moe_ffn_shared_output",
            shared_down.dims(),
            &[flat_token_count, config.hidden_size],
        )?;

        let routed_combined = if flat_token_count == 1 {
            self.forward_single_token_routed_tensors(&routing.dispatch_plan, &flat_tokens, backend)?
        } else {
            self.prefetch_selected_routed_experts(&routing.dispatch_plan)?;
            let mut token_indices = Vec::<u32>::new();
            let mut expert_weights = Vec::<f32>::new();
            let mut expert_output_batches = Vec::<Tensor>::new();

            for dispatch in &routing.dispatch_plan {
                let dispatch_token_indices = dispatch
                    .assignments
                    .iter()
                    .map(|assignment| {
                        u32::try_from(assignment.token_index)
                            .map_err(|_| Error::moe("GGUF MoE token index does not fit u32"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                token_indices.extend(dispatch_token_indices.iter().copied());
                expert_weights.extend(
                    dispatch
                        .assignments
                        .iter()
                        .map(|assignment| assignment.weight),
                );

                let expert_input =
                    backend.moe_gather_tokens(&flat_tokens, &dispatch_token_indices)?;
                let gated = profile::run_layer_stage(
                    self.layer_index,
                    "sparse_moe.routed_gate_up_swiglu",
                    || {
                        self.routed_gate_up.forward_gated_expert_tensor(
                            dispatch.expert_id,
                            &expert_input,
                            backend,
                        )
                    },
                )?;
                let down =
                    profile::run_layer_stage(self.layer_index, "sparse_moe.routed_down", || {
                        self.routed_down
                            .forward_expert_tensor(dispatch.expert_id, &gated, backend)
                    })?;

                expert_output_batches.push(down);
            }

            if expert_output_batches.is_empty() {
                return Err(Error::moe(
                    "GLM-5.2 GGUF MoE FFN routing produced no expert assignments",
                ));
            }

            profile::run_layer_stage(self.layer_index, "sparse_moe.combine", || {
                let expert_output_refs = expert_output_batches.iter().collect::<Vec<_>>();
                let routed_expert_outputs = Tensor::cat(&expert_output_refs, 0)?;
                let token_indices = Tensor::from_vec(
                    token_indices,
                    routed_expert_outputs.dims()[0],
                    backend.device(),
                )?;
                let expert_weights = Tensor::from_vec(
                    expert_weights,
                    routed_expert_outputs.dims()[0],
                    backend.device(),
                )?;
                let routed_accumulator = Tensor::zeros((flat_token_count, config.hidden_size))?;
                backend.moe_weighted_index_add_combine(
                    &routed_accumulator,
                    &token_indices,
                    &routed_expert_outputs,
                    &expert_weights,
                )
            })?
        };
        let ffn_delta_flat = backend.add(&routed_combined, &shared_down)?;
        let ffn_delta = ffn_delta_flat.reshape((batch, tokens, config.hidden_size))?;
        backend.add(hidden_states, &ffn_delta)
    }

    pub fn forward_f32_tensor<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &F32Tensor,
        backend: &B,
    ) -> Result<F32Tensor> {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::moe(format!(
                "GLM-5.2 GGUF native MoE FFN input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            "gguf_native_moe_ffn_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;
        let flat_token_count = batch
            .checked_mul(tokens)
            .ok_or_else(|| Error::moe("GGUF native MoE FFN flat token count overflow"))?;

        let routing = profile::run_layer_stage(self.layer_index, "sparse_moe.router", || {
            self.router
                .route_f32_tensors(config, hidden_states, backend)
        })?;
        let flat_tokens = routing
            .normed_hidden_states
            .clone()
            .reshape([flat_token_count, config.hidden_size])?;

        let shared_down =
            self.run_with_async_selected_routed_expert_prefetch(&routing.dispatch_plan, || {
                let shared_gate =
                    profile::run_layer_stage(self.layer_index, "sparse_moe.shared_gate", || {
                        self.shared_gate.forward_f32_tensor(&flat_tokens, backend)
                    })?;
                let shared_up =
                    profile::run_layer_stage(self.layer_index, "sparse_moe.shared_up", || {
                        self.shared_up.forward_f32_tensor(&flat_tokens, backend)
                    })?;
                let shared_gated = require_native(
                    "SwiGLU",
                    backend.swiglu_f32_tensor(&shared_gate, &shared_up)?,
                )?;
                profile::run_layer_stage(self.layer_index, "sparse_moe.shared_down", || {
                    self.shared_down.forward_f32_tensor(&shared_gated, backend)
                })
            })?;
        validate_exact_shape(
            "gguf_native_moe_ffn_shared_output",
            shared_down.dims(),
            &[flat_token_count, config.hidden_size],
        )?;

        let routed_combined = self.forward_routed_f32_tensor(
            config,
            &routing.dispatch_plan,
            &flat_tokens,
            flat_token_count,
            backend,
        )?;
        let ffn_delta_flat = require_native(
            "add",
            backend.add_f32_tensor(&routed_combined, &shared_down)?,
        )?;
        let ffn_delta = ffn_delta_flat.reshape([batch, tokens, config.hidden_size])?;
        require_native("add", backend.add_f32_tensor(hidden_states, &ffn_delta)?)
    }

    /// Batched device-resident MoE FFN for one-token decode or three-token MTP
    /// verification. Router and expert math stay on Metal. After top-k, one
    /// deliberate synchronization reads the selected IDs so their exact Q2
    /// ranges can be streamed from SSD before the expert kernels run.
    pub(crate) fn forward_device<B: Backend>(
        &self,
        config: &Config,
        hidden_states: &backend::DeviceValue,
        backend: &B,
    ) -> Result<Option<backend::DeviceValue>> {
        let dims = hidden_states.dims();
        if dims.len() != 3 {
            return Err(Error::moe(format!(
                "GLM-5.2 GGUF device MoE FFN input must be rank 3 [B,T,H], got {dims:?}"
            )));
        }
        let batch = dims[0];
        let tokens = dims[1];
        validate_exact_shape(
            "gguf_device_moe_ffn_hidden_states",
            dims,
            &[batch, tokens, config.hidden_size],
        )?;
        let flat_token_count = batch
            .checked_mul(tokens)
            .ok_or_else(|| Error::moe("GGUF device MoE FFN flat token count overflow"))?;
        if flat_token_count == 0 || flat_token_count > 3 {
            return Ok(None);
        }

        let (routing, selected_expert_ids) = crate::try_device!(profile::run_token_device_stage(
            profile::TokenProfileStage::MoeRouting,
            backend,
            || {
                let routing = crate::try_device!(profile::run_layer_stage(
                    self.layer_index,
                    "sparse_moe.router",
                    || self.router.route_device(config, hidden_states, backend),
                ));
                let selected_expert_ids = crate::try_device!(profile::run_layer_stage(
                    self.layer_index,
                    "sparse_moe.router_sync",
                    || backend.moe_router_expert_ids_device(&routing.topk),
                ));
                Ok(Some((routing, selected_expert_ids)))
            },
        ));
        let flat_tokens = routing
            .normed_hidden_states
            .reshape(vec![flat_token_count, config.hidden_size])?;
        let sources = self.selected_routed_expert_sources(&selected_expert_ids)?;
        let (routed, shared_down) = thread::scope(|scope| {
            let routed = scope.spawn(|| {
                profile::run_layer_stage(self.layer_index, "sparse_moe.routed.ready_first", || {
                    backend.ready_routed_experts_device(
                        self.layer_index,
                        self.routed_gate.gguf.path(),
                        &sources.gate,
                        &sources.up,
                        &sources.down,
                        &flat_tokens,
                        &routing.topk,
                        self.routed_gate_up.in_features,
                        self.routed_gate_up.out_features,
                        self.routed_down.out_features,
                    )
                })
            });
            let shared_result: Result<backend::DeviceValue> = (|| {
                let shared_gate = require_moe_device_stage(
                    "sparse_moe.shared_gate",
                    profile::run_layer_stage(
                        self.layer_index,
                        "sparse_moe.shared_gate.device",
                        || self.shared_gate.forward_device(&flat_tokens, backend),
                    ),
                )?;
                let shared_up = require_moe_device_stage(
                    "sparse_moe.shared_up",
                    profile::run_layer_stage(
                        self.layer_index,
                        "sparse_moe.shared_up.device",
                        || self.shared_up.forward_device(&flat_tokens, backend),
                    ),
                )?;
                let shared_gated = require_moe_device_stage(
                    "sparse_moe.shared_swiglu",
                    profile::run_layer_stage(
                        self.layer_index,
                        "sparse_moe.shared_swiglu.device",
                        || backend.swiglu_device(&shared_gate, &shared_up),
                    ),
                )?;
                let shared_down = require_moe_device_stage(
                    "sparse_moe.shared_down",
                    profile::run_layer_stage(
                        self.layer_index,
                        "sparse_moe.shared_down.device",
                        || self.shared_down.forward_device(&shared_gated, backend),
                    ),
                )?;
                // Start the shared-expert command buffer immediately. Routed
                // SSD reads and expert kernels continue independently; the
                // weighted combine adds the precise cross-queue dependency.
                backend.device_submit()?;
                Ok(shared_down)
            })();
            let routed_result = routed
                .join()
                .map_err(|_| Error::moe("ready routed expert thread panicked"))?;
            let routed =
                match require_moe_device_stage("sparse_moe.routed.ready_first", routed_result) {
                    Ok(routed) => routed,
                    Err(error) => {
                        backend.device_flush()?;
                        return Err(error);
                    }
                };
            let shared_down = match shared_result {
                Ok(shared_down) => shared_down,
                Err(error) => {
                    backend.device_flush()?;
                    return Err(error);
                }
            };
            Ok::<_, Error>((routed, shared_down))
        })?;
        tracing::debug!(
            layer_index = self.layer_index,
            selected_experts = routed.selected_experts(),
            cache_hits = routed.cache_hits(),
            cache_misses = routed.cache_misses(),
            transient_experts = routed.transient_experts(),
            read_bytes = routed.read_bytes(),
            ready_waves = routed.ready_waves(),
            expert_ids = ?selected_expert_ids,
            "completed ready-first routed expert execution"
        );
        validate_exact_shape(
            "gguf_device_moe_ffn_shared_output",
            shared_down.dims(),
            &[flat_token_count, config.hidden_size],
        )?;

        backend.wait_for_routed_experts_device(&routed)?;
        let output = require_moe_device_stage(
            "sparse_moe.weighted_combine_residual",
            backend.moe_topk_combine_residual_device(
                &shared_down,
                hidden_states,
                routed.output(),
                &routing.topk,
            ),
        )?;
        Ok(Some(output))
    }

    fn forward_single_token_routed_tensors<B: Backend>(
        &self,
        dispatch_plan: &[ExpertDispatch],
        flat_token: &Tensor,
        backend: &B,
    ) -> Result<Tensor> {
        validate_exact_shape(
            "gguf_moe_single_token_flat_input",
            flat_token.dims(),
            &[1, self.routed_gate.in_features],
        )?;
        if dispatch_plan.is_empty() {
            return Err(Error::moe(
                "GLM-5.2 GGUF MoE FFN routing produced no expert assignments",
            ));
        }

        let mut token_indices = Vec::<u32>::with_capacity(dispatch_plan.len());
        let mut expert_weights = Vec::<f32>::with_capacity(dispatch_plan.len());
        let mut expert_outputs = Vec::<Tensor>::with_capacity(dispatch_plan.len());

        self.prefetch_selected_routed_experts(dispatch_plan)?;

        for dispatch in dispatch_plan {
            validate_exact_shape(
                "gguf_moe_single_token_assignment_count",
                &[dispatch.assignments.len()],
                &[1],
            )?;
            let assignment = dispatch.assignments.first().ok_or_else(|| {
                Error::moe("GLM-5.2 GGUF MoE single-token dispatch has no assignment")
            })?;
            validate_exact_shape(
                "gguf_moe_single_token_assignment_index",
                &[assignment.token_index],
                &[0],
            )?;
            token_indices.push(0);
            expert_weights.push(assignment.weight);

            let gated = profile::run_layer_stage(
                self.layer_index,
                "sparse_moe.routed_gate_up_swiglu",
                || {
                    self.routed_gate_up.forward_gated_expert_tensor(
                        dispatch.expert_id,
                        flat_token,
                        backend,
                    )
                },
            )?;
            let down =
                profile::run_layer_stage(self.layer_index, "sparse_moe.routed_down", || {
                    self.routed_down
                        .forward_expert_tensor(dispatch.expert_id, &gated, backend)
                })?;
            expert_outputs.push(down);
        }

        if expert_outputs.is_empty() {
            return Err(Error::moe(
                "GLM-5.2 GGUF MoE FFN single-token combine produced no output",
            ));
        }

        profile::run_layer_stage(self.layer_index, "sparse_moe.combine", || {
            let expert_output_refs = expert_outputs.iter().collect::<Vec<_>>();
            let routed_expert_outputs = Tensor::cat(&expert_output_refs, 0)?;
            let token_indices = Tensor::from_vec(
                token_indices,
                routed_expert_outputs.dims()[0],
                backend.device(),
            )?;
            let expert_weights = Tensor::from_vec(
                expert_weights,
                routed_expert_outputs.dims()[0],
                backend.device(),
            )?;
            let routed_accumulator = Tensor::zeros((1, self.routed_down.out_features))?;
            backend.moe_weighted_index_add_combine(
                &routed_accumulator,
                &token_indices,
                &routed_expert_outputs,
                &expert_weights,
            )
        })
    }

    fn forward_routed_f32_tensor<B: Backend>(
        &self,
        config: &Config,
        dispatch_plan: &[ExpertDispatch],
        flat_tokens: &F32Tensor,
        flat_token_count: usize,
        backend: &B,
    ) -> Result<F32Tensor> {
        if dispatch_plan.is_empty() {
            return Err(Error::moe(
                "GLM-5.2 GGUF native MoE FFN routing produced no expert assignments",
            ));
        }

        let mut expert_output_batches = Vec::<F32Tensor>::new();

        self.prefetch_selected_routed_experts(dispatch_plan)?;

        for dispatch in dispatch_plan {
            let dispatch_token_indices = dispatch
                .assignments
                .iter()
                .map(|assignment| {
                    u32::try_from(assignment.token_index)
                        .map_err(|_| Error::moe("GGUF native MoE token index does not fit u32"))
                })
                .collect::<Result<Vec<_>>>()?;

            let expert_input = if flat_token_count == 1 {
                flat_tokens.clone()
            } else {
                require_native(
                    "moe_gather_tokens",
                    backend.moe_gather_tokens_f32_tensor(flat_tokens, &dispatch_token_indices)?,
                )?
            };
            let gated = profile::run_layer_stage(
                self.layer_index,
                "sparse_moe.routed_gate_up_swiglu",
                || {
                    self.routed_gate_up.forward_gated_expert_f32_tensor(
                        dispatch.expert_id,
                        &expert_input,
                        backend,
                    )
                },
            )?;
            let down =
                profile::run_layer_stage(self.layer_index, "sparse_moe.routed_down", || {
                    self.routed_down
                        .forward_expert_f32_tensor(dispatch.expert_id, &gated, backend)
                })?;

            expert_output_batches.push(down);
        }

        let (token_indices, expert_weights, routed_expert_outputs) =
            token_major_routed_outputs_f32(
                dispatch_plan,
                &expert_output_batches,
                flat_token_count,
                config.experts_per_token,
                config.hidden_size,
            )?;
        let routed_accumulator = F32Tensor::zeros([flat_token_count, config.hidden_size])?;
        require_native(
            "moe_weighted_index_add_combine",
            backend.moe_weighted_index_add_combine_f32_tensor(
                &routed_accumulator,
                &token_indices,
                &routed_expert_outputs,
                &expert_weights,
            )?,
        )
    }
}

fn routed_expert_prefetch_cache() -> &'static Mutex<ExpertPayloadPrefetchCache> {
    ROUTED_EXPERT_PREFETCH_CACHE.get_or_init(|| {
        Mutex::new(ExpertPayloadPrefetchCache::new(
            ROUTED_EXPERT_PREFETCH_CACHE_BYTES,
        ))
    })
}

fn routed_expert_prefetch_misses<'a>(
    payloads: &[Q2ExpertPayload<'a>],
) -> Result<Vec<Q2ExpertPayload<'a>>> {
    let cache = routed_expert_prefetch_cache();
    let mut cache = cache
        .lock()
        .map_err(|_| Error::moe("routed expert prefetch cache lock poisoned"))?;
    Ok(payloads
        .iter()
        .copied()
        .filter(|payload| !cache.is_warm(payload.key()))
        .collect())
}

fn mark_routed_expert_payloads_prefetched(payloads: &[Q2ExpertPayload<'_>]) -> Result<()> {
    let cache = routed_expert_prefetch_cache();
    let mut cache = cache
        .lock()
        .map_err(|_| Error::moe("routed expert prefetch cache lock poisoned"))?;
    for payload in payloads {
        cache.insert(payload.key());
    }
    Ok(())
}

impl RoutedExpertPrefetchWork<'_> {
    fn is_empty(&self) -> bool {
        self.payloads.is_empty()
    }

    fn run(self, gguf: &GgufFile, layer_index: usize) -> Result<()> {
        if self.payloads.is_empty() {
            return Ok(());
        }

        let ranges = coalesced_payload_prefetch_ranges(&self.payloads)?;
        let prefetch_ranges = ranges
            .iter()
            .map(|range| (range.offset, range.byte_len))
            .collect::<Vec<_>>();
        gguf.prefetch_ranges(&prefetch_ranges)?;
        let prefetched_bytes = ranges.iter().try_fold(0_u64, |total, range| {
            total
                .checked_add(range.byte_len)
                .ok_or_else(|| Error::gguf("routed expert prefetch byte count overflow"))
        })?;
        mark_routed_expert_payloads_prefetched(&self.payloads)?;

        tracing::debug!(
            layer_index,
            payload_count = self.payloads.len(),
            range_count = ranges.len(),
            prefetched_bytes,
            async_prefetch = true,
            "prefetched cold routed expert payloads"
        );

        Ok(())
    }
}

fn coalesced_payload_prefetch_ranges(
    payloads: &[Q2ExpertPayload<'_>],
) -> Result<Vec<PrefetchRange>> {
    let mut ranges = payloads
        .iter()
        .map(|payload| PrefetchRange {
            offset: payload.absolute_offset,
            byte_len: payload.byte_len,
        })
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| range.offset);

    let mut coalesced = Vec::<PrefetchRange>::new();
    for range in ranges {
        if range.byte_len == 0 {
            continue;
        }
        let range_end = range
            .offset
            .checked_add(range.byte_len)
            .ok_or_else(|| Error::gguf("routed expert prefetch range overflow"))?;
        if let Some(last) = coalesced.last_mut() {
            let last_end = last
                .offset
                .checked_add(last.byte_len)
                .ok_or_else(|| Error::gguf("routed expert coalesced prefetch range overflow"))?;
            if range.offset <= last_end {
                if range_end > last_end {
                    last.byte_len = range_end
                        .checked_sub(last.offset)
                        .ok_or_else(|| Error::gguf("routed expert prefetch range underflow"))?;
                }
                continue;
            }
        }
        coalesced.push(range);
    }
    Ok(coalesced)
}

impl Default for RoutedExpertPrefetchWork<'_> {
    fn default() -> Self {
        Self {
            payloads: Vec::new(),
        }
    }
}

impl ExpertPayloadPrefetchCache {
    fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            current_bytes: 0,
            order: VecDeque::new(),
            entries: HashMap::new(),
        }
    }

    fn is_warm(&mut self, key: ExpertPayloadPrefetchKey) -> bool {
        if self.entries.contains_key(&key) {
            self.touch(key);
            return true;
        }
        false
    }

    fn insert(&mut self, key: ExpertPayloadPrefetchKey) {
        if self.entries.contains_key(&key) {
            self.touch(key);
            return;
        }
        if key.byte_len > self.max_bytes {
            return;
        }

        while self.current_bytes.saturating_add(key.byte_len) > self.max_bytes {
            let Some(evicted) = self.order.pop_front() else {
                break;
            };
            if let Some(bytes) = self.entries.remove(&evicted) {
                self.current_bytes = self.current_bytes.saturating_sub(bytes);
            }
        }

        self.order.push_back(key);
        self.entries.insert(key, key.byte_len);
        self.current_bytes = self.current_bytes.saturating_add(key.byte_len);
    }

    fn touch(&mut self, key: ExpertPayloadPrefetchKey) {
        if let Some(index) = self.order.iter().position(|candidate| *candidate == key) {
            self.order.remove(index);
        }
        self.order.push_back(key);
    }
}

struct PackedExpertLinear<'a> {
    gguf: &'a GgufFile,
    tensor_ref: TensorRef,
    in_features: usize,
    out_features: usize,
    expert_count: usize,
    blocks_per_row: u64,
    output_chunk_rows: usize,
    expert_source_payload_bytes: u64,
    full_source_payload_bytes: u64,
    q2_expert_payloads: Option<Vec<Q2ExpertPayload<'a>>>,
}

impl fmt::Debug for PackedExpertLinear<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PackedExpertLinear")
            .field("tensor_ref", &self.tensor_ref)
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("expert_count", &self.expert_count)
            .field("blocks_per_row", &self.blocks_per_row)
            .field("output_chunk_rows", &self.output_chunk_rows)
            .field(
                "expert_source_payload_bytes",
                &self.expert_source_payload_bytes,
            )
            .field("full_source_payload_bytes", &self.full_source_payload_bytes)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct PackedExpertLinearOutput {
    output: Tensor,
    report: PackedExpertLinearForwardReport,
}

#[derive(Debug, Clone, PartialEq)]
struct PackedExpertLinearForwardReport {
    expert_id: usize,
    source_tensor_name: String,
    tensor_type: GgmlType,
    input_shape: Shape,
    logical_weight_shape: Shape,
    output_shape: Shape,
    output_chunk_rows: usize,
    chunk_count: usize,
    source_payload_bytes_read: u64,
    full_source_payload_bytes: u64,
    peak_decoded_f32_bytes: u64,
}

impl<'a> PackedExpertLinear<'a> {
    fn open(
        gguf: &'a GgufFile,
        tensor_ref: &TensorRef,
        expected_in_features: usize,
        expected_out_features: usize,
        expected_expert_count: usize,
        output_chunk_rows: usize,
    ) -> Result<Self> {
        if output_chunk_rows == 0 {
            return Err(Error::gguf(
                "GGUF packed expert linear output_chunk_rows must be positive",
            ));
        }
        validate_quantized_type(tensor_ref)?;
        let (in_features, out_features, expert_count) = packed_linear_dims(tensor_ref)?;
        validate_exact_shape(
            "gguf_packed_expert_linear_shape",
            &[in_features, out_features, expert_count],
            &[
                expected_in_features,
                expected_out_features,
                expected_expert_count,
            ],
        )?;

        let info = gguf.tensor(&tensor_ref.name).ok_or_else(|| {
            Error::gguf(format!("missing GLM-5.2 GGUF tensor {}", tensor_ref.name))
        })?;
        validate_tensor_ref(tensor_ref, info)?;

        let block_size = usize::try_from(GGML_K_QUANT_BLOCK_SIZE)
            .map_err(|_| Error::gguf("K-quant block size does not fit usize"))?;
        if in_features % block_size != 0 {
            return Err(Error::gguf(format!(
                "GGUF packed expert linear input dimension {in_features} must be divisible by {block_size}"
            )));
        }
        let blocks_per_row = u64::try_from(in_features / block_size).map_err(|_| {
            Error::gguf("GGUF packed expert linear blocks_per_row does not fit u64")
        })?;
        let storage = gguf.tensor_quantized_storage(&tensor_ref.name)?;
        let expected_blocks =
            blocks_per_row
                .checked_mul(u64::try_from(out_features).map_err(|_| {
                    Error::gguf("GGUF packed expert output row count does not fit u64")
                })?)
                .and_then(|blocks| blocks.checked_mul(u64::try_from(expert_count).ok()?))
                .ok_or_else(|| Error::gguf("GGUF packed expert block count overflow"))?;
        if storage.block_count != expected_blocks {
            return Err(Error::gguf(format!(
                "GGUF packed expert tensor {} has {} quant blocks but shape requires {expected_blocks}",
                tensor_ref.name, storage.block_count
            )));
        }
        let expert_source_payload_bytes =
            blocks_per_row
                .checked_mul(u64::try_from(out_features).map_err(|_| {
                    Error::gguf("GGUF packed expert output row count does not fit u64")
                })?)
                .and_then(|blocks| blocks.checked_mul(storage.block.block_byte_len()))
                .ok_or_else(|| Error::gguf("GGUF packed expert payload byte count overflow"))?;
        let q2_expert_payloads = if storage.block == GgufQuantBlockKind::Q2K {
            Some(build_q2_expert_payloads(
                storage.bytes,
                gguf.cache_identity(),
                storage.info.absolute_offset,
                blocks_per_row,
                out_features,
                expert_count,
            )?)
        } else {
            None
        };
        Ok(Self {
            gguf,
            tensor_ref: tensor_ref.clone(),
            in_features,
            out_features,
            expert_count,
            blocks_per_row,
            output_chunk_rows,
            expert_source_payload_bytes,
            full_source_payload_bytes: storage.payload_byte_len,
            q2_expert_payloads,
        })
    }

    fn forward_expert<B: Backend>(
        &self,
        expert_id: usize,
        input: &Tensor,
        backend: &B,
    ) -> Result<PackedExpertLinearOutput> {
        let run = self.run_expert_linear(expert_id, input, backend, true)?;
        Ok(PackedExpertLinearOutput {
            output: run.output.clone(),
            report: PackedExpertLinearForwardReport {
                expert_id,
                source_tensor_name: self.tensor_ref.name.clone(),
                tensor_type: self.tensor_ref.ty,
                input_shape: Shape::new(input.dims().to_vec()),
                logical_weight_shape: Shape::new(vec![self.out_features, self.in_features]),
                output_shape: Shape::new(run.output.dims().to_vec()),
                output_chunk_rows: self.output_chunk_rows,
                chunk_count: run.chunk_count,
                source_payload_bytes_read: self.expert_source_payload_bytes,
                full_source_payload_bytes: self.full_source_payload_bytes,
                peak_decoded_f32_bytes: run.peak_decoded_f32_bytes,
            },
        })
    }

    fn is_q2(&self) -> bool {
        self.tensor_ref.ty == GgmlType::Q2K && self.q2_expert_payloads.is_some()
    }

    fn forward_expert_tensor<B: Backend>(
        &self,
        expert_id: usize,
        input: &Tensor,
        backend: &B,
    ) -> Result<Tensor> {
        Ok(self
            .run_expert_linear(expert_id, input, backend, false)?
            .output)
    }

    fn forward_expert_f32_tensor<B: Backend>(
        &self,
        expert_id: usize,
        input: &F32Tensor,
        backend: &B,
    ) -> Result<F32Tensor> {
        self.run_q2_expert_matmul_f32(expert_id, input, backend)
    }

    fn run_expert_linear<B: Backend>(
        &self,
        expert_id: usize,
        input: &Tensor,
        backend: &B,
        collect_report_stats: bool,
    ) -> Result<PackedExpertLinearRun> {
        if expert_id >= self.expert_count {
            return Err(Error::moe(format!(
                "GGUF packed expert id {expert_id} exceeds expert_count {}",
                self.expert_count
            )));
        }
        validate_exact_shape(
            "gguf_packed_expert_linear_input_rank",
            &[input.dims().len()],
            &[2],
        )?;
        validate_exact_shape(
            "gguf_packed_expert_linear_input_features",
            &[input.dims()[1]],
            &[self.in_features],
        )?;

        if !collect_report_stats && self.tensor_ref.ty == GgmlType::Q2K {
            return self.run_q2_expert_matmul(expert_id, input, backend);
        }

        let storage = self.gguf.tensor_quantized_storage(&self.tensor_ref.name)?;
        let expert_row_offset = u64::try_from(expert_id)
            .ok()
            .and_then(|id| id.checked_mul(u64::try_from(self.out_features).ok()?))
            .ok_or_else(|| Error::gguf("GGUF packed expert row offset overflow"))?;
        let mut outputs = Vec::new();
        let mut rows_loaded = 0_usize;
        let mut peak_decoded_f32_bytes = 0_u64;

        while rows_loaded < self.out_features {
            let rows_this_chunk = self
                .output_chunk_rows
                .min(self.out_features.saturating_sub(rows_loaded));
            let block_start =
                expert_row_offset
                    .checked_add(u64::try_from(rows_loaded).map_err(|_| {
                        Error::gguf("GGUF packed expert rows_loaded does not fit u64")
                    })?)
                    .and_then(|row| row.checked_mul(self.blocks_per_row))
                    .ok_or_else(|| Error::gguf("GGUF packed expert chunk offset overflow"))?;
            let block_count = u64::try_from(rows_this_chunk)
                .ok()
                .and_then(|rows| rows.checked_mul(self.blocks_per_row))
                .ok_or_else(|| Error::gguf("GGUF packed expert chunk block count overflow"))?;
            let chunk_values = storage.dequantize_block_range_as_f32(block_start, block_count)?;
            if collect_report_stats {
                let decoded_bytes = u64::try_from(chunk_values.len())
                    .ok()
                    .and_then(|values| values.checked_mul(4))
                    .ok_or_else(|| Error::gguf("GGUF packed expert decoded byte overflow"))?;
                peak_decoded_f32_bytes = peak_decoded_f32_bytes.max(decoded_bytes);
            }
            let chunk_weight = Tensor::from_vec(
                chunk_values,
                (rows_this_chunk, self.in_features),
                backend.device(),
            )?;
            outputs.push(backend.linear(input, &chunk_weight)?);
            rows_loaded += rows_this_chunk;
        }

        let chunk_count = outputs.len();
        let output_refs = outputs.iter().collect::<Vec<_>>();
        let output = Tensor::cat(&output_refs, 1)?;
        Ok(PackedExpertLinearRun {
            output,
            chunk_count,
            peak_decoded_f32_bytes,
        })
    }

    fn run_q2_expert_matmul<B: Backend>(
        &self,
        expert_id: usize,
        input: &Tensor,
        backend: &B,
    ) -> Result<PackedExpertLinearRun> {
        let batch = input.dims()[0];
        let payload = self.q2_expert_payload(expert_id)?;
        if let Some(output) = backend.q2_k_matvec_f32(
            payload.bytes,
            input,
            batch,
            self.in_features,
            self.out_features,
        )? {
            validate_exact_shape(
                "gguf_packed_expert_native_q2_output",
                output.dims(),
                &[batch, self.out_features],
            )?;
            return Ok(PackedExpertLinearRun {
                output,
                chunk_count: 1,
                peak_decoded_f32_bytes: 0,
            });
        }
        reject_missing_native_q2_kernel_on_metal(backend, &self.tensor_ref.name)?;

        let output = q2_payload_matvec_tensor(
            &format!("{} expert {expert_id}", self.tensor_ref.name),
            payload.bytes,
            input,
            batch,
            self.in_features,
            self.out_features,
            backend.device(),
        )?;
        validate_exact_shape(
            "gguf_packed_expert_q2_output",
            output.dims(),
            &[batch, self.out_features],
        )?;

        Ok(PackedExpertLinearRun {
            output,
            chunk_count: 1,
            peak_decoded_f32_bytes: 0,
        })
    }

    fn run_q2_expert_matmul_f32<B: Backend>(
        &self,
        expert_id: usize,
        input: &F32Tensor,
        backend: &B,
    ) -> Result<F32Tensor> {
        if expert_id >= self.expert_count {
            return Err(Error::moe(format!(
                "GGUF native packed expert id {expert_id} exceeds expert_count {}",
                self.expert_count
            )));
        }
        let dims = input.dims();
        validate_exact_shape("gguf_native_packed_expert_input_rank", &[dims.len()], &[2])?;
        validate_exact_shape(
            "gguf_native_packed_expert_input_features",
            &[dims[1]],
            &[self.in_features],
        )?;

        let batch = dims[0];
        let payload = self.q2_expert_payload(expert_id)?;
        if let Some(output) = backend.q2_k_matvec_f32_tensor(
            payload.bytes,
            input,
            batch,
            self.in_features,
            self.out_features,
        )? {
            validate_exact_shape(
                "gguf_native_packed_expert_q2_output",
                output.dims(),
                &[batch, self.out_features],
            )?;
            return Ok(output);
        }
        reject_missing_native_q2_kernel_on_metal(backend, &self.tensor_ref.name)?;

        let output_values = matmul_q2_k_payload_f32(
            &format!("{} expert {expert_id}", self.tensor_ref.name),
            payload.bytes,
            input.values(),
            batch,
            self.in_features,
            self.out_features,
        )?;
        let output = F32Tensor::new(output_values, [batch, self.out_features])?;
        validate_exact_shape(
            "gguf_native_packed_expert_q2_output",
            output.dims(),
            &[batch, self.out_features],
        )?;
        Ok(output)
    }

    fn q2_expert_payload(&self, expert_id: usize) -> Result<&Q2ExpertPayload<'a>> {
        let payloads = self.q2_expert_payloads.as_ref().ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} must be Q2_K for the packed expert Q2 matmul path, got {}",
                self.tensor_ref.name, self.tensor_ref.ty
            ))
        })?;
        payloads.get(expert_id).ok_or_else(|| {
            Error::moe(format!(
                "GGUF packed expert id {expert_id} exceeds expert_count {}",
                self.expert_count
            ))
        })
    }
}

struct PackedExpertGateUp<'a> {
    gate_tensor_name: String,
    up_tensor_name: String,
    in_features: usize,
    out_features: usize,
    expert_count: usize,
    gate_payloads: Vec<Q2ExpertPayload<'a>>,
    up_payloads: Vec<Q2ExpertPayload<'a>>,
}

impl fmt::Debug for PackedExpertGateUp<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PackedExpertGateUp")
            .field("gate_tensor_name", &self.gate_tensor_name)
            .field("up_tensor_name", &self.up_tensor_name)
            .field("in_features", &self.in_features)
            .field("out_features", &self.out_features)
            .field("expert_count", &self.expert_count)
            .finish_non_exhaustive()
    }
}

impl<'a> PackedExpertGateUp<'a> {
    fn open(gate: &PackedExpertLinear<'a>, up: &PackedExpertLinear<'a>) -> Result<Self> {
        validate_exact_shape(
            "gguf_packed_expert_gate_up_features",
            &[gate.in_features, gate.out_features, gate.expert_count],
            &[up.in_features, up.out_features, up.expert_count],
        )?;
        let gate_payloads = gate.q2_expert_payloads.as_ref().ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} must be Q2_K for fused routed gate/up",
                gate.tensor_ref.name
            ))
        })?;
        let up_payloads = up.q2_expert_payloads.as_ref().ok_or_else(|| {
            Error::gguf(format!(
                "GGUF tensor {} must be Q2_K for fused routed gate/up",
                up.tensor_ref.name
            ))
        })?;
        validate_exact_shape(
            "gguf_packed_expert_gate_up_payload_count",
            &[gate_payloads.len(), up_payloads.len()],
            &[gate.expert_count, gate.expert_count],
        )?;
        Ok(Self {
            gate_tensor_name: gate.tensor_ref.name.clone(),
            up_tensor_name: up.tensor_ref.name.clone(),
            in_features: gate.in_features,
            out_features: gate.out_features,
            expert_count: gate.expert_count,
            gate_payloads: gate_payloads.clone(),
            up_payloads: up_payloads.clone(),
        })
    }

    fn forward_expert_tensors<B: Backend>(
        &self,
        expert_id: usize,
        input: &Tensor,
        backend: &B,
    ) -> Result<(Tensor, Tensor)> {
        validate_exact_shape(
            "gguf_packed_expert_gate_up_input_rank",
            &[input.dims().len()],
            &[2],
        )?;
        validate_exact_shape(
            "gguf_packed_expert_gate_up_input_features",
            &[input.dims()[1]],
            &[self.in_features],
        )?;

        let batch = input.dims()[0];
        let gate = self.gate_payload(expert_id)?;
        let up = self.up_payload(expert_id)?;
        if let Some(gate_output) = backend.q2_k_matvec_f32(
            gate.bytes,
            input,
            batch,
            self.in_features,
            self.out_features,
        )? {
            let up_output = backend
                .q2_k_matvec_f32(up.bytes, input, batch, self.in_features, self.out_features)?
                .ok_or_else(|| {
                    Error::backend("native Q2_K gate matvec was available but up matvec was not")
                })?;
            validate_exact_shape(
                "gguf_packed_expert_native_gate_output",
                gate_output.dims(),
                &[batch, self.out_features],
            )?;
            validate_exact_shape(
                "gguf_packed_expert_native_up_output",
                up_output.dims(),
                &[batch, self.out_features],
            )?;
            return Ok((gate_output, up_output));
        }
        reject_missing_native_q2_kernel_on_metal(backend, &self.gate_tensor_name)?;

        let gate = q2_payload_matvec_tensor(
            &format!("{} expert {expert_id}", self.gate_tensor_name),
            gate.bytes,
            input,
            batch,
            self.in_features,
            self.out_features,
            backend.device(),
        )?;
        let up = q2_payload_matvec_tensor(
            &format!("{} expert {expert_id}", self.up_tensor_name),
            up.bytes,
            input,
            batch,
            self.in_features,
            self.out_features,
            backend.device(),
        )?;
        Ok((gate, up))
    }

    fn forward_gated_expert_tensor<B: Backend>(
        &self,
        expert_id: usize,
        input: &Tensor,
        backend: &B,
    ) -> Result<Tensor> {
        validate_exact_shape(
            "gguf_packed_expert_gate_up_swiglu_input_rank",
            &[input.dims().len()],
            &[2],
        )?;
        validate_exact_shape(
            "gguf_packed_expert_gate_up_swiglu_input_features",
            &[input.dims()[1]],
            &[self.in_features],
        )?;

        let batch = input.dims()[0];
        let gate = self.gate_payload(expert_id)?;
        let up = self.up_payload(expert_id)?;

        if let Some(gated) = backend.q2_k_gate_up_swiglu_f32(
            gate.bytes,
            up.bytes,
            input,
            batch,
            self.in_features,
            self.out_features,
        )? {
            validate_exact_shape(
                "gguf_packed_expert_native_gate_up_swiglu_output",
                gated.dims(),
                &[batch, self.out_features],
            )?;
            return Ok(gated);
        }

        let (gate, up) = self.forward_expert_tensors(expert_id, input, backend)?;
        backend.swiglu(&gate, &up)
    }

    fn forward_gated_expert_f32_tensor<B: Backend>(
        &self,
        expert_id: usize,
        input: &F32Tensor,
        backend: &B,
    ) -> Result<F32Tensor> {
        let dims = input.dims();
        validate_exact_shape(
            "gguf_native_packed_expert_gate_up_swiglu_input_rank",
            &[dims.len()],
            &[2],
        )?;
        validate_exact_shape(
            "gguf_native_packed_expert_gate_up_swiglu_input_features",
            &[dims[1]],
            &[self.in_features],
        )?;

        let batch = dims[0];
        let gate = self.gate_payload(expert_id)?;
        let up = self.up_payload(expert_id)?;

        if let Some(gated) = backend.q2_k_gate_up_swiglu_f32_tensor(
            gate.bytes,
            up.bytes,
            input,
            batch,
            self.in_features,
            self.out_features,
        )? {
            validate_exact_shape(
                "gguf_native_packed_expert_gate_up_swiglu_output",
                gated.dims(),
                &[batch, self.out_features],
            )?;
            return Ok(gated);
        }
        reject_missing_native_q2_kernel_on_metal(backend, &self.gate_tensor_name)?;

        let gate_values = matmul_q2_k_payload_f32(
            &format!("{} expert {expert_id}", self.gate_tensor_name),
            gate.bytes,
            input.values(),
            batch,
            self.in_features,
            self.out_features,
        )?;
        let up_values = matmul_q2_k_payload_f32(
            &format!("{} expert {expert_id}", self.up_tensor_name),
            up.bytes,
            input.values(),
            batch,
            self.in_features,
            self.out_features,
        )?;
        let gate = F32Tensor::new(gate_values, [batch, self.out_features])?;
        let up = F32Tensor::new(up_values, [batch, self.out_features])?;
        require_native("SwiGLU", backend.swiglu_f32_tensor(&gate, &up)?)
    }

    fn gate_payload(&self, expert_id: usize) -> Result<&Q2ExpertPayload<'a>> {
        self.gate_payloads.get(expert_id).ok_or_else(|| {
            Error::moe(format!(
                "GGUF packed gate expert id {expert_id} exceeds expert_count {}",
                self.expert_count
            ))
        })
    }

    fn up_payload(&self, expert_id: usize) -> Result<&Q2ExpertPayload<'a>> {
        self.up_payloads.get(expert_id).ok_or_else(|| {
            Error::moe(format!(
                "GGUF packed up expert id {expert_id} exceeds expert_count {}",
                self.expert_count
            ))
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct Q2ExpertPayload<'a> {
    bytes: &'a [u8],
    file_id: usize,
    absolute_offset: u64,
    byte_len: u64,
}

impl Q2ExpertPayload<'_> {
    fn key(&self) -> ExpertPayloadPrefetchKey {
        ExpertPayloadPrefetchKey {
            file_id: self.file_id,
            absolute_offset: self.absolute_offset,
            byte_len: self.byte_len,
        }
    }
}

#[derive(Debug)]
struct PackedExpertLinearRun {
    output: Tensor,
    chunk_count: usize,
    peak_decoded_f32_bytes: u64,
}

fn build_q2_expert_payloads<'a>(
    storage_bytes: &'a [u8],
    file_id: usize,
    storage_absolute_offset: u64,
    blocks_per_row: u64,
    out_features: usize,
    expert_count: usize,
) -> Result<Vec<Q2ExpertPayload<'a>>> {
    let expert_blocks = blocks_per_row
        .checked_mul(
            u64::try_from(out_features)
                .map_err(|_| Error::gguf("GGUF packed expert output row count does not fit u64"))?,
        )
        .ok_or_else(|| Error::gguf("GGUF packed expert block count overflow"))?;
    let block_byte_len = GgufQuantBlockKind::Q2K.block_byte_len();
    let byte_len = expert_blocks
        .checked_mul(block_byte_len)
        .ok_or_else(|| Error::gguf("GGUF packed expert byte length overflow"))?;
    let len = usize::try_from(byte_len)
        .map_err(|_| Error::gguf("GGUF packed expert byte length does not fit usize"))?;

    let mut payloads = Vec::with_capacity(expert_count);
    for expert_id in 0..expert_count {
        let relative_byte_start = u64::try_from(expert_id)
            .ok()
            .and_then(|id| id.checked_mul(byte_len))
            .ok_or_else(|| Error::gguf("GGUF packed expert byte offset overflow"))?;
        let start = usize::try_from(relative_byte_start)
            .map_err(|_| Error::gguf("GGUF packed expert byte start does not fit usize"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| Error::gguf("GGUF packed expert byte range overflow"))?;
        let bytes = storage_bytes.get(start..end).ok_or_else(|| {
            Error::gguf(format!(
                "GGUF packed expert byte range {start}..{end} is outside Q2 payload"
            ))
        })?;
        let absolute_offset = storage_absolute_offset
            .checked_add(relative_byte_start)
            .ok_or_else(|| Error::gguf("GGUF packed expert absolute byte offset overflow"))?;
        payloads.push(Q2ExpertPayload {
            bytes,
            file_id,
            absolute_offset,
            byte_len,
        });
    }
    Ok(payloads)
}

fn reject_missing_native_q2_kernel_on_metal<B: Backend>(
    backend: &B,
    tensor_name: &str,
) -> Result<()> {
    if backend.capabilities().device == DeviceKind::Metal {
        return Err(Error::backend(format!(
            "native Metal Q2_K matvec is required for {tensor_name}; CPU reference fallback is disabled on Metal"
        )));
    }
    Ok(())
}

fn tensor_to_f32_tensor(tensor: &Tensor) -> Result<F32Tensor> {
    let dims = tensor.dims().to_vec();
    let values = tensor
        .to_dtype(common::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    F32Tensor::new(values, dims)
}

fn tensor_from_f32_tensor(tensor: F32Tensor, device: &common::Device) -> Result<Tensor> {
    let (shape, values) = tensor.into_parts();
    Ok(Tensor::from_vec(values, shape.dims(), device)?)
}

fn require_native<T>(operation: &str, value: Option<T>) -> Result<T> {
    value.ok_or_else(|| {
        Error::backend(format!(
            "native Metal {operation} is required for the GLM-5.2 Q2 MoE path"
        ))
    })
}

fn token_major_routed_outputs_f32(
    dispatch_plan: &[ExpertDispatch],
    expert_outputs_by_dispatch: &[F32Tensor],
    token_count: usize,
    top_k: usize,
    hidden_size: usize,
) -> Result<(Vec<u32>, Vec<f32>, F32Tensor)> {
    if dispatch_plan.is_empty() {
        return Err(Error::moe(
            "cannot pack empty routed expert outputs in token-major order",
        ));
    }
    if dispatch_plan.len() != expert_outputs_by_dispatch.len() {
        return Err(Error::moe(format!(
            "routed expert output batch count mismatch: {} dispatches but {} output batches",
            dispatch_plan.len(),
            expert_outputs_by_dispatch.len()
        )));
    }
    if token_count == 0 || top_k == 0 || hidden_size == 0 {
        return Err(Error::moe(
            "token-major routed output packing requires non-zero token_count, top_k and hidden_size",
        ));
    }

    let assignment_count = token_count
        .checked_mul(top_k)
        .ok_or_else(|| Error::moe("token-major routed assignment count overflow"))?;
    let value_count = assignment_count
        .checked_mul(hidden_size)
        .ok_or_else(|| Error::moe("token-major routed output value count overflow"))?;
    let mut token_indices = Vec::with_capacity(assignment_count);
    for token in 0..token_count {
        let token_index = u32::try_from(token)
            .map_err(|_| Error::moe("token-major routed token index exceeds u32"))?;
        for _ in 0..top_k {
            token_indices.push(token_index);
        }
    }
    let mut expert_weights = vec![0.0_f32; assignment_count];
    let mut values = vec![0.0_f32; value_count];
    let mut seen = vec![false; assignment_count];

    for (dispatch, expert_output) in dispatch_plan.iter().zip(expert_outputs_by_dispatch) {
        let dims = expert_output.dims();
        validate_exact_shape("token_major_routed_output_rank", &[dims.len()], &[2])?;
        validate_exact_shape(
            "token_major_routed_output_rows",
            &[dims[0]],
            &[dispatch.assignments.len()],
        )?;
        validate_exact_shape(
            "token_major_routed_output_hidden_size",
            &[dims[1]],
            &[hidden_size],
        )?;

        for (source_row, assignment) in dispatch.assignments.iter().enumerate() {
            if assignment.token_index >= token_count {
                return Err(Error::moe(format!(
                    "routed assignment token {} exceeds token_count {token_count}",
                    assignment.token_index
                )));
            }
            if assignment.topk_rank >= top_k {
                return Err(Error::moe(format!(
                    "routed assignment top-k rank {} exceeds top_k {top_k}",
                    assignment.topk_rank
                )));
            }
            let target_row = assignment
                .token_index
                .checked_mul(top_k)
                .and_then(|base| base.checked_add(assignment.topk_rank))
                .ok_or_else(|| Error::moe("token-major routed row offset overflow"))?;
            if seen[target_row] {
                return Err(Error::moe(format!(
                    "duplicate routed assignment for token {} rank {}",
                    assignment.token_index, assignment.topk_rank
                )));
            }
            seen[target_row] = true;
            expert_weights[target_row] = assignment.weight;

            let source_start = source_row
                .checked_mul(hidden_size)
                .ok_or_else(|| Error::moe("token-major routed source offset overflow"))?;
            let target_start = target_row
                .checked_mul(hidden_size)
                .ok_or_else(|| Error::moe("token-major routed target offset overflow"))?;
            values[target_start..target_start + hidden_size]
                .copy_from_slice(&expert_output.values()[source_start..source_start + hidden_size]);
        }
    }

    if let Some(missing_row) = seen.iter().position(|is_seen| !is_seen) {
        return Err(Error::moe(format!(
            "missing routed assignment for token {} rank {}",
            missing_row / top_k,
            missing_row % top_k
        )));
    }

    let outputs = F32Tensor::new(values, [assignment_count, hidden_size])?;
    Ok((token_indices, expert_weights, outputs))
}

fn q2_payload_matvec_tensor(
    context: &str,
    payload: &[u8],
    input: &Tensor,
    row_count: usize,
    in_features: usize,
    out_features: usize,
    device: &common::Device,
) -> Result<Tensor> {
    let input_values = flatten_rank2_input(input)?;
    let output_values = matmul_q2_k_payload_f32(
        context,
        payload,
        &input_values,
        row_count,
        in_features,
        out_features,
    )?;
    Ok(Tensor::from_vec(
        output_values,
        (row_count, out_features),
        device,
    )?)
}

fn flatten_rank2_input(input: &Tensor) -> Result<Vec<f32>> {
    match input.dims() {
        [_, _] => Ok(input.to_vec2::<f32>()?.into_iter().flatten().collect()),
        dims => Err(Error::model(format!(
            "GGUF packed expert input rank must be 2, got {dims:?}"
        ))),
    }
}

fn validate_sparse_layer(config: &Config, layer_index: usize) -> Result<()> {
    let is_single_mtp_layer =
        layer_index == config.num_layers && config.num_nextn_predict_layers == 1;
    if layer_index >= config.num_layers && !is_single_mtp_layer {
        return Err(Error::moe(format!(
            "GLM-5.2 GGUF MoE FFN layer {layer_index} exceeds num_layers {}",
            config.num_layers
        )));
    }
    if layer_index < config.dense_layers {
        return Err(Error::moe(format!(
            "GLM-5.2 GGUF MoE FFN layer {layer_index} is dense; first sparse layer is {}",
            config.dense_layers
        )));
    }
    Ok(())
}

fn validate_shared_gate_or_up(
    config: &Config,
    shared_width: usize,
    context: &str,
    tensor_ref: &TensorRef,
) -> Result<()> {
    let (input_features, output_features) = linear_dims_2d(tensor_ref)?;
    validate_exact_shape(
        context,
        &[input_features, output_features],
        &[config.hidden_size, shared_width],
    )
}

fn validate_shared_down(
    config: &Config,
    shared_width: usize,
    tensor_ref: &TensorRef,
) -> Result<()> {
    let (input_features, output_features) = linear_dims_2d(tensor_ref)?;
    validate_exact_shape(
        "gguf_moe_shared_down",
        &[input_features, output_features],
        &[shared_width, config.hidden_size],
    )
}

fn validate_packed_gate_or_up(
    config: &Config,
    context: &str,
    tensor_ref: &TensorRef,
) -> Result<()> {
    let (input_features, output_features, expert_count) = packed_linear_dims(tensor_ref)?;
    validate_exact_shape(
        context,
        &[input_features, output_features, expert_count],
        &[
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_routed_experts,
        ],
    )
}

fn validate_packed_down(config: &Config, tensor_ref: &TensorRef) -> Result<()> {
    let (input_features, output_features, expert_count) = packed_linear_dims(tensor_ref)?;
    validate_exact_shape(
        "gguf_moe_packed_down",
        &[input_features, output_features, expert_count],
        &[
            config.moe_intermediate_size,
            config.hidden_size,
            config.num_routed_experts,
        ],
    )
}

fn validate_quantized_type(tensor_ref: &TensorRef) -> Result<()> {
    match tensor_ref.ty {
        GgmlType::Q2K | GgmlType::Q8_0 => Ok(()),
        other => Err(Error::gguf(format!(
            "GLM-5.2 Q2 GGUF MoE tensor {} must be Q2_K or Q8_0, got {other}",
            tensor_ref.name
        ))),
    }
}

fn validate_tensor_ref(tensor_ref: &TensorRef, info: &gguf::GgufTensorInfo) -> Result<()> {
    if tensor_ref.dims != info.dims
        || tensor_ref.ty != info.ty
        || tensor_ref.absolute_offset != info.absolute_offset
        || tensor_ref.storage_byte_len != info.storage_byte_len
    {
        return Err(Error::gguf(format!(
            "GGUF tensor {} metadata changed after indexing",
            tensor_ref.name
        )));
    }
    Ok(())
}

fn linear_dims_2d(tensor_ref: &TensorRef) -> Result<(usize, usize)> {
    validate_quantized_type(tensor_ref)?;
    validate_exact_shape("gguf_moe_linear_rank", &[tensor_ref.dims.len()], &[2])?;
    Ok((
        usize::try_from(tensor_ref.dims[0]).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} input dimension does not fit usize",
                tensor_ref.name
            ))
        })?,
        usize::try_from(tensor_ref.dims[1]).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} output dimension does not fit usize",
                tensor_ref.name
            ))
        })?,
    ))
}

fn packed_linear_dims(tensor_ref: &TensorRef) -> Result<(usize, usize, usize)> {
    validate_quantized_type(tensor_ref)?;
    validate_exact_shape(
        "gguf_packed_expert_linear_rank",
        &[tensor_ref.dims.len()],
        &[3],
    )?;
    Ok((
        usize::try_from(tensor_ref.dims[0]).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} input dimension does not fit usize",
                tensor_ref.name
            ))
        })?,
        usize::try_from(tensor_ref.dims[1]).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} output dimension does not fit usize",
                tensor_ref.name
            ))
        })?,
        usize::try_from(tensor_ref.dims[2]).map_err(|_| {
            Error::gguf(format!(
                "GGUF tensor {} expert dimension does not fit usize",
                tensor_ref.name
            ))
        })?,
    ))
}

fn logical_weight_shape_2d(tensor_ref: &TensorRef) -> Result<Shape> {
    let (input_features, output_features) = linear_dims_2d(tensor_ref)?;
    Ok(Shape::new(vec![output_features, input_features]))
}

fn logical_weight_shape_3d(tensor_ref: &TensorRef) -> Result<Shape> {
    let (input_features, output_features, expert_count) = packed_linear_dims(tensor_ref)?;
    Ok(Shape::new(vec![
        expert_count,
        output_features,
        input_features,
    ]))
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
    use gguf::{
        GgmlType, GgufFile, GgufMetadataValueType, GGML_Q2_K_BLOCK_BYTES, GGUF_MAGIC,
        GGUF_VERSION_V3,
    };

    use crate::{AttentionIndex, LayerKind};

    use super::*;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn token_major_routed_outputs_reorders_expert_grouped_batches() {
        let dispatch_plan = vec![
            ExpertDispatch {
                expert_id: 2,
                assignments: vec![
                    moe::ExpertAssignment {
                        token_index: 0,
                        topk_rank: 1,
                        weight: 0.25,
                    },
                    moe::ExpertAssignment {
                        token_index: 1,
                        topk_rank: 0,
                        weight: 0.60,
                    },
                ],
            },
            ExpertDispatch {
                expert_id: 3,
                assignments: vec![
                    moe::ExpertAssignment {
                        token_index: 0,
                        topk_rank: 0,
                        weight: 0.75,
                    },
                    moe::ExpertAssignment {
                        token_index: 1,
                        topk_rank: 1,
                        weight: 0.40,
                    },
                ],
            },
        ];
        let expert_outputs = vec![
            F32Tensor::new(
                vec![
                    20.0_f32, 21.0, //
                    30.0, 31.0,
                ],
                [2, 2],
            )
            .unwrap(),
            F32Tensor::new(
                vec![
                    10.0_f32, 11.0, //
                    40.0, 41.0,
                ],
                [2, 2],
            )
            .unwrap(),
        ];

        let (token_indices, expert_weights, outputs) =
            token_major_routed_outputs_f32(&dispatch_plan, &expert_outputs, 2, 2, 2).unwrap();

        assert_eq!(token_indices, vec![0, 0, 1, 1]);
        assert_eq!(expert_weights, vec![0.75, 0.25, 0.60, 0.40]);
        assert_eq!(
            outputs.values(),
            &[
                10.0_f32, 11.0, //
                20.0, 21.0, //
                30.0, 31.0, //
                40.0, 41.0,
            ]
        );
    }

    #[test]
    fn token_major_routed_outputs_rejects_duplicate_token_rank() {
        let dispatch_plan = vec![ExpertDispatch {
            expert_id: 1,
            assignments: vec![
                moe::ExpertAssignment {
                    token_index: 0,
                    topk_rank: 0,
                    weight: 0.5,
                },
                moe::ExpertAssignment {
                    token_index: 0,
                    topk_rank: 0,
                    weight: 0.5,
                },
            ],
        }];
        let expert_outputs = vec![F32Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], [2, 2]).unwrap()];

        let err = token_major_routed_outputs_f32(&dispatch_plan, &expert_outputs, 1, 2, 2)
            .expect_err("duplicate token/rank should be rejected");

        assert!(err.to_string().contains("duplicate routed assignment"));
    }

    #[test]
    fn q2_sparse_ffn_executes_shared_and_selected_packed_experts() {
        let path = write_sparse_ffn_fixture(GgmlType::Q2K, false);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let layer = layer_index(&gguf);
        let ffn = MoeFfn::open(&gguf, &config, &layer, &backend, 256).unwrap();
        let hidden_states = Tensor::zeros((1, 1, 256)).unwrap();

        let output = ffn.forward(&config, &hidden_states, &backend).unwrap();

        assert_eq!(ffn.load_report().projection_tensor_type, GgmlType::Q2K);
        assert_eq!(output.hidden_states.dims(), &[1, 1, 256]);
        assert_eq!(output.report.routed_expert_outputs_shape.dims(), &[2, 256]);
        assert_eq!(output.report.routed_expert_count, 2);
        assert_eq!(output.report.routed_assignment_count, 2);
    }

    #[test]
    fn q2_sparse_ffn_tensor_path_matches_reference() {
        let path = write_sparse_ffn_fixture(GgmlType::Q2K, false);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let layer = layer_index(&gguf);
        let ffn = MoeFfn::open(&gguf, &config, &layer, &backend, 256).unwrap();
        let hidden_states = Tensor::zeros((1, 1, 256)).unwrap();

        let reference = ffn
            .forward(&config, &hidden_states, &backend)
            .unwrap()
            .hidden_states;
        let optimized = ffn
            .forward_tensors(&config, &hidden_states, &backend)
            .unwrap();

        assert_eq!(optimized.dims(), &[1, 1, 256]);
        let reference = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let optimized = optimized.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (reference_value, optimized_value) in reference.iter().zip(optimized.iter()) {
            assert!((reference_value - optimized_value).abs() <= 1e-4);
        }
    }

    #[test]
    fn rejects_packed_expert_layout_with_expert_first_dimension() {
        let path = write_sparse_ffn_fixture(GgmlType::Q2K, true);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let backend = MetalBackend::from_device(Device::Cpu).unwrap();
        let layer = layer_index(&gguf);

        let err = MoeFfn::open(&gguf, &config, &layer, &backend, 128)
            .expect_err("expert-first packed layout should fail");

        assert!(err.to_string().contains("gguf_moe_packed_gate"));
    }

    #[test]
    fn q2_expert_payloads_record_absolute_offsets() {
        let bytes = vec![0_u8; (GGML_Q2_K_BLOCK_BYTES * 4) as usize];

        let payloads = build_q2_expert_payloads(&bytes, 7, 1024, 1, 2, 2).unwrap();

        assert_eq!(payloads.len(), 2);
        assert_eq!(
            payloads[0].bytes.len(),
            (GGML_Q2_K_BLOCK_BYTES * 2) as usize
        );
        assert_eq!(payloads[0].file_id, 7);
        assert_eq!(payloads[0].absolute_offset, 1024);
        assert_eq!(payloads[0].byte_len, GGML_Q2_K_BLOCK_BYTES * 2);
        assert_eq!(
            payloads[1].absolute_offset,
            1024 + (GGML_Q2_K_BLOCK_BYTES * 2)
        );
    }

    #[test]
    fn routed_expert_prefetch_cache_uses_lru_capacity() {
        let mut cache = ExpertPayloadPrefetchCache::new(10);
        let first = ExpertPayloadPrefetchKey {
            file_id: 1,
            absolute_offset: 0,
            byte_len: 4,
        };
        let second = ExpertPayloadPrefetchKey {
            file_id: 1,
            absolute_offset: 4,
            byte_len: 4,
        };
        let third = ExpertPayloadPrefetchKey {
            file_id: 1,
            absolute_offset: 8,
            byte_len: 4,
        };

        cache.insert(first);
        cache.insert(second);
        assert!(cache.is_warm(first));

        cache.insert(third);

        assert!(cache.is_warm(first));
        assert!(!cache.is_warm(second));
        assert!(cache.is_warm(third));
    }

    #[test]
    fn routed_expert_prefetch_cache_is_scoped_by_mapped_file() {
        let mut cache = ExpertPayloadPrefetchCache::new(16);
        let first_file = ExpertPayloadPrefetchKey {
            file_id: 1,
            absolute_offset: 4096,
            byte_len: 8,
        };
        let second_file_same_range = ExpertPayloadPrefetchKey {
            file_id: 2,
            absolute_offset: 4096,
            byte_len: 8,
        };

        cache.insert(first_file);

        assert!(cache.is_warm(first_file));
        assert!(!cache.is_warm(second_file_same_range));
    }

    #[test]
    fn routed_expert_prefetch_ranges_are_sorted_and_coalesced() {
        let bytes = [0_u8; 16];
        let payloads = vec![
            Q2ExpertPayload {
                bytes: &bytes[8..12],
                file_id: 1,
                absolute_offset: 108,
                byte_len: 4,
            },
            Q2ExpertPayload {
                bytes: &bytes[0..4],
                file_id: 1,
                absolute_offset: 100,
                byte_len: 4,
            },
            Q2ExpertPayload {
                bytes: &bytes[4..8],
                file_id: 1,
                absolute_offset: 104,
                byte_len: 4,
            },
            Q2ExpertPayload {
                bytes: &bytes[12..16],
                file_id: 1,
                absolute_offset: 200,
                byte_len: 4,
            },
        ];

        let ranges = coalesced_payload_prefetch_ranges(&payloads).unwrap();

        assert_eq!(
            ranges,
            vec![
                PrefetchRange {
                    offset: 100,
                    byte_len: 12,
                },
                PrefetchRange {
                    offset: 200,
                    byte_len: 4,
                },
            ]
        );
    }

    fn layer_index(gguf: &GgufFile) -> LayerIndex {
        LayerIndex {
            layer_index: 1,
            kind: LayerKind::SparseMoe,
            input_norm: tensor_ref(gguf, "blk.1.ffn_norm.weight"),
            post_attention_norm: tensor_ref(gguf, "blk.1.ffn_norm.weight"),
            attention: AttentionIndex {
                q_a: tensor_ref(gguf, "blk.1.ffn_gate_inp.weight"),
                q_a_norm: tensor_ref(gguf, "blk.1.ffn_norm.weight"),
                q_b: tensor_ref(gguf, "blk.1.ffn_gate_inp.weight"),
                kv_a_mqa: tensor_ref(gguf, "blk.1.ffn_gate_inp.weight"),
                kv_a_norm: tensor_ref(gguf, "blk.1.ffn_norm.weight"),
                k_b: tensor_ref(gguf, "blk.1.ffn_gate_inp.weight"),
                v_b: tensor_ref(gguf, "blk.1.ffn_gate_inp.weight"),
                output: tensor_ref(gguf, "blk.1.ffn_gate_inp.weight"),
                indexer: None,
            },
            ffn: FfnIndex::SparseMoe {
                router: tensor_ref(gguf, "blk.1.ffn_gate_inp.weight"),
                router_correction_bias: tensor_ref(gguf, "blk.1.exp_probs_b.bias"),
                shared_experts: SharedExpertIndex {
                    gate: tensor_ref(gguf, "blk.1.ffn_gate_shexp.weight"),
                    up: tensor_ref(gguf, "blk.1.ffn_up_shexp.weight"),
                    down: tensor_ref(gguf, "blk.1.ffn_down_shexp.weight"),
                },
                packed_experts: PackedExpertsIndex {
                    gate: tensor_ref(gguf, "blk.1.ffn_gate_exps.weight"),
                    up: tensor_ref(gguf, "blk.1.ffn_up_exps.weight"),
                    down: tensor_ref(gguf, "blk.1.ffn_down_exps.weight"),
                },
            },
        }
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

    fn tiny_config() -> Config {
        Config {
            model_type: "glm_moe_dsa".to_string(),
            hidden_size: 256,
            num_layers: 2,
            dense_layers: 1,
            sparse_moe_layers: Some(1),
            vocab_size: 8,
            attention_heads: 1,
            qk_head_dim: 256,
            qk_no_rope_dim: 128,
            qk_rope_dim: 128,
            kv_lora_rank: 256,
            v_head_dim: Some(256),
            num_routed_experts: 4,
            experts_per_token: 2,
            moe_intermediate_size: 256,
            num_shared_experts: 1,
            moe_groups: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 2.5,
            scoring_func: "sigmoid".to_string(),
            topk_method: "noaux_tc".to_string(),
            max_context: 32,
            dsa_index_topk: 1,
            index_head_dim: 128,
            index_n_heads: 32,
            index_topk_freq: 4,
            indexer_rope_interleave: true,
            indexer_types: Vec::new(),
            num_nextn_predict_layers: 0,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000_000.0,
        }
    }

    fn write_sparse_ffn_fixture(ty: GgmlType, expert_first_layout: bool) -> PathBuf {
        let path = unique_temp_file("moe-ffn");
        let packed_gate_dims = if expert_first_layout {
            vec![4, 256, 256]
        } else {
            vec![256, 256, 4]
        };
        let specs = vec![
            TensorSpec::f32("blk.1.ffn_norm.weight", vec![256], None),
            TensorSpec::quant("blk.1.ffn_gate_inp.weight", vec![256, 4], ty),
            TensorSpec::f32(
                "blk.1.exp_probs_b.bias",
                vec![4],
                Some(vec![0.0, 0.4, 0.2, 0.8]),
            ),
            TensorSpec::quant("blk.1.ffn_gate_shexp.weight", vec![256, 256], ty),
            TensorSpec::quant("blk.1.ffn_up_shexp.weight", vec![256, 256], ty),
            TensorSpec::quant("blk.1.ffn_down_shexp.weight", vec![256, 256], ty),
            TensorSpec::quant("blk.1.ffn_gate_exps.weight", packed_gate_dims, ty),
            TensorSpec::quant("blk.1.ffn_up_exps.weight", vec![256, 256, 4], ty),
            TensorSpec::quant("blk.1.ffn_down_exps.weight", vec![256, 256, 4], ty),
        ];
        write_gguf(path, &specs)
    }

    #[derive(Debug)]
    struct TensorSpec {
        name: &'static str,
        dims: Vec<u64>,
        ty: GgmlType,
        values: Option<Vec<f32>>,
    }

    impl TensorSpec {
        fn f32(name: &'static str, dims: Vec<u64>, values: Option<Vec<f32>>) -> Self {
            Self {
                name,
                dims,
                ty: GgmlType::F32,
                values,
            }
        }

        fn quant(name: &'static str, dims: Vec<u64>, ty: GgmlType) -> Self {
            Self {
                name,
                dims,
                ty,
                values: None,
            }
        }

        fn payload_len(&self) -> u64 {
            match self.ty {
                GgmlType::F32 => self.dims.iter().product::<u64>() * 4,
                GgmlType::Q2K => self.dims.iter().product::<u64>() / 256 * GGML_Q2_K_BLOCK_BYTES,
                other => panic!("unsupported fixture tensor type {other}"),
            }
        }
    }

    fn write_gguf(path: PathBuf, specs: &[TensorSpec]) -> PathBuf {
        let mut writer = GgufWriter::new();
        writer.header(specs.len() as u64, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);

        let mut offset = 0_u64;
        let offsets = specs
            .iter()
            .map(|spec| {
                let current = offset;
                offset = align_u64(current + spec.payload_len(), 32);
                current
            })
            .collect::<Vec<_>>();
        for (spec, offset) in specs.iter().zip(offsets.iter().copied()) {
            writer.tensor_info(spec.name, &spec.dims, spec.ty, offset);
        }
        writer.pad_to(32);

        for (spec, offset) in specs.iter().zip(offsets.iter().copied()) {
            writer.pad_to_absolute_data_offset(offset);
            match spec.ty {
                GgmlType::F32 => {
                    let element_count = spec.dims.iter().product::<u64>() as usize;
                    let values = spec
                        .values
                        .clone()
                        .unwrap_or_else(|| vec![1.0_f32; element_count]);
                    assert_eq!(values.len(), element_count);
                    for value in values {
                        writer.bytes(&value.to_le_bytes());
                    }
                }
                GgmlType::Q2K => {
                    for _ in 0..spec.payload_len() / GGML_Q2_K_BLOCK_BYTES {
                        writer.bytes(&vec![0_u8; GGML_Q2_K_BLOCK_BYTES as usize]);
                    }
                }
                other => panic!("unsupported fixture tensor type {other}"),
            }
        }
        writer.pad_to_absolute_data_offset(offset);
        writer.finish_to(path)
    }

    struct GgufWriter {
        bytes: Vec<u8>,
        data_start: Option<usize>,
    }

    impl GgufWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                data_start: None,
            }
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
            self.data_start = Some(self.bytes.len());
        }

        fn pad_to_absolute_data_offset(&mut self, offset: u64) {
            let target = self.data_start.unwrap() + offset as usize;
            if self.bytes.len() < target {
                self.bytes.resize(target, 0);
            }
        }

        fn bytes(&mut self, bytes: &[u8]) {
            self.bytes.extend_from_slice(bytes);
        }

        fn finish_to(self, path: PathBuf) -> PathBuf {
            fs::write(&path, self.bytes).unwrap();
            path
        }
    }

    fn align_u64(value: u64, alignment: u64) -> u64 {
        let remainder = value % alignment;
        if remainder == 0 {
            value
        } else {
            value + alignment - remainder
        }
    }

    fn unique_temp_file(label: &str) -> PathBuf {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("inferno-{label}-{}-{id}", std::process::id()))
    }
}
