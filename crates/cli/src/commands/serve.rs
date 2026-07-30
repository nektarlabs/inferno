use std::{
    collections::{HashMap, HashSet},
    net::{SocketAddr, TcpListener},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::Result;
use backend::{Backend, MetalBackend};
use common::{Error, Result as InfernoResult};
use config::{
    detect_model_architecture, load_config, load_generation_config, load_laguna_config, Config,
    LagunaConfig, ModelArchitecture,
};
use gguf::GgufFile;
use inferno_io::EXPERT_PACK_FILE_NAME;
use model::{
    expected_expert_pack_header, validate_routing_policy, LagunaArtifactKind, LagunaModel, Model,
    DEFAULT_GGUF_OUTPUT_CHUNK_ROWS,
};
use runtime::{
    enable_laguna_memory_controller_log, enable_memory_controller_log, enable_memory_telemetry,
    enable_memory_telemetry_file, q2_memory_controller_spec, run_generate_streaming_with_options,
    CacheBudgetSpec, GenerationControl, GenerationOptions, LagunaRuntime, LagunaThinkingGuard,
    LAGUNA_THINKING_END_TOKEN_ID,
};
use server::{
    ResponseUsage, ResponsesHandler, ResponsesRequest, ResponsesStream, ServerAdmission,
    GLM_CODEX_MODEL_ID, LAGUNA_CODEX_MODEL_ID, LAGUNA_GGUF_CODEX_MODEL_ID,
    LAGUNA_XS_GGUF_CODEX_MODEL_ID,
};
use tokenizer::{
    is_supported_codex_function, map_laguna_function_call_to_codex, parse_agent_output,
    parse_complete_agent_tool_call, render_codex_prompt, render_laguna_codex_prompt,
    render_laguna_xs_codex_prompt, AgentFunctionCall, AgentOutput, AgentOutputItem,
    LagunaThinkingMode, Tokenizer,
};
use tracing::{debug, info};

use super::generate::{
    cache_gb_to_bytes, discover_config_path, discover_tokenizer_path, expert_cache_slots_per_layer,
    laguna_runtime_options, laguna_thinking_mode, load_q2_readiness, resolve_q2_artifact,
    validate_generation_request, validate_laguna_prompt, validate_laguna_service_options,
    validate_laguna_tokenizer, validate_memory_controller_options, DecodedTextStream,
};

static NEXT_CALL_ID: AtomicU64 = AtomicU64::new(1);
const LAGUNA_GGUF_SERVICE_CONTEXT_TOKENS: usize = 32_768;
const LAGUNA_GGUF_DEFAULT_MAX_OUTPUT_TOKENS: usize = 8_192;
const LAGUNA_GGUF_MAX_INSTRUCTION_BYTES: usize = 8 * 1_024;
const LAGUNA_CODEX_DUPLICATE_RETRIES: usize = 2;
const LAGUNA_CODEX_MAX_INSPECTIONS_BEFORE_EDIT: usize = 3;
const LAGUNA_CODEX_MAX_IDENTICAL_TOOL_EXECUTIONS: usize = 2;
const CODEX_FALLBACK_INSTRUCTION_PREFIX: &str = "You are a coding agent running in the Codex CLI";
const CLOSED_THINKING_BOUNDARY: &str = "</think>";

#[allow(clippy::too_many_arguments)]
pub fn run(
    model_path: &Path,
    config_path: Option<&Path>,
    tokenizer_path: Option<&Path>,
    bind: SocketAddr,
    page_size: usize,
    max_new_tokens: Option<usize>,
    thinking: bool,
    speculative_mtp: bool,
    enable_unified_memory_controller: bool,
    expert_cache_gb: Option<f64>,
    hot_kv_cache_gb: Option<f64>,
    enable_telemetry: bool,
    telemetry_file: Option<&Path>,
    memory_controller_log: Option<&Path>,
) -> Result<()> {
    let discovered_config = discover_config_path(model_path, config_path)?;
    let discovered_tokenizer = discover_tokenizer_path(model_path, tokenizer_path)?;
    match detect_model_architecture(&discovered_config)? {
        ModelArchitecture::GlmMoeDsa => {
            if thinking {
                return Err(
                    Error::runtime("--thinking currently applies only to Laguna models").into(),
                );
            }
            run_glm(
                model_path,
                &discovered_config,
                &discovered_tokenizer,
                bind,
                page_size,
                max_new_tokens,
                speculative_mtp,
                enable_unified_memory_controller,
                expert_cache_gb,
                hot_kv_cache_gb,
                enable_telemetry,
                telemetry_file,
                memory_controller_log,
            )
        }
        ModelArchitecture::Laguna => run_laguna(
            model_path,
            &discovered_config,
            &discovered_tokenizer,
            bind,
            page_size,
            max_new_tokens,
            thinking,
            speculative_mtp,
            enable_unified_memory_controller,
            expert_cache_gb,
            hot_kv_cache_gb,
            enable_telemetry,
            telemetry_file,
            memory_controller_log,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_glm(
    model_path: &Path,
    config_path: &Path,
    tokenizer_path: &Path,
    bind: SocketAddr,
    page_size: usize,
    max_new_tokens: Option<usize>,
    speculative_mtp: bool,
    enable_unified_memory_controller: bool,
    expert_cache_gb: Option<f64>,
    hot_kv_cache_gb: Option<f64>,
    enable_telemetry: bool,
    telemetry_file: Option<&Path>,
    memory_controller_log: Option<&Path>,
) -> Result<()> {
    validate_memory_controller_options(enable_unified_memory_controller, memory_controller_log)?;
    let generation_config = load_generation_config(&model_path.join("generation_config.json"))?;
    let config = load_config(config_path)?;
    let artifact = resolve_q2_artifact(model_path)?;
    let gguf = GgufFile::open(&artifact.gguf_path)?;
    validate_routing_policy(&config, &gguf)?;
    let readiness = load_q2_readiness(&gguf, &artifact, &config)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;

    let expert_cache_budget_bytes = cache_gb_to_bytes("expert cache", expert_cache_gb)?;
    let hot_kv_cache_budget_bytes = cache_gb_to_bytes("hot KV cache", hot_kv_cache_gb)?;
    let backend = MetalBackend::new()?;
    let expert_cache_slots = expert_cache_budget_bytes
        .map(|budget| {
            expert_cache_slots_per_layer(
                &readiness.index,
                config.num_routed_experts,
                config.num_nextn_predict_layers > 0,
                budget,
            )
        })
        .transpose()?;
    if let Some(slots_per_layer) = expert_cache_slots {
        backend.configure_expert_cache_slots_per_layer(slots_per_layer)?;
    }
    let expert_pack_path = model_path.join(EXPERT_PACK_FILE_NAME);
    if expert_pack_path.is_file() {
        let header = expected_expert_pack_header(&gguf, &config, &readiness.index)?;
        backend.configure_expert_pack(&expert_pack_path, header)?;
    }
    let model = Model::open_from_index(
        &gguf,
        &config,
        readiness.index,
        &backend,
        DEFAULT_GGUF_OUTPUT_CHUNK_ROWS,
    )?;

    if let Some(path) = telemetry_file {
        enable_memory_telemetry_file(path)?;
    } else if enable_telemetry {
        enable_memory_telemetry();
    }
    if let Some(path) = memory_controller_log {
        enable_memory_controller_log(path)?;
    }
    let dynamic_cache_budget = enable_unified_memory_controller
        .then(|| {
            q2_memory_controller_spec(
                &config,
                page_size,
                expert_cache_slots,
                hot_kv_cache_budget_bytes,
                speculative_mtp,
            )
        })
        .transpose()?;

    let listener = TcpListener::bind(bind)?;
    let mut handler = GlmCodexHandler {
        model: &model,
        config: &config,
        backend: &backend,
        tokenizer: &tokenizer,
        eos_token_ids: &generation_config.eos_token_ids,
        artifact_file_name: &readiness.artifact_file_name,
        page_size,
        max_new_tokens,
        speculative_mtp,
        hot_kv_cache_budget_bytes,
        dynamic_cache_budget,
    };
    info!(%bind, model = GLM_CODEX_MODEL_ID, "Inferno model loaded for Codex");
    server::serve(listener, &mut handler, GLM_CODEX_MODEL_ID)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_laguna(
    model_path: &Path,
    config_path: &Path,
    tokenizer_path: &Path,
    bind: SocketAddr,
    page_size: usize,
    max_new_tokens: Option<usize>,
    thinking: bool,
    speculative_mtp: bool,
    enable_unified_memory_controller: bool,
    expert_cache_gb: Option<f64>,
    hot_kv_cache_gb: Option<f64>,
    enable_telemetry: bool,
    telemetry_file: Option<&Path>,
    memory_controller_log: Option<&Path>,
) -> Result<()> {
    validate_laguna_service_options(
        page_size,
        speculative_mtp,
        enable_unified_memory_controller,
        hot_kv_cache_gb,
        enable_telemetry,
        telemetry_file,
        memory_controller_log,
        expert_cache_gb,
    )?;
    let config = load_laguna_config(config_path)?;
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    validate_laguna_tokenizer(&tokenizer, &config)?;
    let backend = MetalBackend::new()?;
    let model = LagunaModel::open(model_path, config.clone(), &backend)?;
    // A server can receive any prompt size, so reserve cache headroom against
    // Laguna's complete configured KV context for Safetensors artifacts.
    let options = laguna_runtime_options(
        &model,
        &config,
        &backend,
        1,
        None,
        expert_cache_gb,
        enable_unified_memory_controller,
    )?;
    if let Some(path) = memory_controller_log {
        enable_laguna_memory_controller_log(path)?;
    }
    let runtime = LagunaRuntime::new(options)?;
    let model_id = match model.artifact_kind() {
        LagunaArtifactKind::SafetensorsInt4 => LAGUNA_CODEX_MODEL_ID,
        LagunaArtifactKind::AntirezGguf => LAGUNA_GGUF_CODEX_MODEL_ID,
        LagunaArtifactKind::PoolsideXsGguf => LAGUNA_XS_GGUF_CODEX_MODEL_ID,
    };

    let listener = TcpListener::bind(bind)?;
    let mut handler = LagunaCodexHandler {
        model: &model,
        config: &config,
        backend: &backend,
        tokenizer: &tokenizer,
        runtime,
        max_new_tokens,
        thinking_mode: laguna_thinking_mode(thinking),
        model_id,
    };
    info!(%bind, model = model_id, "Inferno model loaded for Codex");
    server::serve(listener, &mut handler, model_id)?;
    Ok(())
}

struct GlmCodexHandler<'runtime, 'weights> {
    model: &'runtime Model<'weights>,
    config: &'runtime Config,
    backend: &'runtime MetalBackend,
    tokenizer: &'runtime Tokenizer,
    eos_token_ids: &'runtime [u32],
    artifact_file_name: &'runtime str,
    page_size: usize,
    max_new_tokens: Option<usize>,
    speculative_mtp: bool,
    hot_kv_cache_budget_bytes: Option<usize>,
    dynamic_cache_budget: Option<CacheBudgetSpec>,
}

impl ResponsesHandler for GlmCodexHandler<'_, '_> {
    fn generate(
        &mut self,
        request: ResponsesRequest,
        stream: &mut ResponsesStream<'_>,
    ) -> InfernoResult<ResponseUsage> {
        validate_requested_model(&request.model, GLM_CODEX_MODEL_ID)?;
        let allowed_tools = tool_names(&request.tools)?;
        let prompt = render_codex_prompt(&request.instructions, &request.input, &request.tools)?;
        let encoded = self.tokenizer.encode(&prompt.rendered, false)?;
        validate_generation_request(
            self.config,
            self.eos_token_ids,
            &encoded.token_ids,
            self.max_new_tokens,
            self.page_size,
            self.artifact_file_name,
            &self.model.index().architecture,
            &self.model.index().summary,
        )
        .map_err(|error| Error::runtime(error.to_string()))?;

        let mut decoded = DecodedTextStream::new(self.tokenizer, true);
        let mut generated_text = String::new();
        let mut output_tokens = 0_usize;
        run_generate_streaming_with_options(
            self.model,
            self.config,
            self.backend,
            &encoded.token_ids,
            self.max_new_tokens,
            self.page_size,
            self.eos_token_ids,
            GenerationOptions {
                hot_kv_cache_budget_bytes: self.hot_kv_cache_budget_bytes,
                dynamic_cache_budget: self.dynamic_cache_budget,
                profile_token_costs: false,
                speculative_mtp: self.speculative_mtp,
            },
            |token_id| {
                output_tokens = output_tokens
                    .checked_add(1)
                    .ok_or_else(|| Error::runtime("Codex output token count overflow"))?;
                if let Some(text) = decoded.push(token_id)? {
                    generated_text.push_str(&text);
                }
                stream.heartbeat()
            },
        )?;

        emit_agent_output(&generated_text, &allowed_tools, "GLM", stream)?;

        Ok(ResponseUsage {
            input_tokens: encoded.token_ids.len(),
            output_tokens,
        })
    }
}

struct LagunaCodexHandler<'runtime> {
    model: &'runtime LagunaModel,
    config: &'runtime LagunaConfig,
    backend: &'runtime MetalBackend,
    tokenizer: &'runtime Tokenizer,
    runtime: LagunaRuntime,
    max_new_tokens: Option<usize>,
    thinking_mode: LagunaThinkingMode,
    model_id: &'static str,
}

impl ResponsesHandler for LagunaCodexHandler<'_> {
    fn admission(&self) -> ServerAdmission {
        if is_laguna_gguf_model_id(self.model_id) {
            ServerAdmission::RejectWhenBusy
        } else {
            ServerAdmission::Queue
        }
    }

    fn generate(
        &mut self,
        request: ResponsesRequest,
        stream: &mut ResponsesStream<'_>,
    ) -> InfernoResult<ResponseUsage> {
        validate_requested_model(&request.model, self.model_id)?;
        if is_laguna_gguf_model_id(self.model_id) {
            validate_laguna_gguf_codex_envelope(&request)?;
        }
        let allowed_tools = tool_names(&request.tools)?;
        let tool_phase = laguna_tool_phase(&request.input);
        let prompt_tools = laguna_prompt_tools(tool_phase, &request.tools);
        let prompt_allowed_tools = tool_names(&prompt_tools)?;
        let mut logged_tools = allowed_tools.iter().cloned().collect::<Vec<_>>();
        logged_tools.sort_unstable();
        let mut logged_prompt_tools = prompt_allowed_tools.iter().cloned().collect::<Vec<_>>();
        logged_prompt_tools.sort_unstable();
        debug!(
            model = self.model_id,
            tools = ?logged_tools,
            prompt_tools = ?logged_prompt_tools,
            "accepted Codex tool contract"
        );
        let completed_calls = historical_tool_call_fingerprints(&request.input)?;
        let inspection_limit_reached = tool_phase == LagunaToolPhase::Edit
            && historical_read_only_shell_calls_since_last_mutation(&request.input)
                >= LAGUNA_CODEX_MAX_INSPECTIONS_BEFORE_EDIT;
        let thinking_mode = self.thinking_mode;
        let mut input = request.input.clone();
        let mut total_input_tokens = 0_usize;
        let mut total_output_tokens = 0_usize;
        let mut require_new_tool_call = false;

        for attempt in 0..=LAGUNA_CODEX_DUPLICATE_RETRIES {
            let suppress_exec_replay = inspection_limit_reached
                || require_new_tool_call
                || tool_phase == LagunaToolPhase::Complete;
            let prompt_input = laguna_prompt_input(
                &input,
                &prompt_allowed_tools,
                tool_phase,
                suppress_exec_replay,
            );
            let turn = self.generate_agent_turn(
                &request.instructions,
                &prompt_input,
                &prompt_tools,
                request.max_output_tokens,
                thinking_mode,
                stream,
            )?;
            total_input_tokens = total_input_tokens
                .checked_add(turn.input_tokens)
                .ok_or_else(|| Error::runtime("Codex input token count overflow"))?;
            total_output_tokens = total_output_tokens
                .checked_add(turn.output_tokens)
                .ok_or_else(|| Error::runtime("Codex output token count overflow"))?;

            if let Some(duplicate) = duplicate_tool_call(&turn.output, &completed_calls)? {
                if attempt == LAGUNA_CODEX_DUPLICATE_RETRIES {
                    return Err(Error::runtime(format!(
                        "Laguna repeated completed Codex tool {:?} after {} internal retries",
                        duplicate.name, LAGUNA_CODEX_DUPLICATE_RETRIES
                    )));
                }
                debug!(
                    tool = duplicate.name,
                    attempt = attempt + 1,
                    "retrying Laguna Codex turn after duplicate tool call"
                );
                input.push(duplicate_tool_retry_message(&duplicate));
                require_new_tool_call = true;
                stream.heartbeat()?;
                continue;
            }
            if inspection_limit_reached {
                if let Some(inspection) = first_read_only_shell_call(&turn.output)? {
                    if attempt == LAGUNA_CODEX_DUPLICATE_RETRIES {
                        return Err(Error::runtime(format!(
                            "Laguna repeated workspace inspection with {:?} after {} internal retries",
                            inspection.name, LAGUNA_CODEX_DUPLICATE_RETRIES
                        )));
                    }
                    input.push(repeated_inspection_retry_message());
                    require_new_tool_call = true;
                    stream.heartbeat()?;
                    continue;
                }
            }
            if let Some(disallowed) =
                first_tool_outside_contract(&turn.output, &prompt_allowed_tools)
            {
                if attempt == LAGUNA_CODEX_DUPLICATE_RETRIES {
                    return Err(Error::runtime(format!(
                        "Laguna requested unavailable Codex tool {:?} after {} internal retries",
                        disallowed.name, LAGUNA_CODEX_DUPLICATE_RETRIES
                    )));
                }
                input.push(unavailable_tool_retry_message(&disallowed));
                require_new_tool_call = tool_phase != LagunaToolPhase::Complete;
                stream.heartbeat()?;
                continue;
            }
            if turn.hit_output_limit
                || (require_new_tool_call && !agent_output_has_tool(&turn.output))
            {
                if attempt == LAGUNA_CODEX_DUPLICATE_RETRIES {
                    return Err(Error::runtime(
                        "Laguna did not produce the required new Codex tool call before the internal retry limit",
                    ));
                }
                input.push(required_tool_retry_message(turn.hit_output_limit));
                require_new_tool_call = true;
                stream.heartbeat()?;
                continue;
            }

            emit_laguna_agent_output(
                turn.output,
                "",
                &prompt_allowed_tools,
                "Laguna",
                self.model_id == LAGUNA_XS_GGUF_CODEX_MODEL_ID
                    && thinking_mode == LagunaThinkingMode::Enabled,
                stream,
            )?;
            return Ok(ResponseUsage {
                input_tokens: total_input_tokens,
                output_tokens: total_output_tokens,
            });
        }
        Err(Error::runtime(
            "Laguna Codex duplicate retry loop exhausted",
        ))
    }
}

struct LagunaAgentTurn {
    output: AgentOutput,
    input_tokens: usize,
    output_tokens: usize,
    hit_output_limit: bool,
}

impl LagunaCodexHandler<'_> {
    fn generate_agent_turn(
        &mut self,
        instructions: &str,
        input: &[serde_json::Value],
        tools: &[serde_json::Value],
        requested_max_output_tokens: Option<usize>,
        thinking_mode: LagunaThinkingMode,
        stream: &mut ResponsesStream<'_>,
    ) -> InfernoResult<LagunaAgentTurn> {
        let prompt = if self.model_id == LAGUNA_XS_GGUF_CODEX_MODEL_ID {
            render_laguna_xs_codex_prompt(instructions, input, tools, thinking_mode)?
        } else {
            render_laguna_codex_prompt(instructions, input, tools, thinking_mode)?
        };
        let encoded = self.tokenizer.encode(&prompt.rendered, false)?;
        let max_new_tokens = laguna_codex_max_new_tokens(
            self.model_id,
            self.max_new_tokens,
            requested_max_output_tokens,
            encoded.token_ids.len(),
        )?;
        validate_laguna_prompt(
            self.config,
            &encoded.token_ids,
            max_new_tokens,
            thinking_mode,
        )?;
        debug!(
            model = self.model_id,
            prompt_tokens = encoded.token_ids.len(),
            max_output_tokens = max_new_tokens,
            "accepted Laguna Codex request"
        );

        let mut decoded = DecodedTextStream::new(self.tokenizer, true);
        let mut generated_text = match thinking_mode {
            LagunaThinkingMode::Disabled => String::from(CLOSED_THINKING_BOUNDARY),
            LagunaThinkingMode::Enabled => String::new(),
        };
        let mut completed_output = None;
        let mut output_tokens = 0_usize;
        let mut thinking_guard =
            (thinking_mode == LagunaThinkingMode::Enabled).then(LagunaThinkingGuard::default);
        let heartbeat_during_prefill = is_laguna_gguf_model_id(self.model_id);
        let stream_cell = std::cell::RefCell::new(stream);
        let generation = self
            .runtime
            .generate_streaming_controlled_with_prefill_progress(
                self.model,
                self.backend,
                &encoded.token_ids,
                max_new_tokens,
                &self.config.eos_token_id,
                |progress| {
                    debug!(
                        model = self.model_id,
                        processed_tokens = progress.processed_tokens,
                        prompt_tokens = progress.prompt_tokens,
                        "Laguna Codex prefill progress"
                    );
                    if heartbeat_during_prefill {
                        stream_cell.borrow_mut().heartbeat()?;
                    }
                    Ok(())
                },
                |token_id| {
                    output_tokens = output_tokens
                        .checked_add(1)
                        .ok_or_else(|| Error::runtime("Codex output token count overflow"))?;
                    let is_stop_token = self.config.eos_token_id.contains(&token_id);
                    if !is_stop_token {
                        if let Some(text) = decoded.push(token_id)? {
                            generated_text.push_str(&text);
                        }
                    }
                    let forced_boundary = thinking_guard
                        .as_mut()
                        .and_then(|guard| guard.observe(token_id, is_stop_token));
                    if let Some(reason) = forced_boundary {
                        debug!(?reason, "forcing Laguna reasoning boundary");
                        if let Some(text) = decoded.push(LAGUNA_THINKING_END_TOKEN_ID)? {
                            generated_text.push_str(&text);
                        }
                        if !generated_text.contains(CLOSED_THINKING_BOUNDARY) {
                            generated_text.push_str(CLOSED_THINKING_BOUNDARY);
                        }
                    }
                    let control = match parse_complete_agent_tool_call(&generated_text)? {
                        Some(output) => {
                            completed_output = Some(output);
                            GenerationControl::Stop
                        }
                        None if forced_boundary.is_some() => {
                            GenerationControl::InjectNextToken(LAGUNA_THINKING_END_TOKEN_ID)
                        }
                        None => GenerationControl::Continue,
                    };
                    stream_cell.borrow_mut().heartbeat()?;
                    Ok(control)
                },
            )?;
        let output = completed_output.unwrap_or(parse_agent_output(&generated_text)?);
        Ok(LagunaAgentTurn {
            output,
            input_tokens: encoded.token_ids.len(),
            output_tokens,
            hit_output_limit: max_new_tokens
                .is_some_and(|limit| generation.generated_tokens >= limit),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LagunaToolPhase {
    Initial,
    Edit,
    Validate,
    Complete,
}

fn laguna_prompt_tools(
    phase: LagunaToolPhase,
    tools: &[serde_json::Value],
) -> Vec<serde_json::Value> {
    let has_apply_patch = tools.iter().any(|tool| {
        tool.get("type").and_then(serde_json::Value::as_str) == Some("custom")
            && tool.get("name").and_then(serde_json::Value::as_str) == Some("apply_patch")
    });
    if !has_apply_patch
        || matches!(
            phase,
            LagunaToolPhase::Initial | LagunaToolPhase::Edit | LagunaToolPhase::Validate
        )
    {
        return tools.to_vec();
    }
    if phase == LagunaToolPhase::Complete {
        return Vec::new();
    }

    tools
        .iter()
        .filter(|tool| tool.get("name").and_then(serde_json::Value::as_str) != Some("exec_command"))
        .cloned()
        .collect()
}

fn laguna_prompt_input(
    input: &[serde_json::Value],
    allowed_tools: &HashSet<String>,
    phase: LagunaToolPhase,
    suppress_exec_replay: bool,
) -> Vec<serde_json::Value> {
    if !suppress_exec_replay {
        return input.to_vec();
    }
    if phase != LagunaToolPhase::Edit
        && phase != LagunaToolPhase::Complete
        && !allowed_tools.contains("exec_command")
    {
        return input.to_vec();
    }

    let mut prompt_input = input
        .iter()
        .enumerate()
        .filter(|(index, item)| {
            if is_historical_exec_command(item) {
                return false;
            }
            if item.get("type").and_then(serde_json::Value::as_str) == Some("message")
                && item.get("role").and_then(serde_json::Value::as_str) == Some("assistant")
            {
                return !next_non_reasoning_item_is_exec_command(input, index + 1);
            }
            true
        })
        .map(|(_, item)| item.clone())
        .collect::<Vec<_>>();
    let instruction = match phase {
        LagunaToolPhase::Edit => {
            "Workspace inspection has completed. Do not run another read-only inspection and do not describe the plan. Create or update the requested files now. You may call apply_patch, or emit a mutating shell call in exactly this shape: <tool_call>shell<arg_key>cmd</arg_key><arg_value>mkdir -p src</arg_value></tool_call>. Adapt the command to the task and keep each call small. If the requested work is already complete, return the final answer."
        }
        LagunaToolPhase::Complete => {
            "The requested file change and its validation both succeeded. Do not call another tool. Return a concise final answer now."
        }
        LagunaToolPhase::Initial | LagunaToolPhase::Validate => return prompt_input,
    };
    prompt_input.push(serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": instruction}],
    }));
    prompt_input
}

fn is_historical_exec_command(item: &serde_json::Value) -> bool {
    item.get("type").and_then(serde_json::Value::as_str) == Some("function_call")
        && item.get("name").and_then(serde_json::Value::as_str) == Some("exec_command")
}

fn next_non_reasoning_item_is_exec_command(input: &[serde_json::Value], start: usize) -> bool {
    input[start..]
        .iter()
        .find(|item| item.get("type").and_then(serde_json::Value::as_str) != Some("reasoning"))
        .is_some_and(is_historical_exec_command)
}

fn laguna_tool_phase(input: &[serde_json::Value]) -> LagunaToolPhase {
    let Some((call_index, call)) = input.iter().enumerate().rev().find(|(_, item)| {
        matches!(
            item.get("type").and_then(serde_json::Value::as_str),
            Some("function_call" | "custom_tool_call")
        )
    }) else {
        return LagunaToolPhase::Initial;
    };
    match call.get("name").and_then(serde_json::Value::as_str) {
        Some("apply_patch") if historical_tool_call_succeeded(input, call_index, call) => {
            LagunaToolPhase::Validate
        }
        Some("apply_patch") => LagunaToolPhase::Edit,
        Some("exec_command")
            if prior_successful_apply_patch(input, call_index)
                && historical_tool_call_succeeded(input, call_index, call) =>
        {
            LagunaToolPhase::Complete
        }
        Some("exec_command")
            if historical_tool_call_succeeded(input, call_index, call)
                && exec_command_changes_files(call) =>
        {
            LagunaToolPhase::Validate
        }
        Some("exec_command") => LagunaToolPhase::Edit,
        _ => LagunaToolPhase::Initial,
    }
}

fn prior_successful_apply_patch(input: &[serde_json::Value], before_index: usize) -> bool {
    input[..before_index]
        .iter()
        .enumerate()
        .any(|(index, item)| {
            item.get("type").and_then(serde_json::Value::as_str) == Some("custom_tool_call")
                && item.get("name").and_then(serde_json::Value::as_str) == Some("apply_patch")
                && historical_tool_call_succeeded(input, index, item)
        })
}

fn historical_tool_call_succeeded(
    input: &[serde_json::Value],
    call_index: usize,
    call: &serde_json::Value,
) -> bool {
    let Some(call_id) = call.get("call_id").and_then(serde_json::Value::as_str) else {
        return false;
    };
    let Some(output) = input[call_index + 1..].iter().find(|item| {
        matches!(
            item.get("type").and_then(serde_json::Value::as_str),
            Some("function_call_output" | "custom_tool_call_output")
        ) && item.get("call_id").and_then(serde_json::Value::as_str) == Some(call_id)
    }) else {
        return false;
    };
    let Some(output) = output.get("output").and_then(serde_json::Value::as_str) else {
        return false;
    };
    let normalized = output.to_ascii_lowercase();
    if normalized.contains("process exited with code ") {
        return normalized.contains("process exited with code 0");
    }
    !["error", "failed", "invalid", "rejected"]
        .iter()
        .any(|marker| normalized.contains(marker))
}

fn historical_read_only_shell_calls_since_last_mutation(input: &[serde_json::Value]) -> usize {
    let start = input
        .iter()
        .enumerate()
        .rev()
        .find(|(index, call)| {
            let name = call.get("name").and_then(serde_json::Value::as_str);
            let changes_files = name == Some("apply_patch")
                || (name == Some("exec_command") && exec_command_changes_files(call));
            changes_files && historical_tool_call_succeeded(input, *index, call)
        })
        .map_or(0, |(index, _)| index + 1);
    input[start..]
        .iter()
        .filter(|call| {
            call.get("type").and_then(serde_json::Value::as_str) == Some("function_call")
                && call.get("name").and_then(serde_json::Value::as_str) == Some("exec_command")
                && !exec_command_changes_files(call)
        })
        .count()
}

fn exec_command_changes_files(call: &serde_json::Value) -> bool {
    let Some(arguments) = call
        .get("arguments")
        .and_then(serde_json::Value::as_str)
        .and_then(|arguments| serde_json::from_str::<serde_json::Value>(arguments).ok())
    else {
        return false;
    };
    arguments
        .get("cmd")
        .and_then(serde_json::Value::as_str)
        .is_some_and(shell_command_changes_files)
}

fn shell_command_changes_files(command: &str) -> bool {
    const MUTATION_MARKERS: &[&str] = &[
        "mkdir ",
        "touch ",
        "cargo init",
        "cargo new",
        "cargo fmt",
        "cat >",
        "cat <<",
        "printf ",
        "tee ",
        "cp ",
        "mv ",
        "install ",
        "sed -i",
        "perl -i",
        "git apply",
        "apply_patch",
    ];
    MUTATION_MARKERS
        .iter()
        .any(|marker| command.contains(marker))
}

fn historical_tool_call_fingerprints(
    input: &[serde_json::Value],
) -> InfernoResult<HashMap<String, usize>> {
    let mut fingerprints = HashMap::new();
    for item in input.iter().filter(|item| {
        matches!(
            item.get("type").and_then(serde_json::Value::as_str),
            Some("function_call" | "custom_tool_call")
        )
    }) {
        let name = item
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| Error::tokenizer("historical Codex tool call is missing name"))?;
        let arguments = if item.get("type").and_then(serde_json::Value::as_str)
            == Some("custom_tool_call")
        {
            let input = item
                .get("input")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    Error::tokenizer("historical Codex custom tool call is missing input")
                })?;
            serde_json::json!({"patch": input}).to_string()
        } else {
            item.get("arguments")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| Error::tokenizer("historical Codex tool call is missing arguments"))?
                .to_string()
        };
        let fingerprint = tool_call_fingerprint(&AgentFunctionCall {
            name: name.to_string(),
            arguments,
        })?;
        *fingerprints.entry(fingerprint).or_insert(0) += 1;
    }
    Ok(fingerprints)
}

fn duplicate_tool_call(
    output: &AgentOutput,
    completed_calls: &HashMap<String, usize>,
) -> InfernoResult<Option<AgentFunctionCall>> {
    for item in &output.items {
        let AgentOutputItem::FunctionCall(call) = item else {
            continue;
        };
        let normalized = map_laguna_function_call_to_codex(call.clone());
        if completed_calls
            .get(&tool_call_fingerprint(&normalized)?)
            .copied()
            .unwrap_or(0)
            >= LAGUNA_CODEX_MAX_IDENTICAL_TOOL_EXECUTIONS
        {
            return Ok(Some(normalized));
        }
    }
    Ok(None)
}

fn first_tool_outside_contract(
    output: &AgentOutput,
    allowed_tools: &HashSet<String>,
) -> Option<AgentFunctionCall> {
    output.items.iter().find_map(|item| {
        let AgentOutputItem::FunctionCall(call) = item else {
            return None;
        };
        let normalized = map_laguna_function_call_to_codex(call.clone());
        (!allowed_tools.contains(&normalized.name)).then_some(normalized)
    })
}

fn first_read_only_shell_call(output: &AgentOutput) -> InfernoResult<Option<AgentFunctionCall>> {
    for item in &output.items {
        let AgentOutputItem::FunctionCall(call) = item else {
            continue;
        };
        let normalized = map_laguna_function_call_to_codex(call.clone());
        if normalized.name != "exec_command" {
            continue;
        }
        let arguments: serde_json::Value = serde_json::from_str(&normalized.arguments)?;
        let command = arguments
            .get("cmd")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| Error::runtime("Laguna shell call requires a string cmd argument"))?;
        if !shell_command_changes_files(command) {
            return Ok(Some(normalized));
        }
    }
    Ok(None)
}

fn tool_call_fingerprint(call: &AgentFunctionCall) -> InfernoResult<String> {
    let normalized = map_laguna_function_call_to_codex(call.clone());
    let arguments: serde_json::Value = serde_json::from_str(&normalized.arguments)?;
    if !arguments.is_object() {
        return Err(Error::tokenizer(
            "Codex tool-call arguments must encode a JSON object",
        ));
    }
    Ok(format!(
        "{}\u{0}{}",
        normalized.name,
        serde_json::to_string(&arguments)?
    ))
}

fn duplicate_tool_retry_message(call: &AgentFunctionCall) -> serde_json::Value {
    let mut arguments = call.arguments.chars().take(512).collect::<String>();
    if arguments.len() < call.arguments.len() {
        arguments.push_str("...");
    }
    let next_action = if call.name == "exec_command" {
        "The repeated shell command was not run. Do not inspect again. Create or update the requested files now with apply_patch, or emit a mutating shell call such as <tool_call>shell<arg_key>cmd</arg_key><arg_value>printf 'content' > path</arg_value></tool_call>."
    } else {
        "Inferno did not run it again. Use the existing result and choose a different action that advances the user's unfinished request now."
    };
    serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": format!(
                "Your proposed {name} call with arguments {arguments} exactly duplicates a completed call. {next_action}",
                name = call.name,
            ),
        }],
    })
}

fn unavailable_tool_retry_message(call: &AgentFunctionCall) -> serde_json::Value {
    serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": format!(
                "The {name} tool is unavailable in this turn because the inspection step already completed. Do not inspect again. Use apply_patch now if files still need changes, or provide the final answer if the task is complete.",
                name = call.name,
            ),
        }],
    })
}

fn repeated_inspection_retry_message() -> serde_json::Value {
    serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": "Workspace inspection already completed and Inferno did not run another read-only command. Do not describe the plan. Create or update files now with apply_patch, or emit a mutating shell call in exactly this shape: <tool_call>shell<arg_key>cmd</arg_key><arg_value>mkdir -p src</arg_value></tool_call>. Adapt the command to the task. Do not call ls, find, pwd, cat, head, or another inspection command.",
        }],
    })
}

fn agent_output_has_tool(output: &AgentOutput) -> bool {
    output
        .items
        .iter()
        .any(|item| matches!(item, AgentOutputItem::FunctionCall(_)))
}

fn required_tool_retry_message(hit_output_limit: bool) -> serde_json::Value {
    let reason = if hit_output_limit {
        "Your previous attempt reached the output limit without completing an action."
    } else {
        "Your previous attempt did not provide the required new action."
    };
    serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": format!(
                "{reason} Do not repeat the plan or print repository contents as prose. Call a tool immediately with a different action that advances the user's unfinished request."
            ),
        }],
    })
}

fn validate_laguna_gguf_codex_envelope(request: &ResponsesRequest) -> InfernoResult<()> {
    let instructions = request.instructions.trim_start();
    if instructions.starts_with(CODEX_FALLBACK_INSTRUCTION_PREFIX) {
        return Err(Error::runtime(
            "Codex is using fallback metadata for laguna-s-2.1-gguf; install the current examples/inferno.models.json as ~/.codex/inferno.models.json and restart Codex",
        ));
    }
    if request.instructions.len() > LAGUNA_GGUF_MAX_INSTRUCTION_BYTES {
        return Err(Error::runtime(format!(
            "Laguna GGUF Codex instructions exceed the {} KiB service limit; install the current Inferno model catalog or reduce custom instructions",
            LAGUNA_GGUF_MAX_INSTRUCTION_BYTES / 1_024
        )));
    }
    Ok(())
}

fn is_laguna_gguf_model_id(model_id: &str) -> bool {
    matches!(
        model_id,
        LAGUNA_GGUF_CODEX_MODEL_ID | LAGUNA_XS_GGUF_CODEX_MODEL_ID
    )
}

fn laguna_codex_max_new_tokens(
    model_id: &str,
    configured_limit: Option<usize>,
    requested_limit: Option<usize>,
    prompt_tokens: usize,
) -> InfernoResult<Option<usize>> {
    if !is_laguna_gguf_model_id(model_id) {
        return Ok(configured_limit);
    }
    let available_tokens = LAGUNA_GGUF_SERVICE_CONTEXT_TOKENS
        .checked_sub(prompt_tokens)
        .filter(|available| *available > 0)
        .ok_or_else(|| {
            Error::runtime(format!(
                "Laguna GGUF Codex prompt has {prompt_tokens} tokens, but the production service context is {} tokens; compact the Codex conversation and retry",
                LAGUNA_GGUF_SERVICE_CONTEXT_TOKENS
            ))
        })?;
    let server_limit = configured_limit.unwrap_or(LAGUNA_GGUF_DEFAULT_MAX_OUTPUT_TOKENS);
    let request_limit = requested_limit.unwrap_or(server_limit);
    Ok(Some(server_limit.min(request_limit).min(available_tokens)))
}

fn emit_laguna_agent_output(
    output: AgentOutput,
    streamed_text: &str,
    allowed_tools: &HashSet<String>,
    model_name: &str,
    preserve_reasoning: bool,
    stream: &mut ResponsesStream<'_>,
) -> InfernoResult<()> {
    if preserve_reasoning {
        stream.reasoning_done(&output.reasoning)?;
    }
    let mut streamed_message_finished = false;
    for item in output.items {
        match item {
            AgentOutputItem::Text(text) if !streamed_text.is_empty() => {
                if streamed_message_finished || text != streamed_text.trim() {
                    return Err(Error::runtime(
                        "Laguna streamed text does not match its completed agent output",
                    ));
                }
                stream.message_done(streamed_text)?;
                streamed_message_finished = true;
            }
            AgentOutputItem::Text(text) => {
                stream.text_delta(&text)?;
                stream.message_done(&text)?;
            }
            AgentOutputItem::FunctionCall(call) => {
                emit_function_call(
                    map_laguna_function_call_to_codex(call),
                    allowed_tools,
                    model_name,
                    stream,
                )?;
            }
        }
    }
    Ok(())
}

fn emit_agent_output(
    generated_text: &str,
    allowed_tools: &HashSet<String>,
    model_name: &str,
    stream: &mut ResponsesStream<'_>,
) -> InfernoResult<()> {
    let output = parse_agent_output(generated_text)?;
    emit_parsed_agent_output(output, allowed_tools, model_name, stream)
}

fn emit_parsed_agent_output(
    output: AgentOutput,
    allowed_tools: &HashSet<String>,
    model_name: &str,
    stream: &mut ResponsesStream<'_>,
) -> InfernoResult<()> {
    for item in output.items {
        match item {
            AgentOutputItem::Text(text) => {
                stream.text_delta(&text)?;
                stream.message_done(&text)?;
            }
            AgentOutputItem::FunctionCall(call) => {
                emit_function_call(call, allowed_tools, model_name, stream)?;
            }
        }
    }
    Ok(())
}

fn emit_function_call(
    call: tokenizer::AgentFunctionCall,
    allowed_tools: &HashSet<String>,
    model_name: &str,
    stream: &mut ResponsesStream<'_>,
) -> InfernoResult<()> {
    if !allowed_tools.contains(&call.name) {
        return Err(Error::runtime(format!(
            "{model_name} requested undeclared Codex tool {:?}",
            call.name
        )));
    }
    let call_id = format!(
        "call_inferno_{}",
        NEXT_CALL_ID.fetch_add(1, Ordering::Relaxed)
    );
    if call.name == "apply_patch" {
        let arguments: serde_json::Value = serde_json::from_str(&call.arguments)?;
        let patch = arguments
            .get("patch")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                Error::runtime("Laguna apply_patch call requires a string patch argument")
            })?;
        let patch = normalize_laguna_apply_patch(patch);
        return stream.custom_tool_call_done(&call_id, &call.name, &patch);
    }
    stream.function_call_done(&call_id, &call.name, &call.arguments)
}

fn normalize_laguna_apply_patch(patch: &str) -> String {
    let mut normalized = String::with_capacity(patch.len());
    let mut add_file_body = false;
    for line in patch.lines() {
        if line.starts_with("*** Add File: ") {
            add_file_body = true;
            normalized.push_str(line);
        } else if line.starts_with("*** ") {
            add_file_body = false;
            normalized.push_str(line);
        } else if add_file_body && !line.starts_with('+') {
            normalized.push('+');
            normalized.push_str(line);
        } else {
            normalized.push_str(line);
        }
        normalized.push('\n');
    }
    if !patch.ends_with('\n') {
        normalized.pop();
    }
    normalized
}

fn validate_requested_model(requested: &str, loaded: &str) -> InfernoResult<()> {
    if requested != loaded {
        return Err(Error::runtime(format!(
            "Inferno serves model {loaded:?}, got {requested:?}"
        )));
    }
    Ok(())
}

fn tool_names(tools: &[serde_json::Value]) -> InfernoResult<HashSet<String>> {
    tools
        .iter()
        .filter(|tool| {
            matches!(
                tool.get("type").and_then(serde_json::Value::as_str),
                Some("function" | "custom")
            )
        })
        .filter(|tool| {
            tool.get("name")
                .and_then(serde_json::Value::as_str)
                .is_some_and(is_supported_codex_function)
        })
        .map(|tool| {
            tool.get("name")
                .and_then(serde_json::Value::as_str)
                .filter(|name| !name.is_empty())
                .map(str::to_string)
                .ok_or_else(|| Error::runtime("Codex tool definition is missing a string name"))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tokenizer::streamable_agent_text;

    use super::*;

    #[test]
    fn extracts_only_declared_codex_tool_names() {
        let names = tool_names(&[
            json!({"type": "function", "name": "exec_command"}),
            json!({"type": "function", "name": "write_stdin"}),
            json!({"type": "custom", "name": "apply_patch"}),
            json!({"type": "function", "name": "update_plan"}),
            json!({"type": "namespace", "name": "plugin", "tools": []}),
            json!({"type": "web_search", "external_web_access": false}),
        ])
        .unwrap();

        assert_eq!(names.len(), 3);
        assert!(names.contains("exec_command"));
        assert!(names.contains("write_stdin"));
        assert!(names.contains("apply_patch"));
        assert!(!names.contains("update_plan"));
    }

    #[test]
    fn laguna_moves_through_inspect_edit_validate_and_complete_phases() {
        let tools = vec![
            json!({"type": "function", "name": "exec_command"}),
            json!({"type": "function", "name": "write_stdin"}),
            json!({"type": "custom", "name": "apply_patch"}),
        ];
        let after_inspection = vec![
            json!({"type": "message", "role": "assistant", "content": "Inspecting."}),
            json!({"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\":\"ls\"}", "call_id": "call_1"}),
            json!({"type": "function_call_output", "call_id": "call_1", "output": "README.md"}),
        ];

        let inspection_phase = laguna_tool_phase(&after_inspection);
        assert_eq!(inspection_phase, LagunaToolPhase::Edit);
        assert_eq!(
            historical_read_only_shell_calls_since_last_mutation(&after_inspection),
            1
        );
        let inspection_names = tool_names(&laguna_prompt_tools(inspection_phase, &tools)).unwrap();
        assert!(inspection_names.contains("exec_command"));
        assert!(inspection_names.contains("write_stdin"));
        assert!(inspection_names.contains("apply_patch"));
        let prompt_input =
            laguna_prompt_input(&after_inspection, &inspection_names, inspection_phase, true);
        assert_eq!(prompt_input.len(), 2);
        assert_eq!(
            prompt_input[0]
                .get("type")
                .and_then(serde_json::Value::as_str),
            Some("function_call_output")
        );
        assert_eq!(
            prompt_input[1]
                .get("role")
                .and_then(serde_json::Value::as_str),
            Some("user")
        );

        let mut after_edit = after_inspection;
        after_edit.push(json!({
            "type": "custom_tool_call",
            "name": "apply_patch",
            "input": "*** Begin Patch\n*** End Patch\n",
            "call_id": "call_2"
        }));
        after_edit.push(json!({
            "type": "custom_tool_call_output",
            "call_id": "call_2",
            "output": "Done!"
        }));
        let edit_phase = laguna_tool_phase(&after_edit);
        assert_eq!(edit_phase, LagunaToolPhase::Validate);
        assert_eq!(
            historical_read_only_shell_calls_since_last_mutation(&after_edit),
            0
        );
        let edit_names = tool_names(&laguna_prompt_tools(edit_phase, &tools)).unwrap();
        assert!(edit_names.contains("exec_command"));

        let mut after_failed_edit = after_edit.clone();
        after_failed_edit.last_mut().unwrap()["output"] =
            json!("apply_patch verification failed: invalid hunk");
        let failed_edit_phase = laguna_tool_phase(&after_failed_edit);
        assert_eq!(failed_edit_phase, LagunaToolPhase::Edit);
        let failed_edit_names =
            tool_names(&laguna_prompt_tools(failed_edit_phase, &tools)).unwrap();
        assert!(failed_edit_names.contains("exec_command"));

        after_edit.push(json!({
            "type": "function_call",
            "name": "exec_command",
            "arguments": "{\"cmd\":\"cargo test\"}",
            "call_id": "call_3"
        }));
        after_edit.push(json!({
            "type": "function_call_output",
            "call_id": "call_3",
            "output": "Process exited with code 0\nOutput:\ntest result: ok"
        }));
        let complete_phase = laguna_tool_phase(&after_edit);
        assert_eq!(complete_phase, LagunaToolPhase::Complete);
        assert!(laguna_prompt_tools(complete_phase, &tools).is_empty());
    }

    #[test]
    fn distinguishes_repeated_inspection_from_shell_file_changes() {
        let inspection = AgentOutput {
            reasoning: String::new(),
            items: vec![AgentOutputItem::FunctionCall(AgentFunctionCall {
                name: "shell".to_string(),
                arguments: json!({"cmd": "find . -maxdepth 2 -type f"}).to_string(),
            })],
        };
        assert!(first_read_only_shell_call(&inspection).unwrap().is_some());

        let mutation = AgentOutput {
            reasoning: String::new(),
            items: vec![AgentOutputItem::FunctionCall(AgentFunctionCall {
                name: "shell".to_string(),
                arguments: json!({"cmd": "mkdir -p src"}).to_string(),
            })],
        };
        assert!(first_read_only_shell_call(&mutation).unwrap().is_none());
    }

    #[test]
    fn laguna_keeps_shell_when_apply_patch_is_not_available() {
        let tools = vec![json!({"type": "function", "name": "exec_command"})];
        let input = vec![json!({
            "type": "function_call",
            "name": "exec_command",
            "arguments": "{\"cmd\":\"ls\"}",
            "call_id": "call_1"
        })];

        let names = tool_names(&laguna_prompt_tools(laguna_tool_phase(&input), &tools)).unwrap();
        assert!(names.contains("exec_command"));
    }

    #[test]
    fn detects_generated_tools_outside_the_turn_contract() {
        let output = AgentOutput {
            reasoning: String::new(),
            items: vec![AgentOutputItem::FunctionCall(AgentFunctionCall {
                name: "shell".to_string(),
                arguments: json!({"cmd": "ls"}).to_string(),
            })],
        };

        let call =
            first_tool_outside_contract(&output, &HashSet::from(["apply_patch".to_string()]))
                .unwrap();
        assert_eq!(call.name, "exec_command");
    }

    #[test]
    fn permits_one_repeat_then_blocks_the_third_identical_shell_call() {
        let completed_once = historical_tool_call_fingerprints(&[json!({
            "type": "function_call",
            "name": "exec_command",
            "arguments": "{\"cmd\":\"ls -la\"}",
            "call_id": "call_1"
        })])
        .unwrap();
        let output = AgentOutput {
            reasoning: String::new(),
            items: vec![AgentOutputItem::FunctionCall(AgentFunctionCall {
                name: "shell".to_string(),
                arguments: "{\"cmd\":\"ls -la\"}".to_string(),
            })],
        };
        assert!(duplicate_tool_call(&output, &completed_once)
            .unwrap()
            .is_none());

        let completed_twice = historical_tool_call_fingerprints(&[
            json!({
                "type": "function_call",
                "name": "exec_command",
                "arguments": "{\"cmd\":\"ls -la\"}",
                "call_id": "call_1"
            }),
            json!({
                "type": "function_call",
                "name": "exec_command",
                "arguments": "{\"cmd\":\"ls -la\"}",
                "call_id": "call_2"
            }),
        ])
        .unwrap();
        let duplicate = duplicate_tool_call(&output, &completed_twice)
            .unwrap()
            .expect("expected duplicate");

        assert_eq!(duplicate.name, "exec_command");
        assert_eq!(duplicate.arguments, "{\"cmd\":\"ls -la\"}");
    }

    #[test]
    fn permits_new_tool_arguments_after_a_completed_call() {
        let completed = historical_tool_call_fingerprints(&[json!({
            "type": "function_call",
            "name": "exec_command",
            "arguments": "{\"cmd\":\"ls -la\"}",
            "call_id": "call_1"
        })])
        .unwrap();
        let output = AgentOutput {
            reasoning: String::new(),
            items: vec![AgentOutputItem::FunctionCall(AgentFunctionCall {
                name: "shell".to_string(),
                arguments: "{\"cmd\":\"cargo test\"}".to_string(),
            })],
        };

        assert!(duplicate_tool_call(&output, &completed).unwrap().is_none());
    }

    #[test]
    fn emits_laguna_apply_patch_as_a_responses_custom_tool_call() {
        let output = AgentOutput {
            reasoning: String::new(),
            items: vec![AgentOutputItem::FunctionCall(AgentFunctionCall {
                name: "apply_patch".to_string(),
                arguments: json!({"patch": "*** Begin Patch\n*** End Patch\n"}).to_string(),
            })],
        };
        let allowed_tools = HashSet::from(["apply_patch".to_string()]);
        let mut bytes = Vec::new();
        let mut stream = ResponsesStream::begin(&mut bytes, LAGUNA_GGUF_CODEX_MODEL_ID).unwrap();

        emit_laguna_agent_output(output, "", &allowed_tools, "Laguna", false, &mut stream).unwrap();

        let rendered = String::from_utf8(bytes).unwrap();
        assert!(rendered.contains("\"type\":\"custom_tool_call\""));
        assert!(rendered.contains("\"name\":\"apply_patch\""));
        assert!(rendered.contains("*** Begin Patch"));
    }

    #[test]
    fn normalizes_missing_add_file_prefixes_without_changing_valid_lines() {
        let malformed = "*** Begin Patch\n*** Add File: Cargo.toml\n[package]\n+name = \"merkle\"\n\n*** Add File: src/lib.rs\npub fn root() {}\n*** End Patch";

        assert_eq!(
            normalize_laguna_apply_patch(malformed),
            "*** Begin Patch\n*** Add File: Cargo.toml\n+[package]\n+name = \"merkle\"\n+\n*** Add File: src/lib.rs\n+pub fn root() {}\n*** End Patch"
        );
    }

    #[test]
    fn accepts_only_the_model_loaded_by_this_server_process() {
        validate_requested_model(LAGUNA_CODEX_MODEL_ID, LAGUNA_CODEX_MODEL_ID).unwrap();
        let error =
            validate_requested_model(GLM_CODEX_MODEL_ID, LAGUNA_CODEX_MODEL_ID).unwrap_err();
        assert!(error.to_string().contains(GLM_CODEX_MODEL_ID));
        assert!(error.to_string().contains(LAGUNA_CODEX_MODEL_ID));
    }

    #[test]
    fn gguf_codex_rejects_fallback_model_metadata() {
        let request = ResponsesRequest {
            model: LAGUNA_GGUF_CODEX_MODEL_ID.to_string(),
            instructions:
                "You are a coding agent running in the Codex CLI, a terminal-based assistant."
                    .to_string(),
            input: Vec::new(),
            tools: Vec::new(),
            reasoning: None,
            max_output_tokens: None,
            stream: true,
        };

        let error = validate_laguna_gguf_codex_envelope(&request).unwrap_err();

        assert!(error.to_string().contains("fallback metadata"));
        assert!(error.to_string().contains("inferno.models.json"));
    }

    #[test]
    fn gguf_codex_bounds_output_by_request_server_and_context() {
        assert_eq!(
            laguna_codex_max_new_tokens(LAGUNA_GGUF_CODEX_MODEL_ID, None, None, 1_000).unwrap(),
            Some(LAGUNA_GGUF_DEFAULT_MAX_OUTPUT_TOKENS)
        );
        assert_eq!(
            laguna_codex_max_new_tokens(LAGUNA_GGUF_CODEX_MODEL_ID, Some(512), Some(1_024), 1_000)
                .unwrap(),
            Some(512)
        );
        assert_eq!(
            laguna_codex_max_new_tokens(LAGUNA_GGUF_CODEX_MODEL_ID, Some(512), Some(128), 1_000)
                .unwrap(),
            Some(128)
        );
        assert_eq!(
            laguna_codex_max_new_tokens(LAGUNA_GGUF_CODEX_MODEL_ID, None, None, 32_672).unwrap(),
            Some(96)
        );
    }

    #[test]
    fn gguf_codex_rejects_prompts_outside_the_service_context() {
        let error = laguna_codex_max_new_tokens(
            LAGUNA_GGUF_CODEX_MODEL_ID,
            None,
            None,
            LAGUNA_GGUF_SERVICE_CONTEXT_TOKENS,
        )
        .unwrap_err();

        assert!(error.to_string().contains("compact"));
        assert!(error.to_string().contains("32768"));
    }

    #[test]
    fn safetensors_codex_keeps_its_existing_generation_limit() {
        assert_eq!(
            laguna_codex_max_new_tokens(LAGUNA_CODEX_MODEL_ID, Some(777), Some(12), 40_000)
                .unwrap(),
            Some(777)
        );
    }

    #[test]
    fn completes_streamed_laguna_text_before_its_function_call() {
        let output = parse_agent_output(
            "reasoning</think>Creating the file.<tool_call>exec_command<arg_key>cmd</arg_key><arg_value>pwd</arg_value></tool_call>",
        )
        .unwrap();
        let allowed_tools = HashSet::from(["exec_command".to_string()]);
        let mut bytes = Vec::new();
        let mut stream = ResponsesStream::begin(&mut bytes, LAGUNA_CODEX_MODEL_ID).unwrap();
        stream.text_delta("Creating the file.").unwrap();

        emit_laguna_agent_output(
            output,
            "Creating the file.",
            &allowed_tools,
            "Laguna",
            false,
            &mut stream,
        )
        .unwrap();
        drop(stream);

        let rendered = String::from_utf8(bytes).unwrap();
        assert_eq!(
            rendered
                .matches("event: response.output_text.delta")
                .count(),
            1
        );
        let message_done = rendered
            .find("\"type\":\"message\"")
            .expect("expected completed text message");
        let function_done = rendered
            .find("\"type\":\"function_call\"")
            .expect("expected completed function call");
        assert!(message_done < function_done);
    }

    #[test]
    fn direct_laguna_output_keeps_text_and_tool_parsing_available() {
        let generated = format!(
            "{CLOSED_THINKING_BOUNDARY}Creating the file.<tool_call>exec_command<arg_key>cmd</arg_key><arg_value>pwd</arg_value></tool_call>"
        );

        assert_eq!(
            streamable_agent_text(&generated),
            Some("Creating the file.")
        );
        let output = parse_complete_agent_tool_call(&generated)
            .unwrap()
            .expect("expected a complete tool call");
        assert!(output.reasoning.is_empty());
        assert_eq!(
            output.items.first(),
            Some(&AgentOutputItem::Text("Creating the file.".to_string()))
        );
        assert!(matches!(
            output.items.get(1),
            Some(AgentOutputItem::FunctionCall(call)) if call.name == "exec_command"
        ));
    }

    #[test]
    fn emits_reasoning_only_for_the_xs_preserved_reasoning_path() {
        let output = parse_agent_output(
            "Inspect the workspace first.</think><tool_call>exec_command<arg_key>cmd</arg_key><arg_value>pwd</arg_value></tool_call>",
        )
        .unwrap();
        let allowed_tools = HashSet::from(["exec_command".to_string()]);
        let mut bytes = Vec::new();
        let mut stream = ResponsesStream::begin(&mut bytes, LAGUNA_XS_GGUF_CODEX_MODEL_ID).unwrap();

        emit_laguna_agent_output(output, "", &allowed_tools, "Laguna XS", true, &mut stream)
            .unwrap();
        drop(stream);

        let rendered = String::from_utf8(bytes).unwrap();
        assert!(rendered.contains("\"type\":\"reasoning\""));
        assert!(rendered.contains("Inspect the workspace first."));
        assert!(rendered.contains("\"type\":\"function_call\""));
    }
}
