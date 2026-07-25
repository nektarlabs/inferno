use std::{
    collections::HashSet,
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
    CacheBudgetSpec, GenerationControl, GenerationOptions, LagunaRuntime,
};
use server::{
    ResponseUsage, ResponsesHandler, ResponsesRequest, ResponsesStream, GLM_CODEX_MODEL_ID,
    LAGUNA_CODEX_MODEL_ID, LAGUNA_GGUF_CODEX_MODEL_ID,
};
use tokenizer::{
    is_supported_codex_function, parse_agent_output, parse_complete_agent_tool_call,
    render_codex_prompt, render_laguna_codex_prompt, streamable_agent_text, AgentOutput,
    AgentOutputItem, Tokenizer,
};
use tracing::info;

use super::generate::{
    cache_gb_to_bytes, discover_config_path, discover_tokenizer_path, expert_cache_slots_per_layer,
    laguna_runtime_options, load_q2_readiness, resolve_q2_artifact, validate_generation_request,
    validate_laguna_prompt, validate_laguna_service_options, validate_laguna_tokenizer,
    validate_memory_controller_options, DecodedTextStream,
};

static NEXT_CALL_ID: AtomicU64 = AtomicU64::new(1);

#[allow(clippy::too_many_arguments)]
pub fn run(
    model_path: &Path,
    config_path: Option<&Path>,
    tokenizer_path: Option<&Path>,
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
    let discovered_config = discover_config_path(model_path, config_path)?;
    let discovered_tokenizer = discover_tokenizer_path(model_path, tokenizer_path)?;
    match detect_model_architecture(&discovered_config)? {
        ModelArchitecture::GlmMoeDsa => run_glm(
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
        ),
        ModelArchitecture::Laguna => run_laguna(
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
    };

    let listener = TcpListener::bind(bind)?;
    let mut handler = LagunaCodexHandler {
        model: &model,
        config: &config,
        backend: &backend,
        tokenizer: &tokenizer,
        runtime,
        max_new_tokens,
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
    model_id: &'static str,
}

impl ResponsesHandler for LagunaCodexHandler<'_> {
    fn generate(
        &mut self,
        request: ResponsesRequest,
        stream: &mut ResponsesStream<'_>,
    ) -> InfernoResult<ResponseUsage> {
        validate_requested_model(&request.model, self.model_id)?;
        let allowed_tools = tool_names(&request.tools)?;
        let prompt =
            render_laguna_codex_prompt(&request.instructions, &request.input, &request.tools)?;
        let encoded = self.tokenizer.encode(&prompt.rendered, false)?;
        validate_laguna_prompt(self.config, &encoded.token_ids, self.max_new_tokens)?;

        let mut decoded = DecodedTextStream::new(self.tokenizer, true);
        let mut generated_text = String::new();
        let mut streamed_text = String::new();
        let mut completed_output = None;
        let mut output_tokens = 0_usize;
        self.runtime.generate_streaming_controlled(
            self.model,
            self.backend,
            &encoded.token_ids,
            self.max_new_tokens,
            &self.config.eos_token_id,
            |token_id| {
                output_tokens = output_tokens
                    .checked_add(1)
                    .ok_or_else(|| Error::runtime("Codex output token count overflow"))?;
                if let Some(text) = decoded.push(token_id)? {
                    generated_text.push_str(&text);
                }
                if let Some(visible_text) = streamable_agent_text(&generated_text) {
                    let delta = visible_text.strip_prefix(&streamed_text).ok_or_else(|| {
                        Error::tokenizer("Laguna agent visible text changed after it was streamed")
                    })?;
                    if !delta.is_empty() {
                        stream.text_delta(delta)?;
                        streamed_text.push_str(delta);
                    }
                }
                let control = match parse_complete_agent_tool_call(&generated_text)? {
                    Some(output) => {
                        completed_output = Some(output);
                        GenerationControl::Stop
                    }
                    None => GenerationControl::Continue,
                };
                stream.heartbeat()?;
                Ok(control)
            },
        )?;

        match completed_output {
            Some(output) => {
                emit_laguna_agent_output(output, &streamed_text, &allowed_tools, "Laguna", stream)?
            }
            None => {
                let output = parse_agent_output(&generated_text)?;
                emit_laguna_agent_output(output, &streamed_text, &allowed_tools, "Laguna", stream)?;
            }
        }
        Ok(ResponseUsage {
            input_tokens: encoded.token_ids.len(),
            output_tokens,
        })
    }
}

fn emit_laguna_agent_output(
    output: AgentOutput,
    streamed_text: &str,
    allowed_tools: &HashSet<String>,
    model_name: &str,
    stream: &mut ResponsesStream<'_>,
) -> InfernoResult<()> {
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
                emit_function_call(call, allowed_tools, model_name, stream)?;
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
    stream.function_call_done(&call_id, &call.name, &call.arguments)
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
        .filter(|tool| tool.get("type").and_then(serde_json::Value::as_str) == Some("function"))
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

    use super::*;

    #[test]
    fn extracts_only_declared_codex_tool_names() {
        let names = tool_names(&[
            json!({"type": "function", "name": "exec_command"}),
            json!({"type": "function", "name": "write_stdin"}),
            json!({"type": "function", "name": "update_plan"}),
            json!({"type": "namespace", "name": "plugin", "tools": []}),
            json!({"type": "web_search", "external_web_access": false}),
        ])
        .unwrap();

        assert_eq!(names.len(), 2);
        assert!(names.contains("exec_command"));
        assert!(names.contains("write_stdin"));
        assert!(!names.contains("update_plan"));
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
}
