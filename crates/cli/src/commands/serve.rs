use std::{
    collections::HashSet,
    net::{SocketAddr, TcpListener},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

use anyhow::Result;
use backend::{Backend, MetalBackend};
use common::{Error, Result as InfernoResult};
use config::{load_config, load_generation_config, Config};
use gguf::GgufFile;
use inferno_io::EXPERT_PACK_FILE_NAME;
use model::{
    expected_expert_pack_header, validate_routing_policy, Model, DEFAULT_GGUF_OUTPUT_CHUNK_ROWS,
};
use runtime::{
    enable_memory_controller_log, enable_memory_telemetry, enable_memory_telemetry_file,
    q2_memory_controller_spec, run_generate_streaming_with_options, CacheBudgetSpec,
    GenerationOptions,
};
use server::{ResponseUsage, ResponsesHandler, ResponsesRequest, ResponsesStream, CODEX_MODEL_ID};
use tokenizer::{
    is_supported_codex_function, parse_agent_output, render_codex_prompt, AgentOutputItem,
    Tokenizer,
};
use tracing::info;

use super::generate::{
    cache_gb_to_bytes, discover_config_path, discover_tokenizer_path, expert_cache_slots_per_layer,
    load_q2_readiness, resolve_q2_artifact, validate_generation_request,
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
    validate_memory_controller_options(enable_unified_memory_controller, memory_controller_log)?;
    let discovered_config = discover_config_path(model_path, config_path)?;
    let discovered_tokenizer = discover_tokenizer_path(model_path, tokenizer_path)?;
    let generation_config = load_generation_config(&model_path.join("generation_config.json"))?;
    let config = load_config(&discovered_config)?;
    let artifact = resolve_q2_artifact(model_path)?;
    let gguf = GgufFile::open(&artifact.gguf_path)?;
    validate_routing_policy(&config, &gguf)?;
    let readiness = load_q2_readiness(&gguf, &artifact, &config)?;
    let tokenizer = Tokenizer::from_file(&discovered_tokenizer)?;

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
    let mut handler = CodexHandler {
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
    info!(%bind, model = CODEX_MODEL_ID, "Inferno model loaded for Codex");
    server::serve(listener, &mut handler)?;
    Ok(())
}

struct CodexHandler<'runtime, 'weights> {
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

impl ResponsesHandler for CodexHandler<'_, '_> {
    fn generate(
        &mut self,
        request: ResponsesRequest,
        stream: &mut ResponsesStream<'_>,
    ) -> InfernoResult<ResponseUsage> {
        if request.model != CODEX_MODEL_ID {
            return Err(Error::runtime(format!(
                "Inferno serves model {CODEX_MODEL_ID:?}, got {:?}",
                request.model
            )));
        }
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

        let output = parse_agent_output(&generated_text)?;
        for item in output.items {
            match item {
                AgentOutputItem::Text(text) => {
                    stream.text_delta(&text)?;
                    stream.message_done(&text)?;
                }
                AgentOutputItem::FunctionCall(call) => {
                    if !allowed_tools.contains(&call.name) {
                        return Err(Error::runtime(format!(
                            "GLM requested undeclared Codex tool {:?}",
                            call.name
                        )));
                    }
                    let call_id = format!(
                        "call_inferno_{}",
                        NEXT_CALL_ID.fetch_add(1, Ordering::Relaxed)
                    );
                    stream.function_call_done(&call_id, &call.name, &call.arguments)?;
                }
            }
        }

        Ok(ResponseUsage {
            input_tokens: encoded.token_ids.len(),
            output_tokens,
        })
    }
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
}
