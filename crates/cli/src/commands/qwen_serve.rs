use crate::commands::{
    generate::DecodedTextStream,
    qwen::{self, QwenOptions},
    throughput::LiveThroughputDisplay,
};
use backend::MetalBackend;
use common::{Error, Result};
use config::load_generation_config;
use runtime::QwenRuntime;
use server::{
    ResponseUsage, ResponsesHandler, ResponsesRequest, ResponsesStream, ServerAdmission,
    QWEN_CODEX_MODEL_ID,
};
use std::{
    cell::RefCell,
    net::{SocketAddr, TcpListener},
    path::Path,
};
use tokenizer::{parse_qwen_tool_calls, render_qwen_responses_prompt, Tokenizer};

pub(super) fn run(
    options: &QwenOptions,
    model: &Path,
    config: &Path,
    tokenizer_path: &Path,
    bind: SocketAddr,
    max_new_tokens: Option<usize>,
    thinking: bool,
    throughput_summary: bool,
) -> anyhow::Result<()> {
    let tokenizer = qwen::load_tokenizer(tokenizer_path)?;
    let generation = load_generation_config(&model.join("generation_config.json"))?;
    let backend = MetalBackend::new()?;
    let runtime = options.open(model, config, &backend)?;
    let listener = TcpListener::bind(bind).map_err(|source| Error::Io {
        path: "inferno-listener".into(),
        source,
    })?;
    let mut handler = QwenHandler {
        runtime,
        tokenizer,
        eos: generation.eos_token_ids,
        max_new_tokens,
        context_tokens: options.context_tokens,
        thinking,
        throughput_summary,
    };
    server::serve(listener, &mut handler, QWEN_CODEX_MODEL_ID)?;
    Ok(())
}

struct QwenHandler<'a> {
    runtime: QwenRuntime<'a, MetalBackend>,
    tokenizer: Tokenizer,
    eos: Vec<u32>,
    max_new_tokens: Option<usize>,
    context_tokens: usize,
    thinking: bool,
    throughput_summary: bool,
}

impl ResponsesHandler for QwenHandler<'_> {
    fn context_window(&self) -> Option<usize> {
        Some(self.context_tokens)
    }

    fn admission(&self) -> ServerAdmission {
        ServerAdmission::RejectWhenBusy
    }

    fn generate(
        &mut self,
        request: ResponsesRequest,
        stream: &mut ResponsesStream<'_>,
    ) -> Result<ResponseUsage> {
        super::validate_requested_model(&request.model, QWEN_CODEX_MODEL_ID)?;
        let thinking = match request
            .reasoning
            .as_ref()
            .and_then(|v| v.get("effort"))
            .and_then(|v| v.as_str())
        {
            None => self.thinking,
            Some("none") => false,
            Some("xhigh") => true,
            Some(other) => {
                return Err(Error::runtime(format!(
                    "Qwen supports reasoning effort none or xhigh, got {other:?}"
                )))
            }
        };
        let prompt = render_qwen_responses_prompt(
            &request.instructions,
            &request.input,
            &request.tools,
            thinking,
        )?;
        let ids = self.tokenizer.encode(&prompt.rendered, false)?.token_ids;
        let available = self
            .context_tokens
            .checked_sub(ids.len())
            .filter(|count| *count > 0)
            .ok_or_else(|| {
                Error::runtime("Qwen context limit reached; compact the conversation")
            })?;
        let limit = request
            .max_output_tokens
            .unwrap_or(available)
            .min(self.max_new_tokens.unwrap_or(available))
            .min(available);
        self.runtime.reset_sequence()?;
        let mut decoded = DecodedTextStream::new(&self.tokenizer, true);
        let mut output = OutputStream::new(thinking);
        let mut throughput = LiveThroughputDisplay::new(self.throughput_summary);
        throughput.begin_turn(ids.len())?;
        let stream = RefCell::new(stream);
        let throughput = RefCell::new(throughput);
        let report = self.runtime.generate_with_prefill_progress(
            &ids,
            Some(limit),
            &self.eos,
            |processed| {
                stream.borrow_mut().heartbeat()?;
                throughput.borrow_mut().record_prefill(processed)
            },
            |id| {
                let mut stream = stream.borrow_mut();
                stream.heartbeat()?;
                throughput.borrow_mut().record_token()?;
                if !self.eos.contains(&id) {
                    if let Some(text) = decoded.push(id)? {
                        output.push(&text, &mut stream)?;
                    }
                }
                Ok(())
            },
        )?;
        throughput.borrow_mut().finish_request()?;
        let mut stream = stream.borrow_mut();
        output.finish(&request.tools, report.stopped_on_eos, &mut stream)?;
        if !report.stopped_on_eos {
            stream.mark_output_limit_reached();
        }
        Ok(ResponseUsage {
            input_tokens: ids.len(),
            output_tokens: report.generated_tokens,
        })
    }
}

/// Withholds XML and partial tag prefixes; only visible answer text reaches text deltas.
struct OutputStream {
    generated: String,
    thinking: bool,
    reasoning_sent: bool,
    sent_text: usize,
}

impl OutputStream {
    fn new(thinking: bool) -> Self {
        Self {
            generated: String::new(),
            thinking,
            reasoning_sent: false,
            sent_text: 0,
        }
    }

    fn visible(&self) -> Option<&str> {
        if self.thinking {
            self.generated
                .split_once("</think>")
                .map(|(_, visible)| visible)
        } else {
            Some(&self.generated)
        }
    }

    fn push(&mut self, delta: &str, stream: &mut ResponsesStream<'_>) -> Result<()> {
        self.generated.push_str(delta);
        if self.thinking && !self.reasoning_sent {
            if let Some((reasoning, _)) = self.generated.split_once("</think>") {
                stream.reasoning_done(reasoning.trim())?;
                self.reasoning_sent = true;
            }
        }
        let Some(visible) = self.visible() else {
            return Ok(());
        };
        let end = match visible.find("<tool_call>") {
            Some(position) => position,
            None => stable_prefix(visible, "<tool_call>"),
        };
        if end > self.sent_text {
            stream.text_delta(&visible[self.sent_text..end])?;
            self.sent_text = end;
        }
        Ok(())
    }

    fn finish(
        &mut self,
        tools: &[serde_json::Value],
        stopped_on_eos: bool,
        stream: &mut ResponsesStream<'_>,
    ) -> Result<()> {
        let Some(visible) = self.visible() else {
            if stopped_on_eos {
                return Err(Error::tokenizer(
                    "Qwen ended before closing its reasoning block",
                ));
            }
            // A generation cap can end in reasoning without an answer.
            stream.reasoning_done(self.generated.trim())?;
            return Ok(());
        };
        let split = visible.find("<tool_call>").unwrap_or_else(|| {
            if stopped_on_eos {
                visible.len()
            } else {
                stable_prefix(visible, "<tool_call>")
            }
        });
        let text = &visible[..split];
        // Never dispatch tool calls from a truncated generation, even if one looks complete.
        let calls = if stopped_on_eos {
            parse_qwen_tool_calls(&visible[split..], tools)?
        } else {
            Vec::new()
        };
        if !text.is_empty() {
            if text.len() > self.sent_text {
                stream.text_delta(&text[self.sent_text..])?;
            }
            stream.message_done(text)?;
        }
        for (index, call) in calls.into_iter().enumerate() {
            let id = format!("call_{}_{}", stream.response_id(), index);
            stream.function_call_done(&id, &call.name, &call.arguments)?;
        }
        Ok(())
    }
}

fn stable_prefix(text: &str, marker: &str) -> usize {
    for len in (1..marker.len()).rev() {
        if text.ends_with(&marker[..len]) {
            return text.len() - len;
        }
    }
    text.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn response_events(bytes: &[u8]) -> Vec<serde_json::Value> {
        std::str::from_utf8(bytes)
            .unwrap()
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|data| serde_json::from_str(data).unwrap())
            .collect()
    }

    #[test]
    #[ignore = "requires native Metal and local Qwen Q4, MTP and DFlash2 weights"]
    fn real_qwen_responses_text_tools_and_request_isolation() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let model = workspace.join("models/qwen3.8-27b-4bit");
        let draft = workspace.join("models/qwen3.8-27b-dflash2");
        let backend = MetalBackend::new().unwrap();
        for dflash in [false, true] {
            let options = QwenOptions {
                dflash_model: dflash.then(|| draft.clone()),
                ..QwenOptions::default()
            };
            let runtime = options
                .open(&model, &model.join("config.json"), &backend)
                .unwrap();
            let tokenizer = qwen::load_tokenizer(&model.join("tokenizer.json")).unwrap();
            let generation = load_generation_config(&model.join("generation_config.json")).unwrap();
            let mut handler = QwenHandler {
                runtime,
                tokenizer,
                eos: generation.eos_token_ids,
                max_new_tokens: Some(128),
                context_tokens: options.context_tokens,
                thinking: false,
                throughput_summary: false,
            };
            let mut run = |input: Vec<serde_json::Value>,
                           tools: Vec<serde_json::Value>,
                           instructions: &str| {
                let request = ResponsesRequest::parse(
                    &json!({
                        "model":QWEN_CODEX_MODEL_ID,"stream":true,"input":input,
                        "tools":tools,"instructions":instructions
                    })
                    .to_string(),
                )
                .unwrap();
                let mut bytes = Vec::new();
                let mut stream = ResponsesStream::begin(&mut bytes, QWEN_CODEX_MODEL_ID).unwrap();
                let usage = handler.generate(request, &mut stream).unwrap();
                stream.completed(usage).unwrap();
                let events = response_events(&bytes);
                assert_eq!(events.last().unwrap()["type"], "response.completed");
                events
                    .into_iter()
                    .filter(|event| event["type"] == "response.output_item.done")
                    .map(|event| event["item"].clone())
                    .collect::<Vec<_>>()
            };
            let first = run(
                vec![
                    json!({"role":"user","content":"Tell me the capital of Italy. Answer in one word."}),
                ],
                vec![],
                "",
            );
            assert!(first.iter().any(|item| item["content"][0]["text"]
                .as_str()
                .is_some_and(|text| text.contains("Rome"))));
            let second = run(
                vec![json!({"role":"user","content":"What is 2 + 2? Answer in one digit."})],
                vec![],
                "",
            );
            assert!(second.iter().any(|item| item["content"][0]["text"]
                .as_str()
                .is_some_and(|text| text.contains('4'))));

            let tools = vec![
                json!({"type":"function","name":"echo","description":"Echo the supplied text.","parameters":{
                    "type":"object","properties":{"text":{"type":"string"}},"required":["text"],"additionalProperties":false
                }}),
            ];
            let user = json!({"role":"user","content":"Call echo with text exactly ready. After receiving its result, reply done."});
            let items = run(
                vec![user.clone()],
                tools.clone(),
                "Use the requested function before answering.",
            );
            let call = items
                .iter()
                .find(|item| item["type"] == "function_call")
                .expect("a real native Qwen function call");
            assert_eq!(call["name"], "echo");
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(call["arguments"].as_str().unwrap())
                    .unwrap(),
                json!({"text":"ready"})
            );
            let mut history = vec![user];
            history.extend(items.clone());
            history.push(
                json!({"type":"function_call_output","call_id":call["call_id"],"output":"ready"}),
            );
            let answer = run(
                history,
                tools,
                "Use the requested function before answering.",
            );
            assert!(answer.iter().any(|item| item["type"] == "message"));
        }
    }

    fn render(parts: &[&str], thinking: bool, eos: bool) -> Result<String> {
        let mut bytes = Vec::new();
        let mut stream = ResponsesStream::begin(&mut bytes, QWEN_CODEX_MODEL_ID)?;
        let mut output = OutputStream::new(thinking);
        for part in parts {
            output.push(part, &mut stream)?;
        }
        output.finish(&[json!({"type":"function","name":"run","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}})], eos, &mut stream)?;
        if !eos {
            stream.mark_output_limit_reached();
        }
        stream.completed(ResponseUsage::default())?;
        Ok(String::from_utf8(bytes).unwrap())
    }

    #[test]
    fn qwen_stream_withholds_split_tool_markers() {
        let body = render(&["Ready. <too", "l_call>\n<function=run>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>"], false, true).unwrap();
        assert!(body.contains("\"type\":\"function_call\""));
        assert!(body.contains("\"delta\":\"Ready. \""));
        assert!(!body.contains("<tool_call>"));
        assert!(body.contains("event: response.completed"));
    }

    #[test]
    fn qwen_stream_separates_reasoning_and_answer() {
        let body = render(&["Consider it.</thi", "nk>Rome"], true, true).unwrap();
        assert!(body.contains("\"type\":\"reasoning\""));
        assert!(body.contains("\"delta\":\"Rome\""));
        assert!(!body.contains("\"delta\":\"Consider"));
    }

    #[test]
    fn qwen_stream_never_dispatches_truncated_tools() {
        for tail in [
            "<tool_c",
            "<tool_call><function=run><parameter=cmd>ls",
            "<tool_call><function=run><parameter=cmd>ls</parameter></function></tool_call>",
        ] {
            let body = render(&["Before. ", tail], false, false).unwrap();
            assert!(body.contains("event: response.incomplete"));
            assert!(!body.contains("\"type\":\"function_call\""));
            assert!(!body.contains("\"delta\":\"<tool"));
        }
    }

    #[test]
    fn qwen_stream_rejects_unclosed_reasoning_at_eos() {
        assert!(render(&["Thinking"], true, true).is_err());
        assert!(render(&["Thinking"], true, false)
            .unwrap()
            .contains("response.incomplete"));
    }
}
