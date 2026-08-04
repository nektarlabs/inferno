use std::{
    io::Write,
    sync::atomic::{AtomicU64, Ordering},
};

use common::{Error, Result};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::http::write_sse_headers;

static NEXT_RESPONSE_ID: AtomicU64 = AtomicU64::new(1);

/// The subset of a Codex Responses request consumed by Inferno.
///
/// Input items and tool specifications stay as JSON because Codex can add new
/// item variants without changing the core Responses envelope. The loaded
/// model adapter validates and translates only the item kinds needed for a
/// coding turn.
#[derive(Debug, Clone, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    #[serde(default)]
    pub instructions: String,
    #[serde(default)]
    pub input: Vec<Value>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub reasoning: Option<Value>,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
}

impl ResponsesRequest {
    pub fn parse(body: &str) -> Result<Self> {
        let request: Self = serde_json::from_str(body)?;
        if request.model.trim().is_empty() {
            return Err(Error::runtime("Responses request model is empty"));
        }
        if !request.stream {
            return Err(Error::runtime(
                "Inferno requires stream=true for Codex Responses requests",
            ));
        }
        if request.max_output_tokens == Some(0) {
            return Err(Error::runtime(
                "Responses max_output_tokens must be positive when provided",
            ));
        }
        Ok(request)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResponseUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

/// Writes the exact SSE event subset consumed by Codex.
pub struct ResponsesStream<'a> {
    writer: &'a mut dyn Write,
    response_id: String,
    model: String,
    output_index: usize,
    active_message_id: Option<String>,
}

impl<'a> ResponsesStream<'a> {
    pub fn begin(writer: &'a mut dyn Write, model: &str) -> Result<Self> {
        write_sse_headers(writer)?;
        let response_id = format!(
            "resp_inferno_{}",
            NEXT_RESPONSE_ID.fetch_add(1, Ordering::Relaxed)
        );
        let mut stream = Self {
            writer,
            response_id,
            model: model.to_string(),
            output_index: 0,
            active_message_id: None,
        };
        stream.event(
            "response.created",
            json!({
                "type": "response.created",
                "response": {
                    "id": stream.response_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": stream.model,
                    "output": []
                }
            }),
        )?;
        Ok(stream)
    }

    pub fn response_id(&self) -> &str {
        &self.response_id
    }

    /// Keeps a long local inference request alive and detects a disconnected
    /// Codex client at generated-token boundaries.
    pub fn heartbeat(&mut self) -> Result<()> {
        self.writer
            .write_all(b": inferno\n\n")
            .and_then(|()| self.writer.flush())
            .map_err(|source| Error::Io {
                path: "inferno-sse-response".into(),
                source,
            })
    }

    pub fn text_delta(&mut self, delta: &str) -> Result<()> {
        if delta.is_empty() {
            return Ok(());
        }
        let item_id = self.ensure_message_started()?;
        self.event(
            "response.output_text.delta",
            json!({
                "type": "response.output_text.delta",
                "response_id": self.response_id,
                "output_index": self.output_index,
                "content_index": 0,
                "item_id": item_id,
                "delta": delta
            }),
        )
    }

    pub fn message_done(&mut self, text: &str) -> Result<()> {
        let item_id = self.ensure_message_started()?;
        self.event(
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "response_id": self.response_id,
                "output_index": self.output_index,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]
                }
            }),
        )?;
        self.active_message_id = None;
        self.output_index += 1;
        Ok(())
    }

    /// Emits one completed raw reasoning item so a Responses client can replay
    /// it on the next turn without displaying it as assistant text.
    pub fn reasoning_done(&mut self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let item_id = format!("rs_{}_{}", self.response_id, self.output_index);
        self.event(
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "response_id": self.response_id,
                "output_index": self.output_index,
                "item": {
                    "id": item_id,
                    "type": "reasoning",
                    "status": "in_progress",
                    "summary": [],
                    "content": []
                }
            }),
        )?;
        self.event(
            "response.reasoning_text.done",
            json!({
                "type": "response.reasoning_text.done",
                "response_id": self.response_id,
                "output_index": self.output_index,
                "item_id": item_id,
                "content_index": 0,
                "text": text
            }),
        )?;
        self.event(
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "response_id": self.response_id,
                "output_index": self.output_index,
                "item": {
                    "id": item_id,
                    "type": "reasoning",
                    "status": "completed",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": text}]
                }
            }),
        )?;
        self.output_index += 1;
        Ok(())
    }

    pub fn function_call_done(&mut self, call_id: &str, name: &str, arguments: &str) -> Result<()> {
        if call_id.is_empty() || name.is_empty() {
            return Err(Error::runtime(
                "Responses function call requires non-empty call_id and name",
            ));
        }
        let item_id = format!("fc_{}_{}", self.response_id, self.output_index);
        self.event(
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "response_id": self.response_id,
                "output_index": self.output_index,
                "item": {
                    "id": item_id,
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments
                }
            }),
        )?;
        self.output_index += 1;
        Ok(())
    }

    pub fn custom_tool_call_done(&mut self, call_id: &str, name: &str, input: &str) -> Result<()> {
        if call_id.is_empty() || name.is_empty() {
            return Err(Error::runtime(
                "Responses custom tool call requires non-empty call_id and name",
            ));
        }
        let item_id = format!("ctc_{}_{}", self.response_id, self.output_index);
        self.event(
            "response.output_item.done",
            json!({
                "type": "response.output_item.done",
                "response_id": self.response_id,
                "output_index": self.output_index,
                "item": {
                    "id": item_id,
                    "type": "custom_tool_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "input": input
                }
            }),
        )?;
        self.output_index += 1;
        Ok(())
    }

    pub fn completed(&mut self, usage: ResponseUsage) -> Result<()> {
        let total_tokens = usage
            .input_tokens
            .checked_add(usage.output_tokens)
            .ok_or_else(|| Error::runtime("Responses token usage overflow"))?;
        self.event(
            "response.completed",
            json!({
                "type": "response.completed",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "status": "completed",
                    "model": self.model,
                    "end_turn": true,
                    "usage": {
                        "input_tokens": usage.input_tokens,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": usage.output_tokens,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": total_tokens
                    }
                }
            }),
        )
    }

    pub fn failed(&mut self, message: &str) -> Result<()> {
        self.event(
            "response.failed",
            json!({
                "type": "response.failed",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "status": "failed",
                    "error": {
                        "type": "server_error",
                        "code": "inferno_generation_failed",
                        "message": message
                    }
                }
            }),
        )
    }

    fn ensure_message_started(&mut self) -> Result<String> {
        if let Some(item_id) = &self.active_message_id {
            return Ok(item_id.clone());
        }
        let item_id = format!("msg_{}_{}", self.response_id, self.output_index);
        self.event(
            "response.output_item.added",
            json!({
                "type": "response.output_item.added",
                "response_id": self.response_id,
                "output_index": self.output_index,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [{"type": "output_text", "text": "", "annotations": []}]
                }
            }),
        )?;
        self.active_message_id = Some(item_id.clone());
        Ok(item_id)
    }

    fn event(&mut self, kind: &str, value: Value) -> Result<()> {
        let data = serde_json::to_string(&value)?;
        write!(self.writer, "event: {kind}\ndata: {data}\n\n")
            .and_then(|()| self.writer.flush())
            .map_err(|source| Error::Io {
                path: "inferno-sse-response".into(),
                source,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_codex_request_envelope_and_ignores_new_fields() {
        let request = ResponsesRequest::parse(
            &json!({
                "model": "glm-5.2-q2",
                "instructions": "Use tools.",
                "input": [{"type": "message", "role": "user", "content": []}],
                "tools": [{"type": "function", "name": "exec_command"}],
                "reasoning": {"effort": "high"},
                "stream": true,
                "max_output_tokens": 64,
                "store": false,
                "parallel_tool_calls": false,
                "include": []
            })
            .to_string(),
        )
        .unwrap();

        assert_eq!(request.model, "glm-5.2-q2");
        assert_eq!(request.input.len(), 1);
        assert_eq!(request.tools.len(), 1);
        assert_eq!(request.reasoning, Some(json!({"effort": "high"})));
        assert_eq!(request.max_output_tokens, Some(64));
    }

    #[test]
    fn rejects_non_streaming_requests() {
        let error = ResponsesRequest::parse(
            &json!({"model": "glm-5.2-q2", "input": [], "stream": false}).to_string(),
        )
        .unwrap_err();

        assert!(error.to_string().contains("stream=true"));
    }

    #[test]
    fn rejects_zero_max_output_tokens() {
        let error = ResponsesRequest::parse(
            &json!({
                "model": "glm-5.2-q2",
                "input": [],
                "stream": true,
                "max_output_tokens": 0
            })
            .to_string(),
        )
        .unwrap_err();

        assert!(error.to_string().contains("must be positive"));
    }

    #[test]
    fn emits_codex_text_and_function_call_items() {
        let mut bytes = Vec::new();
        let mut stream = ResponsesStream::begin(&mut bytes, "glm-5.2-q2").unwrap();
        stream.text_delta("done").unwrap();
        stream.message_done("done").unwrap();
        stream
            .function_call_done("call_1", "exec_command", r#"{"cmd":"pwd"}"#)
            .unwrap();
        stream
            .custom_tool_call_done("call_2", "apply_patch", "*** Begin Patch\n*** End Patch\n")
            .unwrap();
        stream
            .completed(ResponseUsage {
                input_tokens: 10,
                output_tokens: 2,
            })
            .unwrap();
        let rendered = String::from_utf8(bytes).unwrap();

        assert!(rendered.contains("HTTP/1.1 200 OK"));
        assert!(rendered.contains("response.output_text.delta"));
        assert!(rendered.contains("response.output_item.added"));
        assert!(rendered.contains("\"type\":\"message\""));
        assert!(rendered.contains("\"type\":\"function_call\""));
        assert!(rendered.contains("\"type\":\"custom_tool_call\""));
        assert!(rendered.contains("\"name\":\"apply_patch\""));
        assert!(rendered.contains("\\\"cmd\\\":\\\"pwd\\\""));
        assert!(rendered.contains("\"total_tokens\":12"));
    }

    #[test]
    fn emits_completed_reasoning_for_client_replay() {
        let mut bytes = Vec::new();
        let mut stream = ResponsesStream::begin(&mut bytes, "laguna-xs-2.1-gguf").unwrap();
        stream.reasoning_done("Inspect the repository.").unwrap();
        let rendered = String::from_utf8(bytes).unwrap();

        assert!(rendered.contains("event: response.reasoning_text.done"));
        assert!(rendered.contains("\"type\":\"reasoning\""));
        assert!(rendered.contains("\"type\":\"reasoning_text\""));
        assert!(rendered.contains("Inspect the repository."));
        assert!(rendered.contains("\"output_index\":0"));
    }
}
