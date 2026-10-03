use crate::{AgentFunctionCall, ChatPrompt};
use common::{Error, Result};
use serde_json::{json, Map, Value};

const TOOL_INSTRUCTIONS: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOL_FORMAT: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>\n\nRequired parameters MUST be specified. You may provide reasoning before a function call, but NOT after.";

/// Converts Responses function declarations to the checkpoint's tool schema.
fn qwen_function_tools(tools: &[Value]) -> Result<Vec<Value>> {
    let mut normalized = Vec::new();
    for tool in tools {
        match tool["type"].as_str() {
            Some("function") => {
                let name = required_string(tool, "name")?;
                validate_name(name)?;
                if normalized
                    .iter()
                    .any(|t: &Value| t["function"]["name"] == name)
                {
                    return Err(Error::tokenizer("duplicate Qwen function name"));
                }
                normalized.push(json!({"type":"function", "function": {
                    "name": name,
                    "description": tool["description"].as_str().unwrap_or(""),
                    "parameters": tool.get("parameters").cloned().unwrap_or_else(|| json!({"type":"object", "properties":{}}))
                }}));
            }
            Some("custom") => {
                return Err(Error::tokenizer(
                    "Qwen currently requires function tools, not custom tools",
                ))
            }
            other => {
                return Err(Error::tokenizer(format!(
                    "unsupported Qwen tool type {other:?}"
                )))
            }
        }
    }
    Ok(normalized)
}

/// Renders text and function/tool history with Qwen's native ChatML/XML grammar.
pub fn render_qwen_responses_prompt(
    instructions: &str,
    input: &[Value],
    tools: &[Value],
    thinking: bool,
) -> Result<ChatPrompt> {
    let tools = qwen_function_tools(tools)?;
    let mut system = String::new();
    if thinking {
        system.push_str("Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.");
    }
    if !tools.is_empty() {
        if !system.is_empty() {
            system.push_str("\n\n");
        }
        system.push_str(TOOL_INSTRUCTIONS);
        for tool in &tools {
            system.push('\n');
            system.push_str(&serde_json::to_string(tool)?);
        }
        system.push_str(TOOL_FORMAT);
    }
    if !instructions.trim().is_empty() {
        if !system.is_empty() {
            system.push_str("\n\n");
        }
        system.push_str(instructions.trim());
    }
    let mut turns: Vec<(String, String)> = Vec::new();
    let mut reasoning = String::new();
    let mut calls = std::collections::HashSet::new();
    for item in input {
        match item["type"].as_str().unwrap_or("message") {
            "message" => {
                let role = required_string(item, "role")?;
                let body = text_content(
                    item.get("content")
                        .ok_or_else(|| Error::tokenizer("Qwen message lacks content"))?,
                )?;
                match role {
                    "system" | "developer" if turns.is_empty() => {
                        if !system.is_empty() {
                            system.push_str("\n\n");
                        }
                        system.push_str(body.trim());
                    }
                    "user" => {
                        if !reasoning.is_empty() {
                            return Err(Error::tokenizer("Qwen reasoning has no assistant item"));
                        }
                        turns.push(("user".into(), body.trim().into()));
                    }
                    "assistant" => {
                        let content =
                            format!("<think>\n{}\n</think>\n\n{}", reasoning.trim(), body.trim());
                        reasoning.clear();
                        turns.push(("assistant".into(), content));
                    }
                    _ => {
                        return Err(Error::tokenizer(format!(
                            "unsupported Qwen message role or position {role:?}"
                        )))
                    }
                }
            }
            "reasoning" => {
                if let Some(content) = item.get("content") {
                    reasoning.push_str(&reasoning_content(content)?);
                }
            }
            "function_call" => {
                let call_id = required_string(item, "call_id")?;
                if !calls.insert(call_id.to_owned()) {
                    return Err(Error::tokenizer("duplicate Qwen call_id"));
                }
                let name = required_string(item, "name")?;
                validate_name(name)?;
                let arguments: Map<String, Value> =
                    serde_json::from_str(required_string(item, "arguments")?)?;
                let mut body = format!("<tool_call>\n<function={name}>\n");
                for (key, value) in arguments {
                    validate_name(&key)?;
                    let value = match value {
                        Value::String(text) => text,
                        other => other.to_string(),
                    };
                    body.push_str(&format!("<parameter={key}>\n{value}\n</parameter>\n"));
                }
                body.push_str("</function>\n</tool_call>");
                if let Some((role, previous)) =
                    turns.last_mut().filter(|(role, _)| role == "assistant")
                {
                    let _ = role;
                    previous.push_str("\n\n");
                    previous.push_str(&body);
                } else {
                    turns.push((
                        "assistant".into(),
                        format!("<think>\n{}\n</think>\n\n{body}", reasoning.trim()),
                    ));
                }
                reasoning.clear();
            }
            "function_call_output" => {
                let id = required_string(item, "call_id")?;
                if !calls.remove(id) {
                    return Err(Error::tokenizer("Qwen tool output has no matching call"));
                }
                let output = text_content(
                    item.get("output")
                        .ok_or_else(|| Error::tokenizer("Qwen tool output lacks output"))?,
                )?;
                let body = format!("<tool_response>\n{}\n</tool_response>", output.trim());
                if let Some((_, previous)) = turns
                    .last_mut()
                    .filter(|(role, body)| role == "user" && body.starts_with("<tool_response>"))
                {
                    previous.push('\n');
                    previous.push_str(&body);
                } else {
                    turns.push(("user".into(), body));
                }
            }
            other => {
                return Err(Error::tokenizer(format!(
                    "unsupported Qwen input item {other:?}"
                )))
            }
        }
    }
    if !reasoning.is_empty() || !calls.is_empty() {
        return Err(Error::tokenizer(
            "Qwen history has unfinished reasoning or tool calls",
        ));
    }
    if !turns
        .iter()
        .any(|(role, body)| role == "user" && !body.starts_with("<tool_response>"))
    {
        return Err(Error::tokenizer("Qwen requires a user message"));
    }
    let mut rendered = String::new();
    if !system.is_empty() {
        rendered.push_str(&format!("<|im_start|>system\n{system}<|im_end|>\n"));
    }
    for (role, body) in turns {
        rendered.push_str(&format!("<|im_start|>{role}\n{body}<|im_end|>\n"));
    }
    rendered.push_str("<|im_start|>assistant\n<think>\n");
    if !thinking {
        rendered.push_str("\n</think>\n\n");
    }
    Ok(ChatPrompt { rendered })
}

pub fn parse_qwen_tool_calls(text: &str, tools: &[Value]) -> Result<Vec<AgentFunctionCall>> {
    let schemas = qwen_function_tools(tools)?;
    let mut remaining = text.trim();
    let mut calls = Vec::new();
    while !remaining.is_empty() {
        let (body, tail) = take(remaining, "<tool_call>", "</tool_call>")?;
        let inner = body
            .trim()
            .strip_prefix("<function=")
            .ok_or_else(|| Error::tokenizer("Qwen tool call lacks function"))?;
        let (name, parameters) = inner
            .split_once('>')
            .ok_or_else(|| Error::tokenizer("Qwen function name is unterminated"))?;
        validate_name(name)?;
        let schema = schemas
            .iter()
            .find(|tool| tool["function"]["name"] == name)
            .ok_or_else(|| Error::tokenizer(format!("Qwen called undeclared function {name:?}")))?;
        let schema = &schema["function"]["parameters"];
        let mut parameters = parameters
            .trim()
            .strip_suffix("</function>")
            .ok_or_else(|| Error::tokenizer("Qwen function is unterminated"))?
            .trim();
        let mut arguments = Map::new();
        while !parameters.is_empty() {
            let rest = parameters
                .strip_prefix("<parameter=")
                .ok_or_else(|| Error::tokenizer("invalid Qwen parameter"))?;
            let (key, rest) = rest
                .split_once('>')
                .ok_or_else(|| Error::tokenizer("unterminated Qwen parameter name"))?;
            validate_name(key)?;
            let (value, tail) = rest
                .split_once("</parameter>")
                .ok_or_else(|| Error::tokenizer("unterminated Qwen parameter value"))?;
            if arguments.contains_key(key) {
                return Err(Error::tokenizer("duplicate Qwen parameter"));
            }
            let property = &schema["properties"][key];
            if property.is_null() && schema["additionalProperties"] == false {
                return Err(Error::tokenizer(format!(
                    "undeclared Qwen parameter {key:?}"
                )));
            }
            let kind = &property["type"];
            let value = value.strip_prefix('\n').unwrap_or(value);
            let value = value.strip_suffix('\n').unwrap_or(value);
            let parsed = if kind == "string"
                || kind
                    .as_array()
                    .is_some_and(|types| types.iter().any(|t| t == "string"))
            {
                Value::String(value.to_owned())
            } else {
                serde_json::from_str(value.trim())?
            };
            if !matches_parameter_type(&parsed, kind) {
                return Err(Error::tokenizer(format!(
                    "Qwen parameter {key:?} has the wrong JSON type"
                )));
            }
            arguments.insert(key.to_owned(), parsed);
            parameters = tail.trim();
        }
        if let Some(required) = schema["required"].as_array() {
            for key in required {
                if let Some(key) = key.as_str() {
                    if !arguments.contains_key(key) {
                        return Err(Error::tokenizer(format!("missing Qwen parameter {key:?}")));
                    }
                }
            }
        }
        calls.push(AgentFunctionCall {
            name: name.to_owned(),
            arguments: serde_json::to_string(&arguments)?,
        });
        remaining = tail.trim();
    }
    Ok(calls)
}

fn validate_name(name: &str) -> Result<()> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'_' | b'-' | b'.'))
    {
        return Err(Error::tokenizer("invalid Qwen tool or parameter name"));
    }
    Ok(())
}

// Preserve JSON types at the protocol boundary; the tool owner validates the full schema.
fn matches_parameter_type(value: &Value, kind: &Value) -> bool {
    if kind.is_null() {
        return true;
    }
    if let Some(types) = kind.as_array() {
        return types.iter().any(|kind| matches_parameter_type(value, kind));
    }
    match kind.as_str() {
        Some("string") => value.is_string(),
        Some("integer") => value.is_i64() || value.is_u64(),
        Some("number") => value.is_number(),
        Some("boolean") => value.is_boolean(),
        Some("object") => value.is_object(),
        Some("array") => value.is_array(),
        Some("null") => value.is_null(),
        _ => false,
    }
}

fn required_string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| Error::tokenizer(format!("Qwen item lacks {key:?}")))
}

fn text_content(value: &Value) -> Result<String> {
    if let Some(text) = value.as_str() {
        return Ok(text.into());
    }
    let parts = value
        .as_array()
        .ok_or_else(|| Error::tokenizer("Qwen supports text content only"))?;
    let mut text = String::new();
    for part in parts {
        if !matches!(
            part["type"].as_str(),
            Some("input_text" | "output_text" | "text")
        ) {
            return Err(Error::tokenizer("Qwen supports text content only"));
        }
        text.push_str(
            part["text"]
                .as_str()
                .ok_or_else(|| Error::tokenizer("Qwen text part lacks text"))?,
        );
    }
    Ok(text)
}

fn reasoning_content(value: &Value) -> Result<String> {
    let parts = value
        .as_array()
        .ok_or_else(|| Error::tokenizer("invalid Qwen reasoning content"))?;
    let mut text = String::new();
    for part in parts {
        if part["type"] != "reasoning_text" {
            return Err(Error::tokenizer("unsupported Qwen reasoning part"));
        }
        text.push_str(
            part["text"]
                .as_str()
                .ok_or_else(|| Error::tokenizer("Qwen reasoning part lacks text"))?,
        );
    }
    Ok(text)
}

fn take<'a>(value: &'a str, open: &str, close: &str) -> Result<(&'a str, &'a str)> {
    value
        .strip_prefix(open)
        .and_then(|rest| rest.split_once(close))
        .ok_or_else(|| Error::tokenizer(format!("incomplete Qwen {open} block")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<Value> {
        vec![
            json!({"type":"function", "name":"exec_command", "parameters":{
                "type":"object", "properties":{"cmd":{"type":"string"},"timeout":{"type":"integer"}},
                "required":["cmd"],"additionalProperties":false
            }}),
        ]
    }

    #[test]
    fn parses_native_function_arguments_without_converting_strings() {
        let text = "<tool_call>\n<function=exec_command>\n<parameter=cmd>\n123\n</parameter>\n<parameter=timeout>15</parameter>\n</function>\n</tool_call>";
        let calls = parse_qwen_tool_calls(text, &tools()).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "exec_command");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args, json!({"cmd":"123","timeout":15}));
    }

    #[test]
    fn rejects_invalid_or_incomplete_calls() {
        for body in [
            "<function=unknown></function>",
            "<function=exec_command></function>",
            "<function=exec_command><parameter=other>x</parameter></function>",
            "<function=exec_command><parameter=cmd>a</parameter><parameter=cmd>b</parameter></function>",
            "<function=exec_command><parameter=cmd>unfinished",
            "<function=exec_command><parameter=cmd>ls</parameter><parameter=timeout>\"wrong\"</parameter></function>",
        ] {
            assert!(parse_qwen_tool_calls(&format!("<tool_call>{body}</tool_call>"), &tools()).is_err(), "{body}");
        }
        assert!(parse_qwen_tool_calls("<tool_call>", &tools()).is_err());
    }

    #[test]
    fn renders_function_result_history_in_native_format() {
        let prompt = render_qwen_responses_prompt("Use tools.", &[
            json!({"role":"user","content":"Where am I?"}),
            json!({"type":"reasoning","content":[{"type":"reasoning_text","text":"Inspect first."}]}),
            json!({"type":"function_call","call_id":"a","name":"exec_command","arguments":"{\"cmd\":\"pwd\"}"}),
            json!({"type":"function_call_output","call_id":"a","output":"/workspace"}),
        ], &tools(), false).unwrap();
        assert!(prompt
            .rendered
            .contains("<think>\nInspect first.\n</think>"));
        assert!(prompt
            .rendered
            .contains("<parameter=cmd>\npwd\n</parameter>"));
        assert!(prompt
            .rendered
            .contains("<tool_response>\n/workspace\n</tool_response>"));
        assert!(prompt
            .rendered
            .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn rejects_unsupported_content_and_orphan_outputs() {
        for input in [
            json!({"role":"user","content":[{"type":"input_image","image_url":"test"}]}),
            json!({"type":"function_call_output","call_id":"unknown","output":"ok"}),
            json!({"type":"unknown"}),
        ] {
            assert!(render_qwen_responses_prompt("", &[input], &[], false).is_err());
        }
        assert!(qwen_function_tools(&[json!({"type":"custom","name":"apply_patch"})]).is_err());
        assert!(qwen_function_tools(&[tools()[0].clone(), tools()[0].clone()]).is_err());
    }

    #[test]
    fn chat_history_keeps_turn_boundaries_and_reasoning() {
        let prompt = crate::render_qwen_chat_prompt(
            &[crate::QwenChatTurn {
                user: "First".into(),
                reasoning: "Thought".into(),
                assistant: "Answer".into(),
            }],
            "Second",
            false,
        );
        assert!(prompt.rendered.contains("<|im_start|>user\nFirst<|im_end|>\n<|im_start|>assistant\n<think>\nThought\n</think>\n\nAnswer<|im_end|>\n<|im_start|>user\nSecond"));
    }
}
