use common::{Error, Result};
use serde_json::{Map, Value};
use std::borrow::Cow;

use crate::{ChatPrompt, LagunaThinkingMode};

const PROMPT_PREFIX: &str = "[gMASK]<sop><|system|>Reasoning Effort: Max";
const LAGUNA_PROMPT_PREFIX: &str = "〈|EOS|〉";
const LAGUNA_DEFAULT_SYSTEM: &str = "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be helpful to users through natural language conversations.";
const THINK_END: &str = "</think>";
const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";
const ARG_KEY_OPEN: &str = "<arg_key>";
const ARG_KEY_CLOSE: &str = "</arg_key>";
const ARG_VALUE_OPEN: &str = "<arg_value>";
const ARG_VALUE_CLOSE: &str = "</arg_value>";
const CODEX_EXEC_COMMAND: &str = "exec_command";
const LAGUNA_SHELL: &str = "shell";
const CODEX_APPLY_PATCH: &str = "apply_patch";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentFunctionCall {
    pub name: String,
    /// JSON object serialized as a string, matching the Responses API.
    pub arguments: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentOutputItem {
    Text(String),
    FunctionCall(AgentFunctionCall),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentOutput {
    pub reasoning: String,
    pub items: Vec<AgentOutputItem>,
}

pub fn map_laguna_function_call_to_codex(mut call: AgentFunctionCall) -> AgentFunctionCall {
    if call.name == LAGUNA_SHELL {
        call.name = CODEX_EXEC_COMMAND.to_string();
    }
    if call.name == CODEX_EXEC_COMMAND {
        call.arguments = normalize_exec_command_arguments(&call.arguments);
    }
    call
}

fn normalize_exec_command_arguments(arguments: &str) -> String {
    let Ok(mut arguments_json) = serde_json::from_str::<Value>(arguments) else {
        return arguments.to_string();
    };
    let Some(arguments_object) = arguments_json.as_object_mut() else {
        return arguments.to_string();
    };
    if arguments_object
        .get("workdir")
        .and_then(Value::as_str)
        .is_none()
    {
        return arguments.to_string();
    }
    let Some(command) = arguments_object.get("cmd").and_then(Value::as_str) else {
        return arguments.to_string();
    };
    let command = command.trim_start();
    let Some(after_cd) = command.strip_prefix("cd ") else {
        return arguments.to_string();
    };
    let Some(separator) = after_cd.find("&&") else {
        return arguments.to_string();
    };
    let target = after_cd[..separator].trim();
    if !target.starts_with('/') && !target.starts_with("'/") && !target.starts_with("\"/") {
        return arguments.to_string();
    }
    let normalized_command = after_cd[separator + 2..].trim_start().to_string();
    arguments_object.insert("cmd".to_string(), Value::String(normalized_command));
    serde_json::to_string(&arguments_json).unwrap_or_else(|_| arguments.to_string())
}

/// Direct Codex functions represented by GLM-5.2's native tool grammar.
pub fn is_supported_codex_function(name: &str) -> bool {
    matches!(name, "exec_command" | "write_stdin" | "apply_patch")
}

/// Renders the Codex Responses input using GLM-5.2's native chat contract.
pub fn render_codex_prompt(
    instructions: &str,
    input: &[Value],
    tools: &[Value],
) -> Result<ChatPrompt> {
    let mut rendered = String::with_capacity(estimate_prompt_capacity(instructions, input, tools));
    rendered.push_str(PROMPT_PREFIX);
    render_tools(&mut rendered, tools)?;
    if !instructions.trim().is_empty() {
        rendered.push_str("<|system|>");
        rendered.push_str(instructions);
    }
    for item in input {
        render_input_item(&mut rendered, item)?;
    }
    rendered.push_str("<|assistant|><think>");
    Ok(ChatPrompt { rendered })
}

/// Renders a Codex Responses turn using Laguna S 2.1's published chat and
/// direct-function tool grammar.
pub fn render_laguna_codex_prompt(
    instructions: &str,
    input: &[Value],
    tools: &[Value],
    thinking_mode: LagunaThinkingMode,
) -> Result<ChatPrompt> {
    let normalized_tools = normalize_laguna_tools(tools)?;
    let system = if instructions.trim().is_empty() {
        LAGUNA_DEFAULT_SYSTEM
    } else {
        instructions.trim()
    };
    let mut rendered = String::with_capacity(
        LAGUNA_PROMPT_PREFIX.len()
            + system.len()
            + input
                .iter()
                .map(|value| value.to_string().len())
                .sum::<usize>()
            + normalized_tools
                .iter()
                .map(|value| value.to_string().len())
                .sum::<usize>()
            + 1_024,
    );
    rendered.push_str(LAGUNA_PROMPT_PREFIX);
    rendered.push_str("<system>");
    rendered.push_str(system);
    render_laguna_tools(&mut rendered, &normalized_tools)?;
    rendered.push_str("</system>\n");
    render_laguna_input_items(&mut rendered, input)?;
    rendered.push_str("<assistant>");
    match thinking_mode {
        LagunaThinkingMode::Enabled => rendered.push_str("<think>"),
        LagunaThinkingMode::Disabled => rendered.push_str(THINK_END),
    }
    Ok(ChatPrompt { rendered })
}

/// Parses one complete GLM or Laguna generation into Responses-compatible
/// output items. Both checkpoints use the same reasoning and tool-call tags.
pub fn parse_agent_output(output: &str) -> Result<AgentOutput> {
    let output = complete_terminated_apply_patch(output);
    let (reasoning, visible) = output
        .rsplit_once(THINK_END)
        .ok_or_else(|| Error::tokenizer("GLM agent output ended before the </think> boundary"))?;
    let mut items = Vec::new();
    let mut remaining = visible;
    while let Some(start) = remaining.find(TOOL_CALL_OPEN) {
        push_text_item(&mut items, &remaining[..start]);
        let tool_body_start = start + TOOL_CALL_OPEN.len();
        let after_open = &remaining[tool_body_start..];
        let close = after_open.find(TOOL_CALL_CLOSE).ok_or_else(|| {
            Error::tokenizer("GLM agent output contains an unterminated <tool_call>")
        })?;
        let call = parse_tool_call(&after_open[..close])?;
        items.push(AgentOutputItem::FunctionCall(call));
        remaining = &after_open[close + TOOL_CALL_CLOSE.len()..];
    }
    push_text_item(&mut items, remaining);
    if items.is_empty() {
        items.push(AgentOutputItem::Text(String::new()));
    }
    Ok(AgentOutput {
        reasoning: reasoning.trim().to_string(),
        items,
    })
}

fn complete_terminated_apply_patch(output: &str) -> Cow<'_, str> {
    const APPLY_PATCH_CALL: &str = "<tool_call>apply_patch<arg_key>patch</arg_key><arg_value>";
    const PATCH_END: &str = "*** End Patch";

    if output.contains(TOOL_CALL_CLOSE) {
        return Cow::Borrowed(output);
    }
    let Some(call_start) = output.rfind(APPLY_PATCH_CALL) else {
        return Cow::Borrowed(output);
    };
    let call = &output[call_start..];
    let Some(patch_end) = call.rfind(PATCH_END) else {
        return Cow::Borrowed(output);
    };
    let after_patch = &call[patch_end + PATCH_END.len()..];
    if !after_patch.trim().is_empty() && after_patch.trim() != ARG_VALUE_CLOSE {
        return Cow::Borrowed(output);
    }

    let mut completed = output.to_string();
    if after_patch.trim().is_empty() {
        completed.push_str(ARG_VALUE_CLOSE);
    }
    completed.push_str(TOOL_CALL_CLOSE);
    Cow::Owned(completed)
}

/// Returns a parsed agent turn as soon as one complete function call exists.
///
/// An incomplete reasoning boundary or function-call closing tag means the
/// model is still generating. Once the closing tag exists, malformed output is
/// reported immediately instead of being mistaken for a usable tool call.
pub fn parse_complete_agent_tool_call(output: &str) -> Result<Option<AgentOutput>> {
    let Some((_, visible)) = output.split_once(THINK_END) else {
        return Ok(None);
    };
    if !visible.contains(TOOL_CALL_CLOSE) {
        return Ok(None);
    }

    let parsed = parse_agent_output(output)?;
    if parsed
        .items
        .iter()
        .any(|item| matches!(item, AgentOutputItem::FunctionCall(_)))
    {
        Ok(Some(parsed))
    } else {
        Ok(None)
    }
}

/// Returns the stable visible-text prefix that can be streamed before the
/// first function call. A partial `<tool_call>` marker is withheld until the
/// next token resolves whether it is text or protocol syntax.
pub fn streamable_agent_text(output: &str) -> Option<&str> {
    let (_, visible) = output.split_once(THINK_END)?;
    let end = visible
        .find(TOOL_CALL_OPEN)
        .unwrap_or_else(|| stable_text_end(visible, TOOL_CALL_OPEN));
    Some(visible[..end].trim_start())
}

fn stable_text_end(text: &str, marker: &str) -> usize {
    for prefix_len in (1..marker.len()).rev() {
        if text.ends_with(&marker[..prefix_len]) {
            return text.len() - prefix_len;
        }
    }
    text.len()
}

fn render_tools(rendered: &mut String, tools: &[Value]) -> Result<()> {
    let normalized = normalize_tools(tools)?;
    if normalized.is_empty() {
        return Ok(());
    }
    rendered.push_str(
        "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
    );
    for tool in normalized {
        rendered.push_str(&serde_json::to_string(&tool)?);
        rendered.push('\n');
    }
    rendered.push_str(
        "</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>",
    );
    Ok(())
}

fn normalize_tools(tools: &[Value]) -> Result<Vec<Value>> {
    tools
        .iter()
        .filter_map(normalize_function_tool)
        .collect::<Result<Vec<_>>>()
}

fn normalize_laguna_tools(tools: &[Value]) -> Result<Vec<Value>> {
    let mut normalized = normalize_tools(tools)?;
    for tool in tools {
        if tool.get("type").and_then(Value::as_str) != Some("custom")
            || tool.get("name").and_then(Value::as_str) != Some(CODEX_APPLY_PATCH)
        {
            continue;
        }
        normalized.push(serde_json::json!({
            "type": "function",
            "name": CODEX_APPLY_PATCH,
            "description": tool
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("Apply a patch to files in the workspace."),
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": "Complete apply_patch payload."
                    }
                },
                "required": ["patch"],
                "additionalProperties": false
            }
        }));
    }
    normalized
        .into_iter()
        .map(|tool| {
            let mut function = tool
                .as_object()
                .cloned()
                .ok_or_else(|| Error::tokenizer("normalized Laguna tool must be an object"))?;
            function.remove("type");
            if function.get("name").and_then(Value::as_str) == Some(CODEX_EXEC_COMMAND) {
                function.insert("name".to_string(), Value::String(LAGUNA_SHELL.to_string()));
            }
            Ok(serde_json::json!({
                "type": "function",
                "function": function,
            }))
        })
        .collect()
}

fn render_laguna_tools(rendered: &mut String, tools: &[Value]) -> Result<()> {
    if tools.is_empty() {
        return Ok(());
    }
    rendered.push_str(
        "\n\n### Tools\n\nYou may call functions to assist with the user query.\nAll available function signatures are listed below:\n<available_tools>\n",
    );
    for tool in tools {
        rendered.push_str(&serde_json::to_string(tool)?);
        rendered.push('\n');
    }
    let has_apply_patch = tools.iter().any(|tool| {
        tool.get("function")
            .and_then(|function| function.get("name"))
            .and_then(Value::as_str)
            == Some(CODEX_APPLY_PATCH)
    });
    rendered.push_str(
        "</available_tools>\n\nWhen repository work requires a tool, call it immediately. Do not describe the planned command, repeat the task, or print file contents as assistant text before the call.\n\nUse exactly this function-call format:\n<tool_call>{function-name}<arg_key>{argument-name}</arg_key><arg_value>{argument-value}</arg_value></tool_call>\n",
    );
    if has_apply_patch {
        rendered.push_str(
            "\nExample:\n<tool_call>apply_patch<arg_key>patch</arg_key><arg_value>*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch</arg_value></tool_call>\n\nThe patch must start with `*** Begin Patch`, end with `*** End Patch`, and prefix every added content line with `+`. Batch workspace inspection into as few shell calls as practical, then use apply_patch for file changes.",
        );
    } else {
        rendered.push_str(
            "\nExample:\n<tool_call>shell<arg_key>cmd</arg_key><arg_value>ls -la</arg_value></tool_call>",
        );
    }
    rendered.push_str(
        "\n\nUse the exact tool and argument names from the schemas. Finish the current response after emitting one complete tool call. After receiving a <tool_response>, use that result and continue the task instead of repeating the completed call.",
    );
    Ok(())
}

fn normalize_function_tool(tool: &Value) -> Option<Result<Value>> {
    let object = match tool
        .as_object()
        .ok_or_else(|| Error::tokenizer("Codex tool definition must be a JSON object"))
    {
        Ok(object) => object,
        Err(error) => return Some(Err(error)),
    };
    match object.get("type").and_then(Value::as_str) {
        Some("function") => {}
        // GLM-5.2's native tool grammar represents direct functions. Codex
        // namespace and hosted tools are omitted instead of being rewritten
        // into an invented naming convention.
        Some(_) => return None,
        None => {
            return Some(Err(Error::tokenizer(
                "Codex tool definition is missing a string type",
            )))
        }
    }
    let name = match object.get("name").and_then(Value::as_str) {
        Some(name) => name,
        None => {
            return Some(Err(Error::tokenizer(
                "Codex function tool definition is missing a string name",
            )))
        }
    };
    if name.is_empty() {
        return Some(Err(Error::tokenizer("Codex function tool name is empty")));
    }
    if !is_supported_codex_function(name) {
        return None;
    }

    let mut normalized = object.clone();
    normalized.remove("strict");
    normalized.remove("defer_loading");
    Some(Ok(Value::Object(normalized)))
}

fn render_input_item(rendered: &mut String, item: &Value) -> Result<()> {
    let item_type = item
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::tokenizer("Codex Responses input item is missing a string type"))?;
    match item_type {
        "message" => render_message(rendered, item),
        "function_call" | "custom_tool_call" => render_historical_tool_call(rendered, item),
        "function_call_output" | "custom_tool_call_output" => render_tool_output(rendered, item),
        // Reasoning is not replayed. GLM receives the resulting assistant action
        // and tool observation, which are the durable state needed for the turn.
        "reasoning" => Ok(()),
        other => Err(Error::tokenizer(format!(
            "unsupported Codex Responses input item type {other:?}"
        ))),
    }
}

fn render_laguna_input_items(rendered: &mut String, items: &[Value]) -> Result<()> {
    let mut assistant_open = false;
    for (index, item) in items.iter().enumerate() {
        let item_type = item.get("type").and_then(Value::as_str).ok_or_else(|| {
            Error::tokenizer("Codex Responses input item is missing a string type")
        })?;
        match item_type {
            "reasoning" => {}
            "message" if item.get("role").and_then(Value::as_str) == Some("assistant") => {
                close_laguna_assistant(rendered, &mut assistant_open);
                let content =
                    visible_content(item.get("content").ok_or_else(|| {
                        Error::tokenizer("Codex message input is missing content")
                    })?)?;
                rendered.push_str("<assistant></think>");
                if !next_non_reasoning_item_is_tool_call(items, index + 1) {
                    rendered.push_str(&content);
                }
                assistant_open = true;
            }
            "function_call" | "custom_tool_call" => {
                if !assistant_open {
                    rendered.push_str("<assistant></think>");
                    assistant_open = true;
                }
                render_laguna_tool_call(rendered, item)?;
            }
            "message" => {
                close_laguna_assistant(rendered, &mut assistant_open);
                render_laguna_message(rendered, item)?;
            }
            "function_call_output" | "custom_tool_call_output" => {
                close_laguna_assistant(rendered, &mut assistant_open);
                render_laguna_tool_output(rendered, item)?;
            }
            other => {
                return Err(Error::tokenizer(format!(
                    "unsupported Codex Responses input item type {other:?}"
                )));
            }
        }
    }
    close_laguna_assistant(rendered, &mut assistant_open);
    Ok(())
}

fn next_non_reasoning_item_is_tool_call(items: &[Value], start: usize) -> bool {
    items[start..]
        .iter()
        .find(|item| item.get("type").and_then(Value::as_str) != Some("reasoning"))
        .and_then(|item| item.get("type"))
        .and_then(Value::as_str)
        .is_some_and(|item_type| matches!(item_type, "function_call" | "custom_tool_call"))
}

fn close_laguna_assistant(rendered: &mut String, assistant_open: &mut bool) {
    if *assistant_open {
        rendered.push_str("</assistant>\n");
        *assistant_open = false;
    }
}

fn render_message(rendered: &mut String, item: &Value) -> Result<()> {
    let role = item
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::tokenizer("Codex message input is missing a string role"))?;
    let content = visible_content(
        item.get("content")
            .ok_or_else(|| Error::tokenizer("Codex message input is missing content"))?,
    )?;
    match role {
        "user" => rendered.push_str("<|user|>"),
        "system" | "developer" => rendered.push_str("<|system|>"),
        "assistant" => rendered.push_str("<|assistant|><think></think>"),
        other => {
            return Err(Error::tokenizer(format!(
                "unsupported Codex message role {other:?}"
            )));
        }
    }
    rendered.push_str(&content);
    Ok(())
}

fn render_laguna_message(rendered: &mut String, item: &Value) -> Result<()> {
    let role = item
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::tokenizer("Codex message input is missing a string role"))?;
    let content = visible_content(
        item.get("content")
            .ok_or_else(|| Error::tokenizer("Codex message input is missing content"))?,
    )?;
    match role {
        "user" => {
            rendered.push_str("<user>");
            rendered.push_str(&content);
            rendered.push_str("</user>\n");
        }
        "system" | "developer" => {
            rendered.push_str("<system>");
            rendered.push_str(&content);
            rendered.push_str("</system>\n");
        }
        "assistant" => {
            return Err(Error::tokenizer(
                "Laguna assistant message reached the non-assistant renderer",
            ));
        }
        other => {
            return Err(Error::tokenizer(format!(
                "unsupported Codex message role {other:?}"
            )));
        }
    }
    Ok(())
}

fn render_historical_tool_call(rendered: &mut String, item: &Value) -> Result<()> {
    let (name, arguments) = historical_tool_call(item)?;
    rendered.push_str("<|assistant|><think></think>");
    render_tool_call(rendered, name, &arguments)
}

fn render_laguna_tool_call(rendered: &mut String, item: &Value) -> Result<()> {
    let (name, arguments) = historical_tool_call(item)?;
    let name = if name == CODEX_EXEC_COMMAND {
        LAGUNA_SHELL
    } else {
        name
    };
    render_tool_call(rendered, name, &arguments)
}

fn historical_tool_call(item: &Value) -> Result<(&str, Map<String, Value>)> {
    let name = item
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::tokenizer("Codex function_call input is missing a string name"))?;
    if item.get("type").and_then(Value::as_str) == Some("custom_tool_call") {
        let input = item.get("input").and_then(Value::as_str).ok_or_else(|| {
            Error::tokenizer("Codex custom_tool_call input is missing string input")
        })?;
        return Ok((
            name,
            Map::from_iter([("patch".to_string(), Value::String(input.to_string()))]),
        ));
    }
    let arguments = item
        .get("arguments")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::tokenizer("Codex function_call input is missing string arguments"))?;
    let arguments: Value = serde_json::from_str(arguments).map_err(|error| {
        Error::tokenizer(format!(
            "Codex function_call arguments are not JSON: {error}"
        ))
    })?;
    let arguments = arguments.as_object().ok_or_else(|| {
        Error::tokenizer("Codex function_call arguments must encode a JSON object")
    })?;
    Ok((name, arguments.clone()))
}

fn render_tool_call(
    rendered: &mut String,
    name: &str,
    arguments: &Map<String, Value>,
) -> Result<()> {
    rendered.push_str("<tool_call>");
    rendered.push_str(name);
    for (key, value) in arguments {
        rendered.push_str(ARG_KEY_OPEN);
        rendered.push_str(key);
        rendered.push_str(ARG_KEY_CLOSE);
        rendered.push_str(ARG_VALUE_OPEN);
        match value {
            Value::String(value) => rendered.push_str(value),
            value => rendered.push_str(&serde_json::to_string(value)?),
        }
        rendered.push_str(ARG_VALUE_CLOSE);
    }
    rendered.push_str(TOOL_CALL_CLOSE);
    Ok(())
}

fn render_tool_output(rendered: &mut String, item: &Value) -> Result<()> {
    let output = item
        .get("output")
        .ok_or_else(|| Error::tokenizer("Codex function_call_output is missing output"))?;
    let output = visible_content(output)?;
    rendered.push_str("<|observation|><tool_response>");
    rendered.push_str(&output);
    rendered.push_str("</tool_response>");
    Ok(())
}

fn render_laguna_tool_output(rendered: &mut String, item: &Value) -> Result<()> {
    let output = item
        .get("output")
        .ok_or_else(|| Error::tokenizer("Codex function_call_output is missing output"))?;
    let output = visible_content(output)?;
    rendered.push_str("<tool_response>");
    rendered.push_str(&output);
    rendered.push_str(
        "</tool_response>\n<system>The tool call above has already run. Its output and exit code are the result, including when the exit code is non-zero. Never repeat the same call or an equivalent inspection. Use the result and perform a different action that advances the next unfinished requirement.</system>\n",
    );
    Ok(())
}

fn visible_content(content: &Value) -> Result<String> {
    match content {
        Value::String(text) => Ok(text.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                let part_type = part.get("type").and_then(Value::as_str).ok_or_else(|| {
                    Error::tokenizer("Codex content part is missing a string type")
                })?;
                match part_type {
                    "input_text" | "output_text" | "text" => {
                        let value = part.get("text").and_then(Value::as_str).ok_or_else(|| {
                            Error::tokenizer("Codex text content part is missing text")
                        })?;
                        text.push_str(value);
                    }
                    "input_image" | "image" | "image_url" => text.push_str(
                        "<reminder>You are unable to process this image because you don't have multi-modal input ability. Try different methods.</reminder>",
                    ),
                    other => {
                        return Err(Error::tokenizer(format!(
                            "unsupported Codex content part type {other:?}"
                        )));
                    }
                }
            }
            Ok(text)
        }
        other => Ok(serde_json::to_string(other)?),
    }
}

fn parse_tool_call(body: &str) -> Result<AgentFunctionCall> {
    let first_argument = body.find(ARG_KEY_OPEN).unwrap_or(body.len());
    let name = body[..first_argument].trim();
    if name.is_empty() {
        return Err(Error::tokenizer(
            "GLM <tool_call> has an empty function name",
        ));
    }

    let mut arguments = Map::new();
    let mut remaining = &body[first_argument..];
    while !remaining.trim().is_empty() {
        remaining = remaining.trim_start();
        let (key, after_key) = take_tag(remaining, ARG_KEY_OPEN, ARG_KEY_CLOSE)?;
        let (raw_value, after_value) = take_tag(after_key, ARG_VALUE_OPEN, ARG_VALUE_CLOSE)?;
        if key.is_empty() {
            return Err(Error::tokenizer(
                "GLM <tool_call> contains an empty argument key",
            ));
        }
        if arguments.contains_key(key) {
            return Err(Error::tokenizer(format!(
                "GLM <tool_call> repeats argument key {key:?}"
            )));
        }
        let value = serde_json::from_str(raw_value)
            .unwrap_or_else(|_| Value::String(raw_value.to_string()));
        arguments.insert(key.to_string(), value);
        remaining = after_value;
    }
    Ok(AgentFunctionCall {
        name: name.to_string(),
        arguments: serde_json::to_string(&Value::Object(arguments))?,
    })
}

fn take_tag<'a>(input: &'a str, open: &str, close: &str) -> Result<(&'a str, &'a str)> {
    let after_open = input
        .strip_prefix(open)
        .ok_or_else(|| Error::tokenizer(format!("expected {open:?} in GLM <tool_call>")))?;
    let close_index = after_open
        .find(close)
        .ok_or_else(|| Error::tokenizer(format!("missing {close:?} in GLM <tool_call>")))?;
    let value = &after_open[..close_index];
    let remaining = &after_open[close_index + close.len()..];
    Ok((value, remaining))
}

fn push_text_item(items: &mut Vec<AgentOutputItem>, text: &str) {
    let text = text.trim();
    if !text.is_empty() {
        items.push(AgentOutputItem::Text(text.to_string()));
    }
}

fn estimate_prompt_capacity(instructions: &str, input: &[Value], tools: &[Value]) -> usize {
    PROMPT_PREFIX.len()
        + instructions.len()
        + input
            .iter()
            .map(|value| value.to_string().len())
            .sum::<usize>()
        + tools
            .iter()
            .map(|value| value.to_string().len())
            .sum::<usize>()
        + 1_024
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn renders_codex_tools_messages_and_observations_for_glm() {
        let prompt = render_codex_prompt(
            "Work carefully.",
            &[
                json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List files"}]}),
                json!({"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\":\"ls\"}", "call_id": "call_1"}),
                json!({"type": "function_call_output", "call_id": "call_1", "output": "README.md"}),
            ],
            &[json!({
                "type": "function",
                "name": "exec_command",
                "description": "Run a command",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                "strict": false
            })],
        )
        .unwrap();

        assert!(prompt.rendered.starts_with(PROMPT_PREFIX));
        assert!(prompt.rendered.contains("# Tools"));
        assert!(prompt.rendered.contains("\"name\":\"exec_command\""));
        assert!(!prompt.rendered.contains("\"strict\""));
        assert!(prompt.rendered.contains("<|system|>Work carefully."));
        assert!(prompt.rendered.contains("<|user|>List files"));
        assert!(prompt.rendered.contains(
            "<tool_call>exec_command<arg_key>cmd</arg_key><arg_value>ls</arg_value></tool_call>"
        ));
        assert!(prompt
            .rendered
            .contains("<|observation|><tool_response>README.md</tool_response>"));
        assert!(prompt.rendered.ends_with("<|assistant|><think>"));
    }

    #[test]
    fn renders_codex_tools_messages_and_observations_for_laguna() {
        let prompt = render_laguna_codex_prompt(
            "Work carefully.",
            &[
                json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List files"}]}),
                json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Inspecting."}]}),
                json!({"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\":\"ls\"}", "call_id": "call_1"}),
                json!({"type": "function_call_output", "call_id": "call_1", "output": "README.md"}),
            ],
            &[json!({
                "type": "function",
                "name": "exec_command",
                "description": "Run a command",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                "strict": false
            })],
            LagunaThinkingMode::Enabled,
        )
        .unwrap();

        assert!(prompt
            .rendered
            .starts_with("〈|EOS|〉<system>Work carefully."));
        assert!(prompt.rendered.contains("### Tools"));
        assert!(prompt.rendered.contains("<available_tools>"));
        assert!(prompt.rendered.contains("\"name\":\"shell\""));
        assert!(!prompt.rendered.contains("\"name\":\"exec_command\""));
        assert!(prompt.rendered.contains(
            "<tool_call>shell<arg_key>cmd</arg_key><arg_value>ls -la</arg_value></tool_call>"
        ));
        assert!(prompt
            .rendered
            .contains("\"function\":{\"description\":\"Run a command\""));
        assert!(!prompt.rendered.contains("\"strict\""));
        assert!(prompt.rendered.contains("<user>List files</user>"));
        assert!(prompt.rendered.contains(
            "<assistant></think><tool_call>shell<arg_key>cmd</arg_key><arg_value>ls</arg_value></tool_call></assistant>"
        ));
        assert!(!prompt.rendered.contains("Inspecting."));
        assert_eq!(prompt.rendered.matches("<assistant></think>").count(), 1);
        assert!(prompt
            .rendered
            .contains("<tool_response>README.md</tool_response>"));
        assert!(prompt
            .rendered
            .contains("Never repeat the same call or an equivalent inspection"));
        assert!(prompt.rendered.ends_with("<assistant><think>"));
    }

    #[test]
    fn maps_laguna_shell_calls_back_to_codex_exec_command() {
        let call = AgentFunctionCall {
            name: LAGUNA_SHELL.to_string(),
            arguments: r#"{"cmd":"pwd"}"#.to_string(),
        };

        assert_eq!(
            map_laguna_function_call_to_codex(call),
            AgentFunctionCall {
                name: CODEX_EXEC_COMMAND.to_string(),
                arguments: r#"{"cmd":"pwd"}"#.to_string(),
            }
        );
    }

    #[test]
    fn removes_a_redundant_absolute_cd_when_codex_already_has_workdir() {
        let call = AgentFunctionCall {
            name: LAGUNA_SHELL.to_string(),
            arguments: json!({
                "cmd": "cd /tmp/misspelled-path && mkdir -p src",
                "workdir": "/tmp/correct-path"
            })
            .to_string(),
        };

        let mapped = map_laguna_function_call_to_codex(call);
        assert_eq!(
            serde_json::from_str::<Value>(&mapped.arguments).unwrap(),
            json!({"cmd": "mkdir -p src", "workdir": "/tmp/correct-path"})
        );
    }

    #[test]
    fn preserves_relative_cd_commands_inside_the_codex_workdir() {
        let call = AgentFunctionCall {
            name: LAGUNA_SHELL.to_string(),
            arguments: json!({
                "cmd": "cd crates/core && cargo test",
                "workdir": "/tmp/project"
            })
            .to_string(),
        };

        let mapped = map_laguna_function_call_to_codex(call);
        assert_eq!(
            serde_json::from_str::<Value>(&mapped.arguments).unwrap(),
            json!({"cmd": "cd crates/core && cargo test", "workdir": "/tmp/project"})
        );
    }

    #[test]
    fn renders_codex_freeform_apply_patch_as_a_laguna_function() {
        let patch = "*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch\n";
        let prompt = render_laguna_codex_prompt(
            "Edit the workspace.",
            &[
                json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Create hello.txt"}]}),
                json!({"type": "custom_tool_call", "name": "apply_patch", "input": patch, "call_id": "call_1"}),
                json!({"type": "custom_tool_call_output", "call_id": "call_1", "output": "Done!"}),
            ],
            &[json!({
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a patch"
            })],
            LagunaThinkingMode::Disabled,
        )
        .unwrap();

        assert!(prompt.rendered.contains("\"name\":\"apply_patch\""));
        assert!(prompt.rendered.contains("\"patch\":{\"description\""));
        assert!(prompt
            .rendered
            .contains("<tool_call>apply_patch<arg_key>patch</arg_key><arg_value>*** Begin Patch"));
        assert!(!prompt.rendered.contains(
            "<tool_call>shell<arg_key>cmd</arg_key><arg_value>ls -la</arg_value></tool_call>"
        ));
        assert!(prompt
            .rendered
            .contains("<tool_response>Done!</tool_response>"));
        assert!(prompt
            .rendered
            .contains("Batch workspace inspection into as few shell calls as practical"));
    }

    #[test]
    fn renders_laguna_codex_prompt_without_reasoning_when_disabled() {
        let prompt = render_laguna_codex_prompt(
            "Answer directly.",
            &[json!({"type": "message", "role": "user", "content": "Hello"})],
            &[],
            LagunaThinkingMode::Disabled,
        )
        .unwrap();

        assert!(prompt.rendered.ends_with("<assistant></think>"));
    }

    #[test]
    fn omits_codex_tools_outside_glms_direct_function_contract() {
        let prompt = render_codex_prompt(
            "",
            &[json!({"type": "message", "role": "user", "content": "Run pwd"})],
            &[
                json!({
                    "type": "function",
                    "name": "exec_command",
                    "description": "Run a command",
                    "parameters": {"type": "object"}
                }),
                json!({
                    "type": "function",
                    "name": "update_plan",
                    "parameters": {"type": "object"}
                }),
                json!({
                    "type": "namespace",
                    "name": "plugin_tools",
                    "tools": [{"type": "function", "name": "nested"}]
                }),
                json!({"type": "web_search", "external_web_access": false}),
            ],
        )
        .unwrap();

        assert!(prompt.rendered.contains("\"name\":\"exec_command\""));
        assert!(!prompt.rendered.contains("update_plan"));
        assert!(!prompt.rendered.contains("plugin_tools"));
        assert!(!prompt.rendered.contains("web_search"));
    }

    #[test]
    fn parses_visible_text_after_reasoning() {
        let output = parse_agent_output("private reasoning</think>The answer is Rome.").unwrap();

        assert_eq!(output.reasoning, "private reasoning");
        assert_eq!(
            output.items,
            vec![AgentOutputItem::Text("The answer is Rome.".to_string())]
        );
    }

    #[test]
    fn uses_the_final_reasoning_boundary_when_laguna_repeats_it() {
        let output = parse_agent_output("</think>draft answer</think>The final answer.").unwrap();

        assert_eq!(
            output.items,
            vec![AgentOutputItem::Text("The final answer.".to_string())]
        );
    }

    #[test]
    fn parses_glm_tool_arguments_into_responses_json() {
        let output = parse_agent_output(
            "inspect files</think><tool_call>exec_command<arg_key>cmd</arg_key><arg_value>cargo test</arg_value><arg_key>yield_time_ms</arg_key><arg_value>30000</arg_value></tool_call>",
        )
        .unwrap();

        assert_eq!(output.items.len(), 1);
        let AgentOutputItem::FunctionCall(call) = &output.items[0] else {
            panic!("expected function call");
        };
        assert_eq!(call.name, "exec_command");
        assert_eq!(
            serde_json::from_str::<Value>(&call.arguments).unwrap(),
            json!({"cmd": "cargo test", "yield_time_ms": 30000})
        );
    }

    #[test]
    fn detects_only_complete_valid_agent_tool_calls() {
        assert_eq!(
            parse_complete_agent_tool_call("still thinking").unwrap(),
            None
        );
        assert_eq!(
            parse_complete_agent_tool_call(
                "done</think><tool_call>exec_command<arg_key>cmd</arg_key>"
            )
            .unwrap(),
            None
        );
        assert_eq!(
            parse_complete_agent_tool_call("done</think>Visible text only.").unwrap(),
            None
        );

        let parsed = parse_complete_agent_tool_call(
            "done</think><tool_call>exec_command<arg_key>cmd</arg_key><arg_value>pwd</arg_value></tool_call>",
        )
        .unwrap()
        .expect("expected a complete function call");
        assert!(matches!(
            parsed.items.as_slice(),
            [AgentOutputItem::FunctionCall(_)]
        ));
    }

    #[test]
    fn streams_visible_text_without_exposing_protocol_prefixes() {
        assert_eq!(streamable_agent_text("still thinking"), None);
        assert_eq!(
            streamable_agent_text("done</think>Creating the file."),
            Some("Creating the file.")
        );
        assert_eq!(
            streamable_agent_text("done</think>\n\nCreating the file."),
            Some("Creating the file.")
        );
        assert_eq!(
            streamable_agent_text("done</think>Creating the file.<tool"),
            Some("Creating the file.")
        );
        assert_eq!(
            streamable_agent_text(
                "done</think>Creating the file.<tool_call>exec_command<arg_key>cmd"
            ),
            Some("Creating the file.")
        );
    }

    #[test]
    fn rejects_a_malformed_completed_agent_tool_call() {
        let error = parse_complete_agent_tool_call(
            "done</think><tool_call>exec_command<arg_key>cmd</arg_key></tool_call>",
        )
        .unwrap_err();
        assert!(error.to_string().contains("<arg_value>"));
    }

    #[test]
    fn recovers_a_complete_apply_patch_with_only_the_xml_suffix_missing() {
        let output = parse_agent_output(
            "done</think><tool_call>apply_patch<arg_key>patch</arg_key><arg_value>*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch",
        )
        .unwrap();
        let AgentOutputItem::FunctionCall(call) = &output.items[0] else {
            panic!("expected function call");
        };
        assert_eq!(call.name, "apply_patch");
        assert_eq!(
            serde_json::from_str::<Value>(&call.arguments).unwrap(),
            json!({"patch": "*** Begin Patch\n*** Add File: hello.txt\n+hello\n*** End Patch"})
        );
    }

    #[test]
    fn rejects_an_apply_patch_whose_patch_body_is_incomplete() {
        assert!(parse_agent_output(
            "done</think><tool_call>apply_patch<arg_key>patch</arg_key><arg_value>*** Begin Patch\n*** Add File: hello.txt\n+hello"
        )
        .is_err());
    }

    #[test]
    fn rejects_unclosed_thinking_and_tool_tags() {
        assert!(parse_agent_output("still thinking").is_err());
        assert!(parse_agent_output("done</think><tool_call>exec_command").is_err());
    }
}
