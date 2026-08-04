use std::{
    io::{self, BufRead, IsTerminal, Write},
    os::fd::{AsFd, BorrowedFd},
    path::{Path, PathBuf},
};

use anyhow::Result;
use backend::{Backend, MetalBackend};
use common::{Error, Result as InfernoResult};
use config::{
    detect_model_architecture, load_config, load_generation_config, load_laguna_config,
    LagunaConfig, LagunaProfile, ModelArchitecture,
};
use gguf::GgufFile;
use inferno_io::EXPERT_PACK_FILE_NAME;
use model::{
    expected_expert_pack_header, validate_routing_policy, LagunaModel, Model,
    DEFAULT_GGUF_OUTPUT_CHUNK_ROWS,
};
use runtime::{
    enable_laguna_memory_controller_log, enable_memory_controller_log, enable_memory_telemetry,
    enable_memory_telemetry_file, q2_memory_controller_spec, run_generate_streaming_with_options,
    CacheBudgetSpec, GenerationControl, GenerationOptions, LagunaRuntime, LagunaThinkingGuard,
    LAGUNA_THINKING_END_TOKEN_ID,
};
use rustix::termios::{
    tcflush, tcgetattr, tcsetattr, LocalModes, OptionalActions, QueueSelector, Termios,
};
use tokenizer::{
    render_chat_prompt, render_laguna_chat_prompt, render_laguna_xs_chat_prompt, ChatPrompt,
    ChatTurn, LagunaReasoningTurn, LagunaThinkingMode, Tokenizer,
};
use tracing::debug;

use super::generate::{
    cache_gb_to_bytes, discover_config_path, discover_tokenizer_path, expert_cache_slots_per_layer,
    laguna_runtime_options, laguna_thinking_mode, load_q2_readiness, resolve_q2_artifact,
    validate_generation_request, validate_laguna_prompt, validate_laguna_service_options,
    validate_laguna_tokenizer, validate_memory_controller_options, write_throughput_summary,
    DecodedTextStream, ThroughputRecorder,
};

const THINK_END: &str = "</think>";
const THINKING_COLOR: &[u8] = b"\x1b[90m";
const ANSWER_COLOR: &[u8] = b"\x1b[97m";
const RESET_COLOR: &[u8] = b"\x1b[0m";

#[allow(clippy::too_many_arguments)]
pub fn run(
    model_path: &Path,
    config_path: Option<&Path>,
    tokenizer_path: Option<&Path>,
    page_size: usize,
    max_new_tokens: Option<usize>,
    thinking: bool,
    throughput_summary: bool,
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
    if detect_model_architecture(&discovered_config)? == ModelArchitecture::Laguna {
        return run_laguna(
            model_path,
            &discovered_config,
            &discovered_tokenizer,
            page_size,
            max_new_tokens,
            thinking,
            throughput_summary,
            speculative_mtp,
            enable_unified_memory_controller,
            expert_cache_gb,
            hot_kv_cache_gb,
            enable_telemetry,
            telemetry_file,
            memory_controller_log,
        );
    }
    if thinking {
        return Err(Error::runtime("--thinking currently applies only to Laguna models").into());
    }

    validate_memory_controller_options(enable_unified_memory_controller, memory_controller_log)?;
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
        .map(|expert_cache_budget_bytes| {
            expert_cache_slots_per_layer(
                &readiness.index,
                config.num_routed_experts,
                config.num_nextn_predict_layers > 0,
                expert_cache_budget_bytes,
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

    if let Some(telemetry_file) = telemetry_file {
        enable_memory_telemetry_file(telemetry_file)?;
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

    run_interactive_loop(
        &model,
        &config,
        &backend,
        &tokenizer,
        &generation_config.eos_token_ids,
        &readiness.artifact_file_name,
        page_size,
        max_new_tokens,
        throughput_summary,
        speculative_mtp,
        hot_kv_cache_budget_bytes,
        dynamic_cache_budget,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_laguna(
    model_path: &Path,
    config_path: &Path,
    tokenizer_path: &Path,
    page_size: usize,
    max_new_tokens: Option<usize>,
    thinking: bool,
    throughput_summary: bool,
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
    // Reserve against Laguna's full configured context because chat history can
    // grow across turns while a Safetensors expert cache remains persistent.
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
    let mut runtime = LagunaRuntime::new(options)?;

    run_laguna_interactive_loop(
        &model,
        &config,
        &backend,
        &tokenizer,
        LagunaChatOptions {
            max_new_tokens,
            thinking_mode: laguna_thinking_mode(thinking),
            throughput_summary,
            profile: config.profile()?,
        },
        &mut runtime,
    )
}

#[derive(Debug, Clone, Copy)]
struct LagunaChatOptions {
    max_new_tokens: Option<usize>,
    thinking_mode: LagunaThinkingMode,
    throughput_summary: bool,
    profile: LagunaProfile,
}

enum LagunaChatHistory {
    S(Vec<ChatTurn>),
    Xs(Vec<LagunaReasoningTurn>),
}

impl LagunaChatHistory {
    fn new(profile: LagunaProfile) -> Self {
        match profile {
            LagunaProfile::S21 => Self::S(Vec::new()),
            LagunaProfile::Xs21 => Self::Xs(Vec::new()),
        }
    }

    fn clear(&mut self) {
        match self {
            Self::S(history) => history.clear(),
            Self::Xs(history) => history.clear(),
        }
    }

    fn render(&self, prompt: &str, thinking_mode: LagunaThinkingMode) -> ChatPrompt {
        match self {
            Self::S(history) => render_laguna_chat_prompt(history, prompt, thinking_mode),
            Self::Xs(history) => render_laguna_xs_chat_prompt(history, prompt, thinking_mode),
        }
    }

    fn push(&mut self, user: &str, assistant: &AssistantStream) {
        match self {
            Self::S(history) => history.push(ChatTurn {
                user: user.to_string(),
                assistant: assistant.answer().trim().to_string(),
            }),
            Self::Xs(history) => history.push(LagunaReasoningTurn {
                user: user.to_string(),
                reasoning: assistant.thinking().trim().to_string(),
                assistant: assistant.answer().trim().to_string(),
            }),
        }
    }
}

fn run_laguna_interactive_loop(
    model: &LagunaModel,
    config: &LagunaConfig,
    backend: &MetalBackend,
    tokenizer: &Tokenizer,
    options: LagunaChatOptions,
    runtime: &mut LagunaRuntime,
) -> Result<()> {
    let stdin = io::stdin();
    let input_is_terminal = stdin.is_terminal();
    let stdout = io::stdout();
    let output_is_terminal = stdout.is_terminal();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    let mut history = LagunaChatHistory::new(options.profile);
    let mut line = String::new();

    loop {
        if input_is_terminal {
            output.write_all(b"inferno> ")?;
            output.flush()?;
        }
        line.clear();
        if input.read_line(&mut line)? == 0 {
            if input_is_terminal {
                output.write_all(b"\n")?;
            }
            return Ok(());
        }

        let prompt = line.trim();
        match classify_input(prompt) {
            ChatInput::Empty => continue,
            ChatInput::Exit => return Ok(()),
            ChatInput::Clear => {
                history.clear();
                output.write_all(b"history cleared\n")?;
                output.flush()?;
                continue;
            }
            ChatInput::Prompt => {}
        }

        let rendered = history.render(prompt, options.thinking_mode);
        let encoded = tokenizer.encode(&rendered.rendered, false)?;
        validate_laguna_prompt(
            config,
            &encoded.token_ids,
            options.max_new_tokens,
            options.thinking_mode,
        )?;

        let mut input_guard = TerminalInputGuard::suspend(&input, input_is_terminal)?;
        set_initial_assistant_color(&mut output, output_is_terminal, options.thinking_mode)?;
        let mut decoded = DecodedTextStream::new(tokenizer, true);
        let mut assistant = AssistantStream::new(options.thinking_mode);
        let mut throughput = ThroughputRecorder::start();
        let mut thinking_guard = (options.thinking_mode == LagunaThinkingMode::Enabled)
            .then(LagunaThinkingGuard::default);
        let generation_result = runtime.generate_streaming_controlled(
            model,
            backend,
            &encoded.token_ids,
            options.max_new_tokens,
            &config.eos_token_id,
            |token_id| {
                throughput.record_token();
                let is_stop_token = config.eos_token_id.contains(&token_id);
                if !is_stop_token {
                    if let Some(text) = decoded.push(token_id)? {
                        write_events(&mut output, assistant.push(&text), output_is_terminal)?;
                    }
                }
                let Some(reason) = thinking_guard
                    .as_mut()
                    .and_then(|guard| guard.observe(token_id, is_stop_token))
                else {
                    return Ok(GenerationControl::Continue);
                };

                debug!(?reason, "forcing Laguna reasoning boundary");
                if let Some(text) = decoded.push(LAGUNA_THINKING_END_TOKEN_ID)? {
                    write_events(&mut output, assistant.push(&text), output_is_terminal)?;
                }
                if assistant.phase == AssistantPhase::Thinking {
                    write_events(&mut output, assistant.push(THINK_END), output_is_terminal)?;
                }
                Ok(GenerationControl::InjectNextToken(
                    LAGUNA_THINKING_END_TOKEN_ID,
                ))
            },
        );
        let finish_result = if generation_result.is_ok() {
            write_events(&mut output, assistant.finish(), output_is_terminal)
        } else {
            Ok(())
        };
        let color_result = reset_color(&mut output, output_is_terminal);
        let input_result = input_guard.restore();

        generation_result?;
        finish_result?;
        color_result?;
        input_result?;
        output.write_all(b"\n")?;
        output.flush()?;
        if options.throughput_summary {
            write_throughput_summary(
                &throughput.finish(encoded.token_ids.len(), config.num_experts_per_tok),
            )?;
        }

        history.push(prompt, &assistant);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_interactive_loop(
    model: &Model<'_>,
    config: &config::Config,
    backend: &MetalBackend,
    tokenizer: &Tokenizer,
    eos_token_ids: &[u32],
    artifact_file_name: &str,
    page_size: usize,
    max_new_tokens: Option<usize>,
    throughput_summary: bool,
    speculative_mtp: bool,
    hot_kv_cache_budget_bytes: Option<usize>,
    dynamic_cache_budget: Option<CacheBudgetSpec>,
) -> Result<()> {
    let stdin = io::stdin();
    let input_is_terminal = stdin.is_terminal();
    let stdout = io::stdout();
    let output_is_terminal = stdout.is_terminal();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    let mut history = Vec::<ChatTurn>::new();
    let mut line = String::new();

    loop {
        if input_is_terminal {
            output.write_all(b"inferno> ")?;
            output.flush()?;
        }
        line.clear();
        if input.read_line(&mut line)? == 0 {
            if input_is_terminal {
                output.write_all(b"\n")?;
            }
            return Ok(());
        }

        let prompt = line.trim();
        match classify_input(prompt) {
            ChatInput::Empty => continue,
            ChatInput::Exit => return Ok(()),
            ChatInput::Clear => {
                history.clear();
                output.write_all(b"history cleared\n")?;
                output.flush()?;
                continue;
            }
            ChatInput::Prompt => {}
        }

        let rendered = render_chat_prompt(&history, prompt);
        let encoded = tokenizer.encode(&rendered.rendered, false)?;
        validate_generation_request(
            config,
            eos_token_ids,
            &encoded.token_ids,
            max_new_tokens,
            page_size,
            artifact_file_name,
            &model.index().architecture,
            &model.index().summary,
        )?;

        let mut input_guard = TerminalInputGuard::suspend(&input, input_is_terminal)?;
        set_thinking_color(&mut output, output_is_terminal)?;
        let mut decoded = DecodedTextStream::new(tokenizer, true);
        let mut assistant = AssistantStream::default();
        let mut throughput = ThroughputRecorder::start();
        let generation_result = run_generate_streaming_with_options(
            model,
            config,
            backend,
            &encoded.token_ids,
            max_new_tokens,
            page_size,
            eos_token_ids,
            GenerationOptions {
                hot_kv_cache_budget_bytes,
                dynamic_cache_budget,
                profile_token_costs: false,
                speculative_mtp,
            },
            |token_id| {
                throughput.record_token();
                if let Some(text) = decoded.push(token_id)? {
                    write_events(&mut output, assistant.push(&text), output_is_terminal)?;
                }
                Ok(())
            },
        );
        let finish_result = if generation_result.is_ok() {
            write_events(&mut output, assistant.finish(), output_is_terminal)
        } else {
            Ok(())
        };
        let color_result = reset_color(&mut output, output_is_terminal);
        let input_result = input_guard.restore();

        generation_result?;
        finish_result?;
        color_result?;
        input_result?;
        output.write_all(b"\n")?;
        output.flush()?;
        if throughput_summary {
            write_throughput_summary(
                &throughput.finish(encoded.token_ids.len(), config.experts_per_token),
            )?;
        }

        let response = assistant.answer().trim().to_string();
        history.push(ChatTurn {
            user: prompt.to_string(),
            assistant: response,
        });
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChatInput {
    Empty,
    Exit,
    Clear,
    Prompt,
}

fn classify_input(input: &str) -> ChatInput {
    match input {
        "" => ChatInput::Empty,
        "/exit" => ChatInput::Exit,
        "/clear" => ChatInput::Clear,
        _ => ChatInput::Prompt,
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum AssistantPhase {
    #[default]
    Thinking,
    Answer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum AssistantEvent {
    Thinking(String),
    AnswerStarted,
    Answer(String),
}

#[derive(Debug, Default)]
struct AssistantStream {
    phase: AssistantPhase,
    pending: String,
    thinking: String,
    answer: String,
}

impl AssistantStream {
    fn new(thinking_mode: LagunaThinkingMode) -> Self {
        Self {
            phase: match thinking_mode {
                LagunaThinkingMode::Disabled => AssistantPhase::Answer,
                LagunaThinkingMode::Enabled => AssistantPhase::Thinking,
            },
            ..Self::default()
        }
    }

    fn push(&mut self, text: &str) -> Vec<AssistantEvent> {
        if text.is_empty() {
            return Vec::new();
        }
        if self.phase == AssistantPhase::Answer {
            self.answer.push_str(text);
            return vec![AssistantEvent::Answer(text.to_string())];
        }

        self.pending.push_str(text);
        if let Some(marker_start) = self.pending.find(THINK_END) {
            let thinking = self.pending[..marker_start].to_string();
            let answer_start = marker_start + THINK_END.len();
            let answer = self.pending[answer_start..].to_string();
            self.pending.clear();
            self.phase = AssistantPhase::Answer;

            let mut events = Vec::with_capacity(3);
            if !thinking.is_empty() {
                self.thinking.push_str(&thinking);
                events.push(AssistantEvent::Thinking(thinking));
            }
            events.push(AssistantEvent::AnswerStarted);
            if !answer.is_empty() {
                self.answer.push_str(&answer);
                events.push(AssistantEvent::Answer(answer));
            }
            return events;
        }

        let retained_bytes = trailing_marker_prefix_bytes(&self.pending);
        let emitted_bytes = self.pending.len() - retained_bytes;
        if emitted_bytes == 0 {
            return Vec::new();
        }
        let emitted = self.pending[..emitted_bytes].to_string();
        self.pending.drain(..emitted_bytes);
        self.thinking.push_str(&emitted);
        vec![AssistantEvent::Thinking(emitted)]
    }

    fn finish(&mut self) -> Vec<AssistantEvent> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        let pending = std::mem::take(&mut self.pending);
        match self.phase {
            AssistantPhase::Thinking => {
                self.thinking.push_str(&pending);
                vec![AssistantEvent::Thinking(pending)]
            }
            AssistantPhase::Answer => {
                self.answer.push_str(&pending);
                vec![AssistantEvent::Answer(pending)]
            }
        }
    }

    fn answer(&self) -> &str {
        &self.answer
    }

    fn thinking(&self) -> &str {
        &self.thinking
    }
}

fn trailing_marker_prefix_bytes(text: &str) -> usize {
    let max_prefix = text.len().min(THINK_END.len() - 1);
    (1..=max_prefix)
        .rev()
        .find(|&bytes| text.ends_with(&THINK_END[..bytes]))
        .unwrap_or(0)
}

fn set_thinking_color(output: &mut impl Write, styled: bool) -> InfernoResult<()> {
    let bytes = if styled { THINKING_COLOR } else { &[] };
    write_terminal_bytes(output, bytes)
}

fn set_initial_assistant_color(
    output: &mut impl Write,
    styled: bool,
    thinking_mode: LagunaThinkingMode,
) -> InfernoResult<()> {
    let bytes = match (styled, thinking_mode) {
        (true, LagunaThinkingMode::Disabled) => ANSWER_COLOR,
        (true, LagunaThinkingMode::Enabled) => THINKING_COLOR,
        (false, _) => &[],
    };
    write_terminal_bytes(output, bytes)
}

fn reset_color(output: &mut impl Write, styled: bool) -> InfernoResult<()> {
    let bytes = if styled { RESET_COLOR } else { &[] };
    write_terminal_bytes(output, bytes)
}

fn write_events(
    output: &mut impl Write,
    events: Vec<AssistantEvent>,
    styled: bool,
) -> InfernoResult<()> {
    for event in events {
        let bytes = match event {
            AssistantEvent::Thinking(text) | AssistantEvent::Answer(text) => text.into_bytes(),
            AssistantEvent::AnswerStarted if styled => {
                let mut bytes = Vec::with_capacity(ANSWER_COLOR.len() + 1);
                bytes.extend_from_slice(ANSWER_COLOR);
                bytes.push(b'\n');
                bytes
            }
            AssistantEvent::AnswerStarted => vec![b'\n'],
        };
        write_terminal_bytes(output, &bytes)?;
    }
    Ok(())
}

fn write_terminal_bytes(output: &mut impl Write, bytes: &[u8]) -> InfernoResult<()> {
    output.write_all(bytes).map_err(|source| Error::Io {
        path: PathBuf::from("<stdout>"),
        source,
    })?;
    output.flush().map_err(|source| Error::Io {
        path: PathBuf::from("<stdout>"),
        source,
    })
}

struct TerminalInputGuard<'a> {
    fd: Option<BorrowedFd<'a>>,
    original: Option<Termios>,
}

impl<'a> TerminalInputGuard<'a> {
    fn suspend(input: &'a impl AsFd, enabled: bool) -> InfernoResult<Self> {
        if !enabled {
            return Ok(Self {
                fd: None,
                original: None,
            });
        }

        let fd = input.as_fd();
        let original = tcgetattr(fd).map_err(terminal_input_error)?;
        let mut suspended = original.clone();
        suspended
            .local_modes
            .remove(LocalModes::ECHO | LocalModes::ECHONL | LocalModes::ISIG);
        tcsetattr(fd, OptionalActions::Now, &suspended).map_err(terminal_input_error)?;
        Ok(Self {
            fd: Some(fd),
            original: Some(original),
        })
    }

    fn restore(&mut self) -> InfernoResult<()> {
        let (Some(fd), Some(original)) = (self.fd, self.original.as_ref()) else {
            return Ok(());
        };

        let flush_error = tcflush(fd, QueueSelector::IFlush).err();
        let restore_result = tcsetattr(fd, OptionalActions::Now, original);
        if restore_result.is_ok() {
            self.fd = None;
            self.original = None;
        }
        restore_result.map_err(terminal_input_error)?;
        if let Some(error) = flush_error {
            return Err(terminal_input_error(error));
        }
        Ok(())
    }
}

impl Drop for TerminalInputGuard<'_> {
    fn drop(&mut self) {
        if let (Some(fd), Some(original)) = (self.fd, self.original.as_ref()) {
            let _ = tcflush(fd, QueueSelector::IFlush);
            let _ = tcsetattr(fd, OptionalActions::Now, original);
        }
    }
}

fn terminal_input_error(error: rustix::io::Errno) -> Error {
    Error::Io {
        path: PathBuf::from("<stdin>"),
        source: io::Error::from_raw_os_error(error.raw_os_error()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_only_the_supported_session_commands() {
        assert_eq!(classify_input(""), ChatInput::Empty);
        assert_eq!(classify_input("/exit"), ChatInput::Exit);
        assert_eq!(classify_input("/clear"), ChatInput::Clear);
        assert_eq!(classify_input("/help"), ChatInput::Prompt);
    }

    #[test]
    fn streams_thinking_and_answer_across_a_split_marker() {
        let mut stream = AssistantStream::default();

        assert_eq!(
            stream.push("I should reason carefully.</thi"),
            vec![AssistantEvent::Thinking(
                "I should reason carefully.".to_string()
            )]
        );
        assert_eq!(
            stream.push("nk>The capital is Rome."),
            vec![
                AssistantEvent::AnswerStarted,
                AssistantEvent::Answer("The capital is Rome.".to_string())
            ]
        );
        assert_eq!(stream.finish(), Vec::<AssistantEvent>::new());
        assert_eq!(stream.thinking, "I should reason carefully.");
        assert_eq!(stream.answer(), "The capital is Rome.");
    }

    #[test]
    fn keeps_unclosed_output_in_the_thinking_channel() {
        let mut stream = AssistantStream::default();

        assert_eq!(
            stream.push("unfinished</thi"),
            vec![AssistantEvent::Thinking("unfinished".to_string())]
        );
        assert_eq!(
            stream.finish(),
            vec![AssistantEvent::Thinking("</thi".to_string())]
        );
        assert_eq!(stream.thinking, "unfinished</thi");
        assert!(stream.answer().is_empty());
    }

    #[test]
    fn streams_direct_laguna_answers_without_a_thinking_boundary() {
        let mut stream = AssistantStream::new(LagunaThinkingMode::Disabled);

        assert_eq!(
            stream.push("The capital is Rome."),
            vec![AssistantEvent::Answer("The capital is Rome.".to_string())]
        );
        assert_eq!(stream.finish(), Vec::<AssistantEvent>::new());
        assert!(stream.thinking.is_empty());
        assert_eq!(stream.answer(), "The capital is Rome.");
    }

    #[test]
    fn terminal_events_use_gray_thinking_and_an_unlabelled_white_answer() {
        let mut output = Vec::new();

        set_thinking_color(&mut output, true).unwrap();
        write_events(
            &mut output,
            vec![
                AssistantEvent::Thinking("reasoning".to_string()),
                AssistantEvent::AnswerStarted,
                AssistantEvent::Answer("Rome".to_string()),
            ],
            true,
        )
        .unwrap();
        reset_color(&mut output, true).unwrap();

        assert_eq!(output, b"\x1b[90mreasoning\x1b[97m\nRome\x1b[0m");
        assert!(!String::from_utf8(output).unwrap().contains('>'));
    }

    #[test]
    fn direct_laguna_answers_start_in_white() {
        let mut output = Vec::new();

        set_initial_assistant_color(&mut output, true, LagunaThinkingMode::Disabled).unwrap();
        write_events(
            &mut output,
            vec![AssistantEvent::Answer("Rome".to_string())],
            true,
        )
        .unwrap();
        reset_color(&mut output, true).unwrap();

        assert_eq!(output, b"\x1b[97mRome\x1b[0m");
    }

    #[test]
    fn non_terminal_events_do_not_emit_ansi_sequences() {
        let mut output = Vec::new();

        set_thinking_color(&mut output, false).unwrap();
        write_events(
            &mut output,
            vec![
                AssistantEvent::Thinking("reasoning".to_string()),
                AssistantEvent::AnswerStarted,
                AssistantEvent::Answer("Rome".to_string()),
            ],
            false,
        )
        .unwrap();
        reset_color(&mut output, false).unwrap();

        assert_eq!(output, b"reasoning\nRome");
    }
}
