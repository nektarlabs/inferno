use super::{
    classify_input, reset_color, set_initial_assistant_color, write_events, AssistantStream,
    ChatInput, DecodedTextStream, TerminalInputGuard, ThroughputRecorder,
};
use crate::commands::{
    generate::write_throughput_summary,
    qwen::{self, QwenOptions},
};
use anyhow::Result;
use backend::MetalBackend;
use config::load_generation_config;
use std::{
    io::{self, BufRead, IsTerminal, Write},
    path::Path,
};
use tokenizer::{render_qwen_chat_prompt, LagunaThinkingMode, QwenChatTurn};

pub(super) fn run(
    options: &QwenOptions,
    model: &Path,
    config: &Path,
    tokenizer_path: &Path,
    max_new_tokens: Option<usize>,
    thinking: bool,
    throughput_summary: bool,
) -> Result<()> {
    let tokenizer = qwen::load_tokenizer(tokenizer_path)?;
    let generation = load_generation_config(&model.join("generation_config.json"))?;
    let backend = MetalBackend::new()?;
    let mut runtime = options.open(model, config, &backend)?;
    let stdin = io::stdin();
    let stdout = io::stdout();
    let input_is_terminal = stdin.is_terminal();
    let output_is_terminal = stdout.is_terminal();
    let mut input = stdin.lock();
    let mut output = stdout.lock();
    let mut history = Vec::<QwenChatTurn>::new();
    let mut line = String::new();
    let mode = if thinking {
        LagunaThinkingMode::Enabled
    } else {
        LagunaThinkingMode::Disabled
    };

    loop {
        if input_is_terminal {
            output.write_all(b"inferno> ")?;
            output.flush()?;
        }
        line.clear();
        if input.read_line(&mut line)? == 0 {
            return Ok(());
        }
        let prompt = line.trim();
        match classify_input(prompt) {
            ChatInput::Empty => continue,
            ChatInput::Exit => return Ok(()),
            ChatInput::Clear => {
                history.clear();
                runtime.reset_sequence()?;
                output.write_all(b"history cleared\n")?;
                output.flush()?;
                continue;
            }
            ChatInput::Prompt => {}
        }
        let rendered = render_qwen_chat_prompt(&history, prompt, thinking);
        let encoded = tokenizer.encode(&rendered.rendered, false)?;
        if encoded.token_ids.len() >= options.context_tokens {
            // Keep the session usable; never silently truncate a conversation.
            writeln!(
                output,
                "Context limit reached; use /clear to start a new conversation."
            )?;
            continue;
        }
        if max_new_tokens
            .is_some_and(|count| count > options.context_tokens - encoded.token_ids.len())
        {
            writeln!(
                output,
                "Not enough context for --max-new-tokens; use /clear to start a new conversation."
            )?;
            continue;
        }
        runtime.reset_sequence()?;
        let mut input_guard = TerminalInputGuard::suspend(&input, input_is_terminal)?;
        set_initial_assistant_color(&mut output, output_is_terminal, mode)?;
        let mut decoded = DecodedTextStream::new(&tokenizer, true);
        let mut assistant = AssistantStream::new(mode);
        let mut throughput = ThroughputRecorder::start();
        let generated = runtime.generate(
            &encoded.token_ids,
            max_new_tokens,
            &generation.eos_token_ids,
            |id| {
                throughput.record_token();
                if !generation.eos_token_ids.contains(&id) {
                    if let Some(text) = decoded.push(id)? {
                        write_events(&mut output, assistant.push(&text), output_is_terminal)?;
                    }
                }
                Ok(())
            },
        );
        let finished = if generated.is_ok() {
            write_events(&mut output, assistant.finish(), output_is_terminal)
        } else {
            Ok(())
        };
        let color = reset_color(&mut output, output_is_terminal);
        let restored = input_guard.restore();
        generated?;
        finished?;
        color?;
        restored?;
        output.write_all(b"\n")?;
        output.flush()?;
        if throughput_summary {
            write_throughput_summary(&throughput.finish(encoded.token_ids.len(), 0))?;
        }
        history.push(QwenChatTurn {
            user: prompt.into(),
            reasoning: assistant.thinking().trim().into(),
            assistant: assistant.answer().trim().into(),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires native Metal and local Qwen Q4, MTP and DFlash2 weights"]
    fn real_qwen_chat_remembers_prior_turn() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let model = workspace.join("models/qwen3.8-27b-4bit");
        let draft = workspace.join("models/qwen3.8-27b-dflash2");
        // Match CLI initialization order, including loading the tokenizer first.
        let tokenizer = qwen::load_tokenizer(&model.join("tokenizer.json")).unwrap();
        let generation = load_generation_config(&model.join("generation_config.json")).unwrap();
        let backend = MetalBackend::new().unwrap();
        for dflash in [false, true] {
            let options = QwenOptions {
                context_tokens: 4096,
                dflash_model: dflash.then(|| draft.clone()),
            };
            let mut runtime = options
                .open(&model, &model.join("config.json"), &backend)
                .unwrap();
            let mut history = Vec::new();
            for prompt in [
                "Remember this code: 7321. Reply only OK.",
                "What was the code? Reply only with the number.",
            ] {
                let rendered = render_qwen_chat_prompt(&history, prompt, false);
                let ids = tokenizer
                    .encode(&rendered.rendered, false)
                    .unwrap()
                    .token_ids;
                runtime.reset_sequence().unwrap();
                let mut decoded = DecodedTextStream::new(&tokenizer, true);
                let mut assistant = AssistantStream::new(LagunaThinkingMode::Disabled);
                runtime
                    .generate(&ids, Some(32), &generation.eos_token_ids, |id| {
                        if !generation.eos_token_ids.contains(&id) {
                            if let Some(text) = decoded.push(id)? {
                                assistant.push(&text);
                            }
                        }
                        Ok(())
                    })
                    .unwrap();
                assistant.finish();
                history.push(QwenChatTurn {
                    user: prompt.into(),
                    reasoning: String::new(),
                    assistant: assistant.answer().into(),
                });
            }
            assert!(
                history[1].assistant.contains("7321"),
                "dflash={dflash}: {}",
                history[1].assistant
            );
        }
    }
}
