use std::path::PathBuf;

use backend::MetalBackend;
use config::load_generation_config;
use tokenizer::{render_qwen_user_prompt, Tokenizer};

use super::*;

const TOKEN_LIMIT: usize = 128;
const FIBONACCI_PROMPT: &str = "Write a JavaScript function to calculate the Fibonacci sequence.";
const EXPLANATION_PROMPT: &str = "Explain how a hash table handles collisions, with a concrete example and the tradeoffs of separate chaining and open addressing.";

struct Sample {
    tokens: Vec<u32>,
    first: Option<Instant>,
    last: Option<Instant>,
    elapsed: Duration,
}

impl Sample {
    fn new() -> Self {
        Self {
            tokens: Vec::new(),
            first: None,
            last: None,
            elapsed: Duration::ZERO,
        }
    }

    fn push(&mut self, token: u32) {
        let now = Instant::now();
        self.first.get_or_insert(now);
        self.last = Some(now);
        self.tokens.push(token);
    }

    fn decode_tps(&self) -> f64 {
        assert!(self.tokens.len() > 1);
        (self.tokens.len() - 1) as f64
            / self
                .last
                .unwrap()
                .duration_since(self.first.unwrap())
                .as_secs_f64()
    }

    fn total_tps(&self) -> f64 {
        self.tokens.len() as f64 / self.elapsed.as_secs_f64()
    }
}

fn autoregressive(
    backend: &MetalBackend,
    config: &QwenConfig,
    weights: &QwenDeviceWeights,
    prompt: &[u32],
    eos: &[u32],
) -> Sample {
    let mut state = QwenModelState::create(backend, config, 1, prompt.len() + TOKEN_LIMIT).unwrap();
    let mut sample = Sample::new();
    let start = Instant::now();
    let mut hidden = forward_qwen_hidden_device(
        backend,
        config,
        weights,
        &mut state,
        prompt,
        &[1, prompt.len()],
    )
    .unwrap();
    loop {
        let token =
            one_batch_token(greedy_qwen_next_tokens_device(backend, weights, &hidden).unwrap())
                .unwrap();
        sample.push(token);
        if eos.contains(&token) || sample.tokens.len() == TOKEN_LIMIT {
            break;
        }
        hidden =
            forward_qwen_hidden_device(backend, config, weights, &mut state, &[token], &[1, 1])
                .unwrap();
    }
    sample.elapsed = start.elapsed();
    sample
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Model loading and state allocation are outside the timer; prefill is included
/// in total throughput. Decode uses first-to-last emitted token, like the CLI.
#[test]
#[ignore = "requires local Qwen W4/DFlash2 artifacts and exclusive native Metal access"]
fn benchmark_real_qwen_dflash_vs_autoregressive() {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let model_dir = std::env::var_os("INFERNO_QWEN_W4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace.join("models/qwen3.8-27b-4bit"));
    let draft_dir = std::env::var_os("INFERNO_QWEN_DFLASH_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace.join("models/qwen3.8-27b-dflash2"));
    let backend = MetalBackend::new().unwrap();
    let config = load_qwen_config(&model_dir.join("config.json")).unwrap();
    let weights = {
        let source = QwenWeightIndex::open(&model_dir, &config).unwrap();
        QwenDeviceWeights::prepare_resident(&source, &config, &backend).unwrap()
    };
    let draft_config = load_dflash_config(&draft_dir.join("config.json")).unwrap();
    validate_dflash_target_compatibility(&config, &draft_config).unwrap();
    let draft_weights = {
        let source = DFlashWeightIndex::open(&draft_dir, &draft_config).unwrap();
        DFlashDeviceWeights::prepare(&source, &backend).unwrap()
    };
    let generation = load_generation_config(&model_dir.join("generation_config.json")).unwrap();
    let tokenizer = Tokenizer::from_file(&model_dir.join("tokenizer.json")).unwrap();

    for (name, text) in [
        ("fibonacci", FIBONACCI_PROMPT),
        ("explanation", EXPLANATION_PROMPT),
    ] {
        let prompt = tokenizer
            .encode(&render_qwen_user_prompt(text, false).rendered, false)
            .unwrap()
            .token_ids;
        let mut decode = [Vec::new(), Vec::new()];
        let mut total = [Vec::new(), Vec::new()];
        // One warmup pair, then three measured pairs with alternating order.
        for repetition in 0..4 {
            let order = if repetition % 2 == 0 { [0, 1] } else { [1, 0] };
            let mut outputs = [Vec::new(), Vec::new()];
            for method in order {
                let sample = if method == 0 {
                    autoregressive(
                        &backend,
                        &config,
                        &weights,
                        &prompt,
                        &generation.eos_token_ids,
                    )
                } else {
                    let mut state = QwenModelState::create_speculative(
                        &backend,
                        &config,
                        1,
                        prompt.len() + TOKEN_LIMIT,
                    )
                    .unwrap();
                    let mut draft_state = DFlashState::create(&backend, &draft_config, 1).unwrap();
                    let mut sample = Sample::new();
                    let start = Instant::now();
                    let report = generate_dflash(
                        &backend,
                        &config,
                        &weights,
                        &mut state,
                        &draft_config,
                        &draft_weights,
                        &mut draft_state,
                        &prompt,
                        TOKEN_LIMIT,
                        &generation.eos_token_ids,
                        |token| {
                            sample.push(token);
                            Ok(())
                        },
                    )
                    .unwrap();
                    sample.elapsed = start.elapsed();
                    assert_eq!(report.generated_tokens, sample.tokens.len());
                    eprintln!(
                        "prompt={name} repetition={repetition} accepted={} proposed={} passes={}",
                        report.accepted_draft_tokens,
                        report.draft_tokens,
                        report.verification_passes
                    );
                    sample
                };
                eprintln!("prompt={name} repetition={repetition} method={} tokens={} decode_tps={:.3} total_tps={:.3}",
                    if method == 0 { "autoregressive" } else { "dflash" },
                    sample.tokens.len(), sample.decode_tps(), sample.total_tps());
                if repetition > 0 {
                    decode[method].push(sample.decode_tps());
                    total[method].push(sample.total_tps());
                }
                outputs[method] = sample.tokens;
            }
            assert_eq!(
                outputs[0], outputs[1],
                "DFlash token mismatch on {name}, repetition {repetition}"
            );
        }
        let ar_decode = median(&mut decode[0]);
        let draft_decode = median(&mut decode[1]);
        let ar_total = median(&mut total[0]);
        let draft_total = median(&mut total[1]);
        eprintln!("prompt={name} median_ar_decode_tps={ar_decode:.3} median_dflash_decode_tps={draft_decode:.3} decode_speedup={:.3} total_speedup={:.3}", draft_decode / ar_decode, draft_total / ar_total);
    }
}
