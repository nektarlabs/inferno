#![deny(unsafe_code)]

mod commands;
mod tracing_init;

use anyhow::Result;
use clap::{ArgAction, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "inferno")]
#[command(about = "Lightweight GLM-5.2 inference engine for Apple Silicon")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate text from a quantized GLM-5.2 model.
    Generate {
        /// Model directory. config.json and tokenizer.json are discovered from this directory unless overridden.
        #[arg(long)]
        model: PathBuf,

        /// Optional config.json path. Defaults to <model>/config.json when present.
        #[arg(long)]
        config: Option<PathBuf>,

        /// Optional tokenizer.json path. Defaults to <model>/tokenizer.json.
        #[arg(long)]
        tokenizer: Option<PathBuf>,

        /// Page size for the paged KV cache.
        #[arg(long, default_value_t = runtime::DEFAULT_KV_PAGE_SIZE)]
        page_size: usize,

        /// Prompt text to encode and generate from.
        #[arg(long)]
        prompt: String,

        /// Optional maximum number of generated token ids. If omitted, generation stops at EOS or context limit.
        #[arg(long)]
        max_new_tokens: Option<usize>,

        /// Ask the tokenizer post-processor to add model special tokens.
        #[arg(long, default_value_t = false)]
        add_special_tokens: bool,

        /// Drop special tokens while decoding generated ids back to text.
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        skip_special_tokens: bool,

        /// Optional TSV output path for Q2 runtime timings.
        #[arg(long)]
        profile_runtime: Option<PathBuf>,

        /// Optional TSV output path for Q2 model-layer timings.
        #[arg(long)]
        profile_layers: Option<PathBuf>,

        /// Write generated-token throughput metrics to stderr after generation.
        #[arg(long, default_value_t = false)]
        measure_tokens_per_second: bool,

        /// Append generated-token throughput metrics to a TSV file.
        #[arg(long)]
        throughput_file: Option<PathBuf>,

        /// Emit runtime memory telemetry to stderr during generation.
        #[arg(long, default_value_t = false)]
        enable_telemetry: bool,

        /// Write runtime memory telemetry to a file instead of interleaving it with streamed text.
        #[arg(long)]
        telemetry_file: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    tracing_init::init(cli.telemetry_to_stderr());
    match cli.command {
        Command::Generate {
            model,
            config,
            tokenizer,
            page_size,
            prompt,
            max_new_tokens,
            add_special_tokens,
            skip_special_tokens,
            profile_runtime,
            profile_layers,
            measure_tokens_per_second,
            throughput_file,
            enable_telemetry,
            telemetry_file,
        } => commands::generate::run(
            model.as_path(),
            config.as_deref(),
            tokenizer.as_deref(),
            page_size,
            &prompt,
            max_new_tokens,
            add_special_tokens,
            skip_special_tokens,
            profile_runtime.as_deref(),
            profile_layers.as_deref(),
            measure_tokens_per_second,
            throughput_file.as_deref(),
            enable_telemetry,
            telemetry_file.as_deref(),
        )?,
    }

    Ok(())
}

impl Cli {
    fn telemetry_to_stderr(&self) -> bool {
        match &self.command {
            Command::Generate {
                enable_telemetry,
                telemetry_file,
                ..
            } => *enable_telemetry && telemetry_file.is_none(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_accepts_explicit_skip_special_tokens_false() {
        let cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
            "--skip-special-tokens",
            "false",
        ])
        .unwrap();

        let Command::Generate {
            skip_special_tokens,
            ..
        } = cli.command;
        assert!(!skip_special_tokens);
    }

    #[test]
    fn generate_defaults_to_skipping_special_tokens() {
        let cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
        ])
        .unwrap();

        let Command::Generate {
            skip_special_tokens,
            ..
        } = cli.command;
        assert!(skip_special_tokens);
    }

    #[test]
    fn generate_defaults_to_paged_cache_page_size() {
        let cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
        ])
        .unwrap();

        let Command::Generate { page_size, .. } = cli.command;
        assert_eq!(page_size, runtime::DEFAULT_KV_PAGE_SIZE);
    }

    #[test]
    fn generate_has_no_default_token_limit() {
        let cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
        ])
        .unwrap();

        let Command::Generate { max_new_tokens, .. } = cli.command;
        assert_eq!(max_new_tokens, None);
    }

    #[test]
    fn generate_accepts_explicit_token_limit() {
        let cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
            "--max-new-tokens",
            "16",
        ])
        .unwrap();

        let Command::Generate { max_new_tokens, .. } = cli.command;
        assert_eq!(max_new_tokens, Some(16));
    }

    #[test]
    fn generate_accepts_page_size() {
        let cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--page-size",
            "64",
            "--prompt",
            "Hello GLM",
        ])
        .unwrap();

        let Command::Generate { page_size, .. } = cli.command;
        assert_eq!(page_size, 64);
    }

    #[test]
    fn generate_accepts_runtime_profile_path() {
        let cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
            "--profile-runtime",
            "/tmp/runtime.tsv",
        ])
        .unwrap();

        let Command::Generate {
            profile_runtime, ..
        } = cli.command;
        assert_eq!(profile_runtime.unwrap(), PathBuf::from("/tmp/runtime.tsv"));
    }

    #[test]
    fn generate_accepts_layer_profile_path() {
        let cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
            "--profile-layers",
            "/tmp/layers.tsv",
        ])
        .unwrap();

        let Command::Generate { profile_layers, .. } = cli.command;
        assert_eq!(profile_layers.unwrap(), PathBuf::from("/tmp/layers.tsv"));
    }

    #[test]
    fn generate_accepts_tokens_per_second_measurement_flag() {
        let cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
            "--measure-tokens-per-second",
        ])
        .unwrap();

        let Command::Generate {
            measure_tokens_per_second,
            ..
        } = cli.command;
        assert!(measure_tokens_per_second);
    }

    #[test]
    fn generate_accepts_throughput_file() {
        let cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
            "--throughput-file",
            "/tmp/inferno-throughput.tsv",
        ])
        .unwrap();

        let Command::Generate {
            throughput_file, ..
        } = cli.command;
        assert_eq!(
            throughput_file.unwrap(),
            PathBuf::from("/tmp/inferno-throughput.tsv")
        );
    }

    #[test]
    fn generate_accepts_telemetry_flag() {
        let cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
            "--enable-telemetry",
        ])
        .unwrap();

        let Command::Generate {
            enable_telemetry, ..
        } = cli.command;
        assert!(enable_telemetry);
    }

    #[test]
    fn generate_accepts_telemetry_file() {
        let cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
            "--telemetry-file",
            "/tmp/inferno-memory.log",
        ])
        .unwrap();

        let Command::Generate { telemetry_file, .. } = cli.command;
        assert_eq!(
            telemetry_file.unwrap(),
            PathBuf::from("/tmp/inferno-memory.log")
        );
    }

    #[test]
    fn generate_rejects_quantization_argument() {
        let err = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--quantization",
            "q2",
            "--prompt",
            "Hello GLM",
        ])
        .expect_err("production CLI is Q2-only and should reject quantization selection");

        assert!(err.to_string().contains("unexpected argument"));
    }
}
