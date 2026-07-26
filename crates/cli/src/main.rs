#![deny(unsafe_code)]

mod commands;
mod tracing_init;

use anyhow::Result;
use clap::{ArgAction, Parser, Subcommand};
use std::{net::SocketAddr, path::PathBuf};

#[derive(Debug, Parser)]
#[command(name = "inferno")]
#[command(about = "Lightweight MoE inference engine for Apple Silicon")]
struct Cli {
    /// Enable opt-in GLM MTP speculative decoding.
    #[arg(long, global = true, default_value_t = false)]
    speculative_mtp: bool,

    /// Enable the model-specific adaptive unified-memory controller.
    #[arg(long, global = true, default_value_t = false)]
    enable_unified_memory_controller: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate text with a supported quantized model.
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

        /// Page size for GLM's paged KV cache.
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

        /// Optional TSV output path for GLM Q2 runtime timings.
        #[arg(long)]
        profile_runtime: Option<PathBuf>,

        /// Optional TSV output path for GLM Q2 model-layer timings.
        #[arg(long)]
        profile_layers: Option<PathBuf>,

        /// Write generated-token throughput metrics to stderr after generation.
        #[arg(long, default_value_t = false)]
        measure_tokens_per_second: bool,

        /// Print only prefill and decode throughput on one stderr line.
        #[arg(
            long,
            default_value_t = false,
            conflicts_with = "measure_tokens_per_second"
        )]
        throughput_summary: bool,

        /// Append generated-token throughput metrics to a TSV file.
        #[arg(long)]
        throughput_file: Option<PathBuf>,

        /// Print a synchronized per-token bottleneck breakdown to stderr.
        #[arg(long, default_value_t = false)]
        profile_token_costs: bool,

        /// Expert-cache working-set budget in decimal GB.
        #[arg(long)]
        expert_cache_gb: Option<f64>,

        /// Total RAM budget in decimal GB for GLM's hot Metal KV tier.
        #[arg(long)]
        hot_kv_cache_gb: Option<f64>,

        /// Emit runtime memory telemetry to stderr during generation.
        #[arg(long, default_value_t = false)]
        enable_telemetry: bool,

        /// Write runtime memory telemetry to a file instead of interleaving it with streamed text.
        #[arg(long)]
        telemetry_file: Option<PathBuf>,

        /// Write every adaptive memory-controller decision to a TSV file. Requires --enable-unified-memory-controller.
        #[arg(long)]
        memory_controller_log: Option<PathBuf>,
    },

    /// Start a persistent local chat session.
    Chat {
        /// Directory containing a supported model.
        #[arg(long)]
        model: PathBuf,

        /// Optional config.json path. Defaults to <model>/config.json.
        #[arg(long)]
        config: Option<PathBuf>,

        /// Optional tokenizer.json path. Defaults to <model>/tokenizer.json.
        #[arg(long)]
        tokenizer: Option<PathBuf>,

        /// Page size for GLM's paged KV cache.
        #[arg(long, default_value_t = runtime::DEFAULT_KV_PAGE_SIZE)]
        page_size: usize,

        /// Optional maximum number of tokens generated for each answer.
        #[arg(long)]
        max_new_tokens: Option<usize>,

        /// Print only prefill and decode throughput after each answer.
        #[arg(long, default_value_t = false)]
        throughput_summary: bool,

        /// Expert-cache working-set budget in decimal GB.
        #[arg(long)]
        expert_cache_gb: Option<f64>,

        /// Total RAM budget in decimal GB for GLM's hot Metal KV tier.
        #[arg(long)]
        hot_kv_cache_gb: Option<f64>,

        /// Emit runtime memory telemetry to stderr during generation.
        #[arg(long, default_value_t = false)]
        enable_telemetry: bool,

        /// Write runtime memory telemetry to a file instead of the terminal.
        #[arg(long)]
        telemetry_file: Option<PathBuf>,

        /// Write every adaptive memory-controller decision to a TSV file. Requires --enable-unified-memory-controller.
        #[arg(long)]
        memory_controller_log: Option<PathBuf>,
    },

    /// Serve a supported model through the local Responses API.
    Serve {
        /// Directory containing a supported model.
        #[arg(long, default_value = "models/glm-5.2")]
        model: PathBuf,

        /// Optional config.json path. Defaults to <model>/config.json.
        #[arg(long)]
        config: Option<PathBuf>,

        /// Optional tokenizer.json path. Defaults to <model>/tokenizer.json.
        #[arg(long)]
        tokenizer: Option<PathBuf>,

        /// Local address exposed to Codex.
        #[arg(long, default_value = "127.0.0.1:11435")]
        bind: SocketAddr,

        /// Page size for GLM's paged KV cache.
        #[arg(long, default_value_t = runtime::DEFAULT_KV_PAGE_SIZE)]
        page_size: usize,

        /// Optional generated-token safety limit for each Codex turn.
        #[arg(long)]
        max_new_tokens: Option<usize>,

        /// Expert-cache working-set budget in decimal GB.
        #[arg(long)]
        expert_cache_gb: Option<f64>,

        /// Total RAM budget in decimal GB for GLM's hot Metal KV tier.
        #[arg(long)]
        hot_kv_cache_gb: Option<f64>,

        /// Emit runtime memory telemetry while serving Codex.
        #[arg(long, default_value_t = false)]
        enable_telemetry: bool,

        /// Write runtime memory telemetry to a file.
        #[arg(long)]
        telemetry_file: Option<PathBuf>,

        /// Write adaptive memory decisions to TSV. Requires --enable-unified-memory-controller.
        #[arg(long)]
        memory_controller_log: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let speculative_mtp = cli.speculative_mtp;
    let enable_unified_memory_controller = cli.enable_unified_memory_controller;
    let command = cli.command.unwrap_or_else(Command::default_chat);
    tracing_init::init(
        command.telemetry_to_stderr(),
        command.token_costs_to_stderr(),
    );
    match command {
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
            throughput_summary,
            throughput_file,
            profile_token_costs,
            expert_cache_gb,
            hot_kv_cache_gb,
            enable_telemetry,
            telemetry_file,
            memory_controller_log,
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
            throughput_summary,
            throughput_file.as_deref(),
            profile_token_costs,
            speculative_mtp,
            enable_unified_memory_controller,
            expert_cache_gb,
            hot_kv_cache_gb,
            enable_telemetry,
            telemetry_file.as_deref(),
            memory_controller_log.as_deref(),
        )?,
        Command::Chat {
            model,
            config,
            tokenizer,
            page_size,
            max_new_tokens,
            throughput_summary,
            expert_cache_gb,
            hot_kv_cache_gb,
            enable_telemetry,
            telemetry_file,
            memory_controller_log,
        } => commands::chat::run(
            model.as_path(),
            config.as_deref(),
            tokenizer.as_deref(),
            page_size,
            max_new_tokens,
            throughput_summary,
            speculative_mtp,
            enable_unified_memory_controller,
            expert_cache_gb,
            hot_kv_cache_gb,
            enable_telemetry,
            telemetry_file.as_deref(),
            memory_controller_log.as_deref(),
        )?,
        Command::Serve {
            model,
            config,
            tokenizer,
            bind,
            page_size,
            max_new_tokens,
            expert_cache_gb,
            hot_kv_cache_gb,
            enable_telemetry,
            telemetry_file,
            memory_controller_log,
        } => commands::serve::run(
            model.as_path(),
            config.as_deref(),
            tokenizer.as_deref(),
            bind,
            page_size,
            max_new_tokens,
            speculative_mtp,
            enable_unified_memory_controller,
            expert_cache_gb,
            hot_kv_cache_gb,
            enable_telemetry,
            telemetry_file.as_deref(),
            memory_controller_log.as_deref(),
        )?,
    }

    Ok(())
}

impl Command {
    fn default_chat() -> Self {
        Self::Chat {
            model: PathBuf::from("models/glm-5.2"),
            config: None,
            tokenizer: None,
            page_size: runtime::DEFAULT_KV_PAGE_SIZE,
            max_new_tokens: None,
            throughput_summary: false,
            expert_cache_gb: None,
            hot_kv_cache_gb: None,
            enable_telemetry: false,
            telemetry_file: None,
            memory_controller_log: None,
        }
    }

    fn telemetry_to_stderr(&self) -> bool {
        match self {
            Command::Generate {
                enable_telemetry,
                telemetry_file,
                ..
            } => *enable_telemetry && telemetry_file.is_none(),
            Command::Chat {
                enable_telemetry,
                telemetry_file,
                ..
            } => *enable_telemetry && telemetry_file.is_none(),
            Command::Serve {
                enable_telemetry,
                telemetry_file,
                ..
            } => *enable_telemetry && telemetry_file.is_none(),
        }
    }

    fn token_costs_to_stderr(&self) -> bool {
        match self {
            Command::Generate {
                profile_token_costs,
                ..
            } => *profile_token_costs,
            Command::Chat { .. } => false,
            Command::Serve { .. } => false,
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
        } = cli.command.expect("expected command")
        else {
            panic!("expected generate command");
        };
        assert!(!skip_special_tokens);
    }

    #[test]
    fn serve_uses_local_codex_defaults() {
        let cli = Cli::try_parse_from(["inferno", "serve"]).unwrap();

        let Command::Serve {
            model,
            bind,
            max_new_tokens,
            ..
        } = cli.command.expect("expected command")
        else {
            panic!("expected serve command");
        };
        assert_eq!(model, PathBuf::from("models/glm-5.2"));
        assert_eq!(bind, "127.0.0.1:11435".parse::<SocketAddr>().unwrap());
        assert_eq!(max_new_tokens, None);
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
        } = cli.command.expect("expected command")
        else {
            panic!("expected generate command");
        };
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

        let Command::Generate { page_size, .. } = cli.command.expect("expected command") else {
            panic!("expected generate command");
        };
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

        let Command::Generate { max_new_tokens, .. } = cli.command.expect("expected command")
        else {
            panic!("expected generate command");
        };
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

        let Command::Generate { max_new_tokens, .. } = cli.command.expect("expected command")
        else {
            panic!("expected generate command");
        };
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

        let Command::Generate { page_size, .. } = cli.command.expect("expected command") else {
            panic!("expected generate command");
        };
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
        } = cli.command.expect("expected command")
        else {
            panic!("expected generate command");
        };
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

        let Command::Generate { profile_layers, .. } = cli.command.expect("expected command")
        else {
            panic!("expected generate command");
        };
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
        } = cli.command.expect("expected command")
        else {
            panic!("expected generate command");
        };
        assert!(measure_tokens_per_second);
    }

    #[test]
    fn generate_accepts_compact_throughput_summary_flag() {
        let cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
            "--throughput-summary",
        ])
        .unwrap();

        let Command::Generate {
            throughput_summary, ..
        } = cli.command.expect("expected command")
        else {
            panic!("expected generate command");
        };
        assert!(throughput_summary);
    }

    #[test]
    fn compact_and_detailed_throughput_flags_conflict() {
        let error = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
            "--throughput-summary",
            "--measure-tokens-per-second",
        ])
        .unwrap_err();

        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn generate_accepts_synchronized_token_cost_profile() {
        let cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
            "--profile-token-costs",
        ])
        .unwrap();

        let Command::Generate {
            profile_token_costs,
            ..
        } = cli.command.expect("expected command")
        else {
            panic!("expected generate command");
        };
        assert!(profile_token_costs);
    }

    #[test]
    fn generate_mtp_is_explicitly_opt_in() {
        let default_cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
        ])
        .unwrap();
        assert!(!default_cli.speculative_mtp);

        let enabled_cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
            "--speculative-mtp",
        ])
        .unwrap();
        assert!(enabled_cli.speculative_mtp);
    }

    #[test]
    fn unified_memory_controller_is_explicitly_opt_in() {
        let default_chat_cli =
            Cli::try_parse_from(["inferno", "--enable-unified-memory-controller"]).unwrap();
        assert!(default_chat_cli.enable_unified_memory_controller);
        assert!(default_chat_cli.command.is_none());

        let default_cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
        ])
        .unwrap();
        assert!(!default_cli.enable_unified_memory_controller);

        let enabled_cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
            "--enable-unified-memory-controller",
        ])
        .unwrap();
        assert!(enabled_cli.enable_unified_memory_controller);

        let chat_cli = Cli::try_parse_from([
            "inferno",
            "chat",
            "--model",
            "/tmp/model",
            "--enable-unified-memory-controller",
        ])
        .unwrap();
        assert!(chat_cli.enable_unified_memory_controller);

        let serve_cli = Cli::try_parse_from([
            "inferno",
            "serve",
            "--model",
            "/tmp/model",
            "--enable-unified-memory-controller",
        ])
        .unwrap();
        assert!(serve_cli.enable_unified_memory_controller);
    }

    #[test]
    fn generate_accepts_cache_budget_overrides() {
        let cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
            "--expert-cache-gb",
            "9.5",
            "--hot-kv-cache-gb",
            "2.0",
        ])
        .unwrap();

        let Command::Generate {
            expert_cache_gb,
            hot_kv_cache_gb,
            ..
        } = cli.command.expect("expected command")
        else {
            panic!("expected generate command");
        };
        assert_eq!(expert_cache_gb, Some(9.5));
        assert_eq!(hot_kv_cache_gb, Some(2.0));
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
        } = cli.command.expect("expected command")
        else {
            panic!("expected generate command");
        };
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
        } = cli.command.expect("expected command")
        else {
            panic!("expected generate command");
        };
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

        let Command::Generate { telemetry_file, .. } = cli.command.expect("expected command")
        else {
            panic!("expected generate command");
        };
        assert_eq!(
            telemetry_file.unwrap(),
            PathBuf::from("/tmp/inferno-memory.log")
        );
    }

    #[test]
    fn generate_accepts_memory_controller_log_file() {
        let cli = Cli::try_parse_from([
            "inferno",
            "generate",
            "--model",
            "/tmp/model",
            "--prompt",
            "Hello GLM",
            "--memory-controller-log",
            "/tmp/inferno-memory-controller.tsv",
        ])
        .unwrap();

        let Command::Generate {
            memory_controller_log,
            ..
        } = cli.command.expect("expected command")
        else {
            panic!("expected generate command");
        };
        assert_eq!(
            memory_controller_log.unwrap(),
            PathBuf::from("/tmp/inferno-memory-controller.tsv")
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

    #[test]
    fn no_subcommand_selects_default_local_chat() {
        let cli = Cli::try_parse_from(["inferno"]).unwrap();
        assert!(!cli.speculative_mtp);
        assert!(!cli.enable_unified_memory_controller);
        let command = cli.command.unwrap_or_else(Command::default_chat);

        let Command::Chat {
            model,
            page_size,
            max_new_tokens,
            ..
        } = command
        else {
            panic!("expected default chat command");
        };
        assert_eq!(model, PathBuf::from("models/glm-5.2"));
        assert_eq!(page_size, runtime::DEFAULT_KV_PAGE_SIZE);
        assert_eq!(max_new_tokens, None);
    }

    #[test]
    fn default_chat_accepts_global_mtp_flag() {
        let cli = Cli::try_parse_from(["inferno", "--speculative-mtp"]).unwrap();
        assert!(cli.speculative_mtp);
        assert!(cli.command.is_none());
    }

    #[test]
    fn chat_accepts_model_and_has_no_default_answer_limit() {
        let cli = Cli::try_parse_from(["inferno", "chat", "--model", "/tmp/model"]).unwrap();

        let Command::Chat {
            model,
            page_size,
            max_new_tokens,
            ..
        } = cli.command.expect("expected command")
        else {
            panic!("expected chat command");
        };
        assert_eq!(model, PathBuf::from("/tmp/model"));
        assert_eq!(page_size, runtime::DEFAULT_KV_PAGE_SIZE);
        assert_eq!(max_new_tokens, None);
    }

    #[test]
    fn chat_accepts_cache_and_telemetry_controls() {
        let cli = Cli::try_parse_from([
            "inferno",
            "chat",
            "--model",
            "/tmp/model",
            "--max-new-tokens",
            "64",
            "--speculative-mtp",
            "--expert-cache-gb",
            "10.5",
            "--hot-kv-cache-gb",
            "1.0",
            "--enable-telemetry",
        ])
        .unwrap();
        assert!(cli.speculative_mtp);

        let Command::Chat {
            max_new_tokens,
            expert_cache_gb,
            hot_kv_cache_gb,
            enable_telemetry,
            ..
        } = cli.command.expect("expected command")
        else {
            panic!("expected chat command");
        };
        assert_eq!(max_new_tokens, Some(64));
        assert_eq!(expert_cache_gb, Some(10.5));
        assert_eq!(hot_kv_cache_gb, Some(1.0));
        assert!(enable_telemetry);
    }

    #[test]
    fn chat_accepts_compact_throughput_summary_flag() {
        let cli = Cli::try_parse_from([
            "inferno",
            "chat",
            "--model",
            "/tmp/model",
            "--throughput-summary",
        ])
        .unwrap();

        let Command::Chat {
            throughput_summary, ..
        } = cli.command.expect("expected command")
        else {
            panic!("expected chat command");
        };
        assert!(throughput_summary);
    }
}
