use std::path::PathBuf;

use clap::Parser;
use config::load_config;
use gguf::GgufFile;
use inferno_io::EXPERT_PACK_FILE_NAME;
use model::{
    antirez_q2_artifact, create_expert_pack, expected_expert_pack_header, validate_routing_policy,
    Index,
};
use tracing_subscriber::{fmt, EnvFilter};

#[derive(Debug, Parser)]
#[command(name = "pack-experts")]
struct Args {
    #[arg(long)]
    model: PathBuf,
}

fn main() -> anyhow::Result<()> {
    init_tracing();
    let args = Args::parse();
    let config = load_config(&args.model.join("config.json"))?;
    let artifact = antirez_q2_artifact();
    let gguf = GgufFile::open(args.model.join(artifact.file_name))?;
    validate_routing_policy(&config, &gguf)?;
    let index = Index::from_gguf(&gguf, &config)?;
    let output_path = args.model.join(EXPERT_PACK_FILE_NAME);
    let header = expected_expert_pack_header(&gguf, &config, &index)?;
    tracing::info!(
        target: "inferno::expert_pack",
        output = %output_path.display(),
        file_gb = header.expected_file_bytes()? as f64 / 1_000_000_000.0,
        layers = header.layer_count,
        experts_per_layer = header.expert_count,
        "preparing lossless routed-expert pack"
    );
    let report = create_expert_pack(&gguf, &config, &index, &output_path)?;
    tracing::info!(
        target: "inferno::expert_pack",
        output = %report.output_path.display(),
        file_gb = report.file_bytes as f64 / 1_000_000_000.0,
        records = report.record_count,
        resumed_records = report.resumed_records,
        already_complete = report.already_complete,
        "routed-expert pack ready"
    );
    Ok(())
}

fn init_tracing() {
    let filter =
        std::env::var("RUST_LOG").unwrap_or_else(|_| "inferno::expert_pack=info".to_string());
    let _ = fmt()
        .with_env_filter(EnvFilter::new(filter))
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init();
}
