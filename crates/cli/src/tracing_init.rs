use tracing_subscriber::{fmt, EnvFilter};

pub fn init(enable_telemetry: bool) {
    let env_filter = match std::env::var("RUST_LOG") {
        Ok(filter) if enable_telemetry && !filter.contains("inferno::memory") => {
            EnvFilter::new(format!("{filter},inferno::memory=info"))
        }
        Ok(filter) => EnvFilter::new(filter),
        Err(_) if enable_telemetry => EnvFilter::new("off,inferno::memory=info"),
        Err(_) => EnvFilter::new("off"),
    };

    let _ = fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init();
}
