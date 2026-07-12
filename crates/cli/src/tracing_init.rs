use tracing_subscriber::{fmt, EnvFilter};

pub fn init(enable_telemetry: bool, enable_token_costs: bool) {
    let mut filter = std::env::var("RUST_LOG").unwrap_or_else(|_| "off".to_string());
    if enable_telemetry && !filter.contains("inferno::memory") {
        filter.push_str(",inferno::memory=info");
    }
    if enable_token_costs && !filter.contains("inferno::token_cost") {
        filter.push_str(",inferno::token_cost=info");
    }
    let env_filter = EnvFilter::new(filter);

    let _ = fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init();
}
