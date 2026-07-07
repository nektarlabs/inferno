use backend::MetalBackend;
use runtime::{benchmark_dsa_selected_attention, DsaAttentionBenchmarkConfig};

fn main() {
    let backend = match MetalBackend::new() {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("inferno dsa-attention benchmark: native Metal backend unavailable: {error}");
            return;
        }
    };

    let config = DsaAttentionBenchmarkConfig {
        layer_count: 8,
        batch: 1,
        attention_heads: 8,
        tokens: 512,
        selected_tokens: 64,
        key_head_dim: 256,
        value_head_dim: 256,
        page_size: 128,
    };

    match benchmark_dsa_selected_attention(&backend, config) {
        Ok(report) => {
            eprintln!("inferno dsa-attention benchmark");
            eprintln!("layers: {}", report.layer_count);
            eprintln!("tokens: {}", report.tokens);
            eprintln!("selected_tokens: {}", report.selected_tokens);
            eprintln!("dense_seconds: {:.6}", report.dense_seconds);
            eprintln!("selected_seconds: {:.6}", report.selected_seconds);
            eprintln!(
                "dense_seconds_per_layer: {:.6}",
                report.dense_seconds_per_layer
            );
            eprintln!(
                "selected_seconds_per_layer: {:.6}",
                report.selected_seconds_per_layer
            );
            eprintln!(
                "selected_vs_dense_speedup: {:.3}",
                report.selected_vs_dense_speedup
            );
        }
        Err(error) => {
            eprintln!("inferno dsa-attention benchmark failed: {error}");
            std::process::exit(1);
        }
    }
}
