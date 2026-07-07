use backend::MetalBackend;
use runtime::{benchmark_ssd_kv_decode_hot_load, SsdKvBenchmarkConfig};

fn main() {
    let Ok(backend) = MetalBackend::new() else {
        eprintln!("inferno ssd-kv benchmark: native Metal backend unavailable");
        return;
    };

    let config = SsdKvBenchmarkConfig {
        layer_count: 2,
        batch: 1,
        attention_heads: 8,
        tokens: 256,
        key_head_dim: 256,
        value_head_dim: 256,
        block_tokens: 128,
        page_size: 128,
    };

    match benchmark_ssd_kv_decode_hot_load(&backend, config) {
        Ok(report) => {
            eprintln!("inferno ssd-kv benchmark");
            eprintln!("layers: {}", report.layer_count);
            eprintln!("tokens: {}", report.tokens);
            eprintln!("stored_bytes: {}", report.stored_bytes);
            eprintln!("raw_f32_bytes: {}", report.raw_f32_bytes);
            eprintln!("compression_ratio: {:.4}", report.compression_ratio);
            eprintln!("write_seconds: {:.6}", report.write_seconds);
            eprintln!("read_seconds: {:.6}", report.read_seconds);
            eprintln!("hot_load_seconds: {:.6}", report.hot_load_seconds);
            eprintln!(
                "ssd_read_gb_per_second: {:.3}",
                report.ssd_read_gb_per_second
            );
        }
        Err(error) => {
            eprintln!("inferno ssd-kv benchmark failed: {error}");
            std::process::exit(1);
        }
    }
}
