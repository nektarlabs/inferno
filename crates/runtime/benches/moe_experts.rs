use backend::MetalBackend;
use runtime::{benchmark_moe_multi_expert_decode, MoeExpertBenchmarkConfig};

fn main() {
    let backend = match MetalBackend::new() {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("inferno moe-experts benchmark: native Metal backend unavailable: {error}");
            return;
        }
    };

    let config = MoeExpertBenchmarkConfig {
        iterations: 8,
        token_count: 1,
        assignment_count: 8,
        hidden_size: 6144,
        intermediate_size: 2048,
        expert_count: 8,
    };

    match benchmark_moe_multi_expert_decode(&backend, config) {
        Ok(report) => {
            eprintln!("inferno moe-experts benchmark");
            eprintln!("iterations: {}", report.iterations);
            eprintln!("token_count: {}", report.token_count);
            eprintln!("assignment_count: {}", report.assignment_count);
            eprintln!("hidden_size: {}", report.hidden_size);
            eprintln!("intermediate_size: {}", report.intermediate_size);
            eprintln!("expert_count: {}", report.expert_count);
            eprintln!(
                "old_single_assignment_seconds: {:.6}",
                report.old_single_assignment_seconds
            );
            eprintln!("multi_expert_seconds: {:.6}", report.multi_expert_seconds);
            eprintln!(
                "old_single_assignment_seconds_per_iteration: {:.6}",
                report.old_single_assignment_seconds_per_iteration
            );
            eprintln!(
                "multi_expert_seconds_per_iteration: {:.6}",
                report.multi_expert_seconds_per_iteration
            );
            eprintln!("scheduling_speedup: {:.3}", report.scheduling_speedup);
        }
        Err(error) => {
            eprintln!("inferno moe-experts benchmark failed: {error}");
            std::process::exit(1);
        }
    }
}
