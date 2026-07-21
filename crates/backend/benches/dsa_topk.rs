use std::time::Instant;

use backend::{Backend, MetalBackend};
use common::F32Tensor;

const KEY_TOKENS: usize = 65_536;
const TOP_K: usize = 2_048;
const ITERATIONS: usize = 3;

fn main() {
    let backend = match MetalBackend::new() {
        Ok(backend) => backend,
        Err(error) => {
            eprintln!("inferno dsa-topk benchmark: native Metal backend unavailable: {error}");
            return;
        }
    };

    let hidden = F32Tensor::new(vec![1.0], [1, 1, 1]).expect("valid hidden shape");
    let q_raw = F32Tensor::new(vec![1.0, 0.0], [1, 1, 2]).expect("valid query shape");
    let mut past_values = Vec::with_capacity((KEY_TOKENS - 1) * 2);
    for token in 0..KEY_TOKENS - 1 {
        past_values.push((token + 1) as f32);
        past_values.push(0.0);
    }
    let past_keys =
        F32Tensor::new(past_values, [1, KEY_TOKENS - 1, 2]).expect("valid past-key shape");
    let current_key = F32Tensor::new(vec![KEY_TOKENS as f32 + 1.0, 0.0], [1, 1, 2])
        .expect("valid current-key shape");
    let weights_proj = F32Tensor::new(vec![1.0], [1, 1]).expect("valid weight shape");

    let hidden = backend
        .device_upload_f32_tensor(&hidden)
        .expect("hidden upload")
        .expect("Metal hidden value");
    let q_raw = backend
        .device_upload_f32_tensor(&q_raw)
        .expect("query upload")
        .expect("Metal query value");
    let past_keys = backend
        .device_upload_f32_tensor(&past_keys)
        .expect("past-key upload")
        .expect("Metal past-key value");
    let current_key = backend
        .device_upload_f32_tensor(&current_key)
        .expect("current-key upload")
        .expect("Metal current-key value");

    let run_once = || {
        backend
            .dsa_decode_topk_device(
                &hidden,
                &q_raw,
                &past_keys,
                &current_key,
                &weights_proj,
                1,
                2,
                2,
                0,
                10_000.0,
                TOP_K,
            )
            .expect("DSA top-k execution")
            .expect("native Metal DSA top-k")
    };

    let warmup = run_once();
    validate_result(&warmup);
    let started_at = Instant::now();
    for _ in 0..ITERATIONS {
        validate_result(&run_once());
    }
    let elapsed = started_at.elapsed().as_secs_f64();

    eprintln!("inferno dsa-topk benchmark");
    eprintln!("key_tokens: {KEY_TOKENS}");
    eprintln!("top_k: {TOP_K}");
    eprintln!("iterations: {ITERATIONS}");
    eprintln!("total_seconds: {elapsed:.6}");
    eprintln!(
        "milliseconds_per_selection: {:.3}",
        elapsed * 1_000.0 / ITERATIONS as f64
    );
}

fn validate_result(token_ids: &[u32]) {
    assert_eq!(token_ids.len(), TOP_K);
    assert_eq!(token_ids[0], (KEY_TOKENS - 1) as u32);
    assert_eq!(token_ids[1], (KEY_TOKENS - 2) as u32);
    assert_eq!(token_ids[TOP_K - 1], (KEY_TOKENS - TOP_K) as u32);
}
