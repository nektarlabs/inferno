#![deny(unsafe_code)]

//! GLM-5.2 production runtime orchestration.

mod telemetry;

use std::{
    fs::File,
    io::Write,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex, OnceLock,
    },
    time::{Duration, Instant},
};

use backend::Backend;
#[cfg(test)]
use backend::BackendCapabilities;
#[cfg(test)]
use cache::LayeredPagedCacheAppendReport;
use cache::{
    LayerKvCacheAppend, LayeredDiskPagedKvCache, LayeredPagedKvCache, LayeredPagedKvCacheSpec,
};
use common::{validate_exact_shape, Error, F32Tensor, PagedKvView, Result, Shape};
use config::Config;
#[cfg(test)]
use model::ModelGreedyOutput;
use model::{set_layer_profile_context, LayerKvCacheTensors, Model};
use telemetry::log_memory_snapshot;

pub const DEFAULT_KV_PAGE_SIZE: usize = 128;
pub const MAX_REFERENCE_PREFILL_ATTENTION_SCORE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub const MAX_REFERENCE_EXPANDED_KV_CACHE_BYTES: u64 = 24 * 1024 * 1024 * 1024;
const F32_BYTES: u64 = 4;

static Q2_RUNTIME_PROFILE_FILE: OnceLock<Mutex<Option<File>>> = OnceLock::new();
static Q2_RUNTIME_PROFILE_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn enable_memory_telemetry() {
    telemetry::enable_memory_telemetry();
}

pub fn enable_memory_telemetry_file(path: &Path) -> Result<()> {
    telemetry::enable_memory_telemetry_file(path)
}

pub fn enable_q2_runtime_profile(path: &Path) -> Result<()> {
    let mut file = File::create(path).map_err(|error| {
        Error::runtime(format!(
            "failed to create Q2 runtime profile file {}: {error}",
            path.display()
        ))
    })?;
    writeln!(file, "step_index\tlayer_index\tmetric\tvalue").map_err(|error| {
        Error::runtime(format!(
            "failed to write Q2 runtime profile header to {}: {error}",
            path.display()
        ))
    })?;

    let profile = q2_runtime_profile_file();
    let mut profile = profile
        .lock()
        .map_err(|_| Error::runtime("Q2 runtime profile lock poisoned"))?;
    if profile.is_some() {
        return Err(Error::runtime("Q2 runtime profile is already enabled"));
    }
    *profile = Some(file);
    Q2_RUNTIME_PROFILE_ENABLED.store(true, Ordering::Release);
    Ok(())
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct GenerationReport {
    pub capabilities: BackendCapabilities,
    pub page_size: usize,
    pub prompt_token_count: usize,
    pub generated_token_ids: Vec<u32>,
    pub total_token_count: usize,
    pub max_new_tokens: usize,
    pub memory: GenerationMemoryEstimate,
    pub prefill: GenerationStepReport,
    pub decode: DecodeLoopReport,
    pub cached_tokens: usize,
    pub next_position: usize,
    pub expert_loads: ExpertLoadReport,
    pub stop_reason: String,
    pub limitations: Vec<String>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct GenerationMemoryEstimate {
    pub batch: usize,
    pub prompt_tokens: usize,
    pub requested_context_tokens: usize,
    pub prefill_attention_scores_shape: Shape,
    pub prefill_attention_scores_bytes: u64,
    pub max_decode_attention_scores_shape: Shape,
    pub max_decode_attention_scores_bytes: u64,
    pub per_layer_k_cache_shape: Shape,
    pub per_layer_v_cache_shape: Shape,
    pub expanded_kv_cache_bytes: u64,
    pub max_prefill_attention_scores_bytes: u64,
    pub max_expanded_kv_cache_bytes: u64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct GenerationStepReport {
    pub step_index: usize,
    pub input_token_count: usize,
    pub hidden_states_shape: Shape,
    pub layer_kv_cache_count: usize,
    pub layer_k_cache_shape: Shape,
    pub layer_v_cache_shape: Shape,
    pub cache_append: PagedCacheAppendReport,
    pub cached_tokens: usize,
    pub next_position: usize,
    pub next_decode_attention_scores_shape: Shape,
    pub logits_shape: Shape,
    pub sampled_token_id: u32,
    pub sampled_token_score: f32,
    pub model: GenerationModelStepReport,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct GenerationModelStepReport {
    pub dense_layer_count: usize,
    pub sparse_layer_count: usize,
    pub max_attention_past_tokens: usize,
    pub materialized_full_logits: bool,
    pub output_projection_chunk_count: usize,
    pub output_projection_source_payload_bytes_read: u64,
    pub output_projection_peak_decoded_f32_bytes: u64,
    pub expert_loads: StepExpertLoadReport,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct StepExpertLoadReport {
    pub loaded_expert_requests: usize,
    pub materialized_expert_bytes_loaded: u64,
    pub source_expert_bytes_loaded: u64,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct DecodeLoopReport {
    pub step_count: usize,
    pub total_appended_tokens: usize,
    pub total_allocated_pages: usize,
    pub final_cached_tokens: usize,
    pub final_next_position: usize,
    pub max_attention_past_tokens: usize,
    pub materialized_full_logits: bool,
    pub total_output_projection_source_payload_bytes_read: u64,
    pub peak_output_projection_decoded_f32_bytes: u64,
    pub last_hidden_states_shape: Option<Shape>,
    pub last_logits_shape: Option<Shape>,
    pub last_next_decode_attention_scores_shape: Option<Shape>,
    pub last_sampled_token_id: Option<u32>,
    pub last_sampled_token_score: Option<f32>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct PagedCacheAppendReport {
    pub page_size: usize,
    pub layer_count: usize,
    pub start_position: usize,
    pub appended_tokens: usize,
    pub end_position_exclusive: usize,
    pub cached_tokens: usize,
    pub next_position: usize,
    pub allocated_pages: usize,
    pub page_count: usize,
    pub next_decode_attention_scores_shape: Shape,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
struct ExpertLoadReport {
    pub loaded_expert_requests: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub hit_rate: f32,
    pub materialized_expert_bytes_loaded: u64,
    pub source_expert_bytes_loaded: u64,
    pub limitations: Vec<String>,
}

pub fn run_generate_token_ids_with_stop_tokens<B: Backend>(
    model: &Model<'_>,
    config: &Config,
    backend: &B,
    prompt_token_ids: &[u32],
    max_new_tokens: Option<usize>,
    page_size: usize,
    stop_token_ids: &[u32],
) -> Result<Vec<u32>> {
    let mut generated_token_ids = Vec::new();
    run_generate_streaming_with_stop_tokens(
        model,
        config,
        backend,
        prompt_token_ids,
        max_new_tokens,
        page_size,
        stop_token_ids,
        |token_id| {
            generated_token_ids.push(token_id);
            Ok(())
        },
    )?;
    Ok(generated_token_ids)
}

pub fn run_generate_streaming_with_stop_tokens<B, F>(
    model: &Model<'_>,
    config: &Config,
    backend: &B,
    prompt_token_ids: &[u32],
    max_new_tokens: Option<usize>,
    page_size: usize,
    stop_token_ids: &[u32],
    mut on_token: F,
) -> Result<()>
where
    B: Backend,
    F: FnMut(u32) -> Result<()>,
{
    let generate_started_at = Instant::now();
    let effective_max_new_tokens =
        validate_generate_request(config, prompt_token_ids, max_new_tokens, page_size)?;
    log_memory_snapshot("generate.start", Some(0), backend, None);

    let prefill_started_at = Instant::now();
    set_layer_profile_context(0, "prefill")?;
    let prefill_output = model.prefill_next_token(config, prompt_token_ids, backend)?;
    record_q2_runtime_stage(0, None, "prefill.model", prefill_started_at.elapsed());
    log_memory_snapshot("prefill.after_model", Some(0), backend, None);
    let cache_started_at = Instant::now();
    let mut kv_cache = PagedRuntimeCache::new(
        model.max_context(),
        page_size,
        &prefill_output.layer_kv_cache,
        backend.capabilities().custom_kernels,
    )?;
    record_q2_runtime_stage(0, None, "prefill.cache_init", cache_started_at.elapsed());
    let cache_started_at = Instant::now();
    kv_cache.append_prefill(&prefill_output.layer_kv_cache)?;
    record_q2_runtime_stage(0, None, "prefill.cache_append", cache_started_at.elapsed());
    log_memory_snapshot(
        "prefill.after_cache_append",
        Some(0),
        backend,
        kv_cache.ssd_kv_bytes(),
    );

    let mut generated_token_count = 1_usize;
    on_token(prefill_output.token_id)?;
    if contains_stop_token(prefill_output.token_id, stop_token_ids) {
        log_memory_snapshot(
            "generate.stop_after_prefill",
            Some(0),
            backend,
            kv_cache.ssd_kv_bytes(),
        );
        record_q2_runtime_stage(0, None, "generate.total", generate_started_at.elapsed());
        return Ok(());
    }

    let mut next_input_ids = [prefill_output.token_id];
    for step_index in 1..effective_max_new_tokens {
        if max_new_tokens.is_none() {
            validate_reference_memory_bounds(
                config,
                prompt_token_ids.len(),
                generated_token_count,
            )?;
        }
        kv_cache.set_profile_step_index(step_index);
        log_memory_snapshot(
            "decode.before_model",
            Some(step_index),
            backend,
            kv_cache.ssd_kv_bytes(),
        );
        let decode_started_at = Instant::now();
        set_layer_profile_context(step_index, "decode")?;
        let decode_output = if backend.capabilities().custom_kernels {
            model.decode_next_token_with_paged_kv_provider(
                config,
                &next_input_ids,
                backend,
                |layer_index| kv_cache.paged_kv_for_layer(layer_index),
            )?
        } else {
            model.decode_next_token_with_past_kv_provider(
                config,
                &next_input_ids,
                backend,
                |layer_index| kv_cache.past_kv_for_layer(layer_index),
            )?
        };
        record_q2_runtime_stage(
            step_index,
            None,
            "decode.model",
            decode_started_at.elapsed(),
        );
        log_memory_snapshot(
            "decode.after_model",
            Some(step_index),
            backend,
            kv_cache.ssd_kv_bytes(),
        );
        let cache_started_at = Instant::now();
        kv_cache.append_decode(&decode_output.layer_kv_cache)?;
        record_q2_runtime_stage(
            step_index,
            None,
            "decode.cache_append",
            cache_started_at.elapsed(),
        );
        log_memory_snapshot(
            "decode.after_cache_append",
            Some(step_index),
            backend,
            kv_cache.ssd_kv_bytes(),
        );

        let token_id = decode_output.token_id;
        generated_token_count = generated_token_count
            .checked_add(1)
            .ok_or_else(|| Error::runtime("generated token count overflow"))?;
        on_token(token_id)?;
        if contains_stop_token(token_id, stop_token_ids) {
            break;
        }
        next_input_ids = [token_id];
    }

    log_memory_snapshot("generate.end", None, backend, kv_cache.ssd_kv_bytes());
    record_q2_runtime_stage(0, None, "generate.total", generate_started_at.elapsed());
    Ok(())
}

#[cfg(test)]
fn run_generate_with_stop_tokens<B: Backend>(
    model: &Model<'_>,
    config: &Config,
    backend: &B,
    prompt_token_ids: &[u32],
    max_new_tokens: usize,
    page_size: usize,
    stop_token_ids: &[u32],
) -> Result<GenerationReport> {
    let effective_max_new_tokens =
        validate_generate_request(config, prompt_token_ids, Some(max_new_tokens), page_size)?;
    let memory = estimate_generate_memory(config, prompt_token_ids.len(), max_new_tokens)?;

    let prefill_output = model.prefill_greedy(config, prompt_token_ids, backend)?;
    let mut kv_cache = PagedRuntimeCache::new(
        model.max_context(),
        page_size,
        &prefill_output.layer_kv_cache,
        backend.capabilities().custom_kernels,
    )?;
    let prefill_cache_append = kv_cache.append_prefill_report(&prefill_output.layer_kv_cache)?;
    let prefill_step = generation_step(
        0,
        prompt_token_ids.len(),
        prefill_output,
        prefill_cache_append,
    )?;

    let mut generated_token_ids = vec![prefill_step.sampled_token_id];
    let mut next_input_ids = [prefill_step.sampled_token_id];
    let mut decode = empty_decode_loop_report();
    let mut expert_loads = empty_expert_load_report();
    accumulate_expert_load_step(&mut expert_loads, &prefill_step);
    let mut stop_reason = if contains_stop_token(prefill_step.sampled_token_id, stop_token_ids) {
        "stop token reached".to_string()
    } else {
        "max_new_tokens reached".to_string()
    };

    for step_index in 1..effective_max_new_tokens {
        if stop_reason == "stop token reached" {
            break;
        }

        let decode_output = model.decode_step_greedy_with_past_kv_provider(
            config,
            &next_input_ids,
            backend,
            |layer_index| kv_cache.past_kv_for_layer(layer_index),
        )?;
        let decode_cache_append = kv_cache.append_decode_report(&decode_output.layer_kv_cache)?;
        let decode_step = generation_step(
            step_index,
            next_input_ids.len(),
            decode_output,
            decode_cache_append,
        )?;

        next_input_ids = [decode_step.sampled_token_id];
        generated_token_ids.push(decode_step.sampled_token_id);
        if contains_stop_token(decode_step.sampled_token_id, stop_token_ids) {
            stop_reason = "stop token reached".to_string();
        }
        accumulate_expert_load_step(&mut expert_loads, &decode_step);
        accumulate_decode_loop_report(&mut decode, &decode_step);
    }
    finalize_expert_load_report(&mut expert_loads);

    let total_token_count = prompt_token_ids
        .len()
        .checked_add(generated_token_ids.len())
        .ok_or_else(|| Error::runtime("generated token count overflow"))?;

    Ok(GenerationReport {
        capabilities: backend.capabilities(),
        page_size,
        prompt_token_count: prompt_token_ids.len(),
        generated_token_ids,
        total_token_count,
        max_new_tokens,
        memory,
        prefill: prefill_step,
        decode,
        cached_tokens: kv_cache.cached_tokens(),
        next_position: kv_cache.next_position(),
        expert_loads,
        stop_reason,
        limitations: vec![
            "GLM-5.2 generation runs the Q2 GGUF embedding, dense prefix, sparse MoE blocks, final norm, and lm_head".to_string(),
            "paged K/V cache is the only runtime cache mode".to_string(),
            "native Metal decode uses SSD-backed paged K/V as canonical storage; K/V is not retained in RAM or Metal buffers between layer attention calls".to_string(),
            "current-step K/V tensors still exist transiently while a layer is computed and appended to SSD".to_string(),
            "GLM DSA sparse attention is not active yet".to_string(),
            "sampling is greedy with tokenizer-derived stop-token support".to_string(),
        ],
    })
}

fn validate_generate_request(
    config: &Config,
    prompt_token_ids: &[u32],
    max_new_tokens: Option<usize>,
    page_size: usize,
) -> Result<usize> {
    if prompt_token_ids.is_empty() {
        return Err(Error::runtime(
            "GLM generation requires at least one prompt token",
        ));
    }
    if max_new_tokens == Some(0) {
        return Err(Error::runtime(
            "max_new_tokens must be positive when provided",
        ));
    }
    if page_size == 0 {
        return Err(Error::cache("paged KV cache page_size must be positive"));
    }
    if page_size > config.max_context {
        return Err(Error::cache(format!(
            "paged KV cache page_size {page_size} exceeds max_context {}",
            config.max_context
        )));
    }
    let effective_max_new_tokens =
        effective_max_new_tokens(config, prompt_token_ids.len(), max_new_tokens)?;
    let requested_context_tokens = prompt_token_ids
        .len()
        .checked_add(effective_max_new_tokens)
        .ok_or_else(|| Error::runtime("requested context token count overflow"))?;
    if requested_context_tokens > config.max_context {
        return Err(Error::runtime(format!(
            "requested context tokens {requested_context_tokens} exceed max_context {}",
            config.max_context
        )));
    }
    if let Some(token_id) = prompt_token_ids
        .iter()
        .copied()
        .find(|token_id| *token_id as usize >= config.vocab_size)
    {
        return Err(Error::tokenizer(format!(
            "prompt token id {token_id} is outside config vocab_size {}",
            config.vocab_size
        )));
    }
    let memory_check_new_tokens = if max_new_tokens.is_some() {
        effective_max_new_tokens
    } else {
        1
    };
    validate_reference_memory_bounds(config, prompt_token_ids.len(), memory_check_new_tokens)?;
    Ok(effective_max_new_tokens)
}

fn effective_max_new_tokens(
    config: &Config,
    prompt_token_count: usize,
    max_new_tokens: Option<usize>,
) -> Result<usize> {
    let effective = match max_new_tokens {
        Some(max_new_tokens) => max_new_tokens,
        None => config
            .max_context
            .checked_sub(prompt_token_count)
            .ok_or_else(|| {
                Error::runtime(format!(
                    "prompt token count {prompt_token_count} exceeds max_context {}",
                    config.max_context
                ))
            })?,
    };

    if effective == 0 {
        return Err(Error::runtime(format!(
            "prompt token count {prompt_token_count} leaves no room for generation within max_context {}",
            config.max_context
        )));
    }

    Ok(effective)
}

#[cfg(test)]
fn estimate_generate_memory(
    config: &Config,
    prompt_tokens: usize,
    max_new_tokens: usize,
) -> Result<GenerationMemoryEstimate> {
    let requested_context_tokens = prompt_tokens
        .checked_add(max_new_tokens)
        .ok_or_else(|| Error::runtime("requested context token count overflow"))?;
    let batch = 1;
    let value_head_dim = config.v_head_dim();

    let prefill_attention_scores_shape = Shape::new(vec![
        batch,
        config.attention_heads,
        prompt_tokens,
        prompt_tokens,
    ]);
    let prefill_attention_scores_bytes = checked_f32_bytes(
        "prefill attention scores",
        &[batch, config.attention_heads, prompt_tokens, prompt_tokens],
    )?;
    let max_decode_attention_scores_shape = Shape::new(vec![
        batch,
        config.attention_heads,
        1,
        requested_context_tokens,
    ]);
    let max_decode_attention_scores_bytes = checked_f32_bytes(
        "decode attention scores",
        &[batch, config.attention_heads, 1, requested_context_tokens],
    )?;
    let per_layer_k_cache_shape = Shape::new(vec![
        batch,
        config.attention_heads,
        requested_context_tokens,
        config.qk_head_dim,
    ]);
    let per_layer_v_cache_shape = Shape::new(vec![
        batch,
        config.attention_heads,
        requested_context_tokens,
        value_head_dim,
    ]);
    let expanded_kv_cache_bytes = checked_f32_bytes(
        "expanded KV cache",
        &[
            config.num_layers,
            batch,
            config.attention_heads,
            requested_context_tokens,
            config
                .qk_head_dim
                .checked_add(value_head_dim)
                .ok_or_else(|| Error::runtime("KV head dimension overflow"))?,
        ],
    )?;

    Ok(GenerationMemoryEstimate {
        batch,
        prompt_tokens,
        requested_context_tokens,
        prefill_attention_scores_shape,
        prefill_attention_scores_bytes,
        max_decode_attention_scores_shape,
        max_decode_attention_scores_bytes,
        per_layer_k_cache_shape,
        per_layer_v_cache_shape,
        expanded_kv_cache_bytes,
        max_prefill_attention_scores_bytes: MAX_REFERENCE_PREFILL_ATTENTION_SCORE_BYTES,
        max_expanded_kv_cache_bytes: MAX_REFERENCE_EXPANDED_KV_CACHE_BYTES,
    })
}

fn validate_reference_memory_bounds(
    config: &Config,
    prompt_tokens: usize,
    max_new_tokens: usize,
) -> Result<()> {
    let requested_context_tokens = prompt_tokens
        .checked_add(max_new_tokens)
        .ok_or_else(|| Error::runtime("requested context token count overflow"))?;
    let batch = 1;
    let value_head_dim = config.v_head_dim();

    let prefill_attention_scores_bytes = checked_f32_bytes(
        "prefill attention scores",
        &[batch, config.attention_heads, prompt_tokens, prompt_tokens],
    )?;
    if prefill_attention_scores_bytes > MAX_REFERENCE_PREFILL_ATTENTION_SCORE_BYTES {
        let shape = Shape::new(vec![
            batch,
            config.attention_heads,
            prompt_tokens,
            prompt_tokens,
        ]);
        return Err(Error::runtime(format!(
            "current dense prefill attention would allocate {prefill_attention_scores_bytes} bytes for attention scores {shape}; limit is {MAX_REFERENCE_PREFILL_ATTENTION_SCORE_BYTES} bytes until GLM DSA/paged attention kernels are active"
        )));
    }

    let key_value_head_dim = config
        .qk_head_dim
        .checked_add(value_head_dim)
        .ok_or_else(|| Error::runtime("KV head dimension overflow"))?;
    let expanded_kv_cache_bytes = checked_f32_bytes(
        "expanded KV cache",
        &[
            config.num_layers,
            batch,
            config.attention_heads,
            requested_context_tokens,
            key_value_head_dim,
        ],
    )?;
    if expanded_kv_cache_bytes > MAX_REFERENCE_EXPANDED_KV_CACHE_BYTES {
        let k_shape = Shape::new(vec![
            batch,
            config.attention_heads,
            requested_context_tokens,
            config.qk_head_dim,
        ]);
        let v_shape = Shape::new(vec![
            batch,
            config.attention_heads,
            requested_context_tokens,
            value_head_dim,
        ]);
        return Err(Error::runtime(format!(
            "current expanded F32 KV cache would allocate {expanded_kv_cache_bytes} bytes for K {k_shape} and V {v_shape} across all layers; limit is {MAX_REFERENCE_EXPANDED_KV_CACHE_BYTES} bytes until compressed/paged Metal KV is active"
        )));
    }

    Ok(())
}

fn checked_f32_bytes(context: &str, dims: &[usize]) -> Result<u64> {
    let mut elements = 1_u128;
    for dim in dims {
        elements = elements
            .checked_mul(*dim as u128)
            .ok_or_else(|| Error::runtime(format!("{context} element count overflow")))?;
    }
    let bytes = elements
        .checked_mul(F32_BYTES as u128)
        .ok_or_else(|| Error::runtime(format!("{context} byte count overflow")))?;
    u64::try_from(bytes)
        .map_err(|_| Error::runtime(format!("{context} byte count does not fit u64")))
}

#[cfg(test)]
fn generation_step(
    step_index: usize,
    input_token_count: usize,
    model_output: ModelGreedyOutput,
    cache_append: PagedCacheAppendReport,
) -> Result<GenerationStepReport> {
    let logits_shape = model_output.report.greedy.logits_shape.clone();
    let model = compact_model_step_report(&model_output);
    let (layer_kv_cache_count, layer_k_cache_shape, layer_v_cache_shape) =
        compact_layer_kv_cache_shapes(&model_output.layer_kv_cache)?;

    Ok(GenerationStepReport {
        step_index,
        input_token_count,
        hidden_states_shape: Shape::new(model_output.hidden_states.dims().to_vec()),
        layer_kv_cache_count,
        layer_k_cache_shape,
        layer_v_cache_shape,
        cached_tokens: cache_append.cached_tokens,
        next_position: cache_append.next_position,
        next_decode_attention_scores_shape: cache_append.next_decode_attention_scores_shape.clone(),
        cache_append,
        logits_shape,
        sampled_token_id: model_output.token_id,
        sampled_token_score: model_output.token_score,
        model,
    })
}

#[cfg(test)]
fn compact_layer_kv_cache_shapes(
    layer_kv_cache: &[LayerKvCacheTensors],
) -> Result<(usize, Shape, Shape)> {
    let first = layer_kv_cache
        .first()
        .ok_or_else(|| Error::runtime("model step produced no layer K/V tensors"))?;
    let k_shape = Shape::new(first.cache_k.dims().to_vec());
    let v_shape = Shape::new(first.cache_v.dims().to_vec());

    for entry in layer_kv_cache.iter().skip(1) {
        validate_exact_shape(
            format!("step_layer_{}_k_cache_shape", entry.layer_index),
            entry.cache_k.dims(),
            k_shape.dims(),
        )?;
        validate_exact_shape(
            format!("step_layer_{}_v_cache_shape", entry.layer_index),
            entry.cache_v.dims(),
            v_shape.dims(),
        )?;
    }

    Ok((layer_kv_cache.len(), k_shape, v_shape))
}

#[cfg(test)]
fn compact_model_step_report(model_output: &ModelGreedyOutput) -> GenerationModelStepReport {
    let layer_stack = &model_output.report.hidden.layer_stack;

    GenerationModelStepReport {
        dense_layer_count: layer_stack.dense_block_count,
        sparse_layer_count: layer_stack.sparse_block_count,
        max_attention_past_tokens: layer_stack.max_attention_past_tokens,
        materialized_full_logits: model_output.report.greedy.materialized_full_logits,
        output_projection_chunk_count: model_output.report.greedy.output_projection_chunk_count,
        output_projection_source_payload_bytes_read: model_output
            .report
            .greedy
            .output_projection_source_payload_bytes_read,
        output_projection_peak_decoded_f32_bytes: model_output
            .report
            .greedy
            .output_projection_peak_decoded_f32_bytes,
        expert_loads: StepExpertLoadReport {
            loaded_expert_requests: layer_stack.loaded_expert_requests,
            materialized_expert_bytes_loaded: layer_stack.routed_peak_decoded_f32_bytes,
            source_expert_bytes_loaded: layer_stack.routed_source_payload_bytes_read,
        },
    }
}

struct PagedRuntimeCache {
    storage: PagedRuntimeCacheStorage,
    profile_step_index: usize,
}

enum PagedRuntimeCacheStorage {
    Ssd(LayeredDiskPagedKvCache),
    Memory(LayeredPagedKvCache),
}

impl PagedRuntimeCache {
    fn new(
        max_context: usize,
        page_size: usize,
        layer_kv_cache: &[LayerKvCacheTensors],
        use_ssd_storage: bool,
    ) -> Result<Self> {
        let spec = cache_spec(max_context, page_size, layer_kv_cache)?;
        let storage = if use_ssd_storage {
            PagedRuntimeCacheStorage::Ssd(LayeredDiskPagedKvCache::new_in_temp(spec)?)
        } else {
            PagedRuntimeCacheStorage::Memory(LayeredPagedKvCache::new(spec)?)
        };
        Ok(Self {
            storage,
            profile_step_index: 0,
        })
    }

    fn set_profile_step_index(&mut self, step_index: usize) {
        self.profile_step_index = step_index;
    }

    #[cfg(test)]
    fn cached_tokens(&self) -> usize {
        match &self.storage {
            PagedRuntimeCacheStorage::Ssd(cache) => cache.cached_tokens(),
            PagedRuntimeCacheStorage::Memory(cache) => cache.cached_tokens(),
        }
    }

    #[cfg(test)]
    fn next_position(&self) -> usize {
        match &self.storage {
            PagedRuntimeCacheStorage::Ssd(cache) => cache.next_position(),
            PagedRuntimeCacheStorage::Memory(cache) => cache.next_position(),
        }
    }

    #[cfg(test)]
    fn storage_kind(&self) -> &'static str {
        match &self.storage {
            PagedRuntimeCacheStorage::Ssd(_) => "ssd",
            PagedRuntimeCacheStorage::Memory(_) => "memory",
        }
    }

    fn ssd_kv_bytes(&self) -> Option<u64> {
        match &self.storage {
            PagedRuntimeCacheStorage::Ssd(cache) => cache.stored_bytes().ok(),
            PagedRuntimeCacheStorage::Memory(_) => None,
        }
    }

    fn append_prefill(&mut self, layer_kv_cache: &[LayerKvCacheTensors]) -> Result<()> {
        let appends = layer_appends(layer_kv_cache);
        match &mut self.storage {
            PagedRuntimeCacheStorage::Ssd(cache) => {
                cache.append_prefill(&appends)?;
            }
            PagedRuntimeCacheStorage::Memory(cache) => {
                cache.append_prefill(&appends)?;
            }
        }
        Ok(())
    }

    fn append_decode(&mut self, layer_kv_cache: &[LayerKvCacheTensors]) -> Result<()> {
        let appends = layer_appends(layer_kv_cache);
        match &mut self.storage {
            PagedRuntimeCacheStorage::Ssd(cache) => {
                cache.append_decode(&appends)?;
            }
            PagedRuntimeCacheStorage::Memory(cache) => {
                cache.append_decode(&appends)?;
            }
        }
        Ok(())
    }

    #[cfg(test)]
    fn append_prefill_report(
        &mut self,
        layer_kv_cache: &[LayerKvCacheTensors],
    ) -> Result<PagedCacheAppendReport> {
        let appends = layer_appends(layer_kv_cache);
        match &mut self.storage {
            PagedRuntimeCacheStorage::Ssd(cache) => {
                paged_append_report(cache.append_prefill(&appends)?, cache.spec().page_size)
            }
            PagedRuntimeCacheStorage::Memory(cache) => {
                paged_append_report(cache.append_prefill(&appends)?, cache.spec().page_size)
            }
        }
    }

    #[cfg(test)]
    fn append_decode_report(
        &mut self,
        layer_kv_cache: &[LayerKvCacheTensors],
    ) -> Result<PagedCacheAppendReport> {
        let appends = layer_appends(layer_kv_cache);
        match &mut self.storage {
            PagedRuntimeCacheStorage::Ssd(cache) => {
                paged_append_report(cache.append_decode(&appends)?, cache.spec().page_size)
            }
            PagedRuntimeCacheStorage::Memory(cache) => {
                paged_append_report(cache.append_decode(&appends)?, cache.spec().page_size)
            }
        }
    }

    fn past_kv_for_layer(&self, layer_index: usize) -> Result<Option<(F32Tensor, F32Tensor)>> {
        let started_at = Instant::now();
        let PagedRuntimeCacheStorage::Memory(cache) = &self.storage else {
            return Err(Error::runtime(
                "SSD KV cache does not support expanded contiguous K/V reconstruction",
            ));
        };
        let (keys, values) = cache.reconstruct_layer_kv(layer_index)?;
        record_q2_runtime_stage(
            self.profile_step_index,
            Some(layer_index),
            "decode.kv_reconstruct",
            started_at.elapsed(),
        );
        Ok(Some((keys, values)))
    }

    fn paged_kv_for_layer(&self, layer_index: usize) -> Result<Option<PagedKvView<'_>>> {
        let started_at = Instant::now();
        let view = match &self.storage {
            PagedRuntimeCacheStorage::Ssd(cache) => cache.layer_view_owned(layer_index)?.view,
            PagedRuntimeCacheStorage::Memory(cache) => cache.layer_view(layer_index)?.view,
        };
        record_q2_runtime_stage(
            self.profile_step_index,
            Some(layer_index),
            "decode.kv_page_view",
            started_at.elapsed(),
        );
        Ok(Some(view))
    }
}

fn cache_spec(
    max_context: usize,
    page_size: usize,
    layer_kv_cache: &[LayerKvCacheTensors],
) -> Result<LayeredPagedKvCacheSpec> {
    let first_layer = layer_kv_cache
        .first()
        .ok_or_else(|| Error::runtime("prefill produced no layer K/V tensors"))?;
    let k_dims = first_layer.cache_k.dims();
    let v_dims = first_layer.cache_v.dims();
    if k_dims.len() != 4 || v_dims.len() != 4 {
        return Err(Error::runtime(format!(
            "K/V tensors must be rank 4 [B,H,T,D], got k={k_dims:?} v={v_dims:?}"
        )));
    }
    validate_exact_shape(
        "cache_batch_heads_tokens",
        &[v_dims[0], v_dims[1], v_dims[2]],
        &[k_dims[0], k_dims[1], k_dims[2]],
    )?;

    Ok(LayeredPagedKvCacheSpec {
        batch: k_dims[0],
        attention_heads: k_dims[1],
        key_head_dim: k_dims[3],
        value_head_dim: v_dims[3],
        max_context,
        page_size,
    })
}

#[cfg(test)]
fn paged_append_report(
    report: LayeredPagedCacheAppendReport,
    page_size: usize,
) -> Result<PagedCacheAppendReport> {
    Ok(PagedCacheAppendReport {
        page_size,
        layer_count: report.layer_count,
        start_position: report.start_position,
        appended_tokens: report.appended_tokens,
        end_position_exclusive: report.end_position_exclusive,
        cached_tokens: report.cached_tokens,
        next_position: report.next_position,
        allocated_pages: report.allocated_pages,
        page_count: report.page_count,
        next_decode_attention_scores_shape: report.next_decode_attention_scores_shape,
    })
}

fn layer_appends(layer_kv_cache: &[LayerKvCacheTensors]) -> Vec<LayerKvCacheAppend<'_>> {
    layer_kv_cache
        .iter()
        .map(|entry| LayerKvCacheAppend {
            layer_index: entry.layer_index,
            k: &entry.cache_k,
            v: &entry.cache_v,
        })
        .collect()
}

#[cfg(test)]
fn empty_expert_load_report() -> ExpertLoadReport {
    ExpertLoadReport {
        loaded_expert_requests: 0,
        cache_hits: 0,
        cache_misses: 0,
        hit_rate: 0.0,
        materialized_expert_bytes_loaded: 0,
        source_expert_bytes_loaded: 0,
        limitations: vec![
            "GLM routed experts use direct Q2 payload dispatch; fused Metal expert dispatch is not active yet"
                .to_string(),
        ],
    }
}

#[cfg(test)]
fn finalize_expert_load_report(report: &mut ExpertLoadReport) {
    report.loaded_expert_requests = report.cache_hits.saturating_add(report.cache_misses);
    if report.loaded_expert_requests > 0 {
        report.hit_rate = report.cache_hits as f32 / report.loaded_expert_requests as f32;
    }
}

#[cfg(test)]
fn accumulate_expert_load_step(report: &mut ExpertLoadReport, step: &GenerationStepReport) {
    report.cache_misses = report
        .cache_misses
        .saturating_add(step.model.expert_loads.loaded_expert_requests);
    report.materialized_expert_bytes_loaded = report
        .materialized_expert_bytes_loaded
        .saturating_add(step.model.expert_loads.materialized_expert_bytes_loaded);
    report.source_expert_bytes_loaded = report
        .source_expert_bytes_loaded
        .saturating_add(step.model.expert_loads.source_expert_bytes_loaded);
}

#[cfg(test)]
fn empty_decode_loop_report() -> DecodeLoopReport {
    DecodeLoopReport {
        step_count: 0,
        total_appended_tokens: 0,
        total_allocated_pages: 0,
        final_cached_tokens: 0,
        final_next_position: 0,
        max_attention_past_tokens: 0,
        materialized_full_logits: false,
        total_output_projection_source_payload_bytes_read: 0,
        peak_output_projection_decoded_f32_bytes: 0,
        last_hidden_states_shape: None,
        last_logits_shape: None,
        last_next_decode_attention_scores_shape: None,
        last_sampled_token_id: None,
        last_sampled_token_score: None,
    }
}

#[cfg(test)]
fn accumulate_decode_loop_report(report: &mut DecodeLoopReport, step: &GenerationStepReport) {
    report.step_count = report.step_count.saturating_add(1);
    report.total_appended_tokens = report
        .total_appended_tokens
        .saturating_add(step.cache_append.appended_tokens);
    report.total_allocated_pages = report
        .total_allocated_pages
        .saturating_add(step.cache_append.allocated_pages);
    report.final_cached_tokens = step.cached_tokens;
    report.final_next_position = step.next_position;
    report.max_attention_past_tokens = report
        .max_attention_past_tokens
        .max(step.model.max_attention_past_tokens);
    report.materialized_full_logits =
        report.materialized_full_logits || step.model.materialized_full_logits;
    report.total_output_projection_source_payload_bytes_read = report
        .total_output_projection_source_payload_bytes_read
        .saturating_add(step.model.output_projection_source_payload_bytes_read);
    report.peak_output_projection_decoded_f32_bytes = report
        .peak_output_projection_decoded_f32_bytes
        .max(step.model.output_projection_peak_decoded_f32_bytes);
    report.last_hidden_states_shape = Some(step.hidden_states_shape.clone());
    report.last_logits_shape = Some(step.logits_shape.clone());
    report.last_next_decode_attention_scores_shape =
        Some(step.next_decode_attention_scores_shape.clone());
    report.last_sampled_token_id = Some(step.sampled_token_id);
    report.last_sampled_token_score = Some(step.sampled_token_score);
}

fn contains_stop_token(token_id: u32, stop_token_ids: &[u32]) -> bool {
    !stop_token_ids.is_empty() && stop_token_ids.contains(&token_id)
}

fn record_q2_runtime_stage(
    step_index: usize,
    layer_index: Option<usize>,
    stage: &str,
    elapsed: Duration,
) {
    let value = format!("{:.3}", elapsed.as_secs_f64() * 1000.0);
    record_q2_runtime_value(step_index, layer_index, stage, &value);
}

fn record_q2_runtime_value(
    step_index: usize,
    layer_index: Option<usize>,
    metric: &str,
    value: &str,
) {
    if !Q2_RUNTIME_PROFILE_ENABLED.load(Ordering::Acquire) {
        return;
    }
    let Ok(mut profile) = q2_runtime_profile_file().lock() else {
        return;
    };
    let Some(file) = profile.as_mut() else {
        return;
    };
    let layer_index = layer_index
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_string());
    let _ = writeln!(file, "{step_index}\t{layer_index}\t{metric}\t{value}",);
}

fn q2_runtime_profile_file() -> &'static Mutex<Option<File>> {
    Q2_RUNTIME_PROFILE_FILE.get_or_init(|| Mutex::new(None))
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use backend::MetalBackend;
    use gguf::{
        GgmlType, GgufFile, GgufMetadataValueType, GGML_Q2_K_BLOCK_BYTES, GGML_Q8_0_BLOCK_BYTES,
        GGUF_MAGIC, GGUF_VERSION_V3,
    };
    use model::{Model, DEFAULT_GGUF_OUTPUT_CHUNK_ROWS};

    use super::*;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn q2_runtime_profile_is_disabled_by_default() {
        assert!(!Q2_RUNTIME_PROFILE_ENABLED.load(Ordering::Acquire));
        assert!(q2_runtime_profile_file().lock().unwrap().is_none());
    }

    #[test]
    fn memory_telemetry_can_be_enabled_explicitly() {
        enable_memory_telemetry();

        assert!(telemetry::memory_telemetry_enabled());
    }

    #[test]
    fn native_runtime_cache_uses_ssd_storage() {
        let layer_kv_cache = tiny_layer_kv_cache(1);
        let ssd =
            PagedRuntimeCache::new(8, 2, &layer_kv_cache, true).expect("SSD cache should build");
        let memory = PagedRuntimeCache::new(8, 2, &layer_kv_cache, false)
            .expect("memory cache should build");

        assert_eq!(ssd.storage_kind(), "ssd");
        assert_eq!(memory.storage_kind(), "memory");
    }

    #[test]
    fn generate_uses_q2_gguf_model_and_paged_cache() {
        let path = write_gguf_model_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let backend = MetalBackend::reference().unwrap();
        let model = Model::open_from_gguf(&gguf, &config, &backend, DEFAULT_GGUF_OUTPUT_CHUNK_ROWS)
            .unwrap();

        let report =
            run_generate_with_stop_tokens(&model, &config, &backend, &[1, 2], 2, 1, &[]).unwrap();

        assert_eq!(report.page_size, 1);
        assert_eq!(report.prompt_token_count, 2);
        assert_eq!(report.generated_token_ids.len(), 2);
        assert_eq!(report.total_token_count, 4);
        assert_eq!(report.prefill.input_token_count, 2);
        assert_eq!(report.prefill.layer_kv_cache_count, 2);
        assert_eq!(report.prefill.layer_k_cache_shape.dims(), &[1, 2, 2, 256]);
        assert_eq!(report.prefill.layer_v_cache_shape.dims(), &[1, 2, 2, 256]);
        assert_eq!(report.prefill.cache_append.layer_count, 2);
        assert_eq!(report.prefill.cache_append.appended_tokens, 2);
        assert_eq!(report.prefill.cache_append.allocated_pages, 4);
        assert_eq!(report.prefill.hidden_states_shape.dims(), &[1, 2, 256]);
        assert_eq!(report.prefill.logits_shape.dims(), &[1, 8]);
        assert_eq!(report.prefill.model.dense_layer_count, 1);
        assert_eq!(report.prefill.model.sparse_layer_count, 1);
        assert_eq!(report.prefill.model.max_attention_past_tokens, 0);
        assert!(!report.prefill.model.materialized_full_logits);
        assert_eq!(report.prefill.model.output_projection_chunk_count, 1);
        assert!(
            report
                .prefill
                .model
                .output_projection_source_payload_bytes_read
                > 0
        );
        assert!(
            report
                .prefill
                .model
                .output_projection_peak_decoded_f32_bytes
                > 0
        );
        assert!(report.prefill.model.expert_loads.loaded_expert_requests > 0);
        assert_eq!(report.decode.step_count, 1);
        assert_eq!(report.decode.total_appended_tokens, 1);
        assert_eq!(report.decode.total_allocated_pages, 2);
        assert_eq!(report.decode.max_attention_past_tokens, 2);
        assert!(!report.decode.materialized_full_logits);
        assert_eq!(
            report
                .decode
                .last_hidden_states_shape
                .as_ref()
                .unwrap()
                .dims(),
            &[1, 1, 256]
        );
        assert_eq!(
            report.decode.last_logits_shape.as_ref().unwrap().dims(),
            &[1, 8]
        );
        assert!(
            report
                .decode
                .total_output_projection_source_payload_bytes_read
                > 0
        );
        assert!(report.decode.peak_output_projection_decoded_f32_bytes > 0);
        assert_eq!(report.cached_tokens, 3);
        assert_eq!(report.next_position, 3);
        assert_eq!(report.memory.prompt_tokens, 2);
        assert_eq!(report.memory.requested_context_tokens, 4);
        assert_eq!(
            report.memory.prefill_attention_scores_shape.dims(),
            &[1, 2, 2, 2]
        );
        assert_eq!(report.memory.prefill_attention_scores_bytes, 32);
        assert_eq!(
            report.memory.per_layer_k_cache_shape.dims(),
            &[1, 2, 4, 256]
        );
        assert_eq!(
            report.memory.per_layer_v_cache_shape.dims(),
            &[1, 2, 4, 256]
        );
        assert_eq!(report.memory.expanded_kv_cache_bytes, 32_768);
        assert_eq!(report.expert_loads.cache_hits, 0);
        assert!(report.expert_loads.cache_misses > 0);
        assert!(report.expert_loads.materialized_expert_bytes_loaded > 0);
        assert!(report.expert_loads.source_expert_bytes_loaded > 0);
    }

    #[test]
    fn generate_token_ids_uses_lean_runtime_path() {
        let path = write_gguf_model_fixture(GgmlType::Q2K);
        let gguf = GgufFile::open(&path).unwrap();
        let config = tiny_config();
        let backend = MetalBackend::reference().unwrap();
        let model = Model::open_from_gguf(&gguf, &config, &backend, DEFAULT_GGUF_OUTPUT_CHUNK_ROWS)
            .unwrap();

        let report =
            run_generate_with_stop_tokens(&model, &config, &backend, &[1, 2], 2, 1, &[]).unwrap();
        let token_ids = run_generate_token_ids_with_stop_tokens(
            &model,
            &config,
            &backend,
            &[1, 2],
            Some(2),
            1,
            &[],
        )
        .unwrap();

        assert_eq!(token_ids, report.generated_token_ids);

        let mut streamed_token_ids = Vec::new();
        run_generate_streaming_with_stop_tokens(
            &model,
            &config,
            &backend,
            &[1, 2],
            Some(2),
            1,
            &[],
            |token_id| {
                streamed_token_ids.push(token_id);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(streamed_token_ids, token_ids);

        let stopped = run_generate_token_ids_with_stop_tokens(
            &model,
            &config,
            &backend,
            &[1, 2],
            Some(2),
            1,
            &[token_ids[0]],
        )
        .unwrap();

        assert_eq!(stopped, vec![token_ids[0]]);
    }

    #[test]
    fn generate_validates_request_shape() {
        let config = tiny_config();
        let err = validate_generate_request(&config, &[1, 2], Some(0), 1)
            .expect_err("zero generated token count must fail");

        assert!(err.to_string().contains("max_new_tokens"));
    }

    #[test]
    fn generate_without_token_limit_uses_remaining_context() {
        let mut config = tiny_config();
        config.max_context = 5;

        let effective = validate_generate_request(&config, &[1, 2], None, 1).unwrap();

        assert_eq!(effective, 3);
    }

    #[test]
    fn generate_without_token_limit_requires_context_room() {
        let mut config = tiny_config();
        config.max_context = 2;

        let err = validate_generate_request(&config, &[1, 2], None, 1)
            .expect_err("full context should leave no room for generation");

        assert!(err.to_string().contains("leaves no room"));
    }

    #[test]
    fn generate_rejects_unsafe_dense_prefill_attention() {
        let mut config = tiny_config();
        config.max_context = 20_000;
        let prompt = vec![1_u32; 17_000];

        let err = validate_generate_request(&config, &prompt, Some(1), 128)
            .expect_err("large dense prefill attention should be rejected");

        assert!(err.to_string().contains("dense prefill attention"));
        assert!(err.to_string().contains("[1, 2, 17000, 17000]"));
    }

    #[test]
    fn generate_rejects_unsafe_expanded_kv_cache() {
        let mut config = tiny_config();
        config.num_layers = 78;
        config.attention_heads = 64;
        config.max_context = 5_000;
        let prompt = vec![1_u32; 16];

        let err = validate_generate_request(&config, &prompt, Some(3_000), 128)
            .expect_err("large expanded F32 KV cache should be rejected");

        assert!(err.to_string().contains("expanded F32 KV cache"));
        assert!(err.to_string().contains("[1, 64, 3016, 256]"));
    }

    fn write_gguf_model_fixture(ty: GgmlType) -> PathBuf {
        let path = unique_temp_file("runtime-gguf-model");
        let mut specs = vec![
            TensorSpec::quant("token_embd.weight", vec![256, 8], ty),
            TensorSpec::f32("output_norm.weight", vec![256], None),
            TensorSpec::quant("output.weight", vec![256, 8], ty),
        ];
        insert_dense_layer(&mut specs, 0, ty);
        insert_sparse_layer(&mut specs, 1, ty);
        write_specs_gguf(path, &specs)
    }

    fn insert_dense_layer(specs: &mut Vec<TensorSpec>, layer: usize, ty: GgmlType) {
        insert_attention(specs, layer, ty);
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_gate.weight"),
            vec![256, 512],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_up.weight"),
            vec![256, 512],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_down.weight"),
            vec![512, 256],
            ty,
        ));
    }

    fn insert_sparse_layer(specs: &mut Vec<TensorSpec>, layer: usize, ty: GgmlType) {
        insert_attention(specs, layer, ty);
        specs.push(TensorSpec::f32(
            format!("blk.{layer}.exp_probs_b.bias"),
            vec![4],
            Some(vec![0.0, 0.4, 0.2, 0.8]),
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_gate_inp.weight"),
            vec![256, 4],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_gate_shexp.weight"),
            vec![256, 256],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_up_shexp.weight"),
            vec![256, 256],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_down_shexp.weight"),
            vec![256, 256],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_gate_exps.weight"),
            vec![256, 256, 4],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_up_exps.weight"),
            vec![256, 256, 4],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.ffn_down_exps.weight"),
            vec![256, 256, 4],
            ty,
        ));
    }

    fn insert_attention(specs: &mut Vec<TensorSpec>, layer: usize, ty: GgmlType) {
        specs.push(TensorSpec::f32(
            format!("blk.{layer}.attn_norm.weight"),
            vec![256],
            None,
        ));
        specs.push(TensorSpec::f32(
            format!("blk.{layer}.ffn_norm.weight"),
            vec![256],
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_q_a.weight"),
            vec![256, 256],
            ty,
        ));
        specs.push(TensorSpec::f32(
            format!("blk.{layer}.attn_q_a_norm.weight"),
            vec![256],
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_q_b.weight"),
            vec![256, 512],
            ty,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_kv_a_mqa.weight"),
            vec![256, 384],
            ty,
        ));
        specs.push(TensorSpec::f32(
            format!("blk.{layer}.attn_kv_a_norm.weight"),
            vec![256],
            None,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_k_b.weight"),
            vec![128, 256, 2],
            GgmlType::Q8_0,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_v_b.weight"),
            vec![256, 256, 2],
            GgmlType::Q8_0,
        ));
        specs.push(TensorSpec::quant(
            format!("blk.{layer}.attn_output.weight"),
            vec![512, 256],
            ty,
        ));
    }

    #[derive(Debug)]
    struct TensorSpec {
        name: String,
        dims: Vec<u64>,
        ty: GgmlType,
        values: Option<Vec<f32>>,
    }

    impl TensorSpec {
        fn f32(name: impl Into<String>, dims: Vec<u64>, values: Option<Vec<f32>>) -> Self {
            Self {
                name: name.into(),
                dims,
                ty: GgmlType::F32,
                values,
            }
        }

        fn quant(name: impl Into<String>, dims: Vec<u64>, ty: GgmlType) -> Self {
            Self {
                name: name.into(),
                dims,
                ty,
                values: None,
            }
        }

        fn payload_len(&self) -> u64 {
            match self.ty {
                GgmlType::F32 => self.dims.iter().product::<u64>() * 4,
                GgmlType::Q2K => self.dims.iter().product::<u64>() / 256 * GGML_Q2_K_BLOCK_BYTES,
                GgmlType::Q8_0 => self.dims.iter().product::<u64>() / 32 * GGML_Q8_0_BLOCK_BYTES,
                other => panic!("unsupported fixture tensor type {other}"),
            }
        }
    }

    fn write_specs_gguf(path: PathBuf, specs: &[TensorSpec]) -> PathBuf {
        let mut writer = GgufWriter::new();
        writer.header(specs.len() as u64, 2);
        writer.metadata_key("general.architecture");
        writer.u32(GgufMetadataValueType::String as u32);
        writer.string("glm-dsa");
        writer.metadata_key("general.alignment");
        writer.u32(GgufMetadataValueType::Uint32 as u32);
        writer.u32(32);

        let mut offset = 0_u64;
        let offsets = specs
            .iter()
            .map(|spec| {
                let current = offset;
                offset = align_u64(current + spec.payload_len(), 32);
                current
            })
            .collect::<Vec<_>>();
        for (spec, offset) in specs.iter().zip(offsets.iter().copied()) {
            writer.tensor_info(&spec.name, &spec.dims, spec.ty, offset);
        }
        writer.pad_to(32);

        for (spec, offset) in specs.iter().zip(offsets.iter().copied()) {
            writer.pad_to_absolute_data_offset(offset);
            match spec.ty {
                GgmlType::F32 => {
                    let element_count = spec.dims.iter().product::<u64>() as usize;
                    let values = spec
                        .values
                        .clone()
                        .unwrap_or_else(|| vec![1.0_f32; element_count]);
                    assert_eq!(values.len(), element_count);
                    for value in values {
                        writer.bytes(&value.to_le_bytes());
                    }
                }
                GgmlType::Q2K | GgmlType::Q8_0 => {
                    writer.bytes(&vec![0_u8; spec.payload_len() as usize]);
                }
                other => panic!("unsupported fixture tensor type {other}"),
            }
        }
        writer.pad_to_absolute_data_offset(offset);
        writer.finish_to(path)
    }

    struct GgufWriter {
        bytes: Vec<u8>,
        data_start: Option<usize>,
    }

    impl GgufWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                data_start: None,
            }
        }

        fn header(&mut self, tensor_count: u64, metadata_kv_count: u64) {
            self.bytes.extend_from_slice(GGUF_MAGIC);
            self.u32(GGUF_VERSION_V3);
            self.u64(tensor_count);
            self.u64(metadata_kv_count);
        }

        fn metadata_key(&mut self, key: &str) {
            self.string(key);
        }

        fn tensor_info(&mut self, name: &str, dims: &[u64], ty: GgmlType, offset: u64) {
            self.string(name);
            self.u32(dims.len() as u32);
            for dim in dims {
                self.u64(*dim);
            }
            self.u32(ty.code());
            self.u64(offset);
        }

        fn string(&mut self, value: &str) {
            self.u64(value.len() as u64);
            self.bytes.extend_from_slice(value.as_bytes());
        }

        fn u32(&mut self, value: u32) {
            self.bytes.extend_from_slice(&value.to_le_bytes());
        }

        fn u64(&mut self, value: u64) {
            self.bytes.extend_from_slice(&value.to_le_bytes());
        }

        fn pad_to(&mut self, alignment: usize) {
            let remainder = self.bytes.len() % alignment;
            if remainder != 0 {
                self.bytes
                    .resize(self.bytes.len() + alignment - remainder, 0);
            }
            self.data_start = Some(self.bytes.len());
        }

        fn pad_to_absolute_data_offset(&mut self, offset: u64) {
            let target = self.data_start.unwrap() + offset as usize;
            if self.bytes.len() < target {
                self.bytes.resize(target, 0);
            }
        }

        fn bytes(&mut self, bytes: &[u8]) {
            self.bytes.extend_from_slice(bytes);
        }

        fn finish_to(self, path: PathBuf) -> PathBuf {
            fs::write(&path, self.bytes).unwrap();
            path
        }
    }

    fn align_u64(value: u64, alignment: u64) -> u64 {
        let remainder = value % alignment;
        if remainder == 0 {
            value
        } else {
            value + alignment - remainder
        }
    }

    fn unique_temp_file(label: &str) -> PathBuf {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("runtime-{label}-{}-{id}", std::process::id()))
    }

    fn tiny_layer_kv_cache(tokens: usize) -> Vec<LayerKvCacheTensors> {
        vec![LayerKvCacheTensors {
            layer_index: 0,
            layer_kind: model::LayerKind::Dense,
            cache_k: F32Tensor::zeros([1, 2, tokens, 256]).unwrap(),
            cache_v: F32Tensor::zeros([1, 2, tokens, 256]).unwrap(),
        }]
    }

    fn tiny_config() -> Config {
        Config {
            model_type: "glm_moe_dsa".to_string(),
            hidden_size: 256,
            num_layers: 2,
            dense_layers: 1,
            sparse_moe_layers: Some(1),
            vocab_size: 8,
            attention_heads: 2,
            qk_head_dim: 256,
            qk_no_rope_dim: 128,
            qk_rope_dim: 128,
            v_head_dim: Some(256),
            num_routed_experts: 4,
            experts_per_token: 2,
            max_context: 128,
            dsa_index_topk: 16,
            moe_intermediate_size: 256,
            num_shared_experts: 1,
            moe_groups: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 2.5,
            scoring_func: "sigmoid".to_string(),
            topk_method: "noaux_tc".to_string(),
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
        .validated()
        .unwrap()
    }
}
