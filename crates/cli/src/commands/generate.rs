use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    time::Duration,
    time::Instant,
};

use anyhow::Result;
use backend::{Backend, ExpertCacheMetrics, MetalBackend};
use common::{Error, Result as InfernoResult};
use config::{load_config, Config};
use gguf::GgufFile;
use model::{
    antirez_q2_artifact, enable_layer_profile, FfnIndex, Index, IndexSummary, Model,
    DEFAULT_GGUF_OUTPUT_CHUNK_ROWS,
};
use runtime::{
    enable_memory_telemetry, enable_memory_telemetry_file, enable_q2_runtime_profile,
    run_generate_streaming_with_options, GenerationOptions, KvCacheMetrics,
};
use tokenizer::{render_user_prompt, Tokenizer, TokenizerMetadata};

pub fn run(
    model_path: &Path,
    config_path: Option<&Path>,
    tokenizer_path: Option<&Path>,
    page_size: usize,
    prompt: &str,
    max_new_tokens: Option<usize>,
    add_special_tokens: bool,
    skip_special_tokens: bool,
    profile_runtime: Option<&Path>,
    profile_layers: Option<&Path>,
    measure_tokens_per_second: bool,
    throughput_file: Option<&Path>,
    profile_token_costs: bool,
    expert_cache_gb: Option<f64>,
    hot_kv_cache_gb: Option<f64>,
    enable_telemetry: bool,
    telemetry_file: Option<&Path>,
    speculative_mtp: bool,
) -> Result<()> {
    let discovered_config = discover_config_path(model_path, config_path)?;
    let discovered_tokenizer = discover_tokenizer_path(model_path, tokenizer_path)?;
    let config = load_config(&discovered_config)?;
    let artifact = resolve_q2_artifact(model_path)?;
    let gguf = GgufFile::open(&artifact.gguf_path)?;
    let readiness = load_q2_readiness(&gguf, &artifact, &config)?;

    let tokenizer = Tokenizer::from_file(&discovered_tokenizer)?;
    let tokenizer_metadata = tokenizer.metadata();
    let rendered_prompt = render_user_prompt(prompt);
    let encoded = tokenizer.encode(&rendered_prompt.rendered, add_special_tokens)?;
    validate_generation_request(
        &config,
        &tokenizer_metadata,
        &encoded.token_ids,
        max_new_tokens,
        page_size,
        &readiness.artifact_file_name,
        &readiness.index.architecture,
        &readiness.index.summary,
    )?;

    let expert_cache_budget_bytes = cache_gb_to_bytes("expert cache", expert_cache_gb)?;
    let hot_kv_cache_budget_bytes = cache_gb_to_bytes("hot KV cache", hot_kv_cache_gb)?;
    let backend = MetalBackend::new()?;
    if let Some(expert_cache_budget_bytes) = expert_cache_budget_bytes {
        let slots_per_layer = expert_cache_slots_per_layer(
            &readiness.index,
            config.num_routed_experts,
            speculative_mtp,
            expert_cache_budget_bytes,
        )?;
        backend.configure_expert_cache_slots_per_layer(slots_per_layer)?;
    }
    let model = Model::open_from_index(
        &gguf,
        &config,
        readiness.index,
        &backend,
        DEFAULT_GGUF_OUTPUT_CHUNK_ROWS,
    )?;
    if let Some(profile_runtime) = profile_runtime {
        enable_q2_runtime_profile(profile_runtime)?;
    }
    if let Some(profile_layers) = profile_layers {
        enable_layer_profile(profile_layers)?;
    }
    if let Some(telemetry_file) = telemetry_file {
        enable_memory_telemetry_file(telemetry_file)?;
    } else if enable_telemetry {
        enable_memory_telemetry();
    }
    let mut stdout = io::stdout().lock();
    let mut stream = DecodedTextStream::new(&tokenizer, skip_special_tokens);
    let mut generated_token_count = 0_usize;
    let mut throughput = ThroughputRecorder::start();
    let generation_report = run_generate_streaming_with_options(
        &model,
        &config,
        &backend,
        &encoded.token_ids,
        max_new_tokens,
        page_size,
        &tokenizer_metadata.eos_token_ids,
        GenerationOptions {
            speculative_mtp,
            hot_kv_cache_budget_bytes,
            dynamic_cache_budget: None,
            profile_token_costs,
        },
        |token_id| {
            throughput.record_token();
            generated_token_count = generated_token_count
                .checked_add(1)
                .ok_or_else(|| Error::runtime("generated token count overflow"))?;
            if let Some(text) = stream.push(token_id)? {
                stdout
                    .write_all(text.as_bytes())
                    .map_err(|source| Error::Io {
                        path: PathBuf::from("<stdout>"),
                        source,
                    })?;
                stdout.flush().map_err(|source| Error::Io {
                    path: PathBuf::from("<stdout>"),
                    source,
                })?;
            }
            Ok(())
        },
    )?;
    let throughput_report = throughput
        .finish(encoded.token_ids.len())
        .with_cache_metrics(backend.expert_cache_metrics()?, generation_report.kv_cache);
    stdout.write_all(b"\n")?;
    if measure_tokens_per_second || throughput_file.is_some() {
        validate_exact_generated_token_count(generated_token_count, &throughput_report)?;
        if measure_tokens_per_second {
            write_tokens_per_second_report(&throughput_report)?;
        }
        if let Some(path) = throughput_file {
            append_tokens_per_second_report(path, &throughput_report)?;
        }
    }
    Ok(())
}

fn validate_exact_generated_token_count(
    generated_token_count: usize,
    report: &ThroughputReport,
) -> InfernoResult<()> {
    if generated_token_count != report.generated_tokens {
        return Err(Error::runtime(format!(
            "throughput token count mismatch: callback counted {generated_token_count}, report counted {}",
            report.generated_tokens
        )));
    }
    Ok(())
}

fn write_tokens_per_second_report(report: &ThroughputReport) -> InfernoResult<()> {
    let mut stderr = io::stderr().lock();
    writeln!(
        stderr,
        "inferno throughput: prompt_tokens={} generated_tokens={} total_seconds={:.3} total_tokens_per_second={:.3} time_to_first_token_seconds={:.3} decode_tokens={} decode_seconds={:.3} decode_tokens_per_second={:.3} decode_token_mean_seconds={:.3} decode_token_p50_seconds={:.3} decode_token_p95_seconds={:.3} expert_lookups={} expert_hits={} expert_misses={} expert_hit_rate={:.4} expert_ssd_read_gb={:.3} expert_cache_allocated_gb={:.3} expert_cache_capacity_gb={:.3} kv_lookups={} kv_hits={} kv_misses={} kv_hit_rate={:.4} kv_miss_rate={:.4} kv_selected_rows={} kv_ssd_read_gb={:.3} hot_kv_gb={:.3} cold_kv_gb={:.3}",
        report.prompt_tokens,
        report.generated_tokens,
        report.total_seconds,
        report.total_tokens_per_second,
        report.time_to_first_token_seconds,
        report.decode_tokens,
        report.decode_seconds,
        report.decode_tokens_per_second,
        report.decode_token_mean_seconds,
        report.decode_token_p50_seconds,
        report.decode_token_p95_seconds,
        report.expert_cache.lookups,
        report.expert_cache.hits,
        report.expert_cache.misses,
        report.expert_cache.hit_rate(),
        bytes_to_gb(report.expert_cache.ssd_read_bytes),
        bytes_to_gb(report.expert_cache.allocated_bytes),
        bytes_to_gb(report.expert_cache.capacity_bytes),
        report.kv_cache.lookups(),
        report.kv_cache.hits(),
        report.kv_cache.misses(),
        report.kv_cache.hit_rate(),
        report.kv_cache.miss_rate(),
        report.kv_cache.selected_rows,
        bytes_to_gb(report.kv_cache.ssd_read_bytes),
        bytes_to_gb(report.kv_cache.hot_bytes),
        bytes_to_gb(report.kv_cache.cold_bytes),
    )
    .map_err(|source| Error::Io {
        path: PathBuf::from("<stderr>"),
        source,
    })
}

fn append_tokens_per_second_report(path: &Path, report: &ThroughputReport) -> InfernoResult<()> {
    let needs_header = match path.metadata() {
        Ok(metadata) => metadata.len() == 0,
        Err(error) if error.kind() == io::ErrorKind::NotFound => true,
        Err(source) => {
            return Err(Error::Io {
                path: path.to_path_buf(),
                source,
            });
        }
    };
    if !needs_header {
        let contents = fs::read_to_string(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let existing_header = contents.lines().next().unwrap_or_default();
        if existing_header != THROUGHPUT_REPORT_HEADER {
            return Err(Error::runtime(format!(
                "throughput report {} uses an incompatible schema; choose a new output file",
                path.display()
            )));
        }
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
    if needs_header {
        writeln!(file, "{THROUGHPUT_REPORT_HEADER}").map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
    }
    writeln!(
        file,
        "{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{:.6}\t{:.6}\t{:.6}",
        report.prompt_tokens,
        report.generated_tokens,
        report.total_seconds,
        report.total_tokens_per_second,
        report.time_to_first_token_seconds,
        report.decode_tokens,
        report.decode_seconds,
        report.decode_tokens_per_second,
        report.decode_token_mean_seconds,
        report.decode_token_p50_seconds,
        report.decode_token_p95_seconds,
        report.expert_cache.lookups,
        report.expert_cache.hits,
        report.expert_cache.misses,
        report.expert_cache.hit_rate(),
        bytes_to_gb(report.expert_cache.ssd_read_bytes),
        bytes_to_gb(report.expert_cache.allocated_bytes),
        bytes_to_gb(report.expert_cache.capacity_bytes),
        report.kv_cache.lookups(),
        report.kv_cache.hits(),
        report.kv_cache.misses(),
        report.kv_cache.hit_rate(),
        report.kv_cache.miss_rate(),
        report.kv_cache.selected_rows,
        bytes_to_gb(report.kv_cache.ssd_read_bytes),
        bytes_to_gb(report.kv_cache.hot_bytes),
        bytes_to_gb(report.kv_cache.cold_bytes),
    )
    .map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })
}

const THROUGHPUT_REPORT_HEADER: &str = "prompt_tokens\tgenerated_tokens\ttotal_seconds\ttotal_tokens_per_second\ttime_to_first_token_seconds\tdecode_tokens\tdecode_seconds\tdecode_tokens_per_second\tdecode_token_mean_seconds\tdecode_token_p50_seconds\tdecode_token_p95_seconds\texpert_lookups\texpert_hits\texpert_misses\texpert_hit_rate\texpert_ssd_read_gb\texpert_cache_allocated_gb\texpert_cache_capacity_gb\tkv_lookups\tkv_hits\tkv_misses\tkv_hit_rate\tkv_miss_rate\tkv_selected_rows\tkv_ssd_read_gb\thot_kv_gb\tcold_kv_gb";

fn bytes_to_gb(bytes: u64) -> f64 {
    bytes as f64 / 1_000_000_000.0
}

fn cache_gb_to_bytes(name: &str, value: Option<f64>) -> InfernoResult<Option<usize>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !value.is_finite() || value <= 0.0 {
        return Err(Error::runtime(format!(
            "{name} size must be a finite positive number of GB, got {value}"
        )));
    }
    let bytes = value * 1_000_000_000.0;
    if bytes > usize::MAX as f64 {
        return Err(Error::runtime(format!(
            "{name} size {value} GB exceeds this platform's address space"
        )));
    }
    Ok(Some(bytes.floor() as usize))
}

fn expert_cache_slots_per_layer(
    index: &Index,
    expert_count: usize,
    include_mtp: bool,
    budget_bytes: usize,
) -> InfernoResult<usize> {
    if expert_count == 0 {
        return Err(Error::weights("routed expert count must be positive"));
    }
    let packed = index
        .layers
        .iter()
        .find_map(|layer| match &layer.ffn {
            FfnIndex::SparseMoe { packed_experts, .. } => Some(packed_experts),
            FfnIndex::Dense(_) => None,
        })
        .ok_or_else(|| Error::weights("GLM index has no routed expert tensors"))?;
    let expert_count_u64 = u64::try_from(expert_count)
        .map_err(|_| Error::weights("routed expert count does not fit u64"))?;
    let bytes_per_expert = [&packed.gate, &packed.up, &packed.down]
        .into_iter()
        .try_fold(0_u64, |total, tensor| {
            if tensor.storage_byte_len % expert_count_u64 != 0 {
                return Err(Error::weights(format!(
                    "packed expert tensor {} byte length {} is not divisible by expert count {expert_count}",
                    tensor.name, tensor.storage_byte_len
                )));
            }
            total
                .checked_add(tensor.storage_byte_len / expert_count_u64)
                .ok_or_else(|| Error::weights("Q2 expert triplet byte size overflow"))
        })?;
    let routed_layer_count = index
        .summary
        .sparse_layer_count
        .checked_add(usize::from(include_mtp && index.mtp.is_some()))
        .ok_or_else(|| Error::weights("routed layer count overflow"))?;
    let bytes_per_layer_slot = usize::try_from(bytes_per_expert)
        .map_err(|_| Error::weights("Q2 expert triplet byte size does not fit usize"))?;
    let bytes_per_global_slot = bytes_per_layer_slot
        .checked_mul(routed_layer_count)
        .ok_or_else(|| Error::weights("Q2 expert cache layer budget overflow"))?;
    let slots = budget_bytes / bytes_per_global_slot;
    if slots == 0 {
        return Err(Error::runtime(format!(
            "expert cache budget {:.3} GB is too small; one slot across {routed_layer_count} routed layers requires {:.3} GB",
            budget_bytes as f64 / 1_000_000_000.0,
            bytes_per_global_slot as f64 / 1_000_000_000.0,
        )));
    }
    Ok(slots.min(expert_count))
}

fn tokens_per_second(token_count: usize, elapsed: Duration) -> f64 {
    let elapsed_seconds = elapsed.as_secs_f64();
    if elapsed_seconds <= 0.0 {
        return 0.0;
    }
    token_count as f64 / elapsed_seconds
}

struct ThroughputRecorder {
    started_at: Instant,
    token_offsets: Vec<Duration>,
}

impl ThroughputRecorder {
    fn start() -> Self {
        Self {
            started_at: Instant::now(),
            token_offsets: Vec::new(),
        }
    }

    fn record_token(&mut self) {
        self.token_offsets.push(self.started_at.elapsed());
    }

    fn finish(self, prompt_tokens: usize) -> ThroughputReport {
        ThroughputReport::from_token_offsets(
            prompt_tokens,
            self.started_at.elapsed(),
            &self.token_offsets,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ThroughputReport {
    prompt_tokens: usize,
    generated_tokens: usize,
    total_seconds: f64,
    total_tokens_per_second: f64,
    time_to_first_token_seconds: f64,
    decode_tokens: usize,
    decode_seconds: f64,
    decode_tokens_per_second: f64,
    decode_token_mean_seconds: f64,
    decode_token_p50_seconds: f64,
    decode_token_p95_seconds: f64,
    expert_cache: ExpertCacheMetrics,
    kv_cache: KvCacheMetrics,
}

impl ThroughputReport {
    fn from_token_offsets(
        prompt_tokens: usize,
        total_elapsed: Duration,
        token_offsets: &[Duration],
    ) -> Self {
        let generated_tokens = token_offsets.len();
        let total_seconds = total_elapsed.as_secs_f64();
        let total_tokens_per_second = tokens_per_second(generated_tokens, total_elapsed);
        let time_to_first_token_seconds = token_offsets
            .first()
            .map(Duration::as_secs_f64)
            .unwrap_or(0.0);
        let intervals = decode_token_intervals(token_offsets);
        let decode_tokens = intervals.len();
        let decode_seconds = match (token_offsets.first(), token_offsets.last()) {
            (Some(first), Some(last)) if generated_tokens > 1 => {
                last.checked_sub(*first).unwrap_or_default().as_secs_f64()
            }
            _ => 0.0,
        };
        let decode_tokens_per_second = if decode_seconds <= 0.0 {
            0.0
        } else {
            decode_tokens as f64 / decode_seconds
        };
        let decode_token_mean_seconds = if intervals.is_empty() {
            0.0
        } else {
            intervals.iter().map(Duration::as_secs_f64).sum::<f64>() / intervals.len() as f64
        };
        let decode_token_p50_seconds = percentile_seconds(intervals.clone(), 50);
        let decode_token_p95_seconds = percentile_seconds(intervals, 95);

        Self {
            prompt_tokens,
            generated_tokens,
            total_seconds,
            total_tokens_per_second,
            time_to_first_token_seconds,
            decode_tokens,
            decode_seconds,
            decode_tokens_per_second,
            decode_token_mean_seconds,
            decode_token_p50_seconds,
            decode_token_p95_seconds,
            expert_cache: ExpertCacheMetrics::default(),
            kv_cache: KvCacheMetrics::default(),
        }
    }

    fn with_cache_metrics(
        mut self,
        expert_cache: ExpertCacheMetrics,
        kv_cache: KvCacheMetrics,
    ) -> Self {
        self.expert_cache = expert_cache;
        self.kv_cache = kv_cache;
        self
    }
}

fn decode_token_intervals(token_offsets: &[Duration]) -> Vec<Duration> {
    token_offsets
        .windows(2)
        .filter_map(|window| window[1].checked_sub(window[0]))
        .collect()
}

fn percentile_seconds(mut values: Vec<Duration>, percentile: usize) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_unstable();
    let last_index = values.len() - 1;
    let index = last_index.saturating_mul(percentile).div_ceil(100);
    values[index.min(last_index)].as_secs_f64()
}

struct DecodedTextStream<'a> {
    tokenizer: &'a Tokenizer,
    skip_special_tokens: bool,
    token_ids: Vec<u32>,
    emitted_text: String,
}

impl<'a> DecodedTextStream<'a> {
    fn new(tokenizer: &'a Tokenizer, skip_special_tokens: bool) -> Self {
        Self {
            tokenizer,
            skip_special_tokens,
            token_ids: Vec::new(),
            emitted_text: String::new(),
        }
    }

    fn push(&mut self, token_id: u32) -> InfernoResult<Option<String>> {
        self.token_ids.push(token_id);
        let decoded = self
            .tokenizer
            .decode(&self.token_ids, self.skip_special_tokens)?;
        if decoded.len() <= self.emitted_text.len() {
            return Ok(None);
        }

        let suffix = decoded
            .strip_prefix(&self.emitted_text)
            .ok_or_else(|| Error::tokenizer("streaming decode produced non-monotonic text"))?
            .to_string();
        self.emitted_text = decoded;

        if suffix.is_empty() {
            Ok(None)
        } else {
            Ok(Some(suffix))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ArtifactSelection {
    gguf_path: PathBuf,
    artifact_file_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Readiness {
    artifact_file_name: String,
    index: Index,
}

fn resolve_q2_artifact(model_path: &Path) -> Result<ArtifactSelection> {
    let artifact = antirez_q2_artifact();
    let gguf_path = model_path.join(artifact.file_name);
    if !gguf_path.exists() {
        return Err(Error::weights(format!(
            "GLM-5.2 Q2 generation requires {}; canonical external artifact source is {}/{} ({})",
            gguf_path.display(),
            artifact.repo_id,
            artifact.file_name,
            artifact.format.as_str()
        ))
        .into());
    }

    Ok(ArtifactSelection {
        gguf_path,
        artifact_file_name: artifact.file_name.to_string(),
    })
}

fn load_q2_readiness(
    gguf: &GgufFile,
    artifact: &ArtifactSelection,
    config: &Config,
) -> Result<Readiness> {
    let index = Index::from_gguf(gguf, config)?;
    Ok(Readiness {
        artifact_file_name: artifact.artifact_file_name.clone(),
        index,
    })
}

fn validate_generation_request(
    config: &Config,
    tokenizer_metadata: &TokenizerMetadata,
    prompt_token_ids: &[u32],
    max_new_tokens: Option<usize>,
    page_size: usize,
    artifact_file_name: &str,
    architecture: &str,
    index: &IndexSummary,
) -> Result<()> {
    if architecture != "glm-dsa" {
        return Err(Error::weights(format!(
            "{} is architecture {}; expected glm-dsa",
            artifact_file_name, architecture
        ))
        .into());
    }
    let indexed_layers = index
        .dense_layer_count
        .saturating_add(index.sparse_layer_count);
    if indexed_layers != config.num_layers {
        return Err(Error::weights(format!(
            "{artifact_file_name} maps {indexed_layers} GLM layers but config has {}",
            config.num_layers
        ))
        .into());
    }
    if let Some(token_id) = tokenizer_metadata
        .eos_token_ids
        .iter()
        .copied()
        .find(|token_id| *token_id as usize >= config.vocab_size)
    {
        return Err(Error::tokenizer(format!(
            "tokenizer EOS token id {token_id} is outside config vocab_size {}",
            config.vocab_size
        ))
        .into());
    }
    if prompt_token_ids.is_empty() {
        return Err(Error::runtime("generate requires at least one prompt token").into());
    }
    if max_new_tokens == Some(0) {
        return Err(Error::runtime("max_new_tokens must be positive when provided").into());
    }
    if page_size == 0 {
        return Err(Error::cache("paged KV cache page_size must be positive").into());
    }
    if page_size > config.max_context {
        return Err(Error::cache(format!(
            "paged KV cache page_size {page_size} exceeds max_context {}",
            config.max_context
        ))
        .into());
    }

    let max_new_tokens = match max_new_tokens {
        Some(max_new_tokens) => max_new_tokens,
        None => config
            .max_context
            .checked_sub(prompt_token_ids.len())
            .ok_or_else(|| {
                Error::runtime(format!(
                    "prompt token count {} exceeds max_context {}",
                    prompt_token_ids.len(),
                    config.max_context
                ))
            })?,
    };
    if max_new_tokens == 0 {
        return Err(Error::runtime(format!(
            "prompt token count {} leaves no room for generation within max_context {}",
            prompt_token_ids.len(),
            config.max_context
        ))
        .into());
    }

    let requested_context_tokens = prompt_token_ids
        .len()
        .checked_add(max_new_tokens)
        .ok_or_else(|| Error::runtime("requested context token count overflow"))?;
    if requested_context_tokens > config.max_context {
        return Err(Error::runtime(format!(
            "requested context tokens {requested_context_tokens} exceed max_context {}",
            config.max_context
        ))
        .into());
    }
    if let Some(token_id) = prompt_token_ids
        .iter()
        .copied()
        .find(|token_id| *token_id as usize >= config.vocab_size)
    {
        return Err(Error::tokenizer(format!(
            "prompt token id {token_id} is outside config vocab_size {}",
            config.vocab_size
        ))
        .into());
    }
    Ok(())
}

fn discover_config_path(model_path: &Path, explicit_config_path: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit_config_path {
        return Ok(path.to_path_buf());
    }
    let candidate = model_path.join("config.json");
    if candidate.exists() {
        return Ok(candidate);
    }
    Err(Error::config(format!(
        "config path was not provided and {} does not exist",
        candidate.display()
    ))
    .into())
}

fn discover_tokenizer_path(
    model_path: &Path,
    explicit_tokenizer_path: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(path) = explicit_tokenizer_path {
        return Ok(path.to_path_buf());
    }
    let candidate = model_path.join("tokenizer.json");
    if candidate.exists() {
        return Ok(candidate);
    }
    Err(Error::tokenizer(format!(
        "tokenizer path was not provided and {} does not exist",
        candidate.display()
    ))
    .into())
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn tokens_per_second_uses_generated_token_count_and_elapsed_time() {
        let rate = tokens_per_second(6, Duration::from_secs(3));

        assert_eq!(rate, 2.0);
    }

    #[test]
    fn tokens_per_second_handles_zero_elapsed_time() {
        let rate = tokens_per_second(6, Duration::from_secs(0));

        assert_eq!(rate, 0.0);
    }

    #[test]
    fn cache_budget_conversion_uses_decimal_gb_and_rejects_invalid_values() {
        assert_eq!(
            cache_gb_to_bytes("cache", Some(1.5)).unwrap(),
            Some(1_500_000_000)
        );
        assert!(cache_gb_to_bytes("cache", Some(0.0)).is_err());
        assert!(cache_gb_to_bytes("cache", Some(f64::NAN)).is_err());
    }

    #[test]
    fn throughput_report_separates_first_token_from_decode_rate() {
        let report = ThroughputReport::from_token_offsets(
            5,
            Duration::from_secs(10),
            &[
                Duration::from_secs(4),
                Duration::from_secs(6),
                Duration::from_secs(7),
                Duration::from_secs(10),
            ],
        );

        assert_eq!(report.prompt_tokens, 5);
        assert_eq!(report.generated_tokens, 4);
        assert_eq!(report.total_seconds, 10.0);
        assert_eq!(report.total_tokens_per_second, 0.4);
        assert_eq!(report.time_to_first_token_seconds, 4.0);
        assert_eq!(report.decode_tokens, 3);
        assert_eq!(report.decode_seconds, 6.0);
        assert_eq!(report.decode_tokens_per_second, 0.5);
        assert_eq!(report.decode_token_mean_seconds, 2.0);
        assert_eq!(report.decode_token_p50_seconds, 2.0);
        assert_eq!(report.decode_token_p95_seconds, 3.0);
    }

    #[test]
    fn throughput_report_handles_single_generated_token() {
        let report = ThroughputReport::from_token_offsets(
            3,
            Duration::from_secs(5),
            &[Duration::from_secs(5)],
        );

        assert_eq!(report.generated_tokens, 1);
        assert_eq!(report.decode_tokens, 0);
        assert_eq!(report.decode_seconds, 0.0);
        assert_eq!(report.decode_tokens_per_second, 0.0);
        assert_eq!(report.decode_token_mean_seconds, 0.0);
        assert_eq!(report.decode_token_p50_seconds, 0.0);
        assert_eq!(report.decode_token_p95_seconds, 0.0);
    }

    #[test]
    fn append_tokens_per_second_report_writes_header_once() {
        let dir = unique_temp_dir("throughput-report");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("throughput.tsv");
        let report = ThroughputReport::from_token_offsets(
            2,
            Duration::from_secs(2),
            &[Duration::from_secs(1)],
        );

        append_tokens_per_second_report(&path, &report).unwrap();
        append_tokens_per_second_report(&path, &report).unwrap();

        let contents = fs::read_to_string(path).unwrap();
        assert_eq!(
            contents
                .lines()
                .filter(|line| line.starts_with("prompt_tokens"))
                .count(),
            1
        );
        assert_eq!(contents.lines().count(), 3);
    }

    #[test]
    fn append_tokens_per_second_report_rejects_old_schema() {
        let dir = unique_temp_dir("throughput-old-schema");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("throughput.tsv");
        fs::write(&path, "prompt_tokens\tgenerated_tokens\n").unwrap();
        let report = ThroughputReport::from_token_offsets(
            2,
            Duration::from_secs(2),
            &[Duration::from_secs(1)],
        );

        let error = append_tokens_per_second_report(&path, &report).unwrap_err();

        assert!(error.to_string().contains("incompatible schema"));
    }

    #[test]
    fn readiness_requires_selected_antirez_gguf_artifact() {
        let dir = unique_temp_dir("missing-gguf");
        fs::create_dir_all(&dir).unwrap();

        let err = resolve_q2_artifact(&dir)
            .expect_err("production GLM generation must require selected GGUF artifact");

        let rendered = err.to_string();
        assert!(rendered.contains("GLM-5.2-UD-Q2_K_RoutedQ2K.gguf"));
        assert!(rendered.contains("antirez/glm-5.2-gguf"));
    }

    #[test]
    fn readiness_maps_local_antirez_q2_gguf_artifact() {
        let dir = unique_temp_dir("q2-gguf");
        fs::create_dir_all(&dir).unwrap();
        let artifact = antirez_q2_artifact();
        fs::write(dir.join(artifact.file_name), tiny_gguf()).unwrap();

        let selection = resolve_q2_artifact(&dir).unwrap();
        let gguf = GgufFile::open(&selection.gguf_path).unwrap();
        let readiness =
            load_q2_readiness(&gguf, &selection, &tiny_config_with_experts(128, 1)).unwrap();

        assert_eq!(readiness.index.architecture, "glm-dsa");
        assert_eq!(readiness.artifact_file_name, artifact.file_name);
        assert_eq!(readiness.index.summary.dense_layer_count, 1);
        assert_eq!(readiness.index.summary.sparse_layer_count, 1);
        assert_eq!(readiness.index.summary.split_kv_b_projection_count, 4);
        assert_eq!(readiness.index.summary.packed_expert_tensor_count, 3);
    }

    #[test]
    fn readiness_uses_q2_artifact_file() {
        let dir = unique_temp_dir("q2-gguf");
        fs::create_dir_all(&dir).unwrap();
        let artifact = antirez_q2_artifact();
        fs::write(dir.join(artifact.file_name), tiny_gguf()).unwrap();

        let selection = resolve_q2_artifact(&dir).unwrap();

        assert!(selection
            .gguf_path
            .ends_with("GLM-5.2-UD-Q2_K_RoutedQ2K.gguf"));
    }

    #[test]
    fn generation_request_rejects_zero_page_size() {
        let config = tiny_config_with_experts(128, 1);
        let index = tiny_index_summary();

        let err = validate_generation_request(
            &config,
            &tokenizer_metadata(16),
            &[1],
            Some(1),
            0,
            "GLM-5.2-UD-Q2_K_RoutedQ2K.gguf",
            "glm-dsa",
            &index,
        )
        .expect_err("zero page size should fail");

        assert!(err.to_string().contains("page_size"));
    }

    #[test]
    fn generation_request_rejects_context_overflow() {
        let config = tiny_config_with_experts(3, 1);
        let index = tiny_index_summary();

        let err = validate_generation_request(
            &config,
            &tokenizer_metadata(16),
            &[1, 2],
            Some(2),
            1,
            "GLM-5.2-UD-Q2_K_RoutedQ2K.gguf",
            "glm-dsa",
            &index,
        )
        .expect_err("context overflow should fail");

        assert!(err.to_string().contains("max_context"));
    }

    #[test]
    fn generation_request_allows_model_vocab_padding_rows() {
        let mut config = tiny_config_with_experts(128, 1);
        config.vocab_size = 32;
        let index = tiny_index_summary();
        let mut metadata = tokenizer_metadata(24);
        metadata.eos_token_ids = vec![23];

        validate_generation_request(
            &config,
            &metadata,
            &[1, 2],
            Some(1),
            1,
            "GLM-5.2-UD-Q2_K_RoutedQ2K.gguf",
            "glm-dsa",
            &index,
        )
        .unwrap();
    }

    #[test]
    fn generation_request_rejects_eos_id_outside_model_vocab() {
        let mut config = tiny_config_with_experts(128, 1);
        config.vocab_size = 32;
        let index = tiny_index_summary();
        let mut metadata = tokenizer_metadata(24);
        metadata.eos_token_ids = vec![32];

        let err = validate_generation_request(
            &config,
            &metadata,
            &[1, 2],
            Some(1),
            1,
            "GLM-5.2-UD-Q2_K_RoutedQ2K.gguf",
            "glm-dsa",
            &index,
        )
        .expect_err("EOS outside model vocab must fail");

        assert!(err.to_string().contains("EOS token id"));
    }

    fn tokenizer_metadata(vocab_size_with_added_tokens: usize) -> TokenizerMetadata {
        TokenizerMetadata {
            vocab_size: vocab_size_with_added_tokens,
            vocab_size_with_added_tokens,
            added_tokens_count: 0,
            encode_special_tokens: false,
            special_tokens: Vec::new(),
            eos_token_ids: Vec::new(),
        }
    }

    fn tiny_index_summary() -> IndexSummary {
        IndexSummary {
            tensor_count: 32,
            metadata_kv_count: 2,
            dense_layer_count: 1,
            sparse_layer_count: 1,
            dsa_indexer_layer_count: 0,
            mtp_layer_count: 0,
            split_kv_b_projection_count: 4,
            packed_expert_tensor_count: 3,
            quantized_tensor_count: 24,
            raw_tensor_count: 8,
        }
    }

    fn tiny_gguf() -> Vec<u8> {
        use gguf::{GgmlType, GgufMetadataValueType, GGUF_MAGIC, GGUF_VERSION_V3};

        let mut tensors = Vec::<(String, Vec<u64>, GgmlType)>::new();
        push_tensor(&mut tensors, "token_embd.weight", &[4, 4], GgmlType::Q2K);
        push_tensor(&mut tensors, "output_norm.weight", &[4], GgmlType::F32);
        push_tensor(&mut tensors, "output.weight", &[4, 4], GgmlType::Q2K);
        push_dense_gguf_layer(&mut tensors, 0);
        push_sparse_gguf_layer(&mut tensors, 1);

        let mut bytes = Vec::new();
        bytes.extend_from_slice(GGUF_MAGIC);
        push_u32(&mut bytes, GGUF_VERSION_V3);
        push_u64(&mut bytes, tensors.len() as u64);
        push_u64(&mut bytes, 2);

        push_string(&mut bytes, "general.architecture");
        push_u32(&mut bytes, GgufMetadataValueType::String as u32);
        push_string(&mut bytes, "glm-dsa");

        push_string(&mut bytes, "general.alignment");
        push_u32(&mut bytes, GgufMetadataValueType::Uint32 as u32);
        push_u32(&mut bytes, 32);

        let mut offset = 0_u64;
        for (name, dims, ty) in tensors {
            push_string(&mut bytes, &name);
            push_u32(&mut bytes, dims.len() as u32);
            for dim in dims {
                push_u64(&mut bytes, dim);
            }
            push_u32(&mut bytes, ty.code());
            push_u64(&mut bytes, offset);
            offset += 32;
        }

        let remainder = bytes.len() % 32;
        if remainder != 0 {
            bytes.resize(bytes.len() + 32 - remainder, 0);
        }
        bytes.resize(bytes.len() + offset as usize + 32, 0);
        bytes
    }

    fn push_dense_gguf_layer(tensors: &mut Vec<(String, Vec<u64>, gguf::GgmlType)>, layer: usize) {
        push_attention_gguf_tensors(tensors, layer, true);
        push_tensor(
            tensors,
            &format!("blk.{layer}.ffn_gate.weight"),
            &[4, 3],
            gguf::GgmlType::Q2K,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.ffn_up.weight"),
            &[4, 3],
            gguf::GgmlType::Q2K,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.ffn_down.weight"),
            &[3, 4],
            gguf::GgmlType::Q2K,
        );
    }

    fn push_sparse_gguf_layer(tensors: &mut Vec<(String, Vec<u64>, gguf::GgmlType)>, layer: usize) {
        push_attention_gguf_tensors(tensors, layer, false);
        push_tensor(
            tensors,
            &format!("blk.{layer}.exp_probs_b.bias"),
            &[1],
            gguf::GgmlType::F32,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.ffn_gate_inp.weight"),
            &[4, 1],
            gguf::GgmlType::Q2K,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.ffn_gate_shexp.weight"),
            &[4, 3],
            gguf::GgmlType::Q2K,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.ffn_up_shexp.weight"),
            &[4, 3],
            gguf::GgmlType::Q2K,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.ffn_down_shexp.weight"),
            &[3, 4],
            gguf::GgmlType::Q2K,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.ffn_gate_exps.weight"),
            &[1, 4, 3],
            gguf::GgmlType::Q2K,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.ffn_up_exps.weight"),
            &[1, 4, 3],
            gguf::GgmlType::Q2K,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.ffn_down_exps.weight"),
            &[1, 3, 4],
            gguf::GgmlType::Q2K,
        );
    }

    fn push_attention_gguf_tensors(
        tensors: &mut Vec<(String, Vec<u64>, gguf::GgmlType)>,
        layer: usize,
        include_indexer: bool,
    ) {
        push_tensor(
            tensors,
            &format!("blk.{layer}.attn_norm.weight"),
            &[4],
            gguf::GgmlType::F32,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.ffn_norm.weight"),
            &[4],
            gguf::GgmlType::F32,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.attn_q_a.weight"),
            &[4, 4],
            gguf::GgmlType::Q2K,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.attn_q_a_norm.weight"),
            &[4],
            gguf::GgmlType::F32,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.attn_q_b.weight"),
            &[4, 4],
            gguf::GgmlType::Q2K,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.attn_kv_a_mqa.weight"),
            &[4, 4],
            gguf::GgmlType::Q2K,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.attn_kv_a_norm.weight"),
            &[4],
            gguf::GgmlType::F32,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.attn_k_b.weight"),
            &[4, 4],
            gguf::GgmlType::Q2K,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.attn_v_b.weight"),
            &[4, 4],
            gguf::GgmlType::Q2K,
        );
        push_tensor(
            tensors,
            &format!("blk.{layer}.attn_output.weight"),
            &[4, 4],
            gguf::GgmlType::Q2K,
        );

        if include_indexer {
            push_tensor(
                tensors,
                &format!("blk.{layer}.indexer.k_norm.bias"),
                &[4],
                gguf::GgmlType::F32,
            );
            push_tensor(
                tensors,
                &format!("blk.{layer}.indexer.k_norm.weight"),
                &[4],
                gguf::GgmlType::F32,
            );
            push_tensor(
                tensors,
                &format!("blk.{layer}.indexer.proj.weight"),
                &[4, 4],
                gguf::GgmlType::Q2K,
            );
            push_tensor(
                tensors,
                &format!("blk.{layer}.indexer.attn_k.weight"),
                &[4, 4],
                gguf::GgmlType::Q2K,
            );
            push_tensor(
                tensors,
                &format!("blk.{layer}.indexer.attn_q_b.weight"),
                &[4, 4],
                gguf::GgmlType::Q2K,
            );
        }
    }

    fn push_tensor(
        tensors: &mut Vec<(String, Vec<u64>, gguf::GgmlType)>,
        name: &str,
        dims: &[u64],
        ty: gguf::GgmlType,
    ) {
        tensors.push((name.to_string(), dims.to_vec(), ty));
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        push_u64(bytes, value.len() as u64);
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn unique_temp_dir(label: &str) -> PathBuf {
        let id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("inferno-{label}-{}-{id}", std::process::id()))
    }

    fn tiny_config_with_experts(max_context: usize, experts: usize) -> Config {
        Config {
            model_type: "glm_moe_dsa".to_string(),
            hidden_size: 4,
            num_layers: 2,
            dense_layers: 1,
            sparse_moe_layers: Some(1),
            vocab_size: 8,
            attention_heads: 2,
            qk_head_dim: 4,
            qk_no_rope_dim: 2,
            qk_rope_dim: 2,
            kv_lora_rank: 2,
            v_head_dim: Some(4),
            num_routed_experts: experts,
            experts_per_token: 1,
            max_context,
            dsa_index_topk: max_context.min(16),
            index_head_dim: 128,
            index_n_heads: 32,
            index_topk_freq: 4,
            indexer_rope_interleave: true,
            indexer_types: Vec::new(),
            moe_intermediate_size: 3,
            num_shared_experts: 1,
            moe_groups: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 2.5,
            scoring_func: "sigmoid".to_string(),
            topk_method: "noaux_tc".to_string(),
            num_nextn_predict_layers: 0,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        }
        .validated()
        .unwrap()
    }
}
