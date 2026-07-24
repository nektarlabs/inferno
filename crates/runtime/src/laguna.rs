use std::time::{Duration, Instant};

use backend::Backend;
use common::{Error, Result};
use model::{LagunaExpertCacheMetrics, LagunaModel, LagunaSession};

use crate::{
    laguna_memory::LagunaMemoryController,
    telemetry::{capture_memory_snapshot, RuntimeKvMemoryBytes},
    LagunaMemoryControllerReport, LagunaMemoryControllerSpec,
};

const LAGUNA_PREFILL_CHUNK_TOKENS: usize = 4_096;
const LAGUNA_INITIAL_DECODE_CAPACITY_TOKENS: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationControl {
    Continue,
    Stop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LagunaGenerationOptions {
    pub expert_cache_capacity: usize,
    pub memory_controller: Option<LagunaMemoryControllerSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LagunaGenerationReport {
    pub generated_tokens: usize,
    pub final_context_tokens: usize,
    pub total_duration: Duration,
    pub time_to_first_token: Option<Duration>,
    pub expert_cache: LagunaExpertCacheMetrics,
    pub memory_controller: Option<LagunaMemoryControllerReport>,
}

impl LagunaGenerationReport {
    pub fn tokens_per_second(self) -> f64 {
        let seconds = self.total_duration.as_secs_f64();
        if seconds == 0.0 {
            0.0
        } else {
            self.generated_tokens as f64 / seconds
        }
    }
}

/// Persistent Laguna generation state for chat and serving.
///
/// Each request starts a fresh KV sequence, while routed experts remain in the
/// same cache across requests. This preserves correctness and avoids turning
/// every chat turn into a cold expert-cache start.
#[derive(Debug)]
pub struct LagunaRuntime {
    expert_cache_capacity: usize,
    memory_controller: Option<LagunaMemoryController>,
    session: Option<LagunaSession>,
}

impl LagunaRuntime {
    pub fn new(options: LagunaGenerationOptions) -> Result<Self> {
        if options.expert_cache_capacity == 0 {
            return Err(Error::cache(
                "Laguna runtime expert-cache capacity must be positive",
            ));
        }
        let memory_controller = options
            .memory_controller
            .map(|spec| {
                if spec.initial_expert_capacity != options.expert_cache_capacity {
                    return Err(Error::runtime(format!(
                        "Laguna memory controller initial capacity {} does not match runtime capacity {}",
                        spec.initial_expert_capacity, options.expert_cache_capacity
                    )));
                }
                LagunaMemoryController::new(spec)
            })
            .transpose()?;
        Ok(Self {
            expert_cache_capacity: options.expert_cache_capacity,
            memory_controller,
            session: None,
        })
    }

    pub fn expert_cache_capacity(&self) -> usize {
        self.expert_cache_capacity
    }

    pub fn generate_streaming<B, F>(
        &mut self,
        model: &LagunaModel,
        backend: &B,
        prompt_token_ids: &[u32],
        max_new_tokens: Option<usize>,
        stop_token_ids: &[u32],
        mut on_token: F,
    ) -> Result<LagunaGenerationReport>
    where
        B: Backend,
        F: FnMut(u32) -> Result<()>,
    {
        self.generate_streaming_controlled(
            model,
            backend,
            prompt_token_ids,
            max_new_tokens,
            stop_token_ids,
            |token_id| {
                on_token(token_id)?;
                Ok(GenerationControl::Continue)
            },
        )
    }

    /// Generates tokens until EOS, the configured limit, or the callback
    /// reports that a complete higher-level response is ready.
    pub fn generate_streaming_controlled<B, F>(
        &mut self,
        model: &LagunaModel,
        backend: &B,
        prompt_token_ids: &[u32],
        max_new_tokens: Option<usize>,
        stop_token_ids: &[u32],
        mut on_token: F,
    ) -> Result<LagunaGenerationReport>
    where
        B: Backend,
        F: FnMut(u32) -> Result<GenerationControl>,
    {
        if prompt_token_ids.is_empty() {
            return Err(Error::runtime(
                "Laguna generation requires at least one prompt token",
            ));
        }
        let config = model.config();
        if prompt_token_ids.len() >= config.max_position_embeddings {
            return Err(Error::runtime(format!(
                "Laguna prompt length {} leaves no generation room in context {}",
                prompt_token_ids.len(),
                config.max_position_embeddings
            )));
        }
        let available_tokens = config.max_position_embeddings - prompt_token_ids.len();
        let generated_limit = max_new_tokens.unwrap_or(available_tokens);
        if generated_limit > available_tokens {
            return Err(Error::runtime(format!(
                "Laguna requested {generated_limit} generated tokens, but only {available_tokens} fit after the prompt"
            )));
        }
        if generated_limit == 0 {
            return Ok(LagunaGenerationReport {
                generated_tokens: 0,
                final_context_tokens: 0,
                total_duration: Duration::ZERO,
                time_to_first_token: None,
                expert_cache: LagunaExpertCacheMetrics::default(),
                memory_controller: self
                    .memory_controller
                    .as_ref()
                    .map(LagunaMemoryController::report),
            });
        }

        let required_context_capacity = initial_context_capacity(
            prompt_token_ids.len(),
            generated_limit,
            config.max_position_embeddings,
        )?;
        self.prepare_session(model, backend, required_context_capacity)?;
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| Error::runtime("Laguna runtime session was not prepared"))?;

        let started_at = Instant::now();
        let mut time_to_first_token = None;
        let mut generated_tokens = 0_usize;
        let final_prompt_chunk_start =
            final_chunk_start(prompt_token_ids.len(), LAGUNA_PREFILL_CHUNK_TOKENS)?;
        for chunk in
            prompt_token_ids[..final_prompt_chunk_start].chunks(LAGUNA_PREFILL_CHUNK_TOKENS)
        {
            model.prefill_chunk(session, chunk, backend)?;
        }
        let mut decode_token = [0_u32; 1];

        let generation_result = (|| -> Result<()> {
            while generated_tokens < generated_limit {
                let input: &[u32] = if generated_tokens == 0 {
                    &prompt_token_ids[final_prompt_chunk_start..]
                } else {
                    &decode_token
                };
                let required_end = session
                    .position()?
                    .checked_add(input.len())
                    .ok_or_else(|| Error::runtime("Laguna sequence capacity overflow"))?;
                if required_end > session.context_capacity() {
                    let next_capacity = next_context_capacity(
                        session.context_capacity(),
                        required_end,
                        config.max_position_embeddings,
                    )?;
                    model.grow_session_capacity(session, next_capacity, backend)?;
                }
                let model_started_at = self.memory_controller.as_ref().map(|_| Instant::now());
                let output = model.forward_next_token(session, input, backend)?;
                let model_elapsed = model_started_at.map(|started_at| started_at.elapsed());
                if time_to_first_token.is_none() {
                    time_to_first_token = Some(started_at.elapsed());
                }

                if let Some(controller) = self.memory_controller.as_mut() {
                    if generated_tokens == 0 {
                        controller.begin_decode(
                            session.expert_cache_metrics(),
                            capture_memory_snapshot(backend, RuntimeKvMemoryBytes::default()),
                        );
                    } else {
                        controller.record_decode_step(model_elapsed.ok_or_else(|| {
                            Error::runtime("Laguna controller decode timer was not started")
                        })?);
                        if controller.observation_due() {
                            let memory =
                                capture_memory_snapshot(backend, RuntimeKvMemoryBytes::default());
                            let decision = controller.observe(
                                session.position()?,
                                session.expert_cache_metrics(),
                                memory,
                            )?;
                            if decision.changes_capacity() {
                                session
                                    .resize_expert_cache_capacity(decision.next_expert_capacity)?;
                                self.expert_cache_capacity = decision.next_expert_capacity;
                            }
                            controller.record_decision(decision, memory)?;
                        }
                    }
                }

                let control = on_token(output.token_id)?;
                generated_tokens = generated_tokens
                    .checked_add(1)
                    .ok_or_else(|| Error::runtime("Laguna generated-token count overflow"))?;
                if control == GenerationControl::Stop
                    || stop_token_ids.contains(&output.token_id)
                    || generated_tokens == generated_limit
                {
                    break;
                }
                decode_token[0] = output.token_id;
            }
            Ok(())
        })();

        if let Some(controller) = self.memory_controller.as_mut() {
            controller.pause_decode(session.expert_cache_metrics());
        }
        generation_result?;

        let expert_cache = session.expert_cache_metrics();
        expert_cache.validate()?;
        Ok(LagunaGenerationReport {
            generated_tokens,
            final_context_tokens: session.position()?,
            total_duration: started_at.elapsed(),
            time_to_first_token,
            expert_cache,
            memory_controller: self
                .memory_controller
                .as_ref()
                .map(LagunaMemoryController::report),
        })
    }

    fn prepare_session<B: Backend>(
        &mut self,
        model: &LagunaModel,
        backend: &B,
        required_context_capacity: usize,
    ) -> Result<()> {
        let Some(session) = self.session.as_mut() else {
            self.session = Some(model.new_session(
                1,
                required_context_capacity,
                self.expert_cache_capacity,
                backend,
            )?);
            return Ok(());
        };

        let allocated_capacity = session.context_capacity();
        let target_capacity = if required_context_capacity > allocated_capacity {
            allocated_capacity
                .saturating_mul(2)
                .max(required_context_capacity)
                .min(model.config().max_position_embeddings)
        } else {
            allocated_capacity
        };
        model.prepare_session(session, 1, target_capacity, backend)
    }
}

fn initial_context_capacity(
    prompt_tokens: usize,
    generated_limit: usize,
    max_context_tokens: usize,
) -> Result<usize> {
    let reserved_outputs = generated_limit.min(LAGUNA_INITIAL_DECODE_CAPACITY_TOKENS);
    let capacity = prompt_tokens
        .checked_add(reserved_outputs.saturating_sub(1))
        .ok_or_else(|| Error::runtime("Laguna initial context capacity overflow"))?;
    if capacity == 0 || capacity > max_context_tokens {
        return Err(Error::runtime(format!(
            "Laguna initial context capacity must be within 1..={max_context_tokens}, got {capacity}"
        )));
    }
    Ok(capacity)
}

fn next_context_capacity(
    current_capacity: usize,
    required_capacity: usize,
    max_context_tokens: usize,
) -> Result<usize> {
    if required_capacity <= current_capacity {
        return Ok(current_capacity);
    }
    if required_capacity > max_context_tokens {
        return Err(Error::runtime(format!(
            "Laguna sequence requires {required_capacity} context tokens, maximum is {max_context_tokens}"
        )));
    }
    Ok(current_capacity
        .saturating_mul(2)
        .max(required_capacity)
        .min(max_context_tokens))
}

/// Runs batch-1 greedy generation with Laguna's native FP8 full/sliding KV
/// caches and indexed on-demand INT4 expert cache.
pub fn run_laguna_generate_streaming<B, F>(
    model: &LagunaModel,
    backend: &B,
    prompt_token_ids: &[u32],
    max_new_tokens: Option<usize>,
    stop_token_ids: &[u32],
    options: LagunaGenerationOptions,
    on_token: F,
) -> Result<LagunaGenerationReport>
where
    B: Backend,
    F: FnMut(u32) -> Result<()>,
{
    LagunaRuntime::new(options)?.generate_streaming(
        model,
        backend,
        prompt_token_ids,
        max_new_tokens,
        stop_token_ids,
        on_token,
    )
}

fn final_chunk_start(token_count: usize, chunk_size: usize) -> Result<usize> {
    if token_count == 0 || chunk_size == 0 {
        return Err(Error::runtime(
            "Laguna prefill token count and chunk size must be positive",
        ));
    }
    Ok(((token_count - 1) / chunk_size) * chunk_size)
}

#[cfg(test)]
mod tests {
    use super::{
        final_chunk_start, initial_context_capacity, next_context_capacity,
        LagunaGenerationOptions, LagunaRuntime, LAGUNA_INITIAL_DECODE_CAPACITY_TOKENS,
        LAGUNA_PREFILL_CHUNK_TOKENS,
    };
    use crate::LagunaMemoryControllerSpec;

    #[test]
    fn persistent_runtime_requires_a_real_expert_cache() {
        let error = LagunaRuntime::new(LagunaGenerationOptions {
            expert_cache_capacity: 0,
            memory_controller: None,
        })
        .unwrap_err();
        assert!(error.to_string().contains("must be positive"));

        let runtime = LagunaRuntime::new(LagunaGenerationOptions {
            expert_cache_capacity: 10,
            memory_controller: None,
        })
        .unwrap();
        assert_eq!(runtime.expert_cache_capacity(), 10);
    }

    #[test]
    fn controller_and_runtime_must_start_with_the_same_capacity() {
        let error = LagunaRuntime::new(LagunaGenerationOptions {
            expert_cache_capacity: 10,
            memory_controller: Some(LagunaMemoryControllerSpec {
                initial_expert_capacity: 11,
                minimum_expert_capacity: 10,
                maximum_expert_capacity: 12,
                expert_capacity_step: 1,
                bytes_per_expert: 1,
                decision_window_tokens: 2,
                trial_warmup_tokens: 1,
                stabilization_windows: 0,
                target_headroom_bytes: 2,
                hard_headroom_bytes: 1,
            }),
        })
        .unwrap_err();

        assert!(error.to_string().contains("does not match"));
    }

    #[test]
    fn prefill_keeps_one_non_empty_final_chunk_for_next_token_projection() {
        assert_eq!(
            final_chunk_start(1, LAGUNA_PREFILL_CHUNK_TOKENS).unwrap(),
            0
        );
        assert_eq!(
            final_chunk_start(LAGUNA_PREFILL_CHUNK_TOKENS, LAGUNA_PREFILL_CHUNK_TOKENS).unwrap(),
            0
        );
        assert_eq!(
            final_chunk_start(LAGUNA_PREFILL_CHUNK_TOKENS + 1, LAGUNA_PREFILL_CHUNK_TOKENS)
                .unwrap(),
            LAGUNA_PREFILL_CHUNK_TOKENS
        );
        assert_eq!(
            final_chunk_start(2 * LAGUNA_PREFILL_CHUNK_TOKENS, LAGUNA_PREFILL_CHUNK_TOKENS)
                .unwrap(),
            LAGUNA_PREFILL_CHUNK_TOKENS
        );
    }

    #[test]
    fn unlimited_generation_reserves_a_small_initial_decode_window() {
        let capacity =
            initial_context_capacity(1_000, 200_000, 262_144).expect("valid context capacity");
        assert_eq!(capacity, 1_000 + LAGUNA_INITIAL_DECODE_CAPACITY_TOKENS - 1);

        assert_eq!(initial_context_capacity(1_000, 8, 262_144).unwrap(), 1_007);
    }

    #[test]
    fn context_growth_doubles_without_exceeding_the_model_limit() {
        assert_eq!(next_context_capacity(1_511, 1_512, 262_144).unwrap(), 3_022);
        assert_eq!(
            next_context_capacity(200_000, 200_001, 262_144).unwrap(),
            262_144
        );
        assert!(next_context_capacity(262_144, 262_145, 262_144).is_err());
    }
}
