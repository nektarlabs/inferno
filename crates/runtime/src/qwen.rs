use std::{path::Path, time::Duration, time::Instant};

use backend::Backend;
use common::{Error, Result};
use config::{
    load_dflash_config, load_qwen_config, load_qwen_mtp_config, DFlashConfig, QwenConfig,
};
use model::{
    draft_qwen_next_token_device_handle, forward_dflash_proposals_device,
    forward_qwen_hidden_device, forward_qwen_hidden_with_dflash_features_device,
    forward_qwen_mtp_hidden_device, forward_qwen_mtp_hidden_from_device_tokens,
    greedy_all_qwen_tokens_device, greedy_qwen_next_tokens_device, DFlashDeviceWeights,
    DFlashState, DFlashWeightIndex, QwenArtifactSummary, QwenDeviceMtpWeights,
    QwenDeviceWeightSummary, QwenDeviceWeights, QwenModelState, QwenMtpState, QwenMtpWeightIndex,
    QwenWeightIndex, QWEN_MTP_DIRECTORY,
};
use tracing::{debug, trace};

const DFLASH_M5_DRAFT_DEPTH: usize = 4;
// DFlash2 checkpoint maximum: one accepted anchor plus seven draft tokens.
const DFLASH_M8_DRAFT_DEPTH: usize = 7;
const DFLASH_ACCEPTANCE_WINDOW: usize = 4;
const DFLASH_PROMOTION_ACCEPTANCE_NUMERATOR: usize = 9;
const DFLASH_PROMOTION_ACCEPTANCE_DENOMINATOR: usize = 10;
const DEFAULT_MTP_DRAFT_DEPTH: usize = 2;

#[cfg(test)]
#[path = "qwen/benchmarks.rs"]
mod benchmarks;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenDraftMethod {
    Mtp,
    DFlash2,
}

impl QwenDraftMethod {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Mtp => "mtp",
            Self::DFlash2 => "dflash2",
        }
    }
}

enum QwenDrafter {
    Mtp {
        weights: Box<QwenDeviceMtpWeights>,
        state: QwenMtpState,
    },
    DFlash2 {
        config: DFlashConfig,
        weights: Box<DFlashDeviceWeights>,
        state: DFlashState,
    },
}

impl QwenDrafter {
    fn method(&self) -> QwenDraftMethod {
        match self {
            Self::Mtp { .. } => QwenDraftMethod::Mtp,
            Self::DFlash2 { .. } => QwenDraftMethod::DFlash2,
        }
    }

    fn state_bytes(&self) -> Result<usize> {
        match self {
            Self::Mtp { state, .. } => state.storage_bytes(),
            Self::DFlash2 { state, .. } => state.storage_bytes(),
        }
    }

    fn max_draft_depth(&self) -> usize {
        match self {
            Self::Mtp { .. } => DEFAULT_MTP_DRAFT_DEPTH,
            Self::DFlash2 { config, .. } => dflash_max_draft_depth(config),
        }
    }
}

pub struct QwenRuntime<'a, B> {
    backend: &'a B,
    config: QwenConfig,
    weights: QwenDeviceWeights,
    state: QwenModelState,
    drafter: QwenDrafter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenRuntimeSummary {
    pub artifact: QwenArtifactSummary,
    pub weights: QwenDeviceWeightSummary,
    pub state_bytes: usize,
    pub draft_method: QwenDraftMethod,
    pub draft_state_bytes: usize,
    pub capacity_tokens: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QwenGenerationReport {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub stopped_on_eos: bool,
    pub draft_method: QwenDraftMethod,
    pub verification_passes: usize,
    pub target_tokens: usize,
    pub draft_tokens: usize,
    pub accepted_draft_tokens: usize,
    pub max_draft_depth: usize,
    pub dflash_m5_passes: usize,
    pub dflash_m8_passes: usize,
    pub dflash_depth_switches: usize,
    pub draft_seconds: f64,
    pub target_seconds: f64,
    pub rollback_seconds: f64,
}

impl<'a, B: Backend> QwenRuntime<'a, B> {
    /// Opens the MLX W4 target with its Q4 MTP companion enabled by default.
    pub fn open(
        model_dir: &Path,
        config_path: &Path,
        backend: &'a B,
        batch: usize,
        capacity_tokens: usize,
    ) -> Result<Self> {
        let (config, weights, state) = open_target(
            model_dir,
            config_path,
            backend,
            batch,
            capacity_tokens,
            true,
        )?;
        let mtp_dir = model_dir.join(QWEN_MTP_DIRECTORY);
        if !mtp_dir.is_dir() {
            return Err(Error::weights(format!(
                "Qwen Q4 requires its default MTP companion in {}; download mlx-community/Qwen3.8-27B-MTP-4bit into that directory",
                mtp_dir.display()
            )));
        }
        load_qwen_mtp_config(&mtp_dir.join("config.json"), &config)?;
        let mtp_source = QwenMtpWeightIndex::open(&mtp_dir, &config)?;
        let mtp_weights = QwenDeviceMtpWeights::prepare(&mtp_source, backend)?;
        let mtp_state = QwenMtpState::create(backend, batch, capacity_tokens)?;
        Ok(Self {
            backend,
            config,
            weights,
            state,
            drafter: QwenDrafter::Mtp {
                weights: Box::new(mtp_weights),
                state: mtp_state,
            },
        })
    }

    /// Opens Qwen with the external DFlash2 draft model. MTP weights and state
    /// are deliberately not loaded on this path.
    pub fn open_with_dflash(
        model_dir: &Path,
        config_path: &Path,
        dflash_model_dir: &Path,
        backend: &'a B,
        batch: usize,
        capacity_tokens: usize,
    ) -> Result<Self> {
        let (config, weights, state) = open_target(
            model_dir,
            config_path,
            backend,
            batch,
            capacity_tokens,
            true,
        )?;
        let dflash_config = load_dflash_config(&dflash_model_dir.join("config.json"))?;
        validate_dflash_target_compatibility(&config, &dflash_config)?;
        let dflash_source = DFlashWeightIndex::open(dflash_model_dir, &dflash_config)?;
        let dflash_weights = DFlashDeviceWeights::prepare(&dflash_source, backend)?;
        let dflash_state = DFlashState::create(backend, &dflash_config, batch)?;
        Ok(Self {
            backend,
            config,
            weights,
            state,
            drafter: QwenDrafter::DFlash2 {
                config: dflash_config,
                weights: Box::new(dflash_weights),
                state: dflash_state,
            },
        })
    }

    pub fn config(&self) -> &QwenConfig {
        &self.config
    }

    pub fn summary(&self) -> Result<QwenRuntimeSummary> {
        Ok(QwenRuntimeSummary {
            artifact: self.weights.artifact,
            weights: self.weights.summary,
            state_bytes: self.state.storage_bytes()?,
            draft_method: self.drafter.method(),
            draft_state_bytes: self.drafter.state_bytes()?,
            capacity_tokens: self.state.capacity_tokens(),
        })
    }

    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: Option<usize>,
        eos_token_ids: &[u32],
        on_token: impl FnMut(u32) -> Result<()>,
    ) -> Result<QwenGenerationReport> {
        let generation_limit = validate_generation_inputs(
            prompt_tokens,
            max_new_tokens,
            eos_token_ids,
            self.state.capacity_tokens(),
        )?;
        if generation_limit == 0 {
            return Ok(empty_report(
                prompt_tokens.len(),
                self.drafter.method(),
                self.drafter.max_draft_depth(),
            ));
        }
        trace!(
            target: "inferno::qwen::tokens",
            prompt_tokens = ?prompt_tokens,
            draft_method = self.drafter.method().as_str(),
            "starting Qwen generation"
        );

        match &mut self.drafter {
            QwenDrafter::Mtp { weights, state } => generate_mtp(
                self.backend,
                &self.config,
                &self.weights,
                &mut self.state,
                weights,
                state,
                prompt_tokens,
                generation_limit,
                eos_token_ids,
                on_token,
            ),
            QwenDrafter::DFlash2 {
                config,
                weights,
                state,
            } => generate_dflash(
                self.backend,
                &self.config,
                &self.weights,
                &mut self.state,
                config,
                weights,
                state,
                prompt_tokens,
                generation_limit,
                eos_token_ids,
                on_token,
            ),
        }
    }
}

fn open_target<B: Backend>(
    model_dir: &Path,
    config_path: &Path,
    backend: &B,
    batch: usize,
    capacity_tokens: usize,
    speculative: bool,
) -> Result<(QwenConfig, QwenDeviceWeights, QwenModelState)> {
    let config = load_qwen_config(config_path)?;
    if capacity_tokens == 0 || capacity_tokens > config.text_config.max_position_embeddings {
        return Err(Error::runtime(format!(
            "Qwen state capacity must be in 1..={}, got {capacity_tokens}",
            config.text_config.max_position_embeddings
        )));
    }
    let source = QwenWeightIndex::open(model_dir, &config)?;
    // Packed U32 payloads are copied once into aligned, persistent Metal
    // buffers because MLX Safetensors does not guarantee four-byte offsets.
    let weights = QwenDeviceWeights::prepare_resident(&source, &config, backend)?;
    let state = if speculative {
        QwenModelState::create_speculative(backend, &config, batch, capacity_tokens)?
    } else {
        QwenModelState::create(backend, &config, batch, capacity_tokens)?
    };
    Ok((config, weights, state))
}

#[allow(clippy::too_many_arguments)]
fn generate_mtp<B: Backend>(
    backend: &B,
    config: &QwenConfig,
    weights: &QwenDeviceWeights,
    state: &mut QwenModelState,
    mtp_weights: &QwenDeviceMtpWeights,
    mtp_state: &mut QwenMtpState,
    prompt_tokens: &[u32],
    generation_limit: usize,
    eos_token_ids: &[u32],
    mut on_token: impl FnMut(u32) -> Result<()>,
) -> Result<QwenGenerationReport> {
    let prompt_hidden = forward_qwen_hidden_device(
        backend,
        config,
        weights,
        state,
        prompt_tokens,
        &[1, prompt_tokens.len()],
    )?;
    let mut primary = one_batch_token(greedy_qwen_next_tokens_device(
        backend,
        weights,
        &prompt_hidden,
    )?)?;
    let mut pending_mtp_hidden = prompt_hidden;
    let mut pending_mtp_tokens = prompt_tokens[1..].to_vec();
    pending_mtp_tokens.push(primary);
    let mut stats = DraftStats::default();
    let mut generated = 0_usize;
    let mut stopped_on_eos = false;

    while generated < generation_limit {
        on_token(primary)?;
        generated += 1;
        if eos_token_ids.contains(&primary) {
            stopped_on_eos = true;
            break;
        }
        if generated == generation_limit {
            break;
        }

        let draft_count = DEFAULT_MTP_DRAFT_DEPTH.min(generation_limit - generated);
        let started = Instant::now();
        let mut draft_hidden = forward_qwen_mtp_hidden_device(
            backend,
            config,
            &weights.root,
            mtp_weights,
            mtp_state,
            &pending_mtp_hidden,
            &pending_mtp_tokens,
        )?;
        let mut draft_handles = Vec::with_capacity(draft_count);
        for draft_index in 0..draft_count {
            let draft = draft_qwen_next_token_device_handle(backend, weights, &draft_hidden)?;
            if draft_index + 1 < draft_count {
                let last_row = draft_hidden.dims()[1] - 1;
                let chain_hidden = backend
                    .qwen_bf16_copy_row_device(&draft_hidden, last_row)?
                    .ok_or_else(|| {
                        Error::backend("Qwen MTP row selection requires native Metal")
                    })?;
                draft_hidden = forward_qwen_mtp_hidden_from_device_tokens(
                    backend,
                    config,
                    &weights.root,
                    mtp_weights,
                    mtp_state,
                    &chain_hidden,
                    &draft,
                )?;
            }
            draft_handles.push(draft);
        }
        let drafts = draft_handles
            .iter()
            .map(|draft| one_batch_token(backend.qwen_read_token_ids(draft)?))
            .collect::<Result<Vec<_>>>()?;
        stats.draft_time += started.elapsed();
        stats.draft_tokens += draft_count;

        let started = Instant::now();
        let mut verify_input = Vec::with_capacity(draft_count + 1);
        verify_input.push(primary);
        verify_input.extend_from_slice(&drafts);
        stats.verification_passes += 1;
        stats.target_tokens += verify_input.len();
        let verify_hidden = forward_qwen_hidden_device(
            backend,
            config,
            weights,
            state,
            &verify_input,
            &[1, verify_input.len()],
        )?;
        let verify_tokens = greedy_all_qwen_tokens_device(backend, weights, &verify_hidden)?;
        stats.target_time += started.elapsed();
        validate_verification_count(&verify_tokens, draft_count)?;
        let accepted = accepted_draft_prefix(&drafts, &verify_tokens);
        stats.accepted += accepted;
        mtp_state.rewind_speculative(draft_count.saturating_sub(1))?;
        restore_target_after_verification(
            backend,
            state,
            accepted,
            draft_count,
            &mut stats.rollback_time,
        )?;

        emit_accepted_drafts(
            &drafts[..accepted],
            eos_token_ids,
            generation_limit,
            &mut generated,
            &mut stopped_on_eos,
            &mut on_token,
        )?;
        if stopped_on_eos || generated == generation_limit {
            break;
        }

        primary = verify_tokens[accepted];
        pending_mtp_hidden = backend
            .qwen_bf16_prefix_rows_device(&verify_hidden, accepted + 1)?
            .ok_or_else(|| Error::backend("Qwen MTP prefix selection requires native Metal"))?;
        pending_mtp_tokens.clear();
        pending_mtp_tokens.extend_from_slice(&drafts[..accepted]);
        pending_mtp_tokens.push(primary);
    }

    log_draft_profile(QwenDraftMethod::Mtp, &stats);
    Ok(stats.report(
        prompt_tokens.len(),
        generated,
        stopped_on_eos,
        QwenDraftMethod::Mtp,
        DEFAULT_MTP_DRAFT_DEPTH,
    ))
}

#[allow(clippy::too_many_arguments)]
fn generate_dflash<B: Backend>(
    backend: &B,
    target_config: &QwenConfig,
    target_weights: &QwenDeviceWeights,
    target_state: &mut QwenModelState,
    dflash_config: &DFlashConfig,
    dflash_weights: &DFlashDeviceWeights,
    dflash_state: &mut DFlashState,
    prompt_tokens: &[u32],
    generation_limit: usize,
    eos_token_ids: &[u32],
    mut on_token: impl FnMut(u32) -> Result<()>,
) -> Result<QwenGenerationReport> {
    let (prompt_hidden, prompt_features) = forward_qwen_hidden_with_dflash_features_device(
        backend,
        target_config,
        dflash_config,
        target_weights,
        target_state,
        prompt_tokens,
        &[1, prompt_tokens.len()],
    )?;
    let mut primary = one_batch_token(greedy_qwen_next_tokens_device(
        backend,
        target_weights,
        &prompt_hidden,
    )?)?;
    let mut pending_features = prompt_features;
    let mut pending_feature_start = prompt_tokens.len() - pending_features.dims()[1];
    let mut target_context_length = prompt_tokens.len();
    let max_draft_depth = dflash_max_draft_depth(dflash_config);
    let mut depth_controller = DFlashDepthController::new();
    let mut stats = DraftStats::default();
    let mut generated = 0_usize;
    let mut stopped_on_eos = false;

    while generated < generation_limit {
        on_token(primary)?;
        generated += 1;
        if eos_token_ids.contains(&primary) {
            stopped_on_eos = true;
            break;
        }
        if generated == generation_limit {
            break;
        }

        let draft_count = depth_controller
            .draft_depth()
            .min(generation_limit - generated);
        stats.record_dflash_pass(draft_count);
        let mut block_tokens = Vec::with_capacity(draft_count + 1);
        block_tokens.push(primary);
        block_tokens.resize(draft_count + 1, dflash_config.dflash_config.mask_token_id);
        let started = Instant::now();
        let draft_handle = forward_dflash_proposals_device(
            backend,
            dflash_config,
            &target_weights.root,
            dflash_weights,
            dflash_state,
            &pending_features,
            pending_feature_start,
            &block_tokens,
        )?;
        let drafts = backend.qwen_read_token_ids(&draft_handle)?;
        stats.draft_time += started.elapsed();
        stats.draft_tokens += draft_count;
        if drafts.len() != draft_count {
            return Err(Error::runtime(format!(
                "DFlash2 returned {} proposals, expected {draft_count}",
                drafts.len()
            )));
        }

        let started = Instant::now();
        let mut verify_input = Vec::with_capacity(draft_count + 1);
        verify_input.push(primary);
        verify_input.extend_from_slice(&drafts);
        stats.verification_passes += 1;
        stats.target_tokens += verify_input.len();
        let (verify_hidden, verify_features) = forward_qwen_hidden_with_dflash_features_device(
            backend,
            target_config,
            dflash_config,
            target_weights,
            target_state,
            &verify_input,
            &[1, verify_input.len()],
        )?;
        let verify_tokens = greedy_all_qwen_tokens_device(backend, target_weights, &verify_hidden)?;
        stats.target_time += started.elapsed();
        validate_verification_count(&verify_tokens, draft_count)?;
        let accepted = accepted_draft_prefix(&drafts, &verify_tokens);
        trace!(
            target: "inferno::qwen::tokens",
            primary,
            draft_count,
            drafts = ?drafts,
            target_tokens = ?verify_tokens,
            accepted_drafts = accepted,
            target_context_length,
            "verified DFlash2 block"
        );
        stats.accepted += accepted;
        if let Some(change) = depth_controller.observe(accepted, draft_count) {
            stats.dflash_depth_switches += 1;
            debug!(
                previous_block_size = change.previous_draft_depth + 1,
                next_block_size = change.next_draft_depth + 1,
                window_accepted = change.window_accepted,
                window_drafts = change.window_drafts,
                window_acceptance_rate =
                    change.window_accepted as f64 / change.window_drafts as f64,
                "adjusted DFlash2 verification block size"
            );
        }
        restore_target_after_verification(
            backend,
            target_state,
            accepted,
            draft_count,
            &mut stats.rollback_time,
        )?;

        emit_accepted_drafts(
            &drafts[..accepted],
            eos_token_ids,
            generation_limit,
            &mut generated,
            &mut stopped_on_eos,
            &mut on_token,
        )?;
        if stopped_on_eos || generated == generation_limit {
            break;
        }

        primary = verify_tokens[accepted];
        pending_features = backend
            .qwen_bf16_prefix_rows_device(&verify_features, accepted + 1)?
            .ok_or_else(|| {
                Error::backend("DFlash2 accepted feature selection requires native Metal")
            })?;
        pending_feature_start = target_context_length;
        target_context_length = target_context_length
            .checked_add(accepted + 1)
            .ok_or_else(|| Error::runtime("DFlash2 target context length overflow"))?;
    }

    log_draft_profile(QwenDraftMethod::DFlash2, &stats);
    Ok(stats.report(
        prompt_tokens.len(),
        generated,
        stopped_on_eos,
        QwenDraftMethod::DFlash2,
        max_draft_depth,
    ))
}

#[derive(Default)]
struct DraftStats {
    verification_passes: usize,
    target_tokens: usize,
    draft_tokens: usize,
    accepted: usize,
    dflash_m5_passes: usize,
    dflash_m8_passes: usize,
    dflash_depth_switches: usize,
    draft_time: Duration,
    target_time: Duration,
    rollback_time: Duration,
}

impl DraftStats {
    fn record_dflash_pass(&mut self, draft_count: usize) {
        match draft_count {
            DFLASH_M5_DRAFT_DEPTH => self.dflash_m5_passes += 1,
            DFLASH_M8_DRAFT_DEPTH => self.dflash_m8_passes += 1,
            _ => {}
        }
    }

    fn report(
        &self,
        prompt_tokens: usize,
        generated_tokens: usize,
        stopped_on_eos: bool,
        draft_method: QwenDraftMethod,
        max_draft_depth: usize,
    ) -> QwenGenerationReport {
        QwenGenerationReport {
            prompt_tokens,
            generated_tokens,
            stopped_on_eos,
            draft_method,
            verification_passes: self.verification_passes,
            target_tokens: self.target_tokens,
            draft_tokens: self.draft_tokens,
            accepted_draft_tokens: self.accepted,
            max_draft_depth,
            dflash_m5_passes: self.dflash_m5_passes,
            dflash_m8_passes: self.dflash_m8_passes,
            dflash_depth_switches: self.dflash_depth_switches,
            draft_seconds: self.draft_time.as_secs_f64(),
            target_seconds: self.target_time.as_secs_f64(),
            rollback_seconds: self.rollback_time.as_secs_f64(),
        }
    }
}

fn validate_generation_inputs(
    prompt_tokens: &[u32],
    max_new_tokens: Option<usize>,
    eos_token_ids: &[u32],
    capacity_tokens: usize,
) -> Result<usize> {
    if prompt_tokens.is_empty() {
        return Err(Error::runtime("Qwen generation requires prompt tokens"));
    }
    if prompt_tokens.len() > capacity_tokens {
        return Err(Error::runtime(format!(
            "Qwen prompt has {} tokens but runtime capacity is {capacity_tokens}",
            prompt_tokens.len()
        )));
    }
    if eos_token_ids.is_empty() {
        return Err(Error::runtime(
            "Qwen generation requires at least one EOS token",
        ));
    }
    let available = capacity_tokens - prompt_tokens.len();
    Ok(max_new_tokens.unwrap_or(available).min(available))
}

fn validate_dflash_target_compatibility(target: &QwenConfig, draft: &DFlashConfig) -> Result<()> {
    let compatible = draft.hidden_size == target.text_config.hidden_size
        && draft.intermediate_size == target.text_config.intermediate_size
        && draft.num_target_layers == target.text_config.num_hidden_layers
        && draft.vocab_size == target.text_config.vocab_size
        && draft.max_position_embeddings == target.text_config.max_position_embeddings
        && draft.eos_token_id == target.text_config.eos_token_id;
    if !compatible {
        return Err(Error::config(
            "DFlash2 artifact is not compatible with the loaded Qwen3.8 target",
        ));
    }
    Ok(())
}

fn dflash_max_draft_depth(config: &DFlashConfig) -> usize {
    (config.dflash_config.block_size - 1).min(DFLASH_M8_DRAFT_DEPTH)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DFlashDepthChange {
    previous_draft_depth: usize,
    next_draft_depth: usize,
    window_accepted: usize,
    window_drafts: usize,
}

struct DFlashDepthController {
    draft_depth: usize,
    accepted: [usize; DFLASH_ACCEPTANCE_WINDOW],
    drafts: [usize; DFLASH_ACCEPTANCE_WINDOW],
    next_sample: usize,
    sample_count: usize,
}

impl DFlashDepthController {
    fn new() -> Self {
        Self {
            draft_depth: DFLASH_M5_DRAFT_DEPTH,
            accepted: [0; DFLASH_ACCEPTANCE_WINDOW],
            drafts: [0; DFLASH_ACCEPTANCE_WINDOW],
            next_sample: 0,
            sample_count: 0,
        }
    }

    fn draft_depth(&self) -> usize {
        self.draft_depth
    }

    fn observe(&mut self, accepted: usize, drafts: usize) -> Option<DFlashDepthChange> {
        // A shortened final block is not representative of the selected mode.
        if drafts != self.draft_depth {
            return None;
        }
        self.accepted[self.next_sample] = accepted;
        self.drafts[self.next_sample] = drafts;
        self.next_sample = (self.next_sample + 1) % DFLASH_ACCEPTANCE_WINDOW;
        self.sample_count = (self.sample_count + 1).min(DFLASH_ACCEPTANCE_WINDOW);

        let window_accepted = self.accepted[..self.sample_count].iter().sum::<usize>();
        let window_drafts = self.drafts[..self.sample_count].iter().sum::<usize>();
        let next_draft_depth = match self.draft_depth {
            DFLASH_M8_DRAFT_DEPTH
                if self.sample_count >= 2 && window_accepted.saturating_mul(2) < window_drafts =>
            {
                DFLASH_M5_DRAFT_DEPTH
            }
            DFLASH_M5_DRAFT_DEPTH
                if self.sample_count == DFLASH_ACCEPTANCE_WINDOW
                    && window_accepted.saturating_mul(DFLASH_PROMOTION_ACCEPTANCE_DENOMINATOR)
                        >= window_drafts.saturating_mul(DFLASH_PROMOTION_ACCEPTANCE_NUMERATOR) =>
            {
                DFLASH_M8_DRAFT_DEPTH
            }
            _ => return None,
        };

        let change = DFlashDepthChange {
            previous_draft_depth: self.draft_depth,
            next_draft_depth,
            window_accepted,
            window_drafts,
        };
        self.draft_depth = next_draft_depth;
        self.accepted.fill(0);
        self.drafts.fill(0);
        self.next_sample = 0;
        self.sample_count = 0;
        Some(change)
    }
}

fn restore_target_after_verification<B: Backend>(
    backend: &B,
    state: &mut QwenModelState,
    accepted: usize,
    draft_count: usize,
    rollback_time: &mut Duration,
) -> Result<()> {
    if accepted == draft_count {
        state.accept_speculative_rows();
    } else {
        let started = Instant::now();
        state.restore_speculative_prefix(backend, accepted, draft_count)?;
        *rollback_time += started.elapsed();
    }
    Ok(())
}

fn emit_accepted_drafts(
    drafts: &[u32],
    eos_token_ids: &[u32],
    generation_limit: usize,
    generated: &mut usize,
    stopped_on_eos: &mut bool,
    on_token: &mut impl FnMut(u32) -> Result<()>,
) -> Result<()> {
    for &draft in drafts {
        on_token(draft)?;
        *generated += 1;
        if eos_token_ids.contains(&draft) {
            *stopped_on_eos = true;
            break;
        }
        if *generated == generation_limit {
            break;
        }
    }
    Ok(())
}

fn validate_verification_count(verify_tokens: &[u32], draft_count: usize) -> Result<()> {
    if verify_tokens.len() != draft_count + 1 {
        return Err(Error::runtime(format!(
            "Qwen verification returned {} tokens for {draft_count} drafts",
            verify_tokens.len()
        )));
    }
    Ok(())
}

fn empty_report(
    prompt_tokens: usize,
    draft_method: QwenDraftMethod,
    max_draft_depth: usize,
) -> QwenGenerationReport {
    QwenGenerationReport {
        prompt_tokens,
        generated_tokens: 0,
        stopped_on_eos: false,
        draft_method,
        verification_passes: 0,
        target_tokens: 0,
        draft_tokens: 0,
        accepted_draft_tokens: 0,
        max_draft_depth,
        dflash_m5_passes: 0,
        dflash_m8_passes: 0,
        dflash_depth_switches: 0,
        draft_seconds: 0.0,
        target_seconds: 0.0,
        rollback_seconds: 0.0,
    }
}

fn log_draft_profile(method: QwenDraftMethod, stats: &DraftStats) {
    debug!(
        draft_method = method.as_str(),
        draft_tokens = stats.draft_tokens,
        accepted_draft_tokens = stats.accepted,
        verification_passes = stats.verification_passes,
        target_tokens = stats.target_tokens,
        acceptance_rate = if stats.draft_tokens == 0 {
            0.0
        } else {
            stats.accepted as f64 / stats.draft_tokens as f64
        },
        draft_ms = stats.draft_time.as_secs_f64() * 1_000.0,
        target_ms = stats.target_time.as_secs_f64() * 1_000.0,
        rollback_ms = stats.rollback_time.as_secs_f64() * 1_000.0,
        "Qwen speculative generation profile"
    );
}

fn one_batch_token(token_ids: Vec<u32>) -> Result<u32> {
    let [token_id]: [u32; 1] = token_ids.try_into().map_err(|token_ids: Vec<u32>| {
        Error::runtime(format!(
            "Qwen single-batch generation returned {} token IDs",
            token_ids.len()
        ))
    })?;
    Ok(token_id)
}

fn accepted_draft_prefix(drafts: &[u32], verify_tokens: &[u32]) -> usize {
    drafts
        .iter()
        .zip(verify_tokens)
        .take_while(|(draft, target)| draft == target)
        .count()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use backend::MetalBackend;

    use super::*;

    #[test]
    fn mtp_is_enabled_by_default_for_qwen() {
        assert_eq!(DEFAULT_MTP_DRAFT_DEPTH, 2);
        assert_eq!(QwenDraftMethod::Mtp.as_str(), "mtp");
    }

    #[test]
    fn dflash_uses_the_checkpoint_maximum_block_size() {
        assert_eq!(DFLASH_M8_DRAFT_DEPTH, 7);
        assert_eq!(QwenDraftMethod::DFlash2.as_str(), "dflash2");
    }

    #[test]
    fn dflash_starts_at_m5() {
        let controller = DFlashDepthController::new();
        assert_eq!(controller.draft_depth(), DFLASH_M5_DRAFT_DEPTH);
    }

    #[test]
    fn dflash_promotes_only_after_a_very_high_acceptance_window() {
        let mut controller = DFlashDepthController::new();
        for _ in 0..DFLASH_ACCEPTANCE_WINDOW - 1 {
            assert_eq!(controller.observe(4, DFLASH_M5_DRAFT_DEPTH), None);
        }
        let change = controller
            .observe(4, DFLASH_M5_DRAFT_DEPTH)
            .expect("four fully accepted M5 blocks must promote to M8");
        assert_eq!(change.previous_draft_depth, DFLASH_M5_DRAFT_DEPTH);
        assert_eq!(change.next_draft_depth, DFLASH_M8_DRAFT_DEPTH);
    }

    #[test]
    fn dflash_does_not_promote_at_seventy_five_percent_acceptance() {
        let mut controller = DFlashDepthController::new();
        for _ in 0..DFLASH_ACCEPTANCE_WINDOW * 2 {
            assert_eq!(controller.observe(3, DFLASH_M5_DRAFT_DEPTH), None);
        }
        assert_eq!(controller.draft_depth(), DFLASH_M5_DRAFT_DEPTH);
    }

    #[test]
    fn dflash_demotes_after_sustained_low_m8_acceptance() {
        let mut controller = DFlashDepthController::new();
        for _ in 0..DFLASH_ACCEPTANCE_WINDOW {
            controller.observe(4, DFLASH_M5_DRAFT_DEPTH);
        }
        assert_eq!(controller.draft_depth(), DFLASH_M8_DRAFT_DEPTH);
        assert_eq!(controller.observe(2, DFLASH_M8_DRAFT_DEPTH), None);
        let change = controller
            .observe(3, DFLASH_M8_DRAFT_DEPTH)
            .expect("two low-acceptance M8 blocks must demote");
        assert_eq!(change.previous_draft_depth, DFLASH_M8_DRAFT_DEPTH);
        assert_eq!(change.next_draft_depth, DFLASH_M5_DRAFT_DEPTH);
        assert_eq!(controller.draft_depth(), DFLASH_M5_DRAFT_DEPTH);
    }

    #[test]
    fn dflash_ignores_short_tail_blocks_when_adapting() {
        let mut controller = DFlashDepthController::new();
        assert_eq!(controller.observe(0, 2), None);
        assert_eq!(controller.draft_depth(), DFLASH_M5_DRAFT_DEPTH);
    }

    #[test]
    fn single_batch_output_rejects_missing_or_extra_ids() {
        assert_eq!(one_batch_token(vec![7]).unwrap(), 7);
        assert!(one_batch_token(Vec::new()).is_err());
        assert!(one_batch_token(vec![1, 2]).is_err());
    }

    #[test]
    fn verification_accepts_only_the_exact_draft_prefix() {
        assert_eq!(accepted_draft_prefix(&[3, 5, 7], &[3, 5, 7, 11]), 3);
        assert_eq!(accepted_draft_prefix(&[3, 5, 7], &[3, 13, 7, 11]), 1);
        assert_eq!(accepted_draft_prefix(&[3, 5, 7], &[17, 5, 7, 11]), 0);
    }

    #[test]
    #[ignore = "requires the local 15 GB Qwen W4 and DFlash2 artifacts"]
    fn real_qwen_m8_rollback_matches_sequential_target_tokens() {
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let model_dir = std::env::var_os("INFERNO_QWEN_W4_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| workspace.join("models/qwen3.8-27b-4bit"));
        let dflash_dir = std::env::var_os("INFERNO_QWEN_DFLASH_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| workspace.join("models/qwen3.8-27b-dflash2"));
        let backend = MetalBackend::new().unwrap();
        let (config, weights, mut candidate_state) = open_target(
            &model_dir,
            &model_dir.join("config.json"),
            &backend,
            1,
            128,
            true,
        )
        .unwrap();
        let dflash_config = load_dflash_config(&dflash_dir.join("config.json")).unwrap();
        let mut reference_state = QwenModelState::create(&backend, &config, 1, 128).unwrap();
        let prompt = [
            248_045, 846, 198, 39_113, 728, 279, 6_511, 314, 14_898, 13, 248_046, 198, 248_045,
            74_455, 198, 248_068, 271, 248_069, 271,
        ];

        let (candidate_prompt, _) = forward_qwen_hidden_with_dflash_features_device(
            &backend,
            &config,
            &dflash_config,
            &weights,
            &mut candidate_state,
            &prompt,
            &[1, prompt.len()],
        )
        .unwrap();
        let reference_prompt = forward_qwen_hidden_device(
            &backend,
            &config,
            &weights,
            &mut reference_state,
            &prompt,
            &[1, prompt.len()],
        )
        .unwrap();
        assert_eq!(
            greedy_qwen_next_tokens_device(&backend, &weights, &candidate_prompt).unwrap(),
            greedy_qwen_next_tokens_device(&backend, &weights, &reference_prompt).unwrap(),
        );

        let blocks: [(&[u32], usize); 3] = [
            (&[760, 6_511, 314, 14_898, 369, 2_972, 208_639, 159_034], 5),
            (&[49, 617, 159_034, 248_046, 248_046, 248_046, 25, 198], 1),
            (&[332, 318, 208_639, 553, 271], 0),
        ];
        for (block_index, (input, accepted_drafts)) in blocks.into_iter().enumerate() {
            let kept_rows = accepted_drafts + 1;
            let (candidate_hidden, _) = forward_qwen_hidden_with_dflash_features_device(
                &backend,
                &config,
                &dflash_config,
                &weights,
                &mut candidate_state,
                input,
                &[1, input.len()],
            )
            .unwrap();
            let candidate_tokens =
                greedy_all_qwen_tokens_device(&backend, &weights, &candidate_hidden).unwrap();
            candidate_state
                .restore_speculative_prefix(&backend, accepted_drafts, input.len() - 1)
                .unwrap();
            let mut reference_tokens = Vec::with_capacity(kept_rows);
            for &token in &input[..kept_rows] {
                let hidden = forward_qwen_hidden_device(
                    &backend,
                    &config,
                    &weights,
                    &mut reference_state,
                    &[token],
                    &[1, 1],
                )
                .unwrap();
                reference_tokens
                    .extend(greedy_qwen_next_tokens_device(&backend, &weights, &hidden).unwrap());
            }
            assert_eq!(
                &candidate_tokens[..kept_rows],
                reference_tokens,
                "target token mismatch in verification block {block_index}"
            );
        }

        for token in [318] {
            let (candidate_hidden, _) = forward_qwen_hidden_with_dflash_features_device(
                &backend,
                &config,
                &dflash_config,
                &weights,
                &mut candidate_state,
                &[token],
                &[1, 1],
            )
            .unwrap();
            let reference_hidden = forward_qwen_hidden_device(
                &backend,
                &config,
                &weights,
                &mut reference_state,
                &[token],
                &[1, 1],
            )
            .unwrap();
            assert_eq!(
                greedy_qwen_next_tokens_device(&backend, &weights, &candidate_hidden).unwrap(),
                greedy_qwen_next_tokens_device(&backend, &weights, &reference_hidden).unwrap(),
                "target state diverged after consecutive M8 rollbacks while probing token {token}"
            );
        }
    }
}
