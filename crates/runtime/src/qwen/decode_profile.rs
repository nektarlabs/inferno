use super::{DraftStats, Instant, QwenModelState, Result};

/// Opt-in block-aligned windows; no extra GPU synchronization or tensor readback.
pub(super) struct DecodeProfile {
    started: Option<Instant>,
    emitted: usize,
    stats: DraftStats,
}

impl DecodeProfile {
    pub(super) fn new() -> Self {
        Self {
            started: tracing::enabled!(target: "inferno::qwen::decode", tracing::Level::DEBUG)
                .then(Instant::now),
            // The first token belongs to prefill, not decode throughput.
            emitted: 1,
            stats: DraftStats::default(),
        }
    }

    pub(super) fn record(
        &mut self,
        emitted: usize,
        stats: &DraftStats,
        state: &QwenModelState,
        final_window: bool,
    ) -> Result<()> {
        let Some(started) = self.started else {
            return Ok(());
        };
        let tokens = emitted.saturating_sub(self.emitted);
        if !window_ready(tokens, final_window) {
            return Ok(());
        }

        let now = Instant::now();
        let elapsed = now.duration_since(started).as_secs_f64();
        let proposed = stats.draft_tokens - self.stats.draft_tokens;
        let accepted = stats.accepted - self.stats.accepted;
        tracing::debug!(
            target: "inferno::qwen::decode",
            generated_start = self.emitted,
            generated_end = emitted,
            context_tokens = state.layers().first().map(|layer| layer.processed_tokens()),
            window_tokens = tokens,
            decode_tps = tokens as f64 / elapsed,
            elapsed_seconds = elapsed,
            draft_seconds = (stats.draft_time - self.stats.draft_time).as_secs_f64(),
            target_seconds = (stats.target_time - self.stats.target_time).as_secs_f64(),
            rollback_seconds = (stats.rollback_time - self.stats.rollback_time).as_secs_f64(),
            proposed,
            accepted,
            acceptance = (proposed > 0).then(|| accepted as f64 / proposed as f64),
            target_state_gib = state.storage_bytes()? as f64 / 1_073_741_824.0,
            "Qwen decode window"
        );
        self.started = Some(now);
        self.emitted = emitted;
        self.stats = stats.clone();
        Ok(())
    }
}

fn window_ready(tokens: usize, final_window: bool) -> bool {
    tokens > 0 && (tokens >= 128 || final_window)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_whole_verification_blocks_and_final_remainder() {
        assert!(!window_ready(0, true));
        assert!(!window_ready(127, false));
        assert!(window_ready(128, false));
        assert!(window_ready(134, false));
        assert!(window_ready(3, true));
    }
}
