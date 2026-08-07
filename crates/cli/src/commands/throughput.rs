use std::{
    io::{self, IsTerminal, Write},
    path::Path,
    time::{Duration, Instant},
};

use common::{Error, Result};

const REFRESH_INTERVAL: Duration = Duration::from_millis(500);

pub(super) struct LiveThroughputDisplay {
    enabled: bool,
    stderr_is_terminal: bool,
    started_at: Instant,
    prompt_tokens: usize,
    processed_prefill_tokens: usize,
    generated_tokens: usize,
    first_token_at: Option<Duration>,
    last_token_at: Option<Duration>,
    last_rendered_at: Option<Duration>,
    terminal_line_visible: bool,
}

impl LiveThroughputDisplay {
    pub(super) fn new(enabled: bool) -> Self {
        Self {
            enabled,
            stderr_is_terminal: io::stderr().is_terminal(),
            started_at: Instant::now(),
            prompt_tokens: 0,
            processed_prefill_tokens: 0,
            generated_tokens: 0,
            first_token_at: None,
            last_token_at: None,
            last_rendered_at: None,
            terminal_line_visible: false,
        }
    }

    pub(super) fn begin_turn(&mut self, prompt_tokens: usize) -> Result<()> {
        self.started_at = Instant::now();
        self.prompt_tokens = prompt_tokens;
        self.processed_prefill_tokens = 0;
        self.generated_tokens = 0;
        self.first_token_at = None;
        self.last_token_at = None;
        self.last_rendered_at = None;
        self.render(true)
    }

    pub(super) fn record_prefill(&mut self, processed_tokens: usize) -> Result<()> {
        self.processed_prefill_tokens = processed_tokens.min(self.prompt_tokens);
        self.render(false)
    }

    pub(super) fn record_token(&mut self) -> Result<()> {
        let elapsed = self.started_at.elapsed();
        self.generated_tokens = self
            .generated_tokens
            .checked_add(1)
            .ok_or_else(|| Error::runtime("live throughput token count overflow"))?;
        self.first_token_at.get_or_insert(elapsed);
        self.last_token_at = Some(elapsed);
        self.processed_prefill_tokens = self.prompt_tokens;
        self.render(self.generated_tokens == 1)
    }

    pub(super) fn finish_request(&mut self) -> Result<()> {
        self.render(true)?;
        if self.enabled && self.stderr_is_terminal && self.terminal_line_visible {
            let mut stderr = io::stderr().lock();
            writeln!(stderr).map_err(stderr_error)?;
            stderr.flush().map_err(stderr_error)?;
            self.terminal_line_visible = false;
        }
        Ok(())
    }

    fn render(&mut self, force: bool) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        let elapsed = self.started_at.elapsed();
        if !force
            && self
                .last_rendered_at
                .is_some_and(|last| elapsed.saturating_sub(last) < REFRESH_INTERVAL)
        {
            return Ok(());
        }

        let (prefill_tps, decode_tps) = rates(
            self.prompt_tokens,
            self.processed_prefill_tokens,
            self.generated_tokens,
            elapsed,
            self.first_token_at,
            self.last_token_at,
        );
        let mut stderr = io::stderr().lock();
        if self.stderr_is_terminal {
            write!(
                stderr,
                "\r\x1b[2Kprefill_tps={prefill_tps:.3} decode_tps={decode_tps:.3}"
            )
            .map_err(stderr_error)?;
            self.terminal_line_visible = true;
        } else {
            writeln!(
                stderr,
                "prefill_tps={prefill_tps:.3} decode_tps={decode_tps:.3}"
            )
            .map_err(stderr_error)?;
        }
        stderr.flush().map_err(stderr_error)?;
        self.last_rendered_at = Some(elapsed);
        Ok(())
    }
}

impl Drop for LiveThroughputDisplay {
    fn drop(&mut self) {
        if self.enabled && self.stderr_is_terminal && self.terminal_line_visible {
            let _ = writeln!(io::stderr().lock());
        }
    }
}

fn rates(
    prompt_tokens: usize,
    processed_prefill_tokens: usize,
    generated_tokens: usize,
    elapsed: Duration,
    first_token_at: Option<Duration>,
    last_token_at: Option<Duration>,
) -> (f64, f64) {
    let prefill_elapsed = first_token_at.unwrap_or(elapsed).as_secs_f64();
    let prefill_tokens = if first_token_at.is_some() {
        prompt_tokens
    } else {
        processed_prefill_tokens
    };
    let prefill_tps = rate(prefill_tokens, prefill_elapsed);
    let decode_tps = match (first_token_at, last_token_at) {
        (Some(first), Some(last)) if generated_tokens > 1 => rate(
            generated_tokens - 1,
            last.saturating_sub(first).as_secs_f64(),
        ),
        _ => 0.0,
    };
    (prefill_tps, decode_tps)
}

fn rate(tokens: usize, seconds: f64) -> f64 {
    if seconds <= 0.0 {
        return 0.0;
    }
    tokens as f64 / seconds
}

fn stderr_error(source: io::Error) -> Error {
    Error::Io {
        path: Path::new("<stderr>").to_path_buf(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_prefill_progress_before_decode() {
        let (prefill_tps, decode_tps) = rates(1_000, 400, 0, Duration::from_secs(2), None, None);

        assert_eq!(prefill_tps, 200.0);
        assert_eq!(decode_tps, 0.0);
    }

    #[test]
    fn separates_prefill_and_decode_rates() {
        let (prefill_tps, decode_tps) = rates(
            1_000,
            1_000,
            5,
            Duration::from_secs(4),
            Some(Duration::from_secs(2)),
            Some(Duration::from_secs(4)),
        );

        assert_eq!(prefill_tps, 500.0);
        assert_eq!(decode_tps, 2.0);
    }
}
