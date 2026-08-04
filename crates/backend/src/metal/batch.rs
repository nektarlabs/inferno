use std::sync::Mutex;

use ::metal::{CommandBuffer, CommandBufferRef, CommandQueue, MTLCommandBufferStatus};
use common::{Error, Result};
use objc::{msg_send, sel, sel_impl};
use tracing::trace;

/// Holds the command buffer that accumulates batched kernel dispatches between
/// host synchronization points.
///
/// Ops on the batched (device-resident) path encode their kernels into this
/// shared command buffer instead of committing one command buffer per kernel.
/// `submit` starts the accumulated work without a host wait; `flush` submits
/// anything still open and waits for every retained submission. Dependent
/// kernels inside one batch are ordered by Metal's automatic hazard tracking
/// on the shared-storage buffers they touch, which is the same guarantee the
/// existing `dispatch_1d_many` chains rely on.
///
/// Safety invariant for callers: CPU code must not read from or write into any
/// buffer referenced by an already-encoded kernel until `flush` has returned.
/// Ops therefore allocate fresh output buffers while a batch is open instead
/// of recycling pooled scratch buffers.
#[derive(Default)]
struct BatchState {
    open: Option<CommandBuffer>,
    submitted: Vec<SubmittedBatch>,
}

struct SubmittedBatch {
    command_buffer: CommandBuffer,
    profile_label: Option<String>,
}

pub(crate) struct BatchSlot {
    inner: Mutex<BatchState>,
}

impl BatchSlot {
    pub(crate) fn new() -> Self {
        Self {
            inner: Mutex::new(BatchState::default()),
        }
    }

    /// Runs `encode` against the open batch command buffer, opening a new one
    /// on `queue` if no batch is currently open.
    pub(crate) fn encode<T>(
        &self,
        queue: &CommandQueue,
        encode: impl FnOnce(&CommandBufferRef) -> Result<T>,
    ) -> Result<T> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| Error::backend("Metal batch command buffer lock poisoned"))?;
        let command_buffer = match guard.open.as_ref() {
            Some(command_buffer) => command_buffer,
            None => {
                trace!(target: "inferno::metal", "opening new batched command buffer");
                guard.open.insert(queue.new_command_buffer().to_owned())
            }
        };
        encode(command_buffer)
    }

    /// Commits the open command buffer without waiting for the GPU.
    ///
    /// Submitted buffers remain retained until `flush` validates them. A new
    /// command buffer can therefore be opened immediately while queue order
    /// preserves dependencies between the two submissions.
    pub(crate) fn submit(&self) -> Result<()> {
        self.submit_with_profile_label(None)
    }

    /// Ends the current profiling segment without waiting for the GPU.
    ///
    /// Callers invoke this only when the Metal profile trace is enabled, so
    /// normal inference keeps its production command-buffer boundaries.
    pub(crate) fn submit_profile_segment(&self, label: &str) -> Result<()> {
        self.submit_with_profile_label(Some(label.to_owned()))
    }

    fn submit_with_profile_label(&self, profile_label: Option<String>) -> Result<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| Error::backend("Metal batch command buffer lock poisoned"))?;
        let Some(command_buffer) = guard.open.take() else {
            return Ok(());
        };

        trace!(target: "inferno::metal", "submitting batched command buffer");
        command_buffer.commit();
        guard.submitted.push(SubmittedBatch {
            command_buffer,
            profile_label,
        });
        Ok(())
    }

    /// Commits the open command buffer, if any, and blocks until every batch
    /// submitted since the previous flush has completed.
    pub(crate) fn flush(&self) -> Result<()> {
        let command_buffers = {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| Error::backend("Metal batch command buffer lock poisoned"))?;
            if let Some(command_buffer) = guard.open.take() {
                trace!(target: "inferno::metal", "submitting final batched command buffer");
                command_buffer.commit();
                guard.submitted.push(SubmittedBatch {
                    command_buffer,
                    profile_label: None,
                });
            }
            std::mem::take(&mut guard.submitted)
        };
        let Some(last) = command_buffers.last() else {
            return Ok(());
        };

        trace!(
            target: "inferno::metal",
            submission_count = command_buffers.len(),
            "waiting for submitted Metal batches"
        );
        last.command_buffer.wait_until_completed();
        for (index, submitted) in command_buffers.iter().enumerate() {
            let command_buffer = &submitted.command_buffer;
            if command_buffer.status() != MTLCommandBufferStatus::Completed {
                return Err(Error::backend(format!(
                    "Metal batched command buffer {index} did not complete: {:?}",
                    command_buffer.status()
                )));
            }
            if tracing::enabled!(
                target: "inferno::metal::batch",
                tracing::Level::TRACE
            ) {
                trace!(
                    target: "inferno::metal::batch",
                    submission_index = index,
                    gpu_duration_ms = command_buffer_gpu_milliseconds(command_buffer),
                    "completed Metal batch"
                );
            }
            if let Some(label) = &submitted.profile_label {
                trace!(
                    target: "inferno::metal::profile",
                    stage = label,
                    gpu_duration_ms = command_buffer_gpu_milliseconds(command_buffer),
                    "completed Metal profile segment"
                );
            }
        }
        Ok(())
    }
}

#[allow(unexpected_cfgs)]
fn command_buffer_gpu_milliseconds(command_buffer: &CommandBufferRef) -> f64 {
    // SAFETY: GPUStartTime and GPUEndTime are read-only MTLCommandBuffer
    // properties, and `flush` calls this only after completion.
    let start: f64 = unsafe { msg_send![command_buffer, GPUStartTime] };
    let end: f64 = unsafe { msg_send![command_buffer, GPUEndTime] };
    if !start.is_finite() || !end.is_finite() || end <= start {
        return 0.0;
    }
    (end - start) * 1_000.0
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use crate::metal::Metal;

    /// A chain of kernels encoded into one batch (no sync between them) must
    /// produce the same values as running the same ops eagerly, one command
    /// buffer + wait each.
    #[test]
    fn batched_chain_matches_eager_ops() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let rows = 2;
        let hidden_size = 4;
        let input = vec![
            0.25_f32, -0.50, 0.75, 1.00, //
            -1.25, 0.50, 0.10, 2.00,
        ];
        let weight = vec![1.0_f32, 1.25, 0.75, 1.50];
        let eps = 1e-5;

        // Eager reference: rms_norm, then add the normed tensor to itself.
        let eager_norm = metal
            .rms_norm_f32(&input, &weight, rows, hidden_size, eps)
            .unwrap();
        let eager_sum = metal.add_f32(&eager_norm, &eager_norm).unwrap();

        // Batched: upload once, chain both kernels through device buffers,
        // and only synchronize at the final read.
        let input_buffer = metal.batch_upload_f32(&input).unwrap();
        let norm_buffer = metal
            .batched_rms_norm(&input_buffer, input.len(), &weight, rows, hidden_size, eps)
            .unwrap();
        let sum_buffer = metal
            .batched_add(&norm_buffer, input.len(), &norm_buffer, input.len())
            .unwrap();
        let batched_sum = metal.batch_read_f32(&sum_buffer, input.len()).unwrap();

        assert_eq!(batched_sum.len(), eager_sum.len());
        for (index, (batched, eager)) in batched_sum.iter().zip(&eager_sum).enumerate() {
            let delta = (batched - eager).abs();
            assert!(
                delta <= 1e-6,
                "value {index} differs: batched={batched}, eager={eager}"
            );
        }
    }

    /// Reading an already-flushed buffer again (nothing pending) must not
    /// fail, and interleaving a new batch afterwards must work.
    #[test]
    fn flush_is_idempotent_and_batches_can_reopen() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let values = vec![1.0_f32, 2.0, 3.0, 4.0];
        let buffer = metal.batch_upload_f32(&values).unwrap();

        metal.batch_flush().unwrap();
        metal.batch_flush().unwrap();

        let doubled = metal
            .batched_add(&buffer, values.len(), &buffer, values.len())
            .unwrap();
        let read = metal.batch_read_f32(&doubled, values.len()).unwrap();
        assert_eq!(read, vec![2.0, 4.0, 6.0, 8.0]);
    }

    /// Queue ordering must preserve data dependencies across an asynchronous
    /// submission boundary without requiring an intermediate CPU wait.
    #[test]
    fn submitted_batch_feeds_next_batch_without_host_wait() {
        let Some(metal) = native_metal_or_skip() else {
            return;
        };
        let values = vec![1.0_f32, 2.0, 3.0, 4.0];
        let input = metal.batch_upload_f32(&values).unwrap();
        let first = metal
            .batched_add(&input, values.len(), &input, values.len())
            .unwrap();

        metal.batch_submit().unwrap();

        let second = metal
            .batched_add(&first, values.len(), &input, values.len())
            .unwrap();
        let output = metal.batch_read_f32(&second, values.len()).unwrap();
        assert_eq!(output, vec![3.0, 6.0, 9.0, 12.0]);
    }

    fn native_metal_or_skip() -> Option<Metal> {
        Metal::new().ok()
    }
}
