use ::metal::{
    Buffer, CommandQueue, ComputePipelineState, MTLCommandBufferStatus, MTLSize, NSUInteger,
};
use common::{Error, Result};
use tracing::trace;

pub(crate) fn dispatch_1d(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    buffers: &[&Buffer],
    threads: usize,
) -> Result<()> {
    if threads == 0 {
        return Err(Error::backend(
            "Metal dispatch requires at least one thread",
        ));
    }

    let command_buffer = queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);

    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(index as NSUInteger, Some(buffer), 0);
    }

    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1);
    let execution_width = pipeline.thread_execution_width().max(1);
    let threads_per_group = max_threads.min(execution_width.max(1));

    trace!(
        target: "inferno::metal",
        threads,
        threads_per_group,
        "dispatching native Metal kernel"
    );

    encoder.dispatch_threads(
        MTLSize::new(threads as NSUInteger, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    match command_buffer.status() {
        MTLCommandBufferStatus::Completed => Ok(()),
        status => Err(Error::backend(format!(
            "Metal command buffer did not complete: {status:?}"
        ))),
    }
}
