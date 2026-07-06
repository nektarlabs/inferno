use ::metal::{
    Buffer, CommandBufferRef, CommandQueue, ComputePipelineState, MTLCommandBufferStatus, MTLSize,
    NSUInteger,
};
use common::{Error, Result};
use tracing::trace;

#[derive(Clone, Copy)]
pub(crate) struct Dispatch1d<'a> {
    pub(crate) pipeline: &'a ComputePipelineState,
    pub(crate) buffers: &'a [&'a Buffer],
    pub(crate) threads: usize,
}

pub(crate) fn dispatch_1d(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    buffers: &[&Buffer],
    threads: usize,
) -> Result<()> {
    dispatch_1d_many(
        queue,
        &[Dispatch1d {
            pipeline,
            buffers,
            threads,
        }],
    )
}

pub(crate) fn dispatch_1d_many(queue: &CommandQueue, dispatches: &[Dispatch1d<'_>]) -> Result<()> {
    if dispatches.is_empty() {
        return Err(Error::backend(
            "Metal command buffer requires at least one dispatch",
        ));
    }

    let command_buffer = queue.new_command_buffer();
    for dispatch in dispatches {
        encode_1d(
            command_buffer,
            dispatch.pipeline,
            dispatch.buffers,
            dispatch.threads,
        )?;
    }
    command_buffer.commit();
    command_buffer.wait_until_completed();

    match command_buffer.status() {
        MTLCommandBufferStatus::Completed => Ok(()),
        status => Err(Error::backend(format!(
            "Metal command buffer did not complete: {status:?}"
        ))),
    }
}

/// Encodes a GPU-side buffer-to-buffer copy (blit) into the command buffer.
/// Offsets and length are in f32 elements.
pub(crate) fn encode_f32_copy(
    command_buffer: &CommandBufferRef,
    source: &Buffer,
    source_offset: usize,
    destination: &Buffer,
    destination_offset: usize,
    len: usize,
) -> Result<()> {
    let element = std::mem::size_of::<f32>();
    let byte_len = len
        .checked_mul(element)
        .ok_or_else(|| Error::backend("Metal blit copy byte length overflow"))?;
    let source_bytes = source_offset
        .checked_mul(element)
        .ok_or_else(|| Error::backend("Metal blit copy source offset overflow"))?;
    let destination_bytes = destination_offset
        .checked_mul(element)
        .ok_or_else(|| Error::backend("Metal blit copy destination offset overflow"))?;
    let source_end = source_bytes
        .checked_add(byte_len)
        .ok_or_else(|| Error::backend("Metal blit copy source range overflow"))?;
    let destination_end = destination_bytes
        .checked_add(byte_len)
        .ok_or_else(|| Error::backend("Metal blit copy destination range overflow"))?;
    if source.length() < source_end as u64 {
        return Err(Error::backend(format!(
            "Metal blit copy source is too small: need {source_end} bytes, got {}",
            source.length()
        )));
    }
    if destination.length() < destination_end as u64 {
        return Err(Error::backend(format!(
            "Metal blit copy destination is too small: need {destination_end} bytes, got {}",
            destination.length()
        )));
    }

    let encoder = command_buffer.new_blit_command_encoder();
    encoder.copy_from_buffer(
        source,
        source_bytes as NSUInteger,
        destination,
        destination_bytes as NSUInteger,
        byte_len as NSUInteger,
    );
    encoder.end_encoding();
    Ok(())
}

pub(crate) fn encode_1d(
    command_buffer: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&Buffer],
    threads: usize,
) -> Result<()> {
    if threads == 0 {
        return Err(Error::backend(
            "Metal dispatch requires at least one thread",
        ));
    }

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
    Ok(())
}
