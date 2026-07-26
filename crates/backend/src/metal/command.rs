use ::metal::{
    Buffer, CommandBufferRef, CommandQueue, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceUsage, MTLSize, NSUInteger,
};
use common::{Error, Result};
use tracing::trace;

#[derive(Clone, Copy)]
pub(crate) struct Dispatch1d<'a> {
    pub(crate) pipeline: &'a ComputePipelineState,
    pub(crate) buffers: &'a [&'a Buffer],
    pub(crate) threads: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct Dispatch1dWithOffsets<'a> {
    pub(crate) pipeline: &'a ComputePipelineState,
    pub(crate) buffers: &'a [(&'a Buffer, usize)],
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

pub(crate) fn dispatch_1d_with_offsets(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    buffers: &[(&Buffer, usize)],
    threads: usize,
) -> Result<()> {
    let command_buffer = queue.new_command_buffer();
    encode_1d_with_offsets(command_buffer, pipeline, buffers, threads)?;
    command_buffer.commit();
    command_buffer.wait_until_completed();

    match command_buffer.status() {
        MTLCommandBufferStatus::Completed => Ok(()),
        status => Err(Error::backend(format!(
            "Metal command buffer did not complete: {status:?}"
        ))),
    }
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

pub(crate) fn dispatch_1d_many_with_offsets(
    queue: &CommandQueue,
    dispatches: &[Dispatch1dWithOffsets<'_>],
) -> Result<()> {
    if dispatches.is_empty() {
        return Err(Error::backend(
            "Metal command buffer requires at least one dispatch",
        ));
    }

    let command_buffer = queue.new_command_buffer();
    for dispatch in dispatches {
        encode_1d_with_offsets(
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

pub(crate) fn dispatch_1d_threadgroups(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    buffers: &[&Buffer],
    threadgroup_count: usize,
    threads_per_group: usize,
) -> Result<()> {
    let command_buffer = queue.new_command_buffer();
    encode_1d_threadgroups(
        command_buffer,
        pipeline,
        buffers,
        threadgroup_count,
        threads_per_group,
    )?;
    command_buffer.commit();
    command_buffer.wait_until_completed();

    match command_buffer.status() {
        MTLCommandBufferStatus::Completed => Ok(()),
        status => Err(Error::backend(format!(
            "Metal command buffer did not complete: {status:?}"
        ))),
    }
}

pub(crate) fn dispatch_2d(
    queue: &CommandQueue,
    pipeline: &ComputePipelineState,
    buffers: &[&Buffer],
    width: usize,
    height: usize,
    threads_per_group_width: usize,
    threads_per_group_height: usize,
) -> Result<()> {
    let command_buffer = queue.new_command_buffer();
    encode_2d(
        command_buffer,
        pipeline,
        buffers,
        width,
        height,
        threads_per_group_width,
        threads_per_group_height,
    )?;
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
    encode_element_copy(
        command_buffer,
        source,
        source_offset,
        destination,
        destination_offset,
        len,
        std::mem::size_of::<f32>(),
    )
}

/// Encodes a GPU-side buffer-to-buffer copy for values of `element_size`
/// bytes. Offsets and length are in logical elements.
pub(crate) fn encode_element_copy(
    command_buffer: &CommandBufferRef,
    source: &Buffer,
    source_offset: usize,
    destination: &Buffer,
    destination_offset: usize,
    len: usize,
    element_size: usize,
) -> Result<()> {
    if element_size == 0 {
        return Err(Error::backend(
            "Metal blit copy element size must be positive",
        ));
    }
    let byte_len = len
        .checked_mul(element_size)
        .ok_or_else(|| Error::backend("Metal blit copy byte length overflow"))?;
    let source_bytes = source_offset
        .checked_mul(element_size)
        .ok_or_else(|| Error::backend("Metal blit copy source offset overflow"))?;
    let destination_bytes = destination_offset
        .checked_mul(element_size)
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

/// One argument of a compute kernel.
///
/// Scalars are bound with `setBytes`, which copies the value straight into the
/// command buffer. Passing them as arena buffers instead costs a real
/// `MTLBuffer` allocation plus hazard-tracking registration — measured at about
/// 3.9µs each on an M4 Max. Laguna decode binds roughly fifty scalars per layer
/// across forty-eight layers, so that allocation alone accounted for most of the
/// CPU time spent encoding a token.
#[derive(Clone, Copy)]
pub(crate) enum KernelArg<'a> {
    Buffer(&'a Buffer),
    BufferOffset(&'a Buffer, usize),
    U32(u32),
}

fn bind_args(encoder: &::metal::ComputeCommandEncoderRef, args: &[KernelArg<'_>]) -> Result<()> {
    for (index, arg) in args.iter().enumerate() {
        let slot = index as NSUInteger;
        match arg {
            KernelArg::Buffer(buffer) => encoder.set_buffer(slot, Some(buffer), 0),
            KernelArg::BufferOffset(buffer, offset) => {
                if *offset > buffer.length() as usize {
                    return Err(Error::backend(format!(
                        "Metal buffer {index} offset {offset} exceeds buffer length {}",
                        buffer.length()
                    )));
                }
                encoder.set_buffer(slot, Some(buffer), *offset as NSUInteger);
            }
            // `setBytes` copies before returning, so pointing at the slice
            // element is sound for the duration of this call.
            KernelArg::U32(value) => encoder.set_bytes(
                slot,
                std::mem::size_of::<u32>() as NSUInteger,
                value as *const u32 as *const std::ffi::c_void,
            ),
        }
    }
    Ok(())
}

fn validate_threadgroup_shape(
    pipeline: &ComputePipelineState,
    threadgroup_count: usize,
    threads_per_group: usize,
) -> Result<()> {
    if threadgroup_count == 0 || threads_per_group == 0 {
        return Err(Error::backend(
            "Metal threadgroup dispatch dimensions must be positive",
        ));
    }
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1) as usize;
    if threads_per_group > max_threads {
        return Err(Error::backend(format!(
            "Metal threadgroup requires {threads_per_group} threads but pipeline allows {max_threads}"
        )));
    }
    let execution_width = pipeline.thread_execution_width().max(1) as usize;
    if !threads_per_group.is_multiple_of(execution_width) {
        return Err(Error::backend(format!(
            "Metal threadgroup size {threads_per_group} must be divisible by SIMD width {execution_width}"
        )));
    }
    Ok(())
}

/// Dispatches a fixed number of threadgroups with mixed buffer and scalar
/// arguments.
pub(crate) fn encode_1d_threadgroups_args(
    command_buffer: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    args: &[KernelArg<'_>],
    threadgroup_count: usize,
    threads_per_group: usize,
) -> Result<()> {
    validate_threadgroup_shape(pipeline, threadgroup_count, threads_per_group)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    bind_args(encoder, args)?;
    encoder.dispatch_thread_groups(
        MTLSize::new(threadgroup_count as NSUInteger, 1, 1),
        MTLSize::new(threads_per_group as NSUInteger, 1, 1),
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

    let threads_per_group = preferred_1d_threadgroup_size(pipeline);

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

pub(crate) fn encode_1d_with_offsets(
    command_buffer: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[(&Buffer, usize)],
    threads: usize,
) -> Result<()> {
    if threads == 0 {
        return Err(Error::backend(
            "Metal dispatch requires at least one thread",
        ));
    }
    validate_buffer_offsets(buffers)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (index, (buffer, offset)) in buffers.iter().enumerate() {
        encoder.set_buffer(index as NSUInteger, Some(buffer), *offset as NSUInteger);
    }

    let threads_per_group = preferred_1d_threadgroup_size(pipeline);
    trace!(
        target: "inferno::metal",
        threads,
        threads_per_group,
        "dispatching native Metal kernel with buffer offsets"
    );
    encoder.dispatch_threads(
        MTLSize::new(threads as NSUInteger, 1, 1),
        MTLSize::new(threads_per_group, 1, 1),
    );
    encoder.end_encoding();
    Ok(())
}

fn validate_buffer_offsets(buffers: &[(&Buffer, usize)]) -> Result<()> {
    for (index, (buffer, offset)) in buffers.iter().enumerate() {
        if *offset > buffer.length() as usize {
            return Err(Error::backend(format!(
                "Metal buffer {index} offset {offset} exceeds buffer length {}",
                buffer.length()
            )));
        }
    }
    Ok(())
}

pub(crate) fn encode_1d_threadgroups(
    command_buffer: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&Buffer],
    threadgroup_count: usize,
    threads_per_group: usize,
) -> Result<()> {
    if threadgroup_count == 0 || threads_per_group == 0 {
        return Err(Error::backend(
            "Metal threadgroup dispatch dimensions must be positive",
        ));
    }
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1) as usize;
    if threads_per_group > max_threads {
        return Err(Error::backend(format!(
            "Metal threadgroup requires {threads_per_group} threads but pipeline allows {max_threads}"
        )));
    }
    let execution_width = pipeline.thread_execution_width().max(1) as usize;
    if threads_per_group % execution_width != 0 {
        return Err(Error::backend(format!(
            "Metal threadgroup size {threads_per_group} must be divisible by SIMD width {execution_width}"
        )));
    }

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(index as NSUInteger, Some(buffer), 0);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(threadgroup_count as NSUInteger, 1, 1),
        MTLSize::new(threads_per_group as NSUInteger, 1, 1),
    );
    encoder.end_encoding();
    Ok(())
}

pub(crate) fn encode_1d_threadgroups_with_offsets(
    command_buffer: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[(&Buffer, usize)],
    threadgroup_count: usize,
    threads_per_group: usize,
) -> Result<()> {
    if threadgroup_count == 0 || threads_per_group == 0 {
        return Err(Error::backend(
            "Metal threadgroup dispatch dimensions must be positive",
        ));
    }
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1) as usize;
    if threads_per_group > max_threads {
        return Err(Error::backend(format!(
            "Metal threadgroup requires {threads_per_group} threads but pipeline allows {max_threads}"
        )));
    }
    let execution_width = pipeline.thread_execution_width().max(1) as usize;
    if !threads_per_group.is_multiple_of(execution_width) {
        return Err(Error::backend(format!(
            "Metal threadgroup size {threads_per_group} must be divisible by SIMD width {execution_width}"
        )));
    }
    validate_buffer_offsets(buffers)?;

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (index, (buffer, offset)) in buffers.iter().enumerate() {
        encoder.set_buffer(index as NSUInteger, Some(buffer), *offset as NSUInteger);
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(threadgroup_count as NSUInteger, 1, 1),
        MTLSize::new(threads_per_group as NSUInteger, 1, 1),
    );
    encoder.end_encoding();
    Ok(())
}

/// Encodes a 1D kernel that dereferences GPU addresses stored in another
/// buffer. Metal cannot infer those dependencies, so every indirectly
/// addressed buffer must be declared explicitly with `use_resource`.
pub(crate) fn encode_1d_with_indirect_reads(
    command_buffer: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&Buffer],
    indirect_reads: &[&Buffer],
    threads: usize,
) -> Result<()> {
    if threads == 0 {
        return Err(Error::backend(
            "Metal indirect dispatch requires at least one thread",
        ));
    }
    if indirect_reads.is_empty() {
        return Err(Error::backend(
            "Metal indirect dispatch requires declared read resources",
        ));
    }

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(index as NSUInteger, Some(buffer), 0);
    }
    for resource in indirect_reads {
        encoder.use_resource(resource.as_ref(), MTLResourceUsage::Read);
    }

    let threads_per_group = preferred_1d_threadgroup_size(pipeline);
    encoder.dispatch_threads(
        MTLSize::new(threads as u64, 1, 1),
        MTLSize::new(threads_per_group as u64, 1, 1),
    );
    encoder.end_encoding();
    Ok(())
}

pub(crate) fn encode_2d(
    command_buffer: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    buffers: &[&Buffer],
    width: usize,
    height: usize,
    threads_per_group_width: usize,
    threads_per_group_height: usize,
) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(Error::backend(
            "Metal 2D dispatch requires non-zero width and height",
        ));
    }
    if threads_per_group_width == 0 || threads_per_group_height == 0 {
        return Err(Error::backend(
            "Metal 2D dispatch requires non-zero threadgroup dimensions",
        ));
    }

    let threads_per_group = threads_per_group_width
        .checked_mul(threads_per_group_height)
        .ok_or_else(|| Error::backend("Metal 2D threadgroup size overflow"))?;
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1) as usize;
    if threads_per_group > max_threads {
        return Err(Error::backend(format!(
            "Metal 2D threadgroup has {threads_per_group} threads but pipeline allows {max_threads}"
        )));
    }

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);

    for (index, buffer) in buffers.iter().enumerate() {
        encoder.set_buffer(index as NSUInteger, Some(buffer), 0);
    }

    trace!(
        target: "inferno::metal",
        width,
        height,
        threads_per_group_width,
        threads_per_group_height,
        threadgroup_count_width = ceil_div(width, threads_per_group_width),
        threadgroup_count_height = ceil_div(height, threads_per_group_height),
        "dispatching native Metal 2D kernel"
    );

    encoder.dispatch_thread_groups(
        MTLSize::new(
            ceil_div(width, threads_per_group_width) as NSUInteger,
            ceil_div(height, threads_per_group_height) as NSUInteger,
            1,
        ),
        MTLSize::new(
            threads_per_group_width as NSUInteger,
            threads_per_group_height as NSUInteger,
            1,
        ),
    );
    encoder.end_encoding();
    Ok(())
}

fn preferred_1d_threadgroup_size(pipeline: &ComputePipelineState) -> NSUInteger {
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(1);
    let execution_width = pipeline.thread_execution_width().max(1);
    let target = max_threads.min(256).max(execution_width);
    let aligned = (target / execution_width) * execution_width;
    aligned.max(1).min(max_threads)
}

fn ceil_div(value: usize, divisor: usize) -> usize {
    value.div_ceil(divisor)
}
