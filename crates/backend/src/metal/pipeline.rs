use ::metal::{ComputePipelineState, Device};
use common::{Error, Result};

use super::library::MetalLibrary;

pub(crate) fn compute_pipeline(
    device: &Device,
    library: &MetalLibrary,
    kernel_name: &str,
) -> Result<ComputePipelineState> {
    let function = library.function(kernel_name)?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|message| {
            Error::backend(format!(
                "failed to create Metal pipeline for {kernel_name}: {message}"
            ))
        })
}
