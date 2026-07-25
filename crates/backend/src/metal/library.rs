use ::metal::{CompileOptions, Device, Function, Library};
use common::{Error, Result};

const ATTENTION_KERNELS: &str = include_str!("kernels/attention_kernels.metal");
const ACTIVATION_KERNELS: &str = include_str!("kernels/activation_kernels.metal");
const BF16_KERNELS: &str = include_str!("kernels/bf16_kernels.metal");
const CAST_KERNELS: &str = include_str!("kernels/cast_kernels.metal");
const LAYOUT_KERNELS: &str = include_str!("kernels/layout_kernels.metal");
const MATMUL_KERNELS: &str = include_str!("kernels/matmul_kernels.metal");
const NORM_KERNELS: &str = include_str!("kernels/norm_kernels.metal");
const Q2_KERNELS: &str = include_str!("kernels/q2_kernels.metal");
const W4_KERNELS: &str = include_str!("kernels/w4_kernels.metal");
const ROPE_KERNELS: &str = include_str!("kernels/rope_kernels.metal");
const MOE_KERNELS: &str = include_str!("kernels/moe_kernels.metal");
const DSA_KERNELS: &str = include_str!("kernels/dsa_kernels.metal");
const FP8_ATTENTION_KERNELS: &str = include_str!("kernels/fp8_attention_kernels.metal");
const F16_ATTENTION_KERNELS: &str = include_str!("kernels/f16_attention_kernels.metal");
const GGUF_MOE_KERNELS: &str = include_str!("kernels/gguf_moe_kernels.metal");
const LAGUNA_VIEWS_KERNELS: &str = include_str!("kernels/laguna_views_kernels.metal");

pub(crate) struct MetalLibrary {
    library: Library,
}

impl MetalLibrary {
    pub(crate) fn compile(device: &Device) -> Result<Self> {
        let options = CompileOptions::new();
        let source = format!(
            "{ATTENTION_KERNELS}\n{ACTIVATION_KERNELS}\n{BF16_KERNELS}\n{CAST_KERNELS}\n{LAYOUT_KERNELS}\n{MATMUL_KERNELS}\n{NORM_KERNELS}\n{Q2_KERNELS}\n{GGUF_MOE_KERNELS}\n{LAGUNA_VIEWS_KERNELS}\n{W4_KERNELS}\n{ROPE_KERNELS}\n{MOE_KERNELS}\n{DSA_KERNELS}\n{FP8_ATTENTION_KERNELS}\n{F16_ATTENTION_KERNELS}"
        );
        let library = device
            .new_library_with_source(&source, &options)
            .map_err(|message| {
                Error::backend(format!("failed to compile native Metal kernels: {message}"))
            })?;

        Ok(Self { library })
    }

    pub(crate) fn function(&self, name: &str) -> Result<Function> {
        self.library
            .get_function(name, None)
            .map_err(|message| Error::backend(format!("missing Metal kernel {name}: {message}")))
    }
}
