#![deny(unsafe_code)]

//! Model configuration parsing, validation, and architecture summaries.

mod architecture;
mod dflash;
mod generation;
mod laguna;
mod loader;
mod qwen;
mod snapshot;
mod types;
mod validation;

pub use architecture::{detect_model_architecture, ModelArchitecture};
pub use dflash::{load_dflash_config, DFlashConfig, DFlashOptions, DFlashRopeParameters};
pub use generation::{load_generation_config, GenerationConfig};
pub use laguna::{
    load_laguna_config, LagunaAttentionKind, LagunaConfig, LagunaKvCacheBudget, LagunaMlpKind,
    LagunaProfile,
};
pub use loader::{load_config, load_embedded_config, ConfigSource};
pub use qwen::{
    load_qwen_config, load_qwen_mtp_config, QwenConfig, QwenLayerKind, QwenMtpConfig,
    QwenMtpTextConfig, QwenQuantizationConfig, QwenRopeParameters, QwenTextConfig,
};
pub use types::{Config, IndexerLayerKind};
