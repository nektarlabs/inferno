#![deny(unsafe_code)]

//! Model configuration parsing, validation, and architecture summaries.

mod architecture;
mod generation;
mod laguna;
mod loader;
mod snapshot;
mod types;
mod validation;

pub use architecture::{detect_model_architecture, ModelArchitecture};
pub use generation::{load_generation_config, GenerationConfig};
pub use laguna::{
    load_laguna_config, LagunaAttentionKind, LagunaConfig, LagunaKvCacheBudget, LagunaMlpKind,
};
pub use loader::{load_config, load_embedded_config, ConfigSource};
pub use types::{Config, IndexerLayerKind};
