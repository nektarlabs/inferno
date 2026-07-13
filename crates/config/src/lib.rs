#![deny(unsafe_code)]

//! GLM configuration parsing, validation, and architecture summaries.

mod generation;
mod loader;
mod snapshot;
mod types;
mod validation;

pub use generation::{load_generation_config, GenerationConfig};
pub use loader::{load_config, load_embedded_config, ConfigSource};
pub use types::{Config, IndexerLayerKind};
