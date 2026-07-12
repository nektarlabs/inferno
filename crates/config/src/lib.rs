#![deny(unsafe_code)]

//! GLM configuration parsing, validation, and architecture summaries.

mod loader;
mod snapshot;
mod types;
mod validation;

pub use loader::{load_config, load_embedded_config, ConfigSource};
pub use types::{Config, IndexerLayerKind};
