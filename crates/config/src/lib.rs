#![deny(unsafe_code)]

//! GLM configuration parsing, validation, and architecture summaries.

mod loader;
mod snapshot;
mod summary;
mod types;
mod validation;

pub use loader::{load_config, load_config_from_path, load_embedded_config, ConfigSource};
pub use snapshot::GLM52_LIKE_CONFIG_JSON;
pub use summary::{ArchitectureSummary, ShapeSummary};
pub use types::{Config, IndexerLayerKind};
