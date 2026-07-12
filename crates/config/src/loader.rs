use std::{
    fs,
    path::{Path, PathBuf},
};

use common::{Error, Result};
use tracing::info;

use crate::{snapshot::GLM52_LIKE_CONFIG_JSON, Config};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigSource {
    EmbeddedInfernoDefaults,
    External(PathBuf),
}

pub fn load_embedded_config() -> Result<Config> {
    info!(source = "embedded_defaults", "loading GLM config");
    Config::from_json_str(GLM52_LIKE_CONFIG_JSON)
}

pub fn load_config(path: &Path) -> Result<Config> {
    info!(path = %path.display(), "loading external GLM config");
    let json = fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Config::from_json_str(&json)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_external_config_returns_typed_io_error() {
        let path = Path::new("/missing/inferno/config.json");
        let error = load_config(path).expect_err("missing config must fail");

        assert!(matches!(error, Error::Io { .. }));
        assert!(error.to_string().contains(path.to_string_lossy().as_ref()));
    }
}
