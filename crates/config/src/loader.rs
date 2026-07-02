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

pub fn load_config(path: Option<&Path>) -> Result<(Config, ConfigSource)> {
    match path {
        Some(path) => load_config_from_path(path)
            .map(|config| (config, ConfigSource::External(path.to_path_buf()))),
        None => {
            load_embedded_config().map(|config| (config, ConfigSource::EmbeddedInfernoDefaults))
        }
    }
}

pub fn load_embedded_config() -> Result<Config> {
    info!(source = "embedded_defaults", "loading GLM config");
    Config::from_json_str(GLM52_LIKE_CONFIG_JSON)
}

pub fn load_config_from_path(path: &Path) -> Result<Config> {
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
    fn load_config_without_path_uses_embedded_source() {
        let (config, source) = load_config(None).unwrap();

        assert_eq!(source, ConfigSource::EmbeddedInfernoDefaults);
        assert_eq!(config.hidden_size, 6144);
    }
}
