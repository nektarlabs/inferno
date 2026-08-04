use std::{fs, path::Path};

use common::{Error, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArchitecture {
    GlmMoeDsa,
    Laguna,
}

#[derive(Debug, Deserialize)]
struct ArchitectureProbe {
    model_type: String,
}

pub fn detect_model_architecture(path: &Path) -> Result<ModelArchitecture> {
    let json = fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let probe: ArchitectureProbe = serde_json::from_str(&json).map_err(|source| {
        Error::config(format!(
            "failed to inspect model architecture at {}: {source}",
            path.display()
        ))
    })?;

    match probe.model_type.as_str() {
        "glm_moe_dsa" => Ok(ModelArchitecture::GlmMoeDsa),
        "laguna" => Ok(ModelArchitecture::Laguna),
        other => Err(Error::config(format!(
            "unsupported model_type {other:?} in {}",
            path.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn detects_supported_architectures_without_parsing_their_full_config() {
        let glm = write_config(r#"{"model_type":"glm_moe_dsa"}"#);
        let laguna = write_config(r#"{"model_type":"laguna"}"#);

        assert_eq!(
            detect_model_architecture(&glm).unwrap(),
            ModelArchitecture::GlmMoeDsa
        );
        assert_eq!(
            detect_model_architecture(&laguna).unwrap(),
            ModelArchitecture::Laguna
        );
    }

    #[test]
    fn rejects_unknown_architecture() {
        let path = write_config(r#"{"model_type":"other"}"#);

        let error = detect_model_architecture(&path).unwrap_err();
        assert!(error.to_string().contains("unsupported model_type"));
    }

    fn write_config(json: &str) -> std::path::PathBuf {
        let test_id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "inferno-architecture-{}-{test_id}.json",
            std::process::id()
        ));
        fs::write(&path, json).unwrap();
        path
    }
}
