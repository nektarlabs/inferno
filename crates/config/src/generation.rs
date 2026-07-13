use std::{collections::HashSet, fs, path::Path};

use common::{Error, Result};
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationConfig {
    pub eos_token_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct RawGenerationConfig {
    eos_token_id: Vec<u32>,
}

pub fn load_generation_config(path: &Path) -> Result<GenerationConfig> {
    let json = fs::read_to_string(path).map_err(|source| Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let raw: RawGenerationConfig = serde_json::from_str(&json).map_err(|source| {
        Error::config(format!(
            "failed to parse generation config at {}: {source}",
            path.display()
        ))
    })?;

    validate_eos_token_ids(&raw.eos_token_id)?;
    Ok(GenerationConfig {
        eos_token_ids: raw.eos_token_id,
    })
}

fn validate_eos_token_ids(token_ids: &[u32]) -> Result<()> {
    if token_ids.is_empty() {
        return Err(Error::config(
            "generation config eos_token_id must not be empty",
        ));
    }

    let mut unique = HashSet::with_capacity(token_ids.len());
    if let Some(duplicate) = token_ids
        .iter()
        .copied()
        .find(|token_id| !unique.insert(*token_id))
    {
        return Err(Error::config(format!(
            "generation config contains duplicate EOS token id {duplicate}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_real_glm_eos_token_shape() {
        let raw: RawGenerationConfig = serde_json::from_str(
            r#"{"eos_token_id":[154820,154827,154829],"pad_token_id":154820}"#,
        )
        .unwrap();

        validate_eos_token_ids(&raw.eos_token_id).unwrap();
        assert_eq!(raw.eos_token_id, vec![154820, 154827, 154829]);
    }

    #[test]
    fn rejects_duplicate_eos_token_ids() {
        let error = validate_eos_token_ids(&[7, 8, 7]).unwrap_err();
        assert!(error.to_string().contains("duplicate EOS token id 7"));
    }
}
