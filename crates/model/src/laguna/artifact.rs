use std::path::{Path, PathBuf};

use backend::Backend;
use common::{Error, Result};
use config::LagunaConfig;
use tracing::info;

use super::{
    LagunaExpertCacheMetrics, LagunaGgufModel, LagunaGgufSession, LagunaSafetensorsModel,
    LagunaSafetensorsSession, LagunaTokenOutput, LagunaWeightSummary, LAGUNA_GGUF_FILE_NAME,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LagunaArtifactKind {
    SafetensorsInt4,
    AntirezGguf,
}

impl LagunaArtifactKind {
    pub fn uses_expert_cache(self) -> bool {
        matches!(self, Self::SafetensorsInt4)
    }
}

/// Loaded Laguna model, independent of the supported weight container.
///
/// The enum keeps artifact-specific storage and kernels behind one runtime
/// contract. GLM remains completely outside this path.
#[derive(Debug)]
pub enum LagunaModel {
    Safetensors(LagunaSafetensorsModel),
    Gguf(LagunaGgufModel),
}

#[derive(Debug)]
pub enum LagunaSession {
    Safetensors(Box<LagunaSafetensorsSession>),
    Gguf(LagunaGgufSession),
}

impl LagunaModel {
    pub fn open<B: Backend>(
        model_dir: impl AsRef<Path>,
        config: LagunaConfig,
        backend: &B,
    ) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let artifact = detect_artifact(model_dir)?;
        info!(
            model_dir = %model_dir.display(),
            ?artifact,
            "loading Laguna model artifact"
        );
        match artifact {
            LagunaArtifactKind::SafetensorsInt4 => {
                LagunaSafetensorsModel::open(model_dir, config, backend).map(Self::Safetensors)
            }
            LagunaArtifactKind::AntirezGguf => {
                LagunaGgufModel::open(model_dir.join(LAGUNA_GGUF_FILE_NAME), config, backend)
                    .map(Self::Gguf)
            }
        }
    }

    pub fn artifact_kind(&self) -> LagunaArtifactKind {
        match self {
            Self::Safetensors(_) => LagunaArtifactKind::SafetensorsInt4,
            Self::Gguf(_) => LagunaArtifactKind::AntirezGguf,
        }
    }

    pub fn config(&self) -> &LagunaConfig {
        match self {
            Self::Safetensors(model) => model.config(),
            Self::Gguf(model) => model.config(),
        }
    }

    pub fn weight_summary(&self) -> Option<LagunaWeightSummary> {
        match self {
            Self::Safetensors(model) => Some(model.weight_summary()),
            Self::Gguf(_) => None,
        }
    }

    pub fn prepared_matrix_bytes(&self) -> Option<u64> {
        match self {
            Self::Safetensors(model) => Some(model.prepared_matrix_bytes()),
            Self::Gguf(_) => None,
        }
    }

    pub fn new_session<B: Backend>(
        &self,
        batch: usize,
        context_capacity: usize,
        expert_cache_capacity: Option<usize>,
        backend: &B,
    ) -> Result<LagunaSession> {
        match self {
            Self::Safetensors(model) => {
                let capacity = expert_cache_capacity.ok_or_else(|| {
                    Error::cache("Laguna Safetensors inference requires an expert-cache capacity")
                })?;
                model
                    .new_session(batch, context_capacity, capacity, backend)
                    .map(Box::new)
                    .map(LagunaSession::Safetensors)
            }
            Self::Gguf(model) => {
                if expert_cache_capacity.is_some() {
                    return Err(Error::cache(
                        "Antirez Laguna GGUF uses mmap-backed Q2/Q3 experts and has no configurable expert cache",
                    ));
                }
                model
                    .new_session(batch, context_capacity, backend)
                    .map(LagunaSession::Gguf)
            }
        }
    }

    pub fn prepare_session<B: Backend>(
        &self,
        session: &mut LagunaSession,
        batch: usize,
        context_capacity: usize,
        backend: &B,
    ) -> Result<()> {
        match (self, session) {
            (Self::Safetensors(model), LagunaSession::Safetensors(session)) => {
                model.prepare_session(session, batch, context_capacity, backend)
            }
            (Self::Gguf(model), LagunaSession::Gguf(session)) => {
                model.prepare_session(session, batch, context_capacity, backend)
            }
            _ => Err(Error::runtime(
                "Laguna model and session artifacts do not match",
            )),
        }
    }

    pub fn grow_session_capacity<B: Backend>(
        &self,
        session: &mut LagunaSession,
        context_capacity: usize,
        backend: &B,
    ) -> Result<()> {
        match (self, session) {
            (Self::Safetensors(model), LagunaSession::Safetensors(session)) => {
                model.grow_session_capacity(session, context_capacity, backend)
            }
            (Self::Gguf(model), LagunaSession::Gguf(session)) => {
                model.grow_session_capacity(session, context_capacity, backend)
            }
            _ => Err(Error::runtime(
                "Laguna model and session artifacts do not match",
            )),
        }
    }

    pub fn forward_next_token<B: Backend>(
        &self,
        session: &mut LagunaSession,
        token_ids: &[u32],
        backend: &B,
    ) -> Result<LagunaTokenOutput> {
        match (self, session) {
            (Self::Safetensors(model), LagunaSession::Safetensors(session)) => {
                model.forward_next_token(session, token_ids, backend)
            }
            (Self::Gguf(model), LagunaSession::Gguf(session)) => {
                model.forward_next_token(session, token_ids, backend)
            }
            _ => Err(Error::runtime(
                "Laguna model and session artifacts do not match",
            )),
        }
    }

    pub fn prefill_chunk<B: Backend>(
        &self,
        session: &mut LagunaSession,
        token_ids: &[u32],
        backend: &B,
    ) -> Result<()> {
        match (self, session) {
            (Self::Safetensors(model), LagunaSession::Safetensors(session)) => {
                model.prefill_chunk(session, token_ids, backend)
            }
            (Self::Gguf(model), LagunaSession::Gguf(session)) => {
                model.prefill_chunk(session, token_ids, backend)
            }
            _ => Err(Error::runtime(
                "Laguna model and session artifacts do not match",
            )),
        }
    }
}

impl LagunaSession {
    pub fn position(&self) -> Result<usize> {
        match self {
            Self::Safetensors(session) => session.position(),
            Self::Gguf(session) => session.position(),
        }
    }

    pub fn context_capacity(&self) -> usize {
        match self {
            Self::Safetensors(session) => session.context_capacity(),
            Self::Gguf(session) => session.context_capacity(),
        }
    }

    pub fn expert_cache_metrics(&self) -> LagunaExpertCacheMetrics {
        match self {
            Self::Safetensors(session) => session.expert_cache_metrics(),
            Self::Gguf(_) => LagunaExpertCacheMetrics::default(),
        }
    }

    pub fn resize_expert_cache_capacity(&mut self, capacity_experts: usize) -> Result<()> {
        match self {
            Self::Safetensors(session) => session.resize_expert_cache_capacity(capacity_experts),
            Self::Gguf(_) => Err(Error::cache(
                "Antirez Laguna GGUF has no configurable expert cache",
            )),
        }
    }
}

fn detect_artifact(model_dir: &Path) -> Result<LagunaArtifactKind> {
    let has_gguf = model_dir.join(LAGUNA_GGUF_FILE_NAME).is_file();
    let has_safetensors = contains_safetensors(model_dir)?;
    artifact_from_presence(has_gguf, has_safetensors).map_err(|error| {
        Error::weights(format!(
            "Laguna model directory {}: {error}",
            model_dir.display()
        ))
    })
}

fn artifact_from_presence(
    has_gguf: bool,
    has_safetensors: bool,
) -> std::result::Result<LagunaArtifactKind, &'static str> {
    match (has_gguf, has_safetensors) {
        (true, false) => Ok(LagunaArtifactKind::AntirezGguf),
        (false, true) => Ok(LagunaArtifactKind::SafetensorsInt4),
        (true, true) => Err(
            "contains both the Antirez GGUF and Safetensors shards; keep one weight artifact per directory",
        ),
        (false, false) => {
            Err("contains neither the exact Antirez GGUF nor Safetensors weight shards")
        }
    }
}

fn contains_safetensors(model_dir: &Path) -> Result<bool> {
    let entries = std::fs::read_dir(model_dir).map_err(|source| Error::Io {
        path: PathBuf::from(model_dir),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| Error::Io {
            path: PathBuf::from(model_dir),
            source,
        })?;
        if entry
            .path()
            .extension()
            .is_some_and(|extension| extension == "safetensors")
        {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::{artifact_from_presence, LagunaArtifactKind};

    #[test]
    fn only_safetensors_artifacts_use_the_expert_cache() {
        assert!(LagunaArtifactKind::SafetensorsInt4.uses_expert_cache());
        assert!(!LagunaArtifactKind::AntirezGguf.uses_expert_cache());
    }

    #[test]
    fn artifact_selection_is_exact_and_unambiguous() {
        assert_eq!(
            artifact_from_presence(true, false).unwrap(),
            LagunaArtifactKind::AntirezGguf
        );
        assert_eq!(
            artifact_from_presence(false, true).unwrap(),
            LagunaArtifactKind::SafetensorsInt4
        );
        assert!(artifact_from_presence(true, true)
            .unwrap_err()
            .contains("both"));
        assert!(artifact_from_presence(false, false)
            .unwrap_err()
            .contains("neither"));
    }
}
