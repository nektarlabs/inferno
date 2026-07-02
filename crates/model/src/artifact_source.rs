pub const ANTIREZ_Q2_GGUF_REPO_ID: &str = "antirez/glm-5.2-gguf";
pub const ANTIREZ_Q2_GGUF_REPO_URL: &str = "https://huggingface.co/antirez/glm-5.2-gguf";
pub const ANTIREZ_Q2_GGUF_FILE: &str = "GLM-5.2-UD-Q2_K_RoutedQ2K.gguf";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactFormat {
    Gguf,
}

impl ArtifactFormat {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gguf => "gguf",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Artifact {
    pub repo_id: &'static str,
    pub repo_url: &'static str,
    pub file_name: &'static str,
    pub format: ArtifactFormat,
}

impl Artifact {
    pub const fn hf_url(self) -> &'static str {
        "https://huggingface.co/antirez/glm-5.2-gguf/blob/main/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf"
    }
}

pub const fn antirez_q2_artifact() -> Artifact {
    Artifact {
        repo_id: ANTIREZ_Q2_GGUF_REPO_ID,
        repo_url: ANTIREZ_Q2_GGUF_REPO_URL,
        file_name: ANTIREZ_Q2_GGUF_FILE,
        format: ArtifactFormat::Gguf,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn antirez_q2_artifact_maps_to_expected_gguf_file() {
        let q2 = antirez_q2_artifact();

        assert_eq!(q2.repo_id, "antirez/glm-5.2-gguf");
        assert_eq!(q2.file_name, "GLM-5.2-UD-Q2_K_RoutedQ2K.gguf");
        assert_eq!(q2.format.as_str(), "gguf");
        assert!(q2.hf_url().contains(q2.file_name));
    }
}
