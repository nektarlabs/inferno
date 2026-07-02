use std::{io, path::PathBuf};

use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("backend error: {message}")]
    Backend { message: String },

    #[error("shape error in {context}: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        context: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("configuration error: {message}")]
    Config { message: String },

    #[error("tokenizer error: {message}")]
    Tokenizer { message: String },

    #[error("weights error: {message}")]
    Weights { message: String },

    #[error("gguf error: {message}")]
    Gguf { message: String },

    #[error("model error: {message}")]
    Model { message: String },

    #[error("cache error: {message}")]
    Cache { message: String },

    #[error("moe error: {message}")]
    Moe { message: String },

    #[error("runtime error: {message}")]
    Runtime { message: String },

    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

impl Error {
    pub fn backend(message: impl Into<String>) -> Self {
        Self::Backend {
            message: message.into(),
        }
    }

    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    pub fn tokenizer(message: impl Into<String>) -> Self {
        Self::Tokenizer {
            message: message.into(),
        }
    }

    pub fn model(message: impl Into<String>) -> Self {
        Self::Model {
            message: message.into(),
        }
    }

    pub fn weights(message: impl Into<String>) -> Self {
        Self::Weights {
            message: message.into(),
        }
    }

    pub fn gguf(message: impl Into<String>) -> Self {
        Self::Gguf {
            message: message.into(),
        }
    }

    pub fn moe(message: impl Into<String>) -> Self {
        Self::Moe {
            message: message.into(),
        }
    }

    pub fn cache(message: impl Into<String>) -> Self {
        Self::Cache {
            message: message.into(),
        }
    }

    pub fn runtime(message: impl Into<String>) -> Self {
        Self::Runtime {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_error_includes_context() {
        let error = Error::ShapeMismatch {
            context: "q_heads".to_string(),
            expected: vec![1, 4, 64, 256],
            actual: vec![1, 4, 64, 128],
        };

        let rendered = error.to_string();
        assert!(rendered.contains("q_heads"));
        assert!(rendered.contains("256"));
        assert!(rendered.contains("128"));
    }
}
