use std::path::{Path, PathBuf};

use backend::MetalBackend;
use common::{Error, Result};
use config::load_qwen_config;
use runtime::QwenRuntime;
use tokenizer::Tokenizer;

pub const DEFAULT_CONTEXT_TOKENS: usize = 262_144;

#[derive(Debug, Clone, clap::Args)]
pub struct QwenOptions {
    /// Use the external Qwen DFlash2 drafter instead of default MTP.
    #[arg(long)]
    pub dflash_model: Option<PathBuf>,

    /// Qwen conversation capacity, including prompt and generated tokens.
    #[arg(long, default_value_t = DEFAULT_CONTEXT_TOKENS)]
    pub context_tokens: usize,
}

impl Default for QwenOptions {
    fn default() -> Self {
        Self {
            dflash_model: None,
            context_tokens: DEFAULT_CONTEXT_TOKENS,
        }
    }
}

impl QwenOptions {
    pub(super) fn reject_for_other_models(&self) -> Result<()> {
        if self.dflash_model.is_some() || self.context_tokens != DEFAULT_CONTEXT_TOKENS {
            return Err(Error::runtime(
                "--dflash-model and --context-tokens apply only to Qwen",
            ));
        }
        Ok(())
    }

    pub(super) fn open<'a>(
        &self,
        model: &Path,
        config: &Path,
        backend: &'a MetalBackend,
    ) -> Result<QwenRuntime<'a, MetalBackend>> {
        let config_value = load_qwen_config(config)?;
        if self.context_tokens < 2
            || self.context_tokens > config_value.text_config.max_position_embeddings
        {
            return Err(Error::runtime(format!(
                "Qwen --context-tokens must be in 2..={}",
                config_value.text_config.max_position_embeddings
            )));
        }
        match &self.dflash_model {
            Some(draft) => {
                QwenRuntime::open_with_dflash(model, config, draft, backend, 1, self.context_tokens)
            }
            None => QwenRuntime::open(model, config, backend, 1, self.context_tokens),
        }
    }
}

pub(super) fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    let tokenizer = Tokenizer::from_file(path)?;
    tokenizer.validate_contract(
        248_077,
        &[
            ("<|endoftext|>", 248_044),
            ("<|im_start|>", 248_045),
            ("<|im_end|>", 248_046),
        ],
    )?;
    Ok(tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_defaults_to_native_context_and_mtp() {
        let options = QwenOptions::default();
        assert_eq!(options.context_tokens, 262_144);
        assert!(options.dflash_model.is_none());
        assert!(options.reject_for_other_models().is_ok());
    }

    #[test]
    fn qwen_overrides_are_rejected_for_other_models() {
        let options = QwenOptions {
            context_tokens: 4096,
            ..QwenOptions::default()
        };
        assert!(options.reject_for_other_models().is_err());
        let options = QwenOptions {
            dflash_model: Some("draft".into()),
            ..QwenOptions::default()
        };
        assert!(options.reject_for_other_models().is_err());
    }
}
