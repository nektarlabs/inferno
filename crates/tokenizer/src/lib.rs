#![deny(unsafe_code)]

//! Tokenizer loading, encoding, decoding, and streaming decode support.

use std::{
    fs,
    path::{Path, PathBuf},
};

use common::{Error, Result};
use tokenizers::Tokenizer as HfTokenizer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerMetadata {
    pub vocab_size: usize,
    pub vocab_size_with_added_tokens: usize,
    pub added_tokens_count: usize,
    pub encode_special_tokens: bool,
    pub special_tokens: Vec<SpecialTokenInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecialTokenInfo {
    pub id: u32,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizedPrompt {
    pub prompt: String,
    pub add_special_tokens: bool,
    pub token_ids: Vec<u32>,
    pub tokens: Vec<String>,
    pub offsets: Vec<(usize, usize)>,
    pub special_tokens_mask: Vec<u32>,
}

impl TokenizedPrompt {
    pub fn token_count(&self) -> usize {
        self.token_ids.len()
    }

    pub fn input_ids_shape(&self, batch: usize) -> Vec<usize> {
        vec![batch, self.token_count()]
    }
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
struct TokenizeReport {
    pub tokenizer_path: PathBuf,
    pub metadata: TokenizerMetadata,
    pub encoded: TokenizedPrompt,
    pub decoded_text: String,
    pub skip_special_tokens_on_decode: bool,
    pub round_trip_exact: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatPrompt {
    pub rendered: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTurn {
    pub user: String,
    pub assistant: String,
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    tokenizer: HfTokenizer,
    source_path: PathBuf,
}

impl Tokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        fs::metadata(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;

        let tokenizer = HfTokenizer::from_file(path).map_err(|error| {
            Error::tokenizer(format!(
                "failed to load tokenizer.json at {}: {error}",
                path.display()
            ))
        })?;

        Ok(Self {
            tokenizer,
            source_path: path.to_path_buf(),
        })
    }

    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub fn metadata(&self) -> TokenizerMetadata {
        let added_tokens = self.tokenizer.get_added_tokens_decoder();
        let mut special_tokens = added_tokens
            .iter()
            .filter_map(|(id, token)| {
                token.special.then_some(SpecialTokenInfo {
                    id: *id,
                    content: token.content.clone(),
                })
            })
            .collect::<Vec<_>>();
        special_tokens.sort_by_key(|token| token.id);

        TokenizerMetadata {
            vocab_size: self.tokenizer.get_vocab_size(false),
            vocab_size_with_added_tokens: self.tokenizer.get_vocab_size(true),
            added_tokens_count: added_tokens.len(),
            encode_special_tokens: self.tokenizer.get_encode_special_tokens(),
            special_tokens,
        }
    }

    pub fn encode(&self, prompt: &str, add_special_tokens: bool) -> Result<TokenizedPrompt> {
        let encoding = self
            .tokenizer
            .encode(prompt, add_special_tokens)
            .map_err(|error| Error::tokenizer(format!("failed to encode prompt: {error}")))?;

        Ok(TokenizedPrompt {
            prompt: prompt.to_string(),
            add_special_tokens,
            token_ids: encoding.get_ids().to_vec(),
            tokens: encoding.get_tokens().to_vec(),
            offsets: encoding.get_offsets().to_vec(),
            special_tokens_mask: encoding.get_special_tokens_mask().to_vec(),
        })
    }

    pub fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(token_ids, skip_special_tokens)
            .map_err(|error| Error::tokenizer(format!("failed to decode token ids: {error}")))
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }
}

#[cfg(test)]
fn encode_decode_round_trip(
    tokenizer_path: impl AsRef<Path>,
    prompt: &str,
    add_special_tokens: bool,
    skip_special_tokens_on_decode: bool,
) -> Result<TokenizeReport> {
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;
    let encoded = tokenizer.encode(prompt, add_special_tokens)?;
    let decoded_text = tokenizer.decode(&encoded.token_ids, skip_special_tokens_on_decode)?;
    let round_trip_exact = decoded_text == prompt;

    Ok(TokenizeReport {
        tokenizer_path: tokenizer.source_path().to_path_buf(),
        metadata: tokenizer.metadata(),
        encoded,
        decoded_text,
        skip_special_tokens_on_decode,
        round_trip_exact,
    })
}

pub fn render_user_prompt(prompt: &str) -> ChatPrompt {
    render_chat_prompt(&[], prompt)
}

pub fn render_chat_prompt(history: &[ChatTurn], prompt: &str) -> ChatPrompt {
    let history_bytes = history.iter().fold(0_usize, |total, turn| {
        total
            .saturating_add(turn.user.len())
            .saturating_add(turn.assistant.len())
    });
    let mut rendered = String::with_capacity(
        "[gMASK]<sop><|system|>Reasoning Effort: Max".len()
            + history_bytes
            + prompt.len()
            + (history.len() + 1) * 48,
    );
    rendered.push_str("[gMASK]<sop><|system|>Reasoning Effort: Max");
    for turn in history {
        rendered.push_str("<|user|>");
        rendered.push_str(&turn.user);
        rendered.push_str("<|assistant|><think></think>");
        rendered.push_str(turn.assistant.trim());
    }
    rendered.push_str("<|user|>");
    rendered.push_str(prompt);
    rendered.push_str("<|assistant|><think>");
    ChatPrompt { rendered }
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use tokenizers::{
        models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace, AddedToken,
        Tokenizer as HfTokenizer,
    };

    use super::*;

    static NEXT_TEST_ID: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn loads_tokenizer_and_round_trips_prompt() {
        let tokenizer_path = write_tiny_tokenizer();
        let report = encode_decode_round_trip(&tokenizer_path, "Hello GLM", false, true).unwrap();

        assert_eq!(report.encoded.token_ids, vec![1, 2]);
        assert_eq!(report.encoded.tokens, vec!["Hello", "GLM"]);
        assert_eq!(report.encoded.input_ids_shape(1), vec![1, 2]);
        assert_eq!(report.decoded_text, "Hello GLM");
        assert!(report.round_trip_exact);
        assert_eq!(report.metadata.vocab_size, 5);
    }

    #[test]
    fn reports_special_tokens_from_tokenizer_json() {
        let tokenizer_path = write_tiny_tokenizer();
        let tokenizer = Tokenizer::from_file(&tokenizer_path).unwrap();
        let metadata = tokenizer.metadata();

        assert!(metadata.added_tokens_count >= 2);
        assert!(metadata
            .special_tokens
            .iter()
            .any(|token| token.content == "<s>"));
        assert!(metadata
            .special_tokens
            .iter()
            .any(|token| token.content == "</s>"));
    }

    #[test]
    fn missing_tokenizer_path_is_typed_io_error() {
        let missing_path = std::env::temp_dir().join("tokenizer-test-missing-tokenizer.json");
        let err = Tokenizer::from_file(&missing_path).expect_err("missing file should fail");

        assert!(err.to_string().contains("io error"));
        assert!(err.to_string().contains("missing-tokenizer.json"));
    }

    #[test]
    fn unknown_token_uses_unk_id() {
        let tokenizer_path = write_tiny_tokenizer();
        let tokenizer = Tokenizer::from_file(&tokenizer_path).unwrap();
        let encoded = tokenizer.encode("not_in_vocab", false).unwrap();

        assert_eq!(encoded.token_ids, vec![0]);
        assert_eq!(tokenizer.decode(&encoded.token_ids, true).unwrap(), "<unk>");
    }

    #[test]
    fn renders_single_user_prompt_from_model_template_contract() {
        let rendered = render_user_prompt("Hello GLM");

        assert_eq!(
            rendered.rendered,
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>Hello GLM<|assistant|><think>"
        );
    }

    #[test]
    fn renders_chat_history_without_replaying_previous_reasoning() {
        let history = vec![ChatTurn {
            user: "What is the capital of Italy?".to_string(),
            assistant: "  Rome.  ".to_string(),
        }];

        let rendered = render_chat_prompt(&history, "And of France?");

        assert_eq!(
            rendered.rendered,
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>What is the capital of Italy?<|assistant|><think></think>Rome.<|user|>And of France?<|assistant|><think>"
        );
    }

    fn write_tiny_tokenizer() -> PathBuf {
        let test_id = NEXT_TEST_ID.fetch_add(1, Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("tokenizer-test-{}-{test_id}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();

        let vocab_path = dir.join("vocab.json");
        fs::write(
            &vocab_path,
            r#"{"<unk>":0,"Hello":1,"GLM":2,"<s>":3,"</s>":4}"#,
        )
        .unwrap();

        let wordlevel =
            WordLevel::from_file(vocab_path.to_str().unwrap(), "<unk>".to_string()).unwrap();
        let mut tokenizer = HfTokenizer::new(wordlevel);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer
            .add_special_tokens([
                AddedToken::from("<s>", true),
                AddedToken::from("</s>", true),
            ])
            .unwrap();

        let tokenizer_path = dir.join("tokenizer.json");
        tokenizer.save(&tokenizer_path, true).unwrap();
        tokenizer_path
    }
}
