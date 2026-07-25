#![deny(unsafe_code)]

//! Tokenizer loading, encoding, decoding, and streaming decode support.

mod agent;

use std::{
    fs,
    path::{Path, PathBuf},
};

use common::{Error, Result};
use tokenizers::{
    DecodeStream, DecoderWrapper, ModelWrapper, NormalizerWrapper, PostProcessorWrapper,
    PreTokenizerWrapper, Tokenizer as HfTokenizer,
};

pub use agent::{
    is_supported_codex_function, parse_agent_output, parse_complete_agent_tool_call,
    render_codex_prompt, render_laguna_codex_prompt, streamable_agent_text, AgentFunctionCall,
    AgentOutput, AgentOutputItem,
};

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

pub struct TokenDecoder<'a> {
    stream: DecodeStream<
        'a,
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
}

impl TokenDecoder<'_> {
    pub fn push(&mut self, token_id: u32) -> Result<Option<String>> {
        self.stream.step(token_id).map_err(|error| {
            Error::tokenizer(format!(
                "failed to decode streaming token {token_id}: {error}"
            ))
        })
    }
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

    pub fn decoder(&self, skip_special_tokens: bool) -> TokenDecoder<'_> {
        TokenDecoder {
            stream: self.tokenizer.decode_stream(skip_special_tokens),
        }
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    /// Validates the vocabulary and control-token IDs required by one exact
    /// model artifact before inference starts.
    pub fn validate_contract(
        &self,
        expected_vocab_size: usize,
        required_tokens: &[(&str, u32)],
    ) -> Result<()> {
        let actual_vocab_size = self.tokenizer.get_vocab_size(true);
        if actual_vocab_size != expected_vocab_size {
            return Err(Error::tokenizer(format!(
                "tokenizer vocabulary must contain {expected_vocab_size} entries, got {actual_vocab_size}"
            )));
        }
        for &(token, expected_id) in required_tokens {
            let actual_id = self.tokenizer.token_to_id(token).ok_or_else(|| {
                Error::tokenizer(format!("tokenizer is missing required token {token:?}"))
            })?;
            if actual_id != expected_id {
                return Err(Error::tokenizer(format!(
                    "tokenizer token {token:?} must have ID {expected_id}, got {actual_id}"
                )));
            }
        }
        Ok(())
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

/// Renders one user turn with Laguna S 2.1's published chat template.
///
/// This is intentionally the exact no-tools, thinking-enabled case used by
/// Inferno's generate command. It matches both the template published beside
/// the source GGUF revision and DwarfStar's Laguna runtime contract. The
/// GGUF's embedded template metadata differs from that tested runtime
/// contract, so Inferno does not interpret it dynamically. General Jinja
/// interpretation does not belong in the inference hot path.
pub fn render_laguna_user_prompt(prompt: &str) -> ChatPrompt {
    render_laguna_chat_prompt(&[], prompt)
}

/// Renders a Laguna S 2.1 conversation using the checkpoint's published
/// thinking-enabled chat contract.
pub fn render_laguna_chat_prompt(history: &[ChatTurn], prompt: &str) -> ChatPrompt {
    const DEFAULT_SYSTEM: &str = "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be helpful to users through natural language conversations.";

    let history_bytes = history.iter().fold(0_usize, |total, turn| {
        total
            .saturating_add(turn.user.len())
            .saturating_add(turn.assistant.len())
    });
    let mut rendered = String::with_capacity(
        "〈|EOS|〉<system></system>\n<user></user>\n<assistant><think>".len()
            + DEFAULT_SYSTEM.len()
            + history_bytes
            + prompt.len(),
    );
    rendered.push_str("〈|EOS|〉<system>");
    rendered.push_str(DEFAULT_SYSTEM);
    rendered.push_str("</system>\n");
    for turn in history {
        rendered.push_str("<user>");
        rendered.push_str(&turn.user);
        rendered.push_str("</user>\n<assistant><think></think>");
        rendered.push_str(turn.assistant.trim());
        rendered.push_str("</assistant>\n");
    }
    rendered.push_str("<user>");
    rendered.push_str(prompt);
    rendered.push_str("</user>\n<assistant><think>");
    ChatPrompt { rendered }
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
    fn validates_exact_vocabulary_and_control_token_ids() {
        let tokenizer_path = write_tiny_tokenizer();
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();

        tokenizer
            .validate_contract(5, &[("<unk>", 0), ("Hello", 1)])
            .unwrap();
        assert!(tokenizer.validate_contract(6, &[("<unk>", 0)]).is_err());
        assert!(tokenizer.validate_contract(5, &[("Hello", 2)]).is_err());
        assert!(tokenizer.validate_contract(5, &[("missing", 4)]).is_err());
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
    fn streaming_decoder_matches_complete_decode() {
        let tokenizer_path = write_tiny_tokenizer();
        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        let token_ids = tokenizer.encode("Hello GLM", false).unwrap().token_ids;
        let mut decoder = tokenizer.decoder(true);
        let streamed = token_ids
            .iter()
            .filter_map(|token_id| decoder.push(*token_id).unwrap())
            .collect::<String>();

        assert_eq!(streamed, tokenizer.decode(&token_ids, true).unwrap());
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
    fn renders_laguna_single_user_prompt_from_published_template() {
        let rendered = render_laguna_user_prompt("Hello Laguna");

        assert_eq!(
            rendered.rendered,
            "〈|EOS|〉<system>You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be helpful to users through natural language conversations.</system>\n<user>Hello Laguna</user>\n<assistant><think>"
        );
    }

    #[test]
    fn laguna_prompt_tokens_match_dwarfstar_reference() {
        let tokenizer_path = Path::new("../../models/laguna-s-2.1-int4/tokenizer.json");
        if !tokenizer_path.is_file() {
            return;
        }

        let tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        let rendered = render_laguna_user_prompt("Tell me the capital of Italy.");
        let encoded = tokenizer.encode(&rendered.rendered, false).unwrap();

        // Produced by DwarfStar's --dump-tokens at the Laguna GGUF reference
        // revision. This covers BPE merges across tag/content boundaries too.
        assert_eq!(
            encoded.token_ids,
            vec![
                2, 97, 6453, 55620, 515, 330, 6408, 81, 12123, 1009, 8286, 10167, 18263, 2637, 565,
                30810, 638, 83, 1239, 515, 1973, 367, 445, 6408, 367, 1667, 1388, 5882, 2930,
                22746, 4187, 6453, 99, 268, 97, 1437, 22021, 753, 756, 340, 9626, 377, 22532, 4187,
                1437, 99, 268, 23, 18,
            ]
        );
    }

    #[test]
    fn renders_laguna_history_with_closed_assistant_turns() {
        let history = vec![ChatTurn {
            user: "What is the capital of Italy?".to_string(),
            assistant: "Rome.".to_string(),
        }];
        let rendered = render_laguna_chat_prompt(&history, "And France?");

        assert_eq!(
            rendered.rendered,
            "〈|EOS|〉<system>You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be helpful to users through natural language conversations.</system>\n<user>What is the capital of Italy?</user>\n<assistant><think></think>Rome.</assistant>\n<user>And France?</user>\n<assistant><think>"
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
