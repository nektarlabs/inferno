use common::{Error, Result};
use config::Config;
use gguf::GgufFile;

const EXPERT_USED_COUNT_KEY: &str = "glm-dsa.expert_used_count";

pub const ROUTED_EXPERTS_PER_TOKEN: usize = 8;

pub fn validate_routing_policy(config: &Config, gguf: &GgufFile) -> Result<()> {
    let gguf_experts_per_token =
        gguf.metadata_unsigned(EXPERT_USED_COUNT_KEY)
            .ok_or_else(|| {
                Error::weights(format!(
                    "GGUF metadata is missing required key {EXPERT_USED_COUNT_KEY}"
                ))
            })?;
    let gguf_experts_per_token = usize::try_from(gguf_experts_per_token)
        .map_err(|_| Error::weights("GGUF expert-used count does not fit usize"))?;

    validate_expert_policy(config.experts_per_token, gguf_experts_per_token)
}

fn validate_expert_policy(
    config_experts_per_token: usize,
    gguf_experts_per_token: usize,
) -> Result<()> {
    if config_experts_per_token != ROUTED_EXPERTS_PER_TOKEN {
        return Err(Error::weights(format!(
            "GLM config declares {config_experts_per_token} routed experts per token; expected {ROUTED_EXPERTS_PER_TOKEN}"
        )));
    }
    if gguf_experts_per_token != ROUTED_EXPERTS_PER_TOKEN {
        return Err(Error::weights(format!(
            "GGUF declares {gguf_experts_per_token} routed experts per token; expected {ROUTED_EXPERTS_PER_TOKEN}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top8_policy_accepts_matching_artifact_contract() {
        validate_expert_policy(8, 8).unwrap();
        assert_eq!(ROUTED_EXPERTS_PER_TOKEN, 8);
    }

    #[test]
    fn top8_policy_rejects_config_contract_mismatch() {
        let error = validate_expert_policy(7, 8).unwrap_err();
        assert!(error.to_string().contains("config declares 7"));
    }

    #[test]
    fn top8_policy_rejects_gguf_contract_mismatch() {
        let error = validate_expert_policy(8, 7).unwrap_err();
        assert!(error.to_string().contains("GGUF declares 7"));
    }
}
