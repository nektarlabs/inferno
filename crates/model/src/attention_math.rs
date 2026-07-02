use common::{validate_exact_shape, Error, Result};
use config::Config;

pub(crate) fn validate_attention_config(config: &Config) -> Result<()> {
    if config.qk_rope_dim % 2 != 0 {
        return Err(Error::model(format!(
            "qk_rope_dim must be even for rotate-half RoPE, got {}",
            config.qk_rope_dim
        )));
    }
    validate_exact_shape(
        "qk_split",
        &[config.qk_no_rope_dim + config.qk_rope_dim],
        &[config.qk_head_dim],
    )
}

#[cfg(test)]
mod tests {
    use config::load_embedded_config;

    use super::*;

    #[test]
    fn rejects_odd_rope_dimension() {
        let mut config = load_embedded_config().unwrap();
        config.qk_rope_dim = 63;
        config.qk_no_rope_dim = 193;

        let err = validate_attention_config(&config).expect_err("odd RoPE dimension should fail");

        assert!(err.to_string().contains("qk_rope_dim"));
    }
}
