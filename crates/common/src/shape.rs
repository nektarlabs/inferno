use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.dims)
    }
}

pub fn validate_exact_shape(
    context: impl Into<String>,
    actual: &[usize],
    expected: &[usize],
) -> Result<()> {
    if actual == expected {
        return Ok(());
    }

    Err(Error::ShapeMismatch {
        context: context.into(),
        expected: expected.to_vec(),
        actual: actual.to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_shape_validation_accepts_match() {
        validate_exact_shape("hidden_states", &[1, 4, 6144], &[1, 4, 6144]).unwrap();
    }

    #[test]
    fn exact_shape_validation_rejects_mismatch() {
        let err = validate_exact_shape("hidden_states", &[1, 4, 4096], &[1, 4, 6144])
            .expect_err("shape should fail");

        assert!(err.to_string().contains("hidden_states"));
    }
}
