use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Dense,
    SparseMoe,
}

impl fmt::Display for LayerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dense => f.write_str("dense"),
            Self::SparseMoe => f.write_str("sparse_moe"),
        }
    }
}
