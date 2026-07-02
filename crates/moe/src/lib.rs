#![deny(unsafe_code)]

//! Sparse MoE routing and dispatch planning primitives.

mod routing;

pub use routing::{build_dispatch_plan, ExpertAssignment, ExpertDispatch, TopKSelection};
