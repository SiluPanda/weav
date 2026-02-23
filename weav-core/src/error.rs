//! Error types for Weav.

use crate::types::{EdgeId, GraphId, NodeId, ShardId};

/// The primary error type used throughout the Weav engine.
#[derive(Debug, thiserror::Error)]
pub enum WeavError {
    #[error("graph '{0}' not found")]
    GraphNotFound(String),

    #[error("node {0} not found in graph {1}")]
    NodeNotFound(NodeId, GraphId),

    #[error("edge {0} not found")]
    EdgeNotFound(EdgeId),

    #[error("duplicate node: entity with key '{0}' already exists")]
    DuplicateNode(String),

    #[error("conflict detected: {0}")]
    Conflict(String),

    #[error("token budget {budget} exceeded: context requires {required} tokens")]
    TokenBudgetExceeded { budget: u32, required: u32 },

    #[error("vector dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: u16, got: u16 },

    #[error("shard {0} unavailable")]
    ShardUnavailable(ShardId),

    #[error("persistence error: {0}")]
    PersistenceError(String),

    #[error("protocol error: {0}")]
    ProtocolError(String),

    #[error("query parse error: {0}")]
    QueryParseError(String),

    #[error("capacity exceeded: {0}")]
    CapacityExceeded(String),

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("internal error: {0}")]
    Internal(String),
}

/// Convenience type alias for `Result<T, WeavError>`.
pub type WeavResult<T> = Result<T, WeavError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = WeavError::GraphNotFound("test-graph".into());
        assert_eq!(err.to_string(), "graph 'test-graph' not found");

        let err = WeavError::NodeNotFound(42, 1);
        assert_eq!(err.to_string(), "node 42 not found in graph 1");

        let err = WeavError::DimensionMismatch {
            expected: 1536,
            got: 768,
        };
        assert_eq!(
            err.to_string(),
            "vector dimension mismatch: expected 1536, got 768"
        );

        let err = WeavError::TokenBudgetExceeded {
            budget: 4096,
            required: 5000,
        };
        assert_eq!(
            err.to_string(),
            "token budget 4096 exceeded: context requires 5000 tokens"
        );
    }
}
