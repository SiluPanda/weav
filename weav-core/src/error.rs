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

    #[test]
    fn test_error_display_all_variants() {
        let variants: Vec<WeavError> = vec![
            WeavError::GraphNotFound("g1".into()),
            WeavError::NodeNotFound(1, 2),
            WeavError::EdgeNotFound(99),
            WeavError::DuplicateNode("alice".into()),
            WeavError::Conflict("merge conflict".into()),
            WeavError::TokenBudgetExceeded {
                budget: 1000,
                required: 2000,
            },
            WeavError::DimensionMismatch {
                expected: 1536,
                got: 768,
            },
            WeavError::ShardUnavailable(5),
            WeavError::PersistenceError("disk full".into()),
            WeavError::ProtocolError("bad frame".into()),
            WeavError::QueryParseError("unexpected token".into()),
            WeavError::CapacityExceeded("max nodes reached".into()),
            WeavError::InvalidConfig("bad port".into()),
            WeavError::Internal("something broke".into()),
        ];

        for err in &variants {
            let display = format!("{}", err);
            assert!(
                !display.is_empty(),
                "Display for {:?} should not be empty",
                err
            );
        }

        // Verify specific messages for the previously untested variants
        assert_eq!(
            WeavError::EdgeNotFound(99).to_string(),
            "edge 99 not found"
        );
        assert_eq!(
            WeavError::DuplicateNode("alice".into()).to_string(),
            "duplicate node: entity with key 'alice' already exists"
        );
        assert_eq!(
            WeavError::Conflict("merge conflict".into()).to_string(),
            "conflict detected: merge conflict"
        );
        assert_eq!(
            WeavError::ShardUnavailable(5).to_string(),
            "shard 5 unavailable"
        );
        assert_eq!(
            WeavError::PersistenceError("disk full".into()).to_string(),
            "persistence error: disk full"
        );
        assert_eq!(
            WeavError::ProtocolError("bad frame".into()).to_string(),
            "protocol error: bad frame"
        );
        assert_eq!(
            WeavError::QueryParseError("unexpected token".into()).to_string(),
            "query parse error: unexpected token"
        );
        assert_eq!(
            WeavError::CapacityExceeded("max nodes reached".into()).to_string(),
            "capacity exceeded: max nodes reached"
        );
        assert_eq!(
            WeavError::InvalidConfig("bad port".into()).to_string(),
            "invalid configuration: bad port"
        );
        assert_eq!(
            WeavError::Internal("something broke".into()).to_string(),
            "internal error: something broke"
        );
    }
}
