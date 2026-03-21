//! Change Data Capture (CDC) event types for graph mutations.
//!
//! Every successful write operation emits a [`GraphEvent`] through the engine's
//! broadcast channel, enabling real-time streaming to subscribers (SSE, gRPC, etc.).

use compact_str::CompactString;
use serde::{Deserialize, Serialize};

use crate::types::{EdgeId, NodeId, Timestamp, Value};

/// A graph mutation event emitted after a successful write operation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphEvent {
    /// Monotonically increasing sequence number.
    pub sequence: u64,
    /// Graph this event belongs to.
    pub graph: CompactString,
    /// When this event occurred (ms since epoch).
    pub timestamp: Timestamp,
    /// The mutation that occurred.
    pub kind: EventKind,
}

/// The specific kind of mutation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EventKind {
    NodeCreated {
        node_id: NodeId,
        label: CompactString,
        properties: Vec<(CompactString, Value)>,
    },
    NodeUpdated {
        node_id: NodeId,
        properties: Vec<(CompactString, Value)>,
    },
    NodeDeleted {
        node_id: NodeId,
    },
    EdgeCreated {
        edge_id: EdgeId,
        source: NodeId,
        target: NodeId,
        label: CompactString,
        weight: f32,
    },
    EdgeDeleted {
        edge_id: EdgeId,
    },
    EdgeInvalidated {
        edge_id: EdgeId,
    },
    GraphCreated {
        name: CompactString,
    },
    GraphDropped {
        name: CompactString,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_event_serialization_roundtrip() {
        let event = GraphEvent {
            sequence: 42,
            graph: CompactString::from("test_graph"),
            timestamp: 1700000000000,
            kind: EventKind::NodeCreated {
                node_id: 1,
                label: CompactString::from("Person"),
                properties: vec![(
                    CompactString::from("name"),
                    Value::String(CompactString::from("Alice")),
                )],
            },
        };
        let json = serde_json::to_string(&event).unwrap();
        let deserialized: GraphEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.sequence, 42);
        assert_eq!(deserialized.graph.as_str(), "test_graph");
        assert_eq!(deserialized.timestamp, 1700000000000);
        match deserialized.kind {
            EventKind::NodeCreated {
                node_id,
                ref label,
                ref properties,
            } => {
                assert_eq!(node_id, 1);
                assert_eq!(label.as_str(), "Person");
                assert_eq!(properties.len(), 1);
            }
            _ => panic!("expected NodeCreated"),
        }
    }

    #[test]
    fn test_all_event_kinds_serialize() {
        let kinds = vec![
            EventKind::NodeCreated {
                node_id: 1,
                label: CompactString::from("X"),
                properties: vec![],
            },
            EventKind::NodeUpdated {
                node_id: 2,
                properties: vec![(CompactString::from("k"), Value::Int(10))],
            },
            EventKind::NodeDeleted { node_id: 3 },
            EventKind::EdgeCreated {
                edge_id: 4,
                source: 1,
                target: 2,
                label: CompactString::from("KNOWS"),
                weight: 1.0,
            },
            EventKind::EdgeDeleted { edge_id: 5 },
            EventKind::EdgeInvalidated { edge_id: 6 },
            EventKind::GraphCreated {
                name: CompactString::from("g1"),
            },
            EventKind::GraphDropped {
                name: CompactString::from("g2"),
            },
        ];
        for kind in kinds {
            let event = GraphEvent {
                sequence: 0,
                graph: CompactString::from("g"),
                timestamp: 0,
                kind,
            };
            let json = serde_json::to_string(&event).unwrap();
            let _: GraphEvent = serde_json::from_str(&json).unwrap();
        }
    }
}
