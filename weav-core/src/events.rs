//! Change Data Capture (CDC) event types for graph mutations.
//!
//! Every successful write operation emits a [`GraphEvent`] through the engine's
//! broadcast channel, enabling real-time streaming to subscribers (SSE, gRPC, etc.).

use compact_str::CompactString;
use serde::{Deserialize, Serialize};
use serde_json::json;

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

/// Minimal public representation of a graph event for transport layers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicGraphEvent {
    pub sequence: u64,
    pub graph: CompactString,
    pub timestamp_ms: Timestamp,
    pub kind: CompactString,
    pub payload_json: String,
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

impl EventKind {
    /// Stable public event kind string.
    pub fn as_str(&self) -> &'static str {
        match self {
            EventKind::NodeCreated { .. } => "node_created",
            EventKind::NodeUpdated { .. } => "node_updated",
            EventKind::NodeDeleted { .. } => "node_deleted",
            EventKind::EdgeCreated { .. } => "edge_created",
            EventKind::EdgeDeleted { .. } => "edge_deleted",
            EventKind::EdgeInvalidated { .. } => "edge_invalidated",
            EventKind::GraphCreated { .. } => "graph_created",
            EventKind::GraphDropped { .. } => "graph_dropped",
        }
    }

    fn payload_value(&self) -> serde_json::Value {
        match self {
            EventKind::NodeCreated {
                node_id,
                label,
                properties,
            } => json!({
                "node_id": node_id,
                "label": label,
                "properties": properties
                    .iter()
                    .map(|(key, value)| json!([key, value_to_json(value)]))
                    .collect::<Vec<_>>(),
            }),
            EventKind::NodeUpdated {
                node_id,
                properties,
            } => json!({
                "node_id": node_id,
                "properties": properties
                    .iter()
                    .map(|(key, value)| json!([key, value_to_json(value)]))
                    .collect::<Vec<_>>(),
            }),
            EventKind::NodeDeleted { node_id } => json!({
                "node_id": node_id,
            }),
            EventKind::EdgeCreated {
                edge_id,
                source,
                target,
                label,
                weight,
            } => json!({
                "edge_id": edge_id,
                "source": source,
                "target": target,
                "label": label,
                "weight": weight,
            }),
            EventKind::EdgeDeleted { edge_id } => json!({
                "edge_id": edge_id,
            }),
            EventKind::EdgeInvalidated { edge_id } => json!({
                "edge_id": edge_id,
            }),
            EventKind::GraphCreated { name } | EventKind::GraphDropped { name } => json!({
                "name": name,
            }),
        }
    }

    pub fn payload_json(&self) -> serde_json::Result<String> {
        serde_json::to_string(&self.payload_value())
    }
}

impl GraphEvent {
    pub fn to_public_event(&self) -> serde_json::Result<PublicGraphEvent> {
        Ok(PublicGraphEvent {
            sequence: self.sequence,
            graph: self.graph.clone(),
            timestamp_ms: self.timestamp,
            kind: CompactString::from(self.kind.as_str()),
            payload_json: self.kind.payload_json()?,
        })
    }
}

fn value_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Null => serde_json::Value::Null,
        Value::Bool(v) => json!(*v),
        Value::Int(v) => json!(*v),
        Value::Float(v) => json!(*v),
        Value::String(v) => json!(v.as_str()),
        Value::Bytes(v) => json!(v),
        Value::Vector(v) => json!(v),
        Value::List(values) => {
            serde_json::Value::Array(values.iter().map(value_to_json).collect::<Vec<_>>())
        }
        Value::Map(entries) => serde_json::Value::Object(
            entries
                .iter()
                .map(|(key, value)| (key.to_string(), value_to_json(value)))
                .collect(),
        ),
        Value::Timestamp(v) => json!(*v),
    }
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

    #[test]
    fn test_graph_event_public_projection() {
        let event = GraphEvent {
            sequence: 7,
            graph: CompactString::from("test_graph"),
            timestamp: 1700000000123,
            kind: EventKind::NodeUpdated {
                node_id: 99,
                properties: vec![(CompactString::from("name"), Value::String("Alice".into()))],
            },
        };

        let public = event.to_public_event().unwrap();
        assert_eq!(public.sequence, 7);
        assert_eq!(public.graph.as_str(), "test_graph");
        assert_eq!(public.timestamp_ms, 1700000000123);
        assert_eq!(public.kind.as_str(), "node_updated");

        let payload: serde_json::Value = serde_json::from_str(&public.payload_json).unwrap();
        assert_eq!(payload["node_id"], 99);
        assert_eq!(payload["properties"][0][0], "name");
        assert_eq!(payload["properties"][0][1], "Alice");
    }
}
