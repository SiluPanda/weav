//! Mapping between RESP3 wire format and Weav command strings / results.
//!
//! Weav commands are sent as RESP3 arrays of strings (like Redis).  This
//! module provides helpers to convert back and forth between the RESP3
//! representation and the internal command / result types.

use weav_core::error::WeavError;
use weav_query::executor::{ContextChunk, ContextResult, RelationshipSummary};

use crate::resp3::Resp3Value;

// ---- RESP3 array --> command string -----------------------------------------

/// Convert a RESP3 array into a raw command string that the
/// `weav_query::parser::parse_command` function can handle.
///
/// Each element of the array should be a `SimpleString` or `BlobString`.
/// Strings that contain spaces are automatically quoted.
///
/// # Examples
///
/// ```text
/// ["PING"]                         --> "PING"
/// ["CONTEXT", "query text", "FROM", "my-graph", "BUDGET", "4096", "TOKENS"]
///   --> "CONTEXT \"query text\" FROM \"my-graph\" BUDGET 4096 TOKENS"
/// ```
pub fn resp3_to_command_string(value: &Resp3Value) -> Result<String, WeavError> {
    let items = match value {
        Resp3Value::Array(items) => items,
        _ => {
            return Err(WeavError::ProtocolError(
                "command must be a RESP3 array".to_string(),
            ));
        }
    };

    if items.is_empty() {
        return Err(WeavError::ProtocolError("empty command array".to_string()));
    }

    let mut parts: Vec<String> = Vec::with_capacity(items.len());
    for item in items {
        let s = match item.as_str() {
            Some(s) => s.to_string(),
            None => {
                return Err(WeavError::ProtocolError(
                    "command array elements must be strings".to_string(),
                ));
            }
        };

        // Quote strings that contain spaces, quotes, or look like JSON.
        if s.contains(' ') || s.contains('"') || s.starts_with('{') || s.starts_with('[') {
            // Escape any existing double-quotes inside the string.
            let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
            parts.push(format!("\"{escaped}\""));
        } else {
            parts.push(s);
        }
    }

    Ok(parts.join(" "))
}

// ---- ContextResult --> RESP3 ------------------------------------------------

/// Convert a `ContextResult` to a RESP3 Map suitable for sending back over the wire.
///
/// The top-level map contains:
///   - `chunks`           : Array of chunk maps
///   - `total_tokens`     : Number
///   - `budget_used`      : Double
///   - `nodes_considered` : Number
///   - `nodes_included`   : Number
///   - `query_time_us`    : Number
pub fn context_result_to_resp3(result: &ContextResult) -> Resp3Value {
    let chunks_array: Vec<Resp3Value> = result.chunks.iter().map(chunk_to_resp3).collect();

    Resp3Value::Map(vec![
        (
            Resp3Value::SimpleString("chunks".to_string()),
            Resp3Value::Array(chunks_array),
        ),
        (
            Resp3Value::SimpleString("total_tokens".to_string()),
            Resp3Value::Number(result.total_tokens as i64),
        ),
        (
            Resp3Value::SimpleString("budget_used".to_string()),
            Resp3Value::Double(result.budget_used as f64),
        ),
        (
            Resp3Value::SimpleString("nodes_considered".to_string()),
            Resp3Value::Number(result.nodes_considered as i64),
        ),
        (
            Resp3Value::SimpleString("nodes_included".to_string()),
            Resp3Value::Number(result.nodes_included as i64),
        ),
        (
            Resp3Value::SimpleString("query_time_us".to_string()),
            Resp3Value::Number(result.query_time_us as i64),
        ),
    ])
}

/// Convert a single `ContextChunk` to a RESP3 Map.
fn chunk_to_resp3(chunk: &ContextChunk) -> Resp3Value {
    let mut pairs: Vec<(Resp3Value, Resp3Value)> = vec![
        (
            Resp3Value::SimpleString("node_id".to_string()),
            Resp3Value::Number(chunk.node_id as i64),
        ),
        (
            Resp3Value::SimpleString("content".to_string()),
            Resp3Value::BlobString(chunk.content.as_bytes().to_vec()),
        ),
        (
            Resp3Value::SimpleString("label".to_string()),
            Resp3Value::SimpleString(chunk.label.clone()),
        ),
        (
            Resp3Value::SimpleString("relevance_score".to_string()),
            Resp3Value::Double(chunk.relevance_score as f64),
        ),
        (
            Resp3Value::SimpleString("depth".to_string()),
            Resp3Value::Number(chunk.depth as i64),
        ),
        (
            Resp3Value::SimpleString("token_count".to_string()),
            Resp3Value::Number(chunk.token_count as i64),
        ),
    ];

    // Optional provenance
    if let Some(ref prov) = chunk.provenance {
        pairs.push((
            Resp3Value::SimpleString("provenance".to_string()),
            Resp3Value::Map(vec![
                (
                    Resp3Value::SimpleString("source".to_string()),
                    Resp3Value::SimpleString(prov.source.to_string()),
                ),
                (
                    Resp3Value::SimpleString("confidence".to_string()),
                    Resp3Value::Double(prov.confidence as f64),
                ),
                (
                    Resp3Value::SimpleString("extraction_method".to_string()),
                    Resp3Value::SimpleString(format!("{:?}", prov.extraction_method)),
                ),
            ]),
        ));
    }

    // Relationships (always include, may be empty array)
    if !chunk.relationships.is_empty() {
        let rels: Vec<Resp3Value> = chunk
            .relationships
            .iter()
            .map(relationship_to_resp3)
            .collect();
        pairs.push((
            Resp3Value::SimpleString("relationships".to_string()),
            Resp3Value::Array(rels),
        ));
    }

    Resp3Value::Map(pairs)
}

/// Convert a `RelationshipSummary` to a RESP3 Map.
fn relationship_to_resp3(rel: &RelationshipSummary) -> Resp3Value {
    let mut pairs = vec![
        (
            Resp3Value::SimpleString("edge_label".to_string()),
            Resp3Value::SimpleString(rel.edge_label.clone()),
        ),
        (
            Resp3Value::SimpleString("target_node_id".to_string()),
            Resp3Value::Number(rel.target_node_id as i64),
        ),
        (
            Resp3Value::SimpleString("direction".to_string()),
            Resp3Value::SimpleString(rel.direction.clone()),
        ),
        (
            Resp3Value::SimpleString("weight".to_string()),
            Resp3Value::Double(rel.weight as f64),
        ),
    ];

    if let Some(ref name) = rel.target_name {
        pairs.push((
            Resp3Value::SimpleString("target_name".to_string()),
            Resp3Value::SimpleString(name.clone()),
        ));
    }

    Resp3Value::Map(pairs)
}

// ---- Error / OK / helpers ---------------------------------------------------

/// Convert a `WeavError` to a RESP3 simple error response.
pub fn error_to_resp3(err: &WeavError) -> Resp3Value {
    Resp3Value::SimpleError(err.to_string())
}

/// Return a `+OK\r\n` acknowledgment value.
pub fn ok_resp3() -> Resp3Value {
    Resp3Value::ok()
}

/// Convert a node id to a RESP3 number.
pub fn node_id_to_resp3(id: u64) -> Resp3Value {
    Resp3Value::Number(id as i64)
}

/// Convert a list of strings to a RESP3 array of bulk strings.
pub fn string_list_to_resp3(items: Vec<String>) -> Resp3Value {
    Resp3Value::Array(
        items
            .into_iter()
            .map(|s| Resp3Value::BlobString(s.into_bytes()))
            .collect(),
    )
}

// ---- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use weav_query::executor::{ContextChunk, ContextResult, RelationshipSummary};

    // -- resp3_to_command_string -----------------------------------------------

    #[test]
    fn test_ping_command() {
        let val = Resp3Value::Array(vec![Resp3Value::BlobString(b"PING".to_vec())]);
        let cmd = resp3_to_command_string(&val).unwrap();
        assert_eq!(cmd, "PING");
    }

    #[test]
    fn test_context_command() {
        let val = Resp3Value::Array(vec![
            Resp3Value::BlobString(b"CONTEXT".to_vec()),
            Resp3Value::BlobString(b"query text".to_vec()),
            Resp3Value::BlobString(b"FROM".to_vec()),
            Resp3Value::BlobString(b"my-graph".to_vec()),
            Resp3Value::BlobString(b"BUDGET".to_vec()),
            Resp3Value::BlobString(b"4096".to_vec()),
            Resp3Value::BlobString(b"TOKENS".to_vec()),
        ]);
        let cmd = resp3_to_command_string(&val).unwrap();
        assert_eq!(
            cmd,
            "CONTEXT \"query text\" FROM my-graph BUDGET 4096 TOKENS"
        );
    }

    #[test]
    fn test_graph_create_command() {
        let val = Resp3Value::Array(vec![
            Resp3Value::SimpleString("GRAPH".to_string()),
            Resp3Value::SimpleString("CREATE".to_string()),
            Resp3Value::SimpleString("my_graph".to_string()),
        ]);
        let cmd = resp3_to_command_string(&val).unwrap();
        assert_eq!(cmd, "GRAPH CREATE my_graph");
    }

    #[test]
    fn test_node_add_with_json() {
        let val = Resp3Value::Array(vec![
            Resp3Value::BlobString(b"NODE".to_vec()),
            Resp3Value::BlobString(b"ADD".to_vec()),
            Resp3Value::BlobString(b"TO".to_vec()),
            Resp3Value::BlobString(b"test_graph".to_vec()),
            Resp3Value::BlobString(b"LABEL".to_vec()),
            Resp3Value::BlobString(b"person".to_vec()),
            Resp3Value::BlobString(b"PROPERTIES".to_vec()),
            Resp3Value::BlobString(b"{\"name\": \"Alice\"}".to_vec()),
        ]);
        let cmd = resp3_to_command_string(&val).unwrap();
        // The JSON object should be quoted.
        assert!(cmd.contains("PROPERTIES"));
        assert!(cmd.contains("Alice"));
    }

    #[test]
    fn test_non_array_is_error() {
        let val = Resp3Value::SimpleString("PING".to_string());
        let result = resp3_to_command_string(&val);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_array_is_error() {
        let val = Resp3Value::Array(vec![]);
        let result = resp3_to_command_string(&val);
        assert!(result.is_err());
    }

    #[test]
    fn test_non_string_element_is_error() {
        let val = Resp3Value::Array(vec![Resp3Value::Number(42)]);
        let result = resp3_to_command_string(&val);
        assert!(result.is_err());
    }

    #[test]
    fn test_simple_string_elements() {
        let val = Resp3Value::Array(vec![
            Resp3Value::SimpleString("GRAPH".to_string()),
            Resp3Value::SimpleString("LIST".to_string()),
        ]);
        let cmd = resp3_to_command_string(&val).unwrap();
        assert_eq!(cmd, "GRAPH LIST");
    }

    // -- context_result_to_resp3 ----------------------------------------------

    fn sample_context_result() -> ContextResult {
        ContextResult {
            chunks: vec![ContextChunk {
                node_id: 1,
                content: "Alice is a software engineer".to_string(),
                label: "person".to_string(),
                relevance_score: 0.95,
                depth: 0,
                token_count: 7,
                provenance: None,
                relationships: vec![RelationshipSummary {
                    edge_label: "knows".to_string(),
                    target_node_id: 2,
                    target_name: Some("Bob".to_string()),
                    direction: "outgoing".to_string(),
                    weight: 0.9,
                }],
                temporal: None,
            }],
            total_tokens: 7,
            budget_used: 0.5,
            nodes_considered: 10,
            nodes_included: 1,
            query_time_us: 123,
            conflicts: Vec::new(),
        }
    }

    #[test]
    fn test_context_result_to_resp3_structure() {
        let result = sample_context_result();
        let val = context_result_to_resp3(&result);

        // Top level should be a Map.
        match &val {
            Resp3Value::Map(pairs) => {
                // Should have 6 top-level keys.
                assert_eq!(pairs.len(), 6);

                // Find chunks key.
                let chunks_pair = pairs.iter().find(|(k, _)| k.as_str() == Some("chunks"));
                assert!(chunks_pair.is_some());

                let (_, chunks_val) = chunks_pair.unwrap();
                match chunks_val {
                    Resp3Value::Array(chunks) => {
                        assert_eq!(chunks.len(), 1);
                        // Each chunk is a map.
                        match &chunks[0] {
                            Resp3Value::Map(chunk_pairs) => {
                                // Should have node_id, content, label, relevance_score,
                                // depth, token_count, relationships.
                                assert!(chunk_pairs.len() >= 6);
                            }
                            _ => panic!("expected chunk to be a Map"),
                        }
                    }
                    _ => panic!("expected chunks to be an Array"),
                }

                // Check total_tokens.
                let tt = pairs
                    .iter()
                    .find(|(k, _)| k.as_str() == Some("total_tokens"));
                assert!(matches!(tt, Some((_, Resp3Value::Number(7)))));

                // Check query_time_us.
                let qt = pairs
                    .iter()
                    .find(|(k, _)| k.as_str() == Some("query_time_us"));
                assert!(matches!(qt, Some((_, Resp3Value::Number(123)))));
            }
            _ => panic!("expected top-level Map"),
        }
    }

    #[test]
    fn test_context_result_empty_chunks() {
        let result = ContextResult {
            chunks: vec![],
            total_tokens: 0,
            budget_used: 0.0,
            nodes_considered: 0,
            nodes_included: 0,
            query_time_us: 1,
            conflicts: Vec::new(),
        };
        let val = context_result_to_resp3(&result);
        match &val {
            Resp3Value::Map(pairs) => {
                let chunks_pair = pairs.iter().find(|(k, _)| k.as_str() == Some("chunks"));
                match chunks_pair {
                    Some((_, Resp3Value::Array(arr))) => assert!(arr.is_empty()),
                    _ => panic!("expected empty chunks array"),
                }
            }
            _ => panic!("expected Map"),
        }
    }

    #[test]
    fn test_context_result_with_provenance() {
        use weav_core::types::{ExtractionMethod, Provenance};

        let result = ContextResult {
            chunks: vec![ContextChunk {
                node_id: 5,
                content: "test".to_string(),
                label: "entity".to_string(),
                relevance_score: 0.8,
                depth: 1,
                token_count: 1,
                provenance: Some(Provenance {
                    source: "gpt-4".into(),
                    confidence: 0.9,
                    extraction_method: ExtractionMethod::LlmExtracted,
                    source_document_id: None,
                    source_chunk_offset: None,
                }),
                relationships: vec![],
                temporal: None,
            }],
            total_tokens: 1,
            budget_used: 0.01,
            nodes_considered: 1,
            nodes_included: 1,
            query_time_us: 50,
            conflicts: Vec::new(),
        };

        let val = context_result_to_resp3(&result);
        // Drill into the first chunk and confirm provenance is present.
        if let Resp3Value::Map(pairs) = &val {
            if let Some((_, Resp3Value::Array(chunks))) =
                pairs.iter().find(|(k, _)| k.as_str() == Some("chunks"))
            {
                if let Resp3Value::Map(chunk_pairs) = &chunks[0] {
                    let prov = chunk_pairs
                        .iter()
                        .find(|(k, _)| k.as_str() == Some("provenance"));
                    assert!(prov.is_some(), "provenance field should be present");
                } else {
                    panic!("expected chunk Map");
                }
            }
        }
    }

    // -- error_to_resp3 -------------------------------------------------------

    #[test]
    fn test_error_to_resp3() {
        let err = WeavError::GraphNotFound("test".to_string());
        let val = error_to_resp3(&err);
        match val {
            Resp3Value::SimpleError(msg) => {
                assert!(msg.contains("test"));
                assert!(msg.contains("not found"));
            }
            _ => panic!("expected SimpleError"),
        }
    }

    // -- ok_resp3 -------------------------------------------------------------

    #[test]
    fn test_ok_resp3() {
        assert_eq!(ok_resp3(), Resp3Value::SimpleString("OK".to_string()));
    }

    // -- node_id_to_resp3 -----------------------------------------------------

    #[test]
    fn test_node_id_to_resp3() {
        assert_eq!(node_id_to_resp3(42), Resp3Value::Number(42));
        assert_eq!(node_id_to_resp3(0), Resp3Value::Number(0));
    }

    // -- string_list_to_resp3 -------------------------------------------------

    #[test]
    fn test_string_list_to_resp3() {
        let items = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        let val = string_list_to_resp3(items);
        match val {
            Resp3Value::Array(arr) => {
                assert_eq!(arr.len(), 3);
                assert_eq!(arr[0].as_str(), Some("alpha"));
                assert_eq!(arr[1].as_str(), Some("beta"));
                assert_eq!(arr[2].as_str(), Some("gamma"));
            }
            _ => panic!("expected Array"),
        }
    }

    #[test]
    fn test_string_list_empty() {
        let val = string_list_to_resp3(vec![]);
        match val {
            Resp3Value::Array(arr) => assert!(arr.is_empty()),
            _ => panic!("expected empty Array"),
        }
    }

    // -- Integration-style: resp3 -> command string -> parseable ---------------

    #[test]
    fn test_command_string_parseable_by_weav_parser() {
        // Build a RESP3 array that should yield a valid PING command string.
        let val = Resp3Value::Array(vec![Resp3Value::BlobString(b"PING".to_vec())]);
        let cmd_str = resp3_to_command_string(&val).unwrap();
        let cmd = weav_query::parser::parse_command(&cmd_str).unwrap();
        assert!(matches!(cmd, weav_query::parser::Command::Ping));
    }

    #[test]
    fn test_command_string_graph_list_parseable() {
        let val = Resp3Value::Array(vec![
            Resp3Value::BlobString(b"GRAPH".to_vec()),
            Resp3Value::BlobString(b"LIST".to_vec()),
        ]);
        let cmd_str = resp3_to_command_string(&val).unwrap();
        let cmd = weav_query::parser::parse_command(&cmd_str).unwrap();
        assert!(matches!(cmd, weav_query::parser::Command::GraphList));
    }

    #[test]
    fn test_command_string_stats_parseable() {
        let val = Resp3Value::Array(vec![
            Resp3Value::BlobString(b"STATS".to_vec()),
            Resp3Value::BlobString(b"my_graph".to_vec()),
        ]);
        let cmd_str = resp3_to_command_string(&val).unwrap();
        let cmd = weav_query::parser::parse_command(&cmd_str).unwrap();
        match cmd {
            weav_query::parser::Command::Stats(Some(name)) => assert_eq!(name, "my_graph"),
            _ => panic!("expected Stats(Some)"),
        }
    }

    // -- error_to_resp3 multiple variants --------------------------------------

    #[test]
    fn test_error_to_resp3_multiple_variants() {
        let cases: Vec<WeavError> = vec![
            WeavError::NodeNotFound(42, 1),
            WeavError::EdgeNotFound(99),
            WeavError::DuplicateNode("alice".to_string()),
            WeavError::QueryParseError("unexpected token".to_string()),
            WeavError::Internal("something broke".to_string()),
        ];

        for err in &cases {
            let val = error_to_resp3(err);
            match &val {
                Resp3Value::SimpleError(msg) => {
                    // The message should match the Display impl of WeavError.
                    assert_eq!(msg, &err.to_string());
                }
                other => panic!(
                    "expected SimpleError for {:?}, got {:?}",
                    err, other
                ),
            }
        }
    }

    // -- context_result with empty relationships ---------------------------------

    #[test]
    fn test_context_result_empty_relationships() {
        let result = ContextResult {
            chunks: vec![ContextChunk {
                node_id: 10,
                content: "Some content".to_string(),
                label: "document".to_string(),
                relevance_score: 0.7,
                depth: 0,
                token_count: 3,
                provenance: None,
                relationships: vec![],
                temporal: None,
            }],
            total_tokens: 3,
            budget_used: 0.1,
            nodes_considered: 5,
            nodes_included: 1,
            query_time_us: 42,
            conflicts: Vec::new(),
        };

        let val = context_result_to_resp3(&result);
        // Drill into the first chunk.
        if let Resp3Value::Map(pairs) = &val {
            if let Some((_, Resp3Value::Array(chunks))) =
                pairs.iter().find(|(k, _)| k.as_str() == Some("chunks"))
            {
                assert_eq!(chunks.len(), 1);
                if let Resp3Value::Map(chunk_pairs) = &chunks[0] {
                    // When relationships is empty, chunk_to_resp3 skips the
                    // "relationships" key entirely, so it should not be present.
                    let has_rels = chunk_pairs
                        .iter()
                        .any(|(k, _)| k.as_str() == Some("relationships"));
                    assert!(
                        !has_rels,
                        "relationships key should be absent when relationships vec is empty"
                    );
                    // Should have exactly 6 fields: node_id, content, label,
                    // relevance_score, depth, token_count.
                    assert_eq!(chunk_pairs.len(), 6);
                } else {
                    panic!("expected chunk Map");
                }
            } else {
                panic!("expected chunks Array");
            }
        } else {
            panic!("expected top-level Map");
        }
    }

    // -- relationship with no target_name ----------------------------------------

    #[test]
    fn test_relationship_no_target_name() {
        let result = ContextResult {
            chunks: vec![ContextChunk {
                node_id: 1,
                content: "test node".to_string(),
                label: "entity".to_string(),
                relevance_score: 0.9,
                depth: 0,
                token_count: 2,
                provenance: None,
                relationships: vec![RelationshipSummary {
                    edge_label: "relates_to".to_string(),
                    target_node_id: 5,
                    target_name: None,
                    direction: "outgoing".to_string(),
                    weight: 0.8,
                }],
                temporal: None,
            }],
            total_tokens: 2,
            budget_used: 0.05,
            nodes_considered: 3,
            nodes_included: 1,
            query_time_us: 10,
            conflicts: Vec::new(),
        };

        let val = context_result_to_resp3(&result);
        // Drill into the first chunk's relationships.
        if let Resp3Value::Map(pairs) = &val {
            if let Some((_, Resp3Value::Array(chunks))) =
                pairs.iter().find(|(k, _)| k.as_str() == Some("chunks"))
            {
                if let Resp3Value::Map(chunk_pairs) = &chunks[0] {
                    let rels_pair = chunk_pairs
                        .iter()
                        .find(|(k, _)| k.as_str() == Some("relationships"));
                    assert!(rels_pair.is_some(), "relationships should be present");

                    if let Some((_, Resp3Value::Array(rels))) = rels_pair {
                        assert_eq!(rels.len(), 1);
                        if let Resp3Value::Map(rel_pairs) = &rels[0] {
                            // When target_name is None, the key should be absent.
                            let has_target_name = rel_pairs
                                .iter()
                                .any(|(k, _)| k.as_str() == Some("target_name"));
                            assert!(
                                !has_target_name,
                                "target_name key should be absent when target_name is None"
                            );
                            // Should have exactly 4 fields: edge_label,
                            // target_node_id, direction, weight.
                            assert_eq!(rel_pairs.len(), 4);
                        } else {
                            panic!("expected relationship Map");
                        }
                    } else {
                        panic!("expected relationships Array");
                    }
                } else {
                    panic!("expected chunk Map");
                }
            } else {
                panic!("expected chunks Array");
            }
        } else {
            panic!("expected top-level Map");
        }
    }
}
