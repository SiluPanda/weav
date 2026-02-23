//! Query executor that runs context queries against the graph stores.

use std::time::Instant;

use weav_core::error::WeavError;
use weav_core::shard::StringInterner;
use weav_core::types::{Direction, NodeId, Provenance, Timestamp};

use weav_graph::adjacency::AdjacencyStore;
use weav_graph::properties::PropertyStore;
use weav_graph::traversal::flow_score;
use weav_vector::index::VectorIndex;
use weav_vector::tokens::TokenCounter;

use crate::budget::enforce_budget;
use crate::parser::{ContextQuery, SeedStrategy};

// ─── Result types ───────────────────────────────────────────────────────────

/// The complete result of a context query.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ContextResult {
    /// Ordered context chunks.
    pub chunks: Vec<ContextChunk>,
    /// Total tokens consumed by included chunks.
    pub total_tokens: u32,
    /// Fraction of token budget used (0.0..1.0).
    pub budget_used: f32,
    /// Total number of nodes considered during traversal.
    pub nodes_considered: u32,
    /// Number of nodes included in the final output.
    pub nodes_included: u32,
    /// Wall-clock query time in microseconds.
    pub query_time_us: u64,
}

/// A single chunk of context extracted from the graph.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ContextChunk {
    /// The node this chunk represents.
    pub node_id: NodeId,
    /// Concatenated text content from the node's string properties.
    pub content: String,
    /// The node's label.
    pub label: String,
    /// Relevance score (from flow scoring).
    pub relevance_score: f32,
    /// BFS/flow depth from the seed nodes.
    pub depth: u8,
    /// Token count for this chunk.
    pub token_count: u32,
    /// Optional provenance metadata.
    pub provenance: Option<Provenance>,
    /// Summary of this node's relationships.
    pub relationships: Vec<RelationshipSummary>,
}

/// Summary of a relationship (edge) for context output.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RelationshipSummary {
    /// The edge label.
    pub edge_label: String,
    /// The node on the other end of the edge.
    pub target_node_id: NodeId,
    /// Optional name/title of the target node.
    pub target_name: Option<String>,
    /// Direction of the edge relative to the chunk's node.
    pub direction: String,
    /// Edge weight.
    pub weight: f32,
}

// ─── Executor ───────────────────────────────────────────────────────────────

/// Execute a context query against the graph stores.
///
/// Steps:
/// 1. Get seed nodes (vector search and/or node key lookup)
/// 2. Run `flow_score` from seeds
/// 3. Build `ContextChunk`s from scored nodes (read properties, build content)
/// 4. Apply temporal filter if requested
/// 5. Apply decay function if set
/// 6. Run budget enforcement
/// 7. Build `RelationshipSummary` for each included node
/// 8. Return `ContextResult` with timing
pub fn execute_context_query(
    query: &ContextQuery,
    adjacency: &AdjacencyStore,
    properties: &PropertyStore,
    vector_index: &VectorIndex,
    token_counter: &TokenCounter,
    interner: &StringInterner,
) -> Result<ContextResult, WeavError> {
    let start = Instant::now();

    // ── Step 1: Get seed nodes ──────────────────────────────────────────
    let mut seeds_with_scores: Vec<(NodeId, f32)> = Vec::new();

    match &query.seeds {
        SeedStrategy::Vector { embedding, top_k } => {
            let results = vector_index.search(embedding, *top_k, None)?;
            for (node_id, distance) in results {
                // Convert distance to similarity score (lower distance = higher score)
                let score = 1.0 / (1.0 + distance);
                seeds_with_scores.push((node_id, score));
            }
        }
        SeedStrategy::Nodes(keys) => {
            for key in keys {
                let key_clone = key.clone();
                let nodes = properties.nodes_where("entity_key", &move |v| {
                    v.as_str() == Some(key_clone.as_str())
                });
                for nid in nodes {
                    seeds_with_scores.push((nid, 1.0));
                }
            }
        }
        SeedStrategy::Both {
            embedding,
            top_k,
            node_keys,
        } => {
            // Vector search
            let results = vector_index.search(embedding, *top_k, None)?;
            for (node_id, distance) in results {
                let score = 1.0 / (1.0 + distance);
                seeds_with_scores.push((node_id, score));
            }
            // Node key lookup
            for key in node_keys {
                let key_clone = key.clone();
                let nodes = properties.nodes_where("entity_key", &move |v| {
                    v.as_str() == Some(key_clone.as_str())
                });
                for nid in nodes {
                    if !seeds_with_scores.iter().any(|(id, _)| *id == nid) {
                        seeds_with_scores.push((nid, 1.0));
                    }
                }
            }
        }
    }

    // ── Step 2: Run flow_score from seeds ───────────────────────────────
    let scored_nodes = flow_score(
        adjacency,
        &seeds_with_scores,
        0.5,  // alpha
        0.01, // theta
        query.max_depth,
    );

    let nodes_considered = scored_nodes.len() as u32;

    // ── Step 3: Build ContextChunks ─────────────────────────────────────
    let mut chunks: Vec<ContextChunk> = Vec::new();

    for scored in &scored_nodes {
        let node_id = scored.node_id;

        // Build content by concatenating all string properties
        let all_props = properties.get_all_node_properties(node_id);
        let content = build_content(&all_props);

        // Get the node's label
        let label = if let Some(label_val) = properties.get_node_property(node_id, "_label") {
            label_val.as_str().unwrap_or("unknown").to_string()
        } else {
            // Try to resolve from the interner if there's a _label_id
            if let Some(label_id_val) = properties.get_node_property(node_id, "_label_id") {
                if let Some(lid) = label_id_val.as_int() {
                    interner
                        .resolve_label(lid as u16)
                        .unwrap_or("unknown")
                        .to_string()
                } else {
                    "unknown".to_string()
                }
            } else {
                "unknown".to_string()
            }
        };

        // Count tokens
        let token_count = token_counter.count(&content);

        // Get provenance if stored
        let provenance = None; // Provenance is stored at the edge level, not node level

        chunks.push(ContextChunk {
            node_id,
            content,
            label,
            relevance_score: scored.score,
            depth: scored.depth,
            token_count,
            provenance,
            relationships: Vec::new(), // Filled in step 7
        });
    }

    // ── Step 4: Temporal filter ─────────────────────────────────────────
    if let Some(ts) = query.temporal_at {
        chunks.retain(|chunk| {
            // Check if node has temporal metadata that's valid at the timestamp
            if let Some(valid_from) = properties.get_node_property(chunk.node_id, "_valid_from") {
                if let Some(vf) = valid_from.as_int() {
                    if (vf as u64) > ts {
                        return false;
                    }
                }
            }
            if let Some(valid_until) = properties.get_node_property(chunk.node_id, "_valid_until") {
                if let Some(vu) = valid_until.as_int() {
                    let vu = vu as u64;
                    if vu != u64::MAX && ts >= vu {
                        return false;
                    }
                }
            }
            true
        });
    }

    // ── Step 5: Apply decay function ────────────────────────────────────
    if let Some(ref decay) = query.decay {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as Timestamp;

        for chunk in &mut chunks {
            // Look up the node's creation timestamp
            let item_ts = if let Some(ts_val) =
                properties.get_node_property(chunk.node_id, "_created_at")
            {
                ts_val.as_int().unwrap_or(0) as Timestamp
            } else {
                0
            };
            chunk.relevance_score = decay.apply(chunk.relevance_score, item_ts, now);
        }

        // Remove chunks with zero relevance after decay
        chunks.retain(|c| c.relevance_score > 0.0);
    }

    // Apply limit if set (before budget enforcement)
    if let Some(limit) = query.limit {
        chunks.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        chunks.truncate(limit as usize);
    }

    // ── Step 6: Budget enforcement ──────────────────────────────────────
    let (final_chunks, total_tokens, budget_used) = if let Some(ref budget) = query.budget {
        let result = enforce_budget(chunks, budget);
        let total = result.total_tokens;
        let util = result.budget_utilization;
        (result.included, total, util)
    } else {
        let total: u32 = chunks.iter().map(|c| c.token_count).sum();
        (chunks, total, 0.0)
    };

    // ── Step 7: Build RelationshipSummary ───────────────────────────────
    let mut result_chunks: Vec<ContextChunk> = final_chunks;

    for chunk in &mut result_chunks {
        let neighbors = adjacency.neighbors_both(chunk.node_id, None);
        for (neighbor_id, edge_id, dir) in &neighbors {
            if let Some(edge_meta) = adjacency.get_edge(*edge_id) {
                let edge_label = interner
                    .resolve_label(edge_meta.label)
                    .unwrap_or("unknown")
                    .to_string();

                let target_name = properties
                    .get_node_property(*neighbor_id, "name")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let direction_str = match dir {
                    Direction::Outgoing => "outgoing".to_string(),
                    Direction::Incoming => "incoming".to_string(),
                    Direction::Both => "both".to_string(),
                };

                chunk.relationships.push(RelationshipSummary {
                    edge_label,
                    target_node_id: *neighbor_id,
                    target_name,
                    direction: direction_str,
                    weight: edge_meta.weight,
                });
            }
        }
    }

    // ── Step 8: Build final result ──────────────────────────────────────
    let elapsed = start.elapsed();
    let nodes_included = result_chunks.len() as u32;

    Ok(ContextResult {
        chunks: result_chunks,
        total_tokens,
        budget_used,
        nodes_considered,
        nodes_included,
        query_time_us: elapsed.as_micros() as u64,
    })
}

/// Build a content string from a node's properties by concatenating
/// all string-typed values.
fn build_content(props: &[(&str, &weav_core::types::Value)]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for (key, val) in props {
        // Skip internal properties
        if key.starts_with('_') {
            continue;
        }
        if let Some(s) = val.as_str() {
            parts.push(format!("{key}: {s}"));
        }
    }
    parts.join("\n")
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use compact_str::CompactString;
    use weav_core::config::TokenCounterType;
    use weav_core::shard::StringInterner;
    use weav_core::types::{BiTemporal, Direction, TokenBudget, Value};
    use weav_graph::adjacency::{AdjacencyStore, EdgeMeta};
    use weav_graph::properties::PropertyStore;
    use weav_vector::index::{VectorConfig, VectorIndex};
    use weav_vector::tokens::TokenCounter;

    use crate::parser::{ContextQuery, SeedStrategy};

    fn setup_test_stores() -> (
        AdjacencyStore,
        PropertyStore,
        VectorIndex,
        TokenCounter,
        StringInterner,
    ) {
        let mut adj = AdjacencyStore::new();
        let mut props = PropertyStore::new();
        let mut interner = StringInterner::new();
        let token_counter = TokenCounter::new(TokenCounterType::CharDiv4);

        // Create nodes
        for i in 1..=4 {
            adj.add_node(i);
        }

        // Set properties
        props.set_node_property(
            1,
            "name",
            Value::String(CompactString::from("Alice")),
        );
        props.set_node_property(
            1,
            "entity_key",
            Value::String(CompactString::from("alice")),
        );
        props.set_node_property(
            1,
            "_label",
            Value::String(CompactString::from("person")),
        );
        props.set_node_property(
            1,
            "description",
            Value::String(CompactString::from("A software engineer")),
        );

        props.set_node_property(
            2,
            "name",
            Value::String(CompactString::from("Bob")),
        );
        props.set_node_property(
            2,
            "entity_key",
            Value::String(CompactString::from("bob")),
        );
        props.set_node_property(
            2,
            "_label",
            Value::String(CompactString::from("person")),
        );
        props.set_node_property(
            2,
            "description",
            Value::String(CompactString::from("A data scientist")),
        );

        props.set_node_property(
            3,
            "name",
            Value::String(CompactString::from("Rust")),
        );
        props.set_node_property(
            3,
            "entity_key",
            Value::String(CompactString::from("rust")),
        );
        props.set_node_property(
            3,
            "_label",
            Value::String(CompactString::from("topic")),
        );
        props.set_node_property(
            3,
            "description",
            Value::String(CompactString::from("A systems programming language")),
        );

        props.set_node_property(
            4,
            "name",
            Value::String(CompactString::from("Python")),
        );
        props.set_node_property(
            4,
            "_label",
            Value::String(CompactString::from("topic")),
        );

        // Add edges
        let knows_label = interner.intern_label("knows");
        let uses_label = interner.intern_label("uses");

        let meta1 = EdgeMeta {
            source: 1,
            target: 2,
            label: knows_label,
            temporal: BiTemporal::new_current(1000),
            provenance: None,
            weight: 0.9,
            token_cost: 0,
        };
        adj.add_edge(1, 2, knows_label, meta1).unwrap();

        let meta2 = EdgeMeta {
            source: 1,
            target: 3,
            label: uses_label,
            temporal: BiTemporal::new_current(1000),
            provenance: None,
            weight: 0.8,
            token_cost: 0,
        };
        adj.add_edge(1, 3, uses_label, meta2).unwrap();

        let meta3 = EdgeMeta {
            source: 2,
            target: 4,
            label: uses_label,
            temporal: BiTemporal::new_current(1000),
            provenance: None,
            weight: 0.7,
            token_cost: 0,
        };
        adj.add_edge(2, 4, uses_label, meta3).unwrap();

        // Create a small vector index
        let vec_config = VectorConfig {
            dimensions: 4,
            ..VectorConfig::default()
        };
        let vec_index = VectorIndex::new(vec_config).unwrap();

        (adj, props, vec_index, token_counter, interner)
    }

    #[test]
    fn test_execute_context_query_with_node_seeds() {
        let (adj, props, vec_index, token_counter, interner) = setup_test_stores();

        let query = ContextQuery {
            query_text: Some("test".to_string()),
            graph: "test".to_string(),
            budget: None,
            seeds: SeedStrategy::Nodes(vec!["alice".to_string()]),
            max_depth: 2,
            direction: Direction::Both,
            edge_filter: None,
            decay: None,
            include_provenance: false,
            temporal_at: None,
            limit: None,
        };

        let result =
            execute_context_query(&query, &adj, &props, &vec_index, &token_counter, &interner)
                .unwrap();

        assert!(result.nodes_considered > 0);
        assert!(result.nodes_included > 0);
        assert!(result.query_time_us > 0);

        // Alice should be in the results as a seed
        let alice_chunk = result.chunks.iter().find(|c| c.node_id == 1);
        assert!(alice_chunk.is_some());
        let alice = alice_chunk.unwrap();
        assert!(alice.content.contains("Alice"));
    }

    #[test]
    fn test_execute_context_query_with_budget() {
        let (adj, props, vec_index, token_counter, interner) = setup_test_stores();

        let query = ContextQuery {
            query_text: Some("test".to_string()),
            graph: "test".to_string(),
            budget: Some(TokenBudget::new(10)), // Very small budget
            seeds: SeedStrategy::Nodes(vec!["alice".to_string()]),
            max_depth: 2,
            direction: Direction::Both,
            edge_filter: None,
            decay: None,
            include_provenance: false,
            temporal_at: None,
            limit: None,
        };

        let result =
            execute_context_query(&query, &adj, &props, &vec_index, &token_counter, &interner)
                .unwrap();

        // Should have budget enforcement applied
        assert!(result.total_tokens <= 10);
        assert!(result.budget_used <= 1.0);
    }

    #[test]
    fn test_execute_context_query_with_limit() {
        let (adj, props, vec_index, token_counter, interner) = setup_test_stores();

        let query = ContextQuery {
            query_text: Some("test".to_string()),
            graph: "test".to_string(),
            budget: None,
            seeds: SeedStrategy::Nodes(vec!["alice".to_string()]),
            max_depth: 3,
            direction: Direction::Both,
            edge_filter: None,
            decay: None,
            include_provenance: false,
            temporal_at: None,
            limit: Some(1),
        };

        let result =
            execute_context_query(&query, &adj, &props, &vec_index, &token_counter, &interner)
                .unwrap();

        assert!(result.nodes_included <= 1);
    }

    #[test]
    fn test_execute_context_query_relationships() {
        let (adj, props, vec_index, token_counter, interner) = setup_test_stores();

        let query = ContextQuery {
            query_text: Some("test".to_string()),
            graph: "test".to_string(),
            budget: None,
            seeds: SeedStrategy::Nodes(vec!["alice".to_string()]),
            max_depth: 1,
            direction: Direction::Both,
            edge_filter: None,
            decay: None,
            include_provenance: false,
            temporal_at: None,
            limit: None,
        };

        let result =
            execute_context_query(&query, &adj, &props, &vec_index, &token_counter, &interner)
                .unwrap();

        // Alice (node 1) should have relationships
        let alice_chunk = result.chunks.iter().find(|c| c.node_id == 1);
        assert!(alice_chunk.is_some());
        let alice = alice_chunk.unwrap();
        assert!(!alice.relationships.is_empty());

        // Check that relationships have proper labels
        let knows_rel = alice
            .relationships
            .iter()
            .find(|r| r.edge_label == "knows");
        assert!(knows_rel.is_some());
    }

    #[test]
    fn test_execute_context_query_empty_seeds() {
        let (adj, props, vec_index, token_counter, interner) = setup_test_stores();

        let query = ContextQuery {
            query_text: Some("test".to_string()),
            graph: "test".to_string(),
            budget: None,
            seeds: SeedStrategy::Nodes(vec!["nonexistent".to_string()]),
            max_depth: 2,
            direction: Direction::Both,
            edge_filter: None,
            decay: None,
            include_provenance: false,
            temporal_at: None,
            limit: None,
        };

        let result =
            execute_context_query(&query, &adj, &props, &vec_index, &token_counter, &interner)
                .unwrap();

        // No seeds found, so no results
        assert_eq!(result.nodes_considered, 0);
        assert_eq!(result.nodes_included, 0);
    }

    #[test]
    fn test_build_content() {
        let v_name = Value::String(CompactString::from("Alice"));
        let v_desc = Value::String(CompactString::from("A developer"));
        let v_internal = Value::String(CompactString::from("skip"));
        let v_age = Value::Int(30);

        let props: Vec<(&str, &Value)> = vec![
            ("name", &v_name),
            ("description", &v_desc),
            ("_internal", &v_internal),
            ("age", &v_age),
        ];

        let content = build_content(&props);
        assert!(content.contains("name: Alice"));
        assert!(content.contains("description: A developer"));
        assert!(!content.contains("_internal"));
        assert!(!content.contains("age")); // Int, not string
    }
}
