//! Query executor that runs context queries against the graph stores.

use std::time::Instant;

use weav_core::error::WeavError;
use weav_core::shard::StringInterner;
use weav_core::types::{BiTemporal, Direction, NodeId, Provenance, Timestamp};

use weav_graph::adjacency::AdjacencyStore;
use weav_graph::properties::PropertyStore;
use weav_graph::traversal::flow_score;
use weav_vector::index::VectorIndex;
use weav_vector::tokens::TokenCounter;

use crate::budget::enforce_budget;
use crate::parser::{ContextQuery, SeedStrategy, SortDirection, SortField};

// ─── Result types ───────────────────────────────────────────────────────────

/// Information about a detected conflict between two nodes.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConflictInfo {
    pub node_a: NodeId,
    pub node_b: NodeId,
    pub property: String,
    pub value_a: String,
    pub value_b: String,
    pub resolution: String,
}

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
    /// Detected conflicts between nodes.
    pub conflicts: Vec<ConflictInfo>,
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
    /// Optional bi-temporal metadata for the node.
    pub temporal: Option<BiTemporal>,
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
    let plan = crate::planner::plan_context_query(query);
    let (alpha, theta) = plan.steps.iter().find_map(|step| {
        if let crate::planner::PlanStep::FlowScore { alpha, theta, .. } = step {
            Some((*alpha, *theta))
        } else {
            None
        }
    }).unwrap_or((0.5, 0.01));

    let scored_nodes = flow_score(
        adjacency,
        &seeds_with_scores,
        alpha,
        theta,
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

        // Get provenance from node properties if available
        let provenance = {
            let source_prop = properties
                .get_node_property(node_id, "provenance")
                .or_else(|| properties.get_node_property(node_id, "source"));
            source_prop.and_then(|v| {
                v.as_str().map(|s| Provenance::new(s, 1.0))
            })
        };

        // Build temporal metadata from node properties if available
        let temporal = {
            let valid_from = properties
                .get_node_property(node_id, "_valid_from")
                .and_then(|v| v.as_int())
                .map(|i| i as u64);
            let valid_until = properties
                .get_node_property(node_id, "_valid_until")
                .and_then(|v| v.as_int())
                .map(|i| i as u64);
            let tx_from = properties
                .get_node_property(node_id, "_tx_from")
                .and_then(|v| v.as_int())
                .map(|i| i as u64);
            let tx_until = properties
                .get_node_property(node_id, "_tx_until")
                .and_then(|v| v.as_int())
                .map(|i| i as u64);
            if let Some(vf) = valid_from {
                Some(BiTemporal {
                    valid_from: vf,
                    valid_until: valid_until.unwrap_or(BiTemporal::OPEN),
                    tx_from: tx_from.unwrap_or(vf),
                    tx_until: tx_until.unwrap_or(BiTemporal::OPEN),
                })
            } else {
                None
            }
        };

        chunks.push(ContextChunk {
            node_id,
            content,
            label,
            relevance_score: scored.score,
            depth: scored.depth,
            token_count,
            provenance,
            relationships: Vec::new(), // Filled in step 7
            temporal,
        });
    }

    // ── Step 3b: Detect conflicts ───────────────────────────────────────
    let mut conflicts: Vec<ConflictInfo> = Vec::new();
    {
        // Group chunks by label
        let mut label_groups: std::collections::HashMap<&str, Vec<usize>> =
            std::collections::HashMap::new();
        for (idx, chunk) in chunks.iter().enumerate() {
            label_groups
                .entry(chunk.label.as_str())
                .or_default()
                .push(idx);
        }

        // Within each group, compare property values
        for (_label, indices) in &label_groups {
            if indices.len() < 2 {
                continue;
            }
            for i in 0..indices.len() {
                for j in (i + 1)..indices.len() {
                    let node_a = chunks[indices[i]].node_id;
                    let node_b = chunks[indices[j]].node_id;

                    let props_a = properties.get_all_node_properties(node_a);
                    let props_b = properties.get_all_node_properties(node_b);

                    for (key_a, val_a) in &props_a {
                        if key_a.starts_with('_') {
                            continue;
                        }
                        for (key_b, val_b) in &props_b {
                            if key_a == key_b {
                                let str_a = format!("{val_a:?}");
                                let str_b = format!("{val_b:?}");
                                if str_a != str_b {
                                    // Resolve: node with higher score wins
                                    let resolution = if chunks[indices[i]].relevance_score
                                        >= chunks[indices[j]].relevance_score
                                    {
                                        format!(
                                            "kept node {} (higher relevance)",
                                            node_a
                                        )
                                    } else {
                                        format!(
                                            "kept node {} (higher relevance)",
                                            node_b
                                        )
                                    };
                                    conflicts.push(ConflictInfo {
                                        node_a,
                                        node_b,
                                        property: key_a.to_string(),
                                        value_a: str_a,
                                        value_b: str_b,
                                        resolution,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
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

    // ── Step 7b: Apply user-requested sort ─────────────────────────────
    if let Some(ref sort) = query.sort {
        result_chunks.sort_by(|a, b| {
            let cmp = match sort.field {
                SortField::Relevance => a
                    .relevance_score
                    .partial_cmp(&b.relevance_score)
                    .unwrap_or(std::cmp::Ordering::Equal),
                SortField::Recency => {
                    // Compare by temporal valid_from (more recent = higher)
                    let ts_a = a.temporal.map(|t| t.valid_from).unwrap_or(0);
                    let ts_b = b.temporal.map(|t| t.valid_from).unwrap_or(0);
                    ts_a.cmp(&ts_b)
                }
                SortField::Confidence => {
                    let conf_a =
                        a.provenance.as_ref().map(|p| p.confidence).unwrap_or(0.0);
                    let conf_b =
                        b.provenance.as_ref().map(|p| p.confidence).unwrap_or(0.0);
                    conf_a
                        .partial_cmp(&conf_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }
            };
            if sort.direction == SortDirection::Asc {
                cmp
            } else {
                cmp.reverse()
            }
        });
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
        conflicts,
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

    use crate::parser::{ContextQuery, SeedStrategy, SortDirection, SortField, SortOrder};

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
            sort: None,
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
            sort: None,
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
            sort: None,
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
            sort: None,
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
            sort: None,
        };

        let result =
            execute_context_query(&query, &adj, &props, &vec_index, &token_counter, &interner)
                .unwrap();

        // No seeds found, so no results
        assert_eq!(result.nodes_considered, 0);
        assert_eq!(result.nodes_included, 0);
    }

    #[test]
    fn test_execute_uses_plan_flow_score_params() {
        // Verify that execute_context_query derives alpha/theta from the planner,
        // not from hardcoded values. The planner defaults to alpha=0.5, theta=0.01
        // so we confirm the plan produces those defaults, then run the executor
        // and verify results are consistent (i.e. the plan path is exercised).
        let (adj, props, vec_index, token_counter, interner) = setup_test_stores();

        let query = ContextQuery {
            query_text: Some("plan test".to_string()),
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
            sort: None,
        };

        // Step 1: Confirm the planner produces the expected default alpha/theta
        let plan = crate::planner::plan_context_query(&query);
        let flow_params = plan.steps.iter().find_map(|step| {
            if let crate::planner::PlanStep::FlowScore { alpha, theta, max_depth } = step {
                Some((*alpha, *theta, *max_depth))
            } else {
                None
            }
        });
        assert_eq!(flow_params, Some((0.5, 0.01, 2)),
            "Planner should produce default alpha=0.5, theta=0.01, max_depth=2");

        // Step 2: Execute the query (which now reads from the plan internally)
        let result =
            execute_context_query(&query, &adj, &props, &vec_index, &token_counter, &interner)
                .unwrap();

        // The executor should produce results using the plan's parameters
        assert!(result.nodes_considered > 0, "Should find nodes via flow scoring");
        assert!(result.nodes_included > 0, "Should include scored nodes");

        // Alice (seed) should be present with a high relevance score
        let alice_chunk = result.chunks.iter().find(|c| c.node_id == 1);
        assert!(alice_chunk.is_some(), "Alice should be in results");
        let alice = alice_chunk.unwrap();
        assert!(alice.relevance_score > 0.0, "Seed node should have positive score");

        // Neighbors should also be scored (via flow propagation with plan's alpha/theta)
        let neighbor_chunks: Vec<_> = result.chunks.iter().filter(|c| c.node_id != 1).collect();
        assert!(!neighbor_chunks.is_empty(), "Flow scoring should reach neighbors");
        for chunk in &neighbor_chunks {
            assert!(chunk.relevance_score > 0.0,
                "Neighbor nodes should have positive scores from flow propagation");
            assert!(chunk.relevance_score < alice.relevance_score,
                "Neighbor scores should be lower than the seed score (alpha=0.5 decay)");
        }
    }

    #[test]
    fn test_execute_seed_strategy_both() {
        let (adj, props, mut vec_index, token_counter, interner) = setup_test_stores();

        // Add embeddings for node 1 (Alice) so vector search can find it
        vec_index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        vec_index.insert(3, &[0.0, 1.0, 0.0, 0.0]).unwrap();

        // Also set an entity_key for node 2 (Bob) so node key lookup can find it
        // (already set in setup_test_stores)

        let query = ContextQuery {
            query_text: Some("test both".to_string()),
            graph: "test".to_string(),
            budget: None,
            seeds: SeedStrategy::Both {
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                top_k: 5,
                node_keys: vec!["bob".to_string()],
            },
            max_depth: 1,
            direction: Direction::Both,
            edge_filter: None,
            decay: None,
            include_provenance: false,
            temporal_at: None,
            limit: None,
            sort: None,
        };

        let result =
            execute_context_query(&query, &adj, &props, &vec_index, &token_counter, &interner)
                .unwrap();

        assert!(result.nodes_considered > 0);
        assert!(result.nodes_included > 0);

        // Both Alice (from vector search) and Bob (from node key lookup) should be in results
        let node_ids: Vec<u64> = result.chunks.iter().map(|c| c.node_id).collect();
        assert!(node_ids.contains(&1), "Alice should be found via vector search");
        assert!(node_ids.contains(&2), "Bob should be found via node key lookup");
    }

    #[test]
    fn test_execute_empty_seeds() {
        let (adj, props, vec_index, token_counter, interner) = setup_test_stores();

        let query = ContextQuery {
            query_text: Some("test".to_string()),
            graph: "test".to_string(),
            budget: None,
            seeds: SeedStrategy::Nodes(vec![]),
            max_depth: 2,
            direction: Direction::Both,
            edge_filter: None,
            decay: None,
            include_provenance: false,
            temporal_at: None,
            limit: None,
            sort: None,
        };

        let result =
            execute_context_query(&query, &adj, &props, &vec_index, &token_counter, &interner)
                .unwrap();

        // Empty seed list means no seeds found, so no results
        assert_eq!(result.nodes_considered, 0);
        assert_eq!(result.nodes_included, 0);
        assert!(result.chunks.is_empty());
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

    #[test]
    fn test_conflict_detection() {
        // Two "person" nodes with the same property key ("name") but different values
        // should produce a ConflictInfo.
        let (adj, mut props, vec_index, token_counter, interner) = setup_test_stores();

        // Node 1 (Alice) and Node 2 (Bob) are both label "person" and both have
        // "name" and "description" properties with different values.
        // Additionally give them a shared property with the SAME value to confirm
        // no false positives.
        props.set_node_property(
            1,
            "role",
            Value::String(CompactString::from("engineer")),
        );
        props.set_node_property(
            2,
            "role",
            Value::String(CompactString::from("engineer")),
        );

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
            sort: None,
        };

        let result =
            execute_context_query(&query, &adj, &props, &vec_index, &token_counter, &interner)
                .unwrap();

        // We should have conflicts for "name" and "description" between Alice and Bob
        assert!(
            !result.conflicts.is_empty(),
            "Should detect conflicts between two person nodes"
        );

        // Find the "name" conflict between person nodes (1 and 2)
        let name_conflict = result
            .conflicts
            .iter()
            .find(|c| {
                c.property == "name"
                    && ((c.node_a == 1 && c.node_b == 2) || (c.node_a == 2 && c.node_b == 1))
            });
        assert!(name_conflict.is_some(), "Should detect name conflict between person nodes 1 and 2");
        let nc = name_conflict.unwrap();
        // Resolution should mention a node
        assert!(
            nc.resolution.contains("kept node"),
            "Resolution should use LastWriteWins strategy"
        );

        // "role" has the same value on both, so NO conflict for "role"
        let role_conflict = result
            .conflicts
            .iter()
            .find(|c| c.property == "role");
        assert!(
            role_conflict.is_none(),
            "Should not report conflict when values are identical"
        );
    }

    #[test]
    fn test_sort_order_applied() {
        // Create nodes with temporal metadata to test sorting by recency.
        let (adj, mut props, vec_index, token_counter, interner) = setup_test_stores();

        // Give nodes valid_from timestamps: Alice=1000, Bob=3000, Rust=2000
        props.set_node_property(1, "_valid_from", Value::Int(1000));
        props.set_node_property(2, "_valid_from", Value::Int(3000));
        props.set_node_property(3, "_valid_from", Value::Int(2000));

        // Query with SORT BY RECENCY ASC (oldest first)
        let query_asc = ContextQuery {
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
            sort: Some(SortOrder {
                field: SortField::Recency,
                direction: SortDirection::Asc,
            }),
        };

        let result_asc =
            execute_context_query(&query_asc, &adj, &props, &vec_index, &token_counter, &interner)
                .unwrap();

        // Should be ordered by valid_from ascending
        assert!(
            result_asc.chunks.len() >= 2,
            "Need at least 2 chunks to verify sorting"
        );
        for i in 1..result_asc.chunks.len() {
            let ts_prev = result_asc.chunks[i - 1]
                .temporal
                .map(|t| t.valid_from)
                .unwrap_or(0);
            let ts_curr = result_asc.chunks[i]
                .temporal
                .map(|t| t.valid_from)
                .unwrap_or(0);
            assert!(
                ts_prev <= ts_curr,
                "ASC sort: chunk {} (valid_from={}) should be <= chunk {} (valid_from={})",
                i - 1,
                ts_prev,
                i,
                ts_curr
            );
        }

        // Query with SORT BY RECENCY DESC (most recent first)
        let query_desc = ContextQuery {
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
            sort: Some(SortOrder {
                field: SortField::Recency,
                direction: SortDirection::Desc,
            }),
        };

        let result_desc =
            execute_context_query(&query_desc, &adj, &props, &vec_index, &token_counter, &interner)
                .unwrap();

        // Should be ordered by valid_from descending
        assert!(
            result_desc.chunks.len() >= 2,
            "Need at least 2 chunks to verify sorting"
        );
        for i in 1..result_desc.chunks.len() {
            let ts_prev = result_desc.chunks[i - 1]
                .temporal
                .map(|t| t.valid_from)
                .unwrap_or(0);
            let ts_curr = result_desc.chunks[i]
                .temporal
                .map(|t| t.valid_from)
                .unwrap_or(0);
            assert!(
                ts_prev >= ts_curr,
                "DESC sort: chunk {} (valid_from={}) should be >= chunk {} (valid_from={})",
                i - 1,
                ts_prev,
                i,
                ts_curr
            );
        }

        // Query with SORT BY RELEVANCE DESC (default-like behavior)
        let query_rel = ContextQuery {
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
            sort: Some(SortOrder {
                field: SortField::Relevance,
                direction: SortDirection::Desc,
            }),
        };

        let result_rel =
            execute_context_query(&query_rel, &adj, &props, &vec_index, &token_counter, &interner)
                .unwrap();

        // Should be ordered by relevance_score descending
        for i in 1..result_rel.chunks.len() {
            assert!(
                result_rel.chunks[i - 1].relevance_score >= result_rel.chunks[i].relevance_score,
                "DESC relevance sort: chunk {} (score={}) should be >= chunk {} (score={})",
                i - 1,
                result_rel.chunks[i - 1].relevance_score,
                i,
                result_rel.chunks[i].relevance_score
            );
        }
    }
}
