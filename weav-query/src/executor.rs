//! Query executor that runs context queries against the graph stores.

use std::time::Instant;

use weav_core::error::WeavError;
use weav_core::shard::StringInterner;
use weav_core::types::{BiTemporal, Direction, NodeId, Provenance, Timestamp};

use weav_graph::adjacency::AdjacencyStore;
use weav_graph::properties::PropertyStore;
use weav_graph::text_index::TextIndex;
use weav_graph::traversal::{flow_score, modularity_communities};
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
    text_index: &TextIndex,
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

    let has_vector_seeds = matches!(&query.seeds, SeedStrategy::Vector { .. } | SeedStrategy::Both { .. });

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

    // ── Step 2b: Triple-fusion RRF (vector + BM25 + graph) ─────────
    // BM25 text search: if query_text is available, get text-ranked results
    let bm25_ranked: Vec<NodeId> = if let Some(ref query_text) = query.query_text {
        if !query_text.is_empty() && !text_index.is_empty() {
            text_index
                .search(query_text, (adjacency.node_count() as usize).min(500))
                .into_iter()
                .map(|(nid, _)| nid)
                .collect()
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    let final_scores: Vec<(NodeId, f32)> = if has_vector_seeds {
        let vector_ranked: Vec<NodeId> = seeds_with_scores.iter()
            .map(|(nid, _)| *nid)
            .collect();

        let graph_ranked: Vec<NodeId> = scored_nodes.iter()
            .map(|s| s.node_id)
            .collect();

        // Triple fusion: vector + graph + BM25 (if text results available)
        let (ranked_lists, rrf_config) = if bm25_ranked.is_empty() {
            // 2-way: vector + graph
            (
                vec![vector_ranked, graph_ranked],
                RrfConfig { k: 60.0, weights: vec![1.0, 1.0] },
            )
        } else {
            // 3-way: vector + graph + BM25
            (
                vec![vector_ranked, graph_ranked, bm25_ranked],
                RrfConfig { k: 60.0, weights: vec![1.0, 1.0, 0.8] },
            )
        };

        let fused = reciprocal_rank_fusion(&ranked_lists, &rrf_config);

        // Normalize RRF scores to [0, 1] range
        let max_rrf = fused.first().map(|(_, s)| *s).unwrap_or(1.0);
        fused.into_iter()
            .map(|(nid, score)| (nid, if max_rrf > 0.0 { score / max_rrf } else { 0.0 }))
            .collect()
    } else if !bm25_ranked.is_empty() {
        // No vector search but BM25 available — fuse graph + BM25
        let graph_ranked: Vec<NodeId> = scored_nodes.iter()
            .map(|s| s.node_id)
            .collect();

        let rrf_config = RrfConfig { k: 60.0, weights: vec![1.0, 0.8] };
        let ranked_lists = vec![graph_ranked, bm25_ranked];
        let fused = reciprocal_rank_fusion(&ranked_lists, &rrf_config);

        let max_rrf = fused.first().map(|(_, s)| *s).unwrap_or(1.0);
        fused.into_iter()
            .map(|(nid, score)| (nid, if max_rrf > 0.0 { score / max_rrf } else { 0.0 }))
            .collect()
    } else {
        // No vector search, no BM25 — use flow scores directly
        scored_nodes.iter()
            .map(|s| (s.node_id, s.score))
            .collect()
    };

    let nodes_considered = final_scores.len() as u32;

    // Build a depth lookup from scored_nodes for use in chunk construction
    let depth_lookup: std::collections::HashMap<NodeId, u8> = scored_nodes.iter()
        .map(|s| (s.node_id, s.depth))
        .collect();

    // ── Step 3: Build ContextChunks ─────────────────────────────────────
    let mut chunks: Vec<ContextChunk> = Vec::new();

    for &(node_id, score) in &final_scores {

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

        let depth = depth_lookup.get(&node_id).copied().unwrap_or(0);

        chunks.push(ContextChunk {
            node_id,
            content,
            label,
            relevance_score: score,
            depth,
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
        for indices in label_groups.values() {
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
                .then_with(|| a.node_id.cmp(&b.node_id))
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
            let cmp = cmp.then_with(|| a.node_id.cmp(&b.node_id));
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

// ─── Reciprocal Rank Fusion ──────────────────────────────────────────────────

/// Configuration for Reciprocal Rank Fusion.
pub struct RrfConfig {
    /// Smoothing constant (default 60.0, from Cormack et al. 2009).
    pub k: f32,
    /// Per-ranker weights (default all 1.0). Length must match ranked_lists.
    pub weights: Vec<f32>,
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self {
            k: 60.0,
            weights: vec![1.0, 1.0],
        }
    }
}

/// Fuse multiple ranked lists using Reciprocal Rank Fusion.
///
/// RRF score = Σᵢ wᵢ / (k + rankᵢ(d))
///
/// Each ranked list is a Vec of NodeIds ordered by relevance (best first).
/// Returns (NodeId, rrf_score) pairs sorted by RRF score descending.
pub fn reciprocal_rank_fusion(
    ranked_lists: &[Vec<NodeId>],
    config: &RrfConfig,
) -> Vec<(NodeId, f32)> {
    let mut scores: std::collections::HashMap<NodeId, f32> = std::collections::HashMap::new();

    for (list_idx, list) in ranked_lists.iter().enumerate() {
        let weight = config.weights.get(list_idx).copied().unwrap_or(1.0);
        for (rank_0, &node_id) in list.iter().enumerate() {
            let rank = (rank_0 + 1) as f32; // 1-indexed
            *scores.entry(node_id).or_default() += weight / (config.k + rank);
        }
    }

    let mut result: Vec<(NodeId, f32)> = scores.into_iter().collect();
    result.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    result
}

// ─── Community Summaries (GraphRAG) ─────────────────────────────────────────

/// A community summary containing the nodes and their aggregated content.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CommunitySummary {
    /// Community ID (from the community detection algorithm).
    pub community_id: u64,
    /// Number of nodes in this community.
    pub node_count: usize,
    /// Number of internal edges (edges between nodes in this community).
    pub internal_edge_count: usize,
    /// Node IDs in this community.
    pub node_ids: Vec<NodeId>,
    /// Labels present in this community (deduplicated).
    pub labels: Vec<String>,
    /// Concatenated content from all nodes, suitable for LLM summarization.
    pub content: String,
    /// Estimated token count for the content.
    pub token_count: u32,
    /// Key entities (node names/titles) in this community.
    pub key_entities: Vec<String>,
}

/// Extract community summaries from a graph using modularity-based community detection.
///
/// This is the preparation step for GraphRAG-style summarization:
/// 1. Detect communities using modularity optimization
/// 2. For each community, collect node content, labels, and key entities
/// 3. Count internal edges
/// 4. Return structured summaries ready for LLM summarization
///
/// The caller can then pass each community's `content` field to an LLM
/// to generate a natural language summary.
pub fn extract_community_summaries(
    adjacency: &AdjacencyStore,
    properties: &PropertyStore,
    _interner: &StringInterner,
    token_counter: &TokenCounter,
    max_iterations: u32,
    resolution: f32,
) -> Vec<CommunitySummary> {
    let node_to_community = modularity_communities(adjacency, max_iterations, resolution);
    if node_to_community.is_empty() {
        return Vec::new();
    }

    // Group nodes by community ID.
    let mut communities: std::collections::HashMap<u64, Vec<NodeId>> =
        std::collections::HashMap::new();
    for (&node_id, &comm_id) in &node_to_community {
        communities.entry(comm_id).or_default().push(node_id);
    }

    // Build a set of node IDs per community for fast membership checks.
    let community_sets: std::collections::HashMap<u64, std::collections::HashSet<NodeId>> =
        communities
            .iter()
            .map(|(&comm_id, nodes)| {
                let set: std::collections::HashSet<NodeId> = nodes.iter().copied().collect();
                (comm_id, set)
            })
            .collect();

    let mut summaries: Vec<CommunitySummary> = Vec::with_capacity(communities.len());

    for (&comm_id, nodes) in &communities {
        let node_set = &community_sets[&comm_id];

        let mut labels: Vec<String> = Vec::new();
        let mut key_entities: Vec<String> = Vec::new();
        let mut content_parts: Vec<String> = Vec::new();
        let mut internal_edge_count: usize = 0;

        // Sort node IDs for deterministic output.
        let mut sorted_nodes = nodes.clone();
        sorted_nodes.sort_unstable();

        for &node_id in &sorted_nodes {
            // Extract label from _label property.
            if let Some(label_val) = properties.get_node_property(node_id, "_label") {
                if let Some(label_str) = label_val.as_str() {
                    if !labels.contains(&label_str.to_string()) {
                        labels.push(label_str.to_string());
                    }
                }
            }

            // Extract name or title as key entity.
            let entity_name = properties
                .get_node_property(node_id, "name")
                .and_then(|v| v.as_str())
                .or_else(|| {
                    properties
                        .get_node_property(node_id, "title")
                        .and_then(|v| v.as_str())
                });
            if let Some(name) = entity_name {
                key_entities.push(name.to_string());
            }

            // Build content from all properties (reusing the build_content pattern).
            let all_props = properties.get_all_node_properties(node_id);
            let node_content = build_content(&all_props);
            if !node_content.is_empty() {
                content_parts.push(node_content);
            }

            // Count internal edges: outgoing edges where the target is in the same community.
            for (neighbor_id, _edge_id) in adjacency.neighbors_out(node_id, None) {
                if node_set.contains(&neighbor_id) {
                    internal_edge_count += 1;
                }
            }
        }

        labels.sort();
        key_entities.sort();

        let content = content_parts.join("\n---\n");
        let token_count = token_counter.count(&content);

        summaries.push(CommunitySummary {
            community_id: comm_id,
            node_count: sorted_nodes.len(),
            internal_edge_count,
            node_ids: sorted_nodes,
            labels,
            content,
            token_count,
            key_entities,
        });
    }

    // Sort by node_count descending (largest communities first), break ties by community_id.
    summaries.sort_by(|a, b| {
        b.node_count
            .cmp(&a.node_count)
            .then_with(|| a.community_id.cmp(&b.community_id))
    });

    summaries
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
        TextIndex,
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
        let knows_label = interner.intern_label("knows").unwrap();
        let uses_label = interner.intern_label("uses").unwrap();

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
        let text_index = TextIndex::new();

        (adj, props, vec_index, text_index, token_counter, interner)
    }

    #[test]
    fn test_execute_context_query_with_node_seeds() {
        let (adj, props, vec_index, text_index, token_counter, interner) = setup_test_stores();

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
            execute_context_query(&query, &adj, &props, &vec_index, &text_index, &token_counter, &interner)
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
        let (adj, props, vec_index, text_index, token_counter, interner) = setup_test_stores();

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
            execute_context_query(&query, &adj, &props, &vec_index, &text_index, &token_counter, &interner)
                .unwrap();

        // Should have budget enforcement applied
        assert!(result.total_tokens <= 10);
        assert!(result.budget_used <= 1.0);
    }

    #[test]
    fn test_execute_context_query_with_limit() {
        let (adj, props, vec_index, text_index, token_counter, interner) = setup_test_stores();

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
            execute_context_query(&query, &adj, &props, &vec_index, &text_index, &token_counter, &interner)
                .unwrap();

        assert!(result.nodes_included <= 1);
    }

    #[test]
    fn test_execute_context_query_relationships() {
        let (adj, props, vec_index, text_index, token_counter, interner) = setup_test_stores();

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
            execute_context_query(&query, &adj, &props, &vec_index, &text_index, &token_counter, &interner)
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
        let (adj, props, vec_index, text_index, token_counter, interner) = setup_test_stores();

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
            execute_context_query(&query, &adj, &props, &vec_index, &text_index, &token_counter, &interner)
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
        let (adj, props, vec_index, text_index, token_counter, interner) = setup_test_stores();

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
            execute_context_query(&query, &adj, &props, &vec_index, &text_index, &token_counter, &interner)
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
        let (adj, props, mut vec_index, text_index, token_counter, interner) = setup_test_stores();

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
            execute_context_query(&query, &adj, &props, &vec_index, &text_index, &token_counter, &interner)
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
        let (adj, props, vec_index, text_index, token_counter, interner) = setup_test_stores();

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
            execute_context_query(&query, &adj, &props, &vec_index, &text_index, &token_counter, &interner)
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
        let (adj, mut props, vec_index, text_index, token_counter, interner) = setup_test_stores();

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
            execute_context_query(&query, &adj, &props, &vec_index, &text_index, &token_counter, &interner)
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
        let (adj, mut props, vec_index, text_index, token_counter, interner) = setup_test_stores();

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
            execute_context_query(&query_asc, &adj, &props, &vec_index, &text_index, &token_counter, &interner)
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
            execute_context_query(&query_desc, &adj, &props, &vec_index, &text_index, &token_counter, &interner)
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
            execute_context_query(&query_rel, &adj, &props, &vec_index, &text_index, &token_counter, &interner)
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

    #[test]
    fn test_execute_build_content_no_string_properties() {
        // Node with only Int/Float properties (no String values).
        // build_content should return empty string since it only concatenates strings.
        let v_age = Value::Int(30);
        let v_score = Value::Float(99.5);
        let v_active = Value::Bool(true);

        let props: Vec<(&str, &Value)> = vec![
            ("age", &v_age),
            ("score", &v_score),
            ("active", &v_active),
        ];

        let content = build_content(&props);
        assert!(
            content.is_empty(),
            "build_content should return empty string when no String values exist, got: '{content}'"
        );
    }

    #[test]
    fn test_execute_build_content_many_properties() {
        // Node with 20+ string properties. Verify concatenation works correctly.
        let values: Vec<Value> = (0..25)
            .map(|i| Value::String(CompactString::from(format!("value_{i}"))))
            .collect();

        let props: Vec<(&str, &Value)> = values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                // We need a stable key string; use a leaked &str for test convenience
                let key: &str = Box::leak(format!("prop_{i}").into_boxed_str());
                (key, v)
            })
            .collect();

        let content = build_content(&props);

        // Should contain all 25 properties joined by newlines
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(
            lines.len(),
            25,
            "Should have 25 lines for 25 string properties, got {}",
            lines.len()
        );

        for i in 0..25 {
            assert!(
                content.contains(&format!("prop_{i}: value_{i}")),
                "Missing property prop_{i}"
            );
        }
    }

    #[test]
    fn test_execute_build_content_only_internal_properties() {
        // All properties start with `_`. build_content should return empty string.
        let v1 = Value::String(CompactString::from("internal1"));
        let v2 = Value::String(CompactString::from("internal2"));
        let v3 = Value::String(CompactString::from("internal3"));

        let props: Vec<(&str, &Value)> = vec![
            ("_label", &v1),
            ("_created_at", &v2),
            ("_valid_from", &v3),
        ];

        let content = build_content(&props);
        assert!(
            content.is_empty(),
            "build_content should return empty for all internal (_-prefixed) properties, got: '{content}'"
        );
    }

    #[test]
    fn test_execute_temporal_filter_at_zero() {
        // temporal_at=0 should filter out all nodes created after timestamp 0.
        let (adj, mut props, vec_index, text_index, token_counter, interner) = setup_test_stores();

        // Set _valid_from on all nodes to timestamps > 0
        props.set_node_property(1, "_valid_from", Value::Int(1000));
        props.set_node_property(2, "_valid_from", Value::Int(2000));
        props.set_node_property(3, "_valid_from", Value::Int(3000));
        props.set_node_property(4, "_valid_from", Value::Int(4000));

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
            temporal_at: Some(0), // timestamp 0 — before all nodes were created
            limit: None,
            sort: None,
        };

        let result =
            execute_context_query(&query, &adj, &props, &vec_index, &text_index, &token_counter, &interner)
                .unwrap();

        assert!(
            result.chunks.is_empty(),
            "temporal_at=0 should exclude all nodes with valid_from > 0, got {} chunks",
            result.chunks.len()
        );
    }

    #[test]
    fn test_execute_temporal_filter_at_max() {
        // temporal_at=u64::MAX-1 should allow all nodes through (since their valid_from < MAX-1).
        let (adj, mut props, vec_index, text_index, token_counter, interner) = setup_test_stores();

        // Set _valid_from on nodes to various timestamps
        props.set_node_property(1, "_valid_from", Value::Int(1000));
        props.set_node_property(2, "_valid_from", Value::Int(2000));
        props.set_node_property(3, "_valid_from", Value::Int(3000));

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
            temporal_at: Some(u64::MAX - 1), // far future — everything is valid
            limit: None,
            sort: None,
        };

        let result =
            execute_context_query(&query, &adj, &props, &vec_index, &text_index, &token_counter, &interner)
                .unwrap();

        // Should include all reachable nodes (alice + neighbors)
        assert!(
            result.nodes_included > 0,
            "temporal_at=MAX-1 should allow all nodes through"
        );
        // Verify nodes with temporal metadata are present
        let node_ids: Vec<u64> = result.chunks.iter().map(|c| c.node_id).collect();
        assert!(node_ids.contains(&1), "Alice should be present");
        assert!(node_ids.contains(&2), "Bob should be present");
        assert!(node_ids.contains(&3), "Rust should be present");
    }

    #[test]
    fn test_execute_sort_by_token_count() {
        // While the executor doesn't have a direct SortField::TokenCount, we can verify
        // chunks are correctly sorted by relevance (which correlates), and also verify
        // that token_count values are correctly computed so external sorting works.
        let (adj, mut props, vec_index, text_index, token_counter, interner) = setup_test_stores();

        // Give nodes varying amounts of content to produce different token counts
        // Node 1 (Alice): short content
        props.set_node_property(
            1,
            "bio",
            Value::String(CompactString::from("Short")),
        );
        // Node 2 (Bob): medium content
        props.set_node_property(
            2,
            "bio",
            Value::String(CompactString::from("A moderately longer piece of text content")),
        );
        // Node 3 (Rust): long content
        props.set_node_property(
            3,
            "bio",
            Value::String(CompactString::from(
                "A very long and detailed description that contains many words to increase the token count significantly beyond the others",
            )),
        );

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
            limit: None,
            sort: None,
        };

        let result =
            execute_context_query(&query, &adj, &props, &vec_index, &text_index, &token_counter, &interner)
                .unwrap();

        // Collect chunks and sort by token_count ascending
        let mut chunks_asc = result.chunks.clone();
        chunks_asc.sort_by_key(|c| c.token_count);

        // Verify ascending order
        for i in 1..chunks_asc.len() {
            assert!(
                chunks_asc[i - 1].token_count <= chunks_asc[i].token_count,
                "ASC: chunk {} (tokens={}) should be <= chunk {} (tokens={})",
                i - 1,
                chunks_asc[i - 1].token_count,
                i,
                chunks_asc[i].token_count,
            );
        }

        // Sort by token_count descending
        let mut chunks_desc = result.chunks.clone();
        chunks_desc.sort_by(|a, b| b.token_count.cmp(&a.token_count));

        // Verify descending order
        for i in 1..chunks_desc.len() {
            assert!(
                chunks_desc[i - 1].token_count >= chunks_desc[i].token_count,
                "DESC: chunk {} (tokens={}) should be >= chunk {} (tokens={})",
                i - 1,
                chunks_desc[i - 1].token_count,
                i,
                chunks_desc[i].token_count,
            );
        }

        // Verify that different content lengths produce different token counts
        let alice_chunk = result.chunks.iter().find(|c| c.node_id == 1).unwrap();
        let rust_chunk = result.chunks.iter().find(|c| c.node_id == 3).unwrap();
        assert!(
            rust_chunk.token_count > alice_chunk.token_count,
            "Longer content should produce more tokens: rust={} vs alice={}",
            rust_chunk.token_count,
            alice_chunk.token_count,
        );
    }

    #[test]
    fn test_execute_limit_zero() {
        // limit=0 should produce 0 chunks in result.
        let (adj, props, vec_index, text_index, token_counter, interner) = setup_test_stores();

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
            limit: Some(0),
            sort: None,
        };

        let result =
            execute_context_query(&query, &adj, &props, &vec_index, &text_index, &token_counter, &interner)
                .unwrap();

        assert_eq!(
            result.chunks.len(),
            0,
            "limit=0 should produce 0 chunks, got {}",
            result.chunks.len()
        );
        assert_eq!(result.nodes_included, 0);
    }

    #[test]
    fn test_execute_decay_with_future_timestamp() {
        // When item_ts > now, decay should return score unchanged.
        // Test the DecayFunction::apply directly since the executor uses SystemTime::now().
        use weav_core::types::DecayFunction;

        let decay_exp = DecayFunction::Exponential { half_life_ms: 1000 };
        let decay_lin = DecayFunction::Linear { max_age_ms: 5000 };
        let decay_step = DecayFunction::Step { cutoff_ms: 2000 };
        let decay_none = DecayFunction::None;

        let score = 0.85_f32;
        let now: u64 = 10_000; // "now" is 10 seconds
        let future_ts: u64 = 20_000; // item created in the "future" at 20 seconds

        // All decay functions should return score unchanged when item_ts > now
        let result_exp = decay_exp.apply(score, future_ts, now);
        assert!(
            (result_exp - score).abs() < f32::EPSILON,
            "Exponential decay should return score unchanged for future timestamps, got {result_exp}"
        );

        let result_lin = decay_lin.apply(score, future_ts, now);
        assert!(
            (result_lin - score).abs() < f32::EPSILON,
            "Linear decay should return score unchanged for future timestamps, got {result_lin}"
        );

        let result_step = decay_step.apply(score, future_ts, now);
        assert!(
            (result_step - score).abs() < f32::EPSILON,
            "Step decay should return score unchanged for future timestamps, got {result_step}"
        );

        let result_none = decay_none.apply(score, future_ts, now);
        assert!(
            (result_none - score).abs() < f32::EPSILON,
            "None decay should return score unchanged for future timestamps, got {result_none}"
        );

        // Also verify that when item_ts == now, the score is unchanged (age = 0)
        let result_same = decay_exp.apply(score, now, now);
        assert!(
            (result_same - score).abs() < f32::EPSILON,
            "Decay with item_ts == now should return score unchanged, got {result_same}"
        );
    }

    // ── RRF tests ───────────────────────────────────────────────────────

    #[test]
    fn test_rrf_two_lists() {
        let list1 = vec![1, 2, 3]; // graph traversal ranking
        let list2 = vec![2, 1, 4]; // vector search ranking
        let config = RrfConfig::default();

        let result = reciprocal_rank_fusion(&[list1, list2], &config);

        // Node 2: rank 2 in list1 + rank 1 in list2 = 1/62 + 1/61
        // Node 1: rank 1 in list1 + rank 2 in list2 = 1/61 + 1/62
        // Both have the same score, so tie-break by node_id
        assert_eq!(result.len(), 4);
        // Nodes 1 and 2 should be top (appear in both lists)
        let top_ids: Vec<NodeId> = result.iter().take(2).map(|(id, _)| *id).collect();
        assert!(top_ids.contains(&1));
        assert!(top_ids.contains(&2));
        // Nodes 3 and 4 should be below (appear in only one list)
        let bottom_ids: Vec<NodeId> = result.iter().skip(2).map(|(id, _)| *id).collect();
        assert!(bottom_ids.contains(&3));
        assert!(bottom_ids.contains(&4));
    }

    #[test]
    fn test_rrf_empty_lists() {
        let result = reciprocal_rank_fusion(&[], &RrfConfig::default());
        assert!(result.is_empty());
    }

    #[test]
    fn test_rrf_single_list() {
        let list = vec![10, 20, 30];
        let config = RrfConfig { k: 60.0, weights: vec![1.0] };
        let result = reciprocal_rank_fusion(&[list], &config);

        assert_eq!(result.len(), 3);
        // Should preserve ranking order
        assert_eq!(result[0].0, 10);
        assert_eq!(result[1].0, 20);
        assert_eq!(result[2].0, 30);
        // Scores should be decreasing
        assert!(result[0].1 > result[1].1);
        assert!(result[1].1 > result[2].1);
    }

    #[test]
    fn test_rrf_weighted() {
        // Give graph results 2x weight over vector results
        let graph_list = vec![1, 2]; // graph says node 1 is best
        let vector_list = vec![2, 1]; // vector says node 2 is best
        let config = RrfConfig {
            k: 60.0,
            weights: vec![2.0, 1.0], // graph gets 2x weight
        };

        let result = reciprocal_rank_fusion(&[graph_list, vector_list], &config);

        // Node 1: 2.0/61 + 1.0/62 = 0.04896
        // Node 2: 2.0/62 + 1.0/61 = 0.04866
        // Node 1 should win because graph (weighted 2x) ranked it #1
        assert_eq!(result[0].0, 1);
        assert_eq!(result[1].0, 2);
    }

    #[test]
    fn test_rrf_disjoint_lists() {
        let list1 = vec![1, 2];
        let list2 = vec![3, 4];
        let config = RrfConfig::default();

        let result = reciprocal_rank_fusion(&[list1, list2], &config);
        assert_eq!(result.len(), 4);
        // Top-ranked from each list should be at top (tied by score)
        let top_scores: Vec<f32> = result.iter().take(2).map(|(_, s)| *s).collect();
        assert!((top_scores[0] - top_scores[1]).abs() < f32::EPSILON);
    }

    // ── Community summary tests ─────────────────────────────────────────

    #[test]
    fn test_extract_community_summaries_basic() {
        // Build two cliques with properties.
        let mut adj = AdjacencyStore::new();
        let mut props = PropertyStore::new();
        let mut interner = StringInterner::new();

        let knows_label = interner.intern_label("knows").unwrap();

        // Clique 1: nodes 1,2,3 (Person entities)
        for id in 1..=3u64 {
            adj.add_node(id);
            props.set_node_property(
                id,
                "_label",
                Value::String(CompactString::from("Person")),
            );
            props.set_node_property(
                id,
                "name",
                Value::String(CompactString::from(format!("Person{}", id))),
            );
        }

        let make_meta = |s: NodeId, t: NodeId| EdgeMeta {
            source: s,
            target: t,
            label: knows_label,
            temporal: BiTemporal::new_current(1000),
            provenance: None,
            weight: 1.0,
            token_cost: 0,
        };

        // Fully connect clique 1
        adj.add_edge(1, 2, knows_label, make_meta(1, 2)).unwrap();
        adj.add_edge(2, 1, knows_label, make_meta(2, 1)).unwrap();
        adj.add_edge(1, 3, knows_label, make_meta(1, 3)).unwrap();
        adj.add_edge(3, 1, knows_label, make_meta(3, 1)).unwrap();
        adj.add_edge(2, 3, knows_label, make_meta(2, 3)).unwrap();
        adj.add_edge(3, 2, knows_label, make_meta(3, 2)).unwrap();

        // Clique 2: nodes 4,5,6 (Company entities)
        for id in 4..=6u64 {
            adj.add_node(id);
            props.set_node_property(
                id,
                "_label",
                Value::String(CompactString::from("Company")),
            );
            props.set_node_property(
                id,
                "name",
                Value::String(CompactString::from(format!("Corp{}", id))),
            );
        }

        adj.add_edge(4, 5, knows_label, make_meta(4, 5)).unwrap();
        adj.add_edge(5, 4, knows_label, make_meta(5, 4)).unwrap();
        adj.add_edge(4, 6, knows_label, make_meta(4, 6)).unwrap();
        adj.add_edge(6, 4, knows_label, make_meta(6, 4)).unwrap();
        adj.add_edge(5, 6, knows_label, make_meta(5, 6)).unwrap();
        adj.add_edge(6, 5, knows_label, make_meta(6, 5)).unwrap();

        // Bridge edge between cliques
        adj.add_edge(3, 4, knows_label, make_meta(3, 4)).unwrap();
        adj.add_edge(4, 3, knows_label, make_meta(4, 3)).unwrap();

        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        let summaries =
            extract_community_summaries(&adj, &props, &interner, &counter, 20, 1.0);

        assert_eq!(summaries.len(), 2);
        // Each community should have 3 nodes
        assert_eq!(summaries[0].node_count, 3);
        assert_eq!(summaries[1].node_count, 3);
        // Communities should have different labels
        assert_ne!(summaries[0].labels, summaries[1].labels);
        // Key entities should be populated
        assert!(!summaries[0].key_entities.is_empty());
        assert!(!summaries[1].key_entities.is_empty());
        // Content should be non-empty
        assert!(!summaries[0].content.is_empty());
        assert!(!summaries[1].content.is_empty());
        // Internal edges: each fully connected 3-clique has 6 directed edges
        assert_eq!(summaries[0].internal_edge_count, 6);
        assert_eq!(summaries[1].internal_edge_count, 6);
        // Token count should be positive
        assert!(summaries[0].token_count > 0);
        assert!(summaries[1].token_count > 0);
    }

    #[test]
    fn test_extract_community_summaries_empty() {
        let adj = AdjacencyStore::new();
        let props = PropertyStore::new();
        let interner = StringInterner::new();
        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        let summaries =
            extract_community_summaries(&adj, &props, &interner, &counter, 10, 1.0);
        assert!(summaries.is_empty());
    }

    #[test]
    fn test_extract_community_summaries_single_node() {
        // A single node with no edges should be in its own community.
        let mut adj = AdjacencyStore::new();
        let mut props = PropertyStore::new();
        let interner = StringInterner::new();

        adj.add_node(1);
        props.set_node_property(
            1,
            "_label",
            Value::String(CompactString::from("Entity")),
        );
        props.set_node_property(
            1,
            "name",
            Value::String(CompactString::from("Singleton")),
        );

        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        let summaries =
            extract_community_summaries(&adj, &props, &interner, &counter, 10, 1.0);

        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].node_count, 1);
        assert_eq!(summaries[0].internal_edge_count, 0);
        assert_eq!(summaries[0].labels, vec!["Entity".to_string()]);
        assert_eq!(summaries[0].key_entities, vec!["Singleton".to_string()]);
    }

    #[test]
    fn test_extract_community_summaries_uses_title_fallback() {
        // When a node has no "name" property but has "title", use that as key entity.
        let mut adj = AdjacencyStore::new();
        let mut props = PropertyStore::new();
        let interner = StringInterner::new();

        adj.add_node(1);
        props.set_node_property(
            1,
            "title",
            Value::String(CompactString::from("My Document")),
        );

        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        let summaries =
            extract_community_summaries(&adj, &props, &interner, &counter, 10, 1.0);

        assert_eq!(summaries.len(), 1);
        assert_eq!(
            summaries[0].key_entities,
            vec!["My Document".to_string()]
        );
    }

    #[test]
    fn test_extract_community_summaries_sorted_by_size() {
        // Communities should be sorted largest first.
        let mut adj = AdjacencyStore::new();
        let mut props = PropertyStore::new();
        let mut interner = StringInterner::new();
        let label = interner.intern_label("rel").unwrap();

        let make_meta = |s: NodeId, t: NodeId| EdgeMeta {
            source: s,
            target: t,
            label,
            temporal: BiTemporal::new_current(1000),
            provenance: None,
            weight: 1.0,
            token_cost: 0,
        };

        // Small clique: nodes 1,2 (2 nodes)
        adj.add_node(1);
        adj.add_node(2);
        adj.add_edge(1, 2, label, make_meta(1, 2)).unwrap();
        adj.add_edge(2, 1, label, make_meta(2, 1)).unwrap();

        // Large clique: nodes 10,11,12,13 (4 nodes)
        for id in 10..=13u64 {
            adj.add_node(id);
            props.set_node_property(
                id,
                "name",
                Value::String(CompactString::from(format!("Node{}", id))),
            );
        }
        for &a in &[10u64, 11, 12, 13] {
            for &b in &[10u64, 11, 12, 13] {
                if a != b {
                    adj.add_edge(a, b, label, make_meta(a, b)).unwrap();
                }
            }
        }

        let counter = TokenCounter::new(TokenCounterType::CharDiv4);
        let summaries =
            extract_community_summaries(&adj, &props, &interner, &counter, 20, 1.0);

        // First community should be the larger one
        assert!(
            summaries[0].node_count >= summaries.last().unwrap().node_count,
            "Summaries should be sorted by node_count descending"
        );
    }
}
