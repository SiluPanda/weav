//! Shared context result types used across weav crates.
//!
//! These types represent the output of context queries and are consumed by
//! the query executor, RESP3 protocol layer, and HTTP server.

use crate::types::{BiTemporal, NodeId, Provenance};

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
    /// Timing breakdown: seed acquisition (microseconds).
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub seed_time_us: u64,
    /// Timing breakdown: flow scoring + fusion (microseconds).
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub flow_time_us: u64,
    /// Timing breakdown: RRF fusion (microseconds).
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub fusion_time_us: u64,
    /// Timing breakdown: chunk building (microseconds).
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub chunk_time_us: u64,
    /// Timing breakdown: budget enforcement (microseconds).
    #[serde(skip_serializing_if = "is_zero_u64")]
    pub budget_time_us: u64,
    /// Query plan description (populated when explain=true).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<String>,
    /// LLM-ready formatted messages (populated when output_format is set).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formatted_messages: Option<String>,
    /// Optional subgraph of relevant nodes and edges (when include_subgraph=true).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subgraph: Option<ContextSubgraph>,
}

/// A subgraph extracted from the context query results.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ContextSubgraph {
    pub nodes: Vec<SubgraphNode>,
    pub edges: Vec<SubgraphEdge>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SubgraphNode {
    pub node_id: u64,
    pub label: String,
    pub importance: f32,
    pub properties: Vec<(String, String)>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SubgraphEdge {
    pub edge_id: u64,
    pub source: u64,
    pub target: u64,
    pub label: String,
    pub weight: f32,
}

/// Helper for serde `skip_serializing_if` on `u64` timing fields.
pub fn is_zero_u64(v: &u64) -> bool {
    *v == 0
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
