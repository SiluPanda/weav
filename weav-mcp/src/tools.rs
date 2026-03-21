//! MCP tool definitions for Weav.
//!
//! Each tool wraps a Weav engine command. Parameters are deserialized from
//! the MCP tool call, translated into the appropriate `Command` variant,
//! executed against the engine, and the `CommandResponse` is serialized
//! back as MCP `CallToolResult` text content.

use std::collections::HashMap;

use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::wrapper::Parameters,
    model::*,
    schemars, tool, tool_handler, tool_router,
};
use serde::Deserialize;

use weav_core::types::{Direction, TokenAllocation, TokenBudget, Value};
use weav_query::parser::{
    Command, ContextQuery, EdgeAddCmd, EdgeDeleteCmd, EdgeGetCmd, GraphCreateCmd,
    IngestCmd, NodeAddCmd, NodeDeleteCmd, NodeGetCmd, NodeUpdateCmd, SchemaGetCmd,
    SchemaSetCmd, SeedStrategy,
};
use weav_server::engine::CommandResponse;

use crate::WeavMcpServer;

// ─── Parameter structs ──────────────────────────────────────────────────────

/// Parameters for graph_info tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GraphInfoParams {
    /// Name of the graph.
    pub graph: String,
}

/// Parameters for graph_create tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GraphCreateParams {
    /// Name of the graph to create.
    pub name: String,
}

/// Parameters for graph_drop tool.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GraphDropParams {
    /// Name of the graph to drop.
    pub name: String,
}

/// Parameters for adding a node.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct NodeAddParams {
    /// Name of the graph to add the node to.
    pub graph: String,
    /// Label for the node (e.g. "Person", "Document", "Concept").
    pub label: String,
    /// Key-value properties for the node.
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
    /// Optional entity key for deduplication.
    pub entity_key: Option<String>,
}

/// Parameters for getting a node.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct NodeGetParams {
    /// Name of the graph.
    pub graph: String,
    /// Node ID to look up (provide either node_id or entity_key).
    pub node_id: Option<u64>,
    /// Entity key to look up (provide either node_id or entity_key).
    pub entity_key: Option<String>,
}

/// Parameters for searching nodes by property.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchNodesParams {
    /// Name of the graph.
    pub graph: String,
    /// Property key to search on.
    pub key: String,
    /// Value to match (compared as string, int, float, or bool).
    pub value: String,
    /// Maximum number of results (default 100).
    pub limit: Option<u32>,
}

/// Parameters for getting neighbors of a node.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetNeighborsParams {
    /// Name of the graph.
    pub graph: String,
    /// Node ID to get neighbors for.
    pub node_id: u64,
}

/// Parameters for exporting a graph.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ExportGraphParams {
    /// Name of the graph to export.
    pub graph: String,
}

/// Parameters for getting graph statistics.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GraphStatsParams {
    /// Name of the graph.
    pub graph: String,
}

/// Parameters for updating a node's properties.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct NodeUpdateParams {
    /// Name of the graph.
    pub graph: String,
    /// Node ID to update.
    pub node_id: u64,
    /// New key-value properties to set (merges with existing).
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
    /// Optional new embedding vector.
    pub embedding: Option<Vec<f32>>,
}

/// Parameters for deleting a node.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct NodeDeleteParams {
    /// Name of the graph.
    pub graph: String,
    /// Node ID to delete.
    pub node_id: u64,
}

/// Parameters for getting an edge.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct EdgeGetParams {
    /// Name of the graph.
    pub graph: String,
    /// Edge ID to retrieve.
    pub edge_id: u64,
}

/// Parameters for deleting an edge.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct EdgeDeleteParams {
    /// Name of the graph.
    pub graph: String,
    /// Edge ID to delete.
    pub edge_id: u64,
}

/// Parameters for context-aware retrieval with token budget.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ContextQueryParams {
    /// Name of the graph.
    pub graph: String,
    /// Natural language query text.
    pub query: Option<String>,
    /// Seed node entity keys to start traversal from.
    #[serde(default)]
    pub seed_keys: Vec<String>,
    /// Embedding vector for semantic seed selection.
    pub embedding: Option<Vec<f32>>,
    /// Number of top-k vector results for seeding (default 10).
    pub top_k: Option<u16>,
    /// Maximum token budget (default 4096).
    pub max_tokens: Option<u32>,
    /// Budget allocation strategy: "auto", "diversity" (MMR), "submodular", or "proportional".
    pub strategy: Option<String>,
    /// Maximum traversal depth (default 3).
    pub max_depth: Option<u8>,
    /// Include provenance metadata in results.
    #[serde(default)]
    pub include_provenance: bool,
    /// Point-in-time temporal query (milliseconds since epoch).
    pub temporal_at: Option<u64>,
    /// If true, return the query plan without executing.
    #[serde(default)]
    pub explain: bool,
    /// Named budget preset: "small"/"4k", "medium"/"8k", "large"/"16k",
    /// "xl"/"32k", "xxl"/"128k". Overrides `max_tokens` when set.
    pub budget_preset: Option<String>,
    /// Output format: "raw" (default), "anthropic", "openai".
    /// When set, the result includes LLM-ready formatted messages.
    pub output_format: Option<String>,
}

/// Parameters for vector similarity search.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct VectorSearchParams {
    /// Name of the graph.
    pub graph: String,
    /// Query embedding vector.
    pub embedding: Vec<f32>,
    /// Number of results to return (default 10).
    pub top_k: Option<usize>,
}

/// Parameters for running a graph algorithm.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RunAlgorithmParams {
    /// Name of the graph.
    pub graph: String,
    /// Algorithm name: "pagerank", "communities", "shortest_path", "connected_components",
    /// "betweenness", "closeness", "degree", "triangle_count", "scc", "topological_sort".
    pub algorithm: String,
    /// Source node ID (for path algorithms).
    pub source: Option<u64>,
    /// Target node ID (for path algorithms).
    pub target: Option<u64>,
    /// Maximum iterations (for iterative algorithms, default 100).
    pub max_iterations: Option<u32>,
    /// Resolution parameter (for community detection, default 1.0).
    pub resolution: Option<f32>,
    /// Maximum results to return (default 50).
    pub limit: Option<usize>,
}

/// Parameters for adding an edge between two nodes.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct EdgeAddParams {
    /// Name of the graph.
    pub graph: String,
    /// Source node ID.
    pub source: u64,
    /// Target node ID.
    pub target: u64,
    /// Label for the edge (e.g. "KNOWS", "CONTAINS", "REFERENCES").
    pub label: String,
    /// Edge weight (0.0 to 1.0). Defaults to 1.0.
    pub weight: Option<f32>,
    /// Key-value properties for the edge.
    #[serde(default)]
    pub properties: HashMap<String, serde_json::Value>,
}

/// Parameters for ingesting a document.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct IngestDocumentParams {
    /// Name of the graph.
    pub graph: String,
    /// Document content (text, markdown, etc.).
    pub content: String,
    /// Format hint: "text", "markdown", "csv", "pdf". Defaults to "text".
    pub format: Option<String>,
    /// Optional document ID for tracking.
    pub document_id: Option<String>,
}

/// Parameters for setting a schema constraint.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SchemaSetParams {
    /// Name of the graph.
    pub graph: String,
    /// Target: "node" or "edge".
    pub target: String,
    /// Label to apply the constraint to (e.g. "Person", "KNOWS").
    pub label: String,
    /// Constraint type: "type", "required", or "unique".
    pub constraint_type: String,
    /// Property name the constraint applies to.
    pub property: String,
    /// Value type (for "type" constraint): "string", "int", "float", "bool", "bytes", "vector", "list", "map", "timestamp".
    pub value_type: Option<String>,
}

/// Parameters for getting the schema of a graph.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SchemaGetParams {
    /// Name of the graph.
    pub graph: String,
}

/// Parameters for setting a config key.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ConfigSetParams {
    /// Config key to set.
    pub key: String,
    /// Value to set.
    pub value: String,
}

/// Parameters for triggering a snapshot.
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TriggerSnapshotParams {}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Convert a `serde_json::Value` to a Weav `Value`.
fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => {
            Value::String(compact_str::CompactString::from(s.as_str()))
        }
        serde_json::Value::Array(arr) => {
            Value::List(arr.iter().map(json_to_value).collect())
        }
        serde_json::Value::Object(obj) => {
            let pairs = obj
                .iter()
                .map(|(k, v)| (compact_str::CompactString::from(k.as_str()), json_to_value(v)))
                .collect();
            Value::Map(pairs)
        }
    }
}

/// Convert a `Value` to a `serde_json::Value`.
fn value_to_json(v: &Value) -> serde_json::Value {
    match v {
        Value::Null => serde_json::Value::Null,
        Value::Bool(b) => serde_json::Value::Bool(*b),
        Value::Int(i) => serde_json::json!(*i),
        Value::Float(f) => serde_json::json!(*f),
        Value::String(s) => serde_json::Value::String(s.to_string()),
        Value::Bytes(b) => serde_json::json!(b),
        Value::Vector(v) => serde_json::json!(v),
        Value::List(items) => {
            serde_json::Value::Array(items.iter().map(value_to_json).collect())
        }
        Value::Map(pairs) => {
            let mut map = serde_json::Map::new();
            for (k, v) in pairs {
                map.insert(k.to_string(), value_to_json(v));
            }
            serde_json::Value::Object(map)
        }
        Value::Timestamp(ts) => serde_json::json!(*ts),
    }
}

/// Convert properties `HashMap` to Weav property pairs.
fn props_to_pairs(props: &HashMap<String, serde_json::Value>) -> Vec<(String, Value)> {
    props
        .iter()
        .map(|(k, v)| (k.clone(), json_to_value(v)))
        .collect()
}

/// Build a success `CallToolResult` from a JSON-serializable value.
fn success_json<T: serde::Serialize>(value: &T) -> Result<CallToolResult, McpError> {
    let text = serde_json::to_string_pretty(value).map_err(|e| {
        McpError::internal_error(format!("JSON serialization failed: {e}"), None)
    })?;
    Ok(CallToolResult::success(vec![Content::text(text)]))
}

/// Build an error `CallToolResult` from a Weav error.
fn weav_error(err: weav_core::error::WeavError) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::error(vec![Content::text(err.to_string())]))
}

// ─── Tool implementations ──────────────────────────────────────────────────

#[tool_router]
impl WeavMcpServer {
    /// Create a new `WeavMcpServer` wrapping the given engine.
    pub fn new(engine: std::sync::Arc<weav_server::engine::Engine>) -> Self {
        Self {
            engine,
            tool_router: Self::tool_router(),
        }
    }

    /// List all graphs in the database.
    #[tool(description = "List all graphs in the Weav database. Returns an array of graph names.")]
    fn graph_list(&self) -> Result<CallToolResult, McpError> {
        match self.engine.execute_command(Command::GraphList, None) {
            Ok(CommandResponse::StringList(names)) => success_json(&names),
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Get information about a graph.
    #[tool(description = "Get information about a graph including its name, node count, and edge count.")]
    fn graph_info(
        &self,
        Parameters(params): Parameters<GraphInfoParams>,
    ) -> Result<CallToolResult, McpError> {
        match self
            .engine
            .execute_command(Command::GraphInfo(params.graph), None)
        {
            Ok(CommandResponse::GraphInfo(info)) => success_json(&serde_json::json!({
                "name": info.name,
                "node_count": info.node_count,
                "edge_count": info.edge_count,
                "vector_count": info.vector_count,
                "label_count": info.label_count,
                "default_ttl_ms": info.default_ttl_ms,
            })),
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Create a new graph.
    #[tool(description = "Create a new empty graph in the Weav database.")]
    fn graph_create(
        &self,
        Parameters(params): Parameters<GraphCreateParams>,
    ) -> Result<CallToolResult, McpError> {
        let cmd = Command::GraphCreate(GraphCreateCmd {
            name: params.name.clone(),
            config: None,
        });
        match self.engine.execute_command(cmd, None) {
            Ok(CommandResponse::Ok) => success_json(&serde_json::json!({
                "status": "created",
                "graph": params.name,
            })),
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Drop (delete) a graph.
    #[tool(description = "Drop (permanently delete) a graph and all its nodes and edges.")]
    fn graph_drop(
        &self,
        Parameters(params): Parameters<GraphDropParams>,
    ) -> Result<CallToolResult, McpError> {
        match self
            .engine
            .execute_command(Command::GraphDrop(params.name.clone()), None)
        {
            Ok(CommandResponse::Ok) => success_json(&serde_json::json!({
                "status": "dropped",
                "graph": params.name,
            })),
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Add a node to a graph.
    #[tool(description = "Add a node to a graph with a label and optional properties. Returns the assigned node ID.")]
    fn node_add(
        &self,
        Parameters(params): Parameters<NodeAddParams>,
    ) -> Result<CallToolResult, McpError> {
        let cmd = Command::NodeAdd(NodeAddCmd {
            graph: params.graph,
            label: params.label,
            properties: props_to_pairs(&params.properties),
            embedding: None,
            entity_key: params.entity_key,
            ttl_ms: None,
        });
        match self.engine.execute_command(cmd, None) {
            Ok(CommandResponse::Integer(node_id)) => success_json(&serde_json::json!({
                "node_id": node_id,
            })),
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Get a node by ID or entity key.
    #[tool(description = "Get a node from a graph by its numeric ID or entity key. Returns the node's label and properties.")]
    fn node_get(
        &self,
        Parameters(params): Parameters<NodeGetParams>,
    ) -> Result<CallToolResult, McpError> {
        let cmd = Command::NodeGet(NodeGetCmd {
            graph: params.graph,
            node_id: params.node_id,
            entity_key: params.entity_key,
        });
        match self.engine.execute_command(cmd, None) {
            Ok(CommandResponse::NodeInfo(info)) => {
                let props: serde_json::Map<String, serde_json::Value> = info
                    .properties
                    .iter()
                    .map(|(k, v)| (k.clone(), value_to_json(v)))
                    .collect();
                success_json(&serde_json::json!({
                    "node_id": info.node_id,
                    "label": info.label,
                    "properties": props,
                }))
            }
            Ok(CommandResponse::Null) => Ok(CallToolResult::success(vec![Content::text(
                "Node not found",
            )])),
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Add an edge between two nodes.
    #[tool(description = "Add a directed edge between two nodes in a graph. Specify source and target node IDs, a label, and an optional weight (0.0 to 1.0).")]
    fn edge_add(
        &self,
        Parameters(params): Parameters<EdgeAddParams>,
    ) -> Result<CallToolResult, McpError> {
        let cmd = Command::EdgeAdd(EdgeAddCmd {
            graph: params.graph,
            source: params.source,
            target: params.target,
            label: params.label,
            weight: params.weight.unwrap_or(1.0),
            properties: props_to_pairs(&params.properties),
            ttl_ms: None,
        });
        match self.engine.execute_command(cmd, None) {
            Ok(CommandResponse::Integer(edge_id)) => success_json(&serde_json::json!({
                "edge_id": edge_id,
            })),
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Search nodes by property key/value.
    #[tool(description = "Search for nodes in a graph by matching a property key against a value. The value is compared as string, integer, float, or boolean. Returns matching node IDs, labels, and properties.")]
    fn search_nodes(
        &self,
        Parameters(params): Parameters<SearchNodesParams>,
    ) -> Result<CallToolResult, McpError> {
        let graph_arc = self.engine.get_graph(&params.graph).map_err(|e| {
            McpError::internal_error(e.to_string(), None)
        })?;
        let gs = graph_arc.read();

        let search_value = params.value.clone();
        let matching_ids = gs.properties.nodes_where(&params.key, &move |v| {
            // Match against string representation.
            if let Some(s) = v.as_str() {
                return s == search_value;
            }
            if let Some(i) = v.as_int()
                && let Ok(parsed) = search_value.parse::<i64>()
            {
                return i == parsed;
            }
            if let Some(f) = v.as_float()
                && let Ok(parsed) = search_value.parse::<f64>()
            {
                return (f - parsed).abs() < f64::EPSILON;
            }
            if let Some(b) = v.as_bool()
                && let Ok(parsed) = search_value.parse::<bool>()
            {
                return b == parsed;
            }
            false
        });

        let limit = params.limit.unwrap_or(100) as usize;
        let results: Vec<serde_json::Value> = matching_ids
            .iter()
            .take(limit)
            .map(|&nid| {
                let label = gs
                    .properties
                    .get_node_property(nid, "_label")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let props: serde_json::Map<String, serde_json::Value> = gs
                    .properties
                    .get_all_node_properties(nid)
                    .into_iter()
                    .filter(|(k, _)| !k.starts_with('_'))
                    .map(|(k, v)| (k.to_string(), value_to_json(v)))
                    .collect();
                serde_json::json!({
                    "node_id": nid,
                    "label": label,
                    "properties": props,
                })
            })
            .collect();

        success_json(&serde_json::json!({
            "count": results.len(),
            "nodes": results,
        }))
    }

    /// Get all neighbors of a node.
    #[tool(description = "Get all neighbors (both incoming and outgoing) of a node in a graph. Returns neighbor node IDs, edge IDs, labels, direction, and edge weight.")]
    fn get_neighbors(
        &self,
        Parameters(params): Parameters<GetNeighborsParams>,
    ) -> Result<CallToolResult, McpError> {
        let graph_arc = self.engine.get_graph(&params.graph).map_err(|e| {
            McpError::internal_error(e.to_string(), None)
        })?;
        let gs = graph_arc.read();

        if !gs.adjacency.has_node(params.node_id) {
            return weav_error(weav_core::error::WeavError::NodeNotFound(
                params.node_id,
                gs.graph_id,
            ));
        }

        let neighbors = gs.adjacency.neighbors_both(params.node_id, None);
        let results: Vec<serde_json::Value> = neighbors
            .iter()
            .map(|&(neighbor_id, edge_id, ref direction)| {
                let label = gs
                    .properties
                    .get_node_property(neighbor_id, "_label")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let dir_str = match direction {
                    weav_core::types::Direction::Outgoing => "outgoing",
                    weav_core::types::Direction::Incoming => "incoming",
                    weav_core::types::Direction::Both => "both",
                };
                let edge_label = gs
                    .adjacency
                    .get_edge(edge_id)
                    .map(|e| {
                        gs.interner
                            .resolve_label(e.label)
                            .unwrap_or("unknown")
                            .to_string()
                    })
                    .unwrap_or_default();
                let weight = gs
                    .adjacency
                    .get_edge(edge_id)
                    .map(|e| e.weight)
                    .unwrap_or(1.0);
                serde_json::json!({
                    "node_id": neighbor_id,
                    "edge_id": edge_id,
                    "label": label,
                    "edge_label": edge_label,
                    "direction": dir_str,
                    "weight": weight,
                })
            })
            .collect();

        success_json(&serde_json::json!({
            "node_id": params.node_id,
            "neighbor_count": results.len(),
            "neighbors": results,
        }))
    }

    /// Export entire graph as JSON.
    #[tool(description = "Export an entire graph as JSON including all nodes with their labels and properties, and all edges with source, target, label, and weight.")]
    fn export_graph(
        &self,
        Parameters(params): Parameters<ExportGraphParams>,
    ) -> Result<CallToolResult, McpError> {
        let graph_arc = self.engine.get_graph(&params.graph).map_err(|e| {
            McpError::internal_error(e.to_string(), None)
        })?;
        let gs = graph_arc.read();

        // Export all nodes.
        let node_ids = gs.adjacency.all_node_ids();
        let nodes: Vec<serde_json::Value> = node_ids
            .iter()
            .map(|&nid| {
                let label = gs
                    .properties
                    .get_node_property(nid, "_label")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let props: serde_json::Map<String, serde_json::Value> = gs
                    .properties
                    .get_all_node_properties(nid)
                    .into_iter()
                    .filter(|(k, _)| !k.starts_with('_'))
                    .map(|(k, v)| (k.to_string(), value_to_json(v)))
                    .collect();
                serde_json::json!({
                    "node_id": nid,
                    "label": label,
                    "properties": props,
                })
            })
            .collect();

        // Export all edges.
        let edges: Vec<serde_json::Value> = gs
            .adjacency
            .all_edges()
            .map(|(eid, meta)| {
                let edge_label = gs
                    .interner
                    .resolve_label(meta.label)
                    .unwrap_or("unknown")
                    .to_string();
                let props: serde_json::Map<String, serde_json::Value> = gs
                    .properties
                    .get_all_edge_properties(eid)
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), value_to_json(v)))
                    .collect();
                serde_json::json!({
                    "edge_id": eid,
                    "source": meta.source,
                    "target": meta.target,
                    "label": edge_label,
                    "weight": meta.weight,
                    "properties": props,
                })
            })
            .collect();

        success_json(&serde_json::json!({
            "graph": params.graph,
            "node_count": nodes.len(),
            "edge_count": edges.len(),
            "nodes": nodes,
            "edges": edges,
        }))
    }

    /// Get detailed graph statistics.
    #[tool(description = "Get detailed statistics for a graph including node count, edge count, vector count, label distribution, and average node degree.")]
    fn graph_stats(
        &self,
        Parameters(params): Parameters<GraphStatsParams>,
    ) -> Result<CallToolResult, McpError> {
        let graph_arc = self.engine.get_graph(&params.graph).map_err(|e| {
            McpError::internal_error(e.to_string(), None)
        })?;
        let gs = graph_arc.read();

        let node_count = gs.adjacency.node_count();
        let edge_count = gs.adjacency.edge_count();
        let vector_count = gs.vector_index.len();

        // Compute label distribution.
        let node_ids = gs.adjacency.all_node_ids();
        let mut label_counts: HashMap<String, u64> = HashMap::new();
        let mut total_degree: u64 = 0;
        for &nid in &node_ids {
            let label = gs
                .properties
                .get_node_property(nid, "_label")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            *label_counts.entry(label).or_insert(0) += 1;
            total_degree += gs.adjacency.neighbors_both(nid, None).len() as u64;
        }

        let avg_degree = if node_count > 0 {
            total_degree as f64 / node_count as f64
        } else {
            0.0
        };

        success_json(&serde_json::json!({
            "graph": params.graph,
            "node_count": node_count,
            "edge_count": edge_count,
            "vector_count": vector_count,
            "label_distribution": label_counts,
            "avg_degree": avg_degree,
        }))
    }

    /// Get comprehensive graph statistics including degree distribution,
    /// label/edge-type distribution, memory estimates, and temporal node count.
    #[tool(description = "Get comprehensive statistics for a graph: degree distribution (min/max/avg/p50/p95/p99), label distribution, edge type distribution, memory estimates, temporal node count, and graph config.")]
    fn graph_stats_detailed(
        &self,
        Parameters(params): Parameters<GraphStatsParams>,
    ) -> Result<CallToolResult, McpError> {
        match self.engine.handle_detailed_stats(&params.graph) {
            Ok(CommandResponse::Text(json_str)) => {
                Ok(CallToolResult::success(vec![Content::text(json_str)]))
            }
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Get server info and statistics.
    #[tool(description = "Get Weav server information including version and number of graphs.")]
    fn server_info(&self) -> Result<CallToolResult, McpError> {
        match self.engine.execute_command(Command::Info, None) {
            Ok(CommandResponse::Text(text)) => {
                Ok(CallToolResult::success(vec![Content::text(text)]))
            }
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Update a node's properties or embedding.
    #[tool(description = "Update an existing node's properties (merge) or embedding vector. Provide the node ID and new properties to set.")]
    fn node_update(
        &self,
        Parameters(params): Parameters<NodeUpdateParams>,
    ) -> Result<CallToolResult, McpError> {
        let cmd = Command::NodeUpdate(NodeUpdateCmd {
            graph: params.graph,
            node_id: params.node_id,
            properties: props_to_pairs(&params.properties),
            embedding: params.embedding,
        });
        match self.engine.execute_command(cmd, None) {
            Ok(CommandResponse::Ok) => success_json(&serde_json::json!({
                "status": "updated",
                "node_id": params.node_id,
            })),
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Delete a node from a graph.
    #[tool(description = "Delete a node and all its connected edges from a graph.")]
    fn node_delete(
        &self,
        Parameters(params): Parameters<NodeDeleteParams>,
    ) -> Result<CallToolResult, McpError> {
        let cmd = Command::NodeDelete(NodeDeleteCmd {
            graph: params.graph,
            node_id: params.node_id,
        });
        match self.engine.execute_command(cmd, None) {
            Ok(CommandResponse::Ok) => success_json(&serde_json::json!({
                "status": "deleted",
                "node_id": params.node_id,
            })),
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Get an edge by ID.
    #[tool(description = "Get an edge from a graph by its edge ID. Returns source, target, label, and weight.")]
    fn edge_get(
        &self,
        Parameters(params): Parameters<EdgeGetParams>,
    ) -> Result<CallToolResult, McpError> {
        let cmd = Command::EdgeGet(EdgeGetCmd {
            graph: params.graph,
            edge_id: params.edge_id,
        });
        match self.engine.execute_command(cmd, None) {
            Ok(CommandResponse::EdgeInfo(info)) => success_json(&serde_json::json!({
                "edge_id": info.edge_id,
                "source": info.source,
                "target": info.target,
                "label": info.label,
                "weight": info.weight,
            })),
            Ok(CommandResponse::Null) => Ok(CallToolResult::success(vec![Content::text(
                "Edge not found",
            )])),
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Delete an edge from a graph.
    #[tool(description = "Delete an edge from a graph by its edge ID.")]
    fn edge_delete(
        &self,
        Parameters(params): Parameters<EdgeDeleteParams>,
    ) -> Result<CallToolResult, McpError> {
        let cmd = Command::EdgeDelete(EdgeDeleteCmd {
            graph: params.graph,
            edge_id: params.edge_id,
        });
        match self.engine.execute_command(cmd, None) {
            Ok(CommandResponse::Ok) => success_json(&serde_json::json!({
                "status": "deleted",
                "edge_id": params.edge_id,
            })),
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Context-aware retrieval with token budget management.
    #[tool(description = "Retrieve context from a graph with intelligent token budget management. This is Weav's unique differentiator — no other graph database offers budget-aware context retrieval. Supports multiple strategies: 'auto' (greedy knapsack), 'diversity' (MMR for diverse results), 'submodular' (facility location for maximum coverage), or 'proportional' (fixed allocation). Provide seed nodes and/or an embedding vector to start traversal.")]
    fn context_query(
        &self,
        Parameters(params): Parameters<ContextQueryParams>,
    ) -> Result<CallToolResult, McpError> {
        let seeds = match (&params.embedding, params.seed_keys.is_empty()) {
            (Some(emb), true) => SeedStrategy::Vector {
                embedding: emb.clone(),
                top_k: params.top_k.unwrap_or(10),
            },
            (None, false) => SeedStrategy::Nodes(params.seed_keys.clone()),
            (Some(emb), false) => SeedStrategy::Both {
                embedding: emb.clone(),
                top_k: params.top_k.unwrap_or(10),
                node_keys: params.seed_keys.clone(),
            },
            (None, true) => {
                return Ok(CallToolResult::error(vec![Content::text(
                    "Must provide either 'embedding' or 'seed_keys' (or both)",
                )]));
            }
        };

        let max_tokens = params.max_tokens.unwrap_or(4096);
        let allocation = match params.strategy.as_deref() {
            Some("diversity") | Some("mmr") => TokenAllocation::DiversityAware { lambda: 0.7 },
            Some("submodular") | Some("facility") => {
                TokenAllocation::SubmodularFacilityLocation { alpha: 0.5 }
            }
            Some("proportional") => TokenAllocation::Proportional {
                entities_pct: 0.4,
                relationships_pct: 0.3,
                text_chunks_pct: 0.2,
                metadata_pct: 0.1,
            },
            _ => TokenAllocation::Auto,
        };

        // Resolve budget from preset or explicit max_tokens.
        let resolved_tokens = if let Some(ref preset) = params.budget_preset {
            weav_query::parser::budget_preset(preset).unwrap_or(max_tokens)
        } else {
            max_tokens
        };

        let cmd = Command::Context(ContextQuery {
            query_text: params.query,
            graph: params.graph,
            budget: Some(TokenBudget {
                max_tokens: resolved_tokens,
                allocation,
            }),
            seeds,
            max_depth: params.max_depth.unwrap_or(3),
            direction: Direction::Both,
            edge_filter: None,
            decay: None,
            include_provenance: params.include_provenance,
            temporal_at: params.temporal_at,
            limit: None,
            sort: None,
            explain: params.explain,
            output_format: params.output_format.clone(),
        });

        match self.engine.execute_command(cmd, None) {
            Ok(CommandResponse::Context(ctx)) => {
                let chunks: Vec<serde_json::Value> = ctx
                    .chunks
                    .iter()
                    .map(|c| {
                        let mut obj = serde_json::json!({
                            "node_id": c.node_id,
                            "label": c.label,
                            "content": c.content,
                            "relevance_score": c.relevance_score,
                            "depth": c.depth,
                            "token_count": c.token_count,
                        });
                        if let Some(ref prov) = c.provenance {
                            obj["provenance"] = serde_json::json!({
                                "source": prov.source.as_str(),
                                "confidence": prov.confidence,
                            });
                        }
                        if let Some(ref temporal) = c.temporal {
                            obj["temporal"] = serde_json::json!({
                                "valid_from": temporal.valid_from,
                                "valid_until": temporal.valid_until,
                            });
                        }
                        if !c.relationships.is_empty() {
                            obj["relationships"] = serde_json::json!(c.relationships.len());
                        }
                        obj
                    })
                    .collect();
                success_json(&serde_json::json!({
                    "total_tokens": ctx.total_tokens,
                    "budget_used": ctx.budget_used,
                    "nodes_considered": ctx.nodes_considered,
                    "nodes_included": ctx.nodes_included,
                    "query_time_us": ctx.query_time_us,
                    "chunks": chunks,
                }))
            }
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Vector similarity search.
    #[tool(description = "Search for the most similar nodes using vector embeddings (HNSW index). Returns nodes ranked by cosine similarity to the query vector.")]
    fn vector_search(
        &self,
        Parameters(params): Parameters<VectorSearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let graph_arc = self.engine.get_graph(&params.graph).map_err(|e| {
            McpError::internal_error(e.to_string(), None)
        })?;
        let gs = graph_arc.read();

        let top_k = params.top_k.unwrap_or(10) as u16;
        let results = gs.vector_index.search(&params.embedding, top_k, None)
            .unwrap_or_default();

        let items: Vec<serde_json::Value> = results
            .iter()
            .map(|&(node_id, score): &(u64, f32)| {
                let label = gs
                    .properties
                    .get_node_property(node_id, "_label")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let props: serde_json::Map<String, serde_json::Value> = gs
                    .properties
                    .get_all_node_properties(node_id)
                    .into_iter()
                    .filter(|(k, _)| !k.starts_with('_'))
                    .map(|(k, v)| (k.to_string(), value_to_json(v)))
                    .collect();
                serde_json::json!({
                    "node_id": node_id,
                    "similarity": score,
                    "label": label,
                    "properties": props,
                })
            })
            .collect();

        success_json(&serde_json::json!({
            "graph": params.graph,
            "count": items.len(),
            "results": items,
        }))
    }

    /// Run a graph algorithm.
    #[tool(description = "Run a graph algorithm on a graph. Supported algorithms: 'pagerank' (Personalized PageRank), 'communities' (Louvain modularity), 'label_propagation', 'shortest_path' (Dijkstra, requires source+target), 'connected_components', 'betweenness' (centrality), 'closeness' (centrality), 'degree' (centrality), 'triangle_count', 'scc' (strongly connected components), 'topological_sort', 'fastrp' (Fast Random Projection node embeddings). Returns algorithm-specific results.")]
    fn run_algorithm(
        &self,
        Parameters(params): Parameters<RunAlgorithmParams>,
    ) -> Result<CallToolResult, McpError> {
        use weav_graph::traversal;

        let graph_arc = self.engine.get_graph(&params.graph).map_err(|e| {
            McpError::internal_error(e.to_string(), None)
        })?;
        let gs = graph_arc.read();
        let limit = params.limit.unwrap_or(50);

        match params.algorithm.as_str() {
            "pagerank" => {
                let max_iter = params.max_iterations.unwrap_or(100);
                let scores = traversal::personalized_pagerank(
                    &gs.adjacency,
                    &[],
                    0.85,
                    max_iter,
                    1e-6,
                );
                let top: Vec<serde_json::Value> = scores
                    .iter()
                    .take(limit)
                    .map(|sn| serde_json::json!({"node_id": sn.node_id, "score": sn.score}))
                    .collect();
                success_json(&serde_json::json!({
                    "algorithm": "pagerank",
                    "count": top.len(),
                    "results": top,
                }))
            }
            "communities" | "louvain" => {
                let max_iter = params.max_iterations.unwrap_or(100);
                let resolution = params.resolution.unwrap_or(1.0);
                let communities =
                    traversal::modularity_communities(&gs.adjacency, max_iter, resolution);
                // Group by community
                let mut groups: HashMap<u64, Vec<u64>> = HashMap::new();
                for (&node, &comm) in &communities {
                    groups.entry(comm).or_default().push(node);
                }
                let mut sorted_groups: Vec<(u64, Vec<u64>)> =
                    groups.into_iter().collect();
                sorted_groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
                let result: Vec<serde_json::Value> = sorted_groups
                    .iter()
                    .take(limit)
                    .map(|(comm, members)| {
                        serde_json::json!({
                            "community_id": comm,
                            "size": members.len(),
                            "members": members,
                        })
                    })
                    .collect();
                success_json(&serde_json::json!({
                    "algorithm": "communities",
                    "community_count": sorted_groups.len(),
                    "results": result,
                }))
            }
            "label_propagation" => {
                let max_iter = params.max_iterations.unwrap_or(100);
                let labels =
                    traversal::label_propagation(&gs.adjacency, max_iter);
                let mut groups: HashMap<u64, Vec<u64>> = HashMap::new();
                for (&node, &label) in &labels {
                    groups.entry(label).or_default().push(node);
                }
                let mut sorted_groups: Vec<(u64, Vec<u64>)> =
                    groups.into_iter().collect();
                sorted_groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
                let result: Vec<serde_json::Value> = sorted_groups
                    .iter()
                    .take(limit)
                    .map(|(label, members)| {
                        serde_json::json!({
                            "label": label,
                            "size": members.len(),
                            "members": members,
                        })
                    })
                    .collect();
                success_json(&serde_json::json!({
                    "algorithm": "label_propagation",
                    "community_count": sorted_groups.len(),
                    "results": result,
                }))
            }
            "shortest_path" | "dijkstra" => {
                let source = params.source.ok_or_else(|| {
                    McpError::invalid_params("'source' is required for shortest_path", None)
                })?;
                let target = params.target.ok_or_else(|| {
                    McpError::invalid_params("'target' is required for shortest_path", None)
                })?;
                match traversal::dijkstra_shortest_path(
                    &gs.adjacency,
                    source,
                    target,
                    u8::MAX,
                ) {
                    Some(path) => success_json(&serde_json::json!({
                        "algorithm": "shortest_path",
                        "path": path.nodes,
                        "total_weight": path.total_weight,
                    })),
                    None => Ok(CallToolResult::error(vec![Content::text(
                        "No path found between source and target",
                    )])),
                }
            }
            "connected_components" => {
                let comp_map = traversal::connected_components(&gs.adjacency);
                let mut groups: HashMap<u32, Vec<u64>> = HashMap::new();
                for (&node, &comp) in &comp_map {
                    groups.entry(comp).or_default().push(node);
                }
                let mut sorted: Vec<Vec<u64>> = groups.into_values().collect();
                sorted.sort_by_key(|b| std::cmp::Reverse(b.len()));
                let result: Vec<serde_json::Value> = sorted
                    .iter()
                    .take(limit)
                    .map(|members: &Vec<u64>| {
                        serde_json::json!({
                            "size": members.len(),
                            "members": members,
                        })
                    })
                    .collect();
                success_json(&serde_json::json!({
                    "algorithm": "connected_components",
                    "component_count": sorted.len(),
                    "results": result,
                }))
            }
            "betweenness" => {
                let scores = traversal::betweenness_centrality(
                    &gs.adjacency,
                    &traversal::EdgeFilter::none(),
                );
                let top: Vec<serde_json::Value> = scores
                    .iter()
                    .take(limit)
                    .map(|&(nid, score)| serde_json::json!({"node_id": nid, "score": score}))
                    .collect();
                success_json(&serde_json::json!({
                    "algorithm": "betweenness_centrality",
                    "count": top.len(),
                    "results": top,
                }))
            }
            "closeness" => {
                let scores = traversal::closeness_centrality(
                    &gs.adjacency,
                    &traversal::EdgeFilter::none(),
                );
                let top: Vec<serde_json::Value> = scores
                    .iter()
                    .take(limit)
                    .map(|&(nid, score)| serde_json::json!({"node_id": nid, "score": score}))
                    .collect();
                success_json(&serde_json::json!({
                    "algorithm": "closeness_centrality",
                    "count": top.len(),
                    "results": top,
                }))
            }
            "degree" => {
                let node_ids = gs.adjacency.all_node_ids();
                let n = node_ids.len();
                let divisor = if n > 1 { (n - 1) as f64 } else { 1.0 };
                let mut scores: Vec<(u64, f64)> = node_ids
                    .iter()
                    .map(|&nid| {
                        let deg = gs.adjacency.neighbors_both(nid, None).len();
                        (nid, deg as f64 / divisor)
                    })
                    .collect();
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top: Vec<serde_json::Value> = scores
                    .iter()
                    .take(limit)
                    .map(|&(nid, score)| serde_json::json!({"node_id": nid, "score": score}))
                    .collect();
                success_json(&serde_json::json!({
                    "algorithm": "degree_centrality",
                    "count": top.len(),
                    "results": top,
                }))
            }
            "triangle_count" => {
                let result = traversal::triangle_count(
                    &gs.adjacency,
                    &traversal::EdgeFilter::none(),
                );
                let top: Vec<serde_json::Value> = result.per_node
                    .iter()
                    .take(limit)
                    .map(|&(nid, count, coeff): &(u64, u32, f64)| {
                        serde_json::json!({
                            "node_id": nid,
                            "triangles": count,
                            "clustering_coefficient": coeff,
                        })
                    })
                    .collect();
                success_json(&serde_json::json!({
                    "algorithm": "triangle_count",
                    "total_triangles": result.total_triangles,
                    "results": top,
                }))
            }
            "scc" => {
                let components = traversal::tarjan_scc(&gs.adjacency);
                let result: Vec<serde_json::Value> = components
                    .iter()
                    .take(limit)
                    .map(|members: &Vec<u64>| {
                        serde_json::json!({
                            "size": members.len(),
                            "members": members,
                        })
                    })
                    .collect();
                success_json(&serde_json::json!({
                    "algorithm": "strongly_connected_components",
                    "component_count": components.len(),
                    "results": result,
                }))
            }
            "topological_sort" => match traversal::topological_sort(&gs.adjacency) {
                Ok(order) => {
                    let limited: Vec<u64> = order.iter().copied().take(limit).collect();
                    success_json(&serde_json::json!({
                        "algorithm": "topological_sort",
                        "order": limited,
                    }))
                }
                Err(e) => weav_error(e),
            },
            "fastrp" => {
                let dim = 128usize;
                let iterations = params.max_iterations.unwrap_or(3) as usize;
                let embeddings = traversal::fastrp_embeddings(&gs.adjacency, dim, iterations, 1.0, 42);
                let results: Vec<serde_json::Value> = embeddings
                    .into_iter()
                    .take(limit)
                    .map(|(nid, emb)| serde_json::json!({"node_id": nid, "embedding": emb}))
                    .collect();
                success_json(&serde_json::json!({
                    "algorithm": "fastrp",
                    "count": results.len(),
                    "results": results,
                }))
            }
            other => Ok(CallToolResult::error(vec![Content::text(format!(
                "Unknown algorithm '{other}'. Supported: pagerank, communities, label_propagation, \
                 shortest_path, connected_components, betweenness, closeness, degree, \
                 triangle_count, scc, topological_sort, fastrp"
            ))])),
        }
    }

    /// Ingest a document into a graph using the LLM extraction pipeline.
    #[tool(description = "Ingest a document into a graph. The extraction pipeline chunks the text, extracts entities and relationships using LLM, and builds the knowledge graph automatically. Supports text, markdown, CSV formats.")]
    fn ingest_document(
        &self,
        Parameters(params): Parameters<IngestDocumentParams>,
    ) -> Result<CallToolResult, McpError> {
        let cmd = Command::Ingest(IngestCmd {
            graph: params.graph,
            content: params.content,
            format: params.format,
            document_id: params.document_id,
            skip_extraction: false,
            skip_dedup: false,
            chunk_size: None,
            entity_types: None,
        });
        match self.engine.execute_command(cmd, None) {
            Ok(CommandResponse::IngestResult(info)) => success_json(&serde_json::json!({
                "document_id": info.document_id,
                "chunks_created": info.chunks_created,
                "entities_created": info.entities_created,
                "entities_merged": info.entities_merged,
                "relationships_created": info.relationships_created,
                "pipeline_duration_ms": info.pipeline_duration_ms,
            })),
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Set a schema constraint on a graph.
    #[tool(description = "Set a schema constraint on node or edge labels. Constraint types: 'type' (enforce property data type), 'required' (property must exist), 'unique' (property value must be unique within the label). Example: set type constraint on Person.age to 'int'.")]
    fn schema_set(
        &self,
        Parameters(params): Parameters<SchemaSetParams>,
    ) -> Result<CallToolResult, McpError> {
        let cmd = Command::SchemaSet(SchemaSetCmd {
            graph: params.graph.clone(),
            target: params.target,
            label: params.label.clone(),
            constraint_type: params.constraint_type,
            property: params.property.clone(),
            value_type: params.value_type,
        });
        match self.engine.execute_command(cmd, None) {
            Ok(CommandResponse::Ok) => success_json(&serde_json::json!({
                "status": "constraint_added",
                "graph": params.graph,
                "label": params.label,
                "property": params.property,
            })),
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Get the schema for a graph.
    #[tool(description = "Get all schema constraints defined for a graph, including property type, required, and uniqueness constraints for both node and edge labels.")]
    fn schema_get(
        &self,
        Parameters(params): Parameters<SchemaGetParams>,
    ) -> Result<CallToolResult, McpError> {
        let cmd = Command::SchemaGet(SchemaGetCmd {
            graph: params.graph,
        });
        match self.engine.execute_command(cmd, None) {
            Ok(CommandResponse::Text(schema_json)) => {
                Ok(CallToolResult::success(vec![Content::text(schema_json)]))
            }
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Set a runtime configuration value.
    #[tool(description = "Set a runtime configuration key-value pair on the Weav server.")]
    fn config_set(
        &self,
        Parameters(params): Parameters<ConfigSetParams>,
    ) -> Result<CallToolResult, McpError> {
        let cmd = Command::ConfigSet(params.key.clone(), params.value.clone());
        match self.engine.execute_command(cmd, None) {
            Ok(CommandResponse::Ok) => success_json(&serde_json::json!({
                "status": "set",
                "key": params.key,
                "value": params.value,
            })),
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }

    /// Trigger a persistence snapshot.
    #[tool(description = "Trigger an immediate persistence snapshot of all graph data to disk. Useful before maintenance or to ensure durability.")]
    fn trigger_snapshot(
        &self,
        #[allow(unused_variables)]
        Parameters(params): Parameters<TriggerSnapshotParams>,
    ) -> Result<CallToolResult, McpError> {
        match self.engine.execute_command(Command::Snapshot, None) {
            Ok(CommandResponse::Ok) => success_json(&serde_json::json!({
                "status": "snapshot_triggered",
            })),
            Ok(resp) => Ok(CallToolResult::success(vec![Content::text(format!(
                "{resp:?}"
            ))])),
            Err(e) => weav_error(e),
        }
    }
}

#[tool_handler]
impl ServerHandler for WeavMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("weav-mcp", env!("CARGO_PKG_VERSION")))
            .with_protocol_version(ProtocolVersion::V_2024_11_05)
            .with_instructions(
                "Weav is an in-memory context graph database for AI/LLM workloads. \
                 25 tools available: graph_list, graph_create, graph_info, graph_stats, graph_stats_detailed, graph_drop, \
                 node_add, node_get, node_update, node_delete, \
                 edge_add, edge_get, edge_delete, \
                 search_nodes, get_neighbors, vector_search, \
                 context_query (token-budget-aware retrieval — Weav's unique feature), \
                 run_algorithm (PageRank, communities, shortest path, centrality, etc.), \
                 export_graph, server_info, \
                 ingest_document (LLM entity extraction pipeline), \
                 schema_set, schema_get (property type/required/unique constraints), \
                 config_set, trigger_snapshot. \
                 Start with graph_list, or graph_create to make a new graph. \
                 Use context_query for intelligent, budget-aware context retrieval. \
                 Use ingest_document to automatically build knowledge graphs from text."
                    .to_string(),
            )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use weav_core::config::WeavConfig;
    use weav_server::engine::Engine;

    use crate::WeavMcpServer;

    fn test_server() -> WeavMcpServer {
        let mut config = WeavConfig::default();
        config.persistence.enabled = false;
        let engine = Arc::new(Engine::new(config));
        WeavMcpServer::new(engine)
    }

    #[test]
    fn test_graph_list_empty() {
        use weav_query::parser::Command;
        use weav_server::engine::CommandResponse;

        let server = test_server();
        let resp = server.engine.execute_command(Command::GraphList, None).unwrap();
        match resp {
            CommandResponse::StringList(names) => assert!(names.is_empty()),
            other => panic!("unexpected response: {other:?}"),
        }
    }

    #[test]
    fn test_graph_create_and_info() {
        use weav_query::parser::{Command, GraphCreateCmd};
        use weav_server::engine::CommandResponse;

        let server = test_server();

        // Create a graph.
        let cmd = Command::GraphCreate(GraphCreateCmd {
            name: "test_graph".to_string(),
            config: None,
        });
        let resp = server.engine.execute_command(cmd, None).unwrap();
        assert!(matches!(resp, CommandResponse::Ok));

        // Verify it appears in the list.
        let resp = server.engine.execute_command(Command::GraphList, None).unwrap();
        match resp {
            CommandResponse::StringList(names) => {
                assert_eq!(names.len(), 1);
                assert_eq!(names[0], "test_graph");
            }
            other => panic!("unexpected response: {other:?}"),
        }

        // Get graph info.
        let resp = server
            .engine
            .execute_command(Command::GraphInfo("test_graph".to_string()), None)
            .unwrap();
        match resp {
            CommandResponse::GraphInfo(info) => {
                assert_eq!(info.name, "test_graph");
                assert_eq!(info.node_count, 0);
                assert_eq!(info.edge_count, 0);
            }
            other => panic!("unexpected response: {other:?}"),
        }
    }

    #[test]
    fn test_node_add_and_get() {
        use weav_query::parser::{Command, GraphCreateCmd, NodeAddCmd, NodeGetCmd};
        use weav_server::engine::CommandResponse;

        let server = test_server();

        // Create graph first.
        server
            .engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "g".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        // Add a node.
        let cmd = Command::NodeAdd(NodeAddCmd {
            graph: "g".to_string(),
            label: "Person".to_string(),
            properties: vec![
                ("name".to_string(), weav_core::types::Value::String("Alice".into())),
            ],
            embedding: None,
            entity_key: Some("alice".to_string()),
            ttl_ms: None,
        });
        let resp = server.engine.execute_command(cmd, None).unwrap();
        let node_id = match resp {
            CommandResponse::Integer(id) => id,
            other => panic!("unexpected response: {other:?}"),
        };
        assert!(node_id > 0);

        // Get the node by ID.
        let cmd = Command::NodeGet(NodeGetCmd {
            graph: "g".to_string(),
            node_id: Some(node_id),
            entity_key: None,
        });
        let resp = server.engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::NodeInfo(info) => {
                assert_eq!(info.node_id, node_id);
                assert_eq!(info.label, "Person");
            }
            other => panic!("unexpected response: {other:?}"),
        }
    }

    #[test]
    fn test_edge_add() {
        use weav_query::parser::{Command, EdgeAddCmd, GraphCreateCmd, NodeAddCmd};
        use weav_server::engine::CommandResponse;

        let server = test_server();

        server
            .engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "g".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        // Add two nodes.
        let n1 = match server
            .engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "g".to_string(),
                    label: "Person".to_string(),
                    properties: vec![],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            other => panic!("unexpected: {other:?}"),
        };

        let n2 = match server
            .engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "g".to_string(),
                    label: "Person".to_string(),
                    properties: vec![],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            other => panic!("unexpected: {other:?}"),
        };

        // Add an edge.
        let cmd = Command::EdgeAdd(EdgeAddCmd {
            graph: "g".to_string(),
            source: n1,
            target: n2,
            label: "KNOWS".to_string(),
            weight: 0.9,
            properties: vec![],
            ttl_ms: None,
        });
        let resp = server.engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::Integer(edge_id) => assert!(edge_id > 0),
            other => panic!("unexpected: {other:?}"),
        }

        // Verify graph info reflects the edge.
        let resp = server
            .engine
            .execute_command(Command::GraphInfo("g".to_string()), None)
            .unwrap();
        match resp {
            CommandResponse::GraphInfo(info) => {
                assert_eq!(info.node_count, 2);
                assert_eq!(info.edge_count, 1);
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_graph_drop() {
        use weav_query::parser::{Command, GraphCreateCmd};
        use weav_server::engine::CommandResponse;

        let server = test_server();

        server
            .engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "to_drop".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        // Verify it exists.
        let resp = server.engine.execute_command(Command::GraphList, None).unwrap();
        match resp {
            CommandResponse::StringList(names) => assert!(names.contains(&"to_drop".to_string())),
            other => panic!("unexpected: {other:?}"),
        }

        // Drop it.
        let resp = server
            .engine
            .execute_command(Command::GraphDrop("to_drop".to_string()), None)
            .unwrap();
        assert!(matches!(resp, CommandResponse::Ok));

        // Verify it's gone.
        let resp = server.engine.execute_command(Command::GraphList, None).unwrap();
        match resp {
            CommandResponse::StringList(names) => {
                assert!(!names.contains(&"to_drop".to_string()))
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_server_info() {
        use weav_query::parser::Command;
        use weav_server::engine::CommandResponse;

        let server = test_server();
        let resp = server.engine.execute_command(Command::Info, None).unwrap();
        match resp {
            CommandResponse::Text(text) => {
                assert!(text.contains("weav-server"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_get_info_capabilities() {
        use rmcp::ServerHandler;

        let server = test_server();
        let info = server.get_info();
        assert_eq!(info.server_info.name, "weav-mcp");
        assert!(info.instructions.is_some());
    }

    #[test]
    fn test_tool_router_has_routes() {
        let server = test_server();
        let router = &server.tool_router;
        assert!(router.has_route("graph_list"));
        assert!(router.has_route("graph_info"));
        assert!(router.has_route("graph_create"));
        assert!(router.has_route("graph_drop"));
        assert!(router.has_route("node_add"));
        assert!(router.has_route("node_get"));
        assert!(router.has_route("node_update"));
        assert!(router.has_route("node_delete"));
        assert!(router.has_route("edge_add"));
        assert!(router.has_route("edge_get"));
        assert!(router.has_route("edge_delete"));
        assert!(router.has_route("server_info"));
        assert!(router.has_route("search_nodes"));
        assert!(router.has_route("get_neighbors"));
        assert!(router.has_route("export_graph"));
        assert!(router.has_route("graph_stats"));
        assert!(router.has_route("context_query"));
        assert!(router.has_route("vector_search"));
        assert!(router.has_route("run_algorithm"));
        assert!(router.has_route("ingest_document"));
        assert!(router.has_route("schema_set"));
        assert!(router.has_route("schema_get"));
        assert!(router.has_route("config_set"));
        assert!(router.has_route("trigger_snapshot"));
    }

    /// Helper: create a graph and add two connected nodes for testing new tools.
    fn setup_graph_with_nodes(server: &WeavMcpServer) -> (u64, u64) {
        use weav_query::parser::{Command, EdgeAddCmd, GraphCreateCmd, NodeAddCmd};
        use weav_server::engine::CommandResponse;

        server
            .engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "tg".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        let n1 = match server
            .engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "tg".to_string(),
                    label: "Person".to_string(),
                    properties: vec![
                        ("name".to_string(), weav_core::types::Value::String("Alice".into())),
                        ("age".to_string(), weav_core::types::Value::Int(30)),
                    ],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            other => panic!("unexpected: {other:?}"),
        };

        let n2 = match server
            .engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "tg".to_string(),
                    label: "Document".to_string(),
                    properties: vec![
                        ("name".to_string(), weav_core::types::Value::String("Report".into())),
                    ],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            other => panic!("unexpected: {other:?}"),
        };

        server
            .engine
            .execute_command(
                Command::EdgeAdd(EdgeAddCmd {
                    graph: "tg".to_string(),
                    source: n1,
                    target: n2,
                    label: "AUTHORED".to_string(),
                    weight: 0.8,
                    properties: vec![],
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap();

        (n1, n2)
    }

    #[test]
    fn test_search_nodes_by_name() {
        let server = test_server();
        let (n1, _n2) = setup_graph_with_nodes(&server);

        let graph_arc = server.engine.get_graph("tg").unwrap();
        let gs = graph_arc.read();

        // Search for "Alice" by name property.
        let matching = gs.properties.nodes_where("name", &|v| {
            v.as_str() == Some("Alice")
        });
        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0], n1);
    }

    #[test]
    fn test_search_nodes_by_int_property() {
        let server = test_server();
        let (n1, _n2) = setup_graph_with_nodes(&server);

        let graph_arc = server.engine.get_graph("tg").unwrap();
        let gs = graph_arc.read();

        // Search for age=30 by integer match.
        let search_value = "30".to_string();
        let matching = gs.properties.nodes_where("age", &move |v| {
            if let Some(i) = v.as_int() {
                if let Ok(parsed) = search_value.parse::<i64>() {
                    return i == parsed;
                }
            }
            false
        });
        assert_eq!(matching.len(), 1);
        assert_eq!(matching[0], n1);
    }

    #[test]
    fn test_search_nodes_no_match() {
        let server = test_server();
        let _ = setup_graph_with_nodes(&server);

        let graph_arc = server.engine.get_graph("tg").unwrap();
        let gs = graph_arc.read();

        let matching = gs.properties.nodes_where("name", &|v| {
            v.as_str() == Some("NonExistent")
        });
        assert!(matching.is_empty());
    }

    #[test]
    fn test_get_neighbors() {
        let server = test_server();
        let (n1, n2) = setup_graph_with_nodes(&server);

        let graph_arc = server.engine.get_graph("tg").unwrap();
        let gs = graph_arc.read();

        // n1 should have one outgoing neighbor (n2).
        let neighbors = gs.adjacency.neighbors_both(n1, None);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, n2);

        // n2 should have one incoming neighbor (n1).
        let neighbors = gs.adjacency.neighbors_both(n2, None);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, n1);
    }

    #[test]
    fn test_get_neighbors_nonexistent_node() {
        let server = test_server();
        let _ = setup_graph_with_nodes(&server);

        let graph_arc = server.engine.get_graph("tg").unwrap();
        let gs = graph_arc.read();

        assert!(!gs.adjacency.has_node(999999));
    }

    #[test]
    fn test_export_graph() {
        let server = test_server();
        let _ = setup_graph_with_nodes(&server);

        let graph_arc = server.engine.get_graph("tg").unwrap();
        let gs = graph_arc.read();

        let node_ids = gs.adjacency.all_node_ids();
        assert_eq!(node_ids.len(), 2);

        let edge_count = gs.adjacency.all_edges().count();
        assert_eq!(edge_count, 1);
    }

    #[test]
    fn test_graph_stats() {
        let server = test_server();
        let _ = setup_graph_with_nodes(&server);

        let graph_arc = server.engine.get_graph("tg").unwrap();
        let gs = graph_arc.read();

        assert_eq!(gs.adjacency.node_count(), 2);
        assert_eq!(gs.adjacency.edge_count(), 1);
        assert_eq!(gs.vector_index.len(), 0);

        // Check label distribution.
        let node_ids = gs.adjacency.all_node_ids();
        let mut label_counts: std::collections::HashMap<String, u64> =
            std::collections::HashMap::new();
        for &nid in &node_ids {
            let label = gs
                .properties
                .get_node_property(nid, "_label")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            *label_counts.entry(label).or_insert(0) += 1;
        }
        assert_eq!(label_counts.get("Person"), Some(&1));
        assert_eq!(label_counts.get("Document"), Some(&1));
    }

    #[test]
    fn test_graph_stats_avg_degree() {
        let server = test_server();
        let _ = setup_graph_with_nodes(&server);

        let graph_arc = server.engine.get_graph("tg").unwrap();
        let gs = graph_arc.read();

        // 2 nodes, 1 edge. Each node sees 1 neighbor (n1->outgoing, n2->incoming).
        // Total degree = 2, avg = 2/2 = 1.0
        let node_ids = gs.adjacency.all_node_ids();
        let total_degree: u64 = node_ids
            .iter()
            .map(|&nid| gs.adjacency.neighbors_both(nid, None).len() as u64)
            .sum();
        let avg_degree = total_degree as f64 / gs.adjacency.node_count() as f64;
        assert!((avg_degree - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_node_update() {
        use weav_query::parser::{Command, GraphCreateCmd, NodeAddCmd, NodeGetCmd, NodeUpdateCmd};
        use weav_server::engine::CommandResponse;

        let server = test_server();
        server
            .engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd { name: "ug".into(), config: None }),
                None,
            )
            .unwrap();

        let nid = match server
            .engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "ug".into(), label: "Person".into(),
                    properties: vec![("name".into(), weav_core::types::Value::String("Alice".into()))],
                    embedding: None, entity_key: None, ttl_ms: None,
                }),
                None,
            )
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            other => panic!("unexpected: {other:?}"),
        };

        let resp = server
            .engine
            .execute_command(
                Command::NodeUpdate(NodeUpdateCmd {
                    graph: "ug".into(), node_id: nid,
                    properties: vec![("age".into(), weav_core::types::Value::Int(25))],
                    embedding: None,
                }),
                None,
            )
            .unwrap();
        assert!(matches!(resp, CommandResponse::Ok));

        let resp = server
            .engine
            .execute_command(
                Command::NodeGet(NodeGetCmd { graph: "ug".into(), node_id: Some(nid), entity_key: None }),
                None,
            )
            .unwrap();
        match resp {
            CommandResponse::NodeInfo(info) => {
                assert!(info.properties.iter().any(|(k, v)| k == "age" && v.as_int() == Some(25)));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_node_delete() {
        use weav_query::parser::{Command, GraphCreateCmd, NodeAddCmd, NodeDeleteCmd};
        use weav_server::engine::CommandResponse;

        let server = test_server();
        server
            .engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd { name: "dg".into(), config: None }),
                None,
            )
            .unwrap();

        let nid = match server
            .engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "dg".into(), label: "Temp".into(),
                    properties: vec![], embedding: None, entity_key: None, ttl_ms: None,
                }),
                None,
            )
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            other => panic!("unexpected: {other:?}"),
        };

        let resp = server
            .engine
            .execute_command(
                Command::NodeDelete(NodeDeleteCmd { graph: "dg".into(), node_id: nid }),
                None,
            )
            .unwrap();
        assert!(matches!(resp, CommandResponse::Ok));

        let graph_arc = server.engine.get_graph("dg").unwrap();
        let gs = graph_arc.read();
        assert!(!gs.adjacency.has_node(nid));
    }

    #[test]
    fn test_edge_get_and_delete() {
        use weav_query::parser::{Command, EdgeAddCmd, EdgeDeleteCmd, EdgeGetCmd, GraphCreateCmd, NodeAddCmd};
        use weav_server::engine::CommandResponse;

        let server = test_server();
        server
            .engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd { name: "eg".into(), config: None }),
                None,
            )
            .unwrap();

        let n1 = match server.engine.execute_command(
            Command::NodeAdd(NodeAddCmd {
                graph: "eg".into(), label: "A".into(),
                properties: vec![], embedding: None, entity_key: None, ttl_ms: None,
            }),
            None,
        ).unwrap() {
            CommandResponse::Integer(id) => id,
            other => panic!("unexpected: {other:?}"),
        };

        let n2 = match server.engine.execute_command(
            Command::NodeAdd(NodeAddCmd {
                graph: "eg".into(), label: "B".into(),
                properties: vec![], embedding: None, entity_key: None, ttl_ms: None,
            }),
            None,
        ).unwrap() {
            CommandResponse::Integer(id) => id,
            other => panic!("unexpected: {other:?}"),
        };

        let eid = match server.engine.execute_command(
            Command::EdgeAdd(EdgeAddCmd {
                graph: "eg".into(), source: n1, target: n2,
                label: "LINK".into(), weight: 0.5, properties: vec![], ttl_ms: None,
            }),
            None,
        ).unwrap() {
            CommandResponse::Integer(id) => id,
            other => panic!("unexpected: {other:?}"),
        };

        // Get the edge.
        match server.engine.execute_command(
            Command::EdgeGet(EdgeGetCmd { graph: "eg".into(), edge_id: eid }),
            None,
        ).unwrap() {
            CommandResponse::EdgeInfo(info) => {
                assert_eq!(info.edge_id, eid);
                assert_eq!(info.source, n1);
                assert_eq!(info.target, n2);
            }
            other => panic!("unexpected: {other:?}"),
        }

        // Delete the edge.
        let resp = server.engine.execute_command(
            Command::EdgeDelete(EdgeDeleteCmd { graph: "eg".into(), edge_id: eid }),
            None,
        ).unwrap();
        assert!(matches!(resp, CommandResponse::Ok));

        let graph_arc = server.engine.get_graph("eg").unwrap();
        let gs = graph_arc.read();
        assert_eq!(gs.adjacency.edge_count(), 0);
    }

    #[test]
    fn test_run_algorithm_pagerank() {
        let server = test_server();
        let _ = setup_graph_with_nodes(&server);
        let graph_arc = server.engine.get_graph("tg").unwrap();
        let gs = graph_arc.read();
        let scores = weav_graph::traversal::personalized_pagerank(
            &gs.adjacency, &[], 0.85, 100, 1e-6,
        );
        // With empty seeds, PageRank may return empty or all nodes.
        // Just verify it doesn't crash.
        assert!(scores.len() <= 2);
    }

    #[test]
    fn test_run_algorithm_communities() {
        let server = test_server();
        let _ = setup_graph_with_nodes(&server);
        let graph_arc = server.engine.get_graph("tg").unwrap();
        let gs = graph_arc.read();
        let communities = weav_graph::traversal::modularity_communities(&gs.adjacency, 100, 1.0);
        assert_eq!(communities.len(), 2);
    }

    #[test]
    fn test_run_algorithm_shortest_path() {
        let server = test_server();
        let (n1, n2) = setup_graph_with_nodes(&server);
        let graph_arc = server.engine.get_graph("tg").unwrap();
        let gs = graph_arc.read();
        let path = weav_graph::traversal::dijkstra_shortest_path(
            &gs.adjacency, n1, n2, u8::MAX,
        );
        assert!(path.is_some());
        assert_eq!(path.unwrap().nodes, vec![n1, n2]);
    }

    #[test]
    fn test_run_algorithm_betweenness() {
        let server = test_server();
        let _ = setup_graph_with_nodes(&server);
        let graph_arc = server.engine.get_graph("tg").unwrap();
        let gs = graph_arc.read();
        let scores = weav_graph::traversal::betweenness_centrality(
            &gs.adjacency, &weav_graph::traversal::EdgeFilter::none(),
        );
        assert_eq!(scores.len(), 2);
    }

    #[test]
    fn test_run_algorithm_triangle_count() {
        let server = test_server();
        let _ = setup_graph_with_nodes(&server);
        let graph_arc = server.engine.get_graph("tg").unwrap();
        let gs = graph_arc.read();
        let result = weav_graph::traversal::triangle_count(
            &gs.adjacency,
            &weav_graph::traversal::EdgeFilter::none(),
        );
        assert_eq!(result.total_triangles, 0);
    }

    #[test]
    fn test_run_algorithm_scc() {
        let server = test_server();
        let _ = setup_graph_with_nodes(&server);
        let graph_arc = server.engine.get_graph("tg").unwrap();
        let gs = graph_arc.read();
        let components = weav_graph::traversal::tarjan_scc(&gs.adjacency);
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_run_algorithm_topological_sort() {
        let server = test_server();
        let (n1, n2) = setup_graph_with_nodes(&server);
        let graph_arc = server.engine.get_graph("tg").unwrap();
        let gs = graph_arc.read();
        let order = weav_graph::traversal::topological_sort(&gs.adjacency).unwrap();
        let pos1 = order.iter().position(|&n| n == n1).unwrap();
        let pos2 = order.iter().position(|&n| n == n2).unwrap();
        assert!(pos1 < pos2);
    }

    #[test]
    fn test_total_tool_count() {
        let server = test_server();
        let router = &server.tool_router;
        let expected = [
            "graph_list", "graph_info", "graph_create", "graph_drop",
            "node_add", "node_get", "node_update", "node_delete",
            "edge_add", "edge_get", "edge_delete",
            "search_nodes", "get_neighbors", "export_graph", "graph_stats",
            "server_info", "context_query", "vector_search", "run_algorithm",
            "ingest_document", "schema_set", "schema_get",
            "config_set", "trigger_snapshot",
        ];
        for tool in &expected {
            assert!(router.has_route(tool), "missing tool: {tool}");
        }
    }

    #[test]
    fn test_schema_set_and_get() {
        use weav_query::parser::{Command, GraphCreateCmd, SchemaSetCmd, SchemaGetCmd};
        use weav_server::engine::CommandResponse;

        let server = test_server();
        server.engine.execute_command(
            Command::GraphCreate(GraphCreateCmd { name: "sg".into(), config: None }),
            None,
        ).unwrap();

        // Set a type constraint.
        let resp = server.engine.execute_command(
            Command::SchemaSet(SchemaSetCmd {
                graph: "sg".into(),
                target: "node".into(),
                label: "Person".into(),
                constraint_type: "type".into(),
                property: "age".into(),
                value_type: Some("int".into()),
            }),
            None,
        ).unwrap();
        assert!(matches!(resp, CommandResponse::Ok));

        // Get schema.
        let resp = server.engine.execute_command(
            Command::SchemaGet(SchemaGetCmd { graph: "sg".into() }),
            None,
        ).unwrap();
        match resp {
            CommandResponse::Text(json) => {
                assert!(json.contains("Person"));
                assert!(json.contains("age"));
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_config_set() {
        use weav_query::parser::Command;
        use weav_server::engine::CommandResponse;

        let server = test_server();
        let resp = server.engine.execute_command(
            Command::ConfigSet("test_key".into(), "test_value".into()),
            None,
        ).unwrap();
        assert!(matches!(resp, CommandResponse::Ok));

        let resp = server.engine.execute_command(
            Command::ConfigGet("test_key".into()),
            None,
        ).unwrap();
        match resp {
            CommandResponse::Text(v) => assert_eq!(v, "test_value"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn test_trigger_snapshot() {
        use weav_query::parser::Command;
        use weav_server::engine::CommandResponse;

        let server = test_server();
        let resp = server.engine.execute_command(Command::Snapshot, None).unwrap();
        assert!(matches!(resp, CommandResponse::Ok));
    }
}
