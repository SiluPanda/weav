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

use weav_core::types::Value;
use weav_query::parser::{
    Command, EdgeAddCmd, GraphCreateCmd, NodeAddCmd, NodeGetCmd,
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
            // Match against integer.
            if let Some(i) = v.as_int() {
                if let Ok(parsed) = search_value.parse::<i64>() {
                    return i == parsed;
                }
            }
            // Match against float.
            if let Some(f) = v.as_float() {
                if let Ok(parsed) = search_value.parse::<f64>() {
                    return (f - parsed).abs() < f64::EPSILON;
                }
            }
            // Match against bool.
            if let Some(b) = v.as_bool() {
                if let Ok(parsed) = search_value.parse::<bool>() {
                    return b == parsed;
                }
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
}

#[tool_handler]
impl ServerHandler for WeavMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("weav-mcp", env!("CARGO_PKG_VERSION")))
            .with_protocol_version(ProtocolVersion::V_2024_11_05)
            .with_instructions(
                "Weav is an in-memory context graph database for AI/LLM workloads. \
                 Use graph_list to see available graphs. \
                 Use graph_create to create a new graph. \
                 Use graph_info to get node/edge counts for a graph. \
                 Use node_add to add nodes with labels and properties. \
                 Use node_get to retrieve nodes by ID or entity_key. \
                 Use search_nodes to find nodes by property key/value. \
                 Use get_neighbors to find all neighbors of a node. \
                 Use edge_add to create edges between nodes. \
                 Use export_graph to export all nodes and edges as JSON. \
                 Use graph_stats for detailed statistics including label distribution. \
                 Use graph_drop to delete a graph."
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
        assert!(router.has_route("edge_add"));
        assert!(router.has_route("server_info"));
        assert!(router.has_route("search_nodes"));
        assert!(router.has_route("get_neighbors"));
        assert!(router.has_route("export_graph"));
        assert!(router.has_route("graph_stats"));
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
}
