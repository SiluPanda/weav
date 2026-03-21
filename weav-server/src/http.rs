//! HTTP REST API using axum.
//!
//! Translates JSON requests into engine `Command`s and engine responses back to JSON.

use std::sync::Arc;

use std::time::Duration;

use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use tower_http::timeout::TimeoutLayer;

use weav_core::types::{DecayFunction, Direction, TokenBudget, Value};
use weav_query::parser::{
    Command, ContextQuery, CypherCmd, EdgeAddCmd, EdgeDeleteCmd, EdgeFilterConfig,
    EdgeInvalidateCmd, GraphCreateCmd, NodeAddCmd, NodeDeleteCmd, NodeGetCmd, NodeUpdateCmd,
    BulkInsertNodesCmd, BulkInsertEdgesCmd, SeedStrategy, SortDirection, SortField, SortOrder,
};

use crate::engine::{CommandResponse, Engine};

// ─── JSON request / response types ──────────────────────────────────────────

#[derive(Deserialize)]
pub struct CreateGraphRequest {
    pub name: String,
}

#[derive(Deserialize)]
pub struct AddNodeRequest {
    pub label: String,
    pub properties: Option<serde_json::Value>,
    pub embedding: Option<Vec<f32>>,
    pub entity_key: Option<String>,
    /// Time-to-live in milliseconds. Node will be auto-expired after this duration.
    pub ttl_ms: Option<u64>,
}

#[derive(Deserialize)]
pub struct AddEdgeRequest {
    pub source: u64,
    pub target: u64,
    pub label: String,
    pub weight: Option<f32>,
    pub properties: Option<serde_json::Value>,
    /// Time-to-live in milliseconds. Edge will be auto-expired after this duration.
    pub ttl_ms: Option<u64>,
}

#[derive(Deserialize)]
pub struct UpdateNodeRequest {
    pub properties: Option<serde_json::Value>,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Deserialize)]
pub struct BulkAddNodesRequest {
    pub nodes: Vec<AddNodeRequest>,
}

#[derive(Deserialize)]
pub struct BulkAddEdgesRequest {
    pub edges: Vec<AddEdgeRequest>,
}

#[derive(Deserialize)]
pub struct CypherRequest {
    pub query: String,
}

#[derive(Deserialize)]
pub struct DecayRequest {
    #[serde(rename = "type")]
    pub decay_type: String,
    pub half_life_ms: Option<u64>,
    pub max_age_ms: Option<u64>,
    pub cutoff_ms: Option<u64>,
}

#[derive(Deserialize)]
pub struct ContextRequest {
    pub graph: String,
    pub query: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub seed_nodes: Option<Vec<String>>,
    pub budget: Option<u32>,
    pub max_depth: Option<u8>,
    pub include_provenance: Option<bool>,
    pub decay: Option<DecayRequest>,
    pub temporal_at: Option<u64>,
    pub limit: Option<u32>,
    pub sort_field: Option<String>,
    pub sort_direction: Option<String>,
    pub edge_labels: Option<Vec<String>>,
    pub direction: Option<String>,
    /// If true, return the query plan without executing.
    pub explain: Option<bool>,
    /// Named budget preset: "small"/"4k", "medium"/"8k", "large"/"16k",
    /// "xl"/"32k", "xxl"/"128k". Overrides `budget` when set.
    pub budget_preset: Option<String>,
    /// Output format: "raw" (default), "anthropic", "openai".
    /// When set, the result includes LLM-ready formatted messages.
    pub output_format: Option<String>,
    /// If true, include the subgraph structure (nodes + edges) in the result.
    pub include_subgraph: Option<bool>,
}

#[derive(Deserialize)]
pub struct SearchParams {
    pub key: String,
    pub value: String,
    pub limit: Option<u32>,
}

#[derive(Deserialize)]
pub struct TextSearchParams {
    #[serde(alias = "q")]
    pub query: String,
    pub limit: Option<u32>,
}

#[derive(Deserialize)]
pub struct TemporalQueryRequest {
    pub timestamp: Option<u64>,
    pub start_time: Option<u64>,
    pub end_time: Option<u64>,
    pub node_ids: Option<Vec<u64>>,
    pub include_edges: Option<bool>,
}

#[derive(Deserialize)]
pub struct BackupRequest {
    pub label: Option<String>,
}

#[derive(Serialize)]
pub struct ApiResponse<T: Serialize> {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl<T: Serialize> ApiResponse<T> {
    fn ok(data: T) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
        }
    }

    fn ok_empty() -> ApiResponse<()> {
        ApiResponse {
            success: true,
            data: None,
            error: None,
        }
    }

    fn err(msg: String) -> ApiResponse<()> {
        ApiResponse {
            success: false,
            data: None,
            error: Some(msg),
        }
    }
}

// ─── Serialization helpers ──────────────────────────────────────────────────

#[derive(Serialize)]
struct HealthResponse {
    status: String,
}

#[derive(Serialize)]
struct NodeInfoJson {
    node_id: u64,
    label: String,
    properties: serde_json::Value,
}

#[derive(Serialize)]
struct GraphInfoJson {
    name: String,
    node_count: u64,
    edge_count: u64,
    vector_count: usize,
    label_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    default_ttl_ms: Option<u64>,
}

#[derive(Serialize)]
struct NodeIdResponse {
    node_id: u64,
}

#[derive(Serialize)]
struct EdgeIdResponse {
    edge_id: u64,
}

#[derive(Serialize)]
struct BulkNodeIdsResponse {
    node_ids: Vec<u64>,
}

#[derive(Serialize)]
struct BulkEdgeIdsResponse {
    edge_ids: Vec<u64>,
}

#[derive(Serialize)]
struct NeighborEntry {
    node_id: u64,
    edge_id: u64,
    direction: String,
    edge_label: String,
    edge_weight: f32,
    node_label: String,
}

#[derive(Serialize)]
struct NeighborsResponse {
    node_id: u64,
    neighbors: Vec<NeighborEntry>,
}

#[derive(Serialize)]
struct EdgeDetailJson {
    edge_id: u64,
    source: u64,
    target: u64,
    label: String,
    weight: f32,
    properties: serde_json::Value,
}

// ─── Router construction ────────────────────────────────────────────────────

/// Build the axum Router with all routes.
pub fn build_router(engine: Arc<Engine>) -> Router {
    Router::new()
        .route("/health", get(health))
        // Graph routes.
        .route("/v1/graphs", post(create_graph))
        .route("/v1/graphs", get(list_graphs))
        .route("/v1/graphs/{name}", get(get_graph_info))
        .route("/v1/graphs/{name}", delete(drop_graph))
        .route("/v1/graphs/{graph}/check", get(check_graph))
        // Node routes.
        .route("/v1/graphs/{graph}/nodes", post(add_node))
        .route("/v1/graphs/{graph}/nodes/bulk", post(bulk_add_nodes))
        .route("/v1/graphs/{graph}/nodes/{id}", get(get_node))
        .route("/v1/graphs/{graph}/nodes/{id}", put(update_node))
        .route("/v1/graphs/{graph}/nodes/{id}", delete(delete_node))
        .route(
            "/v1/graphs/{graph}/nodes/{node_id}/neighbors",
            get(get_node_neighbors),
        )
        // Edge routes.
        .route("/v1/graphs/{graph}/edges", post(add_edge))
        .route("/v1/graphs/{graph}/edges/bulk", post(bulk_add_edges))
        .route("/v1/graphs/{graph}/edges/{id}", get(get_edge))
        .route("/v1/graphs/{graph}/edges/{id}", delete(delete_edge))
        .route(
            "/v1/graphs/{graph}/edges/{id}/invalidate",
            post(invalidate_edge),
        )
        // Snapshot.
        .route("/v1/snapshot", post(snapshot))
        // Backup (snapshot + WAL compact).
        .route("/v1/backup", post(backup))
        // WAL compact.
        .route("/v1/wal/compact", post(wal_compact))
        // Server info.
        .route("/v1/info", get(server_info))
        // Context query.
        .route("/v1/context", post(context_query))
        // Ingest (extraction pipeline).
        .route("/v1/graphs/{graph}/ingest", post(ingest))
        // Schema introspection.
        .route("/v1/graphs/{graph}/schema", get(get_graph_schema))
        // Detailed graph statistics.
        .route(
            "/v1/graphs/{graph}/stats/detailed",
            get(get_detailed_stats),
        )
        // Node search by property.
        .route("/v1/graphs/{graph}/search", get(search_nodes))
        // Full-text search with BM25 scoring.
        .route("/v1/graphs/{graph}/search/text", get(search_text))
        // Graph export/import.
        .route("/v1/graphs/{graph}/export", get(export_graph))
        .route("/v1/graphs/{graph}/import", post(import_graph))
        // CSV import/export.
        .route("/v1/graphs/{graph}/import/csv", post(import_csv))
        .route("/v1/graphs/{graph}/export/csv", get(export_csv))
        // DOT graph export.
        .route("/v1/graphs/{graph}/export/dot", get(export_dot))
        // Graph algorithms.
        .route(
            "/v1/graphs/{graph}/algorithms/{algorithm}",
            post(run_algorithm),
        )
        // Temporal query.
        .route(
            "/v1/graphs/{graph}/temporal",
            post(temporal_query),
        )
        // Node history.
        .route(
            "/v1/graphs/{graph}/nodes/{id}/history",
            get(node_history),
        )
        // Node importance scoring.
        .route(
            "/v1/graphs/{graph}/nodes/{id}/importance",
            get(get_node_importance),
        )
        // Top-N nodes by importance.
        .route(
            "/v1/graphs/{graph}/importance",
            get(get_graph_importance),
        )
        // Graph condensation.
        .route(
            "/v1/graphs/{graph}/condense",
            post(condense_graph),
        )
        // Cypher query.
        .route(
            "/v1/graphs/{graph}/cypher",
            post(cypher_query),
        )
        // Prometheus metrics.
        .route("/metrics", get(metrics_handler))
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024)) // 10 MB
        .layer(TimeoutLayer::with_status_code(StatusCode::REQUEST_TIMEOUT, Duration::from_secs(30)))
        .with_state(engine)
}

// ─── Helper to map WeavError to HTTP responses ─────────────────────────────

fn weav_error_to_response(err: weav_core::error::WeavError) -> impl IntoResponse {
    let status = match &err {
        weav_core::error::WeavError::GraphNotFound(_) => StatusCode::NOT_FOUND,
        weav_core::error::WeavError::NodeNotFound(_, _) => StatusCode::NOT_FOUND,
        weav_core::error::WeavError::EdgeNotFound(_) => StatusCode::NOT_FOUND,
        weav_core::error::WeavError::DuplicateNode(_) => StatusCode::CONFLICT,
        weav_core::error::WeavError::Conflict(_) => StatusCode::CONFLICT,
        weav_core::error::WeavError::QueryParseError(_) => StatusCode::BAD_REQUEST,
        weav_core::error::WeavError::DimensionMismatch { .. } => StatusCode::BAD_REQUEST,
        weav_core::error::WeavError::InvalidConfig(_) => StatusCode::BAD_REQUEST,
        weav_core::error::WeavError::AuthenticationRequired => StatusCode::UNAUTHORIZED,
        weav_core::error::WeavError::AuthenticationFailed(_) => StatusCode::UNAUTHORIZED,
        weav_core::error::WeavError::PermissionDenied(_) => StatusCode::FORBIDDEN,
        weav_core::error::WeavError::ExtractionNotEnabled => StatusCode::SERVICE_UNAVAILABLE,
        weav_core::error::WeavError::DocumentParseError(_) => StatusCode::BAD_REQUEST,
        weav_core::error::WeavError::LlmError(_) => StatusCode::BAD_GATEWAY,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, Json(ApiResponse::<()>::err(err.to_string())))
}

// ─── Auth extraction ────────────────────────────────────────────────────────

/// Extract session identity from the Authorization header.
///
/// Supports two schemes:
///   - `Authorization: Bearer <api_key>` → API key auth
///   - `Authorization: Basic <base64(user:pass)>` → username/password auth
///
/// Returns `None` when auth is disabled or no header is present.
fn extract_identity(
    engine: &Engine,
    headers: &HeaderMap,
) -> Option<weav_auth::identity::SessionIdentity> {
    if !engine.is_auth_enabled() {
        return None;
    }
    let auth_header = headers.get("authorization")?.to_str().ok()?;

    if let Some(token) = auth_header.strip_prefix("Bearer ") {
        engine.authenticate_api_key(token.trim()).ok()
    } else if let Some(basic) = auth_header.strip_prefix("Basic ") {
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(basic.trim())
            .ok()?;
        let creds = String::from_utf8(decoded).ok()?;
        let (user, pass) = creds.split_once(':')?;
        engine.authenticate(user, pass).ok()
    } else {
        None
    }
}

/// Extract group_id from X-Group-Id header for multi-tenancy.
/// When set, node operations store it as `_group_id` property,
/// and search/query operations filter results to the specified group.
fn extract_group_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-group-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

// ─── Handlers ───────────────────────────────────────────────────────────────

async fn health() -> impl IntoResponse {
    Json(ApiResponse::ok(HealthResponse {
        status: "ok".to_string(),
    }))
}

/// Schema introspection: discover node labels, edge labels, and property keys.
/// GET /v1/graphs/{graph}/schema
async fn get_graph_schema(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
) -> impl IntoResponse {
    let _identity = extract_identity(&engine, &headers);
    let graph_arc = match engine.get_graph(&graph) {
        Ok(g) => g,
        Err(e) => return weav_error_to_response(e).into_response(),
    };
    let gs = graph_arc.read();

    // Collect node labels and their property keys
    let mut node_labels: std::collections::HashMap<String, std::collections::HashSet<String>> =
        std::collections::HashMap::new();
    for nid in gs.adjacency.all_node_ids() {
        let label = gs.properties
            .get_node_property(nid, "_label")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let entry = node_labels.entry(label).or_default();
        for (k, _) in gs.properties.get_all_node_properties(nid) {
            if !k.starts_with('_') {
                entry.insert(k.to_string());
            }
        }
    }

    // Collect edge labels and (source_label, target_label) pairs
    let mut edge_labels: std::collections::HashMap<String, Vec<(String, String)>> =
        std::collections::HashMap::new();
    for (_, meta) in gs.adjacency.all_edges() {
        let label = gs.interner
            .resolve_label(meta.label)
            .unwrap_or("unknown")
            .to_string();
        let src_label = gs.properties
            .get_node_property(meta.source, "_label")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let tgt_label = gs.properties
            .get_node_property(meta.target, "_label")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let entry = edge_labels.entry(label).or_default();
        let pair = (src_label, tgt_label);
        if !entry.contains(&pair) {
            entry.push(pair);
        }
    }

    // Format output
    let node_types: Vec<serde_json::Value> = node_labels.iter().map(|(label, keys)| {
        let mut sorted_keys: Vec<&String> = keys.iter().collect();
        sorted_keys.sort();
        serde_json::json!({
            "label": label,
            "count": gs.properties.nodes_where("_label", &|v| v.as_str() == Some(label)).len(),
            "properties": sorted_keys,
        })
    }).collect();

    let edge_types: Vec<serde_json::Value> = edge_labels.iter().map(|(label, pairs)| {
        serde_json::json!({
            "label": label,
            "connections": pairs.iter().map(|(s, t)| {
                serde_json::json!({"from": s, "to": t})
            }).collect::<Vec<_>>(),
        })
    }).collect();

    (
        StatusCode::OK,
        Json(ApiResponse::ok(serde_json::json!({
            "graph": graph,
            "node_types": node_types,
            "edge_types": edge_types,
        }))),
    ).into_response()
}

/// Search nodes by property key/value.
/// GET /v1/graphs/{graph}/search?key=name&value=Alice&limit=100
async fn search_nodes(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
    Query(params): Query<SearchParams>,
) -> impl IntoResponse {
    let _identity = extract_identity(&engine, &headers);
    let graph_arc = match engine.get_graph(&graph) {
        Ok(g) => g,
        Err(e) => return weav_error_to_response(e).into_response(),
    };
    let gs = graph_arc.read();

    let limit = params.limit.unwrap_or(100) as usize;

    // Use secondary index if available for O(1) lookup, otherwise fall back to O(n) scan.
    let matching_nodes: Vec<u64> = if gs.properties.index.has_index(&params.key) {
        gs.properties.index.lookup(&params.key, &params.value)
    } else {
        let value_clone = params.value.clone();
        gs.properties.nodes_where(&params.key, &move |v| {
            match v {
                Value::String(s) => s.as_str() == value_clone,
                Value::Int(i) => i.to_string() == value_clone,
                Value::Float(f) => f.to_string() == value_clone,
                Value::Bool(b) => b.to_string() == value_clone,
                _ => false,
            }
        })
    };

    let results: Vec<serde_json::Value> = matching_nodes.iter().take(limit).map(|&nid| {
        let label = gs.properties
            .get_node_property(nid, "_label")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let props = gs.properties.get_all_node_properties(nid);
        let mut props_map = serde_json::Map::new();
        for (k, v) in props {
            if !k.starts_with('_') {
                props_map.insert(k.to_string(), value_to_json(v));
            }
        }
        serde_json::json!({
            "node_id": nid,
            "label": label,
            "properties": props_map,
        })
    }).collect();

    (
        StatusCode::OK,
        Json(ApiResponse::ok(serde_json::json!({
            "matches": results,
            "total": matching_nodes.len(),
            "limit": limit,
        }))),
    ).into_response()
}

/// Full-text search using BM25 scoring.
/// GET /v1/graphs/{graph}/search/text?query=...&limit=20
async fn search_text(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
    Query(params): Query<TextSearchParams>,
) -> impl IntoResponse {
    let _identity = extract_identity(&engine, &headers);
    let graph_arc = match engine.get_graph(&graph) {
        Ok(g) => g,
        Err(e) => return weav_error_to_response(e).into_response(),
    };
    let gs = graph_arc.read();

    let limit = params.limit.unwrap_or(20) as usize;
    let results = gs.text_index.search(&params.query, limit);

    let matches: Vec<serde_json::Value> = results.iter().map(|&(nid, score)| {
        let label = gs.properties
            .get_node_property(nid, "_label")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let props = gs.properties.get_all_node_properties(nid);
        let mut props_map = serde_json::Map::new();
        for (k, v) in props {
            if !k.starts_with('_') {
                props_map.insert(k.to_string(), value_to_json(v));
            }
        }
        serde_json::json!({
            "node_id": nid,
            "label": label,
            "score": score,
            "properties": props_map,
        })
    }).collect();

    (
        StatusCode::OK,
        Json(ApiResponse::ok(serde_json::json!({
            "matches": matches,
            "total": results.len(),
            "limit": limit,
        }))),
    ).into_response()
}

/// Export an entire graph as JSON.
/// GET /v1/graphs/{graph}/export
async fn export_graph(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
) -> impl IntoResponse {
    let _identity = extract_identity(&engine, &headers);
    let graph_arc = match engine.get_graph(&graph) {
        Ok(g) => g,
        Err(e) => return weav_error_to_response(e).into_response(),
    };
    let gs = graph_arc.read();

    // Export all nodes
    let nodes: Vec<serde_json::Value> = gs.adjacency.all_node_ids().iter().map(|&nid| {
        let label = gs.properties
            .get_node_property(nid, "_label")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let entity_key = gs.properties
            .get_node_property(nid, "entity_key")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let props = gs.properties.get_all_node_properties(nid);
        let mut props_map = serde_json::Map::new();
        for (k, v) in props {
            if !k.starts_with('_') && k != "entity_key" {
                props_map.insert(k.to_string(), value_to_json(v));
            }
        }
        let mut node = serde_json::json!({
            "node_id": nid,
            "label": label,
            "properties": props_map,
        });
        if let Some(key) = entity_key {
            node["entity_key"] = serde_json::Value::String(key);
        }
        if let Some(vec) = gs.vector_index.get_vector(nid) {
            node["embedding"] = serde_json::json!(vec);
        }
        node
    }).collect();

    // Export all edges
    let edges: Vec<serde_json::Value> = gs.adjacency.all_edges().map(|(eid, meta)| {
        let label = gs.interner
            .resolve_label(meta.label)
            .unwrap_or("unknown")
            .to_string();
        let edge_props = gs.properties.get_all_edge_properties(eid);
        let mut props_map = serde_json::Map::new();
        for (k, v) in edge_props {
            props_map.insert(k.to_string(), value_to_json(v));
        }
        serde_json::json!({
            "edge_id": eid,
            "source": meta.source,
            "target": meta.target,
            "label": label,
            "weight": meta.weight,
            "valid_from": meta.temporal.valid_from,
            "valid_until": meta.temporal.valid_until,
            "properties": props_map,
        })
    }).collect();

    (
        StatusCode::OK,
        Json(ApiResponse::ok(serde_json::json!({
            "graph": graph,
            "node_count": nodes.len(),
            "edge_count": edges.len(),
            "nodes": nodes,
            "edges": edges,
        }))),
    ).into_response()
}

/// Import a graph from JSON (same format as export).
/// POST /v1/graphs/{graph}/import
/// Creates the graph if it doesn't exist, then adds all nodes and edges.
async fn import_graph(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);

    // Create graph if it doesn't exist
    let create_cmd = Command::GraphCreate(GraphCreateCmd {
        name: graph.clone(),
        config: None,
    });
    let _ = engine.execute_command(create_cmd, identity.as_ref());

    let mut nodes_imported: u64 = 0;
    let mut edges_imported: u64 = 0;
    let mut errors: Vec<String> = Vec::new();

    // Map from exported node_id to new node_id (in case IDs need remapping)
    let mut id_map: std::collections::HashMap<u64, u64> = std::collections::HashMap::new();

    // Import nodes
    if let Some(nodes) = body.get("nodes").and_then(|v| v.as_array()) {
        for node_json in nodes {
            let label = node_json.get("label").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
            let entity_key = node_json.get("entity_key").and_then(|v| v.as_str()).map(|s| s.to_string());
            let old_id = node_json.get("node_id").and_then(|v| v.as_u64()).unwrap_or(0);

            let properties = if let Some(props) = node_json.get("properties") {
                json_to_props(props)
            } else {
                Vec::new()
            };

            let embedding = node_json.get("embedding").and_then(|v| {
                v.as_array().map(|arr| {
                    arr.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect::<Vec<f32>>()
                })
            });

            let cmd = Command::NodeAdd(NodeAddCmd {
                graph: graph.clone(),
                label,
                properties,
                embedding,
                entity_key,
                ttl_ms: None,
            });

            match engine.execute_command(cmd, identity.as_ref()) {
                Ok(CommandResponse::Integer(new_id)) => {
                    id_map.insert(old_id, new_id);
                    nodes_imported += 1;
                }
                Err(e) => errors.push(format!("node {old_id}: {e}")),
                _ => {}
            }
        }
    }

    // Import edges (remap source/target IDs)
    if let Some(edges) = body.get("edges").and_then(|v| v.as_array()) {
        for edge_json in edges {
            let old_source = edge_json.get("source").and_then(|v| v.as_u64()).unwrap_or(0);
            let old_target = edge_json.get("target").and_then(|v| v.as_u64()).unwrap_or(0);

            let source = id_map.get(&old_source).copied().unwrap_or(old_source);
            let target = id_map.get(&old_target).copied().unwrap_or(old_target);

            let label = edge_json.get("label").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
            let weight = edge_json.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;

            let properties = if let Some(props) = edge_json.get("properties") {
                json_to_props(props)
            } else {
                Vec::new()
            };

            let cmd = Command::EdgeAdd(EdgeAddCmd {
                graph: graph.clone(),
                source,
                target,
                label,
                weight,
                properties,
                ttl_ms: None,
            });

            match engine.execute_command(cmd, identity.as_ref()) {
                Ok(CommandResponse::Integer(_)) => edges_imported += 1,
                Err(e) => errors.push(format!("edge {old_source}->{old_target}: {e}")),
                _ => {}
            }
        }
    }

    let mut response = serde_json::json!({
        "nodes_imported": nodes_imported,
        "edges_imported": edges_imported,
    });
    if !errors.is_empty() {
        response["errors"] = serde_json::json!(errors);
    }

    (StatusCode::OK, Json(ApiResponse::ok(response))).into_response()
}

/// Import nodes from a CSV body.
/// POST /v1/graphs/{graph}/import/csv
///
/// The CSV must have a header row. Required column: `_label` (node label).
/// Optional column: `_id` (ignored). All other columns become node properties.
async fn import_csv(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
    body: String,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);

    // Ensure graph exists (create if not)
    let create_cmd = Command::GraphCreate(GraphCreateCmd {
        name: graph.clone(),
        config: None,
    });
    let _ = engine.execute_command(create_cmd, identity.as_ref());

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(body.as_bytes());

    let csv_headers = match reader.headers() {
        Ok(h) => h.clone(),
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse::<()>::err(format!("invalid CSV headers: {e}"))),
            )
                .into_response();
        }
    };

    // Find the _label column index
    let label_idx = match csv_headers.iter().position(|h| h == "_label") {
        Some(idx) => idx,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse::<()>::err(
                    "CSV must contain a '_label' column".to_string(),
                )),
            )
                .into_response();
        }
    };

    let mut nodes_created: u64 = 0;
    let mut errors: Vec<String> = Vec::new();

    for (row_idx, result) in reader.records().enumerate() {
        let record = match result {
            Ok(r) => r,
            Err(e) => {
                errors.push(format!("row {}: {e}", row_idx + 2));
                continue;
            }
        };

        let label = record.get(label_idx).unwrap_or("unknown").to_string();

        let mut properties: Vec<(String, Value)> = Vec::new();
        for (col_idx, header) in csv_headers.iter().enumerate() {
            if header == "_label" || header == "_id" {
                continue;
            }
            if let Some(val) = record.get(col_idx) {
                if !val.is_empty() {
                    // Try to parse as number, then bool, else string
                    let value = if let Ok(i) = val.parse::<i64>() {
                        Value::Int(i)
                    } else if let Ok(f) = val.parse::<f64>() {
                        Value::Float(f)
                    } else if val.eq_ignore_ascii_case("true") {
                        Value::Bool(true)
                    } else if val.eq_ignore_ascii_case("false") {
                        Value::Bool(false)
                    } else {
                        Value::String(compact_str::CompactString::from(val))
                    };
                    properties.push((header.to_string(), value));
                }
            }
        }

        let cmd = Command::NodeAdd(NodeAddCmd {
            graph: graph.clone(),
            label,
            properties,
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });

        match engine.execute_command(cmd, identity.as_ref()) {
            Ok(CommandResponse::Integer(_)) => nodes_created += 1,
            Err(e) => errors.push(format!("row {}: {e}", row_idx + 2)),
            _ => {}
        }
    }

    let mut response = serde_json::json!({ "nodes_created": nodes_created });
    if !errors.is_empty() {
        response["errors"] = serde_json::json!(errors);
    }

    (StatusCode::OK, Json(ApiResponse::ok(response))).into_response()
}

/// Export all nodes as CSV.
/// GET /v1/graphs/{graph}/export/csv
async fn export_csv(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
) -> impl IntoResponse {
    let _identity = extract_identity(&engine, &headers);
    let graph_arc = match engine.get_graph(&graph) {
        Ok(g) => g,
        Err(e) => return weav_error_to_response(e).into_response(),
    };
    let gs = graph_arc.read();

    let node_ids = gs.adjacency.all_node_ids();
    if node_ids.is_empty() {
        return (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "text/csv")],
            String::new(),
        )
            .into_response();
    }

    // Collect all unique property keys across all nodes (excluding internal keys).
    let mut all_keys: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for &nid in &node_ids {
        for (k, _) in gs.properties.get_all_node_properties(nid) {
            if !k.starts_with('_') && k != "entity_key" {
                all_keys.insert(k.to_string());
            }
        }
    }

    let mut wtr = csv::Writer::from_writer(Vec::new());

    // Write header: _label, then sorted property keys
    let mut header_row: Vec<String> = vec!["_label".to_string()];
    header_row.extend(all_keys.iter().cloned());
    wtr.write_record(&header_row).unwrap();

    // Write data rows
    for &nid in &node_ids {
        let label = gs
            .properties
            .get_node_property(nid, "_label")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let mut row: Vec<String> = vec![label];
        for key in &all_keys {
            let val = gs
                .properties
                .get_node_property(nid, key)
                .map(value_to_csv_string)
                .unwrap_or_default();
            row.push(val);
        }
        wtr.write_record(&row).unwrap();
    }

    let csv_bytes = wtr.into_inner().unwrap();
    let csv_string = String::from_utf8(csv_bytes).unwrap_or_default();

    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "text/csv")],
        csv_string,
    )
        .into_response()
}

/// Convert a Value to a CSV-friendly string.
fn value_to_csv_string(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::String(s) => s.to_string(),
        Value::Timestamp(ts) => ts.to_string(),
        other => serde_json::to_string(&value_to_json(other)).unwrap_or_default(),
    }
}

/// Export graph in DOT (Graphviz) format.
/// GET /v1/graphs/{graph}/export/dot
async fn export_dot(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
) -> impl IntoResponse {
    let _identity = extract_identity(&engine, &headers);
    let graph_arc = match engine.get_graph(&graph) {
        Ok(g) => g,
        Err(e) => return weav_error_to_response(e).into_response(),
    };
    let gs = graph_arc.read();

    let mut dot = String::from("digraph G {\n");
    dot.push_str("  rankdir=LR;\n");
    dot.push_str("  node [shape=record];\n\n");

    // Nodes
    for &nid in &gs.adjacency.all_node_ids() {
        let label = gs
            .properties
            .get_node_property(nid, "_label")
            .and_then(|v| v.as_str())
            .unwrap_or("node");
        let name = gs
            .properties
            .get_node_property(nid, "name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        // Escape double quotes in label and name for DOT format
        let label_escaped = label.replace('"', "\\\"");
        let name_escaped = name.replace('"', "\\\"");
        dot.push_str(&format!(
            "  n{nid} [label=\"{label_escaped}: {name_escaped}\"];\n"
        ));
    }

    dot.push('\n');

    // Edges
    for (eid, meta) in gs.adjacency.all_edges() {
        let edge_label = gs
            .interner
            .resolve_label(meta.label)
            .unwrap_or("unknown");
        let edge_label_escaped = edge_label.replace('"', "\\\"");
        dot.push_str(&format!(
            "  n{} -> n{} [label=\"{}\" id=\"e{}\"];\n",
            meta.source, meta.target, edge_label_escaped, eid
        ));
    }

    dot.push_str("}\n");

    (
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "text/vnd.graphviz")],
        dot,
    )
        .into_response()
}

/// Temporal query: return graph state at a point in time or during a time range.
/// POST /v1/graphs/{graph}/temporal
///
/// Point-in-time: `{ "timestamp": 12345 }` — returns entities valid at that instant.
/// Range: `{ "start_time": 100, "end_time": 500 }` — returns entities valid at
///   any point during the half-open interval [start_time, end_time).
async fn temporal_query(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
    Json(body): Json<TemporalQueryRequest>,
) -> impl IntoResponse {
    let _identity = extract_identity(&engine, &headers);
    let graph_arc = match engine.get_graph(&graph) {
        Ok(g) => g,
        Err(e) => return weav_error_to_response(e).into_response(),
    };
    let gs = graph_arc.read();

    let include_edges = body.include_edges.unwrap_or(true);

    // Determine temporal mode: range [start, end) or point-in-time.
    let is_range = body.start_time.is_some() && body.end_time.is_some();
    let range_start = body.start_time.unwrap_or(0);
    let range_end = body.end_time.unwrap_or(0);
    let timestamp = body.timestamp.unwrap_or(0);

    if !is_range && body.timestamp.is_none() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::<()>::err(
                "either 'timestamp' or both 'start_time' and 'end_time' are required".to_string(),
            )),
        )
            .into_response();
    }

    // Determine which nodes to inspect
    let node_ids: Vec<u64> = match body.node_ids {
        Some(ids) => ids,
        None => gs.adjacency.all_node_ids(),
    };

    // Filter nodes valid at the given timestamp or during the range
    let mut result_nodes: Vec<serde_json::Value> = Vec::new();
    let mut result_edges: Vec<serde_json::Value> = Vec::new();
    let mut seen_edges = std::collections::HashSet::new();

    for &nid in &node_ids {
        if !gs.adjacency.has_node(nid) {
            continue;
        }

        // Check node temporal validity via _tx_from and _ttl_expires_at properties
        let created_at = gs
            .properties
            .get_node_property(nid, "_tx_from")
            .and_then(|v| match v {
                Value::Int(ts) => Some(*ts as u64),
                Value::Timestamp(ts) => Some(*ts),
                _ => None,
            });
        let expires_at = gs
            .properties
            .get_node_property(nid, "_ttl_expires_at")
            .and_then(|v| match v {
                Value::Timestamp(ts) => Some(*ts),
                Value::Int(ts) => Some(*ts as u64),
                _ => None,
            });

        // Construct the node's effective valid window: [created_at, expires_at)
        let node_valid_from = created_at.unwrap_or(0);
        let node_valid_until = expires_at.unwrap_or(u64::MAX);

        let node_passes = if is_range {
            // Overlap check: node_valid_from < range_end && node_valid_until > range_start
            node_valid_from < range_end && node_valid_until > range_start
        } else {
            // Point-in-time: created_at <= timestamp < expires_at
            node_valid_from <= timestamp && timestamp < node_valid_until
        };

        if !node_passes {
            continue;
        }

        // Build node JSON
        let label = gs
            .properties
            .get_node_property(nid, "_label")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let props = gs.properties.get_all_node_properties(nid);
        let mut props_map = serde_json::Map::new();
        for (k, v) in props {
            if !k.starts_with('_') {
                props_map.insert(k.to_string(), value_to_json(v));
            }
        }

        result_nodes.push(serde_json::json!({
            "node_id": nid,
            "label": label,
            "properties": props_map,
        }));

        // Collect edges valid at this timestamp or during the range
        if include_edges {
            if is_range {
                // For range queries, iterate all outgoing edges and check overlap
                let all_neighbors = gs.adjacency.neighbors_out(nid, None);
                for (neighbor_id, edge_id) in all_neighbors {
                    if seen_edges.insert(edge_id)
                        && let Some(meta) = gs.adjacency.get_edge(edge_id)
                        && meta.temporal.is_valid_during(range_start, range_end)
                    {
                        let edge_label = gs
                            .interner
                            .resolve_label(meta.label)
                            .unwrap_or("unknown")
                            .to_string();
                        result_edges.push(serde_json::json!({
                            "edge_id": edge_id,
                            "source": meta.source,
                            "target": meta.target,
                            "label": edge_label,
                            "weight": meta.weight,
                            "neighbor": neighbor_id,
                            "valid_from": meta.temporal.valid_from,
                            "valid_until": if meta.temporal.valid_until == u64::MAX {
                                serde_json::Value::Null
                            } else {
                                serde_json::json!(meta.temporal.valid_until)
                            },
                        }));
                    }
                }
            } else {
                let temporal_neighbors = gs.adjacency.neighbors_at(nid, timestamp, None);
                for (neighbor_id, edge_id) in temporal_neighbors {
                    if seen_edges.insert(edge_id)
                        && let Some(meta) = gs.adjacency.get_edge(edge_id)
                    {
                        let edge_label = gs
                            .interner
                            .resolve_label(meta.label)
                            .unwrap_or("unknown")
                            .to_string();
                        result_edges.push(serde_json::json!({
                            "edge_id": edge_id,
                            "source": meta.source,
                            "target": meta.target,
                            "label": edge_label,
                            "weight": meta.weight,
                            "neighbor": neighbor_id,
                            "valid_from": meta.temporal.valid_from,
                            "valid_until": if meta.temporal.valid_until == u64::MAX {
                                serde_json::Value::Null
                            } else {
                                serde_json::json!(meta.temporal.valid_until)
                            },
                        }));
                    }
                }
            }
        }
    }

    let time_info = if is_range {
        serde_json::json!({
            "start_time": range_start,
            "end_time": range_end,
        })
    } else {
        serde_json::json!({ "timestamp": timestamp })
    };

    (
        StatusCode::OK,
        Json(ApiResponse::ok(serde_json::json!({
            "graph": graph,
            "query": time_info,
            "node_count": result_nodes.len(),
            "edge_count": result_edges.len(),
            "nodes": result_nodes,
            "edges": result_edges,
        }))),
    )
        .into_response()
}

/// Node history: return all temporal versions of a node's data.
/// GET /v1/graphs/{graph}/nodes/{id}/history
async fn node_history(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path((graph, node_id)): Path<(String, u64)>,
) -> impl IntoResponse {
    let _identity = extract_identity(&engine, &headers);
    let graph_arc = match engine.get_graph(&graph) {
        Ok(g) => g,
        Err(e) => return weav_error_to_response(e).into_response(),
    };
    let gs = graph_arc.read();

    if !gs.adjacency.has_node(node_id) {
        return weav_error_to_response(weav_core::error::WeavError::NodeNotFound(
            node_id,
            gs.graph_id,
        ))
        .into_response();
    }

    // Node's current properties
    let label = gs
        .properties
        .get_node_property(node_id, "_label")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let props = gs.properties.get_all_node_properties(node_id);
    let mut props_map = serde_json::Map::new();
    for (k, v) in &props {
        if !k.starts_with('_') {
            props_map.insert(k.to_string(), value_to_json(v));
        }
    }

    // Node temporal metadata from stored properties
    let created_at = gs
        .properties
        .get_node_property(node_id, "_tx_from")
        .and_then(|v| match v {
            Value::Int(ts) => Some(*ts as u64),
            Value::Timestamp(ts) => Some(*ts),
            _ => None,
        });
    let expires_at = gs
        .properties
        .get_node_property(node_id, "_ttl_expires_at")
        .and_then(|v| match v {
            Value::Timestamp(ts) => Some(*ts),
            Value::Int(ts) => Some(*ts as u64),
            _ => None,
        });

    let temporal = serde_json::json!({
        "valid_from": created_at.unwrap_or(0),
        "valid_until": expires_at.map(serde_json::Value::from).unwrap_or(serde_json::Value::Null),
        "tx_from": created_at.unwrap_or(0),
        "tx_until": serde_json::Value::Null,
    });

    // All edges connected to this node (outgoing and incoming), including invalidated ones
    let mut edges: Vec<serde_json::Value> = Vec::new();
    for (edge_id, meta) in gs.adjacency.all_edges() {
        if meta.source != node_id && meta.target != node_id {
            continue;
        }
        let edge_label = gs
            .interner
            .resolve_label(meta.label)
            .unwrap_or("unknown")
            .to_string();
        let is_active = meta.temporal.valid_until == u64::MAX;
        edges.push(serde_json::json!({
            "edge_id": edge_id,
            "source": meta.source,
            "target": meta.target,
            "label": edge_label,
            "weight": meta.weight,
            "active": is_active,
            "temporal": {
                "valid_from": meta.temporal.valid_from,
                "valid_until": if meta.temporal.valid_until == u64::MAX {
                    serde_json::Value::Null
                } else {
                    serde_json::json!(meta.temporal.valid_until)
                },
                "tx_from": meta.temporal.tx_from,
                "tx_until": if meta.temporal.tx_until == u64::MAX {
                    serde_json::Value::Null
                } else {
                    serde_json::json!(meta.temporal.tx_until)
                },
            },
        }));
    }

    (
        StatusCode::OK,
        Json(ApiResponse::ok(serde_json::json!({
            "node_id": node_id,
            "graph": graph,
            "label": label,
            "properties": props_map,
            "temporal": temporal,
            "edges": edges,
            "edge_count": edges.len(),
        }))),
    )
        .into_response()
}

/// Node importance score.
/// GET /v1/graphs/{graph}/nodes/{id}/importance
async fn get_node_importance(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path((graph, node_id)): Path<(String, u64)>,
) -> impl IntoResponse {
    let _identity = extract_identity(&engine, &headers);
    let graph_arc = match engine.get_graph(&graph) {
        Ok(g) => g,
        Err(e) => return weav_error_to_response(e).into_response(),
    };
    let gs = graph_arc.read();

    if !gs.adjacency.has_node(node_id) {
        return weav_error_to_response(weav_core::error::WeavError::NodeNotFound(
            node_id,
            gs.graph_id,
        ))
        .into_response();
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let score = crate::engine::compute_node_importance(&gs, node_id, now);

    (
        StatusCode::OK,
        Json(ApiResponse::ok(serde_json::json!({
            "node_id": score.node_id,
            "importance": score.importance,
            "structural": score.structural,
            "recency": score.recency,
            "access": score.access,
        }))),
    )
        .into_response()
}

/// Top-N nodes by importance.
/// GET /v1/graphs/{graph}/importance?limit=10
#[derive(Deserialize)]
pub struct ImportanceParams {
    pub limit: Option<usize>,
}

async fn get_graph_importance(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
    Query(params): Query<ImportanceParams>,
) -> impl IntoResponse {
    let _identity = extract_identity(&engine, &headers);
    let graph_arc = match engine.get_graph(&graph) {
        Ok(g) => g,
        Err(e) => return weav_error_to_response(e).into_response(),
    };
    let gs = graph_arc.read();

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let limit = params.limit.unwrap_or(10);

    let mut scores: Vec<crate::engine::NodeImportanceScore> = gs
        .adjacency
        .all_node_ids()
        .iter()
        .map(|&nid| crate::engine::compute_node_importance(&gs, nid, now))
        .collect();

    scores.sort_by(|a, b| {
        b.importance
            .partial_cmp(&a.importance)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.node_id.cmp(&b.node_id))
    });
    scores.truncate(limit);

    let results: Vec<serde_json::Value> = scores
        .iter()
        .map(|s| {
            serde_json::json!({
                "node_id": s.node_id,
                "importance": s.importance,
                "structural": s.structural,
                "recency": s.recency,
                "access": s.access,
            })
        })
        .collect();

    (
        StatusCode::OK,
        Json(ApiResponse::ok(serde_json::json!({
            "graph": graph,
            "limit": limit,
            "nodes": results,
        }))),
    )
        .into_response()
}

// ─── Graph condensation ─────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CondenseRequest {
    pub importance_threshold: f32,
}

async fn condense_graph(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
    Json(body): Json<CondenseRequest>,
) -> impl IntoResponse {
    let _identity = extract_identity(&engine, &headers);
    match engine.handle_condense(&graph, body.importance_threshold) {
        Ok(CommandResponse::Text(msg)) => (
            StatusCode::OK,
            Json(ApiResponse::ok(serde_json::json!({
                "graph": graph,
                "message": msg,
            }))),
        )
            .into_response(),
        Ok(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::err("unexpected response type".to_string())),
        )
            .into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn cypher_query(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
    Json(body): Json<CypherRequest>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    let cmd = Command::Cypher(CypherCmd {
        graph,
        query: body.query,
    });

    match engine.execute_command(cmd, identity.as_ref()) {
        Ok(CommandResponse::Text(json_str)) => (
            StatusCode::OK,
            Json(ApiResponse::ok(serde_json::json!({
                "results": serde_json::from_str::<serde_json::Value>(&json_str)
                    .unwrap_or(serde_json::Value::String(json_str)),
            }))),
        )
            .into_response(),
        Ok(CommandResponse::Integer(id)) => (
            StatusCode::CREATED,
            Json(ApiResponse::ok(serde_json::json!({ "node_id": id }))),
        )
            .into_response(),
        Ok(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::err("unexpected response type".to_string())),
        )
            .into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn metrics_handler() -> impl IntoResponse {
    use prometheus::Encoder;
    let encoder = prometheus::TextEncoder::new();
    let mut buffer = Vec::new();
    encoder.encode(&crate::metrics::REGISTRY.gather(), &mut buffer).unwrap();
    (StatusCode::OK, [("content-type", "text/plain; version=0.0.4")], buffer)
}

async fn create_graph(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Json(body): Json<CreateGraphRequest>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    let cmd = Command::GraphCreate(GraphCreateCmd {
        name: body.name,
        config: None,
    });
    match engine.execute_command(cmd, identity.as_ref()) {
        Ok(_) => (
            StatusCode::CREATED,
            Json(ApiResponse::<()>::ok_empty()),
        )
            .into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn list_graphs(State(engine): State<Arc<Engine>>, headers: HeaderMap) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    match engine.execute_command(Command::GraphList, identity.as_ref()) {
        Ok(CommandResponse::StringList(names)) => {
            (StatusCode::OK, Json(ApiResponse::ok(names))).into_response()
        }
        Ok(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::err("unexpected response type".to_string())),
        )
            .into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn get_graph_info(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    let cmd = Command::GraphInfo(name);
    match engine.execute_command(cmd, identity.as_ref()) {
        Ok(CommandResponse::GraphInfo(info)) => (
            StatusCode::OK,
            Json(ApiResponse::ok(GraphInfoJson {
                name: info.name,
                node_count: info.node_count,
                edge_count: info.edge_count,
                vector_count: info.vector_count,
                label_count: info.label_count,
                default_ttl_ms: info.default_ttl_ms,
            })),
        )
            .into_response(),
        Ok(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::err("unexpected response type".to_string())),
        )
            .into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

/// GET /v1/graphs/{graph}/check
async fn check_graph(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    let cmd = Command::GraphCheck(graph);
    match engine.execute_command(cmd, identity.as_ref()) {
        Ok(CommandResponse::Text(report)) => {
            (StatusCode::OK, Json(ApiResponse::ok(report))).into_response()
        }
        Ok(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::err("unexpected response type".to_string())),
        )
            .into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn get_detailed_stats(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    // Permission check: read access on the graph.
    if let Err(e) = engine.check_permission(
        identity.as_ref(),
        &graph,
        weav_auth::identity::GraphPermission::Read,
    ) {
        return weav_error_to_response(e).into_response();
    }
    match engine.handle_detailed_stats(&graph) {
        Ok(CommandResponse::Text(json_str)) => {
            // Parse the pretty-printed JSON so we can wrap it in the standard envelope.
            match serde_json::from_str::<serde_json::Value>(&json_str) {
                Ok(val) => (StatusCode::OK, Json(ApiResponse::ok(val))).into_response(),
                Err(_) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ApiResponse::<()>::err("failed to serialize stats".to_string())),
                )
                    .into_response(),
            }
        }
        Ok(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::err("unexpected response type".to_string())),
        )
            .into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn drop_graph(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    let cmd = Command::GraphDrop(name);
    match engine.execute_command(cmd, identity.as_ref()) {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::<()>::ok_empty())).into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn add_node(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
    Json(body): Json<AddNodeRequest>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    let group_id = extract_group_id(&headers);
    let mut properties = if let Some(ref val) = body.properties {
        json_to_props(val)
    } else {
        Vec::new()
    };
    if let Some(ref gid) = group_id {
        properties.push(("_group_id".to_string(), Value::String(compact_str::CompactString::from(gid.as_str()))));
    }

    let cmd = Command::NodeAdd(NodeAddCmd {
        graph,
        label: body.label,
        properties,
        embedding: body.embedding,
        entity_key: body.entity_key,
        ttl_ms: body.ttl_ms,
    });

    match engine.execute_command(cmd, identity.as_ref()) {
        Ok(CommandResponse::Integer(id)) => (
            StatusCode::CREATED,
            Json(ApiResponse::ok(NodeIdResponse { node_id: id })),
        )
            .into_response(),
        Ok(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::err("unexpected response type".to_string())),
        )
            .into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn get_node(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path((graph, id)): Path<(String, u64)>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    let cmd = Command::NodeGet(NodeGetCmd {
        graph,
        node_id: Some(id),
        entity_key: None,
    });

    match engine.execute_command(cmd, identity.as_ref()) {
        Ok(CommandResponse::NodeInfo(info)) => {
            let props_json = props_to_json(&info.properties);
            (
                StatusCode::OK,
                Json(ApiResponse::ok(NodeInfoJson {
                    node_id: info.node_id,
                    label: info.label,
                    properties: props_json,
                })),
            )
                .into_response()
        }
        Ok(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::err("unexpected response type".to_string())),
        )
            .into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn delete_node(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path((graph, id)): Path<(String, u64)>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    let cmd = Command::NodeDelete(NodeDeleteCmd {
        graph,
        node_id: id,
    });

    match engine.execute_command(cmd, identity.as_ref()) {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::<()>::ok_empty())).into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn get_node_neighbors(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path((graph, node_id)): Path<(String, u64)>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    if let Err(e) = engine.check_permission(
        identity.as_ref(),
        &graph,
        weav_auth::identity::GraphPermission::Read,
    ) {
        return weav_error_to_response(e).into_response();
    }

    let graph_arc = match engine.get_graph(&graph) {
        Ok(arc) => arc,
        Err(e) => return weav_error_to_response(e).into_response(),
    };
    let gs = graph_arc.read();

    if !gs.adjacency.has_node(node_id) {
        return weav_error_to_response(weav_core::error::WeavError::NodeNotFound(
            node_id,
            gs.graph_id,
        ))
        .into_response();
    }

    let both = gs.adjacency.neighbors_both(node_id, None);
    let mut neighbors = Vec::with_capacity(both.len());

    for (neighbor_id, edge_id, direction) in &both {
        let dir_str = match direction {
            Direction::Outgoing => "outgoing",
            Direction::Incoming => "incoming",
            Direction::Both => "both",
        };

        let (edge_label, edge_weight) = if let Some(meta) = gs.adjacency.get_edge(*edge_id) {
            let label = gs
                .interner
                .resolve_label(meta.label)
                .unwrap_or("unknown")
                .to_string();
            (label, meta.weight)
        } else {
            ("unknown".to_string(), 1.0)
        };

        let node_label = gs
            .properties
            .get_node_property(*neighbor_id, "_label")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        neighbors.push(NeighborEntry {
            node_id: *neighbor_id,
            edge_id: *edge_id,
            direction: dir_str.to_string(),
            edge_label,
            edge_weight,
            node_label,
        });
    }

    (
        StatusCode::OK,
        Json(ApiResponse::ok(NeighborsResponse {
            node_id,
            neighbors,
        })),
    )
        .into_response()
}

async fn add_edge(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
    Json(body): Json<AddEdgeRequest>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    let properties = if let Some(ref val) = body.properties {
        json_to_props(val)
    } else {
        Vec::new()
    };

    let cmd = Command::EdgeAdd(EdgeAddCmd {
        graph,
        source: body.source,
        target: body.target,
        label: body.label,
        weight: body.weight.unwrap_or(1.0),
        properties,
        ttl_ms: body.ttl_ms,
    });

    match engine.execute_command(cmd, identity.as_ref()) {
        Ok(CommandResponse::Integer(id)) => (
            StatusCode::CREATED,
            Json(ApiResponse::ok(EdgeIdResponse { edge_id: id })),
        )
            .into_response(),
        Ok(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::err("unexpected response type".to_string())),
        )
            .into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn invalidate_edge(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path((graph, id)): Path<(String, u64)>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    let cmd = Command::EdgeInvalidate(EdgeInvalidateCmd {
        graph,
        edge_id: id,
    });

    match engine.execute_command(cmd, identity.as_ref()) {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::<()>::ok_empty())).into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn get_edge(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path((graph, id)): Path<(String, u64)>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    if let Err(e) = engine.check_permission(
        identity.as_ref(),
        &graph,
        weav_auth::identity::GraphPermission::Read,
    ) {
        return weav_error_to_response(e).into_response();
    }

    let graph_arc = match engine.get_graph(&graph) {
        Ok(arc) => arc,
        Err(e) => return weav_error_to_response(e).into_response(),
    };
    let gs = graph_arc.read();

    let meta = match gs.adjacency.get_edge(id) {
        Some(m) => m,
        None => {
            return weav_error_to_response(weav_core::error::WeavError::EdgeNotFound(id))
                .into_response()
        }
    };

    let label = gs
        .interner
        .resolve_label(meta.label)
        .unwrap_or("unknown")
        .to_string();

    let all_props = gs.properties.get_all_edge_properties(id);
    let props: Vec<(String, Value)> = all_props
        .into_iter()
        .filter(|(k, _)| !k.starts_with('_'))
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect();
    let properties = props_to_json(&props);

    (
        StatusCode::OK,
        Json(ApiResponse::ok(EdgeDetailJson {
            edge_id: id,
            source: meta.source,
            target: meta.target,
            label,
            weight: meta.weight,
            properties,
        })),
    )
        .into_response()
}

async fn delete_edge(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path((graph, id)): Path<(String, u64)>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    let cmd = Command::EdgeDelete(EdgeDeleteCmd {
        graph,
        edge_id: id,
    });

    match engine.execute_command(cmd, identity.as_ref()) {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::<()>::ok_empty())).into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn snapshot(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    match engine.execute_command(Command::Snapshot, identity.as_ref()) {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::<()>::ok_empty())).into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn backup(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    body: Option<Json<BackupRequest>>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    let label = body.and_then(|Json(b)| b.label);
    match engine.execute_command(Command::Backup(label), identity.as_ref()) {
        Ok(CommandResponse::Text(msg)) => {
            (StatusCode::OK, Json(ApiResponse::ok(msg))).into_response()
        }
        Ok(_) => (StatusCode::OK, Json(ApiResponse::<()>::ok_empty())).into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn wal_compact(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    match engine.execute_command(Command::WalCompact, identity.as_ref()) {
        Ok(CommandResponse::Text(msg)) => {
            (StatusCode::OK, Json(ApiResponse::ok(msg))).into_response()
        }
        Ok(_) => (StatusCode::OK, Json(ApiResponse::<()>::ok_empty())).into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn server_info(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    match engine.execute_command(Command::Info, identity.as_ref()) {
        Ok(CommandResponse::Text(text)) => {
            (StatusCode::OK, Json(ApiResponse::ok(text))).into_response()
        }
        Ok(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::err("unexpected response type".to_string())),
        )
            .into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn update_node(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path((graph, id)): Path<(String, u64)>,
    Json(body): Json<UpdateNodeRequest>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    let properties = if let Some(ref val) = body.properties {
        json_to_props(val)
    } else {
        Vec::new()
    };

    let cmd = Command::NodeUpdate(NodeUpdateCmd {
        graph,
        node_id: id,
        properties,
        embedding: body.embedding,
    });

    match engine.execute_command(cmd, identity.as_ref()) {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::<()>::ok_empty())).into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn bulk_add_nodes(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
    Json(body): Json<BulkAddNodesRequest>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    let nodes: Vec<NodeAddCmd> = body
        .nodes
        .into_iter()
        .map(|n| {
            let properties = if let Some(ref val) = n.properties {
                json_to_props(val)
            } else {
                Vec::new()
            };
            NodeAddCmd {
                graph: graph.clone(),
                label: n.label,
                properties,
                embedding: n.embedding,
                entity_key: n.entity_key,
                ttl_ms: n.ttl_ms,
            }
        })
        .collect();

    let cmd = Command::BulkInsertNodes(BulkInsertNodesCmd {
        graph,
        nodes,
    });

    match engine.execute_command(cmd, identity.as_ref()) {
        Ok(CommandResponse::IntegerList(ids)) => {
            (StatusCode::CREATED, Json(ApiResponse::ok(BulkNodeIdsResponse { node_ids: ids }))).into_response()
        }
        Ok(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::err("unexpected response type".to_string())),
        )
            .into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn bulk_add_edges(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
    Json(body): Json<BulkAddEdgesRequest>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    let edges: Vec<EdgeAddCmd> = body
        .edges
        .into_iter()
        .map(|e| {
            let properties = if let Some(ref val) = e.properties {
                json_to_props(val)
            } else {
                Vec::new()
            };
            EdgeAddCmd {
                graph: graph.clone(),
                source: e.source,
                target: e.target,
                label: e.label,
                weight: e.weight.unwrap_or(1.0),
                properties,
                ttl_ms: e.ttl_ms,
            }
        })
        .collect();

    let cmd = Command::BulkInsertEdges(BulkInsertEdgesCmd {
        graph,
        edges,
    });

    match engine.execute_command(cmd, identity.as_ref()) {
        Ok(CommandResponse::IntegerList(ids)) => {
            (StatusCode::CREATED, Json(ApiResponse::ok(BulkEdgeIdsResponse { edge_ids: ids }))).into_response()
        }
        Ok(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::err("unexpected response type".to_string())),
        )
            .into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn context_query(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Json(body): Json<ContextRequest>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    // Build the seed strategy.
    let seeds = match (&body.embedding, &body.seed_nodes) {
        (Some(emb), Some(nodes)) if !nodes.is_empty() => SeedStrategy::Both {
            embedding: emb.clone(),
            top_k: 10,
            node_keys: nodes.clone(),
        },
        (Some(emb), _) => SeedStrategy::Vector {
            embedding: emb.clone(),
            top_k: 10,
        },
        (_, Some(nodes)) if !nodes.is_empty() => SeedStrategy::Nodes(nodes.clone()),
        _ => SeedStrategy::Nodes(Vec::new()),
    };

    // Parse direction.
    let direction = match body.direction.as_deref() {
        Some("outgoing") => Direction::Outgoing,
        Some("incoming") => Direction::Incoming,
        _ => Direction::Both,
    };

    // Parse decay.
    let decay = body.decay.and_then(|d| {
        match d.decay_type.to_lowercase().as_str() {
            "exponential" => d.half_life_ms.map(|ms| DecayFunction::Exponential { half_life_ms: ms }),
            "linear" => d.max_age_ms.map(|ms| DecayFunction::Linear { max_age_ms: ms }),
            "step" => d.cutoff_ms.map(|ms| DecayFunction::Step { cutoff_ms: ms }),
            "none" => Some(DecayFunction::None),
            _ => None,
        }
    });

    // Parse sort.
    let sort = body.sort_field.and_then(|field_str| {
        let field = match field_str.to_lowercase().as_str() {
            "relevance" => SortField::Relevance,
            "recency" => SortField::Recency,
            "confidence" => SortField::Confidence,
            _ => return None,
        };
        let dir = match body.sort_direction.as_deref() {
            Some("asc") => SortDirection::Asc,
            _ => SortDirection::Desc,
        };
        Some(SortOrder { field, direction: dir })
    });

    // Parse edge filter from edge_labels.
    let edge_filter = body.edge_labels.map(|labels| {
        EdgeFilterConfig {
            labels: Some(labels),
            min_weight: None,
            min_confidence: None,
        }
    });

    // Resolve budget: preset overrides explicit budget.
    let resolved_budget = if let Some(ref preset) = body.budget_preset {
        weav_query::parser::budget_preset(preset).map(TokenBudget::new)
    } else {
        body.budget.map(TokenBudget::new)
    };

    let query = ContextQuery {
        query_text: body.query,
        graph: body.graph,
        budget: resolved_budget,
        seeds,
        max_depth: body.max_depth.unwrap_or(2),
        direction,
        edge_filter,
        decay,
        include_provenance: body.include_provenance.unwrap_or(false),
        temporal_at: body.temporal_at,
        limit: body.limit,
        sort,
        explain: body.explain.unwrap_or(false),
        output_format: body.output_format,
        include_subgraph: body.include_subgraph.unwrap_or(false),
    };

    let cmd = Command::Context(query);
    match engine.execute_command(cmd, identity.as_ref()) {
        Ok(CommandResponse::Context(result)) => {
            (StatusCode::OK, Json(ApiResponse::ok(result))).into_response()
        }
        Ok(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::err("unexpected response type".to_string())),
        )
            .into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

// ─── Ingest handler ─────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct IngestRequest {
    pub content: Option<String>,
    pub content_base64: Option<String>,
    pub format: Option<String>,
    pub document_id: Option<String>,
    #[serde(default)]
    pub skip_extraction: bool,
    #[serde(default)]
    pub skip_dedup: bool,
    pub chunk_size: Option<usize>,
    pub entity_types: Option<Vec<String>>,
}

#[derive(Serialize)]
struct IngestResultJson {
    document_id: String,
    chunks_created: usize,
    entities_created: usize,
    entities_merged: usize,
    relationships_created: usize,
    pipeline_duration_ms: u64,
}

async fn ingest(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
    Json(body): Json<IngestRequest>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    // Resolve content from either text or base64.
    let content = if let Some(ref text) = body.content {
        text.clone()
    } else if let Some(ref b64) = body.content_base64 {
        match base64::engine::general_purpose::STANDARD.decode(b64) {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(s) => s,
                Err(e) => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(ApiResponse::<()>::err(format!(
                            "content_base64 is not valid UTF-8: {e}"
                        ))),
                    )
                        .into_response();
                }
            },
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ApiResponse::<()>::err(format!(
                        "invalid base64 in content_base64: {e}"
                    ))),
                )
                    .into_response();
            }
        }
    } else {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::<()>::err(
                "either 'content' or 'content_base64' is required".into(),
            )),
        )
            .into_response();
    };

    use weav_query::parser::{Command, IngestCmd};

    let cmd = Command::Ingest(IngestCmd {
        graph,
        content,
        format: body.format,
        document_id: body.document_id,
        skip_extraction: body.skip_extraction,
        skip_dedup: body.skip_dedup,
        chunk_size: body.chunk_size,
        entity_types: body.entity_types,
    });

    match engine.execute_command_async(cmd, identity.as_ref()).await {
        Ok(CommandResponse::IngestResult(info)) => (
            StatusCode::OK,
            Json(ApiResponse::ok(IngestResultJson {
                document_id: info.document_id,
                chunks_created: info.chunks_created,
                entities_created: info.entities_created,
                entities_merged: info.entities_merged,
                relationships_created: info.relationships_created,
                pipeline_duration_ms: info.pipeline_duration_ms,
            })),
        )
            .into_response(),
        Ok(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::<()>::err("unexpected response type".to_string())),
        )
            .into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

// ─── Value conversion helpers ───────────────────────────────────────────────

/// Convert a serde_json::Value (expected object) to a Vec of (key, Value) pairs.
fn json_to_props(val: &serde_json::Value) -> Vec<(String, Value)> {
    let mut props = Vec::new();
    if let Some(obj) = val.as_object() {
        for (k, v) in obj {
            props.push((k.clone(), json_val_to_value(v)));
        }
    }
    props
}

pub fn json_val_to_value(v: &serde_json::Value) -> Value {
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
            if arr.iter().all(|v| v.is_number()) {
                let floats: Vec<f32> = arr
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                Value::Vector(floats)
            } else {
                let items: Vec<Value> = arr.iter().map(json_val_to_value).collect();
                Value::List(items)
            }
        }
        serde_json::Value::Object(map) => {
            let pairs: Vec<(compact_str::CompactString, Value)> = map
                .iter()
                .map(|(k, v)| (compact_str::CompactString::from(k.as_str()), json_val_to_value(v)))
                .collect();
            Value::Map(pairs)
        }
    }
}

/// Convert a Vec of (key, Value) to a serde_json::Value.
fn props_to_json(props: &[(String, Value)]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (k, v) in props {
        map.insert(k.clone(), value_to_json(v));
    }
    serde_json::Value::Object(map)
}

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
            let arr: Vec<serde_json::Value> = items.iter().map(value_to_json).collect();
            serde_json::Value::Array(arr)
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

// ─── Algorithm response types ────────────────────────────────────────────────

#[derive(Serialize)]
struct AlgoNodeScore {
    node_id: u64,
    score: f64,
}

#[derive(Serialize)]
struct PageRankResponse {
    scores: Vec<AlgoNodeScore>,
}

#[derive(Serialize)]
struct BetweennessResponse {
    scores: Vec<AlgoNodeScore>,
}

#[derive(Serialize)]
struct CommunitiesResponse {
    communities: Vec<Vec<u64>>,
    modularity: f64,
}

#[derive(Serialize)]
struct ShortestPathResponse {
    path: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    length: Option<usize>,
}

#[derive(Serialize)]
struct ComponentsResponse {
    components: Vec<Vec<u64>>,
    count: usize,
}

#[derive(Serialize)]
struct TriangleCountResponse {
    total_triangles: u64,
    per_node: Vec<TriangleNodeInfo>,
}

#[derive(Serialize)]
struct TriangleNodeInfo {
    node_id: u64,
    triangles: u32,
    clustering_coefficient: f64,
}

#[derive(Serialize)]
struct TopologicalSortResponse {
    order: Vec<u64>,
}

#[derive(Serialize)]
struct HitsResponse {
    authorities: Vec<AlgoNodeScore>,
    hubs: Vec<AlgoNodeScore>,
}

#[derive(Serialize)]
struct FastRPEmbedding {
    node_id: u64,
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct FastRPResponse {
    embeddings: Vec<FastRPEmbedding>,
}

// ─── Algorithm handler ──────────────────────────────────────────────────────

async fn run_algorithm(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path((graph, algorithm)): Path<(String, String)>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    if let Err(e) = engine.check_permission(
        identity.as_ref(),
        &graph,
        weav_auth::identity::GraphPermission::Read,
    ) {
        return weav_error_to_response(e).into_response();
    }

    let graph_arc = match engine.get_graph(&graph) {
        Ok(arc) => arc,
        Err(e) => return weav_error_to_response(e).into_response(),
    };
    let gs = graph_arc.read();

    match algorithm.as_str() {
        "pagerank" => {
            let damping = body
                .get("damping")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.85) as f32;
            let iterations = body
                .get("iterations")
                .and_then(|v| v.as_u64())
                .unwrap_or(20) as u32;

            // Uniform PPR = standard PageRank: seed every node with equal weight.
            let all_nodes = gs.adjacency.all_node_ids();
            let seeds: Vec<(u64, f32)> = all_nodes.iter().map(|&nid| (nid, 1.0)).collect();

            let scored = weav_graph::traversal::personalized_pagerank(
                &gs.adjacency,
                &seeds,
                damping,
                iterations,
                1e-6,
            );

            let mut scores: Vec<AlgoNodeScore> = scored
                .iter()
                .map(|s| AlgoNodeScore {
                    node_id: s.node_id,
                    score: s.score as f64,
                })
                .collect();
            scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

            (StatusCode::OK, Json(ApiResponse::ok(PageRankResponse { scores }))).into_response()
        }

        "betweenness" => {
            let filter = weav_graph::traversal::EdgeFilter::none();
            let mut result = weav_graph::traversal::betweenness_centrality(&gs.adjacency, &filter);
            result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let scores: Vec<AlgoNodeScore> = result
                .into_iter()
                .map(|(node_id, score)| AlgoNodeScore { node_id, score })
                .collect();

            (StatusCode::OK, Json(ApiResponse::ok(BetweennessResponse { scores }))).into_response()
        }

        "communities" => {
            let resolution = body
                .get("resolution")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;

            let community_map =
                weav_graph::traversal::modularity_communities(&gs.adjacency, 100, resolution);

            // Group nodes by community.
            let mut groups: std::collections::HashMap<u64, Vec<u64>> =
                std::collections::HashMap::new();
            for (&node_id, &comm_id) in &community_map {
                groups.entry(comm_id).or_default().push(node_id);
            }
            let mut communities: Vec<Vec<u64>> = groups.into_values().collect();
            for c in &mut communities {
                c.sort();
            }
            communities.sort_by_key(|c| std::cmp::Reverse(c.len()));

            // Compute modularity Q.
            let modularity = compute_modularity(&gs.adjacency, &community_map);

            (
                StatusCode::OK,
                Json(ApiResponse::ok(CommunitiesResponse {
                    communities,
                    modularity,
                })),
            )
                .into_response()
        }

        "shortest_path" => {
            let source = match body.get("source").and_then(|v| v.as_u64()) {
                Some(s) => s,
                None => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(ApiResponse::<()>::err(
                            "missing required field: source".to_string(),
                        )),
                    )
                        .into_response()
                }
            };
            let target = match body.get("target").and_then(|v| v.as_u64()) {
                Some(t) => t,
                None => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(ApiResponse::<()>::err(
                            "missing required field: target".to_string(),
                        )),
                    )
                        .into_response()
                }
            };
            let max_depth = body
                .get("max_depth")
                .and_then(|v| v.as_u64())
                .unwrap_or(10) as u8;

            let path =
                weav_graph::traversal::shortest_path(&gs.adjacency, source, target, max_depth);

            let (path_out, length) = match path {
                Some(ref p) => (Some(p.clone()), Some(p.len() - 1)),
                None => (None, None),
            };

            (
                StatusCode::OK,
                Json(ApiResponse::ok(ShortestPathResponse {
                    path: path_out,
                    length,
                })),
            )
                .into_response()
        }

        "components" => {
            let component_map = weav_graph::traversal::connected_components(&gs.adjacency);

            // Group nodes by component.
            let mut groups: std::collections::HashMap<u32, Vec<u64>> =
                std::collections::HashMap::new();
            for (&node_id, &comp_id) in &component_map {
                groups.entry(comp_id).or_default().push(node_id);
            }
            let mut components: Vec<Vec<u64>> = groups.into_values().collect();
            for c in &mut components {
                c.sort();
            }
            components.sort_by_key(|c| std::cmp::Reverse(c.len()));
            let count = components.len();

            (
                StatusCode::OK,
                Json(ApiResponse::ok(ComponentsResponse { components, count })),
            )
                .into_response()
        }

        "closeness" => {
            let filter = weav_graph::traversal::EdgeFilter::none();
            let mut result = weav_graph::traversal::closeness_centrality(&gs.adjacency, &filter);
            result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let scores: Vec<AlgoNodeScore> = result
                .into_iter()
                .map(|(node_id, score)| AlgoNodeScore { node_id, score })
                .collect();
            (StatusCode::OK, Json(ApiResponse::ok(BetweennessResponse { scores }))).into_response()
        }

        "label_propagation" => {
            let iterations = body.get("iterations").and_then(|v| v.as_u64()).unwrap_or(100) as u32;
            let label_map = weav_graph::traversal::label_propagation(&gs.adjacency, iterations);
            let mut groups: std::collections::HashMap<u64, Vec<u64>> = std::collections::HashMap::new();
            for (&node_id, &label) in &label_map {
                groups.entry(label).or_default().push(node_id);
            }
            let mut communities: Vec<Vec<u64>> = groups.into_values().collect();
            for c in &mut communities { c.sort(); }
            communities.sort_by_key(|c| std::cmp::Reverse(c.len()));
            let modularity = compute_modularity(&gs.adjacency, &label_map);
            (StatusCode::OK, Json(ApiResponse::ok(CommunitiesResponse { communities, modularity }))).into_response()
        }

        "leiden" => {
            let resolution = body.get("resolution").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            let gamma = body.get("gamma").and_then(|v| v.as_f64()).unwrap_or(0.3) as f32;
            let community_map = weav_graph::traversal::leiden_communities(&gs.adjacency, 100, resolution, gamma);
            let mut groups: std::collections::HashMap<u64, Vec<u64>> = std::collections::HashMap::new();
            for (&node_id, &comm_id) in &community_map {
                groups.entry(comm_id).or_default().push(node_id);
            }
            let mut communities: Vec<Vec<u64>> = groups.into_values().collect();
            for c in &mut communities { c.sort(); }
            communities.sort_by_key(|c| std::cmp::Reverse(c.len()));
            let modularity = compute_modularity(&gs.adjacency, &community_map);
            (StatusCode::OK, Json(ApiResponse::ok(CommunitiesResponse { communities, modularity }))).into_response()
        }

        "triangle_count" => {
            let filter = weav_graph::traversal::EdgeFilter::none();
            let result = weav_graph::traversal::triangle_count(&gs.adjacency, &filter);
            let per_node: Vec<TriangleNodeInfo> = result.per_node.iter()
                .map(|&(node_id, triangles, clustering_coefficient)| TriangleNodeInfo { node_id, triangles, clustering_coefficient })
                .collect();
            (StatusCode::OK, Json(ApiResponse::ok(TriangleCountResponse { total_triangles: result.total_triangles, per_node }))).into_response()
        }

        "scc" => {
            let sccs = weav_graph::traversal::tarjan_scc(&gs.adjacency);
            let mut components: Vec<Vec<u64>> = sccs;
            for c in &mut components { c.sort(); }
            components.sort_by_key(|c| std::cmp::Reverse(c.len()));
            let count = components.len();
            (StatusCode::OK, Json(ApiResponse::ok(ComponentsResponse { components, count }))).into_response()
        }

        "topological_sort" => {
            match weav_graph::traversal::topological_sort(&gs.adjacency) {
                Ok(order) => (StatusCode::OK, Json(ApiResponse::ok(TopologicalSortResponse { order }))).into_response(),
                Err(e) => weav_error_to_response(e).into_response(),
            }
        }

        "degree" => {
            let all_nodes = gs.adjacency.all_node_ids();
            let n = all_nodes.len();
            let divisor = if n > 1 { (n - 1) as f64 } else { 1.0 };
            let mut scores: Vec<AlgoNodeScore> = all_nodes.iter()
                .map(|&nid| {
                    let deg = gs.adjacency.neighbors_both(nid, None).len();
                    AlgoNodeScore { node_id: nid, score: deg as f64 / divisor }
                })
                .collect();
            scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            (StatusCode::OK, Json(ApiResponse::ok(PageRankResponse { scores }))).into_response()
        }

        "eigenvector" => {
            let max_iter = body.get("iterations").and_then(|v| v.as_u64()).unwrap_or(100) as u32;
            let tolerance = body.get("tolerance").and_then(|v| v.as_f64()).unwrap_or(1e-6);
            let result = weav_graph::traversal::eigenvector_centrality(&gs.adjacency, max_iter, tolerance);
            let scores: Vec<AlgoNodeScore> = result.into_iter()
                .map(|(node_id, score)| AlgoNodeScore { node_id, score })
                .collect();
            (StatusCode::OK, Json(ApiResponse::ok(PageRankResponse { scores }))).into_response()
        }

        "hits" => {
            let max_iter = body.get("iterations").and_then(|v| v.as_u64()).unwrap_or(100) as u32;
            let tolerance = body.get("tolerance").and_then(|v| v.as_f64()).unwrap_or(1e-6);
            let (auth_scores, hub_scores) = weav_graph::traversal::hits(&gs.adjacency, max_iter, tolerance);
            let authorities: Vec<AlgoNodeScore> = auth_scores.into_iter()
                .map(|(node_id, score)| AlgoNodeScore { node_id, score })
                .collect();
            let hubs: Vec<AlgoNodeScore> = hub_scores.into_iter()
                .map(|(node_id, score)| AlgoNodeScore { node_id, score })
                .collect();
            (StatusCode::OK, Json(ApiResponse::ok(HitsResponse { authorities, hubs }))).into_response()
        }

        "fastrp" => {
            let dim = body.get("embedding_dim").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
            let iterations = body.get("iterations").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
            let norm = body.get("normalization_strength").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let seed = body.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);
            let embeddings = weav_graph::traversal::fastrp_embeddings(&gs.adjacency, dim, iterations, norm, seed);
            let mut results: Vec<FastRPEmbedding> = embeddings
                .into_iter()
                .map(|(nid, emb)| FastRPEmbedding { node_id: nid, embedding: emb })
                .collect();
            results.sort_by_key(|e| e.node_id);
            (StatusCode::OK, Json(ApiResponse::ok(FastRPResponse { embeddings: results }))).into_response()
        }

        _ => (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::<()>::err(format!(
                "unknown algorithm: {algorithm}. Supported: pagerank, betweenness, closeness, \
                 communities, label_propagation, leiden, shortest_path, components, scc, \
                 topological_sort, triangle_count, degree, eigenvector, hits, fastrp"
            ))),
        )
            .into_response(),
    }
}

/// Compute the modularity Q for a given community assignment.
///
/// Q = (1/2m) * sum_ij [ A_ij - (k_i * k_j) / (2m) ] * delta(c_i, c_j)
fn compute_modularity(
    adjacency: &weav_graph::adjacency::AdjacencyStore,
    community: &std::collections::HashMap<u64, u64>,
) -> f64 {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return 0.0;
    }

    // Compute total weight m and per-node degree k_i (undirected).
    let mut node_degree: std::collections::HashMap<u64, f64> =
        std::collections::HashMap::with_capacity(all_nodes.len());
    let mut total_weight = 0.0_f64;

    for &node in &all_nodes {
        let mut ki = 0.0_f64;
        for &(_, edge_id) in adjacency.neighbors_out(node, None).iter() {
            let w = adjacency
                .get_edge(edge_id)
                .map(|meta| meta.weight as f64)
                .unwrap_or(1.0);
            ki += w;
        }
        for &(_, edge_id) in adjacency.neighbors_in(node, None).iter() {
            let w = adjacency
                .get_edge(edge_id)
                .map(|meta| meta.weight as f64)
                .unwrap_or(1.0);
            ki += w;
        }
        total_weight += ki;
        node_degree.insert(node, ki);
    }

    let m = total_weight / 2.0;
    if m == 0.0 {
        return 0.0;
    }

    let mut q = 0.0_f64;
    for &node in &all_nodes {
        let ci = community.get(&node).copied().unwrap_or(node);
        let ki = node_degree.get(&node).copied().unwrap_or(0.0);
        for &(neighbor, edge_id) in adjacency.neighbors_out(node, None).iter() {
            let cj = community.get(&neighbor).copied().unwrap_or(neighbor);
            if ci == cj {
                let w = adjacency
                    .get_edge(edge_id)
                    .map(|meta| meta.weight as f64)
                    .unwrap_or(1.0);
                let kj = node_degree.get(&neighbor).copied().unwrap_or(0.0);
                q += w - (ki * kj) / (2.0 * m);
            }
        }
    }

    q / (2.0 * m)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;
    use weav_core::config::WeavConfig;

    fn make_app() -> Router {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        build_router(engine)
    }

    async fn body_to_json(body: Body) -> serde_json::Value {
        let bytes = axum::body::to_bytes(body, usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn test_health() {
        let app = make_app();
        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);
        assert_eq!(json["data"]["status"], "ok");
    }

    #[tokio::test]
    async fn test_create_and_list_graphs() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine);

        // Create a graph.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"name": "test_graph"}"#))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        // List graphs.
        let req = Request::builder()
            .uri("/v1/graphs")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);
        let data = json["data"].as_array().unwrap();
        assert_eq!(data.len(), 1);
        assert_eq!(data[0], "test_graph");
    }

    #[tokio::test]
    async fn test_get_graph_info() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        // Create graph via engine directly.
        engine
            .execute_command(Command::GraphCreate(GraphCreateCmd {
                name: "info_g".to_string(),
                config: None,
            }), None)
            .unwrap();

        let req = Request::builder()
            .uri("/v1/graphs/info_g")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["data"]["name"], "info_g");
        assert_eq!(json["data"]["node_count"], 0);
    }

    #[tokio::test]
    async fn test_graph_not_found() {
        let app = make_app();
        let req = Request::builder()
            .uri("/v1/graphs/nonexistent")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_drop_graph() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(Command::GraphCreate(GraphCreateCmd {
                name: "drop_me".to_string(),
                config: None,
            }), None)
            .unwrap();

        let req = Request::builder()
            .method("DELETE")
            .uri("/v1/graphs/drop_me")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_add_and_get_node() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(Command::GraphCreate(GraphCreateCmd {
                name: "ng".to_string(),
                config: None,
            }), None)
            .unwrap();

        // Add node.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/ng/nodes")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"label": "person", "properties": {"name": "Alice"}, "entity_key": "alice"}"#,
            ))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let json = body_to_json(resp.into_body()).await;
        let node_id = json["data"]["node_id"].as_u64().unwrap();
        assert!(node_id >= 1);

        // Get node.
        let req = Request::builder()
            .uri(format!("/v1/graphs/ng/nodes/{node_id}"))
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["data"]["node_id"], node_id);
        assert_eq!(json["data"]["label"], "person");
    }

    #[tokio::test]
    async fn test_delete_node() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(Command::GraphCreate(GraphCreateCmd {
                name: "dg".to_string(),
                config: None,
            }), None)
            .unwrap();

        // Add and then delete a node.
        let add_resp = engine
            .execute_command(Command::NodeAdd(NodeAddCmd {
                graph: "dg".to_string(),
                label: "x".to_string(),
                properties: vec![],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            }), None)
            .unwrap();
        let node_id = match add_resp {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let req = Request::builder()
            .method("DELETE")
            .uri(format!("/v1/graphs/dg/nodes/{node_id}"))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_add_edge() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(Command::GraphCreate(GraphCreateCmd {
                name: "eg".to_string(),
                config: None,
            }), None)
            .unwrap();

        // Add two nodes via engine.
        let n1 = match engine
            .execute_command(Command::NodeAdd(NodeAddCmd {
                graph: "eg".to_string(),
                label: "a".to_string(),
                properties: vec![],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            }), None)
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let n2 = match engine
            .execute_command(Command::NodeAdd(NodeAddCmd {
                graph: "eg".to_string(),
                label: "b".to_string(),
                properties: vec![],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            }), None)
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let body = serde_json::json!({
            "source": n1,
            "target": n2,
            "label": "knows",
            "weight": 0.8
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/eg/edges")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let json = body_to_json(resp.into_body()).await;
        assert!(json["data"]["edge_id"].as_u64().unwrap() >= 1);
    }

    #[tokio::test]
    async fn test_invalidate_edge() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(Command::GraphCreate(GraphCreateCmd {
                name: "ig".to_string(),
                config: None,
            }), None)
            .unwrap();

        let n1 = match engine
            .execute_command(Command::NodeAdd(NodeAddCmd {
                graph: "ig".to_string(),
                label: "a".to_string(),
                properties: vec![],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            }), None)
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let n2 = match engine
            .execute_command(Command::NodeAdd(NodeAddCmd {
                graph: "ig".to_string(),
                label: "b".to_string(),
                properties: vec![],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            }), None)
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let eid = match engine
            .execute_command(Command::EdgeAdd(EdgeAddCmd {
                graph: "ig".to_string(),
                source: n1,
                target: n2,
                label: "link".to_string(),
                weight: 1.0,
                properties: vec![],
                ttl_ms: None,
            }), None)
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let req = Request::builder()
            .method("POST")
            .uri(format!("/v1/graphs/ig/edges/{eid}/invalidate"))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_update_node() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(Command::GraphCreate(GraphCreateCmd {
                name: "ug".to_string(),
                config: None,
            }), None)
            .unwrap();

        let add_resp = engine
            .execute_command(Command::NodeAdd(NodeAddCmd {
                graph: "ug".to_string(),
                label: "person".to_string(),
                properties: vec![(
                    "name".to_string(),
                    Value::String(compact_str::CompactString::from("Alice")),
                )],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            }), None)
            .unwrap();
        let node_id = match add_resp {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let req = Request::builder()
            .method("PUT")
            .uri(format!("/v1/graphs/ug/nodes/{node_id}"))
            .header("content-type", "application/json")
            .body(Body::from(r#"{"properties": {"name": "Alice Updated", "age": 30}}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_bulk_add_nodes() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(Command::GraphCreate(GraphCreateCmd {
                name: "bg".to_string(),
                config: None,
            }), None)
            .unwrap();

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/bg/nodes/bulk")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"nodes": [{"label": "person", "properties": {"name": "A"}}, {"label": "person", "properties": {"name": "B"}}]}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);
        let ids = json["data"]["node_ids"].as_array().unwrap();
        assert_eq!(ids.len(), 2);
    }

    #[tokio::test]
    async fn test_bulk_add_edges() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(Command::GraphCreate(GraphCreateCmd {
                name: "beg".to_string(),
                config: None,
            }), None)
            .unwrap();

        let n1 = match engine
            .execute_command(Command::NodeAdd(NodeAddCmd {
                graph: "beg".to_string(),
                label: "a".to_string(),
                properties: vec![],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            }), None)
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let n2 = match engine
            .execute_command(Command::NodeAdd(NodeAddCmd {
                graph: "beg".to_string(),
                label: "b".to_string(),
                properties: vec![],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            }), None)
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let body = serde_json::json!({
            "edges": [{"source": n1, "target": n2, "label": "knows", "weight": 0.8}]
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/beg/edges/bulk")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let json = body_to_json(resp.into_body()).await;
        let ids = json["data"]["edge_ids"].as_array().unwrap();
        assert_eq!(ids.len(), 1);
    }

    #[tokio::test]
    async fn test_http_get_graph_not_found() {
        let app = make_app();
        let req = Request::builder()
            .uri("/v1/graphs/nonexistent")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_http_delete_node_not_found() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(Command::GraphCreate(GraphCreateCmd {
                name: "dng".to_string(),
                config: None,
            }), None)
            .unwrap();

        let req = Request::builder()
            .method("DELETE")
            .uri("/v1/graphs/dng/nodes/99999")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_http_metrics_endpoint() {
        let app = make_app();
        let req = Request::builder()
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let text = std::str::from_utf8(&bytes).unwrap();
        // The metrics endpoint should return text (even if empty, it's valid prometheus output).
        assert!(text.is_ascii());
    }

    #[tokio::test]
    async fn test_http_json_null_property() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(Command::GraphCreate(GraphCreateCmd {
                name: "nullg".to_string(),
                config: None,
            }), None)
            .unwrap();

        // Post a node with a null property value.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/nullg/nodes")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"label": "thing", "properties": {"name": "test", "optional": null}}"#,
            ))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);
        let node_id = json["data"]["node_id"].as_u64().unwrap();

        // Get the node and check properties.
        let req = Request::builder()
            .uri(format!("/v1/graphs/nullg/nodes/{node_id}"))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        // The null property should be present as null in JSON.
        assert!(json["data"]["properties"].get("optional").is_some());
        assert!(json["data"]["properties"]["optional"].is_null());
    }

    #[tokio::test]
    async fn test_http_update_node() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(Command::GraphCreate(GraphCreateCmd {
                name: "upg".to_string(),
                config: None,
            }), None)
            .unwrap();

        // Add a node.
        let add_resp = engine
            .execute_command(Command::NodeAdd(NodeAddCmd {
                graph: "upg".to_string(),
                label: "person".to_string(),
                properties: vec![(
                    "name".to_string(),
                    Value::String(compact_str::CompactString::from("Alice")),
                )],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            }), None)
            .unwrap();
        let node_id = match add_resp {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // PUT to update node properties.
        let req = Request::builder()
            .method("PUT")
            .uri(format!("/v1/graphs/upg/nodes/{node_id}"))
            .header("content-type", "application/json")
            .body(Body::from(r#"{"properties": {"name": "Bob", "city": "LA"}}"#))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // GET to verify the updated properties.
        let req = Request::builder()
            .uri(format!("/v1/graphs/upg/nodes/{node_id}"))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["data"]["properties"]["name"], "Bob");
        assert_eq!(json["data"]["properties"]["city"], "LA");
    }

    #[tokio::test]
    async fn test_context_endpoint() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(Command::GraphCreate(GraphCreateCmd {
                name: "cg".to_string(),
                config: None,
            }), None)
            .unwrap();

        // Add a node with entity_key.
        engine
            .execute_command(Command::NodeAdd(NodeAddCmd {
                graph: "cg".to_string(),
                label: "person".to_string(),
                properties: vec![(
                    "name".to_string(),
                    Value::String(compact_str::CompactString::from("Alice")),
                )],
                embedding: None,
                entity_key: Some("alice".to_string()),
                ttl_ms: None,
            }), None)
            .unwrap();

        let body = serde_json::json!({
            "graph": "cg",
            "query": "who is alice",
            "seed_nodes": ["alice"],
            "budget": 4096,
            "max_depth": 2
        });

        let req = Request::builder()
            .method("POST")
            .uri("/v1/context")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);
        assert!(json["data"]["nodes_included"].as_u64().is_some());
    }

    // ─── Auth tests ─────────────────────────────────────────────────────────

    fn make_authed_app() -> (Router, Arc<Engine>) {
        use weav_core::config::{AuthConfig, UserConfig, GraphPatternConfig};
        let mut config = WeavConfig::default();
        config.auth = AuthConfig {
            enabled: true,
            require_auth: true,
            acl_file: None,
            default_password: None,
            users: vec![
                UserConfig {
                    username: "admin".to_string(),
                    password: Some("secret".to_string()),
                    categories: vec!["+@all".to_string()],
                    graph_patterns: vec![GraphPatternConfig {
                        pattern: "*".to_string(),
                        permission: "admin".to_string(),
                    }],
                    api_keys: vec!["wk_test_key_123".to_string()],
                    enabled: true,
                },
                UserConfig {
                    username: "reader".to_string(),
                    password: Some("readonly".to_string()),
                    categories: vec!["+@read".to_string(), "+@connection".to_string()],
                    graph_patterns: vec![GraphPatternConfig {
                        pattern: "*".to_string(),
                        permission: "read".to_string(),
                    }],
                    api_keys: vec![],
                    enabled: true,
                },
            ],
        };
        let engine = Arc::new(Engine::new(config));
        let app = build_router(engine.clone());
        (app, engine)
    }

    #[tokio::test]
    async fn test_http_auth_bearer_token() {
        let (app, _engine) = make_authed_app();

        // Request with valid Bearer token should succeed.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs")
            .header("content-type", "application/json")
            .header("authorization", "Bearer wk_test_key_123")
            .body(Body::from(r#"{"name": "auth_test"}"#))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    #[tokio::test]
    async fn test_http_auth_basic() {
        let (app, _engine) = make_authed_app();

        // Request with valid Basic auth should succeed.
        let creds = base64::engine::general_purpose::STANDARD.encode("admin:secret");
        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs")
            .header("content-type", "application/json")
            .header("authorization", format!("Basic {creds}"))
            .body(Body::from(r#"{"name": "basic_test"}"#))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    #[tokio::test]
    async fn test_http_auth_rejected_no_creds() {
        let (app, _engine) = make_authed_app();

        // Request with no auth header should be rejected (auth required).
        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"name": "no_auth"}"#))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_http_auth_rejected_bad_token() {
        let (app, _engine) = make_authed_app();

        // Request with invalid Bearer token should be rejected.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs")
            .header("content-type", "application/json")
            .header("authorization", "Bearer wk_wrong_key")
            .body(Body::from(r#"{"name": "bad_auth"}"#))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_http_auth_permission_denied() {
        let (app, engine) = make_authed_app();

        // Admin creates a graph first — authenticate to get identity.
        let admin_id = engine.authenticate("admin", "secret").unwrap();
        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "perm_test".to_string(),
                    config: None,
                }),
                Some(&admin_id),
            )
            .unwrap();

        // Reader should be able to list graphs (read op).
        let reader_creds = base64::engine::general_purpose::STANDARD.encode("reader:readonly");
        let req = Request::builder()
            .uri("/v1/graphs")
            .header("authorization", format!("Basic {reader_creds}"))
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Reader should NOT be able to create graphs (admin op).
        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs")
            .header("content-type", "application/json")
            .header("authorization", format!("Basic {reader_creds}"))
            .body(Body::from(r#"{"name": "reader_graph"}"#))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_get_node_neighbors() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        // Create graph.
        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "nbr".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        // Add three nodes.
        let n1 = match engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "nbr".to_string(),
                    label: "person".to_string(),
                    properties: vec![("name".to_string(), Value::String("Alice".into()))],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };
        let n2 = match engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "nbr".to_string(),
                    label: "person".to_string(),
                    properties: vec![("name".to_string(), Value::String("Bob".into()))],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };
        let n3 = match engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "nbr".to_string(),
                    label: "org".to_string(),
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
            _ => panic!("expected Integer"),
        };

        // Add edges: n1 -> n2 (KNOWS), n3 -> n1 (EMPLOYS).
        engine
            .execute_command(
                Command::EdgeAdd(EdgeAddCmd {
                    graph: "nbr".to_string(),
                    source: n1,
                    target: n2,
                    label: "KNOWS".to_string(),
                    weight: 0.9,
                    properties: vec![],
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap();
        engine
            .execute_command(
                Command::EdgeAdd(EdgeAddCmd {
                    graph: "nbr".to_string(),
                    source: n3,
                    target: n1,
                    label: "EMPLOYS".to_string(),
                    weight: 0.5,
                    properties: vec![],
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap();

        // GET neighbors of n1.
        let req = Request::builder()
            .uri(format!("/v1/graphs/nbr/nodes/{n1}/neighbors"))
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);
        assert_eq!(json["data"]["node_id"], n1);

        let neighbors = json["data"]["neighbors"].as_array().unwrap();
        assert_eq!(neighbors.len(), 2);

        // Check we have both directions.
        let outgoing: Vec<&serde_json::Value> = neighbors
            .iter()
            .filter(|n| n["direction"] == "outgoing")
            .collect();
        let incoming: Vec<&serde_json::Value> = neighbors
            .iter()
            .filter(|n| n["direction"] == "incoming")
            .collect();
        assert_eq!(outgoing.len(), 1);
        assert_eq!(incoming.len(), 1);

        // Outgoing neighbor should be n2 with KNOWS label.
        assert_eq!(outgoing[0]["node_id"], n2);
        assert_eq!(outgoing[0]["edge_label"], "KNOWS");
        assert_eq!(outgoing[0]["node_label"], "person");
        assert_eq!(outgoing[0]["edge_weight"], 0.9);

        // Incoming neighbor should be n3 with EMPLOYS label.
        assert_eq!(incoming[0]["node_id"], n3);
        assert_eq!(incoming[0]["edge_label"], "EMPLOYS");
        assert_eq!(incoming[0]["node_label"], "org");
        assert_eq!(incoming[0]["edge_weight"], 0.5);
    }

    #[tokio::test]
    async fn test_get_node_neighbors_not_found() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "nbr_nf".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        let req = Request::builder()
            .uri("/v1/graphs/nbr_nf/nodes/9999/neighbors")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_node_neighbors_no_neighbors() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "nbr_empty".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        let n1 = match engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "nbr_empty".to_string(),
                    label: "solo".to_string(),
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
            _ => panic!("expected Integer"),
        };

        let req = Request::builder()
            .uri(format!("/v1/graphs/nbr_empty/nodes/{n1}/neighbors"))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);
        assert_eq!(json["data"]["node_id"], n1);
        let neighbors = json["data"]["neighbors"].as_array().unwrap();
        assert!(neighbors.is_empty());
    }

    #[tokio::test]
    async fn test_get_edge_with_properties() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "ep".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        let n1 = match engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "ep".to_string(),
                    label: "a".to_string(),
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
            _ => panic!("expected Integer"),
        };
        let n2 = match engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "ep".to_string(),
                    label: "b".to_string(),
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
            _ => panic!("expected Integer"),
        };

        let edge_id = match engine
            .execute_command(
                Command::EdgeAdd(EdgeAddCmd {
                    graph: "ep".to_string(),
                    source: n1,
                    target: n2,
                    label: "RELATES".to_string(),
                    weight: 0.7,
                    properties: vec![
                        ("since".to_string(), Value::String("2024".into())),
                        ("strength".to_string(), Value::Float(0.95)),
                    ],
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // GET edge with properties.
        let req = Request::builder()
            .uri(format!("/v1/graphs/ep/edges/{edge_id}"))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);
        assert_eq!(json["data"]["edge_id"], edge_id);
        assert_eq!(json["data"]["source"], n1);
        assert_eq!(json["data"]["target"], n2);
        assert_eq!(json["data"]["label"], "RELATES");
        assert_eq!(json["data"]["weight"], 0.7);
        assert_eq!(json["data"]["properties"]["since"], "2024");
        assert_eq!(json["data"]["properties"]["strength"], 0.95);
    }

    #[tokio::test]
    async fn test_get_edge_not_found() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "enf".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        let req = Request::builder()
            .uri("/v1/graphs/enf/edges/9999")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ─── Algorithm endpoint tests ───────────────────────────────────────

    /// Helper: create a graph with a triangle (3 nodes, 3 edges) for algorithm tests.
    fn setup_triangle_graph(engine: &Arc<Engine>, name: &str) -> (u64, u64, u64) {
        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: name.to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        let mut ids = Vec::new();
        for label in ["a", "b", "c"] {
            let id = match engine
                .execute_command(
                    Command::NodeAdd(NodeAddCmd {
                        graph: name.to_string(),
                        label: label.to_string(),
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
                _ => panic!("expected Integer"),
            };
            ids.push(id);
        }

        // Edges: a->b, b->c, c->a (triangle)
        for &(s, t) in &[(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])] {
            engine
                .execute_command(
                    Command::EdgeAdd(EdgeAddCmd {
                        graph: name.to_string(),
                        source: s,
                        target: t,
                        label: "link".to_string(),
                        weight: 1.0,
                        properties: vec![],
                        ttl_ms: None,
                    }),
                    None,
                )
                .unwrap();
        }

        (ids[0], ids[1], ids[2])
    }

    #[tokio::test]
    async fn test_algorithm_pagerank() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());
        let (n1, n2, n3) = setup_triangle_graph(&engine, "pr");

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/pr/algorithms/pagerank")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"damping": 0.85, "iterations": 20}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);

        let scores = json["data"]["scores"].as_array().unwrap();
        assert_eq!(scores.len(), 3);
        // In a symmetric triangle, all nodes should have roughly equal PageRank.
        for s in scores {
            assert!(s["score"].as_f64().unwrap() > 0.0);
            let nid = s["node_id"].as_u64().unwrap();
            assert!(nid == n1 || nid == n2 || nid == n3);
        }
    }

    #[tokio::test]
    async fn test_algorithm_pagerank_defaults() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());
        setup_triangle_graph(&engine, "pr_def");

        // Empty body — should use defaults.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/pr_def/algorithms/pagerank")
            .header("content-type", "application/json")
            .body(Body::from(r#"{}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);
        assert_eq!(json["data"]["scores"].as_array().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn test_algorithm_betweenness() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());
        setup_triangle_graph(&engine, "bt");

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/bt/algorithms/betweenness")
            .header("content-type", "application/json")
            .body(Body::from(r#"{}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);

        let scores = json["data"]["scores"].as_array().unwrap();
        assert_eq!(scores.len(), 3);
        // Each score must be a valid number.
        for s in scores {
            assert!(s["node_id"].as_u64().is_some());
            assert!(s["score"].as_f64().is_some());
        }
    }

    #[tokio::test]
    async fn test_algorithm_communities() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());
        setup_triangle_graph(&engine, "cm");

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/cm/algorithms/communities")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"resolution": 1.0}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);

        let communities = json["data"]["communities"].as_array().unwrap();
        assert!(!communities.is_empty());
        // Total nodes across all communities should be 3.
        let total: usize = communities.iter().map(|c| c.as_array().unwrap().len()).sum();
        assert_eq!(total, 3);
        // Modularity should be a finite number.
        assert!(json["data"]["modularity"].as_f64().unwrap().is_finite());
    }

    #[tokio::test]
    async fn test_algorithm_shortest_path() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());
        let (n1, _n2, n3) = setup_triangle_graph(&engine, "sp");

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/sp/algorithms/shortest_path")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({"source": n1, "target": n3, "max_depth": 10}).to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);

        let path = json["data"]["path"].as_array().unwrap();
        assert!(!path.is_empty());
        assert_eq!(path[0].as_u64().unwrap(), n1);
        assert_eq!(path.last().unwrap().as_u64().unwrap(), n3);
        assert!(json["data"]["length"].as_u64().unwrap() >= 1);
    }

    #[tokio::test]
    async fn test_algorithm_shortest_path_no_path() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());
        // Create graph with isolated nodes (no edges).
        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "sp_no".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();
        let n1 = match engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "sp_no".to_string(),
                    label: "x".to_string(),
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
            _ => panic!("expected Integer"),
        };
        let n2 = match engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "sp_no".to_string(),
                    label: "y".to_string(),
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
            _ => panic!("expected Integer"),
        };

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/sp_no/algorithms/shortest_path")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({"source": n1, "target": n2}).to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);
        assert!(json["data"]["path"].is_null());
    }

    #[tokio::test]
    async fn test_algorithm_shortest_path_missing_source() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());
        setup_triangle_graph(&engine, "sp_ms");

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/sp_ms/algorithms/shortest_path")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"target": 1}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_algorithm_components() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());
        setup_triangle_graph(&engine, "cc");

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/cc/algorithms/components")
            .header("content-type", "application/json")
            .body(Body::from(r#"{}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);

        let components = json["data"]["components"].as_array().unwrap();
        // A triangle is a single connected component.
        assert_eq!(json["data"]["count"].as_u64().unwrap(), 1);
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].as_array().unwrap().len(), 3);
    }

    #[tokio::test]
    async fn test_algorithm_unknown() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());
        setup_triangle_graph(&engine, "unk");

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/unk/algorithms/foobar")
            .header("content-type", "application/json")
            .body(Body::from(r#"{}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], false);
        assert!(json["error"].as_str().unwrap().contains("unknown algorithm"));
    }

    #[tokio::test]
    async fn test_algorithm_graph_not_found() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/nonexistent/algorithms/pagerank")
            .header("content-type", "application/json")
            .body(Body::from(r#"{}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_temporal_query_basic() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        // Create graph
        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "tq".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        // Add two nodes
        let n1 = match engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "tq".to_string(),
                    label: "person".to_string(),
                    properties: vec![("name".to_string(), Value::String("Alice".into()))],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };
        let n2 = match engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "tq".to_string(),
                    label: "person".to_string(),
                    properties: vec![("name".to_string(), Value::String("Bob".into()))],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add an edge
        engine
            .execute_command(
                Command::EdgeAdd(EdgeAddCmd {
                    graph: "tq".to_string(),
                    source: n1,
                    target: n2,
                    label: "knows".to_string(),
                    weight: 1.0,
                    properties: vec![],
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap();

        // Query at current timestamp (far future) -- should see both nodes and the edge
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/tq/temporal")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "timestamp": now_ms + 1000,
                    "include_edges": true
                })
                .to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);
        assert_eq!(json["data"]["node_count"].as_u64().unwrap(), 2);
        assert!(json["data"]["edge_count"].as_u64().unwrap() >= 1);
        assert_eq!(json["data"]["nodes"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn test_temporal_query_with_node_ids() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "tq2".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        let n1 = match engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "tq2".to_string(),
                    label: "x".to_string(),
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
            _ => panic!("expected Integer"),
        };
        // Add a second node but don't query it
        engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "tq2".to_string(),
                    label: "y".to_string(),
                    properties: vec![],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap();

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Query only n1
        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/tq2/temporal")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "timestamp": now_ms + 1000,
                    "node_ids": [n1],
                    "include_edges": false
                })
                .to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);
        assert_eq!(json["data"]["node_count"].as_u64().unwrap(), 1);
        assert_eq!(json["data"]["edge_count"].as_u64().unwrap(), 0);
    }

    #[tokio::test]
    async fn test_temporal_query_graph_not_found() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/nonexistent/temporal")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({ "timestamp": 1700000000000_u64 }).to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_temporal_range_query() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "tr".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        // Add nodes
        let n1 = match engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "tr".to_string(),
                    label: "event".to_string(),
                    properties: vec![("name".to_string(), Value::String("A".into()))],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };
        let n2 = match engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "tr".to_string(),
                    label: "event".to_string(),
                    properties: vec![("name".to_string(), Value::String("B".into()))],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add edge
        engine
            .execute_command(
                Command::EdgeAdd(EdgeAddCmd {
                    graph: "tr".to_string(),
                    source: n1,
                    target: n2,
                    label: "related".to_string(),
                    weight: 1.0,
                    properties: vec![],
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap();

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Range query that includes current time should find both nodes
        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/tr/temporal")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({
                    "start_time": now_ms - 5000,
                    "end_time": now_ms + 5000,
                    "include_edges": true
                })
                .to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);
        assert_eq!(json["data"]["node_count"].as_u64().unwrap(), 2);
        assert!(json["data"]["edge_count"].as_u64().unwrap() >= 1);
        assert!(json["data"]["query"]["start_time"].is_number());
        assert!(json["data"]["query"]["end_time"].is_number());

        // Range query should also include edges with range info in response
        let edges = json["data"]["edges"].as_array().unwrap();
        assert!(!edges.is_empty());
        assert!(edges[0]["valid_from"].is_number());

        // Verify response contains range query metadata (not point-in-time)
        assert!(json["data"]["query"]["start_time"].is_number());
        assert!(json["data"]["query"]["end_time"].is_number());
        assert!(json["data"]["query"]["timestamp"].is_null());
    }

    #[tokio::test]
    async fn test_temporal_query_missing_params() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "tp".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        // Neither timestamp nor start_time+end_time => bad request
        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/tp/temporal")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({ "include_edges": true }).to_string(),
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_node_history_endpoint() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "nh".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        let n1 = match engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "nh".to_string(),
                    label: "person".to_string(),
                    properties: vec![("name".to_string(), Value::String("Alice".into()))],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };
        let n2 = match engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "nh".to_string(),
                    label: "person".to_string(),
                    properties: vec![("name".to_string(), Value::String("Bob".into()))],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add edge between them
        engine
            .execute_command(
                Command::EdgeAdd(EdgeAddCmd {
                    graph: "nh".to_string(),
                    source: n1,
                    target: n2,
                    label: "knows".to_string(),
                    weight: 0.9,
                    properties: vec![],
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap();

        // Get node history for n1
        let req = Request::builder()
            .method("GET")
            .uri(format!("/v1/graphs/nh/nodes/{n1}/history"))
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);
        assert_eq!(json["data"]["node_id"].as_u64().unwrap(), n1);
        assert_eq!(json["data"]["graph"], "nh");
        assert_eq!(json["data"]["label"], "person");
        assert!(json["data"]["properties"]["name"].is_string());
        assert!(json["data"]["temporal"]["valid_from"].is_number());
        // Should have at least 1 edge
        assert!(json["data"]["edge_count"].as_u64().unwrap() >= 1);
        let edges = json["data"]["edges"].as_array().unwrap();
        assert!(!edges.is_empty());
        // Verify edge has temporal metadata
        let edge = &edges[0];
        assert!(edge["temporal"]["valid_from"].is_number());
        assert!(edge["active"].is_boolean());
    }

    #[tokio::test]
    async fn test_node_history_not_found() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "nh2".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        let req = Request::builder()
            .method("GET")
            .uri("/v1/graphs/nh2/nodes/9999/history")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ── CSV import/export tests ─────────────────────────────────────────

    #[tokio::test]
    async fn test_csv_import_creates_nodes() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "csv_test".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        let csv_body = "_label,name,age\nPerson,Alice,30\nPerson,Bob,25\nCompany,Acme,50\n";

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/csv_test/import/csv")
            .body(Body::from(csv_body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);
        assert_eq!(json["data"]["nodes_created"], 3);

        let graph_arc = engine.get_graph("csv_test").unwrap();
        let gs = graph_arc.read();
        assert_eq!(gs.adjacency.node_count(), 3);
    }

    #[tokio::test]
    async fn test_csv_import_missing_label_returns_error() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "csv_nolabel".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        let csv_body = "name,age\nAlice,30\nBob,25\n";

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/csv_nolabel/import/csv")
            .body(Body::from(csv_body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], false);
        assert!(json["error"].as_str().unwrap().contains("_label"));
    }

    #[tokio::test]
    async fn test_csv_export_roundtrip() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "csv_rt".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "csv_rt".to_string(),
                    label: "Person".to_string(),
                    properties: vec![
                        (
                            "name".to_string(),
                            Value::String(compact_str::CompactString::from("Alice")),
                        ),
                        ("age".to_string(), Value::Int(30)),
                    ],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap();

        engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "csv_rt".to_string(),
                    label: "Company".to_string(),
                    properties: vec![(
                        "name".to_string(),
                        Value::String(compact_str::CompactString::from("Acme")),
                    )],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap();

        let req = Request::builder()
            .uri("/v1/graphs/csv_rt/export/csv")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let csv_bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let csv_string = String::from_utf8(csv_bytes.to_vec()).unwrap();

        assert!(csv_string.contains("_label"));
        assert!(csv_string.contains("name"));
        assert!(csv_string.contains("Person"));
        assert!(csv_string.contains("Company"));
        assert!(csv_string.contains("Alice"));
        assert!(csv_string.contains("Acme"));

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "csv_rt2".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/csv_rt2/import/csv")
            .body(Body::from(csv_string))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["data"]["nodes_created"], 2);

        let graph_arc = engine.get_graph("csv_rt2").unwrap();
        let gs = graph_arc.read();
        assert_eq!(gs.adjacency.node_count(), 2);
    }

    #[tokio::test]
    async fn test_cancellation_token_lifecycle() {
        use tokio_util::sync::CancellationToken;

        let token = CancellationToken::new();
        assert!(!token.is_cancelled());

        let child = token.child_token();
        assert!(!child.is_cancelled());

        token.cancel();
        assert!(token.is_cancelled());
        assert!(child.is_cancelled());
    }

    #[tokio::test]
    async fn test_body_size_limit_rejects_oversized() {
        let app = make_app();

        // Create a body larger than 10 MB.
        let oversized = "x".repeat(11 * 1024 * 1024);
        let body = serde_json::json!({ "name": oversized });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn test_body_within_limit_succeeds() {
        let app = make_app();

        // A normal-sized request should succeed.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs")
            .header("content-type", "application/json")
            .body(Body::from(r#"{"name": "limit_test"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
    }

    #[tokio::test]
    async fn test_detailed_stats_endpoint() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        // Create a graph and add a node.
        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "dstat_g".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();
        engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "dstat_g".into(),
                    label: "Thing".into(),
                    properties: vec![],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap();

        let req = Request::builder()
            .uri("/v1/graphs/dstat_g/stats/detailed")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let json = body_to_json(resp.into_body()).await;
        let data = &json["data"];
        assert_eq!(data["graph"], "dstat_g");
        assert_eq!(data["node_count"], 1);
        assert_eq!(data["edge_count"], 0);
        assert!(data["degree_distribution"].is_object());
        assert!(data["memory_estimate"].is_object());
        assert!(data["label_distribution"].is_object());
        assert_eq!(data["label_distribution"]["Thing"], 1);
    }

    #[tokio::test]
    async fn test_detailed_stats_not_found() {
        let app = make_app();
        let req = Request::builder()
            .uri("/v1/graphs/nonexistent/stats/detailed")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_check_graph() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        // Create graph with nodes and edge.
        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "chk_http".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        let req = Request::builder()
            .uri("/v1/graphs/chk_http/check")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        let report = json["data"].as_str().unwrap();
        assert!(report.contains("OK - no issues found"));
    }

    #[tokio::test]
    async fn test_check_graph_not_found() {
        let app = make_app();
        let req = Request::builder()
            .uri("/v1/graphs/nonexistent/check")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ── DOT export tests ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_export_dot_empty_graph() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "dot_empty".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        let req = Request::builder()
            .uri("/v1/graphs/dot_empty/export/dot")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let dot = String::from_utf8(body.to_vec()).unwrap();
        assert!(dot.starts_with("digraph G {"));
        assert!(dot.contains("rankdir=LR"));
        assert!(dot.contains("node [shape=record]"));
        assert!(dot.ends_with("}\n"));
    }

    #[tokio::test]
    async fn test_export_dot_with_nodes_and_edges() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "dot_graph".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        // Add two nodes
        let n1 = match engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "dot_graph".to_string(),
                    label: "person".to_string(),
                    properties: vec![("name".to_string(), Value::String("Alice".into()))],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };
        let n2 = match engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "dot_graph".to_string(),
                    label: "person".to_string(),
                    properties: vec![("name".to_string(), Value::String("Bob".into()))],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap()
        {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add an edge
        engine
            .execute_command(
                Command::EdgeAdd(EdgeAddCmd {
                    graph: "dot_graph".to_string(),
                    source: n1,
                    target: n2,
                    label: "knows".to_string(),
                    weight: 1.0,
                    properties: vec![],
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap();

        let req = Request::builder()
            .uri("/v1/graphs/dot_graph/export/dot")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let dot = String::from_utf8(body.to_vec()).unwrap();

        // Verify DOT structure
        assert!(dot.starts_with("digraph G {"));
        assert!(dot.contains(&format!("n{n1}")));
        assert!(dot.contains(&format!("n{n2}")));
        assert!(dot.contains("person: Alice"));
        assert!(dot.contains("person: Bob"));
        assert!(dot.contains(&format!("n{n1} -> n{n2}")));
        assert!(dot.contains("[label=\"knows\""));
        assert!(dot.ends_with("}\n"));
    }

    #[tokio::test]
    async fn test_export_dot_not_found() {
        let app = make_app();
        let req = Request::builder()
            .uri("/v1/graphs/nonexistent/export/dot")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_export_dot_content_type() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "dot_ct".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        let req = Request::builder()
            .uri("/v1/graphs/dot_ct/export/dot")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert_eq!(ct, "text/vnd.graphviz");
    }

    #[tokio::test]
    async fn test_export_dot_special_characters() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "dotspec".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        // Add node with quotes in name
        engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "dotspec".to_string(),
                    label: "Person".to_string(),
                    properties: vec![(
                        "name".to_string(),
                        Value::String("Dr. \"Smith\"".into()),
                    )],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap();

        let req = Request::builder()
            .uri("/v1/graphs/dotspec/export/dot")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let dot = String::from_utf8(body.to_vec()).unwrap();
        // Should be valid DOT (contains digraph)
        assert!(dot.contains("digraph"));
    }

    #[tokio::test]
    async fn test_csv_import_empty_values() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "csvempty".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        let csv_body = "_label,name,age\nPerson,,30\nPerson,Bob,\n";

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/csvempty/import/csv")
            .body(Body::from(csv_body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_csv_import_header_only() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "csvhdr".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        let csv_body = "_label,name\n";

        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/csvhdr/import/csv")
            .body(Body::from(csv_body))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["data"]["nodes_created"], 0);
    }

    // ── Cypher HTTP Tests ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_http_cypher_match() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        // Create graph.
        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "cypher_http".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        // Add a node.
        engine
            .execute_command(
                Command::NodeAdd(NodeAddCmd {
                    graph: "cypher_http".to_string(),
                    label: "Person".to_string(),
                    properties: vec![(
                        "name".to_string(),
                        Value::String(compact_str::CompactString::from("Alice")),
                    )],
                    embedding: None,
                    entity_key: None,
                    ttl_ms: None,
                }),
                None,
            )
            .unwrap();

        // Query via Cypher endpoint.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/cypher_http/cypher")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"query": "MATCH (n:Person) RETURN n"}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);
        let results = &json["data"]["results"];
        assert!(results.is_array());
        assert_eq!(results.as_array().unwrap().len(), 1);
        assert_eq!(results[0]["n"]["labels"][0], "Person");
    }

    #[tokio::test]
    async fn test_http_cypher_create() {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        let app = build_router(engine.clone());

        // Create graph.
        engine
            .execute_command(
                Command::GraphCreate(GraphCreateCmd {
                    name: "cypher_http_c".to_string(),
                    config: None,
                }),
                None,
            )
            .unwrap();

        // Create node via Cypher endpoint.
        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/cypher_http_c/cypher")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"query": "CREATE (n:Person {name: 'Bob', age: 42})"}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);
        let json = body_to_json(resp.into_body()).await;
        assert_eq!(json["success"], true);
        assert!(json["data"]["node_id"].is_number());
    }

    #[tokio::test]
    async fn test_http_cypher_graph_not_found() {
        let app = make_app();
        let req = Request::builder()
            .method("POST")
            .uri("/v1/graphs/no_such_graph/cypher")
            .header("content-type", "application/json")
            .body(Body::from(
                r#"{"query": "MATCH (n) RETURN n"}"#,
            ))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
