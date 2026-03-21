//! HTTP REST API using axum.
//!
//! Translates JSON requests into engine `Command`s and engine responses back to JSON.

use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use base64::Engine as _;
use serde::{Deserialize, Serialize};

use weav_core::types::{DecayFunction, Direction, TokenBudget, Value};
use weav_query::parser::{
    Command, ContextQuery, EdgeAddCmd, EdgeDeleteCmd, EdgeFilterConfig,
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
}

#[derive(Deserialize)]
pub struct SearchParams {
    pub key: String,
    pub value: String,
    pub limit: Option<u32>,
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
        // Server info.
        .route("/v1/info", get(server_info))
        // Context query.
        .route("/v1/context", post(context_query))
        // Ingest (extraction pipeline).
        .route("/v1/graphs/{graph}/ingest", post(ingest))
        // Node search by property.
        .route("/v1/graphs/{graph}/search", get(search_nodes))
        // Graph export/import.
        .route("/v1/graphs/{graph}/export", get(export_graph))
        // Graph algorithms.
        .route(
            "/v1/graphs/{graph}/algorithms/{algorithm}",
            post(run_algorithm),
        )
        // Prometheus metrics.
        .route("/metrics", get(metrics_handler))
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

// ─── Handlers ───────────────────────────────────────────────────────────────

async fn health() -> impl IntoResponse {
    Json(ApiResponse::ok(HealthResponse {
        status: "ok".to_string(),
    }))
}

/// Search nodes by property key/value.
/// GET /v1/graphs/{graph}/search?key=name&value=Alice&limit=100
async fn search_nodes(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
    Query(params): Query<SearchParams>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
    let graph_arc = match engine.get_graph(&graph) {
        Ok(g) => g,
        Err(e) => return weav_error_to_response(e).into_response(),
    };
    let gs = graph_arc.read();

    let limit = params.limit.unwrap_or(100) as usize;
    let value_clone = params.value.clone();

    let matching_nodes: Vec<u64> = gs.properties.nodes_where(&params.key, &move |v| {
        match v {
            Value::String(s) => s.as_str() == value_clone,
            Value::Int(i) => i.to_string() == value_clone,
            Value::Float(f) => f.to_string() == value_clone,
            Value::Bool(b) => b.to_string() == value_clone,
            _ => false,
        }
    });

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

/// Export an entire graph as JSON.
/// GET /v1/graphs/{graph}/export
async fn export_graph(
    State(engine): State<Arc<Engine>>,
    headers: HeaderMap,
    Path(graph): Path<String>,
) -> impl IntoResponse {
    let identity = extract_identity(&engine, &headers);
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
    let properties = if let Some(ref val) = body.properties {
        json_to_props(val)
    } else {
        Vec::new()
    };

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

    let query = ContextQuery {
        query_text: body.query,
        graph: body.graph,
        budget: body.budget.map(TokenBudget::new),
        seeds,
        max_depth: body.max_depth.unwrap_or(2),
        direction,
        edge_filter,
        decay,
        include_provenance: body.include_provenance.unwrap_or(false),
        temporal_at: body.temporal_at,
        limit: body.limit,
        sort,
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

        _ => (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::<()>::err(format!(
                "unknown algorithm: {algorithm}. Supported: pagerank, betweenness, communities, shortest_path, components"
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
}
