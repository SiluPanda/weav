//! HTTP REST API using axum.
//!
//! Translates JSON requests into engine `Command`s and engine responses back to JSON.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use weav_core::types::{DecayFunction, Direction, TokenBudget, Value};
use weav_query::parser::{
    Command, ContextQuery, EdgeAddCmd, EdgeDeleteCmd, EdgeFilterConfig, EdgeGetCmd,
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
}

#[derive(Deserialize)]
pub struct AddEdgeRequest {
    pub source: u64,
    pub target: u64,
    pub label: String,
    pub weight: Option<f32>,
    pub properties: Option<serde_json::Value>,
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
struct EdgeInfoJson {
    edge_id: u64,
    source: u64,
    target: u64,
    label: String,
    weight: f32,
}

#[derive(Serialize)]
struct BulkNodeIdsResponse {
    node_ids: Vec<u64>,
}

#[derive(Serialize)]
struct BulkEdgeIdsResponse {
    edge_ids: Vec<u64>,
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
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, Json(ApiResponse::<()>::err(err.to_string())))
}

// ─── Handlers ───────────────────────────────────────────────────────────────

async fn health() -> impl IntoResponse {
    Json(ApiResponse::ok(HealthResponse {
        status: "ok".to_string(),
    }))
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
    Json(body): Json<CreateGraphRequest>,
) -> impl IntoResponse {
    let cmd = Command::GraphCreate(GraphCreateCmd {
        name: body.name,
        config: None,
    });
    match engine.execute_command(cmd, None) {
        Ok(_) => (
            StatusCode::CREATED,
            Json(ApiResponse::<()>::ok_empty()).into_response(),
        ),
        Err(e) => (
            StatusCode::CONFLICT,
            weav_error_to_response(e).into_response(),
        ),
    }
}

async fn list_graphs(State(engine): State<Arc<Engine>>) -> impl IntoResponse {
    match engine.execute_command(Command::GraphList, None) {
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
    Path(name): Path<String>,
) -> impl IntoResponse {
    let cmd = Command::GraphInfo(name);
    match engine.execute_command(cmd, None) {
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
    Path(name): Path<String>,
) -> impl IntoResponse {
    let cmd = Command::GraphDrop(name);
    match engine.execute_command(cmd, None) {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::<()>::ok_empty())).into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn add_node(
    State(engine): State<Arc<Engine>>,
    Path(graph): Path<String>,
    Json(body): Json<AddNodeRequest>,
) -> impl IntoResponse {
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
    });

    match engine.execute_command(cmd, None) {
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
    Path((graph, id)): Path<(String, u64)>,
) -> impl IntoResponse {
    let cmd = Command::NodeGet(NodeGetCmd {
        graph,
        node_id: Some(id),
        entity_key: None,
    });

    match engine.execute_command(cmd, None) {
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
    Path((graph, id)): Path<(String, u64)>,
) -> impl IntoResponse {
    let cmd = Command::NodeDelete(NodeDeleteCmd {
        graph,
        node_id: id,
    });

    match engine.execute_command(cmd, None) {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::<()>::ok_empty())).into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn add_edge(
    State(engine): State<Arc<Engine>>,
    Path(graph): Path<String>,
    Json(body): Json<AddEdgeRequest>,
) -> impl IntoResponse {
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
    });

    match engine.execute_command(cmd, None) {
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
    Path((graph, id)): Path<(String, u64)>,
) -> impl IntoResponse {
    let cmd = Command::EdgeInvalidate(EdgeInvalidateCmd {
        graph,
        edge_id: id,
    });

    match engine.execute_command(cmd, None) {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::<()>::ok_empty())).into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn get_edge(
    State(engine): State<Arc<Engine>>,
    Path((graph, id)): Path<(String, u64)>,
) -> impl IntoResponse {
    let cmd = Command::EdgeGet(EdgeGetCmd {
        graph,
        edge_id: id,
    });

    match engine.execute_command(cmd, None) {
        Ok(CommandResponse::EdgeInfo(info)) => (
            StatusCode::OK,
            Json(ApiResponse::ok(EdgeInfoJson {
                edge_id: info.edge_id,
                source: info.source,
                target: info.target,
                label: info.label,
                weight: info.weight,
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

async fn delete_edge(
    State(engine): State<Arc<Engine>>,
    Path((graph, id)): Path<(String, u64)>,
) -> impl IntoResponse {
    let cmd = Command::EdgeDelete(EdgeDeleteCmd {
        graph,
        edge_id: id,
    });

    match engine.execute_command(cmd, None) {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::<()>::ok_empty())).into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn snapshot(
    State(engine): State<Arc<Engine>>,
) -> impl IntoResponse {
    match engine.execute_command(Command::Snapshot, None) {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::<()>::ok_empty())).into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn server_info(
    State(engine): State<Arc<Engine>>,
) -> impl IntoResponse {
    match engine.execute_command(Command::Info, None) {
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
    Path((graph, id)): Path<(String, u64)>,
    Json(body): Json<UpdateNodeRequest>,
) -> impl IntoResponse {
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

    match engine.execute_command(cmd, None) {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::<()>::ok_empty())).into_response(),
        Err(e) => weav_error_to_response(e).into_response(),
    }
}

async fn bulk_add_nodes(
    State(engine): State<Arc<Engine>>,
    Path(graph): Path<String>,
    Json(body): Json<BulkAddNodesRequest>,
) -> impl IntoResponse {
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
            }
        })
        .collect();

    let cmd = Command::BulkInsertNodes(BulkInsertNodesCmd {
        graph,
        nodes,
    });

    match engine.execute_command(cmd, None) {
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
    Path(graph): Path<String>,
    Json(body): Json<BulkAddEdgesRequest>,
) -> impl IntoResponse {
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
            }
        })
        .collect();

    let cmd = Command::BulkInsertEdges(BulkInsertEdgesCmd {
        graph,
        edges,
    });

    match engine.execute_command(cmd, None) {
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
    Json(body): Json<ContextRequest>,
) -> impl IntoResponse {
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
    match engine.execute_command(cmd, None) {
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
}
