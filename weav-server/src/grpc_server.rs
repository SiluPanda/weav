//! gRPC server implementation for the Weav context graph database.

use std::sync::Arc;

use base64::Engine as _;
use compact_str::CompactString;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use weav_core::events::GraphEvent;
use weav_core::scope::{ScopeRef, resolve_graph_ref};
use weav_core::types::{Direction, RerankConfig, RetrievalMode, Value};
use weav_proto::grpc::weav_service_server::WeavService;
use weav_proto::grpc::*;
use weav_query::parser::{
    BulkInsertEdgesCmd, BulkInsertNodesCmd, Command, ContextQuery, EdgeAddCmd, EdgeInvalidateCmd,
    GraphCreateCmd, NodeAddCmd, NodeDeleteCmd, NodeGetCmd, NodeUpdateCmd, SeedStrategy,
};

use crate::engine::{CommandResponse, Engine};

pub struct WeavGrpcService {
    pub engine: Arc<Engine>,
}

/// Extract session identity from gRPC request metadata.
fn extract_grpc_identity(
    engine: &Engine,
    metadata: &tonic::metadata::MetadataMap,
) -> Option<weav_auth::identity::SessionIdentity> {
    if !engine.is_auth_enabled() {
        return None;
    }
    let auth = metadata.get("authorization")?.to_str().ok()?;

    if let Some(token) = auth.strip_prefix("Bearer ") {
        engine.authenticate_api_key(token.trim()).ok()
    } else if let Some(basic) = auth.strip_prefix("Basic ") {
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

fn weav_error_to_status(err: weav_core::error::WeavError) -> Status {
    match &err {
        weav_core::error::WeavError::GraphNotFound(_) => Status::not_found(err.to_string()),
        weav_core::error::WeavError::NodeNotFound(_, _) => Status::not_found(err.to_string()),
        weav_core::error::WeavError::EdgeNotFound(_) => Status::not_found(err.to_string()),
        weav_core::error::WeavError::DuplicateNode(_) => Status::already_exists(err.to_string()),
        weav_core::error::WeavError::Conflict(_) => Status::already_exists(err.to_string()),
        weav_core::error::WeavError::QueryParseError(_) => {
            Status::invalid_argument(err.to_string())
        }
        weav_core::error::WeavError::InvalidConfig(_) => Status::invalid_argument(err.to_string()),
        weav_core::error::WeavError::DimensionMismatch { .. } => {
            Status::invalid_argument(err.to_string())
        }
        weav_core::error::WeavError::AuthenticationRequired => {
            Status::unauthenticated(err.to_string())
        }
        weav_core::error::WeavError::AuthenticationFailed(_) => {
            Status::unauthenticated(err.to_string())
        }
        weav_core::error::WeavError::PermissionDenied(_) => {
            Status::permission_denied(err.to_string())
        }
        _ => Status::internal(err.to_string()),
    }
}

fn props_from_proto(props: &[Property]) -> Vec<(String, Value)> {
    props
        .iter()
        .map(|p| {
            let val = if let Ok(v) = serde_json::from_str::<serde_json::Value>(&p.value_json) {
                match v {
                    serde_json::Value::String(s) => Value::String(CompactString::from(s.as_str())),
                    serde_json::Value::Number(n) => {
                        if let Some(i) = n.as_i64() {
                            Value::Int(i)
                        } else if let Some(f) = n.as_f64() {
                            Value::Float(f)
                        } else {
                            Value::Null
                        }
                    }
                    serde_json::Value::Bool(b) => Value::Bool(b),
                    _ => Value::String(CompactString::from(p.value_json.as_str())),
                }
            } else {
                Value::String(CompactString::from(p.value_json.as_str()))
            };
            (p.key.clone(), val)
        })
        .collect()
}

fn props_to_proto(props: &[(String, Value)]) -> Vec<Property> {
    props
        .iter()
        .map(|(k, v)| Property {
            key: k.clone(),
            value_json: match v {
                Value::String(s) => serde_json::json!(s.as_str()).to_string(),
                Value::Int(i) => i.to_string(),
                Value::Float(f) => f.to_string(),
                Value::Bool(b) => b.to_string(),
                _ => format!("{v:?}"),
            },
        })
        .collect()
}

#[allow(clippy::result_large_err)]
fn retrieval_mode_from_proto(raw: i32) -> Result<RetrievalMode, Status> {
    match weav_proto::grpc::RetrievalMode::try_from(raw) {
        Ok(weav_proto::grpc::RetrievalMode::Unspecified)
        | Ok(weav_proto::grpc::RetrievalMode::Local) => Ok(RetrievalMode::Local),
        Ok(weav_proto::grpc::RetrievalMode::Global) => Ok(RetrievalMode::Global),
        Ok(weav_proto::grpc::RetrievalMode::Hybrid) => Ok(RetrievalMode::Hybrid),
        Ok(weav_proto::grpc::RetrievalMode::Drift) => Ok(RetrievalMode::Drift),
        Err(_) => Err(Status::invalid_argument("invalid retrieval_mode")),
    }
}

fn rerank_from_proto(proto: Option<RerankConfigProto>) -> Option<RerankConfig> {
    proto.map(|value| RerankConfig {
        enabled: value
            .enabled
            .unwrap_or(weav_core::types::default_rerank_enabled()),
        provider: (!value.provider.is_empty()).then_some(value.provider),
        model: (!value.model.is_empty()).then_some(value.model),
        candidate_limit: value
            .candidate_limit
            .unwrap_or(weav_core::types::default_rerank_candidate_limit()),
        score_weight: value
            .score_weight
            .unwrap_or(weav_core::types::default_rerank_score_weight()),
    })
}

fn scope_from_proto(proto: Option<ScopeRefProto>) -> Option<ScopeRef> {
    let proto = proto?;
    let scope = ScopeRef {
        workspace_id: (!proto.workspace_id.trim().is_empty()).then_some(proto.workspace_id),
        user_id: (!proto.user_id.trim().is_empty()).then_some(proto.user_id),
        agent_id: (!proto.agent_id.trim().is_empty()).then_some(proto.agent_id),
        session_id: (!proto.session_id.trim().is_empty()).then_some(proto.session_id),
    };

    (!scope.is_empty()).then_some(scope)
}

#[allow(clippy::result_large_err)]
fn resolve_graph_or_scope(graph: &str, scope: Option<ScopeRefProto>) -> Result<String, Status> {
    let scope = scope_from_proto(scope);
    resolve_graph_ref(Some(graph), scope.as_ref()).map_err(weav_error_to_status)
}

fn event_visible_to_identity(
    event: &GraphEvent,
    identity: Option<&weav_auth::identity::SessionIdentity>,
) -> bool {
    identity.is_none_or(|id| id.permissions.can_read_graph(event.graph.as_str()))
}

#[allow(clippy::result_large_err)]
fn graph_event_to_proto(event: &GraphEvent) -> Result<GraphEventProto, Status> {
    let public = event
        .to_public_event()
        .map_err(|err| Status::internal(format!("failed to serialize event: {err}")))?;
    Ok(GraphEventProto {
        sequence: public.sequence,
        graph: public.graph.to_string(),
        timestamp_ms: public.timestamp_ms,
        kind: public.kind.to_string(),
        payload_json: public.payload_json,
    })
}

#[tonic::async_trait]
impl WeavService for WeavGrpcService {
    async fn ping(&self, _: Request<PingRequest>) -> Result<Response<PingResponse>, Status> {
        Ok(Response::new(PingResponse {
            message: "PONG".to_string(),
        }))
    }

    async fn create_graph(
        &self,
        request: Request<CreateGraphRequest>,
    ) -> Result<Response<CreateGraphResponse>, Status> {
        let identity = extract_grpc_identity(&self.engine, request.metadata());
        let req = request.into_inner();
        let name = resolve_graph_or_scope(&req.name, req.scope)?;
        let cmd = Command::GraphCreate(GraphCreateCmd { name, config: None });
        self.engine
            .execute_command(cmd, identity.as_ref())
            .map_err(weav_error_to_status)?;
        Ok(Response::new(CreateGraphResponse {}))
    }

    async fn drop_graph(
        &self,
        request: Request<DropGraphRequest>,
    ) -> Result<Response<DropGraphResponse>, Status> {
        let identity = extract_grpc_identity(&self.engine, request.metadata());
        let req = request.into_inner();
        let graph = resolve_graph_or_scope(&req.name, req.scope)?;
        let cmd = Command::GraphDrop(graph);
        self.engine
            .execute_command(cmd, identity.as_ref())
            .map_err(weav_error_to_status)?;
        Ok(Response::new(DropGraphResponse {}))
    }

    async fn list_graphs(
        &self,
        request: Request<ListGraphsRequest>,
    ) -> Result<Response<ListGraphsResponse>, Status> {
        let identity = extract_grpc_identity(&self.engine, request.metadata());
        match self
            .engine
            .execute_command(Command::GraphList, identity.as_ref())
            .map_err(weav_error_to_status)?
        {
            CommandResponse::StringList(names) => Ok(Response::new(ListGraphsResponse { names })),
            _ => Err(Status::internal("unexpected response type")),
        }
    }

    async fn graph_info(
        &self,
        request: Request<GraphInfoRequest>,
    ) -> Result<Response<GraphInfoResponse>, Status> {
        let identity = extract_grpc_identity(&self.engine, request.metadata());
        let req = request.into_inner();
        let graph = resolve_graph_or_scope(&req.name, req.scope)?;
        let cmd = Command::GraphInfo(graph);
        match self
            .engine
            .execute_command(cmd, identity.as_ref())
            .map_err(weav_error_to_status)?
        {
            CommandResponse::GraphInfo(info) => Ok(Response::new(GraphInfoResponse {
                name: info.name,
                node_count: info.node_count,
                edge_count: info.edge_count,
            })),
            _ => Err(Status::internal("unexpected response type")),
        }
    }

    async fn add_node(
        &self,
        request: Request<AddNodeRequest>,
    ) -> Result<Response<AddNodeResponse>, Status> {
        let identity = extract_grpc_identity(&self.engine, request.metadata());
        let req = request.into_inner();
        let graph = resolve_graph_or_scope(&req.graph, req.scope)?;
        let properties = props_from_proto(&req.properties);
        let embedding = if req.embedding.is_empty() {
            None
        } else {
            Some(req.embedding)
        };
        let entity_key = if req.entity_key.is_empty() {
            None
        } else {
            Some(req.entity_key)
        };

        let cmd = Command::NodeAdd(NodeAddCmd {
            graph,
            label: req.label,
            properties,
            embedding,
            entity_key,
            ttl_ms: None,
        });

        match self
            .engine
            .execute_command(cmd, identity.as_ref())
            .map_err(weav_error_to_status)?
        {
            CommandResponse::Integer(id) => Ok(Response::new(AddNodeResponse { node_id: id })),
            _ => Err(Status::internal("unexpected response type")),
        }
    }

    async fn get_node(
        &self,
        request: Request<GetNodeRequest>,
    ) -> Result<Response<GetNodeResponse>, Status> {
        let identity = extract_grpc_identity(&self.engine, request.metadata());
        let req = request.into_inner();
        let graph = resolve_graph_or_scope(&req.graph, req.scope)?;
        let cmd = Command::NodeGet(NodeGetCmd {
            graph,
            node_id: Some(req.node_id),
            entity_key: None,
        });

        match self
            .engine
            .execute_command(cmd, identity.as_ref())
            .map_err(weav_error_to_status)?
        {
            CommandResponse::NodeInfo(info) => Ok(Response::new(GetNodeResponse {
                node_id: info.node_id,
                label: info.label,
                properties: props_to_proto(&info.properties),
            })),
            _ => Err(Status::internal("unexpected response type")),
        }
    }

    async fn update_node(
        &self,
        request: Request<UpdateNodeRequest>,
    ) -> Result<Response<UpdateNodeResponse>, Status> {
        let identity = extract_grpc_identity(&self.engine, request.metadata());
        let req = request.into_inner();
        let graph = resolve_graph_or_scope(&req.graph, req.scope)?;
        let properties = props_from_proto(&req.properties);
        let embedding = if req.embedding.is_empty() {
            None
        } else {
            Some(req.embedding)
        };

        let cmd = Command::NodeUpdate(NodeUpdateCmd {
            graph,
            node_id: req.node_id,
            properties,
            embedding,
        });

        self.engine
            .execute_command(cmd, identity.as_ref())
            .map_err(weav_error_to_status)?;
        Ok(Response::new(UpdateNodeResponse {}))
    }

    async fn delete_node(
        &self,
        request: Request<DeleteNodeRequest>,
    ) -> Result<Response<DeleteNodeResponse>, Status> {
        let identity = extract_grpc_identity(&self.engine, request.metadata());
        let req = request.into_inner();
        let graph = resolve_graph_or_scope(&req.graph, req.scope)?;
        let cmd = Command::NodeDelete(NodeDeleteCmd {
            graph,
            node_id: req.node_id,
        });

        self.engine
            .execute_command(cmd, identity.as_ref())
            .map_err(weav_error_to_status)?;
        Ok(Response::new(DeleteNodeResponse {}))
    }

    async fn add_edge(
        &self,
        request: Request<AddEdgeRequest>,
    ) -> Result<Response<AddEdgeResponse>, Status> {
        let identity = extract_grpc_identity(&self.engine, request.metadata());
        let req = request.into_inner();
        let graph = resolve_graph_or_scope(&req.graph, req.scope)?;
        let cmd = Command::EdgeAdd(EdgeAddCmd {
            graph,
            source: req.source,
            target: req.target,
            label: req.label,
            weight: req.weight,
            properties: Vec::new(),
            ttl_ms: None,
        });

        match self
            .engine
            .execute_command(cmd, identity.as_ref())
            .map_err(weav_error_to_status)?
        {
            CommandResponse::Integer(id) => Ok(Response::new(AddEdgeResponse { edge_id: id })),
            _ => Err(Status::internal("unexpected response type")),
        }
    }

    async fn invalidate_edge(
        &self,
        request: Request<InvalidateEdgeRequest>,
    ) -> Result<Response<InvalidateEdgeResponse>, Status> {
        let identity = extract_grpc_identity(&self.engine, request.metadata());
        let req = request.into_inner();
        let graph = resolve_graph_or_scope(&req.graph, req.scope)?;
        let cmd = Command::EdgeInvalidate(EdgeInvalidateCmd {
            graph,
            edge_id: req.edge_id,
        });

        self.engine
            .execute_command(cmd, identity.as_ref())
            .map_err(weav_error_to_status)?;
        Ok(Response::new(InvalidateEdgeResponse {}))
    }

    async fn context_query(
        &self,
        request: Request<ContextQueryRequest>,
    ) -> Result<Response<ContextQueryResponse>, Status> {
        let identity = extract_grpc_identity(&self.engine, request.metadata());
        let req = request.into_inner();
        let graph = resolve_graph_or_scope(&req.graph, req.scope.clone())?;
        let retrieval_mode = retrieval_mode_from_proto(req.retrieval_mode)?;
        let rerank = rerank_from_proto(req.rerank);

        let seeds = if !req.embedding.is_empty() && !req.seed_nodes.is_empty() {
            SeedStrategy::Both {
                embedding: req.embedding,
                top_k: 10,
                node_keys: req.seed_nodes,
            }
        } else if !req.embedding.is_empty() {
            SeedStrategy::Vector {
                embedding: req.embedding,
                top_k: 10,
            }
        } else if !req.seed_nodes.is_empty() {
            SeedStrategy::Nodes(req.seed_nodes)
        } else {
            SeedStrategy::Nodes(Vec::new())
        };

        let query = ContextQuery {
            query_text: if req.query.is_empty() {
                None
            } else {
                Some(req.query)
            },
            graph,
            retrieval_mode,
            rerank,
            budget: if req.budget > 0 {
                Some(weav_core::types::TokenBudget::new(req.budget))
            } else {
                None
            },
            seeds,
            max_depth: req.max_depth as u8,
            direction: Direction::Both,
            edge_filter: None,
            decay: None,
            include_provenance: req.include_provenance,
            temporal_at: None,
            limit: None,
            sort: None,
            explain: false,
            output_format: None,
            include_subgraph: false,
        };

        let cmd = Command::Context(query);
        match self
            .engine
            .execute_command(cmd, identity.as_ref())
            .map_err(weav_error_to_status)?
        {
            CommandResponse::Context(result) => {
                let chunks = result
                    .chunks
                    .iter()
                    .map(|c| ContextChunkProto {
                        node_id: c.node_id,
                        content: c.content.clone(),
                        label: c.label.clone(),
                        relevance_score: c.relevance_score,
                        depth: c.depth as u32,
                        token_count: c.token_count,
                    })
                    .collect();

                Ok(Response::new(ContextQueryResponse {
                    chunks,
                    total_tokens: result.total_tokens,
                    budget_used: result.budget_used,
                    nodes_considered: result.nodes_considered,
                    nodes_included: result.nodes_included,
                    query_time_us: result.query_time_us,
                }))
            }
            _ => Err(Status::internal("unexpected response type")),
        }
    }

    async fn bulk_add_nodes(
        &self,
        request: Request<BulkAddNodesRequest>,
    ) -> Result<Response<BulkAddNodesResponse>, Status> {
        let identity = extract_grpc_identity(&self.engine, request.metadata());
        let req = request.into_inner();
        let graph = resolve_graph_or_scope(&req.graph, req.scope)?;
        let nodes: Vec<NodeAddCmd> = req
            .nodes
            .into_iter()
            .map(|n| {
                let properties = props_from_proto(&n.properties);
                let embedding = if n.embedding.is_empty() {
                    None
                } else {
                    Some(n.embedding)
                };
                let entity_key = if n.entity_key.is_empty() {
                    None
                } else {
                    Some(n.entity_key)
                };
                NodeAddCmd {
                    graph: graph.clone(),
                    label: n.label,
                    properties,
                    embedding,
                    entity_key,
                    ttl_ms: None,
                }
            })
            .collect();

        let cmd = Command::BulkInsertNodes(BulkInsertNodesCmd { graph, nodes });

        match self
            .engine
            .execute_command(cmd, identity.as_ref())
            .map_err(weav_error_to_status)?
        {
            CommandResponse::IntegerList(ids) => Ok(Response::new(BulkAddNodesResponse { ids })),
            _ => Err(Status::internal("unexpected response type")),
        }
    }

    async fn bulk_add_edges(
        &self,
        request: Request<BulkAddEdgesRequest>,
    ) -> Result<Response<BulkAddEdgesResponse>, Status> {
        let identity = extract_grpc_identity(&self.engine, request.metadata());
        let req = request.into_inner();
        let graph = resolve_graph_or_scope(&req.graph, req.scope)?;
        let edges: Vec<EdgeAddCmd> = req
            .edges
            .into_iter()
            .map(|e| EdgeAddCmd {
                graph: graph.clone(),
                source: e.source,
                target: e.target,
                label: e.label,
                weight: e.weight,
                properties: Vec::new(),
                ttl_ms: None,
            })
            .collect();

        let cmd = Command::BulkInsertEdges(BulkInsertEdgesCmd { graph, edges });

        match self
            .engine
            .execute_command(cmd, identity.as_ref())
            .map_err(weav_error_to_status)?
        {
            CommandResponse::IntegerList(ids) => Ok(Response::new(BulkAddEdgesResponse { ids })),
            _ => Err(Status::internal("unexpected response type")),
        }
    }

    type ContextQueryStreamStream = ReceiverStream<Result<ContextChunkProto, Status>>;
    type SubscribeEventsStream = ReceiverStream<Result<GraphEventProto, Status>>;

    async fn context_query_stream(
        &self,
        request: Request<ContextQueryRequest>,
    ) -> Result<Response<Self::ContextQueryStreamStream>, Status> {
        let identity = extract_grpc_identity(&self.engine, request.metadata());
        let req = request.into_inner();
        let graph = resolve_graph_or_scope(&req.graph, req.scope.clone())?;
        let retrieval_mode = retrieval_mode_from_proto(req.retrieval_mode)?;
        let rerank = rerank_from_proto(req.rerank);

        let seeds = if !req.embedding.is_empty() && !req.seed_nodes.is_empty() {
            SeedStrategy::Both {
                embedding: req.embedding,
                top_k: 10,
                node_keys: req.seed_nodes,
            }
        } else if !req.embedding.is_empty() {
            SeedStrategy::Vector {
                embedding: req.embedding,
                top_k: 10,
            }
        } else if !req.seed_nodes.is_empty() {
            SeedStrategy::Nodes(req.seed_nodes)
        } else {
            SeedStrategy::Nodes(Vec::new())
        };

        let query = ContextQuery {
            query_text: if req.query.is_empty() {
                None
            } else {
                Some(req.query)
            },
            graph,
            retrieval_mode,
            rerank,
            budget: if req.budget > 0 {
                Some(weav_core::types::TokenBudget::new(req.budget))
            } else {
                None
            },
            seeds,
            max_depth: req.max_depth as u8,
            direction: Direction::Both,
            edge_filter: None,
            decay: None,
            include_provenance: req.include_provenance,
            temporal_at: None,
            limit: None,
            sort: None,
            explain: false,
            output_format: None,
            include_subgraph: false,
        };

        let cmd = Command::Context(query);
        let result = match self
            .engine
            .execute_command(cmd, identity.as_ref())
            .map_err(weav_error_to_status)?
        {
            CommandResponse::Context(result) => result,
            _ => return Err(Status::internal("unexpected response type")),
        };

        let (tx, rx) = mpsc::channel(result.chunks.len().max(1));
        tokio::spawn(async move {
            for c in &result.chunks {
                let chunk = ContextChunkProto {
                    node_id: c.node_id,
                    content: c.content.clone(),
                    label: c.label.clone(),
                    relevance_score: c.relevance_score,
                    depth: c.depth as u32,
                    token_count: c.token_count,
                };
                if tx.send(Ok(chunk)).await.is_err() {
                    break;
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn subscribe_events(
        &self,
        request: Request<SubscribeEventsRequest>,
    ) -> Result<Response<Self::SubscribeEventsStream>, Status> {
        let identity = extract_grpc_identity(&self.engine, request.metadata());
        let req = request.into_inner();
        let graph = (!req.graph.trim().is_empty()).then_some(req.graph);

        if let Some(graph_name) = graph.as_deref() {
            self.engine
                .check_permission(
                    identity.as_ref(),
                    graph_name,
                    weav_auth::identity::GraphPermission::Read,
                )
                .map_err(weav_error_to_status)?;
        } else if self.engine.is_auth_required() && identity.is_none() {
            return Err(Status::unauthenticated("authentication required"));
        }

        let mut live_rx = self.engine.subscribe_events();
        let backlog = self.engine.replay_events(
            graph.as_deref(),
            req.since_sequence,
            req.replay_limit as usize,
        );
        let mut next_sequence = backlog
            .last()
            .map(|event| event.sequence.saturating_add(1))
            .unwrap_or(req.since_sequence.saturating_add(1));

        let (tx, rx) = mpsc::channel(backlog.len().max(16));
        tokio::spawn(async move {
            for event in backlog {
                if !event_visible_to_identity(&event, identity.as_ref()) {
                    continue;
                }
                let proto = match graph_event_to_proto(&event) {
                    Ok(proto) => proto,
                    Err(status) => {
                        let _ = tx.send(Err(status)).await;
                        return;
                    }
                };
                if tx.send(Ok(proto)).await.is_err() {
                    return;
                }
            }

            loop {
                match live_rx.recv().await {
                    Ok(event) => {
                        if event.sequence < next_sequence {
                            continue;
                        }
                        next_sequence = event.sequence.saturating_add(1);
                        if graph
                            .as_deref()
                            .is_some_and(|name| event.graph.as_str() != name)
                            || !event_visible_to_identity(&event, identity.as_ref())
                        {
                            continue;
                        }
                        let proto = match graph_event_to_proto(&event) {
                            Ok(proto) => proto,
                            Err(status) => {
                                let _ = tx.send(Err(status)).await;
                                return;
                            }
                        };
                        if tx.send(Ok(proto)).await.is_err() {
                            return;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn snapshot(
        &self,
        request: Request<SnapshotRequest>,
    ) -> Result<Response<SnapshotResponse>, Status> {
        let identity = extract_grpc_identity(&self.engine, request.metadata());
        let cmd = Command::Snapshot;
        match self.engine.execute_command(cmd, identity.as_ref()) {
            Ok(_) => Ok(Response::new(SnapshotResponse {
                success: true,
                message: "Snapshot completed successfully".to_string(),
            })),
            Err(e) => Ok(Response::new(SnapshotResponse {
                success: false,
                message: format!("Snapshot failed: {e}"),
            })),
        }
    }

    async fn info(&self, request: Request<InfoRequest>) -> Result<Response<InfoResponse>, Status> {
        let identity = extract_grpc_identity(&self.engine, request.metadata());
        let cmd = Command::Info;
        let version = match self
            .engine
            .execute_command(cmd, identity.as_ref())
            .map_err(weav_error_to_status)?
        {
            CommandResponse::Text(text) => text,
            _ => "unknown".to_string(),
        };

        // Get graph list to count graphs and total nodes/edges.
        let (graph_count, total_nodes, total_edges) = match self
            .engine
            .execute_command(Command::GraphList, identity.as_ref())
            .map_err(weav_error_to_status)?
        {
            CommandResponse::StringList(names) => {
                let mut nodes = 0u64;
                let mut edges = 0u64;
                let count = names.len() as u32;
                for name in &names {
                    if let Ok(CommandResponse::GraphInfo(info)) = self
                        .engine
                        .execute_command(Command::GraphInfo(name.clone()), identity.as_ref())
                    {
                        nodes += info.node_count;
                        edges += info.edge_count;
                    }
                }
                (count, nodes, edges)
            }
            _ => (0, 0, 0),
        };

        Ok(Response::new(InfoResponse {
            version,
            uptime_seconds: 0, // Engine doesn't track uptime yet
            graph_count,
            total_nodes,
            total_edges,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_stream::StreamExt;
    use weav_core::config::{AuthConfig, GraphPatternConfig, UserConfig, WeavConfig};
    use weav_proto::grpc::weav_service_server::WeavService;

    fn make_service() -> WeavGrpcService {
        WeavGrpcService {
            engine: Arc::new(Engine::new(WeavConfig::default())),
        }
    }

    fn make_authed_service() -> WeavGrpcService {
        let config = WeavConfig {
            auth: AuthConfig {
                enabled: true,
                require_auth: true,
                default_password: None,
                acl_file: None,
                users: vec![UserConfig {
                    username: "reader".to_string(),
                    password: Some("readonly".to_string()),
                    categories: vec!["+@read".to_string(), "+@connection".to_string()],
                    graph_patterns: vec![GraphPatternConfig {
                        pattern: "*".to_string(),
                        permission: "read".to_string(),
                    }],
                    api_keys: vec!["wk_reader".to_string()],
                    enabled: true,
                }],
            },
            ..WeavConfig::default()
        };
        WeavGrpcService {
            engine: Arc::new(Engine::new(config)),
        }
    }

    #[tokio::test]
    async fn test_grpc_ping() {
        let svc = make_service();
        let resp = svc.ping(Request::new(PingRequest {})).await.unwrap();
        assert_eq!(resp.into_inner().message, "PONG");
    }

    #[tokio::test]
    async fn test_grpc_info() {
        let svc = make_service();

        // Create a graph and add a node so counts are nonzero.
        svc.create_graph(Request::new(CreateGraphRequest {
            name: "info_g".to_string(),
            scope: None,
        }))
        .await
        .unwrap();

        svc.add_node(Request::new(AddNodeRequest {
            graph: "info_g".to_string(),
            label: "person".to_string(),
            properties: vec![Property {
                key: "name".to_string(),
                value_json: "\"Alice\"".to_string(),
            }],
            embedding: vec![],
            entity_key: String::new(),
            scope: None,
        }))
        .await
        .unwrap();

        let resp = svc.info(Request::new(InfoRequest {})).await.unwrap();
        let info = resp.into_inner();
        assert!(info.version.contains("weav-server"));
        assert_eq!(info.graph_count, 1);
        assert_eq!(info.total_nodes, 1);
        assert_eq!(info.total_edges, 0);
    }

    #[tokio::test]
    async fn test_grpc_snapshot() {
        let svc = make_service();

        svc.create_graph(Request::new(CreateGraphRequest {
            name: "snap_g".to_string(),
            scope: None,
        }))
        .await
        .unwrap();

        let resp = svc
            .snapshot(Request::new(SnapshotRequest {}))
            .await
            .unwrap();
        let snap = resp.into_inner();
        assert!(snap.success);
    }

    #[tokio::test]
    async fn test_grpc_bulk_add_nodes() {
        let svc = make_service();

        svc.create_graph(Request::new(CreateGraphRequest {
            name: "bulk_ng".to_string(),
            scope: None,
        }))
        .await
        .unwrap();

        let resp = svc
            .bulk_add_nodes(Request::new(BulkAddNodesRequest {
                graph: "bulk_ng".to_string(),
                nodes: vec![
                    BulkNodeItem {
                        label: "person".to_string(),
                        properties: vec![Property {
                            key: "name".to_string(),
                            value_json: "\"Alice\"".to_string(),
                        }],
                        embedding: vec![],
                        entity_key: String::new(),
                    },
                    BulkNodeItem {
                        label: "person".to_string(),
                        properties: vec![Property {
                            key: "name".to_string(),
                            value_json: "\"Bob\"".to_string(),
                        }],
                        embedding: vec![],
                        entity_key: String::new(),
                    },
                    BulkNodeItem {
                        label: "topic".to_string(),
                        properties: vec![Property {
                            key: "name".to_string(),
                            value_json: "\"Rust\"".to_string(),
                        }],
                        embedding: vec![],
                        entity_key: String::new(),
                    },
                ],
                scope: None,
            }))
            .await
            .unwrap();
        let ids = resp.into_inner().ids;
        assert_eq!(ids.len(), 3);
    }

    #[tokio::test]
    async fn test_grpc_bulk_add_edges() {
        let svc = make_service();

        svc.create_graph(Request::new(CreateGraphRequest {
            name: "bulk_eg".to_string(),
            scope: None,
        }))
        .await
        .unwrap();

        // Add two nodes first.
        let n1 = svc
            .add_node(Request::new(AddNodeRequest {
                graph: "bulk_eg".to_string(),
                label: "a".to_string(),
                properties: vec![],
                embedding: vec![],
                entity_key: String::new(),
                scope: None,
            }))
            .await
            .unwrap()
            .into_inner()
            .node_id;

        let n2 = svc
            .add_node(Request::new(AddNodeRequest {
                graph: "bulk_eg".to_string(),
                label: "b".to_string(),
                properties: vec![],
                embedding: vec![],
                entity_key: String::new(),
                scope: None,
            }))
            .await
            .unwrap()
            .into_inner()
            .node_id;

        let resp = svc
            .bulk_add_edges(Request::new(BulkAddEdgesRequest {
                graph: "bulk_eg".to_string(),
                edges: vec![
                    BulkEdgeItem {
                        source: n1,
                        target: n2,
                        label: "knows".to_string(),
                        weight: 0.9,
                    },
                    BulkEdgeItem {
                        source: n2,
                        target: n1,
                        label: "follows".to_string(),
                        weight: 0.5,
                    },
                ],
                scope: None,
            }))
            .await
            .unwrap();
        let ids = resp.into_inner().ids;
        assert_eq!(ids.len(), 2);
    }

    #[tokio::test]
    async fn test_grpc_context_query() {
        let svc = make_service();

        svc.create_graph(Request::new(CreateGraphRequest {
            name: "ctx_g".to_string(),
            scope: None,
        }))
        .await
        .unwrap();

        // Add nodes with entity_key so we can seed by key.
        svc.add_node(Request::new(AddNodeRequest {
            graph: "ctx_g".to_string(),
            label: "person".to_string(),
            properties: vec![Property {
                key: "name".to_string(),
                value_json: "\"Alice\"".to_string(),
            }],
            embedding: vec![],
            entity_key: "alice".to_string(),
            scope: None,
        }))
        .await
        .unwrap();

        svc.add_node(Request::new(AddNodeRequest {
            graph: "ctx_g".to_string(),
            label: "topic".to_string(),
            properties: vec![Property {
                key: "name".to_string(),
                value_json: "\"Rust\"".to_string(),
            }],
            embedding: vec![],
            entity_key: "rust".to_string(),
            scope: None,
        }))
        .await
        .unwrap();

        let resp = svc
            .context_query(Request::new(ContextQueryRequest {
                graph: "ctx_g".to_string(),
                query: "what does alice use".to_string(),
                embedding: vec![],
                seed_nodes: vec!["alice".to_string()],
                budget: 4096,
                max_depth: 2,
                include_provenance: false,
                retrieval_mode: weav_proto::grpc::RetrievalMode::Local as i32,
                rerank: None,
                scope: None,
            }))
            .await
            .unwrap();
        let result = resp.into_inner();
        // The query should complete successfully with some chunks.
        // The query should complete without error. With seed nodes, it should consider at least one node.
        assert!(result.nodes_considered > 0);
    }

    #[tokio::test]
    async fn test_grpc_scope_resolution() {
        let svc = make_service();
        let scope = ScopeRefProto {
            workspace_id: "acme".to_string(),
            user_id: "u_123".to_string(),
            agent_id: String::new(),
            session_id: String::new(),
        };

        svc.create_graph(Request::new(CreateGraphRequest {
            name: String::new(),
            scope: Some(scope.clone()),
        }))
        .await
        .unwrap();

        svc.add_node(Request::new(AddNodeRequest {
            graph: String::new(),
            label: "person".to_string(),
            properties: vec![Property {
                key: "name".to_string(),
                value_json: "\"Alice\"".to_string(),
            }],
            embedding: vec![],
            entity_key: "alice".to_string(),
            scope: Some(scope.clone()),
        }))
        .await
        .unwrap();

        let resp = svc
            .context_query(Request::new(ContextQueryRequest {
                graph: String::new(),
                query: "who is alice".to_string(),
                embedding: vec![],
                seed_nodes: vec!["alice".to_string()],
                budget: 4096,
                max_depth: 2,
                include_provenance: false,
                retrieval_mode: weav_proto::grpc::RetrievalMode::Local as i32,
                rerank: None,
                scope: Some(scope),
            }))
            .await
            .unwrap();

        assert!(resp.into_inner().nodes_considered > 0);
    }

    #[tokio::test]
    async fn test_grpc_context_query_global_retrieval_mode() {
        let svc = make_service();

        svc.create_graph(Request::new(CreateGraphRequest {
            name: "ctx_global".to_string(),
            scope: None,
        }))
        .await
        .unwrap();

        svc.add_node(Request::new(AddNodeRequest {
            graph: "ctx_global".to_string(),
            label: "_community_summary".to_string(),
            properties: vec![
                Property {
                    key: "summary".to_string(),
                    value_json: "\"Rust systems programming community\"".to_string(),
                },
                Property {
                    key: "member_count".to_string(),
                    value_json: "4".to_string(),
                },
            ],
            embedding: vec![],
            entity_key: "".to_string(),
            scope: None,
        }))
        .await
        .unwrap();

        let resp = svc
            .context_query(Request::new(ContextQueryRequest {
                graph: "ctx_global".to_string(),
                query: "systems programming".to_string(),
                embedding: vec![],
                seed_nodes: vec![],
                budget: 4096,
                max_depth: 2,
                include_provenance: false,
                retrieval_mode: weav_proto::grpc::RetrievalMode::Global as i32,
                rerank: None,
                scope: None,
            }))
            .await
            .unwrap();

        let result = resp.into_inner();
        assert_eq!(result.chunks[0].label, "_community_summary");
    }

    #[tokio::test]
    async fn test_grpc_context_query_rerank_parsing() {
        let svc = make_service();

        svc.create_graph(Request::new(CreateGraphRequest {
            name: "ctx_rerank".to_string(),
            scope: None,
        }))
        .await
        .unwrap();

        let alice_id = svc
            .add_node(Request::new(AddNodeRequest {
                graph: "ctx_rerank".to_string(),
                label: "person".to_string(),
                properties: vec![
                    Property {
                        key: "name".to_string(),
                        value_json: "\"Alice\"".to_string(),
                    },
                    Property {
                        key: "description".to_string(),
                        value_json: "\"A software engineer\"".to_string(),
                    },
                ],
                embedding: vec![],
                entity_key: "alice".to_string(),
                scope: None,
            }))
            .await
            .unwrap()
            .into_inner()
            .node_id;

        let rust_id = svc
            .add_node(Request::new(AddNodeRequest {
                graph: "ctx_rerank".to_string(),
                label: "topic".to_string(),
                properties: vec![
                    Property {
                        key: "name".to_string(),
                        value_json: "\"Rust\"".to_string(),
                    },
                    Property {
                        key: "description".to_string(),
                        value_json: "\"A systems programming language\"".to_string(),
                    },
                ],
                embedding: vec![],
                entity_key: "rust".to_string(),
                scope: None,
            }))
            .await
            .unwrap()
            .into_inner()
            .node_id;

        svc.add_edge(Request::new(AddEdgeRequest {
            graph: "ctx_rerank".to_string(),
            source: alice_id,
            target: rust_id,
            label: "uses".to_string(),
            weight: 1.0,
            scope: None,
        }))
        .await
        .unwrap();

        let resp = svc
            .context_query(Request::new(ContextQueryRequest {
                graph: "ctx_rerank".to_string(),
                query: "systems programming".to_string(),
                embedding: vec![],
                seed_nodes: vec!["alice".to_string()],
                budget: 4096,
                max_depth: 2,
                include_provenance: false,
                retrieval_mode: weav_proto::grpc::RetrievalMode::Local as i32,
                rerank: Some(RerankConfigProto {
                    enabled: Some(true),
                    provider: "cross_encoder".to_string(),
                    model: "bge-reranker-v2-m3".to_string(),
                    candidate_limit: Some(5),
                    score_weight: Some(1.0),
                }),
                scope: None,
            }))
            .await
            .unwrap();

        let result = resp.into_inner();
        assert_eq!(result.chunks[0].node_id, rust_id);
    }

    #[tokio::test]
    async fn test_grpc_subscribe_events_replays_and_streams_live() {
        let svc = make_service();

        svc.create_graph(Request::new(CreateGraphRequest {
            name: "events_g".to_string(),
            scope: None,
        }))
        .await
        .unwrap();

        let first_node = svc
            .add_node(Request::new(AddNodeRequest {
                graph: "events_g".to_string(),
                label: "person".to_string(),
                properties: vec![Property {
                    key: "name".to_string(),
                    value_json: "\"Alice\"".to_string(),
                }],
                embedding: vec![],
                entity_key: String::new(),
                scope: None,
            }))
            .await
            .unwrap()
            .into_inner()
            .node_id;
        assert_eq!(first_node, 1);

        let response = svc
            .subscribe_events(Request::new(SubscribeEventsRequest {
                graph: "events_g".to_string(),
                since_sequence: 0,
                replay_limit: 10,
            }))
            .await
            .unwrap();
        let mut stream = response.into_inner();

        let graph_created = stream.next().await.unwrap().unwrap();
        assert_eq!(graph_created.kind, "graph_created");
        assert_eq!(graph_created.graph, "events_g");

        let node_created = stream.next().await.unwrap().unwrap();
        assert_eq!(node_created.kind, "node_created");
        let payload: serde_json::Value = serde_json::from_str(&node_created.payload_json).unwrap();
        assert_eq!(payload["node_id"], 1);

        svc.add_node(Request::new(AddNodeRequest {
            graph: "events_g".to_string(),
            label: "person".to_string(),
            properties: vec![Property {
                key: "name".to_string(),
                value_json: "\"Bob\"".to_string(),
            }],
            embedding: vec![],
            entity_key: String::new(),
            scope: None,
        }))
        .await
        .unwrap();

        let live_event = tokio::time::timeout(std::time::Duration::from_secs(1), stream.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(live_event.kind, "node_created");
        let payload: serde_json::Value = serde_json::from_str(&live_event.payload_json).unwrap();
        assert_eq!(payload["node_id"], 2);
    }

    #[tokio::test]
    async fn test_grpc_subscribe_events_replay_from_sequence() {
        let svc = make_service();

        svc.create_graph(Request::new(CreateGraphRequest {
            name: "events_seq".to_string(),
            scope: None,
        }))
        .await
        .unwrap();
        svc.add_node(Request::new(AddNodeRequest {
            graph: "events_seq".to_string(),
            label: "person".to_string(),
            properties: vec![Property {
                key: "name".to_string(),
                value_json: "\"Alice\"".to_string(),
            }],
            embedding: vec![],
            entity_key: String::new(),
            scope: None,
        }))
        .await
        .unwrap();
        svc.add_node(Request::new(AddNodeRequest {
            graph: "events_seq".to_string(),
            label: "person".to_string(),
            properties: vec![Property {
                key: "name".to_string(),
                value_json: "\"Bob\"".to_string(),
            }],
            embedding: vec![],
            entity_key: String::new(),
            scope: None,
        }))
        .await
        .unwrap();

        let recent = svc.engine.replay_events(Some("events_seq"), 0, 10);
        assert_eq!(recent.len(), 3);
        let cutoff = recent[1].sequence;

        let response = svc
            .subscribe_events(Request::new(SubscribeEventsRequest {
                graph: "events_seq".to_string(),
                since_sequence: cutoff,
                replay_limit: 10,
            }))
            .await
            .unwrap();
        let mut stream = response.into_inner();

        let replayed = stream.next().await.unwrap().unwrap();
        assert_eq!(replayed.kind, "node_created");
        let payload: serde_json::Value = serde_json::from_str(&replayed.payload_json).unwrap();
        assert_eq!(payload["node_id"], 2);
    }

    #[tokio::test]
    async fn test_grpc_subscribe_events_requires_auth_for_global_stream() {
        let svc = make_authed_service();

        let status = svc
            .subscribe_events(Request::new(SubscribeEventsRequest {
                graph: String::new(),
                since_sequence: 0,
                replay_limit: 1,
            }))
            .await
            .unwrap_err();
        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }

    #[tokio::test]
    async fn test_grpc_lifecycle() {
        let svc = make_service();

        // Create graph.
        svc.create_graph(Request::new(CreateGraphRequest {
            name: "test".to_string(),
            scope: None,
        }))
        .await
        .unwrap();

        // Add node.
        let add_resp = svc
            .add_node(Request::new(AddNodeRequest {
                graph: "test".to_string(),
                label: "person".to_string(),
                properties: vec![Property {
                    key: "name".to_string(),
                    value_json: "\"Alice\"".to_string(),
                }],
                embedding: vec![],
                entity_key: "alice".to_string(),
                scope: None,
            }))
            .await
            .unwrap();
        let node_id = add_resp.into_inner().node_id;
        assert!(node_id >= 1);

        // Get node.
        let get_resp = svc
            .get_node(Request::new(GetNodeRequest {
                graph: "test".to_string(),
                node_id,
                scope: None,
            }))
            .await
            .unwrap();
        let node = get_resp.into_inner();
        assert_eq!(node.node_id, node_id);
        assert_eq!(node.label, "person");

        // List graphs.
        let list_resp = svc
            .list_graphs(Request::new(ListGraphsRequest {}))
            .await
            .unwrap();
        assert_eq!(list_resp.into_inner().names, vec!["test"]);
    }
}
