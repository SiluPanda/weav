//! gRPC server implementation for the Weav context graph database.

use std::sync::Arc;

use compact_str::CompactString;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use weav_core::types::Value;
use weav_proto::grpc::weav_service_server::WeavService;
use weav_proto::grpc::*;
use weav_core::types::Direction;
use weav_query::parser::{
    Command, ContextQuery, EdgeAddCmd, EdgeInvalidateCmd, GraphCreateCmd,
    NodeAddCmd, NodeDeleteCmd, NodeGetCmd, NodeUpdateCmd,
    BulkInsertNodesCmd, BulkInsertEdgesCmd, SeedStrategy,
};

use crate::engine::{CommandResponse, Engine};

pub struct WeavGrpcService {
    pub engine: Arc<Engine>,
}

fn weav_error_to_status(err: weav_core::error::WeavError) -> Status {
    match &err {
        weav_core::error::WeavError::GraphNotFound(_) => Status::not_found(err.to_string()),
        weav_core::error::WeavError::NodeNotFound(_, _) => Status::not_found(err.to_string()),
        weav_core::error::WeavError::EdgeNotFound(_) => Status::not_found(err.to_string()),
        weav_core::error::WeavError::DuplicateNode(_) => {
            Status::already_exists(err.to_string())
        }
        weav_core::error::WeavError::Conflict(_) => Status::already_exists(err.to_string()),
        weav_core::error::WeavError::QueryParseError(_) => {
            Status::invalid_argument(err.to_string())
        }
        weav_core::error::WeavError::DimensionMismatch { .. } => {
            Status::invalid_argument(err.to_string())
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
        let req = request.into_inner();
        let cmd = Command::GraphCreate(GraphCreateCmd {
            name: req.name,
            config: None,
        });
        self.engine
            .execute_command(cmd)
            .map_err(weav_error_to_status)?;
        Ok(Response::new(CreateGraphResponse {}))
    }

    async fn drop_graph(
        &self,
        request: Request<DropGraphRequest>,
    ) -> Result<Response<DropGraphResponse>, Status> {
        let req = request.into_inner();
        let cmd = Command::GraphDrop(req.name);
        self.engine
            .execute_command(cmd)
            .map_err(weav_error_to_status)?;
        Ok(Response::new(DropGraphResponse {}))
    }

    async fn list_graphs(
        &self,
        _: Request<ListGraphsRequest>,
    ) -> Result<Response<ListGraphsResponse>, Status> {
        match self
            .engine
            .execute_command(Command::GraphList)
            .map_err(weav_error_to_status)?
        {
            CommandResponse::StringList(names) => {
                Ok(Response::new(ListGraphsResponse { names }))
            }
            _ => Err(Status::internal("unexpected response type")),
        }
    }

    async fn graph_info(
        &self,
        request: Request<GraphInfoRequest>,
    ) -> Result<Response<GraphInfoResponse>, Status> {
        let req = request.into_inner();
        let cmd = Command::GraphInfo(req.name);
        match self
            .engine
            .execute_command(cmd)
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
        let req = request.into_inner();
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
            graph: req.graph,
            label: req.label,
            properties,
            embedding,
            entity_key,
        });

        match self
            .engine
            .execute_command(cmd)
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
        let req = request.into_inner();
        let cmd = Command::NodeGet(NodeGetCmd {
            graph: req.graph,
            node_id: Some(req.node_id),
            entity_key: None,
        });

        match self
            .engine
            .execute_command(cmd)
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
        let req = request.into_inner();
        let properties = props_from_proto(&req.properties);
        let embedding = if req.embedding.is_empty() {
            None
        } else {
            Some(req.embedding)
        };

        let cmd = Command::NodeUpdate(NodeUpdateCmd {
            graph: req.graph,
            node_id: req.node_id,
            properties,
            embedding,
        });

        self.engine
            .execute_command(cmd)
            .map_err(weav_error_to_status)?;
        Ok(Response::new(UpdateNodeResponse {}))
    }

    async fn delete_node(
        &self,
        request: Request<DeleteNodeRequest>,
    ) -> Result<Response<DeleteNodeResponse>, Status> {
        let req = request.into_inner();
        let cmd = Command::NodeDelete(NodeDeleteCmd {
            graph: req.graph,
            node_id: req.node_id,
        });

        self.engine
            .execute_command(cmd)
            .map_err(weav_error_to_status)?;
        Ok(Response::new(DeleteNodeResponse {}))
    }

    async fn add_edge(
        &self,
        request: Request<AddEdgeRequest>,
    ) -> Result<Response<AddEdgeResponse>, Status> {
        let req = request.into_inner();
        let cmd = Command::EdgeAdd(EdgeAddCmd {
            graph: req.graph,
            source: req.source,
            target: req.target,
            label: req.label,
            weight: req.weight,
            properties: Vec::new(),
        });

        match self
            .engine
            .execute_command(cmd)
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
        let req = request.into_inner();
        let cmd = Command::EdgeInvalidate(EdgeInvalidateCmd {
            graph: req.graph,
            edge_id: req.edge_id,
        });

        self.engine
            .execute_command(cmd)
            .map_err(weav_error_to_status)?;
        Ok(Response::new(InvalidateEdgeResponse {}))
    }

    async fn context_query(
        &self,
        request: Request<ContextQueryRequest>,
    ) -> Result<Response<ContextQueryResponse>, Status> {
        let req = request.into_inner();

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
            graph: req.graph,
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
        };

        let cmd = Command::Context(query);
        match self
            .engine
            .execute_command(cmd)
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
        let req = request.into_inner();
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
                    graph: req.graph.clone(),
                    label: n.label,
                    properties,
                    embedding,
                    entity_key,
                }
            })
            .collect();

        let cmd = Command::BulkInsertNodes(BulkInsertNodesCmd {
            graph: req.graph,
            nodes,
        });

        match self
            .engine
            .execute_command(cmd)
            .map_err(weav_error_to_status)?
        {
            CommandResponse::IntegerList(ids) => {
                Ok(Response::new(BulkAddNodesResponse { ids }))
            }
            _ => Err(Status::internal("unexpected response type")),
        }
    }

    async fn bulk_add_edges(
        &self,
        request: Request<BulkAddEdgesRequest>,
    ) -> Result<Response<BulkAddEdgesResponse>, Status> {
        let req = request.into_inner();
        let edges: Vec<EdgeAddCmd> = req
            .edges
            .into_iter()
            .map(|e| EdgeAddCmd {
                graph: req.graph.clone(),
                source: e.source,
                target: e.target,
                label: e.label,
                weight: e.weight,
                properties: Vec::new(),
            })
            .collect();

        let cmd = Command::BulkInsertEdges(BulkInsertEdgesCmd {
            graph: req.graph,
            edges,
        });

        match self
            .engine
            .execute_command(cmd)
            .map_err(weav_error_to_status)?
        {
            CommandResponse::IntegerList(ids) => {
                Ok(Response::new(BulkAddEdgesResponse { ids }))
            }
            _ => Err(Status::internal("unexpected response type")),
        }
    }

    type ContextQueryStreamStream = ReceiverStream<Result<ContextChunkProto, Status>>;

    async fn context_query_stream(
        &self,
        request: Request<ContextQueryRequest>,
    ) -> Result<Response<Self::ContextQueryStreamStream>, Status> {
        let req = request.into_inner();

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
            graph: req.graph,
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
        };

        let cmd = Command::Context(query);
        let result = match self
            .engine
            .execute_command(cmd)
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

    async fn snapshot(
        &self,
        _request: Request<SnapshotRequest>,
    ) -> Result<Response<SnapshotResponse>, Status> {
        let cmd = Command::Snapshot;
        match self.engine.execute_command(cmd) {
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

    async fn info(
        &self,
        _request: Request<InfoRequest>,
    ) -> Result<Response<InfoResponse>, Status> {
        let cmd = Command::Info;
        let version = match self
            .engine
            .execute_command(cmd)
            .map_err(weav_error_to_status)?
        {
            CommandResponse::Text(text) => text,
            _ => "unknown".to_string(),
        };

        // Get graph list to count graphs and total nodes/edges.
        let (graph_count, total_nodes, total_edges) = match self
            .engine
            .execute_command(Command::GraphList)
            .map_err(weav_error_to_status)?
        {
            CommandResponse::StringList(names) => {
                let mut nodes = 0u64;
                let mut edges = 0u64;
                let count = names.len() as u32;
                for name in &names {
                    if let Ok(CommandResponse::GraphInfo(info)) =
                        self.engine.execute_command(Command::GraphInfo(name.clone()))
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
    use weav_core::config::WeavConfig;
    use weav_proto::grpc::weav_service_server::WeavService;

    fn make_service() -> WeavGrpcService {
        WeavGrpcService {
            engine: Arc::new(Engine::new(WeavConfig::default())),
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
            }))
            .await
            .unwrap();
        let result = resp.into_inner();
        // The query should complete successfully with some chunks.
        // The query should complete without error. With seed nodes, it should consider at least one node.
        assert!(result.nodes_considered > 0);
    }

    #[tokio::test]
    async fn test_grpc_lifecycle() {
        let svc = make_service();

        // Create graph.
        svc.create_graph(Request::new(CreateGraphRequest {
            name: "test".to_string(),
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
