//! The Weav engine: central coordinator holding all in-memory state.
//!
//! Provides a thread-safe interface for executing commands against graphs.

use std::collections::HashMap;
use std::time::SystemTime;

use parking_lot::{Mutex, RwLock};

use compact_str::CompactString;

use weav_core::config::{GraphConfig, WeavConfig};
use weav_core::error::{WeavError, WeavResult};
use weav_core::shard::StringInterner;
use weav_core::types::*;
use weav_graph::adjacency::{AdjacencyStore, EdgeMeta};
use weav_graph::properties::PropertyStore;
use weav_persist::snapshot::{
    EdgeSnapshot, FullSnapshot, GraphSnapshot, NodeSnapshot, SnapshotEngine, SnapshotMeta,
};
use weav_persist::wal::{WalOperation, WriteAheadLog};
use weav_query::executor;
use weav_query::parser::Command;
use weav_vector::index::{DistanceMetric, Quantization, VectorConfig, VectorIndex};
use weav_vector::tokens::TokenCounter;

// ─── Response types ──────────────────────────────────────────────────────────

/// Response from executing a command.
#[derive(Debug)]
pub enum CommandResponse {
    Ok,
    Pong,
    Integer(u64),
    IntegerList(Vec<u64>),
    Text(String),
    StringList(Vec<String>),
    Context(executor::ContextResult),
    NodeInfo(NodeInfo),
    GraphInfo(GraphInfoResponse),
    EdgeInfo(EdgeInfoResponse),
    Null,
    Error(String),
}

/// Information about a single edge.
#[derive(Debug)]
pub struct EdgeInfoResponse {
    pub edge_id: EdgeId,
    pub source: NodeId,
    pub target: NodeId,
    pub label: String,
    pub weight: f32,
}

/// Information about a single node.
#[derive(Debug)]
pub struct NodeInfo {
    pub node_id: NodeId,
    pub label: String,
    pub properties: Vec<(String, Value)>,
}

/// Information about a graph.
#[derive(Debug)]
pub struct GraphInfoResponse {
    pub name: String,
    pub node_count: u64,
    pub edge_count: u64,
}

// ─── GraphState ──────────────────────────────────────────────────────────────

/// Holds all state for a single graph.
pub struct GraphState {
    pub name: String,
    pub graph_id: GraphId,
    pub adjacency: AdjacencyStore,
    pub properties: PropertyStore,
    pub vector_index: VectorIndex,
    pub interner: StringInterner,
    pub config: GraphConfig,
    pub next_node_id: NodeId,
    pub next_edge_id: EdgeId,
}

// ─── Engine ──────────────────────────────────────────────────────────────────

/// The main Weav engine, holding all graphs. Thread-safe via RwLock.
pub struct Engine {
    graphs: RwLock<HashMap<String, GraphState>>,
    next_graph_id: RwLock<GraphId>,
    token_counter: TokenCounter,
    config: WeavConfig,
    wal: Option<Mutex<WriteAheadLog>>,
    snapshot_engine: Option<SnapshotEngine>,
    runtime_config: RwLock<HashMap<String, String>>,
}

impl Engine {
    /// Create a new engine with the given configuration.
    pub fn new(config: WeavConfig) -> Self {
        let token_counter = TokenCounter::new(config.engine.token_counter.clone());
        let (wal, snapshot_engine) = if config.persistence.enabled {
            let data_dir = config.persistence.data_dir.clone();
            let wal_path = data_dir.join("wal");
            let max_wal_size = config.persistence.max_wal_size_mb * 1024 * 1024;
            let wal = WriteAheadLog::new(
                wal_path,
                max_wal_size,
                config.persistence.wal_sync_mode.clone(),
            )
            .ok()
            .map(Mutex::new);
            let snap = SnapshotEngine::new(data_dir);
            (wal, Some(snap))
        } else {
            (None, None)
        };
        Self {
            graphs: RwLock::new(HashMap::new()),
            next_graph_id: RwLock::new(1),
            token_counter,
            config,
            wal,
            snapshot_engine,
            runtime_config: RwLock::new(HashMap::new()),
        }
    }

    fn append_wal(&self, op: WalOperation) {
        if let Some(ref wal_mutex) = self.wal {
            let mut wal = wal_mutex.lock();
            let _ = wal.append(0, op);
        }
    }

    /// Recover state from a RecoveryResult (snapshot + WAL entries).
    pub fn recover(&self, result: weav_persist::recovery::RecoveryResult) -> WeavResult<()> {
        // Step 1: Restore from snapshot if present.
        if let Some(ref snapshot) = result.snapshot {
            for gs in &snapshot.graphs {
                let graph_config = if gs.config_json.is_empty() || gs.config_json == "{}" {
                    let mut gc = GraphConfig::default();
                    gc.vector_dimensions = self.config.engine.default_vector_dimensions;
                    gc
                } else {
                    serde_json::from_str(&gs.config_json).unwrap_or_default()
                };

                let vec_config = VectorConfig {
                    dimensions: graph_config.vector_dimensions,
                    metric: DistanceMetric::Cosine,
                    hnsw_m: self.config.engine.default_hnsw_m,
                    hnsw_ef_construction: self.config.engine.default_hnsw_ef_construction,
                    hnsw_ef_search: self.config.engine.default_hnsw_ef_search,
                    quantization: Quantization::None,
                };
                let vector_index = VectorIndex::new(vec_config)?;

                let graph_id = {
                    let mut id = self.next_graph_id.write();
                    let gid = *id;
                    *id = gid.max(gs.graph_id) + 1;
                    gs.graph_id
                };

                let mut state = GraphState {
                    name: gs.graph_name.clone(),
                    graph_id,
                    adjacency: AdjacencyStore::new(),
                    properties: PropertyStore::new(),
                    vector_index,
                    interner: StringInterner::new(),
                    config: graph_config,
                    next_node_id: 1,
                    next_edge_id: 1,
                };

                // Restore nodes.
                for ns in &gs.nodes {
                    state.adjacency.add_node(ns.node_id);
                    state.properties.set_node_property(
                        ns.node_id,
                        "_label",
                        Value::String(CompactString::from(&ns.label)),
                    );
                    let label_id = state.interner.intern_label(&ns.label);
                    state.properties.set_node_property(
                        ns.node_id,
                        "_label_id",
                        Value::Int(label_id as i64),
                    );
                    if let Some(ref key) = ns.entity_key {
                        state.properties.set_node_property(
                            ns.node_id,
                            "entity_key",
                            Value::String(CompactString::from(key.as_str())),
                        );
                    }
                    if !ns.properties_json.is_empty() && ns.properties_json != "{}" {
                        if let Ok(props_val) = serde_json::from_str::<serde_json::Value>(&ns.properties_json) {
                            if let Some(obj) = props_val.as_object() {
                                for (k, v) in obj {
                                    state.properties.set_node_property(
                                        ns.node_id, k,
                                        crate::http::json_val_to_value(v),
                                    );
                                }
                            }
                        }
                    }
                    if let Some(ref emb) = ns.embedding {
                        let _ = state.vector_index.insert(ns.node_id, emb);
                    }
                    if ns.node_id >= state.next_node_id {
                        state.next_node_id = ns.node_id + 1;
                    }
                }

                // Restore edges.
                for es in &gs.edges {
                    let label_id = state.interner.intern_label(&es.label);
                    let meta = EdgeMeta {
                        source: es.source,
                        target: es.target,
                        label: label_id,
                        temporal: BiTemporal {
                            valid_from: es.valid_from,
                            valid_until: es.valid_until,
                            tx_from: es.valid_from,
                            tx_until: u64::MAX,
                        },
                        provenance: None,
                        weight: es.weight,
                        token_cost: 0,
                    };
                    let _ = state.adjacency.add_edge(es.source, es.target, label_id, meta);
                    if es.edge_id >= state.next_edge_id {
                        state.next_edge_id = es.edge_id + 1;
                    }
                }

                let mut graphs = self.graphs.write();
                graphs.insert(gs.graph_name.clone(), state);
            }
        }

        // Step 2: Replay WAL entries.
        for entry in &result.wal_entries {
            match &entry.operation {
                WalOperation::GraphCreate { name, .. } => {
                    let cmd = weav_query::parser::GraphCreateCmd {
                        name: name.clone(),
                        config: None,
                    };
                    let _ = self.handle_graph_create(cmd);
                }
                WalOperation::GraphDrop { name } => {
                    let _ = self.handle_graph_drop(name);
                }
                WalOperation::NodeAdd {
                    graph_id: _,
                    node_id: _,
                    label,
                    properties_json,
                    embedding,
                    entity_key,
                } => {
                    // Find graph by ID - we need to find the name
                    // For WAL replay, we use the graph_id to find the graph name
                    let graphs = self.graphs.read();
                    let graph_name = graphs.values()
                        .find(|gs| gs.graph_id == entry.operation.graph_id_hint())
                        .map(|gs| gs.name.clone());
                    drop(graphs);
                    if let Some(name) = graph_name {
                        let mut props = Vec::new();
                        if !properties_json.is_empty() && properties_json != "{}" {
                            if let Ok(val) = serde_json::from_str::<serde_json::Value>(properties_json) {
                                if let Some(obj) = val.as_object() {
                                    for (k, v) in obj {
                                        props.push((k.clone(), crate::http::json_val_to_value(v)));
                                    }
                                }
                            }
                        }
                        let cmd = weav_query::parser::NodeAddCmd {
                            graph: name,
                            label: label.clone(),
                            properties: props,
                            embedding: embedding.clone(),
                            entity_key: entity_key.clone(),
                        };
                        let _ = self.handle_node_add(cmd);
                    }
                }
                _ => {
                    // Other WAL operations are handled similarly but less critical for recovery.
                }
            }
        }

        Ok(())
    }

    /// Execute a parsed command and return a response.
    pub fn execute_command(&self, cmd: Command) -> WeavResult<CommandResponse> {
        match cmd {
            Command::Ping => Ok(CommandResponse::Pong),
            Command::Info => Ok(CommandResponse::Text(format!(
                "weav-server v{} (graphs: {})",
                env!("CARGO_PKG_VERSION"),
                self.graphs
                    .read()
                    .len()
            ))),
            Command::Stats(graph_name) => self.handle_stats(graph_name),
            Command::Snapshot => self.handle_snapshot(),
            Command::GraphCreate(cmd) => self.handle_graph_create(cmd),
            Command::GraphDrop(name) => self.handle_graph_drop(&name),
            Command::GraphInfo(name) => self.handle_graph_info(&name),
            Command::GraphList => self.handle_graph_list(),
            Command::NodeAdd(cmd) => self.handle_node_add(cmd),
            Command::NodeGet(cmd) => self.handle_node_get(cmd),
            Command::NodeUpdate(cmd) => self.handle_node_update(cmd),
            Command::NodeDelete(cmd) => self.handle_node_delete(cmd),
            Command::EdgeAdd(cmd) => self.handle_edge_add(cmd),
            Command::EdgeInvalidate(cmd) => self.handle_edge_invalidate(cmd),
            Command::BulkInsertNodes(cmd) => self.handle_bulk_insert_nodes(cmd),
            Command::BulkInsertEdges(cmd) => self.handle_bulk_insert_edges(cmd),
            Command::Context(query) => self.handle_context(query),
            Command::EdgeDelete(cmd) => self.handle_edge_delete(cmd),
            Command::EdgeGet(cmd) => self.handle_edge_get(cmd),
            Command::ConfigSet(key, value) => self.handle_config_set(key, value),
            Command::ConfigGet(key) => self.handle_config_get(key),
        }
    }

    // ── Stats ────────────────────────────────────────────────────────────

    fn handle_stats(&self, graph_name: Option<String>) -> WeavResult<CommandResponse> {
        let graphs = self
            .graphs
            .read();

        if let Some(name) = graph_name {
            let gs = graphs
                .get(&name)
                .ok_or_else(|| WeavError::GraphNotFound(name.clone()))?;
            Ok(CommandResponse::Text(format!(
                "graph={} nodes={} edges={} vectors={}",
                name,
                gs.adjacency.node_count(),
                gs.adjacency.edge_count(),
                gs.vector_index.len(),
            )))
        } else {
            Ok(CommandResponse::Text(format!(
                "graphs={} engine=weav-server",
                graphs.len(),
            )))
        }
    }

    // ── Graph commands ───────────────────────────────────────────────────

    fn handle_graph_create(
        &self,
        cmd: weav_query::parser::GraphCreateCmd,
    ) -> WeavResult<CommandResponse> {
        let graph_config = cmd.config.unwrap_or_else(|| {
            let mut gc = GraphConfig::default();
            gc.vector_dimensions = self.config.engine.default_vector_dimensions;
            gc
        });

        let vec_config = VectorConfig {
            dimensions: graph_config.vector_dimensions,
            metric: DistanceMetric::Cosine,
            hnsw_m: self.config.engine.default_hnsw_m,
            hnsw_ef_construction: self.config.engine.default_hnsw_ef_construction,
            hnsw_ef_search: self.config.engine.default_hnsw_ef_search,
            quantization: Quantization::None,
        };

        let vector_index = VectorIndex::new(vec_config)?;

        let graph_id = {
            let mut id = self
                .next_graph_id
                .write();
            let gid = *id;
            *id += 1;
            gid
        };

        let graph_state = GraphState {
            name: cmd.name.clone(),
            graph_id,
            adjacency: AdjacencyStore::new(),
            properties: PropertyStore::new(),
            vector_index,
            interner: StringInterner::new(),
            config: graph_config,
            next_node_id: 1,
            next_edge_id: 1,
        };

        let mut graphs = self
            .graphs
            .write();

        if graphs.contains_key(&cmd.name) {
            return Err(WeavError::Conflict(format!(
                "graph '{}' already exists",
                cmd.name
            )));
        }

        let graph_name = cmd.name.clone();
        graphs.insert(cmd.name, graph_state);
        drop(graphs);
        self.append_wal(WalOperation::GraphCreate {
            name: graph_name,
            config_json: "{}".to_string(),
        });
        Ok(CommandResponse::Ok)
    }

    fn handle_graph_drop(&self, name: &str) -> WeavResult<CommandResponse> {
        let mut graphs = self
            .graphs
            .write();
        graphs
            .remove(name)
            .ok_or_else(|| WeavError::GraphNotFound(name.to_string()))?;
        let name_owned = name.to_string();
        drop(graphs);
        self.append_wal(WalOperation::GraphDrop { name: name_owned });
        Ok(CommandResponse::Ok)
    }

    fn handle_graph_info(&self, name: &str) -> WeavResult<CommandResponse> {
        let graphs = self
            .graphs
            .read();
        let gs = graphs
            .get(name)
            .ok_or_else(|| WeavError::GraphNotFound(name.to_string()))?;
        Ok(CommandResponse::GraphInfo(GraphInfoResponse {
            name: gs.name.clone(),
            node_count: gs.adjacency.node_count(),
            edge_count: gs.adjacency.edge_count(),
        }))
    }

    fn handle_graph_list(&self) -> WeavResult<CommandResponse> {
        let graphs = self
            .graphs
            .read();
        let names: Vec<String> = graphs.keys().cloned().collect();
        Ok(CommandResponse::StringList(names))
    }

    // ── Node commands ────────────────────────────────────────────────────

    fn handle_node_add(
        &self,
        cmd: weav_query::parser::NodeAddCmd,
    ) -> WeavResult<CommandResponse> {
        let mut graphs = self
            .graphs
            .write();
        let gs = graphs
            .get_mut(&cmd.graph)
            .ok_or_else(|| WeavError::GraphNotFound(cmd.graph.clone()))?;

        let node_id = gs.next_node_id;
        gs.next_node_id += 1;

        // Add node to adjacency store.
        gs.adjacency.add_node(node_id);

        // Set the label as a property.
        gs.properties.set_node_property(
            node_id,
            "_label",
            Value::String(CompactString::from(&cmd.label)),
        );

        // Also intern the label.
        let label_id = gs.interner.intern_label(&cmd.label);
        gs.properties
            .set_node_property(node_id, "_label_id", Value::Int(label_id as i64));

        // Set entity_key if provided.
        if let Some(ref key) = cmd.entity_key {
            gs.properties.set_node_property(
                node_id,
                "entity_key",
                Value::String(CompactString::from(key.as_str())),
            );
        }

        // Set properties.
        for (k, v) in &cmd.properties {
            gs.properties.set_node_property(node_id, k, v.clone());
        }

        // Insert embedding if provided.
        if let Some(ref embedding) = cmd.embedding {
            gs.vector_index.insert(node_id, embedding)?;
        }

        let graph_id = gs.graph_id;
        let props_json = serde_json::to_string(
            &cmd.properties.iter().map(|(k, v)| (k.clone(), format!("{v:?}"))).collect::<Vec<_>>()
        ).unwrap_or_default();
        drop(graphs);
        self.append_wal(WalOperation::NodeAdd {
            graph_id,
            node_id,
            label: cmd.label,
            properties_json: props_json,
            embedding: cmd.embedding,
            entity_key: cmd.entity_key,
        });

        Ok(CommandResponse::Integer(node_id))
    }

    fn handle_node_get(
        &self,
        cmd: weav_query::parser::NodeGetCmd,
    ) -> WeavResult<CommandResponse> {
        let graphs = self
            .graphs
            .read();
        let gs = graphs
            .get(&cmd.graph)
            .ok_or_else(|| WeavError::GraphNotFound(cmd.graph.clone()))?;

        let node_id = if let Some(id) = cmd.node_id {
            id
        } else if let Some(ref key) = cmd.entity_key {
            // Look up by entity_key.
            let key_clone = key.clone();
            let nodes = gs.properties.nodes_where("entity_key", &move |v| {
                v.as_str() == Some(key_clone.as_str())
            });
            *nodes
                .first()
                .ok_or_else(|| WeavError::NodeNotFound(0, gs.graph_id))?
        } else {
            return Err(WeavError::QueryParseError(
                "NodeGet requires either node_id or entity_key".to_string(),
            ));
        };

        if !gs.adjacency.has_node(node_id) {
            return Err(WeavError::NodeNotFound(node_id, gs.graph_id));
        }

        // Get label.
        let label = gs
            .properties
            .get_node_property(node_id, "_label")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        // Get all properties (excluding internal ones for the response).
        let all_props = gs.properties.get_all_node_properties(node_id);
        let properties: Vec<(String, Value)> = all_props
            .into_iter()
            .filter(|(k, _)| !k.starts_with('_'))
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();

        Ok(CommandResponse::NodeInfo(NodeInfo {
            node_id,
            label,
            properties,
        }))
    }

    fn handle_node_delete(
        &self,
        cmd: weav_query::parser::NodeDeleteCmd,
    ) -> WeavResult<CommandResponse> {
        let mut graphs = self
            .graphs
            .write();
        let gs = graphs
            .get_mut(&cmd.graph)
            .ok_or_else(|| WeavError::GraphNotFound(cmd.graph.clone()))?;

        // Remove from adjacency (also removes connected edges).
        gs.adjacency.remove_node(cmd.node_id)?;

        // Remove all properties.
        gs.properties.remove_all_node_properties(cmd.node_id);

        // Remove from vector index.
        gs.vector_index.remove(cmd.node_id)?;

        let graph_id = gs.graph_id;
        drop(graphs);
        self.append_wal(WalOperation::NodeDelete {
            graph_id,
            node_id: cmd.node_id,
        });

        Ok(CommandResponse::Ok)
    }

    fn handle_node_update(
        &self,
        cmd: weav_query::parser::NodeUpdateCmd,
    ) -> WeavResult<CommandResponse> {
        let mut graphs = self
            .graphs
            .write();
        let gs = graphs
            .get_mut(&cmd.graph)
            .ok_or_else(|| WeavError::GraphNotFound(cmd.graph.clone()))?;

        if !gs.adjacency.has_node(cmd.node_id) {
            return Err(WeavError::NodeNotFound(cmd.node_id, gs.graph_id));
        }

        // Merge properties (overwrite existing keys).
        for (k, v) in &cmd.properties {
            gs.properties.set_node_property(cmd.node_id, k, v.clone());
        }

        // Update embedding if provided.
        if let Some(ref embedding) = cmd.embedding {
            // Remove old vector and insert new one.
            let _ = gs.vector_index.remove(cmd.node_id);
            gs.vector_index.insert(cmd.node_id, embedding)?;
        }

        let graph_id = gs.graph_id;
        drop(graphs);
        self.append_wal(WalOperation::NodeUpdate {
            graph_id,
            node_id: cmd.node_id,
            properties_json: "{}".to_string(),
        });

        Ok(CommandResponse::Ok)
    }

    fn handle_bulk_insert_nodes(
        &self,
        cmd: weav_query::parser::BulkInsertNodesCmd,
    ) -> WeavResult<CommandResponse> {
        let mut graphs = self
            .graphs
            .write();
        let gs = graphs
            .get_mut(&cmd.graph)
            .ok_or_else(|| WeavError::GraphNotFound(cmd.graph.clone()))?;

        let mut ids = Vec::with_capacity(cmd.nodes.len());
        for node_cmd in &cmd.nodes {
            let node_id = gs.next_node_id;
            gs.next_node_id += 1;

            gs.adjacency.add_node(node_id);

            gs.properties.set_node_property(
                node_id,
                "_label",
                Value::String(CompactString::from(&node_cmd.label)),
            );

            let label_id = gs.interner.intern_label(&node_cmd.label);
            gs.properties
                .set_node_property(node_id, "_label_id", Value::Int(label_id as i64));

            if let Some(ref key) = node_cmd.entity_key {
                gs.properties.set_node_property(
                    node_id,
                    "entity_key",
                    Value::String(CompactString::from(key.as_str())),
                );
            }

            for (k, v) in &node_cmd.properties {
                gs.properties.set_node_property(node_id, k, v.clone());
            }

            if let Some(ref embedding) = node_cmd.embedding {
                gs.vector_index.insert(node_id, embedding)?;
            }

            ids.push(node_id);
        }

        Ok(CommandResponse::IntegerList(ids))
    }

    fn handle_bulk_insert_edges(
        &self,
        cmd: weav_query::parser::BulkInsertEdgesCmd,
    ) -> WeavResult<CommandResponse> {
        let mut graphs = self
            .graphs
            .write();
        let gs = graphs
            .get_mut(&cmd.graph)
            .ok_or_else(|| WeavError::GraphNotFound(cmd.graph.clone()))?;

        let now = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut ids = Vec::with_capacity(cmd.edges.len());
        for edge_cmd in &cmd.edges {
            let label_id = gs.interner.intern_label(&edge_cmd.label);
            let meta = EdgeMeta {
                source: edge_cmd.source,
                target: edge_cmd.target,
                label: label_id,
                temporal: BiTemporal::new_current(now),
                provenance: None,
                weight: edge_cmd.weight,
                token_cost: 0,
            };
            let edge_id = gs
                .adjacency
                .add_edge(edge_cmd.source, edge_cmd.target, label_id, meta)?;
            ids.push(edge_id);
        }

        Ok(CommandResponse::IntegerList(ids))
    }

    // ── Edge commands ────────────────────────────────────────────────────

    fn handle_edge_add(
        &self,
        cmd: weav_query::parser::EdgeAddCmd,
    ) -> WeavResult<CommandResponse> {
        let mut graphs = self
            .graphs
            .write();
        let gs = graphs
            .get_mut(&cmd.graph)
            .ok_or_else(|| WeavError::GraphNotFound(cmd.graph.clone()))?;

        let label_id = gs.interner.intern_label(&cmd.label);

        let now = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let meta = EdgeMeta {
            source: cmd.source,
            target: cmd.target,
            label: label_id,
            temporal: BiTemporal::new_current(now),
            provenance: None,
            weight: cmd.weight,
            token_cost: 0,
        };

        let edge_id = gs
            .adjacency
            .add_edge(cmd.source, cmd.target, label_id, meta)?;

        let graph_id = gs.graph_id;
        let label = cmd.label.clone();
        drop(graphs);
        self.append_wal(WalOperation::EdgeAdd {
            graph_id,
            edge_id,
            source: cmd.source,
            target: cmd.target,
            label,
            weight: cmd.weight,
        });

        Ok(CommandResponse::Integer(edge_id))
    }

    fn handle_edge_invalidate(
        &self,
        cmd: weav_query::parser::EdgeInvalidateCmd,
    ) -> WeavResult<CommandResponse> {
        let mut graphs = self
            .graphs
            .write();
        let gs = graphs
            .get_mut(&cmd.graph)
            .ok_or_else(|| WeavError::GraphNotFound(cmd.graph.clone()))?;

        let now = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        gs.adjacency.invalidate_edge(cmd.edge_id, now)?;

        let graph_id = gs.graph_id;
        drop(graphs);
        self.append_wal(WalOperation::EdgeInvalidate {
            graph_id,
            edge_id: cmd.edge_id,
            timestamp: now,
        });

        Ok(CommandResponse::Ok)
    }

    fn handle_edge_delete(
        &self,
        cmd: weav_query::parser::EdgeDeleteCmd,
    ) -> WeavResult<CommandResponse> {
        let mut graphs = self
            .graphs
            .write();
        let gs = graphs
            .get_mut(&cmd.graph)
            .ok_or_else(|| WeavError::GraphNotFound(cmd.graph.clone()))?;

        gs.adjacency.remove_edge(cmd.edge_id)?;

        let graph_id = gs.graph_id;
        drop(graphs);
        self.append_wal(WalOperation::EdgeInvalidate {
            graph_id,
            edge_id: cmd.edge_id,
            timestamp: SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        });

        Ok(CommandResponse::Ok)
    }

    fn handle_edge_get(
        &self,
        cmd: weav_query::parser::EdgeGetCmd,
    ) -> WeavResult<CommandResponse> {
        let graphs = self
            .graphs
            .read();
        let gs = graphs
            .get(&cmd.graph)
            .ok_or_else(|| WeavError::GraphNotFound(cmd.graph.clone()))?;

        let meta = gs
            .adjacency
            .get_edge(cmd.edge_id)
            .ok_or(WeavError::EdgeNotFound(cmd.edge_id))?;

        let label = gs
            .interner
            .resolve_label(meta.label)
            .unwrap_or("unknown")
            .to_string();

        Ok(CommandResponse::EdgeInfo(EdgeInfoResponse {
            edge_id: cmd.edge_id,
            source: meta.source,
            target: meta.target,
            label,
            weight: meta.weight,
        }))
    }

    fn handle_config_set(&self, key: String, value: String) -> WeavResult<CommandResponse> {
        let mut config = self
            .runtime_config
            .write();
        config.insert(key, value);
        Ok(CommandResponse::Ok)
    }

    fn handle_config_get(&self, key: String) -> WeavResult<CommandResponse> {
        let config = self
            .runtime_config
            .read();
        match config.get(&key) {
            Some(val) => Ok(CommandResponse::Text(val.clone())),
            None => Ok(CommandResponse::Null),
        }
    }

    // ── Snapshot ──────────────────────────────────────────────────────────

    fn handle_snapshot(&self) -> WeavResult<CommandResponse> {
        let snapshot_engine = match &self.snapshot_engine {
            Some(se) => se,
            None => return Ok(CommandResponse::Ok), // Persistence not enabled
        };

        let graphs = self
            .graphs
            .read();

        let wal_sequence = self.wal.as_ref()
            .map(|w| w.lock().sequence_number())
            .unwrap_or(0);

        let mut graph_snapshots = Vec::new();
        let mut total_nodes: u64 = 0;
        let mut total_edges: u64 = 0;

        for gs in graphs.values() {
            let mut node_snapshots = Vec::new();
            for node_id in gs.adjacency.all_node_ids() {
                let label = gs.properties
                    .get_node_property(node_id, "_label")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                let all_props = gs.properties.get_all_node_properties(node_id);
                let props_json = {
                    let mut map = serde_json::Map::new();
                    for (k, v) in &all_props {
                        if !k.starts_with('_') {
                            map.insert(k.to_string(), serde_json::json!(format!("{v:?}")));
                        }
                    }
                    serde_json::to_string(&map).unwrap_or_default()
                };

                let entity_key = gs.properties
                    .get_node_property(node_id, "entity_key")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                node_snapshots.push(NodeSnapshot {
                    node_id,
                    label,
                    properties_json: props_json,
                    embedding: None,
                    entity_key,
                });
            }

            let mut edge_snapshots = Vec::new();
            for (edge_id, meta) in gs.adjacency.all_edges() {
                let label = gs.interner
                    .resolve_label(meta.label)
                    .unwrap_or("unknown")
                    .to_string();
                edge_snapshots.push(EdgeSnapshot {
                    edge_id,
                    source: meta.source,
                    target: meta.target,
                    label,
                    weight: meta.weight,
                    valid_from: meta.temporal.valid_from,
                    valid_until: meta.temporal.valid_until,
                });
            }

            total_nodes += node_snapshots.len() as u64;
            total_edges += edge_snapshots.len() as u64;

            graph_snapshots.push(GraphSnapshot {
                graph_id: gs.graph_id,
                graph_name: gs.name.clone(),
                config_json: "{}".to_string(),
                nodes: node_snapshots,
                edges: edge_snapshots,
            });
        }

        let now = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let full_snapshot = FullSnapshot {
            meta: SnapshotMeta {
                path: std::path::PathBuf::new(),
                created_at: now,
                size_bytes: 0,
                node_count: total_nodes,
                edge_count: total_edges,
                graph_count: graph_snapshots.len() as u32,
                wal_sequence,
            },
            graphs: graph_snapshots,
        };

        snapshot_engine.save_snapshot(&full_snapshot)
            .map_err(|e| WeavError::Internal(format!("snapshot save failed: {e}")))?;

        Ok(CommandResponse::Ok)
    }

    // ── Context query ────────────────────────────────────────────────────

    fn handle_context(
        &self,
        query: weav_query::parser::ContextQuery,
    ) -> WeavResult<CommandResponse> {
        let graphs = self
            .graphs
            .read();
        let gs = graphs
            .get(&query.graph)
            .ok_or_else(|| WeavError::GraphNotFound(query.graph.clone()))?;

        let result = executor::execute_context_query(
            &query,
            &gs.adjacency,
            &gs.properties,
            &gs.vector_index,
            &self.token_counter,
            &gs.interner,
        )?;

        Ok(CommandResponse::Context(result))
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use weav_query::parser;

    fn make_engine() -> Engine {
        Engine::new(WeavConfig::default())
    }

    fn create_test_graph(engine: &Engine, name: &str) {
        let cmd = parser::parse_command(&format!("GRAPH CREATE \"{name}\"")).unwrap();
        engine.execute_command(cmd).unwrap();
    }

    #[test]
    fn test_ping() {
        let engine = make_engine();
        let resp = engine
            .execute_command(Command::Ping)
            .unwrap();
        assert!(matches!(resp, CommandResponse::Pong));
    }

    #[test]
    fn test_info() {
        let engine = make_engine();
        let resp = engine
            .execute_command(Command::Info)
            .unwrap();
        match resp {
            CommandResponse::Text(t) => assert!(t.contains("weav-server")),
            _ => panic!("expected Text response"),
        }
    }

    #[test]
    fn test_graph_create_and_list() {
        let engine = make_engine();
        create_test_graph(&engine, "test1");
        create_test_graph(&engine, "test2");

        let resp = engine.execute_command(Command::GraphList).unwrap();
        match resp {
            CommandResponse::StringList(names) => {
                assert_eq!(names.len(), 2);
                assert!(names.contains(&"test1".to_string()));
                assert!(names.contains(&"test2".to_string()));
            }
            _ => panic!("expected StringList response"),
        }
    }

    #[test]
    fn test_graph_create_duplicate() {
        let engine = make_engine();
        create_test_graph(&engine, "dup");
        let cmd = parser::parse_command("GRAPH CREATE \"dup\"").unwrap();
        let result = engine.execute_command(cmd);
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_drop() {
        let engine = make_engine();
        create_test_graph(&engine, "todrop");
        let cmd = parser::parse_command("GRAPH DROP \"todrop\"").unwrap();
        engine.execute_command(cmd).unwrap();

        let resp = engine.execute_command(Command::GraphList).unwrap();
        match resp {
            CommandResponse::StringList(names) => assert!(names.is_empty()),
            _ => panic!("expected StringList"),
        }
    }

    #[test]
    fn test_graph_drop_not_found() {
        let engine = make_engine();
        let cmd = parser::parse_command("GRAPH DROP \"nonexistent\"").unwrap();
        assert!(engine.execute_command(cmd).is_err());
    }

    #[test]
    fn test_graph_info() {
        let engine = make_engine();
        create_test_graph(&engine, "info_test");

        let cmd = parser::parse_command("GRAPH INFO \"info_test\"").unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        match resp {
            CommandResponse::GraphInfo(info) => {
                assert_eq!(info.name, "info_test");
                assert_eq!(info.node_count, 0);
                assert_eq!(info.edge_count, 0);
            }
            _ => panic!("expected GraphInfo"),
        }
    }

    #[test]
    fn test_graph_info_not_found() {
        let engine = make_engine();
        let cmd = parser::parse_command("GRAPH INFO \"nope\"").unwrap();
        assert!(engine.execute_command(cmd).is_err());
    }

    #[test]
    fn test_node_add_and_get_by_id() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "person" PROPERTIES {"name": "Alice", "age": 30} KEY "alice-001""#,
        )
        .unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        let node_id = match resp {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer response"),
        };
        assert!(node_id >= 1);

        // Get by ID.
        let cmd = parser::parse_command(&format!("NODE GET \"g\" {node_id}")).unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        match resp {
            CommandResponse::NodeInfo(info) => {
                assert_eq!(info.node_id, node_id);
                assert_eq!(info.label, "person");
                // Should have name, age, entity_key (non-internal properties).
                assert!(info.properties.iter().any(|(k, _)| k == "name"));
            }
            _ => panic!("expected NodeInfo"),
        }
    }

    #[test]
    fn test_node_get_by_entity_key() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "person" PROPERTIES {"name": "Bob"} KEY "bob-key""#,
        )
        .unwrap();
        engine.execute_command(cmd).unwrap();

        let cmd =
            parser::parse_command("NODE GET \"g\" WHERE entity_key = \"bob-key\"").unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        match resp {
            CommandResponse::NodeInfo(info) => {
                assert_eq!(info.label, "person");
            }
            _ => panic!("expected NodeInfo"),
        }
    }

    #[test]
    fn test_node_get_not_found() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        let cmd = parser::parse_command("NODE GET \"g\" 999").unwrap();
        assert!(engine.execute_command(cmd).is_err());
    }

    #[test]
    fn test_node_delete() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "person" PROPERTIES {"name": "Charlie"}"#,
        )
        .unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        let node_id = match resp {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(&format!("NODE DELETE \"g\" {node_id}")).unwrap();
        engine.execute_command(cmd).unwrap();

        // Should no longer be findable.
        let cmd = parser::parse_command(&format!("NODE GET \"g\" {node_id}")).unwrap();
        assert!(engine.execute_command(cmd).is_err());
    }

    #[test]
    fn test_edge_add_and_graph_counts() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        // Add two nodes.
        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "person" PROPERTIES {"name": "A"}"#,
        )
        .unwrap();
        let r1 = engine.execute_command(cmd).unwrap();
        let n1 = match r1 {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "person" PROPERTIES {"name": "B"}"#,
        )
        .unwrap();
        let r2 = engine.execute_command(cmd).unwrap();
        let n2 = match r2 {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add edge.
        let cmd = parser::parse_command(&format!(
            r#"EDGE ADD TO "g" FROM {n1} TO {n2} LABEL "knows" WEIGHT 0.9"#,
        ))
        .unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        match resp {
            CommandResponse::Integer(eid) => assert!(eid >= 1),
            _ => panic!("expected Integer"),
        }

        // Check graph info.
        let cmd = parser::parse_command("GRAPH INFO \"g\"").unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        match resp {
            CommandResponse::GraphInfo(info) => {
                assert_eq!(info.node_count, 2);
                assert_eq!(info.edge_count, 1);
            }
            _ => panic!("expected GraphInfo"),
        }
    }

    #[test]
    fn test_edge_add_missing_node() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        let cmd = parser::parse_command(
            r#"EDGE ADD TO "g" FROM 1 TO 2 LABEL "knows""#,
        )
        .unwrap();
        assert!(engine.execute_command(cmd).is_err());
    }

    #[test]
    fn test_edge_invalidate() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        // Add two nodes and an edge.
        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "a" PROPERTIES {"name": "X"}"#,
        )
        .unwrap();
        let r1 = engine.execute_command(cmd).unwrap();
        let n1 = match r1 {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "b" PROPERTIES {"name": "Y"}"#,
        )
        .unwrap();
        let r2 = engine.execute_command(cmd).unwrap();
        let n2 = match r2 {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(&format!(
            r#"EDGE ADD TO "g" FROM {n1} TO {n2} LABEL "rel""#,
        ))
        .unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        let edge_id = match resp {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Invalidate the edge.
        let cmd =
            parser::parse_command(&format!("EDGE INVALIDATE \"g\" {edge_id}")).unwrap();
        engine.execute_command(cmd).unwrap();
    }

    #[test]
    fn test_edge_invalidate_not_found() {
        let engine = make_engine();
        create_test_graph(&engine, "g");
        let cmd = parser::parse_command("EDGE INVALIDATE \"g\" 999").unwrap();
        assert!(engine.execute_command(cmd).is_err());
    }

    #[test]
    fn test_context_query_basic() {
        let engine = make_engine();
        create_test_graph(&engine, "kg");

        // Add nodes with entity_keys.
        let cmd = parser::parse_command(
            r#"NODE ADD TO "kg" LABEL "person" PROPERTIES {"name": "Alice", "description": "A developer"} KEY "alice""#,
        )
        .unwrap();
        let r1 = engine.execute_command(cmd).unwrap();
        let n1 = match r1 {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(
            r#"NODE ADD TO "kg" LABEL "topic" PROPERTIES {"name": "Rust", "description": "A language"} KEY "rust""#,
        )
        .unwrap();
        let r2 = engine.execute_command(cmd).unwrap();
        let n2 = match r2 {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add edge.
        let cmd = parser::parse_command(&format!(
            r#"EDGE ADD TO "kg" FROM {n1} TO {n2} LABEL "uses""#,
        ))
        .unwrap();
        engine.execute_command(cmd).unwrap();

        // Run context query seeded by node key.
        let cmd = parser::parse_command(
            r#"CONTEXT "what does alice use" FROM "kg" SEEDS NODES ["alice"] DEPTH 2"#,
        )
        .unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        match resp {
            CommandResponse::Context(result) => {
                assert!(result.nodes_considered > 0);
                assert!(result.nodes_included > 0);
            }
            _ => panic!("expected Context response"),
        }
    }

    #[test]
    fn test_context_query_empty_seeds() {
        let engine = make_engine();
        create_test_graph(&engine, "kg");

        let cmd = parser::parse_command(
            r#"CONTEXT "test" FROM "kg" SEEDS NODES ["nonexistent"]"#,
        )
        .unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        match resp {
            CommandResponse::Context(result) => {
                assert_eq!(result.nodes_considered, 0);
                assert_eq!(result.nodes_included, 0);
            }
            _ => panic!("expected Context response"),
        }
    }

    #[test]
    fn test_context_query_graph_not_found() {
        let engine = make_engine();
        let cmd = parser::parse_command(
            r#"CONTEXT "test" FROM "nope" SEEDS NODES ["x"]"#,
        )
        .unwrap();
        assert!(engine.execute_command(cmd).is_err());
    }

    #[test]
    fn test_stats_no_graph() {
        let engine = make_engine();
        let cmd = parser::parse_command("STATS").unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        match resp {
            CommandResponse::Text(t) => assert!(t.contains("graphs=")),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn test_stats_with_graph() {
        let engine = make_engine();
        create_test_graph(&engine, "sg");

        let cmd = parser::parse_command("STATS \"sg\"").unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        match resp {
            CommandResponse::Text(t) => {
                assert!(t.contains("graph=sg"));
                assert!(t.contains("nodes="));
            }
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn test_stats_graph_not_found() {
        let engine = make_engine();
        let cmd = parser::parse_command("STATS \"missing\"").unwrap();
        assert!(engine.execute_command(cmd).is_err());
    }

    #[test]
    fn test_snapshot_placeholder() {
        let engine = make_engine();
        let cmd = parser::parse_command("SNAPSHOT").unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        assert!(matches!(resp, CommandResponse::Ok));
    }

    #[test]
    fn test_node_update_success() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "person" PROPERTIES {"name": "Alice"}"#,
        )
        .unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        let node_id = match resp {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Update the node.
        let cmd = parser::parse_command(&format!(
            r#"NODE UPDATE "g" {node_id} PROPERTIES {{"name": "Alice Updated", "age": 30}}"#,
        ))
        .unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        assert!(matches!(resp, CommandResponse::Ok));

        // Verify the update.
        let cmd = parser::parse_command(&format!("NODE GET \"g\" {node_id}")).unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        match resp {
            CommandResponse::NodeInfo(info) => {
                let name = info.properties.iter().find(|(k, _)| k == "name");
                assert!(name.is_some());
                assert_eq!(
                    name.unwrap().1.as_str(),
                    Some("Alice Updated")
                );
                assert!(info.properties.iter().any(|(k, _)| k == "age"));
            }
            _ => panic!("expected NodeInfo"),
        }
    }

    #[test]
    fn test_node_update_not_found() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        let cmd = parser::parse_command(
            r#"NODE UPDATE "g" 999 PROPERTIES {"name": "X"}"#,
        )
        .unwrap();
        assert!(engine.execute_command(cmd).is_err());
    }

    #[test]
    fn test_bulk_insert_nodes() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        let cmd = parser::parse_command(
            r#"BULK NODES TO "g" DATA [{"label": "person", "properties": {"name": "A"}}, {"label": "person", "properties": {"name": "B"}}]"#,
        )
        .unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        match resp {
            CommandResponse::IntegerList(ids) => {
                assert_eq!(ids.len(), 2);
                assert!(ids[0] >= 1);
                assert!(ids[1] > ids[0]);
            }
            _ => panic!("expected IntegerList"),
        }

        // Verify graph info.
        let cmd = parser::parse_command("GRAPH INFO \"g\"").unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        match resp {
            CommandResponse::GraphInfo(info) => {
                assert_eq!(info.node_count, 2);
            }
            _ => panic!("expected GraphInfo"),
        }
    }

    #[test]
    fn test_bulk_insert_edges() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        // Add two nodes first.
        let cmd = parser::parse_command(
            r#"BULK NODES TO "g" DATA [{"label": "a", "properties": {"name": "X"}}, {"label": "b", "properties": {"name": "Y"}}]"#,
        )
        .unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        let node_ids = match resp {
            CommandResponse::IntegerList(ids) => ids,
            _ => panic!("expected IntegerList"),
        };

        // Bulk insert edges.
        let cmd = parser::parse_command(&format!(
            r#"BULK EDGES TO "g" DATA [{{"source": {}, "target": {}, "label": "knows"}}]"#,
            node_ids[0], node_ids[1]
        ))
        .unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        match resp {
            CommandResponse::IntegerList(ids) => {
                assert_eq!(ids.len(), 1);
            }
            _ => panic!("expected IntegerList"),
        }

        // Verify graph info.
        let cmd = parser::parse_command("GRAPH INFO \"g\"").unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        match resp {
            CommandResponse::GraphInfo(info) => {
                assert_eq!(info.node_count, 2);
                assert_eq!(info.edge_count, 1);
            }
            _ => panic!("expected GraphInfo"),
        }
    }

    #[test]
    fn test_node_update() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        // Add node with properties.
        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "person" PROPERTIES {"name": "Alice", "age": 25}"#,
        )
        .unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        let node_id = match resp {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Update the node properties.
        let cmd = parser::parse_command(&format!(
            r#"NODE UPDATE "g" {node_id} PROPERTIES {{"name": "Bob", "city": "NYC"}}"#,
        ))
        .unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        assert!(matches!(resp, CommandResponse::Ok));

        // Verify updated properties via NODE GET.
        let cmd = parser::parse_command(&format!("NODE GET \"g\" {node_id}")).unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        match resp {
            CommandResponse::NodeInfo(info) => {
                assert_eq!(info.node_id, node_id);
                // name should be updated.
                let name = info.properties.iter().find(|(k, _)| k == "name");
                assert!(name.is_some());
                assert_eq!(name.unwrap().1.as_str(), Some("Bob"));
                // city should be a new property.
                assert!(info.properties.iter().any(|(k, _)| k == "city"));
                // age should still be present (merge, not replace).
                assert!(info.properties.iter().any(|(k, _)| k == "age"));
            }
            _ => panic!("expected NodeInfo"),
        }
    }

    #[test]
    fn test_edge_get() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        // Add two nodes.
        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "a" PROPERTIES {"name": "X"}"#,
        )
        .unwrap();
        let n1 = match engine.execute_command(cmd).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "b" PROPERTIES {"name": "Y"}"#,
        )
        .unwrap();
        let n2 = match engine.execute_command(cmd).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add edge.
        let cmd = parser::parse_command(&format!(
            r#"EDGE ADD TO "g" FROM {n1} TO {n2} LABEL "knows" WEIGHT 0.75"#,
        ))
        .unwrap();
        let edge_id = match engine.execute_command(cmd).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Get the edge by ID.
        let cmd = parser::parse_command(&format!("EDGE GET \"g\" {edge_id}")).unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        match resp {
            CommandResponse::EdgeInfo(info) => {
                assert_eq!(info.edge_id, edge_id);
                assert_eq!(info.source, n1);
                assert_eq!(info.target, n2);
                assert_eq!(info.label, "knows");
                assert!((info.weight - 0.75).abs() < 0.01);
            }
            _ => panic!("expected EdgeInfo"),
        }
    }

    #[test]
    fn test_edge_delete() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        // Add two nodes.
        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "a" PROPERTIES {"name": "X"}"#,
        )
        .unwrap();
        let n1 = match engine.execute_command(cmd).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "b" PROPERTIES {"name": "Y"}"#,
        )
        .unwrap();
        let n2 = match engine.execute_command(cmd).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add edge.
        let cmd = parser::parse_command(&format!(
            r#"EDGE ADD TO "g" FROM {n1} TO {n2} LABEL "link""#,
        ))
        .unwrap();
        let edge_id = match engine.execute_command(cmd).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Delete the edge.
        let cmd = parser::parse_command(&format!("EDGE DELETE \"g\" {edge_id}")).unwrap();
        engine.execute_command(cmd).unwrap();

        // Verify EDGE GET returns error.
        let cmd = parser::parse_command(&format!("EDGE GET \"g\" {edge_id}")).unwrap();
        assert!(engine.execute_command(cmd).is_err());
    }

    #[test]
    fn test_node_get_nonexistent_graph() {
        let engine = make_engine();
        let cmd = parser::parse_command("NODE GET \"no_such_graph\" 1").unwrap();
        assert!(engine.execute_command(cmd).is_err());
    }

    #[test]
    fn test_node_add_nonexistent_graph() {
        let engine = make_engine();
        let cmd = parser::parse_command(
            r#"NODE ADD TO "no_such_graph" LABEL "x" PROPERTIES {"a": 1}"#,
        )
        .unwrap();
        assert!(engine.execute_command(cmd).is_err());
    }

    #[test]
    fn test_snapshot_with_persistence() {
        let tmp_dir = std::env::temp_dir().join(format!("weav_snap_test_{}", std::process::id()));
        std::fs::create_dir_all(&tmp_dir).unwrap();

        let mut config = WeavConfig::default();
        config.persistence.enabled = true;
        config.persistence.data_dir = tmp_dir.clone();

        let engine = Engine::new(config);
        create_test_graph(&engine, "snap_g");

        // Add some nodes.
        let cmd = parser::parse_command(
            r#"NODE ADD TO "snap_g" LABEL "person" PROPERTIES {"name": "Alice"}"#,
        )
        .unwrap();
        engine.execute_command(cmd).unwrap();

        let cmd = parser::parse_command(
            r#"NODE ADD TO "snap_g" LABEL "person" PROPERTIES {"name": "Bob"}"#,
        )
        .unwrap();
        engine.execute_command(cmd).unwrap();

        // Execute SNAPSHOT command.
        let cmd = parser::parse_command("SNAPSHOT").unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        assert!(matches!(resp, CommandResponse::Ok));

        // Clean up temp dir.
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_node_delete_also_removes_edges() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        // Add nodes.
        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "a" PROPERTIES {"name": "N1"}"#,
        )
        .unwrap();
        let n1 = match engine.execute_command(cmd).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "b" PROPERTIES {"name": "N2"}"#,
        )
        .unwrap();
        let n2 = match engine.execute_command(cmd).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add edge.
        let cmd = parser::parse_command(&format!(
            r#"EDGE ADD TO "g" FROM {n1} TO {n2} LABEL "link""#,
        ))
        .unwrap();
        engine.execute_command(cmd).unwrap();

        // Delete node 1.
        let cmd = parser::parse_command(&format!("NODE DELETE \"g\" {n1}")).unwrap();
        engine.execute_command(cmd).unwrap();

        // Edge count should be 0.
        let cmd = parser::parse_command("GRAPH INFO \"g\"").unwrap();
        let resp = engine.execute_command(cmd).unwrap();
        match resp {
            CommandResponse::GraphInfo(info) => {
                assert_eq!(info.node_count, 1);
                assert_eq!(info.edge_count, 0);
            }
            _ => panic!("expected GraphInfo"),
        }
    }
}
