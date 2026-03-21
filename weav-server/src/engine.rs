//! The Weav engine: central coordinator holding all in-memory state.
//!
//! Provides a thread-safe interface for executing commands against graphs.

use std::collections::HashMap;
use std::sync::Arc;
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
    IngestResult(IngestResultResponse),
    Null,
    Error(String),
}

/// Result of an INGEST operation.
#[derive(Debug)]
pub struct IngestResultResponse {
    pub document_id: String,
    pub chunks_created: usize,
    pub entities_created: usize,
    pub entities_merged: usize,
    pub relationships_created: usize,
    pub pipeline_duration_ms: u64,
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
    pub dedup_config: Option<weav_graph::dedup::DedupConfig>,
}

// ─── Engine ──────────────────────────────────────────────────────────────────

/// The main Weav engine, holding all graphs. Thread-safe via per-graph RwLock.
///
/// The outer `RwLock` protects the registry (graph creation/drop). The inner
/// per-graph `Arc<RwLock<GraphState>>` allows concurrent operations on
/// different graphs without contention.
pub struct Engine {
    graphs: RwLock<HashMap<String, Arc<RwLock<GraphState>>>>,
    next_graph_id: RwLock<GraphId>,
    token_counter: TokenCounter,
    config: WeavConfig,
    wal: Option<Mutex<WriteAheadLog>>,
    snapshot_engine: Option<SnapshotEngine>,
    runtime_config: RwLock<HashMap<String, String>>,
    acl_store: Option<weav_auth::acl::AclStore>,
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
        let acl_store = if config.auth.enabled {
            Some(weav_auth::acl::AclStore::from_config(&config.auth)
                .expect("failed to initialize auth — check user password configuration"))
        } else {
            None
        };

        Self {
            graphs: RwLock::new(HashMap::new()),
            next_graph_id: RwLock::new(1),
            token_counter,
            config,
            wal,
            snapshot_engine,
            runtime_config: RwLock::new(HashMap::new()),
            acl_store,
        }
    }

    /// Get a reference to the ACL store, returning an error if auth is not enabled.
    fn acl_store(&self) -> WeavResult<&weav_auth::acl::AclStore> {
        self.acl_store.as_ref().ok_or_else(|| {
            WeavError::Internal("auth not enabled".into())
        })
    }

    /// Get the current timestamp in milliseconds since epoch.
    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    fn append_wal(&self, op: WalOperation) -> WeavResult<()> {
        if let Some(ref wal_mutex) = self.wal {
            let mut wal = wal_mutex.lock();
            wal.append(0, op)
                .map_err(|e| WeavError::PersistenceError(format!("WAL write failed: {e}")))?;
        }
        Ok(())
    }

    /// Force-sync the WAL file to disk. Called periodically for EverySecond mode.
    pub fn sync_wal(&self) -> WeavResult<()> {
        if let Some(ref wal_mutex) = self.wal {
            let mut wal = wal_mutex.lock();
            wal.sync()
                .map_err(|e| WeavError::PersistenceError(format!("WAL sync failed: {e}")))?;
        }
        Ok(())
    }

    /// Return the configured WAL sync mode.
    pub fn wal_sync_mode(&self) -> &weav_core::config::WalSyncMode {
        &self.config.persistence.wal_sync_mode
    }

    /// Sweep expired nodes and edges across all graphs.
    /// Called periodically to enforce TTL.
    /// Returns the total number of expired entities removed.
    pub fn sweep_ttl(&self) -> u64 {
        let now = Self::now_ms();
        let mut total_expired = 0u64;

        let graph_arcs: Vec<(String, Arc<RwLock<GraphState>>)> = {
            let registry = self.graphs.read();
            registry.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        };

        for (graph_name, graph_arc) in &graph_arcs {
            let mut gs = graph_arc.write();

            // Sweep expired nodes (those with _ttl_expires_at <= now)
            let expired_nodes: Vec<NodeId> = gs.properties
                .nodes_where("_ttl_expires_at", &|v| {
                    match v {
                        Value::Timestamp(ts) => *ts <= now,
                        Value::Int(ts) => (*ts as u64) <= now,
                        _ => false,
                    }
                });

            for node_id in &expired_nodes {
                let _ = gs.adjacency.remove_node(*node_id);
                gs.properties.remove_all_node_properties(*node_id);
                let _ = gs.vector_index.remove(*node_id);
            }
            total_expired += expired_nodes.len() as u64;

            // Sweep expired edges (those with valid_until <= now)
            let expired_edges: Vec<EdgeId> = gs.adjacency
                .all_edges()
                .filter(|(_, meta)| {
                    meta.temporal.valid_until != BiTemporal::OPEN
                        && meta.temporal.valid_until <= now
                })
                .map(|(eid, _)| eid)
                .collect();

            for edge_id in &expired_edges {
                let _ = gs.adjacency.remove_edge(*edge_id);
            }
            total_expired += expired_edges.len() as u64;

            // Update metrics after sweep
            if !expired_nodes.is_empty() || !expired_edges.is_empty() {
                crate::metrics::NODES_TOTAL
                    .with_label_values(&[graph_name])
                    .set(gs.adjacency.node_count() as i64);
                crate::metrics::EDGES_TOTAL
                    .with_label_values(&[graph_name])
                    .set(gs.adjacency.edge_count() as i64);
            }
        }

        total_expired
    }

    /// Look up a graph by name, returning a cloned `Arc` to its per-graph lock.
    ///
    /// The outer registry lock is held only for the duration of the HashMap
    /// lookup (nanoseconds). Callers then acquire the inner per-graph lock
    /// for the duration of their operation, allowing different graphs to be
    /// read/written concurrently.
    fn get_graph(&self, name: &str) -> WeavResult<Arc<RwLock<GraphState>>> {
        let registry = self.graphs.read();
        registry
            .get(name)
            .cloned()
            .ok_or_else(|| WeavError::GraphNotFound(name.to_string()))
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
                    dedup_config: None,
                };

                // Restore nodes.
                for ns in &gs.nodes {
                    state.adjacency.add_node(ns.node_id);
                    state.properties.set_node_property(
                        ns.node_id,
                        "_label",
                        Value::String(CompactString::from(&ns.label)),
                    );
                    let label_id = state.interner.intern_label(&ns.label)?;
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
                    let label_id = state.interner.intern_label(&es.label)?;
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
                graphs.insert(gs.graph_name.clone(), Arc::new(RwLock::new(state)));
            }
        }

        // Step 2: Replay WAL entries.
        for entry in &result.wal_entries {
            match &entry.operation {
                WalOperation::GraphCreate { name, config_json } => {
                    let config = if config_json.is_empty() || config_json == "{}" {
                        None
                    } else {
                        serde_json::from_str(config_json).ok()
                    };
                    let cmd = weav_query::parser::GraphCreateCmd {
                        name: name.clone(),
                        config,
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
                    let graph_name = {
                        let registry = self.graphs.read();
                        registry.values()
                            .find_map(|arc| {
                                let gs = arc.read();
                                if gs.graph_id == entry.operation.graph_id_hint() {
                                    Some(gs.name.clone())
                                } else {
                                    None
                                }
                            })
                    };
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
                            ttl_ms: None,
                        };
                        let _ = self.handle_node_add(cmd);
                    }
                }
                WalOperation::NodeUpdate {
                    graph_id,
                    node_id,
                    properties_json,
                } => {
                    let registry = self.graphs.read();
                    let graph_arc = registry.values()
                        .find(|arc| arc.read().graph_id == *graph_id)
                        .cloned();
                    drop(registry);
                    if let Some(graph_arc) = graph_arc {
                        let mut gs = graph_arc.write();
                        if gs.adjacency.has_node(*node_id) {
                            if !properties_json.is_empty() && properties_json != "{}" {
                                if let Ok(val) = serde_json::from_str::<serde_json::Value>(properties_json) {
                                    if let Some(obj) = val.as_object() {
                                        for (k, v) in obj {
                                            gs.properties.set_node_property(
                                                *node_id, k,
                                                crate::http::json_val_to_value(v),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                WalOperation::NodeDelete {
                    graph_id,
                    node_id,
                } => {
                    let registry = self.graphs.read();
                    let graph_arc = registry.values()
                        .find(|arc| arc.read().graph_id == *graph_id)
                        .cloned();
                    drop(registry);
                    if let Some(graph_arc) = graph_arc {
                        let mut gs = graph_arc.write();
                        if gs.adjacency.has_node(*node_id) {
                            let _ = gs.adjacency.remove_node(*node_id);
                            gs.properties.remove_all_node_properties(*node_id);
                            let _ = gs.vector_index.remove(*node_id);
                        }
                    }
                }
                WalOperation::EdgeAdd {
                    graph_id,
                    edge_id,
                    source,
                    target,
                    label,
                    weight,
                } => {
                    let registry = self.graphs.read();
                    let graph_arc = registry.values()
                        .find(|arc| arc.read().graph_id == *graph_id)
                        .cloned();
                    drop(registry);
                    if let Some(graph_arc) = graph_arc {
                        let mut gs = graph_arc.write();
                        if let Ok(label_id) = gs.interner.intern_label(label) {
                            let now = Self::now_ms();
                            let meta = EdgeMeta {
                                source: *source,
                                target: *target,
                                label: label_id,
                                temporal: BiTemporal::new_current(now),
                                provenance: None,
                                weight: *weight,
                                token_cost: 0,
                            };
                            let _ = gs.adjacency.add_edge_with_id(*source, *target, label_id, meta, *edge_id);
                            if *edge_id >= gs.next_edge_id {
                                gs.next_edge_id = *edge_id + 1;
                            }
                        }
                    }
                }
                WalOperation::EdgeInvalidate {
                    graph_id,
                    edge_id,
                    timestamp,
                } => {
                    let registry = self.graphs.read();
                    let graph_arc = registry.values()
                        .find(|arc| arc.read().graph_id == *graph_id)
                        .cloned();
                    drop(registry);
                    if let Some(graph_arc) = graph_arc {
                        let mut gs = graph_arc.write();
                        let _ = gs.adjacency.invalidate_edge(*edge_id, *timestamp);
                    }
                }
                WalOperation::EdgeDelete {
                    graph_id,
                    edge_id,
                } => {
                    let registry = self.graphs.read();
                    let graph_arc = registry.values()
                        .find(|arc| arc.read().graph_id == *graph_id)
                        .cloned();
                    drop(registry);
                    if let Some(graph_arc) = graph_arc {
                        let mut gs = graph_arc.write();
                        let _ = gs.adjacency.remove_edge(*edge_id);
                    }
                }
                WalOperation::VectorUpdate {
                    graph_id,
                    node_id,
                    vector,
                } => {
                    let registry = self.graphs.read();
                    let graph_arc = registry.values()
                        .find(|arc| arc.read().graph_id == *graph_id)
                        .cloned();
                    drop(registry);
                    if let Some(graph_arc) = graph_arc {
                        let mut gs = graph_arc.write();
                        let _ = gs.vector_index.remove(*node_id);
                        let _ = gs.vector_index.insert(*node_id, vector);
                    }
                }
                WalOperation::Ingest { .. } => {
                    // Ingest WAL entries are metadata only; the actual data changes
                    // (NodeAdd, EdgeAdd, etc.) are recorded as separate WAL entries.
                }
            }
        }

        Ok(())
    }

    /// Whether auth is enabled.
    pub fn is_auth_enabled(&self) -> bool {
        self.acl_store.is_some()
    }

    /// Whether auth is required for all connections.
    pub fn is_auth_required(&self) -> bool {
        self.acl_store
            .as_ref()
            .map(|s| s.require_auth())
            .unwrap_or(false)
    }

    /// Authenticate with username + password.
    pub fn authenticate(
        &self,
        username: &str,
        password: &str,
    ) -> WeavResult<weav_auth::identity::SessionIdentity> {
        let store = self.acl_store()?;
        store.authenticate(username, password)
    }

    /// Authenticate with default password (Redis-compat).
    pub fn authenticate_default(
        &self,
        password: &str,
    ) -> WeavResult<weav_auth::identity::SessionIdentity> {
        let store = self.acl_store()?;
        store.authenticate_default(password)
    }

    /// Authenticate with an API key.
    pub fn authenticate_api_key(
        &self,
        key: &str,
    ) -> WeavResult<weav_auth::identity::SessionIdentity> {
        let store = self.acl_store()?;
        store.authenticate_api_key(key)
    }

    /// Execute a parsed command and return a response.
    ///
    /// When auth is enabled, `identity` must be `Some` (unless the command is
    /// AUTH or PING). When auth is disabled, pass `None`.
    pub fn execute_command(
        &self,
        cmd: Command,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        let cmd_type = cmd.type_name();
        let start = std::time::Instant::now();

        let result = self.execute_command_inner(cmd, identity);

        // Record metrics
        let elapsed = start.elapsed().as_secs_f64();
        let status = if result.is_ok() { "ok" } else { "error" };
        crate::metrics::QUERY_DURATION
            .with_label_values(&[cmd_type])
            .observe(elapsed);
        crate::metrics::QUERY_TOTAL
            .with_label_values(&[cmd_type, status])
            .inc();

        result
    }

    fn execute_command_inner(
        &self,
        cmd: Command,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        // Auth gate: check permissions if auth is enabled.
        if let Some(ref acl_store) = self.acl_store {
            // AUTH and PING are always allowed without identity.
            let is_auth_exempt = matches!(cmd, Command::Auth { .. } | Command::Ping);

            if !is_auth_exempt {
                if acl_store.require_auth() && identity.is_none() {
                    return Err(WeavError::AuthenticationRequired);
                }
                if let Some(id) = identity {
                    self.check_authorization(&cmd, id)?;
                }
            }
        }

        match cmd {
            Command::Ping => Ok(CommandResponse::Pong),
            Command::Info => Ok(CommandResponse::Text(format!(
                "weav-server v{} (graphs: {})",
                env!("CARGO_PKG_VERSION"),
                self.graphs
                    .read()
                    .len()
            ))),
            Command::Stats(graph_name) => self.handle_stats(graph_name, identity),
            Command::Snapshot => self.handle_snapshot(),
            Command::GraphCreate(cmd) => self.handle_graph_create_authed(cmd, identity),
            Command::GraphDrop(name) => self.handle_graph_drop_authed(&name, identity),
            Command::GraphInfo(name) => self.handle_graph_info_authed(&name, identity),
            Command::GraphList => self.handle_graph_list(),
            Command::NodeAdd(cmd) => self.handle_node_add_authed(cmd, identity),
            Command::NodeGet(cmd) => self.handle_node_get_authed(cmd, identity),
            Command::NodeUpdate(cmd) => self.handle_node_update_authed(cmd, identity),
            Command::NodeDelete(cmd) => self.handle_node_delete_authed(cmd, identity),
            Command::EdgeAdd(cmd) => self.handle_edge_add_authed(cmd, identity),
            Command::EdgeInvalidate(cmd) => self.handle_edge_invalidate_authed(cmd, identity),
            Command::BulkInsertNodes(cmd) => self.handle_bulk_insert_nodes_authed(cmd, identity),
            Command::BulkInsertEdges(cmd) => self.handle_bulk_insert_edges_authed(cmd, identity),
            Command::Context(query) => self.handle_context_authed(query, identity),
            Command::EdgeDelete(cmd) => self.handle_edge_delete_authed(cmd, identity),
            Command::EdgeGet(cmd) => self.handle_edge_get_authed(cmd, identity),
            Command::ConfigSet(key, value) => self.handle_config_set(key, value),
            Command::ConfigGet(key) => self.handle_config_get(key),
            Command::Auth { username, password } => self.handle_auth(username, password),
            Command::AclSetUser(cmd) => self.handle_acl_set_user(cmd),
            Command::AclDelUser(name) => self.handle_acl_del_user(&name),
            Command::AclList => self.handle_acl_list(),
            Command::AclGetUser(name) => self.handle_acl_get_user(&name),
            Command::AclWhoAmI => self.handle_acl_whoami(identity),
            Command::AclSave => self.handle_acl_save(),
            Command::AclLoad => self.handle_acl_load(),
            Command::Ingest(_) => Err(WeavError::Internal(
                "INGEST requires async execution; use execute_command_async".into(),
            )),
        }
    }

    /// Async command execution — handles INGEST (async LLM calls) and
    /// delegates all other commands to the sync `execute_command`.
    pub async fn execute_command_async(
        &self,
        cmd: Command,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        match cmd {
            Command::Ingest(ingest_cmd) => {
                // Auth gate (same as sync path).
                if let Some(ref acl_store) = self.acl_store {
                    if acl_store.require_auth() && identity.is_none() {
                        return Err(WeavError::AuthenticationRequired);
                    }
                    if let Some(id) = identity {
                        self.check_authorization(
                            &Command::Ingest(ingest_cmd.clone()),
                            id,
                        )?;
                    }
                }
                // Defense-in-depth: verify write permission on the target graph.
                self.check_permission(
                    identity,
                    &ingest_cmd.graph,
                    weav_auth::identity::GraphPermission::ReadWrite,
                )?;
                self.handle_ingest(ingest_cmd).await
            }
            other => self.execute_command(other, identity),
        }
    }

    /// Check authorization for a command against the user's identity.
    ///
    /// This is the centralized auth gate invoked at the `execute_command` entry
    /// point. It enforces both command-category permissions and graph-level
    /// permissions with three tiers:
    ///   - Admin operations (GraphCreate, GraphDrop) require `Admin` graph permission
    ///   - Write operations (NodeAdd, EdgeAdd, ...) require `ReadWrite` graph permission
    ///   - Read operations (NodeGet, Context, ...) require `Read` graph permission
    fn check_authorization(
        &self,
        cmd: &Command,
        identity: &weav_auth::identity::SessionIdentity,
    ) -> WeavResult<()> {
        use weav_auth::identity::CommandCategory;

        // Classify the command.
        let category = match cmd {
            Command::Ping | Command::Info | Command::Auth { .. } => CommandCategory::Connection,
            Command::NodeGet(_) | Command::EdgeGet(_) | Command::GraphInfo(_)
            | Command::GraphList | Command::Stats(_) | Command::Context(_)
            | Command::ConfigGet(_) | Command::AclWhoAmI => CommandCategory::Read,
            Command::NodeAdd(_) | Command::NodeUpdate(_) | Command::NodeDelete(_)
            | Command::EdgeAdd(_) | Command::EdgeDelete(_) | Command::EdgeInvalidate(_)
            | Command::BulkInsertNodes(_) | Command::BulkInsertEdges(_)
            | Command::Ingest(_) => CommandCategory::Write,
            Command::GraphCreate(_) | Command::GraphDrop(_) | Command::Snapshot
            | Command::ConfigSet(_, _) | Command::AclSetUser(_) | Command::AclDelUser(_)
            | Command::AclList | Command::AclGetUser(_) | Command::AclSave
            | Command::AclLoad => CommandCategory::Admin,
        };

        // Check category permission.
        if !identity.permissions.has_category(category) {
            return Err(WeavError::PermissionDenied(format!(
                "user '{}' lacks {:?} permission",
                identity.username, category
            )));
        }

        // Check graph-level permission for commands that target a specific graph.
        // Use three-tier checks to prevent privilege escalation:
        //   Admin commands  -> can_admin_graph  (GraphPermission::Admin)
        //   Write commands  -> can_write_graph  (GraphPermission::ReadWrite)
        //   Read commands   -> can_read_graph   (GraphPermission::Read)
        let graph_name = self.extract_graph_name_from_cmd(cmd);
        if let Some(ref graph) = graph_name {
            match category {
                CommandCategory::Admin => {
                    if !identity.permissions.can_admin_graph(graph) {
                        return Err(WeavError::PermissionDenied(format!(
                            "user '{}' lacks admin access to graph '{}'",
                            identity.username, graph
                        )));
                    }
                }
                CommandCategory::Write => {
                    if !identity.permissions.can_write_graph(graph) {
                        return Err(WeavError::PermissionDenied(format!(
                            "user '{}' lacks write access to graph '{}'",
                            identity.username, graph
                        )));
                    }
                }
                _ => {
                    if !identity.permissions.can_read_graph(graph) {
                        return Err(WeavError::PermissionDenied(format!(
                            "user '{}' lacks read access to graph '{}'",
                            identity.username, graph
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Defense-in-depth permission check for use inside individual handlers.
    ///
    /// Returns `Ok(())` immediately when:
    ///   - Auth is not enabled in the config, OR
    ///   - Auth does not require authentication (unauthenticated access allowed), OR
    ///   - No identity is provided (e.g. during WAL recovery)
    ///
    /// When auth is active and an identity is present, verifies the user holds
    /// at least `required` permission on `graph_name`.
    fn check_permission(
        &self,
        identity: Option<&weav_auth::identity::SessionIdentity>,
        graph_name: &str,
        required: weav_auth::identity::GraphPermission,
    ) -> WeavResult<()> {
        if !self.config.auth.enabled || !self.config.auth.require_auth {
            return Ok(());
        }
        let id = match identity {
            Some(id) => id,
            None => {
                // Auth required but no identity — this is an error when
                // reached through a user-facing path. During recovery the
                // config has auth disabled so we never get here.
                return Err(WeavError::AuthenticationRequired);
            }
        };
        let has_perm = match required {
            weav_auth::identity::GraphPermission::Admin => {
                id.permissions.can_admin_graph(graph_name)
            }
            weav_auth::identity::GraphPermission::ReadWrite => {
                id.permissions.can_write_graph(graph_name)
            }
            weav_auth::identity::GraphPermission::Read => {
                id.permissions.can_read_graph(graph_name)
            }
            weav_auth::identity::GraphPermission::None => true,
        };
        if !has_perm {
            return Err(WeavError::PermissionDenied(format!(
                "insufficient permission on graph '{}'",
                graph_name
            )));
        }
        Ok(())
    }

    /// Extract graph name from a Command variant.
    fn extract_graph_name_from_cmd(&self, cmd: &Command) -> Option<String> {
        match cmd {
            Command::NodeAdd(c) => Some(c.graph.clone()),
            Command::NodeGet(c) => Some(c.graph.clone()),
            Command::NodeUpdate(c) => Some(c.graph.clone()),
            Command::NodeDelete(c) => Some(c.graph.clone()),
            Command::EdgeAdd(c) => Some(c.graph.clone()),
            Command::EdgeInvalidate(c) => Some(c.graph.clone()),
            Command::EdgeDelete(c) => Some(c.graph.clone()),
            Command::EdgeGet(c) => Some(c.graph.clone()),
            Command::BulkInsertNodes(c) => Some(c.graph.clone()),
            Command::BulkInsertEdges(c) => Some(c.graph.clone()),
            Command::Ingest(c) => Some(c.graph.clone()),
            Command::Context(q) => Some(q.graph.clone()),
            Command::GraphCreate(c) => Some(c.name.clone()),
            Command::GraphDrop(name) | Command::GraphInfo(name) => Some(name.clone()),
            Command::Stats(opt) => opt.clone(),
            _ => None,
        }
    }

    // ── Authed handler wrappers ─────────────────────────────────────────
    //
    // These thin wrappers enforce defense-in-depth graph-level permission
    // checks before delegating to the original handler. The original handlers
    // (without `_authed` suffix) remain callable without identity for WAL
    // recovery and internal use.

    fn handle_graph_create_authed(
        &self,
        cmd: weav_query::parser::GraphCreateCmd,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        self.check_permission(identity, &cmd.name, weav_auth::identity::GraphPermission::Admin)?;
        self.handle_graph_create(cmd)
    }

    fn handle_graph_drop_authed(
        &self,
        name: &str,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        self.check_permission(identity, name, weav_auth::identity::GraphPermission::Admin)?;
        self.handle_graph_drop(name)
    }

    fn handle_graph_info_authed(
        &self,
        name: &str,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        self.check_permission(identity, name, weav_auth::identity::GraphPermission::Read)?;
        self.handle_graph_info(name)
    }

    fn handle_node_add_authed(
        &self,
        cmd: weav_query::parser::NodeAddCmd,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        self.check_permission(identity, &cmd.graph, weav_auth::identity::GraphPermission::ReadWrite)?;
        self.handle_node_add(cmd)
    }

    fn handle_node_get_authed(
        &self,
        cmd: weav_query::parser::NodeGetCmd,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        self.check_permission(identity, &cmd.graph, weav_auth::identity::GraphPermission::Read)?;
        self.handle_node_get(cmd)
    }

    fn handle_node_update_authed(
        &self,
        cmd: weav_query::parser::NodeUpdateCmd,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        self.check_permission(identity, &cmd.graph, weav_auth::identity::GraphPermission::ReadWrite)?;
        self.handle_node_update(cmd)
    }

    fn handle_node_delete_authed(
        &self,
        cmd: weav_query::parser::NodeDeleteCmd,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        self.check_permission(identity, &cmd.graph, weav_auth::identity::GraphPermission::ReadWrite)?;
        self.handle_node_delete(cmd)
    }

    fn handle_edge_add_authed(
        &self,
        cmd: weav_query::parser::EdgeAddCmd,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        self.check_permission(identity, &cmd.graph, weav_auth::identity::GraphPermission::ReadWrite)?;
        self.handle_edge_add(cmd)
    }

    fn handle_edge_invalidate_authed(
        &self,
        cmd: weav_query::parser::EdgeInvalidateCmd,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        self.check_permission(identity, &cmd.graph, weav_auth::identity::GraphPermission::ReadWrite)?;
        self.handle_edge_invalidate(cmd)
    }

    fn handle_edge_delete_authed(
        &self,
        cmd: weav_query::parser::EdgeDeleteCmd,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        self.check_permission(identity, &cmd.graph, weav_auth::identity::GraphPermission::ReadWrite)?;
        self.handle_edge_delete(cmd)
    }

    fn handle_edge_get_authed(
        &self,
        cmd: weav_query::parser::EdgeGetCmd,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        self.check_permission(identity, &cmd.graph, weav_auth::identity::GraphPermission::Read)?;
        self.handle_edge_get(cmd)
    }

    fn handle_bulk_insert_nodes_authed(
        &self,
        cmd: weav_query::parser::BulkInsertNodesCmd,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        self.check_permission(identity, &cmd.graph, weav_auth::identity::GraphPermission::ReadWrite)?;
        self.handle_bulk_insert_nodes(cmd)
    }

    fn handle_bulk_insert_edges_authed(
        &self,
        cmd: weav_query::parser::BulkInsertEdgesCmd,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        self.check_permission(identity, &cmd.graph, weav_auth::identity::GraphPermission::ReadWrite)?;
        self.handle_bulk_insert_edges(cmd)
    }

    fn handle_context_authed(
        &self,
        query: weav_query::parser::ContextQuery,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        self.check_permission(identity, &query.graph, weav_auth::identity::GraphPermission::Read)?;
        self.handle_context(query)
    }

    // ── Stats ────────────────────────────────────────────────────────────

    fn handle_stats(
        &self,
        graph_name: Option<String>,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        if let Some(ref name) = graph_name {
            self.check_permission(identity, name, weav_auth::identity::GraphPermission::Read)?;
        }

        if let Some(name) = graph_name {
            let graph_arc = self.get_graph(&name)?;
            let gs = graph_arc.read();
            Ok(CommandResponse::Text(format!(
                "graph={} nodes={} edges={} vectors={}",
                name,
                gs.adjacency.node_count(),
                gs.adjacency.edge_count(),
                gs.vector_index.len(),
            )))
        } else {
            let registry = self.graphs.read();
            Ok(CommandResponse::Text(format!(
                "graphs={} engine=weav-server",
                registry.len(),
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
        let config_json = serde_json::to_string(&graph_config).unwrap_or_default();

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
            dedup_config: None,
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

        // Write-ahead: WAL entry before in-memory mutation
        self.append_wal(WalOperation::GraphCreate {
            name: cmd.name.clone(),
            config_json,
        })?;
        graphs.insert(cmd.name, Arc::new(RwLock::new(graph_state)));
        Ok(CommandResponse::Ok)
    }

    fn handle_graph_drop(&self, name: &str) -> WeavResult<CommandResponse> {
        let mut graphs = self
            .graphs
            .write();
        if !graphs.contains_key(name) {
            return Err(WeavError::GraphNotFound(name.to_string()));
        }
        // Write-ahead: WAL entry before in-memory mutation
        self.append_wal(WalOperation::GraphDrop { name: name.to_string() })?;
        graphs.remove(name);
        Ok(CommandResponse::Ok)
    }

    fn handle_graph_info(&self, name: &str) -> WeavResult<CommandResponse> {
        let graph_arc = self.get_graph(name)?;
        let gs = graph_arc.read();
        Ok(CommandResponse::GraphInfo(GraphInfoResponse {
            name: gs.name.clone(),
            node_count: gs.adjacency.node_count(),
            edge_count: gs.adjacency.edge_count(),
        }))
    }

    fn handle_graph_list(&self) -> WeavResult<CommandResponse> {
        let registry = self.graphs.read();
        let names: Vec<String> = registry.keys().cloned().collect();
        Ok(CommandResponse::StringList(names))
    }

    // ── Node commands ────────────────────────────────────────────────────

    fn handle_node_add(
        &self,
        cmd: weav_query::parser::NodeAddCmd,
    ) -> WeavResult<CommandResponse> {
        let graph_arc = self.get_graph(&cmd.graph)?;
        let mut gs = graph_arc.write();

        // ── Dedup: entity key (always-on, zero false positives) ─────────
        if let Some(ref key) = cmd.entity_key {
            if let Some(existing_id) = weav_graph::dedup::find_duplicate_by_key(
                &gs.properties, "entity_key", key,
            ) {
                // Merge properties into existing node
                weav_graph::dedup::merge_properties(
                    &mut gs.properties,
                    existing_id,
                    &cmd.properties,
                    &ConflictPolicy::LastWriteWins,
                );

                // Update embedding if provided
                if let Some(ref embedding) = cmd.embedding {
                    let _ = gs.vector_index.remove(existing_id);
                    gs.vector_index.insert(existing_id, embedding)?;
                }

                return Ok(CommandResponse::Integer(existing_id));
            }
        }

        // ── Dedup: fuzzy name match (if dedup_config is set) ────────────
        if let Some(ref dedup_cfg) = gs.dedup_config {
            if let Some(ref name_field) = dedup_cfg.name_field {
                // Extract the name value from the incoming properties
                let name_value = cmd.properties.iter()
                    .find(|(k, _)| k == name_field)
                    .and_then(|(_, v)| v.as_str())
                    .map(|s| s.to_string());

                if let Some(ref name_val) = name_value {
                    if let Some((existing_id, _score)) = weav_graph::dedup::find_duplicate_by_name(
                        &gs.properties,
                        name_field,
                        name_val,
                        dedup_cfg.fuzzy_threshold,
                    ) {
                        // If require_same_label: check label matches
                        let label_matches = if dedup_cfg.require_same_label {
                            gs.properties
                                .get_node_property(existing_id, "_label")
                                .and_then(|v| v.as_str())
                                .map(|l| l == cmd.label)
                                .unwrap_or(false)
                        } else {
                            true
                        };

                        if label_matches {
                            weav_graph::dedup::merge_properties(
                                &mut gs.properties,
                                existing_id,
                                &cmd.properties,
                                &ConflictPolicy::LastWriteWins,
                            );

                            if let Some(ref embedding) = cmd.embedding {
                                let _ = gs.vector_index.remove(existing_id);
                                gs.vector_index.insert(existing_id, embedding)?;
                            }

                            return Ok(CommandResponse::Integer(existing_id));
                        }
                    }
                }
            }

            // ── Dedup: vector similarity (if embedding provided) ────────
            if let Some(ref embedding) = cmd.embedding {
                if let Ok(search_results) = gs.vector_index.search(embedding, 5, None) {
                    let similarities: Vec<(NodeId, f32)> = search_results
                        .iter()
                        .map(|&(nid, dist)| (nid, 1.0 / (1.0 + dist)))
                        .collect();

                    if let Some((existing_id, _sim)) = weav_graph::dedup::find_duplicate_by_vector(
                        &similarities,
                        dedup_cfg.vector_threshold,
                    ) {
                        weav_graph::dedup::merge_properties(
                            &mut gs.properties,
                            existing_id,
                            &cmd.properties,
                            &ConflictPolicy::LastWriteWins,
                        );

                        let _ = gs.vector_index.remove(existing_id);
                        gs.vector_index.insert(existing_id, embedding)?;

                        return Ok(CommandResponse::Integer(existing_id));
                    }
                }
            }
        }

        // ── No duplicate found: create new node ─────────────────────────
        let node_id = gs.next_node_id;
        gs.next_node_id += 1;
        let graph_id = gs.graph_id;

        // Write-ahead: WAL entry before in-memory mutation
        let props_json = serde_json::to_string(
            &cmd.properties.iter().map(|(k, v)| (k.as_str(), v)).collect::<std::collections::HashMap<_, _>>()
        ).unwrap_or_default();
        self.append_wal(WalOperation::NodeAdd {
            graph_id,
            node_id,
            label: cmd.label.clone(),
            properties_json: props_json,
            embedding: cmd.embedding.clone(),
            entity_key: cmd.entity_key.clone(),
        })?;

        // Apply in-memory mutation
        gs.adjacency.add_node(node_id);

        gs.properties.set_node_property(
            node_id,
            "_label",
            Value::String(CompactString::from(&cmd.label)),
        );

        let label_id = gs.interner.intern_label(&cmd.label)?;
        gs.properties
            .set_node_property(node_id, "_label_id", Value::Int(label_id as i64));

        if let Some(ref key) = cmd.entity_key {
            gs.properties.set_node_property(
                node_id,
                "entity_key",
                Value::String(CompactString::from(key.as_str())),
            );
        }

        for (k, v) in &cmd.properties {
            gs.properties.set_node_property(node_id, k, v.clone());
        }

        if let Some(ref embedding) = cmd.embedding {
            gs.vector_index.insert(node_id, embedding)?;
        }

        // Store TTL expiry timestamp if provided
        if let Some(ttl_ms) = cmd.ttl_ms {
            let expires_at = Self::now_ms() + ttl_ms;
            gs.properties.set_node_property(
                node_id,
                "_ttl_expires_at",
                Value::Timestamp(expires_at),
            );
        }

        // Update metrics
        crate::metrics::NODES_TOTAL
            .with_label_values(&[&cmd.graph])
            .set(gs.adjacency.node_count() as i64);

        Ok(CommandResponse::Integer(node_id))
    }

    fn handle_node_get(
        &self,
        cmd: weav_query::parser::NodeGetCmd,
    ) -> WeavResult<CommandResponse> {
        let graph_arc = self.get_graph(&cmd.graph)?;
        let gs = graph_arc.read();

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
        let graph_arc = self.get_graph(&cmd.graph)?;
        let mut gs = graph_arc.write();

        if !gs.adjacency.has_node(cmd.node_id) {
            return Err(WeavError::NodeNotFound(cmd.node_id, gs.graph_id));
        }

        // Write-ahead: WAL entry before in-memory mutation
        let graph_id = gs.graph_id;
        self.append_wal(WalOperation::NodeDelete {
            graph_id,
            node_id: cmd.node_id,
        })?;

        // Apply in-memory mutation
        gs.adjacency.remove_node(cmd.node_id)?;
        gs.properties.remove_all_node_properties(cmd.node_id);
        gs.vector_index.remove(cmd.node_id)?;

        // Update metrics
        crate::metrics::NODES_TOTAL
            .with_label_values(&[&cmd.graph])
            .set(gs.adjacency.node_count() as i64);
        crate::metrics::EDGES_TOTAL
            .with_label_values(&[&cmd.graph])
            .set(gs.adjacency.edge_count() as i64);

        Ok(CommandResponse::Ok)
    }

    fn handle_node_update(
        &self,
        cmd: weav_query::parser::NodeUpdateCmd,
    ) -> WeavResult<CommandResponse> {
        let graph_arc = self.get_graph(&cmd.graph)?;
        let mut gs = graph_arc.write();

        if !gs.adjacency.has_node(cmd.node_id) {
            return Err(WeavError::NodeNotFound(cmd.node_id, gs.graph_id));
        }

        // Write-ahead: WAL entry before in-memory mutation
        let graph_id = gs.graph_id;
        let props_json = serde_json::to_string(
            &cmd.properties.iter().map(|(k, v)| (k.as_str(), v)).collect::<std::collections::HashMap<_, _>>()
        ).unwrap_or_default();
        self.append_wal(WalOperation::NodeUpdate {
            graph_id,
            node_id: cmd.node_id,
            properties_json: props_json,
        })?;

        // Apply in-memory mutation
        for (k, v) in &cmd.properties {
            gs.properties.set_node_property(cmd.node_id, k, v.clone());
        }

        if let Some(ref embedding) = cmd.embedding {
            // Write-ahead: persist embedding update to WAL
            self.append_wal(WalOperation::VectorUpdate {
                graph_id,
                node_id: cmd.node_id,
                vector: embedding.clone(),
            })?;
            let _ = gs.vector_index.remove(cmd.node_id);
            gs.vector_index.insert(cmd.node_id, embedding)?;
        }

        Ok(CommandResponse::Ok)
    }

    fn handle_bulk_insert_nodes(
        &self,
        cmd: weav_query::parser::BulkInsertNodesCmd,
    ) -> WeavResult<CommandResponse> {
        let graph_arc = self.get_graph(&cmd.graph)?;
        let mut gs = graph_arc.write();

        let graph_id = gs.graph_id;
        let mut ids = Vec::with_capacity(cmd.nodes.len());
        for node_cmd in &cmd.nodes {
            let node_id = gs.next_node_id;
            gs.next_node_id += 1;

            // Write-ahead: WAL entry before in-memory mutation
            let props_json = serde_json::to_string(
                &node_cmd.properties.iter().map(|(k, v)| (k.as_str(), v)).collect::<std::collections::HashMap<_, _>>()
            ).unwrap_or_default();
            self.append_wal(WalOperation::NodeAdd {
                graph_id,
                node_id,
                label: node_cmd.label.clone(),
                properties_json: props_json,
                embedding: node_cmd.embedding.clone(),
                entity_key: node_cmd.entity_key.clone(),
            })?;

            gs.adjacency.add_node(node_id);

            gs.properties.set_node_property(
                node_id,
                "_label",
                Value::String(CompactString::from(&node_cmd.label)),
            );

            let label_id = gs.interner.intern_label(&node_cmd.label)?;
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
        let graph_arc = self.get_graph(&cmd.graph)?;
        let mut gs = graph_arc.write();

        let now = Self::now_ms();
        let graph_id = gs.graph_id;

        let mut ids = Vec::with_capacity(cmd.edges.len());
        for edge_cmd in &cmd.edges {
            let label_id = gs.interner.intern_label(&edge_cmd.label)?;

            // Pre-allocate edge ID for write-ahead logging
            let edge_id = gs.adjacency.allocate_edge_id();

            // Write-ahead: WAL entry before in-memory mutation
            self.append_wal(WalOperation::EdgeAdd {
                graph_id,
                edge_id,
                source: edge_cmd.source,
                target: edge_cmd.target,
                label: edge_cmd.label.clone(),
                weight: edge_cmd.weight,
            })?;

            // Apply in-memory mutation
            let meta = EdgeMeta {
                source: edge_cmd.source,
                target: edge_cmd.target,
                label: label_id,
                temporal: BiTemporal::new_current(now),
                provenance: None,
                weight: edge_cmd.weight,
                token_cost: 0,
            };
            gs.adjacency.add_edge_with_id(edge_cmd.source, edge_cmd.target, label_id, meta, edge_id)?;
            ids.push(edge_id);
        }

        Ok(CommandResponse::IntegerList(ids))
    }

    // ── Edge commands ────────────────────────────────────────────────────

    fn handle_edge_add(
        &self,
        cmd: weav_query::parser::EdgeAddCmd,
    ) -> WeavResult<CommandResponse> {
        let graph_arc = self.get_graph(&cmd.graph)?;
        let mut gs = graph_arc.write();

        let label_id = gs.interner.intern_label(&cmd.label)?;
        let now = Self::now_ms();

        // Pre-allocate edge ID for write-ahead logging
        let edge_id = gs.adjacency.allocate_edge_id();
        let graph_id = gs.graph_id;

        // Write-ahead: WAL entry before in-memory mutation
        self.append_wal(WalOperation::EdgeAdd {
            graph_id,
            edge_id,
            source: cmd.source,
            target: cmd.target,
            label: cmd.label.clone(),
            weight: cmd.weight,
        })?;

        // Apply in-memory mutation
        let temporal = if let Some(ttl_ms) = cmd.ttl_ms {
            BiTemporal {
                valid_from: now,
                valid_until: now + ttl_ms,
                tx_from: now,
                tx_until: BiTemporal::OPEN,
            }
        } else {
            BiTemporal::new_current(now)
        };
        let meta = EdgeMeta {
            source: cmd.source,
            target: cmd.target,
            label: label_id,
            temporal,
            provenance: None,
            weight: cmd.weight,
            token_cost: 0,
        };
        gs.adjacency.add_edge_with_id(cmd.source, cmd.target, label_id, meta, edge_id)?;

        // Update metrics
        crate::metrics::EDGES_TOTAL
            .with_label_values(&[&cmd.graph])
            .set(gs.adjacency.edge_count() as i64);

        Ok(CommandResponse::Integer(edge_id))
    }

    fn handle_edge_invalidate(
        &self,
        cmd: weav_query::parser::EdgeInvalidateCmd,
    ) -> WeavResult<CommandResponse> {
        let graph_arc = self.get_graph(&cmd.graph)?;
        let mut gs = graph_arc.write();

        let now = Self::now_ms();
        let graph_id = gs.graph_id;

        // Verify edge exists before WAL write
        gs.adjacency.get_edge(cmd.edge_id)
            .ok_or(WeavError::EdgeNotFound(cmd.edge_id))?;

        // Write-ahead: WAL entry before in-memory mutation
        self.append_wal(WalOperation::EdgeInvalidate {
            graph_id,
            edge_id: cmd.edge_id,
            timestamp: now,
        })?;

        gs.adjacency.invalidate_edge(cmd.edge_id, now)?;

        Ok(CommandResponse::Ok)
    }

    fn handle_edge_delete(
        &self,
        cmd: weav_query::parser::EdgeDeleteCmd,
    ) -> WeavResult<CommandResponse> {
        let graph_arc = self.get_graph(&cmd.graph)?;
        let mut gs = graph_arc.write();

        let graph_id = gs.graph_id;

        // Verify edge exists before WAL write
        gs.adjacency.get_edge(cmd.edge_id)
            .ok_or(WeavError::EdgeNotFound(cmd.edge_id))?;

        // Write-ahead: WAL entry before in-memory mutation
        self.append_wal(WalOperation::EdgeDelete {
            graph_id,
            edge_id: cmd.edge_id,
        })?;

        gs.adjacency.remove_edge(cmd.edge_id)?;

        Ok(CommandResponse::Ok)
    }

    fn handle_edge_get(
        &self,
        cmd: weav_query::parser::EdgeGetCmd,
    ) -> WeavResult<CommandResponse> {
        let graph_arc = self.get_graph(&cmd.graph)?;
        let gs = graph_arc.read();

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

        // Clone Arc handles then release registry lock to avoid blocking graph creation/drop.
        let graph_arcs: Vec<Arc<RwLock<GraphState>>> = {
            let registry = self.graphs.read();
            registry.values().cloned().collect()
        };

        let wal_sequence = self.wal.as_ref()
            .map(|w| w.lock().sequence_number())
            .unwrap_or(0);

        let mut graph_snapshots = Vec::new();
        let mut total_nodes: u64 = 0;
        let mut total_edges: u64 = 0;

        for graph_arc in &graph_arcs {
            let gs = graph_arc.read();
            let mut node_snapshots = Vec::new();
            for node_id in gs.adjacency.all_node_ids() {
                let label = gs.properties
                    .get_node_property(node_id, "_label")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                let all_props = gs.properties.get_all_node_properties(node_id);
                let props_json = {
                    let filtered: std::collections::HashMap<&str, &Value> = all_props
                        .into_iter()
                        .filter(|(k, _)| !k.starts_with('_'))
                        .collect();
                    serde_json::to_string(&filtered).unwrap_or_default()
                };

                let entity_key = gs.properties
                    .get_node_property(node_id, "entity_key")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                // Retrieve embedding from vector index for snapshot persistence
                let embedding = gs.vector_index
                    .get_vector(node_id)
                    .map(|v| v.to_vec());

                node_snapshots.push(NodeSnapshot {
                    node_id,
                    label,
                    properties_json: props_json,
                    embedding,
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
                config_json: serde_json::to_string(&gs.config).unwrap_or_default(),
                nodes: node_snapshots,
                edges: edge_snapshots,
            });
        }

        let now = Self::now_ms();

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
        let graph_arc = self.get_graph(&query.graph)?;
        let gs = graph_arc.read();

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

    // ── Ingest handler ────────────────────────────────────────────────────

    async fn handle_ingest(
        &self,
        cmd: weav_query::parser::IngestCmd,
    ) -> WeavResult<CommandResponse> {
        if !self.config.extract.enabled {
            return Err(WeavError::ExtractionNotEnabled);
        }

        // Verify graph exists.
        self.get_graph(&cmd.graph)?;

        // Parse format.
        let format = if let Some(ref fmt_str) = cmd.format {
            weav_extract::types::DocumentFormat::from_str_lossy(fmt_str).ok_or_else(|| {
                WeavError::QueryParseError(format!("unknown document format: {fmt_str}"))
            })?
        } else {
            weav_extract::types::DocumentFormat::PlainText
        };

        // Build input document.
        let document_id = cmd
            .document_id
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let input_doc = weav_extract::types::InputDocument {
            document_id: document_id.clone(),
            format,
            content: weav_extract::types::DocumentContent::Text(cmd.content),
        };

        let options = weav_extract::types::IngestOptions {
            document_id: Some(document_id.clone()),
            format: None,
            skip_extraction: cmd.skip_extraction,
            skip_dedup: cmd.skip_dedup,
            chunk_size: cmd.chunk_size,
            chunk_overlap: None,
            entity_types: cmd.entity_types,
            custom_extraction_prompt: None,
        };

        // Run extraction pipeline.
        let result =
            weav_extract::pipeline::run_pipeline(input_doc, &options, &self.config.extract)
                .await?;

        // Write-ahead: WAL entry before in-memory mutation
        self.append_wal(WalOperation::Ingest {
            graph_name: cmd.graph.clone(),
            document_id: result.document_id.clone(),
            chunks_count: result.stats.total_chunks,
            entities_count: result.stats.total_entities,
            relationships_count: result.stats.total_relationships,
        })?;

        // Apply results to graph.
        let response = self.apply_extraction_result(&cmd.graph, &result)?;

        Ok(response)
    }

    /// Apply extraction results to a graph: insert chunk nodes, entity nodes,
    /// and relationship edges.
    fn apply_extraction_result(
        &self,
        graph_name: &str,
        result: &weav_extract::types::ExtractionResult,
    ) -> WeavResult<CommandResponse> {
        let graph_arc = self.get_graph(graph_name)?;
        let mut gs = graph_arc.write();

        let now = Self::now_ms();
        let mut entities_created = 0;
        let mut entities_merged = result.stats.entities_merged;
        let mut relationships_created = 0;

        // Insert chunk nodes.
        for cwe in &result.chunks {
            let node_id = gs.next_node_id;
            gs.next_node_id += 1;

            gs.adjacency.add_node(node_id);
            let label_id = gs.interner.intern_label("chunk")?;
            gs.properties.set_node_property(
                node_id,
                "_label",
                Value::String(CompactString::from("chunk")),
            );
            gs.properties.set_node_property(
                node_id,
                "_label_id",
                Value::Int(label_id as i64),
            );
            gs.properties.set_node_property(
                node_id,
                "text",
                Value::String(CompactString::from(&cwe.chunk.text)),
            );
            gs.properties.set_node_property(
                node_id,
                "document_id",
                Value::String(CompactString::from(&cwe.chunk.document_id)),
            );
            gs.properties.set_node_property(
                node_id,
                "chunk_index",
                Value::Int(cwe.chunk.chunk_index as i64),
            );
            gs.properties.set_node_property(
                node_id,
                "token_count",
                Value::Int(cwe.chunk.token_count as i64),
            );

            // Set temporal metadata.
            gs.properties.set_node_property(node_id, "_tx_from", Value::Int(now as i64));

            // Insert embedding.
            if !cwe.embedding.is_empty() {
                let _ = gs.vector_index.insert(node_id, &cwe.embedding);
            }
        }

        // Build entity name -> node_id map for relationship linking.
        let mut entity_node_map: HashMap<String, NodeId> = HashMap::new();

        // Insert entity nodes.
        for ewe in &result.entities {
            let entity = &ewe.entity;
            let entity_key = entity.name.to_lowercase();

            // Check for existing entity by key.
            if let Some(existing_id) =
                weav_graph::dedup::find_duplicate_by_key(&gs.properties, "entity_key", &entity_key)
            {
                // Merge properties.
                weav_graph::dedup::merge_properties(
                    &mut gs.properties,
                    existing_id,
                    &entity.properties,
                    &ConflictPolicy::LastWriteWins,
                );
                entity_node_map.insert(entity_key, existing_id);
                entities_merged += 1;
                continue;
            }

            let node_id = gs.next_node_id;
            gs.next_node_id += 1;

            gs.adjacency.add_node(node_id);
            let label_id = gs.interner.intern_label(&entity.entity_type)?;
            gs.properties.set_node_property(
                node_id,
                "_label",
                Value::String(CompactString::from(&entity.entity_type)),
            );
            gs.properties.set_node_property(
                node_id,
                "_label_id",
                Value::Int(label_id as i64),
            );
            gs.properties.set_node_property(
                node_id,
                "entity_key",
                Value::String(CompactString::from(&entity_key)),
            );
            gs.properties.set_node_property(
                node_id,
                "name",
                Value::String(CompactString::from(&entity.name)),
            );
            if !entity.description.is_empty() {
                gs.properties.set_node_property(
                    node_id,
                    "description",
                    Value::String(CompactString::from(&entity.description)),
                );
            }
            gs.properties.set_node_property(
                node_id,
                "confidence",
                Value::Float(entity.confidence as f64),
            );
            gs.properties.set_node_property(
                node_id,
                "document_id",
                Value::String(CompactString::from(&result.document_id)),
            );

            // Set user properties.
            for (k, v) in &entity.properties {
                gs.properties.set_node_property(node_id, k, v.clone());
            }

            // Set temporal metadata.
            gs.properties.set_node_property(node_id, "_tx_from", Value::Int(now as i64));

            // Insert embedding.
            if let Some(ref embedding) = ewe.embedding {
                let _ = gs.vector_index.insert(node_id, embedding);
            }

            entity_node_map.insert(entity_key, node_id);
            entities_created += 1;
        }

        // Insert relationship edges.
        for rel in &result.relationships {
            let source_key = rel.source_entity.to_lowercase();
            let target_key = rel.target_entity.to_lowercase();

            let source_id = entity_node_map.get(&source_key).copied();
            let target_id = entity_node_map.get(&target_key).copied();

            if let (Some(src), Some(tgt)) = (source_id, target_id) {
                let label_id = gs.interner.intern_label(&rel.relationship_type)?;
                let edge_meta = EdgeMeta {
                    source: src,
                    target: tgt,
                    label: label_id,
                    temporal: BiTemporal {
                        valid_from: now,
                        valid_until: u64::MAX,
                        tx_from: now,
                        tx_until: u64::MAX,
                    },
                    provenance: None,
                    weight: rel.weight,
                    token_cost: 0,
                };
                let _ = gs.adjacency.add_edge(src, tgt, label_id, edge_meta);
                relationships_created += 1;
            }
        }

        Ok(CommandResponse::IngestResult(IngestResultResponse {
            document_id: result.document_id.clone(),
            chunks_created: result.stats.total_chunks,
            entities_created,
            entities_merged,
            relationships_created,
            pipeline_duration_ms: result.stats.pipeline_duration_ms,
        }))
    }

    // ── AUTH / ACL handlers ──────────────────────────────────────────────

    fn handle_auth(
        &self,
        username: Option<String>,
        password: String,
    ) -> WeavResult<CommandResponse> {
        let store = self.acl_store()?;

        let identity = match username {
            Some(ref u) => store.authenticate(u, &password)?,
            None => store.authenticate_default(&password)?,
        };

        // Return OK with the username. The protocol layer uses the identity
        // from authenticate() directly; this response is for the client.
        Ok(CommandResponse::Text(format!("OK (user: {})", identity.username)))
    }

    fn handle_acl_set_user(
        &self,
        cmd: weav_query::parser::AclSetUserCmd,
    ) -> WeavResult<CommandResponse> {
        let store = self.acl_store()?;

        // Get or create user.
        let mut user = store.get_user(&cmd.username).unwrap_or_else(|| {
            weav_auth::acl::AclUser {
                username: cmd.username.clone(),
                password_hash: None,
                enabled: true,
                categories: weav_auth::identity::CommandCategorySet::all(),
                graph_acl: Vec::new(),
                api_key_hashes: Vec::new(),
            }
        });

        // Apply modifications.
        if let Some(pw) = &cmd.password {
            user.password_hash = Some(
                weav_auth::password::hash_password(pw)
                    .map_err(|e| WeavError::Internal(e))?,
            );
        }
        if let Some(enabled) = cmd.enabled {
            user.enabled = enabled;
        }
        if !cmd.categories.is_empty() {
            user.categories = weav_auth::identity::CommandCategorySet::from_acl_strings(&cmd.categories);
        }
        if !cmd.graph_patterns.is_empty() {
            user.graph_acl = cmd
                .graph_patterns
                .iter()
                .map(|(pat, perm)| weav_auth::identity::GraphAcl {
                    pattern: pat.clone(),
                    permission: weav_auth::identity::GraphPermission::from_str(perm),
                })
                .collect();
        }

        store.set_user(user);
        Ok(CommandResponse::Ok)
    }

    fn handle_acl_del_user(&self, username: &str) -> WeavResult<CommandResponse> {
        let store = self.acl_store()?;
        if store.delete_user(username) {
            Ok(CommandResponse::Ok)
        } else {
            Err(WeavError::AuthenticationFailed(format!(
                "user '{}' not found",
                username
            )))
        }
    }

    fn handle_acl_list(&self) -> WeavResult<CommandResponse> {
        let store = self.acl_store()?;
        let users = store.list_users();
        Ok(CommandResponse::StringList(users))
    }

    fn handle_acl_get_user(&self, username: &str) -> WeavResult<CommandResponse> {
        let store = self.acl_store()?;
        match store.get_user(username) {
            Some(user) => {
                let info = format!(
                    "user={} enabled={} has_password={}",
                    user.username, user.enabled, user.password_hash.is_some()
                );
                Ok(CommandResponse::Text(info))
            }
            None => Err(WeavError::AuthenticationFailed(format!(
                "user '{}' not found",
                username
            ))),
        }
    }

    fn handle_acl_whoami(
        &self,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        match identity {
            Some(id) => Ok(CommandResponse::Text(id.username.clone())),
            None => Ok(CommandResponse::Text("(unauthenticated)".into())),
        }
    }

    fn handle_acl_save(&self) -> WeavResult<CommandResponse> {
        let store = self.acl_store()?;
        let content = store.serialize_acl();
        if let Some(ref path) = self.config.auth.acl_file {
            std::fs::write(path, &content).map_err(|e| {
                WeavError::PersistenceError(format!("failed to write ACL file: {e}"))
            })?;
            Ok(CommandResponse::Ok)
        } else {
            Ok(CommandResponse::Text(content))
        }
    }

    fn handle_acl_load(&self) -> WeavResult<CommandResponse> {
        let store = self.acl_store()?;
        if let Some(ref path) = self.config.auth.acl_file {
            let content = std::fs::read_to_string(path).map_err(|e| {
                WeavError::PersistenceError(format!("failed to read ACL file: {e}"))
            })?;
            store.load_acl(&content);
            Ok(CommandResponse::Ok)
        } else {
            Err(WeavError::InvalidConfig("no ACL file configured".into()))
        }
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
        engine.execute_command(cmd, None).unwrap();
    }

    #[test]
    fn test_ping() {
        let engine = make_engine();
        let resp = engine
            .execute_command(Command::Ping, None)
            .unwrap();
        assert!(matches!(resp, CommandResponse::Pong));
    }

    #[test]
    fn test_info() {
        let engine = make_engine();
        let resp = engine
            .execute_command(Command::Info, None)
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

        let resp = engine.execute_command(Command::GraphList, None).unwrap();
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
        let result = engine.execute_command(cmd, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_graph_drop() {
        let engine = make_engine();
        create_test_graph(&engine, "todrop");
        let cmd = parser::parse_command("GRAPH DROP \"todrop\"").unwrap();
        engine.execute_command(cmd, None).unwrap();

        let resp = engine.execute_command(Command::GraphList, None).unwrap();
        match resp {
            CommandResponse::StringList(names) => assert!(names.is_empty()),
            _ => panic!("expected StringList"),
        }
    }

    #[test]
    fn test_graph_drop_not_found() {
        let engine = make_engine();
        let cmd = parser::parse_command("GRAPH DROP \"nonexistent\"").unwrap();
        assert!(engine.execute_command(cmd, None).is_err());
    }

    #[test]
    fn test_graph_info() {
        let engine = make_engine();
        create_test_graph(&engine, "info_test");

        let cmd = parser::parse_command("GRAPH INFO \"info_test\"").unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
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
        assert!(engine.execute_command(cmd, None).is_err());
    }

    #[test]
    fn test_node_add_and_get_by_id() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "person" PROPERTIES {"name": "Alice", "age": 30} KEY "alice-001""#,
        )
        .unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        let node_id = match resp {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer response"),
        };
        assert!(node_id >= 1);

        // Get by ID.
        let cmd = parser::parse_command(&format!("NODE GET \"g\" {node_id}")).unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
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
        engine.execute_command(cmd, None).unwrap();

        let cmd =
            parser::parse_command("NODE GET \"g\" WHERE entity_key = \"bob-key\"").unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
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
        assert!(engine.execute_command(cmd, None).is_err());
    }

    #[test]
    fn test_node_delete() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "person" PROPERTIES {"name": "Charlie"}"#,
        )
        .unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        let node_id = match resp {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(&format!("NODE DELETE \"g\" {node_id}")).unwrap();
        engine.execute_command(cmd, None).unwrap();

        // Should no longer be findable.
        let cmd = parser::parse_command(&format!("NODE GET \"g\" {node_id}")).unwrap();
        assert!(engine.execute_command(cmd, None).is_err());
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
        let r1 = engine.execute_command(cmd, None).unwrap();
        let n1 = match r1 {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "person" PROPERTIES {"name": "B"}"#,
        )
        .unwrap();
        let r2 = engine.execute_command(cmd, None).unwrap();
        let n2 = match r2 {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add edge.
        let cmd = parser::parse_command(&format!(
            r#"EDGE ADD TO "g" FROM {n1} TO {n2} LABEL "knows" WEIGHT 0.9"#,
        ))
        .unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::Integer(eid) => assert!(eid >= 1),
            _ => panic!("expected Integer"),
        }

        // Check graph info.
        let cmd = parser::parse_command("GRAPH INFO \"g\"").unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
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
        assert!(engine.execute_command(cmd, None).is_err());
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
        let r1 = engine.execute_command(cmd, None).unwrap();
        let n1 = match r1 {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "b" PROPERTIES {"name": "Y"}"#,
        )
        .unwrap();
        let r2 = engine.execute_command(cmd, None).unwrap();
        let n2 = match r2 {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(&format!(
            r#"EDGE ADD TO "g" FROM {n1} TO {n2} LABEL "rel""#,
        ))
        .unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        let edge_id = match resp {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Invalidate the edge.
        let cmd =
            parser::parse_command(&format!("EDGE INVALIDATE \"g\" {edge_id}")).unwrap();
        engine.execute_command(cmd, None).unwrap();
    }

    #[test]
    fn test_edge_invalidate_not_found() {
        let engine = make_engine();
        create_test_graph(&engine, "g");
        let cmd = parser::parse_command("EDGE INVALIDATE \"g\" 999").unwrap();
        assert!(engine.execute_command(cmd, None).is_err());
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
        let r1 = engine.execute_command(cmd, None).unwrap();
        let n1 = match r1 {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(
            r#"NODE ADD TO "kg" LABEL "topic" PROPERTIES {"name": "Rust", "description": "A language"} KEY "rust""#,
        )
        .unwrap();
        let r2 = engine.execute_command(cmd, None).unwrap();
        let n2 = match r2 {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add edge.
        let cmd = parser::parse_command(&format!(
            r#"EDGE ADD TO "kg" FROM {n1} TO {n2} LABEL "uses""#,
        ))
        .unwrap();
        engine.execute_command(cmd, None).unwrap();

        // Run context query seeded by node key.
        let cmd = parser::parse_command(
            r#"CONTEXT "what does alice use" FROM "kg" SEEDS NODES ["alice"] DEPTH 2"#,
        )
        .unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
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
        let resp = engine.execute_command(cmd, None).unwrap();
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
        assert!(engine.execute_command(cmd, None).is_err());
    }

    #[test]
    fn test_stats_no_graph() {
        let engine = make_engine();
        let cmd = parser::parse_command("STATS").unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
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
        let resp = engine.execute_command(cmd, None).unwrap();
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
        assert!(engine.execute_command(cmd, None).is_err());
    }

    #[test]
    fn test_snapshot_placeholder() {
        let engine = make_engine();
        let cmd = parser::parse_command("SNAPSHOT").unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
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
        let resp = engine.execute_command(cmd, None).unwrap();
        let node_id = match resp {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Update the node.
        let cmd = parser::parse_command(&format!(
            r#"NODE UPDATE "g" {node_id} PROPERTIES {{"name": "Alice Updated", "age": 30}}"#,
        ))
        .unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        assert!(matches!(resp, CommandResponse::Ok));

        // Verify the update.
        let cmd = parser::parse_command(&format!("NODE GET \"g\" {node_id}")).unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
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
        assert!(engine.execute_command(cmd, None).is_err());
    }

    #[test]
    fn test_bulk_insert_nodes() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        let cmd = parser::parse_command(
            r#"BULK NODES TO "g" DATA [{"label": "person", "properties": {"name": "A"}}, {"label": "person", "properties": {"name": "B"}}]"#,
        )
        .unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
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
        let resp = engine.execute_command(cmd, None).unwrap();
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
        let resp = engine.execute_command(cmd, None).unwrap();
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
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::IntegerList(ids) => {
                assert_eq!(ids.len(), 1);
            }
            _ => panic!("expected IntegerList"),
        }

        // Verify graph info.
        let cmd = parser::parse_command("GRAPH INFO \"g\"").unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
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
        let resp = engine.execute_command(cmd, None).unwrap();
        let node_id = match resp {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Update the node properties.
        let cmd = parser::parse_command(&format!(
            r#"NODE UPDATE "g" {node_id} PROPERTIES {{"name": "Bob", "city": "NYC"}}"#,
        ))
        .unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        assert!(matches!(resp, CommandResponse::Ok));

        // Verify updated properties via NODE GET.
        let cmd = parser::parse_command(&format!("NODE GET \"g\" {node_id}")).unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
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
        let n1 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "b" PROPERTIES {"name": "Y"}"#,
        )
        .unwrap();
        let n2 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add edge.
        let cmd = parser::parse_command(&format!(
            r#"EDGE ADD TO "g" FROM {n1} TO {n2} LABEL "knows" WEIGHT 0.75"#,
        ))
        .unwrap();
        let edge_id = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Get the edge by ID.
        let cmd = parser::parse_command(&format!("EDGE GET \"g\" {edge_id}")).unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
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
        let n1 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "b" PROPERTIES {"name": "Y"}"#,
        )
        .unwrap();
        let n2 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add edge.
        let cmd = parser::parse_command(&format!(
            r#"EDGE ADD TO "g" FROM {n1} TO {n2} LABEL "link""#,
        ))
        .unwrap();
        let edge_id = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Delete the edge.
        let cmd = parser::parse_command(&format!("EDGE DELETE \"g\" {edge_id}")).unwrap();
        engine.execute_command(cmd, None).unwrap();

        // Verify EDGE GET returns error.
        let cmd = parser::parse_command(&format!("EDGE GET \"g\" {edge_id}")).unwrap();
        assert!(engine.execute_command(cmd, None).is_err());
    }

    #[test]
    fn test_node_get_nonexistent_graph() {
        let engine = make_engine();
        let cmd = parser::parse_command("NODE GET \"no_such_graph\" 1").unwrap();
        assert!(engine.execute_command(cmd, None).is_err());
    }

    #[test]
    fn test_node_add_nonexistent_graph() {
        let engine = make_engine();
        let cmd = parser::parse_command(
            r#"NODE ADD TO "no_such_graph" LABEL "x" PROPERTIES {"a": 1}"#,
        )
        .unwrap();
        assert!(engine.execute_command(cmd, None).is_err());
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
        engine.execute_command(cmd, None).unwrap();

        let cmd = parser::parse_command(
            r#"NODE ADD TO "snap_g" LABEL "person" PROPERTIES {"name": "Bob"}"#,
        )
        .unwrap();
        engine.execute_command(cmd, None).unwrap();

        // Execute SNAPSHOT command.
        let cmd = parser::parse_command("SNAPSHOT").unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        assert!(matches!(resp, CommandResponse::Ok));

        // Clean up temp dir.
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_sync_wal_with_persistence() {
        let tmp_dir = std::env::temp_dir().join(format!("weav_wal_sync_test_{}", std::process::id()));
        std::fs::create_dir_all(&tmp_dir).unwrap();

        let mut config = WeavConfig::default();
        config.persistence.enabled = true;
        config.persistence.data_dir = tmp_dir.clone();
        config.persistence.wal_sync_mode = weav_core::config::WalSyncMode::EverySecond;

        let engine = Engine::new(config);
        create_test_graph(&engine, "wal_g");

        // Add a node to produce WAL entries.
        let cmd = parser::parse_command(
            r#"NODE ADD TO "wal_g" LABEL "person" PROPERTIES {"name": "Alice"}"#,
        )
        .unwrap();
        engine.execute_command(cmd, None).unwrap();

        // sync_wal should not panic and should succeed.
        engine.sync_wal().unwrap();

        // Verify wal_sync_mode returns the expected mode.
        assert!(matches!(
            engine.wal_sync_mode(),
            weav_core::config::WalSyncMode::EverySecond
        ));

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_sync_wal_without_persistence() {
        // sync_wal on an engine without persistence should be a no-op.
        let engine = make_engine();
        engine.sync_wal().unwrap(); // Should succeed as no-op
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
        let n1 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "b" PROPERTIES {"name": "N2"}"#,
        )
        .unwrap();
        let n2 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add edge.
        let cmd = parser::parse_command(&format!(
            r#"EDGE ADD TO "g" FROM {n1} TO {n2} LABEL "link""#,
        ))
        .unwrap();
        engine.execute_command(cmd, None).unwrap();

        // Delete node 1.
        let cmd = parser::parse_command(&format!("NODE DELETE \"g\" {n1}")).unwrap();
        engine.execute_command(cmd, None).unwrap();

        // Edge count should be 0.
        let cmd = parser::parse_command("GRAPH INFO \"g\"").unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::GraphInfo(info) => {
                assert_eq!(info.node_count, 1);
                assert_eq!(info.edge_count, 0);
            }
            _ => panic!("expected GraphInfo"),
        }
    }

    #[test]
    fn test_entity_key_dedup_merges() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        // Add first node with entity_key "alice"
        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "person" PROPERTIES {"name": "Alice", "age": 25} KEY "alice""#,
        )
        .unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        let first_id = match resp {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add second node with same entity_key "alice" but different props
        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "person" PROPERTIES {"name": "Alice Updated", "city": "NYC"} KEY "alice""#,
        )
        .unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        let second_id = match resp {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Should return the SAME node_id (dedup detected)
        assert_eq!(first_id, second_id, "Dedup should return existing node_id");

        // Verify properties were merged
        let cmd = parser::parse_command(&format!("NODE GET \"g\" {first_id}")).unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::NodeInfo(info) => {
                // "name" should be updated (LastWriteWins)
                let name = info.properties.iter().find(|(k, _)| k == "name");
                assert_eq!(name.unwrap().1.as_str(), Some("Alice Updated"));
                // "city" should be added
                assert!(info.properties.iter().any(|(k, _)| k == "city"));
                // "age" should remain from original
                assert!(info.properties.iter().any(|(k, _)| k == "age"));
            }
            _ => panic!("expected NodeInfo"),
        }

        // Graph should still have only 1 node
        let cmd = parser::parse_command("GRAPH INFO \"g\"").unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::GraphInfo(info) => {
                assert_eq!(info.node_count, 1, "Dedup should not create a second node");
            }
            _ => panic!("expected GraphInfo"),
        }
    }

    #[test]
    fn test_no_dedup_without_entity_key() {
        let engine = make_engine();
        create_test_graph(&engine, "g");

        // Add two nodes without entity_key
        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "person" PROPERTIES {"name": "Alice"}"#,
        )
        .unwrap();
        let id1 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "person" PROPERTIES {"name": "Alice"}"#,
        )
        .unwrap();
        let id2 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        assert_ne!(id1, id2, "Without entity_key, nodes should be distinct");
    }

    #[test]
    fn test_dedup_updates_embedding() {
        let mut config = WeavConfig::default();
        config.engine.default_vector_dimensions = 4;
        let engine = Engine::new(config);
        create_test_graph(&engine, "g");

        // Add node with entity_key and embedding
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "g".to_string(),
            label: "person".to_string(),
            properties: vec![("name".to_string(), Value::String(CompactString::from("Alice")))],
            embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            entity_key: Some("alice".to_string()),
            ttl_ms: None,
        });
        let first_id = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add again with same entity_key but new embedding
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "g".to_string(),
            label: "person".to_string(),
            properties: vec![("name".to_string(), Value::String(CompactString::from("Alice")))],
            embedding: Some(vec![0.0, 1.0, 0.0, 0.0]),
            entity_key: Some("alice".to_string()),
            ttl_ms: None,
        });
        let second_id = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        assert_eq!(first_id, second_id, "Dedup should return existing node_id");

        // Node count should still be 1
        let cmd = parser::parse_command("GRAPH INFO \"g\"").unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::GraphInfo(info) => {
                assert_eq!(info.node_count, 1);
            }
            _ => panic!("expected GraphInfo"),
        }
    }

    // ── Round 10: Edge-case tests ────────────────────────────────────────

    #[test]
    fn test_engine_empty_graph_name() {
        let engine = make_engine();
        // Attempt to create a graph with an empty name.
        let cmd = parser::parse_command("GRAPH CREATE \"\"").unwrap();
        let result = engine.execute_command(cmd, None);
        // The engine may accept or reject an empty name; either outcome is valid.
        // We just verify it does not panic.
        match result {
            Ok(CommandResponse::Ok) => {
                // If accepted, we should be able to query it.
                let cmd = parser::parse_command("GRAPH INFO \"\"").unwrap();
                let resp = engine.execute_command(cmd, None).unwrap();
                assert!(matches!(resp, CommandResponse::GraphInfo(_)));
            }
            Err(_) => {
                // Rejecting an empty name is also valid behaviour.
            }
            other => panic!("unexpected response: {:?}", other),
        }
    }

    #[test]
    fn test_engine_duplicate_graph_create() {
        let engine = make_engine();
        create_test_graph(&engine, "dup_test");

        // Creating the same graph again should return Conflict.
        let cmd = parser::parse_command("GRAPH CREATE \"dup_test\"").unwrap();
        let result = engine.execute_command(cmd, None);
        assert!(result.is_err(), "duplicate GRAPH CREATE should return an error");
        let err = result.unwrap_err();
        assert!(
            matches!(err, WeavError::Conflict(_)),
            "expected Conflict error, got: {:?}",
            err
        );
    }

    #[test]
    fn test_engine_node_add_to_nonexistent_graph() {
        let engine = make_engine();
        let cmd = parser::parse_command(
            r#"NODE ADD TO "nonexistent" LABEL "person" PROPERTIES {"name": "X"}"#,
        )
        .unwrap();
        let result = engine.execute_command(cmd, None);
        assert!(result.is_err(), "NODE ADD to nonexistent graph should fail");
        let err = result.unwrap_err();
        assert!(
            matches!(err, WeavError::GraphNotFound(_)),
            "expected GraphNotFound, got: {:?}",
            err
        );
    }

    #[test]
    fn test_engine_edge_between_nonexistent_nodes() {
        let engine = make_engine();
        create_test_graph(&engine, "edge_test");

        // Try to add an edge referencing node IDs that do not exist.
        let cmd = parser::parse_command(
            r#"EDGE ADD TO "edge_test" FROM 9990 TO 9991 LABEL "link""#,
        )
        .unwrap();
        let result = engine.execute_command(cmd, None);
        assert!(
            result.is_err(),
            "EDGE ADD between nonexistent nodes should fail"
        );
        let err = result.unwrap_err();
        assert!(
            matches!(err, WeavError::NodeNotFound(..)),
            "expected NodeNotFound, got: {:?}",
            err
        );
    }

    #[test]
    fn test_engine_delete_nonexistent_node() {
        let engine = make_engine();
        create_test_graph(&engine, "del_node_test");

        let cmd = parser::parse_command("NODE DELETE \"del_node_test\" 999").unwrap();
        let result = engine.execute_command(cmd, None);
        assert!(
            result.is_err(),
            "NODE DELETE on nonexistent node should fail"
        );
    }

    #[test]
    fn test_engine_graph_info_nonexistent() {
        let engine = make_engine();
        let cmd = parser::parse_command("GRAPH INFO \"nonexistent\"").unwrap();
        let result = engine.execute_command(cmd, None);
        assert!(result.is_err(), "GRAPH INFO for nonexistent graph should fail");
        let err = result.unwrap_err();
        assert!(
            matches!(err, WeavError::GraphNotFound(_)),
            "expected GraphNotFound, got: {:?}",
            err
        );
    }

    // ── Authorization enforcement tests ─────────────────────────────────

    fn make_auth_engine() -> Engine {
        use weav_core::config::{AuthConfig, UserConfig, GraphPatternConfig};
        let mut config = WeavConfig::default();
        config.auth = AuthConfig {
            enabled: true,
            require_auth: true,
            acl_file: None,
            default_password: None,
            users: vec![
                // Admin user: full access to everything
                UserConfig {
                    username: "admin".into(),
                    password: Some("admin_pass".into()),
                    categories: vec!["+@all".into()],
                    graph_patterns: Vec::new(),
                    api_keys: Vec::new(),
                    enabled: true,
                },
                // Read-only user: read category, read permission on "shared"
                UserConfig {
                    username: "reader".into(),
                    password: Some("reader_pass".into()),
                    categories: vec!["+@read".into(), "+@connection".into()],
                    graph_patterns: vec![GraphPatternConfig {
                        pattern: "shared".into(),
                        permission: "read".into(),
                    }],
                    api_keys: Vec::new(),
                    enabled: true,
                },
                // Writer user: read+write categories, readwrite on "app:*"
                UserConfig {
                    username: "writer".into(),
                    password: Some("writer_pass".into()),
                    categories: vec!["+@read".into(), "+@write".into(), "+@connection".into()],
                    graph_patterns: vec![GraphPatternConfig {
                        pattern: "app:*".into(),
                        permission: "readwrite".into(),
                    }],
                    api_keys: Vec::new(),
                    enabled: true,
                },
            ],
        };
        Engine::new(config)
    }

    fn auth_identity(engine: &Engine, username: &str, password: &str)
        -> weav_auth::identity::SessionIdentity
    {
        engine
            .acl_store()
            .unwrap()
            .authenticate(username, password)
            .unwrap()
    }

    #[test]
    fn test_auth_required_no_identity() {
        let engine = make_auth_engine();
        let cmd = parser::parse_command("GRAPH LIST").unwrap();
        let result = engine.execute_command(cmd, None);
        assert!(result.is_err());
        assert!(
            matches!(result.unwrap_err(), WeavError::AuthenticationRequired),
            "should require authentication"
        );
    }

    #[test]
    fn test_auth_ping_exempt() {
        let engine = make_auth_engine();
        // PING should always work without identity
        let result = engine.execute_command(Command::Ping, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_auth_admin_can_create_graph() {
        let engine = make_auth_engine();
        let id = auth_identity(&engine, "admin", "admin_pass");
        let cmd = parser::parse_command("GRAPH CREATE \"test\"").unwrap();
        let result = engine.execute_command(cmd, Some(&id));
        assert!(result.is_ok(), "admin should be able to create graph");
    }

    #[test]
    fn test_auth_reader_cannot_create_graph() {
        let engine = make_auth_engine();
        let id = auth_identity(&engine, "reader", "reader_pass");
        let cmd = parser::parse_command("GRAPH CREATE \"test\"").unwrap();
        let result = engine.execute_command(cmd, Some(&id));
        assert!(result.is_err());
        match result.unwrap_err() {
            WeavError::PermissionDenied(msg) => {
                assert!(msg.contains("reader"), "error should mention username");
            }
            other => panic!("expected PermissionDenied, got: {:?}", other),
        }
    }

    #[test]
    fn test_auth_writer_cannot_create_graph() {
        let engine = make_auth_engine();
        let id = auth_identity(&engine, "writer", "writer_pass");
        // Writer has +@write but not +@admin, so graph create should fail
        let cmd = parser::parse_command("GRAPH CREATE \"app:test\"").unwrap();
        let result = engine.execute_command(cmd, Some(&id));
        assert!(result.is_err());
        match result.unwrap_err() {
            WeavError::PermissionDenied(msg) => {
                assert!(msg.contains("writer"));
            }
            other => panic!("expected PermissionDenied, got: {:?}", other),
        }
    }

    #[test]
    fn test_auth_reader_can_read_shared_graph() {
        let engine = make_auth_engine();
        // Admin creates the graph
        let admin_id = auth_identity(&engine, "admin", "admin_pass");
        let cmd = parser::parse_command("GRAPH CREATE \"shared\"").unwrap();
        engine.execute_command(cmd, Some(&admin_id)).unwrap();

        // Reader can read it
        let reader_id = auth_identity(&engine, "reader", "reader_pass");
        let cmd = parser::parse_command("GRAPH INFO \"shared\"").unwrap();
        let result = engine.execute_command(cmd, Some(&reader_id));
        assert!(result.is_ok(), "reader should be able to read shared graph");
    }

    #[test]
    fn test_auth_reader_cannot_read_other_graph() {
        let engine = make_auth_engine();
        // Admin creates a graph the reader has no access to
        let admin_id = auth_identity(&engine, "admin", "admin_pass");
        let cmd = parser::parse_command("GRAPH CREATE \"secret\"").unwrap();
        engine.execute_command(cmd, Some(&admin_id)).unwrap();

        // Reader cannot read it (pattern "shared" doesn't match "secret")
        let reader_id = auth_identity(&engine, "reader", "reader_pass");
        let cmd = parser::parse_command("GRAPH INFO \"secret\"").unwrap();
        let result = engine.execute_command(cmd, Some(&reader_id));
        assert!(result.is_err());
        match result.unwrap_err() {
            WeavError::PermissionDenied(msg) => {
                assert!(msg.contains("read access"));
            }
            other => panic!("expected PermissionDenied, got: {:?}", other),
        }
    }

    #[test]
    fn test_auth_reader_cannot_write_shared_graph() {
        let engine = make_auth_engine();
        let admin_id = auth_identity(&engine, "admin", "admin_pass");
        let cmd = parser::parse_command("GRAPH CREATE \"shared\"").unwrap();
        engine.execute_command(cmd, Some(&admin_id)).unwrap();

        // Reader has +@read but not +@write category, so NodeAdd should fail
        let reader_id = auth_identity(&engine, "reader", "reader_pass");
        let cmd = parser::parse_command(
            r#"NODE ADD TO "shared" LABEL "doc""#,
        )
        .unwrap();
        let result = engine.execute_command(cmd, Some(&reader_id));
        assert!(result.is_err());
        match result.unwrap_err() {
            WeavError::PermissionDenied(msg) => {
                assert!(msg.contains("reader"));
            }
            other => panic!("expected PermissionDenied, got: {:?}", other),
        }
    }

    #[test]
    fn test_auth_writer_can_write_matching_graph() {
        let engine = make_auth_engine();
        let admin_id = auth_identity(&engine, "admin", "admin_pass");
        let cmd = parser::parse_command("GRAPH CREATE \"app:users\"").unwrap();
        engine.execute_command(cmd, Some(&admin_id)).unwrap();

        // Writer with "app:*" pattern and ReadWrite permission can write
        let writer_id = auth_identity(&engine, "writer", "writer_pass");
        let cmd = parser::parse_command(
            r#"NODE ADD TO "app:users" LABEL "user""#,
        )
        .unwrap();
        let result = engine.execute_command(cmd, Some(&writer_id));
        assert!(result.is_ok(), "writer should be able to add nodes to app:users");
    }

    #[test]
    fn test_auth_writer_cannot_write_non_matching_graph() {
        let engine = make_auth_engine();
        let admin_id = auth_identity(&engine, "admin", "admin_pass");
        let cmd = parser::parse_command("GRAPH CREATE \"secret\"").unwrap();
        engine.execute_command(cmd, Some(&admin_id)).unwrap();

        // Writer with "app:*" pattern cannot write to "secret"
        let writer_id = auth_identity(&engine, "writer", "writer_pass");
        let cmd = parser::parse_command(
            r#"NODE ADD TO "secret" LABEL "doc""#,
        )
        .unwrap();
        let result = engine.execute_command(cmd, Some(&writer_id));
        assert!(result.is_err());
        match result.unwrap_err() {
            WeavError::PermissionDenied(msg) => {
                assert!(msg.contains("write access"));
            }
            other => panic!("expected PermissionDenied, got: {:?}", other),
        }
    }

    #[test]
    fn test_auth_writer_can_read_matching_graph() {
        let engine = make_auth_engine();
        let admin_id = auth_identity(&engine, "admin", "admin_pass");
        let cmd = parser::parse_command("GRAPH CREATE \"app:docs\"").unwrap();
        engine.execute_command(cmd, Some(&admin_id)).unwrap();

        // ReadWrite permission implies Read
        let writer_id = auth_identity(&engine, "writer", "writer_pass");
        let cmd = parser::parse_command("GRAPH INFO \"app:docs\"").unwrap();
        let result = engine.execute_command(cmd, Some(&writer_id));
        assert!(result.is_ok(), "writer should be able to read app:docs");
    }

    #[test]
    fn test_auth_admin_three_tier_graph_create_requires_admin() {
        // This test verifies the critical fix: Admin operations on a graph
        // require Admin-level graph permission, not just ReadWrite.
        use weav_core::config::{AuthConfig, UserConfig, GraphPatternConfig};
        let mut config = WeavConfig::default();
        config.auth = AuthConfig {
            enabled: true,
            require_auth: true,
            acl_file: None,
            default_password: None,
            users: vec![
                // User with Admin category but only ReadWrite graph permission
                UserConfig {
                    username: "rw_admin".into(),
                    password: Some("pass".into()),
                    categories: vec!["+@all".into()],
                    graph_patterns: vec![GraphPatternConfig {
                        pattern: "*".into(),
                        permission: "readwrite".into(),
                    }],
                    api_keys: Vec::new(),
                    enabled: true,
                },
            ],
        };
        let engine = Engine::new(config);
        let id = engine
            .acl_store()
            .unwrap()
            .authenticate("rw_admin", "pass")
            .unwrap();

        // User has Admin category permission but only ReadWrite graph permission.
        // Graph create requires Admin graph permission, so this must fail.
        let cmd = parser::parse_command("GRAPH CREATE \"mydb\"").unwrap();
        let result = engine.execute_command(cmd, Some(&id));
        assert!(result.is_err(), "readwrite graph perm should not allow graph create");
        match result.unwrap_err() {
            WeavError::PermissionDenied(msg) => {
                assert!(msg.contains("admin access"), "should mention admin access, got: {}", msg);
            }
            other => panic!("expected PermissionDenied, got: {:?}", other),
        }
    }

    #[test]
    fn test_auth_admin_three_tier_graph_drop_requires_admin() {
        use weav_core::config::{AuthConfig, UserConfig, GraphPatternConfig};
        let mut config = WeavConfig::default();
        config.auth = AuthConfig {
            enabled: true,
            require_auth: true,
            acl_file: None,
            default_password: None,
            users: vec![
                // Full admin for setup
                UserConfig {
                    username: "superadmin".into(),
                    password: Some("pass".into()),
                    categories: vec!["+@all".into()],
                    graph_patterns: Vec::new(),
                    api_keys: Vec::new(),
                    enabled: true,
                },
                // User with Admin category but only ReadWrite graph permission
                UserConfig {
                    username: "rw_only".into(),
                    password: Some("pass".into()),
                    categories: vec!["+@all".into()],
                    graph_patterns: vec![GraphPatternConfig {
                        pattern: "*".into(),
                        permission: "readwrite".into(),
                    }],
                    api_keys: Vec::new(),
                    enabled: true,
                },
            ],
        };
        let engine = Engine::new(config);

        // Superadmin creates the graph
        let super_id = engine.acl_store().unwrap()
            .authenticate("superadmin", "pass").unwrap();
        let cmd = parser::parse_command("GRAPH CREATE \"todrop\"").unwrap();
        engine.execute_command(cmd, Some(&super_id)).unwrap();

        // User with only ReadWrite graph permission cannot drop it
        let rw_id = engine.acl_store().unwrap()
            .authenticate("rw_only", "pass").unwrap();
        let cmd = parser::parse_command("GRAPH DROP \"todrop\"").unwrap();
        let result = engine.execute_command(cmd, Some(&rw_id));
        assert!(result.is_err(), "readwrite graph perm should not allow graph drop");
        match result.unwrap_err() {
            WeavError::PermissionDenied(msg) => {
                assert!(msg.contains("admin access"), "should mention admin access, got: {}", msg);
            }
            other => panic!("expected PermissionDenied, got: {:?}", other),
        }
    }

    #[test]
    fn test_auth_disabled_allows_all() {
        // With auth disabled (default config), everything works without identity
        let engine = make_engine();
        create_test_graph(&engine, "noauth");
        let cmd = parser::parse_command(r#"NODE ADD TO "noauth" LABEL "test""#).unwrap();
        let result = engine.execute_command(cmd, None);
        assert!(result.is_ok(), "auth disabled should allow all operations");
    }

    #[test]
    fn test_check_permission_auth_disabled() {
        // check_permission should be a no-op when auth is disabled
        let engine = make_engine();
        let result = engine.check_permission(
            None,
            "anything",
            weav_auth::identity::GraphPermission::Admin,
        );
        assert!(result.is_ok(), "check_permission should pass when auth disabled");
    }

    #[test]
    fn test_check_permission_no_identity_auth_enabled() {
        let engine = make_auth_engine();
        let result = engine.check_permission(
            None,
            "test",
            weav_auth::identity::GraphPermission::Read,
        );
        assert!(result.is_err(), "check_permission with no identity should fail when auth required");
        assert!(matches!(result.unwrap_err(), WeavError::AuthenticationRequired));
    }

    #[test]
    fn test_check_permission_insufficient_level() {
        let engine = make_auth_engine();
        let id = auth_identity(&engine, "reader", "reader_pass");
        // Reader has Read on "shared", but requesting ReadWrite should fail
        let result = engine.check_permission(
            Some(&id),
            "shared",
            weav_auth::identity::GraphPermission::ReadWrite,
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            WeavError::PermissionDenied(msg) => {
                assert!(msg.contains("insufficient permission"));
            }
            other => panic!("expected PermissionDenied, got: {:?}", other),
        }
    }

    #[test]
    fn test_check_permission_sufficient_level() {
        let engine = make_auth_engine();
        let id = auth_identity(&engine, "writer", "writer_pass");
        // Writer has ReadWrite on "app:*", requesting Read should succeed
        let result = engine.check_permission(
            Some(&id),
            "app:test",
            weav_auth::identity::GraphPermission::Read,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_auth_edge_operations_require_write() {
        let engine = make_auth_engine();
        let admin_id = auth_identity(&engine, "admin", "admin_pass");

        // Set up: create graph and nodes
        let cmd = parser::parse_command("GRAPH CREATE \"shared\"").unwrap();
        engine.execute_command(cmd, Some(&admin_id)).unwrap();
        let cmd = parser::parse_command(r#"NODE ADD TO "shared" LABEL "a""#).unwrap();
        let n1 = match engine.execute_command(cmd, Some(&admin_id)).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };
        let cmd = parser::parse_command(r#"NODE ADD TO "shared" LABEL "b""#).unwrap();
        let n2 = match engine.execute_command(cmd, Some(&admin_id)).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Reader cannot add edges
        let reader_id = auth_identity(&engine, "reader", "reader_pass");
        let cmd = parser::parse_command(
            &format!(r#"EDGE ADD TO "shared" FROM {n1} TO {n2} LABEL "link""#),
        )
        .unwrap();
        let result = engine.execute_command(cmd, Some(&reader_id));
        assert!(result.is_err(), "reader should not be able to add edges");
    }

    #[test]
    fn test_auth_stats_respects_graph_permission() {
        let engine = make_auth_engine();
        let admin_id = auth_identity(&engine, "admin", "admin_pass");
        let cmd = parser::parse_command("GRAPH CREATE \"secret\"").unwrap();
        engine.execute_command(cmd, Some(&admin_id)).unwrap();

        // Reader cannot access stats for a graph they can't read
        let reader_id = auth_identity(&engine, "reader", "reader_pass");
        let cmd = Command::Stats(Some("secret".into()));
        let result = engine.execute_command(cmd, Some(&reader_id));
        assert!(result.is_err(), "reader should not see stats for inaccessible graph");
    }

    #[test]
    fn test_auth_stats_no_graph_always_allowed() {
        let engine = make_auth_engine();
        let reader_id = auth_identity(&engine, "reader", "reader_pass");
        // Stats without graph name is a read operation, reader has +@read
        let cmd = Command::Stats(None);
        let result = engine.execute_command(cmd, Some(&reader_id));
        assert!(result.is_ok(), "reader should see global stats");
    }

    #[test]
    fn test_metrics_recorded_on_execute() {
        let engine = make_engine();
        create_test_graph(&engine, "metrics_g");

        // Execute a few commands
        let cmd = parser::parse_command(
            r#"NODE ADD TO "metrics_g" LABEL "person" PROPERTIES {"name": "Alice"}"#,
        ).unwrap();
        engine.execute_command(cmd, None).unwrap();

        let cmd = parser::parse_command("PING").unwrap();
        engine.execute_command(cmd, None).unwrap();

        // Verify query_total counter was incremented
        let node_add_ok = crate::metrics::QUERY_TOTAL
            .with_label_values(&["node_add", "ok"])
            .get();
        assert!(node_add_ok >= 1, "node_add ok count should be >= 1, got {node_add_ok}");

        let ping_ok = crate::metrics::QUERY_TOTAL
            .with_label_values(&["ping", "ok"])
            .get();
        assert!(ping_ok >= 1, "ping ok count should be >= 1, got {ping_ok}");

        // Verify duration was recorded (any observation means it works)
        let duration_count = crate::metrics::QUERY_DURATION
            .with_label_values(&["node_add"])
            .get_sample_count();
        assert!(duration_count >= 1, "duration should have at least 1 observation");
    }

    #[test]
    fn test_command_type_name() {
        let cmd = parser::parse_command("PING").unwrap();
        assert_eq!(cmd.type_name(), "ping");

        let cmd = parser::parse_command(
            r#"NODE ADD TO "g" LABEL "x" PROPERTIES {"k": "v"}"#,
        ).unwrap();
        assert_eq!(cmd.type_name(), "node_add");

        let cmd = parser::parse_command("GRAPH LIST").unwrap();
        assert_eq!(cmd.type_name(), "graph_list");
    }

    #[test]
    fn test_node_update_with_embedding_persisted() {
        // Verify that node update with embedding creates in-memory vector
        let engine = make_engine();
        create_test_graph(&engine, "emb_g");

        // Add a node first
        let cmd = parser::parse_command(
            r#"NODE ADD TO "emb_g" LABEL "entity" PROPERTIES {"name": "test"}"#,
        ).unwrap();
        let node_id = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Update with embedding
        let update_cmd = weav_query::parser::NodeUpdateCmd {
            graph: "emb_g".to_string(),
            node_id,
            properties: vec![("updated".to_string(), Value::Bool(true))],
            embedding: Some(vec![1.0; 1536]),
        };
        engine.execute_command(Command::NodeUpdate(update_cmd), None).unwrap();

        // Verify the embedding is searchable via context query
        let graph_arc = engine.get_graph("emb_g").unwrap();
        let gs = graph_arc.read();
        let results = gs.vector_index.search(&vec![1.0; 1536], 1, None).unwrap();
        assert!(!results.is_empty(), "embedding should be searchable after update");
        assert_eq!(results[0].0, node_id);
    }

    #[test]
    fn test_ttl_node_expiry() {
        let engine = make_engine();
        create_test_graph(&engine, "ttl_g");

        // Add a node with TTL of 1ms (already expired by the time we sweep)
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "ttl_g".to_string(),
            label: "temp".to_string(),
            properties: vec![("name".to_string(), Value::String(CompactString::from("ephemeral")))],
            embedding: None,
            entity_key: None,
            ttl_ms: Some(1), // 1ms TTL — will expire almost immediately
        });
        let node_id = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add a permanent node
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "ttl_g".to_string(),
            label: "perm".to_string(),
            properties: vec![("name".to_string(), Value::String(CompactString::from("permanent")))],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let perm_id = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Wait a tiny bit for TTL to expire
        std::thread::sleep(std::time::Duration::from_millis(5));

        // Sweep should remove the expired node
        let expired = engine.sweep_ttl();
        assert!(expired >= 1, "at least 1 expired entity should be removed, got {expired}");

        // Expired node should be gone
        let graph_arc = engine.get_graph("ttl_g").unwrap();
        let gs = graph_arc.read();
        assert!(!gs.adjacency.has_node(node_id), "expired node should be removed");
        assert!(gs.adjacency.has_node(perm_id), "permanent node should remain");
    }

    #[test]
    fn test_ttl_edge_expiry() {
        let engine = make_engine();
        create_test_graph(&engine, "ttl_e");

        // Add two nodes
        let cmd = parser::parse_command(
            r#"NODE ADD TO "ttl_e" LABEL "a" PROPERTIES {"name": "A"}"#,
        ).unwrap();
        let n1 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };
        let cmd = parser::parse_command(
            r#"NODE ADD TO "ttl_e" LABEL "b" PROPERTIES {"name": "B"}"#,
        ).unwrap();
        let n2 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add edge with 1ms TTL
        let cmd = Command::EdgeAdd(parser::EdgeAddCmd {
            graph: "ttl_e".to_string(),
            source: n1,
            target: n2,
            label: "temp_link".to_string(),
            weight: 1.0,
            properties: vec![],
            ttl_ms: Some(1),
        });
        engine.execute_command(cmd, None).unwrap();

        // Add permanent edge
        let cmd = Command::EdgeAdd(parser::EdgeAddCmd {
            graph: "ttl_e".to_string(),
            source: n2,
            target: n1,
            label: "perm_link".to_string(),
            weight: 1.0,
            properties: vec![],
            ttl_ms: None,
        });
        engine.execute_command(cmd, None).unwrap();

        std::thread::sleep(std::time::Duration::from_millis(5));

        let expired = engine.sweep_ttl();
        assert!(expired >= 1, "at least 1 expired edge should be removed");

        // Permanent edge should remain
        let graph_arc = engine.get_graph("ttl_e").unwrap();
        let gs = graph_arc.read();
        assert_eq!(gs.adjacency.edge_count(), 1, "only permanent edge should remain");
    }

    #[test]
    fn test_sweep_ttl_empty_graphs() {
        let engine = make_engine();
        // Sweep on empty engine should not panic
        let expired = engine.sweep_ttl();
        assert_eq!(expired, 0);

        // Sweep on engine with graph but no TTL nodes
        create_test_graph(&engine, "no_ttl");
        let cmd = parser::parse_command(
            r#"NODE ADD TO "no_ttl" LABEL "x" PROPERTIES {"k": "v"}"#,
        ).unwrap();
        engine.execute_command(cmd, None).unwrap();

        let expired = engine.sweep_ttl();
        assert_eq!(expired, 0);
    }
}
