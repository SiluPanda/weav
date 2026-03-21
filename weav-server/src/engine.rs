//! The Weav engine: central coordinator holding all in-memory state.
//!
//! Provides a thread-safe interface for executing commands against graphs.

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::SystemTime;

use parking_lot::{Mutex, RwLock};

use compact_str::CompactString;

use weav_core::config::{EvictionPolicy, GraphConfig, WeavConfig};
use weav_core::error::{WeavError, WeavResult};
use weav_core::events::{EventKind, GraphEvent};
use weav_core::shard::StringInterner;
use weav_core::types::*;
use weav_graph::adjacency::{AdjacencyStore, EdgeMeta};
use weav_graph::properties::PropertyStore;
use weav_graph::text_index::TextIndex;
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
    pub vector_count: usize,
    pub label_count: usize,
    pub default_ttl_ms: Option<u64>,
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
    /// Full-text inverted index with BM25 scoring.
    pub text_index: TextIndex,
    /// Last access time per node for LRU eviction (ms since epoch).
    pub access_times: HashMap<NodeId, u64>,
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
    /// Active connection counter for enforcing max_connections.
    active_connections: std::sync::atomic::AtomicU64,
    /// When true, WAL writes are suppressed (during recovery replay).
    replaying: std::sync::atomic::AtomicBool,
    /// CDC broadcast channel for streaming mutation events to subscribers.
    event_tx: tokio::sync::broadcast::Sender<GraphEvent>,
    /// Monotonically increasing sequence counter for CDC events.
    event_sequence: AtomicU64,
    /// Ring buffer of recent events (last 10,000) for polling/testing.
    event_log: RwLock<Vec<GraphEvent>>,
}

/// Extract graph name and operation type from a WAL operation for metric labels.
fn wal_op_labels(op: &WalOperation) -> (&str, &str) {
    match op {
        WalOperation::GraphCreate { name, .. } => (name.as_str(), "GraphCreate"),
        WalOperation::GraphDrop { name, .. } => (name.as_str(), "GraphDrop"),
        WalOperation::NodeAdd { .. } => ("_", "NodeAdd"),
        WalOperation::NodeUpdate { .. } => ("_", "NodeUpdate"),
        WalOperation::NodeDelete { .. } => ("_", "NodeDelete"),
        WalOperation::EdgeAdd { .. } => ("_", "EdgeAdd"),
        WalOperation::EdgeInvalidate { .. } => ("_", "EdgeInvalidate"),
        WalOperation::EdgeDelete { .. } => ("_", "EdgeDelete"),
        WalOperation::VectorUpdate { .. } => ("_", "VectorUpdate"),
        WalOperation::Ingest { graph_name, .. } => (graph_name.as_str(), "Ingest"),
    }
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

        let (event_tx, _) = tokio::sync::broadcast::channel(4096);

        Self {
            graphs: RwLock::new(HashMap::new()),
            next_graph_id: RwLock::new(1),
            token_counter,
            config,
            wal,
            snapshot_engine,
            runtime_config: RwLock::new(HashMap::new()),
            acl_store,
            active_connections: std::sync::atomic::AtomicU64::new(0),
            replaying: std::sync::atomic::AtomicBool::new(false),
            event_tx,
            event_sequence: AtomicU64::new(0),
            event_log: RwLock::new(Vec::new()),
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

    /// Record an LRU access time for a node.
    fn record_access(gs: &mut GraphState, node_id: NodeId) {
        let now = Self::now_ms();
        gs.access_times.insert(node_id, now);
    }

    /// Evict a single node, removing its edges, properties, vector entry, text
    /// index entry, and access-time tracking.
    #[allow(dead_code)]
    fn evict_node(&self, gs: &mut GraphState, victim_id: NodeId, graph_name: &str) {
        // Remove the node (cascades edges via AdjacencyStore::remove_node).
        let _ = gs.adjacency.remove_node(victim_id);
        gs.properties.remove_all_node_properties(victim_id);
        let _ = gs.vector_index.remove(victim_id);
        gs.text_index.remove_node(victim_id);
        gs.access_times.remove(&victim_id);

        // Update metrics after eviction.
        crate::metrics::NODES_TOTAL
            .with_label_values(&[graph_name])
            .set(gs.adjacency.node_count() as i64);
        crate::metrics::EDGES_TOTAL
            .with_label_values(&[graph_name])
            .set(gs.adjacency.edge_count() as i64);
    }

    // ── CDC event infrastructure ──────────────────────────────────────────

    /// Subscribe to the CDC event stream. Returns a broadcast receiver that
    /// yields every [`GraphEvent`] emitted by mutation handlers.
    pub fn subscribe_events(&self) -> tokio::sync::broadcast::Receiver<GraphEvent> {
        self.event_tx.subscribe()
    }

    /// Return the most recent CDC events (newest first), up to `limit`.
    pub fn recent_events(&self, limit: usize) -> Vec<GraphEvent> {
        let log = self.event_log.read();
        log.iter().rev().take(limit).cloned().collect()
    }

    /// Emit a CDC event after a successful mutation.
    fn emit_event(&self, graph: &str, kind: EventKind) {
        let seq = self
            .event_sequence
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let event = GraphEvent {
            sequence: seq,
            graph: CompactString::from(graph),
            timestamp: Self::now_ms(),
            kind,
        };
        // Broadcast to live subscribers (ignore error = no active receivers).
        let _ = self.event_tx.send(event.clone());
        // Append to ring buffer for polling / testing.
        let mut log = self.event_log.write();
        log.push(event);
        if log.len() > 10_000 {
            log.drain(..5_000);
        }
    }

    fn append_wal(&self, op: WalOperation) -> WeavResult<()> {
        // Skip WAL writes during recovery replay to avoid doubling WAL size
        if self.replaying.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(());
        }
        if let Some(ref wal_mutex) = self.wal {
            // Extract labels and size before moving `op` into append.
            let (graph_label, op_label) = wal_op_labels(&op);
            let graph_label = graph_label.to_string();
            let op_label = op_label.to_string();
            let estimated_bytes = bincode::serialized_size(&op).unwrap_or(0);
            let mut wal = wal_mutex.lock();
            wal.append(0, op)
                .map_err(|e| WeavError::PersistenceError(format!("WAL write failed: {e}")))?;
            crate::metrics::WAL_WRITES_TOTAL
                .with_label_values(&[&graph_label, &op_label])
                .inc();
            crate::metrics::WAL_BYTES_WRITTEN
                .inc_by(4 + estimated_bytes);
        }
        Ok(())
    }

    /// Force-sync the WAL file to disk. Called periodically for EverySecond mode.
    pub fn sync_wal(&self) -> WeavResult<()> {
        if let Some(ref wal_mutex) = self.wal {
            let start = std::time::Instant::now();
            let mut wal = wal_mutex.lock();
            wal.sync()
                .map_err(|e| WeavError::PersistenceError(format!("WAL sync failed: {e}")))?;
            crate::metrics::WAL_SYNC_DURATION.observe(start.elapsed().as_secs_f64());
        }
        Ok(())
    }

    /// Return the configured WAL sync mode.
    pub fn wal_sync_mode(&self) -> &weav_core::config::WalSyncMode {
        &self.config.persistence.wal_sync_mode
    }

    /// Try to acquire a connection slot. Returns Err if max_connections exceeded.
    pub fn try_acquire_connection(&self) -> WeavResult<()> {
        let max = self.config.server.max_connections as u64;
        let current = self.active_connections.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if current >= max {
            self.active_connections.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            return Err(WeavError::CapacityExceeded(format!(
                "max connections ({max}) exceeded"
            )));
        }
        crate::metrics::CONNECTIONS_ACTIVE.set((current + 1) as i64);
        Ok(())
    }

    /// Release a connection slot.
    pub fn release_connection(&self) {
        // Use compare-and-swap loop to prevent underflow
        loop {
            let current = self.active_connections.load(std::sync::atomic::Ordering::Relaxed);
            if current == 0 {
                break;
            }
            if self.active_connections.compare_exchange(
                current,
                current - 1,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            ).is_ok() {
                crate::metrics::CONNECTIONS_ACTIVE.set((current - 1) as i64);
                break;
            }
        }
    }

    /// Return the current active connection count.
    pub fn active_connection_count(&self) -> u64 {
        self.active_connections.load(std::sync::atomic::Ordering::Relaxed)
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
            if !expired_nodes.is_empty() {
                crate::metrics::TTL_EXPIRED_TOTAL
                    .with_label_values(&[graph_name, "node"])
                    .inc_by(expired_nodes.len() as u64);
            }
            if !expired_edges.is_empty() {
                crate::metrics::TTL_EXPIRED_TOTAL
                    .with_label_values(&[graph_name, "edge"])
                    .inc_by(expired_edges.len() as u64);
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
    pub fn get_graph(&self, name: &str) -> WeavResult<Arc<RwLock<GraphState>>> {
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
                    text_index: TextIndex::new(),
                    access_times: HashMap::new(),
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
                        // Try tagged Value deserialization first (full fidelity),
                        // fall back to json_val_to_value for legacy snapshots.
                        if let Ok(props) = serde_json::from_str::<std::collections::HashMap<String, Value>>(&ns.properties_json) {
                            for (k, v) in props {
                                state.properties.set_node_property(ns.node_id, &k, v);
                            }
                        } else if let Ok(props_val) = serde_json::from_str::<serde_json::Value>(&ns.properties_json)
                            && let Some(obj) = props_val.as_object()
                        {
                            for (k, v) in obj {
                                state.properties.set_node_property(
                                    ns.node_id, k,
                                    crate::http::json_val_to_value(v),
                                );
                            }
                        }
                    }
                    // Re-index text content for full-text search during restore
                    let all_props = state.properties.get_all_node_properties(ns.node_id);
                    let text_content: String = all_props.iter()
                        .filter(|(k, _)| !k.starts_with('_'))
                        .filter_map(|(_, v)| v.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    if !text_content.is_empty() {
                        state.text_index.index_node(ns.node_id, &text_content);
                    }

                    if let Some(ref emb) = ns.embedding {
                        let _ = state.vector_index.insert(ns.node_id, emb);
                    }
                    if ns.node_id >= state.next_node_id {
                        state.next_node_id = ns.node_id + 1;
                    }
                }

                // Restore edges with original edge IDs.
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
                    // Use add_edge_with_id to preserve the original edge ID from snapshot
                    let _ = state.adjacency.add_edge_with_id(es.source, es.target, label_id, meta, es.edge_id);
                    if es.edge_id >= state.next_edge_id {
                        state.next_edge_id = es.edge_id + 1;
                        // Sync the adjacency store's internal edge ID counter
                        state.adjacency.set_next_edge_id(es.edge_id + 1);
                    }
                    // Restore edge properties from snapshot.
                    // Properties are serialized as serde-tagged Value enums, so we
                    // deserialize directly into HashMap<String, Value> for fidelity.
                    if !es.properties_json.is_empty() && es.properties_json != "{}"
                        && let Ok(props) = serde_json::from_str::<std::collections::HashMap<String, Value>>(&es.properties_json)
                    {
                        for (k, v) in props {
                            state.properties.set_edge_property(es.edge_id, &k, v);
                        }
                    }
                }

                let mut graphs = self.graphs.write();
                graphs.insert(gs.graph_name.clone(), Arc::new(RwLock::new(state)));
            }
        }

        // Step 2: Replay WAL entries.
        // Set replaying flag to suppress WAL writes during recovery —
        // handle_* methods call append_wal which would otherwise double the WAL size.
        self.replaying.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut replay_errors: u64 = 0;
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
                    if let Err(e) = self.handle_graph_create(cmd) {
                        tracing::warn!("WAL replay GraphCreate '{}' failed: {e}", name);
                        replay_errors += 1;
                    }
                }
                WalOperation::GraphDrop { name } => {
                    if let Err(e) = self.handle_graph_drop(name) {
                        tracing::warn!("WAL replay GraphDrop '{}' failed: {e}", name);
                        replay_errors += 1;
                    }
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
                            // Try tagged Value deserialization first (full fidelity)
                            if let Ok(tagged) = serde_json::from_str::<std::collections::HashMap<String, Value>>(properties_json) {
                                for (k, v) in tagged {
                                    props.push((k, v));
                                }
                            } else if let Ok(val) = serde_json::from_str::<serde_json::Value>(properties_json)
                                && let Some(obj) = val.as_object()
                            {
                                for (k, v) in obj {
                                    props.push((k.clone(), crate::http::json_val_to_value(v)));
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
                        if let Err(e) = self.handle_node_add(cmd) {
                            tracing::warn!("WAL replay NodeAdd failed: {e}");
                            replay_errors += 1;
                        }
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
                                // Try tagged Value deserialization first (full fidelity)
                                if let Ok(props) = serde_json::from_str::<std::collections::HashMap<String, Value>>(properties_json) {
                                    for (k, v) in props {
                                        gs.properties.set_node_property(*node_id, &k, v);
                                    }
                                } else if let Ok(val) = serde_json::from_str::<serde_json::Value>(properties_json)
                                    && let Some(obj) = val.as_object()
                                {
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
                            if let Err(e) = gs.adjacency.remove_node(*node_id) {
                                tracing::warn!("WAL replay NodeDelete({node_id}) failed: {e}");
                                replay_errors += 1;
                            }
                            gs.properties.remove_all_node_properties(*node_id);
                            if let Err(e) = gs.vector_index.remove(*node_id) {
                                tracing::warn!("WAL replay NodeDelete vector cleanup({node_id}) failed: {e}");
                            }
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
                    properties_json,
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
                            // Use add_edge_with_id to preserve the original edge ID
                            if let Err(e) = gs.adjacency.add_edge_with_id(*source, *target, label_id, meta, *edge_id) {
                                tracing::warn!("WAL replay EdgeAdd({source}->{target}, id={edge_id}) failed: {e}");
                                replay_errors += 1;
                            }
                            // Ensure next_edge_id stays ahead of all replayed IDs
                            if *edge_id >= gs.next_edge_id {
                                gs.next_edge_id = *edge_id + 1;
                                // Sync the adjacency store's internal edge ID counter
                                gs.adjacency.set_next_edge_id(*edge_id + 1);
                            }
                            // Restore edge properties from WAL.
                            // Properties are serialized as serde-tagged Value enums, so we
                            // deserialize directly into HashMap<String, Value> for fidelity.
                            if !properties_json.is_empty() && properties_json != "{}"
                                && let Ok(props) = serde_json::from_str::<std::collections::HashMap<String, Value>>(properties_json)
                            {
                                for (k, v) in props {
                                    gs.properties.set_edge_property(*edge_id, &k, v);
                                }
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
                        if let Err(e) = gs.adjacency.invalidate_edge(*edge_id, *timestamp) {
                            tracing::warn!("WAL replay EdgeInvalidate({edge_id}) failed: {e}");
                            replay_errors += 1;
                        }
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
                        if let Err(e) = gs.adjacency.remove_edge(*edge_id) {
                            tracing::warn!("WAL replay EdgeDelete({edge_id}) failed: {e}");
                            replay_errors += 1;
                        }
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
                        // insert already handles remove-then-add internally
                        if let Err(e) = gs.vector_index.insert(*node_id, vector) {
                            tracing::warn!("WAL replay VectorUpdate({node_id}) failed: {e}");
                            replay_errors += 1;
                        }
                    }
                }
                WalOperation::Ingest { .. } => {
                    // Ingest WAL entries are metadata only; the actual data changes
                    // (NodeAdd, EdgeAdd, etc.) are recorded as separate WAL entries.
                }
            }
        }

        // Clear replaying flag — normal WAL writes resume
        self.replaying.store(false, std::sync::atomic::Ordering::Relaxed);

        if replay_errors > 0 {
            tracing::warn!(
                "WAL replay completed with {replay_errors} errors out of {} entries",
                result.wal_entries.len()
            );
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
            Command::Search(cmd) => self.handle_search(cmd),
            Command::SearchText(cmd) => self.handle_search_text(cmd),
            Command::Neighbors(cmd) => self.handle_neighbors(cmd),
            Command::SchemaSet(cmd) => self.handle_schema_set(cmd),
            Command::SchemaGet(cmd) => self.handle_schema_get(cmd),
            Command::NodeMerge(cmd) => self.handle_node_merge_authed(cmd, identity),
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
            | Command::ConfigGet(_) | Command::AclWhoAmI
            | Command::Search(_) | Command::SearchText(_) | Command::Neighbors(_)
            | Command::SchemaGet(_) => CommandCategory::Read,
            Command::NodeAdd(_) | Command::NodeUpdate(_) | Command::NodeDelete(_)
            | Command::EdgeAdd(_) | Command::EdgeDelete(_) | Command::EdgeInvalidate(_)
            | Command::BulkInsertNodes(_) | Command::BulkInsertEdges(_)
            | Command::Ingest(_) | Command::NodeMerge(_) => CommandCategory::Write,
            Command::GraphCreate(_) | Command::GraphDrop(_) | Command::Snapshot
            | Command::ConfigSet(_, _) | Command::AclSetUser(_) | Command::AclDelUser(_)
            | Command::AclList | Command::AclGetUser(_) | Command::AclSave
            | Command::AclLoad | Command::SchemaSet(_) => CommandCategory::Admin,
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
    pub fn check_permission(
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
            Command::Search(c) => Some(c.graph.clone()),
            Command::SearchText(c) => Some(c.graph.clone()),
            Command::Neighbors(c) => Some(c.graph.clone()),
            Command::GraphCreate(c) => Some(c.name.clone()),
            Command::GraphDrop(name) | Command::GraphInfo(name) => Some(name.clone()),
            Command::SchemaSet(c) => Some(c.graph.clone()),
            Command::SchemaGet(c) => Some(c.graph.clone()),
            Command::NodeMerge(c) => Some(c.graph.clone()),
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

    fn handle_node_merge_authed(
        &self,
        cmd: weav_query::parser::NodeMergeCmd,
        identity: Option<&weav_auth::identity::SessionIdentity>,
    ) -> WeavResult<CommandResponse> {
        self.check_permission(identity, &cmd.graph, weav_auth::identity::GraphPermission::ReadWrite)?;
        self.handle_node_merge(cmd)
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

            // Compute label distribution
            let all_nodes = gs.adjacency.all_node_ids();
            let mut label_counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
            for nid in &all_nodes {
                if let Some(label_val) = gs.properties.get_node_property(*nid, "_label") {
                    let label = label_val.as_str().unwrap_or("unknown").to_string();
                    *label_counts.entry(label).or_insert(0) += 1;
                }
            }

            // Count TTL nodes
            let ttl_nodes = gs.properties.nodes_with_property("_ttl_expires_at").len();

            // Compute avg degree
            let node_count = gs.adjacency.node_count();
            let edge_count = gs.adjacency.edge_count();
            let avg_degree = if node_count > 0 {
                (edge_count as f64 * 2.0) / node_count as f64
            } else {
                0.0
            };

            // Build label distribution string
            let mut label_parts: Vec<String> = label_counts
                .iter()
                .map(|(k, v)| format!("{}:{}", k, v))
                .collect();
            label_parts.sort();

            Ok(CommandResponse::Text(format!(
                "graph={} nodes={} edges={} vectors={} labels={{{}}} avg_degree={:.2} ttl_nodes={} interned_labels={}",
                name,
                node_count,
                edge_count,
                gs.vector_index.len(),
                label_parts.join(","),
                avg_degree,
                ttl_nodes,
                gs.interner.label_count(),
            )))
        } else {
            let registry = self.graphs.read();
            let mut total_nodes: u64 = 0;
            let mut total_edges: u64 = 0;
            let mut total_vectors: usize = 0;
            for graph_arc in registry.values() {
                let gs = graph_arc.read();
                total_nodes += gs.adjacency.node_count();
                total_edges += gs.adjacency.edge_count();
                total_vectors += gs.vector_index.len();
            }
            Ok(CommandResponse::Text(format!(
                "graphs={} total_nodes={} total_edges={} total_vectors={} engine=weav-server v{}",
                registry.len(),
                total_nodes,
                total_edges,
                total_vectors,
                env!("CARGO_PKG_VERSION"),
            )))
        }
    }

    // ── Search and neighbors commands ─────────────────────────────────────

    fn handle_search(
        &self,
        cmd: weav_query::parser::SearchCmd,
    ) -> WeavResult<CommandResponse> {
        let graph_arc = self.get_graph(&cmd.graph)?;
        let gs = graph_arc.read();

        let search_value = cmd.value.clone();
        let matching = gs.properties.nodes_where(&cmd.key, &move |v| {
            match v {
                Value::String(s) => s.as_str() == search_value,
                Value::Int(i) => i.to_string() == search_value,
                Value::Float(f) => f.to_string() == search_value,
                Value::Bool(b) => b.to_string() == search_value,
                _ => false,
            }
        });

        let limit = cmd.limit.unwrap_or(100) as usize;
        let result_ids: Vec<u64> = matching.into_iter().take(limit).collect();

        // Build result strings: "node_id:label"
        let results: Vec<String> = result_ids.iter().map(|&nid| {
            let label = gs.properties
                .get_node_property(nid, "_label")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            format!("{}:{}", nid, label)
        }).collect();

        Ok(CommandResponse::StringList(results))
    }

    fn handle_search_text(
        &self,
        cmd: weav_query::parser::SearchTextCmd,
    ) -> WeavResult<CommandResponse> {
        let graph_arc = self.get_graph(&cmd.graph)?;
        let gs = graph_arc.read();
        let limit = cmd.limit.unwrap_or(20) as usize;
        let results = gs.text_index.search(&cmd.query, limit);

        // Build result strings: "node_id:label:score"
        let result_strings: Vec<String> = results.iter().map(|&(nid, score)| {
            let label = gs.properties
                .get_node_property(nid, "_label")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            format!("{}:{}:{:.4}", nid, label, score)
        }).collect();

        Ok(CommandResponse::StringList(result_strings))
    }

    fn handle_neighbors(
        &self,
        cmd: weav_query::parser::NeighborsCmd,
    ) -> WeavResult<CommandResponse> {
        use weav_core::types::Direction;

        let graph_arc = self.get_graph(&cmd.graph)?;
        let gs = graph_arc.read();

        if !gs.adjacency.has_node(cmd.node_id) {
            return Err(WeavError::NodeNotFound(cmd.node_id, gs.graph_id));
        }

        let label_id = cmd.label.as_ref()
            .and_then(|l| gs.interner.resolve_label_id(l));

        let neighbors = match cmd.direction {
            Direction::Outgoing => {
                gs.adjacency.neighbors_out(cmd.node_id, label_id)
                    .into_iter()
                    .map(|(nid, eid)| (nid, eid, Direction::Outgoing))
                    .collect::<Vec<_>>()
            }
            Direction::Incoming => {
                gs.adjacency.neighbors_in(cmd.node_id, label_id)
                    .into_iter()
                    .map(|(nid, eid)| (nid, eid, Direction::Incoming))
                    .collect::<Vec<_>>()
            }
            Direction::Both => {
                gs.adjacency.neighbors_both(cmd.node_id, label_id)
            }
        };

        let results: Vec<String> = neighbors.iter().map(|&(nid, eid, ref dir)| {
            let dir_str = match dir {
                Direction::Outgoing => "OUT",
                Direction::Incoming => "IN",
                Direction::Both => "BOTH",
            };
            let edge_label = gs.adjacency.get_edge(eid)
                .and_then(|e| gs.interner.resolve_label(e.label))
                .unwrap_or("unknown");
            format!("{}:{}:{}:{}", nid, eid, dir_str, edge_label)
        }).collect();

        Ok(CommandResponse::StringList(results))
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
            text_index: TextIndex::new(),
            access_times: HashMap::new(),
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
        let graph_name = cmd.name.clone();
        self.append_wal(WalOperation::GraphCreate {
            name: cmd.name.clone(),
            config_json,
        })?;
        graphs.insert(cmd.name, Arc::new(RwLock::new(graph_state)));
        crate::metrics::GRAPHS_TOTAL.set(graphs.len() as i64);

        self.emit_event(&graph_name, EventKind::GraphCreated {
            name: CompactString::from(&graph_name),
        });

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
        crate::metrics::GRAPHS_TOTAL.set(graphs.len() as i64);

        self.emit_event(name, EventKind::GraphDropped {
            name: CompactString::from(name),
        });

        Ok(CommandResponse::Ok)
    }

    fn handle_graph_info(&self, name: &str) -> WeavResult<CommandResponse> {
        let graph_arc = self.get_graph(name)?;
        let gs = graph_arc.read();
        Ok(CommandResponse::GraphInfo(GraphInfoResponse {
            name: gs.name.clone(),
            node_count: gs.adjacency.node_count(),
            edge_count: gs.adjacency.edge_count(),
            vector_count: gs.vector_index.len(),
            label_count: gs.interner.label_count(),
            default_ttl_ms: gs.config.default_ttl_ms,
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
        if let Some(ref key) = cmd.entity_key
            && let Some(existing_id) = weav_graph::dedup::find_duplicate_by_key(
                &gs.properties, "entity_key", key,
            )
        {
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

        // ── Dedup: fuzzy name match (if dedup_config is set) ────────────
        if let Some(ref dedup_cfg) = gs.dedup_config
            && let Some(ref name_field) = dedup_cfg.name_field
        {
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

        // Moved outside of the collapsed if block
        if let Some(ref dedup_cfg) = gs.dedup_config {
            // ── Dedup: vector similarity (if embedding provided) ────────
            if let Some(ref embedding) = cmd.embedding
                && let Ok(search_results) = gs.vector_index.search(embedding, 5, None)
            {
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

        // ── No duplicate found: create new node ─────────────────────────

        // Enforce max_nodes capacity if configured, with LRU eviction support
        if let Some(max) = gs.config.max_nodes {
            if gs.adjacency.node_count() >= max {
                match gs.config.eviction_policy {
                    EvictionPolicy::LRU => {
                        // Find least recently used node
                        let victim_id = gs.access_times
                            .iter()
                            .min_by_key(|&(_, ts)| *ts)
                            .map(|(&id, _)| id)
                            .or_else(|| gs.adjacency.all_node_ids().first().copied());
                        if let Some(victim) = victim_id {
                            let graph_name = cmd.graph.clone();
                            self.evict_node(&mut gs, victim, &graph_name);
                        }
                    }
                    _ => {
                        return Err(WeavError::CapacityExceeded(format!(
                            "graph '{}' reached max_nodes limit ({})", cmd.graph, max
                        )));
                    }
                }
            }
        }

        // ── Schema validation ────────────────────────────────────────────
        gs.config.schema.validate_node_properties(&cmd.label, &cmd.properties)?;

        // Uniqueness constraint enforcement
        for constraint in gs.config.schema.node_schemas
            .get(cmd.label.as_str())
            .map(|s| s.constraints.as_slice())
            .unwrap_or_default()
        {
            if let weav_core::schema::SchemaConstraint::PropertyUnique { property } = constraint {
                if let Some((_, prop_val)) = cmd.properties.iter().find(|(k, _)| k == property.as_str()) {
                    if *prop_val != Value::Null {
                        let search_val = prop_val.clone();
                        let existing = gs.properties.nodes_where(property.as_str(), &|v| *v == search_val);
                        // Filter to only nodes with the same label
                        let duplicates: Vec<_> = existing.into_iter().filter(|&nid| {
                            gs.properties
                                .get_node_property(nid, "_label")
                                .and_then(|v| v.as_str())
                                .map(|l| l == cmd.label)
                                .unwrap_or(false)
                        }).collect();
                        if !duplicates.is_empty() {
                            return Err(WeavError::Conflict(format!(
                                "unique constraint violation: property '{}' value already exists on label '{}'",
                                property, cmd.label
                            )));
                        }
                    }
                }
            }
        }

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

        // Auto-index text content for full-text search
        let text_content: String = cmd.properties.iter()
            .filter_map(|(_, v)| v.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        if !text_content.is_empty() {
            gs.text_index.index_node(node_id, &text_content);
        }

        if let Some(ref embedding) = cmd.embedding {
            gs.vector_index.insert(node_id, embedding)?;
        }

        // Store TTL expiry timestamp — explicit TTL overrides graph default
        let effective_ttl = cmd.ttl_ms.or(gs.config.default_ttl_ms);
        if let Some(ttl_ms) = effective_ttl {
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

        self.emit_event(&cmd.graph, EventKind::NodeCreated {
            node_id,
            label: CompactString::from(&cmd.label),
            properties: cmd.properties.iter()
                .map(|(k, v)| (CompactString::from(k.as_str()), v.clone()))
                .collect(),
        });

        // Record initial LRU access time for the new node.
        Self::record_access(&mut gs, node_id);

        Ok(CommandResponse::Integer(node_id))
    }

    fn handle_node_get(
        &self,
        cmd: weav_query::parser::NodeGetCmd,
    ) -> WeavResult<CommandResponse> {
        let graph_arc = self.get_graph(&cmd.graph)?;
        let mut gs = graph_arc.write();

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
        // Record LRU access time.
        Self::record_access(&mut gs, node_id);

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
        gs.text_index.remove_node(cmd.node_id);

        // Update metrics
        crate::metrics::NODES_TOTAL
            .with_label_values(&[&cmd.graph])
            .set(gs.adjacency.node_count() as i64);
        crate::metrics::EDGES_TOTAL
            .with_label_values(&[&cmd.graph])
            .set(gs.adjacency.edge_count() as i64);

        self.emit_event(&cmd.graph, EventKind::NodeDeleted {
            node_id: cmd.node_id,
        });

        Ok(CommandResponse::Ok)
    }

    fn handle_node_merge(
        &self,
        cmd: weav_query::parser::NodeMergeCmd,
    ) -> WeavResult<CommandResponse> {
        let graph_arc = self.get_graph(&cmd.graph)?;
        let mut gs = graph_arc.write();

        // 1. Verify both nodes exist
        if !gs.adjacency.has_node(cmd.source_id) {
            return Err(WeavError::NodeNotFound(cmd.source_id, gs.graph_id));
        }
        if !gs.adjacency.has_node(cmd.target_id) {
            return Err(WeavError::NodeNotFound(cmd.target_id, gs.graph_id));
        }
        if cmd.source_id == cmd.target_id {
            return Err(WeavError::Conflict(
                "cannot merge a node into itself".into(),
            ));
        }

        let graph_id = gs.graph_id;

        // 2. Merge properties based on conflict policy
        let source_props: Vec<(String, Value)> = gs
            .properties
            .get_all_node_properties(cmd.source_id)
            .into_iter()
            .filter(|(k, _)| !k.starts_with('_'))
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();

        for (key, src_val) in &source_props {
            let target_has = gs
                .properties
                .get_node_property(cmd.target_id, key)
                .cloned();
            match target_has {
                None => {
                    // Target missing this property, always copy
                    gs.properties
                        .set_node_property(cmd.target_id, key, src_val.clone());
                }
                Some(ref tgt_val) if tgt_val == src_val => {
                    // Same value, nothing to do
                }
                Some(_) => {
                    // Conflict: decide based on policy
                    if cmd.conflict_policy == "keep_source" {
                        gs.properties
                            .set_node_property(cmd.target_id, key, src_val.clone());
                    }
                    // "keep_target" and "merge" both keep target value on conflict
                }
            }
        }

        // 3. Re-link edges from source to target
        //    Collect edge info before mutating (EdgeMeta is not Clone)
        struct EdgeInfo {
            eid: EdgeId,
            source: NodeId,
            target: NodeId,
            label: LabelId,
            temporal: BiTemporal,
            provenance: Option<Provenance>,
            weight: f32,
            token_cost: u16,
        }
        let edge_infos: Vec<EdgeInfo> = gs
            .adjacency
            .all_edges()
            .filter(|(_, meta)| meta.source == cmd.source_id || meta.target == cmd.source_id)
            .map(|(eid, meta)| EdgeInfo {
                eid,
                source: meta.source,
                target: meta.target,
                label: meta.label,
                temporal: meta.temporal,
                provenance: meta.provenance.clone(),
                weight: meta.weight,
                token_cost: meta.token_cost,
            })
            .collect();

        for ei in &edge_infos {
            let new_src = if ei.source == cmd.source_id {
                cmd.target_id
            } else {
                ei.source
            };
            let new_tgt = if ei.target == cmd.source_id {
                cmd.target_id
            } else {
                ei.target
            };

            // Skip self-loops that would result from merge
            if new_src == new_tgt {
                continue;
            }

            // Copy edge properties
            let edge_props: Vec<(String, Value)> = gs
                .properties
                .get_all_edge_properties(ei.eid)
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect();

            // Create new edge with same metadata but updated endpoints
            let new_edge_id = gs.adjacency.allocate_edge_id();
            let new_meta = weav_graph::adjacency::EdgeMeta {
                source: new_src,
                target: new_tgt,
                label: ei.label,
                temporal: ei.temporal,
                provenance: ei.provenance.clone(),
                weight: ei.weight,
                token_cost: ei.token_cost,
            };

            // WAL entry for the new edge
            let props_json = serde_json::to_string(
                &edge_props
                    .iter()
                    .map(|(k, v)| (k.as_str(), v))
                    .collect::<std::collections::HashMap<_, _>>(),
            )
            .unwrap_or_default();
            let label_str = gs
                .interner
                .resolve_label(ei.label)
                .unwrap_or("unknown")
                .to_string();
            self.append_wal(WalOperation::EdgeAdd {
                graph_id,
                edge_id: new_edge_id,
                source: new_src,
                target: new_tgt,
                label: label_str,
                weight: ei.weight,
                properties_json: props_json,
            })?;

            gs.adjacency
                .add_edge_with_id(new_src, new_tgt, ei.label, new_meta, new_edge_id)?;

            for (k, v) in &edge_props {
                gs.properties.set_edge_property(new_edge_id, k, v.clone());
            }
        }

        // 4. If source has a vector embedding, copy to target if target lacks one
        if let Some(src_vec) = gs.vector_index.get_vector(cmd.source_id) {
            let src_vec = src_vec.to_vec();
            if gs.vector_index.get_vector(cmd.target_id).is_none() {
                let _ = gs.vector_index.insert(cmd.target_id, &src_vec);
            }
        }

        // 5. Delete source node (WAL + in-memory)
        self.append_wal(WalOperation::NodeDelete {
            graph_id,
            node_id: cmd.source_id,
        })?;
        gs.adjacency.remove_node(cmd.source_id)?;
        gs.properties.remove_all_node_properties(cmd.source_id);
        let _ = gs.vector_index.remove(cmd.source_id);

        // Update metrics
        crate::metrics::NODES_TOTAL
            .with_label_values(&[&cmd.graph])
            .set(gs.adjacency.node_count() as i64);
        crate::metrics::EDGES_TOTAL
            .with_label_values(&[&cmd.graph])
            .set(gs.adjacency.edge_count() as i64);

        // Emit CDC events: source was deleted, target was updated with merged properties
        let target_props: Vec<(CompactString, Value)> = gs
            .properties
            .get_all_node_properties(cmd.target_id)
            .into_iter()
            .filter(|(k, _)| !k.starts_with('_'))
            .map(|(k, v)| (CompactString::from(k), v.clone()))
            .collect();

        self.emit_event(&cmd.graph, EventKind::NodeDeleted {
            node_id: cmd.source_id,
        });
        self.emit_event(&cmd.graph, EventKind::NodeUpdated {
            node_id: cmd.target_id,
            properties: target_props,
        });

        Ok(CommandResponse::Integer(cmd.target_id))
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

        // Re-index text content for full-text search
        let text_content: String = cmd.properties.iter()
            .filter_map(|(_, v)| v.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        if !text_content.is_empty() {
            gs.text_index.index_node(cmd.node_id, &text_content);
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

        self.emit_event(&cmd.graph, EventKind::NodeUpdated {
            node_id: cmd.node_id,
            properties: cmd.properties.iter()
                .map(|(k, v)| (CompactString::from(k.as_str()), v.clone()))
                .collect(),
        });

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

            self.emit_event(&cmd.graph, EventKind::NodeCreated {
                node_id,
                label: CompactString::from(&node_cmd.label),
                properties: node_cmd.properties.iter()
                    .map(|(k, v)| (CompactString::from(k.as_str()), v.clone()))
                    .collect(),
            });

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

            // Serialize edge properties for WAL persistence
            let props_json = serde_json::to_string(
                &edge_cmd.properties.iter().map(|(k, v)| (k.as_str(), v)).collect::<std::collections::HashMap<_, _>>()
            ).unwrap_or_default();

            // Write-ahead: WAL entry before in-memory mutation
            self.append_wal(WalOperation::EdgeAdd {
                graph_id,
                edge_id,
                source: edge_cmd.source,
                target: edge_cmd.target,
                label: edge_cmd.label.clone(),
                weight: edge_cmd.weight,
                properties_json: props_json,
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

            // Store edge properties
            for (k, v) in &edge_cmd.properties {
                gs.properties.set_edge_property(edge_id, k, v.clone());
            }

            self.emit_event(&cmd.graph, EventKind::EdgeCreated {
                edge_id,
                source: edge_cmd.source,
                target: edge_cmd.target,
                label: CompactString::from(&edge_cmd.label),
                weight: edge_cmd.weight,
            });

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

        // Enforce max_edges capacity if configured
        if let Some(max) = gs.config.max_edges {
            if gs.adjacency.edge_count() >= max {
                return Err(WeavError::CapacityExceeded(format!(
                    "graph '{}' reached max_edges limit ({})", cmd.graph, max
                )));
            }
        }

        // ── Schema validation ────────────────────────────────────────────
        gs.config.schema.validate_edge_properties(&cmd.label, &cmd.properties)?;

        // Uniqueness constraint enforcement for edges
        for constraint in gs.config.schema.edge_schemas
            .get(cmd.label.as_str())
            .map(|s| s.constraints.as_slice())
            .unwrap_or_default()
        {
            if let weav_core::schema::SchemaConstraint::PropertyUnique { property } = constraint {
                if let Some((_, prop_val)) = cmd.properties.iter().find(|(k, _)| k == property.as_str()) {
                    if *prop_val != Value::Null {
                        let search_val = prop_val.clone();
                        let existing = gs.properties.edges_where(property.as_str(), &|v| *v == search_val);
                        // Filter to only edges with the same label
                        let duplicates: Vec<_> = existing.into_iter().filter(|&eid| {
                            gs.adjacency.get_edge(eid)
                                .map(|meta| meta.label == label_id)
                                .unwrap_or(false)
                        }).collect();
                        if !duplicates.is_empty() {
                            return Err(WeavError::Conflict(format!(
                                "unique constraint violation: property '{}' value already exists on edge label '{}'",
                                property, cmd.label
                            )));
                        }
                    }
                }
            }
        }

        // Pre-allocate edge ID for write-ahead logging
        let edge_id = gs.adjacency.allocate_edge_id();
        let graph_id = gs.graph_id;

        // Serialize edge properties for WAL persistence
        let props_json = serde_json::to_string(
            &cmd.properties.iter().map(|(k, v)| (k.as_str(), v)).collect::<std::collections::HashMap<_, _>>()
        ).unwrap_or_default();

        // Write-ahead: WAL entry before in-memory mutation
        self.append_wal(WalOperation::EdgeAdd {
            graph_id,
            edge_id,
            source: cmd.source,
            target: cmd.target,
            label: cmd.label.clone(),
            weight: cmd.weight,
            properties_json: props_json,
        })?;

        // Apply in-memory mutation — explicit TTL overrides graph default
        let effective_ttl = cmd.ttl_ms.or(gs.config.default_ttl_ms);
        let temporal = if let Some(ttl_ms) = effective_ttl {
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

        // Store edge properties
        for (k, v) in &cmd.properties {
            gs.properties.set_edge_property(edge_id, k, v.clone());
        }

        // Update metrics
        crate::metrics::EDGES_TOTAL
            .with_label_values(&[&cmd.graph])
            .set(gs.adjacency.edge_count() as i64);

        self.emit_event(&cmd.graph, EventKind::EdgeCreated {
            edge_id,
            source: cmd.source,
            target: cmd.target,
            label: CompactString::from(&cmd.label),
            weight: cmd.weight,
        });

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

        self.emit_event(&cmd.graph, EventKind::EdgeInvalidated {
            edge_id: cmd.edge_id,
        });

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

        self.emit_event(&cmd.graph, EventKind::EdgeDeleted {
            edge_id: cmd.edge_id,
        });

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

    // ── Schema ───────────────────────────────────────────────────────────

    fn handle_schema_set(
        &self,
        cmd: weav_query::parser::SchemaSetCmd,
    ) -> WeavResult<CommandResponse> {
        use compact_str::CompactString;
        use weav_core::schema::SchemaConstraint;
        use weav_core::types::ValueType;

        let graph_arc = self.get_graph(&cmd.graph)?;
        let mut gs = graph_arc.write();

        let constraint = match cmd.constraint_type.as_str() {
            "type" => {
                let vt = match cmd.value_type.as_deref() {
                    Some("string") => ValueType::String,
                    Some("int") => ValueType::Int,
                    Some("float") => ValueType::Float,
                    Some("bool") => ValueType::Bool,
                    Some("bytes") => ValueType::Bytes,
                    Some("vector") => ValueType::Vector,
                    Some("list") => ValueType::List,
                    Some("map") => ValueType::Map,
                    Some("timestamp") => ValueType::Timestamp,
                    Some("null") => ValueType::Null,
                    Some(other) => {
                        return Err(WeavError::InvalidConfig(format!(
                            "unknown value type: '{}'",
                            other
                        )));
                    }
                    None => {
                        return Err(WeavError::InvalidConfig(
                            "type constraint requires a value_type".into(),
                        ));
                    }
                };
                SchemaConstraint::PropertyType {
                    property: CompactString::new(&cmd.property),
                    expected_type: vt,
                }
            }
            "required" => SchemaConstraint::PropertyRequired {
                property: CompactString::new(&cmd.property),
            },
            "unique" => SchemaConstraint::PropertyUnique {
                property: CompactString::new(&cmd.property),
            },
            other => {
                return Err(WeavError::InvalidConfig(format!(
                    "unknown constraint type: '{}'",
                    other
                )));
            }
        };

        match cmd.target.as_str() {
            "node" => gs.config.schema.add_node_constraint(&cmd.label, constraint),
            "edge" => gs.config.schema.add_edge_constraint(&cmd.label, constraint),
            other => {
                return Err(WeavError::InvalidConfig(format!(
                    "schema target must be 'node' or 'edge', got '{}'",
                    other
                )));
            }
        }

        Ok(CommandResponse::Ok)
    }

    fn handle_schema_get(
        &self,
        cmd: weav_query::parser::SchemaGetCmd,
    ) -> WeavResult<CommandResponse> {
        let graph_arc = self.get_graph(&cmd.graph)?;
        let gs = graph_arc.read();

        let schema_json = serde_json::to_string_pretty(&gs.config.schema)
            .unwrap_or_else(|_| "{}".to_string());
        Ok(CommandResponse::Text(schema_json))
    }

    // ── Snapshot ──────────────────────────────────────────────────────────

    fn handle_snapshot(&self) -> WeavResult<CommandResponse> {
        let snap_start = std::time::Instant::now();
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

                // Serialize edge properties for snapshot persistence
                let all_edge_props = gs.properties.get_all_edge_properties(edge_id);
                let edge_props_json = {
                    let props_map: std::collections::HashMap<&str, &Value> = all_edge_props
                        .into_iter()
                        .collect();
                    serde_json::to_string(&props_map).unwrap_or_default()
                };

                edge_snapshots.push(EdgeSnapshot {
                    edge_id,
                    source: meta.source,
                    target: meta.target,
                    label,
                    weight: meta.weight,
                    valid_from: meta.temporal.valid_from,
                    valid_until: meta.temporal.valid_until,
                    properties_json: edge_props_json,
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

        let saved_path = snapshot_engine.save_snapshot(&full_snapshot)
            .map_err(|e| WeavError::Internal(format!("snapshot save failed: {e}")))?;

        crate::metrics::SNAPSHOT_DURATION.observe(snap_start.elapsed().as_secs_f64());
        if let Ok(meta) = std::fs::metadata(&saved_path) {
            crate::metrics::SNAPSHOT_SIZE_BYTES.set(meta.len() as i64);
        }

        Ok(CommandResponse::Ok)
    }

    // ── Context query ────────────────────────────────────────────────────

    fn handle_context(
        &self,
        query: weav_query::parser::ContextQuery,
    ) -> WeavResult<CommandResponse> {
        let graph_name = query.graph.clone();
        let graph_arc = self.get_graph(&graph_name)?;
        let gs = graph_arc.read();

        let ctx_start = std::time::Instant::now();
        let result = executor::execute_context_query(
            &query,
            &gs.adjacency,
            &gs.properties,
            &gs.vector_index,
            &self.token_counter,
            &gs.interner,
        )?;

        // Vector search timing (context query always involves vector or graph traversal).
        crate::metrics::VECTOR_SEARCH_DURATION
            .with_label_values(&[&graph_name])
            .observe(ctx_start.elapsed().as_secs_f64());
        crate::metrics::VECTOR_INDEX_SIZE
            .with_label_values(&[&graph_name])
            .set(gs.vector_index.len() as i64);

        // Token budget metrics.
        let strategy = query.budget.as_ref().map_or("auto", |b| {
            match &b.allocation {
                weav_core::types::TokenAllocation::Auto => "auto",
                weav_core::types::TokenAllocation::Proportional { .. } => "proportional",
                weav_core::types::TokenAllocation::Priority(_) => "priority",
                weav_core::types::TokenAllocation::DiversityAware { .. } => "mmr",
                weav_core::types::TokenAllocation::SubmodularFacilityLocation { .. } => "submodular",
            }
        });
        crate::metrics::TOKEN_BUDGET_USAGE
            .with_label_values(&[&graph_name, strategy])
            .observe(result.budget_used as f64);
        if result.budget_used >= 1.0 {
            crate::metrics::TOKEN_BUDGET_OVERFLOW
                .with_label_values(&[&graph_name])
                .inc();
        }

        Ok(CommandResponse::Context(result))
    }

    // ── Ingest handler ────────────────────────────────────────────────────

    async fn handle_ingest(
        &self,
        cmd: weav_query::parser::IngestCmd,
    ) -> WeavResult<CommandResponse> {
        let ingest_start = std::time::Instant::now();
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

        let format_label = format!("{:?}", format);

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

        crate::metrics::INGEST_DURATION
            .with_label_values(&[&cmd.graph, &format_label])
            .observe(ingest_start.elapsed().as_secs_f64());
        crate::metrics::INGEST_DOCUMENTS_TOTAL
            .with_label_values(&[&cmd.graph])
            .inc();

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

        let result = match username {
            Some(ref u) => store.authenticate(u, &password),
            None => store.authenticate_default(&password),
        };

        match result {
            Ok(identity) => {
                crate::metrics::AUTH_ATTEMPTS_TOTAL
                    .with_label_values(&["success"])
                    .inc();
                Ok(CommandResponse::Text(format!("OK (user: {})", identity.username)))
            }
            Err(e) => {
                crate::metrics::AUTH_ATTEMPTS_TOTAL
                    .with_label_values(&["failure"])
                    .inc();
                Err(e)
            }
        }
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
                    .map_err(WeavError::Internal)?,
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
                    permission: weav_auth::identity::GraphPermission::parse(perm),
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
            CommandResponse::Text(t) => {
                assert!(t.contains("graphs="), "stats should contain graphs count: {t}");
                assert!(t.contains("total_nodes="), "stats should contain total_nodes: {t}");
                assert!(t.contains("engine=weav-server"), "stats should contain engine name: {t}");
            }
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn test_stats_with_graph() {
        let engine = make_engine();
        create_test_graph(&engine, "sg");

        // Add a node so we have label distribution
        let cmd = parser::parse_command(
            r#"NODE ADD TO "sg" LABEL "person" PROPERTIES {"name": "Alice"}"#,
        ).unwrap();
        engine.execute_command(cmd, None).unwrap();

        let cmd = parser::parse_command("STATS \"sg\"").unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::Text(t) => {
                assert!(t.contains("graph=sg"), "should contain graph name: {t}");
                assert!(t.contains("nodes="), "should contain nodes: {t}");
                assert!(t.contains("labels={"), "should contain label distribution: {t}");
                assert!(t.contains("avg_degree="), "should contain avg_degree: {t}");
                assert!(t.contains("person:"), "should list person label: {t}");
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

    #[test]
    fn test_edge_properties_survive_snapshot_roundtrip() {
        use weav_persist::recovery::RecoveryResult;

        // Step 1: Create engine with persistence enabled, add edge with properties.
        let tmp_dir = std::env::temp_dir().join(format!(
            "weav_edge_props_snap_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp_dir).unwrap();

        let mut config = WeavConfig::default();
        config.persistence.enabled = true;
        config.persistence.data_dir = tmp_dir.clone();

        let engine = Engine::new(config.clone());
        create_test_graph(&engine, "ep_graph");

        // Add two nodes.
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "ep_graph".to_string(),
            label: "person".to_string(),
            properties: vec![("name".to_string(), Value::String(CompactString::from("Alice")))],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let n1 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "ep_graph".to_string(),
            label: "person".to_string(),
            properties: vec![("name".to_string(), Value::String(CompactString::from("Bob")))],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let n2 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add edge with properties.
        let cmd = Command::EdgeAdd(parser::EdgeAddCmd {
            graph: "ep_graph".to_string(),
            source: n1,
            target: n2,
            label: "KNOWS".to_string(),
            weight: 0.9,
            properties: vec![
                ("since".to_string(), Value::String(CompactString::from("2020"))),
                ("strength".to_string(), Value::Float(0.85)),
            ],
            ttl_ms: None,
        });
        let edge_id = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Verify in-memory properties are set.
        {
            let graph_arc = engine.get_graph("ep_graph").unwrap();
            let gs = graph_arc.read();
            let since = gs.properties.get_edge_property(edge_id, "since");
            assert!(since.is_some(), "edge property 'since' should be set");
            assert_eq!(since.unwrap().as_str(), Some("2020"));
            let strength = gs.properties.get_edge_property(edge_id, "strength");
            assert!(strength.is_some(), "edge property 'strength' should be set");
        }

        // Take a snapshot.
        let cmd = parser::parse_command("SNAPSHOT").unwrap();
        engine.execute_command(cmd, None).unwrap();

        // Step 2: Create a new engine and recover from the snapshot.
        let engine2 = Engine::new(config.clone());

        // Load the snapshot file manually.
        let snap_engine = weav_persist::snapshot::SnapshotEngine::new(tmp_dir.clone());
        let latest = snap_engine.latest_snapshot().unwrap().expect("snapshot should exist");
        let snapshot = snap_engine.load_snapshot(&latest).unwrap();

        let recovery = RecoveryResult {
            snapshots_loaded: 1,
            wal_entries_replayed: 0,
            graphs_recovered: 1,
            errors: vec![],
            snapshot: Some(snapshot),
            wal_entries: vec![],
        };
        engine2.recover(recovery).unwrap();

        // Step 3: Verify edge properties survived the roundtrip.
        let graph_arc = engine2.get_graph("ep_graph").unwrap();
        let gs = graph_arc.read();

        // Edge should exist with original ID.
        assert_eq!(gs.adjacency.edge_count(), 1, "edge should be restored");
        let recovered_edge = gs.adjacency.get_edge(edge_id);
        assert!(recovered_edge.is_some(), "edge should have same ID after recovery");

        // Edge properties should be restored.
        let since = gs.properties.get_edge_property(edge_id, "since");
        assert!(since.is_some(), "edge property 'since' should survive snapshot roundtrip");
        assert_eq!(since.unwrap().as_str(), Some("2020"));

        let strength = gs.properties.get_edge_property(edge_id, "strength");
        assert!(strength.is_some(), "edge property 'strength' should survive snapshot roundtrip");

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_edge_id_consistency_after_recovery() {
        use weav_persist::recovery::RecoveryResult;

        // Step 1: Create engine, add nodes + edges, capture edge IDs.
        let tmp_dir = std::env::temp_dir().join(format!(
            "weav_edge_id_recovery_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp_dir).unwrap();

        let mut config = WeavConfig::default();
        config.persistence.enabled = true;
        config.persistence.data_dir = tmp_dir.clone();

        let engine = Engine::new(config.clone());
        create_test_graph(&engine, "eid_graph");

        // Add three nodes.
        let mut node_ids = Vec::new();
        for name in &["Alice", "Bob", "Charlie"] {
            let cmd = Command::NodeAdd(parser::NodeAddCmd {
                graph: "eid_graph".to_string(),
                label: "person".to_string(),
                properties: vec![("name".to_string(), Value::String(CompactString::from(*name)))],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            });
            let nid = match engine.execute_command(cmd, None).unwrap() {
                CommandResponse::Integer(id) => id,
                _ => panic!("expected Integer"),
            };
            node_ids.push(nid);
        }

        // Add edges and capture their IDs.
        let mut original_edge_ids = Vec::new();
        let edges = vec![
            (node_ids[0], node_ids[1], "KNOWS"),
            (node_ids[1], node_ids[2], "WORKS_WITH"),
            (node_ids[0], node_ids[2], "FRIENDS"),
        ];
        for (src, tgt, label) in &edges {
            let cmd = Command::EdgeAdd(parser::EdgeAddCmd {
                graph: "eid_graph".to_string(),
                source: *src,
                target: *tgt,
                label: label.to_string(),
                weight: 1.0,
                properties: vec![],
                ttl_ms: None,
            });
            let eid = match engine.execute_command(cmd, None).unwrap() {
                CommandResponse::Integer(id) => id,
                _ => panic!("expected Integer"),
            };
            original_edge_ids.push(eid);
        }

        // Take a snapshot.
        let cmd = parser::parse_command("SNAPSHOT").unwrap();
        engine.execute_command(cmd, None).unwrap();

        // Step 2: Recover from snapshot into a new engine.
        let engine2 = Engine::new(config.clone());

        let snap_engine = weav_persist::snapshot::SnapshotEngine::new(tmp_dir.clone());
        let latest = snap_engine.latest_snapshot().unwrap().expect("snapshot should exist");
        let snapshot = snap_engine.load_snapshot(&latest).unwrap();

        let recovery = RecoveryResult {
            snapshots_loaded: 1,
            wal_entries_replayed: 0,
            graphs_recovered: 1,
            errors: vec![],
            snapshot: Some(snapshot),
            wal_entries: vec![],
        };
        engine2.recover(recovery).unwrap();

        // Step 3: Verify all edge IDs match.
        let graph_arc = engine2.get_graph("eid_graph").unwrap();
        let gs = graph_arc.read();

        for eid in &original_edge_ids {
            let meta = gs.adjacency.get_edge(*eid);
            assert!(
                meta.is_some(),
                "edge ID {} should be preserved after snapshot recovery",
                eid,
            );
        }

        // Step 4: Verify that adding a new edge after recovery gets a non-conflicting ID.
        drop(gs);
        drop(graph_arc);

        let cmd = Command::EdgeAdd(parser::EdgeAddCmd {
            graph: "eid_graph".to_string(),
            source: node_ids[0],
            target: node_ids[1],
            label: "NEW_EDGE".to_string(),
            weight: 0.5,
            properties: vec![],
            ttl_ms: None,
        });
        let new_eid = match engine2.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // New edge ID should not collide with any recovered edge ID.
        for eid in &original_edge_ids {
            assert_ne!(
                new_eid, *eid,
                "new edge ID should not collide with recovered edge IDs",
            );
        }
        assert!(
            new_eid > *original_edge_ids.iter().max().unwrap(),
            "new edge ID should be greater than all recovered edge IDs",
        );

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_connection_limit_enforcement() {
        let mut config = WeavConfig::default();
        config.server.max_connections = 3;
        let engine = Engine::new(config);

        // Acquire 3 connections (the limit)
        engine.try_acquire_connection().unwrap();
        engine.try_acquire_connection().unwrap();
        engine.try_acquire_connection().unwrap();
        assert_eq!(engine.active_connection_count(), 3);

        // 4th should fail
        let result = engine.try_acquire_connection();
        assert!(result.is_err(), "should reject when at max_connections");
        assert_eq!(engine.active_connection_count(), 3); // unchanged

        // Release one, then 4th should succeed
        engine.release_connection();
        assert_eq!(engine.active_connection_count(), 2);
        engine.try_acquire_connection().unwrap();
        assert_eq!(engine.active_connection_count(), 3);
    }

    #[test]
    fn test_connection_release_underflow_safe() {
        let engine = make_engine();
        // Release without acquire should not underflow (saturating_sub)
        engine.release_connection();
        assert_eq!(engine.active_connection_count(), 0);
    }

    #[test]
    fn test_value_types_survive_snapshot_roundtrip() {
        use weav_persist::recovery::RecoveryResult;

        let tmp_dir = std::env::temp_dir().join(format!(
            "weav_value_types_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&tmp_dir).unwrap();

        let mut config = WeavConfig::default();
        config.persistence.enabled = true;
        config.persistence.data_dir = tmp_dir.clone();

        let engine = Engine::new(config.clone());
        create_test_graph(&engine, "vt_graph");

        // Add node with Timestamp and various Value types
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "vt_graph".to_string(),
            label: "entity".to_string(),
            properties: vec![
                ("name".to_string(), Value::String(CompactString::from("test"))),
                ("count".to_string(), Value::Int(42)),
                ("score".to_string(), Value::Float(3.14)),
                ("active".to_string(), Value::Bool(true)),
                ("created".to_string(), Value::Timestamp(1700000000000)),
            ],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        engine.execute_command(cmd, None).unwrap();

        // Take snapshot
        let cmd = parser::parse_command("SNAPSHOT").unwrap();
        engine.execute_command(cmd, None).unwrap();

        // Recover into new engine
        let engine2 = Engine::new(config.clone());
        let snap_engine = weav_persist::snapshot::SnapshotEngine::new(tmp_dir.clone());
        let latest = snap_engine.latest_snapshot().unwrap().expect("snapshot should exist");
        let snapshot = snap_engine.load_snapshot(&latest).unwrap();

        let recovery = RecoveryResult {
            snapshots_loaded: 1,
            wal_entries_replayed: 0,
            graphs_recovered: 1,
            errors: vec![],
            snapshot: Some(snapshot),
            wal_entries: vec![],
        };
        engine2.recover(recovery).unwrap();

        // Verify all value types survived
        let graph_arc = engine2.get_graph("vt_graph").unwrap();
        let gs = graph_arc.read();
        let node_ids = gs.adjacency.all_node_ids();
        assert!(!node_ids.is_empty());
        let nid = node_ids[0];

        // String
        let name = gs.properties.get_node_property(nid, "name");
        assert_eq!(name.unwrap().as_str(), Some("test"), "String should survive");

        // Int
        let count = gs.properties.get_node_property(nid, "count");
        assert_eq!(count.unwrap().as_int(), Some(42), "Int should survive");

        // Float
        let score = gs.properties.get_node_property(nid, "score");
        assert!(score.is_some(), "Float should survive");

        // Bool
        let active = gs.properties.get_node_property(nid, "active");
        assert_eq!(active.unwrap().as_bool(), Some(true), "Bool should survive");

        // Timestamp — this was the bug! Previously lost during recovery
        let created = gs.properties.get_node_property(nid, "created");
        assert!(created.is_some(), "Timestamp should survive snapshot roundtrip");
        match created.unwrap() {
            Value::Timestamp(ts) => assert_eq!(*ts, 1700000000000),
            other => panic!("expected Timestamp, got {:?}", other),
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_graph_level_default_ttl() {
        let engine = make_engine();

        // Create graph with default TTL of 1ms
        let mut gc = weav_core::config::GraphConfig::default();
        gc.default_ttl_ms = Some(1);
        let cmd = Command::GraphCreate(parser::GraphCreateCmd {
            name: "ttl_default_g".to_string(),
            config: Some(gc),
        });
        engine.execute_command(cmd, None).unwrap();

        // Add node WITHOUT explicit TTL — should inherit graph default
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "ttl_default_g".to_string(),
            label: "auto_ttl".to_string(),
            properties: vec![("x".to_string(), Value::Int(1))],
            embedding: None,
            entity_key: None,
            ttl_ms: None, // no explicit TTL
        });
        let node_id = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Node should have _ttl_expires_at set from graph default
        let graph_arc = engine.get_graph("ttl_default_g").unwrap();
        let gs = graph_arc.read();
        let ttl_prop = gs.properties.get_node_property(node_id, "_ttl_expires_at");
        assert!(ttl_prop.is_some(), "node should have TTL from graph default");

        // Sweep should remove it (1ms TTL already expired)
        drop(gs);
        std::thread::sleep(std::time::Duration::from_millis(5));
        let expired = engine.sweep_ttl();
        assert!(expired >= 1, "node with inherited TTL should be expired");
    }

    #[test]
    fn test_graph_info_includes_new_fields() {
        let engine = make_engine();
        create_test_graph(&engine, "info_g");

        // Add some data
        let cmd = parser::parse_command(
            r#"NODE ADD TO "info_g" LABEL "person" PROPERTIES {"name": "Alice"}"#,
        ).unwrap();
        engine.execute_command(cmd, None).unwrap();

        let cmd = Command::GraphInfo("info_g".to_string());
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::GraphInfo(info) => {
                assert_eq!(info.name, "info_g");
                assert!(info.node_count >= 1);
                assert_eq!(info.vector_count, 0); // no embeddings
                assert!(info.label_count >= 1); // "person"
                assert_eq!(info.default_ttl_ms, None);
            }
            _ => panic!("expected GraphInfo"),
        }
    }

    #[test]
    fn test_max_nodes_enforcement() {
        let engine = make_engine();
        let mut gc = weav_core::config::GraphConfig::default();
        gc.max_nodes = Some(2);
        let cmd = Command::GraphCreate(parser::GraphCreateCmd {
            name: "limited".to_string(),
            config: Some(gc),
        });
        engine.execute_command(cmd, None).unwrap();

        // First two nodes should succeed
        for i in 0..2 {
            let cmd = Command::NodeAdd(parser::NodeAddCmd {
                graph: "limited".to_string(),
                label: "x".to_string(),
                properties: vec![("i".to_string(), Value::Int(i))],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            });
            engine.execute_command(cmd, None).unwrap();
        }

        // Third should fail
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "limited".to_string(),
            label: "x".to_string(),
            properties: vec![],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let result = engine.execute_command(cmd, None);
        assert!(result.is_err(), "should reject when max_nodes reached");
        match result.unwrap_err() {
            WeavError::CapacityExceeded(_) => {}
            other => panic!("expected CapacityExceeded, got: {other}"),
        }
    }

    #[test]
    fn test_max_edges_enforcement() {
        let engine = make_engine();
        let mut gc = weav_core::config::GraphConfig::default();
        gc.max_edges = Some(1);
        let cmd = Command::GraphCreate(parser::GraphCreateCmd {
            name: "edge_limited".to_string(),
            config: Some(gc),
        });
        engine.execute_command(cmd, None).unwrap();

        // Add two nodes
        let n1 = match engine.execute_command(Command::NodeAdd(parser::NodeAddCmd {
            graph: "edge_limited".to_string(),
            label: "a".to_string(),
            properties: vec![],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        }), None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };
        let n2 = match engine.execute_command(Command::NodeAdd(parser::NodeAddCmd {
            graph: "edge_limited".to_string(),
            label: "b".to_string(),
            properties: vec![],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        }), None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // First edge succeeds
        let cmd = Command::EdgeAdd(parser::EdgeAddCmd {
            graph: "edge_limited".to_string(),
            source: n1,
            target: n2,
            label: "link".to_string(),
            weight: 1.0,
            properties: vec![],
            ttl_ms: None,
        });
        engine.execute_command(cmd, None).unwrap();

        // Second edge should fail
        let cmd = Command::EdgeAdd(parser::EdgeAddCmd {
            graph: "edge_limited".to_string(),
            source: n2,
            target: n1,
            label: "link2".to_string(),
            weight: 1.0,
            properties: vec![],
            ttl_ms: None,
        });
        let result = engine.execute_command(cmd, None);
        assert!(result.is_err(), "should reject when max_edges reached");
    }

    #[test]
    fn test_search_command() {
        let engine = make_engine();
        create_test_graph(&engine, "sg");

        // Add nodes with searchable properties
        let cmd = parser::parse_command(
            r#"NODE ADD TO "sg" LABEL "person" PROPERTIES {"name": "Alice", "role": "engineer"}"#,
        ).unwrap();
        engine.execute_command(cmd, None).unwrap();

        let cmd = parser::parse_command(
            r#"NODE ADD TO "sg" LABEL "person" PROPERTIES {"name": "Bob", "role": "designer"}"#,
        ).unwrap();
        engine.execute_command(cmd, None).unwrap();

        let cmd = parser::parse_command(
            r#"NODE ADD TO "sg" LABEL "person" PROPERTIES {"name": "Charlie", "role": "engineer"}"#,
        ).unwrap();
        engine.execute_command(cmd, None).unwrap();

        // Search by role = "engineer" — should find 2
        let cmd = parser::parse_command(
            r#"SEARCH "sg" WHERE role = "engineer""#,
        ).unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::StringList(results) => {
                assert_eq!(results.len(), 2);
                for r in &results {
                    assert!(r.ends_with(":person"), "expected label suffix, got {}", r);
                }
            }
            _ => panic!("expected StringList"),
        }
    }

    #[test]
    fn test_neighbors_command() {
        let engine = make_engine();
        create_test_graph(&engine, "ng");

        let cmd = parser::parse_command(
            r#"NODE ADD TO "ng" LABEL "person" PROPERTIES {"name": "A"}"#,
        ).unwrap();
        let n1 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(
            r#"NODE ADD TO "ng" LABEL "person" PROPERTIES {"name": "B"}"#,
        ).unwrap();
        let n2 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(
            r#"NODE ADD TO "ng" LABEL "person" PROPERTIES {"name": "C"}"#,
        ).unwrap();
        let n3 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add edges: n1->n2, n1->n3
        let cmd = parser::parse_command(&format!(
            r#"EDGE ADD TO "ng" FROM {n1} TO {n2} LABEL "knows""#,
        )).unwrap();
        engine.execute_command(cmd, None).unwrap();

        let cmd = parser::parse_command(&format!(
            r#"EDGE ADD TO "ng" FROM {n1} TO {n3} LABEL "likes""#,
        )).unwrap();
        engine.execute_command(cmd, None).unwrap();

        // Get all neighbors of n1 (default BOTH direction)
        let cmd = parser::parse_command(&format!(
            r#"NEIGHBORS "ng" {n1}"#,
        )).unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::StringList(results) => {
                assert_eq!(results.len(), 2, "n1 should have 2 neighbors");
                for r in &results {
                    let parts: Vec<&str> = r.split(':').collect();
                    assert_eq!(parts.len(), 4, "format should be nid:eid:DIR:label, got {}", r);
                }
            }
            _ => panic!("expected StringList"),
        }
    }

    #[test]
    fn test_neighbors_with_direction() {
        let engine = make_engine();
        create_test_graph(&engine, "nd");

        let cmd = parser::parse_command(
            r#"NODE ADD TO "nd" LABEL "x" PROPERTIES {"v": "1"}"#,
        ).unwrap();
        let n1 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = parser::parse_command(
            r#"NODE ADD TO "nd" LABEL "x" PROPERTIES {"v": "2"}"#,
        ).unwrap();
        let n2 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Edge from n1 -> n2
        let cmd = parser::parse_command(&format!(
            r#"EDGE ADD TO "nd" FROM {n1} TO {n2} LABEL "link""#,
        )).unwrap();
        engine.execute_command(cmd, None).unwrap();

        // DIRECTION OUT from n1: should see n2
        let cmd = parser::parse_command(&format!(
            r#"NEIGHBORS "nd" {n1} DIRECTION OUT"#,
        )).unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::StringList(results) => {
                assert_eq!(results.len(), 1);
                assert!(results[0].contains("OUT"), "direction should be OUT");
            }
            _ => panic!("expected StringList"),
        }

        // DIRECTION OUT from n2: should be empty
        let cmd = parser::parse_command(&format!(
            r#"NEIGHBORS "nd" {n2} DIRECTION OUT"#,
        )).unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::StringList(results) => {
                assert_eq!(results.len(), 0, "n2 has no outgoing edges");
            }
            _ => panic!("expected StringList"),
        }
    }

    #[test]
    fn test_graph_default_ttl_edge() {
        let engine = make_engine();

        // Create graph with default TTL of 1ms
        let mut gc = weav_core::config::GraphConfig::default();
        gc.default_ttl_ms = Some(1);
        let cmd = Command::GraphCreate(parser::GraphCreateCmd {
            name: "ttl_edge_g".to_string(),
            config: Some(gc),
        });
        engine.execute_command(cmd, None).unwrap();

        // Add two nodes
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "ttl_edge_g".to_string(),
            label: "a".to_string(),
            properties: vec![("x".to_string(), Value::Int(1))],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let n1 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "ttl_edge_g".to_string(),
            label: "b".to_string(),
            properties: vec![("x".to_string(), Value::Int(2))],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let n2 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Add edge WITHOUT explicit TTL — should inherit graph default (1ms)
        let cmd = Command::EdgeAdd(parser::EdgeAddCmd {
            graph: "ttl_edge_g".to_string(),
            source: n1,
            target: n2,
            label: "ephemeral".to_string(),
            weight: 1.0,
            properties: vec![],
            ttl_ms: None,
        });
        engine.execute_command(cmd, None).unwrap();

        // Edge should inherit the 1ms TTL via valid_until
        std::thread::sleep(std::time::Duration::from_millis(5));
        let expired = engine.sweep_ttl();
        // Both nodes and the edge should expire (graph default applies to all)
        assert!(expired >= 1, "edge with inherited TTL should be expired, got {}", expired);
    }

    #[test]
    fn test_search_no_results() {
        let engine = make_engine();
        create_test_graph(&engine, "empty_s");

        let cmd = parser::parse_command(
            r#"NODE ADD TO "empty_s" LABEL "item" PROPERTIES {"color": "red"}"#,
        ).unwrap();
        engine.execute_command(cmd, None).unwrap();

        // Search for a value that doesn't match
        let cmd = parser::parse_command(
            r#"SEARCH "empty_s" WHERE color = "blue""#,
        ).unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::StringList(results) => {
                assert!(results.is_empty(), "no nodes should match color=blue");
            }
            _ => panic!("expected StringList"),
        }
    }

    #[test]
    fn test_graph_info_enhanced_fields() {
        let engine = make_engine();
        create_test_graph(&engine, "enh_info");

        // Add nodes with distinct labels
        let cmd = parser::parse_command(
            r#"NODE ADD TO "enh_info" LABEL "person" PROPERTIES {"name": "X"}"#,
        ).unwrap();
        engine.execute_command(cmd, None).unwrap();

        let cmd = parser::parse_command(
            r#"NODE ADD TO "enh_info" LABEL "company" PROPERTIES {"name": "Y"}"#,
        ).unwrap();
        engine.execute_command(cmd, None).unwrap();

        let cmd = Command::GraphInfo("enh_info".to_string());
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::GraphInfo(info) => {
                assert_eq!(info.node_count, 2);
                assert_eq!(info.vector_count, 0);
                assert!(info.label_count >= 2,
                    "should have at least 2 labels, got {}", info.label_count);
            }
            _ => panic!("expected GraphInfo"),
        }
    }

    #[test]
    fn test_replaying_flag_suppresses_wal() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "weav_replay_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&tmp_dir);

        let mut config = WeavConfig::default();
        config.persistence.enabled = true;
        config.persistence.data_dir = tmp_dir.clone();
        let engine = Engine::new(config);

        // Normal operation: create a graph (writes to WAL)
        create_test_graph(&engine, "pre_replay");

        let seq_before = engine.wal.as_ref()
            .expect("WAL should be present").lock().sequence_number();
        assert!(seq_before > 0, "should have WAL entries from graph create");

        // Set replaying flag
        engine.replaying.store(true, std::sync::atomic::Ordering::Relaxed);

        // Create another graph — this should NOT write to WAL
        let cmd = parser::parse_command("GRAPH CREATE \"during_replay\"").unwrap();
        engine.execute_command(cmd, None).unwrap();

        let seq_after = engine.wal.as_ref()
            .expect("WAL should still be present").lock().sequence_number();
        assert_eq!(seq_before, seq_after,
            "WAL should not grow while replaying flag is set");

        // Clear replaying flag
        engine.replaying.store(false, std::sync::atomic::Ordering::Relaxed);

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    // ── Schema enforcement tests ────────────────────────────────────────

    #[test]
    fn test_schema_set_and_get() {
        let engine = make_engine();
        create_test_graph(&engine, "sg");

        let cmd = parser::parse_command(
            r#"SCHEMA SET "sg" node "Person" type "age" int"#,
        )
        .unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        assert!(matches!(resp, CommandResponse::Ok));

        let cmd = parser::parse_command(r#"SCHEMA GET "sg""#).unwrap();
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::Text(json) => {
                assert!(json.contains("Person"));
                assert!(json.contains("age"));
                assert!(json.contains("Int"));
            }
            _ => panic!("expected Text response"),
        }
    }

    #[test]
    fn test_schema_blocks_invalid_node_type() {
        let engine = make_engine();
        create_test_graph(&engine, "sbi");

        let cmd = parser::parse_command(
            r#"SCHEMA SET "sbi" node "Person" type "age" int"#,
        )
        .unwrap();
        engine.execute_command(cmd, None).unwrap();

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "sbi".into(),
            label: "Person".into(),
            properties: vec![
                ("name".into(), Value::String(CompactString::new("Alice"))),
                ("age".into(), Value::String(CompactString::new("thirty"))),
            ],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let result = engine.execute_command(cmd, None);
        assert!(result.is_err(), "should reject wrong type");
    }

    #[test]
    fn test_schema_allows_valid_node() {
        let engine = make_engine();
        create_test_graph(&engine, "sav");

        let cmd = parser::parse_command(
            r#"SCHEMA SET "sav" node "Person" type "age" int"#,
        )
        .unwrap();
        engine.execute_command(cmd, None).unwrap();

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "sav".into(),
            label: "Person".into(),
            properties: vec![
                ("name".into(), Value::String(CompactString::new("Alice"))),
                ("age".into(), Value::Int(30)),
            ],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let result = engine.execute_command(cmd, None);
        assert!(result.is_ok(), "should accept correct type");
    }

    #[test]
    fn test_schema_required_blocks_missing() {
        let engine = make_engine();
        create_test_graph(&engine, "srm");

        let cmd = parser::parse_command(
            r#"SCHEMA SET "srm" node "Person" required "name""#,
        )
        .unwrap();
        engine.execute_command(cmd, None).unwrap();

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "srm".into(),
            label: "Person".into(),
            properties: vec![("age".into(), Value::Int(30))],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let result = engine.execute_command(cmd, None);
        assert!(result.is_err(), "should reject missing required property");
    }

    #[test]
    fn test_schema_uniqueness_constraint() {
        let engine = make_engine();
        create_test_graph(&engine, "su");

        let cmd = parser::parse_command(
            r#"SCHEMA SET "su" node "Person" unique "email""#,
        )
        .unwrap();
        engine.execute_command(cmd, None).unwrap();

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "su".into(),
            label: "Person".into(),
            properties: vec![(
                "email".into(),
                Value::String(CompactString::new("alice@example.com")),
            )],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        assert!(engine.execute_command(cmd, None).is_ok());

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "su".into(),
            label: "Person".into(),
            properties: vec![(
                "email".into(),
                Value::String(CompactString::new("alice@example.com")),
            )],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let result = engine.execute_command(cmd, None);
        assert!(result.is_err(), "should reject duplicate unique value");

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "su".into(),
            label: "Person".into(),
            properties: vec![(
                "email".into(),
                Value::String(CompactString::new("bob@example.com")),
            )],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        assert!(engine.execute_command(cmd, None).is_ok());
    }

    #[test]
    fn test_schema_edge_type_enforcement() {
        let engine = make_engine();
        create_test_graph(&engine, "ste");

        let cmd = parser::parse_command(
            r#"SCHEMA SET "ste" edge "KNOWS" type "strength" float"#,
        )
        .unwrap();
        engine.execute_command(cmd, None).unwrap();

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "ste".into(),
            label: "Person".into(),
            properties: vec![(
                "name".into(),
                Value::String(CompactString::new("Alice")),
            )],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let n1 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "ste".into(),
            label: "Person".into(),
            properties: vec![(
                "name".into(),
                Value::String(CompactString::new("Bob")),
            )],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let n2 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = Command::EdgeAdd(parser::EdgeAddCmd {
            graph: "ste".into(),
            source: n1,
            target: n2,
            label: "KNOWS".into(),
            weight: 1.0,
            properties: vec![("strength".into(), Value::Float(0.95))],
            ttl_ms: None,
        });
        assert!(engine.execute_command(cmd, None).is_ok());

        let cmd = Command::EdgeAdd(parser::EdgeAddCmd {
            graph: "ste".into(),
            source: n1,
            target: n2,
            label: "KNOWS".into(),
            weight: 1.0,
            properties: vec![("strength".into(), Value::Int(1))],
            ttl_ms: None,
        });
        assert!(engine.execute_command(cmd, None).is_err());
    }

    #[test]
    fn test_schema_no_constraints_allows_all() {
        let engine = make_engine();
        create_test_graph(&engine, "snc");

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "snc".into(),
            label: "Anything".into(),
            properties: vec![
                ("random".into(), Value::Bool(true)),
                ("data".into(), Value::Float(3.14)),
            ],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        assert!(engine.execute_command(cmd, None).is_ok());
    }

    #[test]
    fn test_schema_set_nonexistent_graph() {
        let engine = make_engine();
        let cmd = parser::parse_command(
            r#"SCHEMA SET "nonexistent" node "Person" type "age" int"#,
        )
        .unwrap();
        let result = engine.execute_command(cmd, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_schema_set_invalid_value_type() {
        let engine = make_engine();
        create_test_graph(&engine, "sivt");

        let cmd = Command::SchemaSet(parser::SchemaSetCmd {
            graph: "sivt".into(),
            target: "node".into(),
            label: "Person".into(),
            constraint_type: "type".into(),
            property: "age".into(),
            value_type: Some("unknown_type".into()),
        });
        let result = engine.execute_command(cmd, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_schema_uniqueness_different_labels() {
        let engine = make_engine();
        create_test_graph(&engine, "sudl");

        let cmd = parser::parse_command(
            r#"SCHEMA SET "sudl" node "Person" unique "email""#,
        )
        .unwrap();
        engine.execute_command(cmd, None).unwrap();

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "sudl".into(),
            label: "Person".into(),
            properties: vec![(
                "email".into(),
                Value::String(CompactString::new("shared@example.com")),
            )],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        assert!(engine.execute_command(cmd, None).is_ok());

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "sudl".into(),
            label: "Company".into(),
            properties: vec![(
                "email".into(),
                Value::String(CompactString::new("shared@example.com")),
            )],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        assert!(engine.execute_command(cmd, None).is_ok());
    }

    #[test]
    fn test_schema_multiple_constraints_combined() {
        let engine = make_engine();
        create_test_graph(&engine, "smc");

        let cmd = parser::parse_command(
            r#"SCHEMA SET "smc" node "Person" required "name""#,
        )
        .unwrap();
        engine.execute_command(cmd, None).unwrap();

        let cmd = parser::parse_command(
            r#"SCHEMA SET "smc" node "Person" type "age" int"#,
        )
        .unwrap();
        engine.execute_command(cmd, None).unwrap();

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "smc".into(),
            label: "Person".into(),
            properties: vec![
                ("name".into(), Value::String(CompactString::new("Alice"))),
                ("age".into(), Value::Int(30)),
            ],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        assert!(engine.execute_command(cmd, None).is_ok());

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "smc".into(),
            label: "Person".into(),
            properties: vec![("age".into(), Value::Int(25))],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        assert!(engine.execute_command(cmd, None).is_err());

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "smc".into(),
            label: "Person".into(),
            properties: vec![
                ("name".into(), Value::String(CompactString::new("Bob"))),
                ("age".into(), Value::String(CompactString::new("old"))),
            ],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        assert!(engine.execute_command(cmd, None).is_err());
    }

    // ── LRU eviction tests ─────────────────────────────────────────────

    #[test]
    fn test_lru_eviction_basic() {
        let engine = make_engine();
        let mut gc = weav_core::config::GraphConfig::default();
        gc.max_nodes = Some(3);
        gc.eviction_policy = weav_core::config::EvictionPolicy::LRU;
        let cmd = Command::GraphCreate(parser::GraphCreateCmd {
            name: "lru_basic".to_string(),
            config: Some(gc),
        });
        engine.execute_command(cmd, None).unwrap();

        // Add 3 nodes (fills capacity)
        for i in 0..3 {
            let cmd = Command::NodeAdd(parser::NodeAddCmd {
                graph: "lru_basic".to_string(),
                label: "x".to_string(),
                properties: vec![("i".to_string(), Value::Int(i))],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            });
            engine.execute_command(cmd, None).unwrap();
        }

        // Set deterministic access times: node 1 oldest, node 2 mid, node 3 newest
        {
            let graph_arc = engine.get_graph("lru_basic").unwrap();
            let mut gs = graph_arc.write();
            gs.access_times.insert(1, 1000);
            gs.access_times.insert(2, 2000);
            gs.access_times.insert(3, 3000);
        }

        // Add a 4th node -- should evict node 1 (oldest access time)
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "lru_basic".to_string(),
            label: "x".to_string(),
            properties: vec![("i".to_string(), Value::Int(99))],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        engine.execute_command(cmd, None).unwrap();

        // Verify we still have 3 nodes
        let graph_arc = engine.get_graph("lru_basic").unwrap();
        let gs = graph_arc.read();
        assert_eq!(gs.adjacency.node_count(), 3);

        // Node 1 (oldest access time) should be gone
        assert!(!gs.adjacency.has_node(1));
        // Nodes 2, 3, and 4 should still exist
        assert!(gs.adjacency.has_node(2));
        assert!(gs.adjacency.has_node(3));
        assert!(gs.adjacency.has_node(4));
    }

    #[test]
    fn test_lru_eviction_access_updates() {
        let engine = make_engine();
        let mut gc = weav_core::config::GraphConfig::default();
        gc.max_nodes = Some(3);
        gc.eviction_policy = weav_core::config::EvictionPolicy::LRU;
        let cmd = Command::GraphCreate(parser::GraphCreateCmd {
            name: "lru_access".to_string(),
            config: Some(gc),
        });
        engine.execute_command(cmd, None).unwrap();

        // Add 3 nodes
        for i in 0..3 {
            let cmd = Command::NodeAdd(parser::NodeAddCmd {
                graph: "lru_access".to_string(),
                label: "x".to_string(),
                properties: vec![("i".to_string(), Value::Int(i))],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            });
            engine.execute_command(cmd, None).unwrap();
        }

        // Set deterministic access times: node 2 is oldest
        {
            let graph_arc = engine.get_graph("lru_access").unwrap();
            let mut gs = graph_arc.write();
            gs.access_times.insert(1, 1000);
            gs.access_times.insert(2, 500); // node 2 is the oldest
            gs.access_times.insert(3, 2000);
        }

        // Access node 1 via NodeGet to refresh its timestamp (making it recently used)
        let cmd = Command::NodeGet(parser::NodeGetCmd {
            graph: "lru_access".to_string(),
            node_id: Some(1),
            entity_key: None,
        });
        engine.execute_command(cmd, None).unwrap();

        // Add a 4th node -- should evict node 2 (oldest access time = 500)
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "lru_access".to_string(),
            label: "x".to_string(),
            properties: vec![("i".to_string(), Value::Int(99))],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        engine.execute_command(cmd, None).unwrap();

        let graph_arc = engine.get_graph("lru_access").unwrap();
        let gs = graph_arc.read();
        assert_eq!(gs.adjacency.node_count(), 3);

        // Node 1 should still exist (was recently accessed via NodeGet)
        assert!(gs.adjacency.has_node(1));
        // Node 2 should have been evicted (oldest access time)
        assert!(!gs.adjacency.has_node(2));
        // Nodes 3 and 4 should still exist
        assert!(gs.adjacency.has_node(3));
        assert!(gs.adjacency.has_node(4));
    }

    #[test]
    fn test_no_eviction_returns_error() {
        let engine = make_engine();
        let mut gc = weav_core::config::GraphConfig::default();
        gc.max_nodes = Some(2);
        // Default eviction_policy is NoEviction
        assert_eq!(gc.eviction_policy, weav_core::config::EvictionPolicy::NoEviction);
        let cmd = Command::GraphCreate(parser::GraphCreateCmd {
            name: "no_evict".to_string(),
            config: Some(gc),
        });
        engine.execute_command(cmd, None).unwrap();

        // Fill to capacity
        for i in 0..2 {
            let cmd = Command::NodeAdd(parser::NodeAddCmd {
                graph: "no_evict".to_string(),
                label: "x".to_string(),
                properties: vec![("i".to_string(), Value::Int(i))],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            });
            engine.execute_command(cmd, None).unwrap();
        }

        // Third should fail with CapacityExceeded
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "no_evict".to_string(),
            label: "x".to_string(),
            properties: vec![],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let result = engine.execute_command(cmd, None);
        assert!(result.is_err());
        match result.unwrap_err() {
            WeavError::CapacityExceeded(_) => {}
            other => panic!("expected CapacityExceeded, got: {other}"),
        }
    }

    // ── Full-text search (BM25) tests ─────────────────────────────────────

    #[test]
    fn test_fulltext_search_basic() {
        let engine = make_engine();
        create_test_graph(&engine, "ft");

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "ft".to_string(),
            label: "doc".to_string(),
            properties: vec![
                ("content".to_string(), Value::String(CompactString::from(
                    "Rust programming language systems performance",
                ))),
            ],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let id1 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "ft".to_string(),
            label: "doc".to_string(),
            properties: vec![
                ("content".to_string(), Value::String(CompactString::from(
                    "Python scripting language for data science",
                ))),
            ],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let id2 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "ft".to_string(),
            label: "doc".to_string(),
            properties: vec![
                ("content".to_string(), Value::String(CompactString::from(
                    "Rust compiler and borrow checker design",
                ))),
            ],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let id3 = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Search for "rust language" - should find nodes 1 and 3
        let cmd = Command::SearchText(parser::SearchTextCmd {
            graph: "ft".to_string(),
            query: "rust language".to_string(),
            limit: Some(10),
        });
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::StringList(results) => {
                assert!(!results.is_empty(), "should find results for 'rust language'");
                let result_ids: Vec<u64> = results.iter()
                    .filter_map(|s| s.split(':').next()?.parse::<u64>().ok())
                    .collect();
                assert!(result_ids.contains(&id1), "node {} should be in results", id1);
                assert!(result_ids.contains(&id3), "node {} should be in results", id3);
                // Node 1 should rank first (has both "rust" and "language")
                assert_eq!(result_ids[0], id1, "node with both terms should rank first");
            }
            _ => panic!("expected StringList response"),
        }

        // Search for "python" - should find only node 2
        let cmd = Command::SearchText(parser::SearchTextCmd {
            graph: "ft".to_string(),
            query: "python".to_string(),
            limit: Some(10),
        });
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::StringList(results) => {
                assert_eq!(results.len(), 1);
                let first_id: u64 = results[0].split(':').next().unwrap().parse().unwrap();
                assert_eq!(first_id, id2);
            }
            _ => panic!("expected StringList response"),
        }
    }

    #[test]
    fn test_fulltext_search_node_delete_removes_from_index() {
        let engine = make_engine();
        create_test_graph(&engine, "ftdel");

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "ftdel".to_string(),
            label: "doc".to_string(),
            properties: vec![
                ("content".to_string(), Value::String(CompactString::from(
                    "unique searchable content here",
                ))),
            ],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let node_id = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected Integer"),
        };

        // Verify it's searchable
        let cmd = Command::SearchText(parser::SearchTextCmd {
            graph: "ftdel".to_string(),
            query: "unique searchable".to_string(),
            limit: Some(10),
        });
        let resp = engine.execute_command(cmd, None).unwrap();
        match &resp {
            CommandResponse::StringList(results) => assert_eq!(results.len(), 1),
            _ => panic!("expected StringList response"),
        }

        // Delete the node
        let cmd = Command::NodeDelete(parser::NodeDeleteCmd {
            graph: "ftdel".to_string(),
            node_id,
        });
        engine.execute_command(cmd, None).unwrap();

        // Search again - should find nothing
        let cmd = Command::SearchText(parser::SearchTextCmd {
            graph: "ftdel".to_string(),
            query: "unique searchable".to_string(),
            limit: Some(10),
        });
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::StringList(results) => {
                assert!(results.is_empty(), "deleted node should not appear in search");
            }
            _ => panic!("expected StringList response"),
        }
    }

    #[test]
    fn test_fulltext_search_empty_graph() {
        let engine = make_engine();
        create_test_graph(&engine, "ftempty");

        let cmd = Command::SearchText(parser::SearchTextCmd {
            graph: "ftempty".to_string(),
            query: "anything".to_string(),
            limit: Some(10),
        });
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::StringList(results) => {
                assert!(results.is_empty(), "empty graph should return no results");
            }
            _ => panic!("expected StringList response"),
        }
    }

    // ── PropertyUnique enforcement tests ─────────────────────────────────

    #[test]
    fn test_uniqueness_blocks_duplicate_insert() {
        let engine = make_engine();
        create_test_graph(&engine, "uq_dup");

        // Set unique constraint on email for Person
        let cmd = parser::parse_command(
            r#"SCHEMA SET "uq_dup" node "Person" unique "email""#,
        )
        .unwrap();
        engine.execute_command(cmd, None).unwrap();

        // First insert succeeds
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "uq_dup".into(),
            label: "Person".into(),
            properties: vec![(
                "email".into(),
                Value::String(CompactString::new("alice@example.com")),
            )],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        assert!(engine.execute_command(cmd, None).is_ok());

        // Duplicate insert fails with Conflict
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "uq_dup".into(),
            label: "Person".into(),
            properties: vec![(
                "email".into(),
                Value::String(CompactString::new("alice@example.com")),
            )],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        let result = engine.execute_command(cmd, None);
        assert!(result.is_err());
        match result.unwrap_err() {
            WeavError::Conflict(msg) => {
                assert!(msg.contains("email"));
                assert!(msg.contains("Person"));
            }
            other => panic!("expected Conflict, got: {other}"),
        }
    }

    #[test]
    fn test_uniqueness_allows_different_labels() {
        let engine = make_engine();
        create_test_graph(&engine, "uq_labels");

        // Unique constraint only on Person label
        let cmd = parser::parse_command(
            r#"SCHEMA SET "uq_labels" node "Person" unique "email""#,
        )
        .unwrap();
        engine.execute_command(cmd, None).unwrap();

        // Add Person with email
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "uq_labels".into(),
            label: "Person".into(),
            properties: vec![(
                "email".into(),
                Value::String(CompactString::new("shared@example.com")),
            )],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        assert!(engine.execute_command(cmd, None).is_ok());

        // Same email on Company label should succeed (different label)
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "uq_labels".into(),
            label: "Company".into(),
            properties: vec![(
                "email".into(),
                Value::String(CompactString::new("shared@example.com")),
            )],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        assert!(engine.execute_command(cmd, None).is_ok());
    }

    #[test]
    fn test_uniqueness_allows_null_values() {
        let engine = make_engine();
        create_test_graph(&engine, "uq_null");

        let cmd = parser::parse_command(
            r#"SCHEMA SET "uq_null" node "Person" unique "email""#,
        )
        .unwrap();
        engine.execute_command(cmd, None).unwrap();

        // First node with null email
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "uq_null".into(),
            label: "Person".into(),
            properties: vec![("email".into(), Value::Null)],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        assert!(engine.execute_command(cmd, None).is_ok());

        // Second node with null email should also succeed
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "uq_null".into(),
            label: "Person".into(),
            properties: vec![("email".into(), Value::Null)],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        assert!(engine.execute_command(cmd, None).is_ok());
    }

    #[test]
    fn test_edge_uniqueness_enforcement() {
        let engine = make_engine();
        create_test_graph(&engine, "euq");

        // Set unique constraint on edge label KNOWS for property "ref_id"
        let cmd = Command::SchemaSet(parser::SchemaSetCmd {
            graph: "euq".into(),
            target: "edge".into(),
            label: "KNOWS".into(),
            constraint_type: "unique".into(),
            property: "ref_id".into(),
            value_type: None,
        });
        engine.execute_command(cmd, None).unwrap();

        // Add two nodes
        let n1 = match engine.execute_command(
            Command::NodeAdd(parser::NodeAddCmd {
                graph: "euq".into(),
                label: "Person".into(),
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
        let n2 = match engine.execute_command(
            Command::NodeAdd(parser::NodeAddCmd {
                graph: "euq".into(),
                label: "Person".into(),
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

        // First edge with ref_id succeeds
        let cmd = Command::EdgeAdd(parser::EdgeAddCmd {
            graph: "euq".into(),
            source: n1,
            target: n2,
            label: "KNOWS".into(),
            weight: 1.0,
            properties: vec![(
                "ref_id".into(),
                Value::String(CompactString::new("abc123")),
            )],
            ttl_ms: None,
        });
        assert!(engine.execute_command(cmd, None).is_ok());

        // Duplicate ref_id on same edge label fails
        let cmd = Command::EdgeAdd(parser::EdgeAddCmd {
            graph: "euq".into(),
            source: n2,
            target: n1,
            label: "KNOWS".into(),
            weight: 1.0,
            properties: vec![(
                "ref_id".into(),
                Value::String(CompactString::new("abc123")),
            )],
            ttl_ms: None,
        });
        let result = engine.execute_command(cmd, None);
        assert!(result.is_err());
        match result.unwrap_err() {
            WeavError::Conflict(msg) => {
                assert!(msg.contains("ref_id"));
            }
            other => panic!("expected Conflict, got: {other}"),
        }
    }

    // ── NodeMerge tests ──────────────────────────────────────────────────

    #[test]
    fn test_node_merge_basic() {
        let engine = make_engine();
        create_test_graph(&engine, "nm_basic");

        // Add source node with properties
        let n1 = match engine.execute_command(
            Command::NodeAdd(parser::NodeAddCmd {
                graph: "nm_basic".into(),
                label: "Person".into(),
                properties: vec![
                    (
                        "name".into(),
                        Value::String(CompactString::new("Alice")),
                    ),
                    ("age".into(), Value::Int(30)),
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
            _ => panic!("expected Integer"),
        };

        // Add target node with different properties
        let n2 = match engine.execute_command(
            Command::NodeAdd(parser::NodeAddCmd {
                graph: "nm_basic".into(),
                label: "Person".into(),
                properties: vec![(
                    "city".into(),
                    Value::String(CompactString::new("NYC")),
                )],
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

        // Merge n1 into n2
        let cmd = Command::NodeMerge(parser::NodeMergeCmd {
            graph: "nm_basic".into(),
            source_id: n1,
            target_id: n2,
            conflict_policy: "keep_target".into(),
        });
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::Integer(id) => assert_eq!(id, n2),
            _ => panic!("expected Integer response with target id"),
        }

        // Source node should be deleted
        let cmd = Command::NodeGet(parser::NodeGetCmd {
            graph: "nm_basic".into(),
            node_id: Some(n1),
            entity_key: None,
        });
        assert!(engine.execute_command(cmd, None).is_err());

        // Target node should have merged properties
        let cmd = Command::NodeGet(parser::NodeGetCmd {
            graph: "nm_basic".into(),
            node_id: Some(n2),
            entity_key: None,
        });
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::NodeInfo(info) => {
                // Should have name, age (from source) and city (from target)
                assert!(info.properties.iter().any(|(k, _)| k == "name"));
                assert!(info.properties.iter().any(|(k, _)| k == "age"));
                assert!(info.properties.iter().any(|(k, _)| k == "city"));
            }
            _ => panic!("expected NodeInfo"),
        }
    }

    #[test]
    fn test_node_merge_edge_relinking() {
        let engine = make_engine();
        create_test_graph(&engine, "nm_edges");

        // Create three nodes: n1 (source), n2 (target), n3 (neighbor)
        let n1 = match engine.execute_command(
            Command::NodeAdd(parser::NodeAddCmd {
                graph: "nm_edges".into(),
                label: "Person".into(),
                properties: vec![(
                    "name".into(),
                    Value::String(CompactString::new("Source")),
                )],
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

        let n2 = match engine.execute_command(
            Command::NodeAdd(parser::NodeAddCmd {
                graph: "nm_edges".into(),
                label: "Person".into(),
                properties: vec![(
                    "name".into(),
                    Value::String(CompactString::new("Target")),
                )],
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

        let n3 = match engine.execute_command(
            Command::NodeAdd(parser::NodeAddCmd {
                graph: "nm_edges".into(),
                label: "Person".into(),
                properties: vec![(
                    "name".into(),
                    Value::String(CompactString::new("Neighbor")),
                )],
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

        // Create edge from n1 -> n3
        let cmd = Command::EdgeAdd(parser::EdgeAddCmd {
            graph: "nm_edges".into(),
            source: n1,
            target: n3,
            label: "KNOWS".into(),
            weight: 0.9,
            properties: vec![("strength".into(), Value::Float(0.8))],
            ttl_ms: None,
        });
        assert!(engine.execute_command(cmd, None).is_ok());

        // Merge n1 into n2 -- edge should be re-linked to n2 -> n3
        let cmd = Command::NodeMerge(parser::NodeMergeCmd {
            graph: "nm_edges".into(),
            source_id: n1,
            target_id: n2,
            conflict_policy: "keep_target".into(),
        });
        assert!(engine.execute_command(cmd, None).is_ok());

        // n2 should now have a neighbor n3
        let cmd = Command::Neighbors(parser::NeighborsCmd {
            graph: "nm_edges".into(),
            node_id: n2,
            label: None,
            direction: weav_core::types::Direction::Outgoing,
        });
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::StringList(list) => {
                // Should contain an entry referencing n3
                let has_n3 = list.iter().any(|s| s.starts_with(&format!("{}:", n3)));
                assert!(has_n3, "expected edge to n3 after merge, got: {:?}", list);
            }
            _ => panic!("expected StringList"),
        }
    }

    #[test]
    fn test_node_merge_conflict_keep_source() {
        let engine = make_engine();
        create_test_graph(&engine, "nm_ks");

        let n1 = match engine.execute_command(
            Command::NodeAdd(parser::NodeAddCmd {
                graph: "nm_ks".into(),
                label: "Person".into(),
                properties: vec![(
                    "name".into(),
                    Value::String(CompactString::new("SourceName")),
                )],
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

        let n2 = match engine.execute_command(
            Command::NodeAdd(parser::NodeAddCmd {
                graph: "nm_ks".into(),
                label: "Person".into(),
                properties: vec![(
                    "name".into(),
                    Value::String(CompactString::new("TargetName")),
                )],
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

        // Merge with keep_source policy -- source value should win on conflict
        let cmd = Command::NodeMerge(parser::NodeMergeCmd {
            graph: "nm_ks".into(),
            source_id: n1,
            target_id: n2,
            conflict_policy: "keep_source".into(),
        });
        assert!(engine.execute_command(cmd, None).is_ok());

        // Target should have source's name value
        let cmd = Command::NodeGet(parser::NodeGetCmd {
            graph: "nm_ks".into(),
            node_id: Some(n2),
            entity_key: None,
        });
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::NodeInfo(info) => {
                let name = info
                    .properties
                    .iter()
                    .find(|(k, _)| k == "name")
                    .map(|(_, v)| v.clone());
                assert_eq!(
                    name,
                    Some(Value::String(CompactString::new("SourceName")))
                );
            }
            _ => panic!("expected NodeInfo"),
        }
    }

    #[test]
    fn test_node_merge_conflict_keep_target() {
        let engine = make_engine();
        create_test_graph(&engine, "nm_kt");

        let n1 = match engine.execute_command(
            Command::NodeAdd(parser::NodeAddCmd {
                graph: "nm_kt".into(),
                label: "Person".into(),
                properties: vec![(
                    "name".into(),
                    Value::String(CompactString::new("SourceName")),
                )],
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

        let n2 = match engine.execute_command(
            Command::NodeAdd(parser::NodeAddCmd {
                graph: "nm_kt".into(),
                label: "Person".into(),
                properties: vec![(
                    "name".into(),
                    Value::String(CompactString::new("TargetName")),
                )],
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

        // Merge with keep_target policy -- target value should win
        let cmd = Command::NodeMerge(parser::NodeMergeCmd {
            graph: "nm_kt".into(),
            source_id: n1,
            target_id: n2,
            conflict_policy: "keep_target".into(),
        });
        assert!(engine.execute_command(cmd, None).is_ok());

        // Target should keep its own name
        let cmd = Command::NodeGet(parser::NodeGetCmd {
            graph: "nm_kt".into(),
            node_id: Some(n2),
            entity_key: None,
        });
        let resp = engine.execute_command(cmd, None).unwrap();
        match resp {
            CommandResponse::NodeInfo(info) => {
                let name = info
                    .properties
                    .iter()
                    .find(|(k, _)| k == "name")
                    .map(|(_, v)| v.clone());
                assert_eq!(
                    name,
                    Some(Value::String(CompactString::new("TargetName")))
                );
            }
            _ => panic!("expected NodeInfo"),
        }
    }

    #[test]
    fn test_node_merge_nonexistent_source() {
        let engine = make_engine();
        create_test_graph(&engine, "nm_ne");

        let n1 = match engine.execute_command(
            Command::NodeAdd(parser::NodeAddCmd {
                graph: "nm_ne".into(),
                label: "Person".into(),
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

        // Merge with nonexistent source
        let cmd = Command::NodeMerge(parser::NodeMergeCmd {
            graph: "nm_ne".into(),
            source_id: 99999,
            target_id: n1,
            conflict_policy: "keep_target".into(),
        });
        assert!(engine.execute_command(cmd, None).is_err());
    }

    #[test]
    fn test_node_merge_self_merge_rejected() {
        let engine = make_engine();
        create_test_graph(&engine, "nm_self");

        let n1 = match engine.execute_command(
            Command::NodeAdd(parser::NodeAddCmd {
                graph: "nm_self".into(),
                label: "Person".into(),
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

        let cmd = Command::NodeMerge(parser::NodeMergeCmd {
            graph: "nm_self".into(),
            source_id: n1,
            target_id: n1,
            conflict_policy: "keep_target".into(),
        });
        let result = engine.execute_command(cmd, None);
        assert!(result.is_err());
        match result.unwrap_err() {
            WeavError::Conflict(msg) => assert!(msg.contains("itself")),
            other => panic!("expected Conflict, got: {other}"),
        }
    }

    // ── CDC event tests ───────────────────────────────────────────────────

    #[test]
    fn test_cdc_event_node_created() {
        let engine = make_engine();
        create_test_graph(&engine, "cdc_nc");

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "cdc_nc".into(),
            label: "Person".into(),
            properties: vec![
                ("name".into(), Value::String(CompactString::new("Alice"))),
            ],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        engine.execute_command(cmd, None).unwrap();

        let events = engine.recent_events(10);
        // Should have at least 2 events: GraphCreated + NodeCreated
        assert!(events.len() >= 2);
        let node_event = &events[0]; // newest first
        match &node_event.kind {
            weav_core::events::EventKind::NodeCreated { node_id, label, properties } => {
                assert_eq!(*node_id, 1);
                assert_eq!(label.as_str(), "Person");
                assert_eq!(properties.len(), 1);
            }
            other => panic!("expected NodeCreated, got {:?}", other),
        }
        assert_eq!(node_event.graph.as_str(), "cdc_nc");
    }

    #[test]
    fn test_cdc_event_edge_created() {
        let engine = make_engine();
        create_test_graph(&engine, "cdc_ec");

        // Add two nodes
        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "cdc_ec".into(),
            label: "Person".into(),
            properties: vec![],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        engine.execute_command(cmd, None).unwrap();

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "cdc_ec".into(),
            label: "Person".into(),
            properties: vec![],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        engine.execute_command(cmd, None).unwrap();

        // Add edge
        let cmd = Command::EdgeAdd(parser::EdgeAddCmd {
            graph: "cdc_ec".into(),
            source: 1,
            target: 2,
            label: "KNOWS".into(),
            weight: 0.8,
            properties: vec![],
            ttl_ms: None,
        });
        engine.execute_command(cmd, None).unwrap();

        let events = engine.recent_events(10);
        let edge_event = &events[0]; // newest first
        match &edge_event.kind {
            weav_core::events::EventKind::EdgeCreated { edge_id, source, target, label, weight } => {
                assert!(*edge_id > 0);
                assert_eq!(*source, 1);
                assert_eq!(*target, 2);
                assert_eq!(label.as_str(), "KNOWS");
                assert!((*weight - 0.8).abs() < f32::EPSILON);
            }
            other => panic!("expected EdgeCreated, got {:?}", other),
        }
    }

    #[test]
    fn test_cdc_event_node_deleted() {
        let engine = make_engine();
        create_test_graph(&engine, "cdc_nd");

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "cdc_nd".into(),
            label: "Temp".into(),
            properties: vec![],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        engine.execute_command(cmd, None).unwrap();

        let cmd = Command::NodeDelete(parser::NodeDeleteCmd {
            graph: "cdc_nd".into(),
            node_id: 1,
        });
        engine.execute_command(cmd, None).unwrap();

        let events = engine.recent_events(10);
        let del_event = &events[0];
        match &del_event.kind {
            weav_core::events::EventKind::NodeDeleted { node_id } => {
                assert_eq!(*node_id, 1);
            }
            other => panic!("expected NodeDeleted, got {:?}", other),
        }
    }

    #[test]
    fn test_cdc_event_sequence_monotonic() {
        let engine = make_engine();
        create_test_graph(&engine, "cdc_seq");

        // Create several nodes to generate events
        for i in 0..5 {
            let cmd = Command::NodeAdd(parser::NodeAddCmd {
                graph: "cdc_seq".into(),
                label: format!("Type{}", i),
                properties: vec![],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            });
            engine.execute_command(cmd, None).unwrap();
        }

        let events = engine.recent_events(100);
        // Events are returned newest-first, so sequences should be decreasing
        assert!(events.len() >= 6); // 1 GraphCreated + 5 NodeCreated
        for window in events.windows(2) {
            assert!(
                window[0].sequence > window[1].sequence,
                "sequence numbers should be strictly monotonically increasing (newest first): {} vs {}",
                window[0].sequence,
                window[1].sequence
            );
        }
    }

    #[test]
    fn test_cdc_event_graph_lifecycle() {
        let engine = make_engine();
        create_test_graph(&engine, "cdc_gl");

        let cmd = parser::parse_command("GRAPH DROP \"cdc_gl\"").unwrap();
        engine.execute_command(cmd, None).unwrap();

        let events = engine.recent_events(10);
        // Should have GraphDropped as most recent
        match &events[0].kind {
            weav_core::events::EventKind::GraphDropped { name } => {
                assert_eq!(name.as_str(), "cdc_gl");
            }
            other => panic!("expected GraphDropped, got {:?}", other),
        }
        // And GraphCreated before that
        match &events[1].kind {
            weav_core::events::EventKind::GraphCreated { name } => {
                assert_eq!(name.as_str(), "cdc_gl");
            }
            other => panic!("expected GraphCreated, got {:?}", other),
        }
    }

    #[test]
    fn test_cdc_subscribe_returns_receiver() {
        let engine = make_engine();
        let _rx = engine.subscribe_events();
        // Just verify subscribe doesn't panic and returns a receiver
    }

    #[test]
    fn test_cdc_recent_events_limit() {
        let engine = make_engine();
        create_test_graph(&engine, "cdc_lim");

        for _ in 0..10 {
            let cmd = Command::NodeAdd(parser::NodeAddCmd {
                graph: "cdc_lim".into(),
                label: "X".into(),
                properties: vec![],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            });
            engine.execute_command(cmd, None).unwrap();
        }

        // Should have 11 events total (1 GraphCreated + 10 NodeCreated)
        let all = engine.recent_events(100);
        assert_eq!(all.len(), 11);

        // Limit should work
        let limited = engine.recent_events(3);
        assert_eq!(limited.len(), 3);
    }

    #[test]
    fn test_cdc_event_node_updated() {
        let engine = make_engine();
        create_test_graph(&engine, "cdc_nu");

        let cmd = Command::NodeAdd(parser::NodeAddCmd {
            graph: "cdc_nu".into(),
            label: "Person".into(),
            properties: vec![],
            embedding: None,
            entity_key: None,
            ttl_ms: None,
        });
        engine.execute_command(cmd, None).unwrap();

        let cmd = Command::NodeUpdate(parser::NodeUpdateCmd {
            graph: "cdc_nu".into(),
            node_id: 1,
            properties: vec![
                ("age".into(), Value::Int(30)),
            ],
            embedding: None,
        });
        engine.execute_command(cmd, None).unwrap();

        let events = engine.recent_events(10);
        match &events[0].kind {
            weav_core::events::EventKind::NodeUpdated { node_id, properties } => {
                assert_eq!(*node_id, 1);
                assert_eq!(properties.len(), 1);
                assert_eq!(properties[0].0.as_str(), "age");
            }
            other => panic!("expected NodeUpdated, got {:?}", other),
        }
    }

    #[test]
    fn test_cdc_event_edge_invalidated() {
        let engine = make_engine();
        create_test_graph(&engine, "cdc_ei");

        // Add two nodes + edge
        for _ in 0..2 {
            let cmd = Command::NodeAdd(parser::NodeAddCmd {
                graph: "cdc_ei".into(),
                label: "N".into(),
                properties: vec![],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            });
            engine.execute_command(cmd, None).unwrap();
        }
        let cmd = Command::EdgeAdd(parser::EdgeAddCmd {
            graph: "cdc_ei".into(),
            source: 1,
            target: 2,
            label: "REL".into(),
            weight: 1.0,
            properties: vec![],
            ttl_ms: None,
        });
        let resp = engine.execute_command(cmd, None).unwrap();
        let edge_id = match resp {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected integer"),
        };

        // Invalidate the edge
        let cmd = Command::EdgeInvalidate(parser::EdgeInvalidateCmd {
            graph: "cdc_ei".into(),
            edge_id,
        });
        engine.execute_command(cmd, None).unwrap();

        let events = engine.recent_events(10);
        match &events[0].kind {
            weav_core::events::EventKind::EdgeInvalidated { edge_id: eid } => {
                assert_eq!(*eid, edge_id);
            }
            other => panic!("expected EdgeInvalidated, got {:?}", other),
        }
    }

    #[test]
    fn test_cdc_event_edge_deleted() {
        let engine = make_engine();
        create_test_graph(&engine, "cdc_ed");

        // Add two nodes + edge
        for _ in 0..2 {
            let cmd = Command::NodeAdd(parser::NodeAddCmd {
                graph: "cdc_ed".into(),
                label: "N".into(),
                properties: vec![],
                embedding: None,
                entity_key: None,
                ttl_ms: None,
            });
            engine.execute_command(cmd, None).unwrap();
        }
        let cmd = Command::EdgeAdd(parser::EdgeAddCmd {
            graph: "cdc_ed".into(),
            source: 1,
            target: 2,
            label: "REL".into(),
            weight: 1.0,
            properties: vec![],
            ttl_ms: None,
        });
        let resp = engine.execute_command(cmd, None).unwrap();
        let edge_id = match resp {
            CommandResponse::Integer(id) => id,
            _ => panic!("expected integer"),
        };

        // Delete the edge
        let cmd = Command::EdgeDelete(parser::EdgeDeleteCmd {
            graph: "cdc_ed".into(),
            edge_id,
        });
        engine.execute_command(cmd, None).unwrap();

        let events = engine.recent_events(10);
        match &events[0].kind {
            weav_core::events::EventKind::EdgeDeleted { edge_id: eid } => {
                assert_eq!(*eid, edge_id);
            }
            other => panic!("expected EdgeDeleted, got {:?}", other),
        }
    }
}
