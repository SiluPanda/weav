//! The Weav engine: central coordinator holding all in-memory state.
//!
//! Provides a thread-safe interface for executing commands against graphs.

use std::collections::HashMap;
use std::sync::RwLock;
use std::time::SystemTime;

use compact_str::CompactString;

use weav_core::config::{GraphConfig, WeavConfig};
use weav_core::error::{WeavError, WeavResult};
use weav_core::shard::StringInterner;
use weav_core::types::*;
use weav_graph::adjacency::{AdjacencyStore, EdgeMeta};
use weav_graph::properties::PropertyStore;
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
    Text(String),
    StringList(Vec<String>),
    Context(executor::ContextResult),
    NodeInfo(NodeInfo),
    GraphInfo(GraphInfoResponse),
    Error(String),
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
}

impl Engine {
    /// Create a new engine with the given configuration.
    pub fn new(config: WeavConfig) -> Self {
        let token_counter = TokenCounter::new(config.engine.token_counter.clone());
        Self {
            graphs: RwLock::new(HashMap::new()),
            next_graph_id: RwLock::new(1),
            token_counter,
            config,
        }
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
                    .map_err(|e| WeavError::Internal(format!("lock poisoned: {e}")))?
                    .len()
            ))),
            Command::Stats(graph_name) => self.handle_stats(graph_name),
            Command::Snapshot => {
                // Placeholder for snapshot functionality.
                Ok(CommandResponse::Ok)
            }
            Command::GraphCreate(cmd) => self.handle_graph_create(cmd),
            Command::GraphDrop(name) => self.handle_graph_drop(&name),
            Command::GraphInfo(name) => self.handle_graph_info(&name),
            Command::GraphList => self.handle_graph_list(),
            Command::NodeAdd(cmd) => self.handle_node_add(cmd),
            Command::NodeGet(cmd) => self.handle_node_get(cmd),
            Command::NodeDelete(cmd) => self.handle_node_delete(cmd),
            Command::EdgeAdd(cmd) => self.handle_edge_add(cmd),
            Command::EdgeInvalidate(cmd) => self.handle_edge_invalidate(cmd),
            Command::Context(query) => self.handle_context(query),
        }
    }

    // ── Stats ────────────────────────────────────────────────────────────

    fn handle_stats(&self, graph_name: Option<String>) -> WeavResult<CommandResponse> {
        let graphs = self
            .graphs
            .read()
            .map_err(|e| WeavError::Internal(format!("lock poisoned: {e}")))?;

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
                .write()
                .map_err(|e| WeavError::Internal(format!("lock poisoned: {e}")))?;
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
            .write()
            .map_err(|e| WeavError::Internal(format!("lock poisoned: {e}")))?;

        if graphs.contains_key(&cmd.name) {
            return Err(WeavError::Conflict(format!(
                "graph '{}' already exists",
                cmd.name
            )));
        }

        graphs.insert(cmd.name, graph_state);
        Ok(CommandResponse::Ok)
    }

    fn handle_graph_drop(&self, name: &str) -> WeavResult<CommandResponse> {
        let mut graphs = self
            .graphs
            .write()
            .map_err(|e| WeavError::Internal(format!("lock poisoned: {e}")))?;
        graphs
            .remove(name)
            .ok_or_else(|| WeavError::GraphNotFound(name.to_string()))?;
        Ok(CommandResponse::Ok)
    }

    fn handle_graph_info(&self, name: &str) -> WeavResult<CommandResponse> {
        let graphs = self
            .graphs
            .read()
            .map_err(|e| WeavError::Internal(format!("lock poisoned: {e}")))?;
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
            .read()
            .map_err(|e| WeavError::Internal(format!("lock poisoned: {e}")))?;
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
            .write()
            .map_err(|e| WeavError::Internal(format!("lock poisoned: {e}")))?;
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

        Ok(CommandResponse::Integer(node_id))
    }

    fn handle_node_get(
        &self,
        cmd: weav_query::parser::NodeGetCmd,
    ) -> WeavResult<CommandResponse> {
        let graphs = self
            .graphs
            .read()
            .map_err(|e| WeavError::Internal(format!("lock poisoned: {e}")))?;
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
            .write()
            .map_err(|e| WeavError::Internal(format!("lock poisoned: {e}")))?;
        let gs = graphs
            .get_mut(&cmd.graph)
            .ok_or_else(|| WeavError::GraphNotFound(cmd.graph.clone()))?;

        // Remove from adjacency (also removes connected edges).
        gs.adjacency.remove_node(cmd.node_id)?;

        // Remove all properties.
        gs.properties.remove_all_node_properties(cmd.node_id);

        // Remove from vector index.
        gs.vector_index.remove(cmd.node_id)?;

        Ok(CommandResponse::Ok)
    }

    // ── Edge commands ────────────────────────────────────────────────────

    fn handle_edge_add(
        &self,
        cmd: weav_query::parser::EdgeAddCmd,
    ) -> WeavResult<CommandResponse> {
        let mut graphs = self
            .graphs
            .write()
            .map_err(|e| WeavError::Internal(format!("lock poisoned: {e}")))?;
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

        Ok(CommandResponse::Integer(edge_id))
    }

    fn handle_edge_invalidate(
        &self,
        cmd: weav_query::parser::EdgeInvalidateCmd,
    ) -> WeavResult<CommandResponse> {
        let mut graphs = self
            .graphs
            .write()
            .map_err(|e| WeavError::Internal(format!("lock poisoned: {e}")))?;
        let gs = graphs
            .get_mut(&cmd.graph)
            .ok_or_else(|| WeavError::GraphNotFound(cmd.graph.clone()))?;

        let now = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        gs.adjacency.invalidate_edge(cmd.edge_id, now)?;

        Ok(CommandResponse::Ok)
    }

    // ── Context query ────────────────────────────────────────────────────

    fn handle_context(
        &self,
        query: weav_query::parser::ContextQuery,
    ) -> WeavResult<CommandResponse> {
        let graphs = self
            .graphs
            .read()
            .map_err(|e| WeavError::Internal(format!("lock poisoned: {e}")))?;
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
