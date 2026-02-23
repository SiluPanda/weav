//! Shard infrastructure — stub for Phase 1 agent to complete.

// This module will be fully implemented by the Phase 1 agent.
// For now, provide minimal types so other crates can reference them.

use crate::config::GraphConfig;
use crate::types::{GraphId, LabelId, PropertyKeyId, ShardId};
use compact_str::CompactString;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;

/// A single shard owning a portion of the keyspace.
pub struct Shard {
    pub id: ShardId,
    graphs: HashMap<GraphId, GraphShard>,
    string_interner: StringInterner,
    stats: ShardStats,
}

/// Per-graph data within a shard.
pub struct GraphShard {
    pub graph_id: GraphId,
    pub graph_name: CompactString,
    pub config: GraphConfig,
    // topology, properties, vector_index, temporal_index, provenance_index
    // will be added as Phase 2 and Phase 3 types are implemented.
}

/// Bidirectional string interner for labels and property keys.
pub struct StringInterner {
    label_to_id: HashMap<CompactString, LabelId>,
    id_to_label: HashMap<LabelId, CompactString>,
    prop_to_id: HashMap<CompactString, PropertyKeyId>,
    id_to_prop: HashMap<PropertyKeyId, CompactString>,
    next_label_id: LabelId,
    next_prop_id: PropertyKeyId,
}

impl StringInterner {
    pub fn new() -> Self {
        Self {
            label_to_id: HashMap::new(),
            id_to_label: HashMap::new(),
            prop_to_id: HashMap::new(),
            id_to_prop: HashMap::new(),
            next_label_id: 0,
            next_prop_id: 0,
        }
    }

    /// Intern a label string, returning its compact ID.
    /// If already interned, returns the existing ID.
    pub fn intern_label(&mut self, label: &str) -> LabelId {
        if let Some(&id) = self.label_to_id.get(label) {
            return id;
        }
        let id = self.next_label_id;
        self.next_label_id += 1;
        let cs = CompactString::new(label);
        self.label_to_id.insert(cs.clone(), id);
        self.id_to_label.insert(id, cs);
        id
    }

    /// Look up the string for a label ID.
    pub fn resolve_label(&self, id: LabelId) -> Option<&str> {
        self.id_to_label.get(&id).map(|s| s.as_str())
    }

    /// Intern a property key string, returning its compact ID.
    pub fn intern_property_key(&mut self, key: &str) -> PropertyKeyId {
        if let Some(&id) = self.prop_to_id.get(key) {
            return id;
        }
        let id = self.next_prop_id;
        self.next_prop_id += 1;
        let cs = CompactString::new(key);
        self.prop_to_id.insert(cs.clone(), id);
        self.id_to_prop.insert(id, cs);
        id
    }

    /// Look up the string for a property key ID.
    pub fn resolve_property_key(&self, id: PropertyKeyId) -> Option<&str> {
        self.id_to_prop.get(&id).map(|s| s.as_str())
    }

    /// Get label ID without interning (returns None if not found).
    pub fn get_label_id(&self, label: &str) -> Option<LabelId> {
        self.label_to_id.get(label).copied()
    }

    /// Get property key ID without interning (returns None if not found).
    pub fn get_property_key_id(&self, key: &str) -> Option<PropertyKeyId> {
        self.prop_to_id.get(key).copied()
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

/// Lock-free shard statistics.
pub struct ShardStats {
    pub node_count: AtomicU64,
    pub edge_count: AtomicU64,
    pub memory_bytes: AtomicU64,
    pub query_count: AtomicU64,
    pub avg_query_us: AtomicU64,
}

impl ShardStats {
    pub fn new() -> Self {
        Self {
            node_count: AtomicU64::new(0),
            edge_count: AtomicU64::new(0),
            memory_bytes: AtomicU64::new(0),
            query_count: AtomicU64::new(0),
            avg_query_us: AtomicU64::new(0),
        }
    }
}

impl Default for ShardStats {
    fn default() -> Self {
        Self::new()
    }
}

impl Shard {
    pub fn new(id: ShardId) -> Self {
        Self {
            id,
            graphs: HashMap::new(),
            string_interner: StringInterner::new(),
            stats: ShardStats::new(),
        }
    }

    /// Get a reference to the string interner.
    pub fn interner(&self) -> &StringInterner {
        &self.string_interner
    }

    /// Get a mutable reference to the string interner.
    pub fn interner_mut(&mut self) -> &mut StringInterner {
        &mut self.string_interner
    }

    /// Get a reference to the shard stats.
    pub fn stats(&self) -> &ShardStats {
        &self.stats
    }

    /// Get a reference to a graph by ID.
    pub fn get_graph(&self, graph_id: GraphId) -> Option<&GraphShard> {
        self.graphs.get(&graph_id)
    }

    /// Get a mutable reference to a graph by ID.
    pub fn get_graph_mut(&mut self, graph_id: GraphId) -> Option<&mut GraphShard> {
        self.graphs.get_mut(&graph_id)
    }

    /// Insert a new graph into this shard.
    pub fn insert_graph(&mut self, graph: GraphShard) {
        self.graphs.insert(graph.graph_id, graph);
    }

    /// Remove a graph from this shard.
    pub fn remove_graph(&mut self, graph_id: GraphId) -> Option<GraphShard> {
        self.graphs.remove(&graph_id)
    }

    /// List all graph IDs in this shard.
    pub fn graph_ids(&self) -> Vec<GraphId> {
        self.graphs.keys().copied().collect()
    }
}

impl GraphShard {
    pub fn new(graph_id: GraphId, name: CompactString, config: GraphConfig) -> Self {
        Self {
            graph_id,
            graph_name: name,
            config,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_interner_labels() {
        let mut interner = StringInterner::new();
        let id1 = interner.intern_label("entity");
        let id2 = interner.intern_label("relationship");
        let id3 = interner.intern_label("entity"); // duplicate

        assert_eq!(id1, id3);
        assert_ne!(id1, id2);
        assert_eq!(interner.resolve_label(id1), Some("entity"));
        assert_eq!(interner.resolve_label(id2), Some("relationship"));
    }

    #[test]
    fn test_string_interner_properties() {
        let mut interner = StringInterner::new();
        let id1 = interner.intern_property_key("name");
        let id2 = interner.intern_property_key("type");
        let id3 = interner.intern_property_key("name");

        assert_eq!(id1, id3);
        assert_ne!(id1, id2);
        assert_eq!(interner.resolve_property_key(id1), Some("name"));
    }

    #[test]
    fn test_shard_graph_management() {
        let mut shard = Shard::new(0);
        let graph = GraphShard::new(1, "test".into(), GraphConfig::default());
        shard.insert_graph(graph);

        assert!(shard.get_graph(1).is_some());
        assert!(shard.get_graph(99).is_none());
        assert_eq!(shard.graph_ids(), vec![1]);

        let removed = shard.remove_graph(1);
        assert!(removed.is_some());
        assert!(shard.get_graph(1).is_none());
    }
}
