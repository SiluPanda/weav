//! Shard infrastructure for the Weav engine.
//!
//! Provides the core shard types that form the foundation of Weav's
//! thread-per-core architecture with keyspace sharding.

use bumpalo::Bump;

use crate::config::GraphConfig;
use crate::types::{GraphId, LabelId, PropertyKeyId, ShardId, Timestamp};
use compact_str::CompactString;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;

/// A single shard owning a portion of the keyspace.
///
/// Each shard has its own bump allocator for arena-based allocation,
/// a string interner for compact label/property-key encoding, and a
/// set of per-graph metadata entries.
pub struct Shard {
    pub id: ShardId,
    graphs: HashMap<GraphId, GraphShard>,
    string_interner: StringInterner,
    stats: ShardStats,
    allocator: Bump,
}

/// Lightweight per-graph metadata within a shard.
///
/// `GraphShard` in `weav-core` is intentionally a metadata-only struct.
/// The actual graph storage (adjacency lists, property stores, vector
/// indices, temporal and provenance indices) is managed by the
/// [`Engine`](../../weav_server/engine/struct.Engine.html) in `weav-server`,
/// which composes types from `weav-graph` and `weav-vector`. This
/// separation keeps `weav-core` free of upward dependencies while still
/// providing a place for per-graph bookkeeping at the shard level.
pub struct GraphShard {
    pub graph_id: GraphId,
    pub graph_name: CompactString,
    pub config: GraphConfig,
    /// Cached node count for this graph within this shard.
    pub node_count: u64,
    /// Cached edge count for this graph within this shard.
    pub edge_count: u64,
    /// Timestamp (ms since epoch) when this graph shard was created.
    pub created_at: Timestamp,
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
            allocator: Bump::new(),
        }
    }

    /// Returns a reference to this shard's bump allocator.
    ///
    /// The bump allocator provides fast arena-based allocation for
    /// temporary per-request data within the shard.
    pub fn allocator(&self) -> &Bump {
        &self.allocator
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
        let created_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as Timestamp;
        Self {
            graph_id,
            graph_name: name,
            config,
            node_count: 0,
            edge_count: 0,
            created_at,
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

    #[test]
    fn test_interner_get_label_id() {
        let mut interner = StringInterner::new();
        // Non-existent label returns None
        assert_eq!(interner.get_label_id("missing"), None);

        let id = interner.intern_label("Person");
        assert_eq!(interner.get_label_id("Person"), Some(id));
        assert_eq!(interner.get_label_id("NotInterned"), None);
    }

    #[test]
    fn test_interner_get_property_key_id() {
        let mut interner = StringInterner::new();
        // Non-existent property key returns None
        assert_eq!(interner.get_property_key_id("missing"), None);

        let id = interner.intern_property_key("name");
        assert_eq!(interner.get_property_key_id("name"), Some(id));
        assert_eq!(interner.get_property_key_id("age"), None);
    }

    #[test]
    fn test_interner_resolve_nonexistent() {
        let interner = StringInterner::new();
        assert_eq!(interner.resolve_label(999), None);
        assert_eq!(interner.resolve_property_key(999), None);
    }

    #[test]
    fn test_shard_accessors() {
        let mut shard = Shard::new(7);
        assert_eq!(shard.id, 7);

        // allocator() returns a reference
        let _alloc = shard.allocator();

        // interner() returns immutable reference
        let _interner = shard.interner();

        // interner_mut() returns mutable reference - intern something
        let id = shard.interner_mut().intern_label("TestLabel");
        assert_eq!(shard.interner().resolve_label(id), Some("TestLabel"));

        // stats() returns reference
        let stats = shard.stats();
        assert_eq!(
            stats.node_count.load(std::sync::atomic::Ordering::Relaxed),
            0
        );
    }

    #[test]
    fn test_shard_get_graph_mut() {
        let mut shard = Shard::new(0);
        let graph = GraphShard::new(1, "test".into(), GraphConfig::default());
        shard.insert_graph(graph);

        // Verify we can get a mutable reference and modify
        let g = shard.get_graph_mut(1).expect("graph 1 should exist");
        assert_eq!(g.node_count, 0);
        g.node_count = 42;
        g.edge_count = 10;

        // Verify mutation persisted
        let g2 = shard.get_graph(1).unwrap();
        assert_eq!(g2.node_count, 42);
        assert_eq!(g2.edge_count, 10);

        // Non-existent graph returns None
        assert!(shard.get_graph_mut(999).is_none());
    }

    #[test]
    fn test_shard_remove_nonexistent_graph() {
        let mut shard = Shard::new(0);
        let result = shard.remove_graph(999);
        assert!(result.is_none());
    }

    #[test]
    fn test_shard_graph_ids_multiple() {
        let mut shard = Shard::new(0);
        shard.insert_graph(GraphShard::new(10, "g10".into(), GraphConfig::default()));
        shard.insert_graph(GraphShard::new(20, "g20".into(), GraphConfig::default()));
        shard.insert_graph(GraphShard::new(30, "g30".into(), GraphConfig::default()));

        let mut ids = shard.graph_ids();
        ids.sort();
        assert_eq!(ids, vec![10, 20, 30]);
    }

    #[test]
    fn test_shard_graph_ids_empty() {
        let shard = Shard::new(0);
        assert!(shard.graph_ids().is_empty());
    }

    #[test]
    fn test_shard_stats_new() {
        let stats = ShardStats::new();
        use std::sync::atomic::Ordering::Relaxed;
        assert_eq!(stats.node_count.load(Relaxed), 0);
        assert_eq!(stats.edge_count.load(Relaxed), 0);
        assert_eq!(stats.memory_bytes.load(Relaxed), 0);
        assert_eq!(stats.query_count.load(Relaxed), 0);
        assert_eq!(stats.avg_query_us.load(Relaxed), 0);
    }

    // ── Round 2 edge-case tests ──────────────────────────────────────────

    #[test]
    fn test_string_interner_empty_string_label() {
        let mut interner = StringInterner::new();
        let id = interner.intern_label("");
        assert_eq!(interner.resolve_label(id), Some(""));
        // Re-interning returns same ID
        assert_eq!(interner.intern_label(""), id);
    }

    #[test]
    fn test_string_interner_empty_string_property() {
        let mut interner = StringInterner::new();
        let id = interner.intern_property_key("");
        assert_eq!(interner.resolve_property_key(id), Some(""));
        assert_eq!(interner.intern_property_key(""), id);
    }

    #[test]
    fn test_string_interner_case_sensitivity() {
        let mut interner = StringInterner::new();
        let id_upper = interner.intern_label("Person");
        let id_lower = interner.intern_label("person");
        assert_ne!(id_upper, id_lower, "interning is case-sensitive");
        assert_eq!(interner.resolve_label(id_upper), Some("Person"));
        assert_eq!(interner.resolve_label(id_lower), Some("person"));
    }

    #[test]
    fn test_string_interner_special_characters() {
        let mut interner = StringInterner::new();
        let id1 = interner.intern_label("hello world");
        let id2 = interner.intern_label("key/with/slashes");
        let id3 = interner.intern_label("emoji_test");
        let id4 = interner.intern_property_key("prop with spaces");

        assert_eq!(interner.resolve_label(id1), Some("hello world"));
        assert_eq!(interner.resolve_label(id2), Some("key/with/slashes"));
        assert_eq!(interner.resolve_label(id3), Some("emoji_test"));
        assert_eq!(interner.resolve_property_key(id4), Some("prop with spaces"));
    }

    #[test]
    fn test_string_interner_many_labels_monotonic() {
        let mut interner = StringInterner::new();
        let mut ids = Vec::new();
        for i in 0..1000 {
            let id = interner.intern_label(&format!("label_{i}"));
            ids.push(id);
        }
        // IDs should be monotonically increasing
        for w in ids.windows(2) {
            assert!(w[1] > w[0], "IDs should be monotonically increasing");
        }
        // Verify round-trip for a sample
        assert_eq!(interner.resolve_label(ids[0]), Some("label_0"));
        assert_eq!(interner.resolve_label(ids[999]), Some("label_999"));
    }

    #[test]
    fn test_graph_shard_count_increment() {
        let mut gs = GraphShard::new(1, "test".into(), GraphConfig::default());
        assert_eq!(gs.node_count, 0);
        assert_eq!(gs.edge_count, 0);

        gs.node_count += 100;
        gs.edge_count += 50;
        assert_eq!(gs.node_count, 100);
        assert_eq!(gs.edge_count, 50);
    }

    #[test]
    fn test_shard_insert_replace_graph() {
        let mut shard = Shard::new(0);
        shard.insert_graph(GraphShard::new(1, "original".into(), GraphConfig::default()));
        assert_eq!(shard.get_graph(1).unwrap().graph_name.as_str(), "original");

        // Insert with same ID replaces
        shard.insert_graph(GraphShard::new(1, "replaced".into(), GraphConfig::default()));
        assert_eq!(shard.get_graph(1).unwrap().graph_name.as_str(), "replaced");
        assert_eq!(shard.graph_ids().len(), 1);
    }

    #[test]
    fn test_shard_multiple_interners_independent() {
        let mut shard_a = Shard::new(0);
        let mut shard_b = Shard::new(1);

        let id_a = shard_a.interner_mut().intern_label("Person");
        let id_b = shard_b.interner_mut().intern_label("Person");

        // Both start from 0, so IDs should be the same value
        assert_eq!(id_a, id_b);
        assert_eq!(id_a, 0);
    }
}
