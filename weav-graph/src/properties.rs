//! Property storage for nodes and edges.

use std::collections::HashMap;

use weav_core::types::{EdgeId, NodeId, PropertyKeyId, Value, ValueType};

/// Column-oriented storage for a single property key across all nodes.
pub struct PropertyColumn {
    pub key_id: PropertyKeyId,
    pub value_type: Option<ValueType>,
    pub values: HashMap<NodeId, Value>,
}

/// Property store managing node and edge properties.
pub struct PropertyStore {
    schema: HashMap<String, PropertyKeyId>,
    schema_reverse: HashMap<PropertyKeyId, String>,
    next_key_id: PropertyKeyId,
    node_columns: HashMap<PropertyKeyId, PropertyColumn>,
    #[allow(dead_code)]
    edge_overflow: HashMap<EdgeId, Vec<(PropertyKeyId, Value)>>,
}

impl PropertyStore {
    pub fn new() -> Self {
        Self {
            schema: HashMap::new(),
            schema_reverse: HashMap::new(),
            next_key_id: 1,
            node_columns: HashMap::new(),
            edge_overflow: HashMap::new(),
        }
    }

    /// Intern a property key name, returning its compact id.
    /// Returns existing id if already interned.
    pub fn intern_key(&mut self, key: &str) -> PropertyKeyId {
        if let Some(&id) = self.schema.get(key) {
            return id;
        }
        let id = self.next_key_id;
        self.next_key_id += 1;
        self.schema.insert(key.to_string(), id);
        self.schema_reverse.insert(id, key.to_string());
        id
    }

    pub fn set_node_property(&mut self, node: NodeId, key: &str, value: Value) {
        let key_id = self.intern_key(key);
        let column = self
            .node_columns
            .entry(key_id)
            .or_insert_with(|| PropertyColumn {
                key_id,
                value_type: None,
                values: HashMap::new(),
            });
        column.values.insert(node, value);
    }

    pub fn get_node_property(&self, node: NodeId, key: &str) -> Option<&Value> {
        let key_id = self.schema.get(key)?;
        let column = self.node_columns.get(key_id)?;
        column.values.get(&node)
    }

    pub fn get_all_node_properties(&self, node: NodeId) -> Vec<(&str, &Value)> {
        let mut result = Vec::new();
        for (key_id, column) in &self.node_columns {
            if let Some(val) = column.values.get(&node) {
                if let Some(key_name) = self.schema_reverse.get(key_id) {
                    result.push((key_name.as_str(), val));
                }
            }
        }
        result
    }

    pub fn remove_node_property(&mut self, node: NodeId, key: &str) {
        if let Some(&key_id) = self.schema.get(key) {
            if let Some(column) = self.node_columns.get_mut(&key_id) {
                column.values.remove(&node);
            }
        }
    }

    pub fn remove_all_node_properties(&mut self, node: NodeId) {
        for column in self.node_columns.values_mut() {
            column.values.remove(&node);
        }
    }

    pub fn set_node_properties_batch(&mut self, node: NodeId, props: Vec<(String, Value)>) {
        for (key, value) in props {
            self.set_node_property(node, &key, value);
        }
    }

    pub fn nodes_with_property(&self, key: &str) -> Vec<NodeId> {
        if let Some(&key_id) = self.schema.get(key) {
            if let Some(column) = self.node_columns.get(&key_id) {
                return column.values.keys().copied().collect();
            }
        }
        Vec::new()
    }

    pub fn nodes_where(&self, key: &str, predicate: &dyn Fn(&Value) -> bool) -> Vec<NodeId> {
        if let Some(&key_id) = self.schema.get(key) {
            if let Some(column) = self.node_columns.get(&key_id) {
                return column
                    .values
                    .iter()
                    .filter(|(_, v)| predicate(v))
                    .map(|(&nid, _)| nid)
                    .collect();
            }
        }
        Vec::new()
    }

    /// Rough token estimate: sum of string lengths / 4 for all properties on a node.
    pub fn estimate_tokens(&self, node: NodeId) -> u32 {
        let mut total_chars: u32 = 0;
        for column in self.node_columns.values() {
            if let Some(val) = column.values.get(&node) {
                total_chars += value_char_len(val);
            }
        }
        total_chars / 4
    }
}

fn value_char_len(val: &Value) -> u32 {
    match val {
        Value::Null => 4,
        Value::Bool(b) => if *b { 4 } else { 5 },
        Value::Int(i) => i.to_string().len() as u32,
        Value::Float(f) => f.to_string().len() as u32,
        Value::String(s) => s.len() as u32,
        Value::Bytes(b) => b.len() as u32,
        Value::Vector(v) => (v.len() * 6) as u32, // rough estimate
        Value::List(items) => items.iter().map(value_char_len).sum(),
        Value::Map(pairs) => pairs
            .iter()
            .map(|(k, v)| k.len() as u32 + value_char_len(v))
            .sum(),
        Value::Timestamp(_) => 13,
    }
}

impl Default for PropertyStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use compact_str::CompactString;

    #[test]
    fn test_intern_key() {
        let mut store = PropertyStore::new();
        let id1 = store.intern_key("name");
        let id2 = store.intern_key("name");
        let id3 = store.intern_key("age");
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_set_and_get_property() {
        let mut store = PropertyStore::new();
        store.set_node_property(1, "name", Value::String(CompactString::from("Alice")));
        store.set_node_property(1, "age", Value::Int(30));

        assert_eq!(
            store.get_node_property(1, "name"),
            Some(&Value::String(CompactString::from("Alice")))
        );
        assert_eq!(store.get_node_property(1, "age"), Some(&Value::Int(30)));
        assert_eq!(store.get_node_property(1, "missing"), None);
        assert_eq!(store.get_node_property(2, "name"), None);
    }

    #[test]
    fn test_get_all_node_properties() {
        let mut store = PropertyStore::new();
        store.set_node_property(1, "name", Value::String(CompactString::from("Alice")));
        store.set_node_property(1, "age", Value::Int(30));
        store.set_node_property(2, "name", Value::String(CompactString::from("Bob")));

        let props = store.get_all_node_properties(1);
        assert_eq!(props.len(), 2);

        let props2 = store.get_all_node_properties(2);
        assert_eq!(props2.len(), 1);

        let props3 = store.get_all_node_properties(99);
        assert!(props3.is_empty());
    }

    #[test]
    fn test_remove_node_property() {
        let mut store = PropertyStore::new();
        store.set_node_property(1, "name", Value::String(CompactString::from("Alice")));
        store.set_node_property(1, "age", Value::Int(30));

        store.remove_node_property(1, "name");
        assert_eq!(store.get_node_property(1, "name"), None);
        assert_eq!(store.get_node_property(1, "age"), Some(&Value::Int(30)));
    }

    #[test]
    fn test_remove_all_node_properties() {
        let mut store = PropertyStore::new();
        store.set_node_property(1, "name", Value::String(CompactString::from("Alice")));
        store.set_node_property(1, "age", Value::Int(30));

        store.remove_all_node_properties(1);
        assert!(store.get_all_node_properties(1).is_empty());
    }

    #[test]
    fn test_set_node_properties_batch() {
        let mut store = PropertyStore::new();
        store.set_node_properties_batch(
            1,
            vec![
                ("name".to_string(), Value::String(CompactString::from("Alice"))),
                ("age".to_string(), Value::Int(30)),
            ],
        );

        assert_eq!(
            store.get_node_property(1, "name"),
            Some(&Value::String(CompactString::from("Alice")))
        );
        assert_eq!(store.get_node_property(1, "age"), Some(&Value::Int(30)));
    }

    #[test]
    fn test_nodes_with_property() {
        let mut store = PropertyStore::new();
        store.set_node_property(1, "name", Value::String(CompactString::from("Alice")));
        store.set_node_property(2, "name", Value::String(CompactString::from("Bob")));
        store.set_node_property(3, "age", Value::Int(25));

        let mut nodes = store.nodes_with_property("name");
        nodes.sort();
        assert_eq!(nodes, vec![1, 2]);

        assert!(store.nodes_with_property("missing").is_empty());
    }

    #[test]
    fn test_nodes_where() {
        let mut store = PropertyStore::new();
        store.set_node_property(1, "age", Value::Int(20));
        store.set_node_property(2, "age", Value::Int(30));
        store.set_node_property(3, "age", Value::Int(40));

        let mut adults = store.nodes_where("age", &|v| {
            v.as_int().map(|i| i >= 30).unwrap_or(false)
        });
        adults.sort();
        assert_eq!(adults, vec![2, 3]);
    }

    #[test]
    fn test_estimate_tokens() {
        let mut store = PropertyStore::new();
        // "Hello" = 5 chars, 5/4 = 1 token
        store.set_node_property(
            1,
            "greeting",
            Value::String(CompactString::from("Hello World")),
        );
        let tokens = store.estimate_tokens(1);
        assert!(tokens > 0);

        assert_eq!(store.estimate_tokens(99), 0);
    }

    #[test]
    fn test_overwrite_property() {
        let mut store = PropertyStore::new();
        store.set_node_property(1, "name", Value::String(CompactString::from("Alice")));
        store.set_node_property(1, "name", Value::String(CompactString::from("Bob")));

        assert_eq!(
            store.get_node_property(1, "name"),
            Some(&Value::String(CompactString::from("Bob")))
        );
    }
}
