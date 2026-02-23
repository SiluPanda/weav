//! Entity deduplication: exact key matching, fuzzy name matching, and property merging.

use weav_core::types::{ConflictPolicy, NodeId, Value};

use crate::properties::PropertyStore;

/// Configuration for deduplication behavior.
pub struct DedupConfig {
    pub exact_key_field: Option<String>,
    pub name_field: Option<String>,
    pub fuzzy_threshold: f32,
    pub vector_threshold: f32,
    pub require_same_label: bool,
}

impl Default for DedupConfig {
    fn default() -> Self {
        Self {
            exact_key_field: None,
            name_field: None,
            fuzzy_threshold: 0.85,
            vector_threshold: 0.92,
            require_same_label: true,
        }
    }
}

/// Result of a property merge operation.
pub enum MergeResult {
    Merged {
        node_id: NodeId,
        conflicts: Vec<PropertyConflict>,
    },
    NoChange {
        node_id: NodeId,
    },
}

/// A conflict detected during property merging.
pub struct PropertyConflict {
    pub key: String,
    pub existing_value: String,
    pub new_value: String,
}

/// Find a duplicate node by exact key match.
pub fn find_duplicate_by_key(
    properties: &PropertyStore,
    key_field: &str,
    key_value: &str,
) -> Option<NodeId> {
    let nodes = properties.nodes_where(key_field, &|v| {
        v.as_str().map(|s| s == key_value).unwrap_or(false)
    });
    nodes.into_iter().next()
}

/// Find a duplicate node by fuzzy name matching using Jaro-Winkler similarity.
pub fn find_duplicate_by_name(
    properties: &PropertyStore,
    name_field: &str,
    name_value: &str,
    threshold: f32,
) -> Option<(NodeId, f32)> {
    let nodes = properties.nodes_with_property(name_field);
    let mut best: Option<(NodeId, f32)> = None;

    for node_id in nodes {
        if let Some(val) = properties.get_node_property(node_id, name_field) {
            if let Some(existing_name) = val.as_str() {
                let similarity = strsim::jaro_winkler(name_value, existing_name) as f32;
                if similarity >= threshold {
                    match best {
                        Some((_, best_score)) if similarity > best_score => {
                            best = Some((node_id, similarity));
                        }
                        None => {
                            best = Some((node_id, similarity));
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    best
}

/// Merge properties from new data into an existing node.
pub fn merge_properties(
    properties: &mut PropertyStore,
    existing: NodeId,
    new_props: &[(String, Value)],
    policy: &ConflictPolicy,
) -> MergeResult {
    let mut conflicts = Vec::new();
    let mut changed = false;

    for (key, new_value) in new_props {
        let existing_value = properties.get_node_property(existing, key).cloned();

        match existing_value {
            None => {
                // No existing value, just set it
                properties.set_node_property(existing, key, new_value.clone());
                changed = true;
            }
            Some(ref ev) if ev == new_value => {
                // Same value, no conflict
            }
            Some(ev) => {
                // Conflict
                match policy {
                    ConflictPolicy::LastWriteWins => {
                        properties.set_node_property(existing, key, new_value.clone());
                        changed = true;
                    }
                    ConflictPolicy::Reject => {
                        conflicts.push(PropertyConflict {
                            key: key.clone(),
                            existing_value: format!("{:?}", ev),
                            new_value: format!("{:?}", new_value),
                        });
                    }
                    ConflictPolicy::Merge => {
                        // For merge policy, new values win but we record conflicts
                        properties.set_node_property(existing, key, new_value.clone());
                        conflicts.push(PropertyConflict {
                            key: key.clone(),
                            existing_value: format!("{:?}", ev),
                            new_value: format!("{:?}", new_value),
                        });
                        changed = true;
                    }
                    ConflictPolicy::HighestConfidence | ConflictPolicy::TemporalInvalidation => {
                        // Without provenance context, fall back to last-write-wins
                        properties.set_node_property(existing, key, new_value.clone());
                        changed = true;
                    }
                }
            }
        }
    }

    if changed || !conflicts.is_empty() {
        MergeResult::Merged {
            node_id: existing,
            conflicts,
        }
    } else {
        MergeResult::NoChange { node_id: existing }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use compact_str::CompactString;
    use weav_core::types::Value;

    #[test]
    fn test_find_duplicate_by_key_found() {
        let mut props = PropertyStore::new();
        props.set_node_property(1, "email", Value::String(CompactString::from("alice@example.com")));
        props.set_node_property(2, "email", Value::String(CompactString::from("bob@example.com")));

        let found = find_duplicate_by_key(&props, "email", "alice@example.com");
        assert_eq!(found, Some(1));
    }

    #[test]
    fn test_find_duplicate_by_key_not_found() {
        let mut props = PropertyStore::new();
        props.set_node_property(1, "email", Value::String(CompactString::from("alice@example.com")));

        let found = find_duplicate_by_key(&props, "email", "nobody@example.com");
        assert_eq!(found, None);
    }

    #[test]
    fn test_find_duplicate_by_key_wrong_field() {
        let mut props = PropertyStore::new();
        props.set_node_property(1, "email", Value::String(CompactString::from("alice@example.com")));

        let found = find_duplicate_by_key(&props, "phone", "alice@example.com");
        assert_eq!(found, None);
    }

    #[test]
    fn test_find_duplicate_by_name_exact() {
        let mut props = PropertyStore::new();
        props.set_node_property(1, "name", Value::String(CompactString::from("Alice Johnson")));
        props.set_node_property(2, "name", Value::String(CompactString::from("Bob Smith")));

        let found = find_duplicate_by_name(&props, "name", "Alice Johnson", 0.85);
        assert!(found.is_some());
        let (node_id, score) = found.unwrap();
        assert_eq!(node_id, 1);
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_find_duplicate_by_name_fuzzy() {
        let mut props = PropertyStore::new();
        props.set_node_property(1, "name", Value::String(CompactString::from("Alice Johnson")));

        let found = find_duplicate_by_name(&props, "name", "Alice Jonson", 0.85);
        assert!(found.is_some());
        let (_, score) = found.unwrap();
        assert!(score >= 0.85);
    }

    #[test]
    fn test_find_duplicate_by_name_below_threshold() {
        let mut props = PropertyStore::new();
        props.set_node_property(1, "name", Value::String(CompactString::from("Alice Johnson")));

        let found = find_duplicate_by_name(&props, "name", "Completely Different Name", 0.85);
        assert!(found.is_none());
    }

    #[test]
    fn test_merge_properties_no_conflict() {
        let mut props = PropertyStore::new();
        props.set_node_property(1, "name", Value::String(CompactString::from("Alice")));

        let new_props = vec![("age".to_string(), Value::Int(30))];
        let result = merge_properties(&mut props, 1, &new_props, &ConflictPolicy::LastWriteWins);

        match result {
            MergeResult::Merged { node_id, conflicts } => {
                assert_eq!(node_id, 1);
                assert!(conflicts.is_empty());
            }
            MergeResult::NoChange { .. } => panic!("Expected Merged"),
        }
        assert_eq!(props.get_node_property(1, "age"), Some(&Value::Int(30)));
    }

    #[test]
    fn test_merge_properties_same_value() {
        let mut props = PropertyStore::new();
        props.set_node_property(1, "name", Value::String(CompactString::from("Alice")));

        let new_props = vec![(
            "name".to_string(),
            Value::String(CompactString::from("Alice")),
        )];
        let result = merge_properties(&mut props, 1, &new_props, &ConflictPolicy::LastWriteWins);

        match result {
            MergeResult::NoChange { node_id } => assert_eq!(node_id, 1),
            MergeResult::Merged { .. } => panic!("Expected NoChange"),
        }
    }

    #[test]
    fn test_merge_properties_last_write_wins() {
        let mut props = PropertyStore::new();
        props.set_node_property(1, "name", Value::String(CompactString::from("Alice")));

        let new_props = vec![(
            "name".to_string(),
            Value::String(CompactString::from("Alicia")),
        )];
        let result = merge_properties(&mut props, 1, &new_props, &ConflictPolicy::LastWriteWins);

        match result {
            MergeResult::Merged { conflicts, .. } => {
                assert!(conflicts.is_empty());
            }
            MergeResult::NoChange { .. } => panic!("Expected Merged"),
        }
        assert_eq!(
            props.get_node_property(1, "name"),
            Some(&Value::String(CompactString::from("Alicia")))
        );
    }

    #[test]
    fn test_merge_properties_reject() {
        let mut props = PropertyStore::new();
        props.set_node_property(1, "name", Value::String(CompactString::from("Alice")));

        let new_props = vec![(
            "name".to_string(),
            Value::String(CompactString::from("Alicia")),
        )];
        let result = merge_properties(&mut props, 1, &new_props, &ConflictPolicy::Reject);

        match result {
            MergeResult::Merged { conflicts, .. } => {
                assert_eq!(conflicts.len(), 1);
                assert_eq!(conflicts[0].key, "name");
            }
            MergeResult::NoChange { .. } => panic!("Expected Merged with conflicts"),
        }
        // Original value should be preserved under Reject policy
        assert_eq!(
            props.get_node_property(1, "name"),
            Some(&Value::String(CompactString::from("Alice")))
        );
    }

    #[test]
    fn test_merge_properties_merge_policy() {
        let mut props = PropertyStore::new();
        props.set_node_property(1, "name", Value::String(CompactString::from("Alice")));

        let new_props = vec![(
            "name".to_string(),
            Value::String(CompactString::from("Alicia")),
        )];
        let result = merge_properties(&mut props, 1, &new_props, &ConflictPolicy::Merge);

        match result {
            MergeResult::Merged { conflicts, .. } => {
                assert_eq!(conflicts.len(), 1);
            }
            MergeResult::NoChange { .. } => panic!("Expected Merged"),
        }
        // Under Merge, new value wins but conflict is recorded
        assert_eq!(
            props.get_node_property(1, "name"),
            Some(&Value::String(CompactString::from("Alicia")))
        );
    }

    #[test]
    fn test_dedup_config_defaults() {
        let config = DedupConfig::default();
        assert_eq!(config.fuzzy_threshold, 0.85);
        assert_eq!(config.vector_threshold, 0.92);
        assert!(config.require_same_label);
        assert!(config.exact_key_field.is_none());
        assert!(config.name_field.is_none());
    }
}
