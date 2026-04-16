//! Entity deduplication: exact key matching, fuzzy name matching, and property merging.

use std::collections::{HashMap, HashSet};

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

/// Find a duplicate node by vector similarity.
///
/// Accepts a pre-computed slice of `(NodeId, similarity_score)` pairs (as returned
/// by a vector index search) and a similarity threshold. Returns the NodeId with
/// the highest similarity score that exceeds the threshold, if any.
///
/// This design avoids a circular dependency on `weav-vector`; the caller performs
/// the vector search and passes the results here.
pub fn find_duplicate_by_vector(
    similarity_results: &[(NodeId, f32)],
    threshold: f32,
) -> Option<(NodeId, f32)> {
    similarity_results
        .iter()
        .filter(|(_, score)| *score > threshold)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
}

/// Check if two entity names are fuzzy matches using Jaro-Winkler similarity.
/// Returns true if similarity >= threshold.
pub fn fuzzy_name_match(name_a: &str, name_b: &str, threshold: f64) -> bool {
    if name_a.eq_ignore_ascii_case(name_b) {
        return true; // Exact match (fast path)
    }
    let sim = strsim::jaro_winkler(&name_a.to_lowercase(), &name_b.to_lowercase());
    sim >= threshold
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
        if let Some(val) = properties.get_node_property(node_id, name_field)
            && let Some(existing_name) = val.as_str()
        {
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

    best
}

/// N-gram blocking index for efficient fuzzy duplicate detection.
/// Narrows candidate set before expensive string similarity computation.
pub struct BlockingIndex {
    /// Trigram -> set of node IDs that contain this trigram
    trigram_index: HashMap<String, HashSet<NodeId>>,
    /// Node ID -> stored name (for cleanup on removal)
    node_names: HashMap<NodeId, String>,
}

impl BlockingIndex {
    pub fn new() -> Self {
        Self {
            trigram_index: HashMap::new(),
            node_names: HashMap::new(),
        }
    }

    /// Extract character trigrams from a string (lowercased).
    fn trigrams(s: &str) -> Vec<String> {
        let lower = s.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();
        if chars.len() < 3 {
            return vec![lower];
        }
        chars
            .windows(3)
            .map(|w| w.iter().collect::<String>())
            .collect()
    }

    /// Index a node's name for future blocking lookups.
    pub fn insert(&mut self, node_id: NodeId, name: &str) {
        let trigrams = Self::trigrams(name);
        for tri in &trigrams {
            self.trigram_index
                .entry(tri.clone())
                .or_default()
                .insert(node_id);
        }
        self.node_names.insert(node_id, name.to_string());
    }

    /// Remove a node from the blocking index.
    pub fn remove(&mut self, node_id: NodeId) {
        if let Some(name) = self.node_names.remove(&node_id) {
            let trigrams = Self::trigrams(&name);
            for tri in &trigrams {
                if let Some(set) = self.trigram_index.get_mut(tri) {
                    set.remove(&node_id);
                    if set.is_empty() {
                        self.trigram_index.remove(tri);
                    }
                }
            }
        }
    }

    /// Find candidate nodes that share at least `min_shared_trigrams` trigrams with the query.
    /// Returns candidates sorted by number of shared trigrams (most shared first).
    pub fn candidates(&self, name: &str, min_shared_trigrams: usize) -> Vec<NodeId> {
        let query_trigrams = Self::trigrams(name);
        let mut counts: HashMap<NodeId, usize> = HashMap::new();

        for tri in &query_trigrams {
            if let Some(nodes) = self.trigram_index.get(tri) {
                for &nid in nodes {
                    *counts.entry(nid).or_default() += 1;
                }
            }
        }

        let min_shared = min_shared_trigrams.max(1);
        let mut result: Vec<(NodeId, usize)> = counts
            .into_iter()
            .filter(|&(_, count)| count >= min_shared)
            .collect();

        result.sort_by(|a, b| b.1.cmp(&a.1));
        result.into_iter().map(|(nid, _)| nid).collect()
    }

    pub fn len(&self) -> usize {
        self.node_names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.node_names.is_empty()
    }
}

impl Default for BlockingIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Find a potential duplicate by fuzzy name matching.
/// If a `BlockingIndex` is provided, uses it to narrow candidates (fast path).
/// Otherwise falls back to scanning all nodes (slow path).
pub fn find_duplicate_by_name_indexed(
    properties: &PropertyStore,
    name_field: &str,
    name: &str,
    threshold: f32,
    blocking_index: Option<&BlockingIndex>,
    candidate_limit: Option<usize>,
) -> Option<(NodeId, f32)> {
    // Fast path: use blocking index
    if let Some(index) = blocking_index {
        let candidates = index.candidates(name, 1);
        let mut best: Option<(NodeId, f32)> = None;
        for nid in candidates
            .into_iter()
            .take(candidate_limit.unwrap_or(usize::MAX))
        {
            if let Some(val) = properties.get_node_property(nid, name_field)
                && let Some(existing_name) = val.as_str()
            {
                let score = strsim::jaro_winkler(name, existing_name) as f32;
                if score >= threshold {
                    match &best {
                        Some((_, best_score)) if score <= *best_score => {}
                        _ => best = Some((nid, score)),
                    }
                }
            }
        }
        return best;
    }

    // Slow path: scan all nodes (original behavior)
    find_duplicate_by_name(properties, name_field, name, threshold)
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
                        // These policies require provenance/temporal context not available
                        // at the property merge level. Record the conflict and apply
                        // last-write-wins as a fallback.
                        properties.set_node_property(existing, key, new_value.clone());
                        conflicts.push(PropertyConflict {
                            key: key.clone(),
                            existing_value: format!("{:?}", ev),
                            new_value: format!("{:?}", new_value),
                        });
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
        props.set_node_property(
            1,
            "email",
            Value::String(CompactString::from("alice@example.com")),
        );
        props.set_node_property(
            2,
            "email",
            Value::String(CompactString::from("bob@example.com")),
        );

        let found = find_duplicate_by_key(&props, "email", "alice@example.com");
        assert_eq!(found, Some(1));
    }

    #[test]
    fn test_find_duplicate_by_key_not_found() {
        let mut props = PropertyStore::new();
        props.set_node_property(
            1,
            "email",
            Value::String(CompactString::from("alice@example.com")),
        );

        let found = find_duplicate_by_key(&props, "email", "nobody@example.com");
        assert_eq!(found, None);
    }

    #[test]
    fn test_find_duplicate_by_key_wrong_field() {
        let mut props = PropertyStore::new();
        props.set_node_property(
            1,
            "email",
            Value::String(CompactString::from("alice@example.com")),
        );

        let found = find_duplicate_by_key(&props, "phone", "alice@example.com");
        assert_eq!(found, None);
    }

    #[test]
    fn test_find_duplicate_by_name_exact() {
        let mut props = PropertyStore::new();
        props.set_node_property(
            1,
            "name",
            Value::String(CompactString::from("Alice Johnson")),
        );
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
        props.set_node_property(
            1,
            "name",
            Value::String(CompactString::from("Alice Johnson")),
        );

        let found = find_duplicate_by_name(&props, "name", "Alice Jonson", 0.85);
        assert!(found.is_some());
        let (_, score) = found.unwrap();
        assert!(score >= 0.85);
    }

    #[test]
    fn test_find_duplicate_by_name_below_threshold() {
        let mut props = PropertyStore::new();
        props.set_node_property(
            1,
            "name",
            Value::String(CompactString::from("Alice Johnson")),
        );

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

    #[test]
    fn test_find_duplicate_by_vector_above_threshold() {
        let results = vec![(10, 0.95_f32), (20, 0.85), (30, 0.70)];
        let found = find_duplicate_by_vector(&results, 0.92);
        assert!(found.is_some());
        let (node_id, score) = found.unwrap();
        assert_eq!(node_id, 10);
        assert!((score - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_find_duplicate_by_vector_none_above_threshold() {
        let results = vec![(10, 0.80_f32), (20, 0.75), (30, 0.70)];
        let found = find_duplicate_by_vector(&results, 0.92);
        assert!(found.is_none());
    }

    #[test]
    fn test_find_duplicate_by_vector_picks_highest() {
        let results = vec![(10, 0.93_f32), (20, 0.97), (30, 0.94)];
        let found = find_duplicate_by_vector(&results, 0.92);
        assert!(found.is_some());
        let (node_id, score) = found.unwrap();
        assert_eq!(node_id, 20);
        assert!((score - 0.97).abs() < 0.001);
    }

    #[test]
    fn test_find_duplicate_by_vector_empty_results() {
        let results: Vec<(NodeId, f32)> = vec![];
        let found = find_duplicate_by_vector(&results, 0.92);
        assert!(found.is_none());
    }

    #[test]
    fn test_find_duplicate_by_vector_exact_threshold_not_matched() {
        // Score must be strictly greater than threshold
        let results = vec![(10, 0.92_f32)];
        let found = find_duplicate_by_vector(&results, 0.92);
        assert!(found.is_none());
    }

    #[test]
    fn test_merge_highest_confidence_policy() {
        let mut props = PropertyStore::new();
        props.set_node_property(1, "name", Value::String(CompactString::from("Alice")));

        let new_props = vec![(
            "name".to_string(),
            Value::String(CompactString::from("Alicia")),
        )];
        let result = merge_properties(
            &mut props,
            1,
            &new_props,
            &ConflictPolicy::HighestConfidence,
        );

        // HighestConfidence falls through to last-write-wins in the code
        match result {
            MergeResult::Merged { node_id, conflicts } => {
                assert_eq!(node_id, 1);
                assert_eq!(conflicts.len(), 1); // conflict recorded since policy can't be fully applied
            }
            MergeResult::NoChange { .. } => panic!("Expected Merged"),
        }
        // New value should win (last-write-wins fallback)
        assert_eq!(
            props.get_node_property(1, "name"),
            Some(&Value::String(CompactString::from("Alicia")))
        );
    }

    #[test]
    fn test_merge_temporal_invalidation_policy() {
        let mut props = PropertyStore::new();
        props.set_node_property(1, "status", Value::String(CompactString::from("active")));

        let new_props = vec![(
            "status".to_string(),
            Value::String(CompactString::from("inactive")),
        )];
        let result = merge_properties(
            &mut props,
            1,
            &new_props,
            &ConflictPolicy::TemporalInvalidation,
        );

        // TemporalInvalidation falls through to last-write-wins in the code
        match result {
            MergeResult::Merged { node_id, conflicts } => {
                assert_eq!(node_id, 1);
                assert_eq!(conflicts.len(), 1); // conflict recorded since policy can't be fully applied
            }
            MergeResult::NoChange { .. } => panic!("Expected Merged"),
        }
        // New value should win
        assert_eq!(
            props.get_node_property(1, "status"),
            Some(&Value::String(CompactString::from("inactive")))
        );
    }

    #[test]
    fn test_find_duplicate_empty_results() {
        // No nodes have the "name" property at all
        let props = PropertyStore::new();
        let found = find_duplicate_by_name(&props, "name", "Alice", 0.85);
        assert!(found.is_none());
    }

    #[test]
    fn test_merge_properties_empty_new_props() {
        let mut props = PropertyStore::new();
        props.set_node_property(1, "name", Value::String(CompactString::from("Alice")));

        let new_props: Vec<(String, Value)> = vec![];
        let result = merge_properties(&mut props, 1, &new_props, &ConflictPolicy::LastWriteWins);

        match result {
            MergeResult::NoChange { node_id } => assert_eq!(node_id, 1),
            MergeResult::Merged { .. } => panic!("Expected NoChange"),
        }
        // Original properties should remain unchanged
        assert_eq!(
            props.get_node_property(1, "name"),
            Some(&Value::String(CompactString::from("Alice")))
        );
    }

    // ── Round 4 edge-case tests ──────────────────────────────────────────

    #[test]
    fn test_find_duplicate_by_vector_nan_scores() {
        let results = vec![(10, f32::NAN)];
        // NaN > threshold is always false, so no match
        let found = find_duplicate_by_vector(&results, 0.5);
        assert!(found.is_none());
    }

    #[test]
    fn test_find_duplicate_by_vector_threshold_zero() {
        let results = vec![(10, 0.01_f32), (20, 0.0001)];
        // threshold=0.0, any score > 0.0 matches
        let found = find_duplicate_by_vector(&results, 0.0);
        assert!(found.is_some());
        assert_eq!(found.unwrap().0, 10);
    }

    #[test]
    fn test_find_duplicate_by_vector_threshold_one() {
        let results = vec![(10, 0.99_f32), (20, 1.0)];
        // threshold=1.0, only scores > 1.0 match (impossible for normalized similarity)
        // score=1.0 is NOT > 1.0, so no match
        let found = find_duplicate_by_vector(&results, 1.0);
        assert!(found.is_none());
    }

    #[test]
    fn test_find_duplicate_by_name_empty_string() {
        let mut props = PropertyStore::new();
        props.set_node_property(1, "name", Value::String(CompactString::from("")));

        // Empty string vs empty string: Jaro-Winkler similarity is 1.0
        let found = find_duplicate_by_name(&props, "name", "", 0.85);
        assert!(found.is_some());
        let (_, score) = found.unwrap();
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_find_duplicate_by_name_case_sensitive() {
        let mut props = PropertyStore::new();
        props.set_node_property(1, "name", Value::String(CompactString::from("Alice")));

        // "alice" vs "Alice" - Jaro-Winkler is case-sensitive
        let found = find_duplicate_by_name(&props, "name", "alice", 0.99);
        // Similarity should be < 1.0 due to case difference
        assert!(found.is_none());
    }

    #[test]
    fn test_merge_properties_many_conflicts() {
        let mut props = PropertyStore::new();
        // Set 10 properties on node 1
        for i in 0..10 {
            props.set_node_property(1, &format!("key_{i}"), Value::Int(i));
        }

        // New values all differ
        let new_props: Vec<(String, Value)> = (0..10)
            .map(|i| (format!("key_{i}"), Value::Int(i + 100)))
            .collect();

        let result = merge_properties(&mut props, 1, &new_props, &ConflictPolicy::Reject);
        match result {
            MergeResult::Merged { conflicts, .. } => {
                assert_eq!(conflicts.len(), 10);
            }
            MergeResult::NoChange { .. } => panic!("Expected Merged with conflicts"),
        }
        // Under Reject, original values should be preserved
        assert_eq!(props.get_node_property(1, "key_0"), Some(&Value::Int(0)));
    }

    #[test]
    fn test_merge_properties_mixed_types() {
        let mut props = PropertyStore::new();
        props.set_node_property(1, "count", Value::Int(5));

        let new_props = vec![(
            "count".to_string(),
            Value::String(CompactString::from("five")),
        )];
        let result = merge_properties(&mut props, 1, &new_props, &ConflictPolicy::LastWriteWins);

        match result {
            MergeResult::Merged { conflicts, .. } => {
                assert!(conflicts.is_empty());
            }
            MergeResult::NoChange { .. } => panic!("Expected Merged"),
        }
        // Type changed from Int to String
        assert_eq!(
            props.get_node_property(1, "count"),
            Some(&Value::String(CompactString::from("five")))
        );
    }

    #[test]
    fn test_find_duplicate_by_key_non_string_value() {
        let mut props = PropertyStore::new();
        // email is an Int, not a String
        props.set_node_property(1, "email", Value::Int(42));

        // as_str() returns None for Int, so no match
        let found = find_duplicate_by_key(&props, "email", "42");
        assert!(found.is_none());
    }

    // ── BlockingIndex tests ────────────────────────────────────────────

    #[test]
    fn test_blocking_index_insert_and_candidates() {
        let mut index = BlockingIndex::new();
        index.insert(1, "Albert Einstein");
        index.insert(2, "Albert Ellis");
        index.insert(3, "Marie Curie");

        let candidates = index.candidates("Albert Einsten", 1); // typo
        assert!(candidates.contains(&1)); // shares many trigrams with "Albert Einstein"
        assert!(candidates.contains(&2)); // shares "Alb", "lbe", "ber", "ert"
        // Marie Curie may or may not appear (few/no shared trigrams)
    }

    #[test]
    fn test_blocking_index_remove() {
        let mut index = BlockingIndex::new();
        index.insert(1, "Albert Einstein");
        index.insert(2, "Marie Curie");
        assert_eq!(index.len(), 2);

        index.remove(1);
        assert_eq!(index.len(), 1);

        let candidates = index.candidates("Albert Einstein", 1);
        assert!(!candidates.contains(&1)); // removed
    }

    #[test]
    fn test_blocking_index_short_names() {
        let mut index = BlockingIndex::new();
        index.insert(1, "AI");
        index.insert(2, "ML");

        let candidates = index.candidates("AI", 1);
        assert!(candidates.contains(&1));
        assert!(!candidates.contains(&2));
    }

    #[test]
    fn test_blocking_index_empty() {
        let index = BlockingIndex::new();
        assert!(index.is_empty());
        let candidates = index.candidates("test", 1);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_find_duplicate_by_name_indexed_fast_path() {
        let mut props = PropertyStore::new();
        props.set_node_property(
            1,
            "name",
            Value::String(CompactString::from("Albert Einstein")),
        );
        props.set_node_property(2, "name", Value::String(CompactString::from("Marie Curie")));

        let mut index = BlockingIndex::new();
        index.insert(1, "Albert Einstein");
        index.insert(2, "Marie Curie");

        // Should find Albert Einstein via blocking index
        let found = find_duplicate_by_name_indexed(
            &props,
            "name",
            "Albert Einsten",
            0.85,
            Some(&index),
            None,
        );
        assert!(found.is_some());
        let (node_id, score) = found.unwrap();
        assert_eq!(node_id, 1);
        assert!(score >= 0.85);
    }

    #[test]
    fn test_find_duplicate_by_name_indexed_slow_path() {
        let mut props = PropertyStore::new();
        props.set_node_property(
            1,
            "name",
            Value::String(CompactString::from("Alice Johnson")),
        );

        // No blocking index: falls back to full scan
        let found =
            find_duplicate_by_name_indexed(&props, "name", "Alice Jonson", 0.85, None, None);
        assert!(found.is_some());
        let (node_id, score) = found.unwrap();
        assert_eq!(node_id, 1);
        assert!(score >= 0.85);
    }

    // ── fuzzy_name_match tests ───────────────────────────────────────────

    #[test]
    fn test_fuzzy_match_exact() {
        assert!(fuzzy_name_match("Apple Inc", "Apple Inc", 0.85));
    }

    #[test]
    fn test_fuzzy_match_similar() {
        // "Microsoft Corp" vs "Microsoft Corporation" — high Jaro-Winkler similarity
        assert!(fuzzy_name_match(
            "Microsoft Corp",
            "Microsoft Corporation",
            0.85
        ));
    }

    #[test]
    fn test_fuzzy_match_different() {
        assert!(!fuzzy_name_match("Apple", "Samsung", 0.85));
    }

    #[test]
    fn test_fuzzy_match_case_insensitive() {
        // Exact match fast path is case-insensitive
        assert!(fuzzy_name_match("john DOE", "John Doe", 0.85));
    }

    #[test]
    fn test_fuzzy_threshold() {
        // At a very high threshold, partial matches are rejected
        assert!(!fuzzy_name_match(
            "Microsoft Corp",
            "Microsoft Corporation",
            0.99
        ));
    }

    #[test]
    fn test_fuzzy_match_unicode() {
        assert!(fuzzy_name_match("François", "Francois", 0.80));
    }

    #[test]
    fn test_fuzzy_match_empty_strings() {
        // Empty strings should match each other
        assert!(fuzzy_name_match("", "", 0.85));
        // Empty vs non-empty should not match
        assert!(!fuzzy_name_match("", "something", 0.85));
    }

    #[test]
    fn test_fuzzy_match_very_long_strings() {
        let long_a = "a".repeat(1000);
        let long_b = "a".repeat(999) + "b";
        // Very similar long strings
        assert!(fuzzy_name_match(&long_a, &long_b, 0.85));
    }
}
