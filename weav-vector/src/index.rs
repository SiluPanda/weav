use std::collections::HashMap;

use roaring::RoaringBitmap;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
use weav_core::error::WeavError;
use weav_core::types::NodeId;

/// Distance metric for vector similarity search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
}

/// Quantization level for stored vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantization {
    /// Full f32 precision.
    None,
    /// Half-precision f16.
    F16,
    /// 8-bit signed integer quantization.
    I8,
}

/// Configuration for the HNSW vector index.
#[derive(Clone, Debug)]
pub struct VectorConfig {
    pub dimensions: u16,
    pub metric: DistanceMetric,
    pub hnsw_m: u16,
    pub hnsw_ef_construction: u16,
    pub hnsw_ef_search: u16,
    pub quantization: Quantization,
}

impl Default for VectorConfig {
    fn default() -> Self {
        Self {
            dimensions: 1536,
            metric: DistanceMetric::Cosine,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            quantization: Quantization::None,
        }
    }
}

/// HNSW vector index wrapping the `usearch` crate.
pub struct VectorIndex {
    inner: Index,
    dimensions: u16,
    metric: DistanceMetric,
    node_to_key: HashMap<NodeId, u64>,
    key_to_node: HashMap<u64, NodeId>,
    next_key: u64,
    config: VectorConfig,
}

impl VectorIndex {
    /// Create a new vector index with the given configuration.
    pub fn new(config: VectorConfig) -> Result<Self, WeavError> {
        let metric = match config.metric {
            DistanceMetric::Cosine => MetricKind::Cos,
            DistanceMetric::Euclidean => MetricKind::L2sq,
            DistanceMetric::DotProduct => MetricKind::IP,
        };

        let quantization = match config.quantization {
            Quantization::None => ScalarKind::F32,
            Quantization::F16 => ScalarKind::F16,
            Quantization::I8 => ScalarKind::I8,
        };

        let options = IndexOptions {
            dimensions: config.dimensions as usize,
            metric,
            quantization,
            connectivity: config.hnsw_m as usize,
            expansion_add: config.hnsw_ef_construction as usize,
            expansion_search: config.hnsw_ef_search as usize,
            multi: false,
        };

        let inner = Index::new(&options)
            .map_err(|e| WeavError::Internal(format!("failed to create usearch index: {e}")))?;

        Ok(Self {
            inner,
            dimensions: config.dimensions,
            metric: config.metric,
            node_to_key: HashMap::new(),
            key_to_node: HashMap::new(),
            next_key: 0,
            config,
        })
    }

    /// Insert a vector for the given node.
    pub fn insert(&mut self, node_id: NodeId, vector: &[f32]) -> Result<(), WeavError> {
        if vector.len() != self.dimensions as usize {
            return Err(WeavError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len() as u16,
            });
        }

        let key = self.next_key;
        self.next_key += 1;

        // Reserve capacity if needed
        let needed = self.inner.size() + 1;
        if needed > self.inner.capacity() {
            self.inner
                .reserve(needed.max(self.inner.capacity() * 2).max(64))
                .map_err(|e| WeavError::Internal(format!("failed to reserve capacity: {e}")))?;
        }

        self.inner
            .add(key, vector)
            .map_err(|e| WeavError::Internal(format!("failed to insert vector: {e}")))?;

        self.node_to_key.insert(node_id, key);
        self.key_to_node.insert(key, node_id);
        Ok(())
    }

    /// Update the vector for an existing node (remove old + insert new).
    pub fn update(&mut self, node_id: NodeId, vector: &[f32]) -> Result<(), WeavError> {
        if vector.len() != self.dimensions as usize {
            return Err(WeavError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len() as u16,
            });
        }

        // Remove old vector if it exists
        if let Some(&old_key) = self.node_to_key.get(&node_id) {
            let _ = self.inner.remove(old_key);
            self.key_to_node.remove(&old_key);
            self.node_to_key.remove(&node_id);
        }

        // Insert new vector
        self.insert(node_id, vector)
    }

    /// Remove a node's vector from the index.
    pub fn remove(&mut self, node_id: NodeId) -> Result<(), WeavError> {
        if let Some(key) = self.node_to_key.remove(&node_id) {
            self.inner
                .remove(key)
                .map_err(|e| WeavError::Internal(format!("failed to remove vector: {e}")))?;
            self.key_to_node.remove(&key);
            Ok(())
        } else {
            Ok(())
        }
    }

    /// Search for the k nearest neighbors of the query vector.
    pub fn search(
        &self,
        query: &[f32],
        k: u16,
        ef_search: Option<u16>,
    ) -> Result<Vec<(NodeId, f32)>, WeavError> {
        if query.len() != self.dimensions as usize {
            return Err(WeavError::DimensionMismatch {
                expected: self.dimensions,
                got: query.len() as u16,
            });
        }

        if self.inner.size() == 0 {
            return Ok(Vec::new());
        }

        // Temporarily change ef_search if specified
        let original_ef = if let Some(ef) = ef_search {
            let orig = self.inner.expansion_search();
            self.inner.change_expansion_search(ef as usize);
            Some(orig)
        } else {
            None
        };

        let matches = self
            .inner
            .search(query, k as usize)
            .map_err(|e| WeavError::Internal(format!("search failed: {e}")))?;

        // Restore original ef_search
        if let Some(orig) = original_ef {
            self.inner.change_expansion_search(orig);
        }

        let results = matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .filter_map(|(&key, &dist)| {
                self.key_to_node.get(&key).map(|&node_id| (node_id, dist))
            })
            .collect();

        Ok(results)
    }

    /// Search with a pre-filter: only return results whose NodeId is in `candidates`.
    /// Uses usearch's native filtered_search with a predicate callback.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: u16,
        candidates: &RoaringBitmap,
    ) -> Result<Vec<(NodeId, f32)>, WeavError> {
        if query.len() != self.dimensions as usize {
            return Err(WeavError::DimensionMismatch {
                expected: self.dimensions,
                got: query.len() as u16,
            });
        }

        if self.inner.size() == 0 {
            return Ok(Vec::new());
        }

        // Use usearch's native filtered search with a predicate
        let filter = |key: u64| -> bool {
            if let Some(&node_id) = self.key_to_node.get(&key) {
                // NodeId is u64, RoaringBitmap stores u32, so we check if it fits
                if node_id <= u32::MAX as u64 {
                    candidates.contains(node_id as u32)
                } else {
                    false
                }
            } else {
                false
            }
        };

        let matches = self
            .inner
            .filtered_search(query, k as usize, filter)
            .map_err(|e| WeavError::Internal(format!("filtered search failed: {e}")))?;

        let results = matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .filter_map(|(&key, &dist)| {
                self.key_to_node.get(&key).map(|&node_id| (node_id, dist))
            })
            .collect();

        Ok(results)
    }

    /// Returns the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.node_to_key.len()
    }

    /// Returns true if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.node_to_key.is_empty()
    }

    /// Returns the number of dimensions for vectors in this index.
    pub fn dimensions(&self) -> u16 {
        self.dimensions
    }

    /// Returns true if the index contains a vector for the given node.
    pub fn contains(&self, node_id: NodeId) -> bool {
        self.node_to_key.contains_key(&node_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(dims: u16) -> VectorConfig {
        VectorConfig {
            dimensions: dims,
            metric: DistanceMetric::Cosine,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            quantization: Quantization::None,
        }
    }

    #[test]
    fn test_new_index() {
        let config = make_config(4);
        let index = VectorIndex::new(config).unwrap();
        assert_eq!(index.dimensions(), 4);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn test_insert_and_search() {
        let config = make_config(4);
        let mut index = VectorIndex::new(config).unwrap();

        // Insert some vectors
        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        index.insert(3, &[1.0, 1.0, 0.0, 0.0]).unwrap();

        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());
        assert!(index.contains(1));
        assert!(index.contains(2));
        assert!(index.contains(3));
        assert!(!index.contains(99));

        // Search for a vector close to node 1
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 3, None).unwrap();
        assert!(!results.is_empty());
        // The first result should be node 1 (exact match, distance ~0)
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 < 0.01);
    }

    #[test]
    fn test_dimension_mismatch_insert() {
        let config = make_config(4);
        let mut index = VectorIndex::new(config).unwrap();

        let err = index.insert(1, &[1.0, 0.0]).unwrap_err();
        match err {
            WeavError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 4);
                assert_eq!(got, 2);
            }
            _ => panic!("expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_dimension_mismatch_search() {
        let config = make_config(4);
        let index = VectorIndex::new(config).unwrap();

        let err = index.search(&[1.0, 0.0], 5, None).unwrap_err();
        match err {
            WeavError::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 4);
                assert_eq!(got, 2);
            }
            _ => panic!("expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_update() {
        let config = make_config(4);
        let mut index = VectorIndex::new(config).unwrap();

        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(index.len(), 1);

        // Update node 1 with a new vector
        index.update(1, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert_eq!(index.len(), 1);
        assert!(index.contains(1));

        // Search for the updated vector
        let results = index.search(&[0.0, 1.0, 0.0, 0.0], 1, None).unwrap();
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 < 0.01);
    }

    #[test]
    fn test_remove() {
        let config = make_config(4);
        let mut index = VectorIndex::new(config).unwrap();

        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert_eq!(index.len(), 2);

        index.remove(1).unwrap();
        assert_eq!(index.len(), 1);
        assert!(!index.contains(1));
        assert!(index.contains(2));

        // Removing a non-existent node is a no-op
        index.remove(99).unwrap();
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_search_empty_index() {
        let config = make_config(4);
        let index = VectorIndex::new(config).unwrap();

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 5, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_ordering() {
        let config = VectorConfig {
            dimensions: 3,
            metric: DistanceMetric::Euclidean,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            quantization: Quantization::None,
        };
        let mut index = VectorIndex::new(config).unwrap();

        // Insert vectors at known distances from origin-ish query
        index.insert(10, &[0.1, 0.0, 0.0]).unwrap(); // closest
        index.insert(20, &[0.5, 0.0, 0.0]).unwrap(); // middle
        index.insert(30, &[1.0, 0.0, 0.0]).unwrap(); // farthest

        let results = index.search(&[0.0, 0.0, 0.0], 3, None).unwrap();
        assert_eq!(results.len(), 3);
        // Results should be in ascending distance order
        assert!(results[0].1 <= results[1].1);
        assert!(results[1].1 <= results[2].1);
        // Closest should be node 10
        assert_eq!(results[0].0, 10);
    }

    #[test]
    fn test_search_with_ef_override() {
        let config = make_config(4);
        let mut index = VectorIndex::new(config).unwrap();

        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();

        // Search with custom ef_search
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2, Some(100)).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_filtered_search() {
        let config = make_config(4);
        let mut index = VectorIndex::new(config).unwrap();

        // Insert nodes 1..5
        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[0.9, 0.1, 0.0, 0.0]).unwrap();
        index.insert(3, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        index.insert(4, &[0.0, 0.0, 1.0, 0.0]).unwrap();

        // Only allow nodes 3 and 4 in the candidates
        let mut candidates = RoaringBitmap::new();
        candidates.insert(3);
        candidates.insert(4);

        let results = index
            .search_filtered(&[1.0, 0.0, 0.0, 0.0], 4, &candidates)
            .unwrap();

        // All results should be in the candidate set
        for (node_id, _dist) in &results {
            assert!(
                candidates.contains(*node_id as u32),
                "node {} should be in candidates",
                node_id
            );
        }
    }

    #[test]
    fn test_different_metrics() {
        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
        ] {
            let config = VectorConfig {
                dimensions: 4,
                metric,
                ..Default::default()
            };
            let mut index = VectorIndex::new(config).unwrap();
            index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
            let results = index.search(&[1.0, 0.0, 0.0, 0.0], 1, None).unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, 1);
        }
    }

    #[test]
    fn test_update_dimension_mismatch() {
        let config = make_config(4);
        let mut index = VectorIndex::new(config).unwrap();
        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();

        let err = index.update(1, &[1.0, 0.0]).unwrap_err();
        match err {
            WeavError::DimensionMismatch { .. } => {}
            _ => panic!("expected DimensionMismatch"),
        }
    }
}
