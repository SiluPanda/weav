use std::collections::HashMap;

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
    node_to_key: HashMap<NodeId, u64>,
    key_to_node: HashMap<u64, NodeId>,
    /// Raw vector storage for snapshot persistence.
    /// USearch HNSW doesn't support vector retrieval, so we store copies here.
    vectors: HashMap<NodeId, Vec<f32>>,
    next_key: u64,
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
            .map_err(|_| WeavError::Internal("failed to initialize vector index".into()))?;

        Ok(Self {
            inner,
            dimensions: config.dimensions,
            node_to_key: HashMap::new(),
            key_to_node: HashMap::new(),
            vectors: HashMap::new(),
            next_key: 0,
        })
    }

    /// Validate that a vector has the expected number of dimensions.
    fn validate_dims(&self, vector: &[f32]) -> Result<(), WeavError> {
        if vector.len() != self.dimensions as usize {
            return Err(WeavError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len() as u16,
            });
        }
        Ok(())
    }

    /// Insert a vector for the given node.
    /// If the node already has a vector, the old one is removed first to avoid orphaned keys.
    pub fn insert(&mut self, node_id: NodeId, vector: &[f32]) -> Result<(), WeavError> {
        self.validate_dims(vector)?;

        // Remove any existing vector for this node to prevent orphaned keys
        if self.node_to_key.contains_key(&node_id) {
            self.remove(node_id)?;
        }

        let key = self.next_key;
        self.next_key += 1;

        // Reserve capacity if needed
        let needed = self.inner.size() + 1;
        if needed > self.inner.capacity() {
            self.inner
                .reserve(needed.max(self.inner.capacity() * 2).max(64))
                .map_err(|_| WeavError::Internal("failed to grow vector index".into()))?;
        }

        self.inner
            .add(key, vector)
            .map_err(|_| WeavError::Internal("failed to insert vector".into()))?;

        self.node_to_key.insert(node_id, key);
        self.key_to_node.insert(key, node_id);
        self.vectors.insert(node_id, vector.to_vec());
        Ok(())
    }

    /// Remove a node's vector from the index.
    pub fn remove(&mut self, node_id: NodeId) -> Result<(), WeavError> {
        if let Some(key) = self.node_to_key.remove(&node_id) {
            self.inner
                .remove(key)
                .map_err(|_| WeavError::Internal("failed to remove vector".into()))?;
            self.key_to_node.remove(&key);
            self.vectors.remove(&node_id);
            Ok(())
        } else {
            Ok(())
        }
    }

    /// Retrieve the raw vector for a node, if stored.
    pub fn get_vector(&self, node_id: NodeId) -> Option<&[f32]> {
        self.vectors.get(&node_id).map(|v| v.as_slice())
    }

    /// Returns an iterator over all (NodeId, vector) pairs.
    pub fn all_vectors(&self) -> impl Iterator<Item = (NodeId, &[f32])> {
        self.vectors.iter().map(|(&nid, v)| (nid, v.as_slice()))
    }

    /// Search for the k nearest neighbors of the query vector.
    pub fn search(
        &self,
        query: &[f32],
        k: u16,
        ef_search: Option<u16>,
    ) -> Result<Vec<(NodeId, f32)>, WeavError> {
        self.validate_dims(query)?;

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
            .map_err(|_| WeavError::Internal("vector search failed".into()))?;

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

    /// Search with post-filtering: retrieve more candidates and apply a predicate.
    ///
    /// This is the standard approach for filtered vector search when the filter
    /// is not integrated into the index structure. We over-fetch by
    /// `oversample_factor` (default 4x) and then keep the top-k that pass.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: u16,
        filter: &dyn Fn(NodeId) -> bool,
        ef_search: Option<u16>,
    ) -> Result<Vec<(NodeId, f32)>, WeavError> {
        self.validate_dims(query)?;

        if self.inner.size() == 0 {
            return Ok(Vec::new());
        }

        // Over-fetch candidates to compensate for filtering
        let oversample = (k as usize * 4).min(self.inner.size());

        let original_ef = if let Some(ef) = ef_search {
            let orig = self.inner.expansion_search();
            self.inner.change_expansion_search(ef as usize);
            Some(orig)
        } else {
            None
        };

        let matches = self
            .inner
            .search(query, oversample)
            .map_err(|_| WeavError::Internal("vector search failed".into()))?;

        if let Some(orig) = original_ef {
            self.inner.change_expansion_search(orig);
        }

        let results: Vec<(NodeId, f32)> = matches
            .keys
            .iter()
            .zip(matches.distances.iter())
            .filter_map(|(&key, &dist)| {
                self.key_to_node
                    .get(&key)
                    .filter(|&&node_id| filter(node_id))
                    .map(|&node_id| (node_id, dist))
            })
            .take(k as usize)
            .collect();

        Ok(results)
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
    fn test_quantization_f16() {
        let config = VectorConfig {
            dimensions: 4,
            metric: DistanceMetric::Cosine,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            quantization: Quantization::F16,
        };
        let mut index = VectorIndex::new(config).unwrap();

        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert_eq!(index.len(), 2);

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2, None).unwrap();
        assert!(!results.is_empty());
        // Nearest neighbor should be node 1 (exact match)
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_quantization_i8() {
        let config = VectorConfig {
            dimensions: 4,
            metric: DistanceMetric::Cosine,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            quantization: Quantization::I8,
        };
        let mut index = VectorIndex::new(config).unwrap();

        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert_eq!(index.len(), 2);

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2, None).unwrap();
        assert!(!results.is_empty());
        // Nearest neighbor should be node 1 (exact match)
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_search_ef_override_restoration() {
        let config = make_config(4);
        let mut index = VectorIndex::new(config).unwrap();

        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        index.insert(3, &[0.7, 0.7, 0.0, 0.0]).unwrap();

        // First search with an ef_search override of 200
        let results_with_override = index
            .search(&[1.0, 0.0, 0.0, 0.0], 3, Some(200))
            .unwrap();
        assert!(!results_with_override.is_empty());
        assert_eq!(results_with_override[0].0, 1);

        // Second search without override - should use the original ef_search (50)
        // and still return correct results, proving the override was restored
        let results_without_override = index
            .search(&[1.0, 0.0, 0.0, 0.0], 3, None)
            .unwrap();
        assert!(!results_without_override.is_empty());
        assert_eq!(results_without_override[0].0, 1);

        // Verify the internal ef_search is back to the original value (50)
        assert_eq!(index.inner.expansion_search(), 50);
    }

    // ── Round 6 edge-case tests ──────────────────────────────────────────

    #[test]
    fn test_insert_all_zero_vector() {
        let config = make_config(4);
        let mut index = VectorIndex::new(config).unwrap();
        index.insert(1, &[0.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_search_k_zero() {
        let config = make_config(4);
        let mut index = VectorIndex::new(config).unwrap();
        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 0, None).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_k_greater_than_index_size() {
        let config = make_config(4);
        let mut index = VectorIndex::new(config).unwrap();
        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 10, None).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_remove_non_existent_node() {
        let config = make_config(4);
        let mut index = VectorIndex::new(config).unwrap();
        // Node 999 was never inserted; remove should return Ok(())
        assert!(index.remove(999).is_ok());
    }

    #[test]
    fn test_insert_after_remove() {
        let config = make_config(4);
        let mut index = VectorIndex::new(config).unwrap();

        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(index.len(), 1);

        index.remove(1).unwrap();
        assert_eq!(index.len(), 0);

        index.insert(1, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert_eq!(index.len(), 1);
        assert!(index.contains(1));
    }

    #[test]
    fn test_euclidean_metric_distance() {
        let config = VectorConfig {
            dimensions: 4,
            metric: DistanceMetric::Euclidean,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            quantization: Quantization::None,
        };
        let mut index = VectorIndex::new(config).unwrap();

        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();

        // Search for vector [1,0,0,0]; nearest is node 1 (distance 0)
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2, None).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 < 0.01); // exact match

        // Node 2 is at L2sq distance = (1-0)^2 + (0-1)^2 + 0 + 0 = 2.0
        assert_eq!(results[1].0, 2);
        assert!((results[1].1 - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_dot_product_metric() {
        let config = VectorConfig {
            dimensions: 4,
            metric: DistanceMetric::DotProduct,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            quantization: Quantization::None,
        };
        let mut index = VectorIndex::new(config).unwrap();

        // Insert two unit-ish vectors
        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.insert(2, &[0.5, 0.5, 0.0, 0.0]).unwrap();

        // Search with [1,0,0,0]. Inner product with node 1 = 1.0, with node 2 = 0.5.
        // usearch uses IP metric where distance = 1 - dot_product for normalized,
        // so node 1 should be the best match.
        let results = index.search(&[1.0, 0.0, 0.0, 0.0], 2, None).unwrap();
        assert_eq!(results.len(), 2);
        // Node 1 should be the closest (highest dot product = lowest distance)
        assert_eq!(results[0].0, 1);
        // The distances should be ordered
        assert!(results[0].1 <= results[1].1);
    }

    #[test]
    fn test_insert_replaces_existing_vector() {
        let config = VectorConfig {
            dimensions: 4,
            metric: DistanceMetric::Cosine,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            quantization: Quantization::None,
        };
        let mut index = VectorIndex::new(config).unwrap();

        // Insert initial vector for node 1
        index.insert(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(index.get_vector(1).is_some());
        assert_eq!(index.len(), 1);

        // Re-insert with a different vector — should replace, not duplicate
        index.insert(1, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert!(index.get_vector(1).is_some());
        assert_eq!(index.len(), 1); // Still only one entry

        // The new vector should be returned
        let vec = index.get_vector(1).unwrap();
        assert_eq!(vec[1], 1.0); // Second dimension should be 1.0 now

        // Search should only return this node once
        let results = index.search(&[0.0, 1.0, 0.0, 0.0], 5, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }
}
