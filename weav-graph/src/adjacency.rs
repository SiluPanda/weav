//! Graph topology storage using adjacency lists.

use std::collections::HashMap;

use roaring::RoaringBitmap;
use smallvec::SmallVec;

use weav_core::error::WeavError;
use weav_core::types::{
    BiTemporal, Direction, EdgeId, LabelId, NodeId, Provenance, Timestamp,
};

/// Metadata stored for each edge.
pub struct EdgeMeta {
    pub source: NodeId,
    pub target: NodeId,
    pub label: LabelId,
    pub temporal: BiTemporal,
    pub provenance: Option<Provenance>,
    pub weight: f32,
    pub token_cost: u16,
}

/// Per-label directed adjacency: maps a node to its neighbors + edge id.
struct DirectedAdjacency {
    adjacency: HashMap<NodeId, SmallVec<[(NodeId, EdgeId); 8]>>,
}

impl DirectedAdjacency {
    fn new() -> Self {
        Self {
            adjacency: HashMap::new(),
        }
    }

    fn add(&mut self, from: NodeId, to: NodeId, edge_id: EdgeId) {
        self.adjacency
            .entry(from)
            .or_default()
            .push((to, edge_id));
    }

    fn remove_edge(&mut self, from: NodeId, edge_id: EdgeId) {
        if let Some(list) = self.adjacency.get_mut(&from) {
            list.retain(|entry| entry.1 != edge_id);
            if list.is_empty() {
                self.adjacency.remove(&from);
            }
        }
    }

    fn neighbors(&self, node: NodeId) -> &[(NodeId, EdgeId)] {
        self.adjacency
            .get(&node)
            .map(|sv| sv.as_slice())
            .unwrap_or(&[])
    }

    fn degree(&self, node: NodeId) -> u32 {
        self.adjacency
            .get(&node)
            .map(|sv| sv.len() as u32)
            .unwrap_or(0)
    }
}

/// The main adjacency store managing graph topology.
pub struct AdjacencyStore {
    forward: HashMap<LabelId, DirectedAdjacency>,
    backward: HashMap<LabelId, DirectedAdjacency>,
    edge_meta: HashMap<EdgeId, EdgeMeta>,
    node_bitmap: RoaringBitmap,
    next_edge_id: u64,
}

impl AdjacencyStore {
    pub fn new() -> Self {
        Self {
            forward: HashMap::new(),
            backward: HashMap::new(),
            edge_meta: HashMap::new(),
            node_bitmap: RoaringBitmap::new(),
            next_edge_id: 1,
        }
    }

    pub fn add_node(&mut self, node_id: NodeId) {
        self.node_bitmap.insert(node_id as u32);
    }

    pub fn remove_node(&mut self, node_id: NodeId) -> Result<(), WeavError> {
        if !self.has_node(node_id) {
            return Err(WeavError::NodeNotFound(node_id, 0));
        }
        // Collect all edge ids connected to this node
        let mut edge_ids_to_remove: Vec<EdgeId> = Vec::new();
        for fwd in self.forward.values() {
            for &(_, eid) in fwd.neighbors(node_id) {
                edge_ids_to_remove.push(eid);
            }
        }
        for bwd in self.backward.values() {
            for &(_, eid) in bwd.neighbors(node_id) {
                edge_ids_to_remove.push(eid);
            }
        }
        edge_ids_to_remove.sort_unstable();
        edge_ids_to_remove.dedup();

        // Remove edges from adjacency structures and edge_meta
        for eid in &edge_ids_to_remove {
            if let Some(meta) = self.edge_meta.remove(eid) {
                if let Some(fwd) = self.forward.get_mut(&meta.label) {
                    fwd.remove_edge(meta.source, *eid);
                }
                if let Some(bwd) = self.backward.get_mut(&meta.label) {
                    bwd.remove_edge(meta.target, *eid);
                }
            }
        }

        self.node_bitmap.remove(node_id as u32);
        Ok(())
    }

    pub fn add_edge(
        &mut self,
        src: NodeId,
        tgt: NodeId,
        label: LabelId,
        meta: EdgeMeta,
    ) -> Result<EdgeId, WeavError> {
        if !self.has_node(src) {
            return Err(WeavError::NodeNotFound(src, 0));
        }
        if !self.has_node(tgt) {
            return Err(WeavError::NodeNotFound(tgt, 0));
        }
        let edge_id = self.next_edge_id;
        self.next_edge_id += 1;

        self.forward
            .entry(label)
            .or_insert_with(DirectedAdjacency::new)
            .add(src, tgt, edge_id);
        self.backward
            .entry(label)
            .or_insert_with(DirectedAdjacency::new)
            .add(tgt, src, edge_id);

        self.edge_meta.insert(edge_id, meta);
        Ok(edge_id)
    }

    pub fn remove_edge(&mut self, edge_id: EdgeId) -> Result<(), WeavError> {
        let meta = self
            .edge_meta
            .remove(&edge_id)
            .ok_or(WeavError::EdgeNotFound(edge_id))?;
        if let Some(fwd) = self.forward.get_mut(&meta.label) {
            fwd.remove_edge(meta.source, edge_id);
        }
        if let Some(bwd) = self.backward.get_mut(&meta.label) {
            bwd.remove_edge(meta.target, edge_id);
        }
        Ok(())
    }

    pub fn invalidate_edge(
        &mut self,
        edge_id: EdgeId,
        invalid_at: Timestamp,
    ) -> Result<(), WeavError> {
        let meta = self
            .edge_meta
            .get_mut(&edge_id)
            .ok_or(WeavError::EdgeNotFound(edge_id))?;
        meta.temporal.invalidate(invalid_at);
        Ok(())
    }

    pub fn neighbors_out(
        &self,
        node: NodeId,
        label: Option<LabelId>,
    ) -> Vec<(NodeId, EdgeId)> {
        match label {
            Some(l) => self
                .forward
                .get(&l)
                .map(|fwd| fwd.neighbors(node).to_vec())
                .unwrap_or_default(),
            None => {
                let mut result = Vec::new();
                for fwd in self.forward.values() {
                    result.extend_from_slice(fwd.neighbors(node));
                }
                result
            }
        }
    }

    pub fn neighbors_in(
        &self,
        node: NodeId,
        label: Option<LabelId>,
    ) -> Vec<(NodeId, EdgeId)> {
        match label {
            Some(l) => self
                .backward
                .get(&l)
                .map(|bwd| bwd.neighbors(node).to_vec())
                .unwrap_or_default(),
            None => {
                let mut result = Vec::new();
                for bwd in self.backward.values() {
                    result.extend_from_slice(bwd.neighbors(node));
                }
                result
            }
        }
    }

    pub fn neighbors_both(
        &self,
        node: NodeId,
        label: Option<LabelId>,
    ) -> Vec<(NodeId, EdgeId, Direction)> {
        let mut result = Vec::new();
        for (n, e) in self.neighbors_out(node, label) {
            result.push((n, e, Direction::Outgoing));
        }
        for (n, e) in self.neighbors_in(node, label) {
            result.push((n, e, Direction::Incoming));
        }
        result
    }

    pub fn edge_between(
        &self,
        src: NodeId,
        tgt: NodeId,
        label: Option<LabelId>,
    ) -> Option<EdgeId> {
        let neighbors = self.neighbors_out(src, label);
        neighbors
            .into_iter()
            .find(|&(n, _)| n == tgt)
            .map(|(_, eid)| eid)
    }

    pub fn degree_out(&self, node: NodeId) -> u32 {
        let mut total = 0u32;
        for fwd in self.forward.values() {
            total += fwd.degree(node);
        }
        total
    }

    pub fn degree_in(&self, node: NodeId) -> u32 {
        let mut total = 0u32;
        for bwd in self.backward.values() {
            total += bwd.degree(node);
        }
        total
    }

    pub fn node_count(&self) -> u64 {
        self.node_bitmap.len()
    }

    pub fn edge_count(&self) -> u64 {
        self.edge_meta.len() as u64
    }

    pub fn has_node(&self, node_id: NodeId) -> bool {
        self.node_bitmap.contains(node_id as u32)
    }

    pub fn get_edge(&self, edge_id: EdgeId) -> Option<&EdgeMeta> {
        self.edge_meta.get(&edge_id)
    }

    pub fn neighbors_at(
        &self,
        node: NodeId,
        timestamp: Timestamp,
        label: Option<LabelId>,
    ) -> Vec<(NodeId, EdgeId)> {
        self.neighbors_out(node, label)
            .into_iter()
            .filter(|&(_, eid)| {
                self.edge_meta
                    .get(&eid)
                    .map(|m| m.temporal.is_valid_at(timestamp))
                    .unwrap_or(false)
            })
            .collect()
    }

    pub fn all_node_ids(&self) -> Vec<NodeId> {
        self.node_bitmap.iter().map(|id| id as NodeId).collect()
    }

    pub fn all_edges(&self) -> impl Iterator<Item = (EdgeId, &EdgeMeta)> {
        self.edge_meta.iter().map(|(id, meta)| (*id, meta))
    }

    pub fn edge_history(&self, src: NodeId, tgt: NodeId) -> Vec<&EdgeMeta> {
        self.edge_meta
            .values()
            .filter(|m| m.source == src && m.target == tgt)
            .collect()
    }
}

impl Default for AdjacencyStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use weav_core::types::BiTemporal;

    fn make_meta(src: NodeId, tgt: NodeId, label: LabelId) -> EdgeMeta {
        EdgeMeta {
            source: src,
            target: tgt,
            label,
            temporal: BiTemporal::new_current(1000),
            provenance: None,
            weight: 1.0,
            token_cost: 0,
        }
    }

    #[test]
    fn test_add_and_has_node() {
        let mut store = AdjacencyStore::new();
        assert!(!store.has_node(1));
        store.add_node(1);
        assert!(store.has_node(1));
        assert_eq!(store.node_count(), 1);
    }

    #[test]
    fn test_add_edge_requires_nodes() {
        let mut store = AdjacencyStore::new();
        store.add_node(1);
        let meta = make_meta(1, 2, 0);
        assert!(store.add_edge(1, 2, 0, meta).is_err());
    }

    #[test]
    fn test_add_and_query_edges() {
        let mut store = AdjacencyStore::new();
        store.add_node(1);
        store.add_node(2);
        store.add_node(3);

        let meta1 = make_meta(1, 2, 0);
        let eid1 = store.add_edge(1, 2, 0, meta1).unwrap();
        let meta2 = make_meta(1, 3, 0);
        let eid2 = store.add_edge(1, 3, 0, meta2).unwrap();

        assert_ne!(eid1, eid2);
        assert_eq!(store.edge_count(), 2);

        let out = store.neighbors_out(1, None);
        assert_eq!(out.len(), 2);

        let inc = store.neighbors_in(2, None);
        assert_eq!(inc.len(), 1);
        assert_eq!(inc[0].0, 1);
    }

    #[test]
    fn test_neighbors_with_label_filter() {
        let mut store = AdjacencyStore::new();
        store.add_node(1);
        store.add_node(2);
        store.add_node(3);

        let meta1 = make_meta(1, 2, 0);
        store.add_edge(1, 2, 0, meta1).unwrap();
        let meta2 = make_meta(1, 3, 1);
        store.add_edge(1, 3, 1, meta2).unwrap();

        assert_eq!(store.neighbors_out(1, Some(0)).len(), 1);
        assert_eq!(store.neighbors_out(1, Some(1)).len(), 1);
        assert_eq!(store.neighbors_out(1, None).len(), 2);
    }

    #[test]
    fn test_remove_edge() {
        let mut store = AdjacencyStore::new();
        store.add_node(1);
        store.add_node(2);

        let meta = make_meta(1, 2, 0);
        let eid = store.add_edge(1, 2, 0, meta).unwrap();
        assert_eq!(store.edge_count(), 1);

        store.remove_edge(eid).unwrap();
        assert_eq!(store.edge_count(), 0);
        assert!(store.neighbors_out(1, None).is_empty());
    }

    #[test]
    fn test_remove_node_removes_edges() {
        let mut store = AdjacencyStore::new();
        store.add_node(1);
        store.add_node(2);
        store.add_node(3);

        let meta1 = make_meta(1, 2, 0);
        store.add_edge(1, 2, 0, meta1).unwrap();
        let meta2 = make_meta(3, 1, 0);
        store.add_edge(3, 1, 0, meta2).unwrap();

        assert_eq!(store.edge_count(), 2);
        store.remove_node(1).unwrap();
        assert!(!store.has_node(1));
        assert_eq!(store.edge_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_node() {
        let mut store = AdjacencyStore::new();
        assert!(store.remove_node(99).is_err());
    }

    #[test]
    fn test_degree() {
        let mut store = AdjacencyStore::new();
        store.add_node(1);
        store.add_node(2);
        store.add_node(3);

        let meta1 = make_meta(1, 2, 0);
        store.add_edge(1, 2, 0, meta1).unwrap();
        let meta2 = make_meta(1, 3, 0);
        store.add_edge(1, 3, 0, meta2).unwrap();
        let meta3 = make_meta(3, 1, 0);
        store.add_edge(3, 1, 0, meta3).unwrap();

        assert_eq!(store.degree_out(1), 2);
        assert_eq!(store.degree_in(1), 1);
        assert_eq!(store.degree_out(2), 0);
        assert_eq!(store.degree_in(2), 1);
    }

    #[test]
    fn test_edge_between() {
        let mut store = AdjacencyStore::new();
        store.add_node(1);
        store.add_node(2);

        let meta = make_meta(1, 2, 0);
        let eid = store.add_edge(1, 2, 0, meta).unwrap();

        assert_eq!(store.edge_between(1, 2, None), Some(eid));
        assert_eq!(store.edge_between(2, 1, None), None);
        assert_eq!(store.edge_between(1, 2, Some(0)), Some(eid));
        assert_eq!(store.edge_between(1, 2, Some(1)), None);
    }

    #[test]
    fn test_neighbors_both() {
        let mut store = AdjacencyStore::new();
        store.add_node(1);
        store.add_node(2);
        store.add_node(3);

        let meta1 = make_meta(1, 2, 0);
        store.add_edge(1, 2, 0, meta1).unwrap();
        let meta2 = make_meta(3, 1, 0);
        store.add_edge(3, 1, 0, meta2).unwrap();

        let both = store.neighbors_both(1, None);
        assert_eq!(both.len(), 2);

        let out_count = both.iter().filter(|x| x.2 == Direction::Outgoing).count();
        let in_count = both.iter().filter(|x| x.2 == Direction::Incoming).count();
        assert_eq!(out_count, 1);
        assert_eq!(in_count, 1);
    }

    #[test]
    fn test_invalidate_edge() {
        let mut store = AdjacencyStore::new();
        store.add_node(1);
        store.add_node(2);

        let meta = make_meta(1, 2, 0);
        let eid = store.add_edge(1, 2, 0, meta).unwrap();

        store.invalidate_edge(eid, 2000).unwrap();
        let edge = store.get_edge(eid).unwrap();
        assert_eq!(edge.temporal.valid_until, 2000);
        assert!(edge.temporal.is_valid_at(1500));
        assert!(!edge.temporal.is_valid_at(2500));
    }

    #[test]
    fn test_neighbors_at() {
        let mut store = AdjacencyStore::new();
        store.add_node(1);
        store.add_node(2);
        store.add_node(3);

        let meta1 = make_meta(1, 2, 0);
        let eid1 = store.add_edge(1, 2, 0, meta1).unwrap();
        let meta2 = make_meta(1, 3, 0);
        store.add_edge(1, 3, 0, meta2).unwrap();

        // Invalidate edge to node 2 at time 1500
        store.invalidate_edge(eid1, 1500).unwrap();

        // At time 1200, both should be valid
        let at_1200 = store.neighbors_at(1, 1200, None);
        assert_eq!(at_1200.len(), 2);

        // At time 1800, only edge to node 3 should be valid
        let at_1800 = store.neighbors_at(1, 1800, None);
        assert_eq!(at_1800.len(), 1);
        assert_eq!(at_1800[0].0, 3);
    }

    #[test]
    fn test_edge_history() {
        let mut store = AdjacencyStore::new();
        store.add_node(1);
        store.add_node(2);

        let meta1 = make_meta(1, 2, 0);
        let eid1 = store.add_edge(1, 2, 0, meta1).unwrap();
        store.invalidate_edge(eid1, 1500).unwrap();

        let meta2 = EdgeMeta {
            source: 1,
            target: 2,
            label: 0,
            temporal: BiTemporal::new_current(1500),
            provenance: None,
            weight: 1.0,
            token_cost: 0,
        };
        store.add_edge(1, 2, 0, meta2).unwrap();

        let history = store.edge_history(1, 2);
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_get_edge() {
        let mut store = AdjacencyStore::new();
        store.add_node(1);
        store.add_node(2);

        let meta = make_meta(1, 2, 5);
        let eid = store.add_edge(1, 2, 5, meta).unwrap();

        let edge = store.get_edge(eid).unwrap();
        assert_eq!(edge.source, 1);
        assert_eq!(edge.target, 2);
        assert_eq!(edge.label, 5);
        assert_eq!(edge.weight, 1.0);

        assert!(store.get_edge(999).is_none());
    }

    #[test]
    fn test_remove_nonexistent_edge() {
        let mut store = AdjacencyStore::new();
        store.add_node(1);
        store.add_node(2);

        // Add one edge so the store is not empty
        let meta = make_meta(1, 2, 0);
        store.add_edge(1, 2, 0, meta).unwrap();

        // Try to remove an edge_id that does not exist
        let result = store.remove_edge(9999);
        assert!(result.is_err());

        // Existing edge should still be intact
        assert_eq!(store.edge_count(), 1);
    }

    #[test]
    fn test_neighbors_both_with_label() {
        let mut store = AdjacencyStore::new();
        store.add_node(1);
        store.add_node(2);
        store.add_node(3);
        store.add_node(4);

        // Label 0: 1 -> 2 (outgoing from 1)
        let meta1 = make_meta(1, 2, 0);
        store.add_edge(1, 2, 0, meta1).unwrap();

        // Label 1: 1 -> 3 (outgoing from 1, different label)
        let meta2 = make_meta(1, 3, 1);
        store.add_edge(1, 3, 1, meta2).unwrap();

        // Label 0: 4 -> 1 (incoming to 1)
        let meta3 = make_meta(4, 1, 0);
        store.add_edge(4, 1, 0, meta3).unwrap();

        // Label 1: neighbors_both for node 1 with label=Some(1) should only return node 3 (outgoing)
        let both_label1 = store.neighbors_both(1, Some(1));
        assert_eq!(both_label1.len(), 1);
        assert_eq!(both_label1[0].0, 3);
        assert_eq!(both_label1[0].2, Direction::Outgoing);

        // Label 0: neighbors_both for node 1 with label=Some(0) should return node 2 (out) and node 4 (in)
        let both_label0 = store.neighbors_both(1, Some(0));
        assert_eq!(both_label0.len(), 2);
        let out_nodes: Vec<NodeId> = both_label0
            .iter()
            .filter(|x| x.2 == Direction::Outgoing)
            .map(|x| x.0)
            .collect();
        let in_nodes: Vec<NodeId> = both_label0
            .iter()
            .filter(|x| x.2 == Direction::Incoming)
            .map(|x| x.0)
            .collect();
        assert_eq!(out_nodes, vec![2]);
        assert_eq!(in_nodes, vec![4]);

        // No label filter: neighbors_both for node 1 should return all 3 neighbors
        let both_all = store.neighbors_both(1, None);
        assert_eq!(both_all.len(), 3);
    }
}
