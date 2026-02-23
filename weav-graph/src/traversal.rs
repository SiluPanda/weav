//! Graph traversal algorithms: BFS, flow scoring, ego network, shortest path.

use std::collections::{HashMap, HashSet, VecDeque};

use weav_core::types::{Direction, EdgeId, LabelId, NodeId, ScoredNode, Timestamp};

use crate::adjacency::AdjacencyStore;

/// Filter criteria applied to edges during traversal.
pub struct EdgeFilter {
    pub labels: Option<HashSet<LabelId>>,
    pub min_weight: Option<f32>,
    pub max_age_ms: Option<u64>,
    pub min_confidence: Option<f32>,
    pub valid_at: Option<Timestamp>,
}

impl EdgeFilter {
    pub fn none() -> Self {
        Self {
            labels: None,
            min_weight: None,
            max_age_ms: None,
            min_confidence: None,
            valid_at: None,
        }
    }
}

impl Default for EdgeFilter {
    fn default() -> Self {
        Self::none()
    }
}

/// Filter criteria applied to nodes during traversal.
pub struct NodeFilter {
    pub labels: Option<HashSet<LabelId>>,
    pub has_property: Option<Vec<String>>,
    pub valid_at: Option<Timestamp>,
}

impl NodeFilter {
    pub fn none() -> Self {
        Self {
            labels: None,
            has_property: None,
            valid_at: None,
        }
    }
}

impl Default for NodeFilter {
    fn default() -> Self {
        Self::none()
    }
}

/// Result of a BFS traversal.
pub struct TraversalResult {
    pub visited_nodes: Vec<NodeId>,
    pub visited_edges: Vec<EdgeId>,
    pub depth_map: HashMap<NodeId, u8>,
    pub parent_map: HashMap<NodeId, NodeId>,
}

/// A subgraph extracted from the main graph.
pub struct SubGraph {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<EdgeId>,
}

/// Check if an edge passes the filter criteria.
fn edge_passes_filter(adj: &AdjacencyStore, edge_id: EdgeId, filter: &EdgeFilter) -> bool {
    let meta = match adj.get_edge(edge_id) {
        Some(m) => m,
        None => return false,
    };

    if let Some(ref labels) = filter.labels {
        if !labels.contains(&meta.label) {
            return false;
        }
    }

    if let Some(min_w) = filter.min_weight {
        if meta.weight < min_w {
            return false;
        }
    }

    if let Some(ts) = filter.valid_at {
        if !meta.temporal.is_valid_at(ts) {
            return false;
        }
    }

    if let Some(min_conf) = filter.min_confidence {
        match &meta.provenance {
            Some(prov) => {
                if prov.confidence < min_conf {
                    return false;
                }
            }
            None => return false,
        }
    }

    true
}

/// Get neighbors in the specified direction.
fn get_neighbors(
    adj: &AdjacencyStore,
    node: NodeId,
    direction: Direction,
    edge_filter: &EdgeFilter,
) -> Vec<(NodeId, EdgeId)> {
    let raw = match direction {
        Direction::Outgoing => adj.neighbors_out(node, None),
        Direction::Incoming => adj.neighbors_in(node, None),
        Direction::Both => {
            let mut result = adj.neighbors_out(node, None);
            result.extend(adj.neighbors_in(node, None));
            result
        }
    };

    raw.into_iter()
        .filter(|&(_, eid)| edge_passes_filter(adj, eid, edge_filter))
        .collect()
}

/// Breadth-first search from seed nodes.
pub fn bfs(
    adjacency: &AdjacencyStore,
    seeds: &[NodeId],
    max_depth: u8,
    max_nodes: usize,
    edge_filter: &EdgeFilter,
    _node_filter: &NodeFilter,
    direction: Direction,
) -> TraversalResult {
    let mut visited_nodes = Vec::new();
    let mut visited_edges = Vec::new();
    let mut depth_map = HashMap::new();
    let mut parent_map = HashMap::new();
    let mut seen = HashSet::new();
    let mut queue: VecDeque<(NodeId, u8)> = VecDeque::new();

    for &seed in seeds {
        if seen.insert(seed) {
            queue.push_back((seed, 0));
            depth_map.insert(seed, 0);
            visited_nodes.push(seed);
        }
    }

    while let Some((node, depth)) = queue.pop_front() {
        if visited_nodes.len() >= max_nodes {
            break;
        }
        if depth >= max_depth {
            continue;
        }

        let neighbors = get_neighbors(adjacency, node, direction, edge_filter);
        for (neighbor, edge_id) in neighbors {
            if seen.insert(neighbor) {
                let next_depth = depth + 1;
                depth_map.insert(neighbor, next_depth);
                parent_map.insert(neighbor, node);
                visited_nodes.push(neighbor);
                visited_edges.push(edge_id);
                if visited_nodes.len() >= max_nodes {
                    break;
                }
                queue.push_back((neighbor, next_depth));
            } else {
                // Node already visited, but we may still want to record the edge
                if !visited_edges.contains(&edge_id) {
                    visited_edges.push(edge_id);
                }
            }
        }
    }

    TraversalResult {
        visited_nodes,
        visited_edges,
        depth_map,
        parent_map,
    }
}

/// Propagate scores from seed nodes, decaying by alpha per hop.
/// Stops when score falls below theta or max_depth is reached.
pub fn flow_score(
    adjacency: &AdjacencyStore,
    seeds_with_scores: &[(NodeId, f32)],
    alpha: f32,
    theta: f32,
    max_depth: u8,
) -> Vec<ScoredNode> {
    let mut scores: HashMap<NodeId, (f32, u8)> = HashMap::new();
    let mut queue: VecDeque<(NodeId, f32, u8)> = VecDeque::new();

    for &(node, score) in seeds_with_scores {
        scores.insert(node, (score, 0));
        queue.push_back((node, score, 0));
    }

    while let Some((node, score, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }

        let neighbors = adjacency.neighbors_out(node, None);
        for (neighbor, _edge_id) in neighbors {
            let propagated = score * alpha;
            if propagated < theta {
                continue;
            }

            let next_depth = depth + 1;
            let update = match scores.get(&neighbor) {
                Some(&(existing_score, _)) => propagated > existing_score,
                None => true,
            };

            if update {
                scores.insert(neighbor, (propagated, next_depth));
                queue.push_back((neighbor, propagated, next_depth));
            }
        }
    }

    let mut result: Vec<ScoredNode> = scores
        .into_iter()
        .map(|(node_id, (score, depth))| ScoredNode {
            node_id,
            score,
            depth,
        })
        .collect();
    result.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    result
}

/// Extract the ego network around a center node up to the given radius.
pub fn ego_network(
    adjacency: &AdjacencyStore,
    center: NodeId,
    radius: u8,
) -> SubGraph {
    let result = bfs(
        adjacency,
        &[center],
        radius,
        usize::MAX,
        &EdgeFilter::none(),
        &NodeFilter::none(),
        Direction::Both,
    );
    SubGraph {
        nodes: result.visited_nodes,
        edges: result.visited_edges,
    }
}

/// Find the shortest path between source and target using BFS.
/// Returns the path as a list of node ids including source and target.
pub fn shortest_path(
    adjacency: &AdjacencyStore,
    source: NodeId,
    target: NodeId,
    max_depth: u8,
) -> Option<Vec<NodeId>> {
    if source == target {
        return Some(vec![source]);
    }

    let mut seen = HashSet::new();
    let mut parent_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut queue: VecDeque<(NodeId, u8)> = VecDeque::new();

    seen.insert(source);
    queue.push_back((source, 0));

    while let Some((node, depth)) = queue.pop_front() {
        if depth >= max_depth {
            continue;
        }

        let neighbors = adjacency.neighbors_out(node, None);
        for (neighbor, _) in neighbors {
            if !seen.insert(neighbor) {
                continue;
            }
            parent_map.insert(neighbor, node);

            if neighbor == target {
                // Reconstruct path
                let mut path = vec![target];
                let mut current = target;
                while let Some(&parent) = parent_map.get(&current) {
                    path.push(parent);
                    current = parent;
                    if current == source {
                        break;
                    }
                }
                path.reverse();
                return Some(path);
            }

            queue.push_back((neighbor, depth + 1));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adjacency::EdgeMeta;
    use weav_core::types::{BiTemporal, Provenance};

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

    fn build_linear_graph() -> AdjacencyStore {
        // 1 -> 2 -> 3 -> 4
        let mut adj = AdjacencyStore::new();
        for i in 1..=4 {
            adj.add_node(i);
        }
        for i in 1..=3 {
            let meta = make_meta(i, i + 1, 0);
            adj.add_edge(i, i + 1, 0, meta).unwrap();
        }
        adj
    }

    fn build_star_graph() -> AdjacencyStore {
        // center=1, spokes: 1->2, 1->3, 1->4, 1->5
        let mut adj = AdjacencyStore::new();
        for i in 1..=5 {
            adj.add_node(i);
        }
        for i in 2..=5 {
            let meta = make_meta(1, i, 0);
            adj.add_edge(1, i, 0, meta).unwrap();
        }
        adj
    }

    #[test]
    fn test_bfs_linear() {
        let adj = build_linear_graph();
        let result = bfs(
            &adj,
            &[1],
            3,
            100,
            &EdgeFilter::none(),
            &NodeFilter::none(),
            Direction::Outgoing,
        );
        assert_eq!(result.visited_nodes.len(), 4);
        assert_eq!(*result.depth_map.get(&1).unwrap(), 0);
        assert_eq!(*result.depth_map.get(&2).unwrap(), 1);
        assert_eq!(*result.depth_map.get(&3).unwrap(), 2);
        assert_eq!(*result.depth_map.get(&4).unwrap(), 3);
    }

    #[test]
    fn test_bfs_max_depth() {
        let adj = build_linear_graph();
        let result = bfs(
            &adj,
            &[1],
            1,
            100,
            &EdgeFilter::none(),
            &NodeFilter::none(),
            Direction::Outgoing,
        );
        assert_eq!(result.visited_nodes.len(), 2); // 1, 2
    }

    #[test]
    fn test_bfs_max_nodes() {
        let adj = build_linear_graph();
        let result = bfs(
            &adj,
            &[1],
            10,
            2,
            &EdgeFilter::none(),
            &NodeFilter::none(),
            Direction::Outgoing,
        );
        assert_eq!(result.visited_nodes.len(), 2);
    }

    #[test]
    fn test_bfs_with_edge_filter_weight() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);

        let meta1 = EdgeMeta {
            source: 1, target: 2, label: 0,
            temporal: BiTemporal::new_current(1000),
            provenance: None, weight: 0.5, token_cost: 0,
        };
        adj.add_edge(1, 2, 0, meta1).unwrap();

        let meta2 = EdgeMeta {
            source: 1, target: 3, label: 0,
            temporal: BiTemporal::new_current(1000),
            provenance: None, weight: 0.9, token_cost: 0,
        };
        adj.add_edge(1, 3, 0, meta2).unwrap();

        let filter = EdgeFilter {
            min_weight: Some(0.8),
            ..EdgeFilter::none()
        };
        let result = bfs(
            &adj, &[1], 1, 100, &filter, &NodeFilter::none(), Direction::Outgoing,
        );
        // Only node 3 should be reachable (weight 0.9 >= 0.8)
        assert_eq!(result.visited_nodes.len(), 2); // seed + node 3
        assert!(result.visited_nodes.contains(&3));
        assert!(!result.visited_nodes.contains(&2));
    }

    #[test]
    fn test_bfs_with_edge_filter_confidence() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);

        let meta1 = EdgeMeta {
            source: 1, target: 2, label: 0,
            temporal: BiTemporal::new_current(1000),
            provenance: Some(Provenance::new("test", 0.5)),
            weight: 1.0, token_cost: 0,
        };
        adj.add_edge(1, 2, 0, meta1).unwrap();

        let meta2 = EdgeMeta {
            source: 1, target: 3, label: 0,
            temporal: BiTemporal::new_current(1000),
            provenance: Some(Provenance::new("test", 0.9)),
            weight: 1.0, token_cost: 0,
        };
        adj.add_edge(1, 3, 0, meta2).unwrap();

        let filter = EdgeFilter {
            min_confidence: Some(0.7),
            ..EdgeFilter::none()
        };
        let result = bfs(
            &adj, &[1], 1, 100, &filter, &NodeFilter::none(), Direction::Outgoing,
        );
        assert_eq!(result.visited_nodes.len(), 2);
        assert!(result.visited_nodes.contains(&3));
        assert!(!result.visited_nodes.contains(&2));
    }

    #[test]
    fn test_bfs_with_valid_at_filter() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);

        let meta1 = EdgeMeta {
            source: 1, target: 2, label: 0,
            temporal: BiTemporal {
                valid_from: 100,
                valid_until: 200,
                tx_from: 100,
                tx_until: BiTemporal::OPEN,
            },
            provenance: None, weight: 1.0, token_cost: 0,
        };
        adj.add_edge(1, 2, 0, meta1).unwrap();

        let meta2 = EdgeMeta {
            source: 1, target: 3, label: 0,
            temporal: BiTemporal::new_current(100),
            provenance: None, weight: 1.0, token_cost: 0,
        };
        adj.add_edge(1, 3, 0, meta2).unwrap();

        // At timestamp 150: both edges valid
        let filter150 = EdgeFilter {
            valid_at: Some(150),
            ..EdgeFilter::none()
        };
        let result = bfs(
            &adj, &[1], 1, 100, &filter150, &NodeFilter::none(), Direction::Outgoing,
        );
        assert_eq!(result.visited_nodes.len(), 3);

        // At timestamp 250: only edge to 3 is valid
        let filter250 = EdgeFilter {
            valid_at: Some(250),
            ..EdgeFilter::none()
        };
        let result = bfs(
            &adj, &[1], 1, 100, &filter250, &NodeFilter::none(), Direction::Outgoing,
        );
        assert_eq!(result.visited_nodes.len(), 2);
        assert!(result.visited_nodes.contains(&3));
    }

    #[test]
    fn test_flow_score_basic() {
        let adj = build_linear_graph();
        let result = flow_score(&adj, &[(1, 1.0)], 0.5, 0.01, 10);

        // Node 1: 1.0, Node 2: 0.5, Node 3: 0.25, Node 4: 0.125
        assert!(result.len() >= 4);

        let score_map: HashMap<NodeId, f32> =
            result.iter().map(|s| (s.node_id, s.score)).collect();
        assert!((score_map[&1] - 1.0).abs() < 0.001);
        assert!((score_map[&2] - 0.5).abs() < 0.001);
        assert!((score_map[&3] - 0.25).abs() < 0.001);
        assert!((score_map[&4] - 0.125).abs() < 0.001);
    }

    #[test]
    fn test_flow_score_theta_cutoff() {
        let adj = build_linear_graph();
        let result = flow_score(&adj, &[(1, 1.0)], 0.5, 0.3, 10);

        let score_map: HashMap<NodeId, f32> =
            result.iter().map(|s| (s.node_id, s.score)).collect();
        // Node 3 score would be 0.25 < 0.3, so it shouldn't propagate
        assert!(score_map.contains_key(&1));
        assert!(score_map.contains_key(&2));
        assert!(!score_map.contains_key(&3));
    }

    #[test]
    fn test_flow_score_sorted() {
        let adj = build_star_graph();
        let result = flow_score(&adj, &[(1, 1.0)], 0.8, 0.01, 2);
        // Result should be sorted by score descending
        for i in 1..result.len() {
            assert!(result[i - 1].score >= result[i].score);
        }
    }

    #[test]
    fn test_ego_network() {
        let adj = build_star_graph();
        let sub = ego_network(&adj, 1, 1);
        assert_eq!(sub.nodes.len(), 5); // center + 4 spokes
        assert_eq!(sub.edges.len(), 4);
    }

    #[test]
    fn test_ego_network_radius_0() {
        let adj = build_star_graph();
        let sub = ego_network(&adj, 1, 0);
        assert_eq!(sub.nodes.len(), 1);
        assert!(sub.edges.is_empty());
    }

    #[test]
    fn test_shortest_path_direct() {
        let adj = build_linear_graph();
        let path = shortest_path(&adj, 1, 2, 10).unwrap();
        assert_eq!(path, vec![1, 2]);
    }

    #[test]
    fn test_shortest_path_multi_hop() {
        let adj = build_linear_graph();
        let path = shortest_path(&adj, 1, 4, 10).unwrap();
        assert_eq!(path, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_shortest_path_same_node() {
        let adj = build_linear_graph();
        let path = shortest_path(&adj, 1, 1, 10).unwrap();
        assert_eq!(path, vec![1]);
    }

    #[test]
    fn test_shortest_path_unreachable() {
        let adj = build_linear_graph();
        // 4 cannot reach 1 (directed graph)
        let path = shortest_path(&adj, 4, 1, 10);
        assert!(path.is_none());
    }

    #[test]
    fn test_shortest_path_max_depth_limit() {
        let adj = build_linear_graph();
        // Path 1->4 requires 3 hops, but max_depth=2
        let path = shortest_path(&adj, 1, 4, 2);
        assert!(path.is_none());
    }

    #[test]
    fn test_bfs_incoming_direction() {
        let adj = build_linear_graph();
        let result = bfs(
            &adj,
            &[4],
            3,
            100,
            &EdgeFilter::none(),
            &NodeFilter::none(),
            Direction::Incoming,
        );
        assert_eq!(result.visited_nodes.len(), 4);
        assert_eq!(*result.depth_map.get(&4).unwrap(), 0);
        assert_eq!(*result.depth_map.get(&3).unwrap(), 1);
    }

    #[test]
    fn test_bfs_label_filter() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);

        let meta1 = make_meta(1, 2, 0);
        adj.add_edge(1, 2, 0, meta1).unwrap();
        let meta2 = EdgeMeta {
            source: 1, target: 3, label: 1,
            temporal: BiTemporal::new_current(1000),
            provenance: None, weight: 1.0, token_cost: 0,
        };
        adj.add_edge(1, 3, 1, meta2).unwrap();

        let mut labels = HashSet::new();
        labels.insert(0);
        let filter = EdgeFilter {
            labels: Some(labels),
            ..EdgeFilter::none()
        };
        let result = bfs(
            &adj, &[1], 1, 100, &filter, &NodeFilter::none(), Direction::Outgoing,
        );
        assert_eq!(result.visited_nodes.len(), 2);
        assert!(result.visited_nodes.contains(&2));
        assert!(!result.visited_nodes.contains(&3));
    }
}
