//! Graph traversal algorithms: BFS, flow scoring, ego network, shortest path, scored paths.

use std::collections::{HashMap, HashSet, VecDeque};

use weav_core::types::{Direction, EdgeId, LabelId, NodeId, ScoredNode, ScoredPath, Timestamp};

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

    if let Some(max_age) = filter.max_age_ms {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let edge_age = now_ms.saturating_sub(meta.temporal.tx_from);
        if edge_age > max_age {
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

/// Check if a node passes the node filter criteria.
///
/// Implements `valid_at`, `labels`, and `has_property` filtering.
///
/// - `node_label_lookup`: Optional closure that resolves a node to its LabelId.
/// - `property_check`: Optional closure that checks if a node has a given property.
fn node_passes_filter(
    adjacency: &AdjacencyStore,
    node: NodeId,
    node_filter: &NodeFilter,
    node_label_lookup: Option<&dyn Fn(NodeId) -> Option<LabelId>>,
    property_check: Option<&dyn Fn(NodeId, &str) -> bool>,
) -> bool {
    if let Some(ts) = node_filter.valid_at {
        // A node passes the valid_at filter if it has at least one edge
        // (incoming or outgoing) that is valid at the given timestamp.
        let outgoing = adjacency.neighbors_out(node, None);
        let incoming = adjacency.neighbors_in(node, None);

        let has_valid_edge = outgoing
            .iter()
            .chain(incoming.iter())
            .any(|&(_, eid)| {
                adjacency
                    .get_edge(eid)
                    .map(|m| m.temporal.is_valid_at(ts))
                    .unwrap_or(false)
            });

        if !has_valid_edge {
            return false;
        }
    }

    // Label filtering: check if the node's label is in the allowed set.
    if let Some(ref allowed_labels) = node_filter.labels {
        if let Some(ref lookup) = node_label_lookup {
            match lookup(node) {
                Some(label_id) => {
                    if !allowed_labels.contains(&label_id) {
                        return false;
                    }
                }
                None => return false, // No label found, filter out
            }
        }
        // If no lookup provided, skip label filtering (can't check)
    }

    // Property filtering: check if the node has all required properties.
    if let Some(ref required_props) = node_filter.has_property {
        if let Some(ref checker) = property_check {
            for prop_name in required_props {
                if !checker(node, prop_name) {
                    return false;
                }
            }
        }
        // If no checker provided, skip property filtering (can't check)
    }

    true
}

/// Breadth-first search from seed nodes.
///
/// Optional closure params for node filtering:
/// - `node_label_lookup`: Resolves a node to its LabelId (for label filtering).
/// - `property_check`: Checks if a node has a given property (for has_property filtering).
pub fn bfs(
    adjacency: &AdjacencyStore,
    seeds: &[NodeId],
    max_depth: u8,
    max_nodes: usize,
    edge_filter: &EdgeFilter,
    node_filter: &NodeFilter,
    direction: Direction,
    node_label_lookup: Option<&dyn Fn(NodeId) -> Option<LabelId>>,
    property_check: Option<&dyn Fn(NodeId, &str) -> bool>,
) -> TraversalResult {
    let mut visited_nodes = Vec::new();
    let mut visited_edges = Vec::new();
    let mut depth_map = HashMap::new();
    let mut parent_map = HashMap::new();
    let mut seen = HashSet::new();
    let mut queue: VecDeque<(NodeId, u8)> = VecDeque::new();

    let has_node_filter = node_filter.valid_at.is_some()
        || node_filter.labels.is_some()
        || node_filter.has_property.is_some();

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
                // Apply node filter to discovered neighbor
                if has_node_filter
                    && !node_passes_filter(adjacency, neighbor, node_filter, node_label_lookup, property_check)
                {
                    // Node doesn't pass the filter; mark as seen but don't visit
                    continue;
                }

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
    result.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.node_id.cmp(&b.node_id))
    });
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
        None,
        None,
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

/// Find scored paths between anchor nodes (spec 4.3).
///
/// For each pair of anchor nodes, find paths up to `max_path_length` using BFS.
/// Score each path by averaging the anchor scores of nodes on the path.
/// Returns up to `max_paths` paths sorted by reliability score descending.
pub fn scored_paths(
    adjacency: &AdjacencyStore,
    anchors: &[(NodeId, f32)],
    max_paths: u32,
    max_path_length: u8,
) -> Vec<ScoredPath> {
    let anchor_scores: HashMap<NodeId, f32> =
        anchors.iter().copied().collect();

    let mut all_paths: Vec<ScoredPath> = Vec::new();

    // For each pair of anchor nodes, find paths between them
    for i in 0..anchors.len() {
        for j in (i + 1)..anchors.len() {
            let (src, _) = anchors[i];
            let (tgt, _) = anchors[j];

            if src == tgt {
                continue;
            }

            // BFS to find a path from src to tgt up to max_path_length
            if let Some(path_nodes) = shortest_path(adjacency, src, tgt, max_path_length) {
                // Compute reliability as the average anchor score of nodes on the path
                let mut score_sum = 0.0_f32;
                let mut score_count = 0u32;
                for &node in &path_nodes {
                    if let Some(&s) = anchor_scores.get(&node) {
                        score_sum += s;
                        score_count += 1;
                    }
                }
                let reliability = if score_count > 0 {
                    score_sum / score_count as f32
                } else {
                    0.0
                };

                // Collect edge ids along the path
                let mut edges = Vec::new();
                for w in path_nodes.windows(2) {
                    if let Some(eid) = adjacency.edge_between(w[0], w[1], None) {
                        edges.push(eid);
                    }
                }

                all_paths.push(ScoredPath {
                    nodes: path_nodes,
                    edges,
                    reliability,
                });
            }
        }
    }

    // Sort by reliability descending
    all_paths.sort_by(|a, b| {
        b.reliability
            .partial_cmp(&a.reliability)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    all_paths.truncate(max_paths as usize);
    all_paths
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
            None,
            None,
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
            None,
            None,
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
            None,
            None,
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
            None, None,
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
            None, None,
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
            None, None,
        );
        assert_eq!(result.visited_nodes.len(), 3);

        // At timestamp 250: only edge to 3 is valid
        let filter250 = EdgeFilter {
            valid_at: Some(250),
            ..EdgeFilter::none()
        };
        let result = bfs(
            &adj, &[1], 1, 100, &filter250, &NodeFilter::none(), Direction::Outgoing,
            None, None,
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
            None,
            None,
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
            None, None,
        );
        assert_eq!(result.visited_nodes.len(), 2);
        assert!(result.visited_nodes.contains(&2));
        assert!(!result.visited_nodes.contains(&3));
    }

    #[test]
    fn test_bfs_node_filter_valid_at() {
        // Build a graph: 1 -> 2 -> 3
        // Edge 1->2 is valid [100, 200)
        // Edge 2->3 is valid [100, OPEN)
        // Node 2 has edges valid at t=150 (both), so passes filter at t=150
        // Node 3 only reachable via node 2; at t=250, edge 1->2 is invalid so
        // node 2 not reachable via edge filter, but let's test via node filter:
        // We'll make edge 1->2 valid at all times, but node 2's edges:
        //   incoming 1->2 valid [100,200), outgoing 2->3 valid [100,OPEN)
        // At t=250: node 2 still has a valid edge (2->3), so passes node filter
        // At t=50: node 2 has no valid edges (1->2 starts at 100, 2->3 starts at 100)
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);

        // Edge 1->2: always valid
        let meta1 = EdgeMeta {
            source: 1, target: 2, label: 0,
            temporal: BiTemporal::new_current(100),
            provenance: None, weight: 1.0, token_cost: 0,
        };
        adj.add_edge(1, 2, 0, meta1).unwrap();

        // Edge 2->3: valid [200, 300)
        let meta2 = EdgeMeta {
            source: 2, target: 3, label: 0,
            temporal: BiTemporal {
                valid_from: 200,
                valid_until: 300,
                tx_from: 200,
                tx_until: BiTemporal::OPEN,
            },
            provenance: None, weight: 1.0, token_cost: 0,
        };
        adj.add_edge(2, 3, 0, meta2).unwrap();

        // With node_filter valid_at=250: node 2 has edge 1->2 valid at 250 (new_current(100))
        // and edge 2->3 valid at 250 (200..300). Both pass. Node 3 has edge 2->3 valid at 250. Passes.
        let nf = NodeFilter {
            valid_at: Some(250),
            ..NodeFilter::none()
        };
        let result = bfs(
            &adj, &[1], 3, 100, &EdgeFilter::none(), &nf, Direction::Outgoing,
            None, None,
        );
        assert!(result.visited_nodes.contains(&2));
        assert!(result.visited_nodes.contains(&3));

        // With node_filter valid_at=50: node 2 has edge 1->2 (valid_from=100, >50 so invalid)
        // and edge 2->3 (valid_from=200, >50 so invalid). Node 2 fails filter.
        let nf2 = NodeFilter {
            valid_at: Some(50),
            ..NodeFilter::none()
        };
        let result2 = bfs(
            &adj, &[1], 3, 100, &EdgeFilter::none(), &nf2, Direction::Outgoing,
            None, None,
        );
        // Node 2 should be filtered out because none of its edges are valid at t=50
        assert!(!result2.visited_nodes.contains(&2));
        // Node 3 also not reachable since node 2 was filtered
        assert!(!result2.visited_nodes.contains(&3));
    }

    #[test]
    fn test_edge_filter_max_age_ms() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Edge 1->2: created very recently (now - 100ms)
        let meta1 = EdgeMeta {
            source: 1, target: 2, label: 0,
            temporal: BiTemporal::new_current(now_ms - 100),
            provenance: None, weight: 1.0, token_cost: 0,
        };
        adj.add_edge(1, 2, 0, meta1).unwrap();

        // Edge 1->3: created long ago (now - 10_000_000ms = ~2.7 hours ago)
        let meta2 = EdgeMeta {
            source: 1, target: 3, label: 0,
            temporal: BiTemporal::new_current(now_ms - 10_000_000),
            provenance: None, weight: 1.0, token_cost: 0,
        };
        adj.add_edge(1, 3, 0, meta2).unwrap();

        // Filter: max_age 5000ms - only edge 1->2 should pass
        let filter = EdgeFilter {
            max_age_ms: Some(5000),
            ..EdgeFilter::none()
        };
        let result = bfs(
            &adj, &[1], 1, 100, &filter, &NodeFilter::none(), Direction::Outgoing,
            None, None,
        );
        assert_eq!(result.visited_nodes.len(), 2); // seed + node 2
        assert!(result.visited_nodes.contains(&2));
        assert!(!result.visited_nodes.contains(&3));
    }

    #[test]
    fn test_scored_paths_linear() {
        // 1 -> 2 -> 3 -> 4
        let adj = build_linear_graph();
        // Anchors: node 1 (score 1.0), node 4 (score 0.8)
        let anchors = vec![(1, 1.0_f32), (4, 0.8_f32)];
        let paths = scored_paths(&adj, &anchors, 10, 5);

        assert_eq!(paths.len(), 1); // one path between the pair (1,4)
        assert_eq!(paths[0].nodes, vec![1, 2, 3, 4]);
        // Reliability = average of anchor scores on path: (1.0 + 0.8) / 2 = 0.9
        assert!((paths[0].reliability - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_scored_paths_multiple_anchors() {
        // 1 -> 2 -> 3 -> 4
        let adj = build_linear_graph();
        // Three anchors: 1, 2, 4
        let anchors = vec![(1, 1.0_f32), (2, 0.6_f32), (4, 0.4_f32)];
        let paths = scored_paths(&adj, &anchors, 10, 5);

        // Should find paths: (1,2), (1,4), (2,4) = 3 paths
        assert_eq!(paths.len(), 3);

        // Paths should be sorted by reliability descending
        for i in 1..paths.len() {
            assert!(paths[i - 1].reliability >= paths[i].reliability);
        }
    }

    #[test]
    fn test_scored_paths_max_paths_limit() {
        let adj = build_linear_graph();
        let anchors = vec![(1, 1.0_f32), (2, 0.6_f32), (4, 0.4_f32)];
        let paths = scored_paths(&adj, &anchors, 1, 5);
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn test_scored_paths_unreachable() {
        let adj = build_linear_graph();
        // Node 4 cannot reach node 1 (directed), so (4, 1) pair yields no path via shortest_path(4,1)
        // but (1, 4) pair still yields a path via shortest_path(1, 4)
        let anchors = vec![(1, 1.0_f32), (4, 0.8_f32)];
        let paths = scored_paths(&adj, &anchors, 10, 5);
        // shortest_path(1, 4) works, so we get 1 path
        assert_eq!(paths.len(), 1);
    }

    #[test]
    fn test_scored_paths_empty_anchors() {
        let adj = build_linear_graph();
        let paths = scored_paths(&adj, &[], 10, 5);
        assert!(paths.is_empty());
    }

    #[test]
    fn test_scored_paths_single_anchor() {
        let adj = build_linear_graph();
        let anchors = vec![(1, 1.0_f32)];
        let paths = scored_paths(&adj, &anchors, 10, 5);
        // No pairs to connect
        assert!(paths.is_empty());
    }

    #[test]
    fn test_bfs_direction_both() {
        // Node 2 has outgoing edge to 3 and node 4 has edge incoming to 2
        // i.e., 4 -> 2 -> 3
        let mut adj = AdjacencyStore::new();
        adj.add_node(2);
        adj.add_node(3);
        adj.add_node(4);

        let meta_2_3 = make_meta(2, 3, 0);
        adj.add_edge(2, 3, 0, meta_2_3).unwrap();
        let meta_4_2 = make_meta(4, 2, 0);
        adj.add_edge(4, 2, 0, meta_4_2).unwrap();

        let result = bfs(
            &adj,
            &[2],
            2,
            100,
            &EdgeFilter::none(),
            &NodeFilter::none(),
            Direction::Both,
            None,
            None,
        );

        // BFS from 2 with Direction::Both should find 2, 3 (outgoing), and 4 (incoming)
        assert!(result.visited_nodes.contains(&2));
        assert!(result.visited_nodes.contains(&3));
        assert!(result.visited_nodes.contains(&4));
        assert_eq!(result.visited_nodes.len(), 3);
    }

    #[test]
    fn test_bfs_cycle_graph() {
        // Build cycle: 1 -> 2 -> 3 -> 1
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);

        let meta_1_2 = make_meta(1, 2, 0);
        adj.add_edge(1, 2, 0, meta_1_2).unwrap();
        let meta_2_3 = make_meta(2, 3, 0);
        adj.add_edge(2, 3, 0, meta_2_3).unwrap();
        let meta_3_1 = make_meta(3, 1, 0);
        adj.add_edge(3, 1, 0, meta_3_1).unwrap();

        let result = bfs(
            &adj,
            &[1],
            10,
            100,
            &EdgeFilter::none(),
            &NodeFilter::none(),
            Direction::Outgoing,
            None,
            None,
        );

        // All 3 nodes should be found, no infinite loop
        assert_eq!(result.visited_nodes.len(), 3);
        assert!(result.visited_nodes.contains(&1));
        assert!(result.visited_nodes.contains(&2));
        assert!(result.visited_nodes.contains(&3));
    }

    #[test]
    fn test_bfs_disconnected_graph() {
        // Two disconnected components: 1->2 and 3->4
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);
        adj.add_node(4);

        let meta_1_2 = make_meta(1, 2, 0);
        adj.add_edge(1, 2, 0, meta_1_2).unwrap();
        let meta_3_4 = make_meta(3, 4, 0);
        adj.add_edge(3, 4, 0, meta_3_4).unwrap();

        // BFS from node 1 should only find 1 and 2
        let result1 = bfs(
            &adj,
            &[1],
            10,
            100,
            &EdgeFilter::none(),
            &NodeFilter::none(),
            Direction::Outgoing,
            None,
            None,
        );
        let mut nodes1 = result1.visited_nodes.clone();
        nodes1.sort();
        assert_eq!(nodes1, vec![1, 2]);

        // BFS from node 3 should only find 3 and 4
        let result2 = bfs(
            &adj,
            &[3],
            10,
            100,
            &EdgeFilter::none(),
            &NodeFilter::none(),
            Direction::Outgoing,
            None,
            None,
        );
        let mut nodes2 = result2.visited_nodes.clone();
        nodes2.sort();
        assert_eq!(nodes2, vec![3, 4]);
    }

    #[test]
    fn test_bfs_empty_graph() {
        let adj = AdjacencyStore::new();

        // BFS from a non-existent node on an empty graph
        let result = bfs(
            &adj,
            &[42],
            10,
            100,
            &EdgeFilter::none(),
            &NodeFilter::none(),
            Direction::Outgoing,
            None,
            None,
        );

        // The seed is still added to visited_nodes even if it doesn't exist
        // (BFS doesn't check node existence, it just won't find any neighbors)
        assert_eq!(result.visited_nodes.len(), 1);
        assert_eq!(result.visited_nodes[0], 42);
        assert!(result.visited_edges.is_empty());
    }

    #[test]
    fn test_flow_score_empty_seeds() {
        let adj = build_linear_graph();
        let result = flow_score(&adj, &[], 0.5, 0.01, 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_bfs_node_label_filter() {
        // Graph: 1->2, 1->3. Node 2 has label 0, node 3 has label 1.
        // Filter allows only label 0 => only node 2 should be visited.
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);

        let meta1 = make_meta(1, 2, 0);
        adj.add_edge(1, 2, 0, meta1).unwrap();
        let meta2 = make_meta(1, 3, 0);
        adj.add_edge(1, 3, 0, meta2).unwrap();

        // Label lookup: node 2 -> label 0, node 3 -> label 1
        let label_lookup = |nid: NodeId| -> Option<LabelId> {
            match nid {
                1 => Some(0),
                2 => Some(0),
                3 => Some(1),
                _ => None,
            }
        };

        let mut allowed = HashSet::new();
        allowed.insert(0 as LabelId);
        let nf = NodeFilter {
            labels: Some(allowed),
            ..NodeFilter::none()
        };

        let result = bfs(
            &adj, &[1], 1, 100, &EdgeFilter::none(), &nf, Direction::Outgoing,
            Some(&label_lookup), None,
        );
        assert!(result.visited_nodes.contains(&1)); // seed always included
        assert!(result.visited_nodes.contains(&2)); // label 0: passes
        assert!(!result.visited_nodes.contains(&3)); // label 1: filtered out
    }

    #[test]
    fn test_bfs_has_property_filter() {
        // Graph: 1->2, 1->3. Only node 2 has property "status".
        // Filter requires "status" => only node 2 should be visited.
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);

        let meta1 = make_meta(1, 2, 0);
        adj.add_edge(1, 2, 0, meta1).unwrap();
        let meta2 = make_meta(1, 3, 0);
        adj.add_edge(1, 3, 0, meta2).unwrap();

        // Property check: only node 2 has "status"
        let prop_check = |nid: NodeId, prop: &str| -> bool {
            nid == 2 && prop == "status"
        };

        let nf = NodeFilter {
            has_property: Some(vec!["status".to_string()]),
            ..NodeFilter::none()
        };

        let result = bfs(
            &adj, &[1], 1, 100, &EdgeFilter::none(), &nf, Direction::Outgoing,
            None, Some(&prop_check),
        );
        assert!(result.visited_nodes.contains(&1)); // seed always included
        assert!(result.visited_nodes.contains(&2)); // has "status": passes
        assert!(!result.visited_nodes.contains(&3)); // no "status": filtered out
    }

    #[test]
    fn test_shortest_path_no_edges() {
        // Graph with nodes but no edges between them
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);

        let path = shortest_path(&adj, 1, 3, 10);
        assert!(path.is_none());
    }

    #[test]
    fn test_flow_score_deterministic_tie_breaking() {
        // Star graph: center=1, spokes: 1->2, 1->3, 1->4, 1->5
        // All spokes get the same propagated score, so tie-break should be by node_id ascending
        let adj = build_star_graph();
        let result = flow_score(&adj, &[(1, 1.0)], 0.5, 0.01, 2);

        // Node 1 has score 1.0 (seed), nodes 2-5 each get 0.5
        assert!(result.len() >= 5);
        assert_eq!(result[0].node_id, 1); // highest score
        assert!((result[0].score - 1.0).abs() < 0.001);

        // Tied nodes (score=0.5) should be sorted by node_id ascending
        let tied: Vec<&ScoredNode> = result.iter().filter(|s| (s.score - 0.5).abs() < 0.001).collect();
        assert_eq!(tied.len(), 4);
        for i in 1..tied.len() {
            assert!(tied[i - 1].node_id < tied[i].node_id,
                "Tied nodes should be sorted by node_id ascending: {} < {}",
                tied[i - 1].node_id, tied[i].node_id);
        }
    }

    #[test]
    fn test_ego_network_basic() {
        // Build graph with center node 10 connected to 3 spokes: 10->20, 10->30, 10->40
        let mut adj = AdjacencyStore::new();
        adj.add_node(10);
        adj.add_node(20);
        adj.add_node(30);
        adj.add_node(40);

        let meta_10_20 = make_meta(10, 20, 0);
        adj.add_edge(10, 20, 0, meta_10_20).unwrap();
        let meta_10_30 = make_meta(10, 30, 0);
        adj.add_edge(10, 30, 0, meta_10_30).unwrap();
        let meta_10_40 = make_meta(10, 40, 0);
        adj.add_edge(10, 40, 0, meta_10_40).unwrap();

        let sub = ego_network(&adj, 10, 1);

        // Should find center + 3 spokes = 4 nodes
        assert_eq!(sub.nodes.len(), 4);
        assert!(sub.nodes.contains(&10));
        assert!(sub.nodes.contains(&20));
        assert!(sub.nodes.contains(&30));
        assert!(sub.nodes.contains(&40));
        assert_eq!(sub.edges.len(), 3);
    }
}
