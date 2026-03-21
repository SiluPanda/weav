//! Graph traversal algorithms: BFS, flow scoring, ego network, shortest path,
//! scored paths, Dijkstra, connected components, Personalized PageRank,
//! Label Propagation community detection, modularity-based community detection,
//! betweenness centrality, closeness centrality, triangle counting,
//! Tarjan's strongly connected components, topological sort.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

use weav_core::error::{WeavError, WeavResult};
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

    if let Some(ref labels) = filter.labels
        && !labels.contains(&meta.label)
    {
        return false;
    }

    if let Some(min_w) = filter.min_weight
        && meta.weight < min_w
    {
        return false;
    }

    if let Some(ts) = filter.valid_at
        && !meta.temporal.is_valid_at(ts)
    {
        return false;
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
    if let Some(ref allowed_labels) = node_filter.labels
        && let Some(ref lookup) = node_label_lookup
    {
        match lookup(node) {
            Some(label_id) => {
                if !allowed_labels.contains(&label_id) {
                    return false;
                }
            }
            None => return false, // No label found, filter out
        }
        // If no lookup provided, skip label filtering (can't check)
    }

    // Property filtering: check if the node has all required properties.
    if let Some(ref required_props) = node_filter.has_property
        && let Some(ref checker) = property_check
    {
        for prop_name in required_props {
            if !checker(node, prop_name) {
                return false;
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
    let mut visited_edges_set = HashSet::new();
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
                visited_edges_set.insert(edge_id);
                visited_edges.push(edge_id);
                if visited_nodes.len() >= max_nodes {
                    break;
                }
                queue.push_back((neighbor, next_depth));
            } else {
                // Node already visited, but we may still want to record the edge
                if visited_edges_set.insert(edge_id) {
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
    let mut work_count: usize = 0;
    let max_work = seeds_with_scores.len().max(1) * 10000; // bounded work

    for &(node, score) in seeds_with_scores {
        scores.insert(node, (score, 0));
        queue.push_back((node, score, 0));
    }

    while let Some((node, score, depth)) = queue.pop_front() {
        work_count += 1;
        if work_count > max_work {
            break;
        }
        // Skip stale queue entries — a better score was found since this was enqueued
        if let Some(&(current_score, _)) = scores.get(&node)
            && score < current_score
        {
            continue;
        }
        if depth >= max_depth {
            continue;
        }

        let neighbors = adjacency.neighbors_out(node, None);
        for (neighbor, _edge_id) in neighbors {
            let propagated = score * alpha;
            if !propagated.is_finite() || propagated < theta {
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

// ─── Weighted shortest path (Dijkstra) ──────────────────────────────────────

/// A node-distance pair for the priority queue, ordered by minimum distance.
struct DijkstraState {
    node: NodeId,
    cost: f64,
}

impl PartialEq for DijkstraState {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.node == other.node
    }
}

impl Eq for DijkstraState {}

impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (BinaryHeap is a max-heap by default)
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

/// Weighted shortest path result.
pub struct WeightedPath {
    pub nodes: Vec<NodeId>,
    pub total_weight: f64,
}

/// Find the shortest weighted path between source and target using Dijkstra's algorithm.
///
/// Edge weights are used as costs (lower weight = cheaper path).
/// Returns the path and its total weight, or None if no path exists within max_depth hops.
pub fn dijkstra_shortest_path(
    adjacency: &AdjacencyStore,
    source: NodeId,
    target: NodeId,
    max_depth: u8,
) -> Option<WeightedPath> {
    if source == target {
        return Some(WeightedPath {
            nodes: vec![source],
            total_weight: 0.0,
        });
    }

    let mut dist: HashMap<NodeId, f64> = HashMap::new();
    let mut depth: HashMap<NodeId, u8> = HashMap::new();
    let mut parent: HashMap<NodeId, NodeId> = HashMap::new();
    let mut heap = BinaryHeap::new();

    dist.insert(source, 0.0);
    depth.insert(source, 0);
    heap.push(DijkstraState {
        node: source,
        cost: 0.0,
    });

    while let Some(DijkstraState { node, cost }) = heap.pop() {
        if node == target {
            // Reconstruct path
            let mut path = vec![target];
            let mut current = target;
            while let Some(&p) = parent.get(&current) {
                path.push(p);
                current = p;
                if current == source {
                    break;
                }
            }
            path.reverse();
            return Some(WeightedPath {
                nodes: path,
                total_weight: cost,
            });
        }

        // Skip if we've already found a better path to this node
        if let Some(&d) = dist.get(&node)
            && cost > d
        {
            continue;
        }

        let node_depth = depth.get(&node).copied().unwrap_or(0);
        if node_depth >= max_depth {
            continue;
        }

        let neighbors = adjacency.neighbors_out(node, None);
        for (neighbor, edge_id) in neighbors {
            let edge_weight = adjacency
                .get_edge(edge_id)
                .map(|m| m.weight as f64)
                .unwrap_or(1.0);

            // Use 1/weight as cost so higher weight = lower cost
            // (weight represents strength/relevance; we want the strongest path)
            let edge_cost = if edge_weight > 0.0 {
                1.0 / edge_weight
            } else {
                f64::MAX
            };

            let next_cost = cost + edge_cost;
            let is_better = dist.get(&neighbor).is_none_or(|&d| next_cost < d);

            if is_better {
                dist.insert(neighbor, next_cost);
                depth.insert(neighbor, node_depth + 1);
                parent.insert(neighbor, node);
                heap.push(DijkstraState {
                    node: neighbor,
                    cost: next_cost,
                });
            }
        }
    }

    None
}

// ─── Connected Components ───────────────────────────────────────────────────

/// Find all connected components in the graph (treating edges as undirected).
///
/// Returns a map from NodeId to component ID (0-indexed).
pub fn connected_components(adjacency: &AdjacencyStore) -> HashMap<NodeId, u32> {
    let all_nodes = adjacency.all_node_ids();
    let mut component_map: HashMap<NodeId, u32> = HashMap::new();
    let mut component_id: u32 = 0;

    for &node in &all_nodes {
        if component_map.contains_key(&node) {
            continue;
        }

        // BFS from this node, treating edges as undirected
        let mut queue = VecDeque::new();
        queue.push_back(node);
        component_map.insert(node, component_id);

        while let Some(current) = queue.pop_front() {
            // Follow both outgoing and incoming edges
            let out_neighbors = adjacency.neighbors_out(current, None);
            let in_neighbors = adjacency.neighbors_in(current, None);

            for (neighbor, _) in out_neighbors.iter().chain(in_neighbors.iter()) {
                if !component_map.contains_key(neighbor) {
                    component_map.insert(*neighbor, component_id);
                    queue.push_back(*neighbor);
                }
            }
        }

        component_id += 1;
    }

    component_map
}

// ─── Personalized PageRank ──────────────────────────────────────────────────

/// Compute Personalized PageRank scores relative to a set of seed nodes.
///
/// - `seeds`: seed nodes and their teleport weights (will be normalized)
/// - `alpha`: teleport probability (typically 0.15); higher = more biased toward seeds
/// - `max_iterations`: maximum number of power iterations
/// - `tolerance`: convergence threshold (L1 norm of score changes)
pub fn personalized_pagerank(
    adjacency: &AdjacencyStore,
    seeds: &[(NodeId, f32)],
    alpha: f32,
    max_iterations: u32,
    tolerance: f32,
) -> Vec<ScoredNode> {
    if seeds.is_empty() {
        return Vec::new();
    }

    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return Vec::new();
    }

    let n = all_nodes.len();

    // Build node index for fast lookup
    let node_to_idx: HashMap<NodeId, usize> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, &nid)| (nid, i))
        .collect();

    // Normalize seed teleport distribution
    let total_seed_weight: f32 = seeds.iter().map(|(_, w)| w).sum();
    let mut teleport = vec![0.0_f32; n];
    if total_seed_weight > 0.0 {
        for &(nid, w) in seeds {
            if let Some(&idx) = node_to_idx.get(&nid) {
                teleport[idx] = w / total_seed_weight;
            }
        }
    }

    // Initialize scores uniformly
    let init_score = 1.0 / n as f32;
    let mut scores = vec![init_score; n];

    // Pre-compute out-degrees for normalization
    let out_degrees: Vec<usize> = all_nodes
        .iter()
        .map(|&nid| adjacency.neighbors_out(nid, None).len())
        .collect();

    // Power iteration
    for _ in 0..max_iterations {
        let mut new_scores = vec![0.0_f32; n];

        // Distribute score along edges
        for (i, &nid) in all_nodes.iter().enumerate() {
            let out_deg = out_degrees[i];
            if out_deg == 0 {
                // Dangling node: distribute evenly to all seeds
                for &(seed_nid, _) in seeds {
                    if let Some(&seed_idx) = node_to_idx.get(&seed_nid) {
                        new_scores[seed_idx] += scores[i] / seeds.len() as f32;
                    }
                }
            } else {
                let share = scores[i] / out_deg as f32;
                for (neighbor, _) in adjacency.neighbors_out(nid, None) {
                    if let Some(&j) = node_to_idx.get(&neighbor) {
                        new_scores[j] += share;
                    }
                }
            }
        }

        // Apply teleport
        let mut diff = 0.0_f32;
        for i in 0..n {
            new_scores[i] = alpha * teleport[i] + (1.0 - alpha) * new_scores[i];
            diff += (new_scores[i] - scores[i]).abs();
        }

        scores = new_scores;

        if diff < tolerance {
            break;
        }
    }

    // Build result
    let mut result: Vec<ScoredNode> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, &nid)| ScoredNode {
            node_id: nid,
            score: scores[i],
            depth: 0,
        })
        .collect();

    result.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.node_id.cmp(&b.node_id))
    });

    result
}

/// Label Propagation community detection.
///
/// Each node starts with its own label. Iteratively, each node adopts the most
/// frequent label among its neighbors (weighted by edge weight). Converges when
/// no labels change. Treats edges as undirected.
///
/// Returns a map from NodeId to community label (u64).
pub fn label_propagation(
    adjacency: &AdjacencyStore,
    max_iterations: u32,
) -> HashMap<NodeId, u64> {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return HashMap::new();
    }

    // Initialize each node with its own label
    let mut labels: HashMap<NodeId, u64> = all_nodes.iter().map(|&nid| (nid, nid)).collect();

    for iteration in 0..max_iterations {
        // Deterministic shuffle: sort nodes by hash(node_id ^ iteration)
        let mut order: Vec<NodeId> = all_nodes.clone();
        order.sort_by(|&a, &b| {
            let hash_a = {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                (a ^ (iteration as u64)).hash(&mut h);
                h.finish()
            };
            let hash_b = {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                (b ^ (iteration as u64)).hash(&mut h);
                h.finish()
            };
            hash_a.cmp(&hash_b)
        });

        let mut changed = false;

        for &node in &order {
            // Collect all neighbors (both directions) to treat edges as undirected
            let out_neighbors = adjacency.neighbors_out(node, None);
            let in_neighbors = adjacency.neighbors_in(node, None);

            // Count weighted label frequencies among neighbors
            let mut freq: HashMap<u64, f64> = HashMap::new();
            for &(nbr, edge_id) in out_neighbors.iter().chain(in_neighbors.iter()) {
                let weight = adjacency
                    .get_edge(edge_id)
                    .map(|meta| meta.weight as f64)
                    .unwrap_or(1.0);
                let nbr_label = labels[&nbr];
                *freq.entry(nbr_label).or_insert(0.0) += weight;
            }

            if freq.is_empty() {
                continue; // isolated node keeps its label
            }

            // Find the most frequent label; break ties by smallest label
            let best_label = freq
                .iter()
                .max_by(|&(label_a, weight_a), &(label_b, weight_b)| {
                    weight_a
                        .partial_cmp(weight_b)
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| label_b.cmp(label_a)) // prefer smaller label
                })
                .map(|(&label, _)| label)
                .unwrap();

            if labels[&node] != best_label {
                labels.insert(node, best_label);
                changed = true;
            }
        }

        if !changed {
            break; // converged
        }
    }

    labels
}

/// Leiden/Louvain community detection using modularity optimization.
///
/// Iteratively moves nodes between communities to maximize modularity Q.
/// Then aggregates communities into super-nodes and repeats.
/// Returns a map from NodeId to community label (u64).
///
/// Unlike label_propagation, this produces higher-quality communities
/// by optimizing a global quality function (modularity).
///
/// The `resolution` parameter controls community granularity:
/// - 1.0 = standard modularity
/// - \> 1.0 = smaller, more numerous communities
/// - \< 1.0 = larger, fewer communities
pub fn modularity_communities(
    adjacency: &AdjacencyStore,
    max_iterations: u32,
    resolution: f32,
) -> HashMap<NodeId, u64> {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return HashMap::new();
    }

    // Initialize each node in its own community (community label = node id)
    let mut community: HashMap<NodeId, u64> = all_nodes.iter().map(|&nid| (nid, nid)).collect();

    // Compute total edge weight m (treating the directed graph as undirected).
    // We sum weights from both out- and in-neighbors for each node, which
    // double-counts every undirected edge, so we divide by 2 at the end.
    // Also compute per-node weighted degree k_i (again undirected).
    let mut node_degree: HashMap<NodeId, f64> = HashMap::with_capacity(all_nodes.len());
    let mut total_weight_double = 0.0_f64;

    for &node in &all_nodes {
        let mut ki = 0.0_f64;
        for &(_, edge_id) in adjacency.neighbors_out(node, None).iter() {
            let w = adjacency
                .get_edge(edge_id)
                .map(|meta| meta.weight as f64)
                .unwrap_or(1.0);
            ki += w;
        }
        for &(_, edge_id) in adjacency.neighbors_in(node, None).iter() {
            let w = adjacency
                .get_edge(edge_id)
                .map(|meta| meta.weight as f64)
                .unwrap_or(1.0);
            ki += w;
        }
        node_degree.insert(node, ki);
        total_weight_double += ki;
    }

    // m = total undirected edge weight (each edge counted once)
    let m = total_weight_double / 2.0;
    if m == 0.0 {
        // No edges — every node stays in its own community
        return community;
    }

    // sigma_tot[c] = sum of k_i for all nodes i in community c
    let mut sigma_tot: HashMap<u64, f64> = HashMap::with_capacity(all_nodes.len());
    for &node in &all_nodes {
        sigma_tot.insert(node, node_degree[&node]);
    }

    let resolution = resolution as f64;

    for iteration in 0..max_iterations {
        // Deterministic shuffle: sort nodes by hash(node_id ^ iteration)
        let mut order: Vec<NodeId> = all_nodes.clone();
        order.sort_by(|&a, &b| {
            let hash_a = {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                (a ^ (iteration as u64)).hash(&mut h);
                h.finish()
            };
            let hash_b = {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                (b ^ (iteration as u64)).hash(&mut h);
                h.finish()
            };
            hash_a.cmp(&hash_b)
        });

        let mut changed = false;

        for &node in &order {
            let ki = node_degree[&node];
            let current_comm = community[&node];

            // Compute weighted edges from this node to each neighboring community.
            // Treat graph as undirected: consider both out- and in-neighbors.
            let mut edges_to_comm: HashMap<u64, f64> = HashMap::new();
            let out_neighbors = adjacency.neighbors_out(node, None);
            let in_neighbors = adjacency.neighbors_in(node, None);
            for &(nbr, edge_id) in out_neighbors.iter().chain(in_neighbors.iter()) {
                let w = adjacency
                    .get_edge(edge_id)
                    .map(|meta| meta.weight as f64)
                    .unwrap_or(1.0);
                let nbr_comm = community[&nbr];
                *edges_to_comm.entry(nbr_comm).or_insert(0.0) += w;
            }

            if edges_to_comm.is_empty() {
                continue; // isolated node
            }

            // Temporarily remove node from its current community for gain computation.
            // sigma_tot of current community without this node:
            let sigma_tot_old = sigma_tot.get(&current_comm).copied().unwrap_or(0.0) - ki;

            // Edges from node to its own (current) community
            let ki_in = edges_to_comm.get(&current_comm).copied().unwrap_or(0.0);

            // Gain of removing node from current community (loss):
            // We compute relative gains so we can compare. The "removal cost" is:
            //   remove_cost = ki_in / m - resolution * ki * sigma_tot_old / (2 * m^2)
            let remove_cost = ki_in / m - resolution * ki * sigma_tot_old / (2.0 * m * m);

            // Find the best community to move to
            let mut best_comm = current_comm;
            let mut best_gain = 0.0_f64; // must be strictly positive to move

            for (&cand_comm, &ki_cand) in &edges_to_comm {
                if cand_comm == current_comm {
                    continue;
                }
                let sigma_tot_cand = sigma_tot.get(&cand_comm).copied().unwrap_or(0.0);

                // Gain of adding node to candidate community:
                //   add_gain = ki_cand / m - resolution * ki * sigma_tot_cand / (2 * m^2)
                let add_gain =
                    ki_cand / m - resolution * ki * sigma_tot_cand / (2.0 * m * m);

                // Net gain = add_gain - remove_cost
                let net_gain = add_gain - remove_cost;

                if net_gain > best_gain
                    || (net_gain == best_gain && cand_comm < best_comm)
                {
                    best_gain = net_gain;
                    best_comm = cand_comm;
                }
            }

            if best_comm != current_comm {
                // Move node from current_comm to best_comm
                *sigma_tot.entry(current_comm).or_insert(0.0) -= ki;
                *sigma_tot.entry(best_comm).or_insert(0.0) += ki;
                community.insert(node, best_comm);
                changed = true;
            }
        }

        if !changed {
            break; // converged
        }
    }

    community
}

// ─── Betweenness Centrality (Brandes) ────────────────────────────────────────

/// Compute betweenness centrality for every node using Brandes' algorithm.
///
/// Betweenness centrality measures how often a node lies on shortest paths
/// between other node pairs. Edges are treated as undirected and unweighted.
///
/// An optional `edge_filter` restricts which edges are traversed.
///
/// Returns a list of `(NodeId, centrality)` sorted descending by score.
pub fn betweenness_centrality(
    adjacency: &AdjacencyStore,
    edge_filter: &EdgeFilter,
) -> Vec<(NodeId, f64)> {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return Vec::new();
    }

    let mut cb: HashMap<NodeId, f64> = all_nodes.iter().map(|&n| (n, 0.0)).collect();

    for &s in &all_nodes {
        // Single-source shortest-path DAG via BFS
        let mut stack: Vec<NodeId> = Vec::new();
        let mut pred: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        let mut sigma: HashMap<NodeId, f64> = HashMap::new();
        let mut dist: HashMap<NodeId, i64> = HashMap::new();

        for &v in &all_nodes {
            pred.insert(v, Vec::new());
            sigma.insert(v, 0.0);
            dist.insert(v, -1);
        }
        *sigma.get_mut(&s).unwrap() = 1.0;
        *dist.get_mut(&s).unwrap() = 0;

        let mut queue: VecDeque<NodeId> = VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let d_v = dist[&v];

            // Undirected: follow both out and in neighbors
            let out_nbrs = adjacency.neighbors_out(v, None);
            let in_nbrs = adjacency.neighbors_in(v, None);
            for &(w, eid) in out_nbrs.iter().chain(in_nbrs.iter()) {
                if !edge_passes_filter(adjacency, eid, edge_filter) {
                    continue;
                }
                if dist[&w] < 0 {
                    // First visit
                    *dist.get_mut(&w).unwrap() = d_v + 1;
                    queue.push_back(w);
                }
                if dist[&w] == d_v + 1 {
                    *sigma.get_mut(&w).unwrap() += sigma[&v];
                    pred.get_mut(&w).unwrap().push(v);
                }
            }
        }

        // Back-propagation of dependencies
        let mut delta: HashMap<NodeId, f64> = all_nodes.iter().map(|&n| (n, 0.0)).collect();
        while let Some(w) = stack.pop() {
            if sigma[&w] > 0.0 {
                for v in &pred[&w] {
                    let d = (sigma[v] / sigma[&w]) * (1.0 + delta[&w]);
                    *delta.get_mut(v).unwrap() += d;
                }
            }
            if w != s {
                *cb.get_mut(&w).unwrap() += delta[&w];
            }
        }
    }

    // Undirected graph: each pair (s,t) is counted from both directions,
    // so divide by 2.
    for val in cb.values_mut() {
        *val /= 2.0;
    }

    let mut result: Vec<(NodeId, f64)> = cb.into_iter().collect();
    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0)));
    result
}

// ─── Closeness Centrality ────────────────────────────────────────────────────

/// Compute closeness centrality for every node.
///
/// For each node, closeness = (number of reachable nodes - 1) / (sum of shortest
/// path distances to all reachable nodes). If a node cannot reach any other node,
/// its closeness is 0. This uses the Wasserman-Faust normalization so that nodes
/// in smaller components still receive meaningful scores.
///
/// Edges are treated as undirected and unweighted. An optional `edge_filter`
/// restricts which edges are traversed.
///
/// Returns a list of `(NodeId, closeness)` sorted descending by score.
pub fn closeness_centrality(
    adjacency: &AdjacencyStore,
    edge_filter: &EdgeFilter,
) -> Vec<(NodeId, f64)> {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return Vec::new();
    }

    let mut result: Vec<(NodeId, f64)> = Vec::with_capacity(all_nodes.len());

    for &source in &all_nodes {
        // BFS to compute shortest distances from source (undirected)
        let mut dist: HashMap<NodeId, u64> = HashMap::new();
        dist.insert(source, 0);
        let mut queue: VecDeque<NodeId> = VecDeque::new();
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            let d_v = dist[&v];
            let out_nbrs = adjacency.neighbors_out(v, None);
            let in_nbrs = adjacency.neighbors_in(v, None);
            for &(w, eid) in out_nbrs.iter().chain(in_nbrs.iter()) {
                if !edge_passes_filter(adjacency, eid, edge_filter) {
                    continue;
                }
                if !dist.contains_key(&w) {
                    dist.insert(w, d_v + 1);
                    queue.push_back(w);
                }
            }
        }

        let reachable = dist.len() - 1; // exclude self
        if reachable == 0 {
            result.push((source, 0.0));
        } else {
            let total_dist: u64 = dist.values().sum();
            result.push((source, reachable as f64 / total_dist as f64));
        }
    }

    result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0)));
    result
}

// ─── Triangle Counting / Clustering Coefficient ─────────────────────────────

/// Result of triangle counting and clustering coefficient computation.
pub struct TriangleResult {
    /// Total number of triangles in the graph (each triangle counted once).
    pub total_triangles: u64,
    /// Per-node data: `(node_id, triangle_count, clustering_coefficient)`.
    pub per_node: Vec<(NodeId, u32, f64)>,
}

/// Count triangles and compute local clustering coefficients.
///
/// Edges are treated as undirected. A triangle is a set of three mutually
/// connected nodes. The local clustering coefficient for a node with degree k
/// and t triangles is `2t / (k * (k - 1))`.
///
/// An optional `edge_filter` restricts which edges are considered.
pub fn triangle_count(
    adjacency: &AdjacencyStore,
    edge_filter: &EdgeFilter,
) -> TriangleResult {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return TriangleResult {
            total_triangles: 0,
            per_node: Vec::new(),
        };
    }

    // Build undirected neighbor sets (filtered)
    let mut neighbors: HashMap<NodeId, HashSet<NodeId>> = HashMap::with_capacity(all_nodes.len());
    for &node in &all_nodes {
        let mut nbr_set = HashSet::new();
        for &(w, eid) in adjacency.neighbors_out(node, None).iter() {
            if edge_passes_filter(adjacency, eid, edge_filter) {
                nbr_set.insert(w);
            }
        }
        for &(w, eid) in adjacency.neighbors_in(node, None).iter() {
            if edge_passes_filter(adjacency, eid, edge_filter) {
                nbr_set.insert(w);
            }
        }
        nbr_set.remove(&node); // remove self-loops
        neighbors.insert(node, nbr_set);
    }

    // Count triangles per node. For each node u, for each pair of neighbors
    // (v, w) where v < w, check if v and w are neighbors. Each triangle
    // u-v-w is counted once at each of the three vertices.
    let mut node_triangles: HashMap<NodeId, u32> = all_nodes.iter().map(|&n| (n, 0)).collect();
    let mut total: u64 = 0;

    for &u in &all_nodes {
        let u_nbrs = &neighbors[&u];
        let u_nbrs_sorted: Vec<NodeId> = {
            let mut v: Vec<NodeId> = u_nbrs.iter().copied().collect();
            v.sort_unstable();
            v
        };

        for (i, &v) in u_nbrs_sorted.iter().enumerate() {
            if v <= u {
                continue; // only count each triangle once: u < v < w
            }
            for &w in &u_nbrs_sorted[i + 1..] {
                if neighbors[&v].contains(&w) {
                    // Triangle u-v-w found
                    *node_triangles.get_mut(&u).unwrap() += 1;
                    *node_triangles.get_mut(&v).unwrap() += 1;
                    *node_triangles.get_mut(&w).unwrap() += 1;
                    total += 1;
                }
            }
        }
    }

    let per_node: Vec<(NodeId, u32, f64)> = all_nodes
        .iter()
        .map(|&n| {
            let t = node_triangles[&n];
            let k = neighbors[&n].len() as u64;
            let cc = if k >= 2 {
                (2 * t as u64) as f64 / (k * (k - 1)) as f64
            } else {
                0.0
            };
            (n, t, cc)
        })
        .collect();

    TriangleResult {
        total_triangles: total,
        per_node,
    }
}

// ─── Strongly Connected Components (Tarjan's SCC) ───────────────────────────

/// Find all strongly connected components using Tarjan's algorithm.
///
/// Unlike `connected_components` which treats edges as undirected, this respects
/// edge direction. A strongly connected component is a maximal set of nodes
/// where every node is reachable from every other node following directed edges.
///
/// Returns a list of components, each being a `Vec<NodeId>`. Components are
/// returned in reverse topological order of the condensation DAG.
pub fn tarjan_scc(adjacency: &AdjacencyStore) -> Vec<Vec<NodeId>> {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return Vec::new();
    }

    let mut index_counter: u64 = 0;
    let mut stack: Vec<NodeId> = Vec::new();
    let mut on_stack: HashSet<NodeId> = HashSet::new();
    let mut index_map: HashMap<NodeId, u64> = HashMap::new();
    let mut lowlink: HashMap<NodeId, u64> = HashMap::new();
    let mut result: Vec<Vec<NodeId>> = Vec::new();

    // Iterative Tarjan to avoid stack overflow on large graphs.
    // Each frame on the work stack represents either entering a node
    // or returning from a child.
    enum Frame {
        Enter(NodeId),
        Resume {
            node: NodeId,
            neighbors: Vec<NodeId>,
            next_idx: usize,
        },
    }

    for &start in &all_nodes {
        if index_map.contains_key(&start) {
            continue;
        }

        let mut work: Vec<Frame> = vec![Frame::Enter(start)];

        while let Some(frame) = work.pop() {
            match frame {
                Frame::Enter(v) => {
                    if index_map.contains_key(&v) {
                        continue;
                    }
                    let idx = index_counter;
                    index_counter += 1;
                    index_map.insert(v, idx);
                    lowlink.insert(v, idx);
                    stack.push(v);
                    on_stack.insert(v);

                    let neighbors: Vec<NodeId> = adjacency
                        .neighbors_out(v, None)
                        .iter()
                        .map(|&(n, _)| n)
                        .collect();

                    work.push(Frame::Resume {
                        node: v,
                        neighbors: neighbors.clone(),
                        next_idx: 0,
                    });
                }
                Frame::Resume {
                    node: v,
                    neighbors,
                    next_idx,
                } => {
                    // Process the child we just returned from (if any)
                    if next_idx > 0 {
                        let child = neighbors[next_idx - 1];
                        if let Some(&child_ll) = lowlink.get(&child) {
                            let v_ll = lowlink[&v];
                            if child_ll < v_ll {
                                lowlink.insert(v, child_ll);
                            }
                        }
                    }

                    // Find next unvisited/on-stack child
                    let mut i = next_idx;
                    while i < neighbors.len() {
                        let w = neighbors[i];
                        if !index_map.contains_key(&w) {
                            // Push resume frame, then enter child
                            work.push(Frame::Resume {
                                node: v,
                                neighbors: neighbors.clone(),
                                next_idx: i + 1,
                            });
                            work.push(Frame::Enter(w));
                            break;
                        } else if on_stack.contains(&w) {
                            let v_ll = lowlink[&v];
                            let w_idx = index_map[&w];
                            if w_idx < v_ll {
                                lowlink.insert(v, w_idx);
                            }
                        }
                        i += 1;
                    }

                    // If we exhausted all neighbors, check if v is a root
                    if i == neighbors.len() {
                        if lowlink[&v] == index_map[&v] {
                            let mut component = Vec::new();
                            loop {
                                let w = stack.pop().unwrap();
                                on_stack.remove(&w);
                                component.push(w);
                                if w == v {
                                    break;
                                }
                            }
                            result.push(component);
                        }
                    }
                }
            }
        }
    }

    result
}

// ─── Topological Sort ────────────────────────────────────────────────────────

/// Compute a topological ordering of the graph using Kahn's algorithm.
///
/// Returns an error if the graph contains a cycle (i.e., is not a DAG).
/// Nodes with no dependencies appear first in the result.
pub fn topological_sort(adjacency: &AdjacencyStore) -> WeavResult<Vec<NodeId>> {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return Ok(Vec::new());
    }

    // Compute in-degree for each node
    let mut in_degree: HashMap<NodeId, usize> = all_nodes.iter().map(|&n| (n, 0)).collect();
    for &node in &all_nodes {
        for &(target, _) in adjacency.neighbors_out(node, None).iter() {
            *in_degree.entry(target).or_insert(0) += 1;
        }
    }

    // Start with all nodes that have in-degree 0
    // Use a BinaryHeap (max-heap with Reverse) for deterministic output
    let mut queue: BinaryHeap<std::cmp::Reverse<NodeId>> = BinaryHeap::new();
    for (&node, &deg) in &in_degree {
        if deg == 0 {
            queue.push(std::cmp::Reverse(node));
        }
    }

    let mut sorted: Vec<NodeId> = Vec::with_capacity(all_nodes.len());

    while let Some(std::cmp::Reverse(node)) = queue.pop() {
        sorted.push(node);
        for &(target, _) in adjacency.neighbors_out(node, None).iter() {
            if let Some(deg) = in_degree.get_mut(&target) {
                *deg -= 1;
                if *deg == 0 {
                    queue.push(std::cmp::Reverse(target));
                }
            }
        }
    }

    if sorted.len() != all_nodes.len() {
        return Err(WeavError::Conflict(
            "graph contains a cycle — topological sort requires a DAG".into(),
        ));
    }

    Ok(sorted)
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

    // ── Round 5 edge-case tests ──────────────────────────────────────────

    #[test]
    fn test_bfs_empty_graph_no_seeds() {
        let adj = AdjacencyStore::new();
        let result = bfs(&adj, &[], 3, 100, &EdgeFilter::none(), &NodeFilter::none(), Direction::Outgoing, None, None);
        assert!(result.visited_nodes.is_empty());
        assert!(result.visited_edges.is_empty());
    }

    #[test]
    fn test_bfs_seed_not_in_graph() {
        let adj = AdjacencyStore::new();
        let result = bfs(&adj, &[999], 3, 100, &EdgeFilter::none(), &NodeFilter::none(), Direction::Outgoing, None, None);
        // Seed is pushed into visited even if not in graph, but no expansion
        assert_eq!(result.visited_nodes.len(), 1);
        assert_eq!(result.visited_nodes[0], 999);
        assert!(result.visited_edges.is_empty());
    }

    #[test]
    fn test_bfs_disconnected_components() {
        let mut adj = AdjacencyStore::new();
        // Component 1: 1->2
        adj.add_node(1);
        adj.add_node(2);
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        // Component 2: 3->4
        adj.add_node(3);
        adj.add_node(4);
        adj.add_edge(3, 4, 0, make_meta(3, 4, 0)).unwrap();

        let result = bfs(&adj, &[1], 5, 100, &EdgeFilter::none(), &NodeFilter::none(), Direction::Outgoing, None, None);
        assert_eq!(result.visited_nodes.len(), 2);
        assert!(result.visited_nodes.contains(&1));
        assert!(result.visited_nodes.contains(&2));
        assert!(!result.visited_nodes.contains(&3));
        assert!(!result.visited_nodes.contains(&4));
    }

    #[test]
    fn test_bfs_cycle_handling() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let result = bfs(&adj, &[1], 10, 100, &EdgeFilter::none(), &NodeFilter::none(), Direction::Outgoing, None, None);
        // Should visit each node exactly once (seen set prevents revisits)
        assert_eq!(result.visited_nodes.len(), 3);
    }

    #[test]
    fn test_bfs_max_depth_zero() {
        let adj = build_linear_graph();
        let result = bfs(&adj, &[1], 0, 100, &EdgeFilter::none(), &NodeFilter::none(), Direction::Outgoing, None, None);
        // max_depth=0: seeds only, no expansion
        assert_eq!(result.visited_nodes.len(), 1);
        assert_eq!(result.visited_nodes[0], 1);
    }

    #[test]
    fn test_bfs_max_depth_255() {
        let adj = build_linear_graph(); // 4 nodes
        let result = bfs(&adj, &[1], 255, 100, &EdgeFilter::none(), &NodeFilter::none(), Direction::Outgoing, None, None);
        // Should traverse entire graph without overflow
        assert_eq!(result.visited_nodes.len(), 4);
    }

    #[test]
    fn test_bfs_max_nodes_limit() {
        let adj = build_star_graph(); // 5 nodes: 1->2, 1->3, 1->4, 1->5
        let result = bfs(&adj, &[1], 5, 2, &EdgeFilter::none(), &NodeFilter::none(), Direction::Outgoing, None, None);
        // max_nodes=2: should stop after 2 nodes
        assert_eq!(result.visited_nodes.len(), 2);
    }

    #[test]
    fn test_flow_score_no_seeds() {
        let adj = build_linear_graph();
        let result = flow_score(&adj, &[], 0.5, 0.01, 2);
        assert!(result.is_empty());
    }

    #[test]
    fn test_flow_score_unreachable_nodes() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2); // no edges from 1 to 2

        let result = flow_score(&adj, &[(1, 1.0)], 0.5, 0.01, 5);
        // Only seed node should be in results
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].node_id, 1);
    }

    #[test]
    fn test_flow_score_zero_alpha() {
        let adj = build_linear_graph();
        let result = flow_score(&adj, &[(1, 1.0)], 0.0, 0.01, 3);
        // alpha=0: propagated = 0.0 < theta, no propagation
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].node_id, 1);
    }

    #[test]
    fn test_flow_score_theta_larger_than_seed() {
        let adj = build_linear_graph();
        let result = flow_score(&adj, &[(1, 0.5)], 0.5, 1.0, 3);
        // Propagated = 0.5 * 0.5 = 0.25 < theta(1.0), no propagation
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_shortest_path_same_source_target() {
        let adj = build_linear_graph();
        let path = shortest_path(&adj, 1, 1, 10);
        assert_eq!(path, Some(vec![1]));
    }

    #[test]
    fn test_shortest_path_no_path() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2); // disconnected

        let path = shortest_path(&adj, 1, 2, 10);
        assert!(path.is_none());
    }

    #[test]
    fn test_shortest_path_depth_limited() {
        let adj = build_linear_graph(); // 1->2->3->4
        // Path 1->3 requires depth 2, but max_depth=1
        let path = shortest_path(&adj, 1, 3, 1);
        assert!(path.is_none());
    }

    #[test]
    fn test_ego_network_isolated_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(42);

        let sub = ego_network(&adj, 42, 2);
        assert_eq!(sub.nodes.len(), 1);
        assert_eq!(sub.nodes[0], 42);
        assert!(sub.edges.is_empty());
    }

    #[test]
    fn test_scored_paths_one_anchor() {
        let adj = build_linear_graph();
        // Single anchor: no pairs to form, so no paths
        let paths = scored_paths(&adj, &[(1, 1.0)], 10, 5);
        assert!(paths.is_empty());
    }

    // ── Dijkstra tests ──────────────────────────────────────────────────────

    #[test]
    fn test_dijkstra_same_node() {
        let adj = build_linear_graph();
        let result = dijkstra_shortest_path(&adj, 1, 1, 10).unwrap();
        assert_eq!(result.nodes, vec![1]);
        assert_eq!(result.total_weight, 0.0);
    }

    #[test]
    fn test_dijkstra_direct_edge() {
        let adj = build_linear_graph();
        let result = dijkstra_shortest_path(&adj, 1, 2, 10).unwrap();
        assert_eq!(result.nodes, vec![1, 2]);
        assert!(result.total_weight > 0.0);
    }

    #[test]
    fn test_dijkstra_multi_hop() {
        let adj = build_linear_graph();
        let result = dijkstra_shortest_path(&adj, 1, 4, 10).unwrap();
        assert_eq!(result.nodes, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_dijkstra_unreachable() {
        let adj = build_linear_graph();
        let result = dijkstra_shortest_path(&adj, 4, 1, 10);
        assert!(result.is_none());
    }

    #[test]
    fn test_dijkstra_depth_limited() {
        let adj = build_linear_graph();
        // Path 1->4 needs 3 hops, but max_depth=2
        let result = dijkstra_shortest_path(&adj, 1, 4, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_dijkstra_prefers_high_weight() {
        // Two paths from 1 to 3:
        //   1 -> 2 -> 3 (weights 0.1, 0.1)  cost = 1/0.1 + 1/0.1 = 20
        //   1 -> 3 directly (weight 1.0)     cost = 1/1.0 = 1
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);

        let meta_12 = EdgeMeta {
            source: 1, target: 2, label: 0,
            temporal: BiTemporal::new_current(1000),
            provenance: None, weight: 0.1, token_cost: 0,
        };
        adj.add_edge(1, 2, 0, meta_12).unwrap();

        let meta_23 = EdgeMeta {
            source: 2, target: 3, label: 0,
            temporal: BiTemporal::new_current(1000),
            provenance: None, weight: 0.1, token_cost: 0,
        };
        adj.add_edge(2, 3, 0, meta_23).unwrap();

        let meta_13 = EdgeMeta {
            source: 1, target: 3, label: 0,
            temporal: BiTemporal::new_current(1000),
            provenance: None, weight: 1.0, token_cost: 0,
        };
        adj.add_edge(1, 3, 0, meta_13).unwrap();

        let result = dijkstra_shortest_path(&adj, 1, 3, 10).unwrap();
        // Should prefer direct path (1->3) because higher weight = lower cost
        assert_eq!(result.nodes, vec![1, 3]);
    }

    // ── Connected components tests ──────────────────────────────────────────

    #[test]
    fn test_connected_components_single() {
        let adj = build_linear_graph(); // 1->2->3->4 = one component
        let components = connected_components(&adj);
        assert_eq!(components.len(), 4);
        let c = components[&1];
        assert_eq!(components[&2], c);
        assert_eq!(components[&3], c);
        assert_eq!(components[&4], c);
    }

    #[test]
    fn test_connected_components_two() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);
        adj.add_node(4);
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(3, 4, 0, make_meta(3, 4, 0)).unwrap();

        let components = connected_components(&adj);
        assert_eq!(components.len(), 4);
        assert_eq!(components[&1], components[&2]);
        assert_eq!(components[&3], components[&4]);
        assert_ne!(components[&1], components[&3]);
    }

    #[test]
    fn test_connected_components_isolated() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);
        // No edges: each node is its own component
        let components = connected_components(&adj);
        assert_eq!(components.len(), 3);
        let unique: HashSet<u32> = components.values().copied().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_connected_components_empty() {
        let adj = AdjacencyStore::new();
        let components = connected_components(&adj);
        assert!(components.is_empty());
    }

    // ── Personalized PageRank tests ─────────────────────────────────────────

    #[test]
    fn test_ppr_linear_graph() {
        let adj = build_linear_graph(); // 1->2->3->4
        let result = personalized_pagerank(&adj, &[(1, 1.0)], 0.15, 100, 1e-6);
        assert_eq!(result.len(), 4);
        // Node 1 (seed) should have highest PPR score
        assert_eq!(result[0].node_id, 1);
        // Scores should be decreasing (further from seed = lower score)
        let score_map: HashMap<NodeId, f32> = result.iter().map(|s| (s.node_id, s.score)).collect();
        assert!(score_map[&1] > score_map[&2]);
        assert!(score_map[&2] > score_map[&3]);
    }

    #[test]
    fn test_ppr_star_graph() {
        let adj = build_star_graph(); // 1->2, 1->3, 1->4, 1->5
        let result = personalized_pagerank(&adj, &[(1, 1.0)], 0.15, 100, 1e-6);
        assert_eq!(result.len(), 5);
        // Node 1 (seed + hub) should have highest score
        assert_eq!(result[0].node_id, 1);
        // All spokes should have roughly equal scores
        let spoke_scores: Vec<f32> = result
            .iter()
            .filter(|s| s.node_id != 1)
            .map(|s| s.score)
            .collect();
        let max_spoke = spoke_scores.iter().cloned().fold(f32::MIN, f32::max);
        let min_spoke = spoke_scores.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            (max_spoke - min_spoke).abs() < 0.01,
            "Spoke scores should be roughly equal"
        );
    }

    #[test]
    fn test_ppr_empty_seeds() {
        let adj = build_linear_graph();
        let result = personalized_pagerank(&adj, &[], 0.15, 100, 1e-6);
        assert!(result.is_empty());
    }

    #[test]
    fn test_ppr_multiple_seeds() {
        let adj = build_linear_graph(); // 1->2->3->4
        let result = personalized_pagerank(&adj, &[(1, 1.0), (4, 1.0)], 0.15, 100, 1e-6);
        assert_eq!(result.len(), 4);
        // Both seed nodes should have high scores
        let score_map: HashMap<NodeId, f32> = result.iter().map(|s| (s.node_id, s.score)).collect();
        // Node 1 and 4 (seeds) should score higher than interior nodes
        assert!(score_map[&1] > score_map[&3]);
    }

    // ── Label Propagation tests ─────────────────────────────────────────────

    #[test]
    fn test_label_propagation_two_cliques() {
        // Two cliques connected by a single edge:
        // Clique 1: 1-2, 1-3, 2-3
        // Clique 2: 4-5, 4-6, 5-6
        // Bridge: 3-4
        let mut adj = AdjacencyStore::new();
        for i in 1..=6 { adj.add_node(i); }
        // Clique 1
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        // Clique 2
        adj.add_edge(4, 5, 0, make_meta(4, 5, 0)).unwrap();
        adj.add_edge(5, 4, 0, make_meta(5, 4, 0)).unwrap();
        adj.add_edge(4, 6, 0, make_meta(4, 6, 0)).unwrap();
        adj.add_edge(6, 4, 0, make_meta(6, 4, 0)).unwrap();
        adj.add_edge(5, 6, 0, make_meta(5, 6, 0)).unwrap();
        adj.add_edge(6, 5, 0, make_meta(6, 5, 0)).unwrap();
        // Bridge
        adj.add_edge(3, 4, 0, make_meta(3, 4, 0)).unwrap();
        adj.add_edge(4, 3, 0, make_meta(4, 3, 0)).unwrap();

        let communities = label_propagation(&adj, 20);
        assert_eq!(communities.len(), 6);
        // Nodes in same clique should have same label
        assert_eq!(communities[&1], communities[&2]);
        assert_eq!(communities[&1], communities[&3]);
        assert_eq!(communities[&4], communities[&5]);
        assert_eq!(communities[&4], communities[&6]);
        // Different cliques should have different labels
        assert_ne!(communities[&1], communities[&4]);
    }

    #[test]
    fn test_label_propagation_single_component() {
        let adj = build_linear_graph(); // 1->2->3->4
        let communities = label_propagation(&adj, 20);
        assert_eq!(communities.len(), 4);
        // Linear graph: all nodes should converge to same community
        let c = communities[&1];
        assert_eq!(communities[&2], c);
        assert_eq!(communities[&3], c);
        assert_eq!(communities[&4], c);
    }

    #[test]
    fn test_label_propagation_isolated_nodes() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);
        // No edges — each node stays in its own community
        let communities = label_propagation(&adj, 10);
        assert_eq!(communities.len(), 3);
        let unique: HashSet<u64> = communities.values().copied().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_label_propagation_empty_graph() {
        let adj = AdjacencyStore::new();
        let communities = label_propagation(&adj, 10);
        assert!(communities.is_empty());
    }

    // ── Modularity Communities tests ────────────────────────────────────────

    #[test]
    fn test_modularity_communities_two_cliques() {
        // Two cliques connected by a single edge:
        // Clique 1: 1-2, 1-3, 2-3
        // Clique 2: 4-5, 4-6, 5-6
        // Bridge: 3-4
        let mut adj = AdjacencyStore::new();
        for i in 1..=6 { adj.add_node(i); }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(4, 5, 0, make_meta(4, 5, 0)).unwrap();
        adj.add_edge(5, 4, 0, make_meta(5, 4, 0)).unwrap();
        adj.add_edge(4, 6, 0, make_meta(4, 6, 0)).unwrap();
        adj.add_edge(6, 4, 0, make_meta(6, 4, 0)).unwrap();
        adj.add_edge(5, 6, 0, make_meta(5, 6, 0)).unwrap();
        adj.add_edge(6, 5, 0, make_meta(6, 5, 0)).unwrap();
        adj.add_edge(3, 4, 0, make_meta(3, 4, 0)).unwrap();
        adj.add_edge(4, 3, 0, make_meta(4, 3, 0)).unwrap();

        let communities = modularity_communities(&adj, 20, 1.0);
        assert_eq!(communities.len(), 6);
        assert_eq!(communities[&1], communities[&2]);
        assert_eq!(communities[&1], communities[&3]);
        assert_eq!(communities[&4], communities[&5]);
        assert_eq!(communities[&4], communities[&6]);
        assert_ne!(communities[&1], communities[&4]);
    }

    #[test]
    fn test_modularity_communities_empty() {
        let adj = AdjacencyStore::new();
        let communities = modularity_communities(&adj, 10, 1.0);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_modularity_communities_isolated() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        let communities = modularity_communities(&adj, 10, 1.0);
        assert_eq!(communities.len(), 2);
        assert_ne!(communities[&1], communities[&2]);
    }

    #[test]
    fn test_modularity_resolution_parameter() {
        // Higher resolution should produce smaller/more communities
        let adj = build_star_graph(); // 1->2,3,4,5
        let comm_low = modularity_communities(&adj, 20, 0.5);
        let comm_high = modularity_communities(&adj, 20, 2.0);
        let unique_low: HashSet<u64> = comm_low.values().copied().collect();
        let unique_high: HashSet<u64> = comm_high.values().copied().collect();
        // High resolution should have >= as many communities as low resolution
        assert!(unique_high.len() >= unique_low.len());
    }

    #[test]
    fn test_dijkstra_zero_weight_edge() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        let meta = EdgeMeta {
            source: 1, target: 2, label: 0,
            temporal: BiTemporal::new_current(1000),
            provenance: None, weight: 0.0, token_cost: 0,
        };
        adj.add_edge(1, 2, 0, meta).unwrap();
        // Zero weight → cost = f64::MAX, so path exists but is very expensive
        let result = dijkstra_shortest_path(&adj, 1, 2, 10);
        assert!(result.is_some());
        assert_eq!(result.unwrap().nodes, vec![1, 2]);
    }

    #[test]
    fn test_label_propagation_self_loop() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_edge(1, 1, 0, make_meta(1, 1, 0)).unwrap(); // self-loop
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        let communities = label_propagation(&adj, 10);
        assert_eq!(communities.len(), 2);
        // Both should converge to same community
        assert_eq!(communities[&1], communities[&2]);
    }

    #[test]
    fn test_ppr_dangling_node() {
        // Node 3 has no outgoing edges (dangling)
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        // Node 3 is dangling - no outgoing edges
        let result = personalized_pagerank(&adj, &[(1, 1.0)], 0.15, 100, 1e-6);
        assert_eq!(result.len(), 3);
        // Should not crash, all nodes should have scores
        for s in &result {
            assert!(s.score >= 0.0);
            assert!(!s.score.is_nan());
        }
    }

    // ── Flow score cycle / convergence tests ────────────────────────────────

    #[test]
    fn test_flow_score_cycle() {
        // Cycle: 1 -> 2 -> 3 -> 1
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let result = flow_score(&adj, &[(1, 1.0)], 0.85, 0.01, 10);
        // Must terminate and produce scores for all 3 nodes
        assert_eq!(result.len(), 3);
        let score_map: HashMap<NodeId, f32> =
            result.iter().map(|s| (s.node_id, s.score)).collect();
        // Seed has highest score
        assert_eq!(score_map[&1], 1.0);
        // Scores decrease along the cycle
        assert!(score_map[&2] < score_map[&1]);
        assert!(score_map[&3] < score_map[&2]);
    }

    #[test]
    fn test_flow_score_convergent_paths() {
        // Diamond: 1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4
        // Node 4 is reachable via two paths
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);
        adj.add_node(4);
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(2, 4, 0, make_meta(2, 4, 0)).unwrap();
        adj.add_edge(3, 4, 0, make_meta(3, 4, 0)).unwrap();

        let result = flow_score(&adj, &[(1, 1.0)], 0.8, 0.01, 5);
        let score_map: HashMap<NodeId, f32> =
            result.iter().map(|s| (s.node_id, s.score)).collect();
        // Node 4: best path gives 1.0 * 0.8 * 0.8 = 0.64
        assert!((score_map[&4] - 0.64).abs() < 0.001);
        // Node 2 and 3 should have equal scores (symmetric)
        assert!((score_map[&2] - score_map[&3]).abs() < 0.001);
    }

    #[test]
    fn test_flow_score_nan_seed() {
        let adj = build_linear_graph();
        let result = flow_score(&adj, &[(1, f32::NAN)], 0.5, 0.01, 3);
        // NaN propagation: NaN * alpha = NaN, NaN < theta is false,
        // but NaN > existing is also false, so it shouldn't propagate
        // Seed itself is inserted with NaN score
        assert!(result.len() >= 1);
        // No non-seed node should have a score (NaN doesn't propagate)
        let non_seed: Vec<&ScoredNode> = result.iter().filter(|s| s.node_id != 1).collect();
        for s in &non_seed {
            // NaN comparisons: NaN > existing returns false, so no propagation
            assert!(!s.score.is_nan(), "NaN should not propagate to node {}", s.node_id);
        }
    }

    #[test]
    fn test_flow_score_multiple_seeds_different_scores() {
        // Seeds at both ends of a chain: 1(1.0) -> 2 -> 3 <- 5(0.5) <- 4
        let mut adj = AdjacencyStore::new();
        for i in 1..=5 { adj.add_node(i); }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(4, 5, 0, make_meta(4, 5, 0)).unwrap();
        adj.add_edge(5, 3, 0, make_meta(5, 3, 0)).unwrap();

        let result = flow_score(&adj, &[(1, 1.0), (4, 0.5)], 0.8, 0.01, 5);
        let score_map: HashMap<NodeId, f32> =
            result.iter().map(|s| (s.node_id, s.score)).collect();
        // Node 3 reachable from both seeds:
        //   via 1 -> 2 -> 3: 1.0 * 0.8 * 0.8 = 0.64
        //   via 4 -> 5 -> 3: 0.5 * 0.8 * 0.8 = 0.32
        // Best score should win: 0.64
        assert!((score_map[&3] - 0.64).abs() < 0.001);
    }

    #[test]
    fn test_flow_score_self_loop() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_edge(1, 1, 0, make_meta(1, 1, 0)).unwrap();

        let result = flow_score(&adj, &[(1, 1.0)], 0.9, 0.01, 5);
        // Self-loop: propagated = 1.0 * 0.9 = 0.9 < 1.0, no update
        // Should terminate with only the seed
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].node_id, 1);
        assert_eq!(result[0].score, 1.0);
    }

    // ── Betweenness Centrality tests ──────────────────────────────────────────

    #[test]
    fn test_betweenness_empty_graph() {
        let adj = AdjacencyStore::new();
        let result = betweenness_centrality(&adj, &EdgeFilter::none());
        assert!(result.is_empty());
    }

    #[test]
    fn test_betweenness_single_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        let result = betweenness_centrality(&adj, &EdgeFilter::none());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 1);
        assert_eq!(result[0].1, 0.0);
    }

    #[test]
    fn test_betweenness_linear_graph() {
        // 1 -- 2 -- 3 -- 4 (undirected via both directions)
        let mut adj = AdjacencyStore::new();
        for i in 1..=4 { adj.add_node(i); }
        for i in 1..=3 {
            adj.add_edge(i, i + 1, 0, make_meta(i, i + 1, 0)).unwrap();
            adj.add_edge(i + 1, i, 0, make_meta(i + 1, i, 0)).unwrap();
        }

        let result = betweenness_centrality(&adj, &EdgeFilter::none());
        let bc: HashMap<NodeId, f64> = result.into_iter().collect();
        // In a path graph of 4 nodes (undirected), interior nodes 2 and 3
        // have higher betweenness than endpoints 1 and 4.
        assert!(bc[&2] > bc[&1]);
        assert!(bc[&3] > bc[&4]);
        // Endpoints have zero betweenness
        assert_eq!(bc[&1], 0.0);
        assert_eq!(bc[&4], 0.0);
    }

    #[test]
    fn test_betweenness_star_graph() {
        // Center node 1 connects to 2,3,4,5 (undirected)
        let mut adj = AdjacencyStore::new();
        for i in 1..=5 { adj.add_node(i); }
        for i in 2..=5 {
            adj.add_edge(1, i, 0, make_meta(1, i, 0)).unwrap();
            adj.add_edge(i, 1, 0, make_meta(i, 1, 0)).unwrap();
        }

        let result = betweenness_centrality(&adj, &EdgeFilter::none());
        let bc: HashMap<NodeId, f64> = result.into_iter().collect();
        // Center node should have highest betweenness
        assert!(bc[&1] > 0.0);
        // All spokes have zero betweenness
        for i in 2..=5 {
            assert_eq!(bc[&i], 0.0);
        }
    }

    #[test]
    fn test_betweenness_disconnected() {
        // Two isolated nodes
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        let result = betweenness_centrality(&adj, &EdgeFilter::none());
        let bc: HashMap<NodeId, f64> = result.into_iter().collect();
        assert_eq!(bc[&1], 0.0);
        assert_eq!(bc[&2], 0.0);
    }

    // ── Closeness Centrality tests ────────────────────────────────────────────

    #[test]
    fn test_closeness_empty_graph() {
        let adj = AdjacencyStore::new();
        let result = closeness_centrality(&adj, &EdgeFilter::none());
        assert!(result.is_empty());
    }

    #[test]
    fn test_closeness_single_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        let result = closeness_centrality(&adj, &EdgeFilter::none());
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 1);
        assert_eq!(result[0].1, 0.0); // no reachable nodes
    }

    #[test]
    fn test_closeness_linear_graph() {
        // 1 -- 2 -- 3 (undirected)
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 { adj.add_node(i); }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();

        let result = closeness_centrality(&adj, &EdgeFilter::none());
        let cc: HashMap<NodeId, f64> = result.into_iter().collect();
        // Node 2 is central: distances = {1:1, 3:1}, sum=2, closeness=2/2=1.0
        // Node 1: distances = {2:1, 3:2}, sum=3, closeness=2/3
        assert!(cc[&2] > cc[&1]);
        assert!(cc[&2] > cc[&3]);
        // Endpoints should have equal closeness (symmetric)
        assert!((cc[&1] - cc[&3]).abs() < 1e-10);
    }

    #[test]
    fn test_closeness_disconnected() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        let result = closeness_centrality(&adj, &EdgeFilter::none());
        let cc: HashMap<NodeId, f64> = result.into_iter().collect();
        assert_eq!(cc[&1], 0.0);
        assert_eq!(cc[&2], 0.0);
    }

    #[test]
    fn test_closeness_complete_triangle() {
        // 1--2, 2--3, 1--3 (complete, undirected)
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 { adj.add_node(i); }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let result = closeness_centrality(&adj, &EdgeFilter::none());
        let cc: HashMap<NodeId, f64> = result.into_iter().collect();
        // All nodes equidistant: distance sum = 2, closeness = 2/2 = 1.0
        assert!((cc[&1] - 1.0).abs() < 1e-10);
        assert!((cc[&2] - 1.0).abs() < 1e-10);
        assert!((cc[&3] - 1.0).abs() < 1e-10);
    }

    // ── Triangle Counting / Clustering Coefficient tests ──────────────────────

    #[test]
    fn test_triangles_empty_graph() {
        let adj = AdjacencyStore::new();
        let result = triangle_count(&adj, &EdgeFilter::none());
        assert_eq!(result.total_triangles, 0);
        assert!(result.per_node.is_empty());
    }

    #[test]
    fn test_triangles_single_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        let result = triangle_count(&adj, &EdgeFilter::none());
        assert_eq!(result.total_triangles, 0);
        assert_eq!(result.per_node.len(), 1);
        assert_eq!(result.per_node[0], (1, 0, 0.0));
    }

    #[test]
    fn test_triangles_single_triangle() {
        // Triangle: 1-2-3 (undirected)
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 { adj.add_node(i); }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let result = triangle_count(&adj, &EdgeFilter::none());
        assert_eq!(result.total_triangles, 1);
        let node_map: HashMap<NodeId, (u32, f64)> = result
            .per_node.iter().map(|&(n, t, c)| (n, (t, c))).collect();
        // Each node participates in 1 triangle
        assert_eq!(node_map[&1].0, 1);
        assert_eq!(node_map[&2].0, 1);
        assert_eq!(node_map[&3].0, 1);
        // Clustering coefficient = 2*1 / (2*1) = 1.0
        assert!((node_map[&1].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangles_no_triangle() {
        // Star: 1-2, 1-3, 1-4 (no triangles since spokes not connected)
        let mut adj = AdjacencyStore::new();
        for i in 1..=4 { adj.add_node(i); }
        for i in 2..=4 {
            adj.add_edge(1, i, 0, make_meta(1, i, 0)).unwrap();
            adj.add_edge(i, 1, 0, make_meta(i, 1, 0)).unwrap();
        }

        let result = triangle_count(&adj, &EdgeFilter::none());
        assert_eq!(result.total_triangles, 0);
        let node_map: HashMap<NodeId, (u32, f64)> = result
            .per_node.iter().map(|&(n, t, c)| (n, (t, c))).collect();
        // Center has 3 neighbors but 0 triangles → cc = 0
        assert_eq!(node_map[&1].0, 0);
        assert_eq!(node_map[&1].1, 0.0);
    }

    #[test]
    fn test_triangles_disconnected() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        let result = triangle_count(&adj, &EdgeFilter::none());
        assert_eq!(result.total_triangles, 0);
    }

    #[test]
    fn test_triangles_two_triangles_shared_edge() {
        // 1-2-3 triangle + 2-3-4 triangle (share edge 2-3)
        let mut adj = AdjacencyStore::new();
        for i in 1..=4 { adj.add_node(i); }
        // Triangle 1-2-3
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();
        // Triangle 2-3-4
        adj.add_edge(3, 4, 0, make_meta(3, 4, 0)).unwrap();
        adj.add_edge(4, 3, 0, make_meta(4, 3, 0)).unwrap();
        adj.add_edge(2, 4, 0, make_meta(2, 4, 0)).unwrap();
        adj.add_edge(4, 2, 0, make_meta(4, 2, 0)).unwrap();

        let result = triangle_count(&adj, &EdgeFilter::none());
        assert_eq!(result.total_triangles, 2);
        let node_map: HashMap<NodeId, (u32, f64)> = result
            .per_node.iter().map(|&(n, t, c)| (n, (t, c))).collect();
        // Nodes 2 and 3 participate in both triangles
        assert_eq!(node_map[&2].0, 2);
        assert_eq!(node_map[&3].0, 2);
        // Nodes 1 and 4 participate in one triangle each
        assert_eq!(node_map[&1].0, 1);
        assert_eq!(node_map[&4].0, 1);
    }

    // ── Tarjan's SCC tests ────────────────────────────────────────────────────

    #[test]
    fn test_scc_empty_graph() {
        let adj = AdjacencyStore::new();
        let result = tarjan_scc(&adj);
        assert!(result.is_empty());
    }

    #[test]
    fn test_scc_single_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        let result = tarjan_scc(&adj);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], vec![1]);
    }

    #[test]
    fn test_scc_simple_cycle() {
        // 1 -> 2 -> 3 -> 1
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 { adj.add_node(i); }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let result = tarjan_scc(&adj);
        assert_eq!(result.len(), 1); // all 3 nodes in one SCC
        let mut component = result[0].clone();
        component.sort();
        assert_eq!(component, vec![1, 2, 3]);
    }

    #[test]
    fn test_scc_dag() {
        // 1 -> 2 -> 3 (no cycles, each node is its own SCC)
        let adj = build_linear_graph(); // 1->2->3->4
        let result = tarjan_scc(&adj);
        assert_eq!(result.len(), 4); // 4 singleton SCCs
        for component in &result {
            assert_eq!(component.len(), 1);
        }
    }

    #[test]
    fn test_scc_two_components() {
        // SCC 1: 1 -> 2 -> 1
        // SCC 2: 3 -> 4 -> 3
        // Bridge: 2 -> 3 (not creating a larger SCC)
        let mut adj = AdjacencyStore::new();
        for i in 1..=4 { adj.add_node(i); }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 4, 0, make_meta(3, 4, 0)).unwrap();
        adj.add_edge(4, 3, 0, make_meta(4, 3, 0)).unwrap();

        let result = tarjan_scc(&adj);
        assert_eq!(result.len(), 2);
        let mut sizes: Vec<usize> = result.iter().map(|c| c.len()).collect();
        sizes.sort();
        assert_eq!(sizes, vec![2, 2]);
    }

    #[test]
    fn test_scc_disconnected() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        let result = tarjan_scc(&adj);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_scc_self_loop() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_edge(1, 1, 0, make_meta(1, 1, 0)).unwrap();
        let result = tarjan_scc(&adj);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], vec![1]);
    }

    // ── Topological Sort tests ────────────────────────────────────────────────

    #[test]
    fn test_topo_sort_empty_graph() {
        let adj = AdjacencyStore::new();
        let result = topological_sort(&adj).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_topo_sort_single_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        let result = topological_sort(&adj).unwrap();
        assert_eq!(result, vec![1]);
    }

    #[test]
    fn test_topo_sort_linear() {
        let adj = build_linear_graph(); // 1->2->3->4
        let result = topological_sort(&adj).unwrap();
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_topo_sort_diamond() {
        // 1 -> 2, 1 -> 3, 2 -> 4, 3 -> 4
        let mut adj = AdjacencyStore::new();
        for i in 1..=4 { adj.add_node(i); }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(2, 4, 0, make_meta(2, 4, 0)).unwrap();
        adj.add_edge(3, 4, 0, make_meta(3, 4, 0)).unwrap();

        let result = topological_sort(&adj).unwrap();
        // Node 1 must come first, node 4 must come last
        assert_eq!(result[0], 1);
        assert_eq!(result[3], 4);
        // All 4 nodes present
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_topo_sort_cycle_error() {
        // 1 -> 2 -> 3 -> 1
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 { adj.add_node(i); }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let result = topological_sort(&adj);
        assert!(result.is_err());
        match result.unwrap_err() {
            WeavError::Conflict(msg) => assert!(msg.contains("cycle")),
            other => panic!("expected Conflict, got: {:?}", other),
        }
    }

    #[test]
    fn test_topo_sort_disconnected() {
        // Two independent nodes (both have in-degree 0)
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        let result = topological_sort(&adj).unwrap();
        assert_eq!(result.len(), 2);
        // Deterministic ordering: smallest node ID first
        assert_eq!(result, vec![1, 2]);
    }

    #[test]
    fn test_topo_sort_self_loop_error() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_edge(1, 1, 0, make_meta(1, 1, 0)).unwrap();

        let result = topological_sort(&adj);
        assert!(result.is_err());
    }
}
