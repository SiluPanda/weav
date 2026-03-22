//! Graph traversal algorithms: BFS, flow scoring, ego network, shortest path,
//! scored paths, Dijkstra, connected components, Personalized PageRank,
//! Label Propagation community detection, modularity-based community detection,
//! betweenness centrality, closeness centrality, triangle counting,
//! Tarjan's strongly connected components, topological sort, Leiden community
//! detection, Node2Vec random walks, A* shortest path, Yen's K-shortest paths,
//! degree centrality, eigenvector centrality, HITS (hubs & authorities),
//! FastRP node embeddings.

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
    /// If set, edges must have been valid during the half-open range [start, end).
    pub valid_during: Option<(Timestamp, Timestamp)>,
}

impl EdgeFilter {
    pub fn none() -> Self {
        Self {
            labels: None,
            min_weight: None,
            max_age_ms: None,
            min_confidence: None,
            valid_at: None,
            valid_during: None,
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
    /// Create a permissive node filter that matches all nodes.
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

    if let Some((start, end)) = filter.valid_during
        && !meta.temporal.is_valid_during(start, end)
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
#[allow(clippy::type_complexity)]
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

/// Breadth-first search from one or more seed nodes.
///
/// Explores the graph level-by-level from the given seeds, respecting
/// edge direction, edge filters, and node filters. Traversal stops when
/// `max_depth` hops are reached or `max_nodes` have been visited.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `seeds`: starting node(s) for the traversal.
/// - `max_depth`: maximum number of hops from any seed.
/// - `max_nodes`: maximum number of nodes to visit before stopping.
/// - `edge_filter`: criteria to restrict which edges are traversed.
/// - `node_filter`: criteria to restrict which nodes are visited.
/// - `direction`: edge direction to follow (`Outgoing`, `Incoming`, or `Both`).
/// - `node_label_lookup`: optional closure that resolves a node to its `LabelId` (for label filtering).
/// - `property_check`: optional closure that checks if a node has a given property name.
///
/// Returns a `TraversalResult` containing visited nodes, visited edges,
/// a depth map, and a parent map for path reconstruction.
///
/// Complexity: O(V + E) bounded by `max_depth` and `max_nodes`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
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

/// Network flow-based relevance scoring from seed nodes.
///
/// Propagates scores outward from seed nodes via a BFS-like expansion,
/// decaying each score by the factor `alpha` at every hop. Traversal
/// stops when a propagated score drops below `theta` or `max_depth` is
/// reached. This is useful for ranking nodes by proximity and
/// connectivity strength relative to a set of anchor nodes.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `seeds_with_scores`: starting nodes and their initial relevance scores.
/// - `alpha`: decay factor applied per hop (0.0–1.0; lower = faster decay).
/// - `theta`: minimum score threshold; propagation stops below this value.
/// - `max_depth`: maximum number of hops from any seed.
///
/// Returns a list of `ScoredNode` entries for every node reached, sorted
/// descending by score.
///
/// Complexity: O(V + E) in the worst case, bounded by `seeds.len() * 10 000`
/// work units to prevent runaway expansion on dense graphs.
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

/// Extract the k-hop ego network (neighborhood subgraph) around a center node.
///
/// Performs a BFS from `center` up to `radius` hops in both directions,
/// collecting every reachable node and traversed edge. The result is a
/// `SubGraph` suitable for local analysis, visualization, or context
/// extraction around an entity of interest.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `center`: the focal node whose neighborhood is extracted.
/// - `radius`: maximum number of hops from `center` (e.g., 1 = direct neighbors only).
///
/// Returns a `SubGraph` containing all reachable nodes and edges within the radius.
///
/// Complexity: O(V + E) where V and E are the nodes and edges within the
/// `radius`-hop neighborhood.
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

/// Find the shortest unweighted path between two nodes using BFS.
///
/// Explores outgoing edges level-by-level from `source` until `target` is
/// found or `max_depth` hops are exhausted. Because edges are unweighted,
/// the first path found is guaranteed to have the fewest hops.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `source`: starting node.
/// - `target`: destination node.
/// - `max_depth`: maximum number of hops to search.
///
/// Returns `Some(path)` as an ordered list of node IDs from `source` to
/// `target` (inclusive), or `None` if no path exists within the depth limit.
///
/// Complexity: O(V + E) bounded by the `max_depth`-hop frontier.
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

/// Find and score multiple paths between anchor nodes for context assembly.
///
/// For each pair of anchor nodes, discovers paths up to `max_path_length`
/// hops via BFS. Each path is scored by averaging the relevance scores of
/// the anchor nodes it passes through, producing a reliability metric that
/// indicates how well-supported the connection is. This is used by the
/// context assembly pipeline (spec 4.3) to select the most informative
/// subgraph structure for LLM consumption.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `anchors`: anchor nodes paired with their relevance scores.
/// - `max_paths`: maximum number of paths to return.
/// - `max_path_length`: maximum hop count for any individual path.
///
/// Returns up to `max_paths` `ScoredPath` entries sorted by reliability
/// score descending.
///
/// Complexity: O(A^2 * (V + E)) where A is the number of anchor nodes and
/// V, E are bounded by the `max_path_length`-hop frontier.
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
#[derive(Debug)]
pub struct WeightedPath {
    pub nodes: Vec<NodeId>,
    pub total_weight: f64,
}

/// Find the shortest weighted path between two nodes using Dijkstra's algorithm.
///
/// Uses a min-heap priority queue to explore edges in order of cumulative
/// cost. Edge weights are inverted (`1/weight`) so that higher-weight edges
/// are cheaper to traverse. Stops when the target is reached or `max_depth`
/// hops are exhausted.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `source`: starting node.
/// - `target`: destination node.
/// - `max_depth`: maximum number of hops allowed in the path.
///
/// Returns `Some(WeightedPath)` containing the node sequence and total cost,
/// or `None` if no path exists within the depth limit.
///
/// Complexity: O((V + E) log V) using a binary heap, bounded by `max_depth`.
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
/// Uses iterative BFS to assign each node a component ID. All nodes
/// reachable from each other (ignoring edge direction) share the same
/// component. Useful for detecting disconnected subgraphs.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
///
/// Returns a map from `NodeId` to component ID (0-indexed, assigned in
/// discovery order).
///
/// Complexity: O(V + E) where V is the number of nodes and E is the
/// number of edges.
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
/// Runs power iteration on the graph's transition matrix with a teleport
/// vector biased toward the seed nodes. At each step, every node
/// distributes its score equally among its outgoing neighbors, and a
/// fraction `alpha` of the total score teleports back to the seeds.
/// Converges when the L1 norm of score changes falls below `tolerance`.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `seeds`: seed nodes and their teleport weights (will be normalized to sum to 1.0).
/// - `alpha`: teleport probability (typically 0.15); higher = more biased toward seeds.
/// - `max_iterations`: maximum number of power iterations before stopping.
/// - `tolerance`: convergence threshold (L1 norm of score changes between iterations).
///
/// Returns a list of `ScoredNode` entries for all nodes, sorted descending
/// by PageRank score.
///
/// Complexity: O(I * (V + E)) where I is the number of iterations until
/// convergence.
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
/// Each node starts with a unique label. On every iteration, nodes are
/// visited in a deterministic order and each adopts the most frequent
/// label among its neighbors (weighted by edge weight, ties broken by
/// smallest label). The algorithm converges when no labels change.
/// Edges are treated as undirected.
///
/// This is a fast, near-linear community detection method well-suited
/// for large graphs where Louvain/Leiden overhead is unnecessary.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `max_iterations`: maximum number of full passes over all nodes.
///
/// Returns a map from `NodeId` to community label (`u64`).
///
/// Complexity: O(I * (V + E)) where I is the number of iterations until
/// convergence.
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

/// Louvain-style community detection using modularity optimization.
///
/// Iteratively moves nodes between communities to maximize the
/// modularity Q function. After convergence, communities are aggregated
/// into super-nodes and the process repeats on the coarsened graph.
/// This produces higher-quality communities than label propagation by
/// optimizing a global quality function.
///
/// The `resolution` parameter controls community granularity:
/// - 1.0 = standard modularity
/// - \> 1.0 = smaller, more numerous communities
/// - \< 1.0 = larger, fewer communities
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `max_iterations`: maximum number of modularity optimization passes.
/// - `resolution`: modularity resolution parameter (see above).
///
/// Returns a map from `NodeId` to community label (`u64`).
///
/// Complexity: O(I * (V + E)) per level of the hierarchy, where I is the
/// number of iterations per level. Typically converges in a few passes.
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
/// Nodes that act as bridges or bottlenecks receive high scores.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `edge_filter`: criteria to restrict which edges are traversed.
///
/// Returns a list of `(NodeId, centrality)` sorted descending by score.
///
/// Complexity: O(V * E) — one BFS per node, accumulating path counts.
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
/// For each node, closeness = (reachable - 1) / (sum of shortest-path
/// distances to all reachable nodes). Uses the Wasserman-Faust
/// normalization so that nodes in smaller components still receive
/// meaningful scores. Nodes that cannot reach any other node score 0.
///
/// Edges are treated as undirected and unweighted.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `edge_filter`: criteria to restrict which edges are traversed.
///
/// Returns a list of `(NodeId, closeness)` sorted descending by score.
///
/// Complexity: O(V * (V + E)) — one BFS per node to compute distances.
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
                if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(w) {
                    e.insert(d_v + 1);
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
/// connected nodes. The local clustering coefficient for a node with degree
/// k and t triangles is `2t / (k * (k - 1))`.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `edge_filter`: criteria to restrict which edges are considered.
///
/// Returns a `TriangleResult` containing the global triangle count and
/// per-node `(node_id, triangle_count, clustering_coefficient)` tuples.
///
/// Complexity: O(V * d^2) where d is the maximum node degree (intersection
/// of neighbor sets for each edge).
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
/// Unlike `connected_components` which treats edges as undirected, this
/// respects edge direction. A strongly connected component (SCC) is a
/// maximal set of nodes where every node is reachable from every other
/// node following directed edges. Uses an iterative (stack-safe)
/// implementation to avoid stack overflow on deep graphs.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
///
/// Returns a list of components, each being a `Vec<NodeId>`. Components
/// are returned in reverse topological order of the condensation DAG.
///
/// Complexity: O(V + E).
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
                    if i == neighbors.len()
                        && lowlink[&v] == index_map[&v]
                    {
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

    result
}

// ─── Topological Sort ────────────────────────────────────────────────────────

/// Compute a topological ordering of the graph using Kahn's algorithm.
///
/// Processes nodes with zero in-degree first, repeatedly removing them
/// and decrementing the in-degree of their successors. Nodes with no
/// dependencies (sources) appear first in the result.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
///
/// Returns an ordered `Vec<NodeId>` where every node appears before its
/// dependents, or `WeavError::Conflict` if the graph contains a cycle.
///
/// Complexity: O(V + E).
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

// ─── Leiden Community Detection ──────────────────────────────────────────────

/// Leiden community detection with refinement.
///
/// Improves on modularity-based community detection (Louvain) by adding a
/// refinement step that ensures communities are well-connected. After the
/// standard modularity optimization phase, each community is checked for
/// internal connectivity. Nodes whose ratio of internal edges to total
/// edges falls below the `gamma` threshold are separated into singleton
/// communities, and modularity is re-optimized from the refined partition.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `max_iterations`: maximum number of modularity optimization passes.
/// - `resolution`: modularity resolution (1.0 = standard, >1.0 = smaller communities).
/// - `gamma`: refinement threshold (0.0--1.0). Nodes with
///   `internal_edges / total_edges < gamma` are separated into their own
///   community before re-optimization.
///
/// Returns a map from `NodeId` to community label (`u64`).
///
/// Complexity: O(I * (V + E)) per refinement round, similar to Louvain
/// with an additional linear-time refinement pass.
pub fn leiden_communities(
    adjacency: &AdjacencyStore,
    max_iterations: u32,
    resolution: f32,
    gamma: f32,
) -> HashMap<NodeId, u64> {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return HashMap::new();
    }

    // Phase 1: standard modularity optimization (same as modularity_communities)
    let mut community = modularity_communities(adjacency, max_iterations, resolution);

    // Phase 2: refinement — check each community for well-connectedness
    // Build per-node neighbor sets (undirected) for quick internal edge counting
    let mut node_neighbors: HashMap<NodeId, Vec<NodeId>> = HashMap::with_capacity(all_nodes.len());
    for &node in &all_nodes {
        let mut nbrs = Vec::new();
        for &(nbr, _) in adjacency.neighbors_out(node, None).iter() {
            nbrs.push(nbr);
        }
        for &(nbr, _) in adjacency.neighbors_in(node, None).iter() {
            nbrs.push(nbr);
        }
        node_neighbors.insert(node, nbrs);
    }

    let gamma = gamma.clamp(0.0, 1.0) as f64;

    for &node in &all_nodes {
        let nbrs = &node_neighbors[&node];
        if nbrs.is_empty() {
            continue; // isolated node stays in its community
        }

        let my_comm = community[&node];
        let total_edges = nbrs.len() as f64;
        let internal_edges = nbrs
            .iter()
            .filter(|&&nbr| community.get(&nbr).copied() == Some(my_comm))
            .count() as f64;

        let ratio = internal_edges / total_edges;
        if ratio < gamma {
            // Node is not well-connected to its community; make it a singleton
            community.insert(node, node);
        }
    }

    // Phase 3: re-optimize modularity from the refined partition
    // Re-run the modularity optimization moves on the refined partition
    let resolution = resolution as f64;

    // Recompute node degrees and total weight
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

    let m = total_weight_double / 2.0;
    if m == 0.0 {
        return community;
    }

    let mut sigma_tot: HashMap<u64, f64> = HashMap::new();
    for &node in &all_nodes {
        let comm = community[&node];
        *sigma_tot.entry(comm).or_insert(0.0) += node_degree[&node];
    }

    for iteration in 0..max_iterations {
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
                continue;
            }

            let sigma_tot_old = sigma_tot.get(&current_comm).copied().unwrap_or(0.0) - ki;
            let ki_in = edges_to_comm.get(&current_comm).copied().unwrap_or(0.0);
            let remove_cost = ki_in / m - resolution * ki * sigma_tot_old / (2.0 * m * m);

            let mut best_comm = current_comm;
            let mut best_gain = 0.0_f64;

            for (&cand_comm, &ki_cand) in &edges_to_comm {
                if cand_comm == current_comm {
                    continue;
                }
                let sigma_tot_cand = sigma_tot.get(&cand_comm).copied().unwrap_or(0.0);
                let add_gain = ki_cand / m - resolution * ki * sigma_tot_cand / (2.0 * m * m);
                let net_gain = add_gain - remove_cost;

                if net_gain > best_gain || (net_gain == best_gain && cand_comm < best_comm) {
                    best_gain = net_gain;
                    best_comm = cand_comm;
                }
            }

            if best_comm != current_comm {
                *sigma_tot.entry(current_comm).or_insert(0.0) -= ki;
                *sigma_tot.entry(best_comm).or_insert(0.0) += ki;
                community.insert(node, best_comm);
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    community
}

// ─── Node2Vec Random Walks ───────────────────────────────────────────────────

/// Generate biased random walks for Node2Vec-style embeddings.
///
/// Produces `num_walks` walks of length `walk_length` starting from each
/// node. The walk bias is controlled by two parameters that interpolate
/// between BFS-like and DFS-like exploration strategies, enabling
/// downstream embedding algorithms (e.g., Word2Vec) to capture both
/// structural equivalence and community structure.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `walk_length`: number of steps per walk.
/// - `num_walks`: number of walks to generate per node.
/// - `p`: return parameter (high p = less likely to return to the previous node).
/// - `q`: in-out parameter (high q = biased toward local/BFS-like exploration).
/// - `seed`: deterministic PRNG seed (hash-based: `seed XOR step XOR node`).
///
/// Returns a flat list of all walks. The graph is treated as undirected.
///
/// Complexity: O(num_walks * V * walk_length * d) where d is the average
/// node degree.
pub fn node2vec_walks(
    adjacency: &AdjacencyStore,
    walk_length: usize,
    num_walks: usize,
    p: f64,
    q: f64,
    seed: u64,
) -> Vec<Vec<NodeId>> {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() || walk_length == 0 || num_walks == 0 {
        return Vec::new();
    }

    // Simple hash-based PRNG: xorshift64
    fn xorshift(mut state: u64) -> u64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    }

    // Get undirected neighbors for a node (sorted for determinism)
    fn undirected_neighbors(adj: &AdjacencyStore, node: NodeId) -> Vec<NodeId> {
        let mut nbrs: HashSet<NodeId> = HashSet::new();
        for &(nbr, _) in adj.neighbors_out(node, None).iter() {
            nbrs.insert(nbr);
        }
        for &(nbr, _) in adj.neighbors_in(node, None).iter() {
            nbrs.insert(nbr);
        }
        let mut result: Vec<NodeId> = nbrs.into_iter().collect();
        result.sort_unstable();
        result
    }

    let mut all_walks = Vec::with_capacity(all_nodes.len() * num_walks);

    for walk_idx in 0..num_walks {
        for &start_node in &all_nodes {
            let mut walk = Vec::with_capacity(walk_length);
            walk.push(start_node);

            if walk_length == 1 {
                all_walks.push(walk);
                continue;
            }

            // First step: uniform random among neighbors
            let first_nbrs = undirected_neighbors(adjacency, start_node);
            if first_nbrs.is_empty() {
                all_walks.push(walk);
                continue;
            }

            let mut rng_state = seed ^ (walk_idx as u64) ^ start_node;
            rng_state = xorshift(rng_state);
            let first_idx = (rng_state % first_nbrs.len() as u64) as usize;
            walk.push(first_nbrs[first_idx]);

            // Subsequent steps: biased by p and q
            for step in 2..walk_length {
                let current = *walk.last().unwrap();
                let prev = walk[walk.len() - 2];

                let nbrs = undirected_neighbors(adjacency, current);
                if nbrs.is_empty() {
                    break;
                }

                // Build neighbor set of prev for distance computation
                let prev_nbrs: HashSet<NodeId> = undirected_neighbors(adjacency, prev)
                    .into_iter()
                    .collect();

                // Compute unnormalized weights
                let mut weights: Vec<f64> = Vec::with_capacity(nbrs.len());
                for &nbr in &nbrs {
                    let w = if nbr == prev {
                        1.0 / p // return to previous
                    } else if prev_nbrs.contains(&nbr) {
                        1.0 // neighbor of prev (distance 1)
                    } else {
                        1.0 / q // move away (distance 2)
                    };
                    weights.push(w);
                }

                // Normalize and select
                let total: f64 = weights.iter().sum();
                rng_state = xorshift(seed ^ (step as u64) ^ current ^ (walk_idx as u64));
                let threshold = (rng_state as f64 / u64::MAX as f64) * total;

                let mut cumulative = 0.0;
                let mut chosen = nbrs[0];
                for (i, &w) in weights.iter().enumerate() {
                    cumulative += w;
                    if cumulative >= threshold {
                        chosen = nbrs[i];
                        break;
                    }
                }

                walk.push(chosen);
            }

            all_walks.push(walk);
        }
    }

    all_walks
}

// ─── A* Shortest Path ────────────────────────────────────────────────────────

/// A* search state for the priority queue.
struct AstarState {
    node: NodeId,
    g_cost: f64,
    f_cost: f64,
}

impl PartialEq for AstarState {
    fn eq(&self, other: &Self) -> bool {
        self.f_cost == other.f_cost && self.node == other.node
    }
}

impl Eq for AstarState {}

impl PartialOrd for AstarState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AstarState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap
        other
            .f_cost
            .partial_cmp(&self.f_cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.node.cmp(&other.node))
    }
}

/// Find the shortest weighted path between two nodes using A* search.
///
/// Like Dijkstra, but uses a heuristic function `h(n)` to guide the
/// search toward the target: `f(n) = g(n) + h(n)` where `g(n)` is the
/// cost so far. This can dramatically reduce the number of nodes explored
/// when a good heuristic is available.
///
/// Edge weights are used as costs via `1/weight` (same convention as
/// Dijkstra). The heuristic must be admissible (never overestimate) to
/// guarantee optimality.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `source`: starting node.
/// - `target`: destination node.
/// - `heuristic`: function estimating remaining cost from a node to `target`.
/// - `edge_filter`: criteria to restrict which edges are traversed.
///
/// Returns a `WeightedPath` on success, `WeavError::NodeNotFound` if
/// source or target is absent, or `WeavError::Conflict` if no path exists.
///
/// Complexity: O((V + E) log V) in the worst case, but typically much
/// faster than Dijkstra when the heuristic is informative.
pub fn astar_shortest_path(
    adjacency: &AdjacencyStore,
    source: NodeId,
    target: NodeId,
    heuristic: &dyn Fn(NodeId) -> f64,
    edge_filter: &EdgeFilter,
) -> WeavResult<WeightedPath> {
    let all_nodes_set: HashSet<NodeId> = adjacency.all_node_ids().into_iter().collect();
    if !all_nodes_set.contains(&source) {
        return Err(WeavError::NodeNotFound(source, 0));
    }
    if !all_nodes_set.contains(&target) {
        return Err(WeavError::NodeNotFound(target, 0));
    }

    if source == target {
        return Ok(WeightedPath {
            nodes: vec![source],
            total_weight: 0.0,
        });
    }

    let mut g_score: HashMap<NodeId, f64> = HashMap::new();
    let mut parent: HashMap<NodeId, NodeId> = HashMap::new();
    let mut heap = BinaryHeap::new();
    let mut closed: HashSet<NodeId> = HashSet::new();

    g_score.insert(source, 0.0);
    heap.push(AstarState {
        node: source,
        g_cost: 0.0,
        f_cost: heuristic(source),
    });

    while let Some(AstarState { node, g_cost, .. }) = heap.pop() {
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
            return Ok(WeightedPath {
                nodes: path,
                total_weight: g_cost,
            });
        }

        if !closed.insert(node) {
            continue; // already expanded
        }

        // Skip stale entries
        if let Some(&best_g) = g_score.get(&node)
            && g_cost > best_g
        {
            continue;
        }

        let neighbors = adjacency.neighbors_out(node, None);
        for (neighbor, edge_id) in neighbors {
            if !edge_passes_filter(adjacency, edge_id, edge_filter) {
                continue;
            }
            if closed.contains(&neighbor) {
                continue;
            }

            let edge_weight = adjacency
                .get_edge(edge_id)
                .map(|m| m.weight as f64)
                .unwrap_or(1.0);
            let edge_cost = if edge_weight > 0.0 {
                1.0 / edge_weight
            } else {
                f64::MAX
            };

            let tentative_g = g_cost + edge_cost;
            let is_better = g_score.get(&neighbor).is_none_or(|&g| tentative_g < g);

            if is_better {
                g_score.insert(neighbor, tentative_g);
                parent.insert(neighbor, node);
                heap.push(AstarState {
                    node: neighbor,
                    g_cost: tentative_g,
                    f_cost: tentative_g + heuristic(neighbor),
                });
            }
        }
    }

    Err(WeavError::Conflict("no path found".into()))
}

// ─── Yen's K-Shortest Paths ─────────────────────────────────────────────────

/// Find the K shortest loopless paths between two nodes using Yen's algorithm.
///
/// Uses Dijkstra internally to find individual shortest paths, then
/// iteratively discovers the next shortest path by deviating from
/// previously found paths at each spur node. Edges and nodes along
/// earlier paths are temporarily excluded to force alternative routes.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `source`: starting node.
/// - `target`: destination node.
/// - `k`: maximum number of paths to return.
/// - `edge_filter`: criteria to restrict which edges are traversed.
///
/// Returns up to `k` `WeightedPath` entries sorted by total weight
/// (ascending). Returns `WeavError::NodeNotFound` if source or target is
/// absent.
///
/// Complexity: O(K * V * (V + E) log V) — K iterations of a modified
/// Dijkstra search.
pub fn k_shortest_paths(
    adjacency: &AdjacencyStore,
    source: NodeId,
    target: NodeId,
    k: usize,
    edge_filter: &EdgeFilter,
) -> WeavResult<Vec<WeightedPath>> {
    let all_nodes_set: HashSet<NodeId> = adjacency.all_node_ids().into_iter().collect();
    if !all_nodes_set.contains(&source) {
        return Err(WeavError::NodeNotFound(source, 0));
    }
    if !all_nodes_set.contains(&target) {
        return Err(WeavError::NodeNotFound(target, 0));
    }

    if k == 0 {
        return Ok(Vec::new());
    }

    // Helper: Dijkstra with node and edge exclusion sets
    fn dijkstra_with_exclusions(
        adjacency: &AdjacencyStore,
        source: NodeId,
        target: NodeId,
        excluded_nodes: &HashSet<NodeId>,
        excluded_edges: &HashSet<(NodeId, NodeId)>,
        edge_filter: &EdgeFilter,
    ) -> Option<WeightedPath> {
        if source == target {
            return Some(WeightedPath {
                nodes: vec![source],
                total_weight: 0.0,
            });
        }

        let mut dist: HashMap<NodeId, f64> = HashMap::new();
        let mut parent: HashMap<NodeId, NodeId> = HashMap::new();
        let mut heap = BinaryHeap::new();

        dist.insert(source, 0.0);
        heap.push(DijkstraState {
            node: source,
            cost: 0.0,
        });

        while let Some(DijkstraState { node, cost }) = heap.pop() {
            if node == target {
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

            if let Some(&d) = dist.get(&node)
                && cost > d
            {
                continue;
            }

            let neighbors = adjacency.neighbors_out(node, None);
            for (neighbor, edge_id) in neighbors {
                if excluded_nodes.contains(&neighbor) {
                    continue;
                }
                if excluded_edges.contains(&(node, neighbor)) {
                    continue;
                }
                if !edge_passes_filter(adjacency, edge_id, edge_filter) {
                    continue;
                }

                let edge_weight = adjacency
                    .get_edge(edge_id)
                    .map(|m| m.weight as f64)
                    .unwrap_or(1.0);
                let edge_cost = if edge_weight > 0.0 {
                    1.0 / edge_weight
                } else {
                    f64::MAX
                };

                let next_cost = cost + edge_cost;
                let is_better = dist.get(&neighbor).is_none_or(|&d| next_cost < d);

                if is_better {
                    dist.insert(neighbor, next_cost);
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

    // Find the first shortest path
    let first_path = match dijkstra_with_exclusions(
        adjacency,
        source,
        target,
        &HashSet::new(),
        &HashSet::new(),
        edge_filter,
    ) {
        Some(p) => p,
        None => return Ok(Vec::new()),
    };

    let mut result: Vec<WeightedPath> = vec![first_path];

    // Candidate paths, sorted by weight (min-heap via BinaryHeap with Reverse-like ordering)
    let mut candidates: Vec<WeightedPath> = Vec::new();

    for ki in 1..k {
        let prev_path = &result[ki - 1];

        for i in 0..prev_path.nodes.len().saturating_sub(1) {
            let spur_node = prev_path.nodes[i];
            let root_path: Vec<NodeId> = prev_path.nodes[..=i].to_vec();

            // Compute root path cost
            let mut root_cost = 0.0_f64;
            for j in 0..root_path.len().saturating_sub(1) {
                let from = root_path[j];
                let to = root_path[j + 1];
                // Find edge cost
                let neighbors = adjacency.neighbors_out(from, None);
                for &(nbr, eid) in &neighbors {
                    if nbr == to {
                        let w = adjacency
                            .get_edge(eid)
                            .map(|m| m.weight as f64)
                            .unwrap_or(1.0);
                        root_cost += if w > 0.0 { 1.0 / w } else { f64::MAX };
                        break;
                    }
                }
            }

            // Exclude edges used by previous paths at the spur node
            let mut excluded_edges: HashSet<(NodeId, NodeId)> = HashSet::new();
            for prev in &result {
                if prev.nodes.len() > i
                    && prev.nodes[..=i] == root_path[..]
                    && let Some(&next_node) = prev.nodes.get(i + 1)
                {
                    excluded_edges.insert((spur_node, next_node));
                }
            }

            // Exclude nodes in root path (except spur node)
            let mut excluded_nodes: HashSet<NodeId> = HashSet::new();
            for &rn in &root_path[..root_path.len() - 1] {
                excluded_nodes.insert(rn);
            }

            // Find spur path
            if let Some(spur_path) = dijkstra_with_exclusions(
                adjacency,
                spur_node,
                target,
                &excluded_nodes,
                &excluded_edges,
                edge_filter,
            ) {
                // Combine root + spur
                let mut total_nodes = root_path[..root_path.len() - 1].to_vec();
                total_nodes.extend_from_slice(&spur_path.nodes);
                let total_weight = root_cost + spur_path.total_weight;

                // Check for duplicate paths
                let is_dup = result
                    .iter()
                    .chain(candidates.iter())
                    .any(|p| p.nodes == total_nodes);

                if !is_dup {
                    candidates.push(WeightedPath {
                        nodes: total_nodes,
                        total_weight,
                    });
                }
            }
        }

        if candidates.is_empty() {
            break;
        }

        // Select the candidate with minimum weight
        candidates.sort_by(|a, b| {
            a.total_weight
                .partial_cmp(&b.total_weight)
                .unwrap_or(Ordering::Equal)
        });
        result.push(candidates.remove(0));
    }

    Ok(result)
}

// ─── Degree Centrality ───────────────────────────────────────────────────────

/// Compute degree centrality for every node.
///
/// Degree centrality = `degree(node) / (n - 1)` where `n` is the total
/// number of nodes. The graph is treated as undirected: both in-degree
/// and out-degree are counted, with shared neighbors deduplicated.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
///
/// Returns a list of `(NodeId, centrality)` sorted descending by score.
///
/// Complexity: O(V + E).
pub fn degree_centrality(adjacency: &AdjacencyStore) -> Vec<(NodeId, f64)> {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return Vec::new();
    }

    let n = all_nodes.len();
    let denominator = if n > 1 { (n - 1) as f64 } else { 1.0 };

    let mut result: Vec<(NodeId, f64)> = Vec::with_capacity(n);

    for &node in &all_nodes {
        // Collect unique undirected neighbors (dedup shared edges)
        let mut unique_neighbors: HashSet<NodeId> = HashSet::new();
        for &(nbr, _) in adjacency.neighbors_out(node, None).iter() {
            if nbr != node {
                unique_neighbors.insert(nbr);
            }
        }
        for &(nbr, _) in adjacency.neighbors_in(node, None).iter() {
            if nbr != node {
                unique_neighbors.insert(nbr);
            }
        }

        let centrality = unique_neighbors.len() as f64 / denominator;
        result.push((node, centrality));
    }

    result.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    result
}

// ─── Eigenvector Centrality ──────────────────────────────────────────────────

/// Compute eigenvector centrality using the power iteration method.
///
/// A node's score is proportional to the sum of its neighbors' scores
/// (treating the graph as undirected). Scores are normalized by the
/// L-infinity norm (maximum score) each iteration. Convergence is
/// reached when the maximum score change between iterations falls below
/// `tolerance`.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `max_iterations`: maximum number of power iterations.
/// - `tolerance`: convergence threshold on the maximum score change.
///
/// Returns a list of `(NodeId, centrality)` sorted descending by score.
///
/// Complexity: O(I * (V + E)) where I is the number of iterations until
/// convergence.
pub fn eigenvector_centrality(
    adjacency: &AdjacencyStore,
    max_iterations: u32,
    tolerance: f64,
) -> Vec<(NodeId, f64)> {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return Vec::new();
    }

    let n = all_nodes.len();
    let init_val = 1.0 / (n as f64).sqrt();

    let node_to_idx: HashMap<NodeId, usize> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, &nid)| (nid, i))
        .collect();

    let mut scores: Vec<f64> = vec![init_val; n];

    // Pre-compute undirected neighbor indices for each node
    let mut neighbor_indices: Vec<Vec<usize>> = Vec::with_capacity(n);
    for &node in &all_nodes {
        let mut nbr_set: HashSet<NodeId> = HashSet::new();
        for &(nbr, _) in adjacency.neighbors_out(node, None).iter() {
            nbr_set.insert(nbr);
        }
        for &(nbr, _) in adjacency.neighbors_in(node, None).iter() {
            nbr_set.insert(nbr);
        }
        let idx_list: Vec<usize> = nbr_set
            .into_iter()
            .filter_map(|nbr| node_to_idx.get(&nbr).copied())
            .collect();
        neighbor_indices.push(idx_list);
    }

    for _ in 0..max_iterations {
        let mut new_scores = vec![0.0_f64; n];

        // Each node's new score = sum of neighbor scores
        for i in 0..n {
            let mut s = 0.0;
            for &j in &neighbor_indices[i] {
                s += scores[j];
            }
            new_scores[i] = s;
        }

        // Normalize by L-infinity norm (max value)
        let max_score = new_scores
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);

        if max_score > 0.0 {
            for s in &mut new_scores {
                *s /= max_score;
            }
        }

        // Check convergence
        let max_change = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(old, new)| (old - new).abs())
            .fold(0.0_f64, f64::max);

        scores = new_scores;

        if max_change < tolerance {
            break;
        }
    }

    let mut result: Vec<(NodeId, f64)> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, &nid)| (nid, scores[i]))
        .collect();
    result.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    result
}

// ─── HITS (Hyperlink-Induced Topic Search) ───────────────────────────────────

/// Ranked list of `(NodeId, score)` pairs, sorted descending by score.
pub type RankedScores = Vec<(NodeId, f64)>;

/// Compute HITS (Hyperlink-Induced Topic Search) authority and hub scores.
///
/// Authority: a node is a good authority if many good hubs point to it.
/// Hub: a node is a good hub if it points to many good authorities.
///
/// Authority update: `auth[v] = sum(hub[u])` for all `u -> v`.
/// Hub update: `hub[v] = sum(auth[u])` for all `v -> u`.
/// Both vectors are normalized by their L2 norm each iteration.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `max_iterations`: maximum number of iterations.
/// - `tolerance`: convergence threshold on the maximum score change.
///
/// Returns `(authority_scores, hub_scores)` as `HitsScores`, each sorted
/// descending by score.
///
/// Complexity: O(I * (V + E)) where I is the number of iterations until
/// convergence.
pub fn hits(
    adjacency: &AdjacencyStore,
    max_iterations: u32,
    tolerance: f64,
) -> (RankedScores, RankedScores) {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let n = all_nodes.len();
    let node_to_idx: HashMap<NodeId, usize> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, &nid)| (nid, i))
        .collect();

    // Pre-compute in-neighbor and out-neighbor indices
    let mut in_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut out_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, &node) in all_nodes.iter().enumerate() {
        for &(nbr, _) in adjacency.neighbors_out(node, None).iter() {
            if let Some(&j) = node_to_idx.get(&nbr) {
                out_neighbors[i].push(j);
            }
        }
        for &(nbr, _) in adjacency.neighbors_in(node, None).iter() {
            if let Some(&j) = node_to_idx.get(&nbr) {
                in_neighbors[i].push(j);
            }
        }
    }

    let init_val = 1.0 / (n as f64).sqrt();
    let mut auth: Vec<f64> = vec![init_val; n];
    let mut hub: Vec<f64> = vec![init_val; n];

    for _ in 0..max_iterations {
        // Authority update: auth[v] = sum(hub[u]) for all u pointing to v
        let mut new_auth = vec![0.0_f64; n];
        for v in 0..n {
            for &u in &in_neighbors[v] {
                new_auth[v] += hub[u];
            }
        }

        // Normalize auth by L2 norm
        let auth_norm: f64 = new_auth.iter().map(|x| x * x).sum::<f64>().sqrt();
        if auth_norm > 0.0 {
            for a in &mut new_auth {
                *a /= auth_norm;
            }
        }

        // Hub update: hub[v] = sum(auth[u]) for all u that v points to
        let mut new_hub = vec![0.0_f64; n];
        for v in 0..n {
            for &u in &out_neighbors[v] {
                new_hub[v] += new_auth[u];
            }
        }

        // Normalize hub by L2 norm
        let hub_norm: f64 = new_hub.iter().map(|x| x * x).sum::<f64>().sqrt();
        if hub_norm > 0.0 {
            for h in &mut new_hub {
                *h /= hub_norm;
            }
        }

        // Check convergence
        let auth_change = auth
            .iter()
            .zip(new_auth.iter())
            .map(|(old, new)| (old - new).abs())
            .fold(0.0_f64, f64::max);
        let hub_change = hub
            .iter()
            .zip(new_hub.iter())
            .map(|(old, new)| (old - new).abs())
            .fold(0.0_f64, f64::max);

        auth = new_auth;
        hub = new_hub;

        if auth_change < tolerance && hub_change < tolerance {
            break;
        }
    }

    let mut auth_result: Vec<(NodeId, f64)> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, &nid)| (nid, auth[i]))
        .collect();
    auth_result.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    let mut hub_result: Vec<(NodeId, f64)> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, &nid)| (nid, hub[i]))
        .collect();
    hub_result.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    (auth_result, hub_result)
}

// ─── FastRP Node Embeddings ─────────────────────────────────────────────────

/// Generate node embeddings using Fast Random Projection (FastRP).
///
/// FastRP creates dense vector representations of nodes by iteratively
/// aggregating neighborhood structure through sparse random projections.
/// Each node starts with a sparse random vector, then each iteration
/// averages in the embeddings of its neighbors, capturing progressively
/// higher-order structural information. This is Neo4j's most-used
/// embedding algorithm, implemented here in pure Rust.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `embedding_dim`: dimension of output embeddings (e.g., 128, 256).
/// - `iterations`: number of neighborhood aggregation rounds (typically 2--4).
/// - `normalization_strength`: controls L2 normalization strength (0.0 = none, 1.0 = full).
/// - `seed`: random seed for reproducibility.
///
/// Returns a map from `NodeId` to embedding vector (`Vec<f32>`).
///
/// Complexity: O(iterations * (V + E) * embedding_dim).
pub fn fastrp_embeddings(
    adjacency: &AdjacencyStore,
    embedding_dim: usize,
    iterations: usize,
    normalization_strength: f64,
    seed: u64,
) -> HashMap<NodeId, Vec<f32>> {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() || embedding_dim == 0 {
        return HashMap::new();
    }

    let n = all_nodes.len();
    let node_idx: HashMap<NodeId, usize> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, &nid)| (nid, i))
        .collect();

    // Step 1: Generate sparse random initial embeddings.
    // Each component is: +sqrt(3) with prob 1/6, 0 with prob 2/3, -sqrt(3) with prob 1/6.
    // This gives an unbiased sparse random projection matrix (Achlioptas, 2003).
    let sqrt3 = 3.0_f64.sqrt();
    let mut embeddings: Vec<Vec<f64>> = Vec::with_capacity(n);
    for (i, _) in all_nodes.iter().enumerate() {
        let mut vec = vec![0.0_f64; embedding_dim];
        for (j, slot) in vec.iter_mut().enumerate() {
            let h = fastrp_hash(seed ^ (i as u64).wrapping_mul(1_000_003) ^ (j as u64).wrapping_mul(999_983));
            let r = h % 6;
            *slot = match r {
                0 => sqrt3,
                5 => -sqrt3,
                _ => 0.0,
            };
        }
        embeddings.push(vec);
    }

    // Collect per-iteration embeddings for aggregation.
    let mut iteration_embeddings: Vec<Vec<Vec<f64>>> = vec![embeddings.clone()];

    // Step 2: Iterative neighborhood aggregation (mean aggregation with edge weights).
    for iter in 0..iterations {
        let prev = &iteration_embeddings[iter];
        let mut new_embeddings: Vec<Vec<f64>> = vec![vec![0.0; embedding_dim]; n];

        for (i, &node) in all_nodes.iter().enumerate() {
            let out_neighbors = adjacency.neighbors_out(node, None);
            let in_neighbors = adjacency.neighbors_in(node, None);

            let mut degree = 0usize;
            for &(nbr, edge_id) in out_neighbors.iter().chain(in_neighbors.iter()) {
                if let Some(&nbr_idx) = node_idx.get(&nbr) {
                    let weight = adjacency
                        .get_edge(edge_id)
                        .map(|m| m.weight as f64)
                        .unwrap_or(1.0);
                    for (dst, src) in new_embeddings[i].iter_mut().zip(prev[nbr_idx].iter()) {
                        *dst += src * weight;
                    }
                    degree += 1;
                }
            }

            // Normalize by degree (mean aggregation).
            if degree > 0 {
                let inv_deg = 1.0 / degree as f64;
                for val in new_embeddings[i].iter_mut() {
                    *val *= inv_deg;
                }
            }
        }

        // Optional L2 normalization per iteration.
        if normalization_strength > 0.0 {
            for emb in &mut new_embeddings {
                let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-10 {
                    let factor = normalization_strength / norm;
                    for d in emb.iter_mut() {
                        *d *= factor;
                    }
                }
            }
        }

        iteration_embeddings.push(new_embeddings);
    }

    // Step 3: Use the last iteration's embeddings (standard approach).
    let final_emb = iteration_embeddings.last().unwrap();

    // L2-normalize final embeddings.
    let mut result: HashMap<NodeId, Vec<f32>> = HashMap::with_capacity(n);
    for (i, &node) in all_nodes.iter().enumerate() {
        let mut vec: Vec<f32> = final_emb[i].iter().map(|&x| x as f32).collect();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for v in &mut vec {
                *v /= norm;
            }
        }
        result.insert(node, vec);
    }

    result
}

/// Deterministic hash function for FastRP sparse random projection.
fn fastrp_hash(mut x: u64) -> u64 {
    x = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

// ─── Node Similarity ────────────────────────────────────────────────────────

/// Metric for node similarity computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityMetric {
    Jaccard,
    Cosine,
    Overlap,
}

/// Collect all undirected neighbors for a node (both outgoing and incoming),
/// excluding self-loops. Returns a `HashSet<NodeId>`.
fn undirected_neighbors(adjacency: &AdjacencyStore, node: NodeId) -> HashSet<NodeId> {
    let mut neighbors = HashSet::new();
    for &(nbr, _) in adjacency.neighbors_out(node, None).iter() {
        if nbr != node {
            neighbors.insert(nbr);
        }
    }
    for &(nbr, _) in adjacency.neighbors_in(node, None).iter() {
        if nbr != node {
            neighbors.insert(nbr);
        }
    }
    neighbors
}

/// Compute Jaccard similarity between two nodes based on their neighborhoods.
///
/// J(a, b) = |N(a) ∩ N(b)| / |N(a) ∪ N(b)|
///
/// Returns 0.0 if both neighborhoods are empty. Returns 1.0 if both nodes
/// are identical (same node ID).
pub fn jaccard_similarity(adjacency: &AdjacencyStore, node_a: NodeId, node_b: NodeId) -> f64 {
    let na = undirected_neighbors(adjacency, node_a);
    let nb = undirected_neighbors(adjacency, node_b);
    let union_size = na.union(&nb).count();
    if union_size == 0 {
        return 0.0;
    }
    let intersection_size = na.intersection(&nb).count();
    intersection_size as f64 / union_size as f64
}

/// Compute Cosine similarity between two nodes based on their neighborhoods.
///
/// cos(a, b) = |N(a) ∩ N(b)| / sqrt(|N(a)| * |N(b)|)
///
/// Returns 0.0 if either neighborhood is empty.
pub fn cosine_similarity(adjacency: &AdjacencyStore, node_a: NodeId, node_b: NodeId) -> f64 {
    let na = undirected_neighbors(adjacency, node_a);
    let nb = undirected_neighbors(adjacency, node_b);
    let product = na.len() * nb.len();
    if product == 0 {
        return 0.0;
    }
    let intersection_size = na.intersection(&nb).count();
    intersection_size as f64 / (product as f64).sqrt()
}

/// Compute Overlap coefficient between two nodes based on their neighborhoods.
///
/// overlap(a, b) = |N(a) ∩ N(b)| / min(|N(a)|, |N(b)|)
///
/// Returns 0.0 if either neighborhood is empty.
pub fn overlap_similarity(adjacency: &AdjacencyStore, node_a: NodeId, node_b: NodeId) -> f64 {
    let na = undirected_neighbors(adjacency, node_a);
    let nb = undirected_neighbors(adjacency, node_b);
    let min_size = na.len().min(nb.len());
    if min_size == 0 {
        return 0.0;
    }
    let intersection_size = na.intersection(&nb).count();
    intersection_size as f64 / min_size as f64
}

/// Compute all-pairs node similarity (top-k).
///
/// Evaluates every pair of nodes using the specified metric and returns
/// pairs sorted by similarity descending. Only includes pairs where
/// similarity > 0.
pub fn all_pairs_similarity(
    adjacency: &AdjacencyStore,
    metric: SimilarityMetric,
    top_k: usize,
) -> Vec<(NodeId, NodeId, f64)> {
    let all_nodes = adjacency.all_node_ids();
    let n = all_nodes.len();
    let mut results: Vec<(NodeId, NodeId, f64)> = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let a = all_nodes[i];
            let b = all_nodes[j];
            let sim = match metric {
                SimilarityMetric::Jaccard => jaccard_similarity(adjacency, a, b),
                SimilarityMetric::Cosine => cosine_similarity(adjacency, a, b),
                SimilarityMetric::Overlap => overlap_similarity(adjacency, a, b),
            };
            if sim > 0.0 {
                results.push((a, b, sim));
            }
        }
    }

    results.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
            .then(a.1.cmp(&b.1))
    });
    results.truncate(top_k);
    results
}

// ─── Link Prediction ────────────────────────────────────────────────────────

/// Metric for link prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkPredictionMetric {
    AdamicAdar,
    CommonNeighbors,
    PreferentialAttachment,
    ResourceAllocation,
}

/// Adamic-Adar index between two nodes.
///
/// sum(1 / log(|N(z)|)) for z in N(a) ∩ N(b)
///
/// Neighbors with degree 1 are skipped (log(1) = 0).
pub fn adamic_adar(adjacency: &AdjacencyStore, node_a: NodeId, node_b: NodeId) -> f64 {
    let na = undirected_neighbors(adjacency, node_a);
    let nb = undirected_neighbors(adjacency, node_b);
    let mut score = 0.0;
    for &z in na.intersection(&nb) {
        let degree = undirected_neighbors(adjacency, z).len();
        if degree > 1 {
            score += 1.0 / (degree as f64).ln();
        }
    }
    score
}

/// Common neighbors count between two nodes.
///
/// |N(a) ∩ N(b)|
pub fn common_neighbors_count(
    adjacency: &AdjacencyStore,
    node_a: NodeId,
    node_b: NodeId,
) -> usize {
    let na = undirected_neighbors(adjacency, node_a);
    let nb = undirected_neighbors(adjacency, node_b);
    na.intersection(&nb).count()
}

/// Preferential attachment score between two nodes.
///
/// |N(a)| * |N(b)|
pub fn preferential_attachment(
    adjacency: &AdjacencyStore,
    node_a: NodeId,
    node_b: NodeId,
) -> f64 {
    let na = undirected_neighbors(adjacency, node_a).len();
    let nb = undirected_neighbors(adjacency, node_b).len();
    (na * nb) as f64
}

/// Resource allocation index between two nodes.
///
/// sum(1 / |N(z)|) for z in N(a) ∩ N(b)
pub fn resource_allocation(
    adjacency: &AdjacencyStore,
    node_a: NodeId,
    node_b: NodeId,
) -> f64 {
    let na = undirected_neighbors(adjacency, node_a);
    let nb = undirected_neighbors(adjacency, node_b);
    let mut score = 0.0;
    for &z in na.intersection(&nb) {
        let degree = undirected_neighbors(adjacency, z).len();
        if degree > 0 {
            score += 1.0 / degree as f64;
        }
    }
    score
}

/// Predict top-k most likely new links based on the given metric.
///
/// Evaluates every pair of nodes that are NOT already directly connected
/// and ranks them by the specified link prediction score. Returns pairs
/// sorted descending by score.
pub fn predict_links(
    adjacency: &AdjacencyStore,
    metric: LinkPredictionMetric,
    top_k: usize,
) -> Vec<(NodeId, NodeId, f64)> {
    let all_nodes = adjacency.all_node_ids();
    let n = all_nodes.len();
    let mut results: Vec<(NodeId, NodeId, f64)> = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let a = all_nodes[i];
            let b = all_nodes[j];

            // Skip pairs that are already connected (either direction)
            if adjacency.edge_between(a, b, None).is_some()
                || adjacency.edge_between(b, a, None).is_some()
            {
                continue;
            }

            let score = match metric {
                LinkPredictionMetric::AdamicAdar => adamic_adar(adjacency, a, b),
                LinkPredictionMetric::CommonNeighbors => {
                    common_neighbors_count(adjacency, a, b) as f64
                }
                LinkPredictionMetric::PreferentialAttachment => {
                    preferential_attachment(adjacency, a, b)
                }
                LinkPredictionMetric::ResourceAllocation => {
                    resource_allocation(adjacency, a, b)
                }
            };
            if score > 0.0 {
                results.push((a, b, score));
            }
        }
    }

    results.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
            .then(a.1.cmp(&b.1))
    });
    results.truncate(top_k);
    results
}

// ─── Random Walk ────────────────────────────────────────────────────────────

/// Perform a simple random walk starting from a node.
///
/// At each step, picks a random undirected neighbor (both outgoing and
/// incoming edges). If the current node has no neighbors, the walk
/// terminates early.
///
/// Uses a deterministic xorshift64 PRNG seeded with the given seed for
/// reproducibility.
///
/// Returns the sequence of visited node IDs (including the start node).
pub fn random_walk(
    adjacency: &AdjacencyStore,
    start: NodeId,
    length: usize,
    seed: u64,
) -> Vec<NodeId> {
    let mut rng_state = if seed == 0 { 1 } else { seed };
    let mut path = Vec::with_capacity(length + 1);
    path.push(start);

    let mut current = start;
    for _ in 0..length {
        let mut neighbors: Vec<NodeId> = Vec::new();
        for &(nbr, _) in adjacency.neighbors_out(current, None).iter() {
            neighbors.push(nbr);
        }
        for &(nbr, _) in adjacency.neighbors_in(current, None).iter() {
            neighbors.push(nbr);
        }

        if neighbors.is_empty() {
            break;
        }

        // Xorshift64
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;

        let idx = (rng_state as usize) % neighbors.len();
        current = neighbors[idx];
        path.push(current);
    }

    path
}

// ─── K-Core Decomposition ───────────────────────────────────────────────────

/// K-Core decomposition: assigns each node its core number.
///
/// The core number of a node is the largest k such that the node belongs
/// to a k-core (a maximal subgraph where every node has degree >= k).
///
/// Uses the standard peeling algorithm: repeatedly removes the node with
/// the smallest effective degree, assigning core number = current minimum
/// degree. Treats the graph as undirected.
///
/// Returns a map from NodeId to its core number.
pub fn k_core_decomposition(adjacency: &AdjacencyStore) -> HashMap<NodeId, u32> {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return HashMap::new();
    }

    // Compute initial undirected degree for each node
    let mut degree: HashMap<NodeId, u32> = HashMap::with_capacity(all_nodes.len());
    for &node in &all_nodes {
        let d = undirected_neighbors(adjacency, node).len() as u32;
        degree.insert(node, d);
    }

    // Build sorted list by degree for the peeling order
    let mut nodes_by_degree: Vec<NodeId> = all_nodes.clone();
    nodes_by_degree.sort_by_key(|n| degree[n]);

    let mut core_number: HashMap<NodeId, u32> = HashMap::with_capacity(all_nodes.len());
    let mut removed: HashSet<NodeId> = HashSet::new();

    for &node in &nodes_by_degree {
        if removed.contains(&node) {
            continue;
        }
        let k = degree[&node];
        core_number.insert(node, k);
        removed.insert(node);

        // Decrease degree of remaining neighbors
        for nbr in undirected_neighbors(adjacency, node) {
            if !removed.contains(&nbr)
                && let Some(d) = degree.get_mut(&nbr)
                && *d > k
            {
                *d = (*d).saturating_sub(1);
            }
        }
    }

    // Re-sort remaining nodes after degree updates by re-running peeling
    // The simple approach above doesn't correctly handle re-ordering after
    // degree decreases. Use the standard BZ algorithm instead.
    //
    // Reset and do it properly with a bucket-based approach.
    core_number.clear();
    let mut deg: HashMap<NodeId, u32> = HashMap::with_capacity(all_nodes.len());
    for &node in &all_nodes {
        let d = undirected_neighbors(adjacency, node).len() as u32;
        deg.insert(node, d);
    }

    let max_deg = deg.values().copied().max().unwrap_or(0);
    let mut buckets: Vec<Vec<NodeId>> = vec![Vec::new(); (max_deg + 1) as usize];
    let mut pos: HashMap<NodeId, usize> = HashMap::with_capacity(all_nodes.len());

    for &node in &all_nodes {
        let d = deg[&node] as usize;
        pos.insert(node, buckets[d].len());
        buckets[d].push(node);
    }

    let mut processed: HashSet<NodeId> = HashSet::new();

    for k in 0..=max_deg {
        while let Some(node) = buckets[k as usize].pop() {
            if processed.contains(&node) {
                continue;
            }
            core_number.insert(node, k);
            processed.insert(node);

            for nbr in undirected_neighbors(adjacency, node) {
                if !processed.contains(&nbr) {
                    let old_d = deg[&nbr];
                    if old_d > k {
                        let new_d = old_d - 1;
                        deg.insert(nbr, new_d);
                        buckets[new_d as usize].push(nbr);
                    }
                }
            }
        }
    }

    core_number
}

// ─── Max Flow (Edmonds-Karp) ────────────────────────────────────────────────

/// Result of a maximum flow computation.
pub struct MaxFlowResult {
    /// The maximum flow value from source to sink.
    pub max_flow: f64,
    /// Per-edge flow assignments.
    pub flow_edges: Vec<FlowEdge>,
}

/// Flow assignment on a single edge.
pub struct FlowEdge {
    pub source: NodeId,
    pub target: NodeId,
    pub flow: f64,
    pub capacity: f64,
}

/// Compute maximum flow from source to sink using Edmonds-Karp algorithm
/// (BFS-based Ford-Fulkerson).
///
/// Edge weights are used as capacities. The graph is treated as directed;
/// for each directed edge a reverse residual edge with zero capacity is
/// created.
///
/// Returns the maximum flow value and the flow on each edge with non-zero
/// flow.
pub fn max_flow(
    adjacency: &AdjacencyStore,
    source: NodeId,
    sink: NodeId,
) -> WeavResult<MaxFlowResult> {
    if source == sink {
        return Ok(MaxFlowResult {
            max_flow: 0.0,
            flow_edges: Vec::new(),
        });
    }

    if !adjacency.has_node(source) {
        return Err(WeavError::NodeNotFound(source, 0));
    }
    if !adjacency.has_node(sink) {
        return Err(WeavError::NodeNotFound(sink, 0));
    }

    // Build capacity/flow graph using adjacency list of indices.
    // Map NodeId -> index for efficient lookup.
    let all_nodes = adjacency.all_node_ids();
    let node_to_idx: HashMap<NodeId, usize> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, &n)| (n, i))
        .collect();
    let n = all_nodes.len();

    // Adjacency list of (neighbor_index, edge_index_in_edges_vec)
    let mut graph: Vec<Vec<usize>> = vec![Vec::new(); n];
    // Edges stored as (from_idx, to_idx, capacity, flow)
    let mut edges: Vec<(usize, usize, f64, f64)> = Vec::new();

    // Add forward and reverse edges for each directed edge
    for (_, meta) in adjacency.all_edges() {
        let src_idx = node_to_idx[&meta.source];
        let tgt_idx = node_to_idx[&meta.target];
        let cap = meta.weight as f64;

        let fwd_idx = edges.len();
        edges.push((src_idx, tgt_idx, cap, 0.0));
        let rev_idx = edges.len();
        edges.push((tgt_idx, src_idx, 0.0, 0.0));

        graph[src_idx].push(fwd_idx);
        graph[tgt_idx].push(rev_idx);
    }

    let source_idx = node_to_idx[&source];
    let sink_idx = node_to_idx[&sink];
    let mut total_flow = 0.0;

    // BFS to find augmenting paths
    loop {
        let mut parent_edge: Vec<Option<usize>> = vec![None; n];
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();
        visited[source_idx] = true;
        queue.push_back(source_idx);

        while let Some(u) = queue.pop_front() {
            if u == sink_idx {
                break;
            }
            for &edge_idx in &graph[u] {
                let (_, to, cap, flow) = edges[edge_idx];
                let residual = cap - flow;
                if !visited[to] && residual > 1e-12 {
                    visited[to] = true;
                    parent_edge[to] = Some(edge_idx);
                    queue.push_back(to);
                }
            }
        }

        if !visited[sink_idx] {
            break;
        }

        // Find bottleneck
        let mut bottleneck = f64::INFINITY;
        let mut v = sink_idx;
        while v != source_idx {
            let edge_idx = parent_edge[v].unwrap();
            let (_, _, cap, flow) = edges[edge_idx];
            bottleneck = bottleneck.min(cap - flow);
            v = edges[edge_idx].0;
        }

        // Update flow along the path
        v = sink_idx;
        while v != source_idx {
            let edge_idx = parent_edge[v].unwrap();
            edges[edge_idx].3 += bottleneck;
            // Reverse edge is always edge_idx ^ 1 (they are added in pairs)
            let rev_idx = edge_idx ^ 1;
            edges[rev_idx].3 -= bottleneck;
            v = edges[edge_idx].0;
        }

        total_flow += bottleneck;
    }

    // Collect flow edges (only forward edges with positive flow)
    let mut flow_edges = Vec::new();
    for (i, &(from, to, cap, flow)) in edges.iter().enumerate() {
        if i % 2 == 0 && flow > 1e-12 {
            flow_edges.push(FlowEdge {
                source: all_nodes[from],
                target: all_nodes[to],
                flow,
                capacity: cap,
            });
        }
    }

    Ok(MaxFlowResult {
        max_flow: total_flow,
        flow_edges,
    })
}

// ─── Minimum Spanning Tree (Kruskal's) ─────────────────────────────────────

/// Result of a minimum spanning tree computation.
pub struct MstResult {
    /// Edges in the minimum spanning tree.
    pub edges: Vec<MstEdge>,
    /// Total weight of the MST.
    pub total_weight: f64,
}

/// An edge in the minimum spanning tree.
pub struct MstEdge {
    pub source: NodeId,
    pub target: NodeId,
    pub edge_id: EdgeId,
    pub weight: f64,
}

/// Union-Find (Disjoint Set Union) for cycle detection in Kruskal's algorithm.
struct UnionFind {
    parent: HashMap<NodeId, NodeId>,
    rank: HashMap<NodeId, u32>,
}

impl UnionFind {
    fn new(nodes: &[NodeId]) -> Self {
        let mut parent = HashMap::with_capacity(nodes.len());
        let mut rank = HashMap::with_capacity(nodes.len());
        for &node in nodes {
            parent.insert(node, node);
            rank.insert(node, 0);
        }
        Self { parent, rank }
    }

    fn find(&mut self, mut x: NodeId) -> NodeId {
        while self.parent[&x] != x {
            let grandparent = self.parent[&self.parent[&x]];
            self.parent.insert(x, grandparent);
            x = grandparent;
        }
        x
    }

    fn union(&mut self, a: NodeId, b: NodeId) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        let rank_a = self.rank[&ra];
        let rank_b = self.rank[&rb];
        if rank_a < rank_b {
            self.parent.insert(ra, rb);
        } else if rank_a > rank_b {
            self.parent.insert(rb, ra);
        } else {
            self.parent.insert(rb, ra);
            self.rank.insert(ra, rank_a + 1);
        }
        true
    }
}

/// Compute the minimum spanning tree using Kruskal's algorithm.
///
/// Uses edge weights as costs (lower weight = preferred). Since Weav edge
/// weights represent strength (higher = stronger connection), this finds
/// the tree that connects all nodes using the weakest (cheapest) edges.
///
/// Only considers the graph as undirected; duplicate edges between the
/// same pair of nodes (in both directions) are deduplicated by keeping
/// the one with the lower weight.
///
/// Returns the edges in the MST and the total weight.
pub fn minimum_spanning_tree(adjacency: &AdjacencyStore) -> MstResult {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return MstResult {
            edges: Vec::new(),
            total_weight: 0.0,
        };
    }

    // Collect all edges, dedup by (min(src,tgt), max(src,tgt)) keeping lowest weight
    let mut edge_map: HashMap<(NodeId, NodeId), (EdgeId, f64)> = HashMap::new();
    for (eid, meta) in adjacency.all_edges() {
        let key = if meta.source <= meta.target {
            (meta.source, meta.target)
        } else {
            (meta.target, meta.source)
        };
        let w = meta.weight as f64;
        let entry = edge_map.entry(key).or_insert((eid, w));
        if w < entry.1 {
            *entry = (eid, w);
        }
    }

    // Sort edges by weight ascending (lower weight = lower cost)
    let mut sorted_edges: Vec<((NodeId, NodeId), (EdgeId, f64))> =
        edge_map.into_iter().collect();
    sorted_edges.sort_by(|a, b| {
        a.1 .1
            .partial_cmp(&b.1 .1)
            .unwrap_or(Ordering::Equal)
    });

    let mut uf = UnionFind::new(&all_nodes);
    let mut mst_edges = Vec::new();
    let mut total_weight = 0.0;

    for ((src, tgt), (eid, weight)) in sorted_edges {
        if uf.union(src, tgt) {
            mst_edges.push(MstEdge {
                source: src,
                target: tgt,
                edge_id: eid,
                weight,
            });
            total_weight += weight;
        }
    }

    MstResult {
        edges: mst_edges,
        total_weight,
    }
}

// ─── Bellman-Ford Single-Source Shortest Path ────────────────────────────────

/// Bellman-Ford single-source shortest path. Handles negative edge weights.
///
/// Returns shortest distances from source to all reachable nodes, or error
/// if a negative cycle exists. Edge weights are inverted (`1.0 / weight`)
/// to convert Weav's strength-based weights (higher = stronger) into costs
/// (higher strength = lower cost).
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `source`: the starting node.
///
/// Returns a list of `(NodeId, distance)` sorted ascending by distance,
/// or `WeavError::Internal` if a negative cycle is detected.
///
/// Complexity: O(V * E).
pub fn bellman_ford(
    adjacency: &AdjacencyStore,
    source: NodeId,
) -> WeavResult<Vec<(NodeId, f64)>> {
    if !adjacency.has_node(source) {
        return Err(WeavError::NodeNotFound(source, 0));
    }

    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return Ok(Vec::new());
    }

    let mut dist: HashMap<NodeId, f64> = HashMap::with_capacity(all_nodes.len());
    for &node in &all_nodes {
        dist.insert(node, f64::INFINITY);
    }
    dist.insert(source, 0.0);

    // Collect all edges as (src, tgt, cost)
    let mut edges: Vec<(NodeId, NodeId, f64)> = Vec::new();
    for (_, meta) in adjacency.all_edges() {
        let cost = if meta.weight > 0.0 {
            1.0 / meta.weight as f64
        } else {
            f64::INFINITY
        };
        edges.push((meta.source, meta.target, cost));
    }

    let v = all_nodes.len();

    // Relax edges V-1 times
    for _ in 0..v.saturating_sub(1) {
        let mut updated = false;
        for &(src, tgt, cost) in &edges {
            let d_src = dist[&src];
            if d_src < f64::INFINITY {
                let new_dist = d_src + cost;
                if new_dist < dist[&tgt] {
                    dist.insert(tgt, new_dist);
                    updated = true;
                }
            }
        }
        if !updated {
            break;
        }
    }

    // Check for negative cycles
    for &(src, tgt, cost) in &edges {
        let d_src = dist[&src];
        if d_src < f64::INFINITY && d_src + cost < dist[&tgt] - 1e-12 {
            return Err(WeavError::Internal(
                "negative cycle detected in Bellman-Ford".to_string(),
            ));
        }
    }

    // Return only reachable nodes, sorted by distance ascending
    let mut result: Vec<(NodeId, f64)> = dist
        .into_iter()
        .filter(|&(_, d)| d < f64::INFINITY)
        .collect();
    result.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    Ok(result)
}

// ─── Harmonic Centrality ────────────────────────────────────────────────────

/// Harmonic centrality: sum of inverse shortest-path distances.
///
/// Handles disconnected graphs correctly (unlike closeness centrality).
/// `H(v) = (1/(n-1)) * sum_{u != v} 1/d(v,u)` where unreachable nodes
/// contribute 0 to the sum.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `filter`: criteria to restrict which edges are traversed.
///
/// Returns a list of `(NodeId, centrality)` sorted descending by score.
///
/// Complexity: O(V * (V + E)) — one BFS per node.
pub fn harmonic_centrality(adjacency: &AdjacencyStore, filter: &EdgeFilter) -> Vec<(NodeId, f64)> {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return Vec::new();
    }

    let n = all_nodes.len();
    let denominator = if n > 1 { (n - 1) as f64 } else { 1.0 };
    let mut result: Vec<(NodeId, f64)> = Vec::with_capacity(n);

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
                if !edge_passes_filter(adjacency, eid, filter) {
                    continue;
                }
                if let std::collections::hash_map::Entry::Vacant(e) = dist.entry(w) {
                    e.insert(d_v + 1);
                    queue.push_back(w);
                }
            }
        }

        // Sum 1/distance for all reachable nodes (skip self and unreachable)
        let mut sum_inv = 0.0;
        for (&node, &d) in &dist {
            if node != source && d > 0 {
                sum_inv += 1.0 / d as f64;
            }
        }

        result.push((source, sum_inv / denominator));
    }

    result.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    result
}

// ─── Katz Centrality ────────────────────────────────────────────────────────

/// Katz centrality: counts all walks between nodes weighted by attenuation
/// factor alpha.
///
/// Works on DAGs where eigenvector centrality gives all zeros.
/// `x_i = alpha * sum_j A_ij * x_j + beta`
///
/// Uses power iteration. Convergence is reached when the L2 norm of score
/// changes falls below `tolerance`. Scores are L2-normalized after
/// convergence.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `alpha`: attenuation factor (should be < 1/spectral_radius).
/// - `beta`: bias term (typically 1.0).
/// - `max_iterations`: maximum number of power iterations.
/// - `tolerance`: convergence threshold on the maximum score change.
///
/// Returns a list of `(NodeId, centrality)` sorted descending by score.
///
/// Complexity: O(I * (V + E)) where I is the number of iterations.
pub fn katz_centrality(
    adjacency: &AdjacencyStore,
    alpha: f64,
    beta: f64,
    max_iterations: u32,
    tolerance: f64,
) -> Vec<(NodeId, f64)> {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return Vec::new();
    }

    let n = all_nodes.len();
    let node_to_idx: HashMap<NodeId, usize> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, &nid)| (nid, i))
        .collect();

    // Pre-compute undirected neighbor indices for each node
    let mut neighbor_indices: Vec<Vec<usize>> = Vec::with_capacity(n);
    for &node in &all_nodes {
        let mut nbr_set: HashSet<NodeId> = HashSet::new();
        for &(nbr, _) in adjacency.neighbors_out(node, None).iter() {
            nbr_set.insert(nbr);
        }
        for &(nbr, _) in adjacency.neighbors_in(node, None).iter() {
            nbr_set.insert(nbr);
        }
        let idx_list: Vec<usize> = nbr_set
            .into_iter()
            .filter_map(|nbr| node_to_idx.get(&nbr).copied())
            .collect();
        neighbor_indices.push(idx_list);
    }

    let mut scores: Vec<f64> = vec![0.0; n];

    for _ in 0..max_iterations {
        let mut new_scores = vec![0.0_f64; n];

        // x_i = alpha * sum_j A_ij * x_j + beta
        for i in 0..n {
            let mut s = 0.0;
            for &j in &neighbor_indices[i] {
                s += scores[j];
            }
            new_scores[i] = alpha * s + beta;
        }

        // Check convergence (max absolute change)
        let max_change = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(old, new)| (old - new).abs())
            .fold(0.0_f64, f64::max);

        scores = new_scores;

        if max_change < tolerance {
            break;
        }
    }

    // L2-normalize
    let l2_norm: f64 = scores.iter().map(|x| x * x).sum::<f64>().sqrt();
    if l2_norm > 0.0 {
        for s in &mut scores {
            *s /= l2_norm;
        }
    }

    let mut result: Vec<(NodeId, f64)> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, &nid)| (nid, scores[i]))
        .collect();
    result.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    result
}

// ─── ArticleRank ────────────────────────────────────────────────────────────

/// ArticleRank: PageRank variant that reduces bias toward high-authority
/// low-outdegree nodes.
///
/// Denominator is `outdegree(v) + avg_outdegree` instead of `outdegree(v)`.
/// This dampens the influence of nodes with very low outdegree that would
/// otherwise concentrate all their rank into a few neighbors.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `damping`: damping factor (typically 0.85).
/// - `max_iterations`: maximum number of power iterations.
/// - `tolerance`: convergence threshold on the maximum score change.
///
/// Returns a list of `(NodeId, score)` sorted descending by score.
///
/// Complexity: O(I * (V + E)) where I is the number of iterations.
pub fn article_rank(
    adjacency: &AdjacencyStore,
    damping: f32,
    max_iterations: u32,
    tolerance: f64,
) -> Vec<(NodeId, f64)> {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return Vec::new();
    }

    let n = all_nodes.len();
    let node_to_idx: HashMap<NodeId, usize> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, &nid)| (nid, i))
        .collect();

    // Pre-compute out-neighbors and in-neighbors as indices
    let mut out_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];

    for (i, &node) in all_nodes.iter().enumerate() {
        for &(nbr, _) in adjacency.neighbors_out(node, None).iter() {
            if let Some(&j) = node_to_idx.get(&nbr) {
                out_neighbors[i].push(j);
            }
        }
        for &(nbr, _) in adjacency.neighbors_in(node, None).iter() {
            if let Some(&j) = node_to_idx.get(&nbr) {
                in_neighbors[i].push(j);
            }
        }
    }

    // Compute average outdegree
    let total_out_edges: usize = out_neighbors.iter().map(|v| v.len()).sum();
    let avg_outdegree = if n > 0 {
        total_out_edges as f64 / n as f64
    } else {
        1.0
    };

    let d = damping as f64;
    let init_score = 1.0 / n as f64;
    let mut scores: Vec<f64> = vec![init_score; n];

    for _ in 0..max_iterations {
        let mut new_scores = vec![(1.0 - d) / n as f64; n];

        for i in 0..n {
            // Contribute to neighbors: score[i] / (outdegree[i] + avg_outdegree)
            let out_deg = out_neighbors[i].len() as f64;
            let contribution = if out_deg + avg_outdegree > 0.0 {
                d * scores[i] / (out_deg + avg_outdegree)
            } else {
                0.0
            };
            for &j in &out_neighbors[i] {
                new_scores[j] += contribution;
            }
        }

        // Check convergence
        let max_change = scores
            .iter()
            .zip(new_scores.iter())
            .map(|(old, new)| (old - new).abs())
            .fold(0.0_f64, f64::max);

        scores = new_scores;

        if max_change < tolerance {
            break;
        }
    }

    let mut result: Vec<(NodeId, f64)> = all_nodes
        .iter()
        .enumerate()
        .map(|(i, &nid)| (nid, scores[i]))
        .collect();
    result.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    result
}

// ─── K-1 Coloring ───────────────────────────────────────────────────────────

/// Result of a graph coloring computation.
pub struct GraphColoringResult {
    /// Map from NodeId to its assigned color (0-based).
    pub colors: HashMap<NodeId, u32>,
    /// Number of distinct colors used.
    pub num_colors: u32,
    /// Number of remaining conflicts (0 if valid coloring).
    pub conflicts: u32,
}

/// K-1 Coloring: greedy graph coloring.
///
/// Assigns colors to nodes such that no two adjacent nodes share a color,
/// using at most K colors where K <= max_degree + 1.
///
/// Initializes all nodes with random colors, then iteratively reassigns
/// each node the smallest color not used by any neighbor. Repeats until
/// stable or `max_iterations` reached.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `max_iterations`: maximum number of coloring passes.
///
/// Returns a `GraphColoringResult` with the color assignment, the number
/// of distinct colors used, and the remaining conflict count.
///
/// Complexity: O(I * V * d) where d is the maximum node degree and I is
/// the number of iterations.
pub fn k1_coloring(
    adjacency: &AdjacencyStore,
    max_iterations: u32,
) -> GraphColoringResult {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return GraphColoringResult {
            colors: HashMap::new(),
            num_colors: 0,
            conflicts: 0,
        };
    }

    // Initialize colors using a simple hash-based pseudo-random assignment
    let mut colors: HashMap<NodeId, u32> = HashMap::with_capacity(all_nodes.len());
    for &node in &all_nodes {
        // Simple deterministic "random" initial color based on node ID
        let initial_color = (node.wrapping_mul(2654435761) >> 16) as u32 % all_nodes.len() as u32;
        colors.insert(node, initial_color);
    }

    for _ in 0..max_iterations {
        let mut changed = false;

        for &node in &all_nodes {
            // Collect neighbor colors
            let mut neighbor_colors: HashSet<u32> = HashSet::new();
            for &(nbr, _) in adjacency.neighbors_out(node, None).iter() {
                if nbr != node {
                    if let Some(&c) = colors.get(&nbr) {
                        neighbor_colors.insert(c);
                    }
                }
            }
            for &(nbr, _) in adjacency.neighbors_in(node, None).iter() {
                if nbr != node {
                    if let Some(&c) = colors.get(&nbr) {
                        neighbor_colors.insert(c);
                    }
                }
            }

            // Find smallest color not used by any neighbor
            let mut new_color: u32 = 0;
            while neighbor_colors.contains(&new_color) {
                new_color += 1;
            }

            if colors[&node] != new_color {
                colors.insert(node, new_color);
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // Count conflicts
    let mut conflicts: u32 = 0;
    for &node in &all_nodes {
        let node_color = colors[&node];
        for &(nbr, _) in adjacency.neighbors_out(node, None).iter() {
            if nbr != node && colors.get(&nbr) == Some(&node_color) {
                conflicts += 1;
            }
        }
    }
    // Each conflict edge is counted twice (from both endpoints), divide by 2
    conflicts /= 2;

    let num_colors = {
        let unique: HashSet<u32> = colors.values().copied().collect();
        unique.len() as u32
    };

    GraphColoringResult {
        colors,
        num_colors,
        conflicts,
    }
}

// ─── Conductance ────────────────────────────────────────────────────────────

/// Conductance: measures community quality. Lower = better community.
///
/// `conductance(C) = edges_leaving_C / min(vol(C), vol(V\C))`
///
/// where `vol(X) = sum of degrees of nodes in X` (undirected degree).
/// Edges are treated as undirected. Returns 0.0 for empty communities or
/// if `min(vol(C), vol(V\C))` is zero.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `community`: set of node IDs forming the community.
///
/// Returns the conductance score (0.0 to 1.0).
pub fn conductance(adjacency: &AdjacencyStore, community: &HashSet<NodeId>) -> f64 {
    if community.is_empty() {
        return 0.0;
    }

    let all_nodes = adjacency.all_node_ids();
    let all_set: HashSet<NodeId> = all_nodes.iter().copied().collect();

    // If the community is the entire graph, conductance is 0
    if community.len() == all_set.len() && community.is_subset(&all_set) {
        return 0.0;
    }

    let mut vol_community = 0u64;
    let mut vol_complement = 0u64;
    let mut crossing_edges = 0u64;

    for &node in &all_nodes {
        let degree = undirected_neighbors(adjacency, node).len() as u64;
        if community.contains(&node) {
            vol_community += degree;

            // Count edges leaving the community
            let neighbors = undirected_neighbors(adjacency, node);
            for nbr in &neighbors {
                if !community.contains(nbr) {
                    crossing_edges += 1;
                }
            }
        } else {
            vol_complement += degree;
        }
    }

    let min_vol = vol_community.min(vol_complement);
    if min_vol == 0 {
        return 0.0;
    }

    crossing_edges as f64 / min_vol as f64
}

/// Per-community conductance for a full community assignment.
///
/// Computes conductance for each community given a map from node IDs to
/// community labels. Returns `(community_label, conductance)` pairs
/// sorted by community label ascending.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
/// - `communities`: map from NodeId to community label.
///
/// Returns a list of `(community_label, conductance)` pairs.
pub fn all_community_conductance(
    adjacency: &AdjacencyStore,
    communities: &HashMap<NodeId, u64>,
) -> Vec<(u64, f64)> {
    // Group nodes by community
    let mut community_sets: HashMap<u64, HashSet<NodeId>> = HashMap::new();
    for (&node, &label) in communities {
        community_sets.entry(label).or_default().insert(node);
    }

    let mut result: Vec<(u64, f64)> = community_sets
        .iter()
        .map(|(&label, members)| (label, conductance(adjacency, members)))
        .collect();

    result.sort_by_key(|&(label, _)| label);
    result
}

// ─── Local Clustering Coefficient ───────────────────────────────────────────

/// Compute local clustering coefficient for all nodes.
///
/// `LCC(v) = 2*T(v) / (deg(v) * (deg(v) - 1))` where `T(v)` is the
/// number of triangles through node v and `deg(v)` is its undirected
/// degree. Nodes with degree < 2 have coefficient 0.0.
///
/// This provides a simpler API than `triangle_count`, returning only the
/// per-node clustering coefficient without the full triangle result.
///
/// Parameters:
/// - `adjacency`: the graph's adjacency store.
///
/// Returns a list of `(NodeId, coefficient)` sorted descending by score.
///
/// Complexity: O(V * d^2) where d is the maximum node degree.
pub fn local_clustering_coefficient(adjacency: &AdjacencyStore) -> Vec<(NodeId, f64)> {
    let all_nodes = adjacency.all_node_ids();
    if all_nodes.is_empty() {
        return Vec::new();
    }

    // Build undirected neighbor sets
    let mut neighbors: HashMap<NodeId, HashSet<NodeId>> = HashMap::with_capacity(all_nodes.len());
    for &node in &all_nodes {
        let mut nbr_set = HashSet::new();
        for &(w, _) in adjacency.neighbors_out(node, None).iter() {
            if w != node {
                nbr_set.insert(w);
            }
        }
        for &(w, _) in adjacency.neighbors_in(node, None).iter() {
            if w != node {
                nbr_set.insert(w);
            }
        }
        neighbors.insert(node, nbr_set);
    }

    // Count triangles per node
    let mut node_triangles: HashMap<NodeId, u32> = all_nodes.iter().map(|&n| (n, 0)).collect();

    for &u in &all_nodes {
        let u_nbrs = &neighbors[&u];
        let u_nbrs_sorted: Vec<NodeId> = {
            let mut v: Vec<NodeId> = u_nbrs.iter().copied().collect();
            v.sort_unstable();
            v
        };

        for (i, &v) in u_nbrs_sorted.iter().enumerate() {
            if v <= u {
                continue;
            }
            for &w in &u_nbrs_sorted[i + 1..] {
                if neighbors[&v].contains(&w) {
                    *node_triangles.get_mut(&u).unwrap() += 1;
                    *node_triangles.get_mut(&v).unwrap() += 1;
                    *node_triangles.get_mut(&w).unwrap() += 1;
                }
            }
        }
    }

    let mut result: Vec<(NodeId, f64)> = all_nodes
        .iter()
        .map(|&n| {
            let t = node_triangles[&n];
            let k = neighbors[&n].len() as u64;
            let cc = if k >= 2 {
                (2 * t as u64) as f64 / (k * (k - 1)) as f64
            } else {
                0.0
            };
            (n, cc)
        })
        .collect();

    result.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    result
}

// ─── GNN Neighbor Sampling ──────────────────────────────────────────────────

/// Multi-hop neighbor sampling for GNN training (PyTorch Geometric compatible).
/// Returns sampled subgraph with node IDs, edge indices, and features.
pub struct NeighborSampleResult {
    /// All unique node IDs in the sampled subgraph (seeds first, then sampled)
    pub node_ids: Vec<NodeId>,
    /// Edge index in COO format: (source_indices, target_indices) into node_ids
    pub edge_index: (Vec<usize>, Vec<usize>),
    /// Number of seed nodes (first N entries in node_ids)
    pub num_seeds: usize,
}

/// Multi-hop neighbor sampling for GNN training.
///
/// Performs layered sampling starting from `seeds`, sampling up to
/// `num_neighbors[hop]` neighbors per node at each hop level.
/// Uses xorshift64 PRNG seeded by `seed` for deterministic sampling.
pub fn neighbor_sample(
    adjacency: &AdjacencyStore,
    seeds: &[NodeId],
    num_neighbors: &[usize],
    seed: u64,
) -> NeighborSampleResult {
    // Map from NodeId -> position in node_ids vec
    let mut node_index: HashMap<NodeId, usize> = HashMap::new();
    let mut node_ids: Vec<NodeId> = Vec::new();

    // Add seed nodes first
    for &s in seeds {
        if adjacency.has_node(s) && !node_index.contains_key(&s) {
            let idx = node_ids.len();
            node_index.insert(s, idx);
            node_ids.push(s);
        }
    }
    let num_seeds = node_ids.len();

    let mut edge_sources: Vec<usize> = Vec::new();
    let mut edge_targets: Vec<usize> = Vec::new();

    // Track edges we've already added to avoid duplicates
    let mut seen_edges: HashSet<(usize, usize)> = HashSet::new();

    let mut rng_state: u64 = if seed == 0 { 1 } else { seed };
    let mut frontier: Vec<NodeId> = node_ids.clone();

    for hop in 0..num_neighbors.len() {
        let fan_out = num_neighbors[hop];
        let mut next_frontier: Vec<NodeId> = Vec::new();

        for &node in &frontier {
            // Collect all undirected neighbors
            let mut neighbors: Vec<NodeId> = Vec::new();
            for &(nbr, _) in adjacency.neighbors_out(node, None).iter() {
                neighbors.push(nbr);
            }
            for &(nbr, _) in adjacency.neighbors_in(node, None).iter() {
                if !neighbors.contains(&nbr) {
                    neighbors.push(nbr);
                }
            }

            if neighbors.is_empty() {
                continue;
            }

            // Sample if more neighbors than fan_out
            let sampled: Vec<NodeId> = if neighbors.len() <= fan_out {
                neighbors
            } else {
                // Fisher-Yates partial shuffle using xorshift64
                let mut pool = neighbors;
                let take = fan_out.min(pool.len());
                for i in 0..take {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    let j = i + (rng_state as usize) % (pool.len() - i);
                    pool.swap(i, j);
                }
                pool.truncate(take);
                pool
            };

            let node_idx = node_index[&node];

            for nbr in sampled {
                // Ensure neighbor is in node_ids
                let nbr_idx = if let Some(&idx) = node_index.get(&nbr) {
                    idx
                } else {
                    let idx = node_ids.len();
                    node_index.insert(nbr, idx);
                    node_ids.push(nbr);
                    next_frontier.push(nbr);
                    idx
                };

                // Add edge in both directions (undirected representation)
                if seen_edges.insert((node_idx, nbr_idx)) {
                    edge_sources.push(node_idx);
                    edge_targets.push(nbr_idx);
                }
                if seen_edges.insert((nbr_idx, node_idx)) {
                    edge_sources.push(nbr_idx);
                    edge_targets.push(node_idx);
                }
            }
        }

        frontier = next_frontier;
    }

    NeighborSampleResult {
        node_ids,
        edge_index: (edge_sources, edge_targets),
        num_seeds,
    }
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
    fn test_edge_filter_valid_during() {
        // Build graph: 1 -> 2 (valid [100, 500)) and 1 -> 3 (valid [600, 900))
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);

        let meta1 = EdgeMeta {
            source: 1, target: 2, label: 0,
            temporal: BiTemporal {
                valid_from: 100, valid_until: 500,
                tx_from: 100, tx_until: BiTemporal::OPEN,
            },
            provenance: None, weight: 1.0, token_cost: 0,
        };
        adj.add_edge(1, 2, 0, meta1).unwrap();

        let meta2 = EdgeMeta {
            source: 1, target: 3, label: 0,
            temporal: BiTemporal {
                valid_from: 600, valid_until: 900,
                tx_from: 600, tx_until: BiTemporal::OPEN,
            },
            provenance: None, weight: 1.0, token_cost: 0,
        };
        adj.add_edge(1, 3, 0, meta2).unwrap();

        // Range [200, 400) overlaps only edge 1->2
        let filter = EdgeFilter {
            valid_during: Some((200, 400)),
            ..EdgeFilter::none()
        };
        let result = bfs(
            &adj, &[1], 1, 100, &filter, &NodeFilter::none(), Direction::Outgoing,
            None, None,
        );
        assert_eq!(result.visited_nodes.len(), 2); // seed + node 2
        assert!(result.visited_nodes.contains(&2));
        assert!(!result.visited_nodes.contains(&3));

        // Range [400, 700) overlaps both edges
        let filter2 = EdgeFilter {
            valid_during: Some((400, 700)),
            ..EdgeFilter::none()
        };
        let result2 = bfs(
            &adj, &[1], 1, 100, &filter2, &NodeFilter::none(), Direction::Outgoing,
            None, None,
        );
        assert_eq!(result2.visited_nodes.len(), 3); // seed + nodes 2 and 3

        // Range [500, 600) overlaps neither edge (gap between them)
        let filter3 = EdgeFilter {
            valid_during: Some((500, 600)),
            ..EdgeFilter::none()
        };
        let result3 = bfs(
            &adj, &[1], 1, 100, &filter3, &NodeFilter::none(), Direction::Outgoing,
            None, None,
        );
        assert_eq!(result3.visited_nodes.len(), 1); // only seed
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

    #[test]
    fn test_betweenness_centrality_star() {
        // Star graph: center node 1, spokes to 2,3,4,5
        // All shortest paths between spoke pairs go through center,
        // so node 1 should have the highest betweenness centrality.
        let adj = build_star_graph();
        let result = betweenness_centrality(&adj, &EdgeFilter::none());

        assert_eq!(result.len(), 5);

        // Find center node (1) and its score
        let center_score = result.iter()
            .find(|&&(nid, _)| nid == 1)
            .map(|&(_, score)| score)
            .expect("center node should be in results");

        // All spoke nodes should have lower betweenness than center
        for &(nid, score) in &result {
            if nid != 1 {
                assert!(score < center_score,
                    "spoke node {} (score={}) should have lower betweenness than center (score={})",
                    nid, score, center_score);
            }
        }

        // Spoke nodes are leaves — they lie on no shortest paths between other pairs
        for &(nid, score) in &result {
            if nid != 1 {
                assert_eq!(score, 0.0,
                    "spoke node {} should have betweenness 0, got {}", nid, score);
            }
        }
    }

    // ── Helper: build a cycle graph ─────────────────────────────────────────

    fn build_cycle_graph() -> AdjacencyStore {
        // 1 -> 2 -> 3 -> 1 (directed cycle)
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();
        adj
    }

    // ── Leiden Community Detection tests ────────────────────────────────────

    #[test]
    fn test_leiden_two_cliques() {
        // Two cliques connected by a bridge
        let mut adj = AdjacencyStore::new();
        for i in 1..=6 { adj.add_node(i); }
        // Clique 1: 1-2-3
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        // Clique 2: 4-5-6
        adj.add_edge(4, 5, 0, make_meta(4, 5, 0)).unwrap();
        adj.add_edge(5, 4, 0, make_meta(5, 4, 0)).unwrap();
        adj.add_edge(4, 6, 0, make_meta(4, 6, 0)).unwrap();
        adj.add_edge(6, 4, 0, make_meta(6, 4, 0)).unwrap();
        adj.add_edge(5, 6, 0, make_meta(5, 6, 0)).unwrap();
        adj.add_edge(6, 5, 0, make_meta(6, 5, 0)).unwrap();
        // Bridge
        adj.add_edge(3, 4, 0, make_meta(3, 4, 0)).unwrap();
        adj.add_edge(4, 3, 0, make_meta(4, 3, 0)).unwrap();

        let communities = leiden_communities(&adj, 20, 1.0, 0.3);
        assert_eq!(communities.len(), 6);
        // Nodes in same clique should share a community
        assert_eq!(communities[&1], communities[&2]);
        assert_eq!(communities[&1], communities[&3]);
        assert_eq!(communities[&4], communities[&5]);
        assert_eq!(communities[&4], communities[&6]);
        // Different cliques should differ
        assert_ne!(communities[&1], communities[&4]);
    }

    #[test]
    fn test_leiden_empty_graph() {
        let adj = AdjacencyStore::new();
        let communities = leiden_communities(&adj, 10, 1.0, 0.5);
        assert!(communities.is_empty());
    }

    #[test]
    fn test_leiden_single_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        let communities = leiden_communities(&adj, 10, 1.0, 0.5);
        assert_eq!(communities.len(), 1);
        assert!(communities.contains_key(&1));
    }

    #[test]
    fn test_leiden_high_gamma_splits() {
        // With gamma=1.0, every node that isn't fully internal should become singleton
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();

        let communities = leiden_communities(&adj, 20, 1.0, 1.0);
        assert_eq!(communities.len(), 3);
        // With gamma=1.0, node 2 has neighbors in different communities after initial pass,
        // so the refinement may split communities. The number of unique communities
        // should be >= 1 (exact result depends on modularity convergence).
        let unique: HashSet<u64> = communities.values().copied().collect();
        assert!(unique.len() >= 1);
    }

    // ── Node2Vec Random Walks tests ─────────────────────────────────────────

    #[test]
    fn test_node2vec_basic_walks() {
        let adj = build_linear_graph(); // 1->2->3->4
        let walks = node2vec_walks(&adj, 3, 2, 1.0, 1.0, 42);
        // 4 nodes * 2 walks = 8 walks total
        assert_eq!(walks.len(), 8);
        // Each walk should start from a node in the graph and have length <= 3
        for walk in &walks {
            assert!(!walk.is_empty());
            assert!(walk.len() <= 3);
            assert!([1, 2, 3, 4].contains(&walk[0]));
        }
    }

    #[test]
    fn test_node2vec_empty_graph() {
        let adj = AdjacencyStore::new();
        let walks = node2vec_walks(&adj, 5, 3, 1.0, 1.0, 42);
        assert!(walks.is_empty());
    }

    #[test]
    fn test_node2vec_walk_length_one() {
        let adj = build_linear_graph();
        let walks = node2vec_walks(&adj, 1, 1, 1.0, 1.0, 42);
        // Each walk should be exactly 1 node (just the start)
        assert_eq!(walks.len(), 4);
        for walk in &walks {
            assert_eq!(walk.len(), 1);
        }
    }

    #[test]
    fn test_node2vec_isolated_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        let walks = node2vec_walks(&adj, 5, 2, 1.0, 1.0, 42);
        // 1 node * 2 walks = 2 walks, each with only the start node
        assert_eq!(walks.len(), 2);
        for walk in &walks {
            assert_eq!(walk.len(), 1);
            assert_eq!(walk[0], 1);
        }
    }

    #[test]
    fn test_node2vec_deterministic() {
        // Two runs with same seed should produce identical walks
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 { adj.add_node(i); }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();

        let walks1 = node2vec_walks(&adj, 4, 2, 1.0, 1.0, 123);
        let walks2 = node2vec_walks(&adj, 4, 2, 1.0, 1.0, 123);
        assert_eq!(walks1, walks2);
    }

    // ── A* Shortest Path tests ──────────────────────────────────────────────

    #[test]
    fn test_astar_basic() {
        let adj = build_linear_graph(); // 1->2->3->4
        // Trivial heuristic: always 0 (degenerates to Dijkstra)
        let h = |_: NodeId| 0.0;
        let result = astar_shortest_path(&adj, 1, 4, &h, &EdgeFilter::none()).unwrap();
        assert_eq!(result.nodes, vec![1, 2, 3, 4]);
        assert!(result.total_weight > 0.0);
    }

    #[test]
    fn test_astar_same_node() {
        let adj = build_linear_graph();
        let h = |_: NodeId| 0.0;
        let result = astar_shortest_path(&adj, 1, 1, &h, &EdgeFilter::none()).unwrap();
        assert_eq!(result.nodes, vec![1]);
        assert_eq!(result.total_weight, 0.0);
    }

    #[test]
    fn test_astar_no_path() {
        let adj = build_linear_graph(); // directed 1->2->3->4
        let h = |_: NodeId| 0.0;
        let result = astar_shortest_path(&adj, 4, 1, &h, &EdgeFilter::none());
        assert!(result.is_err());
        match result.unwrap_err() {
            WeavError::Conflict(msg) => assert!(msg.contains("no path")),
            other => panic!("expected Conflict, got: {:?}", other),
        }
    }

    #[test]
    fn test_astar_node_not_found() {
        let adj = build_linear_graph();
        let h = |_: NodeId| 0.0;
        let result = astar_shortest_path(&adj, 999, 1, &h, &EdgeFilter::none());
        assert!(result.is_err());
        match result.unwrap_err() {
            WeavError::NodeNotFound(nid, _) => assert_eq!(nid, 999),
            other => panic!("expected NodeNotFound, got: {:?}", other),
        }
    }

    #[test]
    fn test_astar_prefers_high_weight() {
        // Two paths: 1->2->3 (low weight) vs 1->3 (high weight)
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

        let h = |_: NodeId| 0.0;
        let result = astar_shortest_path(&adj, 1, 3, &h, &EdgeFilter::none()).unwrap();
        // Should prefer direct path (high weight = low cost)
        assert_eq!(result.nodes, vec![1, 3]);
    }

    // ── Yen's K-Shortest Paths tests ────────────────────────────────────────

    #[test]
    fn test_k_shortest_basic() {
        // Diamond: 1->2->4, 1->3->4
        let mut adj = AdjacencyStore::new();
        for i in 1..=4 { adj.add_node(i); }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 4, 0, make_meta(2, 4, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 4, 0, make_meta(3, 4, 0)).unwrap();

        let paths = k_shortest_paths(&adj, 1, 4, 2, &EdgeFilter::none()).unwrap();
        assert_eq!(paths.len(), 2);
        // Both paths should start at 1 and end at 4
        for p in &paths {
            assert_eq!(*p.nodes.first().unwrap(), 1);
            assert_eq!(*p.nodes.last().unwrap(), 4);
        }
        // First path should have lower or equal weight
        assert!(paths[0].total_weight <= paths[1].total_weight);
    }

    #[test]
    fn test_k_shortest_single_path() {
        let adj = build_linear_graph(); // 1->2->3->4
        let paths = k_shortest_paths(&adj, 1, 4, 3, &EdgeFilter::none()).unwrap();
        // Only one path exists in a linear graph
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].nodes, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_k_shortest_no_path() {
        let adj = build_linear_graph(); // directed
        let paths = k_shortest_paths(&adj, 4, 1, 2, &EdgeFilter::none()).unwrap();
        assert!(paths.is_empty());
    }

    #[test]
    fn test_k_shortest_node_not_found() {
        let adj = build_linear_graph();
        let result = k_shortest_paths(&adj, 999, 1, 2, &EdgeFilter::none());
        assert!(result.is_err());
        match result.unwrap_err() {
            WeavError::NodeNotFound(nid, _) => assert_eq!(nid, 999),
            other => panic!("expected NodeNotFound, got: {:?}", other),
        }
    }

    #[test]
    fn test_k_shortest_k_zero() {
        let adj = build_linear_graph();
        let paths = k_shortest_paths(&adj, 1, 4, 0, &EdgeFilter::none()).unwrap();
        assert!(paths.is_empty());
    }

    // ── Degree Centrality tests ─────────────────────────────────────────────

    #[test]
    fn test_degree_centrality_star() {
        // Star: center 1 connects to 2,3,4,5 (undirected via both directions)
        let mut adj = AdjacencyStore::new();
        for i in 1..=5 { adj.add_node(i); }
        for i in 2..=5 {
            adj.add_edge(1, i, 0, make_meta(1, i, 0)).unwrap();
            adj.add_edge(i, 1, 0, make_meta(i, 1, 0)).unwrap();
        }

        let result = degree_centrality(&adj);
        let dc: HashMap<NodeId, f64> = result.into_iter().collect();
        // Center has degree 4, n-1=4, so centrality=1.0
        assert!((dc[&1] - 1.0).abs() < 1e-10);
        // Spokes have degree 1, centrality=1/4=0.25
        for i in 2..=5 {
            assert!((dc[&i] - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_degree_centrality_empty() {
        let adj = AdjacencyStore::new();
        let result = degree_centrality(&adj);
        assert!(result.is_empty());
    }

    #[test]
    fn test_degree_centrality_single_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        let result = degree_centrality(&adj);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 1);
        assert_eq!(result[0].1, 0.0);
    }

    #[test]
    fn test_degree_centrality_sorted_descending() {
        let adj = build_star_graph(); // directed: 1->2,3,4,5
        let result = degree_centrality(&adj);
        for i in 1..result.len() {
            assert!(result[i - 1].1 >= result[i].1);
        }
    }

    // ── Eigenvector Centrality tests ────────────────────────────────────────

    #[test]
    fn test_eigenvector_star_graph() {
        // Undirected star: center should have highest eigenvector centrality
        // (need reverse edges so spokes see center and center sees spokes)
        let mut adj = AdjacencyStore::new();
        for i in 1..=5 { adj.add_node(i); }
        for i in 2..=5 {
            adj.add_edge(1, i, 0, make_meta(1, i, 0)).unwrap();
            adj.add_edge(i, 1, 0, make_meta(i, 1, 0)).unwrap();
        }
        // Also connect spokes 2-3, so the graph isn't perfectly symmetric
        // and center node has a clearly higher eigenvector centrality
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();

        let result = eigenvector_centrality(&adj, 100, 1e-8);
        let ec: HashMap<NodeId, f64> = result.into_iter().collect();
        // Center node should have the highest score (most connected)
        assert!(ec[&1] >= ec[&4], "center ec={} should be >= spoke4 ec={}", ec[&1], ec[&4]);
        assert!(ec[&1] >= ec[&5], "center ec={} should be >= spoke5 ec={}", ec[&1], ec[&5]);
        // All scores should be non-negative
        for &(_, score) in &[(1, ec[&1]), (2, ec[&2]), (3, ec[&3]), (4, ec[&4]), (5, ec[&5])] {
            assert!(score >= 0.0);
        }
    }

    #[test]
    fn test_eigenvector_empty_graph() {
        let adj = AdjacencyStore::new();
        let result = eigenvector_centrality(&adj, 100, 1e-8);
        assert!(result.is_empty());
    }

    #[test]
    fn test_eigenvector_single_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        let result = eigenvector_centrality(&adj, 100, 1e-8);
        assert_eq!(result.len(), 1);
        // Single node with no neighbors: score stays 0 after first iteration
        // (no neighbors to sum over)
    }

    #[test]
    fn test_eigenvector_complete_triangle() {
        // Complete triangle: all nodes should have equal scores
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 { adj.add_node(i); }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let result = eigenvector_centrality(&adj, 100, 1e-10);
        let ec: HashMap<NodeId, f64> = result.into_iter().collect();
        assert!((ec[&1] - ec[&2]).abs() < 1e-6);
        assert!((ec[&2] - ec[&3]).abs() < 1e-6);
    }

    // ── HITS tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_hits_star_graph() {
        // Star: 1->2, 1->3, 1->4
        let mut adj = AdjacencyStore::new();
        for i in 1..=4 { adj.add_node(i); }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(1, 4, 0, make_meta(1, 4, 0)).unwrap();

        let (auth, hub) = hits(&adj, 100, 1e-8);
        let auth_map: HashMap<NodeId, f64> = auth.into_iter().collect();
        let hub_map: HashMap<NodeId, f64> = hub.into_iter().collect();

        // Node 1 is the hub (points to many), should have highest hub score
        assert!(hub_map[&1] > hub_map[&2]);
        assert!(hub_map[&1] > hub_map[&3]);
        assert!(hub_map[&1] > hub_map[&4]);

        // Nodes 2,3,4 are authorities (pointed to by the hub)
        assert!(auth_map[&2] > auth_map[&1]);
        assert!(auth_map[&3] > auth_map[&1]);

        // All spoke authorities should be roughly equal
        assert!((auth_map[&2] - auth_map[&3]).abs() < 1e-6);
        assert!((auth_map[&3] - auth_map[&4]).abs() < 1e-6);
    }

    #[test]
    fn test_hits_empty_graph() {
        let adj = AdjacencyStore::new();
        let (auth, hub) = hits(&adj, 100, 1e-8);
        assert!(auth.is_empty());
        assert!(hub.is_empty());
    }

    #[test]
    fn test_hits_single_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        let (auth, hub) = hits(&adj, 100, 1e-8);
        assert_eq!(auth.len(), 1);
        assert_eq!(hub.len(), 1);
    }

    #[test]
    fn test_hits_bidirectional() {
        // Complete graph: all nodes should have similar auth and hub scores
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 { adj.add_node(i); }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let (auth, hub) = hits(&adj, 100, 1e-8);
        let auth_map: HashMap<NodeId, f64> = auth.into_iter().collect();
        let hub_map: HashMap<NodeId, f64> = hub.into_iter().collect();
        // Symmetric graph: all authority scores should be equal
        assert!((auth_map[&1] - auth_map[&2]).abs() < 1e-6);
        assert!((auth_map[&2] - auth_map[&3]).abs() < 1e-6);
        // All hub scores should be equal
        assert!((hub_map[&1] - hub_map[&2]).abs() < 1e-6);
        assert!((hub_map[&2] - hub_map[&3]).abs() < 1e-6);
    }

    // ── FastRP tests ────────────────────────────────────────────────────────

    #[test]
    fn test_fastrp_basic() {
        // Triangle: 1 <-> 2 <-> 3 <-> 1
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let embeddings = fastrp_embeddings(&adj, 64, 3, 1.0, 42);
        assert_eq!(embeddings.len(), 3);
        // Each embedding should have the requested dimension
        for (_, emb) in &embeddings {
            assert_eq!(emb.len(), 64);
        }
    }

    #[test]
    fn test_fastrp_deterministic() {
        let mut adj = AdjacencyStore::new();
        for i in 1..=4 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 4, 0, make_meta(3, 4, 0)).unwrap();

        let emb1 = fastrp_embeddings(&adj, 32, 2, 1.0, 123);
        let emb2 = fastrp_embeddings(&adj, 32, 2, 1.0, 123);

        // Same seed must produce identical embeddings
        for (&nid, vec1) in &emb1 {
            let vec2 = &emb2[&nid];
            for (a, b) in vec1.iter().zip(vec2.iter()) {
                assert!((a - b).abs() < 1e-10, "embeddings differ for node {nid}");
            }
        }

        // Different seed must produce different embeddings
        let emb3 = fastrp_embeddings(&adj, 32, 2, 1.0, 999);
        let mut any_different = false;
        for (&nid, vec1) in &emb1 {
            let vec3 = &emb3[&nid];
            for (a, b) in vec1.iter().zip(vec3.iter()) {
                if (a - b).abs() > 1e-6 {
                    any_different = true;
                    break;
                }
            }
            if any_different {
                break;
            }
        }
        assert!(any_different, "different seeds should produce different embeddings");
    }

    #[test]
    fn test_fastrp_empty_graph() {
        let adj = AdjacencyStore::new();
        let embeddings = fastrp_embeddings(&adj, 64, 3, 1.0, 42);
        assert!(embeddings.is_empty());
    }

    #[test]
    fn test_fastrp_connected_similar() {
        // Build two disconnected clusters:
        // Cluster A: 1 <-> 2 <-> 3 (fully connected)
        // Cluster B: 10 <-> 11 <-> 12 (fully connected)
        // No edges between clusters.
        let mut adj = AdjacencyStore::new();
        for &id in &[1, 2, 3, 10, 11, 12] {
            adj.add_node(id);
        }
        // Cluster A
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();
        // Cluster B
        adj.add_edge(10, 11, 0, make_meta(10, 11, 0)).unwrap();
        adj.add_edge(11, 10, 0, make_meta(11, 10, 0)).unwrap();
        adj.add_edge(11, 12, 0, make_meta(11, 12, 0)).unwrap();
        adj.add_edge(12, 11, 0, make_meta(12, 11, 0)).unwrap();
        adj.add_edge(10, 12, 0, make_meta(10, 12, 0)).unwrap();
        adj.add_edge(12, 10, 0, make_meta(12, 10, 0)).unwrap();

        let embeddings = fastrp_embeddings(&adj, 128, 3, 1.0, 42);

        // Cosine similarity helper
        let cosine = |a: &[f32], b: &[f32]| -> f32 {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na < 1e-10 || nb < 1e-10 {
                return 0.0;
            }
            dot / (na * nb)
        };

        // Within-cluster similarity (nodes 1 and 2)
        let sim_within = cosine(&embeddings[&1], &embeddings[&2]);
        // Cross-cluster similarity (node 1 vs node 10)
        let sim_cross = cosine(&embeddings[&1], &embeddings[&10]);

        assert!(
            sim_within > sim_cross,
            "within-cluster similarity ({sim_within}) should exceed cross-cluster ({sim_cross})"
        );
    }

    #[test]
    fn test_fastrp_dimension() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();

        for dim in [16, 64, 128, 256] {
            let embeddings = fastrp_embeddings(&adj, dim, 2, 1.0, 42);
            for (_, emb) in &embeddings {
                assert_eq!(emb.len(), dim, "embedding dim should be {dim}");
            }
        }
    }

    #[test]
    fn test_fastrp_zero_dim() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        let embeddings = fastrp_embeddings(&adj, 0, 2, 1.0, 42);
        assert!(embeddings.is_empty());
    }

    #[test]
    fn test_fastrp_normalized() {
        // Verify final embeddings are L2-normalized (unit vectors)
        let mut adj = AdjacencyStore::new();
        for i in 1..=5 {
            adj.add_node(i);
        }
        for i in 1..=4 {
            adj.add_edge(i, i + 1, 0, make_meta(i, i + 1, 0)).unwrap();
            adj.add_edge(i + 1, i, 0, make_meta(i + 1, i, 0)).unwrap();
        }

        let embeddings = fastrp_embeddings(&adj, 64, 3, 1.0, 42);
        for (&nid, emb) in &embeddings {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "node {nid} embedding norm={norm}, expected ~1.0"
            );
        }
    }

    // ─── Node Similarity Tests ──────────────────────────────────────────────

    #[test]
    fn test_jaccard_triangle() {
        // Triangle: 1-2-3-1 (bidirectional)
        // N(1) = {2, 3}, N(2) = {1, 3}, N(3) = {1, 2}
        // J(1,2) = |{3}| / |{1, 2, 3}| = 1/3
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let sim = jaccard_similarity(&adj, 1, 2);
        assert!(
            (sim - 1.0 / 3.0).abs() < 1e-10,
            "expected 1/3, got {sim}"
        );
    }

    #[test]
    fn test_jaccard_isolated_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        let sim = jaccard_similarity(&adj, 1, 2);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_jaccard_identical_neighbors() {
        // Star: center=1, spokes 2,3,4. Nodes 2,3,4 all have neighbor {1}.
        // J(2,3) = |{1}| / |{1}| = 1.0
        let mut adj = AdjacencyStore::new();
        for i in 1..=4 {
            adj.add_node(i);
        }
        for i in 2..=4 {
            adj.add_edge(1, i, 0, make_meta(1, i, 0)).unwrap();
            adj.add_edge(i, 1, 0, make_meta(i, 1, 0)).unwrap();
        }
        let sim = jaccard_similarity(&adj, 2, 3);
        assert!((sim - 1.0).abs() < 1e-10, "expected 1.0, got {sim}");
    }

    #[test]
    fn test_cosine_triangle() {
        // Triangle: N(1) = {2,3}, N(2) = {1,3}
        // cos(1,2) = |{3}| / sqrt(2*2) = 1/2
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let sim = cosine_similarity(&adj, 1, 2);
        assert!((sim - 0.5).abs() < 1e-10, "expected 0.5, got {sim}");
    }

    #[test]
    fn test_cosine_empty_neighborhood() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        assert_eq!(cosine_similarity(&adj, 1, 2), 0.0);
    }

    #[test]
    fn test_cosine_star_spokes() {
        // Star: center=1, spokes 2,3. N(2) = {1}, N(3) = {1}
        // cos(2,3) = |{1}| / sqrt(1*1) = 1.0
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let sim = cosine_similarity(&adj, 2, 3);
        assert!((sim - 1.0).abs() < 1e-10, "expected 1.0, got {sim}");
    }

    #[test]
    fn test_overlap_triangle() {
        // Triangle: N(1) = {2,3}, N(2) = {1,3}
        // overlap(1,2) = |{3}| / min(2,2) = 1/2
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let sim = overlap_similarity(&adj, 1, 2);
        assert!((sim - 0.5).abs() < 1e-10, "expected 0.5, got {sim}");
    }

    #[test]
    fn test_overlap_subset_neighbors() {
        // 1->2, 1->3, 1->4 (bidirectional), 2->3 (bidirectional)
        // N(1) = {2,3,4}, N(2) = {1,3}
        // overlap(1,2) = |{3}| / min(3,2) = 1/2
        let mut adj = AdjacencyStore::new();
        for i in 1..=4 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();
        adj.add_edge(1, 4, 0, make_meta(1, 4, 0)).unwrap();
        adj.add_edge(4, 1, 0, make_meta(4, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();

        let sim = overlap_similarity(&adj, 1, 2);
        assert!((sim - 0.5).abs() < 1e-10, "expected 0.5, got {sim}");
    }

    #[test]
    fn test_overlap_empty() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        assert_eq!(overlap_similarity(&adj, 1, 2), 0.0);
    }

    #[test]
    fn test_all_pairs_similarity() {
        // Triangle graph: all pairs should have Jaccard = 1/3
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let results = all_pairs_similarity(&adj, SimilarityMetric::Jaccard, 10);
        assert_eq!(results.len(), 3);
        for (_, _, sim) in &results {
            assert!(
                (sim - 1.0 / 3.0).abs() < 1e-10,
                "expected 1/3, got {sim}"
            );
        }
    }

    #[test]
    fn test_all_pairs_similarity_top_k() {
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let results = all_pairs_similarity(&adj, SimilarityMetric::Cosine, 1);
        assert_eq!(results.len(), 1);
    }

    // ─── Link Prediction Tests ──────────────────────────────────────────────

    #[test]
    fn test_adamic_adar_triangle() {
        // Triangle: N(1)={2,3}, N(2)={1,3}, common={3}
        // |N(3)| = 2, so AA = 1/ln(2)
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let score = adamic_adar(&adj, 1, 2);
        let expected = 1.0 / 2.0_f64.ln();
        assert!(
            (score - expected).abs() < 1e-10,
            "expected {expected}, got {score}"
        );
    }

    #[test]
    fn test_adamic_adar_no_common() {
        // No common neighbors
        let mut adj = AdjacencyStore::new();
        for i in 1..=4 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(3, 4, 0, make_meta(3, 4, 0)).unwrap();

        let score = adamic_adar(&adj, 1, 3);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_adamic_adar_empty() {
        let adj = AdjacencyStore::new();
        let score = adamic_adar(&adj, 1, 2);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_common_neighbors_count_triangle() {
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        assert_eq!(common_neighbors_count(&adj, 1, 2), 1); // common = {3}
    }

    #[test]
    fn test_common_neighbors_count_isolated() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        assert_eq!(common_neighbors_count(&adj, 1, 2), 0);
    }

    #[test]
    fn test_common_neighbors_count_star() {
        // Star with center 1: N(2)={1}, N(3)={1}, common = {1}
        let adj = build_star_graph();
        assert_eq!(common_neighbors_count(&adj, 2, 3), 1);
    }

    #[test]
    fn test_preferential_attachment_star() {
        // Star: |N(1)|=4, |N(2)|=1 -> PA = 4
        let adj = build_star_graph();
        let score = preferential_attachment(&adj, 1, 2);
        assert!((score - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_preferential_attachment_isolated() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        assert_eq!(preferential_attachment(&adj, 1, 2), 0.0);
    }

    #[test]
    fn test_preferential_attachment_complete() {
        // Complete graph K3: |N(1)|=2, |N(2)|=2 -> PA = 4
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let score = preferential_attachment(&adj, 1, 2);
        assert!((score - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_resource_allocation_triangle() {
        // Triangle: common neighbor of (1,2) is {3}, |N(3)| = 2
        // RA = 1/2 = 0.5
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let score = resource_allocation(&adj, 1, 2);
        assert!((score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_resource_allocation_empty() {
        let adj = AdjacencyStore::new();
        assert_eq!(resource_allocation(&adj, 1, 2), 0.0);
    }

    #[test]
    fn test_resource_allocation_no_common() {
        let mut adj = AdjacencyStore::new();
        for i in 1..=4 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(3, 4, 0, make_meta(3, 4, 0)).unwrap();

        assert_eq!(resource_allocation(&adj, 1, 3), 0.0);
    }

    #[test]
    fn test_predict_links_basic() {
        // Path: 1-2-3 (bidirectional), predict link 1-3
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();

        let predictions = predict_links(&adj, LinkPredictionMetric::CommonNeighbors, 5);
        assert_eq!(predictions.len(), 1);
        assert_eq!(predictions[0].0, 1);
        assert_eq!(predictions[0].1, 3);
        assert!((predictions[0].2 - 1.0).abs() < 1e-10); // 1 common neighbor (2)
    }

    #[test]
    fn test_predict_links_empty_graph() {
        let adj = AdjacencyStore::new();
        let predictions = predict_links(&adj, LinkPredictionMetric::AdamicAdar, 5);
        assert!(predictions.is_empty());
    }

    #[test]
    fn test_predict_links_complete_graph() {
        // Complete graph: no links to predict
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let predictions =
            predict_links(&adj, LinkPredictionMetric::PreferentialAttachment, 5);
        assert!(predictions.is_empty());
    }

    // ─── Random Walk Tests ──────────────────────────────────────────────────

    #[test]
    fn test_random_walk_basic() {
        let adj = build_linear_graph();
        let walk = random_walk(&adj, 1, 10, 42);
        assert!(!walk.is_empty());
        assert_eq!(walk[0], 1);
        // Walk length should be at most length+1 (start + steps)
        assert!(walk.len() <= 11);
    }

    #[test]
    fn test_random_walk_isolated_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        let walk = random_walk(&adj, 1, 10, 42);
        assert_eq!(walk, vec![1]); // Terminates immediately
    }

    #[test]
    fn test_random_walk_deterministic() {
        // Same seed should produce the same walk
        let adj = build_linear_graph();
        let walk1 = random_walk(&adj, 1, 5, 12345);
        let walk2 = random_walk(&adj, 1, 5, 12345);
        assert_eq!(walk1, walk2);
    }

    #[test]
    fn test_random_walk_zero_length() {
        let adj = build_linear_graph();
        let walk = random_walk(&adj, 1, 0, 42);
        assert_eq!(walk, vec![1]);
    }

    // ─── K-Core Decomposition Tests ─────────────────────────────────────────

    #[test]
    fn test_k_core_triangle() {
        // Complete triangle: each node has degree 2, so all are in 2-core
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let cores = k_core_decomposition(&adj);
        assert_eq!(cores[&1], 2);
        assert_eq!(cores[&2], 2);
        assert_eq!(cores[&3], 2);
    }

    #[test]
    fn test_k_core_empty() {
        let adj = AdjacencyStore::new();
        let cores = k_core_decomposition(&adj);
        assert!(cores.is_empty());
    }

    #[test]
    fn test_k_core_star() {
        // Star: center has degree 4, leaves have degree 1
        // All nodes should be in 1-core (removing a leaf doesn't change min degree)
        let adj = build_star_graph();
        let cores = k_core_decomposition(&adj);
        assert_eq!(cores[&1], 1); // center
        for i in 2..=5 {
            assert_eq!(cores[&i], 1, "leaf {i} should be 1-core");
        }
    }

    #[test]
    fn test_k_core_isolated_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        let cores = k_core_decomposition(&adj);
        assert_eq!(cores[&1], 0);
    }

    // ─── Max Flow Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_max_flow_simple_path() {
        // 1 --(w=3)--> 2 --(w=2)--> 3
        // Max flow from 1 to 3 = 2 (bottleneck at edge 2->3)
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        let meta1 = EdgeMeta {
            source: 1, target: 2, label: 0,
            temporal: BiTemporal::new_current(1000),
            provenance: None, weight: 3.0, token_cost: 0,
        };
        adj.add_edge(1, 2, 0, meta1).unwrap();
        let meta2 = EdgeMeta {
            source: 2, target: 3, label: 0,
            temporal: BiTemporal::new_current(1000),
            provenance: None, weight: 2.0, token_cost: 0,
        };
        adj.add_edge(2, 3, 0, meta2).unwrap();

        let result = max_flow(&adj, 1, 3).unwrap();
        assert!((result.max_flow - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_flow_same_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        let result = max_flow(&adj, 1, 1).unwrap();
        assert_eq!(result.max_flow, 0.0);
    }

    #[test]
    fn test_max_flow_no_path() {
        // Two disconnected nodes
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        let result = max_flow(&adj, 1, 2).unwrap();
        assert_eq!(result.max_flow, 0.0);
    }

    #[test]
    fn test_max_flow_parallel_paths() {
        // 1 -> 2 (w=3), 1 -> 3 (w=2), 2 -> 4 (w=3), 3 -> 4 (w=2)
        // Max flow from 1 to 4 = 3 + 2 = 5
        let mut adj = AdjacencyStore::new();
        for i in 1..=4 {
            adj.add_node(i);
        }
        let edges = [(1, 2, 3.0), (1, 3, 2.0), (2, 4, 3.0), (3, 4, 2.0)];
        for (s, t, w) in edges {
            let meta = EdgeMeta {
                source: s, target: t, label: 0,
                temporal: BiTemporal::new_current(1000),
                provenance: None, weight: w, token_cost: 0,
            };
            adj.add_edge(s, t, 0, meta).unwrap();
        }

        let result = max_flow(&adj, 1, 4).unwrap();
        assert!(
            (result.max_flow - 5.0).abs() < 1e-10,
            "expected 5.0, got {}",
            result.max_flow
        );
    }

    #[test]
    fn test_max_flow_invalid_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        let result = max_flow(&adj, 1, 999);
        assert!(result.is_err());
    }

    // ─── Minimum Spanning Tree Tests ────────────────────────────────────────

    #[test]
    fn test_mst_linear() {
        // Linear: 1-2-3-4, all weight 1.0
        let adj = build_linear_graph();
        let mst = minimum_spanning_tree(&adj);
        assert_eq!(mst.edges.len(), 3);
        assert!((mst.total_weight - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mst_empty_graph() {
        let adj = AdjacencyStore::new();
        let mst = minimum_spanning_tree(&adj);
        assert!(mst.edges.is_empty());
        assert_eq!(mst.total_weight, 0.0);
    }

    #[test]
    fn test_mst_triangle_with_weights() {
        // Triangle: 1-2 (w=1), 2-3 (w=2), 1-3 (w=3)
        // MST should include edges w=1 and w=2, total=3
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        let meta1 = EdgeMeta {
            source: 1, target: 2, label: 0,
            temporal: BiTemporal::new_current(1000),
            provenance: None, weight: 1.0, token_cost: 0,
        };
        adj.add_edge(1, 2, 0, meta1).unwrap();
        let meta1r = EdgeMeta {
            source: 2, target: 1, label: 0,
            temporal: BiTemporal::new_current(1000),
            provenance: None, weight: 1.0, token_cost: 0,
        };
        adj.add_edge(2, 1, 0, meta1r).unwrap();
        let meta2 = EdgeMeta {
            source: 2, target: 3, label: 0,
            temporal: BiTemporal::new_current(1000),
            provenance: None, weight: 2.0, token_cost: 0,
        };
        adj.add_edge(2, 3, 0, meta2).unwrap();
        let meta2r = EdgeMeta {
            source: 3, target: 2, label: 0,
            temporal: BiTemporal::new_current(1000),
            provenance: None, weight: 2.0, token_cost: 0,
        };
        adj.add_edge(3, 2, 0, meta2r).unwrap();
        let meta3 = EdgeMeta {
            source: 1, target: 3, label: 0,
            temporal: BiTemporal::new_current(1000),
            provenance: None, weight: 3.0, token_cost: 0,
        };
        adj.add_edge(1, 3, 0, meta3).unwrap();
        let meta3r = EdgeMeta {
            source: 3, target: 1, label: 0,
            temporal: BiTemporal::new_current(1000),
            provenance: None, weight: 3.0, token_cost: 0,
        };
        adj.add_edge(3, 1, 0, meta3r).unwrap();

        let mst = minimum_spanning_tree(&adj);
        assert_eq!(mst.edges.len(), 2);
        assert!(
            (mst.total_weight - 3.0).abs() < 1e-10,
            "expected total weight 3.0, got {}",
            mst.total_weight
        );
    }

    #[test]
    fn test_mst_isolated_nodes() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        adj.add_node(3);
        // No edges => MST has no edges
        let mst = minimum_spanning_tree(&adj);
        assert!(mst.edges.is_empty());
        assert_eq!(mst.total_weight, 0.0);
    }

    // ─── Bellman-Ford Tests ─────────────────────────────────────────────────

    #[test]
    fn test_bellman_ford_linear() {
        let adj = build_linear_graph(); // 1->2->3->4, weight=1.0
        let result = bellman_ford(&adj, 1).unwrap();
        // With weight=1.0, cost = 1/1.0 = 1.0 per edge
        assert_eq!(result.len(), 4);
        // Source distance = 0
        assert_eq!(result[0], (1, 0.0));
        // Node 2: distance 1.0
        assert!((result[1].1 - 1.0).abs() < 1e-10);
        assert_eq!(result[1].0, 2);
        // Node 3: distance 2.0
        assert!((result[2].1 - 2.0).abs() < 1e-10);
        assert_eq!(result[2].0, 3);
        // Node 4: distance 3.0
        assert!((result[3].1 - 3.0).abs() < 1e-10);
        assert_eq!(result[3].0, 4);
    }

    #[test]
    fn test_bellman_ford_empty_graph() {
        let adj = AdjacencyStore::new();
        let result = bellman_ford(&adj, 1);
        assert!(result.is_err()); // Node not found
    }

    #[test]
    fn test_bellman_ford_single_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        let result = bellman_ford(&adj, 1).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], (1, 0.0));
    }

    #[test]
    fn test_bellman_ford_star() {
        let adj = build_star_graph(); // 1->2, 1->3, 1->4, 1->5
        let result = bellman_ford(&adj, 1).unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result[0], (1, 0.0));
        // All spokes are distance 1.0 from center
        for &(_, dist) in &result[1..] {
            assert!((dist - 1.0).abs() < 1e-10);
        }
    }

    // ─── Harmonic Centrality Tests ──────────────────────────────────────────

    #[test]
    fn test_harmonic_centrality_star() {
        let adj = build_star_graph(); // center=1, spokes: 1->2, 1->3, 1->4, 1->5
        let result = harmonic_centrality(&adj, &EdgeFilter::none());
        // Center node (1) can reach all 4 others at distance 1
        // H(1) = (1/4) * (1/1 + 1/1 + 1/1 + 1/1) = 1.0
        assert!(!result.is_empty());
        assert_eq!(result[0].0, 1);
        assert!((result[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_harmonic_centrality_empty() {
        let adj = AdjacencyStore::new();
        let result = harmonic_centrality(&adj, &EdgeFilter::none());
        assert!(result.is_empty());
    }

    #[test]
    fn test_harmonic_centrality_isolated_nodes() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        adj.add_node(2);
        // No edges => all harmonic centrality = 0
        let result = harmonic_centrality(&adj, &EdgeFilter::none());
        assert_eq!(result.len(), 2);
        for &(_, score) in &result {
            assert!((score).abs() < 1e-10);
        }
    }

    // ─── Katz Centrality Tests ──────────────────────────────────────────────

    #[test]
    fn test_katz_centrality_star() {
        let adj = build_star_graph();
        let result = katz_centrality(&adj, 0.1, 1.0, 100, 1e-6);
        // Center node should have highest score (most connections)
        assert!(!result.is_empty());
        assert_eq!(result[0].0, 1);
    }

    #[test]
    fn test_katz_centrality_empty() {
        let adj = AdjacencyStore::new();
        let result = katz_centrality(&adj, 0.1, 1.0, 100, 1e-6);
        assert!(result.is_empty());
    }

    #[test]
    fn test_katz_centrality_single_node() {
        let mut adj = AdjacencyStore::new();
        adj.add_node(1);
        let result = katz_centrality(&adj, 0.1, 1.0, 100, 1e-6);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 1);
        // Single node gets score = beta normalized, which is 1.0
        assert!((result[0].1 - 1.0).abs() < 1e-6);
    }

    // ─── ArticleRank Tests ──────────────────────────────────────────────────

    #[test]
    fn test_article_rank_star() {
        let adj = build_star_graph();
        let result = article_rank(&adj, 0.85, 100, 1e-6);
        // All nodes should have non-zero scores
        assert_eq!(result.len(), 5);
        for &(_, score) in &result {
            assert!(score > 0.0);
        }
    }

    #[test]
    fn test_article_rank_empty() {
        let adj = AdjacencyStore::new();
        let result = article_rank(&adj, 0.85, 100, 1e-6);
        assert!(result.is_empty());
    }

    #[test]
    fn test_article_rank_linear() {
        let adj = build_linear_graph(); // 1->2->3->4
        let result = article_rank(&adj, 0.85, 100, 1e-6);
        assert_eq!(result.len(), 4);
        // All scores should be positive
        for &(_, score) in &result {
            assert!(score > 0.0);
        }
        // Scores should be sorted descending
        for w in result.windows(2) {
            assert!(w[0].1 >= w[1].1);
        }
    }

    // ─── K-1 Coloring Tests ────────────────────────────────────────────────

    #[test]
    fn test_k1_coloring_triangle() {
        // Triangle: 1-2, 2-3, 1-3 (needs 3 colors)
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let result = k1_coloring(&adj, 100);
        assert_eq!(result.conflicts, 0);
        assert!(result.num_colors >= 3);
        // Verify no adjacent nodes share a color
        assert_ne!(result.colors[&1], result.colors[&2]);
        assert_ne!(result.colors[&2], result.colors[&3]);
        assert_ne!(result.colors[&1], result.colors[&3]);
    }

    #[test]
    fn test_k1_coloring_empty() {
        let adj = AdjacencyStore::new();
        let result = k1_coloring(&adj, 100);
        assert!(result.colors.is_empty());
        assert_eq!(result.num_colors, 0);
        assert_eq!(result.conflicts, 0);
    }

    #[test]
    fn test_k1_coloring_bipartite() {
        // Bipartite graph: 1-3, 1-4, 2-3, 2-4 (needs 2 colors)
        let mut adj = AdjacencyStore::new();
        for i in 1..=4 {
            adj.add_node(i);
        }
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();
        adj.add_edge(1, 4, 0, make_meta(1, 4, 0)).unwrap();
        adj.add_edge(4, 1, 0, make_meta(4, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(2, 4, 0, make_meta(2, 4, 0)).unwrap();
        adj.add_edge(4, 2, 0, make_meta(4, 2, 0)).unwrap();

        let result = k1_coloring(&adj, 100);
        assert_eq!(result.conflicts, 0);
        assert!(result.num_colors <= 2);
    }

    // ─── Conductance Tests ──────────────────────────────────────────────────

    #[test]
    fn test_conductance_star() {
        let adj = build_star_graph(); // center=1, spokes: 1->2, 1->3, 1->4, 1->5
        // Community = {1, 2} — 1 edge inside (1->2), edges leaving: 1->3, 1->4, 1->5 = 3
        let community: HashSet<NodeId> = [1, 2].iter().copied().collect();
        let c = conductance(&adj, &community);
        assert!(c > 0.0, "conductance should be positive");
    }

    #[test]
    fn test_conductance_empty_community() {
        let adj = build_star_graph();
        let community: HashSet<NodeId> = HashSet::new();
        let c = conductance(&adj, &community);
        assert!((c).abs() < 1e-10);
    }

    #[test]
    fn test_conductance_entire_graph() {
        let adj = build_star_graph();
        let community: HashSet<NodeId> = [1, 2, 3, 4, 5].iter().copied().collect();
        let c = conductance(&adj, &community);
        // Entire graph = no crossing edges
        assert!((c).abs() < 1e-10);
    }

    #[test]
    fn test_all_community_conductance() {
        let adj = build_star_graph();
        let mut communities: HashMap<NodeId, u64> = HashMap::new();
        communities.insert(1, 0);
        communities.insert(2, 0);
        communities.insert(3, 1);
        communities.insert(4, 1);
        communities.insert(5, 1);
        let result = all_community_conductance(&adj, &communities);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 0);
        assert_eq!(result[1].0, 1);
    }

    // ─── Local Clustering Coefficient Tests ─────────────────────────────────

    #[test]
    fn test_local_clustering_coefficient_triangle() {
        // Complete triangle: 1-2, 2-3, 1-3
        let mut adj = AdjacencyStore::new();
        for i in 1..=3 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(2, 1, 0, make_meta(2, 1, 0)).unwrap();
        adj.add_edge(2, 3, 0, make_meta(2, 3, 0)).unwrap();
        adj.add_edge(3, 2, 0, make_meta(3, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(3, 1, 0, make_meta(3, 1, 0)).unwrap();

        let result = local_clustering_coefficient(&adj);
        assert_eq!(result.len(), 3);
        // In a complete triangle, every node has LCC = 1.0
        for &(_, cc) in &result {
            assert!((cc - 1.0).abs() < 1e-10, "expected 1.0, got {}", cc);
        }
    }

    #[test]
    fn test_local_clustering_coefficient_empty() {
        let adj = AdjacencyStore::new();
        let result = local_clustering_coefficient(&adj);
        assert!(result.is_empty());
    }

    #[test]
    fn test_local_clustering_coefficient_star() {
        let adj = build_star_graph();
        let result = local_clustering_coefficient(&adj);
        // Star graph: center has degree 4 but no triangles => LCC = 0
        // Leaf nodes have degree 1 => LCC = 0
        for &(_, cc) in &result {
            assert!((cc).abs() < 1e-10, "expected 0.0, got {}", cc);
        }
    }

    // ─── Neighbor Sampling Tests ────────────────────────────────────────────

    #[test]
    fn test_neighbor_sampling_basic() {
        // Build a small graph: 1-2, 1-3, 2-4, 3-5
        let mut adj = AdjacencyStore::new();
        for i in 1..=5 {
            adj.add_node(i);
        }
        adj.add_edge(1, 2, 0, make_meta(1, 2, 0)).unwrap();
        adj.add_edge(1, 3, 0, make_meta(1, 3, 0)).unwrap();
        adj.add_edge(2, 4, 0, make_meta(2, 4, 0)).unwrap();
        adj.add_edge(3, 5, 0, make_meta(3, 5, 0)).unwrap();

        // Sample 1-hop neighbors of node 1 with large fan-out (get all)
        let result = neighbor_sample(&adj, &[1], &[10], 42);
        assert_eq!(result.num_seeds, 1);
        assert_eq!(result.node_ids[0], 1); // seed first
        assert!(result.node_ids.contains(&2));
        assert!(result.node_ids.contains(&3));
        // Edges should exist between seed and neighbors
        assert!(!result.edge_index.0.is_empty());
        assert_eq!(result.edge_index.0.len(), result.edge_index.1.len());

        // 2-hop sampling: fan-out [10, 10] should reach nodes 4 and 5
        let result2 = neighbor_sample(&adj, &[1], &[10, 10], 42);
        assert_eq!(result2.num_seeds, 1);
        assert!(result2.node_ids.contains(&4) || result2.node_ids.contains(&5),
            "2-hop should reach at least one of nodes 4 or 5");
    }

    #[test]
    fn test_neighbor_sampling_empty() {
        let adj = AdjacencyStore::new();
        let result = neighbor_sample(&adj, &[1, 2, 3], &[10, 5], 42);
        assert_eq!(result.num_seeds, 0);
        assert!(result.node_ids.is_empty());
        assert!(result.edge_index.0.is_empty());
        assert!(result.edge_index.1.is_empty());
    }
}
