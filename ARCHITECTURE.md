# Weav Architecture

Weav is an in-memory context graph database purpose-built for AI and LLM workloads. It stores knowledge as a labeled property graph with vector embeddings, bi-temporal tracking, and provenance metadata. A built-in context retrieval engine assembles token-budgeted context windows by combining vector similarity search, graph traversal, relevance scoring, and temporal decay. Three concurrent protocol interfaces (RESP3, gRPC, HTTP REST) share a single thread-safe engine.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Crate Dependency Graph](#2-crate-dependency-graph)
3. [Core Type System (weav-core)](#3-core-type-system-weav-core)
4. [Graph Storage Layer (weav-graph)](#4-graph-storage-layer-weav-graph)
5. [Vector Index & Token Counting (weav-vector)](#5-vector-index--token-counting-weav-vector)
6. [Query Pipeline (weav-query)](#6-query-pipeline-weav-query)
7. [Persistence Layer (weav-persist)](#7-persistence-layer-weav-persist)
8. [Protocol Layer (weav-proto)](#8-protocol-layer-weav-proto)
9. [Server & Engine (weav-server)](#9-server--engine-weav-server)
10. [CLI Client (weav-cli)](#10-cli-client-weav-cli)
11. [Client SDKs](#11-client-sdks)
12. [Data Flow: Context Query Lifecycle](#12-data-flow-context-query-lifecycle)
13. [Concurrency Model](#13-concurrency-model)
14. [Memory Layout & Compactness](#14-memory-layout--compactness)
15. [Configuration System](#15-configuration-system)
16. [Benchmarks](#16-benchmarks)

---

## 1. System Overview

```
                         Clients
              ┌────────────┼────────────┐
              │            │            │
         ┌────┴────┐  ┌───┴───┐  ┌────┴────┐
         │  RESP3  │  │  gRPC │  │  HTTP   │
         │  :6380  │  │ :6381 │  │  :6382  │
         └────┬────┘  └───┬───┘  └────┬────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────┴──────┐
                    │   Engine    │
                    │  (Arc<RW>)  │
                    └──────┬──────┘
                           │
         ┌─────────┬───────┼───────┬──────────┐
         │         │       │       │          │
    ┌────┴───┐ ┌───┴──┐ ┌──┴──┐ ┌─┴────┐ ┌───┴────┐
    │ Graph  │ │Vector│ │Query│ │Dedup │ │Persist │
    │ Store  │ │Index │ │Exec │ │      │ │(WAL+   │
    │        │ │(HNSW)│ │     │ │      │ │ Snap)  │
    └────────┘ └──────┘ └─────┘ └──────┘ └────────┘
```

**Key design principles:**

- **Single-process, multi-protocol.** One `Engine` instance behind three concurrent servers.
- **AI-native query primitive.** The `CONTEXT` command combines vector search, graph traversal, relevance scoring, temporal filtering, and token budget enforcement in a single call.
- **Bi-temporal by default.** Every entity tracks real-world validity (`valid_from`/`valid_until`) and database transaction time (`tx_from`/`tx_until`).
- **Provenance-aware.** Facts carry source, confidence, and extraction method metadata.
- **Compact in memory.** String interning, `SmallVec` adjacency, `RoaringBitmap` membership, `CompactString` storage.

---

## 2. Crate Dependency Graph

```
weav-core  (foundation — zero intra-workspace deps)
  ├── weav-graph      (adjacency, properties, traversal, dedup)
  ├── weav-vector     (HNSW index, token counting)
  ├── weav-persist    (WAL, snapshots, recovery)
  ├── weav-query      (parser, planner, executor, budget)
  │     └── depends on: weav-graph, weav-vector
  ├── weav-proto      (RESP3 codec, gRPC proto, command mapping)
  │     └── depends on: weav-query
  └── weav-server     (engine, HTTP, RESP3, gRPC servers)
        └── depends on: all above
            ├── weav-cli        (REPL client)
            └── benchmarks      (criterion suite)
```

All crates share `edition = "2024"` (Rust 1.85+). The workspace uses thin LTO, `codegen-units = 1`, and `opt-level = 3` for release builds.

---

## 3. Core Type System (weav-core)

**Source:** `weav-core/src/`

### 3.1 Identifiers

| Type             | Rust Type | Purpose                           |
|------------------|-----------|-----------------------------------|
| `NodeId`         | `u64`     | Node identifier (auto-increment)  |
| `EdgeId`         | `u64`     | Edge identifier (auto-increment)  |
| `GraphId`        | `u32`     | Graph identifier                  |
| `ShardId`        | `u16`     | Shard routing key                 |
| `LabelId`        | `u16`     | Interned node/edge type label     |
| `PropertyKeyId`  | `u16`     | Interned property key             |
| `Timestamp`      | `u64`     | Milliseconds since Unix epoch     |

All identifiers are fixed-width integers for cache-friendly storage and zero-cost comparisons.

### 3.2 Bi-Temporal Tracking

```rust
// weav-core/src/types.rs
pub struct BiTemporal {
    pub valid_from: Timestamp,    // When fact became true in the real world
    pub valid_until: Timestamp,   // When fact stopped being true (u64::MAX = still valid)
    pub tx_from: Timestamp,       // When written to database
    pub tx_until: Timestamp,      // When superseded (u64::MAX = current)
}
```

- **Validity interval** `[valid_from, valid_until)` — models real-world truth.
- **Transaction interval** `[tx_from, tx_until)` — models database history.
- `is_valid_at(ts)` and `is_current_at(tx_ts)` — half-open interval checks.
- `is_active()` — both intervals are open (still valid AND current).

This enables point-in-time queries ("What did the graph look like at timestamp T?") and auditability ("When was this fact recorded?").

### 3.3 Dynamic Value Type

```rust
pub enum Value {
    Null, Bool(bool), Int(i64), Float(f64),
    String(CompactString), Bytes(Vec<u8>), Vector(Vec<f32>),
    List(Vec<Value>), Map(Vec<(CompactString, Value)>), Timestamp(u64),
}
```

Schema-free: properties accept any `Value` variant. Accessor methods (`as_str()`, `as_int()`, etc.) return `Option<&T>`.

### 3.4 Provenance

```rust
pub struct Provenance {
    pub source: CompactString,                     // e.g., "gpt-4-turbo", "user-input"
    pub confidence: f32,                           // [0.0, 1.0], clamped on construction
    pub extraction_method: ExtractionMethod,       // LlmExtracted | NlpPipeline | UserProvided | Derived | Imported
    pub source_document_id: Option<CompactString>,
    pub source_chunk_offset: Option<u32>,
}
```

Every node and edge can carry provenance metadata, enabling confidence-weighted retrieval and source attribution.

### 3.5 Decay Functions

Five time-based relevance decay strategies:

| Variant        | Behavior                                                  |
|----------------|-----------------------------------------------------------|
| `None`         | No decay — score unchanged                                |
| `Exponential`  | Half-life decay: `score * 2^(-age/half_life)`             |
| `Linear`       | Linear ramp: `score * max(0, 1 - age/max_age)`           |
| `Step`         | Hard cutoff: score if `age < cutoff`, else 0              |
| `Custom`       | Piecewise-linear interpolation over `(age_ms, multiplier)` breakpoints |

Applied during context query execution to down-weight stale facts.

### 3.6 Conflict Resolution

```rust
pub enum ConflictPolicy {
    LastWriteWins,             // Default — newest value wins
    HighestConfidence,         // Winner by provenance.confidence
    TemporalInvalidation,      // Mark old as invalid, keep both
    Merge,                     // Attempt property-level merge
    Reject,                    // Reject conflicting writes
}
```

### 3.7 Token Budget

```rust
pub struct TokenBudget {
    pub max_tokens: u32,
    pub allocation: TokenAllocation,
}

pub enum TokenAllocation {
    Proportional { entities_pct, relationships_pct, text_chunks_pct, metadata_pct },
    Priority(Vec<ContentPriority>),
    Auto,
}
```

Controls how many tokens the context retrieval engine packs into the result, and how budget is distributed across content categories.

### 3.8 Input Structures

```rust
pub struct NodeData {
    pub label: CompactString,
    pub properties: Vec<(CompactString, Value)>,
    pub embedding: Option<Vec<f32>>,
    pub entity_key: Option<CompactString>,    // For dedup matching
    pub provenance: Option<Provenance>,
}

pub struct EdgeData {
    pub source: NodeId,
    pub target: NodeId,
    pub label: CompactString,
    pub properties: Vec<(CompactString, Value)>,
    pub weight: f32,
    pub provenance: Option<Provenance>,
}
```

### 3.9 Error Handling

All fallible operations return `WeavResult<T>` (alias for `Result<T, WeavError>`). `WeavError` is a flat `thiserror` enum with 14 variants:

`GraphNotFound`, `NodeNotFound`, `EdgeNotFound`, `DuplicateNode`, `Conflict`, `TokenBudgetExceeded`, `DimensionMismatch`, `ShardUnavailable`, `PersistenceError`, `ProtocolError`, `QueryParseError`, `CapacityExceeded`, `InvalidConfig`, `Internal`.

### 3.10 Shard & String Interning

**Source:** `weav-core/src/shard.rs`

```rust
pub struct Shard {
    pub id: ShardId,
    graphs: HashMap<GraphId, GraphShard>,
    string_interner: StringInterner,
    stats: ShardStats,           // AtomicU64 counters (lock-free)
    allocator: Bump,             // Arena allocator for temporaries
}
```

The `StringInterner` provides bidirectional `CompactString <-> u16` mapping for labels and property keys:

```rust
pub struct StringInterner {
    label_to_id: HashMap<CompactString, LabelId>,
    id_to_label: HashMap<LabelId, CompactString>,
    prop_to_id: HashMap<CompactString, PropertyKeyId>,
    id_to_prop: HashMap<PropertyKeyId, CompactString>,
    // monotonic counters
}
```

Up to 65,535 unique labels and 65,535 unique property keys per interner. This trades interning cost on write for `u16` comparisons on read.

### 3.11 Message Bus

**Source:** `weav-core/src/bus.rs`

Cross-shard communication via bounded `crossbeam_channel` channels. Messages carry embedded response channels for async request/reply:

```rust
pub enum ShardMessage {
    TraverseFrom { node_ids, depth, respond_to: Sender<TraversalResult> },
    VectorSearch { query_vector, k, respond_to: Sender<Vec<ScoredNode>> },
    InsertNode { node, respond_to: Sender<Result<NodeId, WeavError>> },
    Snapshot { respond_to: Sender<()> },
    Shutdown,
}
```

Routing: `MessageBus::route_by_key(graph_id, node_id)` uses **xxHash3** to deterministically map `(graph_id, node_id)` to a shard: `xxh3_hash % num_shards`.

---

## 4. Graph Storage Layer (weav-graph)

**Source:** `weav-graph/src/`

### 4.1 Adjacency Store

```
adjacency.rs
┌──────────────────────────────────────────────────────────┐
│ AdjacencyStore                                           │
│                                                          │
│  forward:  HashMap<LabelId, DirectedAdjacency>           │
│  backward: HashMap<LabelId, DirectedAdjacency>           │
│  edge_meta: HashMap<EdgeId, EdgeMeta>                    │
│  node_bitmap: RoaringBitmap                              │
│                                                          │
│  DirectedAdjacency {                                     │
│    adjacency: HashMap<NodeId, SmallVec<[(NodeId,EdgeId);8]>> │
│  }                                                       │
└──────────────────────────────────────────────────────────┘
```

**Design decisions:**

1. **Label-partitioned adjacency.** Outgoing and incoming edges are stored in separate `DirectedAdjacency` maps keyed by `LabelId`. This enables O(1) label filtering — a traversal restricted to edge label "mentions" only touches that label's adjacency map.

2. **SmallVec<[(NodeId, EdgeId); 8]>.** Inline storage for up to 8 neighbors avoids heap allocation for typical node degrees. When degree exceeds 8, SmallVec spills to a heap Vec transparently.

3. **RoaringBitmap for node membership.** Compressed bitmap enables fast `has_node()` checks and efficient iteration over all node IDs, even for millions of nodes.

4. **Bidirectional storage.** Both `forward` (outgoing) and `backward` (incoming) adjacency maps are maintained, enabling efficient traversal in any direction without full scans.

**Edge metadata:**

```rust
pub struct EdgeMeta {
    pub source: NodeId,
    pub target: NodeId,
    pub label: LabelId,
    pub temporal: BiTemporal,
    pub provenance: Option<Provenance>,
    pub weight: f32,
    pub token_cost: u16,
}
```

**Key operations:**
- `add_edge(src, tgt, label, meta)` — validates both nodes exist, assigns EdgeId, inserts into forward+backward maps.
- `remove_node(node_id)` — cascades: removes all connected edges before removing the node.
- `neighbors_at(node, timestamp, label)` — filters edges by temporal validity at the given timestamp.
- `edge_between(src, tgt, label)` — checks if a specific edge exists.
- `edge_history(src, tgt)` — returns all edges between a node pair (for temporal versioning).

### 4.2 Property Store

```
properties.rs
┌────────────────────────────────────────────────────┐
│ PropertyStore                                      │
│                                                    │
│  node_columns: HashMap<PropertyKeyId,              │
│                  PropertyColumn {                   │
│                    values: HashMap<NodeId, Value>   │
│                  }>                                 │
│                                                    │
│  edge_overflow: HashMap<EdgeId,                    │
│                   Vec<(PropertyKeyId, Value)>>      │
└────────────────────────────────────────────────────┘
```

**Column-oriented for nodes.** Each property key has its own `PropertyColumn` — a `HashMap<NodeId, Value>`. This enables efficient column scans: "find all nodes where `confidence > 0.9`" iterates only the `confidence` column, not all properties of all nodes.

**Row-oriented for edges.** Edge properties are sparse and rarely queried by column, so they use a simple `Vec<(PropertyKeyId, Value)>` per edge.

**Key operations:**
- `nodes_with_property(key)` — returns all NodeIds that have a given property (column scan).
- `nodes_where(key, predicate)` — filter nodes by property value.
- `estimate_tokens(node)` — sum character lengths of all properties / 4 (rough token approximation).

### 4.3 Traversal Algorithms

**Source:** `weav-graph/src/traversal.rs` (~1,470 lines)

Five traversal algorithms, all supporting edge and node filtering:

| Algorithm         | Purpose                                          | Complexity           |
|-------------------|--------------------------------------------------|----------------------|
| **BFS**           | Breadth-first exploration with depth/count limits | O(V + E) bounded     |
| **Flow Score**    | Propagate relevance from seeds with exponential decay | O(V + E) with pruning |
| **Ego Network**   | Extract k-hop neighborhood subgraph              | BFS(center, k)       |
| **Shortest Path** | BFS with early termination on target              | O(V + E) worst case  |
| **Scored Paths**  | Find/score paths between anchor pairs             | O(n^2 * shortest_path) |

**Edge filtering** (`EdgeFilter`):
- Label whitelist, minimum weight, max transaction age, minimum confidence, temporal validity timestamp.

**Node filtering** (`NodeFilter`):
- Label whitelist, required properties, temporal validity.
- Implemented via optional closures (`Fn(NodeId) -> Option<LabelId>`, `Fn(NodeId, &str) -> bool`) to decouple graph structure from schema knowledge.

**Flow scoring** is the primary relevance algorithm for context retrieval:

```
score(neighbor) = score(current) * alpha
```

Propagation stops when `score < theta` (minimum threshold) or `depth >= max_depth`. Scores are sorted descending with deterministic `node_id` tiebreaking.

### 4.4 Entity Deduplication

**Source:** `weav-graph/src/dedup.rs`

Three-tier matching strategy:

1. **Exact key match** — property value equality on a configured key field.
2. **Fuzzy name match** — Jaro-Winkler similarity (via `strsim` crate) above a configurable threshold (default: 0.85).
3. **Vector similarity** — accepts pre-computed similarity results to avoid circular dependency on `weav-vector`.

When a duplicate is found, `merge_properties()` applies the configured `ConflictPolicy`:

- **LastWriteWins** — new value overwrites.
- **Reject** — conflict recorded, value unchanged.
- **Merge** — new value overwrites AND conflict recorded.
- All merges return a `MergeResult` with a list of `PropertyConflict` entries for auditability.

---

## 5. Vector Index & Token Counting (weav-vector)

**Source:** `weav-vector/src/`

### 5.1 HNSW Vector Index

```rust
pub struct VectorIndex {
    inner: Index,                          // usearch HNSW index
    dimensions: u16,
    metric: DistanceMetric,
    node_to_key: HashMap<NodeId, u64>,     // NodeId -> usearch internal key
    key_to_node: HashMap<u64, NodeId>,     // Reverse mapping
    next_key: u64,
    config: VectorConfig,
}
```

**Built on [usearch](https://github.com/unum-cloud/usearch)** — a production HNSW implementation in C++ with Rust bindings.

**Configuration:**

| Parameter          | Default | Description                         |
|--------------------|---------|-------------------------------------|
| `dimensions`       | 1536    | Vector dimensionality (max 4096)    |
| `metric`           | Cosine  | Distance metric (Cosine/Euclidean/DotProduct) |
| `hnsw_m`           | 16      | HNSW connectivity parameter         |
| `hnsw_ef_construction` | 200 | Construction expansion factor       |
| `hnsw_ef_search`   | 50      | Search expansion factor             |
| `quantization`     | None    | Full f32, F16, or I8                |

**Key operations:**
- `insert(node_id, vector)` — auto-reserves capacity with doubling strategy.
- `search(query, k, ef_search)` — returns `Vec<(NodeId, f32)>` ordered by distance ascending. Supports per-query `ef_search` override.
- `search_filtered(query, k, candidates: &RoaringBitmap)` — uses usearch's native filtered search with a predicate callback over a candidate bitmap.
- `update(node_id, vector)` — remove + insert.

**Memory estimation:** `(dimensions * 4 + 64) * num_vectors` bytes (4 bytes per f32 dimension + ~64 bytes HNSW overhead per vector).

### 5.2 Token Counting

```rust
pub struct TokenCounter {
    counter_type: TokenCounterType,
}
```

| Strategy            | Description                                    |
|---------------------|------------------------------------------------|
| `CharDiv4`          | `ceil(byte_length / 4)` — fast approximation   |
| `TiktokenCl100k`    | CL100k BPE encoding (GPT-3.5/4 compatible)     |
| `TiktokenO200k`     | O200k BPE encoding (newer models)              |
| `Exact(path)`       | Custom BPE tokenizer (falls back to CharDiv4)   |

Used throughout the context query pipeline to estimate token costs for budget enforcement.

---

## 6. Query Pipeline (weav-query)

**Source:** `weav-query/src/`

The query pipeline is a four-stage process: Parse -> Plan -> Execute -> Budget Enforce.

### 6.1 Parser

**Source:** `weav-query/src/parser.rs`

Parses text commands into a `Command` enum. Supports 22+ commands:

```
Graph:    GRAPH CREATE | DROP | INFO | LIST
Node:     NODE ADD | GET | UPDATE | DELETE
Edge:     EDGE ADD | GET | DELETE | INVALIDATE
Bulk:     BULK NODES | BULK EDGES
Config:   CONFIG SET | GET
Admin:    PING | INFO | SNAPSHOT | STATS
Context:  CONTEXT (complex multi-clause query)
```

**The CONTEXT command** is the most complex, supporting extensive clause syntax:

```
CONTEXT "query_text" FROM "graph"
  [BUDGET <n> TOKENS]
  [SEEDS VECTOR [...] TOP <k>]
  [SEEDS NODES [...]]
  [DEPTH <d>]
  [DIRECTION OUT|IN|BOTH]
  [FILTER LABELS [...] MIN_WEIGHT <f> MIN_CONFIDENCE <f>]
  [DECAY EXPONENTIAL <ms> | LINEAR <ms> | STEP <ms> | NONE]
  [PROVENANCE]
  [AT <timestamp>]
  [LIMIT <n>]
  [SCORE BY <field> <ASC|DESC>]
```

**Tokenizer:** Custom tokenizer respects double-quoted strings, JSON blocks (`{...}`, `[...]`) with bracket depth tracking, and escape sequences.

**Parsed output:**

```rust
pub struct ContextQuery {
    pub query_text: Option<String>,
    pub graph: String,
    pub budget: Option<TokenBudget>,
    pub seeds: SeedStrategy,         // Vector | Nodes | Both
    pub max_depth: u8,
    pub direction: Direction,
    pub edge_filter: Option<EdgeFilterConfig>,
    pub decay: Option<DecayFunction>,
    pub include_provenance: bool,
    pub temporal_at: Option<Timestamp>,
    pub limit: Option<u32>,
    pub sort: Option<SortOrder>,     // field + direction
}
```

### 6.2 Query Planner

**Source:** `weav-query/src/planner.rs`

Converts a `ContextQuery` into an ordered `QueryPlan` of execution steps:

```
Step 1:  VectorSearch         (if seeds have embedding)
Step 2:  NodeLookup           (if seeds have node keys)
Step 3:  GraphTraversal       (always — BFS from seeds)
Step 4:  FlowScore            (always — relevance propagation, alpha=0.5, theta=0.01)
Step 4b: PathExtraction       (if 2+ node seeds — find connecting paths)
Step 5:  TemporalFilter       (if temporal_at set)
Step 6:  ConflictDetection    (always — find property conflicts within label groups)
Step 7:  RelevanceScore       (apply decay function)
Step 8:  TokenBudgetEnforce   (if budget set)
Step 9:  FormatContext        (always — final assembly)
```

**Optimization rationale:**
- Temporal filtering before decay: early exit removes irrelevant nodes before expensive decay calculations.
- Path extraction only for multi-node seeds: O(n^2) pairwise path finding is too expensive for single seeds.
- Budget enforcement near the end: maximize information available for the greedy knapsack.

### 6.3 Executor

**Source:** `weav-query/src/executor.rs`

Executes the query plan step-by-step, producing a `ContextResult`:

```rust
pub struct ContextResult {
    pub chunks: Vec<ContextChunk>,
    pub total_tokens: u32,
    pub budget_used: f32,           // 0.0 to 1.0+
    pub nodes_considered: u32,
    pub nodes_included: u32,
    pub query_time_us: u64,
    pub conflicts: Vec<ConflictInfo>,
}

pub struct ContextChunk {
    pub node_id: NodeId,
    pub content: String,            // Concatenated string properties
    pub label: String,
    pub relevance_score: f32,
    pub depth: u8,                  // BFS depth from seeds
    pub token_count: u32,
    pub provenance: Option<Provenance>,
    pub relationships: Vec<RelationshipSummary>,
    pub temporal: Option<BiTemporal>,
}
```

**Execution stages:**

1. **Seed resolution.** Vector seeds: `vector_index.search()` with distance-to-score conversion `1.0 / (1.0 + distance)`. Node seeds: `properties.nodes_where()` for entity_key lookup. Both: merge and deduplicate.

2. **Flow scoring.** Calls `weav_graph::traversal::flow_score()` to propagate relevance from seed nodes through the graph.

3. **Chunk construction.** For each scored node: fetch properties, extract label, count tokens, attach provenance and temporal metadata.

4. **Conflict detection.** Groups chunks by label, compares properties pairwise (skipping `_`-prefixed internal properties), records conflicts with resolution ("kept node X — higher relevance").

5. **Temporal filter.** Removes chunks outside the `[valid_from, valid_until)` window.

6. **Decay application.** Applies the configured `DecayFunction` to each chunk's relevance score based on node age. Removes chunks with score <= 0.

7. **Limit.** Sort by relevance descending (node_id tiebreak), truncate to limit.

8. **Budget enforcement.** Delegates to the budget module.

9. **Relationship summaries.** For each included chunk, queries `adjacency.neighbors_both()` to build `RelationshipSummary` entries.

10. **Final sort.** Applies the requested sort order (Relevance/Recency/Confidence, Asc/Desc) with node_id tiebreaking for determinism.

### 6.4 Budget Enforcer

**Source:** `weav-query/src/budget.rs`

Two enforcement strategies, both based on a **greedy knapsack** approach:

**Greedy (for `Auto` and `Priority` allocations):**

```
1. Compute value density for each chunk: relevance_score / token_count
   (zero-token chunks get density = f32::MAX — always included first)
2. Sort by density descending
3. Greedily include chunks until budget exhausted
4. Re-sort included chunks by relevance_score for output
```

**Proportional (for `Proportional` allocation):**

```
1. Categorize chunks by label:
   "relationship"/"edge"/"relation" → relationships pool
   "text"/"chunk"/"document"        → text_chunks pool
   "metadata"/"meta"/"config"       → metadata pool
   Everything else                  → entities pool
2. Allocate budget: max_tokens * category_pct for each pool
3. Apply greedy_fill() independently to each pool
4. Combine and re-sort by relevance
```

Output:

```rust
pub struct BudgetResult {
    pub included: Vec<ContextChunk>,
    pub excluded: Vec<NodeId>,
    pub total_tokens: u32,
    pub budget_utilization: f32,    // total_tokens / max_tokens
}
```

---

## 7. Persistence Layer (weav-persist)

**Source:** `weav-persist/src/`

### 7.1 Write-Ahead Log (WAL)

**Source:** `weav-persist/src/wal.rs`

**Wire format:** Length-prefixed bincode frames.

```
┌─────────────────────────────────────────────┐
│ [u32 length (LE)] [bincode WalEntry bytes]  │  ← repeating
└─────────────────────────────────────────────┘

WalEntry {
    seq: u64,              // Monotonic sequence number
    timestamp: u64,        // Ms since epoch
    shard_id: ShardId,
    operation: WalOperation,
    checksum: u32,         // CRC32 of serialized operation bytes
}
```

**10 WAL operations:**

| Operation          | Fields                                         |
|--------------------|-------------------------------------------------|
| `GraphCreate`      | name, config_json                               |
| `GraphDrop`        | name                                            |
| `NodeAdd`          | graph_id, node_id, label, properties_json, embedding, entity_key |
| `NodeUpdate`       | graph_id, node_id, properties_json              |
| `NodeDelete`       | graph_id, node_id                               |
| `EdgeAdd`          | graph_id, edge_id, source, target, label, weight |
| `EdgeInvalidate`   | graph_id, edge_id, timestamp                    |
| `EdgeDelete`       | graph_id, edge_id                               |
| `VectorUpdate`     | graph_id, node_id, vector                       |

**Checksum:** `crc32fast::hash()` — hardware-accelerated CRC32C with software fallback. Computed over serialized operation bytes only.

**Sync modes:**

| Mode           | Behavior                                          |
|----------------|---------------------------------------------------|
| `Always`       | `fsync` after every append (safest, slowest)      |
| `EverySecond`  | Background tokio task calls `sync_wal()` every 1s |
| `Never`        | No fsync (fastest, risk of data loss)              |

**Rotation:** When `current_size >= max_size` (default 256MB), the WAL file is renamed to `{path}.{timestamp}` and a fresh file is opened. Sequence numbers continue uninterrupted.

**Truncation:** `truncate_before(seq)` rewrites the file keeping only entries with `seq >= cutoff`. Used after successful snapshot to reclaim space.

### 7.2 Snapshots

**Source:** `weav-persist/src/snapshot.rs`

**Format:** Bincode-serialized `FullSnapshot` struct written to `snapshot-{timestamp}.bin`.

```rust
pub struct FullSnapshot {
    pub meta: SnapshotMeta,         // timestamp, size, counts, wal_sequence
    pub graphs: Vec<GraphSnapshot>, // graph name + config + all nodes + all edges
}
```

**Operations:**
- `save_snapshot(snapshot)` — serialize, write, `sync_all()`.
- `load_snapshot(path)` — deserialize from file.
- `list_snapshots()` — scan directory for `snapshot-*.bin`, sorted newest-first.
- `cleanup_old_snapshots(keep)` — retain N most recent, delete rest.

### 7.3 Recovery Manager

**Source:** `weav-persist/src/recovery.rs`

**Recovery procedure:**

```
1. Find latest valid snapshot (newest snapshot-*.bin)
2. Load snapshot → get wal_sequence cutoff
3. Scan data_dir for all WAL files (sorted)
4. Replay WAL entries where seq > wal_sequence_cutoff
5. Validate each entry's CRC32 checksum
6. Stop reading a WAL file on corruption (partial replay)
7. Return RecoveryResult { snapshot, wal_entries, errors }
```

The engine (`weav-server`) then applies the snapshot state and replays WAL entries to reconstruct the full in-memory state.

**Graceful degradation:** Corrupted WAL files are partially replayed up to the first corrupt entry. Corrupted snapshot files are skipped. Non-fatal errors are collected in `RecoveryResult.errors`.

---

## 8. Protocol Layer (weav-proto)

**Source:** `weav-proto/src/`

### 8.1 RESP3 Codec

**Source:** `weav-proto/src/resp3.rs`

Redis Serialization Protocol v3. Implements tokio's `Decoder`/`Encoder` traits for frame-based async I/O.

```rust
pub enum Resp3Value {
    SimpleString(String),      // +text\r\n
    BlobString(Vec<u8>),       // $<len>\r\n<data>\r\n
    SimpleError(String),       // -ERR message\r\n
    Number(i64),               // :<number>\r\n
    Double(f64),               // ,<double>\r\n
    Boolean(bool),             // #t\r\n or #f\r\n
    Null,                      // _\r\n
    Array(Vec<Resp3Value>),    // *<count>\r\n<elements...>
    Map(Vec<(Resp3Value, Resp3Value)>), // %<count>\r\n<pairs...>
    BigNumber(String),         // (<number>\r\n
}
```

**Properties:**
- Handles partial frames gracefully (returns `None` until complete frame received).
- Recursive descent parsing for nested structures.
- UTF-8 validation on string types.
- Default max frame size: 64 MB (configurable).

### 8.2 Command Mapping

**Source:** `weav-proto/src/command.rs`

- `resp3_to_command_string(frame)` — converts RESP3 Array of strings into a command string for the query parser. Handles quoting for strings with spaces/JSON.
- `context_result_to_resp3(result)` — converts `ContextResult` to nested RESP3 Map structure.
- `error_to_resp3(err)` — maps `WeavError` to RESP3 SimpleError.

### 8.3 gRPC Protocol

**Source:** `weav-proto/proto/weav.proto`

Defines `WeavService` with 18 RPC methods covering all operations. Notable:

- `ContextQueryStream` — server-side streaming for memory-efficient chunked delivery.
- Properties passed as `repeated Property { key, value_json }` for flexibility.
- Build-time proto compilation via `tonic-build`.

---

## 9. Server & Engine (weav-server)

**Source:** `weav-server/src/`

### 9.1 Engine (Central Coordinator)

**Source:** `weav-server/src/engine.rs`

```rust
pub struct Engine {
    graphs: RwLock<HashMap<String, GraphState>>,
    next_graph_id: RwLock<GraphId>,
    token_counter: TokenCounter,
    config: WeavConfig,
    wal: Option<Mutex<WriteAheadLog>>,
    snapshot_engine: Option<SnapshotEngine>,
    runtime_config: RwLock<HashMap<String, String>>,
}

pub struct GraphState {
    pub name: String,
    pub graph_id: GraphId,
    pub adjacency: AdjacencyStore,
    pub properties: PropertyStore,
    pub vector_index: VectorIndex,
    pub interner: StringInterner,
    pub config: GraphConfig,
    pub next_node_id: NodeId,
    pub next_edge_id: EdgeId,
    pub dedup_config: Option<DedupConfig>,
}
```

**Locking strategy:**
- `parking_lot::RwLock` for `graphs` — multiple concurrent readers, exclusive writer. No lock poisoning.
- `parking_lot::Mutex` for WAL — single writer for serialized append ordering.

**Node add with deduplication** (three-tier):

```
1. Entity key match (always-on if entity_key provided):
   → Exact match on entity_key property → merge with LastWriteWins

2. Fuzzy name match (if dedup_config.name_field set):
   → Jaro-Winkler similarity above threshold → merge

3. Vector similarity (if embedding provided and dedup_config.vector_threshold set):
   → HNSW search for similar vectors above threshold → merge
```

**Command routing:** `execute_command(cmd)` dispatches 22+ command variants to specialized handler methods.

### 9.2 HTTP REST API

**Source:** `weav-server/src/http.rs`

Built with **Axum 0.8**. All endpoints under `/v1/` prefix.

| Method | Path | Operation |
|--------|------|-----------|
| GET    | `/health` | Health check |
| POST   | `/v1/graphs` | Create graph |
| GET    | `/v1/graphs` | List graphs |
| GET    | `/v1/graphs/{name}` | Graph info |
| DELETE | `/v1/graphs/{name}` | Drop graph |
| POST   | `/v1/graphs/{graph}/nodes` | Add node |
| POST   | `/v1/graphs/{graph}/nodes/bulk` | Bulk add nodes |
| GET    | `/v1/graphs/{graph}/nodes/{id}` | Get node |
| PUT    | `/v1/graphs/{graph}/nodes/{id}` | Update node |
| DELETE | `/v1/graphs/{graph}/nodes/{id}` | Delete node |
| POST   | `/v1/graphs/{graph}/edges` | Add edge |
| POST   | `/v1/graphs/{graph}/edges/bulk` | Bulk add edges |
| GET    | `/v1/graphs/{graph}/edges/{id}` | Get edge |
| DELETE | `/v1/graphs/{graph}/edges/{id}` | Delete edge |
| POST   | `/v1/graphs/{graph}/edges/{id}/invalidate` | Invalidate edge |
| POST   | `/v1/context` | Context query |
| POST   | `/v1/snapshot` | Trigger snapshot |
| GET    | `/v1/info` | Server info |
| GET    | `/metrics` | Prometheus metrics |

**Error mapping:** `WeavError` variants map to HTTP status codes — `NotFound` → 404, `Duplicate`/`Conflict` → 409, parse/validation errors → 400, everything else → 500.

**Response format:** `ApiResponse<T> { success: bool, data: Option<T>, error: Option<String> }`

### 9.3 RESP3 TCP Server

**Source:** `weav-server/src/resp3_server.rs`

- Binds TCP listener on port 6380 (default).
- Spawns async task per client connection.
- Uses `tokio_util::codec::Framed<TcpStream, Resp3Codec>` for frame-based I/O.
- Pipeline: RESP3 frame → command string → parse → execute → RESP3 response.

### 9.4 gRPC Server

**Source:** `weav-server/src/grpc_server.rs`

- Tonic-based gRPC service implementing 18 RPC methods.
- `ContextQueryStream` uses MPSC channel + `ReceiverStream` for server-side streaming.
- Error mapping: `WeavError` → gRPC `Status` codes (NotFound, AlreadyExists, InvalidArgument, Internal).

### 9.5 Startup Sequence

**Source:** `weav-server/src/main.rs`

```
1. Load config (TOML + WEAV_* env overrides)
2. Create Engine (Arc<Engine>)
3. If persistence enabled:
   a. RecoveryManager::recover() — load snapshot + replay WAL
   b. Engine::recover(result) — reconstruct in-memory state
4. If WAL sync mode = EverySecond:
   → Spawn background tokio task (1s interval → engine.sync_wal())
5. Start HTTP server (axum, port 6382)
6. Start RESP3 server (TCP, port 6380)
7. Start gRPC server (tonic, port 6381)
8. tokio::select! { all three server handles }
   → Exit if any server dies
```

All three servers share the same `Arc<Engine>` instance.

### 9.6 Prometheus Metrics

**Source:** `weav-server/src/metrics.rs`

| Metric | Type | Labels |
|--------|------|--------|
| `weav_nodes_total` | IntGaugeVec | graph |
| `weav_edges_total` | IntGaugeVec | graph |
| `weav_query_duration_seconds` | HistogramVec | type |
| `weav_query_total` | IntCounterVec | type, status |
| `weav_connections_active` | IntGauge | — |

Exposed at `/metrics` endpoint in Prometheus text format.

---

## 10. CLI Client (weav-cli)

**Source:** `weav-cli/src/main.rs`

Interactive REPL client using **rustyline** for line editing and history, communicating over RESP3/TCP.

**Modes:**
- Interactive REPL: `weav-cli` — prompt `weav> `, supports history, Ctrl-C/Ctrl-D handling.
- Single command: `weav-cli -c "PING"` — execute and exit.
- Custom endpoint: `weav-cli -H <host> -p <port>`.

**Pipeline per command:**

```
User input → tokenize → build RESP3 Array → TCP send → read response → format_resp3 → display
```

Built-in commands: `quit`/`exit` (exit), `help` (show command reference).

---

## 11. Client SDKs

### 11.1 Python SDK

**Source:** `sdk/python/`

- **Clients:** `WeavClient` (synchronous, httpx) and `AsyncWeavClient` (async, httpx).
- **Target:** Python 3.10+, single dependency: `httpx >= 0.27`.
- **Types:** Python dataclasses — `ContextChunk`, `ContextResult`, `Provenance`, `RelationshipSummary`, `GraphInfo`, `NodeInfo`.
- **LLM integrations:**
  - `ContextResult.to_prompt()` — format as text block for system prompts.
  - `ContextResult.to_messages()` — OpenAI-compatible message format.
  - `context_to_anthropic_messages()` — Anthropic Messages API format.
  - `WeavLangChain` — LangChain retriever interface.
  - `WeavLlamaIndex` — LlamaIndex retriever interface.

### 11.2 Node/TypeScript SDK

**Source:** `sdk/node/`

- **Client:** `WeavClient` (async, native `fetch`).
- **Target:** TypeScript 5.0+, zero runtime dependencies.
- **Types:** TypeScript interfaces — all output types use **camelCase** (`nodeId`, `relevanceScore`, `tokenCount`).
- **Helpers:**
  - `contextToMessages(result)` — chat message format.
  - `contextToPrompt(result)` — text prompt format.

### SDK Field Naming Convention

| Concept | HTTP API (JSON) | Python SDK | Node SDK |
|---------|----------------|------------|----------|
| Node ID | `node_id` | `node_id` | `nodeId` |
| Edge ID | `edge_id` | `edge_id` | `edgeId` |
| Relevance | `relevance_score` | `relevance_score` | `relevanceScore` |
| Token count | `token_count` | `token_count` | `tokenCount` |
| Budget used | `budget_used` | `budget_used` | `budgetUtilization` |

---

## 12. Data Flow: Context Query Lifecycle

This traces a context query end-to-end:

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. CLIENT REQUEST                                                │
│    POST /v1/context                                              │
│    { graph: "knowledge", embedding: [0.1, ...], budget: 4096 }  │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────┐
│ 2. HTTP HANDLER (http.rs)                                        │
│    Parse JSON → Build ContextQuery → Engine::execute_command()   │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────┐
│ 3. QUERY PLANNER (planner.rs)                                    │
│    ContextQuery → QueryPlan {                                    │
│      VectorSearch(k=10),                                         │
│      GraphTraversal(depth=3),                                    │
│      FlowScore(alpha=0.5, theta=0.01),                          │
│      ConflictDetection,                                          │
│      RelevanceScore(decay=Exponential),                          │
│      TokenBudgetEnforce(4096),                                   │
│      FormatContext                                                │
│    }                                                             │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────┐
│ 4. EXECUTOR (executor.rs)                                        │
│                                                                  │
│  Step 1: VectorIndex.search(embedding, k=10)                    │
│          → [(node_42, 0.91), (node_17, 0.87), ...]             │
│                                                                  │
│  Step 2: flow_score(seeds, alpha=0.5, theta=0.01, depth=3)     │
│          → 150 ScoredNodes with propagated relevance            │
│                                                                  │
│  Step 3: Build ContextChunks (properties, labels, tokens)       │
│                                                                  │
│  Step 4: Detect conflicts within same-label groups              │
│                                                                  │
│  Step 5: Apply exponential decay to scores                      │
│                                                                  │
│  Step 6: Greedy knapsack budget enforcement                     │
│          → 23 chunks within 4096 tokens (92% utilized)          │
│                                                                  │
│  Step 7: Attach relationship summaries                           │
│                                                                  │
│  Step 8: Sort by relevance descending                           │
└──────────────────────────────────┬───────────────────────────────┘
                                   │
┌──────────────────────────────────▼───────────────────────────────┐
│ 5. RESPONSE                                                      │
│    ContextResult {                                               │
│      chunks: [23 ContextChunks],                                │
│      total_tokens: 3768,                                         │
│      budget_used: 0.92,                                          │
│      nodes_considered: 150,                                      │
│      nodes_included: 23,                                         │
│      query_time_us: 1247,                                        │
│      conflicts: [...]                                            │
│    }                                                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## 13. Concurrency Model

```
┌──────────────────────────────────────────────┐
│ Engine.graphs: parking_lot::RwLock            │
│                                              │
│  Read lock (concurrent):                     │
│    • GET node/edge                           │
│    • CONTEXT queries                         │
│    • GRAPH LIST / INFO                       │
│                                              │
│  Write lock (exclusive):                     │
│    • NODE ADD / UPDATE / DELETE              │
│    • EDGE ADD / INVALIDATE / DELETE          │
│    • GRAPH CREATE / DROP                     │
│                                              │
│ Engine.wal: parking_lot::Mutex               │
│    • Serialized WAL appends (one at a time)  │
│    • sync_wal() from background task         │
└──────────────────────────────────────────────┘
```

**Why `parking_lot`:**
- No lock poisoning (panics in a locked section don't permanently break the lock).
- Smaller lock objects (1 byte vs 40+ bytes for std).
- Faster uncontended acquisition.

**Async runtime:** Tokio multi-threaded runtime. All three servers (HTTP/RESP3/gRPC) share the runtime and are managed by `tokio::select!`.

---

## 14. Memory Layout & Compactness

Weav prioritizes memory efficiency for in-memory operation at scale:

| Technique | Where Used | Benefit |
|-----------|-----------|---------|
| **String Interning** | Labels, property keys | `u16` comparisons instead of string allocation/comparison |
| **CompactString** | Property values, entity keys | Inline storage for short strings (up to 24 bytes on stack) |
| **SmallVec<[T; 8]>** | Adjacency lists | No heap allocation for nodes with ≤8 edges |
| **RoaringBitmap** | Node membership | Compressed bitmap, efficient for millions of sparse IDs |
| **Column-oriented properties** | Node properties | Efficient scans by property key, sparse storage |
| **u64/u32/u16 IDs** | All identifiers | Fixed-width, cache-friendly, zero allocation |
| **Bump allocator** | Per-shard temporaries | Arena allocation for per-request scratch space |
| **HNSW quantization** | Vector index | F16 or I8 quantization reduces per-vector memory |

**Memory estimation per entity:**
- Node: ~16 bytes (bitmap) + properties + embedding (dims * 4 bytes)
- Edge: ~80 bytes (EdgeMeta) + properties + 2 adjacency entries (~32 bytes)
- Vector: `dims * 4 + 64` bytes (HNSW overhead)

---

## 15. Configuration System

**Source:** `weav-core/src/config.rs`

TOML-based with `WEAV_*` environment variable overrides:

```
┌───────────────────────────────────────────┐
│ WeavConfig                                │
│                                           │
│ ├── ServerConfig                          │
│ │   bind_address: "0.0.0.0"              │
│ │   port: 6380 (RESP3)                   │
│ │   grpc_port: 6381                      │
│ │   http_port: 6382                      │
│ │   max_connections: 10,000              │
│ │   tcp_keepalive_secs: 300              │
│ │   read_timeout_ms: 30,000              │
│ │                                         │
│ ├── EngineConfig                          │
│ │   num_shards: auto (CPU count)         │
│ │   default_vector_dimensions: 1536      │
│ │   max_vector_dimensions: 4096          │
│ │   hnsw_m: 16                           │
│ │   hnsw_ef_construction: 200            │
│ │   hnsw_ef_search: 50                   │
│ │   default_conflict_policy: LastWriteWins│
│ │   enable_temporal: true                │
│ │   enable_provenance: true              │
│ │   token_counter: CharDiv4              │
│ │                                         │
│ ├── PersistenceConfig                     │
│ │   enabled: false                       │
│ │   data_dir: "./weav-data"              │
│ │   wal_enabled: true                    │
│ │   wal_sync_mode: EverySecond           │
│ │   snapshot_interval_secs: 3600         │
│ │   max_wal_size_mb: 256                 │
│ │                                         │
│ └── MemoryConfig                          │
│     max_memory_mb: None (unlimited)      │
│     eviction_policy: NoEviction          │
│     arena_size_mb: 64                    │
└───────────────────────────────────────────┘
```

**Environment overrides:**
- `WEAV_SERVER_PORT` → RESP3 port
- `WEAV_SERVER_HTTP_PORT` → HTTP port
- `WEAV_SERVER_GRPC_PORT` → gRPC port
- `WEAV_ENGINE_NUM_SHARDS` → Shard count
- `WEAV_PERSISTENCE_ENABLED` → Enable/disable persistence
- `WEAV_PERSISTENCE_DATA_DIR` → Data directory
- `WEAV_MEMORY_MAX_MEMORY_MB` → Memory limit

**Per-graph configuration** (`GraphConfig`) allows overriding conflict policy, decay function, node/edge limits, temporal/provenance toggles, and vector dimensions.

---

## 16. Benchmarks

**Source:** `benchmarks/benches/benchmarks.rs`

Criterion-based benchmark suite with HTML report generation:

| Benchmark | Scale | What It Measures |
|-----------|-------|------------------|
| `vector_search_100k_128d_k10` | 100K vectors, 128-dim | Top-10 HNSW nearest neighbor search |
| `bfs_100kn_depth3` | 100K nodes, avg degree 5 | BFS traversal to depth 3 |
| `flow_score_100kn_depth3` | 100K nodes, avg degree 5 | Relevance flow propagation |
| `node_adjacency_10k` | 10K nodes | Node insertion throughput |
| `node_property_10k` | 10K nodes, 2 props each | Property write throughput |
| `edge_10k` | 10K edges (chain) | Edge insertion throughput |
| `token_count_{short,medium,long}` | 11 / ~900 / ~9000 bytes | Token counting throughput |
| `parse_context_query` | Full CONTEXT command | Parser throughput |
| `parse_node_add` | NODE ADD with properties | Parser throughput |
| `budget_enforce_100_chunks` | 100 chunks, 4096 budget | Greedy knapsack throughput |
| `engine_*` | 100 nodes + 99 edges | End-to-end command throughput |
| `context_query_1000n_depth3_budget4096` | 1000 nodes + 999 edges | Full context pipeline |
| `wal_append` | 100 operations | WAL write throughput |
| `snapshot_save_1000n` | 1000 nodes + 999 edges | Snapshot serialization |

Run with `cargo bench`. HTML reports generated to `target/criterion/`.
