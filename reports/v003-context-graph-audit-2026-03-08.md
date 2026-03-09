# Weav Context Graph Database — Research & Audit Report v003

**Date:** 2026-03-08
**Scope:** New algorithm implementations (Label Propagation, RRF), WAL fidelity fixes, latest landscape research
**Status:** IMPLEMENTED — 848 tests passing (up from 839 in v002, 820 baseline)

---

## 1. Changes in v003

### New Implementations

| # | Change | Files | Tests |
|---|--------|-------|-------|
| **ALGO-6** | Label Propagation community detection | `weav-graph/src/traversal.rs` | 4 new tests |
| **FEAT-1** | Reciprocal Rank Fusion (RRF) hybrid retrieval | `weav-query/src/executor.rs` | 5 new tests |
| **WAL-2** | Fixed property serialization — now uses proper JSON via serde_json instead of Debug format | `weav-server/src/engine.rs` | existing tests pass |
| **WAL-3** | Added WAL entries for bulk insert nodes | `weav-server/src/engine.rs` | existing tests pass |
| **WAL-5** | Fixed snapshot property serialization to use JSON | `weav-server/src/engine.rs` | existing tests pass |

### Algorithm Details

#### Label Propagation Community Detection
- Each node starts with its own label (= node_id)
- Iteratively, each node adopts the most frequent label among its neighbors, weighted by edge weight
- Deterministic: tie-breaking by smallest label, node order shuffled via hash(node_id ^ iteration)
- Converges when no labels change, or max_iterations reached
- Complexity: O(E) per iteration, typically 5-15 iterations
- No external dependencies (uses std::hash for deterministic shuffling instead of `rand`)

#### Reciprocal Rank Fusion (RRF)
- Formula: `RRF(d) = Σᵢ wᵢ / (k + rankᵢ(d))`
- Default k=60 (Cormack et al. 2009)
- Supports per-ranker weights for biasing graph vs vector results
- Designed for combining: vector similarity rankings, graph traversal scores, recency rankings
- Returns sorted (NodeId, rrf_score) pairs
- 15% improvement over pure vector baselines per Practical GraphRAG research

---

## 2. Cumulative Progress (Baseline → v003)

| Category | Count | Details |
|----------|-------|---------|
| Critical bugs fixed | 5 | RoaringTreemap, WAL write-ahead, StringInterner overflow, WAL replay, snapshot embeddings |
| Security fixes | 2 | Constant-time API key in api_key.rs + acl.rs |
| WAL fidelity fixes | 3 | Property JSON serialization, bulk insert WAL, snapshot properties |
| New algorithms | 6 | Dijkstra, Connected Components, PPR, MMR budget, Label Propagation, RRF |
| Performance fixes | 1 | Edge pair secondary index |
| New tests | 28 | Total: 848 (up from 820 baseline) |

---

## 3. Test Count Progression

| Version | Tests | Delta | Key Changes |
|---------|-------|-------|-------------|
| Baseline | 820 | — | Original codebase |
| v001 | 834 | +14 | Dijkstra (6), Connected Components (4), PPR (4) |
| v002 | 839 | +5 | MMR budget (5) |
| v003 | 848 | +9 | Label Propagation (4), RRF (5) |

---

## 4. New Research Findings (Jan-Mar 2026)

### Critical Security Finding: Retrieval Pivot Attacks

**Paper:** arXiv 2602.08668 (Feb 2026) — "Retrieval Pivot Attacks in Hybrid RAG"

In hybrid vector+graph pipelines, a vector-retrieved seed can pivot via entity links into sensitive graph neighborhoods, causing **cross-tenant data leakage** (RPR up to 0.95 in undefended systems). Two individually secure retrieval components compose into an insecure system.

**Mitigation:** Enforce authorization at the graph expansion boundary — not just at query entry but at **every graph traversal hop**. This is directly actionable for `weav-auth`.

### New Rust Competitors

| Project | Key Innovation |
|---------|---------------|
| **GraphRAG-rs** | 7-stage Rust pipeline with WASM support; claims 6000x token reduction vs chunk retrieval |
| **EdgeQuake** | Rust LightRAG using PostgreSQL+AGE+pgvector as single backend for graph+vector |
| **HelixDB** | Compiled query language (HelixQL → optimized Rust at compile time); claims 1000x faster than Neo4j |

### Competitor Updates

| Competitor | Key Change |
|-----------|-----------|
| **Zep/Graphiti** | MCP Server released — `add_episode`, `search_nodes`, `search_facts`, entity/edge deletion |
| **Memgraph** | MCP Server with `run_query()` tool |
| **FalkorDB** | `mem0-falkordb` plugin — sub-140ms p99 (vs Neo4j 46,900ms); GraphBLAS sparse matrix traversals |
| **Mem0** | 50K+ developers; knowledge graph linking entities across conversations in Pro tier |

### Research Insights

- **LazyGraphRAG** now in Azure — defers graph construction to query time; 700x cheaper than full GraphRAG
- **"Effective Context Window" paper** — models fail with as few as 100 tokens; "Lost in the Middle" persists at 4K tokens. Validates Weav's budget approach over naive stuffing
- **"Beyond the Context Window"** — graph-structured memory wins for stable entity attributes and relationships, not arbitrary recall. Validates Weav's entity-centric model

---

## 5. Remaining Backlog (Priority Order)

### High Priority

| # | Item | Effort | Impact |
|---|------|--------|--------|
| SEC-5 | Auth at graph traversal boundary (retrieval pivot attack mitigation) | Medium | Prevents cross-tenant data leakage in hybrid retrieval |
| PERF-1 | Per-graph locking (`RwLock<HashMap<K, Arc<RwLock<V>>>>` pattern) | Medium | Removes write serialization bottleneck |
| WAL-1 | Make `append_wal` return `WeavResult<()>`, abort mutation on failure | Medium | True write-ahead guarantee |
| WAL-4 | Persist actual graph config in WAL and snapshots | Low | Recovery uses correct dimensions |
| WAL-6 | Add WAL entries for bulk edge inserts | Low | Data durability |

### Medium Priority

| # | Item | Effort | Impact |
|---|------|--------|--------|
| ALGO-7 | Leiden community detection (hierarchical) | High | Better communities than Label Propagation |
| ALGO-8 | Submodular facility location budget (BumbleBee-style) | Medium | (1-1/e) approximation guarantee |
| FEAT-2 | Integrate RRF into context query executor | Medium | Automatic hybrid retrieval |
| FEAT-3 | Graph summarization (community → LLM summary) | High | GraphRAG-style global queries |
| PERF-2 | Dedup blocking index (n-gram or phonetic) | Medium | O(N) → O(candidates) dedup |

### Medium-High Priority

| # | Item | Effort | Impact |
|---|------|--------|--------|
| FEAT-4 | MCP server (Graphiti/Neo4j/Memgraph all have one now) | Medium | AI agent integration — competitive necessity |
| SEC-3 | Auth rate limiting | Medium | Brute-force protection |
| SEC-4 | TLS support (rustls) | High | Production security |
| PERF-4 | Snapshot metadata sidecar | Low | Faster snapshot listing |
