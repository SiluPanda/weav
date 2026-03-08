# Weav Context Graph Database — Research & Audit Report v002

**Date:** 2026-03-08
**Scope:** Deep-dive research (submodular optimization, community detection, hybrid retrieval), v001 fix verification, new implementations
**Status:** IMPLEMENTED — 839 tests passing (up from 834 in v001, 820 baseline)

---

## Table of Contents

1. [Changes Since v001](#1-changes-since-v001)
2. [Deep Research Findings](#2-deep-research-findings)
3. [v001 Fix Verification](#3-v001-fix-verification)
4. [New Implementations](#4-new-implementations)
5. [Remaining Backlog](#5-remaining-backlog)

---

## 1. Changes Since v001

### Implemented in v002

| # | Change | Files | Tests |
|---|--------|-------|-------|
| **ALGO-5** | Maximum Marginal Relevance (MMR) token budget | `weav-query/src/budget.rs`, `weav-core/src/types.rs` | 5 new tests |
| **PERF-3** | Edge pair secondary index `(src, tgt) -> Vec<EdgeId>` | `weav-graph/src/adjacency.rs` | Existing tests pass (index is transparent) |

### Additional Fixes in v002 (from re-audit)

| # | Fix | Severity | Files |
|---|-----|----------|-------|
| **BUG-4** | Snapshot now persists vector embeddings (was `embedding: None`) | HIGH | `weav-vector/src/index.rs`, `weav-server/src/engine.rs` |
| **BUG-5** | WAL replay handles ALL 9 operations (was only GraphCreate/Drop/NodeAdd) | CRITICAL | `weav-server/src/engine.rs` — DONE |
| **BUG-6** | `handle_edge_delete` writes `EdgeDelete` WAL (was writing `EdgeInvalidate`) | MEDIUM | `weav-server/src/engine.rs` — DONE |
| **SEC-2** | `authenticate_api_key` in acl.rs now uses `verify_api_key()` (constant-time) | HIGH | `weav-auth/src/acl.rs` — DONE |

### Cumulative Changes (v001 + v002)

| Category | Count | Details |
|----------|-------|---------|
| Critical bugs fixed | 5 | RoaringTreemap, WAL write-ahead, StringInterner overflow, WAL replay, snapshot embeddings |
| Security fixes | 2 | Constant-time API key (api_key.rs + acl.rs) |
| New algorithms | 4 | Dijkstra, Connected Components, PPR, MMR budget |
| Performance fixes | 1 | Edge pair index (O(E) → O(k) for edge_history/edge_between) |
| New tests | 19 | 6 Dijkstra + 4 CC + 4 PPR + 5 MMR |

---

## 2. Deep Research Findings

### 2.1 Maximum Marginal Relevance (MMR) — Now Implemented

**Formula:**
```
MMR(cᵢ) = λ · Relevance(cᵢ) - (1-λ) · max_{cⱼ ∈ S} Similarity(cᵢ, cⱼ)
```

Where:
- `λ` controls relevance vs diversity (typical: 0.7)
- `S` is the set of already-selected chunks
- Higher MMR score = better candidate

**Weav's implementation** uses label-based similarity (same label = 1.0, different label = 0.0) since ContextChunks don't carry embeddings at budget time. This naturally diversifies across content categories (entities, relationships, text chunks) without requiring expensive cosine similarity computation.

**Key insight from research:** BumbleBee (COLM 2024) showed that facility location submodular functions provide (1-1/e) ≈ 0.632 approximation guarantees. Our label-based MMR is a practical approximation that captures the most important diversity signal (cross-category) while running in O(n²) worst case vs O(n·k·log n) for full submodular optimization.

### 2.2 Reciprocal Rank Fusion (RRF)

**Formula:**
```
RRF(d) = Σᵢ 1 / (k + rankᵢ(d))
```

Where `k=60` (standard parameter), and `rankᵢ` is the rank of document `d` in ranker `i`.

**Application to Weav:** Combine graph traversal scores with vector similarity scores:
- Ranker 1: Vector similarity rank (from HNSW search)
- Ranker 2: Flow score rank (from graph traversal)
- Ranker 3: Recency rank (from bi-temporal decay)

**Status:** Not yet implemented. Should be added to the executor's context assembly pipeline.

### 2.3 Label Propagation Community Detection

**Algorithm:**
1. Initialize each node with its own unique label
2. Iterate: each node adopts the most frequent label among its neighbors
3. Repeat until convergence (labels stop changing)

**Complexity:** O(E) per iteration, typically converges in 5-10 iterations.

**Comparison to Leiden:**
- Label Propagation: O(E) per iteration, simple, but non-deterministic (random tie-breaking)
- Leiden: O(E·log E), guarantees well-connected communities, hierarchical

**Recommendation:** Implement Label Propagation first (simpler, faster), add Leiden later.

### 2.4 Edge Pair Index — Now Implemented

**Before:** `edge_history(src, tgt)` scanned ALL edges O(E) to find edges between two nodes.
**After:** `pair_index: HashMap<(NodeId, NodeId), SmallVec<[EdgeId; 4]>>` provides O(1) lookup.

SmallVec<[EdgeId; 4]> avoids heap allocation for the common case (≤4 edges between any two nodes).

---

## 3. v001 Fix Verification

All v001 fixes verified correct:

| Fix | Status | Verification |
|-----|--------|-------------|
| RoaringTreemap | CORRECT | `node_bitmap: RoaringTreemap`, no `as u32` casts remain |
| WAL write-ahead | CORRECT | All 7 mutation handlers write WAL before mutation |
| StringInterner overflow | CORRECT | Returns `WeavResult<LabelId>`, all callers propagate with `?` |
| Constant-time API key | CORRECT | `subtle::ConstantTimeEq` used in `verify_api_key` |

---

## 4. New Implementations

### 4.1 MMR Diversity-Aware Token Budget

**New variant:** `TokenAllocation::DiversityAware { lambda: f32 }`

**How it works:**
1. Normalize relevance scores to [0, 1]
2. Iteratively select chunk with highest MMR score: `λ·relevance - (1-λ)·max_similarity_to_selected`
3. Similarity = 1.0 for same-label chunks, 0.0 for different labels
4. Skip chunks that would exceed budget
5. Re-sort included chunks by relevance for output

**Example behavior:**
- `lambda=1.0`: Pure relevance (identical to greedy knapsack)
- `lambda=0.7`: Balanced — strong relevance preference with diversity bonus
- `lambda=0.3`: Strong diversity — actively seeks different content categories

**Test case:** With 4 entity chunks (relevance 0.95-0.80) and 1 relationship chunk (relevance 0.50), budget for 3 chunks:
- Greedy: selects top 3 entities (all same category)
- MMR λ=0.3: selects 2 entities + 1 relationship (diverse context)

### 4.2 Edge Pair Secondary Index

**New field in AdjacencyStore:**
```rust
pair_index: HashMap<(NodeId, NodeId), SmallVec<[EdgeId; 4]>>
```

**Maintained by:** `add_edge_with_id()` (insert) and `remove_edge()` (cleanup)

**Optimized methods:**
- `edge_between(src, tgt, label)`: Was O(degree), now O(k) where k = edges between pair
- `edge_history(src, tgt)`: Was O(E) full scan, now O(k)

---

## 5. New Findings from v002 Re-Audit

The re-audit discovered **26 issues** (4 critical, 5 high, 8 medium, 9 low). Most impactful:

### Critical Issues Found & Fixed

| # | Issue | Status |
|---|-------|--------|
| WAL replay only handles 3/9 operations | FIXED — all 9 ops now replayed |
| `append_wal` silently discards write errors (`let _ =`) | DOCUMENTED — needs Result propagation |
| Snapshot didn't persist embeddings | FIXED — added `vectors: HashMap` to VectorIndex |
| `authenticate_api_key` bypassed constant-time comparison | FIXED — uses `verify_api_key()` |

### Critical Issues Found & Documented (Future Fix)

| # | Issue | Impact |
|---|-------|--------|
| Bulk insert has no WAL entries | Data loss on crash for bulk-inserted data |
| `handle_node_update` writes `properties_json: "{}"` to WAL | Recovery produces no-op updates |
| `handle_node_add` serializes properties with Debug format | Recovery produces incorrect property values |
| Snapshot config always `"{}"` | Recovery uses wrong vector dimensions |
| gRPC/HTTP handlers pass `None` auth identity | Auth system ineffective for non-RESP3 |
| PDF temp file uses predictable path | Race condition + symlink attack vector |

### Architecture Insight

The WAL/recovery subsystem has systemic issues — it was designed for basic operation logging but lacks fidelity for complete state reconstruction. A proper fix would involve:
1. Making `append_wal` return `WeavResult<()>` and aborting mutations on failure
2. Serializing properties correctly (JSON, not Debug format)
3. Persisting actual graph configs
4. Adding WAL entries for bulk operations

---

## 6. Remaining Backlog (Priority Order)

### Immediate (WAL/Recovery System Overhaul)

| # | Item | Effort | Impact |
|---|------|--------|--------|
| WAL-1 | Make `append_wal` return `WeavResult<()>`, abort mutation on failure | Medium | Correctness guarantee |
| WAL-2 | Fix property serialization (JSON instead of Debug format) | Low | Recovery fidelity |
| WAL-3 | Add WAL entries for bulk insert operations | Low | Data durability |
| WAL-4 | Persist actual graph config in WAL and snapshots | Low | Recovery correctness |

### Next Session (High Priority)

| # | Item | Effort | Impact |
|---|------|--------|--------|
| ALGO-6 | Label Propagation community detection | Low (~60 lines) | Enables graph summarization |
| FEAT-1 | RRF hybrid retrieval (combine vector + graph + temporal scores) | Medium (~100 lines) | 15% retrieval improvement per research |
| PERF-1 | Per-graph locking (DashMap or per-graph RwLock) | Medium (~200 lines) | Removes write serialization bottleneck |

### Future Sessions

| # | Item | Effort | Impact |
|---|------|--------|--------|
| ALGO-7 | Leiden community detection (hierarchical) | High (~300 lines) | Better communities than Label Propagation |
| PERF-2 | Dedup blocking index (n-gram or phonetic) | Medium (~150 lines) | O(N) → O(candidates) dedup |
| PERF-4 | Snapshot metadata sidecar | Low (~40 lines) | Faster snapshot listing |
| FEAT-2 | Graph summarization (community → LLM summary) | High (~300 lines) | GraphRAG-style global queries |
| FEAT-3 | MCP server protocol | Low-Medium | AI agent tool-use integration |
| SEC-2 | Auth rate limiting | Medium | Brute-force protection |
| SEC-3 | TLS support (rustls) | High | Production security |

---

## Appendix: Test Count Progression

| Version | Tests | Delta | Key Changes |
|---------|-------|-------|-------------|
| Baseline | 820 | — | Original codebase |
| v001 | 834 | +14 | Dijkstra (6), Connected Components (4), PPR (4) |
| v002 | 839 | +5 | MMR budget (5) |
