# Weav Context Graph Database — Research & Audit Report v005

**Date:** 2026-03-08
**Scope:** Per-graph locking (major architecture change), graph config persistence
**Status:** IMPLEMENTED — 848 tests passing

---

## 1. Changes in v005

### PERF-1: Per-Graph Locking — DONE

**The single biggest performance improvement to the engine.**

**Before:** Single `RwLock<HashMap<String, GraphState>>` — every write to ANY graph blocked ALL reads and writes to ALL graphs.

**After:** `RwLock<HashMap<String, Arc<RwLock<GraphState>>>>` — outer lock held only for graph lookup (nanoseconds), inner per-graph lock held for the operation. Different graphs can be read/written concurrently with zero contention.

**Changes:**
- `Engine.graphs` field type changed
- Added `get_graph()` helper that clones the `Arc` and releases the registry lock
- All 12+ handler methods updated to use `graph_arc.read()` / `graph_arc.write()`
- Graph create/drop still use the outer write lock (necessary for registry mutation)
- Snapshot and recovery updated to work with the new structure

**Impact:** Under concurrent workload with multiple graphs, this eliminates the primary scalability bottleneck. Read operations on different graphs now run in parallel. Write operations on different graphs no longer serialize.

### WAL-4: Graph Config Persistence — DONE

**Before:** Graph configs always serialized as `"{}"` in WAL and snapshots. Recovery used default config (wrong vector dimensions, HNSW parameters, etc.).

**After:** `GraphConfig` now derives `Serialize`. Actual config is serialized in:
- WAL `GraphCreate` entries
- Snapshot `GraphSnapshot.config_json`
- Recovery properly deserializes config from WAL entries

---

## 2. Cumulative Progress (Baseline → v005)

| Category | Count | Details |
|----------|-------|---------|
| Critical bugs fixed | 5 | RoaringTreemap, WAL write-ahead, StringInterner overflow, WAL replay, snapshot embeddings |
| WAL system fixes | 6 | Write ordering, error propagation, property JSON, bulk WAL, replay, config persistence |
| Security fixes | 2 | Constant-time API key (api_key.rs + acl.rs) |
| Architecture improvements | 1 | Per-graph locking (global → per-graph RwLock) |
| New algorithms | 6 | Dijkstra, Connected Components, PPR, MMR budget, Label Propagation, RRF |
| Performance fixes | 1 | Edge pair secondary index |
| New tests | 28 | Total: 848 (up from 820 baseline) |

---

## 3. Test Count Progression

| Version | Tests | Delta | Key Changes |
|---------|-------|-------|-------------|
| Baseline | 820 | — | Original codebase |
| v001 | 834 | +14 | Dijkstra, Connected Components, PPR |
| v002 | 839 | +5 | MMR budget |
| v003 | 848 | +9 | Label Propagation, RRF |
| v004 | 848 | +0 | WAL error propagation |
| v005 | 848 | +0 | Per-graph locking, config persistence |

---

## 4. Remaining Backlog

### High Priority
| # | Item | Effort |
|---|------|--------|
| SEC-5 | Auth at graph traversal boundary (retrieval pivot attack) | Medium |
| FEAT-4 | MCP server | Medium |
| FEAT-2 | Integrate RRF into context query executor | Medium |

### Medium Priority
| # | Item | Effort |
|---|------|--------|
| ALGO-7 | Leiden community detection | High |
| ALGO-8 | Submodular facility location budget | Medium |
| FEAT-3 | Graph summarization (community → LLM) | High |
| PERF-2 | Dedup blocking index | Medium |

### Lower Priority
| # | Item | Effort |
|---|------|--------|
| SEC-3 | Auth rate limiting | Medium |
| SEC-4 | TLS support | High |
| PERF-4 | Snapshot metadata sidecar | Low |
