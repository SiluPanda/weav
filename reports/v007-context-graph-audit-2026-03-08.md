# Weav Context Graph Database — Research & Audit Report v007

**Date:** 2026-03-08
**Scope:** Snapshot metadata sidecar, v003-v006 code committed and pushed, MCP server research
**Status:** 869 tests passing

---

## 1. Changes in v007

### PERF-4: Snapshot Metadata Sidecar — DONE

**Before:** `list_snapshots()` deserialized every full snapshot file (potentially GBs) just to read metadata (created_at, node_count, etc.).

**After:** `save_snapshot()` now writes a small `.meta.json` sidecar alongside each `.bin` file. `list_snapshots()` reads the JSON sidecar first (fast path), falling back to full deserialization only for legacy snapshots without a sidecar.

**Format:** `snapshot-{timestamp}.meta.json`
```json
{
  "created_at": 1709931600000,
  "size_bytes": 1048576,
  "node_count": 5000,
  "edge_count": 12000,
  "graph_count": 3,
  "wal_sequence": 42
}
```

### Git: v003-v006 Committed and Pushed

Commit `39e5e60` pushed to `origin/main` containing all v003-v006 changes:
- Label Propagation, RRF, WAL hardening, per-graph locking, auth enforcement, config persistence

---

## 2. Cumulative Progress Summary (v001-v007)

| Category | Items |
|----------|-------|
| **Bugs fixed** | RoaringTreemap, WAL ordering, StringInterner overflow, WAL replay, snapshot embeddings |
| **WAL hardening** | Error propagation, property JSON, bulk WAL, replay completeness, config persistence, EdgeDelete type |
| **Security** | Constant-time API key (×2), three-tier auth enforcement (21 tests) |
| **Architecture** | Per-graph locking (Arc<RwLock<GraphState>>) |
| **Algorithms** | Dijkstra, Connected Components, PPR, Label Propagation, MMR budget, RRF |
| **Pipeline** | RRF integrated into context queries |
| **Performance** | Edge pair index, snapshot metadata sidecar |
| **Tests** | 820 → 869 (+49) |
| **Reports** | 7 versioned |

---

## 3. Remaining Backlog

| Priority | Item | Status |
|----------|------|--------|
| HIGH | MCP server | Research in progress |
| MEDIUM | Leiden community detection | Future |
| MEDIUM | Submodular facility location budget | Future |
| MEDIUM | Graph summarization (community → LLM) | Future |
| MEDIUM | Dedup blocking index | Future |
| LOW | Auth rate limiting | Future |
| LOW | TLS support | Future |
