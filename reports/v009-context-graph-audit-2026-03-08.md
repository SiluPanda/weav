# Weav Context Graph Database — Research & Audit Report v009

**Date:** 2026-03-08
**Scope:** Dedup blocking index, git push v007-v008
**Status:** 883 tests passing (up from 877, +6 dedup tests)

---

## 1. Changes in v009

### PERF-2: Dedup Blocking Index — DONE

**Before:** `find_duplicate_by_name()` scanned ALL nodes with a property, computing Jaro-Winkler for each — O(N) per insertion.

**After:** `BlockingIndex` uses character trigram inverted index to narrow candidates before expensive string similarity:
1. On insert: extract trigrams ("Albert Einstein" → "alb", "lbe", "ber", "ert", ...), store in inverted index
2. On query: extract query trigrams, intersect candidate sets, compute Jaro-Winkler only on candidates sharing trigrams
3. Typical reduction: 1000 nodes → 5-20 candidates per query

**New API:** `find_duplicate_by_name_indexed(properties, name_field, name, threshold, Some(&blocking_index))` — backward compatible (pass `None` for slow path).

### Git: v007-v008 Committed and Pushed

Commit `758cbee` pushed with MCP server and snapshot metadata sidecar.

---

## 2. Final Cumulative Summary (v001-v009)

| Category | Count |
|----------|-------|
| Tests | 820 → 883 (+63) |
| Bugs fixed | 11 (5 critical + 6 WAL) |
| Security fixes | 3 |
| New algorithms | 6 (Dijkstra, CC, PPR, Label Propagation, MMR, RRF) |
| Architecture | 2 (per-graph locking, MCP server crate) |
| Performance | 3 (edge pair index, snapshot sidecar, dedup blocking index) |
| Pipeline integrations | 1 (RRF in context queries) |
| Reports | 9 versioned |

---

## 3. Remaining Backlog

| Priority | Item |
|----------|------|
| MEDIUM | Leiden community detection |
| MEDIUM | Submodular facility location budget |
| MEDIUM | Graph summarization (community → LLM) |
| MEDIUM | MCP context_query tool |
| LOW | Auth rate limiting |
| LOW | TLS support |
