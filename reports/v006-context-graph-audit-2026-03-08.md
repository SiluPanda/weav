# Weav Context Graph Database — Research & Audit Report v006

**Date:** 2026-03-08
**Scope:** RRF pipeline integration, auth traversal boundary checks
**Status:** IMPLEMENTED — 869 tests passing (up from 848, +21 auth tests)

---

## 1. Changes in v006

### FEAT-2: RRF Integrated into Context Query Pipeline — DONE

**Before:** `reciprocal_rank_fusion()` was a standalone utility only called from tests.

**After:** When a context query uses vector search (`SeedStrategy::Vector` or `Both`), the executor now automatically fuses vector similarity rankings with graph traversal rankings using RRF:

1. Vector seeds produce a ranked list by similarity score
2. Flow scoring produces a ranked list by graph traversal score
3. RRF fuses both: `RRF(d) = 1/(k + rank_vector(d)) + 1/(k + rank_graph(d))`
4. Fused scores are normalized to [0, 1] and used as relevance scores

When seeds come from node keys only (`SeedStrategy::Nodes`), flow scores pass through unchanged — backward compatible.

**Impact:** Per Practical GraphRAG research (arXiv 2507.03226), RRF hybrid retrieval improves context quality by ~15% over pure vector baselines.

### SEC-5: Auth Traversal Boundary & Per-Handler Permission Checks — DONE

**Three-tier graph permission enforcement:**
- Admin operations (GraphCreate, GraphDrop) → require Admin permission
- Write operations (NodeAdd, EdgeAdd, etc.) → require ReadWrite permission
- Read operations (NodeGet, Context, etc.) → require Read permission

**Implementation:**
- Added `check_permission()` defense-in-depth helper to Engine
- Added `_authed` wrapper methods for all 15 graph-targeting handlers
- `execute_command` dispatch routes through wrappers; original handlers used by WAL recovery (no identity)
- Fixed existing bug: Admin ops only required ReadWrite (now properly requires Admin)
- 21 new tests covering full authorization matrix (admin/reader/writer × permitted/denied × read/write/admin ops)

---

## 2. Cumulative Progress (Baseline → v006)

| Category | Count |
|----------|-------|
| Critical bugs fixed | 5 |
| WAL system fixes | 6 |
| Security fixes | 3 (constant-time API key ×2, per-handler auth) |
| Architecture improvements | 1 (per-graph locking) |
| New algorithms | 6 |
| Pipeline integrations | 1 (RRF in context queries) |
| Performance fixes | 1 |
| New tests | +49 (820 → 869) |
| Reports | 6 versioned |

---

## 3. Remaining High-Priority Backlog

| # | Item | Status |
|---|------|--------|
| SEC-5 | Auth per-handler permission checks | In progress |
| FEAT-4 | MCP server | Next priority |
| ALGO-7 | Leiden community detection | Future |
| FEAT-3 | Graph summarization | Future |
