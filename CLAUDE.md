# Weav - In-Memory Context Graph Database

Weav is a Rust in-memory context graph database designed for AI/LLM workloads. Redis-like interface, single-process, thread-per-core with keyspace sharding. Supports RESP3, gRPC, and HTTP protocols.

## Quick Reference

```bash
# Build
export PATH="$HOME/.cargo/bin:$PATH"
cargo build --workspace
cargo build --release              # Optimized (thin LTO, codegen-units=1, opt-level=3)

# Test
cargo test --workspace             # All 812 Rust tests
cargo test -p weav-core            # Single crate
cargo test -p weav-server          # Unit tests
cargo test -p weav-server --test integration  # Integration tests

# Benchmarks
cargo bench                        # Criterion benchmarks (100K scale, HTML reports)

# SDKs
cd sdk/python && pip install -e ".[dev]" && pytest
cd sdk/node && npm install && npm test
```

## Project Structure

```
weav-core/        Core types, errors, config, shard, message bus (no inter-crate deps)
weav-graph/       Adjacency store, property store, traversal algorithms, entity dedup
weav-vector/      HNSW vector index (usearch), token counting (tiktoken-rs)
weav-auth/        Authentication (Argon2id passwords, API keys), ACL store, command classification
weav-query/       Query parser (30 commands), planner, executor, budget enforcer
weav-persist/     WAL (CRC32 checksums), snapshots, recovery manager
weav-proto/       RESP3 codec (tokio-util), gRPC proto (tonic/prost), command mapping
weav-server/      Engine coordinator, HTTP (axum), RESP3 TCP, gRPC (tonic), binary
weav-cli/         Interactive REPL client (rustyline + RESP3)
benchmarks/       Criterion benchmarks
sdk/python/       Python HTTP client (httpx), LLM integrations
sdk/node/         TypeScript HTTP client (@weav/client)
```

## Crate Dependency Graph

```
weav-core (foundation, no deps)
├── weav-graph
├── weav-vector
├── weav-persist
├── weav-auth   (depends on core only)
├── weav-query (depends on graph + vector)
├── weav-proto  (depends on query)
└── weav-server (depends on all above)
    ├── weav-cli
    └── benchmarks
```

## Architecture

- **Engine** (`weav-server/src/engine.rs`): Central coordinator. Holds `HashMap<String, GraphState>` behind `RwLock`. Each `GraphState` owns an `AdjacencyStore`, `PropertyStore`, `VectorIndex`, and `StringInterner`.
- **Command pipeline**: Parse (`weav-query/src/parser.rs`) -> Plan (`planner.rs`) -> Execute (`executor.rs`) -> Budget enforce (`budget.rs`) -> `CommandResponse`
- **String interning**: Labels and property keys stored as `LabelId`/`PropertyKeyId` (u16) via `StringInterner`, not heap strings
- **SmallVec adjacency**: `SmallVec<[(NodeId, EdgeId); 8]>` avoids heap allocation for typical node degrees
- **Bi-temporal tracking**: Every entity has `valid_from/valid_until` + `tx_from/tx_until` for point-in-time queries
- **Token budget**: Greedy knapsack algorithm packs context chunks by value-density within token limits

## Key Types

```rust
// weav-core/src/types.rs
pub type NodeId = u64;
pub type EdgeId = u64;
pub type GraphId = u32;
pub type ShardId = u16;
pub type LabelId = u16;
pub type PropertyKeyId = u16;
pub type Timestamp = u64;  // ms since epoch
```

## Error Handling

All fallible operations return `WeavResult<T>` (alias for `Result<T, WeavError>`). `WeavError` is a `thiserror` enum defined in `weav-core/src/error.rs` with variants: `GraphNotFound`, `NodeNotFound`, `EdgeNotFound`, `DuplicateNode`, `Conflict`, `TokenBudgetExceeded`, `DimensionMismatch`, `ShardUnavailable`, `PersistenceError`, `ProtocolError`, `QueryParseError`, `CapacityExceeded`, `InvalidConfig`, `Internal`.

## Coding Conventions

- **Edition 2024**, minimum Rust 1.85
- **Naming**: Types `CamelCase`, functions/variables `snake_case`, constants `UPPER_SNAKE_CASE`
- **Module layout**: Each crate's `lib.rs` declares public modules only; logic lives in submodules
- **Tests**: Inline `#[cfg(test)] mod tests` blocks in each source file; integration tests in `weav-server/tests/integration.rs`
- **No cyclic dependencies** between crates
- **Concurrency**: `parking_lot::RwLock` for shared state, `std::sync::Mutex` for WAL
- **Serialization**: `serde_json` for HTTP, `bincode` for WAL/snapshots, `rkyv` for zero-copy (future)
- **Compact types**: `CompactString` for interned strings, `RoaringBitmap` for node ID sets

## Server Ports

| Protocol | Default Port |
|----------|-------------|
| RESP3    | 6380        |
| gRPC     | 6381        |
| HTTP     | 6382        |

## Configuration

TOML-based with `WEAV_*` environment variable overrides. Key defaults:
- Vector dimensions: 1536 (max 4096)
- HNSW: m=16, ef_construction=200, ef_search=50
- WAL: enabled, sync every second, max 256MB
- Snapshots: every 3600s
- Shards: auto (CPU count)

## SDK Notes

- **Python**: Sync (`WeavClient`) and async (`AsyncWeavClient`), `httpx`-based, Python 3.10+
- **Node**: `WeavClient` class, zero runtime deps, TypeScript 5.0+, all fields use camelCase (`nodeId`, `relevanceScore`, `tokenCount`)
