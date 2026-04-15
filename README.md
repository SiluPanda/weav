
<p align="center">
  <samp>,_,<br>(O,O)<br>(   )<br>" "<br><br>/-/-/-/-/-/-/-\<br>\-\-\-\-\-\-\-/<br>/-/-/-/-/-/-/-\<br><br>WEAV</samp>
</p>

<h1 align="center">weav</h1>

<p align="center">
  <strong>An in-memory context graph database built for AI.</strong>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#query-language">Query Language</a> &middot;
  <a href="#authentication--authorization">Auth</a> &middot;
  <a href="#api-reference">API Reference</a> &middot;
  <a href="#sdks">SDKs</a> &middot;
  <a href="#benchmarks">Benchmarks</a>
</p>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg">
  <img alt="Rust" src="https://img.shields.io/badge/rust-1.85%2B-orange.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-1340%20passing-brightgreen.svg">
  <img alt="Crates" src="https://img.shields.io/badge/crates-12-purple.svg">
</p>

---

## What is Weav?

Weav is a **Redis-like, in-memory context graph database** purpose-built for AI and LLM workloads. It combines graph topology, vector search, temporal tracking, and token-aware retrieval into a single system — so your AI applications can retrieve *exactly* the right context, within budget, in microseconds.

**The problem:** LLMs need context. RAG gives you chunks. But chunks lack structure — relationships, provenance, temporal validity, and relevance decay all get lost. You end up stitching together a vector DB, a graph DB, and a lot of glue code.

**Weav's answer:** One database that natively understands all of it.

### Key Capabilities

| Capability | Description |
|---|---|
| **Context Graph** | Directed, labeled, weighted graph with property storage |
| **Vector Search** | HNSW index (via usearch) with cosine, euclidean, and dot product metrics |
| **Bi-Temporal** | Track both real-world validity and transaction time for point-in-time queries |
| **Token Budgeting** | Greedy knapsack packing — fit the most relevant context into your LLM's token window |
| **Flow Scoring** | Relevance propagation from seed nodes through the graph topology |
| **Entity Dedup** | Exact key, fuzzy name (Jaro-Winkler), and vector similarity deduplication |
| **Provenance** | Track source, confidence, and extraction method for every piece of knowledge |
| **Decay Functions** | Linear, exponential, and gaussian relevance decay over time |
| **MCP Server** | Model Context Protocol integration — connect directly from Claude, Cursor, etc. |
| **Multi-Protocol** | HTTP REST, RESP3 (Redis protocol), and gRPC — all on one server |
| **Auth & ACL** | Redis-ACL-inspired auth with command categories, graph-level permissions, API keys |
| **Persistence** | WAL with CRC32 checksums + periodic snapshots with full recovery |

---

## Quickstart

### Build & Run

```bash
# Clone and build
git clone https://github.com/SiluPanda/weav.git
cd weav
cargo build --release

# Start the server
./target/release/weav-server

# Default ports:
#   RESP3  → :6380
#   gRPC   → :6381
#   HTTP   → :6382
```

#### Feature-Gated Builds

The LLM provider integration (AWS SDK, Actix) is opt-in to keep default builds lean:

```bash
# Default build — everything except LLM providers (~414 crates)
cargo build --release

# Full build — including LLM extraction (~495 crates)
cargo build --release -p weav-server --features full

# Minimal — HTTP-only graph database
cargo build --release -p weav-server --no-default-features
```

### Connect with the CLI

```bash
# Interactive REPL
./target/release/weav-cli

# Single command
./target/release/weav-cli -c 'PING'

# Connect with authentication
./target/release/weav-cli -u admin -a supersecret
```

### Your First Context Graph

```
weav> GRAPH CREATE "knowledge"
OK

weav> NODE ADD TO "knowledge" LABEL "concept" PROPERTIES {"name": "Transformers", "content": "Self-attention mechanism for sequence modeling"} EMBEDDING [0.1, 0.2, 0.3]
(integer) 0

weav> NODE ADD TO "knowledge" LABEL "concept" PROPERTIES {"name": "BERT", "content": "Bidirectional encoder from transformers"} EMBEDDING [0.12, 0.22, 0.28]
(integer) 1

weav> EDGE ADD TO "knowledge" FROM 1 TO 0 LABEL "derived_from" WEIGHT 0.95
(integer) 0

weav> CONTEXT "attention mechanisms" FROM "knowledge" BUDGET 4096 TOKENS
```

### Python SDK

```bash
pip install httpx  # dependency
```

```python
from weav import WeavClient

client = WeavClient(host="localhost", port=6382)
# or with authentication:
# client = WeavClient(host="localhost", port=6382, api_key="wk_live_abc123")
# client = WeavClient(host="localhost", port=6382, username="admin", password="secret")

# Create a graph
client.create_graph("research")

# Add nodes with embeddings
node_id = client.add_node("research",
    label="paper",
    properties={"title": "Attention Is All You Need", "year": 2017},
    embedding=[0.1, 0.2, 0.3, ...]  # your embedding vector
)

# Query context with token budget
result = client.context("research",
    query="transformer architectures",
    budget=4096,
    include_provenance=True
)

# Ready for your LLM
prompt = result.to_prompt()
messages = result.to_messages()
```

### Node.js / TypeScript SDK

```typescript
import { WeavClient } from "@weav/client";

const client = new WeavClient({ host: "localhost", port: 6382 });
// or with authentication:
// const client = new WeavClient({ host: "localhost", port: 6382, apiKey: "wk_live_abc123" });
// const client = new WeavClient({ host: "localhost", port: 6382, username: "admin", password: "secret" });

await client.createGraph("research");

const nodeId = await client.addNode("research", {
  label: "paper",
  properties: { title: "Attention Is All You Need", year: 2017 },
  embedding: [0.1, 0.2, 0.3],
});

const result = await client.context({
  graph: "research",
  query: "transformer architectures",
  budget: 4096,
});

const prompt = contextToPrompt(result);
```

---

## Architecture

```
                    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
                    │  HTTP REST  │  │  RESP3 TCP  │  │    gRPC     │
                    │   :6382     │  │   :6380     │  │   :6381     │
                    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘
                           │               │                │
                           └───────────────┼────────────────┘
                                           │
                    ┌──────────────────────────────────────────────┐
                    │              AUTH LAYER (opt-in)             │
                    │  Bearer/Basic │  AUTH cmd  │  gRPC metadata  │
                    │  → ACL Store → Category + Graph ACL check → │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────────────────────────────┐
                    │                   ENGINE                     │
                    │  ┌────────────────────────────────────────┐  │
                    │  │            Query Pipeline              │  │
                    │  │  Parse → Plan → Execute → Budget → Out│  │
                    │  └────────────────────────────────────────┘  │
                    │                                              │
                    │  ┌──────────┐ ┌──────────┐ ┌─────────────┐  │
                    │  │ Graph 0  │ │ Graph 1  │ │  Graph N    │  │
                    │  │┌────────┐│ │┌────────┐│ │┌───────────┐│  │
                    │  ││Adjacen.││ ││Adjacen.││ ││ Adjacency ││  │
                    │  │├────────┤│ │├────────┤│ │├───────────┤│  │
                    │  ││Propert.││ ││Propert.││ ││Properties ││  │
                    │  │├────────┤│ │├────────┤│ │├───────────┤│  │
                    │  ││ Vector ││ ││ Vector ││ ││  Vector   ││  │
                    │  ││ (HNSW) ││ ││ (HNSW) ││ ││  (HNSW)  ││  │
                    │  │└────────┘│ │└────────┘│ │└───────────┘│  │
                    │  └──────────┘ └──────────┘ └─────────────┘  │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────────────────────────────┐
                    │              PERSISTENCE                     │
                    │   ┌──────────────┐  ┌────────────────────┐  │
                    │   │     WAL      │  │     Snapshots      │  │
                    │   │  (CRC32)     │  │    (bincode)       │  │
                    │   └──────────────┘  └────────────────────┘  │
                    └──────────────────────────────────────────────┘
```

### Crate Map

| Crate | Purpose |
|---|---|
| `weav-core` | Foundation types, config, errors, shard infrastructure, message bus |
| `weav-graph` | Adjacency store, property store, traversal (BFS, flow scoring, Dijkstra, PPR), entity dedup |
| `weav-vector` | HNSW vector index (usearch), token counting (tiktoken-rs) |
| `weav-extract` | Ingestion pipeline: document parsing (PDF/DOCX/CSV/text), chunking, LLM extraction (opt-in) |
| `weav-query` | Query parser (38 commands), planner, executor, token budget enforcement |
| `weav-auth` | Authentication (Argon2id), API keys (SHA-256), ACL store, command classification |
| `weav-persist` | Write-ahead log, snapshot engine, crash recovery |
| `weav-proto` | RESP3 codec, gRPC protobuf definitions, command mapping |
| `weav-mcp` | Model Context Protocol server (8 tools, stdio + HTTP transports) |
| `weav-server` | Engine coordinator, HTTP/RESP3/gRPC servers (axum, tonic) |
| `weav-cli` | Interactive REPL client with history (rustyline) |
| `benchmarks` | Criterion benchmarks at 100K scale |

### Design Decisions

- **Compact String Interning** — Labels and property keys stored as `u16` IDs, not heap strings
- **Column-Oriented Properties** — Sparse property sets without wasting memory on nulls
- **SmallVec\<8\> Adjacency** — Most nodes have few edges; avoid heap allocation for the common case
- **Roaring Bitmaps** — Efficient set operations for node filtering and membership tests
- **Greedy Knapsack Budget** — Packs the highest value-density chunks (relevance / tokens) first
- **Zero-Copy Ready** — rkyv support for future hot-path serialization

---

## Query Language

Weav uses a Redis-style command language optimized for context retrieval.

### Graph Management

```
GRAPH CREATE "<name>"
GRAPH DROP "<name>"
GRAPH LIST
GRAPH INFO "<name>"
```

### Node Operations

```
NODE ADD TO "<graph>" LABEL "<label>" PROPERTIES {json} [EMBEDDING [f32, ...]]
NODE GET "<graph>" <id>
NODE GET "<graph>" BY ENTITY_KEY "<key>"
NODE UPDATE "<graph>" <id> PROPERTIES {json} [EMBEDDING [f32, ...]]
NODE DELETE "<graph>" <id>
```

### Edge Operations

```
EDGE ADD TO "<graph>" FROM <source> TO <target> LABEL "<label>" [WEIGHT <f32>]
EDGE GET "<graph>" <id>
EDGE DELETE "<graph>" <id>
EDGE INVALIDATE "<graph>" <id>
```

### Bulk Operations

```
BULK NODES TO "<graph>" DATA [{node}, {node}, ...]
BULK EDGES TO "<graph>" DATA [{edge}, {edge}, ...]
```

### Context Query

The star of the show — retrieve structured, budget-aware context for your LLM:

```
CONTEXT "<query>" FROM "<graph>" BUDGET <n> TOKENS
  [SEEDS [node_id, ...]]
  [MAX DEPTH <u8>]
  [DIRECTION IN|OUT|BOTH]
  [EDGE_FILTER {json}]
  [DECAY linear|exponential|gaussian]
  [TEMPORAL AT <timestamp>]
  [LIMIT <u32>]
  [SORT BY relevance|recency|confidence ASC|DESC]
```

**How the context pipeline works:**

```
  Query Text ──→ Vector Search ──→ Seed Nodes
                                      │
  Explicit Seeds ─────────────────────┤
                                      ▼
                              Graph Traversal
                              (BFS to max_depth)
                                      │
                                      ▼
                              Flow Scoring
                              (relevance propagation)
                                      │
                                      ▼
                             Temporal Filtering
                             (bi-temporal validity)
                                      │
                                      ▼
                             Conflict Detection
                             (label-group dedup)
                                      │
                                      ▼
                             Token Budget Enforcement
                             (greedy knapsack)
                                      │
                                      ▼
                             Sorted ContextChunks[]
```

### Server Commands

```
PING
INFO
STATS ["<graph>"]
SNAPSHOT
```

### Authentication & ACL Commands

```
AUTH <password>                              # Redis-compat single-password auth
AUTH <username> <password>                   # Username + password auth

ACL SETUSER <user> [>password] [on|off] [+@cat|-@cat] [~pattern:perm]
ACL DELUSER <username>
ACL LIST
ACL GETUSER <username>
ACL WHOAMI
ACL SAVE                                    # Persist ACL to file
ACL LOAD                                    # Reload ACL from file
```

**Command categories:** `+@connection`, `+@read`, `+@write`, `+@admin`, `+@all`

**Graph patterns:** `~*:readwrite`, `~app:*:read`, `~shared:admin`

---

## Authentication & Authorization

Weav includes a Redis-ACL-inspired auth system that works across all three protocols. **Auth is disabled by default** — zero config change needed for existing deployments.

### How It Works

| Layer | Mechanism |
|---|---|
| **HTTP** | `Authorization: Bearer <api_key>` or `Authorization: Basic <base64>` header |
| **RESP3** | `AUTH [username] password` command (per-connection identity) |
| **gRPC** | `authorization` metadata key |

### Permission Model

**Command categories** control what types of operations a user can perform:

| Category | Commands |
|---|---|
| `connection` | PING, INFO, AUTH |
| `read` | NODE.GET, EDGE.GET, GRAPH.INFO, GRAPH.LIST, STATS, CONTEXT, CONFIG.GET, ACL WHOAMI |
| `write` | NODE.ADD, NODE.UPDATE, NODE.DELETE, EDGE.ADD, EDGE.DELETE, EDGE.INVALIDATE, BULK.INSERT.* |
| `admin` | GRAPH.CREATE, GRAPH.DROP, SNAPSHOT, CONFIG.SET, ACL SETUSER/DELUSER/LIST/GETUSER/SAVE/LOAD |

**Graph-level ACL** controls which graphs a user can access, using glob patterns:

```toml
[[auth.users]]
username = "app_writer"
password = "writepass"
categories = ["+@read", "+@write"]
graph_patterns = [
  { pattern = "app:*", permission = "readwrite" },
  { pattern = "shared", permission = "read" },
]
```

### API Keys

Users can be assigned API keys (prefixed `wk_`) for Bearer token auth. The server stores only SHA-256 hashes — raw keys are never persisted.

```toml
[[auth.users]]
username = "service_account"
categories = ["+@read"]
api_keys = ["wk_live_abc123def456"]
```

### Backward Compatibility

- Auth is **OFF by default** — pass no config and everything works as before
- `require_auth = false` (the default when auth is enabled) allows mixed authenticated/unauthenticated connections during migration
- All SDK auth parameters are optional — existing client code is unchanged

---

## API Reference

### HTTP REST API

Base URL: `http://localhost:6382`

All responses follow `{ "success": bool, "data"?: T, "error"?: string }`.

#### Graphs

| Method | Endpoint | Body | Response |
|---|---|---|---|
| `POST` | `/v1/graphs` | `{ "name": "..." }` | `{ "name": "..." }` |
| `GET` | `/v1/graphs` | — | `[{ "name", "node_count", "edge_count" }]` |
| `GET` | `/v1/graphs/{name}` | — | `{ "name", "node_count", "edge_count" }` |
| `DELETE` | `/v1/graphs/{name}` | — | `"dropped"` |

#### Nodes

| Method | Endpoint | Body | Response |
|---|---|---|---|
| `POST` | `/v1/graphs/{g}/nodes` | `{ "label", "properties?", "embedding?", "entity_key?" }` | `{ "node_id": u64 }` |
| `GET` | `/v1/graphs/{g}/nodes/{id}` | — | `{ "node_id", "label", "properties" }` |
| `PUT` | `/v1/graphs/{g}/nodes/{id}` | `{ "properties?", "embedding?" }` | `"updated"` |
| `DELETE` | `/v1/graphs/{g}/nodes/{id}` | — | `"deleted"` |
| `POST` | `/v1/graphs/{g}/nodes/bulk` | `{ "nodes": [...] }` | `{ "node_ids": [u64] }` |

#### Edges

| Method | Endpoint | Body | Response |
|---|---|---|---|
| `POST` | `/v1/graphs/{g}/edges` | `{ "source", "target", "label", "weight?", "properties?" }` | `{ "edge_id": u64 }` |
| `GET` | `/v1/graphs/{g}/edges/{id}` | — | Edge details |
| `DELETE` | `/v1/graphs/{g}/edges/{id}` | — | `"deleted"` |
| `POST` | `/v1/graphs/{g}/edges/{id}/invalidate` | — | `"invalidated"` |
| `POST` | `/v1/graphs/{g}/edges/bulk` | `{ "edges": [...] }` | `{ "edge_ids": [u64] }` |

#### Context

| Method | Endpoint | Body |
|---|---|---|
| `POST` | `/v1/context` | `{ "graph", "query?", "embedding?", "seed_nodes?", "budget?", "max_depth?", "include_provenance?", "decay?", "temporal_at?", "limit?", "sort_field?", "sort_direction?", "edge_labels?", "direction?" }` |

Returns `ContextResult` with chunks, token counts, and query timing.

**Decay parameter** (object, not string):
```json
{
  "decay": {
    "decay_type": "exponential",
    "half_life_ms": 3600000,
    "max_age_ms": null,
    "cutoff_ms": null
  }
}
```
Supported types: `exponential`, `linear`, `step`, `none`.

#### Server

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/v1/info` | Server info |
| `POST` | `/v1/snapshot` | Trigger snapshot |
| `GET` | `/metrics` | Prometheus metrics |

### RESP3 Protocol

Connect on port `6380` with any Redis client or `weav-cli`. Commands are sent as RESP3 arrays.

### gRPC

Connect on port `6381`. Proto definitions in `weav-proto/proto/weav.proto`. Supports 22 RPC methods including `ContextQueryStream` for streaming results.

---

## SDKs

### Python

```python
from weav import WeavClient, AsyncWeavClient

# Sync client
client = WeavClient(host="localhost", port=6382)

# Async client
client = AsyncWeavClient(host="localhost", port=6382)

# With authentication
client = WeavClient(host="localhost", port=6382, api_key="wk_live_abc123")
client = WeavClient(host="localhost", port=6382, username="admin", password="secret")

# LLM integrations
from weav import WeavLangChain, WeavLlamaIndex
```

**Context result helpers:**
```python
result = client.context("my_graph", query="...", budget=4096)

result.to_prompt()     # Formatted string for system prompt injection
result.to_messages()   # OpenAI-compatible message list
```

**Full parameter support:**

```python
result = client.context("my_graph",
    query="transformer architectures",
    budget=4096,
    decay={"type": "exponential", "half_life_ms": 3600000},
    edge_labels=["derived_from", "related_to"],
    temporal_at=1700000000000,
    direction="outgoing",
    limit=50,
    sort_field="relevance",
    sort_direction="desc",
    include_provenance=True,
    seed_nodes=["node_key_1"],
    embedding=[0.1, 0.2, 0.3],
)
```

### Node.js / TypeScript

```typescript
import { WeavClient, contextToPrompt, contextToMessages } from "@weav/client";

const client = new WeavClient({ host: "localhost", port: 6382 });
// With authentication:
// new WeavClient({ host: "localhost", port: 6382, apiKey: "wk_live_abc123" });
// new WeavClient({ host: "localhost", port: 6382, username: "admin", password: "secret" });

const result = await client.context({
  graph: "my_graph",
  query: "...",
  budget: 4096,
  decay: { type: "exponential", halfLifeMs: 3600000 },
  edgeLabels: ["related_to", "derived_from"],
  temporalAt: Date.now(),
  direction: "outgoing",
  limit: 50,
  sortField: "relevance",
  sortDirection: "desc",
});

contextToPrompt(result);    // Formatted prompt string
contextToMessages(result);  // OpenAI-compatible messages
```

All response fields use **camelCase** (`nodeId`, `relevanceScore`, `tokenCount`).

---

## Configuration

Weav is configured via TOML file or environment variables.

```toml
# weav.toml

[server]
bind_address = "0.0.0.0"
port = 6380                    # RESP3
grpc_port = 6381               # gRPC
http_port = 6382               # HTTP REST
max_connections = 10000
tcp_keepalive_secs = 300
read_timeout_ms = 30000

[engine]
num_shards = 8                 # Defaults to CPU count
default_vector_dimensions = 1536
max_vector_dimensions = 4096
default_hnsw_m = 16
default_hnsw_ef_construction = 200
default_hnsw_ef_search = 50
default_conflict_policy = "LastWriteWins"
enable_temporal = true
enable_provenance = true
token_counter = "CharDiv4"     # or "TiktokenCl100k", "TiktokenO200k"

[persistence]
enabled = false
data_dir = "./weav-data"
wal_enabled = true
wal_sync_mode = "EverySecond"  # or "Always", "Never"
snapshot_interval_secs = 3600
max_wal_size_mb = 256

[memory]
max_memory_mb = 0              # 0 = unlimited
eviction_policy = "NoEviction"
arena_size_mb = 64

[auth]
enabled = false                # Set true to enable auth
require_auth = false           # Set true to reject unauthenticated connections
# default_password = "secret" # Redis-compat: AUTH <password> only
# acl_file = "./weav-data/acl.conf"

# [[auth.users]]
# username = "admin"
# password = "supersecret"
# categories = ["+@all"]
#
# [[auth.users]]
# username = "reader"
# password = "readonly123"
# categories = ["+@read", "+@connection"]
# graph_patterns = [{ pattern = "*", permission = "read" }]
# api_keys = ["wk_live_abc123def456"]
```

### Environment Variable Overrides

| Variable | Description |
|---|---|
| `WEAV_SERVER_PORT` | RESP3 listen port |
| `WEAV_SERVER_HTTP_PORT` | HTTP REST listen port |
| `WEAV_SERVER_GRPC_PORT` | gRPC listen port |
| `WEAV_SERVER_BIND_ADDRESS` | Bind address |
| `WEAV_ENGINE_NUM_SHARDS` | Number of shards |
| `WEAV_PERSISTENCE_ENABLED` | Enable persistence (`true`/`false`) |
| `WEAV_PERSISTENCE_DATA_DIR` | Persistence directory path |
| `WEAV_MEMORY_MAX_MEMORY_MB` | Memory limit in MB |
| `WEAV_AUTH_ENABLED` | Enable authentication (`true`/`false`) |
| `WEAV_AUTH_REQUIRE_AUTH` | Require auth for all connections (`true`/`false`) |
| `WEAV_AUTH_DEFAULT_PASSWORD` | Default password for Redis-compat `AUTH <password>` |

---

## Data Model

### Core Types

```
Node
├── node_id: u64
├── label: LabelId (interned u16)
├── properties: Map<PropertyKeyId, Value>
├── embedding: Option<Vec<f32>>
├── entity_key: Option<String>
└── temporal: BiTemporal

Edge
├── edge_id: u64
├── source: NodeId → target: NodeId
├── label: LabelId (interned u16)
├── weight: f32
├── properties: Map<PropertyKeyId, Value>
├── provenance: Option<Provenance>
└── temporal: BiTemporal

BiTemporal
├── valid_from / valid_until     ← real-world validity window
└── tx_from / tx_until           ← database transaction time

Provenance
├── source: String               ← "gpt-4-turbo", "user-input", "sec-filing-10k"
├── confidence: f32              ← 0.0 to 1.0
├── extraction_method            ← LlmExtracted | NlpPipeline | UserProvided | Derived | Imported
├── source_document_id: Option
└── source_chunk_offset: Option

Value (dynamic type system)
├── Null | Bool | Int | Float
├── String | Bytes | Timestamp
├── Vector(Vec<f32>)
├── List(Vec<Value>)
└── Map(Vec<(String, Value)>)
```

### ContextChunk (query result)

```
ContextChunk
├── node_id: u64
├── content: String              ← concatenated text properties
├── label: String
├── relevance_score: f32         ← flow scoring result
├── depth: u8                    ← hops from seed
├── token_count: u32
├── provenance: Option<Provenance>
├── relationships: Vec<RelationshipSummary>
└── temporal: Option<BiTemporal>
```

---

## Benchmarks

Run with:

```bash
cargo bench
```

| Benchmark | Scale | Description |
|---|---|---|
| `vector_search_100k_128d_k10` | 100K vectors, 128 dims | Top-10 nearest neighbor search |
| `bfs_100kn_depth3` | 100K nodes, avg degree 5 | BFS traversal to depth 3 |
| `flow_score_100kn_depth3` | 100K nodes, avg degree 5 | Relevance flow scoring |
| `node_adjacency_10k` | 10K insertions | Adjacency insert throughput |

Benchmarks produce HTML reports via [criterion](https://github.com/bheisler/criterion.rs).

---

## Testing

```bash
# Run all tests (default features)
cargo test --workspace

# Run all tests including LLM provider tests
cargo test --workspace --features weav-server/full,weav-extract/llm-providers

# Run tests for a specific crate
cargo test -p weav-core
cargo test -p weav-graph
cargo test -p weav-server

# Python SDK tests
cd sdk/python && pip install -e ".[dev]" && pytest

# Node SDK tests
cd sdk/node && npm test
```

**1,340 Rust tests** across all crates, **1,414 total** including SDKs — all passing.

| Crate | Tests |
|---|---|
| weav-core | 134 |
| weav-graph | 343 |
| weav-vector | 32 |
| weav-extract | 32 |
| weav-query | 227 |
| weav-auth | 47 |
| weav-persist | 47 |
| weav-proto | 61 |
| weav-server | 378 (282 unit + 28 integration + 68 E2E) |
| weav-cli | 39 |
| Python SDK | 49 |
| Node SDK | 25 |

---

## Project Structure

```
weav/
├── weav-core/          Core types, config, errors, shard, message bus
├── weav-graph/         Adjacency store, property store, traversal, dedup
├── weav-vector/        HNSW vector index, token counter
├── weav-extract/       Ingestion: document parsing, chunking, LLM extraction (opt-in)
├── weav-query/         Parser (38 commands), planner, executor, budget enforcer
├── weav-auth/          Authentication (Argon2id), API keys, ACL store
├── weav-persist/       WAL, snapshots, recovery manager
├── weav-proto/         RESP3 codec, gRPC proto, command mapping
├── weav-mcp/           MCP server (Model Context Protocol for LLM tools)
├── weav-server/        Engine, HTTP/RESP3/gRPC servers, binary
├── weav-cli/           Interactive REPL client
├── benchmarks/         Criterion benchmarks (100K scale)
├── sdk/
│   ├── python/         Python HTTP client + LLM integrations
│   └── node/           TypeScript HTTP client
└── Cargo.toml          Workspace root
```

---

## Dependencies

Weav is built on battle-tested Rust crates:

| Category | Crates |
|---|---|
| Async | tokio, tokio-util |
| HTTP | axum, tower |
| gRPC | tonic, prost |
| Vector Search | usearch (HNSW) |
| Tokenization | tiktoken-rs (cl100k, o200k) |
| LLM Integration | llm (opt-in via `extract-llm` feature) |
| Serialization | serde, bincode, rkyv |
| Data Structures | roaring, smallvec, compact_str |
| Hashing | xxhash-rust, crc32fast, sha2 |
| Auth | argon2 (Argon2id), rand, glob-match |
| String Matching | strsim (Jaro-Winkler) |
| CLI | clap, rustyline |
| Memory | bumpalo (arena allocator) |
| Concurrency | crossbeam, parking_lot |

---

## License

[MIT](LICENSE)
