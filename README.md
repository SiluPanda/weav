
<p align="center">
  <samp>,_,<br>(O,O)<br>(   )<br>" "<br><br>/-/-/-/-/-/-/-\<br>\-\-\-\-\-\-\-/<br>/-/-/-/-/-/-/-\<br><br>WEAV</samp>
</p>

<h1 align="center">weav</h1>

<p align="center">
  <strong>An in-memory graph + vector database for AI retrieval.</strong>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &middot;
  <a href="#examples">Examples</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#query-language">Query Language</a> &middot;
  <a href="#authentication--authorization">Auth</a> &middot;
  <a href="#api-reference">API Reference</a> &middot;
  <a href="#mcp">MCP</a> &middot;
  <a href="#sdks">SDKs</a> &middot;
  <a href="#benchmarks">Benchmarks</a>
</p>

<p align="center">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg">
  <img alt="Rust" src="https://img.shields.io/badge/rust-1.85%2B-orange.svg">
  <img alt="Protocols" src="https://img.shields.io/badge/protocols-HTTP%20%7C%20RESP3%20%7C%20gRPC%20%7C%20MCP-brightgreen.svg">
  <img alt="Workspace" src="https://img.shields.io/badge/workspace-12%20crates-purple.svg">
</p>

---

## What is Weav?

Weav is an **in-memory graph + vector database** for AI systems that need more than chunk similarity. It stores entities, relationships, embeddings, provenance, and temporal validity in one process, then returns token-budgeted context over HTTP, RESP3, gRPC, or MCP.

Instead of stitching together a vector store, a graph database, and retrieval glue, Weav gives you one query engine for structure-aware context retrieval.

### Why Weav

- **More than vector search**: start from embeddings, then traverse relationships and score context through the graph
- **Grounded retrieval**: provenance, confidence, and bi-temporal validity stay attached to the data
- **Built for LLM windows**: token-budget-aware packing returns context that fits the model budget
- **Fast local deployment**: single-process, in-memory architecture with optional WAL and snapshots
- **Multiple ways in**: HTTP, RESP3, gRPC, CLI, and MCP all target the same engine

### Good Fit

- Agent memory and conversation state
- Knowledge graphs with embeddings
- Timeline-aware or provenance-sensitive retrieval
- Low-latency, single-node AI infrastructure

### Core Capabilities

| Area | What you get |
|---|---|
| Retrieval | HNSW vector search, graph traversal, flow scoring, BM25 text search, rerank hooks |
| Context assembly | Token budgets, provenance-aware chunks, temporal filters, optional subgraph output, LLM-ready formatting |
| Operations | WAL + snapshots, CDC event streams, schema constraints, export/import, graph algorithms |
| Interfaces | HTTP REST, RESP3, gRPC, CLI, MCP, plus Python and Node SDKs |

---

## Quickstart

### Build & Run

Rust 1.85+ is required.

```bash
# Clone
git clone https://github.com/SiluPanda/weav.git
cd weav

# Build the server and CLI used below
cargo build --release -p weav-server -p weav-cli

# Start the server
./target/release/weav-server

# Default ports:
#   RESP3  → :6380
#   gRPC   → :6381
#   HTTP   → :6382
```

If you only want the server binary, `cargo build --release` is enough. The workspace default member is `weav-server`, so `weav-cli` must be built explicitly.

#### Feature-Gated Builds

Common build targets:

```bash
# Default server build
cargo build --release

# Server + CLI (used in the quickstart above)
cargo build --release -p weav-server -p weav-cli

# Full build — including optional LLM providers
cargo build --release -p weav-server --features full

# Minimal — HTTP-only server
cargo build --release -p weav-server --no-default-features
```

### Run with Docker Compose

If you want a prewired local deployment with persistence enabled:

```bash
docker compose up --build
```

This exposes `6380` (RESP3), `6381` (gRPC), and `6382` (HTTP), and persists WAL/snapshots in the `weav-data` Docker volume.

### Smoke Test over HTTP

```bash
# Health check
curl -s http://localhost:6382/health

# Create a graph
curl -sX POST http://localhost:6382/v1/graphs \
  -H 'content-type: application/json' \
  -d '{"name":"knowledge"}'

# Add a node
curl -sX POST http://localhost:6382/v1/graphs/knowledge/nodes \
  -H 'content-type: application/json' \
  -d '{"label":"concept","properties":{"name":"Transformers","content":"Self-attention architecture for sequence modeling"}}'

# Retrieve context
curl -sX POST http://localhost:6382/v1/context \
  -H 'content-type: application/json' \
  -d '{"graph":"knowledge","query":"self attention","budget":1024}'
```

HTTP responses use the standard envelope `{ "success": bool, "data"?: ..., "error"?: ... }`.

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

### SDK Setup

```bash
python -m pip install -e ./sdk/python
cd sdk/node && npm install
```

See [SDKs](#sdks) for full Python and TypeScript examples.

---

## Examples

| Path | What it shows |
|---|---|
| [`examples/quickstart.py`](examples/quickstart.py) | End-to-end HTTP quickstart: graph creation, nodes, edges, search, algorithms, and health checks |
| [`examples/context_query.py`](examples/context_query.py) | Budget-aware retrieval, subgraph output, explain mode, and LLM-oriented formatting |
| [`examples/quickstart.ts`](examples/quickstart.ts) | Minimal Node/TypeScript workflow against the HTTP API |
| [`docker-compose.yml`](docker-compose.yml) | Local deployment with persistence enabled |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Deeper crate-by-crate architecture walkthrough |

If you want a runnable example before reading the full API surface, start with `examples/quickstart.py`.

---

## Architecture

For a crate-by-crate deep dive beyond this overview, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

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
| `weav-mcp` | Model Context Protocol server exposing graph/context tools over stdio |
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
NODE ADD TO "<graph>" LABEL "<label>" PROPERTIES {json} [EMBEDDING [f32, ...]] [ENTITY_KEY "<key>"] [TTL <ms>]
NODE GET "<graph>" <id>
NODE GET "<graph>" WHERE entity_key = "<key>"
NODE UPDATE "<graph>" <id> [PROPERTIES {json}] [EMBEDDING [f32, ...]]
NODE DELETE "<graph>" <id>
NODE MERGE "<graph>" <source_id> INTO <target_id> [POLICY keep_target|keep_source|merge]
```

### Edge Operations

```
EDGE ADD TO "<graph>" FROM <source> TO <target> LABEL "<label>" [WEIGHT <f32>] [PROPERTIES {json}] [TTL <ms>]
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
CONTEXT "<query>" FROM "<graph>" [BUDGET <n> TOKENS]
  [SEEDS VECTOR [f32, ...] TOP <k>]
  [SEEDS NODES ["entity_key", ...]]
  [DEPTH <u8>]
  [RETRIEVAL LOCAL|GLOBAL|HYBRID|DRIFT]
  [RERANK {json}]
  [DIRECTION IN|OUT|BOTH]
  [FILTER LABELS ["label", ...] MIN_WEIGHT <f> MIN_CONFIDENCE <f>]
  [DECAY EXPONENTIAL <ms> | LINEAR <ms> | STEP <ms> | NONE]
  [PROVENANCE]
  [AT <timestamp>]
  [LIMIT <u32>]
  [SCORE BY relevance|recency|confidence ASC|DESC]
```

Vector and node seeds can be combined by repeating `SEEDS` in the same command.

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

All responses follow `{ "success": bool, "data"?: T, "error"?: string }`. The `Response` column below shows the `data` payload.

#### Graphs

| Method | Endpoint | Body | Response |
|---|---|---|---|
| `POST` | `/v1/graphs` | `{ "name": "..." }` or `{ "scope": { ... } }` | empty |
| `GET` | `/v1/graphs` | — | `["graph_a", "graph_b"]` |
| `GET` | `/v1/graphs/{name}` | — | `{ "name", "node_count", "edge_count", "vector_count", "label_count", "default_ttl_ms"? }` |
| `DELETE` | `/v1/graphs/{name}` | — | empty |

#### Nodes

| Method | Endpoint | Body | Response |
|---|---|---|---|
| `POST` | `/v1/graphs/{g}/nodes` | `{ "label", "properties?", "embedding?", "entity_key?", "ttl_ms?" }` | `{ "node_id": u64 }` |
| `GET` | `/v1/graphs/{g}/nodes/{id}` | — | `{ "node_id", "label", "properties" }` |
| `PUT` | `/v1/graphs/{g}/nodes/{id}` | `{ "properties?", "embedding?" }` | empty |
| `DELETE` | `/v1/graphs/{g}/nodes/{id}` | — | empty |
| `POST` | `/v1/graphs/{g}/nodes/merge` | `{ "source_id", "target_id", "conflict_policy?" }` | `{ "node_id": u64 }` |
| `POST` | `/v1/graphs/{g}/nodes/bulk` | `{ "nodes": [...] }` | `{ "node_ids": [u64] }` |

#### Edges

| Method | Endpoint | Body | Response |
|---|---|---|---|
| `POST` | `/v1/graphs/{g}/edges` | `{ "source", "target", "label", "weight?", "properties?", "ttl_ms?" }` | `{ "edge_id": u64 }` |
| `GET` | `/v1/graphs/{g}/edges/{id}` | — | `{ "edge_id", "source", "target", "label", "weight", "properties" }` |
| `DELETE` | `/v1/graphs/{g}/edges/{id}` | — | empty |
| `POST` | `/v1/graphs/{g}/edges/{id}/invalidate` | — | empty |
| `POST` | `/v1/graphs/{g}/edges/bulk` | `{ "edges": [...] }` | `{ "edge_ids": [u64] }` |

#### Context

| Method | Endpoint | Body |
|---|---|---|
| `POST` | `/v1/context` | `{ "graph"?, "scope"?, "query"?, "retrieval_mode"?, "rerank"?, "embedding"?, "seed_nodes"?, "budget"?, "budget_preset"?, "max_depth"?, "include_provenance"?, "decay"?, "temporal_at"?, "limit"?, "sort_field"?, "sort_direction"?, "edge_labels"?, "direction"?, "explain"?, "output_format"?, "include_subgraph"? }` |

Returns `ContextResult` with chunks, token counts, and query timing.

`graph` and `scope` are mutually optional, with `graph` taking precedence when both are supplied.
`scope` resolves to canonical graph names such as `ws:acme:user:u_123`.
`budget_preset` accepts `small`/`4k`, `medium`/`8k`, `large`/`16k`, `xl`/`32k`, and `xxl`/`128k`.
Set `explain: true` to return the query plan without executing it.

**Decay parameter** (object, not string):
```json
{
  "decay": {
    "type": "exponential",
    "half_life_ms": 3600000,
    "max_age_ms": null,
    "cutoff_ms": null
  }
}
```
Supported types: `exponential`, `linear`, `step`, `none`.

**Rerank parameter**:
```json
{
  "rerank": {
    "enabled": true,
    "provider": "cross_encoder",
    "model": "bge-reranker-v2-m3",
    "candidate_limit": 50,
    "score_weight": 0.35
  }
}
```

#### Events

| Method | Endpoint | Query | Description |
|---|---|---|---|
| `GET` | `/v1/events` | `since_sequence?`, `replay_limit?` | Replay recent events across all visible graphs, then continue as SSE |
| `GET` | `/v1/graphs/{g}/events` | `since_sequence?`, `replay_limit?` | Replay recent events for one graph, then continue as SSE |

Each SSE `data:` payload is JSON shaped like:

```json
{
  "sequence": 42,
  "graph": "knowledge",
  "timestamp_ms": 1712345678901,
  "kind": "node_created",
  "payload_json": "{\"node_id\":1,\"label\":\"person\"}"
}
```

#### Server

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/v1/info` | Server info |
| `POST` | `/v1/snapshot` | Trigger snapshot |
| `GET` | `/metrics` | Prometheus metrics (when built with `observability`, enabled by default) |

### RESP3 Protocol

Connect on port `6380` with any Redis client or `weav-cli`. Commands are sent as RESP3 arrays.

### gRPC

Connect on port `6381`. Proto definitions live in `weav-proto/proto/weav.proto` and include unary graph operations plus streaming `ContextQueryStream` and `SubscribeEvents` RPCs.

---

## MCP

`weav-mcp` exposes Weav operations as Model Context Protocol tools for agent runtimes that prefer stdio transport over direct HTTP/gRPC integration.

```bash
cargo run --release -p weav-mcp
```

The MCP server starts an in-memory Weav engine with persistence disabled by default. It is a good fit when you want an MCP client to create graphs, mutate nodes and edges, run context queries, or poll recent graph events without writing a separate adapter layer.

---

## SDKs

### Python

Install from this repository:

```bash
python -m pip install -e ./sdk/python
```

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
result = client.context({"workspace_id": "acme", "user_id": "u_123"},
    query="transformer architectures",
    retrieval_mode="hybrid",
    rerank={"provider": "cross_encoder", "candidate_limit": 25, "score_weight": 0.35},
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

After adding `@weav/client` to your app:

```typescript
import { WeavClient, contextToPrompt, contextToMessages } from "@weav/client";

const client = new WeavClient({ host: "localhost", port: 6382 });
// With authentication:
// new WeavClient({ host: "localhost", port: 6382, apiKey: "wk_live_abc123" });
// new WeavClient({ host: "localhost", port: 6382, username: "admin", password: "secret" });

const result = await client.context({
  scope: { workspaceId: "acme", userId: "u_123" },
  query: "...",
  retrievalMode: "hybrid",
  rerank: { provider: "cross_encoder", candidateLimit: 25, scoreWeight: 0.35 },
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

Request parameters use **camelCase** (`seedNodes`, `retrievalMode`, `includeProvenance`), but some response objects preserve server field names such as `node_id`, `node_count`, and `edge_count`.

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

| Type | Important Fields | Notes |
|---|---|---|
| `Node` | `node_id`, `label`, `properties`, `embedding?`, `entity_key?`, `temporal` | Core entity record; labels and property keys are interned |
| `Edge` | `edge_id`, `source`, `target`, `label`, `weight`, `properties`, `provenance?`, `temporal` | Directed relationship between nodes |
| `BiTemporal` | `valid_from`, `valid_until`, `tx_from`, `tx_until` | Tracks real-world validity and transaction time |
| `Provenance` | `source`, `confidence`, `extraction_method`, `source_document_id?` | Keeps retrieval grounded in source metadata |
| `Value` | `Null`, `Bool`, `Int`, `Float`, `String`, `Bytes`, `Timestamp`, `Vector`, `List`, `Map` | Dynamic property type system |

Context queries return `ContextChunk` values with `node_id`, `content`, `label`, `relevance_score`, `depth`, `token_count`, `relationships`, and optional provenance and temporal metadata.

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
| `duplicate_suppression/*` | 500 canonical entities | Full-scan vs blocked duplicate suppression precision |
| `link_existing_precision_at_1` | 200 canonical entities | Mixed exact-key + fuzzy link-to-existing precision |
| `retrieval_lift/*` | Summary-search eval graph | Precision lift from `global` and `hybrid` retrieval |
| `rerank_lift/*` | Seeded local graph | Top-1 precision lift from cross-encoder reranking |

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

Weav has broad Rust unit, integration, and end-to-end coverage plus Python and Node SDK tests.

---

## License

[MIT](LICENSE)
