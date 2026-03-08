# Weav Feature Parity Research Plan

## Executive Summary

Weav occupies a unique niche at the intersection of **graph databases**, **vector search**, and **AI/LLM context management**. No single competitor offers the exact same combination of in-memory graph with HNSW vector search, bi-temporal data tracking, token budgeting, entity deduplication, and context assembly with relevance scoring and decay -- all in a single-process database with RESP3 + gRPC + HTTP triple-protocol support.

However, the competitive landscape is rapidly converging. Graph databases are adding vector search, vector databases are planning graph features, and a new category of AI memory layers is emerging as direct competitors. This document catalogs Weav's features, maps them against 20+ competitors, identifies parity gaps in both directions, and provides strategic recommendations.

---

## Table of Contents

1. [Weav Feature Inventory](#1-weav-feature-inventory)
2. [Competitive Landscape](#2-competitive-landscape)
3. [Feature Parity Matrix](#3-feature-parity-matrix)
4. [Gap Analysis: Where Weav Leads](#4-gap-analysis-where-weav-leads)
5. [Gap Analysis: Where Competitors Lead](#5-gap-analysis-where-competitors-lead)
6. [Closest Competitors Deep-Dive](#6-closest-competitors-deep-dive)
7. [Market Trends & Strategic Insights](#7-market-trends--strategic-insights)
8. [Recommendations](#8-recommendations)

---

## 1. Weav Feature Inventory

### 1.1 Data Model

| Feature | Details |
|---------|---------|
| **Graph Type** | Labeled property graph (nodes + directed edges) |
| **Node Properties** | Dynamic typed values: Null, Bool, Int(i64), Float(f64), String, Bytes, Vector(f32), List, Map, Timestamp |
| **Edge Properties** | Same value types + weight (f64) |
| **String Interning** | Labels and property keys stored as LabelId/PropertyKeyId (u16) via StringInterner -- max 65K labels/keys |
| **Entity Deduplication** | `entity_key` field on nodes; LastWriteWins merge on duplicate key insert |
| **Provenance Tracking** | Source, confidence (f32), extraction method (5 types), source document reference |
| **Bi-Temporal** | `valid_from/valid_until` (real-world time) + `tx_from/tx_until` (database time) on every entity |
| **Compact Storage** | SmallVec adjacency (inline for degree <= 8), CompactString, RoaringBitmap for ID sets |

### 1.2 Query Language (20+ Commands)

**Graph Management:**
- `GRAPH CREATE`, `GRAPH DROP`, `GRAPH INFO`, `GRAPH LIST`

**Node Operations:**
- `NODE ADD` (label, properties, embedding, entity_key)
- `NODE GET` (by ID or entity_key WHERE clause)
- `NODE UPDATE` (properties, embedding)
- `NODE DELETE` (cascade deletes edges)
- `BULK NODES` (JSON array batch insert)

**Edge Operations:**
- `EDGE ADD` (source, target, label, weight, properties)
- `EDGE GET`, `EDGE DELETE`, `EDGE INVALIDATE` (soft-delete via temporal)
- `BULK EDGES` (JSON array batch insert)

**Context Retrieval (flagship):**
- `CONTEXT` with 12+ clauses: query text, seed strategy (vector/node/both), budget, depth, direction, filter (labels/weight/confidence), decay function, provenance inclusion, temporal AT, limit, sort (field + direction)

**Server Control:**
- `PING`, `INFO`, `SNAPSHOT`, `STATS`, `CONFIG GET/SET`

### 1.3 Traversal Algorithms

| Algorithm | Description |
|-----------|-------------|
| **BFS** | Multi-seed, depth-limited, with edge + node filters (labels, weight, age, confidence, temporal) |
| **Flow Score** | Relevance propagation from seeds with alpha decay per hop, theta cutoff |
| **Ego Network** | Extract subgraph within radius hops of center node |
| **Shortest Path** | BFS-based shortest path between two nodes |
| **Node/Edge Filters** | Label sets, min weight, max age, min confidence, temporal validity, property existence |

### 1.4 Vector Search (HNSW)

| Parameter | Value |
|-----------|-------|
| **Library** | usearch (Rust bindings) |
| **Dimensions** | Default 1536, max 4096 |
| **Metrics** | Cosine, Euclidean, DotProduct |
| **HNSW m** | Default 16 |
| **HNSW ef_construction** | Default 200 |
| **HNSW ef_search** | Default 50 |
| **Quantization** | None (f32), F16, I8 |
| **Filtered Search** | Candidate restriction via RoaringBitmap |
| **Score Formula** | `1.0 / (1.0 + distance)` |

### 1.5 AI/LLM Features

| Feature | Details |
|---------|---------|
| **Token Budgeting** | Greedy knapsack by value-density (`relevance_score / token_count`). Allocation strategies: Auto, Proportional (per category %), Priority (ordered categories) |
| **Token Counting** | CharDiv4 (fast approximation), TiktokenCl100k (GPT-4), TiktokenO200k (GPT-4o), Custom |
| **Context Assembly** | Seeds -> flow score propagation -> chunk building -> conflict detection -> temporal filter -> decay -> budget enforcement -> relationship summaries |
| **Decay Functions** | None, Exponential (half-life), Linear (max-age), Step (cutoff), Custom (piecewise breakpoints) |
| **Conflict Detection** | Groups chunks by label, compares property values, reports conflicts with resolution (higher relevance wins) |
| **Conflict Policies** | LastWriteWins, HighestConfidence, TemporalInvalidation, Merge, Reject |

### 1.6 Persistence

| Component | Details |
|-----------|---------|
| **WAL** | Length-prefixed bincode entries, CRC32 checksums (crc32fast), rotation at max size (256MB default) |
| **WAL Sync Modes** | Always (fsync per write), EverySecond (background tokio task), Never |
| **WAL Operations** | GraphCreate/Drop, NodeAdd/Update/Delete, EdgeAdd/Invalidate/Delete, VectorUpdate |
| **Snapshots** | Full-state bincode serialization, configurable interval (default 3600s) |
| **Recovery** | Load latest snapshot + replay WAL entries |

### 1.7 Protocol Support

| Protocol | Port | Details |
|----------|------|---------|
| **RESP3** | 6380 | Redis-compatible binary protocol via tokio-util codec |
| **gRPC** | 6381 | Tonic/Prost auto-generated service |
| **HTTP** | 6382 | Axum REST API, 20+ endpoints, JSON responses |

### 1.8 Concurrency & Architecture

- Single-process, thread-per-core model
- `parking_lot::RwLock` for shared graph state (no lock poisoning)
- `parking_lot::Mutex` for WAL writes
- Configurable shard count (default = CPU count)
- Max 10,000 concurrent connections

### 1.9 SDKs

| SDK | Language | Transport | Features |
|-----|----------|-----------|----------|
| **Python** | Python 3.10+ | httpx (sync + async) | Full CRUD, context query, bulk ops, LLM integration helpers |
| **Node.js** | TypeScript 5.0+ | fetch (zero deps) | Full CRUD, context query, bulk ops, camelCase types |

### 1.10 Configuration

- TOML file + `WEAV_*` environment variable overrides
- Per-graph config (conflict policy, decay, max nodes/edges, vector dimensions)
- Memory config (max memory, eviction policy: NoEviction/LRU/RelevanceDecay)
- All HNSW parameters configurable

---

## 2. Competitive Landscape

### 2.1 Category Map

| Category | Products Analyzed |
|----------|------------------|
| **General-Purpose Graph DBs** | Neo4j, Amazon Neptune, ArangoDB, TigerGraph, Dgraph, NebulaGraph, SurrealDB, TypeDB, JanusGraph |
| **AI/LLM Memory Systems** | Mem0, Zep/Graphiti, Motorhead, LangGraph, Cognee, Amazon MemoryDB |
| **Vector Databases** | Pinecone, Weaviate, Milvus/Zilliz, Qdrant, Chroma, LanceDB |
| **In-Memory DBs** | Redis, FalkorDB, Memgraph, DragonflyDB |
| **Knowledge Graph Platforms** | TerminusDB, Stardog, Ontotext GraphDB |

### 2.2 Competitor Profiles

#### General-Purpose Graph Databases

**Neo4j** -- The market leader. Cypher query language (now ISO GQL). Vector search via Lucene HNSW. GraphRAG with built-in LLM functions. Disk-based with page cache (not truly in-memory). No native bi-temporal, no token budgeting, no RESP3. GPL/Commercial license. Largest ecosystem.

**Amazon Neptune** -- Fully managed AWS service. RDF (SPARQL) + property graph (openCypher/Gremlin). Neptune Analytics for in-memory processing (separate product). Vector search + Bedrock integration. Cloud-only, expensive. No RESP3, gRPC, or token budgeting.

**ArangoDB (Arango)** -- Multi-model (document + graph + KV). AQL query language. FAISS-based vector search with GPU acceleration (CUDA/cuGraph). HybridGraphRAG. RocksDB storage. No in-memory mode, no bi-temporal, no token budgeting. BSL 1.1 license.

**TigerGraph** -- Massive-scale parallel graph. GSQL language. TigerVector (HNSW, claims 5.2x faster than competitors). Enterprise analytics focus. Free community edition (200GB graph + 100GB vector). No token budgeting, RESP3, or context assembly.

**Dgraph** -- Native graph with DQL + GraphQL. Vector indexing with multi-embedding per node. Fully Apache 2.0 (v25). Supports gRPC. Badger KV storage. No bi-temporal, token budgeting, or RESP3.

**NebulaGraph** -- Distributed property graph. nGQL + GQL. Vector search enterprise-only (v5.1+). MCP server for AI agents. RocksDB + RAFT. No bi-temporal, token budgeting, or RESP3.

**SurrealDB** -- Multi-model (7 data models including graph). SurrealQL. HNSW vector search. **Versioned temporal tables** (partial bi-temporal). Explicitly positioning for "AI agent memory" (v3.0). Disk-based. No RESP3 or token budgeting. BSL 1.1. Raised $23M (Feb 2026).

**TypeDB** -- Hypergraph with strong type system. TypeQL. Symbolic reasoning. Rewritten in Rust (v3.0). gRPC + HTTP. **No vector search**. No token budgeting. MPL 2.0.

**JanusGraph** -- Property graph with Gremlin. Pluggable backends (Cassandra, HBase). Vector search via Elasticsearch only. Apache 2.0. Showing its age -- no AI-specific features.

#### AI/LLM Memory Systems

**Mem0** -- Hybrid (vector + graph + KV) memory layer. Automatic memory extraction from conversations. 91% lower p95 latency, 90% token reduction claims. **Not a database** -- orchestration layer over external stores (Qdrant, Neo4j, etc.). Python + Node SDKs. Apache 2.0. Raised $24M (Oct 2025). AWS Strands memory provider.

**Zep/Graphiti** -- Temporal knowledge graph engine. **True bi-temporal model** (event time + ingestion time). Automatic entity/relationship extraction. Built on Neo4j (not standalone). SOC2 + HIPAA compliant. High token overhead (~600K tokens/conversation). Python + TypeScript SDKs. Graphiti open-source (Apache 2.0), Zep Cloud proprietary.

**Motorhead** -- Session-based message store. Incremental summarization. Redis-backed with VSS. Simpler than Mem0/Zep. LangChain integration. Less actively maintained.

**LangGraph** -- Agent orchestration framework (not a database). Short-term + long-term memory with checkpoint-based time-travel. Pluggable backends. MIT license. Complementary to Weav (could use Weav as backend).

**Cognee** -- Knowledge graph construction pipeline. Graph-aware embeddings. Automatic triplet extraction. Pluggable backends (Memgraph, Neo4j, LanceDB). Python SDK. Apache 2.0. Raised EUR 7.5M (Feb 2026).

**Amazon MemoryDB** -- Redis-compatible with durable vector search. HNSW with 99%+ recall. Multi-AZ replication. No graph capabilities. Needs Neptune pairing for graph features.

#### Vector Databases

**Pinecone** -- Market-leading vector DB. Hybrid search (BM25 + semantic). Integrated inference + re-ranking. gRPC + REST. Serverless. No graph. Proprietary SaaS.

**Weaviate** -- Object-centric with cross-references (graph-like). Hybrid Search 2.0 (60% faster). Auto-vectorization. **Graph-augmented hybrid search planned for 2026**. BSD 3-Clause.

**Milvus/Zilliz** -- Industry-leading scale. GPU-accelerated (NVIDIA cuVS). HNSW, IVF, DiskANN. Full-text 7x faster than Elasticsearch. gRPC + REST. 42K+ GitHub stars. Apache 2.0. No graph.

**Qdrant** -- High-performance Rust vector DB. Dense + sparse + image search. gRPC + REST. WAL-based persistence. Qdrant Edge (on-device). Apache 2.0. No graph.

**Chroma** -- Simplest vector DB. HNSW. 4x faster after Rust rewrite (2025). Embedded or client-server. Python + JS. Apache 2.0. No graph.

**LanceDB** -- Embedded columnar vector DB. Lance format. DuckDB SQL integration. Automatic versioning. Zero-copy. Rust/Python/TS at v1.0. Apache 2.0. No graph.

#### In-Memory Databases

**Redis** -- The most architecturally similar. RESP2/RESP3 protocol. Native vector sets (April 2025). HNSW + FLAT. **RedisGraph discontinued (Jan 2025)** -- no graph support. AOF + RDB persistence. Massive ecosystem. RSALv2 + SSPLv1 license.

**FalkorDB** -- RedisGraph successor. Property graph via GraphBLAS sparse matrices. Vector similarity search. Redis module (requires Redis). Microsecond graph traversal. GraphRAG SDK. SSPL v1. **No bi-temporal, no token budgeting, no HTTP/gRPC**.

**Memgraph** -- In-memory property graph. Cypher. Vector search (2025). WAL + snapshots. Bolt protocol (Neo4j-compatible). AI Toolkit + MCP server. BSL 1.1 / $25K/yr enterprise. **Most architecturally similar general-purpose graph DB**.

**DragonflyDB** -- Redis/Memcached compatible. RESP2/RESP3. HNSW vector search. No graph. BSL 1.1.

#### Knowledge Graph Platforms

**TerminusDB** -- JSON + RDF knowledge graph. Git-for-data model (branch, merge, time-travel). Rust storage backend (v11+). No vector search. Apache 2.0.

**Stardog** -- Enterprise RDF knowledge graph. SPARQL + Gremlin. Data virtualization/federation. Voicebox conversational AI. Commercial license. Enterprise data integration focus.

**Ontotext GraphDB** -- RDF knowledge graph. Graph embedding vector search. MCP protocol support (v11.1). SPARQL. Enterprise knowledge management focus.

---

## 3. Feature Parity Matrix

### 3.1 Core Infrastructure

| Feature | Weav | Neo4j | FalkorDB | Memgraph | SurrealDB | Redis | Milvus | Qdrant |
|---------|------|-------|----------|----------|-----------|-------|--------|--------|
| In-memory | **Yes** | No | **Yes** | **Yes** | No | **Yes** | No | Partial |
| Native graph | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | No | No | No |
| Vector search (HNSW) | **Yes** | **Yes** (Lucene) | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |
| RESP3 protocol | **Yes** | No | **Yes** | No | No | **Yes** | No | No |
| gRPC | **Yes** | No | No | No | No | No | **Yes** | **Yes** |
| HTTP REST | **Yes** | **Yes** | No | **Yes** | **Yes** | No | **Yes** | **Yes** |
| WAL persistence | **Yes** | Custom | Redis AOF | **Yes** | **Yes** | AOF | Distributed | **Yes** |
| Snapshots | **Yes** | Backup | Redis RDB | **Yes** | N/A | RDB | N/A | **Yes** |
| Python SDK | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |
| Node.js SDK | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

### 3.2 AI/LLM Features

| Feature | Weav | Zep | Mem0 | FalkorDB | Neo4j | SurrealDB | Memgraph |
|---------|------|-----|------|----------|-------|-----------|----------|
| Token budgeting | **Yes** | No | Indirect | No | No | No | No |
| Context assembly | **Yes** | Partial | Partial | No | No | No | No |
| Relevance decay (5 functions) | **Yes** | No | No | No | No | No | No |
| Conflict detection | **Yes** | No | No | No | No | No | No |
| Entity deduplication | **Yes** | **Yes** | No | No | No | No | No |
| Provenance tracking | **Yes** | Partial | No | No | No | No | No |
| Multiple tokenizer support | **Yes** | No | No | No | No | No | No |
| Budget allocation strategies | **Yes** | No | No | No | No | No | No |

### 3.3 Temporal & Data Model

| Feature | Weav | Zep | SurrealDB | TerminusDB | Neo4j | FalkorDB | Memgraph |
|---------|------|-----|-----------|------------|-------|----------|----------|
| Bi-temporal (valid + tx time) | **Yes** | **Yes** | Partial | Git-model | Manual | No | No |
| Point-in-time queries | **Yes** | **Yes** | **Yes** | **Yes** | Manual | No | No |
| Temporal invalidation (soft-delete) | **Yes** | **Yes** | No | Via branches | Manual | No | No |
| String interning | **Yes** | No | No | No | No | No | No |
| Dynamic property types (10 types) | **Yes** | No | **Yes** | **Yes** | **Yes** | No | **Yes** |
| SmallVec adjacency | **Yes** | No | No | No | No | GraphBLAS | No |

### 3.4 Query & Traversal

| Feature | Weav | Neo4j | FalkorDB | Memgraph | TigerGraph | Dgraph |
|---------|------|-------|----------|----------|------------|--------|
| Custom query language | **Yes** (20+ cmds) | Cypher/GQL | Cypher | Cypher | GSQL | DQL/GraphQL |
| BFS traversal | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |
| Shortest path | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |
| Flow/relevance scoring | **Yes** | PageRank (plugin) | No | No | **Yes** | No |
| Ego network extraction | **Yes** | Via Cypher | Via Cypher | Via Cypher | Via GSQL | No |
| Filtered vector search | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |
| Multi-seed traversal | **Yes** | Via UNWIND | Partial | Partial | **Yes** | No |
| Bulk operations | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

---

## 4. Gap Analysis: Where Weav Leads

### 4.1 Unique-to-Weav Features (No Competitor Matches)

| Feature | Strategic Value | Nearest Attempt |
|---------|----------------|-----------------|
| **Native token budgeting** (greedy knapsack at DB level) | Eliminates application-layer token management; research shows 40-70% token waste in production RAG | Mem0 achieves token reduction via compression (not budgeting) |
| **5 decay functions** (Exponential, Linear, Step, Custom piecewise, None) | Fine-grained relevance aging per use case | No competitor offers configurable decay at DB level |
| **3 budget allocation strategies** (Auto, Proportional, Priority) | Category-aware context packing (entities vs relationships vs text) | No equivalent in any competitor |
| **Triple protocol** (RESP3 + gRPC + HTTP simultaneously) | Maximum client compatibility, no protocol lock-in | Dgraph has gRPC + HTTP + GraphQL (no RESP3); Redis has RESP only |
| **Integrated context assembly pipeline** (seeds -> flow score -> chunks -> conflict detect -> decay -> budget -> relationships) | Single-query context retrieval vs multi-system orchestration | Zep has partial pipeline but depends on Neo4j |
| **String interning** (LabelId/PropertyKeyId as u16) | Memory efficiency for repeated labels/keys | No competitor interns at this level |
| **Conflict detection with resolution** in context queries | Surfaces contradictory facts to LLMs | No competitor does this at query time |
| **Multiple tokenizer support** (CharDiv4, cl100k, o200k, Custom) | Accurate token counting for any LLM model | No competitor offers model-specific token counting in-database |

### 4.2 Strong Advantages (Few Competitors Match)

| Feature | Weav | Who Else Has It |
|---------|------|-----------------|
| In-memory graph + vector in single process | Yes | FalkorDB (Redis module), Memgraph |
| Bi-temporal entity tracking | Yes | Zep/Graphiti only |
| Entity deduplication with merge | Yes | Zep (extraction-based) |
| Provenance with 5 extraction methods | Yes | Zep (partial) |
| RESP3 + graph capabilities | Yes | FalkorDB only |
| Configurable conflict policies (5 types) | Yes | No one at this level |

---

## 5. Gap Analysis: Where Competitors Lead

### 5.1 Critical Gaps (High Priority)

| Feature | Who Has It | Impact | Complexity |
|---------|-----------|--------|------------|
| **Cypher/GQL query language** | Neo4j, FalkorDB, Memgraph | Industry standard, massive developer familiarity, ISO standardization | High -- full parser + optimizer |
| **ACID transactions** | Neo4j, Memgraph, SurrealDB | Required for write-heavy production workloads | High -- multi-statement tx manager |
| **Horizontal scaling / distribution** | Neo4j, TigerGraph, NebulaGraph, Dgraph | Needed for datasets exceeding single-machine memory | Very High -- consensus, replication, routing |
| **GraphRAG pipeline** | Neo4j, FalkorDB, ArangoDB, Memgraph | Trending paradigm; competitors ship SDKs for it | Medium -- SDK-level integration |
| **Automatic entity/relationship extraction** | Mem0, Zep/Graphiti, Cognee | Key for "just add conversations" UX | Medium -- LLM integration layer |
| **Full-text search** | Neo4j, Milvus, Weaviate, SurrealDB, Qdrant | Common requirement for hybrid retrieval | Medium -- requires index (tantivy/Lucene) |

### 5.2 Important Gaps (Medium Priority)

| Feature | Who Has It | Impact | Complexity |
|---------|-----------|--------|------------|
| **Graph algorithms library** (PageRank, community detection, centrality) | Neo4j (GDS), TigerGraph, Memgraph (MAGE) | Analytics use cases, importance scoring | Medium -- per-algorithm implementation |
| **GPU-accelerated vector search** | ArangoDB (cuGraph), Milvus (cuVS), TigerGraph | 10-100x speedup for large-scale vector ops | Medium -- CUDA/cuVS integration |
| **WebSocket / real-time subscriptions** | SurrealDB, Dgraph, NebulaGraph | Live query results, reactive applications | Medium -- pub/sub + change streams |
| **Multi-tenancy** | FalkorDB, Pinecone, SurrealDB | SaaS deployment model | Low-Medium -- namespace isolation |
| **Schema validation / type system** | TypeDB, SurrealDB, Neo4j | Data integrity for production workloads | Medium |
| **Authentication / authorization** | Neo4j, Memgraph, SurrealDB, most enterprise DBs | Production security requirement | Medium |
| **MCP server protocol** | Memgraph, NebulaGraph, Ontotext GraphDB | AI agent tool-use integration standard | Low -- protocol adapter |
| **Automatic embedding generation** | Weaviate, Pinecone, Chroma | Removes need for external embedding calls | Low-Medium -- model provider integration |

### 5.3 Nice-to-Have Gaps (Low Priority)

| Feature | Who Has It | Notes |
|---------|-----------|-------|
| **GraphQL interface** | Dgraph, SurrealDB, Weaviate | Developer convenience |
| **Data virtualization / federation** | Stardog, ArangoDB | Enterprise data integration |
| **Stored procedures / UDFs** | Neo4j, TigerGraph, Memgraph | Custom in-database logic |
| **Visualization tools** | Neo4j Browser, TigerGraph Studio, TerminusDB | Developer/analyst tooling |
| **Cloud-managed offering** | Most enterprise competitors | Reduces operational burden |
| **Sparse vector support** | Qdrant, Milvus, Pinecone | BM25-style sparse retrieval |

---

## 6. Closest Competitors Deep-Dive

### 6.1 FalkorDB -- Rank #1 Threat

**What it is:** RedisGraph successor. Property graph via GraphBLAS sparse matrix algebra. Redis module with vector search. SSPL v1 license.

**Feature overlap with Weav:**
- In-memory graph + vector search + Redis protocol
- Microsecond graph traversal (GraphBLAS optimized)
- GraphRAG SDK (ahead of Weav here)

**Weav's advantages over FalkorDB:**
- Bi-temporal data model (FalkorDB: none)
- Token budgeting + context assembly (FalkorDB: none)
- Relevance decay functions (FalkorDB: none)
- gRPC + HTTP protocols (FalkorDB: Redis-only)
- Entity deduplication (FalkorDB: none)
- Conflict detection (FalkorDB: none)
- Provenance tracking (FalkorDB: none)
- Standalone binary (FalkorDB: requires Redis server)

**FalkorDB's advantages over Weav:**
- Cypher query language (industry standard)
- GraphBLAS-optimized matrix operations (faster for graph algorithms)
- GraphRAG SDK with dedicated tooling
- Larger community (RedisGraph lineage)
- More mature graph algorithm library

**Strategic response:** Weav should emphasize the AI-native features (token budgeting, context assembly, decay, bi-temporal) that FalkorDB lacks entirely. Consider Cypher compatibility as a long-term roadmap item.

---

### 6.2 Zep/Graphiti -- Rank #2 Threat

**What it is:** Temporal knowledge graph engine for AI memory. Built on Neo4j. Graphiti is open-source (Apache 2.0); Zep Cloud is commercial.

**Feature overlap with Weav:**
- Bi-temporal model (event time + ingestion time)
- Entity deduplication / extraction
- Temporal context management
- Python + TypeScript SDKs

**Weav's advantages over Zep:**
- Standalone database (Zep depends on Neo4j)
- Token budgeting (Zep: none)
- RESP3 + gRPC protocols (Zep: REST only)
- In-memory performance (Zep: Neo4j disk-based)
- Configurable decay functions (Zep: none)
- Conflict detection with resolution (Zep: none)
- Lower token overhead (~7K/conversation potential vs Zep's ~600K)

**Zep's advantages over Weav:**
- Automatic entity/relationship extraction from conversations
- SOC2 Type 2 + HIPAA compliance
- Production-proven with enterprise customers
- 18.5% accuracy improvement (published benchmark)
- More mature temporal reasoning
- No manual entity_key required -- extracts entities automatically

**Strategic response:** The automatic extraction capability is Zep's killer feature. Weav should consider an LLM integration layer for automatic entity extraction (could be SDK-level) to compete here. The compliance certifications are important for enterprise sales.

---

### 6.3 Memgraph -- Rank #3 Threat

**What it is:** In-memory property graph database. Cypher. WAL + snapshots. Vector search (2025). AI Toolkit + MCP server.

**Feature overlap with Weav:**
- In-memory graph database
- Vector search (HNSW)
- WAL + snapshot persistence
- Multiple SDKs

**Weav's advantages over Memgraph:**
- Token budgeting + context assembly (Memgraph: none)
- RESP3 protocol (Memgraph: Bolt only)
- Bi-temporal (Memgraph: none)
- Relevance decay (Memgraph: none)
- Entity deduplication (Memgraph: none)
- Free (Memgraph Enterprise: $25K/year)

**Memgraph's advantages over Weav:**
- Cypher query language (huge ecosystem)
- Graph algorithm library (MAGE: 30+ algorithms)
- ACID transactions
- MCP server for AI agents
- Bolt protocol (Neo4j compatibility)
- Schema constraints and triggers
- Built-in streaming integration (Kafka, Pulsar)

**Strategic response:** Memgraph's broader feature set makes it better for general graph workloads, but Weav's AI-native features (token budgeting, context assembly, decay) make it purpose-built for LLM context management. Different target audiences.

---

### 6.4 SurrealDB -- Rank #4 Threat

**What it is:** Multi-model database (7 data models). SurrealQL. HNSW vector search. Versioned temporal tables. Explicitly positioning for "AI agent memory" (v3.0). Raised $23M (Feb 2026).

**Feature overlap with Weav:**
- Graph capabilities + vector search
- Partial temporal support (versioned tables)
- AI agent memory positioning

**Weav's advantages over SurrealDB:**
- In-memory (SurrealDB: disk-based)
- True bi-temporal (SurrealDB: versioned only)
- Token budgeting + context assembly (SurrealDB: none)
- RESP3 protocol (SurrealDB: none)
- Relevance decay (SurrealDB: none)
- Purpose-built simplicity (SurrealDB: 7 data models = complexity)

**SurrealDB's advantages over Weav:**
- 7 data models (document, graph, KV, time-series, vector, geospatial, full-text)
- SurrealQL (full SQL-like language with graph extensions)
- WebSocket real-time queries
- ACID transactions
- Schema enforcement
- Authentication / authorization built-in
- $23M funding, larger team
- WASM extension system for in-database model execution

**Strategic response:** SurrealDB is broad where Weav is deep. Weav should lean into specialization -- "the best at AI context management" vs SurrealDB's "does everything adequately."

---

### 6.5 Mem0 -- Rank #5 Threat

**What it is:** AI memory layer. Hybrid (vector + graph + KV) over external databases. Apache 2.0. Raised $24M. AWS Strands memory provider.

**Feature overlap with Weav:**
- AI/LLM memory management
- Context optimization (90% token reduction claim)
- Python + Node.js SDKs

**Weav's advantages over Mem0:**
- Standalone database (Mem0: orchestration layer over Qdrant/Neo4j/etc.)
- Native graph with traversal algorithms (Mem0: limited graph)
- Token budgeting at DB level (Mem0: application-level compression)
- Bi-temporal (Mem0: none)
- RESP3 + gRPC protocols (Mem0: REST only)
- Single-process deployment (Mem0: requires multiple backend services)

**Mem0's advantages over Weav:**
- Automatic memory extraction from conversations
- Hierarchical memory (user/session/agent levels)
- Published benchmarks (26% accuracy uplift over OpenAI memory)
- AWS partnership (Strands)
- Larger community and ecosystem
- Simpler developer experience ("just add conversations")
- $24M funding

**Strategic response:** Mem0 validates the market but is fundamentally middleware. Weav should position as the database layer that tools like Mem0 *should be* built on. Consider offering a "Mem0-compatible" API or integration.

---

## 7. Market Trends & Strategic Insights

### 7.1 The AI Memory Market is Exploding

Recent funding rounds validate this market:
- **Mem0**: $24M (Oct 2025)
- **SurrealDB**: $23M (Feb 2026) -- explicitly pivoting to AI agent memory
- **Cognee**: EUR 7.5M (Feb 2026)
- **Zep**: Growing rapidly with enterprise customers

This confirms that Weav's positioning is aligned with a well-funded, growing market.

### 7.2 Convergence is Accelerating

Every database category is moving toward Weav's territory:
- **Graph DBs** adding vector search (Neo4j, TigerGraph, NebulaGraph in 2024-2025)
- **Vector DBs** planning graph features (Weaviate 2026 roadmap)
- **In-memory DBs** adding AI features (Redis vector sets April 2025, DragonflyDB Search 2025)
- **AI memory layers** adding graph storage (Mem0, Cognee 2025-2026)
- **Multi-model DBs** targeting AI agents (SurrealDB 3.0)

**Implication:** Weav's window of unique differentiation is narrowing. Speed to market with the integrated product matters.

### 7.3 Token Budgeting is an Underserved Niche

**No competitor offers native token budgeting at the database level.** This is Weav's clearest differentiator. Research papers confirm this is a real problem solved at the application layer, not the data layer. Production RAG systems waste 40-70% of available tokens due to poor context serialization.

### 7.4 RESP3 + Graph is Extremely Rare

Only FalkorDB (as a Redis module) offers graph capabilities over a Redis-compatible protocol. Weav is the only standalone database combining RESP3 + graph + vector + AI context features. This is a strong moat for teams already invested in Redis tooling.

### 7.5 Licensing Favors Open Source

Most competitors have moved to restrictive licenses (BSL 1.1, SSPLv1, RSALv2). Truly open-source Apache 2.0 options include Mem0, Qdrant, Milvus, LanceDB, Dgraph v25. Weav's licensing decision is strategically important -- an Apache 2.0 license would differentiate against Redis (RSALv2), FalkorDB (SSPL), Memgraph (BSL), and SurrealDB (BSL).

---

## 8. Recommendations

### 8.1 Positioning

**Primary claim:** "The first purpose-built database for AI context management"

| Against | Messaging |
|---------|-----------|
| **Graph DBs** (Neo4j, etc.) | "In-memory speed + token budgeting + context assembly. Your graph DB doesn't understand LLM tokens." |
| **AI memory layers** (Mem0, Zep) | "A real database, not middleware. Native persistence, RESP3 protocol, no external dependencies." |
| **Vector DBs** (Pinecone, Qdrant) | "Graph traversal for multi-hop reasoning. Entity dedup. Bi-temporal tracking. Vectors alone aren't enough." |
| **Redis** | "Redis for AI Knowledge Graphs. Same RESP3 protocol, plus native graph + vector + temporal." |

### 8.2 Priority Feature Roadmap (by competitive impact)

**Tier 1 -- Competitive Necessities (close critical gaps):**

| Feature | Rationale | Effort |
|---------|-----------|--------|
| Authentication / authorization | Production deployment blocker; every competitor has it | Medium |
| MCP server protocol | AI agent integration standard; Memgraph, NebulaGraph, GraphDB have it | Low |
| Full-text search (tantivy) | Required for hybrid retrieval; most competitors have it | Medium |
| GraphRAG SDK/integration | Trending paradigm; FalkorDB, Neo4j, ArangoDB ship it | Medium |

**Tier 2 -- Competitive Differentiators (extend lead):**

| Feature | Rationale | Effort |
|---------|-----------|--------|
| Automatic entity extraction (LLM-powered) | Zep/Mem0's killer feature; SDK-level integration | Medium |
| Streaming/real-time subscriptions | SurrealDB, Dgraph have it; needed for reactive agents | Medium |
| Graph algorithm library (PageRank, community detection) | Neo4j GDS, Memgraph MAGE; useful for importance scoring | Medium |
| Automatic embedding generation | Weaviate, Pinecone, Chroma; removes external API dependency | Low-Medium |

**Tier 3 -- Strategic Investments (long-term moat):**

| Feature | Rationale | Effort |
|---------|-----------|--------|
| Cypher/GQL compatibility layer | Industry standard; massive developer base | High |
| ACID transactions | Production write workloads; Neo4j, Memgraph, SurrealDB | High |
| Horizontal scaling | Enterprise scale; most graph DBs have this | Very High |
| Cloud-managed offering | Reduces operational burden; table-stakes for SaaS | Very High |

### 8.3 Recommended Strategic Actions

1. **Publish benchmarks** -- Every competitor cites performance numbers. Weav needs published benchmark comparisons (context assembly latency, token budget packing efficiency, vs Mem0/Zep token overhead).

2. **Open-source with Apache 2.0** -- Differentiates against Redis (RSALv2), FalkorDB (SSPL), Memgraph (BSL), SurrealDB (BSL). Aligns with Mem0, Qdrant, Milvus in the open-source AI infrastructure trend.

3. **Ship MCP server** -- Low effort, high visibility. AI agent builders are actively looking for MCP-compatible tools. Memgraph and NebulaGraph already have this.

4. **Build a "context benchmark" suite** -- Define the evaluation criteria for AI context management (token efficiency, context relevance, assembly latency, conflict detection accuracy). Own the benchmark = own the category definition.

5. **Integrate with LangChain/LlamaIndex/CrewAI** -- Ecosystem presence matters. Mem0 is the AWS Strands memory provider. Weav needs equivalent framework integration.

6. **Consider a Mem0-compatible API layer** -- Allow Mem0 users to swap in Weav as the backend. This is a low-friction adoption path.

---

## Appendix A: Competitor Funding & Scale

| Competitor | Last Known Funding | Valuation | GitHub Stars | License |
|-----------|-------------------|-----------|--------------|---------|
| Neo4j | $325M Series F (2021) | $2B | 13K+ | GPL/Commercial |
| SurrealDB | $23M (Feb 2026) | N/A | 28K+ | BSL 1.1 |
| Mem0 | $24M (Oct 2025) | N/A | 25K+ | Apache 2.0 |
| Milvus | $60M (Zilliz, 2022) | N/A | 42K+ | Apache 2.0 |
| Qdrant | $28M (2023) | N/A | 23K+ | Apache 2.0 |
| Weaviate | $50M Series B (2023) | N/A | 13K+ | BSD 3-Clause |
| Chroma | $18M (2023) | N/A | 18K+ | Apache 2.0 |
| Pinecone | $100M Series B (2023) | $750M | N/A (proprietary) | Proprietary |
| FalkorDB | $3M Seed (2023) | N/A | 2K+ | SSPLv1 |
| Memgraph | $9M Series A (2022) | N/A | 1.5K+ | BSL 1.1 |
| Cognee | EUR 7.5M (Feb 2026) | N/A | 5K+ | Apache 2.0 |
| TigerGraph | $105M Series C (2021) | $1B | N/A | Freemium |
| DragonflyDB | $40M (2023) | N/A | 28K+ | BSL 1.1 |
| LanceDB | $14.8M Series A (2024) | N/A | 5K+ | Apache 2.0 |

## Appendix B: Key Sources

- Neo4j: neo4j.com/docs, neo4j.com/blog
- FalkorDB: falkordb.com, docs.falkordb.com/design
- Memgraph: memgraph.com/memgraphdb
- SurrealDB: surrealdb.com/features, surrealdb.com/blog
- Mem0: mem0.ai/research, arxiv.org/abs/2504.19413
- Zep: getzep.com, arxiv.org/abs/2501.13956
- Milvus: zilliz.com/what-is-milvus
- Qdrant: qdrant.tech/blog/2025-recap
- Redis: redis.io/redis-for-ai
- Cognee: cognee.ai
- Weaviate: weaviate.io/hybrid-search
- TigerGraph: tigergraph.com, arxiv.org/abs/2501.11216
