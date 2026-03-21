# Competitive Landscape: Context Graph Databases for AI/LLM Workloads

**Date**: 2026-03-21
**Scope**: Feature comparison of 7 products against Weav capabilities

---

## Executive Summary

The context graph database space for AI/LLM workloads is fragmenting into three tiers:

1. **Purpose-built AI memory systems** (Graphiti/Zep, LightRAG, Microsoft GraphRAG) -- focused on LLM context, entity extraction, and RAG patterns. Limited graph algorithm depth.
2. **General-purpose graph databases adding AI features** (Neo4j, FalkorDB, Memgraph) -- deep graph algorithm libraries, adding vector search and LLM integrations. Not designed for token-budget-aware context retrieval.
3. **Embedded/analytical graph engines** (Kuzu) -- high performance, minimal AI-specific features. Kuzu was acquired by Apple (Oct 2025) and archived.

Weav occupies a unique position: purpose-built for AI workloads (like tier 1) but with native graph algorithms, vector search, and a Redis-like protocol (like tier 2), plus in-memory embedded performance (like tier 3). The main gaps to address are entity resolution quality, MCP tool breadth, and SDK ecosystem depth.

---

## Product-by-Product Deep Dive

### 1. Graphiti (by Zep)

**What it is**: Open-source temporal knowledge graph framework for AI agent memory. Not a database -- it sits atop Neo4j, FalkorDB, Kuzu, or Neptune as a storage backend.

**Core strengths**:
- **Entity resolution**: Three-tier cascade: exact match, fuzzy similarity, LLM-based reasoning. Edge deduplication uses the same approach, constrained to edges between the same entity pairs.
- **Bi-temporal model**: Every edge tracks four temporal dimensions (valid_from, valid_until, system_from, system_until). Facts are invalidated but never deleted -- full history preserved.
- **Incremental ingestion**: `add_episode()` processes new data immediately without full graph recomputation. Episodes are the provenance/raw data ground truth.
- **Hybrid retrieval**: Semantic (embedding), BM25 keyword, and graph traversal search with result fusion. Graph-distance reranking.
- **MCP server**: 6 tools -- `add_episode`, `search_nodes`, `search_facts`, `delete_entity_edge`, `delete_episode`, `get_episodes`. Plus group management and graph maintenance.
- **Multi-tenancy**: Via `group_id` field on all nodes/edges. Logical isolation within a single database.
- **Observability**: OTEL tracing support.

**Weaknesses**:
- No native graph algorithms (PageRank, community detection, shortest path). Relies entirely on the backend database.
- No vector index of its own -- uses backend or external embeddings.
- No token budget management.
- Python-only SDK for core (TypeScript and Go SDKs exist but are thinner).
- No TTL / automatic expiry.
- No schema validation.
- No batch import/export tooling.
- Concurrency limited by SEMAPHORE_LIMIT (default 10 concurrent episodes).

**SDKs**: Python (primary), TypeScript, Go
**Backend support**: Neo4j, FalkorDB, Kuzu, Amazon Neptune
**LLM providers**: OpenAI (default), Anthropic, Gemini, Groq, Azure OpenAI

---

### 2. Microsoft GraphRAG

**What it is**: A modular Python framework for graph-based RAG. Builds hierarchical community summaries from document corpora. Not a database -- produces indexed artifacts that power search.

**Core strengths**:
- **Community summaries**: Leiden/hierarchical community detection, LLM-generated summaries at each level. Multiple levels of abstraction.
- **Three search modes**: Global (community summaries for broad questions), Local (entity neighborhood fan-out), DRIFT (dynamic reasoning combining both with follow-up questions).
- **LazyGraphRAG** (June 2025): Dramatically reduced indexing cost by deferring summarization to query time.
- **Incremental indexing** (v0.4.0+): New documents can be added without full re-index, leveraging cache to avoid repeat LLM calls.

**Weaknesses**:
- **No entity resolution**: Explicitly acknowledged as unimplemented. Entities matched by name only, leading to dedup issues when names collide across types.
- **No temporal model**: No bi-temporal tracking, no fact invalidation. Under research.
- **No real-time/streaming**: Batch-oriented indexing pipeline. Graph must be rebuilt for community recomputation.
- **No native graph algorithms**: Community detection is done during indexing, not queryable at runtime.
- **No vector database**: Relies on external vector stores or file-based output.
- **No multi-tenancy**: Single-corpus design.
- **No MCP server**: Community implementations exist (FastMCP wrappers) but nothing official.
- **Extremely token-heavy**: 610,000 tokens per query for full GraphRAG (vs 100 for LightRAG).
- **No TTL, schema validation, transactions, or event subscriptions**.

**SDKs**: Python only (PyPI: `graphrag`, v2.7.0 as of Jan 2026)
**Storage**: File, memory, Azure Blob, CosmosDB

---

### 3. FalkorDB

**What it is**: Redis-compatible in-memory graph database using GraphBLAS sparse matrices. Fork of RedisGraph. Focused on GraphRAG and low-latency AI workloads.

**Core strengths**:
- **Performance**: Sub-millisecond query latency via sparse matrix (GraphBLAS) representation with AVX acceleration.
- **Cypher support**: Full openCypher query language.
- **Vector search**: HNSW index with configurable M (16), efConstruction (200), efRuntime (10). Cosine similarity and Euclidean distance. Flexible dimensions (768, 1536, etc.).
- **Graph algorithms** (8): BFS, shortest path (SPpath), single-source paths (SSpath), minimum spanning forest, PageRank, betweenness centrality, weakly connected components, community detection (label propagation).
- **Multi-tenancy**: Native, up to 10,000+ isolated graph instances per deployment. Full graph isolation.
- **MCP server**: Official `FalkorDB-MCPServer` -- query graphs via OpenCypher, read-only mode, natural language support.
- **Redis compatibility**: RESP protocol, can use existing Redis clients/tooling.

**Weaknesses**:
- **No temporal model**: No bi-temporal tracking, no fact invalidation.
- **No entity resolution**: No built-in dedup/merge.
- **Limited algorithms**: Only 8 algorithms vs Neo4j's 30+ or Memgraph's 40+.
- **No token budget management**.
- **No TTL** (not found in docs).
- **No streaming ingestion** (no Kafka/Pulsar connectors).
- **No event subscriptions / CDC**.
- **No schema validation constraints**.
- **ACID**: Snapshot + transaction log durability, but not full ACID in the PostgreSQL sense.

**SDKs**: Python, Node.js, Java, Rust, Go, C#
**Integrations**: LangChain, LlamaIndex, Graphiti (as backend)

---

### 4. Neo4j

**What it is**: The dominant general-purpose graph database. Enterprise-grade, disk-based with caching. Massive ecosystem.

**Core strengths**:
- **Graph Data Science (GDS) library**: 30+ algorithms across 6 categories:
  - *Centrality*: PageRank, ArticleRank, Betweenness, Degree, Eigenvector, Closeness, HITS, Influence Maximization
  - *Community*: Louvain, Leiden, Label Propagation, WCC, SCC, Speaker-Listener LPA, Triangle Count, K-1 Coloring, K-Means
  - *Similarity*: Node Similarity, K-Nearest Neighbors
  - *Path finding*: Dijkstra, A*, BFS, DFS, Minimum Spanning Tree
  - *Node embeddings*: Node2Vec, FastRP, GraphSAGE, HashGNN
  - *DAG algorithms*: Topological sort, longest path
- **Vector search**: Native HNSW index (since v5.11). Integrated with Cypher.
- **Multi-tenancy**: Multiple databases per instance (Enterprise Edition, since v4.0).
- **Change Data Capture (CDC)**: Real-time change tracking with FULL and DIFF enrichment modes. Kafka connector for streaming changes.
- **TTL**: Via APOC procedures (`apoc.ttl.expire`, `apoc.ttl.expireIn`). Configurable schedule and batch size.
- **Schema constraints**: Unique, existence, node key constraints.
- **Triggers**: APOC trigger procedures for event-driven logic.
- **Temporal types**: Native date, time, datetime, duration types. Bi-temporal modeling via patterns (not built-in).
- **Observability**: Prometheus metrics endpoint, JMX, query logging.
- **ACID transactions**: Full ACID with causal consistency in clusters.

**Weaknesses**:
- **Not designed for AI context**: No token budget management, no context-aware retrieval.
- **No entity resolution built-in**: Requires custom implementation or GenAI tooling.
- **Heavy operational footprint**: JVM-based, significant memory overhead, complex clustering.
- **GDS is licensed separately**: Enterprise/commercial license for full algorithm library.
- **Latency**: Disk-based; not sub-millisecond for complex traversals without warming.
- **No MCP server**: No official implementation.
- **Cost**: Enterprise features (multi-tenancy, CDC, GDS) require paid license.

**SDKs**: Java, Python, JavaScript, .NET, Go (official). Rust (community: neo4rs). Python driver now uses Rust extensions for 1.2-4.3x speedup.
**Integrations**: LangChain, LlamaIndex, Spring Data, GraphQL, extensive ecosystem

---

### 5. Kuzu

**What it is**: Embedded columnar property graph database. Designed for analytical workloads on a single machine. Implements Cypher.

**Core strengths**:
- **Columnar storage**: CSR-based adjacency lists, vectorized query processing. Extremely fast analytical queries.
- **Graph algorithms (algo extension)**: PageRank, Louvain, WCC, SCC, K-Core decomposition. Disk-based execution (can exceed RAM).
- **Vector search**: Native vector index + full-text search.
- **Cypher support**: Full Cypher implementation.
- **ACID transactions**: Serializable isolation.
- **Embeddable**: Runs in-process (Python, Node, Rust, Go, Java, WebAssembly).
- **Multi-core parallelism**: Factorized query processing.

**Weaknesses**:
- **Acquired by Apple (Oct 2025)**: Repository archived, website taken down. No longer available for new adoption.
- **No temporal model**.
- **No entity resolution**.
- **No streaming ingestion**.
- **No MCP server**.
- **No TTL**.
- **Limited algorithm set** (5 algorithms).
- **No multi-tenancy**.
- **No event subscriptions**.

**SDKs**: Python, Node.js, Rust, Go, Java, WebAssembly
**Status**: ARCHIVED -- not viable for new projects

---

### 6. LightRAG

**What it is**: A lightweight graph-based RAG framework focused on cost efficiency. Python library, not a database. Uses external storage backends.

**Core strengths**:
- **Dual-level retrieval**: Low-level (specific entity/edge lookup) and high-level (aggregated multi-hop reasoning). Mixed mode with reranker.
- **Extreme cost efficiency**: ~100 tokens per query vs GraphRAG's ~610,000. 6,000x reduction.
- **~30% lower latency** than standard RAG (~80ms vs ~120ms).
- **Flexible storage backends**:
  - KV: JSON, PostgreSQL, Redis, MongoDB, OpenSearch
  - Vector: NanoVectorDB, pgvector, Milvus, Chroma, FAISS, MongoDB, Qdrant, OpenSearch
  - Graph: NetworkX, Neo4j, PostgreSQL, Apache AGE, OpenSearch
- **Entity extraction**: LLM-based entity/relationship extraction with dedup protocols.
- **Incremental updates**: Documents can be added/deleted without full rebuild. Graph auto-regenerates.
- **node2vec embeddings**: Graph structure encoded as vectors.
- **MCP server**: Community implementations with 22-30+ tools.
- **Multi-tenancy**: Workspace-based isolation via `LIGHTRAG-WORKSPACE` header.
- **Streaming responses**: Supported.
- **Observability**: Langfuse tracing integration.

**Weaknesses**:
- **No native graph algorithms**: Relies on backend (NetworkX for in-memory, Neo4j for production).
- **No temporal model**: No bi-temporal tracking or fact invalidation.
- **No ACID transactions**: Depends on backend.
- **No TTL**.
- **No schema validation**.
- **No CDC / event subscriptions**.
- **Python only**: No TypeScript/Rust/Go SDK.
- **Entity resolution is basic**: LLM-based extraction with dedup, but no cascading resolution like Graphiti.

**SDKs**: Python only
**LLM support**: OpenAI, Ollama, any OpenAI-compatible API

---

### 7. Memgraph

**What it is**: In-memory graph database written in C/C++. Focused on real-time streaming analytics. openCypher query language.

**Core strengths**:
- **MAGE algorithm library**: 40+ algorithms including:
  - *Centrality*: PageRank, Betweenness, Katz, Degree (+ dynamic versions)
  - *Community*: Louvain, Leiden, Label Propagation (+ dynamic)
  - *Path*: BFS, DFS, weighted shortest path, all shortest paths, K shortest paths
  - *ML*: Node2Vec, GNN link prediction, GNN node classification, Temporal Graph Networks
  - *Clustering*: K-means, Spectral, Balanced Cut
  - *Other*: TSP, VRP, Max Flow, Graph Coloring, Bipartite Matching, Bridges, Cycles
- **Dynamic/streaming algorithms**: Online versions of PageRank, betweenness, community detection, Katz centrality that update as the graph changes.
- **Vector search**: Native HNSW via USearch library. f32 and f16 precision. Cosine similarity function.
- **Streaming ingestion**: Native connectors for Kafka, Redpanda, Pulsar. At-least-once semantics.
- **Triggers**: Cypher-based triggers on node/edge creation, with transaction-level execution.
- **ACID transactions**: Full ACID compliance with configurable retry (default 30 retries).
- **Multi-tenancy** (Enterprise): Isolated namespaces per tenant, shared infrastructure.
- **TTL**: Native TTL via labels and properties. Background job for expiration. Supports both node and edge TTL. Replication-aware.
- **GPU acceleration**: NVIDIA cuGraph integration via MAGE for GPU-powered algorithms.
- **Schema constraints**: Indexes and constraints with concurrent read support during schema changes.
- **LLM integration**: Built-in LLM query module using LiteLLM.

**Weaknesses**:
- **No temporal model**: No bi-temporal tracking. TTL is for deletion, not historical queries.
- **No entity resolution**: No built-in dedup/merge for knowledge graph construction.
- **No token budget management**: Not designed for LLM context window optimization.
- **No MCP server**: No official implementation found.
- **No community summaries / GraphRAG patterns**.
- **Limited multi-tenancy in open source**: Enterprise-only feature.
- **No CDC**: Changes not exposed as event stream (triggers exist but are internal).

**SDKs**: Python (pymgclient, GQLAlchemy), C (official). Also compatible with Neo4j Bolt drivers (Python, Java, JS, Go, .NET).
**Integrations**: LangChain, Elasticsearch synchronization, NetworkX, igraph, cuGraph

---

## Feature Comparison Matrix

| Feature | Weav | Graphiti | GraphRAG | FalkorDB | Neo4j | Kuzu | LightRAG | Memgraph |
|---|---|---|---|---|---|---|---|---|
| **Category** | AI graph DB | AI framework | RAG framework | Graph DB | Graph DB | Embedded DB | RAG framework | Graph DB |
| **Language** | Rust | Python | Python | C (GraphBLAS) | Java | C++ | Python | C/C++ |
| **In-memory** | Yes | Via backend | N/A | Yes | Cache layer | Columnar/disk | Via backend | Yes |
| | | | | | | | | |
| **Entity Resolution** | Basic (dedup) | Excellent (3-tier) | None | None | None | None | Basic (LLM) | None |
| **Temporal / Versioning** | Bi-temporal | Bi-temporal | None | None | Pattern-based | None | None | None |
| **Fact Invalidation** | Yes | Yes (never deletes) | None | None | Manual | None | None | None |
| | | | | | | | | |
| **Vector Search** | HNSW (usearch) | Via backend | Via external | HNSW | HNSW (native) | Native vector | Via backend | HNSW (usearch) |
| **Max Dimensions** | 4096 | Backend-dep | External | Flexible | Flexible | Unknown | Backend-dep | Flexible |
| **Similarity Metrics** | Cosine | Backend-dep | External | Cosine, L2 | Cosine, L2 | Unknown | Cosine | Cosine, L2 |
| | | | | | | | | |
| **Graph Algorithms** | | | | | | | | |
| PageRank | Yes | No (backend) | No | Yes | Yes | Yes | No (backend) | Yes |
| Personalized PageRank | Yes | No | No | No | Yes (via config) | No | No | No |
| Community Detection | Louvain, LP | No | Leiden (index) | CDLP | Louvain, Leiden, LP, SLPA | Louvain | No | Louvain, Leiden, LP |
| Shortest Path | Dijkstra | No | No | BFS, SPpath | Dijkstra, A* | No | No | Weighted, K-shortest |
| Connected Components | Yes | No | No | WCC | WCC, SCC | WCC, SCC | No | WCC |
| Betweenness Centrality | No | No | No | Yes | Yes | No | No | Yes (+dynamic) |
| Node Embeddings | No | No | No | No | Node2Vec, FastRP, GraphSAGE | No | node2vec | Node2Vec, GNN |
| Clustering | No | No | No | No | K-Means, K-1 Color | K-Core | No | K-Means, Spectral |
| Dynamic/Online Algos | No | No | No | No | No | No | No | Yes (5 algorithms) |
| Total Algorithms | ~6 | 0 (backend) | 1 (Leiden) | 8 | 30+ | 5 | 0 (backend) | 40+ |
| | | | | | | | | |
| **Token Budget Mgmt** | Yes (knapsack, MMR, submodular) | No | No | No | No | No | No | No |
| **Hybrid Retrieval** | RRF fusion | Semantic+BM25+Graph | Global+Local+DRIFT | Cypher+Vector | Cypher+Vector | Cypher+Vector | Dual-level | Cypher+Vector |
| **GraphRAG Patterns** | Community summaries | No | Full (hierarchical) | No | Via integration | No | Dual-level | No |
| | | | | | | | | |
| **TTL / Auto Expiry** | No | No | No | No | Yes (APOC) | No | No | Yes (native) |
| **Streaming Ingestion** | No | Incremental | Batch only | No | CDC + Kafka | No | Incremental | Kafka, Pulsar, Redpanda |
| **Query Language** | Custom (31 cmds) | Python API | Python API | Cypher | Cypher | Cypher | Python API | Cypher |
| **Multi-tenancy** | No (per-graph) | group_id | No | Native (10K+) | Multi-DB (Ent.) | No | Workspace-based | Namespaces (Ent.) |
| **Observability** | No | OTEL tracing | No | No | Prometheus, JMX | No | Langfuse | Enterprise metrics |
| **ACID Transactions** | Per-graph RwLock | Backend-dep | N/A | Snapshot+log | Full ACID | Serializable | Backend-dep | Full ACID |
| **Batch Import/Export** | No | SEMAPHORE bulk | Incremental index | Bulk Cypher | LOAD CSV, APOC | CSV import | Bulk insert API | CSV, CYPHERL |
| **Schema Validation** | No | Pydantic types | No | No | Constraints | No | No | Constraints |
| **Event Subscriptions** | No | No | No | No | CDC, APOC triggers | No | No | Triggers |
| **MCP Server** | 8 tools | 6 tools | Community only | Official | No | No | Community (22+ tools) | No |
| | | | | | | | | |
| **Protocol** | RESP3, gRPC, HTTP | REST (FastAPI) | Python API | RESP + Bolt | Bolt, HTTP | Embedded | REST API | Bolt |
| **SDKs** | Python, Node.js | Python, TS, Go | Python | Py, Node, Java, Rust, Go, C# | Py, Java, JS, .NET, Go | Py, Node, Rust, Go, Java, Wasm | Python | Py, C (+ Bolt drivers) |
| **LangChain Integration** | No | No | No | Yes | Yes | No | No | Yes |
| **LlamaIndex Integration** | No | No | No | Yes | Yes | No | No | No |

---

## Strategic Gap Analysis for Weav

### Where Weav Already Wins

1. **Token budget management** -- No competitor has anything like the knapsack/MMR/submodular facility location budget algorithms. This is a unique differentiator.
2. **Bi-temporal model** -- Only Graphiti matches this. All general-purpose graph DBs lack it.
3. **Triple-protocol support** (RESP3 + gRPC + HTTP) -- No competitor offers all three.
4. **In-memory Rust performance** -- FalkorDB is the closest competitor here (C/GraphBLAS), but Weav's thread-per-core with keyspace sharding is architecturally superior for AI workloads.
5. **String interning** -- Unique optimization for property-heavy AI workloads.

### Critical Gaps to Close

| Priority | Gap | Competitor Benchmark | Effort Estimate |
|---|---|---|---|
| **P0** | Entity resolution quality | Graphiti's 3-tier (exact/fuzzy/LLM) | High -- needs LLM integration |
| **P0** | MCP server tool breadth | Graphiti (6), LightRAG community (22+) | Medium -- extend existing 8 tools |
| **P1** | TTL / automatic expiry | Memgraph (native node+edge TTL), Neo4j (APOC) | Low -- background sweeper task |
| **P1** | Streaming ingestion | Memgraph (Kafka/Pulsar/Redpanda) | Medium -- tokio async consumer |
| **P1** | More graph algorithms | Neo4j (30+), Memgraph (40+) | Medium -- incremental additions |
| **P1** | Node embeddings | Neo4j (Node2Vec, FastRP, GraphSAGE) | Medium |
| **P2** | CDC / event subscriptions | Neo4j (full CDC), Memgraph (triggers) | Medium |
| **P2** | Schema validation/constraints | Neo4j (unique, existence), Memgraph (constraints) | Low |
| **P2** | Observability (Prometheus/OTEL) | Neo4j (Prometheus), Graphiti (OTEL) | Low-Medium |
| **P2** | LangChain/LlamaIndex integration | Neo4j, FalkorDB, Memgraph | Medium |
| **P3** | Cypher query language | FalkorDB, Neo4j, Memgraph, Kuzu all use it | High -- major parser work |
| **P3** | GPU-accelerated algorithms | Memgraph (cuGraph) | High |

### Recommended Algorithms to Add (by competitive pressure)

| Algorithm | Neo4j | FalkorDB | Memgraph | Weav Has | Priority |
|---|---|---|---|---|---|
| Betweenness Centrality | Yes | Yes | Yes | No | P1 |
| Leiden Community | Yes | No | Yes | No | P1 |
| Node2Vec Embeddings | Yes | No | Yes | No | P1 |
| A* Shortest Path | Yes | No | No | No | P2 |
| K-Shortest Paths | No | No | Yes | No | P2 |
| Triangle Count | Yes | No | No | No | P2 |
| Degree Centrality | Yes | No | Yes | No | P3 |
| Closeness Centrality | Yes | No | No | No | P3 |
| Eigenvector Centrality | Yes | No | No | No | P3 |
| HITS | Yes | No | No | No | P3 |
| K-Means Clustering | Yes | No | Yes | No | P3 |

### Recommended MCP Tools to Add

Current Weav MCP tools (8): graph_list, graph_info, graph_create, graph_drop, node_add, node_get, edge_add, server_info

| Tool | Graphiti Has | LightRAG Has | Priority |
|---|---|---|---|
| context_query (with token budget) | No equivalent | No equivalent | P0 -- unique differentiator |
| search_nodes (semantic) | Yes | Yes | P0 |
| search_facts / search_edges | Yes | Yes | P0 |
| ingest_document | No (add_episode) | Yes | P1 |
| node_update / node_delete | Partial | Yes | P1 |
| edge_get / edge_delete | Yes | Yes | P1 |
| graph_algorithms (run PageRank etc.) | No | No | P1 -- differentiator |
| vector_search | No | Yes | P1 |
| temporal_query (point-in-time) | No | No | P2 -- differentiator |
| bulk_import | No | Yes | P2 |

---

## Key Takeaways

1. **Graphiti is the closest competitor** for AI-native context graphs, but it is a framework (not a database) that depends on Neo4j/FalkorDB underneath. Weav's integrated approach (graph + vector + temporal + budget in one binary) is architecturally superior.

2. **No competitor has token budget management**. This is Weav's strongest unique feature. The MCP server should expose this prominently via a `context_query` tool.

3. **Entity resolution is the biggest quality gap**. Graphiti's 3-tier approach (exact/fuzzy/LLM) is the gold standard. Weav's current dedup is basic.

4. **The algorithm gap matters for enterprise adoption**. Neo4j's 30+ and Memgraph's 40+ algorithms set expectations. Betweenness centrality, Leiden, and node embeddings are the highest-impact additions.

5. **MCP is becoming table stakes**. Graphiti, FalkorDB, and LightRAG all have MCP servers. Weav's 8 tools need to expand to ~15-20 covering search, context queries, and algorithm execution.

6. **Cypher is the de facto standard** -- every serious graph DB supports it. Weav's custom 31-command language is a barrier to adoption. Consider adding a Cypher compatibility layer.

7. **Streaming (Kafka/Pulsar) ingestion** is a differentiator for Memgraph and would be valuable for Weav's real-time AI use cases.

---

## Sources

- [Graphiti GitHub](https://github.com/getzep/graphiti)
- [Zep Temporal Knowledge Graph Paper](https://arxiv.org/abs/2501.13956)
- [Graphiti MCP Server Docs](https://help.getzep.com/graphiti/getting-started/mcp-server)
- [Microsoft GraphRAG GitHub](https://github.com/microsoft/graphrag)
- [Microsoft DRIFT Search](https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/)
- [FalkorDB Algorithms Docs](https://docs.falkordb.com/algorithms/)
- [FalkorDB Vector Index Docs](https://docs.falkordb.com/cypher/indexing/vector-index.html)
- [FalkorDB MCP Server](https://github.com/FalkorDB/FalkorDB-MCPServer)
- [Neo4j GDS Algorithms](https://neo4j.com/docs/graph-data-science/current/algorithms/)
- [Neo4j CDC](https://neo4j.com/docs/cdc/current/)
- [Neo4j TTL (APOC)](https://neo4j.com/labs/apoc/4.2/graph-updates/ttl/)
- [Kuzu GitHub (archived)](https://github.com/kuzudb/kuzu)
- [Kuzu Algo Extension](https://docs.kuzudb.com/extensions/algo/)
- [LightRAG GitHub](https://github.com/HKUDS/LightRAG)
- [LightRAG MCP Server](https://github.com/desimpkins/daniel-lightrag-mcp)
- [Memgraph MAGE Algorithms](https://memgraph.com/docs/advanced-algorithms/available-algorithms)
- [Memgraph Vector Search](https://memgraph.com/docs/querying/vector-search)
- [Memgraph TTL](https://memgraph.com/docs/querying/time-to-live)
- [Memgraph Streaming](https://memgraph.com/docs/data-streams)
- [Memgraph Multi-tenancy](https://memgraph.com/docs/database-management/multi-tenancy)
