//! Comprehensive criterion benchmarks for the Weav context graph database.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use compact_str::CompactString;

use weav_core::config::{TokenCounterType, WalSyncMode, WeavConfig};
use weav_core::types::{BiTemporal, TokenBudget, Value};
use weav_graph::adjacency::{AdjacencyStore, EdgeMeta};
use weav_graph::properties::PropertyStore;
use weav_graph::traversal::{bfs, flow_score, EdgeFilter, NodeFilter};
use weav_persist::snapshot::{
    make_meta, EdgeSnapshot, FullSnapshot, GraphSnapshot, NodeSnapshot, SnapshotEngine,
};
use weav_persist::wal::{WalOperation, WriteAheadLog};
use weav_query::budget::enforce_budget;
use weav_query::executor::ContextChunk;
use weav_query::parser::parse_command;
use weav_server::engine::Engine;
use weav_vector::index::{VectorConfig, VectorIndex};
use weav_vector::tokens::TokenCounter;
use weav_core::types::Direction;

// ---- Deterministic pseudo-random helpers ----

fn pseudo_random_f32(seed: u64) -> f32 {
    let x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (x >> 33) as f32 / (1u64 << 31) as f32
}

fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| pseudo_random_f32(seed + i as u64))
        .collect()
}

// ---- 1. Vector Search Benchmark ----

fn bench_vector_search(c: &mut Criterion) {
    let dim = 128;
    let n = 100_000;

    let config = VectorConfig {
        dimensions: dim as u16,
        metric: weav_vector::index::DistanceMetric::Cosine,
        hnsw_m: 16,
        hnsw_ef_construction: 200,
        hnsw_ef_search: 50,
        quantization: weav_vector::index::Quantization::None,
    };
    let mut index = VectorIndex::new(config).unwrap();

    for i in 0..n {
        let vec = random_vector(dim, i as u64 * 1000);
        index.insert(i as u64 + 1, &vec).unwrap();
    }

    let query = random_vector(dim, 999_999);

    c.bench_function("vector_search_100k_128d_k10", |b| {
        b.iter(|| {
            let results = index.search(black_box(&query), 10, None).unwrap();
            black_box(results);
        });
    });
}

// ---- 2. Graph Traversal Benchmark ----

fn build_bench_graph(node_count: u64, avg_degree: u64) -> AdjacencyStore {
    let mut adj = AdjacencyStore::new();
    for i in 1..=node_count {
        adj.add_node(i);
    }

    // Create deterministic edges using pseudo-random targets
    let mut edge_seed: u64 = 42;
    for src in 1..=node_count {
        for _ in 0..avg_degree {
            edge_seed = edge_seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let tgt = (edge_seed % node_count) + 1;
            if tgt != src {
                let meta = EdgeMeta {
                    source: src,
                    target: tgt,
                    label: 0,
                    temporal: BiTemporal::new_current(1000),
                    provenance: None,
                    weight: 1.0,
                    token_cost: 0,
                };
                let _ = adj.add_edge(src, tgt, 0, meta);
            }
        }
    }
    adj
}

fn bench_traversal(c: &mut Criterion) {
    let adj = build_bench_graph(100_000, 5);

    let edge_filter = EdgeFilter::none();
    let node_filter = NodeFilter::none();

    c.bench_function("bfs_100kn_depth3", |b| {
        b.iter(|| {
            let result = bfs(
                black_box(&adj),
                black_box(&[1]),
                3,
                100_000,
                &edge_filter,
                &node_filter,
                Direction::Outgoing,
                None,
                None,
            );
            black_box(result);
        });
    });

    c.bench_function("flow_score_100kn_depth3", |b| {
        b.iter(|| {
            let result = flow_score(
                black_box(&adj),
                black_box(&[(1, 1.0)]),
                0.5,
                0.01,
                3,
            );
            black_box(result);
        });
    });
}

// ---- 3. Node Insert Benchmark ----

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");
    group.throughput(criterion::Throughput::Elements(10_000));

    group.bench_function("node_adjacency_10k", |b| {
        b.iter_with_setup(
            || AdjacencyStore::new(),
            |mut store| {
                for i in 1..=10_000u64 {
                    store.add_node(i);
                }
                black_box(&store);
            },
        );
    });

    group.bench_function("node_property_10k", |b| {
        b.iter_with_setup(
            || PropertyStore::new(),
            |mut store| {
                for i in 1..=10_000u64 {
                    store.set_node_property(
                        i,
                        "name",
                        Value::String(CompactString::from("test_node")),
                    );
                    store.set_node_property(i, "score", Value::Float(0.5));
                }
                black_box(&store);
            },
        );
    });

    group.bench_function("edge_10k", |b| {
        b.iter_with_setup(
            || {
                let mut adj = AdjacencyStore::new();
                for i in 1..=10_001u64 {
                    adj.add_node(i);
                }
                adj
            },
            |mut adj| {
                for i in 1..=10_000u64 {
                    let meta = EdgeMeta {
                        source: i,
                        target: i + 1,
                        label: 0,
                        temporal: BiTemporal::new_current(1000),
                        provenance: None,
                        weight: 1.0,
                        token_cost: 0,
                    };
                    let _ = adj.add_edge(i, i + 1, 0, meta);
                }
                black_box(&adj);
            },
        );
    });

    group.finish();
}

// ---- 4. Token Counting Benchmark ----

fn bench_token_counting(c: &mut Criterion) {
    let counter = TokenCounter::new(TokenCounterType::CharDiv4);

    let short_text = "Hello world";
    let medium_text = "The quick brown fox jumps over the lazy dog. ".repeat(20);
    let long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(200);

    c.bench_function("token_count_short", |b| {
        b.iter(|| {
            let count = counter.count(black_box(short_text));
            black_box(count);
        });
    });

    c.bench_function("token_count_medium", |b| {
        b.iter(|| {
            let count = counter.count(black_box(&medium_text));
            black_box(count);
        });
    });

    c.bench_function("token_count_long", |b| {
        b.iter(|| {
            let count = counter.count(black_box(&long_text));
            black_box(count);
        });
    });
}

// ---- 5. Query Parsing Benchmark ----

fn bench_parse(c: &mut Criterion) {
    let context_query = r#"CONTEXT "what is rust" FROM "knowledge" BUDGET 4096 TOKENS DEPTH 3 DIRECTION BOTH"#;
    let node_add_query = r#"NODE ADD TO "test_graph" LABEL "person" PROPERTIES {"name": "Alice", "age": 30} KEY "alice-001""#;
    let simple_query = "PING";

    c.bench_function("parse_context_query", |b| {
        b.iter(|| {
            let cmd = parse_command(black_box(context_query)).unwrap();
            black_box(cmd);
        });
    });

    c.bench_function("parse_node_add", |b| {
        b.iter(|| {
            let cmd = parse_command(black_box(node_add_query)).unwrap();
            black_box(cmd);
        });
    });

    c.bench_function("parse_ping", |b| {
        b.iter(|| {
            let cmd = parse_command(black_box(simple_query)).unwrap();
            black_box(cmd);
        });
    });
}

// ---- 6. Budget Enforcement Benchmark ----

fn bench_budget(c: &mut Criterion) {
    let chunks: Vec<ContextChunk> = (0..100)
        .map(|i| {
            let score = pseudo_random_f32(i as u64 * 7 + 13);
            let tokens = ((pseudo_random_f32(i as u64 * 11 + 17) * 200.0) as u32).max(1);
            ContextChunk {
                node_id: i as u64 + 1,
                content: format!("Content for node {}. This is some sample text to fill tokens.", i),
                label: "entity".to_string(),
                relevance_score: score,
                depth: (i % 4) as u8,
                token_count: tokens,
                provenance: None,
                relationships: Vec::new(),
                temporal: None,
            }
        })
        .collect();

    let budget = TokenBudget::new(4096);

    c.bench_function("budget_enforce_100_chunks", |b| {
        b.iter(|| {
            let result = enforce_budget(black_box(chunks.clone()), black_box(&budget));
            black_box(result);
        });
    });
}

// ---- 7. Engine End-to-End Benchmark ----

fn bench_engine(c: &mut Criterion) {
    // Set up engine with a populated graph
    let engine = Engine::new(WeavConfig::default());

    // Create a graph
    let cmd = parse_command("GRAPH CREATE \"bench-graph\"").unwrap();
    engine.execute_command(cmd, None).unwrap();

    // Add 100 nodes with properties
    for i in 0..100u64 {
        let cmd_str = format!(
            r#"NODE ADD TO "bench-graph" LABEL "entity" PROPERTIES {{"name": "node_{}", "description": "Description for node {}"}} KEY "key_{}""#,
            i, i, i
        );
        let cmd = parse_command(&cmd_str).unwrap();
        engine.execute_command(cmd, None).unwrap();
    }

    // Add some edges between nodes
    for i in 1..100u64 {
        let cmd_str = format!(
            r#"EDGE ADD TO "bench-graph" FROM {} TO {} LABEL "relates_to" WEIGHT 0.8"#,
            i,
            i + 1
        );
        let cmd = parse_command(&cmd_str).unwrap();
        engine.execute_command(cmd, None).unwrap();
    }

    c.bench_function("engine_ping", |b| {
        b.iter(|| {
            let cmd = parse_command("PING").unwrap();
            let resp = engine.execute_command(black_box(cmd), None).unwrap();
            black_box(resp);
        });
    });

    c.bench_function("engine_node_get", |b| {
        b.iter(|| {
            let cmd = parse_command("NODE GET \"bench-graph\" 1").unwrap();
            let resp = engine.execute_command(black_box(cmd), None).unwrap();
            black_box(resp);
        });
    });

    c.bench_function("engine_graph_list", |b| {
        b.iter(|| {
            let cmd = parse_command("GRAPH LIST").unwrap();
            let resp = engine.execute_command(black_box(cmd), None).unwrap();
            black_box(resp);
        });
    });

    c.bench_function("engine_graph_info", |b| {
        b.iter(|| {
            let cmd = parse_command("GRAPH INFO \"bench-graph\"").unwrap();
            let resp = engine.execute_command(black_box(cmd), None).unwrap();
            black_box(resp);
        });
    });
}

// ---- 8. Context Query End-to-End Benchmark ----

fn bench_context_query(c: &mut Criterion) {
    // Setup: create Engine, create graph, add 1000 nodes with properties,
    // add edges in chain topology
    let engine = Engine::new(WeavConfig::default());

    let cmd = parse_command("GRAPH CREATE \"ctx-bench\"").unwrap();
    engine.execute_command(cmd, None).unwrap();

    // Add 1000 nodes with properties
    for i in 0..1000u64 {
        let cmd_str = format!(
            r#"NODE ADD TO "ctx-bench" LABEL "document" PROPERTIES {{"title": "doc_{}", "content": "This is document number {} with some searchable content about topic {}"}} KEY "doc_{}""#,
            i, i, i % 50, i
        );
        let cmd = parse_command(&cmd_str).unwrap();
        engine.execute_command(cmd, None).unwrap();
    }

    // Add edges in chain topology
    for i in 1..1000u64 {
        let cmd_str = format!(
            r#"EDGE ADD TO "ctx-bench" FROM {} TO {} LABEL "next" WEIGHT 0.9"#,
            i,
            i + 1
        );
        let cmd = parse_command(&cmd_str).unwrap();
        engine.execute_command(cmd, None).unwrap();
    }

    // Benchmark: execute a CONTEXT query with SEEDS NODES, DEPTH 3, BUDGET 4096
    // This measures the full query pipeline: parse -> plan -> execute -> budget
    c.bench_function("context_query_1000n_depth3_budget4096", |b| {
        b.iter(|| {
            let cmd = parse_command(
                r#"CONTEXT "find documents" FROM "ctx-bench" SEEDS NODES ["doc_0"] DEPTH 3 BUDGET 4096 TOKENS"#,
            )
            .unwrap();
            let resp = engine.execute_command(black_box(cmd), None).unwrap();
            black_box(resp);
        });
    });
}

// ---- 9. Persistence Benchmark ----

fn bench_persistence(c: &mut Criterion) {
    let mut group = c.benchmark_group("persistence");

    // bench_wal_append: Create a WAL in a temp dir, benchmark appending NodeAdd entries
    group.bench_function("wal_append", |b| {
        b.iter_with_setup(
            || {
                let dir = std::env::temp_dir().join(format!(
                    "weav_bench_wal_{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ));
                std::fs::create_dir_all(&dir).unwrap();
                let wal_path = dir.join("bench.wal");
                let wal =
                    WriteAheadLog::new(wal_path, 1024 * 1024 * 64, WalSyncMode::Never).unwrap();
                (wal, dir)
            },
            |(mut wal, dir)| {
                for i in 0..100u64 {
                    wal.append(
                        0,
                        WalOperation::NodeAdd {
                            graph_id: 1,
                            node_id: i,
                            label: "entity".into(),
                            properties_json: format!(r#"{{"name":"node_{}"}}"#, i),
                            embedding: None,
                            entity_key: Some(format!("key_{}", i)),
                        },
                    )
                    .unwrap();
                }
                black_box(&wal);
                std::fs::remove_dir_all(&dir).ok();
            },
        );
    });

    // bench_snapshot_save: Create a FullSnapshot with 1000 nodes, benchmark save_snapshot()
    group.bench_function("snapshot_save_1000n", |b| {
        // Build a snapshot with 1000 nodes once
        let nodes: Vec<NodeSnapshot> = (0..1000)
            .map(|i| NodeSnapshot {
                node_id: i,
                label: "entity".into(),
                properties_json: format!(
                    r#"{{"name":"node_{}","description":"Description for node {}"}}"#,
                    i, i
                ),
                embedding: Some(vec![0.1; 16]),
                entity_key: Some(format!("key_{}", i)),
            })
            .collect();

        let edges: Vec<EdgeSnapshot> = (0..999)
            .map(|i| EdgeSnapshot {
                edge_id: i,
                source: i,
                target: i + 1,
                label: "relates_to".into(),
                weight: 0.8,
                valid_from: 1000,
                valid_until: u64::MAX,
            })
            .collect();

        let snapshot = FullSnapshot {
            meta: make_meta(1000, 999, 1, 0),
            graphs: vec![GraphSnapshot {
                graph_id: 1,
                graph_name: "bench-graph".into(),
                config_json: "{}".into(),
                nodes,
                edges,
            }],
        };

        b.iter_with_setup(
            || {
                let dir = std::env::temp_dir().join(format!(
                    "weav_bench_snap_{}",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ));
                std::fs::create_dir_all(&dir).unwrap();
                let engine = SnapshotEngine::new(dir.clone());
                (engine, dir)
            },
            |(snap_engine, dir)| {
                let path = snap_engine.save_snapshot(&snapshot).unwrap();
                black_box(path);
                std::fs::remove_dir_all(&dir).ok();
            },
        );
    });

    group.finish();
}

// ---- Criterion Groups and Main ----

criterion_group!(
    benches,
    bench_vector_search,
    bench_traversal,
    bench_insert,
    bench_token_counting,
    bench_parse,
    bench_budget,
    bench_engine,
    bench_context_query,
    bench_persistence,
);

criterion_main!(benches);
