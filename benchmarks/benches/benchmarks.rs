//! Comprehensive criterion benchmarks for the Weav context graph database.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use compact_str::CompactString;

use weav_core::config::{TokenCounterType, WeavConfig};
use weav_core::types::{BiTemporal, TokenBudget, Value};
use weav_graph::adjacency::{AdjacencyStore, EdgeMeta};
use weav_graph::properties::PropertyStore;
use weav_graph::traversal::{bfs, flow_score, EdgeFilter, NodeFilter};
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
    let n = 1000;

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

    c.bench_function("vector_search_1000_128d_k10", |b| {
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
    let adj = build_bench_graph(1000, 5);

    let edge_filter = EdgeFilter::none();
    let node_filter = NodeFilter::none();

    c.bench_function("bfs_1000n_depth3", |b| {
        b.iter(|| {
            let result = bfs(
                black_box(&adj),
                black_box(&[1]),
                3,
                10_000,
                &edge_filter,
                &node_filter,
                Direction::Outgoing,
            );
            black_box(result);
        });
    });

    c.bench_function("flow_score_1000n_depth3", |b| {
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
    c.bench_function("insert_node_adjacency", |b| {
        b.iter_with_setup(
            || AdjacencyStore::new(),
            |mut store| {
                for i in 1..=1000u64 {
                    store.add_node(i);
                }
                black_box(&store);
            },
        );
    });

    c.bench_function("insert_node_property", |b| {
        b.iter_with_setup(
            || PropertyStore::new(),
            |mut store| {
                for i in 1..=1000u64 {
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

    c.bench_function("insert_edge", |b| {
        b.iter_with_setup(
            || {
                let mut adj = AdjacencyStore::new();
                for i in 1..=1001u64 {
                    adj.add_node(i);
                }
                adj
            },
            |mut adj| {
                for i in 1..=1000u64 {
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
    engine.execute_command(cmd).unwrap();

    // Add 100 nodes with properties
    for i in 0..100u64 {
        let cmd_str = format!(
            r#"NODE ADD TO "bench-graph" LABEL "entity" PROPERTIES {{"name": "node_{}", "description": "Description for node {}"}} KEY "key_{}""#,
            i, i, i
        );
        let cmd = parse_command(&cmd_str).unwrap();
        engine.execute_command(cmd).unwrap();
    }

    // Add some edges between nodes
    for i in 1..100u64 {
        let cmd_str = format!(
            r#"EDGE ADD TO "bench-graph" FROM {} TO {} LABEL "relates_to" WEIGHT 0.8"#,
            i,
            i + 1
        );
        let cmd = parse_command(&cmd_str).unwrap();
        engine.execute_command(cmd).unwrap();
    }

    c.bench_function("engine_ping", |b| {
        b.iter(|| {
            let cmd = parse_command("PING").unwrap();
            let resp = engine.execute_command(black_box(cmd)).unwrap();
            black_box(resp);
        });
    });

    c.bench_function("engine_node_get", |b| {
        b.iter(|| {
            let cmd = parse_command("NODE GET \"bench-graph\" 1").unwrap();
            let resp = engine.execute_command(black_box(cmd)).unwrap();
            black_box(resp);
        });
    });

    c.bench_function("engine_graph_list", |b| {
        b.iter(|| {
            let cmd = parse_command("GRAPH LIST").unwrap();
            let resp = engine.execute_command(black_box(cmd)).unwrap();
            black_box(resp);
        });
    });

    c.bench_function("engine_graph_info", |b| {
        b.iter(|| {
            let cmd = parse_command("GRAPH INFO \"bench-graph\"").unwrap();
            let resp = engine.execute_command(black_box(cmd)).unwrap();
            black_box(resp);
        });
    });
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
);

criterion_main!(benches);
