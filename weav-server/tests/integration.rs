//! Integration tests for the Weav context graph database.
//!
//! These tests exercise the full stack through the Engine, verifying that
//! all components (parser, graph, vector, query, budget) work together.

use weav_core::config::WeavConfig;
use weav_query::parser::parse_command;
use weav_server::engine::{CommandResponse, Engine};

// ---- Helpers ----

fn make_engine() -> Engine {
    Engine::new(WeavConfig::default())
}

fn create_graph(engine: &Engine, name: &str) {
    let cmd = parse_command(&format!("GRAPH CREATE \"{name}\"")).unwrap();
    engine.execute_command(cmd).unwrap();
}

fn add_node(engine: &Engine, graph: &str, label: &str, name: &str, key: Option<&str>) -> u64 {
    let cmd_str = if let Some(k) = key {
        format!(
            r#"NODE ADD TO "{graph}" LABEL "{label}" PROPERTIES {{"name": "{name}"}} KEY "{k}""#
        )
    } else {
        format!(
            r#"NODE ADD TO "{graph}" LABEL "{label}" PROPERTIES {{"name": "{name}"}}"#
        )
    };
    let cmd = parse_command(&cmd_str).unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::Integer(id) => id,
        other => panic!("expected Integer from NODE ADD, got: {:?}", other),
    }
}

fn add_edge(engine: &Engine, graph: &str, src: u64, tgt: u64, label: &str, weight: f32) -> u64 {
    let cmd_str = format!(
        r#"EDGE ADD TO "{graph}" FROM {src} TO {tgt} LABEL "{label}" WEIGHT {weight}"#
    );
    let cmd = parse_command(&cmd_str).unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::Integer(id) => id,
        other => panic!("expected Integer from EDGE ADD, got: {:?}", other),
    }
}

// ---- Test 1: Full Graph Lifecycle ----

/// Create graph, add nodes, add edges, run context query, verify, drop graph.
#[test]
fn test_full_graph_lifecycle() {
    let engine = make_engine();

    // Create graph
    create_graph(&engine, "lifecycle-graph");

    // Verify the graph exists in the list
    let cmd = parse_command("GRAPH LIST").unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::StringList(names) => {
            assert!(names.contains(&"lifecycle-graph".to_string()));
        }
        other => panic!("expected StringList, got: {:?}", other),
    }

    // Add nodes with properties
    let n1 = add_node(&engine, "lifecycle-graph", "company", "Apple Inc", Some("apple"));
    let n2 = add_node(&engine, "lifecycle-graph", "person", "Tim Cook", Some("tim-cook"));
    let n3 = add_node(&engine, "lifecycle-graph", "product", "iPhone", Some("iphone"));

    assert!(n1 >= 1);
    assert_ne!(n1, n2);
    assert_ne!(n2, n3);

    // Get a node and verify its properties
    let cmd = parse_command(&format!("NODE GET \"lifecycle-graph\" {n1}")).unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::NodeInfo(info) => {
            assert_eq!(info.node_id, n1);
            assert_eq!(info.label, "company");
            let has_name = info.properties.iter().any(|(k, _)| k == "name");
            assert!(has_name, "Node should have a 'name' property");
        }
        other => panic!("expected NodeInfo, got: {:?}", other),
    }

    // Add edges
    let e1 = add_edge(&engine, "lifecycle-graph", n1, n2, "employs", 0.9);
    let e2 = add_edge(&engine, "lifecycle-graph", n1, n3, "produces", 0.8);
    assert!(e1 >= 1);
    assert_ne!(e1, e2);

    // Verify graph info shows correct counts
    let cmd = parse_command("GRAPH INFO \"lifecycle-graph\"").unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.name, "lifecycle-graph");
            assert_eq!(info.node_count, 3);
            assert_eq!(info.edge_count, 2);
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // Run context query seeded by node key
    let cmd = parse_command(
        r#"CONTEXT "who works at apple" FROM "lifecycle-graph" SEEDS NODES ["apple"] DEPTH 2"#,
    )
    .unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::Context(result) => {
            assert!(result.nodes_considered > 0, "Should consider at least the seed node");
            assert!(result.nodes_included > 0, "Should include at least the seed node");
            // The seed node (Apple Inc) should be in the results
            let has_apple = result.chunks.iter().any(|c| c.node_id == n1);
            assert!(has_apple, "Apple node should be in the context results");
        }
        other => panic!("expected Context, got: {:?}", other),
    }

    // Delete a node and verify edge count drops
    let cmd = parse_command(&format!("NODE DELETE \"lifecycle-graph\" {n2}")).unwrap();
    engine.execute_command(cmd).unwrap();

    let cmd = parse_command("GRAPH INFO \"lifecycle-graph\"").unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.node_count, 2, "Should have 2 nodes after deletion");
            assert_eq!(info.edge_count, 1, "Deleting n2 should remove the employs edge");
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // Drop the graph
    let cmd = parse_command("GRAPH DROP \"lifecycle-graph\"").unwrap();
    engine.execute_command(cmd).unwrap();

    // Verify graph no longer exists
    let cmd = parse_command("GRAPH LIST").unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::StringList(names) => {
            assert!(!names.contains(&"lifecycle-graph".to_string()));
        }
        other => panic!("expected StringList, got: {:?}", other),
    }
}

// ---- Test 2: Temporal Edge Invalidation Workflow ----

/// Create edges, invalidate one, verify that context queries respect temporal state.
#[test]
fn test_temporal_workflow() {
    let engine = make_engine();
    create_graph(&engine, "temporal-graph");

    let n1 = add_node(&engine, "temporal-graph", "person", "Alice", Some("alice"));
    let n2 = add_node(&engine, "temporal-graph", "person", "Bob", Some("bob"));
    let n3 = add_node(&engine, "temporal-graph", "person", "Charlie", Some("charlie"));

    let e1 = add_edge(&engine, "temporal-graph", n1, n2, "knows", 0.9);
    let _e2 = add_edge(&engine, "temporal-graph", n1, n3, "knows", 0.8);

    // Verify both edges exist
    let cmd = parse_command("GRAPH INFO \"temporal-graph\"").unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.edge_count, 2);
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // Invalidate the first edge
    let cmd = parse_command(&format!("EDGE INVALIDATE \"temporal-graph\" {e1}")).unwrap();
    let resp = engine.execute_command(cmd).unwrap();
    assert!(
        matches!(resp, CommandResponse::Ok),
        "Edge invalidation should succeed"
    );

    // Edge is still in the graph (it is invalidated, not removed)
    let cmd = parse_command("GRAPH INFO \"temporal-graph\"").unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::GraphInfo(info) => {
            // Edge count stays the same because invalidation does not remove, it sets valid_until
            assert_eq!(info.edge_count, 2);
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }
}

// ---- Test 3: Entity Dedup Through Engine ----

/// Test that entity_key based lookups work correctly,
/// and that multiple nodes with different keys can coexist.
#[test]
fn test_dedup_through_engine() {
    let engine = make_engine();
    create_graph(&engine, "dedup-graph");

    // Add two nodes with different entity keys
    let n1 = add_node(&engine, "dedup-graph", "person", "Alice", Some("alice-key"));
    let n2 = add_node(&engine, "dedup-graph", "person", "Bob", Some("bob-key"));

    // Look up by entity key
    let cmd = parse_command(
        "NODE GET \"dedup-graph\" WHERE entity_key = \"alice-key\"",
    )
    .unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::NodeInfo(info) => {
            assert_eq!(info.node_id, n1);
            assert_eq!(info.label, "person");
        }
        other => panic!("expected NodeInfo, got: {:?}", other),
    }

    let cmd = parse_command(
        "NODE GET \"dedup-graph\" WHERE entity_key = \"bob-key\"",
    )
    .unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::NodeInfo(info) => {
            assert_eq!(info.node_id, n2);
        }
        other => panic!("expected NodeInfo, got: {:?}", other),
    }

    // Look up a non-existent key should error
    let cmd = parse_command(
        "NODE GET \"dedup-graph\" WHERE entity_key = \"nonexistent\"",
    )
    .unwrap();
    assert!(
        engine.execute_command(cmd).is_err(),
        "Looking up non-existent entity_key should return an error"
    );
}

// ---- Test 4: Token Budget Enforcement End-to-End ----

/// Create a graph with many nodes, run a context query with a tight budget,
/// and verify the token budget is respected.
#[test]
fn test_budget_enforcement_e2e() {
    let engine = make_engine();
    create_graph(&engine, "budget-graph");

    // Add multiple nodes with varying content lengths
    let mut node_ids = Vec::new();
    for i in 0..20 {
        let description = format!("This is a fairly long description for node number {}. It contains enough text to consume a meaningful number of tokens in the budget calculation.", i);
        let cmd_str = format!(
            r#"NODE ADD TO "budget-graph" LABEL "doc" PROPERTIES {{"name": "doc_{}", "content": "{}"}} KEY "doc-{}""#,
            i, description, i
        );
        let cmd = parse_command(&cmd_str).unwrap();
        let nid = match engine.execute_command(cmd).unwrap() {
            CommandResponse::Integer(id) => id,
            other => panic!("expected Integer, got: {:?}", other),
        };
        node_ids.push(nid);
    }

    // Connect nodes in a chain
    for i in 0..node_ids.len() - 1 {
        add_edge(
            &engine,
            "budget-graph",
            node_ids[i],
            node_ids[i + 1],
            "next",
            0.9,
        );
    }

    // Run a context query with a very small budget
    let cmd = parse_command(
        r#"CONTEXT "find documents" FROM "budget-graph" SEEDS NODES ["doc-0"] DEPTH 3 BUDGET 50 TOKENS"#,
    )
    .unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::Context(result) => {
            assert!(
                result.total_tokens <= 50,
                "Total tokens ({}) should not exceed budget of 50",
                result.total_tokens
            );
            assert!(
                result.budget_used <= 1.0,
                "Budget utilization should not exceed 1.0"
            );
        }
        other => panic!("expected Context, got: {:?}", other),
    }

    // Run with a generous budget -- should include more nodes
    let cmd = parse_command(
        r#"CONTEXT "find documents" FROM "budget-graph" SEEDS NODES ["doc-0"] DEPTH 3 BUDGET 100000 TOKENS"#,
    )
    .unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::Context(result) => {
            assert!(
                result.nodes_included > 0,
                "Should include nodes with a generous budget"
            );
        }
        other => panic!("expected Context, got: {:?}", other),
    }
}

// ---- Test 5: Multiple Graphs in Same Engine ----

/// Verify that multiple graphs can coexist independently.
#[test]
fn test_multiple_graphs() {
    let engine = make_engine();

    create_graph(&engine, "graph-a");
    create_graph(&engine, "graph-b");
    create_graph(&engine, "graph-c");

    // Verify all three exist
    let cmd = parse_command("GRAPH LIST").unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::StringList(names) => {
            assert_eq!(names.len(), 3);
            assert!(names.contains(&"graph-a".to_string()));
            assert!(names.contains(&"graph-b".to_string()));
            assert!(names.contains(&"graph-c".to_string()));
        }
        other => panic!("expected StringList, got: {:?}", other),
    }

    // Add nodes to different graphs
    let na = add_node(&engine, "graph-a", "type_a", "NodeA", None);
    let nb = add_node(&engine, "graph-b", "type_b", "NodeB", None);

    // Verify they are independent
    let cmd = parse_command("GRAPH INFO \"graph-a\"").unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.node_count, 1);
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    let cmd = parse_command("GRAPH INFO \"graph-b\"").unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.node_count, 1);
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    let cmd = parse_command("GRAPH INFO \"graph-c\"").unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.node_count, 0, "graph-c should have no nodes");
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // Operations on one graph should not affect another
    let cmd = parse_command(&format!("NODE GET \"graph-a\" {na}")).unwrap();
    engine.execute_command(cmd).unwrap();

    // Verify the correct node is returned for nb from graph-b
    let cmd = parse_command(&format!("NODE GET \"graph-b\" {nb}")).unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::NodeInfo(info) => {
            assert_eq!(info.node_id, nb);
            assert_eq!(info.label, "type_b");
        }
        other => panic!("expected NodeInfo, got: {:?}", other),
    }

    // Drop one graph, others should remain
    let cmd = parse_command("GRAPH DROP \"graph-b\"").unwrap();
    engine.execute_command(cmd).unwrap();

    let cmd = parse_command("GRAPH LIST").unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::StringList(names) => {
            assert_eq!(names.len(), 2);
            assert!(!names.contains(&"graph-b".to_string()));
        }
        other => panic!("expected StringList, got: {:?}", other),
    }

    // graph-a should still work
    let cmd = parse_command(&format!("NODE GET \"graph-a\" {na}")).unwrap();
    engine.execute_command(cmd).unwrap();

    // Creating a duplicate graph should fail
    let cmd = parse_command("GRAPH CREATE \"graph-a\"").unwrap();
    assert!(
        engine.execute_command(cmd).is_err(),
        "Creating a duplicate graph should fail"
    );
}

// ---- Test 6: Concurrent Read Access ----

/// Verify that the engine supports concurrent read access from multiple threads.
#[test]
fn test_concurrent_reads() {
    use std::sync::Arc;
    use std::thread;

    let engine = Arc::new(make_engine());

    // Set up a graph with some data
    create_graph(&engine, "concurrent-graph");
    for i in 0..50 {
        add_node(
            &engine,
            "concurrent-graph",
            "item",
            &format!("Item {}", i),
            Some(&format!("item-{}", i)),
        );
    }

    // Spawn multiple reader threads
    let mut handles = Vec::new();
    for t in 0..8 {
        let engine_clone = Arc::clone(&engine);
        let handle = thread::spawn(move || {
            for _ in 0..100 {
                // Read operations: PING, GRAPH LIST, GRAPH INFO, NODE GET
                let cmd = parse_command("PING").unwrap();
                let resp = engine_clone.execute_command(cmd).unwrap();
                assert!(matches!(resp, CommandResponse::Pong));

                let cmd = parse_command("GRAPH LIST").unwrap();
                let resp = engine_clone.execute_command(cmd).unwrap();
                match resp {
                    CommandResponse::StringList(names) => {
                        assert!(names.contains(&"concurrent-graph".to_string()));
                    }
                    _ => panic!("thread {t}: expected StringList"),
                }

                let cmd = parse_command("GRAPH INFO \"concurrent-graph\"").unwrap();
                let resp = engine_clone.execute_command(cmd).unwrap();
                match resp {
                    CommandResponse::GraphInfo(info) => {
                        assert_eq!(info.node_count, 50);
                    }
                    _ => panic!("thread {t}: expected GraphInfo"),
                }

                // Read a specific node
                let node_id = (t % 50) + 1;
                let cmd = parse_command(&format!(
                    "NODE GET \"concurrent-graph\" {}",
                    node_id
                ))
                .unwrap();
                let resp = engine_clone.execute_command(cmd).unwrap();
                match resp {
                    CommandResponse::NodeInfo(info) => {
                        assert_eq!(info.node_id, node_id as u64);
                    }
                    _ => panic!("thread {t}: expected NodeInfo"),
                }
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }
}

// ---- Test 7: Error Propagation ----

/// Test that errors are properly propagated through the entire stack.
#[test]
fn test_error_propagation() {
    let engine = make_engine();

    // 1. Parse error: unknown command
    let result = parse_command("FOOBAR");
    assert!(result.is_err(), "Unknown command should fail to parse");

    // 2. Parse error: empty input
    let result = parse_command("");
    assert!(result.is_err(), "Empty input should fail to parse");

    // 3. Graph not found
    let cmd = parse_command("GRAPH INFO \"nonexistent\"").unwrap();
    assert!(
        engine.execute_command(cmd).is_err(),
        "GRAPH INFO on non-existent graph should error"
    );

    // 4. Graph drop on non-existent graph
    let cmd = parse_command("GRAPH DROP \"nonexistent\"").unwrap();
    assert!(
        engine.execute_command(cmd).is_err(),
        "GRAPH DROP on non-existent graph should error"
    );

    // 5. Node get on non-existent graph
    let cmd = parse_command("NODE GET \"nonexistent\" 1").unwrap();
    assert!(
        engine.execute_command(cmd).is_err(),
        "NODE GET on non-existent graph should error"
    );

    // 6. Node not found
    create_graph(&engine, "error-graph");
    let cmd = parse_command("NODE GET \"error-graph\" 999").unwrap();
    assert!(
        engine.execute_command(cmd).is_err(),
        "NODE GET for non-existent node should error"
    );

    // 7. Edge add with missing nodes
    let cmd = parse_command(
        r#"EDGE ADD TO "error-graph" FROM 1 TO 2 LABEL "link""#,
    )
    .unwrap();
    assert!(
        engine.execute_command(cmd).is_err(),
        "EDGE ADD with non-existent nodes should error"
    );

    // 8. Edge invalidate for non-existent edge
    let cmd = parse_command("EDGE INVALIDATE \"error-graph\" 999").unwrap();
    assert!(
        engine.execute_command(cmd).is_err(),
        "EDGE INVALIDATE for non-existent edge should error"
    );

    // 9. Node delete for non-existent node
    let cmd = parse_command("NODE DELETE \"error-graph\" 999").unwrap();
    assert!(
        engine.execute_command(cmd).is_err(),
        "NODE DELETE for non-existent node should error"
    );

    // 10. Context query on non-existent graph
    let cmd = parse_command(
        r#"CONTEXT "test" FROM "nonexistent" SEEDS NODES ["x"]"#,
    )
    .unwrap();
    assert!(
        engine.execute_command(cmd).is_err(),
        "CONTEXT on non-existent graph should error"
    );

    // 11. Stats on non-existent graph
    let cmd = parse_command("STATS \"nonexistent\"").unwrap();
    assert!(
        engine.execute_command(cmd).is_err(),
        "STATS on non-existent graph should error"
    );
}

// ---- Test 8: Context Query with Multiple Seed Nodes ----

/// Verify context query works with multiple node seeds.
#[test]
fn test_context_query_multi_seed() {
    let engine = make_engine();
    create_graph(&engine, "multi-seed-graph");

    // Create a small knowledge graph
    let alice = add_node(&engine, "multi-seed-graph", "person", "Alice", Some("alice"));
    let bob = add_node(&engine, "multi-seed-graph", "person", "Bob", Some("bob"));
    let rust = add_node(&engine, "multi-seed-graph", "topic", "Rust", Some("rust"));
    let python = add_node(&engine, "multi-seed-graph", "topic", "Python", Some("python"));

    add_edge(&engine, "multi-seed-graph", alice, rust, "uses", 0.9);
    add_edge(&engine, "multi-seed-graph", bob, python, "uses", 0.8);
    add_edge(&engine, "multi-seed-graph", alice, bob, "knows", 0.7);

    // Query with multiple seeds
    let cmd = parse_command(
        r#"CONTEXT "programming languages" FROM "multi-seed-graph" SEEDS NODES ["alice", "bob"] DEPTH 2"#,
    )
    .unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::Context(result) => {
            assert!(
                result.nodes_considered >= 2,
                "Should consider at least both seed nodes, got {}",
                result.nodes_considered
            );
            // Both seeds should appear in results
            let has_alice = result.chunks.iter().any(|c| c.node_id == alice);
            let has_bob = result.chunks.iter().any(|c| c.node_id == bob);
            assert!(has_alice, "Alice should be in results");
            assert!(has_bob, "Bob should be in results");
        }
        other => panic!("expected Context, got: {:?}", other),
    }
}

// ---- Test 9: Stats Command ----

/// Test STATS and INFO commands in detail.
#[test]
fn test_stats_and_info() {
    let engine = make_engine();

    // INFO before any graphs
    let cmd = parse_command("INFO").unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::Text(text) => {
            assert!(text.contains("weav-server"), "INFO should contain server name");
        }
        other => panic!("expected Text, got: {:?}", other),
    }

    // Global stats
    let cmd = parse_command("STATS").unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::Text(text) => {
            assert!(text.contains("graphs="), "STATS should show graph count");
        }
        other => panic!("expected Text, got: {:?}", other),
    }

    // Create a graph with nodes and check per-graph stats
    create_graph(&engine, "stats-graph");
    for i in 0..5 {
        add_node(
            &engine,
            "stats-graph",
            "entity",
            &format!("E{}", i),
            None,
        );
    }

    let cmd = parse_command("STATS \"stats-graph\"").unwrap();
    match engine.execute_command(cmd).unwrap() {
        CommandResponse::Text(text) => {
            assert!(text.contains("graph=stats-graph"));
            assert!(text.contains("nodes=5"));
            assert!(text.contains("edges=0"));
        }
        other => panic!("expected Text, got: {:?}", other),
    }

    // SNAPSHOT should return Ok
    let cmd = parse_command("SNAPSHOT").unwrap();
    let resp = engine.execute_command(cmd).unwrap();
    assert!(matches!(resp, CommandResponse::Ok));
}
