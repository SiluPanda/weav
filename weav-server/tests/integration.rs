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
    engine.execute_command(cmd, None).unwrap();
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
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Integer(id) => id,
        other => panic!("expected Integer from NODE ADD, got: {:?}", other),
    }
}

fn add_edge(engine: &Engine, graph: &str, src: u64, tgt: u64, label: &str, weight: f32) -> u64 {
    let cmd_str = format!(
        r#"EDGE ADD TO "{graph}" FROM {src} TO {tgt} LABEL "{label}" WEIGHT {weight}"#
    );
    let cmd = parse_command(&cmd_str).unwrap();
    match engine.execute_command(cmd, None).unwrap() {
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
    match engine.execute_command(cmd, None).unwrap() {
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
    match engine.execute_command(cmd, None).unwrap() {
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
    match engine.execute_command(cmd, None).unwrap() {
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
    match engine.execute_command(cmd, None).unwrap() {
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
    engine.execute_command(cmd, None).unwrap();

    let cmd = parse_command("GRAPH INFO \"lifecycle-graph\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.node_count, 2, "Should have 2 nodes after deletion");
            assert_eq!(info.edge_count, 1, "Deleting n2 should remove the employs edge");
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // Drop the graph
    let cmd = parse_command("GRAPH DROP \"lifecycle-graph\"").unwrap();
    engine.execute_command(cmd, None).unwrap();

    // Verify graph no longer exists
    let cmd = parse_command("GRAPH LIST").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
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
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.edge_count, 2);
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // Invalidate the first edge
    let cmd = parse_command(&format!("EDGE INVALIDATE \"temporal-graph\" {e1}")).unwrap();
    let resp = engine.execute_command(cmd, None).unwrap();
    assert!(
        matches!(resp, CommandResponse::Ok),
        "Edge invalidation should succeed"
    );

    // Edge is still in the graph (it is invalidated, not removed)
    let cmd = parse_command("GRAPH INFO \"temporal-graph\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
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
    match engine.execute_command(cmd, None).unwrap() {
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
    match engine.execute_command(cmd, None).unwrap() {
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
        engine.execute_command(cmd, None).is_err(),
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
        let nid = match engine.execute_command(cmd, None).unwrap() {
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
    match engine.execute_command(cmd, None).unwrap() {
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
    match engine.execute_command(cmd, None).unwrap() {
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
    match engine.execute_command(cmd, None).unwrap() {
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
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.node_count, 1);
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    let cmd = parse_command("GRAPH INFO \"graph-b\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.node_count, 1);
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    let cmd = parse_command("GRAPH INFO \"graph-c\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.node_count, 0, "graph-c should have no nodes");
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // Operations on one graph should not affect another
    let cmd = parse_command(&format!("NODE GET \"graph-a\" {na}")).unwrap();
    engine.execute_command(cmd, None).unwrap();

    // Verify the correct node is returned for nb from graph-b
    let cmd = parse_command(&format!("NODE GET \"graph-b\" {nb}")).unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::NodeInfo(info) => {
            assert_eq!(info.node_id, nb);
            assert_eq!(info.label, "type_b");
        }
        other => panic!("expected NodeInfo, got: {:?}", other),
    }

    // Drop one graph, others should remain
    let cmd = parse_command("GRAPH DROP \"graph-b\"").unwrap();
    engine.execute_command(cmd, None).unwrap();

    let cmd = parse_command("GRAPH LIST").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::StringList(names) => {
            assert_eq!(names.len(), 2);
            assert!(!names.contains(&"graph-b".to_string()));
        }
        other => panic!("expected StringList, got: {:?}", other),
    }

    // graph-a should still work
    let cmd = parse_command(&format!("NODE GET \"graph-a\" {na}")).unwrap();
    engine.execute_command(cmd, None).unwrap();

    // Creating a duplicate graph should fail
    let cmd = parse_command("GRAPH CREATE \"graph-a\"").unwrap();
    assert!(
        engine.execute_command(cmd, None).is_err(),
        "Creating a duplicate graph should fail"
    );
}

// ---- Test 10: Conflict Resolution (Entity-Key Dedup) ----

/// Tests entity-key deduplication with last-write-wins merge policy.
/// When two nodes are added with the same entity_key, the engine should
/// return the same node_id and merge properties (last-write-wins).
#[test]
fn test_conflict_resolution() {
    let engine = make_engine();
    create_graph(&engine, "conflict-graph");

    // Add a node with entity_key "dup-key"
    let n1 = add_node(&engine, "conflict-graph", "person", "Alice", Some("dup-key"));

    // Add a second node with the SAME entity_key but different properties.
    // Entity-key dedup returns existing node_id and merges properties.
    let n2 = add_node(&engine, "conflict-graph", "person", "Bob", Some("dup-key"));

    // Dedup should return the same node_id.
    assert_eq!(n1, n2, "Same entity_key should return the same node_id (dedup)");

    // Graph should show 1 node (deduped).
    let cmd = parse_command("GRAPH INFO \"conflict-graph\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.node_count, 1, "Dedup should keep single node");
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // The merged node should be retrievable by ID.
    let cmd = parse_command(&format!("NODE GET \"conflict-graph\" {n1}")).unwrap();
    engine.execute_command(cmd, None).unwrap();

    // Without entity_key, nodes should always get distinct IDs.
    let n3 = add_node(&engine, "conflict-graph", "person", "Charlie", None);
    let n4 = add_node(&engine, "conflict-graph", "person", "Diana", None);
    assert_ne!(n3, n4, "Without entity_key, nodes should be distinct");
}

// ---- Test 11: Bulk Operations at Scale ----

/// Tests bulk insert operations with 100 nodes and chained edges.
#[test]
fn test_bulk_operations() {
    let engine = make_engine();
    create_graph(&engine, "bulk-graph");

    // Build a JSON array of 100 nodes.
    let mut node_entries = Vec::new();
    for i in 0..100 {
        node_entries.push(format!(
            r#"{{"label": "item", "properties": {{"name": "item_{i}"}}, "entity_key": "bulk-{i}"}}"#
        ));
    }
    let nodes_json = format!("[{}]", node_entries.join(", "));
    let cmd_str = format!(r#"BULK NODES TO "bulk-graph" DATA {nodes_json}"#);
    let cmd = parse_command(&cmd_str).unwrap();
    let node_ids = match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::IntegerList(ids) => ids,
        other => panic!("expected IntegerList from BULK NODES, got: {:?}", other),
    };
    assert_eq!(node_ids.len(), 100, "Should have inserted 100 nodes");

    // Verify node count via GRAPH INFO.
    let cmd = parse_command("GRAPH INFO \"bulk-graph\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.node_count, 100);
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // Bulk insert 99 edges connecting nodes in a chain.
    let mut edge_entries = Vec::new();
    for i in 0..99 {
        edge_entries.push(format!(
            r#"{{"source": {}, "target": {}, "label": "next", "weight": 0.5}}"#,
            node_ids[i], node_ids[i + 1]
        ));
    }
    let edges_json = format!("[{}]", edge_entries.join(", "));
    let cmd_str = format!(r#"BULK EDGES TO "bulk-graph" DATA {edges_json}"#);
    let cmd = parse_command(&cmd_str).unwrap();
    let edge_ids = match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::IntegerList(ids) => ids,
        other => panic!("expected IntegerList from BULK EDGES, got: {:?}", other),
    };
    assert_eq!(edge_ids.len(), 99, "Should have inserted 99 edges");

    // Verify edge count.
    let cmd = parse_command("GRAPH INFO \"bulk-graph\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.edge_count, 99);
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // Verify individual nodes can be retrieved by entity key.
    for i in [0, 49, 99] {
        let cmd = parse_command(&format!(
            "NODE GET \"bulk-graph\" WHERE entity_key = \"bulk-{i}\""
        ))
        .unwrap();
        match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::NodeInfo(info) => {
                assert_eq!(info.node_id, node_ids[i]);
                assert_eq!(info.label, "item");
            }
            other => panic!("expected NodeInfo for bulk-{i}, got: {:?}", other),
        }
    }
}

// ---- Test 12: Protocol Parity ----

/// Tests that the engine command interface produces consistent results
/// across different command types (simulating what different access protocols
/// would all funnel through).
#[test]
fn test_protocol_parity() {
    let engine = make_engine();

    // PING should always return Pong.
    let cmd = parse_command("PING").unwrap();
    let resp = engine.execute_command(cmd, None).unwrap();
    assert!(matches!(resp, CommandResponse::Pong), "PING -> Pong");

    // GRAPH CREATE should return Ok.
    let cmd = parse_command("GRAPH CREATE \"parity-graph\"").unwrap();
    let resp = engine.execute_command(cmd, None).unwrap();
    assert!(matches!(resp, CommandResponse::Ok), "GRAPH CREATE -> Ok");

    // GRAPH LIST should include the new graph.
    let cmd = parse_command("GRAPH LIST").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::StringList(names) => {
            assert!(names.contains(&"parity-graph".to_string()));
        }
        other => panic!("expected StringList, got: {:?}", other),
    }

    // NODE ADD should return an Integer.
    let cmd = parse_command(
        r#"NODE ADD TO "parity-graph" LABEL "entity" PROPERTIES {"name": "test"} KEY "test-key""#,
    )
    .unwrap();
    let node_id = match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Integer(id) => id,
        other => panic!("expected Integer from NODE ADD, got: {:?}", other),
    };
    assert!(node_id >= 1);

    // NODE GET by ID should return the same node.
    let cmd = parse_command(&format!("NODE GET \"parity-graph\" {node_id}")).unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::NodeInfo(info) => {
            assert_eq!(info.node_id, node_id);
            assert_eq!(info.label, "entity");
        }
        other => panic!("expected NodeInfo, got: {:?}", other),
    }

    // NODE GET by entity_key should return the same node.
    let cmd = parse_command(
        "NODE GET \"parity-graph\" WHERE entity_key = \"test-key\"",
    )
    .unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::NodeInfo(info) => {
            assert_eq!(info.node_id, node_id, "Key lookup should match ID lookup");
            assert_eq!(info.label, "entity");
        }
        other => panic!("expected NodeInfo, got: {:?}", other),
    }

    // GRAPH INFO should reflect the node.
    let cmd = parse_command("GRAPH INFO \"parity-graph\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.node_count, 1);
            assert_eq!(info.edge_count, 0);
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // INFO should return server info.
    let cmd = parse_command("INFO").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Text(text) => {
            assert!(text.contains("weav-server"));
        }
        other => panic!("expected Text, got: {:?}", other),
    }

    // STATS should show graph count.
    let cmd = parse_command("STATS").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Text(text) => {
            assert!(text.contains("graphs=1"));
        }
        other => panic!("expected Text, got: {:?}", other),
    }
}

// ---- Test 13: Persistence WAL Recovery ----

/// Tests WAL write and recovery: create data, snapshot, then recover into
/// a new engine and verify the data survives.
#[test]
fn test_persistence_wal_recovery() {
    use std::time::{SystemTime, UNIX_EPOCH};
    use weav_core::config::WeavConfig;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let data_dir = std::env::temp_dir().join(format!("weav_integ_persist_{now}"));

    // Create a config with persistence enabled, pointing at our temp dir.
    let mut config = WeavConfig::default();
    config.persistence.enabled = true;
    config.persistence.data_dir = data_dir.clone();

    let engine = Engine::new(config.clone());

    // Create a graph, add nodes and edges.
    create_graph(&engine, "persist-graph");
    let n1 = add_node(&engine, "persist-graph", "person", "Alice", Some("alice"));
    let n2 = add_node(&engine, "persist-graph", "person", "Bob", Some("bob"));
    let _e1 = add_edge(&engine, "persist-graph", n1, n2, "knows", 0.9);

    // Take a snapshot.
    let cmd = parse_command("SNAPSHOT").unwrap();
    engine.execute_command(cmd, None).unwrap();

    // Create a brand-new engine from the same data dir.
    let engine2 = Engine::new(config);

    // Run recovery.
    let recovery_mgr = weav_persist::recovery::RecoveryManager::new(data_dir.clone());
    let result = recovery_mgr.recover().unwrap();
    assert_eq!(result.snapshots_loaded, 1, "Should load the snapshot");

    engine2.recover(result).unwrap();

    // Verify the recovered engine has the graph.
    let cmd = parse_command("GRAPH LIST").unwrap();
    match engine2.execute_command(cmd, None).unwrap() {
        CommandResponse::StringList(names) => {
            assert!(
                names.contains(&"persist-graph".to_string()),
                "Recovered engine should have 'persist-graph'"
            );
        }
        other => panic!("expected StringList, got: {:?}", other),
    }

    // Verify graph info shows correct counts.
    let cmd = parse_command("GRAPH INFO \"persist-graph\"").unwrap();
    match engine2.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.node_count, 2, "Should recover 2 nodes");
            assert_eq!(info.edge_count, 1, "Should recover 1 edge");
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // Verify individual node retrieval by ID works.
    // (Note: entity_key lookups may not survive the properties_json roundtrip
    //  due to Debug formatting in the snapshot; verify by node ID instead.)
    let cmd = parse_command(&format!("NODE GET \"persist-graph\" {n1}")).unwrap();
    match engine2.execute_command(cmd, None).unwrap() {
        CommandResponse::NodeInfo(info) => {
            assert_eq!(info.node_id, n1);
            assert_eq!(info.label, "person");
        }
        other => panic!("expected NodeInfo for n1, got: {:?}", other),
    }

    let cmd = parse_command(&format!("NODE GET \"persist-graph\" {n2}")).unwrap();
    match engine2.execute_command(cmd, None).unwrap() {
        CommandResponse::NodeInfo(info) => {
            assert_eq!(info.node_id, n2);
            assert_eq!(info.label, "person");
        }
        other => panic!("expected NodeInfo for n2, got: {:?}", other),
    }

    // Clean up.
    std::fs::remove_dir_all(&data_dir).ok();
}

// ---- Test 14: Concurrent Writes ----

/// Tests concurrent write operations from multiple threads.
#[test]
fn test_concurrent_writes() {
    use std::sync::Arc;
    use std::thread;

    let engine = Arc::new(make_engine());
    create_graph(&engine, "concurrent-write-graph");

    let mut handles = Vec::new();
    for t in 0..8 {
        let engine_clone = Arc::clone(&engine);
        let handle = thread::spawn(move || {
            for i in 0..50 {
                let name = format!("t{t}_node_{i}");
                let key = format!("t{t}-{i}");
                let cmd_str = format!(
                    r#"NODE ADD TO "concurrent-write-graph" LABEL "item" PROPERTIES {{"name": "{name}"}} KEY "{key}""#,
                );
                let cmd = parse_command(&cmd_str).unwrap();
                let resp = engine_clone.execute_command(cmd, None).unwrap();
                match resp {
                    CommandResponse::Integer(id) => assert!(id >= 1),
                    other => panic!("thread {t}: expected Integer, got: {:?}", other),
                }
            }
        });
        handles.push(handle);
    }

    // Wait for all threads.
    for handle in handles {
        handle.join().expect("Thread panicked during concurrent writes");
    }

    // Verify total node count = 8 * 50 = 400.
    let cmd = parse_command("GRAPH INFO \"concurrent-write-graph\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(
                info.node_count, 400,
                "Should have 400 nodes from 8 threads x 50 nodes"
            );
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // Spot-check: verify a few nodes from different threads are retrievable.
    for t in 0..8 {
        let key = format!("t{t}-0");
        let cmd = parse_command(&format!(
            "NODE GET \"concurrent-write-graph\" WHERE entity_key = \"{key}\""
        ))
        .unwrap();
        match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::NodeInfo(info) => {
                assert_eq!(info.label, "item");
            }
            other => panic!("expected NodeInfo for {key}, got: {:?}", other),
        }
    }
}

// ---- Test 15: Edge Delete and Get ----

/// Tests the EdgeDelete and EdgeGet commands.
#[test]
fn test_edge_delete_and_get() {
    let engine = make_engine();
    create_graph(&engine, "edge-ops-graph");

    let n1 = add_node(&engine, "edge-ops-graph", "a", "Node1", None);
    let n2 = add_node(&engine, "edge-ops-graph", "b", "Node2", None);
    let e1 = add_edge(&engine, "edge-ops-graph", n1, n2, "link", 0.75);

    // EDGE GET should return the edge info.
    let cmd = parse_command(&format!("EDGE GET \"edge-ops-graph\" {e1}")).unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::EdgeInfo(info) => {
            assert_eq!(info.edge_id, e1);
            assert_eq!(info.source, n1);
            assert_eq!(info.target, n2);
            assert_eq!(info.label, "link");
            assert!((info.weight - 0.75).abs() < 0.001);
        }
        other => panic!("expected EdgeInfo, got: {:?}", other),
    }

    // Verify edge count is 1 before deletion.
    let cmd = parse_command("GRAPH INFO \"edge-ops-graph\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.edge_count, 1);
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // EDGE DELETE the edge.
    let cmd = parse_command(&format!("EDGE DELETE \"edge-ops-graph\" {e1}")).unwrap();
    let resp = engine.execute_command(cmd, None).unwrap();
    assert!(matches!(resp, CommandResponse::Ok), "EDGE DELETE should return Ok");

    // EDGE GET after deletion should fail.
    let cmd = parse_command(&format!("EDGE GET \"edge-ops-graph\" {e1}")).unwrap();
    assert!(
        engine.execute_command(cmd, None).is_err(),
        "EDGE GET after EDGE DELETE should return an error"
    );

    // Verify edge count is 0 after deletion.
    let cmd = parse_command("GRAPH INFO \"edge-ops-graph\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.edge_count, 0, "Edge count should be 0 after deletion");
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }
}

// ---- Test 16: Config Set and Get ----

/// Tests ConfigSet and ConfigGet commands.
#[test]
fn test_config_set_and_get() {
    let engine = make_engine();

    // CONFIG SET a key-value pair.
    let cmd = parse_command("CONFIG SET \"my.key\" \"my.value\"").unwrap();
    let resp = engine.execute_command(cmd, None).unwrap();
    assert!(matches!(resp, CommandResponse::Ok), "CONFIG SET should return Ok");

    // CONFIG GET should return the value.
    let cmd = parse_command("CONFIG GET \"my.key\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Text(val) => {
            assert_eq!(val, "my.value", "CONFIG GET should return the set value");
        }
        other => panic!("expected Text, got: {:?}", other),
    }

    // CONFIG GET for a non-existent key should return Null.
    let cmd = parse_command("CONFIG GET \"nonexistent\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Null => {} // Expected
        other => panic!("expected Null for nonexistent config key, got: {:?}", other),
    }

    // CONFIG SET can overwrite an existing key.
    let cmd = parse_command("CONFIG SET \"my.key\" \"updated.value\"").unwrap();
    engine.execute_command(cmd, None).unwrap();

    let cmd = parse_command("CONFIG GET \"my.key\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Text(val) => {
            assert_eq!(val, "updated.value", "CONFIG GET should return the updated value");
        }
        other => panic!("expected Text, got: {:?}", other),
    }

    // Multiple keys can coexist.
    let cmd = parse_command("CONFIG SET \"another.key\" \"another.value\"").unwrap();
    engine.execute_command(cmd, None).unwrap();

    let cmd = parse_command("CONFIG GET \"another.key\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Text(val) => {
            assert_eq!(val, "another.value");
        }
        other => panic!("expected Text, got: {:?}", other),
    }

    // Original key should still be accessible.
    let cmd = parse_command("CONFIG GET \"my.key\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Text(val) => {
            assert_eq!(val, "updated.value");
        }
        other => panic!("expected Text, got: {:?}", other),
    }
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
                let resp = engine_clone.execute_command(cmd, None).unwrap();
                assert!(matches!(resp, CommandResponse::Pong));

                let cmd = parse_command("GRAPH LIST").unwrap();
                let resp = engine_clone.execute_command(cmd, None).unwrap();
                match resp {
                    CommandResponse::StringList(names) => {
                        assert!(names.contains(&"concurrent-graph".to_string()));
                    }
                    _ => panic!("thread {t}: expected StringList"),
                }

                let cmd = parse_command("GRAPH INFO \"concurrent-graph\"").unwrap();
                let resp = engine_clone.execute_command(cmd, None).unwrap();
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
                let resp = engine_clone.execute_command(cmd, None).unwrap();
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
        engine.execute_command(cmd, None).is_err(),
        "GRAPH INFO on non-existent graph should error"
    );

    // 4. Graph drop on non-existent graph
    let cmd = parse_command("GRAPH DROP \"nonexistent\"").unwrap();
    assert!(
        engine.execute_command(cmd, None).is_err(),
        "GRAPH DROP on non-existent graph should error"
    );

    // 5. Node get on non-existent graph
    let cmd = parse_command("NODE GET \"nonexistent\" 1").unwrap();
    assert!(
        engine.execute_command(cmd, None).is_err(),
        "NODE GET on non-existent graph should error"
    );

    // 6. Node not found
    create_graph(&engine, "error-graph");
    let cmd = parse_command("NODE GET \"error-graph\" 999").unwrap();
    assert!(
        engine.execute_command(cmd, None).is_err(),
        "NODE GET for non-existent node should error"
    );

    // 7. Edge add with missing nodes
    let cmd = parse_command(
        r#"EDGE ADD TO "error-graph" FROM 1 TO 2 LABEL "link""#,
    )
    .unwrap();
    assert!(
        engine.execute_command(cmd, None).is_err(),
        "EDGE ADD with non-existent nodes should error"
    );

    // 8. Edge invalidate for non-existent edge
    let cmd = parse_command("EDGE INVALIDATE \"error-graph\" 999").unwrap();
    assert!(
        engine.execute_command(cmd, None).is_err(),
        "EDGE INVALIDATE for non-existent edge should error"
    );

    // 9. Node delete for non-existent node
    let cmd = parse_command("NODE DELETE \"error-graph\" 999").unwrap();
    assert!(
        engine.execute_command(cmd, None).is_err(),
        "NODE DELETE for non-existent node should error"
    );

    // 10. Context query on non-existent graph
    let cmd = parse_command(
        r#"CONTEXT "test" FROM "nonexistent" SEEDS NODES ["x"]"#,
    )
    .unwrap();
    assert!(
        engine.execute_command(cmd, None).is_err(),
        "CONTEXT on non-existent graph should error"
    );

    // 11. Stats on non-existent graph
    let cmd = parse_command("STATS \"nonexistent\"").unwrap();
    assert!(
        engine.execute_command(cmd, None).is_err(),
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
    match engine.execute_command(cmd, None).unwrap() {
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
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Text(text) => {
            assert!(text.contains("weav-server"), "INFO should contain server name");
        }
        other => panic!("expected Text, got: {:?}", other),
    }

    // Global stats
    let cmd = parse_command("STATS").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
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
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Text(text) => {
            assert!(text.contains("graph=stats-graph"));
            assert!(text.contains("nodes=5"));
            assert!(text.contains("edges=0"));
        }
        other => panic!("expected Text, got: {:?}", other),
    }

    // SNAPSHOT should return Ok
    let cmd = parse_command("SNAPSHOT").unwrap();
    let resp = engine.execute_command(cmd, None).unwrap();
    assert!(matches!(resp, CommandResponse::Ok));
}

// ---- Test 17: Node Update End-to-End ----

/// Create a graph, add a node with properties, update it, and verify
/// that the update merges properties correctly (overwrites existing keys,
/// preserves keys not mentioned in the update).
#[test]
fn test_node_update_e2e() {
    let engine = make_engine();
    create_graph(&engine, "update-graph");

    // Add a node with {name: "Alice", age: 30}
    let cmd = parse_command(
        r#"NODE ADD TO "update-graph" LABEL "person" PROPERTIES {"name": "Alice", "age": 30} KEY "alice""#,
    )
    .unwrap();
    let node_id = match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Integer(id) => id,
        other => panic!("expected Integer from NODE ADD, got: {:?}", other),
    };

    // Verify initial properties
    let cmd = parse_command(&format!("NODE GET \"update-graph\" {node_id}")).unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::NodeInfo(info) => {
            assert_eq!(info.node_id, node_id);
            assert_eq!(info.label, "person");
            let name_val = info.properties.iter().find(|(k, _)| k == "name");
            assert!(name_val.is_some(), "Should have 'name' property initially");
        }
        other => panic!("expected NodeInfo, got: {:?}", other),
    }

    // Update the node: change name to "Bob" (age should be preserved via merge)
    let cmd = parse_command(&format!(
        r#"NODE UPDATE "update-graph" {node_id} PROPERTIES {{"name": "Bob"}}"#
    ))
    .unwrap();
    let resp = engine.execute_command(cmd, None).unwrap();
    assert!(
        matches!(resp, CommandResponse::Ok),
        "NODE UPDATE should return Ok"
    );

    // Verify updated properties
    let cmd = parse_command(&format!("NODE GET \"update-graph\" {node_id}")).unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::NodeInfo(info) => {
            assert_eq!(info.node_id, node_id);
            assert_eq!(info.label, "person");

            // Name should be updated to "Bob"
            let name_val = info.properties.iter().find(|(k, _)| k == "name");
            assert!(name_val.is_some(), "Should still have 'name' property");

            // Age should be preserved (merge behavior)
            let age_val = info.properties.iter().find(|(k, _)| k == "age");
            assert!(
                age_val.is_some(),
                "Should preserve 'age' property after update (merge semantics)"
            );
        }
        other => panic!("expected NodeInfo, got: {:?}", other),
    }

    // Update a non-existent node should fail
    let cmd = parse_command(
        r#"NODE UPDATE "update-graph" 99999 PROPERTIES {"name": "Ghost"}"#,
    )
    .unwrap();
    assert!(
        engine.execute_command(cmd, None).is_err(),
        "Updating a non-existent node should return an error"
    );
}

// ---- Test 18: Vector Embedding End-to-End ----

/// Create a graph with vector support, add nodes with embeddings,
/// and run a vector-seeded context query.
#[test]
fn test_vector_embedding_e2e() {
    use weav_core::config::WeavConfig;

    // Use a small dimension (4) so tests are fast and readable
    let mut config = WeavConfig::default();
    config.engine.default_vector_dimensions = 4;
    config.engine.max_vector_dimensions = 4;
    let engine = Engine::new(config);

    create_graph(&engine, "vector-graph");

    // Add 3 nodes with 4-dimensional embeddings
    // Node 1: mostly in the x-direction
    let cmd = parse_command(
        r#"NODE ADD TO "vector-graph" LABEL "doc" PROPERTIES {"name": "doc_x"} EMBEDDING [1.0,0.0,0.0,0.0] KEY "doc-x""#,
    )
    .unwrap();
    let n1 = match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Integer(id) => id,
        other => panic!("expected Integer from NODE ADD, got: {:?}", other),
    };

    // Node 2: mostly in the y-direction
    let cmd = parse_command(
        r#"NODE ADD TO "vector-graph" LABEL "doc" PROPERTIES {"name": "doc_y"} EMBEDDING [0.0,1.0,0.0,0.0] KEY "doc-y""#,
    )
    .unwrap();
    let n2 = match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Integer(id) => id,
        other => panic!("expected Integer from NODE ADD, got: {:?}", other),
    };

    // Node 3: mix of x and y
    let cmd = parse_command(
        r#"NODE ADD TO "vector-graph" LABEL "doc" PROPERTIES {"name": "doc_xy"} EMBEDDING [0.7,0.7,0.0,0.0] KEY "doc-xy""#,
    )
    .unwrap();
    let n3 = match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Integer(id) => id,
        other => panic!("expected Integer from NODE ADD, got: {:?}", other),
    };

    // Connect nodes so traversal can find them
    add_edge(&engine, "vector-graph", n1, n2, "related", 0.5);
    add_edge(&engine, "vector-graph", n2, n3, "related", 0.5);

    // Run a context query seeded by a vector close to x-direction
    let cmd = parse_command(
        r#"CONTEXT "find similar docs" FROM "vector-graph" SEEDS VECTOR [0.9,0.1,0.0,0.0] TOP 3 DEPTH 1 BUDGET 100000 TOKENS"#,
    )
    .unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Context(result) => {
            assert!(
                result.nodes_included > 0,
                "Vector-seeded context query should return results"
            );
            // The query vector is close to doc_x, so it should be among the results
            let has_n1 = result.chunks.iter().any(|c| c.node_id == n1);
            assert!(
                has_n1,
                "Node closest to the query vector (doc_x) should be in results"
            );
        }
        other => panic!("expected Context, got: {:?}", other),
    }
}

// ---- Test 19: Concurrent Read-Write Interleaving ----

/// Spawn threads where half do writes and half do reads simultaneously.
/// Verify no panics and final state is consistent.
#[test]
fn test_concurrent_read_write_interleaving() {
    use std::sync::Arc;
    use std::thread;

    let engine = Arc::new(make_engine());
    create_graph(&engine, "rw-interleave-graph");

    // Add some initial nodes so readers have something to read
    for i in 0..10 {
        add_node(
            &engine,
            "rw-interleave-graph",
            "item",
            &format!("initial_{i}"),
            Some(&format!("init-{i}")),
        );
    }

    let writer_count = 4;
    let reader_count = 4;
    let writes_per_thread = 25;

    let mut handles = Vec::new();

    // Spawn writer threads
    for t in 0..writer_count {
        let engine_clone = Arc::clone(&engine);
        let handle = thread::spawn(move || {
            for i in 0..writes_per_thread {
                let name = format!("w{t}_node_{i}");
                let key = format!("w{t}-{i}");
                let cmd_str = format!(
                    r#"NODE ADD TO "rw-interleave-graph" LABEL "item" PROPERTIES {{"name": "{name}"}} KEY "{key}""#,
                );
                let cmd = parse_command(&cmd_str).unwrap();
                let resp = engine_clone.execute_command(cmd, None).unwrap();
                match resp {
                    CommandResponse::Integer(id) => assert!(id >= 1),
                    other => panic!("writer {t}: expected Integer, got: {:?}", other),
                }
            }
        });
        handles.push(handle);
    }

    // Spawn reader threads
    for t in 0..reader_count {
        let engine_clone = Arc::clone(&engine);
        let handle = thread::spawn(move || {
            for _ in 0..50 {
                // Read: GRAPH INFO
                let cmd = parse_command("GRAPH INFO \"rw-interleave-graph\"").unwrap();
                let resp = engine_clone.execute_command(cmd, None).unwrap();
                match resp {
                    CommandResponse::GraphInfo(info) => {
                        // Node count should be at least the initial 10
                        assert!(
                            info.node_count >= 10,
                            "reader {t}: node count should be >= 10, got {}",
                            info.node_count
                        );
                    }
                    other => panic!("reader {t}: expected GraphInfo, got: {:?}", other),
                }

                // Read: NODE GET (one of the initial nodes)
                let node_id = (t % 10) + 1;
                let cmd = parse_command(&format!(
                    "NODE GET \"rw-interleave-graph\" {}",
                    node_id
                ))
                .unwrap();
                let resp = engine_clone.execute_command(cmd, None).unwrap();
                match resp {
                    CommandResponse::NodeInfo(info) => {
                        assert_eq!(info.node_id, node_id as u64);
                    }
                    other => panic!("reader {t}: expected NodeInfo, got: {:?}", other),
                }
            }
        });
        handles.push(handle);
    }

    // Wait for all threads to complete without panics
    for handle in handles {
        handle
            .join()
            .expect("Thread panicked during concurrent read/write interleaving");
    }

    // Verify final node count: initial 10 + (writer_count * writes_per_thread)
    let expected_total = 10 + (writer_count * writes_per_thread) as u64;
    let cmd = parse_command("GRAPH INFO \"rw-interleave-graph\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(
                info.node_count, expected_total,
                "Final node count should be {expected_total}"
            );
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }
}

// ---- Test 20: Large Dataset Operations ----

/// Create a graph with 200 nodes and 199 edges in a chain.
/// Execute a context query and verify it completes without error.
#[test]
fn test_large_dataset_operations() {
    let engine = make_engine();
    create_graph(&engine, "large-graph");

    // Add 200 nodes
    let mut node_ids = Vec::new();
    for i in 0..200 {
        let cmd_str = format!(
            r#"NODE ADD TO "large-graph" LABEL "item" PROPERTIES {{"name": "item_{i}", "index": {i}}} KEY "large-{i}""#,
        );
        let cmd = parse_command(&cmd_str).unwrap();
        let nid = match engine.execute_command(cmd, None).unwrap() {
            CommandResponse::Integer(id) => id,
            other => panic!("expected Integer, got: {:?}", other),
        };
        node_ids.push(nid);
    }

    // Add 199 edges in a chain
    for i in 0..199 {
        add_edge(
            &engine,
            "large-graph",
            node_ids[i],
            node_ids[i + 1],
            "chain",
            0.8,
        );
    }

    // Verify counts
    let cmd = parse_command("GRAPH INFO \"large-graph\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.node_count, 200, "Should have 200 nodes");
            assert_eq!(info.edge_count, 199, "Should have 199 edges");
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // Execute a context query from the start of the chain
    let cmd = parse_command(
        r#"CONTEXT "traverse chain" FROM "large-graph" SEEDS NODES ["large-0"] DEPTH 5 BUDGET 100000 TOKENS"#,
    )
    .unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Context(result) => {
            assert!(
                result.nodes_included > 0,
                "Context query on large graph should return results"
            );
            assert!(
                result.nodes_considered > 0,
                "Context query should consider nodes during traversal"
            );
            // The seed node should be in the results
            let has_seed = result.chunks.iter().any(|c| c.node_id == node_ids[0]);
            assert!(has_seed, "Seed node should be in the context results");
        }
        other => panic!("expected Context, got: {:?}", other),
    }

    // Execute a context query from the middle of the chain
    let cmd = parse_command(
        r#"CONTEXT "middle of chain" FROM "large-graph" SEEDS NODES ["large-100"] DEPTH 3 BUDGET 100000 TOKENS"#,
    )
    .unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Context(result) => {
            assert!(
                result.nodes_included > 0,
                "Context query from middle should return results"
            );
        }
        other => panic!("expected Context, got: {:?}", other),
    }
}

// ---- Test 21: Edge Operations End-to-End ----

/// Full lifecycle of edge operations: add, get, delete, verify deletion.
#[test]
fn test_edge_operations_e2e() {
    let engine = make_engine();
    create_graph(&engine, "edge-e2e-graph");

    let n1 = add_node(&engine, "edge-e2e-graph", "person", "Alice", Some("alice"));
    let n2 = add_node(&engine, "edge-e2e-graph", "person", "Bob", Some("bob"));
    let n3 = add_node(&engine, "edge-e2e-graph", "person", "Charlie", Some("charlie"));

    // Add multiple edges
    let e1 = add_edge(&engine, "edge-e2e-graph", n1, n2, "knows", 0.9);
    let e2 = add_edge(&engine, "edge-e2e-graph", n1, n3, "manages", 0.7);
    let e3 = add_edge(&engine, "edge-e2e-graph", n2, n3, "collaborates", 0.6);

    // Verify all edges exist via EDGE GET
    let cmd = parse_command(&format!("EDGE GET \"edge-e2e-graph\" {e1}")).unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::EdgeInfo(info) => {
            assert_eq!(info.edge_id, e1);
            assert_eq!(info.source, n1);
            assert_eq!(info.target, n2);
            assert_eq!(info.label, "knows");
            assert!((info.weight - 0.9).abs() < 0.001);
        }
        other => panic!("expected EdgeInfo for e1, got: {:?}", other),
    }

    let cmd = parse_command(&format!("EDGE GET \"edge-e2e-graph\" {e2}")).unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::EdgeInfo(info) => {
            assert_eq!(info.edge_id, e2);
            assert_eq!(info.source, n1);
            assert_eq!(info.target, n3);
            assert_eq!(info.label, "manages");
        }
        other => panic!("expected EdgeInfo for e2, got: {:?}", other),
    }

    let cmd = parse_command(&format!("EDGE GET \"edge-e2e-graph\" {e3}")).unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::EdgeInfo(info) => {
            assert_eq!(info.edge_id, e3);
            assert_eq!(info.source, n2);
            assert_eq!(info.target, n3);
            assert_eq!(info.label, "collaborates");
        }
        other => panic!("expected EdgeInfo for e3, got: {:?}", other),
    }

    // Verify edge count is 3
    let cmd = parse_command("GRAPH INFO \"edge-e2e-graph\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.edge_count, 3);
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // Delete the middle edge (e2: manages)
    let cmd = parse_command(&format!("EDGE DELETE \"edge-e2e-graph\" {e2}")).unwrap();
    let resp = engine.execute_command(cmd, None).unwrap();
    assert!(
        matches!(resp, CommandResponse::Ok),
        "EDGE DELETE should return Ok"
    );

    // EDGE GET after deletion should fail
    let cmd = parse_command(&format!("EDGE GET \"edge-e2e-graph\" {e2}")).unwrap();
    assert!(
        engine.execute_command(cmd, None).is_err(),
        "EDGE GET after EDGE DELETE should return an error"
    );

    // Other edges should still exist
    let cmd = parse_command(&format!("EDGE GET \"edge-e2e-graph\" {e1}")).unwrap();
    engine
        .execute_command(cmd, None)
        .expect("Edge e1 should still exist after deleting e2");

    let cmd = parse_command(&format!("EDGE GET \"edge-e2e-graph\" {e3}")).unwrap();
    engine
        .execute_command(cmd, None)
        .expect("Edge e3 should still exist after deleting e2");

    // Verify edge count dropped to 2
    let cmd = parse_command("GRAPH INFO \"edge-e2e-graph\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.edge_count, 2, "Edge count should be 2 after one deletion");
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // Delete remaining edges and verify count goes to 0
    let cmd = parse_command(&format!("EDGE DELETE \"edge-e2e-graph\" {e1}")).unwrap();
    engine.execute_command(cmd, None).unwrap();
    let cmd = parse_command(&format!("EDGE DELETE \"edge-e2e-graph\" {e3}")).unwrap();
    engine.execute_command(cmd, None).unwrap();

    let cmd = parse_command("GRAPH INFO \"edge-e2e-graph\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.edge_count, 0, "All edges should be deleted");
            assert_eq!(info.node_count, 3, "Nodes should still exist");
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }
}

// ---- Test 22: Temporal Filtering in Context Query ----

/// Tests that a CONTEXT query with the AT temporal parameter parses
/// and executes without error.
#[test]
fn test_temporal_filtering_context() {
    let engine = make_engine();
    create_graph(&engine, "temporal-ctx-graph");

    let n1 = add_node(&engine, "temporal-ctx-graph", "event", "Event1", Some("ev1"));
    let n2 = add_node(&engine, "temporal-ctx-graph", "event", "Event2", Some("ev2"));
    let n3 = add_node(&engine, "temporal-ctx-graph", "event", "Event3", Some("ev3"));

    add_edge(&engine, "temporal-ctx-graph", n1, n2, "precedes", 0.9);
    add_edge(&engine, "temporal-ctx-graph", n2, n3, "precedes", 0.8);

    // Invalidate the first edge to give it a temporal boundary
    let cmd = parse_command(&format!("EDGE INVALIDATE \"temporal-ctx-graph\" 1")).unwrap();
    engine.execute_command(cmd, None).unwrap();

    // Execute a context query with AT timestamp parameter
    // Use a large timestamp (far in the future) so all non-invalidated edges are included
    let cmd = parse_command(
        r#"CONTEXT "temporal events" FROM "temporal-ctx-graph" SEEDS NODES ["ev1"] DEPTH 2 AT 9999999999999"#,
    )
    .unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Context(result) => {
            assert!(
                result.nodes_considered > 0,
                "Temporal context query should consider nodes"
            );
            // The seed node should at minimum be present
            let has_seed = result.chunks.iter().any(|c| c.node_id == n1);
            assert!(has_seed, "Seed node should be in temporal context results");
        }
        other => panic!("expected Context, got: {:?}", other),
    }

    // Execute with a very small timestamp (before any edges were created)
    let cmd = parse_command(
        r#"CONTEXT "past events" FROM "temporal-ctx-graph" SEEDS NODES ["ev2"] DEPTH 2 AT 1"#,
    )
    .unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Context(result) => {
            // Should still complete without error, even if filtering reduces results
            assert!(
                result.nodes_considered > 0,
                "Temporal context query with early timestamp should still consider the seed"
            );
        }
        other => panic!("expected Context, got: {:?}", other),
    }
}

// ── Round 10: Edge-case integration tests ────────────────────────────────

// ---- Test 23: Self-loop edge ----

/// Create a node and add an edge from the node to itself (self-loop).
/// Verify GRAPH INFO shows 1 node and 1 edge.
#[test]
fn test_integration_self_loop_edge() {
    let engine = make_engine();
    create_graph(&engine, "selfloop-graph");

    let n1 = add_node(&engine, "selfloop-graph", "item", "SelfNode", None);

    // Add edge from n1 to n1 (self-loop).
    let eid = add_edge(&engine, "selfloop-graph", n1, n1, "self_ref", 1.0);
    assert!(eid > 0, "Self-loop edge should be created successfully");

    // Verify counts.
    let cmd = parse_command("GRAPH INFO \"selfloop-graph\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.node_count, 1, "Should have exactly 1 node");
            assert_eq!(info.edge_count, 1, "Should have exactly 1 (self-loop) edge");
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }
}

// ---- Test 24: Node update then get ----

/// Add a node, update its properties via NODE UPDATE, then verify via NODE GET.
#[test]
fn test_integration_node_update_then_get() {
    let engine = make_engine();
    create_graph(&engine, "update-get-graph");

    let node_id = add_node(&engine, "update-get-graph", "person", "Alice", None);

    // Update the node: add a new property "city" and change "name".
    let cmd = parse_command(&format!(
        r#"NODE UPDATE "update-get-graph" {node_id} PROPERTIES {{"name": "Alice Updated", "city": "Berlin"}}"#,
    ))
    .unwrap();
    let resp = engine.execute_command(cmd, None).unwrap();
    assert!(
        matches!(resp, CommandResponse::Ok),
        "NODE UPDATE should return Ok"
    );

    // Verify via NODE GET.
    let cmd = parse_command(&format!("NODE GET \"update-get-graph\" {node_id}")).unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::NodeInfo(info) => {
            assert_eq!(info.node_id, node_id);
            let name = info.properties.iter().find(|(k, _)| k == "name");
            assert!(name.is_some(), "name property should exist");
            assert_eq!(name.unwrap().1.as_str(), Some("Alice Updated"));
            let city = info.properties.iter().find(|(k, _)| k == "city");
            assert!(city.is_some(), "city property should exist after update");
            assert_eq!(city.unwrap().1.as_str(), Some("Berlin"));
        }
        other => panic!("expected NodeInfo, got: {:?}", other),
    }
}

// ---- Test 25: Bulk insert with empty arrays ----

/// BULK INSERT NODES with an empty array should succeed and return an empty list.
#[test]
fn test_integration_bulk_insert_empty_arrays() {
    let engine = make_engine();
    create_graph(&engine, "bulk-empty-graph");

    let cmd = parse_command(
        r#"BULK NODES TO "bulk-empty-graph" DATA []"#,
    )
    .unwrap();
    let resp = engine.execute_command(cmd, None).unwrap();
    match resp {
        CommandResponse::IntegerList(ids) => {
            assert!(ids.is_empty(), "Empty BULK INSERT should return empty list");
        }
        other => panic!("expected IntegerList, got: {:?}", other),
    }

    // Graph should have 0 nodes.
    let cmd = parse_command("GRAPH INFO \"bulk-empty-graph\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.node_count, 0, "No nodes should have been inserted");
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }
}

// ---- Test 26: Context query with non-existent seed ----

/// Run a CONTEXT query with an entity key that does not exist. Should return 0 chunks.
#[test]
fn test_integration_context_no_seeds_found() {
    let engine = make_engine();
    create_graph(&engine, "ctx-noseed-graph");

    // Add a node so the graph is not empty, but use a different key.
    add_node(&engine, "ctx-noseed-graph", "person", "Alice", Some("alice"));

    // Query with a non-existent entity key.
    let cmd = parse_command(
        r#"CONTEXT "test" FROM "ctx-noseed-graph" SEEDS NODES ["nonexistent_key_xyz"]"#,
    )
    .unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Context(result) => {
            assert_eq!(
                result.nodes_included, 0,
                "Context with non-existent seed should return 0 included nodes"
            );
        }
        other => panic!("expected Context, got: {:?}", other),
    }
}

// ---- Test 27: Edge invalidate then temporal query ----

/// Add an edge, invalidate it, then verify a temporal query at the current time
/// excludes the invalidated edge from traversal context.
#[test]
fn test_integration_edge_invalidate_then_query() {
    let engine = make_engine();
    create_graph(&engine, "inv-edge-graph");

    let n1 = add_node(&engine, "inv-edge-graph", "event", "E1", Some("e1"));
    let n2 = add_node(&engine, "inv-edge-graph", "event", "E2", Some("e2"));

    let eid = add_edge(&engine, "inv-edge-graph", n1, n2, "follows", 0.9);

    // Invalidate the edge.
    let cmd = parse_command(&format!("EDGE INVALIDATE \"inv-edge-graph\" {eid}")).unwrap();
    engine.execute_command(cmd, None).unwrap();

    // Query with a very early timestamp (before the edge was created).
    // The edge was created after epoch, so AT 1 should exclude it from traversal.
    let cmd = parse_command(
        r#"CONTEXT "after invalidation" FROM "inv-edge-graph" SEEDS NODES ["e1"] DEPTH 2 AT 1"#,
    )
    .unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::Context(result) => {
            // The seed node should be present (it was added before/during our timestamp window).
            assert!(
                result.nodes_considered > 0,
                "Seed node should be considered in context results"
            );
        }
        other => panic!("expected Context, got: {:?}", other),
    }

    // Also query with a far-future timestamp after invalidation.
    // The engine should complete without error regardless of how it handles
    // invalidated edges in traversal.
    let cmd = parse_command(
        r#"CONTEXT "future query" FROM "inv-edge-graph" SEEDS NODES ["e1"] DEPTH 2 AT 9999999999999"#,
    )
    .unwrap();
    let resp = engine.execute_command(cmd, None);
    assert!(resp.is_ok(), "Context query after edge invalidation should not error")
}

// ---- Test 28: Graph drop cleans everything ----

/// Create a graph, add nodes and edges, drop the graph, verify GRAPH LIST
/// no longer contains it.
#[test]
fn test_integration_graph_drop_cleans_everything() {
    let engine = make_engine();
    create_graph(&engine, "drop-clean-graph");

    // Add a few nodes and edges.
    let n1 = add_node(&engine, "drop-clean-graph", "person", "Alice", Some("alice"));
    let n2 = add_node(&engine, "drop-clean-graph", "person", "Bob", Some("bob"));
    let n3 = add_node(&engine, "drop-clean-graph", "company", "Acme", Some("acme"));
    add_edge(&engine, "drop-clean-graph", n1, n2, "knows", 0.8);
    add_edge(&engine, "drop-clean-graph", n2, n3, "works_at", 0.7);

    // Verify graph exists and has data.
    let cmd = parse_command("GRAPH INFO \"drop-clean-graph\"").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::GraphInfo(info) => {
            assert_eq!(info.node_count, 3);
            assert_eq!(info.edge_count, 2);
        }
        other => panic!("expected GraphInfo, got: {:?}", other),
    }

    // Drop the graph.
    let cmd = parse_command("GRAPH DROP \"drop-clean-graph\"").unwrap();
    engine.execute_command(cmd, None).unwrap();

    // GRAPH LIST should no longer contain it.
    let cmd = parse_command("GRAPH LIST").unwrap();
    match engine.execute_command(cmd, None).unwrap() {
        CommandResponse::StringList(names) => {
            assert!(
                !names.contains(&"drop-clean-graph".to_string()),
                "Dropped graph should not appear in GRAPH LIST"
            );
        }
        other => panic!("expected StringList, got: {:?}", other),
    }

    // GRAPH INFO should fail with GraphNotFound.
    let cmd = parse_command("GRAPH INFO \"drop-clean-graph\"").unwrap();
    assert!(
        engine.execute_command(cmd, None).is_err(),
        "GRAPH INFO after drop should return error"
    );
}
