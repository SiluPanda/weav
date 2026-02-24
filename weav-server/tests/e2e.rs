//! End-to-end tests for the Weav HTTP server.
//!
//! These tests start a real HTTP server (axum on a random TCP port), make
//! real HTTP requests via reqwest, and verify responses. They test the full
//! HTTP stack as a production user would experience it.
//!
//! Run: `cargo test -p weav-server --test e2e`

use std::sync::Arc;
use std::time::Duration;

use reqwest::{Client, StatusCode};
use serde::Serialize;
use serde_json::{json, Value};

use weav_core::config::WeavConfig;
use weav_server::engine::Engine;
use weav_server::http::build_router;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test server: binds to a real TCP port, serves real HTTP
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct TestServer {
    base_url: String,
    client: Client,
    _shutdown: tokio::sync::oneshot::Sender<()>,
}

impl TestServer {
    async fn start() -> Self {
        let engine = Arc::new(Engine::new(WeavConfig::default()));
        Self::start_with_engine(engine).await
    }

    async fn start_with_engine(engine: Arc<Engine>) -> Self {
        let app = build_router(engine);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async { let _ = rx.await; })
                .await
                .unwrap();
        });

        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();

        TestServer {
            base_url: format!("http://127.0.0.1:{}", port),
            client,
            _shutdown: tx,
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HTTP client wrapper
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct WeavClient {
    client: Client,
    base_url: String,
}

#[derive(Serialize, Clone)]
struct AddNodeRequest {
    label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    properties: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    entity_key: Option<String>,
}

#[derive(Serialize, Clone)]
struct AddEdgeRequest {
    source: u64,
    target: u64,
    label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    weight: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    properties: Option<Value>,
}

impl WeavClient {
    fn new(server: &TestServer) -> Self {
        Self {
            client: server.client.clone(),
            base_url: server.base_url.clone(),
        }
    }

    // ── Health / Info / Metrics ──

    async fn health(&self) -> (StatusCode, Value) {
        let resp = self.client.get(format!("{}/health", self.base_url)).send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    async fn info(&self) -> (StatusCode, Value) {
        let resp = self.client.get(format!("{}/v1/info", self.base_url)).send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    async fn metrics(&self) -> (StatusCode, String) {
        let resp = self.client.get(format!("{}/metrics", self.base_url)).send().await.unwrap();
        let status = resp.status();
        (status, resp.text().await.unwrap())
    }

    // ── Graph operations ──

    async fn create_graph(&self, name: &str) -> (StatusCode, Value) {
        let resp = self.client.post(format!("{}/v1/graphs", self.base_url))
            .json(&json!({"name": name}))
            .send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    async fn list_graphs(&self) -> (StatusCode, Value) {
        let resp = self.client.get(format!("{}/v1/graphs", self.base_url)).send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    async fn get_graph(&self, name: &str) -> (StatusCode, Value) {
        let resp = self.client.get(format!("{}/v1/graphs/{}", self.base_url, name)).send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    async fn delete_graph(&self, name: &str) -> (StatusCode, Value) {
        let resp = self.client.delete(format!("{}/v1/graphs/{}", self.base_url, name)).send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    // ── Node operations ──

    async fn add_node(&self, graph: &str, req: &AddNodeRequest) -> (StatusCode, Value) {
        let resp = self.client.post(format!("{}/v1/graphs/{}/nodes", self.base_url, graph))
            .json(req).send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    async fn bulk_add_nodes(&self, graph: &str, nodes: &[AddNodeRequest]) -> (StatusCode, Value) {
        let resp = self.client.post(format!("{}/v1/graphs/{}/nodes/bulk", self.base_url, graph))
            .json(&json!({"nodes": nodes}))
            .send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    async fn get_node(&self, graph: &str, id: u64) -> (StatusCode, Value) {
        let resp = self.client.get(format!("{}/v1/graphs/{}/nodes/{}", self.base_url, graph, id))
            .send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    async fn update_node(&self, graph: &str, id: u64, body: &Value) -> (StatusCode, Value) {
        let resp = self.client.put(format!("{}/v1/graphs/{}/nodes/{}", self.base_url, graph, id))
            .json(body).send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    async fn delete_node(&self, graph: &str, id: u64) -> (StatusCode, Value) {
        let resp = self.client.delete(format!("{}/v1/graphs/{}/nodes/{}", self.base_url, graph, id))
            .send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    // ── Edge operations ──

    async fn add_edge(&self, graph: &str, req: &AddEdgeRequest) -> (StatusCode, Value) {
        let resp = self.client.post(format!("{}/v1/graphs/{}/edges", self.base_url, graph))
            .json(req).send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    async fn bulk_add_edges(&self, graph: &str, edges: &[AddEdgeRequest]) -> (StatusCode, Value) {
        let resp = self.client.post(format!("{}/v1/graphs/{}/edges/bulk", self.base_url, graph))
            .json(&json!({"edges": edges}))
            .send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    async fn get_edge(&self, graph: &str, id: u64) -> (StatusCode, Value) {
        let resp = self.client.get(format!("{}/v1/graphs/{}/edges/{}", self.base_url, graph, id))
            .send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    async fn delete_edge(&self, graph: &str, id: u64) -> (StatusCode, Value) {
        let resp = self.client.delete(format!("{}/v1/graphs/{}/edges/{}", self.base_url, graph, id))
            .send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    async fn invalidate_edge(&self, graph: &str, id: u64) -> (StatusCode, Value) {
        let resp = self.client.post(format!("{}/v1/graphs/{}/edges/{}/invalidate", self.base_url, graph, id))
            .send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    // ── Context query ──

    async fn context(&self, body: &Value) -> (StatusCode, Value) {
        let resp = self.client.post(format!("{}/v1/context", self.base_url))
            .json(body).send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    // ── Snapshot ──

    async fn snapshot(&self) -> (StatusCode, Value) {
        let resp = self.client.post(format!("{}/v1/snapshot", self.base_url))
            .send().await.unwrap();
        let status = resp.status();
        (status, resp.json().await.unwrap())
    }

    // ── Raw request helpers ──

    async fn raw_post(&self, path: &str, body: &str) -> (StatusCode, String) {
        let resp = self.client.post(format!("{}{}", self.base_url, path))
            .header("content-type", "application/json")
            .body(body.to_string())
            .send().await.unwrap();
        let status = resp.status();
        (status, resp.text().await.unwrap())
    }

    async fn raw_put(&self, path: &str) -> (StatusCode, String) {
        let resp = self.client.put(format!("{}{}", self.base_url, path))
            .send().await.unwrap();
        let status = resp.status();
        (status, resp.text().await.unwrap())
    }

    async fn raw_get(&self, path: &str) -> (StatusCode, String) {
        let resp = self.client.get(format!("{}{}", self.base_url, path))
            .send().await.unwrap();
        let status = resp.status();
        (status, resp.text().await.unwrap())
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Test helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn make_node(label: &str) -> AddNodeRequest {
    AddNodeRequest { label: label.to_string(), properties: None, embedding: None, entity_key: None }
}

fn make_node_with_props(label: &str, props: Value) -> AddNodeRequest {
    AddNodeRequest { label: label.to_string(), properties: Some(props), embedding: None, entity_key: None }
}

fn make_node_with_key(label: &str, key: &str, props: Value) -> AddNodeRequest {
    AddNodeRequest { label: label.to_string(), properties: Some(props), embedding: None, entity_key: Some(key.to_string()) }
}

fn extract_node_id(body: &Value) -> u64 { body["data"]["node_id"].as_u64().unwrap() }
fn extract_edge_id(body: &Value) -> u64 { body["data"]["edge_id"].as_u64().unwrap() }

/// Start a server with a fresh engine, create a graph, return client + graph name.
async fn setup(graph_name: &str) -> (TestServer, WeavClient, String) {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let (status, _) = client.create_graph(graph_name).await;
    assert_eq!(status, StatusCode::CREATED, "failed to create graph {graph_name}");
    (server, client, graph_name.to_string())
}

async fn create_two_nodes(client: &WeavClient, graph: &str) -> (u64, u64) {
    let (_, b1) = client.add_node(graph, &make_node_with_props("a", json!({"name": "node-a"}))).await;
    let (_, b2) = client.add_node(graph, &make_node_with_props("b", json!({"name": "node-b"}))).await;
    (extract_node_id(&b1), extract_node_id(&b2))
}

/// Build a small knowledge graph: Alice --knows--> Bob --works_at--> Acme
async fn build_knowledge_graph(client: &WeavClient, graph: &str) -> (u64, u64, u64) {
    let alice = make_node_with_key("person", "alice", json!({"name": "Alice", "bio": "Alice is a software engineer who loves Rust."}));
    let bob = make_node_with_key("person", "bob", json!({"name": "Bob", "bio": "Bob is a data scientist specializing in ML."}));
    let acme = make_node_with_key("company", "acme", json!({"name": "Acme Corp", "description": "Acme is a tech company."}));

    let (_, a) = client.add_node(graph, &alice).await;
    let (_, b) = client.add_node(graph, &bob).await;
    let (_, c) = client.add_node(graph, &acme).await;
    let (aid, bid, cid) = (extract_node_id(&a), extract_node_id(&b), extract_node_id(&c));

    client.add_edge(graph, &AddEdgeRequest { source: aid, target: bid, label: "knows".to_string(), weight: Some(0.9), properties: None }).await;
    client.add_edge(graph, &AddEdgeRequest { source: bid, target: cid, label: "works_at".to_string(), weight: Some(0.8), properties: None }).await;
    (aid, bid, cid)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Category 1: Health & Server Info (3 tests)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_health_endpoint() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let (status, body) = client.health().await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["success"], true);
    assert_eq!(body["data"]["status"], "ok");
}

#[tokio::test]
async fn test_info_endpoint() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let (status, body) = client.info().await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["success"], true);
    let info_text = body["data"].as_str().unwrap();
    assert!(info_text.contains("weav-server"), "info should contain 'weav-server', got: {info_text}");
}

#[tokio::test]
async fn test_metrics_endpoint() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let (status, text) = client.metrics().await;
    assert_eq!(status, StatusCode::OK);
    assert!(text.is_ascii(), "metrics should be ASCII text");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Category 2: Graph CRUD (8 tests)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_create_graph() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let (status, body) = client.create_graph("test").await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(body["success"], true);
}

#[tokio::test]
async fn test_create_duplicate_graph() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    client.create_graph("dup").await;
    let (status, body) = client.create_graph("dup").await;
    assert_eq!(status, StatusCode::CONFLICT);
    assert_eq!(body["success"], false);
}

#[tokio::test]
async fn test_list_graphs() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    for name in &["g1", "g2", "g3"] {
        client.create_graph(name).await;
    }
    let (status, body) = client.list_graphs().await;
    assert_eq!(status, StatusCode::OK);
    let names: Vec<&str> = body["data"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    assert!(names.contains(&"g1"));
    assert!(names.contains(&"g2"));
    assert!(names.contains(&"g3"));
}

#[tokio::test]
async fn test_list_graphs_empty() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let (_, body) = client.list_graphs().await;
    assert_eq!(body["data"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn test_get_graph_info() {
    let (_server, client, graph) = setup("info_test").await;
    let (status, body) = client.get_graph(&graph).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"]["name"], "info_test");
    assert_eq!(body["data"]["node_count"], 0);
    assert_eq!(body["data"]["edge_count"], 0);
}

#[tokio::test]
async fn test_get_graph_info_not_found() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let (status, _) = client.get_graph("nonexistent").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_delete_graph() {
    let (_server, client, graph) = setup("del_test").await;
    let (status, _) = client.delete_graph(&graph).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = client.get_graph(&graph).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_delete_graph_not_found() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let (status, _) = client.delete_graph("nonexistent").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Category 3: Node CRUD (12 tests)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_add_node_minimal() {
    let (_s, client, graph) = setup("g").await;
    let (status, body) = client.add_node(&graph, &make_node("person")).await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(body["success"], true);
    assert!(extract_node_id(&body) >= 1);
}

#[tokio::test]
async fn test_add_node_with_properties() {
    let (_s, client, graph) = setup("g").await;
    let node = make_node_with_props("person", json!({"name": "Alice", "age": 30}));
    let (_, body) = client.add_node(&graph, &node).await;
    let id = extract_node_id(&body);
    let (status, body) = client.get_node(&graph, id).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"]["label"], "person");
    assert_eq!(body["data"]["properties"]["name"], "Alice");
    assert_eq!(body["data"]["properties"]["age"], 30);
}

#[tokio::test]
async fn test_add_node_with_entity_key() {
    let (_s, client, graph) = setup("g").await;
    let node = make_node_with_key("person", "alice-key", json!({"name": "Alice"}));
    let (status, body) = client.add_node(&graph, &node).await;
    assert_eq!(status, StatusCode::CREATED);
    let id = extract_node_id(&body);
    let (status, body) = client.get_node(&graph, id).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"]["label"], "person");
}

#[tokio::test]
async fn test_add_node_with_embedding() {
    let (_s, client, graph) = setup("g").await;
    let embedding: Vec<f32> = (0..1536).map(|i| (i as f32) * 0.001).collect();
    let node = AddNodeRequest {
        label: "document".to_string(),
        properties: Some(json!({"title": "test doc"})),
        embedding: Some(embedding),
        entity_key: None,
    };
    let (status, body) = client.add_node(&graph, &node).await;
    assert_eq!(status, StatusCode::CREATED);
    assert!(extract_node_id(&body) >= 1);
}

#[tokio::test]
async fn test_get_node() {
    let (_s, client, graph) = setup("g").await;
    let node = make_node_with_props("person", json!({"name": "Bob", "role": "engineer"}));
    let (_, body) = client.add_node(&graph, &node).await;
    let id = extract_node_id(&body);
    let (status, body) = client.get_node(&graph, id).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"]["node_id"], id);
    assert_eq!(body["data"]["label"], "person");
    assert_eq!(body["data"]["properties"]["name"], "Bob");
    assert_eq!(body["data"]["properties"]["role"], "engineer");
}

#[tokio::test]
async fn test_get_node_not_found() {
    let (_s, client, graph) = setup("g").await;
    let (status, body) = client.get_node(&graph, 99999).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["success"], false);
}

#[tokio::test]
async fn test_get_node_graph_not_found() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let (status, _) = client.get_node("nonexistent", 1).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_update_node_properties() {
    let (_s, client, graph) = setup("g").await;
    let node = make_node_with_props("person", json!({"name": "Alice", "age": 25}));
    let (_, body) = client.add_node(&graph, &node).await;
    let id = extract_node_id(&body);
    let (status, _) = client.update_node(&graph, id, &json!({"properties": {"name": "Bob", "city": "NYC"}})).await;
    assert_eq!(status, StatusCode::OK);
    let (_, body) = client.get_node(&graph, id).await;
    assert_eq!(body["data"]["properties"]["name"], "Bob");
    assert_eq!(body["data"]["properties"]["city"], "NYC");
}

#[tokio::test]
async fn test_update_node_preserves_unmodified() {
    let (_s, client, graph) = setup("g").await;
    let node = make_node_with_props("person", json!({"name": "Alice", "age": 25, "role": "dev"}));
    let (_, body) = client.add_node(&graph, &node).await;
    let id = extract_node_id(&body);
    client.update_node(&graph, id, &json!({"properties": {"name": "Bob"}})).await;
    let (_, body) = client.get_node(&graph, id).await;
    assert_eq!(body["data"]["properties"]["name"], "Bob");
    assert_eq!(body["data"]["properties"]["age"], 25);
    assert_eq!(body["data"]["properties"]["role"], "dev");
}

#[tokio::test]
async fn test_update_node_not_found() {
    let (_s, client, graph) = setup("g").await;
    let (status, _) = client.update_node(&graph, 99999, &json!({"properties": {"name": "x"}})).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_delete_node() {
    let (_s, client, graph) = setup("g").await;
    let (_, body) = client.add_node(&graph, &make_node("person")).await;
    let id = extract_node_id(&body);
    let (status, _) = client.delete_node(&graph, id).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = client.get_node(&graph, id).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_delete_node_not_found() {
    let (_s, client, graph) = setup("g").await;
    let (status, _) = client.delete_node(&graph, 99999).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Category 4: Bulk Operations (4 tests)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_bulk_add_nodes() {
    let (_s, client, graph) = setup("g").await;
    let nodes: Vec<AddNodeRequest> = (0..10).map(|i| make_node_with_props("person", json!({"name": format!("p-{}", i)}))).collect();
    let (status, body) = client.bulk_add_nodes(&graph, &nodes).await;
    assert_eq!(status, StatusCode::CREATED);
    let ids = body["data"]["node_ids"].as_array().unwrap();
    assert_eq!(ids.len(), 10);
    let id_set: std::collections::HashSet<u64> = ids.iter().map(|v| v.as_u64().unwrap()).collect();
    assert_eq!(id_set.len(), 10);
}

#[tokio::test]
async fn test_bulk_add_nodes_retrievable() {
    let (_s, client, graph) = setup("g").await;
    let nodes: Vec<AddNodeRequest> = (0..5).map(|i| make_node_with_key("person", &format!("key-{i}"), json!({"name": format!("p-{i}")}))).collect();
    let (_, body) = client.bulk_add_nodes(&graph, &nodes).await;
    let ids: Vec<u64> = body["data"]["node_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    for (i, id) in ids.iter().enumerate() {
        let (status, body) = client.get_node(&graph, *id).await;
        assert_eq!(status, StatusCode::OK, "node {} should be retrievable", id);
        assert_eq!(body["data"]["properties"]["name"], format!("p-{i}"));
    }
}

#[tokio::test]
async fn test_bulk_add_edges() {
    let (_s, client, graph) = setup("g").await;
    let nodes: Vec<AddNodeRequest> = (0..6).map(|_| make_node("node")).collect();
    let (_, body) = client.bulk_add_nodes(&graph, &nodes).await;
    let nids: Vec<u64> = body["data"]["node_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    let edges: Vec<AddEdgeRequest> = (0..5).map(|i| AddEdgeRequest {
        source: nids[i], target: nids[i + 1], label: "links_to".to_string(), weight: None, properties: None,
    }).collect();
    let (status, body) = client.bulk_add_edges(&graph, &edges).await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(body["data"]["edge_ids"].as_array().unwrap().len(), 5);
}

#[tokio::test]
async fn test_bulk_large_batch() {
    let (_s, client, graph) = setup("g").await;
    let nodes: Vec<AddNodeRequest> = (0..100).map(|i| make_node_with_props("item", json!({"index": i}))).collect();
    let (_, body) = client.bulk_add_nodes(&graph, &nodes).await;
    let nids: Vec<u64> = body["data"]["node_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    assert_eq!(nids.len(), 100);
    let edges: Vec<AddEdgeRequest> = (0..99).map(|i| AddEdgeRequest {
        source: nids[i], target: nids[i + 1], label: "next".to_string(), weight: None, properties: None,
    }).collect();
    let (status, body) = client.bulk_add_edges(&graph, &edges).await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(body["data"]["edge_ids"].as_array().unwrap().len(), 99);
    let (_, body) = client.get_graph(&graph).await;
    assert_eq!(body["data"]["node_count"], 100);
    assert_eq!(body["data"]["edge_count"], 99);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Category 5: Edge CRUD (9 tests)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_add_edge() {
    let (_s, client, graph) = setup("g").await;
    let (n1, n2) = create_two_nodes(&client, &graph).await;
    let edge = AddEdgeRequest { source: n1, target: n2, label: "knows".to_string(), weight: None, properties: None };
    let (status, body) = client.add_edge(&graph, &edge).await;
    assert_eq!(status, StatusCode::CREATED);
    assert!(extract_edge_id(&body) >= 1);
}

#[tokio::test]
async fn test_add_edge_with_weight() {
    let (_s, client, graph) = setup("g").await;
    let (n1, n2) = create_two_nodes(&client, &graph).await;
    let edge = AddEdgeRequest { source: n1, target: n2, label: "rated".to_string(), weight: Some(0.75), properties: None };
    let (_, body) = client.add_edge(&graph, &edge).await;
    let eid = extract_edge_id(&body);
    let (status, body) = client.get_edge(&graph, eid).await;
    assert_eq!(status, StatusCode::OK);
    let weight = body["data"]["weight"].as_f64().unwrap();
    assert!((weight - 0.75).abs() < 0.01, "weight should be 0.75, got {weight}");
}

#[tokio::test]
async fn test_add_edge_with_properties() {
    let (_s, client, graph) = setup("g").await;
    let (n1, n2) = create_two_nodes(&client, &graph).await;
    let edge = AddEdgeRequest { source: n1, target: n2, label: "interacted".to_string(), weight: None, properties: Some(json!({"type": "meeting"})) };
    let (status, _) = client.add_edge(&graph, &edge).await;
    assert_eq!(status, StatusCode::CREATED);
}

#[tokio::test]
async fn test_add_edge_invalid_source() {
    let (_s, client, graph) = setup("g").await;
    let (_, n2) = create_two_nodes(&client, &graph).await;
    let edge = AddEdgeRequest { source: 99999, target: n2, label: "broken".to_string(), weight: None, properties: None };
    let (status, body) = client.add_edge(&graph, &edge).await;
    assert!(status.is_client_error() || status.is_server_error(), "invalid source should fail, got {status}");
    assert_eq!(body["success"], false);
}

#[tokio::test]
async fn test_get_edge() {
    let (_s, client, graph) = setup("g").await;
    let (n1, n2) = create_two_nodes(&client, &graph).await;
    let edge = AddEdgeRequest { source: n1, target: n2, label: "knows".to_string(), weight: Some(0.9), properties: None };
    let (_, body) = client.add_edge(&graph, &edge).await;
    let eid = extract_edge_id(&body);
    let (status, body) = client.get_edge(&graph, eid).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"]["edge_id"], eid);
    assert_eq!(body["data"]["source"], n1);
    assert_eq!(body["data"]["target"], n2);
    assert_eq!(body["data"]["label"], "knows");
}

#[tokio::test]
async fn test_get_edge_not_found() {
    let (_s, client, graph) = setup("g").await;
    let (status, _) = client.get_edge(&graph, 99999).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_delete_edge() {
    let (_s, client, graph) = setup("g").await;
    let (n1, n2) = create_two_nodes(&client, &graph).await;
    let edge = AddEdgeRequest { source: n1, target: n2, label: "link".to_string(), weight: None, properties: None };
    let (_, body) = client.add_edge(&graph, &edge).await;
    let eid = extract_edge_id(&body);
    let (_, info) = client.get_graph(&graph).await;
    assert_eq!(info["data"]["edge_count"], 1);
    let (status, _) = client.delete_edge(&graph, eid).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = client.get_edge(&graph, eid).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (_, info) = client.get_graph(&graph).await;
    assert_eq!(info["data"]["edge_count"], 0);
}

#[tokio::test]
async fn test_delete_edge_not_found() {
    let (_s, client, graph) = setup("g").await;
    let (status, _) = client.delete_edge(&graph, 99999).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_invalidate_edge() {
    let (_s, client, graph) = setup("g").await;
    let (n1, n2) = create_two_nodes(&client, &graph).await;
    let edge = AddEdgeRequest { source: n1, target: n2, label: "link".to_string(), weight: None, properties: None };
    let (_, body) = client.add_edge(&graph, &edge).await;
    let eid = extract_edge_id(&body);
    let (status, body) = client.invalidate_edge(&graph, eid).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["success"], true);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Category 6: Context Query (7 tests)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_context_with_seed_nodes() {
    let (_s, client, graph) = setup("g").await;
    build_knowledge_graph(&client, &graph).await;
    let (status, body) = client.context(&json!({
        "graph": graph, "query": "Who does Alice know?", "seed_nodes": ["alice"], "budget": 4096, "max_depth": 2
    })).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["success"], true);
    assert!(body["data"]["nodes_included"].as_u64().unwrap() > 0);
    assert!(body["data"]["total_tokens"].as_u64().unwrap() > 0);
    assert!(body["data"]["budget_used"].as_f64().unwrap() <= 1.0);
    assert!(!body["data"]["chunks"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_context_with_embedding() {
    let (_s, client, graph) = setup("g").await;
    let mut emb_a: Vec<f32> = vec![0.0; 1536]; emb_a[0] = 1.0;
    let mut emb_b: Vec<f32> = vec![0.0; 1536]; emb_b[0] = 0.9; emb_b[1] = 0.1;
    let node_a = AddNodeRequest { label: "doc".to_string(), properties: Some(json!({"content": "Rust programming"})), embedding: Some(emb_a.clone()), entity_key: Some("doc-rust".to_string()) };
    let node_b = AddNodeRequest { label: "doc".to_string(), properties: Some(json!({"content": "Python programming"})), embedding: Some(emb_b), entity_key: Some("doc-python".to_string()) };
    client.add_node(&graph, &node_a).await;
    client.add_node(&graph, &node_b).await;
    let (status, body) = client.context(&json!({"graph": graph, "embedding": emb_a, "budget": 4096})).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["nodes_included"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_context_with_both_seeds() {
    let (_s, client, graph) = setup("g").await;
    let mut emb: Vec<f32> = vec![0.0; 1536]; emb[0] = 1.0;
    let node_a = AddNodeRequest { label: "doc".to_string(), properties: Some(json!({"content": "Vector search"})), embedding: Some(emb.clone()), entity_key: Some("doc-vec".to_string()) };
    let node_b = make_node_with_key("person", "alice", json!({"name": "Alice"}));
    client.add_node(&graph, &node_a).await;
    client.add_node(&graph, &node_b).await;
    let (status, body) = client.context(&json!({"graph": graph, "embedding": emb, "seed_nodes": ["alice"], "budget": 4096})).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["nodes_included"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_context_with_budget() {
    let (_s, client, graph) = setup("g").await;
    let nodes: Vec<AddNodeRequest> = (0..20).map(|i| make_node_with_key("article", &format!("art-{i}"),
        json!({"title": format!("Article {i}"), "body": format!("A moderately long article body number {i} with enough text to consume tokens.")})
    )).collect();
    client.bulk_add_nodes(&graph, &nodes).await;
    let (status, body) = client.context(&json!({"graph": graph, "seed_nodes": ["art-0"], "budget": 50, "max_depth": 1})).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["total_tokens"].as_u64().unwrap() <= 50);
}

#[tokio::test]
async fn test_context_with_provenance() {
    let (_s, client, graph) = setup("g").await;
    build_knowledge_graph(&client, &graph).await;
    let (status, body) = client.context(&json!({"graph": graph, "seed_nodes": ["alice"], "budget": 4096, "include_provenance": true})).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["chunks"].is_array());
}

#[tokio::test]
async fn test_context_empty_graph() {
    let (_s, client, graph) = setup("g").await;
    let (status, body) = client.context(&json!({"graph": graph, "seed_nodes": ["nonexistent"], "budget": 4096})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"]["nodes_included"], 0);
}

#[tokio::test]
async fn test_context_graph_not_found() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let (status, body) = client.context(&json!({"graph": "nonexistent", "seed_nodes": ["foo"], "budget": 4096})).await;
    assert!(status.is_client_error() || status.is_server_error(), "context on missing graph should fail, got {status}");
    assert_eq!(body["success"], false);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Category 7: Error Handling & Edge Cases (10 tests)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_invalid_json_body() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let (status, _) = client.raw_post("/v1/graphs", "{invalid json").await;
    assert!(status.is_client_error(), "invalid JSON should return 4xx, got {status}");
}

#[tokio::test]
async fn test_missing_required_field() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let (status, _) = client.raw_post("/v1/graphs", "{}").await;
    assert!(status.is_client_error(), "missing 'name' field should return 4xx, got {status}");
}

#[tokio::test]
async fn test_wrong_http_method() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let (status, _) = client.raw_put("/v1/graphs").await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn test_nonexistent_route() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let (status, _) = client.raw_get("/v1/nonexistent").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_empty_graph_name() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let (status, _) = client.create_graph("").await;
    assert!(status == StatusCode::CREATED || status.is_client_error(),
        "empty graph name should either create or reject cleanly, got {status}");
}

#[tokio::test]
async fn test_special_chars_in_graph_name() {
    let (_s, client, graph) = setup("special-chars_test").await;
    let (status, body) = client.add_node(&graph, &make_node_with_props("item", json!({"val": 1}))).await;
    assert_eq!(status, StatusCode::CREATED);
    let id = extract_node_id(&body);
    let (status, _) = client.get_node(&graph, id).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = client.delete_graph(&graph).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = client.get_graph(&graph).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_node_properties_null_values() {
    let (_s, client, graph) = setup("g").await;
    let node = make_node_with_props("thing", json!({"name": "test", "optional": null, "active": true}));
    let (_, body) = client.add_node(&graph, &node).await;
    let id = extract_node_id(&body);
    let (status, body) = client.get_node(&graph, id).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["data"]["properties"]["optional"].is_null());
    assert_eq!(body["data"]["properties"]["active"], true);
}

#[tokio::test]
async fn test_node_properties_nested_objects() {
    let (_s, client, graph) = setup("g").await;
    let node = make_node_with_props("person", json!({"name": "Alice", "address": {"city": "NYC", "zip": "10001"}}));
    let (_, body) = client.add_node(&graph, &node).await;
    let id = extract_node_id(&body);
    let (_, body) = client.get_node(&graph, id).await;
    assert_eq!(body["data"]["properties"]["address"]["city"], "NYC");
    assert_eq!(body["data"]["properties"]["address"]["zip"], "10001");
}

#[tokio::test]
async fn test_node_properties_arrays() {
    let (_s, client, graph) = setup("g").await;
    let node = make_node_with_props("item", json!({"tags": ["rust", "database", "graph"], "name": "weav"}));
    let (_, body) = client.add_node(&graph, &node).await;
    let id = extract_node_id(&body);
    let (_, body) = client.get_node(&graph, id).await;
    let tags = body["data"]["properties"]["tags"].as_array().unwrap();
    assert_eq!(tags.len(), 3);
}

#[tokio::test]
async fn test_snapshot_endpoint() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let (status, body) = client.snapshot().await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["success"], true);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Category 8: Data Integrity & Concurrency (6 tests)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_graph_full_lifecycle() {
    let (_s, client, graph) = setup("g").await;
    let (_, b1) = client.add_node(&graph, &make_node_with_props("person", json!({"name": "A"}))).await;
    let (_, b2) = client.add_node(&graph, &make_node_with_props("person", json!({"name": "B"}))).await;
    let (n1, n2) = (extract_node_id(&b1), extract_node_id(&b2));
    client.add_edge(&graph, &AddEdgeRequest { source: n1, target: n2, label: "knows".to_string(), weight: None, properties: None }).await;
    let (_, info) = client.get_graph(&graph).await;
    assert_eq!(info["data"]["node_count"], 2);
    assert_eq!(info["data"]["edge_count"], 1);
    client.delete_node(&graph, n1).await;
    let (_, info) = client.get_graph(&graph).await;
    assert_eq!(info["data"]["node_count"], 1);
    assert_eq!(info["data"]["edge_count"], 0);
    client.delete_graph(&graph).await;
    let (status, _) = client.get_graph(&graph).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_node_deletion_cascades_edges() {
    let (_s, client, graph) = setup("g").await;
    let (n1, n2) = create_two_nodes(&client, &graph).await;
    client.add_edge(&graph, &AddEdgeRequest { source: n1, target: n2, label: "link".to_string(), weight: None, properties: None }).await;
    let (_, info) = client.get_graph(&graph).await;
    assert_eq!(info["data"]["edge_count"], 1);
    client.delete_node(&graph, n1).await;
    let (_, info) = client.get_graph(&graph).await;
    assert_eq!(info["data"]["edge_count"], 0);
}

#[tokio::test]
async fn test_concurrent_node_creation() {
    let engine = Arc::new(Engine::new(WeavConfig::default()));
    engine.execute_command(weav_query::parser::Command::GraphCreate(weav_query::parser::GraphCreateCmd { name: "g".to_string(), config: None })).unwrap();
    let server = TestServer::start_with_engine(engine).await;

    let mut handles = Vec::new();
    for task_id in 0..4u32 {
        let url = format!("{}/v1/graphs/g/nodes", server.base_url);
        let c = server.client.clone();
        handles.push(tokio::spawn(async move {
            for i in 0..25u32 {
                let resp = c.post(&url)
                    .json(&json!({"label": "item", "properties": {"task": task_id, "index": i}}))
                    .send().await.unwrap();
                assert_eq!(resp.status(), StatusCode::CREATED);
            }
        }));
    }
    for h in handles { h.await.unwrap(); }

    let client = WeavClient::new(&server);
    let (_, info) = client.get_graph("g").await;
    assert_eq!(info["data"]["node_count"], 100);
}

#[tokio::test]
async fn test_concurrent_graph_operations() {
    let server = TestServer::start().await;

    let mut handles = Vec::new();
    for task_id in 0..4u32 {
        let url_base = server.base_url.clone();
        let c = server.client.clone();
        let g = format!("g{task_id}");
        handles.push(tokio::spawn(async move {
            c.post(format!("{}/v1/graphs", url_base)).json(&json!({"name": g})).send().await.unwrap();
            for i in 0..20u32 {
                let resp = c.post(format!("{}/v1/graphs/{}/nodes", url_base, g))
                    .json(&json!({"label": "item", "properties": {"i": i}}))
                    .send().await.unwrap();
                assert_eq!(resp.status(), StatusCode::CREATED);
            }
            let resp = c.get(format!("{}/v1/graphs/{}", url_base, g)).send().await.unwrap();
            let body: Value = resp.json().await.unwrap();
            assert_eq!(body["data"]["node_count"], 20);
        }));
    }
    for h in handles { h.await.unwrap(); }
}

#[tokio::test]
async fn test_concurrent_read_write() {
    let engine = Arc::new(Engine::new(WeavConfig::default()));
    engine.execute_command(weav_query::parser::Command::GraphCreate(weav_query::parser::GraphCreateCmd { name: "g".to_string(), config: None })).unwrap();

    // Pre-seed 20 nodes via engine
    for i in 0..20u32 {
        engine.execute_command(weav_query::parser::Command::NodeAdd(weav_query::parser::NodeAddCmd {
            graph: "g".to_string(), label: "item".to_string(),
            properties: vec![("index".to_string(), weav_core::types::Value::Int(i as i64))],
            embedding: None, entity_key: None,
        })).unwrap();
    }

    let server = TestServer::start_with_engine(engine).await;
    let mut handles = Vec::new();

    // 2 writer tasks (25 nodes each)
    for task_id in 0..2u32 {
        let url = format!("{}/v1/graphs/g/nodes", server.base_url);
        let c = server.client.clone();
        handles.push(tokio::spawn(async move {
            for i in 0..25u32 {
                let resp = c.post(&url)
                    .json(&json!({"label": "item", "properties": {"writer": task_id, "i": i}}))
                    .send().await.unwrap();
                assert_eq!(resp.status(), StatusCode::CREATED);
            }
        }));
    }

    // 2 reader tasks
    for _ in 0..2u32 {
        let graph_url = format!("{}/v1/graphs/g", server.base_url);
        let c = server.client.clone();
        handles.push(tokio::spawn(async move {
            for _ in 0..25u32 {
                let resp = c.get(&graph_url).send().await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
                let resp = c.get(format!("{}/nodes/1", graph_url)).send().await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
            }
        }));
    }

    for h in handles { h.await.unwrap(); }

    let client = WeavClient::new(&server);
    let (_, info) = client.get_graph("g").await;
    assert_eq!(info["data"]["node_count"], 70); // 20 seed + 50 writer
}

/// High-concurrency stress test: 8 writer tasks x 50 ops + 4 reader tasks x 50 ops.
/// This exact scenario caused cascading server crashes before the parking_lot fix
/// due to std::sync::RwLock poisoning.
#[tokio::test]
async fn test_high_concurrency_stress() {
    let engine = Arc::new(Engine::new(WeavConfig::default()));
    engine.execute_command(weav_query::parser::Command::GraphCreate(weav_query::parser::GraphCreateCmd { name: "stress".to_string(), config: None })).unwrap();

    // Pre-seed 10 nodes
    for i in 0..10u32 {
        engine.execute_command(weav_query::parser::Command::NodeAdd(weav_query::parser::NodeAddCmd {
            graph: "stress".to_string(), label: "item".to_string(),
            properties: vec![("seed".to_string(), weav_core::types::Value::Int(i as i64))],
            embedding: None, entity_key: None,
        })).unwrap();
    }

    let server = TestServer::start_with_engine(engine).await;
    let mut handles = Vec::new();

    // 8 writer tasks (50 nodes each = 400 total)
    for task_id in 0..8u32 {
        let url = format!("{}/v1/graphs/stress/nodes", server.base_url);
        let c = server.client.clone();
        handles.push(tokio::spawn(async move {
            for i in 0..50u32 {
                let resp = c.post(&url)
                    .json(&json!({"label": "item", "properties": {"task": task_id, "i": i}}))
                    .send().await.unwrap();
                assert_eq!(resp.status(), StatusCode::CREATED);
            }
        }));
    }

    // 4 reader tasks (50 reads each = 200 total)
    for _ in 0..4u32 {
        let graph_url = format!("{}/v1/graphs/stress", server.base_url);
        let c = server.client.clone();
        handles.push(tokio::spawn(async move {
            for _ in 0..50u32 {
                let resp = c.get(&graph_url).send().await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
                let resp = c.get(format!("{}/nodes/1", graph_url)).send().await.unwrap();
                assert_eq!(resp.status(), StatusCode::OK);
            }
        }));
    }

    for h in handles { h.await.unwrap(); }

    // Verify: 10 seed + 400 writer = 410
    let client = WeavClient::new(&server);
    let (_, info) = client.get_graph("stress").await;
    assert_eq!(info["data"]["node_count"], 410);
}

#[tokio::test]
async fn test_multiple_graphs_isolation() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    client.create_graph("g1").await;
    client.create_graph("g2").await;
    client.create_graph("g3").await;
    for _ in 0..3 { client.add_node("g1", &make_node("a")).await; }
    for _ in 0..5 { client.add_node("g2", &make_node("b")).await; }
    for _ in 0..7 { client.add_node("g3", &make_node("c")).await; }
    let (_, i1) = client.get_graph("g1").await;
    let (_, i2) = client.get_graph("g2").await;
    let (_, i3) = client.get_graph("g3").await;
    assert_eq!(i1["data"]["node_count"], 3);
    assert_eq!(i2["data"]["node_count"], 5);
    assert_eq!(i3["data"]["node_count"], 7);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Round 10: E2E Edge-case tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_e2e_malformed_json_body() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    // POST a completely invalid string as the body to the node creation endpoint.
    client.create_graph("malformed_g").await;
    let (status, _) = client
        .raw_post("/v1/graphs/malformed_g/nodes", "not json at all")
        .await;
    assert!(
        status.is_client_error(),
        "malformed JSON body should return 4xx, got {status}"
    );
}

#[tokio::test]
async fn test_e2e_empty_json_body() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    client.create_graph("empty_json_g").await;
    // POST empty JSON object (missing required `label` field).
    let (status, _) = client
        .raw_post("/v1/graphs/empty_json_g/nodes", "{}")
        .await;
    assert!(
        status.is_client_error(),
        "empty JSON body missing 'label' should return 4xx, got {status}"
    );
}

#[tokio::test]
async fn test_e2e_special_characters_in_graph_name() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let special_name = "test-graph_with.special";
    let (status, _) = client.create_graph(special_name).await;
    assert_eq!(
        status,
        StatusCode::CREATED,
        "graph with special characters should be created successfully"
    );
    // Verify it shows up in graph info.
    let (status, body) = client.get_graph(special_name).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["data"]["name"], special_name);
}

#[tokio::test]
async fn test_e2e_self_loop_via_http() {
    let (_server, client, graph) = setup("selfloop_e2e").await;
    let (_, body) = client
        .add_node(&graph, &make_node_with_props("item", json!({"name": "solo"})))
        .await;
    let nid = extract_node_id(&body);

    // Add a self-loop edge: source == target.
    let edge_req = AddEdgeRequest {
        source: nid,
        target: nid,
        label: "self_ref".to_string(),
        weight: Some(1.0),
        properties: None,
    };
    let (status, body) = client.add_edge(&graph, &edge_req).await;
    assert_eq!(status, StatusCode::CREATED, "self-loop edge should succeed");
    let eid = extract_edge_id(&body);
    assert!(eid > 0, "edge ID should be positive");

    // Verify graph info.
    let (_, info) = client.get_graph(&graph).await;
    assert_eq!(info["data"]["node_count"], 1);
    assert_eq!(info["data"]["edge_count"], 1);
}

#[tokio::test]
async fn test_e2e_bulk_nodes_with_duplicates() {
    let (_server, client, graph) = setup("bulk_dup").await;
    // Add two nodes that share the same entity_key.
    let nodes = vec![
        make_node_with_key("person", "same_key", json!({"name": "First"})),
        make_node_with_key("person", "same_key", json!({"name": "Second"})),
    ];
    let (status, body) = client.bulk_add_nodes(&graph, &nodes).await;
    // Bulk insert should succeed even with duplicate entity_keys.
    assert_eq!(
        status,
        StatusCode::CREATED,
        "bulk add with duplicate keys should succeed, got body: {body}"
    );
    // Bulk insert assigns IDs sequentially (no dedup at bulk level).
    let ids = body["data"]["node_ids"].as_array().unwrap();
    assert_eq!(ids.len(), 2, "bulk should return one ID per input node");
    // Both nodes are inserted (bulk does not apply entity_key dedup).
    let (_, info) = client.get_graph(&graph).await;
    assert_eq!(
        info["data"]["node_count"], 2,
        "bulk insert should create both nodes"
    );
}

#[tokio::test]
async fn test_e2e_context_with_large_max_depth() {
    let (_server, client, graph) = setup("deep_ctx").await;
    // Build a small chain: A -> B -> C.
    let (_, a) = client
        .add_node(&graph, &make_node_with_key("item", "a", json!({"name": "A"})))
        .await;
    let (_, b) = client
        .add_node(&graph, &make_node_with_key("item", "b", json!({"name": "B"})))
        .await;
    let (_, c) = client
        .add_node(&graph, &make_node_with_key("item", "c", json!({"name": "C"})))
        .await;
    let aid = extract_node_id(&a);
    let bid = extract_node_id(&b);
    let cid = extract_node_id(&c);
    client
        .add_edge(
            &graph,
            &AddEdgeRequest {
                source: aid,
                target: bid,
                label: "next".to_string(),
                weight: Some(0.9),
                properties: None,
            },
        )
        .await;
    client
        .add_edge(
            &graph,
            &AddEdgeRequest {
                source: bid,
                target: cid,
                label: "next".to_string(),
                weight: Some(0.9),
                properties: None,
            },
        )
        .await;

    // Context with max_depth=255 should work without error.
    let (status, body) = client
        .context(&json!({
            "graph": graph,
            "seed_nodes": ["a"],
            "max_depth": 255,
            "budget": 100000
        }))
        .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "context with large max_depth should succeed, body: {body}"
    );
    let nodes_included = body["data"]["nodes_included"].as_u64().unwrap_or(0);
    assert!(
        nodes_included > 0,
        "context should include at least the seed node"
    );
}

#[tokio::test]
async fn test_e2e_get_nonexistent_node() {
    let (_server, client, graph) = setup("get_missing_node").await;
    let (status, body) = client.get_node(&graph, 999999).await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "GET nonexistent node should return 404"
    );
    assert_eq!(body["success"], false);
}

#[tokio::test]
async fn test_e2e_delete_nonexistent_graph() {
    let server = TestServer::start().await;
    let client = WeavClient::new(&server);
    let (status, body) = client.delete_graph("nonexistent").await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "DELETE nonexistent graph should return 404"
    );
    assert_eq!(body["success"], false);
}
