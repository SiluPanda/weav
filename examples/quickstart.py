#!/usr/bin/env python3
"""Weav Quickstart — Build a knowledge graph and query it in under 50 lines.

Prerequisites:
    pip install httpx
    # Start Weav: docker compose up  (or cargo run --release -p weav-server)

This example:
1. Creates a knowledge graph
2. Adds nodes with properties and embeddings
3. Adds relationships between nodes
4. Runs a context query with token budget
5. Runs graph algorithms (PageRank, community detection)
6. Performs full-text search
"""

import httpx

BASE = "http://localhost:6382/v1"
client = httpx.Client(base_url=BASE)

# 1. Create a graph
client.post("/graphs", json={"name": "quickstart"})
print("Graph created.")

# 2. Add nodes
alice = client.post("/graphs/quickstart/nodes", json={
    "label": "Person",
    "properties": {"name": "Alice", "role": "Engineer", "expertise": "graph databases and AI systems"},
}).json()["data"]["node_id"]

bob = client.post("/graphs/quickstart/nodes", json={
    "label": "Person",
    "properties": {"name": "Bob", "role": "Researcher", "expertise": "natural language processing"},
}).json()["data"]["node_id"]

paper = client.post("/graphs/quickstart/nodes", json={
    "label": "Document",
    "properties": {"name": "GraphRAG Paper", "content": "A novel approach to retrieval-augmented generation using knowledge graphs for context-aware AI systems."},
}).json()["data"]["node_id"]

project = client.post("/graphs/quickstart/nodes", json={
    "label": "Project",
    "properties": {"name": "Weav", "description": "In-memory context graph database for AI/LLM workloads"},
}).json()["data"]["node_id"]

print(f"Added 4 nodes: Alice={alice}, Bob={bob}, Paper={paper}, Project={project}")

# 3. Add edges (relationships)
client.post("/graphs/quickstart/edges", json={"source": alice, "target": bob, "label": "COLLABORATES_WITH", "weight": 0.9})
client.post("/graphs/quickstart/edges", json={"source": alice, "target": paper, "label": "AUTHORED", "weight": 1.0})
client.post("/graphs/quickstart/edges", json={"source": bob, "target": paper, "label": "REVIEWED", "weight": 0.8})
client.post("/graphs/quickstart/edges", json={"source": alice, "target": project, "label": "LEADS", "weight": 1.0})
client.post("/graphs/quickstart/edges", json={"source": bob, "target": project, "label": "CONTRIBUTES_TO", "weight": 0.7})
print("Added 5 edges.")

# 4. Full-text search (BM25)
results = client.get("/graphs/quickstart/search/text", params={"q": "graph database AI", "limit": 5}).json()
print(f"\nBM25 search 'graph database AI': {results['data']['count']} results")
for match in results["data"]["matches"]:
    print(f"  {match['label']}: {match['node_id']} (score={match['score']:.3f})")

# 5. Run PageRank algorithm
pr = client.post("/graphs/quickstart/algorithms/pagerank", json={"damping": 0.85, "iterations": 20}).json()
print(f"\nPageRank scores:")
for node in pr["data"]["scores"][:4]:
    print(f"  Node {node['node_id']}: {node['score']:.4f}")

# 6. Community detection
communities = client.post("/graphs/quickstart/algorithms/communities", json={"resolution": 1.0}).json()
print(f"\nCommunities found: {len(communities['data']['communities'])}")

# 7. Graph info
info = client.get("/graphs/quickstart").json()["data"]
print(f"\nGraph stats: {info['node_count']} nodes, {info['edge_count']} edges")

# 8. Node importance scoring
importance = client.get(f"/graphs/quickstart/importance", params={"limit": 4}).json()
print(f"\nTop nodes by importance:")
for node in importance["data"]["nodes"]:
    print(f"  Node {node['node_id']}: importance={node['importance']:.3f}")

# 9. Graph integrity check
check = client.get("/graphs/quickstart/check").json()
print(f"\nIntegrity check: {check['data']}")

print("\nDone! Weav quickstart complete.")
