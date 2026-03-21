#!/usr/bin/env python3
"""Weav Context Query Example — Token-budget-aware retrieval for LLMs.

This example demonstrates Weav's unique capabilities:
1. Token budget management (no other graph DB has this)
2. Triple-fusion retrieval (vector + BM25 + graph)
3. LLM-ready output formatting (Anthropic/OpenAI native)
4. Context subgraph extraction (graph structure, not just text)
5. Query EXPLAIN for profiling
"""

import httpx
import json

BASE = "http://localhost:6382/v1"
client = httpx.Client(base_url=BASE)

# Setup: create graph with some knowledge
client.post("/graphs", json={"name": "knowledge"})

# Add entities
for i, (label, name, content) in enumerate([
    ("Concept", "GraphRAG", "Graph-based retrieval augmented generation combines knowledge graphs with LLMs"),
    ("Concept", "Token Budget", "Managing the finite context window of LLMs by selecting the most relevant content"),
    ("Concept", "Vector Search", "Finding similar items using embedding vectors and HNSW indexes"),
    ("Concept", "BM25", "Best Match 25 algorithm for keyword-based text relevance scoring"),
    ("Person", "Alice", "Senior engineer specializing in knowledge graphs and AI systems"),
    ("Document", "Design Doc", "Weav uses triple-fusion retrieval combining vector similarity, BM25 text matching, and graph flow scoring"),
]):
    client.post("/graphs/knowledge/nodes", json={
        "label": label, "properties": {"name": name, "content": content},
        "entity_key": name.lower().replace(" ", "_"),
    })

# Add relationships
nodes = {m["label"]: m["node_id"] for m in
    client.get("/graphs/knowledge/search", params={"key": "name", "value": "GraphRAG"}).json()["data"]["matches"]}

print("=== 1. Context Query with Budget Preset ===")
result = client.post("/context", json={
    "graph": "knowledge",
    "query": "How does graph-based retrieval work?",
    "seed_keys": ["graphrag"],
    "budget_preset": "small",  # 4K tokens — named preset
    "max_depth": 2,
}).json()
print(f"Tokens used: {result['data']['total_tokens']}/{4096}")
print(f"Budget utilization: {result['data']['budget_used']:.1%}")
print(f"Nodes considered: {result['data']['nodes_considered']}")
print(f"Nodes included: {result['data']['nodes_included']}")
for chunk in result["data"]["chunks"][:3]:
    print(f"  [{chunk['label']}] {chunk['content'][:80]}...")

print("\n=== 2. Context Query with LLM Output Format ===")
result = client.post("/context", json={
    "graph": "knowledge",
    "query": "Explain token budget management",
    "seed_keys": ["token_budget"],
    "budget_preset": "medium",  # 8K tokens
    "output_format": "anthropic",  # Returns Anthropic API message format
}).json()
if result["data"].get("formatted_messages"):
    messages = json.loads(result["data"]["formatted_messages"])
    print(f"Anthropic format: {len(messages['messages'])} message(s)")
    print(f"  Role: {messages['messages'][0]['role']}")
    print(f"  Content preview: {messages['messages'][0]['content'][:100]}...")

print("\n=== 3. Context Query with Subgraph ===")
result = client.post("/context", json={
    "graph": "knowledge",
    "query": "Knowledge graph concepts",
    "seed_keys": ["graphrag"],
    "budget_preset": "small",
    "include_subgraph": True,  # Return graph structure
}).json()
subgraph = result["data"].get("subgraph")
if subgraph:
    print(f"Subgraph: {len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges")
    for node in subgraph["nodes"]:
        print(f"  Node {node['node_id']}: {node['label']} (importance={node['importance']:.3f})")

print("\n=== 4. Query EXPLAIN (without executing) ===")
result = client.post("/context", json={
    "graph": "knowledge",
    "query": "How does retrieval work?",
    "seed_keys": ["graphrag"],
    "budget_preset": "large",
    "explain": True,  # Return plan without executing
}).json()
if result["data"].get("plan"):
    print("Query plan:")
    for line in result["data"]["plan"].split("\n"):
        print(f"  {line}")

print("\nDone! These features are unique to Weav — no other graph database offers them.")
