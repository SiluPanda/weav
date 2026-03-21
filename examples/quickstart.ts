/**
 * Weav Quickstart — TypeScript/Node.js
 *
 * Prerequisites:
 *   npm install  (from sdk/node/)
 *   Start Weav: docker compose up
 *
 * Run: npx tsx examples/quickstart.ts
 */

const BASE = "http://localhost:6382/v1";

async function main() {
  // 1. Create graph
  await fetch(`${BASE}/graphs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: "quickstart" }),
  });
  console.log("Graph created.");

  // 2. Add nodes
  const alice = await addNode("Person", { name: "Alice", role: "Engineer", expertise: "graph databases and AI" });
  const bob = await addNode("Person", { name: "Bob", role: "Researcher", expertise: "NLP and LLMs" });
  const paper = await addNode("Document", { name: "GraphRAG Paper", content: "Novel approach to RAG using knowledge graphs" });
  console.log(`Added nodes: Alice=${alice}, Bob=${bob}, Paper=${paper}`);

  // 3. Add edges
  await addEdge(alice, bob, "COLLABORATES_WITH", 0.9);
  await addEdge(alice, paper, "AUTHORED", 1.0);
  await addEdge(bob, paper, "REVIEWED", 0.8);
  console.log("Added 3 edges.");

  // 4. Full-text search
  const search = await get(`/graphs/quickstart/search/text?q=graph+databases&limit=5`);
  console.log(`\nBM25 search: ${search.data.count} results`);

  // 5. PageRank
  const pr = await post("/graphs/quickstart/algorithms/pagerank", { damping: 0.85 });
  console.log("\nPageRank:");
  pr.data.scores.slice(0, 3).forEach((s: any) => console.log(`  Node ${s.node_id}: ${s.score.toFixed(4)}`));

  // 6. Graph info
  const info = await get("/graphs/quickstart");
  console.log(`\nGraph: ${info.data.node_count} nodes, ${info.data.edge_count} edges`);

  console.log("\nDone!");
}

async function addNode(label: string, properties: Record<string, string>): Promise<number> {
  const res = await post("/graphs/quickstart/nodes", { label, properties });
  return res.data.node_id;
}

async function addEdge(source: number, target: number, label: string, weight: number) {
  await post("/graphs/quickstart/edges", { source, target, label, weight });
}

async function get(path: string) {
  const res = await fetch(`${BASE}${path}`);
  return res.json();
}

async function post(path: string, body: any) {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

main().catch(console.error);
