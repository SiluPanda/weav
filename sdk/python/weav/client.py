"""Synchronous and asynchronous HTTP clients for the Weav context graph database."""

from __future__ import annotations

from typing import Any, Optional

import httpx

from .types import (
    ContextChunk,
    ContextResult,
    GraphInfo,
    NodeInfo,
    Provenance,
    RelationshipSummary,
)


class WeavError(Exception):
    """Exception raised when the Weav server returns an error response."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# JSON response parsing helpers
# ---------------------------------------------------------------------------

def _parse_provenance(data: dict[str, Any]) -> Provenance:
    """Parse a Provenance from a JSON dict."""
    return Provenance(
        source=data["source"],
        confidence=data["confidence"],
        extraction_method=data.get("extraction_method", "UserProvided"),
        source_document_id=data.get("source_document_id"),
        source_chunk_offset=data.get("source_chunk_offset"),
    )


def _parse_relationship_summary(data: dict[str, Any]) -> RelationshipSummary:
    """Parse a RelationshipSummary from a JSON dict."""
    return RelationshipSummary(
        edge_label=data["edge_label"],
        target_node_id=data["target_node_id"],
        target_name=data.get("target_name"),
        direction=data["direction"],
        weight=data["weight"],
    )


def _parse_context_chunk(data: dict[str, Any]) -> ContextChunk:
    """Parse a ContextChunk from a JSON dict."""
    provenance = None
    if data.get("provenance") is not None:
        provenance = _parse_provenance(data["provenance"])

    relationships = [
        _parse_relationship_summary(r) for r in data.get("relationships", [])
    ]

    return ContextChunk(
        node_id=data["node_id"],
        content=data["content"],
        label=data["label"],
        relevance_score=data["relevance_score"],
        depth=data["depth"],
        token_count=data["token_count"],
        provenance=provenance,
        relationships=relationships,
    )


def _parse_context_result(data: dict[str, Any]) -> ContextResult:
    """Parse a ContextResult from the ``data`` field of an API response."""
    chunks = [_parse_context_chunk(c) for c in data.get("chunks", [])]
    return ContextResult(
        chunks=chunks,
        total_tokens=data["total_tokens"],
        budget_used=data["budget_used"],
        nodes_considered=data["nodes_considered"],
        nodes_included=data["nodes_included"],
        query_time_us=data["query_time_us"],
    )


def _parse_graph_info(data: dict[str, Any]) -> GraphInfo:
    """Parse a GraphInfo from the ``data`` field of an API response."""
    return GraphInfo(
        name=data["name"],
        node_count=data["node_count"],
        edge_count=data["edge_count"],
    )


def _parse_node_info(data: dict[str, Any]) -> NodeInfo:
    """Parse a NodeInfo from the ``data`` field of an API response."""
    return NodeInfo(
        node_id=data["node_id"],
        label=data["label"],
        properties=data.get("properties", {}),
    )


def _check_response(response: httpx.Response) -> Any:
    """Validate an HTTP response and return the ``data`` field.

    Raises :class:`WeavError` if the server returned an error.
    """
    # Attempt to parse as JSON regardless of status code.
    try:
        body = response.json()
    except Exception:
        response.raise_for_status()
        return None

    if not body.get("success", False):
        error_msg = body.get("error", "Unknown server error")
        raise WeavError(error_msg, status_code=response.status_code)

    return body.get("data")


# ---------------------------------------------------------------------------
# Synchronous client
# ---------------------------------------------------------------------------

class WeavClient:
    """Synchronous Weav client using HTTP.

    Usage::

        with WeavClient() as client:
            client.create_graph("my_graph")
            node_id = client.add_node("my_graph", "person", {"name": "Alice"})
            result = client.context("my_graph", query="who is alice", seed_nodes=["alice"])
            print(result.to_prompt())
    """

    def __init__(self, host: str = "localhost", port: int = 6382, *, timeout: float = 30.0):
        self.base_url = f"http://{host}:{port}"
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> WeavClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # -- Graph management --------------------------------------------------

    def create_graph(self, name: str) -> None:
        """Create a new graph with the given name."""
        resp = self._client.post("/v1/graphs", json={"name": name})
        _check_response(resp)

    def drop_graph(self, name: str) -> None:
        """Drop (delete) a graph by name."""
        resp = self._client.delete(f"/v1/graphs/{name}")
        _check_response(resp)

    def list_graphs(self) -> list[str]:
        """List the names of all graphs on the server."""
        resp = self._client.get("/v1/graphs")
        data = _check_response(resp)
        return data if data is not None else []

    def graph_info(self, name: str) -> GraphInfo:
        """Get metadata about a graph (node/edge counts)."""
        resp = self._client.get(f"/v1/graphs/{name}")
        data = _check_response(resp)
        return _parse_graph_info(data)

    # -- Node operations ---------------------------------------------------

    def add_node(
        self,
        graph: str,
        label: str,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        entity_key: str | None = None,
    ) -> int:
        """Add a node to a graph. Returns the new node ID."""
        body: dict[str, Any] = {"label": label}
        if properties is not None:
            body["properties"] = properties
        if embedding is not None:
            body["embedding"] = embedding
        if entity_key is not None:
            body["entity_key"] = entity_key

        resp = self._client.post(f"/v1/graphs/{graph}/nodes", json=body)
        data = _check_response(resp)
        return data["id"]

    def get_node(self, graph: str, node_id: int) -> NodeInfo:
        """Get a node by its ID."""
        resp = self._client.get(f"/v1/graphs/{graph}/nodes/{node_id}")
        data = _check_response(resp)
        return _parse_node_info(data)

    def delete_node(self, graph: str, node_id: int) -> None:
        """Delete a node by its ID (also removes connected edges)."""
        resp = self._client.delete(f"/v1/graphs/{graph}/nodes/{node_id}")
        _check_response(resp)

    def update_node(
        self,
        graph: str,
        node_id: int,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        """Update a node's properties and/or embedding."""
        body: dict[str, Any] = {}
        if properties is not None:
            body["properties"] = properties
        if embedding is not None:
            body["embedding"] = embedding
        resp = self._client.put(f"/v1/graphs/{graph}/nodes/{node_id}", json=body)
        _check_response(resp)

    # -- Edge operations ---------------------------------------------------

    def add_edge(
        self,
        graph: str,
        source: int,
        target: int,
        label: str,
        weight: float = 1.0,
        provenance: dict | None = None,
    ) -> int:
        """Add an edge between two nodes. Returns the new edge ID."""
        body: dict[str, Any] = {
            "source": source,
            "target": target,
            "label": label,
            "weight": weight,
        }
        if provenance is not None:
            body["provenance"] = provenance
        resp = self._client.post(f"/v1/graphs/{graph}/edges", json=body)
        data = _check_response(resp)
        return data["id"]

    def invalidate_edge(self, graph: str, edge_id: int) -> None:
        """Invalidate (soft-delete) an edge by its ID."""
        resp = self._client.post(f"/v1/graphs/{graph}/edges/{edge_id}/invalidate")
        _check_response(resp)

    def bulk_add_nodes(
        self,
        graph: str,
        nodes: list[dict[str, Any]],
    ) -> list[int]:
        """Bulk add nodes. Each dict needs at least 'label'. Returns list of IDs."""
        resp = self._client.post(f"/v1/graphs/{graph}/nodes/bulk", json={"nodes": nodes})
        data = _check_response(resp)
        return data["ids"]

    def bulk_add_edges(
        self,
        graph: str,
        edges: list[dict[str, Any]],
    ) -> list[int]:
        """Bulk add edges. Each dict needs 'source', 'target', 'label'. Returns list of IDs."""
        resp = self._client.post(f"/v1/graphs/{graph}/edges/bulk", json={"edges": edges})
        data = _check_response(resp)
        return data["ids"]

    # -- Context retrieval -------------------------------------------------

    def context(
        self,
        graph: str,
        query: str | None = None,
        embedding: list[float] | None = None,
        seed_nodes: list[str] | None = None,
        budget: int = 4096,
        max_depth: int = 3,
        decay: str | None = None,
        edge_labels: list[str] | None = None,
        temporal_at: str | None = None,
        include_provenance: bool = False,
    ) -> ContextResult:
        """Run a context query against a graph.

        At least one of *query*, *embedding*, or *seed_nodes* should be
        provided so the engine can determine seed nodes for traversal.
        """
        body: dict[str, Any] = {
            "graph": graph,
            "budget": budget,
            "max_depth": max_depth,
            "include_provenance": include_provenance,
        }
        if query is not None:
            body["query"] = query
        if embedding is not None:
            body["embedding"] = embedding
        if seed_nodes is not None:
            body["seed_nodes"] = seed_nodes
        if decay is not None:
            body["decay"] = decay
        if edge_labels is not None:
            body["edge_labels"] = edge_labels
        if temporal_at is not None:
            body["temporal_at"] = temporal_at

        resp = self._client.post("/v1/context", json=body)
        data = _check_response(resp)
        return _parse_context_result(data)

    # -- Health ------------------------------------------------------------

    def health(self) -> bool:
        """Check if the server is healthy. Returns True when reachable and OK."""
        try:
            resp = self._client.get("/health")
            data = _check_response(resp)
            return data is not None and data.get("status") == "ok"
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Asynchronous client
# ---------------------------------------------------------------------------

class AsyncWeavClient:
    """Asynchronous Weav client using HTTP.

    Usage::

        async with AsyncWeavClient() as client:
            await client.create_graph("my_graph")
            node_id = await client.add_node("my_graph", "person", {"name": "Alice"})
            result = await client.context("my_graph", query="who is alice")
            print(result.to_prompt())
    """

    def __init__(self, host: str = "localhost", port: int = 6382, *, timeout: float = 30.0):
        self.base_url = f"http://{host}:{port}"
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> AsyncWeavClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # -- Graph management --------------------------------------------------

    async def create_graph(self, name: str) -> None:
        """Create a new graph with the given name."""
        resp = await self._client.post("/v1/graphs", json={"name": name})
        _check_response(resp)

    async def drop_graph(self, name: str) -> None:
        """Drop (delete) a graph by name."""
        resp = await self._client.delete(f"/v1/graphs/{name}")
        _check_response(resp)

    async def list_graphs(self) -> list[str]:
        """List the names of all graphs on the server."""
        resp = await self._client.get("/v1/graphs")
        data = _check_response(resp)
        return data if data is not None else []

    async def graph_info(self, name: str) -> GraphInfo:
        """Get metadata about a graph (node/edge counts)."""
        resp = await self._client.get(f"/v1/graphs/{name}")
        data = _check_response(resp)
        return _parse_graph_info(data)

    # -- Node operations ---------------------------------------------------

    async def add_node(
        self,
        graph: str,
        label: str,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        entity_key: str | None = None,
    ) -> int:
        """Add a node to a graph. Returns the new node ID."""
        body: dict[str, Any] = {"label": label}
        if properties is not None:
            body["properties"] = properties
        if embedding is not None:
            body["embedding"] = embedding
        if entity_key is not None:
            body["entity_key"] = entity_key

        resp = await self._client.post(f"/v1/graphs/{graph}/nodes", json=body)
        data = _check_response(resp)
        return data["id"]

    async def get_node(self, graph: str, node_id: int) -> NodeInfo:
        """Get a node by its ID."""
        resp = await self._client.get(f"/v1/graphs/{graph}/nodes/{node_id}")
        data = _check_response(resp)
        return _parse_node_info(data)

    async def delete_node(self, graph: str, node_id: int) -> None:
        """Delete a node by its ID (also removes connected edges)."""
        resp = await self._client.delete(f"/v1/graphs/{graph}/nodes/{node_id}")
        _check_response(resp)

    async def update_node(
        self,
        graph: str,
        node_id: int,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        """Update a node's properties and/or embedding."""
        body: dict[str, Any] = {}
        if properties is not None:
            body["properties"] = properties
        if embedding is not None:
            body["embedding"] = embedding
        resp = await self._client.put(f"/v1/graphs/{graph}/nodes/{node_id}", json=body)
        _check_response(resp)

    # -- Edge operations ---------------------------------------------------

    async def add_edge(
        self,
        graph: str,
        source: int,
        target: int,
        label: str,
        weight: float = 1.0,
        provenance: dict | None = None,
    ) -> int:
        """Add an edge between two nodes. Returns the new edge ID."""
        body: dict[str, Any] = {
            "source": source,
            "target": target,
            "label": label,
            "weight": weight,
        }
        if provenance is not None:
            body["provenance"] = provenance
        resp = await self._client.post(f"/v1/graphs/{graph}/edges", json=body)
        data = _check_response(resp)
        return data["id"]

    async def invalidate_edge(self, graph: str, edge_id: int) -> None:
        """Invalidate (soft-delete) an edge by its ID."""
        resp = await self._client.post(
            f"/v1/graphs/{graph}/edges/{edge_id}/invalidate"
        )
        _check_response(resp)

    async def bulk_add_nodes(
        self,
        graph: str,
        nodes: list[dict[str, Any]],
    ) -> list[int]:
        """Bulk add nodes. Each dict needs at least 'label'. Returns list of IDs."""
        resp = await self._client.post(f"/v1/graphs/{graph}/nodes/bulk", json={"nodes": nodes})
        data = _check_response(resp)
        return data["ids"]

    async def bulk_add_edges(
        self,
        graph: str,
        edges: list[dict[str, Any]],
    ) -> list[int]:
        """Bulk add edges. Each dict needs 'source', 'target', 'label'. Returns list of IDs."""
        resp = await self._client.post(f"/v1/graphs/{graph}/edges/bulk", json={"edges": edges})
        data = _check_response(resp)
        return data["ids"]

    # -- Context retrieval -------------------------------------------------

    async def context(
        self,
        graph: str,
        query: str | None = None,
        embedding: list[float] | None = None,
        seed_nodes: list[str] | None = None,
        budget: int = 4096,
        max_depth: int = 3,
        decay: str | None = None,
        edge_labels: list[str] | None = None,
        temporal_at: str | None = None,
        include_provenance: bool = False,
    ) -> ContextResult:
        """Run a context query against a graph.

        At least one of *query*, *embedding*, or *seed_nodes* should be
        provided so the engine can determine seed nodes for traversal.
        """
        body: dict[str, Any] = {
            "graph": graph,
            "budget": budget,
            "max_depth": max_depth,
            "include_provenance": include_provenance,
        }
        if query is not None:
            body["query"] = query
        if embedding is not None:
            body["embedding"] = embedding
        if seed_nodes is not None:
            body["seed_nodes"] = seed_nodes
        if decay is not None:
            body["decay"] = decay
        if edge_labels is not None:
            body["edge_labels"] = edge_labels
        if temporal_at is not None:
            body["temporal_at"] = temporal_at

        resp = await self._client.post("/v1/context", json=body)
        data = _check_response(resp)
        return _parse_context_result(data)

    # -- Health ------------------------------------------------------------

    async def health(self) -> bool:
        """Check if the server is healthy. Returns True when reachable and OK."""
        try:
            resp = await self._client.get("/health")
            data = _check_response(resp)
            return data is not None and data.get("status") == "ok"
        except Exception:
            return False
