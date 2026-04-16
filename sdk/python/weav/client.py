"""Synchronous and asynchronous HTTP clients for the Weav context graph database."""

from __future__ import annotations

from typing import Any, Optional

import httpx

from .types import (
    ContextChunk,
    ContextResult,
    GraphRef,
    GraphInfo,
    IngestParams,
    IngestResult,
    NodeInfo,
    Provenance,
    RerankConfig,
    RetrievalMode,
    ResolutionMode,
    RelationshipSummary,
    ScopeRef,
    scope_to_graph,
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


def _parse_ingest_result(data: dict[str, Any]) -> IngestResult:
    """Parse an IngestResult from the ``data`` field of an API response."""
    return IngestResult(
        document_id=data["document_id"],
        total_tokens=data.get("total_tokens", 0),
        chunks_created=data["chunks_created"],
        entities_created=data["entities_created"],
        entities_merged=data["entities_merged"],
        entities_resolved=data.get("entities_resolved", 0),
        entities_linked_existing=data.get("entities_linked_existing", 0),
        relationships_created=data["relationships_created"],
        pipeline_duration_ms=data["pipeline_duration_ms"],
    )


def _serialize_resolution_mode(
    resolution_mode: ResolutionMode | str,
) -> str:
    """Serialize a resolution mode into the server's JSON shape."""
    if isinstance(resolution_mode, ResolutionMode):
        return resolution_mode.value
    return resolution_mode


def _build_ingest_body(
    *,
    content: str | None = None,
    content_base64: str | None = None,
    format: str | None = None,
    document_id: str | None = None,
    skip_extraction: bool | None = None,
    skip_dedup: bool | None = None,
    chunk_size: int | None = None,
    entity_types: list[str] | None = None,
    resolution_mode: ResolutionMode | str | None = None,
    link_existing_entities: bool | None = None,
    resolution_candidate_limit: int | None = None,
    custom_resolution_prompt: str | None = None,
    params: IngestParams | None = None,
) -> dict[str, Any]:
    """Build the ingest request body from kwargs and/or a params object."""
    body: dict[str, Any] = {}

    def set_if_present(key: str, value: Any) -> None:
        if value is not None and key not in body:
            body[key] = value

    set_if_present("content", content)
    set_if_present("content_base64", content_base64)
    set_if_present("format", format)
    set_if_present("document_id", document_id)
    set_if_present("skip_extraction", skip_extraction)
    set_if_present("skip_dedup", skip_dedup)
    set_if_present("chunk_size", chunk_size)
    set_if_present("entity_types", entity_types)
    if resolution_mode is not None:
        set_if_present("resolution_mode", _serialize_resolution_mode(resolution_mode))
    set_if_present("link_existing_entities", link_existing_entities)
    set_if_present("resolution_candidate_limit", resolution_candidate_limit)
    set_if_present("custom_resolution_prompt", custom_resolution_prompt)

    if params is not None:
        set_if_present("content", params.content)
        set_if_present("content_base64", params.content_base64)
        set_if_present("format", params.format)
        set_if_present("document_id", params.document_id)
        set_if_present("skip_extraction", params.skip_extraction)
        set_if_present("skip_dedup", params.skip_dedup)
        set_if_present("chunk_size", params.chunk_size)
        set_if_present("entity_types", params.entity_types)
        if params.resolution_mode is not None:
            set_if_present(
                "resolution_mode",
                _serialize_resolution_mode(params.resolution_mode),
            )
        set_if_present("link_existing_entities", params.link_existing_entities)
        set_if_present(
            "resolution_candidate_limit",
            params.resolution_candidate_limit,
        )
        set_if_present(
            "custom_resolution_prompt",
            params.custom_resolution_prompt,
        )

    return body


def _serialize_rerank(
    rerank: RerankConfig | dict[str, Any],
) -> dict[str, Any]:
    """Serialize a rerank config into the server's JSON shape."""
    if isinstance(rerank, RerankConfig):
        return {
            "enabled": rerank.enabled,
            "provider": rerank.provider,
            "model": rerank.model,
            "candidate_limit": rerank.candidate_limit,
            "score_weight": rerank.score_weight,
        }
    return rerank


def _serialize_scope(scope: ScopeRef | dict[str, Any]) -> dict[str, Any]:
    """Serialize a scope object into the server's JSON shape."""
    if isinstance(scope, ScopeRef):
        payload: dict[str, Any] = {"workspace_id": scope.workspace_id}
        if scope.user_id is not None:
            payload["user_id"] = scope.user_id
        if scope.agent_id is not None:
            payload["agent_id"] = scope.agent_id
        if scope.session_id is not None:
            payload["session_id"] = scope.session_id
        return payload
    return dict(scope)


def _graph_name(graph: GraphRef) -> str:
    """Resolve a graph-like input into a graph name."""
    if isinstance(graph, str):
        return graph
    return scope_to_graph(graph)


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

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6382,
        *,
        timeout: float = 30.0,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        self.base_url = f"http://{host}:{port}"
        headers: dict[str, str] = {}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        elif username is not None and password is not None:
            import base64
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> WeavClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # -- Graph management --------------------------------------------------

    def create_graph(self, name: GraphRef) -> None:
        """Create a new graph with the given name."""
        if isinstance(name, str):
            body = {"name": name}
        else:
            scope_to_graph(name)
            body = {"scope": _serialize_scope(name)}
        resp = self._client.post("/v1/graphs", json=body)
        _check_response(resp)

    def drop_graph(self, name: GraphRef) -> None:
        """Drop (delete) a graph by name."""
        resp = self._client.delete(f"/v1/graphs/{_graph_name(name)}")
        _check_response(resp)

    def list_graphs(self) -> list[str]:
        """List the names of all graphs on the server."""
        resp = self._client.get("/v1/graphs")
        data = _check_response(resp)
        return data if data is not None else []

    def graph_info(self, name: GraphRef) -> GraphInfo:
        """Get metadata about a graph (node/edge counts)."""
        resp = self._client.get(f"/v1/graphs/{_graph_name(name)}")
        data = _check_response(resp)
        return _parse_graph_info(data)

    # -- Node operations ---------------------------------------------------

    def add_node(
        self,
        graph: GraphRef,
        label: str,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        entity_key: str | None = None,
    ) -> int:
        """Add a node to a graph. Returns the new node ID."""
        graph_name = _graph_name(graph)
        body: dict[str, Any] = {"label": label}
        if properties is not None:
            body["properties"] = properties
        if embedding is not None:
            body["embedding"] = embedding
        if entity_key is not None:
            body["entity_key"] = entity_key

        resp = self._client.post(f"/v1/graphs/{graph_name}/nodes", json=body)
        data = _check_response(resp)
        return data["node_id"]

    def get_node(self, graph: GraphRef, node_id: int) -> NodeInfo:
        """Get a node by its ID."""
        resp = self._client.get(f"/v1/graphs/{_graph_name(graph)}/nodes/{node_id}")
        data = _check_response(resp)
        return _parse_node_info(data)

    def delete_node(self, graph: GraphRef, node_id: int) -> None:
        """Delete a node by its ID (also removes connected edges)."""
        resp = self._client.delete(f"/v1/graphs/{_graph_name(graph)}/nodes/{node_id}")
        _check_response(resp)

    def update_node(
        self,
        graph: GraphRef,
        node_id: int,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        """Update a node's properties and/or embedding."""
        graph_name = _graph_name(graph)
        body: dict[str, Any] = {}
        if properties is not None:
            body["properties"] = properties
        if embedding is not None:
            body["embedding"] = embedding
        resp = self._client.put(f"/v1/graphs/{graph_name}/nodes/{node_id}", json=body)
        _check_response(resp)

    # -- Edge operations ---------------------------------------------------

    def add_edge(
        self,
        graph: GraphRef,
        source: int,
        target: int,
        label: str,
        weight: float = 1.0,
        provenance: dict | None = None,
    ) -> int:
        """Add an edge between two nodes. Returns the new edge ID."""
        graph_name = _graph_name(graph)
        body: dict[str, Any] = {
            "source": source,
            "target": target,
            "label": label,
            "weight": weight,
        }
        if provenance is not None:
            body["provenance"] = provenance
        resp = self._client.post(f"/v1/graphs/{graph_name}/edges", json=body)
        data = _check_response(resp)
        return data["edge_id"]

    def invalidate_edge(self, graph: GraphRef, edge_id: int) -> None:
        """Invalidate (soft-delete) an edge by its ID."""
        resp = self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/edges/{edge_id}/invalidate"
        )
        _check_response(resp)

    def bulk_add_nodes(
        self,
        graph: GraphRef,
        nodes: list[dict[str, Any]],
    ) -> list[int]:
        """Bulk add nodes. Each dict needs at least 'label'. Returns list of IDs."""
        resp = self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/nodes/bulk",
            json={"nodes": nodes},
        )
        data = _check_response(resp)
        return data["node_ids"]

    def bulk_add_edges(
        self,
        graph: GraphRef,
        edges: list[dict[str, Any]],
    ) -> list[int]:
        """Bulk add edges. Each dict needs 'source', 'target', 'label'. Returns list of IDs."""
        resp = self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/edges/bulk",
            json={"edges": edges},
        )
        data = _check_response(resp)
        return data["edge_ids"]

    # -- Context retrieval -------------------------------------------------

    def context(
        self,
        graph: GraphRef,
        query: str | None = None,
        retrieval_mode: RetrievalMode | str | None = None,
        rerank: RerankConfig | dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        seed_nodes: list[str] | None = None,
        budget: int = 4096,
        max_depth: int = 3,
        decay: dict[str, Any] | None = None,
        edge_labels: list[str] | None = None,
        temporal_at: int | None = None,
        include_provenance: bool = False,
        limit: int | None = None,
        sort_field: str | None = None,
        sort_direction: str | None = None,
        direction: str | None = None,
    ) -> ContextResult:
        """Run a context query against a graph.

        At least one of *query*, *embedding*, or *seed_nodes* should be
        provided so the engine can determine seed nodes for traversal.

        *decay* should be a dict like ``{"type": "exponential", "half_life_ms": 3600000}``.
        """
        body: dict[str, Any] = {
            "budget": budget,
            "max_depth": max_depth,
            "include_provenance": include_provenance,
        }
        if isinstance(graph, str):
            body["graph"] = graph
        else:
            scope_to_graph(graph)
            body["scope"] = _serialize_scope(graph)
        if query is not None:
            body["query"] = query
        if retrieval_mode is not None:
            body["retrieval_mode"] = (
                retrieval_mode.value
                if isinstance(retrieval_mode, RetrievalMode)
                else retrieval_mode
            )
        if rerank is not None:
            body["rerank"] = _serialize_rerank(rerank)
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
        if limit is not None:
            body["limit"] = limit
        if sort_field is not None:
            body["sort_field"] = sort_field
        if sort_direction is not None:
            body["sort_direction"] = sort_direction
        if direction is not None:
            body["direction"] = direction

        resp = self._client.post("/v1/context", json=body)
        data = _check_response(resp)
        return _parse_context_result(data)

    # -- Ingest (extraction pipeline) --------------------------------------

    def ingest(
        self,
        graph: GraphRef,
        content: str | None = None,
        content_base64: str | None = None,
        format: str | None = None,
        document_id: str | None = None,
        skip_extraction: bool | None = None,
        skip_dedup: bool | None = None,
        chunk_size: int | None = None,
        entity_types: list[str] | None = None,
        resolution_mode: ResolutionMode | str | None = None,
        link_existing_entities: bool | None = None,
        resolution_candidate_limit: int | None = None,
        custom_resolution_prompt: str | None = None,
        params: IngestParams | None = None,
    ) -> IngestResult:
        """Ingest a document into a graph via the extraction pipeline.

        Provide either *content* (UTF-8 text) or *content_base64* (base64-encoded
        binary). The server will parse, chunk, embed, extract entities/relationships,
        and build the graph automatically.
        """
        graph_name = _graph_name(graph)
        body = _build_ingest_body(
            content=content,
            content_base64=content_base64,
            format=format,
            document_id=document_id,
            skip_extraction=skip_extraction,
            skip_dedup=skip_dedup,
            chunk_size=chunk_size,
            entity_types=entity_types,
            resolution_mode=resolution_mode,
            link_existing_entities=link_existing_entities,
            resolution_candidate_limit=resolution_candidate_limit,
            custom_resolution_prompt=custom_resolution_prompt,
            params=params,
        )

        resp = self._client.post(f"/v1/graphs/{graph_name}/ingest", json=body)
        data = _check_response(resp)
        return _parse_ingest_result(data)

    # -- Search ------------------------------------------------------------

    def search_text(self, graph: GraphRef, query: str, limit: int = 20) -> dict:
        """Full-text BM25 search across node content."""
        resp = self._client.get(
            f"/v1/graphs/{_graph_name(graph)}/search/text",
            params={"q": query, "limit": limit},
        )
        return _check_response(resp)

    # -- Node merge --------------------------------------------------------

    def merge_nodes(
        self,
        graph: GraphRef,
        source_id: int,
        target_id: int,
        conflict_policy: str = "keep_target",
    ) -> dict:
        """Merge two nodes, re-linking edges from source to target."""
        resp = self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/nodes/merge",
            json={
                "source_id": source_id,
                "target_id": target_id,
                "conflict_policy": conflict_policy,
            },
        )
        return _check_response(resp)

    # -- Algorithms --------------------------------------------------------

    def search_vector(
        self,
        graph: GraphRef,
        embedding: list[float],
        k: int = 10,
        labels: list[str] | None = None,
        properties: dict | None = None,
    ) -> dict:
        """Vector similarity search with optional filtering."""
        body: dict[str, Any] = {"embedding": embedding, "k": k}
        if labels is not None:
            body["labels"] = labels
        if properties is not None:
            body["properties"] = properties
        resp = self._client.post(f"/v1/graphs/{_graph_name(graph)}/search/vector", json=body)
        return _check_response(resp)

    # -- Graph diff --------------------------------------------------------

    def graph_diff(self, graph: GraphRef, from_timestamp: int, to_timestamp: int) -> dict:
        """Compare graph state between two timestamps."""
        resp = self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/diff",
            json={"from_timestamp": from_timestamp, "to_timestamp": to_timestamp},
        )
        return _check_response(resp)

    # -- Community operations ----------------------------------------------

    def community_summarize(
        self,
        graph: GraphRef,
        algorithm: str = "leiden",
        resolution: float = 1.0,
    ) -> dict:
        """Run community detection and generate summaries."""
        resp = self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/communities/summarize",
            json={"algorithm": algorithm, "resolution": resolution},
        )
        return _check_response(resp)

    def community_summaries(self, graph: GraphRef) -> dict:
        """Get all community summaries."""
        resp = self._client.get(f"/v1/graphs/{_graph_name(graph)}/communities/summaries")
        return _check_response(resp)

    def community_search(self, graph: GraphRef, query: str, limit: int = 10) -> dict:
        """Search community summaries."""
        resp = self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/communities/search",
            json={"query": query, "limit": limit},
        )
        return _check_response(resp)

    # -- Algorithms --------------------------------------------------------

    def run_algorithm(self, graph: GraphRef, algorithm: str, **kwargs: Any) -> dict:
        """Run a graph algorithm.

        Supported: pagerank, communities, shortest_path, betweenness,
        closeness, degree, triangle_count, scc, topological_sort,
        label_propagation, leiden, eigenvector, hits, similarity,
        link_prediction, random_walk, k_core, max_flow, mst.
        """
        resp = self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/algorithms/{algorithm}",
            json=kwargs,
        )
        return _check_response(resp)

    # -- CSV import/export -------------------------------------------------

    def import_csv(self, graph: GraphRef, csv_content: str) -> dict:
        """Import nodes from CSV. First row is headers (use _label for node label)."""
        resp = self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/import/csv",
            content=csv_content,
            headers={"Content-Type": "text/csv"},
        )
        return _check_response(resp)

    def export_csv(self, graph: GraphRef) -> str:
        """Export all nodes as CSV."""
        resp = self._client.get(f"/v1/graphs/{_graph_name(graph)}/export/csv")
        resp.raise_for_status()
        return resp.text

    # -- CDC events --------------------------------------------------------

    def subscribe_events(
        self,
        graph: GraphRef | None = None,
        *,
        since_sequence: int | None = None,
        replay_limit: int | None = None,
    ):
        """Subscribe to CDC events via SSE. Returns an iterator of GraphEvent dicts."""
        import json

        path = "/v1/events"
        if graph is not None:
            path = f"/v1/graphs/{_graph_name(graph)}/events"
        params = {
            key: value
            for key, value in {
                "since_sequence": since_sequence,
                "replay_limit": replay_limit,
            }.items()
            if value is not None
        }

        with self._client.stream("GET", path, params=params or None) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    # -- Health ------------------------------------------------------------

    def health(self) -> bool:
        """Check if the server is healthy. Returns True when reachable and OK."""
        try:
            resp = self._client.get("/health")
            data = _check_response(resp)
            return data is not None and data.get("status") == "ok"
        except Exception:
            return False

    def info(self) -> str:
        """Return the server info banner."""
        resp = self._client.get("/v1/info")
        data = _check_response(resp)
        return str(data) if data is not None else ""


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

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6382,
        *,
        timeout: float = 30.0,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        self.base_url = f"http://{host}:{port}"
        headers: dict[str, str] = {}
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        elif username is not None and password is not None:
            import base64
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers["Authorization"] = f"Basic {credentials}"
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, headers=headers)

    async def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    async def __aenter__(self) -> AsyncWeavClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()

    # -- Graph management --------------------------------------------------

    async def create_graph(self, name: GraphRef) -> None:
        """Create a new graph with the given name."""
        if isinstance(name, str):
            body = {"name": name}
        else:
            scope_to_graph(name)
            body = {"scope": _serialize_scope(name)}
        resp = await self._client.post("/v1/graphs", json=body)
        _check_response(resp)

    async def drop_graph(self, name: GraphRef) -> None:
        """Drop (delete) a graph by name."""
        resp = await self._client.delete(f"/v1/graphs/{_graph_name(name)}")
        _check_response(resp)

    async def list_graphs(self) -> list[str]:
        """List the names of all graphs on the server."""
        resp = await self._client.get("/v1/graphs")
        data = _check_response(resp)
        return data if data is not None else []

    async def graph_info(self, name: GraphRef) -> GraphInfo:
        """Get metadata about a graph (node/edge counts)."""
        resp = await self._client.get(f"/v1/graphs/{_graph_name(name)}")
        data = _check_response(resp)
        return _parse_graph_info(data)

    # -- Node operations ---------------------------------------------------

    async def add_node(
        self,
        graph: GraphRef,
        label: str,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        entity_key: str | None = None,
    ) -> int:
        """Add a node to a graph. Returns the new node ID."""
        graph_name = _graph_name(graph)
        body: dict[str, Any] = {"label": label}
        if properties is not None:
            body["properties"] = properties
        if embedding is not None:
            body["embedding"] = embedding
        if entity_key is not None:
            body["entity_key"] = entity_key

        resp = await self._client.post(f"/v1/graphs/{graph_name}/nodes", json=body)
        data = _check_response(resp)
        return data["node_id"]

    async def get_node(self, graph: GraphRef, node_id: int) -> NodeInfo:
        """Get a node by its ID."""
        resp = await self._client.get(f"/v1/graphs/{_graph_name(graph)}/nodes/{node_id}")
        data = _check_response(resp)
        return _parse_node_info(data)

    async def delete_node(self, graph: GraphRef, node_id: int) -> None:
        """Delete a node by its ID (also removes connected edges)."""
        resp = await self._client.delete(f"/v1/graphs/{_graph_name(graph)}/nodes/{node_id}")
        _check_response(resp)

    async def update_node(
        self,
        graph: GraphRef,
        node_id: int,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
    ) -> None:
        """Update a node's properties and/or embedding."""
        graph_name = _graph_name(graph)
        body: dict[str, Any] = {}
        if properties is not None:
            body["properties"] = properties
        if embedding is not None:
            body["embedding"] = embedding
        resp = await self._client.put(f"/v1/graphs/{graph_name}/nodes/{node_id}", json=body)
        _check_response(resp)

    # -- Edge operations ---------------------------------------------------

    async def add_edge(
        self,
        graph: GraphRef,
        source: int,
        target: int,
        label: str,
        weight: float = 1.0,
        provenance: dict | None = None,
    ) -> int:
        """Add an edge between two nodes. Returns the new edge ID."""
        graph_name = _graph_name(graph)
        body: dict[str, Any] = {
            "source": source,
            "target": target,
            "label": label,
            "weight": weight,
        }
        if provenance is not None:
            body["provenance"] = provenance
        resp = await self._client.post(f"/v1/graphs/{graph_name}/edges", json=body)
        data = _check_response(resp)
        return data["edge_id"]

    async def invalidate_edge(self, graph: GraphRef, edge_id: int) -> None:
        """Invalidate (soft-delete) an edge by its ID."""
        resp = await self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/edges/{edge_id}/invalidate"
        )
        _check_response(resp)

    async def bulk_add_nodes(
        self,
        graph: GraphRef,
        nodes: list[dict[str, Any]],
    ) -> list[int]:
        """Bulk add nodes. Each dict needs at least 'label'. Returns list of IDs."""
        resp = await self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/nodes/bulk",
            json={"nodes": nodes},
        )
        data = _check_response(resp)
        return data["node_ids"]

    async def bulk_add_edges(
        self,
        graph: GraphRef,
        edges: list[dict[str, Any]],
    ) -> list[int]:
        """Bulk add edges. Each dict needs 'source', 'target', 'label'. Returns list of IDs."""
        resp = await self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/edges/bulk",
            json={"edges": edges},
        )
        data = _check_response(resp)
        return data["edge_ids"]

    # -- Context retrieval -------------------------------------------------

    async def context(
        self,
        graph: GraphRef,
        query: str | None = None,
        retrieval_mode: RetrievalMode | str | None = None,
        rerank: RerankConfig | dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        seed_nodes: list[str] | None = None,
        budget: int = 4096,
        max_depth: int = 3,
        decay: dict[str, Any] | None = None,
        edge_labels: list[str] | None = None,
        temporal_at: int | None = None,
        include_provenance: bool = False,
        limit: int | None = None,
        sort_field: str | None = None,
        sort_direction: str | None = None,
        direction: str | None = None,
    ) -> ContextResult:
        """Run a context query against a graph.

        At least one of *query*, *embedding*, or *seed_nodes* should be
        provided so the engine can determine seed nodes for traversal.

        *decay* should be a dict like ``{"type": "exponential", "half_life_ms": 3600000}``.
        """
        body: dict[str, Any] = {
            "budget": budget,
            "max_depth": max_depth,
            "include_provenance": include_provenance,
        }
        if isinstance(graph, str):
            body["graph"] = graph
        else:
            scope_to_graph(graph)
            body["scope"] = _serialize_scope(graph)
        if query is not None:
            body["query"] = query
        if retrieval_mode is not None:
            body["retrieval_mode"] = (
                retrieval_mode.value
                if isinstance(retrieval_mode, RetrievalMode)
                else retrieval_mode
            )
        if rerank is not None:
            body["rerank"] = _serialize_rerank(rerank)
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
        if limit is not None:
            body["limit"] = limit
        if sort_field is not None:
            body["sort_field"] = sort_field
        if sort_direction is not None:
            body["sort_direction"] = sort_direction
        if direction is not None:
            body["direction"] = direction

        resp = await self._client.post("/v1/context", json=body)
        data = _check_response(resp)
        return _parse_context_result(data)

    # -- Ingest (extraction pipeline) --------------------------------------

    async def ingest(
        self,
        graph: GraphRef,
        content: str | None = None,
        content_base64: str | None = None,
        format: str | None = None,
        document_id: str | None = None,
        skip_extraction: bool | None = None,
        skip_dedup: bool | None = None,
        chunk_size: int | None = None,
        entity_types: list[str] | None = None,
        resolution_mode: ResolutionMode | str | None = None,
        link_existing_entities: bool | None = None,
        resolution_candidate_limit: int | None = None,
        custom_resolution_prompt: str | None = None,
        params: IngestParams | None = None,
    ) -> IngestResult:
        """Ingest a document into a graph via the extraction pipeline."""
        graph_name = _graph_name(graph)
        body = _build_ingest_body(
            content=content,
            content_base64=content_base64,
            format=format,
            document_id=document_id,
            skip_extraction=skip_extraction,
            skip_dedup=skip_dedup,
            chunk_size=chunk_size,
            entity_types=entity_types,
            resolution_mode=resolution_mode,
            link_existing_entities=link_existing_entities,
            resolution_candidate_limit=resolution_candidate_limit,
            custom_resolution_prompt=custom_resolution_prompt,
            params=params,
        )

        resp = await self._client.post(f"/v1/graphs/{graph_name}/ingest", json=body)
        data = _check_response(resp)
        return _parse_ingest_result(data)

    # -- Search ------------------------------------------------------------

    async def search_text(self, graph: GraphRef, query: str, limit: int = 20) -> dict:
        """Full-text BM25 search across node content."""
        resp = await self._client.get(
            f"/v1/graphs/{_graph_name(graph)}/search/text",
            params={"q": query, "limit": limit},
        )
        return _check_response(resp)

    # -- Node merge --------------------------------------------------------

    async def merge_nodes(
        self,
        graph: GraphRef,
        source_id: int,
        target_id: int,
        conflict_policy: str = "keep_target",
    ) -> dict:
        """Merge two nodes, re-linking edges from source to target."""
        resp = await self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/nodes/merge",
            json={
                "source_id": source_id,
                "target_id": target_id,
                "conflict_policy": conflict_policy,
            },
        )
        return _check_response(resp)

    # -- Algorithms --------------------------------------------------------

    async def search_vector(
        self,
        graph: GraphRef,
        embedding: list[float],
        k: int = 10,
        labels: list[str] | None = None,
        properties: dict | None = None,
    ) -> dict:
        """Vector similarity search with optional filtering."""
        body: dict[str, Any] = {"embedding": embedding, "k": k}
        if labels is not None:
            body["labels"] = labels
        if properties is not None:
            body["properties"] = properties
        resp = await self._client.post(f"/v1/graphs/{_graph_name(graph)}/search/vector", json=body)
        return _check_response(resp)

    # -- Graph diff --------------------------------------------------------

    async def graph_diff(self, graph: GraphRef, from_timestamp: int, to_timestamp: int) -> dict:
        """Compare graph state between two timestamps."""
        resp = await self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/diff",
            json={"from_timestamp": from_timestamp, "to_timestamp": to_timestamp},
        )
        return _check_response(resp)

    # -- Community operations ----------------------------------------------

    async def community_summarize(
        self,
        graph: GraphRef,
        algorithm: str = "leiden",
        resolution: float = 1.0,
    ) -> dict:
        """Run community detection and generate summaries."""
        resp = await self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/communities/summarize",
            json={"algorithm": algorithm, "resolution": resolution},
        )
        return _check_response(resp)

    async def community_summaries(self, graph: GraphRef) -> dict:
        """Get all community summaries."""
        resp = await self._client.get(f"/v1/graphs/{_graph_name(graph)}/communities/summaries")
        return _check_response(resp)

    async def community_search(self, graph: GraphRef, query: str, limit: int = 10) -> dict:
        """Search community summaries."""
        resp = await self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/communities/search",
            json={"query": query, "limit": limit},
        )
        return _check_response(resp)

    # -- Algorithms --------------------------------------------------------

    async def run_algorithm(self, graph: GraphRef, algorithm: str, **kwargs: Any) -> dict:
        """Run a graph algorithm.

        Supported: pagerank, communities, shortest_path, betweenness,
        closeness, degree, triangle_count, scc, topological_sort,
        label_propagation, leiden, eigenvector, hits, similarity,
        link_prediction, random_walk, k_core, max_flow, mst.
        """
        resp = await self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/algorithms/{algorithm}",
            json=kwargs,
        )
        return _check_response(resp)

    # -- CSV import/export -------------------------------------------------

    async def import_csv(self, graph: GraphRef, csv_content: str) -> dict:
        """Import nodes from CSV. First row is headers (use _label for node label)."""
        resp = await self._client.post(
            f"/v1/graphs/{_graph_name(graph)}/import/csv",
            content=csv_content,
            headers={"Content-Type": "text/csv"},
        )
        return _check_response(resp)

    async def export_csv(self, graph: GraphRef) -> str:
        """Export all nodes as CSV."""
        resp = await self._client.get(f"/v1/graphs/{_graph_name(graph)}/export/csv")
        resp.raise_for_status()
        return resp.text

    # -- CDC events --------------------------------------------------------

    async def subscribe_events(
        self,
        graph: GraphRef | None = None,
        *,
        since_sequence: int | None = None,
        replay_limit: int | None = None,
    ):
        """Subscribe to CDC events via SSE. Yields GraphEvent dicts."""
        import json

        path = "/v1/events"
        if graph is not None:
            path = f"/v1/graphs/{_graph_name(graph)}/events"
        params = {
            key: value
            for key, value in {
                "since_sequence": since_sequence,
                "replay_limit": replay_limit,
            }.items()
            if value is not None
        }

        async with self._client.stream("GET", path, params=params or None) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    yield json.loads(line[6:])

    # -- Health ------------------------------------------------------------

    async def health(self) -> bool:
        """Check if the server is healthy. Returns True when reachable and OK."""
        try:
            resp = await self._client.get("/health")
            data = _check_response(resp)
            return data is not None and data.get("status") == "ok"
        except Exception:
            return False

    async def info(self) -> str:
        """Return the server info banner."""
        resp = await self._client.get("/v1/info")
        data = _check_response(resp)
        return str(data) if data is not None else ""
