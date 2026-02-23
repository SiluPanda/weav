"""Unit tests for the Weav Python SDK.

These tests do NOT require a running Weav server. They test:
- Data class construction and field access
- ContextResult.to_prompt() formatting
- ContextResult.to_messages() formatting
- Integration helper functions
- Client URL construction
- Response parsing logic (the _parse_* helper functions)
- Error handling via WeavError
"""

from __future__ import annotations

import pytest

from weav.client import (
    WeavClient,
    AsyncWeavClient,
    WeavError,
    _check_response,
    _parse_context_chunk,
    _parse_context_result,
    _parse_graph_info,
    _parse_node_info,
    _parse_provenance,
    _parse_relationship_summary,
)
from weav.integrations import context_to_anthropic_messages, context_to_openai_messages
from weav.types import (
    ContextChunk,
    ContextResult,
    GraphInfo,
    NodeInfo,
    Provenance,
    RelationshipSummary,
)


# ---------------------------------------------------------------------------
# Fixtures: sample data
# ---------------------------------------------------------------------------


def _make_provenance() -> Provenance:
    return Provenance(
        source="gpt-4-turbo",
        confidence=0.95,
        extraction_method="LlmExtracted",
        source_document_id="doc-123",
        source_chunk_offset=42,
    )


def _make_relationship(
    label: str = "knows",
    target_id: int = 2,
    target_name: str | None = "Bob",
    direction: str = "outgoing",
    weight: float = 0.8,
) -> RelationshipSummary:
    return RelationshipSummary(
        edge_label=label,
        target_node_id=target_id,
        target_name=target_name,
        direction=direction,
        weight=weight,
    )


def _make_chunk(
    node_id: int = 1,
    content: str = "Alice is a software engineer.",
    label: str = "person",
    score: float = 0.92,
    depth: int = 0,
    token_count: int = 6,
    provenance: Provenance | None = None,
    relationships: list[RelationshipSummary] | None = None,
) -> ContextChunk:
    return ContextChunk(
        node_id=node_id,
        content=content,
        label=label,
        relevance_score=score,
        depth=depth,
        token_count=token_count,
        provenance=provenance,
        relationships=relationships or [],
    )


def _make_context_result(
    chunks: list[ContextChunk] | None = None,
) -> ContextResult:
    if chunks is None:
        chunks = [
            _make_chunk(
                node_id=1,
                content="Alice is a software engineer.",
                label="person",
                score=0.92,
                relationships=[_make_relationship()],
            ),
            _make_chunk(
                node_id=2,
                content="Bob works at Acme Corp.",
                label="person",
                score=0.75,
            ),
        ]
    total_tokens = sum(c.token_count for c in chunks)
    return ContextResult(
        chunks=chunks,
        total_tokens=total_tokens,
        budget_used=0.42,
        nodes_considered=10,
        nodes_included=len(chunks),
        query_time_us=1234,
    )


# ---------------------------------------------------------------------------
# Type construction and field access
# ---------------------------------------------------------------------------


class TestTypeConstruction:
    """Test that data classes can be constructed and fields accessed."""

    def test_provenance_fields(self) -> None:
        p = _make_provenance()
        assert p.source == "gpt-4-turbo"
        assert p.confidence == 0.95
        assert p.extraction_method == "LlmExtracted"
        assert p.source_document_id == "doc-123"
        assert p.source_chunk_offset == 42

    def test_provenance_defaults(self) -> None:
        p = Provenance(source="user", confidence=1.0)
        assert p.extraction_method == "UserProvided"
        assert p.source_document_id is None
        assert p.source_chunk_offset is None

    def test_relationship_summary_fields(self) -> None:
        r = _make_relationship()
        assert r.edge_label == "knows"
        assert r.target_node_id == 2
        assert r.target_name == "Bob"
        assert r.direction == "outgoing"
        assert r.weight == 0.8

    def test_relationship_summary_no_name(self) -> None:
        r = _make_relationship(target_name=None)
        assert r.target_name is None

    def test_context_chunk_fields(self) -> None:
        c = _make_chunk()
        assert c.node_id == 1
        assert c.content == "Alice is a software engineer."
        assert c.label == "person"
        assert c.relevance_score == 0.92
        assert c.depth == 0
        assert c.token_count == 6
        assert c.provenance is None
        assert c.relationships == []

    def test_context_chunk_with_provenance(self) -> None:
        p = _make_provenance()
        c = _make_chunk(provenance=p)
        assert c.provenance is p

    def test_context_chunk_with_relationships(self) -> None:
        rels = [_make_relationship(), _make_relationship(label="works_with", target_id=3)]
        c = _make_chunk(relationships=rels)
        assert len(c.relationships) == 2

    def test_context_result_fields(self) -> None:
        result = _make_context_result()
        assert len(result.chunks) == 2
        assert result.total_tokens == 12
        assert result.budget_used == 0.42
        assert result.nodes_considered == 10
        assert result.nodes_included == 2
        assert result.query_time_us == 1234

    def test_graph_info_fields(self) -> None:
        g = GraphInfo(name="test_graph", node_count=100, edge_count=200)
        assert g.name == "test_graph"
        assert g.node_count == 100
        assert g.edge_count == 200

    def test_node_info_fields(self) -> None:
        n = NodeInfo(node_id=42, label="person", properties={"name": "Alice", "age": 30})
        assert n.node_id == 42
        assert n.label == "person"
        assert n.properties["name"] == "Alice"
        assert n.properties["age"] == 30


# ---------------------------------------------------------------------------
# ContextResult.to_prompt()
# ---------------------------------------------------------------------------


class TestToPrompt:
    """Test the to_prompt() formatting method."""

    def test_basic_format(self) -> None:
        result = _make_context_result()
        prompt = result.to_prompt()

        # Should contain labels with scores.
        assert "[person] (score: 0.92)" in prompt
        assert "[person] (score: 0.75)" in prompt

        # Should contain content.
        assert "Alice is a software engineer." in prompt
        assert "Bob works at Acme Corp." in prompt

    def test_relationships_in_prompt(self) -> None:
        result = _make_context_result()
        prompt = result.to_prompt()

        # The first chunk has a relationship to "Bob" via "knows".
        assert "  -> knows -> Bob" in prompt

    def test_relationship_fallback_to_id(self) -> None:
        """When target_name is None, the node ID should appear instead."""
        rel = _make_relationship(target_name=None, target_id=99)
        chunk = _make_chunk(relationships=[rel])
        result = _make_context_result(chunks=[chunk])
        prompt = result.to_prompt()
        assert "  -> knows -> 99" in prompt

    def test_empty_chunks(self) -> None:
        result = _make_context_result(chunks=[])
        prompt = result.to_prompt()
        assert prompt == ""

    def test_no_relationships(self) -> None:
        chunk = _make_chunk(relationships=[])
        result = _make_context_result(chunks=[chunk])
        prompt = result.to_prompt()
        # Should not contain relationship arrows.
        assert "  -> " not in prompt

    def test_multiple_relationships(self) -> None:
        rels = [
            _make_relationship(label="knows", target_name="Bob"),
            _make_relationship(label="works_at", target_name="Acme"),
        ]
        chunk = _make_chunk(relationships=rels)
        result = _make_context_result(chunks=[chunk])
        prompt = result.to_prompt()
        assert "  -> knows -> Bob" in prompt
        assert "  -> works_at -> Acme" in prompt


# ---------------------------------------------------------------------------
# ContextResult.to_messages()
# ---------------------------------------------------------------------------


class TestToMessages:
    """Test the to_messages() formatting method."""

    def test_returns_single_system_message(self) -> None:
        result = _make_context_result()
        messages = result.to_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert isinstance(messages[0]["content"], str)

    def test_content_matches_to_prompt(self) -> None:
        result = _make_context_result()
        messages = result.to_messages()
        assert messages[0]["content"] == result.to_prompt()


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


class TestIntegrations:
    """Test the LLM integration helper functions."""

    def test_context_to_anthropic_messages(self) -> None:
        result = _make_context_result()
        messages = context_to_anthropic_messages(result)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert isinstance(messages[0]["content"], list)
        assert len(messages[0]["content"]) == 1
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][0]["text"] == result.to_prompt()

    def test_context_to_openai_messages(self) -> None:
        result = _make_context_result()
        messages = context_to_openai_messages(result)
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == result.to_prompt()

    def test_anthropic_messages_empty_chunks(self) -> None:
        result = _make_context_result(chunks=[])
        messages = context_to_anthropic_messages(result)
        assert messages[0]["content"][0]["text"] == ""

    def test_openai_messages_empty_chunks(self) -> None:
        result = _make_context_result(chunks=[])
        messages = context_to_openai_messages(result)
        assert messages[0]["content"] == ""


# ---------------------------------------------------------------------------
# Client URL construction
# ---------------------------------------------------------------------------


class TestClientURLConstruction:
    """Test that clients build the correct base URL."""

    def test_sync_client_default_url(self) -> None:
        client = WeavClient()
        assert client.base_url == "http://localhost:6382"
        client.close()

    def test_sync_client_custom_url(self) -> None:
        client = WeavClient(host="10.0.0.1", port=9999)
        assert client.base_url == "http://10.0.0.1:9999"
        client.close()

    def test_async_client_default_url(self) -> None:
        client = AsyncWeavClient()
        assert client.base_url == "http://localhost:6382"
        # We cannot await .close() in a sync test, but the base_url is set.

    def test_async_client_custom_url(self) -> None:
        client = AsyncWeavClient(host="weav.example.com", port=8080)
        assert client.base_url == "http://weav.example.com:8080"


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------


class TestResponseParsing:
    """Test the _parse_* helper functions that convert JSON dicts to dataclasses."""

    def test_parse_provenance(self) -> None:
        data = {
            "source": "gpt-4-turbo",
            "confidence": 0.95,
            "extraction_method": "LlmExtracted",
            "source_document_id": "doc-123",
            "source_chunk_offset": 42,
        }
        p = _parse_provenance(data)
        assert isinstance(p, Provenance)
        assert p.source == "gpt-4-turbo"
        assert p.confidence == 0.95
        assert p.extraction_method == "LlmExtracted"
        assert p.source_document_id == "doc-123"
        assert p.source_chunk_offset == 42

    def test_parse_provenance_defaults(self) -> None:
        data = {"source": "user", "confidence": 1.0}
        p = _parse_provenance(data)
        assert p.extraction_method == "UserProvided"
        assert p.source_document_id is None
        assert p.source_chunk_offset is None

    def test_parse_relationship_summary(self) -> None:
        data = {
            "edge_label": "knows",
            "target_node_id": 5,
            "target_name": "Charlie",
            "direction": "outgoing",
            "weight": 0.9,
        }
        r = _parse_relationship_summary(data)
        assert isinstance(r, RelationshipSummary)
        assert r.edge_label == "knows"
        assert r.target_node_id == 5
        assert r.target_name == "Charlie"
        assert r.direction == "outgoing"
        assert r.weight == 0.9

    def test_parse_relationship_summary_null_name(self) -> None:
        data = {
            "edge_label": "uses",
            "target_node_id": 7,
            "target_name": None,
            "direction": "incoming",
            "weight": 1.0,
        }
        r = _parse_relationship_summary(data)
        assert r.target_name is None

    def test_parse_context_chunk_minimal(self) -> None:
        data = {
            "node_id": 1,
            "content": "Hello world",
            "label": "doc",
            "relevance_score": 0.5,
            "depth": 1,
            "token_count": 2,
            "provenance": None,
            "relationships": [],
        }
        c = _parse_context_chunk(data)
        assert isinstance(c, ContextChunk)
        assert c.node_id == 1
        assert c.content == "Hello world"
        assert c.provenance is None
        assert c.relationships == []

    def test_parse_context_chunk_full(self) -> None:
        data = {
            "node_id": 10,
            "content": "Rust is a systems language.",
            "label": "topic",
            "relevance_score": 0.88,
            "depth": 2,
            "token_count": 5,
            "provenance": {
                "source": "nlp",
                "confidence": 0.7,
                "extraction_method": "NlpPipeline",
            },
            "relationships": [
                {
                    "edge_label": "related_to",
                    "target_node_id": 11,
                    "target_name": "C++",
                    "direction": "outgoing",
                    "weight": 0.6,
                },
            ],
        }
        c = _parse_context_chunk(data)
        assert c.provenance is not None
        assert c.provenance.source == "nlp"
        assert len(c.relationships) == 1
        assert c.relationships[0].edge_label == "related_to"

    def test_parse_context_chunk_missing_optional_fields(self) -> None:
        """When provenance and relationships are absent from JSON, they default."""
        data = {
            "node_id": 3,
            "content": "Test",
            "label": "test",
            "relevance_score": 0.1,
            "depth": 0,
            "token_count": 1,
        }
        c = _parse_context_chunk(data)
        assert c.provenance is None
        assert c.relationships == []

    def test_parse_context_result(self) -> None:
        data = {
            "chunks": [
                {
                    "node_id": 1,
                    "content": "Alice",
                    "label": "person",
                    "relevance_score": 0.9,
                    "depth": 0,
                    "token_count": 1,
                    "provenance": None,
                    "relationships": [],
                },
            ],
            "total_tokens": 1,
            "budget_used": 0.01,
            "nodes_considered": 5,
            "nodes_included": 1,
            "query_time_us": 500,
        }
        result = _parse_context_result(data)
        assert isinstance(result, ContextResult)
        assert len(result.chunks) == 1
        assert result.chunks[0].content == "Alice"
        assert result.total_tokens == 1
        assert result.budget_used == 0.01
        assert result.nodes_considered == 5
        assert result.nodes_included == 1
        assert result.query_time_us == 500

    def test_parse_context_result_empty_chunks(self) -> None:
        data = {
            "chunks": [],
            "total_tokens": 0,
            "budget_used": 0.0,
            "nodes_considered": 0,
            "nodes_included": 0,
            "query_time_us": 100,
        }
        result = _parse_context_result(data)
        assert result.chunks == []
        assert result.total_tokens == 0

    def test_parse_graph_info(self) -> None:
        data = {"name": "my_graph", "node_count": 42, "edge_count": 100}
        g = _parse_graph_info(data)
        assert isinstance(g, GraphInfo)
        assert g.name == "my_graph"
        assert g.node_count == 42
        assert g.edge_count == 100

    def test_parse_node_info(self) -> None:
        data = {
            "node_id": 7,
            "label": "company",
            "properties": {"name": "Acme", "founded": 1990},
        }
        n = _parse_node_info(data)
        assert isinstance(n, NodeInfo)
        assert n.node_id == 7
        assert n.label == "company"
        assert n.properties["name"] == "Acme"
        assert n.properties["founded"] == 1990

    def test_parse_node_info_empty_properties(self) -> None:
        data = {"node_id": 1, "label": "empty", "properties": {}}
        n = _parse_node_info(data)
        assert n.properties == {}

    def test_parse_node_info_missing_properties_key(self) -> None:
        """When properties key is absent, default to empty dict."""
        data = {"node_id": 1, "label": "bare"}
        n = _parse_node_info(data)
        assert n.properties == {}


# ---------------------------------------------------------------------------
# _check_response and WeavError
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal fake httpx.Response for testing _check_response."""

    def __init__(self, json_data: dict | None, status_code: int = 200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self) -> dict:
        if self._json_data is None:
            raise ValueError("No JSON body")
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class TestCheckResponse:
    """Test the _check_response helper and WeavError."""

    def test_success_with_data(self) -> None:
        resp = _FakeResponse({"success": True, "data": {"id": 42}})
        data = _check_response(resp)  # type: ignore[arg-type]
        assert data == {"id": 42}

    def test_success_empty(self) -> None:
        resp = _FakeResponse({"success": True})
        data = _check_response(resp)  # type: ignore[arg-type]
        assert data is None

    def test_error_raises_weav_error(self) -> None:
        resp = _FakeResponse(
            {"success": False, "error": "graph not found"}, status_code=404
        )
        with pytest.raises(WeavError) as exc_info:
            _check_response(resp)  # type: ignore[arg-type]
        assert "graph not found" in str(exc_info.value)
        assert exc_info.value.status_code == 404

    def test_error_unknown_message(self) -> None:
        resp = _FakeResponse({"success": False}, status_code=500)
        with pytest.raises(WeavError) as exc_info:
            _check_response(resp)  # type: ignore[arg-type]
        assert "Unknown server error" in str(exc_info.value)

    def test_weav_error_attributes(self) -> None:
        err = WeavError("something failed", status_code=409)
        assert str(err) == "something failed"
        assert err.status_code == 409

    def test_weav_error_no_status(self) -> None:
        err = WeavError("no status")
        assert err.status_code is None


# ---------------------------------------------------------------------------
# Context manager protocol
# ---------------------------------------------------------------------------


class TestContextManager:
    """Test that the sync client supports the context manager protocol."""

    def test_sync_context_manager(self) -> None:
        with WeavClient() as client:
            assert client.base_url == "http://localhost:6382"
        # After exiting, the underlying transport is closed.
        # We just verify no exception is raised.


# ---------------------------------------------------------------------------
# Roundtrip: parse then format
# ---------------------------------------------------------------------------


class TestRoundtrip:
    """Test parsing JSON into types then formatting back to prompt/messages."""

    def test_parse_then_to_prompt(self) -> None:
        """Simulate a full round-trip from server JSON to LLM prompt."""
        server_json = {
            "chunks": [
                {
                    "node_id": 1,
                    "content": "Alice is a developer who loves Rust.",
                    "label": "person",
                    "relevance_score": 0.95,
                    "depth": 0,
                    "token_count": 8,
                    "provenance": None,
                    "relationships": [
                        {
                            "edge_label": "uses",
                            "target_node_id": 2,
                            "target_name": "Rust",
                            "direction": "outgoing",
                            "weight": 1.0,
                        }
                    ],
                },
                {
                    "node_id": 2,
                    "content": "Rust is a systems programming language.",
                    "label": "topic",
                    "relevance_score": 0.78,
                    "depth": 1,
                    "token_count": 6,
                    "provenance": None,
                    "relationships": [],
                },
            ],
            "total_tokens": 14,
            "budget_used": 0.35,
            "nodes_considered": 20,
            "nodes_included": 2,
            "query_time_us": 2500,
        }
        result = _parse_context_result(server_json)

        prompt = result.to_prompt()
        assert "[person] (score: 0.95)" in prompt
        assert "Alice is a developer who loves Rust." in prompt
        assert "  -> uses -> Rust" in prompt
        assert "[topic] (score: 0.78)" in prompt
        assert "Rust is a systems programming language." in prompt

    def test_parse_then_to_openai_messages(self) -> None:
        server_json = {
            "chunks": [
                {
                    "node_id": 1,
                    "content": "Test node",
                    "label": "test",
                    "relevance_score": 1.0,
                    "depth": 0,
                    "token_count": 2,
                    "provenance": None,
                    "relationships": [],
                }
            ],
            "total_tokens": 2,
            "budget_used": 0.1,
            "nodes_considered": 1,
            "nodes_included": 1,
            "query_time_us": 100,
        }
        result = _parse_context_result(server_json)
        messages = context_to_openai_messages(result)
        assert messages[0]["role"] == "system"
        assert "Test node" in messages[0]["content"]

    def test_parse_then_to_anthropic_messages(self) -> None:
        server_json = {
            "chunks": [],
            "total_tokens": 0,
            "budget_used": 0.0,
            "nodes_considered": 0,
            "nodes_included": 0,
            "query_time_us": 50,
        }
        result = _parse_context_result(server_json)
        messages = context_to_anthropic_messages(result)
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["text"] == ""
