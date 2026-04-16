"""Data classes and helpers for the Weav Python SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class RetrievalMode(str, Enum):
    """Retrieval mode for context queries."""

    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    DRIFT = "drift"


class ResolutionMode(str, Enum):
    """Entity resolution mode for ingest requests."""

    OFF = "off"
    HEURISTIC = "heuristic"
    SEMANTIC = "semantic"


@dataclass
class ScopeRef:
    """Hierarchical scope resolved to a canonical graph name."""

    workspace_id: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None


GraphRef = str | ScopeRef | dict[str, Any]


def scope_to_graph(scope: ScopeRef | dict[str, Any]) -> str:
    """Resolve a scope object into its canonical graph name."""
    if isinstance(scope, ScopeRef):
        workspace_id = scope.workspace_id
        user_id = scope.user_id
        agent_id = scope.agent_id
        session_id = scope.session_id
    else:
        workspace_id = scope.get("workspace_id")
        user_id = scope.get("user_id")
        agent_id = scope.get("agent_id")
        session_id = scope.get("session_id")

    def _normalize(value: Any, field: str) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip()
        if not normalized:
            raise ValueError(f"scope field '{field}' cannot be empty")
        return normalized

    workspace_id = _normalize(workspace_id, "workspace_id")
    user_id = _normalize(user_id, "user_id")
    agent_id = _normalize(agent_id, "agent_id")
    session_id = _normalize(session_id, "session_id")

    if workspace_id is None:
        raise ValueError("scope requires non-empty 'workspace_id'")
    if agent_id is not None and user_id is None:
        raise ValueError("scope requires 'user_id' before 'agent_id'")
    if session_id is not None and agent_id is None:
        raise ValueError("scope requires 'agent_id' before 'session_id'")

    graph = f"ws:{workspace_id}"
    if user_id is not None:
        graph += f":user:{user_id}"
    if agent_id is not None:
        graph += f":agent:{agent_id}"
    if session_id is not None:
        graph += f":session:{session_id}"
    return graph


@dataclass
class RerankConfig:
    """Optional reranking config for context queries."""

    enabled: bool = True
    provider: Optional[str] = None
    model: Optional[str] = None
    candidate_limit: int = 50
    score_weight: float = 0.35


@dataclass
class Provenance:
    """Provenance metadata tracking the origin and confidence of a fact."""

    source: str
    confidence: float
    extraction_method: str = "UserProvided"
    source_document_id: Optional[str] = None
    source_chunk_offset: Optional[int] = None


@dataclass
class RelationshipSummary:
    """Summary of a relationship (edge) for context output."""

    edge_label: str
    target_node_id: int
    target_name: Optional[str]
    direction: str
    weight: float


@dataclass
class ContextChunk:
    """A single chunk of context extracted from the graph."""

    node_id: int
    content: str
    label: str
    relevance_score: float
    depth: int
    token_count: int
    provenance: Optional[Provenance] = None
    relationships: list[RelationshipSummary] = field(default_factory=list)


@dataclass
class ContextResult:
    """The complete result of a context query."""

    chunks: list[ContextChunk]
    total_tokens: int
    budget_used: float
    nodes_considered: int
    nodes_included: int
    query_time_us: int

    def to_prompt(self) -> str:
        """Format as a text block suitable for LLM prompt insertion."""
        lines: list[str] = []
        for chunk in self.chunks:
            lines.append(f"[{chunk.label}] (score: {chunk.relevance_score:.2f})")
            lines.append(chunk.content)
            if chunk.relationships:
                for rel in chunk.relationships:
                    target = rel.target_name or str(rel.target_node_id)
                    lines.append(f"  -> {rel.edge_label} -> {target}")
            lines.append("")
        return "\n".join(lines)

    def to_messages(self) -> list[dict[str, str]]:
        """Format as OpenAI-compatible message content blocks."""
        return [{"role": "system", "content": self.to_prompt()}]


@dataclass
class GraphInfo:
    """Information about a graph."""

    name: str
    node_count: int
    edge_count: int


@dataclass
class IngestResult:
    """Result of an ingestion pipeline run."""

    document_id: str
    total_tokens: int
    chunks_created: int
    entities_created: int
    entities_merged: int
    entities_resolved: int
    entities_linked_existing: int
    relationships_created: int
    pipeline_duration_ms: int


@dataclass
class IngestParams:
    """Options for document ingestion."""

    content: Optional[str] = None
    content_base64: Optional[str] = None
    format: Optional[str] = None
    document_id: Optional[str] = None
    skip_extraction: bool | None = None
    skip_dedup: bool | None = None
    chunk_size: Optional[int] = None
    entity_types: list[str] | None = None
    resolution_mode: ResolutionMode | str | None = None
    link_existing_entities: bool | None = None
    resolution_candidate_limit: int | None = None
    custom_resolution_prompt: Optional[str] = None


@dataclass
class NodeInfo:
    """Information about a single node."""

    node_id: int
    label: str
    properties: dict[str, Any]
