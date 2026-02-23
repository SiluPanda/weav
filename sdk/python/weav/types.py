"""Data classes matching the Weav server's JSON responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


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
class NodeInfo:
    """Information about a single node."""

    node_id: int
    label: str
    properties: dict[str, Any]
