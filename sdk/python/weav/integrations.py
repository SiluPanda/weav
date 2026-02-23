"""LLM integration helpers for converting Weav context results to API formats."""

from __future__ import annotations

from .types import ContextResult


def context_to_anthropic_messages(result: ContextResult) -> list[dict]:
    """Convert a ContextResult to Anthropic Messages API format.

    Returns a list with a single user message containing the context as text.
    Suitable for injecting into the ``messages`` parameter of
    ``anthropic.messages.create()``.
    """
    return [
        {
            "role": "user",
            "content": [{"type": "text", "text": result.to_prompt()}],
        }
    ]


def context_to_openai_messages(result: ContextResult) -> list[dict]:
    """Convert a ContextResult to OpenAI Chat Completions API format.

    Returns a list with a single system message containing the context.
    Suitable for prepending to the ``messages`` parameter of
    ``openai.chat.completions.create()``.
    """
    return [{"role": "system", "content": result.to_prompt()}]


class WeavLangChain:
    """LangChain-compatible retriever that uses Weav for context."""

    def __init__(self, client, graph: str, **default_context_kwargs):
        self.client = client
        self.graph = graph
        self.default_kwargs = default_context_kwargs

    def get_relevant_documents(self, query: str) -> list[dict]:
        """Retrieve relevant documents from Weav graph."""
        result = self.client.context(graph=self.graph, query=query, **self.default_kwargs)
        return [
            {
                "page_content": chunk.content,
                "metadata": {
                    "node_id": chunk.node_id,
                    "label": chunk.label,
                    "score": chunk.relevance_score,
                },
            }
            for chunk in result.chunks
        ]

    def as_retriever(self):
        """Return self as a retriever interface."""
        return self


class WeavLlamaIndex:
    """LlamaIndex-compatible retriever that uses Weav for context."""

    def __init__(self, client, graph: str, **default_context_kwargs):
        self.client = client
        self.graph = graph
        self.default_kwargs = default_context_kwargs

    def retrieve(self, query: str) -> list[dict]:
        """Retrieve nodes from Weav graph."""
        result = self.client.context(graph=self.graph, query=query, **self.default_kwargs)
        return [
            {
                "text": chunk.content,
                "score": chunk.relevance_score,
                "metadata": {
                    "node_id": chunk.node_id,
                    "label": chunk.label,
                },
            }
            for chunk in result.chunks
        ]
