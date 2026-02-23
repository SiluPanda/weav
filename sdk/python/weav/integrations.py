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
