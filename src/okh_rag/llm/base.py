"""Backend protocols for the two model calls the paper depends on.

§4.1 uses GPT-4o for extraction, generation and judging; §A.4 uses ``text-embedding-3-small`` for the
1536-d hyperedge embeddings. Those are three different *tasks* over two *APIs*, so the boundary here is
drawn at the API level: one protocol for chat, one for embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass
class Usage:
    """Token accounting, so a full run's §4 cost is measurable rather than estimated."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        """Accumulate another usage record in place."""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.embedding_tokens += other.embedding_tokens
        self.calls += other.calls

    def as_dict(self) -> dict[str, int]:
        """Plain-dict view for run manifests."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embedding_tokens": self.embedding_tokens,
            "calls": self.calls,
        }


@runtime_checkable
class LLMClient(Protocol):
    """A chat completion endpoint that can return strict JSON.

    Implementations must be deterministic for a fixed ``(model, messages, temperature, schema)`` tuple
    when a cache is configured; §A.2 deduplicates on "hashes of ... evidence text", which only produces
    a stable hypergraph if the underlying extraction is stable.
    """

    def complete_text(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float,
    ) -> str:
        """Return the assistant message content as text."""
        ...

    def complete_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float,
        schema: Mapping[str, Any],
        schema_name: str,
    ) -> dict[str, Any]:
        """Return the assistant message parsed as JSON conforming to ``schema``.

        §A.2 — "Output is constrained to a strict JSON schema matching the definitions above", so this
        is a schema-constrained call, not a parse-and-hope.
        """
        ...


@runtime_checkable
class EmbeddingClient(Protocol):
    """An embedding endpoint. §A.4 fixes the output dimension at 1536."""

    @property
    def embedding_dim(self) -> int:
        """The dimensionality of the vectors this client returns."""
        ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of strings.

        Args:
            texts: Strings to embed.

        Returns:
            An ``(len(texts), embedding_dim)`` float32 array, row *i* corresponding to ``texts[i]``.
        """
        ...


@runtime_checkable
class ModelBackend(LLMClient, EmbeddingClient, Protocol):
    """Both capabilities plus shared usage accounting, which is what the pipeline actually holds."""

    @property
    def usage(self) -> Usage:
        """Cumulative usage since construction."""
        ...
