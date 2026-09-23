"""OpenAI backend.

§4.1 names GPT-4o as the generator and ``text-embedding-3-small`` as the embedding model, and §A.2 uses
GPT-4o for extraction too. This module is the only place that talks to the API.

Configuration follows the SDK's own conventions: ``OPENAI_API_KEY`` and (optionally) ``OPENAI_BASE_URL``
are read from the environment by the ``openai`` package itself, so no credential ever passes through
``configs/base.yaml``.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np

from okh_rag.config import LLMConfig
from okh_rag.llm.base import Usage
from okh_rag.llm.cache import DiskCache, cache_key, messages_key

# §A.4 — "Hyperedges are embedded as he ∈ R^1536". text-embedding-3-small is natively 1536-d, so the
# dimension is asserted rather than discovered: a mismatch means the wrong model was configured, and
# failing at call time is far cheaper than failing after a full training run.
_EXPECTED_EMBEDDING_DIM = 1536


class StrictSchemaError(ValueError):
    """Raised when a JSON schema cannot be used in OpenAI's ``strict`` structured-output mode."""


def _assert_strict_schema(schema: Mapping[str, Any], path: str = "$") -> None:
    """Validate that ``schema`` satisfies OpenAI's strict-mode requirements.

    The API rejects schemas where an object type has ``additionalProperties`` other than ``false``, or
    where ``properties`` and ``required`` disagree. Checking locally turns an opaque 400 at request time
    into a clear error at definition time.
    """
    node_type = schema.get("type")
    if node_type == "object":
        if schema.get("additionalProperties") is not False:
            raise StrictSchemaError(f"{path}: object schemas need additionalProperties: false")
        props = schema.get("properties", {})
        required = set(schema.get("required", []))
        missing = set(props) - required
        if missing:
            raise StrictSchemaError(f"{path}: strict mode requires every property in 'required'; missing {sorted(missing)}")
        for name, sub in props.items():
            _assert_strict_schema(sub, f"{path}.{name}")
    elif node_type == "array":
        items = schema.get("items")
        if not isinstance(items, Mapping):
            raise StrictSchemaError(f"{path}: array schemas need an 'items' schema")
        _assert_strict_schema(items, f"{path}[]")


class OpenAIBackend:
    """Concrete :class:`~okh_rag.llm.base.ModelBackend` over the OpenAI SDK.

    Args:
        cfg: The ``llm`` section of ``configs/base.yaml`` — model names, temperatures, retry policy.
        client: An existing ``openai.OpenAI`` instance. When omitted one is constructed with no
            arguments, which makes the SDK read ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` from the
            environment. Injecting a client is what lets the test suite run without network access.
        cache: Optional response cache.
    """

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        client: Any | None = None,
        cache: DiskCache | None = None,
    ) -> None:
        self.cfg = cfg
        self._usage = Usage()
        self.cache = cache

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
                raise ImportError(
                    "the openai package is required for the OpenAI backend; pip install openai"
                ) from exc
            client = OpenAI()
        self._client = client

    # -- usage -------------------------------------------------------------

    @property
    def usage(self) -> Usage:
        """Cumulative token usage since construction."""
        return self._usage

    # -- chat --------------------------------------------------------------

    def complete_text(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float,
    ) -> str:
        """Return the assistant message content as text."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages_key(messages),
            "temperature": temperature,
        }
        key = cache_key(kind="chat", model=model, payload=payload, temperature=temperature)

        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return str(hit)

        response = self._call_with_retry(lambda: self._client.chat.completions.create(**payload))
        content = response.choices[0].message.content or ""
        self._record_chat_usage(response)

        if self.cache is not None:
            self.cache.set(key, content)
        return content

    def complete_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        temperature: float,
        schema: Mapping[str, Any],
        schema_name: str,
    ) -> dict[str, Any]:
        """Schema-constrained chat call, returning parsed JSON.

        §A.2 — "Output is constrained to a strict JSON schema matching the definitions above."

        Raises:
            StrictSchemaError: If ``schema`` is not usable in strict mode.
            ValueError: If the API returns content that is not a JSON object.
        """
        _assert_strict_schema(schema)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages_key(messages),
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": dict(schema)},
            },
        }
        key = cache_key(
            kind="chat_json",
            model=model,
            payload=payload,
            temperature=temperature,
        )

        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return dict(hit)

        response = self._call_with_retry(lambda: self._client.chat.completions.create(**payload))
        content = response.choices[0].message.content or ""
        self._record_chat_usage(response)

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"model returned non-JSON content for schema {schema_name!r}: {content[:400]}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"model returned {type(parsed).__name__}, expected a JSON object")

        if self.cache is not None:
            self.cache.set(key, parsed)
        return parsed

    # -- embeddings --------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """§A.4 fixes the hyperedge embedding at 1536 dimensions."""
        return _EXPECTED_EMBEDDING_DIM

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts``, batching by ``cfg.embedding_batch_size``.

        Args:
            texts: Strings to embed. An empty sequence yields a ``(0, 1536)`` array rather than an error,
                because a horizon block can legitimately produce no facts.

        Returns:
            An ``(len(texts), 1536)`` float32 array in input order.
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        out = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        batch = max(1, self.cfg.embedding_batch_size)

        for start in range(0, len(texts), batch):
            chunk = list(texts[start : start + batch])
            payload: dict[str, Any] = {"model": self.cfg.embedding_model, "input": chunk}

            # Embeddings take no temperature; 0.0 is passed to keep the cache key shape uniform.
            key = cache_key(kind="embed", model=self.cfg.embedding_model, payload=payload, temperature=0.0)

            cached = self.cache.get(key) if self.cache is not None else None
            if cached is not None:
                out[start : start + len(chunk)] = np.asarray(cached, dtype=np.float32)
                continue

            response = self._call_with_retry(lambda: self._client.embeddings.create(**payload))
            vectors = [item.embedding for item in response.data]
            if len(vectors) != len(chunk):
                raise ValueError(f"embedding API returned {len(vectors)} vectors for {len(chunk)} inputs")
            for vec in vectors:
                if len(vec) != self.embedding_dim:
                    raise ValueError(
                        f"embedding model returned {len(vec)} dimensions, §A.4 fixes 1536; "
                        f"is llm.embedding_model really text-embedding-3-small?"
                    )

            out[start : start + len(chunk)] = np.asarray(vectors, dtype=np.float32)
            self._record_embedding_usage(response)

            if self.cache is not None:
                self.cache.set(key, vectors)

        return out

    # -- internals ---------------------------------------------------------

    def _call_with_retry(self, fn: Any) -> Any:
        """Invoke ``fn`` with exponential backoff on transient failures.

        Retries rate limits, connection errors, timeouts and 5xx responses. Authentication and
        bad-request errors are raised immediately: retrying them only burns wall-clock time.
        """
        import openai

        retryable = (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        )
        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                return fn()
            except retryable as exc:
                last_exc = exc
                if attempt == self.cfg.max_retries - 1:
                    break
                delay = self.cfg.retry_base_delay_s * (2**attempt)
                # Jitter avoids a fleet of workers retrying in lockstep after a shared rate limit.
                time.sleep(delay * (0.5 + random.random()))
        assert last_exc is not None
        raise last_exc

    def _record_chat_usage(self, response: Any) -> None:
        self._usage.calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._usage.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self._usage.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    def _record_embedding_usage(self, response: Any) -> None:
        self._usage.calls += 1
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._usage.embedding_tokens += getattr(usage, "prompt_tokens", 0) or 0
