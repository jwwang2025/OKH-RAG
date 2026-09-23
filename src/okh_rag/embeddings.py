"""§A.4 — hyperedge embeddings, and the §A.6 relevance index built on them.

§A.4 specifies three things about the embedding step and nothing more:

    "Each hyperedge is embedded as h_e ∈ R^1536 via OpenAI text-embedding-3-small, which concatenates
     the relation type, evidence string s_e, entity names and types, and key attribute values."

So the embedded text has four specified parts in a specified order, and the model and dimension are
fixed. The *formatting* joining those parts is not stated, and neither is which attributes count as
"key".

This module owns three responsibilities:

* :func:`hyperedge_embedding_text` — assembling the four parts into one string.
* :class:`EmbeddingStore` — a content-addressed on-disk cache. A §4-scale run embeds on the order of
  10^5 hyperedges, and §A.4's text is derived from the extraction output: without a cache, every prompt
  tweak re-bills the whole corpus. Embedding vectors are the only expensive artifact here, so they get a
  dedicated sharded store rather than the JSON cache in :mod:`okh_rag.llm.cache` — 1536 floats as JSON
  is roughly 30 bytes per float, which is 60x the size of the float32 it represents.
* :class:`HyperedgeEmbeddings` — the (N, 1536) matrix plus the id-to-row index that §A.6's cosine
  retrieval and §3.1's transition model both index into.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from okh_rag.config import Config
from okh_rag.llm.base import EmbeddingClient
from okh_rag.schema import Hyperedge, KnowledgeHypergraph

__all__ = [
    "EmbeddingStore",
    "HyperedgeEmbeddings",
    "embed_hypergraph",
    "embed_texts",
    "hyperedge_embedding_text",
]

# §A.4 — "h_e ∈ R^1536". Also the dim the §A.4 parameter count is derived from: 2·64·1536.
EMBEDDING_DIM = 1536


# ---------------------------------------------------------------------------
# §A.4 — the text that gets embedded
# ---------------------------------------------------------------------------


def hyperedge_embedding_text(edge: Hyperedge, hypergraph: KnowledgeHypergraph) -> str:
    """Assemble §A.4's embedding input: relation, then ``s_e``, then entities, then attributes.

    §A.4 — "concatenates the relation type, evidence string s_e, entity names and types, and key
    attribute values". The order is taken from that sentence.

    # [UNSPECIFIED] The paper does not give the joining format, so the four parts are emitted as
    # newline-separated, label-prefixed lines. Alternatives: a bare "relation. evidence. names types.
    # attrs" concatenation (closer to the literal word "concatenates", but loses the field boundaries),
    # or a JSON object (the embedding model would see a lot of punctuation).
    # [UNSPECIFIED] "key attribute values" is not defined. All of ``a_e`` is included, sorted by key so
    # the same hyperedge always embeds to the same vector. Alternatives: a curated key list; dropping
    # attributes entirely.

    Determinism matters here for the same reason it matters in §A.2 step 5: the id is a hash of the
    evidence, so a non-deterministic embedding input would make two runs on identical inputs disagree.
    Entities therefore appear in ``V_e``'s stored order and attributes in key order.

    The temporal anchor (§A.2 step 3) is part of ``V_e``, so it is included like any other entity. That
    is what puts the horizon into the embedding, and §A.6's relevance term is cosine similarity over this
    same vector — so an anchor-bearing embedding is what lets a horizon-specific query prefer the right
    horizon's hyperedges.
    """
    entities = "; ".join(
        f"{entity.name} ({entity.type})" for entity in hypergraph.entity_set(edge)
    )
    attributes = "; ".join(f"{key}={edge.attributes[key]}" for key in sorted(edge.attributes))

    parts = [
        f"relation: {edge.relation}",
        f"evidence: {edge.evidence}",
        f"entities: {entities}",
        f"attributes: {attributes}",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Content-addressed embedding cache
# ---------------------------------------------------------------------------


def _text_key(text: str) -> str:
    """``sha256`` of the text, which is what makes the store independent of hyperedge identity.

    Keying on the *text* rather than the hyperedge id is deliberate: §A.2 step 2 rewrites entity ids and
    step 5 rewrites hyperedge ids, so id-keyed caching would miss on every rebuild even when the
    embedded string is byte-identical.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingStore:
    """Sharded, content-addressed store of embedding vectors.

    Vectors live in ``root/{key[:2]}.npz``, one file per two-hex-character shard — 256 files, each
    holding up to ``1/256`` of the corpus. A single-file store would have to be loaded whole (several GB
    at §4 scale); one file per vector would mean hundreds of thousands of tiny reads. Sharding splits the
    difference: bounded memory per shard, and ~256 sequential reads per flush.

    Shards are held in memory once loaded and written back only when dirty, so a run that reuses its
    cache pays one read and no write.
    """

    def __init__(self, root: str | Path, *, dim: int = EMBEDDING_DIM, enabled: bool = True) -> None:
        self.root = Path(root)
        self.dim = dim
        self.enabled = enabled
        self._shards: dict[str, dict[str, np.ndarray]] = {}
        self._dirty: set[str] = set()
        self.hits = 0
        self.misses = 0
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    # -- storage -----------------------------------------------------------

    def _shard_path(self, key: str) -> Path:
        return self.root / f"{key[:2]}.npz"

    def _shard(self, key: str) -> dict[str, np.ndarray]:
        shard_id = key[:2]
        cached = self._shards.get(shard_id)
        if cached is not None:
            return cached
        path = self._shard_path(key)
        loaded: dict[str, np.ndarray] = {}
        if self.enabled and path.exists():
            try:
                with np.load(path) as data:
                    loaded = {name: data[name].astype(np.float32, copy=False) for name in data.files}
            except (OSError, ValueError):
                # A truncated shard is treated as absent; the vectors it held are simply re-embedded.
                loaded = {}
        self._shards[shard_id] = loaded
        return loaded

    def get(self, text: str) -> np.ndarray | None:
        """Return the cached vector for ``text``, or ``None`` on a miss."""
        if not self.enabled:
            return None
        key = _text_key(text)
        vector = self._shard(key).get(key)
        if vector is None:
            self.misses += 1
            return None
        self.hits += 1
        return vector

    def put(self, text: str, vector: np.ndarray) -> None:
        """Store ``vector`` under ``text``. The vector is copied to float32 and detached from any array it views."""
        if not self.enabled:
            return
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        if array.shape != (self.dim,):
            raise ValueError(f"expected a {self.dim}-d vector, got shape {array.shape}")
        key = _text_key(text)
        self._shard(key)[key] = array
        self._dirty.add(key[:2])

    def flush(self) -> None:
        """Write dirty shards to disk. Called once per document rather than per vector."""
        if not self.enabled:
            return
        for shard_id in sorted(self._dirty):
            shard = self._shards.get(shard_id, {})
            path = self.root / f"{shard_id}.npz"
            tmp = path.with_name(path.name + ".tmp")
            with tmp.open("wb") as handle:
                np.savez(handle, **shard)
            tmp.replace(path)
        self._dirty.clear()

    def stats(self) -> dict[str, int]:
        """Entry count and on-disk size, for the run manifest."""
        if not self.enabled or not self.root.exists():
            return {"entries": 0, "bytes": 0, "hits": self.hits, "misses": self.misses}
        entries = 0
        total = 0
        for path in self.root.glob("*.npz"):
            with np.load(path) as data:
                entries += len(data.files)
            total += path.stat().st_size
        return {"entries": entries, "bytes": total, "hits": self.hits, "misses": self.misses}


# ---------------------------------------------------------------------------
# The embedding matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HyperedgeEmbeddings:
    """``h_e`` for every hyperedge, as an (N, d) matrix plus the id-to-row index.

    Attributes:
        hyperedge_ids: Row *i* of ``matrix`` is the embedding of ``hyperedge_ids[i]``. Sorted, so the
            row order is reproducible across runs.
        matrix: An ``(N, dim)`` float32 array.
    """

    hyperedge_ids: tuple[str, ...]
    matrix: np.ndarray

    def __post_init__(self) -> None:
        if self.matrix.shape != (len(self.hyperedge_ids), self.matrix.shape[1]):
            raise ValueError(
                f"matrix has {self.matrix.shape[0]} rows for {len(self.hyperedge_ids)} hyperedge ids"
            )
        if self.matrix.dtype != np.float32:
            raise ValueError(f"matrix must be float32, got {self.matrix.dtype}")

    def __len__(self) -> int:
        return len(self.hyperedge_ids)

    @property
    def dim(self) -> int:
        """``d`` — §A.4 fixes it at 1536, and §3.1's ``W = U^T V`` has shape ``(d, d)``."""
        return int(self.matrix.shape[1])

    @property
    def index(self) -> dict[str, int]:
        """``hyperedge id -> row number``. Rebuilt per access; small enough not to be worth caching."""
        return {hid: row for row, hid in enumerate(self.hyperedge_ids)}

    def vector(self, hyperedge_id: str) -> np.ndarray:
        """The embedding of one hyperedge.

        Raises:
            KeyError: If the id is absent. Every hyperedge in ``H`` is embedded, so a miss means the
                caller indexed embeddings built from a different hypergraph — worth failing loudly.
        """
        try:
            row = self.index[hyperedge_id]
        except KeyError as exc:
            raise KeyError(f"no embedding for hyperedge {hyperedge_id!r}") from exc
        return self.matrix[row]

    def rows(self, hyperedge_ids: Iterable[str]) -> np.ndarray:
        """Stack the embeddings of several hyperedges into a ``(len(ids), dim)`` array."""
        index = self.index
        return np.stack([self.matrix[index[hid]] for hid in hyperedge_ids])


# ---------------------------------------------------------------------------
# Embedding a corpus
# ---------------------------------------------------------------------------


def embed_texts(
    texts: Sequence[str],
    *,
    backend: EmbeddingClient,
    cfg: Config,
    store: EmbeddingStore | None = None,
) -> np.ndarray:
    """Embed ``texts``, reusing the store for anything already present.

    Args:
        texts: Strings to embed. Duplicates within one call are embedded once.
        backend: The embedding backend. §A.4 uses ``text-embedding-3-small``.
        cfg: The root config; supplies the model name and ``llm.embedding_batch_size``.
        store: Optional content-addressed cache.

    Returns:
        A ``(len(texts), 1536)`` float32 array whose row *i* corresponds to ``texts[i]``.

    Raises:
        ValueError: If ``backend`` returns a dimension other than ``cfg.order_model.embedding_dim``.
            The expected width is read from config rather than hardcoded so that this check and the
            §A.4 parameter-count check in :func:`~okh_rag.config._validate` cannot drift apart.
    """
    expected_dim = cfg.order_model.embedding_dim
    if not texts:
        return np.zeros((0, expected_dim), dtype=np.float32)

    vectors: list[np.ndarray | None] = [None] * len(texts)
    pending: dict[str, list[int]] = {}
    for position, text in enumerate(texts):
        if store is not None:
            cached = store.get(text)
            if cached is not None:
                vectors[position] = cached
                continue
        pending.setdefault(text, []).append(position)

    if pending:
        unique_texts = list(pending)
        batch_size = max(1, cfg.llm.embedding_batch_size)
        for start in range(0, len(unique_texts), batch_size):
            batch = unique_texts[start : start + batch_size]
            produced = np.asarray(backend.embed(batch), dtype=np.float32)
            if produced.ndim != 2 or produced.shape[1] != expected_dim:
                raise ValueError(
                    f"backend returned shape {produced.shape}; §A.4 requires "
                    f"(n, {expected_dim})"
                )
            for text, vector in zip(batch, produced):
                for position in pending[text]:
                    vectors[position] = vector
                if store is not None:
                    store.put(text, vector)

    missing = [i for i, v in enumerate(vectors) if v is None]
    if missing:
        # Only reachable if the store returned a vector for a text that was never embedded, which would
        # silently shift every downstream row. Fail instead of stacking a short matrix.
        raise RuntimeError(f"no vector produced for {len(missing)} text(s) at positions {missing[:5]}")
    return np.stack(vectors).astype(np.float32, copy=False)  # type: ignore[arg-type]


def embed_hypergraph(
    hypergraph: KnowledgeHypergraph,
    *,
    backend: EmbeddingClient,
    cfg: Config,
    store: EmbeddingStore | None = None,
) -> HyperedgeEmbeddings:
    """Embed every hyperedge of ``H`` per §A.4.

    Row order is the hyperedge ids sorted lexicographically, so the matrix is reproducible and can be
    compared across runs even when the ids were produced in a different insertion order.

    The store is flushed once at the end rather than per vector: embedding a corpus is thousands of
    vectors, and one flush per shard is the difference between 256 writes and thousands.
    """
    ordered_ids = tuple(sorted(hypergraph.hyperedges))
    texts = [hyperedge_embedding_text(hypergraph.hyperedges[hid], hypergraph) for hid in ordered_ids]
    matrix = embed_texts(texts, backend=backend, cfg=cfg, store=store)
    if store is not None:
        store.flush()
    return HyperedgeEmbeddings(hyperedge_ids=ordered_ids, matrix=matrix)


def embedding_text_preview(hypergraph: KnowledgeHypergraph, limit: int = 3) -> list[Mapping[str, str]]:
    """The first ``limit`` embedding inputs, for eyeballing §A.4's assembly during a run.

    Worth checking by eye on a new corpus: the four parts of §A.4's text are assembled from three
    different places (relation, evidence, and the entity/attribute tables), so a malformed id or an
    attribute that stringifies badly shows up here and nowhere else.
    """
    out: list[Mapping[str, str]] = []
    for hyperedge_id in sorted(hypergraph.hyperedges)[:limit]:
        edge = hypergraph.hyperedges[hyperedge_id]
        text = hyperedge_embedding_text(edge, hypergraph)
        out.append(
            {
                "hyperedge_id": hyperedge_id,
                "text": text,
                "text_key": _text_key(text),
            }
        )
    return out
