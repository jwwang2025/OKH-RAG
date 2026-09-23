"""§4.1 — the three baselines, each producing the same evidence type as OKH-RAG.

§4.1 defines the comparison by contrast, and the contrast is entirely about *what structure survives
retrieval*:

* **Text-RAG** — "dense retrieval over unstructured text chunks", then "concatenates the top-k results as
  context".
* **GraphRAG** — "retrieves over a binary knowledge graph using graph traversal and community
  summarization", citing Edge et al. 2024.
* **HyperGraphRAG** — "retrieves unordered hyperedges from the same knowledge hypergraph H without modeling
  order".
* **OKH-RAG** — "order-aware hyperedge trajectories from H≺ using the full objective in Equation 5".

All four are run through the same generator, which is only a meaningful control if the generator also
receives **comparable** input. So every retriever here returns an
:class:`~okh_rag.generation.evidence.EvidenceContext` — the same object
:func:`~okh_rag.generation.evidence.context_from_retrieval` builds for OKH-RAG. What differs is what the
context can say:

| source | relation | entity set | horizon | step order | reasoning tags |
|---|---|---|---|---|---|
| `okh_rag` | yes | yes | yes | yes | yes |
| `hypergraph_rag` | yes | yes | yes | **no** | no |
| `graph_rag` | yes | yes | yes | **no** | no |
| `text_rag` | **no** | **no** | **no** | **no** | no |

The blanks are the treatment. Filling them in — numbering the unordered baselines' blocks as ``[Step k]``,
or annotating a prose chunk with a normalised ``[T-48]`` header — would hand each baseline a piece of the
representation under test and make the comparison measure nothing. Each omission is justified at the point
it happens rather than being left implicit.

**Two documented approximations.** §4.1 cites Edge et al. 2024 for GraphRAG, whose pipeline (Leiden
communities, generated community reports) is defined in that paper and not in this one.
``configs/base.yaml`` already flags this; what is done here is the part that can be done from the
hypergraph alone — traverse from the dense-retrieval seeds and summarise the traversed neighbourhood —
without pretending to be the full pipeline. The corpus for Text-RAG is the vendored NOAA text tree, whose
file naming (``{STORM}_{YEAR}_{H}h.txt``) is the release's own input protocol per ``data/CONTRACT.md``
§2.2, so "chunking" is not a free parameter here: a chunk *is* one product at one lead time.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import numpy as np

from okh_rag.data.cyportqa import QAContext, QARecord
from okh_rag.embeddings import HyperedgeEmbeddings
from okh_rag.generation.evidence import (
    EvidenceContext,
    EvidenceTrajectory,
    Passage,
    trajectory_from_hyperedge_ids,
)
from okh_rag.retrieval.candidates import CandidatePool, HypergraphIndex, scope_candidates
from okh_rag.schema import KnowledgeHypergraph
from okh_rag.vocabulary import family_of

if TYPE_CHECKING:  # pragma: no cover - typing only
    from okh_rag.config import BaselinesConfig, RetrievalConfig

__all__ = [
    "BASELINE_SOURCES",
    "ChunkIndex",
    "GraphRagRetriever",
    "HypergraphRagRetriever",
    "PortConditionTable",
    "RawChunk",
    "TextRagRetriever",
    "build_baseline_context",
    "load_raw_text_chunks",
]

#: The baseline names, as they appear in ``source`` on an :class:`EvidenceContext` and in a report.
BASELINE_SOURCES = ("text_rag", "graph_rag", "hypergraph_rag")

#: ``modalities`` value → directory under ``raw_text/``. From ``data/CONTRACT.md`` §2.3: the
#: ``modalities`` strings are *not* the directory names, so the mapping has to be stated explicitly rather
#: than derived by naming convention.
TEXT_MODALITY_DIRS: Mapping[str, str] = {
    "text_advisory": "Cyclone Text Archive Advisory",
    "Table_wind": "Cyclone Text Archive Wind",
}

#: ``{STORM}_{YEAR}_{LEAD}h.txt`` — the release's own input-file naming, per ``data/CONTRACT.md`` §2.2.
_CHUNK_NAME = re.compile(r"^(?P<storm>.+)_(?P<year>\d{4})_(?P<hours>\d+)h\.txt$")


# ---------------------------------------------------------------------------
# Text-RAG's corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawChunk:
    """One NOAA text product at one lead time — Text-RAG's unit of evidence.

    Attributes:
        storm: The storm name as it appears in the filename (``ALEX``).
        year: The storm year, from the filename.
        horizon_hours: The lead time, from the filename.
        modality: The ``modalities`` value this directory corresponds to.
        text: The file's contents.
    """

    storm: str
    year: int
    horizon_hours: int
    modality: str
    text: str

    @property
    def chunk_id(self) -> str:
        """``ALEX_2022_48h/text_advisory`` — stable, and the name the file is found under."""
        return f"{self.storm}_{self.year}_{self.horizon_hours}h/{self.modality}"


def load_raw_text_chunks(root: Path) -> tuple[RawChunk, ...]:
    """Read the vendored NOAA text tree into chunks, deterministically ordered.

    Ordering is by ``(storm, year, horizon, modality)`` rather than by directory-walk order, because the
    chunk list's order decides the tie-break when two chunks score equally, and a tie-break that depends on
    filesystem enumeration is a run that cannot be reproduced.

    Args:
        root: ``data/raw/cyportqa`` — the directory holding ``raw_text/``.

    Returns:
        Every readable ``.txt`` chunk, sorted. A missing tree yields an empty tuple rather than raising:
        Text-RAG is one of four methods, and its corpus not being vendored should drop one row from a
        table, not abort the run.
    """
    base = root / "raw_text" / "NOAA NHC Cyclone Products"
    chunks: list[RawChunk] = []
    for modality, directory in TEXT_MODALITY_DIRS.items():
        folder = base / directory
        if not folder.is_dir():
            continue
        for path in folder.glob("*.txt"):
            match = _CHUNK_NAME.match(path.name)
            if match is None:
                # A filename outside the release's convention cannot be attributed to a storm and lead
                # time, so it could never be cited as evidence. Skipped rather than guessed at.
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            chunks.append(
                RawChunk(
                    storm=match.group("storm"),
                    year=int(match.group("year")),
                    horizon_hours=int(match.group("hours")),
                    modality=modality,
                    text=text,
                )
            )
    chunks.sort(key=lambda c: (c.storm, c.year, c.horizon_hours, c.modality))
    return tuple(chunks)


@dataclass(frozen=True)
class PortConditionTable:
    """The USCG port-condition bulletins, keyed by port.

    Part of the vendored prose corpus but **not** wired into Text-RAG's chunk set: the bulletins are one
    aggregated CSV covering 275 rows across the whole release, with no storm or lead-time column pairing
    them to a scenario, so a retrieved row could not be rendered with the horizon the question is about.

    Kept as a loaded, queryable object because §4.2 treats port conditions as an operation-phase signal and
    a future structured baseline may want it; leaving it unread would make that harder to add and would
    hide the fact that a vendored input is currently unused.
    """

    rows: tuple[Mapping[str, str], ...] = ()

    @classmethod
    def load(cls, root: Path) -> "PortConditionTable":
        """Read ``raw_text/Port_Condition_Bullitens/uscg_port_conditions_agg.csv``."""
        path = root / "raw_text" / "Port_Condition_Bullitens" / "uscg_port_conditions_agg.csv"
        if not path.is_file():
            return cls()
        with path.open(encoding="utf-8", errors="replace", newline="") as handle:
            return cls(rows=tuple(dict(row) for row in csv.DictReader(handle)))

    def conditions_for(self, port: str) -> tuple[dict[str, str], ...]:
        """Rows whose ``PortList`` mentions ``port``.

        Substring rather than equality: ``PortList`` holds a delimited list of ports, and the port names in
        it are the release's own (``Savannah``) rather than the QA records' (``Port of Savannah, GA``).
        """
        needle = port.lower()
        return tuple(row for row in self.rows if needle in str(row.get("PortList", "")).lower())


@dataclass(frozen=True)
class ChunkIndex:
    """Text-RAG's corpus with its embeddings, ready for dense retrieval.

    The matrix is held alongside the chunk tuple rather than looked up per query because the whole
    corpus is under a thousand chunks — a lookup structure would cost more than the matmul it replaces.
    """

    chunks: tuple[RawChunk, ...]
    matrix: np.ndarray

    def __post_init__(self) -> None:
        if len(self.chunks) != self.matrix.shape[0]:
            raise ValueError(
                f"chunk/embedding count mismatch: {len(self.chunks)} chunks, {self.matrix.shape[0]} rows"
            )

    def __len__(self) -> int:
        return len(self.chunks)

    def top_k(self, query_vector: np.ndarray, k: int) -> tuple[tuple[RawChunk, float], ...]:
        """The ``k`` chunks most cosine-similar to ``query_vector``.

        Ties break by chunk order, which :func:`load_raw_text_chunks` made deterministic. Ranking is
        ``argsort`` on the *negated* score so the result is descending; ``np.argsort`` defaulting to
        ascending is the kind of detail that silently reverses a baseline's evidence order.

        Returns:
            ``(chunk, cosine)`` pairs, best first, at most ``k`` of them.
        """
        if not len(self.chunks):
            return ()
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norms = np.linalg.norm(self.matrix, axis=1) * max(float(np.linalg.norm(query)), 1e-12)
        similarities = (self.matrix @ query) / np.maximum(norms, 1e-12)
        order = np.argsort(-similarities, kind="stable")[: max(0, k)]
        return tuple((self.chunks[int(i)], float(similarities[int(i)])) for i in order)


# ---------------------------------------------------------------------------
# Retriever: Text-RAG
# ---------------------------------------------------------------------------


class TextRagRetriever:
    """§4.1's Text-RAG: dense top-k over unstructured chunks, concatenated as context.

    Retrieval is **over the whole chunk corpus**, not restricted to the question's storm. §4.1 describes
    the baseline as "dense retrieval over unstructured text chunks" with no scoping step, and §A.7's
    candidate scoping — the top-K cosine *plus* group-aware expansion — is presented as part of OKH-RAG's
    contribution rather than as a shared prelude. Restricting Text-RAG to the question's own scenario would
    quietly give it a scope filter that the method under test has to earn, and would hide the failure mode
    that makes it a useful baseline: a query naming a port can retrieve another storm's advisory whose text
    names the same port.
    """

    name = "text_rag"

    def __init__(self, index: ChunkIndex, cfg: "BaselinesConfig") -> None:
        self.index = index
        self.cfg = cfg

    def context(self, record: QARecord, query_vector: np.ndarray, **_: object) -> EvidenceContext:
        """Retrieve and render. ``**_`` absorbs the keyword arguments the other retrievers need.

        The uniform ``context(record, query_vector, **kwargs)`` signature is what lets a driver hold the
        four retrievers in one loop. A signature that matched each retriever exactly would put a type test
        at the call site, which is where a baseline comparison is least able to afford one.
        """
        hits = self.index.top_k(query_vector, self.cfg.text_rag.top_k)
        steps = tuple(
            Passage(
                step=position,
                text=chunk.text,
                horizon_hours=chunk.horizon_hours,
                source=f"{chunk.storm} {chunk.year} T-{chunk.horizon_hours} {chunk.modality}",
            )
            for position, (chunk, _score) in enumerate(hits, start=1)
        )
        trajectory = EvidenceTrajectory(
            index=1,
            steps=steps,
            score=float(sum(score for _chunk, score in hits)),
            ordered=False,
        )
        notes = (
            "Retrieved by dense similarity over unstructured NOAA text products; "
            "the chunks are presented as an unordered set.",
        )
        return EvidenceContext(
            question=record.question,
            trajectories=(trajectory,),
            source=self.name,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Retrievers over H: HyperGraphRAG and GraphRAG
# ---------------------------------------------------------------------------


class HypergraphRagRetriever:
    """§4.1's HyperGraphRAG: the same hypergraph, the same pool, **no ordering**.

    §4.1 calls this "retrieves unordered hyperedges from the same knowledge hypergraph H without modeling
    order", and the config's comment makes the equivalence exact: it is "the λ=µ=ν=ρ=0 case of Eq. 5",
    which §3.2 confirms "recovers standard top-k retrieval".

    So the implementation is deliberately **not** a second retriever: it builds §A.7's candidate pool ``C``
    exactly as OKH-RAG does and then takes the top-k by relevance. Sharing the scoping step is what makes
    the comparison a test of ordering — if this baseline also had different candidates, a difference in
    accuracy could be a difference in retrieval scope rather than in order modelling, and no amount of
    ablation elsewhere would recover the distinction.
    """

    name = "hypergraph_rag"

    def __init__(
        self,
        hypergraph: KnowledgeHypergraph,
        embeddings: HyperedgeEmbeddings,
        retrieval_cfg: "RetrievalConfig",
        baselines_cfg: "BaselinesConfig",
    ) -> None:
        self.hypergraph = hypergraph
        self.embeddings = embeddings
        self.retrieval_cfg = retrieval_cfg
        self.cfg = baselines_cfg
        self.index = HypergraphIndex(hypergraph)

    def pool(self, record: QARecord, query_vector: np.ndarray) -> CandidatePool:
        """§A.7's ``C`` for this question, with the question's own group as the reserve scope."""
        return scope_candidates(
            self.hypergraph,
            self.embeddings,
            query_vector,
            cfg=self.retrieval_cfg,
            index=self.index,
            group_id=record.group_id,
        )

    def context(self, record: QARecord, query_vector: np.ndarray, **_: object) -> EvidenceContext:
        """Top-k of the shared pool by relevance, rendered as an unordered set."""
        pool = self.pool(record, query_vector)
        ranked = list(pool.hyperedge_ids)[: self.cfg.hypergraph_rag.top_k]
        relevance = {hid: float(pool.relevance[i]) for i, hid in enumerate(pool.hyperedge_ids)}
        trajectory = trajectory_from_hyperedge_ids(
            self.hypergraph,
            ranked,
            index=1,
            score=float(sum(relevance.get(hid, 0.0) for hid in ranked)),
            ordered=False,
        )
        notes = (
            "Retrieved as unordered hyperedges from the same knowledge hypergraph "
            "(the lambda=mu=nu=rho=0 setting of Eq. 5).",
        )
        return EvidenceContext(
            question=record.question,
            trajectories=(trajectory,),
            source=self.name,
            notes=notes,
        )


class GraphRagRetriever:
    """§4.1's GraphRAG, as far as the hypergraph alone allows.

    §4.1: "retrieves over a binary knowledge graph using graph traversal and community summarization",
    citing Edge et al. 2024. Three parts of that pipeline are defined in [11] and not in this paper:
    decomposing the knowledge graph into a binary one, running Leiden community detection, and generating a
    community report per community. What *is* defined here is the traversal

    # [SIMPLIFICATION] The approximation, stated rather than glossed:
    #   1. seeds  = the top-k hyperedges of §A.7's pool by relevance (the dense-retrieval entry point
    #               every graph RAG shares);
    #   2. expand = the seeds' one-hop neighbourhood, via HypergraphIndex.neighbours — group membership
    #               plus entity overlap, the same traversal OKH-RAG's stage 2 uses;
    #   3. summary = a **compositional** report over the expanded set: how many hyperedges, at which
    #               horizons, in which families. A generated natural-language community report would need
    #               its own LLM call per community, and its wording would then be a second uncontrolled
    #               variable in a comparison whose point is to control the generator.
    # Because the expansion is over entities, this baseline *can* leave the question's own group — which is
    # the property that distinguishes it from HypergraphRAG in §4.1's ordering of the four methods.

    ``include_pairwise_edges`` from ``configs/base.yaml`` is recorded but not acted on: it would require the
    binary-graph decomposition, which is the same [11]-defined step as above. A reader comparing this run to
    §4.1 should know the flag is inert here.
    """

    name = "graph_rag"

    def __init__(
        self,
        hypergraph: KnowledgeHypergraph,
        embeddings: HyperedgeEmbeddings,
        retrieval_cfg: "RetrievalConfig",
        baselines_cfg: "BaselinesConfig",
    ) -> None:
        self.hypergraph = hypergraph
        self.embeddings = embeddings
        self.retrieval_cfg = retrieval_cfg
        self.cfg = baselines_cfg
        self.index = HypergraphIndex(hypergraph)
        self._hypergraph_rag = HypergraphRagRetriever(
            hypergraph, embeddings, retrieval_cfg, baselines_cfg
        )

    def context(self, record: QARecord, query_vector: np.ndarray, **_: object) -> EvidenceContext:
        """Seeds, one-hop traversal, compositional summary, unordered render."""
        pool = self._hypergraph_rag.pool(record, query_vector)
        seeds = list(pool.hyperedge_ids)[: max(1, self.retrieval_cfg.top_k)]

        reached: list[str] = list(seeds)
        seen: set[str] = set(seeds)
        frontier = seeds
        for _hop in range(max(0, self.cfg.graph_rag.expansion_hops)):
            neighbours = sorted(self.index.neighbours(frontier))
            frontier = [hid for hid in neighbours if hid not in seen]
            if not frontier:
                break
            seen.update(frontier)
            reached.extend(frontier)
            # The traversal's budget is the same as HyperGraphRAG's top-k, so both baselines present the
            # same *number* of evidence blocks and differ only in what the blocks are and whether they are
            # ordered. An unequal evidence budget would confound the comparison with a context-length
            # effect, which is a well-known driver of QA accuracy on its own.
            reached = reached[: self.cfg.hypergraph_rag.top_k]

        trajectory = trajectory_from_hyperedge_ids(
            self.hypergraph,
            reached,
            index=1,
            ordered=False,
        )
        notes = tuple(
            note
            for note in (
                "Retrieved by graph traversal from the dense-retrieval seeds; "
                "the traversed neighbourhood is presented as an unordered set.",
                self._community_summary(reached),
            )
            if note
        )
        return EvidenceContext(
            question=record.question,
            trajectories=(trajectory,),
            source=self.name,
            notes=notes,
        )

    def _community_summary(self, hyperedge_ids: Sequence[str]) -> str:
        """The "community summarization" half, as a compositional report over the traversed set.

        Returns an empty string for an empty set rather than a summary of nothing, so the caller's ``if
        note`` filter drops it and the evidence block has no dangling heading.
        """
        if not hyperedge_ids:
            return ""
        horizons: dict[int, int] = {}
        families: dict[str, int] = {}
        for hyperedge_id in hyperedge_ids:
            edge = self.hypergraph.hyperedges.get(hyperedge_id)
            if edge is None:
                continue
            if edge.horizon_hours is not None:
                horizons[edge.horizon_hours] = horizons.get(edge.horizon_hours, 0) + 1
            name = family_of(edge.relation) or "unknown"
            families[name] = families.get(name, 0) + 1
        horizon_text = ", ".join(f"T-{h}: {n}" for h, n in sorted(horizons.items()))
        family_text = ", ".join(f"{name}: {n}" for name, n in sorted(families.items()))
        return (
            f"Community summary: {len(hyperedge_ids)} hyperedges; "
            f"horizons [{horizon_text or 'none'}]; families [{family_text or 'none'}]."
        )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def build_baseline_context(
    source: str,
    record: QARecord,
    query_vector: np.ndarray,
    *,
    retrievers: Mapping[str, object],
) -> EvidenceContext:
    """Render one question's evidence with the named baseline.

    Args:
        source: One of :data:`BASELINE_SOURCES`, or ``okh_rag``. ``okh_rag`` is rejected here rather than
            silently accepted, because OKH-RAG's context is built from a
            :class:`~okh_rag.retrieval.RetrievalResult` (which carries the trajectories, term scores and
            shuffle flag) and constructing one from a raw query vector would discard exactly the
            information the method contributes.
        record: The question.
        query_vector: The embedded question, shared by all four methods per §4.1's "same embedding model".
        retrievers: Registry of constructed retrievers, keyed by ``source``.

    Returns:
        The rendered context.

    Raises:
        KeyError: If ``source`` names a retriever that was not constructed — a run that asked for a
            baseline it did not build should say so rather than fall back to another method.
        ValueError: If ``source`` is ``okh_rag``; see above.
    """
    if source == "okh_rag":
        raise ValueError(
            "okh_rag contexts come from context_from_retrieval(result, hypergraph); "
            "they cannot be rebuilt from a query vector alone"
        )
    if source not in retrievers:
        raise KeyError(f"no retriever named {source!r}; built: {sorted(retrievers)}")
    retriever = retrievers[source]
    context = retriever.context(record, query_vector)  # type: ignore[attr-defined]
    return context


def scenario_chunk_scope(record: QARecord, chunks: Iterable[RawChunk]) -> tuple[RawChunk, ...]:
    """The chunks belonging to the question's own storm and year.

    Not used by :class:`TextRagRetriever`, which retrieves globally — see its docstring. Provided because
    the per-scenario view is the right unit for a report ("how much text exists for the scenarios being
    evaluated?") and because a scenario-scoped ablation of Text-RAG is the obvious next experiment: it
    separates "dense retrieval found the wrong storm" from "the prose does not contain the answer".
    """
    context: QAContext = record.context
    return tuple(
        chunk for chunk in chunks if chunk.storm == context.storm and chunk.year == context.year
    )
