"""§3.2 §A.7 — the retriever: scoring, candidate scoping and inference, wired together.

The three stages map onto the three sibling modules, and this one only does the wiring plus the two
things that belong to no single stage:

* :mod:`okh_rag.retrieval.candidates` — §A.7's two-stage scoping, producing ``C``.
* :mod:`okh_rag.retrieval.scoring` — §A.6's five terms and the ``|C| × |C|`` precomputations.
* :mod:`okh_rag.retrieval.beam` / :mod:`okh_rag.retrieval.viterbi` — §A.7's two inference algorithms.

**The four ``transition_variant`` settings and what each answers.** ``ablation.transition_variant``
selects a Table 2 *Transition* row, and the mapping to code is not uniform, so it is stated once here:

| ``transition_variant`` | transition table          | weights                          | Table 2 row    |
|------------------------|---------------------------|----------------------------------|----------------|
| ``"learned"``          | §3.1 Eq. 3 over ``C``     | §4.1's (1.2, 0.3, 0.2, 0.5)      | full model     |
| ``"heuristic"``        | §A.3 rules only           | §4.1's (1.2, 0.3, 0.2, 0.5)      | heuristic order|
| ``"none"``             | zero table                | (0, 0, 0, 0)                     | no order       |

``"none"`` zeroes the weights as well as the table because that is the configuration §3.2 *defines* as
order-free — "setting all to zero recovers standard top-k retrieval" — and the two halves have to move
together: a zero table with λ ≠ 0 would still run the beam over L distinct steps and, with µ, ν, ρ alive,
would still be an order-aware method. ``ambiguity_audit.md`` records that the Table 2 hierarchy
(0.501 → 0.509 → 0.534) is only reproducible if "no order" means the whole objective collapses, which is
also the reading that makes the row agree with the HyperGraphRAG baseline §4.1 describes as "unordered
hyperedges from the same knowledge hypergraph H without modeling order".

**Multi-trajectory retrieval.** §3.2 — "We therefore retrieve a set of diverse trajectories
Γq = {γ1, ..., γN}, enabling the model to consider alternative explanations and complementary evidence
chains." ``num_trajectories`` is N = 3. Beam search supplies N naturally (Algorithm 1 line 12 returns
"Top-N trajectories with score breakdowns"); Viterbi supplies exactly one, because it solves for a single
argmax. Returning one trajectory from the Viterbi path is what §A.7 means by calling it a ceiling rather
than a replacement.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Protocol, runtime_checkable

import numpy as np

from okh_rag.config import INFERENCE_MODES, Config, RetrievalWeights
from okh_rag.embeddings import EmbeddingStore, HyperedgeEmbeddings, embed_texts
from okh_rag.llm.base import EmbeddingClient
from okh_rag.ordering.precedence import PrecedenceGraph
from okh_rag.retrieval.beam import BeamSearchResult, order_aware_beam_search
from okh_rag.retrieval.candidates import CandidatePool, HypergraphIndex, scope_candidates
from okh_rag.retrieval.scoring import (
    TRANSITION_VARIANTS,
    CandidateScores,
    TrajectoryScorer,
    heuristic_transition_matrix,
    learned_transition_matrix,
    zero_transition_matrix,
)
from okh_rag.retrieval.viterbi import ViterbiResult, viterbi_search
from okh_rag.schema import Hyperedge, KnowledgeHypergraph, Trajectory, TrajectoryScore

__all__ = [
    "INFERENCE_MODES",
    "OrderAwareRetriever",
    "RetrievalResult",
    "Retriever",
    "resolve_weights",
]

# ``INFERENCE_MODES`` is declared in :mod:`okh_rag.config` and re-exported here. §A.7 — "Beam search
# serves as the primary inference algorithm; Viterbi DP provides exact optimization for smaller candidate
# sets." Beam search is the default because the paper calls it primary.


def resolve_weights(cfg: Config, transition_variant: str) -> RetrievalWeights:
    """The four Eq. 5 weights this variant runs with.

    §4.5's ``drop_*`` ablations are applied first (by :meth:`~okh_rag.config.Config.effective_weights`),
    then the order-free variant collapses whatever remains to zero. Doing it in that order means the
    ``−λ`` row and the ``no order`` row cannot be confused: the first removes one term from the full
    objective, the second removes the objective.

    Args:
        cfg: The root config.
        transition_variant: One of :data:`~okh_rag.retrieval.scoring.TRANSITION_VARIANTS`.

    Returns:
        The weights for this run.
    """
    if transition_variant not in TRANSITION_VARIANTS:
        raise ValueError(f"unknown transition variant {transition_variant!r}; expected {TRANSITION_VARIANTS}")
    weights = cfg.effective_weights()
    if transition_variant == "none":
        return RetrievalWeights(lambda_=0.0, mu=0.0, nu=0.0, rho=0.0)
    return weights


class _HorizonIndex:
    """``(group, horizon) -> hyperedge ids``, plus the block-less cross-horizon edges.

    Built once per retriever for the same reason as :class:`~okh_rag.retrieval.candidates.HypergraphIndex`:
    which hyperedges belong to a horizon is a property of ``H``, not of the query, and rebuilding the
    map per question would be a full scan of the hypergraph each time.

    §A.2 step 4's cross-horizon change edges carry ``horizon_hours is None`` — they belong to no block —
    which is why :attr:`cross` is a separate bucket: a caller that wants them asks for them, and one
    that does not is not silently given a hyperedge that straddles every block.
    """

    __slots__ = ("by_horizon", "cross")

    def __init__(self, hypergraph: KnowledgeHypergraph) -> None:
        by_horizon: dict[tuple[str, int], list[str]] = {}
        cross: dict[str, list[str]] = {}
        for hyperedge_id, edge in hypergraph.hyperedges.items():
            if edge.horizon_hours is None:
                cross.setdefault(edge.group_id or "", []).append(hyperedge_id)
            else:
                by_horizon.setdefault((edge.group_id or "", edge.horizon_hours), []).append(hyperedge_id)
        self.by_horizon: Mapping[tuple[str, int], tuple[str, ...]] = {
            key: tuple(value) for key, value in by_horizon.items()
        }
        self.cross: Mapping[str, tuple[str, ...]] = {key: tuple(value) for key, value in cross.items()}

    def universe(
        self, group_id: str | None, horizon_hours: int | None, mode: str
    ) -> tuple[str, ...] | None:
        """The searchable ids for one question, or ``None`` when scoping does not apply.

        ``None`` means "no restriction", not "empty" — an empty universe would make
        :func:`~okh_rag.retrieval.candidates.scope_candidates` return an empty pool and the caller would
        silently get no evidence. A group whose block cannot be found therefore falls back to the whole
        group rather than to nothing.
        """
        if mode == "off" or horizon_hours is None or group_id is None:
            return None
        own = self.by_horizon.get((group_id, horizon_hours), ())
        if not own:
            return None
        if mode == "own_block_plus_cross":
            return tuple(dict.fromkeys((*own, *self.cross.get(group_id, ()))))
        return own


@dataclass(frozen=True)
class RetrievalResult:
    """Everything one query produced, including the intermediate stages that make it auditable.

    The pool and the precomputed scores are kept on the result rather than discarded because both are
    evaluation surface: the template→field oracle in ``data/CONTRACT.md`` §5 asks "were the hyperedges
    built from the field this question needs even *in* ``C``?", which separates a retrieval failure from a
    generation failure. §4.1's end-to-end accuracy cannot make that distinction, so the artifacts have to
    survive the call.

    Attributes:
        query: The query text.
        group_id: The query's knowledge group, when known.
        pool: §A.7's ``C``.
        scores: §A.6's precomputations over ``C``.
        trajectories: ``Γ_q``, best first. Length ``num_trajectories`` for beam search, 1 for Viterbi.
        term_scores: The Eq. 5 breakdown per trajectory, aligned with :attr:`trajectories`.
        variant: ``ablation.transition_variant`` for this result.
        inference: Which §A.7 algorithm produced it.
        shuffled: Whether the §4.5 permutation stress test was applied.
        beam: Algorithm 1's internal trace, when beam search ran. ``None`` otherwise.
        viterbi: The DP's objective value, when Viterbi ran. ``None`` otherwise.
    """

    query: str
    group_id: str | None
    pool: CandidatePool
    scores: CandidateScores
    trajectories: tuple[Trajectory, ...]
    term_scores: tuple[TrajectoryScore, ...]
    variant: str
    inference: str = "beam_search"
    shuffled: bool = False
    beam: BeamSearchResult | None = None
    viterbi: ViterbiResult | None = None

    def best(self) -> Trajectory | None:
        """The highest-scoring trajectory, or ``None`` when nothing was retrieved."""
        return self.trajectories[0] if self.trajectories else None

    def evidence(self, hypergraph: KnowledgeHypergraph) -> list[tuple[int, Hyperedge]]:
        """The best trajectory as the ``(step, hyperedge)`` pairs §A.8's evidence block renders.

        A convenience rather than a second implementation: it delegates to
        :meth:`~okh_rag.schema.Trajectory.to_evidence_sequence`, so the step numbering the generator sees
        and the step numbering the schema defines cannot diverge.
        """
        best = self.best()
        return [] if best is None else best.to_evidence_sequence(hypergraph)


@runtime_checkable
class Retriever(Protocol):
    """What §4.1's baselines and OKH-RAG have in common, so the evaluator can loop over all four.

    §4.4's comparison of OKH-RAG against HyperGraphRAG "isolates the contribution of order-awareness
    beyond the representational advantage of hypergraphs alone", which only holds if both are driven by
    one evaluator over one hypergraph. A shared protocol is the mechanism that guarantees it.
    """

    def retrieve(
        self,
        query: str,
        *,
        group_id: str | None = ...,
        allowed_ids: Iterable[str] | None = ...,
        query_vector: np.ndarray | None = ...,
        length: int | None = ...,
        horizon_hours: int | None = ...,
    ) -> RetrievalResult:
        """Retrieve ``Γ_q`` for a query."""
        ...


class OrderAwareRetriever:
    """§3.2 / §A.7 — order-aware trajectory retrieval over a knowledge hypergraph.

    One instance owns one corpus and is reused across queries: the §A.7 stage-2 indices and the §A.6
    transition table for a query's *pool* are both query-specific, but the reverse maps of
    :class:`~okh_rag.retrieval.candidates.HypergraphIndex` are not, and rebuilding them per question would
    dominate the cost of a §4-scale evaluation.
    """

    def __init__(
        self,
        *,
        cfg: Config,
        hypergraph: KnowledgeHypergraph,
        precedence: PrecedenceGraph,
        embeddings: HyperedgeEmbeddings,
        order_model: object | None = None,
        backend: EmbeddingClient | None = None,
        store: EmbeddingStore | None = None,
        transition_variant: str | None = None,
        inference: str | None = None,
    ) -> None:
        """Bind the retriever to one corpus.

        Args:
            cfg: The root config. Its ``retrieval``, ``ablation``, ``llm`` and ``seed`` sections are read
                at call time, so an ablation switch flipped between two runs of the same object takes
                effect — which is how the Table 2 harness reuses one retriever.
            hypergraph: ``H``.
            precedence: ``≺`` over ``H``'s hyperedges, from §A.3.
            embeddings: ``h_e`` for every hyperedge of ``H``.
            order_model: The §3.1 ``P_θ``. Required when the learned variant is selected; unused by the
                heuristic and order-free variants, which is why it is not a constructor error to omit it
                in those cases.
            backend: The embedding client used to embed *queries*. §A.6's ``Rel`` is a cosine in the same
                space as ``h_e``, so the query must be embedded by the same model. Unused when the caller
                supplies a precomputed ``query_vector``.
            store: Optional content-addressed embedding cache, shared with the corpus embeddings so a
                repeated query text costs nothing.
            transition_variant: Overrides ``ablation.transition_variant``.
            inference: Overrides the default ``beam_search``.

        Raises:
            ValueError: If the learned variant is selected without an ``order_model``. Failing here is
                deliberate: the alternative is a KeyError or an AttributeError inside the first
                ``retrieve`` call, after the corpus has been loaded and the embeddings paid for.
        """
        self.cfg = cfg
        self.hypergraph = hypergraph
        self.precedence = precedence
        self.embeddings = embeddings
        self.order_model = order_model
        self.backend = backend
        self.store = store
        self.index = HypergraphIndex(hypergraph)
        self._horizon_index = _HorizonIndex(hypergraph)

        self.transition_variant = transition_variant or cfg.ablation.transition_variant
        if self.transition_variant not in TRANSITION_VARIANTS:
            raise ValueError(
                f"unknown transition_variant {self.transition_variant!r}; expected {TRANSITION_VARIANTS}"
            )
        if self.transition_variant == "learned" and order_model is None:
            raise ValueError(
                "transition_variant='learned' needs an order_model; pass one, or select 'heuristic' / "
                "'none'"
            )

        self.inference = inference or "beam_search"
        if self.inference not in INFERENCE_MODES:
            raise ValueError(f"unknown inference mode {self.inference!r}; expected {INFERENCE_MODES}")

    # -- query embedding ----------------------------------------------------

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one query string in §A.4's space.

        §A.6 — "cosine similarity in R^1536 between hyperedge and query embeddings". Both sides must come
        from ``text-embedding-3-small``; using a different model would make the cosine a comparison
        between two unrelated spaces, and it would still return a ranking.
        """
        if self.backend is None:
            raise ValueError(
                "no embedding backend configured; pass one to the constructor or supply query_vector"
            )
        matrix = embed_texts([query], backend=self.backend, cfg=self.cfg, store=self.store)
        return matrix[0]

    # -- precomputation -----------------------------------------------------

    def build_scores(self, pool: CandidatePool) -> CandidateScores:
        """§A.6 / Algorithm 1 line 1 — the ``Rel`` vector and the ``|C| × |C|`` transition table.

        Building the transition table is the expensive part of a query: the learned variant is one
        ``(|C|, 1536) × (1536, 64) × (64, |C|)`` chain for ~150 candidates, the heuristic variant is
        ``|C|²`` partial-order lookups. Both are linear in the *pool*, not the corpus, which is the point
        of scoping.
        """
        ids = pool.hyperedge_ids
        if self.transition_variant == "learned":
            transitions = learned_transition_matrix(self.order_model, self.embeddings, ids)
        elif self.transition_variant == "heuristic":
            transitions = heuristic_transition_matrix(self.precedence, ids)
        else:
            transitions = zero_transition_matrix(len(ids))
        return CandidateScores(
            hyperedge_ids=ids,
            relevance=np.asarray(pool.relevance, dtype=np.float64),
            transitions=transitions,
            variant=self.transition_variant,
        )

    # -- the call -----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        group_id: str | None = None,
        allowed_ids: Iterable[str] | None = None,
        query_vector: np.ndarray | None = None,
        length: int | None = None,
        horizon_hours: int | None = None,
    ) -> RetrievalResult:
        """Retrieve ``Γ_q`` for one query.

        Args:
            query: The query text. Carried through to the result for logging and the §A.8 prompt.
            group_id: The query's knowledge group, when known. Drives §A.7's 40% reserve.
            allowed_ids: Optional whitelist for the searchable universe — the group-split guard. Pass the
                training or test subset here when evaluating on a split; without it, retrieval would see
                every group including the one it is being scored on. Intersected with the horizon
                universe when both apply.
            query_vector: A precomputed query embedding. Supplying one skips the API call; this is the
                hook the smoke script and the tests use to run without a backend.
            length: ``L`` override, threaded to the inference algorithm.
            horizon_hours: The question's own horizon block (``T-48`` → ``48``). Drives
                ``retrieval.horizon_scoping``; see :class:`_HorizonIndex`. CyPortQA asks one question
                text per block with a block-dependent gold, so without this the blocks are
                indistinguishable to a cosine retriever and every block gets the same trajectory.

        Returns:
            A :class:`RetrievalResult`.
        """
        vector = query_vector if query_vector is not None else self.embed_query(query)
        universe = self._horizon_index.universe(
            group_id, horizon_hours, self.cfg.retrieval.horizon_scoping
        )
        if universe is not None:
            if allowed_ids is None:
                allowed_ids = universe
            else:
                keep = set(universe)
                allowed_ids = [hid for hid in allowed_ids if hid in keep]
        pool = scope_candidates(
            self.hypergraph,
            self.embeddings,
            vector,
            cfg=self.cfg.retrieval,
            index=self.index,
            group_id=group_id,
            allowed_ids=allowed_ids,
        )
        scores = self.build_scores(pool)
        weights = resolve_weights(self.cfg, self.transition_variant)
        scorer = TrajectoryScorer(
            hypergraph=self.hypergraph,
            precedence=self.precedence,
            scores=scores,
            cfg=self.cfg.retrieval,
            weights=weights,
        )

        if self.inference == "viterbi":
            outcome = viterbi_search(scorer, cfg=self.cfg.retrieval, length=length)
            positions = (outcome.positions,) if outcome.positions else ()
            beam = None
        else:
            outcome = order_aware_beam_search(scorer, cfg=self.cfg.retrieval, length=length)
            positions = tuple(entry.positions for entry in outcome.entries)
            beam = outcome

        term_scores = [scorer.score(p) for p in positions]
        ordered = self._order_trajectories(positions, term_scores, scorer)
        trajectories = tuple(
            Trajectory(
                hyperedge_ids=scorer.ids(p),
                score=score.total(weights.lambda_, weights.mu, weights.nu, weights.rho),
                components=score.as_dict(),
            )
            for p, score in ordered
        )
        term_scores = [score for _, score in ordered]

        return RetrievalResult(
            query=query,
            group_id=pool.group_id,
            pool=pool,
            scores=scores,
            trajectories=trajectories,
            term_scores=tuple(term_scores),
            variant=self.transition_variant,
            inference=self.inference,
            shuffled=self.cfg.ablation.shuffle_trajectory,
            beam=beam,
            viterbi=outcome if self.inference == "viterbi" else None,  # type: ignore[arg-type]
        )

    # -- §4.5 permutation stress test ---------------------------------------

    def _order_trajectories(
        self,
        positions: tuple[tuple[int, ...], ...],
        term_scores: list[TrajectoryScore],
        scorer: TrajectoryScorer,
    ) -> list[tuple[tuple[int, ...], TrajectoryScore]]:
        """Apply the §4.5 shuffle, then sort best-first.

        §4.5 — "OKH-RAG (shuffled) reduces overall accuracy from 0.534 to 0.487, the largest drop observed,
        despite identical retrieved content."

        The shuffle permutes the step order and the trajectory is **re-scored** on the permuted sequence.
        That differs from :meth:`~okh_rag.schema.Trajectory.shuffled`, which deliberately preserves the
        score; the reason for the difference is that a permuted sequence genuinely has a different Eq. 5
        value (``Rel`` and ``Cov`` are order-invariant, ``Trans``, ``Ovlp`` and ``Prec`` are not), so
        re-scoring reports the value of the sequence that is actually being sent to the generator, while
        leaving the pre-permutation value attached to it would report a number no sequence in the result
        ever had. Content is what the two share, and content is what §4.5's claim is about.

        Sorting happens after the shuffle because re-scoring can reorder the ``N`` trajectories — the
        returned ``Γ_q`` must be best-first in the scores it reports, or the "primary trajectory" the
        generator is told to reason from would not be the best one.
        """
        pairs: list[tuple[tuple[int, ...], TrajectoryScore]] = []
        for step, traj in enumerate(positions):
            if self.cfg.ablation.shuffle_trajectory:
                rng = random.Random(self.cfg.seed + step)
                order = list(traj)
                rng.shuffle(order)
                shuffled = tuple(order)
                pairs.append((shuffled, scorer.score(shuffled)))
            else:
                pairs.append((traj, term_scores[step]))

        weights = scorer.weights
        pairs.sort(
            key=lambda item: (
                -item[1].total(weights.lambda_, weights.mu, weights.nu, weights.rho),
                item[0],
            )
        )
        return pairs
