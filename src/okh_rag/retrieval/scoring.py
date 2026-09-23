"""§A.6 — the five scoring terms of Eq. 5, and the precomputations that make them cheap.

§3.2 Eq. 5 defines the retrieval objective as

    γ* = argmax_γ [ Σ_k Rel(e_(k), q)                      relevance
                  + λ Σ_k log P_θ(e_(k+1) | e_(k))         order coherence
                  + µ · Prec(γ)                            precedence consistency  (Eq. 11)
                  + ν · Ovlp(γ)                            entity continuity       (Eq. 12)
                  + ρ · Cov(γ) ]                           phase coverage          (Eq. 13)

and §A.6 opens its definitions with the constraint that determines the whole shape of this module:

    "All terms in Equation 5 are designed for efficient incremental computation during beam search."

So every term is implemented **twice**: once as the whole-trajectory quantity (the definition — used by
Viterbi, by the score-breakdown report, and by anything that verifies a result), and once as a *step
delta* (what Algorithm 1 lines 6-7 accumulate). The two are required to telescope: the sum of a
trajectory's step deltas must equal its whole-trajectory value. Because ``TrajectoryScore.total`` is a
plain weighted sum, linearity guarantees that if the *components* telescope then the weighted totals do
too — and ``tests/test_retrieval.py`` asserts the component identity on random trajectories. A
discrepancy there would mean the beam search optimises a different objective from the one being
reported, which is exactly the class of bug a reproduction must not have.

Three of the five terms are **not** sums over steps, and that is why the incremental form needs stating
rather than assuming:

* ``Prec(γ)`` is a *fraction* over comparable pairs (Eq. 11), so the prefix value is not the sum of the
  per-step values; the delta is the change in the fraction.
* ``Ovlp(γ)`` is a *mean* Jaccard over ``L-1`` pairs (Eq. 12), same situation.
* ``Cov(γ)`` is a *set cardinality* over ``|S|`` (Eq. 13); its delta is ``∆Cov(e', β)`` from Algorithm 1
  line 7, which happens to be exactly the change in the prefix value — the one place where §A.6 and
  Algorithm 1 need no reconciliation.

Where §A.6 and Algorithm 1 do disagree, ``configs/base.yaml`` carries a switch and the default follows
the equation, per this project's standing rule that equations outrank algorithm boxes:

| Term | Eq. 5 (§3.2)             | Algorithm 1 line 6-7                  | Switch                 |
|------|--------------------------|---------------------------------------|------------------------|
| Pre | ``µ · Prec(γ)`` fraction | ``µ · 1[e_(l-1) ≺ e']`` indicator     | ``precedence_mode``    |
| Ovlp | ``ν · Ovlp(γ)`` mean    | ``ν · J(V_last, V_e')`` summed        | ``overlap_mode``       |

The ``overlap_mode="sum"`` variant is worth one sentence because it is the only switch in the project
where the two readings provably cannot change a ranking: ``Σ_k J`` equals ``(L-1) · Ovlp(γ)``, and with
``L`` fixed across a run the two differ by the same constant factor for every trajectory, so only the
*scale* of ν against the other weights moves. Both are implemented so the claim is testable rather than
asserted; the smoke script checks it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from okh_rag.config import (
    OVERLAP_MODES,
    PRECEDENCE_MODES,
    TRANSITION_VARIANTS,
    RetrievalConfig,
    RetrievalWeights,
)
from okh_rag.embeddings import HyperedgeEmbeddings
from okh_rag.precedence import PrecedenceGraph
from okh_rag.schema import Hyperedge, KnowledgeHypergraph, TrajectoryScore
from okh_rag.vocabulary import coverage_phase_of, coverage_phase_reachable

__all__ = [
    "OVERLAP_MODES",
    "PRECEDENCE_MODES",
    "TRANSITION_VARIANTS",
    "CandidateScores",
    "TrajectoryScorer",
    "cosine_relevance",
    "heuristic_transition_matrix",
    "learned_transition_matrix",
    "unreachable_phase_set",
    "zero_transition_matrix",
]

# The three switches that reconcile §3.2 with §A.7 are declared in :mod:`okh_rag.config` and re-exported
# here for the callers that think in terms of scoring. See that module for why the declaration lives there.

# §A.6 — "log Pθ ... computed incrementally from the precomputed |C| × |C| matrix". **No** variant here
# produces ``-inf``: the learned table is a log-softmax over the pool, the heuristic table charges a
# finite ``-1.0`` per ≺-violation (see :func:`heuristic_transition_matrix` for why a hard ``-inf`` was
# rejected), and the order-free table is all zeros. ``TrajectoryScore.total`` still guards the
# ``0 · -inf = NaN`` case, because ``λ`` is not the only weight that can be driven to zero by the §4.5
# ``drop_*`` ablations and a caller may build a score by hand.


# ---------------------------------------------------------------------------
# §A.6 Relevance — Rel(e, q) = cos(h_e, q)
# ---------------------------------------------------------------------------


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation, mapping zero-norm rows to the zero row instead of NaN."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0.0, norms, 1.0)


def cosine_relevance(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """§A.6 — ``Rel(e, q) = cos(h_e, q)`` for every row of ``matrix``.

    §A.6 — "cosine similarity in R^1536 between hyperedge and query embeddings. Precomputed once per
    query for all candidates."

    Args:
        query: The query embedding, ``(d,)``.
        matrix: Hyperedge embeddings, ``(n, d)``.

    Returns:
        An ``(n,)`` float64 array. Computed in float64 even though the stored embeddings are float32:
        the relevance term is added to log-probabilities of order ``-log|C|``, and accumulating a few
        dozen float32 cosines loses more precision than the memory saved is worth.

    Raises:
        ValueError: If the shapes disagree. A silent broadcast here would score every candidate against
            a truncated query and still return a plausible-looking ranking.

    A zero-norm row or query yields 0.0 for that entry rather than NaN. The OpenAI endpoint never
    returns one, but synthetic corpora in tests do, and a single NaN would propagate through the beam
    comparison into an arbitrary winner with no error raised anywhere.
    """
    q = np.asarray(query, dtype=np.float64).reshape(-1)
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"hyperedge embedding matrix must be 2-d, got shape {m.shape}")
    if m.shape[1] != q.shape[0]:
        raise ValueError(f"query has {q.shape[0]} dimensions but matrix has {m.shape[1]}")

    query_norm = float(np.linalg.norm(q))
    if query_norm == 0.0:
        return np.zeros(m.shape[0], dtype=np.float64)
    return _unit_rows(m) @ (q / query_norm)


# ---------------------------------------------------------------------------
# §A.6 Order coherence — the |C| × |C| transition table
# ---------------------------------------------------------------------------


def learned_transition_matrix(
    model: object,
    embeddings: HyperedgeEmbeddings,
    hyperedge_ids: Sequence[str],
) -> np.ndarray:
    """Eq. 3's log-probabilities over the candidate pool, as a dense ``(|C|, |C|)`` table.

    §A.6 restricts ``Trans`` to the candidate set: "cumulative log-transition probability, computed
    incrementally from the precomputed |C| × |C| matrix". So the denominator is Eq. 3's denominator
    **restricted to C**, not the sampled-softmax one of §A.4. ``ambiguity_audit.md`` records that choice
    and its reason: sampling at retrieval time would make the Table 2 rows depend on the RNG.

    Args:
        model: A :class:`~okh_rag.order_model.TransitionScorer`. Typed loosely on purpose — the only
            method used is ``log_prob_full``, and the smoke test supplies a stub that never builds a
            ``torch.nn.Module``.
        embeddings: The corpus embeddings, indexed by hyperedge id.
        hyperedge_ids: The pool ``C``, in the order the table's rows and columns will follow.

    Returns:
        A ``(|C|, |C|)`` float64 array with ``[i, j] = log P_θ(e_j | e_i)``.

    Raises:
        ImportError: If torch is unavailable. The learned variant is the only one that needs it, so the
            import is deferred to here — the heuristic and order-free variants run without torch.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency of the project
        raise ImportError(
            "the learned transition variant needs torch; use transition_variant='heuristic' or 'none' "
            "to run without it"
        ) from exc

    rows = embeddings.rows(hyperedge_ids)
    with torch.no_grad():
        source = torch.from_numpy(np.ascontiguousarray(rows))
        log_probs = model.log_prob_full(source, source)  # type: ignore[attr-defined]
    if tuple(log_probs.shape) != (len(hyperedge_ids), len(hyperedge_ids)):
        raise ValueError(
            f"transition model returned shape {tuple(log_probs.shape)} for a pool of "
            f"{len(hyperedge_ids)}; expected a square table"
        )
    return log_probs.numpy().astype(np.float64, copy=True)


def heuristic_transition_matrix(
    precedence: PrecedenceGraph,
    hyperedge_ids: Sequence[str],
) -> np.ndarray:
    """§A.3's fixed rules in place of the learned model — Table 2's "heuristic order" row.

    §4.5 — "Comparing no order, heuristic order, and learned order reveals a consistent hierarchy:
    performance improves as stronger forms of order modeling are introduced. Heuristic ordering provides
    modest gains over order-free retrieval, while the learned transition model performs best, capturing
    ordering patterns beyond fixed rules."

    The rules live in :mod:`okh_rag.precedence`, so this needs no embeddings — which is why it is here
    and not in :mod:`okh_rag.order_model` (see that module's scope note).

    # [UNSPECIFIED] §4.5 names the row but never defines the score. "Heuristic order" is rendered as a
    # rule-consistency reward in the same units as a log-probability:
    #
    #       score(e_i, e_j) = 0.0   if e_i ≺ e_j          (a §A.3-legal step costs nothing)
    #                       = -1.0  otherwise             (a step against ≺ costs one nat)
    #
    # The units matter: ``λ`` is shared with the learned variant, so a score in nats lets λ transfer
    # between the two rows without rescaling. The step from the trained model is not on a comparable
    # absolute scale — Eq. 3's log-probabilities range over ``[-log|C|, 0]`` and the model's confident
    # transitions drop well below ``-1`` — so the heuristic is deliberately the *weaker* constraint,
    # which is the ordering Table 2 reports.
    # Alternatives: a proper conditional distribution, ``P(e_j|e_i) ∝ 1 + 1[e_i ≺ e_j]`` (never
    # forbidden, but adds a second arbitrary constant and a source-dependent denominator); or a hard
    # ``-inf`` for every non-≺ step, which kills the beam whenever the last hyperedge's successors are
    # all already used.
    """

    n = len(hyperedge_ids)
    matrix = np.full((n, n), -1.0, dtype=np.float64)
    for i, source in enumerate(hyperedge_ids):
        for j, target in enumerate(hyperedge_ids):
            if precedence.precedes(source, target):
                matrix[i, j] = 0.0
    return matrix


def zero_transition_matrix(n: int) -> np.ndarray:
    """The order-free variant — every transition scores 0.

    Used with all four weights at zero (Table 2 "No order"). §3.2 makes the equivalence explicit:
    "The hyperparameters λ, µ, ν, ρ ≥ 0 are modular: setting all to zero recovers standard top-k
    retrieval." A zero table rather than ``None`` keeps the beam search's inner loop branch-free, so the
    ablation cannot accidentally run a different code path from the full model.
    """
    return np.zeros((n, n), dtype=np.float64)


# ---------------------------------------------------------------------------
# §A.6 line 1 — the precomputations, bundled
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateScores:
    """Algorithm 1 line 1 — "Precompute Rel(e, q) and log Pθ(ej | ei) for all e, ei, ej ∈ C".

    Attributes:
        hyperedge_ids: The pool ``C``. Row *i* of both arrays is this hyperedge.
        relevance: ``(|C|,)`` — ``Rel(e, q)``.
        transitions: ``(|C|, |C|)`` — ``[i, j]`` is the order-coherence score for appending ``j``
            after ``i``: ``log P_θ`` for the learned variant, a rule reward for the heuristic one, and
            zero for the order-free one.
        variant: Which of :data:`TRANSITION_VARIANTS` produced ``transitions``. Carried so a result can
            report which Table 2 row it belongs to without the caller threading the flag alongside.
    """

    hyperedge_ids: tuple[str, ...]
    relevance: np.ndarray
    transitions: np.ndarray
    variant: str = "learned"

    def __post_init__(self) -> None:
        n = len(self.hyperedge_ids)
        if self.relevance.shape != (n,):
            raise ValueError(f"relevance has shape {self.relevance.shape}; expected ({n},)")
        if self.transitions.shape != (n, n):
            raise ValueError(f"transitions has shape {self.transitions.shape}; expected ({n}, {n})")
        if self.variant not in TRANSITION_VARIANTS:
            raise ValueError(f"unknown transition variant {self.variant!r}; expected {TRANSITION_VARIANTS}")

    def __len__(self) -> int:
        return len(self.hyperedge_ids)

    @property
    def index(self) -> dict[str, int]:
        """``hyperedge id -> pool position``."""
        return {hid: row for row, hid in enumerate(self.hyperedge_ids)}


# ---------------------------------------------------------------------------
# §A.6 / Algorithm 1 — the scorer
# ---------------------------------------------------------------------------


class TrajectoryScorer:
    """Eq. 5 over one candidate pool, evaluated both whole and incrementally.

    A trajectory is represented as a tuple of **pool positions** (integers), not hyperedge ids. Beam
    search compares and extends trajectories tens of thousands of times on a 150-element pool; integer
    tuples hash and compare far faster than id tuples, and every array lookup becomes an index rather
    than a dict hit. The ids are recovered only when a :class:`~okh_rag.schema.Trajectory` is built at
    the end.

    Attributes:
        scores: The §A.6 precomputations.
        weights: The four Eq. 5 weights, already reduced by the §4.5 ``drop_*`` ablations (and zeroed
            entirely for the order-free variant — the caller resolves that, see
            :meth:`okh_rag.retrieval.OrderAwareRetriever._resolve_weights`).
        phase_set: ``S`` of Eq. 13, as a frozenset.
    """

    def __init__(
        self,
        *,
        hypergraph: KnowledgeHypergraph,
        precedence: PrecedenceGraph,
        scores: CandidateScores,
        cfg: RetrievalConfig,
        weights: RetrievalWeights | None = None,
    ) -> None:
        """Precompute the per-hyperedge quantities the incremental terms need.

        Two per-position lookups dominate the cost of an incremental step — the entity set for Eq. 12
        and the coverage phase for Eq. 13 — and both are pure functions of the hyperedge, so they are
        materialised once here rather than recomputed inside the ``O(B·|C|·L)`` beam loop.

        Missing ids are not an error: a pool built from one hypergraph can legitimately be scored against
        a precedence graph that omits a group, and a hyperedge absent from ``hypergraph`` contributes an
        empty entity set and no coverage phase. That is the behaviour §A.6's terms already have for a
        hyperedge with no comparable neighbours.
        """
        self.hypergraph = hypergraph
        self.precedence = precedence
        self.scores = scores
        self.weights = weights if weights is not None else cfg.weights
        self.phase_set: frozenset[str] = frozenset(cfg.phase_set)
        self.precedence_mode = cfg.precedence_mode
        self.overlap_mode = cfg.overlap_mode

        if self.precedence_mode not in PRECEDENCE_MODES:
            raise ValueError(f"unknown precedence_mode {self.precedence_mode!r}; expected {PRECEDENCE_MODES}")
        if self.overlap_mode not in OVERLAP_MODES:
            raise ValueError(f"unknown overlap_mode {self.overlap_mode!r}; expected {OVERLAP_MODES}")

        self.entity_sets: tuple[frozenset[str], ...] = tuple(
            hypergraph.entity_id_set(hypergraph.hyperedges[hid]) if hid in hypergraph.hyperedges else frozenset()
            for hid in scores.hyperedge_ids
        )
        self.coverage_phases: tuple[str | None, ...] = tuple(
            coverage_phase_of(hypergraph.hyperedges[hid].relation) if hid in hypergraph.hyperedges else None
            for hid in scores.hyperedge_ids
        )

    # -- access ------------------------------------------------------------

    def edge(self, position: int) -> Hyperedge | None:
        """The hyperedge at a pool position, or ``None`` if the pool id is not in the hypergraph."""
        return self.hypergraph.hyperedges.get(self.scores.hyperedge_ids[position])

    def ids(self, positions: Iterable[int]) -> tuple[str, ...]:
        """Pool positions back to hyperedge ids."""
        return tuple(self.scores.hyperedge_ids[p] for p in positions)

    # -- §A.6 Eq. 12 — entity continuity ------------------------------------

    def jaccard(self, i: int, j: int) -> float:
        """``J(V_ei, V_ej) = |∩| / |∪|`` — the per-pair quantity Eq. 12 averages and Algorithm 1 sums."""
        a, b = self.entity_sets[i], self.entity_sets[j]
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    # -- §A.6 Eq. 11 — precedence consistency -------------------------------

    def precedence_pair_counted(self, i: int, j: int) -> bool:
        """Whether the ordered pair counts towards Eq. 11's *denominator*, i.e. is comparable.

        §A.6 Eq. 11 — "Fraction of ordered consecutive pairs consistent with ≺; unrelated pairs are
        excluded from the denominator." The exclusion is not cosmetic: §A.3's order is a *partial* order
        ("accommodating concurrent interactions"), so two hyperedges in unrelated phases at the same
        horizon are genuinely incomparable, and counting them as inconsistent would penalise a
        legitimate trajectory purely for spanning the graph's width.
        """
        return self.precedence.related(self.scores.hyperedge_ids[i], self.scores.hyperedge_ids[j])

    def precedence_pair_consistent(self, i: int, j: int) -> bool:
        """Whether ``e_i ≺ e_j`` — the numerator of Eq. 11 and the indicator of Algorithm 1 line 6."""
        return self.precedence.precedes(self.scores.hyperedge_ids[i], self.scores.hyperedge_ids[j])

    # -- whole-trajectory term values ---------------------------------------

    def relevance_sum(self, positions: Sequence[int]) -> float:
        """``Σ_k Rel(e_(k), q)``."""
        if not positions:
            return 0.0
        return float(self.scores.relevance[list(positions)].sum())

    def order_coherence_sum(self, positions: Sequence[int]) -> float:
        """``Σ_k log P_θ(e_(k+1) | e_(k))``. Returns 0.0 for a trajectory of fewer than two steps."""
        total = 0.0
        for a, b in zip(positions, positions[1:]):
            total += float(self.scores.transitions[a, b])
        return total

    def precedence_consistency(self, positions: Sequence[int]) -> float:
        """§A.6 Eq. 11 — the term as it enters Eq. 5.

        # [UNSPECIFIED] Eq. 11 leaves the denominator-zero case open: a trajectory whose consecutive
        # pairs are *all* incomparable has no fraction. It is reported as 0.0. The term is a non-negative
        # reward in [0, 1], so its neutral element is 0 — a trajectory may not collect credit for
        # precedence it never exercised. Alternatives: 1.0 (treat "nothing violated" as perfect, which
        # hands a free reward to any trajectory that ignores the order entirely); or skip the pair and
        # exclude the term for that trajectory, which makes the reported breakdown depend on which other
        # terms are active.
        Returns:
            The Eq. 11 fraction when ``precedence_mode`` is ``"fraction"``, or the raw count of
            ≺-consistent consecutive pairs when it is ``"indicator"`` (Algorithm 1 line 6). Two
            different quantities share one slot deliberately: both are what the two readings of the paper
            accumulate, and §A.6's own definition is the fraction.
        """
        if self.precedence_mode == "indicator":
            return float(sum(1 for a, b in zip(positions, positions[1:]) if self.precedence_pair_consistent(a, b)))

        consistent = 0
        comparable = 0
        for a, b in zip(positions, positions[1:]):
            if self.precedence_pair_consistent(a, b):
                consistent += 1
                comparable += 1
            elif self.precedence_pair_counted(a, b):
                comparable += 1
        return consistent / comparable if comparable else 0.0

    def entity_continuity(self, positions: Sequence[int]) -> float:
        """§A.6 Eq. 12 — mean Jaccard between consecutive steps, or the sum under ``overlap_mode="sum"``.

        A one-step trajectory has no consecutive pair and so no mean; 0.0 is returned, matching the
        empty-sum convention rather than dividing by the ``L-1 = 0`` denominator.
        """
        if len(positions) < 2:
            return 0.0
        pairs = list(zip(positions, positions[1:]))
        if self.overlap_mode == "sum":
            return float(sum(self.jaccard(a, b) for a, b in pairs))
        return sum(self.jaccard(a, b) for a, b in pairs) / len(pairs)

    def phase_coverage(self, positions: Sequence[int]) -> float:
        """§A.6 Eq. 13 — ``|{φ(e_k)} ∩ S| / |S|``.

        ``φ`` is :func:`okh_rag.vocabulary.coverage_phase_of`, which returns ``None`` for the structural
        families (cyclone state, track, timing, probability, threshold) that §A.6 never lists in ``S``. A
        ``None`` is dropped rather than counted, so a trajectory of purely structural hyperedges scores 0
        on coverage — which is the intent of a term §3.2 introduces to prevent "narrow evidence
        concentration" across the *reasoning* stages.
        """
        if not self.phase_set:
            return 0.0
        seen = {self.coverage_phases[p] for p in positions}
        seen.discard(None)
        return len(seen & self.phase_set) / len(self.phase_set)

    # -- the two views ------------------------------------------------------

    def score(self, positions: Sequence[int]) -> TrajectoryScore:
        """All five Eq. 5 term values for a whole trajectory.

        This is the definitional view: the object Viterbi optimises against, the object the evaluator
        reports, and the reference the incremental view is tested for telescoping against.
        """
        return TrajectoryScore(
            relevance=self.relevance_sum(positions),
            order_coherence=self.order_coherence_sum(positions),
            precedence_consistency=self.precedence_consistency(positions),
            entity_continuity=self.entity_continuity(positions),
            phase_coverage=self.phase_coverage(positions),
        )

    def step_delta(self, prefix: Sequence[int], next_position: int) -> TrajectoryScore:
        """Algorithm 1 lines 6-7 — the increment contributed by appending ``next_position``.

        For the three non-additive terms the increment is defined as *the change in the term's value for
        the prefix*, which is what makes ``Σ_k step_delta(prefix_k, e_(k+1)) == score(full)`` hold for the
        fraction, the mean and the set cardinality alike. Algorithm 1 writes the overlap increment as the
        raw Jaccard, so ``s`` accumulates ``Σ J`` rather than ``ν · mean``; that difference is exactly
        ``overlap_mode``, and under ``"mean"`` the second step's delta still equals ``J`` because a
        one-step prefix has no pair to average over.

        Args:
            prefix: Pool positions already chosen, in order. May be empty, in which case the delta is
                the first step's contribution.
            next_position: The position being appended. Not validated against ``prefix``: Algorithm 1
                line 5 is what excludes repeats, and duplicating that test here would hide a bug in it.

        Returns:
            An unweighted :class:`~okh_rag.schema.TrajectoryScore`. Weights are applied by
            :meth:`~okh_rag.schema.TrajectoryScore.total`, so beam search compares
            ``delta.total(λ, µ, ν, ρ)``.
        """
        prefix = list(prefix)
        relevance = float(self.scores.relevance[next_position])

        if not prefix:
            # The first step: no transition, no pair, no set-growth beyond the single phase. Coverage is
            # included so that the accumulated total equals ``score(full)`` — the singleton's own phase is
            # a member of the trajectory's phase set, and omitting it would under-count Cov by one slot
            # for every trajectory in a run.
            return TrajectoryScore(
                relevance=relevance,
                phase_coverage=self.phase_coverage([next_position]),
            )

        last = prefix[-1]
        transition = float(self.scores.transitions[last, next_position])

        if self.precedence_mode == "indicator":
            precedence = 1.0 if self.precedence_pair_consistent(last, next_position) else 0.0
        else:
            precedence = self.precedence_consistency([*prefix, next_position]) - self.precedence_consistency(prefix)

        if self.overlap_mode == "sum":
            overlap = self.jaccard(last, next_position)
        else:
            overlap = self.entity_continuity([*prefix, next_position]) - self.entity_continuity(prefix)

        coverage = self.phase_coverage([*prefix, next_position]) - self.phase_coverage(prefix)

        return TrajectoryScore(
            relevance=relevance,
            order_coherence=transition,
            precedence_consistency=precedence,
            entity_continuity=overlap,
            phase_coverage=coverage,
        )

    def total(self, positions: Sequence[int]) -> float:
        """Eq. 5 evaluated on a whole trajectory."""
        return self.score(positions).total(
            self.weights.lambda_, self.weights.mu, self.weights.nu, self.weights.rho
        )


def unreachable_phase_set(phase_set: Iterable[str]) -> tuple[str, ...]:
    """Members of ``S`` that no relation can ever map onto, sorted.

    §A.6 Eq. 13 divides by ``|S|``, so an unreachable member silently caps ``Cov`` below 1 for every
    trajectory in the run and quietly deflates the ρ term. :func:`okh_rag.config._validate` calls this to
    fail fast instead — the config file is where a wrong phase set would be introduced, and a run that
    silently loses 1/6 of its ρ contribution is worse than one that refuses to start.
    """
    return tuple(sorted(phase for phase in phase_set if not coverage_phase_reachable(phase)))
