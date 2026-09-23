r"""§A.7 Algorithm 1 — order-aware beam search.

The paper reproduces the algorithm in full, so the implementation is a transcription with the two
under-specified steps made explicit:

```
Require: Query q, candidates C, model Pθ, width B, length L, weights λ, µ, ν, ρ
1: Precompute Rel(e, q) and log Pθ(ej | ei) for all e, ei, ej ∈ C      -> CandidateScores
2: B ← top-2B singletons by Rel(e, q)
3: for ℓ = 2, ..., L do
4:   B′ ← ∅
5:   for each β = (e(1), ..., e(ℓ−1)) ∈ B, each e′ ∈ C \ β do
6:     s ← Rel(e′, q) + λ log Pθ(e′ | e(ℓ−1)) + µ · 1[e(ℓ−1) ≺ e′]
7:     s ← s + ν · J(V_e(ℓ−1), V_e′) + ρ · ∆Cov(e′, β)
8:     B′ ← B′ ∪ {(β ⊕ e′, score(β) + s)}
9:   end for
10:  Apply diversity penalty; retain top-B by score
11: end for
12: return Top-N trajectories with score breakdowns
```

Three things the box leaves open, and how each is resolved:

* **Line 2 fixes the initial *set* but not the initial *score*.** The singletons are chosen by ``Rel``, but
  line 8 accumulates onto ``score(β)``, and a trajectory's ``Cov`` (Eq. 13) includes its first element's
  phase. So each singleton starts at ``Rel(e, q)`` **plus** ``Cov([e])``. Choosing by relevance and
  seeding the score with the first step's full delta is the only way both lines can be satisfied and the
  final accumulated score equal Eq. 5's value for the returned trajectory.

* **Line 6's ``µ · 1[·]`` is not Eq. 5's ``µ · Prec(γ)``.** Eq. 5 weights the *fraction*; the box weights a
  0/1 indicator. ``retrieval.precedence_mode`` selects between them and defaults to the equation. See
  :mod:`okh_rag.retrieval.scoring`.

* **Line 10's diversity penalty has no formula.** The project's recorded decision (``ambiguity_audit.md``)
  is ``total −= penalty · (#beams already retained sharing this beam's head)``, where the head is
  ``e_(1)``. That formula is *only* meaningful as a greedy selection: if the candidates were first sorted
  by their unpenalised score, a penalty applied afterwards could never change which ``B`` survive, and
  "apply the penalty; retain top-B" would be a no-op instruction. :func:`_retain` therefore picks
  repeatedly by current effective score, which is the MMR-shaped behaviour the word "diversity" implies.

Line 5's ``C \ β`` is implemented literally: a trajectory may not repeat a hyperedge. That is a real
constraint on the search — it is what makes ``L = 8`` mean eight distinct pieces of evidence rather than
eight visits to the most relevant one — and the Viterbi variant does *not* impose it (see
:mod:`okh_rag.retrieval.viterbi` for why the paper's stated complexity forces that asymmetry).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

from okh_rag.config import RetrievalConfig
from okh_rag.retrieval.scoring import TrajectoryScorer
from okh_rag.schema import TrajectoryScore, sum_scores

__all__ = [
    "BeamEntry",
    "BeamSearchResult",
    "order_aware_beam_search",
]


@dataclass(frozen=True)
class BeamEntry:
    """One partial or complete trajectory inside the beam.

    Attributes:
        positions: Pool positions in trajectory order. A tuple, so it is hashable and totally ordered —
            both are needed for the deterministic tie-breaks the search relies on.
        score: The accumulated Eq. 5 components, unweighted. Kept separate from :attr:`total` so a
            returned trajectory's breakdown is the objective's own breakdown rather than the search's
            bookkeeping.
        total: The weighted score the search compares, **including** every diversity penalty already
            charged. Equals ``score.total(λ, µ, ν, ρ) − penalty``.
        penalty: Cumulative diversity deduction. Non-zero only after step 2, and reported so the
            difference between a trajectory's objective value and its search value is never inferred
            from a discrepancy. Line 10 runs once per step, so a beam whose head keeps colliding is
            charged repeatedly and this field is the **sum** of those charges — it is not the deduction
            of the last retention alone. The invariant ``total == score.total(...) − penalty`` is
            asserted in ``tests/test_retrieval.py::TestBeamSearch``; it is the reason the field exists
            rather than the caller recomputing the gap.
    """

    positions: tuple[int, ...]
    score: TrajectoryScore
    total: float
    penalty: float = 0.0

    @property
    def head(self) -> int:
        """``e_(1)`` — the pool position of the first step, which is what the diversity penalty keys on."""
        return self.positions[0]

    def __len__(self) -> int:
        return len(self.positions)


@dataclass(frozen=True)
class BeamSearchResult:
    r"""Algorithm 1 line 12 — the trajectories, plus what the search looked like from inside.

    Attributes:
        entries: The returned trajectories, best first, at most ``num_trajectories`` of them.
        beam_size_per_step: ``|B|`` after each step's retention, index 0 being the initial singleton
            beam. Diagnostics only, but the one place a silently starving beam would show: §A.7 line 10
            retains ``B`` every iteration, so a dip below ``beam_width`` can only mean the candidate set
            ran out of unvisited hyperedges.
        considered_per_step: How many candidate extensions each step enumerated, i.e. ``|B| · |C \ β|``.
    """

    entries: tuple[BeamEntry, ...]
    beam_size_per_step: tuple[int, ...] = ()
    considered_per_step: tuple[int, ...] = ()

    def __len__(self) -> int:
        return len(self.entries)

    def best(self) -> BeamEntry | None:
        """The highest-scoring trajectory, or ``None`` when the pool was empty."""
        return self.entries[0] if self.entries else None


def _retain(candidates: Sequence[BeamEntry], width: int, penalty: float) -> list[BeamEntry]:
    """Algorithm 1 line 10 — "Apply diversity penalty; retain top-B by score".

    # [UNSPECIFIED] §A.7 gives the penalty no formula; ``retrieval.diversity_penalty`` supplies the
    # coefficient and ``ambiguity_audit.md`` records the deduction as
    # ``penalty · (#beams already retained sharing this head)``. Implemented as a greedy selection so the
    # penalty can influence retention at all — see the module docstring. With ``penalty == 0`` this is
    # exactly ``sorted(...)[:width]``.
    #
    # Determinism: ties on effective score fall back to the positions tuple, which is a total order over
    # distinct trajectories. Two runs on the same pool therefore retain the same beams in the same order.

    Args:
        candidates: ``B′`` of line 4, already carrying their accumulated scores.
        width: ``B``.
        penalty: The per-shared-head deduction.

    Returns:
        Up to ``width`` entries, in the order they were selected (best first at ``penalty == 0``).
    """
    if width <= 0 or not candidates:
        return []
    if penalty == 0.0:
        return sorted(candidates, key=lambda e: (-e.total, e.positions))[:width]

    remaining = list(candidates)
    retained: list[BeamEntry] = []
    head_counts: dict[int, int] = {}

    while remaining and len(retained) < width:
        best_index = 0
        best_key: tuple[float, tuple[int, ...]] | None = None
        best_deduction = 0.0
        for index, candidate in enumerate(remaining):
            deduction = penalty * head_counts.get(candidate.head, 0)
            key = (-(candidate.total - deduction), candidate.positions)
            if best_key is None or key < best_key:
                best_key = key
                best_index = index
                best_deduction = deduction
        chosen = remaining.pop(best_index)
        retained.append(replace(chosen, total=chosen.total - best_deduction, penalty=chosen.penalty + best_deduction))
        head_counts[chosen.head] = head_counts.get(chosen.head, 0) + 1

    return retained


def order_aware_beam_search(
    scorer: TrajectoryScorer,
    *,
    cfg: RetrievalConfig,
    length: int | None = None,
) -> BeamSearchResult:
    """Run Algorithm 1 over a precomputed candidate pool.

    Args:
        scorer: The Eq. 5 evaluator for this pool, already carrying the resolved weights.
        cfg: The ``retrieval`` section — ``beam_width`` (B), ``beam_length`` (L),
            ``num_trajectories`` (N), ``beam_initial_multiplier`` and ``diversity_penalty``.
        length: Override for ``cfg.beam_length``. ``[UNSPECIFIED]`` §A.7 note: L is a beam *length*
            while §3.1's ``ℓ ∈ {1..L}`` is a sequence index, and CyPortQA has up to 10 horizons, so the
            two are independent — a caller wanting a trajectory that spans all horizons of one scenario
            passes ``length=<that count>``. ``L`` is additionally clamped to ``|C|``, since a trajectory
            cannot hold more distinct hyperedges than the pool has.

    Returns:
        A :class:`BeamSearchResult`. An empty pool yields an empty result rather than an error: a query
        whose scenario has no extracted hyperedges is a data condition, not a programming fault, and the
        evaluator counts it as a retrieval miss.
    """
    pool_size = len(scorer.scores)
    width = max(1, cfg.beam_width)
    steps = cfg.beam_length if length is None else length
    steps = max(1, min(steps, pool_size))
    weights = scorer.weights
    penalty = cfg.diversity_penalty

    if pool_size == 0:
        return BeamSearchResult(entries=())

    def weighted(score: TrajectoryScore) -> float:
        return score.total(weights.lambda_, weights.mu, weights.nu, weights.rho)

    # --- line 2: top-2B singletons by Rel(e, q) --------------------------------
    seed_count = min(pool_size, max(1, cfg.beam_initial_multiplier) * width)
    by_relevance = sorted(range(pool_size), key=lambda p: (-float(scorer.scores.relevance[p]), p))
    beam: list[BeamEntry] = []
    for position in by_relevance[:seed_count]:
        score = scorer.step_delta((), position)
        beam.append(BeamEntry(positions=(position,), score=score, total=weighted(score)))

    widths = [len(beam)]
    considered: list[int] = []

    # --- lines 3-11: extend, score, penalise, retain ---------------------------
    for _ in range(2, steps + 1):
        expanded: list[BeamEntry] = []
        for entry in beam:
            visited = set(entry.positions)
            for position in range(pool_size):
                if position in visited:
                    continue
                delta = scorer.step_delta(entry.positions, position)
                expanded.append(
                    BeamEntry(
                        positions=(*entry.positions, position),
                        score=sum_scores((entry.score, delta)),
                        # ``penalty`` is carried forward, not reset. Line 10 charges a deduction at
                        # *every* retention, and a deducted score stays deducted because the next
                        # step extends this entry's ``total``. Dropping the field here would keep the
                        # deduction inside ``total`` while reporting ``penalty == 0``, breaking the
                        # documented invariant ``total == score.total(...) - penalty`` and making a
                        # genuinely penalised trajectory indistinguishable from a clean one.
                        total=entry.total + weighted(delta),
                        penalty=entry.penalty,
                    )
                )
        considered.append(len(expanded))
        beam = _retain(expanded, width, penalty)
        widths.append(len(beam))

    # --- line 12: top-N trajectories with score breakdowns ---------------------
    ranked = sorted(beam, key=lambda e: (-e.total, e.positions))
    return BeamSearchResult(
        entries=tuple(ranked[: max(1, cfg.num_trajectories)]),
        beam_size_per_step=tuple(widths),
        considered_per_step=tuple(considered),
    )
