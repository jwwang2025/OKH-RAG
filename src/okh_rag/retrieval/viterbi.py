"""§A.7 — Viterbi dynamic programming, the exact optimiser for the two-term objective.

§A.7 gives the recurrence and its complexity:

    dp[0][j] = Rel(e_j, q)                                                          (14)
    dp[ℓ][j] = max_i ( dp[ℓ−1][i] + Rel(e_j, q) + λ log P_θ(e_j | e_i) )            (15)

    "with backpointers for trajectory recovery. Complexity: O(L · |C|²). Viterbi optimizes the two-term
    objective exactly but omits auxiliary terms; it serves primarily as a quality ceiling for beam
    search."

Two consequences of taking those three claims together, both of which shape the implementation:

* **Repeats are permitted.** ``max_i`` ranges over the whole pool, including ``i == j``. Enforcing the
  beam search's ``C \\ β`` would require the subset of visited hyperedges in the DP state, which is
  Held–Karp and costs ``O(L · |C|² · 2^|C|)`` — a different complexity class from the one §A.7 states.
  So the stated complexity *is* evidence for the unconstrained reading, and the code follows it. This
  makes Viterbi's ceiling a slightly different object from Algorithm 1's result, which is exactly what
  "quality ceiling for beam search" claims: a ceiling can be loose.

* **Only two terms.** No ``µ``, ``ν`` or ``ρ``. The point is to isolate what the transition model alone
  can find, so mixing in the structural terms would remove the baseline it exists to provide. A caller
  wanting the full objective on Viterbi's path can ask :class:`~okh_rag.retrieval.scoring.TrajectoryScorer`
  for it — that is what :attr:`ViterbiResult.score` carries.

The recurrence is evaluated as a dense ``(|C|, |C|)`` numpy update rather than a Python loop over pairs:
``cand[i, j] = dp[ℓ−1][i] + λ · trans[i, j]`` is one broadcast add, ``argmax`` along the source axis gives
both the backpointer and, via the max, the value. At ``|C| = 150`` and ``L = 8`` that is eight
150×150 operations per query, which is why the paper can call it "for smaller candidate sets" and still
have it be the reference implementation.

Ties are broken by ``argmax``'s lowest-index rule, and the pool is relevance-ordered (§A.7 stages 1-2), so
a tie resolves towards the more relevant predecessor — deterministically, since :meth:`CandidatePool`
breaks relevance ties on the hyperedge id.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from okh_rag.config import RetrievalConfig
from okh_rag.retrieval.scoring import TrajectoryScorer
from okh_rag.schema import TrajectoryScore

__all__ = [
    "ViterbiResult",
    "viterbi_search",
]


@dataclass(frozen=True)
class ViterbiResult:
    """The optimiser's output: one path, its two-term value, and its full Eq. 5 breakdown.

    Attributes:
        positions: Pool positions along the recovered path.
        objective: ``Σ Rel + λ · Σ log P_θ`` — the two-term quantity Eq. 15 actually maximises. This, not
            :attr:`score`, is the ceiling that can be compared against beam search run with ``µ = ν = ρ =
            0``.
        score: All five Eq. 5 terms for the same path, so the auxiliary terms can be read off the exact
            solution without a second run.
        length: ``L`` after clamping to the pool size.
    """

    positions: tuple[int, ...]
    objective: float
    score: TrajectoryScore
    length: int

    def __len__(self) -> int:
        return len(self.positions)


def viterbi_search(
    scorer: TrajectoryScorer,
    *,
    cfg: RetrievalConfig,
    length: int | None = None,
) -> ViterbiResult:
    """Eq. 14–15 over a precomputed candidate pool.

    Args:
        scorer: The Eq. 5 evaluator for this pool. Its ``weights.lambda_`` is λ; ``weights.mu``,
        ``.nu`` and ``.rho`` are ignored by the recurrence and only used to fill
        :attr:`ViterbiResult.score`.
        cfg: The ``retrieval`` section, for ``beam_length`` when ``length`` is not given.
        length: ``L``. See :func:`~okh_rag.retrieval.beam.order_aware_beam_search` for why L is
            independent of the horizon count; it is clamped to ``|C|`` here because a path of L distinct
            steps needs at least L candidates.

    Returns:
        A :class:`ViterbiResult`. An empty pool returns an empty path with ``objective = 0.0``.
    """
    n = len(scorer.scores)
    if n == 0:
        return ViterbiResult(positions=(), objective=0.0, score=TrajectoryScore(), length=0)

    steps = cfg.beam_length if length is None else length
    steps = max(1, min(steps, n))

    relevance = np.asarray(scorer.scores.relevance, dtype=np.float64)
    transitions = np.asarray(scorer.scores.transitions, dtype=np.float64)
    lam = scorer.weights.lambda_

    # ``λ · trans`` with a zero λ must be the zero matrix, not ``0 · (-inf)``. ``-inf`` is reachable in
    # ``transitions``: the learned table is a log-softmax over the pool, and a log-prob underflows to
    # ``-inf`` once a softmax mass rounds to zero in float64 — rare, but exactly the kind of rare that
    # happens once in a full Table 2 sweep and then has to be explained. The `−λ (order coherence)`
    # ablation would otherwise turn the whole DP into NaN, and NaN loses every ``argmax`` comparison
    # silently, returning an arbitrary path while reporting a perfectly plausible objective.
    weighted = transitions * lam if lam else np.zeros_like(transitions)

    dp = np.empty((steps, n), dtype=np.float64)
    backpointers = np.zeros((steps, n), dtype=np.int64)
    dp[0] = relevance

    for step in range(1, steps):
        # cand[i, j] = dp[step-1][i] + λ log Pθ(e_j | e_i), i.e. Eq. 15's bracket.
        cand = dp[step - 1][:, None] + weighted
        backpointers[step] = np.argmax(cand, axis=0)
        dp[step] = cand[backpointers[step], np.arange(n)] + relevance

    last = int(np.argmax(dp[steps - 1]))
    objective = float(dp[steps - 1, last])

    path = [last]
    for step in range(steps - 1, 0, -1):
        last = int(backpointers[step, last])
        path.append(last)
    path.reverse()

    positions = tuple(path)
    return ViterbiResult(
        positions=positions,
        objective=objective,
        score=scorer.score(positions),
        length=steps,
    )
