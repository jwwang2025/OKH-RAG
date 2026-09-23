"""§3.2 / §A.6–A.7 — order-aware retrieval.

Retrieval here is *trajectory inference*, not ranking. §3.2 puts the difference plainly: existing methods
score hyperedges independently and return a set, while OKH-RAG recovers the ordered sequence
``γ* = argmax_γ`` of Eq. 5. That reframing is what the four modules below implement, one per stage of the
pipeline §A.7 lays out:

* :mod:`~okh_rag.retrieval.candidates` — stage 1 and 2 of the search-space reduction: top-K cosine, then
  group-aware expansion, capped at ``|C| = 150`` with 40% reserved for the query's own group.
* :mod:`~okh_rag.retrieval.scoring` — §A.6's five terms with their ``Σ Rel`` and ``log P_θ``
  precomputations, evaluated both whole and incrementally so the beam search and the reported objective
  cannot drift apart.
* :mod:`~okh_rag.retrieval.beam` — Algorithm 1, order-aware beam search, the primary inference algorithm.
* :mod:`~okh_rag.retrieval.viterbi` — Eq. 14–15, the exact optimiser for the two-term objective, used as
  the quality ceiling.

:class:`~okh_rag.retrieval.OrderAwareRetriever` is the only class most callers need; the rest of the
package is exported because the evaluation and ablation harnesses reason about the intermediate stages
directly (a Table 2 row is a *variant selection*, and the per-term breakdown of a returned trajectory is
the object the component ablations are read from).
"""

from okh_rag.config import (
    GROUP_RESERVE_MODES,
    INFERENCE_MODES,
    OVERLAP_MODES,
    PRECEDENCE_MODES,
    TRANSITION_VARIANTS,
)
from okh_rag.retrieval.beam import BeamEntry, BeamSearchResult, order_aware_beam_search
from okh_rag.retrieval.candidates import CandidatePool, HypergraphIndex, scope_candidates
from okh_rag.retrieval.retriever import (
    OrderAwareRetriever,
    RetrievalResult,
    Retriever,
    resolve_weights,
)
from okh_rag.retrieval.scoring import (
    CandidateScores,
    TrajectoryScorer,
    cosine_relevance,
    heuristic_transition_matrix,
    learned_transition_matrix,
    unreachable_phase_set,
    zero_transition_matrix,
)
from okh_rag.retrieval.viterbi import ViterbiResult, viterbi_search

__all__ = [
    "GROUP_RESERVE_MODES",
    "INFERENCE_MODES",
    "OVERLAP_MODES",
    "PRECEDENCE_MODES",
    "TRANSITION_VARIANTS",
    "BeamEntry",
    "BeamSearchResult",
    "CandidatePool",
    "CandidateScores",
    "HypergraphIndex",
    "OrderAwareRetriever",
    "RetrievalResult",
    "Retriever",
    "TrajectoryScorer",
    "ViterbiResult",
    "cosine_relevance",
    "heuristic_transition_matrix",
    "learned_transition_matrix",
    "order_aware_beam_search",
    "resolve_weights",
    "scope_candidates",
    "unreachable_phase_set",
    "viterbi_search",
    "zero_transition_matrix",
]
