"""§A.7 — candidate scoping, the two-stage reduction of the search space.

§A.7 states the whole algorithm in three sentences:

    "Two stages reduce the search space: (1) top-K (K=80) cosine retrieval from the global index; (2)
    group-aware expansion via group membership and one-hop entity overlap, with 40% of slots reserved for
    the query's own group when known. The pool is capped at |C| = 150."

§3.2 says the same thing in prose ("We scope the candidate set C via top-K cosine retrieval followed by
group-aware expansion (entity overlap and domain group membership), yielding |C|").

Everything between those sentences and code is a decision, and there are exactly two that change *who is
in the pool* rather than how the pool is ordered:

* **Where the reserve is counted from.** 40% of 150 is 60, and stage 1 already contributes 80, so a
  reading in which the reserve is carved *out of* the 150 leaves the expansion stage 10 slots. A reading
  in which it is a *floor* lets the expansion use all the remaining 70 whenever the anchors already supply
  60 own-group hyperedges. The floor reading is the default because it is the only one under which
  "group-aware expansion" does any work beyond the reserve itself, and the alternative is one config key
  away (``retrieval.group_reserve_mode``). §A.7's "40% of slots reserved" is genuinely ambiguous between
  the two, and the difference is not a rounding error.

* **What "group" is.** §A.7 says "the query's own group when known" and never defines the group. §A.1
  defines a knowledge group as the unit over which precedence is built, and this project's group id is
  therefore one scenario (one storm–port–year for CyPortQA). So the reserve draws from the hyperedges
  built from the same scenario document as the question — which for CyPortQA is always known, because a
  question's scenario key is part of its identity.

The second stage's expansion is deliberately *not* re-ranked by anything but cosine. §A.7 gives no
ordering rule for the expansion, and relevance ordering is the only rule already in the paper (stage 1
is explicitly cosine retrieval), so inventing a second ordering would be inventing evidence.

One property here is a requirement rather than a reading: ``len(pool) <= candidate_pool_size``, always.
§A.7 states the cap as a fact about ``|C|`` ("The pool is capped at |C| = 150"), and both stages spend
from it — stage 1's anchors count against it, and the 40% reserve sits *inside* it rather than on top.
Under the shipped numbers (``top_k = 80``, ``|C| = 150``, reserve = 60) the cap has 10 slots of slack, so
it is only reachable by a small pool; but a pool of 6 drawn from a cap of 5 would make every downstream
"|C| = 150" statement in the report a claim the retriever does not honour, which is a silent failure
rather than a crash. :func:`scope_candidates` therefore enforces the cap for both reserve readings and
refuses a ``top_k`` that would overflow it without any reserve at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from okh_rag.config import GROUP_RESERVE_MODES, RetrievalConfig
from okh_rag.embeddings import HyperedgeEmbeddings
from okh_rag.retrieval.scoring import cosine_relevance
from okh_rag.schema import KnowledgeHypergraph

__all__ = [
    "GROUP_RESERVE_MODES",
    "CandidatePool",
    "HypergraphIndex",
    "scope_candidates",
]

# ``group_reserve_mode``'s legal values are declared in :mod:`okh_rag.config` and re-exported here; the
# module docstring explains what the two readings mean.


class HypergraphIndex:
    """The two reverse indices §A.7's expansion stage needs, built once for many queries.

    Stage 2 asks "which hyperedges share a group or an entity with these seeds?" — a question whose answer
    does not depend on the query. Rebuilding the reverse maps per query would dominate the cost of
    retrieval on a corpus of 10^5 hyperedges, where scoping happens once per question and the maps are
    identical every time. So the index is built against the hypergraph and the retriever holds it for the
    life of the run.

    Both maps are built from the hypergraph rather than from ``H``'s candidate subsets so that a
    ``subset()``-derived training/validation hypergraph and its parent share one index.
    """

    def __init__(self, hypergraph: KnowledgeHypergraph) -> None:
        self.group_of: dict[str, str] = {}
        for group_id, member_ids in hypergraph.groups.items():
            for hyperedge_id in member_ids:
                self.group_of[hyperedge_id] = group_id

        entity_to_edges: dict[str, list[str]] = {}
        for hyperedge in hypergraph.hyperedges.values():
            for entity_id in hyperedge.entity_ids:
                entity_to_edges.setdefault(entity_id, []).append(hyperedge.hyperedge_id)
        self.entity_to_edges: Mapping[str, list[str]] = entity_to_edges

        self.hypergraph = hypergraph

    def group_members(self, group_id: str) -> tuple[str, ...]:
        """All hyperedges of one knowledge group, in the hypergraph's insertion order."""
        return tuple(self.hypergraph.groups.get(group_id, ()))

    def neighbours(self, seeds: Sequence[str]) -> set[str]:
        """§A.7 stage 2 — the union of group membership and one-hop entity overlap around ``seeds``.

        "One-hop entity overlap" is read literally: a hyperedge is a neighbour if its entity set
        intersects a seed's. The hop count is over *entities*, which is why the expansion can leave the
        seed's own scenario — a port entity is shared by every storm that affected that port, and that is
        the whole point of expanding beyond the top-K.

        Returns:
            The neighbour set, excluding the seeds themselves. Excluding them keeps the caller's
            "already selected" bookkeeping to one set difference.
        """
        found: set[str] = set()
        for seed in seeds:
            group_id = self.group_of.get(seed)
            if group_id is not None:
                found.update(self.hypergraph.groups.get(group_id, ()))
            edge = self.hypergraph.hyperedges.get(seed)
            if edge is not None:
                for entity_id in edge.entity_ids:
                    found.update(self.entity_to_edges.get(entity_id, ()))
        found.difference_update(seeds)
        return found


@dataclass(frozen=True)
class CandidatePool:
    """§A.7's output — the set ``C`` on which Algorithm 1 and Viterbi operate.

    The three provenance fields exist so the pool can be audited rather than trusted: a run whose
    "group-aware expansion" is empty everywhere is a finding about the corpus, and without the split
    ``anchors`` / ``reserved`` / ``expanded`` it is invisible.

    Attributes:
        hyperedge_ids: ``C``, ordered by decreasing relevance with ties broken by hyperedge id. The order
            is load-bearing: Viterbi's ``argmax`` tie-break and the beam's deterministic sort both read it.
        relevance: ``(|C|,)`` — ``Rel(e, q)`` for each member, the §A.6 precomputation.
        group_id: The query's group, or ``None`` when it was not supplied.
        anchors: Stage-1 survivors, the top-``top_k`` by cosine.
        reserved: Members admitted specifically to satisfy the own-group reserve floor.
        expanded: Members admitted by stage 2 beyond the reserve — the part that does the "expansion".
    """

    hyperedge_ids: tuple[str, ...]
    relevance: np.ndarray
    group_id: str | None = None
    anchors: tuple[str, ...] = ()
    reserved: tuple[str, ...] = ()
    expanded: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.relevance.shape != (len(self.hyperedge_ids),):
            raise ValueError(
                f"relevance has shape {self.relevance.shape}; expected ({len(self.hyperedge_ids)},)"
            )

    def __len__(self) -> int:
        return len(self.hyperedge_ids)

    @property
    def index(self) -> dict[str, int]:
        """``hyperedge id -> pool position``."""
        return {hid: row for row, hid in enumerate(self.hyperedge_ids)}

    def provenance(self) -> dict[str, int]:
        """Counts per admission path, for the run manifest and the audit report."""
        return {
            "anchors": len(self.anchors),
            "group_reserved": len(self.reserved),
            "expanded": len(self.expanded),
            "pool": len(self.hyperedge_ids),
        }


def _rank_by_relevance(hyperedge_ids: Iterable[str], relevance: Mapping[str, float]) -> list[str]:
    """Sort ids by decreasing relevance with the id as tie-break.

    The tie-break is not decoration. Two hyperedges built from the same structured fact at the same
    horizon can embed to byte-identical text and therefore to *identical* relevance; without a total order
    the pool would depend on dict iteration order, and two runs on the same inputs could retrieve
    different trajectories. ``§A.2 step 5`` makes the same argument for the hyperedge hash.
    """
    return sorted(hyperedge_ids, key=lambda hid: (-relevance[hid], hid))


def scope_candidates(
    hypergraph: KnowledgeHypergraph,
    embeddings: HyperedgeEmbeddings,
    query_vector: np.ndarray,
    *,
    cfg: RetrievalConfig,
    index: HypergraphIndex | None = None,
    group_id: str | None = None,
    allowed_ids: Iterable[str] | None = None,
) -> CandidatePool:
    """§A.7 — build the candidate pool ``C`` for one query.

    Args:
        hypergraph: ``H``. Supplies group membership and entity sets.
        embeddings: The corpus embeddings. Its ``hyperedge_ids`` defines the searchable universe —
            anything unembedded cannot be scored, so it is not a candidate.
        query_vector: The query embedding, ``(d,)``.
        cfg: The ``retrieval`` section, supplying ``top_k``, ``candidate_pool_size``,
            ``group_reserve_fraction`` and ``group_reserve_mode``.
        index: A prebuilt :class:`HypergraphIndex`. Built on demand when omitted; pass one when scoping
            more than a single query.
        group_id: The query's knowledge group, when known. §A.7's "when known" is load-bearing — with
            ``None`` no reserve is applied and every slot goes to the cosine and overlap pool, which is
            the behaviour §A.7 specifies for an unknown group.
        allowed_ids: Optional whitelist restricting the universe. Used to evaluate on a held-out group
            split: without it, retrieval would see the validation and test groups while being scored on
            them, and the hyperedge-level split this project rejects for training (``training.split``)
            would leak back in through the retriever.

    Returns:
        The pool, ordered by decreasing relevance.

    Raises:
        ValueError: If ``candidate_pool_size`` is smaller than ``top_k`` or than the reserve it is
            supposed to contain, or if ``group_reserve_fraction`` is outside ``[0, 1]``. Both are config
            errors that would otherwise show up as a pool that silently breaks §A.7's cap.
    """
    if not (0.0 <= cfg.group_reserve_fraction <= 1.0):
        raise ValueError(f"group_reserve_fraction must be in [0, 1], got {cfg.group_reserve_fraction}")
    if cfg.group_reserve_mode not in GROUP_RESERVE_MODES:
        raise ValueError(f"unknown group_reserve_mode {cfg.group_reserve_mode!r}; expected {GROUP_RESERVE_MODES}")
    # §A.7 pipes stage 1 (top-K) into stage 2 (expansion) under the single cap of |C|. ``config._validate``
    # already rejects K > |C|, but this function accepts a ``RetrievalConfig`` that a caller may have built
    # with ``dataclasses.replace`` and never validated, so the cap is re-established here rather than
    # trusted: everything below assumes the anchors cannot exhaust the pool by themselves.
    if cfg.top_k > cfg.candidate_pool_size:
        raise ValueError(
            f"top_k ({cfg.top_k}) exceeds candidate_pool_size ({cfg.candidate_pool_size}); the anchors "
            "alone would overflow §A.7's cap of |C|"
        )

    index = index if index is not None else HypergraphIndex(hypergraph)

    universe = list(embeddings.hyperedge_ids)
    if allowed_ids is not None:
        allowed = set(allowed_ids)
        universe = [hid for hid in universe if hid in allowed]
    if not universe:
        return CandidatePool(hyperedge_ids=(), relevance=np.zeros(0, dtype=np.float64), group_id=group_id)

    positions = {hid: row for row, hid in enumerate(embeddings.hyperedge_ids)}
    all_relevance = cosine_relevance(query_vector, embeddings.matrix)
    relevance = {hid: float(all_relevance[positions[hid]]) for hid in universe}

    # --- stage 1: top-K cosine retrieval from the global index -----------------
    ranked = _rank_by_relevance(universe, relevance)
    anchors = ranked[: cfg.top_k]

    # --- stage 2: group-aware expansion ---------------------------------------
    reserve = round(cfg.group_reserve_fraction * cfg.candidate_pool_size)
    own_group_ranked = (
        _rank_by_relevance(
            [hid for hid in index.group_members(group_id) if hid in relevance], relevance
        )
        if group_id is not None and group_id in hypergraph.groups
        else []
    )

    taken: set[str] = set(anchors)
    neighbour_pool = index.neighbours(anchors)
    neighbour_pool.difference_update(taken)
    neighbour_pool.intersection_update(universe)
    general_ranked = _rank_by_relevance(neighbour_pool, relevance)

    reserved: list[str] = []
    expanded: list[str] = []
    selected: list[str] = []

    def _admit(source: Iterable[str], budget: int, sink: list[str]) -> None:
        """Move up to ``budget`` unseen ids from ``source`` into ``sink``, consuming them from the pool.

        ``budget`` is always relative to ``sink``, never to the pool: the caller states how many ids that
        particular admission path may still contribute, which is the quantity each reading of §A.7
        actually bounds.
        """
        for hyperedge_id in source:
            if len(sink) >= budget:
                return
            if hyperedge_id in taken:
                continue
            sink.append(hyperedge_id)
            taken.add(hyperedge_id)

    if cfg.group_reserve_mode == "quota":
        # The alternative reading: the 150 slots *contain* the reserve, so 90 go to the general pool and
        # 60 are set aside. The general pool is filled first, which is what distinguishes this from
        # "floor": an own-group hyperedge that outranks an anchor still cannot displace it here.
        general_budget = max(0, cfg.candidate_pool_size - reserve)
        selected.extend(anchors[:general_budget])
        taken.update(selected)
        _admit(general_ranked, general_budget - len(selected), expanded)
        selected.extend(expanded)
        _admit(own_group_ranked, reserve, reserved)
        selected.extend(reserved)
    else:
        # Default "floor" reading: the reserve is a guaranteed minimum, so it is satisfied before the
        # expansion is allowed to consume the remaining slots. A *floor* is not a licence to exceed the
        # cap, though: §A.7 fixes |C| = 150 and the reserve lives inside it, so the reserve may only
        # spend the headroom the anchors left. Unbounded, ``top_k=4, |C|=5, reserve=2`` yields six
        # hyperedges from a pool capped at five, and every downstream "|C| = 150" statement in the
        # report becomes a claim the retriever does not honour. The quota branch bounds itself
        # structurally (``general_budget + reserve == |C|``); the floor branch has to be told.
        own_in_anchors = sum(1 for hid in anchors if index.group_of.get(hid) == group_id)
        selected.extend(anchors)
        headroom = max(0, cfg.candidate_pool_size - len(selected))
        _admit(own_group_ranked, min(max(0, reserve - own_in_anchors), headroom), reserved)
        selected.extend(reserved)
        _admit(general_ranked, cfg.candidate_pool_size - len(selected), expanded)
        selected.extend(expanded)

    remaining = cfg.candidate_pool_size - len(selected)
    if remaining > 0:
        # Still short of the cap: the neighbourhood is exhausted, so backfill from the remaining universe by
        # relevance. Without this a sparse corpus yields a pool smaller than |C|, and every downstream
        # "|C| = 150" statement in the report would be quietly wrong.
        backfill: list[str] = []
        _admit(ranked, remaining, backfill)
        expanded.extend(backfill)
        selected.extend(backfill)

    # §A.7 fixes the pool's *contents* but not its order. Re-sorting by relevance gives one canonical
    # order for every downstream tie-break (Viterbi's argmax, the beam's initial singletons, the
    # deterministic sort). The three provenance fields keep their own admission order, so the *reason* a
    # hyperedge is in the pool survives the re-sort even though its position does not.
    ordered = _rank_by_relevance(selected, relevance)
    return CandidatePool(
        hyperedge_ids=tuple(ordered),
        relevance=np.array([relevance[hid] for hid in ordered], dtype=np.float64),
        group_id=group_id,
        anchors=tuple(anchors),
        reserved=tuple(reserved),
        expanded=tuple(expanded),
    )
