"""§A.3 — precedence construction.

Turns the unordered hypergraph ``H`` into the order-aware hypergraph ``H≺ = (V, E, ≺)`` of §3.1 Eq. 2.

§A.3 defines ``≺`` through four rule families, then derives a canonical trajectory per knowledge group:

    | Rule family                | §A.3 wording                                                        |
    |----------------------------|---------------------------------------------------------------------|
    | within-horizon phase order | "ordered by semantic phase following Table 3"                        |
    | cross-horizon evolution    | "decreasing horizon order: T-96 ≺ T-72 ≺ T-48 ≺ T-24 ≺ T-12"        |
    | causal chain rules         | four named phase pairs                                              |
    | family-to-change ordering  | "within-horizon hyperedges ... precede any cross-horizon change"     |

The canonical trajectory is then "topological sorting the resulting DAG with deterministic
tie-breaking (phase rank, family rank, lead time, text position)".
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from okh_rag.config import PrecedenceConfig
from okh_rag.schema import Hyperedge, KnowledgeHypergraph
from okh_rag.vocabulary import (
    ATTR_ORIGIN_FAMILY,
    ATTR_ORIGIN_HORIZONS,
    CAUSAL_CHAIN_PHASE_RULES,
    CROSS_HORIZON_FAMILY,
    FAMILY_BY_NAME,
    PRECEDENCE_PHASE_ORDER,
    PRECEDENCE_PHASE_RANK,
    family_index_of,
    family_of,
    is_cross_horizon,
    precedence_phase_of,
    precedence_rank_of,
)

# Rule names, used as provenance labels on every edge so the §A.5 ablations and the §A.8 "Reasoning:"
# line can say *why* two hyperedges are ordered.
RULE_WITHIN_HORIZON_PHASE = "within_horizon_phase"
RULE_CROSS_HORIZON_EVOLUTION = "cross_horizon_evolution"
RULE_CAUSAL_CHAIN = "causal_chain"
RULE_FAMILY_TO_CHANGE = "family_to_change"

# Drop order when a cycle has to be broken. §A.3 asserts the union of the four rule families is a DAG but
# supplies no proof, and no rule prevents a cycle in principle.
# [UNSPECIFIED] §A.3 does not say what to do on a cycle. Alternatives: raise, drop edges, break
# arbitrarily. We drop from the least "semantic" rule first so the phase ordering — the rule §A.3
# describes as reflecting "logical dependency" — survives longest.
CYCLE_DROP_PRIORITY: tuple[str, ...] = (
    RULE_FAMILY_TO_CHANGE,
    RULE_CAUSAL_CHAIN,
    RULE_CROSS_HORIZON_EVOLUTION,
)

# Attribute keys written by the §A.2 step-4 cross-horizon synthesis and read back here. Declared in
# okh_rag.vocabulary so the writer and the reader cannot drift.


# ---------------------------------------------------------------------------
# The precedence graph
# ---------------------------------------------------------------------------


@dataclass
class PrecedenceGraph:
    """The partial order ``≺`` over hyperedges, plus the provenance of every edge.

    ``≺`` "is a partial order accommodating concurrent interactions", so this is *not* a total order:
    two hyperedges at the same horizon in unrelated phases are simply incomparable, which is what makes
    §A.6 Eq. 11's denominator exclude unrelated pairs.
    """

    rule_of: dict[tuple[str, str], str] = field(default_factory=dict)
    """``(src, dst) -> rule name``. Presence of the key *is* ``src ≺ dst``."""

    dropped_edges: list[tuple[str, str, str]] = field(default_factory=list)
    """``(src, dst, rule)`` triples discarded by :func:`_resolve_cycles`, for diagnostics."""

    def precedes(self, src: str, dst: str) -> bool:
        """Whether ``src ≺ dst``."""
        return (src, dst) in self.rule_of

    def related(self, src: str, dst: str) -> bool:
        """Whether the pair is comparable in either direction, i.e. ``(src, dst) ∈ (≺ ∪ ≻)``.

        §A.6 Eq. 11 restricts its denominator to comparable pairs, so this predicate is exactly the
        "unrelated pairs are excluded from the denominator" clause.
        """
        return (src, dst) in self.rule_of or (dst, src) in self.rule_of

    def rule_between(self, src: str, dst: str) -> str | None:
        """The rule that ordered ``src`` before ``dst``, if any."""
        return self.rule_of.get((src, dst))

    def outgoing(self, src: str) -> list[str]:
        """Successors of ``src``."""
        return [dst for (s, dst) in self.rule_of if s == src]

    def incoming(self, dst: str) -> list[str]:
        """Predecessors of ``dst``."""
        return [src for (src, d) in self.rule_of if d == dst]

    def __len__(self) -> int:
        return len(self.rule_of)


# ---------------------------------------------------------------------------
# Sort keys — §A.3's deterministic tie-breaking
# ---------------------------------------------------------------------------


def _effective_phase_rank(edge: Hyperedge) -> float:
    """The §A.3 phase rank used for tie-breaking.

    A family-13 change edge has no stage of its own, so it borrows the stage of the family it points
    back to and is nudged half a step later. That reproduces §A.3's intent — "the 'before' state is
    presented before the transition connecting it to the next horizon" — without inventing a 12th stage.
    """
    if is_cross_horizon(edge.relation):
        origin = str(edge.attributes.get(ATTR_ORIGIN_FAMILY, ""))
        origin_family = FAMILY_BY_NAME.get(origin)
        if origin_family is None:
            return len(PRECEDENCE_PHASE_ORDER) + 0.5
        ranks = [
            PRECEDENCE_PHASE_RANK[phase]
            for rel in origin_family.relations
            if (phase := precedence_phase_of(rel)) is not None
        ]
        return (max(ranks) if ranks else len(PRECEDENCE_PHASE_ORDER)) + 0.5
    return float(precedence_rank_of(edge.relation))


def precedence_sort_key(edge: Hyperedge) -> tuple[float, int, float, int]:
    """§A.3's four-part tie-breaker, in the order the paper lists them.

    ``(phase rank, family rank, lead time, text position)``

    Two deviations forced by the data model, both documented in ``REPRODUCTION_NOTES.md``:

    * **lead time is negated.** §A.3's cross-horizon rule is "T-96 ≺ T-72 ≺ T-48 ≺ T-24 ≺ T-12", i.e.
      *larger* lead time first, so the tuple must sort descending on lead time to sort ascending as a
      whole. A hyperedge with no horizon (a family-13 change edge) is placed last by using ``-1``.
    * **text position breaks remaining ties.** §A.3 lists it but never defines it; we use the character
      offset of ``s_e`` in its source block.

    Returns:
        A 4-tuple that is sortable and totally deterministic.
    """
    lead_time = edge.horizon_hours if edge.horizon_hours is not None else -1
    return (
        _effective_phase_rank(edge),
        family_index_of(edge.relation),
        -float(lead_time),
        edge.text_position,
    )


# ---------------------------------------------------------------------------
# Rule families
# ---------------------------------------------------------------------------


def _within_horizon_phase_edges(edges: list[Hyperedge]) -> list[tuple[str, str, str]]:
    """§A.3 rule 1 — "At the same horizon, hyperedges are ordered by semantic phase following Table 3".

    The paper's justification: "an operational decision presupposes a hazard assessment, which
    presupposes an advisory."
    """
    out: list[tuple[str, str, str]] = []
    for i, a in enumerate(edges):
        for b in edges[i + 1 :]:
            if a.horizon_hours != b.horizon_hours:
                continue
            ra, rb = precedence_rank_of(a.relation), precedence_rank_of(b.relation)
            if ra < 0 or rb < 0:
                continue  # family 13 is handled by rule 4, not by phase ordering
            if ra < rb:
                out.append((a.hyperedge_id, b.hyperedge_id, RULE_WITHIN_HORIZON_PHASE))
            elif rb < ra:
                out.append((b.hyperedge_id, a.hyperedge_id, RULE_WITHIN_HORIZON_PHASE))
    return out


def _cross_horizon_evolution_edges(edges: list[Hyperedge]) -> list[tuple[str, str, str]]:
    """§A.3 rule 2 — "Hyperedges in the same semantic family at different horizons follow decreasing
    horizon order: T-96 ≺ T-72 ≺ T-48 ≺ T-24 ≺ T-12".

    "This ensures that evolving information is presented chronologically."
    """
    out: list[tuple[str, str, str]] = []
    by_family: dict[str, list[Hyperedge]] = {}
    for e in edges:
        if e.horizon_hours is None or is_cross_horizon(e.relation):
            continue
        fam = family_of(e.relation)
        if fam is not None:
            by_family.setdefault(fam, []).append(e)

    for members in by_family.values():
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                if a.horizon_hours == b.horizon_hours:
                    continue
                # larger lead time first
                if a.horizon_hours > b.horizon_hours:
                    out.append((a.hyperedge_id, b.hyperedge_id, RULE_CROSS_HORIZON_EVOLUTION))
                else:
                    out.append((b.hyperedge_id, a.hyperedge_id, RULE_CROSS_HORIZON_EVOLUTION))
    return out


def _causal_chain_edges(edges: list[Hyperedge], rules: Iterable[tuple[str, str]]) -> list[tuple[str, str, str]]:
    """§A.3 rule 3 — the four named within-horizon causal pathways.

    Note (verified by ``tests/test_precedence.py::test_causal_rules_are_implied_by_phase_order``): all
    four pairs already hold under rule 1, because the §A.3 phase ordering places advisory before hazard,
    hazard before operation and impact, and impact before recovery. These rules therefore add no edges
    for the default configuration. They are still materialised, because

    * §A.3 states them separately and the paper's ablation story treats them as distinct constraints, and
    * the provenance label is what §A.8's "Reasoning: hazard_to_operation" line reports.
    """
    pairs = {(a, b) for a, b in rules}
    out: list[tuple[str, str, str]] = []
    for i, a in enumerate(edges):
        for b in edges[i + 1 :]:
            if a.horizon_hours != b.horizon_hours:
                continue
            pa, pb = precedence_phase_of(a.relation), precedence_phase_of(b.relation)
            if pa is None or pb is None:
                continue
            if (pa, pb) in pairs:
                out.append((a.hyperedge_id, b.hyperedge_id, RULE_CAUSAL_CHAIN))
            elif (pb, pa) in pairs:
                out.append((b.hyperedge_id, a.hyperedge_id, RULE_CAUSAL_CHAIN))
    return out


def _family_to_change_edges(edges: list[Hyperedge]) -> list[tuple[str, str, str]]:
    """§A.3 rule 4 — "Within-horizon hyperedges in a semantic family precede any cross-horizon change
    hyperedges (family 13) for the same family".

    "For the same family" is resolved through the ``origin_family`` attribute written by the §A.2
    step-4 synthesis; a family-13 edge whose origin is unknown is left unconstrained rather than
    attached to an arbitrary family.
    """
    change_edges = [e for e in edges if is_cross_horizon(e.relation)]
    if not change_edges:
        return []

    out: list[tuple[str, str, str]] = []
    for change in change_edges:
        origin = str(change.attributes.get(ATTR_ORIGIN_FAMILY, ""))
        if not origin:
            continue
        for e in edges:
            if is_cross_horizon(e.relation):
                continue
            if family_of(e.relation) == origin:
                out.append((e.hyperedge_id, change.hyperedge_id, RULE_FAMILY_TO_CHANGE))
    return out


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_precedence(hypergraph: KnowledgeHypergraph, cfg: PrecedenceConfig) -> PrecedenceGraph:
    """Construct ``≺`` per knowledge group (§A.1: "produce a DAG within each knowledge group").

    Groups are processed independently on purpose: §A.1 says precedence is built "within each knowledge
    group", and cross-group edges would let an unrelated storm constrain another storm's trajectory.

    Args:
        hypergraph: The unordered hypergraph ``H``.
        cfg: The ``precedence`` section of ``configs/base.yaml``.

    Returns:
        A :class:`PrecedenceGraph` over hyperedge ids. Edges whose source or destination is missing from
        the hypergraph are skipped, so a partially built ``H`` cannot corrupt the order.
    """
    graph = PrecedenceGraph()
    causal_rules = [(a, b) for a, b in cfg.causal_chain_rules]

    for group_id, member_ids in hypergraph.groups.items():
        edges = [hypergraph.hyperedges[h] for h in member_ids if h in hypergraph.hyperedges]
        if len(edges) < 2:
            continue

        candidates = [
            *_within_horizon_phase_edges(edges),
            *_cross_horizon_evolution_edges(edges),
            *_causal_chain_edges(edges, causal_rules),
            *_family_to_change_edges(edges),
        ]

        # Last rule wins on a duplicate pair: rules are applied in §A.3's stated order, so the
        # family-to-change attachment (rule 4) is the most specific label for a change edge.
        for src, dst, rule in candidates:
            if src != dst:
                graph.rule_of[(src, dst)] = rule

        _resolve_cycles(graph, edges, group_id)

    return graph


def _resolve_cycles(graph: PrecedenceGraph, edges: list[Hyperedge], group_id: str) -> None:
    """Break any cycle inside one group, following ``precedence.cycle_policy``.

    §A.3 asserts the result is a DAG but gives no argument that the four rule families cannot combine
    into a cycle. In the default configuration they do not — the phase ordering is already a total order
    on stage, and both the causal rules and the family-to-change rule point forward along it — but a
    customised ``phase_order`` could. This keeps the pipeline from hanging on a topological sort.
    """
    ids = {e.hyperedge_id for e in edges}
    local = {k: v for k, v in graph.rule_of.items() if k[0] in ids and k[1] in ids}

    for rule in CYCLE_DROP_PRIORITY:
        if not _has_cycle(local):
            return
        victims = [pair for pair, r in local.items() if r == rule]
        if not victims:
            continue
        for pair in victims:
            del local[pair]
            graph.rule_of.pop(pair, None)
            graph.dropped_edges.append((pair[0], pair[1], rule))

    if _has_cycle(local):
        raise ValueError(
            f"precedence relation for group {group_id!r} still contains a cycle after dropping every "
            f"non-phase rule family; the configured phase_order is inconsistent with the causal rules"
        )


def _has_cycle(edges: Mapping[tuple[str, str], str]) -> bool:
    """Iterative DFS three-colour cycle detection over a sparse edge map."""
    adjacency: dict[str, list[str]] = {}
    for src, dst in edges:
        adjacency.setdefault(src, []).append(dst)

    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = {node: WHITE for node in adjacency}
    for neighbours in adjacency.values():
        for n in neighbours:
            colour.setdefault(n, WHITE)

    for root in list(colour):
        if colour[root] != WHITE:
            continue
        stack: list[tuple[str, int]] = [(root, 0)]
        colour[root] = GREY
        while stack:
            node, idx = stack[-1]
            neighbours = adjacency.get(node, [])
            if idx < len(neighbours):
                stack[-1] = (node, idx + 1)
                nxt = neighbours[idx]
                if colour.get(nxt, WHITE) == GREY:
                    return True
                if colour.get(nxt, WHITE) == WHITE:
                    colour[nxt] = GREY
                    stack.append((nxt, 0))
            else:
                colour[node] = BLACK
                stack.pop()
    return False


# ---------------------------------------------------------------------------
# Canonical trajectory
# ---------------------------------------------------------------------------


def canonical_order(hypergraph: KnowledgeHypergraph, graph: PrecedenceGraph) -> dict[str, tuple[str, ...]]:
    """§A.3 — the canonical linear ordering per knowledge group.

    "Topological sorting the resulting DAG with deterministic tie-breaking (phase rank, family rank,
    lead time, text position) yields a canonical linear ordering per knowledge group. This ordering
    serves two purposes: it provides the ground-truth trajectory for document-order supervision in
    training (§A.4), and it defines the structural precedence that ``Prec(γ)`` enforces during
    retrieval."

    Args:
        hypergraph: The hypergraph being ordered.
        graph: The precedence relation built by :func:`build_precedence`.

    Returns:
        ``group_id -> ordered hyperedge ids``. Groups with fewer than two members are included as-is.
    """
    order: dict[str, tuple[str, ...]] = {}

    for group_id, member_ids in hypergraph.groups.items():
        members = [hypergraph.hyperedges[h] for h in member_ids if h in hypergraph.hyperedges]
        if not members:
            continue

        ids = {e.hyperedge_id for e in members}
        indegree: dict[str, int] = {h: 0 for h in ids}
        successors: dict[str, list[str]] = {h: [] for h in ids}
        for (src, dst) in graph.rule_of:
            if src in ids and dst in ids:
                successors[src].append(dst)
                indegree[dst] += 1

        by_id = {e.hyperedge_id: e for e in members}
        # Heap keyed on the §A.3 tie-breaker. Hyperedge ids are appended as a final component so two
        # hyperedges with identical keys still order deterministically instead of by dict insertion.
        heap: list[tuple[tuple[float, int, float, int], str]] = [
            (precedence_sort_key(by_id[h]), h) for h, deg in indegree.items() if deg == 0
        ]
        heapq.heapify(heap)

        ordered: list[str] = []
        while heap:
            _, node = heapq.heappop(heap)
            ordered.append(node)
            for nxt in successors[node]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    heapq.heappush(heap, (precedence_sort_key(by_id[nxt]), nxt))

        # Should be unreachable: _resolve_cycles runs first. Kept as a hard failure rather than a
        # silent partial order, because a truncated canonical trajectory would corrupt §A.4 training.
        if len(ordered) != len(ids):
            raise ValueError(
                f"topological sort for group {group_id!r} emitted {len(ordered)}/{len(ids)} hyperedges; "
                "the precedence relation still contains a cycle"
            )

        order[group_id] = tuple(ordered)

    return order


def trajectory_order_index(order: Mapping[str, tuple[str, ...]]) -> dict[str, int]:
    """Invert :func:`canonical_order` into ``hyperedge_id -> position``.

    This is the ``π(·)`` of §A.4's document-order signal: "For co-occurring hyperedges (e_i, e_j) with
    π(e_i) < π(e_j), the pair joins P."
    """
    index: dict[str, int] = {}
    for ids in order.values():
        for position, hid in enumerate(ids):
            index[hid] = position
    return index


def phase_sequence(hypergraph: KnowledgeHypergraph, ordered_ids: Iterable[str]) -> tuple[str, ...]:
    """The §A.3 phase of each hyperedge in a sequence, for diagnostics and the §A.8 evidence block."""
    out: list[str] = []
    for hid in ordered_ids:
        edge = hypergraph.hyperedges.get(hid)
        if edge is None:
            continue
        out.append(precedence_phase_of(edge.relation) or CROSS_HORIZON_FAMILY)
    return tuple(out)
