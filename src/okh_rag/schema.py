"""Core data structures for the order-aware knowledge hypergraph.

§A.1 defines two tuples and this module implements them literally:

    Entity     v = (n_v, τ_v, d_v, c_v)                                  §A.1 Eq. 6
    Hyperedge  e = (V_e, r_e, s_e, a_e, c_e),  |V_e| >= 2                §A.1 Eq. 7

plus the two containers the pipeline passes around: :class:`KnowledgeHypergraph` (the ``H`` of Eq. 1)
and :class:`Trajectory` (the ``γ`` of §3.2 Eq. 5).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, Sequence

# ---------------------------------------------------------------------------
# Entity typing — §A.1's "domain-specific type system"
# ---------------------------------------------------------------------------


class EntityCategory(str, Enum):
    """The three functional categories of §A.1.

    "The type system T distinguishes three functional categories, each serving a distinct role during
    retrieval" — persistent objects give cross-horizon continuity, transient states give
    within-horizon specificity, temporal anchors give the positional scaffolding.
    """

    PERSISTENT = "persistent"
    TRANSIENT = "transient"
    TEMPORAL = "temporal"


@dataclass(frozen=True)
class EntityType:
    """A named type in the domain type system, tagged with its §A.1 category."""

    name: str
    category: EntityCategory


# §A.1 names PORT, CYCLONE as persistent; CYCLONE_STATE, OPERATION_STATUS, HAZARD_FORECAST as transient;
# HORIZON_TIME as temporal. The remaining entries are the entity types that actually appear in Figure 3
# and Figure 4 (cone_status, advisory_status, landfall_forecast, track_forecast, closest_station,
# timing_marker, probability_forecast, probability_timing, first_hazard, gale_probability_set,
# impact_timeline, ...). The paper does not enumerate the full type system, so each of those is
# classified by the §A.1 rule it satisfies: does it persist across the scenario, or is it a snapshot at
# one horizon?
# [PARTIALLY_SPECIFIED] §A.1 gives 3 categories and 6 example type names but not the full registry.
ENTITY_TYPE_REGISTRY: Mapping[str, EntityType] = {
    # --- persistent objects -------------------------------------------------
    "PORT": EntityType("PORT", EntityCategory.PERSISTENT),
    "CYCLONE": EntityType("CYCLONE", EntityCategory.PERSISTENT),
    "LOCATION": EntityType("LOCATION", EntityCategory.PERSISTENT),
    # --- temporal anchors ---------------------------------------------------
    "HORIZON_TIME": EntityType("HORIZON_TIME", EntityCategory.TEMPORAL),
    # --- transient states (§A.1 examples + Figure 3/4 types) ----------------
    "CYCLONE_STATE": EntityType("CYCLONE_STATE", EntityCategory.TRANSIENT),
    "OPERATION_STATUS": EntityType("OPERATION_STATUS", EntityCategory.TRANSIENT),
    "HAZARD_FORECAST": EntityType("HAZARD_FORECAST", EntityCategory.TRANSIENT),
    "HAZARD_OBSERVATION": EntityType("HAZARD_OBSERVATION", EntityCategory.TRANSIENT),
    "ADVISORY_STATUS": EntityType("ADVISORY_STATUS", EntityCategory.TRANSIENT),
    "CONE_STATUS": EntityType("CONE_STATUS", EntityCategory.TRANSIENT),
    "LANDFALL_FORECAST": EntityType("LANDFALL_FORECAST", EntityCategory.TRANSIENT),
    "TRACK_FORECAST": EntityType("TRACK_FORECAST", EntityCategory.TRANSIENT),
    "PROBABILITY_FORECAST": EntityType("PROBABILITY_FORECAST", EntityCategory.TRANSIENT),
    "PROBABILITY_TIMING": EntityType("PROBABILITY_TIMING", EntityCategory.TRANSIENT),
    "TIMING_MARKER": EntityType("TIMING_MARKER", EntityCategory.TRANSIENT),
    "THRESHOLD_STATUS": EntityType("THRESHOLD_STATUS", EntityCategory.TRANSIENT),
    "IMPACT_TIMELINE": EntityType("IMPACT_TIMELINE", EntityCategory.TRANSIENT),
    "RECOVERY_STATUS": EntityType("RECOVERY_STATUS", EntityCategory.TRANSIENT),
    "GALE_PROBABILITY_SET": EntityType("GALE_PROBABILITY_SET", EntityCategory.TRANSIENT),
    "CLOSEST_STATION": EntityType("CLOSEST_STATION", EntityCategory.TRANSIENT),
    "FIRST_HAZARD": EntityType("FIRST_HAZARD", EntityCategory.TRANSIENT),
}

UNKNOWN_ENTITY_CATEGORY = EntityCategory.TRANSIENT
"""§A.1 roll-up: an unrecognised type is treated as a transient state.

Rationale: persistent objects and temporal anchors are a small, closed set that the extraction prompt
names explicitly, so anything unseen is far more likely to be a horizon-local state than something that
persists across the whole scenario. Guessing "persistent" would wrongly let it contribute to
entity-continuity scoring (§A.6 Eq. 12) across unrelated horizons.
"""


def normalize_type_name(type_name: str) -> str:
    """Canonicalise a type name to upper snake case so registry lookups are stable."""
    return "_".join(type_name.strip().upper().replace("-", " ").replace("/", " ").split())


def category_of(type_name: str) -> EntityCategory:
    """Return the §A.1 category for ``type_name``, defaulting to transient for unknown types."""
    entry = ENTITY_TYPE_REGISTRY.get(normalize_type_name(type_name))
    return entry.category if entry is not None else UNKNOWN_ENTITY_CATEGORY


def is_known_type(type_name: str) -> bool:
    """Whether ``type_name`` is present in the §A.1 registry."""
    return normalize_type_name(type_name) in ENTITY_TYPE_REGISTRY


def _clamp_unit_interval(value: Any) -> float:
    """Clamp to ``(0, 1]`` per §A.1, mapping non-finite and <=0 values to a small positive floor."""
    if value is None:
        return 1.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 1.0
    if v != v or v in (float("inf"), float("-inf")):  # NaN / inf
        return 1.0
    if v <= 0.0:
        return 1e-6
    return min(v, 1.0)


# ---------------------------------------------------------------------------
# §A.1 Eq. 6 — Entity
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """§A.1 Eq. 6 — ``v = (n_v, τ_v, d_v, c_v)``.

    Attributes:
        entity_id: The canonical hierarchical identifier built in §A.2 step 2
            (``family:storm:port:horizon``). Distinct from ``name``: the same ``name`` at two horizons
            yields two ids.
        name: ``n_v`` — "the entity name, grounded in the source hyperedge description (n_v ⊆ s_e)".
        type: ``τ_v`` — a type from the domain type system, normalised to upper snake case.
        description: ``d_v`` — "a natural-language explanation of the entity's role".
        confidence: ``c_v ∈ (0, 1]`` — "extraction certainty".
    """

    entity_id: str
    name: str
    type: str
    description: str
    confidence: float

    def __post_init__(self) -> None:
        if not self.entity_id:
            raise ValueError("entity_id must be non-empty")
        if not self.name:
            raise ValueError("entity name (n_v) must be non-empty")
        self.type = normalize_type_name(self.type)
        # §A.1 — c_v ∈ (0, 1]. Clamped rather than rejected: extractor output occasionally emits 0.0 or
        # 1.5, and discarding a whole hyperedge over a confidence field is the costlier failure.
        self.confidence = _clamp_unit_interval(self.confidence)

    @property
    def category(self) -> EntityCategory:
        """The §A.1 functional category of this entity's type."""
        return category_of(self.type)

    @property
    def is_temporal_anchor(self) -> bool:
        """Whether this entity is a ``HORIZON_TIME`` anchor (§A.1)."""
        return self.category is EntityCategory.TEMPORAL


# ---------------------------------------------------------------------------
# §A.1 Eq. 7 — Hyperedge
# ---------------------------------------------------------------------------


@dataclass
class Hyperedge:
    """§A.1 Eq. 7 — ``e = (V_e, r_e, s_e, a_e, c_e)``.

    Attributes:
        hyperedge_id: Hash of ``(r_e, V_e, s_e)`` computed in §A.2 step 5.
        entity_ids: ``V_e`` — the participating entity set, ``|V_e| >= 2``.
        relation: ``r_e`` — a relation normalised onto the Table 3 vocabulary.
        evidence: ``s_e`` — "a natural language description preserving the full semantic content of the
            interaction". Also the text §A.4 embeds (concatenated with relation, entity names and
            types, and attribute values).
        attributes: ``a_e`` — key/value attributes such as quantitative thresholds.
        confidence: ``c_e ∈ (0, 1]``.
        group_id: The knowledge group (§A.4) this hyperedge belongs to, i.e. one scenario.
        horizon_hours: Lead time in hours (``T-48`` → ``48``). ``None`` for a hyperedge that is not
            horizon-grounded, such as a synthesised cross-horizon transition (§A.2 step 4).
        text_position: Character offset of ``evidence`` inside its source document, the last §A.3
            topological tie-breaker.
    """

    hyperedge_id: str
    entity_ids: tuple[str, ...]
    relation: str
    evidence: str
    attributes: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    group_id: str = ""
    horizon_hours: int | None = None
    text_position: int = 0

    def __post_init__(self) -> None:
        self.entity_ids = tuple(dict.fromkeys(self.entity_ids))  # dedupe, preserve order
        if not self.hyperedge_id:
            raise ValueError("hyperedge_id must be non-empty")
        if not self.evidence:
            raise ValueError("evidence (s_e) must be non-empty")
        if not self.relation:
            raise ValueError("relation (r_e) must be non-empty")
        # Guaranteed by the extractor + §A.2 dedup; asserted here so a bug cannot slip a 1-entity
        # "hyperedge" into the retrieval index, where it would behave as a plain graph edge.
        if len(self.entity_ids) < 2:
            raise ValueError(
                f"hyperedge {self.hyperedge_id!r} has |V_e| = {len(self.entity_ids)}; "
                "§A.1 Eq. 7 requires |V_e| >= 2"
            )
        self.confidence = _clamp_unit_interval(self.confidence)

    @property
    def cardinality(self) -> int:
        """``|V_e|`` — the order of the interaction, which is what distinguishes a hyperedge from a
        binary graph edge (§3.1)."""
        return len(self.entity_ids)


# ---------------------------------------------------------------------------
# Eq. 1 / Eq. 9 — the hypergraph
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeHypergraph:
    """``H = (V, E)`` — Eq. 1 / Eq. 9.

    Hyperedges are indexed by id; entities by id. ``groups`` maps a knowledge group (§A.4) to the
    hyperedges extracted from that scenario's document, which is the unit over which §A.3 builds a
    canonical trajectory and §A.4 samples negatives.
    """

    entities: dict[str, Entity] = field(default_factory=dict)
    hyperedges: dict[str, Hyperedge] = field(default_factory=dict)
    groups: dict[str, list[str]] = field(default_factory=dict)

    # -- construction ------------------------------------------------------

    def add_entity(self, entity: Entity) -> None:
        """Insert or overwrite an entity by id."""
        self.entities[entity.entity_id] = entity

    def add_hyperedge(self, hyperedge: Hyperedge) -> None:
        """Insert a hyperedge and register it under its group."""
        self.hyperedges[hyperedge.hyperedge_id] = hyperedge
        if hyperedge.group_id:
            bucket = self.groups.setdefault(hyperedge.group_id, [])
            if hyperedge.hyperedge_id not in bucket:
                bucket.append(hyperedge.hyperedge_id)

    def merge(self, other: "KnowledgeHypergraph") -> None:
        """Fold ``other`` into ``self``; aggregates per-document results (Eq. 9)."""
        for entity in other.entities.values():
            self.add_entity(entity)
        for hyperedge in other.hyperedges.values():
            self.add_hyperedge(hyperedge)

    def subset(self, group_ids: Iterable[str]) -> "KnowledgeHypergraph":
        """A new hypergraph holding only the named knowledge groups.

        Used by §A.4's group-based split: the training and validation corpora are two subsets of the one
        hypergraph, and both must keep the *same* :class:`Hyperedge` objects so that the embeddings
        indexed by hyperedge id stay valid. Only the containers are new.

        Entities are recomputed as the union of the surviving ``V_e``, matching Eq. 9's definition of
        ``V``. An entity used only by a dropped group is therefore absent, which is what keeps
        :meth:`entity_document_frequency` honest on a subset.

        Args:
            group_ids: The groups to keep. Unknown ids are ignored.

        Returns:
            The subset. Returns an empty hypergraph when no id matches.
        """
        wanted = set(group_ids)
        kept = KnowledgeHypergraph()
        for group_id in self.groups:
            if group_id not in wanted:
                continue
            for hyperedge_id in self.groups[group_id]:
                edge = self.hyperedges.get(hyperedge_id)
                if edge is None:
                    continue
                kept.add_hyperedge(edge)
                for entity_id in edge.entity_ids:
                    entity = self.entities.get(entity_id)
                    if entity is not None:
                        kept.add_entity(entity)
        return kept

    # -- access ------------------------------------------------------------

    def entity_set(self, hyperedge: Hyperedge) -> tuple[Entity, ...]:
        """Resolve ``V_e`` to :class:`Entity` objects, silently skipping dangling ids."""
        return tuple(self.entities[e] for e in hyperedge.entity_ids if e in self.entities)

    def entity_names(self, hyperedge: Hyperedge) -> tuple[str, ...]:
        """The names of ``V_e``, in the hyperedge's stored order."""
        return tuple(e.name for e in self.entity_set(hyperedge))

    def entity_id_set(self, hyperedge: Hyperedge) -> frozenset[str]:
        """``V_e`` as a set — the form every scoring term and overlap test needs."""
        return frozenset(hyperedge.entity_ids)

    def group_hyperedges(self, group_id: str) -> list[Hyperedge]:
        """All hyperedges of one knowledge group, in insertion order."""
        return [self.hyperedges[h] for h in self.groups.get(group_id, [])]

    def entity_document_frequency(self) -> dict[str, int]:
        """Map entity id to the number of hyperedges it participates in.

        This is the quantity §A.1 alludes to when it says persistent objects "appear in hyperedges at
        many sequence positions and serve as anchors".
        """
        freq: dict[str, int] = {}
        for edge in self.hyperedges.values():
            for eid in edge.entity_ids:
                freq[eid] = freq.get(eid, 0) + 1
        return freq

    # -- dunder ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.hyperedges)

    def __contains__(self, hyperedge_id: object) -> bool:
        return hyperedge_id in self.hyperedges

    def __iter__(self) -> Iterator[Hyperedge]:
        return iter(self.hyperedges.values())


# ---------------------------------------------------------------------------
# §3.2 — Trajectory
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """An ordered sequence of hyperedges ``γ = (e_(1), ..., e_(L))`` (§3.2).

    Attributes:
        hyperedge_ids: The ordered members. Order is the whole point — two trajectories with the same
            set but a different order are different objects, which is the paper's central claim.
        score: The Eq. 5 objective value.
        components: Per-term breakdown, for the Table 2 component ablations and logging.
    """

    hyperedge_ids: tuple[str, ...]
    score: float = 0.0
    components: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.hyperedge_ids = tuple(self.hyperedge_ids)

    def __len__(self) -> int:
        return len(self.hyperedge_ids)

    def __iter__(self) -> Iterator[str]:
        return iter(self.hyperedge_ids)

    @property
    def as_set(self) -> frozenset[str]:
        """The unordered content — what a permutation-invariant retriever would have returned."""
        return frozenset(self.hyperedge_ids)

    def shuffled(self, rng_seed: int) -> "Trajectory":
        """Permute the members while keeping content and score fixed.

        This is the §4.5 / Table 2 "Shuffle" row: performance drops "from 0.534 to 0.487, the largest
        drop observed, despite identical retrieved content."
        """
        ids = list(self.hyperedge_ids)
        random.Random(rng_seed).shuffle(ids)
        return Trajectory(tuple(ids), self.score, dict(self.components))

    def to_evidence_sequence(self, hypergraph: KnowledgeHypergraph) -> list[tuple[int, Hyperedge]]:
        """Pair each member with its 1-based step index, as §A.8's evidence format requires."""
        out: list[tuple[int, Hyperedge]] = []
        for step, hid in enumerate(self.hyperedge_ids, start=1):
            edge = hypergraph.hyperedges.get(hid)
            if edge is not None:
                out.append((step, edge))
        return out


@dataclass
class TrajectoryScore:
    """The five additive terms of §3.2 Eq. 5, kept separate for the Table 2 component ablations."""

    relevance: float = 0.0
    order_coherence: float = 0.0
    precedence_consistency: float = 0.0
    entity_continuity: float = 0.0
    phase_coverage: float = 0.0

    def total(self, lambda_: float, mu: float, nu: float, rho: float) -> float:
        """Evaluate §3.2 Eq. 5.

        ``Σ Rel(e_(k), q) + λ Σ log P_θ(e_(k+1)|e_(k)) + µ·Prec(γ) + ν·Ovlp(γ) + ρ·Cov(γ)``

        Args:
            lambda_: λ — order-coherence weight.
            mu: µ — precedence-consistency weight.
            nu: ν — entity-continuity weight.
            rho: ρ — phase-coverage weight.

        Returns:
            The scalar objective. With all four weights zero this degenerates to the relevance sum,
            which §3.2 identifies with standard top-k retrieval.

        A zero weight **removes** its term rather than multiplying it. For a finite term the two are the
        same thing; for an infinite one they are not, and ``0.0 * -inf`` is NaN. ``-inf`` is reachable in
        ``order_coherence``: the learned table of §A.6 is a table of log-probabilities, and a log-prob
        underflows to ``-inf`` as soon as a softmax mass rounds to zero in float64. Under the
        ``−λ (order coherence)`` ablation — Table 2's own row — the ordinary implementation would then
        turn that score into NaN and return an arbitrary trajectory while reporting a plausible-looking
        number, since NaN fails every comparison and so loses every ``max`` silently.

        Skipping the term is also the honest reading of "drop λ": the term is not present, rather than
        present with weight zero. The distinction is observable and tested —
        ``tests/test_retrieval.py::TestAblationSwitches::test_zero_weight_times_infinity_is_not_nan``
        feeds an ``-inf`` term in directly and requires a finite total, because that is the only way to
        exercise the guard without depending on a particular embedding to underflow.
        """
        # The relevance term carries no weight in Eq. 5 — it is the unweighted leading sum.
        total = self.relevance
        if lambda_ != 0.0:
            total += lambda_ * self.order_coherence
        if mu != 0.0:
            total += mu * self.precedence_consistency
        if nu != 0.0:
            total += nu * self.entity_continuity
        if rho != 0.0:
            total += rho * self.phase_coverage
        return total

    def as_dict(self) -> dict[str, float]:
        """Plain-dict view for logging and ablation tables."""
        return {
            "relevance": self.relevance,
            "order_coherence": self.order_coherence,
            "precedence_consistency": self.precedence_consistency,
            "entity_continuity": self.entity_continuity,
            "phase_coverage": self.phase_coverage,
        }


def sum_scores(scores: Sequence[TrajectoryScore]) -> TrajectoryScore:
    """Element-wise sum, used when folding incremental beam scores into a whole-trajectory score."""
    total = TrajectoryScore()
    for s in scores:
        total.relevance += s.relevance
        total.order_coherence += s.order_coherence
        total.precedence_consistency += s.precedence_consistency
        total.entity_continuity += s.entity_continuity
        total.phase_coverage += s.phase_coverage
    return total
