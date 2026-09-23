"""§5 of ``data/CONTRACT.md`` — scoring retrieval against what a question actually needs.

§4.1 measures OKH-RAG with one number: end-to-end answer accuracy. That number cannot say
*where* a failure happened, and the paper's own ablation table leans on the difference. When
the ``−λ`` row drops, the drop could be the retriever returning a worse trajectory, the
generator being handed a worse evidence block, or the retriever returning the right evidence
in the wrong order. Accuracy moves for all three.

``CyPortQA_template.json`` removes the ambiguity, because each of the 48 templates names the
field that produces its answer, and §3.2 of ``CONTRACT.md`` names the §A.1 Table 3 relation
that field implies. A question therefore comes with a *closed-form* statement of what evidence
it needs::

    gold(q) = { e ∈ H : e.group = q.scenario  ∧  e.horizon = q.horizon
                     ∧  e.relation ∈ relations(q.template's field) }

and every stage of §A.7 can be scored against it separately:

* did the hyperedges make it into the candidate pool ``C`` at all? (:attr:`~OracleMetrics.pool_recall`)
* did they reach the top-K cosine anchors? (:attr:`~OracleMetrics.anchor_recall`)
* are they in the returned trajectory? (:attr:`~OracleMetrics.trajectory_recall`)
* how much of the trajectory is gold, rather than filler? (:attr:`~OracleMetrics.trajectory_precision`)
* where in the sequence do they appear, and in what order?
  (:attr:`~OracleMetrics.first_gold_step`, :attr:`~OracleMetrics.order_concordance`)

The last two are the ones §4.5 needs and the first four cannot supply. Table 2's shuffle row
permutes the retrieved trajectory "despite identical retrieved content", so every content
metric is invariant across that row by construction — a harness that only measured recall
would report the shuffle ablation as a no-op. Position and order are what actually change, so
:attr:`~OracleMetrics.first_gold_step` and :attr:`~OracleMetrics.order_concordance` are the
oracle's order-aware half, and they measure order *against §A.3's canonical ordering*
(:func:`~okh_rag.precedence.canonical_order`) rather than against the retriever's own score.

**Why the gold set is keyed on relation and not on field.** The obvious definition is "the
hyperedges built from that field", and it is the one ``CONTRACT.md`` §5 states. It is exactly
right on :mod:`~okh_rag.evaluation.reference_graph`, where one hyperedge is one field. It is
unusable on an *extracted* hypergraph, because §A.2 asks the model for evidence sentences and
never for source offsets — there is no field attribute to key on, so a field-keyed oracle
would score zero records on the graph the pipeline actually produces. Keying on the relation
keeps one definition for both, and :func:`gold_matches_field_provenance` checks the reference
graph's gold sets against the field-keyed alternative, so the looser key is verified against
the stricter one wherever both exist rather than being taken on trust.

The looser key is also the *specified* one: §3.2 says the field-to-relation table is what
"makes LLM extraction scoreable" by letting output "be compared against the family each field
implies".
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from okh_rag.config import Config
from okh_rag.data.cyportqa import QARecord, ScenarioId, Template
from okh_rag.evaluation.field_map import FieldKind, FieldResolution, relation_family, relations_for_path
from okh_rag.evaluation.reference_graph import ReferenceGraph
from okh_rag.retrieval.retriever import RetrievalResult
from okh_rag.schema import KnowledgeHypergraph

__all__ = [
    "EvidenceStatus",
    "GoldEvidence",
    "OracleMetrics",
    "OracleOutcome",
    "OracleReport",
    "TemplateCoverage",
    "evaluate_record",
    "gold_for_record",
    "gold_matches_field_provenance",
    "template_coverage",
]


class EvidenceStatus(str, Enum):
    """What the oracle was able to say about a record's evidence requirement."""

    RESOLVED = "resolved"
    """The template's field maps onto a relation, and the graph holds at least one such hyperedge."""

    EMPTY = "empty"
    """The field maps onto a relation, but the graph has no hyperedge there.

    This is the interesting failure, and it is *not* a retrieval failure: the evidence the
    question needs was never built. Two causes are worth telling apart, and the reference
    graph's anomaly counters are what distinguish them — the field may genuinely be absent at
    that horizon (the release's scenarios are ragged), or the block may be misencoded (see
    ``reference_graph``'s ``displaced-block``).
    """

    ENTITY = "entity"
    """The field names a persistent object (§3.2 gives it no relation), so the requirement is an
    entity of ``H`` rather than a hyperedge."""

    UNMAPPED = "unmapped"
    """§3.2 has no row for the field. Counted so the spec's coverage is visible."""


# ---------------------------------------------------------------------------
# Gold evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GoldEvidence:
    """What one question needs from the hypergraph, and whether the hypergraph supplies it."""

    qa_id: str
    template_id: str
    scenario_id: str
    horizon_hours: int
    field: str
    """The template's declared path, after the relative-horizon substitution — e.g.
    ``operation.in_24h.Condition``. This is the path ``groundtruth.resolve_path`` reads."""

    status: EvidenceStatus
    resolution: FieldResolution
    hyperedge_ids: frozenset[str] = frozenset()
    entity_id: str | None = None
    note: str = ""

    @property
    def relations(self) -> tuple[str, ...]:
        return self.resolution.relations

    @property
    def is_scorable(self) -> bool:
        """Whether a recall metric is defined.

        False for the three statuses that have no hyperedge requirement: an unmapped field, a
        persistent-object requirement, and a requirement the graph cannot satisfy. Scoring them
        would put a 0 in the denominator that no retriever could ever move, which is how a
        coverage gap turns into a reported retrieval failure.
        """
        return self.status is EvidenceStatus.RESOLVED

    @property
    def size(self) -> int:
        return len(self.hyperedge_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "qa_id": self.qa_id,
            "template_id": self.template_id,
            "scenario_id": self.scenario_id,
            "horizon_hours": self.horizon_hours,
            "field": self.field,
            "status": self.status.value,
            "relations": list(self.relations),
            "matched_key": self.resolution.matched_key,
            "gold_size": self.size,
            "entity_id": self.entity_id,
            "note": self.note,
        }


def _persistent_entity_id(
    hypergraph: KnowledgeHypergraph, group_id: str, *, name: str, type_name: str
) -> str | None:
    """The id of a group's persistent object, or ``None`` when no hyperedge references one.

    Found by scanning the group's own hyperedges for an entity with the matching name and type
    rather than by reconstructing the §A.2 step-2 id convention. Reconstructing it would mean
    this module owning a second copy of that convention — the one thing
    :mod:`~okh_rag.extraction.normalize` is the authority on — and it would break on any
    extracted graph whose port entity was named slightly differently. The requirement being
    checked is "does the evidence exist in ``H``", and scanning answers exactly that.
    """
    wanted = name.strip().casefold()
    for hyperedge_id in hypergraph.groups.get(group_id, ()):
        edge = hypergraph.hyperedges.get(hyperedge_id)
        if edge is None:
            continue
        for entity_id in edge.entity_ids:
            entity = hypergraph.entities.get(entity_id)
            if entity is None:
                continue
            if entity.type == type_name and entity.name.strip().casefold() == wanted:
                return entity_id
    return None


def gold_for_record(
    record: QARecord,
    template: Template,
    *,
    graph: ReferenceGraph,
) -> GoldEvidence:
    """The evidence requirement of one question, resolved against ``graph``.

    The declared field comes from
    :attr:`~okh_rag.data.cyportqa.Template.static_prefix` with the record's relative horizon
    substituted, which is the same substitution
    :func:`~okh_rag.data.groundtruth.expected_path` performs — so the field named here and the
    field the answer audit reads are the same field, by construction rather than by agreement.

    Args:
        record: The QA record. Its scenario, horizon and relative-horizon tag define *where* the
            evidence has to come from.
        template: The record's template, whose ``answer`` placeholder declares *what*.
        graph: The hypergraph to resolve against.

    Returns:
        A :class:`GoldEvidence`. Never ``None`` and never raising: an unresolvable record is a
        finding, and the report counts findings instead of dropping them.
    """
    field_path = template.static_prefix
    if "<key>" in field_path:
        # Q42 and Q46-Q48. Leaving `<key>` in place when the record carries no tag would make the
        # path unparseable, and `groundtruth.expected_path` deliberately leaves it so that a miss
        # is reported rather than silently reading the wrong sub-key. Here the same choice would
        # turn into a spurious `unmapped` count, so the tag is required and its absence is named.
        if record.relative_horizon is None:
            resolution = relations_for_path(field_path)
            return GoldEvidence(
                qa_id=record.qa_id,
                template_id=record.template_id,
                scenario_id=record.scenario_id,
                horizon_hours=record.context.horizon_hours,
                field=field_path,
                status=EvidenceStatus.UNMAPPED,
                resolution=resolution,
                note="template needs an operation.<key> tag the record does not carry",
            )
        field_path = field_path.replace("<key>", record.relative_horizon)

    resolution = relations_for_path(field_path)
    common = {
        "qa_id": record.qa_id,
        "template_id": record.template_id,
        "scenario_id": record.scenario_id,
        "horizon_hours": record.context.horizon_hours,
        "field": field_path,
        "resolution": resolution,
    }

    if resolution.kind is FieldKind.ENTITY:
        # Q16 and Q19 answer `{senario.port.name}`. §3.1 puts a port under persistent objects, so
        # the requirement is the entity, and it is satisfied once — the port is in every V_e of the
        # scenario, so a hyperedge-level requirement would be "all of them", which measures nothing.
        #
        # Two candidate spellings are tried because the scenario key carries a slugged port
        # (`Port_of_Savannah,_GA`) while the entity holds the encoding's own `port.name`
        # (`Port of Savannah, GA`). The record's context has the second when its arity is 4, and
        # the key has the first always, so trying both covers every record without this module
        # owning a copy of `port_slug` — the one thing `normalize` is the authority on.
        candidates = [record.context.port or "", ScenarioId.parse(record.scenario_id).port]
        entity_id = next(
            (
                found
                for candidate in candidates
                if candidate
                and (found := _persistent_entity_id(
                    graph.hypergraph, record.scenario_id, name=candidate, type_name="PORT"
                ))
            ),
            None,
        )
        return GoldEvidence(
            **common,
            status=EvidenceStatus.ENTITY if entity_id else EvidenceStatus.EMPTY,
            entity_id=entity_id,
            note=(
                "persistent object, not a relation (§3.1, so §3.2 has no row for it)"
                if entity_id
                else "no PORT entity in this group's V_e"
            ),
        )

    if resolution.kind is FieldKind.UNKNOWN:
        return GoldEvidence(
            **common,
            status=EvidenceStatus.UNMAPPED,
            note="no §3.2 row covers this field; CONTRACT.md §3.2 needs a new entry or the field needs routing",
        )

    gold = graph.relation_hyperedges(
        record.scenario_id, record.context.horizon_hours, resolution.relations
    )
    if not gold:
        return GoldEvidence(
            **common,
            status=EvidenceStatus.EMPTY,
            note=(
                "the graph holds no hyperedge with relation(s) "
                f"{list(resolution.relations)} for this group and horizon"
            ),
        )
    return GoldEvidence(**common, status=EvidenceStatus.RESOLVED, hyperedge_ids=gold)


def gold_matches_field_provenance(gold: GoldEvidence, graph: ReferenceGraph) -> bool:
    """Whether a relation-keyed gold set equals the field-keyed one on the reference graph.

    This is the cross-check the module docstring promises: on the reference graph every
    hyperedge records the field it came from, so ``gold`` computed by relation can be compared
    against the set of hyperedges whose recorded field implies one of those relations. On this
    graph the two must agree — each field implies exactly one relation and each within-horizon
    hyperedge comes from exactly one field — so a disagreement means one of the two definitions
    has drifted, and a test asserts this over the whole corpus.

    Returns:
        True when they agree. True also when ``gold`` is not relation-scorable, since there is
        then nothing to compare — the caller checks :attr:`GoldEvidence.status` first when that
        distinction matters.
    """
    if not gold.is_scorable:
        return True
    by_field = frozenset(
        hid
        for hid in gold.hyperedge_ids
        if relation_matches_field(graph.field_of.get(hid, ""), gold.relations)
    )
    return by_field == gold.hyperedge_ids


def relation_matches_field(field_path: str, relations: Sequence[str]) -> bool:
    """Whether ``field_path`` implies any of ``relations``, i.e. it is inside the gold set."""
    if not field_path:
        return False
    resolution = relations_for_path(field_path)
    return bool(set(resolution.relations) & set(relations))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleMetrics:
    """Per-record retrieval quality, relative to the gold evidence.

    Every ratio is ``0.0`` rather than ``None`` when its denominator is empty, except where the
    denominator being empty *means* something different: :attr:`order_concordance` is ``None``
    when fewer than two gold hyperedges survived, because "no pair to compare" is not the same
    fact as "the pair was inverted", and averaging the two as 0.0 would report a corpus of
    single-evidence questions as perfectly mis-ordered.
    """

    gold_size: int
    pool_recall: float
    anchor_recall: float
    trajectory_recall: float
    all_trajectory_recall: float
    trajectory_precision: float

    chance_recall: float
    """``pool_recall × L / |C|`` — what a trajectory of the same length chosen *uniformly at random
    from the pool* would score.

    This is the floor :attr:`trajectory_recall` has to be read against, and without it the number
    is uninterpretable in both directions. A recall of 0.03 sounds like a failure until one
    notices that with ``L = 8`` and ``|C| = 150`` a random draw scores 0.053 × pool_recall, so
    0.03 is *at* chance rather than slightly above it. Conversely a recall of 0.20 on a corpus
    where the pool barely contains any gold is a stronger result than the same 0.20 where it
    contains all of it. The pool and the trajectory length are the two quantities that set the
    floor, and both are already on this record.
    """

    first_gold_step: int | None
    """1-based position of the earliest gold hyperedge in the best trajectory, or ``None`` when
    none of them is present. This is the cheapest order-aware reading: a trajectory that holds
    the right evidence at step 7 and one that holds it at step 1 have identical recall, and the
    §A.8 evidence block the generator reads does not treat them alike."""

    canonical_concordance: float | None
    """Agreement between the trajectory's order and §A.3's canonical order, over every pair of its
    members that the canonical ordering places.

    Defined without reference to gold, which is what makes it usable: a message-passing metric
    restricted to gold hyperedges is undefined whenever fewer than two of them are retrieved, and
    on a corpus where recall is a few percent that is almost always — measured, not assumed: on the
    diagnostic run it had a non-empty denominator for 2 of 120 records. This one has one whenever
    the trajectory holds two placed members, so it always measures something.

    §4.5's shuffle row is exactly what it is for. Shuffling ``L`` members gives expected agreement
    ``0.5``, while an order-aware beam that respects §A.3 approaches ``1.0`` — so the metric
    separates the rows that §A.6's ``Prec`` and ``Ovlp`` are defined to separate.
    """

    gold_order_concordance: float | None
    """The same agreement restricted to the trajectory's gold members.

    Kept beside :attr:`canonical_concordance` because the two answer different questions: this one
    is "is the *needed* evidence in §A.3's order", the other "is the trajectory as a whole". It is
    frequently ``None`` at realistic recall, and that is a fact about the measurement rather than
    about the retriever, which is why it is reported with its own denominator.
    """

    def as_dict(self) -> dict[str, Any]:
        return {
            "gold_size": self.gold_size,
            "pool_recall": round(self.pool_recall, 4),
            "anchor_recall": round(self.anchor_recall, 4),
            "trajectory_recall": round(self.trajectory_recall, 4),
            "all_trajectory_recall": round(self.all_trajectory_recall, 4),
            "trajectory_precision": round(self.trajectory_precision, 4),
            "chance_recall": round(self.chance_recall, 4),
            "first_gold_step": self.first_gold_step,
            "canonical_concordance": (
                None if self.canonical_concordance is None else round(self.canonical_concordance, 4)
            ),
            "gold_order_concordance": (
                None if self.gold_order_concordance is None else round(self.gold_order_concordance, 4)
            ),
        }


@dataclass(frozen=True)
class OracleOutcome:
    """One record's evidence requirement joined to one retrieval result."""

    gold: GoldEvidence
    metrics: OracleMetrics | None
    """``None`` when the record is not scorable — see :attr:`GoldEvidence.is_scorable`."""

    trajectory_size: int = 0
    pool_size: int = 0
    variant: str = ""
    inference: str = ""

    @property
    def qa_id(self) -> str:
        return self.gold.qa_id

    @property
    def template_id(self) -> str:
        return self.gold.template_id

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.gold.as_dict(),
            "pool_size": self.pool_size,
            "trajectory_size": self.trajectory_size,
            "variant": self.variant,
            "inference": self.inference,
            "metrics": None if self.metrics is None else self.metrics.as_dict(),
        }


def _recall(found: Iterable[str], gold: frozenset[str]) -> float:
    if not gold:
        return 0.0
    return len(gold.intersection(found)) / len(gold)


def _order_concordance(
    sequence: Sequence[str],
    reference: Mapping[str, int],
    *,
    restrict_to: frozenset[str] | None = None,
) -> float | None:
    """Agreement between a sequence's order and a reference ordering, as a pair-agreement rate.

    Restricted to members present in both, and to pairs both of which ``reference`` places. A
    hyperedge the reference does not place is one §A.3 never ordered — a family-13 change edge, or
    a member of another group — and excluding it rather than assuming a position keeps the metric
    from inventing an order to agree with.

    Args:
        sequence: The retrieved trajectory, best-first.
        reference: A total order over hyperedge ids, as a position map.
        restrict_to: When given, only members of this set count. Used to score the gold ordering
            separately from the trajectory's ordering as a whole.

    Returns:
        The fraction of comparable pairs whose relative order matches, or ``None`` when there are
        no such pairs. ``None`` distinguishes "nothing to compare" from "the comparison failed",
        which averaging as ``0.0`` would conflate.
    """
    present = [
        hid
        for hid in sequence
        if hid in reference and (restrict_to is None or hid in restrict_to)
    ]
    if len(present) < 2:
        return None
    positions = [reference[hid] for hid in present]
    agree = total = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            total += 1
            if positions[i] < positions[j]:
                agree += 1
    return agree / total if total else None


def evaluate_record(
    record: QARecord,
    template: Template,
    result: RetrievalResult,
    *,
    graph: ReferenceGraph,
    canonical_position: Mapping[str, int] | None = None,
) -> OracleOutcome:
    """Score one retrieval result against one question's gold evidence.

    Args:
        record: The QA record the result was produced for.
        template: Its template.
        result: What §A.7 returned, including the pool and the term scores.
        graph: The hypergraph the result was retrieved from.
        canonical_position: ``hyperedge id -> §A.3 canonical position``, from
            :func:`~okh_rag.precedence.trajectory_order_index`. Omitted, the order metrics are
            not computed and :attr:`OracleMetrics.order_concordance` is ``None``.

    Returns:
        An :class:`OracleOutcome`. Unscorable records come back with ``metrics=None`` and the
        gold status that made them unscorable, so the caller can count them by reason.
    """
    gold = gold_for_record(record, template, graph=graph)
    sizes = {
        "trajectory_size": len(result.trajectories[0]) if result.trajectories else 0,
        "pool_size": len(result.pool),
        "variant": result.variant,
        "inference": result.inference,
    }
    if not gold.is_scorable:
        return OracleOutcome(gold=gold, metrics=None, **sizes)

    pool = frozenset(result.pool.hyperedge_ids)
    anchors = frozenset(result.pool.anchors)
    best = result.best()
    best_ids = tuple(best.hyperedge_ids) if best is not None else ()
    union_of_trajectories = frozenset(hid for traj in result.trajectories for hid in traj)

    first_gold_step: int | None = None
    for step, hid in enumerate(best_ids, start=1):
        if hid in gold.hyperedge_ids:
            first_gold_step = step
            break

    pool_recall = _recall(pool, gold.hyperedge_ids)
    metrics = OracleMetrics(
        gold_size=gold.size,
        pool_recall=pool_recall,
        anchor_recall=_recall(anchors, gold.hyperedge_ids),
        trajectory_recall=_recall(best_ids, gold.hyperedge_ids),
        all_trajectory_recall=_recall(union_of_trajectories, gold.hyperedge_ids),
        trajectory_precision=(
            len(gold.hyperedge_ids.intersection(best_ids)) / len(best_ids) if best_ids else 0.0
        ),
        # Expected recall of `len(best_ids)` members drawn uniformly from the pool. Derived rather
        # than sampled: E[|gold ∩ γ|] = |gold ∩ C| · L/|C| exactly, so the floor is
        # pool_recall · L/|C| with no Monte-Carlo noise of its own.
        chance_recall=pool_recall * (len(best_ids) / len(pool) if len(pool) else 0.0),
        first_gold_step=first_gold_step,
        canonical_concordance=(
            _order_concordance(best_ids, canonical_position)
            if canonical_position is not None
            else None
        ),
        gold_order_concordance=(
            _order_concordance(best_ids, canonical_position, restrict_to=gold.hyperedge_ids)
            if canonical_position is not None
            else None
        ),
    )
    return OracleOutcome(gold=gold, metrics=metrics, **sizes)


# ---------------------------------------------------------------------------
# Template coverage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemplateCoverage:
    """Whether §3.2 can place every template's declared field — a static property of the spec."""

    resolved: Mapping[str, str]
    """``template id -> relation`` for templates whose field names a relation."""

    entities: tuple[str, ...]
    """Template ids whose field names a persistent object rather than a relation (Q16, Q19)."""

    unmapped: Mapping[str, str]
    """``template id -> declared field`` for templates §3.2 does not cover."""

    @property
    def total(self) -> int:
        return len(self.resolved) + len(self.entities) + len(self.unmapped)

    @property
    def is_total(self) -> bool:
        """True when every template is accounted for."""
        return not self.unmapped

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "resolved": len(self.resolved),
            "entity": len(self.entities),
            "unmapped": len(self.unmapped),
            "by_relation": dict(sorted(self.resolved.items(), key=lambda kv: int(kv[0][1:]))),
            "unmapped_templates": dict(self.unmapped),
        }


def template_coverage(templates: Mapping[str, Template]) -> TemplateCoverage:
    """Resolve every template's declared field against §3.2.

    Static and model-free: no corpus is loaded and no graph is built, because the question is
    whether the *spec* covers the *templates*. If a template's field is unmapped the oracle
    cannot score it on any hypergraph, so this runs before a retrieval run does any work.
    """
    resolved: dict[str, str] = {}
    entities: list[str] = []
    unmapped: dict[str, str] = {}
    for template_id, template in templates.items():
        path = template.static_prefix
        resolution = relations_for_path(path)
        if resolution.kind is FieldKind.ENTITY:
            entities.append(template_id)
        elif resolution.kind is FieldKind.RELATION and resolution.relation is not None:
            resolved[template_id] = resolution.relation
        elif resolution.kind is FieldKind.RELATION:
            # A container with several relations. Q33 is the only one on this release, and its
            # single child relation means it lands in the branch above; this arm exists so a
            # future multi-relation container is reported as resolved-but-composite rather than
            # silently dropped into `unmapped`.
            resolved[template_id] = "+".join(resolution.relations)
        else:
            unmapped[template_id] = path
    return TemplateCoverage(
        resolved=resolved, entities=tuple(sorted(entities, key=lambda t: int(t[1:]))), unmapped=unmapped
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class OracleReport:
    """Corpus-level aggregate of :class:`OracleOutcome`, plus the counters that qualify it."""

    outcomes: list[OracleOutcome] = field(default_factory=list)
    graph_stats: Mapping[str, Any] = field(default_factory=dict)
    coverage: Mapping[str, Any] = field(default_factory=dict)
    backend: str = ""

    # -- counting -----------------------------------------------------------

    @property
    def scored(self) -> list[OracleOutcome]:
        return [o for o in self.outcomes if o.metrics is not None]

    def status_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(o.gold.status.value for o in self.outcomes).items()))

    def mean_metrics(self) -> dict[str, float | None]:
        """Mean of each metric over the scorable records.

        ``first_gold_step`` and the two concordances average over the records where they are
        *defined* rather than over all of them, and each reports ``None`` when nothing is defined,
        with its own denominator alongside — see :class:`OracleMetrics` on why an empty
        denominator must not read as 0.0.
        """
        scored = [o.metrics for o in self.outcomes if o.metrics is not None]
        if not scored:
            return {}

        ratios = (
            "pool_recall",
            "anchor_recall",
            "trajectory_recall",
            "all_trajectory_recall",
            "trajectory_precision",
            "chance_recall",
        )
        out: dict[str, float | None] = {
            key: round(sum(getattr(m, key) for m in scored) / len(scored), 4) for key in ratios
        }

        # The gap between the two is the part of retrieval that is *not* explained by the pool and
        # the trajectory length. Negative means the beam did worse than drawing at random, which on
        # this harness is a statement about the embedder rather than about §A.7.
        out["trajectory_recall_over_chance"] = round(
            out["trajectory_recall"] / out["chance_recall"], 3) if out["chance_recall"] else None

        located = [m.first_gold_step for m in scored if m.first_gold_step is not None]
        out["first_gold_step_mean"] = round(sum(located) / len(located), 3) if located else None
        out["gold_fully_retrieved_rate"] = round(
            sum(1 for m in scored if m.trajectory_recall >= 1.0) / len(scored), 4
        )

        for key in ("canonical_concordance", "gold_order_concordance"):
            defined = [getattr(m, key) for m in scored if getattr(m, key) is not None]
            out[f"{key}_mean"] = round(sum(defined) / len(defined), 4) if defined else None
            out[f"{key}_n"] = len(defined)
        return out

    def by_template(self) -> dict[str, dict[str, Any]]:
        """Per-template means, so a spec gap is visible at the template that has it."""
        buckets: dict[str, list[OracleOutcome]] = defaultdict(list)
        for outcome in self.scored:
            buckets[outcome.template_id].append(outcome)
        out: dict[str, dict[str, Any]] = {}
        for template_id, items in buckets.items():
            metrics = [o.metrics for o in items if o.metrics is not None]
            relation = items[0].gold.relations[0] if items[0].gold.relations else ""
            out[template_id] = {
                "n": len(items),
                "trajectory_recall": round(sum(m.trajectory_recall for m in metrics) / len(metrics), 4),
                "pool_recall": round(sum(m.pool_recall for m in metrics) / len(metrics), 4),
                "gold_mean_size": round(sum(m.gold_size for m in metrics) / len(metrics), 3),
                "relation": relation,
                "family": relation_family(relation),
            }
        return dict(sorted(out.items(), key=lambda kv: int(kv[0][1:])))

    def by_horizon(self) -> dict[int, dict[str, Any]]:
        """Per-horizon means — the release's scenarios are ragged, so coverage varies with T-N."""
        buckets: dict[int, list[OracleMetrics]] = defaultdict(list)
        for outcome in self.outcomes:
            if outcome.metrics is not None:
                buckets[outcome.gold.horizon_hours].append(outcome.metrics)
        return {
            horizon: {
                "n": len(items),
                "trajectory_recall": round(sum(m.trajectory_recall for m in items) / len(items), 4),
                "pool_recall": round(sum(m.pool_recall for m in items) / len(items), 4),
                "gold_mean_size": round(sum(m.gold_size for m in items) / len(items), 3),
            }
            for horizon, items in sorted(buckets.items())
        }

    def empty_gold_by_cause(self) -> dict[str, int]:
        """Why gold was empty, split so a data gap is not read as a retrieval failure."""
        counts: Counter[str] = Counter()
        for outcome in self.outcomes:
            if outcome.gold.status is EvidenceStatus.EMPTY:
                counts[outcome.gold.note] += 1
        return dict(counts.most_common())

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "n_records": len(self.outcomes),
            "n_scored": len(self.scored),
            "status_counts": self.status_counts(),
            "mean_metrics": self.mean_metrics(),
            "empty_gold_by_cause": self.empty_gold_by_cause(),
            "by_template": self.by_template(),
            "by_horizon": self.by_horizon(),
            "reference_graph": dict(self.graph_stats),
            "template_coverage": dict(self.coverage),
        }
