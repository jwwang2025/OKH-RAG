"""The deterministic hypergraph the oracle scores against, built straight from the encoding.

§A.2 extracts ``H`` by asking a model to read a scenario document and return facts. For
CyPortQA that step is *checkable*, because the same scenario also ships in structured form:
``Encoded_senario.json`` already separates the four §4.1 blocks, already keys them by the
ten horizons of §A.3, and §3.2 of ``data/CONTRACT.md`` names the §A.1 Table 3 relation each
field implies. So "what would a perfect extractor have produced?" has a closed-form answer
that calls no model:

    one hyperedge per (scenario, horizon, populated field), with the relation §3.2 gives it.

That is what this module builds. It is not a substitute for §A.2 — it cannot be, since the
real pipeline's hyperedges come from model-written evidence sentences that may merge two
fields, split one, or be reworded. It is the *yardstick* for that pipeline, and two things
follow from using it:

1. **The oracle can run with no API key.** Extraction is the only non-deterministic step before
   retrieval, so removing it means retrieval quality can be measured from the release alone.
2. **Extraction becomes scoreable in the same currency as retrieval.** An extracted hypergraph
   and this one are two :class:`~okh_rag.schema.KnowledgeHypergraph` objects with the same
   groups and the same horizon blocks, so recall of this graph's hyperedges is a direct
   measurement of extraction.

**Steps 2-5 are the real ones.** This module emits raw hyperedges with provisional entity ids
and hands them to :func:`~okh_rag.extraction.normalize.normalize_hypergraph`, so the canonical
``family:storm:port:horizon`` ids, the §A.2 step-3 horizon anchors, the step-4 cross-horizon
change edges and the step-5 dedup are all produced by the same code the LLM path uses. Writing
a second implementation here would have been the easier option and the wrong one: the oracle
would then be measuring a hypergraph that no run of the pipeline ever produces.

**Entity naming, and why the field label is the name.** §A.1 defines ``v = (n_v, τ_v, d_v,
c_v)`` with ``n_v ⊆ s_e`` — the name has to occur in the fact's own evidence. The name chosen
here is the field's path with its block prefix stripped (``weather_observation.under_watch`` →
``under_watch``, ``operation.in_12h.Condition`` → ``in_12h.Condition``), and the evidence is
rendered to contain it. That choice, rather than the field's *value*, is load-bearing:

* a *transient state* is a "horizon-specific instance" (§A.1), and one field at one horizon is
  exactly one such instance — naming by value would collapse ``under_watch`` and ``watch_coast``
  into a single entity whenever both read ``"None"``, and step 5 would then merge their two
  hyperedges into one, because their relation is the same;
* the value still has to be reachable, so it goes into ``evidence`` and into
  ``attributes["value"]``, which is also where a reader looks to see what the fact actually says.

**Field provenance is recorded but is not the oracle's key.** Every hyperedge carries
``attributes["field"]``. The oracle keys gold sets on *relation* instead, because an extracted
hypergraph has no field attribute to key on — see :mod:`okh_rag.evaluation.field_map`. Keeping
the attribute makes the two definitions cross-checkable, and
:mod:`~okh_rag.evaluation.oracle` does exactly that on this graph.

**What this graph cannot contain.** A field §3.2 does not cover gets no hyperedge, because
inventing a relation for it would put a fabricated fact into the yardstick. Three classes of
field are dropped and counted rather than dropped silently:

* ``field_map.FieldKind.UNKNOWN`` paths — ``cyclone_condition_and_forecast.name`` is the
  systematic one, and it is correctly absent: the storm is a *persistent object*, an entity of
  every ``V_e``, not the subject of a relation;
* **stray keys** — the release carries model commentary as *keys* in four (scenario, block)
  pairs, e.g. ``cyclone_condition_and_forecast."// once inland Beta is not expected to
  re-emerge near Corpus Christi's coastline"``. See ``KNOWN_STRAY_KEY_BLOCKS``;
* **a displaced block** — in one (scenario, horizon) the whole ``weather_observation`` mapping
  is nested *inside* ``cyclone_condition_and_forecast`` while the real block is empty, so 24
  weather fields arrive under the wrong prefix. Counting these separately matters because
  without it the symptom is "the question's gold set is empty", which is indistinguishable from
  a field that is genuinely absent — and those two need different responses.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Mapping

from okh_rag.config import Config
from okh_rag.data.cyportqa import (
    TIME_BLOCK_FIELDS,
    CyPortQACorpus,
    EncodedScenario,
    TimeBlock,
)
from okh_rag.evaluation.field_map import FieldKind, relations_for_path
from okh_rag.extraction.normalize import normalize_hypergraph
from okh_rag.schema import Entity, Hyperedge, KnowledgeHypergraph

__all__ = [
    "BLOCK_PREFIXES",
    "RELATION_ENTITY_TYPE",
    "ReferenceGraph",
    "ReferenceGraphStats",
    "build_reference_graph",
    "count_group_id_collisions",
    "iter_field_instances",
]

BLOCK_PREFIXES: Final[tuple[str, ...]] = TIME_BLOCK_FIELDS
"""The four §4.1 blocks, in the order a scenario document is read."""

RELATION_ENTITY_TYPE: Final[Mapping[str, str]] = {
    # §A.1's type system, with each of §3.2's relations mapped onto the type whose category it
    # belongs to. Every value is a key of `schema.ENTITY_TYPE_REGISTRY`; a test asserts that, so a
    # typo cannot quietly demote a state to the unknown-type roll-up.
    "has_cyclone_state": "CYCLONE_STATE",
    "has_category_state": "CYCLONE_STATE",
    "has_motion": "CYCLONE_STATE",
    "forecasts_track": "TRACK_FORECAST",
    "forecasts_landfall": "LANDFALL_FORECAST",
    "has_hours_to_landfall": "TIMING_MARKER",
    "has_forecast_window": "TIMING_MARKER",
    "has_watch_status": "ADVISORY_STATUS",
    "has_warning_status": "ADVISORY_STATUS",
    "has_leadtime_probability": "PROBABILITY_TIMING",
    "has_cumulative_probability": "PROBABILITY_FORECAST",
    "forecasts_hazard_at_horizon": "HAZARD_FORECAST",
    "observes_hazard_at_horizon": "HAZARD_OBSERVATION",
    "has_threshold_status": "THRESHOLD_STATUS",
    "has_additional_hazard": "HAZARD_FORECAST",
    "has_operation_status": "OPERATION_STATUS",
    "affects_vessel_handling": "OPERATION_STATUS",
    "has_impact_prediction": "IMPACT_TIMELINE",
    "causes_operational_disruption": "IMPACT_TIMELINE",
    "has_recovery_status": "RECOVERY_STATUS",
}
"""``relation -> §A.1 entity type`` for the transient state each field introduces.

Families 6 and 9 share ``HAZARD_FORECAST`` because both are hazard statements about the horizon
rather than about the storm; family 7 gets ``HAZARD_OBSERVATION``, which is the distinction
§A.6 Eq. 13's six-member phase set exists to preserve.
"""


# ---------------------------------------------------------------------------
# Field instances
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldInstance:
    """One populated field of one horizon block — the unit a reference hyperedge is built from."""

    scenario_id: str
    horizon_hours: int
    path: str
    """Full dotted path, block prefix included, e.g. ``weather_observation.under_watch``."""

    label: str
    """The path with its block prefix stripped — the transient entity's name and evidence subject."""

    value: Any
    text_position: int
    """Monotone enumeration offset within the scenario, used as §A.3's last tie-breaker."""

    @property
    def is_empty(self) -> bool:
        """True when the field is present but carries no value.

        §A.1 needs an evidence string and ``groundtruth`` treats a stored ``None`` as
        answerable-but-degenerate rather than as a missing field, so the hyperedge is still
        built; the count is reported so a reader can see how much of the corpus is degenerate.
        """
        return self.value is None or (isinstance(self.value, str) and not self.value.strip())


def iter_field_instances(scenario: EncodedScenario) -> tuple[list[FieldInstance], list[str]]:
    """Every populated field of one scenario, plus the stray keys found on the way.

    Walks each block depth-first, so a nested mapping (``motion.direction``,
    ``gale_probability."in 12h"``) becomes one instance per leaf and the path is the accessor
    that would retrieve it. Enumeration order is the encoding's own insertion order, which is
    what makes :attr:`FieldInstance.text_position` a reading-order quantity rather than a
    hash of anything.

    Args:
        scenario: The encoding to walk.

    Returns:
        ``(instances, anomalies)``. ``anomalies`` holds human-readable descriptions of every
        structural problem hit, prefixed by kind, so the caller can count them by prefix
        without re-deriving what they mean.
    """
    instances: list[FieldInstance] = []
    anomalies: list[str] = []
    position = 0

    for horizon in scenario.horizons:
        block: TimeBlock = scenario.time[horizon]
        blocks = block.blocks()
        for prefix in BLOCK_PREFIXES:
            payload = blocks[prefix]
            if not isinstance(payload, Mapping):
                anomalies.append(f"block-not-mapping:{prefix}={type(payload).__name__}")
                continue
            # A block name appearing as a *key* of another block means the encoder nested one
            # whole block inside another. This has to be detected at the mapping level, because
            # `_walk` yields only leaves — the intermediate key would otherwise never be seen as
            # a leaf and the misplacement would surface only as "these 24 paths are unmapped",
            # which is indistinguishable from 24 independently unknown fields.
            for nested in payload:
                if nested in BLOCK_PREFIXES and nested != prefix:
                    anomalies.append(f"displaced-block:{prefix}.{nested}")
            for key, value in _walk(payload, prefix):
                # Test the *last* segment, which is the only thing that names a field:
                # `cyclone_condition_and_forecast._comment` is a well-formed path whose final
                # segment is the problem, so testing the whole path would never fire.
                leaf = key.rsplit(".", 1)[-1]
                if not _looks_like_field_name(leaf):
                    anomalies.append(f"stray-key:{leaf}")
                instances.append(
                    FieldInstance(
                        scenario_id=str(scenario.scenario_id),
                        horizon_hours=horizon,
                        path=key,
                        label=key.split(".", 1)[1] if "." in key else key,
                        value=value,
                        text_position=position,
                    )
                )
                position += 1

    return instances, anomalies


def _walk(node: Any, prefix: str) -> Iterable[tuple[str, Any]]:
    """Yield ``(dotted path, leaf value)`` pairs, recursing through mappings.

    A list is treated as a leaf. None of the four blocks contains one in the release — a
    measurement, not an assumption, and it is re-measured by ``scripts/audit_cyportqa.py``.
    """
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield from _walk(value, f"{prefix}.{key}")
    else:
        yield prefix, node


def _looks_like_field_name(key: str) -> bool:
    """Whether ``key`` could name a field, i.e. is not model commentary lifted into a key.

    The release stores two kinds of non-field key, both in the four blocks listed in the module
    docstring: a bare comment (``_comment``, ``_note``) and a prose sentence a model wrote
    instead of a field name, which always begins with a punctuation mark or a ``//``/``--``
    marker. The test is deliberately narrow — the first character must be alphanumeric — because
    a false positive drops a real fact from the yardstick, which is worse than counting one
    stray key as a field and letting §3.2's coverage check reject it anyway.
    """
    return bool(key) and key[0].isalnum()


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


@dataclass
class ReferenceGraphStats:
    """Counters for one build, so the report can state what the yardstick is made of."""

    scenarios: int = 0
    horizon_blocks: int = 0
    fields_seen: int = 0
    fields_kept: int = 0
    fields_empty_value: int = 0
    fields_unmapped: Counter[str] = field(default_factory=Counter)
    anomalies: Counter[str] = field(default_factory=Counter)
    raw_hyperedges: int = 0
    hyperedges: int = 0
    entities: int = 0
    relations: Counter[str] = field(default_factory=Counter)
    id_collisions: dict[str, int] = field(default_factory=dict)
    """Output of :func:`count_group_id_collisions` on the merged graph. Expected to be all zeros."""

    @property
    def coverage(self) -> float:
        """Share of the fields seen that §3.2 could place, i.e. that became hyperedges."""
        return self.fields_kept / self.fields_seen if self.fields_seen else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenarios": self.scenarios,
            "horizon_blocks": self.horizon_blocks,
            "fields_seen": self.fields_seen,
            "fields_kept": self.fields_kept,
            "fields_empty_value": self.fields_empty_value,
            "fields_dropped_unmapped": sum(self.fields_unmapped.values()),
            "unmapped_paths": dict(self.fields_unmapped.most_common()),
            "coverage": round(self.coverage, 4),
            "anomalies": dict(self.anomalies.most_common()),
            "raw_hyperedges": self.raw_hyperedges,
            "hyperedges": self.hyperedges,
            "entities": self.entities,
            "id_collisions": dict(self.id_collisions),
            "hyperedges_by_relation": dict(self.relations.most_common()),
        }


@dataclass(frozen=True)
class ReferenceGraph:
    """The yardstick, plus the counters that say what it is a yardstick *of*."""

    hypergraph: KnowledgeHypergraph
    stats: ReferenceGraphStats
    field_of: Mapping[str, str]
    """``hyperedge id -> the §3.2 field path it was built from``.

    Only hyperedges that came from a field appear: a §A.2 step-4 cross-horizon change edge has
    no source field. The oracle keys gold sets on relation and uses this map to *check* itself,
    so a divergence between the two definitions fails a test instead of being averaged away.
    """

    by_relation: Mapping[tuple[str, int, str], frozenset[str]]
    """``(group id, horizon hours, relation) -> hyperedge ids``.

    Precomputed because the oracle asks this question once per question, and the benchmark has
    117,178 of them: scanning a group's ~500 hyperedges per call would make gold-set
    computation quadratic in the corpus for no reason. §A.7's own argument for scoping applies
    here in miniature — the index is query-independent, so it is built once.
    """

    def horizon_hyperedges(self, group_id: str, horizon_hours: int) -> tuple[str, ...]:
        """Ids of the within-horizon hyperedges of one group at one horizon."""
        return tuple(
            hid
            for hid in self.hypergraph.groups.get(group_id, ())
            if (edge := self.hypergraph.hyperedges.get(hid)) is not None
            and edge.horizon_hours == horizon_hours
        )

    def relation_hyperedges(
        self, group_id: str, horizon_hours: int, relations: Iterable[str]
    ) -> frozenset[str]:
        """Ids of one group's hyperedges at one horizon whose relation is in ``relations``.

        The union over ``relations`` rather than a single lookup, because a container field
        (§3.2's ``gale_probability``) can imply more than one — see
        :func:`~okh_rag.evaluation.field_map.relations_for_path`.
        """
        found: set[str] = set()
        for relation in relations:
            found.update(self.by_relation.get((group_id, horizon_hours, relation), ()))
        return frozenset(found)


def _build_relation_index(
    hypergraph: KnowledgeHypergraph,
) -> dict[tuple[str, int, str], frozenset[str]]:
    """Invert the hypergraph into the ``(group, horizon, relation)`` index the oracle reads."""
    buckets: dict[tuple[str, int, str], set[str]] = {}
    for hyperedge_id, edge in hypergraph.hyperedges.items():
        if edge.horizon_hours is None:
            continue  # a family-13 change edge belongs to no horizon, so it can never be gold
        key = (edge.group_id, edge.horizon_hours, edge.relation)
        buckets.setdefault(key, set()).add(hyperedge_id)
    return {key: frozenset(value) for key, value in buckets.items()}


def build_reference_graph(
    corpus: CyPortQACorpus,
    *,
    cfg: Config,
    include_cross_horizon: bool = True,
    scenario_ids: Iterable[str] | None = None,
) -> ReferenceGraph:
    """Build the reference hypergraph for ``corpus``.

    Args:
        corpus: The loaded release. Only :attr:`~okh_rag.data.cyportqa.CyPortQACorpus.encoded`
            is read — the QA records are the oracle's business, not the yardstick's.
        cfg: The root config. ``hypergraph`` drives relation normalisation, the §A.2 step-2 id
            convention, horizon injection, cross-horizon synthesis and dedup.
        include_cross_horizon: Whether to run §A.2 step 4. On, the graph carries the family-13
            change edges that §4.2 reports as "nearly half" of ``H``; they are never gold
            (§3.2 gives family 13 no source field) but they compete for the pool exactly as they
            do in a real run, so off is only useful for isolating a pool-recall question from
            corpus size.
        scenario_ids: Restrict the build to these knowledge groups. ``None`` builds all 472.

    Returns:
        A :class:`ReferenceGraph`. Groups are keyed by scenario id, matching
        :attr:`~okh_rag.data.cyportqa.QARecord.group_id`.
    """
    stats = ReferenceGraphStats()
    merged = KnowledgeHypergraph()
    field_of: dict[str, str] = {}

    wanted = set(scenario_ids) if scenario_ids is not None else None
    for scenario_id in sorted(corpus.encoded):
        if wanted is not None and scenario_id not in wanted:
            continue
        scenario = corpus.encoded[scenario_id]
        document, document_stats, document_field_of = _build_scenario(scenario, cfg=cfg)
        stats.scenarios += 1
        stats.horizon_blocks += len(scenario.horizons)
        stats.fields_seen += document_stats["fields_seen"]
        stats.fields_kept += document_stats["fields_kept"]
        stats.fields_empty_value += document_stats["fields_empty_value"]
        stats.raw_hyperedges += document_stats["raw_hyperedges"]
        stats.fields_unmapped.update(document_stats["unmapped"])
        stats.anomalies.update(document_stats["anomalies"])
        merged.merge(document)
        field_of.update(document_field_of)

    if not include_cross_horizon:
        _drop_cross_horizon(merged)

    for edge in merged.hyperedges.values():
        stats.relations[edge.relation] += 1
    stats.hyperedges = len(merged.hyperedges)
    stats.entities = len(merged.entities)
    stats.id_collisions = count_group_id_collisions(merged)

    return ReferenceGraph(
        hypergraph=merged,
        stats=stats,
        field_of=field_of,
        by_relation=_build_relation_index(merged),
    )


def _drop_cross_horizon(hypergraph: KnowledgeHypergraph) -> None:
    """Remove every family-13 hyperedge, in place, rebuilding the group index afterwards.

    The group buckets have to be rebuilt rather than filtered because an id may legitimately appear
    in a bucket it no longer belongs to — see :meth:`KnowledgeHypergraph.add_hyperedge`, whose
    overwrite-on-collision behaviour is exactly what
    :func:`count_group_id_collisions` measures on the merged graph.
    """
    from okh_rag.vocabulary import is_cross_horizon

    for hid in [h for h, e in hypergraph.hyperedges.items() if is_cross_horizon(e.relation)]:
        del hypergraph.hyperedges[hid]
    hypergraph.groups = {}
    for hyperedge in hypergraph.hyperedges.values():
        if hyperedge.group_id:
            hypergraph.groups.setdefault(hyperedge.group_id, []).append(hyperedge.hyperedge_id)


def count_group_id_collisions(hypergraph: KnowledgeHypergraph) -> dict[str, int]:
    """Measure ids claimed by more than one group, i.e. aliasing across scenarios.

    §A.2 step 5 makes a hyperedge id a hash of ``(relation, V_e, s_e)``, and §A.2 step 2 makes
    ``V_e`` canonical over ``{family}:{storm}:{port}:{horizon}``. Two scenarios therefore collide
    exactly when their canonical entity ids do, which for this release means **the same storm *name*
    at the same port in two different years** — the id has no year component. The collision is
    silent in the worst way: :meth:`~okh_rag.schema.KnowledgeHypergraph.add_hyperedge` lets the
    second record overwrite the first, while both groups keep listing the id, so the corpus loses a
    fact *and* the survivors get attributed to the wrong knowledge group.

    This returns the counts rather than asserting zero, because the number is a property of the
    corpus and the id convention together and belongs in the report next to the graph it describes.
    :func:`build_reference_graph` qualifies the storm with its year precisely to drive it to zero;
    a non-zero reading means the convention and the corpus have drifted apart again.

    Returns:
        ``{"ids": aliased ids, "groups": distinct groups involved, "stale": bucket entries whose
        edge now belongs to another group}``.
    """
    owner: dict[str, set[str]] = {}
    for group_id, member_ids in hypergraph.groups.items():
        for hid in member_ids:
            owner.setdefault(hid, set()).add(group_id)
    aliased = {hid: groups for hid, groups in owner.items() if len(groups) > 1}
    stale = sum(
        1
        for group_id, member_ids in hypergraph.groups.items()
        for hid in member_ids
        if (edge := hypergraph.hyperedges.get(hid)) is not None and edge.group_id != group_id
    )
    return {
        "ids": len(aliased),
        "groups": len({g for groups in aliased.values() for g in groups}),
        "stale": stale,
    }


def _build_scenario(
    scenario: EncodedScenario, *, cfg: Config
) -> tuple[KnowledgeHypergraph, dict[str, Any], dict[str, str]]:
    """Build one scenario's hypergraph, running the real §A.2 steps 2-5 over it."""
    instances, anomalies = iter_field_instances(scenario)
    stats: dict[str, Any] = {
        "fields_seen": len(instances),
        "fields_kept": 0,
        "fields_empty_value": 0,
        "raw_hyperedges": 0,
        "unmapped": Counter(),
        "anomalies": Counter(anomalies),
    }
    field_of: dict[str, str] = {}

    # The encoding's own names, falling back to the scenario key when a name is blank. Two strings
    # come out of this and they are used for different things:
    #
    # * `storm`/`port` are the *identity* components of §A.2 step 2's canonical id;
    # * `storm_label` is the display name that goes into `n_v` and into the rendered evidence.
    #
    # They differ for the storm, and the difference is load-bearing. §A.2's id template is
    # `{family}:{storm}:{port}:{horizon}` with the example `wind_fcst:IRMA:port_arthur:T-48` — a bare
    # storm name, no year. That is unambiguous only while storm names are unique, and CyPortQA's key
    # space is `STORM_YEAR_PORT`: 69 storms over 9 years, in which names repeat. `CLAUDETTE` is both
    # a 2015 and a 2021 storm, and both made port at Baltimore, so a bare name makes their canonical
    # entity ids identical — and therefore their §A.2 step-5 hyperedge hashes identical whenever the
    # encoded values happen to coincide. The consequence is silent: `add_hyperedge` overwrites, one
    # scenario's fact disappears, and its group keeps listing an id whose edge now belongs to the
    # other scenario, which is what §A.7's group-aware expansion and §A.6 Eq. 12 then match across.
    # Qualifying the storm with its year removes the aliasing without touching the port, which must
    # stay shared: `HypergraphIndex.neighbours` depends on "a port entity is shared by every storm
    # that affected that port". See ``CONTRACT.md`` §3.4.
    storm_label = str(scenario.storm_name or scenario.scenario_id.storm)
    storm = f"{scenario.scenario_id.storm}_{scenario.scenario_id.year}"
    port = str(scenario.port_name or scenario.scenario_id.port)
    hypergraph = KnowledgeHypergraph()

    # §A.1's two persistent objects. One entity each for the whole scenario — §A.2 step 2 gives
    # them no horizon component, which is what lets §A.6 Eq. 12's Jaccard term match a fact at
    # T-120 against one at T-12. They are registered only once a fact has been kept, so a
    # scenario whose every field is unmapped leaves no orphan behind.
    anchors = (
        ("persistent:port", port, "PORT"),
        ("persistent:cyclone", storm_label, "CYCLONE"),
    )

    for instance in instances:
        resolution = relations_for_path(instance.path)
        if resolution.kind is not FieldKind.RELATION:
            stats["unmapped"][instance.path] += 1
            continue

        for relation in resolution.relations:
            state = Entity(
                entity_id=f"state:{instance.horizon_hours}:{instance.label}",
                name=instance.label,
                type=RELATION_ENTITY_TYPE[relation],
                description=f"{instance.label} reported at T-{instance.horizon_hours}",
                confidence=1.0,
            )
            members = tuple(
                Entity(entity_id=eid, name=name, type=type_name, description="", confidence=1.0)
                for eid, name, type_name in anchors
            ) + (state,)
            evidence = _render_evidence(instance, storm=storm_label, port=port)
            hypergraph.add_entity(state)
            for member in members[:-1]:
                hypergraph.add_entity(member)
            hypergraph.add_hyperedge(
                Hyperedge(
                    # Provisional: §A.2 step 5 recomputes the id from the final entity set. The
                    # non-empty-id invariant has to hold in the meantime, so the hash is taken
                    # over the provisional inputs rather than left blank.
                    hyperedge_id=f"raw:{scenario.scenario_id.raw}:{instance.horizon_hours}:{relation}:{instance.text_position}",
                    entity_ids=tuple(m.entity_id for m in members),
                    relation=relation,
                    evidence=evidence,
                    attributes={
                        "field": instance.path,
                        "value": "" if instance.value is None else str(instance.value),
                    },
                    confidence=1.0,
                    group_id=str(scenario.scenario_id),
                    horizon_hours=instance.horizon_hours,
                    text_position=instance.text_position,
                )
            )
            stats["fields_kept"] += 1
            stats["raw_hyperedges"] += 1
            if instance.is_empty:
                stats["fields_empty_value"] += 1

    normalized = normalize_hypergraph(hypergraph, storm=storm, port=port, cfg=cfg.hypergraph)
    # Record the provenance map *after* normalisation, when the ids are final. A cross-horizon
    # change edge inherited no field from its source, so it is correctly absent here.
    for edge in normalized.hyperedges.values():
        path = edge.attributes.get("field")
        if isinstance(path, str) and path:
            field_of[edge.hyperedge_id] = path

    return normalized, stats, field_of


def _render_evidence(instance: FieldInstance, *, storm: str, port: str) -> str:
    """The ``s_e`` of a reference hyperedge — deterministic, and containing the entity name.

    §A.1 requires ``n_v ⊆ s_e``, so the label has to appear; the value follows it so that the
    embedded text of §A.4 carries the fact itself rather than only its field name. The template
    is fixed rather than varied because the id is a hash of this string: two runs over the same
    release must produce the same graph.
    """
    value = "" if instance.value is None else str(instance.value)
    return (
        f"{instance.label} at T-{instance.horizon_hours} for Hurricane {storm} and {port}: {value}"
    )
