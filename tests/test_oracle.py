"""§5 of ``data/CONTRACT.md`` — tests for the retrieval oracle.

The oracle is the half of the evaluation that §4.1's single accuracy number cannot supply:
``CyPortQA_template.json`` names, for every one of the 48 questions, the field that produces
its ground truth, so "which hyperedges does this question need" has a closed-form answer. These
tests pin down the three pieces that answer depends on, in increasing order of cost:

* :mod:`okh_rag.evaluation.field_map` — the §3.2 source-field table, as a spec. Pure, no data,
  so a change to the spec is a test failure rather than a silently different gold set.
* :mod:`okh_rag.evaluation.reference_graph` — the deterministic hypergraph a perfect §A.2
  extractor would have produced from ``Encoded_senario.json``. Needs the release.
* :mod:`okh_rag.evaluation.oracle` — the gold-set definition and the metrics.

The metrics are checked two ways, and the split matters. Their *arithmetic* is checked on a
hand-built graph (four hyperedges, one gold relation) where every ratio can be worked out by
hand, because a metric tested only against a real corpus can be wrong in a way the corpus
happens to hide. Their *behaviour on the release* is checked against the measured aggregate
facts — the coverage, the collision counts, the structural defects — because those are exactly
the quantities the rest of the reproduction reads off this module.

The corpus tests read ``data/raw/cyportqa`` and are skipped when the benchmark is not vendored
(``data/raw`` is git-ignored — 233 MB, reproducible in one command; see ``data/CONTRACT.md``
§2). The format of that guard is copied from ``tests/test_data_cyportqa.py`` deliberately: one
convention for "needs the release" across the suite.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from okh_rag.config import Config, load_config
from okh_rag.data.cyportqa import (
    AnswerFormat,
    CyPortQACorpus,
    QAContext,
    QARecord,
    Template,
    load_encoded,
    load_qa,
    load_templates,
)
from okh_rag.evaluation.field_map import (
    ENTITY_FIELD_PATHS,
    FIELD_TO_RELATION,
    MULTI_SEGMENT_WILDCARDS,
    FieldKind,
    paths_for_relation,
    relation_family,
    relations_for_path,
)
from okh_rag.evaluation.oracle import (
    EvidenceStatus,
    GoldEvidence,
    OracleMetrics,
    OracleOutcome,
    OracleReport,
    evaluate_record,
    gold_for_record,
    gold_matches_field_provenance,
    template_coverage,
)
from okh_rag.evaluation.reference_graph import (
    RELATION_ENTITY_TYPE,
    ReferenceGraph,
    ReferenceGraphStats,
    build_reference_graph,
    count_group_id_collisions,
    iter_field_instances,
)
from okh_rag.retrieval.candidates import CandidatePool
from okh_rag.retrieval.retriever import RetrievalResult
from okh_rag.retrieval.scoring import CandidateScores
from okh_rag.schema import ENTITY_TYPE_REGISTRY, Hyperedge, KnowledgeHypergraph, Trajectory
from okh_rag.vocabulary import RELATION_TO_FAMILY, is_cross_horizon

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "raw" / "cyportqa"
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"

#: The two scenario keys that collide under §A.2's year-less entity-id template, plus their
#: siblings. `CLAUDETTE` is both a 2015 and a 2021 storm, and both made port at each of these
#: four ports, so a bare storm name makes the two scenarios' canonical entity ids identical —
#: see ``tests::TestReferenceGraphIsAliasFree`` and ``reference_graph.count_group_id_collisions``.
ALIASING_SCENARIOS = (
    "CLAUDETTE_2015_Baltimore,_MD",
    "CLAUDETTE_2021_Baltimore,_MD",
    "CLAUDETTE_2015_Wilmington,_DE",
    "CLAUDETTE_2021_Wilmington,_DE",
    "CLAUDETTE_2015_Virginia,_VA,_Port_of",
    "CLAUDETTE_2021_Virginia,_VA,_Port_of",
    "CLAUDETTE_2015_South_Jersey_Port_Corp,_NJ",
    "CLAUDETTE_2021_South_Jersey_Port_Corp,_NJ",
)

#: Every scenario the release encodes a structural anomaly in — 10 stray keys and 1 displaced
#: block. Enumerated rather than searched so the test states which scenarios it is about.
ANOMALOUS_SCENARIOS = (
    "DORIAN_2019_Port_of_Charleston,_SC",
    "FRED_2021_New_York,_NY_&_NJ",
    "PAINE_2016_San_Diego_Unified_Port,_CA",
    "FLORENCE_2018_New_York,_NY_&_NJ",
    "DORIAN_2019_Wilmington,_NC",
    "HUMBERTO_2019_Canaveral_Port_District,_FL",
    "SALLY_2020_Port_Everglades,_FL",
    "BETA_2020_Corpus_Christi,_TX",
)

#: The one structural defect a question could be scored against: `FLORENCE_2018`'s T-84 block
#: nests the whole `weather_observation` mapping inside `cyclone_condition_and_forecast` while
#: the real block is empty, so 24 weather fields arrive under the wrong prefix.
DISPLACED_SCENARIO = "FLORENCE_2018_New_York,_NY_&_NJ"
DISPLACED_HORIZON = 84


def _data_available() -> bool:
    return (DATA_ROOT / "dataset" / "CyPortQA.json").exists() and (
        DATA_ROOT / "source_data" / "Encoded_senario.json"
    ).exists()


requires_data = pytest.mark.skipif(
    not _data_available(),
    reason="CyPortQA not vendored under data/raw/cyportqa; see data/CONTRACT.md section 2",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def encoded():
    if not _data_available():
        pytest.skip("CyPortQA not vendored")
    return load_encoded(DATA_ROOT)


@pytest.fixture(scope="module")
def templates():
    if not _data_available():
        pytest.skip("CyPortQA not vendored")
    return load_templates(DATA_ROOT)


@pytest.fixture(scope="module")
def corpus(encoded, templates) -> CyPortQACorpus:
    """The full corpus, for the record-level checks. QA is the slow part (~2.5 s for 101 MB)."""
    if not _data_available():
        pytest.skip("CyPortQA not vendored")
    return CyPortQACorpus(qa=load_qa(DATA_ROOT), encoded=encoded, templates=templates)


@pytest.fixture(scope="module")
def full_graph(corpus, cfg) -> ReferenceGraph:
    """The reference graph over all 472 scenarios — ~18 s, so module-scoped and only built on demand."""
    return build_reference_graph(corpus, cfg=cfg)


def _build(corpus: CyPortQACorpus, cfg: Config, scenario_ids) -> ReferenceGraph:
    """A reference graph over a scenario subset — cheap, since the subsets here are 8 scenarios."""
    return build_reference_graph(corpus, cfg=cfg, scenario_ids=tuple(scenario_ids))


# ---------------------------------------------------------------------------
# §3.2 — the field table, as a spec
# ---------------------------------------------------------------------------


class TestFieldMapSpec:
    """The source-field table is the oracle's unit, so its shape is asserted, not assumed."""

    def test_every_relation_in_the_table_is_on_the_vocabulary(self):
        """A row pointing off-vocabulary would produce a gold set no hyperedge can ever be in."""
        off = sorted(
            {relation for relation in FIELD_TO_RELATION.values() if relation not in RELATION_TO_FAMILY}
        )
        assert off == [], f"§3.2 rows name relations absent from §A.1 Table 3: {off}"

    def test_family_thirteen_gets_no_source_field(self):
        """§3.2 gives family 13 "*no source field* — synthesised by §A.2 step 4".

        A cross-horizon change edge therefore can never be a question's answer, so giving it a
        field would put a synthetic edge into the gold sets. This is the assertion that keeps it
        out, and it is the one line that would silently break the gold sets if the table grew.
        """
        cross = sorted(key for key, relation in FIELD_TO_RELATION.items() if is_cross_horizon(relation))
        assert cross == [], f"family-13 relations must have no §3.2 field, found {cross}"

    def test_the_wildcard_keys_are_exactly_the_declared_ones(self):
        """Five rows carry a ``*``: four ``operation.*.<field>`` and one ``gale_probability.*``.

        Frozen as a set rather than a count because a wildcard is a *loosening* of the spec — a
        new one would let a path resolve that previously fell through, which is precisely the kind
        of change the oracle's "an unmapped path is a finding" rule depends on being deliberate.
        """
        assert set(MULTI_SEGMENT_WILDCARDS) == {
            "weather_observation.gale_probability.*",
            "operation.*.Condition",
            "operation.*.VesselOps",
            "operation.*.OceanVesselOps",
            "operation.*.FacilityOps",
        }

    def test_relation_entity_types_are_registered(self):
        """Every ``relation -> §A.1 entity type`` value must be a real entity type.

        A typo here would not raise anywhere — :func:`~okh_rag.schema.normalize_type_name` would
        just hand back the misspelling — it would only demote the state to an unknown-type
        roll-up and quietly change which entities §A.6 Eq. 12's Jaccard term can match.
        """
        unknown = sorted({v for v in RELATION_ENTITY_TYPE.values() if v not in ENTITY_TYPE_REGISTRY})
        assert unknown == [], f"RELATION_ENTITY_TYPE names unregistered types: {unknown}"

    def test_every_table_relation_has_an_entity_type(self):
        """The reference builder indexes ``RELATION_ENTITY_TYPE[relation]`` for each kept field."""
        missing = sorted(set(FIELD_TO_RELATION.values()) - set(RELATION_ENTITY_TYPE))
        assert missing == [], f"fields imply relations with no entity type: {missing}"

    @pytest.mark.parametrize(
        ("path", "kind", "relations"),
        [
            # leaf
            ("weather_observation.under_watch", FieldKind.RELATION, ("has_watch_status",)),
            # offset wildcard
            ("operation.in_12h.Condition", FieldKind.RELATION, ("has_operation_status",)),
            ("operation.now.VesselOps", FieldKind.RELATION, ("affects_vessel_handling",)),
            # sub-dict wildcard, key carries a space
            (
                "weather_observation.gale_probability.in 12h",
                FieldKind.RELATION,
                ("has_leadtime_probability",),
            ),
            # entity, not a relation
            ("port.name", FieldKind.ENTITY, ()),
            ("cyclone.name", FieldKind.ENTITY, ()),
            # the systematic gap: the storm's own name is a persistent object, not a relation
            ("cyclone_condition_and_forecast.name", FieldKind.UNKNOWN, ()),
        ],
    )
    def test_resolution_shapes(self, path, kind, relations):
        resolved = relations_for_path(path)
        assert resolved.kind is kind
        assert resolved.relations == relations

    def test_offset_wildcard_matches_every_relative_horizon(self):
        """The wildcard exists to cover all ten offsets; a sample of them is the honest check."""
        for offset in ("now", "in_12h", "in_24h", "in_48h", "in_120h"):
            resolved = relations_for_path(f"operation.{offset}.Condition")
            assert resolved.relation == "has_operation_status", offset
            assert resolved.matched_key == "operation.*.Condition", offset

    def test_container_resolves_to_the_union_of_its_children(self):
        """Q33 names ``gale_probability`` rather than a key inside it; every child is the same relation."""
        container = relations_for_path("weather_observation.gale_probability")
        assert container.kind is FieldKind.RELATION
        assert container.relations == ("has_leadtime_probability",)
        assert container.relation == "has_leadtime_probability"

    def test_a_container_with_several_relations_has_no_single_relation(self):
        """``relation`` is ``None`` exactly when the container spans more than one — callers must use ``relations``."""
        container = relations_for_path("cyclone_condition_and_forecast")
        assert container.kind is FieldKind.RELATION
        assert container.relation is None
        assert container.relations == (
            "has_cyclone_state",
            "has_category_state",
            "has_motion",
            "forecasts_track",
            "forecasts_landfall",
            "has_hours_to_landfall",
        )

    def test_container_deduplicates_a_shared_relation(self):
        """``motion`` holds two keys with the same relation; reporting it twice would read as two."""
        container = relations_for_path("cyclone_condition_and_forecast.motion")
        assert container.relations == ("has_motion",)

    def test_a_typo_falls_through_instead_of_fuzzy_matching(self):
        """The exact pass must stay exact: fuzzy matching would manufacture a gold set from a misspelling."""
        assert relations_for_path("weather_observation.under_watchh").kind is FieldKind.UNKNOWN
        assert relations_for_path("weather_observation.under").kind is FieldKind.UNKNOWN
        assert relations_for_path("not_a_block.under_watch").kind is FieldKind.UNKNOWN

    def test_empty_path_is_unknown_rather_than_an_error(self):
        """``Template.static_prefix`` returns ``""`` when it cannot find a placeholder; never raise."""
        assert relations_for_path("").kind is FieldKind.UNKNOWN

    def test_entity_paths_are_exactly_the_two_persistent_objects(self):
        assert ENTITY_FIELD_PATHS == frozenset({"port.name", "cyclone.name"})

    def test_paths_for_relation_is_the_inverse_map(self):
        assert paths_for_relation("forecasts_track") == (
            "cyclone_condition_and_forecast.trajectory",
            "weather_observation.within_uncertainty_cone",
        )
        assert paths_for_relation("no_such_relation") == ()

    def test_relation_family_reports_off_vocabulary_as_none(self):
        assert relation_family("has_motion") == RELATION_TO_FAMILY["has_motion"]
        assert relation_family("has_motion") == "cyclone_state"
        assert relation_family("not_a_relation") is None


# ---------------------------------------------------------------------------
# Hand-built graph — arithmetic on every metric
# ---------------------------------------------------------------------------

GROUP = "S_2020_P"
HORIZON = 48


def _edge(hyperedge_id: str, relation: str, *, group: str = GROUP, horizon: int = HORIZON, position: int = 0) -> Hyperedge:
    """A minimal §A.1 Eq. 7 hyperedge: two distinct entities so ``|V_e| >= 2`` holds.

    The entity ids are derived from the hyperedge id rather than shared, because a shared id set
    is what §A.2 step 5's dedup keys on — a hand-built graph must not accidentally look deduped.
    """
    return Hyperedge(
        hyperedge_id=hyperedge_id,
        entity_ids=(f"state:{hyperedge_id}:a", f"state:{hyperedge_id}:b"),
        relation=relation,
        evidence=f"{hyperedge_id} reported at T-{horizon}",
        attributes={"field": f"weather_observation.{hyperedge_id}"},
        group_id=group,
        horizon_hours=horizon,
        text_position=position,
    )


def _hand_graph(
    *,
    gold: tuple[str, ...] = ("g1", "g2", "g3"),
    others: tuple[str, ...] = ("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"),
) -> ReferenceGraph:
    """A graph whose only ``has_watch_status`` hyperedges at T-48 are the gold ones.

    ``field_of`` deliberately records the real §3.2 field each edge came from, so
    :func:`~okh_rag.evaluation.oracle.gold_matches_field_provenance` has something to check on it.
    """
    hypergraph = KnowledgeHypergraph()
    field_of: dict[str, str] = {}
    for index, hid in enumerate(gold):
        # Both `under_watch` and `watch_coast` imply has_watch_status; variety here is what makes
        # the union-of-fields check in `test_gold_matches_field_provenance` meaningful.
        field_of[hid] = "weather_observation.under_watch" if index % 2 == 0 else "weather_observation.watch_coast"
        hypergraph.add_hyperedge(_edge(hid, "has_watch_status", position=index))
    for index, hid in enumerate(others):
        hypergraph.add_hyperedge(_edge(hid, "forecasts_track", position=100 + index))

    by_relation: dict[tuple[str, int, str], set[str]] = {}
    for hid, edge in hypergraph.hyperedges.items():
        by_relation.setdefault((edge.group_id, edge.horizon_hours, edge.relation), set()).add(hid)

    return ReferenceGraph(
        hypergraph=hypergraph,
        stats=ReferenceGraphStats(scenarios=1, hyperedges=len(hypergraph.hyperedges)),
        field_of=field_of,
        by_relation={key: frozenset(value) for key, value in by_relation.items()},
    )


def _template(template_id: str, answer: str) -> Template:
    return Template(
        template_id=template_id,
        tags=(),
        context=(),
        modalities=(),
        prompt="",
        question="",
        answer=answer,
    )


def _record(
    *,
    template_id: str = "Q2",
    scenario_id: str = GROUP,
    horizon: int = HORIZON,
    relative_horizon: str | None = None,
    port: str | None = "Port of Nowhere",
) -> QARecord:
    return QARecord(
        scenario_id=scenario_id,
        template_id=template_id,
        context=QAContext(storm="S", year=2020, horizon_hours=horizon, port=port),
        tags=(),
        answer_format=AnswerFormat.TRUE_FALSE,
        stage=None,
        dimensions=(),
        annotations=(),
        modalities=(),
        prompt="",
        question="",
        answer="Yes",
        relative_horizon=relative_horizon,
    )


def _result(
    pool_ids: tuple[str, ...],
    anchor_ids: tuple[str, ...],
    trajectory_ids: tuple[str, ...],
    *,
    variant: str = "heuristic",
) -> RetrievalResult:
    """A :class:`RetrievalResult` with only the stages the oracle reads, filled in honestly."""
    relevance = np.linspace(1.0, 0.1, num=len(pool_ids), dtype=np.float64)
    scores = CandidateScores(
        hyperedge_ids=pool_ids,
        relevance=relevance,
        transitions=np.zeros((len(pool_ids), len(pool_ids)), dtype=np.float64),
        variant=variant,
    )
    pool = CandidatePool(
        hyperedge_ids=pool_ids,
        relevance=relevance,
        group_id=GROUP,
        anchors=anchor_ids,
    )
    return RetrievalResult(
        query="q",
        group_id=GROUP,
        pool=pool,
        scores=scores,
        trajectories=(Trajectory(trajectory_ids),),
        term_scores=(),
        variant=variant,
    )


class TestGoldSetDefinition:
    """``gold(q)`` — the closed-form requirement, on a graph small enough to read."""

    def test_resolved_gold_is_the_groups_edges_at_that_horizon_with_that_relation(self):
        graph = _hand_graph()
        record = _record()
        template = _template("Q2", "{senario.time.<key>.weather_observation.under_watch}")

        gold = gold_for_record(record, template, graph=graph)

        assert gold.status is EvidenceStatus.RESOLVED
        assert gold.is_scorable
        assert gold.relations == ("has_watch_status",)
        assert gold.hyperedge_ids == frozenset({"g1", "g2", "g3"})
        assert gold.size == 3
        assert gold.field == "weather_observation.under_watch"

    def test_the_same_field_at_another_horizon_is_not_gold(self):
        """The horizon term of ``gold(q)`` is load-bearing: T-48 evidence is not T-24 evidence."""
        graph = _hand_graph()
        record = _record(horizon=HORIZON - 12)
        template = _template("Q2", "{senario.time.<key>.weather_observation.under_watch}")

        gold = gold_for_record(record, template, graph=graph)

        assert gold.status is EvidenceStatus.EMPTY
        assert gold.hyperedge_ids == frozenset()
        assert not gold.is_scorable

    def test_empty_gold_names_its_cause(self):
        """An empty gold set is a *finding*, so the note has to say which relation was missing."""
        gold = gold_for_record(
            _record(horizon=12),
            _template("Q2", "{senario.time.<key>.weather_observation.under_watch}"),
            graph=_hand_graph(),
        )
        assert "has_watch_status" in gold.note

    def test_a_container_field_unions_its_relations(self):
        """Q33's shape: the template names the mapping, so the gold set is the union over its children."""
        graph = _hand_graph()
        record = _record(template_id="Q33")
        template = _template("Q33", "{senario.time.<key>.weather_observation.gale_probability}")

        gold = gold_for_record(record, template, graph=graph)

        # The hand graph has no `has_leadtime_probability` edge, so this is the EMPTY arm — and it
        # is the union of relations, not the first of them, that the note reports.
        assert gold.relations == ("has_leadtime_probability",)
        assert gold.status is EvidenceStatus.EMPTY

    def test_relative_horizon_is_substituted_into_an_operation_field(self):
        """Q42/Q46-Q48 declare ``operation.<key>.<field>``; the record's tag fills ``<key>``."""
        graph = _hand_graph()
        record = _record(template_id="Q42", relative_horizon="in_24h")
        template = _template("Q42", "{senario.time.<key>.operation.<key>.VesselOps}")

        gold = gold_for_record(record, template, graph=graph)

        assert gold.field == "operation.in_24h.VesselOps"
        assert gold.relations == ("affects_vessel_handling",)
        assert gold.status is EvidenceStatus.EMPTY  # no such edge in the hand graph

    def test_a_relative_horizon_template_without_a_tag_is_unmapped_not_empty(self):
        """Leaving ``<key>`` in place would read the wrong sub-key; the absence is named instead."""
        gold = gold_for_record(
            _record(template_id="Q42", relative_horizon=None),
            _template("Q42", "{senario.time.<key>.operation.<key>.VesselOps}"),
            graph=_hand_graph(),
        )
        assert gold.status is EvidenceStatus.UNMAPPED
        assert "operation.<key>" in gold.note

    def test_an_unknown_field_is_unmapped_and_says_so(self):
        gold = gold_for_record(
            _record(),
            _template("Q99", "{senario.time.<key>.cyclone_condition_and_forecast.name}"),
            graph=_hand_graph(),
        )
        assert gold.status is EvidenceStatus.UNMAPPED
        assert "§3.2" in gold.note

    def test_an_entity_requirement_is_satisfied_by_the_entity_not_a_hyperedge(self):
        """Q16/Q19 answer ``{senario.port.name}``; §3.1 puts the port among the persistent objects."""
        graph = _hand_graph()
        record = _record(template_id="Q16", port="Port of Nowhere")
        template = _template("Q16", "{senario.port.name}")

        gold = gold_for_record(record, template, graph=graph)

        assert gold.relations == ()
        assert gold.status is EvidenceStatus.EMPTY  # the hand graph registers no PORT entity
        assert gold.entity_id is None
        assert not gold.is_scorable

    def test_an_entity_present_in_the_group_is_found(self):
        graph = _hand_graph()
        # Give one hyperedge a PORT member whose name matches the record's context port.
        from okh_rag.schema import Entity

        port_entity = Entity(
            entity_id="persistent:port", name="Port of Nowhere", type="PORT", description="", confidence=1.0
        )
        graph.hypergraph.add_entity(port_entity)
        edge = graph.hypergraph.hyperedges["g1"]
        graph.hypergraph.hyperedges["g1"] = Hyperedge(
            hyperedge_id=edge.hyperedge_id,
            entity_ids=(*edge.entity_ids, "persistent:port"),
            relation=edge.relation,
            evidence=edge.evidence,
            attributes=edge.attributes,
            group_id=edge.group_id,
            horizon_hours=edge.horizon_hours,
            text_position=edge.text_position,
        )

        gold = gold_for_record(_record(template_id="Q16"), _template("Q16", "{senario.port.name}"), graph=graph)

        assert gold.status is EvidenceStatus.ENTITY
        assert gold.entity_id == "persistent:port"
        assert not gold.is_scorable

    def test_gold_matches_field_provenance_agrees_on_this_graph(self):
        """The relation-keyed and field-keyed definitions must coincide wherever both exist."""
        graph = _hand_graph()
        gold = gold_for_record(
            _record(), _template("Q2", "{senario.time.<key>.weather_observation.under_watch}"), graph=graph
        )
        assert gold_matches_field_provenance(gold, graph)

    def test_gold_matches_field_provenance_rejects_a_drifted_field(self):
        """The cross-check has to be able to fail, or it is decoration rather than a check."""
        graph = _hand_graph()
        # `g1` claims to come from a field that implies `forecasts_track`, not `has_watch_status`.
        drifted = ReferenceGraph(
            hypergraph=graph.hypergraph,
            stats=graph.stats,
            field_of={**graph.field_of, "g1": "cyclone_condition_and_forecast.trajectory"},
            by_relation=graph.by_relation,
        )
        gold = gold_for_record(
            _record(), _template("Q2", "{senario.time.<key>.weather_observation.under_watch}"), graph=drifted
        )
        assert not gold_matches_field_provenance(gold, drifted)

    def test_unscorable_gold_passes_the_provenance_check_vacuously(self):
        """Nothing to compare, so the check must not report a false mismatch."""
        graph = _hand_graph()
        gold = gold_for_record(
            _record(horizon=12), _template("Q2", "{senario.time.<key>.weather_observation.under_watch}"), graph=graph
        )
        assert not gold.is_scorable
        assert gold_matches_field_provenance(gold, graph)


class TestMetricsArithmetic:
    """Every ratio, worked out by hand on a three-gold, ten-member pool."""

    POOL = ("g1", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "g2")
    ANCHORS = ("g1", "x1")
    TRAJECTORY = ("x1", "g1", "x2")

    def _metrics(self, *, canonical: dict[str, int] | None = None) -> OracleMetrics:
        graph = _hand_graph()
        outcome = evaluate_record(
            _record(),
            _template("Q2", "{senario.time.<key>.weather_observation.under_watch}"),
            _result(self.POOL, self.ANCHORS, self.TRAJECTORY),
            graph=graph,
            canonical_position=canonical,
        )
        assert outcome.metrics is not None
        return outcome.metrics

    def test_recall_at_each_stage(self):
        metrics = self._metrics()
        # gold = {g1, g2, g3}; the pool holds g1 and g2, so 2/3.
        assert metrics.pool_recall == pytest.approx(2 / 3)
        # anchors are the top-K cosine survivors; only g1 is among them.
        assert metrics.anchor_recall == pytest.approx(1 / 3)
        # the returned trajectory holds g1 alone.
        assert metrics.trajectory_recall == pytest.approx(1 / 3)
        # the union over trajectories equals the one trajectory here.
        assert metrics.all_trajectory_recall == pytest.approx(1 / 3)
        assert metrics.trajectory_precision == pytest.approx(1 / 3)
        assert metrics.gold_size == 3

    def test_chance_recall_is_the_derived_floor(self):
        """``E[|gold ∩ γ|] = pool_recall · L/|C|`` exactly — derived, not sampled."""
        metrics = self._metrics()
        assert metrics.chance_recall == pytest.approx((2 / 3) * (3 / 10))
        assert metrics.chance_recall == pytest.approx(0.2)

    def test_first_gold_step_is_one_based(self):
        """``g1`` sits at index 1, so it is step 2 — a 0-based reading would report 1."""
        assert self._metrics().first_gold_step == 2

    def test_first_gold_step_is_none_when_no_gold_was_retrieved(self):
        graph = _hand_graph()
        outcome = evaluate_record(
            _record(),
            _template("Q2", "{senario.time.<key>.weather_observation.under_watch}"),
            _result(("x1", "x2"), ("x1",), ("x1", "x2")),
            graph=graph,
        )
        assert outcome.metrics is not None
        assert outcome.metrics.first_gold_step is None

    def test_order_concordance_counts_comparable_pairs(self):
        """``canonical`` places x1 before g1; the trajectory inverts that one pair of the three."""
        metrics = self._metrics(canonical={"g1": 0, "x1": 1, "x2": 2})
        # pairs in trajectory order: (x1,g1) inverted, (x1,x2) ok, (g1,x2) ok -> 2/3.
        assert metrics.canonical_concordance == pytest.approx(2 / 3)

    def test_order_concordance_is_none_with_fewer_than_two_placed_members(self):
        """``None`` distinguishes "nothing to compare" from "inverted" — averaging them as 0 lies."""
        metrics = self._metrics(canonical={"g1": 0})  # x1 and x2 are unplaced
        assert metrics.canonical_concordance is None

    def test_gold_order_concordance_is_none_when_fewer_than_two_gold_members_arrived(self):
        """The gold-restricted variant is frequently ``None`` at realistic recall, by design."""
        metrics = self._metrics(canonical={"g1": 0, "x1": 1, "x2": 2})
        assert metrics.gold_order_concordance is None

    def test_order_metrics_are_none_without_a_canonical_position(self):
        metrics = self._metrics(canonical=None)
        assert metrics.canonical_concordance is None
        assert metrics.gold_order_concordance is None

    def test_an_unscorable_record_comes_back_with_no_metrics(self):
        """The pool and trajectory are still reported, so an unscorable record is auditable."""
        graph = _hand_graph()
        outcome = evaluate_record(
            _record(horizon=12),
            _template("Q2", "{senario.time.<key>.weather_observation.under_watch}"),
            _result(self.POOL, self.ANCHORS, self.TRAJECTORY),
            graph=graph,
        )
        assert outcome.metrics is None
        assert outcome.gold.status is EvidenceStatus.EMPTY
        assert outcome.pool_size == len(self.POOL)
        assert outcome.trajectory_size == len(self.TRAJECTORY)
        assert outcome.variant == "heuristic"

    def test_metrics_round_trip_through_json_safe_dicts(self):
        metrics = self._metrics(canonical={"g1": 0, "x1": 1, "x2": 2})
        payload = metrics.as_dict()
        assert payload["pool_recall"] == pytest.approx(0.6667, abs=1e-4)
        assert payload["chance_recall"] == 0.2
        assert payload["first_gold_step"] == 2
        assert payload["gold_order_concordance"] is None


def _outcome(
    *,
    qa_id: str,
    template_id: str,
    status: EvidenceStatus,
    relations: tuple[str, ...] = ("has_watch_status",),
    metrics: OracleMetrics | None,
    horizon: int = HORIZON,
) -> OracleOutcome:
    gold = GoldEvidence(
        qa_id=qa_id,
        template_id=template_id,
        scenario_id=GROUP,
        horizon_hours=horizon,
        field="weather_observation.under_watch",
        status=status,
        resolution=relations_for_path("weather_observation.under_watch")
        if relations
        else relations_for_path("port.name"),
        hyperedge_ids=frozenset(),
    )
    return OracleOutcome(gold=gold, metrics=metrics)


class TestAggregation:
    """The report's means, and the denominators that qualify them."""

    def _report(self) -> OracleReport:
        a = OracleMetrics(
            gold_size=3,
            pool_recall=2 / 3,
            anchor_recall=1 / 3,
            trajectory_recall=1 / 3,
            all_trajectory_recall=1 / 3,
            trajectory_precision=1 / 3,
            chance_recall=0.2,
            first_gold_step=2,
            canonical_concordance=2 / 3,
            gold_order_concordance=None,
        )
        b = OracleMetrics(
            gold_size=1,
            pool_recall=0.0,
            anchor_recall=0.0,
            trajectory_recall=0.0,
            all_trajectory_recall=0.0,
            trajectory_precision=0.0,
            chance_recall=0.0,
            first_gold_step=None,
            canonical_concordance=0.5,
            gold_order_concordance=None,
        )
        return OracleReport(
            outcomes=[
                _outcome(qa_id="a", template_id="Q2", status=EvidenceStatus.RESOLVED, metrics=a),
                _outcome(qa_id="b", template_id="Q2", status=EvidenceStatus.RESOLVED, metrics=b),
                _outcome(qa_id="c", template_id="Q16", status=EvidenceStatus.ENTITY, relations=(), metrics=None),
            ],
            backend="lexical",
        )

    def test_only_scorable_records_enter_the_ratio_means(self):
        """An unscorable record has no denominator, so averaging it in would floor the retrieval score."""
        means = self._report().mean_metrics()
        # The report rounds to four decimals, hence the tolerance rather than exact division.
        assert means["pool_recall"] == pytest.approx(1 / 3, abs=1e-4)
        assert means["trajectory_recall"] == pytest.approx(1 / 6, abs=1e-4)

    def test_ordinal_metrics_average_over_their_own_denominator(self):
        means = self._report().mean_metrics()
        assert means["first_gold_step_mean"] == pytest.approx(2.0)  # only record `a` defines it
        assert means["canonical_concordance_mean"] == pytest.approx(7 / 12, abs=1e-4)
        assert means["canonical_concordance_n"] == 2
        assert means["gold_order_concordance_mean"] is None
        assert means["gold_order_concordance_n"] == 0

    def test_over_chance_is_the_ratio_of_the_two_means(self):
        means = self._report().mean_metrics()
        assert means["chance_recall"] == pytest.approx(0.1, abs=1e-4)  # (0.2 + 0.0) / 2
        assert means["trajectory_recall_over_chance"] == pytest.approx(
            means["trajectory_recall"] / means["chance_recall"], abs=1e-3
        )

    def test_status_counts_and_empty_causes_are_reported(self):
        report = self._report()
        assert report.status_counts() == {"entity": 1, "resolved": 2}
        # An ENTITY record is not an empty-gold record, so it must not appear in the empty causes.
        assert report.empty_gold_by_cause() == {}

    def test_by_template_groups_on_the_template(self):
        by_template = self._report().by_template()
        assert set(by_template) == {"Q2"}
        assert by_template["Q2"]["n"] == 2
        assert by_template["Q2"]["relation"] == "has_watch_status"
        assert by_template["Q2"]["family"] == "advisory"

    def test_an_empty_report_has_no_means_rather_than_zeros(self):
        assert OracleReport().mean_metrics() == {}


# ---------------------------------------------------------------------------
# The release — the measured structural facts
# ---------------------------------------------------------------------------


@requires_data
class TestReferenceGraphIsAliasFree:
    """§A.2 step 2's entity-id template has no year component; the corpus has repeat storm names.

    ``CLAUDETTE`` made port at four ports in both 2015 and 2021. With a bare storm name the two
    scenarios' canonical entity ids coincide, §A.2 step 5's hash of ``(relation, V_e, s_e)``
    coincides whenever the encoded values also coincide, and
    :meth:`~okh_rag.schema.KnowledgeHypergraph.add_hyperedge` overwrites the first edge with the
    second — silently, while both groups keep listing the id. That is the defect this class
    measures, and ``reference_graph`` qualifies the storm with its year to prevent it.
    """

    def test_the_release_really_does_repeat_a_storm_name_at_a_port(self, encoded):
        """Guard the guard: if this stops holding, the collision test below tests nothing."""
        by_name_and_port: dict[tuple[str, str], list[str]] = {}
        for key in encoded:
            parts = key.split("_")
            by_name_and_port.setdefault((parts[0], "_".join(parts[2:])), []).append(key)
        repeated = {pair: keys for pair, keys in by_name_and_port.items() if len(keys) > 1}
        assert len(repeated) == 4, repeated
        assert set(ALIASING_SCENARIOS) <= {key for keys in repeated.values() for key in keys}

    def test_building_the_colliding_scenarios_produces_no_aliased_ids(self, corpus, cfg):
        graph = _build(corpus, cfg, ALIASING_SCENARIOS)
        assert graph.stats.id_collisions == {"ids": 0, "groups": 0, "stale": 0}

    def test_the_two_years_groups_share_no_hyperedge(self, corpus, cfg):
        """The property the year-qualification buys, stated directly: disjoint group buckets."""
        graph = _build(corpus, cfg, ALIASING_SCENARIOS)
        buckets = {group: set(ids) for group, ids in graph.hypergraph.groups.items()}
        assert len(buckets) == len(ALIASING_SCENARIOS)
        shared = [
            (left, right)
            for left, right in itertools.combinations(sorted(buckets), 2)
            if buckets[left] & buckets[right]
        ]
        assert shared == [], f"groups share hyperedge ids: {shared}"

    def test_every_group_owns_its_own_bucket(self, corpus, cfg):
        """``stale`` counts bucket entries whose edge belongs to another group — the overwrite symptom."""
        graph = _build(corpus, cfg, ALIASING_SCENARIOS)
        for group_id, member_ids in graph.hypergraph.groups.items():
            for hid in member_ids:
                assert graph.hypergraph.hyperedges[hid].group_id == group_id

    def test_the_collision_detector_can_actually_fire(self):
        """A detector that only ever returns zero is not a check; alias a hand-built corpus and see."""
        hypergraph = KnowledgeHypergraph()
        edge = _edge("raw:1", "has_watch_status", group="A_2020_X")
        hypergraph.add_hyperedge(edge)
        # A second group lists the same id, exactly as `add_hyperedge`'s overwrite leaves it.
        hypergraph.groups.setdefault("B_2021_X", []).append("raw:1")

        counts = count_group_id_collisions(hypergraph)

        assert counts["ids"] == 1
        assert counts["groups"] == 2
        assert counts["stale"] == 1


@requires_data
class TestReferenceGraphStructure:
    """The yardstick's shape: what it keeps, what it drops, and why."""

    def test_the_storm_entity_keeps_its_value_and_field_label(self, corpus, cfg):
        """The entity name is the field path without its block prefix, so `n_v ⊆ s_e` holds."""
        graph = _build(corpus, cfg, ("BETA_2020_Corpus_Christi,_TX",))
        edge = next(
            edge
            for edge in graph.hypergraph.hyperedges.values()
            if edge.attributes.get("field", "").startswith("weather_observation.")
        )
        label = edge.attributes["field"].split(".", 1)[1]
        assert label in edge.evidence
        state = [graph.hypergraph.entities[e] for e in edge.entity_ids]
        assert any(entity.name == label for entity in state)
        assert edge.horizon_hours is not None

    def test_every_gold_set_matches_the_field_it_claims_to_come_from(self, corpus, cfg):
        """The oracle's self-check over a sample: relation-keyed and field-keyed gold must agree."""
        graph = _build(corpus, cfg, ALIASING_SCENARIOS)
        checked = 0
        for record in corpus.records():
            if record.scenario_id not in graph.hypergraph.groups:
                continue
            gold = gold_for_record(record, corpus.templates[record.template_id], graph=graph)
            assert gold_matches_field_provenance(gold, graph), gold.qa_id
            checked += 1
        assert checked > 0, "the sample scored no records, so the check proved nothing"

    def test_the_relation_index_agrees_with_the_group_buckets(self, corpus, cfg):
        """``by_relation`` is a precomputed inverse of the groups; a drift would mis-price gold sets."""
        graph = _build(corpus, cfg, ALIASING_SCENARIOS)
        for key, ids in graph.by_relation.items():
            group_id, horizon, relation = key
            for hid in ids:
                edge = graph.hypergraph.hyperedges[hid]
                assert (edge.group_id, edge.horizon_hours, edge.relation) == (group_id, horizon, relation)
                assert hid in graph.hypergraph.groups[group_id]

    def test_relation_hyperedges_is_the_union_over_relations(self, corpus, cfg):
        graph = _build(corpus, cfg, ALIASING_SCENARIOS)
        group = next(iter(graph.hypergraph.groups))
        horizon = max(graph.hypergraph.hyperedges[hid].horizon_hours or 0 for hid in graph.hypergraph.groups[group])
        one = graph.horizon_hyperedges(group, horizon)
        union = graph.relation_hyperedges(
            group, horizon, {graph.hypergraph.hyperedges[hid].relation for hid in one}
        )
        assert union == frozenset(one)

    def test_declaring_no_cross_horizon_edges_leaves_no_family_13_edge(self, corpus, cfg):
        """``include_cross_horizon=False`` is the isolating switch; it must actually remove them."""
        graph = build_reference_graph(corpus, cfg=cfg, include_cross_horizon=False, scenario_ids=ALIASING_SCENARIOS)
        leftovers = [hid for hid, edge in graph.hypergraph.hyperedges.items() if is_cross_horizon(edge.relation)]
        assert leftovers == []
        # The buckets must have been rebuilt, not merely filtered.
        for group_id, member_ids in graph.hypergraph.groups.items():
            for hid in member_ids:
                assert hid in graph.hypergraph.hyperedges
                assert graph.hypergraph.hyperedges[hid].group_id == group_id

    def test_the_cross_horizon_edges_are_a_large_share_of_the_graph(self, corpus, cfg):
        """§4.2 reports family 13 as "nearly half" of ``H``; the sample should reflect that scale."""
        with_cross = _build(corpus, cfg, ALIASING_SCENARIOS)
        without = build_reference_graph(corpus, cfg=cfg, include_cross_horizon=False, scenario_ids=ALIASING_SCENARIOS)
        share = 1 - len(without.hypergraph) / len(with_cross.hypergraph)
        assert 0.3 < share < 0.55, f"cross-horizon share {share:.3f} is far from §4.2's 'nearly half'"


@requires_data
class TestReferenceGraphDefects:
    """The release's structural defects, measured rather than discovered by surprise later.

    Each of these is a field that becomes no hyperedge, and the point of counting them separately
    is that "the question's gold set is empty" is otherwise indistinguishable between a field that
    is genuinely absent (the release's scenarios are ragged) and a field that is misencoded.
    """

    def test_the_storm_name_is_the_only_systematic_unmapped_field(self, corpus, cfg):
        """§3.2 correctly has no row for it — the storm is a persistent object, not a relation."""
        graph = _build(corpus, cfg, ANOMALOUS_SCENARIOS)
        unmapped = set(graph.stats.fields_unmapped)
        assert "cyclone_condition_and_forecast.name" in unmapped
        # Everything else unmapped is a stray key or a symptom of the displaced block, never a
        # well-formed §3.2 path that the table should have covered.
        assert not [path for path in unmapped if path == "weather_observation.under_watch"]

    def test_stray_keys_are_counted_with_their_scenarios(self, corpus, cfg):
        """Model commentary stored as a *key*: 10 across the release, plus the displaced block."""
        graph = _build(corpus, cfg, ANOMALOUS_SCENARIOS)
        anomalies = graph.stats.anomalies
        stray = {key: count for key, count in anomalies.items() if key.startswith("stray-key:")}
        assert sum(stray.values()) == 10
        assert "stray-key:_comment" in stray
        assert "stray-key:_note" in stray
        assert "displaced-block:cyclone_condition_and_forecast.weather_observation" in anomalies

    def test_no_anomaly_fires_on_a_scenario_that_has_none(self, corpus, cfg):
        """A detector that fires everywhere would look identical to a broken encoding."""
        graph = _build(corpus, cfg, ALIASING_SCENARIOS)
        assert graph.stats.anomalies == {}

    def test_a_field_inside_a_displaced_block_becomes_no_hyperedge(self, corpus, cfg):
        """FLORENCE's T-84 weather block is nested inside the cyclone block, so its paths are wrong.

        The consequence is ``EMPTY`` gold for any weather question at that horizon, which is a
        *data* defect rather than a retrieval one — and this is the test that keeps the distinction
        attached to the evidence.
        """
        graph = _build(corpus, cfg, (DISPLACED_SCENARIO,))
        # The nested paths arrive prefixed by the block they were misplaced into.
        misplaced = [p for p in graph.stats.fields_unmapped if p.startswith("cyclone_condition_and_forecast.weather_observation.")]
        assert misplaced, "the displaced weather block was not detected"

        # A Q2 (watch status) record at that horizon therefore has no gold evidence at all.
        record = _record(template_id="Q2", scenario_id=DISPLACED_SCENARIO, horizon=DISPLACED_HORIZON)
        gold = gold_for_record(record, corpus.templates["Q2"], graph=graph)
        assert gold.status is EvidenceStatus.EMPTY
        assert gold.hyperedge_ids == frozenset()

    def test_the_displaced_horizon_is_the_only_ragged_weather_block_there(self, corpus, cfg):
        """Scenarios are ragged in general, so name the specific horizon to keep the test honest."""
        scenario = corpus.encoded[DISPLACED_SCENARIO]
        assert DISPLACED_HORIZON in scenario.horizons
        block = scenario.time[DISPLACED_HORIZON]
        assert not block.weather_observation
        assert "weather_observation" in block.cyclone_condition_and_forecast

    def test_iter_field_instances_reports_the_stray_key_on_the_path_it_was_found(self, corpus):
        """The anomaly text has to name enough to locate the key: the leaf, for strays."""
        scenario = corpus.encoded["SALLY_2020_Port_Everglades,_FL"]
        _, anomalies = iter_field_instances(scenario)
        assert any(anomaly.startswith("stray-key:") for anomaly in anomalies)


@requires_data
class TestCorpusOracle:
    """Aggregate facts over the whole release — the numbers the rest of the reproduction cites."""

    def test_template_coverage_is_total(self, templates):
        """46 templates name a relation, 2 name a persistent object, 0 are uncovered.

        ``is_total`` is the load-bearing property: an uncovered template could not be scored on
        *any* hypergraph, so it would be a permanent hole in the evaluation rather than a bug.
        """
        coverage = template_coverage(templates)
        assert coverage.is_total
        assert coverage.total == 48
        assert len(coverage.resolved) == 46
        assert coverage.entities == ("Q16", "Q19")
        assert coverage.unmapped == {}
        assert set(coverage.resolved.values()) <= set(RELATION_TO_FAMILY)

    def test_coverage_dict_is_ordered_by_template_number(self, templates):
        payload = template_coverage(templates).as_dict()
        keys = list(payload["by_relation"])
        assert keys == sorted(keys, key=lambda key: int(key[1:]))

    def test_the_full_graph_has_the_documented_shape(self, full_graph):
        """Locked deliberately: if the release or the builder changes, this is the intended signal.

        The measured values are those reported in ``reports/retrieval_oracle.json`` over all 472
        scenarios. ``id_collisions`` being all-zero is the year-qualification holding at corpus scale.
        """
        stats = full_graph.stats
        assert stats.scenarios == 472
        assert stats.fields_seen == 138_196
        assert stats.fields_kept == 135_248
        assert stats.hyperedges == 248_652
        assert stats.entities == 139_102
        assert stats.id_collisions == {"ids": 0, "groups": 0, "stale": 0}
        assert stats.coverage == pytest.approx(0.9787, abs=5e-4)

    def test_no_template_is_unscorable_on_the_reference_graph(self, corpus, full_graph):
        """Every question maps onto a relation or an entity — no ``UNMAPPED`` status survives."""
        statuses = set()
        for record in corpus.records():
            gold = gold_for_record(record, corpus.templates[record.template_id], graph=full_graph)
            statuses.add(gold.status)
        assert EvidenceStatus.UNMAPPED not in statuses
        assert EvidenceStatus.EMPTY not in statuses
        assert statuses == {EvidenceStatus.RESOLVED, EvidenceStatus.ENTITY}

    def test_the_provenance_self_check_passes_on_every_scorable_question(self, corpus, full_graph):
        """The whole point of keeping ``field_of``: the loose key is verified, not trusted."""
        mismatches = [
            record.qa_id
            for record in corpus.records()
            if not gold_matches_field_provenance(
                gold_for_record(record, corpus.templates[record.template_id], graph=full_graph), full_graph
            )
        ]
        assert mismatches == []

    def test_most_questions_are_scorable(self, corpus, full_graph):
        """95% of the benchmark; the remainder are the two entity templates' questions."""
        total = scorable = entity = 0
        for record in corpus.records():
            total += 1
            gold = gold_for_record(record, corpus.templates[record.template_id], graph=full_graph)
            if gold.is_scorable:
                scorable += 1
            elif gold.status is EvidenceStatus.ENTITY:
                entity += 1
        assert total == 117_178
        assert entity > 0
        assert scorable / total == pytest.approx(0.95, abs=5e-3)
        assert scorable + entity == total
