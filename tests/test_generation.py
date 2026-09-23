"""§3.3 / §A.8 / §4.1 — tests for the generation module and the answer metrics.

Four layers, tested from the inside out, because each one can be wrong in a way the next one hides:

* :mod:`okh_rag.generation.evidence` — the §A.8 evidence format. The interesting content here is not
  the layout (which is quoted from the paper) but the two *decisions* the paper leaves open: how the
  ``Reasoning:`` line is derived, and what an unordered baseline's evidence is allowed to say. Both are
  tested for the property that makes them load-bearing — that a tag named after a causal hand-off is
  reachable inside a single horizon, and that an unordered render cannot contain a ``[Step k]`` header.
* :mod:`okh_rag.generation.output` — §A.8's ``{answer, confidence, rationale}`` contract. Tested on the
  *strange* inputs the release actually produces (MC golds of ``"A."``, numeric golds of ``"T-48"`` and
  ``"<1%"``, responses wrapped in prose and code fences), because a validator tested only on well-formed
  input is a validator that will pass a malformed run.
* :mod:`okh_rag.generation.generator` — the two §A.8 modes and the fallback's aggregation. The
  aggregation is checked on hand-built answers where every weight is known, and the modes are checked
  against a scripted backend that records what it was asked, so "the two modes differ only in how the
  paths are reconciled" is verified rather than asserted.
* :mod:`okh_rag.evaluation.answers` — §4.1's scoring. The per-regime rules are checked at their
  boundaries (the tolerance is inclusive; a case-folded match counts; an unscorable gold is ``None``, not
  ``0.0``), and the measured release facts are pinned so a change to the corpus handling shows up here.

The corpus-dependent tests read ``data/raw/cyportqa`` and skip when it is not vendored, using the same
guard as the rest of the suite (``data/CONTRACT.md`` §2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from okh_rag.config import Config, EvaluationConfig, load_config
from okh_rag.data.cyportqa import AnswerFormat, QARecord
from okh_rag.evaluation.answers import (
    NUMERIC_GOLD_CAUSES,
    AccuracyReport,
    ScoringRegime,
    numeric_gold_cause,
    regime_of,
    score_answer,
    score_record,
    summarise,
    tolerance_for,
)
from okh_rag.generation.evidence import (
    EvidenceContext,
    EvidenceTrajectory,
    Passage,
    horizon_label,
    reasoning_tags,
    trajectory_from_hyperedge_ids,
)
from okh_rag.generation.generator import Generator, aggregate
from okh_rag.generation.output import (
    AnswerViolation,
    GeneratedAnswer,
    answer_json_schema,
    canonicalise,
    numeric_value,
    parse_generated_answer,
    validate,
    vote_key,
)
from okh_rag.generation.prompts import build_messages, output_contract
from okh_rag.llm.openai_client import _assert_strict_schema
from okh_rag.precedence import build_precedence
from okh_rag.retrieval import OrderAwareRetriever
from okh_rag.schema import Hyperedge, KnowledgeHypergraph
from okh_rag.vocabulary import is_cross_horizon

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "raw" / "cyportqa"
CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"

ALL_FORMATS = tuple(AnswerFormat)


def _data_available() -> bool:
    return (DATA_ROOT / "dataset" / "CyPortQA.json").exists()


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


# ---------------------------------------------------------------------------
# Hand-built hypergraphs
# ---------------------------------------------------------------------------


def _hypergraph() -> KnowledgeHypergraph:
    """A four-hyperedge graph covering four families and two horizons.

    Built by hand rather than taken from the release because the reasoning-tag rule is a function of
    *adjacency*, and a real trajectory's adjacency is whatever the retriever produced. Here the sequence is
    known: advisory at T-48, hazard at T-48 (a within-horizon causal hand-off), operation at T-48, then
    impact at T-12 (a horizon change).
    """
    from okh_rag.schema import Entity

    graph = KnowledgeHypergraph()
    rows = [
        ("h_watch", "has_watch_status", 48, "Tropical Storm Watch issued for the port."),
        ("h_hazard", "forecasts_hazard_at_horizon", 48, "Gale-force winds forecast within 24 hours."),
        ("h_operation", "has_operation_status", 48, "Pilot boarding suspended at the port."),
        ("h_impact", "has_impact_prediction", 12, "Cargo operations suspended for 18 hours."),
    ]
    for hyperedge_id, relation, horizon, evidence in rows:
        port_id = "port:shared"
        storm_id = "storm:shared"
        horizon_id = f"horizon:T-{horizon}"
        for entity_id, name, type_name in (
            (port_id, "Port of Test", "PORT"),
            (storm_id, "TEST", "CYCLONE"),
            (horizon_id, f"horizon:T-{horizon}", "HORIZON_TIME"),
        ):
            if entity_id not in graph.entities:
                graph.entities[entity_id] = Entity(
                    entity_id=entity_id, name=name, type=type_name, description="", confidence=1.0
                )
        graph.hyperedges[hyperedge_id] = Hyperedge(
            hyperedge_id=hyperedge_id,
            entity_ids=(port_id, storm_id, horizon_id),
            relation=relation,
            evidence=evidence,
            attributes={},
            group_id="group",
            horizon_hours=horizon,
            text_position=0,
        )
    graph.groups["group"] = list(graph.hyperedges)
    return graph


def _edges(graph: KnowledgeHypergraph) -> list[Hyperedge]:
    return [graph.hyperedges[h] for h in ("h_watch", "h_hazard", "h_operation", "h_impact")]


# ---------------------------------------------------------------------------
# Reasoning tags (§A.8's one open question)
# ---------------------------------------------------------------------------


class TestReasoningTags:
    def test_a_same_horizon_neighbour_yields_within_horizon(self):
        edges = _edges(_hypergraph())
        assert "within_horizon" in reasoning_tags(edges, 1)

    def test_a_phase_hand_off_inside_one_horizon_is_named(self):
        """§A.8's own example — ``hazard_to_operation`` — must be reachable without a horizon change.

        This is the test that pins the rule. An earlier implementation emitted the phase tag only when the
        horizon *changed*, which made the paper's example unreachable in exactly the within-horizon case the
        ordered evidence exists to express.
        """
        edges = _edges(_hypergraph())
        tags = reasoning_tags(edges, 2)
        assert "hazard_to_operation" in tags, tags
        assert "within_horizon" in tags, "the same pair is also a same-horizon continuation"

    def test_a_horizon_change_is_named_from_both_sides(self):
        edges = _edges(_hypergraph())
        assert "operation_to_impact" in reasoning_tags(edges, 3)
        assert "impact_to_operation" in reasoning_tags(edges, 2)

    def test_the_first_and_last_steps_still_get_a_tag(self):
        """A step with only one neighbour must not be tagless — that would make the ends look special."""
        edges = _edges(_hypergraph())
        assert reasoning_tags(edges, 0) == ("within_horizon", "hazard_to_watch") or (
            "within_horizon" in reasoning_tags(edges, 0)
        )
        assert reasoning_tags(edges, 3)

    def test_cross_horizon_edges_are_labelled_as_such(self):
        graph = _hypergraph()
        edge = Hyperedge(
            hyperedge_id="h_change",
            entity_ids=("port:shared", "storm:shared"),
            relation="forecast_updates_to",
            evidence="the forecast was revised",
            group_id="group",
            horizon_hours=48,
        )
        assert is_cross_horizon(edge.relation)
        graph.hyperedges["h_change"] = edge
        # The family label is ``cross_horizon`` rather than a Table 3 family, because family 13 is the one
        # §A.3's within-horizon ordering excludes — the fact the generator most needs to see.
        assert "cross_horizon" in reasoning_tags([_edges(graph)[0], edge], 1)
        trajectory = trajectory_from_hyperedge_ids(graph, ["h_watch", "h_change"], index=1)
        assert "family=cross_horizon" in trajectory.render(include_quality_summary=False)
        assert "cross_horizon" in trajectory.steps[1].reasoning

    def test_tags_are_capped_and_deduplicated(self):
        edges = _edges(_hypergraph())
        for index in range(len(edges)):
            tags = reasoning_tags(edges, index, max_tags=2)
            assert len(tags) == len(set(tags))
            assert len(tags) <= 2

    def test_an_out_of_range_index_is_an_error_not_an_empty_tuple(self):
        """Silently returning ``()`` would let a caller index past a trajectory and get blank annotations."""
        edges = _edges(_hypergraph())
        with pytest.raises(IndexError):
            reasoning_tags(edges, len(edges))


# ---------------------------------------------------------------------------
# Evidence rendering
# ---------------------------------------------------------------------------


class TestEvidenceRendering:
    def test_horizon_labels_use_the_paper_format(self):
        assert horizon_label(48) == "T-48"
        assert horizon_label(None) == "?"
        assert horizon_label(0) == "T-0"

    def test_an_ordered_step_renders_the_a8_block(self):
        graph = _hypergraph()
        trajectory = trajectory_from_hyperedge_ids(graph, list(graph.hyperedges), index=1, score=1.0)
        text = trajectory.render(include_quality_summary=False)
        assert "[Step 1] [T-48] [phase=advisory] [family=advisory]" in text
        assert "Relation: has_watch_status" in text
        assert "Reasoning: " in text
        assert "Entities: " in text

    def test_an_unordered_trajectory_never_emits_a_step_header_or_a_reasoning_line(self):
        """§4.1's HyperGraphRAG retrieves *without* order; the render must not hand the order back.

        Both halves are needed. A ``[Step k]`` header asserts adjacency, and the ``Reasoning:`` line is
        *derived from* adjacency, so for a set that was never ordered it would describe relationships the
        retriever never established.
        """
        graph = _hypergraph()
        trajectory = trajectory_from_hyperedge_ids(
            graph, list(graph.hyperedges), index=1, ordered=False
        )
        text = trajectory.render(include_quality_summary=False, include_reasoning=True)
        assert "[Step " not in text
        assert "[Evidence 1]" in text
        assert "Reasoning:" not in text

    def test_a_passage_carries_no_relation_no_entities_and_no_horizon(self):
        """Text-RAG's chunk is raw prose; annotating it would give the baseline a free representation."""
        passage = Passage(step=1, text="...T-48 hours before expected landfall: winds 40 kt", horizon_hours=48)
        rendered = passage.render()
        assert rendered.startswith("[Chunk 1]")
        assert "T-48" not in rendered.splitlines()[0]
        assert "Relation:" not in rendered
        assert "Entities:" not in rendered

    def test_the_quality_summary_is_absent_rather_than_zero_for_a_baseline(self):
        """``score=0.000`` would be a false claim: nothing evaluated Eq. 5 for a text chunk."""
        baseline = EvidenceTrajectory(index=1, steps=(Passage(step=1, text="x"),), ordered=False)
        assert baseline.quality_summary() is None
        assert "[Trajectory" not in baseline.render()

    def test_the_quality_summary_names_the_eq5_terms(self):
        graph = _hypergraph()
        trajectory = trajectory_from_hyperedge_ids(
            graph,
            list(graph.hyperedges),
            index=1,
            score=3.5,
            terms={"relevance": 2.0, "order_coherence": -0.5, "coverage": 0.25},
        )
        summary = trajectory.quality_summary()
        assert summary is not None
        assert "score=3.500" in summary
        assert "relevance=2.000" in summary
        assert "order_coherence=-0.500" in summary
        assert "coverage=0.250" in summary

    def test_unknown_hyperedge_ids_are_skipped_rather_than_raising(self):
        graph = _hypergraph()
        trajectory = trajectory_from_hyperedge_ids(
            graph, ["h_watch", "nosuch", "h_hazard"], index=1
        )
        assert len(trajectory) == 2
        assert [step.step for step in trajectory.steps] == [1, 2]

    def test_a_context_with_no_trajectory_is_empty(self):
        context = EvidenceContext(question="q", trajectories=(), source="text_rag")
        assert context.is_empty
        assert not context.is_multi_trajectory
        assert context.render() == ""

    def test_multi_trajectory_needs_two_non_empty_paths(self):
        step = Passage(step=1, text="x")
        one = EvidenceTrajectory(index=1, steps=(step,))
        empty = EvidenceTrajectory(index=2, steps=())
        assert not EvidenceContext("q", (one,), "okh_rag").is_multi_trajectory
        assert not EvidenceContext("q", (one, empty), "okh_rag").is_multi_trajectory
        assert EvidenceContext("q", (one, one), "okh_rag").is_multi_trajectory

    def test_trajectories_are_numbered_and_blank_line_separated(self):
        graph = _hypergraph()
        first = trajectory_from_hyperedge_ids(graph, ["h_watch"], index=1)
        second = trajectory_from_hyperedge_ids(graph, ["h_impact"], index=2)
        text = EvidenceContext("q", (first, second), "okh_rag").render(
            include_quality_summary=False
        )
        assert "[Step 1] [T-48]" in text
        assert "[Step 1] [T-12]" in text, "the step numbering restarts per trajectory"
        assert "\n\n" in text


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


class TestOutputContract:
    def test_a_well_formed_response_has_no_violations(self):
        answer = parse_generated_answer(
            '{"answer": "Yes", "confidence": 0.9, "rationale": "step 2"}', AnswerFormat.TRUE_FALSE
        )
        assert answer.is_conforming
        assert answer.answer == "Yes"
        assert answer.confidence == pytest.approx(0.9)
        assert answer.rationale == "step 2"

    def test_a_fenced_response_wrapped_in_prose_is_recovered(self):
        text = 'Sure!\n```json\n{"answer": "No", "confidence": 0.4, "rationale": "r"}\n```\nHope that helps.'
        answer = parse_generated_answer(text, AnswerFormat.TRUE_FALSE)
        assert answer.is_conforming
        assert answer.answer == "No"

    def test_a_braced_object_without_a_fence_is_recovered(self):
        text = 'Here is my answer: {"answer": "B", "confidence": 1, "rationale": "r"} done'
        answer = parse_generated_answer(text, AnswerFormat.MULTIPLE_CHOICE)
        assert answer.is_conforming

    def test_a_non_json_response_is_reported_not_raised(self):
        answer = parse_generated_answer("I think yes.", AnswerFormat.TRUE_FALSE)
        assert answer.violations == (AnswerViolation.NOT_JSON,)
        assert answer.answer == "I think yes."

    def test_a_json_array_is_not_an_answer_object(self):
        answer = parse_generated_answer('["Yes"]', AnswerFormat.TRUE_FALSE)
        assert AnswerViolation.NOT_JSON in answer.violations

    def test_a_missing_field_is_reported_once(self):
        answer = parse_generated_answer('{"confidence": 0.5}', AnswerFormat.DESCRIPTION)
        assert answer.violations.count(AnswerViolation.MISSING_FIELD) == 1
        assert AnswerViolation.WRONG_TYPE in answer.violations

    def test_an_out_of_range_confidence_is_rejected_not_clamped(self):
        """Clamping 85 to 1.0 would award a contract violation the most influential possible weight."""
        answer = parse_generated_answer(
            '{"answer": "Yes", "confidence": 85, "rationale": ""}', AnswerFormat.TRUE_FALSE
        )
        assert AnswerViolation.BAD_CONFIDENCE in answer.violations
        assert answer.confidence is None
        assert answer.weight == pytest.approx(1.0), "a missing confidence must not delete the vote"

    def test_a_boolean_confidence_is_not_a_confidence(self):
        answer = parse_generated_answer(
            '{"answer": "Yes", "confidence": true, "rationale": ""}', AnswerFormat.TRUE_FALSE
        )
        assert AnswerViolation.BAD_CONFIDENCE in answer.violations

    def test_a_string_confidence_in_range_is_accepted(self):
        answer = parse_generated_answer(
            '{"answer": "Yes", "confidence": "0.7", "rationale": ""}', AnswerFormat.TRUE_FALSE
        )
        assert answer.confidence == pytest.approx(0.7)

    @pytest.mark.parametrize(
        "answer,expected",
        [
            ("Yes", True),
            ("no", True),
            ("False", False),
            ("True", False),
            ("", False),
            (1, False),
        ],
    )
    def test_true_false_accepts_polarity_not_a_convention(self, answer, expected):
        """The check is about polarity: ``"False"`` is not a Yes/No answer, and ``"no"`` is a valid one."""
        violations = validate(answer, AnswerFormat.TRUE_FALSE)
        assert (not violations) is expected

    def test_multiple_choice_rejects_a_label_outside_the_options(self):
        options = (("A", "None"), ("B", "Storm"), ("C", "Hurricane"))
        assert validate("B", AnswerFormat.MULTIPLE_CHOICE, options=options) == ()
        assert validate("D", AnswerFormat.MULTIPLE_CHOICE, options=options) == (
            AnswerViolation.NOT_A_LABEL,
        )

    def test_multiple_choice_accepts_the_four_shapes_a_model_produces(self):
        for text in ("B", "b", "(B)", "B."):
            assert validate(text, AnswerFormat.MULTIPLE_CHOICE) == (), text

    def test_multiple_choice_rejects_a_paragraph(self):
        assert validate("B because of the wind", AnswerFormat.MULTIPLE_CHOICE) == (
            AnswerViolation.NOT_A_LABEL,
        )

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("48", 48.0),
            ("T-48", 48.0),
            ("40 mph", 40.0),
            ("-3.5", -3.5),
            (1500, 1500.0),
            ("5%", 5.0),
            ("None", None),
            ("N/A", None),
            ("", None),
            ("between 12 and 24", None),
            ("32.9N, 78.3W", None),
            ("{'in 12h': '1%'}", None),
        ],
    )
    def test_numeric_extraction(self, value, expected):
        assert numeric_value(value) == expected

    def test_a_horizon_label_is_not_a_negative_number(self):
        """``T-48`` means forty-eight hours. Reading the hyphen as a sign would negate every such answer."""
        assert numeric_value("T-48") == pytest.approx(48.0)

    def test_canonicalisation_is_the_single_definition_of_a_comparable_answer(self):
        assert canonicalise("  Yes  ", AnswerFormat.TRUE_FALSE) == "Yes"
        assert canonicalise("A.", AnswerFormat.MULTIPLE_CHOICE) == "A"
        assert canonicalise("(c)", AnswerFormat.MULTIPLE_CHOICE) == "C"
        assert canonicalise("T-24", AnswerFormat.NUMERIC) == pytest.approx(24.0)
        assert canonicalise("  spaced  ", AnswerFormat.DESCRIPTION) == "spaced"

    def test_canonicalisation_preserves_true_false_case(self):
        """Case is a *scoring* decision, not a canonicalisation one.

        Folding it here would make ``evaluation.case_sensitive`` inert — the switch could be set to ``true``
        and a lowercase answer would still match. :func:`~okh_rag.evaluation.answers.score_answer` applies
        the config's rule through its own comparison.
        """
        assert canonicalise("yes", AnswerFormat.TRUE_FALSE) == "yes"

    def test_canonicalisation_leaves_an_uninterpretable_answer_alone(self):
        """A mismatch report has to quote what the model actually said."""
        assert canonicalise("probably", AnswerFormat.TRUE_FALSE) == "probably"

    def test_the_vote_key_collapses_surface_variation(self):
        assert vote_key("b", AnswerFormat.MULTIPLE_CHOICE) == vote_key("(B)", AnswerFormat.MULTIPLE_CHOICE)
        assert vote_key("yes", AnswerFormat.TRUE_FALSE) == vote_key("Yes", AnswerFormat.TRUE_FALSE)

    def test_the_vote_key_keeps_distinct_descriptions_distinct(self):
        """Near-matching prose inside an integer counter would decide a semantic question by accident."""
        assert vote_key("heavy rain", AnswerFormat.DESCRIPTION) != vote_key("heavy rains", AnswerFormat.DESCRIPTION)

    @pytest.mark.parametrize("answer_format", ALL_FORMATS)
    @pytest.mark.parametrize("multi_trajectory", [False, True])
    def test_every_schema_variant_satisfies_openai_strict_mode(self, answer_format, multi_trajectory):
        """A property the API enforces at call time and that would otherwise fail 117k calls in."""
        schema = answer_json_schema(
            answer_format,
            options=(("A", "x"), ("B", "y")),
            multi_trajectory=multi_trajectory,
        )
        _assert_strict_schema(schema)
        assert ("supporting_trajectory" in schema["properties"]) is multi_trajectory

    def test_the_schema_constrains_the_answer_field_per_format(self):
        assert answer_json_schema(AnswerFormat.TRUE_FALSE)["properties"]["answer"]["enum"] == ["Yes", "No"]
        assert answer_json_schema(AnswerFormat.NUMERIC)["properties"]["answer"]["type"] == "number"
        labels = answer_json_schema(
            AnswerFormat.MULTIPLE_CHOICE, options=(("A", "x"), ("B", "y"))
        )["properties"]["answer"]["enum"]
        assert labels == ["A", "B"]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _record(**overrides: Any) -> QARecord:
    """A minimal, well-formed TF record. Overrides let each test state only what it is about."""
    base: dict[str, Any] = {
        "scenario_id": "TEST_2020_Test_Port,_XX",
        "template_id": "Q1",
        "context": _context(),
        "tags": ("TF", "S1"),
        "answer_format": AnswerFormat.TRUE_FALSE,
        "stage": "S1",
        "dimensions": (),
        "annotations": (),
        "modalities": ("text_advisory",),
        "prompt": "Act as a cyclone and port operations analyst.",
        "question": "Is the port inside the forecast cone? Answer with ONLY 'Yes' or 'No'.",
        "answer": "Yes",
    }
    base.update(overrides)
    return QARecord(**base)


def _context():
    from okh_rag.data.cyportqa import QAContext

    return QAContext(storm="TEST", year=2020, horizon_hours=48, port="Test Port")


class TestPrompts:
    def test_the_release_protocol_is_honoured(self):
        """``prompt`` as system and ``question`` as user — the released runners' own protocol.

        ``data/CONTRACT.md`` §2.2 establishes this from ``models/run_gpt4o.py``. Replacing the release's
        framing with a generic persona would change the input every compared method receives, which is the
        one thing a baseline study may not do.
        """
        messages = build_messages(_record(), EvidenceContext("q", (), "text_rag"))
        assert messages[0]["role"] == "system"
        assert messages[0]["content"].startswith("Act as a cyclone")
        assert messages[1]["role"] == "user"
        assert messages[1]["content"].startswith("Is the port inside the forecast cone?")

    def test_a_missing_system_prompt_falls_back_rather_than_sending_an_empty_message(self):
        messages = build_messages(_record(prompt=""), EvidenceContext("q", (), "text_rag"))
        assert messages[0]["content"].strip()

    def test_the_output_contract_overrides_the_releases_format_clause(self):
        """Both cannot hold; the JSON envelope is placed after the question so it is operative."""
        messages = build_messages(_record(), EvidenceContext("q", (), "text_rag"))
        user = messages[1]["content"]
        assert "ONLY 'Yes' or 'No'" in user, "the release's question text is passed through verbatim"
        assert '"answer"' in user and '"confidence"' in user
        assert user.index("ONLY 'Yes' or 'No'") < user.index('"answer"')

    def test_an_empty_retrieval_is_stated_rather_than_left_blank(self):
        messages = build_messages(_record(), EvidenceContext("q", (), "text_rag"))
        assert "no evidence steps were retrieved" in messages[1]["content"].lower()

    def test_the_multi_trajectory_instruction_appears_only_with_several_paths(self):
        step = Passage(step=1, text="x")
        one = EvidenceContext("q", (EvidenceTrajectory(1, (step,)),), "okh_rag")
        two = EvidenceContext("q", (EvidenceTrajectory(1, (step,)), EvidenceTrajectory(2, (step,))), "okh_rag")
        assert "supporting_trajectory" not in build_messages(_record(), one)[1]["content"]
        assert "supporting_trajectory" in build_messages(_record(), two)[1]["content"]

    def test_the_mc_contract_names_the_admissible_labels(self):
        """2,851 release questions state a stale option count, so "one of the letters above" is ambiguous."""
        contract = output_contract(
            AnswerFormat.MULTIPLE_CHOICE, options=(("A", "None"), ("B", "Tropical Storm"))
        )
        assert '"A" = None' in contract
        assert '"B" = Tropical Storm' in contract

    def test_the_source_is_named_in_the_evidence_section(self):
        messages = build_messages(_record(), EvidenceContext("q", (), "hypergraph_rag"))
        assert "hypergraph_rag" in messages[1]["content"]


# ---------------------------------------------------------------------------
# Aggregation (§A.8's fallback mode)
# ---------------------------------------------------------------------------


def _conforming(answer: Any, confidence: float, rationale: str = "r") -> GeneratedAnswer:
    return GeneratedAnswer(answer=answer, confidence=confidence, rationale=rationale)


class TestAggregation:
    def test_categorical_voting_is_weighted_by_confidence(self):
        """Two weak ``No``s must lose to one strong ``Yes``: this is what "weighted" means."""
        answers = [
            _conforming("Yes", 0.9),
            _conforming("No", 0.1),
            _conforming("No", 0.1),
        ]
        aggregated, notes = aggregate(answers, AnswerFormat.TRUE_FALSE)
        assert aggregated.answer == "Yes"
        assert any("voting" in note for note in notes)

    def test_the_aggregate_confidence_is_the_winning_share_not_the_winner_s_own(self):
        """Reporting 0.9 on a 3-way split would discard the disagreement the mode exists to surface."""
        answers = [_conforming("Yes", 0.9), _conforming("No", 0.1), _conforming("No", 0.1)]
        aggregated, _ = aggregate(answers, AnswerFormat.TRUE_FALSE)
        assert aggregated.confidence == pytest.approx(0.9 / 1.1)

    def test_surface_variation_does_not_split_the_vote(self):
        answers = [_conforming("yes", 0.4), _conforming("Yes", 0.4)]
        aggregated, _ = aggregate(answers, AnswerFormat.TRUE_FALSE)
        assert aggregated.confidence == pytest.approx(1.0), "case-only variation must not split the vote"
        assert aggregated.answer.strip().casefold() == "yes"

    def test_a_tie_goes_to_the_earlier_trajectory(self):
        """Trajectories arrive best-first, so "earlier" is the only ordering information available."""
        answers = [_conforming("Yes", 0.5), _conforming("No", 0.5)]
        aggregated, _ = aggregate(answers, AnswerFormat.TRUE_FALSE)
        assert aggregated.answer == "Yes"

    def test_numeric_answers_are_averaged_not_voted_on(self):
        answers = [_conforming(48, 1.0), _conforming(50, 1.0)]
        aggregated, notes = aggregate(answers, AnswerFormat.NUMERIC)
        assert aggregated.answer == pytest.approx(49.0)
        assert any("averaging" in note for note in notes)

    def test_numeric_averaging_respects_the_weights(self):
        answers = [_conforming(10, 0.75), _conforming(30, 0.25)]
        aggregated, _ = aggregate(answers, AnswerFormat.NUMERIC)
        assert aggregated.answer == pytest.approx(15.0)

    def test_a_spread_out_numeric_answer_lowers_the_confidence(self):
        tight, _ = aggregate([_conforming(48, 1.0), _conforming(49, 1.0)], AnswerFormat.NUMERIC)
        loose, _ = aggregate([_conforming(1, 1.0), _conforming(99, 1.0)], AnswerFormat.NUMERIC)
        assert loose.confidence < tight.confidence

    def test_non_conforming_answers_are_dropped_and_counted(self):
        """A TF answer of ``"maybe"`` cannot be voted against ``"Yes"``; letting it in would give a
        contract violation the same standing as a valid answer."""
        answers = [
            _conforming("Yes", 0.3),
            GeneratedAnswer(answer="maybe", violations=(AnswerViolation.NOT_YES_NO,)),
        ]
        aggregated, notes = aggregate(answers, AnswerFormat.TRUE_FALSE)
        assert aggregated.answer == "Yes"
        assert any("dropped 1/2" in note for note in notes)

    def test_when_everything_is_non_conforming_the_first_is_reported_unaggregated(self):
        answers = [GeneratedAnswer(answer="maybe", violations=(AnswerViolation.NOT_YES_NO,))]
        aggregated, notes = aggregate(answers, AnswerFormat.TRUE_FALSE)
        assert aggregated.answer == "maybe"
        assert any("unaggregated" in note for note in notes)

    def test_a_single_conforming_answer_is_returned_unchanged(self):
        answers = [_conforming("Yes", 0.7)]
        aggregated, notes = aggregate(answers, AnswerFormat.TRUE_FALSE)
        assert aggregated.answer == "Yes"
        assert aggregated.confidence == pytest.approx(0.7)
        assert any("no aggregation" in note for note in notes)

    def test_no_answers_produces_an_empty_result_rather_than_an_exception(self):
        aggregated, notes = aggregate([], AnswerFormat.TRUE_FALSE)
        assert aggregated.answer is None
        assert any("no per-trajectory answers" in note for note in notes)

    def test_the_rationale_comes_from_a_single_member_not_a_concatenation(self):
        answers = [_conforming("Yes", 0.9, "the watch was issued"), _conforming("Yes", 0.1, "because")]
        aggregated, _ = aggregate(answers, AnswerFormat.TRUE_FALSE)
        assert aggregated.rationale == "the watch was issued"


# ---------------------------------------------------------------------------
# The generator, over a scripted backend
# ---------------------------------------------------------------------------


class ScriptedBackend:
    """An ``LLMClient`` that answers from a queue and records every request.

    Recording the request is the point: the claim that ``single_call`` and ``per_trajectory`` differ only
    in how the paths are reconciled is a claim about *what the model was shown*, and it cannot be checked
    from the answers alone.
    """

    def __init__(self, responses: Sequence[Mapping[str, Any] | str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete_text(self, messages, *, model, temperature):  # pragma: no cover - unused
        raise AssertionError("generation uses complete_json")

    def complete_json(self, messages, *, model, temperature, schema, schema_name):
        self.calls.append(
            {"messages": messages, "schema": schema, "model": model, "temperature": temperature}
        )
        if not self._responses:
            raise AssertionError("the scripted backend ran out of responses")
        return self._responses.pop(0)


def _two_path_context(source: str = "okh_rag") -> EvidenceContext:
    step = Passage(step=1, text="winds 40 kt")
    return EvidenceContext(
        question="Is the port inside the cone?",
        trajectories=(
            EvidenceTrajectory(index=1, steps=(step,), score=1.0),
            EvidenceTrajectory(index=2, steps=(step,), score=0.5),
        ),
        source=source,
    )


class TestGenerator:
    def test_single_call_sends_one_request_for_two_paths(self, cfg: Config):
        backend = ScriptedBackend([{"answer": "Yes", "confidence": 0.8, "rationale": "both paths agree",
                                    "supporting_trajectory": "1"}])
        generator = Generator(backend, cfg)
        record = generator.answer(_record(), _two_path_context(), mode="single_call")
        assert len(backend.calls) == 1
        assert record.answer.is_conforming
        assert record.mode == "single_call"
        assert record.trajectories == ()

    def test_single_call_asks_for_the_supporting_trajectory_in_the_schema(self, cfg: Config):
        backend = ScriptedBackend([{"answer": "Yes", "confidence": 0.8, "rationale": "r",
                                    "supporting_trajectory": "2"}])
        Generator(backend, cfg).answer(_record(), _two_path_context(), mode="single_call")
        assert "supporting_trajectory" in backend.calls[0]["schema"]["required"]

    def test_single_call_omits_the_field_when_there_is_one_path(self, cfg: Config):
        """A required-but-meaningless field invites a fabricated path number."""
        backend = ScriptedBackend([{"answer": "Yes", "confidence": 0.8, "rationale": "r"}])
        single = EvidenceContext(
            "q", (EvidenceTrajectory(1, (Passage(step=1, text="x"),)),), "okh_rag"
        )
        Generator(backend, cfg).answer(_record(), single, mode="single_call")
        assert "supporting_trajectory" not in backend.calls[0]["schema"]["properties"]

    def test_per_trajectory_sends_one_request_per_path_and_isolates_them(self, cfg: Config):
        backend = ScriptedBackend(
            [
                {"answer": "Yes", "confidence": 0.9, "rationale": "first"},
                {"answer": "No", "confidence": 0.2, "rationale": "second"},
            ]
        )
        record = Generator(backend, cfg).answer(_record(), _two_path_context(), mode="per_trajectory")
        assert len(backend.calls) == 2
        assert len(record.trajectories) == 2
        for call in backend.calls:
            user = call["messages"][1]["content"]
            # One path per request: the multi-trajectory instruction must not be present, which is what
            # makes this a genuine per-path answer rather than a differently-worded single call.
            assert "supporting_trajectory" not in user
            assert "Several numbered reasoning paths" not in user

    def test_per_trajectory_aggregates_and_records_which_rule_was_used(self, cfg: Config):
        backend = ScriptedBackend(
            [
                {"answer": "Yes", "confidence": 0.9, "rationale": "first"},
                {"answer": "No", "confidence": 0.2, "rationale": "second"},
            ]
        )
        record = Generator(backend, cfg).answer(_record(), _two_path_context(), mode="per_trajectory")
        assert record.answer.answer == "Yes"
        assert any("voting" in note for note in record.notes)

    def test_the_path_number_is_filled_in_when_the_model_omits_it(self, cfg: Config):
        backend = ScriptedBackend(
            [
                {"answer": "Yes", "confidence": 0.5, "rationale": "a"},
                {"answer": "Yes", "confidence": 0.5, "rationale": "b"},
            ]
        )
        record = Generator(backend, cfg).answer(_record(), _two_path_context(), mode="per_trajectory")
        assert [a.supporting_trajectory for a in record.trajectories] == ["1", "2"]

    def test_a_non_json_response_does_not_abort_the_run(self, cfg: Config):
        backend = ScriptedBackend(["I refuse to answer in JSON."])
        record = Generator(backend, cfg).answer(_record(), _two_path_context(), mode="single_call")
        assert not record.answer.is_conforming
        assert record.answer.violations == (AnswerViolation.NOT_JSON,)

    def test_the_shuffle_note_reaches_the_model(self, cfg: Config):
        context = EvidenceContext(
            question="q",
            trajectories=(EvidenceTrajectory(1, (Passage(step=1, text="x"),)),),
            source="okh_rag",
            notes=("Note: the retrieved trajectory was order-shuffled.",),
        )
        assert "order-shuffled" in Generator(ScriptedBackend([]), cfg).messages_for(_record(), context)[1][
            "content"
        ]

    def test_an_unknown_mode_is_rejected_before_any_call(self, cfg: Config):
        backend = ScriptedBackend([])
        with pytest.raises(ValueError, match="unknown generation mode"):
            Generator(backend, cfg).answer(_record(), _two_path_context(), mode="nope")
        assert backend.calls == []

    def test_the_record_carries_the_identity_a_report_keys_on(self, cfg: Config):
        backend = ScriptedBackend([{"answer": "Yes", "confidence": 0.5, "rationale": "r"}])
        record = Generator(backend, cfg).answer(_record(), _two_path_context(), mode="single_call")
        assert record.template_id == "Q1"
        assert record.scenario_id == "TEST_2020_Test_Port,_XX"
        assert record.source == "okh_rag"
        assert record.as_dict()["answer_format"] == "TF"


# ---------------------------------------------------------------------------
# §4.1 scoring
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def evaluation_cfg() -> EvaluationConfig:
    return load_config(CONFIG_PATH).evaluation


class TestScoring:
    def test_regimes_map_onto_the_papers_four_types(self):
        assert regime_of(AnswerFormat.TRUE_FALSE) is ScoringRegime.EXACT
        assert regime_of(AnswerFormat.MULTIPLE_CHOICE) is ScoringRegime.EXACT
        assert regime_of(AnswerFormat.NUMERIC) is ScoringRegime.TOLERANCE
        assert regime_of(AnswerFormat.DESCRIPTION) is ScoringRegime.SEMANTIC

    def test_exact_match_ignores_case_by_default(self, evaluation_cfg):
        score = score_answer("yes", "Yes", AnswerFormat.TRUE_FALSE, evaluation_cfg)
        assert score.correct == pytest.approx(1.0)
        assert score.reason == "match"

    def test_case_sensitivity_can_be_turned_on(self, evaluation_cfg):
        import dataclasses

        strict = dataclasses.replace(evaluation_cfg, case_sensitive=True)
        assert score_answer("yes", "Yes", AnswerFormat.TRUE_FALSE, strict).correct == pytest.approx(0.0)

    def test_multiple_choice_matches_the_releases_trailing_period(self, evaluation_cfg):
        """The release stores ``"A."``; requiring the model to reproduce the period would score a correct
        answer wrong for a publisher's punctuation habit."""
        assert score_answer("A", "A.", AnswerFormat.MULTIPLE_CHOICE, evaluation_cfg).correct == pytest.approx(1.0)
        assert score_answer("(a)", "A.", AnswerFormat.MULTIPLE_CHOICE, evaluation_cfg).matched is True

    def test_a_wrong_letter_is_wrong(self, evaluation_cfg):
        assert score_answer("B", "A.", AnswerFormat.MULTIPLE_CHOICE, evaluation_cfg).correct == pytest.approx(0.0)

    def test_a_non_numeric_answer_to_a_numeric_question_is_wrong_not_unscorable(self, evaluation_cfg):
        """The gold *is* scorable; the model failed. Conflating the two would inflate the accuracy."""
        score = score_answer("about forty", "40", AnswerFormat.NUMERIC, evaluation_cfg)
        assert score.correct == pytest.approx(0.0)
        assert score.reason == "mismatch"

    def test_the_tolerance_is_inclusive_and_relative(self, evaluation_cfg):
        tolerance = tolerance_for("100", evaluation_cfg)
        assert tolerance == pytest.approx(1.0)
        assert score_answer(101.0, "100", AnswerFormat.NUMERIC, evaluation_cfg).matched is True
        assert score_answer(101.0001, "100", AnswerFormat.NUMERIC, evaluation_cfg).matched is False

    def test_the_absolute_floor_protects_a_near_zero_target(self, evaluation_cfg):
        """With a purely relative tolerance a gold of 0 could only be matched exactly."""
        tolerance = tolerance_for("0", evaluation_cfg)
        assert tolerance == pytest.approx(evaluation_cfg.sa_absolute_floor)
        assert score_answer(0, "0", AnswerFormat.NUMERIC, evaluation_cfg).matched is True

    def test_a_numeric_gold_that_is_not_a_number_is_unscorable_not_wrong(self, evaluation_cfg):
        """``32.9N, 78.3W`` is Q6's gold; no tolerance test exists for it, so 0.0 would be a fabrication."""
        score = score_answer("32.9N, 78.3W", "32.9N, 78.3W", AnswerFormat.NUMERIC, evaluation_cfg)
        assert score.correct is None
        assert score.reason == "not_numeric_gold"
        assert not score.scorable
        assert "coordinates" in score.detail

    def test_prose_is_matched_after_folding_case_and_whitespace(self, evaluation_cfg):
        score = score_answer(
            "South Santee   River SC", "South Santee River SC", AnswerFormat.DESCRIPTION, evaluation_cfg
        )
        assert score.correct == pytest.approx(1.0)

    def test_prose_that_differs_is_wrong(self, evaluation_cfg):
        score = score_answer("somewhere else", "South Santee River SC", AnswerFormat.DESCRIPTION, evaluation_cfg)
        assert score.correct == pytest.approx(0.0)

    def test_a_requested_judge_that_is_absent_yields_unscorable_not_zero(self, evaluation_cfg):
        import dataclasses

        judged = dataclasses.replace(evaluation_cfg, td_scoring="llm_judge")
        score = score_answer("prose", "prose", AnswerFormat.DESCRIPTION, judged)
        assert score.correct is None
        assert score.reason == "no_judge"

    def test_the_judge_is_used_and_clamped(self, evaluation_cfg):
        import dataclasses

        judged = dataclasses.replace(evaluation_cfg, td_scoring="llm_judge")
        score = score_answer(
            "anything",
            "gold",
            AnswerFormat.DESCRIPTION,
            judged,
            judge=lambda question, gold, predicted: 0.6,
        )
        assert score.correct == pytest.approx(0.6)
        assert score.reason == "partial"

        clamped = score_answer(
            "anything", "gold", AnswerFormat.DESCRIPTION, judged, judge=lambda *a: 1.7
        )
        assert clamped.correct == pytest.approx(1.0)

    def test_a_raising_judge_is_reported_not_propagated(self, evaluation_cfg):
        import dataclasses

        def boom(*_args):
            raise RuntimeError("judge exploded")

        judged = dataclasses.replace(evaluation_cfg, td_scoring="llm_judge")
        score = score_answer("a", "b", AnswerFormat.DESCRIPTION, judged, judge=boom)
        assert score.correct is None
        assert score.reason == "judge_error"

    def test_an_unanswerable_record_is_excluded_rather_than_scored_wrong(self, evaluation_cfg):
        record = _record(answer="None")
        assert not record.is_answerable
        stored = score_record(record, "Yes", evaluation_cfg)
        assert stored.correct is None
        assert stored.reason == "unanswerable"

    def test_including_unanswerable_records_is_possible_and_marks_them_wrong(self, evaluation_cfg):
        import dataclasses

        lenient = dataclasses.replace(evaluation_cfg, exclude_unanswerable=False)
        record = _record(answer="None")
        assert score_record(record, "Yes", lenient).correct == pytest.approx(0.0)


class TestAccuracyReport:
    def test_per_format_accuracy_and_denominators(self, evaluation_cfg):
        report = AccuracyReport()
        report.add(score_answer("Yes", "Yes", AnswerFormat.TRUE_FALSE, evaluation_cfg), AnswerFormat.TRUE_FALSE)
        report.add(score_answer("No", "Yes", AnswerFormat.TRUE_FALSE, evaluation_cfg), AnswerFormat.TRUE_FALSE)
        report.add(score_answer("A", "A.", AnswerFormat.MULTIPLE_CHOICE, evaluation_cfg), AnswerFormat.MULTIPLE_CHOICE)
        assert report.accuracy(AnswerFormat.TRUE_FALSE) == pytest.approx(0.5)
        assert report.counts(AnswerFormat.TRUE_FALSE) == (2, 2)
        assert report.accuracy(AnswerFormat.MULTIPLE_CHOICE) == pytest.approx(1.0)

    def test_a_format_with_no_scorable_question_has_no_accuracy_not_zero(self, evaluation_cfg):
        """Table 1 has a column per format; ``0.0`` for an absent format would read as total failure."""
        report = AccuracyReport()
        report.add(score_answer("Yes", "Yes", AnswerFormat.TRUE_FALSE, evaluation_cfg), AnswerFormat.TRUE_FALSE)
        assert report.accuracy(AnswerFormat.NUMERIC) is None
        assert report.by_format()["NU"] is None

    def test_micro_and_macro_disagree_when_the_formats_are_unequal(self, evaluation_cfg):
        """§4.1's "Overall" is not the macro-average of its own columns, so both are computed."""
        report = AccuracyReport()
        for _ in range(3):
            report.add(score_answer("Yes", "Yes", AnswerFormat.TRUE_FALSE, evaluation_cfg), AnswerFormat.TRUE_FALSE)
        report.add(score_answer("X", "A.", AnswerFormat.MULTIPLE_CHOICE, evaluation_cfg), AnswerFormat.MULTIPLE_CHOICE)
        assert report.micro() == pytest.approx(0.75)
        assert report.macro() == pytest.approx((1.0 + 0.0) / 2)
        assert report.overall("micro") == pytest.approx(0.75)
        assert report.overall("macro") == pytest.approx(0.5)

    def test_an_unknown_overall_mode_is_rejected(self):
        with pytest.raises(ValueError, match="unknown overall mode"):
            AccuracyReport().overall("median")

    def test_unscorable_questions_leave_the_denominator_and_are_counted(self, evaluation_cfg):
        report = AccuracyReport()
        report.add(score_answer("Yes", "Yes", AnswerFormat.TRUE_FALSE, evaluation_cfg), AnswerFormat.TRUE_FALSE)
        report.add(
            score_answer("N/A", "N/A", AnswerFormat.NUMERIC, evaluation_cfg), AnswerFormat.NUMERIC
        )
        assert report.micro() == pytest.approx(1.0), "the unscorable question must not drag the mean down"
        assert report.exclusion_summary()["unscorable"] == 1
        assert report.as_dict()["reasons"]["not_numeric_gold"] == 1

    def test_the_json_view_keeps_every_accuracy_next_to_its_denominator(self, evaluation_cfg):
        report = AccuracyReport()
        report.add(score_answer("Yes", "Yes", AnswerFormat.TRUE_FALSE, evaluation_cfg), AnswerFormat.TRUE_FALSE)
        payload = report.as_dict()
        assert payload["counts"]["TF"] == [1, 1]
        assert payload["overall_mode"] == "micro"
        assert set(payload["by_format"]) == {fmt.value for fmt in AnswerFormat}

    def test_summarise_keeps_format_and_score_aligned(self, evaluation_cfg):
        scores = [
            (AnswerFormat.TRUE_FALSE, score_answer("Yes", "Yes", AnswerFormat.TRUE_FALSE, evaluation_cfg)),
            (AnswerFormat.MULTIPLE_CHOICE, score_answer("A", "B.", AnswerFormat.MULTIPLE_CHOICE, evaluation_cfg)),
        ]
        report = summarise(scores)
        assert report.counts(AnswerFormat.TRUE_FALSE) == (1, 1)
        assert report.counts(AnswerFormat.MULTIPLE_CHOICE) == (1, 1)
        assert report.accuracy(AnswerFormat.TRUE_FALSE) == pytest.approx(1.0)
        assert report.accuracy(AnswerFormat.MULTIPLE_CHOICE) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Measured facts about the release
# ---------------------------------------------------------------------------


@requires_data
class TestMeasuredReleaseFacts:
    """Counts the module docstrings quote. Pinned here so a corpus-handling change cannot drift silently."""

    @pytest.fixture(scope="class")
    @staticmethod
    def records():
        from okh_rag.data.cyportqa import load_corpus

        corpus = load_corpus(DATA_ROOT)
        return [record for group in corpus.qa.values() for record in group]

    def test_the_four_formats_are_fully_scorable_except_numeric(self, records, evaluation_cfg):
        totals: dict[AnswerFormat, int] = {}
        unscorable: dict[AnswerFormat, int] = {}
        for record in records:
            totals[record.answer_format] = totals.get(record.answer_format, 0) + 1
            if score_answer(None, record.answer, record.answer_format, evaluation_cfg).correct is None:
                unscorable[record.answer_format] = unscorable.get(record.answer_format, 0) + 1
        assert unscorable.get(AnswerFormat.TRUE_FALSE, 0) == 0
        assert unscorable.get(AnswerFormat.MULTIPLE_CHOICE, 0) == 0
        assert unscorable.get(AnswerFormat.DESCRIPTION, 0) == 0
        assert unscorable[AnswerFormat.NUMERIC] == 19574
        assert totals[AnswerFormat.NUMERIC] == 49459

    def test_the_numeric_unscorable_causes_are_the_documented_ones(self, records):
        from collections import Counter

        causes = Counter(
            numeric_gold_cause(record.answer)
            for record in records
            if record.answer_format is AnswerFormat.NUMERIC
        )
        assert causes.pop(None) == 29885, "scorable numeric questions"
        assert dict(causes) == {
            "absent": 8896,
            "date": 4728,
            "coordinates": 2910,
            "multi_horizon": 2908,
            "range": 62,
            "prose": 51,
            "other": 19,
        }
        assert set(causes) <= set(NUMERIC_GOLD_CAUSES)

    def test_every_numeric_gold_that_is_unscorable_has_a_named_cause(self, records):
        for record in records:
            if record.answer_format is not AnswerFormat.NUMERIC:
                continue
            if numeric_value(record.answer) is None:
                assert numeric_gold_cause(record.answer) in NUMERIC_GOLD_CAUSES

    def test_q33_and_q6_are_entirely_unscorable_and_that_is_the_release_s_fault(self, records):
        """Q33 ships a Python ``repr`` of a per-horizon mapping and Q6 a coordinate pair — both tagged NU."""
        for template_id, cause in (("Q33", "multi_horizon"), ("Q6", "coordinates")):
            rows = [r for r in records if r.template_id == template_id]
            assert rows, f"{template_id} is missing from the release"
            assert all(numeric_gold_cause(r.answer) == cause for r in rows), template_id


# ---------------------------------------------------------------------------
# The retrieval-to-generation handoff, on the release
# ---------------------------------------------------------------------------


@requires_data
class TestContextFromRetrieval:
    """The one seam between the two systems: a real ``RetrievalResult`` rendered as §A.8 evidence."""

    @pytest.fixture(scope="class")
    @staticmethod
    def scenario_graph(cfg: Config):
        from okh_rag.data.cyportqa import CyPortQACorpus, load_encoded
        from okh_rag.evaluation.reference_graph import build_reference_graph

        encoded = load_encoded(DATA_ROOT)
        corpus = CyPortQACorpus(qa={}, encoded=encoded, templates={})
        return build_reference_graph(
            corpus, cfg=cfg, scenario_ids=["DORIAN_2019_Port_of_Charleston,_SC"]
        )

    def test_a_real_retrieval_renders_as_numbered_ordered_steps(self, cfg, scenario_graph):
        from okh_rag.embeddings import HyperedgeEmbeddings, embed_texts
        from okh_rag.evaluation.oracle import gold_for_record
        from okh_rag.generation.evidence import context_from_retrieval

        graph = scenario_graph
        ordered = sorted(graph.hypergraph.hyperedges)
        rng = np.random.default_rng(0)
        matrix = rng.standard_normal((len(ordered), cfg.order_model.embedding_dim)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        embeddings = HyperedgeEmbeddings(hyperedge_ids=tuple(ordered), matrix=matrix)

        retriever = OrderAwareRetriever(
            cfg=cfg,
            hypergraph=graph.hypergraph,
            precedence=build_precedence(graph.hypergraph, cfg.precedence),
            embeddings=embeddings,
            backend=None,
            transition_variant="heuristic",
        )
        from okh_rag.data.cyportqa import load_corpus

        corpus = load_corpus(DATA_ROOT)
        record = corpus.qa["DORIAN_2019_Port_of_Charleston,_SC"][0]
        query_vector = embed_texts(
            [record.question], backend=_DeterministicEmbedder(cfg.order_model.embedding_dim), cfg=cfg
        )[0]
        result = retriever.retrieve(record.question, group_id=record.group_id, query_vector=query_vector)
        assert result.trajectories, "retrieval produced no trajectory; the rest of this test is vacuous"

        context = context_from_retrieval(result, graph.hypergraph, question=record.question)
        assert context.source == "okh_rag"
        assert len(context) == len(result.trajectories)
        rendered = context.render()
        assert "[Step 1]" in rendered
        assert "Reasoning:" in rendered
        assert "[T-" in rendered
        assert "phase=" in rendered and "family=" in rendered
        # Every trajectory's steps must be a prefix of what the retriever returned: the render must not
        # reorder. This is the §4.5 property the whole evidence format exists to preserve.
        for trajectory, evidence_trajectory in zip(result.trajectories, context.trajectories, strict=True):
            assert len(evidence_trajectory) == len(trajectory)
        assert gold_for_record is not None  # imported for the oracle's gold definition; unused here


class _DeterministicEmbedder:
    """A hash-based embedder, so this test needs no API key and no cache."""

    def __init__(self, dim: int) -> None:
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in str(text).lower().split():
                out[row, hash(token) % self._dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)
