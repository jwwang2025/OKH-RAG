"""§4.1 — tests for the released CyPortQA data layer.

Two groups:

* **Offline unit tests** for the pure functions — option parsing, ordinal buckets, threshold
  extraction, normalisation. These need no data and always run.
* **Corpus tests** that read ``data/raw/cyportqa``. They are skipped when the benchmark is not
  vendored, since ``data/raw`` is deliberately git-ignored: it is 233 MB and reproducible from
  the upstream repository in one command (see ``data/CONTRACT.md`` §2). To run them, follow
  that section first.

The corpus tests assert the *measured* properties of the release — the counts, the arity
split, the defects — because those are exactly the facts the rest of the reproduction depends
on. If upstream re-releases the data and these fail, that is the intended signal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from okh_rag.data.cyportqa import (
    HORIZON_HOURS,
    KNOWN_DEFECTS,
    PORT_CONDITION_ALIASES,
    AnswerFormat,
    is_placeholder_answer,
    load_corpus,
    load_imbalanced,
    parse_inline_options,
    resolve_mc_letter,
)
from okh_rag.data.groundtruth import (
    MISSING,
    VerdictStatus,
    bucket_letters,
    compare,
    evaluate_predicate,
    expected_path,
    leading_clause,
    normalise,
    normalise_ssh_category,
    parse_ordinal,
    parse_threshold,
    resolve_path,
)

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "cyportqa"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _data_available() -> bool:
    return (DATA_ROOT / "dataset" / "CyPortQA.json").exists() and (
        DATA_ROOT / "source_data" / "Encoded_senario.json"
    ).exists()


requires_data = pytest.mark.skipif(
    not _data_available(),
    reason="CyPortQA not vendored under data/raw/cyportqa; see data/CONTRACT.md section 2",
)


@pytest.fixture(scope="module")
def corpus():
    """The joined corpus. Module-scoped: parsing 233 MB once is enough for the whole file."""
    if not _data_available():
        pytest.skip("CyPortQA not vendored")
    return load_corpus(DATA_ROOT)


# ======================================================================================
# Offline unit tests
# ======================================================================================


class TestParseInlineOptions:
    """The inlined option list, recovered from a question string."""

    def test_four_options_after_a_count_boilerplate(self):
        # The '(A, B, C, or D).' clause is boilerplate and must not be mistaken for options;
        # only the trailing consecutive run starting at A counts.
        question = (
            "Is storm surge expected to exceed 3 feet? Answer with ONLY a choice "
            "(A, B, C, or D). A. Unlikely B. No C. Yes D. Possible"
        )
        assert parse_inline_options(question) == (
            ("A", "Unlikely"),
            ("B", "No"),
            ("C", "Yes"),
            ("D", "Possible"),
        )

    def test_seven_options(self):
        question = (
            "What is the SSHS category? Answer with ONLY a choice (A, B, C, or D). "
            "A. Tropical Depression B. Tropical Storm C. Category 1 D. Category 2 "
            "E. Category 3 F. Category 4 G. Category 5"
        )
        options = parse_inline_options(question)
        assert len(options) == 7
        assert options[-1] == ("G", "Category 5")

    def test_no_options(self):
        assert parse_inline_options("Answer with ONLY a number.") == ()

    def test_option_text_containing_a_decimal_is_not_truncated(self):
        # '1.5 feet' must not look like an option marker at '5'.
        question = "Pick one. A. Under 1.5 feet B. Over 1.5 feet"
        assert parse_inline_options(question) == (("A", "Under 1.5 feet"), ("B", "Over 1.5 feet"))

    def test_letter_to_text(self):
        options = (("A", "Unlikely"), ("B", "No"), ("C", "Yes"), ("D", "Possible"))
        assert resolve_mc_letter("B.", options) == "No"
        assert resolve_mc_letter("d", options) == "Possible"
        assert resolve_mc_letter("Z", options) is None


class TestOrdinalBuckets:
    """An option's text read as an interval."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("3 days", (3.0, 3.0)),
            ("0 hours", (0.0, 0.0)),
            ("1-12 Hours", (1.0, 12.0)),
            ("13-24 hours", (13.0, 24.0)),
            ("24 hours or more", (24.0, float("inf"))),
            ("48 hours or more", (48.0, float("inf"))),
            ("None", (0.0, 0.0)),
            ("Tropical Storm Watch", None),
            ("N/A", None),
            ("", None),
        ],
    )
    def test_parse(self, text, expected):
        assert parse_ordinal(text) == expected

    def test_boundary_value_satisfies_two_buckets(self):
        # Q25's real options. 24 is in '12-24 Hours' and in '24-48 hours' — the ambiguity is
        # the dataset's, and the comparator accepts either while reporting it.
        options = (
            ("A", "0-12 hours"),
            ("B", "12-24 Hours"),
            ("C", "24-48 hours"),
            ("D", "48 hours or more"),
        )
        assert bucket_letters(24.0, options) == (["B", "C"], 4)
        assert bucket_letters(6.0, options) == (["A"], 4)
        assert bucket_letters(96.0, options) == (["D"], 4)

    def test_disjoint_buckets_are_unambiguous(self):
        options = (("A", "2 days"), ("B", "3 days"), ("C", "4 days"), ("D", "5 days"))
        assert bucket_letters(5.0, options) == (["D"], 4)

    def test_non_ordinal_options_reported(self):
        options = (("A", "Yes"), ("B", "No"))
        assert bucket_letters(3.0, options) == ([], 0)


class TestThresholdPredicate:
    """The comparison a yes/no question asks about, read from its own text."""

    def test_q26_question(self):
        question = (
            "Will the probability of 34-kt winds at Port of Savannah, GA reach or exceed 20% "
            "at any point during the forecast period? Answer ONLY 'Yes' or 'No'."
        )
        assert parse_threshold(question) == (">=", 20.0)

    def test_strict_comparison(self):
        assert parse_threshold("Will winds exceed 50 kt?") == (">", 50.0)

    def test_no_threshold(self):
        assert parse_threshold("Is the port inside the cone?") is None

    def test_evaluation(self):
        assert evaluate_predicate(">=", 20.0, 20.0) is True
        assert evaluate_predicate(">", 20.0, 20.0) is False
        assert evaluate_predicate(">=", 20.0, 11.0) is False


class TestNormalisation:
    """Folding the release's surface variation without erasing real differences."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Category 2", "Category 2"),
            ("Category 2 Hurricane", "Category 2"),
            ("Category 2 (105-110 mph, forecast to reach major-hurricane strength)", "Category 2"),
            ("Hurricane (Category 1 now - forecast to strengthen)", "Category 1"),
            ("Tropical Storm (below SSHS Category 1)", "Tropical Storm"),
            ("Potential Tropical Cyclone (not yet on SSHS scale)", "Potential Tropical Cyclone"),
            ("N/A", None),
            ("", None),
        ],
    )
    def test_ssh_category(self, raw, expected):
        assert normalise_ssh_category(raw) == expected

    def test_leading_clause_drops_qualifiers(self):
        assert leading_clause("Unlikely (forecast 1-3 ft)") == "unlikely"
        assert leading_clause("Possible but uncertain") == "possible but uncertain"

    def test_leading_clause_spares_hyphenated_terms(self):
        # 'X-RAY' is a port condition; truncating at the hyphen would yield 'x'.
        assert leading_clause("X-RAY") == "x-ray"

    def test_numeric_normalisation(self):
        assert normalise(2.0) == normalise(2) == "2"

    def test_port_condition_aliases(self):
        assert PORT_CONDITION_ALIASES["XRAY"] == PORT_CONDITION_ALIASES["X-RAY"] == "X-RAY"

    def test_placeholder_predicate(self):
        assert is_placeholder_answer(None)
        assert is_placeholder_answer("None")
        assert is_placeholder_answer("")
        # 'No' and 0 are real answers, not placeholders.
        assert not is_placeholder_answer("No")
        assert not is_placeholder_answer(0)
        assert not is_placeholder_answer("N/A")


# ======================================================================================
# Corpus tests
# ======================================================================================


@requires_data
class TestCorpusShape:
    """§4.1's stated numbers, checked against the release."""

    def test_question_count_matches_the_paper(self, corpus):
        assert len(corpus.qa) == 472
        assert sum(len(records) for records in corpus.qa.values()) == 117178

    def test_horizon_blocks(self, corpus):
        assert sum(len(scenario.horizons) for scenario in corpus.encoded.values()) == 2910

    def test_every_horizon_key_is_a_known_offset(self, corpus):
        seen = {hours for scenario in corpus.encoded.values() for hours in scenario.horizons}
        assert seen <= set(HORIZON_HOURS)
        assert seen == set(HORIZON_HOURS)

    def test_all_four_blocks_present(self, corpus):
        # guaranteed by the loader raising otherwise; assert the loader saw all 2,910
        for scenario in corpus.encoded.values():
            for block in scenario.time.values():
                assert set(block.blocks()) == {
                    "cyclone_condition_and_forecast",
                    "weather_observation",
                    "operation",
                    "impact_prediction",
                }

    def test_answer_format_counts(self, corpus):
        counts: dict[str, int] = {}
        for record in corpus.records():
            counts[record.answer_format.value] = counts.get(record.answer_format.value, 0) + 1
        assert counts == {"DE": 24094, "MC": 37813, "NU": 49459, "TF": 5812}

    def test_paper_stage_counts(self, corpus):
        counts: dict[str, int] = {}
        for record in corpus.records():
            counts[record.stage or ""] = counts.get(record.stage or "", 0) + 1
        assert counts["S1"] == 84244
        assert counts["S2"] == 31502
        assert counts["S3"] == 1432

    def test_release_is_smaller_than_the_paper_claims(self, corpus):
        # §4.1 says 145 ports and 90 storms; the release contains 45 and 69. Recorded in
        # KNOWN_DEFECTS['paper-counts-exceed-release'] so the project reports the real numbers.
        assert corpus.counts()["ports"] == 45
        assert corpus.counts()["storms"] == 69


@requires_data
class TestContextArity:
    """The positional context array, whose arity is not fixed."""

    def test_arity_split(self, corpus):
        arity_3 = sum(1 for record in corpus.records() if record.context.port is None)
        arity_4 = sum(1 for record in corpus.records() if record.context.port is not None)
        assert arity_3 == 31945
        assert arity_4 == 85233

    def test_only_the_port_independent_templates_use_arity_3(self, corpus):
        three: set[str] = set()
        four: set[str] = set()
        for record in corpus.records():
            (three if record.context.port is None else four).add(record.template_id)
        assert three == {"Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q17", "Q18", "Q20", "Q21", "Q23"}
        assert not (three & four)

    def test_horizon_is_always_valid_under_both_arities(self, corpus):
        for record in corpus.records():
            assert record.context.horizon_hours in HORIZON_HOURS


@requires_data
class TestGroundTruth:
    """Every template's declared answer path against the encoding."""

    def test_every_template_resolves_somewhere(self, corpus):
        """No declared path may be a dud — one that resolves for no record at all."""
        resolved: dict[str, int] = {template_id: 0 for template_id in corpus.templates}
        for record in corpus.records():
            template = corpus.templates[record.template_id]
            path = expected_path(record, template)
            value = resolve_path(corpus.encoded[record.scenario_id], record.context.horizon_hours, path)
            if value is not MISSING:
                resolved[record.template_id] += 1
        duds = [template_id for template_id, count in resolved.items() if count == 0]
        assert duds == [], f"templates whose declared path never resolves: {duds}"

    def test_qa_ids_are_unique(self, corpus):
        ids = [record.qa_id for record in corpus.records()]
        assert len(ids) == len(set(ids)) == 117178

    def test_group_id_is_the_scenario(self, corpus):
        for record in corpus.records():
            assert record.group_id == record.scenario_id

    def test_agreement_is_high_except_for_the_known_bad_templates(self, corpus):
        """The headline invariant: the release's answers match their own declared fields.

        Each template gets a floor rather than one global threshold, because the residuals are
        not uniform and pretending otherwise would hide the shape of the problem:

        * ``Q36`` is the broken one — its answers come from ``start_impact_date``, not the
          field its template names — and is expected to score near zero.
        * ``Q25`` and ``Q18`` carry the heaviest normalisation load (ordinal buckets whose
          boundaries overlap; 130+ spellings of an SSHS rung) and sit in the low nineties.
        * ``Q27`` and ``Q31`` sit just under 0.99 from residual label noise, where the option
          set has no entry for the field's value and the generator picked the nearest.
        * everything else must be at 0.99 or better, or the loader or comparator has drifted.
        """
        from collections import Counter

        floors = {"Q36": 0.0, "Q25": 0.94, "Q18": 0.97, "Q27": 0.985, "Q31": 0.985}
        default_floor = 0.99

        verdicts: dict[str, Counter] = {}
        for record in corpus.records():
            template = corpus.templates[record.template_id]
            path = expected_path(record, template)
            value = resolve_path(corpus.encoded[record.scenario_id], record.context.horizon_hours, path)
            verdict = compare(record, value, path, semantics=template.semantics)
            verdicts.setdefault(record.template_id, Counter())[verdict.status.value] += 1

        offenders: dict[str, float] = {}
        for template_id, counter in verdicts.items():
            checked = counter[VerdictStatus.MATCH.value] + counter[VerdictStatus.MISMATCH.value]
            if not checked:
                continue
            agreement = counter[VerdictStatus.MATCH.value] / checked
            floor = floors.get(template_id, default_floor)
            if agreement < floor:
                offenders[template_id] = round(agreement, 4)
        assert offenders == {}, f"templates below their agreement floor: {offenders}"

        # Q36 is not merely imperfect, it is inverted: its answers are taken from a different
        # field. Assert that direction explicitly, so that if upstream ever fixes the template
        # this test fails loudly and the tolerance table gets revisited rather than silently
        # accepting either behaviour.
        #
        # 2,910 records split three ways: 1,334 copy the encoding's own "None" placeholder and so
        # are unscorable, 1,571 disagree with the declared field, and 5 match by coincidence.
        q36 = verdicts["Q36"]
        assert q36[VerdictStatus.UNANSWERABLE.value] == 1334
        q36_checked = q36[VerdictStatus.MATCH.value] + q36[VerdictStatus.MISMATCH.value]
        assert q36_checked == 1576
        assert q36[VerdictStatus.MATCH.value] / q36_checked < 0.05

    def test_multiple_choice_options_come_from_the_choice_array(self, corpus):
        for record in corpus.records():
            if record.answer_format is not AnswerFormat.MULTIPLE_CHOICE:
                continue
            declared = record.extra.get("choice")
            assert declared, "every MC record carries a choice array"
            assert tuple(text for _, text in record.options) == tuple(declared)
        # and no other format carries one
        for record in corpus.records():
            if record.answer_format is not AnswerFormat.MULTIPLE_CHOICE:
                assert not record.extra.get("choice")

    def test_bucket_templates_are_the_declared_ones(self, corpus):
        from okh_rag.data.cyportqa import TemplateSemantics

        by_semantics: dict[str, set[str]] = {}
        for template in corpus.templates.values():
            by_semantics.setdefault(template.semantics.value, set()).add(template.template_id)
        assert by_semantics["ordinal_bucket"] == {"Q23", "Q24", "Q25", "Q27", "Q40"}
        assert by_semantics["threshold_predicate"] == {"Q26"}


@requires_data
class TestKnownDefects:
    """Each defect is asserted here so a change upstream is caught, not discovered."""

    def test_placeholder_answers(self, corpus):
        unanswerable = sum(1 for record in corpus.records() if not record.is_answerable)
        assert unanswerable == 8337

    def test_q36_answers_from_the_wrong_field(self, corpus):
        checked = 0
        from_start = 0
        for record in corpus.records():
            if record.template_id != "Q36":
                continue
            scenario = corpus.encoded[record.scenario_id]
            start = resolve_path(scenario, record.context.horizon_hours, "impact_prediction.start_impact_date")
            checked += 1
            if record.answer_text == str(start):
                from_start += 1
        assert checked == 2910
        assert from_start == checked, (
            "Q36's shipped answers no longer equal start_impact_date; upstream may have fixed the template"
        )

    def test_q33_answer_is_a_python_repr(self, corpus):
        import json

        for record in corpus.records():
            if record.template_id != "Q33":
                continue
            with pytest.raises(json.JSONDecodeError):
                json.loads(record.answer_text)
            parsed = record.structured_answer()
            assert isinstance(parsed, dict)
            assert "in 12h" in parsed
            return
        pytest.fail("no Q33 record found")

    def test_imbalanced_fragments_are_corrupt(self):
        part1 = (DATA_ROOT / "source_data" / "CyPortQA_Imbalanced" / "CyPortQA_Imbalanced_part1.json").read_bytes()
        part2 = (DATA_ROOT / "source_data" / "CyPortQA_Imbalanced" / "CyPortQA_Imbalanced_part2.json").read_bytes()
        blob = (part1 + part2).decode("utf-8-sig")

        import json

        with pytest.raises(json.JSONDecodeError):
            json.loads(blob)

        depth = 0
        in_string = False
        escaped = False
        for char in blob:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char in "{[":
                depth += 1
            elif char in "}]":
                depth -= 1
        assert depth != 0, "the concatenation now returns to brace depth 0; upstream may have fixed the split"

        recovered, note = load_imbalanced(DATA_ROOT)
        assert len(recovered) > 0, note
        assert "unrecoverable" in note

    def test_description_answers_are_verbatim_copies(self, corpus):
        """All 24,094 DE answers equal their source field byte for byte.

        Worth pinning rather than assuming: it is what makes §4.1's GPT-4o judge unnecessary for
        the TD metric, and it is the kind of property that silently stops holding.
        """
        total = 0
        verbatim = 0
        for record in corpus.records():
            if record.answer_format is not AnswerFormat.DESCRIPTION:
                continue
            template = corpus.templates[record.template_id]
            path = expected_path(record, template)
            expected = resolve_path(corpus.encoded[record.scenario_id], record.context.horizon_hours, path)
            if expected is MISSING:
                continue
            total += 1
            verbatim += int(record.answer_text == str(expected))
        assert total == 24094
        assert verbatim == total

    def test_every_declared_defect_is_measurable(self):
        ids = [defect.defect_id for defect in KNOWN_DEFECTS]
        assert len(ids) == len(set(ids))
        assert len(ids) == 10
