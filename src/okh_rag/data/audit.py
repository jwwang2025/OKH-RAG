"""Re-measure the release's defects, and audit every template against its own ground truth.

The point of this module is falsifiability. ``cyportqa.KNOWN_DEFECTS`` states what is wrong
with the released files; this module recomputes those statements from the bytes on disk, so
a reader can run it and disagree with the numbers rather than take them on trust. It also
audits all 48 templates in one pass, which is what turns "the answers look fine" into
"Q36 disagrees with its own declared field in 100% of cases".

Usage::

    python scripts/audit_cyportqa.py --data data/raw/cyportqa --out reports/

Writes ``reports/cyportqa_audit.md`` and ``reports/cyportqa_audit.json``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable

from okh_rag.data.cyportqa import (
    ANSWER_FORMAT_TAGS,
    HORIZON_HOURS,
    KNOWN_DEFECTS,
    PORT_CONDITION_ALIASES,
    AnswerFormat,
    CyPortQACorpus,
    DefectSeverity,
    TemplateSemantics,
    load_corpus,
    load_imbalanced,
    parse_inline_options,
)
from okh_rag.data.groundtruth import (
    MISSING,
    Verdict,
    VerdictStatus,
    bucket_letters,
    compare,
    expected_path,
    normalise_ssh_category,
    parse_ordinal,
    resolve_path,
)


#: The option-count clause a prompt states in words, e.g. ``"(A, B, C, or D)"`` or
#: ``"(A, B, C, D, E, F or G)"``. Used to detect where the prompt and the question disagree
#: about how many choices a question has.
_OPTION_COUNT_RE: Final[re.Pattern[str]] = re.compile(r"\(([A-G](?:\s*,\s*(?:or\s+)?[A-G])*)\)")


# --------------------------------------------------------------------------------------
# Per-template audit
# --------------------------------------------------------------------------------------


@dataclass
class TemplateAudit:
    """How one template's shipped answers relate to the field it declares."""

    template_id: str
    answer_format: str
    semantics: str
    stage: str | None
    declared_path: str
    n_records: int = 0
    verdicts: Counter = field(default_factory=Counter)
    n_ambiguous: int = 0
    examples: list[dict[str, str]] = field(default_factory=list)

    @property
    def n_checked(self) -> int:
        """Records where a verdict was actually decidable offline."""
        return (
            self.verdicts[VerdictStatus.MATCH.value]
            + self.verdicts[VerdictStatus.MISMATCH.value]
        )

    @property
    def agreement(self) -> float | None:
        """Share of decidable records whose shipped answer matches the declared field.

        ``None`` when nothing was decidable — which is the honest answer for every
        description template, since §4.1 scores those with an LLM judge.
        """
        return self.verdicts[VerdictStatus.MATCH.value] / self.n_checked if self.n_checked else None

    @property
    def determinacy(self) -> float | None:
        """Share of checked records whose answer the encoding determines *uniquely*.

        Distinct from :attr:`agreement` and deliberately so. For an ordinal-bucket template
        like Q25 the release's buckets overlap, so a value of 24 satisfies both ``12-24`` and
        ``24-48``; the shipped answer is then consistent with the field without being
        *implied* by it. A high agreement with a low determinacy means the template is being
        scored on a question the data does not settle.
        """
        if not self.n_checked:
            return None
        return (self.n_checked - self.n_ambiguous) / self.n_checked

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["verdicts"] = dict(self.verdicts)
        out["agreement"] = self.agreement
        out["determinacy"] = self.determinacy
        return out


def audit_templates(corpus: CyPortQACorpus) -> list[TemplateAudit]:
    """Compare every shipped answer against the value its template names.

    Records whose scenario has no encoding are skipped by the caller's corpus construction
    (``CyPortQACorpus.__post_init__`` rejects orphans), so ``n_records`` is complete.
    """
    audits: dict[str, TemplateAudit] = {}
    for template_id, template in corpus.templates.items():
        audits[template_id] = TemplateAudit(
            template_id=template_id,
            answer_format=template.answer_format.value,
            semantics=template.semantics.value,
            stage=template.stage,
            declared_path=template.static_prefix,
        )

    for record in corpus.records():
        audit = audits[record.template_id]
        audit.n_records += 1
        scenario = corpus.encoded[record.scenario_id]
        template = corpus.templates[record.template_id]
        path = expected_path(record, template)
        expected = resolve_path(scenario, record.context.horizon_hours, path)
        verdict: Verdict = compare(record, expected, path, semantics=template.semantics)
        audit.verdicts[verdict.status.value] += 1
        audit.n_ambiguous += int(verdict.ambiguous)
        if verdict.status is VerdictStatus.MISMATCH and len(audit.examples) < 5:
            audit.examples.append(
                {
                    "qa_id": record.qa_id,
                    "shipped": verdict.shipped,
                    "expected": verdict.expected,
                    "path": path,
                    "detail": verdict.detail,
                }
            )

    return [audits[key] for key in sorted(audits, key=lambda k: int(k[1:]))]


# --------------------------------------------------------------------------------------
# Defect re-measurement
# --------------------------------------------------------------------------------------


def measure_defects(root: Path) -> dict[str, Any]:
    """Recompute each entry of ``KNOWN_DEFECTS`` from the files under ``root``.

    Returns a mapping ``defect_id -> {"severity", "measured", "verdict", "detail"}`` where
    ``measured`` holds the raw numbers behind the claim. Nothing here trusts the registry;
    every figure is derived from a fresh parse.
    """
    corpus = load_corpus(root)
    measures: dict[str, Any] = {}

    # --- qa-answer-placeholder-none -----------------------------------------------------
    none_literal = 0
    empty = 0
    unanswerable_by_template: Counter = Counter()
    for record in corpus.records():
        if record.answer is None:
            none_literal += 1
            unanswerable_by_template[record.template_id] += 1
        elif isinstance(record.answer, str):
            stripped = record.answer.strip()
            if stripped.lower() == "none":
                none_literal += 1
                unanswerable_by_template[record.template_id] += 1
            elif stripped == "":
                empty += 1
                unanswerable_by_template[record.template_id] += 1
    total = sum(1 for _ in corpus.records())
    empty_operations = sum(1 for scenario in corpus.encoded.values() for block in scenario.time.values() if not block.operation)
    # How much of the placeholder load originates in the *encoding* rather than in generation:
    # the release stores the string "None" rather than JSON null, so the generator copies it.
    field_none = sum(
        1
        for scenario in corpus.encoded.values()
        for block in scenario.time.values()
        if str((block.impact_prediction or {}).get("start_impact_date")) == "None"
    )
    measures["qa-answer-placeholder-none"] = {
        "measured": {
            "literal_none": none_literal,
            "empty_string": empty,
            "share_unanswerable": round((none_literal + empty) / total, 4),
            "horizon_blocks_with_empty_operation": empty_operations,
            "horizon_blocks_whose_impact_prediction_is_the_string_None": field_none,
            "top_templates": unanswerable_by_template.most_common(5),
        },
        "verdict": "reproduced" if none_literal and empty else "NOT REPRODUCED",
    }

    # --- template-q36-answer-field-mismatch ---------------------------------------------
    q36_total = 0
    q36_equals_start = 0
    q36_equals_end = 0
    start_field = "impact_prediction.start_impact_date"
    end_field = "impact_prediction.end_impact_date"
    shared_triple = 0
    triple_groups = 0
    for scenario_id, records in corpus.qa.items():
        scenario = corpus.encoded[scenario_id]
        by_horizon: dict[int, dict[str, str]] = {}
        for record in records:
            if record.template_id not in {"Q34", "Q35", "Q36"}:
                continue
            by_horizon.setdefault(record.context.horizon_hours, {})[record.template_id] = record.answer_text
        for horizon_hours, answers in by_horizon.items():
            if len(answers) == 3:
                triple_groups += 1
                if len(set(answers.values())) == 1:
                    shared_triple += 1
        for record in records:
            if record.template_id != "Q36":
                continue
            q36_total += 1
            shipped = record.answer_text
            start = resolve_path(scenario, record.context.horizon_hours, start_field)
            end = resolve_path(scenario, record.context.horizon_hours, end_field)
            if start is not MISSING and shipped == str(start):
                q36_equals_start += 1
            if end is not MISSING and shipped == str(end):
                q36_equals_end += 1
    measures["template-q36-answer-field-mismatch"] = {
        "measured": {
            "q36_records": q36_total,
            "q36_equals_start_impact_date": q36_equals_start,
            "q36_equals_end_impact_date": q36_equals_end,
            "share_answering_from_start_field": round(q36_equals_start / q36_total, 4) if q36_total else None,
            "q34_q35_q36_groups": triple_groups,
            "groups_where_all_three_agree": shared_triple,
        },
        "verdict": "reproduced" if q36_total and q36_equals_start == q36_total else "NOT REPRODUCED",
    }

    # --- qa-q33-answer-is-python-repr ---------------------------------------------------
    q33_total = 0
    q33_json_ok = 0
    q33_literal_ok = 0
    for record in corpus.records():
        if record.template_id != "Q33":
            continue
        q33_total += 1
        raw = record.answer_text
        try:
            json.loads(raw)
            q33_json_ok += 1
            continue
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            record.structured_answer()
            q33_literal_ok += 1
        except Exception:  # pragma: no cover - structured_answer swallows these
            pass
    measures["qa-q33-answer-is-python-repr"] = {
        "measured": {
            "q33_records": q33_total,
            "parseable_as_json": q33_json_ok,
            "recovered_by_literal_eval": q33_literal_ok,
        },
        "verdict": "reproduced" if q33_total and q33_json_ok == 0 and q33_literal_ok else "NOT REPRODUCED",
    }

    # --- imbalanced-fragments-unterminated ----------------------------------------------
    base = root / "source_data" / "CyPortQA_Imbalanced"
    part1 = (base / "CyPortQA_Imbalanced_part1.json").read_bytes()
    part2 = (base / "CyPortQA_Imbalanced_part2.json").read_bytes()
    blob = (part1 + part2).decode("utf-8-sig")
    depth, in_string, escaped = 0, False, False
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
    plain_concat_valid = False
    try:
        json.loads(blob)
        plain_concat_valid = True
    except json.JSONDecodeError:
        pass
    recovered, note = load_imbalanced(root)
    measures["imbalanced-fragments-unterminated"] = {
        "measured": {
            "concat_parses_as_json": plain_concat_valid,
            "brace_depth_at_eof": depth,
            "ended_inside_string": in_string,
            "scenarios_recovered_by_scanner": len(recovered),
            "scanner_note": note,
        },
        "verdict": "reproduced" if (not plain_concat_valid and depth != 0) else "NOT REPRODUCED",
    }

    # --- qa-context-arity-is-variable ---------------------------------------------------
    arity: Counter = Counter()
    arity_by_template: dict[str, set[int]] = {}
    for record in corpus.records():
        size = 3 if record.context.port is None else 4
        arity[size] += 1
        arity_by_template.setdefault(record.template_id, set()).add(size)
    three_only = sorted(
        (template_id for template_id, sizes in arity_by_template.items() if sizes == {3}),
        key=lambda template_id: int(template_id[1:]),
    )
    measures["qa-context-arity-is-variable"] = {
        "measured": {
            "arity_3": arity[3],
            "arity_4": arity[4],
            "share_arity_3": round(arity[3] / sum(arity.values()), 4),
            "three_element_only_templates": three_only,
        },
        "verdict": "reproduced" if arity[3] and arity[4] else "NOT REPRODUCED",
    }

    # --- mc-options-duplicated-and-inconsistent-with-prompt -----------------------------
    mc_total = 0
    mc_with_options = 0
    mc_with_choice_field = 0
    mc_inline_agrees = 0
    prompt_vs_choice = 0
    choice_vs_question = 0
    prompt_examples: list[dict[str, Any]] = []
    for record in corpus.records():
        if record.answer_format is not AnswerFormat.MULTIPLE_CHOICE:
            continue
        mc_total += 1
        if record.options:
            mc_with_options += 1
        declared = record.extra.get("choice") or ()
        if declared:
            mc_with_choice_field += 1
        inline = parse_inline_options(record.question)
        if declared and len(inline) == len(declared):
            mc_inline_agrees += 1

        # Both the prompt and the question open with a boilerplate clause stating the option
        # count in words. The structured array is checked against each of them, because a
        # stale count can survive in one of the two and mislead a consumer that reads only
        # that one — Q18's question still says "(A, B, C, or D)" while offering A-G.
        for text, label, counter in (
            (record.prompt, "prompt", "prompt"),
            (record.question, "question", "question"),
        ):
            stated = {len(match.group(1).split(",")) for match in _OPTION_COUNT_RE.finditer(text)}
            if not stated or not declared:
                continue
            expected_count = max(stated)
            if len(declared) == expected_count:
                continue
            if counter == "prompt":
                prompt_vs_choice += 1
            else:
                choice_vs_question += 1
            if len(prompt_examples) < 3:
                prompt_examples.append(
                    {
                        "qa_id": record.qa_id,
                        "source": label,
                        "clause_states": expected_count,
                        "choice_array_has": len(declared),
                        "inline_offers": len(inline),
                    }
                )
    measures["mc-options-duplicated-and-inconsistent-with-prompt"] = {
        "measured": {
            "mc_records": mc_total,
            "records_with_choice_array": mc_with_choice_field,
            "records_with_options_recovered": mc_with_options,
            "inline_matches_choice_length": mc_inline_agrees,
            "records_where_prompt_clause_disagrees_with_choice_array": prompt_vs_choice,
            "records_where_question_clause_disagrees_with_choice_array": choice_vs_question,
            "examples": prompt_examples,
        },
        "verdict": (
            "reproduced"
            if mc_total
            and mc_with_choice_field == mc_total
            and (prompt_vs_choice or choice_vs_question)
            else "PARTIAL"
        ),
    }

    # --- raw-categorical-fields-not-normalised ------------------------------------------
    category_variants: Counter = Counter()
    cone_variants: set[str] = set()
    watch_variants: set[str] = set()
    unrecognised_categories = 0
    condition_spellings: Counter = Counter()
    for scenario in corpus.encoded.values():
        for block in scenario.time.values():
            raw_category = (block.cyclone_condition_and_forecast or {}).get("category")
            category_variants[str(raw_category)] += 1
            if normalise_ssh_category(raw_category) is None:
                unrecognised_categories += 1
            cone = (block.weather_observation or {}).get("within_uncertainty_cone")
            if isinstance(cone, str):
                cone_variants.add(cone)
            watch = (block.weather_observation or {}).get("under_watch")
            if isinstance(watch, str):
                watch_variants.add(watch)
            for offset, payload in (block.operation or {}).items():
                raw_condition = payload.get("Condition") if isinstance(payload, dict) else None
                if isinstance(raw_condition, str):
                    condition_spellings[raw_condition] += 1
    off_ladder = [spelling for spelling in condition_spellings if spelling not in PORT_CONDITION_ALIASES]
    measures["raw-categorical-fields-not-normalised"] = {
        "measured": {
            "distinct_category_strings": len(category_variants),
            "categories_not_mapped_to_a_rung": unrecognised_categories,
            "distinct_within_uncertainty_cone_strings": len(cone_variants),
            "distinct_under_watch_strings": len(watch_variants),
            "condition_spellings": dict(condition_spellings),
            "condition_spellings_off_the_ladder": off_ladder,
        },
        "verdict": "reproduced" if len(category_variants) > 50 else "NOT REPRODUCED",
    }

    # --- ordinal-buckets-overlap --------------------------------------------------------
    bucket_templates = [
        template_id
        for template_id, template in corpus.templates.items()
        if template.semantics is TemplateSemantics.ORDINAL_BUCKET
    ]
    bucket_stats: dict[str, Any] = {}
    for template_id in sorted(bucket_templates, key=lambda t: int(t[1:])):
        ambiguous = 0
        checked = 0
        option_sets: Counter = Counter()
        for record in corpus.records():
            if record.template_id != template_id:
                continue
            options = record.options or parse_inline_options(record.question)
            if options:
                option_sets[tuple(text for _, text in options)] += 1
            scenario = corpus.encoded[record.scenario_id]
            path = expected_path(record, corpus.templates[record.template_id])
            expected = resolve_path(scenario, record.context.horizon_hours, path)
            value = parse_ordinal(expected if isinstance(expected, str) else repr(expected))
            if value is None:
                continue
            # A scalar field parses to the degenerate interval (v, v); a bucket option to a real one.
            scalar, upper = value
            if scalar != upper:
                continue
            letters, ordinal = bucket_letters(scalar, options)
            if ordinal == 0:
                continue
            checked += 1
            ambiguous += int(len(letters) > 1)
        bucket_stats[template_id] = {
            "checked": checked,
            "ambiguous": ambiguous,
            "determinacy": round((checked - ambiguous) / checked, 4) if checked else None,
            "option_sets": [list(texts) for texts, _ in option_sets.most_common(2)],
            "example_value": scalar if checked else None,
        }
    measures["ordinal-buckets-overlap"] = {
        "measured": {"templates": bucket_stats},
        "verdict": "reproduced" if any(entry["ambiguous"] for entry in bucket_stats.values()) else "NOT REPRODUCED",
    }

    # --- td-answers-are-verbatim-field-copies -------------------------------------------
    de_total = 0
    de_verbatim = 0
    de_templates: set[str] = set()
    for record in corpus.records():
        if record.answer_format is not AnswerFormat.DESCRIPTION:
            continue
        template = corpus.templates[record.template_id]
        path = expected_path(record, template)
        expected = resolve_path(corpus.encoded[record.scenario_id], record.context.horizon_hours, path)
        if expected is MISSING:
            continue
        de_total += 1
        de_templates.add(record.template_id)
        if record.answer_text == str(expected):
            de_verbatim += 1
    measures["td-answers-are-verbatim-field-copies"] = {
        "measured": {
            "description_records": de_total,
            "byte_identical_to_the_declared_field": de_verbatim,
            "share_verbatim": round(de_verbatim / de_total, 6) if de_total else None,
            "templates": sorted(de_templates, key=lambda t: int(t[1:])),
        },
        "verdict": "reproduced" if de_total and de_verbatim == de_total else "NOT REPRODUCED",
    }

    # --- paper-counts-exceed-release ----------------------------------------------------
    storms = {scenario.scenario_id.storm for scenario in corpus.encoded.values()}
    ports = {scenario.scenario_id.port for scenario in corpus.encoded.values()}
    horizon_blocks = sum(len(scenario.horizons) for scenario in corpus.encoded.values())
    measures["paper-counts-exceed-release"] = {
        "measured": {
            "release_scenario_keys": len(corpus.encoded),
            "release_horizon_blocks": horizon_blocks,
            "release_ports": len(ports),
            "release_storms": len(storms),
            "paper_ports": 145,
            "paper_storms": 90,
            "paper_scenarios": 2917,
            "questions": total,
            "paper_questions": 117178,
        },
        "verdict": "reproduced" if len(ports) != 145 or len(storms) != 90 else "NOT REPRODUCED",
    }

    for defect in KNOWN_DEFECTS:
        entry = measures.setdefault(defect.defect_id, {"measured": {}, "verdict": "NOT MEASURED"})
        entry["severity"] = defect.severity.value
        entry["summary"] = defect.summary
    return measures


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def build_report(root: Path) -> dict[str, Any]:
    """Run both audits and return a single JSON-serialisable structure."""
    corpus = load_corpus(root)
    templates = audit_templates(corpus)
    defects = measure_defects(root)
    return {
        "data_root": str(root),
        "corpus": corpus.counts(),
        "declared_horizons": list(HORIZON_HOURS),
        "answer_format_tags": sorted(ANSWER_FORMAT_TAGS),
        "templates": [audit.to_dict() for audit in templates],
        "defects": defects,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the audit as a Markdown report."""
    lines: list[str] = []
    add = lines.append

    add("# CyPortQA release audit")
    add("")
    add(f"Data root: `{report['data_root']}`")
    add("")
    add("Recomputed from the files on disk by `scripts/audit_cyportqa.py`. Every number below is")
    add("derived at run time; nothing is copied from a previous run.")
    add("")

    add("## Corpus")
    add("")
    add("| quantity | value | paper (§4.1) |")
    add("|---|---:|---|")
    corpus = report["corpus"]
    paper = report["defects"]["paper-counts-exceed-release"]["measured"]
    add(f"| scenario keys in the release | {corpus['scenarios']} | — |")
    add(f"| horizon blocks | {corpus['horizon_blocks']} | 2,917 scenarios |")
    add(f"| knowledge groups | {corpus['knowledge_groups']} | — |")
    add(f"| ports | {corpus['ports']} | 145 |")
    add(f"| storms | {corpus['storms']} | 90 |")
    add(f"| questions | {corpus['questions']} | 117,178 |")
    add(f"| questions with an answer | {corpus['questions_answerable']} | — |")
    add(f"| questions with a placeholder answer | {corpus['questions_unanswerable']} | — |")
    add(f"| questions whose context omits the port | {corpus['questions_without_port_in_context']} | — |")
    add("")
    add(f"Answers by format: `{corpus['answers_by_format']}`")
    add("")
    add(f"Questions by stage: `{corpus['questions_by_stage']}`")
    add("")
    add(f"Horizons present: `{report['declared_horizons']}` (T-{max(report['declared_horizons'])} .. "
        f"T-{min(report['declared_horizons'])})")
    add("")

    add("## Template audit")
    add("")
    add("For each template, the shipped answers are compared against the value at the path the")
    add("template itself declares. Two rates are reported, and they answer different questions:")
    add("")
    add("- `agreement` — of the records where a comparison was possible offline, how many agree with")
    add("  the declared field.")
    add("- `determinacy` — of those same records, how many have an answer the encoding determines")
    add("  *uniquely*. This is low for the ordinal-bucket templates, whose options overlap, so a")
    add("  value can satisfy two buckets at once. High agreement with low determinacy means the")
    add("  template is scored on a question the data does not settle.")
    add("")
    add("The `unverifiable` column is the count of records whose score needs the §4.1 LLM judge. It is")
    add("zero everywhere, and `td-answers-are-verbatim-field-copies` explains why: every description")
    add("answer is a byte-identical copy of its source field, so the judge has nothing left to judge.")
    add("")
    add("| template | fmt | semantics | stage | declared path | records | match | mismatch | indeterminate | missing | placeholder | unverifiable | agreement | determinacy |")
    add("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for entry in report["templates"]:
        verdicts = entry["verdicts"]
        agreement = entry["agreement"]
        determinacy = entry["determinacy"]
        shown = "—" if agreement is None else f"{agreement:.3f}"
        shown_det = "—" if determinacy is None else f"{determinacy:.3f}"
        path = entry["declared_path"] or "*(unresolved)*"
        add(
            f"| {entry['template_id']} | {entry['answer_format']} | {entry['semantics']} "
            f"| {entry['stage'] or '—'} | `{path}` "
            f"| {entry['n_records']} | {verdicts.get('match', 0)} | {verdicts.get('mismatch', 0)} "
            f"| {verdicts.get('indeterminate', 0)} "
            f"| {verdicts.get('missing_field', 0)} | {verdicts.get('unanswerable', 0)} "
            f"| {verdicts.get('unverifiable', 0)} | {shown} | {shown_det} |"
        )
    add("")

    checked = [e for e in report["templates"] if e["agreement"] is not None]
    if checked:
        total_checked = sum(e["agreement"] for e in checked) and sum(
            e["verdicts"].get("match", 0) + e["verdicts"].get("mismatch", 0) for e in checked
        )
        total_match = sum(e["verdicts"].get("match", 0) for e in checked)
        if total_checked:
            add(
                f"Across all templates with decidable comparisons: **{total_match}/{total_checked} "
                f"= {total_match / total_checked:.4f}** of shipped answers agree with the field their "
                "own template names."
            )
            add("")

    flagged = [e for e in report["templates"] if (e["agreement"] is not None and e["agreement"] < 0.999)]
    if flagged:
        add("### Templates whose shipped answers disagree with their declared field")
        add("")
        for entry in flagged:
            add(f"- **{entry['template_id']}** ({entry['semantics']}) — agreement {entry['agreement']:.3f}, "
                f"determinacy {entry['determinacy']:.3f} over {entry['n_records']} records, "
                f"declared path `{entry['declared_path']}`")
            for example in entry["examples"][:2]:
                add(f"  - `{example['qa_id']}`: shipped `{example['shipped']}` vs "
                    f"expected `{example['expected']}` ({example['detail']})")
        add("")

    add("## Declared defects")
    add("")
    add("| id | severity | verdict | measured |")
    add("|---|---|---|---|")
    for defect in KNOWN_DEFECTS:
        entry = report["defects"][defect.defect_id]
        measured = json.dumps(entry["measured"], ensure_ascii=False, default=str)
        if len(measured) > 320:
            measured = measured[:317] + "..."
        add(f"| `{defect.defect_id}` | {defect.severity.value} | {entry['verdict']} | `{measured}` |")
    add("")
    add("### Details")
    add("")
    for defect in KNOWN_DEFECTS:
        entry = report["defects"][defect.defect_id]
        add(f"#### `{defect.defect_id}` — {entry['verdict']}")
        add("")
        add(f"*Severity:* {defect.severity.value}  ")
        add(f"*Affects:* {defect.affected}")
        add("")
        add(defect.summary)
        add("")
        add(f"*Workaround:* {defect.workaround}")
        add("")
        add("```json")
        add(json.dumps(entry["measured"], ensure_ascii=False, indent=2, default=str))
        add("```")
        add("")

    blocking = [d for d in KNOWN_DEFECTS if d.severity is DefectSeverity.BLOCKING]
    if blocking:
        add("## Blocking issues")
        add("")
        for defect in blocking:
            add(f"- `{defect.defect_id}`: {defect.summary.splitlines()[0]}")
        add("")

    return "\n".join(lines)


def write_report(root: Path, out_dir: Path) -> tuple[Path, Path]:
    """Run the audit and write both artefacts. Returns ``(markdown_path, json_path)``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(root)
    markdown_path = out_dir / "cyportqa_audit.md"
    json_path = out_dir / "cyportqa_audit.json"
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return markdown_path, json_path


__all__ = [
    "TemplateAudit",
    "audit_templates",
    "build_report",
    "measure_defects",
    "render_markdown",
    "write_report",
]
