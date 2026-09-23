"""§4.1 — the CyPortQA benchmark as released by its authors.

Upstream: https://github.com/ChenchenMobility/MLLM-Bench-CyPortQA (MIT License)
Benchmark paper: Kuai et al., "CyPortQA: Benchmarking Multimodal Large Language Models
                 for Cyclone Preparedness in Port Operation", arXiv:2508.15846

This is the benchmark §4.1 describes, not a lookalike:

    "We evaluate on CyPortQA (Kuai et al., 2026b), a domain-specific QA benchmark for
     tropical cyclone-port impact assessment. CyPortQA contains 2,917 real-world
     disruption scenarios from 2015 to 2023, spanning 145 U.S. principal ports and 90
     named storms. Each scenario includes multi-horizon descriptions from T-120 to T-12
     covering storm evolution, hazard forecasting, operational response, and impact
     prediction. The benchmark contains 117,178 questions across four types [...]"

Five of the nine OKH-RAG authors (Chenchen Kuai, Zihao Li, Zhengzhong Tu, Yang Zhou and
Keshu Wu) also author CyPortQA, so there is no provenance question to resolve.

What this module provides, and why each part matters to the reproduction
-----------------------------------------------------------------------
``Encoded_senario.json`` is the single most valuable file in the release. It is a
per-scenario *structured* knowledge encoding: one block per lead-time horizon holding
``cyclone_condition_and_forecast``, ``weather_observation``, ``operation`` and
``impact_prediction``. That is exactly the raw material §A.2 extracts a hypergraph
from, and it means the extraction step has a deterministic ground truth to be scored
against instead of being taken on faith:

* entity *categories* (§A.1) fall out of the encoding — ports and storms are persistent
  objects, ``category``/``under_watch``/``Condition`` are transient states, and each
  ``time`` key is a temporal anchor;
* the ten ``time`` keys (``12``..``120``) are the T-120..T-12 horizons of §A.3, in a
  fixed 12-hour cadence, so §A.3's "decreasing horizon order" rule has real data;
* ``CyPortQA_template.json`` names, for every one of the 48 question templates, the
  *exact dotted field path* that produces its ground-truth answer. Retrieval can
  therefore be scored against a closed-form oracle: the hyperedges a question needs
  are precisely the hyperedges built from the fields its template names.

Layout of the release, and which parts this module reads
--------------------------------------------------------
===============================================  =======  =============================
path                                             size     read here
===============================================  =======  =============================
``dataset/CyPortQA.json``                        101 MB   :func:`load_qa`
``source_data/Encoded_senario.json``             9.6 MB   :func:`load_encoded`
``source_data/CyPortQA_template.json``            40 KB   :func:`load_templates`
``source_data/CyPortQA_Imbalanced/*.json``        122 MB   :func:`load_imbalanced`
``dataset/MultiModalInput/**`` and
``source_data/NOAA NHC Cyclone Products/**``     ~1.4 GB  not read — see below
===============================================  =======  =============================

The PNGs are the multimodal half of CyPortQA (uncertainty cones, wind-probability
graphics). OKH-RAG is text-only: §4.1 gives it the same GPT-4o generator and the same
``text-embedding-3-small`` embedder as every baseline, and neither is a vision model.
They are therefore not part of this reproduction, which is why the loader never touches
them — see ``data/CONTRACT.md``.

Known defects in the release
----------------------------
The released files are not internally consistent. Rather than work around that silently,
every defect is enumerated in :data:`KNOWN_DEFECTS` with its measured size, and
:func:`audit_defects` re-measures them from the files on disk so a reader can confirm
each claim instead of trusting this docstring. Two of them are load-bearing for the
reproduction: ``Q36``'s shipped answers disagree with the field its template names, and
the ``CyPortQA_Imbalanced`` fragments cannot be parsed as released.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

# --------------------------------------------------------------------------------------
# Constants derived from the release
# --------------------------------------------------------------------------------------

SCENARIO_ID_RE: Final[re.Pattern[str]] = re.compile(r"^(?P<storm>[A-Z0-9]+)_(?P<year>\d{4})_(?P<port>.+)$")
"""Scenario keys look like ``ANA_2015_Port_of_Savannah,_GA`` — storm, year, then the rest
is the port. Three underscores would be ambiguous, so the port is everything after the
second one (ports legitimately contain ``_`` and ``,``)."""

HORIZON_HOURS: Final[tuple[int, ...]] = (12, 24, 36, 48, 60, 72, 84, 96, 108, 120)
"""§4.1 — "multi-horizon descriptions from T-120 to T-12". The release keys the horizon
blocks by lead time in hours as *strings*, and ships exactly these ten values."""

TIME_BLOCK_FIELDS: Final[tuple[str, ...]] = (
    "cyclone_condition_and_forecast",
    "weather_observation",
    "operation",
    "impact_prediction",
)
"""§4.1 — "storm evolution, hazard forecasting, operational response, and impact
prediction". Every one of the 2,910 horizon blocks carries all four keys, so this is a
hard schema invariant and :func:`load_encoded` asserts it."""

RELATIVE_HORIZON_RE: Final[re.Pattern[str]] = re.compile(r"^in_(\d{2,3})h$")
"""``operation`` is keyed by a *relative* forecast offset (``now``, ``in_12h`` …,
``in_108h``) rather than an absolute lead time. Templates Q42 and Q46-Q48 carry this key
as a question-type tag, which is how the answer path becomes resolvable."""

#: §4.1 — "four types: True/False (TF), Multiple Choice (MC), Short Answer / Numeric
#: (SA), and Text Description (TD)". The paper's four labels are a *coarsening* of the
#: release's tag vocabulary: the real records tag the answer format as one of
#: ``TF``/``MC``/``NU``/``DE``, and ``SA`` is a separate "short answer" modifier that
#: co-occurs with ``TF``, ``NU`` and ``DE``. Mapping the paper's names onto the release:
#: TF→``TF``, MC→``MC``, SA→``NU``, TD→``DE``.
ANSWER_FORMAT_TAGS: Final[frozenset[str]] = frozenset({"TF", "MC", "NU", "DE"})

#: The three preparedness stages of §4.1 / CyPortQA Table 1.
STAGE_TAGS: Final[frozenset[str]] = frozenset({"S1", "S2", "S3"})

#: The ability dimensions a question probes (§4.1: "spatial awareness, temporal
#: understanding, exposure interpretation, and uncertainty quantification"), plus the
#: S2/S3 dimensions the CyPortQA paper introduces.
DIMENSION_TAGS: Final[frozenset[str]] = frozenset({"SA", "TU", "EI", "UQ", "IR", "WP"})

#: Markers that are neither a stage, a dimension, nor an answer format.
ANNOTATION_TAGS: Final[frozenset[str]] = frozenset({"AUTO", "CC", "FC", "CO", "FO", "GV", "OV", "PF"})

#: The USCG port-condition ladder, in escalation order. §A.3's ``operation`` stage and
#: §A.6 Eq. 11's precedence term both need this order, and the release's own data is
#: self-consistent with it: ``Condition`` only ever takes these four values (plus the
#: ``XRAY`` spelling variant — see ``KNOWN_DEFECTS``).
PORT_CONDITIONS: Final[tuple[str, ...]] = ("WHISKEY", "X-RAY", "YANKEE", "ZULU")

#: Accepted spellings of a port condition, normalised to :data:`PORT_CONDITIONS`.
PORT_CONDITION_ALIASES: Final[dict[str, str]] = {
    "WHISKEY": "WHISKEY",
    "X-RAY": "X-RAY",
    "XRAY": "X-RAY",
    "X RAY": "X-RAY",
    "YANKEE": "YANKEE",
    "ZULU": "ZULU",
}

#: The release's ``operation`` block, when present, holds a ``Condition`` and up to three
#: free-text guidance strings.
OPERATION_FIELDS: Final[tuple[str, ...]] = ("Condition", "VesselOps", "OceanVesselOps", "FacilityOps")


# --------------------------------------------------------------------------------------
# Defect registry
# --------------------------------------------------------------------------------------


class DefectSeverity(str, Enum):
    """How a measured property of the release affects this reproduction."""

    BLOCKING = "blocking"  # the file cannot be consumed as shipped
    LABEL = "label"  # the file loads, but a ground-truth answer is wrong or ambiguous
    FORMAT = "format"  # the file loads and the answer is right, but the encoding is hostile
    SIMPLIFICATION = "simplification"  # the release is *easier* to use than the paper implies


@dataclass(frozen=True)
class DatasetDefect:
    """One measured inconsistency in the released files.

    ``measure`` is a callable so the claim stays falsifiable: :func:`audit_defects`
    re-runs it against the files on disk rather than restating a number frozen here.
    """

    defect_id: str
    severity: DefectSeverity
    summary: str
    affected: str
    workaround: str


KNOWN_DEFECTS: Final[tuple[DatasetDefect, ...]] = (
    DatasetDefect(
        defect_id="qa-answer-placeholder-none",
        severity=DefectSeverity.LABEL,
        summary=(
            "7,660 QA records ship the literal string \"None\" as their answer and 677 ship an empty "
            "string — together 7.1% of the benchmark. Importantly, most of these are faithful copies "
            "rather than generation failures: the encoding itself stores the *string* \"None\" (not "
            "JSON null) in `impact_prediction` for 4,002 (scenario, horizon) pairs, and the QA "
            "generator copied it through. The underlying gap is in the source data — the "
            "AIS-derived impact predictions are unavailable for those scenarios — so these questions "
            "are unanswerable from the encoding by construction, not merely by accident. The rest "
            "correlate with `operation`, which is empty in 2,560 of 2,910 horizon blocks."
        ),
        affected="dataset/CyPortQA.json (8,337 records), Encoded_senario.json (4,002 impact_prediction blocks)",
        workaround=(
            "Treat them as unanswerable and exclude them from accuracy denominators — they cannot be "
            "answered from the released data, so scoring them measures nothing. `QARecord.is_answerable` "
            "marks them; `is_placeholder_answer` is the predicate."
        ),
    ),
    DatasetDefect(
        defect_id="template-q36-answer-field-mismatch",
        severity=DefectSeverity.LABEL,
        summary=(
            "Q36 asks 'When is the expected end of impact on freight?' and its template names "
            "`impact_prediction.day_of_recover` as the answer source, but the shipped answer is "
            "neither that field nor `end_impact_date`: it equals `start_impact_date` in 100% of "
            "the 2,910 cases (sample: ANA_2015 Savannah at every horizon answers 5/4/2015 while "
            "`end_impact_date` is 5/6/2015). Q34 and Q35 answer from the same value in 2,079 of "
            "2,910 groups, so all three date questions collapse onto one field."
        ),
        affected="Q34, Q35, Q36 (S2/IR/NU — 8,730 records)",
        workaround=(
            "Score Q34 against `start_impact_date` only, and report Q35/Q36 as label-defective: "
            "`Template.answer_path` returns the template's declared path, and `audit_templates` "
            "flags the discrepancy rather than hiding it. Excluding them entirely is the "
            "conservative option for a headline number."
        ),
    ),
    DatasetDefect(
        defect_id="qa-q33-answer-is-python-repr",
        severity=DefectSeverity.FORMAT,
        summary=(
            "Q33 says 'Answer with ONLY a JSON' but its ground truth is a Python `repr` of a "
            "dict, with single quotes and an unquoted-looking percentage: "
            "\"{'in 12h': '1%', 'in 24h': '3%', ...}\". `json.loads` rejects it."
        ),
        affected="Q33 (S2/IR/NU — 2,908 records)",
        workaround="`QARecord.answer_structured` parses it with `ast.literal_eval` after `json.loads` fails.",
    ),
    DatasetDefect(
        defect_id="imbalanced-fragments-unterminated",
        severity=DefectSeverity.BLOCKING,
        summary=(
            "`CyPortQA_Imbalanced_part1.json` and `_part2.json` are two fragments of one file cut "
            "mid-stream — part1 ends with `},` and part2 begins with a bare key — but concatenating "
            "them still does not yield valid JSON: a brace-depth scan that is string-aware shows the "
            "concatenation never returns to depth 0, so a container opened inside part2 is never "
            "closed. Per-part brace counts are 61,559/61,558 and 88,560/88,560; the second part "
            "should close one more brace than it opens."
        ),
        affected="source_data/CyPortQA_Imbalanced/CyPortQA_Imbalanced_part{1,2}.json (122 MB)",
        workaround=(
            "`load_imbalanced` repairs the stream with a string-aware record scanner and reports how "
            "many scenarios it recovered, refusing to claim a clean parse. The split also uses a "
            "cleaner schema than `dataset/CyPortQA.json` — MCQ options are a separate `choice` array "
            "instead of being inlined into the question string — so it is worth recovering."
        ),
    ),
    DatasetDefect(
        defect_id="qa-context-arity-is-variable",
        severity=DefectSeverity.FORMAT,
        summary=(
            "`context` is a positional array whose arity is not fixed: 85,233 records carry "
            "[port, storm, year, horizon] and 31,945 carry [storm, year, horizon] with the port "
            "omitted. The 3-element form is used by exactly the port-independent templates "
            "(Q4-Q9, Q17, Q18, Q20, Q21, Q23). Indexing `context[3]` positionally, as the array "
            "invites, silently reads the wrong field on 27% of the benchmark."
        ),
        affected="all 117,178 QA records",
        workaround="`QAContext.parse` keys on arity and validates the horizon against `HORIZON_HOURS`.",
    ),
    DatasetDefect(
        defect_id="mc-options-duplicated-and-inconsistent-with-prompt",
        severity=DefectSeverity.FORMAT,
        summary=(
            "All 37,813 multiple-choice records carry their options twice — as a structured "
            "`choice` array and inlined into the `question` string — and the two agree in length "
            "on every record. The inconsistency is in the *boilerplate*: both the `prompt` and the "
            "`question` open with a clause stating the option count in words, and the question's "
            "copy is stale on 2,851 records (all of Q18). Q18 asks for the SSHS category, offers "
            "A-G, and the prompt correctly says '(A, B, C, D, E, F or G)', but the question itself "
            "still says 'Answer with ONLY a choice (A, B, C, or D)'. The shipped answer is a bare "
            "letter with a trailing period ('B.'), so a consumer that trusts the question's own "
            "clause would truncate the option list to four and misread answers E-G."
        ),
        affected="all MC records (37,813); stale clause on 2,851 (Q18)",
        workaround=(
            "`load_qa` prefers the `choice` array and falls back to `parse_inline_options`, keeping "
            "`extra['choice']` for inspection. `resolve_mc_letter` maps the letter to its text. Do "
            "not derive the option list from the question's clause."
        ),
    ),
    DatasetDefect(
        defect_id="raw-categorical-fields-not-normalised",
        severity=DefectSeverity.FORMAT,
        summary=(
            "Fields that §A.2 expects to normalise are shipped in raw natural-language form with "
            "hundreds of surface variants. Measured on `Encoded_senario.json`: "
            "`cyclone_condition_and_forecast.category` has 130+ distinct strings for what is really "
            "the SSHS ladder plus a pre-storm ladder (e.g. 'Category 2', 'Category 2 Hurricane', "
            "'Category 2 (105-110 mph, forecast to reach major-hurricane strength)', "
            "'Hurricane (Category 1 now - forecast to strengthen)'); `within_uncertainty_cone` has "
            "~22 spellings of Yes/No; `under_watch` has ~40. `operation.Condition` is the one clean "
            "field, with the single exception of the `XRAY`/`X-RAY` split."
        ),
        affected="cyclone_condition_and_forecast.category, weather_observation.within_uncertainty_cone, weather_observation.under_watch",
        workaround=(
            "This is the workload §A.2 step 1's alias table and §A.2 step 2's entity-id canonicalisation "
            "exist to absorb. It is evidence *for* the paper's design rather than a defect to route "
            "around, but it means the alias table has to be built against these real strings — "
            "`vocabulary.RELATION_ALIASES` alone is not enough, and `data/CONTRACT.md` records the "
            "expected normalisation for each field."
        ),
    ),
    DatasetDefect(
        defect_id="ordinal-buckets-overlap",
        severity=DefectSeverity.LABEL,
        summary=(
            "Six templates (Q23, Q24, Q25, Q27, Q40 and Q26's predicate sibling) do not answer from "
            "their declared field directly: the field is a scalar that has to be placed in an "
            "ordinal bucket supplied by the question's options. Several of those bucket sets "
            "overlap, so one value satisfies two options and the question is not decidable from the "
            "encoding at all. Q25 offers '0-12 hours', '12-24 Hours', '24-48 hours', '48 hours or "
            "more', making 12, 24 and 48 each admissible under two options; Q27's '0 hours' and "
            "'0-24 Hours' both admit 0; Q24's '13-24 hours' and '24 hours or more' both admit 24. "
            "§4.1 scores these by exact letter match, so part of CyPortQA's headline accuracy is "
            "decided by an ambiguity the field cannot resolve."
        ),
        affected="Q23, Q24, Q25, Q27, Q40 (ordinal buckets); Q26 (threshold predicate)",
        workaround=(
            "Classify the templates in `cyportqa.TEMPLATE_SEMANTICS` and score them through "
            "`groundtruth.bucket_letters`, which accepts any bucket containing the value and reports "
            "whether the value was ambiguous. `TemplateAudit.determinacy` then quantifies the "
            "unavoidable share. Report agreement and determinacy side by side; reporting agreement "
            "alone overstates how well the benchmark measures what it claims to."
        ),
    ),
    DatasetDefect(
        defect_id="td-answers-are-verbatim-field-copies",
        severity=DefectSeverity.SIMPLIFICATION,
        summary=(
            "All 24,094 text-description answers are byte-identical to the value at the field their "
            "template declares — not merely equivalent, identical. §4.1 scores this type with \"an "
            "LLM-judged semantic score in [0,1] using GPT-4o as the evaluator\". On these labels that "
            "judge has nothing to do: the reference answer and the source field are the same string, "
            "so exact match measures the generator's fidelity to the retrieved evidence, which is "
            "what the metric is for, while the judge can only add sampling variance to a number that "
            "needs none."
        ),
        affected="Q4, Q5, Q7, Q9, Q10, Q17, Q20, Q32, Q43-Q48 (24,094 records)",
        workaround=(
            "`compare` scores a description by normalised exact match, which the early equality check "
            "routes to before the format dispatch. Keep the §4.1 GPT-4o judge available as an "
            "ablation — it demonstrates the judge does not change the ranking — but do not make it "
            "the primary metric, or a reproduction imports stochasticity the benchmark does not have."
        ),
    ),
    DatasetDefect(
        defect_id="paper-counts-exceed-release",
        severity=DefectSeverity.FORMAT,
        summary=(
            "§4.1 states '2,917 real-world disruption scenarios ... spanning 145 U.S. principal ports "
            "and 90 named storms'. The release has 472 storm-year-port scenario keys, 45 distinct "
            "ports and 69 distinct storms. The question count (117,178) and the horizon range "
            "(T-120..T-12) match exactly. Reading §4.1's 'scenario' as a (storm, port, year, horizon) "
            "quadruple reconciles the scenario number — `Encoded_senario.json` holds 2,910 horizon "
            "blocks — but not the port and storm counts."
        ),
        affected="§4.1 vs dataset/CyPortQA.json",
        workaround=(
            "Report the counts the release actually contains. Do not write 145 ports / 90 storms into "
            "any table this reproduction emits."
        ),
    ),
)


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------


class AnswerFormat(str, Enum):
    """The release's answer-format tag, mapped onto §4.1's four evaluation regimes."""

    TRUE_FALSE = "TF"  # §4.1 — exact match
    MULTIPLE_CHOICE = "MC"  # §4.1 — exact match
    NUMERIC = "NU"  # §4.1 — "Short Answer / Numeric (SA) [...] tolerance-based accuracy"
    DESCRIPTION = "DE"  # §4.1 — "Text Description (TD) [...] an LLM-judged semantic score in [0,1]"

    @classmethod
    def of(cls, tags: Sequence[str]) -> "AnswerFormat":
        """Pick the format tag out of a record's tag list.

        Raises:
            ValueError: If the record carries no format tag, or more than one. Both would
                mean the taxonomy this module encodes no longer matches the release.
        """
        found = [tag for tag in tags if tag in ANSWER_FORMAT_TAGS]
        if len(found) != 1:
            raise ValueError(f"expected exactly one answer-format tag, got {found} in {list(tags)}")
        return cls(found[0])


@dataclass(frozen=True)
class QAContext:
    """The parsed ``context`` array.

    The array is positional and its arity varies (see the ``qa-context-arity-is-variable``
    defect), so it is never indexed directly — this is the only place that knows the
    layout.
    """

    storm: str
    year: int
    horizon_hours: int
    port: str | None = None

    @classmethod
    def parse(cls, context: Sequence[str]) -> "QAContext":
        """Build from a raw ``context`` array.

        Raises:
            ValueError: If the arity is neither 3 nor 4, or the horizon is not one of
                :data:`HORIZON_HOURS`.
        """
        if len(context) == 4:
            port, storm, year, horizon = context
        elif len(context) == 3:
            port, (storm, year, horizon) = None, context
        else:
            raise ValueError(f"context must have 3 or 4 elements, got {len(context)}: {list(context)}")

        hours = int(horizon)
        if hours not in HORIZON_HOURS:
            raise ValueError(f"horizon {hours!r} is not one of {HORIZON_HOURS}")
        return cls(storm=storm, year=int(year), horizon_hours=hours, port=port)

    def as_dict(self) -> dict[str, Any]:
        return {
            "port": self.port,
            "storm": self.storm,
            "year": self.year,
            "horizon_hours": self.horizon_hours,
        }


def is_placeholder_answer(answer: Any) -> bool:
    """True when the shipped answer carries no information.

    Covers the ``qa-answer-placeholder-none`` defect: the string ``"None"`` and the empty
    string. Deliberately does *not* treat a numeric ``0`` or the string ``"No"`` as
    placeholder — both are legitimate answers in this benchmark.
    """
    if answer is None:
        return True
    if isinstance(answer, str):
        return answer.strip() == "" or answer.strip().lower() == "none"
    return False


_INLINE_OPTION_RE: Final[re.Pattern[str]] = re.compile(r"(?<!\S)([A-G])\.\s")


def parse_inline_options(question: str) -> tuple[tuple[str, str], ...]:
    """Recover ``((letter, text), ...)`` from an MC question with inlined options.

    The question is laid out as ``<stem> (A, B, C, or D). A. alpha B. beta C. gamma
    D. delta``. Splitting naively on ``X.`` misfires on the option-count suffix
    (``or D).``) and on abbreviations, so candidates are collected first and then only the
    trailing run with consecutive letters starting at ``A`` is kept.

    Returns an empty tuple when the question has no inline options — which is the case for
    every non-MC record and for the Imbalanced split, where options are a separate field.
    """
    matches = list(_INLINE_OPTION_RE.finditer(question))
    if not matches:
        return ()

    expected = "A"
    run: list[re.Match[str]] = []
    for match in matches:
        if match.group(1) != expected:
            run.clear()
            if match.group(1) != "A":
                continue
            expected = "A"
        run.append(match)
        expected = chr(ord(expected) + 1)

    if len(run) < 2:
        return ()

    options: list[tuple[str, str]] = []
    for index, match in enumerate(run):
        end = run[index + 1].start() if index + 1 < len(run) else len(question)
        options.append((match.group(1), question[match.end() : end].strip()))
    return tuple(options)


def resolve_mc_letter(answer: str, options: Sequence[tuple[str, str]]) -> str | None:
    """Map a shipped MC answer letter (``"B."``) to its option text.

    Returns ``None`` when the answer is not a letter that appears among ``options``.
    """
    token = answer.strip().rstrip(".").strip().upper()
    if len(token) != 1:
        return None
    for letter, text in options:
        if letter == token:
            return text
    return None


@dataclass(frozen=True)
class QARecord:
    """One row of ``dataset/CyPortQA.json``, plus the identity the release does not give it.

    The release ships no question id, so :attr:`qa_id` is synthesised here. It has to be
    stable across runs because it is what an evaluation table keys on.
    """

    scenario_id: str
    template_id: str  # "Q1".."Q48"
    context: QAContext
    tags: tuple[str, ...]
    answer_format: AnswerFormat
    stage: str | None  # "S1" | "S2" | "S3"
    dimensions: tuple[str, ...]
    annotations: tuple[str, ...]
    modalities: tuple[str, ...]
    prompt: str
    question: str
    answer: Any
    relative_horizon: str | None = None  # e.g. "in_12h" — the operation-block sub-key
    options: tuple[tuple[str, str], ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def qa_id(self) -> str:
        """``<scenario>|<template>|<relative-horizon or ->|<horizon>h``.

        Includes the relative horizon because Q42/Q46-Q48 ask the same template at several
        ``operation`` sub-keys within one absolute horizon; omitting it would collide.
        """
        relative = self.relative_horizon or "-"
        return f"{self.scenario_id}|{self.template_id}|{relative}|T-{self.context.horizon_hours}"

    @property
    def group_id(self) -> str:
        """The knowledge group this question draws on.

        §A.4's negative sampler draws "random cross-group pairs", and a group must not span
        a train/test split or the transition model would be validated on entities it trained
        on. One scenario is the natural unit: its horizon blocks share the port, the storm
        and every temporal anchor.
        """
        return self.scenario_id

    @property
    def is_answerable(self) -> bool:
        """False for the records caught by the ``qa-answer-placeholder-none`` defect."""
        return not is_placeholder_answer(self.answer)

    @property
    def answer_text(self) -> str:
        """The shipped answer as a string, for uniform handling."""
        return self.answer if isinstance(self.answer, str) else json.dumps(self.answer, ensure_ascii=False)

    def structured_answer(self) -> Any:
        """Parse the answer when it is a serialised mapping, else return it unchanged.

        Q33 is the reason this exists: it ships a Python ``repr`` (`{'in 12h': '1%'}`), not
        JSON, so ``ast.literal_eval`` is the fallback rather than the first resort.
        """
        if not isinstance(self.answer, str):
            return self.answer
        try:
            return json.loads(self.answer)
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            return ast.literal_eval(self.answer)
        except (ValueError, SyntaxError):
            return self.answer


@dataclass(frozen=True)
class ScenarioId:
    """A parsed ``STORM_YEAR_PORT`` scenario key."""

    storm: str
    year: int
    port: str
    raw: str

    @classmethod
    def parse(cls, raw: str) -> "ScenarioId":
        match = SCENARIO_ID_RE.match(raw)
        if match is None:
            raise ValueError(f"scenario key {raw!r} does not match STORM_YEAR_PORT")
        return cls(storm=match.group("storm"), year=int(match.group("year")), port=match.group("port"), raw=raw)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.raw


# --------------------------------------------------------------------------------------
# Encoded scenarios
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeBlock:
    """One horizon's worth of knowledge: the four §4.1 blocks.

    Kept as the raw dicts rather than exploded into fields, because §A.2's extraction step
    is defined over these blocks as documents — the hypergraph builder walks them, it does
    not replace them.
    """

    horizon_hours: int
    cyclone_condition_and_forecast: Mapping[str, Any]
    weather_observation: Mapping[str, Any]
    operation: Mapping[str, Any]
    impact_prediction: Mapping[str, Any]

    @property
    def condition(self) -> str | None:
        """The USCG port condition currently in force, normalised through the alias table.

        Returns ``None`` when no bulletin had been issued at this horizon — true for
        2,560 of the 2,910 blocks, which is why CyPortQA's S3 stage is its sparsest.
        """
        raw = (self.operation or {}).get("now", {}).get("Condition")
        if not isinstance(raw, str):
            return None
        return PORT_CONDITION_ALIASES.get(raw.strip().upper())

    @property
    def forecast_conditions(self) -> tuple[tuple[str, str], ...]:
        """``((relative_offset, condition), ...)`` for the *future* bulletin ladder.

        These are what Q42/Q46-Q48 interrogate, and they are the clearest order-sensitive
        signal in the benchmark: the same scenario at the same absolute horizon predicts a
        *sequence* of future conditions.
        """
        out: list[tuple[str, str]] = []
        for offset, payload in (self.operation or {}).items():
            if not RELATIVE_HORIZON_RE.match(offset):
                continue
            raw = payload.get("Condition") if isinstance(payload, Mapping) else None
            if isinstance(raw, str):
                out.append((offset, PORT_CONDITION_ALIASES.get(raw.strip().upper(), raw.strip())))
        return tuple(sorted(out, key=lambda pair: _relative_offset_hours(pair[0])))

    def blocks(self) -> Mapping[str, Mapping[str, Any]]:
        """The four blocks keyed by name, for iteration in a fixed order."""
        return {
            "cyclone_condition_and_forecast": self.cyclone_condition_and_forecast,
            "weather_observation": self.weather_observation,
            "operation": self.operation,
            "impact_prediction": self.impact_prediction,
        }


def _relative_offset_hours(offset: str) -> int:
    """``"in_12h"`` → 12; ``"now"`` → 0. Used only for stable ordering."""
    if offset == "now":
        return 0
    match = RELATIVE_HORIZON_RE.match(offset)
    return int(match.group(1)) if match else 10**6


@dataclass(frozen=True)
class EncodedScenario:
    """``{STORM}_{YEAR}_{PORT}`` → ``senario.{port, cyclone, time}``."""

    scenario_id: ScenarioId
    port: Mapping[str, Any]
    cyclone: Mapping[str, Any]
    time: Mapping[str, TimeBlock]

    @property
    def horizons(self) -> tuple[int, ...]:
        """Available lead times, ascending. Scenarios are ragged — the release holds as
        few as 1 and as many as 10 blocks — so callers must not assume
        :data:`HORIZON_HOURS` is fully populated."""
        return tuple(sorted(self.time))

    @property
    def port_name(self) -> str:
        return str(self.port.get("name", ""))

    @property
    def storm_name(self) -> str:
        return str(self.cyclone.get("name", ""))


def _parse_time_block(hours: int, payload: Mapping[str, Any]) -> TimeBlock:
    missing = [name for name in TIME_BLOCK_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"T-{hours} block is missing {missing}; expected all of {list(TIME_BLOCK_FIELDS)}")
    return TimeBlock(
        horizon_hours=hours,
        cyclone_condition_and_forecast=payload["cyclone_condition_and_forecast"],
        weather_observation=payload["weather_observation"],
        operation=payload["operation"],
        impact_prediction=payload["impact_prediction"],
    )


# --------------------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Template:
    """One of the 48 entries of ``CyPortQA_template.json``.

    The decisive field is :attr:`answer`: a placeholder string such as
    ``{senario.time.<key>.weather_observation.under_watch}`` that names the dotted path
    producing the ground-truth answer. ``time.<key>`` is substituted with the record's
    absolute horizon, and templates that interrogate ``operation`` carry a second
    placeholder, ``operation.<key>``, substituted from the record's ``in_XXh`` tag.
    """

    template_id: str
    tags: tuple[str, ...]
    context: tuple[str, ...]
    modalities: tuple[str, ...]
    prompt: str
    question: str
    answer: str

    @property
    def stage(self) -> str | None:
        found = [tag for tag in self.tags if tag in STAGE_TAGS]
        return found[0] if found else None

    @property
    def answer_format(self) -> AnswerFormat:
        return AnswerFormat.of(self.tags)

    @property
    def static_prefix(self) -> str:
        """The part of the answer placeholder before the horizon substitution.

        For ``{senario.time.<key>.weather_observation.under_watch}`` this is
        ``weather_observation.under_watch``. For templates whose answer is a scenario-level
        field (Q16, Q19 answer ``{senario.port.name}``) it is the scenario-relative part.
        Returns ``""`` when the placeholder cannot be located, which :func:`audit_templates`
        treats as a defect rather than silently ignoring.
        """
        match = re.search(r"\{senario\.time\.<key>\.([^{}]+)\}", self.answer)
        if match:
            return match.group(1)
        match = re.search(r"\{senario\.([^{}]+)\}", self.answer)
        return match.group(1) if match else ""

    @property
    def uses_relative_horizon(self) -> bool:
        """True when the answer path has an ``operation.<key>`` segment needing an
        ``in_XXh`` tag (Q42, Q46, Q47, Q48)."""
        return "<key>" in self.static_prefix

    @property
    def semantics(self) -> "TemplateSemantics":
        """How the declared field relates to the literal answer — see :class:`TemplateSemantics`."""
        return TEMPLATE_SEMANTICS.get(self.template_id, TemplateSemantics.LOOKUP)


class TemplateSemantics(str, Enum):
    """How a template's declared field becomes its answer.

    This distinction is not in the CyPortQA paper and not in OKH-RAG's, but it decides what
    an evaluator may legitimately expect. Most templates name the answer outright; a handful
    name an *operand* that still has to be transformed. Getting this wrong makes correct
    answers look wrong, which is exactly what happens if Q26's "No" is compared as a string
    against the field value ``"11%"``.
    """

    LOOKUP = "lookup"
    """The field value *is* the answer, up to normalisation."""

    ORDINAL_BUCKET = "ordinal_bucket"
    """The field is a scalar that must be placed in an ordinal bucket supplied by the
    question's options — e.g. Q23 asks for the forecast duration and offers ``2 days`` …
    ``5 days`` while the field holds ``"5"``. The option text carries a unit the field
    omits, so string equality always fails and the comparison has to be numeric."""

    THRESHOLD_PREDICATE = "threshold_predicate"
    """The field is an operand of a threshold stated in the question, and the answer is the
    predicate's truth value — e.g. Q26 asks whether 34-kt wind probability "reach[es] or
    exceed[s] 20%" while the field holds ``"11%"``. The threshold lives in the question
    text, not in the encoding, so the answer is not recoverable from the field alone."""


#: Per-template semantics, established by reading each question against its field.
#:
#: Only the templates that need a non-``LOOKUP`` treatment are listed; everything else
#: defaults to :attr:`TemplateSemantics.LOOKUP`. The bucket entries are the ones whose
#: options are ordinal ranges over a scalar field, which the audit measures — see
#: ``KNOWN_DEFECTS`` and ``reports/cyportqa_audit.md``.
TEMPLATE_SEMANTICS: Final[dict[str, TemplateSemantics]] = {
    # "total forecasted duration", options 2/3/4/5 days, field holds bare days
    "Q23": TemplateSemantics.ORDINAL_BUCKET,
    # "duration the port is affected", options 0 / 1-12 / 13-24 / 24+ hours,
    # field holds hours
    "Q24": TemplateSemantics.ORDINAL_BUCKET,
    # "which interval holds the max 34-kt probability", options 0-12 / 12-24 / 24-48 / 48+,
    # field holds the peak's lead time in hours
    "Q25": TemplateSemantics.ORDINAL_BUCKET,
    # "whether 34-kt probability reaches or exceeds 20%", field holds the percentage
    "Q26": TemplateSemantics.THRESHOLD_PREDICATE,
    # "first time 34-kt probability rises above zero", options 0 / 0-24 / 24-48 / 48+
    "Q27": TemplateSemantics.ORDINAL_BUCKET,
    # "total impacted commercial vessels", options None / 1-10 / 10-20 / 20+,
    # field holds a vessel count
    "Q40": TemplateSemantics.ORDINAL_BUCKET,
}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _read_json(path: Path, *, encoding: str = "utf-8") -> Any:
    """Read JSON, tolerating a UTF-8 BOM.

    ``CyPortQA_template.json`` and the Imbalanced fragments are shipped with a BOM while
    ``dataset/CyPortQA.json`` is not. A BOM decodes cleanly as ``utf-8`` into a leading
    ``\\ufeff``, so the decode succeeds and the *parser* is what rejects it — which is why
    the fallback has to catch ``JSONDecodeError`` and not only ``UnicodeDecodeError``.
    """
    raw = path.read_bytes()
    last_error: Exception | None = None
    for candidate in (encoding, "utf-8-sig"):
        try:
            return json.loads(raw.decode(candidate))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            last_error = error
    raise ValueError(f"{path} is not valid JSON under {encoding} or utf-8-sig") from last_error


def options_from_choice(choice: Sequence[Any]) -> tuple[tuple[str, str], ...]:
    """Letter the release's ``choice`` array.

    The array is positional and unlabelled, so the letters are assigned in order — which is
    what the questions themselves do (``A. … B. … C. …``) and what the shipped answer letters
    refer to. Verified against the inlined text: for every one of the 37,813 multiple-choice
    records the two agree.
    """
    return tuple((chr(ord("A") + index), str(text)) for index, text in enumerate(choice))


def load_qa(root: Path) -> dict[str, tuple[QARecord, ...]]:
    """Read ``dataset/CyPortQA.json`` into ``{scenario_id: (QARecord, ...)}``.

    Args:
        root: the repository root, i.e. the directory holding ``dataset/`` and
            ``source_data/``.

    Raises:
        ValueError: On a record whose tag list has no unique answer-format tag, or whose
            context arity or horizon is off-spec. Both indicate the release has changed
            shape underneath this loader, which should be loud rather than silent.
    """
    payload = _read_json(root / "dataset" / "CyPortQA.json")
    out: dict[str, tuple[QARecord, ...]] = {}
    for scenario_id, body in payload.items():
        records: list[QARecord] = []
        for raw in body["qa"]:
            tags = tuple(raw["question_type"])
            by_prefix = [tag for tag in tags if tag.startswith("Q")]
            if len(by_prefix) != 1:
                raise ValueError(f"{scenario_id}: expected one Q-tag, got {by_prefix}")
            relative = next((tag for tag in tags if RELATIVE_HORIZON_RE.match(tag)), None)
            question = raw["question"]
            # The `choice` array is authoritative when present: it is structured, whereas
            # the inlined options have to be recovered by pattern and can be confused by an
            # abbreviation or a decimal in the option text. Both are present on all 37,813
            # multiple-choice records, and they agree.
            declared = raw.get("choice")
            options = options_from_choice(declared) if declared else parse_inline_options(question)
            records.append(
                QARecord(
                    scenario_id=scenario_id,
                    template_id=by_prefix[0],
                    context=QAContext.parse(raw["context"]),
                    tags=tags,
                    answer_format=AnswerFormat.of(tags),
                    stage=next((tag for tag in tags if tag in STAGE_TAGS), None),
                    dimensions=tuple(tag for tag in tags if tag in DIMENSION_TAGS),
                    annotations=tuple(tag for tag in tags if tag in ANNOTATION_TAGS),
                    modalities=tuple(raw.get("modalities", ())),
                    prompt=raw.get("prompt", ""),
                    question=question,
                    answer=raw.get("answer"),
                    relative_horizon=relative,
                    options=options,
                    extra={"choice": tuple(map(str, declared))} if declared else {},
                )
            )
        out[scenario_id] = tuple(records)
    return out


def load_encoded(root: Path) -> dict[str, EncodedScenario]:
    """Read ``source_data/Encoded_senario.json``.

    Raises:
        ValueError: If a horizon block is missing one of :data:`TIME_BLOCK_FIELDS`, or a
            scenario key does not parse. The four-block shape holds for all 2,910 blocks in
            the release, so a violation means the file changed.
    """
    payload = _read_json(root / "source_data" / "Encoded_senario.json")
    out: dict[str, EncodedScenario] = {}
    for scenario_id, body in payload.items():
        senario = body["senario"]
        time_payload = senario["time"]
        blocks: dict[int, TimeBlock] = {}
        for key, block in time_payload.items():
            hours = int(key)
            blocks[hours] = _parse_time_block(hours, block)
        out[scenario_id] = EncodedScenario(
            scenario_id=ScenarioId.parse(scenario_id),
            port=senario["port"],
            cyclone=senario["cyclone"],
            time=blocks,
        )
    return out


def load_templates(root: Path) -> dict[str, Template]:
    """Read ``source_data/CyPortQA_template.json`` into ``{"Q1": Template, ...}``."""
    payload = _read_json(root / "source_data" / "CyPortQA_template.json")
    out: dict[str, Template] = {}
    for raw in payload:
        tags = tuple(raw["question_type"])
        ids = [tag for tag in tags if tag.startswith("Q")]
        if len(ids) != 1:
            raise ValueError(f"template has {len(ids)} Q-tags: {list(tags)}")
        out[ids[0]] = Template(
            template_id=ids[0],
            tags=tags,
            context=tuple(raw.get("context", ())),
            modalities=tuple(raw.get("modalities", ())),
            prompt=raw.get("prompt", ""),
            question=raw.get("question", ""),
            answer=raw.get("answer", ""),
        )
    return out


def load_imbalanced(root: Path) -> tuple[dict[str, list[dict[str, Any]]], str]:
    """Recover the Imbalanced split despite the ``imbalanced-fragments-unterminated`` defect.

    Scans the concatenated stream with a string-aware, brace-counting reader that emits a
    scenario object at every point the depth returns to 1 and a complete ``{...}`` has been
    consumed. Records found after the point where the stream stops being interpretable are
    dropped rather than guessed at.

    Returns:
        ``(scenarios, note)``. ``scenarios`` maps ``scenario_id`` to the raw ``qa`` list;
        ``note`` states what was and was not recovered, so a caller can log it instead of
        reporting a clean parse that did not happen.
    """
    base = root / "source_data" / "CyPortQA_Imbalanced"
    text = (
        (base / "CyPortQA_Imbalanced_part1.json").read_bytes()
        + (base / "CyPortQA_Imbalanced_part2.json").read_bytes()
    ).decode("utf-8-sig")

    scenarios: dict[str, list[dict[str, Any]]] = {}
    depth = 0
    in_string = False
    escaped = False
    string_start = 0
    key: str | None = None
    value_start = 0
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
                if depth == 1 and key is None:
                    key = text[string_start + 1 : index]
        else:
            if char == '"':
                in_string = True
                string_start = index
            elif char == "{":
                if depth == 1 and key is not None:
                    value_start = index
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 1 and key is not None:
                    try:
                        body = json.loads(text[value_start:index + 1])
                        scenarios[key] = body.get("qa", []) if isinstance(body, dict) else []
                    except json.JSONDecodeError:
                        pass
                    key = None
        index += 1

    expected = 700  # measured scenario-key count across both fragments
    note = (
        f"recovered {len(scenarios)} scenario objects from the concatenated Imbalanced fragments "
        f"({expected} keys are present in the raw text); the stream never returns to brace depth 0, "
        f"so records after the unterminated region are unrecoverable. "
        f"final depth was {depth} (expected 0)."
    )
    return scenarios, note


# --------------------------------------------------------------------------------------
# Convenience container
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CyPortQACorpus:
    """QA records joined to their encoded scenario and template.

    Built by :func:`load_corpus`. The three parts are kept separate as well as joined,
    because the hypergraph builder needs the encoding and the evaluator needs the QA, and
    neither should have to carry the other.
    """

    qa: dict[str, tuple[QARecord, ...]]
    encoded: dict[str, EncodedScenario]
    templates: dict[str, Template]

    def __post_init__(self) -> None:
        orphans = sorted(set(self.qa) - set(self.encoded))
        if orphans:
            raise ValueError(
                f"{len(orphans)} scenarios have QA records but no encoding, e.g. {orphans[:3]}; "
                "the two files are expected to cover the same scenario keys"
            )

    @property
    def scenario_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.encoded))

    @property
    def knowledge_groups(self) -> tuple[str, ...]:
        """Group ids for §A.4's split and negative sampler — one per scenario."""
        return self.scenario_ids

    def records(self) -> Iterator[QARecord]:
        for scenario_id in self.scenario_ids:
            yield from self.qa.get(scenario_id, ())

    def counts(self) -> dict[str, Any]:
        """Corpus statistics for the run manifest and ``data/CONTRACT.md``."""
        by_format: dict[str, int] = {}
        by_stage: dict[str, int] = {}
        answerable = 0
        total = 0
        merged = 0
        for record in self.records():
            total += 1
            answerable += int(record.is_answerable)
            by_format[record.answer_format.value] = by_format.get(record.answer_format.value, 0) + 1
            stage = record.stage or "(none)"
            by_stage[stage] = by_stage.get(stage, 0) + 1
            if record.context.port is None:
                merged += 1
        horizon_blocks = sum(len(scenario.horizons) for scenario in self.encoded.values())
        return {
            "scenarios": len(self.qa),
            "scenario_keys_with_encoding": len(self.encoded),
            "knowledge_groups": len(self.knowledge_groups),
            "horizon_blocks": horizon_blocks,
            "questions": total,
            "questions_answerable": answerable,
            "questions_unanswerable": total - answerable,
            "questions_without_port_in_context": merged,
            "storms": len({sid.split("_")[0] for sid in self.encoded}),
            "ports": len({"".join(sid.split("_")[2:]) for sid in self.encoded}),
            "answers_by_format": dict(sorted(by_format.items())),
            "questions_by_stage": dict(sorted(by_stage.items())),
        }


def load_corpus(root: Path) -> CyPortQACorpus:
    """Load QA, encodings and templates from a CyPortQA checkout."""
    return CyPortQACorpus(qa=load_qa(root), encoded=load_encoded(root), templates=load_templates(root))


__all__ = [
    "ANNOTATION_TAGS",
    "ANSWER_FORMAT_TAGS",
    "DIMENSION_TAGS",
    "HORIZON_HOURS",
    "KNOWN_DEFECTS",
    "OPERATION_FIELDS",
    "PORT_CONDITIONS",
    "PORT_CONDITION_ALIASES",
    "RELATIVE_HORIZON_RE",
    "STAGE_TAGS",
    "TEMPLATE_SEMANTICS",
    "TIME_BLOCK_FIELDS",
    "AnswerFormat",
    "CyPortQACorpus",
    "DatasetDefect",
    "DefectSeverity",
    "EncodedScenario",
    "QAContext",
    "QARecord",
    "ScenarioId",
    "Template",
    "TemplateSemantics",
    "TimeBlock",
    "is_placeholder_answer",
    "load_corpus",
    "load_encoded",
    "load_imbalanced",
    "load_qa",
    "load_templates",
    "options_from_choice",
    "parse_inline_options",
    "resolve_mc_letter",
]
