"""§A.8 — the generator's output contract, and the validation that keeps it honest.

§A.8 closes with a format requirement::

    Structured JSON with fields answer, confidence ∈ [0, 1], and rationale. Type-specific constraints
    ensure evaluation-friendly output: "Yes"/"No" for true/false, letter labels for multiple choice,
    numeric values for quantitative questions, free text for descriptions.

Every clause of that sentence is doing work, and they are separable:

* **Structured JSON with three fields** is what makes the answer machine-readable *and* auditable — a
  confidence and a rationale are what let a failed question be diagnosed as "the model was unsure" rather
  than only "the model was wrong".
* **confidence ∈ [0, 1]** is a range constraint, so a model returning ``85`` (meaning percent) is
  violating the contract, not expressing high confidence in a different unit.
* **type-specific constraints** are the part that matters most for §4.1, because the four evaluation
  regimes are not equally forgiving. §4.1 scores TF and MC by exact match, so an answer of ``"no"``
  where the release stores ``"No"`` is a miss unless normalisation is applied deliberately, and an answer
  of ``"False"`` for a True/False question is not an answer at all. SA is scored with a tolerance, which
  requires a *number*; a model that answers ``"about 40 mph"`` has given something a tolerance cannot be
  applied to. TD is scored on free text by construction.

This module therefore does two jobs and keeps them apart:

1. :func:`validate` — reports contract violations **without correcting them**. A violation is a fact about
   the run; silently repairing it would erase the evidence that the generator drifted.
2. :func:`canonicalise` — deliberately repairs what §4.1's own scoring rules say to repair (case folding,
   letter extraction, numeric casting), so that scoring and diagnostics cannot disagree about what the
   model said.

The distinction is the reason violations are a *return value* rather than an exception: a batch run wants
to score the 99% that complied and report the 1% that did not, not abort on the first malformed answer.

# [UNSPECIFIED] §A.8 does not say what to do with a non-conforming answer. The choice here is to record
# the violation, keep the raw text, and let the scorer decide — which is strictly more information than
# either coercing or discarding.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from okh_rag.data.cyportqa import AnswerFormat

__all__ = [
    "ANSWER_JSON_SCHEMA_NAME",
    "AnswerViolation",
    "GeneratedAnswer",
    "answer_json_schema",
    "canonicalise",
    "numeric_value",
    "parse_generated_answer",
    "validate",
    "vote_key",
]


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------


class AnswerViolation(str, Enum):
    """Ways a generator's output can fail §A.8's contract.

    Enumerated rather than free-text so that a run's violations aggregate into counts. The first three are
    contract-level and apply to every format; the rest are format-specific.
    """

    NOT_JSON = "not_json"  # the response was not parseable as a JSON object
    MISSING_FIELD = "missing_field"  # one of answer/confidence/rationale was absent
    BAD_CONFIDENCE = "bad_confidence"  # confidence was absent, non-numeric, or outside [0, 1]
    WRONG_TYPE = "wrong_type"  # answer is not a string/int/float where one is required
    NOT_YES_NO = "not_yes_no"  # a TF question answered with something other than Yes/No
    NOT_A_LABEL = "not_a_label"  # an MC question answered with something that is not an option label
    NOT_NUMERIC = "not_numeric"  # an SA question answered with something that is not a number
    EMPTY_TEXT = "empty_text"  # a TD question answered with nothing


# ---------------------------------------------------------------------------
# The answer object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedAnswer:
    """One generator response, parsed and judged against §A.8.

    Attributes:
        answer: The answer as the model produced it, *before* any §4.1 normalisation. Kept verbatim
            because the raw string is what a violation report has to quote.
        confidence: The ``confidence`` field, or ``None`` when it was missing or unparseable. ``None``
            rather than ``0.0``: "the model did not state a confidence" and "the model stated zero" are
            different, and the second is what a per-trajectory vote would weight by.
        rationale: The ``rationale`` field, or ``""`` when absent. Not optional in the schema, but an
            absent rationale is not a scoring failure, so it degrades instead of violating.
        supporting_trajectory: §A.8's single-call mode asks the model to "note which trajectory supports
            the final answer"; this is that note, kept as the raw label the model returned.
        violations: Everything wrong with the response. Empty for a conforming answer.
        raw: The parsed JSON object, or ``{}`` when the response was not JSON. Retained so that a run
            manifest can show exactly what came back.
    """

    answer: Any
    confidence: float | None = None
    rationale: str = ""
    supporting_trajectory: str | None = None
    violations: tuple[AnswerViolation, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_conforming(self) -> bool:
        """Whether the response satisfied every clause of §A.8's contract."""
        return not self.violations

    @property
    def weight(self) -> float:
        """The confidence to use when aggregating this answer into a vote.

        A missing confidence is treated as a weight of 1.0 — the neutral element of the weighted vote —
        rather than 0.0, which would silently delete the answer from the aggregate. §A.8's fallback mode
        aggregates "independent per-trajectory answers", and an answer with no stated confidence is still
        one of them.
        """
        return 1.0 if self.confidence is None else self.confidence

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view for run manifests and JSON reports."""
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "supporting_trajectory": self.supporting_trajectory,
            "violations": [v.value for v in self.violations],
            "raw": dict(self.raw),
        }


# ---------------------------------------------------------------------------
# The schema handed to the model
# ---------------------------------------------------------------------------

ANSWER_JSON_SCHEMA_NAME = "okh_rag_answer"

# The per-format ``answer`` schema. §A.8's type-specific constraints live here, so the constrained
# decoding the extractor already uses does the enforcing rather than a post-hoc regex.
_ANSWER_FIELD_SCHEMA: Mapping[AnswerFormat, Mapping[str, Any]] = {
    AnswerFormat.TRUE_FALSE: {"type": "string", "enum": ["Yes", "No"]},
    AnswerFormat.MULTIPLE_CHOICE: {"type": "string"},
    AnswerFormat.NUMERIC: {"type": "number"},
    AnswerFormat.DESCRIPTION: {"type": "string"},
}

# The MC enum cannot live in a static dict because the admissible labels differ per question, so it is
# attached by :func:`answer_json_schema`.
_MC_PATTERN = "^[A-Z]$"


def answer_json_schema(
    answer_format: AnswerFormat,
    *,
    options: Sequence[tuple[str, str]] = (),
    multi_trajectory: bool = False,
) -> dict[str, Any]:
    """§A.8's output schema, specialised to one answer format.

    Args:
        answer_format: The §4.1 regime this question is scored under.
        options: For MC, the ``(label, text)`` pairs. §A.8 says "letter labels", and a letter is only
            checkable against the question's own options — a schema that accepts any ``[A-Z]`` would let
            the model answer ``"E"`` on a four-option question and pass validation while being unscoreable.
        multi_trajectory: Include ``supporting_trajectory`` in the schema. §A.8 asks for the supporting
            path only when several were presented, and the field is *omitted entirely* rather than left
            optional because OpenAI's strict structured-output mode requires every declared property to
            be in ``required`` — a property that is present but optional is rejected at call time, and a
            property that is required but meaningless invites a fabricated path number.

    Returns:
        A JSON-schema fragment for the ``complete_json`` call. ``additionalProperties: false`` is set
        because an extra field is a sign the prompt drifted, and it is cheaper to notice here than in a
        report.
    """
    field_schema: dict[str, Any] = dict(_ANSWER_FIELD_SCHEMA[answer_format])
    if answer_format is AnswerFormat.MULTIPLE_CHOICE and options:
        labels = [label for label, _ in options]
        field_schema = {"type": "string", "enum": labels}
    elif answer_format is AnswerFormat.MULTIPLE_CHOICE:
        field_schema = {"type": "string", "pattern": _MC_PATTERN}

    properties: dict[str, Any] = {
        "answer": {**field_schema, "description": "The answer, in the format the question demands."},
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "How confident the answer is, from 0 to 1.",
        },
        "rationale": {
            "type": "string",
            "description": "A brief justification grounded in the evidence steps.",
        },
    }
    required = ["answer", "confidence", "rationale"]
    if multi_trajectory:
        properties["supporting_trajectory"] = {
            "type": "string",
            "description": "The number of the reasoning path that supports the answer.",
        }
        required.append("supporting_trajectory")

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_generated_answer(
    text: str,
    answer_format: AnswerFormat,
    *,
    options: Sequence[tuple[str, str]] = (),
) -> GeneratedAnswer:
    """Parse one raw generator response into a :class:`GeneratedAnswer`.

    Never raises. A response that is not JSON, or is JSON of the wrong shape, becomes a
    :class:`GeneratedAnswer` carrying the appropriate violations — that is the point of the function, and
    the caller gets a value it can aggregate either way.

    The parser is deliberately more permissive than the schema: the schema constrains a live call, but a
    cached, replayed or hand-written response arrives as text, and a run that could not be re-parsed would
    make a cached experiment unreproducible. So the parse accepts a JSON object in a code fence or wrapped
    in prose, and validation is what decides whether the *content* was acceptable.
    """
    violations: list[AnswerViolation] = []
    payload = _extract_json_object(text)
    if payload is None:
        return GeneratedAnswer(answer=text, violations=(AnswerViolation.NOT_JSON,), raw={})

    for required in ("answer", "confidence", "rationale"):
        if required not in payload:
            violations.append(AnswerViolation.MISSING_FIELD)

    answer = payload.get("answer")
    confidence = _as_confidence(payload.get("confidence"))
    if confidence is None:
        violations.append(AnswerViolation.BAD_CONFIDENCE)

    rationale = payload.get("rationale")
    rationale_text = rationale if isinstance(rationale, str) else ""

    supporting = payload.get("supporting_trajectory")
    supporting_text = supporting if isinstance(supporting, str) and supporting.strip() else None

    violations.extend(validate(answer, answer_format, options=options))

    return GeneratedAnswer(
        answer=answer,
        confidence=confidence,
        rationale=rationale_text,
        supporting_trajectory=supporting_text,
        violations=_dedupe(violations),
        raw=dict(payload),
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Recover a JSON object from a model response, tolerating fences and surrounding prose.

    Returns:
        The parsed object, or ``None`` when no JSON object could be recovered. A JSON *array* or scalar is
        rejected: §A.8 asks for an object with named fields, and a bare ``"Yes"`` carries no confidence or
        rationale to check.
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None

    # Fast path, then two repairs in order of how often they are needed: a fenced block, then the widest
    # brace-balanced span. The fallback is last because it is the only one that can pick up the wrong
    # object when a rationale happens to contain braces.
    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1).strip())
    braced = _widest_braced_span(stripped)
    if braced is not None:
        candidates.append(braced)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _widest_braced_span(text: str) -> str | None:
    """The substring from the first ``{`` to its matching ``}``, or ``None`` when unbalanced."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for position in range(start, len(text)):
        char = text[position]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : position + 1]
    return None


def _as_confidence(value: Any) -> float | None:
    """Coerce a ``confidence`` field to a float in ``[0, 1]``, or ``None`` when it is not one.

    Out-of-range values are rejected rather than clamped. §A.8 states the range as part of the contract; a
    ``confidence`` of 1.5 means the model is not following it, and clamping to 1.0 would let that pass as
    a maximally confident answer — the most influential possible value, awarded for a contract violation.
    """
    if isinstance(value, bool):
        # ``bool`` is an ``int`` in Python; a boolean is not a confidence.
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not 0.0 <= number <= 1.0:
        return None
    return number


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(
    answer: Any,
    answer_format: AnswerFormat,
    *,
    options: Sequence[tuple[str, str]] = (),
) -> tuple[AnswerViolation, ...]:
    """Check an ``answer`` field against §A.8's type-specific constraint for its format.

    This reports; it does not repair. :func:`canonicalise` is the repairing path, and keeping them separate
    is what lets a report say "3.1% of TD answers were empty" instead of quietly counting them as wrong.

    Returns:
        The violations found, empty when the answer conforms.
    """
    if answer is None:
        return (AnswerViolation.WRONG_TYPE,)

    if answer_format is AnswerFormat.TRUE_FALSE:
        if not isinstance(answer, str):
            return (AnswerViolation.WRONG_TYPE,)
        # Case-insensitive, because the requirement being checked is *polarity*, and §4.1's own scoring
        # normalises case before exact match. Rejecting "yes" here would report a violation for an answer
        # that scores as correct, which is the kind of false alarm that trains a reader to ignore the count.
        return () if answer.strip().lower() in {"yes", "no"} else (AnswerViolation.NOT_YES_NO,)

    if answer_format is AnswerFormat.MULTIPLE_CHOICE:
        if not isinstance(answer, str):
            return (AnswerViolation.WRONG_TYPE,)
        label = _as_label(answer)
        if label is None:
            return (AnswerViolation.NOT_A_LABEL,)
        if options and label not in {option_label for option_label, _ in options}:
            return (AnswerViolation.NOT_A_LABEL,)
        return ()

    if answer_format is AnswerFormat.NUMERIC:
        return () if numeric_value(answer) is not None else (AnswerViolation.NOT_NUMERIC,)

    if answer_format is AnswerFormat.DESCRIPTION:
        if not isinstance(answer, str):
            return (AnswerViolation.WRONG_TYPE,)
        return () if answer.strip() else (AnswerViolation.EMPTY_TEXT,)

    return ()


def _as_label(answer: str) -> str | None:
    """Extract an option label from a model answer.

    Accepts ``"B"``, ``"(B)"``, ``"b"`` and ``"B."`` — the four forms a model reliably produces when asked
    for a letter — and nothing else. In particular ``"B because ..."`` is *not* accepted here, even though
    :func:`canonicalise` will accept it for scoring: this function feeds a violation count, and a length
    limit is the only thing that keeps "the model wrote a paragraph" from registering as a clean label.
    """
    match = re.fullmatch(r"\s*\(?([A-Za-z])\)?\.?\s*", answer)
    return match.group(1).upper() if match else None


#: A number in free text. The lookbehind is what keeps a horizon label intact: in ``"T-48"`` the hyphen is
#: part of the label, not a minus sign, so the match must start at the ``4``. Without it, a question asking
#: "how many hours before landfall?" and answered ``"T-48"`` would score as **-48**. A hyphen after a space
#: is still read as a sign, which is the common convention in prose ("answered -48").
_NUMBER = re.compile(r"(?<![A-Za-z0-9])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def numeric_value(answer: Any) -> float | None:
    """The number an SA answer carries, or ``None`` when it carries none.

    Accepts a numeric type, a numeric string, and a numeric string with a leading label (``"T-48"`` →
    48, ``"40 mph"`` → 40) because §A.8's own evidence format is full of ``T-XX`` labels and a model
    echoing one is answering the right quantity in the wrong costume.

    Rejects the placeholder strings the release uses for a missing value, and rejects anything with two
    distinct numbers: for ``"between 12 and 24"`` there is no single value that §4.1's tolerance can be
    applied to, and picking the first would score a guess as an answer.
    """
    if isinstance(answer, bool):
        return None
    if isinstance(answer, (int, float)):
        return float(answer)
    if not isinstance(answer, str):
        return None

    text = answer.strip()
    if not text or text.lower() in {"none", "null", "n/a", "na", "unknown"}:
        return None

    numbers = _NUMBER.findall(text)
    if len(numbers) != 1:
        return None
    try:
        return float(numbers[0])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Canonicalisation (the repairing path, used by scoring)
# ---------------------------------------------------------------------------


def canonicalise(answer: Any, answer_format: AnswerFormat) -> Any:
    """Reduce an answer to the form §4.1's scoring rules compare.

    Each branch corresponds to a scoring rule the paper states, not to a convenience:

    * TF → the answer stripped, with **case preserved**. Case folding is a scoring decision, not a
      canonicalisation one: ``evaluation.case_sensitive`` exists precisely so that "yes" can be counted
      wrong when the release's "Yes" was asked for, and folding here would make that switch inert. The
      comparison in :func:`~okh_rag.evaluation.answers._strings_equal` applies the config's rule.
    * MC → the bare upper-case label, because §4.1's answer is "letter labels" and a model that prefixes
      its reasoning has still given a label. Upper-casing is kept here — unlike TF — because a single letter
      has no case distinction in this task, and the release's own golds differ only in punctuation.
    * SA → a float, because §4.1's tolerance arithmetic requires one.
    * TD → the text, stripped. No further normalisation: §4.1 scores descriptions semantically (or, on
      this release, by exact match against a verbatim field copy — see ``data/CONTRACT.md`` §5.2), and
      case-folding prose would be a scoring decision made in the wrong module.

    Returns:
        The canonical value, or the original answer when it cannot be canonicalised — so a caller that
        skips :func:`validate` still sees the raw text in its mismatch report.
    """
    if answer_format is AnswerFormat.TRUE_FALSE:
        return answer.strip() if isinstance(answer, str) else answer

    if answer_format is AnswerFormat.MULTIPLE_CHOICE:
        if isinstance(answer, str):
            match = re.match(r"\s*\(?([A-Za-z])\)?", answer)
            if match:
                return match.group(1).upper()
        return answer

    if answer_format is AnswerFormat.NUMERIC:
        number = numeric_value(answer)
        return answer if number is None else number

    if answer_format is AnswerFormat.DESCRIPTION and isinstance(answer, str):
        return answer.strip()

    return answer


def vote_key(answer: Any, answer_format: AnswerFormat) -> str:
    """The key two answers must share to count as the same vote in §A.8's fallback mode.

    §A.8 says the fallback aggregates "via confidence-weighted voting (categorical) or averaging
    (numeric)". Categorical means TF, MC and TD: two descriptions that are not byte-identical are two
    different answers, and near-matching them would decide a semantic question inside an integer counter.
    Numeric questions are not voted on at all — they are averaged — so this returns a stable string key
    only for the categorical formats and lets the caller route numbers elsewhere.

    Case is folded for TF and MC, and **not** for TD. This is the one place folding is wanted without
    being a scoring decision: ``"yes"`` and ``"Yes"`` are one opinion and splitting the vote between them
    would make agreement look like disagreement. Prose is left alone because two descriptions differing
    only in case are, for voting purposes, still the same claim — but the folding rule for prose is
    deliberately not extended to trimming or punctuation, which is where descriptions genuinely diverge.

    Note this differs from :func:`canonicalise`, which preserves TF case so that
    ``evaluation.case_sensitive`` can decide it at scoring time. Two callers, two rules, stated here so the
    difference reads as a decision rather than an inconsistency.
    """
    canonical = canonicalise(answer, answer_format)
    if isinstance(canonical, str):
        if answer_format in {AnswerFormat.TRUE_FALSE, AnswerFormat.MULTIPLE_CHOICE}:
            return canonical.casefold()
        return canonical
    return json.dumps(canonical, sort_keys=True)


def _dedupe(violations: Sequence[AnswerViolation]) -> tuple[AnswerViolation, ...]:
    """First-seen order, no repeats — so a missing field is counted once, not once per check that misses it."""
    seen: list[AnswerViolation] = []
    for violation in violations:
        if violation not in seen:
            seen.append(violation)
    return tuple(seen)
