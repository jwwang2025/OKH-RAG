"""§4.1 — scoring an answer, and the four regimes the paper's single accuracy number hides.

§4.1 gives one sentence to its metrics: questions come in "four types: True/False (TF), Multiple Choice
(MC), Short Answer / Numeric (SA), and Text Description (TD)", and are scored by "exact match" (TF, MC),
"tolerance-based accuracy" (SA) and "an LLM-judged semantic score in [0,1] using GPT-4o as the evaluator"
(TD). The single accuracy in Table 1 is a micro-average over all of them.

Three things follow from taking that sentence literally, and each is a decision this module makes
explicitly rather than by accident:

**1. Exact match needs a definition of "exact".** The release stores MC answers as ``"A."`` and TD answers
as arbitrary prose, so byte equality would score a correct ``"A"`` as wrong. :func:`canonicalise` in
:mod:`~okh_rag.generation.output` is the single definition, shared with the generator's validator so that
"what the model was asked for" and "what scoring accepts" cannot drift.

**2. A tolerance needs a number on both sides — and 39.6% of SA gold answers are not numbers.**
Measured over the release: of **49,459** NU-tagged questions, **19,574 (39.58%)** have a gold answer that
:func:`~okh_rag.generation.output.numeric_value` cannot reduce to a single value. They are not one problem
but seven, and lumping them together would misreport what the SA column means:

====================  ======  ==========================================================
cause                 count   example (template)
====================  ======  ==========================================================
``absent``             8,896  Q14 ``N/A``, ``Not expected``; Q34 ``None``
``date``               4,728  Q34 ``5/4/2015``, ``9/29/2016 0:00``
``coordinates``        2,910  Q6  ``32.9N, 78.3W`` — 100% of Q6
``multi_horizon``      2,908  Q33 ``{'in 12h': '1%', 'in 24h': '3%', ...}`` — 100% of Q33
``range``                 62  Q21 ``8-12``, Q13 ``0-6``
``prose``                 51  Q8 ``0 mph (will increase to ~10 mph)``
``other``                 19  Q11 ``Data not provided for St. Croix``
====================  ======  ==========================================================

The date, coordinate and multi-horizon cases are the release's own format defects against its own ``NU``
tag (``data/CONTRACT.md`` §7 records the Q33 Python-``repr`` defect and the Q36 wrong-field defect).
``absent`` is a *superset* of the 8,337 questions ``configs/base.yaml`` excludes as unanswerable: that flag
counts the two literal placeholders the audit defect names (7,660 ``"None"``, 677 ``""``), while this
bucket also catches ``N/A``, ``Not Applicable``, ``Not expected`` and ``-``, which the release uses for the
same condition without saying so in a form the defect detector keys on. Only the last two rows are
genuinely ambiguous quantities a tolerance test *could* compare if a convention were chosen, and between
them they are 0.2% of SA questions — which is why this module reports them rather than inventing a rule: the
ambiguity is not where the mass is. Only TF, MC and DE are fully scorable (0 unscorable of 5,812 / 37,813 /
24,094). :attr:`AnswerScore.scorable` and :func:`numeric_gold_cause` report the distinction instead of
silently scoring these as wrong, because "the model failed a tolerance test" and "no tolerance test existed"
are different claims about a system.

**3. A micro-average needs a denominator.** §4.1's "Overall" column is not the macro-average of its four
type columns — Table 1's Text-RAG row averages to 0.374 and is reported as 0.287 — so ``overall: "micro"``
is the config default, and both are computed here so the discrepancy is visible rather than assumed away.

The TD judge is implemented behind a protocol rather than called directly. ``configs/base.yaml`` defaults to
``exact_match`` for TD with a measured justification (all 24,094 description answers are byte-identical to
their source field, so a judge can only add sampling variance); ``llm_judge`` is available because §4.1
says it, and a reproduction that omitted it could not show that its ranking survives the judge.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Protocol, Sequence

from okh_rag.config import EvaluationConfig
from okh_rag.data.cyportqa import AnswerFormat, QARecord
from okh_rag.generation.output import canonicalise, numeric_value

__all__ = [
    "AccuracyReport",
    "AnswerScore",
    "JudgeFn",
    "ScoringRegime",
    "numeric_gold_cause",
    "regime_of",
    "score_answer",
    "score_record",
    "summarise",
    "tolerance_for",
]


#: The causes :func:`numeric_gold_cause` can return, in the order a report should print them.
NUMERIC_GOLD_CAUSES = (
    "absent",
    "date",
    "coordinates",
    "multi_horizon",
    "range",
    "prose",
    "other",
)

# A dash used as a range separator. The en/em dashes are not optional: the release mixes ASCII hyphens with
# typographic dashes inside the same template, so matching only ``-`` misses part of the population.
_RANGE_SEPARATOR = re.compile(r"[\u2010-\u2015\-~]")

# ``5/4/2015`` and ``9/29/2016 0:00`` — the two date shapes Q34-Q36 use.
_DATE = re.compile(r"\d{1,2}/\d{1,2}/\d{4}(?: \d{1,2}:\d{2})?")

# ``32.9N, 78.3W`` — Q6's coordinate pair, which the release tags NU.
_COORDINATES = re.compile(r"[\d.]+[NSns]?,\s*[\d.]+[EWew]?")

#: Strings that mean "there is no value", not "the value is zero". ``not expected`` and ``not applicable``
#: are included because Q14 uses both for the same condition.
_ABSENT_TEXTS = frozenset(
    {"", "n/a", "na", "none", "null", "unknown", "not applicable", "not available", "not expected", "-"}
)


def numeric_gold_cause(gold: Any) -> str | None:
    """Why a gold answer cannot be tolerance-scored, or ``None`` when it can.

    Exists so the honest answer to "how many SA questions were actually scorable?" comes with the breakdown
    that makes it interpretable. The counts in the module docstring are produced by calling this over the
    release, so the table cannot drift from the classifier that decides scorable-ness.

    Returns:
        One of :data:`NUMERIC_GOLD_CAUSES`, or ``None`` for a gold that
        :func:`~okh_rag.generation.output.numeric_value` accepts.
    """
    if numeric_value(gold) is not None:
        return None
    if gold is None:
        return "absent"
    if isinstance(gold, dict):
        # Q33 ships a per-horizon mapping rendered as a Python ``repr`` (audit defect
        # ``qa-q33-answer-is-python-repr``); there is no single number to compare against.
        return "multi_horizon"
    if not isinstance(gold, str):
        return "other"

    text = gold.strip()
    if text.casefold() in _ABSENT_TEXTS:
        return "absent"
    if _DATE.fullmatch(text):
        return "date"
    if _COORDINATES.fullmatch(text):
        return "coordinates"
    if text.startswith("{"):
        # The repr arrives as a string when the release's JSON writer stringified a nested object, which is
        # how Q33's per-horizon mapping reaches this module.
        return "multi_horizon"
    if "(" in text:
        return "prose"
    numbers = re.findall(r"[\d.]+", text)
    if _RANGE_SEPARATOR.search(text) and len(numbers) == 2:
        return "range"
    return "other"


class ScoringRegime(str, Enum):
    """Which of §4.1's three scoring rules applies to a question."""

    EXACT = "exact_match"  # TF, MC
    TOLERANCE = "tolerance"  # SA
    SEMANTIC = "semantic"  # TD


_REGIME_BY_FORMAT: Mapping[AnswerFormat, ScoringRegime] = {
    AnswerFormat.TRUE_FALSE: ScoringRegime.EXACT,
    AnswerFormat.MULTIPLE_CHOICE: ScoringRegime.EXACT,
    AnswerFormat.NUMERIC: ScoringRegime.TOLERANCE,
    AnswerFormat.DESCRIPTION: ScoringRegime.SEMANTIC,
}


def regime_of(answer_format: AnswerFormat) -> ScoringRegime:
    """§4.1's scoring rule for one answer format."""
    return _REGIME_BY_FORMAT[answer_format]


class JudgeFn(Protocol):
    """§4.1's "LLM-judged semantic score in [0, 1]".

    A protocol rather than a concrete call so that the judge can be swapped without touching the scorer,
    and so that a test can assert the ranking is unchanged under a judge that always returns the exact-match
    score. The released ``evaluators/prompt_*.txt`` rubric is an ordinal scale
    (``0.0 / 0.2 / 0.4 / 0.6 / 0.8 / 1.0``) with a hallucination rule that forces 0.0 when invented data
    appears — those files are not vendored (``data/CONTRACT.md`` §2 lists what is), so an implementation
    supplied by the caller is the only honest option.
    """

    def __call__(self, question: str, gold: str, predicted: str) -> float:
        """Return the semantic score of ``predicted`` against ``gold`` for ``question``."""
        ...


@dataclass(frozen=True)
class AnswerScore:
    """One question's score, with enough detail to explain a failure.

    Attributes:
        correct: The score in ``[0, 1]``, or ``None`` when the question is **not scorable** — the gold
            answer is not of the shape the regime can compare (see the module docstring) or the answer was
            excluded as unanswerable. ``None`` rather than ``0.0``: averaging an unscorable question in
            counts it as a failure, which is a claim about the system that the data does not support.
        regime: Which of §4.1's three rules was applied.
        matched: Whether the comparison succeeded. Always ``False`` when :attr:`correct` is ``None``, so a
            caller that ignores the distinction still under-reports rather than over-reports.
        reason: A short machine-readable cause — ``"match"``, ``"mismatch"``, ``"not_numeric_gold"``,
            ``"unanswerable"``, ``"no_judge"``. Aggregated into counts by :class:`AccuracyReport`.
    """

    correct: float | None
    regime: ScoringRegime
    matched: bool = False
    reason: str = ""
    detail: str = ""

    @property
    def scorable(self) -> bool:
        """Whether a score exists. The denominator a report should use."""
        return self.correct is not None


# ---------------------------------------------------------------------------
# Scoring one answer
# ---------------------------------------------------------------------------


def score_answer(
    predicted: Any,
    gold: Any,
    answer_format: AnswerFormat,
    cfg: EvaluationConfig,
    *,
    judge: JudgeFn | None = None,
    question: str = "",
) -> AnswerScore:
    """Score one predicted answer against one gold answer under §4.1's rule for its format.

    Args:
        predicted: The model's answer, canonicalised here.
        gold: The release's answer.
        answer_format: Decides the regime.
        cfg: The ``evaluation`` section — tolerances, case sensitivity, the TD scoring mode.
        judge: Required only when ``cfg.td_scoring == "llm_judge"`` and the format is TD. Absent judge
            yields an unscorable result rather than a guessed score; a run that asked for a judge and did
            not supply one has a configuration error, and 0.0 would hide it inside the accuracy.
        question: Passed to the judge; unused by the other regimes.

    Returns:
        The :class:`AnswerScore`. Never raises: every failure mode is a reason string, because a run over
        117,178 questions cannot afford to abort on the one question whose gold is malformed.
    """
    canonical_predicted = canonicalise(predicted, answer_format)
    canonical_gold = canonicalise(gold, answer_format)
    regime = regime_of(answer_format)

    if regime is ScoringRegime.EXACT:
        if not isinstance(canonical_gold, str):
            return AnswerScore(None, regime, reason="gold_not_a_string")
        matched = _strings_equal(
            _as_text(canonical_predicted), canonical_gold, case_sensitive=cfg.case_sensitive
        )
        return AnswerScore(
            1.0 if matched else 0.0,
            regime,
            matched=matched,
            reason="match" if matched else "mismatch",
            detail="" if matched else f"predicted={predicted!r} gold={gold!r}",
        )

    if regime is ScoringRegime.TOLERANCE:
        target = canonical_gold if isinstance(canonical_gold, (int, float)) else numeric_value(gold)
        if target is None:
            cause = numeric_gold_cause(gold) or "other"
            return AnswerScore(
                None,
                regime,
                reason="not_numeric_gold",
                detail=f"gold={gold!r} is not a single numeric value (cause={cause})",
            )
        guess = canonical_predicted if isinstance(canonical_predicted, (int, float)) else None
        if guess is None:
            return AnswerScore(
                0.0,
                regime,
                reason="mismatch",
                detail=f"predicted={predicted!r} is not numeric; gold={float(target)}",
            )
        tolerance = max(cfg.sa_relative_tolerance * abs(float(target)), cfg.sa_absolute_floor)
        matched = abs(float(guess) - float(target)) <= tolerance
        return AnswerScore(
            1.0 if matched else 0.0,
            regime,
            matched=matched,
            reason="match" if matched else "mismatch",
            detail=f"predicted={float(guess)} gold={float(target)} tolerance={tolerance:g}",
        )

    # TD. §4.1 scores it semantically; the release makes exact match sufficient (see the module docstring).
    text = _as_text(canonical_predicted)
    reference = _as_text(canonical_gold)
    if cfg.td_scoring == "exact_match":
        matched = _normalise_prose(text) == _normalise_prose(reference)
        return AnswerScore(
            1.0 if matched else 0.0,
            regime,
            matched=matched,
            reason="match" if matched else "mismatch",
            detail="" if matched else f"predicted={text[:80]!r} gold={reference[:80]!r}",
        )

    if judge is None:
        return AnswerScore(None, regime, reason="no_judge")
    try:
        value = float(judge(question, reference, text))
    except Exception as error:  # noqa: BLE001 - a judge failure must not abort a 117k-question run
        return AnswerScore(None, regime, reason="judge_error", detail=str(error))
    value = min(1.0, max(0.0, value))
    return AnswerScore(
        value,
        regime,
        matched=value >= 1.0,
        reason="match" if value >= 1.0 else "partial",
        detail=f"judge={value:.2f}",
    )


def score_record(
    record: QARecord,
    predicted: Any,
    cfg: EvaluationConfig,
    *,
    judge: JudgeFn | None = None,
) -> AnswerScore:
    """Score one question's answer, applying the unanswerable exclusion.

    ``cfg.exclude_unanswerable`` drops the 8,337 questions whose gold answer is a placeholder the encoding
    cannot supply (7,660 ``"None"``, 677 ``""``) — see ``data/CONTRACT.md`` §5.2. They are unscorable, not
    wrong: including them measures the release's own gap. The exclusion is applied *here*, at the moment of
    scoring, rather than by filtering the record list beforehand, so every report can quote the count of
    what was excluded instead of having it disappear upstream.
    """
    if cfg.exclude_unanswerable and not record.is_answerable:
        return AnswerScore(
            None,
            regime_of(record.answer_format),
            reason="unanswerable",
            detail=repr(record.answer)[:60],
        )
    return score_answer(
        predicted,
        record.answer,
        record.answer_format,
        cfg,
        judge=judge,
        question=record.question,
    )


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _as_text(value: Any) -> str:
    """A string for comparison. ``None`` becomes ``""`` — an absent answer, not the four-letter word."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        # A tolerance-scored gold of ``24`` canonicalises to ``24.0``; ``str`` would make it ``"24.0"`` and
        # an exact-match comparison against ``"24"`` would fail. This branch keeps the two agreeing when a
        # numeric answer is compared textually.
        return str(int(value))
    return str(value)


def _strings_equal(predicted: str, gold: str, *, case_sensitive: bool) -> bool:
    """Exact match under the config's case rule, with surrounding whitespace ignored.

    Whitespace is always stripped. §4.1's "exact match" is a claim about the answer, and no answer's meaning
    depends on leading or trailing spaces; treating a trailing space as a wrong answer would be a strictness
    the paper does not claim.

    Note the asymmetry with MC: canonicalisation has already reduced ``"A."`` to ``"A"``, so this compares
    bare labels. Doing it the other way round — requiring the model to reproduce the release's trailing
    period — would score correct answers wrong for a publisher's punctuation habit.
    """
    left = predicted.strip()
    right = gold.strip()
    return left == right if case_sensitive else left.casefold() == right.casefold()


_WHITESPACE = re.compile(r"\s+")


def _normalise_prose(text: str) -> str:
    """Whitespace-collapsed, case-folded prose for TD's exact-match mode.

    Case and internal whitespace are folded because a description answer copied from a NOAA product can
    differ from the gold only in line wrapping, and §4.1's semantic intent for TD is plainly not "reproduce
    the source's line breaks".
    """
    return _WHITESPACE.sub(" ", text.strip()).casefold()


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


@dataclass
class AccuracyReport:
    """§4.1's accuracies, computed both ways, with the counts that make them interpretable.

    Every accuracy here is reported alongside the number of questions it divides by. That is not
    decoration: three separate exclusions (unanswerable, non-numeric gold, absent judge) shrink the
    denominator, and an accuracy whose denominator is invisible is the easiest way to publish a number
    that cannot be reproduced.
    """

    scores: list[AnswerScore] = field(default_factory=list)
    formats: list[AnswerFormat] = field(default_factory=list)

    def add(self, score: AnswerScore, answer_format: AnswerFormat) -> None:
        """Record one scored question. The two lists stay index-aligned."""
        self.scores.append(score)
        self.formats.append(answer_format)

    def __len__(self) -> int:
        return len(self.scores)

    # -- per-regime ---------------------------------------------------------

    def accuracy(self, answer_format: AnswerFormat) -> float | None:
        """Mean score over the scorable questions of one format, or ``None`` when there are none.

        ``None`` rather than ``0.0`` for an empty bucket: Table 1 has a column per format, and a format with
        no scorable question has no accuracy — printing ``0.0`` would read as total failure.
        """
        values = [
            score.correct
            for score, fmt in zip(self.scores, self.formats, strict=True)
            if fmt is answer_format and score.correct is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    def counts(self, answer_format: AnswerFormat) -> tuple[int, int]:
        """``(scorable, total)`` for one format."""
        total = sum(1 for fmt in self.formats if fmt is answer_format)
        scorable = sum(
            1
            for score, fmt in zip(self.scores, self.formats, strict=True)
            if fmt is answer_format and score.correct is not None
        )
        return scorable, total

    def by_format(self) -> dict[str, float | None]:
        """§4.1's four columns, keyed by format tag (``TF``/``MC``/``NU``/``DE``)."""
        return {fmt.value: self.accuracy(fmt) for fmt in AnswerFormat}

    # -- overall ------------------------------------------------------------

    def micro(self) -> float | None:
        """The micro-average: mean over every scorable question regardless of format (Table 1's "Overall")."""
        values = [score.correct for score in self.scores if score.correct is not None]
        if not values:
            return None
        return sum(values) / len(values)

    def macro(self) -> float | None:
        """The macro-average: unweighted mean of the four per-format accuracies.

        Computed and reported next to :meth:`micro` because §4.1's Table 1 "Overall" is *not* the
        macro-average of its own columns (the Text-RAG row averages to 0.374, the table reports 0.287), and
        a reproduction that silently picks one cannot be compared against the paper.
        """
        present = [value for value in self.by_format().values() if value is not None]
        if not present:
            return None
        return sum(present) / len(present)

    def overall(self, mode: str = "micro") -> float | None:
        """The overall accuracy under ``configs/base.yaml``'s ``evaluation.overall``."""
        if mode == "micro":
            return self.micro()
        if mode == "macro":
            return self.macro()
        raise ValueError(f"unknown overall mode {mode!r}; expected 'micro' or 'macro'")

    # -- diagnostics --------------------------------------------------------

    def reason_counts(self) -> dict[str, int]:
        """How many questions ended with each reason — the table that explains a surprising accuracy."""
        counts: dict[str, int] = {}
        for score in self.scores:
            counts[score.reason] = counts.get(score.reason, 0) + 1
        return dict(sorted(counts.items()))

    def exclusion_summary(self) -> dict[str, int]:
        """The scorable/unscorable split, with unscorable questions broken down by cause.

        This is the number a reader needs to judge whether an accuracy is comparable to another run's: two
        runs can differ in accuracy only because they excluded different questions.
        """
        scorable = sum(1 for score in self.scores if score.correct is not None)
        summary = {"total": len(self.scores), "scorable": scorable, "unscorable": len(self.scores) - scorable}
        for reason, count in self.reason_counts().items():
            if reason in {"unanswerable", "not_numeric_gold", "no_judge", "judge_error", "gold_not_a_string"}:
                summary[reason] = count
        return summary

    def as_dict(self, *, overall_mode: str = "micro") -> dict[str, Any]:
        """Plain-dict view for reports, with every accuracy next to its denominator.

        ``counts`` is emitted as ``{"format": [scorable, total]}`` with lists rather than tuples: this dict
        is written to JSON, and a tuple that round-trips to a list would make two runs' reports differ in
        type for no reason.
        """
        return {
            "by_format": self.by_format(),
            "counts": {fmt.value: list(self.counts(fmt)) for fmt in AnswerFormat},
            "micro": self.micro(),
            "macro": self.macro(),
            "overall": self.overall(overall_mode),
            "overall_mode": overall_mode,
            "reasons": self.reason_counts(),
            "exclusions": self.exclusion_summary(),
        }


def summarise(scores: Iterable[tuple[AnswerFormat, AnswerScore]]) -> AccuracyReport:
    """Build an :class:`AccuracyReport` from ``(format, score)`` pairs.

    A convenience so that a driver streaming results does not have to keep two parallel lists in step by
    hand — the failure that would silently mis-attribute every accuracy to the wrong column.
    """
    report = AccuracyReport()
    for answer_format, score in scores:
        report.add(score, answer_format)
    return report


def tolerance_for(gold: Any, cfg: EvaluationConfig) -> float | None:
    """The absolute tolerance §4.1's rule produces for one gold value, or ``None`` when it is not numeric.

    Exposed because the tolerance is the one part of SA scoring that a reader will want to check by hand,
    and recomputing ``max(rel * |gold|, floor)`` from a report is exactly the kind of duplicated arithmetic
    that drifts from the scorer.
    """
    target = numeric_value(gold)
    if target is None:
        return None
    return max(cfg.sa_relative_tolerance * abs(target), cfg.sa_absolute_floor)


def geometric_mean(values: Sequence[float]) -> float:
    """Geometric mean, used for no §4.1 metric but needed by ablation ratios.

    Present here rather than in a utility module so that the evaluation package has one place where numbers
    are computed, and ``math`` is imported once.
    """
    if not values:
        raise ValueError("geometric mean of an empty sequence is undefined")
    if any(value <= 0 for value in values):
        raise ValueError("geometric mean requires strictly positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))
