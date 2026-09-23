"""§A.2 + §4.1 — the closed-form ground truth the release ships but the paper never mentions.

``CyPortQA_template.json`` names, for each of the 48 question templates, the dotted path in
``Encoded_senario.json`` that produces its answer:

    "answer": "{senario.time.<key>.weather_observation.within_uncertainty_cone}"

Substituting ``<key>`` with the record's horizon turns that into a field accessor, and
substituting the ``operation.<key>`` form with the record's ``in_XXh`` tag resolves the
four templates (Q42, Q46-Q48) that interrogate the *future* bulletin ladder.

Two things this buys the reproduction
-------------------------------------
1. **A retrieval oracle.** §A.2 extracts a hypergraph from the scenario encoding, and §3.2
   retrieves trajectories from it. Because a question's answer is a named field, the set of
   hyperedges a question *needs* is the set built from that field — so retrieval can be
   scored on whether it recovered them, independently of whether the generator then answers
   correctly. The paper reports only end-to-end accuracy, which cannot separate the two.
2. **A label audit.** Comparing the shipped answer against the value its own template names
   is what surfaces the ``template-q36-answer-field-mismatch`` defect. That comparison is
   not a nicety: without it, a reproduction would inherit a wrong ground truth and report
   it as a method failure.

Nothing here calls a model. ``Resolve`` is pure field access and ``compare`` is string and
number handling, so both run offline and deterministically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Sequence

from okh_rag.data.cyportqa import (
    PORT_CONDITION_ALIASES,
    AnswerFormat,
    EncodedScenario,
    QARecord,
    Template,
    TemplateSemantics,
    parse_inline_options,
    resolve_mc_letter,
)

MISSING: Final[Any] = object()
"""Returned by :func:`resolve_path` when the encoding lacks the field.

A sentinel rather than ``None`` because ``None`` is itself a value the release can store,
and the two cases must not be conflated: an absent field makes a question unanswerable,
a stored ``None`` makes it answerable-but-degenerate.
"""


class VerdictStatus(str, Enum):
    """Outcome of comparing a shipped answer against its template-declared ground truth."""

    MATCH = "match"
    MISMATCH = "mismatch"
    MISSING_FIELD = "missing_field"  # the template's path is absent from the encoding
    INDETERMINATE = "indeterminate"  # the field is present but its value cannot yield an answer
    UNANSWERABLE = "unanswerable"  # the shipped answer is a placeholder ("None" / "")
    UNVERIFIABLE = "unverifiable"  # free text: needs the §4.1 GPT-4o judge, not offline comparison


@dataclass(frozen=True)
class Verdict:
    """One comparison, with enough detail to be argued with."""

    status: VerdictStatus
    shipped: str
    expected: str
    path: str
    detail: str = ""
    ambiguous: bool = False
    """True when the encoding's value lands in more than one of the question's buckets, so
    no model could determine the answer from the field alone. A match is still a match, but
    the question is not a fair measure of reasoning and the audit counts these separately."""

    @property
    def agrees(self) -> bool:
        return self.status is VerdictStatus.MATCH


_DASHES: Final[tuple[str, ...]] = ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015")
_QUOTES: Final[tuple[str, ...]] = ("\u2018", "\u2019", "\u201c", "\u201d")
_STRIP: Final[str] = " \t\r\n.,;:!?\"'()[]{}"


def normalise(text: Any) -> str:
    """Fold a value to a comparison-stable string.

    The release's free-text fields differ from each other only in punctuation, dash flavour
    and spacing ("Yes - on the southwestern edge of the cone" vs "Yes (southern fringe)"),
    and its numeric fields differ only in trailing zeros, so a normalisation that strips
    exactly those is needed before any equality test means anything. It deliberately does
    *not* stem, drop parentheticals or apply synonym tables — those changes would make the
    audit report agreement that is not there. §A.2's alias machinery belongs in the
    hypergraph builder, not in the yardstick.
    """
    if text is None:
        return ""
    if isinstance(text, bool):
        return "yes" if text else "no"
    if isinstance(text, (int, float)):
        # 2.0 and 2 must compare equal; 1e-06-equivalent float noise must not create a diff.
        return f"{float(text):g}"
    out = str(text)
    for dash in _DASHES:
        out = out.replace(dash, "-")
    for quote in _QUOTES:
        out = out.replace(quote, "'")
    out = re.sub(r"\s+", " ", out)
    return out.strip().strip(_STRIP).lower()


_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

_CLAUSE_SPLIT_RE: Final[re.Pattern[str]] = re.compile(r"[\(\[]|;|:|\s-\s")


def leading_clause(text: Any) -> str:
    """A value reduced to its head clause, dropping parentheticals and qualifiers.

    The release's categorical fields carry trailing qualifications that the question
    generator drops when it selects an option: ``"Unlikely (forecast 1-3 ft)"`` becomes the
    option ``Unlikely``. Comparing the full strings would call that a mismatch on all 200-odd
    such records, so the head clause is compared as well.

    Splitting is on ``(``, ``[``, ``;``, ``:`` and a *spaced* hyphen only. That last
    restriction matters: ``"X-RAY"`` is a port condition and must not be truncated to
    ``"x"``, whereas ``"Unlikely - forecast 1-3 ft"`` should be truncated to ``"unlikely"``.
    """
    return _CLAUSE_SPLIT_RE.split(normalise(text))[0].strip()


def _is_word_prefix(shorter: str, longer: str) -> bool:
    """True when ``shorter`` prefixes ``longer`` on a word boundary."""
    if not shorter or not longer or not longer.startswith(shorter):
        return False
    return len(longer) == len(shorter) or not longer[len(shorter)].isalnum()


def _categorical_match(chosen_variants: set[str], wanted_variants: set[str]) -> bool:
    """Test a multiple-choice answer against a ground-truth value.

    Two tests, in order of strength:

    1. **equality** after normalisation — the clean case;
    2. **word-boundary prefix** in either direction — needed because the question generator
       canonicalises a verbose field value to the option that matches it, and the option is
       the shorter of the two. ``"Unlikely at this time"`` becomes the option ``Unlikely``,
       ``"None expected at port"`` becomes ``None``, ``"Category 2 Hurricane"`` becomes
       ``Category 2``. A *word-boundary* prefix is required rather than a bare one so that
       ``"No"`` does not match ``"None expected"``.

    A prefix match is weaker evidence than equality, but the alternative — calling several
    hundred correct answers wrong — would be worse, and the residual after both tests is
    reported as label noise rather than hidden.
    """
    if chosen_variants & wanted_variants:
        return True
    return any(
        _is_word_prefix(chosen, wanted) or _is_word_prefix(wanted, chosen)
        for chosen in chosen_variants
        for wanted in wanted_variants
    )


def _extract_number(text: str) -> float | None:
    """First number in ``text``, or ``None``.

    The answer prompts say "Answer with ONLY a number" but the shipped strings carry units
    ("6 mph", "5%"), so a bare ``float()`` would reject answers that are in fact correct.
    Percentages are *not* rescaled: the encoded field holds "5%" and the answer holds "5%",
    so both sides stay on the same footing.
    """
    match = _NUMBER_RE.search(text.replace(",", ""))
    return float(match.group(0)) if match else None


_YES_NO_RE: Final[re.Pattern[str]] = re.compile(r"^(yes|no|y|n|true|false)\b")


def _polarity(text: str) -> str | None:
    """``"Yes (southern fringe)"`` → ``"yes"``. ``None`` when the text is not a polarity.

    Necessary because a template's expected value can be a qualified yes while the shipped
    answer is the bare one, and treating those as a mismatch would manufacture defects.
    """
    match = _YES_NO_RE.match(normalise(text))
    if match is None:
        return None
    token = match.group(1)
    return "yes" if token in {"yes", "y", "true"} else "no"


#: The SSHS ladder plus the pre-storm rungs the release uses, ordered longest-first so that
#: "Potential Tropical Cyclone" is preferred over "Tropical Cyclone" and "Tropical Storm" over
#: "Tropical". Used only to *find* rungs; the order of preference is decided by position, below.
SSH_LADDER: Final[tuple[str, ...]] = (
    "Potential Tropical Cyclone",
    "Tropical Disturbance",
    "Tropical Depression",
    "Subtropical Storm",
    "Tropical Storm",
    "Hurricane",
    "Remnants",
)

#: The explicit Saffir-Simpson rungs, matched separately because they are the more specific
#: signal when they appear in the head of the string.
CATEGORY_RUNGS: Final[tuple[str, ...]] = (
    "Category 5",
    "Category 4",
    "Category 3",
    "Category 2",
    "Category 1",
)

#: The one ladder rung that is a generic noun rather than a distinct intensity: everything else
#: in :data:`SSH_LADDER` already implies it. "Hurricane" alone is a head clause that the
#: qualifier refines, so a category found later in the string applies to it.
GENERIC_RUNG: Final[str] = "Hurricane"


def normalise_ssh_category(raw: Any) -> str | None:
    """Reduce a raw ``cyclone_condition_and_forecast.category`` to a canonical rung.

    The release ships 130+ distinct strings for what is really one of a dozen rungs, e.g.
    ``"Category 2 (105-110 mph, forecast to reach major-hurricane strength)"`` and
    ``"Hurricane (Category 1 now - forecast to strengthen)"``. This is the normalisation
    §A.2 step 1's alias table is meant to absorb, and it is needed here because Q18's
    expected value is that raw string while its options are the clean ladder.

    The head clause decides, with one exception. Naively scanning the whole string for a rung
    would read ``"Tropical Storm (below SSHS Category 1)"`` as *Category 1* — the exact
    opposite of what it says — so a ``Category N`` inside a parenthetical is ignored unless
    the head clause is the bare generic ``"Hurricane"``, which is the case the parenthetical
    exists to refine (``"Hurricane (Category 1 now …)"``).

    Returns:
        The rung, or ``None`` when no rung is recognisable — which is itself a finding, since
        it means the string needs a hand-written alias rather than a rule. The audit counts
        these.
    """
    text = normalise(raw)
    if not text:
        return None

    head = leading_clause(text)

    # 1. An explicit rung in the head clause wins outright.
    for rung in CATEGORY_RUNGS:
        if rung.lower() in head:
            return rung

    # 2. Otherwise the earliest ladder rung named in the head, longest match at equal position.
    #    Earliest rather than longest because the release stacks rungs — "Tropical Depression /
    #    Potential Tropical Cyclone" — and the leading one is the current intensity.
    best: tuple[int, int, str] | None = None
    for rung in SSH_LADDER:
        position = head.find(rung.lower())
        if position < 0:
            continue
        candidate = (position, -len(rung), rung)
        if best is None or candidate < best:
            best = candidate
    if best is not None and best[2] != GENERIC_RUNG:
        return best[2]

    # 3. The head is the bare generic "Hurricane" (or names no rung at all): a rung stated
    #    anywhere in the string is a refinement of it.
    for rung in CATEGORY_RUNGS:
        if rung.lower() in text:
            return rung
    if best is not None:
        return best[2]

    for rung in SSH_LADDER:
        if rung.lower() in text:
            return rung
    return None


def resolve_path(scenario: EncodedScenario, horizon_hours: int, path: str) -> Any:
    """Resolve a dotted path from a template's answer placeholder.

    Handles the three shapes the release uses:

    * ``port.*`` / ``cyclone.*`` — scenario-level fields (Q16 and Q19 answer
      ``{senario.port.name}``, which is *not* horizon-dependent);
    * ``operation.now.*`` / ``operation.in_12h.*`` — the relative bulletin ladder;
    * ``<block>.<field>[.<subfield>]`` — a field of one of the four horizon blocks.

    Args:
        scenario: the encoding to read.
        horizon_hours: the record's absolute lead time.
        path: a dotted path such as ``weather_observation.under_watch``.

    Returns:
        The value, or :data:`MISSING` when any segment is absent.
    """
    parts = path.split(".")
    if not parts or not parts[0]:
        return MISSING

    head = parts[0]
    if head in {"port", "cyclone"}:
        node: Any = getattr(scenario, head)
        return _descend(node, parts[1:])

    block = scenario.time.get(horizon_hours)
    if block is None:
        return MISSING
    if head not in block.blocks():
        return MISSING
    return _descend(block.blocks()[head], parts[1:])


def _descend(node: Any, parts: list[str]) -> Any:
    """Walk ``parts`` into nested mappings, returning :data:`MISSING` on any miss."""
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return MISSING
    return node


def expected_path(record: QARecord, template: Template) -> str:
    """The concrete dotted path a record's answer should come from.

    Substitutes ``operation.<key>`` with the record's relative-horizon tag. When the
    template needs that substitution but the record does not carry the tag, the raw
    ``<key>`` is left in place so :func:`resolve_path` reports
    :data:`VerdictStatus.MISSING_FIELD` rather than silently reading the wrong sub-key.
    """
    path = template.static_prefix
    if "<key>" in path:
        if record.relative_horizon is None:
            return path
        path = path.replace("<key>", record.relative_horizon)
    return path


def expected_answer(record: QARecord, template: Template, scenario: EncodedScenario) -> tuple[Any, str]:
    """``(value, path)`` for a record — the ground truth its template declares."""
    path = expected_path(record, template)
    return resolve_path(scenario, record.context.horizon_hours, path), path


# --------------------------------------------------------------------------------------
# Ordinal buckets and threshold predicates
# --------------------------------------------------------------------------------------

_RANGE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?P<lo>-?\d+(?:\.\d+)?)\s*(?:-|\u2013|\u2014|\bto\b)\s*(?P<hi>-?\d+(?:\.\d+)?)"
)
_OPEN_TAIL_RE: Final[re.Pattern[str]] = re.compile(
    r"(?P<lo>-?\d+(?:\.\d+)?)\s*(?:\w+\s*)?(?:or\s+(?:more|above|greater|higher)|and\s+(?:above|over|up))",
    re.IGNORECASE,
)


def parse_ordinal(text: Any) -> tuple[float, float] | None:
    """Read an option's text as a closed interval ``[lo, hi]``, with ``hi = inf`` for an open tail.

    Handles the three shapes the release's bucket options use::

        "3 days"            -> (3.0, 3.0)
        "1-12 Hours"        -> (1.0, 12.0)
        "24 hours or more"  -> (24.0, inf)
        "None"              -> (0.0, 0.0)      # Q40's zero-vessel bucket

    Returns ``None`` when the text is not a bucket at all, which is how a genuine
    categorical option (``"Tropical Storm Watch"``) is told apart from an ordinal one.

    Note the release keeps both sides of every bucket comparison in the same unit — Q23's
    options are in days and its ``forecast_window`` field is in days — so no unit
    conversion is attempted here. If that ever stopped holding, this function would need to
    return the unit alongside the interval rather than silently mis-comparing.
    """
    if text is None:
        return None
    raw = str(text)
    # Only "None" denotes a zero-count bucket (Q40's vessel options). "No" must NOT be treated
    # as one: Q30 and Q31 offer "No" as a genuine categorical option alongside "Unlikely",
    # "Yes" and "Possible", and reading it as the scalar 0 would make those templates look like
    # bucket templates and silently mis-score them.
    if normalise(raw) == "none":
        return (0.0, 0.0)

    match = _RANGE_RE.search(raw)
    if match is not None:
        lo, hi = float(match.group("lo")), float(match.group("hi"))
        return (min(lo, hi), max(lo, hi))

    match = _OPEN_TAIL_RE.search(raw)
    if match is not None:
        return (float(match.group("lo")), float("inf"))

    number = _extract_number(raw)
    if number is None:
        return None
    return (number, number)


def bucket_letters(value: float, options: Sequence[tuple[str, str]]) -> tuple[list[str], int]:
    """Which option letters admit ``value``, and how many ordinal options there were.

    The release's buckets overlap: Q25 offers ``0-12``, ``12-24``, ``24-48``, ``48 hours or
    more``, so 12, 24 and 48 each fall in two of them. Every option containing the value is
    accepted, because the ambiguity is the dataset's and not the answerer's — but the
    caller is told how many matched so the ambiguity can be counted rather than averaged
    away.

    Returns:
        ``(letters, n_ordinal_options)``. A caller can tell "this template is not a bucket
        template at all" (``n_ordinal_options == 0``) from "this value lies in several
        buckets" (``len(letters) > 1``).
    """
    letters: list[str] = []
    ordinal = 0
    for letter, text in options:
        interval = parse_ordinal(text)
        if interval is None:
            continue
        ordinal += 1
        lo, hi = interval
        if lo - 1e-9 <= value <= hi + 1e-9:
            letters.append(letter)
    return letters, ordinal


_THRESHOLD_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:reach\s+or\s+exceed|exceed|at\s+least|no\s+less\s+than|above|over|more\s+than|greater\s+than|"
    r"reach|be\s+at\s+least|equal\s+or\s+greater\s+than)\s+(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>%|percent|kt|mph|ft|feet|inch(?:es)?|hours?|days?)?",
    re.IGNORECASE,
)
_INCLUSIVE_RE: Final[re.Pattern[str]] = re.compile(
    r"reach\s+or\s+exceed|at\s+least|no\s+less\s+than|or\s+more|equal\s+or\s+greater", re.IGNORECASE
)


def parse_threshold(question: str) -> tuple[str, float] | None:
    """Extract the comparison a yes/no question asks about.

    Q26's ``"Will the probability of 34-kt winds [...] reach or exceed 20% [...]?"`` yields
    ``(">=", 20.0)``. The threshold is part of the *question*, not the encoding, which is
    why a threshold-predicate template cannot be scored by comparing strings: its declared
    field (``34kt_max`` = ``"11%"``) is the operand, and the answer is the comparison.

    Returns ``None`` when the question states no numeric threshold, in which case the
    caller must fall back to a polarity comparison.
    """
    match = _THRESHOLD_RE.search(question)
    if match is None:
        return None
    operator = ">=" if _INCLUSIVE_RE.search(match.group(0)) else ">"
    return operator, float(match.group("value"))


def evaluate_predicate(operator: str, threshold: float, value: float) -> bool:
    """Apply a :func:`parse_threshold` operator."""
    return value >= threshold if operator == ">=" else value > threshold


def _compare_bucket(record: QARecord, shipped: str, expected: Any, path: str) -> Verdict:
    """Score an ordinal-bucket question: place the field's scalar into the question's options.

    Both the shipped letter and the field are reduced to a *set* of admissible letters, and
    the question counts as answered when the shipped letter is among them. When it is not,
    the verdict names the admissible set so the disagreement is inspectable.
    """
    options = record.options or parse_inline_options(record.question)
    if not options:
        return Verdict(VerdictStatus.INDETERMINATE, shipped, _render(expected), path, "question carries no options")

    value = _extract_number(_render(expected))
    if value is None:
        return Verdict(
            VerdictStatus.INDETERMINATE,
            shipped,
            _render(expected),
            path,
            f"field value {expected!r} is not numeric, so no bucket can be selected "
            "(the release stores 'N/A' and '' here for a minority of horizons)",
        )

    letters, ordinal = bucket_letters(value, options)
    if ordinal == 0:
        return Verdict(
            VerdictStatus.INDETERMINATE,
            shipped,
            _render(expected),
            path,
            "no option is an ordinal bucket, so the template is mis-classified as one",
        )

    chosen = resolve_mc_letter(shipped, options)
    if chosen is None:
        return Verdict(
            VerdictStatus.MISMATCH,
            shipped,
            _render(expected),
            path,
            f"answer {shipped!r} is not one of the option letters {[letter for letter, _ in options]}",
        )

    shipped_letter = shipped.strip().rstrip(".").strip().upper()
    if shipped_letter in letters:
        detail = ""
        if len(letters) > 1:
            detail = (
                f"value {value:g} lies in buckets {letters}; the shipped answer is one of them, "
                "but the question is not decidable from the field alone"
            )
        return Verdict(VerdictStatus.MATCH, chosen, _render(expected), path, detail, ambiguous=len(letters) > 1)

    return Verdict(
        VerdictStatus.MISMATCH,
        chosen,
        _render(expected),
        path,
        f"value {value:g} falls in buckets {letters}, not in {shipped_letter}",
        ambiguous=len(letters) > 1,
    )


def _compare_predicate(record: QARecord, shipped: str, expected: Any, path: str) -> Verdict:
    """Score a threshold-predicate question by evaluating the predicate in the question text."""
    parsed = parse_threshold(record.question)
    if parsed is None:
        return Verdict(
            VerdictStatus.INDETERMINATE,
            shipped,
            _render(expected),
            path,
            "the question states no numeric threshold, so the declared field cannot yield a Yes/No",
        )
    operator, threshold = parsed

    value = _extract_number(_render(expected))
    if value is None:
        return Verdict(
            VerdictStatus.INDETERMINATE,
            shipped,
            _render(expected),
            path,
            f"field value {expected!r} is not numeric, so the predicate is unevaluable",
        )

    want = "yes" if evaluate_predicate(operator, threshold, value) else "no"
    got = _polarity(shipped)
    if got is None:
        return Verdict(VerdictStatus.MISMATCH, shipped, want, path, "shipped answer is not Yes/No")
    detail = f"{value:g} {operator} {threshold:g} -> {want}"
    return Verdict(VerdictStatus.MATCH if got == want else VerdictStatus.MISMATCH, shipped, want, path, detail)


def compare(
    record: QARecord,
    expected: Any,
    path: str,
    *,
    semantics: TemplateSemantics = TemplateSemantics.LOOKUP,
    relative_tolerance: float = 1e-2,
    absolute_floor: float = 1e-6,
) -> Verdict:
    """Test a shipped answer against its template-declared ground truth.

    Dispatch happens on ``semantics`` first, because for a bucket or predicate template the
    declared field is not the answer and a format-based comparison would be comparing the
    wrong things. Within :attr:`TemplateSemantics.LOOKUP` the comparison follows §4.1's
    four evaluation regimes: exact match for TF and MC, tolerance-based accuracy for
    numerics, and an LLM judge for text — which means a description can only ever come back
    :data:`VerdictStatus.UNVERIFIABLE` from here, and saying so is the point.

    Args:
        record: the QA record, whose ``answer`` is the shipped value.
        expected: the resolved ground-truth value, or :data:`MISSING`.
        path: the dotted path ``expected`` came from, echoed into the verdict.
        semantics: how the template's field becomes an answer; see
            :class:`~okh_rag.data.cyportqa.TemplateSemantics`.
        relative_tolerance: §4.1 says "tolerance-based accuracy" without a value; the
            default matches ``configs/base.yaml:evaluation.sa_relative_tolerance``.
        absolute_floor: guards the near-zero case, where a relative tolerance would demand
            exact equality.

    Returns:
        A :class:`Verdict`. ``MISSING_FIELD`` outranks ``UNANSWERABLE``: when both the
        encoding lacks the field *and* the shipped answer is a placeholder, the absence of
        the field is the more informative fact.
    """
    shipped = record.answer_text

    if expected is MISSING:
        return Verdict(VerdictStatus.MISSING_FIELD, shipped, "", path, "the encoding has no such field")
    if not record.is_answerable:
        return Verdict(VerdictStatus.UNANSWERABLE, shipped, _render(expected), path, "shipped answer is a placeholder")

    if not isinstance(expected, (dict, list)) and normalise(shipped) == normalise(expected):
        # Catches the fields whose "no value" marker is a real token rather than a
        # placeholder: `time_to_34kt_max` is "N/A" at 300 horizons and the answer is the
        # same "N/A", which a numeric comparison would reject for having no digits in it.
        return Verdict(VerdictStatus.MATCH, shipped, _render(expected), path, "identical after normalisation")

    if semantics is TemplateSemantics.THRESHOLD_PREDICATE:
        return _compare_predicate(record, shipped, expected, path)
    if semantics is TemplateSemantics.ORDINAL_BUCKET:
        return _compare_bucket(record, shipped, expected, path)

    fmt = record.answer_format

    if fmt is AnswerFormat.DESCRIPTION:
        # §4.1 — "an LLM-judged semantic score in [0, 1]". Reached only when the shipped answer is
        # *not* already equal to the field after normalisation; in the release that never happens,
        # because all 24,094 description answers are byte-identical copies (see
        # KNOWN_DEFECTS['td-answers-are-verbatim-field-copies']). So this branch is a guard against
        # the release changing, not a routine path — and it refuses to guess rather than inventing
        # a similarity score that would look like a real measurement.
        return Verdict(VerdictStatus.UNVERIFIABLE, shipped, _render(expected), path, "free text; §4.1 judge required")

    if fmt is AnswerFormat.MULTIPLE_CHOICE:
        return _compare_multiple_choice(record, shipped, expected, path)

    if fmt is AnswerFormat.TRUE_FALSE:
        return _compare_true_false(shipped, expected, path)

    return _compare_numeric(shipped, expected, path, relative_tolerance, absolute_floor)


def _render(value: Any) -> str:
    return value if isinstance(value, str) else repr(value)


def _compare_true_false(shipped: str, expected: Any, path: str) -> Verdict:
    """§4.1 — "exact match", after folding both sides to a polarity."""
    want = _polarity(expected)
    got = _polarity(shipped)
    if want is None:
        return Verdict(
            VerdictStatus.MISMATCH,
            shipped,
            _render(expected),
            path,
            f"ground truth {expected!r} carries no polarity, so the question has no Yes/No answer",
        )
    if got is None:
        return Verdict(VerdictStatus.MISMATCH, shipped, want, path, "shipped answer is not Yes/No")
    return Verdict(
        VerdictStatus.MATCH if got == want else VerdictStatus.MISMATCH,
        shipped,
        want,
        path,
    )


def _compare_multiple_choice(record: QARecord, shipped: str, expected: Any, path: str) -> Verdict:
    """§4.1 — "exact match", resolved through the option list.

    The shipped answer is a bare letter and the options live in the question string, so the
    letter is first mapped to its text. Both the SSHS normalisation of Q18 and the plain
    string comparison of the other MC templates go through the same path.
    """
    options = record.options or parse_inline_options(record.question)
    chosen = resolve_mc_letter(shipped, options)
    if chosen is None:
        return Verdict(
            VerdictStatus.MISMATCH,
            shipped,
            _render(expected),
            path,
            f"answer {shipped!r} is not one of the option letters {[letter for letter, _ in options]}",
        )

    variants = {normalise(chosen), leading_clause(chosen)}
    if (canonical := normalise_ssh_category(chosen)) is not None:
        variants.add(normalise(canonical))
    if (condition := PORT_CONDITION_ALIASES.get(chosen.strip().upper())) is not None:
        variants.add(normalise(condition))

    wanted = {normalise(expected), leading_clause(expected)}
    if (canonical := normalise_ssh_category(expected)) is not None:
        wanted.add(normalise(canonical))
    if isinstance(expected, str) and (condition := PORT_CONDITION_ALIASES.get(expected.strip().upper())) is not None:
        wanted.add(normalise(condition))

    if _categorical_match(variants, wanted):
        return Verdict(VerdictStatus.MATCH, chosen, _render(expected), path)
    return Verdict(VerdictStatus.MISMATCH, chosen, _render(expected), path, "option text differs from ground truth")


def _compare_numeric(
    shipped: str,
    expected: Any,
    path: str,
    relative_tolerance: float,
    absolute_floor: float,
) -> Verdict:
    """§4.1 — tolerance-based accuracy, plus the Q33 mapping case.

    Q33 ships a Python ``repr`` of a dict rather than the "ONLY a JSON" its question asks
    for; when the ground truth is a mapping, this compares key by key over the whole dict
    instead of falling through to "not a number".
    """
    if isinstance(expected, dict):
        parsed = _parse_mapping(shipped)
        if parsed is None:
            return Verdict(
                VerdictStatus.MISMATCH,
                shipped,
                _render(expected),
                path,
                "ground truth is a mapping but the shipped answer is not parseable as one",
            )
        if {k: normalise(v) for k, v in parsed.items()} == {k: normalise(v) for k, v in expected.items()}:
            return Verdict(VerdictStatus.MATCH, shipped, _render(expected), path)
        return Verdict(VerdictStatus.MISMATCH, shipped, _render(expected), path, "mapping entries differ")

    want = _extract_number(_render(expected))
    got = _extract_number(shipped)
    if want is None:
        return Verdict(
            VerdictStatus.MISMATCH,
            shipped,
            _render(expected),
            path,
            "ground truth is not a number, so tolerance-based accuracy does not apply",
        )
    if got is None:
        return Verdict(VerdictStatus.MISMATCH, shipped, want, path, "shipped answer has no number")

    tolerance = max(absolute_floor, relative_tolerance * abs(want))
    if abs(got - want) <= tolerance:
        return Verdict(VerdictStatus.MATCH, shipped, want, path)
    return Verdict(VerdictStatus.MISMATCH, shipped, want, path, f"|{got} - {want}| > {tolerance:g}")


def _parse_mapping(text: str) -> dict[str, Any] | None:
    """Parse a shipped answer that should be a mapping, accepting the Python-repr variant.

    Delegates to :meth:`QARecord.structured_answer`'s strategy without needing a record:
    JSON first, then ``ast.literal_eval`` for the Q33 case.
    """
    import ast
    import json

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
    return parsed if isinstance(parsed, dict) else None


__all__ = [
    "CATEGORY_RUNGS",
    "MISSING",
    "SSH_LADDER",
    "Verdict",
    "VerdictStatus",
    "bucket_letters",
    "compare",
    "evaluate_predicate",
    "expected_answer",
    "expected_path",
    "leading_clause",
    "normalise",
    "normalise_ssh_category",
    "parse_ordinal",
    "parse_threshold",
    "resolve_path",
]
