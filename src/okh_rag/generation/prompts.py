"""§3.3 / §A.8 — the prompts that turn a question plus retrieved evidence into a structured answer.

Two things in this module are protocol, not style, and both come from outside the paper:

**1. The system message is the release's own.** ``data/CONTRACT.md`` §2.2 records what
``models/run_gpt4o.py`` does — ``Q["prompt"]`` is sent as the **system** message and ``Q["question"]`` as
the user turn. That is the input protocol CyPortQA's own runners use and therefore the protocol §4.1's
numbers were produced under, so it is reproduced here rather than replaced with a generic analyst persona.
Replacing it would change the framing every compared method receives, which is the one thing a baseline
study may not do.

**2. §A.8's output contract is stated in the user turn, and it overrides the release's format clause.**
The release's ``prompt`` typically ends *"Answer with ONLY 'Yes' or 'No'"* or *"Answer ONLY a number"*.
§A.8 requires ``{answer, confidence, rationale}`` as JSON. Both cannot hold, and the resolution is not
arbitrary: the release's format clause exists so that a zero-shot runner's stdout is scorable by string
match, which is a job the JSON envelope does better, and §4.1's comparison needs the confidence and
rationale fields that the bare form has no place for. The format instruction is therefore placed **after**
the question, where a later, more specific instruction is read as the operative one, and the answer field's
*content* rule keeps the release's requirement intact — ``"Yes"``/``"No"``, a letter, a number. The release
keeps its semantics; only the envelope changes. This is recorded as a [CONTRADICTION] in
``.paper2code_work/2604.12185/ambiguity_audit.md``.

The per-format answer rules are generated from :mod:`~okh_rag.generation.output`, not written out twice, so
that the prompt cannot ask for a shape the validator would reject. That coupling is deliberate: the most
expensive failure mode in a generation study is a prompt/validator mismatch, because every question then
fails for a reason that has nothing to do with the method being measured.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from okh_rag.data.cyportqa import AnswerFormat, QARecord
from okh_rag.generation.evidence import EvidenceContext

__all__ = [
    "ANSWER_FIELD_RULES",
    "build_messages",
    "output_contract",
]


#: §A.8's "type-specific constraints", one sentence per §4.1 regime. Kept as data rather than inline
#: f-strings so that the four rules can be read side by side and checked against the paper's sentence.
ANSWER_FIELD_RULES: Mapping[AnswerFormat, str] = {
    AnswerFormat.TRUE_FALSE: 'the single word "Yes" or "No"',
    AnswerFormat.MULTIPLE_CHOICE: "the single letter of one listed option (for example \"B\")",
    AnswerFormat.NUMERIC: (
        "a bare number — no unit, no percent sign, no words, no range; "
        "if the evidence does not state the quantity, use null"
    ),
    AnswerFormat.DESCRIPTION: "the text itself, as short as the question allows",
}


# The role sentence used only when a record ships no ``prompt``. It is not the default: the release's own
# framing is used whenever it exists, and this is the documented fallback so that a missing field produces
# a runnable prompt rather than an exception.
_FALLBACK_SYSTEM = (
    "Act as a cyclone and port operations analyst. Answer the question from the supplied evidence only."
)


def output_contract(
    answer_format: AnswerFormat,
    *,
    options: Sequence[tuple[str, str]] = (),
    extra_guidance: str = "",
) -> str:
    """The instruction block that carries §A.8's output format.

    Args:
        answer_format: Which §4.1 regime the answer will be scored under.
        options: For MC, the ``(label, text)`` pairs, used to state the admissible labels. Passing them
            matters because the release has 2,851 questions whose question text states a stale option
            count (audit defect ``mc-options-duplicated-and-inconsistent-with-prompt``), so "one of the
            letters shown above" is ambiguous on exactly those questions.
        extra_guidance: Appended verbatim. The generator uses it to carry the §4.5 shuffle note, which has
            to reach the model as an instruction about how to read the evidence rather than as a comment
            the model may treat as data.

    Returns:
        The contract text, ending without a trailing newline.
    """
    rule = ANSWER_FIELD_RULES[answer_format]
    lines = ["Respond with a single JSON object and nothing else, with exactly these three fields:"]
    lines.append(f'  "answer": {rule}')
    lines.append('  "confidence": a number between 0 and 1')
    lines.append('  "rationale": one sentence naming the evidence steps the answer rests on')
    if answer_format is AnswerFormat.MULTIPLE_CHOICE and options:
        listed = ", ".join(f'"{label}" = {text}' for label, text in options)
        lines.append(f"Valid answer values are exactly: {listed}.")
    if extra_guidance:
        lines.append(extra_guidance)
    return "\n".join(lines)


# The §A.8 multi-trajectory instruction, quoted almost verbatim from the paper: read all paths, treat
# convergence as a reliability signal, and name the supporting path. It is emitted only when more than one
# trajectory is present, because with a single path "read all the paths" is an instruction about a
# non-existent choice, and instructions that cannot be followed dilute the ones that can.
_MULTI_TRAJECTORY_INSTRUCTION = (
    "Several numbered reasoning paths are given below. Read all of them before answering. "
    "Treat agreement between paths as a reliability signal and disagreement as a reason to lower your "
    "confidence. Name the path that supports your answer in a fourth field, "
    '"supporting_trajectory", using its number (for example "2").'
)

# §3.3's reason for the whole exercise, stated to the model so that the ordered presentation is usable
# rather than merely present: the steps are numbered because the order carries information.
_ORDER_INSTRUCTION = (
    "Each evidence step is numbered and labelled with its horizon and causal phase. "
    "Read them in order: a step that follows another across horizons may be its consequence, and the "
    "step labels tell you which stage of the response each fact belongs to."
)


def build_messages(
    record: QARecord,
    evidence: EvidenceContext,
    *,
    include_quality_summary: bool = True,
    include_reasoning_tags: bool = True,
    extra_guidance: str = "",
) -> tuple[dict[str, str], ...]:
    """Assemble the chat messages for one question.

    Args:
        record: The question being answered. ``record.prompt`` becomes the system message verbatim and
            ``record.question`` the first line of the user turn, per the release's own protocol.
        evidence: The rendered retrieval context. Its ``source`` is named in the user turn so that a
            transcript shows which retriever produced it — useful when several methods' outputs are read
            side by side, and necessary for a run to be auditable after the fact.
        include_quality_summary: Render the §A.8 trajectory-level quality summary.
        include_reasoning_tags: Render the ``Reasoning:`` line.
        extra_guidance: Extra instruction appended to the output contract.

    Returns:
        ``(system, user)`` as a two-element tuple of ``{"role", "content"}`` mappings, ready to pass to
        :meth:`~okh_rag.llm.base.LLMClient.complete_json`.
    """
    system = (record.prompt or "").strip() or _FALLBACK_SYSTEM

    blocks: list[str] = [record.question.strip()]
    blocks.append(_ORDER_INSTRUCTION)

    rendered = evidence.render(
        include_quality_summary=include_quality_summary,
        include_reasoning=include_reasoning_tags,
    )
    if rendered.strip():
        blocks.append(f"Retrieved evidence ({evidence.source}):\n\n{rendered}")
    else:
        # §A.8 has no no-evidence branch, so this is stated rather than left as an empty section that
        # reads like a rendering bug. The model is told the retrieval found nothing, which makes an
        # unanswerable question answerable-as-unanswerable instead of tempting a parametric guess. The
        # source is still named: a transcript of a four-method comparison has to say *which* method came
        # back empty, or the four runs become indistinguishable exactly where they differ most.
        blocks.append(
            f"Retrieved evidence ({evidence.source}): none. "
            "No evidence steps were retrieved for this question. "
            "If the question cannot be answered from general knowledge alone, answer as best you can and "
            "set confidence low."
        )

    if evidence.is_multi_trajectory:
        blocks.append(_MULTI_TRAJECTORY_INSTRUCTION)

    blocks.append(
        output_contract(
            record.answer_format,
            options=record.options,
            extra_guidance=extra_guidance,
        )
    )

    return (
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(blocks)},
    )


def message_preview(messages: Sequence[Mapping[str, Any]], limit: int = 1200) -> str:
    """A truncated transcript of ``messages`` for logs and reports.

    Truncation is on the concatenated length rather than per message because the interesting part of a
    generation trace is where it stops being informative, and that boundary moves between a one-line
    question and a ten-step evidence block.
    """
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "?"))
        content = str(message.get("content", ""))
        parts.append(f"### {role}\n{content}")
    joined = "\n\n".join(parts)
    if len(joined) <= limit:
        return joined
    return joined[:limit] + f"\n... [{len(joined) - limit} more characters]"
