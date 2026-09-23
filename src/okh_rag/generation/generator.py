"""§3.3 / §A.8 — the generator: one question, one evidence context, one answer.

§3.3 states the generation objective as ``y ∼ P(y | q, Γq)`` and then gives two ways to evaluate it:

* **single-call** — *"the primary mode presents all paths in a single prompt for cross-referencing"*;
* **per-trajectory** — *"a fallback mode aggregates independent per-trajectory answers via
  confidence-weighted voting or averaging"*.

§A.8 adds why the first is preferred: *"it enables direct cross-referencing rather than post-hoc
reconciliation"*. That is a strong claim about the mechanism — the gain is supposed to come from the model
seeing the paths *together* — and it is testable here, because the two modes are the same call count and
the same model over the same retrieved content, differing only in whether the paths are reconciled by the
model or by arithmetic. A run that only implements the preferred mode cannot show that its preference was
earned, so both are implemented.

The aggregation rules are inferred from §A.8's one sentence ("confidence-weighted voting (categorical) or
averaging (numeric)") and the choices inside them are recorded on :func:`aggregate`, because the sentence
does not fix a tie-break, does not say what the aggregate's own confidence should be, and does not say what
to do with answers that failed :mod:`~okh_rag.generation.output`'s contract.

Nothing here talks to a model API. The generator is constructed with an
:class:`~okh_rag.llm.base.LLMClient`, so a test can answer from a scripted stub and a production run can
answer from GPT-4o without either path being privileged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from okh_rag.config import Config
from okh_rag.data.cyportqa import AnswerFormat, QARecord
from okh_rag.generation.evidence import EvidenceContext
from okh_rag.generation.output import (
    ANSWER_JSON_SCHEMA_NAME,
    GeneratedAnswer,
    answer_json_schema,
    canonicalise,
    parse_generated_answer,
    vote_key,
)
from okh_rag.generation.prompts import build_messages
from okh_rag.llm.base import LLMClient

__all__ = [
    "GenerationRecord",
    "Generator",
    "GENERATION_MODES",
    "aggregate",
]

#: §A.8's two modes. ``single_call`` is the paper's stated preference and the default in
#: ``configs/base.yaml``; ``per_trajectory`` exists to be compared against it.
GENERATION_MODES = ("single_call", "per_trajectory")


@dataclass(frozen=True)
class GenerationRecord:
    """One question's generation outcome, in enough detail to diagnose a wrong answer.

    Attributes:
        qa_id: The question's stable identity (``QARecord.qa_id``).
        template_id: ``Q1``..``Q48``, so results aggregate by question type.
        scenario_id: The knowledge group, so results aggregate by scenario.
        answer_format: §4.1's regime, which decides the scoring rule.
        source: Which retriever produced the evidence — ``okh_rag`` or a baseline name.
        mode: Which §A.8 mode produced :attr:`answer`.
        answer: The single answer that gets scored. In ``single_call`` mode this is the model's own
            response; in ``per_trajectory`` mode it is :func:`aggregate`'s result.
        trajectories: The raw per-trajectory responses, empty in ``single_call`` mode.
        notes: Anything the run needs to remember — the aggregation rule used, the §4.5 shuffle flag,
            the number of contract violations that were dropped before aggregating.
    """

    qa_id: str
    template_id: str
    scenario_id: str
    answer_format: AnswerFormat
    source: str
    mode: str
    answer: GeneratedAnswer
    trajectories: tuple[GeneratedAnswer, ...] = ()
    notes: tuple[str, ...] = ()
    messages: tuple[Mapping[str, str], ...] = ()

    @property
    def conforming(self) -> bool:
        """Whether the scored answer satisfied §A.8's contract."""
        return self.answer.is_conforming

    def as_dict(self, *, include_messages: bool = False) -> dict[str, Any]:
        """Plain-dict view for run manifests."""
        payload: dict[str, Any] = {
            "qa_id": self.qa_id,
            "template_id": self.template_id,
            "scenario_id": self.scenario_id,
            "answer_format": self.answer_format.value,
            "source": self.source,
            "mode": self.mode,
            "answer": self.answer.as_dict(),
            "trajectories": [a.as_dict() for a in self.trajectories],
            "notes": list(self.notes),
        }
        if include_messages:
            payload["messages"] = [dict(m) for m in self.messages]
        return payload


class Generator:
    """§3.3's ``P(y | q, Γq)``, over any :class:`~okh_rag.llm.base.LLMClient`.

    Args:
        backend: The chat endpoint. Typed as the narrow client protocol rather than
            ``ModelBackend`` because generation needs no embeddings — a run that only answers questions
            should not have to construct an embedding client to do it.
        cfg: The full config. Only ``llm.generator_model``, ``llm.generation_temperature`` and the
            ``generation`` switches are read, but the whole object is taken because that is what the
            pipeline holds and threading three scalars through every call site is how a config and its
            callers drift apart.

    The class is stateless between calls apart from the backend's own cache and usage counter, so one
    instance serves a whole evaluation run.
    """

    def __init__(self, backend: LLMClient, cfg: Config) -> None:
        self.backend = backend
        self.cfg = cfg

    # -- public API ---------------------------------------------------------

    def answer(
        self,
        record: QARecord,
        evidence: EvidenceContext,
        *,
        mode: str | None = None,
        extra_guidance: str = "",
    ) -> GenerationRecord:
        """Answer one question from one evidence context.

        Args:
            record: The question.
            evidence: The retrieved context, from any retriever.
            mode: ``single_call`` or ``per_trajectory``. Defaults to ``cfg.generation.mode``.
            extra_guidance: Instruction appended to the output contract — the §4.5 shuffle note reaches
                the model through here.

        Returns:
            A :class:`GenerationRecord`. Never raises on a bad model response: a malformed answer is a
            recorded outcome (:mod:`~okh_rag.generation.output`), because aborting a run at question
            7,000 of 117,178 costs the whole run to report one bad sample.
        """
        chosen = mode or self.cfg.generation.mode
        if chosen not in GENERATION_MODES:
            raise ValueError(f"unknown generation mode {chosen!r}; expected one of {GENERATION_MODES}")

        if chosen == "single_call":
            answer = self._single_call(record, evidence, extra_guidance=extra_guidance)
            return GenerationRecord(
                qa_id=record.qa_id,
                template_id=record.template_id,
                scenario_id=record.scenario_id,
                answer_format=record.answer_format,
                source=evidence.source,
                mode=chosen,
                answer=answer,
            )

        per_trajectory = self._per_trajectory(record, evidence, extra_guidance=extra_guidance)
        aggregated, notes = aggregate(per_trajectory, record.answer_format)
        return GenerationRecord(
            qa_id=record.qa_id,
            template_id=record.template_id,
            scenario_id=record.scenario_id,
            answer_format=record.answer_format,
            source=evidence.source,
            mode=chosen,
            answer=aggregated,
            trajectories=per_trajectory,
            notes=notes,
        )

    def messages_for(
        self,
        record: QARecord,
        evidence: EvidenceContext,
        *,
        extra_guidance: str = "",
    ) -> tuple[dict[str, str], ...]:
        """The messages :meth:`answer` would send, without sending them.

        Exposed because §4.1's comparison is only interpretable if the prompt is inspectable, and because
        the cheap way to catch a prompt-building regression is to diff the prompt rather than the answer.
        """
        return build_messages(
            record,
            evidence,
            include_quality_summary=self.cfg.generation.include_quality_summary,
            include_reasoning_tags=self.cfg.generation.include_reasoning_tags,
            extra_guidance=extra_guidance,
        )

    # -- modes --------------------------------------------------------------

    def _single_call(
        self,
        record: QARecord,
        evidence: EvidenceContext,
        *,
        extra_guidance: str,
    ) -> GeneratedAnswer:
        """§A.8's primary mode: one call, all paths, one answer."""
        messages = self.messages_for(record, evidence, extra_guidance=extra_guidance)
        payload = self._complete(messages, record, evidence)
        return parse_generated_answer(
            payload,
            record.answer_format,
            options=record.options,
        )

    def _per_trajectory(
        self,
        record: QARecord,
        evidence: EvidenceContext,
        *,
        extra_guidance: str,
    ) -> tuple[GeneratedAnswer, ...]:
        """§A.8's fallback mode: one call per trajectory, aggregated afterwards.

        A trajectory is presented alone, as a single path, so the model is not told there are alternatives
        — that is what makes the mode a genuine per-path answer rather than a differently-phrased
        single-call. When retrieval returned no trajectory to isolate, the shared context is used as the
        single path so that the question is still answered rather than silently skipped.
        """
        contexts = _isolate_trajectories(evidence)
        answers: list[GeneratedAnswer] = []
        for path_index, context in enumerate(contexts, start=1):
            messages = self.messages_for(record, context, extra_guidance=extra_guidance)
            payload = self._complete(messages, record, context)
            answer = parse_generated_answer(payload, record.answer_format, options=record.options)
            # The path number is known here even if the model omitted it, and the aggregation note quotes
            # it, so it is filled in rather than left blank on a conforming-in-every-other-way answer.
            if answer.supporting_trajectory is None:
                answer = GeneratedAnswer(
                    answer=answer.answer,
                    confidence=answer.confidence,
                    rationale=answer.rationale,
                    supporting_trajectory=str(path_index),
                    violations=answer.violations,
                    raw=answer.raw,
                )
            answers.append(answer)
        return tuple(answers)

    # -- transport ----------------------------------------------------------

    def _complete(
        self,
        messages: Sequence[Mapping[str, str]],
        record: QARecord,
        evidence: EvidenceContext,
    ) -> str:
        """One schema-constrained call, returned as the raw assistant text.

        The schema is derived per question rather than fixed, because §A.8's type-specific constraints are
        per question (a TF question's ``answer`` is an enum of two strings; a numeric one's is a number;
        an MC one's admissible labels come from that question's options). Constraining the decoder is
        strictly better than constraining the parser: a model that cannot emit ``"probably"`` for a
        yes/no question cannot make the mistake at all.

        Returns raw text for :func:`~okh_rag.generation.output.parse_generated_answer` to parse, rather
        than the already-parsed dict, so that the conforming path and the replay path run through the same
        parser and cannot diverge.
        """
        multiple = evidence.is_multi_trajectory
        schema = answer_json_schema(
            record.answer_format,
            options=record.options,
            multi_trajectory=multiple,
        )
        parsed: Mapping[str, Any] = self.backend.complete_json(
            messages,
            model=self.cfg.llm.generator_model,
            temperature=self.cfg.llm.generation_temperature,
            schema=schema,
            schema_name=ANSWER_JSON_SCHEMA_NAME,
        )
        # Re-serialised so that parsing is the only path into a ``GeneratedAnswer``. The backend already
        # validated the shape, so this round-trip is cheap; skipping it would mean two constructors with
        # two behaviours to keep in sync.
        return json.dumps(parsed)


# ---------------------------------------------------------------------------
# Aggregation (§A.8's fallback mode)
# ---------------------------------------------------------------------------


def aggregate(
    answers: Sequence[GeneratedAnswer],
    answer_format: AnswerFormat,
) -> tuple[GeneratedAnswer, tuple[str, ...]]:
    """Combine per-trajectory answers into the one that gets scored.

    §A.8 says the fallback aggregates "via confidence-weighted voting (categorical) or averaging
    (numeric)". The sentence is a sketch, so each decision it leaves open is resolved here and labelled:

    * **Non-conforming answers are dropped before aggregating.** An answer that violated §A.8's contract
      has no comparable value — a TF answer of ``"maybe"`` cannot be voted on against ``"Yes"`` — and
      letting it into a vote would give a contract violation the same standing as a valid answer. Dropped
      answers are counted in the returned notes, so the loss is visible instead of silent.
    * **The vote is by canonical value, weighted by confidence.** Canonicalisation reuses
      :func:`~okh_rag.generation.output.canonicalise`, so ``"yes"`` and ``"Yes"`` are one vote. Weights
      come from :attr:`GeneratedAnswer.weight`, which treats a missing confidence as 1.0 rather than 0.0 —
      a missing field should not delete a vote.
    * **Ties break toward the earlier trajectory.** Trajectories arrive best-first, so "earlier" means
      "higher-scoring", which is the only ordering information available. A tie is reported either way.
    * **The aggregate's confidence is the winner's share of total weight.** Not the winner's own stated
      confidence, which would discard the disagreement between paths — the single most useful number this
      mode produces. A unanimous vote yields the mean confidence; a split vote yields less.
    * **The rationale is the winning group's highest-confidence member's.** Concatenating rationales would
      read as one coherent justification for a claim that may have been argued several incompatible ways.
    * **Numeric questions are averaged, not voted on.** §A.8 says averaging, and averaging is also the
      only sensible option: two paths answering 48 and 49 agree and should not be counted as a disagreement
      the way two paths answering ``"Yes"`` and ``"No"`` are.

    Args:
        answers: Per-trajectory answers, in retrieval order (best first).
        answer_format: Decides the aggregation rule.

    Returns:
        ``(aggregated, notes)``. When every answer was non-conforming, the first answer is returned
        unchanged so the caller still has the model's raw output to report, and the notes say so.
    """
    notes: list[str] = []
    usable = [a for a in answers if a.is_conforming]
    dropped = len(answers) - len(usable)
    if dropped:
        notes.append(f"aggregation dropped {dropped}/{len(answers)} non-conforming answers")

    if not usable:
        if not answers:
            return (
                GeneratedAnswer(
                    answer=None,
                    violations=(),
                    raw={},
                ),
                tuple(notes + ["no per-trajectory answers were produced"]),
            )
        notes.append("every answer violated the output contract; the first is reported unaggregated")
        return answers[0], tuple(notes)

    if len(usable) == 1:
        notes.append("only one conforming answer; no aggregation was performed")
        return usable[0], tuple(notes)

    if answer_format is AnswerFormat.NUMERIC:
        aggregated = _average_numeric(usable)
        notes.append("aggregated by confidence-weighted averaging (§A.8, numeric)")
        return aggregated, tuple(notes)

    aggregated = _vote_categorical(usable, answer_format)
    notes.append("aggregated by confidence-weighted voting (§A.8, categorical)")
    return aggregated, tuple(notes)


def _average_numeric(answers: Sequence[GeneratedAnswer]) -> GeneratedAnswer:
    """The confidence-weighted mean of the numeric answers.

    A member whose answer is not a number cannot reach here (it would have been non-conforming), so the
    arithmetic is safe; the guard is written anyway because :func:`aggregate` is public and a future
    caller may pass unvalidated answers.
    """
    weighted_sum = 0.0
    total_weight = 0.0
    for answer in answers:
        value = canonicalise(answer.answer, AnswerFormat.NUMERIC)
        if not isinstance(value, (int, float)):
            continue
        weighted_sum += float(value) * answer.weight
        total_weight += answer.weight
    if total_weight <= 0.0:
        # Every weight was zero — a legal confidence, and the only way to reach this branch. Falling back
        # to the unweighted mean is more informative than returning the first answer, because zero-weight
        # answers are still answers.
        mean = sum(
            float(canonicalise(a.answer, AnswerFormat.NUMERIC))
            for a in answers
            if isinstance(canonicalise(a.answer, AnswerFormat.NUMERIC), (int, float))
        ) / len(answers)
        return GeneratedAnswer(answer=mean, confidence=0.0, rationale=_best_rationale(answers))

    mean = weighted_sum / total_weight
    # Agreement is expressed through the confidence: the mean of the members' own confidences would claim
    # certainty a spread-out set of answers does not have, so the spread is folded in.
    spread = max(
        abs(float(canonicalise(a.answer, AnswerFormat.NUMERIC)) - mean) for a in answers
    )
    scale = max(abs(mean), 1e-9)
    agreement = max(0.0, 1.0 - spread / (2 * scale))
    confidence = agreement * (sum(a.weight for a in answers) / len(answers))
    return GeneratedAnswer(answer=mean, confidence=min(1.0, confidence), rationale=_best_rationale(answers))


def _vote_categorical(
    answers: Sequence[GeneratedAnswer],
    answer_format: AnswerFormat,
) -> GeneratedAnswer:
    """Confidence-weighted plurality vote over canonical values.

    Insertion order is preserved when scanning for the winner, so the tie-break "earlier trajectory wins"
    is the natural consequence of ``max`` over a stable iteration rather than a separate comparison — one
    less thing to get wrong.
    """
    weights: dict[str, float] = {}
    members: dict[str, list[GeneratedAnswer]] = {}
    order: list[str] = []
    for answer in answers:
        key = vote_key(answer.answer, answer_format)
        if key not in weights:
            weights[key] = 0.0
            members[key] = []
            order.append(key)
        weights[key] += answer.weight
        members[key].append(answer)

    winning_key = max(order, key=lambda key: weights[key])
    group = members[winning_key]
    total_weight = sum(weights.values())
    share = weights[winning_key] / total_weight if total_weight > 0 else 0.0

    representative = max(group, key=lambda a: a.weight)
    canonical = canonicalise(representative.answer, answer_format)
    return GeneratedAnswer(
        answer=canonical,
        confidence=min(1.0, share),
        rationale=_best_rationale(group),
        supporting_trajectory=representative.supporting_trajectory,
    )


def _best_rationale(answers: Sequence[GeneratedAnswer]) -> str:
    """The rationale of the highest-confidence member, or the longest when weights are equal.

    Falling back to the longest is not arbitrary: a rationale is the one field whose usefulness grows with
    the evidence it names, and "longest" is a fair proxy when confidence cannot break the tie.
    """
    if not answers:
        return ""
    with_rationale = [a for a in answers if a.rationale]
    if not with_rationale:
        return ""
    best = max(with_rationale, key=lambda a: (a.weight, len(a.rationale)))
    return best.rationale


# ---------------------------------------------------------------------------
# Per-trajectory isolation
# ---------------------------------------------------------------------------


def _isolate_trajectories(evidence: EvidenceContext) -> tuple[EvidenceContext, ...]:
    """One context per **non-empty** trajectory, each containing exactly one path.

    Empty trajectories are dropped rather than isolated. "One call per trajectory" presupposes that the
    trajectory is a path; an empty one is a retrieval artifact, and asking the model to answer from no
    evidence would add a low-information vote to the aggregate — one that carries the same weight as a
    real path while contributing nothing but a parametric guess. If *every* trajectory is empty, the shared
    context is returned as a single path, so a retrieval failure still produces an answer: an evaluation
    that silently skips its hardest questions reports a flattering number.

    The **notes are carried into every isolated context** rather than only the first: a note like the §4.5
    shuffle warning is part of how the evidence must be read, and a note that reaches the model on path 1
    but not paths 2..N would make the mode's own comparison inconsistent.
    """
    paths = [trajectory for trajectory in evidence.trajectories if len(trajectory)]
    if not paths:
        return (evidence,)
    return tuple(
        EvidenceContext(
            question=evidence.question,
            trajectories=(trajectory,),
            source=evidence.source,
            notes=evidence.notes,
        )
        for trajectory in paths
    )
