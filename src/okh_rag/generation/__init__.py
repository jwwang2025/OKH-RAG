"""§3.3 / §A.8 — retrieval-augmented generation over ordered evidence.

§3.3's design claim is narrow and falsifiable: *"Concatenating hyperedge texts into a flat context window
would collapse ordered trajectories back into unordered sets."* Everything in this package exists to make
that claim testable rather than to make it true by construction. The pieces, in the order a run uses them:

* :mod:`~okh_rag.generation.evidence` — renders retrieval output as §A.8's numbered, annotated evidence
  blocks.
* :mod:`~okh_rag.generation.prompts` — assembles the chat messages, honouring the release's own input
  protocol (``prompt`` as system, ``question`` as user) and appending §A.8's output contract.
* :mod:`~okh_rag.generation.output` — §A.8's ``{answer, confidence, rationale}`` envelope, its
  per-format constraints, and the validation that reports violations instead of repairing them.
* :mod:`~okh_rag.generation.generator` — :class:`~okh_rag.generation.generator.Generator`, which drives
  the two §A.8 modes and aggregates the fallback's per-trajectory answers.

Dependency direction: ``generation`` imports from ``data``, ``schema``, ``llm``, ``retrieval`` and
``vocabulary``. Nothing in those packages imports ``generation`` — a retriever that knew how its output
would be phrased could not be compared fairly against one that did not.
"""

from okh_rag.generation.evidence import (
    EvidenceContext,
    EvidenceStep,
    EvidenceTrajectory,
    Passage,
    context_from_retrieval,
    horizon_label,
    reasoning_tags,
    trajectory_from_hyperedge_ids,
)
from okh_rag.generation.generator import (
    GENERATION_MODES,
    GenerationRecord,
    Generator,
    aggregate,
)
from okh_rag.generation.output import (
    AnswerViolation,
    GeneratedAnswer,
    answer_json_schema,
    canonicalise,
    parse_generated_answer,
    validate,
)
from okh_rag.generation.prompts import ANSWER_FIELD_RULES, build_messages, output_contract

__all__ = [
    "ANSWER_FIELD_RULES",
    "GENERATION_MODES",
    "AnswerViolation",
    "EvidenceContext",
    "EvidenceStep",
    "EvidenceTrajectory",
    "GeneratedAnswer",
    "GenerationRecord",
    "Generator",
    "Passage",
    "aggregate",
    "answer_json_schema",
    "build_messages",
    "canonicalise",
    "context_from_retrieval",
    "horizon_label",
    "output_contract",
    "parse_generated_answer",
    "reasoning_tags",
    "trajectory_from_hyperedge_ids",
    "validate",
]
