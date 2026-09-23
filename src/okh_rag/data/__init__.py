"""The released CyPortQA benchmark, as data rather than as prose.

Public entry point is :func:`okh_rag.data.cyportqa.load_corpus`, which joins the three
files that matter — ``dataset/CyPortQA.json``, ``source_data/Encoded_senario.json`` and
``source_data/CyPortQA_template.json`` — into one :class:`CyPortQACorpus`.

Why this package exists at all is worth stating once, here. OKH-RAG's §4.1 delegates every
dataset detail to CyPortQA's own paper, and the release turns out to carry something the
OKH-RAG paper never mentions: a per-template map from each of the 48 questions to the
exact field of the structured encoding that produces its answer. That map makes the whole
reproduction checkable end to end — the hypergraph builder has a deterministic oracle for
which facts a question needs, and the evaluator has the same oracle for what the answer
should be. So the modules here do not merely load files; they carry the ground truth that
the paper's text alone cannot supply.

* :mod:`okh_rag.data.cyportqa` — loaders and record types
* :mod:`okh_rag.data.groundtruth` — template → field-path resolution and answer comparison
* :mod:`okh_rag.data.audit` — re-measurement of the release's known defects
"""

from okh_rag.data.cyportqa import (
    ANSWER_FORMAT_TAGS,
    HORIZON_HOURS,
    KNOWN_DEFECTS,
    OPERATION_FIELDS,
    PORT_CONDITIONS,
    PORT_CONDITION_ALIASES,
    TEMPLATE_SEMANTICS,
    TIME_BLOCK_FIELDS,
    AnswerFormat,
    CyPortQACorpus,
    DatasetDefect,
    DefectSeverity,
    EncodedScenario,
    QAContext,
    QARecord,
    ScenarioId,
    Template,
    TemplateSemantics,
    TimeBlock,
    is_placeholder_answer,
    load_corpus,
    load_encoded,
    load_imbalanced,
    load_qa,
    load_templates,
    options_from_choice,
    parse_inline_options,
    resolve_mc_letter,
)
from okh_rag.data.groundtruth import (
    CATEGORY_RUNGS,
    MISSING,
    Verdict,
    VerdictStatus,
    bucket_letters,
    compare,
    evaluate_predicate,
    expected_answer,
    leading_clause,
    normalise,
    parse_ordinal,
    parse_threshold,
    resolve_path,
)

__all__ = [
    "ANSWER_FORMAT_TAGS",
    "CATEGORY_RUNGS",
    "HORIZON_HOURS",
    "KNOWN_DEFECTS",
    "MISSING",
    "OPERATION_FIELDS",
    "PORT_CONDITIONS",
    "PORT_CONDITION_ALIASES",
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
    "Verdict",
    "VerdictStatus",
    "bucket_letters",
    "compare",
    "evaluate_predicate",
    "expected_answer",
    "is_placeholder_answer",
    "leading_clause",
    "load_corpus",
    "load_encoded",
    "load_imbalanced",
    "load_qa",
    "load_templates",
    "normalise",
    "options_from_choice",
    "parse_inline_options",
    "parse_ordinal",
    "parse_threshold",
    "resolve_mc_letter",
    "resolve_path",
]
