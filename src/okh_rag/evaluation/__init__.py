"""§4.1's evaluation surface, and the retrieval oracle the ``CyPortQA`` release makes possible.

The subpackage exists to keep one thing out of :mod:`okh_rag.retrieval`: the benchmark. §3.2's
retriever knows about a hypergraph, a partial order and a query string, and nothing about
CyPortQA — which is why it can be read as a §3.2 implementation rather than as a solution to
one dataset. Everything that does know about CyPortQA is here.

* :mod:`~okh_rag.evaluation.field_map` — ``CONTRACT.md`` §3.2's source-field to §A.1 relation
  table, as data. A leaf: no corpus, no config, no graph.
* :mod:`~okh_rag.evaluation.reference_graph` — the closed-form hypergraph a perfect §A.2
  extractor would have produced from ``Encoded_senario.json``. This is what lets the oracle run
  with no API key, and what makes §A.2 extraction scoreable in the same currency as retrieval.
* :mod:`~okh_rag.evaluation.oracle` — the gold-set definition and the metrics. Works on any
  hypergraph, so the same metrics score the reference graph and an extracted one.
* :mod:`~okh_rag.evaluation.answers` — §4.1's answer scoring: exact match for TF/MC, tolerance for
  SA, semantic (exact or judged) for TD, plus the micro/macro aggregation and the exclusions each
  accuracy divides by. Separate from the oracle because the two answer different questions: the
  oracle scores *what was retrieved*, this scores *what was said*.

The dependency direction is one-way and deliberate: ``evaluation`` imports from ``data``,
``retrieval`` and ``generation``, and none of them imports ``evaluation``. The one exception is
deliberate and narrow — :mod:`~okh_rag.evaluation.answers` reuses
:func:`~okh_rag.generation.output.canonicalise`, because a scoring rule that disagreed with the
answer contract about what a valid answer is would score correct answers as wrong. That sharing is
the reason the canonical form lives in ``generation`` rather than here: the generator has to know it
to be able to *produce* a conforming answer at all.
"""

from okh_rag.evaluation.answers import (
    NUMERIC_GOLD_CAUSES,
    AccuracyReport,
    AnswerScore,
    JudgeFn,
    ScoringRegime,
    numeric_gold_cause,
    regime_of,
    score_answer,
    score_record,
    summarise,
    tolerance_for,
)
from okh_rag.evaluation.field_map import (
    ENTITY_FIELD_PATHS,
    FIELD_TO_RELATION,
    FieldKind,
    FieldResolution,
    relations_for_path,
)
from okh_rag.evaluation.oracle import (
    EvidenceStatus,
    GoldEvidence,
    OracleMetrics,
    OracleOutcome,
    OracleReport,
    TemplateCoverage,
    evaluate_record,
    gold_for_record,
    gold_matches_field_provenance,
    template_coverage,
)
from okh_rag.evaluation.reference_graph import (
    RELATION_ENTITY_TYPE,
    FieldInstance,
    ReferenceGraph,
    ReferenceGraphStats,
    build_reference_graph,
    iter_field_instances,
)

__all__ = [
    "ENTITY_FIELD_PATHS",
    "FIELD_TO_RELATION",
    "NUMERIC_GOLD_CAUSES",
    "RELATION_ENTITY_TYPE",
    "AccuracyReport",
    "AnswerScore",
    "EvidenceStatus",
    "FieldInstance",
    "FieldKind",
    "FieldResolution",
    "GoldEvidence",
    "JudgeFn",
    "OracleMetrics",
    "OracleOutcome",
    "OracleReport",
    "ReferenceGraph",
    "ReferenceGraphStats",
    "ScoringRegime",
    "TemplateCoverage",
    "build_reference_graph",
    "evaluate_record",
    "gold_for_record",
    "gold_matches_field_provenance",
    "iter_field_instances",
    "numeric_gold_cause",
    "regime_of",
    "relations_for_path",
    "score_answer",
    "score_record",
    "summarise",
    "template_coverage",
    "tolerance_for",
]
