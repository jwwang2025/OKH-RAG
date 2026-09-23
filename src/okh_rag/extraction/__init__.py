"""§A.2 — knowledge hypergraph construction from scenario documents.

The subpackage is split so that each §A.2 step is testable on its own:

* :mod:`~okh_rag.extraction.horizon`   — the per-horizon decomposition
* :mod:`~okh_rag.extraction.prompts`   — the extraction prompt and its strict output schema
* :mod:`~okh_rag.extraction.extractor` — the driver: one LLM call per block, plus the §A.1 checks
* :mod:`~okh_rag.extraction.normalize` — steps 1-5 of §A.2

Only :func:`extract_hypergraph_from_document` and :func:`split_horizons` are normally called from
outside; the rest are exported because the evaluation and ablation scripts reason about intermediate
artifacts directly.
"""

from okh_rag.extraction.extractor import (
    BlockExtraction,
    ExtractionResult,
    ExtractionStats,
    extract_hypergraph_from_document,
    extract_horizon_block,
    is_grounded,
    locate_evidence,
)
from okh_rag.extraction.horizon import (
    HorizonBlock,
    ScenarioDocument,
    offsets_present,
    order_blocks_by_lead_time,
    split_horizons,
)
from okh_rag.extraction.normalize import (
    canonical_entity_id,
    canonicalize_entities,
    deduplicate,
    hyperedge_hash,
    inject_horizon_entities,
    normalize_hypergraph,
    normalize_relations,
    normalization_report,
    prune_orphan_entities,
    synthesize_cross_horizon_edges,
)
from okh_rag.extraction.prompts import (
    EXTRACTION_SCHEMA,
    EXTRACTION_SCHEMA_NAME,
    SYSTEM_PROMPT,
    build_messages,
    build_user_prompt,
    relation_legend,
)

__all__ = [
    "EXTRACTION_SCHEMA",
    "EXTRACTION_SCHEMA_NAME",
    "SYSTEM_PROMPT",
    "BlockExtraction",
    "ExtractionResult",
    "ExtractionStats",
    "HorizonBlock",
    "ScenarioDocument",
    "build_messages",
    "build_user_prompt",
    "canonical_entity_id",
    "canonicalize_entities",
    "deduplicate",
    "extract_hypergraph_from_document",
    "extract_horizon_block",
    "hyperedge_hash",
    "inject_horizon_entities",
    "is_grounded",
    "locate_evidence",
    "normalization_report",
    "normalize_hypergraph",
    "normalize_relations",
    "offsets_present",
    "order_blocks_by_lead_time",
    "prune_orphan_entities",
    "relation_legend",
    "split_horizons",
    "synthesize_cross_horizon_edges",
]
