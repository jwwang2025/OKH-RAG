"""§A.2 / §A.1 — the extraction driver: scenario text in, normalized hypergraph out.

This module is the only place the extraction model is called. It owns the whole path from a
:class:`~okh_rag.extraction.horizon.ScenarioDocument` to a §A.2-normalized
:class:`~okh_rag.schema.KnowledgeHypergraph`:

    split (§A.2, in :mod:`horizon`)
      -> per-block LLM call with a strict schema (:mod:`prompts`)
      -> step 1, relation normalization  (:func:`~okh_rag.extraction.normalize.normalize_relations`)
      -> build Entity / Hyperedge objects with provisional ids
      -> steps 2-5 (:func:`~okh_rag.extraction.normalize.normalize_hypergraph`)

**Why the entities get provisional ids here.** §A.2 step 2 is the step that writes the canonical
``family:storm:port:horizon`` identifier, and it derives a horizon for transient states from the
hyperedges the entity appears in (:func:`~okh_rag.extraction.normalize._horizon_hint`). That
mechanism only works if extraction has *not* already decided the entity's horizon. So this module
mints one provisional id per ``(block, name, type)`` — deliberately over-splitting — and lets step 2
collapse the ones that should collapse:

* a transient state seen at ``T-120`` and at ``T-24`` yields two provisional ids, so step 2 gives them
  two canonical ids with different horizon components. §A.1 calls transient states "horizon-specific
  instances", so this is the intended reading;
* the storm and the port are seen in every block and yield one provisional id each, and step 2 gives
  both the horizon component ``none`` because §A.1 says persistent objects "persist across the
  scenario". They collapse onto a single entity, which is what §A.6 Eq. 12's Jaccard term and §A.4's
  entity-overlap signal need.

The provisional ids never reach the caller: step 2 rewrites every one of them.

**What is enforced here, and why here.** Three §A.1 constraints are checked at this boundary because
they are properties of a single extracted fact:

* ``|V_e| >= 2`` (Eq. 7) — a one-entity "hyperedge" is a node, not an interaction.
* ``n_v ⊆ s_e`` (§A.1) — an entity name that does not occur in the fact's own evidence string is
  ungrounded, and §A.2 step 5 hashes ``s_e``, so an ungrounded name would put a fabricated entity into
  a deduplicated record.
* ``c_e ∈ (0, 1]`` (§A.1) — clamped by :class:`~okh_rag.schema.Hyperedge`, thresholded here against
  ``hypergraph.min_confidence``.

Facts that violate a constraint are dropped and counted, never silently repaired, except for the
grounding case: there the offending *entity* is dropped and the fact survives if enough entities
remain, because one hallucinated entity is not evidence that the interaction is hallucinated.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from okh_rag.config import Config, HypergraphConfig
from okh_rag.extraction.horizon import HorizonBlock, ScenarioDocument, order_blocks_by_lead_time
from okh_rag.extraction.normalize import (
    hyperedge_hash,
    normalize_hypergraph,
    normalize_relations,
    slug,
)
from okh_rag.extraction.prompts import (
    EXTRACTION_SCHEMA,
    EXTRACTION_SCHEMA_NAME,
    build_messages,
)
from okh_rag.llm.base import ModelBackend, Usage
from okh_rag.schema import (
    Entity,
    Hyperedge,
    KnowledgeHypergraph,
    is_known_type,
    normalize_type_name,
)
from okh_rag.vocabulary import RELATION_TO_FAMILY

__all__ = [
    "BlockExtraction",
    "ExtractionResult",
    "ExtractionStats",
    "extract_hypergraph_from_document",
    "extract_horizon_block",
    "is_grounded",
    "locate_evidence",
]

# ---------------------------------------------------------------------------
# Small, total readers for model output
# ---------------------------------------------------------------------------
# The strict JSON schema makes malformed records unlikely, but "unlikely" is not a guarantee: a cached
# response written by an older schema revision, or a stub backend in a test, can still produce them.
# These helpers never raise, so a single bad field degrades one fact instead of aborting a paid run.

_ARTICLE_PREFIX_RE = re.compile(r"^(?:the|a|an)\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_QUOTE_CHARS = " \t\r\n\"'`\u201c\u201d\u2018\u2019"


def _as_text(value: Any) -> str:
    """Coerce a field to a whitespace-collapsed string, returning ``""`` for ``None``.

    Evidence text is hashed in §A.2 step 5, so collapsing whitespace here rather than at hash time
    keeps the stored ``s_e`` identical to the string the hash was computed over.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return " ".join(str(value).split())


def _as_float(value: Any, default: float) -> float:
    """Coerce a field to ``float``, falling back to ``default`` on anything unparseable."""
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _records(value: Any) -> list[Mapping[str, Any]]:
    """The mapping elements of a JSON array field, skipping anything that is not an object."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _attributes(value: Any) -> dict[str, str]:
    """``a_e`` — the ``{key, value}`` list of §A.1 packed into the dict the dataclass stores.

    §A.1 defines ``a_e = {(k_j, u_j)}_{j=1..A}``, a set of pairs. Duplicate keys are last-write-wins:
    two attribute records with the same key are the model repeating itself, not two distinct facts.
    """
    out: dict[str, str] = {}
    for record in _records(value):
        key = _as_text(record.get("key"))
        if not key:
            continue
        out[key] = _as_text(record.get("value"))
    return out


# ---------------------------------------------------------------------------
# §A.1 n_v ⊆ s_e — grounding
# ---------------------------------------------------------------------------


def is_grounded(name: str, evidence: str) -> bool:
    """Whether entity name ``n_v`` occurs in the fact's evidence ``s_e`` (§A.1).

    §A.1 states the relation ("grounded in the source hyperedge description (n_v ⊆ s_e)") but not the
    matching rule.

    # [UNSPECIFIED] The paper does not define the match. Using: case-folded, whitespace-collapsed
    # containment; then the same test after removing a leading article; then a token-subset test that
    # requires every token of the name with >= 3 characters to occur somewhere in the evidence.
    # Alternatives: exact case-sensitive containment (rejects "Hurricane ARTHUR" written against
    # "Hurricane Arthur" in the evidence, which the extractor legitimately produces), character-level
    # containment only (accepts the substring "art" inside "Arthur", too loose to be a check).

    The token-subset branch exists because the prompt asks for the name to be copied verbatim while
    the *evidence* is one model-written sentence; "Hurricane ARTHUR" and "the hurricane Arthur" are the
    same reference, and rejecting that pair would discard a large share of correct facts.
    """
    if not name or not evidence:
        return False
    haystack = " ".join(evidence.lower().split())
    needle = " ".join(name.lower().split()).strip(_QUOTE_CHARS)
    if not needle:
        return False
    if needle in haystack:
        return True
    without_article = _ARTICLE_PREFIX_RE.sub("", needle)
    if without_article and without_article in haystack:
        return True
    tokens = [t for t in _TOKEN_RE.findall(needle) if len(t) >= 3]
    return bool(tokens) and all(t in haystack for t in tokens)


# ---------------------------------------------------------------------------
# §A.3 text_position
# ---------------------------------------------------------------------------


def locate_evidence(block_text: str, block_offset: int, evidence: str) -> int:
    """Approximate character offset of a fact inside its source document.

    §A.3 uses text position as the last tie-breaker once phase rank, family rank and lead time are
    equal, so it only has to be monotone in reading order, not exact.

    ``s_e`` is a model-written sentence, not a copy of the source, so an exact offset usually does not
    exist. Three progressively looser searches are tried — verbatim, case-folded, then a
    whitespace-tolerant regex over the first 12/8/5 words — and the block's own offset is the fallback.
    That fallback still orders facts correctly *across* blocks, which is where the tie-breaker carries
    the most weight, and it is the honest answer when the evidence has been reworded beyond
    recognition.

    Args:
        block_text: The horizon block's raw text.
        block_offset: The block's offset within the parent document.
        evidence: The fact's evidence string.

    Returns:
        A character offset into the parent document.
    """
    needle = " ".join(evidence.split())
    index = -1
    if needle:
        index = block_text.find(needle)
        if index < 0:
            index = block_text.lower().find(needle.lower())
        if index < 0:
            words = needle.split()
            for take in (12, 8, 5):
                if len(words) < take:
                    continue
                pattern = re.compile(
                    r"\s+".join(re.escape(w) for w in words[:take]), re.IGNORECASE
                )
                match = pattern.search(block_text)
                if match is not None:
                    index = match.start()
                    break
    if index < 0:
        return block_offset
    return block_offset + index


# ---------------------------------------------------------------------------
# Run statistics
# ---------------------------------------------------------------------------


@dataclass
class ExtractionStats:
    """Counters for one document's extraction, so a run manifest can explain what was discarded.

    Every field is a count of *records*, not of tokens or calls, and the drop counters are mutually
    exclusive: each rejected fact increments exactly one of them, at the first check it fails.
    """

    documents: int = 0
    blocks: int = 0
    offsets_seen: tuple[int, ...] = ()
    facts_returned: int = 0
    facts_kept: int = 0
    hyperedges_built: int = 0
    dropped_unresolved_relation: int = 0
    dropped_unusable_relation: int = 0
    dropped_empty_evidence: int = 0
    dropped_low_confidence: int = 0
    dropped_small_entity_set: int = 0
    entities_dropped_ungrounded: int = 0
    entities_merged: int = 0
    unknown_entity_types: Counter[str] = field(default_factory=Counter)
    unresolved_relations: Counter[str] = field(default_factory=Counter)

    @property
    def facts_dropped(self) -> int:
        """Facts the model returned that did not become hyperedges."""
        return self.facts_returned - self.facts_kept

    @property
    def keep_rate(self) -> float:
        """``facts_kept / facts_returned``, or ``0.0`` when the model returned nothing.

        §A.2's five steps are lossy by construction, so this is the number to watch when extraction
        quality regresses: a sudden drop means the prompt, the schema enum, or the alias table drifted.
        """
        if self.facts_returned == 0:
            return 0.0
        return self.facts_kept / self.facts_returned

    def as_dict(self) -> dict[str, Any]:
        """Plain, JSON-serialisable view for the run manifest."""
        return {
            "documents": self.documents,
            "blocks": self.blocks,
            "offsets_seen": list(self.offsets_seen),
            "facts_returned": self.facts_returned,
            "facts_kept": self.facts_kept,
            "facts_dropped": self.facts_dropped,
            "keep_rate": round(self.keep_rate, 4),
            "hyperedges_built": self.hyperedges_built,
            "dropped_unresolved_relation": self.dropped_unresolved_relation,
            "dropped_unusable_relation": self.dropped_unusable_relation,
            "dropped_empty_evidence": self.dropped_empty_evidence,
            "dropped_low_confidence": self.dropped_low_confidence,
            "dropped_small_entity_set": self.dropped_small_entity_set,
            "entities_dropped_ungrounded": self.entities_dropped_ungrounded,
            "entities_merged": self.entities_merged,
            "unknown_entity_types": dict(self.unknown_entity_types),
            "unresolved_relations": dict(self.unresolved_relations),
        }


# ---------------------------------------------------------------------------
# Per-document accumulation
# ---------------------------------------------------------------------------


class _DocumentBuilder:
    """Turns validated facts into :class:`Entity` / :class:`Hyperedge` objects for one document.

    Kept as a class rather than a set of functions because the provisional-id registry and the counters
    are per-document state that every fact mutates.
    """

    def __init__(self, cfg: HypergraphConfig) -> None:
        self._cfg = cfg
        self.hypergraph = KnowledgeHypergraph()
        self.stats = ExtractionStats()
        self._raw_ids: dict[tuple[str, str, int], str] = {}

    # -- ids ---------------------------------------------------------------

    def _raw_id(self, name: str, type_name: str, block_index: int) -> str:
        """A provisional entity id, unique per ``(block, name, type)``.

        The block index is part of the key so that a transient state mentioned at two horizons starts
        out as two entities; §A.2 step 2 decides whether they stay separate. See the module docstring.
        """
        key = (name.lower(), normalize_type_name(type_name), block_index)
        existing = self._raw_ids.get(key)
        if existing is not None:
            return existing
        raw = f"raw:{len(self._raw_ids)}:{slug(key[0])}:{slug(key[1])}"
        self._raw_ids[key] = raw
        return raw

    # -- ingestion ---------------------------------------------------------

    def ingest(
        self,
        facts: Sequence[Mapping[str, Any]],
        *,
        block: HorizonBlock,
        block_index: int,
    ) -> None:
        """Fold one block's already-normalized facts into the document hypergraph.

        Args:
            facts: Facts whose ``relation`` is already a §A.1 Table 3 label (§A.2 step 1).
            block: The block they were extracted from; supplies the horizon and text offset.
            block_index: The block's position in lead-time order, used to scope provisional ids.
        """
        for record in facts:
            self.stats.facts_returned += 1

            evidence = _as_text(record.get("evidence"))
            if not evidence:
                self.stats.dropped_empty_evidence += 1
                continue

            relation = _as_text(record.get("relation"))
            if relation not in RELATION_TO_FAMILY:
                # Step 1 resolves every relation, so reaching here means it was bypassed or the
                # vocabulary changed underneath a cached response. Not repairable — a guessed label
                # would corrupt §A.3's phase ordering.
                self.stats.dropped_unusable_relation += 1
                continue

            members = self._collect_entities(record, evidence=evidence, block_index=block_index)
            if len(members) < self._cfg.min_entity_set_size:
                self.stats.dropped_small_entity_set += 1
                continue

            confidence = _as_float(record.get("confidence"), 1.0)
            if confidence < self._cfg.min_confidence:
                self.stats.dropped_low_confidence += 1
                continue

            # Entities are registered only after every check has passed, so a rejected fact can never
            # leave an orphan entity behind. §A.1 / Eq. 9 defines V as the union of the V_e, so an
            # entity in no hyperedge is not part of the hypergraph.
            for entity in members:
                self._register_entity(entity)
            self._register_hyperedge(
                block=block,
                members=members,
                relation=relation,
                evidence=evidence,
                attributes=_attributes(record.get("attributes")),
                confidence=confidence,
            )
            self.stats.facts_kept += 1

    def _collect_entities(
        self,
        record: Mapping[str, Any],
        *,
        evidence: str,
        block_index: int,
    ) -> list[Entity]:
        """Build ``V_e`` for one fact, applying §A.1's grounding requirement."""
        seen: dict[str, Entity] = {}
        for item in _records(record.get("entities")):
            name = _as_text(item.get("name")).strip(_QUOTE_CHARS)
            type_name = normalize_type_name(_as_text(item.get("type")))
            if not name or not type_name:
                continue
            if not is_known_type(type_name):
                # §A.1's type system is domain-specific and the prompt's enum closes it, so this only
                # happens if the provider ignored the enum. schema.category_of rolls an unknown type up
                # to "transient", the conservative reading, but the incident is worth counting.
                self.stats.unknown_entity_types[type_name] += 1
            if not is_grounded(name, evidence):
                self.stats.entities_dropped_ungrounded += 1
                continue

            raw_id = self._raw_id(name, type_name, block_index)
            entity = Entity(
                entity_id=raw_id,
                name=name,
                type=type_name,
                description=_as_text(item.get("description")),
                confidence=_as_float(item.get("confidence"), 1.0),
            )
            previous = seen.get(raw_id)
            if previous is None or entity.confidence > previous.confidence:
                seen[raw_id] = entity
        return list(seen.values())

    def _register_entity(self, entity: Entity) -> None:
        """Add ``entity``, keeping the most confident description when the provisional id repeats."""
        existing = self.hypergraph.entities.get(entity.entity_id)
        if existing is None:
            self.hypergraph.add_entity(entity)
            return
        self.stats.entities_merged += 1
        if entity.confidence > existing.confidence:
            self.hypergraph.add_entity(entity)

    def _register_hyperedge(
        self,
        *,
        block: HorizonBlock,
        members: Sequence[Entity],
        relation: str,
        evidence: str,
        attributes: Mapping[str, str],
        confidence: float,
    ) -> None:
        """Add one §A.1 Eq. 7 hyperedge.

        The id is the §A.2 step-5 hash computed over *provisional* ids, which makes it a placeholder:
        step 5 recomputes it once step 2 has rewritten ``V_e``. Building it here anyway keeps
        :class:`~okh_rag.schema.Hyperedge`'s non-empty-id invariant satisfied throughout, and makes a
        crash mid-pipeline leave an inspectable graph.
        """
        entity_ids = tuple(entity.entity_id for entity in members)
        self.hypergraph.add_hyperedge(
            Hyperedge(
                hyperedge_id=hyperedge_hash(relation, entity_ids, evidence, self._cfg),
                entity_ids=entity_ids,
                relation=relation,
                evidence=evidence,
                attributes=dict(attributes),
                confidence=confidence,
                group_id=block.group_id,
                horizon_hours=block.offset_hours,
                text_position=locate_evidence(block.text, block.char_offset, evidence),
            )
        )
        self.stats.hyperedges_built += 1


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockExtraction:
    """The model's response for one horizon block, after §A.2 step 1.

    Attributes:
        facts: Facts whose relation resolved onto the §A.1 vocabulary. Still raw in every other
            respect — ids, dedup and horizon injection have not run yet.
        unresolved_relations: Relation strings that matched neither the Table 3 vocabulary, the alias
            table, nor fuzzy matching. Reported rather than logged-and-forgotten, because a cluster of
            the same string is the signal that the alias table needs a new entry.
    """

    facts: tuple[Mapping[str, Any], ...]
    unresolved_relations: tuple[str, ...]


def extract_horizon_block(
    block: HorizonBlock,
    *,
    storm: str,
    port: str,
    backend: ModelBackend,
    cfg: Config,
) -> BlockExtraction:
    """Call the extraction model once for one horizon block, then apply §A.2 step 1.

    §A.2 explains why extraction is per block: "Multi-horizon scenario documents can span thousands of
    tokens. Single-call extraction risks truncation, entity conflation across horizons, and reduced
    attribute coverage."

    Args:
        block: The block to extract from.
        storm: Scenario storm name, for the prompt's scenario header.
        port: Scenario port name, for the prompt's scenario header.
        backend: The chat backend. §4.1 uses GPT-4o.
        cfg: The root config; ``llm.generator_model`` and ``llm.extraction_temperature`` are used, and
            ``hypergraph.normalization`` drives step 1.

    Returns:
        The kept facts and the relations that could not be resolved.
    """
    payload = backend.complete_json(
        build_messages(block, storm=storm, port=port),
        model=cfg.llm.generator_model,
        temperature=cfg.llm.extraction_temperature,
        schema=EXTRACTION_SCHEMA,
        schema_name=EXTRACTION_SCHEMA_NAME,
    )
    raw_facts = [dict(record) for record in _records(payload.get("facts"))]
    kept, unresolved = normalize_relations(raw_facts, cfg.hypergraph)
    return BlockExtraction(facts=tuple(kept), unresolved_relations=tuple(unresolved))


@dataclass(frozen=True)
class ExtractionResult:
    """Everything one document's extraction produced, including what it cost.

    Attributes:
        hypergraph: The §A.2-normalized hypergraph. Entities and hyperedges carry canonical ids, every
            horizon-grounded hyperedge carries its temporal anchor, cross-horizon change edges have been
            synthesised, and exact duplicates are gone.
        stats: Drop counters and diagnostics; see :meth:`ExtractionStats.as_dict`.
        usage: Token and call delta attributable to this document.
    """

    hypergraph: KnowledgeHypergraph
    stats: ExtractionStats
    usage: Usage


def extract_hypergraph_from_document(
    document: ScenarioDocument,
    *,
    backend: ModelBackend,
    cfg: Config,
) -> ExtractionResult:
    """Build the §A.2 knowledge hypergraph for one scenario document.

    The full path documented in the module docstring, with §A.2's steps run in the only order that
    works: step 1 per fact (so an off-vocabulary fact is dropped before it can contribute an entity),
    then steps 2-5 on the merged graph.

    Args:
        document: The scenario, already split into horizon blocks.
        backend: The chat backend.
        cfg: The root config.

    Returns:
        An :class:`ExtractionResult`. The hypergraph's ``groups`` has one entry, keyed by
        ``document.group_id``.

    Raises:
        ValueError: Propagated from :func:`~okh_rag.extraction.horizon.split_horizons` if the document
            has no horizon headers. Blocks are assumed already split, so this only fires if the caller
            built a degenerate :class:`ScenarioDocument`.
    """
    builder = _DocumentBuilder(cfg.hypergraph)
    blocks = order_blocks_by_lead_time(list(document.blocks))
    builder.stats.documents = 1
    builder.stats.offsets_seen = tuple(sorted({b.offset_hours for b in blocks}, reverse=True))

    usage_before = _usage_of(backend)
    for block_index, block in enumerate(blocks):
        builder.stats.blocks += 1
        extraction = extract_horizon_block(
            block,
            storm=document.storm,
            port=document.port,
            backend=backend,
            cfg=cfg,
        )
        builder.stats.dropped_unresolved_relation += len(extraction.unresolved_relations)
        builder.stats.unresolved_relations.update(extraction.unresolved_relations)
        builder.ingest(list(extraction.facts), block=block, block_index=block_index)
    usage_after = _usage_of(backend)

    hypergraph = normalize_hypergraph(
        builder.hypergraph,
        storm=document.storm,
        port=document.port,
        cfg=cfg.hypergraph,
    )
    return ExtractionResult(
        hypergraph=hypergraph,
        stats=builder.stats,
        usage=_usage_delta(usage_before, usage_after),
    )


# ---------------------------------------------------------------------------
# Usage accounting
# ---------------------------------------------------------------------------


def _usage_of(backend: ModelBackend) -> Usage | None:
    """Snapshot a backend's cumulative usage, or ``None`` if it does not report any."""
    usage = getattr(backend, "usage", None)
    if usage is None:
        return None
    return Usage(
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0)),
        completion_tokens=int(getattr(usage, "completion_tokens", 0)),
        embedding_tokens=int(getattr(usage, "embedding_tokens", 0)),
        calls=int(getattr(usage, "calls", 0)),
    )


def _usage_delta(before: Usage | None, after: Usage | None) -> Usage:
    """Usage attributable to this document. Missing snapshots yield a zero record, not an error."""
    if before is None or after is None:
        return Usage()
    return Usage(
        prompt_tokens=after.prompt_tokens - before.prompt_tokens,
        completion_tokens=after.completion_tokens - before.completion_tokens,
        embedding_tokens=after.embedding_tokens - before.embedding_tokens,
        calls=after.calls - before.calls,
    )
