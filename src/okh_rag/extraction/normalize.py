"""§A.2 — the five post-processing steps.

§A.2: "Five normalization steps ensure consistency across per-horizon extractions."

    1. Relation normalization      — "mapped to R via the alias table and fuzzy matching"
    2. Entity ID canonicalization  — "identifiers are rewritten to a hierarchical convention encoding
                                      family, storm, port, and horizon (e.g., wind_fcst:IRMA:port_arthur:T-48)"
    3. Horizon entity injection    — "a canonical temporal anchor (horizon:T-48) is created for each
                                      detected horizon and added to all hyperedges at that horizon"
    4. Cross-horizon edge synthesis— "synthetic hyperedges (forecast_updates_to, changes_probability_to)
                                      are created to represent evolution"
    5. Deduplication               — "unique hyperedge IDs are computed as hashes of relation, entity set,
                                      and evidence text; exact duplicates are collapsed"

The steps run in this order because each one's input is the previous one's output: step 4 matches on the
canonical relation from step 1 and the canonical entity ids from step 2, and step 5 hashes the entity set
that steps 2-4 finalise.

:func:`prune_orphan_entities` runs last as a guard rather than as a sixth step: §A.1's entity set is what
the hyperedges use, so an entity no hyperedge references is not part of ``V`` and must not reach the
embedding index.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping

from okh_rag.config import HypergraphConfig
from okh_rag.schema import Entity, EntityCategory, Hyperedge, KnowledgeHypergraph, normalize_type_name
from okh_rag.vocabulary import (
    ATTR_ORIGIN_FAMILY,
    ATTR_ORIGIN_HORIZONS,
    CROSS_HORIZON_FAMILY,
    family_of,
)

# ---------------------------------------------------------------------------
# Step 1 — relation normalization
# ---------------------------------------------------------------------------


def normalize_relations(
    facts: list[dict[str, Any]],
    cfg: HypergraphConfig,
) -> tuple[list[dict[str, Any]], list[str]]:
    """§A.2 step 1 — map every relation string onto the §A.1 Table 3 vocabulary.

    Resolution order: exact match, then the alias table (§A.1's one worked example plus the
    hand-authored variations in :data:`~okh_rag.vocabulary.RELATION_ALIASES`), then fuzzy matching at
    ``cfg.normalization.fuzzy_threshold``.

    Args:
        facts: Raw extraction records, each with a ``relation`` key.
        cfg: The ``hypergraph`` config section.

    Returns:
        ``(kept_facts, unresolved_relations)``. A fact whose relation cannot be resolved is dropped:
        ``r_e`` "is a typed relation from a controlled vocabulary" (§A.1), so a fact with an
        off-vocabulary relation cannot participate in phase ordering, coverage scoring, or the
        alias-table contract, and keeping it under a guessed label would corrupt §A.3.
    """
    from okh_rag.vocabulary import (
        RELATION_ALIASES,
        RELATION_TO_FAMILY,
        fuzzy_match_relation,
        normalize_relation_token,
    )

    kept: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for fact in facts:
        raw = str(fact.get("relation", ""))
        token = normalize_relation_token(raw)

        if token in RELATION_TO_FAMILY:
            canonical = token
        elif token in RELATION_ALIASES:
            canonical = RELATION_ALIASES[token]
        else:
            canonical = fuzzy_match_relation(token, cfg.normalization.fuzzy_threshold) or ""

        if not canonical:
            unresolved.append(raw)
            continue

        fact = dict(fact)
        fact["relation"] = canonical
        kept.append(fact)

    return kept, unresolved


# ---------------------------------------------------------------------------
# Step 2 — entity ID canonicalization
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(value: str) -> str:
    """Lower-case, underscore-separated token used by the §A.2 step-2 id convention."""
    out = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    return out or "unknown"


def entity_family_slug(type_name: str) -> str:
    """The leading ``family`` component of a canonical entity id.

    §A.2's example is ``wind_fcst:IRMA:port_arthur:T-48``, where ``wind_fcst`` is a short family label
    rather than a §A.1 type name. The derivation rule is not published.
    # [UNSPECIFIED] §A.2 gives one example id and no derivation rule. Using: the normalised §A.1 type
    # name as the family slug. Alternatives: a hand-maintained type->slug table, or reusing the
    # hyperedge's relation family.
    """
    return slug(normalize_type_name(type_name))


def port_slug(port: str) -> str:
    """The ``port`` component of a canonical entity id.

    ``port_arthur`` in §A.2's example is the slug of the port name with a ``port_`` prefix, so a
    location entity that is a port is distinguishable from a storm that happens to share its name.
    """
    return f"port_{slug(port)}"


def canonical_entity_id(
    *,
    entity_name: str,
    entity_type: str,
    storm: str,
    port: str,
    horizon: str,
    cfg: HypergraphConfig,
) -> str:
    """§A.2 step 2 — build ``family:storm:port:horizon``.

    "#A.2 — "identifiers are rewritten to a hierarchical convention encoding family, storm, port, and
    horizon (e.g., wind_fcst:IRMA:port_arthur:T-48), ensuring cross-block consistency."

    The entity's own name is folded into the ``family`` component, because the four named components do
    not include it and two entities of the same type at the same horizon (two distinct hazard forecasts)
    would otherwise collide onto one id.

    Args:
        entity_name: ``n_v`` from the extractor.
        entity_type: ``τ_v`` from the extractor.
        storm: Scenario storm name.
        port: Scenario port name.
        horizon: ``T-NN`` label, or ``none`` for a hyperedge with no horizon.
        cfg: The ``hypergraph`` config section, which carries ``entity_id_template``.

    Returns:
        The canonical id.
    """
    family = f"{entity_family_slug(entity_type)}${slug(entity_name)}"
    return cfg.normalization.entity_id_template.format(
        family=family,
        storm=slug(storm),
        port=port_slug(port),
        horizon=slug(horizon),
    )


def canonicalize_entities(
    hypergraph: KnowledgeHypergraph,
    *,
    storm: str,
    port: str,
    cfg: HypergraphConfig,
) -> KnowledgeHypergraph:
    """§A.2 step 2 — rewrite every entity id onto the hierarchical convention, in place.

    Two effects, both required by later stages:

    * Identifiers become comparable across horizon blocks, which §A.2 lists as the step's purpose.
    * Entities that denote the same persistent object at different horizons (the storm, the port) collapse
      onto one id, which is what makes §A.6 Eq. 12's Jaccard overlap between consecutive steps non-zero
      and what §A.4's entity-overlap signal needs.

    Collapsing two extraction records onto one canonical id is the common case, not an edge case: the
    extractor's provisional ids are per ``(block, name, type)``, so the storm and the port arrive once
    per horizon and the same state can arrive twice within a block. When that happens the surviving
    entity keeps the *most confident* description rather than whichever record happened to be inserted
    last, so the result does not depend on dict iteration order.

    Returns:
        The same hypergraph, mutated. Returned for call chaining.
    """
    id_map: dict[str, str] = {}
    new_entities: dict[str, Entity] = {}

    for old_id, entity in hypergraph.entities.items():
        if entity.category is EntityCategory.TEMPORAL:
            # A temporal anchor already encodes its horizon in its name.
            match = re.search(r"T-\d+", entity.name)
            horizon = match.group(0) if match else "none"
        elif entity.category is EntityCategory.PERSISTENT:
            # §A.1 — persistent objects "persist across the scenario", so their identity must not carry
            # a horizon component. Including one would mint a separate id per horizon and destroy the
            # cross-horizon identity that §A.6 Eq. 12's Jaccard term and §A.4's entity-overlap signal
            # both depend on.
            horizon = "none"
        else:
            # Transient states are "horizon-specific instances" (§A.1), so they do carry one.
            horizon = _horizon_hint(hypergraph, old_id)

        new_id = canonical_entity_id(
            entity_name=entity.name,
            entity_type=entity.type,
            storm=storm,
            port=port,
            horizon=horizon,
            cfg=cfg,
        )
        id_map[old_id] = new_id
        candidate = Entity(
            entity_id=new_id,
            name=entity.name,
            type=entity.type,
            description=entity.description,
            confidence=entity.confidence,
        )
        previous = new_entities.get(new_id)
        if previous is None or candidate.confidence > previous.confidence:
            new_entities[new_id] = candidate

    hypergraph.entities = new_entities
    for hid, edge in list(hypergraph.hyperedges.items()):
        mapped = tuple(id_map[e] for e in edge.entity_ids if e in id_map)
        # Dedupe: collapsing two distinct ids onto one (same name, same type, same horizon) would
        # otherwise leave a duplicate member and break the |V_e| >= 2 check for a 2-entity hyperedge.
        deduped = tuple(dict.fromkeys(mapped))
        edge.entity_ids = deduped
        hypergraph.hyperedges[hid] = edge

    return hypergraph


def _horizon_hint(hypergraph: KnowledgeHypergraph, entity_id: str) -> str:
    """The horizon label for a transient-state entity, taken from the hyperedges it appears in.

    §A.1 — transient states are "horizon-specific instances capturing the system's configuration at a
    particular point". When an entity shows up at several horizons (because extraction was per-horizon and
    the same wording recurred), the *earliest* label wins, so the entity joins the horizon at which its
    state first existed rather than being duplicated per horizon.
    """
    horizons = [
        edge.horizon_hours
        for edge in hypergraph.hyperedges.values()
        if entity_id in edge.entity_ids and edge.horizon_hours is not None
    ]
    if not horizons:
        return "none"
    return f"T-{max(horizons)}"


# ---------------------------------------------------------------------------
# Step 3 — horizon entity injection
# ---------------------------------------------------------------------------


def horizon_anchor_id(*, storm: str, port: str, offset_hours: int, cfg: HypergraphConfig) -> str:
    """The canonical id of a horizon anchor.

    §A.2 step 3 says "a canonical temporal anchor (horizon:T-48) is created", and §A.1 gives the same
    notation. But a *globally* shared id is incompatible with §A.4's training signals, which sample
    "random cross-group pairs" as negatives: every hyperedge in the corpus carries exactly one anchor, so
    a global ``horizon:T-48`` would make every cross-group pair share an entity, and the entity-overlap
    signal would have no negative examples at all.

    [CONTRADICTION] resolved by reading ``horizon:T-48`` as the anchor's *name* (§A.1's wording is
    "temporal anchors (e.g., horizon:T-48)") and applying §A.2 step 2's hierarchical id convention to its
    *id*. See REPRODUCTION_NOTES.md (contradiction 6).

    Returns:
        ``horizon_time:{storm}:{port}:T-NN``.
    """
    return canonical_entity_id(
        entity_name=f"horizon:T-{offset_hours}",
        entity_type="HORIZON_TIME",
        storm=storm,
        port=port,
        horizon=f"T-{offset_hours}",
        cfg=cfg,
    )


def inject_horizon_entities(
    hypergraph: KnowledgeHypergraph,
    *,
    storm: str,
    port: str,
    cfg: HypergraphConfig,
) -> KnowledgeHypergraph:
    """§A.2 step 3 — add a temporal anchor to every horizon-grounded hyperedge, in place.

    "[A] canonical temporal anchor (horizon:T-48) is created for each detected horizon and added to all
    hyperedges at that horizon, making horizon membership structurally explicit."

    §A.1 states the invariant this produces: "Every horizon-grounded hyperedge includes exactly one
    temporal anchor, enabling horizon-scoped candidate selection and order enforcement during retrieval."

    Hyperedges without a horizon — the §A.2 step-4 synthetic change edges — are skipped, because they
    span two horizons and giving them one anchor would misreport their position.
    """
    by_horizon: dict[int, str] = {}
    for offset in {e.horizon_hours for e in hypergraph.hyperedges.values() if e.horizon_hours is not None}:
        anchor_id = horizon_anchor_id(storm=storm, port=port, offset_hours=offset, cfg=cfg)
        by_horizon[offset] = anchor_id
        hypergraph.add_entity(
            Entity(
                entity_id=anchor_id,
                name=f"horizon:T-{offset}",
                type="HORIZON_TIME",
                description=f"{offset} hours before expected landfall",
                confidence=1.0,
            )
        )

    for hid, edge in list(hypergraph.hyperedges.items()):
        if edge.horizon_hours is None:
            continue
        anchor = by_horizon[edge.horizon_hours]
        if anchor not in edge.entity_ids:
            edge.entity_ids = (*edge.entity_ids, anchor)
            hypergraph.hyperedges[hid] = edge

    return hypergraph


# ---------------------------------------------------------------------------
# Step 4 — cross-horizon edge synthesis
# ---------------------------------------------------------------------------


def synthesize_cross_horizon_edges(
    hypergraph: KnowledgeHypergraph,
    *,
    cfg: HypergraphConfig,
) -> KnowledgeHypergraph:
    """§A.2 step 4 — create the family-13 change edges that link consecutive horizons, in place.

    "for entity families at multiple horizons, synthetic hyperedges (forecast_updates_to,
     changes_probability_to) are created to represent evolution, providing the cross-horizon links that
     the precedence construction requires."

    §4.2 gives the observed effect: in an evolution-rich scenario "nearly half of its hyperedges are
    cross-horizon transitions, such as forecast_updates_to and changes_status_to, linking states across
    horizons and introducing dependencies that unfold over time."

    Two things the paper leaves open are resolved by config, both flagged ``[UNSPECIFIED]`` in
    ``configs/base.yaml``:

    * **Which families.** ``cfg.normalization.cross_horizon_families`` is ``["all"]`` by default, the
      literal reading of "for entity families at multiple horizons". Family 13 is always excluded: its
      edges already span horizons and were given ``horizon_hours=None``, so they are not indexed here.
    * **Which pairs.** A change edge is drawn only between hyperedges that share at least one entity, so
      the link means "this state, for this port and storm, is superseded at the next horizon" rather
      than joining two unrelated facts. Each source then keeps its
      ``cfg.normalization.cross_horizon_max_fanout`` best destinations, ranked by the number of shared
      entities and tie-broken by text position, which keeps the edge count linear in the corpus instead
      of quadratic in each family's per-horizon size.

    The probability family gets ``changes_probability_to`` and every other family
    ``forecast_updates_to`` — the two relation names §A.2 actually prints.

    Each synthesised edge records ``origin_family`` and ``origin_horizons`` in ``a_e`` so §A.3's
    family-to-change rule can attach it to the right within-horizon family, and so §A.8 can print a
    meaningful "Reasoning: cross_horizon" line.
    """
    wanted = set(cfg.normalization.cross_horizon_families)
    synthesize_every_family = "all" in wanted
    max_fanout = cfg.normalization.cross_horizon_max_fanout
    probability_family = family_of("has_cumulative_probability")

    # family -> {horizon -> members}
    index: dict[str, dict[int, list[Hyperedge]]] = {}
    for edge in hypergraph.hyperedges.values():
        if edge.horizon_hours is None:
            continue
        family = family_of(edge.relation)
        if family is None or family == CROSS_HORIZON_FAMILY:
            continue
        if not synthesize_every_family and family not in wanted:
            continue
        index.setdefault(family, {}).setdefault(edge.horizon_hours, []).append(edge)

    existing = {(e.relation, frozenset(e.entity_ids)) for e in hypergraph.hyperedges.values()}

    for family, by_horizon in index.items():
        horizons = sorted(by_horizon, reverse=True)  # largest lead time first, per §A.3
        relation = (
            cfg.normalization.cross_horizon_probability_relation
            if family == probability_family
            else cfg.normalization.cross_horizon_relation
        )
        for earlier, later in zip(horizons, horizons[1:]):
            destinations = by_horizon[later]
            for src in by_horizon[earlier]:
                src_entities = set(src.entity_ids)
                ranked = []
                for dst in destinations:
                    overlap = src_entities & set(dst.entity_ids)
                    if not overlap:
                        # Without a shared entity the change edge would join two unrelated facts and
                        # would contribute zero to §A.6 Eq. 12's Jaccard term, so it is not a
                        # trajectory link — it is noise.
                        continue
                    ranked.append((-len(overlap), dst.text_position, dst.hyperedge_id, dst))
                if not ranked:
                    continue
                ranked.sort()
                chosen = ranked if max_fanout <= 0 else ranked[:max_fanout]

                for _, _, _, dst in chosen:
                    entities = tuple(sorted(src_entities | set(dst.entity_ids)))
                    if len(entities) < 2:
                        continue
                    if (relation, frozenset(entities)) in existing:
                        continue

                    # The evidence names both states because the edge's meaning *is* the transition, and
                    # §A.4 embeds this string and §A.8 prints it as a reasoning step. It is prefixed so a
                    # reader (and a lexical baseline) can tell a synthesized inference from a source fact
                    # — necessary because the text necessarily quotes the two states it connects.
                    evidence = (
                        f"{family.replace('_', ' ').capitalize()} evolution between T-{earlier} and "
                        f"T-{later}: {src.evidence} Subsequently at T-{later}: {dst.evidence}"
                    )
                    hypergraph.add_hyperedge(
                        Hyperedge(
                            hyperedge_id=hyperedge_hash(relation, entities, evidence, cfg),
                            entity_ids=entities,
                            relation=relation,
                            evidence=evidence,
                            attributes={
                                ATTR_ORIGIN_FAMILY: family,
                                ATTR_ORIGIN_HORIZONS: f"T-{earlier}->T-{later}",
                            },
                            confidence=min(src.confidence, dst.confidence),
                            group_id=src.group_id,
                            horizon_hours=None,
                            text_position=max(src.text_position, dst.text_position),
                        )
                    )
                    existing.add((relation, frozenset(entities)))

    return hypergraph


# ---------------------------------------------------------------------------
# Step 5 — deduplication
# ---------------------------------------------------------------------------


def hyperedge_hash(
    relation: str,
    entity_ids: Iterable[str],
    evidence: str,
    cfg: HypergraphConfig,
) -> str:
    """§A.2 step 5 — the hyperedge id.

    "[U]nique hyperedge IDs are computed as hashes of relation, entity set, and evidence text."

    # [UNSPECIFIED] The paper names the three components but neither the hash function nor a truncation
    # length. Using: `cfg.dedup.hash_algorithm` (sha1) truncated to `cfg.dedup.hash_length` (16 hex
    # chars). Alternatives: md5, blake2b, full digest.

    Entity order is normalised when ``cfg.dedup.sort_entity_names`` is set, so that the id depends on the
    entity *set* as the paper describes rather than on insertion order. Evidence is whitespace-collapsed
    because a trivial reformat of the same sentence is not a different fact.
    """
    ids = sorted(entity_ids) if cfg.dedup.sort_entity_names else list(entity_ids)
    normalized_evidence = " ".join(evidence.split())
    blob = f"{relation}\x1f{'|'.join(ids)}\x1f{normalized_evidence}"
    digest = hashlib.new(cfg.dedup.hash_algorithm, blob.encode("utf-8")).hexdigest()
    return digest[: cfg.dedup.hash_length]


def deduplicate(hypergraph: KnowledgeHypergraph, cfg: HypergraphConfig) -> KnowledgeHypergraph:
    """§A.2 step 5 — recompute every id from the final ``V_e``, then collapse exact duplicates.

    "[U]nique hyperedge IDs are computed as hashes of relation, entity set, and evidence text; exact
    duplicates are collapsed."

    The id is recomputed here rather than read from the record because steps 2-4 change the inputs the
    hash is defined over: step 2 rewrites ``V_e``, and a hash taken before it would encode ids that no
    longer exist. Recomputing is also what makes the step do any work at all — the extractor's
    provisional ids are per block, so the same fact extracted from two blocks has two different
    provisional hashes and only becomes an "exact duplicate" once both are expressed in canonical ids.

    Deduplication is by that hash, so two facts collide only if relation, entity set and evidence all
    agree. Two extractions of the same interaction with different wording therefore both survive, which
    is the intended behaviour: §A.2 calls these "exact duplicates", not paraphrases.

    Returns:
        A rebuilt hypergraph. The first-inserted record wins on collision, so the result does not depend
        on dict iteration order. ``groups`` is rebuilt from the surviving hyperedges.
    """
    rebuilt = KnowledgeHypergraph()
    for edge in hypergraph.hyperedges.values():
        edge.hyperedge_id = hyperedge_hash(edge.relation, edge.entity_ids, edge.evidence, cfg)
        if edge.hyperedge_id in rebuilt.hyperedges:
            continue
        rebuilt.add_hyperedge(edge)
    for entity in hypergraph.entities.values():
        rebuilt.add_entity(entity)
    return rebuilt


def prune_orphan_entities(hypergraph: KnowledgeHypergraph) -> KnowledgeHypergraph:
    """Drop entities that no hyperedge references, in place.

    §A.1 / Eq. 9 defines the entity set as what the hyperedges collectively use; an entity in no ``V_e``
    is not a member of ``V``. Keeping one would let it enter the embedding index and the
    entity-frequency table (:meth:`~okh_rag.schema.KnowledgeHypergraph.entity_document_frequency`) while
    being unreachable from retrieval, which is the kind of state that turns a scoring bug into a mystery.

    The extractor already avoids creating orphans — it registers a fact's entities only after that fact
    has passed every check — so this is a guard, not a workhorse. It matters if a future normalisation
    step removes hyperedges.
    """
    referenced = {eid for edge in hypergraph.hyperedges.values() for eid in edge.entity_ids}
    for entity_id in [e for e in hypergraph.entities if e not in referenced]:
        del hypergraph.entities[entity_id]
    return hypergraph


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def normalize_hypergraph(
    hypergraph: KnowledgeHypergraph,
    *,
    storm: str,
    port: str,
    cfg: HypergraphConfig,
) -> KnowledgeHypergraph:
    """Run §A.2's steps 2-5 in order on a merged per-horizon hypergraph.

    Step 1 (relation normalization) runs earlier, per fact, inside
    :mod:`okh_rag.extraction.extractor`, because a fact whose relation cannot be resolved is dropped
    before it ever becomes an entity or a hyperedge.

    Args:
        hypergraph: The merged hypergraph for one document.
        storm: Scenario storm name.
        port: Scenario port name.
        cfg: The ``hypergraph`` config section.

    Returns:
        A normalized, deduplicated hypergraph.
    """
    hypergraph = canonicalize_entities(hypergraph, storm=storm, port=port, cfg=cfg)
    hypergraph = inject_horizon_entities(hypergraph, storm=storm, port=port, cfg=cfg)
    hypergraph = synthesize_cross_horizon_edges(hypergraph, cfg=cfg)
    hypergraph = deduplicate(hypergraph, cfg=cfg)
    hypergraph = prune_orphan_entities(hypergraph)
    return hypergraph


def normalization_report(hypergraph: KnowledgeHypergraph) -> Mapping[str, int]:
    """Counts for a run manifest: how much of each stage's output actually survived."""
    from okh_rag.vocabulary import is_cross_horizon

    anchors = sum(1 for e in hypergraph.entities.values() if e.category is EntityCategory.TEMPORAL)
    synthetic = sum(1 for e in hypergraph.hyperedges.values() if is_cross_horizon(e.relation))
    return {
        "entities": len(hypergraph.entities),
        "hyperedges": len(hypergraph.hyperedges),
        "temporal_anchors": anchors,
        "cross_horizon_hyperedges": synthetic,
        "groups": len(hypergraph.groups),
    }
