"""§A.1 Table 3 — the controlled relation vocabulary ``R``, and the phase mappings built on it.

Table 3 defines "13 semantic families in canonical phase order". Two different phase notions are needed
downstream and they are deliberately kept apart:

* :data:`PRECEDENCE_PHASE_ORDER` — the **11-stage** ordering of §A.3, used to build the precedence DAG.
* :data:`COVERAGE_PHASE_SET`-compatible labels — the **6-member** set ``S`` of §A.6 Eq. 13, used to
  score phase coverage. The 6-member set is the one the equation side actually enumerates; see the
  [CONTRADICTION] note on :func:`coverage_phase_of`.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Mapping

# ---------------------------------------------------------------------------
# §A.1 Table 3 — the 13 families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RelationFamily:
    """One row of §A.1 Table 3.

    Attributes:
        index: The table's own numbering, 1-13. §A.3 uses this as the ``family_rank`` tie-breaker.
        name: Our slug for the family (the table's "Family" column, lower-cased).
        relations: The "Representative relations" column, verbatim.
    """

    index: int
    name: str
    relations: tuple[str, ...]


# Verbatim from §A.1 Table 3.
RELATION_FAMILIES: tuple[RelationFamily, ...] = (
    RelationFamily(1, "cyclone_state", ("has_cyclone_state", "has_category_state", "has_motion")),
    RelationFamily(2, "track_landfall", ("forecasts_track", "forecasts_landfall")),
    RelationFamily(3, "timing", ("has_hours_to_landfall", "has_forecast_window")),
    RelationFamily(4, "advisory", ("has_watch_status", "has_warning_status")),
    RelationFamily(5, "probability", ("has_leadtime_probability", "has_cumulative_probability")),
    RelationFamily(6, "hazard_forecast", ("forecasts_hazard_at_horizon",)),
    RelationFamily(7, "hazard_observation", ("observes_hazard_at_horizon",)),
    RelationFamily(8, "threshold", ("has_threshold_status",)),
    RelationFamily(9, "additional_hazards", ("has_additional_hazard",)),
    RelationFamily(10, "operations", ("has_operation_status", "affects_vessel_handling")),
    RelationFamily(11, "impact", ("has_impact_prediction", "causes_operational_disruption")),
    RelationFamily(12, "recovery", ("has_recovery_status", "starts_recovery")),
    RelationFamily(13, "cross_horizon", ("forecast_updates_to", "intensifies_to", "changes_status_to")),
)

FAMILY_BY_NAME: Mapping[str, RelationFamily] = {f.name: f for f in RELATION_FAMILIES}
FAMILY_BY_INDEX: Mapping[int, RelationFamily] = {f.index: f for f in RELATION_FAMILIES}

CROSS_HORIZON_FAMILY = "cross_horizon"
"""Family 13. §A.3 — "family 13 operates between horizons". """

# canonical relation -> family. Built from the table so the two can never drift apart.
RELATION_TO_FAMILY: Mapping[str, str] = {
    relation: family.name for family in RELATION_FAMILIES for relation in family.relations
}

# §A.2 step 4 and §A.3 add two more relations that Table 3 does not list, both in family 13.
# §A.2 — "synthetic hyperedges (forecast_updates_to, changes_probability_to) are created to represent
# evolution". forecast_updates_to is already in family 13; changes_probability_to is not, so it is
# registered explicitly.
RELATION_TO_FAMILY = {**RELATION_TO_FAMILY, "changes_probability_to": CROSS_HORIZON_FAMILY}

# ---------------------------------------------------------------------------
# §A.1 — the alias map
# ---------------------------------------------------------------------------

# §A.1 — "All relation strings are normalized to this vocabulary via an alias mapping that absorbs
# variation in LLM-generated labels (e.g., closes_port → has_operation_status)."
#
# [PARTIALLY_SPECIFIED] The paper gives exactly one worked example (closes_port → has_operation_status)
# and does not publish the table. The entries below are hand-authored variations an extraction model
# realistically emits for the 13 families: singular/plural, tense, verb-object reorderings, and the
# domain synonyms the paper's own prose uses. Anything not covered falls through to fuzzy matching.
RELATION_ALIASES: Mapping[str, str] = {
    # --- 1 cyclone state -----------------------------------------------------
    "has_cyclone_status": "has_cyclone_state",
    "cyclone_state": "has_cyclone_state",
    "has_storm_state": "has_cyclone_state",
    "is_categorized_as": "has_category_state",
    "has_category": "has_category_state",
    "intensity_is": "has_category_state",
    "moves_toward": "has_motion",
    "motion_is": "has_motion",
    "moves": "has_motion",
    # --- 2 track & landfall --------------------------------------------------
    "forecast_track": "forecasts_track",
    "predicts_track": "forecasts_track",
    "track_forecast": "forecasts_track",
    "forecast_landfall": "forecasts_landfall",
    "predicts_landfall": "forecasts_landfall",
    "landfall_forecast": "forecasts_landfall",
    "landfall_prediction": "forecasts_landfall",
    # --- 3 timing ------------------------------------------------------------
    "has_time_to_landfall": "has_hours_to_landfall",
    "hours_to_landfall": "has_hours_to_landfall",
    "lead_time": "has_hours_to_landfall",
    "has_forecast_horizon": "has_forecast_window",
    "forecast_window": "has_forecast_window",
    # --- 4 advisory ----------------------------------------------------------
    "has_advisory_status": "has_warning_status",
    "is_under_watch": "has_watch_status",
    "watch_status": "has_watch_status",
    "is_under_warning": "has_warning_status",
    "warning_status": "has_warning_status",
    "advisory_status": "has_warning_status",
    "issues_warning": "has_warning_status",
    # --- 5 probability -------------------------------------------------------
    "has_lead_time_probability": "has_leadtime_probability",
    "leadtime_probability": "has_leadtime_probability",
    "cumulative_probability": "has_cumulative_probability",
    "has_cumulative_prob": "has_cumulative_probability",
    "changes_probability_to": "changes_probability_to",  # §A.2 — synthetic, family 13
    # --- 6 hazard forecast ---------------------------------------------------
    "forecasts_hazard": "forecasts_hazard_at_horizon",
    "hazard_forecast": "forecasts_hazard_at_horizon",
    "forecast_hazard": "forecasts_hazard_at_horizon",
    "predicts_hazard": "forecasts_hazard_at_horizon",
    # --- 7 hazard observation ------------------------------------------------
    "observes_hazard": "observes_hazard_at_horizon",
    "hazard_observation": "observes_hazard_at_horizon",
    "observes": "observes_hazard_at_horizon",
    # --- 8 threshold ---------------------------------------------------------
    "threshold_status": "has_threshold_status",
    "exceeds_threshold": "has_threshold_status",
    "has_threshold": "has_threshold_status",
    # --- 9 additional hazards ------------------------------------------------
    "additional_hazard": "has_additional_hazard",
    "has_hazard": "has_additional_hazard",
    # --- 10 operations -------------------------------------------------------
    "closes_port": "has_operation_status",  # §A.1 — the paper's own worked example
    "closes": "has_operation_status",
    "reopens_port": "has_operation_status",
    "reopens": "has_operation_status",
    "operation_status": "has_operation_status",
    "restricts_vessel_movements": "affects_vessel_handling",
    "suspends_vessel_operations": "affects_vessel_handling",
    "vessel_handling": "affects_vessel_handling",
    # --- 11 impact -----------------------------------------------------------
    "impact_prediction": "has_impact_prediction",
    "predicts_impact": "has_impact_prediction",
    "causes_delay": "causes_operational_disruption",
    "causes_disruption": "causes_operational_disruption",
    "disrupts": "causes_operational_disruption",
    # --- 12 recovery ---------------------------------------------------------
    "recovery_status": "has_recovery_status",
    "recovers": "has_recovery_status",
    "starts_recovery": "starts_recovery",
    "begins_recovery": "starts_recovery",
    # --- 13 cross-horizon ----------------------------------------------------
    "forecast_update": "forecast_updates_to",
    "updates_forecast": "forecast_updates_to",
    "intensifies": "intensifies_to",
    "strengthens_to": "intensifies_to",
    "changes_status": "changes_status_to",
    "status_changes_to": "changes_status_to",
    "weakens_to": "changes_status_to",
}

# ---------------------------------------------------------------------------
# §A.3 — the 11-stage precedence ordering
# ---------------------------------------------------------------------------

# §A.3 — "cyclone state → track → landfall → timing → advisory → probability → hazard → threshold →
#          operation → impact → recovery"
PRECEDENCE_PHASE_ORDER: tuple[str, ...] = (
    "cyclone_state",
    "track",
    "landfall",
    "timing",
    "advisory",
    "probability",
    "hazard",
    "threshold",
    "operation",
    "impact",
    "recovery",
)

_COVERAGE_NONE = ""  # sentinel: this relation does not map onto a member of S

# relation -> §A.3 precedence phase.
# [CONTRADICTION] §A.3's ordering names 11 stages while Table 3 defines 13 families, so families 7
# (hazard observation) and 9 (additional hazards) have no stage of their own; both are folded into
# "hazard", which is the stage they function as. Family 2 splits across two stages
# (forecasts_track → "track", forecasts_landfall → "landfall"), so the mapping is keyed by *relation*,
# not by family. Family 13 has no within-horizon position at all — §A.3's family-to-change rule places
# a change edge immediately after the within-horizon edges of the family it points back to.
RELATION_TO_PRECEDENCE_PHASE: Mapping[str, str] = {
    # 1
    "has_cyclone_state": "cyclone_state",
    "has_category_state": "cyclone_state",
    "has_motion": "cyclone_state",
    # 2 — the one family that spans two stages
    "forecasts_track": "track",
    "forecasts_landfall": "landfall",
    # 3
    "has_hours_to_landfall": "timing",
    "has_forecast_window": "timing",
    # 4
    "has_watch_status": "advisory",
    "has_warning_status": "advisory",
    # 5
    "has_leadtime_probability": "probability",
    "has_cumulative_probability": "probability",
    # 6, 7, 9 — folded into "hazard" (see the CONTRADICTION note above)
    "forecasts_hazard_at_horizon": "hazard",
    "observes_hazard_at_horizon": "hazard",
    "has_additional_hazard": "hazard",
    # 8
    "has_threshold_status": "threshold",
    # 10
    "has_operation_status": "operation",
    "affects_vessel_handling": "operation",
    # 11
    "has_impact_prediction": "impact",
    "causes_operational_disruption": "impact",
    # 12
    "has_recovery_status": "recovery",
    "starts_recovery": "recovery",
    # 13 — no within-horizon position; resolved at build time against the family it points back to
    "forecast_updates_to": _COVERAGE_NONE,
    "intensifies_to": _COVERAGE_NONE,
    "changes_status_to": _COVERAGE_NONE,
    "changes_probability_to": _COVERAGE_NONE,
}

PRECEDENCE_PHASE_RANK: Mapping[str, int] = {
    phase: rank for rank, phase in enumerate(PRECEDENCE_PHASE_ORDER)
}

# relation -> §A.6 Eq. 13 coverage phase, i.e. φ(e) restricted to S.
#
# §A.6 Eq. 13 — "Cov(γ) = |{φ(e_k)} ∩ S| / |S|". The PDF's definition of S is truncated at
# "S = {advisory, hazard_forecast, hazard_observation, operation_stat", while §3.2 glosses the same set
# as 5 stages ("advisory → hazard → operation → impact → recovery").
# [CONTRADICTION] The truncated equation-side text proves hazard_forecast and hazard_observation are
# *separate* members, so the 6-member set is used (retrieval.phase_set in configs/base.yaml). φ maps a
# relation onto one of those 6, or onto _COVERAGE_NONE when the relation is a purely structural
# precondition (cyclone state, track, timing, probability, threshold) that the paper never lists in S.
RELATION_TO_COVERAGE_PHASE: Mapping[str, str] = {
    # --- advisory ------------------------------------------------------------
    "has_watch_status": "advisory",
    "has_warning_status": "advisory",
    # --- hazard (forecast side) ---------------------------------------------
    "forecasts_hazard_at_horizon": "hazard_forecast",
    "has_additional_hazard": "hazard_forecast",
    # --- hazard (observation side) ------------------------------------------
    "observes_hazard_at_horizon": "hazard_observation",
    # --- operation -----------------------------------------------------------
    "has_operation_status": "operation_status",
    "affects_vessel_handling": "operation_status",
    # --- impact --------------------------------------------------------------
    "has_impact_prediction": "impact",
    "causes_operational_disruption": "impact",
    # --- recovery ------------------------------------------------------------
    "has_recovery_status": "recovery",
    "starts_recovery": "recovery",
    # --- everything else is not a member of S -------------------------------
    "has_cyclone_state": _COVERAGE_NONE,
    "has_category_state": _COVERAGE_NONE,
    "has_motion": _COVERAGE_NONE,
    "forecasts_track": _COVERAGE_NONE,
    "forecasts_landfall": _COVERAGE_NONE,
    "has_hours_to_landfall": _COVERAGE_NONE,
    "has_forecast_window": _COVERAGE_NONE,
    "has_leadtime_probability": _COVERAGE_NONE,
    "has_cumulative_probability": _COVERAGE_NONE,
    "has_threshold_status": _COVERAGE_NONE,
    "forecast_updates_to": _COVERAGE_NONE,
    "intensifies_to": _COVERAGE_NONE,
    "changes_status_to": _COVERAGE_NONE,
    "changes_probability_to": _COVERAGE_NONE,
}

# §A.3 — "advisory ≺ hazard forecasts, hazard assessments ≺ operational decisions, hazard assessments
#          ≺ impact predictions, and impact predictions ≺ recovery status", expressed as the four
# precedence-phase pairs those four rules connect.
CAUSAL_CHAIN_PHASE_RULES: tuple[tuple[str, str], ...] = (
    ("advisory", "hazard"),
    ("hazard", "operation"),
    ("hazard", "impact"),
    ("impact", "recovery"),
)

# ---------------------------------------------------------------------------
# §A.8 — reasoning tags shown on the "Reasoning:" line of the evidence block
# ---------------------------------------------------------------------------

# §A.8 — "Reasoning: within_horizon, hazard_to_operation, ..." — the ellipsis hides the full tag set.
# [PARTIALLY_SPECIFIED] The tags are derived from the provenance of the §A.3 rules that created each
# precedence edge, which is the only place the paper defines such relationships.
REASONING_TAGS: tuple[str, ...] = (
    "within_horizon",
    "cross_horizon",
    "advisory_to_hazard",
    "hazard_to_operation",
    "hazard_to_impact",
    "impact_to_recovery",
    "family_to_change",
)

# ---------------------------------------------------------------------------
# Attribute keys on ``a_e`` that the pipeline itself writes and reads
# ---------------------------------------------------------------------------

# Written by §A.2 step 4's cross-horizon synthesis, read by §A.3's family-to-change rule. Sharing the
# constants here rather than redeclaring them in each module is what stops the writer and the reader from
# drifting apart.
ATTR_ORIGIN_FAMILY = "origin_family"
ATTR_ORIGIN_HORIZONS = "origin_horizons"


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def normalize_relation_token(relation: str) -> str:
    """Lower-case, snake-case a raw extractor relation string so lookups are stable."""
    cleaned = relation.strip().lower()
    for ch in ("-", " ", "/", "."):
        cleaned = cleaned.replace(ch, "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def fuzzy_match_relation(relation: str, threshold: float) -> str | None:
    """§A.2 step 1 — "mapped to R via the alias table and fuzzy matching".

    Args:
        relation: An already-token-normalised relation string.
        threshold: Minimum similarity to accept, from ``hypergraph.normalization.fuzzy_threshold``.

    Returns:
        The canonical relation, or ``None`` when nothing clears ``threshold``.
    """
    candidates = list(RELATION_TO_FAMILY)
    matches = difflib.get_close_matches(relation, candidates, n=1, cutoff=threshold)
    return matches[0] if matches else None


def family_of(relation: str) -> str | None:
    """The §A.1 Table 3 family name for a canonical relation, or ``None`` if unknown."""
    return RELATION_TO_FAMILY.get(relation)


def family_index_of(relation: str) -> int:
    """The Table 3 family index (1-13) for a canonical relation.

    Used as the second §A.3 tie-breaker ("phase rank, family rank, lead time, text position").

    Returns:
        The 1-based family index, or ``99`` for a relation outside the vocabulary so it sorts last
        instead of raising mid-trajectory.
    """
    family = family_of(relation)
    if family is None:
        return 99
    return FAMILY_BY_NAME[family].index


def precedence_phase_of(relation: str) -> str | None:
    """The §A.3 precedence stage of a relation, or ``None`` for family-13 change edges.

    ``None`` is not an error: a cross-horizon change edge has no within-horizon stage, and §A.3's
    family-to-change rule positions it relative to the family it points back to instead.
    """
    phase = RELATION_TO_PRECEDENCE_PHASE.get(relation, _COVERAGE_NONE)
    return phase or None


def precedence_rank_of(relation: str) -> int:
    """The rank of a relation's §A.3 stage in :data:`PRECEDENCE_PHASE_ORDER`.

    Cross-horizon change edges get ``-1`` so they sort before every within-horizon stage; the caller is
    expected to reposition them with the family-to-change rule.
    """
    phase = precedence_phase_of(relation)
    if phase is None:
        return -1
    return PRECEDENCE_PHASE_RANK.get(phase, len(PRECEDENCE_PHASE_ORDER))


def coverage_phase_of(relation: str) -> str | None:
    """The §A.6 Eq. 13 coverage phase ``φ(e)`` of a relation, or ``None`` if it is not a member of S."""
    phase = RELATION_TO_COVERAGE_PHASE.get(relation, _COVERAGE_NONE)
    return phase or None


# The image of φ — the phases §A.6 Eq. 13 can actually observe. Derived from the table rather than
# restated so a new relation mapping cannot leave this set stale.
COVERAGE_PHASES: frozenset[str] = frozenset(
    phase for phase in RELATION_TO_COVERAGE_PHASE.values() if phase
)


def coverage_phase_reachable(phase: str) -> bool:
    """Whether any relation in ``R`` maps onto ``phase`` under §A.6 Eq. 13's ``φ``.

    A member of ``S`` that fails this test can never be added to ``{φ(e_k)}``, so it inflates the
    denominator of ``Cov`` without ever contributing to the numerator. Since ``S`` is a config value
    (``retrieval.phase_set``) and ``φ`` is baked into this table, the two can drift; the config validator
    uses this to refuse to start rather than let ρ silently lose part of its range.
    """
    return phase in COVERAGE_PHASES


def is_cross_horizon(relation: str) -> bool:
    """Whether a relation belongs to family 13 (§A.1 Table 3)."""
    return family_of(relation) == CROSS_HORIZON_FAMILY
