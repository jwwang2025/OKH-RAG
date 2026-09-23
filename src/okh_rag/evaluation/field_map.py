"""``data/CONTRACT.md`` §3.2 — the source-field to §A.1 Table 3 relation table, as code.

§A.2 extracts a hypergraph from a scenario document by asking a model for facts and then
normalising their relation labels onto Table 3. CyPortQA ships that same scenario as a
*structured* encoding, so for this dataset the assignment is not free-form: every field of
``Encoded_senario.json`` implies one relation, and §3.2 tabulates which. §3.2 states why the
table is worth having:

    "The mapping below is the specification the hypergraph builder must implement, and it is
     what makes LLM extraction scoreable rather than taken on faith — the extraction output
     can be compared against the family each field implies."

This module is that table, and nothing else. It is deliberately free of imports from
:mod:`okh_rag.data` and :mod:`okh_rag.retrieval`: it is a leaf so that the spec can be tested
without loading a corpus, and so a change to the spec shows up as a test failure rather than
as a silently different gold set.

**Why the table is the oracle's unit.** A question's answer is a named field (§5 of
``CONTRACT.md``), so "the hyperedges this question needs" could be read as "the hyperedges
built from *that exact field*". That reading only works on a hypergraph that records which
field each fact came from, which the LLM-extracted hypergraph does not — §A.2 asks the model
for evidence sentences, not for source offsets. Keying on the *relation* instead makes one
gold-set definition serve both hypergraphs:

* on the reference hypergraph of :mod:`okh_rag.evaluation.reference_graph` — one hyperedge
  per field, so a relation resolves to exactly the fields it covers;
* on an extracted hypergraph — a relation resolves to whatever the model extracted under that
  label, which is the quantity §4.1's end-to-end accuracy cannot isolate.

The relation is therefore the stable identity and the field is the implementation detail, which
is also the reading §3.2's own wording supports ("compared against the family each field
implies").

**Three shapes a declared field can take**, all of which occur in the 48 templates and all of
which have to resolve, or the oracle would silently score a subset of the benchmark:

===========================  ==========================================================
shape                        example
===========================  ==========================================================
leaf                         ``weather_observation.under_watch`` → ``has_watch_status``
offset wildcard              ``operation.<offset>.Condition`` → ``has_operation_status``
sub-dict wildcard / container ``weather_observation.gale_probability`` → (its five children)
===========================  ==========================================================

The third shape is Q33's: its template names the ``gale_probability`` mapping rather than any
one key inside it, and every key inside it is a lead-time probability, so the container
resolves to the union of its children's relations — here a single one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, Mapping

from okh_rag.vocabulary import RELATION_TO_FAMILY

__all__ = [
    "ENTITY_FIELD_PATHS",
    "FIELD_TO_RELATION",
    "MULTI_SEGMENT_WILDCARDS",
    "FieldKind",
    "FieldResolution",
    "paths_for_relation",
    "relation_family",
    "relations_for_path",
]

# ---------------------------------------------------------------------------
# ``CONTRACT.md`` §3.2, transcribed
# ---------------------------------------------------------------------------

FIELD_TO_RELATION: Final[Mapping[str, str]] = {
    # --- family 1: cyclone_state ------------------------------------------
    "cyclone_condition_and_forecast.current_location": "has_cyclone_state",
    "cyclone_condition_and_forecast.coordinates": "has_cyclone_state",
    "cyclone_condition_and_forecast.category": "has_category_state",
    "cyclone_condition_and_forecast.motion.direction": "has_motion",
    "cyclone_condition_and_forecast.motion.speed": "has_motion",
    # --- family 2: track_landfall -----------------------------------------
    "cyclone_condition_and_forecast.trajectory": "forecasts_track",
    "weather_observation.within_uncertainty_cone": "forecasts_track",
    "cyclone_condition_and_forecast.expected_landfall_location": "forecasts_landfall",
    "cyclone_condition_and_forecast.expected_landfall_date": "forecasts_landfall",
    "weather_observation.impacted_coast": "forecasts_landfall",
    "weather_observation.expected_closest_date_to_port": "forecasts_landfall",
    "weather_observation.expected_leaving_port_date": "forecasts_landfall",
    # --- family 3: timing --------------------------------------------------
    "cyclone_condition_and_forecast.hours_to_landfall": "has_hours_to_landfall",
    "cyclone_condition_and_forecast.hours_to_offshore": "has_hours_to_landfall",
    "cyclone_condition_and_forecast.hours_of_strike": "has_hours_to_landfall",
    "weather_observation.forecast_window": "has_forecast_window",
    # --- family 4: advisory ------------------------------------------------
    "weather_observation.under_watch": "has_watch_status",
    "weather_observation.watch_coast": "has_watch_status",
    "weather_observation.under_warning": "has_warning_status",
    "weather_observation.warning_coast": "has_warning_status",
    # --- family 5: probability --------------------------------------------
    "weather_observation.gale_probability.*": "has_leadtime_probability",
    "weather_observation.time_to_34kt_max": "has_leadtime_probability",
    "weather_observation.time_to_34kt_non_zero": "has_leadtime_probability",
    "weather_observation.time_to_50kt_max": "has_leadtime_probability",
    "weather_observation.culmulative_wind_34kt": "has_cumulative_probability",
    "weather_observation.culmulative_wind_50kt": "has_cumulative_probability",
    "weather_observation.34kt_max": "has_cumulative_probability",
    # --- family 6: hazard_forecast ----------------------------------------
    "weather_observation.wind": "forecasts_hazard_at_horizon",
    "weather_observation.storm_surge": "forecasts_hazard_at_horizon",
    "weather_observation.rainfall": "forecasts_hazard_at_horizon",
    "weather_observation.first_hazard": "forecasts_hazard_at_horizon",
    # --- family 7: hazard_observation -------------------------------------
    "weather_observation.closest_wind_data_location": "observes_hazard_at_horizon",
    # --- family 8: threshold ----------------------------------------------
    "weather_observation.rain_fall_above_4_inch": "has_threshold_status",
    "weather_observation.storm_surge_above_3_feet": "has_threshold_status",
    # --- family 9: additional_hazards -------------------------------------
    "weather_observation.other_hazards": "has_additional_hazard",
    # --- family 10: operations --------------------------------------------
    "operation.*.Condition": "has_operation_status",
    "operation.*.VesselOps": "affects_vessel_handling",
    "operation.*.OceanVesselOps": "affects_vessel_handling",
    "operation.*.FacilityOps": "affects_vessel_handling",
    # --- family 11: impact -------------------------------------------------
    "impact_prediction.start_impact_date": "has_impact_prediction",
    "impact_prediction.max_impact_date": "has_impact_prediction",
    "impact_prediction.end_impact_date": "has_impact_prediction",
    "impact_prediction.max_impact": "has_impact_prediction",
    "impact_prediction.day_of_impact": "causes_operational_disruption",
    "impact_prediction.total_impact_value": "causes_operational_disruption",
    # --- family 12: recovery ----------------------------------------------
    "impact_prediction.day_of_recover": "has_recovery_status",
}
"""``static field path -> canonical relation``, one row per field §3.2 names.

``*`` matches exactly one path segment. The two wildcards are §3.2's own:

* ``operation.*.<field>`` — the ``operation`` block is keyed by a *relative* offset
  (``now``, ``in_12h``, …), so §3.2 writes ``operation.<offset>.Condition``. The wildcard is
  what lets one row cover all ten offsets.
* ``gale_probability.<key>`` — §3.2 writes this one as ``gale_probability.*``. The release's
  keys carry a space (``"in 12h"``), which is why segment-wise matching rather than a string
  prefix is used: ``weather_observation.gale_probability.in 12h`` has to match
  ``weather_observation.gale_probability.*``, and it does, because ``*`` matches the whole
  segment ``in 12h``.

**Family 13 has no entry, and must not get one.** §3.2 gives it "*no source field* — synthesised
by §A.2 step 4". A cross-horizon change edge therefore can never be the answer to a question,
so it can never be gold, and it enters the oracle only as pool competition. Adding a field to
family 13 would make a synthetic edge a gold answer, which no template asks for.

**§3.2's table is not exhaustive over the encoding.** ``cyclone_condition_and_forecast.name``
is a real field of every block and appears in no row above — correctly, because it *is* the
storm's persistent identity, which §3.1 lists under persistent objects rather than under a
relation. :func:`relations_for_path` reports such paths as :data:`FieldKind.UNKNOWN` rather
than guessing, and :func:`~okh_rag.evaluation.reference_graph.build_reference_hypergraph`
counts them so the gap is visible instead of silently absent from the corpus.
"""

ENTITY_FIELD_PATHS: Final[frozenset[str]] = frozenset({"port.name", "cyclone.name"})
"""Declared fields that name a *persistent object* rather than a §A.1 Table 3 relation.

Q16 and Q19 answer ``{senario.port.name}``. §3.1 places ``port.name`` under persistent objects,
and §A.1 says persistent objects "serve as anchors" in ``V_e`` rather than being the subject of
a relation — which is why §3.2's table has no row for them and why inventing one would be
wrong. Their evidence requirement is satisfied by the *entity*, not by a hyperedge:
:class:`~okh_rag.evaluation.oracle.GoldEvidence` records the requirement so a missing port
entity is still reported as an unanswerable question instead of as a retrieval miss.
"""

MULTI_SEGMENT_WILDCARDS: Final[tuple[str, ...]] = tuple(
    key for key in FIELD_TO_RELATION if "*" in key
)
"""The keys carrying a ``*`` segment, exposed so a test can assert the set did not grow silently."""


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


class FieldKind(str, Enum):
    """What a template's declared field path turned out to be."""

    RELATION = "relation"
    """It names a §A.1 Table 3 relation's source field; the gold set is relation-defined."""

    ENTITY = "entity"
    """It names a persistent object (see :data:`ENTITY_FIELD_PATHS`); the requirement is an entity."""

    UNKNOWN = "unknown"
    """No row of §3.2 covers it. Reported, never guessed at — see the module docstring."""


@dataclass(frozen=True)
class FieldResolution:
    """The §3.2 lookup for one declared field path, with the row that produced it."""

    path: str
    kind: FieldKind
    relations: tuple[str, ...] = ()
    matched_key: str | None = None

    @property
    def relation(self) -> str | None:
        """The single relation this field implies, or ``None`` when there is not exactly one.

        Callers that can handle a multi-relation container should read
        :attr:`relations` directly; this accessor exists because 47 of the 48 templates
        resolve to exactly one relation and the oracle reads better when it says so.
        """
        return self.relations[0] if len(self.relations) == 1 else None

    @property
    def is_resolved(self) -> bool:
        """True when the path maps onto the §A.1 vocabulary or a persistent object."""
        return self.kind is not FieldKind.UNKNOWN

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "kind": self.kind.value,
            "relations": list(self.relations),
            "matched_key": self.matched_key,
        }


def _segments_match(path_parts: list[str], key_parts: list[str]) -> bool:
    """Whether every segment of ``key_parts`` matches ``path_parts``, ``*`` matching anything."""
    if len(path_parts) != len(key_parts):
        return False
    return all(k == "*" or k == p for p, k in zip(path_parts, key_parts))


def relations_for_path(path: str) -> FieldResolution:
    """Resolve a template-declared dotted field path against §3.2.

    The lookup runs in four passes, in this order:

    1. **exact** — the common case, and the one that must stay exact: a typo in a path has to
       fall through rather than fuzzy-match onto a neighbour and manufacture a gold set.
    2. **segment wildcard** — ``operation.in_12h.Condition`` against ``operation.*.Condition``,
       and ``weather_observation.gale_probability.in 12h`` against
       ``weather_observation.gale_probability.*``.
    3. **container** — the path names a mapping rather than a leaf, so it resolves to the union
       of the relations of every row beneath it. This is Q33's shape.
    4. otherwise :data:`FieldKind.UNKNOWN`.

    Args:
        path: A static field path such as ``weather_observation.under_watch``, or a concrete
            one after the record's relative horizon has been substituted
            (``operation.in_24h.VesselOps``). Both are accepted, because
            :meth:`~okh_rag.data.cyportqa.Template.static_prefix` hands over the first kind and
            :func:`~okh_rag.data.groundtruth.expected_path` the second.

    Returns:
        A :class:`FieldResolution`. Never raises: an unresolvable path is a finding about the
        spec's coverage, and the oracle counts it rather than aborting a run over it.
    """
    if path in ENTITY_FIELD_PATHS:
        return FieldResolution(path=path, kind=FieldKind.ENTITY)

    exact = FIELD_TO_RELATION.get(path)
    if exact is not None:
        return FieldResolution(path=path, kind=FieldKind.RELATION, relations=(exact,), matched_key=path)

    path_parts = path.split(".")
    for key, relation in FIELD_TO_RELATION.items():
        key_parts = key.split(".")
        if len(key_parts) != len(path_parts):
            continue
        if _segments_match(path_parts, key_parts):
            return FieldResolution(
                path=path, kind=FieldKind.RELATION, relations=(relation,), matched_key=key
            )

    # Container: the path is a strict prefix of one or more rows, so it names the mapping those
    # rows live in. Order is the table's declaration order, deduplicated — a container whose
    # children share a relation must not report it twice, or the caller sees "two relations"
    # where the spec says one.
    prefix = path + "."
    children = [key for key in FIELD_TO_RELATION if key.startswith(prefix)]
    if children:
        relations = tuple(dict.fromkeys(FIELD_TO_RELATION[key] for key in children))
        return FieldResolution(
            path=path,
            kind=FieldKind.RELATION,
            relations=relations,
            matched_key=children[0] if len(children) == 1 else None,
        )

    return FieldResolution(path=path, kind=FieldKind.UNKNOWN)


def paths_for_relation(relation: str) -> tuple[str, ...]:
    """The inverse map — every §3.2 field that implies ``relation``.

    Used by the audit to answer "which of §3.2's rows does no template ever ask for?", which is
    the question that tells a reader how much of the extracted hypergraph the oracle can score.
    """
    return tuple(key for key, value in FIELD_TO_RELATION.items() if value == relation)


def relation_family(relation: str) -> str | None:
    """The §A.1 Table 3 family name for a relation, or ``None`` when it is off-vocabulary.

    Thin, but not gratuitous: it is the one place the oracle reaches into
    :mod:`okh_rag.vocabulary`, so a report can group templates by family without importing the
    vocabulary module itself.
    """
    return RELATION_TO_FAMILY.get(relation)
