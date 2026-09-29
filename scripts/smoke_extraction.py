"""Ad-hoc smoke test for the §A.2 extraction path. Not part of the deliverable test suite.

Run from the repo root:
    D:/Anaconda/python.exe -m scripts.smoke_extraction
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from okh_rag.config import load_config  # noqa: E402
from okh_rag.extraction import (  # noqa: E402
    ScenarioDocument,
    extract_hypergraph_from_document,
    is_grounded,
    locate_evidence,
)
from okh_rag.extraction.horizon import HorizonBlock, split_horizons  # noqa: E402
from okh_rag.llm.base import Usage  # noqa: E402
from okh_rag.ordering.precedence import build_precedence, canonical_order, phase_sequence  # noqa: E402
from okh_rag.schema import EntityCategory  # noqa: E402

DOC = """Scenario ARTHUR / Baltimore, MD

T-72 hours before expected landfall:
Hurricane ARTHUR is a Category 2 storm with maximum sustained winds near 100 mph.
The Port of Baltimore is operating under Condition YANKEE.
A gale warning is in effect for the port of Baltimore.

T-24 hours before expected landfall:
Hurricane ARTHUR has strengthened to a Category 3 storm with winds of 120 mph.
The Port of Baltimore is operating under Condition ZULU.
A gale warning is in effect for the port of Baltimore.
A storm surge of 6 feet is forecast for the port of Baltimore.
"""

# Deliberately exercises: a duplicate fact across blocks that must NOT collapse (different horizon),
# an alias relation ("closes_port"), an ungrounded entity ("Port of Norfolk"), a 1-entity fact that
# must be dropped, and an off-vocabulary relation that must be dropped in step 1.
BLOCK_FACTS: dict[int, list[dict]] = {
    72: [
        {
            "relation": "has_category_state",
            "evidence": "Hurricane ARTHUR is a Category 2 storm with maximum sustained winds near 100 mph.",
            "entities": [
                {"name": "Hurricane ARTHUR", "type": "CYCLONE", "description": "the storm", "confidence": 0.9},
                {"name": "Category 2", "type": "CYCLONE_STATE", "description": "intensity", "confidence": 0.8},
            ],
            "attributes": [{"key": "wind_mph", "value": "100"}],
            "confidence": 0.95,
        },
        {
            "relation": "has_operation_status",
            "evidence": "The Port of Baltimore is operating under Condition YANKEE.",
            "entities": [
                {"name": "Port of Baltimore", "type": "PORT", "description": "the port", "confidence": 0.99},
                {"name": "Condition YANKEE", "type": "OPERATION_STATUS", "description": "status", "confidence": 0.9},
            ],
            "attributes": [],
            "confidence": 0.9,
        },
        {
            "relation": "has_warning_status",
            "evidence": "A gale warning is in effect for the port of Baltimore.",
            "entities": [
                {"name": "Port of Baltimore", "type": "PORT", "description": "the port", "confidence": 0.99},
                {"name": "gale warning", "type": "ADVISORY_STATUS", "description": "advisory", "confidence": 0.85},
                # 'Port of Norfolk' is NOT in the evidence -> must be dropped as ungrounded
                {"name": "Port of Norfolk", "type": "PORT", "description": "wrong port", "confidence": 0.4},
            ],
            "attributes": [],
            "confidence": 0.85,
        },
        {
            "relation": "forecasts_hazard_at_horizon",
            "evidence": "A storm surge of 4 feet is forecast for the port of Baltimore.",
            "entities": [
                {"name": "Port of Baltimore", "type": "PORT", "description": "the port", "confidence": 0.99},
                {"name": "storm surge of 4 feet", "type": "HAZARD_FORECAST", "description": "hazard", "confidence": 0.8},
            ],
            "attributes": [{"key": "surge_ft", "value": "4"}],
            "confidence": 0.8,
        },
        {
            "relation": "totally_made_up_relation",
            "evidence": "Something unsupported happens here.",
            "entities": [
                {"name": "Something", "type": "PORT", "description": "x", "confidence": 0.5},
                {"name": "unsupported", "type": "PORT", "description": "y", "confidence": 0.5},
            ],
            "attributes": [],
            "confidence": 0.5,
        },
        {
            "relation": "has_operation_status",
            "evidence": "Only one entity here.",
            "entities": [
                {"name": "Only one entity here", "type": "PORT", "description": "z", "confidence": 0.5},
            ],
            "attributes": [],
            "confidence": 0.5,
        },
    ],
    24: [
        {
            "relation": "has_category_state",
            "evidence": "Hurricane ARTHUR has strengthened to a Category 3 storm with winds of 120 mph.",
            "entities": [
                {"name": "Hurricane ARTHUR", "type": "CYCLONE", "description": "the storm", "confidence": 0.9},
                {"name": "Category 3", "type": "CYCLONE_STATE", "description": "intensity", "confidence": 0.85},
            ],
            "attributes": [{"key": "wind_mph", "value": "120"}],
            "confidence": 0.95,
        },
        {
            "relation": "has_operation_status",
            "evidence": "The Port of Baltimore is operating under Condition ZULU.",
            "entities": [
                {"name": "Port of Baltimore", "type": "PORT", "description": "the port", "confidence": 0.99},
                {"name": "Condition ZULU", "type": "OPERATION_STATUS", "description": "status", "confidence": 0.9},
            ],
            "attributes": [],
            "confidence": 0.92,
        },
        # exact duplicate of the T-72 advisory fact (same relation + entity set + evidence) -> must
        # survive as a *separate* hyperedge, because its horizon differs and step 2 splits the states.
        {
            "relation": "has_warning_status",
            "evidence": "A gale warning is in effect for the port of Baltimore.",
            "entities": [
                {"name": "Port of Baltimore", "type": "PORT", "description": "the port", "confidence": 0.99},
                {"name": "gale warning", "type": "ADVISORY_STATUS", "description": "advisory", "confidence": 0.85},
            ],
            "attributes": [],
            "confidence": 0.85,
        },
        # writes an ALIAS relation name, exercising step 1's alias table
        {
            "relation": "closes_port",
            "evidence": "The Port of Baltimore is closed to all vessel traffic.",
            "entities": [
                {"name": "Port of Baltimore", "type": "PORT", "description": "the port", "confidence": 0.99},
                {"name": "closed to all vessel traffic", "type": "OPERATION_STATUS", "description": "closure", "confidence": 0.9},
            ],
            "attributes": [],
            "confidence": 0.9,
        },
        {
            "relation": "forecasts_hazard_at_horizon",
            "evidence": "A storm surge of 6 feet is forecast for the port of Baltimore.",
            "entities": [
                {"name": "Port of Baltimore", "type": "PORT", "description": "the port", "confidence": 0.99},
                {"name": "storm surge of 6 feet", "type": "HAZARD_FORECAST", "description": "hazard", "confidence": 0.85},
            ],
            "attributes": [{"key": "surge_ft", "value": "6"}],
            "confidence": 0.85,
        },
    ],
}


class StubBackend:
    """Returns the canned facts for the horizon whose block text it is asked about."""

    def __init__(self) -> None:
        self.usage = Usage()
        self.calls: list[int] = []

    def complete_text(self, messages, *, model, temperature):  # pragma: no cover - unused here
        raise NotImplementedError

    def complete_json(self, messages, *, model, temperature, schema, schema_name):
        prompt = messages[-1]["content"]
        self.usage.calls += 1
        self.usage.prompt_tokens += 500
        self.usage.completion_tokens += 300
        for offset, facts in BLOCK_FACTS.items():
            if f"T-{offset} forecast snapshot" in prompt:
                self.calls.append(offset)
                return {"facts": facts}
        return {"facts": []}

    def embed(self, texts):  # pragma: no cover - unused here
        return np.zeros((len(texts), 1536), dtype="float32")

    @property
    def embedding_dim(self) -> int:
        return 1536


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="okh_smoke_"))
    cfg = load_config("configs/base.yaml")
    cfg.paths = cfg.paths.resolve(root)
    cfg.llm = cfg.llm.resolve(root)

    blocks = split_horizons(DOC, group_id="ARTHUR:port_baltimore", cfg=cfg.hypergraph.horizon)
    doc = ScenarioDocument(
        group_id="ARTHUR:port_baltimore",
        storm="ARTHUR",
        port="Baltimore, MD",
        year=2026,
        blocks=tuple(blocks),
    )

    print("== grounding helper ==")
    for name, ev, want in [
        ("Hurricane ARTHUR", "Hurricane ARTHUR is a Category 2 storm", True),
        ("Hurricane ARTHUR", "the storm strengthened", False),
        ("Port of Baltimore", "the port of Baltimore is closed", True),
        ("Port of Norfolk", "the port of Baltimore is closed", False),
    ]:
        got = is_grounded(name, ev)
        print(f"  {name!r:22} -> {got}  (want {want})  {'OK' if got == want else 'FAIL'}")

    print("\n== locate_evidence ==")
    b = blocks[1]
    off = locate_evidence(b.text, b.char_offset, "The Port of Baltimore is operating under Condition ZULU.")
    print(f"  located at {off}, block starts at {b.char_offset}: "
          f"{'inside block' if off >= b.char_offset else 'FELL BACK (ok if reworded)'}")

    print("\n== extraction ==")
    backend = StubBackend()
    result = extract_hypergraph_from_document(doc, backend=backend, cfg=cfg)
    hg, stats = result.hypergraph, result.stats

    print(json.dumps(stats.as_dict(), indent=2, ensure_ascii=False))
    print("usage:", result.usage.as_dict())
    print("blocks queried (lead-time order):", backend.calls)

    print("\n== entities ==")
    for e in sorted(hg.entities.values(), key=lambda x: (x.category.value, x.entity_id)):
        print(f"  [{e.category.value:10}] {e.entity_id}")

    print("\n== hyperedges ==")
    for edge in hg.hyperedges.values():
        print(f"  {edge.relation:26} |V_e|={edge.cardinality} T={edge.horizon_hours} pos={edge.text_position}")
        print(f"      {edge.evidence[:96]}")
        for eid in edge.entity_ids:
            print(f"        - {eid}")

    print("\n== invariants ==")
    failures = []

    def check(name, ok, detail=""):
        print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            failures.append(name)

    check("no orphan entities",
          all(e in {i for d in hg.hyperedges.values() for i in d.entity_ids} for e in hg.entities))

    check("persistent entities carry no horizon",
          all(e.entity_id.endswith(":none") for e in hg.entities.values()
              if e.category is EntityCategory.PERSISTENT))

    anchors = [e for e in hg.entities.values() if e.category is EntityCategory.TEMPORAL]
    check("one temporal anchor per horizon", len(anchors) == 2, f"got {len(anchors)}")

    check("every horizon-grounded hyperedge has one anchor",
          all(sum(1 for i in d.entity_ids
                  if hg.entities[i].category is EntityCategory.TEMPORAL) == 1
              for d in hg.hyperedges.values() if d.horizon_hours is not None))

    check("ungrounded entity dropped", stats.entities_dropped_ungrounded == 1)
    check("off-vocabulary relation dropped", stats.dropped_unresolved_relation == 1,
          str(dict(stats.unresolved_relations)))
    check("1-entity fact dropped", stats.dropped_small_entity_set == 1)
    check("alias 'closes_port' resolved onto a Table 3 relation",
          any(e.relation == "has_operation_status" and "closed to all vessel traffic" in e.evidence
              for e in hg.hyperedges.values()))

    storm_ids = {e.entity_id for e in hg.entities.values() if e.name.upper().startswith("HURRICANE")}
    check("storm collapsed to ONE cross-horizon entity", len(storm_ids) == 1, str(sorted(storm_ids)))

    # Only within-horizon edges: a synthesised change edge quotes the source evidence verbatim into its
    # own evidence, so a plain text search over the graph matches both. That is expected.
    gale = [e for e in hg.hyperedges.values()
            if e.horizon_hours is not None and "gale warning" in e.evidence]
    check("same-wording advisory kept once per horizon", len(gale) == 2, f"got {len(gale)}")

    changes = [e for e in hg.hyperedges.values() if e.horizon_hours is None]
    check("cross-horizon edges synthesised", len(changes) > 0, f"got {len(changes)}")
    check("change edges declare their origin family",
          all(e.attributes.get("origin_family") and e.attributes.get("origin_horizons") for e in changes),
          str([e.attributes for e in changes]))
    check("fan-out cap respected (<= #src per family/horizon pair)",
          len(changes) <= len([e for e in hg.hyperedges.values() if e.horizon_hours is not None]))

    check("all |V_e| >= 2", all(e.cardinality >= 2 for e in hg.hyperedges.values()))

    print(f"\n== §A.3 precedence ==")
    graph = build_precedence(hg, cfg.precedence)
    ordering = canonical_order(hg, graph)
    ordered_ids = ordering[doc.group_id]
    print(f"  precedence edges: {len(graph)}")
    print("  canonical phase sequence:", phase_sequence(hg, ordered_ids))
    for rank, hid in enumerate(ordered_ids, start=1):
        edge = hg.hyperedges[hid]
        print(f"    {rank}. {edge.relation:28} T={edge.horizon_hours}  {edge.evidence[:56]}")

    pos = {hid: i for i, hid in enumerate(ordered_ids)}

    def positions(pred) -> list[int]:
        return [pos[e.hyperedge_id] for e in hg.hyperedges.values() if pred(e)]

    check("precedence relation is non-empty", len(graph) > 0)
    check("canonical order covers every hyperedge", len(ordered_ids) == len(hg.hyperedges))
    check("canonical order is a permutation",
          len(set(ordered_ids)) == len(ordered_ids) == len(hg.hyperedges))

    def at(horizon, relation):
        return positions(lambda e: e.horizon_hours == horizon and e.relation == relation)

    check("§A.3 within-horizon: cyclone_state < advisory < operation at T-72",
          max(at(72, "has_category_state")) < min(at(72, "has_warning_status"))
          < min(at(72, "has_operation_status")))

    check("§A.3 cross-horizon: T-72 state precedes its T-24 counterpart",
          min(at(72, "has_category_state")) < min(at(24, "has_category_state")))

    change_edges = [e for e in hg.hyperedges.values() if e.horizon_hours is None]
    check("§A.3 family-to-change: each change edge follows its origin family's source state",
          all(pos[e.hyperedge_id] > min(at(int(e.attributes["origin_horizons"].split("->")[0][2:]),
                                          {"advisory": "has_warning_status",
                                           "operations": "has_operation_status",
                                           "cyclone_state": "has_category_state",
                                           "hazard_forecast": "forecasts_hazard_at_horizon"}
                                          [e.attributes["origin_family"]]))
              for e in change_edges))

    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
