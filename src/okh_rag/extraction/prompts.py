"""§A.2 — the extraction prompt ``p_ext`` and its output schema.

§A.2 states what the prompt must make the model do:

    "The prompt instructs the model to segment the input into knowledge fragments (yielding s_e),
     recognize and type all entities (yielding V_e with τ_v and d_v), and assign relation labels from R
     with quantitative attributes. Output is constrained to a strict JSON schema matching the definitions
     above."

The paper never prints ``p_ext`` itself, and §3.1 notes the extraction machinery follows reference [20]
(HyperGraphRAG), which is where the actual prompt lives. This prompt is therefore authored from the
sentence above plus §A.1's field definitions; it is flagged ``[UNSPECIFIED]`` in
``REPRODUCTION_NOTES.md`` and every design decision below is traceable to a paper sentence.

Two deliberate schema choices:

* **Relations and entity types are enums.** §A.2 says "assign relation labels from R" and §A.1 says
  "τ_v ∈ T is a type from a domain-specific type system". An enum makes both constraints enforced by the
  API rather than hoped for, which is what lets the §A.2 step-1 alias table act as a *safety net* instead
  of the main mechanism.
* **``attributes`` is a list of ``{key, value}`` pairs, not a free-form object.** §A.1 defines
  ``a_e = {(k_j, u_j)}_{j=1..A}`` — literally a set of pairs — and strict structured-output mode rejects
  objects with unrestricted keys. The list form satisfies both.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from okh_rag.extraction.horizon import HorizonBlock
from okh_rag.schema import ENTITY_TYPE_REGISTRY
from okh_rag.vocabulary import RELATION_FAMILIES

# §A.1 Table 3, flattened to the leaf relations the model may emit. Derived from the vocabulary module so
# the prompt can never drift from the code that consumes the output.
RELATION_ENUM: tuple[str, ...] = tuple(
    sorted({rel for family in RELATION_FAMILIES for rel in family.relations})
)

# §A.1's type system, as far as the paper pins it down. See schema.ENTITY_TYPE_REGISTRY for the
# [PARTIALLY_SPECIFIED] note on the entries the paper does not name.
ENTITY_TYPE_ENUM: tuple[str, ...] = tuple(sorted(ENTITY_TYPE_REGISTRY))

EXTRACTION_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["facts"],
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["relation", "evidence", "entities", "attributes", "confidence"],
                "properties": {
                    "relation": {
                        "type": "string",
                        "enum": list(RELATION_ENUM),
                        "description": (
                            "The relation of §A.1 Table 3 that best describes this interaction. "
                            "Use the relation whose meaning matches the interaction, not the closest "
                            "string."
                        ),
                    },
                    "evidence": {
                        "type": "string",
                        "description": (
                            "s_e: one sentence stating the interaction in natural language, copying "
                            "quantities and thresholds verbatim from the source text. This string is "
                            "what gets embedded, so it must carry the full semantic content."
                        ),
                    },
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["name", "type", "description", "confidence"],
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": (
                                        "n_v. MUST appear verbatim inside the evidence string "
                                        "(§A.1: n_v ⊆ s_e)."
                                    ),
                                },
                                "type": {
                                    "type": "string",
                                    "enum": list(ENTITY_TYPE_ENUM),
                                    "description": (
                                        "τ_v. Persistent objects (PORT, CYCLONE) run through the whole "
                                        "scenario; transient states (CYCLONE_STATE, OPERATION_STATUS, "
                                        "HAZARD_FORECAST, ...) hold at one horizon; HORIZON_TIME is a "
                                        "temporal anchor and is added automatically, so do not emit it."
                                    ),
                                },
                                "description": {
                                    "type": "string",
                                    "description": "d_v. One short clause explaining the entity's role here.",
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "c_v in (0, 1]. Extraction certainty.",
                                },
                            },
                        },
                        "description": (
                            "V_e. At least two entities, including every entity the relation connects."
                        ),
                    },
                    "attributes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["key", "value"],
                            "properties": {
                                "key": {"type": "string"},
                                "value": {"type": "string"},
                            },
                        },
                        "description": (
                            "a_e. Quantitative and categorical properties, e.g. wind speed, probability, "
                            "category number. Emit an empty list when the interaction has none. Values "
                            "must be strings."
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "description": "c_e in (0, 1]. Confidence that this fact is correct and grounded.",
                    },
                },
            },
        }
    },
}

EXTRACTION_SCHEMA_NAME = "okh_rag_horizon_facts"

SYSTEM_PROMPT = """You extract knowledge fragments from tropical-cyclone port-impact reports. \
You reconstruct knowledge as n-ary relational facts: a fact is a single interaction between two or more \
entities, described in one natural-language sentence, not a binary triple.

Rules you must follow:

1. SEGMENTATION. One fact = one interaction the text asserts. Do not merge a hazard forecast and an \
operational decision into one fact. Do not split one interaction into several facts.

2. EVIDENCE. The `evidence` field is the fact's meaning in one sentence. Copy every number, threshold, \
level and place name exactly as written in the source. Never paraphrase a quantity.

3. ENTITIES. Every fact needs at least two entities. An entity `name` must appear verbatim inside that \
fact's `evidence` string. Assign each entity the most specific type available. Use PORT and CYCLONE only \
for the port itself and the storm itself; use transient types for anything that holds at one point in \
time. Do not emit HORIZON_TIME entities — the horizon is attached automatically.

4. RELATIONS. Pick the relation whose meaning matches the interaction. Use has_warning_status for \
watches/warnings, forecasts_hazard_at_horizon for predicted hazards, observes_hazard_at_horizon for \
observed ones, has_operation_status for port/vessel operating decisions, has_impact_prediction for \
predicted consequences, and the cross-horizon relations only when the text explicitly says a value \
changed or was updated from an earlier state.

5. ATTRIBUTES. Record quantitative and categorical details as key/value string pairs.

6. GROUNDING. Every fact must be supported by the supplied text. If the text does not support a fact, \
omit it. Output an empty `facts` list rather than inventing content."""


def build_user_prompt(block: HorizonBlock, *, storm: str, port: str) -> str:
    """Compose the per-horizon user message.

    The scenario header is included so the model can resolve domain references that the horizon block
    itself leaves implicit ("the port", "the storm") — the same reason §A.2 canonicalises entity ids per
    storm and port in post-processing.

    Args:
        block: The horizon block to extract from.
        storm: Storm name, e.g. ``ARTHUR``.
        port: Port name, e.g. ``Baltimore, MD``.

    Returns:
        The user message. The horizon label is stated explicitly so the model can prefer horizon-local
        readings of ambiguous phrases.
    """
    return (
        f"Scenario: storm {storm}, port {port}. "
        f"This text is the {block.horizon_label} forecast snapshot.\n\n"
        f"--- BEGIN {block.horizon_label} TEXT ---\n"
        f"{block.text}\n"
        f"--- END {block.horizon_label} TEXT ---\n\n"
        "Extract every n-ary interaction as a fact."
    )


def build_messages(block: HorizonBlock, *, storm: str, port: str) -> list[dict[str, str]]:
    """Assemble the full message list for one extraction call."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(block, storm=storm, port=port)},
    ]


def relation_legend() -> Sequence[tuple[str, tuple[str, ...]]]:
    """``(family_name, relations)`` pairs for the 13 families, for documentation and logs."""
    return tuple((f.name, f.relations) for f in RELATION_FAMILIES)
