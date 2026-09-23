"""§3.3 / §A.8 — rendering retrieved evidence into the generator's context window.

§3.3 states the design constraint before it states the format: *"Concatenating hyperedge texts into a
flat context window would collapse ordered trajectories back into unordered sets."* So the job of this
module is not to flatten evidence into a string — it is to serialise the **structure** that retrieval
recovered, in a layout the generator can read back out.

§A.8 fixes that layout::

    [Step k] [T-XX] [phase=...] [family=...]
    Relation: r_e
    Evidence: s_e
    Reasoning: within_horizon, hazard_to_operation, ...
    Entities: entity1 [type]; entity2 [type]; ...

and separates two things that are easy to conflate:

* the **step** line (ordering cue, horizon, phase, family — temporal and logical context),
* the **reasoning** line (causal-pathway participation), which is *derived*, not stored.

The reasoning line is where this module has to make a decision the paper leaves open. §A.8 gives one
example — ``within_horizon, hazard_to_operation`` — and says only that the tags *"indicate causal
pathway participation"*. Both tags in that example are relations **between adjacent steps**, so the
tags are computed here from the step's neighbours rather than from the step alone: a step whose
neighbour is at the same horizon contributes ``within_horizon``, and a step following a different
causal phase contributes ``<from>_to_<to>``. See :func:`reasoning_tags` for the exact rule and the
alternatives it rules out.

:class:`EvidenceContext` is the single entry point, and it is deliberately **not** specific to
OKH-RAG. §4.1 requires that every method be scored with the same generator, which is only meaningful if
the generator sees comparable input, so the baselines in :mod:`~okh_rag.generation.baselines` build the
same object from their own retrieval results. What differs is what they can fill in: a text chunk has no
relation and no entity set, so those lines are omitted rather than fabricated. That asymmetry is the
treatment being compared — hiding it by padding the baseline with empty fields would make the
comparison look tidier and mean less.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

from okh_rag.schema import Hyperedge, KnowledgeHypergraph
from okh_rag.vocabulary import (
    coverage_phase_of,
    family_of,
    is_cross_horizon,
    precedence_phase_of,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from okh_rag.retrieval.retriever import RetrievalResult

__all__ = [
    "EvidenceContext",
    "EvidenceStep",
    "EvidenceTrajectory",
    "Passage",
    "horizon_label",
    "reasoning_tags",
    "step_from_hyperedge",
    "trajectory_from_hyperedge_ids",
]


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

#: §A.8 renders horizons as ``[T-XX]``. The number is the lead time in hours, which is how the release
#: keys its blocks, so no unit conversion happens anywhere in this package.
_HORIZON_LABEL = "T-{hours}"

#: Emitted when a value is genuinely absent. A visible placeholder is used rather than an empty string
#: because an empty value in a labelled line reads as "the field exists and is blank", which is a
#: different claim from "this evidence has no such field".
_ABSENT = "?"


def horizon_label(hours: int | None) -> str:
    """``T-48`` for 48 hours, and :data:`_ABSENT` when the horizon is unknown.

    The placeholder matters for the baselines: a text chunk cut from prose may carry no lead time, and
    ``[T-?]`` tells the generator that the ordering cue is missing, where ``[T-None]`` would look like a
    horizon label it failed to understand.
    """
    return _ABSENT if hours is None else _HORIZON_LABEL.format(hours=hours)


def _phase_label(edge: Hyperedge) -> str:
    """The step line's ``phase=`` value.

    §A.8 does not say *which* phase, and the codebase has two phase vocabularies for good reasons:
    §A.3's 11-stage precedence order (for within-horizon ordering) and §A.6 Eq. 13's 6-member set ``S``
    (for coverage scoring). The 11-stage order is the more granular of the two and the one §3.3
    describes in prose — *"recognize preconditions, track escalation across horizons"* — so it is
    preferred, with the 6-member set as a fallback for relations §A.3 leaves out, and the family name as
    a final fallback so the field is never blank for a known relation.
    """
    return (
        precedence_phase_of(edge.relation)
        or coverage_phase_of(edge.relation)
        or family_of(edge.relation)
        or _ABSENT
    )


def _family_label(edge: Hyperedge) -> str:
    """The step line's ``family=`` value — the §A.1 Table 3 family, or a marker for cross-horizon edges.

    Family 13 is the one family §A.3's within-horizon ordering excludes, so labelling it ``cross_horizon``
    is not a fallback: it is the fact the generator most needs to see, because that edge is the
    temporal link the whole method exists to preserve.
    """
    if is_cross_horizon(edge.relation):
        return "cross_horizon"
    return family_of(edge.relation) or _ABSENT


# ---------------------------------------------------------------------------
# The reasoning line
# ---------------------------------------------------------------------------


def reasoning_tags(
    edges: Sequence[Hyperedge],
    index: int,
    *,
    max_tags: int = 3,
) -> tuple[str, ...]:
    """Derive §A.8's ``Reasoning:`` line for step ``index`` from its neighbours in the trajectory.

    §A.8's example is ``within_horizon, hazard_to_operation, ...``. Read against §3.2's terms, both tags
    name the relationship to an **adjacent step**, not a property of the step by itself. Two independent
    tests are applied to each neighbour, so a step can earn both:

    * ``within_horizon`` — the neighbour is at the same horizon, so this step continues the same temporal
      snapshot (the case Eq. 5's order-coherence term rewards).
    * ``<from>_to_<to>`` — the neighbour sits at a different causal phase, so the step is a causal
      hand-off: advisory → hazard → operation → impact → recovery. The tag is built from the two steps'
      phase labels, which is why the example's second tag is ``hazard_to_operation`` rather than a fixed
      vocabulary string.

    The two tests are independent on purpose. Keeping them separate is what makes the example's pair
    reachable in one trajectory: a single block can move advisory → hazard *without* changing horizon, and
    that is the escalation §3.3 says the generator should be able to track. (An earlier form of this rule
    emitted the phase tag only when the horizon *changed*, which made ``hazard_to_operation`` unreachable
    inside the very horizon blocks the paper is about.)

    # [UNSPECIFIED] §A.8 gives one example and no rule. The alternatives considered were: (a) a fixed
    # tag vocabulary attached to relations — rejected because the example's ``hazard_to_operation`` mixes
    # two phase names and so cannot be a relation's own tag; (b) tags from the relation's family only —
    # rejected because it reproduces the step line's ``family=`` field and adds nothing; (c) tags from
    # the Eq. 5 term decomposition — rejected because the terms are trajectory-level, while the line is
    # per step.
    #
    # The rule deliberately looks **both ways**: §A.8's ordering cues are what let the model "track
    # escalation", and escalation is only visible from a step's predecessor. The successor is included
    # because a trajectory's last step would otherwise lose the tag that explains why it is last.

    Trajectory-relative, so this cannot be a method on :class:`Hyperedge` — the same hyperedge appears in
    many trajectories and earns different tags in each.
    """
    if not 0 <= index < len(edges):
        raise IndexError(f"step index {index} out of range for a trajectory of {len(edges)} hyperedges")

    current = edges[index]
    tags: list[str] = []
    current_phase = _phase_label(current)

    for offset in (-1, 1):
        neighbour_index = index + offset
        if not 0 <= neighbour_index < len(edges):
            continue
        neighbour = edges[neighbour_index]

        # Temporal test. Requires both horizons to be known: an unknown horizon is not evidence of a
        # horizon change, and guessing "different" would invent a temporal claim.
        if (
            neighbour.horizon_hours is not None
            and current.horizon_hours is not None
            and neighbour.horizon_hours == current.horizon_hours
        ):
            tags.append("within_horizon")

        # Causal test. Independent of the temporal one — the two are not in competition.
        neighbour_phase = _phase_label(neighbour)
        if neighbour_phase != current_phase:
            tags.append(f"{neighbour_phase}_to_{current_phase}")

    # A family-13 edge *is* the cross-horizon link, and that fact is invisible from the two tags above
    # (whose phases are equal at the endpoints of some change edges). Stated explicitly instead.
    if is_cross_horizon(current.relation):
        tags.append("cross_horizon")

    seen: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.append(tag)
    # §A.8's line is prose budget, not a data dump; the first tags name the nearest neighbours, which
    # are the ones that explain the step's position.
    return tuple(seen[:max_tags])


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceStep:
    """One hyperedge, in the form §A.8's four-line block needs it.

    Frozen because an evidence block is a *record of what was retrieved*: two runs on the same retrieval
    result must render byte-identically, and §3.3's claim that the generator can read the structure back
    out is only testable if the rendering is stable.
    """

    step: int
    relation: str
    evidence: str
    phase: str
    family: str
    horizon_hours: int | None
    entity_labels: tuple[str, ...]
    reasoning: tuple[str, ...] = ()
    cross_horizon: bool = False

    def render(self, *, include_reasoning: bool = True, ordered: bool = True) -> str:
        """The §A.8 block for this step.

        Args:
            include_reasoning: Emit the ``Reasoning:`` line. §A.8 lists it as part of the format, so the
                default is on; the switch exists because the line is derived rather than extracted, and a
                run that ablates it is testing whether the derived tags carry information the step line
                does not already carry.
            ordered: Label the block ``[Step k]``. When ``False`` the label is ``[Evidence k]`` — a
                *display* index rather than a position in a sequence. This distinction is load-bearing for
                §4.1: the HyperGraphRAG baseline "retrieves unordered hyperedges from the same knowledge
                hypergraph H without modeling order", and rendering its output with ``[Step k]`` headers
                would hand it the ordering cue the ablation is supposed to remove. An index is needed to
                let the model refer to a block at all; a *step* number asserts an adjacency that an
                unordered retrieval never established.

        Returns:
            A multi-line block, with no trailing newline.
        """
        label = "Step" if ordered else "Evidence"
        header = (
            f"[{label} {self.step}] [{horizon_label(self.horizon_hours)}] "
            f"[phase={self.phase}] [family={self.family}]"
        )
        lines = [header, f"Relation: {self.relation}", f"Evidence: {self.evidence}"]
        if include_reasoning and self.reasoning:
            lines.append(f"Reasoning: {', '.join(self.reasoning)}")
        if self.entity_labels:
            lines.append(f"Entities: {'; '.join(self.entity_labels)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Passage:
    """A baseline's unit of evidence — a text chunk with no relation and no entity set.

    Kept as a distinct type rather than an :class:`EvidenceStep` with blank fields so that the renderer
    cannot accidentally emit ``Relation: `` for a chunk that has none. §4.1 compares methods with the
    same generator, and a baseline that appears to supply a relation would be claiming structure it did
    not recover.

    The chunk's horizon is recorded but **not rendered**. Text-RAG's chunk *is* the raw prose, which
    usually announces its own lead time in its first line (``"...T-48 hours before expected landfall:"``),
    so a normalised ``[T-48]`` header would add nothing the text does not already say and would add it
    in the method's own representation. That is the treatment being compared, so it is left out.
    """

    step: int
    text: str
    horizon_hours: int | None = None
    source: str = ""

    def render(self, *, include_reasoning: bool = True, ordered: bool = True) -> str:  # noqa: ARG002
        """The block for a passage.

        Both keyword arguments are accepted and ignored, because they are part of the shared interface with
        :meth:`EvidenceStep.render` and a trajectory renders a mixed sequence without a type test. A
        passage is never assigned a ``[Step k]`` header — that would assert adjacency — and it has no
        derived annotation to suppress. ``[Chunk i]`` is a display index only.
        """
        header = f"[Chunk {self.step}]"
        if self.source:
            header += f" [source={self.source}]"
        return f"{header}\nText: {self.text}"


# ---------------------------------------------------------------------------
# Trajectories and contexts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceTrajectory:
    """One reasoning path (§3.2's ``γ``), ready to render.

    ``terms`` carries Eq. 5's decomposition when the trajectory came from the order-aware retriever. §A.8
    says a "trajectory-level quality summary is prepended when available", and the term breakdown is
    exactly that: it tells the generator how much of the path's score came from order coherence versus
    raw relevance, which is the difference between "these steps are relevant" and "these steps are
    relevant *and in an order that means something*".
    """

    index: int
    steps: tuple[EvidenceStep | Passage, ...]
    score: float = 0.0
    terms: Mapping[str, float] = field(default_factory=dict)
    ordered: bool = True

    def __len__(self) -> int:
        return len(self.steps)

    def render(self, *, include_quality_summary: bool = True, include_reasoning: bool = True) -> str:
        """Render the quality summary (when asked and available) followed by the numbered blocks.

        ``include_reasoning`` is forced off for an unordered trajectory: the ``Reasoning:`` line is derived
        from a step's *neighbours*, so for a set that was never ordered it would describe adjacencies the
        retriever did not establish. Letting it through would silently upgrade the baseline into the thing
        it is a baseline for.
        """
        blocks: list[str] = []
        summary = self.quality_summary() if include_quality_summary else None
        if summary:
            blocks.append(summary)
        reasoning = include_reasoning and self.ordered
        blocks.extend(
            step.render(include_reasoning=reasoning, ordered=self.ordered) for step in self.steps
        )
        return "\n".join(blocks)

    def quality_summary(self) -> str | None:
        """The §A.8 quality line, or ``None`` when no trajectory-level score is available.

        Returns ``None`` rather than a zero-filled line for the baseline case. A summary reading
        ``score=0.000`` would be a false statement about a text chunk — those numbers are Eq. 5's, and
        nothing in the pipeline evaluated Eq. 5 for a chunk.
        """
        if not self.terms and not self.score:
            return None
        parts = [f"score={self.score:.3f}"]
        for name in sorted(self.terms):
            parts.append(f"{name}={self.terms[name]:.3f}")
        return f"[Trajectory g{self.index}] steps={len(self.steps)} " + " ".join(parts)


@dataclass(frozen=True)
class EvidenceContext:
    """Everything the generator sees for one question, in the order §A.8 presents it.

    Attributes:
        question: The prompt text handed to the generator.
        trajectories: ``Γ_q``, best first. §A.8's single-call mode numbers these as alternative
            reasoning paths and asks the model to cross-reference them.
        source: Which retriever produced this context. Recorded because §4.1's comparison is between
            methods, and a run that loses track of which context came from which method cannot report it.
        notes: Free-form lines prepended above the trajectories, for anything the retriever knows and
            the model should (for example that a rollout was order-shuffled).
    """

    question: str
    trajectories: tuple[EvidenceTrajectory, ...]
    source: str
    notes: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.trajectories)

    @property
    def is_empty(self) -> bool:
        """Whether retrieval returned nothing to condition on.

        §A.8 has no no-evidence branch, and inventing one would let a retrieval failure masquerade as a
        generation result: the caller decides, and this property exists so it can.
        """
        return not any(len(t) for t in self.trajectories)

    @property
    def is_multi_trajectory(self) -> bool:
        """Whether several **non-empty** paths are on offer — §A.8's trigger for the cross-referencing prompt.

        Lives here rather than in the generator because *two* places act on it: the prompt asks for a
        ``supporting_trajectory`` field and the output schema makes that field required. Two copies of the
        predicate would eventually disagree, and the failure mode is a prompt requesting a field the
        constrained decoder refuses to emit — a whole run of failed calls for a one-line inconsistency.

        Two exclusions, both deliberate. A single non-empty path is not multi-trajectory: asking "which path
        supports this?" when there is one path invites a fabricated choice. And an *empty* trajectory is not
        a path either — it is a retrieval artifact that produced no evidence — so a context holding one real
        path and one empty one is treated as single-path rather than as a comparison between a path and a
        blank.
        """
        return sum(1 for trajectory in self.trajectories if len(trajectory)) > 1

    def render(self, *, include_quality_summary: bool = True, include_reasoning: bool = True) -> str:
        """The full evidence block, with ``YYYY``-numbered trajectories as §A.8's single-call mode wants.

        Numbering is 1-based and per trajectory; the step numbering inside a trajectory restarts too, so
        ``[Step 3]`` in ``g2`` is the third step *of that path*, not the third block overall. That is the
        reading §A.8's "numbered reasoning paths" asks for — the paths are alternatives, not a sequence.
        """
        sections = list(self.notes)
        if not self.trajectories:
            return "\n".join(sections)
        for trajectory in self.trajectories:
            sections.append(
                trajectory.render(
                    include_quality_summary=include_quality_summary,
                    include_reasoning=include_reasoning,
                )
            )
        return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def step_from_hyperedge(
    step: int,
    edge: Hyperedge,
    hypergraph: KnowledgeHypergraph,
    *,
    edges: Sequence[Hyperedge],
    index: int,
    max_tags: int = 3,
) -> EvidenceStep:
    """Build one :class:`EvidenceStep`, resolving ``V_e`` through ``hypergraph``.

    ``edges``/``index`` are the trajectory and this step's position in it, needed for
    :func:`reasoning_tags`. They are passed separately from ``edge`` because the same hyperedge can sit at
    different positions in different trajectories and therefore earn different tags.
    """
    entity_labels = tuple(
        f"{entity.name} [{entity.type}]" for entity in hypergraph.entity_set(edge)
    )
    return EvidenceStep(
        step=step,
        relation=edge.relation,
        evidence=edge.evidence,
        phase=_phase_label(edge),
        family=_family_label(edge),
        horizon_hours=edge.horizon_hours,
        entity_labels=entity_labels,
        reasoning=reasoning_tags(edges, index, max_tags=max_tags),
        cross_horizon=is_cross_horizon(edge.relation),
    )


def trajectory_from_hyperedge_ids(
    hypergraph: KnowledgeHypergraph,
    hyperedge_ids: Iterable[str],
    *,
    index: int = 1,
    score: float = 0.0,
    terms: Mapping[str, float] | None = None,
    max_tags: int = 3,
    ordered: bool = True,
) -> EvidenceTrajectory:
    """Render an ordered list of hyperedge ids as an :class:`EvidenceTrajectory`.

    Unknown ids are **skipped, not raised on**, matching :meth:`KnowledgeHypergraph.entity_set`'s
    treatment of dangling entity ids: an id that survives retrieval but not the hypergraph is a graph
    inconsistency, and the honest response is to render the evidence that does exist rather than lose the
    whole question. Callers that need the inconsistency to be loud should compare lengths.

    Args:
        ordered: ``False`` for a retrieval that produced a *set* — §4.1's HyperGraphRAG baseline. The
            blocks are then labelled ``[Evidence k]`` and carry no reasoning tags; see
            :meth:`EvidenceTrajectory.render` for why the tags must go with the ordering.
    """
    resolved: list[Hyperedge] = []
    for hyperedge_id in hyperedge_ids:
        edge = hypergraph.hyperedges.get(hyperedge_id)
        if edge is not None:
            resolved.append(edge)

    steps: list[EvidenceStep] = []
    for position, edge in enumerate(resolved):
        steps.append(
            step_from_hyperedge(
                position + 1,
                edge,
                hypergraph,
                edges=resolved,
                index=position,
                max_tags=max_tags if ordered else 0,
            )
        )
    return EvidenceTrajectory(
        index=index,
        steps=tuple(steps),
        score=score,
        terms=dict(terms or {}),
        ordered=ordered,
    )


def context_from_retrieval(
    result: "RetrievalResult",
    hypergraph: KnowledgeHypergraph,
    *,
    question: str | None = None,
    include_quality_summary: bool = True,
    include_reasoning_tags: bool = True,
) -> EvidenceContext:
    """Turn an OKH-RAG :class:`~okh_rag.retrieval.RetrievalResult` into a renderable context.

    Two §4.5 facts are propagated here rather than left for the caller to remember:

    * ``result.shuffled`` becomes a ``notes`` line. Table 2's Shuffle row keeps the retrieved content and
      permutes it, and the point of the row is that the *generator* can no longer see the order. A silent
      shuffle would attribute the drop to the wrong cause.
    * ``result.variant`` becomes a ``notes`` line whenever it is not the full model, so an ablated
      rollout's context is self-describing when it is written to a run manifest.

    The switches ``include_quality_summary`` / ``include_reasoning_tags`` are parameters here rather than
    reads of ``cfg.generation`` so this module stays free of configuration — the caller in
    :mod:`~okh_rag.generation.generator` is the place that knows the config.

    Returns:
        A context whose trajectory terms are copied from ``result.term_scores`` when the quality summary
        is enabled, and left empty otherwise (so the summary is genuinely absent, not merely unrendered).
    """
    notes: list[str] = []
    if result.shuffled:
        notes.append(
            "Note: the retrieved trajectory was order-shuffled for the Table 2 stress test; "
            "the content is unchanged but the step order carries no information."
        )
    if result.variant != "learned":
        notes.append(f"Note: retrieved with the '{result.variant}' order model variant.")

    trajectories: list[EvidenceTrajectory] = []
    for position, trajectory in enumerate(result.trajectories):
        terms: Mapping[str, float] = {}
        if include_quality_summary and position < len(result.term_scores):
            terms = _terms_of(result.term_scores[position])
        trajectories.append(
            trajectory_from_hyperedge_ids(
                hypergraph,
                trajectory.hyperedge_ids,
                index=position + 1,
                score=trajectory.score,
                terms=terms,
                max_tags=3 if include_reasoning_tags else 0,
            )
        )

    return EvidenceContext(
        question=question if question is not None else result.query,
        trajectories=tuple(trajectories),
        source="okh_rag",
        notes=tuple(notes),
    )


def _terms_of(score: object) -> dict[str, float]:
    """Eq. 5's five named terms of a :class:`~okh_rag.schema.TrajectoryScore`, as a plain dict.

    The names are written out here rather than derived by iterating the dataclass so that the rendered
    summary is stable if a field is ever added: an unexpected extra term in the generator's prompt is a
    silent change to the experiment, and this way it has to be added deliberately.
    """
    return {
        "relevance": float(getattr(score, "relevance", 0.0)),
        "order_coherence": float(getattr(score, "order_coherence", 0.0)),
        "precedence": float(getattr(score, "precedence_consistency", 0.0)),
        "continuity": float(getattr(score, "entity_continuity", 0.0)),
        "coverage": float(getattr(score, "phase_coverage", 0.0)),
    }
