"""§A.4 — the ordered pairs the transition model is trained on, and the negative pool.

§A.4 describes three self-supervised signals, none of which needs a human label:

    "Document order (P_doc) ... co-occurring hyperedges (e_i, e_j) with π(e_i) < π(e_j) ... form positive
     ordered pairs. Entity-overlap consistency (P_ent) ... Pairs sharing entities (|V_ei ∩ V_ej| ≥ 1)
     within the same knowledge group join P if they appear in canonical trajectory order. Retrieval-induced
     preference (P_ret) ... After initial training, retrieval traces from correct answers yield additional
     positive consecutive pairs, creating a self-training loop."

and one source of negatives:

    "random cross-group pairs"

with, from the same section, "sampled softmax with K = 64 global negatives per example". The negatives are
therefore drawn *per example during training*, not materialised once as a dataset — which is why this
module exposes a sampler rather than a negative-pair list.

Two decisions worth stating, because both change what the signals mean:

**π(·) is the §A.3 index, not a character offset.** The paper writes π(e) without defining it. The
canonical trajectory of §A.3 is already the pipeline's notion of "the order these facts belong in", and
§A.4 elsewhere says the ordering supplies "the ground-truth trajectory for document-order supervision in
training (§A.4)" — so the two are meant to be the same object. Alternative: the raw character offset.

**The entity-overlap signal excludes temporal anchors by default.** §A.2 step 3 puts a per-horizon
``horizon:T-NN`` anchor into *every* horizon-grounded hyperedge, so ``|V_ei ∩ V_ej| ≥ 1`` holds for every
same-horizon pair by construction. Read literally, the signal then supervises "these two are at the same
horizon", which the document-order signal already covers. Excluding anchors restores the reading the name
suggests — that a transition is licensed by the same real entities carrying over — and
``training.entity_overlap_ignore_anchors`` switches back to the literal form.

The overlap relation is expanded as *chains*: for one entity, its occurrences in canonical order are
linked pairwise-consecutively (p1→p2, p2→p3, ...). That is linear in the number of occurrences and
captures "entity continuity" directly, whereas emitting all ``(|V| choose 2)`` overlapping pairs per group
is quadratic in group size — at §4 scale, tens of millions of pairs per corpus pass, dominated by pairs
that are already covered.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from okh_rag.config import TrainingConfig
from okh_rag.schema import EntityCategory, Hyperedge, KnowledgeHypergraph

__all__ = [
    "SIGNAL_DOCUMENT_ORDER",
    "SIGNAL_ENTITY_OVERLAP",
    "SIGNAL_NAMES",
    "SIGNAL_RETRIEVAL_INDUCED",
    "NegativeSampler",
    "OrderedPair",
    "TrainingPairs",
    "build_training_pairs",
    "pair_entity_overlap",
]

SIGNAL_DOCUMENT_ORDER = "document_order"
"""§A.4 ``P_doc`` — consecutive hyperedges in the §A.3 canonical order."""

SIGNAL_ENTITY_OVERLAP = "entity_overlap"
"""§A.4 ``P_ent`` — hyperedges linked by shared entities, in canonical order."""

SIGNAL_RETRIEVAL_INDUCED = "retrieval_induced"
"""§A.4 ``P_ret`` — positives recovered from retrieval traces that produced correct answers."""

SIGNAL_NAMES: tuple[str, ...] = (
    SIGNAL_DOCUMENT_ORDER,
    SIGNAL_ENTITY_OVERLAP,
    SIGNAL_RETRIEVAL_INDUCED,
)


@dataclass(frozen=True)
class OrderedPair:
    """One positive ordered pair ``(e_i, e_j)`` with ``e_i ≺ e_j``.

    Attributes:
        source_id: ``e_i``.
        target_id: ``e_j``.
        signal: Which §A.4 signal contributed the pair, for the per-signal counts in the run manifest.
    """

    source_id: str
    target_id: str
    signal: str


@dataclass(frozen=True)
class TrainingPairs:
    """The positive set ``P``, plus a count of groups that contributed nothing.

    Attributes:
        positives: Every positive ordered pair, deduplicated on ``(source, target)``.
        groups_used: Groups that produced at least one positive.
        groups_missing_order: Groups with no §A.3 canonical ordering. Non-zero means
            :func:`~okh_rag.precedence.canonical_order` was never run, or was run on a different
            hypergraph — a silent source of "training works but the model learned nothing" bugs.
    """

    positives: tuple[OrderedPair, ...]
    groups_used: tuple[str, ...]
    groups_missing_order: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.positives)

    def by_signal(self) -> dict[str, int]:
        """Positive count per §A.4 signal, for the run manifest."""
        counts = {name: 0 for name in SIGNAL_NAMES}
        for pair in self.positives:
            counts[pair.signal] = counts.get(pair.signal, 0) + 1
        return counts

    def sources(self) -> tuple[str, ...]:
        """Unique source ids in first-seen order, i.e. the examples one epoch iterates over."""
        seen: dict[str, None] = {}
        for pair in self.positives:
            seen.setdefault(pair.source_id, None)
        return tuple(seen)


# ---------------------------------------------------------------------------
# §A.4 — positive pair construction
# ---------------------------------------------------------------------------


def _overlap_entities(
    edge: Hyperedge, hypergraph: KnowledgeHypergraph, *, ignore_anchors: bool
) -> tuple[str, ...]:
    """The entities of ``V_e`` that the overlap signal may match on."""
    if not ignore_anchors:
        return edge.entity_ids
    kept: list[str] = []
    for entity_id in edge.entity_ids:
        entity = hypergraph.entities.get(entity_id)
        if entity is not None and entity.category is EntityCategory.TEMPORAL:
            continue
        kept.append(entity_id)
    return tuple(kept)


def build_training_pairs(
    hypergraph: KnowledgeHypergraph,
    ordering: Mapping[str, Sequence[str]],
    cfg: TrainingConfig,
) -> TrainingPairs:
    """Assemble ``P`` from §A.4's document-order and entity-overlap signals.

    The retrieval-induced signal is *not* built here: it needs retrieval traces, so the caller passes the
    positives it recovered to :func:`~okh_rag.training.train.train_order_model` through that function's
    ``extra_positives`` argument.

    Args:
        hypergraph: The hypergraph the pairs are drawn from. Pass a
            :meth:`~okh_rag.schema.KnowledgeHypergraph.subset` to build the training or validation split.
        ordering: ``group_id -> ordered hyperedge ids`` from
            :func:`~okh_rag.precedence.canonical_order`.
        cfg: The ``training`` config section; ``signals`` gates which signals contribute, and
            ``document_order_window``/``entity_overlap_ignore_anchors`` shape them.

    Returns:
        The deduplicated positive pairs. A pair contributed by both signals is recorded under the
        document-order signal, since that is the one §A.3's trajectory is defined by.
    """
    positives: dict[tuple[str, str], OrderedPair] = {}
    used: list[str] = []
    missing: list[str] = []

    for group_id, member_ids in hypergraph.groups.items():
        order = tuple(hid for hid in ordering.get(group_id, ()) if hid in hypergraph.hyperedges)
        if not order:
            missing.append(group_id)
            continue

        before = len(positives)
        if cfg.signals.document_order:
            _add_document_order_pairs(order, cfg.document_order_window, positives)
        if cfg.signals.entity_overlap:
            _add_entity_overlap_pairs(order, hypergraph, cfg, positives)
        if len(positives) > before:
            used.append(group_id)

    return TrainingPairs(
        positives=tuple(positives.values()),
        groups_used=tuple(used),
        groups_missing_order=tuple(missing),
    )


def _add_document_order_pairs(
    order: Sequence[str],
    window: int,
    positives: dict[tuple[str, str], OrderedPair],
) -> None:
    """§A.4 ``P_doc`` — each hyperedge paired with the next ``window`` in canonical order.

    §A.4 writes "co-occurring hyperedges (e_i, e_j) with π(e_i) < π(e_j)" without bounding the gap.
    ``window = 1`` reads "co-occurring" as "adjacent in the trajectory", which is the reading consistent
    with §A.4's own phrase for the retrieval-induced signal ("additional positive *consecutive* pairs")
    and the only one that stays tractable. A larger window is quadratic in group size.
    """
    for i, source_id in enumerate(order):
        for j in range(i + 1, min(i + 1 + window, len(order))):
            target_id = order[j]
            positives.setdefault(
                (source_id, target_id),
                OrderedPair(source_id, target_id, SIGNAL_DOCUMENT_ORDER),
            )


def _add_entity_overlap_pairs(
    order: Sequence[str],
    hypergraph: KnowledgeHypergraph,
    cfg: TrainingConfig,
    positives: dict[tuple[str, str], OrderedPair],
) -> None:
    """§A.4 ``P_ent`` — chain one entity's occurrences in canonical order.

    See the module docstring for why the overlap relation is expanded as per-entity chains rather than as
    all pairwise overlaps, and for why anchors are excluded by default.
    """
    occurrences: dict[str, list[int]] = {}
    for position, hyperedge_id in enumerate(order):
        edge = hypergraph.hyperedges[hyperedge_id]
        for entity_id in _overlap_entities(
            edge, hypergraph, ignore_anchors=cfg.entity_overlap_ignore_anchors
        ):
            occurrences.setdefault(entity_id, []).append(position)

    for positions in occurrences.values():
        for earlier, later in zip(positions, positions[1:]):
            source_id, target_id = order[earlier], order[later]
            if source_id == target_id:
                # Unreachable while V_e is deduplicated, but a chain link from a hyperedge to itself
                # would be a training example the model cannot fail, so it is ruled out explicitly.
                continue
            positives.setdefault(
                (source_id, target_id),
                OrderedPair(source_id, target_id, SIGNAL_ENTITY_OVERLAP),
            )


# ---------------------------------------------------------------------------
# §A.4 — negative sampling
# ---------------------------------------------------------------------------


class NegativeSampler:
    """§A.4 — "sampled softmax with K = 64 global negatives per example", from "random cross-group pairs".

    Negatives are drawn uniformly from the complement of the source's own knowledge group, by rejection
    sampling over the global pool (see :meth:`_sample_one` for why rejection rather than a per-group
    complement list). Every eligible hyperedge therefore has equal probability of being drawn, and a
    single example can never be dominated by one other scenario's hyperedges.

    A corpus with only one group has no cross-group pool at all; the sampler then falls back to the whole
    pool minus the source and records the fact in :attr:`same_group_draws`.
    """

    def __init__(
        self,
        hypergraph: KnowledgeHypergraph,
        *,
        k: int,
        rng: random.Random,
    ) -> None:
        """Index the corpus by group.

        Args:
            hypergraph: The corpus to draw negatives from. This should be the *training* subset, so no
                validation or test hyperedge can leak into a gradient step.
            k: §A.4's ``K`` — negatives per example.
            rng: The RNG. Pass one seeded from
                :func:`~okh_rag.seeding.derive_seed` so the sampler is independent of the split.

        Raises:
            ValueError: If ``k`` is not positive, or the corpus has no hyperedges.
        """
        if k <= 0:
            raise ValueError(f"K must be positive, got {k}")

        self.k = k
        self._rng = rng
        self._group_of: dict[str, str] = {}
        by_group: dict[str, list[str]] = {}
        for group_id, member_ids in hypergraph.groups.items():
            members = [h for h in member_ids if h in hypergraph.hyperedges]
            if not members:
                continue
            by_group[group_id] = members
            for hyperedge_id in members:
                self._group_of[hyperedge_id] = group_id

        self._by_group = {g: tuple(m) for g, m in by_group.items()}
        self._group_ids = tuple(sorted(self._by_group))
        self._all_ids = tuple(sorted(self._group_of))
        if not self._all_ids:
            raise ValueError("cannot sample negatives from an empty hypergraph")

        # Diagnostics, so a corpus that is too small to sample proper cross-group negatives is visible in
        # the run manifest rather than silently producing same-group negatives.
        self.same_group_draws = 0
        self.short_batches = 0

    def stats(self) -> dict[str, int]:
        """Sampler diagnostics for the run manifest."""
        return {
            "k": self.k,
            "groups": len(self._group_ids),
            "pool_size": len(self._all_ids),
            "same_group_draws": self.same_group_draws,
            "short_batches": self.short_batches,
        }

    def eligible(self, source_id: str) -> int:
        """How many hyperedges are eligible as a negative for ``source_id``.

        §A.4 asks for *cross-group* pairs, so this is ``|E| − |group(source)|`` when more than one group
        exists. A corpora-wide :meth:`sample` call is only well-defined for ``K ≤`` this number per source,
        which is what the ``short_batches`` counter reports.
        """
        own = self._group_of.get(source_id)
        if own is None or len(self._group_ids) < 2:
            return len(self._all_ids)
        return len(self._all_ids) - len(self._by_group[own])

    def sample(self, source_ids: Sequence[str]) -> list[list[str]]:
        """Draw ``K`` negatives for each source.

        Args:
            source_ids: The batch's sources, one row of negatives each.

        Returns:
            A list aligned with ``source_ids``; entry *b* holds the negatives for ``source_ids[b]``. May
            hold fewer than ``K`` when the corpus cannot supply that many distinct eligible hyperedges —
            in which case ``short_batches`` is incremented rather than the same negative being repeated,
            because a repeated negative would be counted several times in Eq. 4's ``E_N`` and quietly
            reweight the negative term.
        """
        return [self._sample_one(source_id) for source_id in source_ids]

    def _sample_one(self, source_id: str) -> list[str]:
        """One source's negatives.

        The negative pool is the complement of the source's own group. Drawing uniformly from that
        complement is done by rejection sampling over the global pool: the acceptance rate is
        ``1 − 1/|groups|``, so with a realistic group count this is a few attempts per negative, whereas
        materialising the complement would cost ``O(|E|)`` memory *per group*.

        Rejection also gives *distinct* negatives, which matters because a repeated negative is averaged
        into Eq. 4's ``E_N`` more than once and would silently reweight that term.
        """
        own = self._group_of.get(source_id)
        others = [g for g in self._group_ids if g != own]
        if not others:
            # A single-group corpus. §A.4's "cross-group" rule has no content here, so fall back to the
            # whole pool minus the source and say so through the counter.
            self.same_group_draws += 1
            pool = [h for h in self._all_ids if h != source_id]
            if len(pool) < self.k:
                self.short_batches += 1
            return self._rng.sample(pool, min(self.k, len(pool)))

        picks: list[str] = []
        seen: set[str] = set()
        attempts = 0
        limit = self.k * 20
        while len(picks) < self.k and attempts < limit:
            attempts += 1
            candidate = self._all_ids[self._rng.randrange(len(self._all_ids))]
            if candidate in seen or self._group_of.get(candidate) == own:
                continue
            seen.add(candidate)
            picks.append(candidate)

        if len(picks) < self.k:
            # The complement is smaller than K, or rejection stalled. Top up round-robin over the other
            # groups, which is guaranteed to terminate because there is at least one of them.
            while len(picks) < self.k:
                for group_id in others:
                    if len(picks) == self.k:
                        break
                    picks.append(self._by_group[group_id][self._rng.randrange(len(self._by_group[group_id]))])
            self.short_batches += 1
        return picks

    def explain(self) -> Mapping[str, int | str]:
        """A human-readable summary, for ``--verbose`` logging."""
        return {
            "negatives_per_example": self.k,
            "distinct_groups": len(self._group_ids),
            "global_pool": len(self._all_ids),
            "note": (
                "negatives are drawn cross-group per §A.4; "
                f"{self.same_group_draws} draws fell back to same-group because the corpus has one group"
            ),
        }


def pair_entity_overlap(
    hypergraph: KnowledgeHypergraph,
    source_id: str,
    target_id: str,
    *,
    ignore_anchors: bool = True,
) -> int:
    """``|V_ei ∩ V_ej|``, used by tests and by the coverage report.

    Kept as a free function rather than a method on :class:`~okh_rag.schema.KnowledgeHypergraph` because
    the meaning of the intersection depends on whether anchors count, which is a §A.4 modelling choice
    rather than a property of the hypergraph.
    """
    source = hypergraph.hyperedges[source_id]
    target = hypergraph.hyperedges[target_id]
    return len(
        set(_overlap_entities(source, hypergraph, ignore_anchors=ignore_anchors))
        & set(_overlap_entities(target, hypergraph, ignore_anchors=ignore_anchors))
    )
