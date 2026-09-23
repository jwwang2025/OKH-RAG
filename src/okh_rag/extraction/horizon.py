"""§A.2 — per-horizon decomposition.

§A.2 explains why a multi-horizon document is not extracted in one call:

    "Multi-horizon scenario documents can span thousands of tokens. Single-call extraction risks
     truncation, entity conflation across horizons, and reduced attribute coverage. We therefore split
     each document at horizon headers ... and extract each block independently."

and notes the trade-off explicitly: "This per-horizon strategy yields substantially higher entity and
attribute coverage, at the cost of a subsequent merge step."
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from okh_rag.config import HorizonConfig


@dataclass(frozen=True)
class HorizonBlock:
    """One per-horizon slice of a scenario document.

    Attributes:
        group_id: The knowledge group (§A.4) this block belongs to.
        offset_hours: Lead time in hours before expected landfall, e.g. ``48`` for ``T-48``.
        text: The block's raw text, header line included.
        char_offset: Character offset of ``text`` in the parent document. This is the ``text_position``
            that §A.3 uses as its final topological tie-breaker.
    """

    group_id: str
    offset_hours: int
    text: str
    char_offset: int

    @property
    def horizon_label(self) -> str:
        """The ``T-NN`` label, matching §A.1's ``horizon:T-48`` notation."""
        return f"T-{self.offset_hours}"

    @property
    def anchor_name(self) -> str:
        """§A.1 / §A.2 step 3 — the anchor's *name*, e.g. ``horizon:T-48``.

        Note this is a name, not an id. §A.2 step 2's hierarchical id convention governs the id; see
        :func:`~okh_rag.extraction.normalize.horizon_anchor_id` for why a globally shared
        ``horizon:T-48`` id would break §A.4's negative sampling.
        """
        return f"horizon:{self.horizon_label}"


@dataclass(frozen=True)
class ScenarioDocument:
    """One scenario from the corpus ``K`` of §3.1 Eq. 1, already split into horizon blocks."""

    group_id: str
    storm: str
    port: str
    year: int
    blocks: tuple[HorizonBlock, ...]

    @property
    def full_text(self) -> str:
        """The blocks rejoined in lead-time order, for the Text-RAG baseline's chunking."""
        return "\n\n".join(b.text for b in self.blocks)


def split_horizons(
    text: str,
    *,
    group_id: str,
    cfg: HorizonConfig,
) -> list[HorizonBlock]:
    """Split ``text`` at horizon headers.

    §A.2 detects headers "via regex patterns such as ``T-48 hours before expected landfall:``", which is
    the single example the paper gives; ``cfg.header_pattern`` generalises it in two ways the example
    implies but does not spell out — it accepts ``hour``/``hours`` and trailing whitespace before the
    colon.

    Args:
        text: The raw scenario document.
        group_id: Knowledge group id, carried onto every block.
        cfg: The ``hypergraph.horizon`` config section.

    Returns:
        Blocks in the order the headers appear. Blocks whose header names an offset outside
        ``cfg.offsets_hours`` are still returned — the extractor keeps them, and the offset set is only
        used for validation — but ``offset_hours`` then reflects the header as written, not the
        configured grid.

    Raises:
        ValueError: If no header matches. Falling back to a single whole-document block would silently
            produce an unhorizoned hypergraph in which §A.2's per-horizon benefits are absent and every
            hyperedge would be incomparable under §A.3's phase rules. The caller must decide what to do
            with such a document instead.
    """
    pattern = re.compile(cfg.header_pattern)
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(
            f"no horizon header matched for group {group_id!r} using pattern {cfg.header_pattern!r}. "
            "§A.2 requires horizon headers for per-horizon extraction; either fix the document or "
            "extend the pattern in configs/base.yaml."
        )

    blocks: list[HorizonBlock] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segment = text[start:end]
        offset = int(match.group("h"))
        blocks.append(
            HorizonBlock(
                group_id=group_id,
                offset_hours=offset,
                text=segment.strip(),
                char_offset=start,
            )
        )
    return blocks


def order_blocks_by_lead_time(blocks: list[HorizonBlock]) -> list[HorizonBlock]:
    """Sort blocks largest lead time first, i.e. ``T-120`` before ``T-12``.

    §A.3 — "Hyperedges in the same semantic family at different horizons follow decreasing horizon
    order: T-96 ≺ T-72 ≺ T-48 ≺ T-24 ≺ T-12. This ensures that evolving information is presented
    chronologically."
    """
    return sorted(blocks, key=lambda b: (-b.offset_hours, b.char_offset))


def offsets_present(blocks: list[HorizonBlock], cfg: HorizonConfig) -> list[int]:
    """The subset of ``cfg.offsets_hours`` that actually appears, for coverage diagnostics.

    §4.1 says the benchmark covers "multi-horizon descriptions from T-120 to T-12"; a scenario missing
    most of that grid is a data-quality signal worth reporting before extraction is paid for.
    """
    present = {b.offset_hours for b in blocks}
    return [h for h in cfg.offsets_hours if h in present]
