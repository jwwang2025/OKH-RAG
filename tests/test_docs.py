"""Structural checks on the written deliverables.

``.paper2code_work/<arxiv>/ambiguity_audit.md`` is a deliverable, not scratch: it is the record of every
decision this reproduction made where the paper was silent, ambiguous, or self-contradictory, and
``REPRODUCTION_NOTES.md`` points a reader at it. It is also a large hand-maintained markdown file full of
tables, and a markdown table breaks *silently* — one stray unescaped ``|`` inside an inline code span
splits a row into extra cells, and the rendered page simply shows a mangled table with no error anywhere.

That is not hypothetical. Adding a retrieval decision row during this reproduction turned up three
already-broken rows in the shipped audit: two where a set-cardinality literal (``|V_ei ∩ V_ej|`` and
``|V|``) had been written with unescaped pipes, and one new row of mine that was missing an
``Alternatives`` cell entirely. Nothing else in the project would have caught any of them.

These tests are skipped when the audit is absent, because ``.paper2code_work/`` is git-ignored — it holds
regenerable working state, and the audit is produced while reproducing rather than shipped. A reader
cloning the repository will see them skip; that is the intended behaviour, not a failure.

``data/CONTRACT.md`` is the other hand-maintained markdown deliverable and is guarded the same way, with
one difference: it *does* ship, so :class:`TestContractDoc` always runs. Its tables have different layouts,
so the expected width is each table's own header rather than the audit's single declared column count.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT = REPO_ROOT / ".paper2code_work" / "2604.12185" / "ambiguity_audit.md"
CONTRACT = REPO_ROOT / "data" / "CONTRACT.md"

# Every audit table is Component | Status | Paper Quote | Section | Our Choice | Alternatives.
AUDIT_COLUMNS = 6

requires_audit = pytest.mark.skipif(
    not AUDIT.exists(),
    reason=".paper2code_work is git-ignored regenerable state; run the reproduction to produce it",
)


def _audit_lines() -> list[str]:
    return AUDIT.read_text(encoding="utf-8").split("\n")


def _cell_count(row: str) -> int:
    """Number of table cells in ``row``, counting ``\\|`` as an escaped literal rather than a separator."""
    # ``(?<!\\)`` — a pipe not preceded by a backslash. The lookbehind is what makes ``\|C\|`` one cell.
    pieces = re.split(r"(?<!\\)\|", row.strip())
    # A well-formed row is ``| a | b |``, so splitting yields an empty string on each end.
    return len(pieces) - 2


def _table_blocks(lines: list[str]) -> list[list[tuple[int, str]]]:
    """``[[(1-based line number, row text)]]`` — one list per contiguous block of table rows.

    Generic over a whole document, where :func:`_tables` is scoped to the audit's ``## ``
    sections. A block is separated by any non-table line, which is how Markdown itself delimits
    one table from the next.
    """
    blocks: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for index, line in enumerate(lines, 1):
        if line.strip().startswith("|"):
            current.append((index, line))
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _tables() -> list[tuple[str, list[tuple[int, str]]]]:
    """``[(section name, [(1-based line number, row text)])]`` for every ``## `` section with a table."""
    lines = _audit_lines()
    heads = [i for i, line in enumerate(lines) if line.startswith("## ")]
    out: list[tuple[str, list[tuple[int, str]]]] = []
    for index, start in enumerate(heads):
        end = heads[index + 1] if index + 1 < len(heads) else len(lines)
        rows = [
            (i + 1, lines[i])
            for i in range(start + 1, end)
            if lines[i].strip().startswith("|")
        ]
        if rows:
            out.append((lines[start][3:].strip(), rows))
    return out


@requires_audit
class TestAmbiguityAudit:
    """The audit's tables must stay parseable, and its cross-references must resolve."""

    def test_every_table_row_has_the_declared_columns(self):
        """A row with the wrong cell count renders as a mangled table and raises nothing.

        The usual cause is an unescaped ``|`` inside an inline code span — ``|V_ei ∩ V_ej|`` rather than
        ``\\|V_ei ∩ V_ej\\|``. Since every audit table shares one column layout, a wrong count is always a
        defect and never a legitimate variation.
        """
        offenders = [
            (section, line_no, _cell_count(row), row)
            for section, rows in _tables()
            for line_no, row in rows
            if _cell_count(row) != AUDIT_COLUMNS
        ]
        assert not offenders, "\n".join(
            f"  {section} line {line_no}: {count} cells (expected {AUDIT_COLUMNS})\n    {row[:160]}"
            for section, line_no, count, row in offenders
        )

    def test_every_row_is_closed(self):
        """``| a | b`` with no trailing pipe is the other half of the same failure mode."""
        offenders = [
            (section, line_no, row)
            for section, rows in _tables()
            for line_no, row in rows
            if not row.rstrip().endswith("|")
        ]
        assert not offenders, "\n".join(
            f"  {section} line {line_no}: {row[:160]}" for section, line_no, row in offenders
        )

    def test_no_literal_escape_sequences(self):
        r"""A ``\n`` that reached the file as two characters is invisible in the rendered page.

        It shows up as a row fused to the next one, which the column check would also catch — but only
        because the two rows then have double the cells. Checking directly says *why*.
        """
        offenders = [
            (line_no, line)
            for line_no, line in enumerate(_audit_lines(), 1)
            if re.search(r"\\[nt]", line)
        ]
        assert not offenders, "\n".join(
            f"  line {line_no}: {line[:160]}" for line_no, line in offenders
        )

    def test_cited_tests_exist(self):
        """The audit cites test ids as evidence; a citation that resolves nowhere is worse than none.

        Only ``tests/``-relative ids are checked, because those are the ones a reader can run with one
        command. Wording inside backticks is matched loosely — the guard is against a rename or a typo
        leaving a dangling reference, not against prose.
        """
        pattern = re.compile(r"tests/test_([a-z_]+)\.py::(\w+)::(\w+)")
        lines = _audit_lines()
        cited = sorted({(m.group(1), m.group(2), m.group(3)) for m in pattern.finditer("\n".join(lines))})
        assert cited, "the audit should cite at least one test as evidence"

        missing: list[str] = []
        for module, class_name, method in cited:
            path = REPO_ROOT / "tests" / f"test_{module}.py"
            if not path.exists():
                missing.append(f"tests/test_{module}.py does not exist")
                continue
            text = path.read_text(encoding="utf-8")
            if f"class {class_name}" not in text:
                missing.append(f"tests/test_{module}.py has no class {class_name}")
            elif f"def {method}" not in text:
                missing.append(f"tests/test_{module}.py::{class_name} has no {method}")
        assert not missing, "\n".join(f"  {item}" for item in missing)


@requires_audit
def test_audit_is_present_when_checked():
    """Guard the guard: if this file is being read, the audit parsed rather than being silently skipped."""
    assert _tables(), "the audit has no tables; the parser or the file is wrong"
    sections = [name for name, _ in _tables()]
    for expected in ("Retrieval", "Extraction", "Precedence construction (§A.3)"):
        assert expected in sections, f"the audit lost its {expected!r} table"


class TestContractDoc:
    """``data/CONTRACT.md`` is the other shipped markdown deliverable, and it is mostly tables.

    The same silent failure applies — an unescaped ``|`` inside a cell splits a row and renders a
    mangled table with no error anywhere — and it is arguably worse here, because CONTRACT.md is
    what a reader consults to reproduce the release. Unlike the audit it is not git-ignored, so
    these tests always run.

    The invariant is weaker than the audit's, and deliberately so: CONTRACT.md's fifteen tables
    have different layouts, so the expected width is each block's *own* header rather than one
    declared constant. That still catches the failure mode, since a split row is always wider than
    its header.
    """

    def test_the_contract_ships_and_is_not_empty(self):
        assert CONTRACT.exists(), "data/CONTRACT.md should ship with the repository"
        assert CONTRACT.read_text(encoding="utf-8").strip()

    def test_every_table_block_has_rows(self):
        """Guard the parser: a document with no detected tables would pass the checks below vacuously."""
        blocks = _table_blocks(CONTRACT.read_text(encoding="utf-8").split("\n"))
        assert len(blocks) >= 10, f"only {len(blocks)} table blocks found; the parser or the file is wrong"

    def test_every_row_matches_its_header_width(self):
        offenders: list[str] = []
        for block in _table_blocks(CONTRACT.read_text(encoding="utf-8").split("\n")):
            width = _cell_count(block[0][1])
            for line_no, row in block:
                if _cell_count(row) != width:
                    offenders.append(f"  line {line_no}: {_cell_count(row)} cells (header has {width})\n    {row[:160]}")
        assert not offenders, "a table row does not match its header:\n" + "\n".join(offenders)

    def test_every_row_is_closed(self):
        offenders = [
            (line_no, row)
            for block in _table_blocks(CONTRACT.read_text(encoding="utf-8").split("\n"))
            for line_no, row in block
            if not row.rstrip().endswith("|")
        ]
        assert not offenders, "\n".join(f"  line {line_no}: {row[:160]}" for line_no, row in offenders)

    def test_no_literal_escape_sequences(self):
        """A ``\\n`` that reached the file as two characters fuses two rows and is invisible when rendered."""
        offenders = [
            (line_no, line)
            for line_no, line in enumerate(CONTRACT.read_text(encoding="utf-8").split("\n"), 1)
            if re.search(r"\\[nt]", line)
        ]
        assert not offenders, "\n".join(f"  line {line_no}: {line[:160]}" for line_no, line in offenders)
