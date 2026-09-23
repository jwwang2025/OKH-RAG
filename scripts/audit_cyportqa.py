"""Audit the released CyPortQA benchmark against the paths its own templates declare.

Run from the repo root:

    D:/Anaconda/python.exe -m scripts.audit_cyportqa --data data/raw/cyportqa --out reports

Writes ``reports/cyportqa_audit.md`` and ``reports/cyportqa_audit.json``.

Why this is a deliverable and not a debugging step: OKH-RAG's §4.1 hands every dataset
detail to CyPortQA's own paper and says nothing about preprocessing, splits or prompts, so
the only way to know what the reported numbers were computed over is to read the release.
Doing that turns up a per-template map from each question to the exact field that answers
it — which is the retrieval oracle §3.2 needs — and also a set of label defects that a
naive reproduction would silently inherit as method error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from okh_rag.data.audit import build_report, write_report  # noqa: E402
from okh_rag.data.cyportqa import KNOWN_DEFECTS, load_corpus  # noqa: E402

DEFAULT_DATA = Path("data/raw/cyportqa")
DEFAULT_OUT = Path("reports")


def _resolve_data_root(candidate: Path) -> Path:
    """Accept either the CyPortQA repo root or a directory laid out like it.

    Two layouts are in the wild: a full ``git clone`` of MLLM-Bench-CyPortQA, and the
    trimmed copy this project ships under ``data/raw/cyportqa``. Both put ``dataset/`` and
    ``source_data/`` at the root, so the check is the same either way; the extra probe is
    for a caller who points at the repository's parent by mistake.
    """
    probes = [
        candidate,
        candidate / "MLLM-Bench-CyPortQA",
        candidate / "cyportqa",
    ]
    for probe in probes:
        if (probe / "dataset" / "CyPortQA.json").exists() and (probe / "source_data" / "Encoded_senario.json").exists():
            return probe
    raise SystemExit(
        f"no CyPortQA checkout under {candidate}; expected dataset/CyPortQA.json and "
        "source_data/Encoded_senario.json. See data/CONTRACT.md for how to place the files."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="CyPortQA checkout root")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="directory for the report")
    parser.add_argument("--quiet", action="store_true", help="write the report without echoing a summary")
    args = parser.parse_args(argv)

    root = _resolve_data_root(args.data)
    if not args.quiet:
        print(f"loading {root} ...")

    corpus = load_corpus(root)
    counts = corpus.counts()
    if not args.quiet:
        print(
            f"  scenarios={counts['scenarios']} horizon_blocks={counts['horizon_blocks']} "
            f"questions={counts['questions']}"
        )
        print(f"  ports={counts['ports']} storms={counts['storms']}")
        print(f"  by format: {counts['answers_by_format']}")
        print(f"  by stage:  {counts['questions_by_stage']}")
        print(f"  unanswerable (placeholder answer): {counts['questions_unanswerable']}")

    report = build_report(root)
    markdown_path, json_path = write_report(root, args.out)

    if not args.quiet:
        print()
        print("template agreement (decidable comparisons only):")
        worst = sorted(
            (entry for entry in report["templates"] if entry["agreement"] is not None),
            key=lambda entry: entry["agreement"],
        )[:6]
        for entry in worst:
            print(
                f"  {entry['template_id']:4} {entry['answer_format']:3} "
                f"agreement={entry['agreement']:.3f}  over {entry['n_records']:5} records  "
                f"path={entry['declared_path'] or '(unresolved)'}"
            )

        print()
        print("defect re-measurement:")
        for defect in KNOWN_DEFECTS:
            entry = report["defects"][defect.defect_id]
            mark = "OK " if entry["verdict"] == "reproduced" else "!! "
            print(f"  {mark}{defect.defect_id} [{defect.severity.value}] -> {entry['verdict']}")

        unreproduced = [
            defect.defect_id
            for defect in KNOWN_DEFECTS
            if report["defects"][defect.defect_id]["verdict"] != "reproduced"
        ]
        print()
        if unreproduced:
            print(f"NOT REPRODUCED: {', '.join(unreproduced)}")

        print()
        print(f"wrote {markdown_path}")
        print(f"wrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
