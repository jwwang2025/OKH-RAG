"""Run the retrieval oracle over CyPortQA and write ``reports/retrieval_oracle.{md,json}``.

The oracle answers a question §4.1's single accuracy number cannot: *did retrieval find the
evidence this question needs?* — see :mod:`okh_rag.evaluation.oracle`. This script is the driver,
and it deliberately separates the two halves of the answer, because only one of them needs a
model.

**Stage 1-3 need no API key and no generator.** The template/field spec, the reference
hypergraph, and the gold-set inventory over all 117,178 questions are pure data and closed-form
field access. They are what the run is *for*: they say exactly how much of the benchmark the
oracle can score, and they surface the release's structural defects with measured counts.
``--retrieve 0`` runs only these.

**Stage 4 needs a backend.** Retrieval is measured by §A.4's ``text-embedding-3-small``, so the
headline numbers require ``--backend openai``. ``--backend lexical`` substitutes a token-overlap
embedder defined in this file; it is *not* §A.4's model and its numbers must not be quoted as
method quality. What it is for is validating the instrument, and it does that by running the
same records under three §A.7 configurations whose expected relationship is known in advance:

============================  ==================================================
configuration                 expected
============================  ==================================================
``heuristic`` (§A.3 order)    the §4.5 reference row
``none`` (order-free)         content comparable, order worse
``heuristic`` + shuffle       content *identical* by construction, order worse still
============================  ==================================================

Those three give the harness a falsifiable claim: :attr:`~okh_rag.evaluation.OracleMetrics
.trajectory_recall` must be flat across the shuffle row — §4.5 states the drop happens "despite
identical retrieved content" — while ``order_concordance`` and ``first_gold_step`` must move.
An oracle whose order metrics do not separate those rows cannot measure §4.5's largest reported
effect (0.534 → 0.487), so this is a check on the instrument rather than on the method. It runs
with any embedder, which is why it runs by default.

Run from the repo root::

    D:/Anaconda/python.exe scripts/run_retrieval_oracle.py --retrieve 0
    D:/Anaconda/python.exe scripts/run_retrieval_oracle.py --backend openai --retrieve 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time

# Windows + Anaconda: pip-installed torch and MKL-backed numpy each ship their own
# libiomp5md.dll; loading both in one process aborts with "OMP: Error #15".
# Must be set before numpy/torch initialize their OpenMP runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from okh_rag.config import Config, load_config  # noqa: E402
from okh_rag.data.cyportqa import CyPortQACorpus, load_corpus  # noqa: E402
from okh_rag.embeddings import HyperedgeEmbeddings, embed_texts, hyperedge_embedding_text  # noqa: E402
from okh_rag.evaluation.oracle import (  # noqa: E402
    EvidenceStatus,
    OracleReport,
    evaluate_record,
    gold_for_record,
    gold_matches_field_provenance,
    template_coverage,
)
from okh_rag.evaluation.reference_graph import build_reference_graph  # noqa: E402
from okh_rag.llm.base import EmbeddingClient  # noqa: E402
from okh_rag.ordering.precedence import build_precedence, canonical_order, trajectory_order_index  # noqa: E402
from okh_rag.retrieval import OrderAwareRetriever  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The three §A.7 configurations stage 4 compares. ``(label, transition variant, shuffle)``.
CONFIGURATIONS: tuple[tuple[str, str, bool], ...] = (
    ("heuristic", "heuristic", False),
    ("no_order", "none", False),
    ("heuristic_shuffled", "heuristic", True),
)


# ---------------------------------------------------------------------------
# Diagnostic embedder
# ---------------------------------------------------------------------------


class LexicalEmbedder:
    """A deterministic token-overlap embedder. **Not** §A.4's model — a diagnostic stand-in.

    §A.4 fixes the embedder at ``text-embedding-3-small``, and nothing here can reproduce a
    semantic space: a hashed bag of tokens cannot know that "tropical storm watch" and
    ``under_watch`` are the same fact. What it does preserve is *lexical* overlap, which is
    enough for the instrument check the module docstring describes — the three §A.7
    configurations are compared against each other on the same corpus, so an embedding model
    that ranks poorly still ranks *consistently*, and the order metrics remain meaningful.

    Two properties are non-negotiable and both come from the hashing being keyed on the text:
    determinism across runs, and no dependency on numpy's RNG or Python's ``hash()``
    (which is salted per process and would make the corpus differ between runs).

    The output is L2-normalised so that :func:`~okh_rag.retrieval.scoring.cosine_relevance`,
    which is a plain dot product, is a real cosine. A token counts once per distinct token and
    is signed by a second hash, which keeps collisions from adding up: without the sign, two
    unrelated tokens landing in the same bucket would look like shared evidence.
    """

    _TOKEN_RE = re.compile(r"[a-z0-9]+")

    def __init__(self, dim: int = 1536) -> None:
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        rows = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in set(self._TOKEN_RE.findall(text.lower())):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                value = int.from_bytes(digest, "big")
                column = value % self._dim
                # The top bit chooses the sign, so a collision cancels as often as it compounds.
                rows[row, column] += 1.0 if value >> 63 else -1.0
            norm = float(np.linalg.norm(rows[row]))
            if norm > 0.0:
                rows[row] /= norm
        return rows


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------


def _select_scenarios(corpus: CyPortQACorpus, limit: int | None, seed: int) -> tuple[str, ...]:
    """The knowledge groups to build and retrieve over.

    A subset is not just a speed knob. §A.4's embeddings are ``(N, 1536)`` float32, which for the
    full 248,652-hyperedge reference graph is 1.5 GB — payable with real ``text-embedding-3-small``
    vectors on disk, but not as an in-process diagnostic. The oracle's metrics are per-record and
    already normalised by gold size, so they do not depend on corpus size; only the *pool
    competition* does, and that is reported alongside every number as ``pool_size``.
    """
    ids = sorted(corpus.encoded)
    if limit is None or limit >= len(ids):
        return tuple(ids)
    return tuple(sorted(random.Random(seed).sample(ids, limit)))


def _gold_inventory(
    corpus: CyPortQACorpus, graph, scenario_ids: Iterable[str]
) -> dict[str, Any]:
    """Classify every question by whether the oracle can score it. Model-free and exact."""
    wanted = set(scenario_ids)
    statuses: dict[str, int] = {}
    per_template: dict[str, dict[str, int]] = {}
    sizes: dict[str, int] = {}
    empty_notes: dict[str, int] = {}
    provenance_mismatches: list[str] = []
    scorable = 0
    total = 0

    for record in corpus.records():
        if record.scenario_id not in wanted:
            continue
        total += 1
        gold = gold_for_record(record, corpus.templates[record.template_id], graph=graph)
        statuses[gold.status.value] = statuses.get(gold.status.value, 0) + 1
        bucket = per_template.setdefault(record.template_id, {})
        bucket[gold.status.value] = bucket.get(gold.status.value, 0) + 1

        if gold.is_scorable:
            scorable += 1
            key = str(min(gold.size, 5))
            sizes[key] = sizes.get(key, 0) + 1
            # The oracle's own consistency check: on this graph the relation-keyed gold set and
            # the field-keyed one must agree. A mismatch is a bug in the oracle, not in the data.
            if not gold_matches_field_provenance(gold, graph):
                provenance_mismatches.append(gold.qa_id)
        elif gold.status is EvidenceStatus.EMPTY:
            empty_notes[gold.note] = empty_notes.get(gold.note, 0) + 1
        elif gold.status is EvidenceStatus.UNMAPPED:
            empty_notes[gold.note] = empty_notes.get(gold.note, 0) + 1

    return {
        "n_questions": total,
        "n_scorable": scorable,
        "scorable_rate": round(scorable / total, 4) if total else 0.0,
        "statuses": dict(sorted(statuses.items())),
        "gold_size_histogram_capped_at_5": dict(sorted(sizes.items(), key=lambda kv: int(kv[0]))),
        "unscorable_reasons": dict(sorted(empty_notes.items(), key=lambda kv: -kv[1])),
        "per_template": dict(sorted(per_template.items(), key=lambda kv: int(kv[0][1:]))),
        "provenance_mismatches": provenance_mismatches[:20],
        "n_provenance_mismatches": len(provenance_mismatches),
    }


def _build_embeddings(
    graph, *, cfg: Config, backend: EmbeddingClient, label: str
) -> HyperedgeEmbeddings:
    """§A.4's embedding input, embedded by ``backend``.

    The store is intentionally *not* used for a diagnostic backend. ``EmbeddingStore`` is keyed on
    the sha256 of the text alone, with no record of which model produced the vector, so writing
    ``LexicalEmbedder`` output into ``artifacts/llm_cache/embeddings`` would let a later
    ``text-embedding-3-small`` run read token-overlap vectors back for the same strings and
    measure a corpus that was never embedded by §A.4's model. Using §A.4's cache only for §A.4's
    model is the safe reading of a store that cannot say which model it holds.
    """
    ordered_ids = tuple(sorted(graph.hypergraph.hyperedges))
    texts = [hyperedge_embedding_text(graph.hypergraph.hyperedges[hid], graph.hypergraph) for hid in ordered_ids]
    print(f"  embedding {len(texts)} hyperedges with backend={label} ...")
    matrix = embed_texts(texts, backend=backend, cfg=cfg, store=None)
    return HyperedgeEmbeddings(hyperedge_ids=ordered_ids, matrix=matrix)


def _run_configuration(
    corpus: CyPortQACorpus,
    graph,
    *,
    cfg: Config,
    precedence,
    embeddings: HyperedgeEmbeddings,
    query_vectors: dict[str, np.ndarray],
    canonical_position: dict[str, int],
    records: Sequence[Any],
    variant: str,
    shuffle: bool,
    label: str,
) -> OracleReport:
    """Retrieve and evaluate every sampled record under one §A.7 configuration.

    Two arguments exist to keep one run's cost from being paid three times:

    * ``precedence`` is passed in rather than rebuilt. ``≺`` is a property of the hypergraph alone
      and §A.6 Eq. 11 reads it on every transition evaluation, so rebuilding it per configuration
      would triple the most expensive precomputation in the run without changing an answer.
    * ``query_vectors`` is keyed by ``qa_id`` and computed once for the sample. The same question
      is retrieved under all three configurations, and §A.4's embedder is a network call — so
      embedding inside this loop would bill §A.7's ablation study three times for one vector.
    """
    run_cfg = replace(cfg, ablation=replace(cfg.ablation, shuffle_trajectory=shuffle))
    retriever = OrderAwareRetriever(
        cfg=run_cfg,
        hypergraph=graph.hypergraph,
        precedence=precedence,
        embeddings=embeddings,
        backend=None,
        transition_variant=variant,
        inference="beam_search",
    )
    report = OracleReport(backend=label)
    started = time.time()
    for index, record in enumerate(records, start=1):
        result = retriever.retrieve(
            record.question,
            group_id=record.scenario_id,
            query_vector=query_vectors[record.qa_id],
        )
        report.outcomes.append(
            evaluate_record(
                record,
                corpus.templates[record.template_id],
                result,
                graph=graph,
                canonical_position=canonical_position,
            )
        )
        if index % 25 == 0 or index == len(records):
            print(f"    [{label}] {index}/{len(records)} ({time.time() - started:.1f}s)")
    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _table(rows: Sequence[Sequence[Any]], header: Sequence[str]) -> str:
    out = ["| " + " | ".join(str(c) for c in header) + " |"]
    out.append("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _render_markdown(payload: dict[str, Any]) -> str:
    """Render the JSON payload as the human-facing report."""
    lines: list[str] = []
    add = lines.append

    add("# Retrieval oracle — §A.7 scored against what each question needs")
    add("")
    add(
        "Generated by `scripts/run_retrieval_oracle.py`. The oracle's definition and its metrics "
        "are documented in `src/okh_rag/evaluation/oracle.py`; the gold sets come from "
        "`CyPortQA_template.json` via `CONTRACT.md` §3.2."
    )
    add("")

    add("## 1. Template coverage — a static property of the spec")
    add("")
    coverage = payload["template_coverage"]
    add(
        f"{coverage['resolved']} of {coverage['total']} templates resolve to an §A.1 Table 3 "
        f"relation, {coverage['entity']} name a persistent object instead, "
        f"{coverage['unmapped']} are uncovered."
    )
    add("")
    if coverage["unmapped_templates"]:
        add("Uncovered templates — these cannot be scored on any hypergraph:")
        add("")
        add(_table([[k, v] for k, v in coverage["unmapped_templates"].items()], ["template", "declared field"]))
        add("")

    add("## 2. Reference hypergraph — the yardstick")
    add("")
    stats = payload["reference_graph"]
    add(
        _table(
            [
                ["scenarios", stats["scenarios"]],
                ["horizon blocks", stats["horizon_blocks"]],
                ["fields seen", stats["fields_seen"]],
                ["fields kept", stats["fields_kept"]],
                ["coverage", f"{stats['coverage']:.4f}"],
                ["hyperedges", stats["hyperedges"]],
                ["… of which cross-horizon (family 13)", stats["hyperedges_by_relation"].get("forecast_updates_to", 0) + stats["hyperedges_by_relation"].get("changes_probability_to", 0)],
                ["entities", stats["entities"]],
                ["fields with an empty value", stats["fields_empty_value"]],
                ["hyperedge ids aliased across groups", stats["id_collisions"]["ids"]],
            ],
            ["quantity", "value"],
        )
    )
    add("")
    if stats["anomalies"]:
        add("**Structural anomalies in the encoding.** Each is a field that names no §3.2 row, "
            "so it becomes no hyperedge:")
        add("")
        add(_table([[k, v] for k, v in stats["anomalies"].items()], ["anomaly", "count"]))
        add("")

    add("## 3. Gold-set inventory — how much of the benchmark the oracle can score")
    add("")
    inventory = payload["gold_inventory"]
    add(
        _table(
            [[k, v] for k, v in inventory["statuses"].items()]
            + [["scorable rate", f"{inventory['scorable_rate']:.4f}"]],
            ["status", "questions"],
        )
    )
    add("")
    add("Gold-set size, capped at 5 (questions needing more than five hyperedges are one row):")
    add("")
    add(
        _table(
            [[k, v] for k, v in inventory["gold_size_histogram_capped_at_5"].items()],
            # Escaped, not literal: an unescaped `|` inside a table cell splits the row in Markdown
            # and does so silently, which is how `reports/cyportqa_audit.md` lost a column before
            # `tests/test_docs.py` started checking cell counts.
            [r"\|gold\|", "questions"],
        )
    )
    add("")
    if inventory["unscorable_reasons"]:
        add("Why the unscorable ones are unscorable:")
        add("")
        add(_table([[k, v] for k, v in inventory["unscorable_reasons"].items()], ["reason", "questions"]))
        add("")
    add(
        f"Oracle self-check: relation-keyed gold versus field-keyed gold, over every scorable "
        f"question — **{inventory['n_provenance_mismatches']} mismatches** "
        f"(of {inventory['n_scorable']})."
    )
    add("")

    retrieval = payload.get("retrieval")
    if not retrieval:
        add("## 4. Retrieval — not run")
        add("")
        add("`--retrieve 0` was passed, so only the model-free stages ran.")
        return "\n".join(lines) + "\n"

    add("## 4. Retrieval — §A.7 measured against the gold sets")
    add("")
    add(f"Backend: `{payload['backend']}`.")
    add("")
    if payload["backend"] == "lexical":
        add(
            "> **These are instrument checks, not method measurements.** The backend is a token-overlap "
            "stand-in for §A.4's `text-embedding-3-small`, so the absolute values say nothing about "
            "OKH-RAG's quality. Their use is the *comparison between the three rows below*: the "
            "shuffle row holds retrieved content fixed by construction, so recall must not move "
            "while the order metrics must."
        )
        add("")

    rows = []
    for label, metrics in retrieval["configurations"].items():
        rows.append(
            [
                label,
                metrics.get("pool_recall"),
                metrics.get("anchor_recall"),
                metrics.get("trajectory_recall"),
                metrics.get("chance_recall"),
                metrics.get("trajectory_recall_over_chance"),
                metrics.get("first_gold_step_mean"),
                metrics.get("canonical_concordance_mean"),
                metrics.get("gold_order_concordance_mean"),
            ]
        )
    add(
        _table(
            rows,
            [
                "configuration",
                "pool recall",
                "anchor recall",
                "trajectory recall",
                "chance recall",
                "over chance",
                "first gold step",
                "canonical concordance",
                "gold concordance",
            ],
        )
    )
    add("")
    add(
        "`chance recall` is `pool recall × L / |C|` — what a trajectory of the same length drawn "
        "uniformly from the pool would score. It is the floor `trajectory recall` has to be read "
        "against: the two numbers are the same to two decimals on this backend, which says the beam "
        "is selecting gold at chance and that the absolute recall is a statement about the "
        "embedder, not about §A.7. `over chance` is the ratio, and 1.0 means chance."
    )
    add("")
    add(
        "`canonical concordance` is the agreement between the retrieved sequence and §A.3's "
        "canonical order over all its placed members — 1.0 is perfect, and shuffling `L` members "
        "gives 0.5 in expectation, so it is the metric §4.5's shuffle row must move. "
        "`gold concordance` restricts the same comparison to the gold members and is `None` on most "
        "records at realistic recall; its denominator (`_n`) is in the JSON."
    )
    add("")
    add(f"Records retrieved and scored per configuration: {retrieval['n_records']}.")
    add("")
    add("### Per-template trajectory recall")
    add("")
    reference = next(iter(retrieval["configurations"]))
    add(
        _table(
            [
                [tid, row["n"], row["trajectory_recall"], row["pool_recall"], row["relation"], row["family"] or ""]
                for tid, row in retrieval["configurations"][reference]["by_template"].items()
            ],
            ["template", "n", "trajectory recall", "pool recall", "relation", "family"],
        )
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--data", default=str(REPO_ROOT / "data" / "raw" / "cyportqa"))
    parser.add_argument("--out", default=str(REPO_ROOT / "reports"))
    parser.add_argument(
        "--backend",
        choices=("lexical", "openai"),
        default="lexical",
        help="lexical (offline, diagnostic) or openai (§A.4's text-embedding-3-small)",
    )
    parser.add_argument("--scenario-limit", type=int, default=48, help="groups to build and retrieve over")
    parser.add_argument("--retrieve", type=int, default=120, help="records to retrieve and score; 0 for stages 1-3 only")
    parser.add_argument("--seed", type=int, default=None, help="defaults to config seed")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("== loading the release ==")
    started = time.time()
    corpus = load_corpus(data_dir)
    print(f"  {len(corpus.encoded)} scenarios, {sum(len(v) for v in corpus.qa.values())} questions "
          f"({time.time() - started:.1f}s)")

    scenario_ids = _select_scenarios(corpus, args.scenario_limit, seed)
    print(f"== building the reference hypergraph over {len(scenario_ids)} groups ==")
    started = time.time()
    graph = build_reference_graph(corpus, cfg=cfg, scenario_ids=scenario_ids)
    print(f"  {len(graph.hypergraph)} hyperedges, {len(graph.hypergraph.entities)} entities "
          f"({time.time() - started:.1f}s)")

    coverage = template_coverage(corpus.templates)
    print(f"== template coverage: {len(coverage.resolved)} relation, {len(coverage.entities)} entity, "
          f"{len(coverage.unmapped)} uncovered ==")

    print("== gold-set inventory ==")
    started = time.time()
    inventory = _gold_inventory(corpus, graph, scenario_ids)
    print(f"  {inventory['n_scorable']}/{inventory['n_questions']} scorable "
          f"({inventory['scorable_rate']:.4f}), {inventory['n_provenance_mismatches']} provenance mismatches "
          f"({time.time() - started:.1f}s)")

    payload: dict[str, Any] = {
        "config": args.config,
        "data": str(data_dir),
        "seed": seed,
        "scenario_limit": args.scenario_limit,
        "template_coverage": coverage.as_dict(),
        "reference_graph": graph.stats.as_dict(),
        "gold_inventory": inventory,
        "backend": args.backend,
        "retrieval": None,
    }

    if args.retrieve > 0:
        print("== retrieval ==")
        if args.backend == "openai":
            from okh_rag.llm.openai_client import OpenAIBackend

            # OpenAIBackend reads embedding_model / batch / retry settings off the llm section;
            # passing the root Config crashes on the first embed call with an AttributeError.
            backend: EmbeddingClient = OpenAIBackend(cfg.llm)
        else:
            backend = LexicalEmbedder(dim=cfg.order_model.embedding_dim)

        embeddings = _build_embeddings(graph, cfg=cfg, backend=backend, label=args.backend)
        precedence = build_precedence(graph.hypergraph, cfg.precedence)
        canonical_position = trajectory_order_index(canonical_order(graph.hypergraph, precedence))
        pool = [
            record
            for record in corpus.records()
            if record.scenario_id in set(scenario_ids)
        ]
        sample = random.Random(seed).sample(pool, min(args.retrieve, len(pool)))
        sample.sort(key=lambda r: r.qa_id)
        print(f"  sampled {len(sample)} questions from {len(pool)}")

        print(f"  embedding {len(sample)} query questions ...")
        query_vectors = {
            record.qa_id: vector
            for record, vector in zip(
                sample,
                embed_texts([r.question for r in sample], backend=backend, cfg=cfg, store=None),
            )
        }

        configurations: dict[str, dict[str, Any]] = {}
        for label, variant, shuffle in CONFIGURATIONS:
            print(f"  -- configuration {label} --")
            report = _run_configuration(
                corpus,
                graph,
                cfg=cfg,
                precedence=precedence,
                embeddings=embeddings,
                query_vectors=query_vectors,
                canonical_position=canonical_position,
                records=sample,
                variant=variant,
                shuffle=shuffle,
                label=label,
            )
            report.graph_stats = graph.stats.as_dict()
            configurations[label] = {
                **report.mean_metrics(),
                "status_counts": report.status_counts(),
                "by_template": report.by_template(),
                "by_horizon": report.by_horizon(),
                "empty_gold_by_cause": report.empty_gold_by_cause(),
            }
            print(f"     {json.dumps(report.mean_metrics(), default=str)}")

        payload["retrieval"] = {
            "n_records": len(sample),
            "configurations": configurations,
        }

    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["generated_at"] = stamp
    json_path = out_dir / "retrieval_oracle.json"
    md_path = out_dir / "retrieval_oracle.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"\nwrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
