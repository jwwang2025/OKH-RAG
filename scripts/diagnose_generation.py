"""Diagnose the gap between the reproduced Table 1 and the paper's.

Runs the §4.1 driver's exact pipeline but records per-question detail, so a low column can be
attributed to retrieval, to the generator, or to the scorer instead of guessed at.

Usage::

    python scripts/diagnose_generation.py --retrieve 200
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from okh_rag.config import load_config  # noqa: E402
from okh_rag.data.cyportqa import load_corpus  # noqa: E402
from okh_rag.embeddings import (  # noqa: E402
    EmbeddingStore,
    HyperedgeEmbeddings,
    embed_texts,
    hyperedge_embedding_text,
)
from okh_rag.evaluation.answers import score_record  # noqa: E402
from okh_rag.evaluation.reference_graph import build_reference_graph  # noqa: E402
from okh_rag.generation.evidence import context_from_retrieval  # noqa: E402
from okh_rag.generation.generator import Generator  # noqa: E402
from okh_rag.llm import OpenAIBackend  # noqa: E402
from okh_rag.ordering.model import LearnedTransitionModel  # noqa: E402
from okh_rag.ordering.precedence import build_precedence  # noqa: E402
from okh_rag.retrieval import OrderAwareRetriever  # noqa: E402
from run_generation import select_records  # noqa: E402


def _norm(text: object) -> str:
    return " ".join(str(text).split()).casefold()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--data", default=str(REPO_ROOT / "data" / "raw" / "cyportqa"))
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "generation_diagnosis.json"))
    parser.add_argument(
        "--scenario-limit",
        type=int,
        default=0,
        help="groups to build over; sampled with --seed from all 472. 0 = the full release",
    )
    parser.add_argument(
        "--retrieve", type=int, default=200, help="questions to answer. 0 = all 117,178"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--transition-variant",
        default=None,
        choices=("learned", "heuristic", "none"),
    )
    parser.add_argument(
        "--td-scoring", default=None, choices=("exact_match", "llm_judge")
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    import dataclasses

    if args.transition_variant:
        cfg = dataclasses.replace(
            cfg,
            ablation=dataclasses.replace(
                cfg.ablation, transition_variant=args.transition_variant
            ),
        )
    if args.td_scoring:
        cfg = dataclasses.replace(
            cfg, evaluation=dataclasses.replace(cfg.evaluation, td_scoring=args.td_scoring)
        )
    corpus = load_corpus(Path(args.data))
    records = select_records(
        corpus,
        scenario_limit=args.scenario_limit or None,
        question_limit=args.retrieve or None,
        seed=args.seed,
    )
    scenarios = sorted({r.scenario_id for r in records})
    print(f"{len(records)} questions over {len(scenarios)} scenarios: {scenarios}")

    graph = build_reference_graph(corpus, cfg=cfg, scenario_ids=scenarios)
    hypergraph = graph.hypergraph
    print(f"hypergraph: {len(hypergraph.hyperedges)} hyperedges")

    store = EmbeddingStore(REPO_ROOT / "artifacts" / "llm_cache" / "embeddings", dim=1536)
    ordered = tuple(sorted(hypergraph.hyperedges))
    matrix = embed_texts(
        [hyperedge_embedding_text(hypergraph.hyperedges[h], hypergraph) for h in ordered],
        backend=OpenAIBackend(cfg.llm),
        cfg=cfg,
        store=store,
    )
    embeddings = HyperedgeEmbeddings(hyperedge_ids=ordered, matrix=matrix)

    order_model = None
    checkpoint = REPO_ROOT / "artifacts" / "order_model.pt"
    if cfg.ablation.transition_variant == "learned" and checkpoint.is_file():
        order_model = LearnedTransitionModel.load(checkpoint, cfg=cfg.order_model)

    retriever = OrderAwareRetriever(
        cfg=cfg,
        hypergraph=hypergraph,
        precedence=build_precedence(hypergraph, cfg.precedence),
        embeddings=embeddings,
        order_model=order_model,
        backend=None,
    )

    backend = OpenAIBackend(cfg.llm)
    query_vectors = embed_texts(
        [r.question for r in records], backend=backend, cfg=cfg, store=None
    )
    generator = Generator(backend, cfg)

    rows = []
    started = time.time()
    for record, qv in zip(records, query_vectors):
        retrieval = retriever.retrieve(
            record.question,
            group_id=record.group_id,
            query_vector=qv,
            horizon_hours=record.context.horizon_hours,
        )
        context = context_from_retrieval(
            retrieval,
            hypergraph,
            question=record.question,
            include_quality_summary=cfg.generation.include_quality_summary,
            include_reasoning_tags=cfg.generation.include_reasoning_tags,
        )
        rendered = context.render()
        generated = generator.answer(record, context, mode=cfg.generation.mode)
        score = score_record(record, generated.answer.answer, cfg.evaluation)

        gold_text = str(record.answer)
        rows.append(
            {
                "qa_id": record.qa_id,
                "template_id": record.template_id,
                "scenario_id": record.scenario_id,
                "format": record.answer_format.value,
                "question": record.question,
                "gold": record.answer,
                "predicted": generated.answer.answer,
                "correct": score.correct,
                "reason": score.reason,
                "detail": score.detail,
                "conforming": generated.conforming,
                "violations": [v.value for v in generated.answer.violations],
                "confidence": generated.answer.confidence,
                "gold_in_context": bool(gold_text.strip())
                and _norm(gold_text) in _norm(rendered),
                "n_steps": sum(len(t) for t in context.trajectories),
                "n_trajectories": len(context.trajectories),
                "evidence_chars": len(rendered),
            }
        )
    elapsed = time.time() - started
    print(f"generated {len(rows)} answers in {elapsed:.1f}s")

    payload = {
        "config": {
            "generator_model": cfg.llm.generator_model,
            "generation_temperature": cfg.llm.generation_temperature,
            "td_scoring": cfg.evaluation.td_scoring,
            "overall": cfg.evaluation.overall,
            "sa_relative_tolerance": cfg.evaluation.sa_relative_tolerance,
            "transition_variant": cfg.ablation.transition_variant,
            "mode": cfg.generation.mode,
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")

    _summarise(rows)
    return 0


def _rate(num: int, den: int) -> str:
    return "—" if den == 0 else f"{num}/{den} = {num / den:.1%}"


def _summarise(rows) -> None:
    by_fmt = collections.defaultdict(list)
    for row in rows:
        by_fmt[row["format"]].append(row)

    print("\n=== per-format accuracy (reproduced) ===")
    for fmt in sorted(by_fmt):
        group = by_fmt[fmt]
        scored = [r for r in group if r["correct"] is not None]
        hits = sum(1 for r in scored if r["correct"] >= 1.0)
        partial = sum(r["correct"] for r in scored) / len(scored) if scored else 0.0
        print(
            f"  {fmt}: exact={_rate(hits, len(scored))}  mean={partial:.3f}  "
            f"excluded={len(group) - len(scored)}"
        )
    allscored = [r for r in rows if r["correct"] is not None]
    print(
        f"  ALL: micro={sum(r['correct'] for r in allscored) / max(1, len(allscored)):.3f}"
    )

    print("\n=== accuracy split by whether the gold string is in the retrieved context ===")
    for fmt in sorted(by_fmt):
        group = [r for r in by_fmt[fmt] if r["correct"] is not None]
        inside = [r for r in group if r["gold_in_context"]]
        outside = [r for r in group if not r["gold_in_context"]]
        if not inside and not outside:
            continue
        print(
            f"  {fmt}: in-context {_rate(sum(1 for r in inside if r['correct'] >= 1), len(inside))}"
            f" | not-in-context {_rate(sum(1 for r in outside if r['correct'] >= 1), len(outside))}"
        )

    print("\n=== accuracy split by output-contract conformance ===")
    for fmt in sorted(by_fmt):
        group = [r for r in by_fmt[fmt] if r["correct"] is not None]
        ok = [r for r in group if r["conforming"]]
        bad = [r for r in group if not r["conforming"]]
        print(
            f"  {fmt}: conforming {_rate(sum(1 for r in ok if r['correct'] >= 1), len(ok))}"
            f" | violating {_rate(sum(1 for r in bad if r['correct'] >= 1), len(bad))}"
        )

    violations = collections.Counter(
        v for r in rows for v in r["violations"]
    )
    print("\n=== contract violations ===")
    for name, count in violations.most_common():
        print(f"  {name}: {count}")
    print(f"  questions with >=1 violation: {sum(1 for r in rows if not r['conforming'])}/{len(rows)}")


if __name__ == "__main__":
    raise SystemExit(main())
