"""SA (numeric) column diagnostic — is retrieval scoped to the question's horizon block?

CyPortQA asks the *same* question text once per horizon block of a scenario
(``T-12``, ``T-24``, ``T-36``, ``T-48`` ...). The gold answer is a function of the block.
``QARecord.group_id`` is the scenario only, so a retriever called with just ``group_id``
sees every block of the storm at once and — because the query text is byte-identical —
returns the *same* trajectory for all of them.

This script measures that directly and tests the fix:

    python scripts/diagnose_sa.py --retrieve 200
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

from okh_rag.config import load_config  # noqa: E402
from okh_rag.data.cyportqa import load_corpus  # noqa: E402
from okh_rag.embeddings import (  # noqa: E402
    EmbeddingStore,
    HyperedgeEmbeddings,
    embed_texts,
    hyperedge_embedding_text,
)
from okh_rag.evaluation.oracle import gold_for_record  # noqa: E402
from okh_rag.evaluation.reference_graph import build_reference_graph  # noqa: E402
from okh_rag.llm import OpenAIBackend  # noqa: E402
from okh_rag.ordering.model import LearnedTransitionModel  # noqa: E402
from okh_rag.ordering.precedence import build_precedence  # noqa: E402
from okh_rag.retrieval import OrderAwareRetriever  # noqa: E402
from run_generation import select_records  # noqa: E402


def normalise(value: object) -> str:
    return " ".join(str(value).split()).casefold()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    parser.add_argument("--scenario-limit", type=int, default=24)
    parser.add_argument("--retrieve", type=int, default=200)
    parser.add_argument("--out", default=str(ROOT / "reports/sa_horizon_diagnosis.json"))
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    corpus = load_corpus(ROOT / "data" / "raw" / "cyportqa")
    records = select_records(corpus, scenario_limit=args.scenario_limit, question_limit=args.retrieve)
    scen = sorted({r.scenario_id for r in records})
    graph = build_reference_graph(corpus, cfg=cfg, scenario_ids=scen)
    hg = graph.hypergraph

    store = EmbeddingStore(ROOT / "artifacts/llm_cache/embeddings", dim=1536)
    backend = OpenAIBackend(cfg.llm)
    ordered = tuple(sorted(hg.hyperedges))
    matrix = embed_texts(
        [hyperedge_embedding_text(hg.hyperedges[h], hg) for h in ordered],
        backend=backend,
        cfg=cfg,
        store=store,
    )
    emb = HyperedgeEmbeddings(hyperedge_ids=ordered, matrix=matrix)
    model = LearnedTransitionModel.load(ROOT / "artifacts/order_model.pt", cfg=cfg.order_model)
    retriever = OrderAwareRetriever(
        cfg=cfg,
        hypergraph=hg,
        precedence=build_precedence(hg, cfg.precedence),
        embeddings=emb,
        order_model=model,
        backend=None,
    )
    qstore = EmbeddingStore(ROOT / "artifacts/llm_cache/questions", dim=1536)
    qvecs = embed_texts([r.question for r in records], backend=backend, cfg=cfg, store=qstore)

    # ---- 1. how duplicated is the question text? -------------------------------
    groups: dict[tuple[str, str], list] = collections.defaultdict(list)
    for r in records:
        groups[(r.scenario_id, r.question)].append(r)
    dup = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"questions: {len(records)}   distinct texts: {len(groups)}   duplicated texts: {len(dup)}")
    print(f"questions sitting in a duplicated group: {sum(len(v) for v in dup.values())}")
    sizes = collections.Counter(len(v) for v in dup.values())
    print("group sizes:", dict(sorted(sizes.items())))

    by_fmt = collections.defaultdict(lambda: [0, 0])
    for k, v in dup.items():
        f = v[0].answer_format.value
        by_fmt[f][0] += len(v)
        by_fmt[f][1] += 1
    print("duplicated questions per format:", {f: v[0] for f, v in sorted(by_fmt.items())})

    # ---- 2. candidate universe per horizon -------------------------------------
    print("\n--- hyperedges per (group, horizon) ---")
    for gid in scen:
        rows = []
        for h in sorted({e.horizon_hours for e in hg.hyperedges.values() if e.group_id == gid and e.horizon_hours is not None}):
            rows.append((h, len(graph.horizon_hyperedges(gid, h))))
        print(f"  {gid}: " + "  ".join(f"T-{h}={n}" for h, n in rows))

    # ---- 3. what horizon does retrieval actually look at? ----------------------
    print("\n--- horizon of retrieved hyperedges, Q11 group (same text, 4 blocks) ---")
    q11 = [r for r in records if r.template_id == "Q11"]
    for r, qv in zip(q11, [(qvecs[i]) for i, rr in enumerate(records) if rr.template_id == "Q11"]):
        res = retriever.retrieve(r.question, group_id=r.group_id, query_vector=qv)
        got = [hg.hyperedges[h].horizon_hours for t in res.trajectories for h in t.hyperedge_ids]
        dist = collections.Counter(got)
        own = sum(1 for h in got if h == r.context.horizon_hours)
        print(
            f"  gold={str(r.answer):>5}  block=T-{r.context.horizon_hours:<3}"
            f"  own-block steps={own}/{len(got)}  horizon mix={dict(sorted(dist.items(), key=lambda kv: (kv[0] is None, kv[0])))}"
        )

    # ---- 4. gold-in-context under three candidate universes ---------------------
    #   baseline  : the whole scenario (what the pipeline does today)
    #   scoped    : the question's own horizon block only
    #   scoped_x  : own horizon + the cross-horizon change edges (horizon_hours is None)
    print("\n--- gold evidence and gold string in the retrieved context ---")
    cross: set[str] = {
        hid
        for hid, e in hg.hyperedges.items()
        if e.horizon_hours is None and e.group_id in scen
    }
    rows_out = []
    variants = ("baseline", "scoped", "scoped_x")
    stats = {v: collections.defaultdict(lambda: [0.0, 0]) for v in variants}
    strhit = {v: collections.defaultdict(lambda: [0, 0]) for v in variants}

    for r, qv in zip(records, qvecs):
        fmt = r.answer_format.value
        gold = str(r.answer)
        own = set(graph.horizon_hyperedges(r.group_id, r.context.horizon_hours))
        universes = {
            "baseline": None,
            "scoped": tuple(sorted(own)),
            "scoped_x": tuple(sorted(own | cross)),
        }
        # the gold hyperedges of *this* horizon, per the template->field oracle
        gold_ids = gold_for_record(r, corpus.templates[r.template_id], graph=graph).hyperedge_ids

        row = {
            "qa_id": r.qa_id,
            "template_id": r.template_id,
            "format": fmt,
            "horizon": r.context.horizon_hours,
            "question": r.question,
            "gold": r.answer if isinstance(r.answer, str) else json.dumps(r.answer, ensure_ascii=False),
            "n_own_block": len(own),
            "n_gold_hyperedges": len(gold_ids),
        }
        for name, allowed in universes.items():
            res = retriever.retrieve(
                r.question, group_id=r.group_id, query_vector=qv, allowed_ids=allowed
            )
            ids = [h for t in res.trajectories for h in t.hyperedge_ids]
            text = " ".join(
                f"{hg.hyperedges[h].relation} {hg.hyperedges[h].evidence}" for h in ids
            )
            row[f"{name}_gold_in_context"] = normalise(gold) in normalise(text)
            row[f"{name}_gold_recall"] = (
                len(set(gold_ids) & set(ids)) / len(gold_ids) if gold_ids else None
            )
            strhit[name][fmt][0] += int(row[f"{name}_gold_in_context"])
            strhit[name][fmt][1] += 1
            if row[f"{name}_gold_recall"] is not None:
                stats[name][fmt][0] += row[f"{name}_gold_recall"]
                stats[name][fmt][1] += 1
        rows_out.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"rows": rows_out}, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fmts = sorted({f for v in variants for f, c in stats[v].items() if c[1]})
    print(f"\n{'fmt':5} {'n':>4} " + " ".join(f"{v:>18}" for v in variants))
    for fmt in fmts:
        cells = []
        for v in variants:
            s = stats[v][fmt]
            cells.append(f"{s[0] / s[1]:18.3f}" if s[1] else f"{'--':>18}")
        print(f"{fmt:5} {stats['baseline'][fmt][1]:4d} " + " ".join(cells))
    cells = []
    for v in variants:
        tot = sum(x[0] for x in stats[v].values())
        num = sum(x[1] for x in stats[v].values())
        cells.append(f"{tot / num:18.3f}" if num else f"{'--':>18}")
    print(f"{'ALL':5} {sum(x[1] for x in stats['baseline'].values()):4d} " + " ".join(cells))
    print("  (cells = oracle gold-hyperedge recall)")

    print(f"\n{'fmt':5} {'n':>4} " + " ".join(f"{v:>18}" for v in variants))
    for fmt in sorted(strhit["baseline"]):
        cells = [f"{strhit[v][fmt][0] / strhit[v][fmt][1]:18.3f}" for v in variants]
        print(f"{fmt:5} {strhit['baseline'][fmt][1]:4d} " + " ".join(cells))
    print("  (cells = gold answer string appears in the rendered evidence)")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
