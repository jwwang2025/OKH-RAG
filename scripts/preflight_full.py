"""Preflight for a §4.1-scale run — what does the *whole* release cost to build?

§4.1 evaluates on all of CyPortQA: 117,178 questions over 472 scenario keys. Before committing to
a generation run at that scale, this script measures the two things that decide whether it is
possible at all: how big the reference hypergraph is, and how long its hyperedges take to embed.

    python scripts/preflight_full.py                 # build, report, stop
    python scripts/preflight_full.py --embed-sample 300   # also time a real embedding batch
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from okh_rag.config import load_config  # noqa: E402
from okh_rag.data.cyportqa import load_corpus  # noqa: E402
from okh_rag.embeddings import EmbeddingStore, embed_texts, hyperedge_embedding_text  # noqa: E402
from okh_rag.evaluation.reference_graph import build_reference_graph  # noqa: E402
from okh_rag.llm import OpenAIBackend  # noqa: E402
from run_generation import select_records  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    parser.add_argument("--scenario-limit", type=int, default=0, help="0 = all 472")
    parser.add_argument("--retrieve", type=int, default=0, help="0 = all 117,178")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed-sample", type=int, default=0, help="embed this many hyperedges to time it")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    t0 = time.time()
    corpus = load_corpus(ROOT / "data" / "raw" / "cyportqa")
    print(f"corpus loaded in {time.time() - t0:.1f}s")

    records = select_records(
        corpus,
        scenario_limit=args.scenario_limit or None,
        question_limit=args.retrieve or None,
        seed=args.seed,
    )
    scenarios = sorted({r.scenario_id for r in records})
    fmt = collections.Counter(r.answer_format.value for r in records)
    horizon = collections.Counter(r.context.horizon_hours for r in records)
    print(f"\nSAMPLE (§4.1 = 117,178 questions / 472 scenarios)")
    print(f"  questions : {len(records)}")
    print(f"  scenarios : {len(scenarios)}")
    print(f"  horizons  : {dict(sorted(horizon.items()))}")
    print(f"  formats   : {dict(sorted(fmt.items()))}")
    print(f"     paper  : {{'DE': 24094, 'MC': 37813, 'NU': 49459, 'TF': 5812}}")
    print(f"  templates : {len({r.template_id for r in records})}")
    print(f"  storms    : {len({r.context.storm for r in records})}")

    t1 = time.time()
    graph = build_reference_graph(corpus, cfg=cfg, scenario_ids=scenarios)
    hg = graph.hypergraph
    print(f"\nREFERENCE GRAPH built in {time.time() - t1:.1f}s")
    print(f"  hyperedges : {len(hg.hyperedges)}")
    print(f"  entities   : {len(hg.entities)}")
    print(f"  groups     : {len(hg.groups)}")
    print(f"  matrix     : {len(hg.hyperedges) * 1536 * 4 / 1e9:.2f} GB float32")

    texts = [hyperedge_embedding_text(hg.hyperedges[h], hg) for h in sorted(hg.hyperedges)]
    unique = len(set(texts))
    print(f"  embed texts: {len(texts)} ({unique} unique, {100 * unique / max(len(texts), 1):.1f}%)")

    store = EmbeddingStore(ROOT / "artifacts/llm_cache/embeddings", dim=1536)
    cached = sum(1 for t in texts if store.get(t) is not None)
    print(f"  cached     : {cached}/{len(texts)}")

    if args.embed_sample > 0:
        backend = OpenAIBackend(cfg.llm)
        batch = texts[: args.embed_sample]
        t2 = time.time()
        matrix = embed_texts(batch, backend=backend, cfg=cfg, store=store)
        dt = time.time() - t2
        print(f"\nEMBED TIMING on {len(batch)} texts: {dt:.1f}s")
        remaining = max(len(texts) - cached, 0)
        print(f"  projected for {remaining} uncached: {dt * remaining / len(batch) / 60:.1f} min")
        print(f"  matrix shape: {matrix.shape}")

    # per-query retrieval cost on the real graph needs embeddings, so only print the estimate
    print(f"\nGENERATION (measured earlier at 2.5 s/question on gpt-4o via the relay)")
    print(f"  projected for {len(records)} questions: {2.5 * len(records) / 3600:.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
