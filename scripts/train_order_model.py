"""§3.1 Eq. 3-4 / §A.4 — train the learned transition model ``P_theta`` on the CyPortQA release.

``scripts/smoke_training.py`` exercises the whole training path on a ten-group synthetic corpus;
this driver is its full-corpus counterpart. It runs the stages the paper specifies on the **real**
reference hypergraph, which :mod:`okh_rag.evaluation.reference_graph` builds directly from
``Encoded_senario.json`` — so no §A.2 extraction LLM is needed, only §A.4's embedder:

    reference hypergraph (one hyperedge per populated field + synthesised family-13 edges)
      -> §A.3 precedence and canonical order (the document-order supervision P_doc)
      -> §A.4 embeddings with text-embedding-3-small (content-addressed cache on disk)
      -> §A.4 ordered pairs: document order P_doc + entity-overlap consistency P_ent
      -> §3.1 Eq. 4 contrastive training, sampled softmax K = 64, AdamW
      -> artifacts/order_model.pt  (the checkpoint scripts/run_generation.py loads by default)

Two honest scope notes:

* **P_ret (retrieval-induced preference) is not run here.** §A.4's third signal needs retrieval
  traces "from correct answers", i.e. a generation/eval pass first; ``train_order_model`` accepts
  those pairs as ``extra_positives`` but never calls a retriever itself. This driver trains the
  P_doc + P_ent signals (config ``training.signals``), exactly as ``smoke_training.py`` does.
* **Full release scale is large.** All 472 groups give 248,652 hyperedges — a (248652, 1536)
  float32 matrix is ~1.5 GB in memory plus the one-off text-embedding-3-small bill. Start with
  ``--scenario-limit 48``; embeddings are cached, so a later full run only pays for the new edges.

Needs ``OPENAI_API_KEY`` (and ``OPENAI_BASE_URL`` when going through an OpenAI-compatible relay;
the SDK reads both from the environment).

Run from the repo root::

    # 48-group pilot — cheap, validates the end-to-end learned path
    python scripts/train_order_model.py --scenario-limit 48

    # §4 scale: all 472 groups, then §4.1's table with the learned variant
    python scripts/train_order_model.py --scenario-limit 0
    python scripts/run_generation.py --backend openai
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from okh_rag.config import Config, load_config  # noqa: E402
from okh_rag.data.cyportqa import load_corpus  # noqa: E402
from okh_rag.embeddings import EmbeddingStore, embed_hypergraph  # noqa: E402
from okh_rag.evaluation.reference_graph import build_reference_graph  # noqa: E402
from okh_rag.llm.openai_client import OpenAIBackend  # noqa: E402
from okh_rag.ordering.precedence import build_precedence, canonical_order  # noqa: E402
from okh_rag.ordering.train import summarise_report, train_order_model  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPO_ROOT / "artifacts" / "order_model.pt"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--data", default=str(REPO_ROOT / "data" / "raw" / "cyportqa"))
    parser.add_argument("--out", default=str(DEFAULT_CHECKPOINT), help="checkpoint path (.pt)")
    parser.add_argument(
        "--scenario-limit",
        type=int,
        default=48,
        help="knowledge groups to build and train on; 0 = all 472 (§4 scale)",
    )
    parser.add_argument("--seed", type=int, default=None, help="defaults to config seed")
    return parser.parse_args(argv)


def select_scenarios(corpus, limit: int | None, seed: int) -> tuple[str, ...]:
    """Pick groups deterministically.

    Mirrors ``scripts/run_retrieval_oracle.py``'s selection — sorted ids, and when a subset is
    requested a seeded sample rather than the first N — so a 48-group training graph and the
    oracle's 48-group retrieval graph cover the same knowledge groups for the same seed.
    """
    ids = sorted(corpus.encoded)
    if limit is None or limit >= len(ids):
        return tuple(ids)
    return tuple(sorted(random.Random(seed).sample(ids, limit)))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    seed = cfg.seed if args.seed is None else args.seed
    data_root = Path(args.data)
    out_path = Path(args.out)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. §A.4 fixes text-embedding-3-small as the embedder, so this "
            "driver needs the OpenAI API (set OPENAI_BASE_URL too when using a relay)."
        )

    random.seed(seed)

    print("== loading the release ==")
    started = time.time()
    corpus = load_corpus(data_root)
    print(
        f"  {len(corpus.encoded)} scenarios, "
        f"{sum(len(v) for v in corpus.qa.values())} questions ({time.time() - started:.1f}s)"
    )

    scenario_ids = select_scenarios(corpus, args.scenario_limit or None, seed)
    print(f"== building the reference hypergraph over {len(scenario_ids)} groups ==")
    started = time.time()
    graph = build_reference_graph(corpus, cfg=cfg, scenario_ids=scenario_ids)
    hypergraph = graph.hypergraph
    print(
        f"  {len(hypergraph)} hyperedges, {len(hypergraph.entities)} entities "
        f"({time.time() - started:.1f}s)"
    )

    print("== §A.3 precedence and canonical order ==")
    precedence = build_precedence(hypergraph, cfg.precedence)
    ordering = canonical_order(hypergraph, precedence)
    ordered_groups = {g for g, ids in ordering.items() if ids}
    print(f"  canonical order for {len(ordered_groups)}/{len(hypergraph.groups)} groups")

    print("== §A.4 embeddings (text-embedding-3-small, cached on disk) ==")
    started = time.time()
    backend = OpenAIBackend(cfg.llm)
    store = EmbeddingStore(cfg.llm.cache_dir / "embeddings")
    embeddings = embed_hypergraph(hypergraph, backend=backend, cfg=cfg, store=store)
    store_stats = store.stats()
    print(
        f"  matrix {embeddings.matrix.shape}, store {store_stats['entries']} vectors "
        f"({store_stats['hits']} cached hits, {time.time() - started:.1f}s)"
    )

    print("== §3.1 Eq. 4 contrastive training ==")
    started = time.time()
    outcome = train_order_model(hypergraph, ordering, embeddings, cfg=cfg)
    print(summarise_report(outcome.report))
    print(f"  wall time: {time.time() - started:.1f}s")

    checkpoint = outcome.model.save(out_path)
    print(f"  saved checkpoint: {checkpoint}")

    manifest_path = out_path.with_suffix(".train_report.json")
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": seed,
        "data": str(data_root),
        "n_scenarios": len(scenario_ids),
        "n_hyperedges": len(hypergraph),
        "n_entities": len(hypergraph.entities),
        "embedding_model": cfg.llm.embedding_model,
        "embedding_store": store_stats,
        "checkpoint": str(checkpoint),
        "training": outcome.report.as_dict(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote manifest: {manifest_path}")
    print(
        "\nNext: python scripts/run_generation.py --backend openai "
        f"--order-model {checkpoint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
