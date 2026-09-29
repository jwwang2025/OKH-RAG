"""Ad-hoc smoke test for the §3.1 Eq. 3 + §A.4 training path, end to end. Not part of the test suite.

Runs the real pipeline over a synthetic 6-group corpus:
    extract (§A.2) -> normalize (§A.2 steps 2-5) -> precedence (§A.3)
      -> embed (§A.4) -> pairs (§A.4) -> train (§3.1 Eq. 4)

Run from the repo root:
    D:/Anaconda/python.exe scripts/smoke_training.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from okh_rag.config import Config, load_config  # noqa: E402
from okh_rag.embeddings import EmbeddingStore, embed_hypergraph, hyperedge_embedding_text  # noqa: E402
from okh_rag.extraction import ScenarioDocument, extract_hypergraph_from_document, split_horizons  # noqa: E402
from okh_rag.llm.base import Usage  # noqa: E402
from okh_rag.ordering.model import PAPER_PARAMETER_COUNT, LearnedTransitionModel  # noqa: E402
from okh_rag.ordering.precedence import build_precedence, canonical_order  # noqa: E402
from okh_rag.schema import KnowledgeHypergraph  # noqa: E402
from okh_rag.ordering import (  # noqa: E402
    SIGNAL_DOCUMENT_ORDER,
    NegativeSampler,
    build_training_pairs,
    split_groups,
    summarise_report,
    train_order_model,
)
from okh_rag.vocabulary import precedence_rank_of  # noqa: E402

HORIZONS = (72, 48, 24)

GROUP_SPECS = [
    ("ARTHUR:port_baltimore", "ARTHUR", "Baltimore, MD"),
    ("BERYL:port_galveston", "BERYL", "Galveston, TX"),
    ("CINDY:port_savannah", "CINDY", "Savannah, GA"),
    ("DEBBY:port_charleston", "DEBBY", "Charleston, SC"),
    ("ERNESTO:port_norfolk", "ERNESTO", "Norfolk, VA"),
    ("FRANCINE:port_mobile", "FRANCINE", "Mobile, AL"),
    # Ten groups rather than a handful so the default 80/10/10 split yields a non-empty validation and
    # test split: with six groups, floor(6 * 0.1) == 0 and there would be nothing to validate on.
    ("GORDON:port_tampa", "GORDON", "Tampa, FL"),
    ("HELENE:port_jacksonville", "HELENE", "Jacksonville, FL"),
    ("ISAAC:port_wilmington", "ISAAC", "Wilmington, NC"),
    ("JOYCE:port_providence", "JOYCE", "Providence, RI"),
]

# (relation, [entity name, entity type] x2, label) — one interaction template.
TEMPLATES = [
    ("has_category_state", ("{storm}", "CYCLONE"), ("Category {cat}", "CYCLONE_STATE"), "cat"),
    ("has_hours_to_landfall", ("{storm}", "CYCLONE"), ("{h} hours to landfall", "TIMING_MARKER"), "h"),
    ("has_warning_status", ("Port of {city}", "PORT"), ("gale warning for {city}", "ADVISORY_STATUS"), "city"),
    (
        "forecasts_hazard_at_horizon",
        ("Port of {city}", "PORT"),
        ("storm surge of {surge} feet", "HAZARD_FORECAST"),
        "surge",
    ),
    (
        "has_operation_status",
        ("Port of {city}", "PORT"),
        ("Condition {cond}", "OPERATION_STATUS"),
        "cond",
    ),
    (
        "has_impact_prediction",
        ("Port of {city}", "PORT"),
        ("berth closures at {city}", "IMPACT_TIMELINE"),
        "city",
    ),
]


def build_document(storm: str, city: str) -> str:
    """A three-horizon scenario document with the §A.2 header format."""
    lines = [f"Scenario {storm} / {city}", ""]
    for offset in HORIZONS:
        lines.append(f"T-{offset} hours before expected landfall:")
        lines.append(
            f"Hurricane {storm} is a Category {int(offset / 24)} storm approaching the Port of {city}."
        )
        lines.append(f"There are {offset} hours to landfall for the Port of {city}.")
        lines.append(f"A gale warning for {city} is in effect for the Port of {city}.")
        lines.append(
            f"A storm surge of {int(offset / 6)} feet is forecast for the Port of {city}."
        )
        lines.append(f"The Port of {city} is operating under Condition YANKEE.")
        lines.append(f"Berth closures at {city} are predicted for the Port of {city}.")
        lines.append("")
    return "\n".join(lines)


def build_facts(storm: str, city: str) -> dict[int, list[dict]]:
    """One canned extraction response per horizon, keyed by lead time."""
    out: dict[int, list[dict]] = {}
    cardinals = {1: "2", 2: "3", 3: "4"}
    for offset in HORIZONS:
        facts: list[dict] = []
        cat = cardinals[int(offset / 24)]
        for relation, (n1, t1), (n2, t2), kind in TEMPLATES:
            entity_a = n1.format(storm=storm, city=city)
            entity_b = n2.format(city=city, h=offset, surge=int(offset / 6), cond="YANKEE", cat=cat)
            evidence = f"{entity_a} and {entity_b} are related in this {relation.replace('_', ' ')}."
            facts.append(
                {
                    "relation": relation,
                    "evidence": evidence,
                    "entities": [
                        {"name": entity_a, "type": t1, "description": kind, "confidence": 0.95},
                        {"name": entity_b, "type": t2, "description": kind, "confidence": 0.9},
                    ],
                    "attributes": [{"key": kind, "value": entity_b}],
                    "confidence": 0.9,
                }
            )
        out[offset] = facts
    return out


class CorpusBackend:
    """Serves the canned facts and produces *learnable* fake embeddings.

    The embeddings are not random: a low-dimensional subspace encodes the relation's §A.3 phase rank and
    the horizon, so the positive pairs §A.4 builds (consecutive in canonical order) carry a real,
    generalisable signal. With pure noise the only way to lower the loss would be memorisation, which
    would make "did training work" untestable.
    """

    def __init__(self) -> None:
        self.usage = Usage()
        self.facts = {gid: build_facts(storm, city) for gid, storm, city in GROUP_SPECS}
        self.calls = 0

    def complete_text(self, messages, *, model, temperature):  # pragma: no cover - unused
        raise NotImplementedError

    def complete_json(self, messages, *, model, temperature, schema, schema_name):
        prompt = messages[-1]["content"]
        self.usage.calls += 1
        self.usage.prompt_tokens += 400
        self.usage.completion_tokens += 250
        self.calls += 1
        for group_id, storm, city in GROUP_SPECS:
            if f"storm {storm}, port {city}" not in prompt:
                continue
            for offset, facts in self.facts[group_id].items():
                if f"T-{offset} forecast snapshot" in prompt:
                    return {"facts": facts}
        return {"facts": []}

    @property
    def embedding_dim(self) -> int:
        return 1536

    def embed(self, texts):
        rows = []
        for text in texts:
            relation = text.split("relation: ", 1)[1].split("\n", 1)[0]
            rank = precedence_rank_of(relation)
            horizon = 0.0
            marker = "horizon:T-"
            if marker in text:
                tail = text.split(marker, 1)[1]
                digits = ""
                for ch in tail:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                horizon = float(digits or 0)
            vector = np.zeros(1536, dtype=np.float32)
            # A learnable directed structure: phase rank and lead time.
            vector[0] = rank
            vector[1] = rank * rank
            vector[2] = horizon
            vector[3] = rank * horizon / 100.0
            # Plus fixed per-string noise, so no two hyperedges are identical.
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            vector[4:36] = np.random.default_rng(seed).normal(0, 0.05, 32).astype(np.float32)
            rows.append(vector)
        return np.stack(rows)


def build_corpus(backend: CorpusBackend, cfg: Config) -> KnowledgeHypergraph:
    corpus = KnowledgeHypergraph()
    for group_id, storm, city in GROUP_SPECS:
        document = ScenarioDocument(
            group_id=group_id,
            storm=storm,
            port=city,
            year=2026,
            blocks=tuple(split_horizons(build_document(storm, city), group_id=group_id,
                                        cfg=cfg.hypergraph.horizon)),
        )
        result = extract_hypergraph_from_document(document, backend=backend, cfg=cfg)
        corpus.merge(result.hypergraph)
    return corpus


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="okh_train_"))
    cfg = load_config("configs/base.yaml")
    cfg.paths = cfg.paths.resolve(root)
    cfg.llm = cfg.llm.resolve(root)
    rng = random.Random(cfg.seed)

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            failures.append(name)

    print("== corpus ==")
    backend = CorpusBackend()
    corpus = build_corpus(backend, cfg)
    print(f"  groups={len(corpus.groups)} hyperedges={len(corpus)} entities={len(corpus.entities)} "
          f"extraction calls={backend.calls}")
    check("every group spec produced a group", len(corpus.groups) == len(GROUP_SPECS))
    check("every group has hyperedges", all(corpus.groups[g] for g in corpus.groups))

    print("\n== §A.3 ordering ==")
    graph = build_precedence(corpus, cfg.precedence)
    ordering = canonical_order(corpus, graph)
    check("every group has a canonical order", len(ordering) == len(corpus.groups))
    check("orders are permutations",
          all(len(set(ordering[g])) == len(ordering[g]) for g in ordering))

    print("\n== §A.4 embeddings ==")
    store = EmbeddingStore(cfg.llm.cache_dir / "embeddings")
    embeddings = embed_hypergraph(corpus, backend=backend, cfg=cfg, store=store)
    print(f"  matrix {embeddings.matrix.shape} dim={embeddings.dim}")
    print("  sample embedding text:\n    " + hyperedge_embedding_text(
        corpus.hyperedges[embeddings.hyperedge_ids[0]], corpus).replace("\n", "\n    "))
    check("one row per hyperedge", len(embeddings) == len(corpus))
    check("dim matches §A.4", embeddings.dim == 1536)

    hits_before = store.hits
    embed_hypergraph(corpus, backend=backend, cfg=cfg, store=store)
    check("store caches every vector on a second pass",
          store.hits - hits_before == len(corpus), f"hits={store.hits - hits_before}")

    print("\n== §A.4 pairs ==")
    pairs = build_training_pairs(corpus, ordering, cfg.training)
    print(f"  |P|={len(pairs)} by signal={pairs.by_signal()} missing_order={len(pairs.groups_missing_order)}")
    check("positives built", len(pairs) > 0)
    check("no group lacks an order", not pairs.groups_missing_order)
    check("every positive is canonically ordered",
          all(ordering[_group_of(corpus, p.source_id)].index(p.source_id)
              < ordering[_group_of(corpus, p.source_id)].index(p.target_id)
              for p in pairs.positives))

    print("\n== §A.4 negative sampler ==")
    sampler = NegativeSampler(corpus, k=64, rng=random.Random(1))
    sample_sources = [p.source_id for p in pairs.positives[:200]]
    sampled = sampler.sample(sample_sources)
    group_of = {h: g for g, ids in corpus.groups.items() for h in ids}
    cross = sum(
        1
        for src, row in zip(sample_sources, sampled)
        for neg in row
        if group_of[neg] != group_of[src]
    )
    total_neg = sum(len(r) for r in sampled)
    print(f"  {json.dumps(sampler.stats())}")
    check("K = 64 negatives per example", all(len(r) == 64 for r in sampled))
    check("negatives are cross-group (§A.4)", cross == total_neg, f"{cross}/{total_neg}")
    check("no negative equals its source", all(src not in r for src, r in zip(sample_sources, sampled)))

    print("\n== model ==")
    model = LearnedTransitionModel(cfg.order_model)
    print(f"  {json.dumps(model.describe())}")
    check("§A.4 parameter count reproduced", model.num_parameters == PAPER_PARAMETER_COUNT)
    check("matches_paper_parameter_count flag", model.matches_paper_parameter_count)

    with torch.no_grad():
        probe = torch.randn(4, 1536)
        init_scores = model.score_matrix(probe, probe)
    check("V=0 init makes P_theta uniform",
          bool(torch.allclose(init_scores, init_scores.new_zeros(1).expand_as(init_scores))))

    # Asymmetry is a property of the bilinear form, not of the initialisation: with V = 0 every score is
    # trivially zero and therefore symmetric. Probe a separately initialised model so the check cannot
    # pass vacuously.
    async_cfg = type(cfg.order_model)(
        embedding_dim=cfg.order_model.embedding_dim,
        rank=cfg.order_model.rank,
        l2_normalize_embeddings=cfg.order_model.l2_normalize_embeddings,
        init="xavier",
    )
    probe_model = LearnedTransitionModel(async_cfg)
    with torch.no_grad():
        asymmetric = probe_model.score_matrix(probe, probe)
    check("bilinear form is asymmetric (§3.1)",
          not bool(torch.allclose(asymmetric, asymmetric.t(), atol=1e-5)),
          f"max |S - S^T| = {float((asymmetric - asymmetric.t()).abs().max()):.4f}")

    print("\n== §A.4 split ==")
    split = split_groups(tuple(corpus.groups), cfg.training, rng=random.Random(cfg.seed))
    print(f"  { {k: len(v) for k, v in split.items()} }")
    all_groups = set(corpus.groups)
    check("split partitions the corpus",
          set().union(*[set(v) for v in split.values()]) == all_groups
          and sum(len(v) for v in split.values()) == len(all_groups))
    check("no group appears in two splits",
          sum(len(set(v)) for v in split.values()) == sum(len(v) for v in split.values()))

    print("\n== §3.1 Eq. 4 training (default: negative_term_sign='minus') ==")
    outcome = train_order_model(corpus, ordering, embeddings, cfg=cfg)
    print(summarise_report(outcome.report))
    print(f"  train losses: {[round(x, 4) for x in outcome.report.train_losses]}")
    check("training ran at least one epoch", outcome.report.epochs_run >= 1)
    check("positives were used", outcome.report.positives > 0)
    check("E_P[log P_theta] improved over the run",
          outcome.report.best_mean_positive_logprob > -np.log(1 + cfg.training.sampled_softmax_negatives),
          f"{outcome.report.best_mean_positive_logprob:.4f} "
          f"(uniform baseline {-np.log(1 + cfg.training.sampled_softmax_negatives):.4f})")
    check("loss decreased from first to last epoch",
          outcome.report.train_losses[-1] < outcome.report.train_losses[0],
          f"{outcome.report.train_losses[0]:.4f} -> {outcome.report.train_losses[-1]:.4f}")

    print("\n== checkpoint round-trip ==")
    checkpoint = model.save(root / "order_model.pt")
    reloaded = LearnedTransitionModel.load(checkpoint)
    with torch.no_grad():
        same = torch.allclose(model.score_matrix(probe, probe), reloaded.score_matrix(probe, probe))
    check("save/load reproduces the scores", same)

    print("\n== Eq. 4 sign contradiction, measured ==")
    print("  Training two identical runs, differing only in training.negative_term_sign.")
    results: dict[str, dict[str, float]] = {}
    for sign in ("minus", "literal"):
        variant = load_config("configs/base.yaml")
        variant.paths = variant.paths.resolve(root)
        variant.llm = variant.llm.resolve(root)
        variant.training.negative_term_sign = sign
        variant.training.epochs = 12
        variant.training.early_stop_patience = 0
        run = train_order_model(corpus, ordering, embeddings, cfg=variant, rng=random.Random(cfg.seed))
        last = run.history[-1] if run.history else None
        final_pos_logprob = _final_mean(outcome=run, key="mean_positive_logprob")
        results[sign] = {
            "first_loss": run.report.train_losses[0],
            "final_loss": run.report.train_losses[-1],
            "final_pos_logprob": final_pos_logprob,
            "final_negative_term": last.negative_term if last else float("nan"),
        }
        print(f"  {sign:8} loss {run.report.train_losses[0]:9.4f} -> {run.report.train_losses[-1]:10.4f}   "
              f"E_P[log P] {final_pos_logprob:9.4f}   E_N[log P] "
              f"{(last.negative_term / variant.training.alpha) if last else float('nan'):9.4f}")

    uniform = -np.log(1 + cfg.training.sampled_softmax_negatives)
    print(f"  (uniform-softmax reference: E_P[log P] = {uniform:.4f}; a healthy run must stay above it)")
    check("'minus' keeps E_P[log P_theta] near the uniform reference and improves on it",
          results["minus"]["final_pos_logprob"] > uniform,
          f"{results['minus']['final_pos_logprob']:.4f} > {uniform:.4f}")
    check("'literal' collapses E_P[log P_theta] far below the uniform reference",
          results["literal"]["final_pos_logprob"] < uniform - 1.0,
          f"{results['literal']['final_pos_logprob']:.4f} vs uniform {uniform:.4f}")
    check("'literal' drives the loss far below zero: the objective is unbounded below",
          results["literal"]["final_loss"] < 0.0 < results["minus"]["final_loss"],
          f"literal={results['literal']['final_loss']:.4f} "
          f"minus={results['minus']['final_loss']:.4f}")
    print(
        "  => Eq. 4 as printed is not merely mis-signed: minimising it is unbounded below, because\n"
        "     shrinking E_N[log P_theta] is rewarded without limit, and in practice the negative term\n"
        "     swamps the positive one and destroys E_P[log P_theta]. This is why the default is 'minus'."
    )

    print(f"\n{'ALL CHECKS PASSED' if not failures else 'FAILURES: ' + ', '.join(failures)}")
    return 1 if failures else 0


def _group_of(corpus: KnowledgeHypergraph, hyperedge_id: str) -> str:
    for group_id, ids in corpus.groups.items():
        if hyperedge_id in ids:
            return group_id
    raise KeyError(hyperedge_id)


def _final_mean(outcome, *, key: str) -> float:
    """Mean of one Eq. 4 breakdown field over the last epoch's batches."""
    if not outcome.history:
        return float("nan")
    per_epoch = max(1, len(outcome.history) // max(1, len(outcome.report.train_losses)))
    tail = outcome.history[-per_epoch:]
    return float(np.mean([getattr(b, key) for b in tail]))


if __name__ == "__main__":
    raise SystemExit(main())
