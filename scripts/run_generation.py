"""§4.1 — run the four methods over CyPortQA questions with one generator and score them.

This is the driver for :mod:`okh_rag.generation`. It answers one question end to end — retrieve, render,
generate, score — for each of §4.1's four methods, and writes the accuracy table plus the counts that make
it interpretable.

What it does **not** do is produce §4.1's numbers. Those need GPT-4o (§4.1: "All methods use the same
generator (GPT-4o) and the same embedding model (text-embedding-3-small)"), and this script's default
``--backend lexical`` substitutes a deterministic token-overlap embedder plus a stub generator so that the
whole pipeline can be exercised, and the wiring checked, without an API key. The stub generator does not
read the evidence — see :class:`StubGeneratorBackend` — so its accuracy measures the label distribution,
not the method. Every report says which backend produced it.

The five things this driver exists to keep honest:

1. **One generator, four retrievers.** The generator is constructed once and passed to every method; the
   only per-method difference is the :class:`~okh_rag.generation.evidence.EvidenceContext` it receives.
2. **One question set.** Every method is scored on the same questions, and the count of questions actually
   scored is reported per method. §4.1's table can only be read if the four columns share a denominator.
3. **The exclusions are visible.** Unanswerable questions, non-numeric SA golds, contract violations and
   dropped per-trajectory answers all leave the denominator, and all four counts appear in the report.
4. **The shuffle row is a *condition*, not a flag.** ``--shuffle`` permutes the retrieved trajectory before
   rendering and says so in the prompt, reproducing §4.5 Table 2's stress test, where "performance drops
   from 0.534 to 0.487 [...] despite identical retrieved content".
5. **§A.8's two modes are both runnable**, so the paper's stated preference for single-call can be checked
   rather than trusted.

Usage::

    # wiring check, no API key, no real accuracy
    python scripts/run_generation.py --sources okh_rag hypergraph_rag --retrieve 40

    # §4.1 as written (needs OPENAI_API_KEY)
    python scripts/run_generation.py --backend openai --mode single_call

    # §4.5 Table 2's Shuffle row
    python scripts/run_generation.py --backend openai --shuffle
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from okh_rag.config import Config, load_config  # noqa: E402
from okh_rag.data.cyportqa import QARecord, load_corpus  # noqa: E402
from okh_rag.embeddings import (  # noqa: E402
    HyperedgeEmbeddings,
    embed_texts,
    hyperedge_embedding_text,
)
from okh_rag.evaluation.answers import AccuracyReport, score_record  # noqa: E402
from okh_rag.evaluation.reference_graph import build_reference_graph  # noqa: E402
from okh_rag.generation.baselines import (  # noqa: E402
    BASELINE_SOURCES,
    ChunkIndex,
    GraphRagRetriever,
    HypergraphRagRetriever,
    TextRagRetriever,
    load_raw_text_chunks,
)
from okh_rag.generation.evidence import context_from_retrieval  # noqa: E402
from okh_rag.generation.generator import Generator  # noqa: E402
from okh_rag.precedence import build_precedence  # noqa: E402
from okh_rag.retrieval import OrderAwareRetriever  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_SOURCES = ("okh_rag",) + BASELINE_SOURCES
DEFAULT_SOURCES = ("okh_rag", "text_rag", "graph_rag", "hypergraph_rag")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class LexicalEmbedder:
    """A deterministic bag-of-tokens embedder. **Not** §A.4's model.

    Reused in spirit from ``scripts/run_retrieval_oracle.py``: a hashed token count, L2-normalised, signed
    by a second hash so that unrelated tokens can repel as well as attract. It preserves *lexical* overlap
    and nothing else, which is enough to make retrieval return non-degenerate evidence for a wiring check
    and not enough to say anything about accuracy.

    Kept independent of the existing copy rather than imported from it because a script is not a library:
    ``run_retrieval_oracle.py`` is run as ``__main__``, so importing from it would execute its module body
    or require a path hack, and either would couple two reports' reproducibility to each other.
    """

    def __init__(self, *, dim: int = 1536) -> None:
        self._dim = dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in str(text).lower().split():
                bucket = hash_token(token, self._dim)
                sign = 1.0 if hash_token(f"sign:{token}", 2) == 0 else -1.0
                out[row, bucket] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


def hash_token(token: str, modulus: int) -> int:
    """A stable bucket for a token, independent of ``PYTHONHASHSEED``.

    ``hash()`` is salted per process, so using it would make two runs of the same script disagree about
    the same corpus — which is the one property a validation run needs.
    """
    import hashlib

    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big") % modulus


class StubGeneratorBackend:
    """A generator that returns a valid §A.8 envelope without reading the evidence.

    It answers from the *shape* of the question only — the first admissible option for MC, ``Yes`` for TF, a
    fixed number for SA, and an empty-string description for TD — so that:

    * the pipeline can be exercised end to end with no API key,
    * every response conforms to the contract, which is what makes it usable as a wiring check,
    * and its accuracy is **meaningless as an accuracy**. It is a label prior, and the report says so.

    Returning a conforming-but-uninformed answer is the point: a stub that answered correctly would make a
    broken pipeline look healthy, and a stub that answered invalidly would make the violations count
    unreadable because every call would violate.
    """

    def __init__(self) -> None:
        self.calls = 0

    def complete_text(self, messages, *, model, temperature):  # pragma: no cover - unused
        raise AssertionError("generation uses complete_json")

    def complete_json(self, messages, *, model, temperature, schema, schema_name):
        self.calls += 1
        field_schema = schema["properties"]["answer"]
        if "enum" in field_schema:
            answer: Any = field_schema["enum"][0]
        elif field_schema.get("type") == "number":
            answer = 0
        else:
            # A non-empty string is required for a description: §A.8's contract rejects an empty one
            # (``EMPTY_TEXT``), and a stub that violated the contract on every TD question would make the
            # violations counter unreadable — every run would report its eight worst questions as broken
            # rather than as unanswered.
            answer = "stub answer"
        payload = {"answer": answer, "confidence": 0.5, "rationale": "stub backend: no evidence read"}
        if "supporting_trajectory" in schema["required"]:
            payload["supporting_trajectory"] = "1"
        return payload


def build_backend(name: str, cfg: Config) -> tuple[Any, Any]:
    """Construct the embedding and generator backends named by ``name``.

    Returns:
        ``(embedder, generator_backend)``. ``openai`` uses §4.1's models and requires ``OPENAI_API_KEY``;
        ``lexical`` uses the offline pair above.
    """
    if name == "openai":
        from okh_rag.llm import OpenAIBackend

        backend = OpenAIBackend(cfg.llm)
        return backend, backend
    if name == "lexical":
        return LexicalEmbedder(dim=cfg.order_model.embedding_dim), StubGeneratorBackend()
    raise SystemExit(f"unknown backend {name!r}; expected 'lexical' or 'openai'")


#: Where the training stage is expected to leave its best checkpoint. Named here rather than in the config
#: because it is a driver convention, not a paper parameter.
DEFAULT_ORDER_MODEL = REPO_ROOT / "artifacts" / "order_model.pt"


def load_order_model(cfg: Config, path: str | None) -> object | None:
    """Load the §3.1 ``P_θ`` when the configured variant needs it.

    Retrieval with ``transition_variant='learned'`` and no model raises inside
    :class:`~okh_rag.retrieval.OrderAwareRetriever` (deliberately — see its docstring), which for a user of
    this script would surface as an exception after the corpus and embeddings have been paid for. Checking
    here turns it into one actionable line before anything is loaded.

    Returns:
        The model, or ``None`` for the heuristic and order-free variants, which do not use one.
    """
    variant = cfg.ablation.transition_variant
    if variant != "learned":
        print(f"Order model: not used (transition_variant={variant!r})")
        return None

    candidate = Path(path) if path else DEFAULT_ORDER_MODEL
    if not candidate.is_file():
        raise SystemExit(
            f"transition_variant='learned' needs a trained P_theta, and no checkpoint exists at {candidate}.\n"
            "Either train one first (see src/okh_rag/training/train.py), or pass "
            "--transition-variant heuristic for an offline wiring check."
        )
    from okh_rag.order_model import LearnedTransitionModel

    model = LearnedTransitionModel.load(candidate, cfg=cfg.order_model)
    print(f"Order model: loaded {candidate}")
    return model


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


@dataclass
class RunPlan:
    """What one run will do, printed before it starts so the report is self-describing."""

    backend: str
    mode: str
    sources: tuple[str, ...]
    shuffle: bool
    scenario_limit: int | None
    question_limit: int | None
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "answer_mode": self.mode,
            "sources": list(self.sources),
            "shuffle_trajectory": self.shuffle,
            "scenario_limit": self.scenario_limit,
            "question_limit": self.question_limit,
            "seed": self.seed,
        }


def select_records(corpus, *, scenario_limit: int | None, question_limit: int | None) -> list[QARecord]:
    """A deterministic question sample: scenarios in sorted order, questions in release order.

    Sorted rather than taken in dict order, because dict order is insertion order derived from a JSON object
    and is not a documented contract — a run whose sample changed because the loader's iteration changed
    would be unreproducible for a reason that has nothing to do with the experiment.
    """
    scenario_ids = sorted(corpus.qa)
    if scenario_limit is not None:
        scenario_ids = scenario_ids[:scenario_limit]
    records: list[QARecord] = []
    for scenario_id in scenario_ids:
        records.extend(corpus.qa[scenario_id])
    if question_limit is not None:
        records = records[:question_limit]
    return records


def build_embeddings(graph, cfg: Config, embedder) -> HyperedgeEmbeddings:
    """Embed every hyperedge of the reference graph, in sorted id order so the matrix is reproducible."""
    ordered_ids = tuple(sorted(graph.hypergraph.hyperedges))
    texts = [
        hyperedge_embedding_text(graph.hypergraph.hyperedges[hid], graph.hypergraph) for hid in ordered_ids
    ]
    matrix = embed_texts(texts, backend=embedder, cfg=cfg, store=None)
    return HyperedgeEmbeddings(hyperedge_ids=ordered_ids, matrix=matrix)


def build_retrievers(
    graph, embeddings, cfg: Config, embedder, *, sources, order_model: object | None = None
) -> dict[str, Any]:
    """Construct each requested retriever, sharing the corpus embeddings.

    §4.1 fixes "the same embedding model" across methods, so the hypergraph-side retrievers all receive the
    one :class:`HyperedgeEmbeddings`. Text-RAG needs its own vectors because its corpus is different — the
    NOAA text tree rather than the hypergraph — but the same embedder produces them.
    """
    retrievers: dict[str, Any] = {}

    if "okh_rag" in sources:
        retrievers["okh_rag"] = OrderAwareRetriever(
            cfg=cfg,
            hypergraph=graph.hypergraph,
            precedence=build_precedence(graph.hypergraph, cfg.precedence),
            embeddings=embeddings,
            order_model=order_model,
            backend=None,
        )
    if "hypergraph_rag" in sources:
        retrievers["hypergraph_rag"] = HypergraphRagRetriever(
            graph.hypergraph, embeddings, cfg.retrieval, cfg.baselines
        )
    if "graph_rag" in sources:
        retrievers["graph_rag"] = GraphRagRetriever(
            graph.hypergraph, embeddings, cfg.retrieval, cfg.baselines
        )
    if "text_rag" in sources:
        chunks = load_raw_text_chunks(REPO_ROOT / "data" / "raw" / "cyportqa")
        print(f"  text_rag corpus: {len(chunks)} chunks")
        if chunks:
            matrix = embed_texts(
                [chunk.text for chunk in chunks], backend=embedder, cfg=cfg, store=None
            )
            retrievers["text_rag"] = TextRagRetriever(ChunkIndex(chunks, matrix), cfg.baselines)
        else:
            print("  text_rag: no NOAA text tree vendored; the method will be skipped")
    return retrievers


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass
class SourceResult:
    """One method's outcome across the sampled questions."""

    source: str
    report: AccuracyReport = field(default_factory=AccuracyReport)
    n_empty_context: int = 0
    n_nonconforming: int = 0
    n_questions: int = 0
    notes: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def as_dict(self, *, overall_mode: str) -> dict[str, Any]:
        payload = self.report.as_dict(overall_mode=overall_mode)
        payload.update(
            {
                "source": self.source,
                "n_questions": self.n_questions,
                "n_empty_context": self.n_empty_context,
                "n_nonconforming_answers": self.n_nonconforming,
                "elapsed_s": round(self.elapsed_s, 2),
            }
        )
        return payload


def run_source(
    *,
    source: str,
    retriever: Any,
    records: Sequence[QARecord],
    query_vectors: Mapping[str, np.ndarray],
    graph,
    generator: Generator,
    cfg: Config,
    plan: RunPlan,
) -> SourceResult:
    """Retrieve, render, generate and score every question for one method.

    The retrieval path branches once, at the top: OKH-RAG's context comes from a
    :class:`~okh_rag.retrieval.RetrievalResult` (which carries the trajectories, the Eq. 5 term breakdown
    and the shuffle flag) while the baselines build their own. Everything after that branch — the prompt, the
    generator, the parser, the scorer — is shared, which is the control §4.1's comparison rests on.
    """
    result = SourceResult(source=source)
    started = time.time()

    for record in records:
        query_vector = query_vectors[record.qa_id]
        if source == "okh_rag":
            retrieval = retriever.retrieve(
                record.question,
                group_id=record.group_id,
                query_vector=query_vector,
            )
            context = context_from_retrieval(
                retrieval,
                graph.hypergraph,
                question=record.question,
                include_quality_summary=cfg.generation.include_quality_summary,
                include_reasoning_tags=cfg.generation.include_reasoning_tags,
            )
        else:
            context = retriever.context(record, query_vector)

        if context.is_empty:
            result.n_empty_context += 1

        generated = generator.answer(record, context, mode=plan.mode)
        if not generated.conforming:
            result.n_nonconforming += 1
        score = score_record(record, generated.answer.answer, cfg.evaluation)
        result.report.add(score, record.answer_format)
        result.n_questions += 1

    result.elapsed_s = time.time() - started
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    """A percentage, or an em dash when there is no denominator. Never ``0.0%`` for "none"."""
    return "—" if value is None else f"{100 * value:.1f}%"


def render_markdown(results: Mapping[str, SourceResult], plan: RunPlan, cfg: Config) -> str:
    """§4.1's accuracy table plus the counts that make it readable.

    The report leads with what the run *is* — backend, mode, sample — because the first question a reader
    will have about a generation experiment is whether its numbers are real, and for this driver the honest
    answer is "only if the backend was ``openai``".
    """
    lines: list[str] = []
    lines.append("# §4.1 generation evaluation")
    lines.append("")
    if plan.backend != "openai":
        lines.append(
            "> **These are not §4.1's numbers.** The backend is `lexical`: a deterministic token-overlap "
            "embedder and a stub generator that never reads the evidence. The run checks that the pipeline "
            "is wired — four methods, one generator, one question set, one scorer — and nothing about "
            "accuracy. Re-run with `--backend openai` to reproduce §4.1."
        )
        lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    for key, value in plan.as_dict().items():
        # A list renders as a Python ``repr`` in a markdown cell, which reads like a debugging leftover.
        rendered = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        lines.append(f"| `{key}` | {rendered} |")
    lines.append(f"| `generator_model` | `{cfg.llm.generator_model}` |")
    lines.append(f"| `td_scoring` | `{cfg.evaluation.td_scoring}` |")
    lines.append(f"| `exclude_unanswerable` | {cfg.evaluation.exclude_unanswerable} |")
    lines.append("")

    lines.append("## Accuracy by answer type")
    lines.append("")
    header = ["method", "TF", "MC", "SA (NU)", "TD (DE)", "overall (micro)", "overall (macro)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for source, result in results.items():
        by = result.report.by_format()
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                source,
                _pct(by["TF"]),
                _pct(by["MC"]),
                _pct(by["NU"]),
                _pct(by["DE"]),
                _pct(result.report.overall(cfg.evaluation.overall)),
                _pct(result.report.macro()),
            )
        )
    lines.append("")

    lines.append("## Denominators and exclusions")
    lines.append("")
    header = ["method", "questions", "scorable", "unanswerable", "SA gold not numeric", "empty context", "non-conforming"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for source, result in results.items():
        exclusions = result.report.exclusion_summary()
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                source,
                result.n_questions,
                exclusions["scorable"],
                exclusions.get("unanswerable", 0),
                exclusions.get("not_numeric_gold", 0),
                result.n_empty_context,
                result.n_nonconforming,
            )
        )
    lines.append("")

    lines.append("## Per-format denominators")
    lines.append("")
    covered = sorted({fmt for source in results.values() for fmt in source.report.formats})
    if covered:
        # ``.value`` rather than the enum itself: ``AnswerFormat`` subclasses ``str``, so f-stringing it
        # yields "AnswerFormat.DESCRIPTION" on 3.11+ — a header that names this module's type rather than
        # the release's tag.
        header = ["method"] + [f"{fmt.value} (scorable/total)" for fmt in covered]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for source, result in results.items():
            cells = []
            for fmt in covered:
                scorable, total = result.report.counts(fmt)
                cells.append(f"{scorable}/{total}")
            lines.append("| `{}` | {} |".format(source, " | ".join(cells)))
        lines.append("")
    else:
        lines.append("_No questions were scored._")
        lines.append("")

    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "* **SA is the only column with a large unscorable share.** 39.58% of the release's NU-tagged "
        "golds are not a single numeric value (a coordinate pair, a date, a per-horizon mapping, or a "
        "placeholder), so no tolerance test exists for them. They leave the denominator and are counted "
        "above rather than scored as wrong — see `src/okh_rag/evaluation/answers.py`."
    )
    lines.append(
        "* **The four methods share one question set and one generator.** A column that is high because it "
        "was scored on fewer questions is visible in the denominators table, not hidden in the mean."
    )
    lines.append(
        "* **`overall (micro)` is Table 1's \"Overall\".** It is not the macro-average of the four columns; "
        "both are reported because §4.1's own table is micro while its column layout suggests macro."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--data", default=str(REPO_ROOT / "data" / "raw" / "cyportqa"))
    parser.add_argument("--out", default=str(REPO_ROOT / "reports"))
    parser.add_argument(
        "--backend",
        default="lexical",
        choices=("lexical", "openai"),
        help="lexical = offline wiring check (the default); openai = §4.1's models",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_SOURCES),
        choices=list(ALL_SOURCES),
        help="which methods to run",
    )
    parser.add_argument("--mode", default=None, choices=("single_call", "per_trajectory"))
    parser.add_argument(
        "--transition-variant",
        default=None,
        choices=("learned", "heuristic", "none"),
        help=(
            "the order model P_theta to retrieve with. Defaults to cfg.ablation.transition_variant. "
            "'learned' needs a trained checkpoint — run scripts/run_generation.py only after the training "
            "stage, or pass 'heuristic' for an offline check."
        ),
    )
    parser.add_argument(
        "--order-model",
        default=None,
        help="path to a LearnedTransitionModel checkpoint (only used when the variant is 'learned')",
    )
    parser.add_argument("--scenario-limit", type=int, default=24)
    parser.add_argument("--retrieve", type=int, default=120, help="question cap (0 = all)")
    parser.add_argument("--shuffle", action="store_true", help="Table 2's Shuffle row")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    data_root = Path(args.data)

    # §4.5 Table 2's Shuffle row is a config switch, not a call-site argument: the retriever reads
    # ``ablation.shuffle_trajectory`` per query, and overriding it here (rather than passing a flag down)
    # means the shuffle follows *exactly* the code path the Table 2 harness uses, including the re-scoring
    # of the permuted sequence. A driver that implemented its own shuffling could produce a different drop
    # from the same data and no one would be able to say which one was Table 2.
    if args.shuffle:
        import dataclasses

        cfg = dataclasses.replace(
            cfg, ablation=dataclasses.replace(cfg.ablation, shuffle_trajectory=True)
        )
    if args.transition_variant:
        import dataclasses

        cfg = dataclasses.replace(
            cfg, ablation=dataclasses.replace(cfg.ablation, transition_variant=args.transition_variant)
        )

    order_model = load_order_model(cfg, args.order_model)

    plan = RunPlan(
        backend=args.backend,
        mode=args.mode or cfg.generation.mode,
        sources=tuple(args.sources),
        shuffle=args.shuffle,
        scenario_limit=args.scenario_limit,
        question_limit=args.retrieve or None,
        seed=cfg.seed if args.seed is None else args.seed,
    )
    print("Plan:")
    for key, value in plan.as_dict().items():
        print(f"  {key}: {value}")

    random.seed(plan.seed)
    np.random.seed(plan.seed)

    print("Loading corpus...")
    corpus = load_corpus(data_root)
    records = select_records(
        corpus, scenario_limit=args.scenario_limit, question_limit=args.retrieve or None
    )
    scenarios = sorted({record.scenario_id for record in records})
    print(f"  {len(records)} questions over {len(scenarios)} scenarios")

    print("Building the reference hypergraph...")
    graph = build_reference_graph(corpus, cfg=cfg, scenario_ids=scenarios)
    print(f"  {len(graph.hypergraph)} hyperedges, {len(graph.hypergraph.entities)} entities")

    embedder, generator_backend = build_backend(args.backend, cfg)
    embeddings = build_embeddings(graph, cfg, embedder)

    print("Building retrievers...")
    retrievers = build_retrievers(
        graph, embeddings, cfg, embedder, sources=plan.sources, order_model=order_model
    )

    print("Embedding queries...")
    query_vectors = {
        record.qa_id: vector
        for record, vector in zip(
            records,
            embed_texts([record.question for record in records], backend=embedder, cfg=cfg, store=None),
            strict=True,
        )
    }

    generator = Generator(generator_backend, cfg)
    results: dict[str, SourceResult] = {}
    for source in plan.sources:
        if source not in retrievers:
            print(f"-- {source}: skipped (not constructed) --")
            continue
        print(f"-- {source} --")
        results[source] = run_source(
            source=source,
            retriever=retrievers[source],
            records=records,
            query_vectors=query_vectors,
            graph=graph,
            generator=generator,
            cfg=cfg,
            plan=plan,
        )
        summary = results[source]
        print(
            f"   {summary.n_questions} questions in {summary.elapsed_s:.1f}s; "
            f"overall={_pct(summary.report.overall(cfg.evaluation.overall))}; "
            f"empty={summary.n_empty_context}; nonconforming={summary.n_nonconforming}"
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = out_dir / "generation_eval.md"
    json_path = out_dir / "generation_eval.json"

    markdown_path.write_text(render_markdown(results, plan, cfg), encoding="utf-8")
    payload = {
        "plan": plan.as_dict(),
        "llm": {
            "generator_model": cfg.llm.generator_model,
            "generation_temperature": cfg.llm.generation_temperature,
        },
        "evaluation": {
            "td_scoring": cfg.evaluation.td_scoring,
            "overall": cfg.evaluation.overall,
            "exclude_unanswerable": cfg.evaluation.exclude_unanswerable,
            "sa_relative_tolerance": cfg.evaluation.sa_relative_tolerance,
            "sa_absolute_floor": cfg.evaluation.sa_absolute_floor,
        },
        "generator_calls": getattr(generator_backend, "calls", None),
        "results": {
            source: result.as_dict(overall_mode=cfg.evaluation.overall)
            for source, result in results.items()
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nWrote {markdown_path}")
    print(f"Wrote {json_path}")
    if args.backend != "openai":
        print(
            "\nNOTE: --backend lexical uses a stub generator, so the accuracies above are NOT §4.1's. "
            "They verify the wiring only."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
