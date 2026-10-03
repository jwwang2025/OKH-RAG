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

**§4.1's sample is the whole benchmark.** §4.1 evaluates on "117,178 questions" over CyPortQA's full
release — no subsample is mentioned anywhere in the section. The release on disk is that population:
472 scenario keys, 10 horizon blocks, and a type mix of TF 5,812 / MC 37,813 / NU 49,459 / DE 24,094.
:func:`select_records` therefore draws a **seeded random sample of that population**, never a prefix of
it, and ``--scenario-limit 0 --retrieve 0`` means the full release. See :func:`_print_sample_composition`,
which prints the run's mix next to §4.1's so the two can be compared.

Usage::

    # wiring check, no API key, no real accuracy
    python scripts/run_generation.py --retrieve 40

    # §4.1 as written, on the full release (needs OPENAI_API_KEY; ~81 h sequential)
    python scripts/run_generation.py --backend openai --scenario-limit 0 --retrieve 0 \
        --dump reports/table1.jsonl --progress-every 1000

    # a smaller random subset of the same population, for a quick read
    python scripts/run_generation.py --backend openai --retrieve 2000

    # §4.5 Table 2's Shuffle row
    python scripts/run_generation.py --backend openai --shuffle
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time

# Windows + Anaconda: pip-installed torch and MKL-backed numpy each ship their own
# libiomp5md.dll; loading both in one process aborts with "OMP: Error #15".
# Must be set before numpy/torch initialize their OpenMP runtime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from okh_rag.config import Config, load_config  # noqa: E402
from okh_rag.data.cyportqa import QARecord, load_corpus  # noqa: E402
from okh_rag.embeddings import (  # noqa: E402
    EmbeddingStore,
    HyperedgeEmbeddings,
    embed_texts,
    hyperedge_embedding_text,
)
from okh_rag.evaluation.answers import AccuracyReport, score_record  # noqa: E402
from okh_rag.evaluation.reference_graph import build_reference_graph  # noqa: E402
from okh_rag.generation.evidence import context_from_retrieval  # noqa: E402
from okh_rag.generation.generator import Generator  # noqa: E402
from okh_rag.ordering.precedence import build_precedence  # noqa: E402
from okh_rag.retrieval import OrderAwareRetriever  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


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
            "Either train one first (see scripts/train_order_model.py), or pass "
            "--transition-variant heuristic for an offline wiring check."
        )
    from okh_rag.ordering.model import LearnedTransitionModel

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
    shuffle: bool
    scenario_limit: int | None
    question_limit: int | None
    seed: int
    dump: str | None = None
    progress_every: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "answer_mode": self.mode,
            "shuffle_trajectory": self.shuffle,
            "scenario_limit": self.scenario_limit,
            "question_limit": self.question_limit,
            "seed": self.seed,
            "dump": self.dump,
            "progress_every": self.progress_every,
        }


def select_records(
    corpus,
    *,
    scenario_limit: int | None,
    question_limit: int | None,
    seed: int = 42,
) -> list[QARecord]:
    """A seeded random sample of §4.1's question population.

    §4.1 evaluates on CyPortQA as a whole: **117,178 questions over 472 scenario keys and 10 horizon
    blocks**, no subsample is mentioned. A reproduction's sample therefore has to be a *random subset of
    that population*, not a different population.

    The previous implementation sorted the scenario keys, concatenated them, and truncated the flat
    list. With ``--scenario-limit 24 --retrieve 200`` that stopped inside the second scenario
    (ALEX_2022_Canaveral 160 + Jacksonville 40) — one storm, two ports, 644 hyperedges — so every
    column in Table 1 was estimated on a single storm while the report claimed 24 scenarios. Worse,
    ``--scenario-limit`` was silently inert for any question count below a scenario's own size.

    Both stages are now drawn with a seeded RNG, the way ``scripts/run_retrieval_oracle.py`` already
    does:

    * scenarios — ``random.Random(seed).sample(all_scenarios, scenario_limit)``;
    * questions — ``random.Random(seed).sample(pool, question_limit)`` over the pool of *every*
      question belonging to the selected scenarios.

    ``None`` (``--scenario-limit 0`` / ``--retrieve 0``) means the full release, which is §4.1's
    setting. Sampling rather than sorting-and-truncating is what makes a smaller ``n`` defensible: it
    is the same population measured less precisely, and the seed makes it reproducible.

    Sampling is unstratified on purpose. CyPortQA's own mix (TF 5,812 / MC 37,813 / NU 49,459 /
    DE 24,094) *is* the paper's mix, and a random draw reproduces it in expectation; stratifying would
    silently reweight the micro-average away from the paper's denominator.
    """
    rng = random.Random(seed)
    all_scenarios = sorted(corpus.qa)
    if scenario_limit is None or scenario_limit >= len(all_scenarios):
        scenario_ids = list(all_scenarios)
    else:
        scenario_ids = sorted(rng.sample(all_scenarios, scenario_limit))

    pool: list[QARecord] = []
    for scenario_id in scenario_ids:
        pool.extend(corpus.qa[scenario_id])
    if question_limit is None or question_limit >= len(pool):
        pool.sort(key=lambda record: record.qa_id)
        return pool
    sample = rng.sample(pool, question_limit)
    sample.sort(key=lambda record: record.qa_id)
    return sample


_PAPER_MIX: Mapping[str, int] = {"TF": 5812, "MC": 37813, "NU": 49459, "DE": 24094}
"""§4.1's question mix over the whole release (117,178). Printed next to the run's own so a reader
can see whether a Table 1 was measured on the paper's population or on a convenient corner of it."""


def _print_sample_composition(records: Sequence[QARecord]) -> None:
    """The sample's type / horizon / scenario mix, against §4.1's."""
    formats = collections.Counter(record.answer_format.value for record in records)
    horizons = collections.Counter(record.context.horizon_hours for record in records)
    total = len(records)
    print("  composition (vs §4.1's full release):")
    for fmt in ("TF", "MC", "NU", "DE"):
        share = formats[fmt] / total if total else 0.0
        paper_share = _PAPER_MIX[fmt] / sum(_PAPER_MIX.values())
        print(
            f"    {fmt}: {formats[fmt]:7d} ({share:6.2%})   paper {_PAPER_MIX[fmt]:7d} "
            f"({paper_share:6.2%})"
        )
    print(f"    horizons: {len(horizons)} distinct "
          + " ".join(f"T-{h}={n}" for h, n in sorted(horizons.items())))
    print(
        f"    storms: {len({r.context.storm for r in records})}   "
        f"templates: {len({r.template_id for r in records})}"
    )


def build_embeddings(graph, cfg: Config, embedder, store=None) -> HyperedgeEmbeddings:
    """Embed every hyperedge of the reference graph, in sorted id order so the matrix is reproducible.

    ``store`` must be passed at §4.1 scale. The full release is **248,652 hyperedges**, and embedding
    them from scratch is ~248k API calls — hours of wall clock on top of a generation run already
    measured at ~81 h, and it would have to be repeated after any interruption. The content-addressed
    cache turns a warm run into zero calls; :func:`scripts.preflight_full.py` reports how many are
    already cached.
    """
    ordered_ids = tuple(sorted(graph.hypergraph.hyperedges))
    texts = [
        hyperedge_embedding_text(graph.hypergraph.hyperedges[hid], graph.hypergraph) for hid in ordered_ids
    ]
    matrix = embed_texts(texts, backend=embedder, cfg=cfg, store=store)
    return HyperedgeEmbeddings(hyperedge_ids=ordered_ids, matrix=matrix)


def build_retrievers(
    graph, embeddings, cfg: Config, embedder, *, order_model: object | None = None
) -> dict[str, Any]:
    """Construct the OKH-RAG retriever."""
    return {
        "okh_rag": OrderAwareRetriever(
            cfg=cfg,
            hypergraph=graph.hypergraph,
            precedence=build_precedence(graph.hypergraph, cfg.precedence),
            embeddings=embeddings,
            order_model=order_model,
            backend=None,
        )
    }


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

    # A §4.1-scale run is ~81 h of sequential gpt-4o calls. Two things protect it: a progress line
    # with an ETA, so a stall is distinguishable from slow progress, and an append-only JSONL dump, so
    # an interruption at hour 70 does not discard 70 hours of paid answers. The dump is flushed per
    # question because a buffered write is exactly what a crash loses.
    dump_path = Path(plan.dump) if plan.dump else None
    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text("", encoding="utf-8")

    def _dump(entry: dict) -> None:
        if dump_path is None:
            return
        with dump_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    progress_every = plan.progress_every or 0
    total = len(records)

    for position, record in enumerate(records, start=1):
        if progress_every and (position % progress_every == 0 or position == total):
            elapsed = time.time() - started
            rate = elapsed / position
            remaining = (total - position) * rate
            print(
                f"    {position}/{total} ({position / total:.1%})  "
                f"elapsed {elapsed / 3600:.2f}h  eta {remaining / 3600:.2f}h  "
                f"({rate:.2f}s/question)",
                flush=True,
            )
        query_vector = query_vectors[record.qa_id]
        if source == "okh_rag":
            retrieval = retriever.retrieve(
                record.question,
                group_id=record.group_id,
                query_vector=query_vector,
                horizon_hours=record.context.horizon_hours,
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

        _dump(
            {
                "position": position,
                "qa_id": record.qa_id,
                "template_id": record.template_id,
                "scenario_id": record.scenario_id,
                "horizon_hours": record.context.horizon_hours,
                "format": record.answer_format.value,
                "question": record.question,
                "gold": record.answer if isinstance(record.answer, str) else json.dumps(record.answer, ensure_ascii=False),
                "predicted": generated.answer.answer if isinstance(generated.answer.answer, str) else json.dumps(generated.answer.answer, ensure_ascii=False),
                "correct": score.correct,
                "scorable": score.scorable,
                "reason": score.reason,
                "detail": score.detail,
                "conforming": generated.conforming,
                "empty_context": context.is_empty,
            }
        )

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
    parser.add_argument(
        "--scenario-limit",
        type=int,
        default=0,
        help="knowledge groups to build the hypergraph over; sampled with --seed from all 472. "
        "0 = the full release (§4.1's setting)",
    )
    parser.add_argument(
        "--retrieve",
        type=int,
        default=120,
        help="questions to answer; sampled with --seed from every question of the selected "
        "scenarios. 0 = all 117,178 (§4.1's setting)",
    )
    parser.add_argument("--shuffle", action="store_true", help="Table 2's Shuffle row")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--dump",
        default=None,
        help="append-only JSONL of every question's answer and score. Strongly recommended at "
        "§4.1 scale: an interruption without it discards every answer already paid for",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="print progress and ETA every N questions (0 = silent)",
    )
    parser.add_argument(
        "--cache-root",
        default=str(REPO_ROOT / "artifacts" / "llm_cache"),
        help="content-addressed embedding cache root; hyperedges and questions use subdirectories",
    )
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
        shuffle=args.shuffle,
        scenario_limit=args.scenario_limit,
        question_limit=args.retrieve or None,
        seed=cfg.seed if args.seed is None else args.seed,
        dump=args.dump,
        progress_every=args.progress_every,
    )
    print("Plan:")
    for key, value in plan.as_dict().items():
        print(f"  {key}: {value}")

    random.seed(plan.seed)
    np.random.seed(plan.seed)

    print("Loading corpus...")
    corpus = load_corpus(data_root)
    records = select_records(
        corpus,
        scenario_limit=args.scenario_limit or None,
        question_limit=args.retrieve or None,
        seed=plan.seed,
    )
    scenarios = sorted({record.scenario_id for record in records})
    print(f"  {len(records)} questions over {len(scenarios)} scenarios")
    _print_sample_composition(records)

    print("Building the reference hypergraph...")
    graph = build_reference_graph(corpus, cfg=cfg, scenario_ids=scenarios)
    print(f"  {len(graph.hypergraph)} hyperedges, {len(graph.hypergraph.entities)} entities")

    embedder, generator_backend = build_backend(args.backend, cfg)
    cache_root = Path(args.cache_root)
    hyperedge_store = EmbeddingStore(cache_root / "embeddings", dim=cfg.order_model.embedding_dim)
    print("Embedding hyperedges (cache-first)...")
    embeddings = build_embeddings(graph, cfg, embedder, store=hyperedge_store)

    print("Building retrievers...")
    retrievers = build_retrievers(
        graph, embeddings, cfg, embedder, order_model=order_model
    )

    print("Embedding queries...")
    # Question texts repeat heavily (the same template is asked once per horizon block per port), so
    # the cache collapses 117,178 questions onto a few thousand distinct strings.
    question_store = EmbeddingStore(cache_root / "questions", dim=cfg.order_model.embedding_dim)
    query_vectors = {
        record.qa_id: vector
        for record, vector in zip(
            records,
            embed_texts(
                [record.question for record in records], backend=embedder, cfg=cfg, store=question_store
            ),
            strict=True,
        )
    }
    print(f"  cache: hyperedges {hyperedge_store.hits} hit / {hyperedge_store.misses} miss; "
          f"questions {question_store.hits} hit / {question_store.misses} miss")

    generator = Generator(generator_backend, cfg)
    results: dict[str, SourceResult] = {}
    source = "okh_rag"
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
