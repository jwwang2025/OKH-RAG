"""Typed configuration loaded from ``configs/base.yaml``.

The YAML file is the single source of truth for every hyperparameter, and every value there carries
either a ``§`` citation or an ``[UNSPECIFIED]`` / ``[CONTRADICTION]`` marker. This module only
*transports* those values into typed objects — it introduces no defaults of its own, so a value can
never enter the program without having been justified in the config file.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

DEFAULT_CONFIG_PATH = Path("configs/base.yaml")

T = TypeVar("T")

# ---------------------------------------------------------------------------
# The legal values of the switches that reconcile a contradiction
# ---------------------------------------------------------------------------
#
# Five config keys exist only because the paper states one thing in an equation and another in an
# algorithm box. Each gets a switch, and each switch needs one authoritative list of legal values.
# Declared *here* rather than next to the code that consumes them because this module is a leaf — it
# imports nothing from the package — so it can be imported by every layer without a cycle, whereas
# ``okh_rag.retrieval`` has to import this module and therefore cannot own a list that this module must
# validate against. A duplicated literal in ``_validate`` and in the consuming module is exactly the drift
# the config file's CONVENTION block exists to prevent.

PRECEDENCE_MODES: tuple[str, ...] = ("fraction", "indicator")
"""§3.2 Eq. 5 weights µ by the trajectory *fraction* ``Prec(γ)``; §A.7 Algorithm 1 line 6 weights it by the
0/1 indicator ``1[e_(ℓ−1) ≺ e′]``. Default ``"fraction"`` — equations outrank algorithm boxes."""

OVERLAP_MODES: tuple[str, ...] = ("mean", "sum")
"""§A.6 Eq. 12 defines ``Ovlp(γ)`` as the *mean* Jaccard over consecutive pairs; Algorithm 1 line 7
accumulates the raw Jaccard, i.e. the sum. Default ``"mean"``. The two differ by the constant factor
``L−1`` at fixed ``L``, so they cannot change a ranking — only the scale of ν relative to the other
weights. Both are implemented so that claim is testable rather than asserted."""

GROUP_RESERVE_MODES: tuple[str, ...] = ("floor", "quota")
"""§A.7 reserves "40% of slots ... for the query's own group". ``"floor"`` reads the reserve as a minimum
that is satisfied before the expansion spends the rest; ``"quota"`` reads the 150 slots as *containing* the
reserve, leaving 90 for the general pool. Default ``"floor"``, the reading under which the expansion stage
is not left with 10 slots. See :mod:`okh_rag.retrieval.candidates`."""

TRANSITION_VARIANTS: tuple[str, ...] = ("learned", "heuristic", "none")
"""Table 2's *Transition* group. ``"learned"`` is §3.1 Eq. 3, ``"heuristic"`` is §A.3's rules only, and
``"none"`` collapses the objective to relevance — §3.2's "setting all to zero recovers standard top-k
retrieval". See :mod:`okh_rag.retrieval.retriever` for why ``"none"`` must zero the weights too."""

INFERENCE_MODES: tuple[str, ...] = ("beam_search", "viterbi")
"""§A.7 — "Beam search serves as the primary inference algorithm; Viterbi DP provides exact optimization
for smaller candidate sets." Default ``"beam_search"``."""


# ---------------------------------------------------------------------------
# Leaf sections
# ---------------------------------------------------------------------------


@dataclass
class PaperConfig:
    """Bibliographic anchors. §Title page."""

    arxiv_id: str
    title: str


@dataclass
class PathsConfig:
    """Where user-supplied data lives and where regenerable artifacts are written."""

    data_dir: Path
    reports_dir: Path
    work_dir: Path

    def resolve(self, base: Path) -> "PathsConfig":
        """Make every path absolute relative to ``base`` (usually the repo root)."""
        return PathsConfig(
            data_dir=(base / self.data_dir).resolve(),
            reports_dir=(base / self.reports_dir).resolve(),
            work_dir=(base / self.work_dir).resolve(),
        )


@dataclass
class LLMConfig:
    """§4.1 generator, §A.4 embedding model, and the operational knobs around them."""

    generator_model: str
    embedding_model: str
    embedding_batch_size: int
    extraction_temperature: float
    generation_temperature: float
    judge_temperature: float
    max_retries: int
    retry_base_delay_s: float
    request_timeout_s: int
    cache_dir: Path

    def resolve(self, base: Path) -> "LLMConfig":
        self.cache_dir = (base / self.cache_dir).resolve()
        return self


@dataclass
class HorizonConfig:
    """§A.2 per-horizon decomposition."""

    offsets_hours: list[int]
    header_pattern: str
    anchor_template: str


@dataclass
class NormalizationConfig:
    """§A.2 steps 1-4 post-processing."""

    fuzzy_threshold: float
    entity_id_template: str
    cross_horizon_families: list[str]
    cross_horizon_max_fanout: int
    cross_horizon_relation: str
    cross_horizon_probability_relation: str


@dataclass
class DedupConfig:
    """§A.2 step 5 deduplication."""

    hash_algorithm: str
    hash_length: int
    sort_entity_names: bool


@dataclass
class HypergraphConfig:
    """§3.1 / §A.1-A.2 knowledge hypergraph construction."""

    min_entity_set_size: int
    min_confidence: float
    max_confidence: float
    horizon: HorizonConfig
    normalization: NormalizationConfig
    dedup: DedupConfig


@dataclass
class PrecedenceConfig:
    """§A.3 precedence construction."""

    phase_order: list[str]
    cross_horizon_direction: str
    causal_chain_rules: list[list[str]]
    tie_break_order: list[str]
    cycle_policy: str


@dataclass
class OrderModelConfig:
    """§3.1 Eq. 3 / §A.4 learned transition model."""

    embedding_dim: int
    rank: int
    l2_normalize_embeddings: bool
    init: str


@dataclass
class SignalsConfig:
    """§A.4 the three self-supervised signal families."""

    document_order: bool
    entity_overlap: bool
    retrieval_induced: bool


@dataclass
class SplitConfig:
    """Train/val/test split. [UNSPECIFIED] by the paper."""

    strategy: str
    train: float
    val: float
    test: float


@dataclass
class TrainingConfig:
    """§3.1 Eq. 4 / §A.4 contrastive training."""

    negative_term_sign: str
    alpha: float
    sampled_softmax_negatives: int
    sampled_softmax_correction: bool
    optimizer: str
    lr: float
    betas: list[float]
    eps: float
    weight_decay: float
    batch_size: int
    epochs: int
    early_stop_patience: int
    gradient_clip_norm: float
    signals: SignalsConfig
    document_order_window: int
    entity_overlap_ignore_anchors: bool
    retrieval_induced_rounds: int
    retrieval_induced_min_accuracy: float
    split: SplitConfig


@dataclass
class RetrievalWeights:
    """§4.1 — (λ, µ, ν, ρ) = (1.2, 0.3, 0.2, 0.5).

    ``lambda_`` carries a trailing underscore because ``lambda`` is a Python keyword; it is λ in Eq. 5.
    """

    lambda_: float
    mu: float
    nu: float
    rho: float


@dataclass
class RetrievalConfig:
    """§3.2 Eq. 5 / §A.6-A.7 order-aware retrieval."""

    weights: RetrievalWeights
    precedence_mode: str
    overlap_mode: str
    group_reserve_mode: str
    top_k: int
    candidate_pool_size: int
    group_reserve_fraction: float
    beam_width: int
    beam_length: int
    num_trajectories: int
    beam_initial_multiplier: int
    diversity_penalty: float
    phase_set: list[str]


@dataclass
class GenerationConfig:
    """§3.3 / §A.8 retrieval-augmented generation."""

    mode: str
    include_quality_summary: bool
    include_reasoning_tags: bool


@dataclass
class EvaluationConfig:
    """§4.1 metrics."""

    judge_model: str
    td_scoring: str
    sa_relative_tolerance: float
    sa_absolute_floor: float
    case_sensitive: bool
    overall: str
    exclude_unanswerable: bool


@dataclass
class TextRagConfig:
    chunking: str
    top_k: int


@dataclass
class GraphRagConfig:
    expansion_hops: int
    include_pairwise_edges: bool


@dataclass
class HypergraphRagConfig:
    top_k: int


@dataclass
class BaselinesConfig:
    """§4.1 baselines."""

    text_rag: TextRagConfig
    graph_rag: GraphRagConfig
    hypergraph_rag: HypergraphRagConfig


@dataclass
class AblationConfig:
    """§4.5 / Table 2 ablation switches."""

    shuffle_trajectory: bool
    drop_lambda: bool
    drop_mu: bool
    drop_nu: bool
    drop_rho: bool
    transition_variant: str


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Root config. Mirrors ``configs/base.yaml`` one-to-one."""

    paper: PaperConfig
    seed: int
    paths: PathsConfig
    llm: LLMConfig
    hypergraph: HypergraphConfig
    precedence: PrecedenceConfig
    order_model: OrderModelConfig
    training: TrainingConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    baselines: BaselinesConfig
    ablation: AblationConfig

    # Not from YAML: filled in by :func:`load_config`.
    root: Path = field(default_factory=Path)

    def effective_weights(self) -> RetrievalWeights:
        """Apply the §4.5 component ablations to the §4.1 weights.

        Table 2 removes one structural term at a time, which in Eq. 5 terms means zeroing that weight.
        Returns:
            A copy of ``retrieval.weights`` with the ablated terms set to 0.0.
        """
        w = self.retrieval.weights
        return RetrievalWeights(
            lambda_=0.0 if self.ablation.drop_lambda else w.lambda_,
            mu=0.0 if self.ablation.drop_mu else w.mu,
            nu=0.0 if self.ablation.drop_nu else w.nu,
            rho=0.0 if self.ablation.drop_rho else w.rho,
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _build(cls: type[T], data: Any) -> T:
    """Recursively instantiate a dataclass tree from nested dicts.

    Extra keys raise, so a typo in the YAML cannot silently disable a setting. Missing keys raise too,
    except for fields that declare a default — those are genuinely optional.

    Field types are resolved with :func:`typing.get_type_hints` rather than read off ``f.type``,
    because this module uses ``from __future__ import annotations`` and ``f.type`` is therefore a
    string; ``is_dataclass`` on a string is always ``False`` and nested sections would silently be
    passed through as raw dicts.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping to build {cls.__name__}, got {type(data).__name__}")

    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    optional = {
        f.name
        for f in fields(cls)
        if f.default is not MISSING or f.default_factory is not MISSING
    }

    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown key(s) for {cls.__name__}: {sorted(unknown)}")
    missing = known - set(data) - optional
    if missing:
        raise ValueError(f"missing key(s) for {cls.__name__}: {sorted(missing)}")

    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue  # optional field, let the dataclass default apply
        value = data[f.name]
        ftype = hints.get(f.name)
        if isinstance(ftype, type) and is_dataclass(ftype):
            kwargs[f.name] = _build(ftype, value)
        else:
            kwargs[f.name] = _coerce_path(f.name, value)
    return cls(**kwargs)  # type: ignore[arg-type]


# Fields that must become Path objects after coercion.
_PATH_FIELDS = {"data_dir", "reports_dir", "work_dir", "cache_dir"}


def _coerce_path(name: str, value: Any) -> Any:
    if name in _PATH_FIELDS and isinstance(value, str):
        return Path(value)
    return value


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Read ``configs/base.yaml`` into a :class:`Config`, validating every key.

    Args:
        path: Path to the YAML file. Relative paths resolve against the current working directory.

    Returns:
        A fully populated :class:`Config` whose ``root`` is the repo root (the config file's
        grandparent, i.e. the directory containing ``configs/``).
    """
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = _build(Config, raw)
    root = path.parent.parent
    cfg.root = root
    cfg.paths = cfg.paths.resolve(root)
    cfg.llm = cfg.llm.resolve(root)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """Fail fast on values that would produce silently wrong behaviour."""
    # §A.1 Eq. 7 — |V_e| >= 2.
    if cfg.hypergraph.min_entity_set_size < 2:
        raise ValueError("hypergraph.min_entity_set_size must be >= 2 (§A.1 Eq. 7: |V_e| >= 2)")

    # §A.1 — c in (0, 1].
    if not (0.0 < cfg.hypergraph.max_confidence <= 1.0):
        raise ValueError("hypergraph.max_confidence must be in (0, 1] (§A.1)")

    # §A.4 — parameter count claim: 2 * rank * embedding_dim == 196,608.
    expected = 2 * cfg.order_model.rank * cfg.order_model.embedding_dim
    if expected != 196_608:
        raise ValueError(
            "order_model rank x embedding_dim must reproduce §A.4's parameter count: "
            f"2 * {cfg.order_model.rank} * {cfg.order_model.embedding_dim} = {expected}, expected 196608"
        )

    if not cfg.hypergraph.normalization.cross_horizon_families:
        raise ValueError(
            "hypergraph.normalization.cross_horizon_families must be non-empty (§A.2 step 4); "
            'use ["all"] for every family'
        )

    if cfg.hypergraph.normalization.cross_horizon_max_fanout < 0:
        raise ValueError("hypergraph.normalization.cross_horizon_max_fanout must be >= 0")

    # §A.6 Eq. 11 / Eq. 12 / Eq. 13 depend on the phase set being non-empty.
    if not cfg.retrieval.phase_set:
        raise ValueError("retrieval.phase_set must be non-empty (§A.6 Eq. 13 divides by |S|)")

    if cfg.retrieval.precedence_mode not in PRECEDENCE_MODES:
        raise ValueError(f"retrieval.precedence_mode must be one of {PRECEDENCE_MODES}")

    if cfg.retrieval.overlap_mode not in OVERLAP_MODES:
        raise ValueError(f"retrieval.overlap_mode must be one of {OVERLAP_MODES}")

    if cfg.retrieval.group_reserve_mode not in GROUP_RESERVE_MODES:
        raise ValueError(f"retrieval.group_reserve_mode must be one of {GROUP_RESERVE_MODES}")

    if not (0.0 <= cfg.retrieval.group_reserve_fraction <= 1.0):
        raise ValueError(
            f"retrieval.group_reserve_fraction must be in [0, 1], got {cfg.retrieval.group_reserve_fraction}"
        )

    # §A.7 — stage 1 contributes top-K and the pool is capped at |C|. A K larger than the cap means the
    # anchors alone overflow it, and the two stages stop composing.
    if cfg.retrieval.top_k > cfg.retrieval.candidate_pool_size:
        raise ValueError(
            f"retrieval.top_k ({cfg.retrieval.top_k}) exceeds retrieval.candidate_pool_size "
            f"({cfg.retrieval.candidate_pool_size}); §A.7 pipes stage 1 into stage 2 under one cap"
        )

    if cfg.retrieval.top_k < 1 or cfg.retrieval.candidate_pool_size < 1:
        raise ValueError("retrieval.top_k and retrieval.candidate_pool_size must both be >= 1")

    if cfg.retrieval.beam_width < 1 or cfg.retrieval.beam_length < 1 or cfg.retrieval.num_trajectories < 1:
        raise ValueError("retrieval.beam_width, beam_length and num_trajectories must all be >= 1")

    if cfg.retrieval.beam_initial_multiplier < 1:
        raise ValueError("retrieval.beam_initial_multiplier must be >= 1 (§A.7 line 2 takes top-2B)")

    if cfg.retrieval.diversity_penalty < 0.0:
        raise ValueError("retrieval.diversity_penalty must be >= 0")

    # §A.6 Eq. 13 — S is a config value while φ is baked into the relation table, so a member of S that no
    # relation maps onto is silently unusable: it inflates the denominator of Cov without ever being able
    # to fill the numerator, deflating ρ's range for every trajectory in the run. Imported locally because
    # this module is a leaf and must stay one.
    from okh_rag.retrieval.scoring import unreachable_phase_set

    unreachable = unreachable_phase_set(cfg.retrieval.phase_set)
    if unreachable:
        raise ValueError(
            f"retrieval.phase_set members {list(unreachable)} are unreachable: no relation in §A.1 Table 3 "
            "maps onto them via §A.6 Eq. 13's phi, so Cov could never reach 1.0"
        )

    if cfg.training.negative_term_sign not in {"minus", "literal"}:
        raise ValueError("training.negative_term_sign must be 'minus' (corrected) or 'literal' (Eq. 4 as printed)")

    if cfg.ablation.transition_variant not in {"learned", "heuristic", "none"}:
        raise ValueError("ablation.transition_variant must be 'learned', 'heuristic', or 'none'")

    if cfg.training.document_order_window < 1:
        raise ValueError("training.document_order_window must be >= 1")

    split_total = cfg.training.split.train + cfg.training.split.val + cfg.training.split.test
    if abs(split_total - 1.0) > 1e-6:
        raise ValueError(f"training.split fractions must sum to 1.0, got {split_total}")


def dump_config(cfg: Config) -> dict[str, Any]:
    """Return the config as a plain, JSON-serialisable dict (used when logging a run)."""

    def _plain(obj: Any) -> Any:
        if is_dataclass(obj):
            return {f.name: _plain(getattr(obj, f.name)) for f in fields(obj)}
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_plain(v) for v in obj]
        return obj

    data = _plain(cfg)
    data.pop("root", None)
    return data
