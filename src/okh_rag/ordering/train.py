"""§3.1 Eq. 4 / §A.4 — training the transition model.

§3.1 gives the objective as an equation and nothing else:

    L(θ) = E_P[−log P_θ(e_j | e_i)] + α · E_N[log P_θ(e_m | e_i)]

§A.4 supplies the operational details: the partition function is approximated by "sampled softmax with
K = 64 global negatives per example", negatives are "random cross-group pairs", and the positive
distribution ``P`` is the union of the three self-supervised signals in
:mod:`okh_rag.training.pairs`.

**Eq. 4's sign.** As printed, the second term *adds* ``α · E_N[log P_θ]``. Since ``log P_θ ≤ 0``, that
makes the loss smaller whenever the model assigns higher probability to a negative — it rewards exactly
what a contrastive objective must penalise, and it fights the first term. §A.4 describes the intent as
contrastive ("positive and negative ordered pairs"). Both readings are implemented, selected by
``training.negative_term_sign``: ``"minus"`` (the default, matching the stated intent) and ``"literal"``
(Eq. 4 exactly as printed). Keeping both means the discrepancy can be measured instead of argued about,
which is the only honest thing to do with a published equation that appears to be wrong.

**Why the two terms share a denominator.** Eq. 4's ``P_θ`` is one distribution. :meth:`contrastive_loss`
therefore calls
:meth:`~okh_rag.order_model.LearnedTransitionModel.log_prob_sampled_with_negatives` once, so the positive
and the ``K`` negatives are normalised together, and averages ``log P_θ`` over both sides of *that same*
softmax. Computing the negative term against a differently-normalised softmax would make ``α``
uninterpretable.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from okh_rag.config import Config, TrainingConfig
from okh_rag.embeddings import HyperedgeEmbeddings
from okh_rag.order_model import LearnedTransitionModel
from okh_rag.schema import KnowledgeHypergraph
from okh_rag.seeding import derive_seed, set_global_seed
from okh_rag.training.pairs import (
    SIGNAL_RETRIEVAL_INDUCED,
    NegativeSampler,
    OrderedPair,
    TrainingPairs,
    build_training_pairs,
)

__all__ = [
    "LossBreakdown",
    "TrainOutcome",
    "TrainReport",
    "contrastive_loss",
    "split_groups",
    "train_order_model",
]


# ---------------------------------------------------------------------------
# §A.4 — the group-based split
# ---------------------------------------------------------------------------


def split_groups(
    group_ids: Sequence[str],
    cfg: TrainingConfig,
    *,
    rng: random.Random,
) -> dict[str, tuple[str, ...]]:
    """Partition knowledge groups into ``train`` / ``val`` / ``test``.

    The paper reports no split, but a *per-hyperedge* split would be actively wrong here: §A.2 step 4 links
    hyperedges across horizons and §A.4 samples negatives "cross-group", and both make hyperedges of one
    ``(storm, port)`` scenario depend on each other. Splitting inside a group would put a hyperedge's
    neighbours in the training set and its own evaluation in the test set — leaky, and flattering to every
    method equally. Groups are therefore the atomic unit, per ``training.split.strategy``.

    Args:
        group_ids: All groups in the corpus. Order is irrelevant; the shattering is seeded from ``rng``.
        cfg: The ``training`` config section, supplying the fractions and the strategy name.
        rng: Seeded RNG, so the split is reproducible from the config's ``seed`` alone.

    Returns:
        ``{"train": (...), "val": (...), "test": (...)}``. Sizes follow the configured fractions rounded
        down, and the test split takes the remainder, so the union is always exactly ``group_ids``.

    Raises:
        ValueError: If ``cfg.split.strategy`` is not ``"by_knowledge_group"``.
    """
    if cfg.split.strategy != "by_knowledge_group":
        raise ValueError(
            f"unsupported training.split.strategy {cfg.split.strategy!r}; "
            "only 'by_knowledge_group' is implemented (see configs/base.yaml)"
        )

    shuffled = sorted(group_ids)
    rng.shuffle(shuffled)
    total = len(shuffled)

    n_train = int(math.floor(total * cfg.split.train))
    n_val = int(math.floor(total * cfg.split.val))
    # Test takes whatever is left, so the three parts always sum to the corpus even when the fractions do
    # not divide evenly. Rounding the other way would silently drop groups.
    return {
        "train": tuple(shuffled[:n_train]),
        "val": tuple(shuffled[n_train : n_train + n_val]),
        "test": tuple(shuffled[n_train + n_val :]),
    }


# ---------------------------------------------------------------------------
# §3.1 Eq. 4 — the objective
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LossBreakdown:
    """Eq. 4 split into its parts, so a run can report *which* term moved.

    Attributes:
        total: The scalar optimised.
        positive_term: ``E_P[−log P_θ]`` ≥ 0.
        negative_term: ``α · E_N[log P_θ]`` ≤ 0, *before* the sign convention is applied. Reported
            unsigned-by-convention so the two variants are directly comparable.
        sign: ``+1.0`` for ``literal``, ``-1.0`` for ``minus``.
        mean_positive_logprob: ``E_P[log P_θ] ≤ 0``. This is the quantity to watch: it must rise towards
            0 as training succeeds, and a run whose loss falls while this stays flat is being optimised
            by the negative term alone — which is what the printed Eq. 4 would do.
    """

    total: float
    positive_term: float
    negative_term: float
    sign: float
    mean_positive_logprob: float

    def as_dict(self) -> dict[str, float]:
        """Plain-dict view for logging."""
        return {
            "total": self.total,
            "positive_term": self.positive_term,
            "negative_term": self.negative_term,
            "sign": self.sign,
            "mean_positive_logprob": self.mean_positive_logprob,
        }


def _sign_of(cfg: TrainingConfig) -> float:
    """``+1.0`` for Eq. 4 as printed, ``-1.0`` for the corrected contrastive form."""
    return 1.0 if cfg.negative_term_sign == "literal" else -1.0


def contrastive_loss(
    model: LearnedTransitionModel,
    source: torch.Tensor,
    target: torch.Tensor,
    negatives: torch.Tensor,
    *,
    cfg: TrainingConfig,
) -> tuple[torch.Tensor, LossBreakdown]:
    """§3.1 Eq. 4 with §A.4's sampled softmax.

    Args:
        model: The transition model.
        source: ``(B, d)`` embeddings of ``e_i``.
        target: ``(B, d)`` embeddings of the positive ``e_j``.
        negatives: ``(B, K, d)`` embeddings of ``e_m``.
        cfg: The ``training`` config section; ``alpha`` and ``negative_term_sign`` are used.

    Returns:
        ``(loss, breakdown)``. ``loss`` is a scalar tensor with grad; ``breakdown`` carries the detached
        values for logging.
    """
    positive_logprob, negative_logprob = model.log_prob_sampled_with_negatives(
        source, target, negatives
    )

    positive_term = -positive_logprob.mean()
    negative_term = float(cfg.alpha) * negative_logprob.mean()
    loss = positive_term + _sign_of(cfg) * negative_term

    with torch.no_grad():
        breakdown = LossBreakdown(
            total=float(loss.detach()),
            positive_term=float(positive_term.detach()),
            negative_term=float(negative_term.detach()),
            sign=_sign_of(cfg),
            mean_positive_logprob=float(positive_logprob.mean().detach()),
        )
    return loss, breakdown


# ---------------------------------------------------------------------------
# §A.4 — the training run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainReport:
    """Everything a run manifest needs to explain a training result.

    Attributes:
        epochs_run: Epochs actually executed, which is fewer than ``training.epochs`` when early
            stopping fired.
        best_epoch: The 1-based epoch whose validation loss was lowest; its weights are the ones kept.
        best_val_loss: That epoch's validation loss.
        train_losses: Per-epoch mean training loss.
        val_losses: Per-epoch mean validation loss.
        positives: Size of the positive set ``|P|`` the model was trained on.
        positives_by_signal: Per-§A.4-signal counts, so a run can show which signal dominated.
        split_sizes: Group counts per split.
        negative_sampler: The sampler's diagnostics, including whether it had to fall back.
        early_stopped: Whether training halted before ``training.epochs``.
        best_mean_positive_logprob: ``E_P[log P_θ]`` at the best epoch, on validation data.
    """

    epochs_run: int
    best_epoch: int
    best_val_loss: float
    train_losses: tuple[float, ...]
    val_losses: tuple[float, ...]
    positives: int
    positives_by_signal: Mapping[str, int]
    split_sizes: Mapping[str, int]
    negative_sampler: Mapping[str, Any]
    early_stopped: bool
    best_mean_positive_logprob: float

    def as_dict(self) -> dict[str, Any]:
        """Plain-dict view for the run manifest."""
        return {
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "train_losses": list(self.train_losses),
            "val_losses": list(self.val_losses),
            "positives": self.positives,
            "positives_by_signal": dict(self.positives_by_signal),
            "split_sizes": dict(self.split_sizes),
            "negative_sampler": dict(self.negative_sampler),
            "early_stopped": self.early_stopped,
            "best_mean_positive_logprob": self.best_mean_positive_logprob,
        }


@dataclass
class TrainOutcome:
    """The trained model plus its report."""

    model: LearnedTransitionModel
    report: TrainReport
    history: list[LossBreakdown] = field(default_factory=list)


def _batch_embeddings(
    embeddings: HyperedgeEmbeddings,
    ids: Sequence[str],
) -> torch.Tensor:
    """Stack embeddings for ``ids`` into a ``(len(ids), d)`` tensor, for batching.

    Raises:
        KeyError: If an id has no embedding — meaning the embeddings were built from a different
            hypergraph than the pairs. Worth failing on: silently skipping a pair would shrink ``E_P``
            unevenly across signals.
    """
    return torch.from_numpy(embeddings.rows(ids).copy())


def _mean_loss(
    model: LearnedTransitionModel,
    pairs: Sequence[OrderedPair],
    embeddings: HyperedgeEmbeddings,
    sampler: NegativeSampler,
    cfg: TrainingConfig,
    *,
    batch_size: int,
    shuffle: bool,
    rng: random.Random,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float, list[LossBreakdown]]:
    """Evaluate (and optionally step on) one epoch.

    Args:
        model: The model, in train or eval mode.
        pairs: The epoch's ordered pairs.
        embeddings: Embeddings indexed by hyperedge id.
        sampler: The negative sampler. For validation it should be a sampler built on the *same*
            training pool but with its own RNG, so validation loss is comparable across epochs and does
            not influence training.
        cfg: The ``training`` config section.
        batch_size: Positive pairs per step.
        shuffle: Whether to shuffle the epoch. ``False`` for validation, so its loss is deterministic.
        rng: The RNG for shuffling.
        optimizer: When given, ``backward`` and ``step`` are performed on every batch.

    Returns:
        ``(mean_loss, mean_positive_logprob, breakdowns)``.
    """
    order = list(range(len(pairs)))
    if shuffle:
        rng.shuffle(order)

    total_loss = 0.0
    total_pos_logprob = 0.0
    batches = 0
    breakdowns: list[LossBreakdown] = []

    for start in range(0, len(order), batch_size):
        chunk = [pairs[i] for i in order[start : start + batch_size]]
        if not chunk:
            continue

        source = _batch_embeddings(embeddings, [p.source_id for p in chunk])
        target = _batch_embeddings(embeddings, [p.target_id for p in chunk])
        negative_ids = sampler.sample([p.source_id for p in chunk])
        width = min(len(row) for row in negative_ids)
        if width == 0:
            continue
        negative = _batch_embeddings(
            embeddings, [negative_ids[b][k] for b in range(len(chunk)) for k in range(width)]
        ).reshape(len(chunk), width, -1)

        loss, breakdown = contrastive_loss(model, source, target, negative, cfg=cfg)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            optimizer.step()

        total_loss += breakdown.total
        total_pos_logprob += breakdown.mean_positive_logprob
        batches += 1
        breakdowns.append(breakdown)

    if batches == 0:
        return float("nan"), float("nan"), breakdowns
    return total_loss / batches, total_pos_logprob / batches, breakdowns


def _build_optimizer(model: LearnedTransitionModel, cfg: TrainingConfig) -> torch.optim.Optimizer:
    """Instantiate ``training.optimizer``.

    Raises:
        ValueError: For an unrecognised optimizer name, rather than silently falling back to AdamW — a
            fallback would make two runs comparable that are not.
    """
    kwargs: dict[str, Any] = {"lr": cfg.lr}
    name = cfg.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            betas=tuple(cfg.betas),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            **kwargs,
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            betas=tuple(cfg.betas),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
            **kwargs,
        )
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), weight_decay=cfg.weight_decay, **kwargs)
    raise ValueError(f"unsupported training.optimizer {cfg.optimizer!r}; expected adamw, adam, or sgd")


def train_order_model(
    hypergraph: KnowledgeHypergraph,
    ordering: Mapping[str, Sequence[str]],
    embeddings: HyperedgeEmbeddings,
    *,
    cfg: Config,
    rng: random.Random | None = None,
    extra_positives: Sequence[OrderedPair] = (),
    model: LearnedTransitionModel | None = None,
) -> TrainOutcome:
    """Train ``P_θ`` on the §A.4 self-supervised signals.

    Args:
        hypergraph: The full corpus. Split into train/val/test here by knowledge group.
        ordering: ``group_id -> ordered hyperedge ids`` from §A.3.
        embeddings: §A.4 embeddings for every hyperedge of ``hypergraph``.
        cfg: The root config.
        rng: Optional RNG. When omitted, one is seeded from ``cfg.seed`` via
            :func:`~okh_rag.seeding.derive_seed`, so the run is reproducible from config alone.
        extra_positives: §A.4's retrieval-induced positives ``P_ret``. Ignored unless
            ``training.signals.retrieval_induced`` is set and the pairs are tagged with the
            ``retrieval_induced`` signal. Passing them in is what makes a second self-training round
            possible without this function ever calling the retriever itself.
        model: Optional pre-built model, e.g. resumed from a checkpoint. Its architecture must match
            ``cfg.order_model``.

    Returns:
        A :class:`TrainOutcome` whose model holds the best-validation-loss weights, not the last epoch's.

    Raises:
        ValueError: If the corpus has no positives, which means either the ordering was not built or every
            group was empty. Training on an empty ``E_P`` would report a falling loss driven entirely by
            the negative term.
    """
    if rng is None:
        rng = random.Random(cfg.seed)
    set_global_seed(cfg.seed)

    split = split_groups(tuple(hypergraph.groups), cfg.training, rng=rng)
    train_graph = hypergraph.subset(split["train"])
    val_graph = hypergraph.subset(split["val"])

    signals = cfg.training.signals
    effective_extra = tuple(
        p for p in extra_positives if signals.retrieval_induced and p.signal == SIGNAL_RETRIEVAL_INDUCED
    )
    train_pairs = _with_extra(
        build_training_pairs(train_graph, ordering, cfg.training), effective_extra
    )
    val_pairs = build_training_pairs(val_graph, ordering, cfg.training)

    if not train_pairs.positives:
        raise ValueError(
            "no positive ordered pairs: the §A.3 canonical ordering is missing for every training group, "
            f"or all {len(split['train'])} training groups are empty. "
            f"(groups_without_order={len(train_pairs.groups_missing_order)})"
        )

    if model is None:
        model = LearnedTransitionModel(cfg.order_model)
    if model.rank != cfg.order_model.rank or model.embedding_dim != cfg.order_model.embedding_dim:
        raise ValueError(
            "supplied model does not match cfg.order_model: "
            f"model rank={model.rank} dim={model.embedding_dim}, "
            f"config rank={cfg.order_model.rank} dim={cfg.order_model.embedding_dim}"
        )

    train_sampler = NegativeSampler(
        train_graph,
        k=cfg.training.sampled_softmax_negatives,
        rng=random.Random(derive_seed(cfg.seed, "negatives", "train")),
    )
    # Validation draws its negatives from the *training* pool as well: the val loss measures how well the
    # model ranks held-out positives against the negatives it was trained to separate them from. The
    # sampler has its own derived seed, so the measurement does not depend on training batch order.
    val_sampler = NegativeSampler(
        train_graph,
        k=cfg.training.sampled_softmax_negatives,
        rng=random.Random(derive_seed(cfg.seed, "negatives", "val")),
    )

    optimizer = _build_optimizer(model, cfg.training)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val_loss = float("inf")
    best_epoch = 0
    best_pos_logprob = float("nan")
    epochs_without_improvement = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    history: list[LossBreakdown] = []

    epochs = max(1, cfg.training.epochs)
    batch_size = max(1, cfg.training.batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, _, breakdowns = _mean_loss(
            model,
            train_pairs.positives,
            embeddings,
            train_sampler,
            cfg.training,
            batch_size=batch_size,
            shuffle=True,
            rng=rng,
            optimizer=optimizer,
        )
        history.extend(breakdowns)

        val_loss, val_pos_logprob, _ = _mean_loss(
            model,
            val_pairs.positives,
            embeddings,
            val_sampler,
            cfg.training,
            batch_size=batch_size,
            shuffle=False,
            rng=rng,
            optimizer=None,
        )
        # A corpus with no usable validation groups still trains; fall back to the training loss so early
        # stopping has a signal rather than never firing.
        monitored = val_loss if not math.isnan(val_loss) else train_loss

        train_losses.append(train_loss)
        val_losses.append(monitored)

        if math.isnan(monitored):
            # Only reachable if every batch of the epoch was skipped. Treating NaN as an improvement would
            # make the "best" checkpoint one that was never evaluated.
            continue

        if monitored < best_val_loss - 1e-9:
            best_val_loss = monitored
            best_epoch = epoch
            best_pos_logprob = val_pos_logprob
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if cfg.training.early_stop_patience > 0 and (
                epochs_without_improvement >= cfg.training.early_stop_patience
            ):
                break

    model.load_state_dict(best_state)
    model.eval()

    report = TrainReport(
        epochs_run=len(train_losses),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        train_losses=tuple(train_losses),
        val_losses=tuple(val_losses),
        positives=len(train_pairs.positives),
        positives_by_signal=train_pairs.by_signal(),
        split_sizes={name: len(ids) for name, ids in split.items()},
        negative_sampler={**train_sampler.stats(), "val_sampler": val_sampler.stats()},
        early_stopped=len(train_losses) < epochs,
        best_mean_positive_logprob=best_pos_logprob,
    )
    return TrainOutcome(model=model, report=report, history=history)


def _with_extra(pairs: TrainingPairs, extra: Sequence[OrderedPair]) -> TrainingPairs:
    """Fold the retrieval-induced positives into an existing pair set, deduplicating on ``(source, target)``.

    Existing pairs win, so a pair that both the document-order signal and a retrieval trace produced is
    counted once and keeps its original signal tag — otherwise ``P_ret`` would inflate the documented
    per-signal counts without adding a new training example.
    """
    if not extra:
        return pairs
    merged: dict[tuple[str, str], OrderedPair] = {
        (p.source_id, p.target_id): p for p in pairs.positives
    }
    for pair in extra:
        merged.setdefault((pair.source_id, pair.target_id), pair)
    return TrainingPairs(
        positives=tuple(merged.values()),
        groups_used=pairs.groups_used,
        groups_missing_order=pairs.groups_missing_order,
    )


def summarise_report(report: TrainReport) -> str:
    """One-paragraph human-readable summary, for the end of a training run."""
    lines = [
        f"trained {report.epochs_run} epoch(s); best epoch {report.best_epoch} "
        f"with loss {report.best_val_loss:.6f}"
        + (" (early stopped)" if report.early_stopped else ""),
        f"positives |P| = {report.positives} {dict(report.positives_by_signal)}",
        f"split groups {dict(report.split_sizes)}",
        f"E_P[log P_theta] at best epoch = {report.best_mean_positive_logprob:.6f}",
    ]
    sampler = report.negative_sampler
    if sampler.get("same_group_draws"):
        lines.append(
            f"WARNING: {sampler['same_group_draws']} negative draws fell back to same-group "
            "(the corpus needs more than one knowledge group for §A.4's cross-group negatives)"
        )
    if sampler.get("short_batches"):
        lines.append(
            f"WARNING: {sampler['short_batches']} batches could not supply "
            f"K={sampler.get('k')} distinct negatives; the negative term is averaged over fewer pairs"
        )
    return "\n".join(lines)


def history_matrix(history: Sequence[LossBreakdown]) -> np.ndarray:
    """Stack a run's per-batch breakdowns into an ``(n, 5)`` array, for plotting the loss curves."""
    if not history:
        return np.zeros((0, 5), dtype=np.float64)
    keys = ("total", "positive_term", "negative_term", "sign", "mean_positive_logprob")
    return np.array([[b.as_dict()[k] for k in keys] for b in history], dtype=np.float64)
