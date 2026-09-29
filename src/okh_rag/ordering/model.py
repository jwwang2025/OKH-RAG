"""§3.1 Eq. 3 / §A.4 — the learned asymmetric transition model ``P_θ(e_j | e_i)``.

§3.1 Eq. 3 defines the model in full:

    P_θ(e_j | e_i) = exp(h_{e_i}^T W h_{e_j}) / Σ_{e'} exp(h_{e_i}^T W h_{e'}),   W = U^T V

and §3.1 adds the reason the form was chosen: "The bilinear form is inherently asymmetric" — the
direction of a transition changes its score, which is the whole point of an order-aware retriever.

§A.4 fixes the shapes and the parameter count:

    "W = U^T V with U, V ∈ R^{64×1536} ... yields 196,608 parameters versus ∼2.4M for the full matrix"

so this module implements exactly that: two ``(64, 1536)`` factors, no bias term (a bias would be an
extra parameter the count does not include), no non-linearity, no symmetrisation.

**Two denominators, and why both exist.** §3.1's Eq. 3 normalises over *all* of ``E``; §A.4 says the
training objective uses an approximation instead:

    "The partition function is approximated via sampled softmax with K = 64 global negatives per example"

These are different quantities under different regimes, so the class exposes them separately rather
than hiding the choice:

* :meth:`~LearnedTransitionModel.log_prob_sampled` — Eq. 3 with a positive and ``K`` negatives in the
  denominator. Used by training, because the full denominator over a corpus of 10^5 hyperedges is
  neither affordable per gradient step nor what §A.4 describes.
* :meth:`~LearnedTransitionModel.log_prob_full` — Eq. 3 exactly, over a caller-supplied candidate set.
  Used by retrieval, where §A.6's ``Trans(γ) = Σ log P_θ`` is stated with the equation's own
  denominator. Random sampling at retrieval time would make the Table 2 ablation rows depend on the
  RNG, so the exact form is used there even though it costs one matrix product against ``|E|``.

Scope note: the Table 2 *Transition* group ("no order" / "heuristic order" / "learned order") also needs
an order-free and a rules-only scorer. Those consume the §A.3 precedence graph rather than embeddings,
so they live in the retrieval package next to the graph instead of here, keeping every scorer in this
module a pure ``embeddings in, log-probabilities out`` function.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import torch
import torch.nn as nn

from okh_rag.config import OrderModelConfig

__all__ = [
    "PAPER_PARAMETER_COUNT",
    "LearnedTransitionModel",
    "TransitionScorer",
]

PAPER_PARAMETER_COUNT = 196_608
"""§A.4 — "yields 196,608 parameters versus ∼2.4M for the full matrix".

The check ``2 · rank · embedding_dim == 196_608`` is asserted when config loads, and re-checked here
against the live tensors so a hand-built model that bypassed config cannot quietly violate the claim.
"""


@runtime_checkable
class TransitionScorer(Protocol):
    """What §A.6's ``Trans(γ)`` term needs from an order model.

    Both methods are on the protocol because the two callers genuinely need different denominators —
    see the module docstring. A scorer that has only one of them is not usable by the pipeline.
    """

    def log_prob_sampled(
        self,
        src: torch.Tensor,
        targets: torch.Tensor,
        negatives: torch.Tensor,
        *,
        negative_log_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``log P_θ(target | src)`` under the §A.4 sampled-softmax denominator."""
        ...

    def log_prob_sampled_with_negatives(
        self,
        src: torch.Tensor,
        targets: torch.Tensor,
        negatives: torch.Tensor,
        *,
        negative_log_q: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Positive and negative log-probabilities under one shared denominator."""
        ...

    def log_prob_full(
        self,
        src: torch.Tensor,
        candidates: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``log P_θ(· | src)`` under Eq. 3's full denominator, or the selected log-probs."""
        ...


class LearnedTransitionModel(nn.Module):
    """§3.1 Eq. 3 — ``P_θ(e_j | e_i) ∝ exp((U h_{e_i}) · (V h_{e_j}))``.

    Attributes:
        U: The source factor, ``(rank, dim)``.
        V: The target factor, ``(rank, dim)``. Separating the factors is what makes the score
            asymmetric: ``U h_i · V h_j`` need not equal ``U h_j · V h_i``.
    """

    def __init__(self, cfg: OrderModelConfig, *, generator: torch.Generator | None = None) -> None:
        """Build the model and initialise ``U`` and ``V`` per ``cfg.init``.

        Args:
            cfg: The ``order_model`` config section, supplying ``rank``, ``embedding_dim``, ``init``
                and ``l2_normalize_embeddings``.
            generator: Optional RNG so initialisation is reproducible independently of global torch
                state. Pass one when comparing two configurations from the same starting point.

        Raises:
            ValueError: If ``2 · rank · embedding_dim`` differs from §A.4's published parameter count.
        """
        super().__init__()
        if 2 * cfg.rank * cfg.embedding_dim != PAPER_PARAMETER_COUNT:
            raise ValueError(
                f"order_model rank x embedding_dim must reproduce §A.4's parameter count: "
                f"2 * {cfg.rank} * {cfg.embedding_dim} = "
                f"{2 * cfg.rank * cfg.embedding_dim}, expected {PAPER_PARAMETER_COUNT}"
            )

        self.rank = cfg.rank
        self.embedding_dim = cfg.embedding_dim
        self.init_scheme = cfg.init
        self.l2_normalize_embeddings = cfg.l2_normalize_embeddings

        self.U = nn.Parameter(torch.empty(cfg.rank, cfg.embedding_dim))
        self.V = nn.Parameter(torch.empty(cfg.rank, cfg.embedding_dim))
        self.reset_parameters(generator=generator)

    # -- initialisation ----------------------------------------------------

    def reset_parameters(self, *, generator: torch.Generator | None = None) -> None:
        """(Re-)initialise ``U`` and ``V`` according to ``init_scheme``.

        The default ``xavier_u_zero_v`` sets ``V = 0``. With ``V = 0`` every logit is 0, so ``P_θ``
        starts exactly uniform — which removes the arbitrary tie the first training step would otherwise
        have to break. The gradients are not degenerate: ``∂ logit/∂V = (U h_i) h_j^T`` is non-zero
        because ``U`` is Xavier, so ``V`` moves on the first step and ``U`` follows from the second.
        The alternative, Xavier on both, starts with random preferences that the first few steps mostly
        undo.
        """
        with torch.no_grad():
            if self.init_scheme == "xavier_u_zero_v":
                self._xavier_(self.U, generator)
                self.V.zero_()
            elif self.init_scheme == "xavier":
                self._xavier_(self.U, generator)
                self._xavier_(self.V, generator)
            elif self.init_scheme == "normal":
                for factor in (self.U, self.V):
                    factor.normal_(mean=0.0, std=0.02, generator=generator)
            else:
                raise ValueError(
                    f"unknown order_model.init {self.init_scheme!r}; "
                    "expected 'xavier_u_zero_v', 'xavier', or 'normal'"
                )

    @staticmethod
    def _xavier_(tensor: torch.Tensor, generator: torch.Generator | None) -> None:
        """Xavier-uniform fill, respecting ``generator`` when one is supplied."""
        fan_in, fan_out = tensor.shape[1], tensor.shape[0]
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        with torch.no_grad():
            tensor.uniform_(-bound, bound, generator=generator)

    # -- §A.4 property -----------------------------------------------------

    @property
    def num_parameters(self) -> int:
        """Live parameter count, to be compared against §A.4's ``196,608``."""
        return int(self.U.numel() + self.V.numel())

    @property
    def matches_paper_parameter_count(self) -> bool:
        """Whether the live tensors still reproduce §A.4's published count."""
        return self.num_parameters == PAPER_PARAMETER_COUNT

    # -- scoring -----------------------------------------------------------

    def _prepare(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Optionally L2-normalise before projecting, per ``cfg.l2_normalize_embeddings``."""
        if not self.l2_normalize_embeddings:
            return embeddings
        return embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def score_matrix(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """``(U h_i) · (V h_j)`` for every source/target combination.

        Args:
            src: ``(B, dim)`` source embeddings.
            dst: ``(M, dim)`` target embeddings.

        Returns:
            A ``(B, M)`` matrix of unnormalised logits. ``M`` is ``|E|`` for Eq. 3's full denominator
            and ``K`` for the sampled one.
        """
        src_proj = self._prepare(src) @ self.U.t()
        dst_proj = self._prepare(dst) @ self.V.t()
        return src_proj @ dst_proj.t()

    def pair_logits(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Logits for aligned pairs.

        Args:
            src: ``(B, dim)``.
            dst: ``(B, dim)``.

        Returns:
            A ``(B,)`` tensor. This is the diagonal of :meth:`score_matrix`, computed without
            materialising the ``(B, B)`` matrix.
        """
        src_proj = self._prepare(src) @ self.U.t()
        dst_proj = self._prepare(dst) @ self.V.t()
        return (src_proj * dst_proj).sum(dim=-1)

    def log_prob_full(
        self,
        src: torch.Tensor,
        candidates: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Eq. 3 exactly — log-softmax ``(U h_i) · (V h_e')`` over a supplied candidate set.

        Args:
            src: ``(B, dim)`` source embeddings.
            candidates: ``(M, dim)`` candidate embeddings. Pass every hyperedge embedding to compute
                Eq. 3 literally; pass a per-query candidate pool to compute it over that pool.
            targets: Optional ``(B,)`` row indices into ``candidates`` selecting one candidate per
                source. When omitted the full ``(B, M)`` log-probability matrix is returned.

        Returns:
            ``(B,)`` if ``targets`` is given, else ``(B, M)``. Every value is ``log P_θ ≤ 0``.
        """
        logits = self.score_matrix(src, candidates)
        log_probs = torch.log_softmax(logits, dim=-1)
        if targets is None:
            return log_probs
        return log_probs.gather(1, targets.reshape(-1, 1)).squeeze(1)

    def log_prob_sampled(
        self,
        src: torch.Tensor,
        targets: torch.Tensor,
        negatives: torch.Tensor,
        *,
        negative_log_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Eq. 3 with §A.4's sampled denominator: the positive plus ``K`` negatives.

        Thin wrapper over :meth:`log_prob_sampled_with_negatives` that discards the negative side, so
        the two never compute different denominators.

        §A.4 — "The partition function is approximated via sampled softmax with K = 64 global negatives
        per example." The denominator is therefore ``exp(logit_pos) + Σ_k exp(logit_neg_k)``, not
        ``Σ_{e'∈E}``.

        §A.4 says only "approximated", so ``cfg.sampled_softmax_correction`` is ``false`` by default and
        no ``log Q`` term is applied. That default is also the mathematically neutral one *for this
        sampler*: negatives are drawn uniformly from a fixed global pool, so ``Q`` is the same constant
        for every candidate and subtracting ``log Q`` shifts every logit equally — a softmax is
        invariant to it. The correction only bites for a non-uniform sampler, which is why
        ``negative_log_q`` is accepted at all: it is where such a sampler would plug in.

        Args:
            src: ``(B, dim)`` source embeddings.
            targets: ``(B, dim)`` positive targets, aligned with ``src``.
            negatives: ``(B, K, dim)`` negatives, aligned per source. Sampled negative *pairs*, i.e.
                ``(source, negative)``, are what Eq. 4's ``E_N`` averages over, so the alignment matters.
            negative_log_q: Optional ``(B, K)`` sampling log-probabilities, subtracted from the negative
                logits before normalisation.

        Returns:
            A ``(B,)`` tensor of ``log P_θ(target | src)``.
        """
        positive, _ = self.log_prob_sampled_with_negatives(
            src, targets, negatives, negative_log_q=negative_log_q
        )
        return positive

    def log_prob_sampled_with_negatives(
        self,
        src: torch.Tensor,
        targets: torch.Tensor,
        negatives: torch.Tensor,
        *,
        negative_log_q: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both sides of the sampled softmax, sharing one denominator.

        Eq. 4's two terms must be scored against the *same* normalisation. If the negative term used a
        denominator built without the positive — or vice versa — the two expectations would be over
        different distributions and ``α`` would be uninterpretable. Computing them together is the only
        way to guarantee that, so the loss calls this method rather than two separate ones.

        Returns:
            ``(log_prob_positive, log_prob_negatives)`` with shapes ``(B,)`` and ``(B, K)``.
        """
        if negatives.ndim != 3:
            raise ValueError(f"negatives must be (B, K, dim), got shape {tuple(negatives.shape)}")
        if negatives.shape[0] != src.shape[0]:
            raise ValueError(
                f"negatives has {negatives.shape[0]} sources for {src.shape[0]} inputs; "
                "each source needs its own negatives"
            )

        batch, k, _ = negatives.shape
        logit_pos = self.pair_logits(src, targets).unsqueeze(1)  # (B, 1)
        # Each source is scored against *its own* K negatives, not against every negative in the batch.
        # Expanding the sources to (B*K, d) and taking elementwise logits keeps that alignment, which a
        # (B, B*K) score matrix would break: it would silently mix other examples' negatives into the
        # denominator and make the sampled softmax depend on batch composition.
        flat_neg = negatives.reshape(batch * k, self.embedding_dim)
        src_expanded = src.repeat_interleave(k, dim=0)  # (B*K, d), source repeated K times
        logit_neg = self.pair_logits(src_expanded, flat_neg).reshape(batch, k)  # (B, K)

        if negative_log_q is not None:
            logit_neg = logit_neg - negative_log_q.reshape(batch, k)

        logits = torch.cat([logit_pos, logit_neg], dim=1)  # (B, 1 + K)
        log_denominator = torch.logsumexp(logits, dim=1, keepdim=True)  # (B, 1)
        return (logit_pos - log_denominator).squeeze(1), logit_neg - log_denominator

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the weights and the shapes they were built with to ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "rank": self.rank,
                "embedding_dim": self.embedding_dim,
                "init": self.init_scheme,
                "l2_normalize_embeddings": self.l2_normalize_embeddings,
                "num_parameters": self.num_parameters,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, *, cfg: OrderModelConfig | None = None) -> "LearnedTransitionModel":
        """Rebuild a model from :meth:`save` output.

        Args:
            path: The checkpoint written by :meth:`save`.
            cfg: Optional config to build the architecture from. When omitted, the shapes and flags
                recorded in the checkpoint are used, so a checkpoint is self-describing and does not
                silently inherit a changed config.

        Raises:
            ValueError: If ``cfg`` is given and disagrees with the checkpoint's architecture.
        """
        payload: Mapping[str, Any] = torch.load(Path(path), map_location="cpu", weights_only=True)
        recorded = OrderModelConfig(
            embedding_dim=int(payload["embedding_dim"]),
            rank=int(payload["rank"]),
            l2_normalize_embeddings=bool(payload["l2_normalize_embeddings"]),
            init=str(payload["init"]),
        )
        if cfg is not None and (
            cfg.rank != recorded.rank
            or cfg.embedding_dim != recorded.embedding_dim
            or cfg.l2_normalize_embeddings != recorded.l2_normalize_embeddings
        ):
            raise ValueError(
                "checkpoint architecture does not match the supplied config: "
                f"checkpoint rank={recorded.rank} dim={recorded.embedding_dim} "
                f"l2={recorded.l2_normalize_embeddings}, "
                f"config rank={cfg.rank} dim={cfg.embedding_dim} l2={cfg.l2_normalize_embeddings}"
            )
        model = cls(recorded)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    # -- diagnostics -------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Architecture summary for the run manifest."""
        return {
            **asdict(
                OrderModelConfig(
                    embedding_dim=self.embedding_dim,
                    rank=self.rank,
                    l2_normalize_embeddings=self.l2_normalize_embeddings,
                    init=self.init_scheme,
                )
            ),
            "num_parameters": self.num_parameters,
            "paper_parameter_count": PAPER_PARAMETER_COUNT,
            "matches_paper": self.matches_paper_parameter_count,
        }
