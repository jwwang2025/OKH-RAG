"""Deterministic seeding across the three RNGs the pipeline touches.

``configs/base.yaml`` carries a single ``seed`` and promises it is "applied to ``random``, ``numpy``,
``torch``". A seed that only reaches one of them is worse than no seed: the run looks reproducible and
is not, because §A.4's negative sampling draws from :mod:`random` while the transition model's
initialisation draws from :mod:`torch`.

The paper does not specify a seed. This module exists so that the three are always set together and so
that the split, the negative sampler and the model init all derive from that one value rather than from
each other's side effects.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch

__all__ = ["derive_seed", "set_global_seed"]


def set_global_seed(seed: int) -> None:
    """Seed :mod:`random`, :mod:`numpy` and :mod:`torch` from one value.

    ``torch.use_deterministic_algorithms(True)`` is deliberately *not* enabled: on CPU it can turn a
    supported op into a hard error, and the bilinear form of §3.1 Eq. 3 has no nondeterministic
    component worth guarding. Seeding is enough for the reproducibility the ablation table needs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def derive_seed(base: int, *labels: str) -> int:
    """Derive a stable sub-seed from a base seed and some labels.

    Different stages need independent randomness — if the split and the negative sampler drew from one
    stream, changing the split fraction would change the negatives and two configurations would no
    longer be comparable. Deriving per-stage seeds from ``base`` keeps that independence while leaving a
    single number to record in the run manifest.

    Args:
        base: The value of ``seed`` in ``configs/base.yaml``.
        *labels: Stage names, e.g. ``"split"``, ``"negatives"``, ``"init"``.

    Returns:
        A 63-bit non-negative integer, stable across processes and Python versions (unlike ``hash``).
    """
    payload = "|".join((str(base), *labels)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") >> 1
