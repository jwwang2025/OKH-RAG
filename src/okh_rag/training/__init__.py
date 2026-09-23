"""§A.4 — self-supervised training of the order-aware transition model.

* :mod:`~okh_rag.training.pairs` — §A.4's three positive signals and the cross-group negative sampler
* :mod:`~okh_rag.training.train` — §3.1 Eq. 4's objective and the training loop

The model itself lives in :mod:`okh_rag.order_model`, because it is defined in §3.1 alongside Eq. 3 rather
than as part of §A.4's data construction.
"""

from okh_rag.training.pairs import (
    SIGNAL_DOCUMENT_ORDER,
    SIGNAL_ENTITY_OVERLAP,
    SIGNAL_NAMES,
    SIGNAL_RETRIEVAL_INDUCED,
    NegativeSampler,
    OrderedPair,
    TrainingPairs,
    build_training_pairs,
    pair_entity_overlap,
)
from okh_rag.training.train import (
    LossBreakdown,
    TrainOutcome,
    TrainReport,
    contrastive_loss,
    history_matrix,
    split_groups,
    summarise_report,
    train_order_model,
)

__all__ = [
    "SIGNAL_DOCUMENT_ORDER",
    "SIGNAL_ENTITY_OVERLAP",
    "SIGNAL_NAMES",
    "SIGNAL_RETRIEVAL_INDUCED",
    "LossBreakdown",
    "NegativeSampler",
    "OrderedPair",
    "TrainOutcome",
    "TrainReport",
    "TrainingPairs",
    "build_training_pairs",
    "contrastive_loss",
    "history_matrix",
    "pair_entity_overlap",
    "split_groups",
    "summarise_report",
    "train_order_model",
]
