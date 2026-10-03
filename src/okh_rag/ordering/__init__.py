"""§3.1 / §A.3-§A.4 — the order-learning subsystem.

Consolidates everything that turns the unordered hypergraph ``H`` into the learned asymmetric transition
model ``P_θ(e_j | e_i)`` of §3.1 Eq. 3, plus the §A.4 self-supervision that trains it:

* :mod:`~okh_rag.ordering.precedence` — §A.3 precedence construction and the canonical trajectory
* :mod:`~okh_rag.ordering.model` — §3.1 Eq. 3's learned transition model (the §A.4 parameter count)
* :mod:`~okh_rag.ordering.pairs` — §A.4's three positive signals and the cross-group negative sampler
* :mod:`~okh_rag.ordering.train` — §3.1 Eq. 4's objective and the training loop
"""

from okh_rag.ordering.model import (
    PAPER_PARAMETER_COUNT,
    LearnedTransitionModel,
)
from okh_rag.ordering.pairs import (
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
from okh_rag.ordering.precedence import (
    PrecedenceGraph,
    build_precedence,
    canonical_order,
    phase_sequence,
    precedence_sort_key,
    trajectory_order_index,
)
from okh_rag.ordering.train import (
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
    "PAPER_PARAMETER_COUNT",
    "SIGNAL_DOCUMENT_ORDER",
    "SIGNAL_ENTITY_OVERLAP",
    "SIGNAL_NAMES",
    "SIGNAL_RETRIEVAL_INDUCED",
    "LearnedTransitionModel",
    "LossBreakdown",
    "NegativeSampler",
    "OrderedPair",
    "PrecedenceGraph",
    "TrainOutcome",
    "TrainReport",
    "TrainingPairs",
    "build_precedence",
    "build_training_pairs",
    "canonical_order",
    "contrastive_loss",
    "history_matrix",
    "pair_entity_overlap",
    "phase_sequence",
    "precedence_sort_key",
    "split_groups",
    "summarise_report",
    "train_order_model",
    "trajectory_order_index",
]
