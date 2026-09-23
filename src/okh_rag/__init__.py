"""OKH-RAG — Order-Aware Knowledge Hypergraph Retrieval-Augmented Generation.

Reproduction of:

    Keshu Wu, Chenchen Kuai, Zihao Li, Jiwan Jiang, Shiyu Shen, Shian Wang,
    Chan-Wei Hu, Zhengzhong Tu, Yang Zhou.
    "Knowledge Is Not Static: Order-Aware Hypergraph RAG for Language Models."
    arXiv:2604.12185, 2026.

Module map (paper section -> module):

    §3.1, §A.1   representation + relation vocabulary  ->  okh_rag.types, okh_rag.vocabulary
    §A.2         hypergraph construction               ->  okh_rag.extraction
    §A.3         precedence construction               ->  okh_rag.precedence
    §3.1 Eq. 3   learned transition model              ->  okh_rag.order.transition
    §3.1 Eq. 4   contrastive training                  ->  okh_rag.order.train, okh_rag.order.signals
    §3.2 Eq. 5   retrieval objective                   ->  okh_rag.retrieval.scoring
    §A.7 Alg. 1  order-aware beam search               ->  okh_rag.retrieval.beam_search
    §A.7 Eq.14-15 Viterbi DP                           ->  okh_rag.retrieval.viterbi
    §3.3, §A.8   retrieval-augmented generation        ->  okh_rag.generation
    §4.1         baselines and metrics                 ->  okh_rag.baselines, okh_rag.evaluation
    §4.5         ablation switches                     ->  okh_rag.evaluation.harness
"""

__version__ = "0.1.0"
__paper_arxiv_id__ = "2604.12185"

__all__ = ["__version__", "__paper_arxiv_id__"]
