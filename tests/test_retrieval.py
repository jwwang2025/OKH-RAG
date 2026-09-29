"""Tests for §3.2 / §A.6–A.7 — order-aware retrieval.

The suite is organised by the equation or algorithm each group of assertions belongs to, because that is
the unit a reader has to be able to check against the paper:

* §A.6 ``Rel``      — :class:`TestCosineRelevance`
* §A.6 Eq. 11 ``Prec`` — :class:`TestPrecedenceConsistency`
* §A.6 Eq. 12 ``Ovlp`` — :class:`TestEntityContinuity`
* §A.6 Eq. 13 ``Cov``  — :class:`TestPhaseCoverage`
* §A.6's "designed for efficient incremental computation" constraint — :class:`TestIncrementalScoring`
* §A.7 candidate scoping — :class:`TestCandidateScoping`
* §A.7 Algorithm 1 — :class:`TestBeamSearch`
* §A.7 Eq. 14–15 — :class:`TestViterbi`
* §3.1 Eq. 3's learned table — :class:`TestTransitionMatrices`
* Table 2 switches — :class:`TestAblationSwitches`

Everything runs on a hand-built nine-hyperedge corpus so the arithmetic can be checked by hand. Where a
property is easier to establish on randomly shaped data (telescoping), the test generates trajectories and
asserts the identity rather than a value.

No test needs an API key, torch, or the CyPortQA release, except :class:`TestLearnedTransitionMatrix`,
which is skipped when torch is absent.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from okh_rag.config import (
    GROUP_RESERVE_MODES,
    INFERENCE_MODES,
    OVERLAP_MODES,
    PRECEDENCE_MODES,
    TRANSITION_VARIANTS,
    RetrievalConfig,
    RetrievalWeights,
    load_config,
)
from okh_rag.embeddings import HyperedgeEmbeddings
from okh_rag.ordering.precedence import PrecedenceGraph
from okh_rag.retrieval import (
    CandidateScores,
    HypergraphIndex,
    TrajectoryScorer,
    cosine_relevance,
    heuristic_transition_matrix,
    order_aware_beam_search,
    resolve_weights,
    scope_candidates,
    unreachable_phase_set,
    viterbi_search,
    zero_transition_matrix,
)
from okh_rag.schema import Entity, Hyperedge, KnowledgeHypergraph, sum_scores
from okh_rag.vocabulary import COVERAGE_PHASES

REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Fixtures — a nine-hyperedge corpus with hand-checkable overlaps
# ---------------------------------------------------------------------------

# (id, entity ids, relation, group). Entity ids are shared on purpose so the Eq. 12 Jaccard values are
# exact fractions rather than whatever random vectors happen to produce.
HYPEREDGE_SPECS = (
    ("eA", ("s1", "s2"), "has_warning_status", "g1"),            # advisory
    ("eB", ("s2", "s3"), "forecasts_hazard_at_horizon", "g1"),   # hazard_forecast
    ("eC", ("s3", "s4"), "has_operation_status", "g1"),          # operation_status
    ("eD", ("s4", "s5"), "has_impact_prediction", "g1"),         # impact
    ("eE", ("s5", "s1"), "has_recovery_status", "g1"),           # recovery
    ("eF", ("t1", "t2"), "has_cyclone_state", "g1"),             # structural: not in S
    ("eG", ("u1", "u2"), "has_warning_status", "g2"),
    ("eH", ("u2", "u3"), "has_operation_status", "g2"),
    ("eI", ("u3", "u4"), "has_impact_prediction", "g2"),
)

# A partial order over g1 that is deliberately not a total one: eF (structural) is incomparable with
# everything, and eE is comparable with eD only. §A.6 Eq. 11's "unrelated pairs are excluded from the
# denominator" only means something if unrelated pairs exist.
PRECEDENCE_EDGES = (
    ("eA", "eB"),
    ("eB", "eC"),
    ("eC", "eD"),
    ("eD", "eE"),
)

HORIZON_OF = {"eA": 48, "eB": 48, "eC": 48, "eD": 48, "eE": 48, "eF": 48, "eG": 24, "eH": 24, "eI": 24}


@pytest.fixture(scope="module")
def cfg():
    """The real ``configs/base.yaml``.

    Loaded rather than constructed so the tests run against the shipping hyperparameters — the §4.1
    weights, B = 8, L = 8, N = 3, K = 80, |C| = 150. A test that passes only under a bespoke config would
    not be evidence about the pipeline that will actually be run.
    """
    return load_config(REPO_ROOT / "configs" / "base.yaml")


@pytest.fixture(scope="module")
def tiny() -> KnowledgeHypergraph:
    """The hand-built corpus."""
    hypergraph = KnowledgeHypergraph()
    for hyperedge_id, entity_ids, relation, group_id in HYPEREDGE_SPECS:
        for entity_id in entity_ids:
            hypergraph.add_entity(Entity(entity_id, f"entity {entity_id}", "PORT", "d", 1.0))
        hypergraph.add_hyperedge(
            Hyperedge(
                hyperedge_id=hyperedge_id,
                entity_ids=entity_ids,
                relation=relation,
                evidence=f"evidence for {hyperedge_id}",
                attributes={"k": hyperedge_id},
                group_id=group_id,
                horizon_hours=HORIZON_OF[hyperedge_id],
                text_position=HYPEREDGE_SPECS.index(
                    (hyperedge_id, entity_ids, relation, group_id)
                ),
            )
        )
    return hypergraph


@pytest.fixture(scope="module")
def precedence() -> PrecedenceGraph:
    """``≺`` for :func:`tiny`, built by hand so the denominators in Eq. 11 are known exactly."""
    graph = PrecedenceGraph()
    for src, dst in PRECEDENCE_EDGES:
        graph.rule_of[(src, dst)] = "within_horizon_phase"
    return graph


@pytest.fixture
def identity_embeddings():
    """Embeddings whose rows are one-hot, so ``Rel(e_i, q)`` is a chosen coordinate of ``q``.

    Returns:
        A factory ``make(hyperedge_ids, query) -> HyperedgeEmbeddings`` where the relevance of the *i*-th id
        is exactly ``query[i] / ||query||``.
    """

    def make(hyperedge_ids, query=None) -> HyperedgeEmbeddings:
        n = len(hyperedge_ids)
        matrix = np.eye(n, dtype=np.float32)
        return HyperedgeEmbeddings(hyperedge_ids=tuple(hyperedge_ids), matrix=matrix)

    return make


def overridden(cfg, **cfg_overrides):
    """The ``retrieval`` config a search call needs in order to match a ``make_pool(**)`` override.

    ``make_pool`` applies its overrides to the *scorer* only. ``beam_width``, ``beam_length``,
    ``beam_initial_multiplier``, ``num_trajectories`` and ``diversity_penalty`` are read from the config
    handed to ``order_aware_beam_search`` / ``viterbi_search``, so a test that overrides one of those in
    ``make_pool`` and then passes ``cfg.retrieval`` silently exercises the shipped default instead of its
    own override — the search still runs and still returns trajectories, so nothing fails loudly.

    Returns:
        ``cfg.retrieval`` with ``cfg_overrides`` applied, or ``cfg.retrieval`` unchanged.
    """
    return replace(cfg.retrieval, **cfg_overrides) if cfg_overrides else cfg.retrieval


def make_pool(hypergraph, precedence, cfg, weights=None, variant="heuristic", ids=None, **cfg_overrides):
    """Build a scorer over the tiny corpus, with relevance fixed by the hyperedge id order.

    Args:
        hypergraph: The corpus.
        precedence: ``≺``.
        cfg: The root config.
        weights: Optional weight override.
        variant: Which transition table to materialise.
        ids: Optional subset of the corpus to build the pool from. Needed to test behaviour that only
            appears when the pool is smaller than a beam width or an initial multiplier — on the full
            nine-hyperedge fixture, ``2B = 16 > |C|``, so the seed beam is the whole pool and none of
            those effects are observable.
        **cfg_overrides: Applied to ``cfg.retrieval`` via :func:`dataclasses.replace` — to the **scorer**
            only. A search-side key (``beam_width``, ``beam_length``, ``num_trajectories``,
            ``diversity_penalty``) must also be routed through :func:`overridden` at the call site.

    Returns:
        ``(scorer, scores)``.
    """
    retrieval_cfg = overridden(cfg, **cfg_overrides)
    ids = tuple(sorted(hypergraph.hyperedges)) if ids is None else tuple(ids)
    relevance = np.linspace(1.0, 0.1, len(ids))
    if variant == "heuristic":
        transitions = heuristic_transition_matrix(precedence, ids)
    elif variant == "none":
        transitions = zero_transition_matrix(len(ids))
    else:
        raise ValueError(variant)
    scores = CandidateScores(
        hyperedge_ids=ids, relevance=relevance, transitions=transitions, variant=variant
    )
    resolved = weights if weights is not None else retrieval_cfg.weights
    scorer = TrajectoryScorer(
        hypergraph=hypergraph,
        precedence=precedence,
        scores=scores,
        cfg=retrieval_cfg,
        weights=resolved,
    )
    return scorer, scores


def positions_of(scores: CandidateScores, *hyperedge_ids: str) -> tuple[int, ...]:
    """Pool positions of the named hyperedges, in the order given."""
    index = scores.index
    return tuple(index[hid] for hid in hyperedge_ids)


# ---------------------------------------------------------------------------
# §A.6 Relevance
# ---------------------------------------------------------------------------


class TestCosineRelevance:
    """``Rel(e, q) = cos(h_e, q)``."""

    def test_parallel_vectors_score_one(self):
        assert cosine_relevance(np.array([2.0, 0.0]), np.array([[5.0, 0.0]]))[0] == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert cosine_relevance(np.array([1.0, 0.0]), np.array([[0.0, 3.0]]))[0] == pytest.approx(0.0)

    def test_magnitude_does_not_matter(self):
        """A cosine, not a dot product — otherwise a long evidence string would outrank a relevant one."""
        small = cosine_relevance(np.array([1.0, 1.0]), np.array([[1.0, 1.0]]))[0]
        large = cosine_relevance(np.array([1000.0, 1000.0]), np.array([[1.0, 1.0]]))[0]
        assert small == pytest.approx(large) == pytest.approx(1.0)

    def test_zero_norm_row_is_zero_not_nan(self):
        result = cosine_relevance(np.array([1.0, 0.0]), np.array([[0.0, 0.0], [1.0, 0.0]]))
        assert not np.isnan(result).any()
        assert result[0] == 0.0
        assert result[1] == pytest.approx(1.0)

    def test_zero_norm_query_is_zero_not_nan(self):
        result = cosine_relevance(np.array([0.0, 0.0]), np.array([[1.0, 0.0]]))
        assert result[0] == 0.0 and not np.isnan(result).any()

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimensions"):
            cosine_relevance(np.array([1.0, 0.0, 0.0]), np.array([[1.0, 0.0]]))

    def test_non_matrix_raises(self):
        with pytest.raises(ValueError, match="2-d"):
            cosine_relevance(np.array([1.0, 0.0]), np.array([1.0, 0.0]))


# ---------------------------------------------------------------------------
# §A.6 Eq. 11 Precedence consistency
# ---------------------------------------------------------------------------


class TestPrecedenceConsistency:
    """``Prec(γ) = |{k : e_(k) ≺ e_(k+1)}| / |{k : (e_(k), e_(k+1)) ∈ (≺ ∪ ≻)}|``."""

    def test_fully_consistent_chain_is_one(self, tiny, precedence, cfg):
        scorer, scores = make_pool(tiny, precedence, cfg)
        assert scorer.precedence_consistency(positions_of(scores, "eA", "eB", "eC", "eD")) == pytest.approx(1.0)

    def test_reversed_chain_is_zero(self, tiny, precedence, cfg):
        scorer, scores = make_pool(tiny, precedence, cfg)
        assert scorer.precedence_consistency(positions_of(scores, "eD", "eC", "eB", "eA")) == pytest.approx(0.0)

    def test_incomparable_pairs_leave_the_denominator(self, tiny, precedence, cfg):
        """Eq. 11 — "unrelated pairs are excluded from the denominator".

        ``eF`` is comparable with nothing, so ``(eB, eF)`` reaches neither numerator nor denominator.
        The trajectory therefore has exactly one comparable pair, ``(eA, eB)``, and it is consistent:
        the fraction is ``1/1``. Counting the unrelated pair in the denominator would give ``1/2``, so the
        value itself is the evidence that the exclusion happened.
        """
        scorer, scores = make_pool(tiny, precedence, cfg)
        assert not precedence.related("eA", "eF")
        assert not precedence.related("eB", "eF")
        assert scorer.precedence_consistency(positions_of(scores, "eA", "eB", "eF")) == pytest.approx(1.0)

    def test_empty_denominator_is_reported_as_zero(self, tiny, precedence, cfg):
        """Eq. 11 leaves ``0/0`` open; the recorded choice is ``0.0``.

        ``Prec`` is a reward in ``[0, 1]`` scaled by ``µ ≥ 0``. Returning ``1.0`` would hand the whole µ
        term to a trajectory whose pairs are *all* incomparable — i.e. to one that never engages with ≺
        at all, which is the opposite of what the term rewards. ``0.0`` is the "no evidence, no credit"
        reading, matching the empty-sum convention used for ``Ovlp`` and ``Cov``.
        """
        scorer, scores = make_pool(tiny, precedence, cfg)
        for trajectory in (("eA",), ("eA", "eF"), ("eF", "eG"), ("eA", "eF", "eG")):
            assert not any(precedence.related(a, b) for a, b in zip(trajectory, trajectory[1:])), trajectory
            assert scorer.precedence_consistency(positions_of(scores, *trajectory)) == pytest.approx(0.0)

    def test_half_consistent_is_a_half(self, tiny, precedence, cfg):
        scorer, scores = make_pool(tiny, precedence, cfg)
        # eA ≺ eB (consistent); eB ≻ eA (comparable, inconsistent) -> 1/2.
        assert scorer.precedence_consistency(positions_of(scores, "eA", "eB", "eA")) == pytest.approx(0.5)

    def test_single_step_has_no_denominator(self, tiny, precedence, cfg):
        scorer, scores = make_pool(tiny, precedence, cfg)
        assert scorer.precedence_consistency(positions_of(scores, "eA")) == pytest.approx(0.0)

    def test_cross_group_pairs_are_incomparable(self, tiny, precedence, cfg):
        """§A.1 builds ``≺`` "within each knowledge group", so g1 and g2 never compare.

        ``(eB, eG)`` and ``(eG, eH)`` are both cross-group or unrelated, so the only comparable pair is
        ``(eA, eB)`` and the trajectory scores ``1/1`` rather than ``1/3``. Mixing groups is therefore not
        punished by this term — it is simply not rewarded, which is the same treatment any trajectory of
        incomparable pairs gets (see :meth:`test_empty_denominator_is_reported_as_zero`).
        """
        scorer, scores = make_pool(tiny, precedence, cfg)
        assert not precedence.related("eB", "eG")
        assert not precedence.related("eG", "eH")
        assert scorer.precedence_consistency(
            positions_of(scores, "eA", "eB", "eG", "eH")
        ) == pytest.approx(1.0)

    def test_indicator_mode_counts_pairs(self, tiny, precedence, cfg):
        """Algorithm 1 line 6's ``µ · 1[e_(ℓ−1) ≺ e′]`` is a count, not a fraction."""
        scorer, scores = make_pool(tiny, precedence, cfg, precedence_mode="indicator")
        assert scorer.precedence_consistency(positions_of(scores, "eA", "eB", "eC")) == pytest.approx(2.0)
        assert scorer.precedence_consistency(positions_of(scores, "eA", "eB", "eA")) == pytest.approx(1.0)

    def test_modes_are_the_two_recorded_readings(self):
        assert set(PRECEDENCE_MODES) == {"fraction", "indicator"}


# ---------------------------------------------------------------------------
# §A.6 Eq. 12 Entity continuity
# ---------------------------------------------------------------------------


class TestEntityContinuity:
    """``Ovlp(γ) = 1/(L−1) Σ_k |V∩| / |V∪|``."""

    def test_single_pair_is_its_jaccard(self, tiny, precedence, cfg):
        scorer, scores = make_pool(tiny, precedence, cfg)
        # eA={s1,s2}, eB={s2,s3} -> |∩|=1, |∪|=3.
        assert scorer.jaccard(*positions_of(scores, "eA", "eB")) == pytest.approx(1 / 3)
        assert scorer.entity_continuity(positions_of(scores, "eA", "eB")) == pytest.approx(1 / 3)

    def test_mean_over_pairs(self, tiny, precedence, cfg):
        scorer, scores = make_pool(tiny, precedence, cfg)
        chain = positions_of(scores, "eA", "eB", "eC", "eD")
        expected = (1 / 3 + 1 / 3 + 1 / 3) / 3
        assert scorer.entity_continuity(chain) == pytest.approx(expected)

    def test_disjoint_sets_score_zero(self, tiny, precedence, cfg):
        scorer, scores = make_pool(tiny, precedence, cfg)
        assert scorer.entity_continuity(positions_of(scores, "eA", "eF")) == pytest.approx(0.0)

    def test_one_step_has_no_pair(self, tiny, precedence, cfg):
        scorer, scores = make_pool(tiny, precedence, cfg)
        assert scorer.entity_continuity(positions_of(scores, "eA")) == pytest.approx(0.0)

    def test_sum_mode_is_the_algorithm_1_accumulation(self, tiny, precedence, cfg):
        """Algorithm 1 line 7 accumulates ``J`` instead of the mean — a factor of ``L−1``."""
        chain = ("eA", "eB", "eC", "eD")
        mean_scorer, scores = make_pool(tiny, precedence, cfg)
        sum_scorer, _ = make_pool(tiny, precedence, cfg, overlap_mode="sum")
        mean_value = mean_scorer.entity_continuity(positions_of(scores, *chain))
        sum_value = sum_scorer.entity_continuity(positions_of(scores, *chain))
        assert sum_value == pytest.approx(mean_value * (len(chain) - 1))

    def test_modes_are_the_two_recorded_readings(self):
        assert set(OVERLAP_MODES) == {"mean", "sum"}


# ---------------------------------------------------------------------------
# §A.6 Eq. 13 Phase coverage
# ---------------------------------------------------------------------------


class TestPhaseCoverage:
    """``Cov(γ) = |{φ(e_k)} ∩ S| / |S|``."""

    def test_full_span_is_one(self, tiny, precedence, cfg):
        scorer, scores = make_pool(tiny, precedence, cfg)
        # advisory, hazard_forecast, operation_status, impact, recovery -> 5 of the 6 members of S.
        chain = positions_of(scores, "eA", "eB", "eC", "eD", "eE")
        assert scorer.phase_coverage(chain) == pytest.approx(5 / len(cfg.retrieval.phase_set))

    def test_structural_relations_contribute_nothing(self, tiny, precedence, cfg):
        """``has_cyclone_state`` is a precondition, not a reasoning phase; §A.6 never lists it in S."""
        scorer, scores = make_pool(tiny, precedence, cfg)
        assert scorer.phase_coverage(positions_of(scores, "eF")) == pytest.approx(0.0)

    def test_repeats_do_not_double_count(self, tiny, precedence, cfg):
        scorer, scores = make_pool(tiny, precedence, cfg)
        once = scorer.phase_coverage(positions_of(scores, "eA"))
        thrice = scorer.phase_coverage(positions_of(scores, "eA", "eA", "eA"))
        assert once == thrice == pytest.approx(1 / len(cfg.retrieval.phase_set))

    def test_denominator_is_the_config_phase_set(self, tiny, precedence, cfg):
        """``|S|`` comes from config, so a run with a trimmed ``S`` reports a larger fraction."""
        scorer, scores = make_pool(tiny, precedence, cfg, phase_set=["advisory", "impact"])
        chain = positions_of(scores, "eA", "eD")
        assert scorer.phase_coverage(chain) == pytest.approx(1.0)

    def test_configured_phase_set_is_reachable(self, cfg):
        """Every member of the shipping ``S`` must be the image of some relation, or ρ loses range."""
        assert unreachable_phase_set(cfg.retrieval.phase_set) == ()
        assert set(cfg.retrieval.phase_set) == set(COVERAGE_PHASES)


# ---------------------------------------------------------------------------
# §A.6 — "designed for efficient incremental computation during beam search"
# ---------------------------------------------------------------------------


class TestIncrementalScoring:
    """The step deltas must telescope into the whole-trajectory values.

    This is the single most important correctness property in the package. If it fails, the beam search
    optimises a different objective from the one the evaluator reports, and every Table 2 number is
    measuring something other than Eq. 5.
    """

    @pytest.mark.parametrize("precedence_mode", PRECEDENCE_MODES)
    @pytest.mark.parametrize("overlap_mode", OVERLAP_MODES)
    def test_deltas_telescope(self, tiny, precedence, cfg, precedence_mode, overlap_mode):
        scorer, scores = make_pool(
            tiny, precedence, cfg, precedence_mode=precedence_mode, overlap_mode=overlap_mode
        )
        rng = np.random.default_rng(7)
        pool_size = len(scores)

        for _ in range(60):
            length = int(rng.integers(1, pool_size + 1))
            trajectory = tuple(rng.choice(pool_size, size=length, replace=False).tolist())

            accumulated = []
            prefix: tuple[int, ...] = ()
            for position in trajectory:
                accumulated.append(scorer.step_delta(prefix, position))
                prefix = (*prefix, position)

            folded = sum_scores(accumulated)
            whole = scorer.score(trajectory)

            for field in ("relevance", "order_coherence", "precedence_consistency",
                          "entity_continuity", "phase_coverage"):
                assert getattr(folded, field) == pytest.approx(
                    getattr(whole, field), abs=1e-9
                ), f"{field} does not telescope for {trajectory} under {precedence_mode}/{overlap_mode}"

    @pytest.mark.parametrize("precedence_mode", PRECEDENCE_MODES)
    @pytest.mark.parametrize("overlap_mode", OVERLAP_MODES)
    def test_weighted_totals_telescope_too(self, tiny, precedence, cfg, precedence_mode, overlap_mode):
        """Linearity means the component identity implies the weighted one — asserted, not assumed."""
        weights = RetrievalWeights(lambda_=1.2, mu=0.3, nu=0.2, rho=0.5)
        scorer, scores = make_pool(
            tiny,
            precedence,
            cfg,
            weights=weights,
            precedence_mode=precedence_mode,
            overlap_mode=overlap_mode,
        )
        trajectory = positions_of(scores, "eA", "eB", "eC", "eD")
        incremental = 0.0
        prefix: tuple[int, ...] = ()
        for position in trajectory:
            incremental += scorer.step_delta(prefix, position).total(
                weights.lambda_, weights.mu, weights.nu, weights.rho
            )
            prefix = (*prefix, position)
        assert incremental == pytest.approx(scorer.total(trajectory))

    def test_first_step_carries_its_own_phase(self, tiny, precedence, cfg):
        """The singleton's Cov must be seeded, or every trajectory is short one coverage slot."""
        scorer, scores = make_pool(tiny, precedence, cfg)
        first = positions_of(scores, "eA")[0]
        assert scorer.step_delta((), first).phase_coverage == pytest.approx(
            1 / len(cfg.retrieval.phase_set)
        )


# ---------------------------------------------------------------------------
# §A.7 Candidate scoping
# ---------------------------------------------------------------------------


class TestCandidateScoping:
    """Stage 1 top-K cosine, stage 2 group-aware expansion, capped at ``|C|`` with a 40% reserve."""

    def _embeddings_for(self, hypergraph, query):
        ids = tuple(sorted(hypergraph.hyperedges))
        matrix = np.eye(len(ids), dtype=np.float32)
        return HyperedgeEmbeddings(hyperedge_ids=ids, matrix=matrix), np.asarray(query, dtype=np.float64)

    def test_pool_is_capped(self, tiny, cfg):
        """§A.7 — "The pool is capped at |C| = 150", reserve included.

        The shipped config cannot breach the cap (``top_k + reserve = 80 + 60 = 140 <= 150``) and the tiny
        universe cannot fill it, so the shipped-config assertion below is vacuous on its own. The two
        configurations that *can* breach it are exercised explicitly: a reserve longer than the headroom
        the anchors left, and a ``top_k`` that overflows without any reserve at all.
        """
        ids = tuple(sorted(tiny.hyperedges))
        query = np.array([len(ids) - i for i in range(len(ids))], dtype=np.float64)
        embeddings, query = self._embeddings_for(tiny, query)
        index = HypergraphIndex(tiny)

        pool = scope_candidates(tiny, embeddings, query, cfg=cfg.retrieval, group_id="g1")
        assert len(pool) <= cfg.retrieval.candidate_pool_size

        # Regression: the floor reserve used to spend headroom it did not have. `top_k=4` leaves one slot
        # of a five-slot pool, the 40% reserve asks for two, and the pool came back with six members.
        tight = replace(cfg.retrieval, top_k=4, candidate_pool_size=5, group_reserve_fraction=0.4)
        for mode in GROUP_RESERVE_MODES:
            capped = scope_candidates(
                tiny, embeddings, query, cfg=replace(tight, group_reserve_mode=mode),
                index=index, group_id="g2",
            )
            assert len(capped) == 5, mode
            assert len(capped.reserved) <= 2, mode

        # And the guard that keeps the anchors from doing it on their own.
        with pytest.raises(ValueError, match="cap"):
            scope_candidates(
                tiny, embeddings, query,
                cfg=replace(cfg.retrieval, top_k=6, candidate_pool_size=5),
                index=index, group_id="g1",
            )

    def test_pool_size_is_the_whole_universe_when_smaller_than_the_cap(self, tiny, cfg):
        """A nine-hyperedge corpus cannot fill 150 slots; the pool is the universe, not padded."""
        ids = tuple(sorted(tiny.hyperedges))
        embeddings, query = self._embeddings_for(tiny, np.ones(len(ids)))
        pool = scope_candidates(tiny, embeddings, query, cfg=cfg.retrieval, group_id="g1")
        assert set(pool.hyperedge_ids) == set(ids)

    def test_anchors_are_the_top_k_by_relevance(self, tiny, cfg):
        ids = tuple(sorted(tiny.hyperedges))
        # Relevance falls off with position, so the top-K is a known prefix.
        query = np.array([len(ids) - i for i in range(len(ids))], dtype=np.float64)
        embeddings, query = self._embeddings_for(tiny, query)
        narrowed = replace(cfg.retrieval, top_k=4, candidate_pool_size=4)
        pool = scope_candidates(tiny, embeddings, query, cfg=narrowed, group_id="g1")
        assert pool.anchors == ids[:4]
        assert pool.hyperedge_ids == ids[:4]

    def test_group_reserve_is_satisfied(self, tiny, cfg):
        """With |C|=4 and a 50% reserve, two slots must go to the query's own group."""
        ids = tuple(sorted(tiny.hyperedges))
        query = np.array([len(ids) - i for i in range(len(ids))], dtype=np.float64)
        embeddings, query = self._embeddings_for(tiny, query)
        narrowed = replace(cfg.retrieval, top_k=2, candidate_pool_size=4, group_reserve_fraction=0.5)
        pool = scope_candidates(
            tiny, embeddings, query, cfg=narrowed, group_id="g2", index=HypergraphIndex(tiny)
        )
        own_group = set(tiny.groups["g2"])
        assert len(set(pool.hyperedge_ids) & own_group) >= 2
        assert pool.group_id == "g2"

    def test_unknown_group_gets_no_reserve(self, tiny, cfg):
        ids = tuple(sorted(tiny.hyperedges))
        embeddings, query = self._embeddings_for(tiny, np.ones(len(ids)))
        pool = scope_candidates(tiny, embeddings, query, cfg=cfg.retrieval, group_id=None)
        assert pool.group_id is None
        assert pool.reserved == ()

    def test_floor_and_quota_modes_differ(self, tiny, cfg):
        """The two readings of §A.7's reserve must be able to produce different pools.

        They coincide whenever ``top_k + reserve <= |C|``, because then no anchor has to be denied: both
        readings admit every anchor and fill the reserve from the same own-group list, and the equality is
        structural rather than accidental. They can only diverge when the reserve *squeezes* the anchors,
        i.e. ``top_k > |C| - reserve`` — which is what ``top_k=4, |C|=5, reserve=2`` sets up. There
        "the reserve sits inside the 150" (quota) has to drop the fourth anchor, while "the reserve is a
        floor under the 150" (floor) keeps it and instead trims the reserve to the one slot left.
        """
        ids = tuple(sorted(tiny.hyperedges))
        query = np.array([len(ids) - i for i in range(len(ids))], dtype=np.float64)
        embeddings, query = self._embeddings_for(tiny, query)
        base = replace(cfg.retrieval, top_k=4, candidate_pool_size=5, group_reserve_fraction=0.4)
        index = HypergraphIndex(tiny)
        floor_pool = scope_candidates(
            tiny, embeddings, query, cfg=replace(base, group_reserve_mode="floor"),
            index=index, group_id="g2",
        )
        quota_pool = scope_candidates(
            tiny, embeddings, query, cfg=replace(base, group_reserve_mode="quota"),
            index=index, group_id="g2",
        )
        assert floor_pool.hyperedge_ids != quota_pool.hyperedge_ids
        assert len(floor_pool) == len(quota_pool) == base.candidate_pool_size
        # The fourth anchor, and the slot the quota reading spends on the reserve instead.
        assert "eD" in floor_pool.hyperedge_ids and "eD" not in quota_pool.hyperedge_ids
        assert len(floor_pool.reserved) == 1 and len(quota_pool.reserved) == 2
        assert set(GROUP_RESERVE_MODES) == {"floor", "quota"}

    def test_allowed_ids_restrict_the_universe(self, tiny, cfg):
        """The group-split guard: retrieval must not see the groups it is scored on."""
        ids = tuple(sorted(tiny.hyperedges))
        embeddings, query = self._embeddings_for(tiny, np.ones(len(ids)))
        allowed = tuple(tiny.groups["g1"])
        pool = scope_candidates(
            tiny, embeddings, query, cfg=cfg.retrieval, group_id=None, allowed_ids=allowed
        )
        assert set(pool.hyperedge_ids) <= set(allowed)

    def test_empty_universe_yields_empty_pool(self, tiny, cfg):
        ids = tuple(sorted(tiny.hyperedges))
        embeddings, query = self._embeddings_for(tiny, np.ones(len(ids)))
        pool = scope_candidates(tiny, embeddings, query, cfg=cfg.retrieval, allowed_ids=())
        assert len(pool) == 0

    def test_pool_is_deterministic_and_relevance_ordered(self, tiny, cfg):
        ids = tuple(sorted(tiny.hyperedges))
        query = np.array([len(ids) - i for i in range(len(ids))], dtype=np.float64)
        embeddings, query = self._embeddings_for(tiny, query)
        index = HypergraphIndex(tiny)
        first = scope_candidates(tiny, embeddings, query, cfg=cfg.retrieval, index=index, group_id="g1")
        second = scope_candidates(tiny, embeddings, query, cfg=cfg.retrieval, index=index, group_id="g1")
        assert first.hyperedge_ids == second.hyperedge_ids
        assert list(first.relevance) == sorted(first.relevance, reverse=True)

    def test_ties_break_on_hyperedge_id(self, tiny, cfg):
        """Identical relevance must still yield one order, or two runs could retrieve differently."""
        ids = tuple(sorted(tiny.hyperedges))
        embeddings, query = self._embeddings_for(tiny, np.ones(len(ids)))
        pool = scope_candidates(tiny, embeddings, query, cfg=cfg.retrieval, group_id=None)
        # All relevance equal -> the pool must be in lexicographic id order.
        assert pool.hyperedge_ids == tuple(sorted(pool.hyperedge_ids))

    def test_provenance_accounts_for_the_pool(self, tiny, cfg):
        ids = tuple(sorted(tiny.hyperedges))
        embeddings, query = self._embeddings_for(tiny, np.ones(len(ids)))
        pool = scope_candidates(tiny, embeddings, query, cfg=cfg.retrieval, group_id="g1")
        provenance = pool.provenance()
        assert provenance["pool"] == len(pool)
        assert provenance["anchors"] + provenance["group_reserved"] + provenance["expanded"] == len(pool)

    def test_relevance_array_is_aligned_with_ids(self, tiny, cfg):
        ids = tuple(sorted(tiny.hyperedges))
        query = np.array([len(ids) - i for i in range(len(ids))], dtype=np.float64)
        embeddings, query = self._embeddings_for(tiny, query)
        pool = scope_candidates(tiny, embeddings, query, cfg=cfg.retrieval, group_id="g1")
        expected = cosine_relevance(query, embeddings.matrix)
        for row, hyperedge_id in enumerate(pool.hyperedge_ids):
            assert pool.relevance[row] == pytest.approx(expected[embeddings.index[hyperedge_id]])


# ---------------------------------------------------------------------------
# §A.7 Algorithm 1
# ---------------------------------------------------------------------------


class TestBeamSearch:
    """Order-aware beam search: B, L, N, ``C \\ β``, top-2B singletons, the diversity penalty."""

    def test_returns_at_most_n_trajectories(self, tiny, precedence, cfg):
        scorer, _ = make_pool(tiny, precedence, cfg)
        result = order_aware_beam_search(scorer, cfg=cfg.retrieval)
        assert 0 < len(result) <= cfg.retrieval.num_trajectories

    def test_length_is_min_of_l_and_pool(self, tiny, precedence, cfg):
        scorer, scores = make_pool(tiny, precedence, cfg)
        result = order_aware_beam_search(scorer, cfg=cfg.retrieval)
        expected = min(cfg.retrieval.beam_length, len(scores))
        assert all(len(entry) == expected for entry in result.entries)

    def test_length_override_is_honoured(self, tiny, precedence, cfg):
        scorer, _ = make_pool(tiny, precedence, cfg)
        result = order_aware_beam_search(scorer, cfg=cfg.retrieval, length=3)
        assert all(len(entry) == 3 for entry in result.entries)

    def test_no_trajectory_repeats_a_hyperedge(self, tiny, precedence, cfg):
        """Algorithm 1 line 5's ``C \\ β`` — eight distinct pieces of evidence, not eight visits to one."""
        scorer, _ = make_pool(tiny, precedence, cfg)
        result = order_aware_beam_search(scorer, cfg=cfg.retrieval)
        for entry in result.entries:
            assert len(set(entry.positions)) == len(entry.positions)

    def test_entries_are_best_first(self, tiny, precedence, cfg):
        scorer, _ = make_pool(tiny, precedence, cfg)
        result = order_aware_beam_search(scorer, cfg=cfg.retrieval)
        totals = [entry.total for entry in result.entries]
        assert totals == sorted(totals, reverse=True)

    def test_beam_never_exceeds_width(self, tiny, precedence, cfg):
        """Line 10 retains top-B — but line 2 *seeds* top-2B, so only the later steps are B-bounded.

        Asserting ``<= B`` on every entry of ``beam_size_per_step`` contradicts line 2, whose whole point
        is that the initial beam is twice the width (it is pruned back to ``B`` at the end of the first
        extension). :meth:`test_beam_starts_at_the_initial_multiplier` pins the seed size; this test pins
        the bound that line 10 does impose, on every step after it.
        """
        scorer, scores = make_pool(tiny, precedence, cfg)
        result = order_aware_beam_search(scorer, cfg=cfg.retrieval)
        expected_seed = min(
            len(scores), cfg.retrieval.beam_initial_multiplier * cfg.retrieval.beam_width
        )
        assert result.beam_size_per_step[0] == expected_seed
        assert expected_seed > cfg.retrieval.beam_width  # line 2 really is "top-2B", not "top-B"
        assert all(size <= cfg.retrieval.beam_width for size in result.beam_size_per_step[1:])

    def test_beam_starts_at_the_initial_multiplier(self, tiny, precedence, cfg):
        """Line 2 — "top-2B singletons by Rel(e, q)"."""
        scorer, scores = make_pool(tiny, precedence, cfg)
        result = order_aware_beam_search(scorer, cfg=cfg.retrieval)
        expected = min(len(scores), cfg.retrieval.beam_initial_multiplier * cfg.retrieval.beam_width)
        assert result.beam_size_per_step[0] == expected

    def test_singletons_are_the_most_relevant(self, tiny, precedence, cfg):
        """Line 2 — "B ← top-2B singletons by Rel(e, q)".

        Two of its consequences are observable through :class:`BeamSearchResult`:

        * the seed beam's *size* is ``min(|C|, multiplier · B)``, read from ``beam_size_per_step[0]``.
          It cannot be read from ``len(entries)``: line 12 truncates to ``N``, so running with ``L = 1``
          returns the top-``N`` singletons, not the ``2B`` seeds.
        * those ``N`` singletons are the top-``N`` under a singleton's own objective, which is
          ``Rel(e, q) + ρ · Cov([e])`` — a one-step trajectory has no pair, so ``Trans``, ``Prec`` and
          ``Ovlp`` are all zero and only its own phase contributes.

        What this fixture *cannot* establish is the seed beam's ranking **key**. With ``2B = 16 > |C| = 9``
        the seed set is the whole pool, and the relevance gaps (0.1125) exceed the largest coverage bonus
        (``ρ/6 = 0.0833``), so no coverage-heavy singleton can outrank a more relevant one and "sort the
        seeds by ``Rel``" is indistinguishable from "sort by the singleton objective". Distinguishing them
        needs a pool larger than ``2B`` with a relevance ramp flatter than ``ρ/6``.
        """
        scorer, scores = make_pool(tiny, precedence, cfg)
        result = order_aware_beam_search(scorer, cfg=cfg.retrieval)

        expected_seed = min(
            len(scores), cfg.retrieval.beam_initial_multiplier * cfg.retrieval.beam_width
        )
        assert result.beam_size_per_step[0] == expected_seed
        # Line 2 fixes the seed set by relevance, so no returned trajectory can start outside it. Weak
        # here — the seed set is the whole pool — but it is the invariant, and it is what would break if
        # the seeds were drawn from anywhere else.
        by_relevance = sorted(range(len(scores)), key=lambda p: (-float(scores.relevance[p]), p))
        assert {entry.head for entry in result.entries} <= set(by_relevance[:expected_seed])

        single_step = order_aware_beam_search(scorer, cfg=cfg.retrieval, length=1)
        singleton_objective = {
            p: float(scores.relevance[p]) + cfg.retrieval.weights.rho * scorer.phase_coverage([p])
            for p in range(len(scores))
        }
        by_objective = sorted(range(len(scores)), key=lambda p: (-singleton_objective[p], p))
        assert [entry.positions[0] for entry in single_step.entries] == by_objective[
            : cfg.retrieval.num_trajectories
        ]

    def test_report_scores_match_the_scorer(self, tiny, precedence, cfg):
        """The reported breakdown must be Eq. 5's, not the search's accumulated bookkeeping.

        ``order_coherence`` belongs here more than the other three: §A.6 defines it as a *sum* of
        log-transitions and Algorithm 1 line 6 accumulates it one step at a time, so this is the
        assertion that the incremental accumulation and the definitional sum agree. The other three terms
        are not sums, which is why their step deltas are differences of prefix values.
        """
        scorer, _ = make_pool(tiny, precedence, cfg)
        result = order_aware_beam_search(scorer, cfg=cfg.retrieval)
        for entry in result.entries:
            whole = scorer.score(entry.positions)
            assert entry.score.relevance == pytest.approx(whole.relevance)
            assert entry.score.order_coherence == pytest.approx(whole.order_coherence)
            assert entry.score.entity_continuity == pytest.approx(whole.entity_continuity)
            assert entry.score.phase_coverage == pytest.approx(whole.phase_coverage)

    def test_order_free_variant_returns_the_most_relevant(self, tiny, precedence, cfg):
        """§3.2 — "setting all to zero recovers standard top-k retrieval"."""
        weights = RetrievalWeights(0.0, 0.0, 0.0, 0.0)
        scorer, scores = make_pool(tiny, precedence, cfg, weights=weights, variant="none")
        result = order_aware_beam_search(scorer, cfg=cfg.retrieval)
        best = result.best()
        assert best is not None
        expected = sorted(range(len(scores)), key=lambda p: -float(scores.relevance[p]))[
            : cfg.retrieval.beam_length
        ]
        assert list(best.positions) == expected

    def test_diversity_penalty_is_reported(self, tiny, precedence, cfg):
        """``BeamEntry.total`` carries the deduction, and ``penalty`` says how much it carries.

        The pair is asserted as an invariant rather than a value: ``total == score.total(λ,µ,ν,ρ) −
        penalty`` for every returned trajectory. Line 10 runs once per step, so a beam whose head keeps
        colliding is charged repeatedly and ``penalty`` must report the **sum**, not the last charge —
        propagating ``total`` without propagating ``penalty`` leaves the deduction inside the score while
        the report claims the trajectory was clean.

        The pool is cut to four hyperedges because a charged collision needs more retained beams than
        distinct heads: with ``|C| = 9`` and ``B = 8`` the beam can always keep eight distinct heads and no
        deduction is ever taken. ``num_trajectories`` is raised to ``B`` so the whole retained beam is
        returned — the charged entries are the *lowest*-scoring ones, so a top-3 slice would hide them and
        the test would pass by looking away.
        """
        overrides = {"diversity_penalty": 5.0, "num_trajectories": cfg.retrieval.beam_width}
        scorer, _ = make_pool(tiny, precedence, cfg, ids=("eA", "eB", "eC", "eD"), **overrides)
        result = order_aware_beam_search(scorer, cfg=overridden(cfg, **overrides))
        assert len(result.entries) == cfg.retrieval.beam_width
        for entry in result.entries:
            assert entry.total == pytest.approx(
                entry.score.total(
                    cfg.retrieval.weights.lambda_,
                    cfg.retrieval.weights.mu,
                    cfg.retrieval.weights.nu,
                    cfg.retrieval.weights.rho,
                )
                - entry.penalty
            )
        assert any(entry.penalty > 0 for entry in result.entries)

    def test_zero_penalty_imposes_no_deduction(self, tiny, precedence, cfg):
        """``diversity_penalty = 0`` must make line 10 a plain ``sorted(...)[:B]``.

        Note the ``overridden(...)`` call: ``make_pool``'s overrides reach the scorer, and the penalty is
        read by the *search*, so passing ``cfg.retrieval`` here would silently run with the shipped
        ``diversity_penalty`` of 0.1 and test the opposite of what the name claims.
        """
        overrides = {"diversity_penalty": 0.0}
        scorer, _ = make_pool(tiny, precedence, cfg, **overrides)
        result = order_aware_beam_search(scorer, cfg=overridden(cfg, **overrides))
        assert all(entry.penalty == 0.0 for entry in result.entries)
        assert all(
            entry.total
            == pytest.approx(
                entry.score.total(
                    cfg.retrieval.weights.lambda_,
                    cfg.retrieval.weights.mu,
                    cfg.retrieval.weights.nu,
                    cfg.retrieval.weights.rho,
                )
            )
            for entry in result.entries
        )

    def test_empty_pool_returns_empty_result(self, tiny, precedence, cfg):
        scorer, _ = make_pool(tiny, precedence, cfg)
        empty = CandidateScores(
            hyperedge_ids=(),
            relevance=np.zeros(0),
            transitions=np.zeros((0, 0)),
            variant="none",
        )
        empty_scorer = TrajectoryScorer(
            hypergraph=tiny, precedence=precedence, scores=empty, cfg=cfg.retrieval
        )
        result = order_aware_beam_search(empty_scorer, cfg=cfg.retrieval)
        assert len(result) == 0 and result.best() is None

    def test_heuristic_variant_minimises_precedence_violations(self, tiny, precedence, cfg):
        """The heuristic table charges one nat per ≺-violation, and the beam takes the fewest it can.

        §A.6 defines the order-coherence term as ``Trans(γ) = Σ log P_θ``; the heuristic variant puts its
        rule reward in that same slot — ``0.0`` for a ≺-consistent step, ``-1.0`` otherwise. The term is
        therefore zero **only** when a trajectory's every consecutive pair is ≺-consistent.

        That is impossible at ``L = 8`` on this fixture, and not because anything is wrong: ``≺`` here has
        four edges (``eA ≺ eB ≺ eC ≺ eD ≺ eE``), a trajectory may not repeat a hyperedge (Algorithm 1
        line 5), so at most four of its seven steps can be consistent. The score is bounded below by
        ``-(7 − 4) = -3.0`` and the beam attains that bound, i.e. it is optimal on this term. Asserting
        ``0.0`` would require a ``-inf`` table, which ``heuristic_transition_matrix`` deliberately does
        not use — ``-inf`` kills the beam as soon as a hyperedge's ≺-successors are exhausted, and after
        ``eE`` this chain has none.

        The second half of the test therefore uses ``L = 5``, the smallest length at which a fully
        consistent trajectory fits, and requires the beam to reach ``0.0`` there.
        """
        scorer, scores = make_pool(tiny, precedence, cfg)
        result = order_aware_beam_search(scorer, cfg=cfg.retrieval)
        best = result.best()
        assert best is not None

        forced_violations = (len(best) - 1) - len(PRECEDENCE_EDGES)
        assert forced_violations > 0, "the fixture is what makes this test meaningful"
        assert best.score.order_coherence == pytest.approx(-float(forced_violations))
        # The bound is a maximum, not merely a lower limit: every step's reward is <= 0 and at most
        # len(PRECEDENCE_EDGES) steps can be consistent, so no trajectory of this length scores higher.
        assert best.score.order_coherence <= 0.0
        assert best.score.precedence_consistency == pytest.approx(1.0)

        short = order_aware_beam_search(scorer, cfg=cfg.retrieval, length=len(PRECEDENCE_EDGES) + 1)
        short_best = short.best()
        assert short_best is not None
        assert short_best.score.order_coherence == 0.0
        assert short_best.positions == positions_of(scores, "eA", "eB", "eC", "eD", "eE")


# ---------------------------------------------------------------------------
# §A.7 Eq. 14-15 Viterbi
# ---------------------------------------------------------------------------


class TestViterbi:
    """The exact optimiser for ``Rel + λ · Trans``, and its claim to be a ceiling for beam search."""

    def test_matches_brute_force(self, tiny, precedence, cfg):
        """Eq. 14-15 against exhaustive enumeration on a pool small enough to enumerate."""
        weights = RetrievalWeights(lambda_=1.0, mu=0.0, nu=0.0, rho=0.0)
        scorer, scores = make_pool(tiny, precedence, cfg, weights=weights)
        length = 3
        result = viterbi_search(scorer, cfg=cfg.retrieval, length=length)

        best = -math.inf
        from itertools import product

        for path in product(range(len(scores)), repeat=length):
            value = float(scores.relevance[list(path)].sum())
            value += sum(float(scores.transitions[a, b]) for a, b in zip(path, path[1:]))
            best = max(best, value)
        assert result.objective == pytest.approx(best)

    def test_recovered_path_scores_what_it_claims(self, tiny, precedence, cfg):
        scorer, _ = make_pool(tiny, precedence, cfg)
        result = viterbi_search(scorer, cfg=cfg.retrieval, length=4)
        recomputed = scorer.score(result.positions)
        assert result.objective == pytest.approx(
            recomputed.relevance + scorer.weights.lambda_ * recomputed.order_coherence
        )

    def test_is_a_ceiling_for_beam_search(self, tiny, precedence, cfg):
        """§A.7 — "Viterbi ... serves primarily as a quality ceiling for beam search".

        Compared on the two-term objective the DP actually maximises, with µ, ν and ρ zeroed for both.
        Viterbi searches a relaxed space (repeats allowed, which is what its ``O(L·|C|²)`` complexity
        implies), so its optimum can only be at least the beam's best feasible trajectory.
        """
        weights = RetrievalWeights(lambda_=1.2, mu=0.0, nu=0.0, rho=0.0)
        scorer, _ = make_pool(tiny, precedence, cfg, weights=weights)
        length = 5
        exact = viterbi_search(scorer, cfg=cfg.retrieval, length=length)
        beam = order_aware_beam_search(scorer, cfg=cfg.retrieval, length=length)
        best = beam.best()
        assert best is not None
        beam_two_term = best.score.relevance + weights.lambda_ * best.score.order_coherence
        assert exact.objective >= beam_two_term - 1e-9, (exact.objective, beam_two_term)

    def test_single_step_is_the_most_relevant_candidate(self, tiny, precedence, cfg):
        """Eq. 14 — ``dp[0][j] = Rel(e_j, q)``, so L=1 selects the relevance argmax."""
        scorer, scores = make_pool(tiny, precedence, cfg)
        result = viterbi_search(scorer, cfg=cfg.retrieval, length=1)
        assert result.positions == (int(np.argmax(scores.relevance)),)

    def test_empty_pool_is_safe(self, tiny, precedence, cfg):
        empty = CandidateScores(
            hyperedge_ids=(), relevance=np.zeros(0), transitions=np.zeros((0, 0)), variant="none"
        )
        scorer = TrajectoryScorer(
            hypergraph=tiny, precedence=precedence, scores=empty, cfg=cfg.retrieval
        )
        result = viterbi_search(scorer, cfg=cfg.retrieval)
        assert result.positions == () and result.objective == 0.0


# ---------------------------------------------------------------------------
# §3.1 Eq. 3 / Table 2 transition tables
# ---------------------------------------------------------------------------


class TestTransitionMatrices:
    """The three tables, and the switch that selects them."""

    def test_heuristic_rewards_consistency(self, precedence):
        ids = ("eA", "eB", "eC")
        matrix = heuristic_transition_matrix(precedence, ids)
        assert matrix[0, 1] == 0.0  # eA ≺ eB
        assert matrix[1, 0] == -1.0  # eB ≻ eA
        assert matrix[0, 2] == -1.0  # incomparable is not a licence

    def test_heuristic_diagonal_is_forbidden(self, precedence):
        """``e ≺ e`` is false, so a self-step is a violation. Harmless in beam search (``C \\ β``
        excludes it), load-bearing in Viterbi, which permits repeats — there a self-loop is only chosen
        when it beats every real alternative."""
        matrix = heuristic_transition_matrix(precedence, ("eA", "eB"))
        assert matrix[0, 0] == -1.0 and matrix[1, 1] == -1.0

    def test_zero_matrix_is_zero(self):
        assert not zero_transition_matrix(4).any()
        assert zero_transition_matrix(4).shape == (4, 4)

    def test_learned_matrix_is_a_log_softmax(self, tiny, precedence, cfg):
        """Eq. 3's denominator restricted to the pool: every row must sum to 1 in probability space."""
        torch = pytest.importorskip("torch")
        from okh_rag.retrieval import learned_transition_matrix, zero_transition_matrix as _zero  # noqa: F401

        ids = tuple(sorted(tiny.hyperedges))
        rng = np.random.default_rng(3)
        embeddings = HyperedgeEmbeddings(
            hyperedge_ids=ids, matrix=rng.normal(size=(len(ids), 1536)).astype(np.float32)
        )

        class Stub:
            """A ``TransitionScorer`` without a ``torch.nn.Module``, so the test needs no training."""

            def __init__(self, seed=0):
                generator = torch.Generator().manual_seed(seed)
                self.u = torch.randn(64, 1536, generator=generator)
                self.v = torch.randn(64, 1536, generator=generator)

            def log_prob_full(self, src, candidates, *, targets=None):
                logits = (src @ self.u.t()) @ (candidates @ self.v.t()).t()
                return torch.log_softmax(logits, dim=-1)

        matrix = learned_transition_matrix(Stub(), embeddings, ids)
        assert matrix.shape == (len(ids), len(ids))
        assert np.all(matrix <= 1e-9)  # log P <= 0
        assert np.allclose(np.exp(matrix).sum(axis=1), 1.0, atol=1e-5)

    def test_variant_set_is_the_three_table_2_rows(self):
        assert set(TRANSITION_VARIANTS) == {"learned", "heuristic", "none"}


# ---------------------------------------------------------------------------
# Table 2 switches
# ---------------------------------------------------------------------------


class TestAblationSwitches:
    """The ``drop_*`` weights, the order-free collapse, and the NaN trap they create together."""

    def test_none_variant_zeroes_every_weight(self, cfg):
        weights = resolve_weights(cfg, "none")
        assert (weights.lambda_, weights.mu, weights.nu, weights.rho) == (0.0, 0.0, 0.0, 0.0)

    def test_learned_and_heuristic_keep_the_paper_weights(self, cfg):
        for variant in ("learned", "heuristic"):
            weights = resolve_weights(cfg, variant)
            assert (weights.lambda_, weights.mu, weights.nu, weights.rho) == (1.2, 0.3, 0.2, 0.5)

    def test_drop_switches_remove_one_term_each(self, cfg):
        variant = load_config(REPO_ROOT / "configs" / "base.yaml")
        variant.ablation.drop_mu = True
        weights = resolve_weights(variant, "heuristic")
        assert weights.mu == 0.0
        assert (weights.lambda_, weights.nu, weights.rho) == (1.2, 0.2, 0.5)

    def test_zero_weight_times_infinity_is_not_nan(self):
        """Regression: ``TrajectoryScore.total`` must not turn a zero weight into NaN.

        A naive ``λ · order_coherence`` with ``λ = 0`` evaluates ``0.0 · -inf = NaN``, and the ablation then
        silently returns an arbitrary trajectory while reporting a plausible score. No table in
        ``okh_rag.retrieval`` actually produces ``-inf`` — the heuristic table charges a finite ``-1.0``
        per ≺-violation, precisely so the beam is not killed — but the guard has to hold for a score built
        by hand, which is what this constructs.
        """
        from okh_rag.schema import TrajectoryScore

        score = TrajectoryScore(relevance=1.0, order_coherence=float("-inf"), phase_coverage=0.5)
        assert not math.isnan(score.total(0.0, 0.3, 0.2, 0.5))
        assert score.total(0.0, 0.3, 0.2, 0.5) == pytest.approx(1.0 + 0.2 * 0.0 + 0.5 * 0.5)

    def test_drop_lambda_on_a_heuristic_pool_returns_real_numbers(self, tiny, precedence, cfg, tmp_path):
        """Same regression, end to end: the ``−λ`` row of Table 2 must produce a usable ranking."""
        variant = load_config(REPO_ROOT / "configs" / "base.yaml")
        variant.ablation.drop_lambda = True
        weights = resolve_weights(variant, "heuristic")
        scorer, _ = make_pool(tiny, precedence, cfg, weights=weights)
        result = order_aware_beam_search(scorer, cfg=variant.retrieval)
        assert len(result) > 0
        assert all(math.isfinite(entry.total) for entry in result.entries)

    def test_shuffle_preserves_content(self, tiny, precedence, cfg):
        """§4.5's stress test keeps the retrieved set and changes only its order."""
        from okh_rag.schema import Trajectory

        scorer, scores = make_pool(tiny, precedence, cfg)
        best = order_aware_beam_search(scorer, cfg=cfg.retrieval).best()
        assert best is not None
        trajectory = Trajectory(hyperedge_ids=scorer.ids(best.positions))
        shuffled = trajectory.shuffled(cfg.seed)
        assert shuffled.as_set == trajectory.as_set

    def test_inference_modes_are_the_two_algorithms(self):
        assert set(INFERENCE_MODES) == {"beam_search", "viterbi"}


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """The validator must refuse a config that would silently deflate a term."""

    def test_unreachable_phase_is_rejected(self, tmp_path):
        source = (REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8")
        # Anchor on the *key*, not on a list item. `- advisory` also appears in the §A.3 precedence
        # order, which sits earlier in the file, so replacing a bare `"    - advisory\n"` edits that
        # list instead and leaves `phase_set` intact — the validator then has nothing to reject and
        # the test passes for the wrong reason (or, here, fails for the wrong reason).
        broken = source.replace("  phase_set:\n", "  phase_set:\n    - not_a_phase\n", 1)
        assert broken != source
        assert "not_a_phase" in broken
        path = tmp_path / "broken.yaml"
        path.write_text(broken, encoding="utf-8")
        with pytest.raises(ValueError, match="unreachable"):
            load_config(path)

    def test_unknown_overlap_mode_is_rejected(self, tmp_path):
        source = (REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8")
        broken = source.replace('overlap_mode: "mean"', 'overlap_mode: "median"', 1)
        path = tmp_path / "broken.yaml"
        path.write_text(broken, encoding="utf-8")
        with pytest.raises(ValueError, match="overlap_mode"):
            load_config(path)

    def test_top_k_above_the_cap_is_rejected(self, tmp_path):
        source = (REPO_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8")
        broken = source.replace("top_k: 80", "top_k: 400", 1)
        path = tmp_path / "broken.yaml"
        path.write_text(broken, encoding="utf-8")
        with pytest.raises(ValueError, match="candidate_pool_size"):
            load_config(path)

    def test_reports_dir_is_a_path(self, cfg):
        assert isinstance(cfg.paths.reports_dir, Path)


# ---------------------------------------------------------------------------
# §A.7's own numbers, asserted against the config
# ---------------------------------------------------------------------------


def test_shipping_retrieval_hyperparameters_match_the_paper(cfg):
    """§4.1 and §A.7 state these five values; a config edit must not silently change them."""
    retrieval = cfg.retrieval
    assert (retrieval.weights.lambda_, retrieval.weights.mu, retrieval.weights.nu, retrieval.weights.rho) == (
        1.2,
        0.3,
        0.2,
        0.5,
    )  # §4.1
    assert retrieval.top_k == 80  # §A.7
    assert retrieval.candidate_pool_size == 150  # §A.7
    assert retrieval.group_reserve_fraction == 0.40  # §A.7
    assert (retrieval.beam_width, retrieval.beam_length, retrieval.num_trajectories) == (8, 8, 3)  # §4.1
