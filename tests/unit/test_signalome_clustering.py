"""Unit tests for :mod:`alphaphos.signalome.clustering`.

Covers the four building blocks: score preconditioning, Ward tree
building, cut-tree label extraction, and module-count selection with
the scale-aware ``scoring_mode`` backends.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from alphaphos.signalome import (
    ClusterCandidateScore,
    SignalomeClusteringResult,
    build_ward_tree,
    cluster_sites,
    cut_labels,
    precondition_scores,
    resolve_scoring_mode,
    select_module_count,
    summarize_profile_degeneracy,
)


def _make_block_matrix(n_sites=90, n_kinases=9, block_boost=3.0, seed=0):
    """3-block synthetic prediction matrix: sites 0..30 hit kinases 0..2, etc."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(0, 0.3, size=(n_sites, n_kinases))
    b1, b2 = n_sites // 3, 2 * n_sites // 3
    k1, k2 = n_kinases // 3, 2 * n_kinases // 3
    scores[:b1, :k1] += block_boost
    scores[b1:b2, k1:k2] += block_boost
    scores[b2:, k2:] += block_boost
    return scores


# ---------------------------------------------------------------------------
# precondition_scores
# ---------------------------------------------------------------------------


class TestPreconditionScores:
    def test_fills_nan_with_column_median(self):
        m = np.array([[1.0, np.nan, 5.0], [3.0, 2.0, np.nan], [np.nan, 4.0, 7.0]])
        out = precondition_scores(m)
        # column 0 median (of finites 1, 3) = 2
        assert out[2, 0] == 2.0
        # column 1 median (of finites 2, 4) = 3
        assert out[0, 1] == 3.0
        # column 2 median (of finites 5, 7) = 6
        assert out[1, 2] == 6.0
        # existing values unchanged
        assert out[0, 0] == 1.0

    def test_empty_column_becomes_zero(self):
        m = np.array([[1.0, np.nan], [2.0, np.nan]])
        out = precondition_scores(m)
        assert (out[:, 1] == 0.0).all()

    def test_no_nan_input_unchanged(self):
        m = np.arange(6, dtype=float).reshape(2, 3)
        out = precondition_scores(m)
        np.testing.assert_array_equal(out, m)

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D"):
            precondition_scores(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# build_ward_tree + cut_labels
# ---------------------------------------------------------------------------


class TestWardTree:
    def test_linkage_matrix_shape(self):
        m = _make_block_matrix(n_sites=30, n_kinases=6)
        tree = build_ward_tree(m)
        # scipy linkage: (n-1, 4) for n items
        assert tree.shape == (29, 4)

    def test_empty_input(self):
        tree = build_ward_tree(np.zeros((0, 5)))
        assert tree.shape == (0, 4)

    def test_single_site(self):
        tree = build_ward_tree(np.zeros((1, 5)))
        assert tree.shape == (0, 4)

    def test_deterministic(self):
        m = _make_block_matrix(seed=42)
        assert np.array_equal(build_ward_tree(m), build_ward_tree(m))


class TestCutLabels:
    def test_recovers_3_blocks(self):
        m = _make_block_matrix(n_sites=90, n_kinases=9)
        tree = build_ward_tree(m)
        labels = cut_labels(tree, n_clusters=3, n_sites=90)
        # First block should be one cluster, second another, third another.
        assert len(set(labels[:30])) == 1
        assert len(set(labels[30:60])) == 1
        assert len(set(labels[60:90])) == 1
        assert len(set(labels)) == 3

    def test_k_equals_1(self):
        m = _make_block_matrix(n_sites=30, n_kinases=6)
        tree = build_ward_tree(m)
        labels = cut_labels(tree, n_clusters=1, n_sites=30)
        assert (labels == 0).all()

    def test_k_equals_n(self):
        m = _make_block_matrix(n_sites=10, n_kinases=6)
        tree = build_ward_tree(m)
        labels = cut_labels(tree, n_clusters=10, n_sites=10)
        assert len(set(labels)) == 10

    def test_canonical_label_order(self):
        m = _make_block_matrix(n_sites=60, n_kinases=6)
        tree = build_ward_tree(m)
        labels = cut_labels(tree, n_clusters=3, n_sites=60)
        # Row 0 always gets label 0; label 1 is first non-0 encountered, etc.
        assert labels[0] == 0
        first_1 = int(np.argmax(labels == 1))
        first_2 = int(np.argmax(labels == 2))
        assert first_1 < first_2

    def test_out_of_range_k_raises(self):
        m = _make_block_matrix(n_sites=10, n_kinases=6)
        tree = build_ward_tree(m)
        with pytest.raises(ValueError, match="n_clusters"):
            cut_labels(tree, n_clusters=11, n_sites=10)


# ---------------------------------------------------------------------------
# resolve_scoring_mode + auto behaviour
# ---------------------------------------------------------------------------


class TestResolveScoringMode:
    def test_explicit_exact(self):
        assert resolve_scoring_mode("exact", n_sites=10_000, max_exact_sites=5000) == "exact"

    def test_explicit_sampled(self):
        assert resolve_scoring_mode("sampled", n_sites=100, max_exact_sites=5000) == "sampled"

    def test_auto_small_data_is_exact(self):
        assert resolve_scoring_mode("auto", n_sites=100, max_exact_sites=5000) == "exact"

    def test_auto_large_data_falls_back_to_sampled_with_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = resolve_scoring_mode("auto", n_sites=10_000, max_exact_sites=5000)
        assert got == "sampled"
        assert any(issubclass(w.category, UserWarning) for w in caught)

    def test_bad_mode_raises(self):
        with pytest.raises(ValueError, match="scoring_mode"):
            resolve_scoring_mode("bogus", n_sites=100, max_exact_sites=5000)


# ---------------------------------------------------------------------------
# summarize_profile_degeneracy
# ---------------------------------------------------------------------------


class TestProfileDegeneracy:
    def test_flags_constant_rows(self):
        m = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
        mask = summarize_profile_degeneracy(m)
        assert mask[0]  # constant
        assert not mask[1]  # varying
        assert mask[2]  # constant


# ---------------------------------------------------------------------------
# select_module_count -- selection rule matches PhosPy
# ---------------------------------------------------------------------------


class TestModuleCountSelection:
    def test_requested_module_count_takes_precedence(self):
        m = _make_block_matrix(n_sites=60, n_kinases=9)
        tree = build_ward_tree(m)
        res = select_module_count(m, tree, requested_module_count=4)
        assert res.module_count == 4
        assert res.selection_reason == "requested"
        assert 4 in res.candidate_scores

    def test_primary_rule_min_median_and_max_mean(self):
        # Well-separated 3-block synthetic -- every k >= 3 satisfies min-median
        # threshold; PhosPy's rule picks the k that maximises mean-median-corr.
        m = _make_block_matrix(n_sites=90, n_kinases=9, block_boost=5.0)
        tree = build_ward_tree(m)
        res = select_module_count(
            m, tree, max_modules=10, primary_threshold=0.5, scoring_mode="exact"
        )
        assert res.selection_reason == "primary"
        # With such strong separation, the finest partition (max_modules or feasibility)
        # should be picked.
        assert res.module_count >= 3

    def test_fallback_when_primary_fails(self):
        # All-random noise: no k should pass primary=0.99; fallback should apply.
        rng = np.random.default_rng(0)
        m = rng.normal(0, 1, size=(60, 8))
        tree = build_ward_tree(m)
        res = select_module_count(
            m, tree, primary_threshold=0.99, fallback_threshold=0.01, scoring_mode="exact"
        )
        assert res.selection_reason in {"fallback", "max_default"}

    def test_scoring_mode_recorded(self):
        m = _make_block_matrix(n_sites=60, n_kinases=6)
        tree = build_ward_tree(m)
        res = select_module_count(m, tree, scoring_mode="exact")
        assert res.scoring_mode_used == "exact"
        res2 = select_module_count(m, tree, scoring_mode="sampled", max_samples_per_cluster=20)
        assert res2.scoring_mode_used == "sampled"

    def test_candidate_scores_have_new_fields(self):
        m = _make_block_matrix(n_sites=60, n_kinases=6)
        tree = build_ward_tree(m)
        res = select_module_count(m, tree, scoring_mode="exact", max_modules=5)
        for score in res.candidate_scores.values():
            assert isinstance(score, ClusterCandidateScore)
            assert isinstance(score.min_median_correlation, float)
            assert isinstance(score.mean_median_correlation, float)


# ---------------------------------------------------------------------------
# cluster_sites (combined entry point)
# ---------------------------------------------------------------------------


class TestClusterSitesEntryPoint:
    def test_returns_1_indexed_labels(self):
        m = pd.DataFrame(_make_block_matrix(n_sites=90, n_kinases=9))
        res = cluster_sites(m, requested_module_count=3)
        assert isinstance(res, SignalomeClusteringResult)
        # 0 is reserved for unassigned; labels start at 1.
        assert res.labels.min() >= 1
        assert res.labels.max() == 3

    def test_provenance_recorded(self):
        m = pd.DataFrame(_make_block_matrix(n_sites=60, n_kinases=6))
        res = cluster_sites(m, requested_module_count=3, seed=42)
        for key in (
            "n_sites",
            "n_kinases",
            "module_count",
            "selection_reason",
            "scoring_mode_used",
            "seed",
            "linkage_method",
            "distance_metric",
        ):
            assert key in res.provenance
        assert res.provenance["linkage_method"] == "ward"
        assert res.provenance["distance_metric"] == "euclidean"
        assert res.provenance["seed"] == 42
