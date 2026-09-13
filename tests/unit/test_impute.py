"""Tests for alphaphos.preprocess.impute.

Covers:
  - impute_knn_site_based : the legacy-parity site-based KNN imputer
  - impute_hybrid         : per-cell MAR/MNAR hybrid (site-KNN + Gaussian)
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphaphos.constants import LAYER_INTENSITY_LOG2
from alphaphos.preprocess.impute import (
    _check_complete_features,
    impute_hybrid,
    impute_knn_site_based,
)

# Helper -----------------------------------------------------------------


def _make_adata(X: np.ndarray) -> ad.AnnData:
    n_samples, n_sites = X.shape
    X = X.astype(float).copy()
    # Match the shape collapse_sites emits: both .X and the canonical log2
    # layer carry the same values; impute defaults to the layer.
    adata = ad.AnnData(
        X=X.copy(),
        obs=pd.DataFrame(index=[f"s{i}" for i in range(n_samples)]),
        var=pd.DataFrame(index=[f"site{j}" for j in range(n_sites)]),
    )
    adata.layers[LAYER_INTENSITY_LOG2] = X.copy()
    return adata


# ============================================================================
# _check_complete_features
# ============================================================================


class TestCheckCompleteFeatures:
    def test_passes_when_each_column_has_at_least_one_value(self):
        X = np.array([[1.0, 2.0, np.nan], [np.nan, 3.0, 4.0]])
        _check_complete_features(X)  # no error

    def test_errors_on_all_nan_column(self):
        X = np.array([[1.0, np.nan, 3.0], [2.0, np.nan, 4.0]])
        with pytest.raises(ValueError, match="no observed values"):
            _check_complete_features(X)


# ============================================================================
# impute_knn_site_based
# ============================================================================


class TestImputeKnnSiteBased:
    def test_fills_all_missing_cells(self):
        # 4 samples x 5 sites with some random missingness
        rng = np.random.default_rng(0)
        X = rng.normal(10, 1, size=(4, 5))
        X[0, 0] = np.nan
        X[2, 3] = np.nan
        adata = _make_adata(X)

        impute_knn_site_based(adata)
        assert not np.isnan(adata.layers[LAYER_INTENSITY_LOG2]).any()
        assert adata.layers[LAYER_INTENSITY_LOG2].shape == (4, 5)

    def test_copy_true_returns_anndata(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        adata = _make_adata(X)
        result = impute_knn_site_based(adata, copy=True)
        assert result is not None
        # Original is unchanged
        assert np.isnan(adata.layers[LAYER_INTENSITY_LOG2][0, 1])
        # Returned copy is imputed
        assert not np.isnan(result.layers[LAYER_INTENSITY_LOG2]).any()

    def test_default_k_is_sqrt_n_samples(self):
        """Verify the default n_neighbors path runs without error for k=2."""
        X = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [1.1, 2.1, np.nan, 4.1, 5.1],
                [0.9, np.nan, 2.9, 3.9, 4.9],
                [1.0, 2.0, 3.0, 4.0, 5.0],
            ]
        )
        adata = _make_adata(X)
        impute_knn_site_based(adata)  # k defaults to int(sqrt(4)) = 2
        assert not np.isnan(adata.layers[LAYER_INTENSITY_LOG2]).any()

    def test_uses_layer(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        adata = _make_adata(X)
        adata.layers["alt"] = X.copy()
        # Imputes the requested (non-canonical) layer only, leaves .X and the
        # canonical layer alone
        impute_knn_site_based(adata, layer="alt")
        assert not np.isnan(adata.layers["alt"]).any()
        assert np.isnan(adata.layers[LAYER_INTENSITY_LOG2][0, 1])
        assert np.isnan(adata.X[0, 1])

    def test_default_layer_mirrors_into_X(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        adata = _make_adata(X)
        impute_knn_site_based(adata)
        np.testing.assert_array_equal(adata.X, adata.layers[LAYER_INTENSITY_LOG2])
        assert not np.isnan(adata.X).any()

    def test_layer_none_targets_X(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        adata = _make_adata(X)
        # Explicit override to target .X instead of the log2 layer.
        impute_knn_site_based(adata, layer=None)
        assert not np.isnan(adata.X).any()
        assert np.isnan(adata.layers[LAYER_INTENSITY_LOG2][0, 1])

    def test_errors_on_all_nan_site(self):
        X = np.array([[1.0, np.nan, 3.0], [2.0, np.nan, 4.0]])
        adata = _make_adata(X)
        with pytest.raises(ValueError, match="no observed values"):
            impute_knn_site_based(adata)

    def test_no_op_when_complete(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        adata = _make_adata(X)
        impute_knn_site_based(adata)
        np.testing.assert_array_equal(adata.layers[LAYER_INTENSITY_LOG2], X)


# ============================================================================
# impute_hybrid
# ============================================================================


class TestImputeHybrid:
    def test_fills_all_missing(self):
        rng = np.random.default_rng(0)
        X = rng.normal(10, 1, size=(6, 20))
        # Random missingness on a few cells
        X[0, 0] = np.nan
        X[1, 5] = np.nan
        X[3, 12] = np.nan
        adata = _make_adata(X)
        impute_hybrid(adata)
        assert not np.isnan(adata.layers[LAYER_INTENSITY_LOG2]).any()

    def test_low_abundance_site_uses_gaussian(self):
        """A site whose observed values are well below the threshold should
        get Gaussian (downshifted) imputed values for its NaN cells, not
        site-KNN values.
        """
        X = np.array(
            [
                [10.0, 11.0, 12.0, 13.0, np.nan],  # missing in low-abundance site
                [10.0, 11.0, 12.0, 13.0, 2.0],
                [10.0, 11.0, 12.0, 13.0, 2.1],
                [10.0, 11.0, 12.0, 13.0, 1.9],
            ]
        )
        adata = _make_adata(X)
        impute_hybrid(adata, mnar_threshold_percentile=30, gaussian_seed=42)
        imputed = adata.layers[LAYER_INTENSITY_LOG2][0, 4]
        # Sample 0's observed values (sites 0-3) are {10,11,12,13}, mean=11.5,
        # std ~ 1.29. Downshift_mu = 11.5 - 1.8*1.29 ~ 9.18. So the imputed
        # value should be near 9.18, NOT near 2.0 (which is the site mean
        # KNN would give).
        assert imputed > 7.0  # well above 2.0
        assert imputed < 12.0

    def test_high_abundance_site_uses_knn(self):
        """A site whose observed values are well above the threshold should
        get KNN-imputed values.
        """
        X = np.array(
            [
                [2.0, 2.0, 2.0, 2.0, np.nan, 14.5, 15.0],
                [1.9, 1.9, 1.9, 1.9, 15.0, 14.6, 15.1],
                [2.1, 2.1, 2.1, 2.1, 15.1, 14.7, 15.2],
                [2.0, 2.0, 2.0, 2.0, 14.9, 14.5, 14.9],
            ]
        )
        adata = _make_adata(X)
        impute_hybrid(adata, mnar_threshold_percentile=30, gaussian_seed=42)
        imputed = adata.layers[LAYER_INTENSITY_LOG2][0, 4]
        # Site 4 mean ~ 15 -> MAR -> site-KNN -> borrows from neighbour sites
        # 5 and 6 -> imputed should be near (14.5 + 15.0)/2 = 14.75
        assert imputed > 10.0
        assert imputed < 17.0

    def test_audit_dataframe(self):
        X = np.array(
            [
                [10.0, 11.0, 12.0, 13.0, np.nan],
                [10.0, 11.0, 12.0, 13.0, 2.0],
                [10.0, 11.0, np.nan, 13.0, 2.1],
                [10.0, 11.0, 12.0, 13.0, 1.9],
            ]
        )
        adata = _make_adata(X)
        _, audit = impute_hybrid(adata, return_audit=True)
        assert isinstance(audit, pd.DataFrame)
        assert set(audit.columns) == {"sample_idx", "site_idx", "strategy"}
        # We had 2 missing cells: one low-abundance (MNAR) one high (MAR)
        assert len(audit) == 2
        assert set(audit["strategy"]) <= {"MAR_KNN", "MNAR_Gaussian"}

    def test_deterministic_with_seed(self):
        X = np.array(
            [
                [10.0, 11.0, np.nan, 13.0, np.nan],
                [10.0, 11.0, 2.0, 13.0, 2.0],
                [10.0, 11.0, 2.1, 13.0, 2.1],
                [10.0, 11.0, 1.9, 13.0, 1.9],
            ]
        )
        a1 = _make_adata(X)
        a2 = _make_adata(X)
        impute_hybrid(a1, gaussian_seed=42)
        impute_hybrid(a2, gaussian_seed=42)
        np.testing.assert_array_equal(
            a1.layers[LAYER_INTENSITY_LOG2], a2.layers[LAYER_INTENSITY_LOG2]
        )

    def test_different_seeds_give_different_values(self):
        X = np.array(
            [
                [10.0, 11.0, 12.0, 13.0, np.nan],
                [10.0, 11.0, 12.0, 13.0, 2.0],
                [10.0, 11.0, 12.0, 13.0, 2.1],
                [10.0, 11.0, 12.0, 13.0, 1.9],
            ]
        )
        a1 = _make_adata(X)
        a2 = _make_adata(X)
        impute_hybrid(a1, gaussian_seed=42)
        impute_hybrid(a2, gaussian_seed=7)
        assert a1.layers[LAYER_INTENSITY_LOG2][0, 4] != a2.layers[LAYER_INTENSITY_LOG2][0, 4]

    def test_no_missing_is_no_op(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        adata = _make_adata(X)
        impute_hybrid(adata)
        np.testing.assert_array_equal(adata.layers[LAYER_INTENSITY_LOG2], X)

    def test_copy_true_returns_anndata(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        adata = _make_adata(X)
        result = impute_hybrid(adata, copy=True)
        assert result is not None
        # Original layer unchanged
        assert np.isnan(adata.layers[LAYER_INTENSITY_LOG2][0, 1])
        # Copy has imputed values
        assert not np.isnan(result.layers[LAYER_INTENSITY_LOG2]).any()

    def test_default_imputes_log2_layer_and_mirrors_X(self):
        # The default writes to the canonical log2 layer AND mirrors into .X
        # (collapse contract .X == layers["intensity_log2"]), so .X readers
        # such as filter_by_completeness see the imputed values too.
        X = np.array(
            [
                [10.0, 11.0, 12.0, 13.0, np.nan],
                [10.0, 11.0, 12.0, 13.0, 2.0],
                [10.0, 11.0, 12.0, 13.0, 2.1],
                [10.0, 11.0, 12.0, 13.0, 1.9],
            ]
        )
        adata = _make_adata(X)
        impute_hybrid(adata)
        assert not np.isnan(adata.layers[LAYER_INTENSITY_LOG2]).any()
        np.testing.assert_array_equal(adata.X, adata.layers[LAYER_INTENSITY_LOG2])

    def test_sample_with_single_observation_never_gets_zero_filled(self):
        # Regression: gauss_filled used to be zeros-initialised, so a sample with
        # <2 observed values (no per-sample sigma) had its MNAR cells written as
        # 0.0 on the log2 scale instead of falling through to the global Gaussian.
        rng = np.random.default_rng(0)
        X = rng.normal(20, 2, size=(6, 40))
        X[X < 17.5] = np.nan
        X[5, :] = np.nan
        X[5, 0] = 21.0  # exactly one observed value in sample 5
        adata = _make_adata(X)
        impute_hybrid(adata)
        L = adata.layers[LAYER_INTENSITY_LOG2]
        assert not np.isnan(L).any()
        assert (L[5] > 5.0).all(), "zero-filled cells leaked into the log2 matrix"


# ============================================================================
# Return-value contract: ``adata = ap.impute_*(adata)`` must not clobber
# ============================================================================


class TestReturnContract:
    """Both imputers must ALWAYS return the AnnData, so
    ``adata = ap.impute_hybrid(adata)`` is a safe no-op-and-return."""

    def _adata_with_nan(self) -> ad.AnnData:
        return _make_adata(np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]]))

    def test_impute_hybrid_returns_same_object_in_place(self):
        adata = self._adata_with_nan()
        result = impute_hybrid(adata)
        assert result is adata

    def test_impute_hybrid_returns_copy_when_copy_true(self):
        adata = self._adata_with_nan()
        result = impute_hybrid(adata, copy=True)
        assert result is not adata

    def test_impute_hybrid_assignment_pattern(self):
        # The exact UX pattern the fix targets.
        adata = self._adata_with_nan()
        adata = impute_hybrid(adata)
        assert adata is not None
        assert not np.isnan(adata.layers[LAYER_INTENSITY_LOG2]).any()

    def test_impute_hybrid_audit_tuple(self):
        adata = self._adata_with_nan()
        result, audit = impute_hybrid(adata, return_audit=True)
        assert result is adata
        assert isinstance(audit, pd.DataFrame)

    def test_impute_hybrid_audit_tuple_with_copy(self):
        adata = self._adata_with_nan()
        result, audit = impute_hybrid(adata, return_audit=True, copy=True)
        assert result is not adata
        assert isinstance(audit, pd.DataFrame)

    def test_impute_knn_returns_same_object_in_place(self):
        adata = self._adata_with_nan()
        result = impute_knn_site_based(adata)
        assert result is adata

    def test_impute_knn_returns_copy_when_copy_true(self):
        adata = self._adata_with_nan()
        result = impute_knn_site_based(adata, copy=True)
        assert result is not adata

    def test_impute_knn_assignment_pattern(self):
        adata = self._adata_with_nan()
        adata = impute_knn_site_based(adata)
        assert adata is not None
        assert not np.isnan(adata.layers[LAYER_INTENSITY_LOG2]).any()
