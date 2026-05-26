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

from alphaphos.preprocess.impute import (
    _check_complete_features,
    impute_hybrid,
    impute_knn_site_based,
)

# Helper -----------------------------------------------------------------


def _make_adata(X: np.ndarray) -> ad.AnnData:
    n_samples, n_sites = X.shape
    return ad.AnnData(
        X=X.astype(float).copy(),
        obs=pd.DataFrame(index=[f"s{i}" for i in range(n_samples)]),
        var=pd.DataFrame(index=[f"site{j}" for j in range(n_sites)]),
    )


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
        assert not np.isnan(adata.X).any()
        assert adata.X.shape == (4, 5)

    def test_copy_true_returns_anndata(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        adata = _make_adata(X)
        result = impute_knn_site_based(adata, copy=True)
        assert result is not None
        # Original is unchanged
        assert np.isnan(adata.X[0, 1])
        # Returned copy is imputed
        assert not np.isnan(result.X).any()

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
        assert not np.isnan(adata.X).any()

    def test_uses_layer(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        adata = _make_adata(X)
        adata.layers["alt"] = X.copy()
        # Imputes the layer, leaves .X alone
        impute_knn_site_based(adata, layer="alt")
        assert not np.isnan(adata.layers["alt"]).any()
        assert np.isnan(adata.X[0, 1])

    def test_errors_on_all_nan_site(self):
        X = np.array([[1.0, np.nan, 3.0], [2.0, np.nan, 4.0]])
        adata = _make_adata(X)
        with pytest.raises(ValueError, match="no observed values"):
            impute_knn_site_based(adata)

    def test_no_op_when_complete(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        adata = _make_adata(X)
        impute_knn_site_based(adata)
        # Result should be identical to input
        np.testing.assert_array_equal(adata.X, X)


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
        assert not np.isnan(adata.X).any()

    def test_low_abundance_site_uses_gaussian(self):
        """A site whose observed values are well below the threshold should
        get Gaussian (downshifted) imputed values for its NaN cells, not
        site-KNN values.
        """
        # 4 samples, 5 sites. Site 4 is low-abundance; cell (0, 4) is missing.
        # Make Sample 0's observed-cell distribution wide so the downshift is
        # clearly below the observed range.
        X = np.array(
            [
                [10.0, 11.0, 12.0, 13.0, np.nan],  # missing in low-abundance site
                [10.0, 11.0, 12.0, 13.0, 2.0],
                [10.0, 11.0, 12.0, 13.0, 2.1],
                [10.0, 11.0, 12.0, 13.0, 1.9],
            ]
        )
        # The overall observed-cell distribution has values in {1.9..13.0}.
        # At default 30th percentile, threshold sits around 4–5. Site 4's
        # mean is ~2.0 → BELOW threshold → MNAR → Gaussian.
        adata = _make_adata(X)
        impute_hybrid(adata, mnar_threshold_percentile=30, gaussian_seed=42)
        imputed = adata.X[0, 4]
        # Sample 0's observed values (sites 0–3) are {10,11,12,13}, mean=11.5,
        # std ≈ 1.29. Downshift_mu = 11.5 - 1.8*1.29 ≈ 9.18. So the imputed
        # value should be near 9.18, NOT near 2.0 (which is the site mean
        # KNN would give).
        assert imputed > 7.0  # well above 2.0
        assert imputed < 12.0

    def test_high_abundance_site_uses_knn(self):
        """A site whose observed values are well above the threshold should
        get KNN-imputed values (near other-sample values of similar high-
        abundance sites).
        """
        # Sites 4, 5, 6 are all high-abundance siblings. Site 4 has a missing
        # cell in sample 0; KNN should find sites 5, 6 as nearest neighbours
        # (by Euclidean distance over the 3 non-missing samples) and impute
        # site 4's sample-0 cell at roughly the average of sites 5,6 at
        # sample 0, i.e. ~14.5.
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
        imputed = adata.X[0, 4]
        # Site 4 mean ≈ 15 → MAR → site-KNN → borrows from neighbour sites
        # 5 and 6 → imputed should be near (14.5 + 15.0)/2 = 14.75
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
        np.testing.assert_array_equal(a1.X, a2.X)

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
        assert a1.X[0, 4] != a2.X[0, 4]

    def test_no_missing_is_no_op(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        adata = _make_adata(X)
        impute_hybrid(adata)
        np.testing.assert_array_equal(adata.X, X)

    def test_copy_true_returns_anndata(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        adata = _make_adata(X)
        result = impute_hybrid(adata, copy=True)
        assert result is not None
        assert np.isnan(adata.X[0, 1])  # original unchanged
        assert not np.isnan(result.X).any()
