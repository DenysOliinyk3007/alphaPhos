"""Tests for :mod:`alphaphos.dimred`.

Coverage:
  - Standard PCA on synthetic data (variance ratio, orthogonality)
  - NIPALS matches sklearn PCA on complete data (up to sign flips)
  - NIPALS handles NaN natively; sklearn errors on NaN
  - PPCA matches sklearn PCA on complete data (up to rotation)
  - PPCA handles NaN natively
  - Settings validation
  - Sample distance NaN-safe
  - Hierarchical clustering runs
  - Imputation-impact comparison flags obvious distortion
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA as SklearnPCA

import alphaphos as ap
from alphaphos.dimred.pca import (
    _pca_nipals,
    _pca_ppca,
    _pca_sklearn,
    resolve_pca_settings,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adata(X: np.ndarray) -> ad.AnnData:
    n_samples, n_features = X.shape
    obs = pd.DataFrame(index=[f"s{i}" for i in range(n_samples)])
    var = pd.DataFrame(index=[f"g{j}" for j in range(n_features)])
    a = ad.AnnData(X=X.astype(np.float64), obs=obs, var=var)
    a.layers["intensity_log2"] = a.X.copy()
    return a


def _synthetic_signal(n_samples: int = 12, n_features: int = 100, seed: int = 0) -> np.ndarray:
    """Two-cluster data: first n/2 samples have a systematic offset on the
    first 20 features -- a strong PC1 signal easy to recover."""
    rng = np.random.default_rng(seed)
    X = rng.normal(20.0, 0.5, size=(n_samples, n_features))
    X[n_samples // 2 :, :20] += 3.0  # cluster shift on features 0..19
    return X


def _absolute_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation between two vectors, sign-flip-insensitive."""
    ac = a - a.mean()
    bc = b - b.mean()
    denom = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    return abs(float(ac @ bc)) / denom if denom > 0 else 0.0


# ===========================================================================
# Settings
# ===========================================================================


class TestSettings:
    def test_defaults_returned_when_none(self):
        out = resolve_pca_settings(None)
        assert out["n_components"] == 10
        assert out["handle_missing"] == "error"

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown advanced keys"):
            resolve_pca_settings({"bogus": True})

    def test_bad_handle_missing_raises(self):
        with pytest.raises(ValueError, match="handle_missing must be"):
            resolve_pca_settings({"handle_missing": "svd_impute"})

    def test_bad_n_components_raises(self):
        with pytest.raises(ValueError, match="n_components must be positive"):
            resolve_pca_settings({"n_components": 0})


# ===========================================================================
# Standard PCA on complete data
# ===========================================================================


class TestStandardPca:
    def test_variance_ratio_matches_sklearn(self):
        X = _synthetic_signal()
        adata = _make_adata(X)
        result = ap.dimred.pca(adata, n_components=5)
        # Compare to sklearn directly
        model = SklearnPCA(n_components=5)
        model.fit_transform(X - X.mean(axis=0))
        expected_ratio = (
            model.explained_variance_ / np.var(X - X.mean(axis=0), axis=0, ddof=1).sum()
        )
        got_ratio = result.uns["pca"]["variance_ratio"]
        np.testing.assert_allclose(got_ratio, expected_ratio, atol=1e-6)

    def test_error_on_nan_by_default(self):
        X = _synthetic_signal().copy()
        X[0, 0] = np.nan
        adata = _make_adata(X)
        with pytest.raises(ValueError, match="handle_missing='error' but"):
            ap.dimred.pca(adata)

    def test_pc1_recovers_cluster_signal(self):
        X = _synthetic_signal()
        adata = _make_adata(X)
        result = ap.dimred.pca(adata, n_components=3)
        pc1 = result.obsm["X_pca"][:, 0]
        # Half of samples are +3-shifted; PC1 should separate them.
        n_half = len(pc1) // 2
        assert abs(pc1[:n_half].mean() - pc1[n_half:].mean()) > 1.0

    def test_output_shapes(self):
        X = _synthetic_signal(n_samples=8, n_features=50)
        adata = _make_adata(X)
        result = ap.dimred.pca(adata, n_components=4)
        assert result.obsm["X_pca"].shape == (8, 4)
        assert result.varm["PCs"].shape == (50, 4)
        assert result.uns["pca"]["variance"].shape == (4,)
        assert result.uns["pca"]["variance_ratio"].shape == (4,)


# ===========================================================================
# NIPALS
# ===========================================================================


class TestNipals:
    def test_matches_sklearn_on_complete_data(self):
        """PC1 from NIPALS should be perfectly (anti-)correlated with PC1 from sklearn."""
        X = _synthetic_signal(seed=1)
        Xc = X - X.mean(axis=0)
        std_scores, _, _ = _pca_sklearn(Xc, n_components=3)
        nip_scores, _, _ = _pca_nipals(Xc, n_components=3, max_iter=500, tol=1e-8)
        # Component-wise abs correlation should be ~1.
        for c in range(3):
            r = _absolute_correlation(std_scores[:, c], nip_scores[:, c])
            assert r > 0.99, f"NIPALS PC{c} corr with sklearn: {r:.3f}"

    def test_handles_nan_natively(self):
        X = _synthetic_signal(seed=2).copy()
        # Punch ~10% missing at random
        rng = np.random.default_rng(0)
        mask = rng.random(X.shape) < 0.1
        X[mask] = np.nan
        adata = _make_adata(X)
        result = ap.dimred.pca(adata, n_components=3, handle_missing="nipals")
        # Should still separate the two clusters on PC1
        pc1 = result.obsm["X_pca"][:, 0]
        n_half = len(pc1) // 2
        assert abs(pc1[:n_half].mean() - pc1[n_half:].mean()) > 0.5

    def test_scores_are_uncorrelated_across_pcs(self):
        X = _synthetic_signal()
        Xc = X - X.mean(axis=0)
        scores, _, _ = _pca_nipals(Xc, n_components=3, max_iter=500, tol=1e-8)
        # After deflation, PC1 and PC2 scores should have near-zero correlation
        for i in range(3):
            for j in range(i + 1, 3):
                r = _absolute_correlation(scores[:, i], scores[:, j])
                assert r < 0.3, f"NIPALS PC{i} and PC{j} correlated: {r:.3f}"

    def test_variance_ratio_bounded_on_sparse_masked_data(self):
        # Regression: on sparse NaN data (samples have few observed features),
        # NIPALS previously produced score-column variance blow-ups (variance
        # ratios >1, sometimes reaching 1000+) because t_i = Sum(mask*X*p) /
        # Sum(mask*p^2) divides by a small denominator when few features are
        # observed for sample i. The mask-aware SS-based variance keeps
        # ratios in [0, 1] regardless of masking.
        rng = np.random.default_rng(7)
        X = rng.normal(0, 1, size=(20, 100))
        X[rng.random(X.shape) < 0.3] = np.nan  # 30% missing
        adata = _make_adata(X)
        result = ap.dimred.pca(adata, n_components=5, handle_missing="nipals")
        ratios = result.uns["pca"]["variance_ratio"]
        assert (ratios >= 0).all(), f"negative variance_ratio: {ratios}"
        assert (ratios <= 1).all(), f"variance_ratio > 1 on sparse NIPALS: {ratios}"
        # First 5 PCs of a 20x100 random matrix shouldn't sum above 1 either.
        assert ratios.sum() <= 1.0 + 1e-9, f"cumulative variance_ratio > 1: {ratios.sum()}"

    def test_variance_ratio_matches_sklearn_on_complete_data(self):
        # Backwards-compat check: on complete data, NIPALS variance_ratio must
        # match sklearn's exactly (the SS-based formulation reduces to the
        # eigenvalue formulation when there are no masked cells).
        X = _synthetic_signal(n_samples=20, n_features=100, seed=3)
        adata = _make_adata(X)
        r_nip = ap.dimred.pca(adata, n_components=5, handle_missing="nipals")
        r_sk = ap.dimred.pca(adata, n_components=5, handle_missing="error")
        np.testing.assert_allclose(
            r_nip.uns["pca"]["variance_ratio"],
            r_sk.uns["pca"]["variance_ratio"],
            atol=1e-6,
            err_msg="NIPALS variance_ratio must match sklearn on complete data",
        )


# ===========================================================================
# PPCA
# ===========================================================================


class TestPpca:
    def test_matches_sklearn_on_complete_data(self):
        X = _synthetic_signal(seed=3)
        Xc = X - X.mean(axis=0)
        std_scores, _, _ = _pca_sklearn(Xc, n_components=3)
        ppca_scores, _, _ = _pca_ppca(Xc, n_components=3, max_iter=200, tol=1e-6, seed=42)
        # PPCA converges to the same subspace; PC1 should be highly (anti-)correlated
        r_pc1 = _absolute_correlation(std_scores[:, 0], ppca_scores[:, 0])
        assert r_pc1 > 0.95, f"PPCA PC1 corr with sklearn: {r_pc1:.3f}"

    def test_handles_nan_natively(self):
        X = _synthetic_signal(seed=4).copy()
        # Fewer NaN than NIPALS test -- PPCA is slower and more sensitive to sparse cases
        rng = np.random.default_rng(1)
        mask = rng.random(X.shape) < 0.05
        X[mask] = np.nan
        adata = _make_adata(X)
        result = ap.dimred.pca(adata, n_components=3, handle_missing="ppca")
        pc1 = result.obsm["X_pca"][:, 0]
        n_half = len(pc1) // 2
        assert abs(pc1[:n_half].mean() - pc1[n_half:].mean()) > 0.3

    def test_variance_ratio_matches_sklearn_on_complete_data(self):
        # Regression: PPCA previously returned per-PC variance = var(Z_mean,
        # ddof=1) which is the LATENT-space variance (unit-ish under the
        # prior) rather than the observed-space projection variance.  That
        # made variance_ratio ~10x too small vs sklearn/NIPALS.  The fix
        # projects X onto the SVD-oriented loadings and computes the
        # mask-aware reconstruction SS/(n-1), consistent with NIPALS.
        rng = np.random.default_rng(3)
        X = rng.normal(0, 1, size=(20, 100))
        adata = _make_adata(X)
        r_pp = ap.dimred.pca(
            adata, n_components=5, handle_missing="ppca", advanced={"max_iter": 200, "tol": 1e-7}
        )
        r_sk = ap.dimred.pca(adata, n_components=5, handle_missing="error")
        np.testing.assert_allclose(
            r_pp.uns["pca"]["variance_ratio"].sum(),
            r_sk.uns["pca"]["variance_ratio"].sum(),
            atol=1e-3,
            err_msg="PPCA cumulative variance_ratio must match sklearn on complete data",
        )
        # Per-PC agreement is looser because PPCA's EM introduces small
        # rotations within near-degenerate subspaces.  Total is the strict check.

    def test_variance_ratio_bounded_on_sparse_masked_data(self):
        # Regression: PPCA on masked data must not exceed 1.0 per PC or in
        # aggregate.  Uses the same 30% NaN random matrix as the NIPALS bound test.
        rng = np.random.default_rng(7)
        X = rng.normal(0, 1, size=(20, 100))
        X[rng.random(X.shape) < 0.3] = np.nan
        adata = _make_adata(X)
        result = ap.dimred.pca(
            adata, n_components=5, handle_missing="ppca", advanced={"max_iter": 200, "tol": 1e-7}
        )
        ratios = result.uns["pca"]["variance_ratio"]
        assert (ratios >= 0).all(), f"negative variance_ratio: {ratios}"
        assert (ratios <= 1).all(), f"variance_ratio > 1 on sparse PPCA: {ratios}"
        assert ratios.sum() <= 1.0 + 1e-9, f"cumulative variance_ratio > 1: {ratios.sum()}"


# ===========================================================================
# Sample distance
# ===========================================================================


class TestSampleDistance:
    def test_symmetric_zero_diagonal(self):
        X = _synthetic_signal(n_samples=6, n_features=20)
        adata = _make_adata(X)
        dm = ap.dimred.sample_distance(adata)
        assert dm.shape == (6, 6)
        np.testing.assert_allclose(dm.to_numpy().diagonal(), 0, atol=1e-10)
        np.testing.assert_allclose(dm.to_numpy(), dm.to_numpy().T, atol=1e-10)

    def test_nan_safe(self):
        X = _synthetic_signal(n_samples=4, n_features=20).copy()
        X[0, :5] = np.nan
        X[1, 5:10] = np.nan
        adata = _make_adata(X)
        dm = ap.dimred.sample_distance(adata)
        # No NaN in output (pair 0,1 shares features 10..19)
        assert not np.isnan(dm.to_numpy()).any()

    def test_correlation_metric(self):
        # Identical samples -> corr distance = 0
        X = np.tile(_synthetic_signal(n_samples=1, n_features=20)[0], (3, 1))
        adata = _make_adata(X)
        dm = ap.dimred.sample_distance(adata, metric="correlation")
        assert dm.iloc[0, 1] < 1e-8


class TestHierarchicalCluster:
    def test_returns_expected_keys(self):
        X = _synthetic_signal(n_samples=6, n_features=20)
        adata = _make_adata(X)
        dm = ap.dimred.sample_distance(adata)
        result = ap.dimred.hierarchical_cluster(dm)
        assert set(result.keys()) == {"linkage", "labels", "leaf_order"}
        assert len(result["labels"]) == 6
        assert len(result["leaf_order"]) == 6

    def test_leaf_order_permutation(self):
        X = _synthetic_signal(n_samples=6, n_features=20)
        adata = _make_adata(X)
        dm = ap.dimred.sample_distance(adata)
        result = ap.dimred.hierarchical_cluster(dm)
        # leaf_order should be a permutation of 0..n-1
        assert sorted(result["leaf_order"]) == list(range(6))


# ===========================================================================
# Imputation-impact comparison
# ===========================================================================


class TestImputationComparison:
    def _make_paired_adatas(self):
        """Return (raw_with_nan, imputed) sharing the same sample labels."""
        X = _synthetic_signal(seed=7).astype(np.float64)
        rng = np.random.default_rng(0)
        mask = rng.random(X.shape) < 0.1
        X_raw = X.copy()
        X_raw[mask] = np.nan
        adata_raw = _make_adata(X_raw)
        # "Imputed" version: fill with column means (deliberately simple)
        col_means = np.nanmean(X_raw, axis=0)
        X_imp = np.where(mask, col_means[np.newaxis, :], X_raw)
        adata_imp = _make_adata(X_imp)
        return adata_raw, adata_imp

    def test_returns_expected_keys(self):
        adata_raw, adata_imp = self._make_paired_adatas()
        report = ap.dimred.compare_imputation_impact(adata_raw, adata_imp, n_components=3)
        expected_keys = {
            "pc_coords_imputed",
            "pc_coords_raw",
            "variance_ratio_imputed",
            "variance_ratio_raw",
            "per_pc_correlation",
            "per_sample_distance",
            "sample_labels",
            "n_components",
            "method_raw",
            "verdict",
        }
        assert expected_keys.issubset(report.keys())

    def test_mean_imputed_preserves_pc1_structure(self):
        """Mean-imputation is a mild transformation -- PC1 should be preserved."""
        adata_raw, adata_imp = self._make_paired_adatas()
        report = ap.dimred.compare_imputation_impact(
            adata_raw, adata_imp, n_components=3, method_raw="nipals"
        )
        assert report["per_pc_correlation"][0] > 0.9

    def test_mismatched_obs_names_raises(self):
        adata_raw, adata_imp = self._make_paired_adatas()
        adata_imp.obs.index = [f"other_{i}" for i in range(adata_imp.n_obs)]
        with pytest.raises(ValueError, match="obs_names"):
            ap.dimred.compare_imputation_impact(adata_raw, adata_imp)

    def test_verdict_reflects_distortion(self):
        """Heavy noise imputation should show distortion in the verdict."""
        X = _synthetic_signal(seed=8).astype(np.float64)
        rng = np.random.default_rng(0)
        # Punch 30% missing
        mask = rng.random(X.shape) < 0.3
        X_raw = X.copy()
        X_raw[mask] = np.nan
        # Adversarial "imputation": fill missing with pure noise (breaks structure)
        X_bad = X_raw.copy()
        X_bad[mask] = rng.normal(0.0, 5.0, size=int(mask.sum()))
        adata_raw = _make_adata(X_raw)
        adata_bad = _make_adata(X_bad)
        report = ap.dimred.compare_imputation_impact(
            adata_raw, adata_bad, n_components=3, method_raw="nipals"
        )
        # At least one PC should show reduced correlation with the raw NIPALS PC
        assert report["per_pc_correlation"].min() < 0.9


# ===========================================================================
# Loadings-for-enrichment
# ===========================================================================


def _make_adata_phospho(X: np.ndarray, var_names: list[str]) -> ad.AnnData:
    n_samples, n_features = X.shape
    assert len(var_names) == n_features
    obs = pd.DataFrame(index=[f"s{i}" for i in range(n_samples)])
    var = pd.DataFrame(index=var_names)
    a = ad.AnnData(X=X.astype(np.float64), obs=obs, var=var)
    a.layers["intensity_log2"] = a.X.copy()
    return a


class TestLoadingsForEnrichment:
    def _fit(self, var_names: list[str], n_components: int = 3):
        rng = np.random.default_rng(0)
        X = rng.normal(20.0, 1.0, size=(6, len(var_names)))
        adata = _make_adata_phospho(X, var_names)
        return ap.dimred.pca(adata, n_components=n_components)

    def test_canonicalizes_var_names(self):
        adata = self._fit(
            [
                "Q9H307|PNN|S100|M1",
                "P42229|STAT5A|Y694|M1",
                "Q4KMP7|TBC1D10B|S132|M1",
                "Q8N6T3|ARFGAP1|S361|M1",
            ]
        )
        df = ap.dimred.loadings_for_enrichment(adata)
        assert list(df.index) == ["Q9H307_S100", "P42229_Y694", "Q4KMP7_S132", "Q8N6T3_S361"]
        assert list(df.columns) == ["PC1", "PC2", "PC3"]

    def test_output_feeds_gsea_series_shape(self):
        # gsea() takes a pd.Series indexed by str site_ids with no NaN.
        adata = self._fit(
            ["Q9H307|PNN|S100|M1", "P42229|STAT5A|Y694|M1", "Q4KMP7|TBC1D10B|S132|M1"]
        )
        series = ap.dimred.loadings_for_enrichment(adata)["PC1"]
        assert isinstance(series, pd.Series)
        assert series.dtype.kind == "f"
        assert not series.isna().any()
        assert series.index.map(lambda s: "_" in s and "|" not in s).all()

    def test_dedup_abs_max_keeps_larger_magnitude(self):
        # Two rows for the same canonical site (M1 and M2). Give the second one
        # a much larger signal so it dominates PC1 -- expect it to be retained.
        var_names = [
            "Q9H307|PNN|S100|M1",
            "Q9H307|PNN|S100|M2",  # collides with M1 on canonical Q9H307_S100
            "P42229|STAT5A|Y694|M1",
        ]
        rng = np.random.default_rng(0)
        X = rng.normal(20.0, 0.1, size=(6, 3))
        # M2 row: give it a strong sample-differentiating signal
        X[:3, 1] += 5.0
        adata = _make_adata_phospho(X, var_names)
        adata = ap.dimred.pca(adata, n_components=2)
        pcs_raw = np.asarray(adata.varm["PCs"])
        # Row 1 (M2) should have larger sum-of-squares than row 0 (M1)
        assert (pcs_raw[1] ** 2).sum() > (pcs_raw[0] ** 2).sum()

        df = ap.dimred.loadings_for_enrichment(adata, dedup="abs_max")
        # Q9H307_S100 kept exactly once, with the M2 row's values
        assert (df.index == "Q9H307_S100").sum() == 1
        kept = df.loc["Q9H307_S100"].to_numpy()
        np.testing.assert_allclose(kept, pcs_raw[1, :2])

    def test_dedup_error_raises_on_collision(self):
        var_names = ["Q9H307|PNN|S100|M1", "Q9H307|PNN|S100|M2"]
        rng = np.random.default_rng(0)
        X = rng.normal(20.0, 1.0, size=(6, 2))
        adata = _make_adata_phospho(X, var_names)
        adata = ap.dimred.pca(adata, n_components=2)
        with pytest.raises(ValueError, match="collide on canonical site_id"):
            ap.dimred.loadings_for_enrichment(adata, dedup="error")

    def test_unparseable_keys_dropped(self):
        var_names = ["Q9H307|PNN|S100|M1", "not_an_alphaphos_key", "P42229|STAT5A|Y694|M1"]
        rng = np.random.default_rng(0)
        X = rng.normal(20.0, 1.0, size=(6, 3))
        adata = _make_adata_phospho(X, var_names)
        adata = ap.dimred.pca(adata, n_components=2)
        df = ap.dimred.loadings_for_enrichment(adata)
        assert len(df) == 2
        assert set(df.index) == {"Q9H307_S100", "P42229_Y694"}

    def test_attrs_survive_pandas_concat(self):
        # Regression: .attrs values must not be numpy arrays -- pandas.concat
        # compares them via == which returns element-wise bools on ndarrays and
        # crashes with "truth value of an array is ambiguous". Downstream
        # enrichment functions (e.g. ora) call concat internally.
        adata = self._fit(["Q9H307|PNN|S100|M1", "P42229|STAT5A|Y694|M1"])
        df = ap.dimred.loadings_for_enrichment(adata)
        # Trigger a concat that will exercise .attrs finalization
        _ = pd.concat([df, df.iloc[:0]])
        _ = pd.concat([df["PC1"].to_frame(), df["PC2"].to_frame()], axis=1)


# ===========================================================================
# Feature variance contribution
# ===========================================================================


class TestFeatureVarianceContribution:
    def test_sums_to_100_pct(self):
        X = _synthetic_signal(n_samples=8, n_features=50)
        adata = _make_adata(X)
        adata = ap.dimred.pca(adata, n_components=5)
        df = ap.dimred.feature_variance_contribution(adata, n_components=5)
        np.testing.assert_allclose(df["contribution_pct"].sum(), 100.0, atol=1e-8)

    def test_sorted_descending(self):
        X = _synthetic_signal(n_samples=8, n_features=50)
        adata = _make_adata(X)
        adata = ap.dimred.pca(adata, n_components=5)
        df = ap.dimred.feature_variance_contribution(adata, n_components=5)
        assert df["contribution_pct"].is_monotonic_decreasing

    def test_signal_features_rank_higher_than_noise(self):
        # First 20 features carry the cluster signal (see _synthetic_signal).
        # Top-20 contributors should overlap the signal set at above chance.
        X = _synthetic_signal(n_samples=12, n_features=100, seed=1)
        adata = _make_adata(X)
        adata = ap.dimred.pca(adata, n_components=3)
        df = ap.dimred.feature_variance_contribution(adata, n_components=3)
        top20 = df.head(20).index
        signal_ids = {f"g{i}" for i in range(20)}
        overlap = len(set(top20) & signal_ids)
        assert overlap >= 15  # chance-level for 20/100 into top-20 is ~4

    def test_raises_without_pca(self):
        X = _synthetic_signal()
        adata = _make_adata(X)
        with pytest.raises(KeyError, match="X_pca|PCs"):
            ap.dimred.feature_variance_contribution(adata)
