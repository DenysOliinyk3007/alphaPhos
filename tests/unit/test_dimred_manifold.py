"""Unit tests for :mod:`alphaphos.dimred.tsne` and :mod:`.umap`.

Covers the core happy path (cluster recovery on 2-block synthetic
input), the NaN guard, the `n_pca_components` bridge via
`ap.dimred.pca`, and the accessor DataFrames.  UMAP tests are
auto-skipped when ``umap-learn`` is not installed.
"""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphaphos.dimred import (
    DEFAULT_TSNE_SETTINGS,
    DEFAULT_UMAP_SETTINGS,
    MIN_RECOMMENDED_N_TSNE,
    MIN_RECOMMENDED_N_UMAP,
    get_tsne_dataframe,
    get_umap_dataframe,
    pca,
    resolve_tsne_settings,
    resolve_umap_settings,
    tsne,
)

# UMAP is optional (`[dimred]` extra) -- skip UMAP tests if missing.
umap_learn = pytest.importorskip("umap")
from alphaphos.dimred import umap  # noqa: E402  (guarded by importorskip)


def _make_2cluster_adata(n_per=20, n_features=200, n_signal_features=50, boost=3.0, seed=0):
    rng = np.random.default_rng(seed)
    n_samples = 2 * n_per
    X = rng.normal(0, 1, size=(n_samples, n_features))
    X[:n_per, :n_signal_features] += boost
    obs = pd.DataFrame(
        {"group": ["A"] * n_per + ["B"] * n_per},
        index=[f"s{i}" for i in range(n_samples)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_features)])
    adata = ad.AnnData(X=X.astype(np.float64), obs=obs, var=var)
    adata.layers["intensity_log2"] = adata.X.copy()
    return adata


# ---------------------------------------------------------------------------
# Settings resolvers
# ---------------------------------------------------------------------------


class TestSettingsResolvers:
    def test_tsne_defaults_frozen(self):
        assert DEFAULT_TSNE_SETTINGS["n_components"] == 2
        assert DEFAULT_TSNE_SETTINGS["perplexity"] == 30.0
        assert DEFAULT_TSNE_SETTINGS["seed"] == 42

    def test_tsne_override_layer(self):
        got = resolve_tsne_settings({"layer": "intensity_log2_precombat"})
        assert got["layer"] == "intensity_log2_precombat"
        assert got["perplexity"] == 30.0  # untouched

    def test_tsne_unknown_key_raises(self):
        with pytest.raises(ValueError, match="unknown advanced keys"):
            resolve_tsne_settings({"bogus": 1})

    def test_umap_defaults_frozen(self):
        assert DEFAULT_UMAP_SETTINGS["n_components"] == 2
        assert DEFAULT_UMAP_SETTINGS["n_neighbors"] == 15
        assert DEFAULT_UMAP_SETTINGS["min_dist"] == 0.1

    def test_umap_unknown_key_raises(self):
        with pytest.raises(ValueError, match="unknown advanced keys"):
            resolve_umap_settings({"bogus": 1})


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------


class TestTSNE:
    def test_returns_adata_with_obsm_and_uns(self):
        adata = _make_2cluster_adata()
        result = tsne(adata, perplexity=8, seed=42)
        assert result.obsm["X_tsne"].shape == (adata.n_obs, 2)
        for key in ("perplexity", "seed", "n_components", "method"):
            assert key in result.uns["tsne"]

    def test_perplexity_clipped_when_too_large(self, caplog):
        adata = _make_2cluster_adata(n_per=5)  # n_samples = 10
        with caplog.at_level("WARNING"), pytest.warns(UserWarning, match="< recommended"):
            tsne(adata, perplexity=50, seed=0)
        # sklearn requires perplexity < n_samples; we clip to n_samples - 1
        assert adata.uns["tsne"]["perplexity"] < 10
        assert any("clipping" in rec.message for rec in caplog.records)

    def test_nan_input_raises(self):
        adata = _make_2cluster_adata()
        adata.layers["intensity_log2"][0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            tsne(adata, perplexity=8)

    def test_seed_deterministic(self):
        # Two runs with the same seed on a fresh AnnData produce identical
        # coordinates.
        adata1 = _make_2cluster_adata()
        adata2 = _make_2cluster_adata()
        tsne(adata1, perplexity=8, seed=42)
        tsne(adata2, perplexity=8, seed=42)
        np.testing.assert_array_equal(adata1.obsm["X_tsne"], adata2.obsm["X_tsne"])

    def test_cluster_separation_recovers_structure(self):
        adata = _make_2cluster_adata()
        tsne(adata, perplexity=8, seed=42)
        emb = adata.obsm["X_tsne"]
        centroid_a = emb[:20].mean(axis=0)
        centroid_b = emb[20:].mean(axis=0)
        # Real distance is >~10 on this synthetic 2-block set; bar set loose
        # to allow for cross-version variation of sklearn.TSNE.
        assert np.linalg.norm(centroid_a - centroid_b) > 5.0

    def test_copy_semantics_no_mutation(self):
        adata = _make_2cluster_adata()
        _out = tsne(adata, perplexity=8, seed=42, copy=True)
        assert "X_tsne" not in adata.obsm

    def test_n_pca_components_bridge(self):
        # Pre-reduce with PCA then feed PCs into t-SNE via n_pca_components.
        adata = _make_2cluster_adata()
        pca(adata, n_components=5, layer="intensity_log2")
        tsne(adata, perplexity=8, n_pca_components=5, seed=42)
        assert adata.uns["tsne"]["n_pca_components"] == 5
        # When PCA path is used, layer should be recorded as None.
        assert adata.uns["tsne"]["layer"] is None


# ---------------------------------------------------------------------------
# UMAP -- skipped when umap-learn not installed (via importorskip above)
# ---------------------------------------------------------------------------


class TestUMAP:
    def test_returns_adata_with_obsm_and_uns(self):
        adata = _make_2cluster_adata()
        result = umap(adata, n_neighbors=5, seed=42)
        assert result.obsm["X_umap"].shape == (adata.n_obs, 2)
        for key in ("n_neighbors", "min_dist", "metric", "seed"):
            assert key in result.uns["umap"]

    def test_n_neighbors_clipped_when_too_large(self, caplog):
        adata = _make_2cluster_adata(n_per=5)  # n_samples = 10
        with caplog.at_level("WARNING"), pytest.warns(UserWarning, match="< recommended"):
            umap(adata, n_neighbors=25, seed=0)
        assert adata.uns["umap"]["n_neighbors"] < 10

    def test_nan_input_raises(self):
        adata = _make_2cluster_adata()
        adata.layers["intensity_log2"][0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            umap(adata, n_neighbors=5)

    def test_cluster_separation_recovers_structure(self):
        adata = _make_2cluster_adata()
        umap(adata, n_neighbors=5, seed=42)
        emb = adata.obsm["X_umap"]
        centroid_a = emb[:20].mean(axis=0)
        centroid_b = emb[20:].mean(axis=0)
        assert np.linalg.norm(centroid_a - centroid_b) > 3.0

    def test_n_pca_components_bridge(self):
        adata = _make_2cluster_adata()
        pca(adata, n_components=5, layer="intensity_log2")
        umap(adata, n_neighbors=5, n_pca_components=5, seed=42)
        assert adata.uns["umap"]["n_pca_components"] == 5
        assert adata.uns["umap"]["layer"] is None


# ---------------------------------------------------------------------------
# Small-n reliability warnings
# ---------------------------------------------------------------------------


class TestSmallNWarnings:
    def test_tsne_warns_below_threshold(self):
        # n_samples = 20 < MIN_RECOMMENDED_N_TSNE (30)
        adata = _make_2cluster_adata(n_per=10)
        with pytest.warns(UserWarning, match=r"n_samples=20.*recommended minimum 30"):
            tsne(adata, perplexity=5, seed=0)

    def test_tsne_no_warning_at_threshold(self):
        # n_samples = 30 == MIN_RECOMMENDED_N_TSNE; small-n warning MUST NOT fire.
        # (umap-learn / sklearn may still emit unrelated UserWarnings; we only
        # assert *our* small-n message is absent.)
        adata = _make_2cluster_adata(n_per=15)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tsne(adata, seed=0)
        small_n_msgs = [str(w.message) for w in caught if "< recommended" in str(w.message)]
        assert not small_n_msgs, small_n_msgs

    def test_umap_warns_below_threshold(self):
        # n_samples = 10 < MIN_RECOMMENDED_N_UMAP (20)
        adata = _make_2cluster_adata(n_per=5)
        with pytest.warns(UserWarning, match=r"n_samples=10.*recommended minimum 20"):
            umap(adata, n_neighbors=3, seed=0)

    def test_umap_no_warning_at_threshold(self):
        # n_samples = 20 == MIN_RECOMMENDED_N_UMAP; small-n warning MUST NOT
        # fire.  (umap-learn emits its own unrelated UserWarnings; only
        # assert our small-n message is absent.)
        adata = _make_2cluster_adata(n_per=10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            umap(adata, seed=0)
        small_n_msgs = [str(w.message) for w in caught if "< recommended" in str(w.message)]
        assert not small_n_msgs, small_n_msgs

    def test_constants_exposed(self):
        # Users can override thresholds via module attribute; ensure
        # they're documented + exported.
        assert isinstance(MIN_RECOMMENDED_N_TSNE, int)
        assert isinstance(MIN_RECOMMENDED_N_UMAP, int)
        assert MIN_RECOMMENDED_N_TSNE >= MIN_RECOMMENDED_N_UMAP  # empirical ordering


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


class TestAccessors:
    def test_get_tsne_dataframe_shape_and_metadata(self):
        adata = _make_2cluster_adata()
        tsne(adata, perplexity=8, seed=42)
        df = get_tsne_dataframe(adata)
        assert df.shape == (40, 3)  # tSNE1 + tSNE2 + group
        assert list(df.columns) == ["tSNE1", "tSNE2", "group"]
        assert "perplexity" in df.attrs

    def test_get_tsne_dataframe_no_obs(self):
        adata = _make_2cluster_adata()
        tsne(adata, perplexity=8, seed=42)
        df = get_tsne_dataframe(adata, include_obs=False)
        assert list(df.columns) == ["tSNE1", "tSNE2"]

    def test_get_tsne_dataframe_missing_obsm_raises(self):
        adata = _make_2cluster_adata()
        with pytest.raises(KeyError, match="X_tsne"):
            get_tsne_dataframe(adata)

    def test_get_umap_dataframe_shape_and_metadata(self):
        adata = _make_2cluster_adata()
        umap(adata, n_neighbors=5, seed=42)
        df = get_umap_dataframe(adata)
        assert df.shape == (40, 3)  # UMAP1 + UMAP2 + group
        assert list(df.columns) == ["UMAP1", "UMAP2", "group"]
        assert "n_neighbors" in df.attrs

    def test_get_umap_dataframe_missing_obsm_raises(self):
        adata = _make_2cluster_adata()
        with pytest.raises(KeyError, match="X_umap"):
            get_umap_dataframe(adata)
