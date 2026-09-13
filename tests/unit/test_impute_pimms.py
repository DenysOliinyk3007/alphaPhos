"""Tests for the PIMMS wrapper (ap.impute_pimms).

Skipped when pimms-learn or torch aren't installed -- keeps the core
test suite runnable on a bare install.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("anndata")
pytest.importorskip("torch")
pytest.importorskip("pimmslearn")

import anndata as ad

from alphaphos.preprocess.impute_pimms import (
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_LATENT_DIM,
    MIN_RECOMMENDED_SAMPLES,
    impute_pimms,
)


def _make_adata(
    n_samples: int = 30, n_features: int = 200, mcar: float = 0.15, seed: int = 0
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    mu = rng.normal(20.0, 1.5, size=n_features)
    X = rng.normal(mu, 1.0, size=(n_samples, n_features)).astype(float)
    mask = rng.random(X.shape) < mcar
    X[mask] = np.nan
    obs = pd.DataFrame(
        {"condition": ["A"] * (n_samples // 2) + ["B"] * (n_samples - n_samples // 2)},
        index=[f"s{i:02d}" for i in range(n_samples)],
    )
    var = pd.DataFrame(index=[f"f{i:03d}" for i in range(n_features)])
    a = ad.AnnData(X=X, obs=obs, var=var)
    a.layers["intensity_log2"] = X.copy()
    return a


class TestImputePimms:
    def test_vae_fills_all_nans_and_populates_is_imputed(self):
        adata = _make_adata(n_samples=30, n_features=200, seed=0)
        n_nan_before = int(np.isnan(adata.X).sum())
        assert n_nan_before > 0

        out = impute_pimms(adata, model="VAE", epochs_max=10, seed=0)
        assert int(np.isnan(out.X).sum()) == 0
        assert "is_imputed" in out.layers
        n_imputed = int(out.layers["is_imputed"].sum())
        assert n_imputed == n_nan_before

    def test_provenance_stamped(self):
        adata = _make_adata(n_samples=30, n_features=100, seed=1)
        out = impute_pimms(adata, model="VAE", epochs_max=5, seed=42)
        prov = out.uns["alphaphos"]["pimms_impute"]
        assert prov["method"] == "pimms"
        assert prov["pimms_model"] == "VAE"
        assert prov["latent_dim"] == DEFAULT_LATENT_DIM
        assert list(prov["hidden_layers"]) == list(DEFAULT_HIDDEN_LAYERS)
        assert prov["seed"] == 42
        assert prov["paper_doi"] == "10.1038/s41467-024-48711-5"

    def test_dae_also_runs(self):
        adata = _make_adata(n_samples=30, n_features=100, seed=2)
        out = impute_pimms(adata, model="DAE", epochs_max=5)
        assert int(np.isnan(out.X).sum()) == 0
        assert out.uns["alphaphos"]["pimms_impute"]["pimms_model"] == "DAE"

    def test_small_n_triggers_warning(self):
        adata = _make_adata(n_samples=8, n_features=100, seed=3)
        assert adata.n_obs < MIN_RECOMMENDED_SAMPLES
        with pytest.warns(UserWarning, match="benchmarked at n_samples"):
            impute_pimms(adata, model="VAE", epochs_max=5)

    def test_large_n_no_small_sample_warning(self):
        # n=60 is above the paper's ~50-sample floor -- our small-n
        # UserWarning must NOT fire.  (fastai/fastprogress emit their
        # own unrelated warnings during training; we only care that
        # ours is absent.)
        adata = _make_adata(n_samples=60, n_features=100, seed=4)
        assert adata.n_obs >= MIN_RECOMMENDED_SAMPLES
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            impute_pimms(adata, model="VAE", epochs_max=5)
        for w in caught:
            assert "PIMMS was benchmarked" not in str(w.message), (
                f"unexpected small-n warning: {w.message}"
            )

    def test_bad_model_raises(self):
        adata = _make_adata(n_samples=30, seed=5)
        with pytest.raises(ValueError, match="model must be"):
            impute_pimms(adata, model="XGBoost")  # type: ignore[arg-type]

    def test_cf_not_yet_supported(self):
        adata = _make_adata(n_samples=30, seed=6)
        with pytest.raises(NotImplementedError, match="CF"):
            impute_pimms(adata, model="CF")

    def test_in_place_by_default_copy_opt_in(self):
        # Same contract as impute_hybrid / impute_knn_site_based.
        adata = _make_adata(n_samples=30, seed=7)
        out = impute_pimms(adata, model="VAE", epochs_max=3)
        assert out is adata
        assert int(np.isnan(adata.X).sum()) == 0

        adata2 = _make_adata(n_samples=30, seed=7)
        n_before = int(np.isnan(adata2.X).sum())
        out2 = impute_pimms(adata2, model="VAE", epochs_max=3, copy=True)
        assert out2 is not adata2
        assert int(np.isnan(adata2.X).sum()) == n_before  # original unchanged

    def test_missing_layer_raises(self):
        adata = _make_adata(n_samples=30, seed=8)
        with pytest.raises(KeyError, match="not in adata.layers"):
            impute_pimms(adata, model="VAE", epochs_max=3, layer="does_not_exist")
