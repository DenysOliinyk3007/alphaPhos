"""Tests for :func:`alphaphos.wilson_threshold_sensitivity`."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphaphos import wilson_threshold_sensitivity
from alphaphos.constants import VAR_CLASSI_WILSON_LB
from alphaphos.preprocess.filter import DEFAULT_SENSITIVITY_THRESHOLDS


def _adata(n_samples: int, n_sites: int, seed: int = 0) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.normal(20, 2, size=(n_samples, n_sites)).astype(np.float32)
    X[rng.random(X.shape) < 0.3] = np.nan
    var = pd.DataFrame(
        {
            VAR_CLASSI_WILSON_LB: rng.uniform(0.0, 1.0, size=n_sites),
            "n_samples_detected": rng.integers(1, n_samples + 1, size=n_sites),
        },
        index=[f"site_{i}" for i in range(n_sites)],
    )
    return ad.AnnData(X=X, obs=pd.DataFrame(index=[f"s{i}" for i in range(n_samples)]), var=var)


class TestSensitivityBasic:
    def test_missing_column_raises(self):
        adata = ad.AnnData(
            X=np.zeros((5, 3)),
            obs=pd.DataFrame(index=[f"s{i}" for i in range(5)]),
            var=pd.DataFrame(index=[f"v{i}" for i in range(3)]),
        )
        with pytest.raises(KeyError, match=VAR_CLASSI_WILSON_LB):
            wilson_threshold_sensitivity(adata)

    def test_returns_dataframe_with_expected_columns(self):
        df = wilson_threshold_sensitivity(_adata(200, 500))
        assert set(df.columns) >= {
            "threshold",
            "n_sites",
            "pct_kept",
            "pct_nan",
            "median_sd",
            "median_n_det",
            "vs_ref_delta_pct",
        }
        assert df.attrs.get("total_sites") == 500

    def test_default_thresholds(self):
        df = wilson_threshold_sensitivity(_adata(200, 500))
        assert list(df["threshold"]) == list(DEFAULT_SENSITIVITY_THRESHOLDS)

    def test_custom_thresholds_honoured(self):
        thresholds = [0.1, 0.35, 0.72]
        df = wilson_threshold_sensitivity(_adata(200, 500), thresholds=thresholds)
        assert list(df["threshold"]) == thresholds

    def test_retention_monotonic_nonincreasing_in_threshold(self):
        df = wilson_threshold_sensitivity(_adata(300, 800))
        assert all(
            df["n_sites"].to_numpy()[i] >= df["n_sites"].to_numpy()[i + 1]
            for i in range(len(df) - 1)
        )

    def test_median_n_det_monotonic_nondecreasing(self):
        # Because wilson_lb grows with n_det (for high purity), tighter
        # thresholds should retain sites with larger n_det on average.
        adata = _adata(500, 2000, seed=3)
        df = wilson_threshold_sensitivity(adata)
        # At extreme threshold = 0.75 we may have too few sites; drop empty rows
        df_ok = df[df["n_sites"] > 0].reset_index(drop=True)
        # Median n_det should non-strictly increase (with some noise on synthetic
        # random data; only assert first and last row).
        assert df_ok["median_n_det"].iloc[-1] >= df_ok["median_n_det"].iloc[0]

    def test_reference_delta_column_zero_at_reference_row(self):
        df = wilson_threshold_sensitivity(_adata(200, 500), reference_threshold=0.5)
        ref_row = df[df["threshold"] == 0.5]
        assert not ref_row.empty
        assert ref_row["vs_ref_delta_pct"].iloc[0] == pytest.approx(0.0)

    def test_reference_delta_matches_manual(self):
        df = wilson_threshold_sensitivity(_adata(200, 500), reference_threshold=0.5)
        ref_n = df.loc[df["threshold"] == 0.5, "n_sites"].iloc[0]
        for _, row in df.iterrows():
            expected = (row["n_sites"] - ref_n) / ref_n * 100.0
            assert row["vs_ref_delta_pct"] == pytest.approx(expected, abs=1e-9)


class TestSensitivityEdgeCases:
    def test_all_sites_have_lb_zero(self):
        # Every site's wilson_lb == 0 → threshold=0 keeps all, others keep 0
        adata = _adata(100, 200)
        adata.var[VAR_CLASSI_WILSON_LB] = 0.0
        df = wilson_threshold_sensitivity(adata, thresholds=[0.0, 0.1, 0.5])
        assert df.loc[df["threshold"] == 0.0, "n_sites"].iloc[0] == 200
        assert df.loc[df["threshold"] == 0.1, "n_sites"].iloc[0] == 0

    def test_empty_sites_returns_empty_frame_gracefully(self):
        obs = pd.DataFrame(index=["s0"])
        var = pd.DataFrame(
            {
                VAR_CLASSI_WILSON_LB: pd.Series([], dtype=float),
                "n_samples_detected": pd.Series([], dtype=int),
            },
        )
        adata = ad.AnnData(
            X=np.zeros((1, 0), dtype=np.float32),
            obs=obs,
            var=var,
        )
        df = wilson_threshold_sensitivity(adata, thresholds=[0.5])
        assert (df["n_sites"] == 0).all()

    def test_bad_thresholds_raise(self):
        with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
            wilson_threshold_sensitivity(_adata(50, 100), thresholds=[-0.1, 0.5])
        with pytest.raises(ValueError, match="non-empty"):
            wilson_threshold_sensitivity(_adata(50, 100), thresholds=[])
