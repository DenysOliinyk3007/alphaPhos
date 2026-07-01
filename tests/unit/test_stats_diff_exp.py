"""Unit tests for :mod:`alphaphos.stats.diff_exp`.

Fixture: 40 sites x 12 samples (6 ctrl, 6 trt), log2-scale, heterogeneous
per-site variance (so limma's empirical Bayes prior is finite and the
inmoose df_prior=inf edge case does not fire). First 10 sites are
up-regulated in treatment by 3 log2 units.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphaphos.constants import LAYER_INTENSITY_LOG2
from alphaphos.stats.diff_exp import DEFAULT_STATS_SETTINGS, diff_exp_limma

TRUE_HITS = 10  # first N sites are spiked


@pytest.fixture
def two_group_adata():
    rng = np.random.default_rng(0)
    n_sites, n_samples_per_group = 40, 6
    n_samples = 2 * n_samples_per_group
    # Modest heterogeneous variance -- large enough that limma's empirical
    # Bayes prior is finite (no df_prior=inf edge case), small enough that
    # the +3 log2fc spike is reliably detectable in 6-vs-6.
    per_site_sd = rng.uniform(0.3, 0.8, n_sites)
    X = rng.normal(20.0, per_site_sd[None, :], size=(n_samples, n_sites))
    X[n_samples_per_group:, :TRUE_HITS] += 3.0  # trt samples get +3 log2fc on first 10 sites

    obs = pd.DataFrame(
        {
            "condition": ["ctrl"] * n_samples_per_group + ["trt"] * n_samples_per_group,
            "batch": (["b1"] * 3 + ["b2"] * 3) * 2,
        },
        index=[f"s{i}" for i in range(n_samples)],
    )
    var = pd.DataFrame(index=[f"site_{i:02d}" for i in range(n_sites)])
    adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)
    adata.layers[LAYER_INTENSITY_LOG2] = adata.X.copy()
    return adata


class TestBasicTwoGroup:
    def test_finds_spiked_hits(self, two_group_adata):
        res = diff_exp_limma(
            two_group_adata,
            condition_column="condition",
            comparison=("trt", "ctrl"),
        )
        assert len(res) == two_group_adata.n_vars
        expected = [f"site_{i:02d}" for i in range(TRUE_HITS)]
        # All 10 spiked sites cross FDR<0.05.
        assert (res.loc[expected, "fdr"] < 0.05).all()
        # log2fc is positive for spiked sites (mean(trt) - mean(ctrl))
        assert (res.loc[expected, "log2fc"] > 2.0).all()
        # The Top-10 by FDR is dominated by true positives (>=8 / 10).
        top = res.sort_values("fdr").head(TRUE_HITS).index.tolist()
        assert len(set(top) & set(expected)) >= 8

    def test_result_columns_and_order(self, two_group_adata):
        res = diff_exp_limma(
            two_group_adata,
            condition_column="condition",
            comparison=("trt", "ctrl"),
        )
        assert list(res.columns) == ["log2fc", "se", "t_stat", "p_value", "fdr", "B", "ave_expr"]

    def test_index_preserves_input_order(self, two_group_adata):
        res = diff_exp_limma(
            two_group_adata,
            condition_column="condition",
            comparison=("trt", "ctrl"),
        )
        assert list(res.index) == list(two_group_adata.var_names)

    def test_contrast_direction_attr(self, two_group_adata):
        res = diff_exp_limma(
            two_group_adata,
            condition_column="condition",
            comparison=("trt", "ctrl"),
        )
        assert res.attrs["treatment"] == "trt"
        assert res.attrs["control"] == "ctrl"
        assert "mean(trt) - mean(ctrl)" in res.attrs["contrast_direction"]
        assert res.attrs["contrast_string"] == "condition[trt]-condition[ctrl]"

    def test_swapping_comparison_flips_sign(self, two_group_adata):
        forward = diff_exp_limma(
            two_group_adata,
            condition_column="condition",
            comparison=("trt", "ctrl"),
        )
        backward = diff_exp_limma(
            two_group_adata,
            condition_column="condition",
            comparison=("ctrl", "trt"),
        )
        np.testing.assert_allclose(forward["log2fc"].values, -backward["log2fc"].values, rtol=1e-5)
        # p-values are symmetric under sign flip
        np.testing.assert_allclose(forward["p_value"].values, backward["p_value"].values, rtol=1e-5)

    def test_default_layer_is_intensity_log2(self, two_group_adata):
        # Corrupt .X so it would fail the log-scale check if it were used;
        # a successful run proves the default layer='intensity_log2' is honored.
        two_group_adata.X = (two_group_adata.X.astype(np.float64) * 1e6).astype(np.float32)
        res = diff_exp_limma(
            two_group_adata,
            condition_column="condition",
            comparison=("trt", "ctrl"),
        )
        assert len(res) == two_group_adata.n_vars

    def test_layer_none_uses_X(self, two_group_adata):
        del two_group_adata.layers[LAYER_INTENSITY_LOG2]
        res = diff_exp_limma(
            two_group_adata,
            condition_column="condition",
            comparison=("trt", "ctrl"),
            layer=None,
        )
        assert len(res) == two_group_adata.n_vars


class TestCovariates:
    def test_batch_covariate_runs(self, two_group_adata):
        # Add a real batch effect so adjustment matters.
        two_group_adata.layers[LAYER_INTENSITY_LOG2] = two_group_adata.layers[
            LAYER_INTENSITY_LOG2
        ].copy()
        is_b2 = (two_group_adata.obs["batch"] == "b2").to_numpy()
        two_group_adata.layers[LAYER_INTENSITY_LOG2][is_b2] += 1.5
        res = diff_exp_limma(
            two_group_adata,
            condition_column="condition",
            comparison=("trt", "ctrl"),
            covariates=["batch"],
        )
        assert len(res) == two_group_adata.n_vars

    def test_covariate_equal_to_condition_raises(self, two_group_adata):
        with pytest.raises(ValueError, match="cannot self-adjust"):
            diff_exp_limma(
                two_group_adata,
                condition_column="condition",
                comparison=("trt", "ctrl"),
                covariates=["condition"],
            )

    def test_missing_covariate_raises(self, two_group_adata):
        with pytest.raises(KeyError, match="not in adata.obs"):
            diff_exp_limma(
                two_group_adata,
                condition_column="condition",
                comparison=("trt", "ctrl"),
                covariates=["nope"],
            )


class TestValidation:
    def test_missing_condition_column(self, two_group_adata):
        with pytest.raises(KeyError, match="condition_column"):
            diff_exp_limma(
                two_group_adata,
                condition_column="nope",
                comparison=("trt", "ctrl"),
            )

    def test_treatment_not_present(self, two_group_adata):
        with pytest.raises(ValueError, match="treatment"):
            diff_exp_limma(
                two_group_adata,
                condition_column="condition",
                comparison=("nope", "ctrl"),
            )

    def test_control_not_present(self, two_group_adata):
        with pytest.raises(ValueError, match="control"):
            diff_exp_limma(
                two_group_adata,
                condition_column="condition",
                comparison=("trt", "nope"),
            )

    def test_same_treatment_and_control(self, two_group_adata):
        with pytest.raises(ValueError, match="different levels"):
            diff_exp_limma(
                two_group_adata,
                condition_column="condition",
                comparison=("trt", "trt"),
            )

    def test_nan_rejected(self, two_group_adata):
        two_group_adata.layers[LAYER_INTENSITY_LOG2][0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            diff_exp_limma(
                two_group_adata,
                condition_column="condition",
                comparison=("trt", "ctrl"),
            )

    def test_linear_scale_rejected(self, two_group_adata):
        # 2**20 ~ 1e6 -- linear-scale MS intensity range
        two_group_adata.layers[LAYER_INTENSITY_LOG2] = (
            2.0 ** two_group_adata.layers[LAYER_INTENSITY_LOG2]
        )
        with pytest.raises(ValueError, match="linear-scale"):
            diff_exp_limma(
                two_group_adata,
                condition_column="condition",
                comparison=("trt", "ctrl"),
            )

    def test_duplicate_var_names_rejected(self, two_group_adata):
        two_group_adata.var_names = ["site_0"] * two_group_adata.n_vars
        with pytest.raises(ValueError, match="unique"):
            diff_exp_limma(
                two_group_adata,
                condition_column="condition",
                comparison=("trt", "ctrl"),
            )

    def test_missing_layer_raises(self, two_group_adata):
        with pytest.raises(KeyError, match="layer"):
            diff_exp_limma(
                two_group_adata,
                condition_column="condition",
                comparison=("trt", "ctrl"),
                layer="does_not_exist",
            )

    def test_unknown_advanced_key(self, two_group_adata):
        with pytest.raises(ValueError, match="Unknown advanced keys"):
            diff_exp_limma(
                two_group_adata,
                condition_column="condition",
                comparison=("trt", "ctrl"),
                advanced={"nonsense_flag": True},
            )


class TestDeterminism:
    def test_repeated_runs_identical(self, two_group_adata):
        r1 = diff_exp_limma(
            two_group_adata,
            condition_column="condition",
            comparison=("trt", "ctrl"),
        )
        r2 = diff_exp_limma(
            two_group_adata,
            condition_column="condition",
            comparison=("trt", "ctrl"),
        )
        pd.testing.assert_frame_equal(r1, r2)


class TestSettings:
    def test_default_settings_shape(self):
        assert set(DEFAULT_STATS_SETTINGS) == {"trend", "robust", "winsor_tail_p"}
        assert DEFAULT_STATS_SETTINGS["trend"] is False
        assert DEFAULT_STATS_SETTINGS["robust"] is False
