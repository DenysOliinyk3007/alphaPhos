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
from alphaphos.stats.diff_exp import (
    _HAS_STATS_DEPS,
    DEFAULT_STATS_SETTINGS,
    diff_exp_limma,
)

pytestmark = pytest.mark.skipif(
    not _HAS_STATS_DEPS,
    reason="inmoose + patsy not installed; run `pip install alphaPhos[stats]`.",
)

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


# ============================================================================
# Level sanitisation: inmoose.makeContrasts eval()s the contrast string, so
# level names must be valid Python identifiers.  diff_exp_limma sanitises
# non-identifier characters internally; the user's original labels must still
# round-trip through result.attrs and the comparison= argument.
# ============================================================================


def _relabel_adata(adata: ad.AnnData, mapping: dict[str, str]) -> ad.AnnData:
    """Return a copy with adata.obs["condition"] values remapped."""
    adata = adata.copy()
    adata.obs["condition"] = adata.obs["condition"].astype(str).map(mapping)
    return adata


class TestNonIdentifierConditionLevels:
    def test_plus_minus_labels(self, two_group_adata):
        """The exact case that triggered the bug: EGF+ / EGF-."""
        adata = _relabel_adata(two_group_adata, {"ctrl": "EGF-", "trt": "EGF+"})
        res = diff_exp_limma(
            adata,
            condition_column="condition",
            comparison=("EGF+", "EGF-"),
        )
        # Sanity: spiked hits still found; original labels round-trip via attrs
        assert len(res) == adata.n_vars
        assert (res.loc[[f"site_{i:02d}" for i in range(TRUE_HITS)], "fdr"] < 0.05).all()
        assert res.attrs["treatment"] == "EGF+"
        assert res.attrs["control"] == "EGF-"
        assert "EGF+" in res.attrs["contrast_direction"]
        assert "EGF+" in res.attrs["contrast_string"]

    def test_digit_start_labels(self, two_group_adata):
        """1uM / 10uM: valid category label, invalid Python identifier."""
        adata = _relabel_adata(two_group_adata, {"ctrl": "1uM", "trt": "10uM"})
        res = diff_exp_limma(
            adata,
            condition_column="condition",
            comparison=("10uM", "1uM"),
        )
        assert (res.loc[[f"site_{i:02d}" for i in range(TRUE_HITS)], "fdr"] < 0.05).all()
        assert res.attrs["treatment"] == "10uM"
        assert res.attrs["control"] == "1uM"

    def test_space_in_labels(self, two_group_adata):
        adata = _relabel_adata(two_group_adata, {"ctrl": "not treated", "trt": "treated"})
        res = diff_exp_limma(
            adata,
            condition_column="condition",
            comparison=("treated", "not treated"),
        )
        assert (res.loc[[f"site_{i:02d}" for i in range(TRUE_HITS)], "fdr"] < 0.05).all()
        assert res.attrs["control"] == "not treated"

    def test_slash_in_labels(self, two_group_adata):
        adata = _relabel_adata(two_group_adata, {"ctrl": "WT/WT", "trt": "KO/WT"})
        res = diff_exp_limma(
            adata,
            condition_column="condition",
            comparison=("KO/WT", "WT/WT"),
        )
        assert (res.loc[[f"site_{i:02d}" for i in range(TRUE_HITS)], "fdr"] < 0.05).all()

    def test_collision_disambiguated(self, two_group_adata):
        """EGF+ and EGF- both sanitize to EGF_ -- collision must not cause a
        silent design-matrix mixup."""
        adata = _relabel_adata(two_group_adata, {"ctrl": "EGF-", "trt": "EGF+"})
        res = diff_exp_limma(
            adata,
            condition_column="condition",
            comparison=("EGF+", "EGF-"),
        )
        # log2fc for spiked sites should be positive (mean(EGF+) > mean(EGF-)),
        # which is only true if the collision was resolved bijectively.
        assert (res.loc[[f"site_{i:02d}" for i in range(TRUE_HITS)], "log2fc"] > 2.0).all()

    def test_covariate_with_special_chars(self, two_group_adata):
        """Categorical covariate levels also need sanitising -- they end up
        in the design matrix column names too."""
        adata = two_group_adata.copy()
        # Replace the default batch labels b1/b2 with special-char ones
        adata.obs["batch"] = adata.obs["batch"].astype(str).map({"b1": "run-1", "b2": "run-2"})
        # Should not raise on the design build (covariate levels go through
        # patsy too, and both the "-" and the collision handling need to work)
        res = diff_exp_limma(
            adata,
            condition_column="condition",
            comparison=("trt", "ctrl"),
            covariates=["batch"],
        )
        assert len(res) == adata.n_vars

    def test_sign_convention_still_holds_with_special_chars(self, two_group_adata):
        """After sanitisation, log2fc = mean(treatment) - mean(control) --
        positive for the spiked sites which are up in ``trt`` a.k.a. ``EGF+``."""
        adata = _relabel_adata(two_group_adata, {"ctrl": "EGF-", "trt": "EGF+"})
        res = diff_exp_limma(
            adata,
            condition_column="condition",
            comparison=("EGF+", "EGF-"),
        )
        assert (res.loc[[f"site_{i:02d}" for i in range(TRUE_HITS)], "log2fc"] > 2.0).all()


class TestSanitizeHelpers:
    """Direct tests of the module-private sanitisation helpers."""

    def test_sanitize_level_replaces_common_operators(self):
        from alphaphos.stats.diff_exp import _sanitize_level

        assert _sanitize_level("EGF+") == "EGF_"
        assert _sanitize_level("EGF-") == "EGF_"
        assert _sanitize_level("KO/WT") == "KO_WT"
        assert _sanitize_level("not treated") == "not_treated"
        assert _sanitize_level("a.b") == "a_b"

    def test_sanitize_level_prefixes_digit_start(self):
        from alphaphos.stats.diff_exp import _sanitize_level

        assert _sanitize_level("1uM") == "_1uM"
        assert _sanitize_level("10uM") == "_10uM"

    def test_sanitize_level_empty_becomes_underscore(self):
        from alphaphos.stats.diff_exp import _sanitize_level

        assert _sanitize_level("") == "_"

    def test_sanitize_map_is_bijective_under_collision(self):
        from alphaphos.stats.diff_exp import _sanitize_and_map_levels

        m = _sanitize_and_map_levels(["EGF+", "EGF-", "EGF*"])
        assert len(set(m.values())) == 3  # bijective
        assert m["EGF+"] == "EGF_"
        assert m["EGF-"] == "EGF__2"
        assert m["EGF*"] == "EGF__3"

    def test_sanitize_map_preserves_first_seen_order(self):
        from alphaphos.stats.diff_exp import _sanitize_and_map_levels

        m = _sanitize_and_map_levels(["A-B", "A_B"])
        assert m["A-B"] == "A_B"
        assert m["A_B"] == "A_B_2"
