"""Regression tests from the 2026-09-13 review of :mod:`alphaphos.stats`.

Each class pins one finding.  The inmoose-backed cases are skipped when the
``[stats]`` extra is absent; the clean-room cases always run.
"""

from __future__ import annotations

import logging

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import f as f_dist

from alphaphos.stats.design import covariate_is_continuous, design_matrix
from alphaphos.stats.diff_exp import (
    _HAS_STATS_DEPS,
    _resolve_stats_settings,
    diff_exp_anova,
    diff_exp_limma_contrasts,
)
from alphaphos.stats.linear_model import lm_fit, moderated_f_test
from alphaphos.stats.moderated import fit_f_dist

_needs_inmoose = pytest.mark.skipif(not _HAS_STATS_DEPS, reason="inmoose + patsy not installed")


def _adata(n_per_group=4, groups=("A", "B", "C"), p=300, seed=0, **obs_cols) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    n = n_per_group * len(groups)
    sd = np.exp(rng.normal(-1, 0.6, p))  # heterogeneous variance -> finite prior df
    X = rng.normal(20, 1, (1, p)) + rng.normal(0, 1, (n, p)) * sd
    cond = np.repeat(list(groups), n_per_group)
    X[cond == groups[1]] += np.where(rng.random(p) < 0.1, 1.5, 0.0)
    obs = pd.DataFrame({"condition": cond, **obs_cols}, index=[f"s{i}" for i in range(n)])
    a = ad.AnnData(X=X.copy(), obs=obs, var=pd.DataFrame(index=[f"f{i}" for i in range(p)]))
    a.layers["intensity_log2"] = X.copy()
    return a


# ---------------------------------------------------------------------------
# A1 -- integer-coded covariates must not be fit as a linear trend
# ---------------------------------------------------------------------------


class TestIntegerCovariates:
    def test_helper_rules(self):
        assert covariate_is_continuous(pd.Series([0.1, 0.2]), name="x") is True
        assert covariate_is_continuous(pd.Series(["a", "b"]), name="x") is False
        assert covariate_is_continuous(pd.Series(["a", "b"]).astype("category"), name="x") is False
        for bad in (pd.Series([1, 2, 3]), pd.Series([True, False])):
            with pytest.raises(ValueError, match="ambiguous"):
                covariate_is_continuous(bad, name="batch")

    def test_design_matrix_rejects_int_batch(self):
        a = _adata(batch=[1, 2, 3, 1] * 3)
        with pytest.raises(ValueError, match="batch.*astype\\(str\\)"):
            design_matrix(a, condition_column="condition", covariates=["batch"])

    def test_design_matrix_str_batch_is_dummy_coded_and_float_is_continuous(self):
        a = _adata(batch=[1, 2, 3, 1] * 3)
        a.obs["batch_s"] = a.obs["batch"].astype(str)
        a.obs["age"] = np.linspace(20.0, 60.0, a.n_obs)
        dm = design_matrix(a, condition_column="condition", covariates=["batch_s", "age"])
        assert [c for c in dm.frame.columns if c.startswith("batch_s")] == [
            "batch_s[x_2]",
            "batch_s[x_3]",
        ]
        assert "age" in dm.frame.columns

    @_needs_inmoose
    def test_diff_exp_limma_rejects_int_batch(self):
        import alphaphos as ap

        a = _adata(batch=[1, 2, 3, 1] * 3)
        with pytest.raises(ValueError, match="ambiguous"):
            ap.diff_exp_limma(
                a, condition_column="condition", comparison=("B", "A"), covariates=["batch"]
            )


# ---------------------------------------------------------------------------
# A3 -- joint / ANOVA paths carry the same guards as diff_exp_limma
# ---------------------------------------------------------------------------


class TestJointPathGuards:
    def test_inmoose_only_advanced_keys_raise_on_joint_path(self):
        a = _adata()
        with pytest.raises(ValueError, match="joint=False"):
            diff_exp_limma_contrasts(
                a,
                condition_column="condition",
                contrasts={"c": ("B", "A")},
                advanced={"winsor_tail_p": (0.01, 0.05)},
            )
        # trend IS a clean-room setting and must be accepted.
        r = diff_exp_limma_contrasts(
            a, condition_column="condition", contrasts={"c": ("B", "A")}, advanced={"trend": False}
        )["c"]
        assert r.attrs["trend"] is False

    def test_robust_is_rejected_everywhere(self):
        with pytest.raises(NotImplementedError, match="robust"):
            _resolve_stats_settings({"robust": True})

    def test_duplicate_var_names_rejected(self):
        a = _adata()
        a.var_names = ["dup", "dup", *list(a.var_names[2:])]
        with pytest.raises(ValueError, match="unique"):
            diff_exp_limma_contrasts(a, condition_column="condition", contrasts={"c": ("B", "A")})
        with pytest.raises(ValueError, match="unique"):
            diff_exp_anova(a, condition_column="condition")

    def test_double_batch_guard_fires_on_joint_and_anova(self):
        a = _adata(batch=["x", "y", "z", "x"] * 3)
        a.uns["alphaphos"] = {
            "batch_correction": {"layer": "intensity_log2", "batch_column": "batch"}
        }
        with pytest.raises(ValueError, match="Double batch correction"):
            diff_exp_limma_contrasts(
                a, condition_column="condition", contrasts={"c": ("B", "A")}, covariates=["batch"]
            )
        with pytest.raises(ValueError, match="Double batch correction"):
            diff_exp_anova(a, condition_column="condition", covariates=["batch"])
        # layer=None tests .X, which mirrors the corrected canonical layer.
        with pytest.raises(ValueError, match="Double batch correction"):
            diff_exp_anova(a, condition_column="condition", covariates=["batch"], layer=None)

    def test_nan_condition_label_rejected(self):
        a = _adata()
        a.obs["condition"] = a.obs["condition"].astype(object)
        a.obs.iloc[0, a.obs.columns.get_loc("condition")] = np.nan
        with pytest.raises(ValueError, match="missing value"):
            diff_exp_anova(a, condition_column="condition")

    def test_small_group_warning_on_joint_path(self, caplog):
        a = _adata(n_per_group=2)
        with caplog.at_level(logging.WARNING, logger="alphaphos.stats.diff_exp"):
            diff_exp_limma_contrasts(a, condition_column="condition", contrasts={"c": ("B", "A")})
        assert any("Small groups" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# A4 / C5 -- inmoose path: string-compared levels, confounded covariate message
# ---------------------------------------------------------------------------


@_needs_inmoose
class TestInmoosePathRobustness:
    def test_integer_coded_condition_levels_work(self):
        import alphaphos as ap

        a = _adata(groups=(1, 2, 3))  # int-coded condition column
        r = ap.diff_exp_limma(a, condition_column="condition", comparison=("2", "1"))
        assert len(r) == a.n_vars
        manual = a.X[a.obs["condition"] == 2].mean(0) - a.X[a.obs["condition"] == 1].mean(0)
        np.testing.assert_allclose(r["log2fc"].to_numpy(), manual)
        r2 = ap.diff_exp_limma(a, condition_column="condition", comparison=(2, 1))
        np.testing.assert_array_equal(r2["log2fc"].to_numpy(), r["log2fc"].to_numpy())

    def test_confounded_covariate_gives_clear_error(self):
        import alphaphos as ap

        a = _adata()
        a.obs["site"] = a.obs["condition"].astype(str)  # perfectly confounded
        with pytest.raises(ValueError, match="rank-deficient"):
            ap.diff_exp_limma(
                a, condition_column="condition", comparison=("B", "A"), covariates=["site"]
            )

    def test_settings_type_validation(self):
        with pytest.raises(ValueError, match="trend must be bool"):
            _resolve_stats_settings({"trend": 1})
        with pytest.raises(ValueError, match="winsor_tail_p"):
            _resolve_stats_settings({"winsor_tail_p": (0.05,)})
        with pytest.raises(ValueError, match="winsor_tail_p"):
            _resolve_stats_settings({"winsor_tail_p": (0.05, 0.9)})
        assert _resolve_stats_settings({"winsor_tail_p": [0.05, 0.1]})["winsor_tail_p"] == (
            0.05,
            0.1,
        )


# ---------------------------------------------------------------------------
# C1 / C2 -- limma fidelity of the clean-room stack
# ---------------------------------------------------------------------------


class TestLimmaFidelity:
    def test_f_p_value_uses_moderated_df(self):
        # p = pf(F, rank, df_prior + df_residual) -- limma's definition.  (inmoose
        # 0.9.1 reports the chi-square limit here, which is anti-conservative.)
        a = _adata()
        r = diff_exp_anova(a, condition_column="condition")
        expected = f_dist.sf(r["F"].to_numpy(), r["df_between"].iloc[0], r["df_moderated"].iloc[0])
        np.testing.assert_allclose(r["p_value"].to_numpy(), expected, rtol=0, atol=1e-14)
        assert r["df_between"].iloc[0] == 2
        assert np.isfinite(r["df_moderated"].iloc[0])

    def test_redundant_contrasts_use_rank(self, caplog):
        a = _adata()
        dm = design_matrix(a, condition_column="condition")
        fit = lm_fit(a.X, dm.frame.to_numpy(), coefficient_labels=dm.coefficient_labels)
        C = np.array([[-1.0, 1.0, 0.0], [-2.0, 2.0, 0.0]]).T  # second column redundant
        with caplog.at_level(logging.WARNING, logger="alphaphos.stats.linear_model"):
            r_red = moderated_f_test(fit, C)
        r_one = moderated_f_test(fit, C[:, :1])
        assert r_red["df_between"].iloc[0] == 1
        np.testing.assert_allclose(r_red["F"].to_numpy(), r_one["F"].to_numpy(), rtol=1e-10)
        np.testing.assert_allclose(r_red["p_value"].to_numpy(), r_one["p_value"].to_numpy())
        assert any("rank" in rec.message for rec in caplog.records)

    def test_degenerate_prior_matches_current_limma(self):
        # Constant prior: pooled mean (limma >= 2017).  Trended prior: exp(trend).
        v = np.array([2.0, 2.0, 2.0, 2.0])
        prior = fit_f_dist(v, residual_df=6)
        assert not np.isfinite(prior.prior_df)
        assert prior.prior_variance == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# C4 -- observed-only wrapper imputes the layer it tests
# ---------------------------------------------------------------------------


@_needs_inmoose
class TestObservedOnlyLayerHandling:
    def _sparse(self):
        a = _adata(n_per_group=4, groups=("trt", "ctrl"))
        X = a.layers["intensity_log2"]
        X[0, :50] = np.nan  # sparse cells, still >= 3 observed per group
        a.layers["intensity_log2"] = X
        a.layers["other"] = X.copy()
        a.X = X.copy()
        return a

    def test_non_default_layer_is_imputed_and_tested(self):
        import alphaphos as ap

        a = self._sparse()
        res, on_off = ap.diff_exp_limma_observed_only(
            a, condition_column="condition", comparison=("trt", "ctrl"), layer="other"
        )
        assert len(res) == int((on_off["call"] == "both").sum()) == a.n_vars
        assert not res["imputed_driven"].any()

    def test_layer_none_with_imputer_rejected(self):
        import alphaphos as ap

        a = self._sparse()
        with pytest.raises(ValueError, match="layer=None"):
            ap.diff_exp_limma_observed_only(
                a, condition_column="condition", comparison=("trt", "ctrl"), layer=None
            )


# ---------------------------------------------------------------------------
# A2 -- recommend snippets call the real API
# ---------------------------------------------------------------------------


class TestRecommendSnippets:
    def test_de_snippets_use_real_keywords(self):
        from alphaphos.recommend import _decide_de

        for goal in ("onoff_discovery", "interaction_de", "marginal_de", "primary_de"):
            _name, _why, snippet = _decide_de(goal)
            assert "condition_col=" not in snippet
            assert "design=" not in snippet
            assert "condition_column=" in snippet


# ---------------------------------------------------------------------------
# limma-trend: clean-room trended prior must reproduce inmoose eBayes(trend=True)
# ---------------------------------------------------------------------------


def _trend_adata(n_per_group=4, groups=("A", "B", "C"), p=2000, seed=3) -> ad.AnnData:
    """Residual SD falls with intensity, as in real MS data."""
    rng = np.random.default_rng(seed)
    n = n_per_group * len(groups)
    mu = rng.uniform(12, 28, p)
    sd = 0.6 * np.exp(-0.12 * (mu - 12)) * np.exp(rng.normal(0, 0.3, p))
    cond = np.repeat(list(groups), n_per_group)
    X = mu + rng.normal(0, 1, (n, p)) * sd
    X[cond == groups[1]] += np.where(rng.random(p) < 0.1, 1.0, 0.0)
    obs = pd.DataFrame({"condition": cond}, index=[f"s{i}" for i in range(n)])
    a = ad.AnnData(X=X.copy(), obs=obs, var=pd.DataFrame(index=[f"f{i}" for i in range(p)]))
    a.layers["intensity_log2"] = X.copy()
    return a


class TestTrendPrior:
    def test_trended_prior_is_per_feature_and_follows_intensity(self):
        a = _trend_adata()
        r = diff_exp_limma_contrasts(a, condition_column="condition", contrasts={"c": ("B", "A")})[
            "c"
        ]
        pv = np.asarray(r.attrs["prior_variance"])
        assert r.attrs["trend"] is True
        assert pv.shape == (a.n_vars,)
        lo, hi = a.X.mean(0) < 16, a.X.mean(0) > 24
        assert np.median(pv[lo]) > 3 * np.median(pv[hi])  # variance decreases with intensity

    def test_constant_prior_when_trend_false(self):
        a = _trend_adata()
        r = diff_exp_limma_contrasts(
            a, condition_column="condition", contrasts={"c": ("B", "A")}, advanced={"trend": False}
        )["c"]
        assert r.attrs["trend"] is False
        assert np.ndim(r.attrs["prior_variance"]) == 0

    def test_small_n_spline_fallbacks(self):
        rng = np.random.default_rng(0)
        for n, expect_trend in ((2, True), (4, True), (10, True), (50, True)):
            v = np.exp(rng.normal(-1, 0.5, n))
            prior = fit_f_dist(v, residual_df=4, covariate=rng.uniform(10, 30, n))
            assert prior.trend is expect_trend
            assert np.shape(prior.prior_variance) == (n,)
            assert np.all(np.isfinite(prior.prior_variance))
        # constant covariate -> cannot support a trend -> constant prior broadcast
        prior = fit_f_dist(np.array([1.0, 2.0, 3.0]), residual_df=4, covariate=np.ones(3))
        assert np.allclose(prior.prior_variance, prior.prior_variance[0])

    @_needs_inmoose
    def test_matches_inmoose_trend_true_three_groups(self):
        import patsy
        from inmoose import limma

        a = _trend_adata()
        cr = diff_exp_limma_contrasts(
            a, condition_column="condition", contrasts={"BvA": ("B", "A"), "CvA": ("C", "A")}
        )
        an = diff_exp_anova(a, condition_column="condition")
        d = patsy.dmatrix("~0+condition", a.obs)
        f = limma.lmFit(a.X.T, d)
        lv = list(f.coefficients.columns)
        cm = limma.makeContrasts([f"{lv[1]}-{lv[0]}", f"{lv[2]}-{lv[0]}"], levels=lv)
        fc = limma.eBayes(limma.contrasts_fit(f, cm), trend=True)
        for i, key in enumerate(("BvA", "CvA")):
            tt = pd.DataFrame(
                limma.topTable(fc, coef=fc.coefficients.columns[i], number=np.inf)
            ).sort_index()
            np.testing.assert_allclose(
                tt["stat"].to_numpy(), cr[key]["t_stat"].to_numpy(), atol=1e-9
            )
            np.testing.assert_allclose(
                tt["pvalue"].to_numpy(), cr[key]["p_value"].to_numpy(), atol=1e-12
            )
        np.testing.assert_allclose(
            np.asarray(fc.s2_prior).ravel(),
            np.asarray(cr["BvA"].attrs["prior_variance"]),
            rtol=1e-8,
        )
        assert float(np.asarray(fc.df_prior).ravel()[0]) == pytest.approx(
            cr["BvA"].attrs["prior_df"], rel=1e-8
        )
        # F statistic agrees; the p-value deliberately does not (inmoose uses df2=inf).
        np.testing.assert_allclose(np.asarray(fc.F).ravel(), an["F"].to_numpy(), rtol=1e-9)
