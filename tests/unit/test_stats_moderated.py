"""Tests for :mod:`alphaphos.stats.moderated` and :mod:`.linear_model`.

Focus areas:
* Numerical correctness of ``_trigamma_inverse`` round-trip.
* ``fit_f_dist`` recovers the true prior on synthetic scaled-F data.
* ``lm_fit`` matches ``numpy.linalg.lstsq`` to machine precision.
* Moderated t on a 2-group design matches ``diff_exp_limma`` (inmoose)
  bit-for-bit.
* Moderated F on 2-group == moderated t^2 (Wald identity).
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.special import polygamma

from alphaphos.stats.diff_exp import _HAS_STATS_DEPS
from alphaphos.stats.linear_model import (
    contrasts_fit,
    lm_fit,
    moderated_f_test,
    moderated_t_test,
)
from alphaphos.stats.moderated import (
    EmpiricalBayesPrior,
    _trigamma_inverse,
    fit_f_dist,
    moderate_variance,
)

_needs_inmoose = pytest.mark.skipif(
    not _HAS_STATS_DEPS,
    reason="inmoose + patsy not installed; run `pip install alphaPhos[stats]`.",
)

# ---------------------------------------------------------------------------
# _trigamma_inverse round-trip
# ---------------------------------------------------------------------------


class TestTrigammaInverse:
    @pytest.mark.parametrize("x", [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 200.0])
    def test_round_trip_matches_input(self, x: float):
        y = float(polygamma(1, x))
        got = _trigamma_inverse(y)
        # brentq tolerance is 1e-10 relative; that maps to at most ~x*1e-9 error.
        assert abs(got - x) < max(1e-8, abs(x) * 1e-7), (x, got)

    def test_rejects_non_positive(self):
        with pytest.raises(ValueError, match="positive finite"):
            _trigamma_inverse(0.0)
        with pytest.raises(ValueError, match="positive finite"):
            _trigamma_inverse(-1.0)


# ---------------------------------------------------------------------------
# fit_f_dist: recover known prior from simulated scaled-F variances
# ---------------------------------------------------------------------------


class TestFitFDist:
    def test_recovers_synthetic_prior(self):
        # Draw per-feature variances from the scaled-F model with known
        # (s0^2, df0), then draw observed variances by chi-squared sampling
        # at df_residual.  With n=5000, moment estimator should be tight.
        rng = np.random.default_rng(42)
        n = 5000
        df_res = 12
        true_df0 = 8.0
        true_s0 = 2.5
        tau_sq = true_s0 * true_df0 / rng.chisquare(true_df0, size=n)
        observed = tau_sq * rng.chisquare(df_res, size=n) / df_res

        prior = fit_f_dist(observed, residual_df=df_res)
        assert prior.n_features_used == n
        # Recover s0^2 to within 5%, df0 within 10% (large-n asymptotics of
        # method of moments).
        assert abs(prior.prior_variance - true_s0) / true_s0 < 0.05
        assert abs(prior.prior_df - true_df0) / true_df0 < 0.10

    def test_degenerate_prior_when_variances_homogeneous(self):
        # All variances identical -> e_var <= 0 -> df0 = inf.
        n = 100
        variances = np.full(n, 1.5, dtype=np.float64)
        prior = fit_f_dist(variances, residual_df=10)
        assert not np.isfinite(prior.prior_df)
        # Current limma (post-Jan-2017 fitFDist): the degenerate constant prior
        # is the pooled variance mean(s^2) -- the MLE of the scale.
        assert prior.prior_variance == pytest.approx(1.5)

    def test_moderate_variance_shrinks_toward_prior(self):
        prior = EmpiricalBayesPrior(prior_variance=2.0, prior_df=5.0, n_features_used=100)
        variances = np.array([0.01, 2.0, 200.0])
        moderated, df_mod = moderate_variance(variances, residual_df=10, prior=prior)
        # Extreme variances are pulled toward 2.0; the middle stays put.
        assert moderated[0] > variances[0]
        assert moderated[2] < variances[2]
        assert abs(moderated[1] - 2.0) < 1e-9
        # All moderated df equal df0 + df_residual.
        assert np.all(df_mod == 15.0)


# ---------------------------------------------------------------------------
# lm_fit: matches numpy lstsq per-feature to machine precision
# ---------------------------------------------------------------------------


class TestLmFit:
    def test_coefficients_match_lstsq(self):
        rng = np.random.default_rng(0)
        n_samples, n_coef, n_features = 20, 3, 50
        design = rng.normal(0, 1, size=(n_samples, n_coef))
        Y = rng.normal(0, 1, size=(n_samples, n_features))

        fit = lm_fit(Y, design)
        # Compare against numpy lstsq per feature.
        beta_ref, _res, _rank, _sv = np.linalg.lstsq(design, Y, rcond=None)
        np.testing.assert_allclose(fit.coefficients, beta_ref.T, atol=1e-10)
        assert fit.df_residual == n_samples - n_coef

    def test_rejects_rank_deficient_design(self):
        Y = np.zeros((10, 5))
        # Column 3 duplicates column 0 -> rank deficient.
        design = np.zeros((10, 4))
        design[:, 0] = np.arange(10)
        design[:, 1] = np.arange(10) ** 2
        design[:, 2] = 1.0
        design[:, 3] = design[:, 0]  # collinear
        with pytest.raises(ValueError, match="rank-deficient"):
            lm_fit(Y, design)

    def test_cov_unscaled_equals_xtx_inv(self):
        rng = np.random.default_rng(1)
        design = rng.normal(0, 1, size=(15, 3))
        Y = rng.normal(0, 1, size=(15, 20))
        fit = lm_fit(Y, design)
        expected = np.linalg.inv(design.T @ design)
        np.testing.assert_allclose(fit.cov_unscaled, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Moderated t: bit-for-bit match against inmoose diff_exp_limma
# ---------------------------------------------------------------------------


def _make_two_group_adata(n=8, p=500, effect_features=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(10, 1, size=(n, p))
    X[: n // 2, :effect_features] += 1.5
    obs = pd.DataFrame(
        {"group": ["A"] * (n // 2) + ["B"] * (n - n // 2)}, index=[f"s{i}" for i in range(n)]
    )
    adata = ad.AnnData(
        X=X.astype(np.float64),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(p)]),
    )
    adata.layers["intensity_log2"] = adata.X.copy()
    return adata


@_needs_inmoose
class TestModeratedTMatchesInmoose:
    """Numerical parity check: the clean-room Smyth 2004 stack must match
    the inmoose (R-limma port) 2-group path to floating-point machine
    precision.  This is the strongest possible validation short of R
    itself."""

    def test_two_group_matches_diff_exp_limma(self):
        import alphaphos as ap

        adata = _make_two_group_adata()
        r_inmoose = ap.diff_exp_limma(adata, condition_column="group", comparison=("A", "B"))
        r_joint = ap.diff_exp_limma_contrasts(
            adata,
            condition_column="group",
            contrasts={"A_vs_B": ("A", "B")},
            joint=True,
        )["A_vs_B"]

        common = r_inmoose.index.intersection(r_joint.index)
        # log2fc is a pure linear-model coefficient -> bit-exact.
        np.testing.assert_array_equal(
            r_inmoose.loc[common, "log2fc"].to_numpy(),
            r_joint.loc[common, "log2fc"].to_numpy(),
        )
        # t and p differ only in the last-bit noise of the trigamma_inverse
        # solver + t.sf implementation.
        np.testing.assert_allclose(
            r_inmoose.loc[common, "t_stat"].to_numpy(),
            r_joint.loc[common, "t_stat"].to_numpy(),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            r_inmoose.loc[common, "p_value"].to_numpy(),
            r_joint.loc[common, "p_value"].to_numpy(),
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# Moderated F: 2-group F == t^2 (Wald identity)
# ---------------------------------------------------------------------------


class TestModeratedF:
    def test_f_on_2_group_equals_t_squared(self):
        import alphaphos as ap

        adata = _make_two_group_adata()
        r_t = ap.diff_exp_limma_contrasts(
            adata,
            condition_column="group",
            contrasts={"A_vs_B": ("A", "B")},
            joint=True,
        )["A_vs_B"]
        r_f = ap.diff_exp_anova(adata, condition_column="group")

        # Wald identity: F = t^2 on a 1-df contrast.
        np.testing.assert_allclose(
            r_f["F"].to_numpy(),
            (r_t["t_stat"].to_numpy()) ** 2,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            r_f["p_value"].to_numpy(),
            r_t["p_value"].to_numpy(),
            atol=1e-10,
        )
        assert int(r_f["df_between"].iloc[0]) == 1

    def test_f_rejects_null_on_multi_group_signal(self):
        # 3 groups, real signal on first 20 features -> F picks them up.
        rng = np.random.default_rng(0)
        n_per, p = 5, 100
        X = rng.normal(0, 1, size=(3 * n_per, p))
        X[n_per : 2 * n_per, :20] += 2.0
        X[2 * n_per :, :20] += 4.0
        obs = pd.DataFrame(
            {"group": ["A"] * n_per + ["B"] * n_per + ["C"] * n_per},
            index=[f"s{i}" for i in range(3 * n_per)],
        )
        adata = ad.AnnData(
            X=X.astype(np.float64),
            obs=obs,
            var=pd.DataFrame(index=[f"g{i}" for i in range(p)]),
        )
        adata.layers["intensity_log2"] = adata.X.copy()

        import alphaphos as ap

        r = ap.diff_exp_anova(adata, condition_column="group")
        n_sig = int((r["fdr"] < 0.05).sum())
        assert n_sig >= 15, f"expected >=15 signal features to survive FDR, got {n_sig}"

        # Top-3 should all be in the signal region [0, 20).
        top3 = r.sort_values("F", ascending=False).head(3).index
        top3_ints = [int(name.lstrip("g")) for name in top3]
        assert all(i < 20 for i in top3_ints), f"top-3 F outside signal region: {top3_ints}"


# ---------------------------------------------------------------------------
# diff_exp_limma_contrasts: joint=False path (loop over 2-group)
# ---------------------------------------------------------------------------


@_needs_inmoose
class TestLoopContrasts:
    def test_joint_false_equals_diff_exp_limma_per_pair(self):
        # joint=False should return identical columns to a manual loop of
        # diff_exp_limma calls (since it *is* that loop internally).
        import alphaphos as ap

        adata = _make_two_group_adata()
        r_loop = ap.diff_exp_limma_contrasts(
            adata,
            condition_column="group",
            contrasts={"A_vs_B": ("A", "B")},
            joint=False,
        )["A_vs_B"]
        r_direct = ap.diff_exp_limma(adata, condition_column="group", comparison=("A", "B"))
        pd.testing.assert_frame_equal(r_loop, r_direct)


# ---------------------------------------------------------------------------
# anova_hits: ANOVA output -> (hits, background) for ORA
# ---------------------------------------------------------------------------


class TestAnovaHits:
    def test_splits_by_fdr(self):
        import alphaphos as ap

        df = pd.DataFrame(
            {"fdr": [0.001, 0.02, 0.05, 0.5]},
            index=["a", "b", "c", "d"],
        )
        hits, bg = ap.anova_hits(df, fdr_threshold=0.05)
        assert hits == ["a", "b"]
        assert bg == ["a", "b", "c", "d"]

    def test_drops_nan_from_both_arms(self):
        # Untested features (NaN fdr) must not appear in either arm --
        # they weren't in the Fisher table.
        import alphaphos as ap

        df = pd.DataFrame(
            {"fdr": [0.001, np.nan, 0.5, np.nan]},
            index=["a", "b", "c", "d"],
        )
        hits, bg = ap.anova_hits(df)
        assert hits == ["a"]
        assert bg == ["a", "c"]

    def test_end_to_end_from_diff_exp_anova(self):
        # Real handoff: diff_exp_anova -> anova_hits -> ora-ready lists.
        import alphaphos as ap

        rng = np.random.default_rng(0)
        n_per, p = 5, 60
        X = rng.normal(0, 1, size=(3 * n_per, p))
        X[n_per : 2 * n_per, :10] += 3.0
        X[2 * n_per :, :10] += 6.0
        obs = pd.DataFrame(
            {"group": ["A"] * n_per + ["B"] * n_per + ["C"] * n_per},
            index=[f"s{i}" for i in range(3 * n_per)],
        )
        adata = ad.AnnData(
            X=X.astype(np.float64),
            obs=obs,
            var=pd.DataFrame(index=[f"g{i}" for i in range(p)]),
        )
        adata.layers["intensity_log2"] = adata.X.copy()

        anova = ap.diff_exp_anova(adata, condition_column="group")
        hits, bg = ap.anova_hits(anova, fdr_threshold=0.05)

        # Background is every tested feature (no NaNs here).
        assert len(bg) == p
        # Hits are a strict subset of background.
        assert set(hits).issubset(set(bg))
        # Signal features g0..g9 should dominate the hit list.
        signal_hits = [h for h in hits if int(h.lstrip("g")) < 10]
        assert len(signal_hits) >= 8, hits

    def test_raises_on_missing_fdr_column(self):
        import alphaphos as ap

        df = pd.DataFrame({"F": [1.0]}, index=["a"])
        with pytest.raises(KeyError, match="fdr_col"):
            ap.anova_hits(df)
