"""Per-feature linear model fit with multi-contrast + moderated t / F tests.

Clean-room implementation of the limma pipeline (Smyth 2004) that
alphaPhos needs for multi-contrast + ANOVA-style testing without the
inmoose 3+-column-design bug.  Not a full limma port -- covers the
subset used by :func:`alphaphos.diff_exp_anova` and
:func:`alphaphos.diff_exp_limma_contrasts`:

1. :func:`lm_fit` -- QR-based ordinary least squares per feature.
2. :func:`contrasts_fit` -- apply a linear contrast matrix to the fit
   (converts per-coefficient estimates into per-contrast estimates).
3. :func:`moderated_t_test` -- Smyth 2004 empirical-Bayes moderated t
   statistic per contrast per feature.
4. :func:`moderated_f_test` -- joint moderated F across multiple
   contrasts per feature.

The empirical-Bayes prior is fit once (across all features) and applied
to all downstream contrasts, giving proper joint moderation that
inmoose can't provide today.

Validated against inmoose's limma port (``tests/unit/test_stats_moderated.py``):
moderated t, prior df / s0^2 and F agree to ~1e-13 on 2- and 3-group designs.
The F **p-value** deliberately follows limma (``pf(F, rank(C), df_prior +
df_residual)``); inmoose 0.9.1's ``classifyTestsF`` returns ``df2 = inf`` and
so reports the chi-square limit, which is anti-conservative.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from alphaphos.stats.moderated import (
    EmpiricalBayesPrior,
    fit_f_dist,
    moderate_variance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fit containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LinearModelFit:
    """Per-feature OLS fit results.

    Attributes
    ----------
    coefficients : ndarray, shape (n_features, n_coefficients)
        Least-squares coefficient estimates, one row per feature.
    sigma_sq : ndarray, shape (n_features,)
        Per-feature residual variance (RSS / df_residual).
    df_residual : int
        Residual degrees of freedom (n_samples - n_coefficients).
        Shared across features because the design is shared.
    cov_unscaled : ndarray, shape (n_coefficients, n_coefficients)
        The unscaled variance-covariance matrix ``(X^T X)^-1``.  The
        actual variance of a linear combination ``c^T beta`` is
        ``sigma_sq * (c^T @ cov_unscaled @ c)``.
    coefficient_labels : tuple[str, ...]
        Names of the coefficients (columns of the design matrix), in
        order.  Used to build human-readable contrast tables.
    amean : ndarray, shape (n_features,)
        Average value of each feature across all samples (limma's
        ``Amean``); the covariate for the limma-trend prior.
    """

    coefficients: np.ndarray
    sigma_sq: np.ndarray
    df_residual: int
    cov_unscaled: np.ndarray
    coefficient_labels: tuple[str, ...] = ()
    amean: np.ndarray | None = None

    @property
    def n_features(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def n_coefficients(self) -> int:
        return int(self.coefficients.shape[1])


@dataclass(frozen=True)
class ContrastFit:
    """Per-contrast estimates derived from a :class:`LinearModelFit`.

    Attributes
    ----------
    fit : LinearModelFit
        The underlying joint fit.
    contrasts : ndarray, shape (n_coefficients, n_contrasts)
        The contrast matrix ``C`` such that
        ``estimates = coefficients @ C``.
    estimates : ndarray, shape (n_features, n_contrasts)
        ``coefficients @ C`` -- one linear-combination estimate per
        (feature, contrast).
    unscaled_variance : ndarray, shape (n_contrasts, n_contrasts)
        ``C^T @ cov_unscaled @ C`` -- diagonal gives the unscaled
        per-contrast variance.
    contrast_labels : tuple[str, ...]
        Names of the contrasts (columns of ``C``); optional.
    """

    fit: LinearModelFit
    contrasts: np.ndarray
    estimates: np.ndarray
    unscaled_variance: np.ndarray
    contrast_labels: tuple[str, ...] = ()

    @property
    def n_contrasts(self) -> int:
        return int(self.contrasts.shape[1])


# ---------------------------------------------------------------------------
# Fit + contrasts
# ---------------------------------------------------------------------------


def lm_fit(
    Y: np.ndarray,
    design: np.ndarray,
    *,
    coefficient_labels: Sequence[str] | None = None,
) -> LinearModelFit:
    """Fit ``Y = design @ B + noise`` per feature via QR.

    Parameters
    ----------
    Y : ndarray, shape (n_samples, n_features)
        Observation matrix -- one column per feature.
    design : ndarray, shape (n_samples, n_coefficients)
        Design matrix; must be full column-rank.
    coefficient_labels : sequence of str, optional
        Names for the ``n_coefficients`` columns.  Stored on the fit.

    Returns
    -------
    LinearModelFit
    """
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(design, dtype=np.float64)
    if Y.ndim != 2 or X.ndim != 2:
        raise ValueError(f"Y ({Y.shape}) and design ({X.shape}) must be 2-D")
    if Y.shape[0] != X.shape[0]:
        raise ValueError(f"Y has {Y.shape[0]} rows (samples) but design has {X.shape[0]}")
    n_samples, n_coef = X.shape
    if n_samples <= n_coef:
        raise ValueError(
            f"design has no residual degrees of freedom: {n_samples} samples <= "
            f"{n_coef} coefficients"
        )
    rank = int(np.linalg.matrix_rank(X))
    if rank < n_coef:
        raise ValueError(f"design is rank-deficient: rank {rank} < {n_coef} coefficients")

    # QR factorisation: X = Q R, where Q is (n_samples, n_coef) with
    # orthonormal columns and R is (n_coef, n_coef) upper triangular.
    q, r = np.linalg.qr(X)
    # coefficients = R^-1 Q^T Y
    qty = q.T @ Y  # (n_coef, n_features)
    beta = np.linalg.solve(r, qty)  # (n_coef, n_features)

    # Residuals + per-feature sigma^2.
    fitted = q @ qty  # (n_samples, n_features)
    residuals = Y - fitted
    df_residual = n_samples - n_coef
    sigma_sq = np.sum(residuals**2, axis=0) / df_residual

    # Unscaled covariance: (X^T X)^-1 = R^-1 R^-T
    r_inv = np.linalg.solve(r, np.eye(n_coef))
    cov_unscaled = r_inv @ r_inv.T

    labels = (
        tuple(coefficient_labels)
        if coefficient_labels
        else tuple(f"coef_{i}" for i in range(n_coef))
    )
    if len(labels) != n_coef:
        raise ValueError(f"coefficient_labels has {len(labels)} entries; expected {n_coef}")

    return LinearModelFit(
        coefficients=beta.T,  # transpose so shape is (n_features, n_coef)
        sigma_sq=sigma_sq,
        df_residual=int(df_residual),
        cov_unscaled=cov_unscaled,
        coefficient_labels=labels,
        amean=Y.mean(axis=0),
    )


def fit_prior(fit: LinearModelFit, *, trend: bool) -> EmpiricalBayesPrior:
    """Fit the empirical-Bayes variance prior for a :class:`LinearModelFit`.

    ``trend=True`` is *limma-trend*: the prior scale follows a natural cubic
    spline in ``fit.amean`` (average log-intensity), which is the right model
    when residual variance depends on intensity -- as it does for MS
    intensities.  ``trend=False`` fits the constant prior of Smyth 2004.
    """
    return fit_f_dist(
        fit.sigma_sq,
        residual_df=fit.df_residual,
        covariate=fit.amean if trend else None,
    )


def contrasts_fit(
    fit: LinearModelFit,
    contrasts: np.ndarray,
    *,
    contrast_labels: Sequence[str] | None = None,
) -> ContrastFit:
    """Apply a linear contrast matrix to a :class:`LinearModelFit`.

    Parameters
    ----------
    fit : LinearModelFit
    contrasts : ndarray, shape (n_coefficients, n_contrasts)
        The contrast matrix.  Each column encodes one linear combination
        of coefficients, e.g. ``[1, -1, 0, ...]`` for
        ``coef_0 - coef_1``.
    contrast_labels : sequence of str, optional

    Returns
    -------
    ContrastFit
    """
    C = np.asarray(contrasts, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError(f"contrasts must be 2-D; got shape {C.shape}")
    if C.shape[0] != fit.n_coefficients:
        raise ValueError(
            f"contrast rows ({C.shape[0]}) must match n_coefficients ({fit.n_coefficients})"
        )
    estimates = fit.coefficients @ C  # (n_features, n_contrasts)
    unscaled_var = C.T @ fit.cov_unscaled @ C  # (n_contrasts, n_contrasts)

    labels = (
        tuple(contrast_labels)
        if contrast_labels
        else tuple(f"contrast_{i}" for i in range(C.shape[1]))
    )
    if len(labels) != C.shape[1]:
        raise ValueError(f"contrast_labels has {len(labels)} entries; expected {C.shape[1]}")
    return ContrastFit(
        fit=fit,
        contrasts=C,
        estimates=estimates,
        unscaled_variance=unscaled_var,
        contrast_labels=labels,
    )


# ---------------------------------------------------------------------------
# Moderated t + F tests
# ---------------------------------------------------------------------------


def moderated_t_test(
    contrast_fit: ContrastFit,
    *,
    var_names: Sequence[str] | None = None,
    prior: EmpiricalBayesPrior | None = None,
    trend: bool = False,
) -> dict[str, pd.DataFrame]:
    """Per-contrast moderated t-test using empirical-Bayes shrinkage.

    Parameters
    ----------
    contrast_fit : ContrastFit
    var_names : sequence of str, optional
        Feature names; used as the index of the returned DataFrames.
    prior : EmpiricalBayesPrior, optional
        Pre-fit prior.  ``None`` (default) fits it here via
        :func:`fit_prior` (constant, or trended when ``trend=True``).
    trend : bool
        Only used when ``prior`` is None; see :func:`fit_prior`.

    Returns
    -------
    dict[str, pandas.DataFrame]
        One entry per contrast (keyed by ``contrast_labels``).  Each
        DataFrame has columns::

            estimate     -- contrast value (e.g. log2FC)
            se           -- moderated standard error
            t_stat       -- moderated t
            p_value      -- two-sided p from the moderated t
            fdr          -- BH-adjusted p across features
            ave_expr     -- (attached as .attrs; caller can join if wanted)
            df_moderated -- moderated degrees of freedom
    """
    from scipy.stats import false_discovery_control

    fit = contrast_fit.fit
    n_features = fit.n_features

    if prior is None:
        prior = fit_prior(fit, trend=trend)
    sigma_sq_mod, df_mod = moderate_variance(fit.sigma_sq, residual_df=fit.df_residual, prior=prior)

    idx = pd.Index(var_names) if var_names is not None else None
    results: dict[str, pd.DataFrame] = {}
    for j, name in enumerate(contrast_fit.contrast_labels):
        unscaled = float(contrast_fit.unscaled_variance[j, j])
        if unscaled <= 0:
            raise ValueError(
                f"contrast {name!r} has non-positive unscaled variance "
                f"({unscaled}); check the contrast matrix"
            )
        estimate = contrast_fit.estimates[:, j]
        se = np.sqrt(sigma_sq_mod * unscaled)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(se > 0, estimate / se, np.nan)
        # p_value: two-sided from moderated t.  df_mod may be inf (Gaussian).
        p = np.empty(n_features, dtype=np.float64)
        finite = np.isfinite(t)
        p[:] = np.nan
        if np.isfinite(df_mod[0]):
            p[finite] = 2.0 * stats.t.sf(np.abs(t[finite]), df_mod[finite])
        else:
            p[finite] = 2.0 * stats.norm.sf(np.abs(t[finite]))

        fdr = np.full(n_features, np.nan, dtype=np.float64)
        f_ok = np.isfinite(p)
        if f_ok.any():
            fdr[f_ok] = false_discovery_control(p[f_ok], method="bh")

        df = pd.DataFrame(
            {
                "estimate": estimate,
                "se": se,
                "t_stat": t,
                "p_value": p,
                "fdr": fdr,
                "df_moderated": df_mod,
            },
            index=idx,
        )
        df.attrs["prior_variance"] = prior.prior_variance
        df.attrs["prior_df"] = prior.prior_df
        df.attrs["trend"] = prior.trend
        df.attrs["contrast_name"] = name
        results[name] = df
    return results


def moderated_f_test(
    fit: LinearModelFit,
    contrasts: np.ndarray,
    *,
    var_names: Sequence[str] | None = None,
    prior: EmpiricalBayesPrior | None = None,
    trend: bool = False,
) -> pd.DataFrame:
    """Joint moderated F-test across ``n_contrasts`` linear combinations.

    Tests the null "all contrasts are zero" per feature -- the ANOVA-
    style question.

    Parameters
    ----------
    fit : LinearModelFit
    contrasts : ndarray, shape (n_coefficients, n_contrasts)
    var_names : sequence of str, optional
    prior : EmpiricalBayesPrior, optional
    trend : bool
        Only used when ``prior`` is None; see :func:`fit_prior`.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``var_names`` (or a RangeIndex).  Columns::

            F            -- moderated F statistic
            p_value      -- from the F-distribution
            fdr          -- BH-adjusted p across features
            df_between   -- numerator df = rank(C)
            df_moderated -- moderated denominator df (df_prior + df_residual)

    Notes
    -----
    Formula (see Smyth 2004 eq 10)::

        F = beta_contrasts^T (C^T (X^T X)^{-1} C)^{-1} beta_contrasts
            / (rank(C) * sigma_sq_moderated)

    where ``beta_contrasts = C^T beta``.  Under the null,
    ``F ~ F(rank(C), df_moderated)``.  As in limma, a rank-deficient
    contrast matrix (redundant contrasts) is handled via the pseudo-inverse
    and the numerator df is the rank, not the number of columns.
    """
    from scipy.stats import f as f_dist
    from scipy.stats import false_discovery_control

    cf = contrasts_fit(fit, contrasts)
    n_contrasts = cf.n_contrasts
    if n_contrasts < 1:
        raise ValueError("need >=1 contrast for the moderated F-test")
    rank = int(np.linalg.matrix_rank(cf.contrasts))
    if rank < 1:
        raise ValueError("contrast matrix is all zeros")
    if rank < n_contrasts:
        logger.warning(
            "moderated_f_test: contrast matrix has rank %d < %d columns (redundant "
            "contrasts); numerator df uses the rank.",
            rank,
            n_contrasts,
        )

    if prior is None:
        prior = fit_prior(fit, trend=trend)
    sigma_sq_mod, df_mod = moderate_variance(fit.sigma_sq, residual_df=fit.df_residual, prior=prior)

    # Invert the unscaled contrast covariance once.  Use pinv for
    # robustness -- if the contrast matrix is rank-deficient (redundant
    # contrasts), pinv drops the degenerate directions.
    A = cf.unscaled_variance  # (n_contrasts, n_contrasts)
    A_inv = np.linalg.pinv(A)

    # Quadratic form per feature: q_g = estimates_g^T A^-1 estimates_g
    est = cf.estimates  # (n_features, n_contrasts)
    q = np.einsum("gi,ij,gj->g", est, A_inv, est)

    with np.errstate(divide="ignore", invalid="ignore"):
        F = q / (rank * sigma_sq_mod)
    df_between = rank

    n_features = fit.n_features
    p = np.full(n_features, np.nan, dtype=np.float64)
    finite = np.isfinite(F) & (F >= 0)
    if finite.any():
        if np.isfinite(df_mod[0]):
            # Denominator df is per-feature but constant when moderation is applied.
            p[finite] = f_dist.sf(F[finite], df_between, df_mod[finite])
        else:
            # Degenerate prior -> chi-sq / n_contrasts limit.
            from scipy.stats import chi2

            p[finite] = chi2.sf(q[finite] / sigma_sq_mod[finite], df_between)

    fdr = np.full(n_features, np.nan, dtype=np.float64)
    f_ok = np.isfinite(p)
    if f_ok.any():
        fdr[f_ok] = false_discovery_control(p[f_ok], method="bh")

    idx = pd.Index(var_names) if var_names is not None else None
    result = pd.DataFrame(
        {
            "F": F,
            "p_value": p,
            "fdr": fdr,
            "df_between": df_between,
            "df_moderated": df_mod,
        },
        index=idx,
    )
    result.attrs["method"] = "moderated_anova"
    result.attrs["prior_variance"] = prior.prior_variance
    result.attrs["prior_df"] = prior.prior_df
    result.attrs["trend"] = prior.trend
    result.attrs["n_contrasts"] = n_contrasts
    result.attrs["rank"] = rank
    logger.info(
        "moderated_f_test: n_features=%d, n_contrasts=%d, prior_df=%.3f, prior_var=%.4g%s",
        n_features,
        n_contrasts,
        prior.prior_df,
        float(np.median(prior.prior_variance)),
        " (median of trend)" if prior.trend else "",
    )
    return result
