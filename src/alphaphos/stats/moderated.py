"""Clean-room Smyth 2004 empirical-Bayes variance moderation (limma ``fitFDist`` / ``squeezeVar``).

Implements limma's ``fitFDist`` (method-of-moments prior estimation on
log-scaled residual variances), its intensity-dependent variant used by
*limma-trend* (Law 2014; Phipson 2016 -- the prior scale follows a natural
cubic spline in the average log-intensity), and the moderated variance /
degrees-of-freedom update.  Written from the published algorithm and the
current limma behaviour (post-2017 ``fitFDist``) so the file is MIT-compatible;
independent of R limma, inmoose and PhosPy.

Validated against inmoose's limma port: constant and trended priors, moderated
t and F agree to ~1e-12 (``tests/unit/test_stats_moderated.py``,
``tests/unit/test_stats_review.py``).

The moderation shrinks each per-feature residual variance toward a prior
estimated from all features (a constant ``s0^2``, or a per-feature ``s0_g^2``
along the trend), giving more powerful t / F tests on small-sample designs
where individual per-feature variance estimates are noisy.

Consumed by :mod:`alphaphos.stats.linear_model`, which layers moderation on
top of a QR-based multi-contrast fit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import digamma, polygamma

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmpiricalBayesPrior:
    """Estimated prior parameters for the scaled-F variance model.

    Attributes
    ----------
    prior_variance : float or ndarray
        ``s0^2`` -- the scale parameter of the scaled-F prior.  A scalar for
        the constant prior; an array of shape ``(n_features,)`` when fitted
        with a covariate (limma-trend), giving each feature its own prior
        scale read off the trend.
    prior_df : float
        ``df0`` -- the shape parameter (prior degrees of freedom).  Larger
        values pull individual features harder toward ``prior_variance``.
        ``inf`` means the prior is degenerate (all variances effectively
        equal); in that case moderation reduces to using the prior variance
        directly.
    n_features_used : int
        How many finite variances entered the prior fit (after edge-case
        filtering).
    trend : bool
        Whether ``prior_variance`` follows a covariate trend.
    """

    prior_variance: float | np.ndarray
    prior_df: float
    n_features_used: int
    trend: bool = False


def fit_f_dist(
    variances: np.ndarray,
    *,
    residual_df: float,
    covariate: np.ndarray | None = None,
) -> EmpiricalBayesPrior:
    """Estimate the scaled-F prior via method-of-moments on log-variances.

    Smyth 2004 models the residual sample variances as scaled-F variates::

        s_g^2 ~ s0^2 * F(df_residual, df0)

    Working in log space, with ``z_g = log(s_g^2)`` and ``e_g = z_g -
    digamma(df_residual/2) + log(df_residual/2)``, the mean and variance of
    ``e_g`` yield the two moment equations that identify ``s0^2`` and ``df0``.

    Parameters
    ----------
    variances : ndarray
        Per-feature residual variances (``sigma^2_g``); shape ``(n_features,)``.
    residual_df : float
        Residual degrees of freedom shared across features (``n_samples -
        n_coefficients`` for a standard linear model).
    covariate : ndarray, optional
        Per-feature covariate (limma: ``Amean``, the average log-intensity).
        When given, ``mean(e_g)`` is replaced by a natural cubic spline fit of
        ``e`` on the covariate (limma's ``fitFDist(covariate=)``: spline df =
        ``1 + (n>=3) + (n>=6) + (n>=30)``, knots at quantiles) and the prior
        scale becomes per-feature -- this is *limma-trend*.

    Returns
    -------
    EmpiricalBayesPrior

    Notes
    -----
    Follows current limma (post-Jan-2017 ``fitFDist``):

    * Non-finite / negative variances are dropped from the moment fit; when a
      covariate is given their prior scale is still predicted from the trend.
      Exact zeros are offset to ``1e-5 * median`` (with a warning) rather than
      dropped.
    * If the moment estimator returns ``evar <= 0`` (no room for shrinkage)
      the prior is degenerate: ``df0 = inf`` and ``s0^2 = mean(variances)``
      (the pooled MLE; constant prior) or ``exp(fitted trend)`` (covariate).
    * At most one variance: ``prior_variance = that value``, ``prior_df = 0``.
    """
    variances = np.asarray(variances, dtype=np.float64)
    if variances.ndim != 1:
        raise ValueError(f"variances must be 1-D; got shape {variances.shape}")
    if not np.isfinite(residual_df) or residual_df <= 0:
        raise ValueError(f"residual_df must be finite and > 0 for fit_f_dist; got {residual_df!r}")
    df1 = float(residual_df)
    n = variances.size

    cov = None
    if covariate is not None:
        cov = np.asarray(covariate, dtype=np.float64)
        if cov.shape != variances.shape:
            raise ValueError(
                f"covariate shape {cov.shape} must match variances shape {variances.shape}"
            )
        if np.isnan(cov).any():
            raise ValueError("covariate must not contain NaN")
        finite = np.isfinite(cov)
        if not finite.all():
            # limma: push +-inf just outside the finite range.
            cov = cov.copy()
            if finite.any():
                cov[np.isneginf(cov)] = cov[finite].min() - 1
                cov[np.isposinf(cov)] = cov[finite].max() + 1
            else:
                cov = np.sign(cov)

    ok = np.isfinite(variances) & (variances > -1e-15)
    n_used = int(ok.sum())
    if n_used == 0:
        return EmpiricalBayesPrior(
            prior_variance=float("nan"), prior_df=float("nan"), n_features_used=0
        )
    if n_used == 1:
        return EmpiricalBayesPrior(
            prior_variance=float(variances[ok][0]), prior_df=0.0, n_features_used=1
        )
    v = variances[ok]

    # Spline df for the trend; fall back to the constant prior when the
    # covariate cannot support a trend (limma).
    spline_df = 0
    if cov is not None:
        spline_df = 1 + (n_used >= 3) + (n_used >= 6) + (n_used >= 30)
        spline_df = min(spline_df, int(np.unique(cov[ok]).size))
        if spline_df < 2:
            out = fit_f_dist(v, residual_df=df1)
            return EmpiricalBayesPrior(
                prior_variance=np.full(n, float(out.prior_variance)),
                prior_df=out.prior_df,
                n_features_used=out.n_features_used,
                trend=True,
            )

    # Zero variances: offset away from zero instead of dropping (limma).
    v = np.maximum(v, 0.0)
    m = float(np.median(v))
    if m == 0.0:
        logger.warning(
            "More than half of residual variances are exactly zero: moderation unreliable"
        )
        m = 1.0
    elif (v == 0.0).any():
        logger.warning("Zero sample variances detected; offset away from zero")
    v = np.maximum(v, 1e-5 * m)

    # Log-transform + centre by the F-distribution bias correction.
    z = np.log(v)
    e = z - digamma(df1 / 2.0) + np.log(df1 / 2.0)

    e_mean_all: float | np.ndarray
    if cov is None:
        e_mean_used = float(np.mean(e))
        e_var = float(np.sum((e - e_mean_used) ** 2) / (n_used - 1))
        e_mean_all = e_mean_used
    else:
        knots = _spline_knots(cov[ok], spline_df)
        basis = _natural_spline_basis(cov[ok], knots)
        coef, *_ = np.linalg.lstsq(basis, e, rcond=None)
        fitted = basis @ coef
        rank = int(np.linalg.matrix_rank(basis))
        e_var = float(np.sum((e - fitted) ** 2) / (n_used - rank))
        # Predict the trend for every feature (also the ones excluded above).
        e_mean_all = _natural_spline_basis(cov, knots) @ coef

    # Subtract the intrinsic sampling variance of e (trigamma(df1/2)).
    e_var -= float(polygamma(1, df1 / 2.0))

    if e_var > 0.0:
        df0 = 2.0 * _trigamma_inverse(e_var)
        s0_sq = np.exp(e_mean_all + digamma(df0 / 2.0) - np.log(df0 / 2.0))
    else:
        # Degenerate: no room for shrinkage.
        df0 = float("inf")
        s0_sq = float(np.mean(v)) if cov is None else np.exp(e_mean_all)

    if cov is None:
        return EmpiricalBayesPrior(
            prior_variance=float(s0_sq), prior_df=float(df0), n_features_used=n_used
        )
    return EmpiricalBayesPrior(
        prior_variance=np.asarray(s0_sq, dtype=np.float64),
        prior_df=float(df0),
        n_features_used=n_used,
        trend=True,
    )


def moderate_variance(
    variances: np.ndarray,
    *,
    residual_df: float,
    prior: EmpiricalBayesPrior,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Smyth 2004 shrinkage to per-feature variances (limma ``squeezeVar``).

    ``sigma^2_moderated_g = (df0 * s0_g^2 + df_residual * sigma^2_g) /
    (df0 + df_residual)``, with degrees of freedom
    ``df_moderated_g = df0 + df_residual``.  ``s0_g^2`` is the constant prior
    or, for a trended prior, that feature's value on the trend.

    Parameters
    ----------
    variances : ndarray
        Per-feature residual variances; shape ``(n_features,)``.
    residual_df : float
        Residual degrees of freedom for each feature (shared).
    prior : EmpiricalBayesPrior
        The output of :func:`fit_f_dist`.

    Returns
    -------
    (moderated_variances, moderated_df) : tuple of ndarray
        Both shape ``(n_features,)``.  ``moderated_df`` is a constant
        vector equal to ``df0 + residual_df`` unless ``df0`` is infinite,
        in which case it is ``inf`` (Gaussian moderated statistic, no
        Student t correction needed).
    """
    variances = np.asarray(variances, dtype=np.float64)
    df_res = float(residual_df)
    df0 = float(prior.prior_df)
    s0_sq = np.asarray(prior.prior_variance, dtype=np.float64)
    if s0_sq.ndim == 1 and s0_sq.shape != variances.shape:
        raise ValueError(
            f"trended prior has {s0_sq.size} entries but {variances.size} variances were given"
        )

    if not np.isfinite(df0):
        # Degenerate prior -> all features shrink to s0^2, df is Gaussian.
        moderated = np.broadcast_to(s0_sq, variances.shape).astype(np.float64).copy()
        df_mod = np.full_like(variances, float("inf"), dtype=np.float64)
        return moderated, df_mod

    # Standard limma moderation.
    moderated = (df0 * s0_sq + df_res * variances) / (df0 + df_res)
    df_mod = np.full_like(variances, df0 + df_res, dtype=np.float64)
    return moderated, df_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spline_knots(x: np.ndarray, spline_df: int) -> np.ndarray:
    """Knots of limma's ``ns(covariate, df=spline_df, intercept=TRUE)``.

    ``spline_df - 2`` interior knots at equally spaced quantiles of ``x``
    plus the two boundary knots at ``min(x)`` / ``max(x)``.  (R's ``ns`` with
    an intercept has ``df - 2`` interior knots.)
    """
    n_interior = max(int(spline_df) - 2, 0)
    interior = (
        np.quantile(x, np.linspace(0.0, 1.0, n_interior + 2)[1:-1])
        if n_interior > 0
        else np.empty(0)
    )
    return np.concatenate(
        ([float(np.min(x))], np.asarray(interior, dtype=float), [float(np.max(x))])
    )


def _natural_spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis with the given knots (incl. boundary), intercept included.

    Truncated-power representation (Hastie, Tibshirani & Friedman, *ESL* eq.
    5.4-5.5): ``N_1 = 1``, ``N_2 = x``, ``N_{k+2} = d_k(x) - d_{K-1}(x)`` for
    ``k = 1..K-2`` with ``d_k(x) = [(x - xi_k)_+^3 - (x - xi_K)_+^3] / (xi_K -
    xi_k)``.  Spans the same ``K``-dimensional function space as R's
    ``ns(x, knots, intercept=TRUE)``, so least-squares fitted values are
    identical; only the coefficients differ.  With ``K = 2`` (no interior
    knots) this is the linear basis.
    """
    x = np.asarray(x, dtype=np.float64)
    knots = np.asarray(knots, dtype=np.float64)
    K = knots.size
    cols = [np.ones_like(x), x]
    if K > 2:
        xi_K = knots[-1]

        def d(k: int) -> np.ndarray:
            xi_k = knots[k]
            return (np.clip(x - xi_k, 0, None) ** 3 - np.clip(x - xi_K, 0, None) ** 3) / (
                xi_K - xi_k
            )

        d_last = d(K - 2)
        for k in range(K - 2):
            cols.append(d(k) - d_last)
    return np.column_stack(cols)


def _trigamma_inverse(y: float, *, tol: float = 1e-10) -> float:
    """Invert the trigamma function: find x > 0 such that ``polygamma(1, x) == y``.

    Uses bracketed root-finding (scipy.optimize.brentq).  Trigamma is
    monotonically decreasing on ``x > 0``, going from ``+inf`` at ``x -> 0+``
    to ``0`` as ``x -> inf``.  We bracket around the asymptotic estimates:

    * Small ``x``: ``trigamma(x) ~ 1/x^2`` for very small ``x``,
      ``trigamma(x) ~ 1/x + 1/2 + O(x)`` near ``x = 1``.
    * Large ``x``: ``trigamma(x) ~ 1/x + 1/(2 x^2)``.

    So ``x_guess ~ max(1/sqrt(y), 1/y)`` is in the right ballpark; we
    grow / shrink the bracket until it straddles the root and then let
    brentq do the rest.
    """
    from scipy.optimize import brentq

    y = float(y)
    if not np.isfinite(y) or y <= 0.0:
        raise ValueError(f"trigamma_inverse expects a positive finite y; got {y!r}")

    # An initial guess covering both asymptotic regimes.
    x0 = max(1.0 / np.sqrt(y), 1.0 / (y + 1.0))

    def _f(x: float) -> float:
        return float(polygamma(1, x)) - y

    # Grow the bracket outward until _f changes sign.  polygamma(1, .) is
    # monotone decreasing on (0, inf), so `_f(lo) > 0 > _f(hi)` iff
    # trigamma(lo) > y > trigamma(hi).
    lo = x0
    hi = x0
    for _ in range(60):
        if _f(lo) > 0 and _f(hi) < 0:
            break
        if _f(lo) <= 0:
            lo = lo * 0.5
        if _f(hi) >= 0:
            hi = hi * 2.0
    else:
        raise RuntimeError(
            f"trigamma_inverse: could not bracket the root for y={y!r} "
            f"(final lo={lo}, hi={hi}, f(lo)={_f(lo)}, f(hi)={_f(hi)})"
        )
    return float(brentq(_f, lo, hi, xtol=tol, rtol=tol))
