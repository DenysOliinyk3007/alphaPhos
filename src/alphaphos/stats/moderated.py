"""Clean-room Smyth 2004 empirical-Bayes variance moderation.

Implements limma's ``fitFDist`` (method-of-moments prior estimation on
log-scaled residual variances) and the moderated variance / degrees-of-
freedom update.  Independent of R limma and of PhosPy -- written from
the published algorithm (Smyth GK, *Linear models and empirical Bayes
methods for assessing differential expression in microarray experiments*,
SAGMB 3:3, 2004) so the file is MIT-compatible.

The moderation shrinks each per-feature residual variance toward a
pooled prior estimated from all features, giving more powerful t / F
tests on small-sample designs where individual per-feature variance
estimates are noisy.

Not depended on by any external code -- consumed by
:mod:`alphaphos.stats.linear_model` (which layers moderation on top of
a QR-based multi-contrast fit) and re-exported from
:func:`alphaphos.diff_exp_anova` / :func:`alphaphos.diff_exp_limma_contrasts`.
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
    prior_variance : float
        ``s0^2`` -- the location parameter of the scaled-F prior.
        Interpretation: the "consensus" residual variance across features.
    prior_df : float
        ``df0`` -- the shape parameter (prior degrees of freedom).  Larger
        values pull individual features harder toward ``prior_variance``.
        ``inf`` means the prior is degenerate (all variances effectively
        equal); in that case moderation reduces to using the pooled prior
        variance directly.
    n_features_used : int
        How many finite variances entered the prior fit (after edge-case
        filtering).
    """

    prior_variance: float
    prior_df: float
    n_features_used: int


def fit_f_dist(variances: np.ndarray, *, residual_df: float) -> EmpiricalBayesPrior:
    """Estimate the scaled-F prior via method-of-moments on log-variances.

    Smyth 2004 models the residual sample variances as scaled-F variates::

        s_g^2 ~ s0^2 * F(df_residual, df0)

    Working in log space (variance-stabilised for the F distribution),
    with ``z_g = log(s_g^2)`` and ``e_g = z_g - digamma(df_residual/2) +
    log(df_residual/2)``, the mean and variance of ``e_g`` yield the two
    moment equations that identify ``s0^2`` and ``df0``.

    Parameters
    ----------
    variances : ndarray
        Per-feature residual variances (``sigma^2_g``); shape ``(n_features,)``.
    residual_df : float
        Residual degrees of freedom shared across features (``n_samples -
        n_coefficients`` for a standard linear model).

    Returns
    -------
    EmpiricalBayesPrior
        With ``prior_variance``, ``prior_df``, and ``n_features_used``.

    Notes
    -----
    * Non-finite / non-positive variances are dropped from the fit.
    * If the moment estimator returns ``evar <= 0`` (i.e. the observed
      variance-of-log-variances is smaller than the intrinsic sampling
      variance implied by ``residual_df`` alone), we return ``df0 =
      inf`` and ``s0^2 = mean(variances)`` -- the "no room for
      moderation" degenerate case.
    * At most 1 variance: ``prior_variance = that value``, ``prior_df = 0``
      (no shrinkage).
    """
    variances = np.asarray(variances, dtype=np.float64)
    if variances.ndim != 1:
        raise ValueError(f"variances must be 1-D; got shape {variances.shape}")
    if not np.isfinite(residual_df) or residual_df <= 0:
        raise ValueError(f"residual_df must be finite and > 0 for fit_f_dist; got {residual_df!r}")
    df1 = float(residual_df)

    # Drop non-finite / non-positive entries.  Also stabilise ties at zero.
    ok = np.isfinite(variances) & (variances > 0)
    v = variances[ok]
    n_used = int(v.size)
    if n_used == 0:
        return EmpiricalBayesPrior(
            prior_variance=float("nan"), prior_df=float("nan"), n_features_used=0
        )
    if n_used == 1:
        return EmpiricalBayesPrior(prior_variance=float(v[0]), prior_df=0.0, n_features_used=1)

    # Log-transform + centre by the F-distribution bias correction.
    # e_g = log(s_g^2) - digamma(df1/2) + log(df1/2)
    z = np.log(v)
    e = z - digamma(df1 / 2.0) + np.log(df1 / 2.0)
    e_mean = float(np.mean(e))
    # Sample variance of e, minus the intrinsic sampling variance of the
    # bias correction (trigamma(df1/2) via polygamma(1, df1/2)).
    e_var = float(np.sum((e - e_mean) ** 2) / (n_used - 1)) - float(polygamma(1, df1 / 2.0))

    if e_var > 0.0:
        df0 = 2.0 * _trigamma_inverse(e_var)
        s0_sq = float(np.exp(e_mean + digamma(df0 / 2.0) - np.log(df0 / 2.0)))
    else:
        # Degenerate: no room for shrinkage.
        df0 = float("inf")
        s0_sq = float(np.mean(v))

    return EmpiricalBayesPrior(
        prior_variance=float(s0_sq), prior_df=float(df0), n_features_used=n_used
    )


def moderate_variance(
    variances: np.ndarray,
    *,
    residual_df: float,
    prior: EmpiricalBayesPrior,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Smyth 2004 shrinkage to per-feature variances.

    ``sigma^2_moderated_g = (df0 * s0^2 + df_residual * sigma^2_g) /
    (df0 + df_residual)``, with degrees of freedom
    ``df_moderated_g = df0 + df_residual``.

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
    s0_sq = float(prior.prior_variance)

    if not np.isfinite(df0):
        # Degenerate prior -> all features shrink to s0^2, df is Gaussian.
        moderated = np.full_like(variances, s0_sq, dtype=np.float64)
        df_mod = np.full_like(variances, float("inf"), dtype=np.float64)
        return moderated, df_mod

    # Standard limma moderation.
    moderated = (df0 * s0_sq + df_res * variances) / (df0 + df_res)
    df_mod = np.full_like(variances, df0 + df_res, dtype=np.float64)
    return moderated, df_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
