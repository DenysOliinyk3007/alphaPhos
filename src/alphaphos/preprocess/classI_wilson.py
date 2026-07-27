"""Wilson lower-bound Class-I filter for large phospho cohorts.

Corrects the two failure modes of naive per-site Class-I filtering:

- ``max_loc_prob >= 0.75`` (aka ``global_max`` collapse) is too permissive
  at cohort scale — a single 1/200 measurement keeps the site.
- ``min_loc_prob >= 0.75`` (aka ``per_run`` collapse) is too strict —
  a single 0.74 measurement kills a site with 99/100 Class-I detections.

The Wilson lower bound treats each site's ``(n_classI_samples, n_samples_detected)``
as a binomial proportion and reports the lower confidence bound on the
"true" Class-I rate — automatically penalising sites with few detections.

We use the Jeffreys interval formulation (``Beta.ppf(alpha/2, k+½, n-k+½)``),
which is analytically equivalent to Wilson within 0.01 for ``n >= 5`` and is
better-behaved at the edges ``k=0`` and ``k=n``. Both are standard biostat
references (Wilson 1927; Newcombe 1998; Brown, Cai, DasGupta 2001).

For the empirical basis of the recipe (which threshold to use at which cohort
size) see ``scratchpad/wilson_retention_curve.py`` — retention/quality curves
across the 383-sample uPhosHT full-plate dataset.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import beta

from alphaphos.constants import (
    VAR_CLASSI_WILSON_LB,
)

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


# Small-cohort floor for the "auto" threshold path.  Below this, the
# elbow in the retention curve is too noisy to trust and Wilson itself
# starts to overlap with per_run — refuse and point the user at a
# fixed threshold or the classical filters.
AUTO_MIN_N = 30

# Cohort-size-informed candidate ranges for the "auto" threshold picker.
# See project_recommend_pipeline memory + the retention-curve analysis.
AUTO_RANGES: tuple[tuple[int, float, float], ...] = (
    # (n_upper_exclusive, lo, hi)
    (100, 0.20, 0.40),
    (300, 0.30, 0.55),
    (10_000_000, 0.45, 0.65),
)
AUTO_GRID_POINTS = 31

# Elbow-detection strength: below this the curve is judged too flat
# and we fall back to the midpoint of the candidate range.
AUTO_MIN_ELBOW_STRENGTH = 0.02


def wilson_lower_bound(
    k: np.ndarray | int,
    n: np.ndarray | int,
    *,
    alpha: float = 0.05,
) -> np.ndarray:
    """Return the Jeffreys / Wilson-equivalent 100·(1−α)% lower confidence bound
    on the binomial rate ``p`` given ``k`` successes out of ``n`` trials.

    Vectorised over ``k`` and ``n``.  Handles the ``n == 0`` edge case by
    returning ``0.0`` for those entries.  Raises ``ValueError`` if any
    ``k > n`` or if inputs are negative.

    Parameters
    ----------
    k, n
        Non-negative integers or arrays thereof.  Same shape (or broadcastable).
    alpha
        Two-sided significance level.  Default 0.05 → 95% lower bound.

    Returns
    -------
    ndarray
        Lower bounds in ``[0, 1]``.  Scalar in → scalar out (as a 0-d array).
    """
    k_arr = np.asarray(k)
    n_arr = np.asarray(n)
    if k_arr.shape != n_arr.shape:
        try:
            k_arr, n_arr = np.broadcast_arrays(k_arr, n_arr)
        except ValueError as exc:
            raise ValueError(
                f"k and n must be broadcastable; got shapes {k_arr.shape} and {n_arr.shape}"
            ) from exc

    if np.any(k_arr < 0) or np.any(n_arr < 0):
        raise ValueError("k and n must be non-negative")
    if np.any(k_arr > n_arr):
        raise ValueError("k must not exceed n for any site")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    k_f = k_arr.astype(float)
    n_f = n_arr.astype(float)
    # Jeffreys: Beta(k + 1/2, n - k + 1/2), take α/2 quantile
    lb = beta.ppf(alpha / 2.0, k_f + 0.5, n_f - k_f + 0.5)
    # For n == 0 (undetected site), scipy returns nan; replace with 0
    lb = np.where(n_arr == 0, 0.0, lb)
    # Numerical safety at extremes
    return np.clip(lb, 0.0, 1.0)


def _auto_range(n_samples: int) -> tuple[float, float]:
    """Return the candidate wilson_lb search range for the given cohort size."""
    for upper, lo, hi in AUTO_RANGES:
        if n_samples < upper:
            return lo, hi
    return AUTO_RANGES[-1][1], AUTO_RANGES[-1][2]  # pragma: no cover


def auto_wilson_threshold(
    wilson_lb: np.ndarray,
    n_samples: int,
) -> tuple[float, str]:
    """Pick a Wilson threshold via Kneedle elbow detection on the retention curve.

    The candidate range is set by cohort size (see ``AUTO_RANGES``): small
    cohorts get a permissive band, large cohorts get a strict band.  Within
    the range, we sweep 31 thresholds, plot ``(threshold, n_sites_retained)``,
    and pick the point of maximum perpendicular distance below the diagonal
    from (min_threshold, max_retention) → (max_threshold, min_retention)
    (Satopää et al. 2011).

    Parameters
    ----------
    wilson_lb
        Per-site Wilson lower bounds, e.g. ``adata.var["classI_wilson_lb"]``.
    n_samples
        Cohort size (``adata.n_obs``).

    Returns
    -------
    threshold
        Selected wilson_lb threshold in ``[0, 1]``.
    reason
        One-line explanation of the pick (elbow position or fallback).

    Raises
    ------
    ValueError
        If ``n_samples < AUTO_MIN_N`` (30 by default).
    """
    if n_samples < AUTO_MIN_N:
        raise ValueError(
            f"auto Wilson threshold requires n_samples >= {AUTO_MIN_N}; "
            f"got n_samples={n_samples}. "
            "Use a fixed wilson_threshold or a classical strategy (per_run / global_max)."
        )
    lb_arr = np.asarray(wilson_lb, dtype=float)
    if lb_arr.size == 0:
        raise ValueError("wilson_lb array is empty")

    lo, hi = _auto_range(n_samples)
    thresholds = np.linspace(lo, hi, AUTO_GRID_POINTS)
    retention = np.array([int((lb_arr >= t).sum()) for t in thresholds])

    if retention.max() == retention.min():
        picked = float((lo + hi) / 2)
        return (
            picked,
            f"flat retention curve in [{lo:.2f}, {hi:.2f}]; fell back to midpoint {picked:.2f}",
        )

    # Normalize to unit square
    xn = (thresholds - lo) / (hi - lo)
    yn = (retention - retention.min()) / (retention.max() - retention.min())
    # Retention is monotonically non-increasing, so the curve runs (0, 1) → (1, 0).
    # Perpendicular distance BELOW the diagonal from (0,1) to (1,0) is (1 - x_n) - y_n.
    below = (1.0 - xn) - yn
    strength = float(below.max())

    if strength < AUTO_MIN_ELBOW_STRENGTH:
        picked = float((lo + hi) / 2)
        return (
            picked,
            f"no clear elbow (max deviation {strength:.3f} < {AUTO_MIN_ELBOW_STRENGTH:.2f}) "
            f"in [{lo:.2f}, {hi:.2f}]; fell back to midpoint {picked:.2f}",
        )

    idx = int(np.argmax(below))
    picked = float(thresholds[idx])
    return (
        picked,
        f"elbow at wilson_lb={picked:.2f} within [{lo:.2f}, {hi:.2f}] "
        f"for n_samples={n_samples} (elbow strength {strength:.3f})",
    )


def apply_wilson_filter(
    adata: ad.AnnData,
    threshold: float | str = "auto",
) -> ad.AnnData:
    """Return a copy of ``adata`` with sites failing the Wilson filter dropped.

    Requires ``adata.var[VAR_CLASSI_WILSON_LB]`` to be populated (done
    automatically by :func:`alphaphos.collapse_sites` at collapse time).

    Parameters
    ----------
    adata
        Site-level AnnData.
    threshold
        Either a float in ``[0, 1]`` or the string ``"auto"``.  ``"auto"``
        delegates to :func:`auto_wilson_threshold`.

    Returns
    -------
    ad.AnnData
        New AnnData with sites failing the filter removed.  The applied
        threshold and reasoning are stamped into
        ``adata.uns["wilson_filter"]`` for downstream provenance.
    """
    if VAR_CLASSI_WILSON_LB not in adata.var.columns:
        raise KeyError(
            f"adata.var lacks '{VAR_CLASSI_WILSON_LB}'. "
            "Re-collapse with alphaphos >= 0.22, or compute it manually via "
            "wilson_lower_bound(k=n_classI_samples, n=n_samples_detected)."
        )
    wilson_lb = adata.var[VAR_CLASSI_WILSON_LB].to_numpy()

    if isinstance(threshold, str):
        if threshold != "auto":
            raise ValueError(f"threshold must be a float or the string 'auto'; got {threshold!r}")
        picked, reason = auto_wilson_threshold(wilson_lb, adata.n_obs)
    else:
        picked = float(threshold)
        if not 0.0 <= picked <= 1.0:
            raise ValueError(f"threshold must be in [0, 1]; got {picked}")
        reason = f"fixed wilson_lb >= {picked:.2f}"

    if adata.n_obs < 100:
        warnings.warn(
            f"Wilson filter applied at n_samples={adata.n_obs} < 100; "
            "at this cohort scale the Wilson correction is under-supported and may over-filter. "
            "Consider strategy='global_max' + mean_loc_prob filter, or a lower wilson_threshold.",
            UserWarning,
            stacklevel=2,
        )

    mask = wilson_lb >= picked
    out = adata[:, mask].copy()
    out.uns["wilson_filter"] = {
        "threshold": picked,
        "reason": reason,
        "n_sites_pre_filter": int(adata.n_vars),
        "n_sites_kept": int(mask.sum()),
    }
    logger.info(
        "Wilson filter: kept %d / %d sites (threshold=%.3f; %s)",
        int(mask.sum()),
        adata.n_vars,
        picked,
        reason,
    )
    return out


__all__ = [
    "AUTO_MIN_ELBOW_STRENGTH",
    "AUTO_MIN_N",
    "AUTO_RANGES",
    "apply_wilson_filter",
    "auto_wilson_threshold",
    "wilson_lower_bound",
]
