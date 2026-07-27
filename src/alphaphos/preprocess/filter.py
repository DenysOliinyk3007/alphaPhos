"""Site-completeness filtering for phosphosite AnnData.

Drops sites (``.var`` rows) whose valid-value fraction is below a
threshold. Three group-handling strategies:

- ``"all"`` (default when no group column): compute the valid fraction
  across all samples together and keep sites that meet the threshold
  globally.
- ``"any"`` (needs ``group_column``): compute the valid fraction WITHIN
  each group and keep sites that meet the threshold in AT LEAST ONE
  group. Preserves condition-specific sites (e.g. induced upon
  stimulation, absent in control).
- ``"each"`` (needs ``group_column``): keep sites that meet the
  threshold in EVERY group. Ensures every condition contributes
  reliable data — the safest choice before differential testing.

Dropping var rows dispatches automatically to ``.X``, every layer, and
every ``varm`` matrix (AnnData contract), so this module doesn't touch
any of them explicitly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from alphaphos.constants import VAR_CLASSI_WILSON_LB

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


VALID_STRATEGIES = ("all", "any", "each")

DEFAULT_SENSITIVITY_THRESHOLDS: tuple[float, ...] = (0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75)


def filter_by_completeness(
    adata: ad.AnnData,
    *,
    min_valid_frac: float | None = None,
    min_valid_n: int | None = None,
    group_column: str | None = None,
    keep_strategy: Literal["all", "any", "each"] = "all",
    layer: str | None = None,
) -> ad.AnnData:
    """Return an ``AnnData`` with sites failing the completeness threshold dropped.

    Provide either ``min_valid_frac`` (fractional, 0-1) OR ``min_valid_n``
    (absolute count).  ``min_valid_n`` is often clearer for small-n
    designs (e.g. "keep a site only if observed in at least 3 samples
    of each group"), and matches the completeness-first workflow used
    by :func:`alphaphos.stats.diff_exp_limma_observed_only`.

    Parameters
    ----------
    adata : AnnData
        Shape ``(n_samples, n_sites)``.
    min_valid_frac : float in [0, 1], optional
        Minimum fraction of non-NaN values REQUIRED to keep a site.
        Applied per group (when ``group_column`` is set) or globally
        (``keep_strategy="all"``).  E.g. ``0.7`` = require observations
        in >=70% of samples.  Mutually exclusive with ``min_valid_n``.
    min_valid_n : int, optional
        Minimum absolute count of non-NaN values REQUIRED to keep a site,
        applied per group or globally as above.  Mutually exclusive with
        ``min_valid_frac``.
    group_column : str, optional
        Column in ``adata.obs`` that partitions samples into groups
        (e.g. ``"condition"``). Required for ``keep_strategy in ("any", "each")``;
        MUST be ``None`` for ``keep_strategy="all"``.
    keep_strategy : {"all", "any", "each"}
        - ``"all"``: no groups. A site is kept if its overall valid
          fraction / count meets the threshold. Default.
        - ``"any"``: needs ``group_column``. Keeps a site if it passes the
          threshold in at least one group.
        - ``"each"``: needs ``group_column``. Keeps a site only if it passes
          the threshold in EVERY group -- the strict per-group completeness
          recommended before differential testing.
    layer : str, optional
        Which layer to compute missingness on. ``None`` (default) = ``adata.X``.
        Only affects the completeness computation; the filter DROPS var rows
        from every layer / obsm / varm automatically via AnnData subsetting.

    Returns
    -------
    AnnData
        A new AnnData with the failing sites removed. Always returns; does
        not mutate the input.

    Raises
    ------
    ValueError
        If neither / both of ``min_valid_frac``, ``min_valid_n`` are
        given; if ``min_valid_frac`` is out of range or ``min_valid_n``
        is negative; if ``keep_strategy`` is unknown; or if the strategy
        vs ``group_column`` combination is invalid.
    KeyError
        If ``group_column`` is set but not in ``adata.obs``, or ``layer`` is
        set but not in ``adata.layers``.
    """
    if (min_valid_frac is None) == (min_valid_n is None):
        raise ValueError(
            "provide exactly one of min_valid_frac or min_valid_n (not both, not neither)."
        )
    if min_valid_frac is not None and not 0.0 <= float(min_valid_frac) <= 1.0:
        raise ValueError(f"min_valid_frac must be in [0, 1], got {min_valid_frac!r}")
    if min_valid_n is not None and int(min_valid_n) < 0:
        raise ValueError(f"min_valid_n must be >= 0, got {min_valid_n!r}")
    if keep_strategy not in VALID_STRATEGIES:
        raise ValueError(f"keep_strategy must be one of {VALID_STRATEGIES}, got {keep_strategy!r}")
    if keep_strategy == "all" and group_column is not None:
        raise ValueError(
            "keep_strategy='all' computes a global (ungrouped) filter; "
            "drop group_column, or set keep_strategy to 'any' or 'each'."
        )
    if keep_strategy in ("any", "each") and group_column is None:
        raise ValueError(f"keep_strategy={keep_strategy!r} requires group_column to be set.")
    if group_column is not None and group_column not in adata.obs.columns:
        raise KeyError(
            f"group_column={group_column!r} not in adata.obs. Available: {list(adata.obs.columns)}"
        )
    if layer is not None and layer not in adata.layers:
        raise KeyError(
            f"layer={layer!r} not in adata.layers. Available: {list(adata.layers.keys())}"
        )

    X = np.asarray(adata.X if layer is None else adata.layers[layer], dtype=float)
    not_nan = ~np.isnan(X)  # (n_samples, n_sites) True where observed

    def _passes(n_valid_per_site: np.ndarray, n_samples: int) -> np.ndarray:
        # Use whichever threshold was supplied.
        if min_valid_n is not None:
            return n_valid_per_site >= int(min_valid_n)
        return (n_valid_per_site / max(n_samples, 1)) >= float(min_valid_frac)

    if keep_strategy == "all":
        n_samples = X.shape[0]
        n_valid = not_nan.sum(axis=0)
        keep = _passes(n_valid, n_samples)
    else:
        groups = adata.obs[group_column].astype(str).values
        unique_groups = np.unique(groups)
        # per_group_pass[i, j] = does site j pass in group i?
        per_group_pass = np.zeros((len(unique_groups), X.shape[1]), dtype=bool)
        for gi, g in enumerate(unique_groups):
            mask = groups == g
            n_g = int(mask.sum())
            if n_g == 0:
                continue
            per_group_pass[gi] = _passes(not_nan[mask].sum(axis=0), n_g)
        if keep_strategy == "any":
            keep = per_group_pass.any(axis=0)
        else:  # "each"
            keep = per_group_pass.all(axis=0)

    n_before = adata.n_vars
    result = adata[:, keep].copy()

    threshold_repr = (
        f"min_valid_n={min_valid_n}"
        if min_valid_n is not None
        else f"min_valid_frac={min_valid_frac:.2f}"
    )
    logger.info(
        "filter_by_completeness: %d -> %d sites (strategy=%s, %s%s)",
        n_before,
        result.n_vars,
        keep_strategy,
        threshold_repr,
        f", group={group_column}" if group_column else "",
    )
    return result


def wilson_threshold_sensitivity(
    adata: ad.AnnData,
    *,
    thresholds: tuple[float, ...] | list[float] | None = None,
    reference_threshold: float = 0.50,
) -> pd.DataFrame:
    """Return retention & quality per Wilson threshold — for supplement tables.

    For each threshold, computes what the site count, %NaN, median per-site SD
    (across all samples), and median detection breadth would be after applying
    ``classI_wilson_lb >= threshold``.  Adds a ``vs_ref_delta_pct`` column
    showing the site-count deviation from the ``reference_threshold`` row,
    which converts the table directly into a robustness assessment for a paper.

    Requires ``adata.var[VAR_CLASSI_WILSON_LB]`` to be populated (done
    automatically by :func:`alphaphos.collapse_sites` since 0.22.0).

    Parameters
    ----------
    adata
        Site-level AnnData (post-collapse).
    thresholds
        Iterable of thresholds to sweep.  Defaults to
        :data:`DEFAULT_SENSITIVITY_THRESHOLDS`
        (``0.00, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75``).
    reference_threshold
        The threshold whose site count anchors the ``vs_ref_delta_pct``
        column.  Must be present in ``thresholds`` (or the closest value is
        chosen).

    Returns
    -------
    pd.DataFrame
        One row per threshold, columns: ``threshold``, ``n_sites``,
        ``pct_kept``, ``pct_nan``, ``median_sd``, ``median_n_det``,
        ``vs_ref_delta_pct``.
    """
    if VAR_CLASSI_WILSON_LB not in adata.var.columns:
        raise KeyError(
            f"adata.var lacks '{VAR_CLASSI_WILSON_LB}'. "
            "Re-collapse with alphaphos >= 0.22 to populate it automatically."
        )

    ts = tuple(thresholds) if thresholds is not None else DEFAULT_SENSITIVITY_THRESHOLDS
    if len(ts) == 0:
        raise ValueError("thresholds must be non-empty")
    for t in ts:
        if not 0.0 <= float(t) <= 1.0:
            raise ValueError(f"threshold {t} outside [0, 1]")

    lb = adata.var[VAR_CLASSI_WILSON_LB].to_numpy()
    n_det = (
        adata.var["n_samples_detected"].to_numpy() if "n_samples_detected" in adata.var else None
    )
    X = adata.X
    total_sites = adata.n_vars

    with np.errstate(all="ignore"):
        per_site_sd = np.nanstd(X, axis=0)
        per_site_obs = np.sum(~np.isnan(X), axis=0)
    per_site_sd_valid = np.where(per_site_obs >= 2, per_site_sd, np.nan)

    rows = []
    for t in ts:
        mask = lb >= float(t)
        if mask.sum() == 0:
            rows.append(
                {
                    "threshold": float(t),
                    "n_sites": 0,
                    "pct_kept": 0.0,
                    "pct_nan": float("nan"),
                    "median_sd": float("nan"),
                    "median_n_det": float("nan"),
                }
            )
            continue
        Xk = X[:, mask]
        rows.append(
            {
                "threshold": float(t),
                "n_sites": int(mask.sum()),
                "pct_kept": float(mask.mean()),
                "pct_nan": float(np.isnan(Xk).mean()) if Xk.size else float("nan"),
                "median_sd": float(np.nanmedian(per_site_sd_valid[mask])),
                "median_n_det": float(np.median(n_det[mask]))
                if n_det is not None
                else float("nan"),
            }
        )
    df = pd.DataFrame(rows)

    # Anchor row for delta column
    ref_idx = int((df["threshold"] - reference_threshold).abs().idxmin())
    ref_n = df.loc[ref_idx, "n_sites"]
    df["vs_ref_delta_pct"] = (df["n_sites"] - ref_n) / ref_n * 100.0 if ref_n else float("nan")
    df.attrs["total_sites"] = total_sites
    df.attrs["reference_threshold"] = float(df.loc[ref_idx, "threshold"])
    return df
