"""Localization-based masking of the site quant matrix.

Three strategies for deciding whether a per-(site, run) cell is "trusted":

* ``per_run`` -- the strictest. A cell is kept only when its per-run
  localization probability is >= ``cutoff``. Reproduces Spectronaut's native
  Class-I convention. This is a per-cell mask.

* ``global_max`` -- a permissive dataset-wide filter. Compute each site's
  MAX localization across runs; drop sites whose max is below ``cutoff``.
  Preserves intensities in low-loc runs when the site is confidently
  localized somewhere in the dataset. Matches the Hogrebe SN plugin
  historically.

* ``condition`` -- the default. Applied on top of ``global_max``, this is a
  per-condition majority rule: for each (site, condition), if the fraction
  of replicates whose per-run loc >= ``classI_cutoff`` reaches at least
  ``condition_threshold``, keep ALL replicates of that condition; else
  keep only the Class-I replicates. Recovers information from replicates
  with borderline loc when the site is reliably localized within the
  same biological condition. Implemented in
  :mod:`alphaphos.preprocess.classify`.

All three strategies operate on the LINEAR quant matrix BEFORE log2 (log2
happens after). NaN in the loc matrix is treated as "below cutoff" (i.e.
masking behaves as if the site wasn't observed in that run).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_NULL_LOGGER = logging.getLogger("alphaphos.preprocess._collapse.masking")


VALID_STRATEGIES = ("per_run", "global_max", "condition")


def mask_per_run(
    site_quant: pd.DataFrame,
    site_loc: pd.DataFrame,
    *,
    cutoff: float,
    logger: logging.Logger = _NULL_LOGGER,
) -> pd.DataFrame:
    """Set quant cells to NaN where the loc prob is below ``cutoff``.

    Parameters
    ----------
    site_quant : DataFrame
        Site-level quant matrix, indexed by ``full_key``.
    site_loc : DataFrame
        Same shape and index as ``site_quant``.
    cutoff : float
        Loc-probability threshold in [0, 1].

    Returns
    -------
    DataFrame
        Same shape as ``site_quant``. Cells with loc < ``cutoff`` (or NaN
        loc) become NaN.
    """
    loc_aligned = site_loc.reindex(index=site_quant.index, columns=site_quant.columns)
    keep = loc_aligned >= cutoff  # NaN loc -> False -> masked
    n_before = int(site_quant.notna().sum().sum())
    masked = site_quant.where(keep, np.nan)
    n_after = int(masked.notna().sum().sum())
    logger.info(
        "per_run mask (cutoff=%.2f): %d/%d cells masked (%.1f%%).",
        cutoff,
        n_before - n_after,
        n_before,
        100 * (n_before - n_after) / n_before if n_before else 0.0,
    )
    return masked


def filter_by_global_max(
    site_quant: pd.DataFrame,
    site_loc: pd.DataFrame,
    *,
    cutoff: float,
    logger: logging.Logger = _NULL_LOGGER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop sites whose per-run MAX loc is below ``cutoff``.

    This is a row-level filter (not a cell mask): a site either survives
    completely or is removed from both the quant and loc matrices.

    Returns
    -------
    site_quant_kept, site_loc_kept
        Both filtered to the surviving sites, aligned.
    """
    global_max = site_loc.max(axis=1, skipna=True)
    keep = global_max >= cutoff
    n_before = len(site_quant)
    quant_kept = site_quant.loc[keep]
    loc_kept = site_loc.loc[keep]
    logger.info(
        "global_max filter (cutoff=%.2f): %d -> %d sites (%d dropped).",
        cutoff,
        n_before,
        len(quant_kept),
        n_before - len(quant_kept),
    )
    return quant_kept, loc_kept


def mask_condition_aware(
    site_quant: pd.DataFrame,
    site_loc: pd.DataFrame,
    *,
    condition_df: pd.DataFrame,
    classI_cutoff: float = 0.75,
    condition_threshold: float = 0.50,
    logger: logging.Logger = _NULL_LOGGER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the per-condition majority-rule Class I mask.

    For each ``(site, condition)`` pair:

    * ``N``   = number of replicates in this condition,
    * ``N_I`` = number of those replicates whose loc >= ``classI_cutoff``.

    Then if ``N_I / N >= condition_threshold`` (default 50%), we KEEP the
    quant values for all replicates of this condition (even those with
    loc < cutoff). This recovers signal in replicates with borderline
    localization when the site is confidently localized in the majority of
    the condition.

    Otherwise, we mask non-Class-I replicates of that condition to NaN --
    only trusting the strongly-localized observations.

    Parameters
    ----------
    site_quant : DataFrame
        ``(n_sites x n_samples)`` LINEAR intensity matrix (log2 hasn't
        happened yet in the pipeline order).
    site_loc : DataFrame
        Same shape / index / columns as ``site_quant``.
    condition_df : DataFrame
        Must contain a ``sample`` column matching ``site_quant.columns`` and
        a ``condition`` column grouping replicates.
    classI_cutoff : float
        Loc-probability threshold for Class I. Default 0.75.
    condition_threshold : float
        Fraction of Class-I replicates required to keep the whole condition.
        Default 0.50 (majority rule).

    Returns
    -------
    masked_quant : DataFrame
        Same shape as ``site_quant``; low-loc cells in low-fraction
        conditions become NaN.
    decision_table : DataFrame
        ``(n_sites x n_conditions)`` fraction of Class-I replicates per
        (site, condition). Attached to ``adata.uns`` for diagnostics.
    """
    if "sample" not in condition_df.columns or "condition" not in condition_df.columns:
        raise ValueError("condition_df must contain both 'sample' and 'condition' columns.")
    if not 0 < condition_threshold <= 1:
        raise ValueError(f"condition_threshold must be in (0, 1], got {condition_threshold}")
    if not 0 <= classI_cutoff <= 1:
        raise ValueError(f"classI_cutoff must be in [0, 1], got {classI_cutoff}")

    s2c = condition_df.set_index("sample")["condition"].astype(str)

    quant = site_quant.astype(float).copy()
    loc = site_loc.reindex(index=quant.index, columns=quant.columns)

    is_classI = loc.ge(classI_cutoff).fillna(False)
    keep_mask = pd.DataFrame(True, index=quant.index, columns=quant.columns)

    decision_rows: dict[str, pd.Series] = {}
    for cond in s2c.unique():
        cond_samples = [s for s in s2c.index[s2c == cond] if s in quant.columns]
        if not cond_samples:
            continue
        N = len(cond_samples)
        N_I = is_classI[cond_samples].sum(axis=1)
        fraction = N_I / N
        decision_rows[cond] = fraction

        low_frac_sites = fraction < condition_threshold
        if low_frac_sites.any():
            block = is_classI.loc[low_frac_sites, cond_samples]
            new_keep = keep_mask.loc[low_frac_sites, cond_samples] & block
            keep_mask.loc[low_frac_sites, cond_samples] = new_keep

    masked = quant.where(keep_mask, np.nan)
    n_before = int(quant.notna().sum().sum())
    n_after = int(masked.notna().sum().sum())
    logger.info(
        "condition mask (classI_cutoff=%.2f, condition_threshold=%.2f): "
        "%d/%d cells masked (%.1f%%).",
        classI_cutoff,
        condition_threshold,
        n_before - n_after,
        n_before,
        100 * (n_before - n_after) / n_before if n_before else 0.0,
    )

    decision_table = pd.DataFrame(decision_rows)
    decision_table.index.name = "full_key"
    return masked, decision_table


def drop_all_nan_sites(
    site_quant: pd.DataFrame,
    *others: pd.DataFrame,
    logger: logging.Logger = _NULL_LOGGER,
) -> tuple[pd.DataFrame, ...]:
    """Drop rows in ``site_quant`` where every column is NaN, aligning others.

    Passed additional DataFrames (loc matrix, metadata, etc.) get the same
    rows filtered out to stay index-aligned.

    Returns
    -------
    tuple
        ``(site_quant_filtered, *others_filtered)``. All share the same
        surviving index.
    """
    all_nan = site_quant.isna().all(axis=1)
    keep_idx = site_quant.index[~all_nan]
    n_before = len(site_quant)
    n_after = len(keep_idx)
    logger.info(
        "Drop all-NaN sites: %d -> %d (%d removed).",
        n_before,
        n_after,
        n_before - n_after,
    )
    out = [site_quant.loc[keep_idx]]
    for df in others:
        out.append(df.loc[keep_idx] if df is not None else None)
    return tuple(out)
