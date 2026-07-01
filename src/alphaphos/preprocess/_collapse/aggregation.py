"""Aggregate multiple precursor rows into a single per-site quant value.

After site-level explosion (in :mod:`._collapse.site_pipeline`) each phospho-
site can be covered by multiple precursor rows -- different charges, different
missed cleavages, or different peptides spanning the same site with a shared
phospho position. Before we take log2 and hand off to downstream analysis,
those rows need to be reduced to a single per-(site, run) intensity.

Four strategies are supported, dispatched by a single string:

* ``"sum"`` (default) -- straight nan-safe sum. Matches Spectronaut's native
  PTM Site Report most closely (see docs/design/spectronaut_collapse_synthesis.md).
  Sensitive to differential ion suppression across peptide contexts, but
  reproduces the field standard.

* ``"median"`` / ``"mean"`` -- simple central-tendency aggregators. NOTE:
  ``"median"`` introduces an intensity-dependent log2 offset relative to
  Spectronaut's own output. Kept available for legacy comparisons but not
  recommended as default.

* ``"consolidate"`` -- Hogrebe's ratio-based iterative imputation, then sum.
  See :func:`consolidate` for the full algorithm. Recovers signal when the
  same site is covered by multiple precursors that each have different
  missing-value patterns across samples.

All strategies operate in LINEAR intensity space. Log2 is applied AFTER
aggregation (see :mod:`._collapse.site_pipeline`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------


VALID_AGGREGATION_METHODS = ("sum", "median", "mean", "consolidate")


def aggregate_by_key(
    df: pd.DataFrame,
    method: str,
    sample_cols: list[str],
) -> pd.DataFrame:
    """Aggregate a DataFrame grouped by its index using ``method``.

    ``df`` must already be indexed by the collapse key (e.g. the full site
    key from :mod:`._collapse.keys`). Each row is a precursor observation;
    each column named in ``sample_cols`` is a run's linear intensity.

    Parameters
    ----------
    df : DataFrame
        Precursor-level linear intensities, indexed by collapse key. May
        contain non-quant columns (they will be ignored -- only
        ``sample_cols`` are aggregated).
    method : str
        One of :data:`VALID_AGGREGATION_METHODS`.
    sample_cols : list[str]
        Names of the run/sample columns to aggregate.

    Returns
    -------
    DataFrame
        Indexed by the collapse key (unique); columns are ``sample_cols``.

    Raises
    ------
    ValueError
        If ``method`` is not recognized.
    """
    if method not in VALID_AGGREGATION_METHODS:
        raise ValueError(
            f"aggregation method must be one of {VALID_AGGREGATION_METHODS}, got {method!r}"
        )

    if method == "consolidate":
        # groupby.apply returning a Series-per-group avoids an intermediate
        # dict; each group's ``consolidate`` output is 1-D (n_samples,).
        groups = df.groupby(df.index)[sample_cols]
        return groups.apply(lambda g: pd.Series(consolidate(g.values), index=sample_cols))
    if method == "median":
        return df[sample_cols].groupby(df.index).median()
    if method == "mean":
        return df[sample_cols].groupby(df.index).mean()
    # Default: sum
    return df[sample_cols].groupby(df.index).sum()


# ---------------------------------------------------------------------------
# Hogrebe consolidate (ratio-based imputation + sum)
# ---------------------------------------------------------------------------


def consolidate(matrix: np.ndarray) -> np.ndarray:
    """Ratio-based iterative imputation of missing values, then per-sample sum.

    This reproduces the ``consolidate()`` function from the canonical Hogrebe
    R script (Hogrebe et al., NComms 2018; SN plugin variant). The intent is:
    when the same site is measured by multiple precursors, missing values in
    one precursor can often be filled in from another precursor's observed
    values (assuming ionization efficiencies differ by a stable multiplicative
    factor across the two precursors).

    Algorithm
    ---------
    1. Drop rows with only 0-1 non-NaN values (not enough evidence for
       ratio estimation) -- matches the R implementation's pre-filter.
    2. Sort rows ascending by row median (lowest-signal precursors first;
       these have the most missing values and benefit most from imputation
       later in the loop).
    3. Iterate:
       a. For each row with any NaN, look at every OTHER row. Where the two
          rows both have values, compute ``median(this_row / other_row)``
          -- this is the estimated ratio of the current precursor's signal
          to the other precursor's signal.
       b. Multiply that ratio by the other row's value at each of ``this
          row``'s NaN positions to get a prediction; average predictions
          across all other rows via the median.
       c. Write imputed values into ``this_row``'s NaN cells.
       d. If a full iteration produced no changes (values were unchanged),
          drop the row with the most remaining NaN and try again.
    4. Sum surviving rows per sample.

    Parameters
    ----------
    matrix : np.ndarray
        2-D array of shape ``(n_precursors, n_samples)`` in LINEAR intensity
        space. NaN represents missing.

    Returns
    -------
    np.ndarray
        1-D array of shape ``(n_samples,)``: the consolidated per-sample
        intensity. Samples where every row was NaN return NaN (not 0).

    Notes
    -----
    * The algorithm is guaranteed to terminate: each iteration either
      imputes a NaN (monotonic progress in # of NaN cells) or drops a
      row (monotonic progress in matrix size). A safety cap of
      ``n_rows * 10`` iterations is also enforced.
    * Zero-row / single-row inputs are handled as edge cases -- see
      the shape checks at the top of the function.
    """
    if matrix.shape[0] == 0:
        return np.full(matrix.shape[1], np.nan)

    # Pre-filter: rows with only 0 or 1 non-NaN values can't produce reliable
    # ratio estimates. Matches R script line 1311.
    non_na_per_row = np.sum(~np.isnan(matrix), axis=1)
    cons = matrix[non_na_per_row > 1].copy()

    if cons.shape[0] == 0:
        return np.full(matrix.shape[1], np.nan)
    if cons.shape[0] == 1:
        # Only one precursor left after pre-filter -- return it verbatim.
        return cons[0].copy()

    # Sort by row median ascending. R script line 1234. Lowest-signal
    # precursors get processed first, so their ratios are computed against
    # higher-signal (more reliable) partners.
    row_medians = np.nanmedian(cons, axis=1)
    cons = cons[np.argsort(row_medians)].copy()

    max_iter = cons.shape[0] * 10  # termination guard
    iteration = 0
    while iteration < max_iter:
        row_has_signal = np.any(~np.isnan(cons), axis=1)
        if not np.any(row_has_signal):
            break
        active = cons[row_has_signal]
        if not np.any(np.isnan(active)):
            # Every value across surviving rows is imputed -- done.
            break

        old_cons = cons.copy()
        n_rows = cons.shape[0]

        for r in range(n_rows):
            row = cons[r]
            na_mask = np.isnan(row)
            if not np.any(na_mask) or np.all(na_mask):
                continue

            predictions: list[np.ndarray] = []
            for other_r in range(n_rows):
                if other_r == r:
                    continue
                other_row = cons[other_r]
                shared = ~np.isnan(row) & ~np.isnan(other_row)
                if not np.any(shared):
                    continue
                ratios = row[shared] / other_row[shared]
                ratios = ratios[np.isfinite(ratios)]
                if len(ratios) == 0:
                    continue
                med_ratio = np.nanmedian(ratios)
                if not np.isfinite(med_ratio) or med_ratio == 0:
                    continue
                predictions.append(med_ratio * other_row[na_mask])

            if predictions:
                cons[r, na_mask] = np.nanmedian(np.array(predictions), axis=0)

        if np.array_equal(old_cons, cons, equal_nan=True):
            # Fixed point without eliminating all NaN -- drop worst row and
            # restart. R script line 1271.
            na_counts = np.sum(np.isnan(cons), axis=1)
            worst_row = int(np.argmax(na_counts))
            if na_counts[worst_row] == 0:
                break
            cons = np.delete(cons, worst_row, axis=0)
            if cons.shape[0] == 0:
                return np.full(old_cons.shape[1], np.nan)

        iteration += 1

    # Sum surviving rows per sample. R script line 1288. Samples where every
    # row is still NaN return NaN (via the ``all_nan`` re-mask), not 0.
    result = np.nansum(cons, axis=0)
    all_nan = np.all(np.isnan(cons), axis=0)
    result[all_nan] = np.nan
    return result
