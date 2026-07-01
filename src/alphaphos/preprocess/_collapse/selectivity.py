"""Per-sample phospho-enrichment selectivity.

Selectivity is defined as::

    fraction_phospho_precursors_per_sample = phospho_precursors / total_precursors

Computed from the RAW PSM DataFrame BEFORE any filtering step, so the metric
reflects the actual enrichment efficiency of the sample prep rather than
whatever survived the alphaPhos filters. This is stamped into
``adata.obs["phospho_selectivity_pct"]`` by :func:`collapse_sites` and used
by the QC dashboard's sample-level panel.

A well-enriched phospho sample typically shows >= 70% selectivity. Values
below that indicate a lot of non-phospho co-purified peptides -- worth
noticing but not a hard failure (some workflows deliberately spike in
non-phospho carrier).
"""

from __future__ import annotations

import pandas as pd

from alphaphos.constants import (
    COL_EG_PRECURSOR_ID,
    COL_R_FILENAME,
    OBS_PHOSPHO_SELECTIVITY_PCT,
    OBS_SAMPLE,
)

PHOSPHO_MARKER = "[Phospho (STY)]"


def compute_selectivity(
    psm_df: pd.DataFrame,
    *,
    sample_col: str = COL_R_FILENAME,
    precursor_col: str = COL_EG_PRECURSOR_ID,
) -> pd.DataFrame:
    """Compute per-sample phospho-enrichment fraction from raw PSMs.

    Deduplicates precursor IDs within each sample first (Spectronaut can
    report the same precursor across multiple rows for multi-position
    localization ambiguity; those extras should not double-count).

    Parameters
    ----------
    psm_df : DataFrame
        A PSM-level DataFrame, typically the direct output of
        :func:`alphaphos.io.read_spectronaut` BEFORE any collapse.
        Must contain the ``sample_col`` and ``precursor_col`` columns.
    sample_col : str
        Column identifying the sample (default ``"R.FileName"``).
    precursor_col : str
        Column carrying the precursor id string (default ``"EG.PrecursorId"``).
        Anything containing the substring ``"[Phospho (STY)]"`` is counted
        as phospho.

    Returns
    -------
    DataFrame
        Long-form table with one row per sample and columns::

            sample                  -- the sample id (from ``sample_col``)
            total_precursors        -- count of unique precursor ids
            phospho_precursors      -- count of unique phospho precursor ids
            phospho_selectivity_pct -- 100 * phospho / total, rounded to 2 dp

    Raises
    ------
    KeyError
        When either required column is missing.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "R.FileName": ["s1", "s1", "s2", "s2"],
    ...     "EG.PrecursorId": [
    ...         "_S[Phospho (STY)]K_.2",
    ...         "_PEPTIDE_.2",
    ...         "_S[Phospho (STY)]K_.2",
    ...         "_T[Phospho (STY)]R_.2",
    ...     ],
    ... })
    >>> compute_selectivity(df).sort_values("sample").reset_index(drop=True)
      sample  total_precursors  phospho_precursors  phospho_selectivity_pct
    0     s1                 2                   1                     50.0
    1     s2                 2                   2                    100.0
    """
    missing = [c for c in (sample_col, precursor_col) if c not in psm_df.columns]
    if missing:
        raise KeyError(
            f"compute_selectivity: required column(s) missing from psm_df: {missing}. "
            f"Have: {sorted(psm_df.columns)[:20]}..."
        )

    # Dedupe precursors within each sample so multi-row precursors (from
    # ambiguous localization) don't inflate counts.
    unique = psm_df[[sample_col, precursor_col]].drop_duplicates().copy()
    unique["is_phospho"] = (
        unique[precursor_col].astype(str).str.contains(PHOSPHO_MARKER, regex=False, na=False)
    )

    summary = (
        unique.groupby(sample_col)
        .agg(
            total_precursors=(precursor_col, "count"),
            phospho_precursors=("is_phospho", "sum"),
        )
        .reset_index()
        .rename(columns={sample_col: OBS_SAMPLE})
    )
    summary[OBS_PHOSPHO_SELECTIVITY_PCT] = (
        summary["phospho_precursors"] / summary["total_precursors"] * 100
    ).round(2)
    return summary
