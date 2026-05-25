"""
Condition-aware Class I masking for phosphosite matrices.

Per-run masking treats every (site, run) cell independently; global_max
uses a single per-site aggregate. This module implements a middle
strategy where the *condition* is the unit of decision:

For each (site, condition):
    N   = number of replicates in this condition
    N_I = number of those replicates where loc_prob >= classI_cutoff

    if N_I / N >= condition_threshold:
        keep ALL quant values for this (site, condition), including
        cells with loc_prob < classI_cutoff (the site is reliably
        localized in the majority of the condition, so non-Class-I
        replicates are taken to be the same site at lower localization
        confidence).
    else:
        mask cells with loc_prob < classI_cutoff to NaN (only trust
        the strongly-localized observations).

Operates on a PeptideCollapse v4 output produced WITHOUT the per-run
mask -- i.e. obtained via ``localization_strategy='global_max'`` and a
permissive cutoff (e.g. 0.0) so the quant matrix still contains the
non-Class-I observations.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Tuple, Union

import numpy as np
import pandas as pd

# Canonical metadata columns emitted by PeptideCollapse (mirrors
# PeptideCollapse._finalize_collapsed_output's internal _meta_set). Any name
# listed here that doesn't exist in a given DataFrame is silently ignored by
# the helpers below.
META_COLS: set[str] = {
    "PTM_Collapse_key",
    "UPD_seq",
    "PTM_localization",
    "Protein_group",
    "Gene_group",
    "kinase_sequence",
    "Protein_Collapse_key",
    "PG.Genes",
    "PG.ProteinGroups",
}

DEFAULT_META_COLS: Tuple[str, ...] = tuple(sorted(META_COLS))


def _coerce_s2c(
    sample_to_condition: Union[Mapping[str, str], pd.Series, pd.DataFrame],
    sample_col: str = "sample",
    condition_col: str = "condition",
) -> pd.Series:
    if isinstance(sample_to_condition, pd.Series):
        return sample_to_condition.astype(str)
    if isinstance(sample_to_condition, pd.DataFrame):
        return (sample_to_condition
                .set_index(sample_col)[condition_col]
                .astype(str))
    return pd.Series(dict(sample_to_condition), dtype=str)


def apply_condition_aware_classI_mask(
    df_sites: pd.DataFrame,
    loc_per_run: pd.DataFrame,
    sample_to_condition: Union[Mapping[str, str], pd.Series, pd.DataFrame],
    classI_cutoff: float = 0.75,
    condition_threshold: float = 0.50,
    meta_cols: Iterable[str] = DEFAULT_META_COLS,
    drop_all_nan: bool = True,
    return_decision_table: bool = False,
):
    """
    Apply the condition-aware Class I masking rule.

    Parameters
    ----------
    df_sites : DataFrame
        UNMASKED site-level matrix (PeptideCollapse global_max output with
        cutoff=0). Rows are sites, columns are metadata + sample-quant
        columns. ``PTM_Collapse_key`` must be one of the metadata columns.
    loc_per_run : DataFrame
        Per-(site, run) localization probability matrix. Index must align
        with the ``PTM_Collapse_key`` values in ``df_sites``; columns must
        align with the sample columns in ``df_sites``.
        Typically obtained as ``pc.site_localization_per_run`` after
        running PeptideCollapse v4.
    sample_to_condition : dict, Series, or DataFrame
        Maps sample column name -> condition label. Samples not in the
        mapping (e.g. blank wells) are left untouched in the matrix but
        do not participate in the per-condition fraction calculation.
    classI_cutoff : float
        Localization-probability cutoff for Class I. Default 0.75
        (Olsen/Mann).
    condition_threshold : float
        Fraction of replicates in a condition that must be Class I for
        the "keep all quants" branch. Default 0.50 (majority rule).
    meta_cols : iterable of str
        Metadata columns to skip when identifying sample columns.
    drop_all_nan : bool
        Drop sites whose entire quant row becomes NaN after masking.
        Default True.
    return_decision_table : bool
        If True, also return a (site x condition) DataFrame of the
        per-(site, condition) Class I fractions, useful for diagnostics.

    Returns
    -------
    DataFrame, or (DataFrame, decision_table) if ``return_decision_table``.
    """
    if not 0 < condition_threshold <= 1:
        raise ValueError(
            f"condition_threshold must be in (0, 1], got {condition_threshold}"
        )
    if not 0 <= classI_cutoff <= 1:
        raise ValueError(
            f"classI_cutoff must be in [0, 1], got {classI_cutoff}"
        )

    s2c = _coerce_s2c(sample_to_condition)

    meta_present = [c for c in meta_cols if c in df_sites.columns]
    sample_cols = [c for c in df_sites.columns if c not in meta_present]
    if "PTM_Collapse_key" not in df_sites.columns:
        raise ValueError(
            "df_sites must contain a 'PTM_Collapse_key' column for alignment."
        )

    # Work in (site x sample) form indexed by PTM_Collapse_key for alignment.
    quant = (df_sites
             .set_index("PTM_Collapse_key")[sample_cols]
             .astype(float)
             .copy())

    loc = loc_per_run.reindex(index=quant.index, columns=quant.columns)

    # Per-cell Class I status: True iff loc >= cutoff. NaN loc => False.
    is_classI = loc.ge(classI_cutoff).fillna(False)

    # Initialize: keep everything.
    keep_mask = pd.DataFrame(True, index=quant.index, columns=quant.columns)

    # Build the decision table while applying the mask.
    decision_rows = {}
    for cond in s2c.unique():
        cond_samples = [s for s in s2c.index[s2c == cond] if s in quant.columns]
        if not cond_samples:
            continue
        N = len(cond_samples)
        N_I = is_classI[cond_samples].sum(axis=1)
        fraction = N_I / N
        decision_rows[cond] = fraction

        # Sites with fraction < threshold: in this condition only, drop
        # the non-Class-I cells (where loc < cutoff or loc is NaN).
        low_frac_sites = fraction < condition_threshold
        if low_frac_sites.any():
            block = is_classI.loc[low_frac_sites, cond_samples]
            keep_block = keep_mask.loc[low_frac_sites, cond_samples] & block
            keep_mask.loc[low_frac_sites, cond_samples] = keep_block

    masked_quant = quant.where(keep_mask, np.nan)

    # Reassemble the full DataFrame.
    out = df_sites.copy().set_index("PTM_Collapse_key")
    out[sample_cols] = masked_quant
    out = out.reset_index()

    if drop_all_nan:
        all_nan = out[sample_cols].isna().all(axis=1)
        out = out.loc[~all_nan].reset_index(drop=True)

    if return_decision_table:
        decision = pd.DataFrame(decision_rows)
        decision.index.name = "PTM_Collapse_key"
        return out, decision

    return out
