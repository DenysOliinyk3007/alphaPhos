"""Aggregate a precursor-level Spectronaut proteome report to protein groups.

Companion to :func:`alphaphos.proteome.read_spectronaut_long` -- takes the
precursor DataFrame it returns and produces the same AnnData shape that
:func:`alphaphos.proteome.read_spectronaut_short` produces from the short
report, so downstream (filter / impute / batch-correct / diff-exp / PCA /
pathway enrichment) works identically on either path.
"""

from __future__ import annotations

import logging
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AggregationMethod = Literal["sum", "median", "top3"]


def collapse_proteome(
    prec_df: pd.DataFrame,
    *,
    condition_df: pd.DataFrame | None = None,
    aggregation_method: AggregationMethod = "sum",
    min_precursors: int | None = None,
    quant_column: str | None = None,
) -> ad.AnnData:
    """Aggregate precursors → protein groups → AnnData.

    Parameters
    ----------
    prec_df
        Precursor-level DataFrame as returned by
        :func:`alphaphos.proteome.read_spectronaut_long`.  Must have columns
        ``R.FileName``, ``PG.ProteinGroups``, and the quant column recorded
        in ``prec_df.attrs["resolved_quant_column"]``.
    condition_df
        Optional per-sample metadata to attach to ``adata.obs``; must carry
        a ``sample`` column matching ``R.FileName`` values.
    aggregation_method
        - ``"sum"`` (default): sum precursor quantities per (sample, PG).
          Standard for LFQ.
        - ``"median"``: median of precursor quantities.  Robust to outliers,
          but drops linear-scale intuition and diverges from Spectronaut's
          native PG.Quantity.
        - ``"top3"``: sum of the three most intense precursors per (sample,
          PG).  Approximates the "iBAQ-lite" convention some labs use.
    min_precursors
        If set, drop protein groups whose (sample-level) precursor count in
        *any* sample is below this threshold.  ``None`` (default) disables
        the filter.  Typical value ``2``.
    quant_column
        Override the quantitative column name.  Default: read from
        ``prec_df.attrs["resolved_quant_column"]``.

    Returns
    -------
    AnnData
        Shape ``(n_samples, n_protein_groups)``.  Layout matches
        :func:`read_spectronaut_short`:
        - ``.X`` and ``.layers["intensity_log2"]`` -- log2 of the aggregated
          linear intensity.  Zeros / negatives → NaN.
        - ``.var`` indexed by ``PG_ProteinGroups`` with ``PG_Genes``
          (mode-collapsed if multiple values appear across precursors).
        - ``.obs`` joined with ``condition_df`` (if supplied).
        - ``.uns["alphaphos_proteome"]`` carries a full audit:
          ``{"reader": "long+collapse", "aggregation_method": ...,
          "min_precursors": ..., "quant_column": ..., "stats": {...}}``.
    """
    quant_col = quant_column or prec_df.attrs.get("resolved_quant_column")
    if quant_col is None:
        raise ValueError(
            "collapse_proteome: no quant column provided and "
            "prec_df.attrs['resolved_quant_column'] is not set. "
            "Pass quant_column=... or read the precursor DataFrame via "
            "alphaphos.proteome.read_spectronaut_long."
        )

    for required in ("R.FileName", "PG.ProteinGroups", quant_col):
        if required not in prec_df.columns:
            raise ValueError(f"collapse_proteome: missing required column {required!r} in prec_df")
    if aggregation_method not in ("sum", "median", "top3"):
        raise ValueError(
            f"aggregation_method must be one of ('sum', 'median', 'top3'), "
            f"got {aggregation_method!r}"
        )

    # 1. Aggregate precursors → (sample, PG) linear intensity.
    #    Precursor rows with NaN or non-positive quant are excluded from
    #    the aggregation (they can't contribute useful signal to sum/median).
    quant = pd.to_numeric(prec_df[quant_col], errors="coerce")
    valid = quant.notna() & (quant > 0)
    slim = prec_df.loc[valid, ["R.FileName", "PG.ProteinGroups"]].copy()
    slim[quant_col] = quant[valid].to_numpy()

    grouped = slim.groupby(["R.FileName", "PG.ProteinGroups"], sort=False)[quant_col]
    if aggregation_method == "sum":
        long_agg = grouped.sum()
    elif aggregation_method == "median":
        long_agg = grouped.median()
    else:  # top3
        long_agg = grouped.apply(lambda s: s.nlargest(3).sum())
    long_agg.name = "linear"

    # Optionally filter PGs by precursor count.
    if min_precursors is not None:
        counts = grouped.size().unstack(0, fill_value=0)  # (PGs x samples)
        keep_pgs = counts.max(axis=1) >= min_precursors
        n_before = long_agg.index.get_level_values(1).nunique()
        long_agg = long_agg.loc[long_agg.index.get_level_values(1).isin(keep_pgs[keep_pgs].index)]
        n_after = long_agg.index.get_level_values(1).nunique()
        logger.info(
            "collapse_proteome: min_precursors=%d filter kept %d/%d PGs.",
            min_precursors,
            n_after,
            n_before,
        )

    # 2. Pivot to wide (samples × PGs) linear matrix.
    wide = long_agg.unstack("PG.ProteinGroups")  # (samples x PGs)
    samples = wide.index.astype(str).tolist()
    pgs = wide.columns.astype(str).tolist()
    linear = wide.to_numpy(dtype=np.float64)

    # 3. log2, with non-positive → NaN.
    with np.errstate(invalid="ignore", divide="ignore"):
        X = np.where(linear > 0, np.log2(linear, where=linear > 0), np.nan)

    # 4. Build .var (protein metadata).  Where multiple precursors carry
    #    differing PG.Genes for the same PG (rare but happens with
    #    subgroup rewrites), keep the first non-null value.
    var = pd.DataFrame(index=pd.Index(pgs, name="PG_ProteinGroups"))
    if "PG.Genes" in prec_df.columns:
        gene_map = (
            prec_df[["PG.ProteinGroups", "PG.Genes"]]
            .dropna()
            .drop_duplicates(subset=["PG.ProteinGroups"])
            .set_index("PG.ProteinGroups")["PG.Genes"]
        )
        var["PG_Genes"] = gene_map.reindex(pgs).values
    var["PG_ProteinGroups"] = pgs

    # 5. Build .obs.
    obs = pd.DataFrame(index=pd.Index(samples, name="sample"))
    n_matched, n_unmatched_report, n_unmatched_cond = 0, 0, 0
    if condition_df is not None:
        if "sample" not in condition_df.columns:
            raise ValueError("condition_df must have a 'sample' column to match R.FileName values.")
        cdf = condition_df.set_index("sample")
        matched = obs.index.intersection(cdf.index)
        n_matched = len(matched)
        n_unmatched_report = len(obs.index.difference(cdf.index))
        n_unmatched_cond = len(cdf.index.difference(obs.index))
        obs = obs.join(cdf)

    # 6. Assemble AnnData.
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["intensity_log2"] = adata.X.copy()

    n_missing = int(np.isnan(adata.X).sum())
    total = int(adata.X.size)
    adata.uns["alphaphos_proteome"] = {
        "reader": "long+collapse",
        "aggregation_method": aggregation_method,
        "min_precursors": min_precursors,
        "quant_column": quant_col,
        "source_path": prec_df.attrs.get("path"),
        "stats": {
            "n_samples": int(adata.n_obs),
            "n_proteins": int(adata.n_vars),
            "n_precursor_rows_input": int(prec_df.attrs.get("n_rows_final", len(prec_df))),
            "n_precursor_rows_used_for_agg": int(valid.sum()),
            "n_missing_cells": n_missing,
            "missing_frac": n_missing / total if total else 0.0,
            "n_samples_matched_condition": n_matched,
            "n_samples_unmatched_in_condition_df": n_unmatched_report,
            "n_condition_rows_unmatched_in_report": n_unmatched_cond,
        },
    }
    logger.info(
        "collapse_proteome: %d precursor rows -> %d samples x %d proteins "
        "(method=%s, %.1f%% missing).",
        len(prec_df),
        adata.n_obs,
        adata.n_vars,
        aggregation_method,
        100 * n_missing / total if total else 0.0,
    )
    return adata
