"""QC metric computations — tidy DataFrames per dashboard section.

Each function takes an AnnData (and optionally supplementary inputs) and
returns a small DataFrame ready for plotting. Kept separate from
:mod:`alphaphos.qc.plots` so the metrics are independently testable and
the dashboard can render either the cached tables or fresh computations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# §1. Pipeline waterfall
# ---------------------------------------------------------------------------


def compute_pipeline_waterfall(adata: ad.AnnData) -> pd.DataFrame:
    """Per-step PSM-row counts from ``adata.uns["source_attrs"]``.

    Reads the ``.attrs`` dict that ``alphaphos.io.read_psm`` populates and
    re-emits as a tidy step-by-step waterfall: row counts at each filter
    stage. Returns empty if no source attrs are present.

    Returns
    -------
    pd.DataFrame with columns ``stage`` (str), ``n_rows`` (int),
    ``dropped`` (int — drop count vs previous stage), ``pct_dropped``.
    """
    source = adata.uns.get("source_attrs", {}) or {}
    stages = [
        ("loaded", source.get("n_rows_loaded")),
        ("after_qvalue_and_decoys", source.get("n_rows_returned")),
        # n_rows_after_contaminant_filter is post-contaminant, pre-top_n
        ("after_contaminants", source.get("n_rows_after_contaminant_filter")),
        ("after_top_n_attribution", source.get("n_rows_after_top_n")),
    ]
    rows = []
    prev = None
    for stage, n in stages:
        if n is None:
            continue
        dropped = int(prev - n) if (prev is not None and n is not None) else 0
        pct = (100.0 * dropped / prev) if prev else 0.0
        rows.append(
            {
                "stage": stage,
                "n_rows": int(n),
                "dropped_from_prev": dropped,
                "pct_dropped_from_prev": round(pct, 2),
            }
        )
        prev = n
    # Final stage: after collapse to sites (n_sites in the AnnData)
    rows.append(
        {
            "stage": "after_collapse_to_sites",
            "n_rows": int(adata.n_vars),
            "dropped_from_prev": 0,  # site collapse changes units, not a drop
            "pct_dropped_from_prev": 0.0,
        }
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §2. Sample-level QC
# ---------------------------------------------------------------------------


def compute_sty_ratio(adata: ad.AnnData) -> pd.DataFrame:
    """Per-sample S/T/Y composition of detected sites.

    For each sample, counts how many S, T, Y sites have a non-NaN value in
    that sample's column. Returns a long-form DataFrame ready for a stacked
    bar plot.

    Returns
    -------
    pd.DataFrame with columns ``sample``, ``site_aa`` (S/T/Y), ``n``, ``pct``.
    """
    if "site_aa" not in adata.var.columns:
        return pd.DataFrame(columns=["sample", "site_aa", "n", "pct"])

    site_aa = adata.var["site_aa"].astype(str).values  # (n_sites,)
    present = ~np.isnan(np.asarray(adata.X, dtype=float))  # (n_obs, n_sites)
    rows = []
    for i, s in enumerate(adata.obs.index):
        mask = present[i, :]
        for aa in ("S", "T", "Y"):
            n = int(np.sum(mask & (site_aa == aa)))
            rows.append({"sample": str(s), "site_aa": aa, "n": n})
    df = pd.DataFrame(rows)
    totals = df.groupby("sample")["n"].transform("sum")
    df["pct"] = (df["n"] / totals.replace(0, np.nan) * 100).round(2).fillna(0.0)
    return df


def compute_localization_distribution(adata: ad.AnnData) -> pd.DataFrame:
    """Per-sample localization-probability distribution from
    ``adata.layers["localization"]``.

    Returns one row per (sample, site) observation with a non-NaN loc value
    — long-form, ready for a violin/density plot.

    Returns
    -------
    pd.DataFrame with columns ``sample``, ``loc_prob``.
    """
    if "localization" not in adata.layers:
        return pd.DataFrame(columns=["sample", "loc_prob"])
    loc = np.asarray(adata.layers["localization"], dtype=float)  # (n_obs, n_sites)
    rows = []
    for i, s in enumerate(adata.obs.index):
        vals = loc[i, :]
        vals = vals[~np.isnan(vals)]
        for v in vals:
            rows.append({"sample": str(s), "loc_prob": float(v)})
    return pd.DataFrame(rows)


def compute_multiplicity_distribution(adata: ad.AnnData) -> pd.DataFrame:
    """Per-sample M1/M2/M3 (multiplicity) composition of detected sites.

    Returns
    -------
    pd.DataFrame with columns ``sample``, ``multiplicity`` (1/2/3+), ``n``, ``pct``.
    """
    if "multiplicity" not in adata.var.columns:
        return pd.DataFrame(columns=["sample", "multiplicity", "n", "pct"])

    mult = pd.to_numeric(adata.var["multiplicity"], errors="coerce").fillna(0).astype(int).values
    present = ~np.isnan(np.asarray(adata.X, dtype=float))
    rows = []
    for i, s in enumerate(adata.obs.index):
        mask = present[i, :]
        for m in (1, 2, 3):
            tag = "3+" if m == 3 else str(m)
            if m == 3:
                n = int(np.sum(mask & (mult >= 3)))
            else:
                n = int(np.sum(mask & (mult == m)))
            rows.append({"sample": str(s), "multiplicity": tag, "n": n})
    df = pd.DataFrame(rows)
    totals = df.groupby("sample")["n"].transform("sum")
    df["pct"] = (df["n"] / totals.replace(0, np.nan) * 100).round(2).fillna(0.0)
    return df


def compute_missingness(adata: ad.AnnData) -> pd.DataFrame:
    """Per-sample missing-value count + fraction.

    Returns
    -------
    pd.DataFrame with columns ``sample``, ``n_missing``, ``n_total``,
    ``pct_missing``, ``condition`` (if obs has it).
    """
    X = np.asarray(adata.X, dtype=float)
    rows = []
    for i, s in enumerate(adata.obs.index):
        missing = int(np.sum(np.isnan(X[i, :])))
        total = int(X.shape[1])
        row = {
            "sample": str(s),
            "n_missing": missing,
            "n_total": total,
            "pct_missing": round(100 * missing / max(total, 1), 2),
        }
        if "condition" in adata.obs.columns:
            row["condition"] = str(adata.obs["condition"].iloc[i])
        rows.append(row)
    return pd.DataFrame(rows)


def compute_n_classI_per_sample(
    adata: ad.AnnData,
    *,
    classI_cutoff: float = 0.75,
) -> pd.DataFrame:
    """Per-sample count of Class I-localized sites (loc >= classI_cutoff)."""
    if "localization" not in adata.layers:
        return pd.DataFrame(columns=["sample", "n_classI", "n_present"])
    loc = np.asarray(adata.layers["localization"], dtype=float)
    rows = []
    for i, s in enumerate(adata.obs.index):
        vals = loc[i, :]
        present_mask = ~np.isnan(vals)
        n_present = int(present_mask.sum())
        n_classI = int(np.sum((vals >= classI_cutoff) & present_mask))
        rows.append(
            {
                "sample": str(s),
                "n_classI": n_classI,
                "n_present": n_present,
                "fraction_classI": round(n_classI / max(n_present, 1), 4),
            }
        )
    return pd.DataFrame(rows)


def compute_per_condition_cv_by_aa(adata: ad.AnnData) -> pd.DataFrame:
    """Per-condition CV (within-replicates), stratified by site_aa (S/T/Y).

    For each (condition, site, AA), computes CV = stdev(linear-space) /
    mean(linear-space). adata.X is log2 → exponentiated before CV. Returns
    one row per (condition, site, site_aa).

    Returns
    -------
    pd.DataFrame with columns ``condition``, ``site_aa``, ``cv``.
    """
    if "condition" not in adata.obs.columns or "site_aa" not in adata.var.columns:
        return pd.DataFrame(columns=["condition", "site_aa", "cv"])

    X = np.asarray(adata.X, dtype=float)
    lin = np.power(2.0, X)
    site_aa = adata.var["site_aa"].astype(str).values
    conditions = adata.obs["condition"].astype(str).values

    rows = []
    for cond in np.unique(conditions):
        mask = conditions == cond
        block = lin[mask, :]  # (n_replicates, n_sites)
        if block.shape[0] < 2:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nanmean(block, axis=0)
            sd = np.nanstd(block, axis=0, ddof=1)
            cv = sd / mean
        for j, aa in enumerate(site_aa):
            if aa not in ("S", "T", "Y"):
                continue
            cv_j = cv[j]
            if not np.isfinite(cv_j):
                continue
            rows.append({"condition": cond, "site_aa": aa, "cv": float(cv_j)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §3. Reproducibility
# ---------------------------------------------------------------------------


def compute_replicate_correlation(adata: ad.AnnData) -> pd.DataFrame:
    """Pearson correlation matrix between sample columns of adata.X
    (using only sites observed in both samples).

    Returns
    -------
    pd.DataFrame (n_samples × n_samples) of Pearson r, indexed/columned by sample.
    """
    X = np.asarray(adata.X, dtype=float)  # (n_obs, n_sites)
    # Compute sample × sample correlation
    samples = list(adata.obs.index.astype(str))
    df = pd.DataFrame(X, index=samples).T  # sites as rows, samples as columns
    return df.corr(method="pearson", min_periods=10)


def compute_sample_dendrogram_order(adata: ad.AnnData) -> list[str]:
    """Order samples by hierarchical clustering (Ward linkage on Euclidean).

    Used by the dashboard to re-order the correlation heatmap. Falls back
    to original order if scipy isn't importable.
    """
    try:
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import pdist
    except ImportError:
        return list(adata.obs.index.astype(str))

    X = np.asarray(adata.X, dtype=float)
    # Replace NaN with column means to avoid pdist errors
    col_means = np.nanmean(X, axis=0)
    X_filled = np.where(np.isnan(X), col_means, X)
    if X_filled.shape[0] < 2:
        return list(adata.obs.index.astype(str))
    try:
        Z = hierarchy.linkage(pdist(X_filled), method="ward")
        leaves = hierarchy.leaves_list(Z)
    except Exception as exc:
        logger.info("dendrogram failed (%s); using original sample order", exc)
        return list(adata.obs.index.astype(str))
    samples = list(adata.obs.index.astype(str))
    return [samples[i] for i in leaves]


# ---------------------------------------------------------------------------
# §4. Class I diagnostics
# ---------------------------------------------------------------------------


def compute_classI_comparison(
    adata: ad.AnnData,
    *,
    classI_cutoff: float = 0.75,
) -> pd.DataFrame:
    """Compare per-cell binary Class I vs the condition-aware mask.

    Counts how many (site, sample) cells survive each policy:

    - ``per_cell_binary``: cells where ``loc >= classI_cutoff`` AND the cell
      is observed
    - ``condition_aware``: cells with a present value in ``adata.X`` (which
      already reflects the condition-aware mask applied at collapse time)

    The difference is the "recovery" by the condition-aware policy — cells
    that wouldn't make the binary cut but were kept because the rest of the
    condition's replicates support the site.

    Returns
    -------
    pd.DataFrame with columns ``policy``, ``n_cells``.
    """
    rows: list[dict] = []
    if "localization" in adata.layers:
        loc = np.asarray(adata.layers["localization"], dtype=float)
        present_mask = ~np.isnan(loc)
        binary = int(np.sum((loc >= classI_cutoff) & present_mask))
        rows.append({"policy": "per_cell_binary (loc >= cutoff)", "n_cells": binary})

    X = np.asarray(adata.X, dtype=float)
    cond_aware = int(np.sum(~np.isnan(X)))
    rows.append({"policy": "condition_aware (present in adata.X)", "n_cells": cond_aware})

    return pd.DataFrame(rows)


def compute_contaminant_breakdown(
    psm_df: pd.DataFrame,
    *,
    contaminant_filter_attr: str = "contaminants_filter_applied",
    n_rows_after_contaminants_attr: str = "n_rows_after_contaminant_filter",
) -> pd.DataFrame:
    """Top-15 contaminant genes by PSM-row count (from a raw PSM DataFrame).

    The PSM DataFrame should be the output of ``alphaphos.io.read_psm``
    with ``drop_contaminants=False`` so the contaminant rows are still
    present. We then re-run the filter to count what *would have been*
    dropped per gene.

    Returns empty DataFrame if PSM df has no contaminant rows or required
    columns are missing.
    """
    from alphaphos.preprocess.contaminants import (
        DEFAULT_CONTAMINANT_PREFIXES,
        _is_contaminant_protein,
        get_default_contaminants_fasta,
        parse_fasta_accessions,
    )

    if "PG.ProteinGroups" not in psm_df.columns or "PG.Genes" not in psm_df.columns:
        return pd.DataFrame(columns=["gene", "n_rows"])

    accessions = parse_fasta_accessions(get_default_contaminants_fasta())

    def _row_is_cont(value):
        if pd.isna(value):
            return False
        proteins = [p for p in str(value).split(";") if p.strip()]
        if not proteins:
            return False
        return all(
            _is_contaminant_protein(p, accessions, DEFAULT_CONTAMINANT_PREFIXES) for p in proteins
        )

    mask = psm_df["PG.ProteinGroups"].map(_row_is_cont)
    dropped = psm_df.loc[mask]
    if dropped.empty:
        return pd.DataFrame(columns=["gene", "n_rows"])
    counts = dropped["PG.Genes"].fillna("(no_gene)").value_counts().head(15).reset_index()
    counts.columns = ["gene", "n_rows"]
    return counts


# ---------------------------------------------------------------------------
# §5. Imputation diagnostics
# ---------------------------------------------------------------------------


def compute_imputation_summary(impute_audit: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-cell audit DataFrame from ``impute_hybrid(return_audit=True)``.

    Parameters
    ----------
    impute_audit
        DataFrame with columns ``sample_idx``, ``site_idx``, ``strategy``
        (``"MAR_KNN"`` or ``"MNAR_Gaussian"``).

    Returns
    -------
    pd.DataFrame with columns ``sample_idx``, ``strategy``, ``n``.
    """
    if impute_audit is None or impute_audit.empty:
        return pd.DataFrame(columns=["sample_idx", "strategy", "n"])
    return (
        impute_audit.groupby(["sample_idx", "strategy"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
