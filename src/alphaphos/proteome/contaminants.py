"""Contaminant protein groups in a protein-level AnnData: measure, then remove.

Uses :func:`alphaphos.io.contaminants.contaminant_mask`, the same definition
the precursor-level reader filter and QC metric use, so a protein report and
a PSM report of one study agree on what a contaminant is.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from alphaphos.io.contaminants import DEFAULT_CONTAMINANT_PREFIXES, contaminant_mask
from alphaphos.proteome.io_short import PG_INDEX
from alphaphos.qc.psm_metrics import DEFAULT_CONTAMINANT_FLAG_FRACTION


def protein_contaminant_mask(
    adata: ad.AnnData,
    *,
    contaminants_fasta: str | Path | None = None,
    prefix_patterns: tuple[str, ...] = DEFAULT_CONTAMINANT_PREFIXES,
    require_all_contaminant: bool = True,
) -> np.ndarray:
    """True for each protein group (``adata.var``) that is a contaminant."""
    groups = adata.var[PG_INDEX] if PG_INDEX in adata.var else adata.var_names.to_series()
    return contaminant_mask(
        groups,
        contaminants_fasta=contaminants_fasta,
        prefix_patterns=prefix_patterns,
        require_all_contaminant=require_all_contaminant,
    )


def contaminant_fraction(
    adata: ad.AnnData,
    *,
    contaminants_fasta: str | Path | None = None,
    prefix_patterns: tuple[str, ...] = DEFAULT_CONTAMINANT_PREFIXES,
    require_all_contaminant: bool = True,
    flag_fraction_threshold: float = DEFAULT_CONTAMINANT_FLAG_FRACTION,
) -> pd.DataFrame:
    """Per-run fraction of protein intensity from contaminant protein groups.

    Intensities are summed on the linear scale (``2 ** intensity_log2``;
    missing values count as 0).  Returns the columns of
    :func:`alphaphos.qc.compute_contaminant_fraction_per_sample` that apply
    at protein level -- ``contaminant_intensity``, ``total_intensity``,
    ``contaminant_fraction``, ``n_contaminant_proteins``,
    ``n_total_proteins``, ``is_flagged`` -- so ``ap.viz.contaminant_fraction``
    plots either.
    """
    mask = protein_contaminant_mask(
        adata,
        contaminants_fasta=contaminants_fasta,
        prefix_patterns=prefix_patterns,
        require_all_contaminant=require_all_contaminant,
    )
    log2 = adata.layers.get("intensity_log2", adata.X)
    log2 = np.asarray(log2, dtype=np.float64)
    linear = np.where(np.isfinite(log2), np.exp2(log2), 0.0)
    detected = np.isfinite(log2)

    contaminant = linear[:, mask].sum(axis=1)
    total = linear.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = np.where(total > 0, contaminant / total, 0.0)
    out = pd.DataFrame(
        {
            "contaminant_intensity": contaminant,
            "total_intensity": total,
            "contaminant_fraction": fraction,
            "n_contaminant_proteins": detected[:, mask].sum(axis=1),
            "n_total_proteins": detected.sum(axis=1),
            "is_flagged": fraction >= flag_fraction_threshold,
        },
        index=pd.Index(adata.obs_names.astype(str), name="sample"),
    )
    out.attrs["provenance"] = {
        "level": "protein",
        "n_contaminant_protein_groups": int(mask.sum()),
        "require_all_contaminant": require_all_contaminant,
        "flag_fraction_threshold": flag_fraction_threshold,
        "cohort_median_fraction": float(np.median(fraction)) if len(fraction) else float("nan"),
    }
    return out


def drop_contaminant_proteins(
    adata: ad.AnnData,
    *,
    contaminants_fasta: str | Path | None = None,
    prefix_patterns: tuple[str, ...] = DEFAULT_CONTAMINANT_PREFIXES,
    require_all_contaminant: bool = True,
) -> ad.AnnData:
    """Copy of ``adata`` without contaminant protein groups; count in ``uns``."""
    mask = protein_contaminant_mask(
        adata,
        contaminants_fasta=contaminants_fasta,
        prefix_patterns=prefix_patterns,
        require_all_contaminant=require_all_contaminant,
    )
    out = adata[:, ~mask].copy()
    stats = out.uns.setdefault("alphaphos_proteome", {}).setdefault("stats", {})
    stats["n_contaminant_proteins_dropped"] = int(mask.sum())
    stats["n_proteins"] = int(out.n_vars)
    return out
