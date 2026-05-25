"""Convert alphaPhos collapse output to AnnData (scverse / alphapepttools).

AnnData conventions:
  - rows of .X / .obs are SAMPLES (observations)
  - columns of .X / .var are SITES (variables/features)
  - .layers hold parallel matrices of the same shape as .X
  - .obsm / .varm hold sample- and site-level matrix annotations
  - .uns holds free-form metadata (pipeline parameters, FDR, etc.)

Our ``collapse_sites`` output is the opposite layout (sites × samples), so this
converter transposes the numeric block while preserving per-site metadata
(``META_COLS``) as ``.var`` and any sample-level metadata as ``.obs``.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from alphaphos.preprocess.classify import META_COLS


_KEY_RE = re.compile(
    r"^(?P<protein_group>[^~]+)~(?P<gene>[^_]*)_"
    r"(?P<aa>[A-Z])(?P<position>\d+)_M(?P<multiplicity>\d+)$"
)


def to_anndata(
    sites: pd.DataFrame,
    *,
    loc_per_run: Optional[pd.DataFrame] = None,
    condition_df: Optional[pd.DataFrame] = None,
    decision_table: Optional[pd.DataFrame] = None,
    main_layer: str = "intensity_log2",
):
    """Convert ``collapse_sites`` output to an AnnData object.

    Shape: ``adata.X`` is ``(n_samples × n_sites)`` (samples as observations,
    sites as variables) — the scverse and alphapepttools convention.

    Parameters
    ----------
    sites
        Wide DataFrame from ``collapse_sites``. Rows are sites
        (``PTM_Collapse_key``), columns are sample names + ``META_COLS``.
    loc_per_run
        Optional ``(sites × samples)`` DataFrame of per-(site, run)
        localization probabilities (from ``pc.site_localization_per_run`` or
        the second return of ``collapse_sites``). If provided, stored as
        ``adata.layers["localization"]`` with the same shape as ``adata.X``.
    condition_df
        Optional DataFrame mapping sample names to condition labels (and any
        extra columns). Must contain ``"sample"`` and ``"condition"`` columns;
        extra columns are joined into ``adata.obs`` as-is.
    decision_table
        Optional ``(sites × conditions)`` DataFrame from the condition-aware
        mask. If provided, stored as ``adata.uns["classI_decision_table"]``.
    main_layer
        Name for the principal data layer (also stored under ``adata.X``).
        Default ``"intensity_log2"`` reflects the log2 transform applied by
        PeptideCollapse.

    Returns
    -------
    anndata.AnnData
        Ready for downstream alphapepttools / scanpy / scverse workflows.
        ``obs.index`` is sample names, ``var.index`` is ``PTM_Collapse_key``.

    Notes
    -----
    Parses the ``PTM_Collapse_key`` (format
    ``{ProteinGroup}~{Gene}_{S|T|Y}{position}_M{multiplicity}``) and exposes
    its components as first-class columns in ``adata.var``:
    ``protein_group_id``, ``gene_first``, ``site_aa``, ``site_position``,
    ``multiplicity``. Original ``META_COLS`` columns are also preserved.

    Examples
    --------
    >>> from alphaphos.io import read_spectronaut
    >>> from alphaphos.preprocess import collapse_sites, to_anndata
    >>> df = read_spectronaut("report.parquet", quant_level="MS2",
    ...                       drop_decoys=True, pg_qvalue_max=0.01)
    >>> sites, loc_per_run, decision = collapse_sites(
    ...     df, localization_strategy="condition",
    ...     condition_df=condition_df, return_decision_table=True,
    ... )
    >>> adata = to_anndata(
    ...     sites, loc_per_run=loc_per_run,
    ...     condition_df=condition_df, decision_table=decision,
    ... )
    >>> adata
    AnnData object with n_obs × n_vars = ...
    """
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "anndata is required. Install via `pip install anndata` or it "
            "is already a core dependency of alphaPhos."
        ) from exc

    sample_cols = [c for c in sites.columns if c not in META_COLS]
    if "PTM_Collapse_key" not in sites.columns:
        raise ValueError(
            "sites must contain 'PTM_Collapse_key' column — pass the output "
            "of collapse_sites() directly."
        )

    # site_index: per-site identifier (PTM_Collapse_key)
    site_index = sites["PTM_Collapse_key"].astype(str).values

    # X: samples × sites (transpose the numeric block)
    X = sites[sample_cols].astype(float).T.values  # (n_samples, n_sites)

    # obs: per-sample metadata (sample name as index)
    obs = pd.DataFrame(index=pd.Index(sample_cols, name="sample"))
    if condition_df is not None:
        required = {"sample", "condition"}
        missing = required - set(condition_df.columns)
        if missing:
            raise ValueError(
                f"condition_df must contain columns {sorted(required)}; "
                f"missing: {sorted(missing)}"
            )
        obs = obs.join(
            condition_df.drop_duplicates("sample").set_index("sample"),
            how="left",
        )

    # var: per-site metadata
    meta_present = [c for c in META_COLS if c in sites.columns and c != "PTM_Collapse_key"]
    var = sites[meta_present].copy() if meta_present else pd.DataFrame()
    var.index = pd.Index(site_index, name="PTM_Collapse_key")

    # Parse PTM_Collapse_key components into first-class var columns
    parsed = sites["PTM_Collapse_key"].astype(str).str.extract(_KEY_RE)
    var["protein_group_id"] = parsed["protein_group"].values
    var["gene_first"] = parsed["gene"].values
    var["site_aa"] = parsed["aa"].values
    var["site_position"] = pd.to_numeric(parsed["position"], errors="coerce").values
    var["multiplicity"] = pd.to_numeric(parsed["multiplicity"], errors="coerce").values

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers[main_layer] = X.copy()

    # Localization layer (if provided): align to samples × sites
    if loc_per_run is not None:
        loc_aligned = loc_per_run.reindex(index=site_index, columns=sample_cols)
        adata.layers["localization"] = loc_aligned.astype(float).T.values

    # Decision table (uns; preserved with both row and column labels)
    if decision_table is not None:
        adata.uns["classI_decision_table"] = decision_table

    # Pipeline metadata
    adata.uns["alphaphos"] = {
        "main_layer": main_layer,
        "n_sites": int(adata.n_vars),
        "n_samples": int(adata.n_obs),
    }
    if hasattr(sites, "attrs") and sites.attrs:
        adata.uns["source_attrs"] = dict(sites.attrs)

    return adata
