"""Assemble the collapse pipeline's outputs into a single ``AnnData``.

At the end of the pipeline we have three parallel DataFrames indexed by
``full_key``:

* ``site_quant``    -- log2 intensity, sites x samples
* ``site_loc``      -- localization probability, sites x samples
* ``site_meta``     -- one row per site with all derived columns (gene,
                       protein_group_id, site_aa, site_position,
                       multiplicity, UPD_seq, short_key, pg_key)

The AnnData contract requires the OPPOSITE orientation for ``.X`` (samples
as observations along the first axis). This module handles the transpose
and populates every ``.var``, ``.obs``, ``.layers``, ``.uns`` slot in one
call so users get a fully-ready object with no missing metadata.

For advanced users, :mod:`alphaphos.preprocess.anndata` also exposes a
lower-level ``to_anndata`` that accepts externally-collapsed data; the
function here calls into it for the actual construction.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

# Type-only import: anndata.AnnData
try:
    import anndata as ad
except ImportError:  # pragma: no cover - anndata is a hard dep
    ad = None  # type: ignore[assignment]


_NULL_LOGGER = logging.getLogger("alphaphos.preprocess._collapse.output_format")


def assemble_anndata(
    site_quant: pd.DataFrame,
    site_loc: pd.DataFrame,
    site_meta: pd.DataFrame,
    *,
    condition_df: pd.DataFrame | None = None,
    settings: dict[str, Any],
    stats: dict[str, Any],
    selectivity: pd.DataFrame | None = None,
    decision_table: pd.DataFrame | None = None,
    source_attrs: dict[str, Any] | None = None,
    short_key_collisions: list | None = None,
    version: str = "0.0.0",
    logger: logging.Logger = _NULL_LOGGER,
) -> ad.AnnData:
    """Build an ``AnnData`` from collapse pipeline outputs.

    Parameters
    ----------
    site_quant : DataFrame
        ``(sites x samples)`` log2 intensity matrix, indexed by ``full_key``.
        Column order becomes the AnnData ``obs_names`` order.
    site_loc : DataFrame
        Same shape/index as ``site_quant``.
    site_meta : DataFrame
        One row per site (indexed by ``full_key``) with the following
        columns: ``short_key``, ``pg_key``, ``protein_group_id``, ``gene``,
        ``site_aa``, ``site_position`` (aliased from ``absolute_position``),
        ``multiplicity``, ``UPD_seq``. Additional columns are silently
        forwarded to ``adata.var`` as-is.
    condition_df : DataFrame, optional
        Sample metadata. Must contain ``sample`` and ``condition`` columns;
        extra columns are joined into ``adata.obs``. Samples in the quant
        matrix that aren't in ``condition_df`` are still kept -- with NaN
        in the joined columns.
    settings : dict
        The resolved settings dict (from :func:`resolve_settings`). Stored
        as-is at ``adata.uns['alphaphos']['pipeline_params']``.
    stats : dict
        Per-stage counts / percentages collected during the pipeline.
        Stored at ``adata.uns['alphaphos']['stats']``.
    selectivity : DataFrame, optional
        Output of :func:`compute_selectivity`. When provided, its
        ``phospho_selectivity_pct`` column is joined into ``adata.obs``
        by sample id.
    decision_table : DataFrame, optional
        Per-(site, condition) Class-I fraction table (only produced when
        the condition-aware masking strategy is active). Stored at
        ``adata.uns['alphaphos']['classI_decision_table']``.
    source_attrs : dict, optional
        The ``psm_df.attrs`` dict from ``read_spectronaut``. Stored at
        ``adata.uns['source_attrs']``. Powers the QC dashboard's PSM
        pipeline waterfall panel.
    short_key_collisions : list, optional
        Output of :func:`resolve_short_key_collisions`. Stored at
        ``adata.uns['alphaphos']['short_key_collisions']`` when non-empty
        so users can inspect which gene:site labels got suffixed.
    version : str
        The alphaPhos version string, stored at
        ``adata.uns['alphaphos']['version']`` for provenance.

    Returns
    -------
    anndata.AnnData
        Shape ``(n_samples, n_sites)``:

        * ``.X`` == ``layers["intensity_log2"]`` (log2 intensities)
        * ``layers["localization"]`` (per-cell loc probability)
        * ``.var.index`` = ``full_key`` (``"PG|Gene|Site|Mult"``)
        * ``.obs.index`` = sample id
    """
    if ad is None:
        raise ImportError(
            "anndata is required to assemble collapse output. Install with `pip install anndata`."
        )

    # Sanity: matrices must be identically shaped and index-aligned.
    if list(site_quant.index) != list(site_loc.index):
        raise ValueError("site_quant and site_loc must share the same index.")
    if list(site_quant.columns) != list(site_loc.columns):
        raise ValueError("site_quant and site_loc must share the same columns.")
    if list(site_quant.index) != list(site_meta.index):
        raise ValueError("site_quant and site_meta must share the same index.")

    # AnnData wants (obs, var) == (samples, sites); our quant is (sites, samples).
    X = site_quant.T.to_numpy(dtype=np.float32, copy=True)
    loc_layer = site_loc.T.to_numpy(dtype=np.float32, copy=True)

    obs = pd.DataFrame(index=site_quant.columns.astype(str))
    obs.index.name = "sample"
    if condition_df is not None:
        cdf = condition_df.copy()
        if "sample" not in cdf.columns:
            raise KeyError("condition_df must contain a 'sample' column.")
        cdf = cdf.set_index("sample")
        obs = obs.join(cdf, how="left")
    if selectivity is not None:
        # Join by sample id; missing samples get NaN.
        sel = selectivity.set_index("sample")[["phospho_selectivity_pct"]]
        obs = obs.join(sel, how="left")

    var = site_meta.copy()
    var.index = var.index.astype(str)
    var.index.name = "full_key"

    # Compute site-level QC columns for adata.var (analog to what to_anndata does).
    var["n_samples_detected"] = np.asarray((~np.isnan(X)).sum(axis=0)).astype(int)
    with np.errstate(all="ignore"):
        loc_arr = np.asarray(loc_layer)
        var["mean_loc_prob"] = np.nanmean(loc_arr, axis=0)
        var["max_loc_prob"] = np.nanmax(loc_arr, axis=0)
        var["min_loc_prob"] = np.nanmin(loc_arr, axis=0)
    classI_cutoff = float(settings.get("classI_cutoff", 0.75))
    is_classI = loc_arr >= classI_cutoff  # NaN loc -> False
    var["n_classI_samples"] = is_classI.sum(axis=0).astype(int)
    var["fraction_classI"] = var["n_classI_samples"] / max(loc_arr.shape[0], 1)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["intensity_log2"] = X.copy()
    adata.layers["localization"] = loc_layer

    # uns package: everything alphaPhos-specific goes under 'alphaphos'.
    alphaphos_ns: dict[str, Any] = {
        "version": version,
        "pipeline_params": dict(settings),
        "stats": dict(stats),
    }
    if short_key_collisions:
        alphaphos_ns["short_key_collisions"] = short_key_collisions
    if decision_table is not None:
        alphaphos_ns["classI_decision_table"] = decision_table
    adata.uns["alphaphos"] = alphaphos_ns
    if source_attrs is not None:
        adata.uns["source_attrs"] = dict(source_attrs)

    logger.info(
        "Assembled AnnData: %d samples x %d sites; obs cols=%s; layers=%s.",
        adata.n_obs,
        adata.n_vars,
        list(adata.obs.columns),
        list(adata.layers.keys()),
    )
    return adata
