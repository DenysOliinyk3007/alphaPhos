"""Assemble the collapse pipeline's outputs into a single ``AnnData``.

Takes three parallel DataFrames indexed by ``full_key``:

* ``site_quant``    -- log2 intensity, sites x samples
* ``site_loc``      -- localization probability, sites x samples
* ``site_meta``     -- one row per site with all derived columns (gene,
                       protein_group_id, site_aa, site_position,
                       multiplicity, UPD_seq, short_key, pg_key)

The AnnData contract requires the OPPOSITE orientation for ``.X`` (samples
as observations along the first axis). This module handles the transpose
and populates every ``.var``, ``.obs``, ``.layers``, ``.uns`` slot.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from alphaphos.constants import (
    LAYER_INTENSITY_LOG2,
    LAYER_LOCALIZATION,
    OBS_PHOSPHO_SELECTIVITY_PCT,
    OBS_SAMPLE,
    UNS_ALPHAPHOS,
    UNS_SOURCE_ATTRS,
    VAR_CLASSI_WILSON_LB,
    VAR_FRACTION_CLASSI,
    VAR_FULL_KEY,
    VAR_MAX_LOC_PROB,
    VAR_MEAN_LOC_PROB,
    VAR_MIN_LOC_PROB,
    VAR_N_CLASSI_SAMPLES,
    VAR_N_SAMPLES_DETECTED,
)

# Type-only import: anndata.AnnData
try:
    import anndata as ad
except ImportError:  # pragma: no cover - anndata is a hard dep
    ad = None  # type: ignore[assignment]


_NULL_LOGGER = logging.getLogger(__name__)


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
        in the joined columns. Sample ids are compared as strings and a
        duplicated ``sample`` row collapses to its first occurrence.
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
        ``adata.uns['source_attrs']``.
    short_key_collisions : list, optional
        Output of :func:`resolve_short_key_collisions`. Stored at
        ``adata.uns['alphaphos']['short_key_collisions']`` when non-empty, as
        a ``{short_key: [full_key, ...]}`` dict (h5ad-serialisable; the raw
        list of tuples is not).
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
    obs.index.name = OBS_SAMPLE
    if condition_df is not None:
        cdf = condition_df.copy()
        if OBS_SAMPLE not in cdf.columns:
            raise KeyError("condition_df must contain a 'sample' column.")
        # str-cast so integer sample ids still match obs.index; dedup so a
        # repeated sample row can't multiply obs rows.
        cdf[OBS_SAMPLE] = cdf[OBS_SAMPLE].astype(str)
        cdf = cdf.drop_duplicates(OBS_SAMPLE).set_index(OBS_SAMPLE)
        obs = obs.join(cdf, how="left")
    if selectivity is not None:
        # Join by sample id; missing samples get NaN.
        sel = selectivity.set_index(OBS_SAMPLE)[[OBS_PHOSPHO_SELECTIVITY_PCT]]
        obs = obs.join(sel, how="left")

    var = site_meta.copy()
    var.index = var.index.astype(str)
    var.index.name = VAR_FULL_KEY

    # Compute site-level QC columns for adata.var (analog to what to_anndata does).
    var[VAR_N_SAMPLES_DETECTED] = np.asarray((~np.isnan(X)).sum(axis=0)).astype(int)
    with np.errstate(all="ignore"):
        loc_arr = np.asarray(loc_layer)
        var[VAR_MEAN_LOC_PROB] = np.nanmean(loc_arr, axis=0)
        var[VAR_MAX_LOC_PROB] = np.nanmax(loc_arr, axis=0)
        var[VAR_MIN_LOC_PROB] = np.nanmin(loc_arr, axis=0)
    # collapse_precursors allows classI_cutoff=None (gate disabled); the var QC
    # columns still need a threshold, so fall back to the Class-I convention.
    _cutoff = settings.get("classI_cutoff")
    classI_cutoff = 0.75 if _cutoff is None else float(_cutoff)
    is_classI = loc_arr >= classI_cutoff  # NaN loc -> False
    var[VAR_N_CLASSI_SAMPLES] = is_classI.sum(axis=0).astype(int)
    var[VAR_FRACTION_CLASSI] = var[VAR_N_CLASSI_SAMPLES] / max(loc_arr.shape[0], 1)

    # Per-site 95% Jeffreys/Wilson lower confidence bound on the Class-I fraction.
    # See alphaphos.preprocess.classI_wilson — cheap, always populated, backs the
    # ``strategy="wilson"`` and ``wilson_threshold_sensitivity`` public APIs.
    # For Wilson we need k <= n; because ``VAR_N_CLASSI_SAMPLES`` is counted on
    # the loc matrix and ``VAR_N_SAMPLES_DETECTED`` on the intensity matrix (which
    # may be more aggressively filtered — noise floor, per-run mask, etc.), we
    # intersect the two masks so k is the count of samples that are BOTH
    # Class-I and quantified.
    from alphaphos.preprocess.classI_wilson import wilson_lower_bound

    intensity_observed = ~np.isnan(X)
    k_wilson = (is_classI & intensity_observed).sum(axis=0).astype(int)
    var[VAR_CLASSI_WILSON_LB] = wilson_lower_bound(
        k_wilson,
        var[VAR_N_SAMPLES_DETECTED].to_numpy(),
    )

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers[LAYER_INTENSITY_LOG2] = X.copy()
    adata.layers[LAYER_LOCALIZATION] = loc_layer

    # uns package: everything alphaPhos-specific goes under 'alphaphos'.
    alphaphos_ns: dict[str, Any] = {
        "version": version,
        "pipeline_params": dict(settings),
        "stats": dict(stats),
    }
    if short_key_collisions:
        # dict[str, list[str]] round-trips through h5ad; the (str, list)
        # tuples from resolve_short_key_collisions do not.
        alphaphos_ns["short_key_collisions"] = {
            short: list(fulls) for short, fulls in short_key_collisions
        }
    if decision_table is not None:
        alphaphos_ns["classI_decision_table"] = decision_table
    adata.uns[UNS_ALPHAPHOS] = alphaphos_ns
    if source_attrs is not None:
        adata.uns[UNS_SOURCE_ATTRS] = dict(source_attrs)

    logger.info(
        "Assembled AnnData: %d samples x %d sites; obs cols=%s; layers=%s.",
        adata.n_obs,
        adata.n_vars,
        list(adata.obs.columns),
        list(adata.layers.keys()),
    )
    return adata
