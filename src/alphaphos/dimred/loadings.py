"""Loadings-analysis accessors for PCA results.

The core :func:`alphaphos.dimred.pca` attaches loadings to
``adata.varm["PCs"]`` as a raw numpy array indexed positionally by
``adata.var_names`` (alphaPhos ``Protein|Gene|Site|Mult`` full keys).

This module provides two accessors that convert the raw loadings into
shapes the downstream enrichment submodules accept directly:

* :func:`loadings_for_enrichment` -- DataFrame indexed by canonical
  ``Protein_AApos`` site IDs, one column per PC.  Drops straight into
  :func:`alphaphos.enrichment.gsea`,
  :func:`alphaphos.enrichment.kinase_activity`, and
  :func:`alphaphos.enrichment.ora` (see docstring for usage).
* :func:`feature_variance_contribution` -- per-feature % of the top-K PC
  variance explained.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)

DedupPolicy = Literal["abs_max", "error"]


def loadings_for_enrichment(
    adata: ad.AnnData,
    *,
    n_components: int | None = None,
    dedup: DedupPolicy = "abs_max",
) -> pd.DataFrame:
    """PCA loadings reshaped for direct use with the enrichment submodules.

    Converts the raw ``adata.varm["PCs"]`` matrix into a DataFrame with

    * **index** -- canonical ``Protein_AApos`` site IDs (e.g. ``Q9H307_S100``)
    * **columns** -- ``PC1..PCK``

    which is the shape accepted by :func:`alphaphos.enrichment.gsea`,
    :func:`alphaphos.enrichment.kinase_activity`, and
    :func:`alphaphos.enrichment.ora`.  No wrappers required::

        loadings = ap.dimred.loadings_for_enrichment(adata)

        # Preranked GSEA on PC1
        ap.enrichment.gsea(loadings["PC1"], libraries=libs)

        # KSEA on PC1 (kinase_activity auto-detects the canonical index)
        ap.enrichment.kinase_activity(loadings, stat_col="PC1")

        # Categorical ORA on top-200 |loaders| of PC1
        hits = loadings["PC1"].abs().nlargest(200).index.tolist()
        ap.enrichment.ora(hits, loadings.index.tolist(), libraries=libs)

    Parameters
    ----------
    adata
        AnnData that has been through :func:`alphaphos.dimred.pca`.
    n_components
        Number of PCs to include (``None`` = all available).
    dedup
        Collision policy when two features canonicalize to the same
        site ID (typically different multiplicities of the same site):

        - ``"abs_max"`` (default) -- keep the row whose sum-of-squares
          across PC columns is largest.  Follows the convention already
          used in :func:`alphaphos.enrichment.ksea` for site-level dedup.
        - ``"error"`` -- raise ``ValueError`` on any collision.

    Returns
    -------
    DataFrame indexed by canonical site IDs.  Features whose key does
    not parse to a valid alphaPhos key are dropped (with an info log).
    """
    # Deferred import: keep dimred independent of enrichment at module scope.
    from alphaphos.enrichment.db import site_id
    from alphaphos.enrichment.matching import parse_alphaphos_key

    if "PCs" not in adata.varm:
        raise KeyError("adata.varm['PCs'] not found. Run alphaphos.dimred.pca(adata) first.")
    loadings = np.asarray(adata.varm["PCs"])
    k = loadings.shape[1] if n_components is None else min(n_components, loadings.shape[1])

    var_names = adata.var_names.astype(str)
    canonical: list[str | None] = []
    for key in var_names:
        parsed = parse_alphaphos_key(str(key))
        canonical.append(
            site_id(parsed.protein, parsed.residue, parsed.position) if parsed else None
        )

    df = pd.DataFrame(
        loadings[:, :k],
        columns=[f"PC{i + 1}" for i in range(k)],
        index=pd.Index(canonical, name="site_id"),
    )

    n_unparsed = df.index.isna().sum()
    if n_unparsed:
        logger.info(
            "loadings_for_enrichment: dropped %d/%d features whose var_name did not parse "
            "as a Protein|Gene|Site|Mult key",
            n_unparsed,
            len(var_names),
        )
        df = df.loc[df.index.notna()]

    # Collision handling
    collision_mask = df.index.duplicated(keep=False)
    n_collisions = int(collision_mask.sum())
    if n_collisions:
        if dedup == "error":
            colliding = df.index[collision_mask].unique().tolist()[:5]
            raise ValueError(
                f"loadings_for_enrichment: {n_collisions} rows collide on "
                f"canonical site_id (e.g. {colliding}). Pass dedup='abs_max' to "
                "keep the highest-magnitude row per site, or pre-collapse "
                "multiplicities upstream."
            )
        # abs_max: within each canonical site, keep the row with max sum-of-squares.
        # Using positional indices (not labels) is essential: labels are the
        # duplicated site_ids themselves, so .loc[label] would grab all colliders.
        magnitudes = (df.values**2).sum(axis=1)
        tmp = df.reset_index()
        tmp["_mag"] = magnitudes
        # groupby(sort=False) preserves first-appearance order of sites;
        # idxmax() returns the RangeIndex position of the winning row per site.
        keep_positions = tmp.groupby("site_id", sort=False)["_mag"].idxmax().to_numpy()
        df = df.iloc[keep_positions]
        logger.info(
            "loadings_for_enrichment: %d rows collided on canonical site_id; "
            "kept max-|loading| per site (%d unique sites retained)",
            n_collisions,
            df.shape[0],
        )

    # Store metadata as tuples: pandas concat compares .attrs via ==,
    # which on numpy arrays returns element-wise booleans and breaks.
    pca_ns = adata.uns.get("pca", {}) if hasattr(adata, "uns") else {}
    if "variance_ratio" in pca_ns:
        df.attrs["variance_ratio"] = tuple(np.asarray(pca_ns["variance_ratio"])[:k].tolist())
    if "method" in pca_ns:
        df.attrs["method"] = pca_ns["method"]
    return df


def feature_variance_contribution(
    adata: ad.AnnData,
    *,
    n_components: int | None = None,
) -> pd.DataFrame:
    """Per-feature share of the top-K PC variance.

    For feature ``j`` and PC ``k``, the contribution to that PC's variance
    is ``loading[j,k]**2`` (loadings are unit-norm per PC, so column sums
    to 1).  Weighted across the top-K PCs by ``variance_ratio``::

        contribution[j] = sum_k( loading[j,k]**2 * variance_ratio[k] )

    ``contribution`` sums to ``sum(variance_ratio[:K])`` across all
    features, and answers: "over the top-K PCs, which sites carry the
    most variance?"  Useful as a variance-weighted feature-importance
    score independent of any particular PC direction.

    Parameters
    ----------
    adata
        AnnData that has been through :func:`alphaphos.dimred.pca`.
    n_components
        Number of PCs to weight over (``None`` = all available).

    Returns
    -------
    DataFrame indexed by ``adata.var_names``, sorted by
    ``contribution_pct`` descending.  Columns:

    - ``contribution`` -- raw weighted sum
    - ``contribution_pct`` -- as % of total (i.e. of ``sum(variance_ratio[:K])``)
    """
    if "PCs" not in adata.varm:
        raise KeyError("adata.varm['PCs'] not found. Run alphaphos.dimred.pca(adata) first.")
    if "pca" not in adata.uns or "variance_ratio" not in adata.uns["pca"]:
        raise KeyError(
            "adata.uns['pca']['variance_ratio'] not found. Run alphaphos.dimred.pca(adata) first."
        )
    loadings = np.asarray(adata.varm["PCs"])
    var_ratio = np.asarray(adata.uns["pca"]["variance_ratio"])
    k = loadings.shape[1] if n_components is None else min(n_components, loadings.shape[1])
    contribution = loadings[:, :k] ** 2 @ var_ratio[:k]
    total = float(var_ratio[:k].sum())
    df = pd.DataFrame(
        {
            "contribution": contribution,
            "contribution_pct": 100.0 * contribution / total
            if total > 0
            else np.zeros_like(contribution),
        },
        index=adata.var_names.astype(str),
    )
    return df.sort_values("contribution_pct", ascending=False)
