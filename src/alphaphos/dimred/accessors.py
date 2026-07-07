"""Plot-ready DataFrame accessors for PCA / UMAP / t-SNE results.

The core dimred functions attach coordinates to ``.obsm`` and loadings to
``.varm`` following the scverse convention -- numpy arrays without column
labels or sample metadata joined in.  These helpers wrap the arrays as
``pandas.DataFrame`` objects with sensible column names + ``.obs``/``.var``
metadata joined in, ready to hand to matplotlib / seaborn / plotly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


def get_pca_dataframe(
    adata: ad.AnnData,
    *,
    n_components: int | None = None,
    include_obs: bool = True,
) -> pd.DataFrame:
    """One row per sample, columns = PC scores + selected ``.obs`` columns.

    Parameters
    ----------
    adata
        AnnData that has been through :func:`alphaphos.dimred.pca`.
    n_components
        Number of PCs to include.  ``None`` (default) returns all
        available.
    include_obs
        If True (default), all ``.obs`` columns are joined onto the
        DataFrame (condition, batch, replicate, etc.) so you can pass
        the frame directly to a plotting library that uses column names.

    Returns
    -------
    DataFrame indexed by ``adata.obs_names``.  Columns: ``PC1..PCK``
    then all ``.obs`` columns (if included).  ``.attrs["variance_ratio"]``
    and ``.attrs["method"]`` carry the scree data + backend name.
    """
    if "X_pca" not in adata.obsm:
        raise KeyError("adata.obsm['X_pca'] not found. Run alphaphos.dimred.pca(adata) first.")
    scores = np.asarray(adata.obsm["X_pca"])
    k = scores.shape[1] if n_components is None else min(n_components, scores.shape[1])
    df = pd.DataFrame(
        scores[:, :k],
        columns=[f"PC{i + 1}" for i in range(k)],
        index=adata.obs_names.astype(str),
    )
    if include_obs and adata.obs.shape[1] > 0:
        df = df.join(adata.obs)
    # Store metadata as tuples: pandas concat compares .attrs via ==,
    # which on numpy arrays returns element-wise booleans and breaks.
    pca_ns = adata.uns.get("pca", {}) if hasattr(adata, "uns") else {}
    if "variance_ratio" in pca_ns:
        df.attrs["variance_ratio"] = tuple(np.asarray(pca_ns["variance_ratio"])[:k].tolist())
    if "variance" in pca_ns:
        df.attrs["variance"] = tuple(np.asarray(pca_ns["variance"])[:k].tolist())
    if "method" in pca_ns:
        df.attrs["method"] = pca_ns["method"]
    return df


def get_pca_loadings(
    adata: ad.AnnData,
    *,
    n_components: int | None = None,
    top_n_per_pc: int | None = None,
) -> pd.DataFrame:
    """One row per feature, columns = PC loadings.

    Parameters
    ----------
    adata
        AnnData that has been through :func:`alphaphos.dimred.pca`.
    n_components
        Number of PCs to include.  ``None`` (default) returns all.
    top_n_per_pc
        If set, restrict output to the union of features that are in the
        top-N (by absolute loading) for any of the returned PCs.  Useful
        for building "top contributing sites" lists.  ``None`` (default)
        returns loadings for every feature.

    Returns
    -------
    DataFrame indexed by ``adata.var_names``.  Columns: ``PC1..PCK``.
    """
    if "PCs" not in adata.varm:
        raise KeyError("adata.varm['PCs'] not found. Run alphaphos.dimred.pca(adata) first.")
    loadings = np.asarray(adata.varm["PCs"])
    k = loadings.shape[1] if n_components is None else min(n_components, loadings.shape[1])
    df = pd.DataFrame(
        loadings[:, :k],
        columns=[f"PC{i + 1}" for i in range(k)],
        index=adata.var_names.astype(str),
    )
    if top_n_per_pc is not None:
        keep_idx: set[int] = set()
        for col in df.columns:
            top = df[col].abs().nlargest(top_n_per_pc).index
            keep_idx.update(df.index.get_indexer(top).tolist())
        df = df.iloc[sorted(keep_idx)]
    return df
