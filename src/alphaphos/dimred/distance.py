"""Sample-sample distance + hierarchical clustering.

Utility functions for the sample-space QC branch of the pipeline.  Both
functions are NaN-safe (missing values are handled per pair).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


def sample_distance(
    adata: ad.AnnData,
    *,
    layer: str = "intensity_log2",
    metric: Literal["euclidean", "correlation", "cosine"] = "euclidean",
) -> pd.DataFrame:
    """Pairwise sample-sample distance matrix.

    NaN-safe: for each pair, only the features where both samples have
    non-missing values contribute to the distance.

    Parameters
    ----------
    adata
        ``(n_samples, n_features)`` AnnData.
    layer
        Which layer to compute on.  Default ``"intensity_log2"``.
    metric
        - ``"euclidean"`` -- L2 distance.  Scales with n_features so pair
          coverage matters; we normalise by ``sqrt(n_common)`` so pairs
          with fewer shared observations aren't penalised.
        - ``"correlation"`` -- ``1 - pearson_r`` on the common features.
        - ``"cosine"`` -- ``1 - cosine_similarity`` on the common features.

    Returns
    -------
    DataFrame indexed and columned by ``adata.obs_names``, symmetric,
    diagonal = 0.
    """
    X = _get_matrix(adata, layer=layer)
    n = X.shape[0]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            mask = ~(np.isnan(X[i]) | np.isnan(X[j]))
            k = int(mask.sum())
            if k == 0:
                dist[i, j] = dist[j, i] = np.nan
                continue
            a = X[i, mask]
            b = X[j, mask]
            if metric == "euclidean":
                d = float(np.linalg.norm(a - b)) / max(np.sqrt(k), 1.0)
            elif metric == "correlation":
                a_c = a - a.mean()
                b_c = b - b.mean()
                denom = float(np.linalg.norm(a_c) * np.linalg.norm(b_c))
                d = 1.0 - float(a_c @ b_c) / denom if denom > 0 else 1.0
            elif metric == "cosine":
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                d = 1.0 - float(a @ b) / denom if denom > 0 else 1.0
            else:
                raise ValueError(f"Unknown metric: {metric!r}")
            dist[i, j] = dist[j, i] = d
    labels = list(adata.obs_names.astype(str))
    return pd.DataFrame(dist, index=labels, columns=labels)


def hierarchical_cluster(
    distance_df: pd.DataFrame,
    *,
    method: Literal["average", "complete", "single", "ward"] = "average",
) -> dict[str, object]:
    """Hierarchical clustering from a distance matrix.

    Returns a dict with ``"linkage"`` (scipy linkage matrix), ``"labels"``
    (sample names in matrix order), and ``"leaf_order"`` (index order
    from the dendrogram, for reordering heatmaps).
    """
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    labels = list(distance_df.index.astype(str))
    matrix = distance_df.to_numpy(dtype=np.float64)
    # Convert to condensed form; NaN in the matrix is not tolerated by scipy
    # so we replace with the max finite entry (worst-case distance).
    finite_mask = np.isfinite(matrix)
    if not finite_mask.all():
        max_finite = float(np.nanmax(matrix[finite_mask])) if finite_mask.any() else 1.0
        matrix = np.where(finite_mask, matrix, max_finite)
    condensed = squareform(matrix, checks=False)
    link = linkage(condensed, method=method)
    return {
        "linkage": link,
        "labels": labels,
        "leaf_order": leaves_list(link).tolist(),
    }


def _get_matrix(adata: ad.AnnData, *, layer: str) -> np.ndarray:
    if layer in adata.layers:
        X = adata.layers[layer]
    elif layer == "intensity_log2":
        X = adata.X
    else:
        raise KeyError(f"Layer {layer!r} not in adata.layers.")
    return np.asarray(X, dtype=np.float64)
