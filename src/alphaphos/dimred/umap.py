"""UMAP (Uniform Manifold Approximation and Projection) for phospho samples.

Wraps ``umap-learn`` for AnnData -- sample-level manifold embedding into
a 2-D (or higher) visualisation space.  Complements
:func:`alphaphos.dimred.pca` (linear) and
:func:`alphaphos.dimred.tsne` (non-linear, neighbour-preserving,
tends to over-cluster) with a manifold-learning approach that better
preserves both local and global structure.

Output written to the input ``AnnData``:

- ``.obsm["X_umap"]``          -- (n_samples, n_components) sample coords
- ``.uns["umap"]``             -- settings + provenance (n_neighbors, min_dist, metric, ...)

Dependency
----------
Requires the ``umap-learn`` package.  Install with::

    pip install "alphaphos[dimred]"

or::

    pip install umap-learn>=0.5

Reference
---------
McInnes, Healy & Melville (2018), *arXiv:1802.03426* -- the canonical
UMAP paper.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


# Minimum n_samples at which UMAP gives reliable silhouette recovery in
# an empirical sweep across 2- and 3-group synthetic data (10 seeds per
# n, boost = 2-3 std, features = 400).  Below this, the embedding tends
# to collapse (all points cluster together regardless of injected
# structure).  Empirically observed on the EGF+/- benchmark (n=6): all
# samples land within a radius of ~1 in the embedding.  See
# scratchpad/tsne_umap_n_threshold_study.py for the sweep.
MIN_RECOMMENDED_N_UMAP = 20


DEFAULT_UMAP_SETTINGS: dict[str, Any] = {
    "n_components": 2,
    "layer": "intensity_log2",
    # umap-learn defaults; these are also scanpy defaults.
    "n_neighbors": 15,
    "min_dist": 0.1,
    "metric": "euclidean",
    "spread": 1.0,
    "n_epochs": None,  # umap-learn picks 200 or 500 by dataset size
    # Optional PCA pre-reduction (scanpy convention).  When set we take
    # the first ``n_pca_components`` columns of adata.obsm["X_pca"] as
    # the UMAP input.  ap.dimred.pca must have been run first.
    "n_pca_components": None,
    "seed": 42,
}


def resolve_umap_settings(advanced: dict[str, Any] | None) -> dict[str, Any]:
    """Merge overrides over :data:`DEFAULT_UMAP_SETTINGS` with light validation."""
    out = dict(DEFAULT_UMAP_SETTINGS)
    if advanced:
        unknown = set(advanced) - set(out)
        if unknown:
            raise ValueError(
                f"umap: unknown advanced keys: {sorted(unknown)}.  Allowed: {sorted(out)}."
            )
        out.update(advanced)
    return out


def umap(
    adata: ad.AnnData,
    *,
    n_components: int | None = None,
    layer: str | None = None,
    n_neighbors: int | None = None,
    min_dist: float | None = None,
    metric: str | None = None,
    n_pca_components: int | None = None,
    seed: int | None = None,
    advanced: dict[str, Any] | None = None,
    copy: bool = False,
) -> ad.AnnData:
    """UMAP embedding on samples of ``adata``.

    Parameters
    ----------
    adata
        Site-level ``AnnData``.  ``(n_samples, n_features)``.
    n_components, layer, n_neighbors, min_dist, metric, n_pca_components, seed
        Convenience overrides for the corresponding advanced setting.
        Passing these takes precedence over the ``advanced`` dict.
    advanced
        Overrides for :data:`DEFAULT_UMAP_SETTINGS`.
    copy
        If ``True``, mutate a fresh copy of ``adata``; else in-place.
        The (possibly-mutated) AnnData is returned in both cases.

    Notes
    -----
    - UMAP has no native NaN handling.  Impute upstream
      (:func:`alphaphos.impute_hybrid`) or pass ``n_pca_components``
      to feed a NaN-aware PCA embedding (requires
      :func:`alphaphos.dimred.pca` to have been run first).
    - ``n_neighbors`` is auto-clipped to ``< n_samples``.
    - Default seed makes runs deterministic; note that UMAP's stochastic
      optimisation can still show small variation across sklearn
      versions.
    - **Small-n caveat**: UMAP tends to collapse all points into a tiny
      cluster when ``n_samples <= ~10``.  Empirically observed on the
      EGF +/- benchmark (n=6): all samples land within a radius of ~1
      in the embedding regardless of ``n_neighbors`` / ``min_dist``.
      This is a known UMAP behaviour (the manifold-learning objective
      is under-constrained with too few points), not a bug in the
      wrapper.  For very small studies use :func:`alphaphos.dimred.pca`
      or :func:`alphaphos.dimred.tsne` instead.
    """
    try:
        import umap as umap_lib  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for alphaphos.dimred.umap.  Install with "
            "`pip install 'alphaphos[dimred]'` or `pip install umap-learn`."
        ) from exc

    advanced_merged = dict(advanced or {})
    if n_components is not None:
        advanced_merged["n_components"] = n_components
    if layer is not None:
        advanced_merged["layer"] = layer
    if n_neighbors is not None:
        advanced_merged["n_neighbors"] = n_neighbors
    if min_dist is not None:
        advanced_merged["min_dist"] = min_dist
    if metric is not None:
        advanced_merged["metric"] = metric
    if n_pca_components is not None:
        advanced_merged["n_pca_components"] = n_pca_components
    if seed is not None:
        advanced_merged["seed"] = seed
    settings = resolve_umap_settings(advanced_merged)

    adata = adata.copy() if copy else adata

    if settings["n_pca_components"] is not None:
        if "X_pca" not in adata.obsm:
            raise KeyError(
                "n_pca_components requested but adata.obsm['X_pca'] not found. "
                "Run alphaphos.dimred.pca(adata) first, or drop n_pca_components "
                "to run UMAP on the raw layer."
            )
        pca_scores = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
        k = int(settings["n_pca_components"])
        if k > pca_scores.shape[1]:
            raise ValueError(
                f"n_pca_components={k} > available PC scores ({pca_scores.shape[1]}). "
                "Re-run pca with a larger n_components or lower n_pca_components."
            )
        X = pca_scores[:, :k]
        source = f"obsm['X_pca'][:, :{k}]"
    else:
        X = _get_layer_matrix(adata, layer=settings["layer"])
        source = f"layers[{settings['layer']!r}]" if settings["layer"] else "X"

    n_missing = int(np.isnan(X).sum())
    if n_missing > 0:
        raise ValueError(
            f"UMAP input from {source} has {n_missing:,} NaN entries.  Impute "
            "upstream (e.g. ap.impute_hybrid) or pre-reduce with a NaN-aware "
            "PCA (ap.dimred.pca(..., handle_missing='nipals')) and pass "
            "n_pca_components= to feed the PCA scores in."
        )

    # Clip n_neighbors: umap-learn accepts n_neighbors up to n_samples but
    # >= n_samples triggers warnings / edge behaviour.
    neigh = int(settings["n_neighbors"])
    n_samples = int(X.shape[0])
    max_neigh = max(2, n_samples - 1)
    if neigh >= n_samples:
        logger.warning(
            "umap: n_neighbors=%d >= n_samples=%d; clipping to %d.",
            neigh,
            n_samples,
            max_neigh,
        )
        neigh = max_neigh

    # Empirically-derived reliability threshold.  See the constant's
    # docstring.
    if n_samples < MIN_RECOMMENDED_N_UMAP:
        warnings.warn(
            f"UMAP with n_samples={n_samples} < recommended minimum "
            f"{MIN_RECOMMENDED_N_UMAP}.  Empirically the embedding tends "
            "to collapse (silhouette near 0) below this threshold.  "
            "Consider alphaphos.dimred.pca or alphaphos.dimred.tsne for "
            "small studies.  Silence with warnings.filterwarnings("
            "'ignore', category=UserWarning).",
            UserWarning,
            stacklevel=2,
        )

    reducer_kwargs = {
        "n_components": int(settings["n_components"]),
        "n_neighbors": neigh,
        "min_dist": float(settings["min_dist"]),
        "metric": str(settings["metric"]),
        "spread": float(settings["spread"]),
        "random_state": int(settings["seed"]),
    }
    if settings["n_epochs"] is not None:
        reducer_kwargs["n_epochs"] = int(settings["n_epochs"])
    reducer = umap_lib.UMAP(**reducer_kwargs)
    embedding = np.asarray(reducer.fit_transform(X), dtype=np.float64)

    adata.obsm["X_umap"] = embedding
    adata.uns["umap"] = {
        "n_components": int(settings["n_components"]),
        "n_neighbors": int(neigh),
        "min_dist": float(settings["min_dist"]),
        "metric": str(settings["metric"]),
        "spread": float(settings["spread"]),
        "n_epochs": (int(settings["n_epochs"]) if settings["n_epochs"] is not None else None),
        "n_pca_components": (
            int(settings["n_pca_components"]) if settings["n_pca_components"] is not None else None
        ),
        "layer": settings["layer"] if settings["n_pca_components"] is None else None,
        "seed": int(settings["seed"]),
        "n_samples": int(X.shape[0]),
        "n_features_input": int(X.shape[1]),
        "umap_learn_version": getattr(umap_lib, "__version__", None),
    }
    logger.info(
        "umap: %d components on %d samples x %d features (%s), n_neighbors=%d, min_dist=%g",
        settings["n_components"],
        X.shape[0],
        X.shape[1],
        source,
        neigh,
        settings["min_dist"],
    )
    return adata


def _get_layer_matrix(adata: ad.AnnData, *, layer: str | None) -> np.ndarray:
    """Copy semantics identical to ``alphaphos.dimred.pca._get_matrix``."""
    if layer is None:
        return np.asarray(adata.X, dtype=np.float64)
    if layer not in adata.layers:
        raise KeyError(
            f"layer {layer!r} not in adata.layers; available: "
            f"{sorted(adata.layers.keys())}.  Pass layer=None to target adata.X."
        )
    return np.asarray(adata.layers[layer], dtype=np.float64)
