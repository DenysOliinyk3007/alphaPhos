"""t-SNE (t-distributed Stochastic Neighbor Embedding) for phospho samples.

Wraps :class:`sklearn.manifold.TSNE` for AnnData -- sample-level embedding
into a 2-D (or higher) visualisation space.  Complements
:func:`alphaphos.dimred.pca` (linear, variance-preserving) with a
neighbour-preserving non-linear projection that separates
condition clusters better on datasets where variance is not the
discriminating axis.

Output written to the input ``AnnData``:

- ``.obsm["X_tsne"]``          -- (n_samples, n_components) sample coords
- ``.uns["tsne"]["perplexity"]``, ``["seed"]``, ``["method"]``, ``["n_pca_components"]``

Missing values
--------------
t-SNE has no native NaN handling.  If the input layer has NaN, this
function raises a clear error and points at
:func:`alphaphos.impute_hybrid`.  When ``use_pca=True`` is set the PCA
step handles NaN via NIPALS / PPCA if the user requested that method
in the upstream ``ap.dimred.pca`` call.

Reference
---------
van der Maaten & Hinton (2008), *Journal of Machine Learning Research*
9: 2579-2605 -- the canonical t-SNE paper.  Barnes-Hut approximation
(Barnes & Hut 1986; van der Maaten 2014) is the sklearn default and
what we use.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import anndata as ad

try:
    from sklearn.manifold import TSNE as _SklearnTSNE

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - sklearn is a core dep
    _HAS_SKLEARN = False

logger = logging.getLogger(__name__)


# Minimum n_samples at which t-SNE gives reliable silhouette recovery in
# an empirical sweep across 2- and 3-group synthetic data (10 seeds per
# n, boost = 2-3 std, features = 400).  Below this, the mean silhouette
# is noisy across seeds -- some runs recover the structure, others
# don't.  See scratchpad/tsne_umap_n_threshold_study.py for the sweep.
MIN_RECOMMENDED_N_TSNE = 30


DEFAULT_TSNE_SETTINGS: dict[str, Any] = {
    "n_components": 2,
    "layer": "intensity_log2",
    # sklearn's TSNE default.  For n_samples < 50, drop this manually --
    # perplexity must be < n_samples.  We clip below.
    "perplexity": 30.0,
    "learning_rate": "auto",
    # Barnes-Hut is the default sklearn method and the only one we use.
    # "exact" is O(n^2) and rarely worth it for phospho scales.
    "method": "barnes_hut",
    "init": "pca",
    "max_iter": 1000,
    # Optional PCA pre-reduction (scanpy convention).  When set, we take
    # the first ``n_pca_components`` columns of adata.obsm["X_pca"] as
    # the t-SNE input.  ap.dimred.pca must have been run first.
    "n_pca_components": None,
    "seed": 42,
}


def resolve_tsne_settings(advanced: dict[str, Any] | None) -> dict[str, Any]:
    """Merge overrides over :data:`DEFAULT_TSNE_SETTINGS` with light validation."""
    out = dict(DEFAULT_TSNE_SETTINGS)
    if advanced:
        unknown = set(advanced) - set(out)
        if unknown:
            raise ValueError(
                f"tsne: unknown advanced keys: {sorted(unknown)}.  Allowed: {sorted(out)}."
            )
        out.update(advanced)
    return out


def tsne(
    adata: ad.AnnData,
    *,
    n_components: int | None = None,
    layer: str | None = None,
    perplexity: float | None = None,
    n_pca_components: int | None = None,
    seed: int | None = None,
    advanced: dict[str, Any] | None = None,
    copy: bool = False,
) -> ad.AnnData:
    """Barnes-Hut t-SNE on samples of ``adata``.

    Parameters
    ----------
    adata
        Site-level ``AnnData``.  ``(n_samples, n_features)``.
    n_components, layer, perplexity, n_pca_components, seed
        Convenience overrides for the corresponding advanced setting.
        Passing these takes precedence over the ``advanced`` dict.
    advanced
        Overrides for :data:`DEFAULT_TSNE_SETTINGS`.
    copy
        If ``True``, mutate a fresh copy of ``adata``; else in-place.
        The (possibly-mutated) AnnData is returned in both cases.

    Notes
    -----
    - t-SNE has no native NaN handling.  Impute upstream
      (:func:`alphaphos.impute_hybrid`) or pass ``n_pca_components``
      (which reads ``adata.obsm["X_pca"]`` -- run
      :func:`alphaphos.dimred.pca` first with a NaN-aware backend).
    - ``perplexity`` is auto-clipped to ``< n_samples`` (sklearn hard
      constraint).  A warning is logged when clipping fires.
    - ``learning_rate="auto"`` follows the sklearn 1.2+ recommendation
      (``max(N/12, 50)``).
    """
    if not _HAS_SKLEARN:  # pragma: no cover
        raise ImportError("scikit-learn is required for alphaphos.dimred.tsne.")

    advanced_merged = dict(advanced or {})
    if n_components is not None:
        advanced_merged["n_components"] = n_components
    if layer is not None:
        advanced_merged["layer"] = layer
    if perplexity is not None:
        advanced_merged["perplexity"] = perplexity
    if n_pca_components is not None:
        advanced_merged["n_pca_components"] = n_pca_components
    if seed is not None:
        advanced_merged["seed"] = seed
    settings = resolve_tsne_settings(advanced_merged)

    adata = adata.copy() if copy else adata

    # Input matrix -- either the PCA scores or the raw layer.
    if settings["n_pca_components"] is not None:
        if "X_pca" not in adata.obsm:
            raise KeyError(
                "n_pca_components requested but adata.obsm['X_pca'] not found. "
                "Run alphaphos.dimred.pca(adata) first, or drop n_pca_components "
                "to run t-SNE on the raw layer."
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
            f"t-SNE input from {source} has {n_missing:,} NaN entries.  Impute "
            "upstream (e.g. ap.impute_hybrid) or pre-reduce with a NaN-aware "
            "PCA (ap.dimred.pca(..., handle_missing='nipals')) and pass "
            "n_pca_components= to feed the PCA scores in."
        )

    # Clip perplexity to sklearn's hard constraint: perplexity < n_samples.
    perp = float(settings["perplexity"])
    n_samples = int(X.shape[0])
    max_perp = max(1.0, n_samples - 1.0)
    if perp >= n_samples:
        logger.warning(
            "tsne: perplexity=%g >= n_samples=%d; clipping to %g "
            "(sklearn requires perplexity < n_samples).",
            perp,
            n_samples,
            max_perp,
        )
        perp = max_perp

    # Empirically-derived reliability threshold.  See the constant's
    # docstring.  We emit a UserWarning (not an error): the caller may
    # know what they're doing (e.g. exploratory viz where any embedding
    # is acceptable), but the default assumption is that they'd rather
    # know.
    if n_samples < MIN_RECOMMENDED_N_TSNE:
        warnings.warn(
            f"t-SNE with n_samples={n_samples} < recommended minimum "
            f"{MIN_RECOMMENDED_N_TSNE}.  Silhouette recovery on synthetic "
            "benchmarks is highly variable across seeds in this regime.  "
            "Consider alphaphos.dimred.pca for small studies, or expect "
            "seed-dependent output.  Silence with warnings.filterwarnings("
            "'ignore', category=UserWarning).",
            UserWarning,
            stacklevel=2,
        )

    # sklearn 1.5 renamed n_iter -> max_iter (n_iter deprecated, removed in 1.7).
    # Prefer max_iter when available; fall back to n_iter on older sklearn.
    tsne_kwargs: dict[str, Any] = {
        "n_components": int(settings["n_components"]),
        "perplexity": perp,
        "learning_rate": settings["learning_rate"],
        "method": settings["method"],
        "init": settings["init"],
        "random_state": int(settings["seed"]),
    }
    import inspect

    if "max_iter" in inspect.signature(_SklearnTSNE).parameters:
        tsne_kwargs["max_iter"] = int(settings["max_iter"])
    else:  # pragma: no cover  -- old sklearn (<1.5)
        tsne_kwargs["n_iter"] = int(settings["max_iter"])
    estimator = _SklearnTSNE(**tsne_kwargs)
    embedding = np.asarray(estimator.fit_transform(X), dtype=np.float64)

    adata.obsm["X_tsne"] = embedding
    adata.uns["tsne"] = {
        "method": "barnes_hut",
        "n_components": int(settings["n_components"]),
        "perplexity": float(perp),
        "learning_rate": str(settings["learning_rate"]),
        "init": str(settings["init"]),
        "max_iter": int(settings["max_iter"]),
        "n_pca_components": (
            int(settings["n_pca_components"]) if settings["n_pca_components"] is not None else None
        ),
        "layer": settings["layer"] if settings["n_pca_components"] is None else None,
        "seed": int(settings["seed"]),
        "n_samples": int(X.shape[0]),
        "n_features_input": int(X.shape[1]),
    }
    logger.info(
        "tsne: %d components on %d samples x %d features (%s), perplexity=%.2f",
        settings["n_components"],
        X.shape[0],
        X.shape[1],
        source,
        perp,
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
