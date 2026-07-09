"""Dimensionality reduction for phosphosite AnnData.

Answers three exploratory questions on a phospho-experiment ``AnnData``:

1. **Sample-space clustering** -- do samples group by condition (biology)
   or by batch (artifact)?  PCA / distance heatmaps.
2. **Effective dimensionality** -- how many components explain the
   variance?  Scree plots.
3. **Imputation impact** -- does the imputation step distort the sample-
   space structure vs a missing-value-aware PCA on the raw matrix?

Missing values
--------------
Standard PCA requires a complete matrix.  For phospho data, the natural
workflow is ``filter_by_completeness -> impute_hybrid -> pca``.  Two
alternative methods handle missing values directly, without imputation:

- **NIPALS** (Wold 1966) -- iterative regression that skips NaN entries
  in the sum computations.  Standard in chemometrics/metabolomics
  (MetaboAnalyst, mixOmics).  Fast, well-understood.
- **PPCA** (Tipping & Bishop 1999) -- probabilistic PCA with EM
  handling of missing values.  Slower but principled.

Compare imputation-vs-raw PCA with
:func:`compare_imputation_impact` to detect imputation-induced
distortions of the sample-space structure before publishing figures.

Public API
----------
:func:`pca`                          -- standard / NIPALS / PPCA PCA.
:func:`tsne`                         -- Barnes-Hut t-SNE (van der Maaten
                                        & Hinton 2008) via sklearn.
:func:`umap`                         -- UMAP (McInnes et al. 2018) via
                                        ``umap-learn`` (``[dimred]`` extra).
:func:`get_pca_dataframe`,
:func:`get_tsne_dataframe`,
:func:`get_umap_dataframe`           -- plot-ready sample-level frames.
:func:`get_pca_loadings`             -- feature-level loadings as DataFrame.
:func:`loadings_for_enrichment`      -- canonicalized loadings ready for
                                        :mod:`alphaphos.enrichment` (gsea /
                                        ksea / ora) with no wrappers.
:func:`feature_variance_contribution` -- per-feature % of top-K PC variance.
:func:`compare_imputation_impact`    -- imputed-vs-raw PC diagnostic.
:func:`sample_distance`              -- pairwise sample distances.
:func:`hierarchical_cluster`         -- linkage + dendrogram data.

t-SNE / UMAP note
-----------------
Neither method has native NaN handling.  Impute upstream
(:func:`alphaphos.impute_hybrid`) OR pre-reduce with a NaN-aware
:func:`pca` (``handle_missing="nipals"`` / ``"ppca"``) and pass
``n_pca_components=`` to the manifold call -- both ``tsne`` and
``umap`` will read ``adata.obsm["X_pca"]`` when that argument is set.

**Small-n caveat**: t-SNE and UMAP are non-linear methods that need
enough samples to define local neighbourhoods reliably.  Empirical
sweeps on synthetic 2/3-group data (see
``scratchpad/tsne_umap_n_threshold_study.py``) show:

- ``pca`` is reliable at ``n_samples >= 6``.
- ``umap`` becomes reliable at
  ``n_samples >= MIN_RECOMMENDED_N_UMAP`` (=20).
- ``tsne`` becomes reliable at
  ``n_samples >= MIN_RECOMMENDED_N_TSNE`` (=30).

Below the recommended minimums, ``tsne`` and ``umap`` emit
``UserWarning`` -- the embedding is still returned so the caller can
inspect it, but silhouette recovery is variable across seeds and
often worse than PCA.  Silence via
``warnings.filterwarnings("ignore", category=UserWarning)`` when you
know what you're doing.
"""

from alphaphos.dimred.accessors import (
    get_pca_dataframe,
    get_pca_loadings,
    get_tsne_dataframe,
    get_umap_dataframe,
)
from alphaphos.dimred.comparison import compare_imputation_impact
from alphaphos.dimred.distance import hierarchical_cluster, sample_distance
from alphaphos.dimred.loadings import (
    feature_variance_contribution,
    loadings_for_enrichment,
)
from alphaphos.dimred.pca import DEFAULT_PCA_SETTINGS, pca, resolve_pca_settings
from alphaphos.dimred.tsne import (
    DEFAULT_TSNE_SETTINGS,
    MIN_RECOMMENDED_N_TSNE,
    resolve_tsne_settings,
    tsne,
)
from alphaphos.dimred.umap import (
    DEFAULT_UMAP_SETTINGS,
    MIN_RECOMMENDED_N_UMAP,
    resolve_umap_settings,
    umap,
)

__all__ = [
    # Core dimensionality reductions
    "pca",
    "tsne",
    "umap",
    # Settings + advanced-key resolvers
    "DEFAULT_PCA_SETTINGS",
    "DEFAULT_TSNE_SETTINGS",
    "DEFAULT_UMAP_SETTINGS",
    "MIN_RECOMMENDED_N_TSNE",
    "MIN_RECOMMENDED_N_UMAP",
    "resolve_pca_settings",
    "resolve_tsne_settings",
    "resolve_umap_settings",
    # Plot-ready DataFrame accessors
    "get_pca_dataframe",
    "get_pca_loadings",
    "get_tsne_dataframe",
    "get_umap_dataframe",
    # PC loadings -> enrichment bridges
    "loadings_for_enrichment",
    "feature_variance_contribution",
    # Diagnostics + distances
    "compare_imputation_impact",
    "sample_distance",
    "hierarchical_cluster",
]
