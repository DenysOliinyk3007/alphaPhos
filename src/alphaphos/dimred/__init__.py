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
:func:`get_pca_dataframe`            -- plot-ready sample-level PCs + ``.obs``.
:func:`get_pca_loadings`             -- feature-level loadings as DataFrame.
:func:`loadings_for_enrichment`      -- canonicalized loadings ready for
                                        :mod:`alphaphos.enrichment` (gsea /
                                        ksea / ora) with no wrappers.
:func:`feature_variance_contribution` -- per-feature % of top-K PC variance.
:func:`compare_imputation_impact`    -- imputed-vs-raw PC diagnostic.
:func:`sample_distance`              -- pairwise sample distances.
:func:`hierarchical_cluster`         -- linkage + dendrogram data.
"""

from alphaphos.dimred.accessors import get_pca_dataframe, get_pca_loadings
from alphaphos.dimred.comparison import compare_imputation_impact
from alphaphos.dimred.distance import hierarchical_cluster, sample_distance
from alphaphos.dimred.loadings import (
    feature_variance_contribution,
    loadings_for_enrichment,
)
from alphaphos.dimred.pca import DEFAULT_PCA_SETTINGS, pca, resolve_pca_settings

__all__ = [
    "pca",
    "DEFAULT_PCA_SETTINGS",
    "resolve_pca_settings",
    "get_pca_dataframe",
    "get_pca_loadings",
    "loadings_for_enrichment",
    "feature_variance_contribution",
    "compare_imputation_impact",
    "sample_distance",
    "hierarchical_cluster",
]
