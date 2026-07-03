"""Imputation for phosphosite quant matrices.

Two functions, both operating on an ``AnnData`` with shape
``(n_samples, n_sites)`` (the scverse convention):

- :func:`impute_knn_site_based` — site-based k-NN imputation. The "right
  direction" of KNN for phospho data with n_samples << n_features:
  finds the k most similar SITES for each missing cell, not the k most
  similar SAMPLES. Matches Dublin's ``impute_phosphosites`` exactly at
  defaults (``n_neighbors=int(sqrt(n_samples))``, ``weights='uniform'``)
  and reproduces R-limma to numerical precision in the EGF benchmark
  (Pearson r = 1.0000 on logFC).

- :func:`impute_hybrid` — per-cell hybrid MAR/MNAR imputation. Combines
  site-based KNN (for MAR cells, where the value is "missed but
  measurable") with Perseus-style downshifted Gaussian (for MNAR cells,
  where the value is below the LOD). The MAR/MNAR split is per-cell,
  driven by the site's overall abundance vs an intensity percentile.

Both functions mutate ``adata`` in place by default (``copy=False``) **and**
return it, so ``ap.impute_hybrid(adata)`` and ``adata = ap.impute_hybrid(adata)``
both work.  Pass ``copy=True`` to receive a fresh copy instead.  Both require a
complete-features matrix — call :func:`alphaphos.filter_by_completeness` first.

The site-based KNN direction is established as correct for phospho 3v3
designs in ``docs/benchmark/_run_imputation_benchmark.py``. The hybrid
is the statistically-defensible default for new analyses (it uses each
imputer where its assumption holds); the pure site-KNN is the legacy-
parity path against R-limma references.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from alphaphos.constants import LAYER_INTENSITY_LOG2

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


def _check_complete_features(X: np.ndarray) -> None:
    """Error if any feature (column) is entirely NaN — KNN can't impute these.

    Caller should run ``filter_data_completeness`` first to drop such columns.
    """
    all_nan_cols = np.isnan(X).all(axis=0)
    if all_nan_cols.any():
        n = int(all_nan_cols.sum())
        raise ValueError(
            f"{n} feature(s) have no observed values; KNN cannot impute them. "
            f"Filter first with alphaphos.filter_by_completeness(adata, "
            f"min_valid_frac=<some positive number>)."
        )


def _site_knn_impute(
    X: np.ndarray,
    n_neighbors: int,
    weights: Literal["uniform", "distance"],
) -> np.ndarray:
    """KNN imputation in the site direction.

    Transposes ``X`` (samples × sites → sites × samples) so KNNImputer
    treats sites as the rows (samples) and samples as the features. For
    each missing cell, the k nearest sites (in the n_samples-dim space)
    contribute their value in that sample. Returns the (samples × sites)
    imputed matrix.
    """
    from sklearn.impute import KNNImputer

    Xt = X.T
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    Xt_imputed = imputer.fit_transform(Xt)
    return Xt_imputed.T


def impute_knn_site_based(
    adata: ad.AnnData,
    *,
    n_neighbors: int | None = None,
    weights: Literal["uniform", "distance"] = "uniform",
    layer: str | None = LAYER_INTENSITY_LOG2,
    copy: bool = False,
) -> ad.AnnData:
    """Site-based KNN imputation (the right direction for phospho 3v3 data).

    Drop-in replacement for Dublin's ``impute_phosphosites``: same algorithm,
    same defaults. Reproduces R-limma on the EGF benchmark at Pearson
    r = 1.0000 on logFC.

    Parameters
    ----------
    adata
        AnnData with shape (n_samples, n_sites). All site columns must
        have at least one observed value (run
        ``apt.pp.filter_data_completeness(action='drop')`` first).
    n_neighbors
        Number of neighbor sites to use. Default ``int(sqrt(n_samples))``
        — Dublin's convention.
    weights
        ``"uniform"`` (default, Dublin) or ``"distance"``.
    layer
        Which layer to impute. Default ``"intensity_log2"`` -- the canonical
        log2 slot produced by ``collapse_sites``, and the slot every
        downstream step (``diff_exp_limma``, viz, QC) reads by default.
        Pass ``None`` to target ``adata.X`` instead (they start equal after
        ``collapse_sites`` but diverge once any layer is mutated).
    copy
        If ``True``, mutate a fresh copy of ``adata`` and return it.
        If ``False`` (default), mutate ``adata`` in place.  Either way
        the (possibly-mutated) AnnData is returned so the call is safe
        with or without assignment::

            ap.impute_knn_site_based(adata)          # in-place, return ignored
            adata = ap.impute_knn_site_based(adata)  # equivalent, same object
            new_ad = ap.impute_knn_site_based(adata, copy=True)   # fresh copy

    Returns
    -------
    AnnData
        The mutated AnnData (same object when ``copy=False``, fresh copy
        when ``copy=True``).
    """
    adata = adata.copy() if copy else adata
    X = adata.X if layer is None else adata.layers[layer]
    X = np.asarray(X, dtype=float)

    _check_complete_features(X)

    n_samples = X.shape[0]
    k = n_neighbors if n_neighbors is not None else max(1, int(np.sqrt(n_samples)))

    imputed = _site_knn_impute(X, n_neighbors=k, weights=weights)

    if layer is None:
        adata.X = imputed
    else:
        adata.layers[layer] = imputed
    return adata


def impute_hybrid(
    adata: ad.AnnData,
    *,
    mnar_threshold_percentile: float = 30.0,
    knn_n_neighbors: int | None = None,
    knn_weights: Literal["uniform", "distance"] = "uniform",
    gaussian_std_offset: float = 1.8,
    gaussian_std_factor: float = 0.3,
    gaussian_seed: int = 42,
    layer: str | None = LAYER_INTENSITY_LOG2,
    return_audit: bool = False,
    copy: bool = False,
) -> ad.AnnData | tuple[ad.AnnData, pd.DataFrame]:
    """Per-cell hybrid MAR/MNAR imputation.

    Each missing cell ``(site, sample)`` is classified independently:

    - **MNAR** (Missing Not At Random): the site has no observed values
      anywhere, OR the mean of its observed values is below the
      ``mnar_threshold_percentile`` of the dataset's overall intensity
      distribution. The cell is imputed by drawing from a downshifted
      Gaussian ``N(μ_sample − offset·σ_sample, factor·σ_sample)`` — the
      Perseus convention.
    - **MAR** (Missing At Random): otherwise. The cell is imputed via
      site-based KNN (see :func:`impute_knn_site_based`).

    The split lets each imputer operate where its statistical assumption
    holds: KNN borrows from neighbors when the site IS detectable
    (technical miss), Gaussian models the below-LOD tail when the site is
    below the noise floor.

    Parameters
    ----------
    adata
        AnnData with shape (n_samples, n_sites).
    mnar_threshold_percentile
        Percentile (0–100) of the dataset's observed-cell distribution
        below which a site is treated as low-abundance / MNAR. Default
        30 — sites whose mean observed intensity is in the bottom 30% of
        the dataset get Gaussian-imputed for their missing cells.
    knn_n_neighbors
        k for the MAR-path KNN. Default ``int(sqrt(n_samples))``.
    knn_weights
        ``"uniform"`` (default) or ``"distance"``.
    gaussian_std_offset, gaussian_std_factor
        Perseus parameters: imputed value ~ ``N(μ − offset·σ, factor·σ)``.
        Defaults 1.8 and 0.3 match Perseus and ``apt.pp.impute_gaussian``.
    gaussian_seed
        RNG seed for reproducible Gaussian draws.
    layer
        Which layer to impute. Default ``"intensity_log2"`` -- the canonical
        log2 slot produced by ``collapse_sites``, and the slot every
        downstream step (``diff_exp_limma``, viz, QC) reads by default.
        Pass ``None`` to target ``adata.X`` instead (they start equal after
        ``collapse_sites`` but diverge once any layer is mutated).
    return_audit
        If True, also return a per-cell DataFrame logging which strategy
        was applied (only for missing cells).
    copy
        If ``True``, mutate a fresh copy of ``adata`` and return it.
        If ``False`` (default), mutate ``adata`` in place.  Either way
        the (possibly-mutated) AnnData is returned so the call is safe
        with or without assignment::

            ap.impute_hybrid(adata)          # in-place, return ignored
            adata = ap.impute_hybrid(adata)  # equivalent, same object
            new_ad = ap.impute_hybrid(adata, copy=True)   # fresh copy

    Returns
    -------
    AnnData or (AnnData, audit DataFrame)
        The mutated AnnData (same object when ``copy=False``, fresh copy
        when ``copy=True``).  When ``return_audit=True``, a
        ``(adata, audit)`` tuple.
    """
    adata = adata.copy() if copy else adata
    X_orig = adata.X if layer is None else adata.layers[layer]
    X = np.asarray(X_orig, dtype=float).copy()  # we mutate

    _check_complete_features(X)

    n_samples = X.shape[0]
    missing_mask = np.isnan(X)
    if not missing_mask.any():
        # Nothing to do
        if layer is None:
            adata.X = X
        else:
            adata.layers[layer] = X
        if return_audit:
            return adata, pd.DataFrame(columns=["sample_idx", "site_idx", "strategy"])
        return adata

    # ---- 1. Determine MAR vs MNAR per site -------------------------------
    observed_all = X[~missing_mask]
    threshold = float(np.percentile(observed_all, mnar_threshold_percentile))
    with np.errstate(all="ignore"):  # silence the all-NaN warnings
        site_means = np.nanmean(X, axis=0)
    site_is_low = np.isnan(site_means) | (site_means < threshold)

    mnar_mask = missing_mask & site_is_low[np.newaxis, :]
    mar_mask = missing_mask & ~site_is_low[np.newaxis, :]

    # ---- 2. Pre-compute per-sample Gaussian draws for MNAR cells ---------
    rng = np.random.default_rng(gaussian_seed)
    gauss_filled = np.zeros_like(X)
    for j in range(n_samples):
        sample_observed = X[j, ~np.isnan(X[j, :])]
        if len(sample_observed) < 2:
            # Can't compute stdev from <2 points; leave NaN, warn later
            continue
        mu = float(sample_observed.mean())
        sigma = float(sample_observed.std(ddof=1))
        downshift_mu = mu - gaussian_std_offset * sigma
        width = gaussian_std_factor * sigma
        n_cells = int(mnar_mask[j, :].sum())
        if n_cells > 0:
            gauss_filled[j, mnar_mask[j, :]] = rng.normal(downshift_mu, width, n_cells)

    # ---- 3. Site-based KNN for MAR cells ---------------------------------
    if mar_mask.any():
        k = knn_n_neighbors if knn_n_neighbors is not None else max(1, int(np.sqrt(n_samples)))
        knn_filled = _site_knn_impute(X, n_neighbors=k, weights=knn_weights)
    else:
        knn_filled = np.zeros_like(X)

    # ---- 4. Assemble ------------------------------------------------------
    out = X.copy()
    out[mar_mask] = knn_filled[mar_mask]
    out[mnar_mask] = gauss_filled[mnar_mask]

    # Sanity check: any cell still NaN means the per-sample Gaussian failed
    # (sample had <2 observed cells); fall back to global Gaussian for those.
    remaining_nan = np.isnan(out)
    if remaining_nan.any():
        global_mu = float(observed_all.mean())
        global_sigma = float(observed_all.std(ddof=1))
        fallback_mu = global_mu - gaussian_std_offset * global_sigma
        fallback_width = gaussian_std_factor * global_sigma
        n_left = int(remaining_nan.sum())
        out[remaining_nan] = rng.normal(fallback_mu, fallback_width, n_left)
        logger.info(
            "impute_hybrid: %d cells fell through per-sample Gaussian "
            "(samples with <2 observed values); filled from global Gaussian.",
            n_left,
        )

    n_mar = int(mar_mask.sum())
    n_mnar = int(mnar_mask.sum())
    logger.info(
        "impute_hybrid: %d cells imputed (%d MAR via site-KNN, %d MNAR via Gaussian) "
        "[threshold = %s percentile, intensity %.3f]",
        n_mar + n_mnar,
        n_mar,
        n_mnar,
        mnar_threshold_percentile,
        threshold,
    )

    if layer is None:
        adata.X = out
    else:
        adata.layers[layer] = out

    if return_audit:
        rows = []
        if mar_mask.any():
            i_idx, j_idx = np.where(mar_mask)
            rows.extend(
                [
                    {"sample_idx": int(i), "site_idx": int(j), "strategy": "MAR_KNN"}
                    for i, j in zip(i_idx, j_idx, strict=True)
                ]
            )
        if mnar_mask.any():
            i_idx, j_idx = np.where(mnar_mask)
            rows.extend(
                [
                    {"sample_idx": int(i), "site_idx": int(j), "strategy": "MNAR_Gaussian"}
                    for i, j in zip(i_idx, j_idx, strict=True)
                ]
            )
        audit = pd.DataFrame(rows)
        return adata, audit

    return adata
