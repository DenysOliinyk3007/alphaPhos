"""ComBat batch-effect correction (empirical-Bayes) via inmoose.pycombat.

Sits between imputation and differential testing: reads the canonical
log2 layer (must be complete -- no NaN, no all-NaN sites), removes known
batch effects, writes the corrected values back to the same layer. The
pre-correction values are copied to ``layers["intensity_log2_precombat"]``
by default so ``diff_exp_limma`` can test the raw data with a batch
covariate (the statistically preferred path).

The routine stamps ``adata.uns["alphaphos"]["batch_correction"]`` with the
method + parameters used. ``diff_exp_limma`` reads that stamp and refuses
to run when the same batch column is also passed as a covariate -- see
:func:`alphaphos.stats.diff_exp.diff_exp_limma`.

Community convention (Nekrutenko/Smyth, and how most limma tutorials
describe it):

* ComBat-correct for **visualisation** (PCA/UMAP/clustering) so batch
  differences don't dominate the projection.
* Run statistics on the **raw** data with batch as a limma covariate;
  the linear model handles the adjustment more cleanly than acting on
  the same numbers twice.

Both branches are available here because both layers are kept.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from alphaphos.constants import (
    LAYER_INTENSITY_LOG2,
    LAYER_INTENSITY_LOG2_PRECOMBAT,
    LOG_SCALE_MEDIAN_CEILING,
    MIN_REPLICATES_WARNING,
    UNS_ALPHAPHOS,
    UNS_BATCH_CORRECTION,
)

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


try:
    import inmoose  # type: ignore[import-not-found]
    import patsy  # type: ignore[import-not-found]
    from inmoose.pycombat import pycombat_norm  # type: ignore[import-not-found]

    _HAS_COMBAT_DEPS = True
    _INMOOSE_VERSION = inmoose.__version__
except ImportError:  # pragma: no cover
    _HAS_COMBAT_DEPS = False
    _INMOOSE_VERSION = None


DEFAULT_COMBAT_SETTINGS: dict = {
    "par_prior": True,
    "mean_only": False,
    "ref_batch": None,
}


def batch_correct_combat(
    adata: ad.AnnData,
    *,
    batch_column: str,
    covariates: list[str] | None = None,
    layer: str | None = LAYER_INTENSITY_LOG2,
    keep_precombat: bool = True,
    advanced: dict | None = None,
    copy: bool = False,
) -> ad.AnnData | None:
    """Empirical-Bayes batch correction via ComBat.

    Parameters
    ----------
    adata : AnnData
        Shape ``(n_samples, n_sites)``. The target layer must be
        log-scale and NaN-free.
    batch_column : str
        Column in ``adata.obs`` giving the batch label per sample.
        Categorical; at least 2 levels, at least 2 samples per level.
    covariates : list[str], optional
        Columns in ``adata.obs`` whose variance should be PRESERVED
        during correction (typically the biology-of-interest, e.g.
        ``["condition"]``). Categorical only; continuous covariates are
        not supported by pycombat. The batch column itself may not be
        listed here.
    layer : str, optional
        Which ``adata.layers`` slot to correct. Default
        ``"intensity_log2"``. Pass ``None`` to target ``adata.X``.
    keep_precombat : bool, default True
        If True, copy the pre-correction values to
        ``layers["intensity_log2_precombat"]`` (regardless of ``layer=``)
        so downstream ``diff_exp_limma`` can run on the raw data with a
        batch covariate. Cost: ~one extra layer's memory footprint.
    advanced : dict, optional
        Overrides for :data:`DEFAULT_COMBAT_SETTINGS`:

        - ``par_prior`` (bool, default True) -- parametric vs
          non-parametric EB estimation of batch effect priors.
        - ``mean_only`` (bool, default False) -- adjust only the batch
          means, not variances.
        - ``ref_batch`` (str or None) -- if set, use this batch as
          reference and adjust the others to match it.
    copy : bool, default False
        If True, return a corrected copy; otherwise mutate in-place
        and return ``None``.

    Returns
    -------
    None or AnnData
        AnnData (if ``copy=True``) with the target layer corrected,
        ``layers["intensity_log2_precombat"]`` populated (if
        ``keep_precombat=True``), and ``.uns["alphaphos"]["batch_correction"]``
        stamped with the provenance dict.

    Raises
    ------
    ImportError
        If ``inmoose`` / ``patsy`` are not installed. Install via
        ``pip install alphaPhos[stats]``.
    ValueError
        For validation failures: <2 batches, batch with <2 samples,
        NaN in the target layer, non-log-scale data, duplicate
        ``var_names``, unknown ``advanced`` keys, ``ref_batch`` not a
        real batch, covariate equal to batch column.
    KeyError
        For missing ``batch_column`` / ``covariates`` / ``layer``.
    """
    if not _HAS_COMBAT_DEPS:
        raise ImportError(
            "batch_correct_combat requires inmoose and patsy. Install with "
            "`pip install alphaPhos[stats]`."
        )

    settings = _resolve_combat_settings(advanced)
    _validate_inputs(
        adata,
        batch_column=batch_column,
        covariates=covariates,
        layer=layer,
        ref_batch=settings["ref_batch"],
    )

    adata = adata.copy() if copy else adata
    batches = adata.obs[batch_column].astype(str).to_numpy()
    unique_batches, batch_sizes = np.unique(batches, return_counts=True)
    _warn_on_small_batches(unique_batches, batch_sizes)
    if covariates:
        _warn_on_confounding(adata, batch_column=batch_column, covariates=covariates)

    source_matrix = _get_matrix(adata, layer=layer)
    _validate_no_nan(source_matrix, layer=layer)
    _validate_log_scale(source_matrix, layer=layer)

    covar_mod = _build_covar_mod(adata, covariates=covariates)

    # pycombat wants features x samples; AnnData is samples x features.
    counts = source_matrix.T
    corrected = pycombat_norm(
        counts=counts,
        batch=list(batches),
        covar_mod=covar_mod,
        par_prior=settings["par_prior"],
        mean_only=settings["mean_only"],
        ref_batch=settings["ref_batch"],
    )
    corrected = np.asarray(corrected).T  # back to samples x features

    if keep_precombat:
        adata.layers[LAYER_INTENSITY_LOG2_PRECOMBAT] = source_matrix.copy()

    if layer is None:
        adata.X = corrected
    else:
        adata.layers[layer] = corrected

    _stamp_provenance(
        adata,
        batch_column=batch_column,
        covariates=covariates,
        layer=layer,
        keep_precombat=keep_precombat,
        settings=settings,
        unique_batches=unique_batches,
        batch_sizes=batch_sizes,
    )

    logger.info(
        "batch_correct_combat: layer=%s | n_batches=%d | batch_sizes=%s | covariates=%s%s",
        layer,
        len(unique_batches),
        dict(zip(unique_batches.tolist(), batch_sizes.tolist(), strict=True)),
        covariates or [],
        f" | ref_batch={settings['ref_batch']!r}" if settings["ref_batch"] else "",
    )

    return adata if copy else None


def _resolve_combat_settings(advanced: dict | None) -> dict:
    out = dict(DEFAULT_COMBAT_SETTINGS)
    if not advanced:
        return out
    unknown = set(advanced) - set(DEFAULT_COMBAT_SETTINGS)
    if unknown:
        raise ValueError(
            f"Unknown advanced keys: {sorted(unknown)}. Allowed: {sorted(DEFAULT_COMBAT_SETTINGS)}"
        )
    out.update(advanced)
    return out


def _validate_inputs(
    adata: ad.AnnData,
    *,
    batch_column: str,
    covariates: list[str] | None,
    layer: str | None,
    ref_batch,
) -> None:
    if not adata.var_names.is_unique:
        raise ValueError(
            "adata.var_names must be unique; duplicates would break the "
            "features -> samples mapping ComBat expects."
        )
    if batch_column not in adata.obs.columns:
        raise KeyError(
            f"batch_column={batch_column!r} not in adata.obs. Available: {list(adata.obs.columns)}"
        )
    batches = adata.obs[batch_column].astype(str)
    unique_batches = batches.unique()
    if len(unique_batches) < 2:
        raise ValueError(
            f"batch_column={batch_column!r} must have >=2 levels. Found: {list(unique_batches)}"
        )
    counts = batches.value_counts()
    underpopulated = counts[counts < 2].index.tolist()
    if underpopulated:
        raise ValueError(
            f"Every batch must have >=2 samples. Under-populated batches: {underpopulated}. "
            "ComBat cannot estimate a batch effect from a single sample."
        )
    if ref_batch is not None and str(ref_batch) not in set(unique_batches):
        raise ValueError(
            f"ref_batch={ref_batch!r} is not a level of {batch_column!r}. "
            f"Available: {list(unique_batches)}"
        )
    for cov in covariates or ():
        if cov not in adata.obs.columns:
            raise KeyError(
                f"covariate={cov!r} not in adata.obs. Available: {list(adata.obs.columns)}"
            )
        if cov == batch_column:
            raise ValueError(f"covariate={cov!r} is the batch column; cannot self-adjust.")
    if layer is not None and layer not in adata.layers:
        raise KeyError(
            f"layer={layer!r} not in adata.layers. Available: {list(adata.layers.keys())}"
        )


def _get_matrix(adata: ad.AnnData, *, layer: str | None) -> np.ndarray:
    X = adata.X if layer is None else adata.layers[layer]
    return np.asarray(X, dtype=float)


def _validate_no_nan(X: np.ndarray, *, layer: str | None) -> None:
    if np.isnan(X).any():
        n_nan = int(np.isnan(X).sum())
        where = f"layer={layer!r}" if layer else "adata.X"
        raise ValueError(
            f"{n_nan} NaN values in {where}. ComBat cannot fit rows with missing "
            "values; run alphaphos.filter_by_completeness + alphaphos.impute_hybrid first."
        )


def _validate_log_scale(X: np.ndarray, *, layer: str | None) -> None:
    med = float(np.nanmedian(X))
    if med > LOG_SCALE_MEDIAN_CEILING:
        where = f"layer={layer!r}" if layer else "adata.X"
        raise ValueError(
            f"Data in {where} looks linear-scale (median={med:.3g}). "
            "ComBat's normal-model variant expects log2-scale input. Pass "
            f"layer='{LAYER_INTENSITY_LOG2}' or log2-transform first."
        )


def _warn_on_small_batches(unique_batches: np.ndarray, batch_sizes: np.ndarray) -> None:
    small = [
        (b, int(n))
        for b, n in zip(unique_batches.tolist(), batch_sizes.tolist(), strict=True)
        if n < MIN_REPLICATES_WARNING
    ]
    if small:
        logger.warning(
            "Batches with <%d samples: %s. ComBat will run but the empirical-Bayes "
            "prior on batch variance will be very weak -- interpret with caution.",
            MIN_REPLICATES_WARNING,
            small,
        )


def _warn_on_confounding(
    adata: ad.AnnData,
    *,
    batch_column: str,
    covariates: list[str],
) -> None:
    # If any covariate has a level that appears in only ONE batch, its
    # variance is fully collinear with the batch effect there. ComBat's
    # linear-model step will complain about a singular design; better to
    # warn upfront with a clear diagnosis.
    for cov in covariates:
        crosstab = pd.crosstab(adata.obs[cov].astype(str), adata.obs[batch_column].astype(str))
        cov_batch_counts = (crosstab > 0).sum(axis=1)
        confounded = cov_batch_counts[cov_batch_counts < 2].index.tolist()
        if confounded:
            logger.warning(
                "Covariate %r has levels %s that appear in only ONE batch each -- "
                "those biological effects cannot be separated from the batch "
                "effect and ComBat may fail with a singular design.",
                cov,
                confounded,
            )


def _build_covar_mod(adata: ad.AnnData, *, covariates: list[str] | None):
    if not covariates:
        return None
    obs_df = adata.obs.copy()
    for cov in covariates:
        obs_df[cov] = obs_df[cov].astype(str)
    formula = "~ " + " + ".join(covariates)
    # patsy's default design INCLUDES an intercept; pycombat expects a
    # full-rank matrix (a no-intercept design with one categorical becomes
    # singular in pycombat's internal linear model).
    design = patsy.dmatrix(formula, data=obs_df)
    return np.asarray(design)


def _stamp_provenance(
    adata: ad.AnnData,
    *,
    batch_column: str,
    covariates: list[str] | None,
    layer: str | None,
    keep_precombat: bool,
    settings: dict,
    unique_batches: np.ndarray,
    batch_sizes: np.ndarray,
) -> None:
    if UNS_ALPHAPHOS not in adata.uns:
        adata.uns[UNS_ALPHAPHOS] = {}
    adata.uns[UNS_ALPHAPHOS][UNS_BATCH_CORRECTION] = {
        "method": "combat",
        "backend": "inmoose.pycombat.pycombat_norm",
        "inmoose_version": _INMOOSE_VERSION,
        "batch_column": batch_column,
        "covariates": list(covariates or []),
        "layer": layer,
        "kept_precombat_layer": keep_precombat,
        "par_prior": settings["par_prior"],
        "mean_only": settings["mean_only"],
        "ref_batch": settings["ref_batch"],
        "n_batches": len(unique_batches),
        "batch_sizes": {
            str(b): int(n)
            for b, n in zip(unique_batches.tolist(), batch_sizes.tolist(), strict=True)
        },
    }
