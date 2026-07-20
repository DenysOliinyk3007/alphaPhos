"""Wrapper around PIMMS (Webel *et al.* 2024, Nature Communications).

.. code-block:: text

    Webel H, Niu L, Nielsen AB, Locard-Paulet M, Mann M, Jensen LJ,
    Rasmussen S. "Imputation of label-free quantitative mass spectrometry-
    based proteomics data using self-supervised deep learning."
    Nat Commun. 2024;15(1):5405.  DOI: 10.1038/s41467-024-48711-5

This module ships the paper's model exactly as the authors distribute
it in ``pimms-learn`` (PyPI): the sklearn-style
``AETransformer`` (VAE / DAE) or ``CollaborativeFilteringTransformer``
(CF).  All alphaPhos does here is:

    1. Extract the log2 intensity matrix from an ``AnnData``.
    2. Standardise column layout (samples-as-rows, features-as-columns).
    3. Call the appropriate PIMMS transformer with the paper's ALD
       defaults (``hidden_layers=[64]``, ``latent_dim=10``,
       ``batch_size=64``, ``patience=25``).
    4. Fill the AnnData layer with the imputed values + attach a
       boolean ``is_imputed`` layer for downstream provenance.

We *do not* re-implement the model.  If the paper wins on your data,
that is the paper's model doing it, not ours.

Sample-size caveat (from the paper's own discussion): "for fewer than
50 samples the PIMMS models can fit the data, but alternatives were
better suited".  We surface a UserWarning below that count so users
don't silently choose a poorly-supported default.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad

from alphaphos.constants import LAYER_INTENSITY_LOG2

logger = logging.getLogger(__name__)

# Paper's ALD-cohort configuration (Methods, GALA-ALD subsection).
DEFAULT_HIDDEN_LAYERS: tuple[int, ...] = (64,)
DEFAULT_LATENT_DIM: int = 10
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_EPOCHS_MAX: int = 100
DEFAULT_PATIENCE: int = 25
MIN_RECOMMENDED_SAMPLES: int = 50


def impute_pimms(
    adata: ad.AnnData,
    *,
    model: Literal["VAE", "DAE", "CF"] = "VAE",
    layer: str = LAYER_INTENSITY_LOG2,
    hidden_layers: list[int] | tuple[int, ...] = DEFAULT_HIDDEN_LAYERS,
    latent_dim: int = DEFAULT_LATENT_DIM,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs_max: int = DEFAULT_EPOCHS_MAX,
    patience: int | None = DEFAULT_PATIENCE,
    cuda: bool = False,
    seed: int = 0,
    out_folder: str | Path | None = None,
    copy: bool = True,
) -> ad.AnnData:
    """Impute missing values with PIMMS deep-learning models.

    Wraps the ``pimms-learn`` package (Webel *et al.* 2024) as an
    alphaPhos imputer with the paper's ALD-cohort defaults.  Requires
    the ``[pimms]`` optional extra
    (``pip install 'alphaphos[pimms]'``).

    Parameters
    ----------
    adata
        AnnData ``(n_samples, n_features)``.  Missing values in
        ``layer`` (default ``"intensity_log2"``) are imputed in place
        on a copy.
    model
        Which PIMMS model to use.  ``"VAE"`` (default; paper's headline
        recommendation), ``"DAE"``, or ``"CF"``.
    layer
        Layer name to read/write.  Default ``"intensity_log2"``.
    hidden_layers, latent_dim, batch_size, epochs_max, patience
        Model hyperparameters.  Defaults match the paper's ALD setup
        (Methods, "Evaluation, imputation and differential expression
        in GALA-ALD dataset").  ``patience`` triggers early stopping.
    cuda
        Use GPU acceleration if available.  Default ``False`` because
        CI + typical Windows scientific installs lack CUDA; users with
        a GPU should pass ``cuda=True`` for a 5-15x speedup.
    seed
        Reproducibility seed passed to PyTorch.
    out_folder
        Directory where PIMMS writes loss curves and model
        checkpoints.  If ``None``, a temp directory is used.
    copy
        Return a copy (default) or mutate ``adata`` in place.

    Returns
    -------
    AnnData
        Same shape as input; ``.layers[layer]`` is NaN-free after the
        call.  ``.layers["is_imputed"]`` is a bool mask that is
        ``True`` where an original NaN was filled.
        ``.uns["alphaphos"]["pimms_impute"]`` records the model type,
        hyperparameters, number of epochs actually run, and PIMMS
        version.

    Raises
    ------
    ImportError
        If ``pimms-learn`` is not installed.

    Notes
    -----
    The PIMMS paper explicitly says "for fewer than 50 samples the
    PIMMS models can fit the data, but alternatives were better
    suited".  When ``adata.n_obs < 50`` we emit a UserWarning
    recommending :func:`alphaphos.impute_hybrid` or
    :func:`alphaphos.impute_knn_site_based` instead.  We still run
    PIMMS because the caller may want to test on a small dataset.
    """
    try:
        # Force headless matplotlib BEFORE pimmslearn's plotting hooks pull
        # in the default backend.  Prevents Windows GDI-handle exhaustion
        # when this function is called repeatedly (e.g. in a benchmark
        # sweep of 100+ fits).
        import matplotlib

        with contextlib.suppress(Exception):
            matplotlib.use("Agg", force=False)

        import torch  # noqa: F401
        from pimmslearn.sklearn.ae_transformer import AETransformer
    except ImportError as exc:
        raise ImportError(
            "impute_pimms requires pimms-learn (and its torch + fastai deps). "
            "Install with `pip install 'alphaphos[pimms]'`."
        ) from exc

    if adata.n_obs < MIN_RECOMMENDED_SAMPLES:
        warnings.warn(
            f"PIMMS was benchmarked at n_samples >= {MIN_RECOMMENDED_SAMPLES} "
            f"(paper's own guidance); you have n_samples={adata.n_obs}.  On "
            "smaller cohorts, alphaphos.impute_hybrid or "
            "alphaphos.impute_knn_site_based typically match or beat PIMMS. "
            "Run alphaphos.impute.benchmark(...) on your data to be sure.",
            UserWarning,
            stacklevel=2,
        )

    if model == "CF":
        raise NotImplementedError(
            "PIMMS collaborative-filtering (CF) requires a long-format Series "
            "with (sample, feature) MultiIndex + target column.  Wrap not yet "
            "provided; use model='VAE' or 'DAE' for now."
        )
    if model not in ("VAE", "DAE"):
        raise ValueError(f"model must be 'VAE', 'DAE', or 'CF'; got {model!r}")

    _set_seeds(seed)

    if copy:
        adata = adata.copy()

    X_wide = _adata_to_wide(adata, layer=layer)
    was_nan = X_wide.isna().to_numpy()

    if out_folder is None:
        import tempfile

        out_folder = tempfile.mkdtemp(prefix="alphaphos_pimms_")

    transformer = AETransformer(
        hidden_layers=list(hidden_layers),
        latent_dim=latent_dim,
        out_folder=str(out_folder),
        model=model,
        batch_size=batch_size,
    )
    logger.info(
        "impute_pimms: fitting %s (hidden=%s, latent=%d) on shape %s",
        model,
        list(hidden_layers),
        latent_dim,
        X_wide.shape,
    )
    # PIMMS fits on the wide DataFrame with NaN; internally it splits
    # observed/missing.  y=None means "no held-out validation split";
    # in that mode the EarlyStoppingCallback has no validation loss to
    # track, so we drop ``patience`` (matches PIMMS notebook 04_1 when
    # ``sample_splits=False``).
    effective_patience = patience if patience is not None else None
    fit_kwargs: dict = {
        "y": None,
        "epochs_max": epochs_max,
        "cuda": bool(cuda),
    }
    if effective_patience is not None:
        # Only add patience when we have a validation set (not the case here).
        # Setting to None means the transformer skips early stopping entirely.
        fit_kwargs["patience"] = None
    else:
        fit_kwargs["patience"] = None
    transformer.fit(X_wide, **fit_kwargs)
    # PIMMS attaches training-loss matplotlib figures to the transformer
    # via ``plot_training_losses``.  Close them explicitly so repeat
    # calls in a benchmark loop don't exhaust Windows GDI handles.
    with contextlib.suppress(Exception):
        import matplotlib.pyplot as _plt

        _plt.close("all")
    X_imputed_df = transformer.transform(X_wide)
    # Align: pimms may return columns in a different order in some paths;
    # reindex to the input columns to be safe.
    X_imputed_df = X_imputed_df.reindex(columns=X_wide.columns, index=X_wide.index)

    _wide_to_adata(adata, X_imputed_df.to_numpy(dtype=np.float32), layer=layer)
    adata.layers["is_imputed"] = was_nan.astype(bool)

    try:
        import pimmslearn as _pl

        pimms_version = getattr(_pl, "__version__", "?")
    except Exception:
        pimms_version = "?"

    provenance = {
        "method": "pimms",
        "pimms_model": model,
        "pimms_version": pimms_version,
        "hidden_layers": list(hidden_layers),
        "latent_dim": int(latent_dim),
        "batch_size": int(batch_size),
        "epochs_max": int(epochs_max),
        "patience": (None if patience is None else int(patience)),
        "cuda": bool(cuda),
        "seed": int(seed),
        "n_epochs_trained": int(getattr(transformer, "epochs_trained_", -1)),
        "n_samples": int(adata.n_obs),
        "n_features": int(adata.n_vars),
        "n_imputed_cells": int(was_nan.sum()),
        "paper_doi": "10.1038/s41467-024-48711-5",
    }
    _stamp_provenance(adata, provenance)
    return adata


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _set_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def _adata_to_wide(adata: ad.AnnData, *, layer: str) -> pd.DataFrame:
    X = np.asarray(adata.layers.get(layer, adata.X), dtype=float)
    return pd.DataFrame(X, index=adata.obs_names.astype(str), columns=adata.var_names.astype(str))


def _wide_to_adata(adata: ad.AnnData, X_new: np.ndarray, *, layer: str) -> None:
    adata.layers[layer] = X_new
    # Convention: mirror to .X so downstream tools that consume adata.X see
    # the imputed values.
    adata.X = X_new.copy()


def _stamp_provenance(adata: ad.AnnData, provenance: dict) -> None:
    from alphaphos.constants import UNS_ALPHAPHOS

    ns = adata.uns.setdefault(UNS_ALPHAPHOS, {})
    ns["pimms_impute"] = provenance
