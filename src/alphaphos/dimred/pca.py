"""PCA with three missing-value strategies.

Design decision
---------------
When the input matrix is complete (no NaN), the three methods below
converge to the same result numerically (down to sign flips and rotation
within tied-variance components).  They differ only in HOW they treat
missing values:

- ``handle_missing="error"`` -- standard PCA (sklearn).  Raises on NaN.
- ``handle_missing="nipals"`` -- iterative NIPALS.  Skips NaN in the
  regression sums.  No imputation required.
- ``handle_missing="ppca"`` -- probabilistic PCA (Tipping & Bishop 1999).
  EM algorithm; NaN values are integrated out of the observed-data
  likelihood.

Output written to the ``AnnData``:

- ``.obsm["X_pca"]``          -- (n_samples, n_components) sample scores
- ``.varm["PCs"]``            -- (n_features, n_components) loadings
- ``.uns["pca"]["variance"]`` -- (n_components,) eigenvalues
- ``.uns["pca"]["variance_ratio"]`` -- (n_components,) fraction of total variance
- ``.uns["pca"]["method"]``   -- algorithm actually run: "standard" | "nipals" | "ppca"
- ``.uns["pca"]["n_missing_in_layer"]`` -- NaN count in the input layer
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import anndata as ad

try:
    from sklearn.decomposition import PCA as _SklearnPCA

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


DEFAULT_PCA_SETTINGS: dict[str, Any] = {
    "n_components": 10,
    "layer": "intensity_log2",
    "handle_missing": "error",
    # Ceiling on iterations for NIPALS / PPCA convergence loops.
    "max_iter": 1000,
    # Relative convergence tolerance for NIPALS / PPCA.
    "tol": 1e-6,
    # Centre (subtract feature mean) before PCA.  Standard.
    "center": True,
    # Scale (divide by feature std) before PCA.  Off by default -- log2
    # intensities are already comparable across sites and scaling amplifies
    # low-abundance noise.
    "scale": False,
    # RNG seed for PPCA random initialisation.
    "seed": 42,
}

_ALLOWED_HANDLE_MISSING = ("error", "nipals", "ppca")


def resolve_pca_settings(advanced: dict[str, Any] | None) -> dict[str, Any]:
    """Merge overrides over :data:`DEFAULT_PCA_SETTINGS` with validation."""
    out = dict(DEFAULT_PCA_SETTINGS)
    if not advanced:
        return out
    unknown = set(advanced) - set(DEFAULT_PCA_SETTINGS)
    if unknown:
        raise ValueError(
            f"Unknown advanced keys: {sorted(unknown)}. Allowed: {sorted(DEFAULT_PCA_SETTINGS)}"
        )
    out.update(advanced)
    if not isinstance(out["n_components"], int) or out["n_components"] < 1:
        raise ValueError(f"n_components must be positive int; got {out['n_components']!r}")
    if out["handle_missing"] not in _ALLOWED_HANDLE_MISSING:
        raise ValueError(
            f"handle_missing must be one of {_ALLOWED_HANDLE_MISSING}; "
            f"got {out['handle_missing']!r}"
        )
    if not isinstance(out["max_iter"], int) or out["max_iter"] < 1:
        raise ValueError(f"max_iter must be positive int; got {out['max_iter']!r}")
    if not isinstance(out["tol"], (int, float)) or out["tol"] <= 0:
        raise ValueError(f"tol must be positive float; got {out['tol']!r}")
    for bkey in ("center", "scale"):
        if not isinstance(out[bkey], bool):
            raise ValueError(f"{bkey} must be bool; got {type(out[bkey]).__name__}")
    if not isinstance(out["seed"], int):
        raise ValueError(f"seed must be int; got {out['seed']!r}")
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def pca(
    adata: ad.AnnData,
    *,
    n_components: int | None = None,
    layer: str | None = None,
    handle_missing: Literal["error", "nipals", "ppca"] | None = None,
    advanced: dict[str, Any] | None = None,
    copy: bool = False,
) -> ad.AnnData:
    """Principal Component Analysis on samples.

    Parameters
    ----------
    adata
        Site-level ``AnnData``.  ``(n_samples, n_features)``.
    n_components, layer, handle_missing
        Convenience overrides for the corresponding advanced setting.
        Passing these takes precedence over the ``advanced`` dict.
    advanced
        Overrides for :data:`DEFAULT_PCA_SETTINGS`.
    copy
        If ``True``, mutate a fresh copy of ``adata``; else in-place.
        The (possibly-mutated) AnnData is returned in both cases.
    """
    if not _HAS_SKLEARN:  # pragma: no cover
        raise ImportError("scikit-learn is required for alphaphos.dimred.pca.")

    # Resolve settings; convenience-arg overrides take precedence over advanced.
    advanced_merged = dict(advanced or {})
    if n_components is not None:
        advanced_merged["n_components"] = n_components
    if layer is not None:
        advanced_merged["layer"] = layer
    if handle_missing is not None:
        advanced_merged["handle_missing"] = handle_missing
    settings = resolve_pca_settings(advanced_merged)

    adata = adata.copy() if copy else adata
    X = _get_matrix(adata, layer=settings["layer"])
    n_missing = int(np.isnan(X).sum())

    # Center / scale (mask-aware -- ignore NaN)
    Xc, feature_means, feature_stds = _center_scale(
        X, center=settings["center"], scale=settings["scale"]
    )

    # Dispatch on missing-value strategy.
    # ``handle_missing`` is the *policy* the user asked for; ``method`` records
    # the *algorithm* that actually ran, which is what belongs in .uns.
    handle_missing = settings["handle_missing"]
    if handle_missing == "error":
        if n_missing > 0:
            raise ValueError(
                f"handle_missing='error' but layer {settings['layer']!r} has "
                f"{n_missing:,} NaN entries.  Impute upstream (e.g. "
                "ap.impute_hybrid) or pass handle_missing='nipals'/'ppca'."
            )
        scores, loadings, variance = _pca_sklearn(Xc, n_components=settings["n_components"])
        method = "standard"
    elif handle_missing == "nipals":
        scores, loadings, variance = _pca_nipals(
            Xc,
            n_components=settings["n_components"],
            max_iter=settings["max_iter"],
            tol=settings["tol"],
        )
        method = "nipals"
    else:  # ppca
        scores, loadings, variance = _pca_ppca(
            Xc,
            n_components=settings["n_components"],
            max_iter=settings["max_iter"],
            tol=settings["tol"],
            seed=settings["seed"],
        )
        method = "ppca"

    # Variance ratio: eigenvalues / total variance (mask-aware)
    total_var = _total_variance(Xc)
    variance_ratio = variance / total_var if total_var > 0 else np.zeros_like(variance)

    adata.obsm["X_pca"] = np.asarray(scores, dtype=np.float64)
    adata.varm["PCs"] = np.asarray(loadings, dtype=np.float64)
    adata.uns["pca"] = {
        "variance": np.asarray(variance, dtype=np.float64),
        "variance_ratio": np.asarray(variance_ratio, dtype=np.float64),
        "method": method,
        "n_components": int(settings["n_components"]),
        "n_missing_in_layer": n_missing,
        "layer": settings["layer"],
        "center": bool(settings["center"]),
        "scale": bool(settings["scale"]),
        "feature_means": np.asarray(feature_means, dtype=np.float64),
        "feature_stds": np.asarray(feature_stds, dtype=np.float64),
    }
    logger.info(
        "pca(%s): %d components on (%d samples, %d features), variance_ratio_first_3=%s",
        method,
        settings["n_components"],
        X.shape[0],
        X.shape[1],
        np.round(variance_ratio[:3], 4).tolist() if len(variance_ratio) >= 3 else variance_ratio,
    )
    return adata


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


def _pca_sklearn(X: np.ndarray, *, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard PCA via sklearn.  ``X`` must have no NaN."""
    k = min(n_components, min(X.shape))
    model = _SklearnPCA(n_components=k, svd_solver="full")
    scores = model.fit_transform(X)
    loadings = model.components_.T  # (n_features, k)
    return scores, loadings, model.explained_variance_


def _pca_nipals(
    X: np.ndarray,
    *,
    n_components: int,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NIPALS PCA with native missing-value handling.

    For each component:
      1. Initialise scores ``t`` with the column of ``X`` that has the
         highest sum-of-squares over its non-missing entries.
      2. Loadings ``p_j = sum_i(mask_ij * t_i * X_ij) / sum_i(mask_ij * t_i^2)``.
         Normalise ``p`` to unit length.
      3. Scores ``t_new = sum_j(mask_ij * X_ij * p_j) / sum_j(mask_ij * p_j^2)``.
      4. Iterate until ``||t_new - t|| / ||t|| < tol`` or ``max_iter``.
      5. Store ``(t, p)`` as PC ``k``; deflate ``X`` (subtract ``t p^T``,
         leaving NaN cells NaN).

    NaN-safe via a boolean mask and pre-zeroed X.  Vectorised: no
    per-element loops.
    """
    n, m = X.shape
    k = min(n_components, min(n, m))
    mask = ~np.isnan(X)
    mask_f = mask.astype(np.float64)
    Xw = np.where(mask, X, 0.0).astype(np.float64)

    scores = np.zeros((n, k), dtype=np.float64)
    loadings = np.zeros((m, k), dtype=np.float64)
    variance = np.zeros(k, dtype=np.float64)

    for comp in range(k):
        # Initialise t with column of highest available SS (over non-missing entries)
        col_ss = np.nansum(Xw * mask_f, axis=0) ** 2 + (Xw**2 * mask_f).sum(axis=0)
        best_col = int(np.argmax(col_ss))
        t = Xw[:, best_col].copy()
        if np.linalg.norm(t) == 0:
            t = np.ones(n, dtype=np.float64)

        for _ in range(max_iter):
            # p_j = (X_zeroed^T @ t) / (mask^T @ t^2), NaN-safe
            p_num = Xw.T @ t
            p_den = mask_f.T @ (t**2)
            p_den = np.where(p_den > 0, p_den, 1.0)
            p = p_num / p_den
            p_norm = np.linalg.norm(p)
            if p_norm == 0:
                break
            p = p / p_norm

            # t_i = (X_zeroed_i @ p) / (mask_i @ p^2)
            t_num = Xw @ p
            t_den = mask_f @ (p**2)
            t_den = np.where(t_den > 0, t_den, 1.0)
            t_new = t_num / t_den

            denom = np.linalg.norm(t)
            if denom > 0 and np.linalg.norm(t_new - t) / denom < tol:
                t = t_new
                break
            t = t_new

        # Deflate: subtract t*p^T from non-missing cells
        deflate = np.outer(t, p)
        Xw = np.where(mask, Xw - deflate, 0.0)

        scores[:, comp] = t
        loadings[:, comp] = p
        # Variance explained: SS of this component's reconstruction at OBSERVED
        # cells, divided by (n-1) to match sklearn/PPCA units.  The classical
        # ``np.var(t, ddof=1)`` NIPALS eigenvalue blows up on sparse data
        # because the least-squares update ``t_i = Sum(mask * X * p) /
        # Sum(mask * p^2)`` divides by a small denominator when few features
        # are observed for sample i, inflating ``|t|`` without a matching
        # counter-inflation of the true signal.  The mask-aware
        # reconstruction-SS/(n-1) is bounded by the mask-aware total variance
        # (see ``_total_variance``), so variance_ratio stays in [0, 1].
        recon_ss = float(np.sum(np.where(mask, deflate, 0.0) ** 2))
        variance[comp] = recon_ss / (n - 1) if n > 1 else 0.0

    return scores, loadings, variance


def _pca_ppca(
    X: np.ndarray,
    *,
    n_components: int,
    max_iter: int,
    tol: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Probabilistic PCA with missing values via EM (Tipping-Bishop 1999).

    Model: ``x = W z + mu + eps``, ``z ~ N(0, I)``, ``eps ~ N(0, sigma^2 I)``.
    With missing values, the log-likelihood factorises per sample over its
    observed entries; the EM updates use the posterior on ``z`` given only
    the observed entries.

    Assumes ``X`` is already centred (feature mean subtracted).  The
    returned scores/loadings satisfy the same shape contract as the
    other backends; ``variance`` is the marginal per-component variance
    of the posterior scores.
    """
    rng = np.random.default_rng(seed)
    n, m = X.shape
    k = min(n_components, min(n, m))
    mask = ~np.isnan(X)
    mask_f = mask.astype(np.float64)

    # Initialise W randomly (small values), sigma^2 = variance-per-cell floor
    W = rng.normal(0.0, 0.1, size=(m, k))
    obs_var = float(np.nanvar(X)) if np.any(mask) else 1.0
    sigma2 = max(obs_var * 0.5, 1e-6)

    # Precompute per-sample observed indices
    per_sample_obs = [np.where(mask[i, :])[0] for i in range(n)]

    prev_ll = -np.inf
    Z_mean = np.zeros((n, k), dtype=np.float64)  # posterior means
    for _it in range(max_iter):
        # ---- E-step: posterior on z_i given observed x_i ----
        Z_mean_new = np.zeros_like(Z_mean)
        # Per-sample posterior covariance sum for M-step
        ExxT_sum = np.zeros((k, k), dtype=np.float64)
        Wnew_num = np.zeros((m, k), dtype=np.float64)  # sum_i x_i z_i^T (observed only)
        Wnew_den = np.zeros((m, k, k), dtype=np.float64)  # sum_i E[z z^T] over obs of feature j
        resid_ss = 0.0
        n_obs_total = 0

        for i in range(n):
            obs_i = per_sample_obs[i]
            if len(obs_i) == 0:
                continue
            Wo = W[obs_i, :]  # (|O|, k)
            x_o = X[i, obs_i]  # (|O|,)
            M = Wo.T @ Wo + sigma2 * np.eye(k)
            M_inv = np.linalg.inv(M)
            z_mean = M_inv @ (Wo.T @ x_o)  # (k,)
            Z_mean_new[i, :] = z_mean
            Ezz = sigma2 * M_inv + np.outer(z_mean, z_mean)  # (k, k)
            ExxT_sum += Ezz

            # M-step accumulators, per feature (only observed features contribute)
            for j_idx, j in enumerate(obs_i):
                Wnew_num[j, :] += x_o[j_idx] * z_mean
                Wnew_den[j, :, :] += Ezz

            # Residual sum of squares (for sigma^2 update)
            recon = Wo @ z_mean  # (|O|,)
            r = x_o - recon
            resid_ss += float(np.dot(r, r))
            n_obs_total += len(obs_i)

        # ---- M-step: update W per feature ----
        W_new = np.zeros_like(W)
        for j in range(m):
            n_obs_j = int(mask_f[:, j].sum())
            if n_obs_j == 0:
                continue
            denom = Wnew_den[j]
            # Small ridge to keep denom invertible
            try:
                W_new[j, :] = np.linalg.solve(denom + 1e-8 * np.eye(k), Wnew_num[j, :])
            except np.linalg.LinAlgError:
                W_new[j, :] = np.linalg.pinv(denom) @ Wnew_num[j, :]

        # ---- M-step: update sigma^2 ----
        sigma2_new = max(resid_ss / max(n_obs_total, 1), 1e-8)

        # Convergence check via approximate log-likelihood (obs-fit only).
        # First iteration compares against -inf, so skip convergence then.
        ll = -0.5 * resid_ss / sigma2 - 0.5 * n_obs_total * np.log(2 * np.pi * sigma2)
        rel_delta = abs(ll - prev_ll) / max(abs(prev_ll), 1.0) if np.isfinite(prev_ll) else np.inf
        W = W_new
        sigma2 = sigma2_new
        Z_mean = Z_mean_new
        if rel_delta < tol:
            break
        prev_ll = ll

    # Orient with SVD of W: makes loadings orthonormal and orders components
    # by descending singular value (matches sklearn convention).
    U, _S, _VT = np.linalg.svd(W, full_matrices=False)
    loadings = U  # (m, k), unit-norm columns

    # Compute projection-style scores mask-aware, matching the NIPALS + sklearn
    # convention (scores = X projected onto loading directions).  The PPCA EM
    # output ``Z_mean`` is the posterior of the LATENT variable z which has
    # unit variance in the prior; treating it as PC scores under-reports the
    # observed-space variance explained by each component (Z_mean has variance
    # ~1 while projections have variance ~ eigenvalue of the sample
    # covariance).  Projecting X onto U puts scores in the same units as
    # sklearn/NIPALS so variance_ratio is comparable across methods.
    Xw = np.where(mask, X, 0.0)
    score_num = Xw @ loadings  # (n, k)
    score_den = mask_f @ (loadings**2)  # (n, k)
    score_den = np.where(score_den > 0, score_den, 1.0)
    scores = score_num / score_den

    # Variance per component: SS of reconstruction at observed cells / (n-1).
    # Same formulation as NIPALS -- bounded by mask-aware total variance.
    variance = np.zeros(k, dtype=np.float64)
    for comp in range(k):
        deflate = np.outer(scores[:, comp], loadings[:, comp])
        recon_ss = float(np.sum(np.where(mask, deflate, 0.0) ** 2))
        variance[comp] = recon_ss / (n - 1) if n > 1 else 0.0
    return scores, loadings, variance


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def _get_matrix(adata: ad.AnnData, *, layer: str) -> np.ndarray:
    if layer not in adata.layers:
        # Fall back to .X if the requested layer isn't there (with a note).
        if layer == "intensity_log2":
            X = adata.X
        else:
            raise KeyError(f"Layer {layer!r} not in adata.layers ({list(adata.layers.keys())}).")
    else:
        X = adata.layers[layer]
    return np.asarray(X, dtype=np.float64)


def _center_scale(
    X: np.ndarray, *, center: bool, scale: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature-wise mean-centre and (optionally) unit-scale, NaN-safe.

    Returns the transformed matrix, the per-feature means, and per-feature stds
    (both 1D of length ``n_features``).  When ``center=False`` the means array
    is zeros; when ``scale=False`` the stds array is ones.
    """
    _n, m = X.shape
    if center:
        with np.errstate(all="ignore"):
            means = np.nanmean(X, axis=0)
        means = np.where(np.isnan(means), 0.0, means)
    else:
        means = np.zeros(m, dtype=np.float64)
    Xc = X - means[np.newaxis, :]

    if scale:
        with np.errstate(all="ignore"):
            stds = np.nanstd(X, axis=0, ddof=1)
        stds = np.where((~np.isnan(stds)) & (stds > 0), stds, 1.0)
    else:
        stds = np.ones(m, dtype=np.float64)
    if scale:
        Xc = Xc / stds[np.newaxis, :]
    return Xc, means, stds


def _total_variance(X: np.ndarray) -> float:
    """Total variance = SS at observed cells / (n-1).

    Mask-aware: NaN cells contribute nothing.  On complete data this
    matches ``sum_j(var(X[:,j], ddof=1))`` (the classical trace of the
    covariance matrix); on masked data it stays in the same units so
    variance ratios against ``_pca_nipals`` / ``_pca_ppca`` per-component
    variances (also SS/(n-1)) stay in [0, 1].
    """
    n = X.shape[0]
    if n <= 1:
        return 0.0
    mask = ~np.isnan(X)
    ss = float(np.sum(np.where(mask, X, 0.0) ** 2))
    return ss / (n - 1)
