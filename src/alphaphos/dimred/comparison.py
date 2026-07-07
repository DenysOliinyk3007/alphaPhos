"""Imputation-impact diagnostic.

Question this answers: "does the imputation step distort the sample-space
structure enough to worry me?"

Runs standard PCA on the imputed matrix and NIPALS (or PPCA) on the raw
matrix (with NaN), then reports per-sample agreement in PC space and
per-PC correlation.  Sign-flips are handled automatically (each PC
direction is arbitrary up to sign).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from alphaphos.dimred.pca import pca

if TYPE_CHECKING:
    import anndata as ad


def compare_imputation_impact(
    adata_raw: ad.AnnData,
    adata_imputed: ad.AnnData,
    *,
    n_components: int = 5,
    layer: str = "intensity_log2",
    method_raw: Literal["nipals", "ppca"] = "nipals",
    advanced_raw: dict[str, Any] | None = None,
    advanced_imputed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare PCA on raw (NaN-containing) vs imputed data.

    Parameters
    ----------
    adata_raw
        AnnData BEFORE imputation.  ``.layers[layer]`` typically has NaN.
    adata_imputed
        AnnData AFTER imputation.  ``.layers[layer]`` must be complete.
        Sample identities and order must match ``adata_raw``.
    n_components
        Number of PCs to compare.  Default 5 (usually enough for QC).
    layer
        Layer to compute on in both objects.  Default ``"intensity_log2"``.
    method_raw
        Missing-value PCA method for ``adata_raw``.  Default ``"nipals"``.
    advanced_raw, advanced_imputed
        Extra overrides forwarded to :func:`pca` for each run.

    Returns
    -------
    Dict with these keys:

    - ``pc_coords_imputed``   -- (n_samples, n_components)
    - ``pc_coords_raw``       -- (n_samples, n_components)
    - ``variance_ratio_imputed`` / ``variance_ratio_raw``  -- (n_components,)
    - ``per_pc_correlation``  -- (n_components,) abs Pearson r between the two
                                 PC axes.  1.0 = imputation had no effect on
                                 that axis.  Below ~0.9 suggests distortion.
    - ``per_sample_distance`` -- (n_samples,) L2 distance between the two
                                 samples' coordinates in the first
                                 ``n_components`` PCs.
    - ``verdict``             -- short human-readable summary string.
    """
    if list(adata_raw.obs_names.astype(str)) != list(adata_imputed.obs_names.astype(str)):
        raise ValueError(
            "adata_raw and adata_imputed must have identical obs_names in the "
            "same order (they are the same experiment, one pre- and one "
            "post-imputation)."
        )
    if adata_raw.n_vars != adata_imputed.n_vars:
        raise ValueError("adata_raw and adata_imputed must have the same number of features.")

    # Run PCA on the imputed matrix (standard sklearn, requires complete data)
    adata_i2 = pca(
        adata_imputed,
        n_components=n_components,
        layer=layer,
        handle_missing="error",
        advanced=advanced_imputed,
        copy=True,
    )

    # Run PCA on the raw matrix with missing-value-aware method
    adata_r2 = pca(
        adata_raw,
        n_components=n_components,
        layer=layer,
        handle_missing=method_raw,
        advanced=advanced_raw,
        copy=True,
    )

    Zi = np.asarray(adata_i2.obsm["X_pca"], dtype=np.float64)  # imputed scores
    Zr = np.asarray(adata_r2.obsm["X_pca"], dtype=np.float64)  # raw scores
    var_i = np.asarray(adata_i2.uns["pca"]["variance_ratio"], dtype=np.float64)
    var_r = np.asarray(adata_r2.uns["pca"]["variance_ratio"], dtype=np.float64)

    # Sign-flip alignment: each PC is arbitrary up to sign.  Flip the raw PC
    # to correlate positively with the imputed one before computing distance.
    per_pc_corr = np.zeros(n_components, dtype=np.float64)
    Zr_aligned = np.zeros_like(Zr)
    for c in range(n_components):
        a = Zi[:, c] - Zi[:, c].mean()
        b = Zr[:, c] - Zr[:, c].mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            per_pc_corr[c] = 0.0
            Zr_aligned[:, c] = Zr[:, c]
            continue
        r = float(a @ b) / denom
        per_pc_corr[c] = abs(r)
        # Flip sign of the raw axis if it anti-correlates with the imputed
        Zr_aligned[:, c] = Zr[:, c] * (1.0 if r >= 0 else -1.0)

    # Per-sample L2 distance in PC space (after alignment).  Normalise by
    # sqrt(n_components) so the number can be interpreted as an average
    # per-PC displacement, comparable across choices of n_components.
    per_sample_distance = np.linalg.norm(Zi - Zr_aligned, axis=1) / np.sqrt(n_components)

    # Verdict heuristic: PCs with corr < 0.7 = "distorted"; overall summary
    n_distorted = int((per_pc_corr < 0.7).sum())
    if n_distorted == 0:
        verdict = (
            f"OK: imputation preserved all top-{n_components} PCs "
            f"(min per-PC |r| = {per_pc_corr.min():.3f})."
        )
    elif n_distorted <= n_components // 2:
        verdict = (
            f"BORDERLINE: {n_distorted}/{n_components} PCs show distortion "
            f"(|r| < 0.7).  Inspect per-sample distances before publishing figures."
        )
    else:
        verdict = (
            f"DISTORTED: {n_distorted}/{n_components} PCs disagree between "
            "imputed and raw.  Imputation is materially changing the sample "
            "structure; consider a NIPALS/PPCA scatter for publication."
        )

    return {
        "pc_coords_imputed": Zi,
        "pc_coords_raw": Zr_aligned,
        "variance_ratio_imputed": var_i,
        "variance_ratio_raw": var_r,
        "per_pc_correlation": per_pc_corr,
        "per_sample_distance": per_sample_distance,
        "sample_labels": list(adata_raw.obs_names.astype(str)),
        "n_components": int(n_components),
        "method_raw": method_raw,
        "verdict": verdict,
    }


def _format_report(report: dict[str, Any]) -> str:
    """Human-readable formatted output for logging."""
    lines = [report["verdict"], ""]
    lines.append("PC   var_imp   var_raw   |r|")
    for c in range(report["n_components"]):
        lines.append(
            f"{c + 1:>3}  {report['variance_ratio_imputed'][c]:>7.4f}  "
            f"{report['variance_ratio_raw'][c]:>7.4f}  "
            f"{report['per_pc_correlation'][c]:>5.3f}"
        )
    lines.append("")
    lines.append("Per-sample PC-space displacement:")
    lines.append(
        pd.Series(
            report["per_sample_distance"],
            index=report["sample_labels"],
            name="distance",
        ).to_string()
    )
    return "\n".join(lines)
