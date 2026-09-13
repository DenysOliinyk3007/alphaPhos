"""Differential testing entry points.

* :func:`diff_exp_limma` -- two-group moderated t-test on the inmoose limma
  port (``lmFit -> contrasts_fit -> eBayes -> topTable``).
* :func:`diff_exp_limma_contrasts` -- several named contrasts on ONE joint
  fit, moderated with alphaPhos's clean-room Smyth 2004 stack
  (:mod:`alphaphos.stats.linear_model`); ``joint=False`` loops
  :func:`diff_exp_limma` instead.
* :func:`diff_exp_anova` -- moderated F-test across all levels (same stack).
* :func:`anova_hits` -- split an F-test table into ORA hits / background.

The clean-room stack reproduces inmoose's moderated t, prior and F to ~1e-13
(``tests/unit/test_stats_moderated.py``); its F p-value follows limma
(``pf(F, rank, df_prior + df_residual)``) where inmoose 0.9.1 reports the
chi-square limit (``df2 = inf``), which is anti-conservative.

Result columns (two-group / per-contrast tables):

* ``log2fc``   -- ``mean(treatment) - mean(control)`` on the input layer's scale.
* ``se``       -- standard error of ``log2fc``.
* ``t_stat``   -- moderated t-statistic.
* ``p_value``  -- p-value from the moderated t-test.
* ``fdr``      -- Benjamini-Hochberg adjusted p-value across all sites.
* ``B``        -- log-odds of differential expression (limma's ``B``).
* ``ave_expr`` -- mean across all samples of the input layer.

The returned DataFrame is indexed by ``adata.var_names`` (input order
preserved).  ``.attrs`` carries a ``contrast_direction`` note stating
the sign convention.
"""

from __future__ import annotations

import importlib.util
import logging
import numbers
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
from alphaphos.stats.design import covariate_is_continuous, sanitize_and_map_levels

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


# patsy + inmoose are gated behind the [stats] extra AND imported lazily
# (inmoose alone costs ~0.4 s at import).  Everything in this module imports
# cleanly without them; the inmoose-backed entry points call
# _require_stats_deps() first so users get an install hint, not a traceback.


def _require_stats_deps(caller: str) -> None:
    try:
        import patsy  # noqa: F401  # type: ignore[import-not-found]
        from inmoose import limma  # noqa: F401  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"{caller} requires inmoose and patsy. Install with `pip install alphaPhos[stats]`."
        ) from exc


# Import-free availability flag (find_spec only); tests use it for skipif.
_HAS_STATS_DEPS = all(importlib.util.find_spec(m) is not None for m in ("inmoose", "patsy"))


DEFAULT_STATS_SETTINGS: dict = {
    # limma-trend: prior variance follows a spline in average log-intensity.
    # MS residual variance falls steeply with intensity (EGF HeLa: residual SD
    # 0.33 -> 0.16 from the lowest to the highest quintile), so the constant
    # prior is misspecified; on by default since 0.24.
    "trend": True,
    # Phipson 2016 robust prior: NOT available -- inmoose 0.9.1 raises
    # NotImplementedError inside squeezeVar and the clean-room stack does not
    # implement it.  Kept so that robust=True fails fast with a clear message.
    "robust": False,
    "winsor_tail_p": (0.05, 0.1),
}

# advanced keys the clean-room (joint / ANOVA) path understands.
_JOINT_PATH_KEYS = frozenset({"trend"})

INMOOSE_TOPTABLE_COLS = {
    "log2FoldChange": "log2fc",
    "lfcSE": "se",
    "stat": "t_stat",
    "pvalue": "p_value",
    "adj_pvalue": "fdr",
    "AveExpr": "ave_expr",
    "B": "B",
}


def diff_exp_limma(
    adata: ad.AnnData,
    *,
    condition_column: str,
    comparison: tuple[str, str],
    covariates: list[str] | None = None,
    layer: str | None = LAYER_INTENSITY_LOG2,
    advanced: dict | None = None,
) -> pd.DataFrame:
    """Run a two-group moderated t-test on ``adata``.

    Parameters
    ----------
    adata : AnnData
        Shape ``(n_samples, n_sites)``. Values must be log-scale and free
        of NaN in the samples used for the comparison.
    condition_column : str
        Column in ``adata.obs`` that holds the group label.
    comparison : (treatment, control)
        Two levels present in ``adata.obs[condition_column]``. Log2FC is
        computed as ``mean(treatment) - mean(control)``; positive =
        higher in ``treatment``.
    covariates : list[str], optional
        Additional ``adata.obs`` columns to include in the design matrix
        (``~ 0 + condition + covariate_1 + ...``). Use for batch / run /
        plate adjustment. Categorical covariates are dummy-coded by patsy.
    layer : str, optional
        Which ``adata.layers`` slot to test. Default ``"intensity_log2"``;
        pass ``None`` to test ``adata.X`` directly.
    advanced : dict, optional
        Overrides for ``DEFAULT_STATS_SETTINGS``:

        - ``trend`` (bool, default True): limma-trend -- the prior variance
          follows a natural cubic spline in average log-intensity (Law 2014).
          Set False for the constant prior of Smyth 2004.
        - ``robust`` (bool): Phipson-2016 robust EB estimator.  **Not
          available** (inmoose 0.9.1 does not implement it); ``True`` raises
          ``NotImplementedError``.
        - ``winsor_tail_p`` ((float, float)): passed to ``eBayes``; only
          meaningful with ``robust``.

    Returns
    -------
    pandas.DataFrame
        One row per site, indexed by ``adata.var_names`` (input order
        preserved). See module docstring for column semantics.
        ``result.attrs["contrast_direction"]`` names the convention.

    Raises
    ------
    ImportError
        If ``inmoose`` is not installed. Install with
        ``pip install alphaPhos[stats]``.
    ValueError
        If input validation fails (bad levels, NaN in the tested layer,
        duplicate var names, non-log data, unknown ``advanced`` keys).
    KeyError
        If ``condition_column`` / ``covariates`` / ``layer`` are absent.
    """
    _require_stats_deps("diff_exp_limma")

    settings = _resolve_stats_settings(advanced)

    # Levels are compared as strings throughout (obs columns are often
    # categorical / integer-coded); the user's original values are kept for
    # the result attrs.
    treatment, control = str(comparison[0]), str(comparison[1])
    _validate_inputs(
        adata,
        condition_column=condition_column,
        treatment=treatment,
        control=control,
        covariates=covariates,
        layer=layer,
    )
    _refuse_double_batch_correction(adata, covariates=covariates, layer=layer)

    sample_mask = adata.obs[condition_column].astype(str).isin([treatment, control]).to_numpy()
    sub = adata[sample_mask, :]

    _warn_on_small_groups(sub, condition_column, [treatment, control])

    X = _get_matrix(sub, layer=layer)
    _validate_no_nan(X, layer=layer)
    _validate_log_scale(X, layer=layer)

    design, level_names = _build_design(
        sub,
        condition_column=condition_column,
        treatment=treatment,
        control=control,
        covariates=covariates,
    )

    # Internal contrast string uses the sanitized levels (they must be valid
    # Python identifiers because inmoose.makeContrasts evals the string).
    contrast_string_internal = f"{level_names['treatment']}-{level_names['control']}"
    # User-facing contrast string preserves the original labels for display /
    # provenance in result.attrs.
    contrast_string_display = f"{condition_column}[{treatment}]-{condition_column}[{control}]"
    logger.info(
        "diff_exp_limma: contrast=%s | n_sites=%d | design=%s",
        contrast_string_display,
        sub.n_vars,
        design.shape,
    )

    from inmoose import limma  # lazy: [stats] extra, checked above

    fit = limma.lmFit(X.T, design)
    contrast_mat = limma.makeContrasts(
        [contrast_string_internal], levels=list(fit.coefficients.columns)
    )
    fit2 = limma.contrasts_fit(fit, contrast_mat)
    fit2 = _safe_ebayes(fit2, settings)

    contrast_name = fit2.coefficients.columns[0]
    tt = pd.DataFrame(limma.topTable(fit2, coef=contrast_name, number=np.inf)).sort_index()
    tt.index = sub.var_names

    return _finalize_result(
        tt,
        treatment=treatment,
        control=control,
        contrast_string=contrast_string_display,
    )


def _refuse_double_batch_correction(
    adata: ad.AnnData,
    *,
    covariates: list[str] | None,
    layer: str | None,
) -> None:
    # If ComBat has already been applied to the layer we're testing and the
    # caller adds the same batch column as a covariate, the batch effect
    # gets subtracted twice: once from the data, once from the model.
    # Result is anti-conservative p-values (inflated false positives).
    bc = adata.uns.get(UNS_ALPHAPHOS, {}).get(UNS_BATCH_CORRECTION)
    if not bc:
        return
    corrected_layer = bc.get("layer")
    # ``layer=None`` tests ``.X``, which by the collapse contract mirrors the
    # canonical log2 layer -- so a correction of that layer applies to it too.
    tested_layer = layer if layer is not None else LAYER_INTENSITY_LOG2
    if corrected_layer != tested_layer:
        # Different layer -- e.g. testing the untouched precombat slot -- is fine.
        return
    prior_batch_col = bc.get("batch_column")
    if prior_batch_col not in (covariates or ()):
        return
    raise ValueError(
        f"Double batch correction detected. adata.layers[{layer!r}] has "
        f"already been ComBat-corrected on batch column "
        f"{prior_batch_col!r} (see adata.uns[{UNS_ALPHAPHOS!r}]"
        f"[{UNS_BATCH_CORRECTION!r}]), and you passed "
        f"covariates=[{prior_batch_col!r}] to diff_exp_limma. Subtracting "
        "the same batch effect twice (once from the data, once in the "
        "linear model) gives anti-conservative p-values.\n\n"
        "Community convention: use ComBat for visualisation only, and "
        "run stats on the raw data with batch as a limma covariate.\n\n"
        "Fix -- pick ONE of:\n"
        f"  A) Statistically preferred: test the pre-correction layer\n"
        f"     with the batch covariate:\n"
        f"        diff_exp_limma(..., covariates=[{prior_batch_col!r}], "
        f"layer={LAYER_INTENSITY_LOG2_PRECOMBAT!r})\n"
        f"  B) Test the ComBat-corrected layer WITHOUT the batch covariate:\n"
        f"        diff_exp_limma(..., covariates=None)  # (or drop "
        f"{prior_batch_col!r} from covariates)"
    )


def _resolve_stats_settings(advanced: dict | None) -> dict:
    """Merge ``advanced`` on top of ``DEFAULT_STATS_SETTINGS``, rejecting unknown keys."""
    out = dict(DEFAULT_STATS_SETTINGS)
    if not advanced:
        return out
    unknown = set(advanced) - set(DEFAULT_STATS_SETTINGS)
    if unknown:
        raise ValueError(
            f"Unknown advanced keys: {sorted(unknown)}. Allowed: {sorted(DEFAULT_STATS_SETTINGS)}"
        )
    out.update(advanced)
    for key in ("trend", "robust"):
        if not isinstance(out[key], bool):
            raise ValueError(f"{key} must be bool, got {type(out[key]).__name__}")
    wt = out["winsor_tail_p"]
    if (
        not isinstance(wt, (tuple, list))
        or len(wt) != 2
        or not all(isinstance(x, numbers.Real) and not isinstance(x, bool) for x in wt)
        or not all(0 <= float(x) < 0.5 for x in wt)
    ):
        raise ValueError(f"winsor_tail_p must be a pair of floats in [0, 0.5), got {wt!r}")
    out["winsor_tail_p"] = (float(wt[0]), float(wt[1]))
    if out["robust"]:
        raise NotImplementedError(
            "advanced={'robust': True}: the Phipson 2016 robust prior is not available -- "
            "inmoose 0.9.1 raises inside squeezeVar and alphaPhos's clean-room stack does "
            "not implement it. Use the default (trend=True) or trend=False."
        )
    return out


def _validate_frame(
    adata: ad.AnnData,
    *,
    condition_column: str,
    covariates: list[str] | None,
    layer: str | None,
) -> None:
    """Checks shared by all differential-testing entry points."""
    if not adata.var_names.is_unique:
        raise ValueError(
            "adata.var_names must be unique; duplicates would silently break "
            "row alignment of the result table."
        )
    if condition_column not in adata.obs.columns:
        raise KeyError(
            f"condition_column={condition_column!r} not in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )
    n_nan = int(adata.obs[condition_column].isna().sum())
    if n_nan:
        raise ValueError(
            f"adata.obs[{condition_column!r}] has {n_nan} missing value(s); "
            "drop those samples or fill the label first."
        )
    for cov in covariates or ():
        if cov not in adata.obs.columns:
            raise KeyError(
                f"covariate={cov!r} not in adata.obs. Available: {list(adata.obs.columns)}"
            )
        if cov == condition_column:
            raise ValueError(f"covariate={cov!r} is the condition column; cannot self-adjust.")
    if layer is not None and layer not in adata.layers:
        raise KeyError(
            f"layer={layer!r} not in adata.layers. Available: {list(adata.layers.keys())}"
        )


def _validate_inputs(
    adata: ad.AnnData,
    *,
    condition_column: str,
    treatment: str,
    control: str,
    covariates: list[str] | None,
    layer: str | None,
) -> None:
    _validate_frame(adata, condition_column=condition_column, covariates=covariates, layer=layer)
    levels_present = set(adata.obs[condition_column].astype(str).unique())
    if treatment not in levels_present:
        raise ValueError(
            f"treatment={treatment!r} not in adata.obs[{condition_column!r}]. "
            f"Available: {sorted(levels_present)}"
        )
    if control not in levels_present:
        raise ValueError(
            f"control={control!r} not in adata.obs[{condition_column!r}]. "
            f"Available: {sorted(levels_present)}"
        )
    if treatment == control:
        raise ValueError("treatment and control must be different levels.")


def _get_matrix(adata: ad.AnnData, *, layer: str | None) -> np.ndarray:
    X = adata.X if layer is None else adata.layers[layer]
    return np.asarray(X, dtype=float)


def _validate_no_nan(X: np.ndarray, *, layer: str | None) -> None:
    if np.isnan(X).any():
        n_nan = int(np.isnan(X).sum())
        where = f"layer={layer!r}" if layer else "adata.X"
        raise ValueError(
            f"{n_nan} NaN values in {where}. limma cannot fit rows with missing values; "
            "run alphaphos.filter_by_completeness and alphaphos.impute_hybrid first."
        )


def _validate_log_scale(X: np.ndarray, *, layer: str | None) -> None:
    # Linear-scale MS intensities have medians in the millions; log2-scale
    # values sit in the 10-30 range. If we see medians well above the log2
    # ceiling, the caller almost certainly passed raw intensities.
    med = float(np.nanmedian(X))
    if med > LOG_SCALE_MEDIAN_CEILING:
        where = f"layer={layer!r}" if layer else "adata.X"
        raise ValueError(
            f"Data in {where} looks linear-scale (median={med:.3g}). "
            "limma expects log2-scale input. Pass "
            f"layer='{LAYER_INTENSITY_LOG2}' or log2-transform first."
        )


def _warn_on_small_groups(adata: ad.AnnData, condition_column: str, levels: list[str]) -> None:
    """Warn when any of ``levels`` has fewer than ``MIN_REPLICATES_WARNING`` samples."""
    counts = adata.obs[condition_column].astype(str).value_counts()
    small = {
        lv: int(counts.get(lv, 0)) for lv in levels if counts.get(lv, 0) < MIN_REPLICATES_WARNING
    }
    if small:
        logger.warning(
            "Small groups: %s. limma is designed for small n but n<%d gives unstable "
            "moderated statistics -- interpret with caution.",
            ", ".join(f"{lv}={n}" for lv, n in small.items()),
            MIN_REPLICATES_WARNING,
        )


# Level sanitisation lives in alphaphos.stats.design so the inmoose path and
# the clean-room path encode levels identically; these names are kept for
# backwards compatibility of the private helpers.
def _sanitize_level(label: str) -> str:
    return sanitize_and_map_levels([str(label)])[str(label)]


_sanitize_and_map_levels = sanitize_and_map_levels


def _build_design(
    adata: ad.AnnData,
    *,
    condition_column: str,
    treatment: str,
    control: str,
    covariates: list[str] | None,
):
    obs_df = adata.obs.copy()

    # Sanitize the condition column BEFORE it hits patsy, so patsy's dummy
    # column names (which later flow through inmoose.makeContrasts eval())
    # are guaranteed to be valid Python identifiers.  Handles level names
    # like "EGF+", "EGF-", "1uM", "KO/WT", "not treated" -- see
    # _sanitize_level for the rules.
    cond_str = obs_df[condition_column].astype(str)
    cond_map = _sanitize_and_map_levels(cond_str.unique().tolist())
    obs_df[condition_column] = cond_str.map(cond_map)
    safe_treatment = cond_map[str(treatment)]
    safe_control = cond_map[str(control)]

    # Same for any categorical covariate columns.  Continuous (float)
    # covariates flow through patsy as plain names; integer / bool columns
    # are rejected as ambiguous (see design.covariate_is_continuous).
    if covariates:
        for cov in covariates:
            col = obs_df[cov]
            if not covariate_is_continuous(col, name=cov):
                col_str = col.astype(str)
                cov_map = sanitize_and_map_levels(col_str.unique().tolist())
                obs_df[cov] = col_str.map(cov_map)

    formula_parts = [f"0 + {condition_column}"]
    if covariates:
        formula_parts.extend(covariates)
    formula = "~ " + " + ".join(formula_parts)
    import patsy  # lazy: [stats] extra

    design = patsy.dmatrix(formula, data=obs_df)
    # Patsy names columns as ``condition[<level>]`` for the no-intercept case.
    level_treat = f"{condition_column}[{safe_treatment}]"
    level_ctrl = f"{condition_column}[{safe_control}]"
    cols = list(design.design_info.column_names)
    if level_treat not in cols or level_ctrl not in cols:
        raise ValueError(
            f"Design matrix missing expected levels {level_treat!r}/{level_ctrl!r}. Got: {cols}"
        )
    rank = int(np.linalg.matrix_rank(np.asarray(design)))
    if rank < len(cols):
        raise ValueError(
            f"design matrix is rank-deficient (rank {rank} < {len(cols)} coefficients). "
            "Common cause: a covariate is perfectly confounded with the condition within "
            f"the compared samples. Coefficients: {cols}"
        )
    return design, {"treatment": level_treat, "control": level_ctrl}


def _safe_ebayes(fit2, settings: dict):
    # inmoose 0.9.1 has a bug when ``df_prior`` returns as a scalar
    # ``float('inf')`` (pathological homogeneous variance): the check
    # ``Infdf = df_prior > 1e6`` becomes a Python ``True`` and ``~True == -2``
    # then tries a DataFrame column lookup, raising ``KeyError: -2``. Real
    # biological data with heterogeneous per-site variance never produces
    # inf ``df_prior``. Translate the raw KeyError into a targeted message.
    from inmoose.limma import eBayes  # lazy: [stats] extra

    try:
        return eBayes(
            fit2,
            trend=settings["trend"],
            robust=settings["robust"],
            winsor_tail_p=settings["winsor_tail_p"],
        )
    except KeyError as exc:
        if str(exc) == "-2":
            raise RuntimeError(
                "inmoose 0.9.1 eBayes failed with a degenerate variance prior "
                "(df_prior = inf). This usually means residual variance is "
                "identical across features -- often synthetic data with "
                "identical noise. Real biology data with heterogeneous "
                "per-site variance should not hit this. If you see this on "
                "real data, add small jitter to X or use trend=True."
            ) from exc
        raise


def _finalize_result(
    tt: pd.DataFrame,
    *,
    treatment: str,
    control: str,
    contrast_string: str,
) -> pd.DataFrame:
    renamed = tt.rename(columns=INMOOSE_TOPTABLE_COLS)
    ordered_cols = ["log2fc", "se", "t_stat", "p_value", "fdr", "B", "ave_expr"]
    present = [c for c in ordered_cols if c in renamed.columns]
    result = renamed[present].copy()
    result.attrs["contrast_string"] = contrast_string
    result.attrs["contrast_direction"] = (
        f"log2fc = mean({treatment}) - mean({control}); positive = up in {treatment!r}"
    )
    result.attrs["treatment"] = treatment
    result.attrs["control"] = control
    return result


# =============================================================================
# Multi-contrast + ANOVA (H3 from the PhosPy borrowings audit).
# =============================================================================


def diff_exp_limma_contrasts(
    adata: ad.AnnData,
    *,
    condition_column: str,
    contrasts: list[tuple[str, str]] | dict[str, tuple[str, str]],
    covariates: list[str] | None = None,
    block_column: str | None = None,
    layer: str | None = LAYER_INTENSITY_LOG2,
    joint: bool = True,
    advanced: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """Multi-contrast moderated t-tests on one ``AnnData``.

    Parameters
    ----------
    adata, condition_column, covariates, layer
        As in :func:`diff_exp_limma`.
    advanced
        As in :func:`diff_exp_limma`.  With ``joint=True`` only ``trend``
        is meaningful (``robust`` / ``winsor_tail_p`` belong to the inmoose
        backend and raise).
    contrasts
        Either a list of ``(treatment, control)`` tuples (keys auto-
        generated as ``f"{treatment}_vs_{control}"``) or a dict
        ``{name: (treatment, control)}``.
    block_column
        Optional paired-block factor (e.g. ``"patient"``) added to the
        design as fixed effects.  Only used when ``joint=True``.
    joint
        - ``True`` (default): fit **one** linear model across all samples
          with all condition levels + covariates + block, then test each
          contrast on that joint fit.  Uses alphaPhos's own moderation
          (:mod:`alphaphos.stats.linear_model`) -- proper cross-feature
          variance shrinkage jointly across all contrasts.  Independent
          of inmoose.
        - ``False``: fall back to a loop of :func:`diff_exp_limma` calls,
          each on the 2-group subset for its contrast.  Uses inmoose per
          contrast, so ``advanced`` overrides apply.  Per-pair variance
          estimation (no joint moderation).

    Returns
    -------
    dict[str, pandas.DataFrame]
        One entry per contrast, insertion order preserved.  Columns::

            log2fc / estimate  se  t_stat  p_value  fdr  ave_expr [ df_moderated ]

        ``joint=True`` results carry a ``df_moderated`` column; the
        ``.attrs["prior_variance"]`` / ``prior_df`` fields record the
        joint EB prior estimated across all features.  ``joint=False``
        results match :func:`diff_exp_limma`'s columns exactly (log2fc,
        se, t_stat, p_value, fdr, B, ave_expr).
    """
    if isinstance(contrasts, list):
        contrasts = {f"{t}_vs_{c}": (t, c) for t, c in contrasts}
    if not contrasts:
        raise ValueError("contrasts must be non-empty")

    if not joint:
        # Per-pair loop.  advanced kwargs pass through to diff_exp_limma.
        _require_stats_deps("diff_exp_limma_contrasts(joint=False)")
        if block_column is not None:
            raise ValueError(
                "block_column is only supported when joint=True.  For "
                "joint=False, encode the block factor as a categorical "
                "covariate via covariates=..."
            )
        results: dict[str, pd.DataFrame] = {}
        for name, pair in contrasts.items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(f"contrasts[{name!r}] must be a (treatment, control) tuple")
            treatment, control = pair
            results[name] = diff_exp_limma(
                adata,
                condition_column=condition_column,
                comparison=(treatment, control),
                covariates=covariates,
                layer=layer,
                advanced=advanced,
            )
        return results

    # ----- joint=True path (default): use our own moderated stack.
    settings = _resolve_joint_settings(advanced)
    _validate_frame(adata, condition_column=condition_column, covariates=covariates, layer=layer)
    if block_column is not None and block_column not in adata.obs.columns:
        raise KeyError(f"block_column {block_column!r} not in adata.obs")
    _refuse_double_batch_correction(adata, covariates=covariates, layer=layer)

    from alphaphos.stats.design import design_matrix
    from alphaphos.stats.linear_model import (
        contrasts_fit,
        fit_prior,
        lm_fit,
        moderated_t_test,
    )

    # Validate the requested contrasts against the observed levels.
    obs_levels = set(adata.obs[condition_column].astype(str).unique())
    for name, pair in contrasts.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"contrasts[{name!r}] must be a (treatment, control) tuple")
        for lv in pair:
            if str(lv) not in obs_levels:
                raise ValueError(
                    f"contrasts[{name!r}] references level {lv!r} which is "
                    f"not in adata.obs[{condition_column!r}] "
                    f"(available: {sorted(obs_levels)})"
                )
    _warn_on_small_groups(
        adata, condition_column, sorted({str(lv) for pair in contrasts.values() for lv in pair})
    )

    X = _get_matrix(adata, layer=layer)
    _validate_no_nan(X, layer=layer)
    _validate_log_scale(X, layer=layer)

    dm = design_matrix(
        adata,
        condition_column=condition_column,
        covariates=covariates,
        block_column=block_column,
    )
    fit = lm_fit(X, dm.frame.to_numpy(), coefficient_labels=dm.coefficient_labels)
    # Fit the EB prior once (constant or intensity trend), reused across all contrasts.
    prior = fit_prior(fit, trend=settings["trend"])

    # Build one column per contrast.  Condition-column encoding is
    # ``condition[<sanitized_level>]`` -- map raw level -> sanitized.
    coef_index = {label: i for i, label in enumerate(dm.coefficient_labels)}
    C = np.zeros((len(dm.coefficient_labels), len(contrasts)), dtype=np.float64)
    for j, (_name, (trt, ctrl)) in enumerate(contrasts.items()):
        trt_safe = dm.level_sanitization[str(trt)]
        ctrl_safe = dm.level_sanitization[str(ctrl)]
        trt_coef = f"{condition_column}[{trt_safe}]"
        ctrl_coef = f"{condition_column}[{ctrl_safe}]"
        C[coef_index[trt_coef], j] = 1.0
        C[coef_index[ctrl_coef], j] = -1.0

    cf = contrasts_fit(fit, C, contrast_labels=list(contrasts.keys()))
    raw_results = moderated_t_test(
        cf,
        var_names=list(adata.var_names),
        prior=prior,
    )

    # Normalise column names to match diff_exp_limma's convention (log2fc,
    # ave_expr) so downstream code that consumes both is uniform.
    ave_expr = X.mean(axis=0)
    finalized: dict[str, pd.DataFrame] = {}
    for name, r in raw_results.items():
        trt, ctrl = contrasts[name]
        r = r.rename(columns={"estimate": "log2fc"})
        r["ave_expr"] = ave_expr
        r.attrs["contrast_string"] = f"{condition_column}[{trt}]-{condition_column}[{ctrl}]"
        r.attrs["contrast_direction"] = (
            f"log2fc = mean({trt}) - mean({ctrl}); positive = up in {trt!r}"
        )
        r.attrs["treatment"] = trt
        r.attrs["control"] = ctrl
        r.attrs["prior_variance"] = prior.prior_variance
        r.attrs["prior_df"] = prior.prior_df
        r.attrs["trend"] = prior.trend
        finalized[name] = r
    return finalized


def _resolve_joint_settings(advanced: dict | None) -> dict:
    """Settings for the clean-room path: full validation, but only ``trend`` may be set."""
    settings = _resolve_stats_settings(advanced)
    extra = set(advanced or ()) - _JOINT_PATH_KEYS
    if extra:
        raise ValueError(
            f"advanced keys {sorted(extra)} belong to the inmoose eBayes backend and have no "
            "effect on the clean-room fit (joint=True / diff_exp_anova). Only 'trend' applies "
            "here; pass joint=False to use the others."
        )
    return settings


def diff_exp_anova(
    adata: ad.AnnData,
    *,
    condition_column: str,
    covariates: list[str] | None = None,
    layer: str | None = LAYER_INTENSITY_LOG2,
    advanced: dict | None = None,
) -> pd.DataFrame:
    """Moderated F-test per feature across all condition levels (ANOVA-style).

    ``advanced={"trend": bool}`` selects the limma-trend (default) or the
    constant prior; other keys raise (they belong to the inmoose backend).

    Answers **"does the mean differ anywhere across the K levels of
    ``condition_column``?"** as a single moderated F-statistic per
    feature.  Complements :func:`diff_exp_limma` (pairwise) and
    :func:`diff_exp_limma_contrasts` (multiple named pairwise).

    Uses alphaPhos's own limma-style empirical-Bayes moderation
    (Smyth 2004 fitFDist / moderated F-statistic; see
    :mod:`alphaphos.stats.linear_model`).  Independent of inmoose --
    fully working on multi-condition designs.

    See :func:`alphaphos.stats.linear_model.moderated_f_test` for the
    full docstring; this function is a thin wrapper that builds the
    design + contrast matrices from an ``AnnData`` + ``condition_column``.

    Downstream compatibility
    ------------------------
    ANOVA output is **unsigned** (F ≥ 0, no direction).  So the
    downstream analyses that make sense on it are ORA-style
    (direction-agnostic), not signed:

    - **Site ORA** (:func:`alphaphos.enrichment.ora`): supported
      natively via :func:`alphaphos.anova_hits` -- ``hits, bg =
      ap.anova_hits(anova); ap.enrichment.ora(hits, bg, libraries)``.
    - **Gene pathway ORA** (:func:`alphaphos.enrichment.pathway_enrichment`):
      pass ``direction="any"`` -- ``ap.enrichment.pathway_enrichment(anova,
      direction="any", background=[...])``.  Works identically for
      phospho (default key parser) and proteome (with ``gene_column=``).

    Signed methods (:func:`alphaphos.enrichment.gsea`,
    :func:`alphaphos.enrichment.pathway_gsea`,
    :func:`alphaphos.enrichment.kinase_activity`) **cannot run on ANOVA
    output** -- they need a per-feature direction (up vs down).  Re-run
    per-contrast with :func:`diff_exp_limma_contrasts` and hand each
    resulting DataFrame to those consumers.
    """
    # Deferred to avoid a circular import at module load.
    from alphaphos.stats.design import design_matrix
    from alphaphos.stats.linear_model import lm_fit, moderated_f_test

    settings = _resolve_joint_settings(advanced)
    _validate_frame(adata, condition_column=condition_column, covariates=covariates, layer=layer)
    _refuse_double_batch_correction(adata, covariates=covariates, layer=layer)
    obs = adata.obs
    levels = sorted(obs[condition_column].astype(str).unique())
    if len(levels) < 2:
        raise ValueError(f"condition_column {condition_column!r} needs >=2 levels; got {levels}")
    _warn_on_small_groups(adata, condition_column, levels)

    X = _get_matrix(adata, layer=layer)
    _validate_no_nan(X, layer=layer)
    _validate_log_scale(X, layer=layer)

    dm = design_matrix(
        adata,
        condition_column=condition_column,
        covariates=covariates,
    )
    fit = lm_fit(X, dm.frame.to_numpy(), coefficient_labels=dm.coefficient_labels)

    # Contrast matrix: every non-reference condition minus the reference.
    # Joint F across these K-1 contrasts is the ANOVA "any differs" test.
    ref_coef = dm.condition_coefficients[0]
    other_coefs = dm.condition_coefficients[1:]
    C = np.zeros((len(dm.coefficient_labels), len(other_coefs)))
    ref_idx = dm.coefficient_labels.index(ref_coef)
    for j, other in enumerate(other_coefs):
        idx = dm.coefficient_labels.index(other)
        C[idx, j] = 1.0
        C[ref_idx, j] = -1.0

    result = moderated_f_test(
        fit, contrasts=C, var_names=list(adata.var_names), trend=settings["trend"]
    )
    result.attrs["condition_column"] = condition_column
    result.attrs["condition_levels"] = tuple(levels)
    result.attrs["reference_level"] = dm.reference_level
    result.attrs["covariates"] = tuple(covariates or ())
    return result


def anova_hits(
    anova_result: pd.DataFrame,
    *,
    fdr_threshold: float = 0.05,
    fdr_col: str = "fdr",
) -> tuple[list[str], list[str]]:
    """Split an ANOVA / F-test result into ``(hits, background)`` for ORA.

    Two-line convenience over :func:`alphaphos.enrichment.ora` for the
    common "which pathways / site-sets are enriched among ANOVA hits?"
    question:

    >>> anova = ap.diff_exp_anova(adata, condition_column="disease")
    >>> hits, bg = ap.anova_hits(anova, fdr_threshold=0.05)
    >>> ap.enrichment.ora(hits, bg, libraries)

    Parameters
    ----------
    anova_result
        Output of :func:`diff_exp_anova` -- any DataFrame with an
        ``fdr`` column also works.
    fdr_threshold
        Rows with FDR strictly less than this are hits.  Default 0.05.
    fdr_col
        Name of the FDR column.  Default ``"fdr"``.

    Returns
    -------
    hits, background : list[str], list[str]
        Row labels for the two ORA arms.  Both suitable as direct inputs
        to :func:`alphaphos.enrichment.ora`.  Rows with NaN FDR are
        dropped from **both** lists -- they were not tested and do not
        belong in either arm of the Fisher table.
    """
    if fdr_col not in anova_result.columns:
        raise KeyError(
            f"fdr_col {fdr_col!r} not in anova_result columns; "
            f"available: {list(anova_result.columns)}"
        )
    fdr = anova_result[fdr_col]
    tested = fdr.notna()
    tested_index = anova_result.index[tested]
    hits_index = anova_result.index[tested & (fdr < fdr_threshold)]
    return [str(x) for x in hits_index], [str(x) for x in tested_index]
