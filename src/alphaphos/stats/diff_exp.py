"""Two-group moderated t-test via limma (inmoose backend).

Implements a single pairwise comparison. ANOVA / multi-contrast F-tests
are intentionally NOT included in this module (see 0.5.0 changelog):
they will land later once inmoose's multi-contrast eBayes has been
regression-tested against R limma on real data.

Result columns:

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
import re
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
    "trend": False,
    "robust": False,
    "winsor_tail_p": (0.05, 0.1),
}

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

        - ``trend`` (bool): pass to ``eBayes``; enables mean-variance
          trend fitting.
        - ``robust`` (bool): pass to ``eBayes``; enables the
          Phipson-2016 robust EB estimator.
        - ``winsor_tail_p`` ((float, float)): passed to ``eBayes``.

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

    treatment, control = comparison
    _validate_inputs(
        adata,
        condition_column=condition_column,
        treatment=treatment,
        control=control,
        covariates=covariates,
        layer=layer,
    )
    _refuse_double_batch_correction(adata, covariates=covariates, layer=layer)

    sample_mask = adata.obs[condition_column].isin([treatment, control]).to_numpy()
    if not sample_mask.any():
        raise ValueError(
            f"No samples found with {condition_column} in {{{treatment!r}, {control!r}}}"
        )
    sub = adata[sample_mask, :]

    _warn_on_small_groups(sub, condition_column, treatment, control)

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
    if corrected_layer != layer:
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
    return out


def _validate_inputs(
    adata: ad.AnnData,
    *,
    condition_column: str,
    treatment: str,
    control: str,
    covariates: list[str] | None,
    layer: str | None,
) -> None:
    if not adata.var_names.is_unique:
        raise ValueError(
            "adata.var_names must be unique; duplicates would silently break "
            "row alignment after sorting the limma output."
        )
    if condition_column not in adata.obs.columns:
        raise KeyError(
            f"condition_column={condition_column!r} not in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )
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


def _warn_on_small_groups(
    adata: ad.AnnData, condition_column: str, treatment: str, control: str
) -> None:
    counts = adata.obs[condition_column].value_counts()
    n_treat = int(counts.get(treatment, 0))
    n_ctrl = int(counts.get(control, 0))
    if n_treat < MIN_REPLICATES_WARNING or n_ctrl < MIN_REPLICATES_WARNING:
        logger.warning(
            "Small groups: %s=%d, %s=%d. limma is designed for small n but n<%d "
            "gives unstable moderated statistics -- interpret with caution.",
            treatment,
            n_treat,
            control,
            n_ctrl,
            MIN_REPLICATES_WARNING,
        )


_UNSAFE_LEVEL_CHAR_RE = re.compile(r"[^A-Za-z0-9_]")


def _sanitize_level(label: str) -> str:
    """Map a categorical level to a Python-identifier-safe token.

    Levels flow through patsy dummy column names (``condition[<level>]``)
    and then through ``inmoose.limma.makeContrasts``, which evaluates the
    contrast string via ``eval()``.  Any non-identifier character
    (``+``, ``-``, ``.``, ``/``, space, leading digit, ...) is a
    ``SyntaxError`` waiting to happen.  Replaces unsafe characters with
    ``_``, prefixes an ``_`` when the result starts with a digit, and
    falls back to ``_`` for an empty string.
    """
    safe = _UNSAFE_LEVEL_CHAR_RE.sub("_", str(label))
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe or "_"


def _sanitize_and_map_levels(labels: list[str]) -> dict[str, str]:
    """Build a bijective ``original -> safe`` mapping over the given labels.

    Collisions (``EGF+`` and ``EGF-`` both sanitize to ``EGF_``) are broken
    by suffixing ``_2``, ``_3``, ... in first-seen order.  The mapping is
    an implementation detail; the sanitized names never leak to the user.
    """
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for orig in labels:
        base = _sanitize_level(orig)
        safe = base
        i = 2
        while safe in used:
            safe = f"{base}_{i}"
            i += 1
        mapping[str(orig)] = safe
        used.add(safe)
    return mapping


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

    # Same for any categorical covariate columns.  Continuous (numeric)
    # covariates flow through patsy as plain names and are unaffected.
    if covariates:
        for cov in covariates:
            col = obs_df[cov]
            if col.dtype == object or isinstance(col.dtype, pd.CategoricalDtype):
                col_str = col.astype(str)
                cov_map = _sanitize_and_map_levels(col_str.unique().tolist())
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
    from alphaphos.stats.design import design_matrix
    from alphaphos.stats.linear_model import (
        contrasts_fit,
        fit_f_dist,
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
    # Fit the EB prior once, reused across all contrasts.
    prior = fit_f_dist(fit.sigma_sq, residual_df=fit.df_residual)

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
        finalized[name] = r
    return finalized


def diff_exp_anova(
    adata: ad.AnnData,
    *,
    condition_column: str,
    covariates: list[str] | None = None,
    layer: str | None = LAYER_INTENSITY_LOG2,
) -> pd.DataFrame:
    """Moderated F-test per feature across all condition levels (ANOVA-style).

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

    obs = adata.obs
    if condition_column not in obs.columns:
        raise KeyError(f"condition_column {condition_column!r} not in adata.obs")
    if obs[condition_column].isna().any():
        raise ValueError(f"adata.obs[{condition_column!r}] has NaN; drop or impute first.")
    levels = sorted(obs[condition_column].astype(str).unique())
    if len(levels) < 2:
        raise ValueError(f"condition_column {condition_column!r} needs >=2 levels; got {levels}")

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

    result = moderated_f_test(fit, contrasts=C, var_names=list(adata.var_names))
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
