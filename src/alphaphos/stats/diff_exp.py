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


# patsy + inmoose are gated behind the [stats] extra. Everything in this
# module still imports cleanly without them; diff_exp_limma raises
# ImportError with an install hint when called.
try:
    import patsy  # type: ignore[import-not-found]
    from inmoose.limma import (  # type: ignore[import-not-found]
        contrasts_fit,
        eBayes,
        lmFit,
        makeContrasts,
        topTable,
    )

    _HAS_STATS_DEPS = True
except ImportError:  # pragma: no cover
    _HAS_STATS_DEPS = False


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
    if not _HAS_STATS_DEPS:
        raise ImportError(
            "diff_exp_limma requires inmoose and patsy. Install with "
            "`pip install alphaPhos[stats]`."
        )

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

    fit = lmFit(X.T, design)
    contrast_mat = makeContrasts([contrast_string_internal], levels=list(fit.coefficients.columns))
    fit2 = contrasts_fit(fit, contrast_mat)
    fit2 = _safe_ebayes(fit2, settings)

    contrast_name = fit2.coefficients.columns[0]
    tt = pd.DataFrame(topTable(fit2, coef=contrast_name, number=np.inf)).sort_index()
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
