"""Observed-only differential expression + on/off detection.

Guards against a well-known failure mode of impute-then-limma on small
groups: when hybrid imputation fills a whole comparison group with the
MNAR down-shift (a near-constant, tiny-variance value), the moderated
t-test reports an inflated fold-change and over-confident FDR for
every one of those "on/off" features -- entirely driven by invented
numbers.

Three tools, composable + also wrapped in one call:

- :func:`on_off_detection` -- report per-feature detection call
  (``on_in_treatment``, ``on_in_control``, ``both``, ``absent``) with the
  raw counts.  No p-values; on/off is a **detection** result, not a
  statistical test.
- :func:`annotate_imputed_provenance` -- add per-hit observation counts
  and an ``imputed_driven`` flag to an existing ``diff_exp_limma``
  result.  Users then either filter or clearly mark those rows.
- :func:`diff_exp_limma_observed_only` -- headline convenience: on/off
  detect + strict per-group completeness filter + hybrid impute (which
  now only fills sparse cells, not whole groups) + limma + provenance.
  Returns ``(limma_result, on_off_table)``.

The pattern matches the community convention (MSstats, Perseus's
present/absent call, DEqMS) of separating **detection** (feature seen
in one group, not the other) from **quantification** (fold-change and
p-value between two well-observed groups).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


DetectionCall = Literal["on_in_treatment", "on_in_control", "both", "absent"]


def on_off_detection(
    adata: ad.AnnData,
    *,
    condition_column: str,
    comparison: tuple[str, str],
    min_observed_per_group: int = 3,
    layer: str | None = "intensity_log2",
) -> pd.DataFrame:
    """Per-feature detection call for a two-group contrast.

    A feature is:

    - ``both``            -- observed in >= ``min_observed_per_group`` samples
                            of BOTH groups.  Eligible for limma.
    - ``on_in_treatment`` -- observed in >= threshold of ``treatment`` and
                            < threshold in ``control``.  "Up" detection.
    - ``on_in_control``   -- observed in >= threshold of ``control`` and
                            < threshold in ``treatment``.  "Down" detection.
    - ``absent``          -- fails threshold in both groups.  Neither
                            testable nor detected reliably in either arm.

    Parameters
    ----------
    adata
        AnnData ``(n_samples, n_sites)``.  Uses ``layer`` (default
        ``"intensity_log2"``) to determine missingness; pass ``None`` to
        use ``adata.X``.  **Must be pre-imputation** -- the whole point
        is to count observed vs missing before any imputation fills in.
    condition_column
        Column in ``adata.obs`` that carries the group labels.
    comparison
        ``(treatment, control)`` tuple.  Same convention as
        :func:`diff_exp_limma`; positive log2FC / "on_in_treatment"
        means up in ``treatment``.
    min_observed_per_group
        Threshold count for calling a feature "present" in a group.
        Default 3 (matches the reviewer feedback that motivated this
        module).
    layer
        Which layer's missingness to use.  Default ``"intensity_log2"``.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``adata.var_names``, with columns:

        - ``n_observed_treatment`` : int
        - ``n_observed_control`` : int
        - ``call`` : one of ``{"both", "on_in_treatment", "on_in_control", "absent"}``

        ``.attrs["provenance"]`` records the contrast and threshold used.
    """
    treatment, control = comparison
    _validate_contrast(adata, condition_column, treatment, control)

    X = _get_layer(adata, layer=layer)
    obs_col = adata.obs[condition_column].astype(str).to_numpy()

    treat_mask = obs_col == treatment
    ctrl_mask = obs_col == control
    if not treat_mask.any() or not ctrl_mask.any():
        raise ValueError(
            f"Contrast has 0 samples on one side: n_treatment={int(treat_mask.sum())}, "
            f"n_control={int(ctrl_mask.sum())}."
        )

    not_nan = ~np.isnan(X)
    n_treat = not_nan[treat_mask].sum(axis=0).astype(int)
    n_ctrl = not_nan[ctrl_mask].sum(axis=0).astype(int)

    ok_treat = n_treat >= min_observed_per_group
    ok_ctrl = n_ctrl >= min_observed_per_group
    call = np.full(adata.n_vars, "absent", dtype=object)
    call[ok_treat & ok_ctrl] = "both"
    call[ok_treat & ~ok_ctrl] = "on_in_treatment"
    call[~ok_treat & ok_ctrl] = "on_in_control"

    df = pd.DataFrame(
        {
            "n_observed_treatment": n_treat,
            "n_observed_control": n_ctrl,
            "call": call,
        },
        index=adata.var_names.copy(),
    )
    df.attrs["provenance"] = {
        "condition_column": condition_column,
        "treatment": treatment,
        "control": control,
        "min_observed_per_group": int(min_observed_per_group),
        "n_samples_treatment": int(treat_mask.sum()),
        "n_samples_control": int(ctrl_mask.sum()),
        "layer": layer,
        "n_both": int((call == "both").sum()),
        "n_on_in_treatment": int((call == "on_in_treatment").sum()),
        "n_on_in_control": int((call == "on_in_control").sum()),
        "n_absent": int((call == "absent").sum()),
    }
    logger.info(
        "on_off_detection: %s vs %s (min_obs=%d) -> both=%d, on_treatment=%d, "
        "on_control=%d, absent=%d",
        treatment,
        control,
        min_observed_per_group,
        df.attrs["provenance"]["n_both"],
        df.attrs["provenance"]["n_on_in_treatment"],
        df.attrs["provenance"]["n_on_in_control"],
        df.attrs["provenance"]["n_absent"],
    )
    return df


def annotate_imputed_provenance(
    diff_exp_result: pd.DataFrame,
    adata_pre_imputation: ad.AnnData,
    *,
    condition_column: str,
    comparison: tuple[str, str],
    min_observed_per_group: int = 3,
    layer: str | None = "intensity_log2",
) -> pd.DataFrame:
    """Attach observation counts + an ``imputed_driven`` flag to a limma result.

    Adds columns to ``diff_exp_result`` (returns a copy):

    - ``n_observed_treatment`` : int
    - ``n_observed_control`` : int
    - ``n_missing_treatment`` : int
    - ``n_missing_control`` : int
    - ``imputed_driven`` : bool -- True if EITHER group has fewer than
      ``min_observed_per_group`` observed values.  A True here means the
      hit's fold-change and p-value are (at least partly) driven by
      imputed numbers rather than measured evidence.

    The typical use is: run the standard impute → limma flow, then call
    this to mark which rows are trustworthy.  Users filter
    ``imputed_driven == False`` for their headline table.

    Parameters
    ----------
    diff_exp_result
        Output of :func:`diff_exp_limma` (or ``diff_exp_limma_contrasts``
        for a single contrast).  Indexed by feature; index must be a
        subset of ``adata_pre_imputation.var_names``.
    adata_pre_imputation
        The AnnData used to fit limma **before** imputation replaced any
        NaN values.  Missingness in ``layer`` (default
        ``"intensity_log2"``) drives the counts.
    condition_column, comparison, min_observed_per_group, layer
        Same semantics as :func:`on_off_detection`.
    """
    treatment, control = comparison
    _validate_contrast(adata_pre_imputation, condition_column, treatment, control)

    missing_in_var = set(diff_exp_result.index) - set(adata_pre_imputation.var_names)
    if missing_in_var:
        raise KeyError(
            f"{len(missing_in_var)} rows of diff_exp_result are not in "
            f"adata_pre_imputation.var_names (e.g. {sorted(missing_in_var)[:3]}). "
            "Pass the pre-imputation AnnData that was collapsed/filtered "
            "identically to the one limma tested."
        )

    var_index = adata_pre_imputation.var_names.get_indexer(diff_exp_result.index)
    X = _get_layer(adata_pre_imputation, layer=layer)
    obs_col = adata_pre_imputation.obs[condition_column].astype(str).to_numpy()

    treat_mask = obs_col == treatment
    ctrl_mask = obs_col == control
    n_treat_samples = int(treat_mask.sum())
    n_ctrl_samples = int(ctrl_mask.sum())

    not_nan = ~np.isnan(X[:, var_index])
    n_treat_obs = not_nan[treat_mask].sum(axis=0).astype(int)
    n_ctrl_obs = not_nan[ctrl_mask].sum(axis=0).astype(int)
    n_treat_missing = n_treat_samples - n_treat_obs
    n_ctrl_missing = n_ctrl_samples - n_ctrl_obs

    out = diff_exp_result.copy()
    out["n_observed_treatment"] = n_treat_obs
    out["n_observed_control"] = n_ctrl_obs
    out["n_missing_treatment"] = n_treat_missing
    out["n_missing_control"] = n_ctrl_missing
    out["imputed_driven"] = (n_treat_obs < min_observed_per_group) | (
        n_ctrl_obs < min_observed_per_group
    )

    prov = dict(out.attrs.get("provenance", {}))
    prov.update(
        {
            "imputed_provenance_min_observed_per_group": int(min_observed_per_group),
            "imputed_provenance_layer": layer,
            "n_imputed_driven": int(out["imputed_driven"].sum()),
        }
    )
    out.attrs["provenance"] = prov
    logger.info(
        "annotate_imputed_provenance: %d / %d rows flagged imputed_driven "
        "(min_observed_per_group=%d)",
        int(out["imputed_driven"].sum()),
        len(out),
        min_observed_per_group,
    )
    return out


def diff_exp_limma_observed_only(
    adata: ad.AnnData,
    *,
    condition_column: str,
    comparison: tuple[str, str],
    min_observed_per_group: int = 3,
    covariates: list[str] | None = None,
    layer: str | None = "intensity_log2",
    imputer: str | None = "hybrid",
    imputer_kwargs: dict | None = None,
    limma_advanced: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Headline: run limma only on features observed in both comparison groups.

    The default alphaPhos flow (impute the entire matrix, then run limma)
    inflates the differential-hit count on the "down" side when a whole
    group's values for a feature are absent and get filled with the MNAR
    down-shift.  This function guards against that by:

    1. Computing the on/off table on the pre-imputation AnnData (no NaN
       filled in yet).
    2. Filtering to features where BOTH groups have >=
       ``min_observed_per_group`` observed values.  Features present in
       only one group are moved to the on/off table, NOT tested by
       limma.
    3. Imputing the filtered subset with hybrid imputation (which now
       only fills sparse cells within an otherwise-observed group; a
       whole group cannot be imputed because the filter removed those
       features).
    4. Running :func:`diff_exp_limma` on the strict-filtered, imputed
       matrix.
    5. Annotating each result row with observation counts and an
       ``imputed_driven`` flag (kept for transparency; should always be
       False by construction after the strict filter, but exposed for
       audit).

    Parameters
    ----------
    adata
        AnnData ``(n_samples, n_sites)`` **before** imputation.  Must
        carry ``layer`` (default ``"intensity_log2"``) with NaN in
        missing cells.
    condition_column, comparison
        As in :func:`diff_exp_limma`.
    min_observed_per_group
        Minimum observed count per group for a feature to be tested by
        limma.  Default 3.  Features below this in either group are
        moved to the on/off detection table.
    covariates
        Passed through to :func:`diff_exp_limma`.
    layer
        The log2 layer to test.  Default ``"intensity_log2"``.
    imputer
        Which imputer to run on the filtered subset.  ``"hybrid"``
        (default) calls :func:`alphaphos.impute_hybrid`.  ``None`` skips
        imputation -- users who prefer a truly imputation-free run
        should pass ``None`` and expect ``diff_exp_limma`` to still
        error on any residual NaN (rare on strict-filtered data).
    imputer_kwargs
        Passed through to the imputer.
    limma_advanced
        Passed to :func:`diff_exp_limma` as ``advanced=``.

    Returns
    -------
    (limma_result, on_off_table)
        ``limma_result`` : same shape as :func:`diff_exp_limma`'s output,
        with the extra provenance columns from
        :func:`annotate_imputed_provenance`.  Index restricted to
        features with ``call == "both"`` in the on/off table.

        ``on_off_table`` : as returned by :func:`on_off_detection`.
        Its rows with ``call in ("on_in_treatment", "on_in_control")``
        are the detection-only hits -- reported alongside limma, not as
        p-values.
    """
    from alphaphos.stats.diff_exp import diff_exp_limma

    on_off = on_off_detection(
        adata,
        condition_column=condition_column,
        comparison=comparison,
        min_observed_per_group=min_observed_per_group,
        layer=layer,
    )
    testable = on_off.index[on_off["call"] == "both"]
    if len(testable) == 0:
        raise ValueError(
            "No features pass strict per-group completeness (call == 'both'); "
            "loosen min_observed_per_group or check the input."
        )

    sub = adata[:, testable].copy()
    if imputer is not None:
        from alphaphos.preprocess.impute import impute_hybrid, impute_knn_site_based

        imputer_kwargs = dict(imputer_kwargs or {})
        if imputer == "hybrid":
            sub = impute_hybrid(sub, **imputer_kwargs)
        elif imputer == "knn":
            sub = impute_knn_site_based(sub, **imputer_kwargs)
        else:
            raise ValueError(f"imputer={imputer!r} not recognised; use 'hybrid', 'knn', or None")
        # impute_* rewrites the layer; also propagate to .X for limma default.
        if layer is not None and layer in sub.layers:
            sub.X = sub.layers[layer].copy()

    result = diff_exp_limma(
        sub,
        condition_column=condition_column,
        comparison=comparison,
        covariates=covariates,
        layer=layer,
        advanced=limma_advanced,
    )
    result = annotate_imputed_provenance(
        result,
        adata_pre_imputation=adata,
        condition_column=condition_column,
        comparison=comparison,
        min_observed_per_group=min_observed_per_group,
        layer=layer,
    )
    return result, on_off


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _validate_contrast(
    adata: ad.AnnData, condition_column: str, treatment: str, control: str
) -> None:
    if condition_column not in adata.obs.columns:
        raise KeyError(
            f"condition_column={condition_column!r} not in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )
    if treatment == control:
        raise ValueError("treatment and control must be different levels.")
    levels = set(adata.obs[condition_column].astype(str).unique())
    missing = [x for x in (treatment, control) if x not in levels]
    if missing:
        raise ValueError(
            f"Level(s) {missing} not in adata.obs[{condition_column!r}]. "
            f"Available: {sorted(levels)}"
        )


def _get_layer(adata: ad.AnnData, *, layer: str | None) -> np.ndarray:
    if layer is None:
        X = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(
                f"layer={layer!r} not in adata.layers. Available: {list(adata.layers.keys())}"
            )
        X = adata.layers[layer]
    return np.asarray(X, dtype=float)
