"""Advisory decision tree for filter + impute pipeline choices.

Reads an ``AnnData`` (typically the output of :func:`collapse_sites`) plus a
stated analytical goal, and prints a copy-paste-ready pipeline recipe:
Class-I cutoff, per-cell design audit, completeness filter parameters,
whether to impute, which imputer, and which DE call.  A single cheap
filter dry-run at the end shows what the recipe would leave behind so
the caller can adjust parameters before committing.

Executes nothing beyond that dry-run.  No AnnData is modified; the
downstream pipeline is the caller's to run.

Design decisions encoded here
-----------------------------
* Class-I cutoff defaults to 0.75 (kinase-substrate / pathway work);
  0.90 for site-level publication claims.  Skipped entirely for
  ``data_type="proteome"``.
* Any per-cell group with n < 3 samples is dropped before filtering;
  keeps ``keep_strategy="each"`` viable for interaction analyses.
* ``min_valid_n = max(3, min(10, round(0.6 * smallest_cell)))``.
  Capped at 10 because beyond ~10 observations per group the test is
  well-powered; higher just drops good sites without adding rigour.
* Imputation skipped when the DE method handles NaN natively (limma
  observed-only, msqrob2, on/off detection) AND post-filter NaN < 60%.
* Imputer selection: KNN for n < 50 (below PIMMS's paper floor);
  PIMMS-DAE for n >= 300 (10 s vs ~25 min for KNN); PIMMS-DAE when
  post-filter NaN >= 40%; KNN otherwise.

For the empirical basis of these thresholds see
``scratchpad/sfphospho_grouping_grid.py`` (10-config grid over
sfPhospho 304-fiber cohort) and ``scratchpad/imputation_benchmark_*``
(EGF / cardio / bulk phospho / SF phospho / proteome MAE comparisons).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from alphaphos.preprocess.filter import filter_by_completeness

if TYPE_CHECKING:
    import anndata as ad


Goal = Literal[
    "primary_de",
    "marginal_de",
    "interaction_de",
    "onoff_discovery",
    "profiling",
    "viz_only",
]
DataType = Literal["proteome", "phospho", "other_ptm"]

MIN_CELL_N = 3
LARGE_COHORT = 300
PIMMS_FLOOR = 50
NAN_HEAVY = 0.40
N_CAP = 10
CLASSI_DEFAULT = 0.75

# Above this cohort size we prescribe the Wilson lower-bound filter on the
# per-site Class-I fraction instead of the simple mean_loc_prob cutoff.
# Empirical basis: sfPhospho/uPhosHT retention curves — Wilson clearly beats
# naive filters at n >= 100 and is the demonstrably-correct default at n >= 300.
WILSON_MIN_N = 100


def _decide_classI(data_type: DataType, adata: ad.AnnData) -> tuple[str, float | str | None, str]:
    """Return ``(kind, threshold, why)``.

    - kind == "none"           : threshold is ``None``; no Class-I filter
    - kind == "mean_loc_prob"  : threshold is a float (e.g. 0.75)
    - kind == "wilson"         : threshold is a float or the string ``"auto"``
    """
    if data_type == "proteome":
        return "none", None, "proteome → no PTM localization filter"
    if "mean_loc_prob" not in adata.var.columns:
        return "none", None, "no `mean_loc_prob` column → skip PTM filter"

    n = adata.n_obs
    if n >= WILSON_MIN_N:
        # For cohorts big enough for sample-size correction to matter, prescribe
        # the Wilson filter.  "auto" lets the pipeline elbow-detect the threshold
        # within a cohort-size-informed range (see classI_wilson.AUTO_RANGES).
        return (
            "wilson",
            "auto",
            f'phospho/PTM at n={n} >= {WILSON_MIN_N} → strategy="wilson", '
            'wilson_threshold="auto" (elbow on retention curve)',
        )
    return (
        "mean_loc_prob",
        CLASSI_DEFAULT,
        f"phospho/PTM at n={n} < {WILSON_MIN_N} → mean_loc_prob >= {CLASSI_DEFAULT} "
        "(Wilson under-supported at this cohort scale)",
    )


def _audit_and_drop_cells(
    adata: ad.AnnData,
    primary: str,
    secondary: str | None,
) -> tuple[list[str], str]:
    if primary not in adata.obs.columns:
        return [], f"primary factor '{primary}' not in obs → no drop check"

    if secondary is None:
        counts = adata.obs[primary].astype(str).value_counts()
        bad = counts[counts < MIN_CELL_N].index.tolist()
        if not bad:
            return [], f"all {primary} groups have ≥ {MIN_CELL_N} samples"
        mask = adata.obs[primary].astype(str).isin(bad)
        return (
            adata.obs.index[mask].tolist(),
            f"drop {int(mask.sum())} sample(s) in {primary} groups {bad} (n<{MIN_CELL_N})",
        )

    cell = adata.obs[primary].astype(str) + "|" + adata.obs[secondary].astype(str)
    counts = cell.value_counts()
    bad = counts[counts < MIN_CELL_N].index.tolist()
    if not bad:
        return [], f"all {primary} x {secondary} cells have ≥ {MIN_CELL_N} samples"
    mask = cell.isin(bad)
    return (
        adata.obs.index[mask].tolist(),
        f"drop {int(mask.sum())} sample(s) in {primary}x{secondary} cells "
        f"{bad[:3]}{'...' if len(bad) > 3 else ''} (n<{MIN_CELL_N})",
    )


def _n_from_smallest(smallest: int) -> int:
    return max(3, min(N_CAP, round(0.6 * smallest)))


def _decide_filter(
    goal: Goal,
    adata: ad.AnnData,
    primary: str,
    secondary: str | None,
) -> tuple[dict, str]:
    if primary not in adata.obs.columns:
        return (
            {"min_valid_frac": 0.5, "keep_strategy": "all"},
            "no primary factor → global frac=0.5",
        )

    if goal == "profiling":
        return (
            {"min_valid_frac": 0.7, "keep_strategy": "all"},
            "profiling → global frac=0.7 (no groups)",
        )

    if goal == "onoff_discovery":
        smallest = int(adata.obs[primary].astype(str).value_counts().min())
        n = _n_from_smallest(smallest)
        return (
            {"group_column": primary, "keep_strategy": "any", "min_valid_n": n},
            f"onoff_discovery → group={primary}, any, n={n} (smallest group = {smallest})",
        )

    if goal == "interaction_de":
        if secondary is None:
            smallest = int(adata.obs[primary].astype(str).value_counts().min())
            n = _n_from_smallest(smallest)
            return (
                {"group_column": primary, "keep_strategy": "each", "min_valid_n": n},
                f"interaction_de but no secondary → group={primary}, each, n={n} "
                "(cannot form interaction cells; falling back to primary)",
            )
        cell = adata.obs[primary].astype(str) + "|" + adata.obs[secondary].astype(str)
        smallest = int(cell.value_counts().min())
        n = _n_from_smallest(smallest)
        return (
            {
                "group_column": f"{primary}_x_{secondary}",
                "keep_strategy": "each",
                "min_valid_n": n,
                "_synth_from": (primary, secondary),
            },
            f"interaction_de → group={primary}x{secondary} (synth), each, n={n} "
            f"(smallest cell = {smallest})",
        )

    if goal == "primary_de":
        if secondary is not None:
            cell = adata.obs[primary].astype(str) + "|" + adata.obs[secondary].astype(str)
            smallest = int(cell.value_counts().min())
            if smallest >= 5:
                n = _n_from_smallest(smallest)
                return (
                    {
                        "group_column": f"{primary}_x_{secondary}",
                        "keep_strategy": "each",
                        "min_valid_n": n,
                        "_synth_from": (primary, secondary),
                    },
                    f"primary_de → group={primary}x{secondary} (synth), each, n={n} "
                    f"(smallest cell = {smallest}; interaction-testable)",
                )
        smallest = int(adata.obs[primary].astype(str).value_counts().min())
        n = _n_from_smallest(smallest)
        return (
            {"group_column": primary, "keep_strategy": "each", "min_valid_n": n},
            f"primary_de → group={primary}, each, n={n} "
            "(interaction cells too small or secondary absent)",
        )

    if goal == "marginal_de":
        smallest = int(adata.obs[primary].astype(str).value_counts().min())
        n = _n_from_smallest(smallest)
        return (
            {"group_column": primary, "keep_strategy": "each", "min_valid_n": n},
            f"marginal_de → group={primary}, each, n={n}",
        )

    if goal == "viz_only":
        smallest = (
            int(adata.obs[primary].astype(str).value_counts().min()) if primary in adata.obs else 5
        )
        n = max(3, round(0.3 * smallest))
        return (
            {"group_column": primary, "keep_strategy": "any", "min_valid_n": n},
            f"viz_only → group={primary}, any, n={n} (loose; imputer fills the rest)",
        )

    return ({"min_valid_frac": 0.5, "keep_strategy": "all"}, "fallback: frac=0.5, all")


def _decide_de(goal: Goal) -> tuple[str, str, str]:
    """Return (short-name, one-line reasoning, code snippet template)."""
    if goal == "onoff_discovery":
        return (
            "ap.on_off_detection",
            "on/off → dedicated on/off detection; plain limma drops on/off features",
            'results = ap.on_off_detection(adata, condition_col="{primary}")',
        )
    if goal in ("viz_only", "profiling"):
        return ("(none — visualization/profiling only)", "no DE step requested", "")
    if goal == "interaction_de":
        return (
            "ap.diff_exp_limma_contrasts",
            "interaction → limma-trend with explicit interaction contrasts",
            (
                "results = ap.diff_exp_limma_contrasts(\n"
                "    adata,\n"
                '    design="~ 0 + {primary}_x_{secondary}",\n'
                '    contrasts={{"interaction_A_vs_B": "..."}},\n'
                ")"
            ),
        )
    if goal == "marginal_de":
        return (
            "ap.diff_exp_limma_observed_only",
            "marginal DE → limma on observed values, per-feature complete-case",
            'results = ap.diff_exp_limma_observed_only(adata, condition_col="{primary}")',
        )
    return (
        "ap.diff_exp_limma_observed_only (+ ap.diff_exp_limma_contrasts)",
        "primary DE → main effects + interaction contrasts, per-feature complete-case",
        (
            'main = ap.diff_exp_limma_observed_only(adata, condition_col="{primary}")\n'
            "# interaction contrasts:\n"
            "# inter = ap.diff_exp_limma_contrasts(adata, ...)"
        ),
    )


def _decide_imputer(
    goal: Goal,
    n_samples: int,
    pct_nan_post_filter: float,
) -> tuple[str | None, str, str]:
    """Return (method, reasoning, code snippet)."""
    if goal != "viz_only" and pct_nan_post_filter < 0.60:
        return (
            None,
            "DE handles NaN natively per-feature → skip imputation for stats",
            "# no imputation — DE runs on the observed matrix",
        )

    if n_samples < PIMMS_FLOOR:
        return (
            "knn",
            f"n_samples={n_samples} < {PIMMS_FLOOR} (PIMMS floor) → KNN",
            "adata_imputed = ap.impute_knn_site_based(adata, copy=True)",
        )

    if n_samples >= LARGE_COHORT:
        return (
            "pimms_dae",
            f"n_samples={n_samples} ≥ {LARGE_COHORT} → PIMMS-DAE (10 s vs ~25 min for KNN)",
            'adata_imputed = ap.impute_pimms(adata, model="DAE", copy=True)',
        )

    if pct_nan_post_filter >= NAN_HEAVY:
        return (
            "pimms_dae",
            f"post-filter NaN={pct_nan_post_filter:.0%} ≥ {NAN_HEAVY:.0%} → PIMMS-DAE",
            'adata_imputed = ap.impute_pimms(adata, model="DAE", copy=True)',
        )

    return (
        "knn",
        f"post-filter NaN={pct_nan_post_filter:.0%} < {NAN_HEAVY:.0%}, n_samples={n_samples} → KNN",
        "adata_imputed = ap.impute_knn_site_based(adata, copy=True)",
    )


def _preview(
    adata: ad.AnnData,
    classI_kind: str,
    classI_threshold: float | str | None,
    dropped: list[str],
    filter_cfg: dict,
    primary: str,
    secondary: str | None,
) -> dict:
    """Cheap filter dry-run: applies classI + drop + completeness, returns shape/NaN."""
    a = adata
    if classI_kind == "mean_loc_prob" and "mean_loc_prob" in a.var.columns:
        a = a[:, (a.var["mean_loc_prob"] >= float(classI_threshold)).values]
    elif classI_kind == "wilson" and "classI_wilson_lb" in a.var.columns:
        from alphaphos.preprocess.classI_wilson import (
            auto_wilson_threshold as _auto_wilson,
        )

        lb = a.var["classI_wilson_lb"].to_numpy()
        if classI_threshold == "auto":
            try:
                t_val, _ = _auto_wilson(lb, a.n_obs)
            except ValueError:
                t_val = 0.5  # dry-run fallback; execution path raises for real
        else:
            t_val = float(classI_threshold)
        a = a[:, lb >= t_val]
    if dropped:
        a = a[~a.obs.index.isin(dropped)]
    a = a.copy()
    if "_synth_from" in filter_cfg:
        p, s = filter_cfg["_synth_from"]
        a.obs[filter_cfg["group_column"]] = a.obs[p].astype(str) + "|" + a.obs[s].astype(str)
    kwargs = {k: v for k, v in filter_cfg.items() if not k.startswith("_")}
    try:
        filt = filter_by_completeness(a, **kwargs)
    except Exception as exc:
        return {"error": str(exc)[:120]}
    nan_pct = float(np.isnan(filt.X).mean()) if filt.X.size else 0.0
    result = {
        "n_samples_final": filt.n_obs,
        "n_sites_final": filt.n_vars,
        "pct_nan_final": nan_pct,
    }
    if secondary and primary in filt.obs.columns and secondary in filt.obs.columns:
        cell = filt.obs[primary].astype(str) + "|" + filt.obs[secondary].astype(str)
        ready = np.ones(filt.n_vars, dtype=bool)
        for c in cell.unique():
            m = (cell == c).values
            ready &= (~np.isnan(filt.X[m])).sum(axis=0) >= 3
        result["frac_interaction_testable"] = float(ready.mean())
    return result


def _render_code(
    classI_kind: str,
    classI_threshold: float | str | None,
    dropped: list[str],
    filter_cfg: dict,
    primary: str,
    secondary: str | None,
    de_snippet: str,
    imputer_snippet: str,
) -> str:
    lines = ["import alphaphos as ap", ""]
    step = 1

    if dropped:
        lines.append(f"# {step}. Drop {len(dropped)} sample(s) in degenerate cells")
        lines.append(f"adata = adata[~adata.obs.index.isin({dropped!r})].copy()")
        lines.append("")
        step += 1

    if classI_kind == "mean_loc_prob":
        lines.append(f"# {step}. Class-I localization filter (small-cohort default)")
        lines.append(f'adata = adata[:, adata.var["mean_loc_prob"] >= {classI_threshold}].copy()')
        lines.append("")
        step += 1
    elif classI_kind == "wilson":
        threshold_repr = "'auto'" if classI_threshold == "auto" else f"{classI_threshold}"
        lines.append(f"# {step}. Class-I Wilson-lb filter (large-cohort default)")
        lines.append(
            "# NOTE: assumes `adata` was produced by collapse_sites with "
            'advanced={"localization_strategy": "wilson", '
            f'"wilson_threshold": {threshold_repr}}}'
        )
        lines.append(
            "# The filter is applied inside collapse_sites itself; no separate step needed."
        )
        lines.append("")
        step += 1

    if "_synth_from" in filter_cfg:
        p, s = filter_cfg["_synth_from"]
        lines.append(f"# {step}. Build interaction group column")
        lines.append(f'adata.obs["{filter_cfg["group_column"]}"] = (')
        lines.append(f'    adata.obs["{p}"].astype(str) + "|" + adata.obs["{s}"].astype(str)')
        lines.append(")")
        lines.append("")
        step += 1

    lines.append(f"# {step}. Completeness filter")
    clean_cfg = {k: v for k, v in filter_cfg.items() if not k.startswith("_")}
    kwargs_str = ", ".join(f"{k}={v!r}" for k, v in clean_cfg.items())
    lines.append(f"adata = ap.filter_by_completeness(adata, {kwargs_str})")
    lines.append("")
    step += 1

    if imputer_snippet.strip() and not imputer_snippet.lstrip().startswith("#"):
        lines.append(f"# {step}. Impute")
        lines.append(imputer_snippet)
        lines.append("")
        step += 1
    else:
        lines.append(f"# {step}. {imputer_snippet.lstrip('# ').strip()}")
        lines.append("")
        step += 1

    if de_snippet.strip():
        lines.append(f"# {step}. DE")
        lines.append(de_snippet.format(primary=primary, secondary=secondary or "?"))

    return "\n".join(lines)


def _render_advice(
    goal: Goal,
    data_type: DataType,
    adata: ad.AnnData,
    classI_kind: str,
    classI_threshold: float | str | None,
    why_classI: str,
    dropped: list[str],
    why_drop: str,
    filter_cfg: dict,
    why_filter: str,
    why_de: str,
    de_snippet: str,
    imputer: str | None,
    why_imputer: str,
    imputer_snippet: str,
    preview: dict,
    primary: str,
    secondary: str | None,
) -> str:
    W = 74
    bar = "─" * W
    hbar = "═" * W
    input_nan = float(np.isnan(adata.X).mean()) if adata.X.size else 0.0
    lines = [
        hbar,
        "  ALPHAPHOS PIPELINE RECOMMENDATION",
        hbar,
        f"  goal:      {goal}",
        f"  data_type: {data_type}",
        f"  input:     {adata.n_obs} samples x {adata.n_vars:,} features ({input_nan:.1%} NaN)",
        "",
        f"  ┌ Decision trace {bar[16:]}",
        f"  │  · Class-I     : {why_classI}",
        f"  │  · Design audit: {why_drop}",
        f"  │  · Completeness: {why_filter}",
        f"  │  · DE          : {why_de}",
        f"  │  · Imputer     : {why_imputer}",
        f"  └{bar}",
        "",
        f"  ┌ Recommended code (copy-paste) {bar[31:]}",
    ]
    code = _render_code(
        classI_kind,
        classI_threshold,
        dropped,
        filter_cfg,
        primary,
        secondary,
        de_snippet,
        imputer_snippet,
    )
    for code_line in code.splitlines():
        lines.append(f"  │    {code_line}")
    lines.append(f"  └{bar}")
    lines.append("")

    lines.append(f"  ┌ Expected outcome {bar[18:]}")
    if "error" in preview:
        lines.append(f"  │  ⚠  filter dry-run failed: {preview['error']}")
    else:
        lines.append(
            f"  │    · Final matrix   : {preview['n_samples_final']} x {preview['n_sites_final']:,}"
        )
        lines.append(
            f"  │    · Missingness    : {preview['pct_nan_final']:.1%} NaN (before imputation)"
        )
        if "frac_interaction_testable" in preview:
            lines.append(
                f"  │    · Interaction OK : "
                f"{preview['frac_interaction_testable']:.1%} of sites "
                f"testable in every {primary}x{secondary} cell"
            )
        if imputer:
            lines.append(f"  │    · After impute   : {preview['n_sites_final']:,} x 0.0% NaN")
    lines.append(f"  └{bar}")
    lines.append(hbar)
    return "\n".join(lines)


def recommend_pipeline(
    adata: ad.AnnData,
    *,
    goal: Goal,
    data_type: DataType,
    primary_factor: str,
    secondary_factor: str | None = None,
    subject_col: str | None = None,
) -> None:
    """Print filter + impute + DE recommendations for ``adata``.

    Advisory only — executes nothing beyond a cheap filter dry-run used
    to estimate what the recommended recipe would retain.  Read the
    printed trace, copy the code, apply it yourself.

    Parameters
    ----------
    adata
        Site- or protein-level AnnData, typically the output of
        :func:`collapse_sites`.
    goal
        Analytical goal — controls filter grouping strategy and DE method.
        ``"primary_de"`` covers main-effects + interaction; specialize to
        ``"marginal_de"`` / ``"interaction_de"`` / ``"onoff_discovery"`` /
        ``"profiling"`` / ``"viz_only"`` when appropriate.
    data_type
        ``"phospho"`` / ``"other_ptm"`` triggers a Class-I localization
        filter on ``adata.var["mean_loc_prob"]``.  ``"proteome"`` skips it.
    primary_factor
        Column in ``adata.obs`` naming the primary condition (e.g.
        ``"fiber_type"``, ``"disease"``).
    secondary_factor
        Optional second design factor (e.g. ``"time_point"``).  When
        present and every ``primary × secondary`` cell has ≥ 5 samples,
        the recommended filter groups on the interaction cell so every
        retained site is interaction-testable.
    subject_col
        Column identifying subjects (accepted for API parity with
        downstream random-effects models; not used to shape the recipe
        directly).
    """
    _ = subject_col  # reserved for future random-effects reasoning
    classI_kind, classI_threshold, why_classI = _decide_classI(data_type, adata)
    dropped, why_drop = _audit_and_drop_cells(adata, primary_factor, secondary_factor)
    a_sub = adata[~adata.obs.index.isin(dropped)] if dropped else adata
    filter_cfg, why_filter = _decide_filter(goal, a_sub, primary_factor, secondary_factor)
    _de_name, why_de, de_snippet = _decide_de(goal)

    preview = _preview(
        adata,
        classI_kind,
        classI_threshold,
        dropped,
        filter_cfg,
        primary_factor,
        secondary_factor,
    )

    n_final = preview.get("n_samples_final", a_sub.n_obs)
    nan_final = preview.get("pct_nan_final", float(np.isnan(a_sub.X).mean()))
    imputer, why_imputer, imputer_snippet = _decide_imputer(goal, n_final, nan_final)

    print(
        _render_advice(
            goal=goal,
            data_type=data_type,
            adata=adata,
            classI_kind=classI_kind,
            classI_threshold=classI_threshold,
            why_classI=why_classI,
            dropped=dropped,
            why_drop=why_drop,
            filter_cfg=filter_cfg,
            why_filter=why_filter,
            why_de=why_de,
            de_snippet=de_snippet,
            imputer=imputer,
            why_imputer=why_imputer,
            imputer_snippet=imputer_snippet,
            preview=preview,
            primary=primary_factor,
            secondary=secondary_factor,
        )
    )


__all__ = ["recommend_pipeline", "Goal", "DataType"]
