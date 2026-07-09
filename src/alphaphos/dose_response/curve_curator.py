"""CurveCurator wrapper for dose-response phospho data.

`CurveCurator <https://github.com/kusterlab/curve_curator>`_ (Kuster lab,
TUM) fits 4-parameter log-logistic dose-response curves to MS data and
reports per-curve quality statistics (pEC50, fold change, R², F-test
p/q-value with target-decoy FDR option). It is **CLI-only** — the public
interface is ``python -m curve_curator <config.toml>``.

This module:

1. Builds CurveCurator's input TSV (Name + ``Raw <experiment>`` columns,
   linear intensities) from an AnnData's expression matrix.
2. Builds the TOML config file describing the experimental design
   (doses, controls, treatment time, fit parameters).
3. Runs the CLI in a subprocess.
4. Parses ``curves.txt`` back into a tidy DataFrame.

For multi-timepoint experiments (one dose-response per timepoint),
:func:`fit_dose_response` orchestrates one CurveCurator run per timepoint
and concatenates the per-curve results. For single-timepoint experiments,
pass ``timepoint_col=None``.

Ported from the uPhosHT lab convention and adapted to alphaPhos's
AnnData-centric idiom: doses, timepoints, and DMSO membership live in
``adata.obs`` columns; site ids come from ``adata.var.index``;
intensities come from ``adata.X`` (log2; this module converts to linear
for CurveCurator).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TOML helpers (CurveCurator uses TOML config files)
# ---------------------------------------------------------------------------


def _toml_quote(s: object) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _toml_str_array(items: list) -> str:
    return "[" + ", ".join(_toml_quote(x) for x in items) + "]"


def _toml_num_array(items: list) -> str:
    return "[" + ", ".join(repr(float(x)) for x in items) + "]"


# ---------------------------------------------------------------------------
# Input building
# ---------------------------------------------------------------------------


def _build_input_tsv(
    adata: ad.AnnData,
    *,
    output_path: Path,
    dose_col: str,
    timepoint_col: str | None,
    timepoint: float | None,
    layer: str | None,
) -> dict:
    """Build a CurveCurator OTHER-format input TSV from an AnnData.

    Parameters
    ----------
    adata
        AnnData with shape ``(n_samples, n_sites)``; ``.X`` (or ``layer``)
        contains log2 intensities.
    output_path
        Where to write the TSV (will overwrite).
    dose_col, timepoint_col
        Column names in ``adata.obs``. ``timepoint_col=None`` means
        single-timepoint design — all samples go into the one run.
    timepoint
        Value of ``timepoint_col`` to select. Ignored if
        ``timepoint_col is None``.
    layer
        Which ``adata.layers`` to use as intensities. ``None`` uses
        ``adata.X``.

    Returns
    -------
    dict with keys:

    - ``tsv_path``: Path to written TSV
    - ``experiments``: list of sample ids used as ``Raw <id>`` suffixes
    - ``doses``: dose values for each experiment
    - ``controls``: subset of experiments where dose == 0
    - ``n_sites``, ``n_wells``
    """
    if timepoint_col is None:
        obs_subset = adata.obs.copy()
    else:
        obs_subset = adata.obs.loc[adata.obs[timepoint_col] == timepoint].copy()
        if obs_subset.empty:
            raise ValueError(
                f"No samples found at {timepoint_col}={timepoint!r}. "
                f"Available values: {sorted(adata.obs[timepoint_col].unique())}"
            )

    sample_ids = obs_subset.index.tolist()
    doses = obs_subset[dose_col].astype(float).tolist()
    controls = [s for s, d in zip(sample_ids, doses, strict=True) if d == 0.0]

    if not controls:
        raise ValueError(
            "No DMSO controls found (dose == 0). CurveCurator requires "
            "at least one zero-dose sample per run."
        )

    # Pull intensities for these samples; convert log2 -> linear (NaN-preserving)
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]
    idx = adata.obs.index.get_indexer(sample_ids)
    log2_vals = np.asarray(X[idx, :], dtype=float)  # (n_wells, n_sites)
    raw_vals = np.power(2.0, log2_vals)
    raw_vals[~np.isfinite(raw_vals)] = np.nan

    raw_cols = [f"Raw {e}" for e in sample_ids]
    tsv = pd.DataFrame({"Name": adata.var.index.astype(str).values})
    for col, vals in zip(raw_cols, raw_vals, strict=True):
        tsv[col] = vals

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tsv.to_csv(output_path, sep="\t", index=False)

    return {
        "tsv_path": output_path,
        "experiments": sample_ids,
        "doses": doses,
        "controls": controls,
        "n_sites": len(tsv),
        "n_wells": len(sample_ids),
    }


# ---------------------------------------------------------------------------
# TOML config building
# ---------------------------------------------------------------------------


def _build_toml(
    *,
    input_tsv: Path,
    output_dir: Path,
    experiments: list,
    doses: list,
    controls: list,
    treatment_time: float | str,
    dose_unit: str = "nM",
    dose_scale: float = 1e-9,
    run_id: str | None = None,
    condition: str = "treatment",
    description: str = "",
    alpha: float = 0.05,
    fc_lim: float = 0.45,
    pec50_filter: tuple[float, float] = (2.0, 9.5),
    imputation: bool = False,
    normalization: bool = False,
    max_missing: int | None = None,
    available_cores: int = 4,
    fit_type: Literal["OLS"] = "OLS",
    fit_speed: Literal["fast", "standard", "extensive"] = "standard",
    control_fold_change: bool = True,
    dashboard_backend: str = "webgl",
) -> Path:
    """Write a CurveCurator TOML config file for one run.

    Returns the path to the written ``config.toml``.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_tsv = Path(input_tsv).resolve()

    if run_id is None:
        run_id = f"{condition}_t{treatment_time}"

    curves_file = output_dir / "curves.txt"
    decoys_file = output_dir / "decoys.txt"

    if isinstance(treatment_time, (int, float)):
        treatment_time_str = f"{int(treatment_time)} min"
    else:
        treatment_time_str = str(treatment_time)

    lines = [
        "[Meta]",
        f"id = {_toml_quote(run_id)}",
        f"condition = {_toml_quote(condition)}",
        f"description = {_toml_quote(description)}",
        f"treatment_time = {_toml_quote(treatment_time_str)}",
        "",
        "[Experiment]",
        f"experiments = {_toml_str_array(experiments)}",
        f"control_experiment = {_toml_str_array(controls)}",
        f"doses = {_toml_num_array(doses)}",
        f"dose_scale = {float(dose_scale)!r}",
        f"dose_unit = {_toml_quote(dose_unit)}",
        'measurement_type = "OTHER"',
        'data_type = "OTHER"',
        'search_engine = "OTHER"',
        "",
        "[Paths]",
        f"input_file = {_toml_quote(input_tsv.as_posix())}",
        f"curves_file = {_toml_quote(curves_file.as_posix())}",
        f"decoys_file = {_toml_quote(decoys_file.as_posix())}",
        "",
        "[Processing]",
        f"available_cores = {int(available_cores)}",
        f"imputation = {'true' if imputation else 'false'}",
        f"normalization = {'true' if normalization else 'false'}",
    ]
    if max_missing is not None:
        lines.append(f"max_missing = {int(max_missing)}")
    lines.extend(
        [
            "",
            '["Curve Fit"]',
            f"type = {_toml_quote(fit_type)}",
            f"speed = {_toml_quote(fit_speed)}",
            f"control_fold_change = {'true' if control_fold_change else 'false'}",
            "",
            '["F Statistic"]',
            f"alpha = {float(alpha)!r}",
            f"fc_lim = {float(fc_lim)!r}",
            f"pEC50_filter = {_toml_num_array(pec50_filter)}",
            "",
            "[Dashboard]",
            f"backend = {_toml_quote(dashboard_backend)}",
            "",
        ]
    )

    toml_path = output_dir / "config.toml"
    toml_path.write_text("\n".join(lines), encoding="utf-8")
    return toml_path


# ---------------------------------------------------------------------------
# Subprocess execution
# ---------------------------------------------------------------------------


def _run_curve_curator(
    toml_path: Path,
    *,
    fdr: bool = True,
    mad: bool = False,
    capture_output: bool = True,
    check: bool = False,
    python_executable: str | None = None,
) -> subprocess.CompletedProcess:
    """Invoke CurveCurator: ``python -m curve_curator [opts] <toml_path>``.

    Parameters
    ----------
    toml_path
        Config file to run.
    fdr
        Pass ``--fdr`` (target-decoy FDR; default True — recommended).
    mad
        Pass ``--mad`` (MAD outlier analysis).
    capture_output
        If True, stdout/stderr are captured.
    check
        If True, raise on non-zero exit. Default False so the caller can
        inspect a partial result before deciding what to do.
    python_executable
        Path to the Python that has ``curve_curator`` installed. Default
        ``sys.executable``.
    """
    cmd = [python_executable or sys.executable, "-m", "curve_curator"]
    if fdr:
        cmd.append("--fdr")
    if mad:
        cmd.append("--mad")
    cmd.append(str(toml_path))
    return subprocess.run(
        cmd,
        capture_output=capture_output,
        check=check,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

# Subset of CurveCurator output columns we expose by default. Anything else
# stays accessible via the raw curves.txt file in the output directory.
DEFAULT_CURVE_COLS = [
    "Name",
    "pEC50",
    "Curve Slope",
    "Curve Front",
    "Curve Back",
    "Curve Fold Change",
    "Curve AUC",
    "Curve RMSE",
    "Curve R2",
    "Curve F_Value",
    "Curve P_Value",
    "Curve Log P_Value",
    "pEC50 Error",
    "Curve Slope Error",
    "Signal Quality",
    "Control Ratio Std",
    # FDR-mode columns (only when fdr=True at run time)
    "Curve F_Value SAM Corrected",
    "Curve Relevance Score",
    "Curve q_Value",
    "Curve Regulation",
    "Curve qFDR",
]


def _parse_curves(
    curves_path: Path,
    *,
    timepoint_label: float | str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read ``curves.txt`` produced by CurveCurator into a tidy DataFrame.

    Adds an ``EC50_nM`` column derived from ``pEC50`` (``10**(9 - pEC50)``).
    Renames ``Name`` -> ``site_key``.
    """
    df = pd.read_csv(curves_path, sep="\t")
    if columns is None:
        columns = [c for c in DEFAULT_CURVE_COLS if c in df.columns]
    df = df[columns].copy().rename(columns={"Name": "site_key"})
    if timepoint_label is not None:
        df["timepoint"] = timepoint_label
        # Move timepoint after site_key for readability
        cols = ["site_key", "timepoint"] + [
            c for c in df.columns if c not in ("site_key", "timepoint")
        ]
        df = df[cols]
    if "pEC50" in df.columns:
        df["EC50_nM"] = 10 ** (9 - df["pEC50"])
    return df


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------


def fit_dose_response(
    adata: ad.AnnData,
    *,
    output_root: str | Path,
    dose_col: str = "dose",
    timepoint_col: str | None = "timepoint",
    timepoints: list | None = None,
    layer: str | None = None,
    # CurveCurator [Meta]
    condition: str = "treatment",
    description: str = "",
    # CurveCurator [Experiment]
    dose_unit: str = "nM",
    dose_scale: float = 1e-9,
    # CurveCurator [Processing]
    imputation: bool = False,
    normalization: bool = False,
    max_missing: int | None = None,
    available_cores: int = 4,
    # CurveCurator ["Curve Fit"]
    fit_type: Literal["OLS"] = "OLS",
    fit_speed: Literal["fast", "standard", "extensive"] = "standard",
    control_fold_change: bool = True,
    # CurveCurator ["F Statistic"]
    alpha: float = 0.05,
    fc_lim: float = 0.45,
    pec50_filter: tuple[float, float] = (2.0, 9.5),
    # CLI flags
    fdr: bool = True,
    mad: bool = False,
    # logging
    verbose: bool = True,
) -> pd.DataFrame:
    """Run CurveCurator dose-response fits and return a concatenated per-curve DataFrame.

    For multi-timepoint experiments, fits one curve per (site, timepoint)
    independently — CurveCurator does NOT fit dose × time jointly. For
    single-timepoint, runs once.

    Parameters
    ----------
    adata
        AnnData with shape ``(n_samples, n_sites)``. ``.X`` (or ``layer``)
        contains log2 intensities. ``adata.obs`` must have a numeric
        ``dose_col`` and (optionally) a ``timepoint_col`` column.
        DMSO/control samples must have ``dose == 0``.
    output_root
        Root directory for CurveCurator output. Per-timepoint subdirs are
        created at ``output_root/t<timepoint>/``. Each contains
        ``input.tsv``, ``config.toml``, ``curves.txt`` (and the
        interactive HTML dashboard).
    dose_col, timepoint_col
        Column names in ``adata.obs``. ``timepoint_col=None`` means a
        single-timepoint design — all samples go into one run; the
        output dir is ``output_root/single/``.
    timepoints
        Optional subset of timepoints to fit. ``None`` = all unique
        non-NaN values of ``timepoint_col``.
    layer
        Which ``adata.layers`` to use. ``None`` = ``adata.X``.
    condition, description
        Free-text labels stored in CurveCurator's ``[Meta]`` block.
    dose_unit, dose_scale
        Units of ``adata.obs[dose_col]`` and the multiplier to molar.
        ``dose_unit='nM'``, ``dose_scale=1e-9`` is the lab default.
    imputation, normalization
        CurveCurator's own pre-processing flags. Default off — alphaPhos
        already does these steps upstream.
    max_missing
        Max NaN raw values per curve. ``None`` = CurveCurator default
        (no missing-value filter). A common choice is the number of
        non-control wells (so any site missing in a single dose is
        dropped).
    available_cores
        Parallel cores for the fit. Default 4.
    fit_type, fit_speed, control_fold_change
        CurveCurator ``[Curve Fit]`` parameters. ``OLS`` is the standard
        4-parameter log-logistic fit.
    alpha, fc_lim, pec50_filter
        CurveCurator ``[F Statistic]`` thresholds for the "regulated"
        call. Default lab settings.
    fdr, mad
        CLI flags. ``fdr=True`` (default) runs target-decoy FDR; ``mad``
        runs MAD outlier analysis.
    verbose
        Print progress per timepoint.

    Returns
    -------
    pd.DataFrame
        Concatenated curves across timepoints. Columns include
        ``site_key``, ``timepoint`` (if multi-timepoint), ``pEC50``,
        ``EC50_nM``, ``Curve Fold Change``, ``Curve F_Value``,
        ``Curve P_Value``, ``Curve q_Value`` (when fdr=True), and a few
        more — see :data:`DEFAULT_CURVE_COLS`.

    Raises
    ------
    ImportError
        If ``curve_curator`` is not importable in the active Python.
    RuntimeError
        If a CurveCurator subprocess returns a non-zero exit code or
        produces no curves output.
    """
    # Sanity: CurveCurator must be importable in this Python
    try:
        import curve_curator  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "curve_curator is required. Install with: pip install curve_curator"
        ) from exc

    if dose_col not in adata.obs.columns:
        raise ValueError(
            f"adata.obs is missing required dose column {dose_col!r}. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Resolve the iteration plan
    if timepoint_col is None:
        iter_plan: list[tuple[Path, float | None]] = [(output_root / "single", None)]
    else:
        if timepoint_col not in adata.obs.columns:
            raise ValueError(
                f"adata.obs is missing {timepoint_col!r}. "
                f"Pass timepoint_col=None for single-timepoint design."
            )
        if timepoints is None:
            tps = sorted(adata.obs[timepoint_col].dropna().unique().tolist())
        else:
            tps = list(timepoints)
        iter_plan = [(output_root / f"t{int(t) if float(t).is_integer() else t}", t) for t in tps]

    all_curves = []
    for tdir, t in iter_plan:
        label = "single" if t is None else f"t={t}"
        if verbose:
            print(f"\n=== {label} ===")
            print(f"  output dir: {tdir}")

        info = _build_input_tsv(
            adata,
            output_path=tdir / "input.tsv",
            dose_col=dose_col,
            timepoint_col=timepoint_col,
            timepoint=t,
            layer=layer,
        )
        if verbose:
            print(
                f"  TSV: {info['n_sites']:,} sites x {info['n_wells']} wells "
                f"({len(info['controls'])} DMSO controls)"
            )

        run_id = f"{condition}_{label}".replace(" ", "").replace("=", "")
        treatment_time = t if t is not None else 0
        toml_path = _build_toml(
            input_tsv=info["tsv_path"],
            output_dir=tdir,
            experiments=info["experiments"],
            doses=info["doses"],
            controls=info["controls"],
            treatment_time=treatment_time,
            dose_unit=dose_unit,
            dose_scale=dose_scale,
            run_id=run_id,
            condition=condition,
            description=description,
            alpha=alpha,
            fc_lim=fc_lim,
            pec50_filter=pec50_filter,
            imputation=imputation,
            normalization=normalization,
            max_missing=max_missing,
            available_cores=available_cores,
            fit_type=fit_type,
            fit_speed=fit_speed,
            control_fold_change=control_fold_change,
        )

        proc = _run_curve_curator(toml_path, fdr=fdr, mad=mad)
        if proc.returncode != 0:
            tail_out = (proc.stdout or "")[-2000:]
            tail_err = (proc.stderr or "")[-2000:]
            raise RuntimeError(
                f"CurveCurator failed at {label} (returncode={proc.returncode}).\n"
                f"STDOUT (tail):\n{tail_out}\n\n"
                f"STDERR (tail):\n{tail_err}"
            )

        curves_file = tdir / "curves.txt"
        if not curves_file.exists():
            tail = (proc.stdout or "")[-2000:]
            raise RuntimeError(
                f"CurveCurator exited 0 at {label} but produced no curves.txt. "
                f"Most common cause: every site failed max_missing or quality "
                f"filtering. Last output:\n{tail}"
            )

        curves = _parse_curves(curves_file, timepoint_label=t)
        all_curves.append(curves)
        if verbose:
            msg = f"  fit: {len(curves):,} curves"
            if "Curve P_Value" in curves.columns:
                msg += f" | P<{alpha}: {int((curves['Curve P_Value'] < alpha).sum()):,}"
            if fdr and "Curve q_Value" in curves.columns:
                msg += f" | q<{alpha}: {int((curves['Curve q_Value'] < alpha).sum()):,}"
            print(msg)

    return pd.concat(all_curves, ignore_index=True)
