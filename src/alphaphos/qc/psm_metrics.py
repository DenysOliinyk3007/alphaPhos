"""QC metrics computed from a raw PSM DataFrame (pre-collapse).

These functions operate on the DataFrame returned by
:func:`alphaphos.read_spectronaut` (or ``read_diann``) BEFORE
:func:`alphaphos.collapse_sites`.  Some signals -- charge state
distribution, retention-time drift, per-precursor Q-value dispersion --
are irreversibly lost once collapse aggregates PSMs into site-level
observations, so they must be measured at this stage.

All functions are pure: take the PSM DataFrame (plus optional
supplementary inputs like an acquisition queue), return a tidy
``pd.DataFrame``, no plotting side-effects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from typing import Any

# Default column names -- match the alphaPhos convention after
# read_spectronaut() (dots → underscores).
DEFAULT_SAMPLE_COLUMN = "R_FileName"
DEFAULT_RT_COLUMN = "EG_ApexRT"
DEFAULT_PRECURSOR_COLUMN = "EG_PrecursorId"

# Empirically-sane defaults.  Users override per-study.
DEFAULT_MIN_SAMPLES_PER_PRECURSOR = 3
DEFAULT_OFFSET_FLAG_THRESHOLD_MIN = 1.0  # minutes

# Robust z-score threshold for low-depth flagging.  -3.0 = "3 MAD-SDs
# below the cohort median" -- catches the C8-style catastrophic
# injection failures while ignoring normal cohort spread.
DEFAULT_DEPTH_FLAG_Z = -3.0
DEFAULT_PROTEIN_COLUMN = "PG_ProteinGroups"

# TIC (Total Ion Current) proxy at the precursor level.  Spectronaut's
# "EG.TotalQuantity (Settings)" column carries the precursor's total
# intensity (sum-of-fragments after Spectronaut's processing).  Post-
# read_spectronaut it becomes "EG_TotalQuantity_(Settings)".  DIA-NN
# users should override to "Precursor.Quantity".
DEFAULT_INTENSITY_COLUMN = "EG_TotalQuantity_(Settings)"

# Contaminant-fraction QC threshold.  Phospho enrichment protocols
# typically clear well below 10% contaminant intensity; anything above
# 10% signals a sample-prep or column-carryover problem.
DEFAULT_CONTAMINANT_FLAG_FRACTION = 0.10


def compute_retention_time_drift(
    psm_df: pd.DataFrame,
    *,
    acquisition_queue: pd.DataFrame | None = None,
    sample_column: str = DEFAULT_SAMPLE_COLUMN,
    rt_column: str = DEFAULT_RT_COLUMN,
    precursor_column: str = DEFAULT_PRECURSOR_COLUMN,
    min_samples_per_precursor: int = DEFAULT_MIN_SAMPLES_PER_PRECURSOR,
    offset_flag_threshold: float = DEFAULT_OFFSET_FLAG_THRESHOLD_MIN,
) -> pd.DataFrame:
    """Per-sample RT drift vs the cohort median, on shared precursors.

    For every precursor detected in ``>= min_samples_per_precursor``
    samples, take the cohort median RT.  For each sample, report the
    distribution of ``(sample_RT - cohort_median_RT)`` across the
    precursors it shares.

    Parameters
    ----------
    psm_df
        Raw PSM DataFrame from ``ap.read_spectronaut`` /
        ``ap.read_diann``.
    acquisition_queue
        Optional queue DataFrame from
        :func:`alphaphos.qc.load_acquisition_queue`.  Provides the
        ``injection_order`` axis.  Must be joined via **exact match**
        on the sample identifier (i.e. queue's ``sample`` column
        equals ``psm_df[sample_column]`` verbatim -- raw file names
        should never be edited between Xcalibur and Spectronaut).
    sample_column, rt_column, precursor_column
        Column names in ``psm_df``.  Defaults follow the
        ``read_spectronaut`` post-normalisation schema.
    min_samples_per_precursor
        Precursors detected in fewer than this many samples are
        excluded from the drift calculation (drift-vs-cohort is
        undefined without enough shared observations).  Default 3.
    offset_flag_threshold
        A sample is flagged (``is_flagged=True``) if
        ``abs(median_offset) > offset_flag_threshold`` minutes.  Default
        1.0 min -- sensible for a ~60 min gradient.

    Returns
    -------
    pandas.DataFrame indexed by ``sample`` with columns:

    - ``n_shared_precursors`` : int
    - ``median_rt``           : float, sample's median RT (min) across shared precursors
    - ``median_offset``       : float, sample_median - cohort_median (min, signed)
    - ``iqr_offset``          : float, IQR of per-precursor offsets within the sample
    - ``gradient_slope``      : float, signed slope of offset vs cohort-precursor-RT
    - ``is_flagged``          : bool

    When ``acquisition_queue`` is provided, adds:

    - ``injection_order``     : int
    - ``offset_slope_vs_order`` : float, global slope (same value per row)
    - ``is_early_batch``      : bool

    ``.attrs['provenance']`` carries counts + Spearman /
    Pearson correlations vs ``injection_order`` when the queue is
    supplied.
    """
    _validate_columns(psm_df, [sample_column, rt_column, precursor_column])

    # Per (precursor, sample) median RT.  Handles Spectronaut's fragment-
    # level duplicates within a precursor+sample group.
    core = psm_df.loc[:, [sample_column, precursor_column, rt_column]].copy()
    core = core.dropna(subset=[sample_column, precursor_column, rt_column])
    pivot = (
        core.groupby([precursor_column, sample_column], observed=True)[rt_column]
        .median()
        .unstack(sample_column)
    )
    n_samples_per_precursor = pivot.notna().sum(axis=1)
    shared = pivot.loc[n_samples_per_precursor >= min_samples_per_precursor]

    samples = list(pivot.columns.astype(str))
    if shared.empty or len(samples) == 0:
        return _empty_drift_frame(samples=samples, queue=acquisition_queue)

    # Cohort median RT per precursor
    cohort_median_rt = shared.median(axis=1)  # index = precursor
    # (precursor, sample) offsets
    offsets = shared.sub(cohort_median_rt, axis=0)

    rows: list[dict[str, Any]] = []
    for sample in samples:
        if sample not in offsets.columns:
            rows.append(_null_sample_row(sample))
            continue
        sample_offsets = offsets[sample].dropna()
        if len(sample_offsets) < 2:
            rows.append(_null_sample_row(sample, n=len(sample_offsets)))
            continue
        median_offset = float(sample_offsets.median())
        iqr = float(sample_offsets.quantile(0.75) - sample_offsets.quantile(0.25))
        sample_median_rt = float(shared[sample].dropna().median())
        # Slope of within-sample offset vs cohort-precursor-RT (gradient distortion)
        x = cohort_median_rt.reindex(sample_offsets.index).to_numpy(dtype=float)
        y = sample_offsets.to_numpy(dtype=float)
        gradient_slope = _linear_slope(x, y)
        rows.append(
            {
                "sample": sample,
                "n_shared_precursors": len(sample_offsets),
                "median_rt": sample_median_rt,
                "median_offset": median_offset,
                "iqr_offset": iqr,
                "gradient_slope": gradient_slope,
                "is_flagged": bool(abs(median_offset) > offset_flag_threshold),
            }
        )

    df = pd.DataFrame(rows).set_index("sample")

    provenance: dict[str, Any] = {
        "n_samples_input": len(samples),
        "n_samples_with_drift": int(df["n_shared_precursors"].notna().sum()),
        "n_shared_precursors_total": len(shared),
        "min_samples_per_precursor": int(min_samples_per_precursor),
        "offset_flag_threshold_min": float(offset_flag_threshold),
        "sample_column": sample_column,
        "rt_column": rt_column,
        "precursor_column": precursor_column,
    }

    if acquisition_queue is not None:
        df = _attach_queue(df, acquisition_queue, provenance=provenance)

    df.attrs["provenance"] = provenance
    return df


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def compute_psm_counts_per_sample(
    psm_df: pd.DataFrame,
    *,
    acquisition_queue: pd.DataFrame | None = None,
    sample_column: str = DEFAULT_SAMPLE_COLUMN,
    precursor_column: str = DEFAULT_PRECURSOR_COLUMN,
    protein_column: str = DEFAULT_PROTEIN_COLUMN,
    flag_z_threshold: float = DEFAULT_DEPTH_FLAG_Z,
) -> pd.DataFrame:
    """Per-sample identification depth (PSMs / precursors / protein groups).

    The first metric anyone looks at after an MS batch: **did every
    sample produce a reasonable number of identifications?**  Samples
    with unusually low depth are candidates for drop (failed injection,
    low sample amount, degraded sample).

    Robustness: the cohort statistic is a MAD-based z-score, not
    mean/std.  MAD is unaffected by extreme outliers (e.g. a catastrophic
    failed injection dragging the mean), so ``depth_z_score`` accurately
    tells you "how many robust SDs below the cohort median is this
    sample?" even when a few samples are broken.

    Parameters
    ----------
    psm_df
        Raw PSM DataFrame from ``ap.read_spectronaut`` /
        ``ap.read_diann``.
    acquisition_queue
        Optional queue DataFrame from
        :func:`alphaphos.qc.load_acquisition_queue`.  When provided,
        adds ``injection_order`` to the output; join is exact-match on
        the sample identifier.
    sample_column, precursor_column, protein_column
        Column names in ``psm_df``.  Defaults follow ``read_spectronaut``
        post-normalisation (dots → underscores).
    flag_z_threshold
        A sample is flagged (``is_flagged=True``) when its
        ``depth_z_score < flag_z_threshold``.  Default -3.0.  Flags only
        the low-depth tail -- high-depth outliers are not flagged.

    Returns
    -------
    pandas.DataFrame indexed by ``sample`` with columns:

    - ``n_psms`` : int, total PSM rows for this sample (fragment-level,
      may double-count when Spectronaut emits per-fragment rows)
    - ``n_unique_precursors`` : int, unique ``EG_PrecursorId`` --
      the honest identification-depth metric
    - ``n_unique_proteins`` : int, unique ``PG_ProteinGroups``
    - ``depth_z_score`` : float, MAD-based z-score of
      ``n_unique_precursors`` vs cohort median
    - ``is_flagged`` : bool

    When ``acquisition_queue`` is provided, adds ``injection_order``.
    Provenance dict in ``.attrs['provenance']`` records cohort statistics
    + queue-match counts.
    """
    _validate_columns(
        psm_df,
        [sample_column, precursor_column, protein_column],
    )
    core = psm_df.loc[:, [sample_column, precursor_column, protein_column]].dropna(
        subset=[sample_column]
    )

    grouped = core.groupby(sample_column, observed=True)
    n_psms = grouped.size().astype(int)
    n_precursors = grouped[precursor_column].nunique().astype(int)
    n_proteins = grouped[protein_column].nunique().astype(int)

    # MAD-based robust z-score on unique-precursor counts
    values = n_precursors.to_numpy(dtype=float)
    median = float(np.median(values)) if values.size else 0.0
    mad = float(np.median(np.abs(values - median))) if values.size else 0.0
    if mad > 0.0:
        robust_sd = 1.4826 * mad
        z_scores = (values - median) / robust_sd
    else:
        # Degenerate: all samples have the same count -> everybody is z=0
        z_scores = np.zeros_like(values)

    df = pd.DataFrame(
        {
            "n_psms": n_psms,
            "n_unique_precursors": n_precursors,
            "n_unique_proteins": n_proteins,
            "depth_z_score": z_scores,
            "is_flagged": z_scores < flag_z_threshold,
        }
    )
    df.index.name = "sample"

    provenance: dict[str, Any] = {
        "n_samples": len(df),
        "cohort_median_unique_precursors": float(median),
        "cohort_mad_unique_precursors": float(mad),
        "flag_z_threshold": float(flag_z_threshold),
        "n_flagged": int(df["is_flagged"].sum()),
        "sample_column": sample_column,
        "precursor_column": precursor_column,
        "protein_column": protein_column,
    }

    if acquisition_queue is not None:
        df = _attach_queue_simple(df, acquisition_queue, provenance=provenance)

    df.attrs["provenance"] = provenance
    return df


def compute_run_tic_per_sample(
    psm_df: pd.DataFrame,
    *,
    acquisition_queue: pd.DataFrame | None = None,
    sample_column: str = DEFAULT_SAMPLE_COLUMN,
    precursor_column: str = DEFAULT_PRECURSOR_COLUMN,
    intensity_column: str = DEFAULT_INTENSITY_COLUMN,
    flag_z_threshold: float = DEFAULT_DEPTH_FLAG_Z,
) -> pd.DataFrame:
    """Per-sample TIC-proxy (summed precursor intensity).

    Not a true TIC (which is the raw-file sum across all m/z bins), but
    the practical equivalent for identified precursors: for each sample,
    take one intensity per unique precursor (dedup on Spectronaut's
    fragment-level rows), then sum.  The z-score is computed on
    ``log2(total_intensity)`` because intensity spans orders of
    magnitude and MAD on the linear scale would be dominated by
    high-intensity samples.

    Instrument QC signal.  Combined with ``compute_psm_counts_per_sample``
    tells you: is a low-depth sample low because *nothing ionised well*
    (low TIC) or because *many things ionised but few got matched*
    (normal TIC, low ID count)?  Different root causes.

    Parameters
    ----------
    psm_df
        Raw PSM DataFrame from ``ap.read_spectronaut`` /
        ``ap.read_diann``.
    acquisition_queue
        Optional queue DataFrame from
        :func:`alphaphos.qc.load_acquisition_queue`.
    sample_column, precursor_column, intensity_column
        Column names.  Defaults follow ``read_spectronaut``.  DIA-NN
        users typically pass ``intensity_column='Precursor.Quantity'``.
    flag_z_threshold
        MAD-based z-score threshold on ``log2_total_intensity``.  A
        sample is flagged when its z falls below this.  Default -3.0.

    Returns
    -------
    pandas.DataFrame indexed by ``sample`` with columns:

    - ``total_intensity`` : float, sum of per-precursor intensity
    - ``log2_total_intensity`` : float, log2 of the total
    - ``median_precursor_intensity`` : float, robust central tendency
    - ``n_precursors_with_intensity`` : int, count of finite intensities
    - ``tic_z_score`` : float, MAD z-score on ``log2_total_intensity``
    - ``is_flagged`` : bool

    ``.attrs['provenance']`` records cohort statistics + queue match
    counts.
    """
    _validate_columns(psm_df, [sample_column, precursor_column, intensity_column])

    core = psm_df.loc[:, [sample_column, precursor_column, intensity_column]].dropna(
        subset=[sample_column, precursor_column]
    )
    # One intensity per (sample, precursor).  Spectronaut may emit
    # multiple fragment rows for a single precursor -- they share the
    # precursor's EG_TotalQuantity, so ``.first()`` is safe.
    per_precursor = (
        core.groupby([sample_column, precursor_column], observed=True)[intensity_column]
        .first()
        .rename("intensity")
        .reset_index()
    )
    per_precursor = per_precursor.loc[per_precursor["intensity"] > 0]

    grouped = per_precursor.groupby(sample_column, observed=True)
    total = grouped["intensity"].sum().astype(float)
    median = grouped["intensity"].median().astype(float)
    n_prec = grouped["intensity"].size().astype(int)

    # Log2 transform for the z-score computation (intensities span orders
    # of magnitude; MAD on the raw scale is dominated by high-intensity
    # samples).
    log2_total = np.log2(total.replace(0, np.nan))
    values = log2_total.to_numpy(dtype=float)
    finite = np.isfinite(values)
    if finite.any():
        median_log2 = float(np.median(values[finite]))
        mad_log2 = float(np.median(np.abs(values[finite] - median_log2)))
    else:
        median_log2 = float("nan")
        mad_log2 = 0.0
    if mad_log2 > 0.0:
        robust_sd = 1.4826 * mad_log2
        z_scores = (values - median_log2) / robust_sd
    else:
        z_scores = np.zeros_like(values)

    df = pd.DataFrame(
        {
            "total_intensity": total,
            "log2_total_intensity": log2_total,
            "median_precursor_intensity": median,
            "n_precursors_with_intensity": n_prec,
            "tic_z_score": pd.Series(z_scores, index=total.index),
            "is_flagged": pd.Series(z_scores < flag_z_threshold, index=total.index),
        }
    )
    df.index.name = "sample"

    provenance: dict[str, Any] = {
        "n_samples": len(df),
        "intensity_column": intensity_column,
        "cohort_median_log2_total_intensity": float(median_log2),
        "cohort_mad_log2_total_intensity": float(mad_log2),
        "flag_z_threshold": float(flag_z_threshold),
        "n_flagged": int(df["is_flagged"].sum()),
        "sample_column": sample_column,
        "precursor_column": precursor_column,
    }

    if acquisition_queue is not None:
        df = _attach_queue_simple(df, acquisition_queue, provenance=provenance)

    df.attrs["provenance"] = provenance
    return df


def compute_contaminant_fraction_per_sample(
    psm_df: pd.DataFrame,
    *,
    acquisition_queue: pd.DataFrame | None = None,
    sample_column: str = DEFAULT_SAMPLE_COLUMN,
    precursor_column: str = DEFAULT_PRECURSOR_COLUMN,
    protein_column: str = DEFAULT_PROTEIN_COLUMN,
    intensity_column: str = DEFAULT_INTENSITY_COLUMN,
    contaminant_prefixes: tuple[str, ...] | None = None,
    contaminants_fasta: str | Path | None = None,
    require_all_contaminant: bool = True,
    flag_fraction_threshold: float = DEFAULT_CONTAMINANT_FLAG_FRACTION,
) -> pd.DataFrame:
    """Per-sample fraction of precursor intensity from contaminants.

    Two-mechanism contaminant detection matching
    :func:`alphaphos.preprocess.filter_contaminants`:

    (A) **Prefix match** on protein IDs -- ``CON__`` (MaxQuant),
        ``Cont_`` (Spectronaut), ``contam_`` (Bilbao/Frankenfield).

    (B) **FASTA accession lookup** against the bundled MaxQuant
        ``contaminants.fasta`` (246 sequences: keratins, BSA, trypsin,
        serum proteins, common buffer/lab contaminants).  Applied
        even when prefixes are absent -- catches reports where the
        search engine did not prefix-tag contaminants.

    A precursor is contaminant when **every** protein in its
    ``PG_ProteinGroups`` matches (A) or (B), by default.  Ambiguous
    precursors mapped to a contaminant *and* a real protein are kept
    as non-contaminant.  Set ``require_all_contaminant=False`` to
    count them as contaminant instead (more aggressive).

    Parameters
    ----------
    psm_df
        Raw PSM DataFrame from ``ap.read_spectronaut`` /
        ``ap.read_diann``, **before** ``drop_contaminants=True`` has
        been applied (this metric measures the raw contamination level;
        run it upstream of contaminant filtering).
    acquisition_queue
        Optional queue from :func:`alphaphos.qc.load_acquisition_queue`.
    sample_column, precursor_column, protein_column, intensity_column
        Column names.  Defaults follow the ``read_spectronaut``
        post-normalisation schema (dots -> underscores).
    contaminant_prefixes
        Prefixes marking a contaminant.  ``None`` (default) uses
        :data:`alphaphos.preprocess.contaminants.DEFAULT_CONTAMINANT_PREFIXES`.
    contaminants_fasta
        FASTA to source curated contaminant accessions.  ``None``
        (default) uses the bundled MaxQuant contaminants FASTA.  Pass
        an explicit path to use a different list, or ``""`` to disable
        FASTA lookup (prefix detection only).
    require_all_contaminant
        See mechanism above.  Default ``True`` matches
        ``filter_contaminants``.
    flag_fraction_threshold
        Flag samples with ``contaminant_fraction > threshold``.
        Default 0.10 (10% of total precursor intensity).

    Returns
    -------
    pandas.DataFrame indexed by ``sample`` with columns:

    - ``contaminant_intensity`` : summed intensity of contaminant precursors
    - ``total_intensity`` : summed intensity of all precursors
    - ``contaminant_fraction`` : 0.0-1.0
    - ``n_contaminant_precursors`` : int
    - ``n_total_precursors`` : int
    - ``is_flagged`` : bool

    ``.attrs['provenance']`` records the detection config + cohort
    median + queue match counts.

    Notes
    -----
    On phospho-enriched samples the metric typically reads near zero
    (contaminants are largely non-phospho, so IMAC/TiO2 enrichment
    depletes them).  A non-zero result on such a run is a strong
    QC signal for column carryover, tip breakthrough, or a fouled
    trap.  On whole-cell proteome runs, baseline can be 1-5% even
    for a clean prep.
    """
    from alphaphos.preprocess.contaminants import (
        DEFAULT_CONTAMINANT_PREFIXES,
        _is_contaminant_protein,
        get_default_contaminants_fasta,
        parse_fasta_accessions,
    )

    if contaminant_prefixes is None:
        contaminant_prefixes = DEFAULT_CONTAMINANT_PREFIXES

    if contaminants_fasta is None:
        contaminants_fasta = get_default_contaminants_fasta()
    accession_set = parse_fasta_accessions(contaminants_fasta) if contaminants_fasta else set()

    _validate_columns(psm_df, [sample_column, precursor_column, protein_column, intensity_column])

    core = psm_df.loc[
        :, [sample_column, precursor_column, protein_column, intensity_column]
    ].dropna(subset=[sample_column, precursor_column])

    per_precursor = (
        core.groupby([sample_column, precursor_column], observed=True)
        .agg(
            intensity=(intensity_column, "first"),
            protein_group=(protein_column, "first"),
        )
        .reset_index()
    )
    per_precursor = per_precursor.loc[per_precursor["intensity"] > 0]

    def _pg_is_contaminant(pg: object) -> bool:
        if pg is None or (isinstance(pg, float) and pd.isna(pg)):
            return False
        proteins = [p for p in str(pg).split(";") if p.strip()]
        if not proteins:
            return False
        flags = [
            _is_contaminant_protein(p, accession_set, tuple(contaminant_prefixes)) for p in proteins
        ]
        return all(flags) if require_all_contaminant else any(flags)

    per_precursor["is_contam"] = per_precursor["protein_group"].map(_pg_is_contaminant)

    grouped = per_precursor.groupby(sample_column, observed=True)
    total_intensity = grouped["intensity"].sum().astype(float)
    n_total = grouped["intensity"].size().astype(int)

    contam_rows = per_precursor.loc[per_precursor["is_contam"]]
    contam_grouped = contam_rows.groupby(sample_column, observed=True)
    contam_intensity = (
        contam_grouped["intensity"]
        .sum()
        .astype(float)
        .reindex(total_intensity.index, fill_value=0.0)
    )
    n_contam = (
        contam_grouped["intensity"].size().astype(int).reindex(total_intensity.index, fill_value=0)
    )

    fraction = (contam_intensity / total_intensity.replace(0, np.nan)).fillna(0.0)

    df = pd.DataFrame(
        {
            "contaminant_intensity": contam_intensity,
            "total_intensity": total_intensity,
            "contaminant_fraction": fraction,
            "n_contaminant_precursors": n_contam,
            "n_total_precursors": n_total,
            "is_flagged": fraction > flag_fraction_threshold,
        }
    )
    df.index.name = "sample"

    provenance: dict[str, Any] = {
        "n_samples": len(df),
        "protein_column": protein_column,
        "intensity_column": intensity_column,
        "contaminant_prefixes": list(contaminant_prefixes),
        "n_fasta_accessions": len(accession_set),
        "contaminants_fasta": str(contaminants_fasta) if contaminants_fasta else None,
        "require_all_contaminant": bool(require_all_contaminant),
        "flag_fraction_threshold": float(flag_fraction_threshold),
        "n_flagged": int(df["is_flagged"].sum()),
        "cohort_median_fraction": float(fraction.median()) if len(fraction) else float("nan"),
        "sample_column": sample_column,
        "precursor_column": precursor_column,
    }

    if acquisition_queue is not None:
        df = _attach_queue_simple(df, acquisition_queue, provenance=provenance)

    df.attrs["provenance"] = provenance
    return df


def _validate_columns(psm_df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in psm_df.columns]
    if missing:
        raise KeyError(
            f"psm_df is missing required columns: {missing}.  Have: "
            f"{list(psm_df.columns)[:15]}... "
            "For DIA-NN input, the RT column is typically 'RT' -- pass "
            "rt_column='RT' explicitly."
        )


def _null_sample_row(sample: str, *, n: int = 0) -> dict[str, Any]:
    return {
        "sample": sample,
        "n_shared_precursors": int(n),
        "median_rt": float("nan"),
        "median_offset": float("nan"),
        "iqr_offset": float("nan"),
        "gradient_slope": float("nan"),
        "is_flagged": False,
    }


def _empty_drift_frame(
    *,
    samples: list[str],
    queue: pd.DataFrame | None,
) -> pd.DataFrame:
    """When no shared precursors exist, return an all-null frame."""
    rows = [_null_sample_row(s) for s in samples]
    df = pd.DataFrame(rows).set_index("sample")
    if queue is not None:
        df = _attach_queue(df, queue, provenance={})
    df.attrs["provenance"] = {
        "n_samples_input": len(samples),
        "n_samples_with_drift": 0,
        "n_shared_precursors_total": 0,
    }
    return df


def _linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of y ~ x with sensible NaN + zero-variance handling."""
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return float("nan")
    x_f = x[finite]
    y_f = y[finite]
    x_var = float(np.var(x_f))
    if x_var == 0.0:
        return float("nan")
    slope = float(np.cov(x_f, y_f, ddof=0)[0, 1] / x_var)
    return slope


def _attach_queue_simple(
    df: pd.DataFrame,
    queue: pd.DataFrame,
    *,
    provenance: dict[str, Any],
) -> pd.DataFrame:
    """Attach injection_order to a per-sample metric DataFrame.

    Simpler than :func:`_attach_queue`: no Spearman / slope /
    is_early_batch columns.  For metrics where the panel just wants to
    plot vs injection_order without deriving batch-level drift stats.
    """
    required = {"sample", "injection_order"}
    missing = required - set(queue.columns)
    if missing:
        raise ValueError(
            f"acquisition_queue is missing required columns: {sorted(missing)}. "
            "Use alphaphos.qc.load_acquisition_queue to build it, or "
            "construct a DataFrame with columns 'sample' and 'injection_order'."
        )
    queue_index = queue.set_index("sample")
    if not queue_index.index.is_unique:
        dups = queue_index.index[queue_index.index.duplicated()].tolist()
        raise ValueError(f"acquisition_queue has duplicate sample IDs: {dups[:3]}")

    matched = df.index.intersection(queue_index.index)
    unmatched = df.index.difference(queue_index.index).tolist()

    df = df.copy()
    df["injection_order"] = queue_index.loc[matched, "injection_order"].reindex(df.index)
    provenance["n_queue_samples"] = len(queue)
    provenance["n_matched_queue"] = len(matched)
    provenance["n_unmatched_psm_samples"] = len(unmatched)
    provenance["unmatched_psm_samples"] = unmatched[:10]
    return df


def _attach_queue(
    df: pd.DataFrame,
    queue: pd.DataFrame,
    *,
    provenance: dict[str, Any],
) -> pd.DataFrame:
    """Attach injection_order + global-drift stats via exact sample match."""
    required = {"sample", "injection_order"}
    missing = required - set(queue.columns)
    if missing:
        raise ValueError(
            f"acquisition_queue is missing required columns: {sorted(missing)}. "
            "Use alphaphos.qc.load_acquisition_queue to build it, or "
            "construct a DataFrame with columns 'sample' and 'injection_order'."
        )
    queue_index = queue.set_index("sample")
    if not queue_index.index.is_unique:
        dups = queue_index.index[queue_index.index.duplicated()].tolist()
        raise ValueError(f"acquisition_queue has duplicate sample IDs: {dups[:3]}")

    matched = df.index.intersection(queue_index.index)
    unmatched = df.index.difference(queue_index.index).tolist()

    df["injection_order"] = queue_index.loc[matched, "injection_order"].reindex(df.index)
    if "acquisition_time" in queue.columns:
        df["acquisition_time"] = queue_index.loc[matched, "acquisition_time"].reindex(df.index)

    # Global slope / correlation of median_offset vs injection_order
    joined = df.loc[matched, ["median_offset", "injection_order"]].dropna()
    if len(joined) >= 3:
        slope = _linear_slope(
            joined["injection_order"].to_numpy(dtype=float),
            joined["median_offset"].to_numpy(dtype=float),
        )
        pearson_r = float(np.corrcoef(joined["injection_order"], joined["median_offset"])[0, 1])
        # Spearman via rankdata to avoid scipy dep
        from scipy.stats import spearmanr

        spearman = spearmanr(joined["injection_order"], joined["median_offset"])
        provenance["pearson_r_order_vs_offset"] = float(pearson_r)
        provenance["spearman_r_order_vs_offset"] = float(spearman.statistic)
        provenance["spearman_p_value"] = float(spearman.pvalue)
    else:
        slope = float("nan")
        provenance["pearson_r_order_vs_offset"] = float("nan")
        provenance["spearman_r_order_vs_offset"] = float("nan")
        provenance["spearman_p_value"] = float("nan")

    df["offset_slope_vs_order"] = slope
    if df["injection_order"].notna().any():
        median_order = float(df["injection_order"].median())
        df["is_early_batch"] = df["injection_order"] < median_order
    else:
        df["is_early_batch"] = False

    provenance["n_queue_samples"] = len(queue)
    provenance["n_matched_queue"] = len(matched)
    provenance["n_unmatched_psm_samples"] = len(unmatched)
    provenance["unmatched_psm_samples"] = unmatched[:10]  # cap for readability
    return df
