"""FragPipe DIA phospho-site reader.

Reads FragPipe's site-level abundance file
(``abundance_single-site_MS{1,2}quant_{None,Norm}.tsv``) and returns an
already-collapsed ``AnnData``. Unlike the Spectronaut and DIA-NN readers,
this one **bypasses** :func:`alphaphos.collapse_sites` because FragPipe's
DIA workflow already emits site-level matrices via its own IonQuant +
PTM-Prophet aggregation. Trusting the FragPipe developers on collapse.

Two design goals:

1. **Trust FragPipe's site collapse.** The output is ready to consume;
   we only parse the site identifiers and construct the AnnData wrapper.
2. **Match the AnnData contract of ``collapse_sites`` output as far as the
   input allows** so downstream tooling (QC dashboard, imputation, kinase
   annotation, diff-exp, dose-response) doesn't care which engine produced
   the AnnData. See "Contract gaps" below for what FragPipe cannot provide.

Input file layout (per FragPipe DIA docs)::

    Index    Gene   ProteinID   Peptide            SequenceWindow   Multiplicity   Best Localization   Best Scan   Best Precursor   <run1>   <run2>   ...
    P10644_S77  PRKAR1A  P10644  TDsREDEIsPPPPNPVVK KAGTRTDsREDEIsP  2              0.8982              ...         ...              123.4    98.7     ...

Where:

- ``Index`` = ``{ProteinID}_{aa}{position}`` (e.g. ``P10644_S77``).
- ``SequenceWindow`` = +/- 7 residues around the site, lowercase for
  phospho residues.
- ``Multiplicity`` = number of phospho on the source peptide.
- ``Best Localization`` = highest per-scan localization probability for
  this site across all scans (proxy for "Class I" filtering).

The output ``AnnData`` follows :func:`alphaphos.collapse_sites`'s contract:

- ``.X`` = ``(n_samples, n_sites)`` log2 intensity
- ``.layers["intensity_log2"]`` = same as ``.X``
- ``.var.index`` = full key ``"{ProteinID}|{Gene}|{aa}{pos}|M{mult}"``
- ``.var`` columns: ``short_key``, ``pg_key``, ``protein_group_id``,
  ``gene``, ``site_aa``, ``site_position``, ``multiplicity``,
  ``n_samples_detected``, ``best_localization``, ``max_loc_prob``
  (alias of ``best_localization``), ``sequence_window``, ``kinase_sequence``
- ``.obs.index`` = sample id (raw file basename, ``_uncalibrated`` stripped)
- ``.uns["alphaphos"]`` = ``version``, ``pipeline_params``, ``source_file``

Contract gaps (vs. ``collapse_sites``)
---------------------------------------

FragPipe reports ONE localization probability per site (``Best
Localization``), not one per run, so the per-run-derived columns cannot be
computed: ``mean_loc_prob``, ``min_loc_prob``, ``n_classI_samples``,
``fraction_classI``, ``classI_wilson_lb``, and ``layers["localization"]`` are
absent. ``UPD_seq`` is also absent (no modified-sequence string per site).
Consequences: :func:`alphaphos.recommend_pipeline` skips its Class-I step
on FragPipe data (use ``min_best_localization`` here instead), and
``localization_strategy="wilson"`` / :func:`alphaphos.apply_wilson_filter`
are not applicable.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path, PureWindowsPath
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from alphaphos._version import __version__ as _alphaphos_version
from alphaphos.constants import (
    FRAGPIPE_BEST_LOCALIZATION,
    FRAGPIPE_GENE,
    FRAGPIPE_INDEX,
    FRAGPIPE_META_COLUMNS,
    FRAGPIPE_MULTIPLICITY,
    FRAGPIPE_SEQUENCE_WINDOW,
    LAYER_INTENSITY_LOG2,
    OBS_CONDITION,
    OBS_SAMPLE,
    UNS_ALPHAPHOS,
    VAR_FULL_KEY,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public defaults
# ---------------------------------------------------------------------------


DEFAULT_FRAGPIPE_IO_SETTINGS: dict[str, Any] = {
    "quant_level": "MS2",  # "MS2" or "MS1" -- picks between abundance_single-site_MS{1,2}quant_*.tsv
    "normalized": False,  # True -> "_Norm" file variant, False -> "_None"
    "site_type": "single",  # "single" | "multi" -- abundance_{single,multi}-site_*.tsv
    "min_best_localization": 0.75,  # Class-I equivalent cutoff; None to disable
    "add_kinase_sequence": True,  # convert SequenceWindow -> alphaPhos kinase_sequence format
}


# ---------------------------------------------------------------------------
# Settings resolution
# ---------------------------------------------------------------------------


def resolve_fragpipe_io_settings(advanced: dict[str, Any] | None) -> dict[str, Any]:
    """Merge ``advanced`` overrides on top of :data:`DEFAULT_FRAGPIPE_IO_SETTINGS`.

    Unknown keys and bad values raise ``ValueError``. Returns a fresh dict.
    """
    settings = dict(DEFAULT_FRAGPIPE_IO_SETTINGS)
    if advanced is None:
        return settings
    if not isinstance(advanced, dict):
        raise TypeError(f"'advanced' must be a dict or None, got {type(advanced).__name__}")
    unknown = set(advanced) - set(DEFAULT_FRAGPIPE_IO_SETTINGS)
    if unknown:
        raise ValueError(
            f"Unknown keys in 'advanced': {sorted(unknown)}. "
            f"Allowed: {sorted(DEFAULT_FRAGPIPE_IO_SETTINGS)}."
        )
    settings.update(advanced)

    if settings["quant_level"] not in ("MS1", "MS2"):
        raise ValueError(f"quant_level must be 'MS1' or 'MS2', got {settings['quant_level']!r}")
    if settings["site_type"] not in ("single", "multi"):
        raise ValueError(f"site_type must be 'single' or 'multi', got {settings['site_type']!r}")
    if not isinstance(settings["normalized"], bool):
        raise ValueError(f"normalized must be bool, got {type(settings['normalized']).__name__}")
    if not isinstance(settings["add_kinase_sequence"], bool):
        raise ValueError(
            f"add_kinase_sequence must be bool, got {type(settings['add_kinase_sequence']).__name__}"
        )
    v = settings["min_best_localization"]
    if v is not None and not (isinstance(v, (int, float)) and 0 <= float(v) <= 1):
        raise ValueError(f"min_best_localization must be None or float in [0, 1], got {v!r}")
    return settings


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_abundance_file(path: str | Path, settings: dict[str, Any]) -> Path:
    """Return the concrete abundance-file path.

    If ``path`` is a specific file, returns it unchanged. If it's a
    directory, resolves to the file matching ``settings`` (quant_level,
    normalized, site_type).
    """
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        norm_tag = "Norm" if settings["normalized"] else "None"
        fname = (
            f"abundance_{settings['site_type']}-site_{settings['quant_level']}quant_{norm_tag}.tsv"
        )
        target = p / fname
        if not target.exists():
            raise FileNotFoundError(
                f"Expected FragPipe abundance file not found: {target}. "
                f"Directory contents: {sorted(x.name for x in p.iterdir())[:15]}..."
            )
        return target
    raise FileNotFoundError(f"Path does not exist: {p}")


# ---------------------------------------------------------------------------
# Site-key parsing helpers (pure functions)
# ---------------------------------------------------------------------------


_INDEX_RE = re.compile(r"^(?P<prot>.+)_(?P<aa>[A-Za-z])(?P<pos>\d+)$")


def parse_fragpipe_index(idx: str) -> tuple[str, str, int] | None:
    """Parse ``"P10644_S77"`` -> ``("P10644", "S", 77)``.

    Returns ``None`` for anything that doesn't match the expected format.
    """
    if not isinstance(idx, str):
        return None
    m = _INDEX_RE.match(idx)
    if m is None:
        return None
    return m.group("prot"), m.group("aa").upper(), int(m.group("pos"))


def sequence_window_to_kinase_sequence(sequence_window: str, target_aa: str) -> str:
    """Convert FragPipe's ``SequenceWindow`` to the alphaPhos kinase format.

    FragPipe SequenceWindow uses lowercase for phospho residues; the
    target site is at the middle of the window (position ``len//2``).

    Examples
    --------
    >>> sequence_window_to_kinase_sequence("KAGTRTDsREDEIsP", "S")
    '_KAGTRTD*S*REDEISP_'
    """
    if not isinstance(sequence_window, str) or len(sequence_window) < 3:
        return ""
    center_idx = len(sequence_window) // 2
    upper = sequence_window.upper()
    return (
        "_"
        + upper[:center_idx]
        + "*"
        + str(target_aa).upper()
        + "*"
        + upper[center_idx + 1 :]
        + "_"
    )


# ---------------------------------------------------------------------------
# Sample-column normalization
# ---------------------------------------------------------------------------


def _normalize_sample_col(col: str) -> str:
    """Strip path + common FragPipe suffixes from a sample column header.

    FragPipe writes the full raw-file path as the column header and is
    almost always run on Windows, so headers may use ``\\`` separators even
    when alphaPhos runs on macOS / Linux. ``PureWindowsPath`` splits on both
    ``\\`` and ``/`` regardless of host OS.

    Examples
    --------
    >>> _normalize_sample_col("V:/foo/20250721_run_01_uncalibrated.mzML")
    '20250721_run_01'
    >>> _normalize_sample_col(r"D:\\data\\20250721_run_01_uncalibrated.mzML")
    '20250721_run_01'
    """
    stem = PureWindowsPath(col).stem
    # Strip FragPipe's ``_uncalibrated`` suffix if present.
    if stem.endswith("_uncalibrated"):
        stem = stem[: -len("_uncalibrated")]
    return stem


# ---------------------------------------------------------------------------
# Public reader
# ---------------------------------------------------------------------------


def read_fragpipe_sites(
    path: str | Path,
    *,
    condition_df: pd.DataFrame | None = None,
    advanced: dict[str, Any] | None = None,
) -> ad.AnnData:
    """Read a FragPipe DIA site-abundance file into an ``AnnData``.

    Parameters
    ----------
    path : str | Path
        Either a FragPipe output DIRECTORY (in which case the abundance
        file is resolved from ``advanced``'s ``quant_level``, ``site_type``,
        ``normalized``) OR a specific ``abundance_*-site_*.tsv`` file.
    condition_df : DataFrame, optional
        Sample metadata. Must contain columns ``sample`` (matching the
        normalized sample id -- see :func:`_normalize_sample_col`) and
        ``condition``. Extra columns are joined into ``adata.obs``.
    advanced : dict, optional
        Overrides for :data:`DEFAULT_FRAGPIPE_IO_SETTINGS`. Unknown keys raise.

    Returns
    -------
    anndata.AnnData
        Shape ``(n_samples, n_sites)``:

        * ``.X`` = ``layers["intensity_log2"]`` (log2 intensity)
        * ``.var.index`` = ``"Protein|Gene|aa+pos|Mmult"``
        * ``.var`` = ``short_key``, ``pg_key``, ``protein_group_id``,
          ``gene``, ``site_aa``, ``site_position``, ``multiplicity``,
          ``n_samples_detected``, ``best_localization``, ``max_loc_prob``,
          ``sequence_window``, ``kinase_sequence``.  Per-run localization
          columns (``mean_loc_prob``, ``classI_wilson_lb``, ...) are NOT
          available -- see the module docstring's "Contract gaps".
        * ``.obs`` = ``condition`` (from ``condition_df`` if given)
        * ``.uns["alphaphos"]`` = ``version``, ``pipeline_params``,
          ``source_file``

    Raises
    ------
    FileNotFoundError
        If ``path`` doesn't exist or the resolved abundance file is missing.
    ValueError
        If required columns are absent, the file has no numeric sample
        columns, two sample headers normalize to the same id, or
        ``Multiplicity`` has missing values.
    """
    settings = resolve_fragpipe_io_settings(advanced)
    abundance_path = resolve_abundance_file(path, settings)

    df = pd.read_csv(abundance_path, sep="\t", low_memory=False)
    n_loaded_rows = len(df)

    missing = set(FRAGPIPE_META_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"FragPipe abundance file is missing required column(s): {sorted(missing)}. "
            f"Present: {sorted(df.columns)[:20]}..."
        )

    # Optional Class-I filter on Best Localization.
    if settings["min_best_localization"] is not None:
        n_before = len(df)
        df = df[df[FRAGPIPE_BEST_LOCALIZATION].fillna(-1) >= settings["min_best_localization"]]
        logger.info(
            "Best Localization >= %.2f filter: %d -> %d rows.",
            settings["min_best_localization"],
            n_before,
            len(df),
        )

    # Parse the site keys.
    parsed = [parse_fragpipe_index(x) for x in df[FRAGPIPE_INDEX]]
    valid = [p is not None for p in parsed]
    n_unparseable = sum(1 for v in valid if not v)
    if n_unparseable:
        logger.warning(
            "Dropping %d row(s) with unparseable Index (expected 'ProteinID_AApos').",
            n_unparseable,
        )
        df = df.loc[valid].reset_index(drop=True)
        parsed = [p for p in parsed if p is not None]

    protein_ids = [p[0] for p in parsed]
    site_aas = [p[1] for p in parsed]
    site_positions = [p[2] for p in parsed]
    # Missing gene -> "" (not the string "nan") so keys stay well-formed.
    genes = df[FRAGPIPE_GENE].fillna("").astype(str).tolist()
    n_mult_missing = int(df[FRAGPIPE_MULTIPLICITY].isna().sum())
    if n_mult_missing:
        raise ValueError(
            f"{n_mult_missing} row(s) have a missing '{FRAGPIPE_MULTIPLICITY}' value; "
            "cannot build site keys."
        )
    mults = df[FRAGPIPE_MULTIPLICITY].astype(int).tolist()

    # Build canonical keys.
    full_keys = [
        f"{prot}|{gene}|{aa}{pos}|M{mult}"
        for prot, gene, aa, pos, mult in zip(
            protein_ids, genes, site_aas, site_positions, mults, strict=True
        )
    ]
    short_keys = [
        f"{gene}|{aa}{pos}|M{mult}"
        for gene, aa, pos, mult in zip(genes, site_aas, site_positions, mults, strict=True)
    ]
    pg_keys = [
        f"{prot}|{aa}{pos}|M{mult}"
        for prot, aa, pos, mult in zip(protein_ids, site_aas, site_positions, mults, strict=True)
    ]

    # Identify sample columns: everything not in the fixed metadata set that
    # is numeric. Other FragPipe versions may add extra text metadata columns
    # (e.g. a protein description); those are excluded with a warning rather
    # than crashing the float conversion.
    candidate_cols = [c for c in df.columns if c not in FRAGPIPE_META_COLUMNS]
    sample_cols_raw: list[str] = []
    numeric_cols: dict[str, pd.Series] = {}
    for c in candidate_cols:
        coerced = pd.to_numeric(df[c], errors="coerce")
        if df[c].notna().any() and coerced.isna().all():
            logger.warning(
                "Column %r is non-numeric and not a known FragPipe metadata column; "
                "excluding it from the sample matrix.",
                c,
            )
            continue
        sample_cols_raw.append(c)
        numeric_cols[c] = coerced
    if not sample_cols_raw:
        raise ValueError(
            "FragPipe abundance file has no numeric sample columns (after removing the "
            f"{len(FRAGPIPE_META_COLUMNS)} metadata columns)."
        )
    sample_ids = [_normalize_sample_col(c) for c in sample_cols_raw]

    # Normalization must be injective, otherwise obs.index is non-unique and
    # the condition_df join silently mis-assigns metadata.
    if len(set(sample_ids)) != len(sample_ids):
        collisions: dict[str, list[str]] = {}
        for raw_col, sid in zip(sample_cols_raw, sample_ids, strict=True):
            collisions.setdefault(sid, []).append(raw_col)
        dupes = {sid: cols for sid, cols in collisions.items() if len(cols) > 1}
        raise ValueError(
            "Sample column headers collide after normalization (same file stem in "
            f"different directories?): {dupes}"
        )

    # Linear intensities -> log2. Zeros / missing -> NaN.
    # copy=True: to_numpy() may hand back a read-only view; we mutate in place below.
    raw = pd.DataFrame(numeric_cols, columns=sample_cols_raw).to_numpy(
        dtype=float, na_value=np.nan, copy=True
    )
    raw[raw <= 0] = np.nan
    log2_intensity = np.log2(raw)

    # AnnData wants (obs, var) = (samples, sites). Our matrix is (sites, samples).
    X = log2_intensity.T.astype(np.float32, copy=False)

    var = pd.DataFrame(
        {
            "short_key": short_keys,
            "pg_key": pg_keys,
            "protein_group_id": protein_ids,
            "gene": genes,
            "site_aa": site_aas,
            "site_position": site_positions,
            "multiplicity": mults,
            # Same name as collapse_sites so downstream completeness helpers
            # find it. Counted on the log2 matrix (zeros already -> NaN).
            "n_samples_detected": np.sum(~np.isnan(log2_intensity), axis=1).astype(int),
            "best_localization": df[FRAGPIPE_BEST_LOCALIZATION].values,
            # FragPipe's "Best Localization" is the max over scans -- expose
            # it under the collapse_sites column name too.
            "max_loc_prob": df[FRAGPIPE_BEST_LOCALIZATION].values,
            "sequence_window": df[FRAGPIPE_SEQUENCE_WINDOW].values,
        },
        index=pd.Index(full_keys, name=VAR_FULL_KEY),
    )

    if settings["add_kinase_sequence"]:
        var["kinase_sequence"] = [
            sequence_window_to_kinase_sequence(sw, aa)
            for sw, aa in zip(var["sequence_window"], var["site_aa"], strict=True)
        ]

    obs = pd.DataFrame(index=pd.Index(sample_ids, name=OBS_SAMPLE))
    if condition_df is not None:
        cdf = condition_df.copy()
        if OBS_SAMPLE not in cdf.columns:
            raise KeyError("condition_df must contain a 'sample' column.")
        if OBS_CONDITION not in cdf.columns:
            raise KeyError("condition_df must contain a 'condition' column.")
        # Mirror preprocess.anndata.to_anndata: a duplicated sample row in
        # condition_df must not multiply obs rows.
        obs = obs.join(cdf.drop_duplicates(OBS_SAMPLE).set_index(OBS_SAMPLE), how="left")

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers[LAYER_INTENSITY_LOG2] = X.copy()

    adata.uns[UNS_ALPHAPHOS] = {
        "version": _alphaphos_version,
        "source": "fragpipe_site_matrix",
        "source_file": str(abundance_path),
        "pipeline_params": dict(settings),
        "stats": {
            "n_rows_loaded": n_loaded_rows,
            "n_sites_after_localization_filter": len(df),
            "n_unparseable_indexes": n_unparseable,
            "n_samples": len(sample_ids),
        },
    }

    logger.info(
        "read_fragpipe_sites(%s): %d samples x %d sites.",
        abundance_path.name,
        adata.n_obs,
        adata.n_vars,
    )
    return adata
