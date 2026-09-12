"""Spectronaut PSM-level report reader.

Reads a Spectronaut Normal-report export (Parquet or TSV) into a DataFrame
ready for the collapse pipeline. Two design goals:

1. **Minimal memory footprint** -- the reader prunes columns AT LOAD TIME
   using the pyarrow / pandas ``columns=`` (parquet) or ``usecols=`` (TSV)
   argument. A raw Spectronaut Normal report typically has 30-50+ columns;
   we consume ~15. Skipping the rest at read time avoids materializing
   them into memory at all. Typical saving: 2-3x on large reports.

2. **Just-read-and-normalize** -- the reader applies BOUNDARY filters
   (decoy drop, contaminant drop, q-value cutoffs) and normalizes column
   names to the dotted convention. It does NOT pick the quant column and
   does NOT run top-N attribution -- those belong to :mod:`collapse` where
   they can be tuned via ``advanced`` and where a user has a chance to
   ``.head()`` the raw PSM output first.

Configuration via a settings dict, mirroring the pattern in
:mod:`alphaphos.preprocess.collapse`::

    import alphaphos as ap

    # 1) Defaults -- decoy drop on, contaminants dropped, no q-value cutoffs
    psm = ap.read_spectronaut("report.parquet")

    # 2) Selective override
    psm = ap.read_spectronaut(
        "report.parquet",
        advanced={"eg_qvalue_max": 0.01, "drop_contaminants": False},
    )

    # 3) Full customization -- inspect + edit the defaults
    ios = dict(ap.DEFAULT_IO_SETTINGS)
    ios["drop_decoys"] = False
    psm = ap.read_spectronaut("report.parquet", advanced=ios)

Unknown keys in ``advanced`` raise, so typos surface immediately.

The returned DataFrame carries ``.attrs`` populated with lineage counts
(rows loaded, rows after each filter, source path, engine).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from alphaphos.constants import (
    COL_EG_IS_DECOY,
    COL_EG_QVALUE,
    COL_PG_QVALUE,
)
from alphaphos.io.contaminants import DEFAULT_CONTAMINANT_PREFIXES, filter_contaminants
from alphaphos.io.schemas import (
    REQUIRED_COLUMNS,
    all_needed_columns,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public defaults
# ---------------------------------------------------------------------------


DEFAULT_IO_SETTINGS: dict[str, Any] = {
    "drop_decoys": True,  # drop EG.IsDecoy==True rows if col present
    "drop_contaminants": True,  # drop rows whose PG is entirely contam
    "contaminants_fasta": None,  # None -> bundled MaxQuant fasta
    "contaminant_prefixes": DEFAULT_CONTAMINANT_PREFIXES,
    "eg_qvalue_max": None,  # drop EG.Qvalue > threshold if set
    "pg_qvalue_max": None,  # drop PG.Qvalue > threshold if set
}


# ---------------------------------------------------------------------------
# Settings resolution
# ---------------------------------------------------------------------------


def resolve_io_settings(advanced: dict[str, Any] | None) -> dict[str, Any]:
    """Merge ``advanced`` overrides on top of :data:`DEFAULT_IO_SETTINGS`.

    Unknown keys and bad values raise ``ValueError`` naming the allowed set.
    Returns a fresh dict.
    """
    settings = dict(DEFAULT_IO_SETTINGS)
    if advanced is None:
        return settings
    if not isinstance(advanced, dict):
        raise TypeError(f"'advanced' must be a dict or None, got {type(advanced).__name__}")
    unknown = set(advanced) - set(DEFAULT_IO_SETTINGS)
    if unknown:
        raise ValueError(
            f"Unknown keys in 'advanced': {sorted(unknown)}. "
            f"Allowed: {sorted(DEFAULT_IO_SETTINGS)}."
        )
    settings.update(advanced)

    # Type / range checks (fail-fast).
    for bool_key in ("drop_decoys", "drop_contaminants"):
        if not isinstance(settings[bool_key], bool):
            raise ValueError(f"{bool_key} must be bool, got {type(settings[bool_key]).__name__}")
    for prob_key in ("eg_qvalue_max", "pg_qvalue_max"):
        v = settings[prob_key]
        # bool is an int subclass -- reject it explicitly so True doesn't pass as 1.0.
        if v is not None and (
            isinstance(v, bool) or not (isinstance(v, (int, float)) and 0 <= float(v) <= 1)
        ):
            raise ValueError(f"{prob_key} must be None or a float in [0, 1], got {v!r}")
    prefixes = settings["contaminant_prefixes"]
    if not isinstance(prefixes, (tuple, list)) or not all(isinstance(x, str) for x in prefixes):
        raise ValueError("contaminant_prefixes must be a tuple / list of strings")

    return settings


# ---------------------------------------------------------------------------
# Column-name normalization
# ---------------------------------------------------------------------------


def _to_dotted_column(name: str) -> str:
    """Spectronaut underscore-form column name -> dotted form.

    Spectronaut TSV exports use ``R.FileName`` style; parquet exports often
    use ``R_FileName`` style. Downstream code expects the dotted form.

    Examples
    --------
    >>> _to_dotted_column("R_FileName")
    'R.FileName'
    >>> _to_dotted_column("EG_TotalQuantity_(Settings)")
    'EG.TotalQuantity (Settings)'
    >>> _to_dotted_column("R.FileName")   # already dotted -- passthrough
    'R.FileName'
    """
    if "_" not in name or name.startswith("_"):
        return name
    # First underscore is the R_/PG_/PEP_/EG_/FG_ prefix separator.
    i = name.index("_")
    out = name[:i] + "." + name[i + 1 :]
    # Any remaining "_(" pattern is Spectronaut's flag suffix separator.
    return out.replace("_(", " (")


# ---------------------------------------------------------------------------
# Column scanning (schema-only, no data materialized)
# ---------------------------------------------------------------------------


def _scan_available_columns(path: Path, engine: str) -> tuple[dict[str, str], int]:
    """Return ``{dotted_name: raw_column_name}`` for columns we care about.

    Reads ONLY the file schema (parquet) or first header line (TSV) --
    no data is materialized. Filters to the intersection of what the file
    contains and :func:`all_needed_columns(engine)`. Handles both dotted
    and underscored raw column-name conventions.

    Returns
    -------
    (mapping, n_total_cols)
        ``mapping`` keys are dotted (canonical) column names; values are the
        RAW column names in the file, so the caller can pass them straight to
        ``pd.read_parquet(columns=...)`` or ``pd.read_csv(usecols=...)``.
        ``n_total_cols`` is the number of columns in the file before pruning.
    """
    if path.suffix.lower() == ".parquet":
        raw_names = pq.read_schema(str(path)).names
    else:
        # TSV / TXT: read only the header line to enumerate columns.
        # "utf-8-sig" transparently strips a leading BOM (Windows exports).
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            header = f.readline().rstrip("\r\n").split("\t")
        raw_names = header

    needed = all_needed_columns(engine)
    mapping: dict[str, str] = {}
    for raw in raw_names:
        dotted = _to_dotted_column(raw)
        if dotted in needed:
            # Only take the first occurrence if a file has both raw and
            # dotted variants of the same column (Spectronaut sometimes ships
            # duplicate columns after a re-export). Deterministic tie-break.
            mapping.setdefault(dotted, raw)
    return mapping, len(raw_names)


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce Spectronaut string-typed numeric columns to float.

    Some Spectronaut versions export q-values, EG.CScore etc. as strings
    (e.g. ``"2.97e-20"``). We coerce these so filters work reliably.
    """
    for col in (COL_EG_QVALUE, COL_PG_QVALUE):
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


_DECOY_TRUE = frozenset({"true", "1"})
_DECOY_FALSE = frozenset({"false", "0", ""})


def _decoy_mask(series: pd.Series) -> np.ndarray:
    """Return a boolean array that is True where the row is a decoy.

    ``EG.IsDecoy`` is a proper bool column in most exports, but some TSV /
    parquet exports carry it as strings (``"True"``/``"False"``) or as
    object dtype when NaNs are mixed in. ``Series.astype(bool)`` on those
    maps the *string* ``"False"`` (and NaN) to ``True`` and would drop every
    row -- so parse explicitly. Missing values count as not-decoy.
    """
    if series.dtype == bool:
        return series.to_numpy()
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).to_numpy() != 0
    text = series.astype("string").str.strip().str.lower()
    bad = text.notna() & ~text.isin(_DECOY_TRUE | _DECOY_FALSE)
    if bad.any():
        raise ValueError(
            f"Unrecognized values in '{COL_EG_IS_DECOY}': "
            f"{sorted(text[bad].unique().tolist())[:5]}. Expected True/False."
        )
    return text.isin(_DECOY_TRUE).fillna(False).to_numpy(dtype=bool)


# ---------------------------------------------------------------------------
# Public reader
# ---------------------------------------------------------------------------


def read_psm(
    path: str | Path,
    *,
    advanced: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Read a Spectronaut Normal-report PSM file.

    Prunes columns AT READ TIME to just what alphaPhos needs (union of
    required + optional + all quant candidates), applies the configured
    boundary filters, and returns a normalized DataFrame with lineage
    counts in ``.attrs``.

    Parameters
    ----------
    path : str | Path
        Path to a ``.parquet`` or ``.tsv``/``.txt`` Spectronaut Normal report.
    advanced : dict, optional
        Overrides for :data:`DEFAULT_IO_SETTINGS`. Unknown keys raise.

    Returns
    -------
    pandas.DataFrame
        PSM rows with dotted column names. All Spectronaut quant column
        variants that were present in the file are preserved (collapse
        picks one later). ``df.attrs`` records lineage::

            source_path            -- str, absolute path read
            engine                 -- "SN"
            n_rows_loaded          -- int, rows before any filter
            n_rows_after_decoys    -- int, only if drop_decoys=True
            n_rows_after_qvalue    -- int, only if q-value cutoffs set
            n_rows_after_contaminants -- int, only if drop_contaminants=True
            n_rows_returned        -- int, final row count
            columns_read           -- list[str], dotted names actually loaded
            columns_dropped        -- int, count of columns pruned at read time

    Raises
    ------
    FileNotFoundError
        If ``path`` doesn't exist.
    ValueError
        On bad settings or missing required columns.
    """
    settings = resolve_io_settings(advanced)
    engine = "SN"

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Spectronaut report not found: {p}")

    # ------- column pruning: figure out what to read before reading anything -------
    col_map, n_total_cols = _scan_available_columns(p, engine)
    if not col_map:
        raise ValueError(
            f"No usable columns found in {p}. Is this actually a Spectronaut "
            f"Normal report? Expected e.g. one of {sorted(REQUIRED_COLUMNS[engine])}."
        )

    raw_cols_to_read = list(col_map.values())
    n_dropped_cols = n_total_cols - len(raw_cols_to_read)

    # ------- load, materializing only the pruned column set -------
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p, columns=raw_cols_to_read)
    elif p.suffix.lower() in (".tsv", ".txt"):
        df = pd.read_csv(
            p, sep="\t", usecols=raw_cols_to_read, low_memory=False, encoding="utf-8-sig"
        )
    else:
        raise ValueError(
            f"Unsupported file extension {p.suffix!r}; expected .parquet, .tsv, or .txt"
        )

    n_loaded = len(df)

    # ------- normalize columns to dotted form -------
    df = df.rename(columns={raw: dotted for dotted, raw in col_map.items()})
    df = _coerce_dtypes(df)

    # ------- validate required columns arrived -------
    missing = set(REQUIRED_COLUMNS[engine]) - set(df.columns)
    if missing:
        raise ValueError(
            f"Spectronaut report is missing required column(s): {sorted(missing)}. "
            f"Present columns (dotted): {sorted(df.columns)}"
        )

    # ------- boundary filters -------
    n_after_decoys = None
    if settings["drop_decoys"] and COL_EG_IS_DECOY in df.columns:
        df = df.loc[~_decoy_mask(df[COL_EG_IS_DECOY])]
        n_after_decoys = len(df)
        logger.info("Dropped decoys: %d rows remaining.", n_after_decoys)

    n_after_qvalue = None
    if settings["eg_qvalue_max"] is not None and COL_EG_QVALUE in df.columns:
        df = df.loc[df[COL_EG_QVALUE].fillna(np.inf) <= settings["eg_qvalue_max"]]
        n_after_qvalue = len(df)
    if settings["pg_qvalue_max"] is not None and COL_PG_QVALUE in df.columns:
        df = df.loc[df[COL_PG_QVALUE].fillna(np.inf) <= settings["pg_qvalue_max"]]
        n_after_qvalue = len(df)

    df = df.reset_index(drop=True)

    n_after_contam = None
    if settings["drop_contaminants"]:
        df = filter_contaminants(
            df,
            contaminants_fasta=settings["contaminants_fasta"],
            prefix_patterns=tuple(settings["contaminant_prefixes"]),
        )
        n_after_contam = len(df)
        logger.info("Dropped contaminants: %d rows remaining.", n_after_contam)

    # ------- stamp lineage -------
    df.attrs["source_path"] = str(p)
    df.attrs["engine"] = engine
    df.attrs["n_rows_loaded"] = n_loaded
    if n_after_decoys is not None:
        df.attrs["n_rows_after_decoys"] = n_after_decoys
    if n_after_qvalue is not None:
        df.attrs["n_rows_after_qvalue"] = n_after_qvalue
    if n_after_contam is not None:
        df.attrs["n_rows_after_contaminants"] = n_after_contam
    df.attrs["n_rows_returned"] = len(df)
    df.attrs["columns_read"] = sorted(col_map)
    df.attrs["columns_dropped"] = n_dropped_cols

    logger.info(
        "read_spectronaut(%s): %d rows, %d cols kept / %d dropped at read time.",
        p.name,
        len(df),
        len(col_map),
        n_dropped_cols,
    )
    return df


__all__ = [
    "DEFAULT_IO_SETTINGS",
    "resolve_io_settings",
    "read_psm",
]
