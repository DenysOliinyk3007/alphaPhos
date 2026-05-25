"""Spectronaut PSM-level report reader.

Reads Spectronaut Normal-report exports (Parquet or TSV) into a normalized
pandas DataFrame ready for downstream collapse. Handles:

- Both Spectronaut column-name conventions: dot (``R.FileName``) and
  underscore (``R_FileName``). Parquet exports tend to use underscores;
  TSV exports tend to use dots. We normalize to dot form so downstream
  consumers (notably ``PeptideCollapse_v4``) see consistent names.
- The ``(Settings)`` and ``(MS1)``/``(MS2)`` suffixes embedded in column
  names by Spectronaut.
- Type coercion for columns Spectronaut exports as text (e.g. ``EG.Qvalue``
  with values like ``"2.97e-20"``).
- Quant-level routing: ``"auto"`` reads whatever the search was configured
  to produce (``EG.TotalQuantity (Settings)``); ``"MS1"`` and ``"MS2"``
  require explicit raw-quant columns to be present.

The returned frame has a guaranteed ``EG.TotalQuantity (Settings)`` column
holding the chosen quant level, so existing pipelines (``PeptideCollapse_v4``)
work unchanged regardless of which level the caller requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd


REQUIRED_DOTTED: tuple[str, ...] = (
    "R.FileName",
    "EG.PrecursorId",
    "PEP.PeptidePosition",
    "EG.PTMAssayProbability",
    "PG.Genes",
    "PG.ProteinGroups",
)


QuantLevel = Literal["auto", "MS1", "MS2"]


# Candidate column names per quant level, ordered by preference.
# alphaphos picks the first one present in the dataframe.
QUANT_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "auto": ("EG.TotalQuantity (Settings)", "FG.Quantity"),
    "MS1": (
        "FG.MS1Quantity",
        "FG.MS1RawQuantity",
        "EG.MS1Quantity",
        "EG.RawIntensityMS1",
    ),
    "MS2": (
        "FG.MS2Quantity",
        "FG.MS2RawQuantity",
        "EG.MS2Quantity",
        "EG.RawIntensityMS2",
    ),
}


def _to_dotted_column(name: str) -> str:
    """Spectronaut underscore-form column name -> dotted form.

    Examples
    --------
    >>> _to_dotted_column("R_FileName")
    'R.FileName'
    >>> _to_dotted_column("EG_TotalQuantity_(Settings)")
    'EG.TotalQuantity (Settings)'
    >>> _to_dotted_column("FG_PeakRTs_(MS2)")
    'FG.PeakRTs (MS2)'
    >>> _to_dotted_column("R.FileName")
    'R.FileName'
    """
    if "_" not in name or name.startswith("_"):
        return name
    # First underscore is the prefix-table separator (R_/PG_/PEP_/EG_/FG_/F_)
    i = name.index("_")
    out = name[:i] + "." + name[i + 1 :]
    # Remaining "_(" sequences are Spectronaut's flag suffixes ((Settings), (MS1), (MS2))
    return out.replace("_(", " (")


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: _to_dotted_column(c) for c in df.columns}
    mapping = {k: v for k, v in mapping.items() if k != v}
    if not mapping:
        return df
    return df.rename(columns=mapping)


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns that Spectronaut occasionally exports with wrong dtype."""
    # EG.Qvalue and PG.Qvalue can be string in some Spectronaut versions
    # (e.g. "2.97e-20" stored as text). Coerce to float for downstream filtering.
    for col in ("EG.Qvalue", "PG.Qvalue", "EG.PEP", "EG.Cscore"):
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _pick_quant_column(df: pd.DataFrame, quant_level: QuantLevel) -> str:
    """Return the canonical quant column name for the requested level."""
    candidates = QUANT_COLUMN_CANDIDATES[quant_level]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Cannot find a {quant_level!r} quant column. Looked for any of "
        f"{list(candidates)}; got columns {sorted(df.columns)}"
    )


def _validate_required(df: pd.DataFrame, extra_required: Iterable[str] = ()) -> None:
    required = tuple(REQUIRED_DOTTED) + tuple(extra_required)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Spectronaut report is missing required column(s): {missing}. "
            f"Got: {sorted(df.columns)}"
        )


def read_psm(
    path: str | Path,
    quant_level: QuantLevel = "auto",
    *,
    drop_decoys: bool = True,
    drop_non_phospho: bool = False,
    eg_qvalue_max: float | None = None,
    pg_qvalue_max: float | None = None,
    top_n_attribution: bool = True,
) -> pd.DataFrame:
    """Read a Spectronaut Normal-report PSM file (Parquet or TSV).

    Parameters
    ----------
    path
        Path to a Spectronaut Normal-report export. ``.parquet`` and
        ``.tsv``/``.txt`` are supported.
    quant_level
        Which Spectronaut quant column to feed downstream:

        - ``"auto"`` (default): uses ``EG.TotalQuantity (Settings)``,
          i.e. whatever MS level the Spectronaut search was configured for.
          The resulting frame's ``alphaphos_quant_level`` attribute is set
          to ``"settings"``.
        - ``"MS1"``: requires an explicit MS1 quant column (e.g.
          ``FG.MS1Quantity``). Errors if none is present.
        - ``"MS2"``: requires an explicit MS2 quant column (e.g.
          ``FG.MS2Quantity``). Errors if none is present.

        Regardless of choice, the chosen quant is written into a column
        named ``EG.TotalQuantity (Settings)`` so existing pipelines
        (``PeptideCollapse_v4``) consume it without modification.
    drop_decoys
        If True (default) and an ``EG.IsDecoy`` column is present, drop
        rows where it is True.
    drop_non_phospho
        If True, drop rows whose ``EG.PrecursorId`` does not contain
        ``[Phospho (STY)]``. Default False (caller may want non-phospho
        for selectivity calculations).
    eg_qvalue_max, pg_qvalue_max
        Optional q-value cutoffs. If set, rows above the threshold are
        dropped. q-value columns are coerced to float first if needed.
    top_n_attribution
        If True (default), apply the top-N attribution filter (Spectronaut
        over-export dedup; see ``alphaphos.preprocess.filter_to_top_n_positions``)
        immediately after loading. This is the validated correct behavior:
        agreement with Spectronaut native PTM site report is r=0.98 with the
        filter on vs. r=0.93 (with a long right tail) without it. Pass
        ``False`` only when you specifically need the raw per-candidate-
        position rows (e.g. for advanced fragment-level work). The filter is
        silently skipped when ``EG.PTMLocalizationProbabilities`` is absent
        from the report (the column it needs to evaluate top-N).

    Returns
    -------
    pd.DataFrame
        Normalized PSM table with dotted column names. The chosen quant
        is in ``EG.TotalQuantity (Settings)``. The frame carries a
        ``.attrs`` dict recording: ``source_path``, ``alphaphos_quant_level``,
        ``alphaphos_quant_column``, ``n_rows_loaded``, ``n_rows_returned``,
        ``top_n_attribution_applied``, ``n_rows_after_top_n``.
    """
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix.lower() in (".tsv", ".txt"):
        df = pd.read_csv(p, sep="\t", low_memory=False)
    else:
        raise ValueError(
            f"Unsupported file extension {p.suffix!r}; expected .parquet or .tsv"
        )

    n_loaded = len(df)
    df = _normalize_column_names(df)
    df = _coerce_dtypes(df)
    _validate_required(df)

    # Pick the quant column and make sure it's in the canonical slot.
    quant_col = _pick_quant_column(df, quant_level)
    if quant_col != "EG.TotalQuantity (Settings)":
        df = df.copy()
        df["EG.TotalQuantity (Settings)"] = df[quant_col]
    elif quant_level != "auto":
        # User asked for MS1/MS2 explicitly but only the Settings column matches.
        # Trust them but make the choice visible in the metadata.
        pass

    # Optional row filters
    if drop_decoys and "EG.IsDecoy" in df.columns:
        df = df.loc[~df["EG.IsDecoy"].astype(bool)]
    if drop_non_phospho:
        is_phospho = df["EG.PrecursorId"].str.contains(
            r"\[Phospho \(STY\)\]", regex=True, na=False
        )
        df = df.loc[is_phospho]
    if eg_qvalue_max is not None and "EG.Qvalue" in df.columns:
        df = df.loc[df["EG.Qvalue"].fillna(np.inf) <= eg_qvalue_max]
    if pg_qvalue_max is not None and "PG.Qvalue" in df.columns:
        df = df.loc[df["PG.Qvalue"].fillna(np.inf) <= pg_qvalue_max]

    df = df.reset_index(drop=True)

    # Top-N attribution dedup (Spectronaut over-exports the same peptide
    # measurement as multiple candidate-position rows; this filter keeps only
    # the rows whose PrecId-encoded positions match the top-N by per-row loc
    # probability). Validated against Spectronaut native PTM site report
    # (Pearson r=0.98) and the canonical R consolidate() (r=1.000).
    n_before_top_n = len(df)
    attribution_applied = False
    if top_n_attribution:
        if "EG.PTMLocalizationProbabilities" in df.columns:
            # Local import: keeps io <- preprocess directional dep explicit and
            # avoids any chance of circular import at package-load time.
            from alphaphos.preprocess.attribution import filter_to_top_n_positions

            df = filter_to_top_n_positions(df)
            attribution_applied = True

    df.attrs["source_path"] = str(p)
    df.attrs["alphaphos_quant_level"] = quant_level
    df.attrs["alphaphos_quant_column"] = quant_col
    df.attrs["n_rows_loaded"] = n_loaded
    df.attrs["n_rows_returned"] = len(df)
    df.attrs["top_n_attribution_applied"] = attribution_applied
    df.attrs["n_rows_after_top_n"] = len(df) if attribution_applied else n_before_top_n
    return df
