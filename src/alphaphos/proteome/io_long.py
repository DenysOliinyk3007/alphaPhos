"""Read Spectronaut's long (precursor-level) proteome report.

Parallel to :func:`alphaphos.read_spectronaut` but without the PTM-column
requirement.  Applies the same QC filters (decoys, contaminants, Q-value
thresholds) so the output has the same trust profile as the phospho path.

Output shape
------------
Precursor-level ``pd.DataFrame`` with columns (dotted convention):

- ``R.FileName``          -- sample identifier
- ``PG.ProteinGroups``    -- protein-group ID (semicolon-joined)
- ``PG.Genes``            -- gene symbols
- ``PG.Qvalue``           -- protein-group Q-value
- ``EG.PrecursorId``      -- precursor identifier
- ``EG.Qvalue``           -- precursor Q-value (coerced string -> float)
- ``EG.IsDecoy``          -- decoy flag
- ``PEP.StrippedSequence``-- peptide sequence
- ``<quant_col>``         -- resolved quantity column (see quant selection)

Hand the returned DataFrame to :func:`collapse_proteome` for aggregation to
a protein-level AnnData.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from alphaphos.preprocess.contaminants import (
    DEFAULT_CONTAMINANT_PREFIXES,
    filter_contaminants,
)

logger = logging.getLogger(__name__)

# The output columns we always keep (dotted names) when present.  Others in
# the source are dropped early to save memory.  ``EG.PrecursorId`` /
# ``EG.ModifiedSequence`` / ``EG.ModifiedPeptide`` are alternative precursor
# identifiers -- at least one must be present.
_KEEP_COLUMNS_DOTTED: tuple[str, ...] = (
    "R.FileName",
    "PG.ProteinGroups",
    "PG.Genes",
    "PG.Qvalue",
    "EG.PrecursorId",
    "EG.ModifiedSequence",
    "EG.ModifiedPeptide",
    "EG.Qvalue",
    "EG.GlobalPrecursorQvalue",
    "EG.IsDecoy",
    "PEP.StrippedSequence",
)

# Precursor-identifier column fallback order.  The reader picks the first
# present and stamps the choice into ``df.attrs["precursor_id_column"]``.
_PRECURSOR_ID_CANDIDATES: tuple[str, ...] = (
    "EG.PrecursorId",
    "EG.ModifiedSequence",
    "EG.ModifiedPeptide",
)

# Quant column candidates, in preference order per level.  Mirrors the phospho
# reader's chain so both paths pick the same quant when both are present.
_QUANT_CANDIDATES_DOTTED: dict[str, tuple[str, ...]] = {
    "auto": ("EG.TotalQuantity (Settings)", "FG.Quantity", "PG.Quantity"),
    "MS1": ("FG.MS1Quantity", "FG.MS1RawQuantity", "EG.MS1Quantity"),
    "MS2": ("FG.MS2Quantity", "FG.MS2RawQuantity", "EG.MS2Quantity"),
}
_FALLBACK_CHAIN: dict[str, tuple[str, ...]] = {
    "MS2": ("MS2", "MS1", "auto"),
    "MS1": ("MS1", "auto"),
    "auto": ("auto",),
}


def read_spectronaut_long(
    path: str | Path,
    *,
    quantification_level: str = "MS2",
    pg_qvalue_max: float | None = 0.01,
    eg_qvalue_max: float | None = 0.01,
    drop_decoys: bool = True,
    drop_contaminants: bool = True,
    contaminants_fasta: str | Path | None = None,
    contaminant_prefixes: tuple[str, ...] = DEFAULT_CONTAMINANT_PREFIXES,
    verbose: bool = False,
) -> pd.DataFrame:
    """Read a Spectronaut precursor-level proteome report → filtered DataFrame.

    Parameters
    ----------
    path
        Path to ``.parquet`` (preferred) or ``.tsv`` long-format report.
    quantification_level
        ``"MS2"`` (default), ``"MS1"``, or ``"auto"``.  A per-level candidate
        list is walked; the first present column is used.  Falls back down
        the chain (MS2 → MS1 → auto) if the requested level is unavailable.
    pg_qvalue_max
        Protein-group Q-value cutoff; rows with ``PG.Qvalue`` above this are
        dropped.  ``None`` disables the filter.  Default ``0.01``.
    eg_qvalue_max
        Precursor Q-value cutoff; rows with ``EG.Qvalue`` above this are
        dropped.  ``EG.Qvalue`` is coerced string → float (Spectronaut
        parquet exports sometimes ship it as string).  ``None`` disables.
        Default ``0.01``.
    drop_decoys
        Drop rows where ``EG.IsDecoy`` is True (if column present).
    drop_contaminants
        Drop rows whose protein group is entirely contaminants (via the
        bundled MaxQuant ``contaminants.fasta`` plus the prefix patterns
        below).  Reuses :func:`alphaphos.preprocess.filter_contaminants`.
    contaminants_fasta
        Override the bundled contaminants FASTA path.  ``None`` uses the
        bundled default.
    contaminant_prefixes
        Prefixes marking search-time-tagged contaminants (default matches
        MaxQuant / Spectronaut / common workflows).
    verbose
        Log per-stage row counts at INFO level.

    Returns
    -------
    pd.DataFrame
        Precursor-level, filtered.  ``.attrs`` carries a full audit:
        ``n_rows_loaded``, ``n_rows_after_decoy``, ``n_rows_after_pg_q``,
        ``n_rows_after_eg_q``, ``n_rows_after_contaminants``, ``n_rows_final``,
        ``resolved_quant_column``, ``requested_quant_level``.
    """
    path = Path(path)
    ext = path.suffix.lower()

    # 1. Scan available columns without reading data.
    if ext == ".parquet":
        raw_names = [f.name for f in pq.ParquetFile(path).schema_arrow]
    elif ext in {".tsv", ".txt"}:
        # Read only the header line
        with path.open("r", encoding="utf-8") as f:
            raw_names = f.readline().rstrip("\r\n").split("\t")
    else:
        raise ValueError(f"Unsupported extension {ext!r}; expected .parquet, .tsv, or .txt")

    raw_to_dotted = {r: _to_dotted(r) for r in raw_names}
    dotted_to_raw = {v: k for k, v in raw_to_dotted.items()}

    # 2. Resolve the quant column via the fallback chain.
    resolved_quant_dotted, level_used = _resolve_quant_column(
        dotted_to_raw, requested_level=quantification_level
    )

    # 3. Assemble the list of raw columns to read.
    keep_dotted = [*_KEEP_COLUMNS_DOTTED, resolved_quant_dotted]
    read_raw = [dotted_to_raw[d] for d in keep_dotted if d in dotted_to_raw]

    # Hard requirements: R.FileName, PG.ProteinGroups, and at least one of the
    # precursor-id candidates.  Everything else is optional.
    hard_required = {"R.FileName", "PG.ProteinGroups"}
    hard_missing = hard_required - set(dotted_to_raw)
    if hard_missing:
        raise ValueError(
            f"Required columns missing from {path.name}: {sorted(hard_missing)}. "
            f"Found columns: {sorted(dotted_to_raw)[:15]}..."
        )
    precursor_id_col = next((c for c in _PRECURSOR_ID_CANDIDATES if c in dotted_to_raw), None)
    if precursor_id_col is None:
        raise ValueError(
            f"None of the candidate precursor-id columns "
            f"{list(_PRECURSOR_ID_CANDIDATES)} are present in {path.name}. "
            "Cannot identify unique precursors."
        )
    logger.info(
        "read_spectronaut_long: using %r as the precursor identifier.",
        precursor_id_col,
    )

    # 4. Load only the needed columns.
    if ext == ".parquet":
        df = pd.read_parquet(path, columns=read_raw)
    else:
        df = pd.read_csv(path, sep="\t", usecols=read_raw, low_memory=False)
    # Rename to dotted.
    df = df.rename(columns=raw_to_dotted)
    df.attrs["n_rows_loaded"] = len(df)
    if verbose:
        logger.info("read_spectronaut_long: loaded %d rows from %s", len(df), path.name)

    # 5. Filter: decoys.
    if drop_decoys and "EG.IsDecoy" in df.columns:
        mask = df["EG.IsDecoy"].fillna(False).astype(bool)
        df = df.loc[~mask].copy()
    df.attrs["n_rows_after_decoy"] = len(df)
    if verbose and drop_decoys:
        logger.info("  after decoy drop: %d rows", len(df))

    # 6. Filter: PG.Qvalue.
    if pg_qvalue_max is not None and "PG.Qvalue" in df.columns:
        pg_q = pd.to_numeric(df["PG.Qvalue"], errors="coerce")
        mask = pg_q <= pg_qvalue_max
        # Rows with a NaN Q-value are conservatively kept (Spectronaut sometimes
        # omits it for high-confidence PGs) -- matches the phospho reader's
        # behaviour where a missing Q-value doesn't force a drop.
        df = df.loc[mask.fillna(True)].copy()
    df.attrs["n_rows_after_pg_q"] = len(df)
    if verbose and pg_qvalue_max is not None:
        logger.info("  after PG.Qvalue<=%.4g: %d rows", pg_qvalue_max, len(df))

    # 7. Filter: EG.Qvalue (coerce string -> float).
    if eg_qvalue_max is not None and "EG.Qvalue" in df.columns:
        eg_q = pd.to_numeric(df["EG.Qvalue"], errors="coerce")
        mask = eg_q <= eg_qvalue_max
        df = df.loc[mask.fillna(True)].copy()
    df.attrs["n_rows_after_eg_q"] = len(df)
    if verbose and eg_qvalue_max is not None:
        logger.info("  after EG.Qvalue<=%.4g: %d rows", eg_qvalue_max, len(df))

    # 8. Filter: contaminants (reuses the shared PSM-level helper).
    if drop_contaminants and "PG.ProteinGroups" in df.columns:
        df = filter_contaminants(
            df,
            contaminants_fasta=contaminants_fasta,
            prefix_patterns=tuple(contaminant_prefixes),
            protein_groups_column="PG.ProteinGroups",
            verbose=verbose,
        )
    df.attrs["n_rows_after_contaminants"] = len(df)

    # 9. Record provenance.
    df.attrs["n_rows_final"] = len(df)
    df.attrs["resolved_quant_column"] = resolved_quant_dotted
    df.attrs["precursor_id_column"] = precursor_id_col
    df.attrs["requested_quant_level"] = quantification_level
    df.attrs["quant_level_used"] = level_used
    df.attrs["path"] = str(path)

    logger.info(
        "read_spectronaut_long: %s -> %d rows after filters (loaded=%d, quant=%r, level=%r).",
        path.name,
        len(df),
        df.attrs["n_rows_loaded"],
        resolved_quant_dotted,
        level_used,
    )
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_dotted(name: str) -> str:
    """Spectronaut underscore-column name → dotted.  Mirrors the phospho reader.

    Examples
    --------
    >>> _to_dotted("R_FileName")
    'R.FileName'
    >>> _to_dotted("EG_TotalQuantity_(Settings)")
    'EG.TotalQuantity (Settings)'
    >>> _to_dotted("R.FileName")
    'R.FileName'
    """
    if "_" not in name or name.startswith("_"):
        return name
    i = name.index("_")
    out = name[:i] + "." + name[i + 1 :]
    return out.replace("_(", " (")


def _resolve_quant_column(
    dotted_to_raw: dict[str, str], *, requested_level: str
) -> tuple[str, str]:
    """Walk the fallback chain and return the first available quant column.

    Returns (dotted_column_name, level_actually_used).  Raises if nothing
    matches across the full chain.
    """
    if requested_level not in _FALLBACK_CHAIN:
        raise ValueError(
            f"quantification_level must be one of {sorted(_FALLBACK_CHAIN)}, "
            f"got {requested_level!r}"
        )
    for level in _FALLBACK_CHAIN[requested_level]:
        for candidate in _QUANT_CANDIDATES_DOTTED[level]:
            if candidate in dotted_to_raw:
                if level != requested_level:
                    logger.info(
                        "read_spectronaut_long: quantification_level=%r unavailable "
                        "in the input; falling back to level=%r via column %r.",
                        requested_level,
                        level,
                        candidate,
                    )
                return candidate, level
    tried: list[str] = []
    for lvl in _FALLBACK_CHAIN[requested_level]:
        tried.extend(_QUANT_CANDIDATES_DOTTED[lvl])
    raise ValueError(
        f"None of the candidate quant columns for level {requested_level!r} "
        f"(with fallback chain) were found in the input. Candidates tried: {tried}"
    )
