"""DIA-NN PSM-level report reader.

Reads a DIA-NN main report (``*.parquet`` or ``*.tsv``) and adapts it to
the Spectronaut-canonical PSM column schema so the same collapse pipeline
runs on both engines. Two design goals:

1. **Adapter, not reimplementation.** The core algorithmic logic (site key
   construction, aggregation, log2, noise floor, condition-aware Class-I
   masking) lives in :mod:`alphaphos.preprocess.collapse`. This module
   only rewires DIA-NN columns into the columns that collapse consumes.

2. **Minimal memory footprint.** Same column-pruning approach as
   :mod:`alphaphos.io.spectronaut` -- load only what
   :func:`alphaphos.io.schemas.all_needed_columns("Diann")` requires.

Column mapping to Spectronaut canonical schema
----------------------------------------------

    R.FileName                        <-  Run
    EG.PrecursorId                    <-  Modified.Sequence + Precursor.Charge
    PEP.PeptidePosition               <-  Protein.Sites + Modified.Sequence
    EG.PTMAssayProbability            <-  PTM.Site.Confidence
    EG.PTMLocalizationProbabilities   <-  Site.Occupancy.Probabilities
    PG.Genes                          <-  Genes.split(';')[0]
    PG.ProteinGroups                  <-  Protein.Group

Precursor.Quantity, Precursor.Normalised, Ms1.Translated, Ms1.Area are
preserved as-is so ``collapse_sites``'s ``quantification_level`` selector
can pick the appropriate one at collapse time via the fallback chain in
:mod:`alphaphos.io.schemas`.

Phospho-only. Rows without ``(UniMod:21)`` in ``Modified.Sequence`` are
dropped; other PTMs are ignored and their bracket markers stripped from
``Modified.Sequence`` when building ``EG.PrecursorId``.

Ported from ``nanoPhos_env/nanoPhos_Figure3_DIANN_v00.ipynb`` (validated
against Spectronaut on the HeLa dilution series; Jaccard 40-48% site
overlap across 100..3000 cells, with the residual difference attributable
to low-abundance sites at the detection threshold rather than pipeline
disagreement).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from alphaphos.constants import (
    COL_EG_PRECURSOR_ID,
    COL_EG_PTM_ASSAY_PROB,
    COL_EG_PTM_LOC_PROBS,
    COL_PEP_PEPTIDE_POSITION,
    COL_PG_GENES,
    COL_PG_PROTEIN_GROUPS,
    COL_R_FILENAME,
    DIANN_GENES,
    DIANN_GLOBAL_PG_Q_VALUE,
    DIANN_LIB_PG_Q_VALUE,
    DIANN_MODIFIED_SEQUENCE,
    DIANN_PG_MAXLFQ_QUALITY,
    DIANN_PG_Q_VALUE,
    DIANN_PRECURSOR_CHARGE,
    DIANN_PROTEIN_GROUP,
    DIANN_PROTEIN_SITES,
    DIANN_PTM_SITE_CONFIDENCE,
    DIANN_QUANTITY_QUALITY,
    DIANN_RUN,
    DIANN_SITE_OCCUPANCY_PROBS,
    UNIMOD_PHOSPHO,
)
from alphaphos.io.schemas import REQUIRED_COLUMNS, all_needed_columns

logger = logging.getLogger("alphaphos.io.diann")


# ---------------------------------------------------------------------------
# Public defaults
# ---------------------------------------------------------------------------


# Defaults ported verbatim from the manuscript pipeline (nanoPhos §8.2).
# When MBR is on, DIANN_LIB_PG_Q_MAX is applied on top of the others.
DEFAULT_DIANN_IO_SETTINGS: dict[str, Any] = {
    "mbr": True,  # match-between-runs mode toggles the Lib.PG.Q.Value filter
    "pg_qvalue_max": 0.05,
    "global_pg_qvalue_max": 0.01,
    "lib_pg_qvalue_max": 0.01,  # only applied when mbr=True
    "quantity_quality_min": 0.5,
    "pg_maxlfq_quality_min": 0.7,
    "require_locprobs": True,  # drop rows missing Site.Occupancy.Probabilities
}


_UNIMOD = re.compile(r"\(UniMod:\d+\)")


# ---------------------------------------------------------------------------
# Settings resolution
# ---------------------------------------------------------------------------


def resolve_diann_io_settings(advanced: dict[str, Any] | None) -> dict[str, Any]:
    """Merge ``advanced`` overrides on top of :data:`DEFAULT_DIANN_IO_SETTINGS`.

    Unknown keys and bad values raise ``ValueError`` naming the allowed set.
    Returns a fresh dict.
    """
    settings = dict(DEFAULT_DIANN_IO_SETTINGS)
    if advanced is None:
        return settings
    if not isinstance(advanced, dict):
        raise TypeError(f"'advanced' must be a dict or None, got {type(advanced).__name__}")
    unknown = set(advanced) - set(DEFAULT_DIANN_IO_SETTINGS)
    if unknown:
        raise ValueError(
            f"Unknown keys in 'advanced': {sorted(unknown)}. "
            f"Allowed: {sorted(DEFAULT_DIANN_IO_SETTINGS)}."
        )
    settings.update(advanced)

    if not isinstance(settings["mbr"], bool):
        raise ValueError(f"mbr must be bool, got {type(settings['mbr']).__name__}")
    if not isinstance(settings["require_locprobs"], bool):
        raise ValueError(
            f"require_locprobs must be bool, got {type(settings['require_locprobs']).__name__}"
        )
    for k in (
        "pg_qvalue_max",
        "global_pg_qvalue_max",
        "lib_pg_qvalue_max",
        "quantity_quality_min",
        "pg_maxlfq_quality_min",
    ):
        v = settings[k]
        if v is not None and not (isinstance(v, (int, float)) and 0 <= float(v) <= 1):
            raise ValueError(f"{k} must be None or a float in [0, 1], got {v!r}")
    return settings


# ---------------------------------------------------------------------------
# String-parsing helpers (pure functions -- individually unit-testable)
# ---------------------------------------------------------------------------


def diann_precursor_id(modseq: str, charge: int) -> str:
    """Return the Spectronaut-format precursor id from a DIA-NN row.

    Replaces every ``(UniMod:21)`` (phospho STY) with ``[Phospho (STY)]``
    and strips all OTHER ``(UniMod:X)`` tags. Wraps in underscores and
    appends the charge suffix.

    Examples
    --------
    >>> diann_precursor_id("AAS(UniMod:21)PLK", 2)
    '_AAS[Phospho (STY)]PLK_.2'
    >>> diann_precursor_id("AC(UniMod:4)AS(UniMod:21)PLK", 2)
    '_ACAS[Phospho (STY)]PLK_.2'
    """
    s = modseq.replace(UNIMOD_PHOSPHO, "[Phospho (STY)]")
    return "_" + _UNIMOD.sub("", s) + "_." + str(int(charge))


def diann_loc_probs(occ: str | float) -> float | str:
    """Convert DIA-NN ``Site.Occupancy.Probabilities`` -> Spectronaut format.

    DIA-NN format::

        AAS(UniMod:21){1.000000}LPT{0.848000}K2

    Spectronaut format::

        _AAS[Phospho (STY): 100%]LPT[Phospho (STY): 84.8%]K_

    Returns ``np.nan`` for missing / non-string input.
    """
    if not isinstance(occ, str):
        return np.nan
    # Strip trailing charge digit and UniMod tags.
    s = re.sub(r"\d+$", "", _UNIMOD.sub("", occ))
    # {prob_as_fraction} -> [Phospho (STY): prob_as_percent%]
    s = re.sub(
        r"\{([\d.]+)\}",
        lambda m: "[Phospho (STY): %.4g%%]" % (float(m.group(1)) * 100),
        s,
    )
    return "_" + s + "_"


def first_phospho_abs_position(protein_sites: Any) -> float:
    """Return the absolute protein position of the FIRST phospho S/T/Y.

    Parses DIA-NN's ``Protein.Sites`` column, which looks like
    ``[P35221:C116,S118]`` (brackets, protein id, colon, comma-separated
    positions). Non-STY entries (carbamidomethyl-C etc.) are ignored.
    Returns ``np.nan`` when no S/T/Y entry is present or the string is
    unparseable.

    Examples
    --------
    >>> first_phospho_abs_position("[P12345:S117,T119]")
    117
    >>> first_phospho_abs_position("[P35221:C116,S118]")
    118
    >>> first_phospho_abs_position("[P00000:]")
    nan
    """
    inner = str(protein_sites).strip("[]").split(";")[0]
    if ":" not in inner:
        return np.nan
    sty = [t for t in inner.split(":", 1)[1].split(",") if t[:1] in ("S", "T", "Y")]
    if not sty:
        return np.nan
    return int(re.sub(r"\D", "", sty[0]))


def diann_peptide_start(modseq: str, protein_sites: Any) -> float:
    """Return the peptide's start position in the parent protein (1-indexed).

    Derives from the DIA-NN row: the first phospho's absolute position
    (from ``Protein.Sites``) minus its intra-peptide 1-indexed position
    (from ``Modified.Sequence``, using the position immediately before
    the first ``(UniMod:21)`` marker), plus 1.

    Returns ``np.nan`` if either input is missing / unparseable.

    Examples
    --------
    >>> # AAS(UniMod:21)PLK with S at protein pos 100 -> peptide start = 98
    >>> diann_peptide_start("AAS(UniMod:21)PLK", "[P12345:S100]")
    98
    """
    if UNIMOD_PHOSPHO not in modseq:
        return np.nan
    within = len(_UNIMOD.sub("", modseq[: modseq.index(UNIMOD_PHOSPHO)]))
    abs_pos = first_phospho_abs_position(protein_sites)
    if abs_pos != abs_pos:  # NaN check
        return np.nan
    return int(abs_pos) - within + 1


# ---------------------------------------------------------------------------
# Adapter: DIA-NN raw df -> Spectronaut-canonical PSM df
# ---------------------------------------------------------------------------


def _diann_to_psm(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Adapt a filtered DIA-NN DataFrame to the Spectronaut PSM schema.

    Returns
    -------
    (psm_df, n_unmappable)
        ``psm_df`` has the seven Spectronaut-canonical columns needed by
        :func:`alphaphos.collapse_sites` PLUS the raw DIA-NN quant columns
        (``Precursor.Quantity`` and any other present quant candidates)
        preserved as-is. ``n_unmappable`` counts rows where the peptide
        start position could not be derived (dropped from the output).
    """
    # Build the canonical metadata columns.
    out = pd.DataFrame(
        {
            COL_R_FILENAME: df[DIANN_RUN].astype(str),
            COL_EG_PRECURSOR_ID: [
                diann_precursor_id(m, c)
                for m, c in zip(
                    df[DIANN_MODIFIED_SEQUENCE], df[DIANN_PRECURSOR_CHARGE], strict=True
                )
            ],
            COL_PEP_PEPTIDE_POSITION: [
                diann_peptide_start(m, p)
                for m, p in zip(df[DIANN_MODIFIED_SEQUENCE], df[DIANN_PROTEIN_SITES], strict=True)
            ],
            COL_EG_PTM_ASSAY_PROB: df[DIANN_PTM_SITE_CONFIDENCE].values,
            COL_PG_GENES: df[DIANN_GENES].astype(str).str.split(";").str[0],
            COL_PG_PROTEIN_GROUPS: df[DIANN_PROTEIN_GROUP].astype(str),
        }
    )
    if DIANN_SITE_OCCUPANCY_PROBS in df.columns:
        out[COL_EG_PTM_LOC_PROBS] = [diann_loc_probs(o) for o in df[DIANN_SITE_OCCUPANCY_PROBS]]

    # Carry through the DIA-NN quant columns so downstream can pick via
    # QUANT_COLUMN_CANDIDATES["Diann"][level].
    for quant_col in (
        "Precursor.Quantity",
        "Precursor.Normalised",
        "Ms1.Translated",
        "Ms1.Area",
    ):
        if quant_col in df.columns:
            out[quant_col] = df[quant_col].values

    # Drop rows where the peptide start couldn't be derived.
    n_unmappable = int(out[COL_PEP_PEPTIDE_POSITION].isna().sum())
    out = out.dropna(subset=[COL_PEP_PEPTIDE_POSITION]).copy()
    out[COL_PEP_PEPTIDE_POSITION] = out[COL_PEP_PEPTIDE_POSITION].astype(int)
    return out, n_unmappable


# ---------------------------------------------------------------------------
# Column-scanning (schema-only, no data materialized)
# ---------------------------------------------------------------------------


def _scan_available_columns(path: Path) -> set[str]:
    """Return the set of columns that DIA-NN wrote in ``path``.

    Reads only the file schema (parquet) or first header line (TSV).
    """
    if path.suffix.lower() == ".parquet":
        return set(pq.read_schema(str(path)).names)
    with open(path, encoding="utf-8", errors="replace") as f:
        header = f.readline().rstrip("\r\n").split("\t")
    return set(header)


# ---------------------------------------------------------------------------
# QC filter (per manuscript §8.2)
# ---------------------------------------------------------------------------


def _apply_qc_filter(df: pd.DataFrame, settings: dict[str, Any]) -> tuple[pd.DataFrame, dict]:
    """Apply the six-step DIA-NN QC filter chain.

    Returns
    -------
    (df, funnel)
        ``df`` is the filtered DataFrame. ``funnel`` is a dict of row
        counts at each stage, for provenance stamping.
    """
    funnel: dict[str, int] = {"raw_precursors": len(df)}

    df = df[df[DIANN_GENES].notna() & (df[DIANN_GENES].astype(str).str.strip() != "")]
    funnel["has_gene"] = len(df)

    mask = pd.Series(True, index=df.index)
    if settings["pg_qvalue_max"] is not None and DIANN_PG_Q_VALUE in df.columns:
        mask &= df[DIANN_PG_Q_VALUE] <= settings["pg_qvalue_max"]
    if settings["global_pg_qvalue_max"] is not None and DIANN_GLOBAL_PG_Q_VALUE in df.columns:
        mask &= df[DIANN_GLOBAL_PG_Q_VALUE] <= settings["global_pg_qvalue_max"]
    if settings["quantity_quality_min"] is not None and DIANN_QUANTITY_QUALITY in df.columns:
        mask &= df[DIANN_QUANTITY_QUALITY] >= settings["quantity_quality_min"]
    if settings["pg_maxlfq_quality_min"] is not None and DIANN_PG_MAXLFQ_QUALITY in df.columns:
        mask &= df[DIANN_PG_MAXLFQ_QUALITY] >= settings["pg_maxlfq_quality_min"]
    if (
        settings["mbr"]
        and settings["lib_pg_qvalue_max"] is not None
        and DIANN_LIB_PG_Q_VALUE in df.columns
    ):
        mask &= df[DIANN_LIB_PG_Q_VALUE] <= settings["lib_pg_qvalue_max"]
    df = df[mask]
    funnel["after_qc"] = len(df)

    # Phospho-only: keep rows with UniMod:21 in Modified.Sequence.
    df = df[df[DIANN_MODIFIED_SEQUENCE].str.contains("UniMod:21", na=False)].copy()
    funnel["phospho"] = len(df)

    if settings["require_locprobs"] and DIANN_SITE_OCCUPANCY_PROBS in df.columns:
        df = df[df[DIANN_PROTEIN_SITES].notna() & df[DIANN_SITE_OCCUPANCY_PROBS].notna()]
    else:
        df = df[df[DIANN_PROTEIN_SITES].notna()]
    funnel["localizable"] = len(df)

    return df, funnel


# ---------------------------------------------------------------------------
# Public reader
# ---------------------------------------------------------------------------


def read_psm(
    path: str | Path,
    *,
    advanced: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Read a DIA-NN report and adapt to the Spectronaut-canonical PSM schema.

    Parameters
    ----------
    path : str | Path
        Path to a DIA-NN ``*.parquet`` or ``*.tsv`` main report.
        Requires DIA-NN >= 1.9 (needs ``Protein.Sites`` and
        ``Site.Occupancy.Probabilities``).
    advanced : dict, optional
        Overrides for :data:`DEFAULT_DIANN_IO_SETTINGS`. Unknown keys raise.

    Returns
    -------
    pandas.DataFrame
        PSM rows with Spectronaut-canonical metadata columns plus the
        native DIA-NN quant columns preserved. Ready to feed into
        :func:`alphaphos.collapse_sites` with ``advanced["search_engine"]="Diann"``.

        ``df.attrs`` records lineage::

            source_path            -- str, absolute path read
            engine                 -- "Diann"
            n_rows_loaded          -- int, rows before any filter
            n_rows_after_qc        -- int, after the six-step QC filter
            n_rows_phospho         -- int, after the UniMod:21 filter
            n_rows_localizable     -- int, after the loc-string presence filter
            n_rows_unmappable      -- int, rows dropped because peptide-start
                                      could not be derived from Protein.Sites
            n_rows_returned        -- int, final row count

    Raises
    ------
    FileNotFoundError
        If ``path`` doesn't exist.
    ValueError
        If required columns are missing.
    """
    settings = resolve_diann_io_settings(advanced)
    engine = "Diann"

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DIA-NN report not found: {p}")

    # Column pruning: figure out what to read before reading anything.
    available_raw = _scan_available_columns(p)
    needed = all_needed_columns(engine)
    cols_to_read = sorted(available_raw & needed)

    missing = set(REQUIRED_COLUMNS[engine]) - available_raw
    if missing:
        raise ValueError(
            f"DIA-NN report is missing required column(s): {sorted(missing)}. "
            f"Present columns: {sorted(available_raw)[:30]}..."
        )

    n_total_cols = len(available_raw)
    n_dropped_cols = n_total_cols - len(cols_to_read)

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p, columns=cols_to_read)
    elif p.suffix.lower() in (".tsv", ".txt"):
        df = pd.read_csv(p, sep="\t", usecols=cols_to_read, low_memory=False)
    else:
        raise ValueError(
            f"Unsupported file extension {p.suffix!r}; expected .parquet, .tsv, or .txt"
        )

    n_loaded = len(df)

    # Apply the six-step DIA-NN QC filter chain (funnel counts stamped later).
    df, funnel = _apply_qc_filter(df, settings)

    # Adapt to Spectronaut-canonical schema.
    df_out, n_unmappable = _diann_to_psm(df)
    df_out = df_out.reset_index(drop=True)

    df_out.attrs["source_path"] = str(p)
    df_out.attrs["engine"] = engine
    df_out.attrs["n_rows_loaded"] = n_loaded
    df_out.attrs["n_rows_after_qc"] = funnel["after_qc"]
    df_out.attrs["n_rows_phospho"] = funnel["phospho"]
    df_out.attrs["n_rows_localizable"] = funnel["localizable"]
    df_out.attrs["n_rows_unmappable"] = n_unmappable
    df_out.attrs["n_rows_returned"] = len(df_out)
    df_out.attrs["columns_dropped"] = n_dropped_cols

    logger.info(
        "read_diann(%s): %d rows returned; funnel=%s; n_unmappable=%d; %d cols pruned.",
        p.name,
        len(df_out),
        funnel,
        n_unmappable,
        n_dropped_cols,
    )
    return df_out
