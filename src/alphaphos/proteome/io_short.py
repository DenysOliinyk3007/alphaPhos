"""Read Spectronaut's short (wide, pre-collapsed) protein-group report.

Format expectation
------------------
- Two identifier columns: ``PG_ProteinGroups`` and ``PG_Genes``
  (semicolon-joined group members allowed, e.g. ``P1;P2;P3``).
- One quantitative column per run named
  ``[<index>]_<runname>_raw_PG_Quantity`` (or ``.PG.Quantity`` in some
  Spectronaut exports).  Values are **raw linear intensity**; the reader
  log2-transforms them (positive values only; zeros and negatives → NaN).
- Non-matching columns (e.g. ``PG_UniProtIds``) are propagated to
  ``adata.var`` verbatim.

Sample matching
---------------
Run names are extracted from the quant column headers via a strict regex.
If a ``condition_df`` is supplied it must carry a ``sample`` column; matching
is by run name.  Unmatched samples on either side are surfaced through
``.uns["alphaphos_proteome"]["stats"]`` so callers can audit without a
silent drop.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)

# Matches: [<n>]_<runname>_raw_PG_Quantity   (accepts both '_' and '.' in the
# trailing PG_Quantity marker; Spectronaut has historically shipped both).
_QUANT_COL_RE = re.compile(
    r"""
    ^\[(?P<idx>\d+)\]           # leading [N]
    _
    (?P<run>.+?)                # run name (non-greedy)
    _raw
    [_.]                        # separator
    PG
    [_.]                        # separator
    Quantity$                   # trailing marker
    """,
    re.VERBOSE,
)

# Columns that are per-protein metadata (not per-sample quant).  Anything
# that doesn't match _QUANT_COL_RE and isn't in this list is still copied to
# .var as-is, but these are the ones we always expect.
_METADATA_COLS_EXPECTED = ("PG_ProteinGroups", "PG_Genes")


def read_spectronaut_short(
    path: str | Path,
    *,
    condition_df: pd.DataFrame | None = None,
) -> ad.AnnData:
    """Read a Spectronaut short (wide) protein-group report → AnnData.

    Parameters
    ----------
    path
        Path to the ``.parquet`` (preferred) or ``.tsv`` report.  Format is
        auto-detected from the file extension.
    condition_df
        Optional ``pd.DataFrame`` with a ``sample`` column matching the
        run names extracted from the quant columns, plus any per-sample
        annotation columns (``condition``, ``batch``, etc.).  Attached to
        ``adata.obs``.  If ``None``, ``adata.obs`` is empty (samples only).

    Returns
    -------
    AnnData
        Shape ``(n_samples, n_proteins)``.
        ``.X`` -- log2 of the raw quant (zeros/negatives → NaN).
        ``.layers["intensity_log2"]`` -- copy of ``.X`` (matches phospho
        convention so downstream filter / impute / diff-exp find the
        expected layer).
        ``.var`` -- indexed by ``PG_ProteinGroups``, with ``PG_Genes`` and
        any other non-quant columns preserved.
        ``.obs`` -- indexed by run name, joined with ``condition_df``.
        ``.uns["alphaphos_proteome"]`` -- ``{"reader": "short",
        "format": <parquet|tsv>, "path": <str>, "stats": {...}}``.

    Notes
    -----
    Sample run names come from the quant column headers, not the file
    contents.  If ``condition_df`` uses e.g. ``phosphoDVP`` in the run
    names but the proteome columns say ``proteomeDVP``, direct matching
    fails and the samples are recorded as unmatched.  Provide a proteome-
    specific ``condition_df`` in that case, or use the well-ID pairing
    logic in :func:`alphaphos.proteome.phospho_over_proteome` for the
    phospho ↔ proteome bridge.
    """
    path = Path(path)
    df, fmt = _read_report(path)

    quant_cols, meta_cols = _classify_columns(df.columns)
    logger.info(
        "read_spectronaut_short: %s -> %d proteins x %d samples (metadata cols: %s)",
        path.name,
        len(df),
        len(quant_cols),
        meta_cols,
    )

    if not quant_cols:
        raise ValueError(
            f"No columns matching the pattern '[N]_<runname>_raw_PG_Quantity' "
            f"were found in {path.name}. Got {len(df.columns)} columns; first few: "
            f"{list(df.columns[:5])}"
        )

    # Build sample-name → column-name map from the quant col regex.
    sample_to_col: dict[str, str] = {}
    for col in quant_cols:
        m = _QUANT_COL_RE.match(col)
        run = m.group("run")
        if run in sample_to_col:
            raise ValueError(
                f"Duplicate run name after parsing: {run!r} appears in both "
                f"{sample_to_col[run]!r} and {col!r}"
            )
        sample_to_col[run] = col

    samples = list(sample_to_col.keys())

    # Assemble the (n_samples, n_proteins) log2 matrix.
    quant = df[[sample_to_col[s] for s in samples]].to_numpy(
        dtype=np.float64
    )  # (n_proteins, n_samples)
    with np.errstate(invalid="ignore", divide="ignore"):
        # log2 of non-positive → NaN (matches how phospho path treats missing)
        quant_log2 = np.where(quant > 0, np.log2(quant, where=quant > 0), np.nan)
    X = quant_log2.T  # (n_samples, n_proteins)

    # Build var (per-protein metadata).
    var = df[list(meta_cols)].copy()
    var.index = df["PG_ProteinGroups"].astype(str)
    var.index.name = "PG_ProteinGroups"

    # Build obs (per-sample metadata).
    obs = pd.DataFrame(index=pd.Index(samples, name="sample"))
    n_matched, n_unmatched_cond, unmatched_samples = 0, 0, []
    if condition_df is not None:
        if "sample" not in condition_df.columns:
            raise ValueError("condition_df must have a 'sample' column to match against run names.")
        cdf = condition_df.set_index("sample")
        matched = obs.index.intersection(cdf.index)
        n_matched = len(matched)
        unmatched_samples = obs.index.difference(cdf.index).tolist()
        n_unmatched_cond = len(cdf.index.difference(obs.index))
        obs = obs.join(cdf)  # unmatched samples get NaN in condition columns

    # Assemble AnnData (deferred import so alphaphos.proteome package can be
    # imported at top-level without anndata being installed).
    import anndata as ad

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers["intensity_log2"] = adata.X.copy()

    n_missing = int(np.isnan(adata.X).sum())
    total = int(adata.X.size)
    adata.uns["alphaphos_proteome"] = {
        "reader": "short",
        "format": fmt,
        "path": str(path),
        "stats": {
            "n_samples": int(adata.n_obs),
            "n_proteins": int(adata.n_vars),
            "n_missing_cells": n_missing,
            "missing_frac": n_missing / total if total else 0.0,
            "n_samples_matched_condition": n_matched,
            "n_samples_unmatched_in_condition_df": len(unmatched_samples),
            "n_condition_rows_unmatched_in_report": n_unmatched_cond,
        },
    }
    if unmatched_samples:
        logger.warning(
            "read_spectronaut_short: %d/%d samples in the report have no row in "
            "condition_df (their .obs values will be NaN). First few: %s",
            len(unmatched_samples),
            adata.n_obs,
            unmatched_samples[:5],
        )
    if n_unmatched_cond:
        logger.warning(
            "read_spectronaut_short: %d rows in condition_df have no matching "
            "sample in the report and were dropped from .obs.",
            n_unmatched_cond,
        )
    return adata


def _read_report(path: Path) -> tuple[pd.DataFrame, str]:
    """Read either a .parquet or .tsv report; return (df, format_str)."""
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path), "parquet"
    if ext in {".tsv", ".txt"}:
        # Spectronaut's short TSVs can be large; keep the read tight.
        return pd.read_csv(path, sep="\t", low_memory=False), "tsv"
    raise ValueError(
        f"Unsupported extension {ext!r} for a Spectronaut short report; "
        "expected .parquet, .tsv, or .txt."
    )


def _classify_columns(columns) -> tuple[list[str], list[str]]:
    """Split columns into (quant_cols, metadata_cols)."""
    quant, meta = [], []
    for col in columns:
        if _QUANT_COL_RE.match(str(col)):
            quant.append(col)
        else:
            meta.append(col)
    return quant, meta
