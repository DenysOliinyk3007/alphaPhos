"""Read DIA-NN's protein-group matrix (``report.pg_matrix.tsv``).

Format expectation
------------------
- Annotation columns ``Protein.Group``, ``Protein.Names``, ``Genes``,
  ``First.Protein.Description`` and (DIA-NN >= 1.9) ``N.Sequences``,
  ``N.Proteotypic.Sequences``; ``Protein.Ids`` in some versions.
- Every other column is one run, headed by the raw file's full path
  (``V:\\data\\20260627_..._A1.raw``).  The run name is that path's file
  name without the acquisition extension, the same name ``ap.read_diann``
  gives the phospho runs, so well-ID pairing works across the two.
- Values are linear MaxLFQ intensities; the reader log2-transforms them.

The output AnnData has the same layout as
:func:`alphaphos.proteome.read_spectronaut_short`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from alphaphos.proteome.io_short import PG_INDEX, _build_protein_adata

if TYPE_CHECKING:
    import anndata as ad

DIANN_PG_COLUMN = "Protein.Group"
DIANN_PG_METADATA = (
    DIANN_PG_COLUMN,
    "Protein.Ids",
    "Protein.Names",
    "Genes",
    "First.Protein.Description",
    "N.Sequences",
    "N.Proteotypic.Sequences",
)
# DIA-NN annotation name -> the protein AnnData convention shared with Spectronaut.
_RENAME = {DIANN_PG_COLUMN: PG_INDEX, "Genes": "PG_Genes"}
_ACQUISITION_EXT = re.compile(r"\.(raw|d|wiff2?|mzml|mzxml|dia)$", re.IGNORECASE)


def run_name_from_path(column: str) -> str:
    """``V:\\data\\run_A1.raw`` -> ``run_A1``; works for Windows and POSIX paths."""
    name = re.split(r"[\\/]", str(column).strip().rstrip("\\/"))[-1]
    return _ACQUISITION_EXT.sub("", name)


def read_diann_pg_matrix(
    path: str | Path,
    *,
    condition_df: pd.DataFrame | None = None,
) -> ad.AnnData:
    """Read a DIA-NN ``pg_matrix`` (protein groups x runs) report → AnnData.

    Parameters
    ----------
    path
        ``report.pg_matrix.tsv`` (or a ``.parquet`` re-save of it).
    condition_df
        Optional table with a ``sample`` column matching the run names
        (file names without path or extension); joined into ``.obs``.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df, fmt = pd.read_parquet(path), "parquet"
    elif suffix in (".tsv", ".txt"):
        df, fmt = pd.read_csv(path, sep="\t", low_memory=False), "tsv"
    else:
        raise ValueError(f"Unsupported extension {suffix!r}; expected .tsv, .txt or .parquet.")
    if DIANN_PG_COLUMN not in df.columns:
        raise ValueError(f"{path.name} has no {DIANN_PG_COLUMN!r} column; not a DIA-NN pg_matrix.")

    run_cols = [c for c in df.columns if c not in DIANN_PG_METADATA]
    if not run_cols:
        raise ValueError(f"{path.name} has no run columns.")
    runs = [run_name_from_path(c) for c in run_cols]
    duplicates = sorted({r for r in runs if runs.count(r) > 1})
    if duplicates:
        raise ValueError(f"Run names repeat after removing paths: {duplicates[:3]}")

    quant = df[run_cols].set_axis(runs, axis=1)
    var = df[[c for c in df.columns if c in DIANN_PG_METADATA]].rename(columns=_RENAME)
    return _build_protein_adata(
        quant,
        var,
        condition_df=condition_df,
        reader="diann_pg_matrix",
        fmt=fmt,
        path=path,
        caller="read_diann_pg_matrix",
    )
