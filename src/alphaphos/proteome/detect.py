"""Tell a proteome report's search engine and layout apart from its header alone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from alphaphos.proteome.io_diann_short import DIANN_PG_COLUMN
from alphaphos.proteome.io_short import _QUANT_COL_RE

SPECTRONAUT, DIANN = "Spectronaut", "DIA-NN"
SHORT, LONG = "short", "long"


@dataclass(frozen=True)
class ReportFormat:
    engine: str  # "Spectronaut" | "DIA-NN"
    level: str  # "short" (protein x run matrix) | "long" (one row per precursor and run)


def report_columns(path: str | Path) -> list[str]:
    """Header of a TSV / TXT / parquet report, without reading its rows."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return list(pq.read_schema(path).names)
    with open(path, encoding="utf-8-sig") as handle:
        return handle.readline().rstrip("\r\n").split("\t")


def detect_report_format(path: str | Path) -> ReportFormat:
    """Engine and layout of a proteome report, from its column names.

    - Spectronaut short: ``[N] <run>.raw.PG.Quantity`` columns.
    - Spectronaut long: ``R.FileName`` plus ``PG.ProteinGroups``.
    - DIA-NN short (``pg_matrix``): ``Protein.Group`` and no ``Precursor.Id``.
    - DIA-NN long (main report): ``Run`` plus ``Precursor.Id``.

    Dotted and underscored column spellings are both recognised.
    """
    columns = report_columns(path)
    dotted = {c.replace("_", ".") for c in columns}
    if any(_QUANT_COL_RE.match(c) for c in columns):
        return ReportFormat(SPECTRONAUT, SHORT)
    if "R.FileName" in dotted and "PG.ProteinGroups" in dotted:
        return ReportFormat(SPECTRONAUT, LONG)
    if "Precursor.Id" in columns and "Run" in columns:
        return ReportFormat(DIANN, LONG)
    if DIANN_PG_COLUMN in columns:
        return ReportFormat(DIANN, SHORT)
    raise ValueError(
        f"{Path(path).name} does not look like a Spectronaut or DIA-NN proteome report "
        f"(first columns: {columns[:5]})."
    )
