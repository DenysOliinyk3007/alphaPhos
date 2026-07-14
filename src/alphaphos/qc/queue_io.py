"""MS acquisition queue loader.

Parses a Thermo Xcalibur sequence CSV into the alphaPhos acquisition-
queue contract used by ``alphaphos.qc.psm_metrics`` and related QC
functions.

Xcalibur format (as of Xcalibur 4.x / 5.x):

- Optional preamble line ``Bracket Type=N`` (integer 1-4)
- Header row: ``File Name,Path,Instrument Method,Position``
- Data rows -- row order is injection order (top-to-bottom)

Blank / wash injections are filtered by default: they don't make it to
the search software so no PSM data is ever computed for them and they
clutter downstream analysis.  Original ``injection_order`` values are
retained -- gaps in the numbering indicate a blank was skipped, useful
for detecting column re-equilibration effects.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import pandas as pd

QueueFormat = Literal["auto", "xcalibur"]

# Default regex: match rows whose sample name contains 'blank' (case-
# insensitive).  Users with different blank/wash naming conventions
# (e.g. 'BLK', '_wash_') can pass a custom regex.
DEFAULT_BLANK_PATTERN = r"(?i)blank"

_XCALIBUR_HEADER = "File Name,Path,Instrument Method,Position"
_XCALIBUR_PREAMBLE_RE = re.compile(r"^Bracket Type=\d+\s*$")


def load_acquisition_queue(
    path: str | Path,
    *,
    format: QueueFormat = "auto",
    blank_pattern: str | None = DEFAULT_BLANK_PATTERN,
) -> pd.DataFrame:
    """Parse an MS acquisition queue file into the alphaPhos contract.

    Parameters
    ----------
    path
        Path to the queue CSV.  Currently only Thermo Xcalibur sequence
        CSV is supported.
    format
        ``"auto"`` (default) sniffs the first line for the Xcalibur
        preamble; ``"xcalibur"`` forces Xcalibur parsing regardless.
    blank_pattern
        Regex identifying blank / wash rows to drop.  Default matches
        ``blank`` case-insensitive anywhere in the sample name.  Pass
        ``None`` to keep all rows.

    Returns
    -------
    pandas.DataFrame
        One row per non-blank sample with columns::

            sample            : str  -- from Xcalibur's 'File Name'
            injection_order   : int  -- 1-indexed row order in the SOURCE file
                                        (gaps preserve blank-wash positions)
            position          : str  -- rack:well (e.g. 'S5:D9')
            instrument_method : str  -- absolute path to the method file

        ``.attrs['provenance']`` carries:

            format_detected     : str
            n_total_rows        : int
            n_blank_removed     : int
            blank_samples       : list[str]
            source_path         : str
    """
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"acquisition queue file not found: {resolved_path}")

    resolved_format = _resolve_format(resolved_path, format=format)
    if resolved_format == "xcalibur":
        raw = _read_xcalibur(resolved_path)
    else:  # pragma: no cover - guarded by the resolver
        raise ValueError(f"Unsupported queue format: {resolved_format!r}")

    # Row order in the source = injection order.  1-indexed.  Preserved
    # across the blank filter so gaps in the numbering are visible.
    raw = raw.reset_index(drop=True)
    raw.insert(0, "injection_order", raw.index + 1)

    blank_mask, blank_samples = _blank_mask(raw["File Name"], blank_pattern)
    kept = raw.loc[~blank_mask].copy()

    result = pd.DataFrame(
        {
            "sample": kept["File Name"].astype(str).to_numpy(),
            "injection_order": kept["injection_order"].astype(int).to_numpy(),
            "position": kept["Position"].astype(str).to_numpy(),
            "instrument_method": kept["Instrument Method"].astype(str).to_numpy(),
        }
    )
    result.attrs["provenance"] = {
        "format_detected": resolved_format,
        "n_total_rows": len(raw),
        "n_blank_removed": len(blank_samples),
        "blank_samples": blank_samples,
        "source_path": str(resolved_path),
    }
    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_format(path: Path, *, format: QueueFormat) -> str:
    """Return the concrete format label after auto-detection if needed."""
    if format == "xcalibur":
        return "xcalibur"
    if format != "auto":  # pragma: no cover - defensive
        raise ValueError(f"format must be 'auto' or 'xcalibur'; got {format!r}")
    # Auto-detect: read the first ~200 bytes and look for Xcalibur signature.
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        first_line = fh.readline().strip()
        second_line = (
            fh.readline().strip() if _XCALIBUR_PREAMBLE_RE.match(first_line) else first_line
        )

    if _XCALIBUR_PREAMBLE_RE.match(first_line) and _looks_like_xcalibur_header(second_line):
        return "xcalibur"
    if _looks_like_xcalibur_header(first_line):
        return "xcalibur"
    raise ValueError(
        f"Could not auto-detect queue format for {path.name}.  Expected an "
        "Xcalibur sequence CSV with header "
        f"'{_XCALIBUR_HEADER}'.  Pass format='xcalibur' explicitly if you're "
        "sure, or open an issue with your queue format."
    )


def _looks_like_xcalibur_header(line: str) -> bool:
    fields = {f.strip().lower() for f in line.split(",")}
    required = {"file name", "path", "instrument method", "position"}
    return required.issubset(fields)


def _read_xcalibur(path: Path) -> pd.DataFrame:
    """Read the CSV, skipping the ``Bracket Type=N`` preamble if present."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        first_line = fh.readline().strip()
    skip = 1 if _XCALIBUR_PREAMBLE_RE.match(first_line) else 0

    df = pd.read_csv(path, skiprows=skip, encoding="utf-8-sig")
    required = ["File Name", "Path", "Instrument Method", "Position"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Xcalibur queue is missing required columns: {missing}.  "
            f"Found columns: {list(df.columns)}"
        )
    return df.loc[:, required]


def _blank_mask(
    sample_names: pd.Series,
    blank_pattern: str | None,
) -> tuple[pd.Series, list[str]]:
    """Return (mask, filtered_names) for blank filtering."""
    if blank_pattern is None:
        empty_mask = pd.Series(False, index=sample_names.index)
        return empty_mask, []
    compiled = re.compile(blank_pattern)
    mask = sample_names.astype(str).apply(lambda s: bool(compiled.search(s)))
    return mask, sample_names[mask].astype(str).tolist()
