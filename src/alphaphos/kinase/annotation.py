"""FASTA-based kinase-window annotation for a collapsed AnnData.

Given a collapsed AnnData (from :func:`alphaphos.collapse_sites`), this
module attaches a ``kinase_sequence`` column to ``adata.var`` that carries
the +/-N residue window around each modified residue. This is the exact
input format expected by the Yaffe Kinase Library scoring in
:mod:`alphaphos.kinase.library` and by GSEA / KSEA workflows that need
sequence context.

Format
------
For each site::

    protein_group_id --lookup--> FASTA sequence
    site_position    --index--> center residue
    site_aa          --sanity check--> AA at that position

Output string per site::

    _..._{left_flank}{aa_lower}*{right_flank}_..._

where ``aa_lower`` is the lowercased center residue wrapped in ``*``s, the
flanks are the window of ``window_size`` residues on each side, and ``_``
padding is inserted at protein N- and C-termini when the window overflows
the sequence.

Example (``window_size=7``, center at position 4 of "PEPTIDES...")::

    _PEPTIDEs*ABCDEFG_

Sentinel strings for failures (matches the historical alphaPhos convention;
downstream :mod:`kinase.library` recognizes these and treats them as missing):

* ``FASTA_ERROR:``       -- protein_group_id not in the FASTA dictionary.
* ``POSITION_ERROR:``    -- ``site_position`` out of bounds of the protein.
* ``SEQUENCE_MISMATCH:`` -- AA at that position in FASTA disagrees with the
                            recorded ``site_aa``.

Usage
-----
>>> import alphaphos as ap
>>> adata = ap.collapse_sites(psm_df, condition_df=cdf)
>>> adata = ap.add_kinase_windows(adata, fasta_path="human.fasta")
>>> adata.var["kinase_sequence"].head()

Species mismatch: use the SAME FASTA that the search engine used. Common
mistake is running collapse on CHO data and then annotating against a
human FASTA -- most lookups will fail with ``FASTA_ERROR:``.
"""

from __future__ import annotations

import logging
from pathlib import Path

try:
    import anndata as ad
except ImportError:  # pragma: no cover
    ad = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


ERROR_PREFIXES = ("FASTA_ERROR:", "POSITION_ERROR:", "SEQUENCE_MISMATCH:", "PARSING_ERROR:")


# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------


def load_fasta(path: str | Path) -> dict[str, str]:
    """Load a FASTA file into ``{accession: sequence}``.

    Accepts three common header conventions:

    * UniProt: ``>sp|P12345|GENE_HUMAN Some description`` -- the accession
      is ``P12345`` (between the first two pipes).
    * Bare header: ``>P12345 Some description`` -- the accession is the
      first whitespace-delimited token.
    * Anything else with a leading ``>``: the first whitespace-delimited
      token after ``>`` is treated as the accession.

    Sequences are joined across continuation lines and uppercased. Blank
    lines are ignored.

    Parameters
    ----------
    path : str | Path
        Path to a text FASTA file. Gzipped FASTAs are NOT auto-decompressed;
        wrap with ``gzip.open`` upstream if needed.

    Returns
    -------
    dict[str, str]
        Non-empty dictionary. Raises when the file is empty or unreadable.

    Raises
    ------
    FileNotFoundError
        If ``path`` doesn't exist.
    ValueError
        If no sequences could be parsed (empty file, only headers, etc.).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    fasta_dict: dict[str, str] = {}
    current_id: str | None = None
    current_sequence: list[str] = []

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if line.startswith(">"):
                # Flush previous entry.
                if current_id is not None:
                    fasta_dict[current_id] = "".join(current_sequence)
                # Parse new header.
                header = line[1:]
                parts = header.split("|")
                if len(parts) >= 2:
                    current_id = parts[1]
                else:
                    current_id = header.split()[0] if header.split() else None
                current_sequence = []
            elif line and current_id is not None:
                current_sequence.append(line.upper())

    if current_id is not None:
        fasta_dict[current_id] = "".join(current_sequence)

    if not fasta_dict:
        raise ValueError(f"No sequences parsed from FASTA: {path}")

    logger.info("Loaded FASTA %s: %d protein entries.", path, len(fasta_dict))
    return fasta_dict


# ---------------------------------------------------------------------------
# Window extraction (site level)
# ---------------------------------------------------------------------------


def extract_window(
    fasta_dict: dict[str, str],
    protein_id: str,
    position: int,
    aa: str,
    *,
    window_size: int = 7,
) -> str:
    """Return the +/- window around one site, or a sentinel error string.

    Parameters
    ----------
    fasta_dict : dict[str, str]
        Output of :func:`load_fasta`.
    protein_id : str
        Accession key to look up in ``fasta_dict``.
    position : int
        1-indexed position of the modified residue in the protein.
    aa : str
        Expected AA at that position (used for a sanity check against the
        FASTA; ``"S"``/``"T"``/``"Y"`` for phospho).
    window_size : int
        Number of residues on EACH side of the center. Default 7 (matches
        the Yaffe Kinase Library convention of 15-mers).

    Returns
    -------
    str
        Either the ``_..._`` wrapped window with ``*aa_lower*`` marker, or
        one of the sentinel strings listed in the module docstring.
    """
    if protein_id not in fasta_dict:
        return f"FASTA_ERROR: Protein '{protein_id}' not found in FASTA dictionary"

    sequence = fasta_dict[protein_id]
    seq_len = len(sequence)
    zero_pos = int(position) - 1

    if zero_pos < 0 or zero_pos >= seq_len:
        return (
            f"POSITION_ERROR: Position {position} out of bounds for protein "
            f"'{protein_id}' (length: {seq_len})"
        )

    actual_aa = sequence[zero_pos]
    if actual_aa != str(aa).upper():
        return (
            f"SEQUENCE_MISMATCH: Expected '{aa}' at position {position} in "
            f"'{protein_id}', found '{actual_aa}'"
        )

    # Build the window with `_` padding at termini.
    left_start = zero_pos - window_size
    right_end = zero_pos + window_size  # inclusive

    parts: list[str] = []
    if left_start < 0:
        parts.append("_" * abs(left_start))
        actual_left = 0
    else:
        actual_left = left_start
    parts.append(sequence[actual_left:zero_pos])
    parts.append(f"*{str(aa).upper()}*")
    actual_right = min(right_end, seq_len - 1)
    parts.append(sequence[zero_pos + 1 : actual_right + 1])
    if right_end >= seq_len:
        parts.append("_" * (right_end - seq_len + 1))

    return "_" + "".join(parts) + "_"


# ---------------------------------------------------------------------------
# Public: annotate an AnnData in place (returning a modified copy)
# ---------------------------------------------------------------------------


def add_kinase_windows(
    adata: ad.AnnData,
    fasta_path: str | Path,
    *,
    window_size: int = 7,
    protein_col: str = "protein_group_id",
    position_col: str = "site_position",
    aa_col: str = "site_aa",
    out_col: str = "kinase_sequence",
    copy: bool = True,
) -> ad.AnnData:
    """Add a per-site kinase window column to ``adata.var``.

    Reads three columns from ``adata.var`` and looks up each site's flanking
    residues in the FASTA. Failures produce sentinel strings prefixed with
    ``FASTA_ERROR:``, ``POSITION_ERROR:``, or ``SEQUENCE_MISMATCH:``.

    Parameters
    ----------
    adata : AnnData
        The collapsed AnnData from :func:`alphaphos.collapse_sites`.
    fasta_path : str | Path
        Path to the FASTA file whose accessions match ``adata.var[protein_col]``.
        Use the SAME FASTA that the search engine used, otherwise most
        lookups will fail with ``FASTA_ERROR:``.
    window_size : int
        Number of residues on each side of the center residue. Default 7
        (a 15-mer, matching the Yaffe library).
    protein_col, position_col, aa_col : str
        Column names in ``adata.var`` for the FASTA accession, 1-indexed
        residue position in the parent protein, and the modified AA
        letter respectively. Defaults match what :func:`collapse_sites`
        populates.
    out_col : str
        Name of the new column to add. Default ``"kinase_sequence"``.
    copy : bool
        If True (default), return a modified copy; the input is untouched.
        If False, mutate ``adata.var`` in place and return the same object.

    Returns
    -------
    AnnData
        Same shape as input; ``.var[out_col]`` contains the window strings.
        Also stamps ``adata.uns["alphaphos"]["kinase_annotation"] = {...}``
        with the FASTA path, window size, and per-status counts.

    Raises
    ------
    KeyError
        If any of ``protein_col``, ``position_col``, ``aa_col`` is missing
        from ``adata.var``.
    FileNotFoundError
        If ``fasta_path`` doesn't exist.
    """
    if ad is None:  # pragma: no cover
        raise ImportError("anndata is required. Install with `pip install anndata`.")

    for col in (protein_col, position_col, aa_col):
        if col not in adata.var.columns:
            raise KeyError(
                f"adata.var is missing required column '{col}'. Have: {sorted(adata.var.columns)}"
            )

    fasta_dict = load_fasta(fasta_path)
    ad_out = adata.copy() if copy else adata

    sequences: list[str] = []
    n_ok = n_fasta_err = n_pos_err = n_mismatch = 0
    for pid, pos, aa in zip(
        ad_out.var[protein_col].tolist(),
        ad_out.var[position_col].tolist(),
        ad_out.var[aa_col].tolist(),
        strict=True,
    ):
        try:
            window = extract_window(
                fasta_dict, str(pid), int(pos), str(aa), window_size=window_size
            )
        except Exception as exc:  # defensive: don't crash on one bad row
            window = f"PARSING_ERROR: {exc}"
        if window.startswith("FASTA_ERROR:"):
            n_fasta_err += 1
        elif window.startswith("POSITION_ERROR:"):
            n_pos_err += 1
        elif window.startswith("SEQUENCE_MISMATCH:"):
            n_mismatch += 1
        else:
            n_ok += 1
        sequences.append(window)

    ad_out.var[out_col] = sequences

    # Provenance / diagnostics.
    ns = ad_out.uns.setdefault("alphaphos", {})
    ns["kinase_annotation"] = {
        "fasta_path": str(fasta_path),
        "window_size": int(window_size),
        "n_ok": n_ok,
        "n_fasta_error": n_fasta_err,
        "n_position_error": n_pos_err,
        "n_sequence_mismatch": n_mismatch,
    }

    logger.info(
        "add_kinase_windows: %d ok, %d fasta_error, %d position_error, "
        "%d sequence_mismatch (total %d).",
        n_ok,
        n_fasta_err,
        n_pos_err,
        n_mismatch,
        len(sequences),
    )
    return ad_out
