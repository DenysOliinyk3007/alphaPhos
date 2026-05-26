"""Contaminant filter for PSM-level Spectronaut reports.

Identifies and removes peptide-to-protein assignments that map exclusively
to common-contaminant proteins (trypsin, BSA, serum albumin, keratins,
collagens, etc.) using two complementary mechanisms:

1. **Prefix detection.** Protein IDs starting with one of
   ``("CON__", "Cont_", "contam_")``. Common when the search engine
   (MaxQuant, Spectronaut) tagged contaminants at search time.

2. **FASTA accession lookup.** Protein accessions present in a
   contaminants FASTA file. Use this when the report was not
   prefix-tagged, or as a redundant cross-check.

The default contaminants FASTA bundled with alphaPhos is the MaxQuant
``contaminants.fasta`` (246 sequences: keratins, serum proteins,
proteases, common buffer/lab contaminants); see
``alphaphos/resources/contaminants.fasta``.

A row is dropped only if **every** protein in its ``PG.ProteinGroups`` is
a contaminant (prefix or FASTA match). Peptides ambiguously assigned to a
contaminant *and* a real protein are kept — their real-protein evidence
may be valid.

The filter operates at the PSM (row) level, before site collapse.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from importlib.resources import files
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CONTAMINANT_PREFIXES: tuple[str, ...] = ("CON__", "Cont_", "contam_")


def get_default_contaminants_fasta() -> Path:
    """Return a filesystem path to the bundled MaxQuant contaminants FASTA.

    The bundled file (``alphaphos/resources/contaminants.fasta``, 246
    sequences) is the canonical MaxQuant-derived list used across the
    proteomics community since ~2010 (keratins, BSA, trypsin, common
    buffer contaminants).

    Returns
    -------
    Path
        Concrete filesystem path to the bundled FASTA.
    """
    resource = files("alphaphos.resources") / "contaminants.fasta"
    # importlib.resources returns a Traversable; converting to Path works
    # when the package is installed as a real directory (the normal case).
    return Path(str(resource))


def parse_fasta_accessions(fasta_path: str | Path) -> set[str]:
    """Extract protein accessions from a FASTA file's header lines.

    Handles the four header conventions found in the MaxQuant
    ``contaminants.fasta``:

    - UniProt: ``>P00761 SWISS-PROT:P00761|TRYP_PIG Trypsin ...``
    - TREMBL with secondary IDs: ``>Q32MB2 TREMBL:Q32MB2;Q86Y46 ...``
    - ENSEMBL: ``>ENSEMBL:ENSBTAP00000034412 (Bos taurus) ...``
    - REFSEQ: ``>REFSEQ:XP_585019 (Bos taurus) ...``

    For each header, the first whitespace-delimited token is parsed (with
    any leading ``DB:`` prefix stripped and any ``|name`` suffix stripped).
    Subsequent tokens of the form ``DB:ACC[;ACC2;...]`` are also scanned
    so that TREMBL secondary IDs are captured.

    Parameters
    ----------
    fasta_path
        Path to a FASTA file with ``>`` headers.

    Returns
    -------
    set[str]
        Unique protein accessions found in the headers.
    """
    path = Path(fasta_path)
    known_db_prefixes = ("SWISS-PROT", "TREMBL", "REFSEQ", "ENSEMBL", "H-INV", "GI")
    accessions: set[str] = set()

    def _yield_accessions_from_token(tok: str) -> Iterable[str]:
        # Strip leading "DB:" if present (case where the primary token starts
        # with a DB prefix, e.g. "ENSEMBL:ENSBTAP00000034412")
        if ":" in tok:
            tok = tok.split(":", 1)[1]
        # ; separates secondary IDs (e.g. "Q32MB2;Q86Y46")
        for sub in tok.split(";"):
            # | separates name suffix (e.g. "P00761|TRYP_PIG")
            acc = sub.split("|", 1)[0].strip()
            if acc:
                yield acc

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            tokens = header.split()
            if not tokens:
                continue
            # Always include the primary token
            accessions.update(_yield_accessions_from_token(tokens[0]))
            # Also include any subsequent DB:ACCESSION tokens (these carry
            # secondary IDs in the MaxQuant FASTA format)
            for tok in tokens[1:3]:
                if ":" in tok and tok.split(":", 1)[0] in known_db_prefixes:
                    accessions.update(_yield_accessions_from_token(tok))
    return accessions


def _is_contaminant_protein(
    protein: str,
    accession_set: set[str],
    prefix_patterns: tuple[str, ...],
) -> bool:
    """True if a single protein ID is a contaminant.

    A protein is a contaminant if it starts with any of ``prefix_patterns``
    OR if (after stripping a known prefix, if any) its accession is in
    ``accession_set``.
    """
    protein = protein.strip()
    if not protein:
        return False
    for prefix in prefix_patterns:
        if protein.startswith(prefix):
            return True
    # No prefix match — fall back to accession lookup
    if not accession_set:
        return False
    # Strip any UniProt-style isoform suffix (e.g. "P00761-2" → "P00761")
    accession = protein.split("-", 1)[0]
    return accession in accession_set


def filter_contaminants(
    df: pd.DataFrame,
    *,
    contaminants_fasta: str | Path | None = None,
    prefix_patterns: tuple[str, ...] = DEFAULT_CONTAMINANT_PREFIXES,
    protein_groups_column: str = "PG.ProteinGroups",
    require_all_contaminant: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Drop PSM rows whose protein groups are entirely contaminants.

    Parameters
    ----------
    df
        PSM-level DataFrame (e.g. output of ``alphaphos.io.read_psm``).
        Must contain ``protein_groups_column``.
    contaminants_fasta
        Path to a contaminants FASTA. If ``None`` (default), uses the
        bundled MaxQuant ``contaminants.fasta``. Pass an empty FASTA, or
        an empty ``prefix_patterns``, to disable that detection mechanism.
    prefix_patterns
        Protein-ID prefixes that mark a contaminant. Default catches
        MaxQuant's ``CON__``, Spectronaut's ``Cont_``, and the
        Frankenfield/Bilbao ``contam_`` conventions. Pass an empty tuple
        to disable prefix detection.
    protein_groups_column
        Column holding ``;``-separated protein IDs. Default
        ``PG.ProteinGroups`` (Spectronaut convention).
    require_all_contaminant
        If True (default, conservative), drop only rows where **every**
        protein in the group is a contaminant; rows with a mix of real
        proteins and contaminants are kept. If False, drop rows where
        **any** protein is a contaminant.
    verbose
        Log a summary at INFO level.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with the index reset.

    Examples
    --------
    >>> df = read_psm("report.parquet", drop_contaminants=False)
    >>> df_clean = filter_contaminants(df)  # uses bundled MaxQuant FASTA
    >>> df_strict = filter_contaminants(df, require_all_contaminant=False)
    """
    if protein_groups_column not in df.columns:
        if verbose:
            logger.info(
                "filter_contaminants: column %r absent; returning input unchanged",
                protein_groups_column,
            )
        return df

    if contaminants_fasta is None:
        contaminants_fasta = get_default_contaminants_fasta()
    accession_set = parse_fasta_accessions(contaminants_fasta)

    def _row_is_contaminant(value: object) -> bool:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return False
        proteins = [p for p in str(value).split(";") if p.strip()]
        if not proteins:
            return False
        flags = (_is_contaminant_protein(p, accession_set, prefix_patterns) for p in proteins)
        return all(flags) if require_all_contaminant else any(flags)

    mask = df[protein_groups_column].map(_row_is_contaminant)
    n_dropped = int(mask.sum())
    if verbose:
        logger.info(
            "filter_contaminants: dropped %d / %d rows (%.2f%%) "
            "[require_all=%s, fasta_accessions=%d, prefixes=%s]",
            n_dropped,
            len(df),
            100 * n_dropped / max(len(df), 1),
            require_all_contaminant,
            len(accession_set),
            prefix_patterns,
        )
    return df.loc[~mask].reset_index(drop=True)
