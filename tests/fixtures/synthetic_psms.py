"""Synthetic Spectronaut PSM factory for spike-in integration tests.

Emits PSM rows in the same tabular schema Spectronaut writes, so that a set
of hand-crafted peptides can be concatenated onto a real ``read_spectronaut``
output (typically the small ``tests/data/egf_mini.tsv`` fixture) and put
through ``collapse_sites``. The factory owns the format details so
tests can express *biology* (protein, sequence, phospho positions, loc probs)
rather than string encoding.

Format facts covered here (verified against ``EGF_diff_exp.tsv``):

* ``EG.PrecursorId`` = ``"_<seq-with-mod-tags>_.<charge>"``. Modification
  tags are ``[Phospho (STY)]``, ``[Carbamidomethyl (C)]``, ``[Oxidation (M)]``
  and are inserted immediately after the modified residue.
* ``EG.PTMLocalizationProbabilities`` = ``"_<seq-with-per-STY-percentages>_"``.
  Every candidate STY residue in the peptide gets a
  ``[Phospho (STY): X%]`` suffix; percentages must sum to
  ``100 * multiplicity``.
* ``PEP.PeptidePosition`` = 1-indexed start of the peptide in the protein.
  Stored as a string (semicolon-joined for multi-protein groups).
* ``PG.ProteinGroups`` / ``PG.Genes`` = semicolon-joined UniProt IDs / gene
  symbols.
* ``EG.TotalQuantity (Settings)`` = LINEAR intensity (collapse takes log2
  internally).

For the phospho-mod tag, the factory places the tag at the positions
provided in ``phospho_positions`` (0-indexed within the peptide). Those
positions are the "top-N localized" residues that Spectronaut would emit
in the ``EG.PrecursorId``. The ``loc_probs`` dict specifies the
per-candidate localization percentage; the top ``multiplicity`` entries
should match ``phospho_positions``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

STY = frozenset("STY")


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _encode_precursor_id(
    base_sequence: str,
    phospho_positions: list[int],
    other_mods: dict[int, str] | None,
    charge: int,
) -> str:
    """Emit ``_ABC[Phospho (STY)]DE[Carbamidomethyl (C)]F_.charge``."""
    mods_at: dict[int, str] = {}
    for p in phospho_positions:
        if base_sequence[p] not in STY:
            raise ValueError(
                f"phospho at position {p} lands on {base_sequence[p]!r}, not STY."
            )
        mods_at[p] = "Phospho (STY)"
    for p, m in (other_mods or {}).items():
        if p in mods_at:
            raise ValueError(
                f"position {p} already has a phospho; cannot also add {m!r}."
            )
        mods_at[p] = m

    out: list[str] = []
    for i, aa in enumerate(base_sequence):
        out.append(aa)
        if i in mods_at:
            out.append(f"[{mods_at[i]}]")
    return f"_{''.join(out)}_.{charge}"


def _encode_loc_probs_string(
    base_sequence: str,
    loc_probs: dict[int, float],
    other_mods: dict[int, str] | None,
) -> str:
    """Emit ``_A_S[Phospho (STY): 56.9%]P..._``."""
    for p in loc_probs:
        if base_sequence[p] not in STY:
            raise ValueError(
                f"loc_probs position {p} is on {base_sequence[p]!r}, not STY."
            )
    out: list[str] = []
    for i, aa in enumerate(base_sequence):
        out.append(aa)
        if i in loc_probs:
            out.append(f"[Phospho (STY): {loc_probs[i]:g}%]")
        elif other_mods and i in other_mods:
            out.append(f"[{other_mods[i]}: 100%]")
    return f"_{''.join(out)}_"


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


SPECTRONAUT_COLUMNS: tuple[str, ...] = (
    "R.Condition",
    "R.FileName",
    "PG.Genes",
    "PG.Organisms",
    "PG.ProteinDescriptions",
    "PG.ProteinGroups",
    "PG.ProteinNames",
    "PG.UniProtIds",
    "PEP.GroupingKey",
    "PEP.PeptidePosition",
    "PEP.StrippedSequence",
    "PEP.Quantity",
    "EG.IsDecoy",
    "EG.ModifiedSequence",
    "EG.PrecursorId",
    "EG.ApexRT",
    "EG.PTMAssayCandidateScore",
    "EG.PTMAssayProbability",
    "EG.PTMLocalizationProbabilities",
    "EG.ProteinPTMLocations",
    "EG.NormalizationFactor",
    "EG.TotalQuantity (Settings)",
    "FG.Charge",
    "FG.PrecMzCalibrated",
    "FG.ApexIonMobility",
    "FG.FWHM",
)


def make_phospho_psm(
    *,
    protein_id: str,
    gene: str,
    base_sequence: str,
    peptide_start: int,
    phospho_positions: list[int],
    loc_probs: dict[int, float],
    intensities_per_run: dict[str, float],
    charge: int = 3,
    other_mods: dict[int, str] | None = None,
    ptm_assay_probability: float = 0.99,
    protein_ids_semicolon: str | None = None,
    genes_semicolon: str | None = None,
) -> pd.DataFrame:
    """Build one Spectronaut PSM row per run.

    Parameters
    ----------
    protein_id
        UniProt accession that ends up in ``PG.UniProtIds`` /
        ``PG.ProteinGroups``. Use unique test IDs (e.g. ``"TEST1_P0"``) so
        the spike-in rows don't collide with real proteins in the base
        dataset.
    gene
        Gene symbol; goes into ``PG.Genes``.
    base_sequence
        Unmodified peptide sequence (uppercase 1-letter AA).
    peptide_start
        1-indexed start position of the peptide in the parent protein.
        Stored in ``PEP.PeptidePosition``.
    phospho_positions
        0-indexed intra-peptide positions where the phospho is *placed*
        (i.e. the top-N localized residues Spectronaut emits in
        ``EG.PrecursorId``). Must be a subset of STY residues.
    loc_probs
        ``{pos_0indexed: percentage}`` covering every candidate STY
        residue in the peptide. Sum should equal ``100 * len(phospho_positions)``
        (this is a soft convention -- not enforced, since Spectronaut
        sometimes rounds).
    intensities_per_run
        ``{R.FileName: linear intensity}``. One PSM row is emitted per key.
    charge
        Precursor charge state; goes into ``FG.Charge`` and the
        ``EG.PrecursorId`` suffix.
    other_mods
        Non-phospho fixed/variable modifications, keyed by 0-indexed
        residue position. Example: ``{4: "Carbamidomethyl (C)"}``.
    ptm_assay_probability
        Value in ``[0, 1]`` for ``EG.PTMAssayProbability`` (Spectronaut's
        overall assay-level confidence). Defaults to 0.99 (high).
    protein_ids_semicolon / genes_semicolon
        Override for multi-protein-group cases where the peptide maps to
        multiple proteins. E.g. ``"P57059;A0A0B4J2F2"``. When None, uses
        the single-protein value from ``protein_id`` / ``gene``.
    """
    _validate(
        base_sequence=base_sequence,
        phospho_positions=phospho_positions,
        loc_probs=loc_probs,
        other_mods=other_mods,
    )
    precursor_id = _encode_precursor_id(
        base_sequence, phospho_positions, other_mods, charge
    )
    loc_string = _encode_loc_probs_string(base_sequence, loc_probs, other_mods)
    pg_ids = protein_ids_semicolon or protein_id
    genes_field = genes_semicolon or gene
    n_prots = pg_ids.count(";") + 1
    pep_pos_field = ";".join(str(peptide_start) for _ in range(n_prots))

    ptm_locations = _encode_protein_ptm_locations(
        base_sequence, peptide_start, phospho_positions, other_mods, n_prots
    )

    rows: list[dict] = []
    for run, intensity in intensities_per_run.items():
        rows.append(
            {
                "R.Condition": "+",
                "R.FileName": run,
                "PG.Genes": genes_field,
                "PG.Organisms": "Homo sapiens",
                "PG.ProteinDescriptions": f"Synthetic test protein {protein_id}",
                "PG.ProteinGroups": pg_ids,
                "PG.ProteinNames": f"{gene}_SYNTHETIC",
                "PG.UniProtIds": pg_ids,
                "PEP.GroupingKey": base_sequence,
                "PEP.PeptidePosition": pep_pos_field,
                "PEP.StrippedSequence": base_sequence,
                "PEP.Quantity": intensity,
                "EG.IsDecoy": False,
                "EG.ModifiedSequence": precursor_id.rsplit(".", 1)[0],
                "EG.PrecursorId": precursor_id,
                "EG.ApexRT": 30.0,
                "EG.PTMAssayCandidateScore": 100.0,
                "EG.PTMAssayProbability": ptm_assay_probability,
                "EG.PTMLocalizationProbabilities": loc_string,
                "EG.ProteinPTMLocations": ptm_locations,
                "EG.NormalizationFactor": 1.0,
                "EG.TotalQuantity (Settings)": intensity,
                "FG.Charge": charge,
                "FG.PrecMzCalibrated": 500.0,
                "FG.ApexIonMobility": 1.0,
                "FG.FWHM": 0.1,
            }
        )
    return pd.DataFrame(rows, columns=list(SPECTRONAUT_COLUMNS))


def _validate(
    *,
    base_sequence: str,
    phospho_positions: list[int],
    loc_probs: dict[int, float],
    other_mods: dict[int, str] | None,
) -> None:
    if not base_sequence.isupper() or not base_sequence.isalpha():
        raise ValueError(f"base_sequence must be uppercase letters, got {base_sequence!r}")
    n = len(base_sequence)
    for p in phospho_positions:
        if not 0 <= p < n:
            raise ValueError(f"phospho position {p} out of range for peptide of length {n}")
    for p in loc_probs:
        if not 0 <= p < n:
            raise ValueError(f"loc_probs position {p} out of range for peptide of length {n}")
    for p in other_mods or ():
        if not 0 <= p < n:
            raise ValueError(f"other_mods position {p} out of range for peptide of length {n}")


def _encode_protein_ptm_locations(
    base_sequence: str,
    peptide_start: int,
    phospho_positions: list[int],
    other_mods: dict[int, str] | None,
    n_prots: int,
) -> str:
    """Emit "(S123);(S123)" style -- one clause per protein in the group."""
    parts: list[str] = []
    for p in sorted(phospho_positions):
        aa = base_sequence[p]
        parts.append(f"{aa}{peptide_start + p}")
    for p, m in sorted((other_mods or {}).items()):
        if m.startswith("Carbamidomethyl"):
            parts.append(f"C{peptide_start + p}")
    if not parts:
        return ";".join([""] * n_prots)
    per_prot = "(" + ",".join(parts) + ")"
    return ";".join([per_prot] * n_prots)


# ---------------------------------------------------------------------------
# Sugar for the common "spike a bunch of PSMs into a real reader output" flow.
# ---------------------------------------------------------------------------


@dataclass
class Spike:
    """One synthetic PSM specification for the composer below."""

    protein_id: str
    gene: str
    base_sequence: str
    peptide_start: int
    phospho_positions: list[int]
    loc_probs: dict[int, float]
    intensities_per_run: dict[str, float]
    charge: int = 3
    other_mods: dict[int, str] | None = None
    ptm_assay_probability: float = 0.99
    protein_ids_semicolon: str | None = None
    genes_semicolon: str | None = None


def spike_into(base_psm_df: pd.DataFrame, spikes: Iterable[Spike]) -> pd.DataFrame:
    """Append synthetic PSMs onto a real reader output.

    Preserves ``base_psm_df.attrs`` (which ``read_spectronaut`` populates
    with source provenance) so downstream pipeline stages keep working.
    """
    rows = [
        make_phospho_psm(
            protein_id=s.protein_id,
            gene=s.gene,
            base_sequence=s.base_sequence,
            peptide_start=s.peptide_start,
            phospho_positions=s.phospho_positions,
            loc_probs=s.loc_probs,
            intensities_per_run=s.intensities_per_run,
            charge=s.charge,
            other_mods=s.other_mods,
            ptm_assay_probability=s.ptm_assay_probability,
            protein_ids_semicolon=s.protein_ids_semicolon,
            genes_semicolon=s.genes_semicolon,
        )
        for s in spikes
    ]
    if not rows:
        return base_psm_df.copy()
    keep_cols = [c for c in SPECTRONAUT_COLUMNS if c in base_psm_df.columns]
    combined = pd.concat(
        [base_psm_df[keep_cols], *[r[keep_cols] for r in rows]],
        ignore_index=True,
    )
    combined.attrs = dict(base_psm_df.attrs)
    return combined


def uniform_intensities(runs: Iterable[str], value: float) -> dict[str, float]:
    """Shortcut for equal intensity across all runs."""
    return dict.fromkeys(runs, float(value))
