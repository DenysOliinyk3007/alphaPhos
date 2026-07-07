"""Map user site identifiers to the canonical ``Protein_AApos`` IDs used
in the emitted GMT libraries.

This is the "identity spine" of the enrichment module: get it wrong and
every enrichment p-value is silently biased.  The engine is designed so
that dropped hits are always **visible** to the caller.

Two-tier matching (Phase 2 scope):

1. **UniProt + residue + position** (exact).  The DB stores
   ``substrate_uniprot`` as either a bare accession (``P00533``) or a
   semicolon-joined multi-mapping (``Q7Z6Z7;A6NHW2``).  We explode the DB
   into a per-accession lookup so a user's single accession matches any
   position in a multi-mapped DB row.
2. **Gene + residue + position** (fallback).  Trades isoform robustness
   for gene-symbol ambiguity — kept as a rescue tier when UniProt lookup
   fails (typical for old datasets that carry only gene symbols).

Sequence-window (±7) matching is architecturally in scope but deferred
to a later patch: it requires attaching ±7 windows to every DB site
from a reference proteome.  Tiers 2+3 hit >95% match rate on typical
Spectronaut output; window matching earns its place only if that number
drops for a specific study.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from alphaphos.enrichment.db import (
    COL_GENE,
    COL_POSITION,
    COL_RESIDUE,
    COL_UNIPROT,
    load_ptm_db,
    site_id,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# Regex for alphaphos site keys: Protein|Gene|<AA><pos>|M<mult>
# Only accepts S / T / Y as the phospho residue.
_ALPHAPHOS_KEY_RE = re.compile(r"^([^|]+)\|([^|]+)\|([STY])(\d+)\|M(\d+)$")


@dataclass(frozen=True)
class ParsedKey:
    """Structured view of an alphaphos ``Protein|Gene|Site|Mult`` key."""

    query_key: str
    protein: str  # first accession from a semicolon-joined group
    gene: str
    residue: str  # S / T / Y
    position: int
    multiplicity: int

    @property
    def canonical_site_id(self) -> str:
        return site_id(self.protein, self.residue, self.position)


@dataclass(frozen=True)
class MatchResult:
    """Outcome of :func:`match_sites`.

    Attributes
    ----------
    matched
        One row per successfully-matched query.  Columns:
        ``query_key``, ``site_id``, ``match_tier``
        (``"uniprot_position"`` or ``"gene_position"``).
    unmatched
        One row per dropped query.  Columns: ``query_key``, ``reason``.
    stats
        Summary counters: ``n_input``, ``n_parsed``, ``n_matched_tier2``,
        ``n_matched_tier3``, ``n_unmatched``, ``match_rate``.
    """

    matched: pd.DataFrame
    unmatched: pd.DataFrame
    stats: dict[str, int | float]

    def log_summary(self) -> None:
        s = self.stats
        logger.info(
            "match_sites: %d/%d matched (%.1f%%) — tier2=%d tier3=%d unmatched=%d",
            s["n_input"] - s["n_unmatched"],
            s["n_input"],
            s["match_rate"] * 100,
            s["n_matched_tier2"],
            s["n_matched_tier3"],
            s["n_unmatched"],
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_alphaphos_key(key: str) -> ParsedKey | None:
    """Parse a single ``Protein|Gene|Site|Mult`` key.

    Multi-mapped protein groups (``P1;P2``) collapse to the first
    accession — same convention as :func:`alphaphos.enrichment.site_id`.
    Returns ``None`` on malformed input; caller decides whether that's
    a hard error or a soft drop.
    """
    if not isinstance(key, str):
        return None
    m = _ALPHAPHOS_KEY_RE.match(key)
    if m is None:
        return None
    protein_group, gene, residue, pos_str, mult_str = m.groups()
    protein = protein_group.split(";", 1)[0]
    return ParsedKey(
        query_key=key,
        protein=protein,
        gene=gene,
        residue=residue,
        position=int(pos_str),
        multiplicity=int(mult_str),
    )


def match_sites(
    query_keys: pd.Index | list[str],
    *,
    db_path: str | Path | None = None,
    db: pd.DataFrame | None = None,
) -> MatchResult:
    """Match alphaphos-style ``Protein|Gene|Site|Mult`` keys to the DB.

    Parameters
    ----------
    query_keys
        Iterable of alphaphos var-name strings.  ``adata.var_names``,
        ``result.index``, and plain lists are all accepted.
    db_path
        Path to the DB parquet.  Ignored if ``db`` is provided.
    db
        Pre-loaded DB DataFrame.  Skips the file-read; useful when
        matching several query sets in the same session.

    Returns
    -------
    MatchResult
        See the dataclass.  ``matched.site_id`` is the canonical
        ``Protein_AApos`` ID that indexes the GMT libraries; hand it to
        the Phase-3 enrichment engine.
    """
    keys = list(query_keys)
    if db is None:
        db = load_ptm_db(db_path)

    up_index, gene_index = _build_lookup_indices(db)

    parsed_rows: list[ParsedKey] = []
    unmatched_rows: list[dict] = []
    for k in keys:
        parsed = parse_alphaphos_key(k)
        if parsed is None:
            unmatched_rows.append({"query_key": k, "reason": "unparseable_key"})
            continue
        parsed_rows.append(parsed)

    matched_rows: list[dict] = []
    n_tier2 = 0
    n_tier3 = 0
    for parsed in parsed_rows:
        canonical = parsed.canonical_site_id
        up_key = (parsed.protein, parsed.residue, parsed.position)
        if up_key in up_index:
            matched_rows.append(
                {
                    "query_key": parsed.query_key,
                    "site_id": canonical,
                    "match_tier": "uniprot_position",
                }
            )
            n_tier2 += 1
            continue
        gene_key = (parsed.gene.upper(), parsed.residue, parsed.position)
        if gene_key in gene_index:
            # Gene-tier match: canonical id uses the DB's uniprot for
            # this gene+site (not the user's), so the resulting site_id
            # still points into the emitted GMT libraries.
            db_uniprot = gene_index[gene_key]
            matched_rows.append(
                {
                    "query_key": parsed.query_key,
                    "site_id": site_id(db_uniprot, parsed.residue, parsed.position),
                    "match_tier": "gene_position",
                }
            )
            n_tier3 += 1
            continue
        unmatched_rows.append({"query_key": parsed.query_key, "reason": "not_in_db"})

    matched = pd.DataFrame(matched_rows, columns=["query_key", "site_id", "match_tier"])
    unmatched = pd.DataFrame(unmatched_rows, columns=["query_key", "reason"])
    stats = {
        "n_input": len(keys),
        "n_parsed": len(parsed_rows),
        "n_matched_tier2": n_tier2,
        "n_matched_tier3": n_tier3,
        "n_unmatched": len(unmatched_rows),
        "match_rate": (len(matched) / max(len(keys), 1)),
    }
    return MatchResult(matched=matched, unmatched=unmatched, stats=stats)


def attach_site_ids(
    result: pd.DataFrame,
    *,
    db_path: str | Path | None = None,
    db: pd.DataFrame | None = None,
    key_column: str | None = None,
) -> pd.DataFrame:
    """Add a ``site_id`` column to a diff-exp DataFrame indexed by
    alphaphos keys.

    Convenience wrapper for the enrichment-input prep step.  Rows whose
    key doesn't match are kept in the output with ``site_id = NaN``;
    the caller must decide whether to drop them.  The full
    :class:`MatchResult` is attached at ``result.attrs["match_result"]``
    for post-hoc inspection.

    Parameters
    ----------
    result
        DataFrame from :func:`alphaphos.diff_exp_limma` (indexed by
        var_names) or any DataFrame carrying alphaphos-style keys.
    key_column
        Column holding the keys.  If ``None``, uses ``result.index``.
    """
    keys = result[key_column] if key_column is not None else result.index
    mr = match_sites(keys, db_path=db_path, db=db)
    key_to_id = dict(zip(mr.matched["query_key"], mr.matched["site_id"], strict=True))
    out = result.copy()
    lookup_source = out[key_column] if key_column is not None else out.index
    out["site_id"] = [key_to_id.get(k) for k in lookup_source]
    out.attrs["match_result"] = mr
    mr.log_summary()
    return out


# ---------------------------------------------------------------------------
# Lookup index construction
# ---------------------------------------------------------------------------


def _build_lookup_indices(
    db: pd.DataFrame,
) -> tuple[set[tuple[str, str, int]], dict[tuple[str, str, int], str]]:
    """Emit (uniprot_set, gene_map) for two-tier matching.

    * ``uniprot_set`` contains ``(uniprot, residue, position)`` triples.
      Multi-mapped DB rows (``P1;P2``) are expanded so a query hitting
      P2 matches even if the DB stored P1 first.
    * ``gene_map`` maps ``(gene_upper, residue, position)`` to a
      representative uniprot from the DB.  First-hit wins on collisions
      (gene ambiguity is why this is the *fallback* tier).
    """
    uniprot_triples: set[tuple[str, str, int]] = set()
    gene_map: dict[tuple[str, str, int], str] = {}

    # fillna("") before astype(str) so a NaN/NA in any pandas backend
    # (object / pyarrow-string / nullable-string) becomes "" rather than
    # leaking through as a float/pd.NA that later .split() calls choke on.
    for uniprot_field, gene, residue, position in zip(
        db[COL_UNIPROT].fillna("").astype(str),
        db[COL_GENE].fillna("").astype(str),
        db[COL_RESIDUE].fillna("").astype(str),
        db[COL_POSITION],
        strict=True,
    ):
        if pd.isna(position) or not uniprot_field:
            continue
        pos = int(position)
        # Uniprot tier: expand semicolon-joined groups
        for accession in uniprot_field.split(";"):
            accession = accession.strip()
            if not accession or accession == "nan":
                continue
            # Strip isoform tag (P00533-2 -> P00533).  Isoform-specific
            # positions rarely match the canonical proteome downstream;
            # falling back to canonical is the pragmatic choice.
            base_accession = accession.split("-", 1)[0]
            uniprot_triples.add((base_accession, residue, pos))
        # Gene tier: first occurrence wins.  Store one representative
        # uniprot per gene+site so we can emit a stable canonical id.
        first_accession = uniprot_field.split(";", 1)[0].split("-", 1)[0]
        gene_key = (gene.upper(), residue, pos)
        if gene_key not in gene_map:
            gene_map[gene_key] = first_accession

    return uniprot_triples, gene_map
