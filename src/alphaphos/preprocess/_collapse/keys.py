"""Build the canonical site identifiers used across alphaPhos.

Two identifier flavors are produced for each phosphosite, both using ``|`` as
the field delimiter (chosen because it never appears in any of the source
fields: gene names, UniProt / Ensembl / RefSeq accessions, or amino-acid
letters):

* **Full key** (``adata.var.index``, canonical unique identifier)::

      Protein|Gene|Site|Mult
      A0A0B4J2F2|SIK1B|S575|M1

  Includes the protein-group id so that isoforms / paralogs with the same
  gene name but different sequences don't collide. This is what analysis
  code should join on.

* **Short label** (``adata.var["short_key"]``, human-readable)::

      Gene|Site|Mult
      SIK1B|S575|M1

  Convenient for plotting axis labels and tables. When two different
  protein groups yield the same short label, we auto-suffix ``#2``,
  ``#3``, ... in a deterministic order (ascending by full key) so the
  ``short_key`` column stays unique too. Collision events are surfaced
  in ``adata.uns["alphaphos"]["short_key_collisions"]`` for provenance.

The delimiter migration from ``~`` and ``_`` to ``|``:

* Old format: ``A0A0B4J2F2~SIK1B_S575_M1`` -- used two different delimiters
  because gene names historically could contain ``_``.
* New format: ``A0A0B4J2F2|SIK1B|S575|M1`` -- ``|`` is safe against every
  gene / accession convention we've encountered.

If you're chasing a bug where a downstream tool broke on the format change,
this module is the ONLY place that produces the string; every parser that
consumes it should be updated together.
"""

from __future__ import annotations

from collections import defaultdict

# ---------------------------------------------------------------------------
# String builders (pure, no state)
# ---------------------------------------------------------------------------


def build_full_key(
    protein_group: str,
    gene: str,
    aa: str,
    position: int,
    multiplicity: int,
) -> str:
    """Return ``"{ProteinGroup}|{Gene}|{aa}{position}|M{multiplicity}"``.

    This is the canonical site identifier (``adata.var.index``). It is
    proteoform-specific: two protein groups sharing a gene name will produce
    two different full keys.

    Parameters
    ----------
    protein_group : str
        First protein-group id (semicolons split off upstream).
    gene : str
        Gene name (semicolons split off upstream). May contain any characters
        except ``|``.
    aa : str
        Single-letter amino acid (``S``, ``T``, or ``Y`` for phospho).
    position : int
        1-indexed absolute position of the residue in the parent protein
        (peptide_start + intra-peptide_offset - 1).
    multiplicity : int
        Number of phospho groups on the peptide of origin, clamped to
        ``max=3`` (the ``M1``/``M2``/``M3+`` convention).

    Returns
    -------
    str
        The composed key. Returns ``"Error_key"`` if any input is bad, so
        the calling row can be filtered out downstream without a crash.
    """
    try:
        return f"{protein_group}|{gene}|{aa}{int(position)}|M{int(multiplicity)}"
    except (TypeError, ValueError):
        return "Error_key"


def build_short_key(gene: str, aa: str, position: int, multiplicity: int) -> str:
    """Return ``"{Gene}|{aa}{position}|M{multiplicity}"``.

    The human-readable label used in plots and tables. Two protein groups
    with the same gene name at the same residue produce the same short
    label -- :func:`resolve_short_key_collisions` disambiguates by suffixing
    ``#N`` when this happens.
    """
    try:
        return f"{gene}|{aa}{int(position)}|M{int(multiplicity)}"
    except (TypeError, ValueError):
        return "Error_short_key"


def build_pg_key(protein_group: str, aa: str, position: int, multiplicity: int) -> str:
    """Return ``"{ProteinGroup}|{aa}{position}|M{multiplicity}"``.

    Used when analysis needs to group at protein-group granularity ignoring
    the gene-name annotation (e.g. some FASTA lookups where gene isn't the
    primary key).
    """
    try:
        return f"{protein_group}|{aa}{int(position)}|M{int(multiplicity)}"
    except (TypeError, ValueError):
        return "Error_pg_key"


# ---------------------------------------------------------------------------
# Modified-sequence helpers (used for the human-readable UPD_seq column)
# ---------------------------------------------------------------------------


def build_modified_sequence(clean_sequence: str, phospho_position: int) -> str:
    """Insert a lowercased-and-starred marker at the phospho position.

    The output is a compact visualization of "where on this peptide the
    phospho sits" that's readable at a glance:

    >>> build_modified_sequence("PEPTIDE", 4)
    'PEPt*IDE'

    Position is 1-indexed. Out-of-range positions return the sequence
    unchanged (defensive; upstream filtering already drops these).
    """
    if phospho_position < 1 or phospho_position > len(clean_sequence):
        return clean_sequence

    idx = phospho_position - 1
    return clean_sequence[:idx] + clean_sequence[idx].lower() + "*" + clean_sequence[idx + 1 :]


def get_phospho_amino_acid(sequence: str, position: int) -> str:
    """Return the amino acid at ``position`` (1-indexed), or ``"X"`` on error.

    Used to fill the ``site_aa`` column when we've exploded a precursor into
    per-position rows. ``"X"`` is the "unknown" sentinel and lets the row
    survive to be filtered later based on whether the AA is in {S, T, Y}.
    """
    try:
        if position < 1 or position > len(sequence):
            return "X"
        result = sequence[position - 1 : position]
        return result if result else "X"
    except (IndexError, TypeError):
        return "X"


# ---------------------------------------------------------------------------
# Collision resolution for short keys
# ---------------------------------------------------------------------------


def resolve_short_key_collisions(
    short_keys: list[str],
    full_keys: list[str],
) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Deterministically suffix duplicate ``short_keys``.

    When two rows share the same ``short_key`` (typically because the same
    gene has multiple protein-group entries), we keep the first occurrence
    (ordered by ascending ``full_key``) unchanged and suffix the rest with
    ``#2``, ``#3``, ...

    Parameters
    ----------
    short_keys : list[str]
        One short key per site (``adata.var["short_key"]`` candidate).
    full_keys : list[str]
        One full key per site, parallel to ``short_keys``. Used only as the
        tiebreaker for ordering suffixes.

    Returns
    -------
    resolved_short_keys : list[str]
        Same length as ``short_keys``; each entry is either the original
        value or ``value#N`` (N >= 2).
    collisions : list[tuple[str, list[str]]]
        One entry per short_key that had duplicates. ``(short_key, [full_keys ...])``
        sorted for stable diagnostic output. Empty when there are none.

    Examples
    --------
    >>> shorts = ['SIK1B|S575|M1', 'SIK1B|S575|M1', 'AKT1|T308|M1']
    >>> fulls  = ['P2|SIK1B|S575|M1', 'P1|SIK1B|S575|M1', 'Q1|AKT1|T308|M1']
    >>> resolve_short_key_collisions(shorts, fulls)
    (['SIK1B|S575|M1#2', 'SIK1B|S575|M1', 'AKT1|T308|M1'], [('SIK1B|S575|M1', ['P1|SIK1B|S575|M1', 'P2|SIK1B|S575|M1'])])
    """
    if len(short_keys) != len(full_keys):
        raise ValueError(
            f"short_keys ({len(short_keys)}) and full_keys ({len(full_keys)}) "
            "must be the same length."
        )

    # Group row indices by short_key so we can find duplicates in one pass.
    by_short: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(short_keys):
        by_short[s].append(i)

    resolved = list(short_keys)  # will mutate in place
    collisions: list[tuple[str, list[str]]] = []

    for short_key, indices in by_short.items():
        if len(indices) < 2:
            continue
        # Order the duplicates by ascending full_key -- deterministic across
        # runs regardless of input row order.
        ordered = sorted(indices, key=lambda i: full_keys[i])
        # The first entry keeps the original short key; the rest are suffixed.
        for suffix_n, i in enumerate(ordered[1:], start=2):
            resolved[i] = f"{short_key}#{suffix_n}"
        collisions.append((short_key, sorted(full_keys[i] for i in indices)))

    # Deterministic output ordering for logging / diagnostics.
    collisions.sort(key=lambda kv: kv[0])
    return resolved, collisions
