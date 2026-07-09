"""Site-cluster labels -> protein-level module IDs.

Collapses per-site hierarchical-clustering labels into protein-level module
assignments via **cluster-signature grouping**: two proteins get the same
module ID iff they have sites in the same set of clusters (i.e. share the
same 0/1 cluster-membership pattern).

Clean-room port of PhosPy's ``clustering.protein_modules.derive_protein_modules``
(MIT-compatible re-implementation of the same algorithm; PhosPy is
GPL-3.0 and is used only as a numerical oracle in parity tests).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaphos.signalome.constants import MODULE_ID_COLUMN, PROTEIN_COLUMN


def derive_protein_modules(
    *,
    site_clusters: pd.Series,
    site_to_protein: pd.Series,
) -> pd.Series:
    """Group proteins by their cluster-participation signature.

    Parameters
    ----------
    site_clusters
        pd.Series indexed by site_id, values are cluster labels (1..K) as
        returned by :attr:`SignalomeClusteringResult.labels`.  Sites with
        cluster label 0 are treated as unassigned (participation = 0 in
        every cluster's membership).
    site_to_protein
        pd.Series indexed by site_id, values are protein identifiers.

    Returns
    -------
    pd.Series
        Indexed by protein, values are module IDs (1..M, where M is the
        number of distinct cluster-participation patterns).  Proteins with
        no clustered sites get module_id 0 -- but this shouldn't happen
        when ``site_clusters`` and ``site_to_protein`` cover the same
        site set.

    Notes
    -----
    Two proteins get the same module_id iff they have sites in exactly
    the same set of clusters, regardless of how many sites per cluster.
    Module IDs are assigned in the order the unique patterns first
    appear when iterating proteins in their input order.
    """
    aligned = site_to_protein.copy()
    aligned.index = pd.Index(aligned.index.astype(str))
    cluster_index = pd.Index(site_clusters.index.astype(str))
    missing = [s for s in cluster_index if s not in aligned.index]
    if missing:
        preview = ", ".join(missing[:3])
        suffix = "..." if len(missing) > 3 else ""
        raise ValueError(f"site_to_protein is missing clustered sites: {preview}{suffix}")
    aligned = aligned.loc[cluster_index].astype(str)

    proteins = pd.Index(aligned.tolist(), dtype=object)
    # crosstab rows = clusters, columns = proteins, cell = count of sites
    membership = pd.crosstab(site_clusters, proteins)
    membership = (membership > 0).astype(int)

    pattern_to_module: dict[tuple[int, ...], int] = {}
    assignments: dict[str, int] = {}
    next_module_id = 1
    for protein in membership.columns:
        pattern = tuple(int(v) for v in membership.loc[:, protein].tolist())
        if pattern not in pattern_to_module:
            pattern_to_module[pattern] = next_module_id
            next_module_id += 1
        assignments[str(protein)] = pattern_to_module[pattern]

    result = pd.Series(assignments, dtype="int64", name=MODULE_ID_COLUMN)
    result.index.name = PROTEIN_COLUMN
    return result


def extract_site_to_protein(
    var: pd.DataFrame,
    *,
    protein_column: str = "protein_group_id",
) -> pd.Series:
    """Pull ``site_key -> protein`` from an alphaPhos ``adata.var``.

    Falls back to parsing ``var.index`` as ``Protein|Gene|Site|Mult`` keys
    when ``protein_column`` is absent (keeps behaviour robust across older
    var schemas).
    """
    if protein_column in var.columns:
        return pd.Series(
            var[protein_column].astype(str).str.split(";").str[0].to_numpy(),
            index=pd.Index(var.index.astype(str)),
            name=PROTEIN_COLUMN,
        )
    # fallback: parse the first field of the alphaPhos site key
    parsed = pd.Index(var.index.astype(str)).str.split("|").str[0].str.split(";").str[0]
    if parsed.isna().any() or (parsed == "").any():
        raise ValueError(
            f"could not extract protein from var.index (first-field parse); "
            f"provide protein_column= explicitly (var columns: {list(var.columns)})"
        )
    return pd.Series(parsed.to_numpy(), index=pd.Index(var.index.astype(str)), name=PROTEIN_COLUMN)


def extract_site_metadata(
    var: pd.DataFrame,
    *,
    site_key_column: str | None = None,
    gene_column: str = "gene",
    residue_column: str = "site_aa",
    position_column: str = "site_position",
    protein_column: str = "protein_group_id",
) -> pd.DataFrame:
    """Extract signalome ``site_metadata`` from an alphaPhos ``adata.var``.

    Populates the columns signalome downstream requires (``site_key``,
    ``display_id``, ``gene_symbol``, ``site``, ``protein``,
    ``protein_accession``, ``isoform_id``) using alphaPhos conventions.

    - ``site_key`` = the var index string (unique per row)
    - ``display_id`` = ``<protein>_<residue><position>`` (matches PhosPy)
    - ``gene_symbol`` = ``var[gene_column]``
    - ``site`` = ``<residue><position>`` (e.g. ``"Y1172"``)
    - ``protein`` / ``protein_accession`` = first accession of a
      protein group
    - ``isoform_id`` = "" (alphaPhos does not track isoforms explicitly)

    Callers can also build ``site_metadata`` themselves and pass it
    directly to ``build_module_assignments`` -- this helper is a
    convenience for the common alphaPhos varm layout.
    """
    from alphaphos.signalome.constants import (
        DISPLAY_ID_COLUMN,
        GENE_SYMBOL_COLUMN,
        ISOFORM_ID_COLUMN,
        PROTEIN_ACCESSION_COLUMN,
        SITE_COLUMN,
        SITE_KEY_COLUMN,
    )

    site_keys = pd.Index(var.index.astype(str))
    if site_key_column is not None and site_key_column in var.columns:
        site_key_values = var[site_key_column].astype(str).to_numpy()
    else:
        site_key_values = site_keys.to_numpy()

    if gene_column in var.columns:
        genes = var[gene_column].astype(str).fillna("").to_numpy()
    else:
        genes = np.array([""] * len(site_keys), dtype=object)

    if residue_column in var.columns and position_column in var.columns:
        residues = var[residue_column].astype(str).to_numpy()
        positions = var[position_column].astype(str).to_numpy()
        sites = np.array([f"{r}{p}" for r, p in zip(residues, positions, strict=True)])
    else:
        sites = np.array([""] * len(site_keys), dtype=object)

    if protein_column in var.columns:
        proteins = var[protein_column].astype(str).str.split(";").str[0].to_numpy()
    else:
        proteins = np.array([""] * len(site_keys), dtype=object)

    display_ids = np.array(
        [
            f"{p}_{s}" if p and s else str(k)
            for p, s, k in zip(proteins, sites, site_key_values, strict=True)
        ]
    )

    return pd.DataFrame(
        {
            SITE_KEY_COLUMN: site_key_values,
            DISPLAY_ID_COLUMN: display_ids,
            GENE_SYMBOL_COLUMN: genes,
            SITE_COLUMN: sites,
            PROTEIN_ACCESSION_COLUMN: proteins,
            ISOFORM_ID_COLUMN: [""] * len(site_keys),
        },
        index=site_keys,
    )
