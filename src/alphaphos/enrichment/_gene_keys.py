"""Shared helpers for the gene-level pathway wrappers (ORA + preranked GSEA).

Both :mod:`alphaphos.enrichment.pathway` and
:mod:`alphaphos.enrichment.pathway_gsea` consume a per-site differential
result indexed by alphaPhos keys, project sites to gene symbols and query
Enrichr gene-set libraries through gseapy.  The pieces they share live here
so the two stay identical (they used to be copy-pasted).
"""

from __future__ import annotations

import pandas as pd

DEFAULT_LIBRARIES_HUMAN: list[str] = [
    "GO_Biological_Process_2023",
    "GO_Molecular_Function_2023",
    "GO_Cellular_Component_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
    "MSigDB_Hallmark_2020",
]

DEFAULT_LIBRARIES_MOUSE: list[str] = [
    "GO_Biological_Process_2023",
    "GO_Molecular_Function_2023",
    "GO_Cellular_Component_2023",
    "KEGG_2019_Mouse",
    "Reactome_2022",
    "MSigDB_Hallmark_2020",
]


def normalise_organism(organism: str) -> str:
    """Lower-case and validate the organism (gseapy wants lower-case)."""
    org = str(organism).lower()
    if org not in ("human", "mouse"):
        raise ValueError(f"organism must be 'human' or 'mouse'; got {organism!r}")
    return org


def default_libraries(organism: str) -> list[str]:
    return (DEFAULT_LIBRARIES_HUMAN if organism == "human" else DEFAULT_LIBRARIES_MOUSE).copy()


def require_gseapy(caller: str):
    """Import gseapy or raise with the install hint."""
    try:
        import gseapy as gp
    except ImportError as exc:
        raise ImportError(
            f"{caller} requires gseapy. Install with: "
            "pip install 'alphaphos[enrichment]' or pip install gseapy>=1.1"
        ) from exc
    return gp


def extract_keys(result: pd.DataFrame, key_column: str | None) -> list[str]:
    if key_column is not None:
        return list(result[key_column])
    return list(result.index)


def keys_to_genes(keys: list[str]) -> dict[str, str]:
    """Extract the gene symbol from an alphaPhos site or precursor key.

    Both key formats share the ``Protein|Gene|...`` prefix -- site keys are
    ``Protein|Gene|Site|Mult`` (4 fields), precursor keys are
    ``Protein|Gene|Peptide|Charge|Mods`` (5 fields).  The gene is always the
    second pipe-delimited field, so a permissive split works for both.  Keys
    with an empty / ``nan`` gene are omitted.
    """
    out: dict[str, str] = {}
    for k in keys:
        parts = str(k).split("|")
        if len(parts) < 2:
            continue
        gene = parts[1].strip()
        if gene and gene.lower() != "nan":
            out[str(k)] = gene
    return out


def gene_map(
    result: pd.DataFrame,
    keys: list[str],
    *,
    gene_column: str | None,
) -> dict[str, str]:
    """``key -> gene`` from an explicit gene column (proteome input) or the key parser."""
    if gene_column is None:
        mapping = keys_to_genes(keys)
    else:
        if gene_column not in result.columns:
            raise ValueError(
                f"gene_column={gene_column!r} not found in diff_exp_result "
                f"columns (available: {list(result.columns)})"
            )
        # Semicolon-joined multi-gene entries: keep the first name.
        gene_series = result[gene_column].astype(str).str.split(";").str[0]
        mapping = {
            str(k): g
            for k, g in zip(keys, gene_series, strict=True)
            if isinstance(g, str) and g and g.lower() != "nan"
        }
    if not mapping:
        raise ValueError(
            "No parseable gene names extracted from diff_exp_result.  For "
            "phospho input, keys must be in 'Protein|Gene|Site|Mult' format.  "
            "For proteome input, attach a gene column and pass "
            "gene_column='<name>' (e.g. gene_column='PG_Genes')."
        )
    return mapping
