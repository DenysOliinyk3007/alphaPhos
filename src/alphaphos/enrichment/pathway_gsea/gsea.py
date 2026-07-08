"""Gene-level preranked GSEA on limma diff-exp output via gseapy.

Classical (Subramanian 2005) preranked GSEA against the same Enrichr
libraries used by :func:`alphaphos.enrichment.pathway_enrichment`.
Complements the ORA path -- GSEA uses the full ranked list without a
significance threshold, so it can pick up coherently-moving pathways
that ORA misses on subtle effects.

Phospho-specific wrinkle: GSEA needs **one number per gene**, but a
phospho experiment produces many sites per gene with different log2FCs.
The site-to-gene collapse strategy is configurable
(:attr:`site_to_gene_agg`); the default keeps the site with the largest
``|log2fc|`` (signed) per gene, which preserves peak regulation without
averaging away opposing sites.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


_OUTPUT_COLUMNS: list[str] = [
    "library",
    "term",
    "es",
    "nes",
    "p_value",
    "fdr",
    "size",
    "leading_edge",
    "direction",
]


def pathway_gsea(
    diff_exp_result: pd.DataFrame,
    *,
    stat_col: str = "log2fc",
    libraries: list[str] | None = None,
    site_to_gene_agg: Literal["max_abs", "top_significant"] = "max_abs",
    min_set_size: int = 15,
    max_set_size: int = 500,
    n_permutations: int = 1000,
    seed: int = 42,
    organism: Literal["human", "mouse"] = "human",
    key_column: str | None = None,
    gene_column: str | None = None,
    fdr_col: str = "fdr",
    cache_dir: str | Path | None = None,
    threads: int = 1,
) -> pd.DataFrame:
    """Preranked GSEA on gene-collapsed limma output.

    Parameters
    ----------
    diff_exp_result
        DataFrame from :func:`alphaphos.diff_exp_limma`, indexed by
        alphaPhos ``Protein|Gene|Site|Mult`` keys. Must carry the
        ranking column (default ``log2fc``); the FDR column is only
        required when ``site_to_gene_agg="top_significant"``.
    stat_col
        Column used as the ranking metric. Default ``"log2fc"``;
        ``"t_stat"`` (moderated) is a defensible alternative that
        accounts for per-site variability.
    libraries
        Enrichr library slugs. Defaults to GO BP/MF/CC + KEGG + Reactome
        + MSigDB Hallmark for the chosen organism. Any Enrichr library
        works; see https://maayanlab.cloud/Enrichr/#libraries.
    site_to_gene_agg
        How to collapse multiple sites per gene down to one ranked value:

        - ``"max_abs"`` (default) -- keep the site with the largest
          ``|stat|`` per gene, retain its signed value. Captures peak
          regulation; ignores contradicting sites on the same gene.
        - ``"top_significant"`` -- keep the site with the lowest
          per-site FDR per gene, retain its signed stat. Prioritises
          statistical evidence over effect size.
    min_set_size, max_set_size
        Gene-set size bounds (after restricting to the ranked universe).
        Defaults follow the gseapy prerank convention (15 / 500).
    n_permutations
        Permutations for the empirical null. 1000 is the gseapy default;
        10000 gives FDR resolution below 0.001.
    seed
        RNG seed passed to gseapy for reproducibility.
    organism
        ``"human"`` or ``"mouse"``. Selects the default library set.
    key_column
        Column carrying site keys if they live outside the index.
    gene_column
        Column in ``diff_exp_result`` carrying gene symbols directly.
        Use for **proteome input** (protein-group keys with no embedded
        gene name), e.g. ``result["gene"] = adata_prot.var.loc[result.index,
        "PG_Genes"].values`` then ``gene_column="gene"``.  Semicolon-joined
        multi-gene entries take the first name.  When set, overrides the
        phospho-key parser.  ``None`` (default) uses the phospho
        ``Protein|Gene|Site|Mult`` parser.
    fdr_col
        Column carrying per-site FDR. Only consulted when
        ``site_to_gene_agg="top_significant"``.
    cache_dir
        Where gseapy caches downloaded library GMTs. Passed through as
        ``outdir``; ``None`` uses gseapy's own cache location.
    threads
        Worker threads for gseapy's permutation loop. Default 1
        (deterministic on the same seed); raise for larger runs.

    Returns
    -------
    pandas.DataFrame with columns::

        library, term, es, nes, p_value, fdr, size, leading_edge, direction

    ``es`` is the raw enrichment score, ``nes`` is the size-normalised
    score, ``size`` is the number of collapsed genes in the pathway
    overlap, ``leading_edge`` is the semicolon-joined list of genes
    driving the enrichment, ``direction`` is ``"up"`` if ``nes >= 0``
    else ``"down"``. Sorted by ``fdr`` within each library.
    ``.attrs["provenance"]`` records libraries, method, stat_col,
    site_to_gene_agg, n_permutations, seed, gseapy version, and
    site-to-gene collapse statistics.
    """
    if site_to_gene_agg not in ("max_abs", "top_significant"):
        raise ValueError(
            f"site_to_gene_agg must be 'max_abs' or 'top_significant'; got {site_to_gene_agg!r}"
        )
    organism = organism.lower()  # type: ignore[assignment]
    if organism not in ("human", "mouse"):
        raise ValueError(f"organism must be 'human' or 'mouse'; got {organism!r}")

    try:
        import gseapy as gp
    except ImportError as exc:
        raise ImportError(
            "pathway_gsea requires gseapy. Install with: "
            "pip install 'alphaphos[enrichment]' or pip install gseapy>=1.1"
        ) from exc

    if libraries is None:
        libraries = (
            DEFAULT_LIBRARIES_HUMAN.copy()
            if organism == "human"
            else DEFAULT_LIBRARIES_MOUSE.copy()
        )
    if not libraries:
        raise ValueError("libraries must be non-empty")

    keys = _extract_keys(diff_exp_result, key_column)
    if stat_col not in diff_exp_result.columns:
        if "F" in diff_exp_result.columns:
            raise ValueError(
                f"stat_col={stat_col!r} not in diff_exp_result columns, but "
                "an 'F' column is present -- this looks like diff_exp_anova "
                "output.  Preranked GSEA requires a **signed** ranking "
                "metric (positive = up), which ANOVA does not produce.  "
                "For direction-agnostic pathway enrichment on ANOVA hits, "
                "use pathway_enrichment(direction='any').  For signed GSEA, "
                "re-run per-contrast with diff_exp_limma_contrasts(...) and "
                "pass its per-contrast DataFrames."
            )
        raise ValueError(
            f"stat_col={stat_col!r} not in diff_exp_result columns "
            f"(available: {list(diff_exp_result.columns)})"
        )
    stats = diff_exp_result[stat_col].to_numpy()
    fdrs = diff_exp_result[fdr_col].to_numpy() if site_to_gene_agg == "top_significant" else None

    if gene_column is not None:
        if gene_column not in diff_exp_result.columns:
            raise ValueError(
                f"gene_column={gene_column!r} not found in diff_exp_result "
                f"columns (available: {list(diff_exp_result.columns)})"
            )
        gene_series = diff_exp_result[gene_column].astype(str).str.split(";").str[0]
        key_to_gene = {
            str(k): g
            for k, g in zip(keys, gene_series, strict=True)
            if isinstance(g, str) and g and g.lower() != "nan"
        }
    else:
        key_to_gene = _keys_to_genes(keys)
    if not key_to_gene:
        raise ValueError(
            "No parseable gene names extracted from diff_exp_result.  For "
            "phospho input, keys must be in 'Protein|Gene|Site|Mult' format.  "
            "For proteome input, attach a gene column and pass "
            "gene_column='<name>'."
        )

    gene_to_stat, collapse_stats = _collapse_sites_to_genes(
        keys=keys,
        stats=stats,
        fdrs=fdrs,
        key_to_gene=key_to_gene,
        strategy=site_to_gene_agg,
    )
    if not gene_to_stat:
        raise ValueError(
            "Empty ranked list after site->gene collapse -- check "
            f"{stat_col!r} for NaN-only or that keys parse correctly."
        )

    ranked = pd.Series(gene_to_stat, name="stat").sort_values(ascending=False)
    logger.info(
        "pathway_gsea: %d sites -> %d ranked genes (strategy=%s), %d libraries, %d permutations",
        collapse_stats["n_sites_used"],
        len(ranked),
        site_to_gene_agg,
        len(libraries),
        n_permutations,
    )

    outdir = str(cache_dir) if cache_dir is not None else None
    frames: list[pd.DataFrame] = []
    for library in libraries:
        try:
            pre = gp.prerank(
                rnk=ranked,
                gene_sets=library,
                permutation_num=n_permutations,
                min_size=min_set_size,
                max_size=max_set_size,
                seed=seed,
                threads=threads,
                outdir=outdir,
                verbose=False,
                no_plot=True,
            )
        except Exception as exc:
            logger.warning("pathway_gsea: library %s failed: %s", library, exc)
            continue
        frame = _normalise_prerank_output(pre.res2d, library=library)
        frames.append(frame)

    if not frames:
        out = pd.DataFrame(columns=_OUTPUT_COLUMNS)
    else:
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["library", "fdr"], kind="stable").reset_index(drop=True)

    out.attrs["provenance"] = {
        "method": "gsea_prerank",
        "libraries": libraries,
        "stat_col": stat_col,
        "site_to_gene_agg": site_to_gene_agg,
        "n_permutations": n_permutations,
        "n_ranked_genes": len(ranked),
        "n_sites_used": collapse_stats["n_sites_used"],
        "n_sites_dropped": collapse_stats["n_sites_dropped"],
        "n_unparseable_keys": collapse_stats["n_unparseable_keys"],
        "min_set_size": min_set_size,
        "max_set_size": max_set_size,
        "organism": organism,
        "seed": seed,
        "gseapy_version": getattr(gp, "__version__", None),
    }
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _extract_keys(
    result: pd.DataFrame,
    key_column: str | None,
) -> list[str]:
    if key_column is not None:
        return list(result[key_column])
    return list(result.index)


def _keys_to_genes(keys: list[str]) -> dict[str, str]:
    """Extract the gene symbol from an alphaPhos site or precursor key.

    Both formats share the ``Protein|Gene|...`` prefix (site key has 4
    fields, precursor key has 5); a permissive pipe split works for both.
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


def _collapse_sites_to_genes(
    *,
    keys: list[str],
    stats: np.ndarray,
    fdrs: np.ndarray | None,
    key_to_gene: dict[str, str],
    strategy: str,
) -> tuple[dict[str, float], dict[str, int]]:
    """Collapse per-site stats to per-gene stats.

    Returns ``(gene_to_signed_stat, stats_dict)`` where ``stats_dict``
    reports how many input sites were used vs dropped, for provenance.
    """
    if strategy == "max_abs":
        return _collapse_max_abs(keys, stats, key_to_gene)
    if strategy == "top_significant":
        if fdrs is None:
            raise ValueError("top_significant requires per-site FDR column")
        return _collapse_top_significant(keys, stats, fdrs, key_to_gene)
    raise ValueError(f"unknown strategy: {strategy!r}")


def _collapse_max_abs(
    keys: list[str],
    stats: np.ndarray,
    key_to_gene: dict[str, str],
) -> tuple[dict[str, float], dict[str, int]]:
    """Per gene, keep the site with the largest ``|stat|`` (signed).

    Ties broken by first occurrence.
    """
    best: dict[str, tuple[float, float]] = {}  # gene -> (abs_stat, signed_stat)
    n_used = 0
    n_dropped = 0
    n_unparseable = 0
    for k, s in zip(keys, stats, strict=True):
        gene = key_to_gene.get(str(k))
        if gene is None:
            n_unparseable += 1
            continue
        if pd.isna(s):
            n_dropped += 1
            continue
        n_used += 1
        signed = float(s)
        abs_s = abs(signed)
        prev = best.get(gene)
        if prev is None or abs_s > prev[0]:
            best[gene] = (abs_s, signed)
    gene_to_stat = {gene: signed for gene, (_, signed) in best.items()}
    return gene_to_stat, {
        "n_sites_used": n_used,
        "n_sites_dropped": n_dropped,
        "n_unparseable_keys": n_unparseable,
    }


def _collapse_top_significant(
    keys: list[str],
    stats: np.ndarray,
    fdrs: np.ndarray,
    key_to_gene: dict[str, str],
) -> tuple[dict[str, float], dict[str, int]]:
    """Per gene, keep the site with the lowest per-site FDR (signed stat)."""
    best: dict[str, tuple[float, float]] = {}  # gene -> (fdr, signed_stat)
    n_used = 0
    n_dropped = 0
    n_unparseable = 0
    for k, s, q in zip(keys, stats, fdrs, strict=True):
        gene = key_to_gene.get(str(k))
        if gene is None:
            n_unparseable += 1
            continue
        if pd.isna(s) or pd.isna(q):
            n_dropped += 1
            continue
        n_used += 1
        signed = float(s)
        q_val = float(q)
        prev = best.get(gene)
        if prev is None or q_val < prev[0]:
            best[gene] = (q_val, signed)
    gene_to_stat = {gene: signed for gene, (_, signed) in best.items()}
    return gene_to_stat, {
        "n_sites_used": n_used,
        "n_sites_dropped": n_dropped,
        "n_unparseable_keys": n_unparseable,
    }


def _normalise_prerank_output(raw: pd.DataFrame, *, library: str) -> pd.DataFrame:
    """Reshape gseapy prerank ``.res2d`` into our schema.

    gseapy 1.1 returns object-dtype columns for numerics; cast explicitly
    and derive ``size`` from ``Tag %`` ("k/n") and ``direction`` from NES
    sign.
    """
    if raw.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    df = raw.copy()
    df["library"] = library
    df["term"] = df["Term"].astype(str)
    df["es"] = pd.to_numeric(df["ES"], errors="coerce")
    df["nes"] = pd.to_numeric(df["NES"], errors="coerce")
    df["p_value"] = pd.to_numeric(df["NOM p-val"], errors="coerce")
    df["fdr"] = pd.to_numeric(df["FDR q-val"], errors="coerce")
    df["size"] = df["Tag %"].astype(str).map(_parse_tag_size)
    df["leading_edge"] = df["Lead_genes"].astype(str)
    df["direction"] = np.where(df["nes"] >= 0, "up", "down")
    return df[_OUTPUT_COLUMNS]


def _parse_tag_size(tag: str) -> int:
    """``"8/20"`` -> ``8`` (leading-edge overlap size)."""
    try:
        return int(tag.split("/", 1)[0])
    except (ValueError, AttributeError):
        return 0
