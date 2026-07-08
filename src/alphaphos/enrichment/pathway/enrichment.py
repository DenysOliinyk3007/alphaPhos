"""Gene-level pathway over-representation analysis via gseapy Enrichr.

Takes a per-site differential-expression result, splits significantly
up- vs down-regulated sites, extracts their gene symbols, and runs
one-sided hypergeometric enrichment against Enrichr gene-set libraries
(GO BP/MF/CC, KEGG, Reactome, MSigDB Hallmark by default).

The **background** is the entire tested phosphoproteome by default
(all site keys in ``diff_exp_result.index``, projected to unique gene
symbols).  A proteome gene list should be passed as ``background=[...]``
when available -- it is the biologically rigorous choice.
Genome-wide background is available as an explicit opt-in with a
UserWarning; using it in a phospho-only study inflates significance.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Literal

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
    "direction",
    "library",
    "term",
    "overlap",
    "p_value",
    "fdr",
    "odds_ratio",
    "combined_score",
    "genes",
    "n_foreground",
    "n_background",
]


def pathway_enrichment(
    diff_exp_result: pd.DataFrame,
    *,
    fdr_threshold: float = 0.05,
    log2fc_threshold: float = 0.0,
    direction: Literal["up", "down", "both", "split", "any"] = "split",
    libraries: list[str] | None = None,
    background: Literal["phosphoproteome", "genome"] | list[str] | pd.DataFrame = "phosphoproteome",
    organism: Literal["human", "mouse"] = "human",
    key_column: str | None = None,
    gene_column: str | None = None,
    cache_dir: str | Path | None = None,
    stat_col: str | None = "log2fc",
    fdr_col: str = "fdr",
) -> pd.DataFrame:
    """Gene-level pathway ORA on significant phosphosites.

    Parameters
    ----------
    diff_exp_result
        DataFrame from :func:`alphaphos.diff_exp_limma` (or
        :func:`diff_exp_anova` for direction-agnostic ORA), indexed by
        alphaPhos ``Protein|Gene|Site|Mult`` keys.  Must carry the
        ``fdr`` column (or the column named by ``fdr_col``).  Signed
        modes (``direction`` in ``{"up", "down", "split", "both"}``)
        additionally require the column named by ``stat_col``.
    fdr_threshold
        FDR cutoff for calling a site significant.  Default 0.05.
    log2fc_threshold
        Minimum absolute ``log2fc`` for a site to count.  Default 0
        (any signed effect).  Ignored when ``direction="any"``.
    direction
        - ``"split"`` (default) -- run up- and down-regulated sets
          separately and stack the results (adds a ``direction`` column).
        - ``"up"`` / ``"down"`` -- one side only.
        - ``"both"`` -- combine up and down into one gene list (loses
          sign information; use only when the question is direction-
          agnostic).
        - ``"any"`` -- direction-agnostic; take all sites below
          ``fdr_threshold`` regardless of sign.  **Use this for ANOVA
          output** (:func:`diff_exp_anova` has no signed statistic).
          ``stat_col`` may be ``None`` in this mode.
    libraries
        Enrichr library slugs.  Defaults to GO BP/MF/CC + KEGG +
        Reactome + MSigDB Hallmark for the chosen organism.  See
        https://maayanlab.cloud/Enrichr/#libraries for the full list.
    background
        - ``"phosphoproteome"`` (default) -- use every gene symbol in
          ``diff_exp_result.index`` (every tested site).  This is the
          right choice when no proteome data is available.
        - A ``list[str]`` or ``pd.DataFrame`` -- caller-supplied gene
          symbols.  Pass proteome-identified genes here when available;
          this is the rigorous choice.
        - ``"genome"`` -- deliberate opt-in to Enrichr's whole-genome
          default.  Emits a ``UserWarning``: using genome-wide background
          for a phospho experiment inflates significance.
    organism
        ``"human"`` or ``"mouse"`` (case-insensitive).  Selects the
        organism-specific default libraries and Enrichr backend.
    key_column
        If site keys live in a column rather than the index, name it
        here.
    gene_column
        Column in ``diff_exp_result`` carrying gene symbols directly.  Use
        this for **proteome input** (where the row index is a protein-group
        key without an embedded gene), e.g.::

            result["gene"] = adata_prot.var.loc[result.index, "PG_Genes"].values
            ap.enrichment.pathway_enrichment(result, gene_column="gene", ...)

        Semicolon-joined multi-gene entries take the first name.  When set,
        overrides the phospho-key parser.  ``None`` (default) uses the
        phospho ``Protein|Gene|Site|Mult`` parser.
    cache_dir
        Directory for gseapy to cache Enrichr library GMTs.  Defaults
        to gseapy's own cache (``~/.gseapy/`` on POSIX,
        ``%LOCALAPPDATA%\\gseapy\\`` on Windows).
    stat_col
        Column carrying the signed effect.  Default ``"log2fc"``.
    fdr_col
        Column carrying the per-site FDR.  Default ``"fdr"``.

    Returns
    -------
    pandas.DataFrame with columns::

        direction, library, term, overlap, p_value, fdr, odds_ratio,
        combined_score, genes, n_foreground, n_background

    Sorted by ``fdr`` ascending within each ``direction`` + ``library``.
    ``.attrs["provenance"]`` records libraries, background_type,
    n_background_genes, thresholds, gseapy version, and n_foreground per
    direction.
    """
    if direction not in ("up", "down", "both", "split", "any"):
        raise ValueError(
            f"direction must be 'up', 'down', 'both', 'split', or 'any'; got {direction!r}"
        )
    organism = organism.lower()  # type: ignore[assignment]
    if organism not in ("human", "mouse"):
        raise ValueError(f"organism must be 'human' or 'mouse'; got {organism!r}")

    try:
        import gseapy as gp
    except ImportError as exc:
        raise ImportError(
            "pathway_enrichment requires gseapy. Install with: "
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
    if fdr_col not in diff_exp_result.columns:
        raise ValueError(
            f"fdr_col={fdr_col!r} not in diff_exp_result columns "
            f"(available: {list(diff_exp_result.columns)})"
        )
    fdrs = diff_exp_result[fdr_col].to_numpy()
    if direction == "any":
        stats = None
    else:
        if stat_col is None or stat_col not in diff_exp_result.columns:
            raise ValueError(
                f"stat_col={stat_col!r} required for direction={direction!r} "
                f"(available columns: {list(diff_exp_result.columns)}). "
                "For ANOVA / F-test output use direction='any', which needs "
                "only fdr_col."
            )
        stats = diff_exp_result[stat_col].to_numpy()

    if gene_column is not None:
        if gene_column not in diff_exp_result.columns:
            raise ValueError(
                f"gene_column={gene_column!r} not found in diff_exp_result "
                f"columns (available: {list(diff_exp_result.columns)})"
            )
        # Direct gene lookup (proteome workflow).  Semicolon-joined
        # multi-gene entries: keep the first name.
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
            "gene_column='<name>' (e.g. gene_column='PG_Genes')."
        )

    bg_genes, background_type = _resolve_background(
        background,
        diff_exp_keys=keys,
        key_to_gene=key_to_gene,
    )

    outdir = str(cache_dir) if cache_dir is not None else None

    to_run: list[tuple[str, list[str]]] = []
    if direction == "any":
        any_genes = _select_any_hits(
            keys=keys,
            fdrs=fdrs,
            key_to_gene=key_to_gene,
            fdr_threshold=fdr_threshold,
        )
        to_run.append(("any", any_genes))
    else:
        up_genes, down_genes = _select_hits(
            keys=keys,
            stats=stats,
            fdrs=fdrs,
            key_to_gene=key_to_gene,
            fdr_threshold=fdr_threshold,
            log2fc_threshold=log2fc_threshold,
        )
        if direction in ("up", "split"):
            to_run.append(("up", up_genes))
        if direction in ("down", "split"):
            to_run.append(("down", down_genes))
        if direction == "both":
            to_run.append(("both", sorted(set(up_genes) | set(down_genes))))

    frames: list[pd.DataFrame] = []
    n_foreground_per_direction: dict[str, int] = {}
    for tag, genes in to_run:
        n_foreground_per_direction[tag] = len(genes)
        if not genes:
            logger.info(
                "pathway_enrichment: no %s-regulated hits at fdr<%g, skipping", tag, fdr_threshold
            )
            continue
        logger.info(
            "pathway_enrichment: %s | %d foreground genes, %d background genes, %d libraries",
            tag,
            len(genes),
            len(bg_genes) if bg_genes is not None else -1,
            len(libraries),
        )
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=libraries,
            background=bg_genes,
            organism=organism.capitalize(),
            outdir=outdir,
            verbose=False,
            no_plot=True,
        )
        frame = _normalise_enrichr_output(
            enr.results,
            direction=tag,
            n_foreground=len(genes),
            n_background=len(bg_genes) if bg_genes is not None else 0,
        )
        frames.append(frame)

    if not frames:
        out = pd.DataFrame(columns=_OUTPUT_COLUMNS)
    else:
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["direction", "library", "fdr"], kind="stable").reset_index(drop=True)

    out.attrs["provenance"] = {
        "libraries": libraries,
        "background_type": background_type,
        "n_background_genes": len(bg_genes) if bg_genes is not None else None,
        "fdr_threshold": fdr_threshold,
        "log2fc_threshold": log2fc_threshold,
        "direction": direction,
        "organism": organism,
        "n_foreground_per_direction": n_foreground_per_direction,
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

    Both key formats share the same ``Protein|Gene|...`` prefix -- site keys
    are ``Protein|Gene|Site|Mult`` (4 fields), precursor keys are
    ``Protein|Gene|Peptide|Charge|Mods`` (5 fields).  The gene is always
    the second pipe-delimited field, so a permissive split works for both.
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


def _select_hits(
    *,
    keys: list[str],
    stats,
    fdrs,
    key_to_gene: dict[str, str],
    fdr_threshold: float,
    log2fc_threshold: float,
) -> tuple[list[str], list[str]]:
    """Split significant sites into up/down gene lists (deduplicated)."""
    up: set[str] = set()
    down: set[str] = set()
    for k, s, q in zip(keys, stats, fdrs, strict=True):
        gene = key_to_gene.get(str(k))
        if gene is None:
            continue
        if pd.isna(s) or pd.isna(q):
            continue
        if q >= fdr_threshold:
            continue
        if s > log2fc_threshold:
            up.add(gene)
        elif s < -log2fc_threshold:
            down.add(gene)
    return sorted(up), sorted(down)


def _select_any_hits(
    *,
    keys: list[str],
    fdrs,
    key_to_gene: dict[str, str],
    fdr_threshold: float,
) -> list[str]:
    """Direction-agnostic significant-gene list -- for ANOVA / F-test input."""
    hits: set[str] = set()
    for k, q in zip(keys, fdrs, strict=True):
        gene = key_to_gene.get(str(k))
        if gene is None:
            continue
        if pd.isna(q):
            continue
        if q < fdr_threshold:
            hits.add(gene)
    return sorted(hits)


def _resolve_background(
    background,
    *,
    diff_exp_keys: list[str],
    key_to_gene: dict[str, str],
) -> tuple[list[str] | None, str]:
    """Return (background_gene_list, tag).

    ``tag`` is one of ``"phosphoproteome"``, ``"proteome"``, ``"genome"``,
    or ``"custom"`` -- used for provenance and diagnostic logging.
    """
    if isinstance(background, str):
        low = background.lower()
        if low == "phosphoproteome":
            genes = sorted({g for g in (key_to_gene.get(str(k)) for k in diff_exp_keys) if g})
            return genes, "phosphoproteome"
        if low == "genome":
            warnings.warn(
                "background='genome' uses Enrichr's whole-genome default, "
                "which overstates enrichment for phospho-only experiments "
                "(the detection universe is much smaller than the genome). "
                "Pass a proteome-identified gene list as background=[...] "
                "when available; otherwise use the phosphoproteome default.",
                UserWarning,
                stacklevel=3,
            )
            return None, "genome"
        raise ValueError(
            f"background str must be 'phosphoproteome' or 'genome'; got {background!r}"
        )
    if isinstance(background, pd.DataFrame):
        if "gene" in background.columns:
            genes = background["gene"].dropna().astype(str).unique().tolist()
        elif len(background.columns) == 1:
            genes = background.iloc[:, 0].dropna().astype(str).unique().tolist()
        else:
            raise ValueError("background DataFrame must have a 'gene' column or be a single column")
        return sorted(genes), "custom"
    if isinstance(background, (list, tuple, set, pd.Index)):
        genes = sorted({str(g) for g in background if pd.notna(g) and str(g).strip()})
        if not genes:
            raise ValueError("background list is empty after cleaning")
        return genes, "custom"
    raise TypeError(
        f"background must be 'phosphoproteome'/'genome', a list, or a DataFrame; "
        f"got {type(background).__name__}"
    )


def _normalise_enrichr_output(
    raw: pd.DataFrame,
    *,
    direction: str,
    n_foreground: int,
    n_background: int,
) -> pd.DataFrame:
    """Rename gseapy's Enrichr columns to our schema."""
    rename_map = {
        "Gene_set": "library",
        "Term": "term",
        "Overlap": "overlap",
        "P-value": "p_value",
        "Adjusted P-value": "fdr",
        "Odds Ratio": "odds_ratio",
        "Combined Score": "combined_score",
        "Genes": "genes",
    }
    df = raw.rename(columns=rename_map).copy()
    df["direction"] = direction
    df["n_foreground"] = n_foreground
    df["n_background"] = n_background
    keep = [c for c in _OUTPUT_COLUMNS if c in df.columns]
    return df[keep]
