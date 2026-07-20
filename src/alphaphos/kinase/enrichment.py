"""Kinase enrichment / activity inference (KSEA) using the Yaffe Kinase Library.

Three statistical frameworks, all wrapping ``kinase_library``:

- :func:`kinase_enrichment_from_diffexp` — **Fisher's exact** per direction
  (up- and down-regulated sites separately). Inputs: a diff_exp results
  DataFrame with logFC + p-value columns. Most direct equivalent of
  classical KSEA. Returns log2 frequency factor + Fisher p-value per
  kinase, per direction (up / down / most-significant).

- :func:`kinase_mea` — **GSEA-style** weighted Kolmogorov-Smirnov via
  ``gseapy``. Operates on the full ranked list (no hard threshold).
  Returns NES + p-value + FDR per kinase. The most rigorous of the three
  (uses every site's rank, not a binarised threshold).

- :func:`kinase_enrichment_binary` — direct **foreground/background**
  Fisher's exact. For when you have a custom site set (e.g. a cluster
  from PCA, a list of upstream kinase targets) rather than a full
  diff_exp table.

All three operate on the same machinery: kinase-library scores each site
against every kinase PWM, then statistical tests pick out kinases whose
substrate sets are non-randomly distributed in the foreground / ranking.

Pipeline position
-----------------
These run **downstream** of differential analysis (e.g.
:func:`alphaphos.diff_exp_limma` or :func:`alphaphos.diff_exp_anova`),
using the diff_exp table plus a per-site sequence lookup as inputs.
Complements per-site PWM prediction in ``alphaphos.kinase.library``.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Literal

import pandas as pd

from alphaphos.kinase.library import _strip_alphaphos_markers

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoding fix
# ---------------------------------------------------------------------------
# kinase_library's tqdm progress bars emit Unicode block characters that
# crash on Windows cp1252 stdout. Reconfigure once at module import.
def _reconfigure_streams_for_utf8() -> None:
    import contextlib

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(OSError, ValueError):
            reconfigure(encoding="utf-8", errors="replace")


_reconfigure_streams_for_utf8()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

KinType = Literal["ser_thr", "tyrosine"]
KLMethod = Literal["score", "percentile"]


def _build_seq_frame(
    site_ids: pd.Index | list[str],
    sequence_lookup: pd.Series,
) -> pd.DataFrame:
    """Build (id, seq) for kinase-library, dropping sites with invalid sequences."""
    seqs = pd.Series(site_ids).map(sequence_lookup).map(_strip_alphaphos_markers)
    valid = seqs.notna() & seqs.str.len().mod(2).eq(1)
    return pd.DataFrame({"id": pd.Series(site_ids)[valid].values, "seq": seqs[valid].values})


def _merge_in_seq(
    diff_results: pd.DataFrame,
    sequence_lookup: pd.Series,
    id_col: str,
) -> pd.DataFrame:
    """Add a 'seq' column to ``diff_results`` from the sequence lookup; drop rows
    whose sequence is missing / invalid (the library would drop them anyway,
    but this keeps the counts honest)."""
    out = diff_results.copy()
    out["seq"] = out[id_col].map(sequence_lookup).map(_strip_alphaphos_markers)
    keep = out["seq"].notna() & out["seq"].str.len().mod(2).eq(1)
    n_dropped = int((~keep).sum())
    if n_dropped:
        logger.info("Dropped %d / %d rows with invalid kinase_sequence", n_dropped, len(out))
    return out.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. KSEA from a diff_exp result (Fisher's exact, per direction)
# ---------------------------------------------------------------------------


def kinase_enrichment_from_diffexp(
    diff_results: pd.DataFrame,
    sequence_lookup: pd.Series,
    *,
    id_col: str = "protein",
    lfc_col: str = "log2fc",
    pval_col: str = "fdr",
    lfc_thresh: float = 0.585,
    pval_thresh: float = 0.05,
    kl_method: KLMethod = "percentile",
    kl_thresh: int = 90,
    kin_types: tuple[KinType, ...] = ("ser_thr", "tyrosine"),
) -> dict[str, pd.DataFrame]:
    """KSEA-style kinase enrichment via Fisher's exact (per direction).

    Splits ``diff_results`` into up- and down-regulated sets using
    (lfc_thresh, pval_thresh), then asks for each kinase whether its
    substrates are over-represented in either set vs the unchanged
    background. Returns a combined per-kinase summary with both directions
    plus a "most significant" direction.

    Parameters
    ----------
    diff_results
        A differential analysis output (e.g. :func:`alphaphos.diff_exp_limma`).
        Must contain the columns named by ``id_col``, ``lfc_col``,
        ``pval_col``.
    sequence_lookup
        Series mapping site id (matching ``diff_results[id_col]``) to the
        alphaPhos-format ``kinase_sequence`` (with ``*X*`` markers). Sites
        with invalid sequences are silently dropped.
    id_col, lfc_col, pval_col
        Column names in ``diff_results``.
    lfc_thresh, pval_thresh
        Thresholds for calling a site "up", "down", or unchanged.
    kl_method
        Per-site kinase scoring metric — ``"percentile"`` (recommended;
        normalises against a reference phosphoproteome) or ``"score"``.
    kl_thresh
        Cutoff above which a site is counted as a candidate substrate of a
        kinase. For ``"percentile"`` this is in 0-100 (default 90 = top
        10%); for ``"score"`` it's the raw PWM log-score.
    kin_types
        Which kinase pools to enrich. By default both — tyrosine kinases
        are tested separately because they share no PWMs with ser/thr.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``{"ser_thr": df, "tyrosine": df}``. Each DataFrame is
        indexed by kinase, columns include (per direction): ``fg_counts``,
        ``log2_freq_factor``, ``fisher_pval``, ``fisher_adj_pval``, plus
        ``most_sig_direction``, ``most_sig_log2_freq_factor``,
        ``most_sig_fisher_adj_pval``.

    Notes
    -----
    Direct port of ``DiffPhosData.kinase_enrichment``; pandas-friendly
    output extracted from ``combined_enrichment_results``.
    """
    from kinase_library import DiffPhosData

    df = _merge_in_seq(diff_results, sequence_lookup, id_col)

    dpd = DiffPhosData(
        dp_data=df,
        lfc_col=lfc_col,
        pval_col=pval_col,
        lfc_thresh=lfc_thresh,
        pval_thresh=pval_thresh,
        seq_col="seq",
        suppress_warnings=True,
    )

    out: dict[str, pd.DataFrame] = {}
    for kt in kin_types:
        try:
            res = dpd.kinase_enrichment(
                kin_type=kt,
                kl_method=kl_method,
                kl_thresh=kl_thresh,
                enrichment_type="both",
            )
        except Exception as exc:
            logger.info("kinase_enrichment(%s) failed: %s", kt, exc)
            continue
        out[kt] = res.combined_enrichment_results.copy()
    return out


# ---------------------------------------------------------------------------
# 2. MEA — GSEA-style (continuous, no threshold)
# ---------------------------------------------------------------------------


def kinase_mea(
    diff_results: pd.DataFrame,
    sequence_lookup: pd.Series,
    *,
    id_col: str = "protein",
    rank_col: str = "log2fc",
    kl_method: KLMethod = "percentile",
    kl_thresh: int = 90,
    permutation_num: int = 1000,
    seed: int = 42,
    threads: int = 4,
    min_size: int = 5,
    max_size: int = 100000,
    kin_types: tuple[KinType, ...] = ("ser_thr", "tyrosine"),
    gseapy_verbose: bool = False,
) -> dict[str, pd.DataFrame]:
    """Motif Enrichment Analysis — GSEA-style weighted Kolmogorov-Smirnov.

    Operates on the **full ranked list** (no hard threshold). For each
    kinase, asks whether its substrates are enriched at the top
    (activated) or bottom (inhibited) of the ranking. Uses ``gseapy``'s
    pre-ranked GSEA implementation under the hood.

    Parameters
    ----------
    diff_results
        Differential analysis output.
    sequence_lookup
        site_id -> alphaPhos kinase_sequence (with ``*X*`` markers).
    id_col, rank_col
        Columns in ``diff_results``. ``rank_col`` is what the sites get
        ordered by — typically ``"log2fc"`` (signed) or ``"stat"``
        (moderated-t).
    kl_method, kl_thresh
        How each site's "is this a substrate of kinase X" call is made
        (see :func:`kinase_enrichment_from_diffexp`).
    permutation_num
        Number of permutations for the null distribution. 1000 is the
        gseapy default; bump to 10000 for publication-grade p-values.
    seed
        Permutation RNG seed.
    threads
        gseapy parallelism.
    min_size, max_size
        Minimum / maximum number of substrates per kinase to test.
    kin_types
        Which kinase pools to test.
    gseapy_verbose
        Pass-through to gseapy's stderr verbosity.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``{"ser_thr": df, "tyrosine": df}``. Each DataFrame is
        indexed by kinase, columns: ``ES``, ``NES`` (normalised
        enrichment score — the headline statistic), ``p-value``, ``FDR``,
        ``Subs fraction``, ``Leading substrates``.
    """
    from kinase_library import RankedPhosData

    df = _merge_in_seq(diff_results, sequence_lookup, id_col)

    rpd = RankedPhosData(
        dp_data=df,
        rank_col=rank_col,
        seq_col="seq",
        suppress_warnings=True,
    )

    out: dict[str, pd.DataFrame] = {}
    for kt in kin_types:
        try:
            res = rpd.mea(
                kin_type=kt,
                kl_method=kl_method,
                kl_thresh=kl_thresh,
                permutation_num=permutation_num,
                seed=seed,
                threads=threads,
                min_size=min_size,
                max_size=max_size,
                gseapy_verbose=gseapy_verbose,
            )
        except Exception as exc:
            logger.info("mea(%s) failed: %s", kt, exc)
            continue
        out[kt] = res.enrichment_results.copy()
    return out


# ---------------------------------------------------------------------------
# 3. Binary foreground/background enrichment
# ---------------------------------------------------------------------------


def kinase_enrichment_binary(
    foreground_sites: pd.Index | list[str],
    background_sites: pd.Index | list[str],
    sequence_lookup: pd.Series,
    *,
    kl_method: KLMethod = "percentile",
    kl_thresh: int = 90,
    enrichment_type: Literal["enriched", "depleted", "both"] = "both",
    kin_types: tuple[KinType, ...] = ("ser_thr", "tyrosine"),
) -> dict[str, pd.DataFrame]:
    """Foreground-vs-background kinase enrichment via Fisher's exact.

    Use this when you have a custom site set (a PCA cluster, a curated
    list, sites belonging to a specific pathway, etc.) rather than a full
    diff_exp table. Asks: for each kinase, is its substrate set
    over-represented (or depleted) among the foreground sites compared to
    the background?

    Parameters
    ----------
    foreground_sites
        Sites of interest (e.g. up-regulated ones).
    background_sites
        Reference universe (typically all detected sites). Often
        ``adata.var.index``.
    sequence_lookup
        site_id -> alphaPhos kinase_sequence.
    kl_method, kl_thresh, enrichment_type
        See :func:`kinase_enrichment_from_diffexp`.
    kin_types
        Which kinase pools to test.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``{"ser_thr": df, "tyrosine": df}``. Each DataFrame is
        indexed by kinase, columns include: ``fg_counts``, ``bg_counts``,
        ``log2_freq_factor``, ``fisher_pval``, ``fisher_adj_pval``.
    """
    from kinase_library import EnrichmentData

    fg = _build_seq_frame(foreground_sites, sequence_lookup)
    bg = _build_seq_frame(background_sites, sequence_lookup)

    ed = EnrichmentData(
        foreground=fg,
        background=bg,
        fg_seq_col="seq",
        bg_seq_col="seq",
        suppress_warnings=True,
    )

    out: dict[str, pd.DataFrame] = {}
    for kt in kin_types:
        try:
            res = ed.kinase_enrichment(
                kin_type=kt,
                kl_method=kl_method,
                kl_thresh=kl_thresh,
                enrichment_type=enrichment_type,
            )
        except Exception as exc:
            logger.info("kinase_enrichment(%s) failed: %s", kt, exc)
            continue
        out[kt] = res.enrichment_results.copy()
    return out
