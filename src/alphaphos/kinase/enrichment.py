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

Input conventions
-----------------
* ``diff_results`` is indexed by alphaPhos site keys (``id_col=None``,
  default), or carries them in ``id_col``.
* ``sequence_lookup`` maps site key -> alphaPhos ``kinase_sequence``
  (``adata.var["kinase_sequence"]`` after :func:`alphaphos.add_kinase_windows`);
  its index must be unique.
* Multiplicity variants of one site (``...|M1`` / ``...|M2``) share a window
  and would be counted as two substrates; ``dedup_sequences=True`` (default)
  keeps one row per sequence (strongest statistic).
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from alphaphos.kinase.library import (
    _strip_alphaphos_markers,
    kl_context,
    require_kinase_library,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

KinType = Literal["ser_thr", "tyrosine"]
KLMethod = Literal["score", "percentile"]


def _check_lookup(sequence_lookup: pd.Series) -> pd.Series:
    if not isinstance(sequence_lookup, pd.Series):
        raise TypeError("sequence_lookup must be a pandas Series (site key -> kinase_sequence)")
    if not sequence_lookup.index.is_unique:
        dups = sequence_lookup.index[sequence_lookup.index.duplicated()][:3].tolist()
        raise ValueError(
            f"sequence_lookup index must be unique; duplicated keys e.g. {dups}. "
            "Use adata.var['kinase_sequence'] (unique var_names)."
        )
    return sequence_lookup


def _site_ids(diff_results: pd.DataFrame, id_col: str | None) -> pd.Series:
    if id_col is None:
        return pd.Series(diff_results.index.astype(str), index=diff_results.index)
    if id_col not in diff_results.columns:
        raise KeyError(
            f"id_col={id_col!r} not in diff_results columns {list(diff_results.columns)}; "
            "pass id_col=None to use the index (alphaPhos results are indexed by site key)."
        )
    return diff_results[id_col].astype(str)


def _build_seq_frame(
    site_ids: pd.Index | list[str],
    sequence_lookup: pd.Series,
    *,
    dedup_sequences: bool = True,
) -> pd.DataFrame:
    """Build (id, seq) for kinase-library, dropping sites with invalid sequences."""
    sequence_lookup = _check_lookup(sequence_lookup)
    ids = pd.Series(list(site_ids), dtype=object)
    seqs = ids.map(sequence_lookup).map(_strip_alphaphos_markers)
    valid = seqs.notna() & seqs.str.len().mod(2).eq(1)
    out = pd.DataFrame({"id": ids[valid].values, "seq": seqs[valid].values})
    if dedup_sequences:
        n0 = len(out)
        out = out.drop_duplicates("seq").reset_index(drop=True)
        if len(out) < n0:
            logger.info("dedup_sequences: %d -> %d unique windows", n0, len(out))
    return out


def _merge_in_seq(
    diff_results: pd.DataFrame,
    sequence_lookup: pd.Series,
    id_col: str | None,
    *,
    dedup_sequences: bool = True,
    strength_col: str | None = None,
    strength: Literal["abs_max", "min"] = "abs_max",
) -> pd.DataFrame:
    """Add a 'seq' column to ``diff_results`` from the sequence lookup; drop rows
    whose sequence is missing / invalid (the library would drop them anyway,
    but this keeps the counts honest).  With ``dedup_sequences`` one row per
    window is kept -- the largest ``|strength_col|`` (``abs_max``) or the
    smallest ``strength_col`` (``min``, for p-values)."""
    sequence_lookup = _check_lookup(sequence_lookup)
    out = diff_results.copy()
    out["_site_id"] = _site_ids(diff_results, id_col).to_numpy()
    out["seq"] = out["_site_id"].map(sequence_lookup).map(_strip_alphaphos_markers)
    keep = out["seq"].notna() & out["seq"].str.len().mod(2).eq(1)
    n_dropped = int((~keep).sum())
    if n_dropped:
        logger.info("Dropped %d / %d rows with invalid kinase_sequence", n_dropped, len(out))
    out = out.loc[keep]
    if dedup_sequences and strength_col is not None and out["seq"].duplicated().any():
        n0 = len(out)
        key = out[strength_col].abs() if strength == "abs_max" else -out[strength_col]
        out = (
            out.assign(_k=key)
            .sort_values("_k", ascending=False)
            .drop_duplicates("seq")
            .drop(columns="_k")
        )
        logger.info("dedup_sequences: %d -> %d rows (one per window)", n0, len(out))
    return out.reset_index(drop=True)


def _run_pools(label: str, kin_types: tuple[str, ...], fn) -> dict[str, pd.DataFrame]:
    """Call ``fn(kin_type)`` per pool; warn on failures, raise if every pool failed."""
    out: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    for kt in kin_types:
        try:
            out[kt] = fn(kt)
        except Exception as exc:
            failures[kt] = f"{type(exc).__name__}: {exc}"
            logger.warning("%s(%s) failed: %s", label, kt, failures[kt])
    if failures and not out:
        raise RuntimeError(
            f"{label}: every kinase pool failed -- "
            + "; ".join(f"{k}: {v}" for k, v in failures.items())
        )
    return out


# ---------------------------------------------------------------------------
# 1. KSEA from a diff_exp result (Fisher's exact, per direction)
# ---------------------------------------------------------------------------


def kinase_enrichment_from_diffexp(
    diff_results: pd.DataFrame,
    sequence_lookup: pd.Series,
    *,
    id_col: str | None = None,
    lfc_col: str = "log2fc",
    pval_col: str = "fdr",
    lfc_thresh: float = 0.585,
    pval_thresh: float = 0.05,
    kl_method: KLMethod = "percentile",
    kl_thresh: int = 90,
    kin_types: tuple[KinType, ...] = ("ser_thr", "tyrosine"),
    dedup_sequences: bool = True,
    quiet: bool = True,
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
        Series mapping site key to the alphaPhos-format ``kinase_sequence``
        (with ``*X*`` markers) -- ``adata.var["kinase_sequence"]``.  Sites
        with invalid sequences are dropped (count logged).
    id_col
        Column of ``diff_results`` holding the site keys; ``None`` (default)
        uses the index, which is what alphaPhos's ``diff_exp_*`` return.
    lfc_col, pval_col
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
    dedup_sequences
        Keep one row per sequence window (the most significant by
        ``pval_col``) so multiplicity variants are not double-counted.
    quiet
        Silence kinase_library's stdout output.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``{"ser_thr": df, "tyrosine": df}``. Each DataFrame is
        indexed by kinase, columns include (per direction): ``fg_counts``,
        ``log2_freq_factor``, ``fisher_pval``, ``fisher_adj_pval``, plus
        ``most_sig_direction``, ``most_sig_log2_freq_factor``,
        ``most_sig_fisher_adj_pval``.  A pool that cannot be tested (e.g.
        too few tyrosine sites) is omitted with a warning; if every pool
        fails a ``RuntimeError`` is raised.

    Notes
    -----
    Direct port of ``DiffPhosData.kinase_enrichment``; pandas-friendly
    output extracted from ``combined_enrichment_results``.
    """
    kinase_library = require_kinase_library("kinase_enrichment_from_diffexp")

    df = _merge_in_seq(
        diff_results,
        sequence_lookup,
        id_col,
        dedup_sequences=dedup_sequences,
        strength_col=pval_col,
        strength="min",
    )

    with kl_context(quiet):
        dpd = kinase_library.DiffPhosData(
            dp_data=df,
            lfc_col=lfc_col,
            pval_col=pval_col,
            lfc_thresh=lfc_thresh,
            pval_thresh=pval_thresh,
            seq_col="seq",
            suppress_warnings=True,
        )

    def _one(kt: str) -> pd.DataFrame:
        with kl_context(quiet):
            res = dpd.kinase_enrichment(
                kin_type=kt, kl_method=kl_method, kl_thresh=kl_thresh, enrichment_type="both"
            )
        return res.combined_enrichment_results.copy()

    return _run_pools("kinase_enrichment_from_diffexp", kin_types, _one)


# ---------------------------------------------------------------------------
# 2. MEA — GSEA-style (continuous, no threshold)
# ---------------------------------------------------------------------------


def kinase_mea(
    diff_results: pd.DataFrame,
    sequence_lookup: pd.Series,
    *,
    id_col: str | None = None,
    rank_col: str = "log2fc",
    kl_method: KLMethod = "percentile",
    kl_thresh: int = 90,
    permutation_num: int = 1000,
    seed: int = 42,
    threads: int = 1,
    min_size: int = 5,
    max_size: int = 100000,
    kin_types: tuple[KinType, ...] = ("ser_thr", "tyrosine"),
    gseapy_verbose: bool = False,
    dedup_sequences: bool = True,
    quiet: bool = True,
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
    id_col
        Column holding the site keys; ``None`` (default) uses the index.
    rank_col
        What the sites get ordered by — ``"log2fc"`` (signed effect) or
        ``"t_stat"`` (moderated t; accounts for per-site variability).
    kl_method, kl_thresh
        How each site's "is this a substrate of kinase X" call is made
        (see :func:`kinase_enrichment_from_diffexp`).
    permutation_num
        Number of permutations for the null distribution. 1000 is the
        gseapy default; bump to 10000 for publication-grade p-values.
    seed
        Permutation RNG seed.
    threads
        gseapy parallelism.  Default 1 (deterministic for a given seed, like
        :func:`alphaphos.enrichment.pathway_gsea`); raise for large runs.
    min_size, max_size
        Minimum / maximum number of substrates per kinase to test.
    kin_types
        Which kinase pools to test.
    gseapy_verbose
        Pass-through to gseapy's stderr verbosity.
    dedup_sequences
        Keep one row per sequence window (largest ``|rank_col|``) so
        multiplicity variants are not double-counted.
    quiet
        Silence kinase_library's stdout output.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``{"ser_thr": df, "tyrosine": df}``. Each DataFrame is
        indexed by kinase, columns: ``ES``, ``NES`` (normalised
        enrichment score — the headline statistic), ``p-value``, ``FDR``,
        ``Subs fraction``, ``Leading substrates``.  Pools that fail are
        omitted with a warning; ``RuntimeError`` if all fail.
    """
    kinase_library = require_kinase_library("kinase_mea")

    df = _merge_in_seq(
        diff_results,
        sequence_lookup,
        id_col,
        dedup_sequences=dedup_sequences,
        strength_col=rank_col,
        strength="abs_max",
    )

    with kl_context(quiet):
        rpd = kinase_library.RankedPhosData(
            dp_data=df, rank_col=rank_col, seq_col="seq", suppress_warnings=True
        )

    def _one(kt: str) -> pd.DataFrame:
        with kl_context(quiet):
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
        return res.enrichment_results.copy()

    return _run_pools("kinase_mea", kin_types, _one)


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
    dedup_sequences: bool = True,
    quiet: bool = True,
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
    dedup_sequences
        Count each sequence window once in foreground and background.
    quiet
        Silence kinase_library's stdout output.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``{"ser_thr": df, "tyrosine": df}``. Each DataFrame is
        indexed by kinase, columns include: ``fg_counts``, ``bg_counts``,
        ``log2_freq_factor``, ``fisher_pval``, ``fisher_adj_pval``.  Pools
        that fail are omitted with a warning; ``RuntimeError`` if all fail.
    """
    kinase_library = require_kinase_library("kinase_enrichment_binary")

    fg = _build_seq_frame(foreground_sites, sequence_lookup, dedup_sequences=dedup_sequences)
    bg = _build_seq_frame(background_sites, sequence_lookup, dedup_sequences=dedup_sequences)

    with kl_context(quiet):
        ed = kinase_library.EnrichmentData(
            foreground=fg, background=bg, fg_seq_col="seq", bg_seq_col="seq", suppress_warnings=True
        )

    def _one(kt: str) -> pd.DataFrame:
        with kl_context(quiet):
            res = ed.kinase_enrichment(
                kin_type=kt,
                kl_method=kl_method,
                kl_thresh=kl_thresh,
                enrichment_type=enrichment_type,
            )
        return res.enrichment_results.copy()

    return _run_pools("kinase_enrichment_binary", kin_types, _one)
