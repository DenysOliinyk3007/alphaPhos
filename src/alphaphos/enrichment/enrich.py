"""Site-set enrichment scoring — ORA + preranked GSEA — with FDR control.

Two conventional, field-standard methods:

**ORA (over-representation analysis).**  Fisher's exact test on a 2x2
contingency table of hits vs non-hits × in-set vs out-of-set, restricted
to the caller-supplied background of measured sites.  Two-sided by
default (tests both enrichment AND depletion).  BH-adjusted FDR across
all sets tested in a given library type (tiered by default -- one BH
per library) or optionally joint across all libraries.

Reference: Fisher 1922; Benjamini & Hochberg 1995 (J R Stat Soc B
57:289-300); pathway-enrichment best-practice review by Reimand et al.
2019 (Nat Protoc 14:482-517).

**GSEA (preranked, fgsea-style).**  For a ranked list of sites (ranked
by e.g. limma t-statistic or log2FC), computes the Subramanian 2005
running-sum enrichment score per set, then permutes set membership
(gene-set permutation, appropriate for preranked input) to build a
null.  Normalized enrichment score (NES) accounts for set-size
differences.  BH-adjusted FDR is applied to the empirical permutation
p-values, tiered by library.

References: Subramanian et al. 2005 (PNAS 102:15545-15550);
Korotkevich et al. 2019 (bioRxiv 060012, "Fast gene set enrichment
analysis").

**Silent-wrong-result guards baked in:**

1. Background restriction — set members not measured in the background
   are removed BEFORE testing (otherwise ORA over-estimates enrichment
   because it treats database sites as "assayed").
2. Sets are filtered on their *restricted size* (``min_set_size``), which
   does not depend on the hits, so the BH denominator is data-independent
   and two-sided tests really do test depletion (a set with 0 hits is the
   most depleted one).  An ``min_overlap`` filter is available as an
   opt-in for one-sided enrichment questions.
3. Every result carries both nominal p and BH-adjusted q; users see
   both.
4. :func:`gsea` refuses duplicated site ids (multiplicity variants that
   canonicalise to one site) instead of silently mis-positioning them.
5. :func:`library_redundancy` reports pairwise Jaccard overlap between
   sets so users know when apparent "independent" hits are one signal.
"""

from __future__ import annotations

import logging
import warnings
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

from alphaphos.enrichment.libraries import load_libraries

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


DEFAULT_MIN_OVERLAP = 0  # opt-in only; see ora()
DEFAULT_MIN_SET_SIZE = 5  # matches PTM-SEA / library-emit convention
# No upper cap by default: fgsea's 500 is a gene-level heuristic; the shipped
# site-set libraries have 750-49k members and would be skipped wholesale.
DEFAULT_MAX_SET_SIZE: int | None = None
DEFAULT_N_PERMUTATIONS = 10_000
DEFAULT_SEED = 42

# BH is the field-standard default; BY is the arbitrary-dependence
# alternative (more conservative under set-set overlap).
_FDR_METHODS = {"bh": "bh", "by": "by"}


# ---------------------------------------------------------------------------
# ORA
# ---------------------------------------------------------------------------


def ora(
    hits: Iterable[str],
    background: Iterable[str],
    libraries: dict[str, dict[str, list[str]]] | str | Path,
    *,
    min_set_size: int = DEFAULT_MIN_SET_SIZE,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
    alternative: str = "two-sided",
    fdr_method: str = "bh",
    fdr_per_library: bool = True,
    library_names: list[str] | None = None,
) -> pd.DataFrame:
    """Over-representation analysis via Fisher's exact + BH-FDR.

    Parameters
    ----------
    hits
        Site IDs (``Protein_AApos``) called significant by upstream
        analysis (e.g. FDR<0.05 in ``diff_exp_limma``).
    background
        Site IDs measured by the assay (the "universe").  This MUST be
        the assayed sites, not the whole DB -- see docstring for why.
    libraries
        Either a pre-loaded nested dict
        ``{library: {set: [sites]}}`` (from
        :func:`alphaphos.enrichment.libraries.load_libraries`) or a
        directory of ``.gmt`` files.
    min_set_size
        Minimum number of set members *present in the background* for the
        set to be tested.  Default 5.  Independent of the hits, so it does
        not bias the FDR.
    min_overlap
        Optional minimum hit/set overlap (default 0 = no filter).  Only
        sensible for ``alternative="greater"``; with a two-sided test it
        removes exactly the most *depleted* sets and makes the number of
        tests depend on the hits, so a ``UserWarning`` is emitted.
    alternative
        ``"two-sided"`` (default): tests both enrichment AND depletion.
        ``"greater"``: enrichment only (one-sided; more powerful when
        depletion is not of interest).
    fdr_method
        ``"bh"`` (Benjamini-Hochberg, default) or ``"by"``
        (Benjamini-Yekutieli, arbitrary-dependence variant).
    fdr_per_library
        If True (default), BH is applied *within* each library type
        (tiered).  If False, BH is applied jointly across all sets in
        all libraries (more conservative).
    library_names
        Optional subset of library names to test.  ``None`` tests all.

    Returns
    -------
    pandas.DataFrame
        One row per tested set, columns:

        - ``library`` -- source library name
        - ``set_name``, ``description`` -- from the GMT
        - ``n_set`` -- total sites in the set (after background
          restriction)
        - ``n_hits`` -- total hit list size
        - ``n_background`` -- total background size
        - ``n_overlap`` -- observed count in the intersection
        - ``expected`` -- expected overlap under H0
          (``n_hits * n_set / n_background``)
        - ``odds_ratio`` -- Fisher's odds ratio
        - ``log2_fold_enrichment`` -- ``log2(observed / expected)``
        - ``p_value`` -- nominal Fisher p (using ``alternative``)
        - ``fdr`` -- BH- (or BY-) adjusted q
        - ``direction`` -- ``"enriched"`` if observed > expected
          otherwise ``"depleted"``
        - ``overlap_sites`` -- semicolon-joined intersection (all sites)

    Notes
    -----
    Set members not present in ``background`` are removed before the
    Fisher call -- otherwise a set of 5000 with only 100 assayed
    members would be tested as if 5000 were assayable, wildly inflating
    the p-value.  This restriction is a critical correctness step and
    is enforced silently (the ``n_set`` column reports the RESTRICTED
    size).
    """
    if fdr_method not in _FDR_METHODS:
        raise ValueError(f"fdr_method must be one of {sorted(_FDR_METHODS)}, got {fdr_method!r}")
    if alternative not in ("two-sided", "greater"):
        raise ValueError(f"alternative must be 'two-sided' or 'greater', got {alternative!r}")

    libraries_dict = _resolve_libraries(libraries, library_names=library_names)
    if min_overlap > 0 and alternative == "two-sided":
        warnings.warn(
            f"ora(min_overlap={min_overlap}, alternative='two-sided'): the overlap filter "
            "removes the most depleted sets before testing, so depletion is only partially "
            "tested and the BH denominator depends on the hits. Use min_overlap=0 (default) "
            "or alternative='greater'.",
            UserWarning,
            stacklevel=2,
        )
    hits_set = frozenset(hits)
    bg_set = frozenset(background)
    if not hits_set.issubset(bg_set):
        # hits must be a subset of background -- otherwise we're testing
        # enrichment against a universe that doesn't contain the hits.
        stray = hits_set - bg_set
        raise ValueError(
            f"{len(stray)} hit sites are not in the background. "
            "The background should be the union of all assayed sites; "
            f"first stray: {sorted(stray)[:3]}"
        )

    n_hits = len(hits_set)
    n_background = len(bg_set)
    if n_hits == 0 or n_background == 0:
        raise ValueError("both hits and background must be non-empty")

    rows: list[dict] = []
    n_skipped_size = 0
    n_skipped_overlap = 0
    for library_name, sets in libraries_dict.items():
        for set_name, members in sets.items():
            restricted = frozenset(members) & bg_set
            n_set = len(restricted)
            if n_set < max(min_set_size, 1):
                n_skipped_size += 1
                continue
            overlap = restricted & hits_set
            n_overlap = len(overlap)
            if n_overlap < min_overlap:
                n_skipped_overlap += 1
                continue

            a = n_overlap
            b = n_hits - n_overlap
            c = n_set - n_overlap
            d = n_background - n_hits - n_set + n_overlap
            if min(a, b, c, d) < 0:
                # Should not happen if inputs are consistent, but guard
                # against arithmetic error rather than let scipy raise.
                logger.warning(
                    "ORA: skipping %s/%s -- inconsistent 2x2 table (a=%d b=%d c=%d d=%d)",
                    library_name,
                    set_name,
                    a,
                    b,
                    c,
                    d,
                )
                continue
            odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative=alternative)
            expected = n_hits * n_set / n_background
            log2fe = float(np.log2(a / expected)) if expected > 0 and a > 0 else float("nan")
            direction = "enriched" if a > expected else "depleted"
            rows.append(
                {
                    "library": library_name,
                    "set_name": set_name,
                    "n_set": n_set,
                    "n_hits": n_hits,
                    "n_background": n_background,
                    "n_overlap": n_overlap,
                    "expected": expected,
                    "odds_ratio": float(odds_ratio),
                    "log2_fold_enrichment": log2fe,
                    "p_value": float(p_value),
                    "direction": direction,
                    "overlap_sites": ";".join(sorted(overlap)),
                }
            )

    result = pd.DataFrame(rows)
    if n_skipped_size or n_skipped_overlap:
        logger.info(
            "ora: %d set(s) skipped (restricted size < %d), %d skipped (overlap < %d); %d tested",
            n_skipped_size,
            min_set_size,
            n_skipped_overlap,
            min_overlap,
            len(result),
        )
    if result.empty:
        result["fdr"] = pd.Series(dtype=float)
    else:
        result = _apply_fdr(result, method=fdr_method, per_library=fdr_per_library)
        result = result.sort_values(["fdr", "p_value"]).reset_index(drop=True)
    result.attrs["n_sets_skipped_size"] = n_skipped_size
    result.attrs["n_sets_skipped_overlap"] = n_skipped_overlap
    return result


def library_redundancy(
    libraries: dict[str, dict[str, list[str]]] | str | Path,
    *,
    background: Iterable[str] | None = None,
    min_jaccard: float = 0.0,
) -> pd.DataFrame:
    """Pairwise Jaccard overlap between sets of the same library.

    Enrichment results for two sets that share most members are one
    signal, not two.  Call this on the libraries you test (optionally
    restricted to the assayed ``background``) and read the high-Jaccard
    pairs alongside :func:`ora` / :func:`gsea` output.

    Returns a DataFrame with ``library``, ``set_a``, ``set_b``, ``n_a``,
    ``n_b``, ``n_shared``, ``jaccard`` sorted by ``jaccard`` descending;
    pairs below ``min_jaccard`` are omitted.
    """
    libraries_dict = _resolve_libraries(libraries, library_names=None)
    bg = frozenset(background) if background is not None else None
    rows: list[dict] = []
    for library_name, sets in libraries_dict.items():
        members = {
            name: (frozenset(m) & bg if bg is not None else frozenset(m))
            for name, m in sets.items()
        }
        for a, b in combinations(members, 2):
            union = members[a] | members[b]
            if not union:
                continue
            shared = len(members[a] & members[b])
            jac = shared / len(union)
            if jac < min_jaccard:
                continue
            rows.append(
                {
                    "library": library_name,
                    "set_a": a,
                    "set_b": b,
                    "n_a": len(members[a]),
                    "n_b": len(members[b]),
                    "n_shared": shared,
                    "jaccard": jac,
                }
            )
    cols = ["library", "set_a", "set_b", "n_a", "n_b", "n_shared", "jaccard"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("jaccard", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preranked GSEA
# ---------------------------------------------------------------------------


def gsea(
    ranked_stats: pd.Series | dict[str, float],
    libraries: dict[str, dict[str, list[str]]] | str | Path,
    *,
    min_set_size: int = DEFAULT_MIN_SET_SIZE,
    max_set_size: int | None = DEFAULT_MAX_SET_SIZE,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    seed: int = DEFAULT_SEED,
    weight: float = 1.0,
    fdr_method: str = "bh",
    fdr_per_library: bool = True,
    library_names: list[str] | None = None,
) -> pd.DataFrame:
    """Preranked GSEA (Subramanian 2005, fgsea-style) with permutation-based
    FDR.

    Parameters
    ----------
    ranked_stats
        Site-level ranking metric (e.g. limma t-statistic or log2FC).
        Series indexed by site_id (``Protein_AApos``).  Sign matters:
        positive = up in condition of interest.  Ties are broken by
        original order.  **Ids must be unique**: multiplicity variants of
        one site (``...|M1`` / ``...|M2``) canonicalise to the same id and
        must be aggregated first (e.g. keep the largest ``|stat|`` per
        site, as :func:`alphaphos.enrichment.kinase_activity` does);
        duplicates raise.
    libraries
        Nested dict or GMT directory (see :func:`ora`).
    min_set_size, max_set_size
        Sets with fewer members (after background restriction to the
        ranking's universe) than ``min_set_size`` or, when
        ``max_set_size`` is given, more than that are skipped.  Default
        ``None`` (no cap): fgsea's 500 is a gene-level heuristic and would
        skip most shipped site-set libraries.  Skipped sets are logged and
        listed in ``result.attrs["skipped_sets"]``.
    n_permutations
        Number of set-membership permutations for the null.  10k is the
        conventional target for FDR<0.001 resolution.
    seed
        RNG seed for reproducibility of the permutation sample.
    weight
        Weight exponent ``p`` in the Subramanian running-sum increment
        (``|r|^p``).  ``p=1`` (default) is classical GSEA;  ``p=0`` is
        the unweighted Kolmogorov-Smirnov variant.
    fdr_method, fdr_per_library
        As in :func:`ora`.
    library_names
        Optional subset of libraries to score.

    Returns
    -------
    pandas.DataFrame
        One row per tested set, columns:

        - ``library``, ``set_name``, ``n_set``
        - ``ES`` (raw enrichment score, signed)
        - ``NES`` (normalized enrichment score)
        - ``p_value`` (empirical from permutations)
        - ``fdr`` (BH-adjusted)
        - ``direction`` (``"up"`` if ES>=0 else ``"down"``)
        - ``leading_edge`` (semicolon-joined leading-edge sites)
    """
    if fdr_method not in _FDR_METHODS:
        raise ValueError(f"fdr_method must be one of {sorted(_FDR_METHODS)}, got {fdr_method!r}")

    libraries_dict = _resolve_libraries(libraries, library_names=library_names)

    if isinstance(ranked_stats, dict):
        ranked_stats = pd.Series(ranked_stats)
    if ranked_stats.empty:
        raise ValueError("ranked_stats is empty")
    if ranked_stats.isna().any():
        raise ValueError(
            f"{int(ranked_stats.isna().sum())} NaN values in ranked_stats; "
            "drop them or impute before calling gsea()"
        )
    dup = ranked_stats.index.astype(str).duplicated()
    if dup.any():
        examples = sorted(set(ranked_stats.index.astype(str)[dup]))[:3]
        raise ValueError(
            f"{int(dup.sum())} duplicated site ids in ranked_stats (e.g. {examples}). "
            "Multiplicity variants collapse to the same Protein_AApos id; aggregate "
            "first, e.g. keep the largest |stat| per site: "
            "s.groupby(level=0).agg(lambda v: v.iloc[int(v.abs().to_numpy().argmax())])."
        )
    # Sort descending by score.  Ties broken by original position (stable sort).
    ranked = ranked_stats.sort_values(ascending=False, kind="mergesort")
    universe = list(ranked.index.astype(str))
    n_universe = len(universe)
    universe_pos = {sid: i for i, sid in enumerate(universe)}
    ranks_signed = ranked.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    skipped: dict[str, str] = {}
    for library_name, sets in libraries_dict.items():
        for set_name, members in sets.items():
            in_universe_idx = np.array(
                sorted(universe_pos[m] for m in members if m in universe_pos),
                dtype=np.int64,
            )
            n_set = int(in_universe_idx.size)
            if n_set < min_set_size:
                skipped[f"{library_name}/{set_name}"] = (
                    f"n_set={n_set} < min_set_size={min_set_size}"
                )
                continue
            if max_set_size is not None and n_set > max_set_size:
                skipped[f"{library_name}/{set_name}"] = (
                    f"n_set={n_set} > max_set_size={max_set_size}"
                )
                continue

            es, hit_positions = _compute_es(ranks_signed, in_universe_idx, weight=weight)
            # Permutation null: draw n_set random positions from the universe.
            perm_es = _permute_es(ranks_signed, n_set, n_universe, n_permutations, weight, rng)
            same_sign = perm_es[np.sign(perm_es) == np.sign(es)]
            if len(same_sign) == 0:
                # Extremely rare — every permutation flipped sign; fall back.
                p_value = 1.0 / (n_permutations + 1)
                nes = float("nan")
            else:
                # Empirical p using same-sign permutations (Subramanian's
                # convention). Add-one smoothing prevents zero p-values.
                n_more_extreme = int(np.sum(np.abs(same_sign) >= abs(es)))
                p_value = (n_more_extreme + 1) / (len(same_sign) + 1)
                nes = float(es / np.mean(np.abs(same_sign)))

            leading_edge = _leading_edge(ranks_signed, hit_positions, es, weight=weight)
            leading_ids = [universe[i] for i in leading_edge]

            rows.append(
                {
                    "library": library_name,
                    "set_name": set_name,
                    "n_set": n_set,
                    "ES": float(es),
                    "NES": nes,
                    "p_value": float(p_value),
                    "direction": "up" if es >= 0 else "down",
                    "leading_edge": ";".join(leading_ids),
                }
            )

    result = pd.DataFrame(rows)
    if skipped:
        logger.info(
            "gsea: %d set(s) skipped by size filters (see result.attrs['skipped_sets']): %s",
            len(skipped),
            "; ".join(f"{k} ({v})" for k, v in list(skipped.items())[:5]),
        )
    if result.empty:
        result["fdr"] = pd.Series(dtype=float)
    else:
        result = _apply_fdr(result, method=fdr_method, per_library=fdr_per_library)
        result = result.sort_values(["fdr", "p_value"]).reset_index(drop=True)
    result.attrs["skipped_sets"] = skipped
    return result


# ---------------------------------------------------------------------------
# GSEA internals
# ---------------------------------------------------------------------------


def _compute_es(
    ranks_signed: np.ndarray,
    in_universe_idx: np.ndarray,
    *,
    weight: float,
) -> tuple[float, np.ndarray]:
    """Compute Subramanian's running-sum enrichment score.

    Given a signed rank vector and the positions (indices into that
    vector) of the current set's members, walk the ranked list summing
    the appropriate increment at each position. Returns (ES, hit_positions).
    """
    n = len(ranks_signed)
    n_hits = len(in_universe_idx)
    n_miss = n - n_hits
    if n_hits == 0 or n_miss == 0:
        return 0.0, in_universe_idx

    # Precompute per-position increments.
    hit_mask = np.zeros(n, dtype=bool)
    hit_mask[in_universe_idx] = True
    r_pow = np.abs(ranks_signed) ** weight
    hit_weight_sum = r_pow[hit_mask].sum()
    if hit_weight_sum == 0:
        return 0.0, in_universe_idx
    increments = np.where(hit_mask, r_pow / hit_weight_sum, -1.0 / n_miss)
    running = np.cumsum(increments)
    # ES is the signed value of the maximum absolute deviation.
    imax = int(np.argmax(np.abs(running)))
    es = float(running[imax])
    return es, in_universe_idx


def _permute_es(
    ranks_signed: np.ndarray,
    n_set: int,
    n_universe: int,
    n_permutations: int,
    weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute ES for ``n_permutations`` random sets of size ``n_set``.

    Set-membership permutation (uniform sampling without replacement)
    is the appropriate null for preranked GSEA -- phenotype
    permutation would require the raw expression matrix, which
    preranked GSEA doesn't have.
    """
    out = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm_idx = np.sort(rng.choice(n_universe, size=n_set, replace=False))
        out[i], _ = _compute_es(ranks_signed, perm_idx, weight=weight)
    return out


def _leading_edge(
    ranks_signed: np.ndarray,
    in_universe_idx: np.ndarray,
    es: float,
    *,
    weight: float = 1.0,
) -> list[int]:
    """Return the leading-edge indices (subset of the set's members
    driving the enrichment).

    Following Subramanian 2005 §2.5: for positive ES, these are the
    set members appearing BEFORE the peak of the running-sum in the
    ranked list; for negative ES, those appearing AFTER the trough.
    Uses the same ``weight`` exponent as the ES computation.
    """
    n = len(ranks_signed)
    n_hits = len(in_universe_idx)
    n_miss = n - n_hits
    if n_hits == 0 or n_miss == 0:
        return []
    hit_mask = np.zeros(n, dtype=bool)
    hit_mask[in_universe_idx] = True
    r_pow = np.abs(ranks_signed) ** weight
    hit_weight_sum = r_pow[hit_mask].sum()
    increments = np.where(hit_mask, r_pow / hit_weight_sum, -1.0 / n_miss)
    running = np.cumsum(increments)
    imax = int(np.argmax(np.abs(running)))
    if es >= 0:
        # Members with rank position <= peak.
        edge = [int(i) for i in in_universe_idx if i <= imax]
    else:
        # Members with rank position >= trough.
        edge = [int(i) for i in in_universe_idx if i >= imax]
    return edge


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_libraries(
    libraries: dict[str, dict[str, list[str]]] | str | Path,
    *,
    library_names: list[str] | None,
) -> dict[str, dict[str, list[str]]]:
    if isinstance(libraries, dict):
        loaded = libraries
    else:
        loaded = load_libraries(libraries)
    if library_names is not None:
        missing = [n for n in library_names if n not in loaded]
        if missing:
            raise KeyError(f"library_names {missing} not found among {sorted(loaded)}")
        loaded = {n: loaded[n] for n in library_names}
    return loaded


def _apply_fdr(
    result: pd.DataFrame,
    *,
    method: str,
    per_library: bool,
) -> pd.DataFrame:
    """Attach a ``fdr`` column with adjusted q-values.

    Uses ``scipy.stats.false_discovery_control`` (scipy >=1.11).
    """
    result = result.copy()
    if per_library:
        result["fdr"] = float("nan")
        for _lib_name, group in result.groupby("library", sort=False):
            q = stats.false_discovery_control(group["p_value"].to_numpy(), method=method)
            result.loc[group.index, "fdr"] = q
    else:
        result["fdr"] = stats.false_discovery_control(result["p_value"].to_numpy(), method=method)
    return result
