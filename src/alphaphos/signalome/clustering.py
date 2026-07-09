"""Hierarchical clustering of sites + module count selection.

Two-stage design (mirrors PhosPy's signalome clustering, MIT clean-room):

1. **Ward tree**: ``scipy.cluster.hierarchy.linkage`` with Ward method on
   Euclidean distance of site kinase-prediction profiles.  Deterministic
   given the same input matrix.  O(N^2) memory + O(N^2 log N) time — the
   scaling limit for the whole subpackage.
2. **Module-count selection**: score each candidate ``k in {2..max_modules}``
   by the median within-cluster Pearson correlation of the raw site profiles;
   pick the smallest ``k`` whose median cluster-correlation is
   >= ``primary_threshold``.  If none reaches primary, fall back to
   ``fallback_threshold``.  If neither is reached, use ``max_modules``.

Scale-aware correlation backend (``scoring_mode``):

- ``"exact"`` -- full N x N pairwise Pearson correlation matrix.  O(N^2)
  memory; recommended for N <= ``max_exact_sites`` (default 5000).
- ``"sampled"`` -- for each candidate cluster, downsample to
  ``max_samples_per_cluster`` sites (default 200) and compute the median
  pairwise correlation on the sample.  O(N * S) memory where S =
  n_clusters * max_samples_per_cluster.  Deterministic given ``seed``.
- ``"auto"`` -- pick ``"exact"`` for N <= max_exact_sites, else
  ``"sampled"``.  Emits a warning when auto-approximation kicks in.

Score preconditioning: NaN cells filled with the column median (or 0.0 if
the column is entirely NaN).  This is the same behaviour PhosPy uses; it
keeps the Ward linkage well-defined on datasets with missing kinase-score
cells (e.g. sites the kinase library rejected).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import pdist

from alphaphos.signalome.constants import (
    DEFAULT_FALLBACK_THRESHOLD,
    DEFAULT_MAX_APPROX_SAMPLES_PER_CLUSTER,
    DEFAULT_MAX_EXACT_SITES,
    DEFAULT_MAX_MODULES,
    DEFAULT_PRIMARY_THRESHOLD,
    NEAR_CONSTANT_VARIANCE_TOLERANCE,
    SCORING_MODE_AUTO,
    SCORING_MODE_EXACT,
    SCORING_MODE_SAMPLED,
    SCORING_MODES,
)

logger = logging.getLogger(__name__)


_LINKAGE_METHOD = "ward"
_DISTANCE_METRIC = "euclidean"


@dataclass(frozen=True, slots=True)
class ClusterCandidateScore:
    """Score for one candidate module count during selection.

    ``min_median_correlation`` is the smallest median-cluster-correlation
    across all non-singleton clusters -- passing the primary/fallback
    threshold requires this minimum to clear it.  Ensures every cluster
    is coherent, not just the median cluster.

    ``mean_median_correlation`` is the mean across cluster medians;
    used as the tie-breaker (higher wins).

    ``median_of_cluster_medians`` is retained for backward-compat and
    diagnostics.
    """

    module_count: int
    cluster_median_correlations: tuple[float, ...]
    min_median_correlation: float
    mean_median_correlation: float
    median_of_cluster_medians: float
    n_singleton_clusters: int


@dataclass(frozen=True, slots=True)
class ModuleCountSelectionResult:
    """Outcome of :func:`select_module_count`."""

    module_count: int
    """The chosen ``k``."""

    selection_reason: str
    """One of ``"primary"``, ``"fallback"``, ``"max_default"``, ``"requested"``."""

    threshold_used: float | None
    candidate_scores: dict[int, ClusterCandidateScore]
    scoring_mode_used: Literal["exact", "sampled"]
    n_sites: int
    max_exact_sites: int


@dataclass(frozen=True, slots=True)
class SignalomeClusteringResult:
    """Combined output of :func:`cluster_sites`."""

    labels: np.ndarray
    """1-indexed integer array, shape (n_sites,).  0 reserved for un-clustered sites."""

    module_count: int
    selection: ModuleCountSelectionResult
    linkage_matrix: np.ndarray
    site_index: pd.Index
    provenance: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def precondition_scores(scoring_values: np.ndarray) -> np.ndarray:
    """Fill NaN with per-column median, empty columns with 0.0.

    Mirrors PhosPy's ``prepare_scoring_values_for_clustering``.  Keeps the
    scipy Ward linkage well-defined on datasets with missing kinase scores.
    """
    values = np.asarray(scoring_values, dtype=float)
    if values.ndim != 2:
        raise ValueError("scoring_values must be 2D (sites, kinases)")
    prepared = values.copy()
    for j in range(prepared.shape[1]):
        col = prepared[:, j]
        finite = np.isfinite(col)
        if finite.all():
            continue
        if not finite.any():
            prepared[:, j] = 0.0
            continue
        median = float(np.median(col[finite]))
        prepared[~finite, j] = median
    return prepared


def build_ward_tree(scoring_values: np.ndarray) -> np.ndarray:
    """Return the scipy Ward linkage matrix for ``scoring_values``.

    ``scoring_values`` must be preconditioned (no NaN).  For n_sites <= 1
    returns an empty (0, 4) linkage.
    """
    values = np.asarray(scoring_values, dtype=float)
    if values.ndim != 2:
        raise ValueError("scoring_values must be 2D (sites, kinases)")
    n_sites = int(values.shape[0])
    if n_sites <= 1:
        return np.zeros((0, 4), dtype=float)
    condensed = np.asarray(pdist(values, metric=_DISTANCE_METRIC), dtype=float)
    return np.asarray(linkage(condensed, method=_LINKAGE_METHOD), dtype=float)


def cut_labels(linkage_matrix: np.ndarray, *, n_clusters: int, n_sites: int) -> np.ndarray:
    """Cut the Ward tree to obtain 0-indexed labels for ``n_clusters``.

    Canonicalises labels so cluster 0 is the one containing the first row,
    cluster 1 the next distinct one, etc. -- deterministic ordering across
    runs.
    """
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if n_clusters > n_sites:
        raise ValueError(f"n_clusters={n_clusters} > n_sites={n_sites}")
    if n_sites == 0:
        return np.zeros(0, dtype=int)
    if n_sites == 1:
        return np.zeros(1, dtype=int)
    if n_clusters == 1:
        return np.zeros(n_sites, dtype=int)
    if n_clusters == n_sites:
        return np.arange(n_sites, dtype=int)
    raw = np.asarray(cut_tree(linkage_matrix, n_clusters=[int(n_clusters)]), dtype=int).reshape(-1)
    return _canonicalize_labels(raw)


def _canonicalize_labels(raw_labels: np.ndarray) -> np.ndarray:
    """Re-label so that cluster 0 contains the first row, cluster 1 next, etc."""
    labels = np.asarray(raw_labels, dtype=int).reshape(-1)
    if labels.size == 0:
        return np.zeros(0, dtype=int)
    seen: dict[int, int] = {}
    out = np.empty(labels.shape[0], dtype=int)
    next_id = 0
    for i, lbl in enumerate(labels.tolist()):
        if lbl not in seen:
            seen[lbl] = next_id
            next_id += 1
        out[i] = seen[lbl]
    return out


def resolve_scoring_mode(
    scoring_mode: str,
    *,
    n_sites: int,
    max_exact_sites: int,
) -> Literal["exact", "sampled"]:
    """Resolve ``"auto"`` to ``"exact"`` or ``"sampled"`` based on n_sites."""
    if scoring_mode not in SCORING_MODES:
        raise ValueError(f"scoring_mode must be one of {SCORING_MODES}, got {scoring_mode!r}")
    if scoring_mode == SCORING_MODE_EXACT:
        return SCORING_MODE_EXACT
    if scoring_mode == SCORING_MODE_SAMPLED:
        return SCORING_MODE_SAMPLED
    # auto
    if n_sites <= max_exact_sites:
        return SCORING_MODE_EXACT
    warnings.warn(
        f"signalome.cluster_sites: n_sites={n_sites} > max_exact_sites="
        f"{max_exact_sites}; falling back to sampled correlation scoring for "
        "module count selection.  Pass scoring_mode='exact' to force exact "
        "(O(N^2) memory) or bump max_exact_sites if your machine has room.",
        UserWarning,
        stacklevel=3,
    )
    return SCORING_MODE_SAMPLED


def score_candidate_exact(
    scoring_values: np.ndarray,
    labels: np.ndarray,
    *,
    excluded_mask: np.ndarray | None = None,
) -> ClusterCandidateScore:
    """Median within-cluster Pearson correlation, full N x N matrix path."""
    correlation = _pairwise_pearson(scoring_values, excluded_mask=excluded_mask)
    return _summarise_from_correlation_matrix(labels=labels, correlation=correlation)


def score_candidate_sampled(
    scoring_values: np.ndarray,
    labels: np.ndarray,
    *,
    max_samples_per_cluster: int,
    excluded_mask: np.ndarray | None,
    seed: int,
) -> ClusterCandidateScore:
    """Median within-cluster Pearson correlation via per-cluster subsampling."""
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    rng = np.random.default_rng(seed)
    cluster_medians: list[float] = []
    n_singletons = 0
    for cluster_id in range(n_clusters):
        positions = np.flatnonzero(labels == cluster_id)
        if excluded_mask is not None:
            positions = positions[~excluded_mask[positions]]
        if positions.size <= 1:
            n_singletons += 1
            continue
        if positions.size > max_samples_per_cluster:
            positions = np.sort(rng.choice(positions, size=max_samples_per_cluster, replace=False))
        sub = scoring_values[positions, :]
        corr = _pairwise_pearson(sub, excluded_mask=None)
        np.fill_diagonal(corr, np.nan)
        finite = corr[np.isfinite(corr)]
        cluster_medians.append(float(np.median(finite)) if finite.size else 0.0)
    if not cluster_medians:
        median_of_medians = 0.0
        min_med = 0.0
        mean_med = 0.0
    else:
        median_of_medians = float(np.median(cluster_medians))
        min_med = float(min(cluster_medians))
        mean_med = float(np.mean(cluster_medians))
    return ClusterCandidateScore(
        module_count=n_clusters,
        cluster_median_correlations=tuple(cluster_medians),
        min_median_correlation=min_med,
        mean_median_correlation=mean_med,
        median_of_cluster_medians=median_of_medians,
        n_singleton_clusters=n_singletons,
    )


def _pairwise_pearson(
    matrix: np.ndarray,
    *,
    excluded_mask: np.ndarray | None,
) -> np.ndarray:
    """Return the N x N Pearson correlation between rows of ``matrix``.

    ``excluded_mask`` is a per-row bool array; excluded rows produce NaN
    correlations (used to skip constant/degenerate profiles from
    ``summarize_profile_degeneracy``).
    """
    values = np.asarray(matrix, dtype=float)
    n = values.shape[0]
    corr = np.full((n, n), np.nan, dtype=float)
    if excluded_mask is None:
        active = np.ones(n, dtype=bool)
    else:
        active = ~np.asarray(excluded_mask, dtype=bool)
    if active.sum() < 2:
        return corr
    active_idx = np.flatnonzero(active)
    sub = values[active_idx, :]
    # Row-centred data
    row_mean = sub.mean(axis=1, keepdims=True)
    centred = sub - row_mean
    row_norm = np.sqrt((centred * centred).sum(axis=1))
    # Guard against zero-variance rows
    ok = row_norm > NEAR_CONSTANT_VARIANCE_TOLERANCE
    if ok.sum() < 2:
        return corr
    centred = centred[ok, :]
    row_norm = row_norm[ok]
    sub_corr = (centred @ centred.T) / np.outer(row_norm, row_norm)
    idx = active_idx[ok]
    corr[np.ix_(idx, idx)] = sub_corr
    return corr


def _summarise_from_correlation_matrix(
    *,
    labels: np.ndarray,
    correlation: np.ndarray,
) -> ClusterCandidateScore:
    n_clusters = int(labels.max()) + 1 if labels.size else 0
    cluster_medians: list[float] = []
    n_singletons = 0
    for cluster_id in range(n_clusters):
        positions = np.flatnonzero(labels == cluster_id)
        if positions.size <= 1:
            n_singletons += 1
            continue
        block = correlation[np.ix_(positions, positions)].copy()
        np.fill_diagonal(block, np.nan)
        finite = block[np.isfinite(block)]
        cluster_medians.append(float(np.median(finite)) if finite.size else 0.0)
    if cluster_medians:
        median_of_medians = float(np.median(cluster_medians))
        min_med = float(min(cluster_medians))
        mean_med = float(np.mean(cluster_medians))
    else:
        median_of_medians = 0.0
        min_med = 0.0
        mean_med = 0.0
    return ClusterCandidateScore(
        module_count=n_clusters,
        cluster_median_correlations=tuple(cluster_medians),
        min_median_correlation=min_med,
        mean_median_correlation=mean_med,
        median_of_cluster_medians=median_of_medians,
        n_singleton_clusters=n_singletons,
    )


def summarize_profile_degeneracy(
    scoring_values: np.ndarray,
    *,
    variance_tolerance: float = NEAR_CONSTANT_VARIANCE_TOLERANCE,
) -> np.ndarray:
    """Return a bool mask marking rows with near-constant profiles.

    Rows flagged True are excluded from Pearson correlation computations
    (they would produce undefined values).  Same conservative rule as
    PhosPy's ``summarize_profile_degeneracy``.
    """
    values = np.asarray(scoring_values, dtype=float)
    variances = np.nanvar(values, axis=1)
    return ~np.isfinite(variances) | (variances <= variance_tolerance)


def select_module_count(
    scoring_values: np.ndarray,
    linkage_matrix: np.ndarray,
    *,
    requested_module_count: int | None = None,
    max_modules: int = DEFAULT_MAX_MODULES,
    primary_threshold: float = DEFAULT_PRIMARY_THRESHOLD,
    fallback_threshold: float = DEFAULT_FALLBACK_THRESHOLD,
    scoring_mode: str = SCORING_MODE_AUTO,
    max_exact_sites: int = DEFAULT_MAX_EXACT_SITES,
    max_samples_per_cluster: int = DEFAULT_MAX_APPROX_SAMPLES_PER_CLUSTER,
    seed: int = 0,
) -> ModuleCountSelectionResult:
    """Pick the smallest ``k`` whose median cluster correlation meets threshold.

    If ``requested_module_count`` is set, that ``k`` is used directly (still
    returns candidate scores for the requested count for diagnostics).
    """
    _validate_thresholds(primary_threshold, fallback_threshold)
    if max_modules < 1:
        raise ValueError("max_modules must be >= 1")
    n_sites = int(scoring_values.shape[0])
    if n_sites <= 1:
        return ModuleCountSelectionResult(
            module_count=1,
            selection_reason="requested" if requested_module_count == 1 else "trivial",
            threshold_used=None,
            candidate_scores={},
            scoring_mode_used=SCORING_MODE_EXACT,
            n_sites=n_sites,
            max_exact_sites=max_exact_sites,
        )
    resolved_mode = resolve_scoring_mode(
        scoring_mode, n_sites=n_sites, max_exact_sites=max_exact_sites
    )
    max_modules_feasible = min(max_modules, n_sites)
    if requested_module_count is not None:
        if requested_module_count < 1 or requested_module_count > n_sites:
            raise ValueError(
                f"requested_module_count={requested_module_count} outside [1, {n_sites}]"
            )
        labels = cut_labels(linkage_matrix, n_clusters=requested_module_count, n_sites=n_sites)
        score = _score_one(
            scoring_values=scoring_values,
            labels=labels,
            resolved_mode=resolved_mode,
            max_samples_per_cluster=max_samples_per_cluster,
            seed=seed,
        )
        return ModuleCountSelectionResult(
            module_count=int(requested_module_count),
            selection_reason="requested",
            threshold_used=None,
            candidate_scores={int(requested_module_count): score},
            scoring_mode_used=resolved_mode,
            n_sites=n_sites,
            max_exact_sites=max_exact_sites,
        )
    candidate_scores: dict[int, ClusterCandidateScore] = {}
    for k in range(2, max_modules_feasible + 1):
        labels = cut_labels(linkage_matrix, n_clusters=k, n_sites=n_sites)
        candidate_scores[k] = _score_one(
            scoring_values=scoring_values,
            labels=labels,
            resolved_mode=resolved_mode,
            max_samples_per_cluster=max_samples_per_cluster,
            seed=seed,
        )
    selection_reason, threshold_used, chosen_k = _select_by_threshold(
        candidate_scores=candidate_scores,
        primary_threshold=primary_threshold,
        fallback_threshold=fallback_threshold,
        max_modules_feasible=max_modules_feasible,
    )
    return ModuleCountSelectionResult(
        module_count=chosen_k,
        selection_reason=selection_reason,
        threshold_used=threshold_used,
        candidate_scores=candidate_scores,
        scoring_mode_used=resolved_mode,
        n_sites=n_sites,
        max_exact_sites=max_exact_sites,
    )


def _score_one(
    *,
    scoring_values: np.ndarray,
    labels: np.ndarray,
    resolved_mode: str,
    max_samples_per_cluster: int,
    seed: int,
) -> ClusterCandidateScore:
    excluded_mask = summarize_profile_degeneracy(scoring_values)
    if resolved_mode == SCORING_MODE_EXACT:
        return score_candidate_exact(scoring_values, labels, excluded_mask=excluded_mask)
    return score_candidate_sampled(
        scoring_values,
        labels,
        max_samples_per_cluster=max_samples_per_cluster,
        excluded_mask=excluded_mask,
        seed=seed,
    )


def _select_by_threshold(
    *,
    candidate_scores: dict[int, ClusterCandidateScore],
    primary_threshold: float,
    fallback_threshold: float,
    max_modules_feasible: int,
) -> tuple[str, float | None, int]:
    """PhosPy's rule: filter by min-median >= threshold, pick k that maximises
    mean-median-correlation, tie-break by smallest k.  Fallback threshold
    is tried when nothing passes primary."""
    primary_pass = {
        k: score
        for k, score in candidate_scores.items()
        if score.min_median_correlation >= primary_threshold
    }
    if primary_pass:
        best = max(
            primary_pass.items(),
            key=lambda kv: (kv[1].mean_median_correlation, -kv[0]),
        )
        return "primary", primary_threshold, best[0]
    fallback_pass = {
        k: score
        for k, score in candidate_scores.items()
        if score.min_median_correlation >= fallback_threshold
    }
    if fallback_pass:
        best = max(
            fallback_pass.items(),
            key=lambda kv: (kv[1].mean_median_correlation, -kv[0]),
        )
        return "fallback", fallback_threshold, best[0]
    return "max_default", None, max_modules_feasible


def _validate_thresholds(primary: float, fallback: float) -> None:
    for name, value in (("primary_threshold", primary), ("fallback_threshold", fallback)):
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError(f"{name} must lie in [0, 1]")


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------


def cluster_sites(
    prediction_matrix: pd.DataFrame,
    *,
    requested_module_count: int | None = None,
    max_modules: int = DEFAULT_MAX_MODULES,
    primary_threshold: float = DEFAULT_PRIMARY_THRESHOLD,
    fallback_threshold: float = DEFAULT_FALLBACK_THRESHOLD,
    scoring_mode: str = SCORING_MODE_AUTO,
    max_exact_sites: int = DEFAULT_MAX_EXACT_SITES,
    max_samples_per_cluster: int = DEFAULT_MAX_APPROX_SAMPLES_PER_CLUSTER,
    seed: int = 0,
) -> SignalomeClusteringResult:
    """Run the full clustering pipeline on ``prediction_matrix``.

    Parameters
    ----------
    prediction_matrix
        (n_sites, n_kinases) DataFrame; rows are sites, columns are kinases,
        cells are per-site kinase-prediction scores.  Typically the output of
        :func:`alphaphos.score_kinases` (``adata.varm["kinase_score_ser_thr"]``
        or ``_tyrosine``).
    requested_module_count
        Skip the threshold-based selection and use this ``k`` directly.
        Diagnostics are still recorded.
    max_modules
        Upper bound on the candidate ``k``s tried.  Default 10.
    primary_threshold, fallback_threshold
        Median-cluster-correlation thresholds for module-count selection.
        Defaults 0.5 / 0.1 (matches PhosPy).
    scoring_mode
        ``"auto"`` (default), ``"exact"``, or ``"sampled"``.  See module
        docstring.
    max_exact_sites
        The N boundary at which ``"auto"`` switches from exact to sampled.
        Default 5000.
    max_samples_per_cluster
        Sample size per cluster in sampled mode.  Default 200.
    seed
        RNG seed for the sampled path.

    Returns
    -------
    SignalomeClusteringResult
    """
    values_prep = precondition_scores(prediction_matrix.to_numpy(dtype=float, copy=True))
    linkage_matrix = build_ward_tree(values_prep)
    selection = select_module_count(
        values_prep,
        linkage_matrix,
        requested_module_count=requested_module_count,
        max_modules=max_modules,
        primary_threshold=primary_threshold,
        fallback_threshold=fallback_threshold,
        scoring_mode=scoring_mode,
        max_exact_sites=max_exact_sites,
        max_samples_per_cluster=max_samples_per_cluster,
        seed=seed,
    )
    n_sites = int(prediction_matrix.shape[0])
    zero_indexed = cut_labels(linkage_matrix, n_clusters=selection.module_count, n_sites=n_sites)
    # Convention: module_id 0 is reserved for "unassigned" downstream; shift
    # cluster labels to 1..k so 0 remains reserved.
    labels = zero_indexed + 1
    provenance = {
        "n_sites": n_sites,
        "n_kinases": int(prediction_matrix.shape[1]),
        "module_count": int(selection.module_count),
        "selection_reason": selection.selection_reason,
        "threshold_used": selection.threshold_used,
        "scoring_mode_requested": scoring_mode,
        "scoring_mode_used": selection.scoring_mode_used,
        "max_exact_sites": max_exact_sites,
        "max_samples_per_cluster": max_samples_per_cluster,
        "seed": seed,
        "linkage_method": _LINKAGE_METHOD,
        "distance_metric": _DISTANCE_METRIC,
    }
    return SignalomeClusteringResult(
        labels=labels,
        module_count=int(selection.module_count),
        selection=selection,
        linkage_matrix=linkage_matrix,
        site_index=pd.Index(prediction_matrix.index.astype(str)),
        provenance=provenance,
    )
