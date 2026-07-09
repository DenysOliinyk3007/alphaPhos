"""Kinase-kinase network from the prediction-score matrix.

Compute pairwise Pearson correlations across the **kinase columns** of the
prediction matrix (each column = a kinase's per-site prediction vector).
Two kinases with highly correlated per-site score profiles co-regulate
similar sites -- an edge in the signalome kinase network.

Three edge-selection policies (matches PhosPy conventions):

- ``"signed"`` (default): keep the signed correlation; edge exists when
  ``|corr| >= threshold``.
- ``"positive_only"``: keep only positive correlations; edge exists when
  ``corr >= threshold``.
- ``"absolute"``: use ``|corr|`` as the edge weight; edge exists when
  ``|corr| >= threshold``.

Diagnostics: correlation status per candidate pair (finite / missing /
constant / insufficient / non-finite), returned as a separate DataFrame
alongside the edges.  Node table carries per-kinase ``degree`` (edge
count) and ``n_substrates`` (from ``kinase_substrates`` map).

Clean-room re-implementation of PhosPy's ``science.signalomes.network``
(MIT; PhosPy GPL-3.0 used only as numerical oracle).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaphos.signalome.constants import (
    CORRELATION_COLUMN,
    DEGREE_COLUMN,
    KINASE_COLUMN,
    N_SUBSTRATES_COLUMN,
    NETWORK_POLICIES,
    NETWORK_POLICY_ABSOLUTE,
    NETWORK_POLICY_POSITIVE_ONLY,
    NETWORK_POLICY_SIGNED,
    SOURCE_KINASE_COLUMN,
    TARGET_KINASE_COLUMN,
)

_CORRELATION_STATUS_COLUMN = "correlation_status"
_CORRELATION_REASON_COLUMN = "correlation_reason"
_VALID_OBSERVATIONS_COLUMN = "valid_observations"

_STATUS_FINITE = "finite"
_STATUS_CONSTANT = "constant_profile"
_STATUS_INSUFFICIENT = "insufficient_observations"
_STATUS_MISSING = "missing_values"
_STATUS_NON_FINITE = "non_finite_values"
_STATUS_UNDEFINED = "undefined"


@dataclass(frozen=True, slots=True)
class KinaseNetwork:
    """Kinase-kinase network output."""

    edges: pd.DataFrame
    """Columns: source_kinase, target_kinase, correlation."""

    nodes: pd.DataFrame
    """Columns: degree, n_substrates.  Indexed by kinase."""

    candidate_correlations: pd.DataFrame
    """Every candidate pair with its correlation status + diagnostic reason."""

    provenance: dict[str, object]


def build_kinase_network(
    *,
    prediction_matrix: pd.DataFrame,
    kinase_order: Sequence[str],
    kinase_substrates: Mapping[str, Sequence[str]],
    threshold: float,
    network_policy: str = NETWORK_POLICY_SIGNED,
    min_paired_observations: int = 2,
) -> KinaseNetwork:
    """Build the kinase-kinase network.

    Parameters
    ----------
    prediction_matrix
        (n_sites, n_kinases).  Rows are sites, columns are kinases.
    kinase_order
        Explicit kinase-column order to include in the network.  Kinases
        missing from ``prediction_matrix.columns`` raise a ValueError.
    kinase_substrates
        ``{kinase: [substrate_site_ids]}`` used for the node table's
        ``n_substrates`` column.  Only affects the nodes table, not the
        edges.
    threshold
        Absolute-correlation threshold for edge inclusion.  Default 0.5.
    network_policy
        ``"signed"`` (default), ``"positive_only"``, or ``"absolute"``.
    min_paired_observations
        Minimum finite paired observations required to compute a
        correlation.  Below this the pair is skipped with an
        ``insufficient_observations`` status.
    """
    if network_policy not in NETWORK_POLICIES:
        raise ValueError(
            f"network_policy must be one of {NETWORK_POLICIES}, got {network_policy!r}"
        )
    kinase_names = list(dict.fromkeys(str(k) for k in kinase_order))
    if not kinase_names:
        raise ValueError("kinase_order must contain at least one kinase")
    available = set(prediction_matrix.columns.astype(str).tolist())
    missing = [k for k in kinase_names if k not in available]
    if missing:
        preview = ", ".join(missing[:3])
        suffix = "..." if len(missing) > 3 else ""
        raise ValueError(
            f"prediction_matrix is missing kinases from kinase_order: {preview}{suffix}"
        )
    aligned = _precondition_scores(prediction_matrix=prediction_matrix, kinase_names=kinase_names)
    candidate = _build_candidate_correlations(
        aligned=aligned,
        kinase_names=kinase_names,
        min_paired_observations=min_paired_observations,
    )
    edges = _resolve_edges(
        candidate=candidate, threshold=float(threshold), network_policy=network_policy
    )
    nodes = _build_nodes(
        kinase_names=kinase_names, edges=edges, kinase_substrates=kinase_substrates
    )
    provenance = {
        "n_kinases": len(kinase_names),
        "n_candidate_pairs": int(candidate.shape[0]),
        "n_edges": int(edges.shape[0]),
        "threshold": float(threshold),
        "network_policy": network_policy,
        "min_paired_observations": int(min_paired_observations),
    }
    return KinaseNetwork(
        edges=edges,
        nodes=nodes,
        candidate_correlations=candidate,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _precondition_scores(
    *,
    prediction_matrix: pd.DataFrame,
    kinase_names: list[str],
) -> pd.DataFrame:
    aligned = prediction_matrix.loc[:, kinase_names].astype(float)
    # Drop rows that are entirely NaN across the selected kinases -- keeps
    # per-pair valid_observations honest.
    any_valid = aligned.notna().any(axis=1).to_numpy(dtype=bool)
    if any_valid.all():
        return aligned
    return aligned.iloc[any_valid, :]


def _build_candidate_correlations(
    *,
    aligned: pd.DataFrame,
    kinase_names: list[str],
    min_paired_observations: int,
) -> pd.DataFrame:
    if len(kinase_names) < 2:
        return _empty_candidate_table()
    score_values = {k: aligned[k].to_numpy(dtype=float) for k in kinase_names}
    rows: list[dict[str, object]] = []
    for i, source in enumerate(kinase_names[:-1]):
        source_vec = score_values[source]
        for target in kinase_names[i + 1 :]:
            correlation, status, valid_n, reason = _classify_pair(
                source=source_vec,
                target=score_values[target],
                min_paired_observations=min_paired_observations,
            )
            rows.append(
                {
                    SOURCE_KINASE_COLUMN: source,
                    TARGET_KINASE_COLUMN: target,
                    CORRELATION_COLUMN: correlation,
                    _CORRELATION_STATUS_COLUMN: status,
                    _VALID_OBSERVATIONS_COLUMN: valid_n,
                    _CORRELATION_REASON_COLUMN: reason,
                }
            )
    if not rows:
        return _empty_candidate_table()
    df = pd.DataFrame.from_records(rows)
    return df.astype(
        {
            SOURCE_KINASE_COLUMN: str,
            TARGET_KINASE_COLUMN: str,
            CORRELATION_COLUMN: float,
            _CORRELATION_STATUS_COLUMN: str,
            _VALID_OBSERVATIONS_COLUMN: "int64",
        }
    )


def _classify_pair(
    *,
    source: np.ndarray,
    target: np.ndarray,
    min_paired_observations: int,
) -> tuple[float, str, int, str | None]:
    """Return (correlation, status, valid_n, reason).

    Guards against NaN, infinite, constant, and insufficient-observation
    inputs.  Correlation returned as NaN when undefined; caller decides
    whether to admit the pair to the edge set.
    """
    src = np.asarray(source, dtype=float)
    tgt = np.asarray(target, dtype=float)
    finite_pair = np.isfinite(src) & np.isfinite(tgt)
    valid_n = int(finite_pair.sum())
    if np.isinf(src).any() or np.isinf(tgt).any():
        return float("nan"), _STATUS_NON_FINITE, valid_n, "input has +/-inf"
    if valid_n < int(min_paired_observations):
        if np.isnan(src).any() or np.isnan(tgt).any():
            return (
                float("nan"),
                _STATUS_MISSING,
                valid_n,
                "NaN reduced paired obs below minimum",
            )
        return (
            float("nan"),
            _STATUS_INSUFFICIENT,
            valid_n,
            "fewer than min_paired_observations paired finite observations",
        )
    src_valid = src[finite_pair]
    tgt_valid = tgt[finite_pair]
    src_var = float(np.var(src_valid, ddof=0))
    tgt_var = float(np.var(tgt_valid, ddof=0))
    if src_var == 0.0 or tgt_var == 0.0:
        return (
            float("nan"),
            _STATUS_CONSTANT,
            valid_n,
            "zero variance in paired finite observations",
        )
    corr = float(np.corrcoef(src_valid, tgt_valid)[0, 1])
    if np.isfinite(corr):
        return float(np.clip(corr, -1.0, 1.0)), _STATUS_FINITE, valid_n, None
    return (
        float("nan"),
        _STATUS_UNDEFINED,
        valid_n,
        "np.corrcoef returned non-finite",
    )


def _resolve_edges(
    *,
    candidate: pd.DataFrame,
    threshold: float,
    network_policy: str,
) -> pd.DataFrame:
    finite = candidate.loc[
        candidate[_CORRELATION_STATUS_COLUMN].eq(_STATUS_FINITE)
        & candidate[CORRELATION_COLUMN].notna(),
        [SOURCE_KINASE_COLUMN, TARGET_KINASE_COLUMN, CORRELATION_COLUMN],
    ].reset_index(drop=True)
    if finite.empty:
        return _empty_edges_table()
    corr = finite[CORRELATION_COLUMN].to_numpy(dtype=float)
    if network_policy == NETWORK_POLICY_POSITIVE_ONLY:
        edge_vals, edge_mask = corr, corr >= float(threshold)
    elif network_policy == NETWORK_POLICY_ABSOLUTE:
        edge_vals, edge_mask = np.abs(corr), np.abs(corr) >= float(threshold)
    else:  # signed (default)
        edge_vals, edge_mask = corr, np.abs(corr) >= float(threshold)
    selected = finite.loc[edge_mask].copy()
    selected[CORRELATION_COLUMN] = edge_vals[edge_mask]
    selected = selected.astype(
        {
            SOURCE_KINASE_COLUMN: str,
            TARGET_KINASE_COLUMN: str,
            CORRELATION_COLUMN: float,
        }
    )
    return selected.sort_values(
        [SOURCE_KINASE_COLUMN, TARGET_KINASE_COLUMN], ascending=[True, True], kind="stable"
    ).reset_index(drop=True)


def _build_nodes(
    *,
    kinase_names: list[str],
    edges: pd.DataFrame,
    kinase_substrates: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    idx = pd.Index(kinase_names, name=KINASE_COLUMN)
    degree = pd.Series(0, index=idx.copy(), dtype="int64")
    if not edges.empty:
        combined = pd.concat(
            [edges[SOURCE_KINASE_COLUMN], edges[TARGET_KINASE_COLUMN]], axis=0
        ).value_counts()
        common = idx.intersection(combined.index)
        degree.loc[common] = combined.loc[common].astype("int64")
    n_substrates = np.asarray(
        [len(tuple(kinase_substrates.get(str(k), ()))) for k in kinase_names],
        dtype=np.int64,
    )
    return pd.DataFrame(
        {DEGREE_COLUMN: degree.to_numpy(dtype=np.int64), N_SUBSTRATES_COLUMN: n_substrates},
        index=idx.copy(),
    ).astype({DEGREE_COLUMN: "int64", N_SUBSTRATES_COLUMN: "int64"})


def _empty_edges_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[SOURCE_KINASE_COLUMN, TARGET_KINASE_COLUMN, CORRELATION_COLUMN]
    ).astype({SOURCE_KINASE_COLUMN: str, TARGET_KINASE_COLUMN: str, CORRELATION_COLUMN: float})


def _empty_candidate_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            SOURCE_KINASE_COLUMN,
            TARGET_KINASE_COLUMN,
            CORRELATION_COLUMN,
            _CORRELATION_STATUS_COLUMN,
            _VALID_OBSERVATIONS_COLUMN,
            _CORRELATION_REASON_COLUMN,
        ]
    ).astype(
        {
            SOURCE_KINASE_COLUMN: str,
            TARGET_KINASE_COLUMN: str,
            CORRELATION_COLUMN: float,
            _CORRELATION_STATUS_COLUMN: str,
            _VALID_OBSERVATIONS_COLUMN: "int64",
        }
    )
