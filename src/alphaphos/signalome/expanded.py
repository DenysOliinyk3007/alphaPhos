"""Expanded per-kinase signalome view.

Emits a denormalized "kinase perspective" table.  For each kinase K in
``kinase_order`` this returns either:

- One or more **"site" rows** -- for every site whose top-kinase (or a
  tied co-top-kinase) supports K, restricted to the sites that live in
  modules where K (or a network-linked kinase) is a substantial
  substrate contributor.
- One **"summary" row** -- when there is no such support (either no
  regulated modules, or none of them contain a supported site).

This is the format PhosPy's ``expanded_signalome`` table uses -- useful
as a flat export for graph-visualisation tools (each row = one edge in
the site -> kinase graph) and for downstream reasoning about how kinases
share modules.

Clean-room re-implementation of PhosPy's
``science.signalomes.expanded.build_expanded_signalome_table``
(MIT; PhosPy GPL-3.0 used only as numerical oracle).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from alphaphos.signalome.constants import (
    DISPLAY_ID_COLUMN,
    GENE_SYMBOL_COLUMN,
    ISOFORM_ID_COLUMN,
    KINASE_COLUMN,
    MODULE_ID_COLUMN,
    PROTEIN_ACCESSION_COLUMN,
    PROTEIN_COLUMN,
    SITE_COLUMN,
    SITE_KEY_COLUMN,
    SOURCE_KINASE_COLUMN,
    TARGET_KINASE_COLUMN,
    TOP_KINASE_COLUMN,
    TOP_KINASE_WEIGHTS_COLUMN,
    TOP_SCORE_COLUMN,
)

_ROW_KIND_COLUMN = "row_kind"
_ROW_KIND_SITE = "site"
_ROW_KIND_SUMMARY = "summary"
_ASSIGNMENT_POLICY_COLUMN = "assignment_policy"
_LINKED_KINASES_COLUMN = "linked_kinases"
_REGULATED_MODULE_IDS_COLUMN = "regulated_module_ids"
_SUPPORT_KINASES_COLUMN = "support_kinases"
_SUPPORT_WEIGHT_COLUMN = "support_weight"
_SITE_ORDER_COLUMN = "site_order"
_SITE_ID_COLUMN = "site_id"

_JSON_EMPTY_ARRAY = "[]"
_MIN_MODULE_SHARE_PERCENT = 1.0

_ASSIGNMENT_POLICY_CUTOFF_BINARY = "cutoff_binary"
_ASSIGNMENT_POLICY_WEIGHTED_TOP = "weighted_top"
_ASSIGNMENT_POLICIES = (
    _ASSIGNMENT_POLICY_CUTOFF_BINARY,
    _ASSIGNMENT_POLICY_WEIGHTED_TOP,
)


def build_expanded_table(
    *,
    module_assignments: pd.DataFrame,
    module_table: pd.DataFrame,
    network_edges: pd.DataFrame,
    kinase_substrates: Mapping[str, Sequence[str]],
    kinase_order: Sequence[str],
    assignment_policy: str = _ASSIGNMENT_POLICY_CUTOFF_BINARY,
    min_module_share_percent: float = _MIN_MODULE_SHARE_PERCENT,
) -> pd.DataFrame:
    """Emit the denormalized per-kinase expanded view.

    Parameters
    ----------
    module_assignments
        Per-site DataFrame from
        :func:`alphaphos.signalome.build_module_assignments`.
    module_table
        Modules x kinases percent-share DataFrame from
        :func:`alphaphos.signalome.build_module_table`.
    network_edges
        (source_kinase, target_kinase, correlation) DataFrame from
        :func:`alphaphos.signalome.build_kinase_network`.
    kinase_substrates
        ``{kinase: [substrate_site_id, ...]}`` mapping.
    kinase_order
        Explicit iteration order over kinases (one row set per focal
        kinase).
    assignment_policy
        ``"cutoff_binary"`` (default) or ``"weighted_top"``.  Determines
        how per-site support weights are computed:

        - ``cutoff_binary``: weight = 1.0 for each ``(site, kinase)``
          where site is in ``kinase_substrates[kinase]``.
        - ``weighted_top``: weights come from the per-site
          ``top_kinase_weights`` column of ``module_assignments``.

    min_module_share_percent
        A kinase's "regulated modules" are the modules where the kinase
        has share >= this cutoff in ``module_table``.  Default 1.0
        (matches PhosPy).
    """
    if assignment_policy not in _ASSIGNMENT_POLICIES:
        raise ValueError(
            f"assignment_policy must be one of {_ASSIGNMENT_POLICIES}, got {assignment_policy!r}"
        )
    kinase_names = list(dict.fromkeys(str(k) for k in kinase_order))
    neighbor_map = _build_neighbor_map(edges=network_edges)
    support_by_kinase = _build_support_arrays(
        module_assignments=module_assignments,
        kinase_substrates=kinase_substrates,
        kinase_names=kinase_names,
        assignment_policy=assignment_policy,
    )
    regulated_by_kinase = _build_regulated_modules(
        module_table=module_table,
        kinase_names=kinase_names,
        min_share=float(min_module_share_percent),
    )
    module_site_positions = _build_module_site_positions(
        module_ids=module_assignments[MODULE_ID_COLUMN].astype("int64").to_numpy()
    )
    identity = _extract_identity(module_assignments)

    n_sites = int(module_assignments.shape[0])
    rows: list[dict[str, object]] = []
    for focal in kinase_names:
        linked = tuple(dict.fromkeys((focal, *neighbor_map.get(focal, ()))))
        regulated = regulated_by_kinase.get(focal, ())
        linked_json = json.dumps(list(linked), separators=(",", ":"), ensure_ascii=True)
        regulated_json = json.dumps(list(regulated), separators=(",", ":"), ensure_ascii=True)
        candidate_positions = _collect_candidate_positions(
            regulated=regulated,
            module_site_positions=module_site_positions,
            n_sites=n_sites,
        )
        if candidate_positions.size == 0:
            rows.append(
                _summary_row(
                    focal=focal,
                    assignment_policy=assignment_policy,
                    linked_json=linked_json,
                    regulated_json=regulated_json,
                )
            )
            continue
        # Sum support across all linked kinases at each candidate position.
        candidate_weights = np.zeros(candidate_positions.size, dtype=float)
        contributing_supports: list[tuple[str, np.ndarray]] = []
        for lk in linked:
            arr = support_by_kinase.get(lk)
            if arr is None:
                continue
            contributing_supports.append((lk, arr))
            candidate_weights += arr[candidate_positions]
        supported_mask = candidate_weights > 0.0
        if not supported_mask.any():
            rows.append(
                _summary_row(
                    focal=focal,
                    assignment_policy=assignment_policy,
                    linked_json=linked_json,
                    regulated_json=regulated_json,
                )
            )
            continue
        matched_positions = candidate_positions[supported_mask]
        matched_weights = candidate_weights[supported_mask]
        for position, weight in zip(matched_positions, matched_weights, strict=True):
            support_kinases_for_site = tuple(
                lk for lk, arr in contributing_supports if arr[position] > 0.0
            )
            support_kinases_json = json.dumps(
                list(support_kinases_for_site),
                separators=(",", ":"),
                ensure_ascii=True,
            )
            rows.append(
                _site_row(
                    focal=focal,
                    assignment_policy=assignment_policy,
                    linked_json=linked_json,
                    regulated_json=regulated_json,
                    identity=identity,
                    position=int(position),
                    support_weight=float(weight),
                    support_kinases_json=support_kinases_json,
                )
            )

    if not rows:
        return _empty_expanded_table()
    return pd.DataFrame(rows, columns=_EXPANDED_COLUMNS).astype(_EXPANDED_DTYPES)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_EXPANDED_COLUMNS = (
    KINASE_COLUMN,
    _ROW_KIND_COLUMN,
    _ASSIGNMENT_POLICY_COLUMN,
    _LINKED_KINASES_COLUMN,
    _REGULATED_MODULE_IDS_COLUMN,
    SITE_KEY_COLUMN,
    DISPLAY_ID_COLUMN,
    _SITE_ID_COLUMN,
    _SITE_ORDER_COLUMN,
    GENE_SYMBOL_COLUMN,
    SITE_COLUMN,
    PROTEIN_COLUMN,
    PROTEIN_ACCESSION_COLUMN,
    ISOFORM_ID_COLUMN,
    MODULE_ID_COLUMN,
    _SUPPORT_KINASES_COLUMN,
    _SUPPORT_WEIGHT_COLUMN,
    TOP_KINASE_COLUMN,
    TOP_SCORE_COLUMN,
)


_EXPANDED_DTYPES = {
    KINASE_COLUMN: str,
    _ROW_KIND_COLUMN: str,
    _ASSIGNMENT_POLICY_COLUMN: str,
    _LINKED_KINASES_COLUMN: str,
    _REGULATED_MODULE_IDS_COLUMN: str,
    SITE_KEY_COLUMN: str,
    DISPLAY_ID_COLUMN: str,
    _SITE_ID_COLUMN: str,
    _SITE_ORDER_COLUMN: "Int64",
    GENE_SYMBOL_COLUMN: str,
    SITE_COLUMN: str,
    PROTEIN_COLUMN: str,
    PROTEIN_ACCESSION_COLUMN: str,
    ISOFORM_ID_COLUMN: str,
    MODULE_ID_COLUMN: "Int64",
    _SUPPORT_KINASES_COLUMN: str,
    _SUPPORT_WEIGHT_COLUMN: float,
    TOP_KINASE_COLUMN: str,
    TOP_SCORE_COLUMN: float,
}


def _build_neighbor_map(*, edges: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Undirected neighbours from edges DataFrame."""
    if edges.empty:
        return {}
    neighbours: dict[str, list[str]] = {}
    for src, tgt in zip(
        edges[SOURCE_KINASE_COLUMN].astype(str),
        edges[TARGET_KINASE_COLUMN].astype(str),
        strict=True,
    ):
        neighbours.setdefault(src, []).append(tgt)
        neighbours.setdefault(tgt, []).append(src)
    # Preserve order and deduplicate.
    return {k: tuple(dict.fromkeys(v)) for k, v in neighbours.items()}


def _build_support_arrays(
    *,
    module_assignments: pd.DataFrame,
    kinase_substrates: Mapping[str, Sequence[str]],
    kinase_names: list[str],
    assignment_policy: str,
) -> dict[str, np.ndarray]:
    """Per-kinase per-site support-weight vector aligned to module_assignments.index."""
    site_to_position = {str(sid): i for i, sid in enumerate(module_assignments.index.astype(str))}
    n_sites = int(module_assignments.shape[0])
    out: dict[str, np.ndarray] = {}
    if assignment_policy == _ASSIGNMENT_POLICY_CUTOFF_BINARY:
        for kinase in kinase_names:
            support = np.zeros(n_sites, dtype=float)
            for site_id in kinase_substrates.get(kinase, ()):
                pos = site_to_position.get(str(site_id))
                if pos is not None:
                    support[pos] = 1.0
            out[kinase] = support
        return out
    # weighted_top
    if TOP_KINASE_WEIGHTS_COLUMN not in module_assignments.columns:
        raise ValueError(
            f"module_assignments missing {TOP_KINASE_WEIGHTS_COLUMN!r} column "
            "required for assignment_policy='weighted_top'"
        )
    for kinase in kinase_names:
        out[kinase] = np.zeros(n_sites, dtype=float)
    weights_lists = module_assignments[TOP_KINASE_WEIGHTS_COLUMN].tolist()
    for pos, weights in enumerate(weights_lists):
        if not weights:
            continue
        for entry in weights:
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                continue
            kinase_name = str(entry[0])
            weight_value = float(entry[1])
            if kinase_name in out and weight_value > 0.0:
                out[kinase_name][pos] = max(out[kinase_name][pos], weight_value)
    return out


def _build_regulated_modules(
    *,
    module_table: pd.DataFrame,
    kinase_names: list[str],
    min_share: float,
) -> dict[str, tuple[int, ...]]:
    """For each kinase, the modules where its %-share meets the threshold."""
    if module_table.empty:
        return {k: () for k in kinase_names}
    out: dict[str, tuple[int, ...]] = {}
    for kinase in kinase_names:
        if kinase not in module_table.columns:
            out[kinase] = ()
            continue
        col = module_table[kinase].astype(float)
        mask = col >= float(min_share)
        modules_selected = tuple(sorted(int(m) for m in col.index[mask].tolist()))
        out[kinase] = modules_selected
    return out


def _build_module_site_positions(
    *,
    module_ids: np.ndarray,
) -> dict[int, np.ndarray]:
    positions_by_module: dict[int, list[int]] = {}
    for i, m in enumerate(module_ids.astype(int)):
        if m <= 0:
            continue
        positions_by_module.setdefault(int(m), []).append(i)
    return {m: np.asarray(pos, dtype=int) for m, pos in positions_by_module.items()}


def _collect_candidate_positions(
    *,
    regulated: tuple[int, ...],
    module_site_positions: dict[int, np.ndarray],
    n_sites: int,
) -> np.ndarray:
    """Union of site positions from all regulated modules, sorted."""
    if not regulated:
        return np.zeros(0, dtype=int)
    seen = np.zeros(n_sites, dtype=bool)
    for m in regulated:
        pos_arr = module_site_positions.get(int(m))
        if pos_arr is not None and pos_arr.size:
            seen[pos_arr] = True
    return np.flatnonzero(seen).astype(int)


def _extract_identity(module_assignments: pd.DataFrame) -> dict[str, np.ndarray]:
    """Pull the columns we'll need per-site into numpy arrays for fast lookup."""
    idx = module_assignments.index.astype(str).to_numpy()
    return {
        SITE_KEY_COLUMN: (
            module_assignments[SITE_KEY_COLUMN].astype(str).to_numpy()
            if SITE_KEY_COLUMN in module_assignments.columns
            else idx
        ),
        DISPLAY_ID_COLUMN: module_assignments[DISPLAY_ID_COLUMN].astype(str).to_numpy(),
        GENE_SYMBOL_COLUMN: module_assignments[GENE_SYMBOL_COLUMN].astype(str).to_numpy(),
        SITE_COLUMN: module_assignments[SITE_COLUMN].astype(str).to_numpy(),
        PROTEIN_COLUMN: module_assignments[PROTEIN_COLUMN].astype(str).to_numpy(),
        PROTEIN_ACCESSION_COLUMN: (
            module_assignments[PROTEIN_ACCESSION_COLUMN].astype(str).to_numpy()
            if PROTEIN_ACCESSION_COLUMN in module_assignments.columns
            else np.array([""] * len(idx), dtype=object)
        ),
        ISOFORM_ID_COLUMN: (
            module_assignments[ISOFORM_ID_COLUMN].astype(str).to_numpy()
            if ISOFORM_ID_COLUMN in module_assignments.columns
            else np.array([""] * len(idx), dtype=object)
        ),
        MODULE_ID_COLUMN: module_assignments[MODULE_ID_COLUMN].astype("int64").to_numpy(),
        TOP_KINASE_COLUMN: module_assignments[TOP_KINASE_COLUMN].astype(str).to_numpy(),
        TOP_SCORE_COLUMN: module_assignments[TOP_SCORE_COLUMN].astype(float).to_numpy(),
    }


def _summary_row(
    *,
    focal: str,
    assignment_policy: str,
    linked_json: str,
    regulated_json: str,
) -> dict[str, object]:
    return {
        KINASE_COLUMN: focal,
        _ROW_KIND_COLUMN: _ROW_KIND_SUMMARY,
        _ASSIGNMENT_POLICY_COLUMN: assignment_policy,
        _LINKED_KINASES_COLUMN: linked_json,
        _REGULATED_MODULE_IDS_COLUMN: regulated_json,
        SITE_KEY_COLUMN: "",
        DISPLAY_ID_COLUMN: "",
        _SITE_ID_COLUMN: "",
        _SITE_ORDER_COLUMN: pd.NA,
        GENE_SYMBOL_COLUMN: "",
        SITE_COLUMN: "",
        PROTEIN_COLUMN: "",
        PROTEIN_ACCESSION_COLUMN: "",
        ISOFORM_ID_COLUMN: "",
        MODULE_ID_COLUMN: pd.NA,
        _SUPPORT_KINASES_COLUMN: _JSON_EMPTY_ARRAY,
        _SUPPORT_WEIGHT_COLUMN: 0.0,
        TOP_KINASE_COLUMN: "",
        TOP_SCORE_COLUMN: float("nan"),
    }


def _site_row(
    *,
    focal: str,
    assignment_policy: str,
    linked_json: str,
    regulated_json: str,
    identity: dict[str, np.ndarray],
    position: int,
    support_weight: float,
    support_kinases_json: str,
) -> dict[str, object]:
    return {
        KINASE_COLUMN: focal,
        _ROW_KIND_COLUMN: _ROW_KIND_SITE,
        _ASSIGNMENT_POLICY_COLUMN: assignment_policy,
        _LINKED_KINASES_COLUMN: linked_json,
        _REGULATED_MODULE_IDS_COLUMN: regulated_json,
        SITE_KEY_COLUMN: str(identity[SITE_KEY_COLUMN][position]),
        DISPLAY_ID_COLUMN: str(identity[DISPLAY_ID_COLUMN][position]),
        _SITE_ID_COLUMN: str(identity[DISPLAY_ID_COLUMN][position]),
        _SITE_ORDER_COLUMN: int(position),
        GENE_SYMBOL_COLUMN: str(identity[GENE_SYMBOL_COLUMN][position]),
        SITE_COLUMN: str(identity[SITE_COLUMN][position]),
        PROTEIN_COLUMN: str(identity[PROTEIN_COLUMN][position]),
        PROTEIN_ACCESSION_COLUMN: str(identity[PROTEIN_ACCESSION_COLUMN][position]),
        ISOFORM_ID_COLUMN: str(identity[ISOFORM_ID_COLUMN][position]),
        MODULE_ID_COLUMN: int(identity[MODULE_ID_COLUMN][position]),
        _SUPPORT_KINASES_COLUMN: support_kinases_json,
        _SUPPORT_WEIGHT_COLUMN: float(support_weight),
        TOP_KINASE_COLUMN: str(identity[TOP_KINASE_COLUMN][position]),
        TOP_SCORE_COLUMN: float(identity[TOP_SCORE_COLUMN][position]),
    }


def _empty_expanded_table() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_EXPANDED_COLUMNS)).astype(_EXPANDED_DTYPES)
