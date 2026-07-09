"""Module x kinase table -- percent shares per module.

Two policies for building the table:

- **cutoff_binary** (default): for each ``(module, kinase)`` cell, count
  the number of **distinct proteins** in the module that have at least
  one substrate of the kinase in ``kinase_substrates``.  Then rescale
  each row so it sums to 100 (percent shares).
- **weighted_top**: use the site-level ``top_kinase_weights`` (fractional
  weights when a site's top kinase is a tie); per protein take the max
  weight; per module sum across proteins; rescale to percent shares.

The ``cutoff_binary`` output is what PhosPy ships as the primary
signalome module table.

Clean-room re-implementation of PhosPy's
``science.signalomes.modules.build_signalome_module_table`` (MIT; PhosPy
GPL-3.0 used only as numerical oracle).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from alphaphos.signalome.constants import (
    KINASE_COLUMN,
    MODULE_ID_COLUMN,
    PROTEIN_COLUMN,
    TOP_KINASE_WEIGHTS_COLUMN,
)

_POLICY_CUTOFF_BINARY = "cutoff_binary"
_POLICY_WEIGHTED_TOP = "weighted_top"
_POLICIES = (_POLICY_CUTOFF_BINARY, _POLICY_WEIGHTED_TOP)


def build_module_table(
    *,
    module_assignments: pd.DataFrame,
    kinase_substrates: Mapping[str, Sequence[str]],
    kinase_order: Sequence[str],
    assignment_policy: str = _POLICY_CUTOFF_BINARY,
) -> pd.DataFrame:
    """Build the module x kinase percent-share table.

    Parameters
    ----------
    module_assignments
        Output of :func:`alphaphos.signalome.build_module_assignments` --
        one row per site with at minimum ``module_id`` and ``protein``
        columns.  ``top_kinase_weights`` is additionally required when
        ``assignment_policy="weighted_top"``.
    kinase_substrates
        ``{kinase: [substrate_site_id, ...]}`` mapping.
    kinase_order
        Explicit kinase order for the output columns.  Ensures the table
        is comparable across runs and against PhosPy.
    assignment_policy
        ``"cutoff_binary"`` (default) or ``"weighted_top"``.

    Returns
    -------
    pd.DataFrame
        Rows = module_id (1..M), columns = kinase (in ``kinase_order``),
        cells = percent shares per module (rows sum to 100 for modules
        that have any support, else 0).  Rounded to 3 decimals to match
        PhosPy's output.
    """
    if assignment_policy not in _POLICIES:
        raise ValueError(f"assignment_policy must be one of {_POLICIES}, got {assignment_policy!r}")

    module_index = pd.Index(
        sorted(
            {
                int(v)
                for v in module_assignments[MODULE_ID_COLUMN].astype("int64").tolist()
                if int(v) > 0
            }
        ),
        name=MODULE_ID_COLUMN,
    )
    kinase_index = pd.Index([str(k) for k in kinase_order], name=KINASE_COLUMN)

    if module_index.empty or kinase_index.empty:
        return pd.DataFrame(0.0, index=module_index.copy(), columns=kinase_index.copy()).astype(
            float
        )

    protein_to_module = (
        module_assignments[[PROTEIN_COLUMN, MODULE_ID_COLUMN]]
        .drop_duplicates(subset=[PROTEIN_COLUMN])
        .set_index(PROTEIN_COLUMN)[MODULE_ID_COLUMN]
        .astype("int64")
    )
    protein_to_module = protein_to_module.loc[protein_to_module > 0]

    if assignment_policy == _POLICY_CUTOFF_BINARY:
        table = _build_cutoff_binary(
            module_assignments=module_assignments,
            kinase_substrates=kinase_substrates,
            module_index=module_index,
            kinase_index=kinase_index,
            protein_to_module=protein_to_module,
        )
    else:
        table = _build_weighted_top(
            module_assignments=module_assignments,
            module_index=module_index,
            kinase_index=kinase_index,
            protein_to_module=protein_to_module,
        )

    # Row-normalise to percent shares.  Rows with zero support stay all-zero.
    row_totals = table.sum(axis=1)
    non_zero = row_totals > 0.0
    if non_zero.any():
        table.loc[non_zero] = table.loc[non_zero].div(row_totals.loc[non_zero], axis=0) * 100.0
    return table.astype(float).round(3)


def _build_cutoff_binary(
    *,
    module_assignments: pd.DataFrame,
    kinase_substrates: Mapping[str, Sequence[str]],
    module_index: pd.Index,
    kinase_index: pd.Index,
    protein_to_module: pd.Series,
) -> pd.DataFrame:
    module_positions = {int(m): i for i, m in enumerate(module_index.tolist())}
    site_to_protein = module_assignments[PROTEIN_COLUMN].astype(str)
    site_to_protein.index = pd.Index(site_to_protein.index.astype(str))
    site_to_protein_lookup = site_to_protein.to_dict()
    protein_to_module_lookup = {str(p): int(m) for p, m in protein_to_module.items() if int(m) > 0}
    kinases = kinase_index.astype(str).tolist()
    counts = np.zeros((len(module_index), len(kinases)), dtype=float)
    for k_idx, kinase in enumerate(kinases):
        substrates = kinase_substrates.get(kinase, ())
        if not substrates:
            continue
        seen_proteins: set[str] = set()
        for site_id in substrates:
            protein_id = site_to_protein_lookup.get(str(site_id))
            if protein_id is None or protein_id in seen_proteins:
                continue
            seen_proteins.add(protein_id)
            module_id = protein_to_module_lookup.get(protein_id)
            if module_id is None:
                continue
            module_pos = module_positions.get(module_id)
            if module_pos is None:
                continue
            counts[module_pos, k_idx] += 1.0
    return pd.DataFrame(counts, index=module_index.copy(), columns=kinase_index.copy())


def _build_weighted_top(
    *,
    module_assignments: pd.DataFrame,
    module_index: pd.Index,
    kinase_index: pd.Index,
    protein_to_module: pd.Series,
) -> pd.DataFrame:
    if TOP_KINASE_WEIGHTS_COLUMN not in module_assignments.columns:
        raise ValueError(
            f"module_assignments missing '{TOP_KINASE_WEIGHTS_COLUMN}' column "
            "required for assignment_policy='weighted_top'"
        )
    protein_to_module_lookup = {str(p): int(m) for p, m in protein_to_module.items() if int(m) > 0}
    kinase_membership = {str(k) for k in kinase_index.tolist()}
    rows: list[tuple[int, str, str, float]] = []
    site_index = module_assignments.index.astype(str)
    proteins = module_assignments[PROTEIN_COLUMN].astype(str).tolist()
    weights_lists = module_assignments[TOP_KINASE_WEIGHTS_COLUMN].tolist()
    for site_id, protein, weights in zip(site_index, proteins, weights_lists, strict=True):
        module_id = protein_to_module_lookup.get(protein)
        if module_id is None:
            continue
        for kinase, weight in _normalize_weights(weights, site_id=str(site_id)):
            if kinase not in kinase_membership:
                continue
            rows.append((module_id, kinase, protein, float(weight)))
    if not rows:
        return pd.DataFrame(0.0, index=module_index.copy(), columns=kinase_index.copy()).astype(
            float
        )
    df = pd.DataFrame(
        rows, columns=[MODULE_ID_COLUMN, KINASE_COLUMN, PROTEIN_COLUMN, "weight"]
    ).astype(
        {
            MODULE_ID_COLUMN: "int64",
            KINASE_COLUMN: str,
            PROTEIN_COLUMN: str,
            "weight": float,
        }
    )
    per_protein = (
        df.groupby([MODULE_ID_COLUMN, KINASE_COLUMN, PROTEIN_COLUMN], sort=False)["weight"]
        .max()
        .reset_index()
    )
    module_hits = (
        per_protein.groupby([MODULE_ID_COLUMN, KINASE_COLUMN], sort=False)["weight"]
        .sum()
        .unstack(KINASE_COLUMN, fill_value=0.0)
    )
    return module_hits.reindex(
        index=module_index.copy(), columns=kinase_index.copy(), fill_value=0.0
    ).astype(float)


def _normalize_weights(
    value: object,
    *,
    site_id: str,
) -> tuple[tuple[str, float], ...]:
    if value is None:
        return ()
    if isinstance(value, dict):
        pairs = tuple((str(k), float(w)) for k, w in value.items())
    elif isinstance(value, (tuple, list)):
        pairs_out: list[tuple[str, float]] = []
        for item in value:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(
                    f"top_kinase_weights entries must be (kinase, weight) pairs "
                    f"at site_id={site_id!r}"
                )
            try:
                pairs_out.append((str(item[0]), float(item[1])))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"top_kinase_weights entries must have float-compatible weights "
                    f"at site_id={site_id!r}"
                ) from exc
        pairs = tuple(pairs_out)
    else:
        raise ValueError(
            f"top_kinase_weights entries must be dicts or (kinase, weight) "
            f"sequences at site_id={site_id!r}"
        )
    return tuple((k, w) for k, w in pairs if w > 0.0)
