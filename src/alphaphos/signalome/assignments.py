"""Per-site module assignments + top-kinase attribution.

For each site, this module resolves:

1. **module_id** — the site's protein's module (from :func:`derive_protein_modules`).
2. **top_kinase** — the highest-scoring kinase in the site's row of the
   prediction matrix, with lexicographic tie-breaking on equal scores.
3. **Site-level tie diagnostics** — candidates + tie count + ambiguity flag.
4. **Module-level top-kinase** — the majority-vote top-kinase across all
   sites in the module (same tie-breaking policy).

Clean-room re-implementation of PhosPy's ``science.signalomes.assignments``
(MIT-compatible; PhosPy is GPL-3.0 and is used only as a numerical oracle
in parity tests).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaphos.signalome.constants import (
    DISPLAY_ID_COLUMN,
    GENE_SYMBOL_COLUMN,
    ISOFORM_ID_COLUMN,
    LEXICOGRAPHIC_TIE_BREAK_POLICY,
    MODULE_ID_COLUMN,
    MODULE_TOP_KINASE_CANDIDATES_COLUMN,
    MODULE_TOP_KINASE_COLUMN,
    MODULE_TOP_KINASE_IS_AMBIGUOUS_COLUMN,
    MODULE_TOP_KINASE_SELECTION_POLICY_COLUMN,
    MODULE_TOP_KINASE_TIE_COUNT_COLUMN,
    NO_SUPPORT_SELECTION_POLICY,
    PROTEIN_ACCESSION_COLUMN,
    PROTEIN_COLUMN,
    SITE_COLUMN,
    SITE_KEY_COLUMN,
    TOP_KINASE_CANDIDATES_COLUMN,
    TOP_KINASE_COLUMN,
    TOP_KINASE_IS_AMBIGUOUS_COLUMN,
    TOP_KINASE_SELECTION_POLICY_COLUMN,
    TOP_KINASE_TIE_COUNT_COLUMN,
    TOP_KINASE_WEIGHTS_COLUMN,
    TOP_SCORE_COLUMN,
    UNSUPPORTED_KINASE,
)


def build_module_assignments(
    *,
    prediction_matrix: pd.DataFrame,
    site_to_protein: pd.Series,
    protein_modules: pd.Series,
    site_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Build the site-level signalome assignments DataFrame.

    Parameters
    ----------
    prediction_matrix
        (n_sites, n_kinases).  Cell = per-site kinase prediction score.
    site_to_protein
        pd.Series indexed by site_key, values are protein IDs.
    protein_modules
        pd.Series indexed by protein, values are module IDs.  Typically the
        output of :func:`alphaphos.signalome.derive_protein_modules` after
        clustering.
    site_metadata
        DataFrame indexed by site_key with columns ``site_key``,
        ``display_id``, ``gene_symbol``, ``site``, ``protein_accession``,
        ``isoform_id``.  Build via
        :func:`alphaphos.signalome.extract_site_metadata` or supply your
        own.

    Returns
    -------
    pd.DataFrame
        One row per site, indexed by site_key, with the following columns::

            site_key, display_id, gene_symbol, site, protein,
            protein_accession, isoform_id, module_id,
            top_kinase, top_score,
            top_kinase_candidates, top_kinase_weights,
            top_kinase_tie_count, top_kinase_is_ambiguous,
            top_kinase_selection_policy,
            module_top_kinase, module_top_kinase_candidates,
            module_top_kinase_tie_count, module_top_kinase_is_ambiguous,
            module_top_kinase_selection_policy.
    """
    if prediction_matrix.shape[1] == 0:
        raise ValueError("prediction_matrix must contain at least one kinase column")
    site_index = _validate_unique_string_index(prediction_matrix.index, context="prediction_matrix")
    aligned_site_to_protein = _align_site_to_protein(
        site_index=site_index, site_to_protein=site_to_protein
    )
    identity = _align_site_identity(
        site_index=site_index,
        aligned_site_to_protein=aligned_site_to_protein,
        site_metadata=site_metadata,
    )
    (
        top_kinases,
        top_scores,
        top_kinase_candidates,
        top_kinase_weights,
        top_kinase_tie_counts,
        top_kinase_policies,
    ) = _resolve_top_kinase_per_site(prediction_matrix=prediction_matrix)

    # Assign module_id to each site via its protein
    normalized_protein_modules = _normalize_protein_modules(protein_modules)
    module_ids_per_site = (
        aligned_site_to_protein.map(normalized_protein_modules).fillna(0).astype("int64")
    )

    # Module-level top-kinase attribution (majority vote of site top_kinases)
    site_level = pd.DataFrame(
        {
            PROTEIN_COLUMN: aligned_site_to_protein.to_numpy(),
            TOP_KINASE_COLUMN: top_kinases,
            MODULE_ID_COLUMN: module_ids_per_site.to_numpy(),
        },
        index=site_index.copy(),
    )
    module_resolution = _derive_module_top_kinase(site_level)

    # Broadcast module-level attributions back to per-site rows.
    site_module_ids = site_level[MODULE_ID_COLUMN].astype("int64")
    module_view = module_resolution.reindex(site_module_ids.tolist()).reset_index(drop=True)
    module_view.index = site_index.copy()

    result = pd.DataFrame(
        {
            SITE_KEY_COLUMN: identity[SITE_KEY_COLUMN].to_numpy(),
            DISPLAY_ID_COLUMN: identity[DISPLAY_ID_COLUMN].to_numpy(),
            GENE_SYMBOL_COLUMN: identity[GENE_SYMBOL_COLUMN].to_numpy(),
            SITE_COLUMN: identity[SITE_COLUMN].to_numpy(),
            PROTEIN_COLUMN: aligned_site_to_protein.to_numpy(),
            PROTEIN_ACCESSION_COLUMN: identity[PROTEIN_ACCESSION_COLUMN].to_numpy(),
            ISOFORM_ID_COLUMN: identity[ISOFORM_ID_COLUMN].to_numpy(),
            MODULE_ID_COLUMN: module_ids_per_site.to_numpy(dtype=np.int64),
            TOP_KINASE_COLUMN: top_kinases,
            TOP_SCORE_COLUMN: top_scores,
            TOP_KINASE_CANDIDATES_COLUMN: top_kinase_candidates,
            TOP_KINASE_WEIGHTS_COLUMN: top_kinase_weights,
            TOP_KINASE_TIE_COUNT_COLUMN: top_kinase_tie_counts,
            TOP_KINASE_IS_AMBIGUOUS_COLUMN: [c > 1 for c in top_kinase_tie_counts],
            TOP_KINASE_SELECTION_POLICY_COLUMN: top_kinase_policies,
            MODULE_TOP_KINASE_COLUMN: module_view[MODULE_TOP_KINASE_COLUMN].to_numpy(),
            MODULE_TOP_KINASE_CANDIDATES_COLUMN: module_view[
                MODULE_TOP_KINASE_CANDIDATES_COLUMN
            ].to_numpy(),
            MODULE_TOP_KINASE_TIE_COUNT_COLUMN: module_view[
                MODULE_TOP_KINASE_TIE_COUNT_COLUMN
            ].to_numpy(dtype=np.int64),
            MODULE_TOP_KINASE_IS_AMBIGUOUS_COLUMN: module_view[
                MODULE_TOP_KINASE_IS_AMBIGUOUS_COLUMN
            ].to_numpy(),
            MODULE_TOP_KINASE_SELECTION_POLICY_COLUMN: module_view[
                MODULE_TOP_KINASE_SELECTION_POLICY_COLUMN
            ].to_numpy(),
        },
        index=site_index.copy(),
    )
    return result.astype(
        {
            SITE_KEY_COLUMN: str,
            DISPLAY_ID_COLUMN: str,
            GENE_SYMBOL_COLUMN: str,
            SITE_COLUMN: str,
            PROTEIN_COLUMN: str,
            PROTEIN_ACCESSION_COLUMN: str,
            ISOFORM_ID_COLUMN: str,
            MODULE_ID_COLUMN: "int64",
            TOP_KINASE_COLUMN: str,
            TOP_SCORE_COLUMN: float,
            TOP_KINASE_TIE_COUNT_COLUMN: "int64",
            TOP_KINASE_IS_AMBIGUOUS_COLUMN: bool,
            TOP_KINASE_SELECTION_POLICY_COLUMN: str,
            MODULE_TOP_KINASE_COLUMN: str,
            MODULE_TOP_KINASE_TIE_COUNT_COLUMN: "int64",
            MODULE_TOP_KINASE_IS_AMBIGUOUS_COLUMN: bool,
            MODULE_TOP_KINASE_SELECTION_POLICY_COLUMN: str,
        }
    )


def select_kinase_substrates(
    *,
    prediction_matrix: pd.DataFrame,
    cutoff: float,
) -> dict[str, tuple[str, ...]]:
    """For each kinase, list the sites whose prediction score exceeds ``cutoff``."""
    site_ids = prediction_matrix.index.astype(str).to_numpy()
    kinase_names = prediction_matrix.columns.astype(str).to_numpy()
    mask = prediction_matrix.to_numpy(dtype=float) > float(cutoff)
    return {
        str(kinase): tuple(site_ids[mask[:, j]].tolist()) for j, kinase in enumerate(kinase_names)
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_unique_string_index(index: pd.Index, *, context: str) -> pd.Index:
    resolved = pd.Index(index.astype(str))
    if resolved.has_duplicates:
        dups = sorted({str(v) for v in resolved[resolved.duplicated(keep=False)]})
        preview = ", ".join(dups[:3])
        suffix = "..." if len(dups) > 3 else ""
        raise ValueError(f"{context} has duplicate site identifiers: {preview}{suffix}")
    return resolved


def _align_site_to_protein(
    *,
    site_index: pd.Index,
    site_to_protein: pd.Series,
) -> pd.Series:
    resolved = site_to_protein.copy()
    resolved.index = pd.Index(resolved.index.astype(str))
    missing = [s for s in site_index if s not in resolved.index]
    if missing:
        preview = ", ".join(missing[:3])
        suffix = "..." if len(missing) > 3 else ""
        raise ValueError(
            f"site_to_protein missing sites present in prediction_matrix: {preview}{suffix}"
        )
    aligned = resolved.loc[site_index].astype(str).str.strip()
    if (aligned == "").any():
        raise ValueError("site_to_protein contains empty protein identifiers")
    aligned.index = site_index.copy()
    aligned.name = PROTEIN_COLUMN
    return aligned


def _align_site_identity(
    *,
    site_index: pd.Index,
    aligned_site_to_protein: pd.Series,
    site_metadata: pd.DataFrame,
) -> pd.DataFrame:
    required = (SITE_KEY_COLUMN, DISPLAY_ID_COLUMN, GENE_SYMBOL_COLUMN, SITE_COLUMN)
    missing_cols = [c for c in required if c not in site_metadata.columns]
    if missing_cols:
        raise ValueError(f"site_metadata missing required columns: {', '.join(missing_cols)}")
    metadata = site_metadata.copy(deep=False)
    metadata.index = pd.Index(metadata.index.astype(str))
    missing_sites = [s for s in site_index if s not in metadata.index]
    if missing_sites:
        preview = ", ".join(missing_sites[:3])
        suffix = "..." if len(missing_sites) > 3 else ""
        raise ValueError(
            f"site_metadata missing sites present in prediction_matrix: {preview}{suffix}"
        )
    aligned = metadata.loc[site_index].copy()
    for col in (SITE_KEY_COLUMN, DISPLAY_ID_COLUMN, GENE_SYMBOL_COLUMN, SITE_COLUMN):
        aligned[col] = aligned[col].fillna("").astype(str).str.strip()
    for col in (PROTEIN_ACCESSION_COLUMN, ISOFORM_ID_COLUMN):
        if col in aligned.columns:
            aligned[col] = aligned[col].fillna("").astype(str).str.strip()
        else:
            aligned[col] = ""
    # site_key must match the row index -- guard the caller
    if aligned[SITE_KEY_COLUMN].tolist() != site_index.astype(str).tolist():
        raise ValueError("site_metadata.site_key does not align with prediction_matrix.index")
    return aligned


def _resolve_top_kinase_per_site(
    *,
    prediction_matrix: pd.DataFrame,
) -> tuple[
    list[str],
    np.ndarray,
    list[tuple[str, ...]],
    list[tuple[tuple[str, float], ...]],
    list[int],
    list[str],
]:
    """Vectorized max-per-row + lex tie-break resolution."""
    sorted_columns = sorted(str(k) for k in prediction_matrix.columns)
    sorted_matrix = prediction_matrix.loc[:, sorted_columns].astype(float)
    values = sorted_matrix.to_numpy(dtype=float)
    kinase_names = np.asarray(sorted_columns, dtype=object)

    row_max = values.max(axis=1)
    tie_mask = values == row_max[:, None]
    tie_counts = tie_mask.sum(axis=1).astype(int)

    top_kinases: list[str] = []
    top_kinase_candidates: list[tuple[str, ...]] = []
    top_kinase_weights: list[tuple[tuple[str, float], ...]] = []
    top_kinase_policies: list[str] = []

    for row_mask, tie_count in zip(tie_mask, tie_counts, strict=True):
        candidates = tuple(str(k) for k in kinase_names[row_mask])
        top_kinase_candidates.append(candidates)
        if tie_count == 0:
            top_kinase_weights.append(())
            top_kinases.append(UNSUPPORTED_KINASE)
            top_kinase_policies.append(NO_SUPPORT_SELECTION_POLICY)
            continue
        weight = 1.0 / float(tie_count)
        top_kinase_weights.append(tuple((k, weight) for k in candidates))
        # sorted_columns was already lex-sorted, so candidates are in lex
        # order.  First element wins ties.
        top_kinases.append(candidates[0])
        top_kinase_policies.append(LEXICOGRAPHIC_TIE_BREAK_POLICY)

    return (
        top_kinases,
        row_max,
        top_kinase_candidates,
        top_kinase_weights,
        list(tie_counts.astype(int).tolist()),
        top_kinase_policies,
    )


def _normalize_protein_modules(protein_modules: pd.Series) -> pd.Series:
    resolved = protein_modules.copy()
    resolved.index = pd.Index(resolved.index.astype(str))
    if resolved.index.has_duplicates:
        dups = sorted({str(v) for v in resolved.index[resolved.index.duplicated()]})
        preview = ", ".join(dups[:3])
        suffix = "..." if len(dups) > 3 else ""
        raise ValueError(f"protein_modules has duplicate protein identifiers: {preview}{suffix}")
    numeric = pd.to_numeric(resolved, errors="coerce")
    if numeric.isna().any():
        raise ValueError("protein_modules must contain integer module IDs")
    rounded = np.floor(numeric.to_numpy(dtype=float))
    if not np.allclose(numeric.to_numpy(dtype=float), rounded):
        raise ValueError("protein_modules must contain integer module IDs")
    integer = numeric.astype("int64")
    integer.loc[integer < 0] = 0
    return integer.astype("int64")


def _derive_module_top_kinase(site_level: pd.DataFrame) -> pd.DataFrame:
    """Per-module majority-vote of top_kinase across the module's sites."""
    module_ids = sorted({int(v) for v in site_level[MODULE_ID_COLUMN].astype("int64").tolist()})
    if 0 not in module_ids:
        module_ids = [0, *module_ids]
    rows: list[dict[str, object]] = []
    for module_id in module_ids:
        group = site_level.loc[site_level[MODULE_ID_COLUMN].astype("int64") == int(module_id)]
        supported = group.loc[group[TOP_KINASE_COLUMN].astype(str) != UNSUPPORTED_KINASE]
        counts = supported[TOP_KINASE_COLUMN].astype(str).value_counts()
        if not counts.empty and int(module_id) > 0:
            max_count = int(counts.iloc[0])
            tied = tuple(sorted(k for k in counts[counts == max_count].index.tolist()))
            dominant = tied[0]
            policy = LEXICOGRAPHIC_TIE_BREAK_POLICY
        else:
            tied = ()
            dominant = UNSUPPORTED_KINASE
            policy = NO_SUPPORT_SELECTION_POLICY
        rows.append(
            {
                MODULE_ID_COLUMN: int(module_id),
                MODULE_TOP_KINASE_COLUMN: dominant,
                MODULE_TOP_KINASE_CANDIDATES_COLUMN: tied,
                MODULE_TOP_KINASE_TIE_COUNT_COLUMN: len(tied),
                MODULE_TOP_KINASE_IS_AMBIGUOUS_COLUMN: len(tied) > 1,
                MODULE_TOP_KINASE_SELECTION_POLICY_COLUMN: policy,
            }
        )
    result = pd.DataFrame(rows).set_index(MODULE_ID_COLUMN)
    result.index = pd.Index(result.index.astype("int64"), name=MODULE_ID_COLUMN)
    return result
