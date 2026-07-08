"""Normalize phospho intensities by the parent-protein intensity.

The killer use case for pairing a phospho AnnData with a proteome AnnData:
a change in per-site phospho signal can just reflect more protein.  Dividing
(subtracting in log2) removes that confounder, yielding a per-(site, sample)
"log fraction phosphorylated" quantity.

Sample pairing is by trailing DVP well-ID regex (``_A1`` .. ``_G11``) since
phospho runs and proteome runs are named differently but originate from the
same well of the same plate.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Matches the DVP well-ID suffix.  Restricted to the standard 96-well plate
# layout (A-H, 1-12); accepts a trailing extra tag (e.g. Spectronaut re-run
# timestamp ``_20260706070624``) after the well ID.
_WELL_ID_RE = re.compile(r"_([A-H])(1[0-2]|[1-9])(?:_[^_]+)?$")

ProteinGroupPolicy = Literal["first", "best_q"]
MissingProteinPolicy = Literal["drop", "carry", "fail"]


def phospho_over_proteome(
    adata_phos: ad.AnnData,
    adata_prot: ad.AnnData,
    *,
    sample_pairing: dict[str, str] | Literal["auto"] = "auto",
    missing_protein: MissingProteinPolicy = "drop",
    protein_group_policy: ProteinGroupPolicy = "first",
    phospho_layer: str = "intensity_log2",
    proteome_layer: str = "intensity_log2",
) -> ad.AnnData:
    """Return phospho AnnData with intensities divided by protein abundance.

    In log2 space, division is subtraction::

        adata_norm.X[i, j] = adata_phos.X[phos_sample, site]
                           - adata_prot.X[matched_prot_sample, protein_of(site)]

    Parameters
    ----------
    adata_phos
        Phospho AnnData (samples × sites).  ``.var_names`` must parse as
        the alphaPhos ``Protein|Gene|Site|Mult`` full-key format (this is
        what :func:`alphaphos.collapse_sites` produces).
    adata_prot
        Proteome AnnData (samples × protein groups), as returned by
        :func:`read_spectronaut_short` or
        :func:`read_spectronaut_long` + :func:`collapse_proteome`.
    sample_pairing
        - ``"auto"`` (default): extract the DVP well ID (``_A1``..``_G11``)
          from both AnnData's ``.obs_names`` and pair by well.  Wells that
          don't appear in both are unpaired and dropped from the output.
        - ``dict[str, str]``: explicit ``{phos_sample: prot_sample}``
          mapping.  Phospho samples not in the mapping are dropped.
    missing_protein
        Behaviour when a site's parent protein is not quantified in the
        matched proteome sample:

        - ``"drop"`` (default): that cell becomes NaN.  Downstream imputer
          then handles it as MAR / MNAR.
        - ``"carry"``: keep the raw phospho log2 value for those cells; a
          ``.layers["was_normalized"]`` bool matrix records which cells
          are un-normalized.  Downstream DE sees mixed semantics -- flag
          this in the manuscript methods.
        - ``"fail"``: raise if any (site, sample) has phospho quant but
          no parent-protein quant.  Most conservative.
    protein_group_policy
        When a site's protein-group key contains multiple accessions
        (``P1;P2;P3|GENE|S100|M1``), which proteome PG to normalize by:

        - ``"first"`` (default): first accession that matches (P1 → its PG).
        - ``"best_q"``: accession whose PG has the fewest NaN across all
          proteome samples (proxy for best-quantified).
    phospho_layer, proteome_layer
        Which layers to read from.  Default ``"intensity_log2"``.

    Returns
    -------
    AnnData
        Same feature axis as ``adata_phos`` but with samples restricted to
        the paired subset.  ``.X`` and ``.layers["intensity_log2"]`` carry
        the normalized values.  ``.var`` gains:

        - ``matched_pg`` (str or NaN): proteome PG name used for this site.
        - ``matched_pg_hit`` (bool): whether a match was found.

        ``.layers["was_normalized"]`` is a bool matrix of shape (n_samples,
        n_sites): True where the cell was normalized (protein was
        quantified), False where the raw phospho carries through or was
        dropped to NaN.

        ``.uns["alphaphos_proteome_normalization"]`` records:
        ``pairing_used`` (dict), ``n_paired_samples``,
        ``n_sites_matched_pg`` / ``n_sites_no_pg_match``,
        ``policy_missing_protein``, ``policy_protein_group``,
        ``phospho_source_shape``, ``proteome_source_shape``.
    """
    # 1. Sample pairing.
    if sample_pairing == "auto":
        pairing = _pair_by_well(adata_phos.obs_names, adata_prot.obs_names)
    else:
        pairing = dict(sample_pairing)
    if not pairing:
        raise ValueError(
            "phospho_over_proteome: no phospho ↔ proteome sample pairs resolved. "
            "For sample_pairing='auto', expected DVP well-ID suffixes "
            "('_A1'..'_G11') in both .obs_names.  Otherwise pass an explicit "
            "sample_pairing={phos: prot} dict."
        )
    logger.info(
        "phospho_over_proteome: paired %d/%d phospho samples with proteome samples.",
        len(pairing),
        adata_phos.n_obs,
    )

    # Restrict phospho to paired samples, in pairing insertion order.
    phos_samples = list(pairing.keys())
    prot_samples = [pairing[s] for s in phos_samples]
    phos = adata_phos[phos_samples, :].copy()
    prot = adata_prot[prot_samples, :]

    # 2. Resolve which proteome PG each site normalizes by.
    pg_index, col_to_name = _build_pg_lookup(prot.var_names.astype(str))
    prot_X = np.asarray(prot.layers.get(proteome_layer, prot.X), dtype=np.float64)
    site_to_pg, pg_col_idx = _resolve_site_to_pg(
        phos.var_names.astype(str),
        pg_index=pg_index,
        col_to_name=col_to_name,
        policy=protein_group_policy,
        prot_X=prot_X,
    )
    hit_mask = pg_col_idx >= 0
    n_sites_hit = int(hit_mask.sum())
    n_sites_miss = int((~hit_mask).sum())
    logger.info(
        "phospho_over_proteome: %d/%d sites matched a proteome PG (policy=%s).",
        n_sites_hit,
        phos.n_vars,
        protein_group_policy,
    )

    # 3. Build the aligned proteome matrix (n_paired_samples, n_sites).
    aligned_prot = np.full((len(phos_samples), phos.n_vars), np.nan, dtype=np.float64)
    hit_j = np.where(pg_col_idx >= 0)[0]
    if hit_j.size:
        aligned_prot[:, hit_j] = prot_X[:, pg_col_idx[hit_j]]

    # 4. Compute the normalized values (log2 subtraction).
    phos_X = np.asarray(phos.layers.get(phospho_layer, phos.X), dtype=np.float64)
    norm_X = phos_X - aligned_prot  # NaN where either input is NaN
    was_normalized = ~np.isnan(norm_X)

    # 5. Apply missing_protein policy.
    orphan_cells = np.isnan(norm_X) & ~np.isnan(phos_X)
    n_orphan = int(orphan_cells.sum())
    if missing_protein == "fail" and n_orphan:
        raise ValueError(
            f"phospho_over_proteome: {n_orphan:,} (site, sample) cells have "
            "phospho quant but no matched protein quant "
            "(missing_protein='fail'). Switch to 'drop' or 'carry'."
        )
    if missing_protein == "carry":
        norm_X = np.where(orphan_cells, phos_X, norm_X)

    # 6. Assemble output AnnData.
    out = phos.copy()
    out.X = norm_X
    out.layers["intensity_log2"] = norm_X.copy()
    out.layers["was_normalized"] = was_normalized
    out.var["matched_pg"] = pd.Series(site_to_pg, index=out.var_names, dtype="object")
    out.var["matched_pg_hit"] = hit_mask

    # h5ad doesn't serialize tuples -- store shapes as lists so
    # ``adata.write_h5ad`` round-trips cleanly.
    out.uns["alphaphos_proteome_normalization"] = {
        "pairing_used": pairing,
        "n_paired_samples": len(pairing),
        "n_sites_matched_pg": n_sites_hit,
        "n_sites_no_pg_match": n_sites_miss,
        "n_orphan_cells_pre_policy": n_orphan,
        "policy_missing_protein": missing_protein,
        "policy_protein_group": protein_group_policy,
        "phospho_source_shape": list(adata_phos.shape),
        "proteome_source_shape": list(adata_prot.shape),
    }
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pair_by_well(phos_names: pd.Index, prot_names: pd.Index) -> dict[str, str]:
    """Return ``{phos_sample: prot_sample}`` via trailing DVP well-ID match."""

    def _extract(name: str) -> str | None:
        m = _WELL_ID_RE.search(str(name))
        return f"{m.group(1)}{m.group(2)}" if m else None

    phos_well = {_extract(n): n for n in phos_names if _extract(n) is not None}
    prot_well = {_extract(n): n for n in prot_names if _extract(n) is not None}
    if not phos_well:
        raise ValueError(
            "phospho_over_proteome: sample_pairing='auto' could not extract any "
            "DVP well IDs from the phospho .obs_names.  Sample names must end "
            "with '_<A-H><1-12>' (optionally followed by a re-run tag).  Pass "
            "an explicit sample_pairing dict if your naming differs."
        )
    if not prot_well:
        raise ValueError(
            "phospho_over_proteome: sample_pairing='auto' could not extract any "
            "DVP well IDs from the proteome .obs_names."
        )
    shared = set(phos_well) & set(prot_well)
    dropped_phos = set(phos_well) - shared
    dropped_prot = set(prot_well) - shared
    if dropped_phos:
        logger.info(
            "phospho_over_proteome: %d phospho wells have no proteome match: %s",
            len(dropped_phos),
            sorted(dropped_phos)[:5],
        )
    if dropped_prot:
        logger.info(
            "phospho_over_proteome: %d proteome wells have no phospho match: %s",
            len(dropped_prot),
            sorted(dropped_prot)[:5],
        )
    # Preserve phos ordering.
    return {phos_well[w]: prot_well[w] for w in phos_well if w in shared}


def _build_pg_lookup(
    pg_names: pd.Index,
) -> tuple[dict[str, int], dict[int, str]]:
    """Return ``({accession_or_full: column_index}, {column_index: canonical_name})``.

    A proteome PG stored as ``"P1;P2;P3"`` becomes reachable by any of
    ``"P1"``, ``"P2"``, ``"P3"``, or the full string.  Later entries win
    on collision.  ``col_to_name`` maps back to the canonical proteome
    ``.var_name`` for each column.
    """
    lookup: dict[str, int] = {}
    col_to_name: dict[int, str] = {}
    for j, name in enumerate(pg_names):
        name_s = str(name)
        col_to_name[j] = name_s
        lookup[name_s] = j
        for acc in name_s.split(";"):
            acc = acc.strip()
            if acc:
                lookup.setdefault(acc, j)
    return lookup, col_to_name


def _resolve_site_to_pg(
    site_keys: pd.Index,
    *,
    pg_index: dict[str, int],
    col_to_name: dict[int, str],
    policy: ProteinGroupPolicy,
    prot_X: np.ndarray,
) -> tuple[list[str | None], np.ndarray]:
    """Map each site → (proteome-PG canonical name, proteome column index).

    Returns (list of PG names (or None), ndarray of column indices where
    -1 means unmatched).  For ``policy="best_q"``, picks the accession
    whose column has the fewest missing samples.
    """
    n_sites = len(site_keys)
    site_to_pg: list[str | None] = [None] * n_sites
    col_idx = np.full(n_sites, -1, dtype=np.int64)
    n_missing_per_col = np.isnan(prot_X).sum(axis=0)

    for i, key in enumerate(site_keys):
        # phospho keys look like "P1;P2|GENE|S100|M1"; take the PG component.
        pg_component = str(key).split("|", 1)[0]
        accs = [a.strip() for a in pg_component.split(";") if a.strip()]
        if not accs:
            continue

        # Candidate columns (deduped, preserving first-hit order).
        seen_cols: set[int] = set()
        candidates: list[int] = []
        for acc in accs:
            col = pg_index.get(acc)
            if col is not None and col not in seen_cols:
                candidates.append(col)
                seen_cols.add(col)

        if not candidates:
            continue

        chosen_col = (
            candidates[0]
            if policy == "first" or len(candidates) == 1
            else min(candidates, key=lambda c: n_missing_per_col[c])
        )
        col_idx[i] = chosen_col
        site_to_pg[i] = col_to_name[chosen_col]
    return site_to_pg, col_idx
