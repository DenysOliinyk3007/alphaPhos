"""Site-level collapse pipeline: PSM DataFrame -> (quant, loc, metadata).

This is where the multi-stage collapse happens, glued together as a sequence
of pure functions. Each stage's job is small and its input/output is a
DataFrame or (DataFrame, DataFrame) so it can be tested in isolation with a
minimal fixture.

Pipeline overview (per stage docstrings below have full detail):

    PSM DataFrame  (rows = precursor observations, one per (peptide, sample))
        |
        |  prepare_psms:   parse EG.PrecursorId, filter to phospho,
        |                  parse per-position loc probabilities
        v
    Prepared DataFrame
        |
        |  explode_to_sites:  one row per (peptide, phospho_position, sample).
        |                     Adds site_aa, UPD_seq, and the per-site loc prob.
        v
    Exploded DataFrame
        |
        |  build_precursor_pivots:  pivot quant + loc + metadata to
        |                            (peptide, position) x sample.
        v
    Wide precursor tables
        |
        |  compute_site_metadata:  compute absolute site positions,
        |                          multiplicity, and site keys (full,
        |                          short, pg). Deduplicate to one row
        |                          per site.
        v
    Precursor tables + site metadata
        |
        |  aggregate_precursors_to_sites:  group precursor rows by
        |                                  collapse key, apply the chosen
        |                                  aggregation method.
        v
    (sites x samples) quant matrix + (sites x samples) loc matrix + site metadata
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from alphaphos.preprocess._collapse.aggregation import aggregate_by_key
from alphaphos.preprocess._collapse.keys import (
    build_full_key,
    build_modified_sequence,
    build_pg_key,
    build_short_key,
    get_phospho_amino_acid,
    resolve_short_key_collisions,
)
from alphaphos.preprocess._collapse.parsing import (
    extract_first_valid_position,
    extract_sequence_modifications,
    parse_localization_probabilities,
)

_NULL_LOGGER = logging.getLogger("alphaphos.preprocess._collapse.site_pipeline")


# ---------------------------------------------------------------------------
# Stage 1: prepare -- parse PSMs into structured columns
# ---------------------------------------------------------------------------


def prepare_psms(psm_df: pd.DataFrame, *, logger: logging.Logger = _NULL_LOGGER) -> pd.DataFrame:
    """Parse ``EG.PrecursorId`` columns and filter to phospho-containing rows.

    Adds several derived columns and drops any row without a phospho marker
    (they can't contribute to site-level analysis).

    Parameters
    ----------
    psm_df : DataFrame
        Raw PSM-level input. Required columns::

            R.FileName
            EG.PrecursorId
            EG.TotalQuantity (Settings)
            PEP.PeptidePosition
            EG.PTMAssayProbability
            PG.Genes
            PG.ProteinGroups

        Optional::

            EG.PTMLocalizationProbabilities  -- enables per-site loc probs.

    Returns
    -------
    DataFrame
        A COPY (input is untouched) with additional columns::

            clean_sequence   -- amino acids only
            phospho_positions -- list[int], 1-indexed within clean_sequence
            phospho_count    -- int, number of phospho groups on the peptide
            all_modifications -- list[str], all bracket contents in order
            _loc_dict        -- dict[int, float] parsed per-position loc probs;
                                only present when EG.PTMLocalizationProbabilities
                                exists in the input.

    Notes
    -----
    * Underscores in gene names get replaced with ``#`` here (they're a
      historical alphaPhos convention that never survives to the output;
      restored to ``_`` at finalize time). This is a workaround for the
      old key delimiter ``_``; with the new ``|`` delimiter it's largely
      cosmetic but kept for parity with legacy pipelines.
    """
    df = psm_df.copy()
    logger.info("Prepare PSMs: %d rows in", len(df))

    # Sanity: check duplicate raw files (same sample name, same first-100 precursors).
    _log_duplicate_raw_files(df, logger)

    # Underscore-in-gene-name mitigation. Preserves legacy behavior; the
    # replacement is inverted at finalize time so users see ``_`` again.
    if "PG.Genes" in df.columns:
        underscore_count = df["PG.Genes"].astype(str).str.contains("_", na=False).sum()
        if underscore_count > 0:
            df["PG.Genes"] = df["PG.Genes"].astype(str).str.replace("_", "#", regex=False)
            logger.warning(
                "%d gene names contained underscores and have been temporarily "
                "replaced with '#' (restored at output).",
                underscore_count,
            )

    # Parse EG.PrecursorId into the modification columns.
    mods = df["EG.PrecursorId"].apply(extract_sequence_modifications)
    df["clean_sequence"] = [m["clean_sequence"] for m in mods]
    df["phospho_positions"] = [m["phospho_positions"] for m in mods]
    df["phospho_count"] = [m["phospho_count"] for m in mods]
    df["all_modifications"] = [m["all_modifications"] for m in mods]

    # Drop non-phospho rows (they can't contribute to site-level output).
    n_before = len(df)
    df = df[df["phospho_count"] > 0].copy()
    logger.info(
        "Phospho filter: %d -> %d rows (%d non-phospho dropped).",
        n_before,
        len(df),
        n_before - len(df),
    )

    # Per-position loc probabilities (optional: only if the column exists).
    if "EG.PTMLocalizationProbabilities" in df.columns:
        df["_loc_dict"] = df["EG.PTMLocalizationProbabilities"].apply(
            parse_localization_probabilities
        )

    return df


# ---------------------------------------------------------------------------
# Stage 2: explode -- one row per (peptide, phospho_position)
# ---------------------------------------------------------------------------


def explode_to_sites(
    prepared_df: pd.DataFrame,
    *,
    logger: logging.Logger = _NULL_LOGGER,
) -> pd.DataFrame:
    """Explode ``phospho_positions`` -> one row per (peptide, position).

    Each precursor row can carry multiple phospho residues (M1/M2/M3+); we
    split them out so each row represents ONE modified residue. Then we
    look up the per-site localization probability (either from the parsed
    ``_loc_dict`` set in :func:`prepare_psms`, or falling back to the joint
    ``EG.PTMAssayProbability`` when per-position parsing wasn't possible).

    Adds columns::

        PTM_group        -- alias of EG.PrecursorId (used as pivot index)
        PTM_0_pos_val    -- int, the phospho position within the peptide
        PTM_base_seq     -- alias of clean_sequence
        PTM_0_num        -- int, phospho_count (kept for multiplicity calc)
        UPD_seq          -- "sequence with * marker" visualization
        PTM_0_aa         -- amino acid at PTM_0_pos_val
        PTM_localization -- per-site loc prob (or joint fallback)

    Returns
    -------
    DataFrame
        Long-form: N rows per input row, one per phospho position on the
        peptide. Non-phospho input rows would have been dropped by
        :func:`prepare_psms` and don't appear here.
    """
    df = prepared_df.copy()

    # Alias the columns to their downstream names.
    df["PTM_group"] = df["EG.PrecursorId"]
    df["PTM_base_seq"] = df["clean_sequence"]
    df["PTM_0_pos_val"] = df["phospho_positions"]
    df["PTM_0_num"] = df["phospho_count"]

    df = df.explode("PTM_0_pos_val")
    # After explode, PTM_0_pos_val is object dtype; coerce to Python int for
    # downstream key building.
    df["PTM_0_pos_val"] = df["PTM_0_pos_val"].astype(int)

    df["UPD_seq"] = df.apply(
        lambda x: build_modified_sequence(x["PTM_base_seq"], x["PTM_0_pos_val"]),
        axis=1,
    )
    df["PTM_0_aa"] = df.apply(
        lambda x: get_phospho_amino_acid(x["PTM_base_seq"], x["PTM_0_pos_val"]),
        axis=1,
    )

    # Per-site localization: look up in the parsed dict; fall back to the
    # joint EG.PTMAssayProbability when per-position parsing wasn't possible
    # for this row (e.g. no EG.PTMLocalizationProbabilities column, or an
    # empty / unparseable string).
    if "_loc_dict" in df.columns:
        loc_series = pd.Series(
            [
                d.get(int(pos), np.nan) if d else np.nan
                for d, pos in zip(df["_loc_dict"], df["PTM_0_pos_val"], strict=True)
            ],
            index=df.index,
        )
        df["PTM_localization"] = loc_series
        fallback = df["PTM_localization"].isna()
        # Where per-site parsing produced NaN, use the joint prob as a stand-in.
        df.loc[fallback, "PTM_localization"] = df.loc[fallback, "EG.PTMAssayProbability"].astype(
            float
        )

        n_per_site = int((~fallback).sum())
        n_fallback = int(fallback.sum())
        pct = 100 * n_fallback / (n_per_site + n_fallback) if (n_per_site + n_fallback) else 0.0
        logger.info(
            "Per-site localization: %d rows with per-site prob, %d fell back to joint "
            "prob (%.1f%% fallback).",
            n_per_site,
            n_fallback,
            pct,
        )
        df = df.drop(columns=["_loc_dict"])
    else:
        df["PTM_localization"] = df["EG.PTMAssayProbability"].astype(float)
        logger.info(
            "No EG.PTMLocalizationProbabilities column; using joint "
            "EG.PTMAssayProbability for all rows."
        )

    logger.info("Exploded to %d (precursor, position) rows.", len(df))
    return df


# ---------------------------------------------------------------------------
# Stage 3: pivots -- quant and loc, at (precursor, position) granularity
# ---------------------------------------------------------------------------


def build_precursor_pivots(
    exploded_df: pd.DataFrame,
    *,
    logger: logging.Logger = _NULL_LOGGER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pivot the exploded long form to wide (precursor, position) x sample tables.

    Returns three parallel-indexed frames:

    1. ``quant_wide`` -- linear intensities. Rows = ``(PTM_group, PTM_0_pos_val)``.
       Columns = sample id (``R.FileName``). Zero-quants are converted to NaN.
    2. ``loc_wide`` -- per-(precursor, position) localization probability at
       run granularity. Same index as ``quant_wide``. Uses max across duplicate
       (precursor, position, run) cells (rare but possible).
    3. ``meta_wide`` -- first-observed values of the metadata columns
       (``PEP.PeptidePosition``, ``PG.ProteinGroups``, ``PG.Genes``,
       ``PTM_0_num``, ``PTM_0_aa``, ``UPD_seq``). One row per unique
       ``(PTM_group, PTM_0_pos_val)``.

    All three have the SAME index, which lets us join them later without
    reindexing.
    """
    quant_wide = exploded_df.pivot_table(
        index=["PTM_group", "PTM_0_pos_val"],
        columns="R.FileName",
        values="EG.TotalQuantity (Settings)",
        aggfunc="sum",
    ).replace(0, np.nan)

    logger.info(
        "Quant pivot: %d keys x %d samples (%d NaN cells).",
        quant_wide.shape[0],
        quant_wide.shape[1],
        int(quant_wide.isna().sum().sum()),
    )

    loc_wide = exploded_df.pivot_table(
        index=["PTM_group", "PTM_0_pos_val"],
        columns="R.FileName",
        values="PTM_localization",
        aggfunc="max",  # take strongest per-run evidence if duplicated
    )

    meta_keep = [
        "PEP.PeptidePosition",
        "PG.ProteinGroups",
        "PG.Genes",
        "PTM_0_num",
        "PTM_0_aa",
        "UPD_seq",
    ]
    meta_keep = [c for c in meta_keep if c in exploded_df.columns]
    meta_wide = exploded_df.pivot_table(
        index=["PTM_group", "PTM_0_pos_val"],
        values=meta_keep,
        aggfunc="first",
    )

    return quant_wide, loc_wide, meta_wide


# ---------------------------------------------------------------------------
# Stage 4: compute per-site metadata and canonical keys
# ---------------------------------------------------------------------------


def compute_site_metadata(
    meta_wide: pd.DataFrame,
    *,
    collapse_level: str = "PG",
    logger: logging.Logger = _NULL_LOGGER,
) -> pd.DataFrame:
    """Compute absolute site positions and canonical keys.

    Given the (precursor, position) metadata table, this stage:

    1. Extracts the peptide's start position in the parent protein (from
       ``PEP.PeptidePosition``) as a single integer.
    2. Computes the site's absolute position in the protein::

           absolute_position = peptide_start_position + intra_peptide_position - 1

    3. Clamps multiplicity (``PTM_0_num``) to ``3`` (the M1/M2/M3+ convention).
    4. Splits ``PG.ProteinGroups`` and ``PG.Genes`` at ``";"`` and keeps the
       first entry when ``collapse_level == "PG"``. When ``"P"``, keeps the
       full group string (protein-resolved output explodes later).
    5. Builds three keys per row::

           full_key   = "{ProteinGroup}|{Gene}|{aa}{position}|M{mult}"
           short_key  = "{Gene}|{aa}{position}|M{mult}" (may need collision fix)
           pg_key     = "{ProteinGroup}|{aa}{position}|M{mult}"

    Rows with any of the following are dropped::

        * unparseable PEP.PeptidePosition
        * PTM_0_aa == "X" (position out of range or non-STY residue)

    Returns
    -------
    DataFrame
        Indexed by ``(PTM_group, PTM_0_pos_val)`` (same as input).
        Adds columns: ``peptide_start``, ``absolute_position``,
        ``multiplicity``, ``protein_group_id``, ``gene``, ``full_key``,
        ``short_key``, ``pg_key``. Preserves ``UPD_seq``, ``PTM_0_aa``,
        ``PG.ProteinGroups``, ``PG.Genes``, ``PTM_0_num``.
    """
    meta = meta_wide.copy()

    if collapse_level not in ("PG", "P"):
        raise ValueError(f"collapse_level must be 'PG' or 'P', got {collapse_level!r}")

    # Parse PEP.PeptidePosition -> peptide_start (Int)
    meta["peptide_start"] = meta["PEP.PeptidePosition"].apply(extract_first_valid_position)

    # Drop rows without a valid peptide start (would produce garbage keys).
    n_before = len(meta)
    meta = meta.dropna(subset=["peptide_start"]).copy()
    meta["peptide_start"] = meta["peptide_start"].astype(np.int64)
    logger.info(
        "Peptide-position filter: %d -> %d sites (%d dropped for missing position).",
        n_before,
        len(meta),
        n_before - len(meta),
    )

    # Also drop rows where PTM_0_aa is 'X' (out-of-range or non-STY).
    n_before = len(meta)
    meta = meta[meta["PTM_0_aa"].isin(["S", "T", "Y"])].copy()
    if n_before - len(meta) > 0:
        logger.warning(
            "AA filter: dropped %d sites with non-STY residues in their position.",
            n_before - len(meta),
        )

    # Absolute site position in the parent protein (1-indexed).
    intra = meta.index.get_level_values("PTM_0_pos_val")
    meta["absolute_position"] = (meta["peptide_start"] + intra.astype(int) - 1).astype(np.int64)

    # Multiplicity clamped to 3 (M1/M2/M3+ convention).
    meta["multiplicity"] = meta["PTM_0_num"].astype(np.int64).clip(upper=3)

    # Protein group / gene canonicalization for the key.
    if collapse_level == "PG":
        meta["protein_group_id"] = meta["PG.ProteinGroups"].astype(str).str.split(";").str[0]
        meta["gene"] = meta["PG.Genes"].astype(str).str.split(";").str[0]
    else:  # "P" -- keep full string, downstream explodes it later
        meta["protein_group_id"] = meta["PG.ProteinGroups"].astype(str)
        meta["gene"] = meta["PG.Genes"].astype(str)

    # Build the three key flavors.
    meta["full_key"] = [
        build_full_key(pg, g, aa, p, m)
        for pg, g, aa, p, m in zip(
            meta["protein_group_id"],
            meta["gene"],
            meta["PTM_0_aa"],
            meta["absolute_position"],
            meta["multiplicity"],
            strict=True,
        )
    ]
    meta["short_key"] = [
        build_short_key(g, aa, p, m)
        for g, aa, p, m in zip(
            meta["gene"],
            meta["PTM_0_aa"],
            meta["absolute_position"],
            meta["multiplicity"],
            strict=True,
        )
    ]
    meta["pg_key"] = [
        build_pg_key(pg, aa, p, m)
        for pg, aa, p, m in zip(
            meta["protein_group_id"],
            meta["PTM_0_aa"],
            meta["absolute_position"],
            meta["multiplicity"],
            strict=True,
        )
    ]

    return meta


# ---------------------------------------------------------------------------
# Stage 5: aggregate to site level
# ---------------------------------------------------------------------------


def aggregate_precursors_to_sites(
    quant_wide: pd.DataFrame,
    loc_wide: pd.DataFrame,
    site_meta: pd.DataFrame,
    *,
    aggregation_method: str,
    logger: logging.Logger = _NULL_LOGGER,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group precursor-level rows by ``full_key`` and aggregate.

    Multiple precursors can map to the same site (different charges,
    different peptides spanning the same residue). We aggregate their
    per-run linear intensities via the requested method (see
    :mod:`._collapse.aggregation`).

    The localization matrix is aggregated by taking the ``max`` per-run
    (any evidence localizing the site in a run is sufficient).

    Parameters
    ----------
    quant_wide, loc_wide : DataFrame
        Precursor-level pivots from :func:`build_precursor_pivots`.
    site_meta : DataFrame
        Metadata from :func:`compute_site_metadata`. Indexed the same way
        as ``quant_wide``. We use its ``full_key`` column as the grouping
        variable.
    aggregation_method : str
        One of ``"sum"``, ``"median"``, ``"mean"``, ``"consolidate"``.

    Returns
    -------
    site_quant : DataFrame
        ``(n_sites x n_samples)`` linear intensities, indexed by ``full_key``.
    site_loc : DataFrame
        ``(n_sites x n_samples)`` localization probabilities, indexed by
        ``full_key``. Column order matches ``site_quant``.
    site_meta_dedup : DataFrame
        One row per ``full_key`` (first occurrence wins for tie-breakable
        metadata like ``PG.ProteinGroups``, ``PG.Genes``, ``UPD_seq``).
    """
    # Align: attach full_key to the precursor pivots via join on the shared
    # (PTM_group, PTM_0_pos_val) index.
    keys_by_index = site_meta["full_key"]
    quant_with_keys = quant_wide.join(keys_by_index, how="inner")
    loc_with_keys = loc_wide.join(keys_by_index, how="inner")

    # Sample columns are everything except full_key
    sample_cols = [c for c in quant_wide.columns]

    quant_indexed = quant_with_keys.set_index("full_key")[sample_cols]
    site_quant = aggregate_by_key(quant_indexed, aggregation_method, sample_cols)

    loc_indexed = loc_with_keys.set_index("full_key")[sample_cols]
    site_loc = loc_indexed.groupby(level=0).max()

    # Deduplicate metadata to one row per full_key.
    site_meta_dedup = site_meta.reset_index().drop_duplicates("full_key").set_index("full_key")

    logger.info(
        "Aggregated (%s) to %d sites x %d samples.",
        aggregation_method,
        len(site_quant),
        len(sample_cols),
    )
    return site_quant, site_loc, site_meta_dedup


# ---------------------------------------------------------------------------
# Stage 6: post-aggregate transforms (log2, noise floor)
# ---------------------------------------------------------------------------


def log2_transform(
    site_quant_linear: pd.DataFrame,
    *,
    apply_noise_floor: bool = True,
    logger: logging.Logger = _NULL_LOGGER,
) -> pd.DataFrame:
    """Take log2, optionally remove log2 ∈ {0, 1} as a noise floor.

    Log2 is applied AFTER aggregation so that values summed / consolidated
    across precursors are in the same linear scale. Zero-quants are
    converted to NaN before the log to avoid ``-inf``.

    The noise floor filter removes cells where log2 is exactly 0 or 1
    (linear intensity 1 or 2). These are Spectronaut's convention for
    "signal detected but at the very bottom of the dynamic range" and are
    usually treated as noise rather than real signal.

    Parameters
    ----------
    site_quant_linear : DataFrame
        Aggregated quant matrix in LINEAR intensity space.
    apply_noise_floor : bool
        If True, values in ``{0, 1}`` after log2 are set to NaN.

    Returns
    -------
    DataFrame
        Same shape as input, log2 intensities.
    """
    out = np.log2(site_quant_linear.replace(0, np.nan))

    neg_inf = int((out == -np.inf).sum().sum())
    if neg_inf > 0:
        logger.warning("log2 produced %d -inf values; replacing with NaN.", neg_inf)
        out = out.replace(-np.inf, np.nan)

    if apply_noise_floor:
        n_before = int(out.notna().sum().sum())
        out = out.replace(0, np.nan).replace(1, np.nan)
        n_after = int(out.notna().sum().sum())
        logger.info("Noise floor filter removed %d values.", n_before - n_after)

    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_duplicate_raw_files(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Warn if two ``R.FileName`` values share their first-100 precursors."""
    import hashlib

    file_hashes: dict[str, list[str]] = {}
    for fname in df["R.FileName"].unique():
        subset = df.loc[df["R.FileName"] == fname, "EG.PrecursorId"]
        sorted_precursors = sorted(subset.dropna().astype(str).tolist())[:100]
        h = hashlib.md5("||".join(sorted_precursors).encode()).hexdigest()
        file_hashes.setdefault(h, []).append(fname)
    for fnames in file_hashes.values():
        if len(fnames) > 1:
            logger.warning(
                "Potential duplicated run files: %s share the first-100 precursor pattern.",
                fnames,
            )


# ---------------------------------------------------------------------------
# Convenience: apply short-key collision resolution
# ---------------------------------------------------------------------------


def resolve_short_keys(site_meta: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Suffix duplicate ``short_key`` values so they're unique.

    Wraps :func:`._collapse.keys.resolve_short_key_collisions` to operate
    on the metadata DataFrame. Returns the modified meta table and the
    list of collision events (for logging into ``adata.uns``).
    """
    resolved, collisions = resolve_short_key_collisions(
        site_meta["short_key"].tolist(),
        site_meta.index.tolist()
        if site_meta.index.name == "full_key"
        else site_meta["full_key"].tolist(),
    )
    site_meta = site_meta.copy()
    site_meta["short_key"] = resolved
    return site_meta, collisions
