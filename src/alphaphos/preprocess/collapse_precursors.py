"""Precursor-level collapse for phospho DIA experiments.

Sibling of :mod:`alphaphos.preprocess.collapse`. Quantifies each identified
**precursor** (peptide sequence + charge + modifications) as one feature per
sample, without residue attribution and without localization-probability
masking. Trades site-level resolution for a more complete, less-punctured
quantification matrix -- valuable when localization is unreliable in the
dataset and the primary question is detection / differential rather than
per-site kinase inference.

Precedent: some phospho DIA workflows (Krug 2019 PTM-SEA, Meier 2020
diaPASEF phosphoproteomics guidance) advocate precursor-level readout when
the localization metric itself is noisy on low-abundance features. The
core practical rationale for this repo is documented in
``D:/Projects/miniBinders/CHO_TAB2_H2F/docs/peptide_level_collapse_rationale.md``
(TRKA activation loop pY680: 56/60 runs, 99.9% localization on Y680, still
culled by site-level masking because per-run loc was inconsistent).

Public API
----------
:func:`collapse_precursors`             -- entry point.
:func:`precursor_to_site_view`          -- annotate precursor .var with the
                                           best-guess protein-absolute site
                                           key so KSEA / pathway analyses can
                                           still be run downstream on the
                                           site-level parallel object.
:func:`aggregate_to_site_level`         -- collapse a precursor-level
                                           diff-exp DataFrame to a
                                           site-indexed one, dedupelicating
                                           the many-precursors-per-site
                                           join.  Feeds :func:`kinase_activity`.
:data:`DEFAULT_PRECURSOR_COLLAPSE_SETTINGS` -- the settings dict.
:func:`resolve_precursor_settings`      -- exposed for testing / advanced use.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    import anndata as ad
except ImportError:  # pragma: no cover
    ad = None  # type: ignore[assignment]

from alphaphos._version import __version__ as _ALPHAPHOS_VERSION
from alphaphos.constants import (
    COL_CANONICAL_QUANT,
    COL_EG_PRECURSOR_ID,
    COL_EG_PTM_LOC_PROBS,
    COL_PEP_PEPTIDE_POSITION,
    COL_PG_GENES,
    COL_PG_PROTEIN_GROUPS,
    COL_R_FILENAME,
)
from alphaphos.preprocess._collapse.output_format import assemble_anndata
from alphaphos.preprocess._collapse.parsing import (
    extract_first_valid_position,
    extract_sequence_modifications,
    parse_localization_probabilities,
    rank_select_positions,
)
from alphaphos.preprocess._collapse.selectivity import compute_selectivity
from alphaphos.preprocess._collapse.shared import (
    enable_verbose_logging,
    select_quantification_column,
)
from alphaphos.preprocess._collapse.site_pipeline import log2_transform

logger = logging.getLogger(__name__)


DEFAULT_PRECURSOR_COLLAPSE_SETTINGS: dict[str, Any] = {
    "search_engine": "SN",
    "quantification_level": None,  # engine default: SN -> MS2, DIA-NN -> MS1
    "aggregation_method": "sum",
    "noise_floor_filter": True,
    "drop_all_nan": True,
    "phospho_only": True,
    "annotate_localization": True,
    # Class I gate on the PEAK localization probability across all PSM rows
    # of a precursor. Default 0.75 matches Spectronaut's Class I convention.
    # This is the natural analog of site-level ``localization_strategy="global_max"``
    # at precursor granularity: keep the precursor if it was confidently
    # localized in ANY run, drop if it was never confident anywhere.
    # Pass ``None`` to disable (rationale-doc fallback for datasets where the
    # loc metric is known unreliable).
    "classI_cutoff": 0.75,
}

_ALLOWED_ENGINES = ("SN",)
_ALLOWED_QUANT_LEVELS = (None, "MS2", "MS1", "auto")
_ALLOWED_AGG_METHODS = ("sum", "mean", "median")

_CHARGE_TRAILING_RE = re.compile(r"\.(\d+)$")
_PHOSPHO_MARKER = "[Phospho (STY)]"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def resolve_precursor_settings(advanced: dict[str, Any] | None) -> dict[str, Any]:
    """Merge ``advanced`` overrides over :data:`DEFAULT_PRECURSOR_COLLAPSE_SETTINGS`.

    Validates every key + value; unknown keys or invalid values raise
    ``ValueError``. Returns a fresh dict.
    """
    out = dict(DEFAULT_PRECURSOR_COLLAPSE_SETTINGS)
    if advanced is None:
        return out
    unknown = set(advanced) - set(DEFAULT_PRECURSOR_COLLAPSE_SETTINGS)
    if unknown:
        raise ValueError(
            f"Unknown advanced keys: {sorted(unknown)}. "
            f"Allowed: {sorted(DEFAULT_PRECURSOR_COLLAPSE_SETTINGS)}"
        )
    out.update(advanced)
    if out["search_engine"] not in _ALLOWED_ENGINES:
        raise ValueError(
            f"search_engine must be one of {_ALLOWED_ENGINES}; got "
            f"{out['search_engine']!r}. DIA-NN and FragPipe support will "
            "land in a follow-up."
        )
    if out["quantification_level"] not in _ALLOWED_QUANT_LEVELS:
        raise ValueError(
            f"quantification_level must be one of {_ALLOWED_QUANT_LEVELS}; "
            f"got {out['quantification_level']!r}"
        )
    if out["aggregation_method"] not in _ALLOWED_AGG_METHODS:
        raise ValueError(
            f"aggregation_method must be one of {_ALLOWED_AGG_METHODS}; "
            f"got {out['aggregation_method']!r}"
        )
    for bool_key in ("noise_floor_filter", "drop_all_nan", "phospho_only", "annotate_localization"):
        if not isinstance(out[bool_key], bool):
            raise ValueError(f"{bool_key} must be a bool; got {type(out[bool_key]).__name__}")
    cutoff = out["classI_cutoff"]
    if cutoff is not None:
        if not isinstance(cutoff, (int, float)) or isinstance(cutoff, bool):
            raise ValueError(f"classI_cutoff must be a float in [0, 1] or None; got {cutoff!r}")
        if not (0.0 <= float(cutoff) <= 1.0):
            raise ValueError(f"classI_cutoff must be in [0, 1]; got {cutoff!r}")
        if not out["annotate_localization"]:
            raise ValueError(
                "classI_cutoff requires annotate_localization=True (no loc info "
                "to gate on otherwise). Set classI_cutoff=None to disable the gate."
            )
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collapse_precursors(
    data: pd.DataFrame,
    *,
    condition_df: pd.DataFrame | None = None,
    advanced: dict[str, Any] | None = None,
    verbose: bool = False,
) -> ad.AnnData:
    """Collapse a PSM-level DataFrame to a precursor-level ``AnnData``.

    One row per unique precursor identifier
    ``(peptide_sequence, charge, modifications)`` per protein group, one
    column per sample.  No residue-level parsing.  Modification
    information is preserved on ``.var``; the best-localized site (if
    any) is annotated too.

    **Localization gating**: by default a Class-I gate on the precursor's
    best localization probability (``classI_cutoff=0.75``) drops precursors
    whose PTMs are all poorly localized -- see
    :data:`DEFAULT_PRECURSOR_COLLAPSE_SETTINGS`.  Pass
    ``advanced={"classI_cutoff": None}`` for a truly localization-agnostic
    peptide view (e.g. when the downstream analysis wants EVERY precursor,
    including those with only a diffuse phospho signal).

    Parameters
    ----------
    data
        PSM-level DataFrame from :func:`alphaphos.read_spectronaut`.
        Currently only Spectronaut is supported (see ``advanced["search_engine"]``).
    condition_df
        Optional sample metadata (``sample`` + ``condition`` columns at
        minimum; extras joined into ``.obs``).
    advanced
        Overrides for :data:`DEFAULT_PRECURSOR_COLLAPSE_SETTINGS`.  Unknown
        keys raise.
    verbose
        Emit INFO-level stage progress to stderr.

    Returns
    -------
    anndata.AnnData
        Shape ``(n_samples, n_precursors)``:

        * ``.X`` == ``layers["intensity_log2"]`` -- log2 intensity per
          precursor per sample.
        * ``layers["localization"]`` -- per-cell best localization
          probability (NaN when unavailable).  Never used for masking.
        * ``.var.index`` = ``"Protein|Gene|Peptide|Charge|Mods"`` (alphaPhos
          precursor key).
        * ``.var`` columns: ``protein_group_id``, ``gene``,
          ``peptide_sequence``, ``charge``, ``mods``, ``n_phospho``,
          ``best_localization_prob``, ``best_localization_pos_peptide``,
          ``best_localization_pos_protein``, ``peptide_start``.
        * ``.obs`` = sample metadata + ``phospho_selectivity_pct``.
        * ``.uns["alphaphos"]`` = version + resolved settings + stats.
    """
    if verbose:
        enable_verbose_logging(logger)

    settings = resolve_precursor_settings(advanced)

    if COL_EG_PRECURSOR_ID not in data.columns:
        raise KeyError(f"Input DataFrame is missing required column {COL_EG_PRECURSOR_ID!r}.")

    # Capture PSM lineage before ``data`` is rebound by the filters below.
    source_attrs = dict(data.attrs) if getattr(data, "attrs", None) else None

    # ---- Selectivity (per-sample phospho-selectivity %) is engine-agnostic
    selectivity = compute_selectivity(data)

    # ---- Pick the quant column (shared with collapse_sites) --------------
    data, chosen_col, level_used = select_quantification_column(
        data,
        engine=settings["search_engine"],
        requested_level=settings["quantification_level"],
        logger=logger,
    )

    # ---- Parse precursor identifiers -------------------------------------
    parsed = _parse_precursors(data[COL_EG_PRECURSOR_ID])
    #   parsed is a DataFrame indexed like `data`, columns:
    #     peptide_sequence, charge, mods, n_phospho, phospho_positions_peptide

    if settings["phospho_only"]:
        keep = parsed["n_phospho"] > 0
        n_dropped = int((~keep).sum())
        if n_dropped:
            logger.info(
                "collapse_precursors: dropped %d non-phospho precursor rows (phospho_only=True)",
                n_dropped,
            )
        data = data.loc[keep].copy()
        parsed = parsed.loc[keep].copy()

    if data.empty:
        raise ValueError(
            "No precursors survived filtering.  Check that the input actually "
            "contains phospho PSMs and that quantification_level is achievable "
            "for this engine."
        )

    # ---- Build the precursor key & work frame ---------------------------
    work = pd.DataFrame(
        {
            "protein_group_id": data[COL_PG_PROTEIN_GROUPS].astype(str).to_numpy(),
            "gene": data[COL_PG_GENES].astype(str).to_numpy(),
            "peptide_sequence": parsed["peptide_sequence"].to_numpy(),
            "charge": parsed["charge"].to_numpy(),
            "mods": parsed["mods"].to_numpy(),
            "n_phospho": parsed["n_phospho"].to_numpy(),
            "phospho_positions_peptide": parsed["phospho_positions_peptide"].to_numpy(),
            "peptide_start": (
                data[COL_PEP_PEPTIDE_POSITION].map(extract_first_valid_position).to_numpy()
                if COL_PEP_PEPTIDE_POSITION in data.columns
                else np.array([None] * len(data), dtype=object)
            ),
            "loc_prob_string": (
                data[COL_EG_PTM_LOC_PROBS].to_numpy()
                if COL_EG_PTM_LOC_PROBS in data.columns
                else np.array([None] * len(data), dtype=object)
            ),
            "sample": data[COL_R_FILENAME].astype(str).to_numpy(),
            "quant": data[COL_CANONICAL_QUANT].to_numpy(),
        }
    )
    work["precursor_key"] = _build_keys(
        work["protein_group_id"],
        work["gene"],
        work["peptide_sequence"],
        work["charge"],
        work["mods"],
    )

    # ---- Aggregate to (precursor x sample) linear intensity -------------
    quant_linear = work.pivot_table(
        index="precursor_key",
        columns="sample",
        values="quant",
        aggfunc=settings["aggregation_method"],
    )

    # ---- Build the per-precursor loc matrix (best-loc-prob per run) -----
    if settings["annotate_localization"] and COL_EG_PTM_LOC_PROBS in data.columns:
        loc_per_run = _build_loc_matrix(work)
        # Align to the quant matrix (same index + columns)
        loc_per_run = loc_per_run.reindex(index=quant_linear.index, columns=quant_linear.columns)
    else:
        loc_per_run = pd.DataFrame(
            np.nan,
            index=quant_linear.index,
            columns=quant_linear.columns,
            dtype=float,
        )

    # ---- log2 (+ noise floor) -------------------------------------------
    quant_log2 = log2_transform(
        quant_linear,
        apply_noise_floor=settings["noise_floor_filter"],
        logger=logger,
    )

    if settings["drop_all_nan"]:
        keep = ~quant_log2.isna().all(axis=1)
        n_dropped = int((~keep).sum())
        if n_dropped:
            logger.info(
                "collapse_precursors: dropped %d all-NaN precursors after log2/noise-floor",
                n_dropped,
            )
        quant_log2 = quant_log2.loc[keep]
        loc_per_run = loc_per_run.loc[keep]

    # ---- Build per-precursor .var metadata ------------------------------
    var_meta = _build_var_metadata(
        work,
        quant_log2.index,
        annotate_localization=settings["annotate_localization"],
    )

    # ---- Class I gate: drop precursors whose PEAK localization probability
    # across all PSM rows never met the cutoff.  Natural analog of the
    # site-level "global_max" strategy at precursor granularity: a
    # phospho precursor that was NEVER confidently localized in any run
    # is dropped; a precursor that was confident in ANY run is kept.
    # NaN best_localization_prob (no parseable loc string anywhere)
    # counts as "below the cutoff" and is dropped.
    #
    # Non-phospho precursors (n_phospho==0, only present when
    # ``phospho_only=False``) bypass the gate -- localization confidence
    # is not a meaningful concept for them.
    #
    # Pass ``classI_cutoff=None`` to disable the gate entirely.
    n_dropped_classI = 0
    if settings["classI_cutoff"] is not None:
        cutoff = float(settings["classI_cutoff"])
        is_phospho = var_meta["n_phospho"].fillna(0).astype(int) > 0
        loc_ok = var_meta["best_localization_prob"] >= cutoff
        # NaN comparisons yield False -> those precursors get dropped.
        # Non-phospho precursors are always kept (loc not applicable).
        keep = (~is_phospho) | loc_ok
        n_dropped_classI = int((~keep).sum())
        if n_dropped_classI:
            logger.info(
                "collapse_precursors: dropped %d/%d precursors with "
                "best_localization_prob < %s (classI_cutoff)",
                n_dropped_classI,
                len(var_meta),
                cutoff,
            )
        var_meta = var_meta.loc[keep]
        quant_log2 = quant_log2.loc[keep]
        loc_per_run = loc_per_run.loc[keep]

    if var_meta.empty:
        raise ValueError(
            "No precursors survived the classI_cutoff gate. Either lower the "
            "cutoff, disable it (classI_cutoff=None), or verify that "
            "EG.PTMLocalizationProbabilities is present in the input."
        )

    # ---- Package + adapt AnnData ----------------------------------------
    n_input_rows = len(data)
    n_precursors = int(quant_log2.shape[0])
    n_samples = int(quant_log2.shape[1])
    stats = {
        "n_input_rows": n_input_rows,
        "n_precursors": n_precursors,
        "n_samples": n_samples,
        "quantification_column_used": chosen_col,
        "quantification_level_used": level_used,
        "n_dropped_classI": n_dropped_classI,
        "classI_cutoff": settings["classI_cutoff"],
    }

    # The resolved settings are stamped verbatim into uns["pipeline_params"]
    # (including classI_cutoff=None when the gate is disabled -- assemble_anndata
    # falls back to 0.75 for the informational var QC columns in that case).
    adata = assemble_anndata(
        quant_log2,
        loc_per_run,
        var_meta,
        condition_df=condition_df,
        settings=settings,
        stats=stats,
        selectivity=selectivity,
        decision_table=None,
        source_attrs=source_attrs,
        short_key_collisions=None,
        version=_ALPHAPHOS_VERSION,
        logger=logger,
    )
    # Rename the hardcoded var index and strip site-centric QC columns that
    # don't make sense at precursor level.  The n_samples_detected column
    # IS still meaningful (# samples with a finite log2 intensity for this
    # precursor) so it's kept.
    adata.var.index.name = "precursor_key"
    for col in (
        "n_classI_samples",
        "fraction_classI",
        "mean_loc_prob",
        "max_loc_prob",
        "min_loc_prob",
    ):
        if col in adata.var.columns and not settings["annotate_localization"]:
            adata.var.drop(columns=col, inplace=True)

    return adata


# ---------------------------------------------------------------------------
# Bridge: precursor.var -> best-guess site key for KSEA / pathway hand-off
# ---------------------------------------------------------------------------


def precursor_to_site_view(
    adata: ad.AnnData,
    *,
    require_localization: float | None = 0.75,
    residue_alphabet: tuple[str, ...] = ("S", "T", "Y"),
) -> pd.DataFrame:
    """Return a per-precursor DataFrame with a best-guess alphaPhos site key.

    Reads the precursor-level ``adata.var`` columns produced by
    :func:`collapse_precursors` and constructs a site key
    ``"Protein|Gene|<AA><absolute_pos>|M<n_phospho>"`` for each precursor
    whose best-localization probability meets ``require_localization``.
    Not a re-collapse -- just annotation for downstream KSEA / pathway
    analyses that need a specific residue.

    Parameters
    ----------
    adata
        The output of :func:`collapse_precursors`.  Must carry the
        ``best_localization_prob``, ``best_localization_pos_protein``,
        ``peptide_sequence``, ``peptide_start``, ``gene``,
        ``protein_group_id``, and ``n_phospho`` columns in ``.var``.
    require_localization
        Minimum ``best_localization_prob`` for a precursor to be assigned a
        site key.  Precursors below the threshold get a ``NaN`` site key.
        Pass ``None`` to skip the gate (every precursor with a resolvable
        site gets a key).
    residue_alphabet
        Residues that count as phospho-acceptors.  Non-STY residues at the
        best-localization position get a ``NaN`` site key.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``adata.var.index`` (precursor_key), columns:

        * ``site_key`` -- ``"Protein|Gene|<AA><pos>|M<mult>"`` or NaN
        * ``site_residue`` -- ``S``/``T``/``Y`` or NaN
        * ``site_position_protein`` -- 1-indexed absolute AA position or NaN
        * ``best_localization_prob`` -- verbatim from ``.var``
        * ``passes_localization`` -- bool
    """
    required = {
        "peptide_sequence",
        "peptide_start",
        "gene",
        "protein_group_id",
        "n_phospho",
        "best_localization_prob",
        "best_localization_pos_peptide",
        "best_localization_pos_protein",
    }
    missing = required - set(adata.var.columns)
    if missing:
        raise ValueError(
            "precursor_to_site_view: adata.var is missing columns "
            f"{sorted(missing)}. Did you produce this AnnData via "
            "collapse_precursors(annotate_localization=True)?"
        )
    v = adata.var
    prob = pd.to_numeric(v["best_localization_prob"], errors="coerce")
    if require_localization is None:
        passes = prob.notna().to_numpy()
    else:
        passes = (prob >= float(require_localization)).fillna(False).to_numpy(dtype=bool)

    # Column-wise extraction + one plain zip loop (no DataFrame.iterrows, which
    # builds a Series per row and dominated runtime at 1e5 precursors).
    peps = v["peptide_sequence"].tolist()
    pep_pos = pd.to_numeric(v["best_localization_pos_peptide"], errors="coerce").to_numpy()
    prot_pos = pd.to_numeric(v["best_localization_pos_protein"], errors="coerce").to_numpy()
    mults = pd.to_numeric(v["n_phospho"], errors="coerce").fillna(0).to_numpy()
    pgs = v["protein_group_id"].astype(str).tolist()
    genes = v["gene"].astype(str).tolist()

    site_keys: list[str | None] = []
    residues: list[str | None] = []
    positions: list[float] = []
    for ok, pep, pp, pr, m, pg, gene in zip(
        passes, peps, pep_pos, prot_pos, mults, pgs, genes, strict=True
    ):
        key: str | None = None
        residue: str | None = None
        pos = np.nan
        if ok and isinstance(pep, str) and pep and not np.isnan(pp) and not np.isnan(pr):
            i = int(pp) - 1  # 1-indexed -> 0-indexed
            if 0 <= i < len(pep) and pep[i] in residue_alphabet:
                residue = pep[i]
                key = f"{pg}|{gene}|{residue}{int(pr)}|M{int(m)}"
                pos = float(int(pr))
        site_keys.append(key)
        residues.append(residue)
        positions.append(pos)

    out = pd.DataFrame(
        {
            "site_key": site_keys,
            "site_residue": residues,
            "site_position_protein": positions,
            "best_localization_prob": prob.to_numpy(),
            "passes_localization": passes,
        },
        index=pd.Index(v.index, name="precursor_key"),
    )
    return out


# ---------------------------------------------------------------------------
# Aggregation: precursor-level diff-exp result -> site-level diff-exp result
# ---------------------------------------------------------------------------


def aggregate_to_site_level(
    precursor_result: pd.DataFrame,
    site_view: pd.DataFrame,
    *,
    stat_col: str = "log2fc",
    stat_agg: Literal["max_abs", "mean", "first"] = "max_abs",
    fdr_col: str | None = "fdr",
    fdr_agg: Literal["min", "mean"] = "min",
) -> pd.DataFrame:
    """Collapse a precursor-indexed diff-exp DataFrame to a site-indexed one.

    Uses the mapping produced by :func:`precursor_to_site_view` to route each
    precursor to a phosphosite.  Multiple precursors covering the same site
    (different peptides, charges, or multiplicities) are aggregated per
    ``stat_agg``.  Precursors with no site assignment (``site_key`` is NaN
    -- e.g. below the localization gate, or a non-STY peak position) are
    dropped.

    Enables the KSEA / kinase-activity hand-off from a precursor-level
    ``diff_exp_limma`` result::

        adata = ap.collapse_precursors(psm, ...)
        result = ap.diff_exp_limma(adata, ...)
        site_view = ap.precursor_to_site_view(adata, require_localization=0.75)
        site_result = ap.aggregate_to_site_level(result, site_view)
        ksea = ap.enrichment.kinase_activity(
            site_result, stat_col="log2fc", network="omnipath",
        )

    Parameters
    ----------
    precursor_result
        DataFrame indexed by precursor_key (matching ``site_view.index``).
        Usually the output of :func:`diff_exp_limma` run on a precursor-level
        AnnData.
    site_view
        Output of :func:`precursor_to_site_view`.  Must carry a ``site_key``
        column.
    stat_col
        Column in ``precursor_result`` carrying the signed effect statistic.
    stat_agg
        How to combine precursors that map to the same site:

        - ``"max_abs"`` (default) -- keep the precursor with the largest
          ``|stat|``, retain its signed value.  Preserves peak regulation.
        - ``"mean"`` -- arithmetic mean of the stat across precursors.
        - ``"first"`` -- first-seen precursor's value.
    fdr_col
        FDR column to also aggregate (if present).  Pass ``None`` to skip.
    fdr_agg
        ``"min"`` (default) -- take the smallest FDR across precursors at
        the same site.  ``"mean"`` -- arithmetic mean.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``site_key``.  Columns: the aggregated ``stat_col`` and
        (if not ``None``) the aggregated ``fdr_col``, plus ``n_precursors``
        (how many precursors were aggregated at each site).
    """
    if stat_agg not in ("max_abs", "mean", "first"):
        raise ValueError(f"stat_agg must be one of 'max_abs'/'mean'/'first'; got {stat_agg!r}")
    if fdr_agg not in ("min", "mean"):
        raise ValueError(f"fdr_agg must be 'min' or 'mean'; got {fdr_agg!r}")
    if "site_key" not in site_view.columns:
        raise ValueError("site_view must have a 'site_key' column (see precursor_to_site_view).")
    if stat_col not in precursor_result.columns:
        raise KeyError(f"stat_col {stat_col!r} not in precursor_result columns.")

    joined = precursor_result.join(site_view[["site_key"]], how="left").dropna(subset=["site_key"])
    if joined.empty:
        raise ValueError(
            "No precursors have a resolvable site_key in site_view.  "
            "Lower require_localization or verify the bridge is aligned."
        )

    grouped = joined.groupby("site_key", sort=False)

    def _agg_stat(sub: pd.Series) -> float:
        if stat_agg == "max_abs":
            i = sub.abs().idxmax()
            return float(sub.loc[i])
        if stat_agg == "mean":
            return float(sub.mean())
        return float(sub.iloc[0])  # "first"

    out = pd.DataFrame(
        {
            stat_col: grouped[stat_col].apply(_agg_stat),
            "n_precursors": grouped.size().astype(int),
        }
    )
    if fdr_col is not None and fdr_col in precursor_result.columns:
        if fdr_agg == "min":
            out[fdr_col] = grouped[fdr_col].min()
        else:
            out[fdr_col] = grouped[fdr_col].mean()
    out.index.name = "site_key"
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _parse_precursors(precursor_ids: pd.Series) -> pd.DataFrame:
    """Parse a Series of ``EG.PrecursorId`` values.

    Returns a DataFrame with the input's index and columns:

        peptide_sequence -- plain amino acids (e.g. "STDNAFENPFFK")
        charge           -- int precursor charge from the ``.N`` suffix
        mods             -- pipe-joined bracket contents (empty string if none)
        n_phospho        -- int number of ``[Phospho (STY)]`` markers
        phospho_positions_peptide -- tuple of 1-indexed peptide positions
    """
    peptide_seqs: list[str] = []
    charges: list[int | None] = []
    mods_strs: list[str] = []
    n_phosphos: list[int] = []
    phos_positions: list[tuple[int, ...]] = []

    for raw in precursor_ids.astype(str):
        # Charge is the trailing ".<digits>" -- pulled off the raw before wrapper strip
        charge = None
        m = _CHARGE_TRAILING_RE.search(raw)
        if m:
            charge = int(m.group(1))
        parsed = extract_sequence_modifications(raw)
        peptide_seqs.append(parsed["clean_sequence"])
        charges.append(charge)
        mods_strs.append("|".join(parsed["all_modifications"]))
        n_phosphos.append(int(parsed["phospho_count"]))
        phos_positions.append(tuple(int(p) for p in parsed["phospho_positions"]))

    return pd.DataFrame(
        {
            "peptide_sequence": peptide_seqs,
            "charge": charges,
            "mods": mods_strs,
            "n_phospho": n_phosphos,
            "phospho_positions_peptide": phos_positions,
        },
        index=precursor_ids.index,
    )


def _build_keys(
    protein: pd.Series,
    gene: pd.Series,
    peptide: pd.Series,
    charge: pd.Series,
    mods: pd.Series,
) -> pd.Series:
    """Build the alphaPhos precursor key ``Protein|Gene|Peptide|Charge|Mods``.

    ``Mods`` uses ``+`` as the internal separator (safer than pipe since the
    pipe is our field separator).  Empty mods slot is rendered as ``none``
    so the key never contains a bare trailing pipe.
    """
    mods_safe = mods.fillna("").astype(str).str.replace("|", "+", regex=False)
    mods_safe = mods_safe.where(mods_safe != "", "none")
    charge_str = charge.fillna(0).astype(int).astype(str)
    return (
        protein.astype(str)
        + "|"
        + gene.astype(str)
        + "|"
        + peptide.astype(str)
        + "|"
        + charge_str
        + "|"
        + mods_safe
    )


def _build_loc_matrix(work: pd.DataFrame) -> pd.DataFrame:
    """Best per-(precursor, sample) localization probability.

    For each PSM row, parses ``EG.PTMLocalizationProbabilities`` to get the
    per-position map, picks the peak probability (over the peptide-local
    phospho positions), and aggregates per (precursor_key, sample) as the
    max across duplicate rows.  Rows without a parseable loc string
    contribute NaN.
    """
    tmp = work.copy()
    best_probs: list[float] = []
    for loc_str in tmp["loc_prob_string"]:
        loc_map = parse_localization_probabilities(loc_str)
        if not loc_map:
            best_probs.append(np.nan)
            continue
        _, probs = rank_select_positions(loc_map, 1)
        best_probs.append(float(probs[0]) if probs else np.nan)
    tmp["_best_prob"] = best_probs
    return tmp.pivot_table(
        index="precursor_key",
        columns="sample",
        values="_best_prob",
        aggfunc="max",
    )


def _build_var_metadata(
    work: pd.DataFrame,
    precursor_index: pd.Index,
    *,
    annotate_localization: bool,
) -> pd.DataFrame:
    """Aggregate per-precursor metadata across PSM rows.

    For each precursor_key surviving in ``precursor_index``, pick the first
    row's protein_group_id / gene / peptide_sequence / charge / mods /
    n_phospho / peptide_start (these are constant by definition of the key,
    except peptide_start which can occasionally vary; we take the first).
    Best-localization is aggregated as the peak across rows when
    ``annotate_localization=True``; otherwise the localization columns are
    populated with NaN so the ``.var`` schema is stable regardless of the
    flag.
    """
    grouped = work.set_index("precursor_key")
    #   Non-loc metadata: keep first per precursor
    static_cols = [
        "protein_group_id",
        "gene",
        "peptide_sequence",
        "charge",
        "mods",
        "n_phospho",
        "peptide_start",
    ]
    meta = grouped[static_cols].groupby(level=0).first()

    if annotate_localization:
        # Best-localization info per precursor across PSM rows
        def _best_row(sub: pd.DataFrame) -> pd.Series:
            best_pos_pep: int | None = None
            best_prob: float = float("nan")
            for loc_str in sub["loc_prob_string"]:
                loc_map = parse_localization_probabilities(loc_str)
                if not loc_map:
                    continue
                positions, probs = rank_select_positions(loc_map, 1)
                if not probs:
                    continue
                if np.isnan(best_prob) or probs[0] > best_prob:
                    best_prob = float(probs[0])
                    best_pos_pep = int(positions[0])
            return pd.Series(
                {
                    "best_localization_prob": best_prob,
                    "best_localization_pos_peptide": best_pos_pep,
                }
            )

        loc_meta = grouped[["loc_prob_string"]].groupby(level=0).apply(_best_row)
        if isinstance(loc_meta, pd.Series):
            loc_meta = loc_meta.unstack()
        meta = meta.join(loc_meta, how="left")

        # Absolute (protein-level) best-loc position:
        #   protein_position = peptide_start + peptide_local_pos - 1
        def _abs_pos(row: pd.Series) -> float:
            start = row.get("peptide_start")
            local = row.get("best_localization_pos_peptide")
            if start is None or pd.isna(start) or local is None or pd.isna(local):
                return float("nan")
            return float(int(start) + int(local) - 1)

        meta["best_localization_pos_protein"] = meta.apply(_abs_pos, axis=1)
    else:
        # Stable schema: columns present but NaN.
        meta["best_localization_prob"] = np.nan
        meta["best_localization_pos_peptide"] = np.nan
        meta["best_localization_pos_protein"] = np.nan

    meta = meta.reindex(precursor_index)
    return meta
