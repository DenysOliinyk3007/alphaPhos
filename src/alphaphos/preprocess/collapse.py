"""Public entry point for the peptide -> site collapse pipeline.

The single public function :func:`collapse_sites` composes the stages in
:mod:`._collapse` (parsing -> explode -> pivot -> aggregate -> mask ->
log2 -> assemble AnnData) into a one-call pipeline that returns a fully-
packaged :class:`anndata.AnnData` ready for downstream analysis.

Configuration
-------------
All algorithmic knobs live in :data:`DEFAULT_COLLAPSE_SETTINGS`. Users
change behavior by passing a dict of overrides via the ``advanced``
kwarg::

    import alphaphos as ap

    # 1) Defaults
    adata = ap.collapse_sites(df, condition_df=cdf)

    # 2) Selective override
    adata = ap.collapse_sites(
        df, condition_df=cdf,
        advanced={"aggregation_method": "median", "classI_cutoff": 0.85},
    )

    # 3) Full customization -- inspect + edit the defaults
    adv = dict(ap.DEFAULT_COLLAPSE_SETTINGS)
    adv["localization_strategy"] = "per_run"
    adv["cutoff"] = 0.65
    adata = ap.collapse_sites(df, advanced=adv)

Unknown keys in ``advanced`` raise, so typos surface immediately.
Value-level validation (allowed strategies, allowed engines, etc.) also
raises with a message naming the accepted set.

FASTA / kinase annotation
-------------------------
This module is DELIBERATELY decoupled from FASTA lookups: the kinase
sequence window is added by :func:`alphaphos.add_kinase_windows` on the
resulting AnnData. This lets you run collapse on non-human data without
needing a FASTA at all, and lets you annotate late in the pipeline only
when kinase-level analysis actually needs it.

Public API
----------
:func:`collapse_sites`             -- the entry point.
:data:`DEFAULT_COLLAPSE_SETTINGS`  -- the settings dict.
:func:`resolve_settings`           -- exposed for testing / advanced users.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import pandas as pd

try:
    import anndata as ad
except ImportError:  # pragma: no cover
    ad = None  # type: ignore[assignment]

from alphaphos import __version__ as _alphaphos_version
from alphaphos.constants import (
    COL_CANONICAL_QUANT,
    COL_EG_PTM_LOC_PROBS,
    OBS_CONDITION,
    OBS_SAMPLE,
)
from alphaphos.io.schemas import resolve_quant_column
from alphaphos.preprocess._collapse.masking import (
    VALID_STRATEGIES,
    drop_all_nan_sites,
    filter_by_global_max,
    mask_condition_aware,
    mask_per_run,
)
from alphaphos.preprocess._collapse.output_format import assemble_anndata
from alphaphos.preprocess._collapse.selectivity import compute_selectivity
from alphaphos.preprocess._collapse.site_pipeline import (
    aggregate_precursors_to_sites,
    build_precursor_pivots,
    compute_site_metadata,
    explode_to_sites,
    log2_transform,
    prepare_psms,
    resolve_short_keys,
)
from alphaphos.preprocess.attribution import filter_to_top_n_positions

logger = logging.getLogger("alphaphos.preprocess.collapse")


# ---------------------------------------------------------------------------
# Public defaults + engine dispatch
# ---------------------------------------------------------------------------


DEFAULT_COLLAPSE_SETTINGS: dict[str, Any] = {
    "search_engine": "SN",  # "SN" | "Diann" | "Fragpipe" | "Peaks" (only SN implemented)
    "quantification_level": "MS2",  # "MS2" | "MS1" | "auto" (fallback chain per schemas.py)
    "top_n_attribution": True,  # Spectronaut over-export dedup (safe default; on)
    "cutoff": 0.75,  # loc cutoff for per_run / global_max
    "classI_cutoff": 0.75,  # loc cutoff for the condition-aware mask
    "condition_threshold": 0.50,  # min fraction of Class-I reps to keep condition
    "collapse_level": "PG",  # "PG" (protein group) | "P" (protein resolved)
    "aggregation_method": "sum",  # "sum" | "median" | "mean" | "consolidate"
    "localization_strategy": "condition",  # "condition" | "per_run" | "global_max"
    "noise_floor_filter": True,  # remove log2 values in {0, 1}
    "drop_all_nan": True,  # drop sites fully NaN after the mask
}

_ALLOWED_ENGINES = ("SN", "Diann", "Fragpipe", "Peaks")
_ALLOWED_QUANT_LEVELS = ("MS2", "MS1", "auto")


# ---------------------------------------------------------------------------
# Settings resolution
# ---------------------------------------------------------------------------


def resolve_settings(advanced: dict[str, Any] | None) -> dict[str, Any]:
    """Merge ``advanced`` overrides on top of :data:`DEFAULT_COLLAPSE_SETTINGS`.

    Validates every key and value; unknown keys or bad values raise ``ValueError``
    with a message naming the allowed set. Returns a fresh dict so callers
    can mutate the result without affecting the module-level defaults.

    Parameters
    ----------
    advanced : dict | None
        Overrides. Missing keys inherit the default.

    Returns
    -------
    dict[str, Any]
        Fully-resolved settings.
    """
    settings = dict(DEFAULT_COLLAPSE_SETTINGS)
    if advanced is None:
        return settings

    if not isinstance(advanced, dict):
        raise TypeError(f"'advanced' must be a dict or None, got {type(advanced).__name__}")

    unknown = set(advanced) - set(DEFAULT_COLLAPSE_SETTINGS)
    if unknown:
        raise ValueError(
            f"Unknown keys in 'advanced': {sorted(unknown)}. "
            f"Allowed: {sorted(DEFAULT_COLLAPSE_SETTINGS)}."
        )

    settings.update(advanced)

    # Value-level validation
    if settings["search_engine"] not in _ALLOWED_ENGINES:
        raise ValueError(
            f"search_engine must be one of {_ALLOWED_ENGINES}, got {settings['search_engine']!r}"
        )
    if settings["quantification_level"] not in _ALLOWED_QUANT_LEVELS:
        raise ValueError(
            f"quantification_level must be one of {_ALLOWED_QUANT_LEVELS}, "
            f"got {settings['quantification_level']!r}"
        )
    if not isinstance(settings["top_n_attribution"], bool):
        raise ValueError(
            f"top_n_attribution must be bool, got {type(settings['top_n_attribution']).__name__}"
        )
    if settings["localization_strategy"] not in VALID_STRATEGIES:
        raise ValueError(
            f"localization_strategy must be one of {VALID_STRATEGIES}, "
            f"got {settings['localization_strategy']!r}"
        )
    if settings["collapse_level"] not in ("PG", "P"):
        raise ValueError(f"collapse_level must be 'PG' or 'P', got {settings['collapse_level']!r}")
    if settings["aggregation_method"] not in ("sum", "median", "mean", "consolidate"):
        raise ValueError(
            f"aggregation_method must be one of ('sum', 'median', 'mean', 'consolidate'), "
            f"got {settings['aggregation_method']!r}"
        )
    for prob_key in ("cutoff", "classI_cutoff"):
        if not 0 <= float(settings[prob_key]) <= 1:
            raise ValueError(f"{prob_key} must be in [0, 1], got {settings[prob_key]}")
    if not 0 < float(settings["condition_threshold"]) <= 1:
        raise ValueError(
            f"condition_threshold must be in (0, 1], got {settings['condition_threshold']}"
        )

    return settings


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------


def collapse_sites(
    data: pd.DataFrame,
    *,
    condition_df: pd.DataFrame | None = None,
    advanced: dict[str, Any] | None = None,
    verbose: bool = False,
) -> ad.AnnData:
    """Collapse a PSM-level DataFrame to a site-level ``AnnData``.

    Runs the canonical Hogrebe-style pipeline: parse ``EG.PrecursorId``,
    explode to per-site rows, pivot to a (site x sample) linear intensity
    matrix, aggregate multi-precursor evidence, mask by localization
    probability, log2-transform, and pack everything into an ``AnnData``.

    Parameters
    ----------
    data : DataFrame
        PSM-level input. For Spectronaut, this is the output of
        :func:`alphaphos.io.read_spectronaut`. Required columns depend on
        the ``search_engine`` (currently only ``"SN"`` is implemented;
        others raise ``NotImplementedError``).

        Any ``data.attrs`` dict is preserved in
        ``adata.uns["source_attrs"]``.
    condition_df : DataFrame, optional
        Sample-level metadata. Must contain columns ``sample`` (matching
        ``R.FileName`` values) and ``condition`` (replicate grouping);
        extra columns are joined into ``adata.obs`` as-is. **Required**
        when ``advanced["localization_strategy"] == "condition"`` (the
        default). Optional otherwise.
    advanced : dict, optional
        Overrides for :data:`DEFAULT_COLLAPSE_SETTINGS`. See module
        docstring for examples. Unknown keys raise.
    verbose : bool
        If True, INFO-level logs are printed to stderr. False (default) is
        library-quiet; all logs still go to the ``alphaphos.preprocess.collapse``
        logger for external configuration.

    Returns
    -------
    anndata.AnnData
        Shape ``(n_samples, n_sites)``.

        * ``.X`` == ``layers["intensity_log2"]``: log2 intensities.
        * ``layers["localization"]``: per-cell localization probability.
        * ``.var.index`` = full key ``"Protein|Gene|Site|Mult"``.
        * ``.var`` columns: ``short_key``, ``pg_key``, ``protein_group_id``,
          ``gene``, ``site_aa``, ``site_position``, ``multiplicity``,
          ``UPD_seq``, ``n_samples_detected``, ``mean/max/min_loc_prob``,
          ``n_classI_samples``, ``fraction_classI``.
        * ``.obs.index`` = sample id (from ``R.FileName``).
        * ``.obs`` columns: ``condition`` (from ``condition_df`` if given),
          ``phospho_selectivity_pct`` (fraction of phospho-containing
          precursors in that sample's raw PSMs).
        * ``.uns["alphaphos"]`` = ``version``, resolved ``pipeline_params``,
          per-stage ``stats``, and (when applicable) ``classI_decision_table``
          and ``short_key_collisions``.
        * ``.uns["source_attrs"]`` = ``data.attrs`` (PSM lineage).

    Raises
    ------
    ValueError
        If ``advanced`` contains unknown keys or invalid values.
    NotImplementedError
        If ``search_engine`` is set to any value other than ``"SN"``.
    KeyError
        If required PSM columns are missing.
    """
    settings = resolve_settings(advanced)
    _configure_logger(verbose)

    if settings["search_engine"] not in ("SN", "Diann"):
        raise NotImplementedError(
            f"search_engine={settings['search_engine']!r} is not yet implemented. "
            f"Currently supported: 'SN' (Spectronaut), 'Diann'. Others planned: "
            f"'Fragpipe', 'Peaks'."
        )

    if settings["localization_strategy"] == "condition" and condition_df is None:
        raise ValueError(
            "localization_strategy='condition' requires condition_df. Pass a "
            "DataFrame with columns 'sample' (matching R.FileName) and "
            "'condition' (grouping replicates)."
        )
    if condition_df is not None and not {OBS_SAMPLE, OBS_CONDITION}.issubset(condition_df.columns):
        missing = {OBS_SAMPLE, OBS_CONDITION} - set(condition_df.columns)
        raise KeyError(f"condition_df is missing required columns: {sorted(missing)}")

    stats: dict[str, Any] = {"n_psms_loaded": len(data)}

    # -- Selectivity: per-sample phospho fraction from RAW PSMs (before any filter).
    selectivity = compute_selectivity(data)

    # -- Pre-stage A: pick the quant column (fallback chain per schemas.py; warn on fallback)
    data, chosen_quant_col, quant_level_used = _select_quantification_column(
        data,
        engine=settings["search_engine"],
        requested_level=settings["quantification_level"],
    )
    stats["quantification_column_used"] = chosen_quant_col
    stats["quantification_level_used"] = quant_level_used

    # -- Pre-stage B: top-N attribution dedup (Spectronaut over-export fix; on by default)
    if settings["top_n_attribution"]:
        data = _apply_top_n_attribution(data)
    stats["n_psms_after_top_n"] = len(data)

    # -- Stage 1-2: parse + explode
    prepared = prepare_psms(data, logger=logger)
    stats["n_psms_phospho"] = len(prepared)

    exploded = explode_to_sites(prepared, logger=logger)

    # -- Stage 3: pivot precursor-level tables
    quant_precursor, loc_precursor, meta_precursor = build_precursor_pivots(exploded, logger=logger)

    # -- Stage 4: canonical keys + metadata
    meta_precursor = compute_site_metadata(
        meta_precursor,
        collapse_level=settings["collapse_level"],
        logger=logger,
    )

    # -- Stage 5: aggregate precursors to sites
    # Note: this uses the aggregation method for the QUANT matrix. The LOC
    # matrix is always max-aggregated by site (per-run cell = strongest evidence).
    site_quant, site_loc, site_meta = aggregate_precursors_to_sites(
        quant_precursor,
        loc_precursor,
        meta_precursor,
        aggregation_method=settings["aggregation_method"],
        logger=logger,
    )
    stats["n_sites_pre_mask"] = len(site_quant)

    # -- Stage 6: apply localization masking (per-run mask, or condition-aware, or none)
    strategy = settings["localization_strategy"]
    decision_table = None
    if strategy == "per_run":
        site_quant = mask_per_run(
            site_quant,
            site_loc,
            cutoff=settings["cutoff"],
            logger=logger,
        )
    elif strategy == "global_max":
        site_quant, site_loc = filter_by_global_max(
            site_quant,
            site_loc,
            cutoff=settings["cutoff"],
            logger=logger,
        )
        site_meta = site_meta.loc[site_quant.index]
    elif strategy == "condition":
        # Condition mask works on LINEAR intensity + linear loc; the sequence
        # remains: mask -> log2 -> noise floor.
        site_quant, decision_table = mask_condition_aware(
            site_quant,
            site_loc,
            condition_df=condition_df,
            classI_cutoff=settings["classI_cutoff"],
            condition_threshold=settings["condition_threshold"],
            logger=logger,
        )

    # -- Stage 7: log2 + noise floor
    site_quant = log2_transform(
        site_quant,
        apply_noise_floor=settings["noise_floor_filter"],
        logger=logger,
    )

    # -- Stage 8: drop all-NaN sites (applies to per_run and condition modes).
    if strategy != "global_max" and settings["drop_all_nan"]:
        site_quant, site_loc, site_meta = drop_all_nan_sites(
            site_quant,
            site_loc,
            site_meta,
            logger=logger,
        )
        if decision_table is not None:
            decision_table = decision_table.reindex(site_quant.index)

    stats["n_sites_post_mask"] = len(site_quant)

    # -- Resolve short-key collisions (for the human-readable label column).
    site_meta, collisions = resolve_short_keys(site_meta)
    stats["n_short_key_collisions"] = len(collisions)

    # -- Site position column is what downstream code expects; rename absolute_position
    #    to site_position and drop internals.
    site_meta = site_meta.rename(
        columns={
            "absolute_position": "site_position",
            "PTM_0_aa": "site_aa",
        }
    )
    # Restore ``_`` in gene names (they were temporarily replaced with ``#`` in prepare_psms).
    if "gene" in site_meta.columns:
        site_meta["gene"] = site_meta["gene"].astype(str).str.replace("#", "_", regex=False)

    # -- Stage 9: build AnnData
    adata = assemble_anndata(
        site_quant=site_quant,
        site_loc=site_loc,
        site_meta=site_meta,
        condition_df=condition_df,
        settings=settings,
        stats=stats,
        selectivity=selectivity,
        decision_table=decision_table,
        source_attrs=dict(data.attrs) if hasattr(data, "attrs") and data.attrs else None,
        short_key_collisions=collisions,
        version=_alphaphos_version,
        logger=logger,
    )
    return adata


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _configure_logger(verbose: bool) -> None:
    """Attach an INFO-level stderr handler when ``verbose=True``.

    Library-quiet by default. All logs still propagate through the
    ``alphaphos.preprocess.collapse`` logger for external configuration.
    """
    if not verbose:
        logger.setLevel(logging.WARNING)
        return
    if any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        # Already configured for this session.
        logger.setLevel(logging.INFO)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Pre-collapse stages: quant-column selection + top-N attribution
#
# These run BEFORE ``prepare_psms`` because they choose / prune what
# ``prepare_psms`` consumes. Kept in this module (rather than under
# ``_collapse``) because they compose engine settings with schema knowledge
# and are logically part of the entry-point wiring.
# ---------------------------------------------------------------------------


def _select_quantification_column(
    df: pd.DataFrame,
    *,
    engine: str,
    requested_level: str,
) -> tuple[pd.DataFrame, str, str]:
    """Pick the quant column and copy its values into the canonical slot.

    Walks the ``MS2 -> MS1 -> auto`` fallback chain via
    :func:`alphaphos.io.schemas.resolve_quant_column`. If a fallback was
    needed, emits a ``UserWarning`` AND logs a warning.

    Returns
    -------
    (df, chosen_col, level_used)
        Modified DataFrame (only touched if the source column was NOT
        already the canonical slot), the actual source column name, and
        the level the source column belongs to.
    """
    available = set(df.columns)
    chosen_col, level_used = resolve_quant_column(
        available, engine=engine, requested_level=requested_level
    )
    if level_used != requested_level:
        msg = (
            f"quantification_level={requested_level!r} unavailable in the input; "
            f"falling back to level={level_used!r} via column {chosen_col!r}."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
        logger.warning(msg)

    if chosen_col != COL_CANONICAL_QUANT:
        df = df.copy()
        df[COL_CANONICAL_QUANT] = df[chosen_col]

    logger.info("Using quantification column: %r (level=%s)", chosen_col, level_used)
    return df, chosen_col, level_used


def _apply_top_n_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the Spectronaut top-N over-export dedup, if the loc string is present.

    Spectronaut exports the same peptide measurement as N separate rows,
    one per candidate localization position. This filter drops the extras,
    keeping only the rows whose ``EG.PrecursorId``-encoded positions equal
    the top-N from the per-row ``EG.PTMLocalizationProbabilities`` string.
    Validated at Pearson r = 0.98 vs Spectronaut's native PTM Site Report.

    Silently no-ops when ``EG.PTMLocalizationProbabilities`` is absent
    (older Spectronaut exports, or reports that were pruned before load).
    """
    if COL_EG_PTM_LOC_PROBS not in df.columns:
        logger.info(
            "Skipping top_n_attribution: EG.PTMLocalizationProbabilities column not present."
        )
        return df

    n_before = len(df)
    df = filter_to_top_n_positions(df)
    logger.info(
        "top_n_attribution dedup: %d -> %d rows (%d dropped).",
        n_before,
        len(df),
        n_before - len(df),
    )
    return df
