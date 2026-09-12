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
import numbers
from typing import Any

import pandas as pd

try:
    import anndata as ad
except ImportError:  # pragma: no cover
    ad = None  # type: ignore[assignment]

from alphaphos._version import __version__ as _alphaphos_version
from alphaphos.constants import (
    COL_EG_PTM_LOC_PROBS,
    OBS_CONDITION,
    OBS_SAMPLE,
)
from alphaphos.preprocess._collapse.masking import (
    VALID_STRATEGIES,
    drop_all_nan_sites,
    filter_by_global_max,
    mask_condition_aware,
    mask_per_run,
)
from alphaphos.preprocess._collapse.output_format import assemble_anndata
from alphaphos.preprocess._collapse.selectivity import compute_selectivity
from alphaphos.preprocess._collapse.shared import (
    enable_verbose_logging,
    select_quantification_column,
)
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public defaults + engine dispatch
# ---------------------------------------------------------------------------


DEFAULT_COLLAPSE_SETTINGS: dict[str, Any] = {
    "search_engine": "SN",  # "SN" | "Diann" implemented; "Fragpipe" | "Peaks" raise
    # "MS2" | "MS1" | "auto".  Default "MS2" prefers ``FG.MS2Quantity``
    # (Spectronaut MS2 Fragment Ion Report).  When that column is absent
    # (as in a standard Spectronaut "Normal" report), the reader logs a
    # UserWarning and falls back to ``EG.TotalQuantity (Settings)`` -- the
    # precursor-level total intensity that ships with the Normal report.
    # None -> the engine's default level (schemas.DEFAULT_QUANT_LEVEL):
    # "MS2" for Spectronaut, "MS1" (Ms1.Translated) for DIA-NN -- alphaPhos's
    # deliberate DIA-NN choice.  "MS2" on DIA-NN gives the conventional
    # Precursor.Quantity.  See ``alphaphos.io.schemas`` for the fallback
    # chain per engine; pass ``"auto"`` explicitly to skip the warning.
    "quantification_level": None,
    # Spectronaut over-export dedup: True | False | "auto".  Spectronaut writes
    # one row per candidate localization of an ambiguous precursor, each
    # carrying the full intensity, so the extras must be removed.  DIA-NN
    # writes ONE peptidoform row per precursor-run -- nothing to dedup, and the
    # filter would only delete low-confidence peptidoforms the Class-I mask
    # handles anyway.  "auto" applies it for search_engine "SN" only.
    "top_n_attribution": "auto",
    "cutoff": 0.75,  # loc cutoff for per_run / global_max
    "classI_cutoff": 0.75,  # loc cutoff for the condition-aware mask
    "condition_threshold": 0.50,  # min fraction of Class-I reps to keep condition
    "collapse_level": "PG",  # only "PG": site keys use the first protein-group accession
    "aggregation_method": "sum",  # "sum" | "median" | "mean" | "consolidate"
    # Per-(precursor, run) Class-I gate BEFORE aggregation: a precursor adds
    # to a site's intensity in a run only if its own loc prob there reaches
    # the strategy's cutoff -- unless no precursor of the site is Class-I in
    # that run (then all are aggregated and the site-level mask decides).
    # Mirrors Spectronaut's PTM consolidation; validated on the EGF HeLa
    # series (docs/benchmark/spectronaut_native_benchmark.md §9).
    "precursor_loc_gate": True,
    "localization_strategy": "condition",  # "condition" | "per_run" | "global_max" | "wilson"
    # Only consumed when localization_strategy == "wilson": either a float in
    # [0, 1] or the string "auto" (elbow-detected from cohort size + retention
    # curve; requires n_samples >= 30).  See
    # :func:`alphaphos.preprocess.classI_wilson.apply_wilson_filter`.
    "wilson_threshold": "auto",
    "noise_floor_filter": True,  # remove log2 values in {0, 1}
    "drop_all_nan": True,  # drop sites fully NaN after the mask
}

_ALLOWED_ENGINES = ("SN", "Diann", "Fragpipe", "Peaks")
_ALLOWED_QUANT_LEVELS = (None, "MS2", "MS1", "auto")


# ---------------------------------------------------------------------------
# Settings resolution
# ---------------------------------------------------------------------------


def _check_unit_interval(name: str, value: Any, *, exclusive_low: bool = False) -> None:
    """Fail-fast check that ``value`` is a real number in [0, 1] (or (0, 1]).

    Rejects ``bool`` (an ``int`` subclass, so ``True`` would otherwise pass as
    1.0) and strings (``float("0.5")`` would silently accept ``"0.5"``).
    """
    interval = "(0, 1]" if exclusive_low else "[0, 1]"
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a number in {interval}, got {value!r}")
    v = float(value)
    in_range = (0.0 < v <= 1.0) if exclusive_low else (0.0 <= v <= 1.0)
    if not in_range:
        raise ValueError(f"{name} must be in {interval}, got {value!r}")


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
    top_n = settings["top_n_attribution"]
    if not (isinstance(top_n, bool) or top_n == "auto"):
        raise ValueError(f"top_n_attribution must be bool or 'auto', got {top_n!r}")
    for bool_key in (
        "precursor_loc_gate",
        "noise_floor_filter",
        "drop_all_nan",
    ):
        if not isinstance(settings[bool_key], bool):
            raise ValueError(f"{bool_key} must be bool, got {type(settings[bool_key]).__name__}")
    if settings["localization_strategy"] not in VALID_STRATEGIES:
        raise ValueError(
            f"localization_strategy must be one of {VALID_STRATEGIES}, "
            f"got {settings['localization_strategy']!r}"
        )
    if settings["collapse_level"] != "PG":
        raise ValueError(
            "collapse_level must be 'PG'. The former 'P' option never resolved proteins (it "
            "only kept the raw semicolon-joined group string) and has been removed; got "
            f"{settings['collapse_level']!r}"
        )
    if settings["aggregation_method"] not in ("sum", "median", "mean", "consolidate"):
        raise ValueError(
            f"aggregation_method must be one of ('sum', 'median', 'mean', 'consolidate'), "
            f"got {settings['aggregation_method']!r}"
        )
    _check_unit_interval("cutoff", settings["cutoff"])
    _check_unit_interval("classI_cutoff", settings["classI_cutoff"])
    _check_unit_interval("condition_threshold", settings["condition_threshold"], exclusive_low=True)
    wt = settings["wilson_threshold"]
    if isinstance(wt, str):
        if wt != "auto":
            raise ValueError(f"wilson_threshold string must be 'auto', got {wt!r}")
    elif isinstance(wt, bool) or not isinstance(wt, numbers.Real):
        raise ValueError(f"wilson_threshold must be a float or 'auto', got {wt!r}")
    else:
        _check_unit_interval("wilson_threshold", wt)

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
        :func:`alphaphos.io.read_spectronaut`; for DIA-NN, the output of
        :func:`alphaphos.io.read_diann` (with
        ``advanced={"search_engine": "Diann"}``). Required columns depend on
        the ``search_engine``; engines other than ``"SN"`` / ``"Diann"``
        raise ``NotImplementedError``.

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
        If True, attach a stderr handler to the ``alphaphos.preprocess.collapse``
        logger and emit INFO-level stage logs. False (default) leaves the
        logger untouched -- library-quiet unless you configure it yourself
        (a level you set, e.g. DEBUG, is never overridden).

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
        If ``search_engine`` is set to any value other than ``"SN"`` or ``"Diann"``.
    KeyError
        If required PSM columns are missing.
    """
    settings = resolve_settings(advanced)
    if verbose:
        enable_verbose_logging(logger)

    # Capture PSM lineage now: ``data`` is rebound below (quant-column copy,
    # top-N filter) and relying on pandas to propagate .attrs through those
    # operations is fragile.
    source_attrs = dict(data.attrs) if getattr(data, "attrs", None) else None

    if settings["search_engine"] not in ("SN", "Diann"):
        raise NotImplementedError(
            f"search_engine={settings['search_engine']!r} is not yet implemented. "
            f"Currently supported: 'SN' (Spectronaut), 'Diann'. Others planned: "
            f"'Fragpipe', 'Peaks'."
        )

    if settings["localization_strategy"] == "condition" and condition_df is None:
        raise ValueError(
            "localization_strategy='condition' (the default) requires condition_df. "
            "Pass a DataFrame with columns 'sample' (matching R.FileName) and "
            "'condition' (grouping replicates). If you don't have condition labels, "
            "use advanced={'localization_strategy': 'per_run'} (strict) or "
            "advanced={'localization_strategy': 'global_max'} (permissive) — "
            "these skip condition-aware masking entirely."
        )
    if condition_df is not None and not {OBS_SAMPLE, OBS_CONDITION}.issubset(condition_df.columns):
        missing = {OBS_SAMPLE, OBS_CONDITION} - set(condition_df.columns)
        raise KeyError(f"condition_df is missing required columns: {sorted(missing)}")

    stats: dict[str, Any] = {"n_psms_loaded": len(data)}

    # -- Selectivity: per-sample phospho fraction from RAW PSMs (before any filter).
    selectivity = compute_selectivity(data)

    # -- Pre-stage A: pick the quant column (fallback chain per schemas.py; warn on fallback)
    data, chosen_quant_col, quant_level_used = select_quantification_column(
        data,
        engine=settings["search_engine"],
        requested_level=settings["quantification_level"],
        logger=logger,
    )
    stats["quantification_level_requested"] = settings["quantification_level"]
    stats["quantification_column_used"] = chosen_quant_col
    stats["quantification_level_used"] = quant_level_used

    # -- Pre-stage B: top-N attribution dedup (Spectronaut over-export fix).
    # "auto" -> Spectronaut only; DIA-NN emits one peptidoform row per
    # precursor-run so there is nothing to dedup (see DEFAULT_COLLAPSE_SETTINGS).
    top_n = settings["top_n_attribution"]
    apply_top_n = (settings["search_engine"] == "SN") if top_n == "auto" else bool(top_n)
    if apply_top_n:
        data = _apply_top_n_attribution(data)
    stats["top_n_attribution_applied"] = apply_top_n
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

    strategy = settings["localization_strategy"]
    # "wilson" delegates to global_max at the precursor-mask stage; the Wilson
    # post-collapse site filter is applied after assemble_anndata (Stage 10).
    strategy_effective = "global_max" if strategy == "wilson" else strategy

    # -- Stage 5: aggregate precursors to sites
    # The QUANT matrix uses the aggregation method; the LOC matrix is always
    # max-aggregated by site (per-run cell = strongest evidence).  With
    # precursor_loc_gate the per-precursor Class-I gate uses the same cutoff
    # the strategy applies at site level afterwards.
    gate_cutoff = (
        settings["classI_cutoff"] if strategy_effective == "condition" else settings["cutoff"]
    )
    site_quant, site_loc, site_meta = aggregate_precursors_to_sites(
        quant_precursor,
        loc_precursor,
        meta_precursor,
        aggregation_method=settings["aggregation_method"],
        precursor_loc_gate=gate_cutoff if settings["precursor_loc_gate"] else None,
        logger=logger,
    )
    stats["n_sites_pre_mask"] = len(site_quant)
    stats["n_precursor_cells_gated"] = int(site_quant.attrs.get("n_precursor_cells_gated", 0))

    # -- Stage 6: apply localization masking (per-run mask, or condition-aware, or none)
    decision_table = None
    if strategy_effective == "per_run":
        site_quant = mask_per_run(
            site_quant,
            site_loc,
            cutoff=settings["cutoff"],
            logger=logger,
        )
    elif strategy_effective == "global_max":
        site_quant, site_loc = filter_by_global_max(
            site_quant,
            site_loc,
            cutoff=settings["cutoff"],
            logger=logger,
        )
        site_meta = site_meta.loc[site_quant.index]
    elif strategy_effective == "condition":
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

    # -- Stage 8: drop all-NaN sites (applies to per_run and condition modes;
    # skipped for global_max and its wilson-strategy delegate).
    if strategy_effective != "global_max" and settings["drop_all_nan"]:
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
        source_attrs=source_attrs,
        short_key_collisions=collisions,
        version=_alphaphos_version,
        logger=logger,
    )

    # -- Stage 10: Wilson site filter (only when strategy == "wilson")
    if strategy == "wilson":
        from alphaphos.preprocess.classI_wilson import apply_wilson_filter

        adata = apply_wilson_filter(adata, threshold=settings["wilson_threshold"])

    return adata


# ---------------------------------------------------------------------------
# Pre-collapse stage: top-N attribution
#
# Runs BEFORE ``prepare_psms`` because it prunes what ``prepare_psms``
# consumes.  (Quant-column selection, the other pre-stage, is shared with
# collapse_precursors -- see ``_collapse.shared``.)
# ---------------------------------------------------------------------------


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
