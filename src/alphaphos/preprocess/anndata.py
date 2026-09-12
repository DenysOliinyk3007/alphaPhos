"""Build an ``AnnData`` from a (sites × samples) matrix that was NOT produced
by ``collapse_sites`` -- the escape hatch for externally collapsed data.

:func:`alphaphos.collapse_sites` and :func:`alphaphos.collapse_precursors`
already return a fully packaged ``AnnData``.  Use :func:`to_anndata` when the
site matrix comes from somewhere else (a legacy pipeline, Spectronaut's own
PTM site report, an R export) and its row index follows the alphaPhos site-key
convention ``Protein|Gene|<S|T|Y><position>|M<multiplicity>``.  The result has
the same ``.var`` / ``.obs`` / ``.layers`` / ``.uns`` layout the rest of
alphaPhos expects, so filtering, imputation, DE and enrichment run unchanged.

Populated slots
---------------

- ``.X`` and ``.layers[main_layer]`` -- the transposed matrix (samples × sites).
- ``.var`` -- key components (``protein_group_id``, ``gene``, ``site_aa``,
  ``site_position``, ``multiplicity``, ``short_key``, ``pg_key``), every
  column of ``var_meta``, and

  * **motif flags** when ``var_meta`` carries a ``kinase_sequence`` column
    (``p_minus_1``, ``p_plus_1``, ``is_proline_directed``, ``is_basophilic``,
    ``is_acidic_motif``; see :func:`_derive_motif_flags`);
  * **site QC** when ``loc_per_run`` is given (``n_samples_detected``,
    ``mean/max/min_loc_prob``, ``n_classI_samples``, ``fraction_classI``,
    ``classI_wilson_lb``) plus ``.layers["localization"]``.

- ``.obs`` -- ``condition_df`` joined by sample id.
- ``.uns["alphaphos"]`` -- version, ``pipeline_params``, processing timestamp,
  optional ``classI_decision_table``; ``.uns["source_attrs"]`` from ``psm_attrs``.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
from typing import Any

import pandas as pd

from alphaphos._version import __version__ as _alphaphos_version
from alphaphos.constants import (
    LAYER_INTENSITY_LOG2,
    LAYER_LOCALIZATION,
    OBS_CONDITION,
    OBS_SAMPLE,
    UNS_ALPHAPHOS,
    UNS_SOURCE_ATTRS,
    VAR_FULL_KEY,
)

logger = logging.getLogger(__name__)

# Current alphaPhos site key: ``Protein|Gene|<AA><pos>|M<mult>``.  ``|`` never
# occurs inside any field (see ``_collapse/keys.py``); the gene field may be
# empty.
_KEY_RE = re.compile(
    r"^(?P<protein_group>[^|]+)\|(?P<gene>[^|]*)\|"
    r"(?P<aa>[A-Z])(?P<position>\d+)\|M(?P<multiplicity>\d+)$"
)

# A kinase_sequence string looks like ``_AAVKRGT*S*ELLIQAA_`` (outer ``_`` is
# protein-end padding, two ``*`` bracket the modified residue). This pattern
# captures the left flank, the modified residue, and the right flank.
_KINASE_SEQ_RE = re.compile(r"^_+([A-Z_]*?)\*([A-Z])\*([A-Z_]*?)_+$")

# Acidic residues for the CK1/CK2 motif check.
_ACIDIC = ("D", "E")
# Basophilic residues for the AGC-kinase motif check (PKA/PKC/AKT).
_BASOPHILIC = ("R", "K")

_MOTIF_COLUMNS = (
    "p_minus_1",
    "p_plus_1",
    "is_proline_directed",
    "is_basophilic",
    "is_acidic_motif",
)


def _derive_motif_flags(kinase_sequence: str | None) -> dict[str, object]:
    """Parse a kinase_sequence string into motif-position columns.

    The kinase_sequence format (produced by
    :func:`alphaphos.add_kinase_windows`) is::

        ``_<left_flank>*<modified_aa>*<right_flank>_``

    with leading/trailing ``_`` chars padding protein N/C termini. Returns
    a dict with the keys:

    - ``p_minus_1`` / ``p_plus_1``: residues at -1 / +1 around the phospho
      (``None`` if the position falls beyond the protein terminus)
    - ``is_proline_directed``: ``p_plus_1 == 'P'`` (CMGC/CDK/MAPK consensus)
    - ``is_basophilic``: any of residues at -2, -3, -5 is R or K
      (AGC consensus, e.g. PKA R-X-X-S/T, AKT R-X-R-X-X-S/T)
    - ``is_acidic_motif``: ``p_plus_1`` ∈ D/E (CK1) OR ``p_plus_3`` ∈ D/E (CK2)

    Returns ``{flag: None}`` for invalid/error/missing inputs so the columns
    still align across the var index.
    """
    empty: dict[str, object] = dict.fromkeys(_MOTIF_COLUMNS)
    if not isinstance(kinase_sequence, str) or not kinase_sequence:
        return empty
    # Error-string sentinels from annotation -- keep flags empty for those rows
    if kinase_sequence.startswith(
        ("FASTA_ERROR:", "POSITION_ERROR:", "SEQUENCE_MISMATCH:", "PARSING_ERROR:")
    ):
        return empty

    m = _KINASE_SEQ_RE.match(kinase_sequence)
    if m is None:
        return empty

    left, _modified_aa, right = m.group(1), m.group(2), m.group(3)

    # offset is 1-indexed distance from the modified residue. left[-k] reads
    # backward (closest residue last); right[k-1] reads forward.
    def _left(offset: int) -> str | None:
        if offset < 1 or offset > len(left):
            return None
        ch = left[-offset]
        return None if ch == "_" else ch

    def _right(offset: int) -> str | None:
        if offset < 1 or offset > len(right):
            return None
        ch = right[offset - 1]
        return None if ch == "_" else ch

    p_minus_1 = _left(1)
    p_plus_1 = _right(1)
    p_plus_3 = _right(3)

    # Basophilic: any of -2/-3/-5 is R/K (covers PKA, PKC, AKT consensus)
    basophilic_positions = [_left(k) for k in (2, 3, 5)]
    is_basophilic = any(r in _BASOPHILIC for r in basophilic_positions if r is not None)

    # Acidic motif: CK1 (p+1 ∈ D/E) or CK2 (p+3 ∈ D/E)
    is_acidic_motif = (p_plus_1 in _ACIDIC) or (p_plus_3 in _ACIDIC)

    return {
        "p_minus_1": p_minus_1,
        "p_plus_1": p_plus_1,
        "is_proline_directed": p_plus_1 == "P",
        "is_basophilic": is_basophilic,
        "is_acidic_motif": is_acidic_motif,
    }


def _compute_site_qc(
    loc_per_run: pd.DataFrame,
    site_index: pd.Index,
    sample_cols: list[str],
    classI_cutoff: float = 0.75,
) -> pd.DataFrame:
    """Compute per-site QC metrics from the (sites × samples) loc-prob matrix.

    Returns a DataFrame indexed by ``site_index`` with columns:

    - ``n_samples_detected``: count of samples where the site has a loc value
    - ``mean_loc_prob``, ``max_loc_prob``, ``min_loc_prob``: across samples
    - ``n_classI_samples``: count of samples with loc ≥ classI_cutoff
    - ``fraction_classI``: n_classI_samples / total_samples
    - ``classI_wilson_lb``: Jeffreys / Wilson-equivalent 95% lower confidence
      bound on the per-site Class-I rate — sample-size-aware correction to
      ``n_classI_samples / n_samples_detected`` (see
      :mod:`alphaphos.preprocess.classI_wilson`).
    """
    from alphaphos.preprocess.classI_wilson import wilson_lower_bound

    # Align to the requested site index and sample columns; missing entries
    # become NaN, which the aggregations handle natively.
    loc = loc_per_run.reindex(index=site_index, columns=sample_cols)
    n_total = len(sample_cols)
    n_samples_detected = loc.notna().sum(axis=1).astype(int)
    n_classI_samples = (loc >= classI_cutoff).sum(axis=1).astype(int)
    wilson_lb = wilson_lower_bound(n_classI_samples.to_numpy(), n_samples_detected.to_numpy())
    return pd.DataFrame(
        {
            "n_samples_detected": n_samples_detected,
            "mean_loc_prob": loc.mean(axis=1).astype(float),
            "max_loc_prob": loc.max(axis=1).astype(float),
            "min_loc_prob": loc.min(axis=1).astype(float),
            "n_classI_samples": n_classI_samples,
            "fraction_classI": (n_classI_samples / max(n_total, 1)).astype(float),
            "classI_wilson_lb": wilson_lb.astype(float),
        },
        index=site_index,
    )


def _parse_site_keys(site_index: pd.Index) -> pd.DataFrame:
    """Split ``Protein|Gene|<AA><pos>|M<mult>`` keys into var columns.

    Keys that don't match the convention get NaN in every derived column
    (and a single summary warning) rather than raising -- the matrix is still
    usable, only key-derived annotations are missing for those rows.
    """
    parsed = site_index.to_series().str.extract(_KEY_RE)
    n_bad = int(parsed["protein_group"].isna().sum())
    if n_bad:
        logger.warning(
            "%d / %d site keys do not match 'Protein|Gene|<AA><pos>|M<mult>' "
            "(e.g. %r); their key-derived var columns are NaN.",
            n_bad,
            len(site_index),
            site_index[parsed["protein_group"].isna().to_numpy()][0],
        )
    short_keys: list[str | None] = []
    pg_keys: list[str | None] = []
    for pg, gene, aa, pos, mult in zip(
        parsed["protein_group"],
        parsed["gene"],
        parsed["aa"],
        parsed["position"],
        parsed["multiplicity"],
        strict=True,
    ):
        if pd.isna(pg):
            short_keys.append(None)
            pg_keys.append(None)
            continue
        short_keys.append(f"{gene}|{aa}{pos}|M{mult}")
        pg_keys.append(f"{pg}|{aa}{pos}|M{mult}")
    return pd.DataFrame(
        {
            "short_key": short_keys,
            "pg_key": pg_keys,
            "protein_group_id": parsed["protein_group"].to_numpy(),
            "gene": parsed["gene"].to_numpy(),
            "site_aa": parsed["aa"].to_numpy(),
            "site_position": pd.to_numeric(parsed["position"], errors="coerce").to_numpy(),
            "multiplicity": pd.to_numeric(parsed["multiplicity"], errors="coerce").to_numpy(),
        },
        index=site_index,
    )


def to_anndata(
    sites: pd.DataFrame,
    *,
    var_meta: pd.DataFrame | None = None,
    loc_per_run: pd.DataFrame | None = None,
    condition_df: pd.DataFrame | None = None,
    decision_table: pd.DataFrame | None = None,
    main_layer: str = LAYER_INTENSITY_LOG2,
    classI_cutoff: float = 0.75,
    pipeline_params: dict[str, Any] | None = None,
    psm_attrs: dict[str, Any] | None = None,
):
    """Package an externally collapsed (sites × samples) matrix as an ``AnnData``.

    Parameters
    ----------
    sites
        Numeric DataFrame, rows = sites, columns = samples.  The index must be
        unique alphaPhos site keys ``Protein|Gene|<S|T|Y><pos>|M<mult>``
        (log2 intensities are expected -- pass ``main_layer`` accordingly if
        not).  Non-numeric columns raise; put per-site annotations in
        ``var_meta`` instead.
    var_meta
        Optional per-site annotations indexed like ``sites`` (e.g. a
        ``kinase_sequence`` column from :func:`alphaphos.add_kinase_windows`).
        Every column is forwarded to ``adata.var``; ``kinase_sequence``
        additionally triggers the motif flags.
    loc_per_run
        Optional ``(sites × samples)`` localization-probability matrix.  Stored
        as ``adata.layers["localization"]`` and used for the site-QC columns.
    condition_df
        Optional sample metadata with ``sample`` + ``condition`` columns
        (extras are joined into ``adata.obs``).  Duplicated sample rows are
        collapsed to the first occurrence.
    decision_table
        Optional ``(sites × conditions)`` Class-I fraction table; stored at
        ``adata.uns["alphaphos"]["classI_decision_table"]``.
    main_layer
        Name of the principal layer (also mirrored in ``adata.X``).
    classI_cutoff
        Threshold for the ``n_classI_samples`` / ``fraction_classI`` columns.
    pipeline_params
        Free-form provenance dict for ``adata.uns["alphaphos"]["pipeline_params"]``.
        Defaults to ``sites.attrs["alphaphos_pipeline"]`` when present.
    psm_attrs
        PSM-level lineage (``read_spectronaut(...).attrs``) for
        ``adata.uns["source_attrs"]``.  Falls back to the non-pipeline keys of
        ``sites.attrs``.

    Returns
    -------
    anndata.AnnData
        ``obs.index`` = sample ids, ``var.index`` = site keys (named
        ``full_key``), same layout as :func:`alphaphos.collapse_sites` output.

    Raises
    ------
    ValueError
        If ``sites`` is empty, has non-numeric columns, or a non-unique index;
        if ``condition_df`` lacks ``sample`` / ``condition``; if a
        ``var_meta`` column clashes with a key-derived column.

    Examples
    --------
    >>> sites = pd.DataFrame(
    ...     {"s1": [20.1, 18.3], "s2": [20.4, np.nan]},
    ...     index=["P31749|AKT1|S473|M1", "P31749|AKT1|T308|M1"],
    ... )
    >>> adata = to_anndata(sites, condition_df=cond_df)   # doctest: +SKIP
    """
    import anndata as ad

    if not isinstance(sites, pd.DataFrame) or sites.empty:
        raise ValueError("sites must be a non-empty (sites x samples) DataFrame.")
    non_numeric = [c for c in sites.columns if not pd.api.types.is_numeric_dtype(sites[c])]
    if non_numeric:
        raise ValueError(
            f"sites must contain only numeric sample columns; non-numeric: {non_numeric[:5]}. "
            "Pass per-site annotations via var_meta= instead."
        )
    if not sites.index.is_unique:
        raise ValueError("sites.index (site keys) must be unique.")

    site_index = pd.Index(sites.index.astype(str), name=VAR_FULL_KEY)
    sample_cols = [str(c) for c in sites.columns]
    X = sites.to_numpy(dtype=float).T  # (n_samples, n_sites)

    # ---- obs -------------------------------------------------------------
    obs = pd.DataFrame(index=pd.Index(sample_cols, name=OBS_SAMPLE))
    if condition_df is not None:
        required = {OBS_SAMPLE, OBS_CONDITION}
        missing = required - set(condition_df.columns)
        if missing:
            raise ValueError(
                f"condition_df must contain columns {sorted(required)}; missing: {sorted(missing)}"
            )
        cdf = condition_df.copy()
        cdf[OBS_SAMPLE] = cdf[OBS_SAMPLE].astype(str)
        obs = obs.join(cdf.drop_duplicates(OBS_SAMPLE).set_index(OBS_SAMPLE), how="left")

    # ---- var -------------------------------------------------------------
    var = _parse_site_keys(site_index)

    if var_meta is not None:
        vm = var_meta.copy()
        vm.index = vm.index.astype(str)
        clash = [c for c in vm.columns if c in var.columns]
        if clash:
            raise ValueError(
                f"var_meta column(s) {clash} clash with key-derived var columns; rename them."
            )
        extra = vm.reindex(site_index)
        for col in extra.columns:
            var[col] = extra[col].to_numpy()

    if "kinase_sequence" in var.columns:
        flags = pd.DataFrame(
            [_derive_motif_flags(s) for s in var["kinase_sequence"]], index=site_index
        )
        for col in _MOTIF_COLUMNS:
            var[col] = flags[col].to_numpy()

    loc_aligned: pd.DataFrame | None = None
    if loc_per_run is not None:
        loc = loc_per_run.copy()
        loc.index = loc.index.astype(str)
        loc.columns = loc.columns.astype(str)
        loc_aligned = loc.reindex(index=site_index, columns=sample_cols)
        qc = _compute_site_qc(loc_aligned, site_index, sample_cols, classI_cutoff=classI_cutoff)
        for col in qc.columns:
            var[col] = qc[col].to_numpy()

    # ---- assemble --------------------------------------------------------
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers[main_layer] = X.copy()
    if loc_aligned is not None:
        adata.layers[LAYER_LOCALIZATION] = loc_aligned.to_numpy(dtype=float).T

    sites_attrs = dict(getattr(sites, "attrs", {}) or {})
    if pipeline_params is None:
        pipeline_params = dict(sites_attrs.get("alphaphos_pipeline", {}) or {})
    ns: dict[str, Any] = {
        "version": _alphaphos_version,
        "main_layer": main_layer,
        "n_sites": int(adata.n_vars),
        "n_samples": int(adata.n_obs),
        "pipeline_params": dict(pipeline_params),
        "processing_timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    if decision_table is not None:
        ns["classI_decision_table"] = decision_table
    adata.uns[UNS_ALPHAPHOS] = ns

    if psm_attrs is not None:
        adata.uns[UNS_SOURCE_ATTRS] = dict(psm_attrs)
    else:
        source = {k: v for k, v in sites_attrs.items() if k != "alphaphos_pipeline"}
        if source:
            adata.uns[UNS_SOURCE_ATTRS] = source

    return adata
