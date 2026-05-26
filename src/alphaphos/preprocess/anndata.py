"""Convert alphaPhos collapse output to AnnData (scverse / alphapepttools).

AnnData conventions:
  - rows of .X / .obs are SAMPLES (observations)
  - columns of .X / .var are SITES (variables/features)
  - .layers hold parallel matrices of the same shape as .X
  - .obsm / .varm hold sample- and site-level matrix annotations
  - .uns holds free-form metadata (pipeline parameters, FDR, etc.)

Our ``collapse_sites`` output is the opposite layout (sites × samples), so this
converter transposes the numeric block while preserving per-site metadata
(``META_COLS``) as ``.var`` and any sample-level metadata as ``.obs``.

In addition to the raw collapse output, ``to_anndata`` enriches ``.var`` with
phospho-unique annotations computed from the available inputs:

- **Motif flags** (when a ``kinase_sequence`` column is present): the immediate
  flanking residues (``p_minus_1``, ``p_plus_1``) and quick-motif booleans
  (``is_proline_directed``, ``is_basophilic``, ``is_acidic_motif``).
- **Site QC** (when ``loc_per_run`` is provided): ``mean_loc_prob``,
  ``max_loc_prob``, ``min_loc_prob``, ``n_samples_detected``,
  ``n_classI_samples``, ``fraction_classI``.

And ``.uns['alphaphos']`` carries pipeline provenance: alphaPhos version,
pipeline parameters (lifted from ``sites.attrs['alphaphos_pipeline']``),
processing timestamp, source paths.
"""

from __future__ import annotations

import datetime as _dt
import re

import pandas as pd

from alphaphos import __version__ as _alphaphos_version
from alphaphos.preprocess.classify import META_COLS

_KEY_RE = re.compile(
    r"^(?P<protein_group>[^~]+)~(?P<gene>[^_]*)_"
    r"(?P<aa>[A-Z])(?P<position>\d+)_M(?P<multiplicity>\d+)$"
)

# A kinase_sequence string looks like ``_AAVKRGT*S*ELLIQAA_`` (outer ``_`` is
# protein-end padding, two ``*`` bracket the modified residue). This pattern
# captures the left flank, the modified residue, and the right flank.
_KINASE_SEQ_RE = re.compile(r"^_+([A-Z_]*?)\*([A-Z])\*([A-Z_]*?)_+$")

# Acidic residues for the CK1/CK2 motif check.
_ACIDIC = ("D", "E")
# Basophilic residues for the AGC-kinase motif check (PKA/PKC/AKT).
_BASOPHILIC = ("R", "K")


def _derive_motif_flags(kinase_sequence: str | None) -> dict[str, object]:
    """Parse a kinase_sequence string into motif-position columns.

    The kinase_sequence format (produced by
    ``PeptideCollapse._create_kinase_sequence``) is::

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
    empty = {
        "p_minus_1": None,
        "p_plus_1": None,
        "is_proline_directed": None,
        "is_basophilic": None,
        "is_acidic_motif": None,
    }
    if not isinstance(kinase_sequence, str) or not kinase_sequence:
        return empty
    # PeptideCollapse error-string sentinels — keep flags empty for those rows
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
    """
    # Align to the requested site index and sample columns; missing entries
    # become NaN, which the aggregations handle natively.
    loc = loc_per_run.reindex(index=site_index, columns=sample_cols)
    n_total = len(sample_cols)
    return pd.DataFrame(
        {
            "n_samples_detected": loc.notna().sum(axis=1).astype(int),
            "mean_loc_prob": loc.mean(axis=1).astype(float),
            "max_loc_prob": loc.max(axis=1).astype(float),
            "min_loc_prob": loc.min(axis=1).astype(float),
            "n_classI_samples": (loc >= classI_cutoff).sum(axis=1).astype(int),
            "fraction_classI": ((loc >= classI_cutoff).sum(axis=1) / max(n_total, 1)).astype(float),
        },
        index=site_index,
    )


def to_anndata(
    sites: pd.DataFrame,
    *,
    loc_per_run: pd.DataFrame | None = None,
    condition_df: pd.DataFrame | None = None,
    decision_table: pd.DataFrame | None = None,
    main_layer: str = "intensity_log2",
    classI_cutoff: float = 0.75,
):
    """Convert ``collapse_sites`` output to an AnnData object.

    Shape: ``adata.X`` is ``(n_samples × n_sites)`` (samples as observations,
    sites as variables) — the scverse and alphapepttools convention.

    Parameters
    ----------
    sites
        Wide DataFrame from ``collapse_sites``. Rows are sites
        (``PTM_Collapse_key``), columns are sample names + ``META_COLS``.
    loc_per_run
        Optional ``(sites × samples)`` DataFrame of per-(site, run)
        localization probabilities (from ``pc.site_localization_per_run`` or
        the second return of ``collapse_sites``). If provided, stored as
        ``adata.layers["localization"]`` AND used to compute per-site QC
        columns in ``adata.var``.
    condition_df
        Optional DataFrame mapping sample names to condition labels (and any
        extra columns). Must contain ``"sample"`` and ``"condition"`` columns;
        extra columns are joined into ``adata.obs`` as-is.
    decision_table
        Optional ``(sites × conditions)`` DataFrame from the condition-aware
        mask. If provided, stored as ``adata.uns["classI_decision_table"]``.
    main_layer
        Name for the principal data layer (also stored under ``adata.X``).
        Default ``"intensity_log2"`` reflects the log2 transform applied by
        PeptideCollapse.
    classI_cutoff
        Threshold for the ``n_classI_samples`` / ``fraction_classI`` QC
        columns. Default 0.75 matches Spectronaut's Class I convention.

    Returns
    -------
    anndata.AnnData
        Ready for downstream alphapepttools / scanpy / scverse workflows.
        ``obs.index`` is sample names, ``var.index`` is ``PTM_Collapse_key``.

    Notes
    -----
    Parses the ``PTM_Collapse_key`` (format
    ``{ProteinGroup}~{Gene}_{S|T|Y}{position}_M{multiplicity}``) and exposes
    its components as first-class columns in ``adata.var``:
    ``protein_group_id``, ``gene_first``, ``site_aa``, ``site_position``,
    ``multiplicity``. Original ``META_COLS`` columns are also preserved.

    Phospho-specific annotations added when their inputs are available:

    - ``kinase_sequence`` column → motif flags ``p_minus_1``, ``p_plus_1``,
      ``is_proline_directed``, ``is_basophilic``, ``is_acidic_motif``.
    - ``loc_per_run`` parameter → site-QC columns ``n_samples_detected``,
      ``mean_loc_prob``, ``max_loc_prob``, ``min_loc_prob``,
      ``n_classI_samples``, ``fraction_classI``.

    Pipeline provenance is written to ``adata.uns['alphaphos']``:
    package version, pipeline parameters (lifted from
    ``sites.attrs['alphaphos_pipeline']``), processing timestamp, and any
    other ``sites.attrs`` keys.

    Examples
    --------
    >>> from alphaphos.io import read_spectronaut
    >>> from alphaphos.preprocess import collapse_sites, to_anndata
    >>> df = read_spectronaut("report.parquet")
    >>> sites, loc, decision = collapse_sites(
    ...     df, condition_df=condition_df, return_decision_table=True,
    ...     fasta_path="proteome.fasta",  # auto-enables kinase windows
    ... )
    >>> adata = to_anndata(
    ...     sites, loc_per_run=loc,
    ...     condition_df=condition_df, decision_table=decision,
    ... )
    """
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError(
            "anndata is required. Install via `pip install anndata` or it "
            "is already a core dependency of alphaPhos."
        ) from exc

    sample_cols = [c for c in sites.columns if c not in META_COLS]
    if "PTM_Collapse_key" not in sites.columns:
        raise ValueError(
            "sites must contain 'PTM_Collapse_key' column — pass the output "
            "of collapse_sites() directly."
        )

    # site_index: per-site identifier (PTM_Collapse_key)
    site_index = sites["PTM_Collapse_key"].astype(str).values

    # X: samples × sites (transpose the numeric block)
    X = sites[sample_cols].astype(float).T.values  # (n_samples, n_sites)

    # obs: per-sample metadata (sample name as index)
    obs = pd.DataFrame(index=pd.Index(sample_cols, name="sample"))
    if condition_df is not None:
        required = {"sample", "condition"}
        missing = required - set(condition_df.columns)
        if missing:
            raise ValueError(
                f"condition_df must contain columns {sorted(required)}; missing: {sorted(missing)}"
            )
        obs = obs.join(
            condition_df.drop_duplicates("sample").set_index("sample"),
            how="left",
        )

    # var: per-site metadata
    meta_present = [c for c in META_COLS if c in sites.columns and c != "PTM_Collapse_key"]
    var = sites[meta_present].copy() if meta_present else pd.DataFrame()
    var.index = pd.Index(site_index, name="PTM_Collapse_key")

    # Parse PTM_Collapse_key components into first-class var columns
    parsed = sites["PTM_Collapse_key"].astype(str).str.extract(_KEY_RE)
    var["protein_group_id"] = parsed["protein_group"].values
    var["gene_first"] = parsed["gene"].values
    var["site_aa"] = parsed["aa"].values
    var["site_position"] = pd.to_numeric(parsed["position"], errors="coerce").values
    var["multiplicity"] = pd.to_numeric(parsed["multiplicity"], errors="coerce").values

    # Motif flags from kinase_sequence (Tier 1)
    if "kinase_sequence" in var.columns:
        flags = var["kinase_sequence"].map(_derive_motif_flags).tolist()
        flags_df = pd.DataFrame(flags, index=var.index)
        for col in (
            "p_minus_1",
            "p_plus_1",
            "is_proline_directed",
            "is_basophilic",
            "is_acidic_motif",
        ):
            var[col] = flags_df[col].values

    # Site QC from loc_per_run (Tier 2)
    if loc_per_run is not None:
        qc = _compute_site_qc(loc_per_run, var.index, sample_cols, classI_cutoff=classI_cutoff)
        for col in qc.columns:
            var[col] = qc[col].values

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.layers[main_layer] = X.copy()

    # Localization layer (if provided): align to samples × sites
    if loc_per_run is not None:
        loc_aligned = loc_per_run.reindex(index=site_index, columns=sample_cols)
        adata.layers["localization"] = loc_aligned.astype(float).T.values

    # Decision table (uns; preserved with both row and column labels)
    if decision_table is not None:
        adata.uns["classI_decision_table"] = decision_table

    # Pipeline provenance (Tier 4)
    pipeline_params = (
        dict(sites.attrs["alphaphos_pipeline"])
        if "alphaphos_pipeline" in getattr(sites, "attrs", {})
        else {}
    )
    adata.uns["alphaphos"] = {
        "version": _alphaphos_version,
        "main_layer": main_layer,
        "n_sites": int(adata.n_vars),
        "n_samples": int(adata.n_obs),
        "pipeline_params": pipeline_params,
        "processing_timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    if hasattr(sites, "attrs") and sites.attrs:
        adata.uns["source_attrs"] = {
            k: v for k, v in sites.attrs.items() if k != "alphaphos_pipeline"
        }

    return adata
