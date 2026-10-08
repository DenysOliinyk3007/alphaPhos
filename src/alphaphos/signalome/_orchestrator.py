"""Single-call orchestrator: prediction matrix -> full signalome result.

Chains the six stages of the signalome pipeline behind one function so
callers who don't need per-stage control don't have to wire the pieces
manually.

Users who want per-stage control should call the underlying functions
directly (see :mod:`alphaphos.signalome.clustering`,
:mod:`~.assignments`, :mod:`~.protein_resolution`, :mod:`~.modules`,
:mod:`~.network`, :mod:`~.expanded`).
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from alphaphos.signalome.assignments import build_module_assignments
from alphaphos.signalome.clustering import (
    SignalomeClusteringResult,
    cluster_sites,
)
from alphaphos.signalome.constants import (
    DEFAULT_FALLBACK_THRESHOLD,
    DEFAULT_MAX_APPROX_SAMPLES_PER_CLUSTER,
    DEFAULT_MAX_EXACT_SITES,
    DEFAULT_MAX_MODULES,
    DEFAULT_MIN_TOP_SITES,
    DEFAULT_NETWORK_CORRELATION_THRESHOLD,
    DEFAULT_PRIMARY_THRESHOLD,
    DEFAULT_SUBSTRATE_SUPPORT_CUTOFF,
    MAX_SUBSTRATE_DENSITY_WARN,
    NETWORK_POLICY_SIGNED,
    SCORING_MODE_AUTO,
)
from alphaphos.signalome.expanded import build_expanded_table
from alphaphos.signalome.modules import build_module_table
from alphaphos.signalome.network import KinaseNetwork, build_kinase_network
from alphaphos.signalome.protein_resolution import (
    derive_protein_modules,
)

if TYPE_CHECKING:
    import anndata as ad

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SignalomeResult:
    """End-to-end signalome pipeline output.

    Attributes
    ----------
    site_assignments
        Per-site table with module_id, top_kinase, and tie diagnostics.
        Output of :func:`build_module_assignments`.
    protein_modules
        pd.Series indexed by protein, values are module IDs.  Output of
        :func:`derive_protein_modules`.
    module_table
        (modules x kinases) percent-share table.  Output of
        :func:`build_module_table`.
    network
        Kinase-kinase network with edges + nodes + candidate correlations.
        Output of :func:`build_kinase_network`.
    expanded
        Denormalized per-focal-kinase view.  Output of
        :func:`build_expanded_table`.
    clustering
        Clustering diagnostics from :func:`cluster_sites` -- module-count
        selection scores, mode used, provenance.
    provenance
        Full pipeline provenance dict (settings + counts + version).
    """

    site_assignments: pd.DataFrame
    protein_modules: pd.Series
    module_table: pd.DataFrame
    network: KinaseNetwork
    expanded: pd.DataFrame
    clustering: SignalomeClusteringResult
    provenance: dict[str, object] = field(default_factory=dict)


def build_signalome(
    prediction_matrix: pd.DataFrame,
    *,
    site_to_protein: pd.Series | None = None,
    site_metadata: pd.DataFrame | None = None,
    kinase_substrates: Mapping[str, Sequence[str]] | None = None,
    substrate_support_cutoff: float = DEFAULT_SUBSTRATE_SUPPORT_CUTOFF,
    kinase_order: Sequence[str] | None = None,
    requested_module_count: int | None = None,
    max_modules: int = DEFAULT_MAX_MODULES,
    primary_threshold: float = DEFAULT_PRIMARY_THRESHOLD,
    fallback_threshold: float = DEFAULT_FALLBACK_THRESHOLD,
    scoring_mode: str = SCORING_MODE_AUTO,
    max_exact_sites: int = DEFAULT_MAX_EXACT_SITES,
    max_samples_per_cluster: int = DEFAULT_MAX_APPROX_SAMPLES_PER_CLUSTER,
    network_correlation_threshold: float = DEFAULT_NETWORK_CORRELATION_THRESHOLD,
    network_policy: str = NETWORK_POLICY_SIGNED,
    assignment_policy: str = "cutoff_binary",
    min_module_share_percent: float | None = None,
    seed: int = 0,
) -> SignalomeResult:
    """Run the full signalome pipeline in one call.

    Parameters
    ----------
    prediction_matrix
        (n_sites, n_kinases) DataFrame with a unique site index.  Cells are
        per-site kinase prediction values **on a [0, 1] scale** -- build it
        with :func:`prediction_matrix_from_adata` (Yaffe percentile / 100,
        restricted to kinases that have confident substrates among your
        sites), or supply PhosR-style kinase-substrate scores.  Raw log2 PWM
        scores work mechanically but ``substrate_support_cutoff`` then has
        no meaning and all 300+ kinases produce a near-uniform module table;
        a warning is emitted when values fall outside [0, 1] or when the
        cutoff admits more than 30 % of cells.  Rows that are entirely NaN
        (sites the kinase library rejected) get ``module_id`` 0.
    site_to_protein
        Optional site_key -> protein Series.  If ``None``, derived from
        ``prediction_matrix.index`` (assumed alphaPhos
        ``Protein|Gene|Site|Mult`` format).
    site_metadata
        Optional per-site metadata DataFrame with the columns
        ``site_key``, ``display_id``, ``gene_symbol``, ``site``,
        ``protein_accession``, ``isoform_id``.  If ``None``, extracted
        from ``prediction_matrix.index``.
    kinase_substrates
        ``{kinase: [substrate_site_ids]}`` reference network.  If ``None``,
        derived from ``prediction_matrix`` itself by thresholding at
        ``substrate_support_cutoff`` (each kinase's substrates =
        sites whose score for that kinase exceeds the cutoff).
    substrate_support_cutoff
        Score cutoff used when deriving ``kinase_substrates`` from the
        prediction matrix itself.  Default 0.5.
    kinase_order
        Explicit kinase-column order.  If ``None``, uses the
        ``prediction_matrix`` columns in their natural order.
    requested_module_count
        Skip the threshold-based selection and use this ``k`` directly.
    max_modules
        Upper bound on candidate ``k``.  Default 10.
    primary_threshold, fallback_threshold
        Median-cluster-correlation thresholds.  Defaults 0.5 / 0.1.
    scoring_mode
        ``"auto"`` (default), ``"exact"``, or ``"sampled"`` -- see
        :mod:`alphaphos.signalome.clustering` for the scale-aware
        module-count-selection backends.
    max_exact_sites, max_samples_per_cluster
        Scaling knobs.  Defaults 5000 / 200.
    network_correlation_threshold, network_policy
        Kinase-kinase network edge cutoff + policy
        (``"signed"``, ``"positive_only"``, ``"absolute"``).  Defaults
        0.5 / ``"signed"``.
    assignment_policy
        ``"cutoff_binary"`` (default) or ``"weighted_top"`` for the
        module-table + expanded-table.
    min_module_share_percent
        Share (in %) a kinase needs in a module to count it as "regulated" in
        the expanded view.  ``None`` (default) = ``max(1.0, 200 / n_kinases)``,
        i.e. at least twice the uniform share, so the cutoff stays meaningful
        with hundreds of kinases (PhosPy's fixed 1.0 is barely above uniform
        at 311 kinases).
    seed
        RNG seed for the sampled scoring path.
    """
    prediction_matrix = _validate_prediction_matrix(prediction_matrix)
    # 1) Site -> protein mapping.
    if site_to_protein is None:
        site_to_protein = _default_site_to_protein(prediction_matrix)
    # 2) Site metadata.
    if site_metadata is None:
        site_metadata = _default_site_metadata(prediction_matrix)
    # 3) Kinase substrates (fallback: threshold the prediction matrix itself).
    if kinase_substrates is None:
        kinase_substrates = _default_kinase_substrates(
            prediction_matrix, cutoff=float(substrate_support_cutoff)
        )
    # 4) Kinase order.
    if kinase_order is None:
        kinase_order = list(prediction_matrix.columns.astype(str))
    kinase_order = list(kinase_order)
    if min_module_share_percent is None:
        min_module_share_percent = max(1.0, 200.0 / max(len(kinase_order), 1))

    # 5) Cluster sites -> module labels + selection diagnostics.
    clustering = cluster_sites(
        prediction_matrix,
        requested_module_count=requested_module_count,
        max_modules=max_modules,
        primary_threshold=primary_threshold,
        fallback_threshold=fallback_threshold,
        scoring_mode=scoring_mode,
        max_exact_sites=max_exact_sites,
        max_samples_per_cluster=max_samples_per_cluster,
        seed=seed,
    )
    site_cluster_series = pd.Series(
        clustering.labels, index=clustering.site_index, name="site_cluster"
    )
    # 6) Protein-level module assignment via cluster-signature grouping.
    protein_modules = derive_protein_modules(
        site_clusters=site_cluster_series, site_to_protein=site_to_protein
    )
    # 7) Site-level module assignments (broadcast protein module to sites).
    site_assignments = build_module_assignments(
        prediction_matrix=prediction_matrix,
        site_to_protein=site_to_protein,
        protein_modules=protein_modules,
        site_metadata=site_metadata,
    )
    # 8) Module x kinase table.
    module_table = build_module_table(
        module_assignments=site_assignments,
        kinase_substrates=kinase_substrates,
        kinase_order=kinase_order,
        assignment_policy=assignment_policy,
    )
    # 9) Kinase-kinase network.
    network = build_kinase_network(
        prediction_matrix=prediction_matrix,
        kinase_order=kinase_order,
        kinase_substrates=kinase_substrates,
        threshold=network_correlation_threshold,
        network_policy=network_policy,
    )
    # 10) Expanded per-focal-kinase view.
    expanded = build_expanded_table(
        module_assignments=site_assignments,
        module_table=module_table,
        network_edges=network.edges,
        kinase_substrates=kinase_substrates,
        kinase_order=kinase_order,
        assignment_policy=assignment_policy,
        min_module_share_percent=float(min_module_share_percent),
    )

    provenance = {
        "n_sites": int(prediction_matrix.shape[0]),
        "n_sites_all_nan": int(clustering.provenance.get("n_sites_all_nan", 0)),
        "n_kinases": int(prediction_matrix.shape[1]),
        "n_proteins": int(protein_modules.size),
        "n_proteins_assigned": int((protein_modules > 0).sum()),
        "n_modules": int(protein_modules[protein_modules > 0].nunique()),
        "module_count": int(clustering.module_count),
        "selection_reason": clustering.selection.selection_reason,
        "scoring_mode_used": clustering.selection.scoring_mode_used,
        "primary_threshold": float(primary_threshold),
        "fallback_threshold": float(fallback_threshold),
        "network_correlation_threshold": float(network_correlation_threshold),
        "network_policy": network_policy,
        "assignment_policy": assignment_policy,
        "substrate_support_cutoff": float(substrate_support_cutoff),
        "substrate_density": _substrate_density(prediction_matrix, float(substrate_support_cutoff)),
        "min_module_share_percent": float(min_module_share_percent),
        "seed": int(seed),
        "n_network_edges": int(network.edges.shape[0]),
        "n_expanded_rows": int(expanded.shape[0]),
    }
    return SignalomeResult(
        site_assignments=site_assignments,
        protein_modules=protein_modules,
        module_table=module_table,
        network=network,
        expanded=expanded,
        clustering=clustering,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Input construction + validation
# ---------------------------------------------------------------------------


def prediction_matrix_from_adata(
    adata: ad.AnnData,
    *,
    pool: Literal["ser_thr", "tyrosine"] = "ser_thr",
    metric: Literal["percentile", "score"] = "percentile",
    varm_prefix: str = "kinase",
    sites: Sequence[str] | pd.Index | None = None,
    min_top_sites: int = DEFAULT_MIN_TOP_SITES,
) -> pd.DataFrame:
    """Build the ``build_signalome`` input from :func:`alphaphos.score_kinases` output.

    Returns a (sites x kinases) DataFrame on a [0, 1] scale: the Yaffe
    **percentile / 100** (default) -- the per-site ranking metric that is
    comparable across kinases -- restricted to the *informative* kinases: those
    that are the top-ranked kinase (ties included) of at least ``min_top_sites``
    of the selected sites.  With all 311 Ser/Thr PWMs the module x kinase table
    is near-uniform; a percentile cutoff cannot fix that because by construction
    every kinase has ~10 % of any site set at percentile >= 0.9.  Restricting to
    the kinases that actually win sites in *your* data is what makes modules
    interpretable (PhosR's Signalomes likewise works on a few dozen kinases).

    Parameters
    ----------
    adata
        AnnData after :func:`alphaphos.score_kinases` (varm
        ``<prefix>_<metric>_<pool>`` present).
    pool
        ``"ser_thr"`` (default) or ``"tyrosine"``.
    metric
        ``"percentile"`` (default; divided by 100) or ``"score"`` (raw log2
        PWM score -- returned unscaled with a warning).
    sites
        Optional subset of ``adata.var_names`` (e.g. the significant sites of
        a contrast).  Default: all sites.
    min_top_sites
        Keep kinases that are top-ranked for at least this many sites.
        ``0`` keeps every kinase.
    """
    key = f"{varm_prefix}_{metric}_{pool}"
    if key not in adata.varm:
        raise KeyError(
            f"adata.varm[{key!r}] not found (have {sorted(adata.varm.keys())}); run "
            "alphaphos.score_kinases(adata) first."
        )
    mat = adata.varm[key]
    if not isinstance(mat, pd.DataFrame):
        cols = adata.uns.get("alphaphos", {}).get("kinase_scores", {}).get(f"{pool}_kinases")
        mat = pd.DataFrame(np.asarray(mat, dtype=float), index=adata.var_names, columns=cols)
    mat = mat.astype(float)
    mat.index = pd.Index(mat.index.astype(str))
    if sites is not None:
        wanted = pd.Index(pd.Index(sites).astype(str))
        missing = wanted.difference(mat.index)
        if len(missing):
            raise KeyError(
                f"{len(missing)} requested sites not in adata.var_names, e.g. {list(missing[:3])}"
            )
        mat = mat.loc[wanted]
    if metric == "percentile":
        mat = mat / 100.0
    else:
        warnings.warn(
            "prediction_matrix_from_adata(metric='score'): raw log2 PWM scores are not on a "
            "[0, 1] scale and are not comparable across kinases; prefer metric='percentile'.",
            UserWarning,
            stacklevel=2,
        )
    informative = mat.notna().any(axis=1)
    mat = mat.loc[informative]
    if mat.empty:
        raise ValueError(f"no site in the selection has {pool} kinase scores")
    values = mat.to_numpy(dtype=float)
    row_max = np.nanmax(values, axis=1, keepdims=True)
    top_counts = pd.Series((values == row_max).sum(axis=0), index=mat.columns)
    keep = top_counts[top_counts >= int(min_top_sites)].index
    if len(keep) < 2:
        raise ValueError(
            f"only {len(keep)} kinase(s) are top-ranked for >= {min_top_sites} of {len(mat)} "
            "sites; lower min_top_sites."
        )
    out = mat.loc[:, keep]
    logger.info(
        "prediction_matrix_from_adata: %d sites x %d kinases (of %d) top-ranked for >= %d "
        "sites (%s); %d all-NaN sites dropped",
        out.shape[0],
        out.shape[1],
        mat.shape[1],
        min_top_sites,
        metric,
        int((~informative).sum()),
    )
    out.attrs["metric"] = metric
    out.attrs["pool"] = pool
    out.attrs["top_site_counts"] = top_counts.loc[keep].astype(int).to_dict()
    return out


def _validate_prediction_matrix(prediction_matrix: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(prediction_matrix, pd.DataFrame):
        raise TypeError("prediction_matrix must be a pandas DataFrame (sites x kinases)")
    if prediction_matrix.shape[0] == 0 or prediction_matrix.shape[1] == 0:
        raise ValueError(f"prediction_matrix is empty: shape {prediction_matrix.shape}")
    idx = pd.Index(prediction_matrix.index.astype(str))
    if idx.has_duplicates:
        dups = sorted(set(idx[idx.duplicated()]))[:3]
        raise ValueError(f"prediction_matrix has duplicate site ids, e.g. {dups}")
    try:
        values = prediction_matrix.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("prediction_matrix must be numeric") from exc
    finite = values[np.isfinite(values)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        warnings.warn(
            f"prediction_matrix values span [{finite.min():.2f}, {finite.max():.2f}], not [0, 1]. "
            "substrate_support_cutoff assumes a [0, 1] kinase-substrate score; build the "
            "input with alphaphos.signalome.prediction_matrix_from_adata() (percentile / 100).",
            UserWarning,
            stacklevel=3,
        )
    out = prediction_matrix.copy()
    out.index = idx
    return out


def _substrate_density(prediction_matrix: pd.DataFrame, cutoff: float) -> float:
    values = prediction_matrix.to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return 0.0
    density = float((values[finite] > cutoff).mean())
    if density > MAX_SUBSTRATE_DENSITY_WARN:
        warnings.warn(
            f"substrate_support_cutoff={cutoff} admits {density:.0%} of prediction-matrix cells "
            "as substrates; the module x kinase table will be close to uniform. Raise the cutoff "
            "(0.9 on a percentile scale) or restrict the kinases "
            "(prediction_matrix_from_adata(min_top_sites=...)).",
            UserWarning,
            stacklevel=3,
        )
    return density


# ---------------------------------------------------------------------------
# Defaults derived from prediction_matrix.index (alphaPhos site-key format)
# ---------------------------------------------------------------------------


def _default_site_to_protein(prediction_matrix: pd.DataFrame) -> pd.Series:
    idx = pd.Index(prediction_matrix.index.astype(str))
    proteins = idx.str.split("|").str[0].str.split(";").str[0]
    if proteins.isna().any() or (proteins == "").any():
        raise ValueError(
            "could not derive site_to_protein from prediction_matrix.index "
            "(expected alphaPhos 'Protein|Gene|Site|Mult' keys); pass "
            "site_to_protein= explicitly"
        )
    return pd.Series(proteins.to_numpy(), index=idx, name="protein")


def _default_site_metadata(prediction_matrix: pd.DataFrame) -> pd.DataFrame:
    from alphaphos.signalome.constants import (
        DISPLAY_ID_COLUMN,
        GENE_SYMBOL_COLUMN,
        ISOFORM_ID_COLUMN,
        PROTEIN_ACCESSION_COLUMN,
        SITE_COLUMN,
        SITE_KEY_COLUMN,
    )

    idx = pd.Index(prediction_matrix.index.astype(str))
    parts = idx.str.split("|")
    proteins = parts.str[0].str.split(";").str[0]
    genes = parts.str[1].fillna("")
    sites = parts.str[2].fillna("")
    display = (proteins + "_" + sites).where((proteins != "") & (sites != ""), idx.astype(str))
    return pd.DataFrame(
        {
            SITE_KEY_COLUMN: idx.to_numpy(),
            DISPLAY_ID_COLUMN: display.to_numpy(),
            GENE_SYMBOL_COLUMN: genes.to_numpy(),
            SITE_COLUMN: sites.to_numpy(),
            PROTEIN_ACCESSION_COLUMN: proteins.to_numpy(),
            ISOFORM_ID_COLUMN: [""] * len(idx),
        },
        index=idx,
    )


def _default_kinase_substrates(
    prediction_matrix: pd.DataFrame,
    *,
    cutoff: float,
) -> dict[str, tuple[str, ...]]:
    from alphaphos.signalome.assignments import select_kinase_substrates

    return select_kinase_substrates(prediction_matrix=prediction_matrix, cutoff=float(cutoff))


__all__ = [
    "SignalomeResult",
    "build_signalome",
    "prediction_matrix_from_adata",
]
