"""Classical kinase-activity inference via decoupler's linear-model methods.

Wraps ``decoupler-py`` (Badia-i-Mompel et al. 2022 *Bioinformatics
Advances* 2:vbac016) to score per-kinase activity from a differential
expression result and a kinase-substrate network.

Default method is **ULM** (Univariate Linear Model): regresses the
observed site-level statistic against each kinase's substrate-indicator
vector; the slope's t-value is the activity score.  This is what
decoupler and Saez-Rodriguez et al. established as the modern
replacement for the classical Wiredja 2017 KSEA z-score -- essentially
the same test framed as a linear model, numerically equivalent on
typical phospho data.

**MLM** (Multivariate Linear Model) additionally accounts for
substrate sharing across kinases by fitting all kinases as covariates
in one joint regression.  Useful when many kinases share substrates.

Both methods return BH-adjusted FDR (decoupler applies this internally
across all tested kinases for the contrast).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from alphaphos.enrichment.db import site_id
from alphaphos.enrichment.ksea.network import (
    fetch_omnipath_ks_network,
    load_ptm_ks_network,
    validate_network,
)
from alphaphos.enrichment.matching import parse_alphaphos_key

logger = logging.getLogger(__name__)


_OBS_LABEL = "contrast"


def kinase_activity(
    diff_exp_result: pd.DataFrame,
    *,
    stat_col: str = "log2fc",
    network: str | pd.DataFrame = "omnipath",
    method: Literal["ulm", "mlm"] = "ulm",
    min_substrates: int = 5,
    organism: Literal["human", "mouse", "rat"] = "human",
    fdr_method: str = "bh",
    seed: int = 42,
    cache_path: str | Path | None = None,
    key_column: str | None = None,
) -> pd.DataFrame:
    """Kinase-activity inference via decoupler's linear-model methods.

    Parameters
    ----------
    diff_exp_result
        DataFrame from :func:`alphaphos.diff_exp_limma` (indexed by
        ``Protein|Gene|Site|Mult`` keys) or an already-canonicalised
        DataFrame with ``Protein_AApos`` site IDs.  Auto-detected from
        the index format.
    stat_col
        Column carrying the per-site statistic.  Typical values are
        ``"log2fc"`` (signed effect size) or ``"t_stat"`` (moderated).
    network
        Kinase-substrate network source:

        - ``"omnipath"`` (default) -- fetch via the ``omnipath`` package
          for the given ``organism``.
        - ``"ptm_db"`` -- our curated PTM functional-database network.
        - A DataFrame -- caller-supplied; must have ``source`` and
          ``target`` columns (weight defaults to 1.0 if missing).
    method
        ``"ulm"`` (default) -- univariate linear model per kinase.
        ``"mlm"`` -- multivariate linear model with joint fit across
        kinases (accounts for substrate sharing).
    min_substrates
        Minimum number of the kinase's substrates that must be present
        in ``diff_exp_result``.  Default 5 (decoupler convention).
    organism
        Passed to OmniPath when ``network="omnipath"``.
    fdr_method
        Currently only ``"bh"`` is supported (decoupler applies BH
        internally).  Documented for API forward-compatibility.
    seed
        Unused -- ULM / MLM are deterministic closed-form fits.  Kept for
        backward compatibility and recorded in provenance.
    cache_path
        OmniPath fetch cache location (parquet).
    key_column
        If ``diff_exp_result`` carries site IDs in a column rather than
        the index, pass its name here.

    Returns
    -------
    pandas.DataFrame with a ``kinase`` column and:

    - ``score`` -- signed activity (the ULM / MLM t-value; positive =
      activated, negative = inhibited)
    - ``fdr`` -- BH-adjusted p across all kinases in this run (decoupler
      2.x returns adjusted p only; verified against a manual univariate fit
      + BH -- there is no nominal ``p_value`` column)
    - ``n_substrates`` -- substrates observed in ``diff_exp_result``
    - ``direction`` -- ``"up"`` if score >= 0 else ``"down"``

    ``result.attrs["provenance"]`` records method, network source,
    seed, decoupler version, n_kinases_tested.
    """
    if method not in ("ulm", "mlm"):
        raise ValueError(f"method must be 'ulm' or 'mlm', got {method!r}")
    if fdr_method != "bh":
        raise ValueError(
            f"fdr_method must be 'bh' (decoupler applies BH internally), got {fdr_method!r}"
        )

    import decoupler as dc

    net = _resolve_network(network, organism=organism, cache_path=cache_path)

    if stat_col not in diff_exp_result.columns:
        if "F" in diff_exp_result.columns:
            raise ValueError(
                f"stat_col={stat_col!r} not in diff_exp_result columns, but "
                "an 'F' column is present -- this looks like diff_exp_anova "
                "output.  Kinase activity inference (ULM/MLM) requires a "
                "**signed** per-site statistic to infer activation vs "
                "inhibition, which ANOVA does not produce.  Re-run "
                "per-contrast with diff_exp_limma_contrasts(...) and pass "
                "each contrast's DataFrame to kinase_activity separately."
            )
        raise ValueError(
            f"stat_col={stat_col!r} not in diff_exp_result columns "
            f"(available: {list(diff_exp_result.columns)})"
        )

    if key_column is not None:
        site_series = diff_exp_result[key_column]
        stat_series = diff_exp_result[stat_col]
    else:
        site_series = pd.Series(diff_exp_result.index, index=diff_exp_result.index)
        stat_series = diff_exp_result[stat_col]

    canonical, n_input, n_matched = _canonicalise_site_ids(site_series)
    if canonical.empty:
        raise ValueError(
            "No site IDs could be canonicalised to OmniPath format "
            f"({n_input} inputs). Are these alphaPhos keys or "
            "Protein_AApos IDs?"
        )
    stat_aligned = stat_series.reindex(canonical.index).dropna()
    canonical = canonical.loc[stat_aligned.index]

    # decoupler's ULM/MLM want a (n_observations, n_features) matrix.
    # For a single contrast we build a 1-row matrix.
    data = pd.DataFrame(
        [stat_aligned.to_numpy()],
        index=[_OBS_LABEL],
        columns=canonical.to_numpy(),
    ).astype(float)
    # Drop duplicate columns (same site_id reached via two alphaphos keys —
    # multiplicity variants). Keep the largest-magnitude stat.
    if data.columns.duplicated().any():
        data = _dedup_columns_by_abs_max(data)

    overlap = set(net["target"]) & set(data.columns)
    if not overlap:
        raise ValueError(
            f"No overlap between {n_matched} matched site IDs and "
            f"network targets ({net['target'].nunique()} targets in "
            f"the network). Check that the network uses the same site-ID format."
        )
    logger.info(
        "kinase_activity(%s): %d/%d input sites canonical, %d overlap network, "
        "%d unique network kinases",
        method,
        n_matched,
        n_input,
        len(overlap),
        net["source"].nunique(),
    )

    fn = dc.mt.ulm if method == "ulm" else dc.mt.mlm
    try:
        score_df, padj_df = fn(data=data, net=net, tmin=min_substrates, verbose=False)
    except np.linalg.LinAlgError as exc:
        # MLM fails on rank-deficient designs -- happens when the network has
        # many overlapping-substrate kinases (typical of OmniPath).  ULM has
        # no such constraint.
        raise RuntimeError(
            f"decoupler {method.upper()} raised LinAlgError: {exc}. This usually "
            "means the kinase-substrate design matrix is rank-deficient (many "
            "kinases share substrates). Options: switch to method='ulm' (recommended), "
            "or use a smaller / higher-confidence network (e.g. "
            "load_ptm_ks_network(min_curation_confidence='high') or "
            "fetch_omnipath_ks_network(min_n_sources=3))."
        ) from exc

    # Reshape to per-kinase output
    result = pd.DataFrame(
        {
            "score": score_df.iloc[0],
            "fdr": padj_df.iloc[0],
        }
    )
    # decoupler 2.x returns BH-adjusted p-values only (checked against a
    # manual univariate fit + BH), so `fdr` is the honest name and there is
    # no nominal p column.
    result.index.name = "kinase"

    substrates_by_kinase = _substrates_per_kinase(net, present=set(data.columns))
    result["n_substrates"] = result.index.to_series().map(
        lambda k: len(substrates_by_kinase.get(k, ()))
    )
    result["direction"] = result["score"].apply(lambda s: "up" if s >= 0 else "down")
    result = result.sort_values("fdr").reset_index()

    network_label = network if isinstance(network, str) else "user_supplied"
    result.attrs["provenance"] = {
        "method": method,
        "network": network_label,
        "organism": organism if network_label == "omnipath" else None,
        "min_substrates": min_substrates,
        "n_input_sites": n_input,
        "n_canonicalised_sites": n_matched,
        "n_overlap_with_network": len(overlap),
        "n_kinases_tested": len(result),
        "seed": seed,
        "decoupler_version": getattr(dc, "__version__", None),
    }
    return result


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_network(
    network: str | pd.DataFrame,
    *,
    organism: str,
    cache_path: str | Path | None,
) -> pd.DataFrame:
    if isinstance(network, pd.DataFrame):
        return validate_network(network)
    if network == "omnipath":
        return fetch_omnipath_ks_network(organism=organism, cache_path=cache_path)
    if network == "ptm_db":
        try:
            return load_ptm_ks_network()
        except FileNotFoundError as exc:
            # The PTM functional-DB parquet is a planned Zenodo release
            # (v0.20.x era) but is not shipped with the package.  Surface
            # that clearly so users don't guess it's a broken install.
            raise FileNotFoundError(
                "network='ptm_db' requires the PTM functional-database parquet, "
                "which is not shipped with alphaPhos (planned Zenodo release). "
                "Options:\n"
                "  A) Use the OmniPath-backed network: "
                "network='omnipath' (requires internet on first call, then cached).\n"
                "  B) Build the DB locally from the source xlsx via "
                "``scripts/build_ptm_db.py``, then pass its output path to "
                "``load_ptm_ks_network(db_path=...)``.\n"
                "  C) Supply your own K-S DataFrame with 'source'/'target' "
                "columns to ``kinase_activity(network=<df>)``."
            ) from exc
    raise ValueError(f"network must be 'omnipath', 'ptm_db', or a DataFrame; got {network!r}")


def _canonicalise_site_ids(
    site_series: pd.Series,
) -> tuple[pd.Series, int, int]:
    """Convert alphaPhos ``Protein|Gene|Site|Mult`` to ``Protein_AApos``.

    Auto-detects: if the first non-null looks like an alphaPhos key,
    apply :func:`parse_alphaphos_key`; otherwise assume the IDs are
    already canonical.  Returns (canonical_series_indexed_like_input,
    n_input, n_matched).
    """
    n_input = len(site_series)
    if n_input == 0:
        return pd.Series(dtype=str), 0, 0

    sample = str(site_series.dropna().iloc[0])
    if "|" in sample:
        # alphaphos format
        def _convert(k):
            parsed = parse_alphaphos_key(str(k))
            if parsed is None:
                return None
            return site_id(parsed.protein, parsed.residue, parsed.position)

        canonical = site_series.map(_convert).dropna()
    else:
        # Assume already Protein_AApos
        canonical = site_series.astype(str).where(site_series.notna()).dropna()

    n_matched = len(canonical)
    return canonical, n_input, n_matched


def _dedup_columns_by_abs_max(data: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate columns by keeping the entry with maximum |value|.

    Occurs when several alphaphos keys (multiplicity variants) canonicalise
    to the same ``Protein_AApos`` site ID.  Keeping the strongest signal
    per site is the defensible choice; averaging would silently mute
    genuine biology.
    """
    row = data.iloc[0]
    abs_max_per_col: dict[str, float] = {}
    for col, val in zip(data.columns, row.to_numpy(), strict=True):
        prior = abs_max_per_col.get(col)
        if prior is None or abs(val) > abs(prior):
            abs_max_per_col[col] = val
    return pd.DataFrame(
        [list(abs_max_per_col.values())],
        index=data.index,
        columns=list(abs_max_per_col.keys()),
    ).astype(float)


def _substrates_per_kinase(net: pd.DataFrame, *, present: set[str]) -> dict[str, set[str]]:
    """Group the network by kinase, restricted to substrates in `present`."""
    scoped = net.loc[net["target"].isin(present), ["source", "target"]]
    out: dict[str, set[str]] = {}
    for src, tgt in zip(scoped["source"], scoped["target"], strict=True):
        out.setdefault(str(src), set()).add(str(tgt))
    return out
