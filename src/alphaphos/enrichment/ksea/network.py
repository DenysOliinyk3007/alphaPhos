"""Kinase-substrate network sources for KSEA.

Two bundled sources plus BYO:

- **OmniPath Enzsub** (default) -- aggregates ~50k phospho-interactions
  from PhosphoSitePlus, SIGNOR, KEA, NetworKIN, HPRD, ProtMapper,
  phosphoELM, dbPTM.  Community standard; fetched via the ``omnipath``
  Python package and cached to parquet.
- **PTM functional DB** -- our own curated 8-source aggregation.  Fewer
  edges than OmniPath overall but each carries a ``curation_confidence``
  tier so callers can filter for higher precision.
- **User-supplied DataFrame** -- validated against the same schema
  (``source``, ``target``, ``weight``).

All three feed decoupler's ULM / MLM methods via
:func:`alphaphos.enrichment.ksea.kinase_activity`.

Target-id format
----------------
Both bundled sources return the OmniPath-canonical
``"{UniProtAC}_{residue}{position}"`` format (e.g. ``P00533_Y1172``) --
matches :func:`alphaphos.enrichment.site_id` so the ``target`` column
joins directly against sites emerged from the alphaPhos pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from alphaphos.enrichment.db import site_id

logger = logging.getLogger(__name__)


NETWORK_COLUMNS: tuple[str, ...] = ("source", "target", "weight")


# ---------------------------------------------------------------------------
# OmniPath
# ---------------------------------------------------------------------------


def fetch_omnipath_ks_network(
    *,
    organism: Literal["human", "mouse", "rat"] = "human",
    source_key: Literal["enzyme_genesymbol", "enzyme"] = "enzyme_genesymbol",
    target_id_style: Literal["uniprot", "genesymbol"] = "uniprot",
    modification: Literal["phosphorylation"] = "phosphorylation",
    min_n_sources: int = 1,
    cache_path: str | Path | None = None,
    overwrite_cache: bool = False,
) -> pd.DataFrame:
    """Fetch a kinase-substrate network from OmniPath, formatted for decoupler.

    Parameters
    ----------
    organism
        ``"human"``, ``"mouse"``, or ``"rat"``.
    source_key
        Which kinase identifier to use as the ``source`` column.
        ``"enzyme_genesymbol"`` (e.g. ``"LCK"``) is more readable;
        ``"enzyme"`` (UniProt AC) is more stable.
    target_id_style
        ``"uniprot"`` produces ``"{UniProtAC}_{AA}{pos}"``;
        ``"genesymbol"`` produces ``"{Gene}_{AA}{pos}"``.
    modification
        Only ``"phosphorylation"`` is used for KSEA.
    min_n_sources
        Minimum number of primary source DBs supporting an interaction.
    cache_path
        Optional parquet cache path.  When present and non-stale, loaded
        instead of re-fetching.  A fresh fetch takes ~10-30 s.
    overwrite_cache
        Force re-fetch and overwrite the cache.

    Returns
    -------
    pandas.DataFrame
        Columns: ``source``, ``target``, ``weight`` (=1.0), ``n_sources``,
        ``n_primary_sources``.  Deduplicated on (source, target).
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not overwrite_cache:
            logger.info("Loading cached KS network from %s", cache_path)
            return pd.read_parquet(cache_path)

    # Local import keeps `omnipath` optional at runtime.
    from omnipath.requests import Enzsub

    logger.info("Fetching OmniPath Enzsub for organism=%s ...", organism)
    es = Enzsub.get(organisms=organism, genesymbols=True)
    logger.info("Got %d enzyme-substrate rows from OmniPath", len(es))

    es = es.loc[es["modification"] == modification].copy()
    es = es.loc[es["n_primary_sources"] >= min_n_sources]
    es = es.dropna(subset=[source_key, "substrate", "residue_type", "residue_offset"])

    target_proto = es["substrate" if target_id_style == "uniprot" else "substrate_genesymbol"]
    es["target"] = (
        target_proto.astype(str)
        + "_"
        + es["residue_type"].astype(str)
        + es["residue_offset"].astype(int).astype(str)
    )
    out = pd.DataFrame(
        {
            "source": es[source_key].astype(str),
            "target": es["target"].astype(str),
            "weight": 1.0,
            "n_sources": es["n_sources"].astype(int),
            "n_primary_sources": es["n_primary_sources"].astype(int),
        }
    )
    out = (
        out.sort_values("n_sources", ascending=False)
        .drop_duplicates(subset=["source", "target"], keep="first")
        .reset_index(drop=True)
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cache_path)
        logger.info("Cached KS network to %s", cache_path)

    logger.info(
        "OmniPath KS network: %d edges, %d kinases, %d sites",
        len(out),
        out["source"].nunique(),
        out["target"].nunique(),
    )
    return out


# ---------------------------------------------------------------------------
# PTM functional DB
# ---------------------------------------------------------------------------


def load_ptm_ks_network(
    *,
    db_path: str | Path | None = None,
    db: pd.DataFrame | None = None,
    require_curated: bool = True,
    min_curation_confidence: Literal["high", "medium", "low"] | None = None,
) -> pd.DataFrame:
    """Build a decoupler-compatible KS network from the PTM functional DB.

    Uses ``substrate_uniprot`` + ``residue`` + ``position`` for the target
    site id and ``enzyme_gene`` for the source kinase.  Deduplicates on
    (source, target), keeping the row with the highest curation
    confidence.

    Parameters
    ----------
    db_path
        PTM DB parquet path.  Defaults to
        ``resources/ptm_functional_db.parquet``.
    db
        Optional pre-loaded DB DataFrame; bypasses ``db_path``.
    require_curated
        If True (default), restrict to rows where ``is_curated`` is
        True.  Trades recall for precision.
    min_curation_confidence
        Optional stricter filter: ``"high"``, ``"medium"``, or ``"low"``.
        Kept edges have ``curation_confidence`` at or above this level.

    Returns
    -------
    pandas.DataFrame with columns ``source``, ``target``, ``weight``,
    ``curation_confidence``, ``n_sources``.
    """
    from alphaphos.enrichment.db import (
        COL_CURATION_CONFIDENCE,
        COL_ENZYME_GENE,
        COL_IS_CURATED,
        COL_POSITION,
        COL_RESIDUE,
        COL_UNIPROT,
        load_ptm_db,
    )

    if db is None:
        db = load_ptm_db(db_path)

    df = db.dropna(subset=[COL_ENZYME_GENE, COL_UNIPROT, COL_RESIDUE, COL_POSITION])
    if require_curated and COL_IS_CURATED in df.columns:
        df = df[df[COL_IS_CURATED].astype("boolean").fillna(False)]

    if min_curation_confidence is not None:
        _rank = {"low": 0, "medium": 1, "high": 2}
        threshold = _rank[min_curation_confidence]
        df = df[
            df[COL_CURATION_CONFIDENCE].astype(str).map(lambda s: _rank.get(s, -1) >= threshold)
        ]

    df = df.copy()
    df["target"] = [
        site_id(u, r, p)
        for u, r, p in zip(
            df[COL_UNIPROT].astype(str),
            df[COL_RESIDUE].astype(str),
            df[COL_POSITION],
            strict=True,
        )
    ]
    df["source"] = df[COL_ENZYME_GENE].astype(str)

    n_sources_col = "n_sources" if "n_sources" in df.columns else None
    out = df[
        ["source", "target", COL_CURATION_CONFIDENCE] + ([n_sources_col] if n_sources_col else [])
    ].copy()
    out["weight"] = 1.0
    out = out.rename(columns={COL_CURATION_CONFIDENCE: "curation_confidence"})

    # Deduplicate (source, target) keeping the highest curation-confidence row.
    _rank = {"high": 3, "medium": 2, "low": 1}
    out["_rank"] = out["curation_confidence"].astype(str).map(_rank).fillna(0)
    out = (
        out.sort_values("_rank", ascending=False)
        .drop_duplicates(subset=["source", "target"], keep="first")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )

    logger.info(
        "PTM-DB KS network: %d edges, %d kinases, %d sites (curated=%s, min_conf=%s)",
        len(out),
        out["source"].nunique(),
        out["target"].nunique(),
        require_curated,
        min_curation_confidence,
    )
    return out


# ---------------------------------------------------------------------------
# User-supplied network validation
# ---------------------------------------------------------------------------


def validate_network(net: pd.DataFrame) -> pd.DataFrame:
    """Verify a caller-supplied network has the decoupler schema.

    Required columns: ``source``, ``target``.  ``weight`` is auto-filled
    with 1.0 when missing.  All other columns are preserved for
    downstream use (annotation, filtering).

    Raises
    ------
    ValueError
        If required columns are missing or empty.
    """
    if not isinstance(net, pd.DataFrame):
        raise ValueError(f"network must be a DataFrame, got {type(net).__name__}")
    missing = [c for c in ("source", "target") if c not in net.columns]
    if missing:
        raise ValueError(f"network is missing required columns {missing}. Got: {list(net.columns)}")
    if net.empty:
        raise ValueError("network is empty")
    out = net.copy()
    if "weight" not in out.columns:
        out["weight"] = 1.0
    return out
