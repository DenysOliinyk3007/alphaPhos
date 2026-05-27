"""Fetch curated kinase-substrate networks from OmniPath / Enzsub.

OmniPath's ``Enzsub`` endpoint aggregates site-level enzyme-substrate
relationships from PhosphoSitePlus, SIGNOR, KEA, NetworKIN, HPRD,
ProtMapper, phosphoELM, dbPTM, and more (~50k phospho-interactions for
human, ~20k for mouse). This is the canonical "kinase-substrate priors"
network used by Saez-Rodriguez-style KSEA via `decoupler-py`.

This module:

- :func:`fetch_omnipath_ks_network` — pulls the network, formats it for
  decoupler (``source`` = kinase, ``target`` = site_id, ``weight`` = 1.0),
  and optionally caches to a parquet file (the network is ~30k rows; a
  fresh fetch takes 10-30s over the network).

- :func:`alphaphos_site_to_omnipath` — converts our ``PTM_Collapse_key``
  format (``"P00533~EGFR_Y1172_M1"``) to OmniPath's site format
  (``"P00533_Y1172"``) so the ``net.target`` ids match our diff_exp
  feature ids.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


# alphaPhos PTM_Collapse_key looks like ``{ProteinGroup}~{Gene}_{AA}{pos}_M{mult}``
# We extract the first protein from ProteinGroup (drop ``;`` alt forms),
# the AA letter (S/T/Y), and the position (digits).
_ALPHAPHOS_KEY_RE = re.compile(
    r"^(?P<protein>[^~;]+)(?:;[^~]+)?~[^_]*_(?P<aa>[STY])(?P<pos>\d+)_M\d+$"
)


def alphaphos_site_to_omnipath(site_id: str) -> str | None:
    """Convert an alphaPhos site id to OmniPath ``{UniProtAC}_{AA}{pos}`` format.

    >>> alphaphos_site_to_omnipath("P00533~EGFR_Y1172_M1")
    'P00533_Y1172'
    >>> alphaphos_site_to_omnipath("P12345;Q67890~MYGENE_S123_M2")
    'P12345_S123'  # takes the first protein from a multi-protein group

    Returns ``None`` for inputs that don't match the expected format.
    """
    if not isinstance(site_id, str):
        return None
    m = _ALPHAPHOS_KEY_RE.match(site_id)
    if m is None:
        return None
    return f"{m['protein']}_{m['aa']}{m['pos']}"


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
        ``"enzyme"`` (UniProt AC, e.g. ``"P06239"``) is more stable.
    target_id_style
        How to build ``target`` ids — ``"uniprot"`` gives
        ``"{UniProtAC}_{AA}{pos}"`` (matches :func:`alphaphos_site_to_omnipath`),
        ``"genesymbol"`` gives ``"{Gene}_{AA}{pos}"``.
    modification
        Only ``"phosphorylation"`` is supported (the relevant subset of
        Enzsub for KSEA).
    min_n_sources
        Minimum number of primary source DBs supporting an interaction.
        Default 1 (keep all). Set to 2 or 3 for stricter, higher-confidence
        networks.
    cache_path
        Optional path to a parquet cache. If the file exists, it's loaded
        instead of re-fetching from OmniPath. Cleared with
        ``overwrite_cache=True``.
    overwrite_cache
        If True, always re-fetch and overwrite the cache.

    Returns
    -------
    pd.DataFrame
        Columns: ``source``, ``target``, ``weight``, ``n_sources``,
        ``n_primary_sources``. One row per (kinase, site) pair. ``weight``
        is 1.0 for every row; consumers can re-weight by ``n_sources``
        if desired.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not overwrite_cache:
            logger.info("Loading cached KS network from %s", cache_path)
            return pd.read_parquet(cache_path)

    # Local import keeps omnipath optional (only needed for fetch)
    from omnipath.requests import Enzsub

    logger.info("Fetching OmniPath Enzsub for organism=%s ...", organism)
    es = Enzsub.get(organisms=organism, genesymbols=True)
    logger.info("Got %d enzyme-substrate rows from OmniPath", len(es))

    # Filter to phospho only
    es = es.loc[es["modification"] == modification].copy()
    es = es.loc[es["n_primary_sources"] >= min_n_sources]
    es = es.dropna(subset=[source_key, "substrate", "residue_type", "residue_offset"])
    # Build the target site id
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
    # Drop duplicate (source, target) rows (Enzsub can have the same pair via
    # multiple databases — collapse to a single edge keeping the max n_sources).
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
        "KS network: %d edges, %d kinases, %d sites",
        len(out),
        out["source"].nunique(),
        out["target"].nunique(),
    )
    return out
