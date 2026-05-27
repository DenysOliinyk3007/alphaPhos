"""Kinase-Substrate Enrichment Analysis (KSEA) — network-based, via
decoupler-py + OmniPath (Saez-Rodriguez framework).

Distinct from ``alphaphos.kinase.enrichment`` which uses sequence-based
PWMs (Yaffe Kinase Library). This module uses **curated kinase-substrate
databases** (PhosphoSitePlus, SIGNOR, KEA, NetworKIN, etc. via OmniPath)
as priors, then runs decoupler's statistical methods (ULM, MLM, ORA,
GSEA) to score per-kinase activity.

Workflow:

    from alphaphos.ksea import fetch_omnipath_ks_network, kinase_activity_ulm

    net = fetch_omnipath_ks_network(organism="human",
                                     cache_path="ks_human.parquet")
    activity = kinase_activity_ulm(results, net,
                                    id_col="protein", stat_col="log2fc")
    print(activity.sort_values("score").tail(10))  # top activated

Modules:

- ``alphaphos.ksea.network`` — fetch + format the kinase-substrate
  network from OmniPath. Includes the alphaPhos -> OmniPath site-id
  converter.
- ``alphaphos.ksea.activity`` — four kinase activity methods:
  :func:`kinase_activity_ulm` (recommended default), :func:`mlm`,
  :func:`ora`, :func:`gsea`.
"""

from alphaphos.ksea.activity import (
    kinase_activity_gsea,
    kinase_activity_mlm,
    kinase_activity_ora,
    kinase_activity_ulm,
)
from alphaphos.ksea.network import (
    alphaphos_site_to_omnipath,
    fetch_omnipath_ks_network,
)

__all__ = [
    # Network
    "fetch_omnipath_ks_network",
    "alphaphos_site_to_omnipath",
    # Activity
    "kinase_activity_ulm",
    "kinase_activity_mlm",
    "kinase_activity_ora",
    "kinase_activity_gsea",
]
