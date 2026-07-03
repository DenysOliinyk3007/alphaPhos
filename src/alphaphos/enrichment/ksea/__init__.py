"""Kinase-activity inference — the first submodule of alphaphos.enrichment.

Wraps decoupler-py's linear-model methods (Badia-i-Mompel et al. 2022)
to score per-kinase activity from a differential-expression result.
Default is **ULM** (univariate linear model), the modern replacement
for the classical Wiredja 2017 KSEA z-score.  MLM (multivariate) is
available as an option when substrate sharing across kinases is a
concern.

Two bundled networks (OmniPath default, PTM-DB alternative) plus
bring-your-own via a DataFrame.
"""

from alphaphos.enrichment.ksea.activity import kinase_activity
from alphaphos.enrichment.ksea.network import (
    fetch_omnipath_ks_network,
    load_ptm_ks_network,
    validate_network,
)

__all__ = [
    "kinase_activity",
    "fetch_omnipath_ks_network",
    "load_ptm_ks_network",
    "validate_network",
]
