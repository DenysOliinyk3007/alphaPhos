"""Backward-compatibility shim -- the contaminant filter now lives in
:mod:`alphaphos.io.contaminants`.

It moved because it is a PSM-boundary filter used by the readers
(``read_spectronaut``, ``proteome.read_spectronaut_long``); keeping it under
``preprocess`` created an ``io -> preprocess -> io`` import cycle.  Existing
imports from ``alphaphos.preprocess.contaminants`` keep working.
"""

from alphaphos.io.contaminants import (
    DEFAULT_CONTAMINANT_PREFIXES,
    filter_contaminants,
    get_default_contaminants_fasta,
    parse_fasta_accessions,
)

__all__ = [
    "DEFAULT_CONTAMINANT_PREFIXES",
    "filter_contaminants",
    "get_default_contaminants_fasta",
    "parse_fasta_accessions",
]
