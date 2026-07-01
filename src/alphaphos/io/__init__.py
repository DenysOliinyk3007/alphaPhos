"""PSM-level readers for search-engine outputs.

Currently only Spectronaut Normal reports are supported. Planned engines
(DIA-NN, FragPipe, PEAKS, MaxQuant) will follow the same shape: each
reader normalizes into the Spectronaut-canonical dotted column schema
that :func:`alphaphos.collapse_sites` consumes.

Readers are designed to minimize memory: they prune columns AT READ TIME
using pyarrow / pandas ``columns=`` / ``usecols=``, so unused columns are
never materialized into RAM. See :func:`read_spectronaut` for the pattern.
"""

from alphaphos.io.spectronaut import (
    DEFAULT_IO_SETTINGS,
    resolve_io_settings,
)
from alphaphos.io.spectronaut import (
    read_psm as read_spectronaut,
)

__all__ = [
    "read_spectronaut",
    "DEFAULT_IO_SETTINGS",
    "resolve_io_settings",
]
