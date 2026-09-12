"""PSM-level readers for search-engine outputs.

Currently supported:

- Spectronaut Normal reports (``read_spectronaut``)
- DIA-NN main reports (``read_diann``)
- FragPipe DIA site-abundance files (``read_fragpipe_sites``)

Spectronaut and DIA-NN readers normalize to a common canonical dotted
column schema that :func:`alphaphos.collapse_sites` consumes. The
FragPipe reader is different: FragPipe already emits site-level matrices,
so the reader BYPASSES ``collapse_sites`` and returns a ready-to-use
``AnnData`` directly.

All readers minimize memory by pruning columns at read time
(``pandas.read_parquet(columns=...)`` / ``read_csv(usecols=...)``).

The PSM-boundary contaminant filter (``filter_contaminants``) also lives
here because the readers apply it before anything downstream runs.
"""

from alphaphos.io.contaminants import (
    DEFAULT_CONTAMINANT_PREFIXES,
    filter_contaminants,
    get_default_contaminants_fasta,
    parse_fasta_accessions,
)
from alphaphos.io.diann import (
    DEFAULT_DIANN_IO_SETTINGS,
    resolve_diann_io_settings,
)
from alphaphos.io.diann import (
    read_psm as read_diann,
)
from alphaphos.io.fragpipe import (
    DEFAULT_FRAGPIPE_IO_SETTINGS,
    read_fragpipe_sites,
    resolve_fragpipe_io_settings,
)
from alphaphos.io.spectronaut import (
    DEFAULT_IO_SETTINGS,
    resolve_io_settings,
)
from alphaphos.io.spectronaut import (
    read_psm as read_spectronaut,
)

__all__ = [
    "read_spectronaut",
    "read_diann",
    "read_fragpipe_sites",
    "DEFAULT_IO_SETTINGS",
    "DEFAULT_DIANN_IO_SETTINGS",
    "DEFAULT_FRAGPIPE_IO_SETTINGS",
    "resolve_io_settings",
    "resolve_diann_io_settings",
    "resolve_fragpipe_io_settings",
    "filter_contaminants",
    "get_default_contaminants_fasta",
    "parse_fasta_accessions",
    "DEFAULT_CONTAMINANT_PREFIXES",
]
