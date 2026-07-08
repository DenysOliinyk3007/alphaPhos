"""Proteome analysis branch of alphaPhos.

Two Spectronaut input formats are supported:

* **Short (wide, pre-collapsed)** -- ``PG_ProteinGroups`` × sample matrix,
  one column per run named like ``[N]_<runname>_raw_PG_Quantity``.  This is
  Spectronaut's protein-group report, ready for downstream stats.  Fast to
  read, trusts Spectronaut's precursor → protein-group quantification.
* **Long (precursor-level)** -- one row per precursor observation, needs a
  collapse step (precursor → protein group).  Same shape as the phospho long
  parquet but without PTM-localization columns.  Slower, gives full control
  over the aggregation.

Both paths land in the same AnnData shape at the output: samples × proteins,
``.X`` in log2, ``.var`` indexed by ``PG_ProteinGroups``, ``.obs`` carrying
the caller-supplied ``condition_df``.  Downstream (filter / impute / batch-
correct / limma / PCA / pathway enrichment) then works unmodified on either.

**Pairing with phospho**: :func:`phospho_over_proteome` divides phospho
intensities by their parent-protein intensities (in log2 space, so
subtraction) to yield "fraction phosphorylated" per (site, sample).  Pairing
between phospho and proteome runs is by DVP well ID (regex the trailing
``_A1``..``_G11`` from the run name).

Public API
----------
:func:`read_spectronaut_short`    -- wide pre-collapsed report → AnnData.
:func:`read_spectronaut_long`     -- precursor-level report → precursor DataFrame.
:func:`collapse_proteome`         -- precursor DataFrame → protein AnnData.
:func:`phospho_over_proteome`     -- phospho AnnData / proteome AnnData → normalized phospho AnnData.
"""

from alphaphos.proteome.collapse import collapse_proteome
from alphaphos.proteome.io_long import read_spectronaut_long
from alphaphos.proteome.io_short import read_spectronaut_short
from alphaphos.proteome.pairing import phospho_over_proteome

__all__ = [
    "read_spectronaut_short",
    "read_spectronaut_long",
    "collapse_proteome",
    "phospho_over_proteome",
]
