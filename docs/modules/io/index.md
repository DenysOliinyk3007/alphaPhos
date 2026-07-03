# `alphaphos.io`

Search-engine report readers. Three engines are supported; each has its own reader that
returns data in a form ready for the collapse or downstream pipeline:

| Engine | Reader | Output | When to use |
| --- | --- | --- | --- |
| Spectronaut | [`read_spectronaut`](spectronaut.md) | PSM `pd.DataFrame` | DIA data processed with Spectronaut. |
| DIA-NN | [`read_diann`](diann.md) | PSM `pd.DataFrame` | DIA data processed with DIA-NN &ge;1.9. |
| FragPipe | [`read_fragpipe_sites`](fragpipe.md) | site-level `AnnData` (already collapsed) | FragPipe's DIA workflow (IonQuant + PTM-Prophet). |

## PSM DataFrame schema (Spectronaut / DIA-NN)

Both `read_spectronaut` and `read_diann` return a **PSM-level `pd.DataFrame`** ready for
`ap.collapse_sites`. Columns use the dotted Spectronaut convention throughout
(`R.FileName`, `EG.PrecursorId`, ...); the DIA-NN reader adapts DIA-NN's native columns
into this canonical schema so the same downstream collapse pipeline runs on both.

The single source of truth for column names is
[`alphaphos.io.schemas`](../../../src/alphaphos/io/schemas.py).

### Required (Spectronaut)

| Column | Meaning |
| --- | --- |
| `R.FileName` | Sample / run identifier. Becomes `adata.obs_names` after collapse. |
| `EG.PrecursorId` | Modified sequence + charge (`_[Phospho (STY)]SPMK_.2` etc.). Parsed into (protein, site, multiplicity) during collapse. |
| `PEP.PeptidePosition` | 1-based peptide start position in the protein. |
| `EG.PTMAssayProbability` | Peptide-level assay confidence (not used as a filter internally — gate upstream in Spectronaut). |
| `PG.Genes` | First gene symbol of the protein group. |
| `PG.ProteinGroups` | Semicolon-joined UniProt accessions of the protein group. |

### Optional (Spectronaut)

| Column | Meaning |
| --- | --- |
| `EG.PTMLocalizationProbabilities` | Per-position localization probability string. Needed for top-N attribution. |
| `EG.IsDecoy` | Decoy flag; enables `drop_decoys`. |
| `EG.Qvalue`, `PG.Qvalue` | Per-peptide / per-protein-group q-values; enable q-value cutoffs. |

### DIA-NN adapter mapping

`read_diann` renames DIA-NN columns into the Spectronaut-canonical set:

| Spectronaut column (target) | DIA-NN column (source) |
| --- | --- |
| `R.FileName` | `Run` |
| `EG.PrecursorId` | `Modified.Sequence` + `Precursor.Charge` |
| `PEP.PeptidePosition` | derived from `Protein.Sites` + `Modified.Sequence` |
| `EG.PTMAssayProbability` | `PTM.Site.Confidence` |
| `EG.PTMLocalizationProbabilities` | `Site.Occupancy.Probabilities` |
| `PG.Genes` | `Genes.split(';')[0]` |
| `PG.ProteinGroups` | `Protein.Group` |

Phospho-only: rows without `(UniMod:21)` in `Modified.Sequence` are dropped; other PTM
bracket markers are stripped and ignored.

## Quant column selection

Neither reader picks the quantification column -- collapse does. Every available quant
column (`FG.MS2Quantity`, `FG.MS1Quantity`, `FG.Quantity (Settings)`, etc. for Spectronaut;
`Precursor.Quantity`, `Ms1.Translated`, etc. for DIA-NN) is preserved in the PSM DataFrame,
and `collapse_sites` picks one based on `advanced["quantification_level"]` with a fallback
chain per engine. See the [collapse docs](../preprocess/collapse.md) for details.

## `df.attrs` lineage

Both PSM readers stamp `df.attrs` with row-count checkpoints so the collapse pipeline
(and the QC dashboard's waterfall plot) can trace how many PSMs survived each filter.
Common keys:

- `source_path` -- absolute path read.
- `engine` -- `"SN"` or `"Diann"`.
- `n_rows_loaded` -- rows in the raw report before any filter.
- `n_rows_returned` -- rows in the returned DataFrame.
- (Spectronaut) `n_rows_after_decoys`, `n_rows_after_qvalue`, `n_rows_after_contaminants`, `columns_read`, `columns_dropped`.
- (DIA-NN) `n_rows_after_qc`, `n_rows_phospho`, `n_rows_localizable`, `n_rows_unmappable`.

## Design goals

- **Minimal memory** -- both PSM readers prune columns at load time via pyarrow / pandas
  `columns=` / `usecols=` args. Raw reports often have 30-50+ columns; ~15 are consumed
  downstream, so skipping the rest at read time saves 2-3&times; memory on large reports.
- **Read-and-normalise, nothing more** -- readers apply only *boundary* filters (decoys,
  contaminants, q-value cutoffs) and normalise column names. Quant-column selection,
  top-N attribution, and site collapse all live in `alphaphos.preprocess.collapse` where
  they can be tuned.
