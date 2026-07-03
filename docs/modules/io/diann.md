# `alphaphos.io.diann`

Read a DIA-NN main report and adapt it to the Spectronaut-canonical PSM schema.

`read_diann(path, *, advanced=None)` accepts a DIA-NN `.parquet` or `.tsv` main report,
prunes columns at load time, applies a six-step QC filter, restricts to phospho-only rows,
and rewires DIA-NN columns onto the Spectronaut PSM contract so the same
[`collapse_sites`](../preprocess/collapse.md) pipeline runs on both engines.

## Signature

```python
ap.read_diann(
    path: str | Path,
    *,
    advanced: dict[str, Any] | None = None,
) -> pd.DataFrame
```

Also exposed as `ap.io.diann.read_psm`.

## Input

A DIA-NN main report:

- `.parquet` -- preferred; opened via pyarrow with column pruning.
- `.tsv` -- opened via pandas `read_csv(usecols=...)`.

**Requires DIA-NN &ge; 1.9** -- needs `Protein.Sites` and `Site.Occupancy.Probabilities`.

## Parameters

| Parameter | Default | What it does | Alternatives |
| --- | --- | --- | --- |
| `path` | *required* | Path to DIA-NN main report. | `.parquet` or `.tsv`. |
| `advanced` | `None` | Dict of overrides for `DEFAULT_DIANN_IO_SETTINGS`. Unknown keys raise. | See below. |

### `advanced` keys (see `DEFAULT_DIANN_IO_SETTINGS`)

| Key | Default | What it does |
| --- | --- | --- |
| `mbr` | `True` | Whether match-between-runs is enabled (affects which QC columns are consulted). |
| `pg_qvalue_max` | `0.05` | Drop rows where `PG.Q.Value > pg_qvalue_max`. |
| `global_pg_qvalue_max` | `0.01` | Drop rows where `Global.PG.Q.Value > global_pg_qvalue_max`. |
| `lib_pg_qvalue_max` | `0.01` | Drop rows where `Lib.PG.Q.Value > lib_pg_qvalue_max`. |
| `quantity_quality_min` | `0.5` | Drop rows where `Quantity.Quality < quantity_quality_min`. |
| `pg_maxlfq_quality_min` | `0.7` | Drop rows where `PG.MaxLFQ.Quality < pg_maxlfq_quality_min`. |
| `require_locprobs` | `True` | Drop rows missing `Site.Occupancy.Probabilities`. |

Use `ap.resolve_diann_io_settings(advanced)` to preview the resolved settings.

## Adapter details

DIA-NN's native columns are rewired to Spectronaut-canonical names before the returned
DataFrame is emitted:

| Spectronaut target | DIA-NN source | Notes |
| --- | --- | --- |
| `R.FileName` | `Run` | Verbatim. |
| `EG.PrecursorId` | `Modified.Sequence` + `Precursor.Charge` | Joined with `_.<charge>`. |
| `PEP.PeptidePosition` | derived | 1-based peptide start position parsed from `Protein.Sites` + `Modified.Sequence`. |
| `EG.PTMAssayProbability` | `PTM.Site.Confidence` | Verbatim (per-site score). |
| `EG.PTMLocalizationProbabilities` | `Site.Occupancy.Probabilities` | Verbatim. |
| `PG.Genes` | `Genes.split(';')[0]` | First gene of the protein group. |
| `PG.ProteinGroups` | `Protein.Group` | Verbatim. |

**Phospho-only**: rows without `(UniMod:21)` in `Modified.Sequence` are dropped; other
PTM bracket markers (methylation, acetylation, etc.) are stripped and ignored.

**Quant columns**: DIA-NN's native quant columns (`Precursor.Quantity`,
`Precursor.Normalised`, `Ms1.Translated`, `Ms1.Area`) are preserved in the returned
DataFrame; collapse picks one based on `advanced["quantification_level"]`. Note: DIA-NN
has no MS2-level precursor quant, so `quantification_level="MS2"` falls back to MS1 with
a warning.

## Output

A `pd.DataFrame`:

- PSM-level rows with Spectronaut-canonical metadata columns + native DIA-NN quant
  columns.
- Ready to feed into `ap.collapse_sites` with `advanced={"search_engine": "Diann"}`.
- `df.attrs` populated with lineage: `source_path`, `engine="Diann"`, `n_rows_loaded`,
  `n_rows_after_qc`, `n_rows_phospho`, `n_rows_localizable`, `n_rows_unmappable`,
  `n_rows_returned`.

## Raises

- `FileNotFoundError` -- path doesn't exist.
- `ValueError` -- required columns absent from the report.

## Example

```python
import alphaphos as ap

# Default: manuscript-validated QC filters + phospho-only
psm = ap.read_diann("report.parquet")

# Loosen QC filters (larger, noisier hit set)
psm = ap.read_diann(
    "report.parquet",
    advanced={"pg_qvalue_max": 0.10, "quantity_quality_min": 0.3},
)

# Feed into collapse (note: switch engine)
adata = ap.collapse_sites(psm, condition_df=conditions,
                          advanced={"search_engine": "Diann"})
```

## Design goals

- **Adapter, not reimplementation** -- core collapse logic (site-key construction,
  aggregation, log2, noise-floor, condition-aware Class-I masking) lives in
  [`collapse_sites`](../preprocess/collapse.md). This reader only rewires DIA-NN columns
  onto the Spectronaut PSM contract.
- **Minimal memory footprint** -- same column-pruning approach as
  [`read_spectronaut`](spectronaut.md).

## Known caveats

- **Requires DIA-NN &ge; 1.9** for the `Protein.Sites` and `Site.Occupancy.Probabilities`
  columns. Older DIA-NN builds fail with a `ValueError`.
- MS2 quant is not available in DIA-NN reports; `quantification_level="MS2"` in collapse
  falls back to MS1 with a warning.
