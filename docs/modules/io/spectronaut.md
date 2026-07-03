# `alphaphos.io.spectronaut`

Read a Spectronaut Normal-report PSM export into a PSM `pd.DataFrame`.

`read_spectronaut(path, *, advanced=None)` accepts a `.parquet` or `.tsv`/`.txt` file,
prunes columns at load time, applies boundary filters (decoys, contaminants,
q-value cutoffs), and normalises column names to the dotted convention. It does **not**
pick the quant column or run top-N attribution -- those live in
[`collapse_sites`](../preprocess/collapse.md).

## Signature

```python
ap.read_spectronaut(
    path: str | Path,
    *,
    advanced: dict[str, Any] | None = None,
) -> pd.DataFrame
```

Also exposed as `ap.io.spectronaut.read_psm`.

## Input

A Spectronaut Normal report:

- `.parquet` -- preferred; opened via pyarrow with column pruning.
- `.tsv` / `.txt` -- opened via pandas `read_csv(usecols=...)`.

Required columns are listed in the [io overview](index.md#required-spectronaut).
Underscored column names (e.g. `R_FileName` sometimes emitted by parquet exports) are
normalised to dotted form (`R.FileName`).

## Parameters

| Parameter | Default | What it does | Alternatives |
| --- | --- | --- | --- |
| `path` | *required* | Path to Spectronaut Normal report. | `.parquet`, `.tsv`, or `.txt`. |
| `advanced` | `None` | Dict of overrides for `DEFAULT_IO_SETTINGS`. Unknown keys raise. | See below. |

### `advanced` keys (see `DEFAULT_IO_SETTINGS`)

| Key | Default | What it does |
| --- | --- | --- |
| `drop_decoys` | `True` | Drop rows where `EG.IsDecoy == True`. |
| `drop_contaminants` | `True` | Drop rows whose protein-group id matches a contaminant prefix. |
| `contaminants_fasta` | `None` | Optional path to a contaminants FASTA (extra accessions considered contaminants). |
| `contaminant_prefixes` | `("CON__", "Cont_", "contam_")` | Prefixes that identify contaminant proteins. |
| `eg_qvalue_max` | `None` | Drop rows where `EG.Qvalue > eg_qvalue_max`. `None` disables. |
| `pg_qvalue_max` | `None` | Drop rows where `PG.Qvalue > pg_qvalue_max`. `None` disables. |

Use `ap.resolve_io_settings(advanced)` to preview the fully-resolved settings dict.

## Output

A `pd.DataFrame` with:

- PSM-level rows -- one row per (precursor, run) observation.
- Columns normalised to the dotted convention.
- All Spectronaut quant column variants preserved (collapse picks one later).
- `df.attrs` populated with lineage: `source_path`, `engine="SN"`, `n_rows_loaded`,
  `n_rows_after_decoys`, `n_rows_after_qvalue`, `n_rows_after_contaminants`,
  `n_rows_returned`, `columns_read`, `columns_dropped`.

## Raises

- `FileNotFoundError` -- path doesn't exist.
- `ValueError` -- bad settings, or a required column is missing from the report.

## Example

```python
import alphaphos as ap

# Default: drop decoys + contaminants, no q-value gating
psm = ap.read_spectronaut("report.parquet")

# Strict q-value gating for a manuscript re-run
psm = ap.read_spectronaut(
    "report.parquet",
    advanced={"eg_qvalue_max": 0.01, "pg_qvalue_max": 0.01},
)

# Custom contaminants FASTA
psm = ap.read_spectronaut(
    "report.tsv",
    advanced={"contaminants_fasta": "contaminants.fasta"},
)

# Feed into the collapse pipeline
adata = ap.collapse_sites(psm, condition_df=conditions)
```

## Design goals

- **Minimal memory footprint** -- columns are pruned AT LOAD TIME via pyarrow's
  `columns=` or pandas's `usecols=` arg. Typical saving: 2-3&times; on large reports.
- **Boundary filters only** -- decoys, contaminants, q-values are applied here because
  they cannot be reversed downstream. Quant-column pick, top-N attribution, and site
  collapse all live in [`collapse_sites`](../preprocess/collapse.md) so they can be
  tuned by strategy without re-parsing the raw report.

## Known caveats

- `EG.PTMAssayProbability` (peptide-level assay confidence) is **not** used as a filter
  anywhere in alphaPhos. Gate it upstream in Spectronaut if you need that filter.
- If `EG.PTMLocalizationProbabilities` is missing (older Spectronaut exports or reports
  pruned before load), collapse's top-N attribution dedup silently no-ops.
