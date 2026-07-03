# `alphaphos.io.fragpipe`

Read a FragPipe DIA site-abundance file directly into an `AnnData`.

Unlike the Spectronaut and DIA-NN readers, this one **bypasses**
[`collapse_sites`](../preprocess/collapse.md) because FragPipe's DIA workflow already
emits site-level matrices via its own IonQuant + PTM-Prophet aggregation. Trusting the
FragPipe developers on collapse; alphaPhos parses the site identifiers and constructs the
AnnData wrapper to match the contract that downstream tooling expects.

## Signature

```python
ap.read_fragpipe_sites(
    path: str | Path,
    *,
    condition_df: pd.DataFrame | None = None,
    advanced: dict[str, Any] | None = None,
) -> ad.AnnData
```

## Input

Either:

- A **FragPipe output directory** -- the abundance file is resolved from `advanced`'s
  `quant_level` / `site_type` / `normalized` settings.
- A specific `abundance_*-site_*.tsv` file -- e.g.
  `abundance_single-site_MS2quant_None.tsv`.

## Parameters

| Parameter | Default | What it does | Alternatives |
| --- | --- | --- | --- |
| `path` | *required* | FragPipe output directory OR specific abundance TSV. | -- |
| `condition_df` | `None` | Sample metadata (columns `sample` + `condition` at minimum; extras joined into `.obs`). | -- |
| `advanced` | `None` | Dict of overrides for `DEFAULT_FRAGPIPE_IO_SETTINGS`. Unknown keys raise. | See below. |

### `advanced` keys (see `DEFAULT_FRAGPIPE_IO_SETTINGS`)

| Key | Default | What it does |
| --- | --- | --- |
| `quant_level` | `"MS2"` | Selects `abundance_*-site_MS{2,1}quant_*.tsv`. |
| `normalized` | `False` | `True` -> selects the `_Norm.tsv` variant; `False` -> `_None.tsv`. |
| `site_type` | `"single"` | `"single"` -> single-site aggregation; `"multi"` -> multi-site. **`"multi"` has a known silent bug that can emit garbled protein IDs -- avoid until fixed.** |
| `min_best_localization` | `0.75` | Drop sites whose `best_localization` is below this. |
| `add_kinase_sequence` | `True` | Whether to attach the &plusmn;7-residue kinase sequence window to `.var["kinase_sequence"]`. |

## Output

An `anndata.AnnData` with shape `(n_samples, n_sites)` matching the contract of the
Spectronaut/DIA-NN collapse pipeline:

- `.X` = `.layers["intensity_log2"]` (log2 intensity).
- `.var.index` = `"Protein|Gene|aa+pos|Mmult"` (alphaPhos site keys).
- `.var` columns: `short_key`, `pg_key`, `protein_group_id`, `gene`, `site_aa`,
  `site_position`, `multiplicity`, `best_localization`, `sequence_window`,
  `kinase_sequence` (if `add_kinase_sequence=True`).
- `.obs` = `condition` (from `condition_df` if given) + any extra `condition_df` columns.
- `.uns["alphaphos"]` = `version`, `pipeline_params`, `source_file`.

## Raises

- `FileNotFoundError` -- path doesn't exist, or the resolved abundance file is missing.
- `ValueError` -- required columns are absent, or the file has no sample columns.

## Example

```python
import alphaphos as ap
import pandas as pd

conditions = pd.DataFrame({
    "sample":    ["s1", "s2", "s3", "s4"],
    "condition": ["ctrl", "ctrl", "trt", "trt"],
})

# Point at the FragPipe output directory; default MS2 quant, unnormalised
adata = ap.read_fragpipe_sites("fragpipe_out/", condition_df=conditions)

# Or point at a specific file
adata = ap.read_fragpipe_sites(
    "fragpipe_out/abundance_single-site_MS1quant_Norm.tsv",
    condition_df=conditions,
    advanced={"quant_level": "MS1", "normalized": True},
)

# The output is already a site-level AnnData -- skip collapse and go straight to
# filter / impute / diff-exp:
adata = ap.filter_by_completeness(adata, min_valid_frac=2/3,
                                  group_column="condition", keep_strategy="each")
adata = ap.impute_hybrid(adata)
result = ap.diff_exp_limma(adata, condition_column="condition",
                           comparison=("trt", "ctrl"))
```

## Design goals

- **Trust FragPipe's collapse** -- IonQuant + PTM-Prophet already aggregate to
  site-level. We parse identifiers and wrap; we don't re-collapse.
- **Match the AnnData contract** -- downstream tooling (QC dashboard, imputation,
  kinase annotation, diff-exp, dose-response) is engine-agnostic because the AnnData
  produced here uses the same layer/var/obs schema as `collapse_sites`.

## Known caveats

- **`site_type="multi"` has a silent bug** -- can emit garbled protein IDs. Stick to
  the default `"single"` until fixed.
- FragPipe's site-level output is what you get; there is no PSM-level path through
  alphaPhos for FragPipe DIA. If you need to alter aggregation (e.g. change quant
  method), regenerate the abundance file from FragPipe.
