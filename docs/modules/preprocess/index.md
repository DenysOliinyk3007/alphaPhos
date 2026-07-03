# `alphaphos.preprocess`

PSM &rarr; site &rarr; ready-for-stats pipeline. Five steps, five functions.

| Step | Function | Input | Output |
| --- | --- | --- | --- |
| 1. Collapse | [`collapse_sites`](collapse.md) | PSM `pd.DataFrame` | site `AnnData` |
| 2. Filter | [`filter_by_completeness`](filter.md) | `AnnData` | `AnnData` (fewer sites) |
| 3. Impute | [`impute_hybrid`](imputation.md) / [`impute_knn_site_based`](imputation.md) | `AnnData` | `AnnData` (no NaN in log2 layer) |
| 4. Batch correct (optional) | [`batch_correct_combat`](batch_correct.md) | `AnnData` | `AnnData` (batch-adjusted + pre-correction copy) |
| 5. Differential test | [`diff_exp_limma`](../stats/index.md) | `AnnData` | per-site `pd.DataFrame` |

## Typical pipeline

```python
import alphaphos as ap
import pandas as pd

# --- 1. Read + collapse ---
psm = ap.read_spectronaut("report.parquet")
conditions = pd.DataFrame({"sample": [...], "condition": [...]})
adata = ap.collapse_sites(psm, condition_df=conditions)

# --- 2. Filter ---
adata = ap.filter_by_completeness(
    adata, min_valid_frac=2/3,
    group_column="condition", keep_strategy="each",
)

# --- 3. Impute ---
ap.impute_hybrid(adata)   # in-place; fills layers["intensity_log2"]

# --- 4. (Optional) batch correction ---
# ap.batch_correct_combat(adata, batch_column="batch", covariates=["condition"])

# --- 5. Differential test ---
result = ap.diff_exp_limma(
    adata,
    condition_column="condition",
    comparison=("trt", "ctrl"),
)
```

## AnnData contract

Throughout the pipeline the working object is an `anndata.AnnData` with:

- **Shape** `(n_samples, n_sites)` -- samples as observations, sites as variables (the
  scverse convention).
- **`.X`** and `.layers["intensity_log2"]` -- log2 intensity, same matrix.
- **`.layers["localization"]`** -- per-cell localization probability (populated by
  `collapse_sites`).
- **`.layers["intensity_log2_precombat"]`** -- pre-batch-correction copy, populated only
  after `batch_correct_combat` runs (see [Batch correction](batch_correct.md)).
- **`.obs`** -- sample metadata. Index is sample name. Always includes `condition` when a
  `condition_df` was supplied to collapse.
- **`.var`** -- site metadata. Index is `Protein|Gene|Site|Mult`. Carries parsed key
  fields, motif flags, and site QC.
- **`.uns["alphaphos"]`** -- pipeline provenance (version, pipeline params, timestamps,
  per-stage stats, ComBat stamp).
- **`.uns["source_attrs"]`** -- PSM-level lineage (`n_rows_*` checkpoints).

This same contract is produced by both the Spectronaut/DIA-NN collapse path and the
FragPipe direct-read path, so downstream code doesn't care which engine produced the data.

## Batch correction: when + how

**Community convention** for phospho DIA experiments: ComBat-correct only for
**visualisation** (PCA / UMAP / clustering); run statistics on the **raw** log2 data with
batch as a limma covariate. `batch_correct_combat` supports both paths -- it keeps the
pre-correction values in `layers["intensity_log2_precombat"]` so `diff_exp_limma` can
still test the raw data with a batch covariate.

`diff_exp_limma` has a **double-correct guard**: if the layer you're testing has already
been ComBat-corrected on the same batch column you're passing as a covariate, it raises
`ValueError`. This prevents anti-conservative p-values from subtracting batch twice.
See [`batch_correct_combat`](batch_correct.md#double-correct-guard) for the fix.
