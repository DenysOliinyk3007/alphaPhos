# `alphaphos.preprocess.batch_correct`

ComBat batch-effect correction via `inmoose.pycombat`.

`batch_correct_combat(adata, *, batch_column, ...)` sits between imputation and
differential testing: reads the canonical log2 layer (must be complete -- no NaN,
no all-NaN sites), removes known batch effects, and writes the corrected values back
to the same layer. The pre-correction values are copied to
`layers["intensity_log2_precombat"]` by default, so `diff_exp_limma` can still test the
raw data with a batch covariate (the statistically preferred path).

## Scientific basis

Empirical-Bayes ComBat (Johnson, Li, Rabinovic 2007 *Biostatistics*) fits a location-scale
model per batch and shrinks per-batch effect estimates toward a common prior. Adjusts
means (and optionally variances) so batch differences no longer dominate the numerics
while preserving covariate-of-interest structure passed via `covariates=`. Implemented
via [`inmoose.pycombat`](https://github.com/epigenelabs/pycombat) -- the Python port
that's numerically consistent with Bioconductor's `sva::ComBat`.

**Community convention** for phospho experiments: ComBat-correct for **visualisation**
(PCA / UMAP / clustering) so batch differences don't dominate the projection. Run
statistics on the **raw** data with batch as a limma covariate -- the linear model
handles the adjustment more cleanly than acting on the same numbers twice. Both
branches are available here because both layers are kept when `keep_precombat=True`.

## Signature

```python
ap.batch_correct_combat(
    adata: ad.AnnData,
    *,
    batch_column: str,
    covariates: list[str] | None = None,
    layer: str | None = "intensity_log2",
    keep_precombat: bool = True,
    advanced: dict | None = None,
    copy: bool = False,
) -> ad.AnnData | None
```

## Input

- `AnnData` shape `(n_samples, n_sites)`.
- Target `layer` must be **log-scale** and **NaN-free**. Run
  [`impute_hybrid`](imputation.md) or `impute_knn_site_based` first.

## Parameters

| Parameter | Default | What it does |
| --- | --- | --- |
| `adata` | *required* | `(n_samples, n_sites)`, log-scale, NaN-free in target layer. |
| `batch_column` | *required* | Column in `.obs` giving batch label per sample. Categorical; &ge; 2 levels; &ge; 2 samples per level. |
| `covariates` | `None` | List of `.obs` columns whose variance should be **preserved** (typically the biology of interest, e.g. `["condition"]`). Categorical only. Batch column itself may **not** appear here. |
| `layer` | `"intensity_log2"` | Which layer to correct. `None` targets `.X`. |
| `keep_precombat` | `True` | Copy pre-correction values to `layers["intensity_log2_precombat"]`. Keeps the raw log2 slot available to `diff_exp_limma` for the batch-as-covariate path. |
| `advanced` | `None` | Overrides for `DEFAULT_COMBAT_SETTINGS`. See below. |
| `copy` | `False` | If `True`, return a corrected copy; else mutate in-place. |

### `advanced` keys (see `DEFAULT_COMBAT_SETTINGS`)

| Key | Default | What it does |
| --- | --- | --- |
| `par_prior` | `True` | Parametric vs non-parametric EB estimation of batch effect priors. Parametric is faster and standard; non-parametric handles non-Gaussian effects. |
| `mean_only` | `False` | Adjust only batch means, not variances. Useful when variances are stable across batches. |
| `ref_batch` | `None` | If set, use this batch as reference and adjust the others to match it. `None` (default) centres all batches to the pooled mean. |

## Output

- If `copy=False`: `None` (in-place). Otherwise: modified `AnnData`.
- Target layer is now batch-corrected.
- `layers["intensity_log2_precombat"]` populated (if `keep_precombat=True`).
- `.uns["alphaphos"]["batch_correction"]` stamped with `{method: "combat", batch_column,
  covariates, par_prior, mean_only, ref_batch, layer, alphaphos_version}` for provenance
  and the double-correct guard.

## Raises

- `ImportError` -- `inmoose` / `patsy` not installed. Install via `pip install
  "alphaphos[stats]"`.
- `ValueError` -- fewer than 2 batches, batch with fewer than 2 samples, NaN in target
  layer, non-log-scale data (heuristic check), duplicate var-names, unknown `advanced`
  keys, `ref_batch` not a real batch, or a covariate equal to the batch column.
- `KeyError` -- missing `batch_column` / any `covariates` / `layer`.

## Example

```python
import alphaphos as ap

# Correct on 'batch', preserve 'condition' structure
ap.batch_correct_combat(
    adata,
    batch_column="batch",
    covariates=["condition"],
)

# Now for visualisation, use the corrected layer (default X):
# sc.pp.pca(adata); sc.pl.pca(adata, color="condition")

# For statistics, use the pre-correction layer with batch as a covariate --
# the statistically preferred path:
result = ap.diff_exp_limma(
    adata,
    condition_column="condition",
    comparison=("trt", "ctrl"),
    covariates=["batch"],
    layer="intensity_log2_precombat",   # <-- raw log2, pre-ComBat
)
```

## Double-correct guard

If you accidentally test the **ComBat-corrected layer** while also passing the batch as
a limma covariate, the batch effect gets subtracted twice (once by ComBat, once by the
linear model). The result is anti-conservative p-values (inflated false positives).

`diff_exp_limma` reads `adata.uns["alphaphos"]["batch_correction"]` and **raises
`ValueError`** in this case, pointing you at one of the two safe paths:

- **(A) statistically preferred**: test `layer="intensity_log2_precombat"` (the raw
  log2 slot) with `covariates=[batch_column]`.
- **(B) simpler**: test `layer="intensity_log2"` (ComBat-corrected) **without** listing
  the batch column in `covariates`. Report the pipeline in Methods so reviewers see
  which path you took.

## Known caveats

- Continuous covariates are **not supported** by pycombat -- if you have a continuous
  confounder, model it in limma directly instead.
- ComBat assumes at least 2 samples per batch level; a singleton batch raises.
- Numerical check on log-scale: the routine refuses to run if the values look linear
  (large positive range with an obvious noise floor). Bypass by explicitly passing
  a linear layer only if you know what you're doing.

## References

- Johnson, Li, Rabinovic 2007. *Adjusting batch effects in microarray expression data
  using empirical Bayes methods.* Biostatistics 8:118-127.
- Behdenna, Haziza, Azencott, Nordor 2021. *pyComBat, a Python tool for batch effects
  correction in high-throughput molecular data.* bioRxiv 2020.03.17.995431. (The Python
  port that inmoose ships.)
