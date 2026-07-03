# `alphaphos.stats`

Differential expression for phosphosites via limma (inmoose backend).

Currently a single public function: [`diff_exp_limma`](#diff_exp_limma) -- two-group
moderated t-test with optional covariate adjustment.

Multi-contrast ANOVA / F-tests are **not** included yet; the inmoose multi-contrast
`eBayes` path has a known bug on certain data shapes and support is deferred pending
regression testing against R limma on real data.

## Scientific basis

Two-group **moderated** t-test via `inmoose.limma`'s
`lmFit &rarr; contrasts_fit &rarr; eBayes &rarr; topTable` pipeline. The empirical-Bayes
step shrinks per-site variance estimates toward a common prior, stabilising inference on
small sample sizes (typical for phospho: `n = 3` per condition). This is the same
algorithm as Bioconductor's `limma::eBayes` -- alphaPhos ships the Python port via
`inmoose`, and the EGF benchmark reproduces R-limma's logFC and moderated t to numerical
precision.

The design matrix uses `~ 0 + condition + covariate_1 + ...` (no intercept, so the
condition coefficients directly represent group levels), and the contrast is
`condition[treatment] - condition[control]`.

## `diff_exp_limma`

### Signature

```python
ap.diff_exp_limma(
    adata: ad.AnnData,
    *,
    condition_column: str,
    comparison: tuple[str, str],
    covariates: list[str] | None = None,
    layer: str | None = "intensity_log2",
    advanced: dict | None = None,
) -> pd.DataFrame
```

### Input

- `AnnData` shape `(n_samples, n_sites)`.
- Target `layer` must be **log-scale** and **NaN-free** (in the samples actually used
  for the comparison). Run [`impute_hybrid`](../preprocess/imputation.md) first, and
  [`filter_by_completeness`](../preprocess/filter.md) before that.

### Parameters

| Parameter | Default | What it does | Alternatives |
| --- | --- | --- | --- |
| `adata` | *required* | Site-level AnnData; log-scale, NaN-free target layer. | -- |
| `condition_column` | *required* | Column in `.obs` holding the group label per sample. | Any `.obs` column. |
| `comparison` | *required* | `(treatment, control)`. Two levels in `adata.obs[condition_column]`. Log2FC is `mean(treatment) &minus; mean(control)`; positive = higher in treatment. | Any two valid levels. |
| `covariates` | `None` | Extra `.obs` columns for the design matrix (`~ 0 + condition + cov1 + ...`). Categorical (patsy dummy-coded). | E.g. `["batch"]`, `["batch", "sex"]`. |
| `layer` | `"intensity_log2"` | Which layer to test. `None` targets `.X`. | `"intensity_log2_precombat"` when ComBat has been applied and you want the batch-as-covariate path. |
| `advanced` | `None` | Overrides for `DEFAULT_STATS_SETTINGS`. See below. | -- |

### `advanced` keys (see `DEFAULT_STATS_SETTINGS`)

| Key | Default | What it does |
| --- | --- | --- |
| `trend` | `False` | Enable mean-variance trend fitting in `eBayes`. Recommended for RNA-seq-like data with mean-dependent variance; usually off for log2 phospho intensities. |
| `robust` | `False` | Enable the Phipson-2016 robust EB estimator. Downweights outlier sites when estimating the prior; helpful when a few sites have wild residual variance. |
| `winsor_tail_p` | `(0.05, 0.1)` | Passed to `eBayes` when `robust=True`. Winsorization tail fractions. |

### Output

A `pd.DataFrame`, one row per site, **indexed by `adata.var_names` (input order preserved)**:

| Column | Type | Meaning |
| --- | --- | --- |
| `log2fc` | float | `mean(treatment) &minus; mean(control)` on the input layer's scale (log2 by default). |
| `se` | float | Standard error of `log2fc`. |
| `t_stat` | float | Moderated t-statistic. |
| `p_value` | float | Nominal p from the moderated t-test. |
| `fdr` | float | Benjamini-Hochberg adjusted p across all sites. |
| `B` | float | limma's `B`: log-odds of differential expression. |
| `ave_expr` | float | Mean across all samples of the input layer for this site. |

`result.attrs["contrast_direction"]` names the sign convention:
`"log2fc = mean(treatment) - mean(control); positive = higher in treatment"`.

### Raises

- `ImportError` -- `inmoose` not installed. Install via `pip install "alphaphos[stats]"`.
- `ValueError` -- bad levels in `comparison`, NaN in tested layer, duplicate var-names,
  non-log data (heuristic), unknown `advanced` keys, **or the double-correct guard
  triggering** (see below).
- `KeyError` -- missing `condition_column` / `covariates` / `layer`.

### Example

```python
import alphaphos as ap

# Simplest two-group test
result = ap.diff_exp_limma(
    adata,
    condition_column="condition",
    comparison=("EGF", "ctrl"),   # positive log2fc = higher in EGF
)

# Significant sites at 5% FDR, sorted by effect
sig = result[result["fdr"] < 0.05].sort_values("log2fc", key=abs, ascending=False)

# With batch adjustment (recommended path when data has known batches)
result = ap.diff_exp_limma(
    adata,
    condition_column="condition",
    comparison=("EGF", "ctrl"),
    covariates=["batch"],
)

# Robust EB — downweight outlier sites when estimating the prior
result = ap.diff_exp_limma(
    adata,
    condition_column="condition",
    comparison=("EGF", "ctrl"),
    advanced={"robust": True},
)
```

## Double-correct guard

If the input layer has already been ComBat-corrected on a batch column and you also pass
that same batch column as a covariate, the batch effect gets subtracted twice
(once from the data, once from the model) &rarr; anti-conservative p-values.

`diff_exp_limma` reads `adata.uns["alphaphos"]["batch_correction"]` and **raises
`ValueError`** in this case, pointing to two safe paths:

- **(A) statistically preferred**: test `layer="intensity_log2_precombat"` (the raw log2
  slot saved by ComBat when `keep_precombat=True`) with `covariates=[batch_column]`.
- **(B) simpler**: test `layer="intensity_log2"` (ComBat-corrected) **without** the
  batch column in `covariates`.

See [`batch_correct_combat`](../preprocess/batch_correct.md#double-correct-guard) for
the full explanation.

## Known caveats

- **Two-group only.** Multi-contrast ANOVA / F-tests are not yet implemented. The
  inmoose multi-contrast eBayes path has a bug on certain data shapes; deferred pending
  regression tests.
- **inmoose 0.9.1 degenerate-variance bug**: when residual variance is identical across
  features (pathological, typically synthetic data), `eBayes` fails with `df_prior = inf`.
  Real biological data with heterogeneous per-site variance should not hit this.
  Workaround: add small jitter to `X`, or use `advanced={"trend": True}`.
- **`comparison` sign convention** is `(treatment, control)`. Reviewers often ask which
  direction is up -- `result.attrs["contrast_direction"]` documents it. Report it in
  Methods.

## References

- Smyth 2004. *Linear models and empirical Bayes methods for assessing differential
  expression in microarray experiments.* Stat Appl Genet Mol Biol 3:Article3. (The
  original moderated t-test.)
- Ritchie et al. 2015. *limma powers differential expression analyses for RNA-sequencing
  and microarray studies.* Nucleic Acids Res 43:e47.
- Phipson et al. 2016. *Robust hyperparameter estimation protects against hypervariable
  genes and improves power to detect differential expression.* Annals of Applied
  Statistics 10:946-963. (The `robust=True` estimator.)
- Behdenna et al. 2020 / inmoose -- the Python port of limma + ComBat used internally.
