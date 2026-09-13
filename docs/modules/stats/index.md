# `alphaphos.stats`

Differential testing on a site-level `AnnData`: limma-style moderated t and F tests,
plus a detection ("on/off") layer that keeps imputation-driven artefacts out of the
p-value table.

| Function | Question it answers | Backend |
| --- | --- | --- |
| [`diff_exp_limma`](#diff_exp_limma) | Two groups: is site *g* different between treatment and control? | inmoose (`lmFit → contrasts_fit → eBayes → topTable`) |
| [`diff_exp_limma_contrasts`](#diff_exp_limma_contrasts) | Several named pairwise contrasts on **one** joint fit | alphaPhos clean-room Smyth 2004 stack (`joint=True`, default) or an inmoose loop (`joint=False`) |
| [`diff_exp_anova`](#diff_exp_anova) | Does the mean differ *anywhere* across K levels? (moderated F) | clean-room stack |
| [`anova_hits`](#anova_hits) | Split an F-test table into `(hits, background)` for ORA | -- |
| [`on_off_detection`](#on-off-detection) | Which sites are *present in one group and absent in the other*? | counts, no p-values |
| [`annotate_imputed_provenance`](#on-off-detection) | Which limma hits rest on imputed values? | counts |
| [`diff_exp_limma_observed_only`](#diff_exp_limma_observed_only) | Headline two-group flow: on/off → strict completeness → impute → limma → provenance | inmoose |
| [`design_matrix`](#design_matrix) | Typed `~ 0 + condition + covariates (+ block)` design with provenance | -- |

`inmoose` + `patsy` come with `pip install "alphaPhos[stats]"`; the clean-room stack
(`diff_exp_limma_contrasts(joint=True)`, `diff_exp_anova`) needs only numpy/scipy.

## Scientific basis

All tests are **moderated** (Smyth 2004): the per-site residual variance is shrunk toward a
prior estimated from all sites (`fitFDist` → `s0²`, `df0`), which stabilises inference at the
`n = 3` per condition typical for phospho. By default the prior is **intensity-dependent**
(*limma-trend*, Law 2014): `s0²` follows a natural cubic spline in the site's average
log-intensity. MS residual variance falls steeply with intensity — on the EGF HeLa series the
residual SD drops from 0.33 to 0.16 between the lowest and highest intensity quintile — so a
constant prior over-shrinks bright sites and under-shrinks dim ones.
`advanced={"trend": False}` gives the constant Smyth 2004 prior.

The design is the no-intercept parameterisation `~ 0 + condition + covariate_1 + …`, so
each condition level is one coefficient and a contrast is simply
`condition[treatment] − condition[control]`.

### Validation

The clean-room stack was checked against inmoose's limma port on 2- and 3-group designs
(synthetic, and 12,029 complete sites of the EGF HeLa series), with the constant and the
trended prior: `log2fc` identical, moderated t, prior `df0` / per-site `s0²` and F identical
to ~1e-9 or better (`tests/unit/test_stats_moderated.py`, `test_stats_review.py`).

**Statistical audit** (EGF HeLa, 12,029 complete sites, 3 vs 3, 2026-09-13):

- `fdr` is Benjamini–Hochberg of `p_value` (verified against scipy and statsmodels), per
  contrast, over exactly the tested sites; 3,347 sites at p < 0.05 → 1,782 at FDR < 0.05.
- `t = log2fc / se`, `p = 2·t.sf(|t|, df0 + df_res)` two-sided, `se² = s²_mod·(1/n₁ + 1/n₂)`
  with the pooled variance shrunk toward the prior — all exact.
- Null calibration (parametric bootstrap from the real per-site variances, 10 × 12k sites):
  5.3% of p < 0.05 (Student t 5.0%, Welch 3.5%), KS D = 0.007, 0.1 false FDR-hits per 12k.
- Spike-ins (10% of sites changed): empirical FDR **0.050** at |log2FC| = 1 and 0.023 at 0.6,
  nominal 0.05; power 0.64 vs 0.42 for a Student t at effect 1.
- Impute-everything-then-test produced 1,913 hits of which 30% were imputed-driven, with twice
  the effect size and a 3:1 "up" skew; `diff_exp_limma_observed_only` reports those as
  detection events instead (see below).

One deliberate difference: the **F p-value** follows limma,
`pf(F, rank(C), df_prior + df_residual)`. inmoose 0.9.1's `classifyTestsF` returns
`df2 = inf` and therefore reports the chi-square limit, which is anti-conservative
(Δp up to 0.05 on the test data). This is why `joint=True` is the default for multi-contrast
work and why `diff_exp_anova` does not use inmoose.

## Conventions shared by all functions

- **Input**: `AnnData (n_samples, n_sites)`; the tested `layer` (default
  `"intensity_log2"`, `None` = `.X`) must be **log-scale** (median < 30 is enforced) and
  **NaN-free** in the samples used — run [`filter_by_completeness`](../preprocess/filter.md)
  and [`impute_hybrid`](../preprocess/imputation.md) first, or use
  `diff_exp_limma_observed_only`.
- **Levels are compared as strings**; `comparison=("2", "1")` works on an integer-coded
  `obs` column. Special characters (`EGF+`, `1uM`, `KO/WT`, spaces) are fine: levels are
  sanitised to identifiers internally (`x_1uM`, `KO_WT`), your original labels are what you
  get back in `result.attrs`.
- **Sign**: `log2fc = mean(treatment) − mean(control)`; positive = up in treatment.
  `result.attrs["contrast_direction"]` spells it out — cite it in Methods.
- **Covariates** (`covariates=[...]`): string / categorical columns are dummy-coded
  (reference level dropped); **float** columns enter as one continuous term; **integer or
  bool columns raise** — a batch coded `1, 2, 3` would otherwise be fit as a linear trend.
  Cast explicitly: `.astype(str)` for a factor, `.astype(float)` for a continuous covariate.
- **Guards** (all entry points): unique `var_names`; no NaN condition labels; a covariate
  perfectly confounded with the condition raises "rank-deficient"; groups with fewer than 3
  samples log a warning; and the **double-correction guard** (below).
- **Output**: one `DataFrame` per contrast, indexed by `adata.var_names` in input order.

| Column | `diff_exp_limma` | `diff_exp_limma_contrasts(joint=True)` | `diff_exp_anova` |
| --- | --- | --- | --- |
| `log2fc`, `se`, `t_stat`, `p_value`, `fdr` (BH across sites) | ✓ | ✓ | -- |
| `B` (log-odds) | ✓ | -- | -- |
| `ave_expr` | ✓ | ✓ | -- |
| `df_moderated` | -- | ✓ | ✓ |
| `F`, `df_between` | -- | -- | ✓ |

## `diff_exp_limma`

```python
ap.diff_exp_limma(adata, *, condition_column, comparison, covariates=None,
                  layer="intensity_log2", advanced=None) -> pd.DataFrame
```

| Parameter | Default | What it does |
| --- | --- | --- |
| `condition_column` | *required* | `.obs` column with the group label. |
| `comparison` | *required* | `(treatment, control)`. |
| `covariates` | `None` | Extra `.obs` columns for the design (see conventions). |
| `layer` | `"intensity_log2"` | Layer to test; `"intensity_log2_precombat"` for the batch-as-covariate path after ComBat. |
| `advanced` | `None` | `trend` (bool, default **True**: limma-trend prior; `False` = constant prior), `robust` (**not available** — inmoose 0.9.1 lacks it; `True` raises `NotImplementedError`), `winsor_tail_p` (pair in [0, 0.5); only meaningful with `robust`). Type-checked. |

```python
res = ap.diff_exp_limma(adata, condition_column="condition", comparison=("EGF", "ctrl"))
sig = res[res["fdr"] < 0.05].sort_values("log2fc", key=abs, ascending=False)

# with batch adjustment
res = ap.diff_exp_limma(adata, condition_column="condition", comparison=("EGF", "ctrl"),
                        covariates=["batch"])          # batch must be str/categorical
```

## `diff_exp_limma_contrasts`

```python
ap.diff_exp_limma_contrasts(adata, *, condition_column, contrasts, covariates=None,
                            block_column=None, layer="intensity_log2", joint=True,
                            advanced=None) -> dict[str, pd.DataFrame]
```

`contrasts` is a list of `(treatment, control)` tuples (keys become `"trt_vs_ctrl"`) or a
dict `{name: (treatment, control)}`.

- `joint=True` (default): **one** linear model over all samples and levels
  (+ covariates, + `block_column` as fixed effects for paired designs), the EB prior
  (trended by default) fitted once and shared by every contrast. Only
  `advanced={"trend": ...}` applies here; inmoose-only keys raise.
- `joint=False`: loops `diff_exp_limma` on each two-group subset; all `advanced` keys apply;
  `block_column` is not supported (encode it as a covariate).

```python
res = ap.diff_exp_limma_contrasts(
    adata, condition_column="condition",
    contrasts={"EGF5_vs_ctrl": ("EGF_5min", "ctrl"), "EGF30_vs_ctrl": ("EGF_30min", "ctrl")},
    block_column="replicate",            # paired design
)
res["EGF5_vs_ctrl"].attrs["prior_df"]    # shared EB prior
```

## `diff_exp_anova`

```python
ap.diff_exp_anova(adata, *, condition_column, covariates=None, layer="intensity_log2",
                  advanced=None) -> pd.DataFrame
```

Moderated F per site for "any level differs" across all `K` levels
(`df_between = K − 1`). The output is **unsigned**, so pair it with direction-agnostic
downstream tools:

```python
anova = ap.diff_exp_anova(adata, condition_column="dose")
hits, bg = ap.anova_hits(anova, fdr_threshold=0.05)   # NaN-FDR rows dropped from both
ap.enrichment.ora(hits, bg, libraries)
ap.enrichment.pathway_enrichment(anova, direction="any", background=bg)
```

Signed methods (`gsea`, `pathway_gsea`, `kinase_activity`) need a per-site direction —
run `diff_exp_limma_contrasts` for those.

## On/off detection

Imputing a whole group with the MNAR down-shift and then running limma yields inflated
fold-changes and over-confident FDR for every "on/off" site. The community convention
(MSstats, Perseus present/absent, DEqMS) separates **detection** from **quantification**:

```python
on_off = ap.on_off_detection(adata_pre_imputation, condition_column="condition",
                             comparison=("EGF", "ctrl"), min_observed_per_group=3)
on_off["call"].value_counts()   # both / on_in_treatment / on_in_control / absent

res = ap.annotate_imputed_provenance(limma_result, adata_pre_imputation,
                                     condition_column="condition", comparison=("EGF", "ctrl"))
res[~res["imputed_driven"]]     # headline table
```

`call == "both"` sites are eligible for limma; `on_in_*` sites are reported as detection
results without p-values.

## `diff_exp_limma_observed_only`

```python
ap.diff_exp_limma_observed_only(adata, *, condition_column, comparison,
    min_observed_per_group=3, covariates=None, layer="intensity_log2",
    imputer="hybrid", imputer_kwargs=None, limma_advanced=None) -> (limma_result, on_off_table)
```

The headline two-group flow on a **pre-imputation** `AnnData`: on/off table → keep
`call == "both"` → impute that subset in the tested `layer` (`"hybrid"`, `"knn"`, or `None`)
→ `diff_exp_limma` → `annotate_imputed_provenance`. `imputed_driven` is `False` for every
row by construction. `layer=None` cannot be combined with an imputer.

## `design_matrix`

```python
ap.stats.design_matrix(adata, *, condition_column, reference_level=None,
                       covariates=None, block_column=None, sample_mask=None) -> DesignMatrix
```

Builds the `~ 0 + condition + covariates + block` frame used by the clean-room path and
returns it with provenance (`condition_levels`, `reference_level`, `condition_coefficients`,
`level_sanitization`, `block_levels`). Rank-deficient designs raise. Useful for
inspecting exactly what will be fit, or for custom contrast matrices with
`alphaphos.stats.linear_model`.

## Double-correct guard

If the tested layer has been ComBat-corrected on a batch column
(`adata.uns["alphaphos"]["batch_correction"]`) and the same column is passed as a
covariate, the batch effect is removed twice → anti-conservative p-values. Every entry
point raises `ValueError` in that case (also for `layer=None`, since `.X` mirrors the
corrected canonical layer) and points to the two safe options:

- **(A) preferred**: test `layer="intensity_log2_precombat"` with `covariates=[batch]`;
- **(B)**: test the corrected layer without the batch covariate.

See [`batch_correct_combat`](../preprocess/batch_correct.md#double-correct-guard).

## Known caveats

- **inmoose 0.9.1 degenerate-variance bug**: when residual variance is identical across
  sites (synthetic data), `eBayes` fails with `df_prior = inf`; alphaPhos translates the
  raw `KeyError` into a message. Real data does not hit this.
- **inmoose F p-value** is anti-conservative (see Validation); alphaPhos never uses it.
- **`robust=True` is not available** on any path (inmoose 0.9.1 does not implement the
  Phipson 2016 robust prior); it raises rather than silently running the plain estimator.
- **Unequal within-group variances.** limma pools the variance across groups. On the EGF
  data the stimulated group is less variable (median variance ratio 0.36). With balanced
  groups the pooled moderated t remains well calibrated (see the bootstrap above); for
  strongly unbalanced designs with unequal variances, interpret borderline hits with care.
- `block_column` enters as fixed effects (limma's "fixed block" convention), not as a
  random effect (`duplicateCorrelation`).

## References

- Smyth 2004. *Linear models and empirical Bayes methods for assessing differential
  expression in microarray experiments.* Stat Appl Genet Mol Biol 3:Article3.
- Ritchie et al. 2015. *limma powers differential expression analyses for RNA-sequencing
  and microarray studies.* Nucleic Acids Res 43:e47.
- Phipson et al. 2016. *Robust hyperparameter estimation protects against hypervariable
  genes and improves power to detect differential expression.* Ann Appl Stat 10:946–963.
- Behdenna et al. 2020 / inmoose — Python port of limma + ComBat used for the two-group path.
