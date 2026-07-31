# Design principles

The "why" doc for expert readers. If you're evaluating whether to trust alphaPhos on your data, this is the page that explains the choices, grounds them in empirical evidence, and names the limits.

Every other doc in this repo (the top-level tutorial, the quickstart, the module deep-dives) is organized around *pipeline stages* — read left-to-right through PSM → collapse → filter → impute → DE. This doc is organized around the axis that actually changes the answers: **cohort size**. What's a sensible default for `n = 6` is measurably wrong for `n = 300`, and vice versa. The advisor (`ap.recommend_pipeline`) encodes this dependence; the rest of the doc explains why.

---

## Contents

1. [Data model — why AnnData, and which slots hold what](#1-data-model)
2. [Cohort regimes](#cohort-regimes) (Class-I, completeness, imputation, DE by scale)
    - [§2.1 Small cohorts — `n < 30`](#21-small-cohorts--n--30)
    - [§2.2 Medium cohorts — `30 ≤ n < 100`](#22-medium-cohorts--30--n--100)
    - [§2.3 Large cohorts — `100 ≤ n < 300`](#23-large-cohorts--100--n--300)
    - [§2.4 Very large cohorts — `n ≥ 300`](#24-very-large-cohorts--n--300)
3. [Cross-cutting choices](#3-cross-cutting-choices)
    - Aggregation, condition-aware collapse, DE on observed values, provenance
4. [What's still rough](#4-whats-still-rough)
5. [Empirical basis — how we know the defaults are right](#5-empirical-basis)

---

## 1. Data model

**alphaPhos uses `AnnData` (scanpy / scverse convention) with samples as observations (rows) and sites as variables (columns).**

Concretely, after `ap.collapse_sites(psm, condition_df=cond)`:

```
adata.X                      -- log2 intensity matrix, shape (n_samples, n_sites)
adata.obs                    -- sample metadata (condition, batch, subject, ...)
adata.var                    -- site metadata:
                                  PG.Genes, PG.ProteinGroups, site_aa, site_position,
                                  multiplicity, protein_group_id, gene,
                                  n_samples_detected, mean_loc_prob, max_loc_prob, min_loc_prob,
                                  n_classI_samples, fraction_classI, classI_wilson_lb
adata.layers["intensity_log2"] -- same content as X (canonical log2 layer)
adata.layers["localization"]   -- per-cell localization probability
adata.uns["alphaphos"]         -- pipeline provenance (version, params, per-stage stats)
adata.uns["wilson_filter"]     -- present when strategy="wilson" or apply_wilson_filter was used
```

The samples-as-obs orientation is deliberate. It's the opposite of some phospho tools (Perseus rows = features, columns = samples), and it matches every scverse tool downstream (`scanpy.pl.*`, `mudata`, `sklearn` estimators, `plotly.express`). Two-line rule: **anything you'd loop over goes into `.obs`. Anything you'd summarize per feature goes into `.var`.**

Multiple representations of the same intensity matrix live in **layers**, not in separate AnnDatas. When `ap.impute_hybrid` fills missing values, it writes to `adata.layers["intensity_log2"]` and leaves `adata.X` unchanged — that's how downstream code can distinguish "the observed matrix" from "the imputed matrix" without keeping two objects around. Every alphaPhos imputer respects this convention.

**Provenance is not optional.** Every pipeline step that changes the matrix stamps its parameters and result-counts into `adata.uns`. When a reviewer asks "what filter did you apply?", the answer is a one-liner:

```python
adata.uns["alphaphos"]["pipeline_params"]
adata.uns["wilson_filter"]     # threshold, reason, n_sites_pre, n_sites_kept
adata.uns["alphaphos"]["stats"]  # per-stage counts (rows before/after collapse, etc.)
```

If a downstream analysis result surprises you later, the whole provenance chain is one lookup away.

---

## Cohort regimes

The single most important input to any design choice is **how big is the cohort**. Different defaults are correct for a 3v3 pilot vs a 384-well plate; a single "reasonable default" across all n is a false economy. `ap.recommend_pipeline` encodes exactly this branching, and the sections below explain what changes when.

**Quick reference:**

| Regime | Class-I filter | Imputer for viz | DE default | Advisor prescribes |
|---|---|---|---|---|
| `n < 30` | `mean_loc_prob ≥ 0.75` | KNN | limma-observed-only, no imputation | `mean_loc_prob` recipe |
| `30 ≤ n < 100` | `mean_loc_prob ≥ 0.75` (Wilson optional) | KNN (PIMMS-DAE if `n ≥ 50`) | limma-observed-only | `mean_loc_prob` recipe |
| `100 ≤ n < 300` | **`strategy="wilson"` recommended** | PIMMS-DAE if `%NaN ≥ 40%`, else KNN | limma-observed-only + interaction contrasts | **Wilson recipe, auto threshold** |
| `n ≥ 300` | **Wilson essential** (naive filters demonstrably wrong) | **PIMMS-DAE** (KNN is 10-250× slower) | limma-observed-only + interaction contrasts | **Wilson recipe, auto threshold** |

---

### 2.1 Small cohorts — `n < 30`

**The regime**: pilots, dose-response studies, benchmark fixtures. Classical Perseus territory.

**Class-I filter**: `adata.var["mean_loc_prob"] >= 0.75`. Wilson's `auto` path explicitly refuses `n < 30` because the retention curve is too noisy to elbow-detect at this scale. You *can* use Wilson with a fixed threshold at `n = 6-29`, but our benchmarks show it collapses to a `per_run + min_valid_n` filter with no real added value from the sample-size correction.

**Imputation**: KNN (`ap.impute_knn_site_based`) or hybrid (`ap.impute_hybrid`) for visualization; PIMMS is below its paper's `n ≥ 50` floor and returns implausibly-smooth values at this scale. Better: **skip imputation entirely** for DE — see [§3](#3-cross-cutting-choices).

**Completeness filter**: strict (`min_valid_n=3` or `min_valid_frac=2/3` per group, `keep_strategy="each"`). Small cohorts don't survive loose filtering; the DE model needs full observation to be meaningful.

**DE**: `ap.diff_exp_limma_observed_only(imputer=None)`. Features present in only one group go to the on/off table; features in both groups get a moderated t-test on the observed values. **Do not impute at n < 30 before DE** — the imputer will fill missing cells with shift-normal draws that look like real down-regulation to limma, inflating false positives. This is the single most common mistake at this scale.

Empirical validation: EGF benchmark (n=6, 2 conditions × 3 reps). See `scratchpad/egf_precursor_informed_imputation.py` for the head-to-head — PIMMS-DAE at n=6 achieves MAE = 0.55, but the DE conclusions are noisier than the observed-only pipeline's.

---

### 2.2 Medium cohorts — `30 ≤ n < 100`

**The regime**: proteome-of-proteome studies, patient cohorts with 5-15 samples per group, condition-aware DIA panels.

**Class-I filter**: `mean_loc_prob >= 0.75` is still the default. Wilson `auto` becomes technically valid at `n ≥ 30` but its permissive band (`[0.20, 0.40]`) is a soft correction to `mean_loc_prob` — worth it if you're already reaching for Wilson elsewhere in your analysis, but not clearly better on this regime.

**Imputation**: **PIMMS-DAE crosses its viability threshold at `n ≥ 50`**. At `n = 50-99` with `%NaN < 40%` post-filter, KNN and PIMMS-DAE are roughly equivalent (KNN sometimes slightly better in MAE, PIMMS-DAE always faster). Above `%NaN = 40%`, PIMMS-DAE pulls ahead — the network learns cohort structure that KNN's per-site neighborhood cannot.

**Completeness filter**: The advisor computes `min_valid_n = max(3, min(10, round(0.6 × smallest_group_cell)))`. The cap at 10 matters — beyond that the test is already well-powered and higher thresholds drop good sites without adding statistical rigor. See `scratchpad/sfphospho_grouping_grid.py` for the empirical grid that motivated the cap.

**DE**: still observed-only by default. At n ≈ 60 with 4 conditions × 15 samples per condition, you have enough data to fit interaction terms — `ap.recommend_pipeline(goal="primary_de", ...)` will suggest the interaction-testable filter grouping (`primary × secondary`, `keep_strategy="each"`).

Empirical validation: cardio (n=69, patient × disease × region) — Wilson at threshold 0.30 retains 65% of sites at 45% NaN; PIMMS-DAE beats KNN by 13% MAE.

---

### 2.3 Large cohorts — `100 ≤ n < 300`

**The regime**: cross-lab studies, plate-based DIA acquisitions, clinical cohorts with 10-30 patients × several conditions.

**This is where Wilson-lb starts *mattering*, not just working.** At this scale:

- The `per_run` classical filter (any measurement below 0.75 kills the site) discards 80-90% of sites — atrocious for completeness.
- The `global_max` classical filter (any single Class-I measurement retains the site) barely filters anything (~94% retained).
- `mean_loc_prob >= 0.75` sits in the middle but doesn't correct for sample size — a site with `k = 5/5` Class-I measurements looks identical to a site with `k = 100/100`.

The Wilson lower bound resolves this: it rejects `5/5` at `p_lb = 0.57` and accepts `100/100` at `p_lb = 0.96`. `ap.recommend_pipeline` prescribes `strategy="wilson"` with `wilson_threshold="auto"` at `n ≥ 100`. The auto path elbow-detects on the retention curve within a cohort-size-informed range (`[0.30, 0.55]` for this regime).

**Imputation**: PIMMS-DAE is the default for visualization at this scale. KNN still works — at `n = 100`, KNN's ~1-minute runtime is fine — but PIMMS-DAE is 10-30× faster and matches or beats it in MAE on our benchmarks. For DE, still no imputation.

**Completeness filter**: The advisor's `min_valid_n = round(0.6 × smallest_group_cell)` (capped at 10) works well here. Interaction analyses become feasible with `primary_factor × secondary_factor` grouping.

**On/off detection**: worth explicitly considering at this scale. With 100+ samples, detection-only patterns (a site present in condition A, absent in condition B, in a non-trivial fraction of samples) become common and biologically meaningful. `ap.recommend_pipeline(goal="onoff_discovery", ...)` recipe uses `keep_strategy="any"` and routes to `ap.on_off_detection`.

Empirical validation: sfPhospho bulk phospho (n=99), sfPhospho SF (n=304), uPhosHT full plate (n=383). Retention curves showing Wilson's advantage over classical filters live in `scratchpad/wilson_retention_curve.py` and `scratchpad/classI_wilson_filter.py`.

---

### 2.4 Very large cohorts — `n ≥ 300`

**The regime**: high-throughput DIA studies. Plate-format acquisitions, cross-institutional cohorts, single-fiber phosphoproteomics.

**Wilson is not optional here; it's the correct default.** The scale at which naive Class-I filtering fails is now obvious:

- `per_run ≥ 0.75` on 383-sample uPhosHT plate: retains ~9% of sites.
- `global_max ≥ 0.75`: retains ~100% (no-op).
- Naive purity `k/n ≥ 0.75`: retains ~46%, but includes 5000-6000 sites with `n_det ≤ 5` — false-confidence hits from tiny per-site sample sizes.
- **Wilson-lb ≥ 0.5**: retains ~42% at 50% NaN, correctly rejects the false-confidence low-n hits. See `scratchpad/wilson_retention_curve.py` for the full curve.

**Imputation runtime becomes decisive at this scale.** On sfPhospho SF (n=304, ~24k sites):

- shift_rsn: < 1 second (but MAE ≈ 3.8; unusable)
- PIMMS-DAE: **10 seconds** (MAE ≈ 0.65)
- KNN: **25 minutes** (MAE ≈ 0.71)
- hybrid: 24 minutes (MAE ≈ 1.1; inherits shift_rsn's downside on MNAR-flagged features)

PIMMS-DAE is the only realistic imputer at this scale — an order of magnitude faster than KNN, with better or equal MAE across every regime we've tested.

**Completeness filter**: same formula (`min_valid_n = max(3, min(10, round(0.6 × smallest_cell)))`). The cap now matters critically — without it a naive "60% of samples per group" rule at `n_per_cell = 25` gives `min_valid_n = 15`, discarding half the sites for no statistical benefit.

**Memory considerations**: at `n ≥ 500`, `~50k` sites, dense KNN neighborhood computation starts to feel; PIMMS-DAE runs in bounded memory (chunked). Not currently a blocker; not currently validated beyond `n = 383`.

Empirical validation: uPhosHT full plate (n=383), PhosphoScape rapamycin (n=305), sfPhospho SF (n=304). Detailed retention + MAE tables in `scratchpad/wilson_retention_curve.py`, `scratchpad/imputation_benchmark_*`.

---

## 3. Cross-cutting choices

These principles apply at every cohort size:

### 3.1 DE on observed values is the default

Every alphaPhos DE call has an `imputer` argument. **The default (`imputer='hybrid'`) is optimistic; the recommendation (`imputer=None`) is honest.**

Imputation-then-DE is the classical Perseus workflow. It has a well-known failure mode: the imputer fills MNAR cells with shift-normal draws, which look like real down-regulation to limma. Features present in only one group get systematically mis-called as "significantly down in the other group" — inflating false positives on the down side. The magnitude of this artifact scales with the fraction of features that are "one-group-only detected", which is exactly the fraction that becomes non-trivial at cohort scales where imputation is *most tempting* (n ≥ 50).

`ap.diff_exp_limma_observed_only` splits the problem:
1. Features observed in both groups → moderated t-test (limma, no imputation)
2. Features present in only one group → on/off table with `call in {on_in_treatment, on_in_control, both, absent}`

Both tables get reported. Total hits from a paper using this pipeline are the union — but the reader can see which came from a real fold-change and which came from a presence/absence pattern.

Downstream tools that need a fully-imputed matrix (PCA, UMAP, clustering, some ML) can still call `ap.impute_pimms(adata)` or KNN — the imputer runs on a copy, and the imputed layer is separate from `adata.X`. Provenance stays clean.

### 3.2 Aggregation: sum in linear space (default)

`collapse_sites` aggregates multiple precursors per site via sum in linear (non-log) space, then log2-transforms the sum. This matches the convention in the Hogrebe / Mann lab workflow and preserves the "one precursor per (site, sample)" invariant that downstream tools assume.

Alternatives via `advanced={"aggregation_method": ...}`:
- `sum` (default): faithful to the underlying quantitative measurement
- `median`: robust to outlier precursors, at the cost of losing dynamic range
- `mean`: exposed for parity with older workflows; rarely a better choice than `sum`
- `consolidate`: use only when there's a specific reason to keep one representative precursor per site

Nothing about the design forbids other aggregations, but the empirical evidence is that `sum` is almost always the right default. Alternative choices should be justified in the methods section.

### 3.3 Condition-aware localization strategy

`collapse_sites` has three (now four, with Wilson) `localization_strategy` options controlling how the Class-I mask (`classI_cutoff = 0.75` by default) is applied at the precursor × run level:

- `"per_run"` (strict): mask each measurement below 0.75. Common in older workflows; **atrocious for completeness at large cohorts** — see §2.4.
- `"global_max"` (permissive): retain a precursor if it hits 0.75 in *any* run. Cheap but keeps low-confidence sites.
- `"condition"` (default; needs `condition_df`): retain a precursor in a given run if that run's condition has ≥ 50% Class-I measurements for that precursor. Balances scientific rigor with completeness.
- `"wilson"` (new in 0.22): runs `global_max` at the precursor level, then applies the per-site Wilson lower-bound filter on the aggregated data. **Recommended default for `n ≥ 100`.**

Choose based on cohort size, following §2. The advisor picks for you.

### 3.4 Provenance is always populated

Every alphaPhos function that mutates the AnnData stamps its parameters and result-counts into `adata.uns`. This is the design choice that lets a reviewer ask "what filter did you apply?" and get an answer without re-running the pipeline. If you write a wrapper around one of the low-level functions, respect this convention — put your params in `adata.uns` in a namespaced key.

---

## 4. What's still rough

Honest inventory of gaps as of 0.22.0. None of these block the alpha; some will block the beta.

- **`viz` submodule is a stub**. `alphaphos.viz` exists but doesn't ship reusable plotters. Publication-quality volcano, heatmap, PCA scatter, KSEA dotplot, and enrichment dotplot are all coming — right now users build these themselves against the DE results. `ap.qc.generate_dashboard` covers the exploratory-plot side; deliberate figure design is manual.
- **Memory scaling beyond `n = 500` is not validated**. `impute_knn_site_based` becomes expensive; PIMMS-DAE stays bounded but hasn't been stress-tested at n = 1000+. If you're running a study that big, run once at a subsample first and confirm before committing.
- **Error messages are inconsistent in quality**. Some are polished (`Wilson_threshold must be a float or "auto", got ...`); others let a Python stack trace bubble up. An audit is on the pre-beta list; feedback from you on which ones you hit is genuinely useful.
- **`dose_response.fit_dose_response` requires the optional `curve_curator` dependency**. Not installed by default. Not blocking for standard DIA phospho DE workflows.
- **KSEA (`ap.enrichment.kinase_activity`) requires the optional `decoupler` dependency** and currently fails on numpy 2.4+ due to a numba/numpy incompatibility upstream. Pin numpy < 2.4 if you need KSEA today. This is a known upstream regression, not an alphaPhos bug.
- **UMAP (`ap.dimred.umap`) requires the `[dimred]` extra and has the same numpy 2.4 constraint.** t-SNE (`ap.dimred.tsne`) has no such constraint; use it if UMAP is unavailable.
- **The precursor-informed imputation line** (three variants explored 2026-07-29) was benchmarked and dropped — the ideas didn't beat PIMMS-DAE on the EGF pilot, and the theoretical basis for expecting improvement at large cohorts turned out weaker than initially hoped. Full details in `scratchpad/egf_precursor_informed_imputation.py`.
- **msqrob2-style hierarchical DE** — genuinely more principled than limma-observed-only for cohorts with per-peptide variance structure. Not currently implemented; parked for post-1.0.

---

## 5. Empirical basis

The defaults above aren't inherited from Perseus or borrowed from a tutorial. They come from head-to-head benchmarks on real cohorts across the size spectrum. Every claim in this doc traces to reproducible scripts in `scratchpad/`:

| Dataset | `n_samples` | Data type | Benchmarks driven |
|---|---:|---|---|
| EGF benchmark | 6 | Phospho (Spectronaut DIA) | Baseline imputer comparison; DE-on-observed correctness; smallest-cohort behaviour of every filter/imputer |
| Cardio | 69 | Phospho (Spectronaut DIA) | Medium-cohort validation; multi-factor design (patient × disease × region); Wilson permissive band |
| sfPhospho bulk phospho | 99 | Phospho (Spectronaut DIA) | Large-cohort boundary case; batch effects |
| sfPhospho SF phospho | 304 | Phospho (Spectronaut DIA, single-fiber) | High-throughput retention curves; Wilson threshold selection; PIMMS-DAE scaling |
| sfPhospho proteome | 367 | Proteome | Proteome-scale defaults; classical no-loc_prob path |
| uPhosHT full plate | 383 | Phospho (Spectronaut DIA, 4-plate × 96-well) | Very-large-cohort validation; Wilson retention curves; auto threshold detection |
| PhosphoScape rapamycin | 305 | Phospho (Spectronaut DIA) | Cross-validation of Wilson defaults; independent cohort check |

Cross-dataset findings driving the defaults:

- **PIMMS-DAE < KNN < hybrid ≪ shift_rsn** in held-out MAE, at every cohort size. See `scratchpad/imputation_benchmark_*.pkl` and `scratchpad/sfphospho_run/impute_bench_*.pkl`.
- **PIMMS-DAE is 10-250× faster than KNN** at `n ≥ 300`. See runtime columns in the same pickles.
- **Wilson-lb correctly rejects ~14% false-confidence low-n_det sites** that naive `k/n ≥ 0.75` retains on the 305-sample rapamycin dataset. See `scratchpad/classI_wilson_filter.py`.
- **The `min_valid_n = round(0.6 × smallest_cell)` formula with cap at 10** was derived from `scratchpad/sfphospho_grouping_grid.py` — a 10-config grouping-strategy grid where the cap becomes measurably better than uncapped at `n ≥ 100`.

The empirical work is deliberately kept in scratchpad rather than baked into the package: the scripts are reproducible artifacts (readable, runnable) but not part of the API contract. When a reader wants to audit "did they really do this?", the answer is a git-clone away.
