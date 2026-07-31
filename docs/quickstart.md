# Quickstart — 30 minutes from install to differential-expression tables

The end state: two tables printed in your terminal or notebook.

- **`limma_result`** — features where both conditions had enough observations for a t-test; columns include `log2fc`, `p_value`, `fdr`.
- **`on_off_table`** — features present in only one condition (detection-only hits); columns include `n_observed_treatment`, `n_observed_control`, `call`.

No imputation, no plotting, no chart libraries. Just the numbers you'd feed into a volcano plot or an enrichment analysis next.

If this quickstart works end-to-end on your machine, the rest of the package is available and healthy. If any step fails, that's exactly the feedback we want — see [`docs/alpha-tester-guide.md`](alpha-tester-guide.md) for how to report.

---

## 0. Prerequisites

- Python 3.10, 3.11, or 3.12
- `git` on your PATH (for the install command below)
- A fresh virtual environment recommended

```bash
python -m venv .venv-alphaphos
# Windows:  .venv-alphaphos\Scripts\activate
# macOS/Linux:  source .venv-alphaphos/bin/activate
```

---

## 1. Clone and install (~3 minutes)

Currently alpha; the quickstart uses a bundled test fixture (EGF Spectronaut TSV) that lives at the repo root, so a clone is easier than a git-URL install:

```bash
git clone https://github.com/DenysOliinyk3007/alphaPhos.git
cd alphaPhos
git checkout add-large-batch-handling   # until this branch is merged to main

pip install -e ".[stats]"    # base + differential-expression deps (inmoose + patsy)
```

The `-e` (editable) install lets you edit alphaPhos source and see changes immediately; harmless for pure users. The `[stats]` extra is needed for `diff_exp_limma` and everything that calls it (including this quickstart).

To verify:

```python
import alphaphos as ap
print(ap.__version__)   # should print '0.22.0' or newer
```

**Optional extras** (skip for the quickstart; needed for deep-learning imputation, kinase enrichment, UMAP, and the QC dashboard):

```bash
pip install -e ".[enrichment]"   # KSEA + pathway enrichment (decoupler, gseapy, omnipath)
pip install -e ".[dimred]"       # UMAP (adds umap-learn; t-SNE works without this)
pip install -e ".[qc]"           # generate_dashboard() (bokeh + plotly + kaleido)
pip install -e ".[pimms]"        # PIMMS deep imputers (torch + pimms-learn)
pip install -e ".[all]"          # everything above in one call
```

Note: `[pimms]`, `[enrichment]`, and `[dimred]` currently require numpy < 2.4 (upstream numba/numpy incompatibility). If your environment has numpy 2.4+ and you don't need those specific features, skip the affected extras.

---

## 2. Get the test data (~5 seconds)

You're already there — the bundled EGF fixture is at `test_data/benchmark/EGF_diff_exp.tsv` relative to the repo root. From within the `alphaPhos/` directory:

```python
from pathlib import Path

data_path = Path("test_data") / "benchmark" / "EGF_diff_exp.tsv"
print("exists:", data_path.exists())   # should print True
```

Spectronaut TSV, 6 samples (3× EGF+, 3× EGF-), ~35k phospho sites. You'll swap this for your own Spectronaut / DIA-NN / FragPipe report later.

---

## 3. Load and collapse (~2 minutes)

```python
import alphaphos as ap
import pandas as pd

# --- 3a.  Read the Spectronaut PSM table
psm = ap.read_spectronaut(data_path)
print("PSM rows:", len(psm), "  unique samples:", psm['R.FileName'].nunique())

# --- 3b.  Build sample metadata (2 columns required: 'sample' + 'condition')
samples = psm["R.FileName"].unique()
cond_df = pd.DataFrame({
    "sample": samples,
    "condition": ["EGF+" if "withEGF" in s else "EGF-" for s in samples],
})
print(cond_df)

# --- 3c.  Collapse PSMs to a site-level AnnData
#     The default localization_strategy is "condition"; this needs the
#     condition_df to compute the per-condition Class-I mask.
adata = ap.collapse_sites(psm, condition_df=cond_df)
print("site AnnData shape:", adata.shape)
```

What just happened:

- **`read_spectronaut`** — parsed the TSV, extracted the localization-probability strings, deduplicated over-exports.
- **`collapse_sites`** — exploded precursors to their phospho sites, aggregated multiple precursors per site (default: sum in linear space), applied the Class-I mask (≥0.75 loc_prob per measurement, condition-aware), log2-transformed.

The returned `adata`:

- `adata.X` — `(n_samples, n_sites)` log2 intensities
- `adata.var` — per-site metadata: `PG.Genes`, `site_position`, `site_aa`, `mean_loc_prob`, `n_samples_detected`, `classI_wilson_lb`, plus more
- `adata.obs` — per-sample metadata from your `condition_df`

---

## 4. Class-I filter (~10 seconds)

At n=6 the EGF cohort is well below where sample-size correction matters, so use the classical Class-I cutoff on the per-site average localization probability:

```python
before = adata.n_vars
adata = adata[:, adata.var["mean_loc_prob"] >= 0.75].copy()
print(f"Class-I filter: {before:,} → {adata.n_vars:,} sites")
```

At larger cohorts (n≥100), the Wilson lower-bound filter is preferred — see [`docs/how-to-use-alphaphos.md`](how-to-use-alphaphos.md) §3.

---

## 5. Completeness filter (~10 seconds)

Drop sites that were never detected in either condition, but *keep* sites detected in only one condition — those become detection-only hits in the on/off table. `diff_exp_limma_observed_only` (step 6) will further split the retained sites into "limma-tested" (observed in both groups) and "on/off" (present in only one).

```python
adata = ap.filter_by_completeness(
    adata,
    min_valid_n=1,               # at least one observation …
    group_column="condition",
    keep_strategy="any",         # … in AT LEAST ONE condition
)
print("after completeness filter:", adata.shape)
```

Larger cohorts typically go stricter (e.g. `min_valid_n=5, keep_strategy="each"`) — but for a 3v3 study we lose too much biology being strict at the completeness stage, and the on/off detector at step 6 is the right place to make the observed-vs-detection-only call.

---

## 6. Differential expression (~5 seconds)

**This is the default alphaPhos DE path: no imputation, observed values only.** Features where both conditions have enough observations are limma-tested; features present in only one condition are moved to the on/off table.

```python
limma_result, on_off_table = ap.diff_exp_limma_observed_only(
    adata,
    condition_column="condition",
    comparison=("EGF+", "EGF-"),      # (treatment, control)
    min_observed_per_group=3,          # matches the completeness filter above
    imputer=None,                      # NO imputation on the surviving matrix
)
print("limma-tested features:", len(limma_result))
print("on/off (detection-only) hits:", len(on_off_table))
```

Why no imputation is the default:

- Imputing whole-group missing features inflates false-positive down-hits (the imputer fills MNAR cells with a shifted-normal draw; limma reads that as "signal is down"). Removing those features from limma and reporting them separately in the on/off table is more honest.
- For features that survive the completeness filter (observed in both groups), plain limma is enough — the missingness is sparse enough that a trend-fitted moderated t-test is well-behaved.

---

## 7. Inspect the results

**Top 10 differentially-abundant sites** (by FDR):

```python
top10 = limma_result.sort_values("fdr").head(10)
print(top10[["log2fc", "p_value", "fdr", "n_observed_treatment", "n_observed_control"]])
```

Note the column names: `log2fc` (not `log2FC`), `p_value` (not `pvalue`). `fdr` is Benjamini-Hochberg adjusted.

**Sites present only in EGF+ or only in EGF-** (detection-only):

```python
detection_only = on_off_table[on_off_table["call"].isin(["on_in_treatment", "on_in_control"])]
print(detection_only.head(10)[["call", "n_observed_treatment", "n_observed_control"]])
```

You should see approximately:
- **~11,900 features limma-tested** (all 3 samples observed in both conditions)
- **~5,100 detection-only hits** in the on/off table (~2,800 on_in_treatment + ~2,300 on_in_control)
- **~12,800 "absent"** features filtered out entirely (too few observations in either group)

EGF is a strong biological stimulus, so top log2fc values will be large; FDRs on the top hits are limited by the small 3v3 design more than by biology.

---

## 8. What next?

- **Try your own data**: swap `EGF_diff_exp.tsv` for your Spectronaut report, adjust the `cond_df` builder to reflect your sample naming, keep the rest.
- **Let the advisor tell you the recipe**: instead of picking filter/imputer parameters by hand, call `ap.recommend_pipeline(adata, goal="primary_de", data_type="phospho", primary_factor="condition")` after `collapse_sites`. It prints copy-paste code tuned to your cohort's size and design. See [`docs/how-to-use-alphaphos.md`](how-to-use-alphaphos.md) §5.
- **The full walkthroughs**: `examples/egf_walkthrough.ipynb` (2-group), `cardio_anova_walkthrough.ipynb` (multi-group ANOVA), `proteome_walkthrough.ipynb` (paired phospho/proteome), `cardio_signalome.py` (module + kinase network).
- **When it goes wrong**: [`docs/alpha-tester-guide.md`](alpha-tester-guide.md) has what feedback we want, how to report, and what's currently out of scope.
- **The "why these defaults" doc**: [`docs/design-principles.md`](design-principles.md) — for expert readers who want to know why we pick the defaults we do.

---

## Full script

If you want to paste one block into a notebook cell and run:

```python
import alphaphos as ap
import pandas as pd
from pathlib import Path

# 1. Read (run from the alphaPhos repo root)
data_path = Path("test_data") / "benchmark" / "EGF_diff_exp.tsv"
psm = ap.read_spectronaut(data_path)

# 2. Metadata
samples = psm["R.FileName"].unique()
cond_df = pd.DataFrame({
    "sample": samples,
    "condition": ["EGF+" if "withEGF" in s else "EGF-" for s in samples],
})

# 3. Collapse
adata = ap.collapse_sites(psm, condition_df=cond_df)

# 4. Class-I filter (small cohort → simple cutoff on mean_loc_prob)
adata = adata[:, adata.var["mean_loc_prob"] >= 0.75].copy()

# 5. Completeness filter (keep sites detected in either condition)
adata = ap.filter_by_completeness(
    adata, min_valid_n=1, group_column="condition", keep_strategy="any",
)

# 6. Differential expression on observed values (NO imputation)
#    → limma-tests only sites with ≥3 obs in BOTH groups
#    → moves the rest to the on/off table as detection-only hits
limma_result, on_off_table = ap.diff_exp_limma_observed_only(
    adata,
    condition_column="condition",
    comparison=("EGF+", "EGF-"),
    min_observed_per_group=3,
    imputer=None,
)

# 7. Inspect
print("Top 10 by FDR:")
print(limma_result.sort_values("fdr").head(10)[
    ["log2FC", "pvalue", "fdr", "n_observed_treatment", "n_observed_control"]
])
print("\nDetection-only hits:")
detection_only = on_off_table[on_off_table["call"].isin(["on_in_treatment", "on_in_control"])]
print(detection_only.head(10)[["call", "n_observed_treatment", "n_observed_control"]])
```
