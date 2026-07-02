# alphaPhos

Phosphoproteomics analysis toolkit.

**Status:** pre-alpha (v0.6.2). Standard 2-condition workflows (Spectronaut / DIA-NN / FragPipe → sites → filter → impute → optional ComBat → limma) are covered end-to-end and validated on real data. Multi-contrast ANOVA, PCA/UMAP, and PTM-SEA enrichment are not yet implemented.

## Install

```bash
pip install -e .                # core (Python-only)
pip install -e ".[stats]"       # + inmoose for limma differential testing + ComBat batch correction
pip install -e ".[dev,docs]"    # development
```

`inmoose` (Python limma + ComBat) is behind the `[stats]` extra; without it, `ap.diff_exp_limma` and `ap.batch_correct_combat` raise `ImportError` and everything else works.

## Quickstart

```python
import pandas as pd
import alphaphos as ap

# 1. Read PSMs from your search engine
psm_df = ap.read_spectronaut("report.tsv")     # or ap.read_diann(...) / ap.read_fragpipe_sites(...)

# 2. Sample metadata (must contain 'sample' and 'condition' columns)
condition_df = pd.DataFrame({
    "sample":    ["s1", "s2", "s3", "s4"],
    "condition": ["ctrl", "ctrl", "trt", "trt"],
})

# 3. Collapse PSMs -> site-level AnnData
adata = ap.collapse_sites(psm_df, condition_df=condition_df)

# 4. Keep sites observed in >= 2/3 of each condition
adata = ap.filter_by_completeness(
    adata, min_valid_frac=2/3,
    group_column="condition", keep_strategy="each",
)

# 5. Hybrid MAR/MNAR imputation (fills layers["intensity_log2"])
ap.impute_hybrid(adata)

# 6. (Optional) ComBat batch correction if the design has a known batch column
# ap.batch_correct_combat(adata, batch_column="batch", covariates=["condition"])

# 7. Two-group moderated t-test via limma
result = ap.diff_exp_limma(
    adata,
    condition_column="condition",
    comparison=("trt", "ctrl"),  # log2fc = mean(trt) - mean(ctrl)
)

# result: per-site DataFrame indexed by adata.var_names with columns
#   log2fc, se, t_stat, p_value, fdr, B, ave_expr
```

## What ships in v0.6.2

| Module | Function | Notes |
| --- | --- | --- |
| `alphaphos.io.spectronaut` | `read_spectronaut` | Column pruning at read time. Handles decoys + contaminants. |
| `alphaphos.io.diann` | `read_diann` | Manuscript-validated on nanoPhos. Requires DIA-NN ≥1.9. |
| `alphaphos.io.fragpipe` | `read_fragpipe_sites` | Reads FragPipe's pre-collapsed abundance files. |
| `alphaphos.preprocess.collapse` | `collapse_sites` | PSM → site AnnData with `Protein\|Gene\|Site\|Mult` keys, top-N attribution, three localization strategies. |
| `alphaphos.preprocess.filter` | `filter_by_completeness` | Global / any-group / each-group strategies. |
| `alphaphos.preprocess.impute` | `impute_hybrid`, `impute_knn_site_based` | Per-cell MAR-KNN + MNAR-Gaussian hybrid, or legacy-parity KNN. |
| `alphaphos.preprocess.batch_correct` | `batch_correct_combat` | inmoose pycombat wrapper with double-correct guard. |
| `alphaphos.stats.diff_exp` | `diff_exp_limma` | Two-group moderated t-test. Batch/covariate adjustment supported. |
| `alphaphos.kinase.annotation` | `add_kinase_windows`, `load_fasta` | ±7-residue windows around each site from a proteome FASTA. |
| `alphaphos.kinase.library` | Yaffe PWM scoring | Requires the optional `kinase_library` package. |
| `alphaphos.kinase.enrichment` | Kinase library enrichment | Same optional dep. |
| `alphaphos.ksea` | `kinase_activity_ulm`, `_mlm`, `_ora`, `_gsea` | OmniPath-based KSEA via `decoupler` (optional). |
| `alphaphos.dose_response` | `fit_dose_response` | CurveCurator wrapper (optional `curve_curator`). |
| `alphaphos.qc` | `generate_dashboard` | Bokeh QC HTML report (optional `bokeh`). |

## Known limitations to flag before use

- **No multi-contrast ANOVA / F-test.** `diff_exp_limma` is two-group only. The multi-contrast eBayes path has an inmoose bug on certain data shapes; support is deferred.
- **FragPipe multi-site mode has a silent bug.** Default `site_type="single"` works; `"multi"` can emit garbled protein IDs.
- **`.var` schema carries 9 legacy/internal columns** (`PTM_group`, `PTM_0_pos_val`, `UPD_seq`, `pg_key`, ...) with no downstream consumers. Slated for cleanup in a future minor.
- **`peptide_start = 0` silently emits `S0`** rather than rejecting. Locked by a characterization test; will surface if a future validator lands.
- **`EG.PTMAssayProbability` is not used as a filter anywhere.** Peptide-level assay confidence has to be gated upstream in Spectronaut.

## License

MIT — see [LICENSE](LICENSE).
