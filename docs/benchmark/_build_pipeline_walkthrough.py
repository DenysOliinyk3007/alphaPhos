"""Build docs/benchmark/pipeline_walkthrough.ipynb.

A clean, didactic walkthrough of the canonical alphaPhos analysis pipeline
on the EGF benchmark dataset. One section per step, with rationale + the
exact call. Empty outputs — user runs cell-by-cell to see results.

Pipeline shown:
  Spectronaut PSM TSV
    -> alphaphos.io.read_spectronaut           (q-value + decoys + top_n_attr + contaminants)
    -> alphaphos.preprocess.collapse_sites     (sum agg + condition-aware Class I)
    -> alphaphos.preprocess.to_anndata          (motif flags + site QC + provenance)
    -> alphapepttools.pp.filter_data_completeness
    -> alphaphos.preprocess.impute_hybrid       (p30 MAR/MNAR split)
    -> alphapepttools.tl.pca                   (sample-level QC)
    -> alphapepttools.tl.diff_exp_ebayes        (limma via inmoose, no R)
    -> alphapepttools.pl.volcano               (top hits)
    -> adata.write_h5ad                        (save for downstream)
"""

from pathlib import Path

import nbformat as nbf

NB_PATH = Path("D:/Projects/alphaPhos/docs/benchmark/pipeline_walkthrough.ipynb")


# ============================================================================
# Cells
# ============================================================================

INTRO_MD = """# alphaPhos canonical phospho-proteomics analysis pipeline

This notebook walks through the **recommended end-to-end workflow** for a
phospho-MS analysis: from a raw Spectronaut PSM report to a volcano plot of
differentially regulated sites.

It uses the **EGF benchmark dataset** (6-run nanoPhos DIA, 3 withEGF + 3
woEGF) as a worked example. Swap in your own report path + condition map
to use the same pipeline on new data.

## Pipeline at a glance

| Step | Tool | What it does |
|---|---|---|
| 1 | `alphaphos.io.read_spectronaut` | PSM ingestion + dedup + contaminant filter |
| 2 | (your `condition_df`) | sample → condition map |
| 3 | `alphaphos.preprocess.collapse_sites` | precursors → phosphosites, condition-aware Class I |
| 4 | `alphaphos.preprocess.to_anndata` | wide quants + phospho-specific var/uns metadata |
| 5 | `apt.pp.filter_data_completeness` | drop low-coverage sites |
| 6 | `alphaphos.preprocess.impute_hybrid` | per-cell MAR (KNN) / MNAR (Gaussian) split |
| 7 | `apt.tl.pca` + `apt.pl.plot_pca` | sample-level QC |
| 8 | `apt.tl.diff_exp_ebayes` | limma via `inmoose` (pure Python, no R) |
| 9 | `apt.pl.volcano` + top hits | results |
| 10 | `alphaphos.kinase.predict_kinases` | per-site upstream-kinase prediction (Yaffe PWMs) |
| 11 | `alphaphos.kinase.kinase_enrichment_from_diffexp` + `kinase_mea` | KSEA — per-kinase activity z-scores / NES |
| 12 | `adata.write_h5ad` | persist for downstream / sharing |

**Division of labor:** alphaPhos owns the phospho-specific parts (steps 1–4, 6).
alphapepttools owns the generic downstream (5, 7–9). Both operate on the same
`AnnData` object — handoff is zero-friction.
"""


IMPORTS_CODE = """from pathlib import Path
import sys
import warnings; warnings.filterwarnings("ignore")
import logging; logging.basicConfig(level=logging.INFO, format="%(message)s")

import numpy as np
import pandas as pd

# alphaPhos — phospho-specific upstream
import alphaphos
from alphaphos.io import read_spectronaut
from alphaphos.preprocess import (
    collapse_sites,
    to_anndata,
    impute_hybrid,            # MAR/MNAR per-cell split (p30 default)
    impute_knn_site_based,    # legacy-parity alternative (Pearson r=1.000 vs R-limma)
)

# alphapepttools — generic downstream
import alphapepttools as apt

# --- Preflight: ensure the kernel is on the right env -----------------------
print(f"Python interpreter: {sys.executable}")
print(f"alphaPhos:          {alphaphos.__version__}")
print(f"alphapepttools:     {apt.__version__ if hasattr(apt,'__version__') else '?'}")
# Older alphapepttools (<0.2.1) lacks `keep_strategy` in filter_data_completeness.
# If you hit a TypeError on that arg later, upgrade with:
#   !python -m pip install --upgrade "git+https://github.com/MannLabs/alphapepttools.git"
# then RESTART THE KERNEL.
import inspect as _inspect
_filter_params = _inspect.signature(apt.pp.filter_data_completeness).parameters
assert "keep_strategy" in _filter_params, (
    "alphapepttools is too old (no keep_strategy on filter_data_completeness). "
    "Run !python -m pip install --upgrade "
    "git+https://github.com/MannLabs/alphapepttools.git "
    "and restart the kernel."
)

ROOT = Path("D:/Projects/alphaPhos")
PSM_TSV = ROOT / "test_data/benchmark/EGF_report_new_ms2.tsv"   # change for your data
OUT_DIR = ROOT / "test_data/benchmark_output"
OUT_DIR.mkdir(exist_ok=True)
"""


STEP1_MD = """## §1. Read the Spectronaut PSM report

`read_spectronaut` is the gate. It loads the report, normalizes column names,
and applies four PSM-level filters with sensible phospho-aware defaults:

- `drop_decoys=True` — drop `EG.IsDecoy` rows
- `pg_qvalue_max=0.01` — protein-group FDR cap
- `top_n_attribution=True` — Spectronaut over-export dedup (drops redundant
   peptide rows that map to the same site but encode different candidate
   positions). Validated against SN classI at Pearson r=0.99.
- `drop_contaminants=True` — filter PSMs whose protein group is entirely
   common contaminants (trypsin, BSA, keratins). Uses the bundled MaxQuant
   `contaminants.fasta`. Drops ~0.5% of rows on a typical phospho run.

The function records every stage's row count in `df.attrs` for audit.
"""

STEP1_CODE = """df = read_spectronaut(
    PSM_TSV,
    quant_level="MS2",     # this dataset is MS2; use "auto" to auto-detect
    drop_decoys=True,
    pg_qvalue_max=0.01,
    top_n_attribution=True,
    drop_contaminants=True,
)

# Audit trail — exact row counts at each filter stage
a = df.attrs
print(f"Loaded:                            {a['n_rows_loaded']:>9,}")
print(f"  after top-N attribution:         {a['n_rows_after_top_n']:>9,}  "
      f"(dropped {a['n_rows_loaded'] - a['n_rows_after_top_n']:,})")
print(f"  after contaminant filter:        {a['n_rows_after_contaminant_filter']:>9,}  "
      f"(dropped {a['n_rows_after_top_n'] - a['n_rows_after_contaminant_filter']:,})")
print(f"Final rows returned:               {a['n_rows_returned']:>9,}")
print(f"Quant column used:                 {a['alphaphos_quant_column']}")
df.head(3)
"""


STEP2_MD = """## §2. Sample → condition mapping

`condition_df` is a two-column DataFrame mapping each run (`sample`) to its
biological condition. Extra columns (batch, donor, etc.) are propagated
into `adata.obs` and can be used downstream for filtering / coloring /
covariates.
"""

STEP2_CODE = """SAMPLES = sorted(df["R.FileName"].unique())
print(f"{len(SAMPLES)} samples in the report:")
for s in SAMPLES:
    print(f"  {s}")

# Build condition map by parsing the sample name
condition_df = pd.DataFrame({
    "sample": SAMPLES,
    "condition": ["withEGF" if "withEGF" in s else "woEGF" for s in SAMPLES],
})
print()
print(condition_df.to_string(index=False))
"""


STEP3_MD = """## §3. Collapse precursors to phosphosites

`collapse_sites` runs the full Hogrebe-style collapse with the
SN-validated defaults:

- `aggregation_method='sum'` — best match to Spectronaut's native PTM
   site report (Pearson r=0.99 on per-cell log2 quant; `'median'` introduces
   an intensity-dependent bias).
- `localization_strategy='condition'` — site collapse runs `global_max`
   upstream, then the per-condition Class-I majority mask is applied
   (`classI_cutoff=0.75`, `condition_threshold=0.50`). Site-set Jaccard 0.98
   vs SN classI on the EGF dataset.
- `return_decision_table=True` — optional Class-I per-condition decision
   table for downstream auditing.

If you have a proteome FASTA, pass `fasta_path="proteome.fasta"` and kinase
windows (`±7` residues around each phospho-S/T/Y) will be auto-computed —
needed for the kinase-Library / KSEA workflows in `alphaphos.kinase`.
"""

STEP3_CODE = """sites, loc_per_run, decision_table = collapse_sites(
    df,
    condition_df=condition_df,
    return_decision_table=True,
    # fasta_path="path/to/proteome.fasta",   # uncomment to enable kinase windows
)

print(f"Sites collapsed: {len(sites):,}")
print(f"loc_per_run shape: {loc_per_run.shape}  (sites x samples)")
print(f"decision_table shape: {decision_table.shape}  (sites x conditions)")
print()
print("Pipeline parameters recorded in sites.attrs:")
for k, v in sites.attrs.get("alphaphos_pipeline", {}).items():
    print(f"  {k:>25s} = {v}")
"""


STEP4_MD = """## §4. Convert to AnnData

`to_anndata` builds a scverse-conventions AnnData (samples × sites in `.X`)
and enriches it with three categories of phospho-specific metadata:

### `adata.var` (per-site)
- **Structural** (parsed from `PTM_Collapse_key`): `protein_group_id`,
  `gene_first`, `site_aa` (S/T/Y), `site_position`, `multiplicity`
- **Motif flags** (when `kinase_sequence` is present): `p_minus_1`, `p_plus_1`,
  `is_proline_directed` (+1 == P), `is_basophilic` (R/K at -2/-3/-5),
  `is_acidic_motif` (D/E at +1 or +3)
- **Site QC** (from `loc_per_run`): `n_samples_detected`, `mean_loc_prob`,
  `max_loc_prob`, `min_loc_prob`, `n_classI_samples`, `fraction_classI`

### `adata.obs` (per-sample)
- `condition` (and any extra columns from `condition_df`)

### `adata.uns['alphaphos']`
- `version`, `pipeline_params`, `processing_timestamp` — full provenance
"""

STEP4_CODE = """adata = to_anndata(
    sites,
    loc_per_run=loc_per_run,
    condition_df=condition_df,
    decision_table=decision_table,
    psm_attrs=dict(df.attrs),       # propagate PSM-level row counts for the QC waterfall
)

print(f"adata: {adata.n_obs} samples x {adata.n_vars} sites")
print(f"  layers: {list(adata.layers.keys())}")
print(f"  obs cols: {list(adata.obs.columns)}")
print(f"  var cols ({len(adata.var.columns)}): {list(adata.var.columns)}")
print()
print(f"  uns['alphaphos']['version'] = {adata.uns['alphaphos']['version']}")
print(f"  uns['alphaphos']['processing_timestamp'] = "
      f"{adata.uns['alphaphos']['processing_timestamp']}")
print()
# Per-site QC summary
print("Site QC distribution (first 5 sites):")
qc_cols = ["n_samples_detected", "mean_loc_prob", "n_classI_samples", "fraction_classI"]
adata.var[qc_cols].head()
"""


STEP5_MD = """## §5. Filter low-coverage sites

`apt.pp.filter_data_completeness` drops sites that don't meet a
completeness threshold. Two flags drive its behaviour:

- `max_missing=0.3` — at most 30% of samples in a group can be missing
- `keep_strategy='any'` (lenient) — keep if ≥1 group passes;
  `'all'` (strict) — require every group to pass

The lenient default matches Dublin's `filter_phosphosites(how='condition',
cutoff=0.7)` (presence ≥0.7 in at least one condition).
"""

STEP5_CODE = """adata = apt.pp.filter_data_completeness(
    adata,
    max_missing=0.3,
    group_column="condition",
    keep_strategy="any",
    action="drop",
)

# missingness stats post-filter
n_total = adata.X.size
n_miss = int(np.isnan(adata.X).sum())
print(f"adata after filter: {adata.shape}")
print(f"Missing cells:      {n_miss:,} / {n_total:,} ({n_miss/n_total*100:.1f}%)")
"""


STEP6_MD = """## §6. Hybrid imputation (MAR + MNAR)

The key phospho-specific step that alphapepttools doesn't ship.

`impute_hybrid` splits per-cell:

- **MNAR (Missing Not At Random)** — site mean intensity below the 30th
  percentile of the dataset. The value is below the limit of detection.
  Imputed via per-sample downshifted Gaussian (`μ − 1.8σ`, width `0.3σ`)
  — the Perseus convention.
- **MAR (Missing At Random)** — everywhere else. The site IS detectable;
  the missing value is a technical miss. Imputed via **site-based KNN**
  (transposed direction; `n_neighbors=int(sqrt(n_samples))`,
  `weights='uniform'`).

Why hybrid and not pure KNN or pure Gaussian:
- **Pure KNN** correctly imputes MAR but *inflates* MNAR (borrows from
  detectable neighbours, dampening true low-abundance fold-changes).
- **Pure Gaussian** correctly imputes MNAR but *inflates fold-changes
  for high-abundance sites* (where the missing cell isn't really below
  LOD). On the EGF benchmark, pure Gaussian gives EGFR Y1172 a logFC of
  −7.55, vs the correct −6.59 from KNN.
- **Hybrid** uses each where it applies. EGFR Y1172 stays at −6.59
  (high abundance → MAR → KNN), and ~30% of low-abundance MNAR sites
  get correctly downshifted instead of inflated.

`return_audit=True` returns a per-cell DataFrame showing which strategy
was used — useful for debugging or for stratification plots.
"""

STEP6_CODE = """_, audit = impute_hybrid(
    adata,
    mnar_threshold_percentile=30.0,    # default; bottom 30% of sites get Gaussian
    return_audit=True,
)

n_mar = (audit["strategy"] == "MAR_KNN").sum()
n_mnar = (audit["strategy"] == "MNAR_Gaussian").sum()
print(f"Imputed {len(audit):,} cells total:")
print(f"  MAR (site-KNN):     {n_mar:>7,}  ({n_mar/len(audit)*100:.1f}%)")
print(f"  MNAR (Gaussian):    {n_mnar:>7,}  ({n_mnar/len(audit)*100:.1f}%)")
print()
print(f"Remaining NaNs after imputation: {int(np.isnan(adata.X).sum())}  (should be 0)")
"""


STEP7_MD = """## §7. PCA — sample-level QC

`apt.tl.pca` adds the embedding to `adata.obsm["X_pca"]` and the explained
variance to `adata.uns["pca"]`. `apt.pl.plot_pca` produces a PC1/PC2 scatter
colored by any obs column.

Sanity checks at this step:
- Do replicates of each condition cluster together?
- Is PC1 capturing the condition contrast (vs e.g. a batch effect)?
- Any obvious outliers?
"""

STEP7_CODE = """import matplotlib.pyplot as plt

apt.tl.pca(adata, n_comps=2)

# alphapepttools namespaces results by method + dim_space. The defaults
# (method='pca', dim_space='obs') write to:
#   adata.obsm['X_pca_obs']
#   adata.varm['PCs_pca_obs']
#   adata.uns['variance_pca_obs']  -> {'variance_ratio': [...], 'variance': [...]}
var = adata.uns["variance_pca_obs"]["variance_ratio"]
print(f"PC1 / PC2 explained variance: {var[0]*100:.1f}% / {var[1]*100:.1f}%")

fig, ax = plt.subplots(figsize=(6, 5))
apt.pl.plot_pca(adata, color_map_column="condition", ax=ax)
plt.show()
"""


STEP8_MD = """## §8. Differential analysis — limma via inmoose

`apt.tl.diff_exp_ebayes` wraps `inmoose` (pure-Python Bioconductor limma port).
Same algorithm as R limma; same numerical results to floating-point precision.
No R subprocess; works on any platform.

Pass:
- `between_column` — the obs column whose values define groups
- `comparison=(treatment, control)` — `logFC = mean(treatment) - mean(control)`,
   so positive = up in treatment

Result is a long DataFrame with columns including:
- `protein` — the site identifier (alphaPhos `PTM_Collapse_key`)
- `log2fc` — moderated log2 fold-change
- `p_value`, `fdr` — moderated-t p-value and BH-adjusted
- `stat`, `B` — moderated-t statistic, log-odds of differential expression
"""

STEP8_CODE = """treatment, control = "woEGF", "withEGF"  # logFC = mean(treatment) - mean(control)

comp_id, results = apt.tl.diff_exp_ebayes(
    adata,
    between_column="condition",
    comparison=(treatment, control),
)

# Add significance flag (adj.P < 0.05 & |logFC| > 0.585 == ~1.5x fold-change)
results["sig"] = (results["fdr"] < 0.05) & (results["log2fc"].abs() > 0.585)

print(f"Comparison ID: {comp_id}")
print(f"Features tested: {len(results):,}")
print(f"Significant hits (FDR<0.05, |logFC|>0.585): {int(results['sig'].sum()):,}")
print(f"  up in {treatment}:   "
      f"{int(((results['sig']) & (results['log2fc'] > 0)).sum()):,}")
print(f"  up in {control}: "
      f"{int(((results['sig']) & (results['log2fc'] < 0)).sum()):,}")
"""


STEP9_MD = """## §9. Volcano plot + top hits

Standard volcano: x = logFC, y = −log10(FDR), colored by significance.

The top hits should include the textbook EGF signaling readouts:
- **EGFR autophosphorylation sites**: Y1172, Y1197 — strong negative logFC
  in woEGF (= up under EGF) is the canonical positive control.
- **Downstream signaling**: SHC1, GAB1, CRK, PIK3 components, MAPK pathway.
- **Negative-feedback regulators**: TSC2, INPPL1, etc.
"""

STEP9_CODE = """import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(7, 6))
apt.pl.volcano(
    results,
    x_column="log2fc",                    # results has 'log2fc' column already
    y_column="-log10(fdr)",               # FDR-axis volcano (standard for proteomics)
    x_thresholds=(-0.585, 0.585),         # |logFC| > 0.585 (~1.5x)
    y_thresholds=-np.log10(0.05),         # adj.P < 0.05
    ax=ax,
)
plt.tight_layout()
plt.show()
"""

STEP9_CODE_TOP = """# Top 15 hits by FDR — annotate with gene/site for biology check
top = results.nsmallest(15, "fdr")[["protein", "log2fc", "p_value", "fdr"]].copy()
top["log2fc"] = top["log2fc"].round(3)
top["p_value"] = top["p_value"].apply(lambda x: f"{x:.2e}")
top["fdr"] = top["fdr"].apply(lambda x: f"{x:.2e}")
top.reset_index(drop=True, inplace=True)
top
"""


STEP_KIN_MD = """## §10. Per-site kinase prediction (Yaffe Kinase Library)

**Per-site PWM scoring — answers: "for *this specific site*, which
kinase is the most likely regulator?"** This is sequence-based prediction
of upstream kinase(s) using the Yaffe Kinase Library PWMs (Johnson et al.
*Nature* 2023) — 311 Ser/Thr + 78 tyrosine kinases. Per-site PWM
prediction; NOT kinase activity / enrichment (that's §11).

**Prerequisite:** `kinase_sequence` must be populated in `adata.var`. That
happens automatically when you pass `fasta_path="proteome.fasta"` to
`collapse_sites()`. If your `adata` doesn't have this column, re-run §3
with a FASTA path.

Two functions:
- `score_kinases(adata)` writes the full (sites × kinases) score matrices
  to `adata.varm['kinase_score_ser_thr']` and `adata.varm['kinase_score_tyrosine']`.
- `predict_kinases(adata, top_k=5)` returns a per-site DataFrame with the
  top-k kinases. Calls `score_kinases` automatically on first run.

**What this answers:** "For *this specific site*, which kinase is most
likely the regulator?" — complementary to the network-based KSEA
question ("which kinase *activities* changed?") which is the planned
`alphaphos.ksea` module.
"""

STEP_KIN_CODE = """from alphaphos.kinase import predict_kinases

# Only run if we have kinase_sequence (i.e. collapse was run with fasta_path)
if "kinase_sequence" not in adata.var.columns:
    print("Skipping — no kinase_sequence column in adata.var.")
    print("Re-run §3 with collapse_sites(..., fasta_path='proteome.fasta')")
else:
    pred = predict_kinases(adata, top_k=5)

    # Summary
    info = adata.uns["alphaphos_kinase"]
    print(f"Scored {info['n_sites_scored']:,} sites "
          f"({info['n_ser_thr_scored']:,} S/T + {info['n_tyrosine_scored']:,} Y); "
          f"{info['n_sites_dropped']:,} dropped (invalid sequences)")
    print()

    # Attach the top1 prediction to adata.var for easy filtering/coloring
    adata.var["top1_kinase"] = pred["top1_kinase"]
    adata.var["top1_kinase_score"] = pred["top1_score"]
    adata.var["kin_type"] = pred["kin_type"]

    # Show the top-15 differentially regulated sites, annotated with their
    # predicted upstream kinases
    top_hits = (results.nsmallest(15, "fdr")
                       .set_index("protein")
                       .join(pred[["top1_kinase", "top1_score", "top_kinases"]],
                             how="left")
                       [["log2fc", "fdr", "top1_kinase", "top1_score", "top_kinases"]])
    top_hits.round(3)
"""


STEP_KSEA_MD = """## §11. Kinase enrichment / KSEA (Yaffe Kinase Library)

Where §10 asked "for *this site*, which kinase?", §11 asks **"across the
whole experiment, which kinase *activities* are changed?"** This is
classical KSEA — the headline kinase-level statistic in most phospho
papers.

`alphaphos.kinase.enrichment` exposes three complementary frameworks:

- **`kinase_enrichment_from_diffexp`** — Fisher's exact, per direction
  (up- and down-regulated sites separately). Takes the diff_exp result
  as input. Returns: per-kinase log2 frequency factor + Fisher p-value,
  per direction. **Most direct equivalent of classical KSEA.**

- **`kinase_mea`** — GSEA-style (weighted Kolmogorov-Smirnov via
  `gseapy`). Operates on the full ranked list (no hard threshold).
  Returns: per-kinase NES + p-value + FDR. **Most rigorous** (uses every
  site's rank, not a binarised threshold).

- **`kinase_enrichment_binary`** — Fisher's exact, custom foreground vs
  background (e.g. for a PCA cluster, or a pathway-curated site set).

For the EGF dataset we'd expect:
- **RTK / Src-family** activity *up* (EGFR autophos drives the cascade)
- **AKT / mTOR / RSK** activity *up* (canonical EGF downstream)
- **MAPK family** (ERK1/2) activity *up* (EGF→RAS→RAF→MEK→ERK)
- Phosphatases or feedback regulators *down*
"""

STEP_KSEA_CODE = """from alphaphos.kinase import kinase_enrichment_from_diffexp, kinase_mea

if "kinase_sequence" not in adata.var.columns:
    print("Skipping — no kinase_sequence column (re-run §3 with fasta_path).")
else:
    # 1. Classical KSEA (Fisher per direction) — runs in seconds
    print("Running Fisher-based KSEA (kinase_enrichment_from_diffexp)...")
    ksea = kinase_enrichment_from_diffexp(
        diff_results=results,
        sequence_lookup=adata.var["kinase_sequence"],
        id_col="protein",
        lfc_col="log2fc",
        pval_col="fdr",
        lfc_thresh=0.585,
        pval_thresh=0.05,
        kl_method="percentile",
        kl_thresh=90,
    )

    # Show the top activated/inhibited kinases in each kin_type
    for kt, df_kt in ksea.items():
        print(f"\\n=== {kt} — top 10 by 'most significant' direction ===")
        top = (df_kt
                 .sort_values("most_sig_fisher_adj_pval")
                 .head(10)
                 [["most_sig_direction", "most_sig_log2_freq_factor",
                   "most_sig_fisher_adj_pval", "fg_counts_upreg", "fg_counts_downreg"]])
        print(top.round(3))

    # Stash in adata.uns for downstream / sharing
    adata.uns["alphaphos_ksea_fisher"] = {k: v.copy() for k, v in ksea.items()}
"""

STEP_MEA_CODE = """# 2. GSEA-style MEA — uses the full rank, no threshold. Slower (permutation).
print("Running GSEA-style MEA (kinase_mea, 1000 permutations)...")
mea = kinase_mea(
    diff_results=results,
    sequence_lookup=adata.var["kinase_sequence"],
    id_col="protein",
    rank_col="log2fc",
    kl_method="percentile",
    kl_thresh=90,
    permutation_num=1000,
    seed=42,
)

# Top activated (positive NES) and inhibited (negative NES) kinases per type
for kt, df_kt in mea.items():
    nonna = df_kt.dropna(subset=["NES"]).sort_values("NES", ascending=False)
    print(f"\\n=== {kt} — MEA NES leaders ===")
    print(f"Top 10 ACTIVATED (positive NES):")
    print(nonna.head(10)[["NES", "p-value", "FDR", "Subs fraction"]].round(3))
    print(f"\\nTop 10 INHIBITED (negative NES):")
    print(nonna.tail(10)[::-1][["NES", "p-value", "FDR", "Subs fraction"]].round(3))

adata.uns["alphaphos_ksea_mea"] = {k: v.copy() for k, v in mea.items()}
"""

STEP_NETWORK_KSEA_MD = """## §12. Network-based KSEA (Saez-Rodriguez / decoupler-py + OmniPath)

Where §11 used **sequence-based PWMs** (Yaffe Kinase Library) to identify
kinase activity, §12 uses **curated kinase-substrate databases**
(PhosphoSitePlus, SIGNOR, KEA, NetworKIN, ...) via OmniPath as the
substrate set priors, and decoupler-py's statistical methods to score
per-kinase activity.

The two approaches are complementary:

| Question | Method |
|---|---|
| "What does the *motif* around this site predict?" | PWM-based (§11) |
| "What does the *biological literature* say this site is regulated by?" | Network-based (§12) |

Both should converge on the same conclusions for well-studied pathways.
For novel sites with no DB record, §11 still works but §12 can't score
them (no substrate set to match against).

`alphaphos.ksea` exposes four methods:

- **`kinase_activity_ulm`** — Univariate Linear Model (the classical
  KSEA z-score equivalent; recommended default)
- **`kinase_activity_mlm`** — Multivariate (joint fit; controls for
  shared substrates)
- **`kinase_activity_ora`** — Over-Representation Analysis (Fisher tail)
- **`kinase_activity_gsea`** — GSEA on the full rank
"""

STEP_NETWORK_KSEA_CODE = """from pathlib import Path
from alphaphos.ksea import fetch_omnipath_ks_network, kinase_activity_ulm

# 1. Fetch (and cache) the curated kinase-substrate network for human.
#    First call hits OmniPath over the network (10–30s). Subsequent calls
#    read the parquet cache.
cache = Path("D:/Projects/alphaPhos/test_data/ks_human.parquet")
net = fetch_omnipath_ks_network(
    organism="human",
    source_key="enzyme_genesymbol",
    target_id_style="uniprot",
    min_n_sources=1,        # keep all sources for max coverage
    cache_path=cache,
)
print(f"KS network: {len(net):,} edges, {net['source'].nunique():,} kinases, "
      f"{net['target'].nunique():,} unique sites")

# 2. Run ULM (the classical KSEA-z-score equivalent).
print("\\nRunning network-based KSEA (ULM)...")
ksea_net = kinase_activity_ulm(
    results,                     # diff_exp result from §8
    net,
    id_col="protein",
    stat_col="log2fc",
    min_targets=5,
    convert_site_ids=True,       # alphaPhos -> OmniPath site-id conversion
)

# 3. Sort + show top activated / inhibited
top_act = ksea_net.sort_values("score", ascending=False).head(15)
top_inh = ksea_net.sort_values("score").head(15)
print("\\nTop 15 ACTIVATED kinases (sorted by score desc):")
print(top_act.round(3))
print("\\nTop 15 INHIBITED kinases (sorted by score asc):")
print(top_inh.round(3))

# 4. Stash in adata.uns for downstream / sharing
adata.uns["alphaphos_ksea_network_ulm"] = ksea_net.copy()
"""

STEP_DOSE_MD = """## §13. Dose-response analysis (CurveCurator)

**Use this section only if your experiment is a dose-response study**
(varying concentrations of a drug/ligand at one or more timepoints).
Skip if your study is a binary contrast (e.g. +EGF vs -EGF — already
analysed in §8).

`alphaphos.dose_response.fit_dose_response` wraps the
[CurveCurator](https://github.com/kusterlab/curve_curator) CLI (Kuster
lab, TUM) which fits **4-parameter log-logistic** dose-response curves
to MS data and reports per-curve quality (pEC50, fold change, R²,
target-decoy FDR q-value).

Inputs needed from `adata.obs`:
- **`dose_col`** (default `"dose"`) — numeric, in nM. **DMSO/control
  samples must have dose == 0.**
- **`timepoint_col`** (default `"timepoint"`) — pass `None` for
  single-timepoint experiments. For multi-timepoint, CurveCurator is
  run **independently per timepoint** (CurveCurator doesn't fit
  dose×time jointly) and the curves are concatenated.

The orchestrator:
1. Builds CurveCurator's TSV input (sites as rows, `Raw <sample>`
   columns with linear intensities — converted from log2 internally).
2. Generates the TOML config (experimental design, fit parameters).
3. Runs `python -m curve_curator --fdr config.toml` as a subprocess.
4. Parses `curves.txt` back into a tidy pandas DataFrame.

For each (site, timepoint) you get: `pEC50`, `EC50_nM`,
`Curve Fold Change`, `Curve R2`, `Curve F_Value`, `Curve P_Value`, and
(when `fdr=True`) `Curve q_Value` + `Curve Regulation` (the FDR-call).

The interactive dashboard HTML is written to each per-timepoint
output dir (open in browser to interactively browse curves).
"""

STEP_DOSE_CODE = """from pathlib import Path
from alphaphos.dose_response import fit_dose_response

# Pre-flight: only run this section if obs has a 'dose' column with a 0-dose group
if "dose" not in adata.obs.columns or (adata.obs["dose"] == 0).sum() == 0:
    print("Skipping §13 — adata.obs has no 'dose' column with DMSO (dose==0) samples.")
    print("This pipeline section is for dose-response experiments only.")
else:
    out_root = Path("D:/Projects/alphaPhos/test_data/cc_runs")
    curves = fit_dose_response(
        adata,
        output_root=out_root,
        dose_col="dose",
        timepoint_col="timepoint" if "timepoint" in adata.obs.columns else None,
        condition="drug",
        description="alphaPhos pipeline walkthrough",
        max_missing=12,       # tweak based on n_wells
        available_cores=4,
        fdr=True,
        verbose=True,
    )

    print(f"\\nFit {len(curves):,} curves across {curves.get('timepoint', pd.Series([0])).nunique()} timepoint(s).")
    print("\\nTop 15 regulated sites (smallest q-value):")
    cols = [c for c in ("site_key","timepoint","pEC50","EC50_nM",
                         "Curve Fold Change","Curve R2","Curve q_Value")
            if c in curves.columns]
    print(curves.sort_values("Curve q_Value").head(15)[cols].round(3))

    # Stash for downstream / sharing
    adata.uns["alphaphos_dose_response"] = curves.copy()
"""

STEP_QC_MD = """## §14. QC dashboard (one-call HTML)

`alphaphos.qc.generate_dashboard(adata, output)` writes a single,
self-contained, interactive HTML file with all phospho-aware QC panels:

- **§1 Pipeline waterfall** — row counts at each filter step
- **§2 Sample QC grid** — S/T/Y composition, localization distribution,
  multiplicity, missingness, n_classI, per-condition CV by AA
- **§3 Reproducibility** — replicate correlation matrix (Pearson r),
  hierarchically clustered for sample order
- **§4 Class I + contaminants** — per-cell binary vs condition-aware
  policy comparison, optional contaminant breakdown
- **§5 Imputation** — MAR vs MNAR cells per sample (when hybrid imputer
  audit is provided)
- **§6 Provenance** — pipeline parameters from `adata.uns['alphaphos']`

Built on bokeh — every plot is hover/zoom/pan interactive. Open the
output HTML in any browser, share, or archive.
"""

STEP_QC_CODE = """from pathlib import Path
from alphaphos.qc import generate_dashboard

out_path = Path("D:/Projects/alphaPhos/test_data/qc_dashboard.html")

# Optional: pass the raw PSM df to enable the contaminant panel.
# (we already used df_raw with drop_contaminants=False if you have it;
# otherwise leave psm_df=None)
psm_df_optional = None  # or: read_psm(PSM_TSV, drop_contaminants=False, ...)

# Optional: pass impute_hybrid's audit DataFrame to enable the MAR/MNAR panel.
# Re-run impute with return_audit=True if you want this.
impute_audit_optional = None

dashboard_path = generate_dashboard(
    adata,
    output_path=out_path,
    psm_df=psm_df_optional,
    impute_audit=impute_audit_optional,
    title="alphaPhos QC | EGF nanoPhos benchmark",
)
print(f"Wrote dashboard: {dashboard_path}  ({dashboard_path.stat().st_size:,} bytes)")
print("Open it in your browser to inspect interactively.")
"""

STEP10_MD = """## §15. Save the analysed AnnData

`adata.write_h5ad` persists everything in one HDF5 file:
- `.X` (imputed log2 quants)
- `.obs` (sample metadata)
- `.var` (per-site metadata: structural + motif flags + site QC)
- `.layers["intensity_log2"]` (the imputed quants)
- `.layers["localization"]` (per-site, per-sample loc probs)
- `.uns["alphaphos"]` (pipeline provenance)
- `.uns["pca"]` (PCA result)
- `.obsm["X_pca"]` (PCA embedding)
- `.uns["classI_decision_table"]` (the per-condition Class-I decision table)

Differential results live in their own DataFrame — save alongside.
"""

STEP10_CODE = """adata_path = OUT_DIR / "pipeline_walkthrough_adata.h5ad"
adata.write_h5ad(adata_path)
print(f"AnnData saved to: {adata_path}")
print(f"  size: {adata_path.stat().st_size / 1e6:.1f} MB")

results_path = OUT_DIR / "pipeline_walkthrough_results.parquet"
results.to_parquet(results_path)
print(f"Differential results saved to: {results_path}")
print(f"  size: {results_path.stat().st_size / 1e6:.1f} MB")
"""


OUTRO_MD = """## What you have at this point

- `adata` — full quantitative matrix with phospho-specific metadata, ready
  for any scverse-compatible downstream tool
- `results` — differential analysis table (1 row per site)
- Two persisted files: `pipeline_walkthrough_adata.h5ad` +
  `pipeline_walkthrough_results.parquet`

## Natural next steps (none of these are shipped yet)

| What | Where it would live |
|---|---|
| Kinase activity inference (KSEA, Kinase Library) | `alphaphos.kinase` |
| Sequence motif logos + enrichment | `alphaphos.motif` |
| PhosphoSitePlus annotation enrichment | `alphaphos.annotate.phosphositeplus` |
| Per-S/T/Y stratified analysis | `alphaphos.stratify.by_aa` |
| Multiplicity-aware analysis (M1/M2/M3 co-regulation) | `alphaphos.stratify.multiplicity` |

See **`docs/benchmark/spectronaut_native_benchmark.md`** for the full validation
of every default in this pipeline against Spectronaut's native classI PTM site
report.
"""


# ============================================================================
# Build the notebook
# ============================================================================

def main():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(INTRO_MD),
        nbf.v4.new_code_cell(IMPORTS_CODE),
        nbf.v4.new_markdown_cell(STEP1_MD),
        nbf.v4.new_code_cell(STEP1_CODE),
        nbf.v4.new_markdown_cell(STEP2_MD),
        nbf.v4.new_code_cell(STEP2_CODE),
        nbf.v4.new_markdown_cell(STEP3_MD),
        nbf.v4.new_code_cell(STEP3_CODE),
        nbf.v4.new_markdown_cell(STEP4_MD),
        nbf.v4.new_code_cell(STEP4_CODE),
        nbf.v4.new_markdown_cell(STEP5_MD),
        nbf.v4.new_code_cell(STEP5_CODE),
        nbf.v4.new_markdown_cell(STEP6_MD),
        nbf.v4.new_code_cell(STEP6_CODE),
        nbf.v4.new_markdown_cell(STEP7_MD),
        nbf.v4.new_code_cell(STEP7_CODE),
        nbf.v4.new_markdown_cell(STEP8_MD),
        nbf.v4.new_code_cell(STEP8_CODE),
        nbf.v4.new_markdown_cell(STEP9_MD),
        nbf.v4.new_code_cell(STEP9_CODE),
        nbf.v4.new_code_cell(STEP9_CODE_TOP),
        nbf.v4.new_markdown_cell(STEP_KIN_MD),
        nbf.v4.new_code_cell(STEP_KIN_CODE),
        nbf.v4.new_markdown_cell(STEP_KSEA_MD),
        nbf.v4.new_code_cell(STEP_KSEA_CODE),
        nbf.v4.new_code_cell(STEP_MEA_CODE),
        nbf.v4.new_markdown_cell(STEP_NETWORK_KSEA_MD),
        nbf.v4.new_code_cell(STEP_NETWORK_KSEA_CODE),
        nbf.v4.new_markdown_cell(STEP_DOSE_MD),
        nbf.v4.new_code_cell(STEP_DOSE_CODE),
        nbf.v4.new_markdown_cell(STEP_QC_MD),
        nbf.v4.new_code_cell(STEP_QC_CODE),
        nbf.v4.new_markdown_cell(STEP10_MD),
        nbf.v4.new_code_cell(STEP10_CODE),
        nbf.v4.new_markdown_cell(OUTRO_MD),
    ]
    # Standard kernelspec
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}

    NB_PATH.parent.mkdir(exist_ok=True, parents=True)
    nbf.write(nb, NB_PATH)
    print(f"Wrote {NB_PATH} with {len(nb.cells)} cells.")


if __name__ == "__main__":
    main()
