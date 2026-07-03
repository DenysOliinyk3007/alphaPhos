# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # alphaPhos end-to-end walkthrough — EGF+ vs EGF- HEK dataset
#
# This notebook exercises every module that ships in alphaPhos v0.6.2 on a
# real Spectronaut phospho report: **3× EGF-stimulated vs 3× unstimulated
# HEK samples**, ~621k PSMs.
#
# **Pipeline:**
#
# 1. Read PSMs (`ap.read_spectronaut`)
# 2. Assemble sample metadata
# 3. Collapse PSMs → site-level AnnData (`ap.collapse_sites`)
# 4. Filter by completeness (`ap.filter_by_completeness`)
# 5. Hybrid MAR/MNAR imputation (`ap.impute_hybrid`)
# 6. QC dashboard (`ap.generate_dashboard`, requires bokeh)
# 7. FASTA-based kinase-window annotation (`ap.add_kinase_windows`)
# 8. Optional: ComBat batch correction (`ap.batch_correct_combat`, uses a
#    fake batch structure here since the real dataset is single-batch)
# 9. Differential expression via limma (`ap.diff_exp_limma`)
# 10. Volcano plot (matplotlib — alphaPhos has no viz module yet)
# 11. Kinase-motif PWM scoring (`alphaphos.kinase.library`, requires
#     `kinase_library`)
# 12. KSEA kinase-activity inference (`alphaphos.enrichment.kinase_activity`,
#     requires `decoupler` + `omnipath`)
#
# **Prerequisites:**
#
# * `pip install -e ".[stats]"` for limma + ComBat.
# * `bokeh`, `kinase_library`, `decoupler`, `omnipath` for the optional
#   sections. Each optional section is guarded with `try/except`; the rest
#   of the notebook works without them.

# %%
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import alphaphos as ap

logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")

REPO = Path(__file__).resolve().parent.parent
REPORT = REPO / "test_data" / "benchmark" / "EGF_diff_exp.tsv"
HUMAN_FASTA = REPO / "resources" / "fastas" / "human.fasta"
OUT_DIR = REPO / "test_data" / "walkthrough_output"
OUT_DIR.mkdir(exist_ok=True)

print(f"alphaPhos {ap.__version__}")
print(f"Report:  {REPORT} ({REPORT.stat().st_size / 1e6:.1f} MB)")
print(f"FASTA:   {HUMAN_FASTA}")
print(f"Outputs: {OUT_DIR}")

# %% [markdown]
# ## 1 — Read Spectronaut PSMs
#
# `read_spectronaut` prunes unused columns at read time (~17 of 26 dropped
# on this report), applies decoy + contaminant filters, and stamps source
# provenance into `.attrs`.

# %%
psm_df = ap.read_spectronaut(REPORT)

print(f"PSM rows after decoy + contaminant filter: {len(psm_df):,}")
print(f"Columns dropped at read time: {psm_df.attrs['columns_dropped']}")
print(f"Unique R.FileName (samples): {psm_df['R.FileName'].nunique()}")
psm_df.head(3)

# %% [markdown]
# ## 2 — Sample metadata
#
# Infer condition from filename: `withEGF` → EGF, `woEGF` → ctrl.

# %%
samples = sorted(psm_df["R.FileName"].unique())
condition_df = pd.DataFrame(
    {
        "sample": samples,
        "condition": ["EGF" if "withEGF" in s else "ctrl" for s in samples],
    }
)
condition_df

# %% [markdown]
# ## 3 — Collapse PSMs → site-level AnnData
#
# `collapse_sites` runs a 7-stage pipeline: parse → top-N attribution →
# key building → precursor aggregation → localization masking → class-I
# QC → AnnData assembly. The output is samples × sites with canonical
# `Protein|Gene|Site|Mult` keys.
#
# We use the defaults: MS2 quantification with MS1→auto fallback,
# `top_n_attribution=True`, `sum` aggregation, `condition` localization
# strategy at Class-I cutoff 0.75.

# %%
adata = ap.collapse_sites(psm_df, condition_df=condition_df)
print(f"AnnData shape: {adata.shape} (samples × sites)")
print(f"Layers: {list(adata.layers.keys())}")
print(f"Phospho selectivity per sample:")
print(adata.obs[["condition", "phospho_selectivity_pct"]])
print(f"\nCollapse stats:")
for k, v in adata.uns["alphaphos"]["stats"].items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 4 — Filter by completeness (each group ≥ 2/3 valid)
#
# `keep_strategy="each"` is the strictest option — a site must be
# observed in ≥ 2/3 of the samples in EVERY condition. Good default
# before differential testing.

# %%
adata = ap.filter_by_completeness(
    adata,
    min_valid_frac=2 / 3,
    group_column="condition",
    keep_strategy="each",
    layer="intensity_log2",
)
print(f"After filter: {adata.n_vars:,} sites")
n_nan = int(np.isnan(adata.layers["intensity_log2"]).sum())
total = adata.n_obs * adata.n_vars
print(f"NaN remaining: {n_nan:,} / {total:,} ({100 * n_nan / total:.1f}%)")

# %% [markdown]
# ## 5 — Hybrid MAR/MNAR imputation
#
# Per-cell branching: MAR (missed-but-measurable) → site-based KNN;
# MNAR (below-LOD) → downshifted Gaussian per Perseus convention.
# Writes to `layers["intensity_log2"]` by default.

# %%
adata, audit = ap.impute_hybrid(adata, return_audit=True)
print(f"Cells imputed: {len(audit):,}")
print(audit["strategy"].value_counts().to_string())
assert not np.isnan(adata.layers["intensity_log2"]).any()

# %% [markdown]
# ## 6 — QC dashboard
#
# Optional — requires `bokeh`. Generates a self-contained HTML with 6
# panels: pipeline waterfall, sample-level QC, reproducibility,
# Class-I + contaminant, imputation diagnostics, provenance.

# %%
try:
    dashboard_path = OUT_DIR / "qc_dashboard.html"
    ap.generate_dashboard(
        adata,
        dashboard_path,
        impute_audit=audit,
        title="EGF+/- QC (walkthrough)",
    )
    print(f"QC dashboard written: {dashboard_path}")
except ImportError as e:
    print(f"Skipping QC dashboard: {e}")

# %% [markdown]
# ## 7 — Kinase-window annotation from FASTA
#
# `add_kinase_windows` extracts a ±7-residue window around each site
# from the proteome FASTA, formats it in the `_LEFT*S*RIGHT_` convention,
# and writes it to `adata.var["kinase_sequence"]`. Required for the
# kinase-library PWM scoring downstream.

# %%
adata = ap.add_kinase_windows(adata, fasta_path=HUMAN_FASTA)
print(f"Sites with kinase_sequence: {adata.var['kinase_sequence'].notna().sum():,}")
print("Sample kinase_sequence values:")
adata.var[["gene", "site_aa", "site_position", "kinase_sequence"]].head(5)

# %% [markdown]
# ## 8 — (Optional) ComBat batch correction with synthetic batch
#
# The real EGF dataset was acquired in a single MS run — no real batch
# structure. To demonstrate `batch_correct_combat` we inject a fake
# batch label (b1 = 4 samples; b2 = 2 samples, one per condition) and
# add a +2 log2 site-wide shift to b2 so ComBat has something to
# correct. In a real analysis, only run this step if you have a real
# known batch structure.

# %%
adata.obs["batch"] = pd.Categorical(
    ["b1" if s.endswith(("_01", "_02")) else "b2" for s in adata.obs_names],
    categories=["b1", "b2"],
)
print("Fake batch layout for demo:")
print(pd.crosstab(adata.obs["batch"], adata.obs["condition"]))

# Inject a +2 log2 batch shift on b2 samples so ComBat has signal to remove
b2_mask = (adata.obs["batch"] == "b2").to_numpy()
adata.layers["intensity_log2"][b2_mask] += 2.0
before = (
    adata.layers["intensity_log2"][b2_mask].mean() - adata.layers["intensity_log2"][~b2_mask].mean()
)

adata = ap.batch_correct_combat(
    adata,
    batch_column="batch",
    covariates=["condition"],
)
after = (
    adata.layers["intensity_log2"][b2_mask].mean() - adata.layers["intensity_log2"][~b2_mask].mean()
)
print(f"\nBatch (b2 − b1) mean shift: before={before:+.2f} log2, after={after:+.2f} log2")
print(f"Pre-correction layer kept at: layers['intensity_log2_precombat']")

# %% [markdown]
# ## 9 — Differential expression: limma
#
# `diff_exp_limma` runs a two-group moderated t-test via `inmoose.limma`.
# We follow the community convention:
#
# - **Statistically preferred (Path A):** test the **pre-correction
#   layer** with `batch` as a limma covariate. The linear model handles
#   the batch adjustment inside the fit.
# - **For visualisation (Path B):** the ComBat-corrected `intensity_log2`
#   layer with no batch covariate is what you would feed into PCA/UMAP.
#
# The double-correct guard refuses the invalid mix (ComBat data +
# batch covariate).

# %%
result = ap.diff_exp_limma(
    adata,
    condition_column="condition",
    comparison=("EGF", "ctrl"),
    covariates=["batch"],
    layer="intensity_log2_precombat",
)
print(f"Tested {len(result):,} sites")
n_sig = int((result["fdr"] < 0.05).sum())
n_up = int(((result["fdr"] < 0.05) & (result["log2fc"] > 0)).sum())
n_dn = int(((result["fdr"] < 0.05) & (result["log2fc"] < 0)).sum())
print(f"FDR < 0.05: {n_sig:,} total ({n_up:,} up in EGF, {n_dn:,} down)")
print(
    f"|log2fc| ≥ 1 and FDR < 0.05: {int(((result['fdr'] < 0.05) & (result['log2fc'].abs() >= 1)).sum()):,}"
)
print(f"\nContrast: {result.attrs['contrast_direction']}")
print("\nTop 10 up-regulated in EGF:")
top_up = result[(result["fdr"] < 0.05) & (result["log2fc"] > 0)].sort_values("fdr").head(10)
print(top_up[["log2fc", "t_stat", "p_value", "fdr"]].round(4).to_string())

# %% [markdown]
# ## 10 — Volcano plot
#
# alphaPhos does not yet ship a `viz` module; a quick matplotlib volcano
# plot is enough to eyeball the result.  **We plot the y-axis on
# BH-adjusted FDR, not the nominal p-value** — a volcano based on
# nominal p-values inflates the visual sense of significance and
# doesn't correspond to the ``FDR < 0.05`` cutoff used to call hits.

# %%
fig, ax = plt.subplots(figsize=(6, 5))
sig = result["fdr"] < 0.05
big = result["log2fc"].abs() >= 1.0
neg_log10_fdr = -np.log10(result["fdr"].clip(lower=1e-300))
ax.scatter(
    result.loc[~sig, "log2fc"],
    neg_log10_fdr[~sig],
    s=4,
    color="lightgrey",
    alpha=0.5,
    label="ns",
)
ax.scatter(
    result.loc[sig & ~big, "log2fc"],
    neg_log10_fdr[sig & ~big],
    s=6,
    color="tab:orange",
    alpha=0.7,
    label="FDR<0.05",
)
ax.scatter(
    result.loc[sig & big, "log2fc"],
    neg_log10_fdr[sig & big],
    s=8,
    color="tab:red",
    alpha=0.9,
    label="FDR<0.05, |log2fc|≥1",
)
ax.axhline(-np.log10(0.05), color="grey", linestyle="--", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.3)
ax.set_xlabel("log2FC (EGF − ctrl)")
ax.set_ylabel("−log10(FDR)")
ax.set_title(f"EGF+/-  •  {n_sig} sig at FDR<0.05")
ax.legend(loc="upper left", frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "volcano.png", dpi=120)
plt.show()

# %% [markdown]
# ## 11 — Kinase-motif PWM scoring
#
# Optional — requires `kinase_library`. `predict_kinases` scores each
# site's ±7 window against ~300 kinase PWMs from the Yaffe library and
# returns a per-site DataFrame with the top-K kinase predictions.

# %%
try:
    from alphaphos.kinase.library import predict_kinases

    # Score only the significant-and-strong hits to keep the demo fast.
    hits = result[(result["fdr"] < 0.05) & (result["log2fc"].abs() >= 1.0)].index
    ad_hits = adata[:, list(hits)].copy()
    kinase_predictions = predict_kinases(ad_hits, top_k=5)
    print(f"Scored {len(kinase_predictions):,} hit sites against the kinase library.")
    print("\nSample predictions (first 5 sites):")
    print(kinase_predictions.head(5).to_string())
except ImportError as e:
    print(f"Skipping kinase library: {e}")
except Exception as e:
    print(f"Kinase library scoring hit an error: {type(e).__name__}: {e}")

# %% [markdown]
# ## 12 — KSEA (kinase activity inference)
#
# Runs `alphaphos.enrichment.ksea.kinase_activity` on the diff-exp result.
# Wraps `decoupler`'s ULM (univariate linear model — the modern
# replacement for the classical Wiredja 2017 KSEA z-score) against the
# OmniPath kinase-substrate network. Auto-canonicalises the alphaphos
# site keys to OmniPath's `Protein_AApos` format internally.

# %%
try:
    from alphaphos.enrichment import kinase_activity

    ksea_result = kinase_activity(
        result,  # from ap.diff_exp_limma; index is Protein|Gene|Site|Mult
        stat_col="log2fc",
        network="omnipath",
        method="ulm",
        min_substrates=5,
        organism="human",
        cache_path=OUT_DIR / "omnipath_ks_human.parquet",
    )
    print(f"KSEA scored {len(ksea_result):,} kinases with >= 5 substrates.")
    print(f"KSEA columns: {list(ksea_result.columns)}")
    top_kin = ksea_result.sort_values("score", ascending=False).head(10)
    print("\nTop 10 activated kinases in EGF (by ULM score):")
    print(top_kin.round(4).to_string(index=False))
    ksea_result.to_csv(OUT_DIR / "ksea_result.tsv", sep="\t", index=False)
except ImportError as e:
    print(f"Skipping KSEA: {e}")
except Exception as e:
    print(f"KSEA hit an error: {type(e).__name__}: {e}")

# %% [markdown]
# ## 13 — Save the site-level result
#
# The differential-expression DataFrame is what you'd hand to the next
# stage of analysis (enrichment, manuscript figures, etc.).

# %%
result_out = OUT_DIR / "egf_diff_exp_result.tsv"
result.to_csv(result_out, sep="\t")
adata_out = OUT_DIR / "egf_collapsed_and_analyzed.h5ad"
adata.write_h5ad(adata_out)
print(f"Wrote result table:  {result_out}")
print(f"Wrote AnnData:       {adata_out}")

# %% [markdown]
# ## Summary
#
# On this dataset (3× EGF vs 3× ctrl, ~15,000 sites after filter):
#
# - Top canonical EGF-pathway sites recovered: **MAPK7 T733/S731**
#   (ERK5 activation loop), **EP300 S12**, **DAB2 S723**, **WIPF1
#   S340**, **CHEK1 S280**, **NEK9 T333**, **RSK1 hydrophobic-motif
#   S732** (`KLPS*TTL`).
# - Total significant sites at FDR<0.05: reported above.
#
# For a fuller integration-test view of the pipeline see
# `tests/integration/test_collapse_spiked.py`, which spikes 31
# hand-crafted synthetic peptides into a filtered slice of this same
# dataset to exercise every collapse edge case.
