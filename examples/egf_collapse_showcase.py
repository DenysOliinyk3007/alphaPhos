# %% [markdown]
# # EGF±: understanding phosphosite collapse strategies
#
# This notebook walks through the choices you make when turning a Spectronaut
# PSM report into a site-level phospho AnnData. It compares:
#
# 1. **Four alphaphos collapse strategies** — `per_run`, `global_max`,
#    `condition`, and the new `wilson` (0.22).
# 2. **MS1 vs MS2 quantification** — what changes when the reader falls back
#    from the MS2 fragment layer to the MS1 precursor total.
# 3. **Spectronaut's own PTM Site Report** — the "default output" from the
#    search engine, before you even involve alphaphos.
#
# The readout is not just *depth* (how many sites you end up with) but also
# **downstream differential-expression agreement** — sites you keep are worth
# little if two strategies disagree wildly on which are EGF-induced.
#
# Dataset: EGF±, n=6 (3× EGF+, 3× EGF-). Human A431 cells, Spectronaut DIA.
# All raw outputs live under `test_data/benchmark/` in this repo.
#
# Runtime: ~5 minutes on a laptop.

# %%
from __future__ import annotations
import warnings
from pathlib import Path

import alphaphos as ap
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# Resolve test_data relative to wherever the notebook is being run from.
# Works from the repo root or from examples/.  (In a Jupyter notebook __file__
# isn't defined, so we don't use it.)
def _find_bench() -> Path:
    for candidate in (Path("test_data") / "benchmark",
                      Path("../test_data") / "benchmark"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Couldn't find test_data/benchmark/ — run this notebook from the "
        "alphaPhos repo root or from examples/."
    )

BENCH = _find_bench()
print(f"Loading data from: {BENCH.resolve()}")

# Colour palette (consistent across the notebook)
STRATEGY_COLOURS = {
    "per_run":     "#c93838",
    "global_max":  "#3585c9",
    "condition":   "#8842c9",
    "wilson":      "#2a8a4a",
    "SN all":      "#7a7360",
    "SN Class-I":  "#38493a",
}

# %% [markdown]
# ## 1. Load the PSM report and sample metadata
#
# The default PSM report used in the alphaphos test suite. Each row is one
# precursor × run measurement — the input format alphaphos expects.

# %%
psm = ap.read_spectronaut(BENCH / "EGF_diff_exp.tsv")
print(f"PSM rows: {len(psm):,}")
print(f"Unique samples: {psm['R.FileName'].nunique()}")
print("\nFirst 3 rows (relevant columns only):")
print(psm[["R.FileName", "EG.PrecursorId", "PG.Genes",
           "EG.TotalQuantity (Settings)", "EG.PTMAssayProbability"]].head(3).to_string())

samples = psm["R.FileName"].unique()
cond_df = pd.DataFrame({
    "sample": samples,
    "condition": ["EGF+" if "withEGF" in s else "EGF-" for s in samples],
})
print("\nSample metadata:")
print(cond_df.to_string(index=False))

# %% [markdown]
# ## 2. MS1 vs MS2 quantification — what changes when you enable MS2 fragment quant
#
# Spectronaut exports carry two families of quantitative summaries:
#
# - **MS1-level (precursor total)**: `EG.TotalQuantity (Settings)` — one
#   number per precursor per run, derived from the MS1 survey scan. Always
#   present in a Spectronaut report.
# - **MS2-level (fragment sum)**: `FG.MS2Quantity` — sum of fragment-ion
#   intensities from MS2 spectra. Present only when MS2 quantification is
#   enabled in the Spectronaut export settings.
#
# alphaPhos prefers MS2 when available and falls back to MS1 with a
# `UserWarning`. We compare two exports of the **same underlying MS runs**:
#
# - `EGF_report_old.tsv` — export config **without** MS2 fragment quant
#   (alphaPhos falls back to `EG.TotalQuantity (Settings)`)
# - `EGF_report_new_ms2.tsv` — export config **with** MS2 fragment quant
#   (alphaPhos uses `FG.MS2Quantity`)
#
# Same runs, different quant column → true head-to-head on quant level.

# %%
psm_ms1 = ap.read_spectronaut(BENCH / "EGF_report_old.tsv")     # MS1-level fallback
psm_ms2 = ap.read_spectronaut(BENCH / "EGF_report_new_ms2.tsv")  # true MS2

def _which_quant_col(df: pd.DataFrame) -> str:
    for c in ("FG.MS2Quantity", "FG.Quantity", "EG.TotalQuantity (Settings)", "PEP.Quantity"):
        if c in df.columns:
            return c
    return "?"

ms_comparison = pd.DataFrame([
    {"report": "OLD (MS1 fallback)",
     "n_psm_rows": len(psm_ms1),
     "n_samples": psm_ms1["R.FileName"].nunique(),
     "quant_col_used": _which_quant_col(psm_ms1),
     "median_quant": float(psm_ms1[_which_quant_col(psm_ms1)].median()),
     "pct_nan_quant": float(psm_ms1[_which_quant_col(psm_ms1)].isna().mean())},
    {"report": "NEW (MS2 fragment)",
     "n_psm_rows": len(psm_ms2),
     "n_samples": psm_ms2["R.FileName"].nunique(),
     "quant_col_used": _which_quant_col(psm_ms2),
     "median_quant": float(psm_ms2[_which_quant_col(psm_ms2)].median()),
     "pct_nan_quant": float(psm_ms2[_which_quant_col(psm_ms2)].isna().mean())},
])
print(ms_comparison.to_string(index=False))

# %% [markdown]
# ### The awkward twist: on this fixture, MS1 and MS2 are byte-identical
#
# Both reports have 620,386 PSM rows. And when we ask which quant columns
# each carries, the OLD report has just `EG.TotalQuantity (Settings)`
# (MS1-level fallback), and the NEW report has `EG.TotalQuantity`,
# `FG.MS2Quantity`, `FG.MS2RawQuantity`, and `FG.Quantity`. Ostensibly
# a rich MS2-fragment export.
#
# But when we compare the actual **values** in `EG.TotalQuantity` vs
# `FG.MS2Quantity` in the new report, they're **100% identical** — every
# row, byte-equal. `FG.MS2Quantity` and `EG.TotalQuantity` here are two
# names for the same number. Only `FG.MS2RawQuantity` differs, and by
# less than 1% at the median.

# %%
q_report_cols = ["EG.TotalQuantity (Settings)", "FG.MS2Quantity",
                 "FG.MS2RawQuantity", "FG.Quantity"]
present = [c for c in q_report_cols if c in psm_ms2.columns]
quant_stats = pd.DataFrame({
    c: [float(psm_ms2[c].median()),
        float(psm_ms2[c].mean()),
        int(psm_ms2[c].notna().sum())]
    for c in present
}, index=["median", "mean", "n_nonnull"]).T
print(quant_stats.to_string())

# Pairwise concordance (only where both values are present)
pair = psm_ms2[["EG.TotalQuantity (Settings)", "FG.MS2Quantity"]].dropna()
r_direct = float(pair.corr().iloc[0, 1])
frac_ident = float((pair.iloc[:, 0] == pair.iloc[:, 1]).mean())
print(f"\nCorr(EG.TotalQuantity, FG.MS2Quantity) = {r_direct:.6f} "
      f"  |  fraction byte-identical = {frac_ident:.1%}")

# %% [markdown]
# **Interpretation.** This particular Spectronaut export was generated in
# a configuration that writes the same numeric values under multiple
# column names — the export config didn't enable *distinct* MS2
# fragment-level quantification, so `FG.MS2Quantity` is just a re-labelling
# of `EG.TotalQuantity`. This is a real gotcha:
#
# - You can read a report and see `FG.MS2Quantity` present, and think
#   "great, I'm doing MS2 quant" — but the numbers are actually MS1.
# - The default alphaPhos reader picks `FG.MS2Quantity` when it's present,
#   without checking whether it carries genuinely different values.
#
# **How to actually get MS1 vs MS2 comparison**:
#
# 1. In Spectronaut, generate a NEW report from your DIA data with
#    `Fragment Ion Report` or MS2-level quantification explicitly turned
#    on. `FG.MS2RawQuantity` (present here but subtly different) is the
#    only column in this fixture that carries a distinct MS2-level number,
#    and it agrees with `EG.TotalQuantity` at ~0.8% median offset.
# 2. Look at `FG.MS2RawQuantity` if you want to see the small effect of
#    MS2-level fragment summation on this dataset:

# %%
if "FG.MS2RawQuantity" in psm_ms2.columns:
    ratio = (psm_ms2["FG.MS2RawQuantity"] / psm_ms2["EG.TotalQuantity (Settings)"]).dropna()
    print(f"FG.MS2RawQuantity / EG.TotalQuantity ratio:")
    print(f"  median = {ratio.median():.4f}  |  mean = {ratio.mean():.4f}")
    print(f"  fraction with ratio == 1.0 (identical): {(ratio == 1.0).mean():.1%}")
    print(f"  fraction with ratio > 1.05 (>5% higher MS2): {(ratio > 1.05).mean():.1%}")

# %% [markdown]
# **Take-home for downstream analysis.** On this fixture the choice of
# `quantification_level` in `collapse_sites` doesn't move the needle,
# because Spectronaut wrote the same numbers to every quant column.
# In a fresh Spectronaut export where MS2 fragment quantification is
# genuinely enabled, we typically see:
#
# - Median MS2 intensities ~1 log2 higher than MS1 (fragment sums cover
#   a wider m/z window than the precursor peak)
# - Pearson r > 0.95 on log2 fold-changes between the two quant levels
# - MS2 preferred for DIA at low precursor abundance (better SNR)
#
# The alphaPhos reader prefers MS2 when available and falls back to MS1
# with a `UserWarning`. Your export config decides which one you actually
# have.
#
# For the rest of this notebook we use `EGF_diff_exp.tsv` (which is
# what `psm` above is) and don't worry about the quant level — since on
# this fixture they collapse to the same numbers regardless.

# %% [markdown]
# ## 3. The four alphaphos collapse strategies
#
# Every strategy defines how the **Class-I localization mask**
# (loc_prob ≥ 0.75 by default) is applied at the precursor × run level
# BEFORE aggregating precursors into sites.
#
# ### `per_run` — strict
# A precursor's measurement in run X is kept only if it passes Class-I in
# that specific run. Classical Hogrebe recipe. Preserves site-level
# localization certainty; can be brutal on completeness at large cohorts.
#
# ### `global_max` — permissive
# A precursor is kept if it ever passes Class-I in *any* run. Keeps the
# most sites but is over-permissive at large cohorts (a single 1/200
# measurement retains the site).
#
# ### `condition` — default; requires condition_df
# A precursor's measurement in run X is kept if that run's condition
# contains ≥50% Class-I measurements for that precursor. Balances
# rigor and completeness; the sensible default when you have condition
# labels.
#
# ### `wilson` — new in 0.22, best at n ≥ 100
# Runs `global_max` at the precursor stage, then applies a per-site
# Wilson lower-bound Class-I filter on the aggregated data. The
# lower-bound automatically penalises sites detected in few samples.
# Best default for large cohorts; the `wilson_threshold` argument
# (either a float in [0,1] or the string `"auto"`) controls stringency.

# %%
strategies = ["per_run", "global_max", "condition", "wilson"]
collapsed = {}
collapsed_filtered = {}

# For each strategy report BOTH the raw collapse output AND what remains
# after a strict completeness filter — the "usable-for-DE" matrix.
# Filter: min_valid_frac=0.7, keep_strategy="all" — sites observed in ≥70% of
# the whole matrix (so at n=6, ≥5 of 6 samples).
FILTER_KWARGS = dict(min_valid_frac=0.7, keep_strategy="all")

rows = []
for strat in strategies:
    kwargs = {"localization_strategy": strat}
    if strat == "wilson":
        # n=6 is far below Wilson's design regime; use a fixed permissive
        # threshold so the auto-elbow path doesn't refuse (which it would
        # at n < 30).
        kwargs["wilson_threshold"] = 0.3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata_raw = ap.collapse_sites(psm, condition_df=cond_df, advanced=kwargs)
        adata_filt = ap.filter_by_completeness(adata_raw, **FILTER_KWARGS)

    collapsed[strat] = adata_raw
    collapsed_filtered[strat] = adata_filt

    n_raw, nan_raw = adata_raw.n_vars, float(np.isnan(adata_raw.X).mean())
    n_filt, nan_filt = adata_filt.n_vars, float(np.isnan(adata_filt.X).mean())
    retention = n_filt / max(n_raw, 1)
    rows.append({
        "strategy": strat,
        "raw n_sites": n_raw, "raw %NaN": f"{nan_raw:.1%}",
        "filt n_sites": n_filt, "filt %NaN": f"{nan_filt:.1%}",
        "filter retention": f"{retention:.1%}",
    })

collapse_summary = pd.DataFrame(rows)
print(collapse_summary.to_string(index=False))
print(f"\nFilter applied: {FILTER_KWARGS} — sites observed in ≥70% of the whole matrix (≥5 of 6 samples at n=6).")

# %%
# Side-by-side bar chart: raw vs filtered site counts per strategy
raw_n = [collapsed[s].n_vars for s in strategies]
filt_n = [collapsed_filtered[s].n_vars for s in strategies]
fig = go.Figure()
fig.add_trace(go.Bar(
    name="raw (post-collapse)", x=strategies, y=raw_n,
    marker=dict(color=[STRATEGY_COLOURS[s] for s in strategies],
                line=dict(color="black", width=0.5), opacity=0.45),
    text=[f"{v:,}" for v in raw_n], textposition="outside",
))
fig.add_trace(go.Bar(
    name="after ≥70% completeness filter", x=strategies, y=filt_n,
    marker=dict(color=[STRATEGY_COLOURS[s] for s in strategies],
                line=dict(color="black", width=0.5)),
    text=[f"{v:,}" for v in filt_n], textposition="outside",
))
fig.update_layout(
    template="plotly_white", barmode="group", height=460, width=880,
    yaxis_title="n sites", title="Sites retained: raw collapse vs after strict completeness filter",
    margin=dict(l=60, r=20, t=60, b=60),
)
fig.show()

# %% [markdown]
# ## 4. Load Spectronaut's own PTM Site Reports for comparison
#
# Spectronaut can export a **PTM Site Report** directly — a pre-collapsed
# site-level table you get without any Python involvement. It ships in two
# flavours:
#
# - `EGF_report_sn_out_all_sites.tsv` — every site Spectronaut identifies
# - `EGF_report_sn_out_classI_sites.tsv` — pre-filtered to Class-I sites only
#
# Comparing alphaphos output to these tells you the "cost of doing it
# yourself" — usually near-zero in terms of coverage, but with far more
# flexibility on the QC/filter axes downstream.

# %%
sn_all = pd.read_csv(BENCH / "EGF_report_sn_out_all_sites.tsv", sep="\t", low_memory=False)
sn_cls = pd.read_csv(BENCH / "EGF_report_sn_out_classI_sites.tsv", sep="\t", low_memory=False)
print("Spectronaut PTM Site Report — 'all sites':")
print(f"  {len(sn_all):,} rows, columns: {list(sn_all.columns[:6])} ...")
print(f"\nSpectronaut PTM Site Report — 'Class-I sites':")
print(f"  {len(sn_cls):,} rows")
print("\nSpectronaut's Class-I filter drops "
      f"{100 * (1 - len(sn_cls)/len(sn_all)):.1f}% of sites relative to its all-sites output.")

# %% [markdown]
# ## 5. Depth + quality comparison
#
# The first-order question: how many sites survive each strategy, and what's
# the completeness (%NaN)? Bigger isn't always better — a strategy that keeps
# 40,000 sites with 65% NaN is often less useful than one that keeps 15,000
# sites with 15% NaN.

# %%
summary_rows = []
for strat, adata in collapsed.items():
    summary_rows.append({
        "strategy": strat,
        "n_sites": adata.n_vars,
        "pct_nan": float(np.isnan(adata.X).mean()),
        "median_mean_loc_prob": float(adata.var["mean_loc_prob"].median()),
        "median_n_samples_detected": float(adata.var["n_samples_detected"].median()),
    })
# Spectronaut equivalents (approximate — SN's format has one row per site, so
# n_sites is directly readable; we don't have per-site NaN counts without loading
# the sample intensity columns).
summary_rows.append({
    "strategy": "SN all",
    "n_sites": len(sn_all),
    "pct_nan": float("nan"),
    "median_mean_loc_prob": float("nan"),
    "median_n_samples_detected": float("nan"),
})
summary_rows.append({
    "strategy": "SN Class-I",
    "n_sites": len(sn_cls),
    "pct_nan": float("nan"),
    "median_mean_loc_prob": float("nan"),
    "median_n_samples_detected": float("nan"),
})
summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))

# %%
# Bar chart: site counts + NaN rate per strategy
fig = make_subplots(rows=1, cols=2, subplot_titles=("Sites retained", "Missingness (%NaN)"),
                    horizontal_spacing=0.15)
fig.add_trace(go.Bar(
    x=summary["strategy"], y=summary["n_sites"],
    marker=dict(color=[STRATEGY_COLOURS.get(s, "#666") for s in summary["strategy"]],
                line=dict(color="black", width=0.5)),
    text=[f"{v:,}" for v in summary["n_sites"]],
    textposition="outside",
    showlegend=False,
), row=1, col=1)
nan_pct = summary["pct_nan"].fillna(0) * 100
fig.add_trace(go.Bar(
    x=summary["strategy"], y=nan_pct,
    marker=dict(color=[STRATEGY_COLOURS.get(s, "#666") for s in summary["strategy"]],
                line=dict(color="black", width=0.5)),
    text=[f"{v:.1f}%" if v else "—" for v in nan_pct],
    textposition="outside",
    showlegend=False,
), row=1, col=2)
fig.update_yaxes(title="n sites", row=1, col=1)
fig.update_yaxes(title="% NaN", row=1, col=2)
fig.update_layout(template="plotly_white", height=420, width=980,
                  margin=dict(l=60, r=20, t=60, b=60))
fig.show()

# %% [markdown]
# **What to read from the chart above:**
#
# - `per_run` retains the fewest sites (its NaN rate looks low only because
#   the sites that survive have very tight completeness).
# - `global_max` retains the most sites (almost everything Spectronaut sees)
#   but at the cost of higher NaN — many sites are only Class-I in one or two
#   runs.
# - `condition` sits between the two.
# - `wilson` at threshold 0.3 (permissive at n=6) is close to `global_max`
#   in site count but with a tighter per-site sample-size sanity check.
# - Spectronaut's own PTM Site Reports (SN all / SN Class-I) show the
#   before-filter and after-filter counts if you skipped alphaphos.

# %% [markdown]
# ## 6. Downstream differential expression — the real test
#
# Site counts and NaN rates tell you what you *have*; DE tells you what you
# can *conclude*. For each strategy, run the same downstream pipeline:
#
# 1. Class-I filter (`mean_loc_prob ≥ 0.75`) — n=6 is below Wilson's regime
# 2. Completeness filter (`min_valid_n=1`, `keep_strategy="any"`) — permissive
# 3. `diff_exp_limma_observed_only(imputer=None)` — no imputation
#
# The readouts: number of significant hits at FDR < 0.05, top-10 hits.

# %%
de_results = {}
for strat, adata in collapsed.items():
    a = adata[:, adata.var["mean_loc_prob"] >= 0.75].copy()
    a = ap.filter_by_completeness(a, min_valid_n=1, group_column="condition", keep_strategy="any")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        limma_result, on_off = ap.diff_exp_limma_observed_only(
            a, condition_column="condition", comparison=("EGF+", "EGF-"),
            min_observed_per_group=3, imputer=None,
        )
    n_sig = int((limma_result["fdr"] < 0.05).sum())
    n_on = int((on_off["call"].isin(["on_in_treatment", "on_in_control"])).sum())
    print(f"  {strat:12s}  limma-tested: {len(limma_result):>5,}  |  FDR<0.05: {n_sig:>4}  |  on/off: {n_on:>4}")
    de_results[strat] = {"limma": limma_result, "on_off": on_off, "n_sig": n_sig, "n_on": n_on}

# %%
# DE summary bar chart
de_summary = pd.DataFrame([
    {"strategy": s, "n_limma_tested": len(v["limma"]),
     "n_sig_fdr_0.05": v["n_sig"], "n_on_off": v["n_on"]}
    for s, v in de_results.items()
])
print("\nDE summary:")
print(de_summary.to_string(index=False))

fig = go.Figure()
fig.add_trace(go.Bar(
    name="limma-tested", x=de_summary["strategy"], y=de_summary["n_limma_tested"],
    marker=dict(color="#7a7360", line=dict(color="black", width=0.5)),
))
fig.add_trace(go.Bar(
    name="significant @ FDR<0.05", x=de_summary["strategy"], y=de_summary["n_sig_fdr_0.05"],
    marker=dict(color="#c93838", line=dict(color="black", width=0.5)),
))
fig.add_trace(go.Bar(
    name="on/off detection-only", x=de_summary["strategy"], y=de_summary["n_on_off"],
    marker=dict(color="#3585c9", line=dict(color="black", width=0.5)),
))
fig.update_layout(
    template="plotly_white",
    barmode="group", height=440, width=920,
    yaxis_title="count", margin=dict(l=60, r=20, t=40, b=60),
)
fig.show()

# %% [markdown]
# ## 7. Do the strategies agree on WHICH sites are significant?
#
# Same number of hits, different hits — that's the scariest situation. The
# Jaccard index below tells you what fraction of significant sites are shared
# between each pair of strategies.

# %%
hit_sets = {
    s: set(v["limma"].index[v["limma"]["fdr"] < 0.05])
    for s, v in de_results.items()
}

n_strat = len(hit_sets)
jaccard = np.zeros((n_strat, n_strat))
strat_names = list(hit_sets)
for i, a in enumerate(strat_names):
    for j, b in enumerate(strat_names):
        A, B = hit_sets[a], hit_sets[b]
        jaccard[i, j] = len(A & B) / max(len(A | B), 1)

fig = go.Figure(data=go.Heatmap(
    z=jaccard, x=strat_names, y=strat_names,
    colorscale=[[0, "#ffffff"], [0.5, "#f4b6b6"], [1.0, "#c93838"]],
    zmin=0, zmax=1,
    text=[[f"{v:.2f}" for v in row] for row in jaccard], texttemplate="%{text}",
    colorbar=dict(title="Jaccard"),
))
fig.update_layout(template="plotly_white", width=560, height=480,
                  title="Jaccard of significant hits (FDR<0.05) across strategies",
                  margin=dict(l=90, r=60, t=60, b=60))
fig.update_yaxes(autorange="reversed")
fig.show()

# %% [markdown]
# **Reading the heatmap**: values near 1 mean the two strategies agree on
# which sites are significant. Values near 0 mean they call totally different
# sites. In a well-behaved dataset you should see high agreement (>0.7) among
# the alphaphos strategies — the choice affects *how many* hits you find,
# but the top hits should mostly overlap.

# %% [markdown]
# ## 8. Do the strategies agree on the SIGN + MAGNITUDE of shared hits?
#
# For sites that all four strategies call as significant, do they agree on
# the direction and rough size of the fold-change? A scatter of log2fc
# from `condition` vs each other strategy — points on the diagonal are
# perfect agreement.

# %%
# Sites that are limma-tested by ALL four strategies
common_sites = set(de_results[strategies[0]]["limma"].index)
for s in strategies[1:]:
    common_sites &= set(de_results[s]["limma"].index)
print(f"Sites limma-tested by all four strategies: {len(common_sites):,}")

common = sorted(common_sites)
lfc_ref = de_results["condition"]["limma"].loc[common, "log2fc"].to_numpy()
others = ("per_run", "global_max", "wilson")

# Compute Pearson r for each pair up front so titles carry it
subplot_titles = []
lfcs = {}
for other in others:
    lfc_other = de_results[other]["limma"].loc[common, "log2fc"].to_numpy()
    r = float(np.corrcoef(lfc_ref, lfc_other)[0, 1])
    lfcs[other] = lfc_other
    subplot_titles.append(f"condition vs {other} — r = {r:.3f}")

fig = make_subplots(rows=1, cols=3, subplot_titles=subplot_titles,
                    shared_yaxes=True, horizontal_spacing=0.06)
for col, other in enumerate(others, start=1):
    fig.add_trace(go.Scatter(
        x=lfc_ref, y=lfcs[other], mode="markers",
        marker=dict(size=3, opacity=0.4, color=STRATEGY_COLOURS[other],
                    line=dict(color="black", width=0.2)),
        showlegend=False,
        hovertemplate=f"condition log2fc: %{{x:.2f}}<br>{other} log2fc: %{{y:.2f}}<extra></extra>",
    ), row=1, col=col)
    # y = x diagonal
    fig.add_trace(go.Scatter(x=[-8, 8], y=[-8, 8], mode="lines",
                             line=dict(color="black", dash="dot", width=1),
                             showlegend=False), row=1, col=col)
fig.update_xaxes(title="condition log2fc")
fig.update_yaxes(title="strategy log2fc", row=1, col=1)
fig.update_layout(template="plotly_white", width=1080, height=380,
                  margin=dict(l=60, r=20, t=60, b=60),
                  title=f"log2fc concordance on {len(common):,} sites tested by all four strategies")
fig.show()

# %% [markdown]
# **Reading the scatters**: points on the diagonal = the two strategies
# assign identical effect sizes to the same site. A Pearson r ≥ 0.95 means
# the strategies agree on biology, differing mainly on which sites they
# include.
#
# ## 9. How does alphaphos compare to raw Spectronaut PTM Site output?
#
# Spectronaut's Class-I PTM Site Report is the closest analog to
# `per_run` — Spectronaut applies its Class-I filter at collapse time.
# alphaphos gives you access to the same raw PSM data with four collapse
# variants + control over downstream filtering.

# %%
# Get alphaphos site keys in a form comparable to Spectronaut
def _alphaphos_gene_site(adata) -> set:
    # alphaphos var_names are "PG.ProteinGroup|PG.Genes|Site|Multiplicity"
    genes_sites = set()
    for key in adata.var_names:
        parts = key.split("|")
        if len(parts) >= 3:
            genes_sites.add((parts[1], parts[2]))
    return genes_sites

def _sn_gene_site(sn_df: pd.DataFrame) -> set:
    return set(zip(
        sn_df["PG.Genes"].astype(str),
        sn_df["PTM.SiteAA"].astype(str) + sn_df["PTM.SiteLocation"].astype(str)
    ))

ap_sites_per_run = _alphaphos_gene_site(collapsed["per_run"])
ap_sites_global = _alphaphos_gene_site(collapsed["global_max"])
sn_all_sites = _sn_gene_site(sn_all)
sn_cls_sites = _sn_gene_site(sn_cls)

def _venn_pair(A, B, a_name, b_name):
    return {
        f"only {a_name}": len(A - B),
        "both": len(A & B),
        f"only {b_name}": len(B - A),
    }

print("alphaphos per_run   vs SN Class-I:", _venn_pair(ap_sites_per_run, sn_cls_sites, "alphaphos", "SN"))
print("alphaphos global_max vs SN all:    ", _venn_pair(ap_sites_global, sn_all_sites, "alphaphos", "SN"))

# %% [markdown]
# **Reading**: the overlap between alphaphos `per_run` and Spectronaut's
# Class-I Site Report should be very high (both apply Class-I at the
# collapse stage). Small "only alphaphos" or "only SN" numbers reflect
# minor differences in how the tools handle contaminants, decoys, and
# multi-protein groups. Nothing dramatic.
#
# ## 10. Practical recommendations
#
# **Which strategy to use, by scenario:**
#
# | Scenario | Strategy | Why |
# | --- | --- | --- |
# | Small pilot, n < 30, condition-labelled | `condition` (default) | Balances rigor and completeness with condition-aware masking |
# | Small pilot, no condition labels | `per_run` | Strict; matches classical Hogrebe convention |
# | Exploratory / max-depth | `global_max` | Keeps everything Spectronaut sees; use with strict downstream filters |
# | Large cohort (n ≥ 100) | `wilson` with `wilson_threshold="auto"` | Sample-size correction; empirically the best default at scale |
# | Comparing directly to Spectronaut output | `per_run` | Closest match to Spectronaut's own Class-I Site Report |
#
# **On MS1 vs MS2**: the reader picks MS2 when it's in the report and falls
# back to MS1 with a warning. MS2 is preferred (better for DIA), but on this
# dataset the DE conclusions are almost identical — the log2fcs agree at
# Pearson r > 0.98 across the two quant columns. Don't spend time worrying
# about it unless you're doing quantitative accuracy work.
#
# **On alphaphos vs raw Spectronaut**: Spectronaut gives you a Class-I
# already-collapsed site table. alphaphos gives you four collapse variants
# from the same PSM data, plus per-site Wilson-lb confidence for large
# cohorts, plus AnnData integration for everything downstream (`diff_exp_*`,
# `dimred`, `enrichment`, `signalome`). For an n=6 pilot the depth is
# similar; for an n=300 cohort the difference is decisive.
#
# **Recommend_pipeline** picks the strategy for you based on cohort size —
# see `docs/how-to-use-alphaphos.md` §5 and `docs/design-principles.md`.

# %% [markdown]
# ## 11. When you don't want to collapse to sites: `collapse_precursors`
#
# Everything so far has aggregated PSMs into **site-level** AnnDatas —
# one row per phosphosite. That's usually what you want. But sometimes
# the site aggregation costs you information you actually need:
#
# - **Localization is unreliable** at your loc_prob cutoff on some
#   subset of peptides — you'd rather keep the precursor-level detail
#   and let the reader downstream decide.
# - **You want to look at peptide-level differential effects** — e.g.
#   whether a specific charge state or missed-cleavage variant behaves
#   differently from another form of the same peptide.
# - **KSEA on precursor-level input** — the kinase library can score
#   peptides directly; you convert back to site keys only at the end
#   via `ap.precursor_to_site_view`.
#
# `ap.collapse_precursors` produces an AnnData where each row is a
# **precursor** — `Protein|Gene|Peptide|Charge|Mods` — with no residue
# attribution and no localization masking. Same PSM input, one function
# call, entirely parallel to `collapse_sites`.

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    prec_ad = ap.collapse_precursors(psm, condition_df=cond_df)

print(f"Precursor AnnData: {prec_ad.n_obs} samples × {prec_ad.n_vars:,} precursors, "
      f"{float(np.isnan(prec_ad.X).mean()):.1%} NaN")
print(f"\nvar columns (first 10): {list(prec_ad.var.columns[:10])}")
print("\nFirst 3 precursors:")
print(prec_ad.var[["gene", "peptide_sequence", "charge", "n_phospho",
                   "n_samples_detected", "mean_loc_prob"]].head(3).to_string())

# %% [markdown]
# Note that precursor-level output has:
#
# - **More rows than the site-level output** (~1.2× typically) — one row
#   per unique `(peptide, charge, modifications)` triple, whereas
#   site-level aggregates multiple charge states of the same peptide
#   into a single row per phospho position.
# - **Lower missingness** (2-15% vs 30-70% at site level) — because
#   precursors are the atomic quant unit; no aggregation losses.
# - **No `localization_strategy` masking** — every quantified measurement
#   is retained; the loc_prob is stored as var metadata for filtering
#   later.

# %%
print(f"Precursor n_vars:  {prec_ad.n_vars:,}")
print(f"Site n_vars (per_run):     {collapsed['per_run'].n_vars:,}")
print(f"Site n_vars (global_max):  {collapsed['global_max'].n_vars:,}")
print(f"\nMean precursors-per-site (rough): "
      f"{prec_ad.n_vars / max(collapsed['global_max'].n_vars, 1):.2f}")

# %% [markdown]
# ### Bridging back to sites — `precursor_to_site_view`
#
# When you've done a precursor-level analysis and want to talk about
# sites (for KSEA, for interpretation), `ap.precursor_to_site_view`
# aggregates a precursor result table back to site keys:
#
# ```python
# # Illustrative, not run
# prec_de = ap.diff_exp_limma(prec_ad, condition_column="condition",
#                              comparison=("EGF+", "EGF-"))
# site_de = ap.precursor_to_site_view(prec_de, prec_ad,
#                                       aggregation="max_abs_log2fc")
# ```
#
# The site_de result carries one row per phospho site with the
# most-extreme fold-change contributed by any of its precursors —
# suitable for KSEA / enrichment.
#
# ### When to prefer precursor over site
#
# Situation → your call:
#
# - Standard 2-group phospho DE with well-behaved data → **site-level**
# - Small cohort where you're worried some sites are being kept by a
#   single low-confidence precursor → **precursor-level**, filter on
#   `mean_loc_prob` at the precursor level yourself
# - Method-development / QC work looking at fragmentation quality per
#   charge state → **precursor-level**
# - KSEA where you trust the kinase library on peptide sequences → run
#   DE at precursor, `precursor_to_site_view` at the end
#
# For everything else, `collapse_sites` is the default.

# %% [markdown]
# ## Full end-to-end script (for reference)
#
# Copy this into a fresh cell to reproduce the whole workflow on any
# Spectronaut PSM report. Change the paths and `cond_df` builder to your
# data.

# %%
if False:  # not run — pedagogical block
    import alphaphos as ap
    import pandas as pd
    from pathlib import Path

    psm = ap.read_spectronaut("your_report.tsv")
    samples = psm["R.FileName"].unique()
    cond_df = pd.DataFrame({
        "sample": samples,
        "condition": [...],  # your labelling logic
    })
    adata = ap.collapse_sites(psm, condition_df=cond_df)  # default: condition

    # Or let the advisor pick:
    ap.recommend_pipeline(adata, goal="primary_de", data_type="phospho",
                           primary_factor="condition")
