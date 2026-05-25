"""Investigate the right-tail disagreement between alphaPhos and Spectronaut native.

Baseline finding (compare_to_spectronaut_native.py):
    consolidate + global_max + cutoff 0.75 →
    median log2 diff = +0.007 (effectively zero),
    mean = +0.36 (long right tail),
    p95 = +2.33 (alphaPhos up to 5x higher).

Three hypotheses tested in parallel here:

  H1 (strategy)    — global_max keeps cells Spectronaut filters per-run.
                     Test by switching to per_run; if right tail collapses,
                     Spectronaut effectively filters per-cell at consolidate time.
  H2 (aggregation) — consolidate's ratio imputation adds signal sum doesn't.
                     Test by using sum aggregation; if right tail collapses,
                     consolidate is over-imputing missing precursor values.
  H3 (noise floor) — our noise_floor_filter is a no-op (we already drop
                     log2={0,1}), but worth sanity-checking; pre-printed for
                     completeness.

Plus stratification of the diff distribution by:
  * native PTM.SiteProbability quartile  (loc quality)
  * PTM.Multiplicity (1, 2, 3+)
  * native PTM.Quantity log10 bucket    (signal magnitude)
  * native vs alphaPhos cell-count overlap per site
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "d:/Projects/Dublin/testscripts/src")

from alphaphos.io import read_spectronaut  # noqa: E402
from PeptideCollapse_v4 import PeptideCollapse  # noqa: E402

OUT_DIR = Path(__file__).parent / "investigate_right_tail_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET = (
    "d:/Projects/alphaPhos/test_data/"
    "nanoPhos_dilser_noEGF_1000ng_Report_ms1_ms2.parquet"
)
NATIVE_REPORT = (
    "Z:/Denys_nanoPhos/PRIDE/analysis_data/revision/figure2/"
    "20260506_113358_nanoPhos_dilser_noEGF_1000ng_Report.tsv"
)

CUTOFF = 0.75

KEY_PATTERN = re.compile(
    r"^(?P<prot>[^~]+)~(?P<gene>[^_]*)_(?P<aa>[A-Z])(?P<pos>\d+)_M(?P<mult>\d+)$"
)


def load_native_long(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.loc[df["PTM.ModificationTitle"] == "Phospho (STY)"].copy()
    prob_cols = [c for c in df.columns if "PTM.SiteProbability" in c]
    quant_cols = [c for c in df.columns if "PTM.Quantity" in c]

    def _sample_of(c: str) -> str:
        m = re.match(r"\[\d+\]\s+(.+?)\.raw\.PTM\.\w+", c)
        return m.group(1) if m else c

    prob = df.melt(
        id_vars=["PTM.ProteinId", "PTM.SiteAA", "PTM.SiteLocation", "PTM.Multiplicity"],
        value_vars=prob_cols, var_name="sample_col", value_name="prob_raw",
    )
    prob["sample"] = prob["sample_col"].map(_sample_of)
    quant = df.melt(
        id_vars=["PTM.ProteinId", "PTM.SiteAA", "PTM.SiteLocation", "PTM.Multiplicity"],
        value_vars=quant_cols, var_name="sample_col", value_name="quant_raw",
    )
    quant["sample"] = quant["sample_col"].map(_sample_of)
    out = prob.merge(
        quant[["PTM.ProteinId", "PTM.SiteAA", "PTM.SiteLocation", "PTM.Multiplicity",
               "sample", "quant_raw"]],
        on=["PTM.ProteinId", "PTM.SiteAA", "PTM.SiteLocation",
            "PTM.Multiplicity", "sample"],
    )
    out["native_loc"] = pd.to_numeric(
        out["prob_raw"].replace("Filtered", np.nan), errors="coerce"
    )
    out["native_quant"] = pd.to_numeric(
        out["quant_raw"].replace("Filtered", np.nan), errors="coerce"
    )
    out["mult_capped"] = out["PTM.Multiplicity"].clip(upper=3).astype(int)
    out["match_key"] = list(zip(
        out["PTM.ProteinId"], out["PTM.SiteAA"],
        out["PTM.SiteLocation"].astype(int), out["mult_capped"],
    ))
    return out[["match_key", "sample", "native_loc", "native_quant",
                "PTM.Multiplicity", "mult_capped"]]


def run_config(
    df_raw: pd.DataFrame, aggregation: str, strategy: str, noise_floor: bool,
) -> tuple[pd.DataFrame, dict]:
    pc = PeptideCollapse(verbose=False)
    t0 = time.time()
    sites = pc.process_complete_pipeline(
        df_raw,
        cutoff=CUTOFF,
        collapse_level="P",
        aggregation_method=aggregation,
        add_kinase_sequences=False,
        noise_floor_filter=noise_floor,
        localization_strategy=strategy,
    )
    elapsed = time.time() - t0
    return sites, {"elapsed_s": round(elapsed, 1), **dict(pc.processing_stats)}


def alpha_to_long(sites: pd.DataFrame, sample_cols: list[str]) -> pd.DataFrame:
    parsed = sites["Protein_Collapse_key"].astype(str).str.extract(KEY_PATTERN)
    bad = parsed["prot"].isna()
    sites = sites.loc[~bad].copy()
    parsed = parsed.loc[~bad]
    sites["_prot"] = parsed["prot"].values
    sites["_aa"] = parsed["aa"].values
    sites["_pos"] = parsed["pos"].astype(int).values
    sites["_mult"] = parsed["mult"].astype(int).values
    long = sites.melt(
        id_vars=["_prot", "_aa", "_pos", "_mult"],
        value_vars=sample_cols, var_name="sample", value_name="alpha_log2",
    )
    long["match_key"] = list(zip(long["_prot"], long["_aa"], long["_pos"], long["_mult"]))
    return long[["match_key", "sample", "alpha_log2"]]


def pair_and_diff(native: pd.DataFrame, alpha: pd.DataFrame) -> pd.DataFrame:
    merged = native.merge(alpha, on=["match_key", "sample"], how="inner")
    both = merged.dropna(subset=["native_quant", "alpha_log2"]).copy()
    both["native_log2"] = np.log2(both["native_quant"].replace(0, np.nan))
    both = both.dropna(subset=["native_log2"])
    both["log2_diff"] = both["alpha_log2"] - both["native_log2"]
    both["abs_diff"] = both["log2_diff"].abs()
    return both


def summarize_diff(diff: pd.Series, label: str) -> dict:
    return {
        "config": label,
        "n_cells": int(len(diff)),
        "mean": round(float(diff.mean()), 4),
        "median": round(float(diff.median()), 4),
        "sd": round(float(diff.std()), 4),
        "p05": round(float(diff.quantile(0.05)), 4),
        "p25": round(float(diff.quantile(0.25)), 4),
        "p75": round(float(diff.quantile(0.75)), 4),
        "p95": round(float(diff.quantile(0.95)), 4),
        "pct_within_0.1": round(float((diff.abs() < 0.1).mean() * 100), 2),
        "pct_within_0.5": round(float((diff.abs() < 0.5).mean() * 100), 2),
        "pct_within_1.0": round(float((diff.abs() < 1.0).mean() * 100), 2),
        "pct_alpha_higher_by_1_log2": round(float((diff > 1.0).mean() * 100), 2),
    }


def stratify(both: pd.DataFrame, by: str, bins, labels=None) -> pd.DataFrame:
    grp = pd.cut(both[by], bins=bins, labels=labels, include_lowest=True)
    out = both.groupby(grp, observed=True).agg(
        n=("log2_diff", "size"),
        mean=("log2_diff", "mean"),
        median=("log2_diff", "median"),
        p95=("log2_diff", lambda s: s.quantile(0.95)),
        pct_within_0_1=("log2_diff", lambda s: (s.abs() < 0.1).mean() * 100),
        pct_alpha_higher_1_log2=("log2_diff", lambda s: (s > 1.0).mean() * 100),
    )
    return out.round(3)


def main() -> None:
    print("=== Loading Spectronaut native ===")
    native = load_native_long(NATIVE_REPORT)
    n_native_quants = int(native["native_quant"].notna().sum())
    print(f"  native long rows: {len(native):,}, non-NaN quants: {n_native_quants:,}")

    print("\n=== Loading alphaPhos input (MS2) ===")
    df_raw = read_spectronaut(PARQUET, quant_level="MS2", drop_decoys=True, pg_qvalue_max=0.01)
    print(f"  {len(df_raw):,} rows")
    sample_cols = [
        c for c in df_raw["R.FileName"].drop_duplicates().tolist()
    ]
    # Sites use R.FileName as column names in their wide layout
    print(f"  samples: {sample_cols}")

    # === Run 3 configs ===
    configs = [
        ("consolidate / global_max", "consolidate", "global_max", True),
        ("consolidate / per_run",    "consolidate", "per_run",    True),
        ("sum / global_max",         "sum",         "global_max", True),
    ]
    all_runs = {}
    for label, agg, strat, nf in configs:
        print(f"\n--- {label} ---")
        sites, stats = run_config(df_raw, agg, strat, nf)
        sc = [c for c in sample_cols if c in sites.columns]
        if not sc:
            sc = [c for c in sites.columns if "nanoPhos_dilser" in str(c)]
        long = alpha_to_long(sites, sc)
        both = pair_and_diff(native, long)
        all_runs[label] = {
            "sites": sites, "long": long, "both": both, "stats": stats,
            "n_alpha_sites": int(long["match_key"].nunique()),
            "n_paired_cells_with_quants": int(len(both)),
        }
        print(f"  alphaPhos sites: {all_runs[label]['n_alpha_sites']:,}, "
              f"paired non-NaN cells: {all_runs[label]['n_paired_cells_with_quants']:,}, "
              f"runtime: {stats['elapsed_s']}s")

    # === Top-level diff summary per config ===
    print("\n\n========== Per-config diff summary (log2 space) ==========")
    summaries = [summarize_diff(r["both"]["log2_diff"], label)
                 for label, r in all_runs.items()]
    summary_df = pd.DataFrame(summaries)
    print(summary_df.to_string(index=False))

    # === Stratification — focus on the baseline (consolidate + global_max) ===
    baseline_label = "consolidate / global_max"
    both = all_runs[baseline_label]["both"]
    print(f"\n========== Stratification of '{baseline_label}' ==========")

    print("\n--- by native_loc (PTM.SiteProbability) quartile ---")
    loc_strat = stratify(
        both, "native_loc",
        bins=[0.75, 0.85, 0.95, 0.99, 1.001],
        labels=["0.75-0.85", "0.85-0.95", "0.95-0.99", "0.99-1.00"],
    )
    print(loc_strat)

    print("\n--- by PTM.Multiplicity (raw, before our cap) ---")
    mult_strat = both.groupby("PTM.Multiplicity").agg(
        n=("log2_diff", "size"),
        mean=("log2_diff", "mean"),
        median=("log2_diff", "median"),
        p95=("log2_diff", lambda s: s.quantile(0.95)),
        pct_within_0_1=("log2_diff", lambda s: (s.abs() < 0.1).mean() * 100),
        pct_alpha_higher_1_log2=("log2_diff", lambda s: (s > 1.0).mean() * 100),
    ).round(3)
    print(mult_strat)

    print("\n--- by native quant magnitude (log10) ---")
    both["native_log10"] = np.log10(both["native_quant"].replace(0, np.nan))
    quant_strat = stratify(
        both, "native_log10",
        bins=[-np.inf, 1, 2, 3, 4, np.inf],
        labels=["<10", "10-100", "100-1k", "1k-10k", ">10k"],
    )
    print(quant_strat)

    # === Per-site disagreement count ===
    print("\n--- top 20 most-disagreeing sites by max abs diff ---")
    per_site = (
        both.groupby("match_key")
        .agg(n_cells=("log2_diff", "size"),
             mean_diff=("log2_diff", "mean"),
             max_abs_diff=("abs_diff", "max"),
             max_alpha=("alpha_log2", "max"),
             max_native_log2=("native_log2", "max"))
        .sort_values("max_abs_diff", ascending=False)
        .head(20)
    )
    print(per_site.round(3).to_string())

    # === Was the right tail driven by alphaPhos including cells where native is "Filtered"? ===
    # The pair_and_diff above only keeps cells with non-NaN native_quant, so
    # this comparison ALREADY excludes the "Filtered" cells. But let's also
    # look at alphaPhos cells *paired with NaN native* to count what got
    # excluded.
    print("\n--- alphaPhos cells WITHOUT a non-NaN native counterpart (baseline) ---")
    long = all_runs[baseline_label]["long"]
    merged_full = native.merge(long, on=["match_key", "sample"], how="outer", indicator=True)
    alpha_orphans = merged_full.loc[
        (merged_full["_merge"] == "right_only") | (
            (merged_full["_merge"] == "both") &
            merged_full["native_quant"].isna() &
            merged_full["alpha_log2"].notna()
        )
    ]
    n_filtered_native = int((
        (merged_full["_merge"] == "both")
        & merged_full["native_quant"].isna()
        & merged_full["alpha_log2"].notna()
    ).sum())
    print(f"  alphaPhos cells with native NaN (Filtered or absent): {len(alpha_orphans):,}")
    print(f"  ... where native quant was 'Filtered' (paired site, masked cell): {n_filtered_native:,}")

    # === Save artifacts ===
    summary_df.to_csv(OUT_DIR / "config_summary.csv", index=False)
    loc_strat.to_csv(OUT_DIR / "stratify_by_loc.csv")
    mult_strat.to_csv(OUT_DIR / "stratify_by_mult.csv")
    quant_strat.to_csv(OUT_DIR / "stratify_by_quant.csv")
    per_site.to_csv(OUT_DIR / "top20_disagreeing_sites.csv")

    (OUT_DIR / "summary.json").write_text(
        json.dumps({
            "configs": [
                {
                    "label": label,
                    "alphaphos_sites": r["n_alpha_sites"],
                    "paired_cells": r["n_paired_cells_with_quants"],
                    "diff_summary": summaries[i],
                }
                for i, (label, r) in enumerate(all_runs.items())
            ],
        }, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nArtifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
