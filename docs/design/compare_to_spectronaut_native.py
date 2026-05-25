"""Validate alphaPhos collapse output against Spectronaut's native PTM site report.

We run our pipeline on the SAME raw Spectronaut export that produced the
native PTM site report (`collapse_level='P'`, `consolidate`, `global_max`,
cutoff=0.75 — the closest match to Spectronaut's defaults), then match
sites by (ProteinId, SiteAA, SiteLocation, Multiplicity_capped) and
compare per-cell quants in log2 space.

This is the canonical *validation* — if our pipeline reproduces Spectronaut's
own consolidate (which IS the Hogrebe R script embedded in the .NET plugin,
per the decompile finding) then the Python port is verified end-to-end.
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

OUT_DIR = Path(__file__).parent / "compare_native_output"
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


def load_spectronaut_native(path: str) -> pd.DataFrame:
    """Load Spectronaut PTM site report, filter to phospho, coerce numerics.

    Returns a long-form DataFrame:
        match_key : (protein_id, site_aa, site_location, multiplicity_capped)
        sample    : run filename (without [n] prefix and .raw suffix)
        site_prob : float (NaN if 'Filtered')
        quantity  : float (NaN if 'Filtered')
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.loc[df["PTM.ModificationTitle"] == "Phospho (STY)"].copy()

    # Identify per-sample columns
    prob_cols = [c for c in df.columns if "PTM.SiteProbability" in c]
    quant_cols = [c for c in df.columns if "PTM.Quantity" in c]

    def _sample_from_col(c: str) -> str:
        # '[1] 20250721_OA4_..._01.raw.PTM.Quantity' -> '20250721_OA4_..._01'
        m = re.match(r"\[\d+\]\s+(.+?)\.raw\.PTM\.\w+", c)
        return m.group(1) if m else c

    # Wide -> long, separately for probability and quantity, then merge
    meta_cols = [
        "PTM.ProteinId", "PG.Genes", "PTM.SiteAA", "PTM.SiteLocation",
        "PTM.Multiplicity",
    ]
    prob_long = (
        df[meta_cols + prob_cols]
        .melt(id_vars=meta_cols, value_vars=prob_cols,
              var_name="sample_col", value_name="site_prob_raw")
    )
    prob_long["sample"] = prob_long["sample_col"].map(_sample_from_col)

    quant_long = (
        df[meta_cols + quant_cols]
        .melt(id_vars=meta_cols, value_vars=quant_cols,
              var_name="sample_col", value_name="quantity_raw")
    )
    quant_long["sample"] = quant_long["sample_col"].map(_sample_from_col)

    long = prob_long.merge(
        quant_long[meta_cols + ["sample", "quantity_raw"]],
        on=meta_cols + ["sample"], how="outer",
    )

    long["site_prob"] = pd.to_numeric(
        long["site_prob_raw"].replace("Filtered", np.nan), errors="coerce"
    )
    long["quantity"] = pd.to_numeric(
        long["quantity_raw"].replace("Filtered", np.nan), errors="coerce"
    )
    long["multiplicity_capped"] = long["PTM.Multiplicity"].clip(upper=3).astype(int)

    long["match_key"] = list(zip(
        long["PTM.ProteinId"],
        long["PTM.SiteAA"],
        long["PTM.SiteLocation"].astype(int),
        long["multiplicity_capped"],
    ))

    return long[
        ["match_key", "PTM.ProteinId", "PTM.SiteAA", "PTM.SiteLocation",
         "PTM.Multiplicity", "multiplicity_capped",
         "sample", "site_prob", "quantity"]
    ]


def run_alphaphos_collapse(path: str) -> tuple[pd.DataFrame, dict]:
    """Run PeptideCollapse_v4 at protein-resolved level to match Spectronaut layout."""
    print(f"  Reading {Path(path).name} (MS2)")
    df_raw = read_spectronaut(path, quant_level="MS2", drop_decoys=True, pg_qvalue_max=0.01)
    print(f"    -> {len(df_raw):,} rows")

    print("  Running PeptideCollapse_v4 (collapse_level=P, consolidate, global_max, cutoff=0.75)")
    pc = PeptideCollapse(verbose=False)
    t0 = time.time()
    sites = pc.process_complete_pipeline(
        df_raw,
        cutoff=CUTOFF,
        collapse_level="P",
        aggregation_method="consolidate",
        add_kinase_sequences=False,
        noise_floor_filter=True,
        localization_strategy="global_max",
    )
    print(f"    -> {len(sites)} sites in {time.time()-t0:.1f}s")
    return sites, dict(pc.processing_stats)


def alphaphos_to_long(sites: pd.DataFrame, sample_cols: list[str]) -> pd.DataFrame:
    """Convert alphaPhos site matrix to long form keyed by Spectronaut's identity tuple.

    At collapse_level='P', `Protein_Collapse_key` exists per protein. We parse it
    into (protein_id, site_aa, site_position, multiplicity).
    """
    if "Protein_Collapse_key" not in sites.columns:
        raise ValueError(
            f"Expected 'Protein_Collapse_key' from collapse_level='P'. "
            f"Got: {sorted(sites.columns)}"
        )

    key_pattern = re.compile(r"^(?P<prot>[^~]+)~(?P<gene>[^_]*)_(?P<aa>[A-Z])(?P<pos>\d+)_M(?P<mult>\d+)$")

    parsed = sites["Protein_Collapse_key"].astype(str).str.extract(key_pattern)
    bad = parsed["prot"].isna()
    if bad.any():
        print(f"  WARNING: {int(bad.sum())} Protein_Collapse_keys did not parse; dropping")
        sites = sites.loc[~bad].copy()
        parsed = parsed.loc[~bad]

    sites = sites.copy()
    sites["_prot"] = parsed["prot"].values
    sites["_aa"] = parsed["aa"].values
    sites["_pos"] = parsed["pos"].astype(int).values
    sites["_mult"] = parsed["mult"].astype(int).values

    long = sites.melt(
        id_vars=["_prot", "_aa", "_pos", "_mult", "PTM_localization"],
        value_vars=sample_cols,
        var_name="sample",
        value_name="log2_intensity",
    )
    long["match_key"] = list(zip(long["_prot"], long["_aa"], long["_pos"], long["_mult"]))
    return long.rename(columns={"PTM_localization": "alphaphos_loc"})


def main() -> None:
    print(f"Output dir: {OUT_DIR}")

    print("\n=== Loading Spectronaut native PTM site report ===")
    native = load_spectronaut_native(NATIVE_REPORT)
    print(f"  Phospho long rows: {len(native):,}")
    n_native_sites = native["match_key"].nunique()
    print(f"  Unique Spectronaut sites (after phospho filter + cap to M<=3): {n_native_sites:,}")
    n_native_cells = native["quantity"].notna().sum()
    print(f"  Non-NaN quant cells (Spectronaut): {n_native_cells:,}")

    print("\n=== Running alphaPhos pipeline on the same Spectronaut search ===")
    sites, stats = run_alphaphos_collapse(PARQUET)

    # Identify alphaPhos sample columns (the long filenames)
    sample_cols = [
        c for c in sites.columns
        if isinstance(c, str) and "nanoPhos_dilser" in c
    ]
    print(f"  alphaPhos sample columns: {sample_cols}")

    alpha = alphaphos_to_long(sites, sample_cols)
    n_alpha_sites = alpha["match_key"].nunique()
    print(f"  Unique alphaPhos sites (P-level, capped to M<=3 by pipeline): {n_alpha_sites:,}")

    # Sanity: do sample names align?
    native_samples = sorted(native["sample"].unique())
    alpha_samples = sorted(alpha["sample"].unique())
    print(f"  Spectronaut samples ({len(native_samples)}): {native_samples}")
    print(f"  alphaPhos samples ({len(alpha_samples)}): {alpha_samples}")
    if native_samples != alpha_samples:
        print("  WARNING: sample names differ; aligning may produce gaps")

    # === Site-set overlap ===
    native_keys = set(native["match_key"].unique())
    alpha_keys = set(alpha["match_key"].unique())
    intersection = native_keys & alpha_keys
    overlap_summary = {
        "native_total_sites": len(native_keys),
        "alphaphos_total_sites": len(alpha_keys),
        "intersection": len(intersection),
        "native_only": len(native_keys - alpha_keys),
        "alphaphos_only": len(alpha_keys - native_keys),
        "jaccard": round(len(intersection) / max(1, len(native_keys | alpha_keys)), 4),
    }
    print()
    print("=== Site-set overlap ===")
    for k, v in overlap_summary.items():
        print(f"  {k:25s}: {v:,}" if isinstance(v, int) else f"  {k:25s}: {v}")

    # === Per-cell paired comparison ===
    print("\n=== Per-cell quant agreement (matched by match_key + sample) ===")
    merged = native.merge(alpha, on=["match_key", "sample"], how="inner",
                          suffixes=("_native", "_alpha"))
    print(f"  paired cells (any value): {len(merged):,}")

    both = merged.dropna(subset=["quantity", "log2_intensity"]).copy()
    both["log2_native"] = np.log2(both["quantity"].replace(0, np.nan))
    both = both.dropna(subset=["log2_native"])
    print(f"  paired cells with non-NaN quants in BOTH: {len(both):,}")

    if len(both) < 10:
        print("  Too few paired cells for stats; aborting comparison.")
        return

    diff = both["log2_intensity"] - both["log2_native"]
    pearson_log2 = float(both["log2_intensity"].corr(both["log2_native"]))
    spearman_log2 = float(both["log2_intensity"].corr(both["log2_native"], method="spearman"))
    abs_diff_log2 = diff.abs()

    cell_stats = {
        "n_paired_cells": int(len(both)),
        "pearson_log2": round(pearson_log2, 4),
        "spearman_log2": round(spearman_log2, 4),
        "mean_log2_diff": round(float(diff.mean()), 4),
        "median_log2_diff": round(float(diff.median()), 4),
        "sd_log2_diff": round(float(diff.std()), 4),
        "p05_log2_diff": round(float(diff.quantile(0.05)), 4),
        "p95_log2_diff": round(float(diff.quantile(0.95)), 4),
        "pct_within_0.1_log2": round(float((abs_diff_log2 < 0.1).mean() * 100), 2),
        "pct_within_0.5_log2": round(float((abs_diff_log2 < 0.5).mean() * 100), 2),
        "pct_within_1.0_log2": round(float((abs_diff_log2 < 1.0).mean() * 100), 2),
    }
    print("  Cell-level stats:")
    for k, v in cell_stats.items():
        print(f"    {k:25s}: {v}")

    # === Localization agreement (sanity: should be near-identical, both use Spectronaut MS2 loc) ===
    print("\n=== Localization probability agreement ===")
    loc_pairs = merged.dropna(subset=["site_prob", "alphaphos_loc"]).copy()
    if len(loc_pairs):
        loc_diff = (loc_pairs["site_prob"] - loc_pairs["alphaphos_loc"]).abs()
        loc_stats = {
            "n_paired_with_both_loc": int(len(loc_pairs)),
            "fraction_exact_4dp": round(float((loc_diff < 1e-4).mean()), 4),
            "fraction_within_0.01": round(float((loc_diff < 0.01).mean()), 4),
            "fraction_within_0.05": round(float((loc_diff < 0.05).mean()), 4),
            "max_abs_diff": round(float(loc_diff.max()), 4),
            "median_abs_diff": round(float(loc_diff.median()), 6),
        }
        for k, v in loc_stats.items():
            print(f"  {k:25s}: {v}")
    else:
        loc_stats = {"n_paired_with_both_loc": 0}

    # === Per-site relative error (linear space) ===
    print("\n=== Per-site median relative error (linear space) ===")
    both["lin_alpha"] = np.power(2.0, both["log2_intensity"])
    both["lin_native"] = both["quantity"]
    both["rel_err_pct"] = (
        (both["lin_alpha"] - both["lin_native"]).abs() / both["lin_native"].abs() * 100
    )
    per_site = both.groupby("match_key").agg(
        n_cells=("rel_err_pct", "size"),
        median_rel_err_pct=("rel_err_pct", "median"),
    )
    print(f"  sites with >=1 paired cell: {len(per_site):,}")
    print(f"  median of per-site median rel_err: {per_site['median_rel_err_pct'].median():.2f}%")
    print(f"  p25 / p75 of per-site median rel_err: "
          f"{per_site['median_rel_err_pct'].quantile(0.25):.2f}% / "
          f"{per_site['median_rel_err_pct'].quantile(0.75):.2f}%")

    # === Save artifacts ===
    summary = {
        "overlap": overlap_summary,
        "cells": cell_stats,
        "localization": loc_stats,
        "alphaphos_pipeline_stats": stats,
        "config": {
            "parquet": PARQUET,
            "native_report": NATIVE_REPORT,
            "collapse_level": "P",
            "aggregation": "consolidate",
            "strategy": "global_max",
            "cutoff": CUTOFF,
        },
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    both[["match_key", "sample", "log2_native", "log2_intensity"]].to_csv(
        OUT_DIR / "paired_cells.csv", index=False
    )
    print(f"\nArtifacts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
