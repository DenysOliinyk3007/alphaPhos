"""Paired alphaPhos-vs-Spectronaut-native validation.

Generalized version of compare_to_spectronaut_native.py — takes the PSM path
and native PTM site report path as arguments so we can run it on additional
datasets (lungAC, DVPP shapes, etc.) without copy-pasting.

Default config: alphaPhos with top-N attribution fix + collapse_level=P +
consolidate + global_max + cutoff 0.75 (closest match to Spectronaut native).
"""

from __future__ import annotations

import argparse
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
from alphaphos.preprocess import filter_to_top_n_positions  # noqa: E402
from PeptideCollapse_v4 import PeptideCollapse  # noqa: E402

KEY = re.compile(r"^(?P<prot>[^~]+)~(?P<gene>[^_]*)_(?P<aa>[A-Z])(?P<pos>\d+)_M(?P<mult>\d+)$")


def load_native(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.loc[df["PTM.ModificationTitle"] == "Phospho (STY)"].copy()
    prob_cols = [c for c in df.columns if "PTM.SiteProbability" in c]
    quant_cols = [c for c in df.columns if "PTM.Quantity" in c]

    def _sample(c: str) -> str:
        m = re.match(r"\[\d+\]\s+(.+?)\.raw\.PTM\.\w+", c)
        return m.group(1) if m else c

    meta = ["PTM.ProteinId", "PTM.SiteAA", "PTM.SiteLocation", "PTM.Multiplicity"]
    prob = df.melt(id_vars=meta, value_vars=prob_cols,
                   var_name="sample_col", value_name="prob_raw")
    prob["sample"] = prob["sample_col"].map(_sample)
    quant = df.melt(id_vars=meta, value_vars=quant_cols,
                    var_name="sample_col", value_name="quant_raw")
    quant["sample"] = quant["sample_col"].map(_sample)
    out = prob.merge(quant[meta + ["sample", "quant_raw"]], on=meta + ["sample"])
    out["native_quant"] = pd.to_numeric(
        out["quant_raw"].replace("Filtered", np.nan), errors="coerce"
    )
    out["mult_capped"] = out["PTM.Multiplicity"].clip(upper=3).astype(int)
    out["match_key"] = list(zip(
        out["PTM.ProteinId"], out["PTM.SiteAA"],
        out["PTM.SiteLocation"].astype(int), out["mult_capped"],
    ))
    return out[["match_key", "sample", "native_quant",
                "mult_capped"]]


def alpha_to_long(sites: pd.DataFrame, sample_keyword: str) -> pd.DataFrame:
    parsed = sites["Protein_Collapse_key"].astype(str).str.extract(KEY)
    sites = sites.loc[parsed["prot"].notna()].copy()
    parsed = parsed.loc[parsed["prot"].notna()]
    sites["_prot"] = parsed["prot"].values
    sites["_aa"] = parsed["aa"].values
    sites["_pos"] = parsed["pos"].astype(int).values
    sites["_mult"] = parsed["mult"].astype(int).values
    sample_cols = [c for c in sites.columns
                   if isinstance(c, str) and sample_keyword in c]
    long = sites.melt(
        id_vars=["_prot", "_aa", "_pos", "_mult"],
        value_vars=sample_cols, var_name="sample", value_name="alpha_log2",
    )
    long["match_key"] = list(zip(long["_prot"], long["_aa"], long["_pos"], long["_mult"]))
    return long[["match_key", "sample", "alpha_log2"]], sample_cols


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--psm", required=True, type=Path, help="PSM-level PSM TSV/parquet")
    ap.add_argument("--native", required=True, type=Path,
                    help="Native PTM site report TSV")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "validate_native_out")
    ap.add_argument("--sample-keyword", default="nanoPhos",
                    help="Substring to identify sample columns in alphaPhos output")
    ap.add_argument("--quant-level", default="auto", choices=["auto", "MS1", "MS2"])
    ns = ap.parse_args()

    ns.out.mkdir(parents=True, exist_ok=True)
    label = ns.psm.stem

    print(f"=== Paired validation: {label} ===")
    print(f"  PSM:    {ns.psm}")
    print(f"  native: {ns.native}\n")

    print("[1/3] Loading native...")
    t = time.time()
    native = load_native(ns.native)
    print(f"  {len(native):,} long rows ({time.time()-t:.1f}s)")
    print(f"  unique native sites: {native['match_key'].nunique():,}")

    print("\n[2/3] Loading PSM + applying top-N filter + alphaPhos pipeline...")
    t = time.time()
    df = read_spectronaut(ns.psm, quant_level=ns.quant_level,
                          drop_decoys=True, pg_qvalue_max=0.01)
    print(f"  PSM rows: {len(df):,} ({time.time()-t:.1f}s)")
    df = filter_to_top_n_positions(df)
    print(f"  after top-N: {len(df):,} rows")
    t = time.time()
    pc = PeptideCollapse(verbose=False)
    sites = pc.process_complete_pipeline(
        df, cutoff=0.75, collapse_level="P",
        aggregation_method="consolidate",
        add_kinase_sequences=False, noise_floor_filter=True,
        localization_strategy="global_max",
    )
    print(f"  {len(sites)} alphaPhos sites ({time.time()-t:.1f}s)")

    alpha, sample_cols = alpha_to_long(sites, ns.sample_keyword)
    print(f"  alphaPhos sample columns: {len(sample_cols)}")

    print("\n[3/3] Cross-comparison...")
    merged = native.merge(alpha, on=["match_key", "sample"], how="inner")
    both = merged.dropna(subset=["native_quant", "alpha_log2"]).copy()
    both["native_log2"] = np.log2(both["native_quant"].replace(0, np.nan))
    both = both.dropna(subset=["native_log2"])
    both["log2_diff"] = both["alpha_log2"] - both["native_log2"]
    diff = both["log2_diff"]

    # Overlap
    native_keys = set(native["match_key"].unique())
    alpha_keys = set(alpha["match_key"].unique())
    overlap_sites = native_keys & alpha_keys
    overlap = {
        "n_native_sites": len(native_keys),
        "n_alphaphos_sites": len(alpha_keys),
        "intersection_sites": len(overlap_sites),
        "native_only": len(native_keys - alpha_keys),
        "alphaphos_only": len(alpha_keys - native_keys),
        "jaccard": round(len(overlap_sites) / max(1, len(native_keys | alpha_keys)), 4),
    }

    cell_stats = {
        "n_paired_cells_with_quants": int(len(both)),
        "pearson_log2": round(float(both["alpha_log2"].corr(both["native_log2"])), 4),
        "spearman_log2": round(float(both["alpha_log2"].corr(both["native_log2"],
                                                              method="spearman")), 4),
        "pearson_linear": round(float(np.power(2, both["alpha_log2"]).corr(
            np.power(2, both["native_log2"]))), 4),
        "mean_log2_diff": round(float(diff.mean()), 4),
        "median_log2_diff": round(float(diff.median()), 4),
        "sd_log2_diff": round(float(diff.std()), 4),
        "p05_log2_diff": round(float(diff.quantile(0.05)), 4),
        "p95_log2_diff": round(float(diff.quantile(0.95)), 4),
        "pct_within_0.1_log2": round(float((diff.abs() < 0.1).mean() * 100), 2),
        "pct_within_0.5_log2": round(float((diff.abs() < 0.5).mean() * 100), 2),
        "pct_within_1.0_log2": round(float((diff.abs() < 1.0).mean() * 100), 2),
        "pct_alpha_higher_by_1_log2": round(float((diff > 1.0).mean() * 100), 2),
    }

    summary = {"label": label, "overlap": overlap, "cells": cell_stats}
    print("\n=== Overlap ===")
    for k, v in overlap.items():
        print(f"  {k:25s}: {v}")
    print("\n=== Cell-level agreement ===")
    for k, v in cell_stats.items():
        print(f"  {k:30s}: {v}")

    (ns.out / f"summary_{label}.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nWrote {ns.out / ('summary_' + label + '.json')}")


if __name__ == "__main__":
    main()
