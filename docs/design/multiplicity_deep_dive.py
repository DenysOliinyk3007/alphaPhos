"""M2/M3/M4/M5 multiplicity-stratified validation vs Spectronaut native.

The earlier native comparison (compare_to_spectronaut_native.py +
prototype_top_n_attribution.py) found ~3% residual disagreement between
alphaPhos (with top-N fix) and Spectronaut's native PTM site report.

Hypothesis: this residual comes mainly from multiplicity capping —
alphaPhos clips multiplicity to M3, lumping M3+M4+M5 sites together,
while Spectronaut reports them as separate site objects.

Here we stratify the agreement specifically by `PTM.Multiplicity` and
explicitly count how many cells are in each multiplicity class.
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
from alphaphos.preprocess import filter_to_top_n_positions  # noqa: E402
from PeptideCollapse_v4 import PeptideCollapse  # noqa: E402

OUT_DIR = Path(__file__).parent / "multiplicity_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET = "d:/Projects/alphaPhos/test_data/nanoPhos_dilser_noEGF_1000ng_Report_ms1_ms2.parquet"
NATIVE = "Z:/Denys_nanoPhos/PRIDE/analysis_data/revision/figure2/20260506_113358_nanoPhos_dilser_noEGF_1000ng_Report.tsv"

KEY = re.compile(r"^(?P<prot>[^~]+)~(?P<gene>[^_]*)_(?P<aa>[A-Z])(?P<pos>\d+)_M(?P<mult>\d+)$")


def load_native_long(path: str) -> pd.DataFrame:
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
    out["native_quant"] = pd.to_numeric(out["quant_raw"].replace("Filtered", np.nan),
                                         errors="coerce")
    out["mult_raw"] = out["PTM.Multiplicity"].astype(int)
    out["mult_capped"] = out["mult_raw"].clip(upper=3)
    out["match_key_capped"] = list(zip(
        out["PTM.ProteinId"], out["PTM.SiteAA"],
        out["PTM.SiteLocation"].astype(int), out["mult_capped"],
    ))
    return out[["match_key_capped", "mult_raw", "mult_capped",
                "sample", "native_quant"]]


def alpha_to_long(sites: pd.DataFrame, sample_cols: list[str]) -> pd.DataFrame:
    parsed = sites["Protein_Collapse_key"].astype(str).str.extract(KEY)
    sites = sites.loc[parsed["prot"].notna()].copy()
    parsed = parsed.loc[parsed["prot"].notna()]
    sites["_prot"] = parsed["prot"].values
    sites["_aa"] = parsed["aa"].values
    sites["_pos"] = parsed["pos"].astype(int).values
    sites["_mult"] = parsed["mult"].astype(int).values
    long = sites.melt(id_vars=["_prot", "_aa", "_pos", "_mult"],
                      value_vars=sample_cols, var_name="sample",
                      value_name="alpha_log2")
    long["match_key_capped"] = list(zip(long["_prot"], long["_aa"], long["_pos"], long["_mult"]))
    long["alpha_mult"] = long["_mult"]
    return long[["match_key_capped", "alpha_mult", "sample", "alpha_log2"]]


def main() -> None:
    print("=== Loading native (with raw multiplicity preserved) ===")
    native = load_native_long(NATIVE)
    print(f"  long rows: {len(native):,}")
    print(f"  unique native sites (capped): {native['match_key_capped'].nunique():,}")
    print(f"  multiplicity distribution (raw):")
    print(native.groupby("mult_raw")["match_key_capped"].nunique().rename("n_sites"))

    print("\n=== Running alphaPhos with top-N fix ===")
    df = read_spectronaut(PARQUET, quant_level="MS2", drop_decoys=True, pg_qvalue_max=0.01)
    df = filter_to_top_n_positions(df)
    pc = PeptideCollapse(verbose=False)
    t = time.time()
    sites = pc.process_complete_pipeline(
        df, cutoff=0.75, collapse_level="P",
        aggregation_method="consolidate",
        add_kinase_sequences=False, noise_floor_filter=True,
        localization_strategy="global_max",
    )
    print(f"  {len(sites)} sites in {time.time()-t:.1f}s")

    sample_cols = [c for c in sites.columns if isinstance(c, str) and "nanoPhos_dilser" in c]
    alpha = alpha_to_long(sites, sample_cols)

    # Pair on capped key
    merged = native.merge(alpha, on=["match_key_capped", "sample"], how="inner")
    both = merged.dropna(subset=["native_quant", "alpha_log2"]).copy()
    both["native_log2"] = np.log2(both["native_quant"].replace(0, np.nan))
    both = both.dropna(subset=["native_log2"])
    both["log2_diff"] = both["alpha_log2"] - both["native_log2"]
    print(f"\n  paired non-NaN cells: {len(both):,}")

    print("\n=== Stratification by RAW native multiplicity ===")
    strat = both.groupby("mult_raw").agg(
        n=("log2_diff", "size"),
        mean=("log2_diff", "mean"),
        median=("log2_diff", "median"),
        sd=("log2_diff", "std"),
        p95=("log2_diff", lambda s: s.quantile(0.95)),
        pct_within_0_1=("log2_diff", lambda s: (s.abs() < 0.1).mean() * 100),
        pct_within_0_5=("log2_diff", lambda s: (s.abs() < 0.5).mean() * 100),
        pct_alpha_high_1log2=("log2_diff", lambda s: (s > 1.0).mean() * 100),
    ).round(4)
    print(strat.to_string())

    print("\n=== Stratification by CAPPED multiplicity (what alphaPhos sees) ===")
    strat2 = both.groupby("alpha_mult").agg(
        n=("log2_diff", "size"),
        mean=("log2_diff", "mean"),
        median=("log2_diff", "median"),
        sd=("log2_diff", "std"),
        p95=("log2_diff", lambda s: s.quantile(0.95)),
        pct_within_0_1=("log2_diff", lambda s: (s.abs() < 0.1).mean() * 100),
        pct_within_0_5=("log2_diff", lambda s: (s.abs() < 0.5).mean() * 100),
    ).round(4)
    print(strat2.to_string())

    # How much of the right tail (alpha higher by >1 log2) comes from M3+ sites?
    tail = both[both["log2_diff"] > 1.0]
    print(f"\n=== Right-tail (alpha higher by >1 log2) breakdown ===")
    print(f"  total tail cells: {len(tail):,} ({len(tail)/len(both)*100:.2f}% of paired)")
    print(f"  mult_raw distribution in tail:")
    print(tail.groupby("mult_raw").size().to_string())
    print(f"  tail cells that are mult_raw >= 3: {len(tail[tail['mult_raw'] >= 3]):,} "
          f"({len(tail[tail['mult_raw'] >= 3])/len(tail)*100:.1f}% of tail)")
    print(f"  tail cells that are mult_raw >= 4 (capping victims): "
          f"{len(tail[tail['mult_raw'] >= 4]):,} "
          f"({len(tail[tail['mult_raw'] >= 4])/len(tail)*100:.1f}% of tail)")

    # Save artifacts
    strat.to_csv(OUT_DIR / "stratify_by_native_mult_raw.csv")
    strat2.to_csv(OUT_DIR / "stratify_by_alpha_mult_capped.csv")
    summary = {
        "n_paired_cells": int(len(both)),
        "stratify_native_mult_raw": strat.reset_index().to_dict("records"),
        "stratify_alpha_mult_capped": strat2.reset_index().to_dict("records"),
        "tail_total": int(len(tail)),
        "tail_from_mult4plus": int(len(tail[tail["mult_raw"] >= 4])),
        "tail_from_mult4plus_pct": round(
            len(tail[tail["mult_raw"] >= 4]) / max(1, len(tail)) * 100, 2
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                          encoding="utf-8")
    print(f"\nArtifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
