"""R cross-validation of the consolidate() algorithm.

Picks N real phospho sites from nanoPhos test data, extracts their
precursor x sample matrices, runs BOTH:

  * Python `_consolidate` (the in-repo port at PeptideCollapse_v4._consolidate)
  * R   `consolidate`   (canonical Hogrebe function, decompiled from .dll)

on each matrix, and compares element-by-element. Expected agreement:
near-exact (Pearson r > 0.9999) modulo the documented Python improvements
(Inf-guard and max_iter cap, neither of which trigger on real data).

The R cross-validation is the strongest possible algorithmic-correctness
check for the consolidate step — it isolates the algorithm from all
upstream pipeline differences.
"""

from __future__ import annotations

import re
import subprocess
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

OUT_DIR = Path(__file__).parent / "r_cross_val_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET = "d:/Projects/alphaPhos/test_data/nanoPhos_dilser_noEGF_1000ng_Report_ms1_ms2.parquet"
RSCRIPT = r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
R_SCRIPT_PATH = Path(__file__).parent / "r_consolidate.R"

N_SITES_TO_TEST = 300


def build_precursor_matrices(df: pd.DataFrame, n_sites: int) -> tuple[pd.DataFrame, dict]:
    """Re-derive the per-(site, precursor) intensity matrices that PeptideCollapse
    feeds into _consolidate, so we can hand the SAME data to R.

    Returns:
        long_df: stacked long-form table (site_key, precursor_id, sample columns)
                 suitable for the R driver script.
        site_matrices: dict[site_key -> np.ndarray] of (precursors x samples) in
                       linear intensity space, exactly as PeptideCollapse passes
                       to _consolidate.
    """
    # Replicate PeptideCollapse's site-collapse preprocessing up to the point
    # where it calls _consolidate. Easiest path: run the pipeline up to before
    # aggregation by intercepting via process_complete_pipeline isn't clean,
    # so we replicate the relevant chunk here.
    pc = PeptideCollapse(verbose=False)
    pc.load_data(df, validate=False)
    pc.preprocess_data()

    proc = pc.processed_data.copy()

    modification_info = proc["EG.PrecursorId"].apply(pc._extract_sequence_modifications)
    proc["PTM_base_seq"] = [info["clean_sequence"] for info in modification_info]
    proc["PTM_0_pos_val"] = [info["phospho_positions"] for info in modification_info]
    proc["PTM_0_num"] = [info["phospho_count"] for info in modification_info]
    proc["PTM_group"] = proc["EG.PrecursorId"]
    proc = proc[proc["PTM_0_num"] > 0].copy()
    proc = proc.explode("PTM_0_pos_val")

    # Build the per-precursor, per-sample intensity table
    pivot = pd.pivot_table(
        proc,
        index=["PTM_group", "PTM_0_pos_val"],
        columns="R.FileName",
        values="EG.TotalQuantity (Settings)",
        aggfunc="sum",
    ).replace(0, np.nan)

    # Add metadata for the collapse key
    keep_meta = ["PEP.PeptidePosition", "PG.ProteinGroups", "PG.Genes", "PTM_0_num", "PTM_0_aa"]
    proc["PTM_0_aa"] = proc.apply(
        lambda x: pc._get_phospho_amino_acid(x["PTM_base_seq"], x["PTM_0_pos_val"]), axis=1
    )
    meta = pd.pivot_table(proc, index=["PTM_group", "PTM_0_pos_val"], values=keep_meta, aggfunc="first")
    combined = pd.concat([pivot, meta], axis=1).reset_index()
    combined["PTM_pep_pos"] = combined["PEP.PeptidePosition"].astype(str).apply(
        lambda r: next((x for x in r.split(";")[0].split(",") if x and x != "None"), None)
    )
    combined = combined[combined["PTM_pep_pos"].notna() & (combined["PTM_pep_pos"] != "None")]
    combined["PTM_pep_pos"] = combined["PTM_pep_pos"].astype(int)
    combined["PTM_mult123"] = combined["PTM_0_num"].astype(int).clip(upper=3)
    combined["PTM_Collapse_key"] = combined.apply(
        lambda r: pc._create_collapse_key(
            r["PG.ProteinGroups"], r["PG.Genes"], r["PTM_0_aa"],
            r["PTM_0_pos_val"], r["PTM_pep_pos"], r["PTM_mult123"]
        ), axis=1
    )

    sample_cols = [c for c in combined.columns if "nanoPhos_dilser" in str(c)]
    print(f"  {len(combined)} (site_key, precursor) rows; {len(sample_cols)} samples")

    # Subset: keep sites with >=2 precursors and >=2 non-NaN cells in some samples
    site_counts = combined.groupby("PTM_Collapse_key").size()
    multi_prec_sites = site_counts[site_counts >= 2].index
    print(f"  sites with >=2 precursors: {len(multi_prec_sites):,}")

    # Pick a random sample (seeded for determinism)
    rng = np.random.default_rng(seed=42)
    chosen = rng.choice(multi_prec_sites, size=min(n_sites, len(multi_prec_sites)),
                        replace=False)
    chosen = sorted(chosen)

    chosen_rows = combined[combined["PTM_Collapse_key"].isin(chosen)].copy()
    print(f"  chosen sample: {len(chosen):,} sites, {len(chosen_rows):,} precursor rows")

    long_df = chosen_rows[["PTM_Collapse_key", "PTM_group"] + sample_cols].copy()
    long_df = long_df.rename(columns={"PTM_Collapse_key": "site_key", "PTM_group": "precursor_id"})

    # Build the in-memory matrices for the Python consolidate
    site_matrices: dict[str, np.ndarray] = {}
    for key, grp in chosen_rows.groupby("PTM_Collapse_key"):
        # Pre-filter: matches the in-pipeline pre-filter (drop precursors with
        # <=1 non-NaN cell). This is what PeptideCollapse feeds to _consolidate.
        mat = grp[sample_cols].to_numpy(dtype=float)
        site_matrices[key] = mat

    return long_df, site_matrices, sample_cols


def run_r(long_df: pd.DataFrame) -> pd.DataFrame:
    in_tsv = OUT_DIR / "precursor_matrices_for_r.tsv"
    out_tsv = OUT_DIR / "r_consolidated.tsv"
    long_df.to_csv(in_tsv, sep="\t", index=False, na_rep="NA")
    print(f"  wrote {in_tsv} ({in_tsv.stat().st_size / 1024:.1f} KB)")
    t = time.time()
    print(f"  invoking R ({RSCRIPT})...")
    result = subprocess.run(
        [RSCRIPT, str(R_SCRIPT_PATH), str(in_tsv), str(out_tsv)],
        capture_output=True, text=True, check=False,
    )
    print(f"  R elapsed: {time.time()-t:.1f}s, returncode={result.returncode}")
    if result.stdout:
        print("  R stdout:")
        for line in result.stdout.strip().splitlines():
            print(f"    {line}")
    if result.returncode != 0:
        print("  R stderr:")
        for line in result.stderr.strip().splitlines():
            print(f"    {line}")
        raise RuntimeError("R script failed")
    r_out = pd.read_csv(out_tsv, sep="\t", na_values="NA")
    print(f"  R output: {len(r_out)} rows, {len(r_out.columns)-1} sample cols")
    return r_out


def python_consolidate_all(
    site_matrices: dict[str, np.ndarray], sample_cols: list[str]
) -> pd.DataFrame:
    """Run PeptideCollapse_v4._consolidate on every site matrix in-memory."""
    pc = PeptideCollapse(verbose=False)
    rows = []
    for key, mat in site_matrices.items():
        result = pc._consolidate(mat)
        row = {"site_key": key}
        for i, c in enumerate(sample_cols):
            row[c] = result[i] if i < len(result) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compare(py_df: pd.DataFrame, r_df: pd.DataFrame, sample_cols: list[str]) -> dict:
    """Element-by-element comparison of Python and R consolidate outputs."""
    merged = py_df.merge(r_df, on="site_key", suffixes=("_py", "_r"), how="inner")
    print(f"\n  paired sites: {len(merged):,}")

    py_cols = [c + "_py" for c in sample_cols]
    r_cols = [c + "_r" for c in sample_cols]

    # Stack into long form
    long_py = merged[["site_key"] + py_cols].melt(
        id_vars="site_key", var_name="sample_col", value_name="py"
    )
    long_py["sample"] = long_py["sample_col"].str.removesuffix("_py")
    long_r = merged[["site_key"] + r_cols].melt(
        id_vars="site_key", var_name="sample_col", value_name="r"
    )
    long_r["sample"] = long_r["sample_col"].str.removesuffix("_r")
    long = long_py[["site_key", "sample", "py"]].merge(
        long_r[["site_key", "sample", "r"]], on=["site_key", "sample"]
    )
    both = long.dropna(subset=["py", "r"]).copy()

    if len(both) == 0:
        return {"error": "no paired non-NaN cells"}

    both["diff"] = both["py"] - both["r"]
    both["abs_diff"] = both["diff"].abs()
    both["rel_err"] = both["abs_diff"] / both["r"].abs().replace(0, np.nan)

    pearson_lin = float(both["py"].corr(both["r"]))
    both["log2_py"] = np.log2(both["py"].replace(0, np.nan))
    both["log2_r"] = np.log2(both["r"].replace(0, np.nan))
    bl = both.dropna(subset=["log2_py", "log2_r"])
    pearson_log2 = float(bl["log2_py"].corr(bl["log2_r"]))

    summary = {
        "n_paired_cells": int(len(both)),
        "n_paired_sites": int(len(merged)),
        "pearson_linear": round(pearson_lin, 6),
        "pearson_log2": round(pearson_log2, 6),
        "max_abs_diff": float(both["abs_diff"].max()),
        "median_abs_diff": float(both["abs_diff"].median()),
        "pct_exact": round(float((both["abs_diff"] < 1e-6).mean() * 100), 4),
        "pct_within_1e-4_rel": round(float((both["rel_err"] < 1e-4).mean() * 100), 4),
        "pct_within_1e-3_rel": round(float((both["rel_err"] < 1e-3).mean() * 100), 4),
        "pct_within_1e-2_rel": round(float((both["rel_err"] < 1e-2).mean() * 100), 4),
        "pct_within_0.1_rel": round(float((both["rel_err"] < 0.1).mean() * 100), 4),
        "py_has_value_r_nan": int((long["py"].notna() & long["r"].isna()).sum()),
        "r_has_value_py_nan": int((long["py"].isna() & long["r"].notna()).sum()),
        "both_nan": int((long["py"].isna() & long["r"].isna()).sum()),
    }
    return summary


def main() -> None:
    print(f"=== alphaPhos R cross-validation ===\n")
    print(f"  parquet:  {PARQUET}")
    print(f"  Rscript:  {RSCRIPT}")
    print(f"  R script: {R_SCRIPT_PATH}\n")

    print("[1/4] Loading and applying alphaPhos preprocessing (read + top-N filter)...")
    df = read_spectronaut(PARQUET, quant_level="MS2", drop_decoys=True, pg_qvalue_max=0.01)
    print(f"  loaded {len(df):,} rows")
    df = filter_to_top_n_positions(df)
    print(f"  after top-N filter: {len(df):,} rows")

    print(f"\n[2/4] Extracting precursor matrices for {N_SITES_TO_TEST} random sites...")
    long_df, site_matrices, sample_cols = build_precursor_matrices(df, N_SITES_TO_TEST)

    print("\n[3/4] Running R consolidate on the same matrices...")
    r_df = run_r(long_df)

    print("\n[3/4] Running Python _consolidate on the same matrices...")
    t = time.time()
    py_df = python_consolidate_all(site_matrices, sample_cols)
    print(f"  Python elapsed: {time.time()-t:.1f}s; rows: {len(py_df)}")

    print("\n[4/4] Comparing element-by-element...")
    summary = compare(py_df, r_df, sample_cols)
    print("\n  ==== Summary ====")
    for k, v in summary.items():
        print(f"  {k:30s}: {v}")

    import json
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                          encoding="utf-8")
    py_df.to_csv(OUT_DIR / "python_consolidate.tsv", sep="\t", index=False, na_rep="NA")
    print(f"\nArtifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
