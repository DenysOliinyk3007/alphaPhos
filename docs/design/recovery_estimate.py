"""Recovery estimate: combining linear-model imputation with condition-aware Class I.

Compares four configurations on Dublin's nanoPhos test data (12 samples,
4 conditions x 3 reps each):
    baseline : median aggregation + condition-aware classify (current Dublin pipeline)
    A        : consolidate (linear-model) aggregation + condition-aware classify
    B        : median + classify + post-classify within-condition recovery
    A+B      : consolidate + classify + post-classify within-condition recovery

Recovery rule (B): impute cells that are
    (a) NaN after classify, AND
    (b) NaN before classify (pre-existing genuine quant dropouts, NOT mask-induced), AND
    (c) in a condition that is "high-trust" for this site (classify fraction >= 0.5)
using the median of the non-NaN replicates of the same site in the same condition.
This avoids contradicting the classify mask (we never impute mask-induced NaNs) and
avoids cross-condition imputation (we never impute from a different biological group).
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

DUBLIN_SRC = Path("d:/Projects/Dublin/testscripts/src")
DATA_TSV = Path(
    "d:/Projects/Dublin/testscripts/raw_data/20260516_092715_e112e115_test_nanoPhos_Report.tsv"
)

sys.path.insert(0, str(DUBLIN_SRC))

from PeptideCollapse_v4 import PeptideCollapse  # noqa: E402
from condition_aware_classI import apply_condition_aware_classI_mask  # noqa: E402

# Switch cwd so PeptideCollapse.log is written next to the script, not to repo root.
import os  # noqa: E402
os.chdir(Path(__file__).parent)


def deduce_condition(fname: str) -> str:
    exp = re.search(r"(e11[25])", fname, flags=re.IGNORECASE)
    treat = re.search(r"(IgG|PD1)", fname, flags=re.IGNORECASE)
    if exp is None or treat is None:
        return "UNKNOWN"
    return f"{exp.group(1).lower()}_{treat.group(1)}"


def run_collapse(df_raw: pd.DataFrame, aggregation: str, label: str):
    t0 = time.time()
    pc = PeptideCollapse(verbose=False)
    sites = pc.process_complete_pipeline(
        df_raw,
        cutoff=0.0,
        collapse_level="PG",
        aggregation_method=aggregation,
        add_kinase_sequences=False,
        noise_floor_filter=True,
        localization_strategy="global_max",
    )
    elapsed = time.time() - t0
    print(f"  collapse({aggregation:11s}): {len(sites):>6d} sites pre-classify  [{elapsed:5.1f}s]")
    return sites, pc.site_localization_per_run


def run_classify(sites, loc, sample_to_condition, drop_all_nan: bool):
    out, decision = apply_condition_aware_classI_mask(
        sites,
        loc,
        sample_to_condition,
        classI_cutoff=0.75,
        condition_threshold=0.50,
        drop_all_nan=drop_all_nan,
        return_decision_table=True,
    )
    return out, decision


def within_condition_recovery(
    sites_pre: pd.DataFrame,
    sites_post: pd.DataFrame,
    decision: pd.DataFrame,
    sample_to_cond: dict,
    condition_threshold: float = 0.50,
):
    """Impute high-trust pre-existing-NaN cells from same-condition median.

    Returns (recovered_df, n_cells_imputed).
    """
    sample_cols = [c for c in sites_post.columns if c in sample_to_cond]
    s2c = pd.Series(sample_to_cond)

    pre_idx = sites_pre.set_index("PTM_Collapse_key")[sample_cols]
    post_idx = sites_post.set_index("PTM_Collapse_key")[sample_cols].copy()
    common = post_idx.index.intersection(pre_idx.index).intersection(decision.index)
    pre_idx = pre_idx.loc[common]
    post_idx = post_idx.loc[common]
    decision = decision.loc[common]

    n_imputed = 0
    for cond in decision.columns:
        cond_samples = [c for c in s2c[s2c == cond].index if c in sample_cols]
        if len(cond_samples) < 2:
            continue
        high_trust = decision[cond] >= condition_threshold
        if not high_trust.any():
            continue

        sub_post = post_idx.loc[high_trust, cond_samples]
        sub_pre = pre_idx.loc[high_trust, cond_samples]
        row_med = sub_post.median(axis=1, skipna=True)

        for col in cond_samples:
            cand = sub_post[col].isna() & sub_pre[col].isna() & row_med.notna()
            if not cand.any():
                continue
            sub_post.loc[cand, col] = row_med[cand]
            n_imputed += int(cand.sum())

        post_idx.loc[high_trust, cond_samples] = sub_post

    out = sites_post.set_index("PTM_Collapse_key").copy()
    out.update(post_idx)
    out = out.reset_index()
    return out, n_imputed


def drop_all_nan(df: pd.DataFrame, sample_to_cond: dict) -> pd.DataFrame:
    sample_cols = [c for c in df.columns if c in sample_to_cond]
    keep = ~df[sample_cols].isna().all(axis=1)
    return df.loc[keep].reset_index(drop=True)


def metrics(df: pd.DataFrame, label: str, sample_to_cond: dict) -> dict:
    sample_cols = [c for c in df.columns if c in sample_to_cond]
    block = df[sample_cols]
    n_sites = len(df)
    n_cells = block.size
    n_nan = int(block.isna().sum().sum())
    n_present = n_cells - n_nan
    completeness = n_present / n_cells * 100 if n_cells > 0 else 0.0
    return {
        "config": label,
        "n_sites": n_sites,
        "cells_present": n_present,
        "cells_nan": n_nan,
        "completeness_pct": round(completeness, 2),
    }


def main() -> None:
    print(f"Loading {DATA_TSV.name} ...")
    t0 = time.time()
    df_raw = pd.read_csv(DATA_TSV, sep="\t", low_memory=False)
    print(f"  loaded {len(df_raw):>7d} rows in {time.time()-t0:.1f}s")

    samples = df_raw["R.FileName"].drop_duplicates().tolist()
    sample_to_cond = {s: deduce_condition(s) for s in samples}
    assert all(c != "UNKNOWN" for c in sample_to_cond.values())
    cond_counts = pd.Series(sample_to_cond).value_counts().to_dict()
    print(f"  conditions: {cond_counts}\n")

    # === Collapse runs ===
    print("Collapse stage:")
    sites_med, loc_med = run_collapse(df_raw, "median", "median")
    sites_con, loc_con = run_collapse(df_raw, "consolidate", "consolidate")
    print()

    # === Classify (keep all-NaN rows so we can measure recovery's effect on site survival) ===
    print("Classify (condition-aware, drop_all_nan=False):")
    sites_med_C_nodrop, decision_med = run_classify(
        sites_med, loc_med, sample_to_cond, drop_all_nan=False
    )
    sites_con_C_nodrop, decision_con = run_classify(
        sites_con, loc_con, sample_to_cond, drop_all_nan=False
    )

    # === B: post-classify within-condition recovery ===
    print("Within-condition recovery (B):")
    sites_med_CB_nodrop, n_imp_med = within_condition_recovery(
        sites_med, sites_med_C_nodrop, decision_med, sample_to_cond
    )
    print(f"  median+classify: imputed {n_imp_med:>6d} cells")
    sites_con_CB_nodrop, n_imp_con = within_condition_recovery(
        sites_con, sites_con_C_nodrop, decision_con, sample_to_cond
    )
    print(f"  consolidate+classify: imputed {n_imp_con:>6d} cells\n")

    # === Now drop all-NaN to get final site counts ===
    print("Final drop_all_nan:")
    final = {
        "baseline (median + classify)": drop_all_nan(sites_med_C_nodrop, sample_to_cond),
        "A (consolidate + classify)": drop_all_nan(sites_con_C_nodrop, sample_to_cond),
        "B (median + classify + recovery)": drop_all_nan(sites_med_CB_nodrop, sample_to_cond),
        "A+B (consolidate + classify + recovery)": drop_all_nan(sites_con_CB_nodrop, sample_to_cond),
    }
    for label, df in final.items():
        print(f"  {label:<45s}: {len(df):>6d} sites")
    print()

    # === Summary table ===
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    rows = []
    # Pre-classify references
    rows.append(metrics(sites_med, "0a. median   (pre-classify reference)", sample_to_cond))
    rows.append(metrics(sites_con, "0b. consolidate (pre-classify reference)", sample_to_cond))
    rows.append({"config": "---", "n_sites": "---", "cells_present": "---",
                 "cells_nan": "---", "completeness_pct": "---"})
    rows.append(metrics(final["baseline (median + classify)"],
                        "1. baseline: median + classify", sample_to_cond))
    rows.append(metrics(final["A (consolidate + classify)"],
                        "2. A: consolidate + classify", sample_to_cond))
    rows.append(metrics(final["B (median + classify + recovery)"],
                        "3. B: median + classify + recovery", sample_to_cond))
    rows.append(metrics(final["A+B (consolidate + classify + recovery)"],
                        "4. A+B: consolidate + classify + recovery", sample_to_cond))

    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    print()

    # === Per-condition class I fractions ===
    print("Mean condition-aware class I fraction (per site, then averaged):")
    print(f"  median collapse:      {decision_med.mean().round(3).to_dict()}")
    print(f"  consolidate collapse: {decision_con.mean().round(3).to_dict()}")
    print()

    # === Recovery deltas vs baseline ===
    base = metrics(final["baseline (median + classify)"], "baseline", sample_to_cond)
    print("Deltas vs baseline:")
    for cfg in ["A (consolidate + classify)",
                "B (median + classify + recovery)",
                "A+B (consolidate + classify + recovery)"]:
        m = metrics(final[cfg], cfg, sample_to_cond)
        ds = m["n_sites"] - base["n_sites"]
        dc = m["cells_present"] - base["cells_present"]
        print(f"  {cfg:<45s}: sites {ds:+6d} ({ds/base['n_sites']*100:+5.1f}%), "
              f"cells {dc:+7d} ({dc/base['cells_present']*100:+5.1f}%)")


if __name__ == "__main__":
    main()
