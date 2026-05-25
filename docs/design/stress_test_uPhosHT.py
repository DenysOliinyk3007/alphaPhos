"""Stress-test alphaPhos pipeline (baseline vs top-N fix) across uPhosHT datasets.

For each Spectronaut PSM export we run:
  * baseline: PeptideCollapse_v4 with default config (no attribution fix)
  * top-N fix: same config, with filter_to_top_n_positions applied first

and compute internal metrics:
  * site count, completeness, per-sample ID counts, median per-site CV (linear),
    runtime, % rows retained by the top-N filter

Goal: verify the top-N attribution fix produces sane, consistent behavior across
a range of datasets (varying size, conditions, protocols), not just nanoPhos.
No native PTM site reports exist in uPhosHT, so this is INTERNAL CONSISTENCY
testing — we're checking whether the fix breaks anything or behaves uniformly.

Datasets are processed in size-ascending order so failures surface fast.
Results are checkpointed after each file (results.json).
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "d:/Projects/Dublin/testscripts/src")
sys.path.insert(0, str(Path(__file__).parent))

from alphaphos.io import read_spectronaut  # noqa: E402
from PeptideCollapse_v4 import PeptideCollapse  # noqa: E402
from prototype_top_n_attribution import filter_to_top_n_positions  # noqa: E402

OUT_DIR = Path(__file__).parent / "stress_test_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("elution_A1",      "d:/Projects/uPhosHT/raw_data/elution_test_data/uPhosHT_elution_test_A1_Report.tsv"),
    ("elution_F3",      "d:/Projects/uPhosHT/raw_data/elution_test_data/uPhosHT_elution_test_F3_Report.tsv"),
    ("gradient_500SPD", "d:/Projects/uPhosHT/raw_data/uPhos_HT_gradient_test_20uL_500SPD_Report.tsv"),
    ("gradient_300SPD", "d:/Projects/uPhosHT/raw_data/uPhos_HT_gradient_test_20uL_300SPD_Report.tsv"),
    ("gradient_200SPD", "d:/Projects/uPhosHT/raw_data/uPhosHT_gradient_test_20uL_200SPD_Report.tsv"),
    ("volume_10ul",     "d:/Projects/uPhosHT/raw_data/20260223_032012_uPhosHT_volume_test_10ul_Report.tsv"),
    ("volume_all",      "d:/Projects/uPhosHT/raw_data/20260406_090154_uPhosHT_volume_test_all_Report.tsv"),
]


def collect_metrics(sites: pd.DataFrame, samples: list[str]) -> dict:
    """Common per-dataset metrics: site count, completeness, per-sample IDs, per-site CV."""
    sample_cols = [c for c in sites.columns if c in samples]
    if not sample_cols:
        # Fallback — sites' sample columns are R.FileName values
        sample_cols = [c for c in sites.columns if isinstance(c, str) and any(s in c for s in samples)]
    block = sites[sample_cols].select_dtypes(include=[np.number])
    n_present = int(block.notna().sum().sum())
    n_total = int(block.size)
    completeness = round(n_present / n_total * 100, 2) if n_total > 0 else 0.0
    per_sample_ids = block.notna().sum(axis=0)
    ids_summary = {
        "min": int(per_sample_ids.min()) if len(per_sample_ids) else None,
        "max": int(per_sample_ids.max()) if len(per_sample_ids) else None,
        "median": int(per_sample_ids.median()) if len(per_sample_ids) else None,
    }
    cv_median = None
    if len(sample_cols) >= 2 and len(block) > 0:
        linear = np.power(2.0, block)
        mean = linear.mean(axis=1, skipna=True)
        sd = linear.std(axis=1, skipna=True, ddof=1)
        cv = (sd / mean * 100).dropna()
        if len(cv):
            cv_median = round(float(cv.median()), 2)
    return {
        "n_sites": int(len(sites)),
        "n_samples": len(sample_cols),
        "completeness_pct": completeness,
        "cells_present": n_present,
        "cells_total": n_total,
        "ids_per_sample": ids_summary,
        "per_site_cv_median_pct": cv_median,
    }


def run_one(label: str, path: str) -> dict:
    print(f"\n{'='*72}\n[{label}] {Path(path).name}\n{'='*72}")
    size_mb = round(Path(path).stat().st_size / 1e6, 1)
    print(f"  size: {size_mb} MB")

    t = time.time()
    df = read_spectronaut(path, quant_level="auto", drop_decoys=True, pg_qvalue_max=0.01)
    load_s = round(time.time() - t, 1)
    n_rows = len(df)
    samples = sorted(df["R.FileName"].unique())
    n_samples = len(samples)
    n_conditions = (
        int(df["R.Condition"].nunique()) if "R.Condition" in df.columns else None
    )
    quant_col = df.attrs["alphaphos_quant_column"]
    print(f"  loaded {n_rows:,} rows in {load_s}s, {n_samples} samples, "
          f"n_conditions={n_conditions}, quant_col={quant_col!r}")

    schema = {c: c in df.columns for c in [
        "FG.MS1Quantity", "FG.MS2Quantity", "FG.MS2RawQuantity",
        "EG.TotalQuantity (Settings)", "EG.PTMLocalizationProbabilities",
        "EG.PTMAssayProbability", "PG.Qvalue", "EG.Qvalue", "EG.IsDecoy",
        "EG.Cscore", "R.Condition",
    ]}

    # --- Baseline ---
    print("  [baseline] PeptideCollapse_v4, median + per_run, cutoff 0.75 ...")
    t = time.time()
    pc = PeptideCollapse(verbose=False)
    sites_base = pc.process_complete_pipeline(
        df, cutoff=0.75, collapse_level="PG",
        aggregation_method="median", add_kinase_sequences=False,
        noise_floor_filter=True, localization_strategy="per_run",
    )
    base_s = round(time.time() - t, 1)
    base_stats = dict(pc.processing_stats)
    base_metrics = collect_metrics(sites_base, samples)
    print(f"    -> {base_metrics['n_sites']:,} sites, "
          f"completeness {base_metrics['completeness_pct']}%, "
          f"CV med {base_metrics['per_site_cv_median_pct']}%, {base_s}s")

    # --- top-N filter ---
    print("  [filter] applying top-N attribution ...")
    t = time.time()
    df_fix = filter_to_top_n_positions(df)
    filter_s = round(time.time() - t, 1)
    n_fix_rows = len(df_fix)
    retention = round(n_fix_rows / n_rows * 100, 2)
    print(f"    -> {n_fix_rows:,} rows retained ({retention}%), {filter_s}s")

    # --- Fixed ---
    print("  [fixed] PeptideCollapse_v4 on filtered data ...")
    t = time.time()
    pc2 = PeptideCollapse(verbose=False)
    sites_fix = pc2.process_complete_pipeline(
        df_fix, cutoff=0.75, collapse_level="PG",
        aggregation_method="median", add_kinase_sequences=False,
        noise_floor_filter=True, localization_strategy="per_run",
    )
    fix_s = round(time.time() - t, 1)
    fix_metrics = collect_metrics(sites_fix, samples)
    print(f"    -> {fix_metrics['n_sites']:,} sites, "
          f"completeness {fix_metrics['completeness_pct']}%, "
          f"CV med {fix_metrics['per_site_cv_median_pct']}%, {fix_s}s")

    # --- Deltas ---
    def _delta_pct(a, b):
        return round((b - a) / a * 100, 2) if a else None

    delta = {
        "delta_sites": fix_metrics["n_sites"] - base_metrics["n_sites"],
        "delta_sites_pct": _delta_pct(base_metrics["n_sites"], fix_metrics["n_sites"]),
        "delta_cells": fix_metrics["cells_present"] - base_metrics["cells_present"],
        "delta_cells_pct": _delta_pct(base_metrics["cells_present"], fix_metrics["cells_present"]),
        "delta_completeness_pp": round(
            fix_metrics["completeness_pct"] - base_metrics["completeness_pct"], 2
        ),
        "delta_cv_pp": (
            round(fix_metrics["per_site_cv_median_pct"] - base_metrics["per_site_cv_median_pct"], 2)
            if base_metrics["per_site_cv_median_pct"] and fix_metrics["per_site_cv_median_pct"]
            else None
        ),
    }

    print(f"  [delta] sites: {delta['delta_sites']:+d} ({delta['delta_sites_pct']}%),  "
          f"cells: {delta['delta_cells']:+d} ({delta['delta_cells_pct']}%),  "
          f"completeness: {delta['delta_completeness_pp']:+.2f} pp,  "
          f"CV: {delta['delta_cv_pp']} pp")

    return {
        "label": label,
        "path": path,
        "size_MB": size_mb,
        "n_rows_raw": n_rows,
        "n_samples": n_samples,
        "n_conditions": n_conditions,
        "schema": schema,
        "quant_column": quant_col,
        "load_s": load_s,
        "baseline_runtime_s": base_s,
        "filter_runtime_s": filter_s,
        "fixed_runtime_s": fix_s,
        "top_n_retention_pct": retention,
        "baseline": base_metrics,
        "fixed": fix_metrics,
        "delta": delta,
        "baseline_processing_stats": base_stats,
    }


def render_summary(results: list[dict]) -> str:
    rows = []
    for r in results:
        if "error" in r:
            rows.append({"label": r["label"], "error": r["error"]})
            continue
        rows.append({
            "label": r["label"],
            "size_MB": r["size_MB"],
            "n_rows": r["n_rows_raw"],
            "n_samples": r["n_samples"],
            "n_conditions": r["n_conditions"],
            "top_n_keep_%": r["top_n_retention_pct"],
            "base_sites": r["baseline"]["n_sites"],
            "fix_sites": r["fixed"]["n_sites"],
            "Δsites_%": r["delta"]["delta_sites_pct"],
            "base_compl_%": r["baseline"]["completeness_pct"],
            "fix_compl_%": r["fixed"]["completeness_pct"],
            "Δcompl_pp": r["delta"]["delta_completeness_pp"],
            "base_CV_%": r["baseline"]["per_site_cv_median_pct"],
            "fix_CV_%": r["fixed"]["per_site_cv_median_pct"],
            "ΔCV_pp": r["delta"]["delta_cv_pp"],
            "base_t_s": r["baseline_runtime_s"],
            "fix_t_s": r["fixed_runtime_s"],
        })
    df = pd.DataFrame(rows)
    return df.to_string(index=False)


def main() -> None:
    results: list[dict] = []
    for label, path in DATASETS:
        if not Path(path).exists():
            print(f"SKIP: {path} not found")
            continue
        try:
            res = run_one(label, path)
        except Exception as exc:
            print(f"\n!!! ERROR on {label}:\n{traceback.format_exc()}")
            res = {"label": label, "path": path, "error": str(exc),
                   "traceback": traceback.format_exc()}
        results.append(res)
        # Checkpoint after every dataset
        (OUT_DIR / "results.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        print(f"  [checkpoint] wrote results.json ({len(results)} datasets so far)")

    summary_text = render_summary(results)
    print("\n\n=" * 36)
    print("STRESS TEST SUMMARY")
    print("=" * 72)
    print(summary_text)

    # Markdown report
    md = [
        f"# uPhosHT stress test — alphaPhos baseline vs top-N attribution fix\n",
        f"_run: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n",
        f"Pipeline config: PeptideCollapse_v4 / collapse_level=PG / "
        f"aggregation=median / strategy=per_run / cutoff=0.75 / noise_floor=True\n",
        "## Summary",
        "",
        f"```\n{summary_text}\n```",
        "",
        "## Per-dataset details",
        "",
    ]
    for r in results:
        if "error" in r:
            md.extend([f"### {r['label']} — ERROR", "", f"```\n{r.get('traceback','')}\n```", ""])
            continue
        md.extend([
            f"### {r['label']}",
            f"- Path: `{r['path']}`",
            f"- Size: {r['size_MB']} MB, rows {r['n_rows_raw']:,}, "
            f"samples {r['n_samples']}, conditions {r['n_conditions']}",
            f"- Quant column used: `{r['quant_column']}`",
            f"- Baseline:  {r['baseline']['n_sites']:,} sites, "
            f"{r['baseline']['completeness_pct']}% complete, "
            f"CV med {r['baseline']['per_site_cv_median_pct']}%",
            f"- Top-N fix: {r['fixed']['n_sites']:,} sites, "
            f"{r['fixed']['completeness_pct']}% complete, "
            f"CV med {r['fixed']['per_site_cv_median_pct']}%",
            f"- Top-N filter retained {r['top_n_retention_pct']}% of rows",
            f"- Delta: sites {r['delta']['delta_sites']:+d} ({r['delta']['delta_sites_pct']}%), "
            f"completeness {r['delta']['delta_completeness_pp']:+.2f} pp, "
            f"CV {r['delta']['delta_cv_pp']} pp",
            "",
        ])
    (OUT_DIR / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nReport: {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
