"""Exhaustive MS1 vs MS2 benchmark for Spectronaut phospho exports.

Runs the canonical PeptideCollapse_v4 pipeline at multiple aggregation
methods x localization strategies on one or two Spectronaut exports, and
produces:

  * a per-file characterization (ID counts, completeness, CV, loc-prob
    distribution, classify-mask aggressiveness)
  * cross-file agreement metrics if two files are supplied (site overlap,
    per-cell intensity correlation, Bland-Altman, loc-prob consistency)
  * a Markdown report + CSV dump of every metric

USAGE
-----
    python benchmark_ms_levels.py <ms2_file> [<ms1_file>] [--out DIR]

Designed for paired MS1/MS2 exports of the SAME Spectronaut search (same
samples, same library, only Quantity MS Level differing). Files may also
be passed in either order, or alone; the runner auto-detects MS level
from each file's quant-column presence and labels accordingly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "d:/Projects/Dublin/testscripts/src")

from alphaphos.io import read_spectronaut  # noqa: E402
from alphaphos.io.spectronaut import QUANT_COLUMN_CANDIDATES  # noqa: E402
from PeptideCollapse_v4 import PeptideCollapse  # noqa: E402


STRATEGIES = ("per_run", "global_max")
AGGREGATIONS = ("median", "consolidate")
CUTOFF = 0.75


@dataclass
class PipelineResult:
    """A single (aggregation, strategy) pipeline run, with all metrics."""

    label: str
    aggregation: str
    strategy: str
    sites: pd.DataFrame = field(repr=False)
    processing_stats: dict = field(repr=False)
    runtime_s: float = 0.0


@dataclass
class FileBenchmark:
    """Everything we measure about one Spectronaut export."""

    path: str
    ms_level: str           # "MS1", "MS2", or "settings"
    quant_column: str
    n_rows_raw: int
    n_rows_phospho: int
    n_precursors: int
    n_phospho_precursors: int
    n_samples: int
    samples: list[str]
    selectivity_per_sample: dict[str, float]
    loc_prob_quantiles: dict[str, float]
    runs: dict[str, PipelineResult] = field(default_factory=dict)


def _detect_ms_level(df: pd.DataFrame) -> str:
    """Best-effort label for a loaded frame's MS level based on its quant col."""
    qcol = df.attrs.get("alphaphos_quant_column", "")
    if "MS1" in qcol:
        return "MS1"
    if "MS2" in qcol:
        return "MS2"
    return "settings"


def _selectivity_per_sample(df: pd.DataFrame) -> dict[str, float]:
    out = {}
    for fname, sub in df.groupby("R.FileName"):
        total = sub["EG.PrecursorId"].nunique()
        phospho = sub.loc[
            sub["EG.PrecursorId"].str.contains(r"\[Phospho \(STY\)\]", regex=True, na=False),
            "EG.PrecursorId",
        ].nunique()
        out[fname] = round(100.0 * phospho / total, 2) if total > 0 else 0.0
    return out


def _loc_prob_quantiles(df: pd.DataFrame) -> dict[str, float]:
    s = pd.to_numeric(df["EG.PTMAssayProbability"], errors="coerce").dropna()
    q = s.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    return {f"p{int(p*100):02d}": round(float(v), 3) for p, v in q.items()}


def _run_pipeline(
    df: pd.DataFrame, aggregation: str, strategy: str, label: str
) -> PipelineResult:
    t0 = time.time()
    pc = PeptideCollapse(verbose=False)
    sites = pc.process_complete_pipeline(
        df,
        cutoff=CUTOFF,
        collapse_level="PG",
        aggregation_method=aggregation,
        add_kinase_sequences=False,
        noise_floor_filter=True,
        localization_strategy=strategy,
    )
    elapsed = time.time() - t0
    print(f"    {label} ({aggregation}/{strategy}): {len(sites):>6d} sites  [{elapsed:5.1f}s]")
    return PipelineResult(
        label=label,
        aggregation=aggregation,
        strategy=strategy,
        sites=sites,
        processing_stats=dict(pc.processing_stats),
        runtime_s=round(elapsed, 2),
    )


def _site_sample_block(sites: pd.DataFrame, sample_names: list[str]) -> pd.DataFrame:
    """Return (sites x samples) numeric block keyed by PTM_Collapse_key."""
    cols = [c for c in sample_names if c in sites.columns]
    block = sites.set_index("PTM_Collapse_key")[cols].astype(float)
    return block


def _per_sample_metrics(sites: pd.DataFrame, sample_names: list[str]) -> pd.DataFrame:
    block = _site_sample_block(sites, sample_names)
    n_present = block.notna().sum(axis=0)
    n_total = pd.Series(len(block), index=block.columns, name="n_sites")
    completeness = (n_present / len(block) * 100).round(2)
    # Linear-space CV per sample is across-site noise, not really a "CV"; report
    # log2 median intensity and IQR instead.
    log2_med = block.median(axis=0).round(3)
    log2_iqr = (block.quantile(0.75, axis=0) - block.quantile(0.25, axis=0)).round(3)
    return pd.DataFrame(
        {
            "n_total": n_total,
            "n_present": n_present,
            "completeness_pct": completeness,
            "log2_median": log2_med,
            "log2_iqr": log2_iqr,
        }
    )


def _per_site_cv(sites: pd.DataFrame, sample_names: list[str]) -> pd.Series:
    """Across-replicate CV per site in LINEAR space.

    Treats all samples as one group (we have only one condition in dilser
    nanoPhos). For multi-condition datasets the caller should group first.
    """
    block = _site_sample_block(sites, sample_names)
    linear = np.power(2.0, block)  # log2 -> linear
    mean = linear.mean(axis=1, skipna=True)
    sd = linear.std(axis=1, skipna=True, ddof=1)
    cv = (sd / mean * 100).dropna()
    return cv


def benchmark_one(path: Path, quant_level: str) -> FileBenchmark:
    print(f"\n=== Loading {path.name} (quant_level={quant_level}) ===")
    t0 = time.time()
    df = read_spectronaut(path, quant_level=quant_level, drop_decoys=True, pg_qvalue_max=0.01)
    print(f"  loaded {len(df):>7d} rows in {time.time()-t0:.1f}s "
          f"(quant_col={df.attrs['alphaphos_quant_column']})")

    ms_level = _detect_ms_level(df)
    is_phospho = df["EG.PrecursorId"].str.contains(r"\[Phospho \(STY\)\]", regex=True, na=False)
    samples = sorted(df["R.FileName"].unique())

    fb = FileBenchmark(
        path=str(path),
        ms_level=ms_level,
        quant_column=df.attrs["alphaphos_quant_column"],
        n_rows_raw=len(df),
        n_rows_phospho=int(is_phospho.sum()),
        n_precursors=int(df["EG.PrecursorId"].nunique()),
        n_phospho_precursors=int(df.loc[is_phospho, "EG.PrecursorId"].nunique()),
        n_samples=len(samples),
        samples=samples,
        selectivity_per_sample=_selectivity_per_sample(df),
        loc_prob_quantiles=_loc_prob_quantiles(df),
    )

    print(f"  rows: {fb.n_rows_raw} (phospho {fb.n_rows_phospho})")
    print(f"  precursors: {fb.n_precursors} (phospho {fb.n_phospho_precursors})")
    print(f"  samples: {fb.n_samples}")
    print(f"  loc-prob quantiles: {fb.loc_prob_quantiles}")
    print("  Pipeline runs:")
    for aggregation in AGGREGATIONS:
        for strategy in STRATEGIES:
            key = f"{aggregation}_{strategy}"
            fb.runs[key] = _run_pipeline(df, aggregation, strategy, key)
    return fb


def compare_two(a: FileBenchmark, b: FileBenchmark) -> dict:
    """Cross-file metrics: site overlap, intensity agreement, loc consistency.

    Assumes a and b are paired exports (same samples, same library, only
    differing in quant level).
    """
    out = {}
    out["both_samples_match"] = sorted(a.samples) == sorted(b.samples)
    out["sample_intersection"] = sorted(set(a.samples) & set(b.samples))
    out["sample_a_only"] = sorted(set(a.samples) - set(b.samples))
    out["sample_b_only"] = sorted(set(b.samples) - set(a.samples))

    # Per-strategy overlap on PTM_Collapse_key
    overlaps = {}
    for key in a.runs:
        if key not in b.runs:
            continue
        sa = set(a.runs[key].sites["PTM_Collapse_key"])
        sb = set(b.runs[key].sites["PTM_Collapse_key"])
        overlaps[key] = {
            "a_total": len(sa),
            "b_total": len(sb),
            "intersection": len(sa & sb),
            "a_only": len(sa - sb),
            "b_only": len(sb - sa),
            "jaccard": round(len(sa & sb) / max(1, len(sa | sb)), 3),
        }
    out["overlap_per_run"] = overlaps

    # Per-cell intensity agreement (log2 space) on the median+per_run config
    if "median_per_run" in a.runs and "median_per_run" in b.runs:
        samples = out["sample_intersection"]
        block_a = _site_sample_block(a.runs["median_per_run"].sites, samples)
        block_b = _site_sample_block(b.runs["median_per_run"].sites, samples)
        common_sites = block_a.index.intersection(block_b.index)
        block_a = block_a.loc[common_sites]
        block_b = block_b.loc[common_sites]
        # Long-form for cell-level pairing
        flat_a = block_a.stack(dropna=False).rename("a")
        flat_b = block_b.stack(dropna=False).rename("b")
        merged = pd.concat([flat_a, flat_b], axis=1).dropna()
        if len(merged) > 100:
            pearson = float(merged["a"].corr(merged["b"]))
            spearman = float(merged["a"].corr(merged["b"], method="spearman"))
            diff = merged["a"] - merged["b"]
            bland_altman = {
                "mean_diff_log2": round(float(diff.mean()), 4),
                "sd_diff_log2": round(float(diff.std()), 4),
                "p05_diff": round(float(diff.quantile(0.05)), 4),
                "p95_diff": round(float(diff.quantile(0.95)), 4),
            }
            out["cell_agreement_median_per_run"] = {
                "n_paired_cells": int(len(merged)),
                "n_common_sites": int(len(common_sites)),
                "pearson_log2": round(pearson, 4),
                "spearman_log2": round(spearman, 4),
                "bland_altman": bland_altman,
            }

    # Localization consistency — should be IDENTICAL between MS1 and MS2 since
    # loc is derived from MS2 fragments regardless of quantification level.
    if "median_per_run" in a.runs and "median_per_run" in b.runs:
        loc_a = a.runs["median_per_run"].sites.set_index("PTM_Collapse_key")["PTM_localization"]
        loc_b = b.runs["median_per_run"].sites.set_index("PTM_Collapse_key")["PTM_localization"]
        common = loc_a.index.intersection(loc_b.index)
        if len(common) > 100:
            la = loc_a.loc[common]
            lb = loc_b.loc[common]
            agree = (la.round(4) == lb.round(4)).mean()
            out["localization_agreement"] = {
                "n_common": int(len(common)),
                "fraction_identical_4dp": round(float(agree), 4),
                "max_abs_diff": round(float((la - lb).abs().max()), 6),
            }

    return out


def _runs_summary_df(fb: FileBenchmark) -> pd.DataFrame:
    rows = []
    for key, r in fb.runs.items():
        psm = _per_sample_metrics(r.sites, fb.samples)
        # If we have replicates of one condition, per-site CV is meaningful
        cv = _per_site_cv(r.sites, fb.samples)
        rows.append(
            {
                "ms_level": fb.ms_level,
                "config": key,
                "aggregation": r.aggregation,
                "strategy": r.strategy,
                "n_sites": len(r.sites),
                "cells_present_total": int(psm["n_present"].sum()),
                "completeness_pct": round(float(psm["completeness_pct"].mean()), 2),
                "log2_median_avg": round(float(psm["log2_median"].mean()), 3),
                "log2_iqr_avg": round(float(psm["log2_iqr"].mean()), 3),
                "per_site_cv_median_pct": round(float(cv.median()), 1) if len(cv) else None,
                "per_site_cv_p25_pct": round(float(cv.quantile(0.25)), 1) if len(cv) else None,
                "per_site_cv_p75_pct": round(float(cv.quantile(0.75)), 1) if len(cv) else None,
                "runtime_s": r.runtime_s,
                "per_run_cells_masked": r.processing_stats.get("per_run_cells_masked", 0),
                "per_run_cells_total": r.processing_stats.get("per_run_cells_total", 0),
                "sites_dropped_all_nan": r.processing_stats.get("sites_dropped_all_nan", 0),
            }
        )
    return pd.DataFrame(rows)


def render_markdown_report(fbs: list[FileBenchmark], cross: Optional[dict], out_path: Path) -> None:
    lines: list[str] = []
    a = lines.append
    a("# MS1 vs MS2 collapse benchmark — alphaPhos\n")
    a(f"_generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n")

    a("## Files\n")
    for fb in fbs:
        a(f"- **{fb.ms_level}** — `{Path(fb.path).name}` "
          f"(quant_column = `{fb.quant_column}`, samples = {fb.n_samples}, "
          f"rows = {fb.n_rows_raw:,})")
    a("")

    a("## Per-file characterization\n")
    for fb in fbs:
        a(f"### {fb.ms_level}")
        a(f"- Rows: {fb.n_rows_raw:,} (phospho {fb.n_rows_phospho:,}; "
          f"{100*fb.n_rows_phospho/fb.n_rows_raw:.1f}%)")
        a(f"- Precursors: {fb.n_precursors:,} (phospho {fb.n_phospho_precursors:,})")
        a(f"- Samples: {fb.n_samples}")
        a(f"- Phospho selectivity per sample: "
          f"{ {k.split('_')[-1]: v for k, v in fb.selectivity_per_sample.items()} }")
        a(f"- Localization probability quantiles: {fb.loc_prob_quantiles}")
        a("")

    a("## Pipeline runs\n")
    full = pd.concat([_runs_summary_df(fb).assign(ms_level=fb.ms_level) for fb in fbs])
    a(full.to_markdown(index=False))
    a("")

    if cross is not None:
        a("## Cross-file comparison\n")
        a(f"- Same samples in both files: **{cross['both_samples_match']}**")
        if cross.get("sample_a_only") or cross.get("sample_b_only"):
            a(f"- Sample mismatch: {cross.get('sample_a_only')=}, "
              f"{cross.get('sample_b_only')=}")
        a("")

        a("### Site overlap per (aggregation, strategy)")
        ovr = pd.DataFrame(cross["overlap_per_run"]).T
        a(ovr.to_markdown())
        a("")

        if "cell_agreement_median_per_run" in cross:
            ca = cross["cell_agreement_median_per_run"]
            a("### Per-cell intensity agreement (median + per_run)")
            a(f"- N paired cells: {ca['n_paired_cells']:,}")
            a(f"- N common sites: {ca['n_common_sites']:,}")
            a(f"- Pearson r (log2): **{ca['pearson_log2']}**")
            a(f"- Spearman r (log2): **{ca['spearman_log2']}**")
            a(f"- Bland-Altman: mean diff (log2) = "
              f"{ca['bland_altman']['mean_diff_log2']}, "
              f"sd = {ca['bland_altman']['sd_diff_log2']}, "
              f"p5/p95 = [{ca['bland_altman']['p05_diff']}, "
              f"{ca['bland_altman']['p95_diff']}]")
            a("")

        if "localization_agreement" in cross:
            la = cross["localization_agreement"]
            a("### Localization probability consistency")
            a("Localization comes from MS2 fragments regardless of quant level — "
              "the two files should agree exactly.")
            a(f"- N common sites: {la['n_common']:,}")
            a(f"- Fraction identical (4 dp): **{la['fraction_identical_4dp']}**")
            a(f"- Max abs difference: {la['max_abs_diff']}")
            a("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written: {out_path}")


def _has_both_levels(path: Path) -> bool:
    """Quickly peek at the file's column names to see if it carries both
    explicit MS1 and MS2 quant columns. Avoids loading the full file."""
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        cols = {f.name for f in pq.ParquetFile(path).schema_arrow}
    else:
        cols = set(pd.read_csv(path, sep="\t", nrows=0).columns)
    has_ms1 = any(c.replace("_", ".").replace(".", "_") in cols
                  or any(c2 in cols for c2 in (c, c.replace(".", "_")))
                  for c in QUANT_COLUMN_CANDIDATES["MS1"])
    has_ms2 = any(c.replace("_", ".").replace(".", "_") in cols
                  or any(c2 in cols for c2 in (c, c.replace(".", "_")))
                  for c in QUANT_COLUMN_CANDIDATES["MS2"])
    # Simpler equivalent — check the underscore variants directly:
    has_ms1 = any(c.replace(".", "_").replace(" (", "_(") in cols
                  for c in QUANT_COLUMN_CANDIDATES["MS1"])
    has_ms2 = any(c.replace(".", "_").replace(" (", "_(") in cols
                  for c in QUANT_COLUMN_CANDIDATES["MS2"])
    return has_ms1 and has_ms2


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", type=Path,
                    help="One or two Spectronaut report files (.parquet or .tsv). "
                         "A single file carrying both MS1 and MS2 quant columns is "
                         "auto-treated as a paired benchmark.")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "benchmark_output",
                    help="Output directory for report + CSVs")
    ns = ap.parse_args(argv)

    out_dir: Path = ns.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Expand a single dual-quant file into two virtual loads (MS1 and MS2).
    expanded: list[tuple[Path, str]] = []
    if len(ns.files) == 1 and _has_both_levels(ns.files[0]):
        print(f"[auto] {ns.files[0].name} carries both MS1 and MS2 quant columns — "
              f"running paired benchmark.")
        expanded = [(ns.files[0], "MS1"), (ns.files[0], "MS2")]
    else:
        for f in ns.files:
            ql = "MS1" if "ms1" in f.name.lower() else "MS2" if "ms2" in f.name.lower() else "auto"
            expanded.append((f, ql))

    fbs: list[FileBenchmark] = []
    for f, ql in expanded:
        fbs.append(benchmark_one(f, ql))

    # Save per-file summaries
    for fb in fbs:
        _runs_summary_df(fb).assign(ms_level=fb.ms_level).to_csv(
            out_dir / f"runs_{fb.ms_level}.csv", index=False
        )
        (out_dir / f"file_meta_{fb.ms_level}.json").write_text(
            json.dumps(
                {k: v for k, v in asdict(fb).items() if k != "runs"},
                indent=2,
            ),
            encoding="utf-8",
        )

    cross = None
    if len(fbs) == 2:
        print("\n=== Cross-file comparison ===")
        cross = compare_two(fbs[0], fbs[1])
        (out_dir / "cross_file.json").write_text(json.dumps(cross, indent=2, default=str),
                                                 encoding="utf-8")

    render_markdown_report(fbs, cross, out_dir / "report.md")
    print(f"\nDone. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
