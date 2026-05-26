"""Dublin IgG vs PD1 — biological reproducibility under baseline vs top-N-fixed.

Replicates Dublin's processing.ipynb pipeline end-to-end on the same data,
running it twice:

  BASELINE: no top-N filter (matches the as-published behavior).
  FIXED:    apply top-N filter immediately after Spectronaut read.

Then runs the same downstream pipeline for both:
  PeptideCollapse_v4 (cutoff=0, global_max, median)
  -> condition-aware Class I mask
  -> filter_phosphosites (condition presence-based, cutoff 0.7)
  -> impute_phosphosites (KNN)
  -> limma per batch (e112 IgG vs PD1, e115 IgG vs PD1)

Compares:
  - n significant sites per batch per pipeline
  - Jaccard overlap of significant sites
  - logFC concordance on shared significants (Pearson r)
  - Top-K hits

This answers: does the top-N fix change BIOLOGICAL conclusions, not just numbers?
If overlap is >90% and logFC r > 0.95, the fix is non-disruptive for the
published Dublin analysis (just more honest absolute intensities).
If overlap < 70%, the bias mattered and analyses need re-running.
"""

from __future__ import annotations

import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "d:/Projects/Dublin/testscripts/src")

from alphaphos.io import read_spectronaut  # noqa: E402
from alphaphos.preprocess import filter_to_top_n_positions  # noqa: E402
from PeptideCollapse_v4 import PeptideCollapse  # noqa: E402
from condition_aware_classI import apply_condition_aware_classI_mask  # noqa: E402
from core import filter_phosphosites, impute_phosphosites  # noqa: E402

OUT_DIR = Path(__file__).parent / "dublin_repro_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DUBLIN_TSV = "d:/Projects/Dublin/testscripts/raw_data/20260516_092715_e112e115_test_nanoPhos_Report.tsv"

# Reproduce the notebook's exact configuration
CUTOFF = 0.0   # no early filter; condition_aware does the work
AGG = "median"
STRATEGY = "global_max"  # the notebook uses this so condition_aware can run downstream
CLASS_I_CUTOFF = 0.75
CONDITION_THRESHOLD = 0.50
PRESENCE_CUTOFF = 0.7   # filter_phosphosites condition-presence cutoff
LIMMA_P = 0.05
LIMMA_LFC = 0.585


def deduce_condition(fname: str) -> str:
    exp = re.search(r"(e11[25])", fname, flags=re.IGNORECASE)
    treat = re.search(r"(IgG|PD1)", fname, flags=re.IGNORECASE)
    if exp is None or treat is None:
        return "UNKNOWN"
    return f"{exp.group(1).lower()}_{treat.group(1)}"


def run_pipeline_to_imputed(df_raw: pd.DataFrame, sample_to_cond: dict, label: str) -> pd.DataFrame:
    """Runs the full pre-limma pipeline: collapse -> classI mask -> presence filter -> KNN impute.

    Returns a wide DataFrame (rows = samples, columns = sites + 'condition').
    """
    print(f"  [{label}] PeptideCollapse...")
    t = time.time()
    pc = PeptideCollapse(verbose=False)
    sites = pc.process_complete_pipeline(
        df_raw,
        cutoff=CUTOFF,
        collapse_level="PG",
        aggregation_method=AGG,
        add_kinase_sequences=False,
        noise_floor_filter=True,
        localization_strategy=STRATEGY,
    )
    loc = pc.site_localization_per_run
    print(f"    -> {len(sites)} pre-classify sites ({time.time()-t:.1f}s)")

    print(f"  [{label}] condition-aware Class I mask...")
    sites_classI, _ = apply_condition_aware_classI_mask(
        sites, loc, sample_to_cond,
        classI_cutoff=CLASS_I_CUTOFF,
        condition_threshold=CONDITION_THRESHOLD,
        drop_all_nan=True,
        return_decision_table=True,
    )
    print(f"    -> {len(sites_classI)} post-classI sites")

    print(f"  [{label}] reshape + presence filter (cutoff={PRESENCE_CUTOFF})...")
    META = {"PTM_Collapse_key", "UPD_seq", "PTM_localization", "Protein_group",
            "Gene_group", "kinase_sequence", "Protein_Collapse_key",
            "PG.Genes", "PG.ProteinGroups"}
    sample_cols = [c for c in sites_classI.columns if c not in META]
    wide = (sites_classI.set_index("PTM_Collapse_key")[sample_cols].T)
    wide.index.name = "R.FileName"
    wide["condition"] = wide.index.map(sample_to_cond)

    filtered = filter_phosphosites(wide, how="condition", cutoff=PRESENCE_CUTOFF,
                                    condition_col="condition")
    n_sites_filt = sum("~" in str(c) for c in filtered.columns)
    print(f"    -> {n_sites_filt} sites after presence filter")

    print(f"  [{label}] KNN impute...")
    imputed = impute_phosphosites(filtered)
    site_cols = [c for c in imputed.columns if "~" in str(c)]
    print(f"    -> {len(site_cols)} sites in imputed matrix")
    return imputed


RSCRIPT = r"C:\Program Files\R\R-4.5.2\bin\Rscript.exe"
LIMMA_WRAPPER = Path(__file__).parent / "limma_wrapper.R"


def _call_limma(expr_b: pd.DataFrame, meta_b: pd.DataFrame, label: str, batch: str) -> pd.DataFrame:
    """Run limma via Rscript subprocess. Returns a DataFrame with the topTable output."""
    import subprocess
    expr_path = OUT_DIR / f"_expr_{label}_{batch}.tsv"
    meta_path = OUT_DIR / f"_meta_{label}_{batch}.tsv"
    out_path = OUT_DIR / f"_limma_out_{label}_{batch}.tsv"
    # Write expr with 'feature' index name
    expr_out = expr_b.copy()
    expr_out.index.name = "feature"
    expr_out.to_csv(expr_path, sep="\t", na_rep="NA")
    meta_b.to_csv(meta_path, sep="\t", index=False)
    result = subprocess.run(
        [RSCRIPT, str(LIMMA_WRAPPER),
         str(expr_path), str(meta_path), "treatment", str(out_path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(f"    [R stderr]\n{result.stderr}")
        raise RuntimeError(f"limma failed for {label}/{batch}")
    res = pd.read_csv(out_path, sep="\t")
    res["significant"] = (res["adj.P.Val"] < LIMMA_P) & (res["logFC"].abs() > LIMMA_LFC)
    return res


def run_limma_per_batch(imputed: pd.DataFrame, label: str) -> dict[str, pd.DataFrame]:
    """Run limma IgG-vs-PD1 separately per batch (e112, e115).
    Uses Rscript subprocess wrapper (limma_wrapper.R) — rpy2 is broken on this
    Windows setup (R not built as library), so we shell out.
    """
    meta_full = pd.DataFrame({
        "sample_id": imputed.index,
        "condition": imputed["condition"].values,
    })
    meta_full[["batch", "treatment"]] = meta_full["condition"].str.split("_", expand=True)
    site_cols = [c for c in imputed.columns if "~" in str(c)]
    expr_full = imputed[site_cols].T
    expr_full.columns = imputed.index

    results = {}
    for batch in ["e112", "e115"]:
        print(f"  [{label}] {batch} limma...")
        meta_b = meta_full[meta_full["batch"] == batch].reset_index(drop=True)
        expr_b = expr_full[meta_b["sample_id"].values]
        try:
            res = _call_limma(expr_b, meta_b, label, batch)
        except Exception as e:
            print(f"    limma subprocess failed ({e}); using Welch fallback")
            res = _welch_test(expr_b, meta_b)
        n_sig = int(res["significant"].sum())
        n_p05 = int((res["P.Value"] < 0.05).sum())
        print(f"    -> {n_sig} sites at adj.P<{LIMMA_P} & |lfc|>{LIMMA_LFC}, "
              f"{n_p05} sites at raw P<0.05")
        results[batch] = res
    return results


def _welch_test(expr_b: pd.DataFrame, meta_b: pd.DataFrame) -> pd.DataFrame:
    """Per-site Welch's t-test fallback. expr_b: features x samples, meta_b has 'treatment'."""
    from scipy import stats
    igg = meta_b.loc[meta_b["treatment"] == "IgG", "sample_id"].tolist()
    pd1 = meta_b.loc[meta_b["treatment"] == "PD1", "sample_id"].tolist()
    rows = []
    for site, row in expr_b.iterrows():
        a = row.loc[igg].dropna().values
        b = row.loc[pd1].dropna().values
        if len(a) < 2 or len(b) < 2:
            continue
        try:
            t, p = stats.ttest_ind(b, a, equal_var=False, nan_policy="omit")
            lfc = float(np.nanmean(b) - np.nanmean(a))  # PD1 - IgG
        except Exception:
            continue
        rows.append({"feature": site, "logFC": lfc, "t": t, "P.Value": p})
    out = pd.DataFrame(rows)
    if len(out):
        # BH adjust
        m = len(out)
        out = out.sort_values("P.Value").reset_index(drop=True)
        out["rank"] = np.arange(1, m + 1)
        out["adj.P.Val"] = (out["P.Value"] * m / out["rank"]).clip(upper=1.0)
        # Enforce monotonicity (BH)
        out["adj.P.Val"] = out["adj.P.Val"][::-1].cummin()[::-1]
        out["significant"] = (out["adj.P.Val"] < LIMMA_P) & (out["logFC"].abs() > LIMMA_LFC)
        out = out.drop(columns=["rank"])
    else:
        out = pd.DataFrame(columns=["feature", "logFC", "P.Value", "adj.P.Val", "significant"])
    return out


def compare(base: dict, fix: dict) -> dict:
    """Per-batch comparison: significant-hit overlap + logFC concordance + top-K ranking."""
    summary = {}
    for batch in ["e112", "e115"]:
        b = base[batch]; f = fix[batch]
        b_sig = set(b.loc[b["significant"], "feature"])
        f_sig = set(f.loc[f["significant"], "feature"])
        intersect = b_sig & f_sig
        union = b_sig | f_sig
        jaccard = round(len(intersect) / len(union), 4) if union else 1.0

        m = b[["feature", "logFC", "adj.P.Val", "P.Value"]].merge(
            f[["feature", "logFC", "adj.P.Val", "P.Value"]],
            on="feature", suffixes=("_base", "_fix")
        )
        r_all = float(m["logFC_base"].corr(m["logFC_fix"])) if len(m) > 100 else None

        # Stratify logFC correlation by magnitude
        big = m[m["logFC_base"].abs() > 0.585]  # at least 1.5x change
        r_big = float(big["logFC_base"].corr(big["logFC_fix"])) if len(big) > 20 else None

        # Top-50 hits by P.Value per pipeline
        top50_b = set(b.sort_values("P.Value").head(50)["feature"])
        top50_f = set(f.sort_values("P.Value").head(50)["feature"])
        top50_intersect = top50_b & top50_f

        summary[batch] = {
            "n_features_baseline": int(len(b)),
            "n_features_fixed": int(len(f)),
            "n_significant_baseline": int(len(b_sig)),
            "n_significant_fixed": int(len(f_sig)),
            "n_shared_significant": int(len(intersect)),
            "n_baseline_only_sig": int(len(b_sig - f_sig)),
            "n_fixed_only_sig": int(len(f_sig - b_sig)),
            "jaccard_significant": jaccard,
            "n_shared_features": int(len(m)),
            "pearson_logfc_all_features": round(r_all, 4) if r_all is not None else None,
            "pearson_logfc_big_only": round(r_big, 4) if r_big is not None else None,
            "n_big_features": int(len(big)),
            "top50_overlap_count": int(len(top50_intersect)),
            "top50_overlap_pct": round(len(top50_intersect) / 50 * 100, 2),
        }
    return summary


def main() -> None:
    print(f"=== Dublin biological reproducibility test ===")
    print(f"  TSV: {DUBLIN_TSV}\n")

    print("[1/4] Loading Dublin data...")
    df_raw = read_spectronaut(DUBLIN_TSV, quant_level="auto", drop_decoys=True)
    print(f"  loaded {len(df_raw):,} rows")
    samples = sorted(df_raw["R.FileName"].unique())
    sample_to_cond = {s: deduce_condition(s) for s in samples}
    assert all(c != "UNKNOWN" for c in sample_to_cond.values())
    print(f"  samples: {len(samples)}, conditions: {set(sample_to_cond.values())}")

    print("\n[2a/4] BASELINE pipeline (no top-N fix)...")
    imputed_base = run_pipeline_to_imputed(df_raw, sample_to_cond, "BASE")

    print("\n[2b/4] FIXED pipeline (with top-N filter)...")
    df_fix = filter_to_top_n_positions(df_raw)
    print(f"  top-N retention: {len(df_fix)}/{len(df_raw)} = {len(df_fix)/len(df_raw)*100:.1f}%")
    imputed_fix = run_pipeline_to_imputed(df_fix, sample_to_cond, "FIX")

    print("\n[3/4] limma per batch (e112, e115) under both pipelines...")
    res_base = run_limma_per_batch(imputed_base, "BASE")
    res_fix = run_limma_per_batch(imputed_fix, "FIX")

    print("\n[4/4] Comparing significant hits...")
    summary = compare(res_base, res_fix)
    print()
    print(json.dumps(summary, indent=2))

    # Save artifacts
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for batch in ["e112", "e115"]:
        res_base[batch].to_csv(OUT_DIR / f"baseline_{batch}.csv", index=False)
        res_fix[batch].to_csv(OUT_DIR / f"fixed_{batch}.csv", index=False)
    print(f"\nArtifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
