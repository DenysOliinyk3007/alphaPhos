"""Prototype: 'top-N' precursor attribution fix.

Diagnosis (from investigate_right_tail.py + manual tracing):
  Spectronaut over-exports a peptide as N candidate-position rows (one per
  potential phospho site, all with similar quants for the same chromatographic
  measurement). Spectronaut's PTM consolidate dedups by selecting only the
  TOP-N positions by per-row loc probability and assigning the peptide there.

  alphaPhos PeptideCollapse_v4 currently parses position from EG.PrecursorId
  (i.e. trusts each candidate-position row independently), so it sums every
  candidate row at its own position -> over-attribution to non-top sites.

Fix tested here:
  filter_to_top_n_positions(df) keeps only the precursor rows whose PrecId
  position set EQUALS the top-N positions by EG.PTMLocalizationProbabilities.
  For each (peptide, charge, sample) group this typically reduces ~10-20
  candidate-position rows to ONE canonical row.

Validation: re-run compare_to_spectronaut_native.py logic on filtered data and
check whether the right-tail anomaly collapses.
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

OUT_DIR = Path(__file__).parent / "prototype_top_n_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET = "d:/Projects/alphaPhos/test_data/nanoPhos_dilser_noEGF_1000ng_Report_ms1_ms2.parquet"
NATIVE = "Z:/Denys_nanoPhos/PRIDE/analysis_data/revision/figure2/20260506_113358_nanoPhos_dilser_noEGF_1000ng_Report.tsv"

KEY_PATTERN = re.compile(
    r"^(?P<prot>[^~]+)~(?P<gene>[^_]*)_(?P<aa>[A-Z])(?P<pos>\d+)_M(?P<mult>\d+)$"
)


def parse_loc_dict(s: str | float) -> dict[int, float]:
    """Parse EG.PTMLocalizationProbabilities -> {peptide_position_1indexed: prob}."""
    if pd.isna(s) or not isinstance(s, str):
        return {}
    s = s.strip("_.* ")
    out: dict[int, float] = {}
    pos = 0
    i = 0
    while i < len(s):
        if s[i] == "[":
            try:
                end = s.index("]", i)
            except ValueError:
                break
            bracket = s[i + 1 : end]
            if bracket.startswith("Phospho (STY)"):
                m = re.search(r":\s*([\d.]+)%", bracket)
                if m:
                    out[pos] = float(m.group(1)) / 100.0
            i = end + 1
        elif s[i].isalpha():
            pos += 1
            i += 1
        else:
            i += 1
    return out


def parse_precid_phospho_positions(precid: str) -> tuple[int, ...]:
    """Return the set of phospho positions (1-indexed) encoded in EG.PrecursorId."""
    if pd.isna(precid):
        return ()
    # Strip non-phospho brackets so the only brackets left are [Phospho (STY)]
    only_phos = re.sub(r"\[(?!Phospho \(STY\))[^\]]*\]", "", precid)
    # Strip wrapper underscores and trailing .charge
    only_phos = only_phos.strip("_*. ")
    # Re-strip trailing .digit (charge state) if present after a '.'
    only_phos = re.sub(r"\.\d+$", "", only_phos)
    parts = only_phos.split("[Phospho (STY)]")
    if len(parts) < 2:
        return ()
    positions: list[int] = []
    cur = 0
    for seg in parts[:-1]:
        cur += len(seg)
        positions.append(cur)
    return tuple(sorted(positions))


def filter_to_top_n_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only precursor rows whose PrecId positions equal top-N by loc prob.

    This deduplicates Spectronaut's per-candidate-position over-export by
    keeping only the canonical row (whose PrecId already encodes the
    rank-selected positions).
    """
    df = df.copy()
    loc_dicts = df["EG.PTMLocalizationProbabilities"].map(parse_loc_dict)
    precid_pos = df["EG.PrecursorId"].map(parse_precid_phospho_positions)
    df["_loc_dict"] = loc_dicts.values
    df["_precid_pos"] = precid_pos.values
    df["_phospho_count"] = df["_precid_pos"].map(len)

    def is_top_n(row) -> bool:
        n = row["_phospho_count"]
        loc = row["_loc_dict"]
        if n == 0 or not loc:
            # Keep non-phospho rows untouched (downstream filters phospho-only anyway)
            return True
        # Top-N positions by descending prob, tie-break by lower position (matches R `rank(ties.method='first')`)
        ranked = sorted(loc.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
        top_n = tuple(sorted(p for p, _ in ranked))
        return tuple(row["_precid_pos"]) == top_n

    print("  Computing top-N attribution mask (this is the slow row-wise step)...")
    t0 = time.time()
    keep = df.apply(is_top_n, axis=1)
    print(f"    elapsed {time.time()-t0:.1f}s")
    print(f"  Rows: {len(df)} -> {int(keep.sum())} after top-N filter "
          f"({keep.mean()*100:.1f}% retained)")

    return df.loc[keep].drop(columns=["_loc_dict", "_precid_pos", "_phospho_count"]).reset_index(drop=True)


# === Reuse comparison logic from compare_to_spectronaut_native.py ===

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
    out["native_quant"] = pd.to_numeric(
        out["quant_raw"].replace("Filtered", np.nan), errors="coerce"
    )
    out["mult_capped"] = out["PTM.Multiplicity"].clip(upper=3).astype(int)
    out["match_key"] = list(zip(
        out["PTM.ProteinId"], out["PTM.SiteAA"],
        out["PTM.SiteLocation"].astype(int), out["mult_capped"],
    ))
    return out[["match_key", "sample", "native_quant"]]


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


def run_pipeline(df_raw: pd.DataFrame, label: str) -> dict:
    print(f"\n--- {label} ---")
    pc = PeptideCollapse(verbose=False)
    t0 = time.time()
    sites = pc.process_complete_pipeline(
        df_raw, cutoff=0.75, collapse_level="P",
        aggregation_method="consolidate",
        add_kinase_sequences=False, noise_floor_filter=True,
        localization_strategy="global_max",
    )
    elapsed = time.time() - t0
    print(f"  {len(sites)} sites, {elapsed:.1f}s")
    sample_cols = [c for c in sites.columns if isinstance(c, str) and "nanoPhos_dilser" in c]
    long = alpha_to_long(sites, sample_cols)
    return {"sites": sites, "long": long}


def summarize_diff(both: pd.DataFrame, label: str) -> dict:
    d = both["log2_diff"]
    return {
        "config": label,
        "n_paired_non_nan": int(len(both)),
        "mean": round(float(d.mean()), 4),
        "median": round(float(d.median()), 4),
        "sd": round(float(d.std()), 4),
        "p05": round(float(d.quantile(0.05)), 4),
        "p95": round(float(d.quantile(0.95)), 4),
        "pct_within_0.1": round(float((d.abs() < 0.1).mean() * 100), 2),
        "pct_within_0.5": round(float((d.abs() < 0.5).mean() * 100), 2),
        "pct_within_1.0": round(float((d.abs() < 1.0).mean() * 100), 2),
        "pct_alpha_higher_1_log2": round(float((d > 1.0).mean() * 100), 2),
    }


def main() -> None:
    print("=== Load native ===")
    native = load_native_long(NATIVE)
    print(f"  native paired-long rows: {len(native):,}")
    print(f"  unique native sites: {native['match_key'].nunique():,}")

    print("\n=== Load alphaPhos raw (MS2) ===")
    df_raw = read_spectronaut(PARQUET, quant_level="MS2",
                              drop_decoys=True, pg_qvalue_max=0.01)
    print(f"  raw rows: {len(df_raw):,}")

    # Baseline (no fix)
    base = run_pipeline(df_raw, "BASELINE — no fix")

    # Apply the top-N attribution filter
    print("\n=== Applying top-N attribution filter ===")
    df_fixed = filter_to_top_n_positions(df_raw)
    fix = run_pipeline(df_fixed, "FIXED — top-N attribution")

    # Compare both vs native
    summaries = []
    for label, payload in [("baseline", base), ("top_n_fix", fix)]:
        merged = native.merge(payload["long"], on=["match_key", "sample"], how="inner")
        both = merged.dropna(subset=["native_quant", "alpha_log2"]).copy()
        both["native_log2"] = np.log2(both["native_quant"].replace(0, np.nan))
        both = both.dropna(subset=["native_log2"])
        both["log2_diff"] = both["alpha_log2"] - both["native_log2"]
        summaries.append(summarize_diff(both, label))

    print("\n\n========== Verdict (log2 alpha - native) ==========")
    print(pd.DataFrame(summaries).to_string(index=False))

    # Persist
    pd.DataFrame(summaries).to_csv(OUT_DIR / "diff_summary.csv", index=False)
    (OUT_DIR / "summary.json").write_text(
        json.dumps({"summaries": summaries}, indent=2), encoding="utf-8"
    )
    print(f"\nArtifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
