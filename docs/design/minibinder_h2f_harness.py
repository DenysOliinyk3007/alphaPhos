"""End-to-end miniBinder H2F pipeline test: 12 configurations -> CurveCurator.

3 filters x 2 MS levels x baseline/fix = 12 runs. For each, runs the full
alphaPhos pipeline (PeptideCollapse + optional top-N attribution + optional
condition-aware mask) on the H2F subset, generates a replicate-level
CurveCurator input + config, invokes CurveCurator, and records the resulting
number of significantly regulated phosphosites + global FDR.

Output structure (under docs/design/minibinder_h2f_output/):
    runs/<strategy>_<ms_level>_<base|fix>/
        input.tsv      <- one row per site, Raw <expt> columns for each sample
        config.toml    <- CurveCurator config
        curves.txt     <- CurveCurator output
        decoys.txt
        fdr.txt
        dashboard.html
        summary.json   <- per-run stats
    aggregate.csv      <- 12 rows, one per config
    aggregate.md       <- human-readable comparison
"""

from __future__ import annotations

import json
import re
import sys
import subprocess
import time
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "d:/Projects/miniBinders/scripts")  # PeptideCollapse_v4 + condition_aware_classI live here

from alphaphos.io.spectronaut import _normalize_column_names, _coerce_dtypes  # noqa: E402
from alphaphos.preprocess import (  # noqa: E402
    filter_to_top_n_positions,
    parse_loc_dict,
    parse_precid_phospho_positions,
    top_n_positions,
)
from PeptideCollapse_v4 import PeptideCollapse  # noqa: E402
from condition_aware_classI import apply_condition_aware_classI_mask  # noqa: E402


def compute_top_n_mask_inline(df: pd.DataFrame) -> np.ndarray:
    """Compute the top-N attribution mask DIRECTLY as a boolean array aligned
    with df's positions (not its index). Avoids the index-reset bug from
    filter_to_top_n_positions returning a reset index.
    """
    loc_dicts = df["EG.PTMLocalizationProbabilities"].map(parse_loc_dict).to_list()
    precid_positions = df["EG.PrecursorId"].map(parse_precid_phospho_positions).to_list()

    def _is_top_n(loc, pp) -> bool:
        n = len(pp)
        if n == 0:
            return True
        if not loc:
            return False
        return top_n_positions(loc, n) == pp

    return np.array(
        [_is_top_n(loc, pp) for loc, pp in zip(loc_dicts, precid_positions)],
        dtype=bool,
    )

PARQUET = Path("D:/Projects/alphaPhos/docs/design/minibinder_data/minibinder_full.parquet")
COLNAMES = Path("D:/Projects/miniBinders/colnames_upd.csv")
OUT_ROOT = Path(__file__).parent / "minibinder_h2f_output"
RUNS_DIR = OUT_ROOT / "runs"
CC_EXE = Path("D:/Projects/miniBinders/CurveCuratorEnv/Scripts/CurveCurator.exe")

CONFIGS = [
    {"strategy": s, "ms_level": m, "use_top_n": t}
    for s in ["global_max", "per_run", "condition_aware"]
    for m in ["MS1", "MS2"]
    for t in [False, True]
]


# ============================================================================
# Helpers
# ============================================================================

def normalize_spectronaut(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same column normalization the alphaphos reader does."""
    df = _normalize_column_names(df)
    df = _coerce_dtypes(df)
    return df


def swap_quant(df: pd.DataFrame, ms_level: str) -> pd.DataFrame:
    """Replace 'EG.TotalQuantity (Settings)' with the chosen FG.{MS_LEVEL}Quantity."""
    src_col = f"FG.{ms_level}Quantity"
    if src_col not in df.columns:
        raise KeyError(f"Missing column {src_col!r}; have: {sorted(df.columns)[:30]}...")
    out = df.copy()
    out["EG.TotalQuantity (Settings)"] = out[src_col]
    return out


def build_sample_to_condition(sample_names: list[str]) -> dict[str, str]:
    """For H2F samples like 'H2F_NT_1', 'H2F_512nM_3' -> 'H2F_NT', 'H2F_512nM'."""
    pat = re.compile(r"^(H2F)_(NT|\d+nM)_\d+$")
    s2c = {}
    for s in sample_names:
        m = pat.match(s)
        if m:
            s2c[s] = f"{m.group(1)}_{m.group(2)}"
    return s2c


def write_curve_curator_input(
    sites_df: pd.DataFrame, sample_cols: list[str], out_tsv: Path,
) -> tuple[list[str], list[float]]:
    """Replicate-level input: one row per site, one column per sample.

    Each sample column is named 'Raw <sample_name>'. Returns the sorted list
    of (experiment_name, dose_nM) needed for the TOML.
    """
    def parse_sample(s: str) -> tuple[float, int, str]:
        m = re.match(r"^H2F_(NT|\d+nM)_(\d+)$", s)
        if not m:
            return (1e9, 9999, s)  # sort unparseable to end
        dose = m.group(1)
        rep = int(m.group(2))
        dose_nM = 0.0 if dose == "NT" else float(dose.replace("nM", ""))
        return (dose_nM, rep, s)

    triples = sorted([parse_sample(s) for s in sample_cols], key=lambda t: (t[0], t[1]))
    ordered_samples = [t[2] for t in triples]
    doses = [t[0] for t in triples]

    # Build the DataFrame: Name + Raw <expt> per sample
    rows = sites_df[["PTM_Collapse_key"] + ordered_samples].copy()
    rows = rows.rename(columns={"PTM_Collapse_key": "Name"})
    raw_cols = {s: f"Raw {s}" for s in ordered_samples}
    rows = rows.rename(columns=raw_cols)
    rows.to_csv(out_tsv, sep="\t", index=False, na_rep="")
    return ordered_samples, doses


def write_curve_curator_config(
    out_toml: Path, input_tsv: Path, run_dir: Path,
    experiments: list[str], doses: list[float], strategy_label: str,
) -> None:
    nt_experiments = [e for e, d in zip(experiments, doses) if d == 0.0]
    def _ascii(s: str) -> str:
        return s.replace("\\", "/")
    cfg = f'''[Meta]
id = "H2F_{strategy_label}"
condition = "H2F_minibinder"
description = "miniBinder H2F dose response - {strategy_label}"
treatment_time = "15 min"

[Experiment]
experiments = {experiments!r}
doses = {doses!r}
dose_scale = 1e-09
dose_unit = "M"
control_experiment = {nt_experiments!r}
measurement_type = "OTHER"
data_type = "OTHER"
search_engine = "OTHER"

[Paths]
input_file = "{_ascii(str(input_tsv))}"
curves_file = "{_ascii(str(run_dir / "curves.txt"))}"
decoys_file = "{_ascii(str(run_dir / "decoys.txt"))}"
fdr_file = "{_ascii(str(run_dir / "fdr.txt"))}"
mad_file = "{_ascii(str(run_dir / "mad.txt"))}"
dashboard = "{_ascii(str(run_dir / "dashboard.html"))}"

[Processing]
available_cores = 5
imputation = false
normalization = true
max_missing = 12

["Curve Fit"]
type = "OLS"
speed = "standard"
control_fold_change = true

["F Statistic"]
alpha = 0.05
fc_lim = 0.45
pEC50_filter = [6.0, 10.0]

[Dashboard]
backend = "svg"
'''
    out_toml.write_text(cfg, encoding="utf-8")


def run_curve_curator(config_toml: Path) -> dict:
    """Invoke CurveCurator.exe with -f flag for FDR estimation."""
    t = time.time()
    result = subprocess.run(
        [str(CC_EXE), "-f", str(config_toml)],
        capture_output=True, text=True, check=False, timeout=900,
    )
    elapsed = time.time() - t
    out = {
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 1),
        "stdout_tail": "\n".join(result.stdout.strip().splitlines()[-15:]),
        "stderr_tail": "\n".join(result.stderr.strip().splitlines()[-15:]),
    }
    return out


def parse_cc_output(run_dir: Path) -> dict:
    """Parse curves.txt + fdr.txt to extract significance counts."""
    out = {
        "n_sites_total": None, "n_up": None, "n_down": None,
        "n_not": None, "n_regulated": None,
        "global_fdr": None, "filtered_fdr": None,
        "median_pEC50_regulated": None,
        "median_R2_regulated": None,
    }
    curves_path = run_dir / "curves.txt"
    fdr_path = run_dir / "fdr.txt"
    if curves_path.exists():
        curves = pd.read_csv(curves_path, sep="\t")
        out["n_sites_total"] = int(len(curves))
        reg = curves["Curve Regulation"].value_counts().to_dict()
        out["n_up"] = int(reg.get("up", 0))
        out["n_down"] = int(reg.get("down", 0))
        out["n_not"] = int(reg.get("not", 0))
        out["n_regulated"] = out["n_up"] + out["n_down"]
        sig = curves[curves["Curve Regulation"].isin(["up", "down"])]
        if len(sig):
            out["median_pEC50_regulated"] = round(float(sig["pEC50"].median()), 3)
            out["median_R2_regulated"] = round(float(sig["Curve R2"].median()), 3)
    if fdr_path.exists():
        text = fdr_path.read_text().strip()
        for line in text.splitlines():
            if "Global FDR" in line:
                out["global_fdr"] = float(line.split(":")[1].strip())
            elif "Filtered FDR" in line:
                out["filtered_fdr"] = float(line.split(":")[1].strip())
    return out


# ============================================================================
# Pipeline (one config)
# ============================================================================

def run_one_config(
    df_base: pd.DataFrame, top_n_mask: pd.Series, cfg: dict[str, Any],
    colnames: pd.DataFrame,
) -> dict:
    label = f"{cfg['strategy']}_{cfg['ms_level']}_{'fix' if cfg['use_top_n'] else 'base'}"
    run_dir = RUNS_DIR / label
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}\n[{label}]\n{'='*72}")

    # 1. Apply top-N mask if needed (positional indexing, not label-based)
    t0 = time.time()
    if cfg["use_top_n"]:
        df = df_base.iloc[top_n_mask].reset_index(drop=True)
    else:
        df = df_base
    print(f"  rows after top-N filter: {len(df):,} ({len(df)/len(df_base)*100:.1f}%)")

    # 2. Swap quant column to chosen MS level
    df_q = swap_quant(df, cfg["ms_level"])

    # 3. PeptideCollapse
    pc_strategy = "global_max" if cfg["strategy"] == "condition_aware" else cfg["strategy"]
    pc_cutoff = 0.0 if cfg["strategy"] == "condition_aware" else 0.75
    print(f"  PeptideCollapse (strategy={pc_strategy}, cutoff={pc_cutoff}) ...")
    t = time.time()
    pc = PeptideCollapse(verbose=False)
    sites = pc.process_complete_pipeline(
        df_q,
        cutoff=pc_cutoff,
        collapse_level="PG",
        aggregation_method="median",
        localization_strategy=pc_strategy,
        noise_floor_filter=True,
        add_kinase_sequences=False,
    )
    loc_per_run = pc.site_localization_per_run
    elapsed_collapse = time.time() - t
    print(f"    -> {len(sites):,} sites in {elapsed_collapse:.1f}s")

    # 4. Rename columns
    rename = dict(zip(colnames["Old_cols"], colnames["New_cols"]))
    sites = sites.rename(columns=rename)
    loc_per_run = loc_per_run.rename(columns=rename)

    # 5. condition_aware mask if applicable
    sample_cols_all = [c for c in sites.columns if isinstance(c, str)
                       and re.match(r"^(TAB2|H2F)_(NT|\d+nM)_\d+$", c)]
    if cfg["strategy"] == "condition_aware":
        s2c = {c: _condition_of(c) for c in sample_cols_all if _condition_of(c) is not None}
        print(f"  condition_aware mask ({len(s2c)} samples, "
              f"{len(set(s2c.values()))} conditions) ...")
        sites, _ = apply_condition_aware_classI_mask(
            sites, loc_per_run, s2c,
            classI_cutoff=0.75, condition_threshold=0.50,
            drop_all_nan=True, return_decision_table=True,
        )
        print(f"    -> {len(sites):,} sites after mask")

    # 6. Filter to H2F sample columns only
    h2f_cols = [c for c in sites.columns if isinstance(c, str) and c.startswith("H2F_")]
    print(f"  H2F sample cols: {len(h2f_cols)}")

    # Drop sites with all-NaN across H2F samples
    block = sites[h2f_cols]
    keep = block.notna().any(axis=1)
    sites_h2f = sites.loc[keep].reset_index(drop=True)
    print(f"  H2F sites with at least one quant: {len(sites_h2f):,}")

    # 7. Write CurveCurator input + config
    input_tsv = run_dir / "input.tsv"
    experiments, doses = write_curve_curator_input(sites_h2f, h2f_cols, input_tsv)
    config_toml = run_dir / "config.toml"
    write_curve_curator_config(config_toml, input_tsv, run_dir,
                                experiments, doses, label)
    print(f"  wrote input.tsv ({input_tsv.stat().st_size/1024:.0f} KB), config.toml")

    # 8. Run CurveCurator
    print("  running CurveCurator -f ...")
    cc_run = run_curve_curator(config_toml)
    print(f"    rc={cc_run['returncode']}, {cc_run['elapsed_s']}s")
    if cc_run["returncode"] != 0:
        print(f"    stderr_tail: {cc_run['stderr_tail']}")

    # 9. Parse outputs
    parsed = parse_cc_output(run_dir)
    print(f"  -> total sites: {parsed['n_sites_total']}, "
          f"regulated: {parsed['n_regulated']} "
          f"({parsed['n_up']} up + {parsed['n_down']} down), "
          f"global FDR: {parsed['global_fdr']}")

    summary = {
        **cfg,
        "label": label,
        "n_alphaphos_sites_pre_h2f": int(len(sites)),
        "n_alphaphos_sites_h2f": int(len(sites_h2f)),
        "elapsed_collapse_s": round(elapsed_collapse, 1),
        "elapsed_total_s": round(time.time() - t0, 1),
        "curve_curator": {
            "returncode": cc_run["returncode"],
            "elapsed_s": cc_run["elapsed_s"],
        },
        **parsed,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                          encoding="utf-8")
    return summary


def _condition_of(s: str) -> str | None:
    m = re.match(r"^(TAB2|H2F)_(NT|\d+nM)_\d+$", s)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}"


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_ROOT}\n")

    print("[1/4] Reading parquet...")
    t = time.time()
    df = pd.read_parquet(PARQUET)
    print(f"  {len(df):,} rows in {time.time()-t:.1f}s")

    print("\n[2/4] Normalizing columns (dot/underscore unification)...")
    df = normalize_spectronaut(df)
    print(f"  columns: {len(df.columns)}")

    # Sanity: ensure required columns are present
    needed = ["EG.PrecursorId", "EG.PTMLocalizationProbabilities", "FG.MS1Quantity",
              "FG.MS2Quantity", "R.FileName", "PG.Genes", "PG.ProteinGroups",
              "PEP.PeptidePosition", "EG.PTMAssayProbability"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")

    # Set canonical quant column for PeptideCollapse (will be swapped per-run)
    df["EG.TotalQuantity (Settings)"] = df["FG.MS2Quantity"]  # default

    # PG q-value filter (PeptideCollapse caller-side filter normally)
    if "PG.Qvalue" in df.columns:
        before = len(df)
        df = df[df["PG.Qvalue"].fillna(0) <= 0.01].reset_index(drop=True)
        print(f"  PG.Qvalue<=0.01: {len(df):,} rows ({len(df)/before*100:.1f}% kept)")

    print("\n[3/4] Computing top-N attribution mask (one-time, ~1-2 min for 12M rows)...")
    t = time.time()
    top_n_mask = compute_top_n_mask_inline(df)
    print(f"  retained {top_n_mask.sum():,} / {len(df):,} rows "
          f"({top_n_mask.sum()/len(df)*100:.1f}%) in {time.time()-t:.1f}s")

    print("\n[4/4] Loading colnames + starting 12-config loop...")
    colnames = pd.read_csv(COLNAMES)

    results = []
    for i, cfg in enumerate(CONFIGS, 1):
        print(f"\n\n#### Config {i}/{len(CONFIGS)} ####")
        label = f"{cfg['strategy']}_{cfg['ms_level']}_{'fix' if cfg['use_top_n'] else 'base'}"
        existing = RUNS_DIR / label / "summary.json"
        if existing.exists():
            print(f"  [skip] {label} already has summary.json")
            try:
                summary = json.loads(existing.read_text(encoding="utf-8"))
                results.append(summary)
            except Exception:
                print(f"  (could not parse existing summary; will re-run)")
                existing.unlink()
                summary = run_one_config(df, top_n_mask, cfg, colnames)
                results.append(summary)
            continue
        try:
            summary = run_one_config(df, top_n_mask, cfg, colnames)
            results.append(summary)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"  !! FAILED: {e}\n{tb}")
            results.append({**cfg, "label": "FAILED", "error": str(e), "traceback": tb})
        # Checkpoint after every config
        (OUT_ROOT / "aggregate.json").write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )

    # Build aggregate CSV + Markdown
    rows = []
    for r in results:
        if "error" in r:
            rows.append({"label": r.get("label", "?"), **r})
            continue
        rows.append({
            "label": r["label"],
            "strategy": r["strategy"],
            "ms_level": r["ms_level"],
            "use_top_n": r["use_top_n"],
            "n_alphaphos_sites_h2f": r["n_alphaphos_sites_h2f"],
            "n_sites_total": r.get("n_sites_total"),
            "n_regulated": r.get("n_regulated"),
            "n_up": r.get("n_up"),
            "n_down": r.get("n_down"),
            "global_fdr": r.get("global_fdr"),
            "median_pEC50_reg": r.get("median_pEC50_regulated"),
            "median_R2_reg": r.get("median_R2_regulated"),
            "elapsed_collapse_s": r.get("elapsed_collapse_s"),
            "elapsed_total_s": r.get("elapsed_total_s"),
        })
    agg = pd.DataFrame(rows)
    agg.to_csv(OUT_ROOT / "aggregate.csv", index=False)
    print("\n\n" + "=" * 72)
    print("AGGREGATE")
    print("=" * 72)
    print(agg.to_string(index=False))
    md = ["# miniBinder H2F pipeline test\n",
          f"_run: {time.strftime('%Y-%m-%d %H:%M:%S')}_\n",
          "## Per-config results\n",
          agg.to_markdown(index=False), ""]
    (OUT_ROOT / "aggregate.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {OUT_ROOT / 'aggregate.csv'} and aggregate.md")


if __name__ == "__main__":
    main()
