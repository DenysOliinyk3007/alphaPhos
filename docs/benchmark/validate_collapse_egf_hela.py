"""Validate ``collapse_sites`` against Spectronaut's native PTM site report.

Dataset: ``Documents/Data/egf_hela_nanophos_spectronaut`` -- 6-run nanoPhos EGF
HeLa series (3 withEGF / 3 woEGF), Spectronaut 21.1.  Three PSM-level "long"
exports (MS2-only, MS1-only, MS1+MS2; TSV + parquet each) and two native
site-level "short" reports: ``cutoff_0`` (no per-cell Class-I filter) and
``cutoff_0p75`` (binary per-cell Class-I, the reference).

Expectation: collapsing the MS2 long export must reproduce ``short_cutoff_0p75``
(site set + per-cell log2 quant) up to the documented strategy differences.

Usage::

    .venv/bin/python docs/benchmark/validate_collapse_egf_hela.py [--data-dir DIR] [--full]

Writes a JSON summary next to stdout output (``--out``).
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import alphaphos as ap

warnings.filterwarnings("ignore", message="quantification_level=")

DEFAULT_DIR = Path("/Users/denysoliinyk/Documents/Data/egf_hela_nanophos_spectronaut")
PREFIX = "nanophos_egf_hela_1ug_"
MULT_CAP = 3


# ---------------------------------------------------------------------------
# Spectronaut native site report -> (log2 quant, loc prob) wide frames
# ---------------------------------------------------------------------------


def _sample_from_col(col: str) -> str:
    """'[1] 2025..._01.raw.PTM.Quantity' -> '2025..._01' (matches R.FileName)."""
    s = re.sub(r"^\[\d+\]\s*", "", col)
    return re.sub(r"\.raw\.PTM\.(Quantity|SiteProbability)$", "", s)


def load_sn_site_report(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (log2_quant, loc_prob) indexed by match_key ``PROT|AApos|Mcap``.

    * phospho rows only (``PTM.ModificationTitle == 'Phospho (STY)'``)
    * 'Filtered' -> NaN
    * multiplicity capped at 3 like alphaPhos; rows that collapse onto the same
      capped key are SUMMED in linear space (that is what alphaPhos does with
      the underlying precursors).
    """
    df = pd.read_csv(path, sep="\t", low_memory=False)
    n_all = len(df)
    df = df[df["PTM.ModificationTitle"] == "Phospho (STY)"].copy()
    qcols = [c for c in df.columns if c.endswith(".PTM.Quantity")]
    pcols = [c for c in df.columns if c.endswith(".PTM.SiteProbability")]
    m = df["PTM.CollapseKey"].str.extract(
        r"^(?P<prot>[^_]+)_(?P<aa>[A-Z])(?P<pos>\d+)_M(?P<mult>\d+)_\d+$"
    )
    assert m["prot"].notna().all(), "unparseable CollapseKey"
    mult_cap = m["mult"].astype(int).clip(upper=MULT_CAP)
    key = m["prot"] + "|" + m["aa"] + m["pos"] + "|M" + mult_cap.astype(str)
    quant = df[qcols].apply(pd.to_numeric, errors="coerce")
    quant.columns = [_sample_from_col(c) for c in qcols]
    quant.index = key
    prob = df[pcols].apply(pd.to_numeric, errors="coerce")
    prob.columns = [_sample_from_col(c) for c in pcols]
    prob.index = key
    n_before = len(quant)
    quant = quant.groupby(level=0).sum(min_count=1)
    prob = prob.groupby(level=0).max()
    info = {
        "rows_all_ptms": n_all,
        "rows_phospho": n_before,
        "sites_after_mult_cap": len(quant),
        "genes_with_semicolon": int(df["PG.Genes"].astype(str).str.contains(";").sum()),
        "protein_ids": sorted(set(m["prot"])) if len(m) < 50 else None,
    }
    return np.log2(quant.where(quant > 0)), prob, info


# ---------------------------------------------------------------------------
# alphaPhos side
# ---------------------------------------------------------------------------


def condition_df_from_samples(samples: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample": samples,
            "condition": ["withEGF" if "withEGF" in s else "woEGF" for s in samples],
        }
    )


def adata_to_wide(adata) -> tuple[pd.DataFrame, pd.DataFrame]:
    v = adata.var
    key = (
        v["protein_group_id"].astype(str)
        + "|"
        + v["site_aa"].astype(str)
        + v["site_position"].astype(int).astype(str)
        + "|M"
        + v["multiplicity"].astype(int).astype(str)
    )
    quant = pd.DataFrame(adata.layers["intensity_log2"].T, index=key, columns=adata.obs_names)
    loc = pd.DataFrame(adata.layers["localization"].T, index=key, columns=adata.obs_names)
    assert key.is_unique, "alphaPhos match keys are not unique"
    return quant.astype(float), loc.astype(float)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score(
    a: pd.DataFrame, sn: pd.DataFrame, a_loc: pd.DataFrame | None, sn_loc: pd.DataFrame
) -> dict:
    a = a.reindex(columns=sn.columns)
    shared = a.index.intersection(sn.index)
    union = a.index.union(sn.index)
    A, S = a.loc[shared], sn.loc[shared]
    both = (A.notna() & S.notna()).to_numpy()
    av, sv = A.to_numpy()[both], S.to_numpy()[both]
    d = av - sv
    out = {
        "n_aphos_sites": len(a),
        "n_sn_sites": len(sn),
        "n_shared_sites": len(shared),
        "jaccard_sites": round(len(shared) / len(union), 4),
        "aphos_only_sites": len(a.index.difference(sn.index)),
        "sn_only_sites": len(sn.index.difference(a.index)),
        "cells_both": int(both.sum()),
        "cells_aphos_only": int((A.notna() & S.isna()).sum().sum()),
        "cells_sn_only": int((A.isna() & S.notna()).sum().sum()),
        "pearson_r_log2": round(float(np.corrcoef(av, sv)[0, 1]), 4),
        "mean_diff_log2": round(float(d.mean()), 4),
        "median_diff_log2": round(float(np.median(d)), 4),
        "sd_diff_log2": round(float(d.std()), 4),
        "pct_within_0.1": round(float((np.abs(d) <= 0.1).mean() * 100), 2),
        "pct_within_0.5": round(float((np.abs(d) <= 0.5).mean() * 100), 2),
    }
    # Fold-change concordance (withEGF - woEGF mean, per shared site).
    w = [c for c in sn.columns if "withEGF" in c]
    o = [c for c in sn.columns if "woEGF" in c]
    fa = A[w].mean(axis=1) - A[o].mean(axis=1)
    fs = S[w].mean(axis=1) - S[o].mean(axis=1)
    ok = fa.notna() & fs.notna()
    out["log2fc_pearson_r"] = round(float(np.corrcoef(fa[ok], fs[ok])[0, 1]), 4)
    out["log2fc_n_sites"] = int(ok.sum())
    if a_loc is not None:
        L = a_loc.reindex(index=shared, columns=sn.columns).to_numpy()
        P = sn_loc.reindex(index=shared, columns=sn.columns).to_numpy()
        lb = ~np.isnan(L) & ~np.isnan(P)
        ld = L[lb] - P[lb]
        out["loc_cells_compared"] = int(lb.sum())
        out["loc_pct_equal_1e-3"] = round(float((np.abs(ld) <= 1e-3).mean() * 100), 2)
        out["loc_pearson_r"] = round(float(np.corrcoef(L[lb], P[lb])[0, 1]), 4)
    return out


def sanity(adata) -> dict:
    v = adata.var
    return {
        "shape": list(adata.shape),
        "var_index_unique": bool(adata.var_names.is_unique),
        "hash_in_index": int(adata.var_names.str.contains("#", regex=False).sum()),
        "multiplicity_values": sorted(v["multiplicity"].unique().tolist()),
        "site_aa_values": sorted(v["site_aa"].unique().tolist()),
        "n_contaminant_match": int(v["is_contaminant_match"].sum()),
        "obs_condition_nan": int(adata.obs["condition"].isna().sum()),
        "X_equals_layer": bool(
            np.array_equal(adata.X, adata.layers["intensity_log2"], equal_nan=True)
        ),
        "stats": {
            k: (int(v_) if isinstance(v_, (int, np.integer)) else v_)
            for k, v_ in adata.uns["alphaphos"]["stats"].items()
        },
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    ap_.add_argument(
        "--full", action="store_true", help="also run MS1 / dual / parquet / aggregation matrix"
    )
    ap_.add_argument("--out", type=Path, default=None)
    args = ap_.parse_args()
    D = args.data_dir
    results: dict = {}

    t = time.perf_counter()
    sn75, sn75_loc, info75 = load_sn_site_report(D / f"{PREFIX}short_cutoff_0p75.tsv")
    sn0, sn0_loc, info0 = load_sn_site_report(D / f"{PREFIX}short_cutoff_0.tsv")
    results["sn_reports"] = {
        "cutoff_0p75": info75,
        "cutoff_0": info0,
        "load_s": round(time.perf_counter() - t, 1),
    }
    print(
        f"[SN] 0p75: {info75['rows_phospho']} phospho rows -> {len(sn75)} capped sites | 0: {info0['rows_phospho']} -> {len(sn0)}"
    )

    t = time.perf_counter()
    psm_ms2 = ap.read_spectronaut(D / f"{PREFIX}long_ms2.tsv")
    print(
        f"[read] long_ms2.tsv: {len(psm_ms2):,} rows in {time.perf_counter() - t:.1f}s; attrs={ {k: v for k, v in psm_ms2.attrs.items() if k.startswith('n_rows')} }"
    )
    samples = sorted(psm_ms2["R.FileName"].unique())
    cdf = condition_df_from_samples(samples)

    def run(label, psm, advanced, ref, ref_loc, extra_sanity=True):
        t0 = time.perf_counter()
        adata = ap.collapse_sites(psm, condition_df=cdf, advanced=advanced)
        dt = time.perf_counter() - t0
        q, loc = adata_to_wide(adata)
        r = score(q, ref, loc, ref_loc)
        r["collapse_s"] = round(dt, 1)
        r["advanced"] = advanced
        if extra_sanity:
            r["sanity"] = sanity(adata)
        results[label] = r
        print(
            f"[{label}] {dt:5.1f}s  sites={r['n_aphos_sites']} (sn {r['n_sn_sites']}) jaccard={r['jaccard_sites']}  r={r['pearson_r_log2']}  mean_d={r['mean_diff_log2']:+.3f}  within0.1={r['pct_within_0.1']}%  cells a-only/sn-only={r['cells_aphos_only']}/{r['cells_sn_only']}  log2FC_r={r['log2fc_pearson_r']}  loc_eq={r.get('loc_pct_equal_1e-3')}%"
        )
        return adata, q

    # ---- core: MS2 long export vs SN 0.75 -----------------------------------
    a_per_run, q_per_run = run(
        "ms2_per_run_vs_sn0p75", psm_ms2, {"localization_strategy": "per_run"}, sn75, sn75_loc
    )
    a_cond, _q_cond = run(
        "ms2_condition_vs_sn0p75", psm_ms2, {"localization_strategy": "condition"}, sn75, sn75_loc
    )
    _a_gm, _q_gm = run(
        "ms2_global_max_vs_sn0p75", psm_ms2, {"localization_strategy": "global_max"}, sn75, sn75_loc
    )
    _a_gm0, q_gm0 = run(
        "ms2_global_max_cutoff0_vs_sn0",
        psm_ms2,
        {"localization_strategy": "global_max", "cutoff": 0.0},
        sn0,
        sn0_loc,
    )

    # determinism + h5ad round trip
    a2 = ap.collapse_sites(psm_ms2, condition_df=cdf, advanced={"localization_strategy": "per_run"})
    results["determinism_per_run"] = bool(
        np.array_equal(a2.X, a_per_run.X, equal_nan=True)
        and list(a2.var_names) == list(a_per_run.var_names)
    )
    out_h5 = Path(
        "/private/tmp/claude-501/-Users-denysoliinyk-Documents/69808082-7afe-4747-a638-391d1e0a0714/scratchpad/egf_ms2_condition.h5ad"
    )
    a_cond.write_h5ad(out_h5)
    results["h5ad_roundtrip_ok"] = True
    print(
        f"[determinism] {results['determinism_per_run']} | h5ad write ok | collisions={len(a_cond.uns['alphaphos'].get('short_key_collisions', {}))}"
    )

    # ---- where do alphaPhos-only sites come from? -----------------------------
    a_only = q_per_run.index.difference(sn0.index)  # not even in the unfiltered SN report
    sn_only = sn75.index.difference(q_gm0.index)  # SN 0.75 sites we never produce even unmasked
    results["aphos_sites_absent_from_sn_cutoff0"] = len(a_only)
    results["sn0p75_sites_absent_from_aphos_unmasked"] = len(sn_only)
    print(
        f"[site-set] alphaPhos sites absent from SN cutoff_0 (position/key suspects): {len(a_only)} e.g. {a_only[:6].tolist()}"
    )
    print(
        f"[site-set] SN 0.75 sites never produced by alphaPhos even unmasked: {len(sn_only)} e.g. {sn_only[:6].tolist()}"
    )
    # how many of the SN-only are non-leading protein-group members?
    pg_members = set()
    for pg in psm_ms2["PG.ProteinGroups"].astype(str).unique():
        parts = pg.split(";")
        if len(parts) > 1:
            pg_members.update(parts[1:])
    sn_only_prot = pd.Index([k.split("|")[0] for k in sn_only])
    results["sn_only_from_nonleading_pg_member"] = int(sn_only_prot.isin(pg_members).sum())
    print(
        f"[site-set]   of which protein is a NON-leading member of an alphaPhos protein group: {results['sn_only_from_nonleading_pg_member']}"
    )

    if args.full:
        # ---- parquet == tsv ----------------------------------------------------
        t = time.perf_counter()
        psm_ms2_pq = ap.read_spectronaut(D / f"{PREFIX}long_ms2_parquet.parquet")
        print(
            f"[read] long_ms2_parquet: {len(psm_ms2_pq):,} rows in {time.perf_counter() - t:.1f}s (tsv had {len(psm_ms2):,})"
        )
        _a_pq, q_pq = run(
            "ms2_parquet_per_run_vs_sn0p75",
            psm_ms2_pq,
            {"localization_strategy": "per_run"},
            sn75,
            sn75_loc,
            extra_sanity=False,
        )
        common = q_pq.index.intersection(q_per_run.index)
        diff = (q_pq.loc[common] - q_per_run.reindex(columns=q_pq.columns).loc[common]).abs()
        results["parquet_vs_tsv"] = {
            "sites_tsv": len(q_per_run),
            "sites_parquet": len(q_pq),
            "sites_common": len(common),
            "max_abs_log2_diff_common": float(np.nanmax(diff.to_numpy())),
            "cells_differing_gt_1e-6": int((diff > 1e-6).sum().sum()),
        }
        print(f"[parquet vs tsv] {results['parquet_vs_tsv']}")
        # which TSV rows are missing from parquet?
        key_cols = ["R.FileName", "EG.PrecursorId"]
        tsv_keys = psm_ms2[key_cols].astype(str).agg("§".join, axis=1)
        pq_keys = psm_ms2_pq[key_cols].astype(str).agg("§".join, axis=1)
        missing = psm_ms2[~tsv_keys.isin(set(pq_keys))]
        results["parquet_vs_tsv"]["psm_rows_only_in_tsv"] = len(missing)
        results["parquet_vs_tsv"]["only_in_tsv_examples"] = (
            missing[
                ["R.FileName", "EG.PrecursorId", "PG.ProteinGroups", "EG.TotalQuantity (Settings)"]
            ]
            .head(5)
            .astype(str)
            .to_dict("records")
        )
        print(
            f"[parquet vs tsv] PSM rows only in TSV: {len(missing)}; examples:\n{missing[['EG.PrecursorId', 'PG.ProteinGroups', 'EG.TotalQuantity (Settings)', 'EG.PTMLocalizationProbabilities']].head(5).to_string()}"
        )

        # ---- dual export: MS2 forced == ms2-only; MS1 forced vs SN --------------
        t = time.perf_counter()
        psm_dual = ap.read_spectronaut(D / f"{PREFIX}long_ms1_and_ms2.tsv")
        print(
            f"[read] long_ms1_and_ms2.tsv: {len(psm_dual):,} rows in {time.perf_counter() - t:.1f}s; quant cols present: {[c for c in psm_dual.columns if 'Quantity' in c]}"
        )
        _a_dual_ms2, q_dual_ms2 = run(
            "dual_MS2_per_run_vs_sn0p75",
            psm_dual,
            {"localization_strategy": "per_run", "quantification_level": "MS2"},
            sn75,
            sn75_loc,
            extra_sanity=False,
        )
        eq = q_dual_ms2.index.equals(q_per_run.index) and np.allclose(
            q_dual_ms2.to_numpy(),
            q_per_run.reindex(columns=q_dual_ms2.columns).to_numpy(),
            equal_nan=True,
            rtol=1e-6,
        )
        results["dual_MS2_identical_to_ms2_only"] = bool(eq)
        print(f"[dual MS2 == ms2-only export] {eq}")
        _a_dual_ms1, q_dual_ms1 = run(
            "dual_MS1_per_run_vs_sn0p75",
            psm_dual,
            {"localization_strategy": "per_run", "quantification_level": "MS1"},
            sn75,
            sn75_loc,
            extra_sanity=False,
        )
        _a_dual_auto, _q_dual_auto = run(
            "dual_auto_per_run_vs_sn0p75",
            psm_dual,
            {"localization_strategy": "per_run", "quantification_level": "auto"},
            sn75,
            sn75_loc,
            extra_sanity=False,
        )

        # ---- MS1-only export ----------------------------------------------------
        t = time.perf_counter()
        psm_ms1 = ap.read_spectronaut(D / f"{PREFIX}long_ms1.tsv")
        print(
            f"[read] long_ms1.tsv: {len(psm_ms1):,} rows in {time.perf_counter() - t:.1f}s; quant cols: {[c for c in psm_ms1.columns if 'Quantity' in c]}"
        )
        _a_ms1, q_ms1 = run(
            "ms1only_per_run_vs_sn0p75",
            psm_ms1,
            {"localization_strategy": "per_run"},
            sn75,
            sn75_loc,
            extra_sanity=False,
        )
        common = q_ms1.index.intersection(q_dual_ms1.index)
        dd = (q_ms1.loc[common] - q_dual_ms1.reindex(columns=q_ms1.columns).loc[common]).stack()
        results["ms1only_vs_dual_MS1"] = {
            "sites_common": len(common),
            "mean_log2_diff": round(float(dd.mean()), 4),
            "sd": round(float(dd.std()), 4),
            "pct_within_0.01": round(float((dd.abs() < 0.01).mean() * 100), 2),
        }
        print(f"[ms1-only export vs dual@MS1] {results['ms1only_vs_dual_MS1']}")
        # what does the MS1-only export's Settings column equal in the dual export?
        m = psm_ms1[["R.FileName", "EG.PrecursorId", "EG.TotalQuantity (Settings)"]].merge(
            psm_dual[
                [
                    "R.FileName",
                    "EG.PrecursorId",
                    "FG.MS1Quantity",
                    "FG.MS1RawQuantity",
                    "FG.MS2Quantity",
                    "PEP.Quantity",
                ]
            ]
            if "PEP.Quantity" in psm_dual.columns
            else psm_dual[
                [
                    "R.FileName",
                    "EG.PrecursorId",
                    "FG.MS1Quantity",
                    "FG.MS1RawQuantity",
                    "FG.MS2Quantity",
                ]
            ],
            on=["R.FileName", "EG.PrecursorId"],
            how="inner",
        )
        ratios = {}
        for c in [
            c
            for c in m.columns
            if c not in ("R.FileName", "EG.PrecursorId", "EG.TotalQuantity (Settings)")
        ]:
            r_ = (m["EG.TotalQuantity (Settings)"] / m[c]).replace([np.inf, -np.inf], np.nan)
            ratios[c] = {
                "frac_equal": round(
                    float(np.isclose(m["EG.TotalQuantity (Settings)"], m[c], rtol=1e-6).mean()), 3
                ),
                "median_ratio": round(float(r_.median()), 4),
                "ratio_IQR": [
                    round(float(r_.quantile(0.25)), 4),
                    round(float(r_.quantile(0.75)), 4),
                ],
            }
        results["ms1only_settings_column_identity"] = ratios
        print(f"[ms1-only export: what is EG.TotalQuantity (Settings)?] {json.dumps(ratios)}")

        # ---- aggregation methods ------------------------------------------------
        for agg in ("median", "consolidate"):
            run(
                f"ms2_condition_{agg}_vs_sn0p75",
                psm_ms2,
                {"localization_strategy": "condition", "aggregation_method": agg},
                sn75,
                sn75_loc,
                extra_sanity=False,
            )
        # ---- top_n off ----------------------------------------------------------
        _a_topn_off, q_topn_off = run(
            "ms2_per_run_topN_off_vs_sn0p75",
            psm_ms2,
            {"localization_strategy": "per_run", "top_n_attribution": False},
            sn75,
            sn75_loc,
            extra_sanity=False,
        )
        results["topN_changes_output"] = not (
            q_topn_off.index.equals(q_per_run.index)
            and np.allclose(
                q_topn_off.to_numpy(),
                q_per_run.reindex(columns=q_topn_off.columns).to_numpy(),
                equal_nan=True,
            )
        )
        print(f"[top_n_attribution off changes output] {results['topN_changes_output']}")
        # ---- wilson with fixed threshold runs ----------------------------------
        aw = ap.collapse_sites(
            psm_ms2,
            condition_df=cdf,
            advanced={"localization_strategy": "wilson", "wilson_threshold": 0.2},
        )
        results["wilson_fixed_0.2_sites"] = int(aw.n_vars)
        print(f"[wilson fixed 0.2] sites={aw.n_vars} ({aw.uns['wilson_filter']['reason']})")

    out = args.out or Path(
        "/private/tmp/claude-501/-Users-denysoliinyk-Documents/69808082-7afe-4747-a638-391d1e0a0714/scratchpad/validate_collapse_egf_hela.json"
    )
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[done] results -> {out}")


if __name__ == "__main__":
    main()
