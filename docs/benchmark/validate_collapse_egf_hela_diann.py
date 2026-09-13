"""Validate ``collapse_sites`` on a DIA-NN 2.x report against DIA-NN's own site tables.

Dataset: ``Documents/Data/egf_hela_nanophos_diann`` -- the same six EGF HeLa raw
files as the Spectronaut benchmark, searched with DIA-NN 2.2.0 (library-free,
``--var-mod UniMod:21,79.966331,STY``, MBR).  DIA-NN 2.x ships

* ``report.parquet``               -- main report (input to ``read_diann``)
* ``report.site_report.parquet``   -- per (run, precursor, candidate residue):
                                      ``Occupied``, ``Probability``, ``Fragment.Sum``
* ``report.phosphosites_90/99.tsv`` -- native site matrices (Top-1 method, see §2)

Three references are used:

1. **Positions**: alphaPhos absolute positions must be a subset of the residues
   DIA-NN marks ``Occupied`` for that precursor (the leading protein).
2. **Oracle**: an independent gated sum built from the site table --
   ``sum(Precursor.Quantity)`` over occupied precursors with ``Probability >= cutoff``
   per (protein, site, run).  This is what a correct collapse of DIA-NN
   output *should* produce; it tests our parsing/gating/aggregation mechanics.
3. **DIA-NN's own matrices**: site set only.  Their quantity is the *maximum*
   ``Fragment.Sum x Normalisation.Factor`` over qualifying precursors (a top-3
   fragment sum; documented by DIA-NN as "intended only for quick preliminary
   analyses"), so per-cell quant is not comparable to a precursor-quantity sum.

Optionally (``--cross-engine``) compares against the Spectronaut collapse of the
same raw files.

Usage::

    .venv/bin/python docs/benchmark/validate_collapse_egf_hela_diann.py [--data-dir DIR] [--cross-engine]
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
from alphaphos.io.diann import diann_precursor_id
from alphaphos.preprocess._collapse.parsing import extract_sequence_modifications

warnings.filterwarnings("ignore")

DEFAULT_DIR = Path("/Users/denysoliinyk/Documents/Data/egf_hela_nanophos_diann")
SN_DIR = Path("/Users/denysoliinyk/Documents/Data/egf_hela_nanophos_spectronaut")
SN_LONG = SN_DIR / "nanophos_egf_hela_1ug_long_ms2.tsv"


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def run_from_path(col: str) -> str:
    return re.sub(r"\.raw$", "", col.replace("\\", "/").split("/")[-1])


def load_matrix(path: Path) -> pd.DataFrame:
    """DIA-NN phosphosites_XX.tsv -> linear matrix indexed by ``Protein|AApos``."""
    m = pd.read_csv(path, sep="\t")
    runcols = [c for c in m.columns if ".raw" in c]
    m = m.rename(columns={c: run_from_path(c) for c in runcols})
    m["key"] = m["Protein"] + "|" + m["Residue"] + m["Site"].astype(str)
    return m.set_index("key")[[run_from_path(c) for c in runcols]].replace(0, np.nan)


def load_site_table(D: Path) -> pd.DataFrame:
    """Occupied rows of site_report joined to the main report's quantities."""
    sr = pd.read_parquet(D / "report.site_report.parquet")
    rep = pd.read_parquet(
        D / "report.parquet",
        columns=[
            "Run",
            "Precursor.Id",
            "Modified.Sequence",
            "Precursor.Charge",
            "Protein.Group",
            "Precursor.Quantity",
            "Precursor.Normalised",
            "Normalisation.Factor",
            "Q.Value",
            "PG.Q.Value",
            "Peptidoform.Q.Value",
        ],
    )
    occ = sr[sr["Occupied"] == 1].merge(rep, on=["Run", "Precursor.Id"], how="left")
    occ["key"] = occ["Protein"] + "|" + occ["Residue"] + occ["Site"].astype(str)
    return occ


def oracle_matrix(
    occ: pd.DataFrame, cutoff: float, quant: str = "Precursor.Quantity"
) -> pd.DataFrame:
    d = occ[occ["Probability"] >= cutoff]
    return np.log2(d.groupby(["key", "Run"])[quant].sum(min_count=1).unstack("Run"))


def alphaphos_nomult(adata) -> pd.DataFrame:
    """alphaPhos site matrix re-keyed as ``Protein|AApos`` with multiplicities summed."""
    v = adata.var
    key = (
        v["protein_group_id"].astype(str)
        + "|"
        + v["site_aa"].astype(str)
        + v["site_position"].astype(int).astype(str)
    )
    q = pd.DataFrame(adata.layers["intensity_log2"].T, index=key.values, columns=adata.obs_names)
    return np.log2(np.exp2(q.astype(float)).groupby(level=0).sum(min_count=1))


def compare(a: pd.DataFrame, ref: pd.DataFrame) -> dict:
    a = a.reindex(columns=ref.columns)
    shared = a.index.intersection(ref.index)
    A, B = a.loc[shared].to_numpy(), ref.loc[shared].to_numpy()
    both = ~np.isnan(A) & ~np.isnan(B)
    d = A[both] - B[both]
    w = [c for c in ref.columns if "withEGF" in c]
    o = [c for c in ref.columns if "woEGF" in c]
    fa = a.loc[shared, w].mean(axis=1) - a.loc[shared, o].mean(axis=1)
    fb = ref.loc[shared, w].mean(axis=1) - ref.loc[shared, o].mean(axis=1)
    ok = fa.notna() & fb.notna()
    return {
        "n_a": len(a),
        "n_ref": len(ref),
        "n_shared": len(shared),
        "jaccard": round(len(shared) / len(a.index.union(ref.index)), 4),
        "cells_both": int(both.sum()),
        "cells_a_only": int((~np.isnan(A) & np.isnan(B)).sum()),
        "cells_ref_only": int((np.isnan(A) & ~np.isnan(B)).sum()),
        "pearson_r_log2": round(float(np.corrcoef(A[both], B[both])[0, 1]), 4)
        if both.sum() > 2
        else None,
        "mean_diff_log2": round(float(d.mean()), 4),
        "sd_diff_log2": round(float(d.std()), 4),
        "pct_within_0.1": round(float((np.abs(d) <= 0.1).mean() * 100), 2),
        "log2fc_pearson_r": round(float(np.corrcoef(fa[ok], fb[ok])[0, 1]), 4)
        if ok.sum() > 2
        else None,
    }


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_positions(psm: pd.DataFrame, occ: pd.DataFrame, D: Path) -> dict:
    rep = pd.read_parquet(
        D / "report.parquet", columns=["Precursor.Id", "Modified.Sequence", "Precursor.Charge"]
    ).drop_duplicates("Precursor.Id")
    rep["EG.PrecursorId"] = [
        diann_precursor_id(m, c)
        for m, c in zip(rep["Modified.Sequence"], rep["Precursor.Charge"], strict=True)
    ]
    prec = psm.drop_duplicates("EG.PrecursorId").merge(
        rep[["EG.PrecursorId", "Precursor.Id"]], on="EG.PrecursorId"
    )
    prec["prot"] = prec["PG.ProteinGroups"].str.split(";").str[0]
    prec["aphos"] = [
        {int(s) + p - 1 for p in m["phospho_positions"]}
        for m, s in zip(
            prec["EG.PrecursorId"].map(extract_sequence_modifications),
            prec["PEP.PeptidePosition"],
            strict=True,
        )
    ]
    truth = occ.groupby(["Precursor.Id", "Protein"])["Site"].apply(set).reset_index()
    chk = prec.merge(
        truth, left_on=["Precursor.Id", "prot"], right_on=["Precursor.Id", "Protein"], how="inner"
    )
    subset = np.array([a <= t for a, t in zip(chk["aphos"], chk["Site"], strict=True)])
    bad = chk[~subset]
    return {
        "precursors_compared": len(chk),
        "pct_positions_subset_of_diann": round(float(subset.mean() * 100), 4),
        "n_disagree": int((~subset).sum()),
        "disagree_examples": bad[["EG.PrecursorId", "prot", "PEP.PeptidePosition"]]
        .head(5)
        .astype(str)
        .to_dict("records"),
        "n_multi_mapping": int(
            sum(len(t) > len(a) for a, t in zip(chk["aphos"], chk["Site"], strict=True))
        ),
    }


def check_matrix_recipe(occ: pd.DataFrame, m90: pd.DataFrame) -> dict:
    """DIA-NN docs: Top-1 = max over qualifying precursors of Fragment.Sum x Normalisation.Factor."""
    d = occ[occ["Probability"] >= 0.9].copy()
    d["fs_norm"] = d["Fragment.Sum"] * d["Normalisation.Factor"]
    n = d.groupby(["key", "Run"]).size()
    lg = m90.stack().rename("dn").reset_index()
    lg.columns = ["key", "Run", "dn"]
    single = d.merge(n[n == 1].reset_index()[["key", "Run"]], on=["key", "Run"]).merge(
        lg, on=["key", "Run"]
    )
    ratio = single["dn"] / single["fs_norm"]
    top1 = (
        d.groupby(["key", "Run"])["fs_norm"]
        .max()
        .rename("rc")
        .reset_index()
        .merge(lg, on=["key", "Run"])
    )
    return {
        "single_precursor_cells": len(single),
        "pct_within_1pct_of_FragmentSum_x_NormFactor": round(
            float((np.abs(ratio - 1) < 0.01).mean() * 100), 2
        ),
        "all_cells_top1_pct_within_1pct": round(
            float((np.abs(top1["dn"] / top1["rc"] - 1) < 0.01).mean() * 100), 2
        ),
        "note": "DIA-NN matrices = max(Fragment.Sum x Normalisation.Factor); a top-3-fragment quantity, not Precursor.Quantity",
    }


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--data-dir", type=Path, default=DEFAULT_DIR)
    ap_.add_argument("--cross-engine", action="store_true")
    ap_.add_argument("--out", type=Path, default=None)
    args = ap_.parse_args()
    D = args.data_dir
    results: dict = {}

    t = time.perf_counter()
    psm = ap.read_diann(D / "report.parquet")
    results["read_diann_attrs"] = {k: v for k, v in psm.attrs.items() if k.startswith("n_rows")}
    print(
        f"[read_diann] {len(psm):,} rows in {time.perf_counter() - t:.1f}s; funnel={results['read_diann_attrs']}"
    )
    occ = load_site_table(D)
    m90, m99 = (
        load_matrix(D / "report.phosphosites_90.tsv"),
        load_matrix(D / "report.phosphosites_99.tsv"),
    )
    samples = sorted(psm["R.FileName"].unique())
    cdf = pd.DataFrame(
        {
            "sample": samples,
            "condition": ["withEGF" if "withEGF" in s else "woEGF" for s in samples],
        }
    )

    results["positions"] = check_positions(psm, occ, D)
    print(f"[positions] {results['positions']}")
    results["diann_matrix_recipe"] = check_matrix_recipe(occ, m90)
    print(f"[DIA-NN matrix recipe] {results['diann_matrix_recipe']}")

    for cutoff, mat in ((0.9, m90), (0.99, m99)):
        adata = ap.collapse_sites(
            psm,
            condition_df=cdf,
            advanced={
                "search_engine": "Diann",
                "quantification_level": "auto",
                "localization_strategy": "per_run",
                "cutoff": cutoff,
            },
        )
        a = alphaphos_nomult(adata)
        r_or = compare(a, oracle_matrix(occ, cutoff))
        r_dn = compare(a, np.log2(mat))
        results[f"cutoff_{cutoff}"] = {
            "stats": {
                k: (int(v) if isinstance(v, (int, np.integer)) else v)
                for k, v in adata.uns["alphaphos"]["stats"].items()
            },
            "vs_oracle": r_or,
            "vs_diann_matrix": r_dn,
        }
        print(
            f"[cutoff={cutoff}] sites={adata.n_vars} (gated cells {adata.uns['alphaphos']['stats']['n_precursor_cells_gated']}, top_n applied={adata.uns['alphaphos']['stats']['top_n_attribution_applied']})"
        )
        print(
            f"   vs ORACLE gated sum : r={r_or['pearson_r_log2']} mean_d={r_or['mean_diff_log2']:+.3f} sd={r_or['sd_diff_log2']} within0.1={r_or['pct_within_0.1']}% jaccard={r_or['jaccard']} a-only/ref-only={r_or['cells_a_only']}/{r_or['cells_ref_only']} log2FC_r={r_or['log2fc_pearson_r']}"
        )
        print(
            f"   vs DIA-NN matrix    : site jaccard={r_dn['jaccard']} (a={r_dn['n_a']}, dn={r_dn['n_ref']}) r={r_dn['pearson_r_log2']} [Top-1 fragment quantity: offset not meaningful]"
        )

    if args.cross_engine and SN_LONG.exists():
        psm_sn = ap.read_spectronaut(SN_LONG)
        a_sn = alphaphos_nomult(
            ap.collapse_sites(
                psm_sn, condition_df=cdf, advanced={"localization_strategy": "per_run"}
            )
        )
        a_dn = alphaphos_nomult(
            ap.collapse_sites(
                psm,
                condition_df=cdf,
                advanced={
                    "search_engine": "Diann",
                    "quantification_level": "auto",
                    "localization_strategy": "per_run",
                },
            )
        )
        # engines report intensities on different scales -> per-sample median centring
        r = compare(a_dn - a_dn.median(), a_sn - a_sn.median())
        shared = a_dn.index.intersection(a_sn.index)
        w = [c for c in a_sn.columns if "withEGF" in c]
        o = [c for c in a_sn.columns if "woEGF" in c]
        cc = a_sn.loc[shared, w + o].notna().all(axis=1) & a_dn.reindex(columns=a_sn.columns).loc[
            shared, w + o
        ].notna().all(axis=1)
        fs = a_sn.loc[shared, w].mean(axis=1) - a_sn.loc[shared, o].mean(axis=1)
        fd = a_dn.loc[shared, w].mean(axis=1) - a_dn.loc[shared, o].mean(axis=1)
        r["log2fc_r_complete_cases"] = round(float(np.corrcoef(fs[cc], fd[cc])[0, 1]), 4)
        r["n_complete_cases"] = int(cc.sum())
        r["egfr_autophosphorylation_log2fc"] = {
            k: {"spectronaut": round(float(fs[k]), 2), "diann": round(float(fd[k]), 2)}
            for k in shared
            if k.startswith("P00533|Y") and pd.notna(fs[k]) and pd.notna(fd[k])
        }
        results["cross_engine_diann_vs_spectronaut"] = r
        print(f"[cross-engine] {json.dumps(r)}")

    out = args.out or Path(
        "/private/tmp/claude-501/-Users-denysoliinyk-Documents/69808082-7afe-4747-a638-391d1e0a0714/scratchpad/validate_collapse_egf_hela_diann.json"
    )
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[done] results -> {out}")


if __name__ == "__main__":
    main()
