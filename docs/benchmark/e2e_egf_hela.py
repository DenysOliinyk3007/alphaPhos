"""End-to-end run of the alphaPhos pipeline on the EGF HeLa series, both engines.

read -> collapse (defaults) -> completeness filter -> DE (observed-only limma, plus
impute+limma for comparison) -> kinase activity (OmniPath) -> pathway ORA (Enrichr)
-> pathway GSEA -> site-level ORA/GSEA on the shipped PTM libraries, for the
Spectronaut and the DIA-NN search of the same six raw files, followed by a
cross-engine comparison of the biology that comes out.

Needs the ``[enrichment]`` extra (gseapy, decoupler, omnipath) and network access
for Enrichr / the first OmniPath fetch.  Data paths default to the lab machine.

    python docs/benchmark/e2e_egf_hela.py [--sn PATH] [--diann PATH] [--cache DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import alphaphos as ap
from alphaphos.enrichment import (
    canonicalise_site_ids,
    fetch_omnipath_ks_network,
    gsea,
    kinase_activity,
    load_libraries,
    ora,
    pathway_enrichment,
    pathway_gsea,
)
from alphaphos.resources import LIBRARIES_DIR

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

DATA = Path("/Users/denysoliinyk/Documents/Data")
SN_PATH = DATA / "egf_hela_nanophos_spectronaut" / "nanophos_egf_hela_1ug_long_ms2.tsv"
DIANN_PATH = DATA / "egf_hela_nanophos_diann" / "report.parquet"
CONTRAST = ("withEGF", "woEGF")
KNOWN_UP_KINASES = ["MAPK1", "MAPK3", "MAP2K1", "MAP2K2", "RPS6KA1", "RPS6KA3", "AKT1", "MAPKAPK2"]


def condition_df(psm: pd.DataFrame) -> pd.DataFrame:
    s = sorted(psm["R.FileName"].unique())
    return pd.DataFrame(
        {"sample": s, "condition": ["withEGF" if "withEGF" in x else "woEGF" for x in s]}
    )


def timed(label: str, fn, report: dict):
    t = time.perf_counter()
    out = fn()
    report.setdefault("timings_s", {})[label] = round(time.perf_counter() - t, 1)
    return out


def run_engine(name: str, path: Path, *, net: pd.DataFrame, libs: dict, cache: Path) -> dict:
    rep: dict = {"engine": name}
    print(f"\n{'=' * 78}\n{name}: {path.name}\n{'=' * 78}")

    # ---- 1. read + collapse with the engine's defaults -----------------------
    if name == "Spectronaut":
        psm = timed("read", lambda: ap.read_spectronaut(path), rep)
        adv = {}
    else:
        psm = timed("read", lambda: ap.read_diann(path), rep)
        adv = {"search_engine": "Diann"}
    cdf = condition_df(psm)
    adata = timed("collapse", lambda: ap.collapse_sites(psm, condition_df=cdf, advanced=adv), rep)
    st = adata.uns["alphaphos"]["stats"]
    rep["psm_rows"] = len(psm)
    rep["sites"] = int(adata.n_vars)
    rep["quant_column"] = st["quantification_column_used"]
    rep["missing_pct"] = round(float(np.isnan(adata.X).mean() * 100), 1)
    print(
        f"[collapse] {len(psm):,} PSM rows -> {adata.n_vars:,} site x mult (quant={st['quantification_column_used']}, "
        f"strategy={adata.uns['alphaphos']['pipeline_params']['localization_strategy']}, missing {rep['missing_pct']}%)"
    )

    # ---- 2. completeness filter ---------------------------------------------
    filt = timed(
        "filter",
        lambda: ap.filter_by_completeness(
            adata, min_valid_n=3, group_column="condition", keep_strategy="any"
        ),
        rep,
    )
    rep["sites_after_filter"] = int(filt.n_vars)
    print(f"[filter] >=3 valid in at least one condition: {filt.n_vars:,} sites")

    # ---- 3. differential expression -----------------------------------------
    de, on_off = timed(
        "de_observed_only",
        lambda: ap.diff_exp_limma_observed_only(
            adata, condition_column="condition", comparison=CONTRAST
        ),
        rep,
    )
    sig = de[de["fdr"] < 0.05]
    rep["de_tested"] = len(de)
    rep["de_sig"] = len(sig)
    rep["de_up"] = int((sig["log2fc"] > 0).sum())
    rep["de_down"] = int((sig["log2fc"] < 0).sum())
    rep["on_off"] = {k: int(v) for k, v in on_off["call"].value_counts().items()}
    imp = ap.impute_hybrid(filt.copy())
    de_imp = ap.diff_exp_limma(imp, condition_column="condition", comparison=CONTRAST)
    de_imp = ap.annotate_imputed_provenance(
        de_imp, filt, condition_column="condition", comparison=CONTRAST
    )
    s_imp = de_imp["fdr"] < 0.05
    rep["de_impute_all_sig"] = int(s_imp.sum())
    rep["de_impute_all_imputed_driven"] = int((s_imp & de_imp["imputed_driven"]).sum())
    print(
        f"[DE observed-only] {len(de):,} tested, {len(sig):,} sig @5% FDR (up {rep['de_up']} / down {rep['de_down']}); "
        f"on/off: {rep['on_off']}"
    )
    print(
        f"[DE impute-all]    {len(de_imp):,} tested, {int(s_imp.sum()):,} sig of which {rep['de_impute_all_imputed_driven']} imputed-driven"
    )
    # EGFR autophosphorylation sites are unquantified in unstimulated cells, so
    # they are detection (on/off) events, not limma hits -- report them from there.
    egfr = on_off[on_off.index.str.startswith("P00533|EGFR|Y")]
    print("[EGFR pTyr sites -- on/off table]\n" + egfr.to_string())
    rep["egfr_on_off"] = egfr["call"].to_dict()

    # ---- 4. kinase activity (OmniPath, ULM on moderated t) --------------------
    ka = timed(
        "kinase_activity",
        lambda: kinase_activity(de, stat_col="t_stat", network=net, min_substrates=5),
        rep,
    )
    top = ka.head(10)
    rep["kinases_tested"] = len(ka)
    rep["top_kinases"] = top["kinase"].tolist()
    known = ka[ka["kinase"].isin(KNOWN_UP_KINASES)].set_index("kinase")
    rep["known_egf_kinases_up_fdr05"] = int(((known["score"] > 0) & (known["fdr"] < 0.05)).sum())
    rep["known_egf_kinases_present"] = len(known)
    print(f"[kinase_activity] {len(ka)} kinases; top 10: {', '.join(top['kinase'])}")
    print(
        f"   known EGF-responsive kinases up @5% FDR: {rep['known_egf_kinases_up_fdr05']}/{rep['known_egf_kinases_present']} "
        f"({', '.join(k for k in known.index if known.loc[k, 'score'] > 0 and known.loc[k, 'fdr'] < 0.05)})"
    )
    rep["_ka"] = ka.set_index("kinase")["score"]

    # ---- 5. pathway ORA (Enrichr, phosphoproteome background) -----------------
    pe = timed(
        "pathway_enrichment",
        lambda: pathway_enrichment(
            de, libraries=["KEGG_2021_Human", "MSigDB_Hallmark_2020"], direction="split"
        ),
        rep,
    )
    pe_sig = pe[pe["fdr"] < 0.1].sort_values("fdr")
    rep["pathway_ora_terms_fdr10"] = len(pe_sig)
    rep["pathway_ora_top"] = pe_sig.head(8)[["direction", "term", "overlap", "fdr"]].to_dict(
        "records"
    )
    print(f"[pathway ORA] {len(pe)} terms, {len(pe_sig)} @10% FDR; top:")
    print(pe_sig.head(8)[["direction", "library", "term", "overlap", "fdr"]].to_string(index=False))
    rep["_pe"] = set(pe_sig["term"])

    # ---- 6. pathway GSEA (gseapy prerank on gene-collapsed t) -----------------
    pg = timed(
        "pathway_gsea",
        lambda: pathway_gsea(
            de, stat_col="t_stat", libraries=["MSigDB_Hallmark_2020"], n_permutations=1000, seed=1
        ),
        rep,
    )
    pg_sig = pg[pg["fdr"] < 0.25].sort_values("fdr")
    rep["pathway_gsea_terms_fdr25"] = len(pg_sig)
    print(f"[pathway GSEA Hallmark] {len(pg)} terms, {len(pg_sig)} @25% FDR (GSEA convention):")
    print(pg_sig.head(6)[["term", "nes", "fdr", "n_set", "n_leading_edge"]].to_string(index=False))
    rep["_pg"] = pg.set_index("term")["nes"]

    # ---- 7. site-level ORA / GSEA on the shipped PTM libraries ----------------
    bg = canonicalise_site_ids(de.index)
    hits = canonicalise_site_ids(sig.index)
    so = timed("site_ora", lambda: ora(hits, bg, libs), rep)
    print(
        "[site ORA]\n"
        + so[
            ["library", "set_name", "n_set", "n_overlap", "p_value", "fdr", "direction"]
        ].to_string(index=False)
    )
    ranked = pd.Series(de["t_stat"].to_numpy(), index=bg)
    ranked = ranked.groupby(level=0).agg(lambda v: v.iloc[int(v.abs().to_numpy().argmax())])
    sg = timed("site_gsea", lambda: gsea(ranked, libs, n_permutations=2000, seed=1), rep)
    print(
        "[site GSEA]\n"
        + sg[["library", "set_name", "n_set", "NES", "fdr", "direction"]].to_string(index=False)
    )
    rep["site_ora_enriched_fdr05"] = so.loc[
        (so["fdr"] < 0.05) & (so["direction"] == "enriched"), "set_name"
    ].tolist()
    rep["site_ora_depleted_fdr05"] = so.loc[
        (so["fdr"] < 0.05) & (so["direction"] == "depleted"), "set_name"
    ].tolist()
    rep["_so"] = so.set_index("set_name")["log2_fold_enrichment"]
    return rep


def cross_engine(sn: dict, dn: dict) -> dict:
    out: dict = {}
    ka = pd.concat([sn["_ka"].rename("sn"), dn["_ka"].rename("diann")], axis=1).dropna()
    out["kinases_in_both"] = len(ka)
    out["kinase_score_spearman"] = round(float(ka.corr(method="spearman").iloc[0, 1]), 3)
    top_sn, top_dn = set(sn["top_kinases"]), set(dn["top_kinases"])
    out["top10_kinase_overlap"] = sorted(top_sn & top_dn)
    out["pathway_ora_fdr10_shared"] = sorted(sn["_pe"] & dn["_pe"])
    out["pathway_ora_fdr10_sn_only"] = len(sn["_pe"] - dn["_pe"])
    out["pathway_ora_fdr10_diann_only"] = len(dn["_pe"] - sn["_pe"])
    pg = pd.concat([sn["_pg"].rename("sn"), dn["_pg"].rename("diann")], axis=1).dropna()
    out["hallmark_nes_spearman"] = round(float(pg.corr(method="spearman").iloc[0, 1]), 3)
    so = pd.concat([sn["_so"].rename("sn"), dn["_so"].rename("diann")], axis=1).dropna()
    out["site_ora_log2fe_pearson"] = round(float(so.corr().iloc[0, 1]), 3)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sn", type=Path, default=SN_PATH)
    p.add_argument("--diann", type=Path, default=DIANN_PATH)
    p.add_argument("--cache", type=Path, default=Path.home() / ".alphaphos" / "cache")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    net = fetch_omnipath_ks_network(
        organism="human", cache_path=args.cache / "omnipath_ks_human.parquet"
    )
    libs = load_libraries(LIBRARIES_DIR)
    print(
        f"OmniPath KS network: {len(net):,} edges, {net['source'].nunique()} kinases; PTM libraries: {list(libs)}"
    )

    reports = {
        "Spectronaut": run_engine("Spectronaut", args.sn, net=net, libs=libs, cache=args.cache),
        "DIA-NN": run_engine("DIA-NN", args.diann, net=net, libs=libs, cache=args.cache),
    }
    xe = cross_engine(reports["Spectronaut"], reports["DIA-NN"])
    print(f"\n{'=' * 78}\nCROSS-ENGINE\n{'=' * 78}")
    for k, v in xe.items():
        print(f"  {k}: {v}")

    summary = {
        k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in reports.items()
    }
    summary["cross_engine"] = xe
    out = args.out or Path(
        "/private/tmp/claude-501/-Users-denysoliinyk-Documents/69808082-7afe-4747-a638-391d1e0a0714/scratchpad/e2e_egf_hela.json"
    )
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[done] summary -> {out}")


if __name__ == "__main__":
    main()
