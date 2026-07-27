"""Benchmark hard-site (A) vs soft-attribution site (C) [B=peptide, protein-level ref]
against the SIGNOR EGF gold-standard phosphosites. EGF vs ctrl, observed-only.

A gold site counts as RECOVERED if it is significantly UP (BH FDR<0.05, log2FC>0)
OR on-in-EGF (detected >=2 EGF, <2 ctrl). Reports coverage, sensitivity,
enrichment, and the head-to-head 'rescued by soft' set.
"""
from __future__ import annotations
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
AA3 = {"Ser": "S", "Thr": "T", "Tyr": "Y"}

# ---- gold standard: human phospho target sites from SIGNOR ----
def gold_sites():
    files = ["D:/Projects/nanoPhos_env/docs/phosphonetworks/SIGNOR_egfr.tsv",
             "D:/Projects/nanoPhos_env/docs/SIGNOR-EGF_25_06_26.tsv"]
    sites = set()
    for f in files:
        s = pd.read_csv(f, sep="\t")
        s = s[s["MECHANISM"].astype(str).str.contains("phosphorylation", case=False, na=False)]
        s = s[s["TAX_ID"].astype(str) == "9606"]
        for idb, res in zip(s["IDB"].astype(str), s["RESIDUE"].astype(str)):
            m = pd.Series(res).str.extract(r"^(Ser|Thr|Tyr)(\d+)").iloc[0]
            if pd.notna(m[0]):
                sites.add((idb, f"{AA3[m[0]]}{m[1]}"))
    return sites

GOLD = gold_sites()
print(f"Gold standard: {len(GOLD)} human (UniProt,residue) sites across "
      f"{len({g[0] for g in GOLD})} proteins\n")

# ---- observed-only DE (Welch + BH) + on/off ----
def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.empty(n); r[o] = np.arange(1, n+1)
    q = p * n / r; qs = np.minimum.accumulate((q[o])[::-1])[::-1]; out = np.empty(n); out[o] = qs
    return np.clip(out, 0, 1)

def de(adata):
    X = np.asarray(adata.layers["intensity_log2"], float)
    c = adata.obs["condition"].to_numpy()
    E, C = X[c == "EGF"], X[c == "ctrl"]
    nE, nC = (~np.isnan(E)).sum(0), (~np.isnan(C)).sum(0)
    res = pd.DataFrame(index=adata.var_names)
    res["nE"], res["nC"] = nE, nC
    both = (nE >= 2) & (nC >= 2)
    mE, mC = np.nanmean(E, 0), np.nanmean(C, 0)
    res["log2fc"] = mE - mC
    p = np.full(adata.n_vars, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for j in np.where(both)[0]:
            p[j] = stats.ttest_ind(E[:, j][~np.isnan(E[:, j])], C[:, j][~np.isnan(C[:, j])],
                                   equal_var=False).pvalue
    res["p"] = p
    res["fdr"] = np.nan
    res.loc[both, "fdr"] = bh(p[both])
    res["sig_up"] = (res["fdr"] < 0.05) & (res["log2fc"] > 0)
    res["on_in_EGF"] = (nE >= 2) & (nC < 2)
    res["recovered_up"] = res["sig_up"] | res["on_in_EGF"]
    return res

# ---- map features -> (UniProt, residue) ----
def site_keys_from_index(idx):
    # 'UNIPROT|GENE|RESIDUE[|MULT]' ; residue like S473 (strip multiplicity)
    parts = idx.to_series().str.split("|")
    prot = parts.str[0]
    resi = parts.str[2]
    return list(zip(prot, resi))

def score(res, name, level="site"):
    if level == "site":
        keys = site_keys_from_index(res.index)
        res = res.assign(prot=[k[0] for k in keys], resi=[k[1] for k in keys])
        res["is_gold"] = [(p, r) in GOLD for p, r in keys]
        tested = res[(res["nE"] >= 2) | (res["nC"] >= 2)]          # observed at all
        gold_detected = {(p, r) for p, r, obs in zip(res["prot"], res["resi"], (res["nE"]+res["nC"]))
                         if (p, r) in GOLD and obs >= 2}
        gold_recovered = {(p, r) for p, r, rec in zip(res["prot"], res["resi"], res["recovered_up"])
                          if (p, r) in GOLD and rec}
        # enrichment of gold among recovered-up
        rec = res["recovered_up"]; gold = res["is_gold"]
        a = int((rec & gold).sum()); b = int((rec & ~gold).sum())
        cc = int((~rec & gold).sum()); d = int((~rec & ~gold).sum())
        OR, pF = stats.fisher_exact([[a, b], [cc, d]], alternative="greater")
        print(f"[{name}] tested_sites={len(tested):,}  gold_detected={len(gold_detected)}/{len(GOLD)}  "
              f"gold_recovered_up={len(gold_recovered)}  "
              f"enrichment OR={OR:.1f} p={pF:.1e}")
        return gold_recovered, gold_detected
    return set(), set()

A = de(ad.read_h5ad(HERE / "egf_A_hard_site.h5ad"))
C = de(ad.read_h5ad(HERE / "egf_C_soft_site.h5ad"))
recA, detA = score(A, "A hard-site")
recC, detC = score(C, "C soft-site")

print(f"\nGold recovered — A only: {len(recA-recC)}   C only: {len(recC-recA)}   both: {len(recA&recC)}")
print("\nRescued by soft-attribution (gold up in C, not in A):")
for p, r in sorted(recC - recA):
    print(f"   {p} {r}")
print("\nRecovered only by hard-site (gold up in A, not in C):")
for p, r in sorted(recA - recC):
    print(f"   {p} {r}")

# per-gold-site table
rows = []
for (p, r) in sorted(GOLD):
    rows.append({"protein": p, "residue": r,
                 "A_detected": (p, r) in detA, "A_up": (p, r) in recA,
                 "C_detected": (p, r) in detC, "C_up": (p, r) in recC})
tab = pd.DataFrame(rows)
tab.to_csv(HERE / "gold_site_recovery.csv", index=False)
print(f"\nPer-gold-site table -> {HERE/'gold_site_recovery.csv'}")
print(tab[tab[["A_detected","C_detected"]].any(axis=1)].to_string(index=False))
