"""Specificity / null check: run the observed-only DE under the TRUE EGF/ctrl
labels and under all other 3-vs-3 label assignments (permutation null).
If soft-attribution (C) is picking up real signal (not a feature-count
artifact), the true labels should give far more significant-up sites AND the
gold enrichment should appear ONLY under the true labels.
"""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
AA3 = {"Ser": "S", "Thr": "T", "Tyr": "Y"}

def gold_sites():
    sites = set()
    for f in ["D:/Projects/nanoPhos_env/docs/phosphonetworks/SIGNOR_egfr.tsv",
              "D:/Projects/nanoPhos_env/docs/SIGNOR-EGF_25_06_26.tsv"]:
        s = pd.read_csv(f, sep="\t")
        s = s[s["MECHANISM"].astype(str).str.contains("phosphorylation", case=False, na=False)]
        s = s[s["TAX_ID"].astype(str) == "9606"]
        for idb, res in zip(s["IDB"].astype(str), s["RESIDUE"].astype(str)):
            m = pd.Series(res).str.extract(r"^(Ser|Thr|Tyr)(\d+)").iloc[0]
            if pd.notna(m[0]):
                sites.add((idb, f"{AA3[m[0]]}{m[1]}"))
    return sites
GOLD = gold_sites()

def bh(p):
    p = np.asarray(p, float); n = len(p)
    if n == 0: return p
    o = np.argsort(p); r = np.empty(n); r[o] = np.arange(1, n+1)
    q = p*n/r; qs = np.minimum.accumulate((q[o])[::-1])[::-1]; out = np.empty(n); out[o] = qs
    return np.clip(out, 0, 1)

def de_vec(X, is_E):
    """Vectorized observed-only Welch per column. is_E: bool sample mask."""
    E, C = X[is_E], X[~is_E]
    nE, nC = (~np.isnan(E)).sum(0), (~np.isnan(C)).sum(0)
    both = (nE >= 2) & (nC >= 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mE, mC = np.nanmean(E, 0), np.nanmean(C, 0)
        vE, vC = np.nanvar(E, 0, ddof=1), np.nanvar(C, 0, ddof=1)
        se = np.sqrt(vE/nE + vC/nC)
        t = (mE - mC)/se
        dfree = (vE/nE + vC/nC)**2 / ((vE/nE)**2/(nE-1) + (vC/nC)**2/(nC-1))
        p = 2*stats.t.sf(np.abs(t), dfree)
    fc = mE - mC
    fdr = np.full(X.shape[1], np.nan); fdr[both] = bh(p[both])
    sig_up = (fdr < 0.05) & (fc > 0)
    on_E = (nE >= 2) & (nC < 2)
    return sig_up | on_E, fc  # recovered_up mask

def keys_gold_mask(var_index):
    parts = var_index.to_series().str.split("|")
    return np.array([(p, r) in GOLD for p, r in zip(parts.str[0], parts.str[2])]), \
           list(zip(parts.str[0], parts.str[2]))

def run(name, path):
    a = ad.read_h5ad(path)
    X = np.asarray(a.layers["intensity_log2"], float)
    samples = list(a.obs_names); cond = a.obs["condition"].to_numpy()
    is_gold, keys = keys_gold_mask(a.var_names)
    true_E = cond == "EGF"
    rows = []
    for combo in combinations(range(6), 3):          # 20 labelings; which 3 are 'EGF'
        isE = np.zeros(6, bool); isE[list(combo)] = True
        rec, fc = de_vec(X, isE)
        a_ = int((rec & is_gold).sum()); b_ = int((rec & ~is_gold).sum())
        c_ = int((~rec & is_gold).sum()); d_ = int((~rec & ~is_gold).sum())
        OR, pF = stats.fisher_exact([[a_, b_], [c_, d_]], alternative="greater")
        gold_rec = len({keys[i] for i in np.where(rec & is_gold)[0]})
        rows.append({"is_true": bool(np.array_equal(isE, true_E)),
                     "n_sig_up": int(rec.sum()), "gold_recovered": gold_rec,
                     "enrich_OR": OR, "enrich_p": pF})
    r = pd.DataFrame(rows)
    true = r[r.is_true].iloc[0]; null = r[~r.is_true]
    print(f"\n=== {name} ===")
    print(f"  n_sig_up      : TRUE={true.n_sig_up:6d} | null median={null.n_sig_up.median():.0f} "
          f"max={null.n_sig_up.max():.0f}")
    print(f"  gold_recovered: TRUE={true.gold_recovered:6d} | null median={null.gold_recovered.median():.1f} "
          f"max={null.gold_recovered.max():.0f}")
    print(f"  enrichment p  : TRUE={true.enrich_p:.1e} | null min p={null.enrich_p.min():.1e} "
          f"({(null.enrich_p<0.05).sum()}/{len(null)} nulls reach p<0.05)")
    return r

run("A hard-site", HERE / "egf_A_hard_site.h5ad")
run("C soft-site", HERE / "egf_C_soft_site.h5ad")
