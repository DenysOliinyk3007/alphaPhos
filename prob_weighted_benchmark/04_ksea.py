"""Pathway-level readout: self-contained KSEA (Casado 2013 z-score) using
kinase_substrate_db.gmt. For each representation, infer kinase activity from
EGF-vs-ctrl log2FC and check whether canonical EGF-pathway kinases are among
the top-activated. Pools many substrates -> far more power than 30 gold sites.
Compares hard-site (A) vs soft-site (C): substrate coverage + EGF-kinase z.
"""
from __future__ import annotations
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
GMT = "D:/Projects/miniBinders/scripts/kinase_substrate_db.gmt"
EGF_KINASES = ["EGFR","ERBB2","SRC","MAP2K1","MAP2K2","BRAF","RAF1","MAPK1","MAPK3",
               "RPS6KA1","RPS6KA2","RPS6KA3","RPS6KB1","AKT1","AKT2","AKT3","MTOR"]

# kinase -> set of substrate 'GENE_RESIDUE'
ksdb = {}
for line in open(GMT):
    parts = line.rstrip("\n").split("\t")
    if len(parts) > 2:
        ksdb[parts[0]] = set(parts[2:])

def de_log2fc(path):
    a = ad.read_h5ad(path); X = np.asarray(a.layers["intensity_log2"], float)
    c = a.obs["condition"].to_numpy(); E, C = X[c=="EGF"], X[c=="ctrl"]
    nE, nC = (~np.isnan(E)).sum(0), (~np.isnan(C)).sum(0)
    both = (nE >= 2) & (nC >= 2)
    with np.errstate(invalid="ignore"):
        fc = np.nanmean(E,0) - np.nanmean(C,0)
    parts = a.var_names.to_series().str.split("|")
    gene, resi = parts.str[1], parts.str[2]
    df = pd.DataFrame({"gene_res": (gene+"_"+resi).values, "log2fc": fc, "both": both})
    df = df[df["both"] & np.isfinite(df["log2fc"])]
    # collapse multiplicities: mean log2fc per gene_residue
    return df.groupby("gene_res")["log2fc"].mean()

def ksea(fc_series):
    mu, delta = fc_series.mean(), fc_series.std(ddof=0)
    universe = set(fc_series.index)
    rows = []
    for k, subs in ksdb.items():
        hit = [s for s in subs if s in universe]
        if len(hit) < 3:
            continue
        m = fc_series.loc[hit].mean()
        z = (m - mu) * np.sqrt(len(hit)) / delta
        rows.append({"kinase": k, "n_sub": len(hit), "mean_log2fc": m, "z": z,
                     "p": 2*stats.norm.sf(abs(z))})
    r = pd.DataFrame(rows).sort_values("z", ascending=False).reset_index(drop=True)
    r["rank"] = np.arange(1, len(r)+1)
    return r

for name, path in [("A hard-site", HERE/"egf_A_hard_site.h5ad"),
                   ("C soft-site", HERE/"egf_C_soft_site.h5ad")]:
    fc = de_log2fc(path); r = ksea(fc)
    print(f"\n===== {name}: {len(fc):,} quantified sites, {len(r)} kinases scored (>=3 substrates) =====")
    print("Top 10 activated kinases:")
    print(r.head(10)[["rank","kinase","n_sub","z","p"]].to_string(index=False))
    egf = r[r.kinase.isin(EGF_KINASES)].copy()
    n_top50 = int((egf["rank"] <= 50).sum())
    print(f"EGF-pathway kinases scored: {len(egf)}/{len(EGF_KINASES)}; "
          f"{(egf['z']>0).sum()} activated (z>0); {n_top50} in top-50 by z")
    print(egf.sort_values("z", ascending=False)[["rank","kinase","n_sub","z","p"]].to_string(index=False))
    r.to_csv(HERE/f"ksea_{name.split()[0]}.csv", index=False)
