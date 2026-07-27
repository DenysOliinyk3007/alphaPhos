"""Build 3 representations of the EGF+/- dataset from ONE raw report, same quant:
  (A) hard Class-I site level  (collapse_sites, per_run, cutoff 0.75)
  (B) peptide/precursor level  (collapse_precursors, localization-agnostic)
  (C) probability-weighted soft-attribution site level  (our approach)

All use quantification_level='auto' (EG.TotalQuantity (Settings)) so the only
difference is how localization is handled. Saves three .h5ad files.
"""
from __future__ import annotations
import re
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import alphaphos as ap

HERE = Path(__file__).resolve().parent
REPORT = Path("D:/Projects/alphaPhos/test_data/benchmark/EGF_report_new_ms2.tsv")
QUANT = "EG.TotalQuantity (Settings)"
ADV_SITE = {"quantification_level": "auto", "localization_strategy": "per_run", "cutoff": 0.75}
ADV_PREC = {"quantification_level": "auto", "classI_cutoff": None}

# ---- condition metadata ----
cond_raw = pd.read_csv(REPORT, sep="\t", usecols=["R.FileName", "R.Condition"]).drop_duplicates()
cond_raw["condition"] = cond_raw["R.Condition"].astype(str).str.strip().map({"+": "EGF", "-": "ctrl"})
condition_df = cond_raw.rename(columns={"R.FileName": "sample"})[["sample", "condition"]]
print("Conditions:\n", condition_df.to_string(index=False))

# ---- (A) + (B) via alphaPhos ----
psm = ap.read_spectronaut(REPORT)
site = ap.collapse_sites(psm, condition_df=condition_df, advanced=ADV_SITE)
prec = ap.collapse_precursors(psm, condition_df=condition_df, advanced=ADV_PREC)
for a in (site, prec):
    a.uns.pop("alphaphos", None)  # avoid non-serializable uns
site.write_h5ad(HERE / "egf_A_hard_site.h5ad")
prec.write_h5ad(HERE / "egf_B_peptide.h5ad")
print(f"(A) hard-site: {site.shape}   (B) peptide: {prec.shape}")

# ---- (C) probability-weighted soft-attribution site matrix ----
cols = ["R.FileName", "PG.Genes", "PG.ProteinGroups", "PEP.PeptidePosition",
        "EG.ModifiedSequence", "EG.PrecursorId", "EG.IsDecoy",
        "EG.PTMLocalizationProbabilities", QUANT]
df = pd.read_csv(REPORT, sep="\t", usecols=cols)
df = df[df["EG.IsDecoy"] != True]                                   # drop decoys
df = df[df["PG.ProteinGroups"].astype(str).str.len() > 0]
df = df[df["EG.PTMLocalizationProbabilities"].astype(str).str.contains("Phospho", na=False)]
# one intensity per precursor-observation
df = df.drop_duplicates(["R.FileName", "EG.PrecursorId"])
print(f"(C) phospho precursor-observations to attribute: {len(df):,}")

PHOS = re.compile(r"Phospho.*?:\s*([\d.]+)%")

def parse(modprob, pep_start):
    s = str(modprob).strip("_"); out = []; pep_idx = 0; i = 0
    while i < len(s):
        c = s[i]
        if c.isalpha():
            pep_idx += 1; j = i + 1
            while j < len(s) and s[j] == "[":
                k = s.index("]", j); tag = s[j+1:k]
                m = PHOS.search(tag)
                if m:
                    out.append((c, pep_start + pep_idx - 1, float(m.group(1)) / 100.0))
                j = k + 1
            i = j
        elif c == "[":
            i = s.index("]", i) + 1
        else:
            i += 1
    return out

# accumulate soft intensity per (protein, gene, residue) x sample
from collections import defaultdict
acc = defaultdict(lambda: defaultdict(float))
n_skip = 0
for fn, gene, pg, pos, modprob, inten in zip(
        df["R.FileName"], df["PG.Genes"], df["PG.ProteinGroups"],
        df["PEP.PeptidePosition"], df["EG.PTMLocalizationProbabilities"], df[QUANT]):
    p0 = str(pos).split(";")[0]
    if not p0.lstrip("-").isdigit() or not np.isfinite(inten) or inten <= 0:
        n_skip += 1; continue
    prot = str(pg).split(";")[0]
    for aa, ppos, prob in parse(modprob, int(p0)):
        if prob <= 0:
            continue
        acc[(prot, str(gene).split(";")[0], f"{aa}{ppos}")][fn] += inten * prob
print(f"(C) sites accumulated: {len(acc):,}  (skipped {n_skip} rows w/ bad pos/intensity)")

samples = condition_df["sample"].tolist()
keys = sorted(acc.keys())
X = np.full((len(samples), len(keys)), np.nan)
for j, k in enumerate(keys):
    for i, s in enumerate(samples):
        v = acc[k].get(s)
        if v:
            X[i, j] = np.log2(v)
var = pd.DataFrame(
    {"protein": [k[0] for k in keys], "gene": [k[1] for k in keys],
     "residue": [k[2] for k in keys]},
    index=[f"{k[0]}|{k[1]}|{k[2]}" for k in keys])
soft = ad.AnnData(X=X, obs=condition_df.set_index("sample"), var=var)
soft.layers["intensity_log2"] = soft.X.copy()
soft.write_h5ad(HERE / "egf_C_soft_site.h5ad")
print(f"(C) soft-site: {soft.shape}")

# sanity: are canonical EGF genes present in each?
egf_genes = ["EGFR", "GRB2", "GAB1", "SOS1", "SHC1", "MAPK1", "MAPK3", "ELK1"]
def genes_of(a, col):
    return set(a.var[col].astype(str)) if col in a.var else set(
        a.var_names.to_series().str.split("|").str[1])
print("\nCanonical EGF genes present:")
print("  A hard-site:", sorted(g for g in egf_genes if g in genes_of(site, "gene")))
print("  B peptide  :", sorted(g for g in egf_genes if g in genes_of(prec, "gene")))
print("  C soft-site:", sorted(g for g in egf_genes if g in set(soft.var['gene'])))
