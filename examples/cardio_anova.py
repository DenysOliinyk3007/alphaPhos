"""Cardiomyocyte DVP: 5-disease ANOVA + downstream ORA on 3 layers.

End-to-end validation of the ANOVA -> enrichment path introduced in
alphaPhos 0.17.0.  Runs on the cardiomyocyte DVP dataset (Mann lab
single-cell DVP, 5 disease groups × 3 tissue regions) with three parallel
layers of biological signal:

- **proteome**       -- Spectronaut short-format protein-group quant
- **phospho**        -- Spectronaut PSMs -> site-level collapse
- **normalized**     -- phospho intensities / matched-sample parent-protein
                        intensity (log2 subtraction; removes protein-
                        abundance confounding from phospho fold-changes)

For each layer we run:

1. ``ap.diff_exp_anova`` -- moderated F-test across all 5 disease groups.
2. ``ap.enrichment.pathway_enrichment(direction="any")`` -- gene-level
   Enrichr ORA on Hallmark (direction-agnostic; ANOVA has no sign).
3. ``ap.enrichment.ora`` on the PTM-DB site-set GMTs (phospho + normalized
   only; proteome has no per-site annotation).

Prints hit-count summary + top-10 enriched terms per layer.

Run from the repo root::

    python examples/cardio_anova.py
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import alphaphos as ap

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / "test_data" / "proteome_path"
OUT = DATA / "cardio_output"
OUT.mkdir(exist_ok=True)

PHOSPHO_PARQUET = DATA / "cardiomyocytes_dvp_phospho_raw.parquet"
PROTEOME_SHORT = DATA / "cardiomyocytes_dvp_proteome_spectronaut_short_parquet.parquet"
CONDITION_CSV = DATA / "cardiomyocytes_condition_df.csv"
PTM_LIBS = OUT / "ptm_libs"  # pre-built PTM DB GMTs

print(f"alphaPhos {ap.__version__}")
print(f"Data:  {DATA}")
print(f"Out:   {OUT}")

# ---------------------------------------------------------------------------
# 1. Read + parse conditions
# ---------------------------------------------------------------------------

cond = pd.read_csv(CONDITION_CSV)
cond_prot = cond.copy()
cond_prot["sample"] = cond_prot["sample"].str.replace("phosphoDVP", "proteomeDVP", regex=False)


def _parse_condition(c: object) -> dict[str, str | None]:
    if not isinstance(c, str) or c == "BLANK_BLANK_BLANK" or not c:
        return {
            "patient": None,
            "disease": "BLANK" if isinstance(c, str) and c else None,
            "region": None,
        }
    parts = c.split("_")
    if len(parts) == 3:
        return {"patient": parts[0], "disease": parts[1], "region": parts[2]}
    return {"patient": None, "disease": None, "region": None}


def _attach_and_filter(adata):
    parsed = adata.obs["condition"].map(_parse_condition).apply(pd.Series)
    for col in ("patient", "disease", "region"):
        adata.obs[col] = parsed[col].values
    keep = adata.obs["disease"].notna() & (adata.obs["disease"] != "BLANK")
    return adata[keep].copy()


psm_df = ap.read_spectronaut(PHOSPHO_PARQUET)
adata_phos = ap.collapse_sites(psm_df, condition_df=cond)
adata_phos = _attach_and_filter(adata_phos)

adata_prot = ap.proteome.read_spectronaut_short(PROTEOME_SHORT, condition_df=cond_prot)
adata_prot = _attach_and_filter(adata_prot)

print("\nRaw shapes (after BLANK drop + condition parse)")
print(f"  phospho:  {adata_phos.shape}")
print(f"  proteome: {adata_prot.shape}")
print(f"  disease levels: {sorted(adata_phos.obs['disease'].unique())}")

# ---------------------------------------------------------------------------
# 2. Filter + impute per layer
# ---------------------------------------------------------------------------


def _filter_impute(adata):
    a = ap.filter_by_completeness(
        adata,
        min_valid_frac=2 / 3,
        group_column="disease",
        keep_strategy="any",
        layer="intensity_log2",
    )
    a = ap.impute_hybrid(a)
    a.X = a.layers["intensity_log2"].copy()
    return a


phos_ready = _filter_impute(adata_phos)
prot_ready = _filter_impute(adata_prot)

# ---------------------------------------------------------------------------
# 3. Normalized layer: phospho / parent-protein (log2 subtraction)
# ---------------------------------------------------------------------------

adata_norm = ap.proteome.phospho_over_proteome(
    adata_phos,
    adata_prot,
    sample_pairing="auto",
    missing_protein="drop",
    protein_group_policy="first",
)
adata_norm = _attach_and_filter(adata_norm)
norm_ready = _filter_impute(adata_norm)

layers = {
    "proteome": prot_ready,
    "phospho": phos_ready,
    "normalized": norm_ready,
}

print("\nPost-filter-impute shapes")
for name, a in layers.items():
    n_nan = int(np.isnan(a.X).sum())
    disease_counts = a.obs["disease"].value_counts().to_dict()
    print(f"  {name:11s}  {a.shape}   nan={n_nan}   diseases={disease_counts}")

# ---------------------------------------------------------------------------
# 4. ANOVA per layer (moderated F across 5 disease levels)
# ---------------------------------------------------------------------------

anova = {}
print("\nANOVA (moderated F across all 5 disease groups)")
for name, a in layers.items():
    r = ap.diff_exp_anova(a, condition_column="disease", layer="intensity_log2")
    n_sig05 = int((r["fdr"] < 0.05).sum())
    n_sig01 = int((r["fdr"] < 0.01).sum())
    anova[name] = r
    print(
        f"  {name:11s}  tested={len(r):<5d}  fdr<0.05={n_sig05:<5d}  "
        f"fdr<0.01={n_sig01:<5d}  top_F={r['F'].max():.2f}"
    )
    r.to_csv(OUT / f"anova_{name}.tsv", sep="\t")

# ---------------------------------------------------------------------------
# 5. Gene-level pathway ORA per layer -- direction="any" (ANOVA-friendly)
# ---------------------------------------------------------------------------

HALL = ["MSigDB_Hallmark_2020"]

# For the proteome branch, attach gene symbols from adata.var["PG_Genes"].
r_prot = anova["proteome"].copy()
if "PG_Genes" in prot_ready.var.columns:
    r_prot["gene"] = prot_ready.var.loc[r_prot.index, "PG_Genes"].values
else:
    r_prot["gene"] = ""

pathway_inputs = {
    "proteome": (r_prot, {"gene_column": "gene", "stat_col": None}),
    "phospho": (anova["phospho"], {"stat_col": None}),
    "normalized": (anova["normalized"], {"stat_col": None}),
}

print("\nGene-level pathway ORA (pathway_enrichment, direction='any', Hallmark)")
pathway_results = {}
for name, (r, kwargs) in pathway_inputs.items():
    try:
        pe = ap.enrichment.pathway_enrichment(
            r,
            fdr_threshold=0.05,
            direction="any",
            libraries=HALL,
            organism="human",
            **kwargs,
        )
    except Exception as e:  # network / API blip
        print(f"  {name:11s}  SKIPPED  ({type(e).__name__}: {e})")
        continue
    pathway_results[name] = pe
    n_sig = int((pe["fdr"] < 0.05).sum())
    n_fg = pe.attrs.get("provenance", {}).get("n_foreground_per_direction", {}).get("any", 0)
    print(f"  {name:11s}  tested={len(pe):<4d}  sig(fdr<0.05)={n_sig:<3d}  foreground={n_fg} genes")
    pe.to_csv(OUT / f"pathway_ora_anova_{name}.tsv", sep="\t", index=False)

# ---------------------------------------------------------------------------
# 6. Site-level ORA on PTM-DB (phospho + normalized only; proteome has no sites)
# ---------------------------------------------------------------------------

print("\nSite-level ORA on PTM-DB (ap.enrichment.ora + anova_hits)")

if not PTM_LIBS.exists() or not any(PTM_LIBS.glob("*.gmt")):
    print(f"  SKIPPED  no PTM libraries at {PTM_LIBS} -- rebuild with ap.emit_libraries")
else:
    site_ora_results = {}
    for name in ("phospho", "normalized"):
        r = anova[name]
        hits_raw, bg_raw = ap.anova_hits(r, fdr_threshold=0.05)
        # alphaPhos site keys (Protein|Gene|Site|Mult) -> canonical
        # Protein_AApos IDs that index the PTM-DB GMT libraries.
        hits = ap.enrichment.canonicalise_site_ids(hits_raw)
        bg = ap.enrichment.canonicalise_site_ids(bg_raw)
        # Ensure hits subset of bg (canonicalisation can drop keys unevenly).
        hits = [h for h in hits if h in set(bg)]
        try:
            ora_df = ap.enrichment.ora(hits, bg, libraries=PTM_LIBS)
        except Exception as e:
            print(f"  {name:11s}  SKIPPED  ({type(e).__name__}: {e})")
            continue
        site_ora_results[name] = ora_df
        n_tested = len(ora_df)
        n_sig = int((ora_df["fdr"] < 0.05).sum()) if "fdr" in ora_df.columns else 0
        print(
            f"  {name:11s}  hits={len(hits):<5d}  bg={len(bg):<5d}  "
            f"sets_tested={n_tested:<4d}  sig(fdr<0.05)={n_sig}"
        )
        ora_df.to_csv(OUT / f"site_ora_anova_{name}.tsv", sep="\t", index=False)

    if site_ora_results:
        print("\nTop-5 enriched PTM-DB site sets per phospho layer")
        for name, ora_df in site_ora_results.items():
            top = ora_df.sort_values("fdr").head(5)
            cols = [
                c
                for c in (
                    "library",
                    "set_name",
                    "n_overlap",
                    "n_set",
                    "log2_fold_enrichment",
                    "direction",
                    "fdr",
                )
                if c in top.columns
            ]
            print(f"\n  [{name}]")
            if len(top):
                view = top[cols].copy()
                if "log2_fold_enrichment" in view.columns:
                    view["log2_fold_enrichment"] = view["log2_fold_enrichment"].round(2)
                if "fdr" in view.columns:
                    view["fdr"] = view["fdr"].apply(lambda v: f"{v:.2e}")
                print(view.to_string(index=False))
            else:
                print("    (no enriched sets)")

# ---------------------------------------------------------------------------
# 7. Top hits summary per layer
# ---------------------------------------------------------------------------

print("\nTop-10 ANOVA hits per layer (by F stat)")
for name, r in anova.items():
    top = r.sort_values("F", ascending=False).head(10)
    if name == "proteome":
        # Show gene symbols where available.
        top = top.copy()
        top["gene"] = prot_ready.var.loc[top.index, "PG_Genes"].values
        cols = ["gene", "F", "fdr"]
    else:
        # Phospho + normalized carry gene in the site key.
        top = top.copy()
        top["gene"] = top.index.to_series().str.split("|").str[1]
        cols = ["gene", "F", "fdr"]
    view = top[cols].copy()
    view["F"] = view["F"].round(2)
    view["fdr"] = view["fdr"].apply(lambda v: f"{v:.2e}")
    print(f"\n  [{name}]")
    print(view.to_string())

if pathway_results:
    print("\nTop-5 enriched Hallmark pathways per layer (pathway_enrichment, direction='any')")
    for name, pe in pathway_results.items():
        top = pe.sort_values("fdr").head(5)
        cols = [c for c in ("term", "overlap", "odds_ratio", "fdr") if c in top.columns]
        print(f"\n  [{name}]")
        if len(top):
            print(top[cols].to_string(index=False))
        else:
            print("    (no enriched terms)")

print("\nDone.")
