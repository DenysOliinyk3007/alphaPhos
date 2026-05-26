"""Phase 1 smoke test — does alphapepttools' downstream pipeline reproduce
the R-limma reference on the EGF benchmark dataset?

Pipeline:
  cached aphos_only_MS2_sum_classI.parquet
       -> to_anndata (samples × sites, with condition obs + loc layer)
       -> apt.pp.filter_data_completeness (drop sites failing 70% in any condition)
       -> apt.pp.impute_knn  (group_column='condition', n_neighbors=2)
       -> apt.tl.diff_exp_ebayes (limma via inmoose)
       -> compare against test_data/benchmark_output/limma/aphos_sum_classI_topTable.tsv

Reports: feature overlap, logFC concordance (Pearson/Spearman), adj.P
concordance, significant-hit Jaccard.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import alphapepttools as apt
from alphaphos.preprocess import to_anndata

# ------------------------------------------------------------------ paths
ROOT = Path("D:/Projects/alphaPhos")
SITES_PARQUET = ROOT / "test_data/benchmark_output/aphos_only_MS2_sum_classI.parquet"
LOC_PARQUET = ROOT / "test_data/benchmark_output/aphos_only_MS2_loc_per_run.parquet"
R_LIMMA_REF = ROOT / "test_data/benchmark_output/limma/aphos_sum_classI_topTable.tsv"

SAMPLES = [
    "20250729_OA4_Evo11_16p3min_DeOl_SA_nanoPhos_dilser_new_withEGF_1000ng_01",
    "20250729_OA4_Evo11_16p3min_DeOl_SA_nanoPhos_dilser_new_withEGF_1000ng_02",
    "20250729_OA4_Evo11_16p3min_DeOl_SA_nanoPhos_dilser_new_withEGF_1000ng_03",
    "20250729_OA4_Evo11_16p3min_DeOl_SA_nanoPhos_dilser_new_woEGF_1000ng_01",
    "20250729_OA4_Evo11_16p3min_DeOl_SA_nanoPhos_dilser_new_woEGF_1000ng_02",
    "20250729_OA4_Evo11_16p3min_DeOl_SA_nanoPhos_dilser_new_woEGF_1000ng_03",
]
CONDITION_DF = pd.DataFrame(
    {"sample": SAMPLES, "condition": ["withEGF"] * 3 + ["woEGF"] * 3}
)


def main():
    # ----------------------------------------------------- 1. inputs
    sites = pd.read_parquet(SITES_PARQUET)
    loc = pd.read_parquet(LOC_PARQUET)
    print(f"sites: {sites.shape}; loc_per_run: {loc.shape}")

    # ----------------------------------------------------- 2. AnnData
    adata = to_anndata(
        sites,
        loc_per_run=loc,
        condition_df=CONDITION_DF,
    )
    print(f"adata: {adata.n_obs} samples × {adata.n_vars} sites")
    print(f"  obs cols: {list(adata.obs.columns)}")
    print(f"  var cols: {list(adata.var.columns)}")
    print(f"  layers:   {list(adata.layers.keys())}")
    print(f"  uns keys: {list(adata.uns.keys())}")
    print()

    # ----------------------------------------------------- 3. filter completeness
    # Match Dublin's "70% valid in at least one condition" -> max_missing=0.3
    # in any one group.
    print("Filtering by data completeness (max_missing=0.3, any condition) ...")
    adata = apt.pp.filter_data_completeness(
        adata,
        max_missing=0.3,
        group_column="condition",
        keep_strategy="any",
        action="drop",
    )
    print(f"  -> {adata.n_vars} sites kept")
    print()

    # ----------------------------------------------------- 4. impute
    # Dublin's impute_phosphosites runs KNN GLOBALLY (not per-condition) —
    # required because our keep_strategy='any' filter permits sites that are
    # 100% missing in one of the two conditions. Per-condition imputation
    # would fail on those.
    print("Imputing (KNN, n_neighbors=2, GLOBAL — no group_column) ...")
    apt.pp.impute_knn(adata, n_neighbors=2)
    n_nan_after = int(np.isnan(adata.X).sum())
    print(f"  remaining NaN cells after imputation: {n_nan_after}")
    print()

    # ----------------------------------------------------- 5. differential
    # We try ebayes (inmoose-backed limma) first; on Windows without MSVC
    # build tools, inmoose can't install, in which case fall back to the
    # alphapepttools NaN-safe t-test for the smoke comparison.
    backend = None
    ebayes = None
    try:
        print("Running apt.tl.diff_exp_ebayes (limma via inmoose) ...")
        _comp_id, ebayes = apt.tl.diff_exp_ebayes(
            adata,
            between_column="condition",
            comparison=("woEGF", "withEGF"),
        )
        backend = "ebayes"
    except ImportError as exc:
        print(f"  ebayes unavailable: {exc}")
        print()
        print("Falling back to apt.tl.diff_exp_ttest (scipy, no C++ build needed) ...")
        ebayes = apt.tl.diff_exp_ttest(
            adata,
            between_column="condition",
            comparison=("woEGF", "withEGF"),
        )
        backend = "ttest"
    print(f"  backend used:  {backend}")
    print(f"  result rows:   {len(ebayes)}")
    print(f"  result cols:   {list(ebayes.columns)}")
    print()

    # ----------------------------------------------------- 6. R-limma reference
    r_ref = pd.read_csv(R_LIMMA_REF, sep="\t")
    print(f"R-limma reference: {len(r_ref)} rows, cols: {list(r_ref.columns)}")
    print()

    # ----------------------------------------------------- 7. align via feature id
    # ebayes path: 'feature' index column.
    # ttest  path: 'protein' column holds the feature name; index is RangeIndex.
    ebayes_keyed = ebayes.copy()
    if "feature" in ebayes_keyed.columns:
        ebayes_keyed = ebayes_keyed.set_index("feature")
    elif "protein" in ebayes_keyed.columns:
        ebayes_keyed = ebayes_keyed.set_index("protein")

    r_ref_keyed = r_ref.set_index("feature")

    # Normalize column names — alphapepttools backends use different conventions
    def _pick(df, *candidates):
        for c in candidates:
            if c in df.columns:
                return df[c]
        raise KeyError(f"none of {candidates} in {df.columns.tolist()}")

    e_logfc = _pick(ebayes_keyed, "logFC", "log_fc", "log2fc")
    e_padj = _pick(ebayes_keyed, "adj.P.Val", "adj_p_value", "fdr")
    e_pval = _pick(ebayes_keyed, "P.Value", "p_value")

    r_logfc = r_ref_keyed["logFC"]
    r_padj = r_ref_keyed["adj.P.Val"]
    r_pval = r_ref_keyed["P.Value"]

    shared = e_logfc.index.intersection(r_logfc.index)
    print(f"Shared features: {len(shared)} (ebayes={len(e_logfc)}, R={len(r_logfc)})")

    pair = pd.DataFrame({
        "logfc_apt": e_logfc.loc[shared],
        "logfc_r":   r_logfc.loc[shared],
        "padj_apt":  e_padj.loc[shared],
        "padj_r":    r_padj.loc[shared],
        "pval_apt":  e_pval.loc[shared],
        "pval_r":    r_pval.loc[shared],
    }).dropna()
    print(f"After dropna: {len(pair):,} comparable rows")
    print()

    # ----------------------------------------------------- 8. concordance
    pearson_logfc = float(pair["logfc_apt"].corr(pair["logfc_r"]))
    spear_logfc = float(pair["logfc_apt"].corr(pair["logfc_r"], method="spearman"))
    med_abs_diff = float((pair["logfc_apt"] - pair["logfc_r"]).abs().median())
    print("logFC concordance (alphapepttools vs R-limma):")
    print(f"  Pearson r:    {pearson_logfc:.6f}")
    print(f"  Spearman r:   {spear_logfc:.6f}")
    print(f"  median |diff|: {med_abs_diff:.6f}")
    print()

    pearson_padj = float(pair["padj_apt"].corr(pair["padj_r"]))
    spear_padj = float(pair["padj_apt"].corr(pair["padj_r"], method="spearman"))
    print("adj.P.Val concordance:")
    print(f"  Pearson r:    {pearson_padj:.6f}")
    print(f"  Spearman r:   {spear_padj:.6f}")
    print()

    # ----------------------------------------------------- 9. sig-hit overlap
    sig_apt = set(pair.index[(pair["padj_apt"] < 0.05) & (pair["logfc_apt"].abs() > 0.585)])
    sig_r = set(pair.index[(pair["padj_r"] < 0.05) & (pair["logfc_r"].abs() > 0.585)])
    inter = sig_apt & sig_r
    union = sig_apt | sig_r
    print("Significant hits (adj.P<0.05 & |logFC|>0.585):")
    print(f"  alphapepttools:  {len(sig_apt):>5,}")
    print(f"  R-limma:         {len(sig_r):>5,}")
    print(f"  shared:          {len(inter):>5,}")
    print(f"  apt-only:        {len(sig_apt - sig_r):>5,}")
    print(f"  R-only:          {len(sig_r - sig_apt):>5,}")
    print(f"  Jaccard:         {len(inter)/max(len(union),1):.4f}")
    print()

    # ----------------------------------------------------- 10. show disagreements
    only_apt = sig_apt - sig_r
    only_r = sig_r - sig_apt
    if only_apt or only_r:
        print("Top apt-only hits (by padj_apt):")
        print(pair.loc[list(only_apt)].sort_values("padj_apt").head(10)[["logfc_apt","logfc_r","padj_apt","padj_r"]].to_string())
        print()
        print("Top R-only hits (by padj_r):")
        print(pair.loc[list(only_r)].sort_values("padj_r").head(10)[["logfc_apt","logfc_r","padj_apt","padj_r"]].to_string())


if __name__ == "__main__":
    main()
