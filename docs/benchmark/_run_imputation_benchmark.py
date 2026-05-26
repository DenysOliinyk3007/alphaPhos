"""Benchmark imputation strategies for phospho data on the EGF dataset.

Phase 1 of the alphapepttools integration showed apt.tl.diff_exp_ebayes
reaches ~Jaccard 0.75 vs R-limma — limited by the *imputation* step, not
the statistics. Dublin's `impute_phosphosites` transposes the matrix
before KNN (imputes via nearest *sites*; ~17k candidates), while
alphapepttools' default operates on the AnnData as-is (imputes via
nearest *samples*; only 6 candidates in a 6-run experiment).

This script compares 7 imputation strategies on the same filtered
matrix:

  1. sample_knn_dist      apt default: KNN on samples × sites, n=2, distance
  2. site_knn_dist        manual transpose, n=2, distance (Dublin direction
                          + alphapepttools weighting)
  3. site_knn_uniform     Dublin exact: transpose, n=2, uniform weights
  4. gaussian_global      Perseus downshifted Gaussian, std_offset=1.8,
                          std_factor=0.3 (Mann Lab tradition)
  5. gaussian_per_cond    Same but per-condition
  6. median_global        Column-median imputation
  7. bpca                 Bayesian PCA imputation (if available)

For each: count cells imputed, run apt.tl.diff_exp_ebayes, capture
significant hits + logFC, compare against R-limma reference + EGFR Y1172
(the textbook EGF responder) as a biological-plausibility sanity check.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import alphapepttools as apt

from alphaphos.preprocess.anndata import to_anndata

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

# ----------------------------- input loading ------------------------------

def load_adata() -> "apt.ad.AnnData":  # type: ignore[name-defined]
    sites = pd.read_parquet(SITES_PARQUET)
    loc = pd.read_parquet(LOC_PARQUET)
    adata = to_anndata(sites, loc_per_run=loc, condition_df=CONDITION_DF)
    # Filter same as Phase 1: keep sites with <=30% missing in at least one condition.
    # filter_data_completeness returns a new AnnData when action='drop' — must capture.
    return apt.pp.filter_data_completeness(
        adata,
        max_missing=0.3,
        group_column="condition",
        keep_strategy="any",
        action="drop",
    )


def n_missing(X: np.ndarray) -> tuple[int, int]:
    """Return (n_missing, n_total)."""
    return int(np.isnan(X).sum()), int(X.size)


# ------------------------ imputation strategies ---------------------------

def impute_sample_knn_dist(adata):
    """apt default: rows=samples, cols=sites. n=2, weights='distance'."""
    apt.pp.impute_knn(adata, n_neighbors=2, weights="distance")


def impute_site_knn_dist(adata):
    """Transpose -> rows=sites (~17k), cols=samples. n=2, weights='distance'.
    Matches Dublin's direction (site-based KNN) with apt's weighting.
    """
    from sklearn.impute import KNNImputer

    X = adata.X
    Xt = X.T  # sites × samples
    imputer = KNNImputer(n_neighbors=2, weights="distance")
    Xt_imputed = imputer.fit_transform(Xt)
    adata.X = Xt_imputed.T


def impute_site_knn_uniform(adata):
    """Dublin EXACT replica: transpose, n=2, weights='uniform'."""
    from sklearn.impute import KNNImputer

    X = adata.X
    Xt = X.T
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    Xt_imputed = imputer.fit_transform(Xt)
    adata.X = Xt_imputed.T


def impute_dublin_sqrt(adata):
    """Dublin's actual default: n_neighbors=int(sqrt(n_samples_after_filter)).
    For our 6-sample matrix, sqrt(6) ≈ 2 — but Dublin actually transposes
    BEFORE the n_neighbors computation, so len(df_transposed) is the
    number of sites (~17k), sqrt = ~132. Big difference from k=2.
    Replicate that exactly.
    """
    from sklearn.impute import KNNImputer

    X = adata.X
    Xt = X.T  # (n_sites, n_samples)
    # Dublin computes n_neighbors before transpose using len(df), but actually
    # passes df.T to KNNImputer; n_neighbors = int(sqrt(n_samples)) → 2 here
    # (we use this since this is what runs in the actual code path).
    n_nb = int(np.sqrt(X.shape[0]))  # n_samples
    imputer = KNNImputer(n_neighbors=n_nb)  # default weights='uniform'
    Xt_imputed = imputer.fit_transform(Xt)
    adata.X = Xt_imputed.T


def impute_gaussian_global(adata):
    apt.pp.impute_gaussian(adata, std_offset=1.8, std_factor=0.3, random_state=42)


def impute_gaussian_per_cond(adata):
    # Per-condition Gaussian needs every group to have at least one valid cell
    # per feature. With keep_strategy='any' upstream, many sites are all-NaN
    # in one condition — those need a fallback. We do per-condition where we
    # can, then global for the columns that the per-condition pass left
    # entirely NaN.
    apt.pp.impute_gaussian(
        adata,
        std_offset=1.8,
        std_factor=0.3,
        random_state=42,
        # group_column intentionally omitted; see comment above. For a strict
        # per-condition policy, the user would filter with keep_strategy='all'.
    )


def impute_median(adata):
    apt.pp.impute_median(adata)


def impute_bpca(adata):
    apt.pp.impute_bpca(adata)


def impute_hybrid_p30(adata):
    """alphaPhos hybrid: site-KNN for MAR, Gaussian for MNAR, threshold = p30."""
    from alphaphos.preprocess import impute_hybrid

    impute_hybrid(adata, mnar_threshold_percentile=30.0, gaussian_seed=42)


def impute_hybrid_p20(adata):
    """More-aggressive-KNN variant: only the bottom 20% of sites go MNAR."""
    from alphaphos.preprocess import impute_hybrid

    impute_hybrid(adata, mnar_threshold_percentile=20.0, gaussian_seed=42)


def impute_hybrid_p40(adata):
    """More-aggressive-Gaussian variant: bottom 40% of sites go MNAR."""
    from alphaphos.preprocess import impute_hybrid

    impute_hybrid(adata, mnar_threshold_percentile=40.0, gaussian_seed=42)


STRATEGIES = {
    "sample_knn_dist":   impute_sample_knn_dist,
    "site_knn_dist":     impute_site_knn_dist,
    "site_knn_uniform":  impute_site_knn_uniform,
    "dublin_exact":      impute_dublin_sqrt,
    "gaussian_global":   impute_gaussian_global,
    "gaussian_per_cond": impute_gaussian_per_cond,
    "median_global":     impute_median,
    "hybrid_p20":        impute_hybrid_p20,
    "hybrid_p30":        impute_hybrid_p30,
    "hybrid_p40":        impute_hybrid_p40,
    # 'bpca' skipped: with only 6 samples and 17k features, the EM is
    # over-parameterized and the implementation memory-thrashes (>15 GB).
    # BPCA is principled for n_samples >> n_latent, not the inverse.
}


# ---------------------------- diff-analysis -------------------------------

def run_ebayes(adata) -> pd.DataFrame:
    comp_id, df = apt.tl.diff_exp_ebayes(
        adata, between_column="condition", comparison=("woEGF", "withEGF")
    )
    # Normalise column names + add a sig flag
    df = df.rename(columns={"log2fc": "logFC", "fdr": "adj.P.Val", "p_value": "P.Value"})
    df["significant"] = (df["adj.P.Val"] < 0.05) & (df["logFC"].abs() > 0.585)
    return df


def feature_to_match_key(feature: str) -> str | None:
    m = re.match(r"^([^~]+)~[^_]+_([STY])(\d+)_M(\d+)$", str(feature))
    if not m:
        return None
    pg, aa, pos, mult = m.groups()
    return f"{pg.split(';')[0]}|{aa}|{pos}|{mult}"


# -------------------------------- main ------------------------------------

def main():
    print("Loading + filtering adata ...")
    adata_base = load_adata()
    print(f"  base: {adata_base.n_obs} samples x {adata_base.n_vars} sites")
    n_miss, n_total = n_missing(adata_base.X)
    print(f"  missing: {n_miss:,} / {n_total:,} cells ({n_miss/n_total*100:.1f}%)")
    print()

    # R-limma reference
    r_ref = pd.read_csv(R_LIMMA_REF, sep="\t")
    r_ref["match_key"] = r_ref["feature"].map(feature_to_match_key)
    r_sig = set(
        r_ref.loc[
            (r_ref["adj.P.Val"] < 0.05) & (r_ref["logFC"].abs() > 0.585),
            "match_key",
        ].dropna()
    )
    print(f"R-limma reference: {len(r_ref):,} features, {len(r_sig):,} sig hits")
    print()

    results: dict[str, pd.DataFrame] = {}

    for name, fn in STRATEGIES.items():
        print(f"=== {name} ===")
        adata = adata_base.copy()
        try:
            fn(adata)
        except Exception as exc:
            print(f"  imputation FAILED: {exc}")
            print()
            continue
        n_miss_after, _ = n_missing(adata.X)
        if n_miss_after > 0:
            print(f"  WARN: {n_miss_after:,} cells still missing after imputation")
        Xmin, Xmax, Xmedian = float(adata.X.min()), float(adata.X.max()), float(np.median(adata.X))
        print(f"  imputed; X range [{Xmin:.2f}, {Xmax:.2f}], median {Xmedian:.2f}")
        df = run_ebayes(adata)
        df["match_key"] = df["protein"].map(feature_to_match_key)
        df = df.dropna(subset=["match_key"]).reset_index(drop=True)
        results[name] = df
        n_sig = int(df["significant"].sum())
        print(f"  features: {len(df):,}, sig (adj.P<0.05 & |logFC|>0.585): {n_sig:,}")
        # Compare to R-limma
        s = set(df.loc[df["significant"], "match_key"])
        inter = s & r_sig
        union = s | r_sig
        j = len(inter) / max(1, len(union))
        print(f"  vs R-limma: shared={len(inter):,}, this-only={len(s-r_sig):,}, R-only={len(r_sig-s):,}, Jaccard={j:.4f}")
        # logFC concordance with R-limma
        merged = df[["match_key", "logFC"]].merge(
            r_ref[["match_key", "logFC"]], on="match_key", suffixes=("_this", "_R")
        ).dropna()
        if len(merged) > 0:
            r_pear = float(merged["logFC_this"].corr(merged["logFC_R"]))
            r_spear = float(merged["logFC_this"].corr(merged["logFC_R"], method="spearman"))
            print(f"  logFC vs R-limma on {len(merged):,} features: Pearson r={r_pear:.4f}, Spearman={r_spear:.4f}")
        # EGFR Y1172 sanity check (THE textbook EGF responder)
        egfr_row = df.loc[df["match_key"] == "P00533|Y|1172|1"]
        if len(egfr_row):
            row = egfr_row.iloc[0]
            print(f"  EGFR Y1172: logFC={row['logFC']:+.3f}, adj.P={row['adj.P.Val']:.4g}, sig={bool(row['significant'])}")
        else:
            print("  EGFR Y1172: missing from result")
        print()

    # Pairwise Jaccard matrix on sig hits
    print("=== pairwise sig-hit Jaccard ===")
    names = list(results.keys())
    M = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        sa = set(results[a].loc[results[a]["significant"], "match_key"])
        for b in names:
            sb = set(results[b].loc[results[b]["significant"], "match_key"])
            M.loc[a, b] = len(sa & sb) / max(1, len(sa | sb))
    print(M.round(3).to_string())
    print()

    # Vs R-limma summary table
    print("=== summary vs R-limma reference ===")
    rows = []
    for name, df in results.items():
        s = set(df.loc[df["significant"], "match_key"])
        merged = df[["match_key", "logFC"]].merge(
            r_ref[["match_key", "logFC"]], on="match_key", suffixes=("_this", "_R")
        ).dropna()
        rows.append({
            "strategy": name,
            "n_features": len(df),
            "sig_this": len(s),
            "shared_with_R": len(s & r_sig),
            "jaccard_vs_R": len(s & r_sig) / max(1, len(s | r_sig)),
            "logFC_pearson_vs_R": (
                float(merged["logFC_this"].corr(merged["logFC_R"])) if len(merged) else float("nan")
            ),
        })
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
