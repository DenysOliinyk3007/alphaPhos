"""Benchmark: alphaPhos pipeline with vs without the contaminant filter.

Same input as the SN classI benchmark (EGF nanoPhos, 6 runs, MS2 quant).
Compares two configurations:

  - WITHOUT contaminant filter:  read_psm(..., drop_contaminants=False)
  - WITH contaminant filter:     read_psm(..., drop_contaminants=True)  # default

For each: row counts at each pipeline stage, site counts, the set of sites
that disappear, top dropped genes, per-cell quant agreement against
sn_classI (the cross-tool reference), and finally limma differential
analysis (filter + impute + limma) with significant-hit overlap and logFC
concordance.

Run from the repo root with:
    python docs/benchmark/_run_contaminant_benchmark.py
"""

from __future__ import annotations

import io
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from alphaphos.io import read_spectronaut as read_psm
from alphaphos.preprocess import (
    apply_condition_aware_classI_mask,
    collapse_sites,
)

# ------------------------------------------------------------------ paths
ROOT = Path("D:/Projects/alphaPhos")
PSM_TSV = ROOT / "test_data/benchmark/EGF_report_new_ms2.tsv"
SN_CLASSI = ROOT / "test_data/benchmark/EGF_report_sn_out_classI_sites.tsv"
OUT = ROOT / "test_data/benchmark_output"
LIMMA_WRAPPER = ROOT / "docs/benchmark/limma_wrapper.R"
RSCRIPT = Path("C:/Program Files/R/R-4.5.2/bin/Rscript.exe")

# ------------------------------------------------------------ condition map
SAMPLES = [
    "20250729_OA4_Evo11_16p3min_DeOl_SA_nanoPhos_dilser_new_withEGF_1000ng_01",
    "20250729_OA4_Evo11_16p3min_DeOl_SA_nanoPhos_dilser_new_withEGF_1000ng_02",
    "20250729_OA4_Evo11_16p3min_DeOl_SA_nanoPhos_dilser_new_withEGF_1000ng_03",
    "20250729_OA4_Evo11_16p3min_DeOl_SA_nanoPhos_dilser_new_woEGF_1000ng_01",
    "20250729_OA4_Evo11_16p3min_DeOl_SA_nanoPhos_dilser_new_woEGF_1000ng_02",
    "20250729_OA4_Evo11_16p3min_DeOl_SA_nanoPhos_dilser_new_woEGF_1000ng_03",
]
CONDITION_DF = pd.DataFrame({
    "sample": SAMPLES,
    "condition": ["withEGF"] * 3 + ["woEGF"] * 3,
})


# ============================================================================
# Stage 1: PSM-level counts at each filter step
# ============================================================================

def stage1_psm_counts():
    print("### Stage 1: PSM-level row counts at each filter step\n")

    t0 = time.time()
    df_no_filter = read_psm(
        PSM_TSV,
        quant_level="MS2",
        drop_decoys=True,
        pg_qvalue_max=0.01,
        top_n_attribution=True,
        drop_contaminants=False,
    )
    t_no = time.time() - t0

    t0 = time.time()
    df_with_filter = read_psm(
        PSM_TSV,
        quant_level="MS2",
        drop_decoys=True,
        pg_qvalue_max=0.01,
        top_n_attribution=True,
        drop_contaminants=True,
    )
    t_with = time.time() - t0

    print(f"read_psm timing:  without filter = {t_no:.1f}s | with filter = {t_with:.1f}s\n")

    # Both runs go through the same filter cascade except the final
    # contaminant step. Pull stages from .attrs.
    a_no = df_no_filter.attrs
    a_with = df_with_filter.attrs

    rows = [
        ("Loaded from TSV", a_no.get("n_rows_loaded"), a_with.get("n_rows_loaded")),
        ("After q-value + decoy filters + top-N attribution",
         a_no.get("n_rows_after_top_n"), a_with.get("n_rows_after_top_n")),
        ("After contaminant filter",
         a_no.get("n_rows_after_contaminant_filter"),
         a_with.get("n_rows_after_contaminant_filter")),
        ("Final returned (= n_rows_returned)",
         a_no.get("n_rows_returned"), a_with.get("n_rows_returned")),
    ]
    table = pd.DataFrame(rows, columns=["Stage", "no_filter", "with_filter"])
    table["delta"] = table["no_filter"] - table["with_filter"]
    table["pct_dropped"] = (table["delta"] / table["no_filter"] * 100).round(2)
    print(table.to_string(index=False))
    print()
    return df_no_filter, df_with_filter


# ============================================================================
# Stage 2: which contaminants are being dropped (gene + protein distribution)
# ============================================================================

def stage2_dropped_contaminant_breakdown(df_no_filter):
    print("### Stage 2: Which contaminants are dropped\n")

    # Re-derive the mask using the same logic as filter_contaminants
    from alphaphos.preprocess.contaminants import (
        DEFAULT_CONTAMINANT_PREFIXES,
        _is_contaminant_protein,
        get_default_contaminants_fasta,
        parse_fasta_accessions,
    )
    accessions = parse_fasta_accessions(get_default_contaminants_fasta())

    def _row_is_cont(value):
        if pd.isna(value):
            return False
        proteins = [p for p in str(value).split(";") if p.strip()]
        if not proteins:
            return False
        return all(
            _is_contaminant_protein(p, accessions, DEFAULT_CONTAMINANT_PREFIXES)
            for p in proteins
        )

    mask = df_no_filter["PG.ProteinGroups"].map(_row_is_cont)
    dropped = df_no_filter.loc[mask]
    print(f"Total PSM rows that contaminant filter would drop: {len(dropped):,}\n")

    # Top dropped genes (rows are precursor-level, so one peptide -> many rows)
    print("Top 20 dropped contaminant genes by PSM-row count:")
    print(dropped["PG.Genes"].fillna("(no_gene)").value_counts().head(20).to_string())
    print()

    # Top dropped protein groups
    print("Top 20 dropped contaminant protein groups (PSM-row count):")
    print(dropped["PG.ProteinGroups"].value_counts().head(20).to_string())
    print()

    # How are they identified?
    by_prefix = dropped["PG.ProteinGroups"].str.contains(
        r"^(?:CON__|Cont_|contam_)", regex=True, na=False
    ).sum()
    by_fasta_only = len(dropped) - by_prefix
    print(f"Dropped via prefix detection (CON__/Cont_/contam_): {by_prefix:,}")
    print(f"Dropped via FASTA accession match only:             {by_fasta_only:,}")
    print()
    return dropped


# ============================================================================
# Stage 3: site-level effect — collapse to sites and compare
# ============================================================================

def stage3_site_level(df_no_filter, df_with_filter):
    print("### Stage 3: Site-level counts after collapse_sites\n")

    def _collapse(df, label):
        print(f"  collapsing {label} ...")
        t0 = time.time()
        sites, _loc = collapse_sites(
            df,
            cutoff=0.0,
            aggregation_method="sum",
            localization_strategy="condition",
            condition_df=CONDITION_DF,
            classI_cutoff=0.75,
            condition_threshold=0.50,
        )
        print(f"    -> {len(sites):,} sites ({time.time()-t0:.1f}s)")
        return sites

    sites_no = _collapse(df_no_filter, "no_filter")
    sites_with = _collapse(df_with_filter, "with_filter")

    # Build the same match_key format as the SN classI benchmark: ProteinGroup|AA|pos|mult
    def _to_match_key(sites_df):
        df = sites_df.copy()
        df.index.name = None
        # PTM_Collapse_key format: "{PG}~{Gene}_{AA}{pos}_M{mult}"
        m = df["PTM_Collapse_key"].str.extract(
            r"^(?P<pg>[^~]+)~[^_]+_(?P<aa>[STY])(?P<pos>\d+)_M(?P<mult>\d+)$"
        )
        # Take just the first protein from the group for the match key
        pg_first = m["pg"].str.split(";").str[0]
        df["match_key"] = pg_first + "|" + m["aa"] + "|" + m["pos"] + "|" + m["mult"]
        return df

    keyed_no = _to_match_key(sites_no)
    keyed_with = _to_match_key(sites_with)

    set_no = set(keyed_no["match_key"].dropna())
    set_with = set(keyed_with["match_key"].dropna())

    print()
    print(f"Sites without filter: {len(set_no):,}")
    print(f"Sites with filter:    {len(set_with):,}")
    print(f"Sites lost when filter is on: {len(set_no - set_with):,}")
    print(f"Sites gained when filter is on (sanity, should be 0): {len(set_with - set_no):,}")
    print()
    return sites_no, sites_with, keyed_no, keyed_with, set_no, set_with


# ============================================================================
# Stage 4: comparison against sn_classI
# ============================================================================

def stage4_sn_comparison(keyed_no, keyed_with):
    print("### Stage 4: Agreement against sn_classI (the cross-tool reference)\n")
    # Re-use sn_classI from the saved parquet if present; else parse the TSV
    sn_pivot = pd.read_csv(SN_CLASSI, sep="\t", low_memory=False)
    print(f"Loaded sn_classI pivot: shape {sn_pivot.shape}")

    # The SN pivot has 'PTM.Group' or similar key columns. Build the same
    # match_key by extracting (ProteinId, AA, Pos, Multiplicity).
    # Conventional SN column names:
    sn_pivot.columns = [c.strip() for c in sn_pivot.columns]

    # SN classI pivot has fixed column names; use them directly
    sn_pivot["match_key"] = (
        sn_pivot["PTM.ProteinId"].astype(str).str.split(";").str[0]
        + "|" + sn_pivot["PTM.SiteAA"].astype(str)
        + "|" + sn_pivot["PTM.SiteLocation"].astype(str)
        + "|" + sn_pivot["PTM.Multiplicity"].astype(str)
    )
    sn_pivot = sn_pivot.set_index("match_key")

    # Quant columns: end with '.PTM.Quantity' AND contain '.raw.' (per-sample);
    # the bracketed [1]..[6] indices are just label prefixes Spectronaut writes.
    quant_cols = [c for c in sn_pivot.columns if c.endswith(".PTM.Quantity") and ".raw." in c]
    print(f"  sn_classI quant cols: {len(quant_cols)} samples")

    # Map SN column names back to short sample names (matching CONDITION_DF['sample'])
    def _normalize_sn_col(c):
        # e.g. '[1] 20250729_OA4_...withEGF_1000ng_01.raw.PTM.Quantity' -> '20250729...01'
        c = re.sub(r"^\[\d+\]\s*", "", c)
        return c.replace(".raw.PTM.Quantity", "")

    rename_sn = {c: _normalize_sn_col(c) for c in quant_cols}
    # SN writes 'Filtered' (string) for cells below the Class I threshold;
    # pd.to_numeric coerces those to NaN cleanly.
    quant_df = sn_pivot[quant_cols].rename(columns=rename_sn).apply(
        lambda s: pd.to_numeric(s, errors="coerce")
    )
    sn_log2 = np.log2(quant_df)
    sn_log2 = sn_log2.replace([np.inf, -np.inf], np.nan)
    print(f"  sn_classI sites: {len(sn_log2):,}, samples: {sn_log2.shape[1]}")
    print()

    # Reduce aphos sites to sample columns + match_key
    def _aphos_long(keyed):
        keyed = keyed.set_index("match_key")
        # the wide-form columns include the sample names (R.FileName values).
        # The condition_df['sample'] entries should match.
        aphos_samples = [c for c in keyed.columns if c in SAMPLES]
        return keyed[aphos_samples]

    A_no = _aphos_long(keyed_no)
    A_with = _aphos_long(keyed_with)

    # Align A_no and sn_log2 on samples (re-index columns)
    sn_aligned = sn_log2.reindex(columns=SAMPLES)

    def _agreement(label, A):
        shared = A.index.intersection(sn_aligned.index)
        a = A.loc[shared]
        b = sn_aligned.loc[shared]
        both = pd.DataFrame({"a": a.values.flatten(), "b": b.values.flatten()}).dropna()
        if len(both) < 50:
            return None
        diff = both["a"] - both["b"]
        return {
            "config":       label,
            "n_sites":      len(A),
            "shared_keys":  len(shared),
            "jaccard":      len(set(A.index) & set(sn_aligned.index)) / max(
                              1, len(set(A.index) | set(sn_aligned.index))
                            ),
            "paired_cells": len(both),
            "pearson_log2": round(float(both["a"].corr(both["b"])), 4),
            "mean_log2_diff":   round(float(diff.mean()), 4),
            "median_log2_diff": round(float(diff.median()), 4),
            "sd_log2_diff":     round(float(diff.std()), 4),
            "pct_within_0.1":   round(float((diff.abs() < 0.1).mean() * 100), 2),
        }

    rows = [_agreement("no_filter (aphos+sum)", A_no),
            _agreement("with_filter (aphos+sum)", A_with)]
    rows = [r for r in rows if r is not None]
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    return sn_log2


# ============================================================================
# Stage 5: differential analysis — does the filter change limma hits?
# ============================================================================

def stage5_limma(sites_no, sites_with):
    print("### Stage 5: limma differential analysis (filter + impute + limma)\n")

    # Local imports: borrow Dublin's filter/impute helpers via sys.path; this
    # mirrors what §7 of the benchmark notebook does.
    import sys
    sys.path.insert(0, "D:/Projects/Dublin/testscripts/src")
    from core import filter_phosphosites, impute_phosphosites  # type: ignore  # noqa

    # Build samples-as-rows form (the form filter_phosphosites + limma_wrapper want)
    def _to_wide(sites_df):
        # Sample columns are the file-name-like strings; condition col goes alongside
        cols = [c for c in sites_df.columns if c in SAMPLES]
        out = sites_df[cols].T  # transpose → rows = samples, cols = sites
        out.index.name = "sample"
        # column names → simple "ProtGroup~AAPos_M{m}" form, but renormalize the
        # separators so filter_phosphosites' '~' detection works
        new_cols = []
        for c in sites_df["PTM_Collapse_key"]:
            # e.g. "A0A087WUV0~ZNF892_S118_M1" already has '~' separator
            new_cols.append(c.replace("|", "~"))
        out.columns = new_cols
        out["condition"] = out.index.map(dict(zip(CONDITION_DF["sample"], CONDITION_DF["condition"])))
        return out

    wide_no = _to_wide(sites_no)
    wide_with = _to_wide(sites_with)

    def _run_pipeline(wide, label):
        print(f"  [{label}] filtering ...")
        filt = filter_phosphosites(wide, how="condition", cutoff=0.7, condition_col="condition")
        n_kept = sum("~" in str(c) for c in filt.columns)
        print(f"    sites kept after filter: {n_kept:,}")

        print(f"  [{label}] imputing ...")
        imputed = impute_phosphosites(filt)

        print(f"  [{label}] running limma ...")
        # Write to temp expr/meta files, call R, parse top table
        expr_path = OUT / f"contamtest_{label}_expr.tsv"
        meta_path = OUT / f"contamtest_{label}_meta.tsv"
        top_path = OUT / f"contamtest_{label}_topTable.tsv"

        site_cols = [c for c in imputed.columns if "~" in str(c)]
        expr = imputed[site_cols].T
        expr.index.name = "feature"
        expr.to_csv(expr_path, sep="\t")

        # limma_wrapper.R expects meta column "sample_id" not "sample"
        meta = pd.DataFrame(
            {"sample_id": imputed.index, "condition": imputed["condition"].values}
        )
        meta.to_csv(meta_path, sep="\t", index=False)

        # Wrapper signature: Rscript limma_wrapper.R <expr> <meta> <group_col> <out>
        result = subprocess.run(
            [str(RSCRIPT), str(LIMMA_WRAPPER),
             str(expr_path), str(meta_path), "condition", str(top_path)],
            check=False, capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"    [limma] FAILED: rc={result.returncode}")
            print(f"    stderr: {result.stderr[:500]}")
            raise RuntimeError("limma_wrapper.R failed")
        top = pd.read_csv(top_path, sep="\t")
        top.columns = [c.strip() for c in top.columns]
        if "feature" not in top.columns and top.columns[0] != "feature":
            top = top.rename(columns={top.columns[0]: "feature"})
        top["significant"] = (top["adj.P.Val"] < 0.05) & (top["logFC"].abs() > 0.585)
        print(f"    features: {len(top):,}, sig: {int(top['significant'].sum()):,}")
        return top

    top_no = _run_pipeline(wide_no, "no_filter")
    top_with = _run_pipeline(wide_with, "with_filter")

    # Compare
    def _match_key(feat):
        # feature ids look like "ProteinGroup~Gene_AA{pos}_M{mult}"
        m = re.match(r"^([^~]+)~[^_]+_([STY])(\d+)_M(\d+)$", feat)
        if not m:
            return None
        pg, aa, pos, mult = m.groups()
        pg_first = pg.split(";")[0]
        return f"{pg_first}|{aa}|{pos}|{mult}"

    top_no["match_key"] = top_no["feature"].map(_match_key)
    top_with["match_key"] = top_with["feature"].map(_match_key)

    sig_no = set(top_no.loc[top_no["significant"], "match_key"].dropna())
    sig_with = set(top_with.loc[top_with["significant"], "match_key"].dropna())
    inter = sig_no & sig_with
    print()
    print("Significant hits (adj.P<0.05, |logFC|>0.585):")
    print(f"  no_filter:    {len(sig_no):>5,}")
    print(f"  with_filter:  {len(sig_with):>5,}")
    print(f"  shared:       {len(inter):>5,}")
    print(f"  no-only:      {len(sig_no - sig_with):>5,}")
    print(f"  with-only:    {len(sig_with - sig_no):>5,}")
    print(f"  Jaccard:      {len(inter)/max(1, len(sig_no | sig_with)):.4f}")
    print()

    # logFC concordance on shared features
    merged = top_no[["match_key", "logFC"]].merge(
        top_with[["match_key", "logFC"]], on="match_key", suffixes=("_no", "_with")
    ).dropna()
    pear = float(merged["logFC_no"].corr(merged["logFC_with"]))
    spear = float(merged["logFC_no"].corr(merged["logFC_with"], method="spearman"))
    med_abs_diff = float((merged["logFC_no"] - merged["logFC_with"]).abs().median())
    print(f"logFC concordance on {len(merged):,} shared features:")
    print(f"  Pearson r:   {pear:.4f}")
    print(f"  Spearman r:  {spear:.4f}")
    print(f"  median |diff|: {med_abs_diff:.4f}")
    print()

    # Top hits that ONLY appear in no_filter (these should be contaminants)
    only_no = sig_no - sig_with
    if only_no:
        rows = top_no.loc[top_no["match_key"].isin(only_no)].copy()
        rows = rows.sort_values("adj.P.Val").head(20)
        print("Top 20 hits that DISAPPEAR when contaminant filter is on:")
        print(rows[["feature", "logFC", "adj.P.Val"]].to_string(index=False))
        print()


# ============================================================================
def main():
    df_no, df_with = stage1_psm_counts()
    dropped = stage2_dropped_contaminant_breakdown(df_no)
    sites_no, sites_with, keyed_no, keyed_with, set_no, set_with = stage3_site_level(
        df_no, df_with
    )
    stage4_sn_comparison(keyed_no, keyed_with)
    stage5_limma(sites_no, sites_with)
    print("=== Done. ===")


if __name__ == "__main__":
    main()
