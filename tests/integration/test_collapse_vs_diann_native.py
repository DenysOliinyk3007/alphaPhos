"""Regression guard: ``collapse_sites`` on a DIA-NN 2.x report vs DIA-NN's own site tables.

Runs only when the EGF HeLa DIA-NN output is present locally
(``ALPHAPHOS_DIANN_BENCH_DIR`` or the default lab path); skipped in CI.
Thresholds are the 2026-09-12 values with a small margin -- see
``docs/benchmark/spectronaut_native_benchmark.md`` §10 and
``docs/benchmark/validate_collapse_egf_hela_diann.py``.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import alphaphos as ap
from alphaphos.io.diann import diann_precursor_id
from alphaphos.preprocess._collapse.parsing import extract_sequence_modifications

DATA_DIR = Path(
    os.environ.get(
        "ALPHAPHOS_DIANN_BENCH_DIR", "/Users/denysoliinyk/Documents/Data/egf_hela_nanophos_diann"
    )
)
REPORT = DATA_DIR / "report.parquet"
SITE_REPORT = DATA_DIR / "report.site_report.parquet"
M90 = DATA_DIR / "report.phosphosites_90.tsv"
SN_LONG = Path(
    "/Users/denysoliinyk/Documents/Data/egf_hela_nanophos_spectronaut/nanophos_egf_hela_1ug_long_ms2.tsv"
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (REPORT.exists() and SITE_REPORT.exists() and M90.exists()),
        reason="EGF HeLa DIA-NN benchmark output not available locally",
    ),
]


def _run(col: str) -> str:
    return re.sub(r"\.raw$", "", col.replace("\\", "/").split("/")[-1])


def _nomult(adata) -> pd.DataFrame:
    v = adata.var
    key = (
        v["protein_group_id"].astype(str)
        + "|"
        + v["site_aa"].astype(str)
        + v["site_position"].astype(int).astype(str)
    )
    q = pd.DataFrame(adata.layers["intensity_log2"].T, index=key.values, columns=adata.obs_names)
    return np.log2(np.exp2(q.astype(float)).groupby(level=0).sum(min_count=1))


@pytest.fixture(scope="module")
def psm():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ap.read_diann(REPORT)


@pytest.fixture(scope="module")
def occ():
    sr = pd.read_parquet(SITE_REPORT)
    rep = pd.read_parquet(REPORT, columns=["Run", "Precursor.Id", "Precursor.Quantity"])
    o = sr[sr["Occupied"] == 1].merge(rep, on=["Run", "Precursor.Id"], how="left")
    o["key"] = o["Protein"] + "|" + o["Residue"] + o["Site"].astype(str)
    return o


@pytest.fixture(scope="module")
def cdf(psm):
    s = sorted(psm["R.FileName"].unique())
    return pd.DataFrame(
        {"sample": s, "condition": ["withEGF" if "withEGF" in x else "woEGF" for x in s]}
    )


@pytest.fixture(scope="module")
def per_run_090(psm, cdf):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ap.collapse_sites(
            psm,
            condition_df=cdf,
            advanced={
                "search_engine": "Diann",
                "quantification_level": "auto",
                "localization_strategy": "per_run",
                "cutoff": 0.9,
            },
        )


def test_reader_funnel_and_defaults(psm, per_run_090):
    assert psm.attrs["n_rows_unmappable"] == 0
    assert "n_rows_after_contaminants" in psm.attrs
    assert not psm["PG.ProteinGroups"].str.startswith("cRAP").any()
    stats = per_run_090.uns["alphaphos"]["stats"]
    assert stats["top_n_attribution_applied"] is False  # "auto" -> off for DIA-NN
    assert stats["quantification_column_used"] == "Precursor.Quantity"


def test_positions_are_subset_of_diann_occupied_sites(psm, occ):
    rep = pd.read_parquet(
        REPORT, columns=["Precursor.Id", "Modified.Sequence", "Precursor.Charge"]
    ).drop_duplicates("Precursor.Id")
    rep["EG.PrecursorId"] = [
        diann_precursor_id(m, c)
        for m, c in zip(rep["Modified.Sequence"], rep["Precursor.Charge"], strict=True)
    ]
    prec = psm.drop_duplicates("EG.PrecursorId").merge(
        rep[["EG.PrecursorId", "Precursor.Id"]], on="EG.PrecursorId"
    )
    prec["prot"] = prec["PG.ProteinGroups"].str.split(";").str[0]
    aphos = [
        {int(s) + p - 1 for p in m["phospho_positions"]}
        for m, s in zip(
            prec["EG.PrecursorId"].map(extract_sequence_modifications),
            prec["PEP.PeptidePosition"],
            strict=True,
        )
    ]
    truth = occ.groupby(["Precursor.Id", "Protein"])["Site"].apply(set).reset_index()
    chk = prec.assign(aphos=aphos).merge(
        truth, left_on=["Precursor.Id", "prot"], right_on=["Precursor.Id", "Protein"]
    )
    assert len(chk) > 30_000
    bad = [
        (r["EG.PrecursorId"], r["prot"]) for _, r in chk.iterrows() if not (r["aphos"] <= r["Site"])
    ]
    assert bad == [], (
        f"{len(bad)} precursors with positions outside DIA-NN's occupied sites, e.g. {bad[:3]}"
    )


def test_matches_gated_sum_oracle_from_site_table(per_run_090, occ):
    a = _nomult(per_run_090)
    d = occ[occ["Probability"] >= 0.9]
    oracle = np.log2(
        d.groupby(["key", "Run"])["Precursor.Quantity"].sum(min_count=1).unstack("Run")
    )
    a = a.reindex(columns=oracle.columns)
    shared = a.index.intersection(oracle.index)
    A, B = a.loc[shared].to_numpy(), oracle.loc[shared].to_numpy()
    both = ~np.isnan(A) & ~np.isnan(B)
    dd = A[both] - B[both]
    assert both.sum() > 50_000
    assert np.corrcoef(A[both], B[both])[0, 1] >= 0.995
    assert abs(dd.mean()) <= 0.03
    assert (np.abs(dd) <= 0.1).mean() >= 0.98
    assert (~np.isnan(A) & np.isnan(B)).sum() == 0  # nothing alphaPhos keeps that the oracle lacks


def test_site_set_vs_diann_phosphosites_90(per_run_090):
    m = pd.read_csv(M90, sep="\t")
    keys = set(m["Protein"] + "|" + m["Residue"] + m["Site"].astype(str))
    ours = set(_nomult(per_run_090).index)
    jaccard = len(ours & keys) / len(ours | keys)
    assert jaccard >= 0.90, jaccard


@pytest.mark.skipif(
    not SN_LONG.exists(), reason="Spectronaut export of the same raw files not available"
)
def test_egfr_autophosphorylation_reproduces_across_engines(psm, cdf):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dn = _nomult(
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
        sn = _nomult(
            ap.collapse_sites(
                ap.read_spectronaut(SN_LONG),
                condition_df=cdf,
                advanced={"localization_strategy": "per_run"},
            )
        )
    w = [c for c in sn.columns if "withEGF" in c]
    o = [c for c in sn.columns if "woEGF" in c]
    for site in ("P00533|Y1172", "P00533|Y1197"):
        for mat in (sn, dn.reindex(columns=sn.columns)):
            fc = mat.loc[site, w].mean() - mat.loc[site, o].mean()
            assert fc > 3.0, f"{site}: EGF-induced log2FC {fc:.2f} not recovered"
