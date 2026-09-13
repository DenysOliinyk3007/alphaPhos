"""Regression guard: ``collapse_sites`` vs Spectronaut's native PTM site report.

Runs only when the EGF HeLa nanoPhos Spectronaut exports are present locally
(``ALPHAPHOS_SN_BENCH_DIR`` or the default lab path); skipped in CI.  The
thresholds are the values measured on 2026-09-12 with a small safety margin
-- see ``docs/benchmark/spectronaut_native_benchmark.md`` §9 and
``docs/benchmark/validate_collapse_egf_hela.py`` for the full matrix.
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
from alphaphos.preprocess._collapse.parsing import (
    extract_first_valid_position,
    extract_sequence_modifications,
)

DATA_DIR = Path(
    os.environ.get(
        "ALPHAPHOS_SN_BENCH_DIR",
        "/Users/denysoliinyk/Documents/Data/egf_hela_nanophos_spectronaut",
    )
)
PREFIX = "nanophos_egf_hela_1ug_"
LONG_MS2 = DATA_DIR / f"{PREFIX}long_ms2.tsv"
LONG_DUAL = DATA_DIR / f"{PREFIX}long_ms1_and_ms2.tsv"
SHORT_0P75 = DATA_DIR / f"{PREFIX}short_cutoff_0p75.tsv"
SHORT_0 = DATA_DIR / f"{PREFIX}short_cutoff_0.tsv"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not (LONG_MS2.exists() and SHORT_0P75.exists()),
        reason="EGF HeLa Spectronaut benchmark exports not available locally",
    ),
]


def _sample(col: str) -> str:
    return re.sub(r"\.raw\.PTM\.(Quantity|SiteProbability)$", "", re.sub(r"^\[\d+\]\s*", "", col))


def _load_sn(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SN site report -> (log2 quant, loc prob) indexed by ``PROT|AApos|Mcap``."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df[df["PTM.ModificationTitle"] == "Phospho (STY)"]
    m = df["PTM.CollapseKey"].str.extract(
        r"^(?P<p>[^_]+)_(?P<aa>[A-Z])(?P<pos>\d+)_M(?P<m>\d+)_\d+$"
    )
    key = m["p"] + "|" + m["aa"] + m["pos"] + "|M" + m["m"].astype(int).clip(upper=3).astype(str)
    q = df[[c for c in df.columns if c.endswith(".PTM.Quantity")]].apply(
        pd.to_numeric, errors="coerce"
    )
    q.columns = [_sample(c) for c in q.columns]
    q.index = key
    p = df[[c for c in df.columns if c.endswith(".PTM.SiteProbability")]].apply(
        pd.to_numeric, errors="coerce"
    )
    p.columns = [_sample(c) for c in p.columns]
    p.index = key
    q = q.groupby(level=0).sum(min_count=1)
    return np.log2(q.where(q > 0)), p.groupby(level=0).max()


def _wide(adata) -> tuple[pd.DataFrame, pd.DataFrame]:
    v = adata.var
    key = (
        v["protein_group_id"].astype(str)
        + "|"
        + v["site_aa"].astype(str)
        + v["site_position"].astype(int).astype(str)
        + "|M"
        + v["multiplicity"].astype(int).astype(str)
    )
    q = pd.DataFrame(adata.layers["intensity_log2"].T, index=key, columns=adata.obs_names)
    loc = pd.DataFrame(adata.layers["localization"].T, index=key, columns=adata.obs_names)
    return q.astype(float), loc.astype(float)


@pytest.fixture(scope="module")
def sn():
    return _load_sn(SHORT_0P75)


@pytest.fixture(scope="module")
def psm():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ap.read_spectronaut(LONG_MS2)


@pytest.fixture(scope="module")
def cdf(psm):
    s = sorted(psm["R.FileName"].unique())
    return pd.DataFrame(
        {"sample": s, "condition": ["withEGF" if "withEGF" in x else "woEGF" for x in s]}
    )


@pytest.fixture(scope="module")
def per_run(psm, cdf):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ap.collapse_sites(
            psm, condition_df=cdf, advanced={"localization_strategy": "per_run"}
        )


class TestPositions:
    def test_site_positions_match_spectronauts_own(self):
        """alphaPhos absolute positions == EG.ProteinPTMLocations for every phospho precursor."""
        raw = pd.read_csv(
            LONG_MS2,
            sep="\t",
            usecols=["EG.PrecursorId", "EG.ProteinPTMLocations", "PEP.PeptidePosition"],
        ).drop_duplicates("EG.PrecursorId")
        mods = raw["EG.PrecursorId"].map(extract_sequence_modifications)
        keep = np.array([m["phospho_count"] > 0 for m in mods])
        raw, mods = raw[keep], [m for m, k in zip(mods, keep, strict=True) if k]
        start = raw["PEP.PeptidePosition"].map(extract_first_valid_position)
        aphos = [
            tuple(sorted(int(s) + p - 1 for p in m["phospho_positions"])) if pd.notna(s) else None
            for m, s in zip(mods, start, strict=True)
        ]

        def sn_first_group(loc):
            if not isinstance(loc, str):
                return None
            groups = re.findall(r"\(([^)]*)\)", loc.split(";")[0])
            return tuple(sorted(int(x[1:]) for x in groups[0].split(",") if x and x[0] in "STY"))

        sn_pos = raw["EG.ProteinPTMLocations"].map(sn_first_group)
        ok = [a is not None and b is not None for a, b in zip(aphos, sn_pos, strict=True)]
        agree = [a == b for a, b, k in zip(aphos, sn_pos, ok, strict=True) if k]
        assert sum(ok) > 100_000
        assert all(agree), (
            f"{len(agree) - sum(agree)} precursors disagree with EG.ProteinPTMLocations"
        )


class TestSiteSet:
    def test_jaccard_against_class_i_report(self, per_run, sn):
        q, _ = _wide(per_run)
        shared = q.index.intersection(sn[0].index)
        jaccard = len(shared) / len(q.index.union(sn[0].index))
        assert jaccard >= 0.97, jaccard

    def test_no_alphaphos_site_outside_spectronauts_universe(self, per_run):
        sn0, _ = _load_sn(SHORT_0)
        q, _ = _wide(per_run)
        stray = q.index.difference(sn0.index)
        assert len(stray) == 0, stray[:10].tolist()


class TestQuant:
    def test_per_cell_agreement(self, per_run, sn):
        q, loc = _wide(per_run)
        S, P = sn
        shared = q.index.intersection(S.index)
        A = q.loc[shared].reindex(columns=S.columns).to_numpy()
        B = S.loc[shared].to_numpy()
        both = ~np.isnan(A) & ~np.isnan(B)
        d = A[both] - B[both]
        assert both.sum() > 100_000
        assert np.corrcoef(A[both], B[both])[0, 1] >= 0.995
        assert abs(d.mean()) <= 0.03, d.mean()
        assert (np.abs(d) <= 0.1).mean() >= 0.95, (np.abs(d) <= 0.1).mean()
        # localization probabilities are Spectronaut's own numbers
        L = loc.loc[shared].reindex(columns=P.columns).to_numpy()
        Pp = P.loc[shared].to_numpy()
        lb = ~np.isnan(L) & ~np.isnan(Pp)
        assert (np.abs(L[lb] - Pp[lb]) <= 1e-3).mean() >= 0.99

    def test_log2fc_concordance(self, per_run, sn):
        q, _ = _wide(per_run)
        S, _ = sn
        shared = q.index.intersection(S.index)
        A, B = q.loc[shared].reindex(columns=S.columns), S.loc[shared]
        w = [c for c in S.columns if "withEGF" in c]
        o = [c for c in S.columns if "woEGF" in c]
        fa, fb = A[w].mean(axis=1) - A[o].mean(axis=1), B[w].mean(axis=1) - B[o].mean(axis=1)
        ok = fa.notna() & fb.notna()
        assert np.corrcoef(fa[ok], fb[ok])[0, 1] >= 0.97


@pytest.mark.skipif(not LONG_DUAL.exists(), reason="dual MS1+MS2 export not available")
def test_dual_export_at_ms2_is_identical_to_ms2_only_export(per_run, cdf):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dual = ap.collapse_sites(
            ap.read_spectronaut(LONG_DUAL),
            condition_df=cdf,
            advanced={"localization_strategy": "per_run", "quantification_level": "MS2"},
        )
    assert list(dual.var_names) == list(per_run.var_names)
    np.testing.assert_allclose(dual.X, per_run.X, rtol=1e-6, equal_nan=True)
