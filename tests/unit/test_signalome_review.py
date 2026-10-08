"""Regression tests from the 2026-09-14 review of :mod:`alphaphos.signalome`."""

from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from alphaphos.signalome import (
    build_signalome,
    cluster_sites,
    derive_protein_modules,
    extract_site_metadata,
    prediction_matrix_from_adata,
)


def _blocks(n_per=30, seed=0):
    """Three clean blocks of sites on a [0, 1] scale, 8 kinases (2 never hot), 3 sites/protein."""
    rng = np.random.default_rng(seed)

    def block(hot):
        m = rng.uniform(0.0, 0.2, (n_per, 8))
        m[:, hot] += rng.uniform(0.6, 0.8, (n_per, len(hot)))
        return m

    M = np.vstack([block([0, 1]), block([2, 3]), block([4, 5])])
    idx = [f"P{i // 3:03d}|G{i // 3}|S{i}|M1" for i in range(3 * n_per)]
    return pd.DataFrame(M, index=idx, columns=[f"K{j}" for j in range(8)])


class TestProvenanceAndValidation:
    def test_n_modules_counts_modules_not_proteins(self):
        res = build_signalome(_blocks(), substrate_support_cutoff=0.5)
        prov = res.provenance
        assert prov["n_modules"] == int(res.protein_modules[res.protein_modules > 0].nunique())
        assert prov["n_proteins_assigned"] <= prov["n_proteins"] == 30
        assert prov["n_modules"] < prov["n_proteins"]
        assert 0.0 <= prov["substrate_density"] <= 1.0

    def test_duplicate_index_rejected_upfront(self):
        m = _blocks()
        m.index = [m.index[0], m.index[0], *m.index[2:]]
        with pytest.raises(ValueError, match="duplicate site ids"):
            build_signalome(m)

    def test_empty_and_non_numeric_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            build_signalome(pd.DataFrame(index=["a|b|S1|M1"]))
        bad = _blocks().astype(object)
        bad.iloc[0, 0] = "x"
        with pytest.raises(TypeError, match="numeric"):
            build_signalome(bad)

    def test_scale_and_density_warnings(self):
        raw = _blocks() * 20 - 10  # log2-score-like range
        with pytest.warns(UserWarning, match="not \\[0, 1\\]"):
            build_signalome(raw, substrate_support_cutoff=0.5)
        with pytest.warns(UserWarning, match="close to uniform"):
            build_signalome(_blocks(), substrate_support_cutoff=0.05)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            build_signalome(_blocks(), substrate_support_cutoff=0.5)  # clean: no warning


class TestAllNanRows:
    def test_all_nan_sites_are_unassigned_and_do_not_change_k(self):
        clean = _blocks()
        ref = build_signalome(clean, substrate_support_cutoff=0.5)
        nan_rows = pd.DataFrame(
            np.nan, index=[f"Q{i:03d}|GQ{i}|Y{i}|M1" for i in range(20)], columns=clean.columns
        )
        mixed = build_signalome(pd.concat([clean, nan_rows]), substrate_support_cutoff=0.5)
        labels = pd.Series(mixed.clustering.labels, index=mixed.clustering.site_index)
        assert (labels[nan_rows.index] == 0).all()
        assert (labels[clean.index] >= 1).all()
        assert mixed.provenance["module_count"] == ref.provenance["module_count"]
        assert mixed.provenance["n_sites_all_nan"] == 20
        sa = mixed.site_assignments.loc[nan_rows.index]
        assert (sa["module_id"] == 0).all() and (sa["top_kinase"] == "unsupported").all()
        # proteins with only unassigned sites get module 0, not a module of their own
        assert (mixed.protein_modules.loc[[f"Q{i:03d}" for i in range(20)]] == 0).all()
        pd.testing.assert_series_equal(
            mixed.protein_modules.loc[ref.protein_modules.index], ref.protein_modules
        )

    def test_cluster_sites_labels_zero_for_all_nan(self):
        m = _blocks()
        m.iloc[:5] = np.nan
        res = cluster_sites(m, requested_module_count=3)
        assert (res.labels[:5] == 0).all() and res.labels[5:].min() == 1
        assert res.provenance["n_sites_clustered"] == len(m) - 5

    def test_derive_protein_modules_ignores_label_zero(self):
        clusters = pd.Series([1, 1, 0, 2, 0], index=list("abcde"))
        proteins = pd.Series(["X", "X", "Y", "Z", "Z"], index=list("abcde"))
        pm = derive_protein_modules(site_clusters=clusters, site_to_protein=proteins)
        assert pm["Y"] == 0  # only an unassigned site
        assert pm["X"] > 0 and pm["Z"] > 0 and pm["X"] != pm["Z"]


class TestPredictionMatrixHelper:
    def _adata(self):
        rng = np.random.default_rng(1)
        n = 60
        idx = [f"P{i:03d}|G{i}|S{i}|M1" for i in range(n)]
        pct = pd.DataFrame(rng.uniform(0, 50, (n, 5)), index=idx, columns=["A", "B", "C", "D", "E"])
        pct["E"] = 60.0  # E is top-ranked wherever no other kinase is hot
        pct.loc[pct.index[:20], "A"] = 99.0  # A is top for 20 sites
        pct.loc[pct.index[20:35], "B"] = 95.0  # B is top for 15
        pct.loc[pct.index[35:38], "C"] = 92.0  # C is top for 3 -> filtered
        pct.iloc[-2:] = np.nan  # two sites the library dropped
        a = ad.AnnData(X=np.zeros((1, n)), var=pd.DataFrame(index=idx))
        a.varm["kinase_percentile_ser_thr"] = pct
        a.varm["kinase_score_ser_thr"] = pct - 50.0
        return a, idx

    def test_percentile_scaled_filtered_and_nan_dropped(self):
        a, idx = self._adata()
        m = prediction_matrix_from_adata(a, min_top_sites=10)
        assert list(m.columns) == ["A", "B", "E"]
        assert m.shape[0] == 58 and m.to_numpy().max() <= 1.0 and m.to_numpy().min() >= 0.0
        assert m.attrs["metric"] == "percentile"
        assert m.attrs["top_site_counts"]["A"] == 20 and m.attrs["top_site_counts"]["B"] == 15
        sub = prediction_matrix_from_adata(a, sites=idx[:30], min_top_sites=10)
        assert list(sub.columns) == ["A", "B"] and len(sub) == 30  # E tops only 0 of the first 30

    def test_errors_and_score_warning(self):
        a, _ = self._adata()
        with pytest.raises(KeyError, match="kinase_percentile_tyrosine"):
            prediction_matrix_from_adata(a, pool="tyrosine")
        with pytest.raises(ValueError, match="only 0 kinase"):
            prediction_matrix_from_adata(a, min_top_sites=21)  # A tops 20, E 20, B 15
        with pytest.raises(KeyError, match="not in adata.var_names"):
            prediction_matrix_from_adata(a, sites=["nope|G|S1|M1"])
        with pytest.warns(UserWarning, match="raw log2"):
            prediction_matrix_from_adata(a, metric="score", min_top_sites=0)

    def test_helper_output_runs_without_warnings(self):
        a, _ = self._adata()
        m = prediction_matrix_from_adata(a, min_top_sites=10)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = build_signalome(m, substrate_support_cutoff=0.9, requested_module_count=2)
        assert res.provenance["min_module_share_percent"] == pytest.approx(200.0 / 3)


def test_extract_site_metadata_missing_gene_is_empty_not_nan():
    var = pd.DataFrame(
        {
            "gene": [None, "EGFR"],
            "site_aa": ["S", "Y"],
            "site_position": [1, 2],
            "protein_group_id": ["P1", "P2"],
        },
        index=["P1||S1|M1", "P2|EGFR|Y2|M1"],
    )
    md = extract_site_metadata(var)
    assert md["gene_symbol"].tolist() == ["", "EGFR"]
