"""End-to-end signalome pipeline tests.

Combined into one file because the assignments -> modules -> network ->
expanded stages share the same setup and the shared fixture keeps the
test count honest.  Bit-exact parity vs PhosPy lives in
:mod:`tests.unit.test_signalome_phospy_parity`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaphos.signalome import (
    KinaseNetwork,
    SignalomeResult,
    build_expanded_table,
    build_kinase_network,
    build_module_assignments,
    build_module_table,
    build_signalome,
    cluster_sites,
    derive_protein_modules,
    select_kinase_substrates,
)


@pytest.fixture(scope="module")
def synthetic_pipeline():
    """3-module synthetic set + reference substrates."""
    n_sites, n_kinases = 90, 9
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 0.3, size=(n_sites, n_kinases))
    scores[:30, :3] += 3.0
    scores[30:60, 3:6] += 3.0
    scores[60:, 6:] += 3.0
    sites = [f"P{i:03d}|GENE{i:03d}|S{100 + i}|M1" for i in range(n_sites)]
    kinases = [f"KIN_{k}" for k in range(n_kinases)]
    mat = pd.DataFrame(scores, index=sites, columns=kinases)
    proteins = pd.Series([f"P{i:03d}" for i in range(n_sites)], index=sites)
    substrates = {}
    for k in range(n_kinases):
        if k < 3:
            substrates[f"KIN_{k}"] = tuple(sites[:30])
        elif k < 6:
            substrates[f"KIN_{k}"] = tuple(sites[30:60])
        else:
            substrates[f"KIN_{k}"] = tuple(sites[60:])
    metadata = pd.DataFrame(
        {
            "site_key": sites,
            "display_id": [f"{s.split('|')[0]}_{s.split('|')[2]}" for s in sites],
            "gene_symbol": [s.split("|")[1] for s in sites],
            "site": [s.split("|")[2] for s in sites],
            "protein_accession": [s.split("|")[0] for s in sites],
            "isoform_id": [""] * n_sites,
        },
        index=sites,
    )
    return {
        "matrix": mat,
        "sites": sites,
        "kinases": kinases,
        "proteins": proteins,
        "substrates": substrates,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Protein resolution
# ---------------------------------------------------------------------------


class TestProteinResolution:
    def test_cluster_signature_grouping(self, synthetic_pipeline):
        # 3-block synthetic + 1 protein per site => 3 modules, 30 proteins each.
        m = synthetic_pipeline["matrix"]
        proteins = synthetic_pipeline["proteins"]
        cl = cluster_sites(m, requested_module_count=3)
        series = pd.Series(cl.labels, index=cl.site_index)
        pm = derive_protein_modules(site_clusters=series, site_to_protein=proteins)
        assert sorted(pm.unique()) == [1, 2, 3]
        counts = pm.value_counts()
        assert counts.to_dict() == {1: 30, 2: 30, 3: 30}

    def test_missing_site_raises(self, synthetic_pipeline):
        m = synthetic_pipeline["matrix"]
        cl = cluster_sites(m, requested_module_count=3)
        series = pd.Series(cl.labels, index=cl.site_index)
        proteins = synthetic_pipeline["proteins"].iloc[:-1]  # drop last
        with pytest.raises(ValueError, match="missing clustered sites"):
            derive_protein_modules(site_clusters=series, site_to_protein=proteins)


# ---------------------------------------------------------------------------
# Assignments
# ---------------------------------------------------------------------------


class TestAssignments:
    def test_module_id_broadcast_from_protein(self, synthetic_pipeline):
        d = synthetic_pipeline
        cl = cluster_sites(d["matrix"], requested_module_count=3)
        series = pd.Series(cl.labels, index=cl.site_index)
        pm = derive_protein_modules(site_clusters=series, site_to_protein=d["proteins"])
        a = build_module_assignments(
            prediction_matrix=d["matrix"],
            site_to_protein=d["proteins"],
            protein_modules=pm,
            site_metadata=d["metadata"],
        )
        # Each block's 30 sites should share a module_id.
        assert len(set(a["module_id"].iloc[:30])) == 1
        assert len(set(a["module_id"].iloc[30:60])) == 1
        assert len(set(a["module_id"].iloc[60:])) == 1

    def test_top_kinase_lex_tie_break(self, synthetic_pipeline):
        # Row with two identical top scores -> lex-smaller kinase wins.
        idx = ["P|G|S1|M1"]
        mat = pd.DataFrame([[1.0, 1.0, 0.0]], index=idx, columns=["Z_K", "A_K", "B_K"])
        pm = pd.Series([1], index=["P"])
        meta = pd.DataFrame(
            {
                "site_key": idx,
                "display_id": idx,
                "gene_symbol": ["G"],
                "site": ["S1"],
                "protein_accession": ["P"],
                "isoform_id": [""],
            },
            index=idx,
        )
        s2p = pd.Series(["P"], index=idx)
        a = build_module_assignments(
            prediction_matrix=mat,
            site_to_protein=s2p,
            protein_modules=pm,
            site_metadata=meta,
        )
        assert a["top_kinase"].iloc[0] == "A_K"
        assert a["top_kinase_tie_count"].iloc[0] == 2
        assert a["top_kinase_is_ambiguous"].iloc[0]

    def test_module_top_kinase_majority_vote(self, synthetic_pipeline):
        d = synthetic_pipeline
        cl = cluster_sites(d["matrix"], requested_module_count=3)
        series = pd.Series(cl.labels, index=cl.site_index)
        pm = derive_protein_modules(site_clusters=series, site_to_protein=d["proteins"])
        a = build_module_assignments(
            prediction_matrix=d["matrix"],
            site_to_protein=d["proteins"],
            protein_modules=pm,
            site_metadata=d["metadata"],
        )
        # Module 1 sites should elect a kinase from KIN_0..KIN_2; module 2 from
        # KIN_3..KIN_5; module 3 from KIN_6..KIN_8.
        mod_to_top = a.groupby("module_id")["module_top_kinase"].first().to_dict()
        assert mod_to_top[1] in {"KIN_0", "KIN_1", "KIN_2"}
        assert mod_to_top[2] in {"KIN_3", "KIN_4", "KIN_5"}
        assert mod_to_top[3] in {"KIN_6", "KIN_7", "KIN_8"}


# ---------------------------------------------------------------------------
# Module table
# ---------------------------------------------------------------------------


class TestModuleTable:
    def test_rows_sum_to_100(self, synthetic_pipeline):
        d = synthetic_pipeline
        cl = cluster_sites(d["matrix"], requested_module_count=3)
        series = pd.Series(cl.labels, index=cl.site_index)
        pm = derive_protein_modules(site_clusters=series, site_to_protein=d["proteins"])
        a = build_module_assignments(
            prediction_matrix=d["matrix"],
            site_to_protein=d["proteins"],
            protein_modules=pm,
            site_metadata=d["metadata"],
        )
        table = build_module_table(
            module_assignments=a,
            kinase_substrates=d["substrates"],
            kinase_order=d["kinases"],
        )
        # Rows with any support sum to ~100 (rounding).
        for row_sum in table.sum(axis=1):
            assert abs(row_sum - 100.0) < 0.01

    def test_block_biology_recovered(self, synthetic_pipeline):
        d = synthetic_pipeline
        cl = cluster_sites(d["matrix"], requested_module_count=3)
        series = pd.Series(cl.labels, index=cl.site_index)
        pm = derive_protein_modules(site_clusters=series, site_to_protein=d["proteins"])
        a = build_module_assignments(
            prediction_matrix=d["matrix"],
            site_to_protein=d["proteins"],
            protein_modules=pm,
            site_metadata=d["metadata"],
        )
        table = build_module_table(
            module_assignments=a,
            kinase_substrates=d["substrates"],
            kinase_order=d["kinases"],
        )
        # Each block should be dominated by its 3 injected kinases (~33% each,
        # totalling ~100%).
        for module_id in (1, 2, 3):
            row = table.loc[module_id]
            top3 = row.sort_values(ascending=False).head(3)
            assert top3.sum() > 95.0

    def test_bad_policy_raises(self, synthetic_pipeline):
        d = synthetic_pipeline
        cl = cluster_sites(d["matrix"], requested_module_count=3)
        series = pd.Series(cl.labels, index=cl.site_index)
        pm = derive_protein_modules(site_clusters=series, site_to_protein=d["proteins"])
        a = build_module_assignments(
            prediction_matrix=d["matrix"],
            site_to_protein=d["proteins"],
            protein_modules=pm,
            site_metadata=d["metadata"],
        )
        with pytest.raises(ValueError, match="assignment_policy"):
            build_module_table(
                module_assignments=a,
                kinase_substrates=d["substrates"],
                kinase_order=d["kinases"],
                assignment_policy="bogus",
            )


# ---------------------------------------------------------------------------
# Kinase network
# ---------------------------------------------------------------------------


class TestKinaseNetwork:
    def test_within_block_kinases_are_edges(self, synthetic_pipeline):
        d = synthetic_pipeline
        net = build_kinase_network(
            prediction_matrix=d["matrix"],
            kinase_order=d["kinases"],
            kinase_substrates=d["substrates"],
            threshold=0.5,
            network_policy="signed",
        )
        assert isinstance(net, KinaseNetwork)
        edge_pairs = set(zip(net.edges["source_kinase"], net.edges["target_kinase"], strict=True))
        # Within-block kinases should all be highly correlated -> edges present
        # under either direction (source<target ordering).
        for pair in [("KIN_0", "KIN_1"), ("KIN_3", "KIN_4"), ("KIN_6", "KIN_7")]:
            assert pair in edge_pairs

    def test_node_degrees_match_edge_incidence(self, synthetic_pipeline):
        d = synthetic_pipeline
        net = build_kinase_network(
            prediction_matrix=d["matrix"],
            kinase_order=d["kinases"],
            kinase_substrates=d["substrates"],
            threshold=0.5,
            network_policy="signed",
        )
        from collections import Counter

        counted: Counter = Counter()
        for src, tgt in zip(net.edges["source_kinase"], net.edges["target_kinase"], strict=True):
            counted[src] += 1
            counted[tgt] += 1
        for kinase in net.nodes.index:
            assert net.nodes.loc[kinase, "degree"] == counted.get(kinase, 0)

    def test_bad_policy_raises(self, synthetic_pipeline):
        d = synthetic_pipeline
        with pytest.raises(ValueError, match="network_policy"):
            build_kinase_network(
                prediction_matrix=d["matrix"],
                kinase_order=d["kinases"],
                kinase_substrates=d["substrates"],
                threshold=0.5,
                network_policy="bogus",
            )

    def test_positive_only_drops_negative_edges(self, synthetic_pipeline):
        d = synthetic_pipeline
        net = build_kinase_network(
            prediction_matrix=d["matrix"],
            kinase_order=d["kinases"],
            kinase_substrates=d["substrates"],
            threshold=0.5,
            network_policy="positive_only",
        )
        assert (net.edges["correlation"] >= 0.5).all()


# ---------------------------------------------------------------------------
# Expanded table
# ---------------------------------------------------------------------------


class TestExpandedTable:
    def test_all_kinases_get_at_least_one_row(self, synthetic_pipeline):
        d = synthetic_pipeline
        cl = cluster_sites(d["matrix"], requested_module_count=3)
        series = pd.Series(cl.labels, index=cl.site_index)
        pm = derive_protein_modules(site_clusters=series, site_to_protein=d["proteins"])
        a = build_module_assignments(
            prediction_matrix=d["matrix"],
            site_to_protein=d["proteins"],
            protein_modules=pm,
            site_metadata=d["metadata"],
        )
        table = build_module_table(
            module_assignments=a,
            kinase_substrates=d["substrates"],
            kinase_order=d["kinases"],
        )
        net = build_kinase_network(
            prediction_matrix=d["matrix"],
            kinase_order=d["kinases"],
            kinase_substrates=d["substrates"],
            threshold=0.5,
            network_policy="signed",
        )
        exp = build_expanded_table(
            module_assignments=a,
            module_table=table,
            network_edges=net.edges,
            kinase_substrates=d["substrates"],
            kinase_order=d["kinases"],
        )
        assert set(exp["kinase"]) == set(d["kinases"])

    def test_linked_kinases_json_valid(self, synthetic_pipeline):
        import json

        d = synthetic_pipeline
        cl = cluster_sites(d["matrix"], requested_module_count=3)
        series = pd.Series(cl.labels, index=cl.site_index)
        pm = derive_protein_modules(site_clusters=series, site_to_protein=d["proteins"])
        a = build_module_assignments(
            prediction_matrix=d["matrix"],
            site_to_protein=d["proteins"],
            protein_modules=pm,
            site_metadata=d["metadata"],
        )
        table = build_module_table(
            module_assignments=a,
            kinase_substrates=d["substrates"],
            kinase_order=d["kinases"],
        )
        net = build_kinase_network(
            prediction_matrix=d["matrix"],
            kinase_order=d["kinases"],
            kinase_substrates=d["substrates"],
            threshold=0.5,
            network_policy="signed",
        )
        exp = build_expanded_table(
            module_assignments=a,
            module_table=table,
            network_edges=net.edges,
            kinase_substrates=d["substrates"],
            kinase_order=d["kinases"],
        )
        for value in exp["linked_kinases"].unique():
            parsed = json.loads(value)
            assert isinstance(parsed, list)
            # Focal kinase always appears first.
            assert len(parsed) >= 1


# ---------------------------------------------------------------------------
# Orchestrator (single-call build_signalome)
# ---------------------------------------------------------------------------


class TestBuildSignalome:
    def test_defaults_from_alphaphos_keys(self, synthetic_pipeline):
        d = synthetic_pipeline
        result = build_signalome(d["matrix"], kinase_substrates=d["substrates"])
        assert isinstance(result, SignalomeResult)
        # Every attribute wired
        assert result.site_assignments.shape[0] == 90
        assert result.protein_modules.size == 90  # 1 site per protein
        assert not result.module_table.empty
        assert isinstance(result.network, KinaseNetwork)
        assert not result.expanded.empty
        # Provenance carries the pipeline settings
        for k in (
            "n_sites",
            "n_kinases",
            "module_count",
            "selection_reason",
            "assignment_policy",
        ):
            assert k in result.provenance

    def test_derived_substrates_from_matrix(self, synthetic_pipeline):
        # Skip kinase_substrates; orchestrator threshold-derives them.
        d = synthetic_pipeline
        result = build_signalome(d["matrix"], substrate_support_cutoff=1.5)
        # Sanity: still recovers 3 modules
        modules = set(result.site_assignments["module_id"].unique())
        assert len(modules) >= 3

    def test_requested_module_count_honoured(self, synthetic_pipeline):
        d = synthetic_pipeline
        result = build_signalome(
            d["matrix"], kinase_substrates=d["substrates"], requested_module_count=3
        )
        assert result.provenance["module_count"] == 3

    def test_select_kinase_substrates_cutoff(self, synthetic_pipeline):
        d = synthetic_pipeline
        subs = select_kinase_substrates(prediction_matrix=d["matrix"], cutoff=1.5)
        # High-block kinases pick up ~30 sites each; low-block kinases few.
        for kinase in ("KIN_0", "KIN_1", "KIN_3", "KIN_6"):
            assert len(subs[kinase]) >= 20
