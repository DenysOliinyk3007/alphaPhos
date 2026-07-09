"""Stage-by-stage bit-exact parity against PhosPy on real algorithms.

Runs each signalome stage on the SAME input via both alphaPhos and PhosPy;
asserts identical output.  Marked ``skipif`` on machines without PhosPy
installed.

PhosPy is GPL-3.0 (alphaPhos is MIT), so we treat it purely as a
numerical oracle: never imported at package-import time, never installed
by default -- only present in this dev-time parity harness.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

phospy = pytest.importorskip("phospy")


# ---------------------------------------------------------------------------
# Shared fixtures -- 3 sizes to sample the scaling axis + auto-k rule
# ---------------------------------------------------------------------------


def _make_pipeline(n_sites: int, n_kinases: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    scores = rng.normal(0, 0.3, size=(n_sites, n_kinases))
    b1, b2 = n_sites // 3, 2 * n_sites // 3
    k1, k2 = n_kinases // 3, 2 * n_kinases // 3
    scores[:b1, :k1] += 3.0
    scores[b1:b2, k1:k2] += 3.0
    scores[b2:, k2:] += 3.0
    sites = [f"P{i:04d}|G{i:04d}|S{100 + i}|M1" for i in range(n_sites)]
    kinases = [f"KIN_{k:02d}" for k in range(n_kinases)]
    mat = pd.DataFrame(scores, index=sites, columns=kinases)
    proteins = pd.Series([f"P{i:04d}" for i in range(n_sites)], index=sites)
    subs = {}
    for k in range(n_kinases):
        if k < k1:
            subs[f"KIN_{k:02d}"] = tuple(sites[:b1])
        elif k < k2:
            subs[f"KIN_{k:02d}"] = tuple(sites[b1:b2])
        else:
            subs[f"KIN_{k:02d}"] = tuple(sites[b2:])
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
    return mat, proteins, subs, metadata


PARITY_SIZES = [(200, 9), (1000, 15)]


# ---------------------------------------------------------------------------
# Stage 1: Ward linkage + label cutting
# ---------------------------------------------------------------------------


class TestClusteringParity:
    @pytest.mark.parametrize(("n_sites", "n_kinases"), PARITY_SIZES)
    def test_linkage_and_labels_bit_exact(self, n_sites, n_kinases):
        from phospy.science.signalomes.clustering.backends.scipy_hierarchical import (
            build_cluster_labels_from_tree as pp_labels_from_tree,
        )
        from phospy.science.signalomes.clustering.backends.scipy_hierarchical import (
            build_cluster_tree as pp_build_tree,
        )

        from alphaphos.signalome.clustering import (
            build_ward_tree,
            cut_labels,
            precondition_scores,
        )

        mat, _, _, _ = _make_pipeline(n_sites, n_kinases)
        prep = precondition_scores(mat.to_numpy(dtype=float))

        # Linkage
        pp_tree = pp_build_tree(prep)
        ap_tree = build_ward_tree(prep)
        max_diff = float(np.max(np.abs(np.asarray(pp_tree.linkage_matrix) - ap_tree)))
        assert max_diff == 0.0, f"Ward linkage diverges: max diff {max_diff}"

        # Labels for k=3 (matches injected structure)
        k = 3
        pp_lab = pp_labels_from_tree(cluster_tree=pp_tree, cluster_counts=[k])[k]
        ap_lab = cut_labels(ap_tree, n_clusters=k, n_sites=prep.shape[0])
        np.testing.assert_array_equal(pp_lab, ap_lab)


# ---------------------------------------------------------------------------
# Stage 2: Module count auto-selection
# ---------------------------------------------------------------------------


class TestAutoModuleCountParity:
    @pytest.mark.parametrize(("n_sites", "n_kinases"), [(200, 9), (1000, 15)])
    def test_auto_k_matches_phospy(self, n_sites, n_kinases):
        from phospy.science.signalomes.clustering.selection import (
            select_module_count as pp_select,
        )

        from alphaphos.signalome.clustering import (
            build_ward_tree,
            precondition_scores,
        )
        from alphaphos.signalome.clustering import (
            select_module_count as ap_select,
        )

        mat, _, _, _ = _make_pipeline(n_sites, n_kinases)
        prep = precondition_scores(mat.to_numpy(dtype=float))
        tree = build_ward_tree(prep)
        pp_k = pp_select(
            prep,
            primary_threshold=0.5,
            fallback_threshold=0.1,
            max_clusters=10,
            scoring_mode="exact",
            max_exact_tree_sites=5000,
        )
        ap_result = ap_select(
            prep,
            tree,
            primary_threshold=0.5,
            fallback_threshold=0.1,
            max_modules=10,
            scoring_mode="exact",
            max_exact_sites=5000,
        )
        assert pp_k == ap_result.module_count


# ---------------------------------------------------------------------------
# Stage 3: protein_modules (cluster-signature grouping)
# ---------------------------------------------------------------------------


class TestProteinModulesParity:
    @pytest.mark.parametrize(("n_sites", "n_kinases"), PARITY_SIZES)
    def test_protein_modules_exact(self, n_sites, n_kinases):
        from phospy.science.signalomes.clustering.protein_modules import (
            derive_protein_modules as pp_derive,
        )

        from alphaphos.signalome import cluster_sites, derive_protein_modules

        mat, proteins, _, _ = _make_pipeline(n_sites, n_kinases)
        cl = cluster_sites(mat, requested_module_count=3, seed=0)
        series = pd.Series(cl.labels, index=cl.site_index)

        ap_pm = derive_protein_modules(site_clusters=series, site_to_protein=proteins)
        pp_pm = pp_derive(site_clusters=series, site_to_protein=proteins)

        common = ap_pm.index.intersection(pp_pm.index)
        assert len(common) == len(ap_pm)
        assert ap_pm.loc[common].astype(int).equals(pp_pm.loc[common].astype(int))


# ---------------------------------------------------------------------------
# Stage 4: Module table (percent shares)
# ---------------------------------------------------------------------------


class TestModuleTableParity:
    @pytest.mark.parametrize(("n_sites", "n_kinases"), PARITY_SIZES)
    def test_module_table_bit_exact(self, n_sites, n_kinases):
        from phospy.contracts.configs import SIGNALOME_ASSIGNMENT_POLICY_CUTOFF_BINARY
        from phospy.science.signalomes.assignments import (
            build_module_assignments as pp_assign,
        )
        from phospy.science.signalomes.modules import build_signalome_module_table

        from alphaphos.signalome import (
            build_module_assignments as ap_assign,
        )
        from alphaphos.signalome import (
            build_module_table,
            cluster_sites,
            derive_protein_modules,
        )

        mat, proteins, subs, metadata = _make_pipeline(n_sites, n_kinases)
        cl = cluster_sites(mat, requested_module_count=3, seed=0)
        series = pd.Series(cl.labels, index=cl.site_index)
        pm = derive_protein_modules(site_clusters=series, site_to_protein=proteins)
        kinase_order = list(mat.columns.astype(str))

        ap_a = ap_assign(
            prediction_matrix=mat,
            site_to_protein=proteins,
            protein_modules=pm,
            site_metadata=metadata,
        )
        pp_a = pp_assign(
            prediction_matrix=mat,
            site_to_protein=proteins,
            site_metadata=metadata,
            protein_modules=pm,
        )
        ap_mt = build_module_table(
            module_assignments=ap_a,
            kinase_substrates=subs,
            kinase_order=kinase_order,
            assignment_policy="cutoff_binary",
        )
        pp_mt = build_signalome_module_table(
            module_assignments=pp_a,
            kinase_substrates=subs,
            kinase_order=kinase_order,
            assignment_policy=SIGNALOME_ASSIGNMENT_POLICY_CUTOFF_BINARY,
        )
        max_diff = float(np.max(np.abs(ap_mt.to_numpy() - pp_mt.to_numpy())))
        assert max_diff == 0.0, f"module_table diverges: max diff {max_diff}"


# ---------------------------------------------------------------------------
# Stage 5: Kinase network (edges + nodes)
# ---------------------------------------------------------------------------


class TestNetworkParity:
    @pytest.mark.parametrize(("n_sites", "n_kinases"), PARITY_SIZES)
    def test_network_edges_bit_exact(self, n_sites, n_kinases):
        from phospy.contracts.configs import (
            SIGNALOME_KINASE_NETWORK_POLICY_SIGNED,
        )
        from phospy.science.signalomes.network import (
            build_kinase_network as pp_net,
        )

        from alphaphos.signalome import build_kinase_network as ap_net

        mat, _, subs, _ = _make_pipeline(n_sites, n_kinases)
        kinase_order = list(mat.columns.astype(str))
        threshold = 0.5

        ap_r = ap_net(
            prediction_matrix=mat,
            kinase_order=kinase_order,
            kinase_substrates=subs,
            threshold=threshold,
            network_policy="signed",
        )
        pp_edges, _pp_nodes = pp_net(
            downstream_score_matrix=mat,
            kinase_order=kinase_order,
            kinase_substrates=subs,
            threshold=threshold,
            network_policy=SIGNALOME_KINASE_NETWORK_POLICY_SIGNED,
        )
        key = ["source_kinase", "target_kinase"]
        ap_sorted = ap_r.edges.set_index(key)["correlation"].sort_index()
        pp_sorted = pp_edges.set_index(key)["correlation"].sort_index()
        assert ap_sorted.index.equals(pp_sorted.index)
        max_diff = float(np.max(np.abs(ap_sorted.to_numpy() - pp_sorted.to_numpy())))
        assert max_diff == 0.0

    @pytest.mark.parametrize(("n_sites", "n_kinases"), PARITY_SIZES)
    def test_network_nodes_bit_exact(self, n_sites, n_kinases):
        from phospy.contracts.configs import SIGNALOME_KINASE_NETWORK_POLICY_SIGNED
        from phospy.science.signalomes.network import build_kinase_network as pp_net

        from alphaphos.signalome import build_kinase_network as ap_net

        mat, _, subs, _ = _make_pipeline(n_sites, n_kinases)
        kinase_order = list(mat.columns.astype(str))
        ap_r = ap_net(
            prediction_matrix=mat,
            kinase_order=kinase_order,
            kinase_substrates=subs,
            threshold=0.5,
            network_policy="signed",
        )
        _, pp_nodes = pp_net(
            downstream_score_matrix=mat,
            kinase_order=kinase_order,
            kinase_substrates=subs,
            threshold=0.5,
            network_policy=SIGNALOME_KINASE_NETWORK_POLICY_SIGNED,
        )
        ap_sorted = ap_r.nodes.sort_index()
        pp_sorted = pp_nodes.sort_index()
        assert ap_sorted["degree"].tolist() == pp_sorted["degree"].tolist()
        assert ap_sorted["n_substrates"].tolist() == pp_sorted["n_substrates"].tolist()


# ---------------------------------------------------------------------------
# Stage 6: Expanded table -- row count + per-column content
# ---------------------------------------------------------------------------


class TestExpandedTableParity:
    def test_expanded_content_bit_exact_all_columns(self):
        # Smaller size for column-wise content comparison.
        n_sites, n_kinases = 200, 9

        from phospy.contracts.configs import (
            SIGNALOME_ASSIGNMENT_POLICY_CUTOFF_BINARY,
            SIGNALOME_KINASE_NETWORK_POLICY_SIGNED,
        )
        from phospy.science.signalomes.assignments import (
            build_module_assignments as pp_assign,
        )
        from phospy.science.signalomes.expanded import (
            build_expanded_signalome_table,
        )
        from phospy.science.signalomes.modules import build_signalome_module_table
        from phospy.science.signalomes.network import build_kinase_network as pp_net

        from alphaphos.signalome import (
            build_expanded_table,
            build_module_table,
            cluster_sites,
            derive_protein_modules,
        )
        from alphaphos.signalome import (
            build_kinase_network as ap_net,
        )
        from alphaphos.signalome import (
            build_module_assignments as ap_assign,
        )

        mat, proteins, subs, metadata = _make_pipeline(n_sites, n_kinases)
        kinase_order = list(mat.columns.astype(str))
        threshold = 0.5

        cl = cluster_sites(mat, requested_module_count=3, seed=0)
        series = pd.Series(cl.labels, index=cl.site_index)
        pm = derive_protein_modules(site_clusters=series, site_to_protein=proteins)

        ap_a = ap_assign(
            prediction_matrix=mat,
            site_to_protein=proteins,
            protein_modules=pm,
            site_metadata=metadata,
        )
        ap_mt = build_module_table(
            module_assignments=ap_a,
            kinase_substrates=subs,
            kinase_order=kinase_order,
            assignment_policy="cutoff_binary",
        )
        ap_nw = ap_net(
            prediction_matrix=mat,
            kinase_order=kinase_order,
            kinase_substrates=subs,
            threshold=threshold,
            network_policy="signed",
        )
        ap_exp = build_expanded_table(
            module_assignments=ap_a,
            module_table=ap_mt,
            network_edges=ap_nw.edges,
            kinase_substrates=subs,
            kinase_order=kinase_order,
            assignment_policy="cutoff_binary",
        )

        pp_a = pp_assign(
            prediction_matrix=mat,
            site_to_protein=proteins,
            site_metadata=metadata,
            protein_modules=pm,
        )
        pp_mt = build_signalome_module_table(
            module_assignments=pp_a,
            kinase_substrates=subs,
            kinase_order=kinase_order,
            assignment_policy=SIGNALOME_ASSIGNMENT_POLICY_CUTOFF_BINARY,
        )
        pp_edges, _ = pp_net(
            downstream_score_matrix=mat,
            kinase_order=kinase_order,
            kinase_substrates=subs,
            threshold=threshold,
            network_policy=SIGNALOME_KINASE_NETWORK_POLICY_SIGNED,
        )
        pp_exp = build_expanded_signalome_table(
            module_assignments=pp_a,
            signalome_modules=pp_mt,
            kinase_network_edges=pp_edges,
            kinase_substrates=subs,
            assignment_policy=SIGNALOME_ASSIGNMENT_POLICY_CUTOFF_BINARY,
        )

        assert len(ap_exp) == len(pp_exp)
        common_cols = sorted(set(ap_exp.columns) & set(pp_exp.columns))
        sort_keys = [c for c in ("kinase", "site_key", "module_id") if c in common_cols]
        ap_v = ap_exp[common_cols].sort_values(sort_keys).reset_index(drop=True)
        pp_v = pp_exp[common_cols].sort_values(sort_keys).reset_index(drop=True)
        for c in common_cols:
            a = ap_v[c].astype("object").fillna("<NA>").astype(str).to_numpy()
            p = pp_v[c].astype("object").fillna("<NA>").astype(str).to_numpy()
            n_match = int((a == p).sum())
            assert n_match == len(a), f"column {c!r}: only {n_match}/{len(a)} rows match PhosPy"
