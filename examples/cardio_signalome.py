"""End-to-end signalome walkthrough on the cardiomyocyte DVP dataset.

Runs:

1. Cardio phospho -> collapse -> filter -> impute (standard alphaPhos pipeline).
2. ``ap.add_kinase_windows`` + ``ap.score_kinases`` -- populates
   ``adata.varm["kinase_score_ser_thr"]`` (Yaffe / Johnson 2023 PSSMs).
3. ``ap.build_signalome`` -- module detection + kinase network on that
   prediction matrix.
4. Real-data PhosPy parity: same prediction matrix + reference substrates
   fed into both alphaPhos and PhosPy signalome; per-stage bit-exact check.

Prints:

- Cardio pipeline shapes at each step.
- Module count selected + selection reason.
- Top 5 modules (by size) + their attributed kinases.
- Top 10 kinase-network edges.
- Full stage-by-stage parity vs PhosPy on the real cardio prediction matrix.

Run from the repo root::

    python examples/cardio_signalome.py
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import alphaphos as ap

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s %(message)s")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA = REPO / "test_data" / "proteome_path"
OUT = DATA / "cardio_output"
OUT.mkdir(exist_ok=True)

PHOSPHO_PARQUET = DATA / "cardiomyocytes_dvp_phospho_raw.parquet"
CONDITION_CSV = DATA / "cardiomyocytes_condition_df.csv"
HUMAN_FASTA = REPO / "resources" / "fastas" / "human.fasta"

print(f"alphaPhos {ap.__version__}")
print(f"Data:  {PHOSPHO_PARQUET}")
print(f"Out:   {OUT}\n")

# ---------------------------------------------------------------------------
# 1. Read + collapse + parse conditions
# ---------------------------------------------------------------------------

cond = pd.read_csv(CONDITION_CSV)
psm = ap.read_spectronaut(PHOSPHO_PARQUET)
adata = ap.collapse_sites(psm, condition_df=cond)


def _parse(c):
    if not isinstance(c, str) or c == "BLANK_BLANK_BLANK":
        return {"patient": None, "disease": None, "region": None}
    parts = c.split("_")
    if len(parts) == 3:
        return {"patient": parts[0], "disease": parts[1], "region": parts[2]}
    return {"patient": None, "disease": None, "region": None}


parsed = adata.obs["condition"].map(_parse).apply(pd.Series)
for col in ("patient", "disease", "region"):
    adata.obs[col] = parsed[col].values
adata = adata[adata.obs["disease"].notna()].copy()
print(
    f"Post collapse+condition-parse: {adata.shape}   diseases: {sorted(adata.obs['disease'].unique())}"
)

# ---------------------------------------------------------------------------
# 2. Filter + impute
# ---------------------------------------------------------------------------

adata = ap.filter_by_completeness(
    adata,
    min_valid_frac=2 / 3,
    group_column="disease",
    keep_strategy="any",
    layer="intensity_log2",
)
adata = ap.impute_hybrid(adata)
adata.X = adata.layers["intensity_log2"].copy()
print(f"Post filter+impute:            {adata.shape}   NaN cells: {int(np.isnan(adata.X).sum())}")

# ---------------------------------------------------------------------------
# 3. Kinase windows + kinase-library PSSM scoring
# ---------------------------------------------------------------------------

try:
    from alphaphos.kinase.library import score_kinases

    adata = ap.add_kinase_windows(adata, fasta_path=str(HUMAN_FASTA))
    score_kinases(adata)
except ImportError as exc:
    print(f"\nSKIPPED (kinase-library not installed): {exc}")
    raise SystemExit(0) from exc

st = adata.varm["kinase_score_ser_thr"]
ty = adata.varm.get("kinase_score_tyrosine")
n_st_sites, n_st_kinases = st.shape
n_ty_sites, n_ty_kinases = ty.shape if ty is not None else (0, 0)
print(
    f"score_kinases: {n_st_sites} S/T sites x {n_st_kinases} S/T kinases; "
    f"{n_ty_sites} Y sites x {n_ty_kinases} Y kinases"
)

# ---------------------------------------------------------------------------
# 4. Build signalome (Ser/Thr branch; the larger dataset)
# ---------------------------------------------------------------------------

# Drop rows that were unscored (all-NaN) so clustering has finite input.
mask = st.notna().any(axis=1)
st_active = st.loc[mask]
print(f"S/T sites carried into signalome: {st_active.shape}")

t0 = time.perf_counter()
result = ap.build_signalome(
    st_active,
    scoring_mode="auto",
    substrate_support_cutoff=1.5,  # PSSM percentile score ~ 1.5 (kinase_library scale)
    network_correlation_threshold=0.5,
    seed=0,
)
elapsed = time.perf_counter() - t0
print(f"\nbuild_signalome finished in {elapsed:.1f}s")
print(
    f"  module count selected: {result.provenance['module_count']} ({result.provenance['selection_reason']})"
)
print(f"  scoring_mode used:     {result.provenance['scoring_mode_used']}")
print(f"  n modules with support: {result.provenance['n_modules']}")
print(f"  n network edges:       {result.provenance['n_network_edges']}")
print(f"  n expanded rows:       {result.provenance['n_expanded_rows']}")

# Persist tables for downstream inspection.
result.site_assignments.to_csv(OUT / "signalome_site_assignments.tsv", sep="\t")
result.protein_modules.to_csv(OUT / "signalome_protein_modules.tsv", sep="\t")
result.module_table.to_csv(OUT / "signalome_module_table.tsv", sep="\t")
result.network.edges.to_csv(OUT / "signalome_network_edges.tsv", sep="\t", index=False)
result.network.nodes.to_csv(OUT / "signalome_network_nodes.tsv", sep="\t")
result.expanded.to_csv(OUT / "signalome_expanded.tsv", sep="\t", index=False)

# ---------------------------------------------------------------------------
# 5. Show top modules + top kinases per module + top network edges
# ---------------------------------------------------------------------------

sizes = result.site_assignments["module_id"].value_counts().sort_index()
print(f"\nTop-5 modules by site count:")
top_mods = sizes.sort_values(ascending=False).head(5)
mt = result.module_table
for module_id, n_sites in top_mods.items():
    if int(module_id) == 0:
        continue
    if int(module_id) not in mt.index:
        continue
    row = mt.loc[int(module_id)].sort_values(ascending=False)
    top_k = row.head(3)
    top_kinase_summary = ", ".join(f"{k} {v:.1f}%" for k, v in top_k.items())
    print(f"  module {int(module_id):>2}  n_sites={int(n_sites):>4}  top-3: {top_kinase_summary}")

print(f"\nTop-10 kinase-network edges (by |corr|):")
edges = result.network.edges.copy()
edges["abs_corr"] = edges["correlation"].abs()
top_edges = edges.sort_values("abs_corr", ascending=False).head(10)
print(top_edges[["source_kinase", "target_kinase", "correlation"]].round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Real-data PhosPy parity on the SAME prediction matrix
# ---------------------------------------------------------------------------

try:
    from phospy.contracts.configs import (
        SIGNALOME_ASSIGNMENT_POLICY_CUTOFF_BINARY,
        SIGNALOME_KINASE_NETWORK_POLICY_SIGNED,
    )
    from phospy.science.signalomes.assignments import (
        build_module_assignments as pp_assign,
    )
    from phospy.science.signalomes.clustering.backends.scipy_hierarchical import (
        build_cluster_labels_from_tree as pp_labels_from_tree,
    )
    from phospy.science.signalomes.clustering.backends.scipy_hierarchical import (
        build_cluster_tree as pp_build_tree,
    )
    from phospy.science.signalomes.clustering.protein_modules import (
        derive_protein_modules as pp_derive_pm,
    )
    from phospy.science.signalomes.modules import build_signalome_module_table
    from phospy.science.signalomes.network import build_kinase_network as pp_net
except ImportError as exc:
    print(f"\nSKIPPED PhosPy parity: PhosPy not installed ({exc}).")
    raise SystemExit(0) from exc

print(f"\n----- PhosPy real-data parity on the cardio prediction matrix -----")

# Sub-sample rows to fit under PhosPy's own hard limits (max_full_candidate_scoring
# and pdist memory) if needed.  We compare on the same subset both sides.
max_parity_sites = 4000
if len(st_active) > max_parity_sites:
    print(
        f"  Subsampling {max_parity_sites} of {len(st_active)} sites deterministically "
        "so PhosPy fits under its 5000-site scale guard while comparison stays honest."
    )
    st_parity = st_active.iloc[:max_parity_sites].copy()
else:
    st_parity = st_active

# Rebuild alphaPhos side on the parity subset (so the compared inputs match).
substrates = ap.signalome.select_kinase_substrates(prediction_matrix=st_parity, cutoff=1.5)
proteins = ap.signalome.extract_site_to_protein(adata.var.loc[st_parity.index])
metadata = ap.signalome.extract_site_metadata(adata.var.loc[st_parity.index])
kinase_order = list(st_parity.columns.astype(str))

# --- Stage 1: Ward linkage + labels ---
prep = ap.signalome.precondition_scores(st_parity.to_numpy(dtype=float))
ap_tree = ap.signalome.build_ward_tree(prep)
pp_tree = pp_build_tree(prep)
linkage_diff = float(np.max(np.abs(np.asarray(pp_tree.linkage_matrix) - ap_tree)))

# Cluster labels at the alphaPhos-selected module count
k = int(result.provenance["module_count"])
ap_lab_0 = ap.signalome.cut_labels(ap_tree, n_clusters=k, n_sites=prep.shape[0])
pp_lab = pp_labels_from_tree(cluster_tree=pp_tree, cluster_counts=[k])[k]
labels_match = bool(np.array_equal(ap_lab_0, pp_lab))
print(f"  Ward linkage max-diff:    {linkage_diff:.2e}")
print(f"  Cluster labels (k={k}):    identical={labels_match}")

# --- Stage 2: protein_modules ---
cl = ap.signalome.cluster_sites(st_parity, requested_module_count=k, seed=0)
series = pd.Series(cl.labels, index=cl.site_index)
ap_pm = ap.signalome.derive_protein_modules(site_clusters=series, site_to_protein=proteins)
pp_pm = pp_derive_pm(site_clusters=series, site_to_protein=proteins)
common = ap_pm.index.intersection(pp_pm.index)
pm_match = bool(ap_pm.loc[common].astype(int).equals(pp_pm.loc[common].astype(int)))
print(f"  protein_modules match:     {pm_match} ({len(common)} proteins)")

# --- Stage 3: module_table ---
ap_a = ap.signalome.build_module_assignments(
    prediction_matrix=st_parity,
    site_to_protein=proteins,
    protein_modules=ap_pm,
    site_metadata=metadata,
)
pp_a = pp_assign(
    prediction_matrix=st_parity,
    site_to_protein=proteins,
    site_metadata=metadata,
    protein_modules=ap_pm,
)
ap_mt = ap.signalome.build_module_table(
    module_assignments=ap_a,
    kinase_substrates=substrates,
    kinase_order=kinase_order,
    assignment_policy="cutoff_binary",
)
pp_mt = build_signalome_module_table(
    module_assignments=pp_a,
    kinase_substrates=substrates,
    kinase_order=kinase_order,
    assignment_policy=SIGNALOME_ASSIGNMENT_POLICY_CUTOFF_BINARY,
)
mt_diff = float(np.max(np.abs(ap_mt.to_numpy() - pp_mt.to_numpy())))
print(f"  module_table max-diff:     {mt_diff:.2e}   shape={ap_mt.shape}")

# --- Stage 4: kinase network (edges + nodes) ---
ap_nw = ap.signalome.build_kinase_network(
    prediction_matrix=st_parity,
    kinase_order=kinase_order,
    kinase_substrates=substrates,
    threshold=0.5,
    network_policy="signed",
)
pp_edges, pp_nodes = pp_net(
    downstream_score_matrix=st_parity,
    kinase_order=kinase_order,
    kinase_substrates=substrates,
    threshold=0.5,
    network_policy=SIGNALOME_KINASE_NETWORK_POLICY_SIGNED,
)
key = ["source_kinase", "target_kinase"]
ap_sorted = ap_nw.edges.set_index(key)["correlation"].sort_index()
pp_sorted = pp_edges.set_index(key)["correlation"].sort_index()
edges_same_set = bool(ap_sorted.index.equals(pp_sorted.index))
edges_max_diff = (
    float(np.max(np.abs(ap_sorted.to_numpy() - pp_sorted.to_numpy())))
    if edges_same_set
    else float("nan")
)
ap_n_sorted = ap_nw.nodes.sort_index()
pp_n_sorted = pp_nodes.sort_index()
nodes_ok = (
    ap_n_sorted["degree"].tolist() == pp_n_sorted["degree"].tolist()
    and ap_n_sorted["n_substrates"].tolist() == pp_n_sorted["n_substrates"].tolist()
)
print(
    f"  network edges: n_ap={len(ap_nw.edges)}, n_pp={len(pp_edges)}, "
    f"same-set={edges_same_set}, max-diff={edges_max_diff:.2e}, nodes-match={nodes_ok}"
)

print("\nDone.  Outputs written to", OUT)
