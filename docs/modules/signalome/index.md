# `alphaphos.signalome`

Module detection + kinase network extraction from phosphoproteomics data.

Signalome analysis surfaces a **network / module-level view** of the
phosphosignalling landscape, complementing site-level (limma / ANOVA),
kinase-level (KSEA), and pathway-level (Enrichr) tools that alphaPhos
already ships. Instead of asking *"which sites move?"* or *"which kinases
are active?"*, it asks *"which groups of phosphosites move together, and
which kinases drive each group?"*.

Ships in alphaPhos **0.19.0+** as a clean-room MIT re-implementation of
the algorithm described below.

## Scientific basis + attribution

The algorithm was introduced in [**PhosR**](https://github.com/PYangLab/PhosR)
(Kim et al. 2021, *Cell Reports Methods* 1: 100056, doi:
[10.1016/j.crmeth.2021.100056](https://doi.org/10.1016/j.crmeth.2021.100056)),
an R package for phosphoproteomics.  PhosR ships the original R
implementation of kinase-substrate prediction + signalome extraction.

The alphaPhos port was written **clean-room from published algorithms +
[PhosPy](https://github.com/falconsmilie/phospy)** (a Python port of
PhosR) as a numerical oracle.  PhosPy is GPL-3.0 licensed and alphaPhos
is MIT — no PhosPy code appears in alphaPhos; every algorithm was
re-implemented from method descriptions and validated bit-exact against
PhosPy in [`tests/unit/test_signalome_phospy_parity.py`](../../../tests/unit/test_signalome_phospy_parity.py).

**If you use `alphaphos.signalome` in a publication, please cite:**

> Kim, H.J., Kim, T., Xiao, D., Yang, P. (2021). Protocol for the
> processing and downstream analysis of phosphoproteomic data with
> PhosR. *Cell Reports Methods* 1(6): 100056.

## Pipeline overview

The signalome pipeline is a **six-stage transformation** of a
`(n_sites, n_kinases)` prediction-score matrix into a set of structural
tables:

```
                        ┌─────────────────────────────────────┐
                        │  prediction matrix (sites x kinases) │
                        │  cells = per-site kinase prediction  │
                        │  e.g. Yaffe/Johnson PSSM scores       │
                        └────────────┬─────────────────────────┘
                                     │
                                     ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ 1. Precondition scores (fill NaN with column median)         │
    └──────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ 2. Ward hierarchical clustering (Euclidean distance)          │
    │    scipy.cluster.hierarchy.linkage(method="ward")             │
    └──────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ 3. Select module count k (2..max_k)                           │
    │    Score each k by median within-cluster Pearson correlation. │
    │    Pick k that maximises mean-median-correlation among k's    │
    │    where every cluster has median >= primary_threshold (0.5). │
    │    Fall back to threshold=0.1 if none pass primary.           │
    │    Scale-aware: "exact" for n<=5000, "sampled" otherwise.     │
    └──────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ 4. Protein-level modules via cluster-signature grouping      │
    │    Two proteins get the same module_id iff they have sites   │
    │    in the same set of clusters (0/1 membership vector).      │
    │    (derive_protein_modules)                                   │
    └──────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ 5. Per-site + module-level assignments                       │
    │    Site's module_id = its protein's module_id.               │
    │    Site's top_kinase = argmax of its prediction row          │
    │      (lexicographic tie-break).                              │
    │    Module's top_kinase = majority vote across sites.         │
    │    (build_module_assignments)                                │
    └──────────────────────────────────────────────────────────────┘
                                     │
                        ┌────────────┴────────────┐
                        ▼                         ▼
    ┌───────────────────────────┐  ┌──────────────────────────────────┐
    │ 6a. Module x kinase table │  │ 6b. Kinase-kinase network        │
    │  cells = % of module's    │  │  Pearson correlation across       │
    │  proteins supporting each │  │  kinase columns.                  │
    │  kinase; rows sum to 100. │  │  Edges: |corr| >= threshold (0.5) │
    │  (build_module_table)     │  │  with policy (signed / positive / │
    └────────────┬──────────────┘  │  absolute).                        │
                 │                  │  (build_kinase_network)           │
                 │                  └────────────────┬──────────────────┘
                 └─────────────────────┬─────────────┘
                                       ▼
                     ┌──────────────────────────────────────┐
                     │ 7. Expanded per-focal-kinase view     │
                     │  Denormalised: one row per (kinase,   │
                     │  supported site).  linked_kinases +   │
                     │  regulated_module_ids as JSON columns.│
                     │  (build_expanded_table)               │
                     └──────────────────────────────────────┘
```

Every stage is a standalone function; the whole pipeline is also
available as a single call, [`build_signalome`](#build_signalome).

## Public API

### Single-call entry point

#### `build_signalome`

```python
result = ap.build_signalome(
    prediction_matrix,                # DataFrame (sites x kinases) -- Yaffe PSSM scores or similar
    kinase_substrates=None,            # {kinase: [substrate_site_id, ...]}; auto-derived if None
    site_to_protein=None,              # optional; auto-derived from alphaPhos site keys
    site_metadata=None,                # optional; auto-derived from alphaPhos site keys
    requested_module_count=None,       # skip auto-selection, use this k
    max_modules=10,                    # upper bound on candidate k
    primary_threshold=0.5,             # min cluster-median correlation
    fallback_threshold=0.1,            # relaxed threshold if primary fails
    scoring_mode="auto",               # "auto" | "exact" | "sampled"
    max_exact_sites=5000,              # auto -> sampled boundary
    max_samples_per_cluster=200,       # sampled-mode cluster subsampling
    network_correlation_threshold=0.5,
    network_policy="signed",           # "signed" | "positive_only" | "absolute"
    assignment_policy="cutoff_binary", # or "weighted_top"
    seed=0,
) -> SignalomeResult
```

Returns a frozen `SignalomeResult` with:

| Attribute | Type | Description |
| --- | --- | --- |
| `.site_assignments` | `pd.DataFrame` | One row per site with `module_id`, `top_kinase`, tie diagnostics |
| `.protein_modules` | `pd.Series` | Indexed by protein, values are module IDs |
| `.module_table` | `pd.DataFrame` | `(modules x kinases)` percent-share matrix (rows sum to 100) |
| `.network` | `KinaseNetwork` | `.edges`, `.nodes`, `.candidate_correlations`, `.provenance` |
| `.expanded` | `pd.DataFrame` | Denormalised per-focal-kinase view (JSON list columns) |
| `.clustering` | `SignalomeClusteringResult` | Clustering diagnostics: labels, tree, selection scores |
| `.provenance` | `dict[str, object]` | Settings + counts + module-count-selection reason |

### Minimal example

```python
import alphaphos as ap
import pandas as pd

# Given a phospho AnnData that has gone through the standard pipeline
# up to kinase library scoring.
adata = ap.collapse_sites(psm_df, condition_df=cond)
adata = ap.filter_by_completeness(adata, min_valid_frac=2/3)
adata = ap.impute_hybrid(adata)
adata = ap.add_kinase_windows(adata, fasta_path="resources/fastas/human.fasta")
ap.score_kinases(adata)  # populates adata.varm["kinase_score_ser_thr"]

# Run signalome.  If you have a curated substrate network (OmniPath / PhosphoSitePlus),
# pass it via kinase_substrates=.  Otherwise the pipeline derives one by thresholding
# the prediction matrix at substrate_support_cutoff (default 0.5).
result = ap.build_signalome(adata.varm["kinase_score_ser_thr"])

# What the modules look like
print(result.module_table.head())
#             KIN_A    KIN_B  ...
# module_id
# 1           42.31    38.47  ...
# 2            5.23     6.11  ...
# ...

# Which kinases co-regulate
print(result.network.edges.head())
#   source_kinase target_kinase  correlation
# 0        AURKA        AURKB       0.847
# ...
```

### Per-stage functions (for advanced use)

Skip the orchestrator and call stages directly when you want to
intervene between them (e.g. run clustering with a bespoke method count,
supply your own protein-to-module mapping, etc.):

- `precondition_scores`, `build_ward_tree`, `cut_labels`
- `summarize_profile_degeneracy`
- `select_module_count` → `ModuleCountSelectionResult`
- `cluster_sites` → `SignalomeClusteringResult` (combines the above)
- `derive_protein_modules` → protein → module_id `pd.Series`
- `build_module_assignments` → per-site DataFrame
- `select_kinase_substrates` → auto-derive `kinase_substrates` from a
  prediction matrix by thresholding
- `build_module_table` → `(modules x kinases)` % shares
- `build_kinase_network` → `KinaseNetwork` dataclass
- `build_expanded_table` → per-focal-kinase denormalised view
- `extract_site_to_protein`, `extract_site_metadata` — helpers that
  pull the required inputs from an `adata.var` in alphaPhos conventions

Each is imported off the `alphaphos.signalome` namespace and is
documented via its own docstring.

## Scaling and the `scoring_mode` parameter

The **clustering step** (scipy Ward `linkage` + `cut_tree`) has
`O(N²)` memory and `O(N² log N)` time.  On a modern workstation this is
practical up to `N ≈ 20,000` sites; beyond that expect several minutes
and gigabytes.  For very large studies (proteome-wide baselines), you
may need to downsample before feeding to `cluster_sites`.

The **module-count selection step** additionally computes a
`(N x N)` Pearson correlation matrix to score candidate k's.  This is
the *primary* scaling bottleneck; alphaPhos offers three modes:

| Mode | When it fires | What it does |
| --- | --- | --- |
| `"exact"` | Any `N` (user forces) | Full `N x N` Pearson.  `O(N²)` memory. |
| `"sampled"` | Any `N` (user forces) | Per-cluster subsample to `max_samples_per_cluster` sites; median correlation on the sample.  `O(N * S)` memory. |
| `"auto"` (default) | `N <= max_exact_sites` (default 5000) → exact; else sampled with a `UserWarning`. |

`sampled` mode is deterministic given `seed`.  When `auto` falls back
to `sampled`, provenance records
`scoring_mode_used="sampled"` and users can re-run with `scoring_mode="exact"`
if they have RAM for it and want reproducibility against PhosPy's exact
path.

## Deliberate deviations from PhosPy

None affect numerical results (all verified bit-exact via the parity
suite).  These are packaging / API decisions:

| Deviation | Rationale |
| --- | --- |
| Flat functions, no validator / interpreter / executor contract layer | ~3000 lines of PhosPy's framework isn't needed for our public-API shape.  |
| Return `SignalomeResult` dataclass, not PhosPy's `SignalomeWorkflowResult` | Matches alphaPhos conventions (see `ap.diff_exp_limma` output pattern). |
| Use scipy Ward backend directly, skip PhosPy's exact-Python fallback | scipy is a core dep; PhosPy's pure-Python backend is a niche fallback for tiny-N cases we don't need. |
| Cluster labels are 1-indexed downstream (module 0 = unassigned) | Removes a class of "off-by-one" bugs in module_id joins.  Internally we canonicalise 0-indexed then shift. |

## Testing + PhosPy parity guarantee

We validate signalome outputs against PhosPy on every stage:

- `tests/unit/test_signalome_phospy_parity.py` — bit-exact parity on
  Ward linkage, cluster labels, auto-selected module count,
  `protein_modules`, `module_table`, network `edges` + `nodes`, and
  per-column content of the `expanded` table.  Parity holds at
  multiple sizes (`n_sites=200`, `1000`).  Skipped automatically on
  machines where PhosPy is not installed.
- `tests/unit/test_signalome_clustering.py` — 25 unit tests on
  precondition/Ward/cut-tree/module-count-selection internals.
- `tests/unit/test_signalome_pipeline.py` — 15 tests spanning
  protein-resolution → assignments → module-table → network →
  expanded-view → orchestrator, using synthetic 3-block data with
  ground-truth modules.

Total: **57 signalome tests + full-suite regression** — see
[CHANGELOG.md](../../../CHANGELOG.md) 0.19.0 for the release note.

## License

alphaPhos is MIT.  PhosR is GPL-2.0 (via R package distribution) and
PhosPy is GPL-3.0.  Because signalome is a **clean-room
re-implementation from published methods**, alphaPhos code carries the
MIT license unchanged.  We nonetheless recommend citing both PhosR
(algorithm) and, if you use the parity tests during development,
PhosPy (numerical oracle) — see the "Scientific basis" section above.
