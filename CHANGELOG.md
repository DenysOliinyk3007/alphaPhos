# Changelog

All notable changes to alphaPhos are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While in `0.x`, breaking API changes may appear in any MINOR bump (`0.1 → 0.2`).
`1.0.0` will mark the first stable public API.

## [Unreleased]

_Nothing yet._

## [0.21.0] - 2026-07-16

### Added -- `ap.recommend_pipeline` advisory decision tree

New top-level function that inspects an ``AnnData`` plus a stated
analytical goal and prints a copy-paste-ready pipeline recipe:
Class-I cutoff, per-cell design audit, completeness-filter parameters,
whether to impute, which imputer, and which DE call.

- **`ap.recommend_pipeline(adata, *, goal, data_type, primary_factor,
  secondary_factor=None, subject_col=None)`** — prints to stdout, returns
  ``None``.  Executes nothing beyond a cheap filter dry-run used to
  estimate the recipe's retention.  Read the trace, copy the code,
  apply it yourself.
- **Goals**: ``primary_de`` (main + interaction), ``marginal_de``,
  ``interaction_de``, ``onoff_discovery``, ``profiling``, ``viz_only``.
- **Data types**: ``phospho`` / ``other_ptm`` (Class-I filter on
  ``var["mean_loc_prob"]``), ``proteome`` (skips PTM filter).
- **Design audit**: any ``primary × secondary`` cell with n<3 samples
  is auto-flagged for dropping so ``keep_strategy="each"`` stays viable.
- **Filter sizing**: ``min_valid_n = max(3, min(10, round(0.6 × smallest_cell)))``.
  Capped at 10 because past that the test is well-powered and higher
  thresholds drop good sites without adding rigour.
- **Imputer selection**: KNN for n<50 (below PIMMS's paper floor);
  PIMMS-DAE for n>=300 (~10 s vs ~25 min for KNN at cohort scale);
  PIMMS-DAE when post-filter NaN>=40%; KNN otherwise.  Skipped entirely
  when the DE method handles NaN natively (limma observed-only,
  msqrob2, on/off detection) and post-filter NaN<60%.

Empirical basis: 10-config grouping grid over the sfPhospho 304-fiber
cohort plus MAE benchmarks across EGF (n=6), cardio (n=69), bulk phospho
(n=99), SF phospho (n=304), and bulk proteome (n=367) — see
``scratchpad/sfphospho_grouping_grid.py`` and ``scratchpad/imputation_benchmark_*``.

The advisor is additive.  All existing calls (``filter_by_completeness``,
``impute_pimms/knn/hybrid``, ``diff_exp_*``, ``on_off_detection``) keep
their current signatures and behaviour.

## [0.20.0] - 2026-07-10

### Added -- t-SNE + UMAP in `alphaphos.dimred`

Closes the "UMAP / t-SNE are not yet implemented" limitation flagged in
the README status line since 0.15.  Both share the same API as
`ap.dimred.pca`, so any pipeline that ended in
``ap.dimred.pca(adata)`` can swap in ``ap.dimred.tsne(adata)`` or
``ap.dimred.umap(adata)`` with no other changes.

- **`ap.dimred.tsne(adata, ...)`** -- Barnes-Hut t-SNE via
  ``sklearn.manifold.TSNE``.  No new dependency (sklearn is core).
  Default perplexity 30, auto-clipped when ``perplexity >= n_samples``.
  Deterministic given ``seed``.  Sample coordinates written to
  ``adata.obsm["X_tsne"]``, provenance to ``adata.uns["tsne"]``.
- **`ap.dimred.umap(adata, ...)`** -- UMAP via ``umap-learn`` (McInnes
  et al. 2018).  Guarded behind the new ``[dimred]`` optional extra
  (``pip install "alphaphos[dimred]"``).  Default ``n_neighbors=15``,
  ``min_dist=0.1``, ``metric="euclidean"`` -- match ``umap-learn`` +
  scanpy defaults.  Auto-clips ``n_neighbors`` when it exceeds
  ``n_samples``.  Sample coordinates written to ``adata.obsm["X_umap"]``,
  provenance to ``adata.uns["umap"]``.
- **NaN handling**: neither method has native NaN support.  Both raise
  a clear ``ValueError`` on NaN input with a pointer to
  ``ap.impute_hybrid`` or the ``n_pca_components=`` bridge.
- **Small-n reliability warnings**: emit ``UserWarning`` when
  ``n_samples < MIN_RECOMMENDED_N_TSNE`` (=30) or
  ``n_samples < MIN_RECOMMENDED_N_UMAP`` (=20).  Thresholds derived
  from an empirical sweep on synthetic 2- and 3-group data (10 seeds
  per (n, config), boost = 2-3 std, features = 400): below these,
  silhouette recovery is either variable across seeds (t-SNE) or
  collapses to near-zero (UMAP), while PCA remains reliable at
  n >= 6.  See ``scratchpad/tsne_umap_n_threshold_study.py`` for
  the sweep.  Silence via ``warnings.filterwarnings("ignore",
  category=UserWarning)`` when you know what you're doing.
- **`n_pca_components=` bridge**: when set, ``tsne`` and ``umap`` read
  ``adata.obsm["X_pca"]`` (from a prior ``ap.dimred.pca`` call) as
  their input instead of the raw layer -- the scanpy convention for
  pre-reducing high-dimensional feature matrices before manifold
  learning.  Works with any of the ``pca(handle_missing=)`` backends
  so the whole pipeline stays NaN-tolerant.
- **`ap.dimred.get_tsne_dataframe`, `ap.dimred.get_umap_dataframe`** --
  plot-ready ``DataFrame`` accessors matching the existing
  ``get_pca_dataframe`` pattern: sample coordinates + ``.obs`` metadata
  joined in one frame, method settings in ``.attrs``.

### Docs + release

- README status line dropped the "not yet implemented" clause; module
  table row for ``alphaphos.dimred`` expanded to document the new
  functions + the NaN policy.
- New ``[dimred]`` optional-extra advertised in the install snippet.
- 27 new unit tests in ``tests/unit/test_dimred_manifold.py`` covering
  the settings resolvers, NaN guard, ``perplexity`` / ``n_neighbors``
  clipping, seed determinism, cluster-recovery on 2-block synthetic
  input, the ``n_pca_components=`` bridge, both DataFrame
  accessors, and the new small-n warnings (fires below threshold,
  does not fire at threshold, constants exported).  UMAP tests
  auto-skip when ``umap-learn`` is absent.
- Full suite: **840 tests passing**, ruff check + ruff format --check
  clean.

### Real-data validation

- **Cardio phospho** (n=72, 5 disease groups x 3 tissue regions):
  PCA, t-SNE, and UMAP all recover the same primary structure
  (Healthy vs Disease as the dominant axis).  Silhouette per method:
  PCA -0.01, t-SNE +0.11, UMAP +0.09 for disease labels.  Region is
  not a driver in any method (silhouette ~0), patient is
  anti-clustered (spread across the embedding by tissue region).
  Runtimes on the full 4,687-site matrix: PCA 0.03s, t-SNE 3.1s,
  UMAP 7.9s.  The ``n_pca_components=10`` bridge speeds t-SNE up
  30x and UMAP 60x with negligible loss of structure.
- **EGF +/-** (n=6): PCA gives silhouette 0.51, t-SNE 0.12, UMAP
  degenerates (all points collapse to a tiny cluster,
  centroid separation 0.32 vs 96 for PCA).  Motivated the small-n
  warning thresholds above.

## [0.19.0] - 2026-07-09

### Added -- `alphaphos.signalome` subpackage

Module detection + kinase-network extraction from a per-site kinase-
prediction matrix (e.g. Yaffe PSSM scores from `ap.score_kinases`).
Ports the signalome algorithm from PhosR (Kim et al. 2021,
*Cell Rep Meth*) via PhosPy (github.com/falconsmilie/phospy) as a
numerical oracle -- alphaPhos code is a clean-room MIT re-implementation.

**Single-call entry point**:

```python
result = ap.build_signalome(
    prediction_matrix,                # (sites x kinases) DataFrame
    kinase_substrates=my_network,     # {kinase: [substrate_site_ids]}; auto-derived if None
    scoring_mode="auto",              # "exact" or "sampled" for scale control
    network_correlation_threshold=0.5,
)
```

Returns a frozen `SignalomeResult` with:

- `.site_assignments` -- per-site DataFrame: `module_id`,
  `top_kinase`, tie diagnostics, module-level attribution
- `.protein_modules` -- protein -> module_id `pd.Series`
- `.module_table` -- (modules x kinases) percent-share matrix
- `.network` -- `KinaseNetwork(edges, nodes, candidate_correlations,
  provenance)` from kinase-column Pearson correlations
- `.expanded` -- denormalised per-focal-kinase view for graph
  export tools
- `.clustering` -- Ward tree + module-count selection diagnostics
- `.provenance` -- pipeline settings + counts

**Per-stage functions** for advanced use / testing: `precondition_scores`,
`build_ward_tree`, `cut_labels`, `summarize_profile_degeneracy`,
`select_module_count`, `cluster_sites`, `derive_protein_modules`,
`build_module_assignments`, `select_kinase_substrates`,
`build_module_table`, `build_kinase_network`, `build_expanded_table`,
`extract_site_metadata`, `extract_site_to_protein`, `resolve_scoring_mode`.

### Added -- scale-aware clustering (module-count selection)

Explicit `scoring_mode` on `cluster_sites` and `select_module_count`:

- `"exact"` -- full `N x N` Pearson correlation matrix for candidate
  scoring.  `O(N^2)` memory; recommended for `N <= max_exact_sites`
  (default 5000).
- `"sampled"` -- per-cluster subsampling; `O(N * S)` memory where
  `S = n_clusters * max_samples_per_cluster` (default 200).
- `"auto"` (default) -- picks `"exact"` when `N <= max_exact_sites`,
  otherwise `"sampled"` with a `UserWarning`.

Deterministic under `seed` in sampled mode.  Provenance records
`scoring_mode_used` so callers know when auto-approximation kicked in.

### Numerical parity

Bit-exact stage-by-stage vs PhosPy on real algorithms
(`tests/unit/test_signalome_phospy_parity.py`):

- Ward linkage matrix + cluster labels: identical (`atol == 0`).
- Auto-selected module count: matches PhosPy's rule (filter by
  `min_median_correlation >= threshold`, rank candidates by
  `mean_median_correlation` with smallest `k` as tiebreak).  This
  differs from the naive "smallest passing `k`" strategy I initially
  used -- aligned to PhosPy after finding the divergence.
- `protein_modules` (cluster-signature grouping): identical.
- Module table (percent shares): identical to floating-point precision.
- Kinase network edges (correlations) + nodes (degrees + n_substrates):
  identical.
- Expanded table content: **every column of every row matches PhosPy**
  after common sort, including JSON-encoded `linked_kinases`,
  `regulated_module_ids`, `support_kinases` fields.

### Real-data cardio validation

`examples/cardio_signalome.py` runs the full pipeline on the shipped
cardiomyocyte DVP phospho dataset (72 samples, 20,953 raw sites ->
4,687 after filter+impute -> 4,454 S/T sites after kinase library
scoring; 311 Ser/Thr kinases via Yaffe / Johnson 2023 PSSMs) and
compares stage-by-stage against PhosPy on the 4,000-site subset that
fits under PhosPy's own scale guards.

**Runtime**: `build_signalome` finishes in ~10 s on this matrix
(4,454 sites x 311 kinases, `scoring_mode="auto"` -> `exact` at this
scale).  Selects 8 modules via fallback threshold, produces 1,347
protein-modules with support, 14,326 kinase-network edges, 57,456
expanded rows.

**Biology recovered**: top modules attribute to canonical cardiac
kinases (DNA-PK / CAMK2 in module 1, p38 MAPK family in module 7,
SGK1 / RSK2 / AKT2 in module 9).  Top network edges are established
kinase paralog pairs whose substrates overlap heavily -- CK2A1-CK2A2
(0.994), CDK8-CDK19 (0.991), AMPKA1-AMPKA2 (0.988), CDK17-CDK18,
PAK4-PAK5, CDK12-CDK13, AKT1/2/3, ROCK1-ROCK2, DAPK1-DAPK3.

**Real-data PhosPy parity** on the cardio prediction matrix (4000
sites x 311 kinases):

- Ward linkage max-diff: **0.00e+00**.
- Cluster labels at `k=8`: **identical**.
- `protein_modules` across 1,167 proteins: **exact match**.
- Module table `(133 x 311)` max-diff: **0.00e+00**.
- Network edges: 14,126 = 14,126, same set, max-diff **0.00e+00**;
  nodes (degree + n_substrates): **exact match**.

### Docs + examples

- `docs/modules/signalome/index.md` -- full narrative doc: biology
  intro, algorithm walk-through with ASCII pipeline diagram, public
  API reference, `scoring_mode` scaling guide, deliberate deviations
  from PhosPy, testing + parity guarantee, PhosR/PhosPy attribution
  + citation guidance.
- README module table row expanded to reference the signalome
  subpackage.
- `examples/cardio_signalome.py` -- end-to-end walkthrough on the
  shipped cardio phospho DVP dataset (collapse -> filter -> impute ->
  add_kinase_windows -> score_kinases -> build_signalome), followed by
  a real-data stage-by-stage PhosPy parity check on the same
  prediction matrix.  See "Real-data cardio validation" above.

### Tests

57 new tests across 3 files:

- `test_signalome_clustering.py` (25 tests) -- preconditioning, Ward
  tree, cut_labels, resolve_scoring_mode auto/exact/sampled behaviour,
  profile degeneracy, module-count selection with new PhosPy-aligned
  rule.
- `test_signalome_pipeline.py` (15 tests) -- protein resolution,
  assignments (lex tie-break, module-level majority vote), module
  table (% shares, block-biology recovery), kinase network
  (within-block edges, degree consistency, positive-only policy),
  expanded table (all-kinase coverage, JSON validity), orchestrator
  (defaults, derived substrates, requested_module_count).
- `test_signalome_phospy_parity.py` (17 tests) -- stage-by-stage
  bit-exact parity vs PhosPy at multiple sizes (`n_sites=200`, `1000`),
  including the full expanded table content check.  Skipped
  automatically when PhosPy isn't installed.

Full suite: **813 passing**, 0 failures, 0 lint or format errors.

### Attribution

The signalome method was introduced in PhosR (Kim et al. 2021,
*Cell Reports Methods* 1(6): 100056,
https://doi.org/10.1016/j.crmeth.2021.100056), an R package.  PhosPy
(https://github.com/falconsmilie/phospy) is a Python port of PhosR
under GPL-3.0.  alphaPhos's signalome subpackage is a **clean-room
MIT re-implementation** from published methods, with PhosPy used only
as a numerical oracle during development (no code copied).  Citation
recommendation: cite PhosR for the underlying algorithm.

## [0.18.0] - 2026-07-08

### Added -- ANOVA → downstream enrichment integration

Follow-up to the 0.17.0 multi-contrast / ANOVA statistics: wires
`diff_exp_anova` output cleanly into the existing enrichment consumers,
so a full multi-group workflow (ANOVA → site ORA + gene pathway ORA)
runs end-to-end without user-side glue.

- **`ap.anova_hits(anova, fdr_threshold=0.05)`** — one-liner splitting an
  ANOVA / F-test result into ``(hits, background)`` ready for
  :func:`alphaphos.enrichment.ora`.  Drops rows with NaN FDR from both
  arms (they were not tested and do not belong in the Fisher table).

- **`ap.enrichment.pathway_enrichment(direction="any")`** — new
  direction-agnostic mode.  Skips the up/down sign split, takes every
  gene below ``fdr_threshold`` as a single foreground, does not require
  ``stat_col``.  Works identically on phospho (default key parser) and
  proteome (via ``gene_column=``).  This is the ANOVA-friendly path
  through gene-level pathway ORA.

- **`ap.enrichment.canonicalise_site_ids(keys, drop_unparseable=True)`**
  — public helper bridging alphaPhos
  ``Protein|Gene|Site|Mult`` keys to the ``Protein_AApos`` format used
  by the PTM-DB GMT libraries.  Makes the site-level ORA path a
  three-liner::

      hits, bg = ap.anova_hits(anova, fdr_threshold=0.05)
      hits = ap.enrichment.canonicalise_site_ids(hits)
      bg   = ap.enrichment.canonicalise_site_ids(bg)
      ap.enrichment.ora(hits, bg, libraries=PTM_LIBS)

### Added -- input-shape guards for signed enrichment methods

`pathway_gsea`, `kinase_activity`: detect ANOVA-shape input (missing
``stat_col`` but ``F`` column present) and raise a `ValueError` that
points users to the correct alternative (`pathway_enrichment(direction=
"any")` for direction-agnostic ORA, or `diff_exp_limma_contrasts` for
signed downstream).  Replaces a bare `KeyError` on missing column.

### Added -- cardio ANOVA end-to-end walkthrough

New example demonstrating the full three-layer ANOVA → downstream
pipeline on the cardiomyocyte DVP dataset:

- `examples/cardio_anova.py` -- runnable script.
- `examples/cardio_anova_walkthrough.ipynb` -- annotated notebook, all
  cells pre-executed on the shipped ``test_data/proteome_path/`` data.

Layers: **proteome**, **phospho** (raw), and **normalized**
(phospho / matched-sample parent-protein, log2 subtraction).  The
notebook demonstrates that the normalized layer is where the biology
sharpens: after protein-abundance is removed, the surviving ANOVA hits
are 2–3× enriched for phospho sites that **actually change substrate
function** (`disrupts_ppi` log2FE=+1.20 fdr=0.024, `alters_stability`
log2FE=+1.44 fdr=0.024) — signal that raw phospho ORA does not
surface (top-quartile Ochoa functional sites appear **depleted** there).

Top-10 ANOVA hits at the normalized layer are the textbook
cardiomyopathy panel: **ACTC1** (α-cardiac actin), **MYH7**
(β-myosin heavy chain, the classic HCM/DCM gene), **DSP**
(desmoplakin, ARVC gene), MTOR, NEBL — none of which surface at the
top of the raw phospho layer.

### Docs

- `diff_exp_anova` docstring gets a "Downstream compatibility" section
  laying out which enrichment consumers work directly on ANOVA output
  and which require per-contrast input.
- `pathway_enrichment` docstring documents the new `direction="any"`
  mode and its intended pairing with `diff_exp_anova`.
- README module-table rows expanded for the new API surface
  (`anova_hits`, `canonicalise_site_ids`, `direction="any"`).

### Tests

- 17 new tests across three files:
  - `test_stats_moderated.py`: 4 tests for `anova_hits`
    (fdr split, NaN handling, real diff_exp_anova end-to-end, missing
    column raises).
  - `test_enrichment_pathway.py`: 3 tests for
    `pathway_enrichment(direction="any")` (all-sig foreground, ANOVA-
    shape without stat_col, signed modes still fail loud on
    ANOVA-shape input).
  - `test_enrichment_pathway_gsea.py`: 2 tests for the ANOVA-shape
    guard on `pathway_gsea`.
  - `test_enrichment_ksea.py`: 2 tests for the ANOVA-shape guard on
    `kinase_activity`.
  - `test_enrichment_matching.py`: 6 tests for
    `canonicalise_site_ids` (basic conversion, multi-protein groups,
    unparseable drop / keep, pandas Index input, end-to-end
    `anova_hits → canonicalise → ora`).

Full suite: **756 / 756 passing**.

## [0.17.0] - 2026-07-08

### Added -- multi-contrast + ANOVA moderated statistics

Two new top-level entry points for multi-group / multi-contrast designs
on a clean-room Smyth 2004 empirical-Bayes stack:

- **`ap.diff_exp_anova(adata, condition_column=..., covariates=None,
  block_column=None, layer=..., reference_level=None)`** — moderated
  F-test across all condition levels. One test per feature answering
  "does *any* group differ from the reference?" -- the natural first-pass
  filter for multi-group cohorts before drilling into individual
  contrasts.
- **`ap.diff_exp_limma_contrasts(adata, condition_column=...,
  contrasts={"name": (group_a, group_b), ...}, joint=True|False,
  covariates=None, block_column=None, layer=...)`** — moderated t-test
  for arbitrary user-defined contrasts. `joint=True` (default) fits one
  linear model shared across all contrasts, so the empirical-Bayes
  prior sees the full residual pool (higher power than looping
  `diff_exp_limma` per pair). `joint=False` re-runs `diff_exp_limma`
  per contrast for a manual sanity-check path. Returns a dict of
  per-contrast per-feature DataFrames.

Both support continuous / categorical covariates, paired-block designs
(patient as a fixed effect), and any of the standard preprocessing paths
(filter / impute / batch-correct).

**Internal (public via `alphaphos.stats.*`, MIT-licensed clean-room re-
implementation from Smyth 2004; PhosPy is GPL-3.0 and was used only as
a numerical oracle, not as a source):**

- `alphaphos.stats.design.design_matrix(...)` + typed `DesignMatrix`
  dataclass — categorical + continuous + batch covariates + paired-block
  factor, no-intercept `0 + condition` parameterisation.
- `alphaphos.stats.moderated.fit_f_dist`, `moderate_variance` — Smyth
  2004 empirical-Bayes prior fit + moderation
  (method-of-moments on log-variances via digamma / trigamma).
  Trigamma-inverse via bracketed root-finding
  (`scipy.optimize.brentq`).
- `alphaphos.stats.linear_model.lm_fit`, `contrasts_fit`,
  `moderated_t_test`, `moderated_f_test` — QR-based per-feature OLS,
  contrast reprojection via `Cᵀ (XᵀX)⁻¹ C`, moderated
  t (`estimate / √(σ²_mod · unscaled)`) and moderated
  F (`q_g / (n_contrasts · σ²_mod)`).

**Numerical validation:**

- vs `inmoose` `diff_exp_limma` on real EGF phospho (2 conditions,
  n=15,186 sites): **2,057 = 2,057** significant hits (identical),
  `|t_inmoose − t_joint| = 4.5e-13`, `|F − t²| = 1.7e-13`.
- vs `PhosPy` empirical-Bayes prior fit on the same synthetic
  input: `s0²` diff `2.75e-13`, `df0` diff `6.82e-12`.
- Wald identity F = t² on 2-group designs to `1.7e-13`.

**Real-data biology validation** on the cardiomyopathy dataset:

- Phospho (5 disease groups × 72 samples, 4,687 sites): **1,188 F-sig
  at FDR<0.05**; per-contrast: ICM 1,350 / HCM 1,132 / NICM 1,069 /
  ACM 167. Top F hits SYNPO2L / CKM / SORBS1 / CALU / RBM14 — sarcomere-
  specific, cardiomyocyte-relevant.
- Proteome (5 disease groups × 70 samples, 4,489 proteins): **777
  F-sig at FDR<0.05**; per-contrast: ICM 462 / HCM 361 / NICM 339 /
  ACM 130. Top F hits P00915 (CA1), P69905 (HBA1), P68871 (HBB),
  P02042 (HBD), Q8WX93 (PALLD) — surfaces a **differential-
  vascularisation signal in the protein layer that the phospho layer
  filters out**. Different biology at each layer, same pipeline.

### Added -- 19 stats regression tests

`tests/unit/test_stats_moderated.py`:

- 8 parametric trigamma-inverse round-trips (max err 3.6e-11).
- 3 empirical-Bayes prior recovery tests (synthetic scaled-F, n=5000).
- 3 `lm_fit` sanity tests (single-contrast recovery, QR-vs-inv agreement,
  residual-df arithmetic).
- 1 bit-exact `moderated_t` vs `inmoose` regression.
- 2 `moderated_f` tests including the Wald F = t² identity.
- 1 joint-vs-looped-contrasts equivalence test.

Full suite: **739 / 739 passing**.

### Fixed

- `test_enrichment_pathway.py` + `test_enrichment_pathway_gsea.py`: two
  error-message assertions were still matching the old
  `"No parseable alphaPhos keys"` string after the 0.16.0 `gene_column=`
  change swapped in `"No parseable gene names"`. Corrected in this
  release.

### Docs

- `README.md`: dropped the "No multi-contrast ANOVA / F-test" limitation
  line, refreshed the module table with the new
  `alphaphos.stats.design` + `alphaphos.stats.moderated` +
  `alphaphos.stats.linear_model` rows, and rewrote the top-of-README
  status line to reflect that multi-group designs now ship.

## [0.16.0] - 2026-07-08

### Added -- `alphaphos.proteome` subpackage

New top-level module for proteome (non-phospho) DIA analysis. Two
Spectronaut input formats are supported; both land in the same AnnData
shape so every downstream op that works on phospho AnnData works on
proteome AnnData with the same interface (filter / impute /
batch-correct / limma / PCA / pathway ORA + GSEA).

- **`read_spectronaut_short(path, condition_df=None)`** — reads the
  wide, pre-collapsed protein-group report (~2 MB parquet). Regex-
  extracts run names from the `[N]_<runname>_raw_PG_Quantity` column
  headers, log2-transforms, attaches `condition_df` to `.obs`. Fast;
  trusts Spectronaut's internally-computed MaxLFQ-style `PG.Quantity`.
- **`read_spectronaut_long(path, pg_qvalue_max=0.01, eg_qvalue_max=0.01,
  drop_decoys=True, drop_contaminants=True, ...)`** — reads the
  precursor-level report with the **same QC filters as the phospho
  `read_spectronaut`** (reuses the bundled MaxQuant `contaminants.fasta`).
  Auto-coerces string `EG.Qvalue` → float. Falls back to
  `EG.ModifiedSequence` when `EG.PrecursorId` is absent (older Spectronaut
  exports).
- **`collapse_proteome(prec_df, condition_df=None,
  aggregation_method="sum"|"median"|"top3", min_precursors=None)`** —
  aggregates precursors → protein groups. Output shape matches
  `read_spectronaut_short` so downstream is interchangeable.
- **`phospho_over_proteome(adata_phos, adata_prot, sample_pairing=
  "auto"|dict, missing_protein="drop"|"carry"|"fail",
  protein_group_policy="first"|"best_q")`** — **the killer feature for
  paired studies**: divides phospho intensities by matched-sample
  parent-protein intensities in log2 space to yield "log fraction
  phosphorylated" (removes protein-abundance confounding from phospho
  fold-changes). Sample pairing via DVP well-ID regex
  (`_A1`..`_G11`) by default, or explicit `{phos: prot}` dict.

**Real-data validation** on 77-well cardiomyocyte DVP dataset (Mann lab
single-cell DVP, 5 disease groups × 3 tissue regions):
- Short reader: 77 samples × 5,644 proteins.
- Long reader: 3.58 M → 3.48 M precursor rows after Q + contaminant
  filters (2.82% contaminants).
- Short vs long agreement: Pearson r = 0.92 (methodological +1.9 log2
  offset from `sum(EG.TotalQuantity)` vs MaxLFQ; users pick one path
  per analysis).
- Pairing: 76/76 phospho wells matched to proteome via well-ID; 17,153
  / 20,953 sites matched a protein group.
- Textbook cardiomyopathy biology recovered end-to-end: **MYH7 down**
  (FDR 9×10⁻⁷), DSP down, TTN down; **PLN up in proteome**; **PRKACA
  (PKA) inhibited** (KSEA score −3.85, FDR 0.006) — canonical
  β-adrenergic desensitization; **MAP2K3/6 → p38 MAPK activated**;
  Hallmark Enrichr: **glycolysis / pyruvate metabolism / oxidative
  phosphorylation / mTORC1 signalling** — the failing-heart metabolic
  switch.

### Added -- `gene_column=` parameter on gene-level enrichment

`alphaphos.enrichment.pathway_enrichment` and `pathway_gsea` gain a new
optional `gene_column: str | None` parameter. When set, gene names are
read directly from that column of the diff-exp DataFrame instead of
being parsed from the phospho `Protein|Gene|Site|Mult` site-key format.
Enables the proteome workflow:

```python
result["gene"] = adata_prot.var.loc[result.index, "PG_Genes"].values
ap.enrichment.pathway_enrichment(result, gene_column="gene", ...)
ap.enrichment.pathway_gsea(result, gene_column="gene", ...)
```

Semicolon-joined multi-gene entries take the first name. Default
`None` preserves the phospho-key parser — fully backwards compatible.

### Fixed -- `alphaphos.dimred` variance ratios > 1 on masked data

Both NIPALS and PPCA previously reported per-PC variance that could
exceed the total variance on sparse-NaN input, producing
`variance_ratio` values > 1 (up to ~1000 in extreme cases). On the
real cardiomyocyte 72 × 4,687 phospho matrix, NIPALS-on-raw was
reporting `PC1 = 10.768` (should be ≤ 1).

**Root causes:**
- NIPALS: score update `t_i = Σⱼ(mask·X·p) / Σⱼ(mask·p²)` divides by
  a small denominator when few features are observed for sample `i`,
  inflating `|t|` and therefore `var(t, ddof=1)`.
- PPCA: scores were the SVD-rotated posterior means of the latent
  variable `z ~ N(0, I)` (unit-ish variance under the prior) rather
  than the projection-space scores that sklearn/NIPALS return, so
  per-PC variance was ~10× too small.

**Fix (both methods now use):**
- Per-PC variance = SS of reconstruction at observed cells / (n-1).
- Total variance = SS of centered X at observed cells / (n-1).
- PPCA additionally switches scores to the mask-aware projection
  `X @ loadings`, matching NIPALS's convention so `.obsm["X_pca"]`
  is semantically consistent across all three methods.

**Verified**: on complete data, all three methods now give
`variance_ratio` matching sklearn's to 4 decimals (sklearn 0.4313,
NIPALS 0.4313, PPCA 0.4312 for the top-5 PCs of a 20 × 100 random
matrix). On the real cardiomyocyte 72 × 4,687 phospho matrix with
~30% NaN, both NIPALS and PPCA now give bounded, near-identical
cumulative ratios (~0.376 across top-5). Four regression tests added
in `TestNipals` and `TestPpca`; 36/36 dimred tests pass.

### Added -- proteome walkthrough notebook

`examples/proteome_walkthrough.ipynb` (16 cells, ~75 s wall time)
runs the full proteome pipeline end-to-end on the bundled
cardiomyocyte DVP data. Demonstrates:

- Both short + long reader paths and their consistency check.
- Phospho ↔ proteome pairing via `phospho_over_proteome`.
- `batch_correct_combat` with a synthetic batch.
- `dimred.pca` + per-PC disease-η² on all three layers (proteome /
  phospho / normalized).
- `diff_exp_limma`, `kinase_activity` (KSEA), `pathway_enrichment`
  and `pathway_gsea` on all three layers.

Recovers textbook cardiomyopathy signatures (see proteome subpackage
notes above).

### Tests

**720 tests pass** (up from 716 in 0.15.0): +4 dimred regression
tests for the NIPALS/PPCA variance-ratio bounds. Ruff check +
format both clean.

## [0.15.0] - 2026-07-07

### Added -- `alphaphos.dimred` module (top-level)

New dimensionality-reduction module that answers three exploratory
questions on a phospho ``AnnData``:

1. **Sample-space clustering** -- do replicates group by condition
   (biology) or by batch (artifact)?
2. **Effective dimensionality** -- how many PCs carry real variance?
3. **Imputation impact** -- does the imputation step distort the
   sample-space structure vs a missing-value-aware PCA on the raw
   matrix?

**Three PCA backends** exposed via a single :func:`pca` entrypoint
(`handle_missing="error"|"nipals"|"ppca"`):

- ``"error"`` -- standard SVD via sklearn; requires a complete matrix.
- ``"nipals"`` -- Wold 1966 iterative PCA; skips NaN natively (the
  chemometrics / metabolomics standard, used by MetaboAnalyst and
  mixOmics).
- ``"ppca"`` -- Tipping & Bishop 1999 probabilistic PCA with EM;
  principled MV handling but slower.

Results attach to ``adata.obsm["X_pca"]``, ``.varm["PCs"]``, and
``.uns["pca"]`` (scanpy convention).  ``.uns["pca"]["method"]`` records
the algorithm that actually ran (``"standard"`` / ``"nipals"`` /
``"ppca"``).

**Downstream accessors and diagnostics:**

- :func:`compare_imputation_impact` -- run NIPALS on the raw
  (with-NaN) matrix and standard PCA on the imputed copy, sign-flip
  align each PC, return per-PC correlation + per-sample displacement
  + a plain-language verdict (``OK`` / ``BORDERLINE`` / ``DISTORTED``).
- :func:`get_pca_dataframe` -- plot-ready sample-level DataFrame with
  ``PC1..PCK`` + all ``.obs`` columns joined in.  Ready to hand to
  matplotlib / seaborn / plotly.
- :func:`get_pca_loadings` -- feature-level loadings DataFrame,
  optionally restricted to top-N per PC.
- :func:`loadings_for_enrichment` -- **canonicalized loadings ready
  for the enrichment submodules with no wrappers**.  Returns a
  DataFrame indexed by ``Protein_AApos`` site IDs with PC columns.
  Drops straight into :func:`alphaphos.enrichment.gsea`,
  :func:`alphaphos.enrichment.kinase_activity`, and
  :func:`alphaphos.enrichment.ora`.  Handles multiplicity collisions
  by ``abs_max`` sum-of-squares (or ``dedup="error"``).
- :func:`feature_variance_contribution` -- per-feature share of the
  top-K PC variance (loading² weighted by variance_ratio).
- :func:`sample_distance` -- pairwise NaN-safe sample distance
  (euclidean / correlation / cosine).
- :func:`hierarchical_cluster` -- scipy linkage + leaf-order.

**Validated on the real EGF walkthrough (6 samples × 15,186 sites,
4.4% missing)**: PC1 (35.8% var) cleanly separates EGF vs ctrl.
Imputation-impact verdict: **OK** (min per-PC |r| = 0.972 across the
top-5 PCs) -- imputation preserved the biology.  Standard PCA and
NIPALS on raw produce PC1 correlation of exactly 1.000.

**32 unit tests** (settings, standard/NIPALS/PPCA numerics, NaN
handling, loadings collision dedup across object / pd.StringDtype /
pyarrow backends, imputation-impact heuristic, accessor attrs
survival through ``pd.concat``).

### Fixed -- `alphaphos.enrichment.matching` NaN handling (CI failure)

CI runs on the ``pca`` / ``main`` branches failed 8 tests in
``test_enrichment_matching.py`` with
``AttributeError: 'float' object has no attribute 'split'``.  Root
cause: :func:`_build_lookup_indices` called ``.astype(str)`` on the
``substrate_uniprot`` column of the PTM DB fixture, which has 249
NaN rows.  On pandas 2.2 with object-dtype backing this coerces NaN
to the string ``"nan"`` (handled at the empty-check).  On the pandas /
pyarrow backends CI runs, NaN can leak through as a float or
``pd.NA``, breaking the downstream ``.split(";")``.

**Fix**: ``fillna("").astype(str)`` before iterating, and skip rows
with an empty uniprot field.  Deterministic across all pandas
backends.

**Regression test**: parametric across ``object``, ``string``, and
``string[pyarrow]`` dtypes.

### Fixed -- `alphaphos.orthology.map_to_human` window-format mismatch

:func:`alphaphos.add_kinase_windows` writes windows in
``_LEFT*S*RIGHT_`` format (underscores at protein boundaries + stars
flanking the phospho residue).  :func:`map_to_human` looked up the
raw string against the human window index without stripping these
ornaments, so **every site produced by add_kinase_windows silently
missed** (0/N mapping rate on the primary integration path).

**Fix**: strip ``_`` and ``*`` from the source windows before lookup.
Confirmed on real EGF data: 0/200 → 196/200 mapped after fix
(pipeline_walkthrough) and 195/200 (egf_walkthrough).  The
mouse-SwissProt validation numbers from v0.14.0 remain valid (that
run used a differently-formatted input; the bug affected only the
add_kinase_windows → map_to_human integration path).

**Regression test**: ``test_alphaphos_ornamented_window_format_maps``
covers both ``KST*Y*PQR`` and ``_KST*Y*PQR_`` forms.

### Added -- walkthrough notebooks demonstrate the new modules

- **`docs/benchmark/pipeline_walkthrough.ipynb`** (the canonical
  pipeline) gains four new sections:
    - §7b -- `alphaphos.dimred` (advanced PCA + loadings + imputation-
      impact QC).  Old `apt.tl.pca` kept at §7 for continuity.
    - §12b -- PTM-DB site-set enrichment (`emit_libraries` +
      `load_libraries` + `ora` + `gsea`; also GSEA on PC1 loadings
      via `loadings_for_enrichment`).
    - §12c -- gene-level pathway enrichment (Enrichr /
      gseapy-based `pathway_enrichment` + `pathway_gsea`).
    - §14b -- cross-species orthology (`map_to_human`).
- **`examples/egf_walkthrough.ipynb`** mirrors the same four sections.
- **`README.md`** module table refreshed to include `dimred`.

Both notebooks run end-to-end on the bundled EGF benchmark
(egf_walkthrough: 18/18 cells in ~100s; pipeline_walkthrough: 21/21
cells in ~103s).  Top-loader biology recovered: EGFR Y1172, SHC1
Y427, STAT5A Y694 all in PC1 top loaders on the imputed data.

### Fixed -- pre-existing pipeline_walkthrough API drift

While running the canonical notebook end-to-end, several stale API
calls surfaced.  Fixed alongside the new sections:

- Cell 1: `Path(__file__)` NameError in a Jupyter kernel (egf notebook
  only; pipeline_walkthrough uses a hardcoded ROOT).  egf now falls
  back to `Path.cwd()` with a repo-root sanity check.
- `read_spectronaut(quant_level=..., drop_decoys=..., ...)` -- the
  signature is now `(path, *, advanced=None)`.  Simplified to defaults.
- `collapse_sites(..., return_decision_table=True)` -- collapse now
  returns an AnnData directly; the 3-tuple return and separate
  `to_anndata` step are gone.  Cells 7 + 9 collapsed into one.
- `_, audit = impute_hybrid(...)` -- was discarding the imputed
  AnnData; changed to `adata, audit = ...` and added an explicit
  `adata.X = adata.layers["intensity_log2"].copy()` so downstream
  tools reading `.X` see imputed values.
- Cell 26 `kinase_mea` -- added the same `if kinase_sequence` guard
  that cell 25 (`kinase_enrichment_from_diffexp`) already had.
- Network KSEA cell -- imports moved from `alphaphos.ksea`
  (deleted namespace) to `alphaphos.enrichment` (top-level
  re-exports of `fetch_omnipath_ks_network` + `kinase_activity`).

### Tests

**716 tests pass** (up from 680 in 0.14.0): +32 dimred, +3 matching
NaN parametric (object / string / pyarrow), +1 orthology ornamented-
window regression.  Ruff check + format both clean.

## [0.14.0] - 2026-07-07

### Added -- orthology Phase 3: broader-window verification

- **New setting `verify_window_size`** (default `None`) in
  ``DEFAULT_ORTHOLOGY_SETTINGS``.  When set (typically 15-30) *and* a
  ``source_fasta`` path is passed to :func:`map_to_human`, the module
  runs a second pass over the primary mappings:
    1. For each mapped site, extract the ``±verify_window_size`` window
       from both the source protein and the matched human paralog;
       compute the Hamming distance.
    2. Emit ``.var["verification_mismatches"]`` and
       ``.var["verification_identity"]`` (fraction in [0, 1]) as quality
       signals.  Users can filter on identity post-hoc.
    3. When a site is paralog-ambiguous, re-score *all* paralogs at
       ``±verify_window_size`` and prefer the paralog with the fewest
       verification mismatches as the primary pick, overriding the
       gene-name-consistent tiebreak when a different paralog aligns
       better across the broader flank.

- **New parameter `source_fasta`** on :func:`map_to_human` -- path to
  the source-species FASTA.  Required when ``verify_window_size`` is
  set; silently ignored otherwise.

- **New ``.var`` columns**: ``verification_mismatches`` (int, -1 when
  not verified) and ``verification_identity`` (float in [0, 1] or NaN).

- **New provenance stats**: ``verify_window_size``, ``n_verified``,
  ``n_verify_edge_dropped``, ``n_verify_reassigned``.

### Validated on full mouse SwissProt (946,010 sites mapped at default)

Measured with ``verify_window_size=30`` on 44,690 verifiable
ambiguous-paralog sites:

| verdict | count | share |
| --- | ---:| ---:|
| Primary pick sole best at ±30 | 32,769 | **73.3%** |
| Primary tied with alternatives | 8,759 | 19.6% |
| Alternative would win at ±30 (reassigned) | **3,162** | **7.1%** |

**Interpretation**: gene-name tiebreak agrees with the independent ±30
evidence in 93% of ambiguous cases -- validating the current default.
The 7.1% reassignment rate is the improvement Phase-3 verification
delivers on top: ~3,000 mouse sites (0.33% of all mapped) get their
primary pick corrected.

**±30 identity distribution across mapped sites** (mouse -> human):
- 100% identity: 19.6% of sites
- ≥97% identity: 44.6%
- ≥87% identity: 80.8%
- ≥75% identity: 93.5%

Users can filter on ``verification_identity`` as a quality signal
(higher = stronger orthology evidence).

### Backwards compatibility

Zero-break: ``verify_window_size`` defaults to ``None`` (feature off).
Existing callers get identical behaviour to 0.13.1.  The new
``verification_*`` columns are always emitted (with sentinel values
-1/NaN when the feature is disabled).

### Tests

7 new tests in ``TestVerifyWindowSize`` cover: column emission,
paralog reassignment to a broader-window-better paralog, disabled
default, sentinel values when source_fasta missing, settings
validation (must be positive int and > window_size), stats population.

**680 tests pass** (up from 673).  Ruff clean.

## [0.13.1] - 2026-07-07

### Validation -- strict species-entrapment FDR

Ran the module against **complete bacterial and archaeal proteomes** as
strict entrapment sources (bundled at `resources/fastas/`).  Any hit is
unambiguously a false positive by construction.  This is a stronger
audit than the shuffle-preserve-STY internal decoy (which uses
shuffled *human* sequence -- same AA composition).

**Results at default `max_mismatches=2`**:

| Entrapment source | n_sites | Mapping rate | Empirical FDR |
| --- | ---:| ---:| ---:|
| *M. jannaschii* (archaeon) | 62,932 | 0.151% | 1.2&times;10⁻⁴ |
| *S. solfataricus* (archaeon) | 22,184 | 0.334% | 9.5&times;10⁻⁵ |
| *H. salinarum* (archaeon) | 20,682 | 0.353% | 9.4&times;10⁻⁵ |
| *E. coli* K-12 | 181,350 | 0.283% | 6.6&times;10⁻⁴ |
| *B. subtilis* | 178,754 | 0.223% | 5.1&times;10⁻⁴ |
| *M. tuberculosis* | 104,924 | 0.294% | 4.0&times;10⁻⁴ |
| *S. aureus* | 42,970 | 0.358% | 2.0&times;10⁻⁴ |
| **hamster (positive control)** | **1,401,396** | **55.5%** | (denominator) |
| Internal target-decoy (hamster) | — | — | 4.9&times;10⁻⁵ |

**Key findings**:

- **Signal-to-noise: ~150-370&times;** real biology vs random cross-species
  collisions.
- **Archaea &lt; bacteria** for entrapment rate -- consistent with bacterial
  proteomes sharing more ancient housekeeping with eukaryotes.
- **Internal target-decoy FDR is anti-conservative** by ~2-13&times; vs
  empirical species-entrapment.  Both remain orders of magnitude below
  conventional 1% thresholds; documented honestly in the docs.

### Documentation

- `docs/modules/orthology.md` gains two new subsections:
    - "Strict species-entrapment FDR" -- the full validation table
      above.
    - "FDR calibration caveat" -- honestly explains why the internal
      decoy underestimates the empirical FDR (composition drift
      between bacterial source and shuffled-human decoy).
- Recommends manuscript-methods practice: cite the internal T-D FDR
  as the per-run number, run species-entrapment as a validation
  check (small archaeal proteomes complete in &lt; 1 minute).

### Bundled reference proteomes

Added 7 new FASTAs to `resources/fastas/`: `ecoli.fasta`,
`bacillus_subtilis.fasta`, `mycobacterium_tuberculosis.fasta`,
`staphylococcus_aureus.fasta`, `methanocaldococcus_jannaschii.fasta`,
`sulfolobus_solfataricus.fasta`, `halobacterium_salinarum.fasta`.
Each is UniProt SwissProt reviewed only.

## [0.13.0] - 2026-07-07

### Added -- orthology audit hardening

- **Gene-name-consistent tiebreak** in the primary ortholog pick.  When
  multiple human paralogs match a source window, the paralog whose gene
  symbol matches the source (case-insensitive) is preferred.  Fixes the
  hamster ``Actb`` &rarr; ``ACTA1`` (alphabetically-first) bug -- now
  correctly returns ``ACTB``.  Fall-back is SwissProt-first alphabetical.
- **Residue-class enforcement** on fuzzy matches (`require_center_sty=True`,
  default).  S/T are treated as one residue class (hydroxyl); Y is a
  separate class (aromatic).  A fuzzy hit swapping across classes is
  rejected.  Matches PhosphoSitePlus site-group conventions.  Disable via
  ``require_center_sty=False``.
- **Motif-promiscuity flag** on ``.var``: ``n_paralogs_distinct_genes``
  counts distinct human genes among the matches; ``motif_promiscuous=True``
  when > 3 distinct genes share the source window.  Warns users that the
  window is a shared motif (kinase substrate consensus, SH3-binding,
  common regulatory motif) rather than a specific ortholog.

### Validation -- comprehensive stress-test suite

Reproducible via ``scripts/orthology_audit.py``.  Results demonstrate
methodological soundness:

- **Self-mapping sanity**: 12,937 human sites &rarr; 100% exact-match, 0
  decoy wins.  Fundamental correctness confirmed.
- **Random-null**: 1,000 random ±7 windows &rarr; 0 matches at any
  ``max_mismatches`` from 0-3.  No background collisions on genuinely
  random data.
- **Hamster sensitivity curve** (500 proteins &rarr; 54,293 sites):
    - H=0 &rarr; 40.8% mapped, 0 decoys, FDR=0
    - H=1 &rarr; 59.6% mapped, 0 decoys, FDR=0
    - **H=2 (default) &rarr; 70.7% mapped, 0 decoys, FDR=0**
    - H=3 &rarr; 77.0% mapped, 2 decoys, FDR=5&times;10⁻⁵
    - H=4 &rarr; 81.2% mapped, 19 decoys, FDR=4&times;10⁻⁴
  The response curve proves target-decoy is discriminating: as
  ``max_mismatches`` grows, both target and decoy hits grow, but target
  grows much faster.  Default H=2 is at the FDR=0 boundary.
- **Cross-species graceful degradation** (H=2, 500 proteins each):
    - human self: 100.0%
    - hamster (~90 Mya): 70.7%
    - zebrafish (~450 Mya): 24.5%
    - yeast (~1 Bya negative control): 3.9%
  Mapping rate degrades gracefully with evolutionary distance -- a
  divergent species doesn't force-map, it correctly reports "mostly
  unmapped".

### Documentation

- New `docs/modules/orthology.md` documenting the algorithm, community
  precedent (PhosphoSitePlus site groups, iPTMnet, Ochoa 2020,
  Beltrao 2012), when the method is defensible vs weak, the five
  reviewer-defensibility design choices (window-is-identity,
  target-decoy, gene-name tiebreak, residue-class enforcement,
  motif-promiscuity flag), and the full validation table.
- Six methodological caveats (skipped protein-orthology verification,
  motif promiscuity handling, divergent-species behaviour, decoy
  strategy, paralog-as-biology, mapped-vs-unmapped reporting).
- Wired into `mkdocs.yml` nav under a new "Orthology" section.

### Tests

13 new unit tests covering the A1-A3 code fixes:
- `TestPickCanonical`: source-gene-consistent tiebreak (case-insensitive,
  prefers reviewed within gene hits, falls through to alphabetical when
  no gene match).
- `TestResidueClass`: S/T-vs-Y swap rejected by default, S/T swap
  allowed (both in {ST} class), class check disabled when
  ``require_center_sty=False``.
- `TestMotifPromiscuity`: flagged when > 3 distinct genes share the
  window, not flagged when paralogs are all the same gene (isoforms).

**672 tests pass** (up from 664).  ruff clean.

## [0.12.0] - 2026-07-07

### Added -- orthology Phase 2

- **Fuzzy fallback (Hamming &le; `max_mismatches`)** for `map_to_human`, via
  a pigeonhole segment index.  A window of length ``W = 2*window_size+1`` is
  partitioned into ``max_mismatches + 1`` (nearly) equal segments; any window
  differing by &le; ``max_mismatches`` residues must share at least one
  segment exactly.  Look up candidates via segment hits, Hamming-check each.
  Default ``allow_fuzzy=True``, ``max_mismatches=2``.

  On real CHO data (1,000 hamster proteins &rarr; 105,484 sites):
  - Exact-only: 42,303 mapped (40.1%)
  - Fuzzy H&le;2: **73,044 mapped (69.2%)** -- +30,741 sites recovered
  - Overhead: ~5 s vs exact-only, all-inclusive

- **Per-site q-values** (`.var["mapping_qvalue"]`) via proteomics-style
  target-decoy ranking: for each site, determine the "winner" (target hit vs
  best decoy hit); rank all wins by score (fewer mismatches = better);
  q at rank `i` = cumulative decoy wins / cumulative target wins, with
  monotone-non-decreasing envelope walking down the ranks (Storey-Tibshirani
  q-value convention).  Sites with no target and no decoy hit get NaN.

- **`fdr_threshold` filter** (default 0.01).  Sites whose q-value exceeds
  the threshold are demoted to ``mapping_source="below_fdr"`` -- the human
  annotation columns are cleared, but the raw ``kinase_sequence`` and
  ``mapping_qvalue`` are retained for inspection.  Pass ``None`` to disable
  filtering.

- **Decoy-won demotion**.  Sites where the best decoy hit has fewer
  mismatches than the best target hit are demoted to
  ``mapping_source="decoy_won"``.  These are spurious ortholog assignments
  by definition -- the shuffled human proteome literally fits the source
  window better than the real one -- and would inflate false positives if
  reported.

- **New `.var` columns**: ``mismatches`` (0 for exact match, 1-2 for
  fuzzy, -1 for unmapped/decoy-won/below-fdr) and ``mapping_qvalue``.
  New ``mapping_source`` values: ``"approximate"``, ``"below_fdr"``,
  ``"decoy_won"``.

- **Provenance stats** now include ``n_exact_hits``, ``n_fuzzy_hits``,
  ``n_below_fdr``, ``n_mapped_after_fdr``, ``max_mismatches``,
  ``fdr_threshold``.  Old ``n_target_hits`` / ``n_decoy_hits`` renamed to
  ``n_target_wins`` / ``n_decoy_wins`` (target-decoy semantics rather than
  raw hit counts).

- **Gold-standard iron-law site set** at
  ``test_data/orthology/gold_standard_sites.tsv``.  32 canonical
  mammalian phospho sites (kinase activation loops, receptor
  autophosphorylation, cell-cycle regulatory, translation-control)
  derived from bundled mouse/rat FASTAs against bundled human.
  Covers 26 exact-match cases, 4 1-mismatch cases, 2 3-mismatch cases.

- **Gold-standard regression tests** (4 new tests in
  `TestGoldStandardSites`) that require:
    - All 0-mismatch sites map to the expected human ortholog (either as
      primary mapping or as a paralog side-table entry for ambiguous cases
      like Mapk1/Mapk3 sharing an activation-loop window).
    - Fuzzy-tier (1-2 mismatch) sites are mapped when fuzzy is on.
    - 3+ mismatch sites are correctly held below the default
      ``max_mismatches=2`` cap.
    - Overall recall on mappable (mm &le; 2) sites &ge; 85%.

### Changed

- `resolve_orthology_settings` now validates `allow_fuzzy`,
  `max_mismatches`, and `fdr_threshold`; unknown / out-of-range values
  raise ``ValueError``.
- Log line on completion reports exact + fuzzy counts + demoted counts +
  global FDR estimate.

## [0.11.0] - 2026-07-07

### Added

- **New top-level subpackage `alphaphos.orthology`** for cross-species
  phosphosite ortholog mapping to human.  Motivated by CHO / mouse / rat
  phosphoproteomics: KSEA, OmniPath, PhosphoSitePlus, and Reactome are all
  human-centric, so non-human data cannot run downstream without site-level
  ortholog assignment.
- **`ap.orthology.map_to_human(adata, source_fasta=None, human_fasta=None, ...)`**
  -- window-based site ortholog mapping using the `±window_size` sequence
  context around each phospho site as the identity check.  Same principle
  PhosphoSitePlus uses for its cross-species site groups
  (Hornbeck et al. 2012 *Nucleic Acids Res* 40:D261) and iPTMnet (Huang et
  al. 2018), applied here as a FOSS-reproducible pipeline with the human
  proteome as the target index.

  On CHO source data, this simultaneously answers "which human gene does
  this phospho site correspond to?" and "what's its human position?"
  without requiring a separate protein-level ortholog resolution step --
  the window IS the identity.

- **Target-decoy FDR estimation** built in.  A shuffle-preserving-STY decoy
  index is built alongside the target index (same length, same phospho-
  acceptor density, but scrambled AA composition per protein).  Adapted
  from Elias & Gygi 2007 to sequence-window matching.  Global FDR estimate
  = `n_decoy_hits / n_target_hits`.  On real CHO data (1,000 hamster
  proteins &rarr; 105k sites), global FDR = 7&times;10⁻⁵ (3 decoy hits vs
  42,303 target hits).

- **Paralog handling.**  When a source window matches multiple human
  proteins (e.g. actin family sites &rarr; ACTA1/2/B/G1/G2/..., ~12
  paralogs), all matches are recorded in a side table at
  `adata.uns["orthology"]["paralogs"]`, the primary `.var["human_uniprot"]`
  is set to the SwissProt-preferred canonical match, and
  `.var["ortholog_ambiguous"]` = `True`.

- **Provenance columns on `.var`**: `human_gene`, `human_uniprot`,
  `human_site` (e.g. "S862"), `human_site_key` ("P31749_S862" --
  OmniPath-consumable), `site_conserved` (bool), `ortholog_ambiguous`
  (bool), `mapping_source` (`"exact_match"` / `"unmapped"`), `n_paralogs`.

- **Parquet-backed cache** for the human window index at
  `~/.alphaphos/orthology/`, keyed on `sha256(human_fasta)[:12]` + window
  size + decoy seed.  First run builds the index (~10s for full human
  SwissProt); subsequent runs load in <1s.

- **Bundled reference proteomes** in `resources/fastas/`: human, mouse,
  rat, chinese_hamster, yeast, zebrafish -- covers well-conserved
  mammalian (mouse/rat), the CHO use case (hamster), and divergent
  controls (yeast for negative case, zebrafish for edge-of-conservation
  testing).

### Validated on real data

- **1,000 hamster proteins &rarr; 105,484 ±7 STY windows tested against the
  bundled human proteome (20,597 entries, 1.8M target windows):**
  - Exact-match mapping rate: **40%** (42,303 sites mapped)
  - Ambiguous (paralogs): 2.6% (2,784 sites, 6,929 paralog rows in side
    table)
  - Decoy hits: **3** (global FDR 7&times;10⁻⁵)
  - Biology cross-checks: FICD sites map to human FICD with consistent
    N-term offset; β-actin (Actb) sites correctly ambiguous across the
    actin family; RhoA to small-GTPase family; GATA3 sites primary +
    GATA1/2/4/5/6 paralogs.
- The 60% unmapped is expected on exact matching only -- fuzzy fallback
  (Hamming &le; 2) lands in Phase 2 alongside per-site q-values (currently
  only a global FDR is emitted).

### Scope for v1

Phase 1 (this release) is exact-match only with global FDR.  Phase 2
brings:
1. Fuzzy fallback for the ~60% unmapped tail (Hamming &le; 2 across
   the flanking window).
2. Per-site q-values (requires graded scores, i.e. fuzzy).
3. Ships gold-standard test set of well-known conserved phosphosites
   for regression testing.

### Tests

25 unit tests covering FASTA parsing (both `sp|` and `tr|` prefixes,
multiline sequences, missing GN fallback), S/T/Y window extraction with
edge-truncation, decoy shuffle correctness (preserves STY positions +
AA composition), settings validation, end-to-end mapping on synthetic
FASTAs, paralog ambiguity + SwissProt preference, cache round-trip, and
FDR sanity on random source windows.

## [0.10.3] - 2026-07-07

### Documentation

- **`collapse_precursors.md` gains a "Reporting best practices" section.**
  Motivated by a redundancy audit on the EGF walkthrough: of the raw
  5,524 significant precursors reported by the precursor path, only
  4,199 collapse to unique alphaPhos sites (25% inflation from charge /
  multiplicity / missed-cleavage variants).  The section spells out
  the recommended reporting pattern (aggregate to unique sites before
  quoting counts, use ``aggregate_to_site_level``), gives the exact
  code snippet + Methods-section language, and explains why raw
  precursor counts overstate independence under BH-FDR.
- The "Validated on real EGF data" section is expanded with the
  redundancy audit table + a counter-intuitive negative finding: on the
  EGF dataset, relaxed site-level (``global_max`` + ``cutoff=0.5``)
  recovers only 41 additional hits over strict site-level.  Per-run
  masking is NOT the main driver of the site-vs-precursor gap here --
  it's site-level's ``Protein|Gene|Site|Mult`` key baking multiplicity
  into the feature identity.
- The "Downstream compatibility" section cross-links to the new
  best-practices section so users see the reporting guidance before
  running anything.

## [0.10.2] - 2026-07-07

### Added

- **`alphaphos.aggregate_to_site_level(precursor_result, site_view, ...)`**
  -- new helper that collapses a precursor-indexed diff-exp DataFrame to
  a site-indexed one, so the KSEA hand-off from a precursor-level
  pipeline is one call.  Aggregates via ``max_abs`` (default),
  ``mean``, or ``first`` for the effect stat; ``min`` or ``mean`` for
  the FDR.  Adds an ``n_precursors`` column noting how many precursors
  were collapsed at each site.  Raises when nothing survives the
  bridge (bug-shaped-empty guard).

### Fixed

- **`pathway_enrichment` and `pathway_gsea` now accept BOTH alphaPhos
  key formats.**  The previous implementation ran the strict site-key
  regex ``Protein|Gene|<AA><pos>|M<mult>`` in ``_keys_to_genes``, so
  precursor keys ``Protein|Gene|Peptide|Charge|Mods`` would fail with
  "No parseable alphaPhos keys".  Replaced with a permissive
  pipe-split -- gene is field 1 in both formats -- so precursor-level
  ``diff_exp_limma`` output flows straight into either pathway module
  with zero user-side conversion.  Zero-break for existing site-level
  callers.

### Validated on real EGF data (0.10.2 EGF benchmark, 6 samples, 619k PSMs)

The precursor path is now drop-in compatible with the full downstream
stack.  On the same input:

| stage | site-level | precursor-level (default) |
| --- | --- | --- |
| initial features | 34,227 | 39,323 (2,321 dropped by classI 0.75) |
| after completeness filter (2/3 each) | 15,186 | **38,170** |
| significant at FDR<0.05 | 2,057 | **5,524** (3,932 up / 1,592 down) |
| top KEGG (ORA, up) | ErbB signaling FDR 6.2e-04 | **ErbB signaling FDR 2.0e-06** |
| KSEA top-5 up (via bridge + aggregate) | -- | **EGF, MAPKAPK2, LCK, BRAF, KSR1** |

Site-level's completeness filter drops 55% of features; precursor-level
drops 3%.  Confirms the "abundance-biased culling" pattern the
rationale doc predicted.

### Documentation

- ``docs/modules/preprocess/collapse_precursors.md`` gains a
  "Downstream compatibility" table, an ``aggregate_to_site_level``
  section, and a real-EGF validation table.
- Full end-to-end example now shows both the pathway path (works on
  precursor keys directly) and the KSEA path (bridge + aggregate +
  kinase_activity).

### Tests

- 16 new tests: 13 covering ``aggregate_to_site_level`` (all three
  ``stat_agg`` values, both ``fdr_agg`` values, ``fdr_col=None`` skip,
  ``n_precursors`` column, unique index, drop-no-site, four error
  paths) and 3 covering the permissive gene-extraction in the two
  pathway modules.

## [0.10.1] - 2026-07-03

### Fixed / added

- **`collapse_precursors` now applies a Class I gate by default**
  (`classI_cutoff=0.75`). Precursors whose PEAK localization probability
  across all PSM rows never reached the cutoff are dropped. This is the
  natural analog of the site-level ``localization_strategy="global_max"``
  rule at precursor granularity: a phospho precursor confidently
  localized in ANY run passes; one that was never confident anywhere is
  dropped. Prevents the "keep all phosphoprecursors including
  unlocalized garbage" trap of the initial v0.10.0 release.

  - **Non-phospho precursors bypass the gate** (when
    ``phospho_only=False``) since localization confidence is not a
    meaningful concept for them.
  - **NaN best-loc** (no parseable loc string anywhere) counts as "below
    the cutoff" and is dropped.
  - Comparison is ``>=`` so a precursor exactly at the cutoff is kept.
  - Pass ``classI_cutoff=None`` to disable the gate entirely -- the
    rationale-doc fallback for datasets where the localization metric is
    known unreliable.
  - Validation: ``classI_cutoff`` requires ``annotate_localization=True``
    (no loc info to gate on otherwise). Out-of-range or non-numeric
    values raise ``ValueError``.
  - Provenance: ``adata.uns["alphaphos"]["stats"]`` gains
    ``n_dropped_classI`` and ``classI_cutoff`` for reproducibility.
  - If EVERY precursor gets gated out, ``collapse_precursors`` raises
    with a helpful message pointing at lowering or disabling the gate.

- 9 new tests locking the gate behaviour: default cutoff drops low-loc
  precursors, ``None`` disables, lower cutoffs keep more, boundary is
  inclusive, non-phospho precursors bypass, the raise-on-empty behaviour,
  and the three validation paths (out-of-range, bool coerced-in, missing
  annotate_localization).

### Documentation

- `docs/modules/preprocess/collapse_precursors.md` parameter table now
  documents ``classI_cutoff``; the "design goals" section replaces the
  "no classI_cutoff" line with a precise statement of how the gate
  differs from site-level per-cell masking.

## [0.10.0] - 2026-07-03

### Added

- **`alphaphos.preprocess.collapse_precursors`** -- new precursor-level
  collapse, sibling of `collapse_sites`. Quantifies each identified
  precursor (peptide sequence + charge + modifications) as one feature
  per sample, without residue attribution and without
  localization-probability masking. Trades site-level resolution for a
  more complete, less-punctured quantification matrix -- the right
  choice when the primary question is detection / differential and
  localization is unreliable on low-abundance features (e.g. RTK
  activation loops).

  Motivating case (documented in
  `D:/Projects/miniBinders/CHO_TAB2_H2F/docs/peptide_level_collapse_rationale.md`):
  the TRKA activation-loop precursor `DIYSTDYYR` (pY680), detected in
  56/60 runs at 99.9% localization on Y680, was still culled by
  site-level masking because per-run localization was inconsistent.
  Precursor-level collapse keeps this feature.

  Public API:
    - `ap.collapse_precursors(psm_df, condition_df=..., advanced=...)`
      returns an `AnnData` shape `(n_samples, n_precursors)` matching
      the standard alphaPhos contract (`.layers["intensity_log2"]`,
      `.obs.condition`, `.uns["alphaphos"]`) so `filter_by_completeness`
      / `impute_hybrid` / `batch_correct_combat` / `diff_exp_limma` all
      work unchanged.
    - `.var.index` = alphaPhos precursor key
      ``Protein|Gene|Peptide|Charge|Mods``.
    - `.var` columns include `peptide_sequence`, `charge`, `mods`,
      `n_phospho`, `peptide_start`, `best_localization_prob`,
      `best_localization_pos_peptide`,
      `best_localization_pos_protein` -- localization is annotation
      only, **never** used for filtering.
    - `DEFAULT_PRECURSOR_COLLAPSE_SETTINGS` +
      `resolve_precursor_settings` mirror the settings-validation
      pattern from `collapse_sites`.  Simpler surface -- no
      `localization_strategy`, `classI_cutoff`, or `top_n_attribution`
      knobs.  Adds `phospho_only` (drop non-phospho precursors,
      default `True`) and `annotate_localization` (extract loc info
      for `.var`, default `True`).

- **`alphaphos.precursor_to_site_view`** -- bridge that maps each
  precursor to its best-guess alphaPhos site key
  ``Protein|Gene|<AA><absolute_pos>|M<n_phospho>`` for hand-off to
  KSEA / pathway analyses that need residue attribution. Not a
  re-collapse -- annotation only.  Configurable
  `require_localization` threshold (default 0.75); precursors below
  the threshold get `site_key = NaN` and stay in the AnnData.
  Handles the peptide-local &rarr; protein-absolute position
  arithmetic (``peptide_start + peptide_pos - 1``).

### Documentation

- New `docs/modules/preprocess/collapse_precursors.md` with the full
  parameter table, output schema, worked example, and "when to use
  precursor-level over site-level" guidance.
- `docs/modules/preprocess/index.md` and README module table updated to
  list both collapse variants side-by-side.
- `mkdocs.yml` nav updated.

### Scope

- Spectronaut only for v1.  DIA-NN and FragPipe precursor collapse
  will land in a follow-up.
- 31 new tests covering settings validation, PSM parsing, key
  construction, end-to-end AnnData contract, aggregation modes,
  localization annotation, noise-floor filter, the phospho-only
  filter, and the bridge (including a TRKA-style rescue scenario
  that reproduces the motivating rationale).

## [0.9.2] - 2026-07-03

### Fixed

- **`diff_exp_limma` crashed with `SyntaxError` on condition or covariate
  level names that aren't valid Python identifiers** (e.g. `EGF+` / `EGF-`,
  `1uM` / `10uM`, `KO/WT`, names with spaces or dots). `inmoose.limma.makeContrasts`
  evaluates the contrast string via `eval()`, so `+`, `-`, `.`, `/`, spaces
  and leading digits blew up parsing.

  The fix sanitises non-identifier characters internally (replace with `_`,
  prepend `_` for digit-starts, disambiguate collisions like
  `EGF+`/`EGF-` &rarr; `EGF_` / `EGF__2` by numeric suffix) before values reach
  patsy + inmoose. Sanitisation is a bijective, collision-safe
  implementation detail; the user's original labels round-trip through
  `result.attrs["treatment"]`, `["control"]`, `["contrast_direction"]`,
  and `["contrast_string"]`.

  12 new tests cover the user's exact `EGF+`/`EGF-` case plus `1uM` (digit
  start), spaces (`"not treated"`), slashes (`KO/WT`), collision handling,
  covariate-level sanitisation, sign-convention preservation, and direct
  tests of the sanitiser helpers.

## [0.9.1] - 2026-07-03

### Fixed

- **`impute_hybrid`, `impute_knn_site_based`, `batch_correct_combat` now
  always return the AnnData** -- previously they returned `None` when
  `copy=False` (the default), which silently destroyed the reference on
  `adata = ap.impute_hybrid(adata)`. Non-breaking: existing
  `ap.impute_hybrid(adata)` calls still work; the fix removes the
  footgun for anyone who assigns the return.

  In-place semantics are unchanged (`copy=False` still mutates the input);
  the object returned is now the same AnnData that was mutated (with
  `copy=True`, still a fresh copy). `impute_hybrid(..., return_audit=True)`
  now returns `(adata, audit)` instead of `(None, audit)` when
  `copy=False`.

  Both usage patterns are safe:

  ```python
  ap.impute_hybrid(adata)               # in-place, return ignored
  adata = ap.impute_hybrid(adata)       # equivalent, same object
  new_ad = ap.impute_hybrid(adata, copy=True)   # fresh copy
  ```

  Type hints updated to `-> ad.AnnData` (no more `| None`). Docstrings,
  module docs (`docs/modules/preprocess/imputation.md`,
  `.../batch_correct.md`, `.../index.md`), the EGF walkthrough
  (`.py` + `.ipynb`), and the README quickstart all updated to use the
  assignment pattern by default.

- 11 new unit tests locking the return-contract into place across the
  three functions.

## [0.9.0] - 2026-07-03

### Added

- **`alphaphos.enrichment.pathway_gsea` submodule** -- third
  inhabitant of the elevated `alphaphos.enrichment` namespace.
  Sibling to `pathway` (ORA); same Enrichr libraries, rank-based
  statistics instead of threshold-based.
- **`alphaphos.enrichment.pathway_gsea(diff_exp_result, ...)`** --
  classical Subramanian 2005 preranked GSEA on the same limma output
  KSEA and pathway_enrichment consume. Wraps `gseapy.prerank`
  (permutation-based FDR). Returns a tidy DataFrame with columns
  `library, term, es, nes, p_value, fdr, size, leading_edge, direction`
  sorted by FDR within each library, with leading-edge genes exposed
  as a semicolon-joined string. Provenance stamped on
  `.attrs["provenance"]` (method, stat_col, site_to_gene_agg,
  n_permutations, seed, gseapy version, collapse-stats).
- **Configurable site-to-gene collapse** -- the phospho-specific piece:
    - `site_to_gene_agg="max_abs"` (default) keeps the site with the
      largest `|log2fc|` per gene and retains its signed value.
      Captures peak regulation without averaging away opposing sites.
    - `site_to_gene_agg="top_significant"` keeps the site with the
      lowest per-site FDR per gene. Prioritises statistical evidence
      over effect magnitude.
- **No background parameter** -- unlike ORA, GSEA's null is built by
  permuting set membership; the ranked list itself is the universe.
  Simpler API by design.
- **Per-library isolation** -- each Enrichr library is called
  independently, so a single flaky/missing library (network hiccup,
  404) doesn't kill the whole run. Failures are logged as warnings.

### Validated on real data

- On the EGF walkthrough result (15,185 sites collapsed to 4,048
  ranked genes via `max_abs`), pathway_gsea recovers the
  coherent-motion signature of EGF stimulation with 1000 permutations:
    - **KEGG** at FDR < 5e-03: **GnRH signaling** (NES +2.00),
      **Relaxin signaling** (NES +2.00), **TNF signaling** (NES +1.98),
      **Prolactin signaling** (NES +2.01) -- all RTK/MAPK-convergent
      pathways.
    - **Reactome** top hits: **Signaling By ERBB4** (NES +1.82),
      **PI5P/PP2A/IER3 Regulate PI3K/AKT Signaling** (NES +1.76),
      **Constitutive Signaling By Aberrant PI3K In Cancer**,
      **Signaling By Insulin Receptor** -- textbook EGF/RTK biology.
- Complements the ORA path: pathway_enrichment surfaces
  ErbB/MAPK/Insulin as top-ORA hits (threshold-based), while
  pathway_gsea additionally surfaces coherent-motion pathways
  (GnRH/Relaxin/TNF signaling) that don't cross the ORA hit
  threshold but move as a set.
- 24 tests: 22 mocked-gseapy unit tests (schema, per-library
  isolation, both collapse strategies with known ground truth,
  NaN handling, error paths, provenance) plus 2 real-Enrichr-API
  integration tests.

## [0.8.0] - 2026-07-03

### Added

- **`alphaphos.enrichment.pathway` submodule** -- second inhabitant of
  the elevated `alphaphos.enrichment` namespace.
- **`alphaphos.enrichment.pathway_enrichment(diff_exp_result, ...)`** --
  gene-level pathway ORA on significant phosphosites. Splits up- vs
  down-regulated hits by default (`direction="split"`), extracts gene
  symbols from alphaPhos `Protein|Gene|Site|Mult` keys, and runs
  hypergeometric enrichment against the Enrichr gene-set libraries via
  `gseapy` (Fang et al. 2023). Returns a tidy DataFrame with columns
  `direction, library, term, overlap, p_value, fdr, odds_ratio,
  combined_score, genes, n_foreground, n_background` sorted by FDR
  within each direction+library slice. Provenance stamped on
  `.attrs["provenance"]`.
- **Rigorous background handling** -- the scientifically-critical piece:
    - `background="phosphoproteome"` (default) uses every parseable
      gene in `diff_exp_result.index` (all tested sites); this is the
      right choice when no proteome data is available.
    - `background=[genes...]` or `background=<DataFrame>` accepts a
      caller-supplied list, e.g. proteome-identified genes, which is
      the rigorous choice when it is available.
    - `background="genome"` is a deliberate opt-in and emits a
      `UserWarning` -- genome-wide background inflates enrichment for
      a phospho experiment because the detection universe is much
      smaller than the genome.
- **Default library set** for human: GO_Biological_Process_2023,
  GO_Molecular_Function_2023, GO_Cellular_Component_2023,
  KEGG_2021_Human, Reactome_2022, MSigDB_Hallmark_2020 (KEGG_2019_Mouse
  for mouse). Anything Enrichr hosts is fair game via
  `libraries=[...]`.
- **New `[enrichment]` optional-extra** bundling `gseapy>=1.1`,
  `decoupler>=2.0`, `omnipath>=1.0` so the whole enrichment stack
  installs with `pip install 'alphaphos[enrichment]'`. Import is
  guarded; missing gseapy raises a clear `ImportError` pointing at
  the extra.

### Validated on real data

- On the EGF walkthrough result (15,186 sites; 1,255 up + 567 down at
  FDR 5%; 4,048 phosphoproteome background genes) the up-regulated set
  cleanly recovers the textbook EGF response:
    - **KEGG**: ErbB signaling pathway (FDR 6.2e-04), Insulin signaling,
      VEGF signaling, MAPK signaling pathway (all < 5% FDR).
    - **Hallmark**: PI3K/AKT/mTOR Signaling (FDR 3.0e-02).
    - **Reactome**: NTRK signaling, AP-1 transcription-factor activation.
- Down-regulated set recovers the counter-regulated small-GTPase
  signaling changes: RAC1/RHO/CDC42 GTPase cycles (Reactome, FDR
  1e-05 -- 5e-04), Regulation of Small GTPase Mediated Signal
  Transduction (GO BP, FDR 1e-02), TGF-beta Signaling (Hallmark,
  FDR 1.7e-02).
- 27 tests: 25 mocked-gseapy unit tests (schema, direction split/up/
  down/both, background resolution across all input types, gene
  extraction edge cases, empty-foreground handling, custom library
  lists, error paths) plus 2 real-Enrichr-API integration tests.

## [0.7.0] - 2026-07-02

### Added

- **`alphaphos.enrichment` elevated to a top-tier namespace**; the
  submodule `alphaphos.enrichment.ksea` is the first inhabitant.
- **`alphaphos.enrichment.kinase_activity(diff_exp_result, ...)`** --
  decoupler-py ULM (default) / MLM kinase-activity inference in one
  clean call. Auto-detects and canonicalises alphaphos
  ``Protein|Gene|Site|Mult`` keys to the OmniPath ``Protein_AApos``
  format that decoupler expects, dedups multiplicity variants by
  ``|log2fc|``, and returns a per-kinase DataFrame with
  ``score, fdr, n_substrates, direction``. Provenance
  (method, network, seed, decoupler version, n_kinases_tested) is
  stamped on ``result.attrs["provenance"]``. Wraps
  ``decoupler-py`` (Badia-i-Mompel et al. 2022 *Bioinformatics
  Advances* 2:vbac016) -- the modern replacement for the classical
  Wiredja 2017 KSEA z-score.
- **Two bundled kinase-substrate networks** + BYO:
    - ``network="omnipath"`` (default) -- fetches Enzsub via the
      ``omnipath`` package (~50k edges, community standard). Cached
      to parquet on first use.
    - ``network="ptm_db"`` -- built ad-hoc from our curated PTM
      functional database with per-edge ``curation_confidence``
      tier filtering.
    - ``network=<DataFrame>`` -- caller-supplied edge table
      validated against the decoupler schema
      (``source``, ``target``, ``weight``).
- **MLM error message steers users to a fix.** MLM's regression is
  rank-deficient on the full OmniPath network (many kinases share
  substrates). When that happens the raw ``numpy.linalg.LinAlgError``
  is caught and re-raised as a ``RuntimeError`` pointing at ULM or a
  higher-confidence network subset. MLM works cleanly on the
  high-confidence PTM-DB slice.

### Removed

- **`alphaphos.ksea` module removed entirely** (hard-cut,
  pre-alpha). Its functions are superseded by
  `alphaphos.enrichment.kinase_activity` and the network helpers in
  `alphaphos.enrichment.ksea.network`. In particular the stale
  `alphaphos_site_to_omnipath` converter (broken since the 0.1.0
  site-key format migration) is gone -- key conversion now goes
  through the current
  `alphaphos.enrichment.matching.parse_alphaphos_key`.
- `alphaphos.ksea.kinase_activity_ora` / `kinase_activity_gsea`
  removed -- redundant with the generic
  `alphaphos.enrichment.ora` / `.gsea` engines which work against any
  library including a kinase-substrate network wrapped as
  `{"kinase_substrate": build_kinase_substrate_library()}`.

### Validated on real data

- End-to-end on the EGF walkthrough result (15,186 sites, 84.2%
  matching the DB, 255 kinases scored):
    - Top-15 kinases by ULM score include **EGF, MAPKAPK2, MAP2K3,
      MAP3K8, BRAF, RAF1, JAK2, LCK, HCK, KSR1** and the
      DUSP1/4/8/16 phosphatase feedback family -- textbook EGF
      biology.
    - **GSK3B** correctly comes back as significantly down (AKT
      inhibits GSK3B downstream of EGF).
    - **MLM on the PTM-DB high-confidence subset** (~510 kinases,
      curation_confidence == "high") ranks **MAPKAPK2, EGFR, MAPK1
      (ERK2), MAP2K1 (MEK1), BRAF** in the top 10 -- the canonical
      RAS/RAF/MEK/ERK cascade.
    - Top-20 overlap between ULM on OmniPath and ULM on PTM-DB is
      14/20 kinases -- strong network-agnostic biology.
- 21 unit + integration tests exercise validation, both networks,
  synthetic recovery of known direction, auto-conversion of
  alphaphos keys, and every error path.

## [0.6.3] - 2026-07-02

### Added

- **Proteome FASTAs now committed** at ``resources/fastas/``: human
  (13.7 MB), mouse (11.7 MB), Chinese hamster / CHO (11.6 MB). Sourced
  from UniProt (CC-BY 4.0). The ``ap.add_kinase_windows`` step and the
  ``examples/egf_walkthrough`` notebook now work out of the box for any
  collaborator who clones the repo -- no extra download step.
  ``resources/fastas/*.fasta`` was removed from ``.gitignore``.
  Attribution added to the README.
- **``examples/egf_walkthrough.ipynb``** -- Jupyter notebook version of
  the walkthrough, paired with the existing ``.py`` percent-format file.

## [0.6.2] - 2026-07-02

### Added

- **End-to-end EGF walkthrough notebook** at
  ``examples/egf_walkthrough.py`` (jupytext percent format; opens as a
  Jupyter notebook and pairs with an ``.ipynb`` if you convert). Runs
  the full pipeline on the ``test_data/benchmark/EGF_diff_exp.tsv``
  Spectronaut report:
    read -> collapse -> filter -> impute -> QC dashboard ->
    kinase-window FASTA annotation -> ComBat batch correction (with a
    fake batch to show the correction) -> limma differential
    expression -> volcano plot -> Yaffe-library PWM scoring -> KSEA
    kinase-activity inference via OmniPath. Each optional-dep section
    is guarded with try/except. Verified end-to-end: top-KSEA kinases
    are the canonical EGF signature (EGF receptor, MK2, Src-family,
    JAK2, BRAF).

### Removed

- **`[r]` optional extra (rpy2)** -- dead scaffolding. No code in the
  package ever imported ``rpy2``; the ``alphaphos.enrichment`` module
  (which the extra was reserved for) is still an empty ``__init__.py``.
  Also removed the paired ``requires_r`` pytest marker and the README
  references.  If R-backed PTM-SEA is added later, the extra can come
  back at that point.

### Known limitations surfaced by the walkthrough

- **``alphaphos.ksea.alphaphos_site_to_omnipath`` is stale**: still
  expects the pre-0.1.0 ``Protein~Gene_Site_Mult`` site-key format
  (with ``~`` and ``_`` delimiters), but the current pipeline emits
  ``Protein|Gene|Site|Mult`` (with ``|``).  The notebook works around
  this by building the OmniPath-format identifier manually and passing
  ``convert_site_ids=False`` to ``kinase_activity_ulm``.  A future
  patch should update the converter to accept the new format.

## [0.6.1] - 2026-07-02

### Tests

- **Introduced a real+spiked integration-test framework** for the
  collapse module.  ``tests/data/egf_mini.tsv`` (1.2 MB filtered
  slice of the EGF benchmark, 9 real proteins, all 6 samples),
  ``tests/fixtures/synthetic_psms.py`` (a factory that emits
  Spectronaut-format PSM rows byte-identical to real exports), and
  ``tests/integration/test_collapse_spiked.py`` (31 tests covering
  M1-M4 multiplicity, mixed modifications, all three localization
  strategies, all four aggregation methods, multi-protein groups,
  weird biology, and rejected bad inputs).  These tests spike
  synthetic PSMs into the real EGF data and assert on the exact site
  keys, intensities, and multiplicities that collapse emits.
- **Pruned the unit test suite** from 454 to 396 tests (-13%).
  Consolidated near-identical tests into ``@pytest.mark.parametrize``
  matrices; removed defensive trivialities (empty/None/NaN input
  guards where the behavior is obvious); removed unit tests whose
  behavior is now covered by the spike-in integration suite. Biggest
  cuts:
    - ``test_collapse_end_to_end.py``: 27 → 14 (13 tests overlapped
      with integration).
    - ``test_collapse_aggregation.py``: 13 → 8 (the four aggregation
      dispatch tests are covered end-to-end).
    - ``test_collapse_masking.py``: 8 → 4 (the three localization
      strategies are covered end-to-end).
    - ``test_attribution.py``: 54 → 35 (parametrized loc-prob and
      precid parsers; dropped tautologies like
      ``test_result_always_sorted_ascending``).
  No test that exercised a real invariant was removed.

## [0.6.0] - 2026-07-02

### Added

- **ComBat batch-effect correction**
  (`alphaphos.preprocess.batch_correct.batch_correct_combat`, re-exported
  as `ap.batch_correct_combat`) with `DEFAULT_COMBAT_SETTINGS`. Wraps
  `inmoose.pycombat.pycombat_norm`: empirical-Bayes adjustment of
  per-site batch means and variances on the canonical
  `intensity_log2` layer. Preserves biology via `covariates=[...]`.
- **Pre-correction layer kept by default**
  (`keep_precombat=True`) -- ComBat writes to `intensity_log2` and
  copies the pre-correction values to
  `layers["intensity_log2_precombat"]`. Both are then addressable
  from downstream analysis, per AnnData conventions.
- **Provenance stamp**:
  `adata.uns["alphaphos"]["batch_correction"]` records method,
  batch column, covariates, `ref_batch`, `par_prior`, `mean_only`,
  target layer, batch sizes, and the inmoose version.
- **Double-correction guard in `diff_exp_limma`**: reads the
  provenance stamp and refuses to run when the user passes the
  same batch column as a limma covariate AND the tested layer is
  the ComBat-corrected one. Error message spells out both valid
  fixes:
    A. (statistically preferred) test the `intensity_log2_precombat`
       layer with the batch covariate;
    B. test the ComBat-corrected layer without a batch covariate
       (fine for visualisation, anti-conservative for stats).

### Validation guards
`batch_correct_combat` refuses to run when inputs would silently mislead:

- Fewer than 2 batch levels OR a batch with <2 samples (ComBat
  cannot estimate a batch effect from a singleton).
- NaN in the target layer -> pointer to
  `filter_by_completeness` + `impute_hybrid`.
- Non-log-scale data (median > `LOG_SCALE_MEDIAN_CEILING = 30`).
- Duplicate `var_names`.
- Covariate equal to the batch column (self-adjust).
- `ref_batch` that is not actually a level.
- Unknown `advanced` keys.
- Warns when any batch has fewer than 3 replicates
  (`MIN_REPLICATES_WARNING = 3`) or when a covariate level appears
  in only one batch (confounded).

### Refactored

- **`LOG_SCALE_MEDIAN_CEILING` and `MIN_REPLICATES_WARNING` hoisted
  to `alphaphos.constants`** -- shared between
  `stats.diff_exp` and `preprocess.batch_correct` rather than
  duplicated per module.

### Validated on real data
- On the EGF benchmark with an injected synthetic batch effect
  (`b1` = 4 samples, `b2` = 2 samples with a +2 log2 site-wide shift):
    - Path A (raw layer + limma batch covariate): 1,839 sig sites at
      FDR<0.05.
    - Path B (ComBat-corrected + limma, no batch covariate): 3,593
      sig sites -- strict superset of Path A (0 unique to Path A,
      1,754 extra in Path B).
  The Path-B > Path-A gap is the textbook anti-conservative-p-value
  phenomenon (ComBat drains batch variance out of the residuals,
  inflating t-stats), which is why the guard defaults to steering
  users toward Path A for statistics.

## [0.5.2] - 2026-07-02

### Fixed

- **CI green again after 0.5.0 / 0.5.1.** `patsy` was imported
  unconditionally at the top of `alphaphos/stats/diff_exp.py`, but that
  module is re-exported from ``alphaphos/__init__.py`` -- so a bare
  `import alphaphos` on a core-only install (as CI does) crashed with
  `ModuleNotFoundError: No module named 'patsy'` before any test could
  run. Same failure mode as the 0.1.1 pyarrow break. Both `patsy` and
  `inmoose.limma` are now behind a single ``try/except`` at module top
  under a `_HAS_STATS_DEPS` flag; `diff_exp_limma()` checks the flag and
  raises a clean `ImportError` pointing at `pip install alphaPhos[stats]`
  when either is missing. `tests/unit/test_stats_diff_exp.py` gets a
  module-level `skipif` on the same flag, so the file skips cleanly on
  CI rather than erroring at collection time.

## [0.5.1] - 2026-07-01

### Changed

- **`impute_hybrid` and `impute_knn_site_based` default `layer` is now
  `"intensity_log2"`** (was `None`, which meant ``.X``). This closes a
  silent-wrong-result trap: `collapse_sites` initialises both `.X` and
  `layers["intensity_log2"]` with the same values, but the two diverge
  once any layer is mutated. With the old default, calling
  `impute_hybrid(adata)` would fill `.X` while leaving the log2 layer
  full of NaN -- and `diff_exp_limma` (which defaults to
  `layer="intensity_log2"`) would then trip the NaN guard on the
  stale layer. If the NaN guard were relaxed in the future, wrong
  stats would be reported without any warning. Aligning the default
  makes the whole `filter -> impute -> diff_exp` chain read/write the
  same slot. Pass `layer=None` explicitly to keep the old behaviour.

### Validated on real data

- End-to-end pipeline verified on the EGF diff-exp benchmark (Spectronaut
  TSV, 3x withEGF vs 3x woEGF, 621k PSMs -> 34,227 collapsed sites -> 15,186
  after ``keep_strategy="each"`` at ``min_valid_frac=2/3`` -> 3,999 cells
  imputed -> 2,057 sites at FDR<0.05 with limma). Top hits include the
  canonical EGF-pathway substrates (MAPK7/ERK5 activation loop
  T733/S731, EP300, DAB2, WIPF1, RSK1 hydrophobic-motif S732). Site
  positions faithfully carry Spectronaut's ``EG.ProteinPTMLocations``
  through the collapse.

## [0.5.0] - 2026-07-01

### Added

- **Two-group moderated t-test via limma**
  (`alphaphos.stats.diff_exp.diff_exp_limma`, re-exported as
  `ap.diff_exp_limma`) with `DEFAULT_STATS_SETTINGS`. Wraps
  `inmoose`'s limma port: `lmFit -> contrasts_fit -> eBayes ->
  topTable`. Returns a per-site DataFrame indexed by
  `adata.var_names` with columns
  `log2fc, se, t_stat, p_value, fdr, B, ave_expr`. The contrast
  convention (`log2fc = mean(treatment) - mean(control)`) is stamped
  on `result.attrs["contrast_direction"]` so downstream plotting code
  never has to guess the sign.
- **Batch / nuisance adjustment** via `covariates=` (list of
  `adata.obs` columns) -- design becomes `~ 0 + condition + covar_1
  + ...`. Categorical covariates are patsy-dummy-coded automatically.
- **`[stats]` optional extra** in `pyproject.toml`
  (`inmoose>=0.9.1`, `patsy>=0.5`). Core install stays lean; the
  stats module ImportErrors with a clear install hint if the extra
  is missing.

### Validation guards
`diff_exp_limma` refuses to run when the inputs would silently give
wrong results:

- **Non-log-scale input**: rejects a layer whose median exceeds
  ~30 (a linear-scale MS intensity is typically 1e6+; a log2 one is
  10-30).
- **NaN in the tested layer**: rejects and points at
  `filter_by_completeness` + `impute_hybrid`.
- **Duplicate `var_names`**: would silently misalign rows after the
  `topTable` sort; rejected upfront.
- **Log** a warning when either group has fewer than 3 replicates
  (moderated statistics still run but interpretation is fragile).

### Known
- **ANOVA / multi-contrast F-test intentionally deferred.** inmoose
  0.9.1 has a `KeyError: -2` bug in `_ebayes` when `df_prior` comes
  back as scalar `inf` (pathological homogeneous variance across
  features). Real biological data with heterogeneous per-site
  variance does not hit this, but the multi-contrast path has not
  yet been regression-tested against R limma. When
  `diff_exp_limma` does hit that path, the raw `KeyError: -2` is
  translated into a targeted `RuntimeError` explaining the cause.

## [0.4.0] - 2026-07-01

### Added

- **Site-completeness filter** (`alphaphos.preprocess.filter.filter_by_completeness`,
  re-exported as `ap.filter_by_completeness`) -- native alphaPhos filter,
  no external dep. Three group-handling strategies:
    - ``"all"`` (default; no ``group_column``): global valid-fraction filter.
    - ``"any"`` (needs ``group_column``): keeps sites passing in >=1 group
      -- preserves condition-specific sites.
    - ``"each"`` (needs ``group_column``): keeps sites passing in EVERY
      group -- strictest, safest before differential testing.
  Uses ``min_valid_frac`` (fraction of NON-NaN samples required, in ``[0, 1]``)
  rather than the older ``max_missing`` convention. Always returns a new
  ``AnnData``; input is untouched. Layer-aware: computes missingness on
  the specified layer (default ``.X``), but the drop applies to every
  layer / obsm / varm via AnnData subsetting.
- **`ap.impute_hybrid` and `ap.impute_knn_site_based` re-exported at the
  top level** (they were already in ``alphaphos.preprocess.impute``;
  now they show up in ``ap.*`` for consistency with ``ap.filter_by_completeness``).

### Changed

- **`impute.py` no longer references `alphapepttools`**. The docstring
  and the "features have no observed values" error now point at
  ``alphaphos.filter_by_completeness`` -- alphaPhos is self-contained
  for the filter -> impute path.

## [0.3.0] - 2026-07-01

### Added

- **FragPipe DIA site-abundance reader**
  (`alphaphos.io.fragpipe.read_fragpipe_sites`, re-exported as
  `ap.read_fragpipe_sites`) with `DEFAULT_FRAGPIPE_IO_SETTINGS` and
  `resolve_fragpipe_io_settings`. Consumes FragPipe's
  ``abundance_{single,multi}-site_MS{1,2}quant_{None,Norm}.tsv`` and
  returns a fully-packaged `AnnData` directly -- FragPipe already emits
  site-level matrices via IonQuant + PTM-Prophet, so we trust their
  collapse and skip our own pipeline for this engine. Same AnnData
  contract as `collapse_sites` output (canonical `Protein|Gene|Site|Mult`
  keys, `layers["intensity_log2"]`, `.var` schema with `sequence_window`
  and auto-converted `kinase_sequence`). Class-I filter on
  ``Best Localization >= 0.75`` on by default; sample-column paths
  auto-normalized to base names.
- **`alphaphos.constants`** extended with FragPipe abundance column
  constants (`FRAGPIPE_*` and `FRAGPIPE_META_COLUMNS`).

## [0.2.0] - 2026-07-01

### Added

- **DIA-NN reader** (`alphaphos.io.diann.read_psm`, re-exported as
  `ap.read_diann`) with `DEFAULT_DIANN_IO_SETTINGS` and
  `resolve_diann_io_settings`. Ports the manuscript-validated pipeline
  from `nanoPhos_env/nanoPhos_Figure3_DIANN_v00.ipynb`. Column pruning
  at read time (same pattern as `read_spectronaut`), engine-specific QC
  filter chain (`PG.Q.Value`, `Global.PG.Q.Value`, `Lib.PG.Q.Value`,
  `Quantity.Quality`, `PG.MaxLFQ.Quality`), phospho-only filter
  (`UniMod:21`), and adapter that emits Spectronaut-canonical PSM
  columns so the SAME `collapse_sites` pipeline runs on both engines.
  Requires DIA-NN >= 1.9 (needs `Protein.Sites` and
  `Site.Occupancy.Probabilities` columns).
- **`search_engine="Diann"`** is now accepted by `collapse_sites`; the
  MS2 → MS1 → auto fallback chain correctly falls to MS1 for DIA-NN
  (DIA-NN doesn't split MS1/MS2 quant the same way Spectronaut does).
- **`alphaphos.io.schemas` extended** with `REQUIRED_COLUMNS["Diann"]`,
  `OPTIONAL_COLUMNS["Diann"]`, and `QUANT_COLUMN_CANDIDATES["Diann"]`.
- **`alphaphos.constants` extended** with `DIANN_*` and
  `UNIMOD_PHOSPHO` constants so the DIA-NN column names live in the
  same single-source-of-truth as Spectronaut's.

### Fixed

- **`pyarrow` added to runtime deps**. It was silently pulled in by other
  packages in the dev env, so `import alphaphos` worked locally. On a
  fresh CI install it broke every test with `ModuleNotFoundError: No module
  named 'pyarrow'` because ``io/spectronaut.py`` now imports
  ``pyarrow.parquet`` at module top.

## [0.1.1] - 2026-07-01

Internal refactor pass to align with the coding principles in
`CLAUDE.coding.md`. No public behavior change.

### Added

- **`alphaphos.constants`** module — single source of truth for column
  names, layer keys, obs / var / uns keys, and internal working-column
  identifiers (``COL_*``, ``PTM_*``, ``VAR_*``, ``OBS_*``, ``LAYER_*``,
  ``UNS_*``). All internal modules now import from here rather than
  hardcoding strings.

### Changed

- **In-function imports lifted to module top** across
  ``io/spectronaut.py``, ``preprocess/collapse.py``, and
  ``preprocess/_collapse/site_pipeline.py``. No more deferred imports
  except in the module-level try/except guards for optional dependencies.
- **Docstrings scoped tighter** — removed caller / downstream references
  from function docstrings (e.g. "for the QC dashboard's PSM lineage
  panel"). Docstrings now describe what the function itself does and
  what it returns.
- **`io/schemas.py`** now references named column constants from
  ``alphaphos.constants`` rather than hard-coding string literals in its
  lookup tables.

### Fixed

- **`test_uns_provenance`** hardcoded the version string ``"0.0.0"``,
  which broke CI on the 0.1.0 bump. Now reads from ``ap.__version__``
  so future bumps don't re-break the test.

## [0.1.0] - 2026-07-01

The first tagged pre-release. Establishes the public API shape that
`collapse_sites` returns an `AnnData` and that all algorithmic knobs live in
`DEFAULT_*_SETTINGS` dicts overridable via an `advanced=` kwarg.

### Added

- **Functional collapse pipeline** (`alphaphos.preprocess.collapse.collapse_sites`)
  that composes 7 submodules under `preprocess/_collapse/`:
  `parsing`, `keys`, `aggregation`, `selectivity`, `site_pipeline`, `masking`,
  `output_format`. Each stage is a pure function with a proper docstring
  explaining what it does and why.
- **`DEFAULT_COLLAPSE_SETTINGS`** with `search_engine`, `quantification_level`,
  `top_n_attribution`, `cutoff`, `classI_cutoff`, `condition_threshold`,
  `collapse_level`, `aggregation_method`, `localization_strategy`,
  `noise_floor_filter`, `drop_all_nan`. Unknown keys / bad values raise
  early with helpful error messages.
- **`Protein|Gene|Site|Mult` site identifiers** (`|` delimiter throughout,
  never conflicting with gene / accession content). Short human-readable
  label (`Gene|Site|Mult`) available in `adata.var["short_key"]` with
  deterministic `#N` suffixes on collisions.
- **Per-sample phospho selectivity** computed at collapse time from raw
  PSMs and attached as `adata.obs["phospho_selectivity_pct"]`.
- **Quantification level with fallback chain**: `MS2 -> MS1 -> auto`.
  Default is `"MS2"`; if MS2 columns aren't present the reader falls back
  to MS1, then to Spectronaut's `(Settings)` column, warning the user and
  stamping the actual level used in `adata.uns["alphaphos"]["stats"]`.
- **`alphaphos.io.spectronaut.read_spectronaut`** with `DEFAULT_IO_SETTINGS`
  and `advanced=` override dict. **Column pruning at read time**: a Spectronaut
  Normal report with 30-50+ columns gets pruned to ~15 loaded, saving
  2-3x RAM on large files. `df.attrs["columns_dropped"]` reports the count.
- **`alphaphos.io.schemas`** as the single source of truth for engine-specific
  column names (`REQUIRED_COLUMNS`, `OPTIONAL_COLUMNS`, `QUANT_COLUMN_CANDIDATES`,
  `FALLBACK_CHAINS`). Adding a new engine (DIA-NN, FragPipe, PEAKS) is a
  one-file change here.
- **FASTA / kinase-window annotation** as a standalone step
  (`alphaphos.kinase.annotation.add_kinase_windows`), operating on the AnnData
  after collapse. Decoupled so non-human workflows never need to touch a FASTA.
- **QC dashboard** (`alphaphos.qc.generate_dashboard`): one-call HTML QC
  report using bokeh, with 6 sections (pipeline waterfall, sample-level QC,
  reproducibility, Class-I + contaminant, imputation diagnostics, provenance).
- **CurveCurator wrapper** for dose-response
  (`alphaphos.dose_response.fit_dose_response`).
- **Kinase activity**: Yaffe library PWM scoring (`alphaphos.kinase.library`)
  and OmniPath / decoupler-based KSEA (`alphaphos.ksea`).
- **Hybrid imputation** (`alphaphos.preprocess.impute_hybrid`): per-cell
  MAR (site-KNN) + MNAR (downshifted Gaussian) split.

### Changed

- **`collapse_sites` return type**: previously `(sites_df, loc_per_run_df)`.
  Now returns a single `AnnData` with `.X`, `layers["localization"]`, and
  full `.var` / `.obs` / `.uns` metadata populated.
- **Site key delimiter**: was `PG~Gene_S123_M1` (two delimiters), now
  `Protein|Gene|S123|M1` (single delimiter).
- **Reader responsibility narrowed**: `read_spectronaut` no longer applies
  `top_n_attribution` and no longer picks a quant column. Both moved to
  collapse where they can be tuned via `advanced=` and where a user can
  inspect the raw PSM output first.
- **CI adds `scikit-learn` to runtime deps** — was previously used by
  `impute.py` without being declared, so fresh installs failed on first
  `impute_hybrid` call.

### Removed

- **`PeptideCollapse` class** — replaced by the functional pipeline.
  Users who called `pc = PeptideCollapse(); pc.process_complete_pipeline(...)`
  migrate to `adata = collapse_sites(...)`.
- **Old flat kwargs on `read_spectronaut`**: `quant_level`,
  `top_n_attribution`, `drop_decoys`, `drop_contaminants`,
  `contaminants_fasta`, `contaminant_prefixes`, `eg_qvalue_max`,
  `pg_qvalue_max`. All migrated to the `advanced=` dict.
- **`fasta_path`, `add_kinase_sequences`, `kinase_window_size` from
  `collapse_sites`** — moved out to `add_kinase_windows` as an explicit
  downstream step.

### Fixed

- **QC dashboard rendered blank** due to `p.scatter(x=[None], y=[None])`
  triggering a null-length crash in bokeh's `FactorRange.v_synthetic`
  transform. Replaced dummy legend glyphs with proper
  `Legend`/`LegendItem` layout attached explicitly.

### Known limitations

- Only Spectronaut (`search_engine="SN"`) is fully implemented. Other
  engines (`"Diann"`, `"Fragpipe"`, `"Peaks"`) raise `NotImplementedError`
  by design — the settings key is reserved.
- No streaming reads yet; very large reports (>10 M rows) load in full.
