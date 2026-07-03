# Changelog

All notable changes to alphaPhos are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While in `0.x`, breaking API changes may appear in any MINOR bump (`0.1 → 0.2`).
`1.0.0` will mark the first stable public API.

## [Unreleased]

_Nothing yet._

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
