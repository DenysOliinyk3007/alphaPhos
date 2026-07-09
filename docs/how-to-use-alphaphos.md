# How to use alphaPhos

Landing page for new users.  Read this first: it inventories the public
API and walks a decision tree from raw MS output to publication-ready
tables and figures.

Version-tied to **alphaPhos 0.19.0** (September 2026); check
`ap.__version__` if you're on a newer release.  Per-module deep dives
live under [`docs/modules/`](modules/).

---

## 1. Module inventory (all public entry points)

Every name below is accessible as `ap.<name>` (top level) or
`ap.<namespace>.<name>` (sub-package).  ~65 public names total.

| Layer | Namespace | What's in it |
| --- | --- | --- |
| **IO -- phospho** | `ap.` top | `read_spectronaut`, `read_diann`, `read_fragpipe_sites` |
| **IO -- proteome** | `ap.proteome` | `read_spectronaut_short`, `read_spectronaut_long`, `collapse_proteome`, `phospho_over_proteome` |
| **PSM &rarr; AnnData** | `ap.` top | `collapse_sites` (site-level), `collapse_precursors` + `precursor_to_site_view` (precursor-level) |
| **Preprocessing** | `ap.` top | `filter_by_completeness`, `impute_hybrid`, `impute_knn_site_based`, `batch_correct_combat`, `add_kinase_windows`, `load_fasta` |
| **Statistics** | `ap.stats.*` | `diff_exp_limma` (2-group), `diff_exp_limma_contrasts` (multi-contrast), `diff_exp_anova` (multi-group F), `anova_hits`, `design_matrix`, `DesignMatrix` |
| **Kinase scoring** | `ap.kinase.library` | `score_kinases` (Yaffe / Johnson 2023 PSSMs via the `kinase_library` package) |
| **KSEA (kinase activity)** | `ap.enrichment.*` | `kinase_activity` (decoupler ULM/MLM), `fetch_omnipath_ks_network`, `load_ptm_ks_network` |
| **Site enrichment** | `ap.enrichment.*` | `ora`, `gsea`, `canonicalise_site_ids`, `emit_libraries`, `load_libraries`, `load_gmt`, `match_sites`, `attach_site_ids`, `parse_alphaphos_key` |
| **Gene enrichment** | `ap.enrichment.*` | `pathway_enrichment` (Enrichr ORA), `pathway_gsea` (preranked GSEA) |
| **Dimensionality reduction** | `ap.dimred.*` | `pca` (standard / NIPALS / PPCA), `compare_imputation_impact`, `sample_distance`, `hierarchical_cluster`, `loadings_for_enrichment`, `feature_variance_contribution`, `get_pca_dataframe`, `get_pca_loadings` |
| **Signalome (module + kinase network)** | `ap.signalome.*` | `build_signalome`, `SignalomeResult`, per-stage helpers (`cluster_sites`, `derive_protein_modules`, `build_module_assignments`, `build_module_table`, `build_kinase_network`, `build_expanded_table`, `extract_site_metadata`, `extract_site_to_protein`) |
| **Cross-species** | `ap.orthology` | `map_to_human` (validated on mouse SwissProt) |
| **QC** | `ap.` top | `generate_dashboard` (Bokeh HTML report; optional `bokeh` dep) |
| **PhosPy-EV validation harness** | `ap.enrichment.*` | `load_ev2`, `load_ev3`, `ev3_expectations_for_condition`, `score_against_ev3`, `build_kinase_substrate_library` |

Everything in this table is stable public API.  Anything else (dunder-
prefixed, `_orchestrator`, etc.) is internal and can change without
notice.

---

## 2. Decision tree -- "which function do I call?"

Read the questions top-to-bottom.  Each level narrows the choice.

### Level 1 -- What input data do you have?

```
Phospho DIA (Spectronaut) ────────────────▶ ap.read_spectronaut
Phospho DIA (DIA-NN)     ────────────────▶ ap.read_diann
Phospho DDA (FragPipe)   ────────────────▶ ap.read_fragpipe_sites
Proteome (Spectronaut short / wide)      ▶ ap.proteome.read_spectronaut_short
Proteome (Spectronaut long / precursor)  ▶ ap.proteome.read_spectronaut_long
Already have an AnnData                  ▶ jump to Level 3
```

### Level 2 -- How do you want the AnnData shaped?

```
Phospho, sites are the unit of interest        ─▶ ap.collapse_sites
Phospho, precursors are the unit (low local.)  ─▶ ap.collapse_precursors
                                                    (+ ap.precursor_to_site_view when going into KSEA)
Proteome, short (already-collapsed) report     ─▶ nothing extra: the reader returns AnnData
Proteome, long (precursor) report              ─▶ ap.proteome.collapse_proteome
Paired phospho / proteome normalisation        ─▶ ap.proteome.phospho_over_proteome
                                                    (log2 subtraction; removes protein-abundance
                                                     confounding from phospho fold-changes)
```

### Level 3 -- Standard preprocessing (almost always in this order)

```
1. Filter by completeness   ap.filter_by_completeness(
                              min_valid_frac=2/3,
                              group_column="condition",
                              keep_strategy="each"|"any"|"global")
2. Impute (if needed)       ap.impute_hybrid   (default; per-cell MAR-KNN + MNAR-Gaussian)
                            ap.impute_knn_site_based (legacy parity)
3. Batch correct (optional) ap.batch_correct_combat   OR   pass batch= as covariate to diff-exp
4. QC / exploratory         ap.dimred.pca + get_pca_dataframe / sample_distance
                            ap.dimred.compare_imputation_impact   (does imputation distort clusters?)
                            ap.generate_dashboard("qc.html")      (Bokeh HTML report)
```

### Level 4 -- What scientific question are you asking?

```
"How do samples cluster?"
    ├── ap.dimred.pca(..., method="standard"|"nipals"|"ppca")
    ├── ap.dimred.get_pca_dataframe   (samples with PC coords)
    ├── ap.dimred.sample_distance     (distance matrix)
    ├── ap.dimred.hierarchical_cluster
    └── ap.dimred.compare_imputation_impact  (imputation vs raw)

"Which SITES differ between conditions?"
    ├── Two groups          ─▶ ap.diff_exp_limma(condition_column, comparison=(a, b))
    ├── Multiple groups     ─▶ ap.diff_exp_anova(condition_column)
    │                            (moderated F-test; any-group-differs)
    └── Named contrasts     ─▶ ap.diff_exp_limma_contrasts(
                                  condition_column, contrasts={"name": (trt, ctrl), ...})

"Which KINASES are active?"
    └── ap.enrichment.kinase_activity(
              diff_exp_result, network="omnipath"|"ptm_db"|DataFrame,
              method="ulm"|"mlm")
        (needs SIGNED input -- per-contrast log2fc, NOT ANOVA output)

"Are my hits enriched for pathways / functional sets?"
    ├── Site-set (PTM-DB)  ORA           ─▶ ap.enrichment.ora(hits, background, libraries)
    ├── Site-set preranked GSEA          ─▶ ap.enrichment.gsea(ranked_series, libraries)
    ├── Gene-level Enrichr ORA           ─▶ ap.enrichment.pathway_enrichment(
    │                                          diff_exp_result,
    │                                          direction="split"|"up"|"down"|"both"|"any")
    │                                       • direction="any" is the ANOVA-friendly path
    │                                         (unsigned, uses fdr only)
    └── Gene-level preranked GSEA        ─▶ ap.enrichment.pathway_gsea(diff_exp_result)

"What phospho MODULES / kinase network exist?"
    └── ap.build_signalome(prediction_matrix, kinase_substrates=..., ...)
        (input: adata.varm["kinase_score_ser_thr"] from ap.kinase.library.score_kinases)
        Returns SignalomeResult with:
          .site_assignments   per-site: module_id, top_kinase, ties
          .protein_modules    protein -> module_id
          .module_table       (modules x kinases) percent shares
          .network            .edges, .nodes, .candidate_correlations
          .expanded           denormalised per-focal-kinase view
          .clustering         Ward tree + module-count selection diagnostics

"Map mouse sites to human orthologs?"
    └── ap.orthology.map_to_human(adata, target_fasta="human.fasta")

"Dose-response curve fitting?"
    └── ap.dose_response.fit_dose_response  (optional `curve_curator` dep)
```

### Level 5 -- Bridge helpers (often overlooked)

These sit between the main functions and unblock common patterns.

| Bridge | What it does |
| --- | --- |
| `ap.anova_hits(anova_df)` | Returns `(hits, background)` from a `diff_exp_anova` result -- feed straight into `ap.enrichment.ora`. |
| `ap.enrichment.canonicalise_site_ids(keys)` | alphaPhos `Protein\|Gene\|Site\|Mult` &rarr; PTM-DB `Protein_AApos` IDs.  Needed when calling `ap.enrichment.ora` with PTM-DB GMTs. |
| `ap.enrichment.attach_site_ids(diff_exp_result)` | Same conversion, but attaches the canonical IDs as a new column of the diff-exp table. |
| `ap.dimred.loadings_for_enrichment(pca_result)` | Turns PC loadings into a signed ranking that plugs straight into `ap.enrichment.gsea` / `pathway_gsea` / `kinase_activity`. |
| `ap.signalome.extract_site_metadata(var)` | Builds the `site_metadata` DataFrame `ap.build_signalome` requires from an alphaPhos-conventioned `adata.var`. |
| `ap.signalome.select_kinase_substrates(matrix, cutoff)` | Threshold a per-site kinase-score matrix to synthesise `{kinase: [substrate_sites]}`. |

---

## 3. Canonical end-to-end workflows

Four common study designs.  Each block runs top-to-bottom.

### Workflow A -- 2-group phospho study (the standard case)

```python
import alphaphos as ap

# 1. IO + collapse
psm = ap.read_spectronaut("report.tsv")
adata = ap.collapse_sites(psm, condition_df=cond_df)

# 2. Preprocess
adata = ap.filter_by_completeness(adata, min_valid_frac=2/3,
                                    group_column="condition", keep_strategy="each")
adata = ap.impute_hybrid(adata)

# 3. Two-group diff-exp
result = ap.diff_exp_limma(adata, condition_column="condition",
                            comparison=("trt", "ctrl"))

# 4. Enrichment (all three flavours)
hits = result.index[result["fdr"] < 0.05]
bg   = result.index
site_ora = ap.enrichment.ora(
    ap.enrichment.canonicalise_site_ids(hits),
    ap.enrichment.canonicalise_site_ids(bg),
    libraries=PTM_LIBS,
)
kin_ksea  = ap.enrichment.kinase_activity(result, network="omnipath")
gene_ora  = ap.enrichment.pathway_enrichment(result, direction="split")
gene_gsea = ap.enrichment.pathway_gsea(result)
```

### Workflow B -- Multi-group study (ANOVA + per-contrast follow-up)

```python
# Steps 1-2 as Workflow A, but with >2 disease groups in condition_column.

# 3. Moderated F-test across all levels
anova = ap.diff_exp_anova(adata, condition_column="disease")

# 4a. Gene enrichment on ANOVA hits (direction-agnostic)
gene_ora = ap.enrichment.pathway_enrichment(anova, direction="any")

# 4b. Site-level ORA on ANOVA hits
hits, bg = ap.anova_hits(anova)
site_ora = ap.enrichment.ora(
    ap.enrichment.canonicalise_site_ids(hits),
    ap.enrichment.canonicalise_site_ids(bg),
    libraries=PTM_LIBS,
)

# 4c. Per-contrast follow-up for SIGNED downstream (KSEA / preranked GSEA)
per_contrast = ap.diff_exp_limma_contrasts(
    adata, condition_column="disease",
    contrasts={"HCM_vs_H": ("HCM", "Healthy"),
                "ICM_vs_H": ("ICM", "Healthy"),
                "NICM_vs_H": ("NICM", "Healthy")},
)
for name, df in per_contrast.items():
    ksea = ap.enrichment.kinase_activity(df, network="omnipath")
```

### Workflow C -- Signalome (module + kinase network view)

```python
# After Workflow A / B steps 1-2 (adata is filtered + imputed):
adata = ap.add_kinase_windows(adata, fasta_path="resources/fastas/human.fasta")

from alphaphos.kinase.library import score_kinases
score_kinases(adata)   # populates adata.varm["kinase_score_ser_thr"]

# Signalome on the S/T branch (the larger one for most human studies)
mask = adata.varm["kinase_score_ser_thr"].notna().any(axis=1)
result = ap.build_signalome(
    adata.varm["kinase_score_ser_thr"].loc[mask],
    scoring_mode="auto",      # exact for n<=5000, sampled otherwise
    network_correlation_threshold=0.5,
)

result.module_table    # module x kinase %-share
result.network.edges   # kinase-kinase co-regulation edges
result.expanded        # denormalised per-focal-kinase view (JSON list cols)
```

### Workflow D -- Paired phospho + proteome (remove abundance confounding)

```python
# Both branches:  read -> collapse -> filter -> impute
adata_phos = ...  # phospho pipeline
adata_prot = ap.proteome.read_spectronaut_short("proteome.tsv", condition_df=cond)
# ... standard preprocess on adata_prot ...

# Pair samples + normalise phospho by matched-sample parent-protein abundance
adata_norm = ap.proteome.phospho_over_proteome(
    adata_phos, adata_prot,
    sample_pairing="auto",   # by DVP well-ID regex, or dict, or Series
    missing_protein="drop",   # or "carry" / "fail"
)

# Now diff-exp / KSEA / enrichment all three layers with identical calls
# (phos / prot / norm).  Compare which biology survives normalisation.
```

---

## 4. Things you probably don't need day-to-day

- **`ap.collapse_precursors` + `precursor_to_site_view`** -- only when
  localisation is unreliable and you want to stay at precursor
  granularity.  Bridges back to site keys for KSEA.
- **`ap.dimred.compare_imputation_impact`** -- one-off sanity check per
  study, not a routine step.
- **`ap.enrichment.emit_libraries`** -- refreshing the PTM-DB GMTs from
  the raw parquet.  Ship-time operation.
- **`ap.enrichment.match_sites`** / **`attach_site_ids`** -- diagnostic
  variants of `canonicalise_site_ids`; prefer the latter in normal flows.
- **`ap.enrichment.load_ev2` / `load_ev3` / `score_against_ev3`** --
  ev2 / ev3 validation harness (matches a PhosPy-derived benchmark).
- **`ap.dose_response.fit_dose_response`** -- CurveCurator wrapper for
  compound / time-course dose-response studies.

---

## 5. Where else to look

- **`README.md`** -- top-level table of contents, install instructions,
  quickstart.
- **`docs/modules/`** -- per-subpackage deep dives.  Signalome
  in particular has its own biology-level explainer at
  [`docs/modules/signalome/index.md`](modules/signalome/index.md).
- **`examples/`** -- runnable scripts + Jupyter notebooks against the
  shipped `test_data/`, including:
  - `egf_walkthrough.{ipynb,py}` -- 2-group EGF benchmark (Workflow A).
  - `cardio_anova.py`, `cardio_anova_walkthrough.ipynb` -- 5-group
    cardiomyopathy ANOVA (Workflow B).
  - `proteome_walkthrough.ipynb` -- proteome + paired phospho / proteome
    (Workflow D).
  - `cardio_signalome.py` -- signalome on real cardio data (Workflow C).
- **`CHANGELOG.md`** -- release notes with what changed in each version.
