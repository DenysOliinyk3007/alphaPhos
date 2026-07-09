# alphaPhos

Phosphoproteomics analysis toolkit.

**Status:** pre-alpha (v0.20.0). Standard 2-condition workflows (Spectronaut / DIA-NN / FragPipe → sites → filter → impute → optional ComBat → limma → PCA / KSEA / enrichment) are covered end-to-end and validated on real data. Multi-contrast / multi-group designs (moderated F-test, paired-block, arbitrary contrasts) ship via `ap.diff_exp_anova` + `ap.diff_exp_limma_contrasts` on a clean-room Smyth 2004 stack (bit-exact vs inmoose on the 2-group case). Cross-species orthology (mouse ↔ human) is validated on full SwissProt. PCA / t-SNE / UMAP all ship in `ap.dimred` with a shared API.

## Install

```bash
pip install -e .                # core (Python-only)
pip install -e ".[stats]"       # + inmoose for limma differential testing + ComBat batch correction
pip install -e ".[dimred]"      # + umap-learn for ap.dimred.umap (t-SNE ships with sklearn)
pip install -e ".[dev,docs]"    # development
```

`inmoose` (Python limma + ComBat) is behind the `[stats]` extra; without it, `ap.diff_exp_limma` and `ap.batch_correct_combat` raise `ImportError` and everything else works.

## Quickstart

```python
import pandas as pd
import alphaphos as ap

# 1. Read PSMs from your search engine
psm_df = ap.read_spectronaut("report.tsv")     # or ap.read_diann(...) / ap.read_fragpipe_sites(...)

# 2. Sample metadata (must contain 'sample' and 'condition' columns)
condition_df = pd.DataFrame({
    "sample":    ["s1", "s2", "s3", "s4"],
    "condition": ["ctrl", "ctrl", "trt", "trt"],
})

# 3. Collapse PSMs -> site-level AnnData
adata = ap.collapse_sites(psm_df, condition_df=condition_df)

# 4. Keep sites observed in >= 2/3 of each condition
adata = ap.filter_by_completeness(
    adata, min_valid_frac=2/3,
    group_column="condition", keep_strategy="each",
)

# 5. Hybrid MAR/MNAR imputation (in-place; also returns adata so either
#    `ap.impute_hybrid(adata)` or `adata = ap.impute_hybrid(adata)` works)
adata = ap.impute_hybrid(adata)

# 6. (Optional) ComBat batch correction if the design has a known batch column
# adata = ap.batch_correct_combat(adata, batch_column="batch", covariates=["condition"])

# 7. Two-group moderated t-test via limma
result = ap.diff_exp_limma(
    adata,
    condition_column="condition",
    comparison=("trt", "ctrl"),  # log2fc = mean(trt) - mean(ctrl)
)

# result: per-site DataFrame indexed by adata.var_names with columns
#   log2fc, se, t_stat, p_value, fdr, B, ave_expr
```

## What ships in v0.20.0

| Module | Function | Notes |
| --- | --- | --- |
| `alphaphos.io.spectronaut` | `read_spectronaut` | Column pruning at read time. Handles decoys + contaminants. |
| `alphaphos.io.diann` | `read_diann` | Manuscript-validated on nanoPhos. Requires DIA-NN ≥1.9. |
| `alphaphos.io.fragpipe` | `read_fragpipe_sites` | Reads FragPipe's pre-collapsed abundance files. |
| `alphaphos.proteome` | `read_spectronaut_short`, `read_spectronaut_long`, `collapse_proteome`, `phospho_over_proteome` | Proteome (non-phospho) DIA analysis with Spectronaut short + long report formats.  Same AnnData shape as phospho — filter / impute / batch-correct / limma / PCA all work unchanged.  `phospho_over_proteome` divides phospho intensities by parent-protein intensities to yield log-fraction-phosphorylated (removes protein-abundance confounding). |
| `alphaphos.preprocess.collapse` | `collapse_sites` | PSM → site AnnData with `Protein\|Gene\|Site\|Mult` keys, top-N attribution, three localization strategies. |
| `alphaphos.preprocess.collapse_precursors` | `collapse_precursors`, `precursor_to_site_view` | PSM → precursor AnnData (`Protein\|Gene\|Peptide\|Charge\|Mods`). No residue attribution, no localization masking — use for detection / differential when localization is unreliable on low-abundance features. Bridges back to site keys via `precursor_to_site_view` for KSEA. |
| `alphaphos.preprocess.filter` | `filter_by_completeness` | Global / any-group / each-group strategies. |
| `alphaphos.preprocess.impute` | `impute_hybrid`, `impute_knn_site_based` | Per-cell MAR-KNN + MNAR-Gaussian hybrid, or legacy-parity KNN. |
| `alphaphos.preprocess.batch_correct` | `batch_correct_combat` | inmoose pycombat wrapper with double-correct guard. |
| `alphaphos.stats.diff_exp` | `diff_exp_limma`, `diff_exp_limma_contrasts`, `diff_exp_anova`, `anova_hits` | Two-group moderated t-test (`diff_exp_limma`, inmoose backend), multi-contrast moderated t (`diff_exp_limma_contrasts`, joint fit — one linear model, N contrasts), and moderated F-test across all condition levels (`diff_exp_anova`, ANOVA-style). Batch / covariate / paired-block adjustment supported. `anova_hits(anova, fdr_threshold=0.05)` is a two-line convenience for `(hits, background)` ready for `ap.enrichment.ora`. |
| `alphaphos.stats.design` | `design_matrix`, `DesignMatrix` | Typed design-matrix builder for the multi-contrast + ANOVA path. Categorical + continuous + batch covariates + paired-block factor, no-intercept parameterisation. |
| `alphaphos.stats.moderated` + `linear_model` | (internal) | Clean-room Smyth 2004 empirical-Bayes stack (`fit_f_dist`, `moderate_variance`, `lm_fit`, `contrasts_fit`, `moderated_t_test`, `moderated_f_test`). MIT-licensed, numerically bit-exact vs inmoose on 2-group and vs PhosPy on the EB-prior fit to ~1e-13. |
| `alphaphos.dimred` | `pca`, `tsne`, `umap`, `get_pca_dataframe`, `get_tsne_dataframe`, `get_umap_dataframe`, `compare_imputation_impact`, `loadings_for_enrichment`, `feature_variance_contribution`, `sample_distance`, `hierarchical_cluster` | **PCA** (standard / NIPALS / PPCA backends behind one entrypoint, NaN-aware). **t-SNE** via `sklearn.manifold.TSNE` (Barnes-Hut) and **UMAP** via `umap-learn` (via optional `[dimred]` extra) -- both share the same API: layer / `n_pca_components` / `seed` / `copy`, embedding to `.obsm["X_{pca,tsne,umap}"]`, provenance to `.uns[...]`. NaN handling: impute upstream or pre-reduce with a NaN-aware PCA and pass `n_pca_components=`. `compare_imputation_impact` flags whether imputation distorts sample-space structure. `loadings_for_enrichment` bridges PC loadings straight into `alphaphos.enrichment.gsea` / `.ora` / `.kinase_activity` with no wrappers. |
| `alphaphos.kinase.annotation` | `add_kinase_windows`, `load_fasta` | ±7-residue windows around each site from a proteome FASTA. |
| `alphaphos.kinase.library` | Yaffe PWM scoring | Requires the optional `kinase_library` package. |
| `alphaphos.kinase.enrichment` | Kinase library enrichment | Same optional dep. |
| `alphaphos.enrichment.ksea` | `kinase_activity` | Kinase-activity inference via decoupler ULM (default) / MLM against OmniPath (default), curated PTM DB, or a user-supplied network. |
| `alphaphos.enrichment` (site-set) | `emit_libraries`, `load_libraries`, `ora`, `gsea` | PTM functional DB (8 curated sources, ~504k relations) → GMT libraries → Fisher ORA + Subramanian preranked GSEA. Site-level counterpart to `pathway_enrichment` (which is gene-level). |
| `alphaphos.enrichment.pathway` | `pathway_enrichment` | Gene-level pathway ORA via gseapy Enrichr (GO BP/MF/CC + KEGG + Reactome + Hallmark by default). Phosphoproteome background by default; proteome background preferred if available.  Signed input (`direction="split"|"up"|"down"|"both"`) uses `log2fc`; **for ANOVA output use `direction="any"`** (direction-agnostic, needs only `fdr`). |
| `alphaphos.enrichment.pathway_gsea` | `pathway_gsea` | Gene-level preranked GSEA (Subramanian 2005 via gseapy prerank) on the same Enrichr libraries. Complements ORA for coherent-motion pathways. Site→gene collapse via max-\|log2fc\| default or lowest-per-site-FDR. |
| `alphaphos.orthology` | `map_to_human` | Cross-species phospho-site translation via ±7 flanking-window matching against the target proteome, with target-decoy FDR (Elias-Gygi 2007), optional ±30 broader-window verification, and paralog disambiguation. Validated on full mouse SwissProt (60.2% mapping rate; species-entrapment FDR ≤ 5×10⁻⁴). |
| `alphaphos.signalome` | `build_signalome`, `SignalomeResult`, per-stage helpers (`cluster_sites`, `derive_protein_modules`, `build_module_assignments`, `build_module_table`, `build_kinase_network`, `build_expanded_table`) | Module detection + kinase-network extraction on kinase-prediction matrices (e.g. Yaffe PSSM scores from `score_kinases`). Ward hierarchical clustering + auto module-count selection (scale-aware `scoring_mode`), protein-level module resolution via cluster-signature grouping, module × kinase % share table, kinase-kinase correlation network (signed / positive-only / absolute policies), denormalised expanded view. Clean-room MIT re-implementation of the [PhosR](https://github.com/PYangLab/PhosR) (Kim et al. 2021 *Cell Rep Meth*) / [PhosPy](https://github.com/falconsmilie/phospy) signalome algorithm — bit-exact parity validated stage-by-stage. See [docs/modules/signalome/](docs/modules/signalome/index.md). |
| `alphaphos.dose_response` | `fit_dose_response` | CurveCurator wrapper (optional `curve_curator`). |
| `alphaphos.qc` | `generate_dashboard` | Bokeh QC HTML report (optional `bokeh`). |

## Known limitations to flag before use

- **FragPipe multi-site mode has a silent bug.** Default `site_type="single"` works; `"multi"` can emit garbled protein IDs.
- **`.var` schema carries 9 legacy/internal columns** (`PTM_group`, `PTM_0_pos_val`, `UPD_seq`, `pg_key`, ...) with no downstream consumers. Slated for cleanup in a future minor.
- **`peptide_start = 0` silently emits `S0`** rather than rejecting. Locked by a characterization test; will surface if a future validator lands.
- **`EG.PTMAssayProbability` is not used as a filter anywhere.** Peptide-level assay confidence has to be gated upstream in Spectronaut.

## Bundled proteome FASTAs

The `resources/fastas/` directory ships **human**, **mouse**, and
**Chinese hamster (CHO)** proteomes (~37 MB total, needed by
`ap.add_kinase_windows` for sequence-window annotation). Sourced from
[UniProt](https://www.uniprot.org/) — CC-BY 4.0. Please cite UniProt if
you use these downstream:

> The UniProt Consortium. *UniProt: the Universal Protein
> Knowledgebase in 2023.* Nucleic Acids Res. 51:D523-D531 (2023).

The versions here are pinned snapshots; refresh from UniProt when you
need a newer release.

## License

MIT — see [LICENSE](LICENSE).
