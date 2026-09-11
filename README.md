# alphaPhos

**Python-native phosphoproteomics analysis toolkit for DIA workflows and large-cohort studies.** Empirically-derived defaults, scale-aware filtering and imputation, and full scanpy / AnnData interoperability.

**Status:** alpha (v0.22.0). Core science modules (IO → collapse → filter → impute → DE → enrichment → dimred → signalome) are populated and tested (916 unit tests, CI green on Windows + Linux, Python 3.10–3.12). API is stable enough for real analyses but may still change in minor bumps. Send feedback: see [`docs/alpha-tester-guide.md`](docs/alpha-tester-guide.md).

## Start here

- **[`docs/quickstart.md`](docs/quickstart.md)** — 30 minutes from install to a differential-expression table on the bundled EGF fixture.
- **[`docs/how-to-use-alphaphos.md`](docs/how-to-use-alphaphos.md)** — the full API tour: module inventory, decision tree ("which function do I call?"), four canonical workflow templates.
- **[`docs/design-principles.md`](docs/design-principles.md)** — the "why these defaults?" doc, organised by cohort-size regime (`n < 30` / `30–100` / `100–300` / `300+`). Read this if you want to know why alphaPhos chooses what it chooses on your data.
- **[`docs/alpha-tester-guide.md`](docs/alpha-tester-guide.md)** — what feedback we want, where to report, what's known-rough.

## Install

```bash
git clone https://github.com/DenysOliinyk3007/alphaPhos.git
cd alphaPhos
pip install -e ".[stats]"        # base + differential expression (inmoose, patsy)
```

Optional extras (each gated at call time — install only what you need):

| extra | pulls in | needed for |
| --- | --- | --- |
| `[stats]` | `inmoose`, `patsy` | `diff_exp_limma` and everything that calls it |
| `[enrichment]` | `decoupler`, `gseapy`, `omnipath` | KSEA, pathway ORA / GSEA |
| `[dimred]` | `umap-learn` | `ap.dimred.umap` (t-SNE works without) |
| `[qc]` | `bokeh`, `plotly`, `kaleido` | `ap.generate_dashboard`, QC v2 panels |
| `[pimms]` | `pimms-learn` (torch + fastai) | `ap.impute_pimms` deep-learning imputation |
| `[all]` | all of the above | one-shot install |
| `[dev,docs]` | `ruff`, `pytest`, `mkdocs-material`, … | contributing |

`[pimms]`, `[enrichment]`, and `[dimred]` currently require `numpy < 2.4` (upstream numba / numpy incompatibility).

## Minimal usage

```python
import alphaphos as ap
import pandas as pd

psm = ap.read_spectronaut("report.tsv")
cond_df = pd.DataFrame({"sample": samples, "condition": labels})
adata = ap.collapse_sites(psm, condition_df=cond_df)

# The advisor prints a copy-paste recipe tuned to your cohort size + design:
ap.recommend_pipeline(
    adata, goal="primary_de", data_type="phospho",
    primary_factor="condition",
)
```

The full quickstart walks through filter → completeness → differential expression on the bundled EGF fixture; see [`docs/quickstart.md`](docs/quickstart.md).

## What's in the package

- **Readers** — Spectronaut, DIA-NN, FragPipe, plus Spectronaut short + long report formats for proteome
- **Collapse** — PSM → site- or precursor-level AnnData with four localization-masking strategies (`per_run` / `global_max` / `condition` / `wilson`)
- **Filtering** — completeness (`filter_by_completeness`), Wilson lower-bound Class-I (`apply_wilson_filter`, `wilson_threshold_sensitivity`), plus per-site QC columns auto-populated in `.var`
- **Imputation** — KNN, MAR/MNAR hybrid, PIMMS-DAE / VAE (Webel et al. 2024)
- **Batch correction** — ComBat via `inmoose` (double-correct-guarded)
- **Differential expression** — clean-room Smyth 2004 empirical-Bayes stack (`diff_exp_limma`, `_contrasts`, `_anova`, `_observed_only`) with on/off detection
- **Enrichment** — KSEA (decoupler ULM / MLM), site-set ORA + GSEA, gene-level pathway ORA + GSEA
- **Kinase** — sequence-window annotation, Yaffe PSSM scoring
- **Signalome** — module detection + kinase network (clean-room re-implementation of Kim et al. 2021)
- **Dimensionality reduction** — PCA (standard / NIPALS / PPCA), t-SNE, UMAP; same shared API
- **Cross-species** — mouse ↔ human orthology (`map_to_human`, validated on full SwissProt)
- **QC** — Bokeh dashboard + composable Plotly panels
- **Advisor** — `recommend_pipeline` inspects your AnnData and prints a scale-aware recipe

Full module inventory + per-function reference: [`docs/how-to-use-alphaphos.md`](docs/how-to-use-alphaphos.md) §1.

## Bundled resources

`resources/fastas/` ships human, mouse, and Chinese hamster (CHO) proteomes (~37 MB total, needed by `ap.add_kinase_windows` for sequence-window annotation). Sourced from [UniProt](https://www.uniprot.org/) (CC-BY 4.0) — please cite UniProt if you use these downstream:

> The UniProt Consortium. *UniProt: the Universal Protein Knowledgebase in 2023.* Nucleic Acids Res. 51:D523-D531 (2023).

## License

MIT — see [LICENSE](LICENSE).

## Feedback

Alpha — see [`docs/alpha-tester-guide.md`](docs/alpha-tester-guide.md).
