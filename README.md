# alphaPhos

Phosphoproteomics analysis toolkit.

**Status:** pre-alpha (v0.3.0). Structure scaffolded; modules being extracted from prior scripts.

## Scope

A Python library for reproducible phosphoproteomics analysis: parse search-engine outputs (MaxQuant, Spectronaut, DIA-NN, AlphaPept), collapse peptides to sites, normalize and filter, run differential statistics, KSEA, functional enrichment, and produce standard visualizations. The in-memory representation is an AnnData-based `PhosphoExperiment` with peptide / site / protein layers.

## Install

```bash
pip install -e .                # core (Python-only)
pip install -e ".[r]"           # + R backend for limma and PTM-SEA
pip install -e ".[dev,docs]"    # development
```

R/rpy2 is **optional**. Without it, `alphaphos.stats.limma` and `alphaphos.enrichment.ptmsea` are unavailable; everything else works.

## Layout

| Path | Purpose |
| --- | --- |
| `src/alphaphos/io/` | Search-engine output readers |
| `src/alphaphos/preprocess/` | Peptide collapse, normalization, filtering, imputation |
| `src/alphaphos/ptm/` | Site mapping, localization, multi-PTM handling |
| `src/alphaphos/stats/` | Differential testing (limma, ANOVA, paired) |
| `src/alphaphos/ksea/` | Kinase-Substrate Enrichment Analysis |
| `src/alphaphos/enrichment/` | GO/KEGG/Reactome ORA, PTM-SEA |
| `src/alphaphos/acquisition/` | MS acquisition queue building |
| `src/alphaphos/viz/` | Plotting |
| `src/alphaphos/qc/` | QC checks |
| `apps/` | Standalone apps: KSEAplus binary wrapper, phosphoscape Streamlit app |
| `studies/` | Per-study analyses that import alphaPhos |
| `manuscripts/` | Figure code for published methods papers |
| `tests/regression/` | Golden-snapshot regression tests against legacy pipelines |

## License

MIT — see [LICENSE](LICENSE).
