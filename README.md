# alphaPhos

Phosphoproteomics analysis toolkit.

**Status:** pre-alpha (v0.6.1). Structure scaffolded; modules being extracted from prior scripts.

## Scope

A Python library for reproducible phosphoproteomics analysis

## Install

```bash
pip install -e .                # core (Python-only)
pip install -e ".[stats]"       # + inmoose for the limma two-group test
pip install -e ".[r]"           # + R backend for PTM-SEA
pip install -e ".[dev,docs]"    # development
```

`inmoose` (Python limma) is behind the `[stats]` extra; without it, `ap.diff_exp_limma` raises `ImportError` and everything else works. `rpy2` remains optional for R-backed enrichment (`alphaphos.enrichment.ptmsea`).

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
