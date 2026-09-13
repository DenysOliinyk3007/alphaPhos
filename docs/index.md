# alphaPhos

Python-native phosphoproteomics analysis toolkit for DIA workflows and
large-cohort studies: empirically-derived defaults, scale-aware filtering and
imputation, and full scanpy / AnnData interoperability.

## Start here

- **[Quickstart](quickstart.md)** — 30 minutes from install to a
  differential-expression table on the bundled EGF fixture.
- **[How to use alphaPhos](how-to-use-alphaphos.md)** — the full API tour:
  module inventory, decision tree, four canonical workflow templates.
- **[Design principles](design-principles.md)** — why the defaults are what
  they are, organised by cohort-size regime.
- **[Alpha-tester guide](alpha-tester-guide.md)** — what feedback we want and
  what's known-rough.

## Module reference

Per-module deep dives live under *Modules* in the navigation:
IO readers, the collapse pipeline and preprocessing, statistics, enrichment
(KSEA / ORA / GSEA), the signalome, and cross-species orthology.

## Install

```bash
git clone https://github.com/DenysOliinyk3007/alphaPhos.git
cd alphaPhos
pip install -e ".[stats]"
```

See the [README](https://github.com/DenysOliinyk3007/alphaPhos#install) for
the optional extras table.
