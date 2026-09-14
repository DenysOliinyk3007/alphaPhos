# `alphaphos.kinase`

Sequence-based kinase analysis: attach the ±7 residue window to every collapsed site,
score the windows against the Yaffe Kinase Library position-weight matrices (Johnson et al.
*Nature* 2023; Yaromenko et al. 2024 for the tyrosine kinome), and run kinase-set
enrichment on a differential result. Everything here works from **sequence alone**, so it
covers sites with no database evidence — the complement of the network-based
[`enrichment.kinase_activity`](enrichment/ksea.md) (OmniPath / PTM-DB substrates).

| Function | Question | Needs |
| --- | --- | --- |
| [`add_kinase_windows`](#add_kinase_windows) | What is the ±7 sequence context of each site? | a proteome FASTA |
| [`predict_kinases`](#score_kinases-predict_kinases) / `score_kinases` | Which kinases best match each site's motif? | `kinase-library` |
| [`kinase_mea`](#kinase-enrichment-ksea) | Which kinases' motif-substrates move coherently in the ranked result? (GSEA-style) | `kinase-library` |
| [`kinase_enrichment_from_diffexp`](#kinase-enrichment-ksea) | Which kinases' motif-substrates are over-represented among up / down hits? (Fisher) | `kinase-library` |
| [`kinase_enrichment_binary`](#kinase-enrichment-ksea) | Same, for an arbitrary foreground / background site set | `kinase-library` |

## Install

`add_kinase_windows` needs only a FASTA. Everything else wraps the
[`kinase-library`](https://github.com/TheYaffeLab/kinase_library) package, whose 1.8 release
pins `numpy~=1.26`, `pandas~=2.2`, `matplotlib~=3.8.3` — unsatisfiable next to alphaPhos on
Python 3.13 / pandas 3. The code itself works with current versions:

```bash
pip install --no-deps kinase-library
pip install biopython adjustText natsort seaborn statsmodels tqdm openpyxl matplotlib
```

Without it, the wrappers raise an `ImportError` carrying this recipe.

## `add_kinase_windows`

```python
ap.add_kinase_windows(adata, fasta_path, *, window_size=7, protein_col="protein_group_id",
                      position_col="site_position", aa_col="site_aa",
                      out_col="kinase_sequence", copy=True) -> AnnData
```

Looks each site up in the FASTA and writes `adata.var["kinase_sequence"]` in the alphaPhos
format `_{7 left}*{AA}*{7 right}_` (19 characters; `_` padding at protein termini; e.g. EGFR
Y1172 → `_ISLDNPD*Y*QQDFFPK_`). Failures are sentinel strings (`FASTA_ERROR:`,
`POSITION_ERROR:`, `SEQUENCE_MISMATCH:`) that every downstream function treats as missing.
Counts land in `adata.uns["alphaphos"]["kinase_annotation"]`.

- **Use the FASTA the search engine used.** The bundled proteomes live in the repo-level
  `resources/fastas/` (`alphaphos.resources.external("fastas")`); a species mismatch shows
  up as a wall of `FASTA_ERROR`.
- **Accession fallbacks**: `collapse_sites` keys a site under a contaminant-tagged twin when
  the group contains one (`cRAP-P00441` for `P00441;cRAP-P00441`), and engines may report
  isoforms (`P00533-2`). Neither is a FASTA key; `resolve_fasta_accession` falls back to
  the untagged / base accession and the counts are reported
  (`n_fallback_contaminant_tag`, `n_fallback_isoform`).
- Validated on the EGF HeLa series: 34,279 / 34,279 (Spectronaut) and 22,256 / 22,256
  (DIA-NN, 24 via the contaminant fallback) windows, zero position or residue mismatches.

## `score_kinases` / `predict_kinases`

```python
ap.kinase.score_kinases(adata, *, sequence_col="kinase_sequence",
                        metrics=("score", "percentile"), varm_prefix="kinase",
                        round_digits=3, quiet=True) -> None
ap.kinase.predict_kinases(adata, *, top_k=5, metric="percentile", overwrite=False,
                          quiet=True) -> pd.DataFrame
```

`score_kinases` writes one (sites × kinases) matrix per metric and kinase pool to
`adata.varm` — `kinase_score_ser_thr`, `kinase_score_tyrosine`, `kinase_percentile_ser_thr`,
`kinase_percentile_tyrosine` (311 Ser/Thr + 78 Tyr kinases in kinase-library 1.8) — with
provenance in `adata.uns["alphaphos"]["kinase_scores"]`. `predict_kinases` picks the top-k
kinases per site by one metric (`top1_kinase`, `top1_score`, …, `top_kinases`, `kin_type`).

**Which metric?** The raw log2 `score` is comparable *across sites for one kinase* but not
*across kinases for one site*: promiscuous kinases score high everywhere. The
`percentile` — the score's rank against that kinase's distribution on the reference
phosphoproteome — is Johnson 2023's per-site ranking metric and the default here. On the
EGF HeLa data:

| site | top-3 by `score` | top-3 by `percentile` |
| --- | --- | --- |
| MAPK1 T185 (the MEK site) | PRP4, BMPR1A, YANK2 | **MEK2** (100), **MEK1** (99), YANK2 (99) |
| RPS6 S235 | AKT3, AKT1, SGK1 | **AKT1** (100), PKG2 (100), MRCKA (100) |
| EGFR Y1172 | SYK, DDR1, EGFR | DDR1 (99), IRR (96), SYK (95) |

Per-site prediction is a motif statement, not a proof of regulation; the enrichment
functions below aggregate it over many sites.

## Kinase enrichment (KSEA)

All three take the differential table plus `adata.var["kinase_sequence"]` as the
`sequence_lookup`; site keys come from the table's index (`id_col=None`, alphaPhos's
convention) or from `id_col`.

```python
res = ap.diff_exp_limma_observed_only(adata, condition_column="condition",
                                      comparison=("EGF", "ctrl"))[0]
seq = adata.var["kinase_sequence"]

mea = ap.kinase.kinase_mea(res, seq, rank_col="t_stat")           # {"ser_thr": df, "tyrosine": df}
mea["ser_thr"].sort_values("NES", ascending=False).head()          # ES, NES, p-value, FDR, Leading substrates

ksea = ap.kinase.kinase_enrichment_from_diffexp(res, seq, lfc_col="log2fc", pval_col="fdr",
                                                lfc_thresh=0.585, pval_thresh=0.05)
ksea["ser_thr"][["most_sig_direction", "most_sig_log2_freq_factor", "most_sig_fisher_adj_pval"]]

fg = res.index[res["fdr"] < 0.05]
binary = ap.kinase.kinase_enrichment_binary(fg, res.index, seq)
```

- `kinase_mea` — GSEA-style weighted Kolmogorov–Smirnov on the full ranking (no
  threshold; the most rigorous). `rank_col="t_stat"` is preferable to `"log2fc"` because it
  accounts for per-site variability. `threads=1` by default (deterministic for a seed).
- `kinase_enrichment_from_diffexp` — Fisher's exact per direction (up / down vs
  unchanged); classical KSEA.
- `kinase_enrichment_binary` — Fisher's exact for any foreground / background.

Shared conventions:

- `kl_method="percentile"`, `kl_thresh=90`: a site counts as a candidate substrate of a
  kinase when its percentile for that kinase is ≥ 90 (top 10 %). `kl_method="score"` uses the
  raw log2 score threshold instead.
- **Multiplicity variants** (`…|M1`, `…|M2`) share one window; `dedup_sequences=True`
  (default) keeps one row per window (largest `|rank_col|` for MEA, smallest p for Fisher)
  so no substrate is counted twice.
- Sites with missing / sentinel windows are dropped (count logged). A kinase pool that
  cannot be tested (typically tyrosine, when few Y sites are quantified) is omitted with a
  warning; if every pool fails a `RuntimeError` is raised.
- `quiet=True` silences kinase-library's stdout output.

On the EGF HeLa series (DIA-NN, observed-only limma, 12k sites) `kinase_mea` ranks
ERK1/ERK2, RSK2, p90RSK, AKT1, MAPKAPK2/3 and MSK1/2 up at FDR 0 — the same picture as
the network-based [`kinase_activity`](enrichment/ksea.md). The tyrosine pool had only 41
quantified sites (EGFR autophosphorylation sites are on/off events, absent from the limma
table), so it is uninformative there.

## Kinase names

kinase-library uses its own kinase names (`ERK1`, `MEK1`, `P90RSK`, `PKACA`, …), not
gene symbols (`MAPK3`, `MAP2K1`, `RPS6KA1`, `PRKACA`); OmniPath-based results use gene
symbols. Map before comparing the two.

## References

- Johnson JL et al. 2023. *An atlas of substrate specificities for the human serine/threonine
  kinome.* Nature 613:759–766.
- Yaromenko A et al. 2024. *The tyrosine kinome atlas.* Nature 629:1174–1181.
- Wiredja DD et al. 2017. *The KSEA App.* Bioinformatics 33:3489–3491 (classical KSEA).
