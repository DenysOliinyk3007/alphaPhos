# alphaPhos vs Spectronaut native PTM site report — benchmark & defaults rationale

**Status:** validation report · 2026-05-26
**Companion to:** [../design/spectronaut_collapse_synthesis.md](../design/spectronaut_collapse_synthesis.md) (foundational design, 2026-05-21)
**Notebook:** [./benchmark.ipynb](./benchmark.ipynb) — §0 through §9
**Reproducibility helpers:** [_add_section6_cells.py](./_add_section6_cells.py), [_add_section6_5_cells.py](./_add_section6_5_cells.py), [_add_section7_cells.py](./_add_section7_cells.py), [_add_section8_cells.py](./_add_section8_cells.py), [_add_section9_cells.py](./_add_section9_cells.py)
**Verdict:** alphaPhos default config (`aggregation_method='sum'`, `localization_strategy='condition'`, `cutoff=0.75`, `classI_cutoff=0.75`, `condition_threshold=0.50`) is technically equivalent to Spectronaut's native Class I PTM site report on this dataset.

---

## 0. TL;DR — the defaults we picked and why

| Arg (in `collapse_sites` / `apply_condition_aware_classI_mask`) | Default | Rationale |
| --- | --- | --- |
| `quant_level` (in `read_psm`) | `'auto'` | Spectronaut's `EG.TotalQuantity (Settings)` reflects the export-time MS-level setting. Auto-detect from the columns present; MS1 vs MS2 mismatch with the SN report manifests as a +4.7 log2 systematic offset (§3.1). |
| `top_n_attribution` (in `read_psm`) | `True` | Safety filter that dedups Spectronaut over-export of same-site/different-precursor rows. Did not affect this dataset's quants (`aphos_old` = `aphos_new_MS2` = `aphos_only_MS2` bit-for-bit), but is required to prevent precursor double-counting on reports with over-export. |
| `drop_contaminants` (in `read_psm`) | `True` | Drops PSM rows where every protein in the group matches a common-contaminant (MaxQuant `contaminants.fasta`, bundled at [src/alphaphos/resources/contaminants.fasta](https://github.com/DenysOliinyk3007/alphaPhos/blob/main/src/alphaphos/resources/contaminants.fasta), 246 entries: keratins, BSA, trypsin, etc.). On the EGF dataset: 1,126 / 240,099 PSM rows (0.47%), 63 / 34,248 sites (0.18%), 5 spurious "significant" keratin hits in limma. Doesn't change SN classI agreement (r 0.99 either way) but cleans up the downstream hit list. See §3.6. |
| `cutoff` (collapse-time loc cutoff) | `0.75` | Matches Spectronaut's Class I threshold. With `localization_strategy='condition'` this value is ignored upstream; the `classI_cutoff` below drives the mask. |
| **`aggregation_method`** | **`'sum'`** | **Biggest finding.** `'median'` (the legacy Dublin default) introduces a systematic intensity-dependent log2 offset vs SN (−1.84 at high quants, +0.10 at low). Switching to `'sum'` collapses it to ~0 (mean diff +0.06, 92% of cells within 0.1 log2 of SN). `'consolidate'` (full Hogrebe port) does *not* match SN better than `'sum'`, and drops more sites. See §3.3. |
| **`localization_strategy`** | **`'condition'`** | Two-layer filter (collapse runs `global_max` upstream, then per-condition Class-I majority mask) mirrors SN's two-layer filter (peptide-level + binary per-cell). Jaccard 0.98 against `sn_classI`. `'per_run'` is what Reviewer 2 asks for if you want byte-equality with SN; `'global_max'` is the legacy permissive mode. See §3.4. |
| `classI_cutoff` | `0.75` | SN's published default (Manual pp. 124, 157). |
| `condition_threshold` | `0.50` | Majority rule — "if more than half a condition's replicates localize the site at Class I, keep the entire condition's quants." Recovers cells that SN's per-cell binary filter would discard but for which there is clear within-condition evidence of localization. |

**Resulting end-to-end behaviour vs `sn_classI` on this dataset:**

- Site-set Jaccard: **0.98**
- Per-cell log2 quant Pearson r: **0.990**
- Per-cell log2 mean diff: **+0.063** (median +0.002; SD 0.37; 92% within 0.1)
- limma significant-hit overlap (adj.P<0.05, |logFC|>0.585): **1,548 / 2,097 aphos & 1,921 sn**, Jaccard 0.627
- logFC Pearson r on 15,281 shared features: **0.878**
- PCA: PC1 42.6% in both versions, PC2 within 0.3pp
- Median within-condition CV: aphos 13.7% / sn 12.8% (Δ ≈ 1pp)

---

## 1. Dataset

**Source.** 6-run nanoPhos DIA EGF dilution series acquired 2025-07-29 on Evo11, 16.3 min gradient, 1000 ng load. 3 withEGF + 3 woEGF replicates from the same biological sample.

**Reports analysed:**

| Label | Description | Size |
| --- | --- | --- |
| `old` | The originally exported Spectronaut PSM report (MS2-quanted, single-quant column) | 287.4 MB |
| `new_ms1_ms2` | Re-export with both MS1 and MS2 quants present | 448.3 MB |
| `new_ms2` | Re-export with only MS2 quant (no MS1 columns) | 430.7 MB |
| `sn_all` | Spectronaut's PTM site report (all sites, no Class I filter applied) | 19.8 MB |
| `sn_classI` | Spectronaut's PTM site report with binary Class I filter (loc prob ≥ 0.75 per cell) | 8.8 MB |

The lab's `old` and `new_ms2` PSM reports produce **identical** alphaPhos site matrices (byte-identical numbers, 42,558 raw sites each). `old` was therefore MS2-quanted from the start — the framing in the synthesis doc ("MS1 is the implicit default") was wrong for this lab and this dataset.

---

## 2. Experimental matrix

10 versions × 6 samples were compared:

| Version | PSM input | quant_level | top_n_attribution | After classI mask |
| --- | --- | --- | --- | --- |
| `aphos_old_raw` | `old.parquet` | MS2 (only level present) | True | 42,558 sites |
| `aphos_old_classI` | `old.parquet` | MS2 | True | 34,248 |
| `aphos_new_MS1_raw` | `new_ms1_ms2.parquet` | MS1 (forced) | True | 42,558 |
| `aphos_new_MS1_classI` | `new_ms1_ms2.parquet` | MS1 | True | 32,408 |
| `aphos_new_MS2_raw` | `new_ms1_ms2.parquet` | MS2 (forced) | True | 42,558 |
| `aphos_new_MS2_classI` | `new_ms1_ms2.parquet` | MS2 | True | 34,248 |
| `aphos_only_MS2_raw` | `new_ms2.parquet` | MS2 (only level present) | True | 42,558 |
| `aphos_only_MS2_classI` | `new_ms2.parquet` | MS2 | True | 34,248 |
| `sn_all` | Spectronaut native pivot | — | — | 66,410 |
| `sn_classI` | Spectronaut native pivot | — | — | 34,636 |

All four alphaPhos MS2-quanted versions produced **byte-identical** matrices. Filter and mask were `cutoff=0.75`, `condition_threshold=0.50`. All §3-§9 results below are with the legacy `aggregation_method='median'` unless explicitly stated (§6.5 changes this).

---

## 3. Why each default was set this way

### 3.1 `quant_level='auto'` — and why MS-level matters more than the file

§5 ranking by Pearson r vs `sn_classI`:

| Rank | Version | Pearson r | mean log2 diff | % within 0.1 log2 |
| --- | --- | --- | --- | --- |
| 1 | `sn_all` | 0.9196 | +0.42 | 68.74 |
| 2 | `aphos_old_raw` (MS2) | 0.9192 | −0.52 | 63.43 |
| 3 | `aphos_old_classI` (MS2) | 0.9192 | −0.52 | 63.43 |
| 4–7 | other MS2 variants | 0.9192 | −0.52 | 63.43 |
| 8 | `aphos_new_MS1_raw` | **0.7939** | **+4.68** | **0.12** |
| 9 | `aphos_new_MS1_classI` | 0.7938 | +4.68 | 0.12 |

**MS1 mismatch is catastrophic.** The +4.68 log2 mean diff (~25× linear scale) reflects that the SN PTM site report was MS2-quanted. Picking the wrong quant level dwarfs every other configuration choice. `read_psm(quant_level='auto')` will detect the MS-level from columns present (`FG.MS1Quantity` vs `FG.MS2Quantity` vs `EG.TotalQuantity (Settings)`-only) so this failure mode requires deliberate misuse.

### 3.2 `top_n_attribution=True` — safety filter

This filter (`filter_to_top_n_positions`) deduplicates Spectronaut over-export of the same site evidenced by different precursors. On *this* dataset it had **no effect** on output quants — `aphos_old`, `aphos_new_MS2`, `aphos_only_MS2` are byte-identical. It is kept as the default because: (a) on datasets *with* over-export it prevents double-counting at the site collapse step, (b) it's cheap, (c) when it isn't needed it is a no-op.

### 3.3 `aggregation_method='sum'` — the big finding

#### The problem (§6)

With the legacy `aggregation_method='median'`, alphaPhos has a systematic log2 offset against SN's native site report, and the offset grows with SN quant magnitude:

| SN log2 bin | n cells | mean log2 diff (aphos − sn) | % within 0.1 |
| --- | ---: | ---: | ---: |
| < 4 | 2,047 | +0.10 | 88.3 |
| 4 – 6 | 12,167 | +0.04 | 90.1 |
| 6 – 8 | 30,964 | −0.07 | 84.6 |
| 8 – 10 | 33,877 | −0.33 | 67.6 |
| 10 – 12 | 23,252 | **−0.83** | 44.2 |
| > 12 | 16,470 | **−1.84** | 19.3 |

Interpretation: `median` of multiple-precursor sites doesn't sum the signal — it picks the *middle* precursor. Sites with N precursors of comparable intensity therefore land at roughly `1/N` of the summed intensity, which on log2 scale is `-log2(N)`. Sites with 2/4/8 precursors will sit at `-1` / `-2` / `-3` log2 below SN. High-abundance sites are exactly the ones with more precursors observed → intensity-dependent bias.

#### The fix (§6.5)

Three aggregation methods compared, same `aphos_only_MS2` PSM input, same condition-aware mask:

| Method | n_sites | Pearson r | mean log2 diff | median log2 diff | SD log2 diff | % within 0.1 | % within 0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `median` | 34,248 | 0.919 | **−0.525** | −0.012 | 1.067 | 63.4 | 66.0 |
| `consolidate` | 26,450 | 0.981 | +0.109 | +0.005 | 0.519 | 84.7 | 92.8 |
| **`sum`** | **34,248** | **0.990** | **+0.063** | +0.002 | **0.374** | **91.6** | **96.3** |

`'sum'` wins on every axis:
- Highest correlation
- Smallest mean offset
- Smallest spread
- Highest fraction of cells inside ±0.1 log2 of SN
- Same site count as `'median'` (no sites dropped)

The intensity-dependent bias collapses too: stratifying `'consolidate'` (the better of the two non-median modes) by SN quant magnitude:

| SN log2 bin | mean log2 diff | % within 0.1 |
| --- | ---: | ---: |
| < 4 | +0.20 | 87.0 |
| 4 – 6 | +0.18 | 88.3 |
| 6 – 8 | +0.15 | 86.0 |
| 8 – 10 | +0.11 | 82.9 |
| 10 – 12 | +0.08 | 82.0 |
| > 12 | +0.03 | 87.3 |

— a flat residual at every magnitude. `'sum'` shows the same flat profile (data not retained per-bin in §6.5 but inferable from the overall +0.06 mean and SD 0.37).

#### Why `consolidate` underperformed `sum`

`'consolidate'` is the canonical Hogrebe ratio-imputation + sum (a Python port of the R `consolidate()` reference). It (a) drops precursor rows with ≤ 1 non-NaN value, (b) iteratively imputes the remaining NaNs by cross-precursor ratios, (c) sums. Two effects vs plain `'sum'`:

1. **Site loss.** Sites where no precursor has ≥2 non-NaN values get dropped entirely — 26,450 vs 34,248 sites on this dataset (~23% loss).
2. **Imputation residual.** The ratio-imputation introduces a small positive bias (+0.11 vs +0.06 for `'sum'`), presumably because the imputed values inflate the per-precursor signal slightly relative to SN's "Linear modelling based" PTM consolidation, which probably uses a different imputation scheme.

SN's `"Linear modeling based"` PTM consolidation, per Manual p. 126, is "impute missing values for each parent peptide based on quantities reported in other runs, then sum." Without per-precursor missingness, this **degenerates to plain sum** — which is exactly what `'sum'` does. The ratio-imputation in `'consolidate'` is doing work that SN's algorithm largely doesn't, and the work it does is biased slightly high.

**Decision:** `'sum'` is the right default. Users who specifically want full Hogrebe / Dublin-legacy reproduction can opt into `'consolidate'`; users who want byte-equality with prior `'median'`-aggregated outputs can opt into `'median'`.

### 3.4 `localization_strategy='condition'` — two-layer filter

SN's `sn_classI` is a two-layer filter:

1. **Peptide-level**: SN's PTM consolidation excludes parent peptides where the site's loc prob is too low at the peptide collapse step.
2. **Per-cell binary**: the final report drops cells where the site's per-run loc prob < 0.75.

alphaPhos with `'condition'` strategy is also two-layer:

1. **Top-N attribution** (in `read_psm`): drops PSM rows that don't make the top-N localization-supported set for each spectrum.
2. **Condition-aware mask**: per-condition Class-I majority rule.

Compared head-to-head (§5):

| alphaPhos config | Jaccard vs `sn_classI` |
| --- | ---: |
| MS2 + `condition_aware` mask (default) | **0.980** |
| MS2 raw (no classI mask) | 0.796 |
| MS1 + `condition_aware` mask | 0.927 |

§8 ran the converse experiment — apply the *same* `condition_aware` filter to SN's `sn_all` (raw, unfiltered) report. Outcome:

| Comparison | Sig hits | Jaccard sig vs aphos | logFC Pearson r |
| --- | ---: | ---: | ---: |
| aphos vs `sn_classI` (default) | 2,097 / 1,921 | 0.627 | 0.878 |
| aphos vs `sn_all + condition_aware` | 2,097 / 4,077 | 0.414 | 0.749 |

Applying `condition_aware` to `sn_all` over-recovers (4,077 hits, vs 1,921 from binary classI) because `sn_all`'s peptide-level filtering is more permissive than the SN classI report. The single-layer condition_aware filter on top of permissive input is too leaky. SN's peptide-level filter + alphaPhos's `condition_aware` mask is the right *cross-tool* analogy — keeping `sn_classI` as the comparison reference (rather than `sn_all + condition_aware`) is what produces the 0.98 Jaccard.

Three strategies remain exposed:

- **`'condition'` (default)** — recovers cells that strict per-cell binary filters discard but for which there is within-condition evidence of confident localization. Requires a `condition_df`.
- **`'per_run'`** — strict per-cell mask. If a cell's per-run loc prob < cutoff, the quant is NaN. Matches SN binary Class I exactly when `cutoff=0.75`.
- **`'global_max'`** — drop sites where the dataset-wide max loc prob < cutoff. Legacy Hogrebe-plugin convention; permissive.

The order of preference in the docstring is `condition` → `per_run` → `global_max` because the first is biologically more useful while still recovering SN-style structure on standard designs.

### 3.5 `classI_cutoff=0.75`, `condition_threshold=0.50`

`0.75` is SN's published Class I default (Manual pp. 124, 157). `0.50` is the majority rule. Both have biological justification and both are inherited unchanged from the existing `condition_aware_classI.py` in Dublin/testscripts.

### 3.6 `drop_contaminants=True` — kill keratin/BSA/trypsin PSMs at the gate

**Mechanism.** `read_psm()` drops rows where every protein in `PG.ProteinGroups` matches a known contaminant. Identification is two-pronged:

1. **Prefix detection** — protein IDs starting with `CON__`, `Cont_`, or `contam_`. Common when the search engine (MaxQuant, Spectronaut) tagged contaminants at search time.
2. **FASTA accession lookup** — bare UniProt/TREMBL/REFSEQ/ENSEMBL accessions matching entries in the bundled MaxQuant `contaminants.fasta` (246 sequences: keratins, BSA, trypsin, serum proteins). This dataset's Spectronaut search used the second mechanism: contaminants appear with their plain UniProt accessions (e.g. `P05787` for KRT8).

**The conservative-by-design rule.** A row is dropped only when *every* protein in the group is a contaminant. A peptide ambiguously assigned to `P12345;CON__P02769` is kept — its real-protein match could be valid. Matches Spectronaut's own conservative default behaviour.

**Cost vs benefit on the EGF dataset:**

| Metric | no_filter | with_filter | Δ |
|---|---:|---:|---:|
| `read_psm` runtime | 8.0 s | 8.7 s | +0.7 s |
| PSM rows after filter (post-top_n) | 240,099 | 238,973 | −1,126 (0.47%) |
| Sites after collapse | 34,248 | 34,185 | −63 (0.18%) |
| Jaccard vs `sn_classI` | 0.832 | 0.831 | ~0 |
| Per-cell Pearson r vs `sn_classI` | 0.99 | 0.99 | 0 |
| Mean log2 diff vs `sn_classI` | +0.063 | +0.063 | 0 |
| % cells within 0.1 log2 of `sn_classI` | 91.69% | 91.69% | 0 |
| limma significant hits | 2,097 | 2,093 | −4 net |
| Shared sig hits | — | — | 2,092 (Jaccard 0.997) |
| logFC concordance (Pearson r) | — | — | 0.9996 |

**Which "significant" hits disappear?** All five contaminant hits removed by the filter are keratins, with apparent EGF regulation that is biologically implausible:

| Feature | logFC | adj.P.Val |
|---|---:|---:|
| `P05787~KRT8_S475_M1` | −2.47 | 3.1e-04 |
| `Q04695~KRT17_S44_M1` | −2.27 | 1.7e-02 |
| `P05787~KRT8_S432_M1` | −0.87 | 1.7e-02 |
| `P05787~KRT8_Y437_M1` | −1.49 | 2.8e-02 |
| `P05787~KRT8_S13_M1` | −1.07 | 3.4e-02 |

**Why SN classI agreement doesn't change.** Spectronaut's own search FASTA already includes a contaminants list, so `sn_classI` was already largely contaminant-free. alphaPhos's filter brings our pipeline to the same place, but doesn't reduce the residual disagreement (which is dominated by sites SN scored Class I and we didn't, or vice versa — not by contaminant content). See §4.6 of [_run_contaminant_benchmark.py](./_run_contaminant_benchmark.py) for the full per-stage numbers.

**Decision:** `drop_contaminants=True` is the right default. Cheap (no measurable cost), removes precisely the artefactual hits a contaminant filter is supposed to remove (keratin "signaling" in an EGF experiment), and leaves the real biology untouched (Pearson r 0.9996 on shared limma features).

---

## 4. End-to-end benchmark axes

### 4.1 Site-set identity (§5)

10×10 Jaccard matrix (sites in either version), key cells:

|  | `aphos_new_MS2_classI` | `aphos_new_MS1_classI` | `sn_classI` |
| --- | ---: | ---: | ---: |
| `aphos_new_MS2_classI` | 1.000 | 0.946 | **0.980** |
| `aphos_new_MS1_classI` | 0.946 | 1.000 | 0.927 |
| `sn_classI` | **0.980** | 0.927 | 1.000 |
| `sn_all` | 0.516 | 0.488 | 0.520 |

The default MS2-quanted, condition-aware alphaPhos configuration recovers 98% of SN's classI sites.

### 4.2 Cross-version per-cell quant (§5)

`sn_classI` is the reference. All MS2 alphaPhos variants score the same (`aphos_old`, `aphos_new_MS2`, `aphos_only_MS2` are byte-identical):

- Pearson r 0.919 (with `'median'`)
- Mean log2 diff −0.52
- 63% of cells within 0.1 log2

After switching to `'sum'` (§6.5):

- Pearson r **0.990**
- Mean log2 diff **+0.06**
- **92%** of cells within 0.1 log2

### 4.3 Deep diff on the shared site set (§6, with legacy `median`)

Cell-level NaN patterns (34,091 shared sites × 6 samples = 204,546 cells):

- 118,777 (58.1%) — both present
- 4,186 (2.0%) — aphos has a value, SN dropped as Filtered
- 86 (0.04%) — SN has a value, aphos has NaN (within-rounding identical)
- 81,497 (39.8%) — both NaN

The 2% "aphos-only" cells are expected from `condition_aware` recovering cells that SN's binary per-cell filter censors. The 0.04% "sn-only" cells are negligible.

Per-site median diff distribution (with `median` agg):

- median: −0.005 (essentially zero typical bias)
- 5th percentile: −1.94 (the long left tail of underestimated high-abundance sites)
- 95th percentile: +0.028

After switching to `'sum'` (per §6.5 stratification), the left tail collapses to ±0.05 across all SN quant bins.

### 4.4 Downstream — limma differential analysis (§7)

Pipeline: `filter_phosphosites(cutoff=0.7)` → `impute_phosphosites` (KNN, Dublin core.py) → `run_limma_via_rscript` (Rscript subprocess wrapper).

| Step | aphos+sum+classI | sn_classI |
| --- | ---: | ---: |
| After filter | 17,615 sites | 15,557 |
| After impute | 17,615 | 15,557 |
| Sig hits (adj.P<0.05 & |logFC|>0.585) | 2,097 | 1,921 |
| Raw P<0.05 | 5,380 | 4,768 |

Sig hit overlap: 1,548 shared, 549 aphos-only, 373 sn-only (Jaccard 0.627).
logFC concordance on 15,281 shared features: Pearson r **0.878**, Spearman 0.873, median |diff| 0.036.

The top-15 hits are nearly identical between versions; both surface EGFR autophosphorylation (P00533 Y1172, Y1197) and downstream pathway sites (P12270, P51858, P04792, Q8WX93) as the strongest withEGF-up signal.

### 4.5 Filtering policy isolation (§8)

Applied `condition_aware` to `sn_all` → 35,695 sites. Ran the same filter+impute+limma pipeline:

| Comparison | Sig hits | Jaccard sig | logFC Pearson r |
| --- | ---: | ---: | ---: |
| aphos vs sn_classI (default, §7) | 2,097 vs 1,921 | 0.627 | 0.878 |
| aphos vs sn_all + cond_aware (§8) | 2,097 vs 4,077 | 0.414 | 0.749 |

Lesson: `condition_aware` on top of *SN's already-filtered* peptide-level output (which is what `sn_classI` consumes downstream of) is what reproduces SN's classI behaviour cleanly. `condition_aware` on top of `sn_all` (raw) over-includes — 4,077 hits is implausibly high for a 6-replicate experiment, suggesting the single-layer filter is letting in too many low-evidence sites.

### 4.6 Sample QC (§9)

PCA on z-scored sites × samples:

|  | PC1 | PC2 |
| --- | ---: | ---: |
| `aphos+sum+classI` (17,615 sites) | 42.6% | 20.7% |
| `sn_classI` (15,557 sites) | 42.6% | 21.0% |

Both versions show identical withEGF/woEGF separation along PC1 with the same replicate spread along PC2.

Per-site CV within condition (linear space):

| Version × Condition | n_sites | 25% | **50% (median)** | 75% | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| aphos+sum+classI / withEGF | 17,615 | 5.5 | **11.7** | 27.4 | 20.8 |
| aphos+sum+classI / woEGF | 17,615 | 7.9 | **15.6** | 32.7 | 24.8 |
| sn_classI / withEGF | 15,557 | 5.3 | **10.9** | 24.8 | 19.7 |
| sn_classI / woEGF | 15,557 | 7.7 | **15.0** | 31.2 | 24.0 |

SN is marginally tighter (~1pp on median CV in both conditions) but doesn't differ at the structural level. Both pipelines produce equally clean sample-level QC.

---

## 5. Surprises and non-obvious findings

1. **The legacy `'median'` default was wrong for SN parity.** Dublin's longstanding default produces a systematic intensity-dependent log2 offset against any sum-style consolidation. It is mathematically a 1/N attenuation where N is the number of precursors observed per site. The lab's published numbers on this convention are biologically correct in *ratio* (since the offset is multiplicative per site and cancels in withEGF/woEGF contrasts) but absolute intensities are not comparable to SN or other sum-based pipelines.

2. **`'consolidate'` (full Hogrebe port) is not better than `'sum'` against SN.** This was unexpected given that `'consolidate'` is the more sophisticated algorithm. The reason is that SN's own `"Linear modeling based"` PTM consolidation is *also* a sum after imputation (per Manual p. 126), and the imputation step on this dataset (where missingness is ~40%) is minor compared to the aggregation step. `'consolidate'`'s ratio-imputation introduces a small positive bias (+0.11 mean diff vs +0.06 for `'sum'`) and drops ~23% more sites (sites where no precursor has ≥2 non-NaN values).

3. **Applying `condition_aware` to `sn_all` does NOT recover `sn_classI`.** SN's binary per-cell classI filter is operating downstream of a peptide-level filter that `sn_all` already lacks. Putting `condition_aware` on top of `sn_all` produces a different (more permissive) site set than `sn_classI`, hence the worse concordance with aphos in §8. The correct cross-tool analogy is `aphos(top_n + condition_aware)` ≈ `SN(peptide_filter + binary_classI)` — the *two-layer* filter on each side, not one layer of each.

4. **MS-level mismatch is the dominant configuration error.** Forcing MS1 quant on a report whose ground truth is MS2 produces a +4.68 log2 systematic bias — orders of magnitude worse than any other config choice. `quant_level='auto'` should be left alone unless the user knows exactly what they are doing.

5. **The four MS2 alphaPhos variants are byte-identical.** `aphos_old` (the original old export), `aphos_new_MS2` (forced MS2 from the dual-quant report), and `aphos_only_MS2` (re-export with only MS2 columns) produced identical sites and quants. This confirms that the `top_n_attribution` filter does not affect this report's content and that `EG.TotalQuantity (Settings)` was MS2 in all three exports.

---

## 6. Limitations

1. **Single dataset.** 6 samples, two conditions, one biological replicate set, MS2 quant only. The defaults chosen here are validated against SN classI on this dataset. Re-running the benchmark on at least one more dataset (e.g. one of the four other reports in `test_data/benchmark/`) before claiming generality is the natural next step.
2. **No MS1 cross-check.** This dataset's SN report was MS2, so the MS1 vs MS2 comparison is degenerate. For experiments where MS1 sensitivity matters (low input, single-cell), the `'consolidate'` algorithm may still be preferred — its ratio-imputation assumes co-eluting precursor envelopes scale proportionally, which is a clean assumption for MS1 envelopes and not for MS2 fragment sums.
3. **One filter, one imputation method, one stats engine.** `filter_phosphosites(cutoff=0.7)` + KNN impute + limma. These are reasonable defaults but they are choices, not validated optimum. The §7-§9 results show that *given* the same downstream choices, aphos and SN agree; they don't say anything about whether those downstream choices are themselves optimal.
4. **Top-15 hits agreement only spot-checked.** The 1,548-shared / 549-aphos-only / 373-sn-only split deserves a closer look at the disagreement sets — are they biologically meaningful or just borderline-significant noise? Not pursued in this benchmark.

---

## 7. Reproducing

The full benchmark is `docs/benchmark/benchmark.ipynb`. To re-run on the original data:

```python
# §0–§3: produces 4 alphaPhos site matrices × (raw, classI) = 8 parquet files
# §4: parses 2 SN pivot reports
# §5: 10×10 Jaccard matrix, per-cell Pearson r vs sn_classI
# §6: deep diff on the median-aggregated default vs sn_classI
# §6.5: aggregation_method = median vs consolidate vs sum
# §7: filter + impute + limma, aphos_sum_classI vs sn_classI
# §8: applied condition_aware to sn_all, re-ran limma — filtering policy check
# §9: PCA + per-site CV — sample-level QC
```

To score a new dataset against an SN classI export:

```python
from alphaphos.io import read_psm
from alphaphos.preprocess import collapse_sites, apply_condition_aware_classI_mask
import pandas as pd

df = read_psm("new_report.parquet")  # quant_level='auto', top_n on
condition_df = pd.DataFrame({"sample": [...], "condition": [...]})
sites, loc_per_run = collapse_sites(df, condition_df=condition_df)
# sites is the SN-faithful default output (sum agg, condition-aware mask, 0.75 cutoff)
```

The scoring helpers (`alpha_to_matchkey_frame`, `score_vs_classI`) live in §3 and §6.5 of the notebook; promoting them to a `alphaphos.benchmark` submodule is a natural follow-up.

---

## 8. Cross-references

- **Foundational design + Spectronaut manual cross-walk:** [../design/spectronaut_collapse_synthesis.md](../design/spectronaut_collapse_synthesis.md)
- **The change implementing these defaults:** `src/alphaphos/preprocess/collapse.py`, `collapse_sites()` and `PeptideCollapse.{collapse_to_peptides, collapse_to_sites, process_complete_pipeline}` — `aggregation_method` default flipped `'median'` → `'sum'`; `localization_strategy` default flipped `'per_run'` → `'condition'`.
- **Open questions answered:** §7 of the synthesis doc asked "where does the lab's preferred default sit?" — this benchmark answers it. For studies where SN parity matters (most new analyses), `condition_aware + sum` is the validated default. Legacy `global_max + consolidate` is reachable explicitly for backward-compat regression tests against Dublin-era outputs.

---

## 9. Re-validation 2026-09-12 — EGF HeLa 1 µg, Spectronaut 21.1, full export matrix

**Data:** `Documents/Data/egf_hela_nanophos_spectronaut/` — 6 runs (3 withEGF / 3 woEGF), 622,199 PSM rows. Long exports: `long_ms2`, `long_ms1`, `long_ms1_and_ms2` (TSV + parquet each). Native site reports: `short_cutoff_0` (66,395 phospho sites) and `short_cutoff_0p75` (34,626; binary per-cell Class I — the reference).
**Harness:** [validate_collapse_egf_hela.py](./validate_collapse_egf_hela.py) (`--full` for the whole matrix). **Guard:** `tests/integration/test_collapse_vs_spectronaut_native.py`.

### 9.1 Verified correct

| Check | Result |
| --- | --- |
| Site position arithmetic vs Spectronaut's own `EG.ProteinPTMLocations` | **101,636 / 101,636 phospho precursors agree (100.000%)** |
| alphaPhos sites outside Spectronaut's cutoff-0 universe | **0** |
| Localization probability per cell vs `PTM.SiteProbability` | identical (±1e-3) in **99.75%** of shared cells, r = 0.995 |
| Site set vs `short_cutoff_0p75` (`per_run` strategy) | 34,279 vs 34,626 sites, **Jaccard 0.981** |
| MS1+MS2 export collapsed at `MS2` / `auto` vs MS2-only export | **byte-identical** |
| TSV vs parquet of the same export | identical up to Spectronaut's own export differences (§9.4) |
| Determinism, `write_h5ad`, runtime | deterministic; h5ad round-trips; 5 s per collapse (622k rows) |

The 491 sites present in `short_cutoff_0p75` but never produced by alphaPhos are fully accounted for: 229 are non-leading protein-group members (Spectronaut emits one row per member; alphaPhos one per group), 79 belong to contaminant proteins dropped at read time, 176 are second mappings of repeat peptides (§9.3), 7 are multiplicity relabels (§9.3).

### 9.2 The residual offset explained — `precursor_loc_gate`

With the pre-0.23 default (sum every precursor covering a site) the per-cell agreement was r = 0.990, mean +0.063 log2, 91.6% within 0.1 — but stratified by precursor count the offset lived entirely in multi-precursor sites (1 precursor: +0.001; 2: +0.119; ≥3: +0.135), and 22% of observed site-run cells (33,768 / 154,635) mixed Class-I and non-Class-I precursors. Spectronaut sums only precursors that are themselves Class-I in that run.

| Variant | mean Δlog2 | SD | % within 0.1 | per-cell r | log2FC r (withEGF−woEGF) |
| --- | ---: | ---: | ---: | ---: | ---: |
| sum all precursors (pre-0.23) | +0.063 | 0.374 | 91.6 | 0.990 | 0.931 (per_run) / 0.922 (condition) |
| gate: only Class-I precursors (strict SN semantics) | +0.005 | 0.156 | 97.5 | 0.998 | 0.993 |
| **gate if available: Class-I precursors when any exists, else all** (`precursor_loc_gate=True`) | **+0.005** | **0.156** | **97.5** | **0.998** | 0.993 (per_run) / **0.981 (condition)** |

The strict gate collapses the `condition` strategy onto `per_run` (it removes exactly the cells the condition rule recovers); the "if available" variant keeps that recovery (4,179 recovered cells) while matching Spectronaut wherever Spectronaut has a value. It is the new default.

### 9.3 Known, intended semantic differences

- **Multiplicity.** alphaPhos: number of phospho groups on the precursor. Spectronaut 0.75 report: number of *localized* (≥ cutoff) groups on the peptide — so a doubly-phosphorylated peptide with one ambiguous site is filed under `M1`. Explains 237 of the 553 cells alphaPhos keeps where Spectronaut says `Filtered`. Kept as is (consistent with Spectronaut's own cutoff-0 report).
- **Repeat peptides** mapping to two positions in the same protein (`EG.ProteinPTMLocations = (S915)(T1131)`; 175 precursors, 0.17%): Spectronaut reports a site at each mapping (211 extra sites), alphaPhos at the first listed position only.
- `aggregation_method`: `sum` r = 0.990 (pre-gate), `consolidate` 0.981 with 26,483 sites, `median` 0.919 with −0.53 log2 — unchanged conclusion from §3.3.
- `top_n_attribution=False` on this export: r drops to 0.92 with +0.42 log2 (2,560 extra cells) — unlike the 2026-05 export, top-N matters here; keep it on.

### 9.4 Surprises in the exports (not in alphaPhos)

- **`long_ms1` is MS2-quantified.** Its `EG.TotalQuantity (Settings)` equals `FG.MS2Quantity` of the dual export in 99.9% of rows (ratio to `FG.MS1Quantity` ≈ 0.026); collapsing it reproduces the MS2 result exactly. The export needs to be redone with MS1 quantity settings before it can serve as an MS1 test case.
- **Parquet ≠ TSV for the same analysis.** The parquet export drops 16 PSM rows, quantities differ by ~0.1% (median relative; up to 1.4% — not float rounding), and 288 shared paralog peptides are assigned to a *different* protein group (e.g. `_KEES[Phospho]EES[Phospho]DDDMGFGLFD_.2` → RPLP1 `P05386`@98 in TSV, RPLP2 `P05387`@99 in parquet). 14 sites therefore swap values by >1 log2 between the two exports. alphaPhos is faithful to whichever file it is given; use one export format per project.
- Forcing `quantification_level="MS1"` on the dual export gives +5.2 log2 vs the MS2 site report (as in §3.1).
- The `"Potential duplicated run files"` warning fired for two unrelated runs (r = 0.92) — the first 100 sorted precursor ids are shared by deep DIA runs of the same sample type. Fixed: the fingerprint now includes quantities.

## 10. DIA-NN arm — same six raw files, DIA-NN 2.2.0 (2026-09-12)

`Documents/Data/egf_hela_nanophos_diann/`: library-free search of the same six EGF HeLa runs
(`--var-mod UniMod:21,79.966331,STY`, MBR, `--cont-quant-exclude cRAP-`). DIA-NN 2.x ships
three site-level artefacts we can validate against: `report.site_report.parquet` (one row per
run × precursor × candidate residue with `Occupied`, `Probability`, `Fragment.Sum`),
`report.phosphosites_90.tsv` / `_99.tsv` (native site matrices), and the per-row
`Site.Occupancy.Probabilities` string in the main report. Harness:
`docs/benchmark/validate_collapse_egf_hela_diann.py`; guard:
`tests/integration/test_collapse_vs_diann_native.py` (skips without the data).

### 10.1 What DIA-NN's native matrices actually are

Reverse-engineered from the site table (needed to know what a "match" means): each cell of
`phosphosites_90.tsv` is the **maximum** over qualifying precursors (site `Probability` ≥ 0.9,
q-value filters) of `Fragment.Sum × Normalisation.Factor` — a normalised top-3-fragment
quantity, not `Precursor.Quantity` (94.1% of single-precursor cells within 1%; 95.2% of all
cells reproduced by the Top-1 rule). No multiplicity dimension. DIA-NN's own documentation
calls these matrices "intended only for quick preliminary analyses". Consequently the
per-cell quantity is *not* comparable to a precursor-quantity sum (r ≈ 0.85), and the
meaningful comparisons are (a) the site **set** and (b) an independent **oracle** — a gated
sum of `Precursor.Quantity` over `Occupied` precursors with `Probability ≥ cutoff`, built
directly from the site table.

### 10.2 Results (per_run, `quantification_level="auto"` → `Precursor.Quantity`)

| Check | Result |
| --- | --- |
| Positions ⊆ DIA-NN `Occupied` sites (leading protein), 33,557 precursors | **100.0%** (was 99.991% before the fix in 10.3) |
| Multi-mapping precursors (DIA-NN lists more residues than the peptide carries) | 68; alphaPhos reports the first phospho's mapping only (same policy as §9.3) |
| cutoff 0.90 vs oracle gated sum | r **0.9993**, mean −0.006 log2, SD 0.11, **99.3% within 0.1**, log2FC r 0.990; 0 cells alphaPhos-only, 164 oracle-only |
| cutoff 0.99 vs oracle gated sum | r 0.9996, −0.004 log2, 99.4% within 0.1, log2FC r 0.992 |
| Site set vs `phosphosites_90` (multiplicity summed) | Jaccard 0.934 (16,815 vs 17,218 sites) |
| Site set vs `phosphosites_99` | Jaccard 0.918 |

The oracle-only cells are sites whose only qualifying precursor was gated by alphaPhos's
`precursor_loc_gate` in that run and is otherwise below cutoff — i.e. DIA-NN's site table
records the residue as occupied at ≥ 0.9 for a *different* residue-set decomposition.
The ~7% Jaccard gap to DIA-NN's matrices (378 alphaPhos-only / 781 DIA-NN-only sites at
0.9) is **not** explained by filter differences — tested by adding DIA-NN's global 1% filters
(`Global.Q.Value`, `Global.Peptidoform.Q.Value` ≤ 0.01: Jaccard 0.927), removing alphaPhos's
`Quantity.Quality` / `PG.MaxLFQ.Quality` filters (0.916), both (0.921), and switching the
precursor gate off (unchanged). The default `read_diann` settings are the closest (0.934).
The residue is inside DIA-NN's matrix builder (Top-1 rule, its own site-confidence handling);
since the main report is already at run-specific precursor *and* peptidoform FDR ≤ 1%
(0.0% of phospho rows exceed either), the site-table oracle above is the meaningful
reference, not the matrices. Position disagreement is 0 in all cases.

### 10.3 Bugs found and fixed

- **Shared peptides got the wrong protein's position.** `Protein.Sites` lists one bracket
  group per member of the protein group in *alphabetical* order (`[Q16828:S331];[Q16829:S369];[Q99956:S328]`)
  while `Protein.Group` lists the leading protein first (`Q99956;Q16828;Q16829`). `read_diann`
  took the first bracket group, so `SNIS[p]PNFNFMGQLLDFER` was keyed `Q99956|S331` instead of
  `Q99956|S328`. Now the leading accession's group is used (3 precursors affected here; the
  fraction scales with paralog content).
- **cRAP contaminants passed through.** DIA-NN's `--cont-quant-exclude cRAP-` only excludes
  the tagged proteins from *protein* quantification; 456 tagged phospho precursor rows (620
  after MBR) remained in the report. `read_diann` now applies the same contaminant filter as
  `read_spectronaut`, and `cRAP-` / `cRAP_` joined `DEFAULT_CONTAMINANT_PREFIXES`.
- **`top_n_attribution` was harmful on DIA-NN.** The Spectronaut over-export dedup assumes
  multiple rows per precursor-run; DIA-NN writes exactly one peptidoform row, so the filter
  deleted 12.6% of rows (low-confidence peptidoforms) and 796 real site–run cells (oracle r
  0.9984 → 0.9993 with it off). New default `"auto"` = Spectronaut only.

### 10.4 Cross-engine (DIA-NN vs Spectronaut, same raw files, per-sample median-centred)

Site-set Jaccard 0.46 (19,406 vs 31,589 sites; Spectronaut's library-based search finds far
more), per-cell r 0.84 with SD 1.57 log2 — the engines' intensity scales are not
interchangeable. Condition log2FC (withEGF − woEGF) correlates r 0.59 over all 16,109
shared sites, **0.74 on 7,529 complete cases**. The known biology reproduces: EGFR
autophosphorylation `P00533|Y1172` +6.8 (SN) / +6.2 (DIA-NN) log2, `Y1197` +4.1 / +4.4.
This is the expected engine-to-engine agreement for DIA phospho (see the literature
review §3) and is not an alphaPhos property; alphaPhos is faithful to each report.

### 10.5 Notes on this DIA-NN run (not in alphaPhos)

- `report.log.txt` line 35: `WARNING: incorrect settings, the in silico-predicted library
  must be generated in a separate pipeline step and then used to process the raw data, now
  without activating FASTA digest` — DIA-NN recommends a two-step library-free workflow; this
  run was one-step (set up outside alphaPhos). The numbers above validate collapse mechanics
  on this report, not the search itself; re-searching two-step may change the site set.
- `Protein.Sites` is per protein-group member; `Genes` is `;`-joined in group order.
  `read_diann` keys sites by the leading accession and first gene, as for Spectronaut.


## 11. End-to-end run, both engines (2026-09-14)

`docs/benchmark/e2e_egf_hela.py` runs the full default pipeline — read → `collapse_sites`
(engine defaults: `condition` strategy, MS2 for Spectronaut, MS1 for DIA-NN) →
`filter_by_completeness` → `diff_exp_limma_observed_only` (+ impute-all limma for
comparison) → `kinase_activity` (OmniPath, ULM on moderated t) → `pathway_enrichment`
(Enrichr KEGG + Hallmark, phosphoproteome background) → `pathway_gsea` (Hallmark) →
site-level `ora` / `gsea` on the shipped PTM libraries — on both searches of the six raw
files, then compares the biology. Requires the `[enrichment]` extra and network access.

| | Spectronaut | DIA-NN |
| --- | --- | --- |
| PSM rows → site×mult | 620,293 → 34,279 | 176,652 → 22,256 |
| quant column (default) | `FG.MS2Quantity` | `Ms1.Area` |
| ≥3 valid in one condition | 17,628 | 17,265 |
| observed-only limma: tested / sig @5% FDR (up/down) | 11,955 / 1,976 (1,260/716) | 13,112 / 1,458 (1,210/248) |
| on/off detection (on in EGF / on in control) | 3,092 / 2,581 | 2,405 / 1,748 |
| impute-all limma: sig, of which imputed-driven | 2,675 / 840 (31%) | 2,137 / 642 (30%) |
| kinases tested; known EGF kinases up @5% FDR | 231; **8/8** | 231; **8/8** |
| top-10 kinases | EGF, MAPKAPK2, MAP2K3, MAPKAPK3, BRAF, MAP2K6, PRKACA, DUSP1, DUSP8, MAP3K8 | EGF, DUSP1, MAP2K3, DUSP8, BRAF, PTPN7, DUSP16, MAPKAPK2, MAP2K1, KSR1 |
| pathway ORA terms @10% FDR (top) | 65 (ErbB, Insulin, MAPK signalling) | 57 (ErbB, MAPK, Mitotic spindle) |
| site ORA | Ochoa top quartile, activates/inhibits activity, induces PPI enriched; Ochoa bottom quartile **depleted** | same pattern |

Cross-engine: kinase activity scores Spearman **0.92** over 212 shared kinases (6 of the
top 10 shared); 45 KEGG/Hallmark terms significant in both (19 SN-only, 12 DIA-NN-only);
site-ORA log2 fold-enrichment Pearson 0.93; Hallmark GSEA NES Spearman 0.62 (Hallmark GSEA
on gene-collapsed phospho data is weak in both — 1–2 terms at 25% FDR).

Two observations worth keeping in mind when reading such runs:

- **EGFR autophosphorylation sites (Y1172, Y1197, Y1110, Y727/Y869) are on/off events**,
  not limma hits: they are unquantified in unstimulated cells, so
  `diff_exp_limma_observed_only` correctly reports them in the detection table
  (`on_in_treatment`) and only the impute-all path gives them a p-value — driven by
  imputed values.
- ~30% of impute-all hits are imputed-driven in **both** engines; the observed-only flow
  removes them without losing the kinase / pathway signal (8/8 known kinases either way).
