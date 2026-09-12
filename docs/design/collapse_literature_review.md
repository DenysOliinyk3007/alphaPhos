# Peptide → phosphosite collapse: what the field does, where it is fragile, and how alphaPhos compares

**Status:** literature review + defensibility assessment · 2026-09-12
**Companions:** [spectronaut_collapse_synthesis.md](./spectronaut_collapse_synthesis.md) (design), [../benchmark/spectronaut_native_benchmark.md](../benchmark/spectronaut_native_benchmark.md) (empirical validation, §9)

> **TL;DR.** The field has one de-facto convention for turning phosphopeptide precursors into a site table: *(protein accession, residue+position, multiplicity ≤ 3)* as the key, Class I = localization probability ≥ 0.75, per-precursor-run localization filtering before aggregation, and **summing** the intensities of every precursor that maps to the key. alphaPhos follows that convention exactly with `localization_strategy="per_run"` (r = 0.998 vs Spectronaut's own report) and extends it with a documented middle path (`"condition"`). Its real deviations from the literature are (i) the condition-aware mask (novel; must be described as such), (ii) `top_n_attribution` (a Spectronaut-specific fix), and (iii) multiplicity counted on the precursor rather than on *localized* groups. None of these is indefensible, but all three must be stated explicitly in a methods section. The caveats the literature itself flags — localization scores are tool-specific and only loosely calibrated to false-localization rate, multiplicity fragments biology, summing ignores ionization efficiency, shared/repeat peptides are mis- or un-assigned — apply to alphaPhos as much as to every other pipeline.

---

## 1. The conventions

### 1.1 What a "site" is: key construction

Every mainstream tool builds the same three-part identifier. The *msproteomics sitereport* paper (Bioinformatics 2024), the only work that formalizes site-level reporting for DIA, states it plainly: each phosphosite identifier "is composed of a protein identifier, the site position in the protein, and the so-called phosphorylation multiplicity", where multiplicity "originates from MaxQuant in which 1 denotes single phosphorylation, 2 for double phosphorylation of a peptide, and 3 for three or more phosphorylation sites" ([Pham et al. 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11239223/)). The Perseus *Peptide Collapse* plugin — the reference implementation that most DIA phospho papers of 2020-2024 used — builds "gene/protein identifier _ PTM amino acid type & position _ multiplicity", multiplicity "capped at a maximum of 3" ([Hogrebe, GitHub](https://github.com/AlexHgO/Perseus_Plugin_Peptide_Collapse)). Spectronaut's native `PTM.CollapseKey` (`A0A087WUV0_S118_M1_1`) is the same thing with a modification index appended.

**alphaPhos:** `Protein|Gene|<AA><pos>|M<mult>` with `mult` clipped to 3. Same information, safer delimiter. ✔ Convention.

### 1.2 Class I: the 0.75 localization threshold

The threshold is 20 years old. Olsen et al. (Cell 2006) defined Class I sites as localization probability ≥ 0.75 (with a PTM-score difference ≥ 5 in the MaxQuant implementation) ([Olsen 2006](https://pubmed.ncbi.nlm.nih.gov/17081983/); [summary](https://link.springer.com/protocol/10.1007/978-1-4939-3049-4_20)). It carried unchanged into DIA: Bekker-Jensen et al. (Nat Commun 2020) introduced Spectronaut's DIA localization score — fragments are "classified as either confirming or refuting a specific site candidate" and the score is "a fractional probability … in relation to all candidate scores" — and required "at least 0.75 localization probability (Class I sites)", which on a synthetic phosphopeptide library "equates to an estimated FDR of 1.5%" (1.7% incorrectly assigned sites in the DIA arm) ([Bekker-Jensen 2020 preprint](https://www.biorxiv.org/content/10.1101/657858v1.full); [paper](https://www.nature.com/articles/s41467-020-14609-1)). Kitata et al. 2021 report "36,350 phosphosites (19,755 class 1)" in the same terms ([Kitata 2021](https://www.nature.com/articles/s41467-021-22759-z)).

**Two caveats the field itself raises:**

1. **The number 0.75 is not portable across tools.** Lou et al. (Nat Commun 2023) benchmarked Spectronaut and DIA-NN on synthetic phosphopeptides and, "unlike most previous studies defining class-I sites based on a non-discriminant confidence score cutoff of 0.75", derived tool-specific cutoffs: Spectronaut 0.75 (regular) / 0.99 (stringent) vs DIA-NN `PTM.Site.Confidence` **0.01 / 0.51** ([Lou 2023](https://www.nature.com/articles/s41467-022-35740-1)). *sitereport* likewise uses "a threshold of 0.75 on site localization in Spectronaut is equivalent to a threshold of 0.01 in DIA-NN" ([Pham 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11239223/)). Locard-Paulet et al. (JPR 2020) showed across 22 DDA pipelines that localization scores "follow very different score distributions, which can lead to different false localization rates for the same threshold" ([Locard-Paulet 2020](https://pubmed.ncbi.nlm.nih.gov/31975593/)).
2. **Probability ≠ false localization rate.** Ramsbottom et al. (JPR 2022) note that localization scores "have generally been calibrated using synthetic datasets, and their statistical reliability on real datasets is largely unknown", and propose independent FLR estimation with decoy amino acids ([Ramsbottom 2022](https://pubmed.ncbi.nlm.nih.gov/35640880/)). A 0.75 probability does *not* mean a 25% FLR (Bekker-Jensen's synthetic estimate was ~1.5%), but neither is the FLR of a real dataset known.

**alphaPhos:** default `cutoff = classI_cutoff = 0.75`, applied to Spectronaut's per-position probabilities (`EG.PTMLocalizationProbabilities`; identical to Spectronaut's own `PTM.SiteProbability` in 99.75% of cells — benchmark §9.1). ✔ Convention for Spectronaut. For the DIA-NN arm the same 0.75 is applied to DIA-NN's per-site `Site.Occupancy.Probabilities`. Note the version dependence of the literature's mapping: Lou 2023 (DIA-NN 1.8) and Pham 2024 map Spectronaut 0.75 to a DIA-NN "site confidence" cutoff of 0.01 — a *score* on that version's scale, not a per-site probability. DIA-NN 2.x reports genuine per-site posterior probabilities (`Site.Occupancy.Probabilities`, and per residue in `site_report.parquet`) and its own site matrices use **0.90 / 0.99** on exactly that quantity (benchmark §10). alphaPhos's 0.75 is therefore the same *kind* of threshold as DIA-NN's, just at the Class I level rather than DIA-NN's stricter 0.9; document which you used.

### 1.3 Localization filtering happens per precursor-run, before aggregation

This is the point that our benchmark had to rediscover empirically, but it is the literature's convention. The Perseus plugin's localization cutoff "will kick out all peptides with localization probability below the set cutoff (e.g. 0.75 or 0.99)" — and because the plugin consumes the *long* Spectronaut report (one row per precursor × run, each with its own `EG.PTMLocalizationProbabilities`), the filter acts on precursor-run rows ([plugin README](https://github.com/AlexHgO/Perseus_Plugin_Peptide_Collapse)). Spectronaut's own PTM site report behaves the same way (benchmark §9.2: only precursors that are individually Class I in a run enter that run's sum). Field papers describe exactly this: Skowronek et al. (MCP 2022) collapsed "using default settings and a localization cutoff of 0.75 (class I sites)" and, where several peptides mapped to one site, "the intensities were summed up" ([Skowronek 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9465115/)).

**alphaPhos:** until 0.22 it summed *all* precursors covering a site and applied Class I on the max — inflating 22% of site-run cells by ~+0.12 log2 (§9.2). Since 0.23, `precursor_loc_gate=True` reproduces the convention (only Class I precursors are aggregated when any exists) → r 0.998, +0.005 log2, 97.5% of cells within 0.1 log2 of Spectronaut. ✔ Convention (now).

### 1.4 Aggregation: sum in linear space

Summing is universal: MaxQuant's `Phospho (STY)Sites` table reports per-multiplicity intensity columns summed over the contributing evidences; the Perseus plugin offers "simple summation" or "linear regression extrapolation for missing values between collapsed peptides"; *sitereport* describes the status quo as "site quantification is currently done by summing up associated intensities, ignoring missing values and not taking into account differences such as ionization efficiency or fragmentation of observed peptides" and adds `max` and MaxLFQ (via the *iq* package) as alternatives ([Pham 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11239223/)). Spectronaut's manual describes its consolidation as "linear modelling based" — impute missing precursor values from other runs, then sum (see synthesis doc §2.3).

**alphaPhos:** `aggregation_method="sum"` (default) and `"consolidate"` (a port of the plugin's linear-regression extrapolation). The benchmark (§3.3, §9.3) shows `sum` tracks Spectronaut best (r 0.998), `consolidate` loses 23% of sites and adds a small positive bias, `median` introduces a −log2(N) attenuation and is kept only for legacy comparisons. ✔ Convention. ✚ MaxLFQ-style site quantification is the one aggregation the field now offers that alphaPhos does not (§4).

### 1.5 Multiplicity

The convention (MaxQuant → plugin → Spectronaut → sitereport) counts phospho groups **on the peptide**, capped at 3, and treats `S473_M1` and `S473_M2` as different features. The literature's own criticism is that this fragments biology and complicates stoichiometry: Hogrebe et al. (Nat Commun 2018) note that stoichiometry "for multiply-phosphorylated peptides is more complex and may not accurately reflect values at individual sites" ([Hogrebe 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC5849679/)); PhosR argues that "site-centric [analysis] treat[s] phosphosites on the same protein independently … while protein-centric … los[es] information from individual phosphosites" and sits in between with `phosCollapse` ([Kim 2021, PhosR](https://www.sciencedirect.com/science/article/pii/S221112472100084X)).

A subtlety no paper spells out: **which** phospho groups are counted. Spectronaut's Class I report counts only *localized* (≥ cutoff) groups — a doubly phosphorylated peptide with one ambiguous site is filed under `M1` — whereas MaxQuant, the plugin and alphaPhos count all groups on the peptide (benchmark §9.3; explains 237 of the 553 cells alphaPhos keeps where Spectronaut says `Filtered`).

**alphaPhos:** total count on the precursor, capped at 3. ✔ Convention (MaxQuant/plugin flavour). ⚠ Differs from Spectronaut's 0.75 report; document.

### 1.6 Protein groups, isoforms, repeat peptides

The convention is to assign the site to the group's leading (first) accession and to compute the position from `PEP.PeptidePosition` for that protein. The literature is explicit that this is where sites get lost or mis-assigned: *sitereport* enumerates five scenarios (single protein; peptide repeated within a protein; doubly phosphorylated; ≥3; multiple isoforms) and notes that the Perseus plugin "misses several phosphosites due to repetition of peptides in the protein sequence (scenario 2) and multiple isoforms (scenario 5)" ([Pham 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11239223/)). Standard practice "either collapse[s] proteins with shared peptides into protein groups or ignore[s] shared peptides altogether" ([PAQu, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC13131628/)).

**alphaPhos:** first accession of the group (contaminant-tagged accession preferred so the tag survives), first mapping position. Positions verified exact against Spectronaut's `EG.ProteinPTMLocations` for 100.000% of 101,636 precursors (§9.1). Same limitation as the plugin for repeat peptides (175 precursors / 211 sites on the EGF set, 0.17%). ✔ Convention, with the field's known blind spot. Note the benchmark also showed that Spectronaut's *own* leading-protein choice for shared paralog peptides differs between its TSV and parquet exports (288 precursors) — the ambiguity is upstream of any collapse tool.

### 1.7 What happens to cells that fail the filter

Here the literature is silent, and tools differ silently. Spectronaut's report writes `Filtered` per cell (strict per-run). MaxQuant's sites table keeps the site if *any* evidence localizes it (`Localization prob` is the best evidence) and reports intensities from all evidences — effectively any-run. The Perseus plugin filters rows below the cutoff, i.e. per-run. Nobody discusses the consequence: strict per-run masking creates missingness that is *correlated with intensity* (low-abundance runs localize worse), which propagates into imputation and DE; permissive any-run masking keeps mislocalized cells.

**alphaPhos** exposes both ends (`per_run`, `global_max`) and defaults to a middle path, `condition`: keep a run's cell if ≥ 50% of that condition's replicates are Class I, else mask it. This has no direct precedent in the collapse literature; it is analogous in spirit to the "quantified in ≥ X% of replicates in at least one condition" completeness filters used downstream (e.g. PhosR), but applied to localization. It recovers ~2% of cells relative to `per_run` on the EGF set at r 0.998 and log2FC r 0.98 vs Spectronaut (§9.2). ⚠ **Deviation — defensible, but it is alphaPhos's choice, not the field's, and must be named in a methods section** (see §5).

### 1.8 Spectronaut over-export (`top_n_attribution`)

Spectronaut's long report emits one row per candidate localization for ambiguous peptides, each carrying the full precursor intensity. A naive collapse attributes the full quant to every candidate site. No paper describes this because the plugin and Spectronaut's own report avoid it implicitly through their per-row localization filter. alphaPhos handles it explicitly (`filter_to_top_n_positions`, validated r 0.98 vs the native report; without it r drops to 0.92 with +0.42 log2 on the current export, §9.3). ⚠ alphaPhos-specific mechanism; describe it.

### 1.9 Adjacent conventions (outside collapse proper)

- **Protein-level confounding.** MSstatsPTM (Kohler et al., MCP 2023) fits separate models for modified and unmodified peptides because of "confounding between changes in the abundance of PTM and the overall changes in the protein abundance" ([Kohler 2023](https://www.mcponline.org/article/S1535-9476(22)00285-7/fulltext)). alphaPhos offers `proteome.phospho_over_proteome` for the same purpose.
- **Contaminants.** Spectronaut/MaxQuant search with a contaminant FASTA; alphaPhos additionally drops PSMs whose whole group is a contaminant at read time (benchmark §3.6).
- **Match-between-runs and positional isomers.** In DDA, MBR can transfer an ID to a co-eluting positional isomer ([review](https://pmc.ncbi.nlm.nih.gov/articles/PMC12188548/)); in DIA the analogue is library-driven localization of isomers with near-identical fragment sets — the per-run probability is the only guard, which is another argument for per-precursor-run gating (§1.3).

---

## 2. Caveats the literature flags — and their status in alphaPhos

| Caveat (source) | Consequence | alphaPhos status |
| --- | --- | --- |
| Localization scores are tool-specific; 0.75 is not portable (Lou 2023; Locard-Paulet 2020; Pham 2024) | Thresholds must be read on each tool's scale; the Lou/Pham "SN 0.75 ≈ DIA-NN 0.01" mapping refers to DIA-NN 1.8's site-confidence *score* | ✔ For DIA-NN 2.x alphaPhos filters the per-site posterior probabilities that DIA-NN itself thresholds at 0.90/0.99 in its matrices; validated against DIA-NN's site table (positions 100%, gated-sum oracle r 0.999 — benchmark §10). The stringency difference 0.75 vs 0.90 is a documented choice (§4.1). |
| Probability thresholds are calibrated on synthetic data; real-data FLR unknown (Ramsbottom 2022) | Class I ≠ 1.5% FLR on your data | Inherent to all pipelines; alphaPhos exposes the per-cell probabilities (`layers["localization"]`, `var["mean/min/max_loc_prob"]`) and the Wilson lower bound for cohort-scale filtering. Decoy-residue FLR estimation is not implemented (§4.3). |
| Multiplicity fragments one site into up to three features; stoichiometry unreliable for multiply phosphorylated peptides (Hogrebe 2018; PhosR) | Diluted power, awkward biology, inconsistent counting across tools | Convention followed (total count, cap 3); `collapse_precursors` + `aggregate_to_site_level` offers the peptide-level alternative; difference to Spectronaut's *localized* count documented (§9.3). |
| Summing ignores ionization efficiency and missingness (Pham 2024) | Site intensity dominated by the most intense precursor; missing precursors bias sums downward | `sum` default matches the field and Spectronaut; `consolidate` available; MaxLFQ-style site quant not implemented (§4.2). |
| Shared / isoform / repeat peptides mis- or un-assigned (Pham 2024; PAQu) | Sites attributed to one paralog; second mapping positions dropped | Same behaviour as the plugin; positions exact for the chosen mapping; export-level protein-inference nondeterminism documented (§9.4). |
| Per-run vs any-run localization is an unstated design choice | Missingness structure and mislocalization trade off silently | All three regimes exposed; default is the documented `condition` rule (§1.7). |
| Site-level analysis loses peptidoform context (Hogrebe 2018; PhosFox) | Co-occupancy invisible | `collapse_precursors` keeps precursor granularity; `precursor_to_site_view` bridges to KSEA. |

---

## 3. Is alphaPhos's collapse defensible? — point by point

| Design choice | Field convention | alphaPhos | Verdict |
| --- | --- | --- | --- |
| Site key | protein_site_mult (MaxQuant, plugin, SN, sitereport) | `Protein|Gene|Site|Mmult`, cap 3 | **Convention** |
| Class I threshold | 0.75 (Olsen 2006 → Bekker-Jensen 2020) | 0.75 on Spectronaut per-position probabilities | **Convention** (Spectronaut arm) |
| Where the threshold is applied | per precursor-run row, before summing (plugin, SN) | `precursor_loc_gate=True` (0.23) | **Convention** — validated r 0.998 |
| Aggregation | sum (all tools); linear-model imputation optional | `sum` default; `consolidate` optional | **Convention** |
| Multiplicity | count on peptide, cap 3 | total count on precursor, cap 3 | **Convention** (MaxQuant/plugin); differs from SN Class I report — document |
| Protein assignment | leading accession, first position | first accession (contaminant-preferring), first mapping | **Convention**, shared blind spot for repeat peptides |
| Per-run vs any-run | tools pick one silently | `per_run` / `global_max` / **`condition`** (default) | **Deviation (extension)** — state explicitly with parameters |
| Over-export dedup | implicit in plugin / SN | explicit `top_n_attribution` | alphaPhos-specific — state explicitly |
| Cohort-scale Class I | none (per-cell or any-run) | `wilson` strategy (0.22) | alphaPhos-specific — state explicitly |
| Contaminant removal | search-time FASTA | search-time + read-time group filter | Conservative extension |

**Overall:** with `localization_strategy="per_run"` alphaPhos *is* the conventional pipeline (and reproduces Spectronaut's native report to within 0.1 log2 in 97.5% of cells). The default configuration adds two well-motivated but non-standard layers (`condition`, `top_n_attribution`) on top of the convention. Both are defensible — the first is a principled completeness/rigour compromise, the second corrects an export artefact — provided they are disclosed. What would *not* be defensible is presenting the default as "Spectronaut-equivalent Class I filtering" without qualification.

---

## 4. Gaps worth closing

1. **DIA-NN localization threshold.** Resolved for DIA-NN 2.x (benchmark §10): alphaPhos filters the same per-site posterior probabilities DIA-NN thresholds in its own matrices; positions and gated sums reproduce DIA-NN's site table. Remaining choice: alphaPhos keeps the Class I value 0.75 where DIA-NN's matrices use 0.90 / 0.99 — pass `cutoff=0.9` to match DIA-NN's convention. For DIA-NN 1.8-era reports (`PTM.Site.Confidence` as a score, no `site_report`), the Lou 2023 mapping (0.01 / 0.51) applies and alphaPhos's 0.75 is *not* equivalent.
2. **MaxLFQ-style site quantification.** *sitereport* shows it is feasible (via *iq*) and addresses the "summing ignores ionization efficiency" critique. A candidate `aggregation_method="maxlfq"`; benchmark against the Spectronaut report before recommending.
3. **Independent FLR estimate.** Ramsbottom's decoy-amino-acid approach (alanine/leucine as decoy acceptors) at search time is outside alphaPhos's scope, but a QC panel that reports the *distribution* of per-cell probabilities near the threshold would make the "Class I ≠ 1.5% FLR" caveat visible to users.
4. **Repeat-peptide second mappings** (`multi_mapping="all"`) — Spectronaut emits them; alphaPhos and the plugin do not. 0.17% of precursors here; larger for repetitive proteomes.
5. **Multiplicity as localized count** (`multiplicity="localized"`) as an opt-in to match Spectronaut's Class I report exactly.

---

## 5. Suggested methods-section wording

> Phosphopeptide precursor reports (Spectronaut *n*, PTM localization enabled) were collapsed to phosphosites with alphaPhos *v*. Sites were keyed by protein accession, residue position and phosphorylation multiplicity (1, 2, ≥3; Olsen et al. 2006; Hogrebe et al. 2018). For each site and run, the linear intensities of all precursors whose localization probability for that site was ≥ 0.75 (Class I) were summed; where no precursor reached 0.75 in a run, all precursors were summed and the cell was then subjected to the localization mask. Spectronaut's per-candidate over-export was resolved by retaining, per precursor, only the top-*N* localization candidates (*N* = number of phosphogroups). Class I masking was applied per condition: a site's measurement in a run was retained if at least 50% of that condition's replicates localized the site at ≥ 0.75, otherwise only Class I measurements were kept [*or*, for `per_run`: measurements with localization probability < 0.75 were set to missing, equivalent to Spectronaut's Class I PTM site report]. Sites with no remaining measurement were dropped. PSMs whose protein group consisted entirely of common contaminants (MaxQuant contaminant list) were removed before collapse. Localization probabilities are Spectronaut's and were not recalibrated; a probability of 0.75 corresponded to an estimated 1.5–1.7% false localization rate on synthetic phosphopeptides (Bekker-Jensen et al. 2020) and should not be read as a per-dataset FLR.

---

## 6. Sources

- Olsen JV et al. *Global, in vivo, and site-specific phosphorylation dynamics in signaling networks.* Cell 2006 — [PubMed](https://pubmed.ncbi.nlm.nih.gov/17081983/); Class I definition summarized in [Springer protocol](https://link.springer.com/protocol/10.1007/978-1-4939-3049-4_20)
- Hogrebe A et al. *Benchmarking common quantification strategies for large-scale phosphoproteomics.* Nat Commun 2018 — [PMC5849679](https://pmc.ncbi.nlm.nih.gov/articles/PMC5849679/)
- Hogrebe A. *Perseus Plugin Peptide Collapse* (reference implementation; now retired in favour of Spectronaut's native report) — [GitHub](https://github.com/AlexHgO/Perseus_Plugin_Peptide_Collapse)
- Bekker-Jensen DB et al. *Rapid and site-specific deep phosphoproteome profiling by DIA without the need for spectral libraries.* Nat Commun 2020 — [paper](https://www.nature.com/articles/s41467-020-14609-1); [preprint full text](https://www.biorxiv.org/content/10.1101/657858v1.full)
- Kitata RB et al. *A DIA-based global phosphoproteomics system enables deep profiling.* Nat Commun 2021 — [paper](https://www.nature.com/articles/s41467-021-22759-z)
- Steger M et al. *Time-resolved in vivo ubiquitinome profiling by DIA-MS.* Nat Commun 2021 — [paper](https://www.nature.com/articles/s41467-021-25454-1)
- Skowronek P et al. *Rapid and in-depth coverage of the (phospho-)proteome with deep libraries and optimal window design for dia-PASEF.* MCP 2022 — [PMC9465115](https://pmc.ncbi.nlm.nih.gov/articles/PMC9465115/)
- Lou R et al. *Benchmarking commonly used software suites and analysis workflows for DIA proteomics and phosphoproteomics.* Nat Commun 2023 — [paper](https://www.nature.com/articles/s41467-022-35740-1)
- Pham TV et al. *msproteomics sitereport: reporting DIA-MS phosphoproteomics experiments at site level with ease.* Bioinformatics 2024 — [PMC11239223](https://pmc.ncbi.nlm.nih.gov/articles/PMC11239223/)
- Locard-Paulet M et al. *Comparing 22 popular phosphoproteomics pipelines for peptide identification and site localization.* JPR 2020 — [PubMed](https://pubmed.ncbi.nlm.nih.gov/31975593/)
- Ramsbottom KA et al. *Method for independent estimation of the false localization rate for phosphoproteomics.* JPR 2022 — [PubMed](https://pubmed.ncbi.nlm.nih.gov/35640880/)
- Kohler D et al. *MSstatsPTM: statistical relative quantification of PTMs in bottom-up MS proteomics.* MCP 2023 — [paper](https://www.mcponline.org/article/S1535-9476(22)00285-7/fulltext)
- Kim HJ et al. *PhosR enables processing and functional analysis of phosphoproteomic data.* Cell Rep 2021 — [paper](https://www.sciencedirect.com/science/article/pii/S221112472100084X)
- Demichev V. *DIA-NN discussion #427: assessing PTM localization confidence* — [GitHub](https://github.com/vdemichev/DiaNN/discussions/427)
- *Computational approaches to identify sites of phosphorylation* (review, 2025) — [PMC12188548](https://pmc.ncbi.nlm.nih.gov/articles/PMC12188548/); *PAQu* (isoform abundance) — [PMC13131628](https://pmc.ncbi.nlm.nih.gov/articles/PMC13131628/)
