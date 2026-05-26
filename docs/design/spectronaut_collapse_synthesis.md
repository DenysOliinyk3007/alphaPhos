# Spectronaut PTM collapse — current MS1 pipeline vs. Spectronaut-native MS2, and the alphaPhos synthesis

**Status:** design draft · 2026-05-21
**Update (2026-05-26):** Empirical defaults validated. The open questions in §7 of this doc were resolved by the benchmark in **[../benchmark/spectronaut_native_benchmark.md](../benchmark/spectronaut_native_benchmark.md)** — the package defaults are now `aggregation_method='sum'` + `localization_strategy='condition'` based on cross-version comparison against SN classI on a 6-run EGF dataset.
**Scope:** Foundational design for `alphaphos.io.spectronaut`, `alphaphos.preprocess.collapse`, and `alphaphos.preprocess.classify`.
**Inputs analysed:**
- Spectronaut 20 Manual, `docs/Spectronaut-20-Manual.pdf` (esp. Box 10 p. 55, §Quantity MS Level p. 121, PTM consolidation p. 126, PTM site report Appendix 8 pp. 200–205).
- Canonical current implementation: [Dublin/testscripts/src/PeptideCollapse_v4.py](../../../Dublin/testscripts/src/PeptideCollapse_v4.py) (1577 lines, identical across aggDVP, miniBinders, nanoPhos_env, T2D; uPhosHT has only memory-projection tweaks).
- Canonical current Class I filter: [Dublin/testscripts/src/condition_aware_classI.py](../../../Dublin/testscripts/src/condition_aware_classI.py) (170 lines, identical across all 5 study copies).

---

## 0. TL;DR

The "MS1 vs MS2" framing is partially a misnomer. The Python pipeline is **MS-level agnostic** — it consumes whatever Spectronaut wrote into `EG.TotalQuantity (Settings)`, which reflects Spectronaut's *active* `Quantity MS Level` setting at export time. The substantive differences are not at the quant column but in **how peptides are collapsed to sites** and **how localization confidence is enforced**:

| Layer | Current Python | Spectronaut native |
| --- | --- | --- |
| Quant source | Whatever Spectronaut exported (`EG.TotalQuantity (Settings)`) | Same column, but Spectronaut typically defaults to MS2 |
| Peptide → site collapse | `consolidate` = ratio-based imputation + sum (R `consolidate()` port) — built around MS1 envelope semantics | Sum *or* Linear-model (impute + sum), Bekker-Jensen 2020 |
| Localization | Per-(site, run) cell mask OR dataset-wide max OR per-(site, condition) majority rule (three modes) | Single binary cutoff (default 0.75) on the strongest per-site observation |
| Multiplicity (M1/M2/M3) | Tracked in collapse key (`_M1`/`_M2`/`_M3`), separate site objects per multiplicity | Tracked in `PTM.Multiplicity`, separate site objects per multiplicity (same logic) |
| Output schema | `PTM_Collapse_key` = `{ProteinGroup}~{Gene}_{S\|T\|Y}{abs_pos}_M{1\|2\|3}`, wide log2 matrix | `PTM.ProteinId` + `PTM.SiteAA` + `PTM.SiteLocation` + `PTM.Multiplicity` (tuple), linear quantity |

**Synthesis decision** (proposed): `alphaphos` exposes both collapse algorithms and all three localization-confidence strategies as named, orthogonal choices. The default for new analyses is **MS2 quant + linear-model collapse + condition-aware Class I**, but the current pipeline's exact behavior is reproducible via **MS1 quant + ratio-consolidate + global_max** for byte-equality regression tests against legacy studies. Justification and trade-offs below.

---

## 1. What the current Python pipeline does

### 1.1 Inputs assumed

`PeptideCollapse_v4.required_columns["essential"]`:

```
R.FileName, EG.PrecursorId, EG.TotalQuantity (Settings),
PEP.PeptidePosition, EG.PTMAssayProbability,
PG.Genes, PG.ProteinGroups
```

Optional (improves localization resolution): `EG.PTMLocalizationProbabilities`, `EG.ProteinPTMLocations`, `PEP.StrippedSequence`, `PG.UniProtIds`.

**Critical observation:** `EG.TotalQuantity (Settings)` is whatever Spectronaut was *configured* to export. The Python code never inspects whether that's MS1 or MS2 — it operates on the column as opaque intensities. So when we say "the current pipeline is MS1," what we really mean is **"the typical Spectronaut search method in this lab is configured with Quantity MS Level = MS1, so this column is MS1."** Confirming the lab default before migration is a one-line check.

### 1.2 Pipeline shape

1. **Load & validate** ([PeptideCollapse_v4.py:101–120](../../../Dublin/testscripts/src/PeptideCollapse_v4.py#L101-L120)). Hashes first 100 precursors per run to flag accidentally-duplicated raw files.
2. **Parse modifications from `EG.PrecursorId`** ([_extract_sequence_modifications:580–622](../../../Dublin/testscripts/src/PeptideCollapse_v4.py#L580-L622)). Custom regex strips Spectronaut's `[Phospho (STY)]` brackets, computes phospho positions within the peptide, counts modifications.
3. **Per-site explosion** ([_create_site_level_collapse:809–844](../../../Dublin/testscripts/src/PeptideCollapse_v4.py#L809-L844)). One row per (peptide, phospho-position) pair.
4. **Per-site localization probability** ([:838–882](../../../Dublin/testscripts/src/PeptideCollapse_v4.py#L838-L882)). Parses `EG.PTMLocalizationProbabilities` into `{position: prob}`; falls back to the joint `EG.PTMAssayProbability` if the per-site string is missing (logged as `per_site_localization_fallback_pct`).
5. **Pivot quants** ([:887–898](../../../Dublin/testscripts/src/PeptideCollapse_v4.py#L887-L898)) to `(PTM_group, PTM_0_pos_val) × R.FileName` using `aggfunc="sum"` (sums multiple precursor rows mapping to the same site/run, i.e. charge states / mis-cleaves).
6. **Build `PTM_Collapse_key`** ([_create_collapse_key:1181–1191](../../../Dublin/testscripts/src/PeptideCollapse_v4.py#L1181-L1191)): `{ProteinGroup}~{Gene}_{S|T|Y}{abs_pos}_M{1|2|3}`. The `M{1|2|3}` is multiplicity (3+ phosphos clamp to M3).
7. **Aggregate precursors → site quant** ([:970–987](../../../Dublin/testscripts/src/PeptideCollapse_v4.py#L970-L987)). Four modes:
   - `sum` (additive aggregation)
   - `mean`
   - `median`
   - `consolidate` — the algorithmically distinctive one. See §1.3.
8. **Per-run localization mask** ([:999–1023](../../../Dublin/testscripts/src/PeptideCollapse_v4.py#L999-L1023)) — only in `localization_strategy="per_run"` mode. For each (site, run) cell, mask quant → NaN if the per-run loc prob < cutoff. NaN loc is treated as below cutoff. Applied in linear space before `log2`.
9. **Log2 transform + noise floor filter** ([:1026–1042](../../../Dublin/testscripts/src/PeptideCollapse_v4.py#L1026-L1042)). Replaces log2 values of 0 or 1 with NaN (noise floor heuristic, optional).
10. **Site filtering** ([:1060–1082](../../../Dublin/testscripts/src/PeptideCollapse_v4.py#L1060-L1082)):
    - `per_run` mode: drop sites whose entire row is NaN after masking.
    - `global_max` mode: filter on dataset-wide max loc prob ≥ cutoff (legacy Hogrebe-plugin convention).

### 1.3 The `consolidate` algorithm — what makes it "MS1-flavoured"

[_consolidate:1215–1313](../../../Dublin/testscripts/src/PeptideCollapse_v4.py#L1215-L1313) is a faithful port of an R `consolidate()` function (the docstring labels it "matching the OG R consolidate()"). For a matrix of precursors-of-the-same-site × samples in **linear intensity space**:

1. Drop precursor rows with ≤1 non-NaN value.
2. Sort rows by ascending median intensity.
3. **Ratio-based imputation loop:** while any sample still has NaN in a row that has some signal, for each row with NaN values, estimate them as `median(this_row / other_row) × other_row[nan_positions]` across all other rows; take the median across all such predictions. If a full iteration makes no progress, drop the worst-NaN row and retry.
4. Sum surviving rows per sample.

This algorithm assumes that **co-eluting precursor isotopes scale proportionally across samples** — a defensible assumption for MS1 envelope intensities of co-eluting precursors of the same chemical species, less so for MS2 fragment-ion intensities (which are subject to per-fragment selection, dynamic range, and interference filtering effects that don't scale linearly precursor-wide). This is the sense in which the current pipeline is structurally MS1-oriented even though the column itself is opaque.

### 1.4 `condition_aware_classI` — the middle-ground filter

[condition_aware_classI.py:55–170](../../../Dublin/testscripts/src/condition_aware_classI.py#L55-L170) operates **downstream** of `PeptideCollapse_v4` run in `global_max` mode with `cutoff=0` (i.e. the matrix has not yet been masked). For each (site, condition):

- `N` = replicates in condition
- `N_I` = replicates where the site's per-run loc prob ≥ `classI_cutoff` (default 0.75)
- If `N_I / N ≥ condition_threshold` (default 0.50, "majority rule"): keep *all* quants for this (site, condition), including the non-Class-I cells. Interpretation: when the majority of a condition's replicates localize the site confidently, the minority observations are treated as the same site at lower confidence.
- Else: mask the non-Class-I cells in this condition to NaN.

This is a biologically motivated compromise between strict per-cell masking (`per_run`) and permissive dataset-wide filtering (`global_max`).

---

## 2. What Spectronaut does natively

### 2.1 Quantification level — a single config knob (Manual p. 121)

> "**Quantity MS Level** — Choose which MS level you want to use to perform quantification: MS1 or MS2. MS2 level = sum of the quantities of the number of fragment ions per precursor, as specified in the spectral library."

Defaults to MS2 in BGS Factory Settings (implied throughout Appendix 5 XIC plots, pp. 167–169). MS1 means "sum of the integrated isotope envelope peaks for the precursor"; MS2 means "sum of the integrated fragment-ion XICs for the assay's library fragments." Both are AUC by default (`Quantity Type`, p. 121).

Interference Correction (p. 123, Bilbao 2015) operates on both levels — Spectronaut excludes fragments/precursors flagged as outliers across runs from the quant.

### 2.2 PTM site localization — Box 10 (p. 55)

Spectronaut uses an in-house DIA-specific algorithm (Bekker-Jensen, Bernhardt et al. *Nat Commun* 11:787, 2020 — cited Manual p. 110), **not** PTMScore, Ascore, or phosphoRS. Inputs scored:

- Full isotopic pattern of each fragment ion (not just monoisotopic).
- Short elution chromatograms per fragment, correlated to the targeted peak shape (interference removal).
- Fragment mass accuracy.
- Fragment intensity.

The output is `EG.PTMAssayProbability` (per assay), `EG.PTMLocalizationProbabilities` (full per-position string), `PTM.SiteProbability` (max per site per run/sample).

**No Class I/II/III terminology in the manual.** The only "Class I/II" references are MHC immunopeptidomics (p. 10). The single documented threshold is **probability ≥ 0.75 (default)** applied in three places (pp. 124, 157, plus directDIA settings). Spectronaut is binary: localized vs. not.

### 2.3 Peptide → site collapse (PTM consolidation, Manual p. 126)

Verbatim:

> "PTM consolidation specifies how to derive quantity from set of parent peptides carrying a particular modification on a given modification site:
> - **Sum** would summarize quantities of all parent peptides that are carrying particular modification on a given modification site.
> - **Linear model** would firstly impute missing values for each parent peptide based on quantities reported in other runs. Afterwards, it would summarize quantities of all parent peptides as above."

So exactly **two** collapse modes: `Sum` and `Linear-model` (impute → sum). No median, no `consolidate`-style ratio imputation, no MaxLFQ at the site level (MaxLFQ is protein-level only, p. 121).

### 2.4 Multiplicity (Manual pp. 55, 124)

Identical to current Python:

> "If the parent peptides carry more modifications of the same type, a separate collapse is performed according to the modification multiplicity … M1, M2, M3."

`PTM.Multiplicity` column distinguishes M1/M2/M3 site objects. Differential testing is run per-multiplicity.

### 2.5 Output schema (Appendix 8, pp. 200–205)

The site is identified by the tuple `(PTM.ProteinId, PTM.SiteAA, PTM.SiteLocation, PTM.Multiplicity)` — there is no single concatenated "site name" column. Downstream tooling (incl. the current pipeline) concatenates as `{ProteinId}_{SiteAA}{SiteLocation}_M{Multiplicity}`.

Key columns: `PTM.Quantity`, `PTM.QuantityPerProtein` (input-normalized), `PTM.NrOfCollapsedPeptides`, `PTM.Group` (parent peptides), `PTM.SiteProbability` (the localization filter target), `PTM.CollapseKey`.

---

## 3. Detailed MS1 vs MS2 comparison

The Spectronaut manual contains **no explicit prose comparison** of MS1 vs MS2 for PTM quantification (Section 3 of the agent's findings). The comparison below is from primary literature (Bekker-Jensen 2020, Bruderer 2015/2017) and from inspection of where each approach breaks.

| Dimension | MS1 (precursor envelope) | MS2 (fragment-ion sum) |
| --- | --- | --- |
| **Signal definition** | AUC of isotope peaks of the precursor | Sum of AUCs of assay's library fragments (typically top ~6 b/y) |
| **Selectivity / interference** | Lower. Co-eluting peptides with overlapping isotope envelopes contaminate the signal; DIA window size dictates this. | Higher. Fragment ions are more specific; Bilbao 2015 interference correction filters bad fragments. Bekker-Jensen 2020 leverages the per-fragment chromatograms for both localization and quantification. |
| **Positional isomer discrimination** | None at the quantification step. Two peptides differing only in phospho-position have *identical* MS1 envelopes — MS1 quant cannot tell them apart and the localization probability is the only handle. | Some, via site-determining ions (b/y ions spanning the modified residue). MS2 quant for a peptide implicitly weights fragments that survived the localization scoring — they tend to be the localizing ones. |
| **Sensitivity per identification** | Higher. One MS1 peak per peptide; integrated over a single XIC. | Distributed across many fragment ions. Sum-of-top-6 is often comparable but is more sensitive to dropout of individual fragments. |
| **Dynamic range** | Wider (MS1 has the full precursor signal). | Narrower per fragment; sum-of-fragments recovers most of it. |
| **Missing-value pattern** | Smoother — MS1 envelopes are more often detectable. The `consolidate` ratio-imputation assumes this. | More fragment-level missingness; Spectronaut's Linear-model imputation operates at the parent-peptide level (downstream of fragment summing), not at the fragment level. |
| **Default in Spectronaut** | Available, opt-in. | Default. |
| **Default in the current Python pipeline** | Implicit (whatever lab's Spectronaut method exports). | Not used. |
| **PSU-of-truth for site quantification** | Less direct — relies on the precursor signal being a clean proxy for the site. | More direct in the sense that the fragments are the same evidence the localization scoring used. |
| **Behavior under co-isolation** | Inflates quant (other peptides in the isolation window contribute to the precursor envelope). | More robust — co-isolated species fragment to different ions, only contaminate fragments matching the assay m/z. |

**Net practical implication for the lab:**
- Studies acquired with a wide DIA window (e.g. 25 m/z) and long gradients: MS1 quant is at higher risk of co-isolation; MS2 should help.
- Studies acquired with narrow DIA windows (e.g. 5 m/z) and modern instruments: both work; MS1 has marginal sensitivity advantage but MS2 has selectivity advantage.
- For sites with **ambiguous localization** (multiple S/T/Y in the same peptide near the modification), MS2 quantification is structurally more honest because the quant signal weights site-determining fragments. MS1 quant is the same regardless of which S/T/Y is phosphorylated within the peptide — the localization probability column is *all* the localization evidence.

---

## 4. Class I / classification: three modes vs. one

| Mode | Decision unit | Mask granularity | Source |
| --- | --- | --- | --- |
| **Spectronaut binary** (PTM Localization Filter, default cutoff 0.75) | Site (per-run max) | Sites below cutoff dropped entirely | Manual pp. 124, 157 |
| **`global_max`** (current Python, legacy) | Site (dataset-wide max) | Sites below cutoff dropped entirely; rough parity with Spectronaut binary at dataset level | [PeptideCollapse_v4.py:1080–1082](../../../Dublin/testscripts/src/PeptideCollapse_v4.py#L1080-L1082) |
| **`per_run`** (current Python, "Reviewer 2 bulletproof") | Cell `(site, run)` | Quant → NaN per cell, no dataset-wide aggregation | [PeptideCollapse_v4.py:999–1023](../../../Dublin/testscripts/src/PeptideCollapse_v4.py#L999-L1023) |
| **`condition_aware`** (middle) | Cell `(site, run)`, conditioned on per-condition Class I fraction | If majority of a condition's replicates is Class I, keep all; else mask non-Class-I in that condition | [condition_aware_classI.py](../../../Dublin/testscripts/src/condition_aware_classI.py) |

**Spectronaut ≈ `global_max`** in spirit (single binary cutoff on the strongest per-site evidence), though Spectronaut applies it earlier in its pipeline. `per_run` is **stricter than anything Spectronaut offers**; `condition_aware` is **smarter than anything Spectronaut offers** because Spectronaut has no notion of conditions during collapse.

The current pipeline's three-mode design is a real capability over Spectronaut's binary cutoff, and should be preserved.

---

## 5. The alphaPhos synthesis

### 5.1 Design principles

1. **Quant level is an export-time setting, not a pipeline setting.** `alphaphos.io.spectronaut.read_psm()` reads whatever Spectronaut wrote and records the active level in `PhosphoExperiment.uns["spectronaut_quantity_ms_level"]` (auto-detected from `EG.PrecursorQuantityMS1`/`EG.PrecursorQuantityMS2` columns if present, else inferred from settings metadata, else falls back to "unknown" with a warning).
2. **Collapse algorithm is orthogonal to quant level.** A user can run MS1 quant + linear-model collapse, MS2 quant + `consolidate`, etc. Some combinations are bad ideas (see §5.5), but we don't prevent them — we warn.
3. **Localization confidence strategy is a separate, named choice.** Three strategies exposed; Spectronaut-compatible (`global_max`) is one of them. Default is `condition_aware` because it dominates `global_max` for typical experiment designs and dominates `per_run` for typical localization-prob distributions.
4. **All choices are recorded in `PhosphoExperiment.uns["processing_history"]`** as a list of operations with parameters. A `PhosphoExperiment` is self-describing about how it got to its current state.
5. **Backward-compatibility path is exact, not approximate.** `MS1 + consolidate + global_max + 0.75 cutoff` must reproduce `PeptideCollapse_v4` byte-for-byte (within float tolerance) on Dublin's test data. This is the regression-validation contract that lets us migrate existing studies without changing published numbers.

### 5.2 Module shape

```python
# alphaphos.io.spectronaut
read_psm(path, *, quantity_level="auto", ...) -> PhosphoExperiment
# Reads Spectronaut Normal-report long format. quantity_level:
#   "auto" -> infer from columns / settings
#   "MS1"  -> require EG.TotalQuantity (Settings) to be MS1
#   "MS2"  -> require it to be MS2
# Populates pe.uns["spectronaut_version"], pe.uns["spectronaut_quantity_ms_level"].

read_ptm_site_report(path, ...) -> PhosphoExperiment
# Alternative: consume Spectronaut's already-collapsed PTM site report.
# Bypasses alphaphos.preprocess.collapse entirely (site_data already exists).
```

```python
# alphaphos.preprocess.collapse
collapse_to_sites(
    pe: PhosphoExperiment,
    *,
    algorithm: Literal["consolidate", "linear_model", "sum", "median", "mean"] = "linear_model",
    aggregate_charge_states: Literal["sum"] = "sum",  # matches both Spectronaut and current code
    multiplicity_split: bool = True,                  # M1/M2/M3 as separate site objects
    cutoff: float = 0.75,
    classify_strategy: Literal["global_max", "per_run", "condition_aware"] = "condition_aware",
    classify_condition_threshold: float = 0.50,       # only used when condition_aware
    classify_obs_column: str | None = None,           # condition column in pe.obs for condition_aware
    log_transform: bool = True,
    noise_floor: bool = True,
) -> PhosphoExperiment
```

Where:
- `algorithm="consolidate"` reproduces the current Python `_consolidate`.
- `algorithm="linear_model"` reproduces Spectronaut's PTM consolidation Linear-model (impute via cross-run linear model + sum). This is the algorithm we need to implement from scratch — Spectronaut does it internally and the Manual only describes it at the prose level (p. 126). First-pass implementation: per-precursor, regress quant against the rest of the runs (or against the mean of the rest), use the regression to impute, then sum.
- `algorithm="sum"` is the Spectronaut "Sum" mode — simplest, matches Spectronaut default when Linear-model is off.

### 5.3 The collapse-key schema

Keep the current `PTM_Collapse_key` format `{ProteinGroup}~{Gene}_{S|T|Y}{abs_pos}_M{1|2|3}` as the canonical site identifier — it preserves the regression-test contract. In `PhosphoExperiment.site.var`, expose the components as first-class columns:

```
site.var:
    protein_group         str   # PG.ProteinGroups
    gene                  str   # PG.Genes (first group)
    site_aa               cat   # S | T | Y
    site_position_abs     int   # absolute position in protein
    multiplicity          cat   # M1 | M2 | M3
    collapse_key          str   # the legacy concatenated key, for regression alignment
    spectronaut_ptm_protein_id    str   # PTM.ProteinId, if from PTM site report
    spectronaut_ptm_site_location int   # PTM.SiteLocation
    flanking_window       str   # ±N residues, for kinase motif analyses
    n_collapsed_peptides  int   # how many parent peptides contributed
    parent_peptides       list[str]  # for full traceability
```

### 5.4 Layered transforms in `PhosphoExperiment.site.layers`

```
layers:
    intensity_raw           # right out of io (linear)
    intensity_collapsed     # after collapse(), linear
    intensity_log2          # after log2 transform
    intensity_normalized    # after normalize()
    intensity_imputed       # after impute()
    localization_per_run    # (site, run) matrix of per-run loc probs — needed for any later re-classification
    classify_mask           # boolean (site, run) — which cells the active classify_strategy kept
```

This way the user can re-classify with a different strategy without re-collapsing — `localization_per_run` is persisted and `reclassify(pe, strategy=...)` becomes a cheap operation on the existing object.

### 5.5 Recommendation matrix

| Scenario | Recommended config |
| --- | --- |
| New acquisition, modern Spectronaut (>=18), standard DIA | `quantity_level="MS2", algorithm="linear_model", classify_strategy="condition_aware"` |
| Single-cell / very low input, where MS1 sensitivity matters | `quantity_level="MS1", algorithm="consolidate", classify_strategy="condition_aware"` |
| Reproducing pre-2026 published study from this lab | `quantity_level="MS1", algorithm="consolidate", classify_strategy="global_max", cutoff=0.75` (legacy mode) |
| Cross-tool benchmarking against Spectronaut PTM site report | Skip `collapse_to_sites` entirely — use `read_ptm_site_report` |
| Plate-scale screens (308 / 384 wells) | Same as new-acquisition + memory-optimized path (fold uPhosHT projection tweaks into `_create_site_level_collapse` by default) |

**Bad combinations to warn on:**
- `quantity_level="MS2" + algorithm="consolidate"`: ratio-imputation assumes co-eluting precursor envelopes scale proportionally — not a clean assumption for fragment sums. Warn.
- `classify_strategy="condition_aware"` without `classify_obs_column` set: error (no condition information).

---

## 6. Implementation order (to slot into the migration roadmap)

1. **`alphaphos.io.spectronaut.read_psm`** with `PhosphoExperiment` output, populating `obs` (per-run metadata from `R.*` columns), `var` (per-precursor metadata), `X` (linear quant from `EG.TotalQuantity (Settings)`).
2. **Port `_extract_sequence_modifications` + `_parse_localization_probabilities`** as `alphaphos.preprocess._parsing` (pure functions, easy to unit-test).
3. **Port `_create_site_level_collapse`** as `alphaphos.preprocess.collapse.collapse_to_sites` with `algorithm="consolidate"` and `classify_strategy="global_max"` as default → first golden-snapshot regression target against Dublin.
4. **Add `classify_strategy="per_run"`** (port the existing per-run mask logic) → second regression target.
5. **Port `condition_aware_classI`** as `classify_strategy="condition_aware"` → third regression target.
6. **Implement `algorithm="linear_model"`** as the Spectronaut-equivalent path. This is new code — needs its own unit tests against a tiny synthetic example, plus a side-by-side comparison against a Spectronaut PTM site report on Dublin's data to verify the imputation matches.
7. **Implement `algorithm="sum"`** (trivial — just `groupby.sum`) — included for completeness and as the simplest Spectronaut-equivalent.
8. **`read_ptm_site_report`** for users who want to skip our collapse entirely.

### 6.1 Validation contract

The regression test in [tests/regression/test_dublin_collapse.py](../../../tests/regression/) must assert:

```python
def test_legacy_mode_byte_equality(dublin_input, dublin_legacy_output):
    pe = alphaphos.io.spectronaut.read_psm(dublin_input, quantity_level="MS1")
    pe = alphaphos.preprocess.collapse.collapse_to_sites(
        pe,
        algorithm="consolidate",
        classify_strategy="global_max",
        cutoff=0.75,
        multiplicity_split=True,
    )
    matrix_new = pe.site.layers["intensity_log2"]
    matrix_old = pd.read_parquet(dublin_legacy_output)
    pd.testing.assert_frame_equal(
        matrix_new, matrix_old, atol=1e-6, check_dtype=False,
    )
```

If this passes, **migrating Dublin / T2D / aggDVP / VEGF / CytoDerm / miniBinders to the package cannot change their published numbers** — only add new capabilities on top.

---

## 7. Open questions for the user

1. **Is the lab's standard Spectronaut method actually configured with `Quantity MS Level = MS1`?** This whole MS1-vs-MS2 framing depends on this. Worth running `grep -l "MS1Quantity" Dublin/data/*.txt` (or similar) to confirm from a real export.
2. **Linear-model implementation source.** Spectronaut's exact linear-model imputation is not documented in the manual beyond "imputes missing values for each parent peptide based on quantities reported in other runs." Options for our re-implementation:
   - (a) Reverse-engineer by running both on the same data and minimizing residual — empirical match.
   - (b) Implement a defensible linear-model imputation (e.g. per-precursor regression on a low-rank row-mean basis) and document it as *our* linear-model, not Spectronaut's. Reproducible, but won't match Spectronaut byte-for-byte.
   - Recommendation: (b) for v1, with a separate `algorithm="spectronaut_linear_model"` reserved for if/when we work out (a).
3. **Where does the lab's preferred default sit?** Current studies use `global_max` (legacy) or `per_run` (Reviewer 2). For new studies, do we default to `condition_aware`, or hold that back as opt-in until it's been stress-tested on a fresh dataset?
4. **Charge-state aggregation.** Current code sums precursor rows mapping to the same (site, run) — including different charge states of the same sequence. Spectronaut does the same (within `PTM.NrOfCollapsedPeptides`). Worth confirming this is intended (alternative: MaxLFQ-style top-N selection at the charge-state level before site collapse, more robust to one bad charge state).
