# `alphaphos.preprocess.collapse_precursors`

Precursor-level collapse — sibling of [`collapse_sites`](collapse.md).

`collapse_precursors(psm_df, ...)` quantifies each identified **precursor**
(peptide sequence + charge + modifications) as one feature per sample, without
residue attribution and without localization-probability masking. Trades
site-level resolution for a more complete, less-punctured quantification
matrix.

## When to use precursor-level over site-level

Use **precursor-level** when the primary question is *detection / differential*
and one or more of these holds:

- The experiment is a **modest-effect study** where every percent of matrix
  completeness matters (dose-response, subtle time-courses).
- The features of scientific interest are **low-abundance receptors / kinases**
  whose per-run localization probability is borderline — precisely the
  population most likely to be culled by site-level masking.
- The **Spectronaut assay probability** disagrees with the localization
  probability (loc high, assay low), which signals that the localization
  metric itself is noisy on this dataset.

Use **site-level** when:

- You need residue resolution for KSEA / pathway / kinase-library motif
  analyses.
- Localization is reliable across your abundance range (well-behaved
  high-abundance targets).

A common workflow keeps **both** in parallel: precursor-level as the primary
detection / differential object, site-level for the downstream analyses that
genuinely require residue attribution. See [`precursor_to_site_view`](#bridging-back-to-site-level-analyses)
below.

## Scientific basis

Some phospho DIA workflows (Krug 2019 PTM-SEA; Meier 2020 diaPASEF
phosphoproteomics guidance) advocate precursor-level readout when the
localization metric is noisy on low-abundance features. In practice, the
argument is empirical: site-level masking gates quantification on a
probability that is itself unreliable in some datasets, and it does so with a
**per-run** rule that trips on borderline features precisely where you'd
want power. The concrete motivator for this module is documented in
`D:/Projects/miniBinders/CHO_TAB2_H2F/docs/peptide_level_collapse_rationale.md`:
TRKA activation loop pY680 detected in 56/60 runs at 99.9% localization on
Y680, still culled from the site-level analysis because per-run loc was
inconsistent.

Precursor collapse skips those two masking steps and keeps the precursor
as-measured. Localization info is still extracted and stored on `.var`, but
purely as annotation — never for filtering.

## Signature

```python
ap.collapse_precursors(
    data: pd.DataFrame,
    *,
    condition_df: pd.DataFrame | None = None,
    advanced: dict[str, Any] | None = None,
    verbose: bool = False,
) -> ad.AnnData
```

## Input

- **`data`**: PSM-level DataFrame from [`read_spectronaut`](../io/spectronaut.md).
  Currently only Spectronaut is supported (DIA-NN / FragPipe adaptation in
  follow-up). See the [io schema](../io/index.md#required-spectronaut) for
  required columns.
- **`condition_df`**: Sample metadata (`sample` + `condition` columns at
  minimum; extras joined into `.obs`).

## Parameters

| Parameter | Default | What it does | Alternatives |
| --- | --- | --- | --- |
| `data` | *required* | PSM DataFrame from `read_spectronaut`. | -- |
| `condition_df` | `None` | Sample metadata. | Optional but recommended. |
| `advanced` | `None` | Dict of overrides for `DEFAULT_PRECURSOR_COLLAPSE_SETTINGS`. Unknown keys raise. | See below. |
| `verbose` | `False` | Log INFO-level stage progress to stderr. | `True` for pipeline debugging. |

### `advanced` keys (see `DEFAULT_PRECURSOR_COLLAPSE_SETTINGS`)

| Key | Default | What it does | Alternatives |
| --- | --- | --- | --- |
| `search_engine` | `"SN"` | Which PSM schema to expect. | Only Spectronaut for v1. |
| `quantification_level` | `"MS2"` | Which quant column to consume. | `"MS1"`, `"auto"`. Falls back per engine if unavailable. |
| `aggregation_method` | `"sum"` | How to combine multiple PSM rows for the same (precursor, sample) into one intensity. | `"mean"`, `"median"`. `"sum"` is the conventional choice; the other two are useful when duplicate rows represent redundant measurements rather than distinct fragments. |
| `noise_floor_filter` | `True` | Drop log2 values in `{0, 1}` (Spectronaut noise-floor convention: linear intensity 1 or 2 = "signal detected but at the very bottom of the dynamic range"). | `False` to keep them. |
| `drop_all_nan` | `True` | Drop precursors with no observed values in any sample after log2 / noise-floor. | `False` to keep NaN rows. |
| `phospho_only` | `True` | Drop precursors with no `[Phospho (STY)]` marker. | `False` to keep unphosphorylated precursors (rare; use only when explicitly analysing the non-phospho background). |
| `annotate_localization` | `True` | Parse `EG.PTMLocalizationProbabilities` and store best-position + best-probability in `.var`. Never used for filtering. | `False` to skip parsing (loc columns present but NaN). |

Use `ap.resolve_precursor_settings(advanced)` to preview the fully-resolved
settings dict.

## Output

An `anndata.AnnData` with shape `(n_samples, n_precursors)`:

- `.X` == `.layers["intensity_log2"]` — log2 intensity per precursor per sample.
- `.layers["localization"]` — per-cell best localization probability. **Never
  used for masking** — annotation only.
- `.var.index` = **alphaPhos precursor key** `Protein|Gene|Peptide|Charge|Mods`.
  Modifications field uses `+` as internal separator (safer than `|`), or
  literal `none` when the precursor is unmodified.
- `.var` columns:

  | Column | Meaning |
  | --- | --- |
  | `protein_group_id` | `PG.ProteinGroups` value (semicolon-joined UniProt accessions). |
  | `gene` | First gene symbol from `PG.Genes`. |
  | `peptide_sequence` | Plain amino acids (no bracket mods). |
  | `charge` | Precursor charge (int). |
  | `mods` | `+`-joined list of bracket contents (e.g. `Phospho (STY)+Phospho (STY)`). |
  | `n_phospho` | Number of `[Phospho (STY)]` markers on the precursor (drives the `M<n>` field of the site key when bridging). |
  | `peptide_start` | 1-indexed absolute position of the peptide's first AA in the parent protein (from `PEP.PeptidePosition`). |
  | `best_localization_prob` | Peak localization probability across all PSM rows for this precursor. NaN when `annotate_localization=False` or unavailable. |
  | `best_localization_pos_peptide` | 1-indexed peptide-local position of the peak. |
  | `best_localization_pos_protein` | Absolute (protein-level) position = `peptide_start + peptide_pos - 1`. |
  | `n_samples_detected` | Count of samples with a finite log2 intensity for this precursor. |

- `.obs` columns: `condition` (from `condition_df`), `phospho_selectivity_pct`
  (fraction of phospho-containing precursors in that sample's raw PSMs), plus
  any extras from `condition_df`.
- `.uns["alphaphos"]` = `version`, resolved `pipeline_params`, `stats`
  (`n_input_rows`, `n_precursors`, `n_samples`, `quantification_column_used`,
  `quantification_level_used`).
- `.uns["source_attrs"]` = `data.attrs` (PSM lineage from the reader).

## Bridging back to site-level analyses

`ap.precursor_to_site_view(adata, *, require_localization=0.75)` returns a
per-precursor DataFrame with a best-guess **alphaPhos site key**
(`Protein|Gene|<AA><absolute_pos>|M<n_phospho>`), so downstream KSEA /
pathway analyses that need residue attribution can consume it.

- Below `require_localization`: `site_key` is `NaN` (the precursor is kept in
  the precursor-level AnnData; only the site annotation is withheld).
- Non-STY residue at the peak-localization position: `site_key` is `NaN`.
- Pass `require_localization=None` to skip the confidence gate.

### Signature

```python
ap.precursor_to_site_view(
    adata: ad.AnnData,
    *,
    require_localization: float | None = 0.75,
    residue_alphabet: tuple[str, ...] = ("S", "T", "Y"),
) -> pd.DataFrame
```

### Return columns

| Column | Type | Meaning |
| --- | --- | --- |
| *index* | str | `precursor_key` (matches `adata.var.index`). |
| `site_key` | str \| NaN | alphaPhos site key `"Protein\|Gene\|S473\|M1"`, or NaN. |
| `site_residue` | str \| NaN | `S`/`T`/`Y`. |
| `site_position_protein` | int \| NaN | 1-indexed absolute position. |
| `best_localization_prob` | float | Verbatim from `.var`. |
| `passes_localization` | bool | Did this precursor clear the gate? |

## Example

```python
import alphaphos as ap
import pandas as pd

psm = ap.read_spectronaut("phospho_report.parquet")
conditions = pd.DataFrame({
    "sample":    ["s1", "s2", "s3", "s4"],
    "condition": ["ctrl", "ctrl", "trt", "trt"],
})

# --- Precursor-level pipeline ---
adata = ap.collapse_precursors(psm, condition_df=conditions)
adata = ap.filter_by_completeness(
    adata, min_valid_frac=2/3, group_column="condition", keep_strategy="each"
)
adata = ap.impute_hybrid(adata)
result = ap.diff_exp_limma(
    adata, condition_column="condition", comparison=("trt", "ctrl"),
)

# --- Bridge for KSEA (still on the precursor-level result) ---
bridge = ap.precursor_to_site_view(adata, require_localization=0.75)
result_with_sites = result.join(bridge[["site_key"]])
ksea = ap.enrichment.kinase_activity(
    result_with_sites.dropna(subset=["site_key"]).set_index("site_key"),
    stat_col="log2fc",
)
```

## Design goals

- **Feature-parity contract with `collapse_sites` downstream.** Returns the
  same AnnData shape / layers / `.obs` schema, so `filter_by_completeness`,
  `impute_hybrid`, `batch_correct_combat`, `diff_exp_limma` all work
  unchanged.
- **Localization is annotation, never masking.** `EG.PTMLocalizationProbabilities`
  is parsed and stored on `.var` for provenance / downstream bridging, but
  it never removes rows or NaN's cells.
- **Minimum-viable settings surface.** No `localization_strategy`, no
  `classI_cutoff`, no `top_n_attribution` — these are the site-level knobs
  that don't apply here.

## Known caveats

- **Spectronaut only in v1.** DIA-NN precursor parsing requires a separate
  parser (`(UniMod:21)` bracket-free encoding); FragPipe emits site-level
  already. Both will land as follow-ups.
- **Duplicate PSM rows aggregate by `aggregation_method`.** If your input
  carries the same precursor twice for the same run (e.g. Spectronaut's
  candidate-position over-export before top-N attribution), `"sum"` will
  double-count. The site-level pipeline runs top-N attribution to dedupe;
  the precursor pipeline currently doesn't. In practice this is fine
  because Spectronaut's over-export mostly duplicates *sites*, not
  *precursors* — but if you see unexpectedly high intensities, this is
  the first suspect.
- **`peptide_start` can vary within a single precursor** when the peptide
  maps to multiple proteins in the same protein group. The first valid
  integer wins (Hogrebe R script convention), so absolute-position
  annotations are always relative to the first matched protein.
- **`fdr_col`-style bridging assumes STY phospho.** Non-STY phospho
  (`H`, `R`, ...) or other PTMs annotated via
  `EG.PTMLocalizationProbabilities` will get `site_key = NaN` in
  `precursor_to_site_view` -- widen `residue_alphabet` to include them.

## References

- Hogrebe et al. 2018. *Benchmarking common quantification strategies for
  large-scale phosphoproteomics.* Nature Communications 9:1045. (The
  peptide-vs-site debate framed for DDA phosphoproteomics.)
- Krug et al. 2019. *A curated resource for phosphosite-specific signature
  analysis.* Molecular & Cellular Proteomics 18:576-593. (PTM-SEA;
  precedent for precursor-level readout in DIA.)
- Meier et al. 2020. *diaPASEF: parallel accumulation-serial fragmentation
  combined with data-independent acquisition.* Nature Methods 17:1229-1236.
