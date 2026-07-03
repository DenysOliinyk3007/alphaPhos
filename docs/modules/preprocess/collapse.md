# `alphaphos.preprocess.collapse`

Collapse a PSM-level DataFrame to a site-level `AnnData`.

`collapse_sites` runs the canonical Hogrebe-style pipeline: parse `EG.PrecursorId`,
explode to per-site rows, pivot to a (site &times; sample) linear intensity matrix,
aggregate multi-precursor evidence, mask by localization probability, log2-transform, and
pack everything into an `AnnData`.

## Scientific basis

Peptide-level PSMs from DIA search engines contain multiple observations of the same
phosphosite (different precursor charges, multiple peptides mapping to the same
residue with the same or different multiplicities). Collapse aggregates them into a
single per-site intensity per sample, applies a localization-probability confidence
gate, and packs the result into the scverse-standard `AnnData`. The routine follows the
Hogrebe et al. 2018 Spectronaut plugin conventions (top-N attribution, condition-aware
Class-I masking) with three configurable localization strategies.

## Signature

```python
ap.collapse_sites(
    data: pd.DataFrame,
    *,
    condition_df: pd.DataFrame | None = None,
    advanced: dict[str, Any] | None = None,
    verbose: bool = False,
) -> ad.AnnData
```

## Input

- **`data`**: PSM-level DataFrame from [`read_spectronaut`](../io/spectronaut.md) or
  [`read_diann`](../io/diann.md). See the [io schema](../io/index.md#psm-dataframe-schema-spectronaut--dia-nn)
  for required columns. `data.attrs` is preserved into `adata.uns["source_attrs"]`.
- **`condition_df`**: Sample metadata DataFrame. Must contain `sample` (matching
  `R.FileName` values) and `condition`; extra columns are joined into `adata.obs` as-is.
  **Required** when `advanced["localization_strategy"] == "condition"` (the default);
  optional otherwise.

## Parameters

| Parameter | Default | What it does | Alternatives |
| --- | --- | --- | --- |
| `data` | *required* | PSM DataFrame from `read_spectronaut` / `read_diann`. | -- |
| `condition_df` | `None` | Sample metadata (`sample` + `condition` columns). | Required when `localization_strategy="condition"`. |
| `advanced` | `None` | Dict of overrides for `DEFAULT_COLLAPSE_SETTINGS`. Unknown keys raise. | See below. |
| `verbose` | `False` | Log INFO-level stage progress to stderr. | `True` for pipeline debugging. |

### `advanced` keys (see `DEFAULT_COLLAPSE_SETTINGS`)

| Key | Default | What it does | Alternatives |
| --- | --- | --- | --- |
| `search_engine` | `"SN"` | Which PSM schema to expect. | `"Diann"`. Others raise `NotImplementedError`. |
| `quantification_level` | `"MS2"` | Which quant column to consume. | `"MS1"`, `"auto"`. Falls back per engine if unavailable. |
| `top_n_attribution` | `True` | Deduplicate PSMs by keeping the highest-loc-prob variant per (precursor, site). | `False` to skip (matters if the loc-prob column is missing). |
| `cutoff` | `0.75` | Global-max loc-prob threshold. Sites whose max across runs is below this are dropped. | Any float in [0, 1]. Raise to be stricter. |
| `classI_cutoff` | `0.75` | Per-run loc-prob threshold used by the condition strategy. | Any float in [0, 1]. Matches Spectronaut's native Class-I cutoff. |
| `condition_threshold` | `0.50` | Per-condition majority-rule threshold. If &ge;this fraction of replicates in a condition are Class-I, keep all replicates of that condition. | Any float in [0, 1]. |
| `collapse_level` | `"PG"` | Site-key uniqueness scope. | `"PG"` (protein group). |
| `aggregation_method` | `"sum"` | How to combine multiple precursor rows into one site intensity per sample. | `"sum"` -- summed intensity. |
| `localization_strategy` | `"condition"` | Which loc-prob masking rule to apply. | `"per_run"`, `"global_max"`, `"condition"`. See below. |
| `noise_floor_filter` | `True` | Drop sites whose linear intensities collapse to a monolithic noise-floor value across all samples. | `False` to skip. |
| `drop_all_nan` | `True` | Drop sites with no observed values in any sample after masking. | `False` to keep all sites (produces NaN rows). |

Use `ap.resolve_settings(advanced)` to preview the fully-resolved settings dict.

### Localization strategies

All three operate on the **linear** intensity matrix, before log2. `NaN` in the loc
matrix is treated as "below cutoff".

| Strategy | Rule | When to use |
| --- | --- | --- |
| `"per_run"` (strictest) | Per-cell mask -- keep `(site, run)` only when its own loc-prob &ge; `cutoff`. | Reproduces Spectronaut's native Class-I. Most conservative; loses cells from replicates with borderline loc. |
| `"global_max"` (permissive) | Per-site mask -- keep site only if its MAX loc across runs &ge; `cutoff`. Then keep every non-NaN cell of surviving sites. | Preserves intensity in low-loc runs when the site is confidently localized *somewhere*. Matches the Hogrebe SN plugin historically. |
| `"condition"` (default) | Applied on top of `"global_max"`: per `(site, condition)`, if &ge; `condition_threshold` of replicates have per-run loc &ge; `classI_cutoff`, keep ALL replicates of that condition; else keep only the Class-I replicates. | Recovers information from borderline-loc replicates when the site is reliably localized within a biological condition. Requires `condition_df`. |

## Output

An `anndata.AnnData` with shape `(n_samples, n_sites)`:

- `.X` == `.layers["intensity_log2"]` -- log2 intensity.
- `.layers["localization"]` -- per-cell localization probability.
- `.var.index` = `"Protein|Gene|Site|Mult"` (alphaPhos site keys).
- `.var` columns: `short_key`, `pg_key`, `protein_group_id`, `gene`, `site_aa`,
  `site_position`, `multiplicity`, `UPD_seq`, `n_samples_detected`,
  `mean/max/min_loc_prob`, `n_classI_samples`, `fraction_classI`.
- `.obs.index` = sample id (`R.FileName`).
- `.obs` columns: `condition` (from `condition_df`),
  `phospho_selectivity_pct` (fraction of phospho-containing precursors in that sample's
  raw PSMs), plus any extra columns from `condition_df`.
- `.uns["alphaphos"]` = `version`, resolved `pipeline_params`, per-stage `stats`, and
  (when applicable) `classI_decision_table` and `short_key_collisions`.
- `.uns["source_attrs"]` = `data.attrs` (PSM lineage).

## Raises

- `ValueError` -- unknown keys or invalid values in `advanced`.
- `NotImplementedError` -- `search_engine` set to a value other than `"SN"` or `"Diann"`.
- `KeyError` -- required PSM columns are missing from `data`.

## Example

```python
import alphaphos as ap
import pandas as pd

psm = ap.read_spectronaut("report.parquet")
conditions = pd.DataFrame({
    "sample":    ["s1", "s2", "s3", "s4"],
    "condition": ["ctrl", "ctrl", "trt", "trt"],
})

# Default: MS2 quant, top-N attribution, condition-aware Class-I masking
adata = ap.collapse_sites(psm, condition_df=conditions)

# Strict global-max masking, MS1 quant
adata = ap.collapse_sites(
    psm,
    condition_df=conditions,
    advanced={
        "quantification_level": "MS1",
        "localization_strategy": "global_max",
        "cutoff": 0.90,
    },
)

# For DIA-NN input:
psm = ap.read_diann("report.parquet")
adata = ap.collapse_sites(psm, condition_df=conditions,
                          advanced={"search_engine": "Diann"})
```

## Design goals

- **Deliberately decoupled from FASTA lookups.** The kinase &plusmn;7 sequence window is
  attached later via `ap.add_kinase_windows` on the resulting AnnData. Lets you run
  collapse on non-human data without needing a FASTA at all.
- **Every knob validated.** Unknown `advanced` keys or invalid values raise with a
  message naming the accepted set. Typos surface immediately.

## Known caveats

- `peptide_start = 0` silently emits site key `S0` rather than rejecting. Locked by a
  characterization test; will surface if a future validator lands.
- **9 legacy `.var` columns** (`PTM_group`, `PTM_0_pos_val`, `UPD_seq`, `pg_key`, ...)
  carry no downstream consumers. Slated for cleanup in a future minor.
