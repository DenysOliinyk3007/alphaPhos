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
| `quantification_level` | `None` | Which quant column to consume. `None` = the engine's default (`DEFAULT_QUANT_LEVEL`): `"MS2"` for Spectronaut, `"MS1"` (`Ms1.Translated`) for DIA-NN -- alphaPhos's deliberate, dilution-series-validated choice. | `"MS2"` (conventional fragment quant on both engines; `Precursor.Quantity` on DIA-NN), `"MS1"`, `"auto"` (engine's canonical column). Falls back along `MS2 -> MS1 -> auto` with a warning if the requested level is absent. |
| `top_n_attribution` | `"auto"` | Spectronaut over-export dedup: Spectronaut writes one row per candidate localization of an ambiguous precursor, each with the full intensity; only the rows whose encoded positions are the top-*N* candidates are kept. `"auto"` applies it for `search_engine="SN"` only -- DIA-NN writes one peptidoform row per precursor-run, so there is nothing to dedup and the filter would only delete low-confidence peptidoforms (validated: it removed 796 real cells on the EGF HeLa DIA-NN report). | `True` / `False` to force. |
| `cutoff` | `0.75` | Global-max loc-prob threshold. Sites whose max across runs is below this are dropped. | Any float in [0, 1]. Raise to be stricter. |
| `classI_cutoff` | `0.75` | Per-run loc-prob threshold used by the condition strategy. | Any float in [0, 1]. Matches Spectronaut's native Class-I cutoff. |
| `condition_threshold` | `0.50` | Per-condition majority-rule threshold. If &ge;this fraction of replicates in a condition are Class-I, keep all replicates of that condition. | Any float in [0, 1]. |
| `collapse_level` | `"PG"` | Site keys use the first protein-group accession (a contaminant-tagged accession wins over its untagged twin). | Only `"PG"`. The former `"P"` option was removed in 0.23 -- it never resolved proteins, it only kept the raw `;`-joined group string. |
| `aggregation_method` | `"sum"` | How to combine multiple precursor rows into one site intensity per sample. | `"sum"` (default), `"mean"`, `"median"`, `"consolidate"` (Hogrebe ratio-imputation). |
| `precursor_loc_gate` | `True` | Before aggregating, a precursor contributes to a site in a run only if *its own* loc-prob for that site in that run &ge; the strategy's cutoff (`classI_cutoff` for `condition`, `cutoff` otherwise). If **no** precursor of the site is Class-I in that run, all are aggregated and the site-level mask decides. Mirrors Spectronaut's PTM consolidation; on the EGF HeLa benchmark it removes a +0.06 log2 inflation of multi-precursor sites (97.5% vs 91.6% of cells within 0.1 log2 of the native report). | `False` reproduces pre-0.23 output (every precursor summed). |
| `localization_strategy` | `"condition"` | Which loc-prob masking rule to apply. | `"per_run"`, `"global_max"`, `"condition"`, `"wilson"`. See below. |
| `noise_floor_filter` | `True` | Set cells whose log2 value is exactly 0 or 1 (linear 1 or 2 -- Spectronaut's noise-floor convention) to NaN. | `False` to skip. |
| `drop_all_nan` | `True` | Drop sites with no observed values in any sample after masking. | `False` to keep all sites (produces NaN rows). |

Use `ap.resolve_settings(advanced)` to preview the fully-resolved settings dict.

### Localization strategies

All three operate on the **linear** intensity matrix, before log2. `NaN` in the loc
matrix is treated as "below cutoff".

| Strategy | Rule | When to use |
| --- | --- | --- |
| `"per_run"` (strictest) | Per-cell mask -- keep `(site, run)` only when its own loc-prob &ge; `cutoff`. | Reproduces Spectronaut's native Class-I. Most conservative; loses cells from replicates with borderline loc. |
| `"global_max"` (permissive) | Per-site mask -- keep site only if its MAX loc across runs &ge; `cutoff`. Then keep every non-NaN cell of surviving sites. | Preserves intensity in low-loc runs when the site is confidently localized *somewhere*. Matches the Hogrebe SN plugin historically. |
| `"condition"` (default) | Per `(site, condition)`, if &ge; `condition_threshold` of replicates have per-run loc &ge; `classI_cutoff`, keep ALL replicates of that condition; else keep only the Class-I replicates. Sites never Class-I anywhere end up all-NaN and are removed by `drop_all_nan`, so `"global_max"` is implied. Samples missing from `condition_df` fall back to the `"per_run"` rule (with a warning) -- they never bypass masking. | Recovers information from borderline-loc replicates when the site is reliably localized within a biological condition. Requires `condition_df`. |

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
  (when applicable) `classI_decision_table` and `short_key_collisions`
  (`{short_key: [full_key, ...]}`; everything in `.uns` round-trips through `write_h5ad`).
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
