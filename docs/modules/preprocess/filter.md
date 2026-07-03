# `alphaphos.preprocess.filter`

Site-completeness filter for phosphosite AnnData.

`filter_by_completeness(adata, min_valid_frac, ...)` returns a new `AnnData` with sites
that fail a valid-value threshold dropped. Three group-handling strategies let you tune
whether the threshold is applied globally, per group, or across every group.

## Scientific basis

DIA phosphoproteomics data is missing-not-at-random on the tail (below-LOD sites are
missing preferentially in the low-intensity condition), so imputation without a prior
completeness filter can inject noise or hide biology. The standard practice is to require
observations in a fixed fraction of samples per condition before imputation, so imputed
values back-fill occasional MAR gaps rather than reconstructing whole low-abundance
sites.

## Signature

```python
ap.filter_by_completeness(
    adata: ad.AnnData,
    *,
    min_valid_frac: float,
    group_column: str | None = None,
    keep_strategy: Literal["all", "any", "each"] = "all",
    layer: str | None = None,
) -> ad.AnnData
```

## Input

An `AnnData` with shape `(n_samples, n_sites)`. Missingness is computed on `layer`
(default `adata.X`).

## Parameters

| Parameter | Default | What it does | Alternatives |
| --- | --- | --- | --- |
| `adata` | *required* | Site-level AnnData from `collapse_sites` (or `read_fragpipe_sites`). | -- |
| `min_valid_frac` | *required* | Float in [0, 1]. Fraction of non-NaN values required to keep a site. | E.g. `0.7` = &ge; 70% non-NaN. |
| `group_column` | `None` | Column in `adata.obs` partitioning samples into groups. | Required for `"any"` / `"each"`; must be `None` for `"all"`. |
| `keep_strategy` | `"all"` | How the threshold is applied. | See below. |
| `layer` | `None` | Which layer to compute missingness on. | `None` = `adata.X`. Any layer name. |

### `keep_strategy` in detail

| Value | Rule | Requires `group_column`? |
| --- | --- | --- |
| `"all"` | Keep site if its **overall** valid fraction &ge; `min_valid_frac`. No grouping. | No -- must be `None`. |
| `"any"` | Keep site if it passes the threshold **in at least one group**. | Yes. |
| `"each"` | Keep site only if it passes the threshold **in every group**. | Yes. |

Filter dropping affects **all** layers / obsm / varm automatically via AnnData
subsetting.

## Output

A new `AnnData` (always -- does not mutate input) with failing sites removed.

## Raises

- `ValueError` -- `min_valid_frac` out of range, unknown `keep_strategy`, or invalid
  strategy/group combination (e.g. `"all"` with `group_column` set, or `"any"`/`"each"`
  without).
- `KeyError` -- `group_column` not in `adata.obs`, or `layer` not in `adata.layers`.

## Example

```python
import alphaphos as ap

# --- Keep sites present in >=2/3 samples of EVERY condition (strict) ---
# Recommended default for 3-vs-3 designs — every site is guaranteed at least
# 2 observations in each group before imputation.
adata = ap.filter_by_completeness(
    adata,
    min_valid_frac=2/3,
    group_column="condition",
    keep_strategy="each",
)

# --- Present in AT LEAST ONE condition (rescues condition-specific sites) ---
adata = ap.filter_by_completeness(
    adata,
    min_valid_frac=2/3,
    group_column="condition",
    keep_strategy="any",
)

# --- Simple global completeness (no groups) ---
adata = ap.filter_by_completeness(adata, min_valid_frac=0.7)
```

## When to use which strategy

- **`"each"`** (strict) is the safe default for a balanced 2-condition design. Every
  surviving site has enough support in every group to survive imputation cleanly.
- **`"any"`** rescues sites that are condition-specifically present -- for example, a
  phospho site that only appears after stimulation. Combined with imputation, this can
  yield hits that `"each"` filters away. Use when biology-specific presence/absence is
  scientifically meaningful and your imputer treats the missing group as MNAR.
- **`"all"`** is the correct choice when there is no meaningful grouping (e.g. a
  time-course with no discrete groups, or a pilot with a single condition).
