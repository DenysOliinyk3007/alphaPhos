# `alphaphos.io.silac` — multi-channel (SILAC) support

Status: **alpha**, added in 0.25.0. Spectronaut only; DIA-NN multi-channel is
not wired up.

## The one thing to get right in the export

Spectronaut encodes the metabolic label as a residue tag inside **`FG.Id`**:

```
_AAAAAATAPPS[Phospho (STY)]PGPAQPGPR_.2           <- light
_AAAAAATAPPS[Phospho (STY)]PGPAQPGPR[Arg6]_.2     <- medium
_AAAAAATAPPS[Phospho (STY)]PGPAQPGPR[Arg10]_.2    <- heavy
```

**No other column carries the channel.** `EG.PrecursorId` and
`EG.ModifiedSequence` have the label stripped, so all three rows above share
one precursor id. `EG.Workflow` only says `LABEL` or `LABEL_FREE`, and
`LABEL_FREE` rows carry label tags too, so it cannot be used either.

Include `FG.Id` in the report schema. Without it a labelled report is
indistinguishable from a label-free one, and its channels would be summed into
a single number per run.

alphaPhos refuses to let that happen: a report with several rows per
`(R.FileName, EG.PrecursorId)` and no `FG.Id` raises at read time.

## Reading

`read_spectronaut` detects the label tags, adds a `Channel` column, and records
the per-channel row counts in `df.attrs["channels"]`.

```python
import alphaphos as ap

df = ap.read_spectronaut("SILAC_Report.parquet")
df.attrs["channels"]
# {'light': 119613, 'medium': 102265, 'heavy': 103626}
```

The default plex is the standard Lys/Arg triplex:

| tag | channel |
| --- | --- |
| *(none)* | `light` |
| `Arg6`, `Lys4` | `medium` |
| `Arg10`, `Lys8` | `heavy` |

A 2-plex simply never produces the medium tags. For a non-standard label, pass
your own mapping; the light channel is always the absence of a tag and must not
appear in it.

```python
df = ap.read_spectronaut(path, advanced={"label_channels": {"Lys8": "heavy"}})
```

A precursor carrying tags from two different channels (`Arg6` with `Lys8`)
raises. That cannot happen in a correctly searched experiment and would
corrupt every ratio the precursor contributes to.

## Collapsing: a channel is a sample

In SILAC the channels of one run are different biological material, so a sample
is **one run times one channel**. `collapse_sites` folds the channel into the
sample id and every downstream stage treats channels as the separate samples
they are.

```python
adata = ap.collapse_sites(df, advanced={"localization_strategy": "global_max"})
adata.shape          # (18, 15080)  <- 6 runs x 3 channels
adata.obs["run"]     # the run, for use as a blocking factor
adata.obs["channel"] # light / medium / heavy
```

Sample ids are `run::channel`. A run name containing `::` raises.

`condition_df` must therefore name channels, not runs, because in a SILAC
design the condition belongs to the channel. Passing bare run names raises with
that explanation rather than silently leaving every condition empty.

## ❗️Class-I evidence is counted per run, not per channel

Spectronaut assigns the site **once per channel group**. On the reference
dataset, zero of 101,076 groups differ between their channels on localisation
probability, assay probability, protein PTM locations or Q-value.

So the channels of a run do not carry independent localisation evidence.
Counting them as independent observations inflates the Wilson/Jeffreys
denominator and makes the Class-I lower bound anti-conservative:

| evidence | k/n | Wilson lower bound |
| --- | --- | --- |
| 6 runs, channels collapsed | 6/6 | 0.670 |
| the same evidence, channels counted separately | 18/18 | **0.871** |
| 4 of 6 runs Class-I | 4/6 | 0.286 |
| the same evidence, channels counted separately | 12/18 | **0.437** |

alphaPhos therefore computes the bound on the **run grid**: a run counts once
for a site if any of its channels qualifies. Two extra `var` columns record it,
and they appear only for multi-channel data:

| column | meaning |
| --- | --- |
| `n_runs_detected` | runs quantifying the site in at least one channel |
| `n_classI_runs` | of those, runs where the site is Class-I |

`classI_wilson_lb` is computed from these two, so `strategy="wilson"` and
`wilson_threshold_sensitivity` stay correct on SILAC data without any change at
the call site.

`n_samples_detected` keeps counting the channel grid, because quantification
genuinely is per channel.

## Analysis: use the design, not division

Treat the result as intensities and get the within-run pairing from the design:

```python
res = ap.diff_exp_limma(
    adata,
    condition_column="channel",
    comparison=("heavy", "light"),
    covariates=["run"],          # the within-run pairing
)
```

Limma with the run as a blocking factor on log2 intensities is the paired
analysis. It keeps every site instead of only those with a complete channel
pair, and reuses filtering, imputation and DE unchanged.

**Do not form site-level ratios by dividing the collapsed channels.** Two
thirds of sites have more than one precursor, and in 75.5% of those the
channels do not cover the same precursors. Dividing per-channel sums then puts
a different set of precursors in the numerator and the denominator, so the
ratio partly reflects coverage rather than biology. A ratio has to be formed at
precursor level, where numerator and denominator are the same molecule. That is
not implemented yet; see the limitations below.

## ❗️Normalise before testing

alphaPhos has no normalisation step, and SILAC needs one. The channels of a run
are mixed by hand and never land at exactly 1:1:1, so the loading offset shows
up as a global fold change on top of the biology. On the reference dataset the
uncorrected paired test gives a median log2 heavy/light of **-0.25**; centring
each sample on its median moves it to **+0.012**, which is where a phospho
time course should sit, since EGF moves a minority of sites in each direction.

```python
import numpy as np
X = adata.X - np.nanmedian(adata.X, axis=1, keepdims=True)
adata.X = X
adata.layers["intensity_log2"] = X.copy()
```

Do this before any testing. Without it every contrast carries the mixing ratio.

## What the reference dataset validates

The `id_test` "mixed" dataset is a real EGF time course, not a same-material
control: light is unstimulated HeLa, medium is 5 min EGF, heavy is 10 min EGF,
lysates mixed 1:1:1. Six replicate runs.

It validates the whole chain, because the expected biology appears with the
right sign, the right magnitude and the right **order**. Mean log2 change
against unstimulated, over the six runs:

| site | 5 min | 10 min |
| --- | --- | --- |
| RPS6 S235 | +2.08 | +3.39 |
| RPS6 S236 | +2.14 | +3.34 |
| RPS6 S240 | +1.99 | +3.29 |
| SOS1 S1134 | +2.14 | +3.05 |
| STMN1 S25 | +1.96 | +2.16 |
| CTTN S405 | +0.83 | +1.94 |

RPS6 S235/S236/S240 are the canonical RSK and S6K readout downstream of ERK.
SOS1 S1134 is the textbook ERK negative-feedback site. A monotonic rise across
three channels can only appear if the channels are assigned correctly, kept
separate, and quantified per channel, so this is a direct check on collapse
rather than a plausibility argument.

Supporting coherence: Spearman 0.59 between the 5 min and 10 min fold changes,
and 173 of the 344 sites up more than two-fold at 10 min are already up more
than two-fold at 5 min.

**What it does not show.** Global motif enrichment among the induced sites is
weak. Proline-directed (ERK/CDK) sites are 48.8% of the induced set against
44.8% of the non-significant set (odds ratio 1.18, p = 0.16), and no basophilic
enrichment is detectable. The one significant signal is a *depletion* of the
acidic CK2 motif (23.3% vs 29.5%, odds ratio 0.72, p = 0.018), consistent with
EGF-responsive sites not being constitutive CK2 targets. Read the site-level
evidence, not the global signature.

Two caveats on scope. The numbers above come from the 5,116 sites quantified in
all 18 samples, out of 15,080 collapsed, which biases towards abundant sites.
And the per-sample median centring below is required before any of it holds.

## Current limitations

- Spectronaut only.
- No precursor-level ratio output. Use the blocking-factor design above.
- No label-incorporation or arginine-to-proline conversion QC, which the
  reference dataset suggests are needed to explain its residual channel
  differences.
- No normalisation step anywhere in the package; median-centre by hand.
- The channel-to-condition mapping is supplied by the user via `condition_df`;
  nothing is inferred from the run name.
