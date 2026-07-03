# `alphaphos.enrichment`

Downstream enrichment for phosphoproteomics differential-expression results.

Given a per-site limma output (a `pd.DataFrame` from `ap.diff_exp_limma` indexed by
`Protein|Gene|Site|Mult` keys), this module answers three complementary biological questions:

| Question | Submodule | Function |
| --- | --- | --- |
| Which **kinases** are (in)activated? | [`ksea`](ksea.md) | [`kinase_activity`](ksea.md) |
| Which **pathways** are enriched among the significant hits? | [`pathway`](pathway.md) | [`pathway_enrichment`](pathway.md) |
| Which **pathways** move coherently (no threshold)? | [`pathway_gsea`](pathway_gsea.md) | [`pathway_gsea`](pathway_gsea.md) |

All three share the same input schema (limma `diff_exp_result`), auto-parse the alphaPhos
site keys, and stamp provenance on `result.attrs["provenance"]`.

## Which one should I use?

They're not alternatives -- they're complementary lenses.

**`kinase_activity`** operates at the **site** level and asks *"which kinases are the
sites moving as substrates of?"*. Directly interpretable as kinase (in)activation.
Best when you care about upstream regulators.

**`pathway_enrichment`** (ORA) operates at the **gene** level and asks *"which pathways
are over-represented in my significant-hit list vs my phospho background?"*. Threshold-based
(FDR + log2FC); throws away sub-threshold sites. Best when you have a clean strong signal.

**`pathway_gsea`** operates at the **gene** level and asks *"reading the full ranked
list, which pathways move coherently (up or down)?"*. Rank-based; no threshold; keeps
subtle-but-consistent motion. Best when your effect is broad rather than concentrated in
a few outliers.

A common workflow runs all three on the same limma output and compares the top hits: if
`kinase_activity` fingers a kinase, `pathway_enrichment` picks up its canonical downstream
pathway among the up-hits, and `pathway_gsea` shows the same pathway as coherently
positive -- that's a triangulated finding worth reporting.

## Scientific basis (one line each)

- **`ksea.kinase_activity`** wraps [`decoupler-py`](https://github.com/saezlab/decoupler-py)'s
  ULM / MLM linear models (Badia-i-Mompel et al. 2022) against OmniPath or the curated
  PTM functional DB.
- **`pathway.pathway_enrichment`** wraps
  [`gseapy.enrichr`](https://gseapy.readthedocs.io/en/latest/introduction.html) (Fang et
  al. 2023) against Enrichr gene-set libraries with client-side custom background.
- **`pathway_gsea.pathway_gsea`** wraps
  [`gseapy.prerank`](https://gseapy.readthedocs.io/en/latest/introduction.html) --
  Subramanian 2005 running-sum ES with permutation-based FDR -- against the same Enrichr
  libraries.

## Also in this module

Lower-level primitives used internally, exposed for advanced use:

- **`load_ptm_db`**, **`site_id`**, **`emit_libraries`**, **`load_gmt`** -- access to the
  bundled 235k-site PTM functional database and its GMT emission machinery.
- **`match_sites`**, **`attach_site_ids`**, **`parse_alphaphos_key`** -- site-key matching
  engine (`Protein|Gene|Site|Mult` &harr; `Protein_AApos`).
- **`ora`**, **`gsea`** -- site-level engines used by the PTM-signature validation
  pipeline. Not the same as `pathway_enrichment`/`pathway_gsea` (which are gene-level).
- **`build_kinase_substrate_library`**, **`load_ev3`**, **`ev2`**,
  **`score_against_ev3`** -- validation harness against the
  Hernández-Armenta 2017 / Ochoa 2020 EV3 benchmark.

## Install

The enrichment stack is behind the `[enrichment]` optional extra:

```bash
pip install "alphaphos[enrichment]"
```

That pulls `gseapy>=1.1`, `decoupler>=2.0`, `omnipath>=1.0`. Without it,
`kinase_activity`/`pathway_enrichment`/`pathway_gsea` raise `ImportError` on call and
everything else in alphaPhos works.
