"""Internal collapse pipeline stages.

This subpackage holds the individual stages of the peptide -> site collapse
pipeline. It is intentionally an implementation detail (leading underscore in
the name): the public entry point is :func:`alphaphos.collapse_sites`, which
composes these stages into a single call returning a fully-packaged
``AnnData``.

If you're reading this trying to understand the algorithm, follow this order:

1. :mod:`._collapse.parsing`     -- parse Spectronaut ``EG.PrecursorId`` and
                                    ``EG.PTMLocalizationProbabilities`` strings
                                    into structured Python data.
2. :mod:`._collapse.keys`        -- build the canonical site identifiers used
                                    as ``adata.var.index`` and elsewhere.
3. :mod:`._collapse.aggregation` -- how multiple precursor rows are combined
                                    into a single per-site quant (sum /
                                    median / mean / Hogrebe "consolidate").
4. :mod:`._collapse.selectivity` -- per-sample phospho-enrichment metric,
                                    computed from raw PSMs BEFORE filtering.
5. :mod:`._collapse.site_pipeline` -- the orchestrated stages that turn a
                                     PSM-level DataFrame into a (sites x samples)
                                     quant matrix + per-(site, run) loc matrix.
6. :mod:`._collapse.masking`     -- localization-based masking: per-run,
                                    global_max, or the condition-aware
                                    majority rule.
7. :mod:`._collapse.output_format` -- assemble the final ``AnnData`` with
                                     ``.var``, ``.obs``, ``.layers``,
                                     ``.uns`` populated.

See :mod:`alphaphos.preprocess.collapse` for the public :func:`collapse_sites`
that composes all of the above.
"""
