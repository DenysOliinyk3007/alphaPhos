# Alpha-tester guide

Thanks for trying alphaPhos while it's still alpha. This page is short on purpose — it tells you what feedback we're most interested in, what's fair to complain about, and what's already known-rough so you don't waste time reporting it.

**tl;dr**: run the [quickstart](quickstart.md) on your own data. Every step that surprises, confuses, or breaks is useful feedback. Nothing is too small to mention.

---

## What you're testing

alphaPhos 0.22.0 — a Python-native phosphoproteomics analysis package built for DIA workflows and large-cohort studies. Core science modules (IO → collapse → filter → impute → DE → enrichment) are populated and tested; polish (visualization, error messages, docs) is uneven.

The public API you should try is inventoried in [`how-to-use-alphaphos.md`](how-to-use-alphaphos.md). The design choices (why these defaults, why these gates) are explained in [`design-principles.md`](design-principles.md). The 30-minute smoke test is [`quickstart.md`](quickstart.md).

**Version compatibility**: Python 3.10, 3.11, or 3.12. Windows or Linux. numpy < 2.4 currently required for KSEA and UMAP (upstream numba/numpy incompatibility; use everything else at any numpy).

---

## What feedback we most want

Roughly in decreasing order of value:

1. **The quickstart failed at step N on your machine.** Absolute gold. Include: OS, Python version, exact command, full error message. If it was your own data, the first 3 lines of your TSV column headers help too.

2. **A default choice surprised you.** You expected `filter_by_completeness` to keep more sites; or `strategy="wilson"` to be more/less strict; or `recommend_pipeline` to prescribe something else. We want to hear this even if the choice turns out to be defensible — the surprise itself is useful signal about the docs. See [`design-principles.md`](design-principles.md) for the reasoning that should already have prevented the surprise; if the reasoning is unconvincing, that's the feedback.

3. **A result disagreed with a reference tool.** You ran the same data through Perseus / MSstatsPTM / msqrob2 and the hit list is different. We want the *why*, not just the observation — is our filter too strict, our imputer wrong, our DE model different in a way that matters?

4. **A workflow you needed isn't in the package.** Concretely: "I wanted to X and had to write it myself because ap.Y doesn't handle Z." Include what you ended up writing.

5. **The docs are wrong.** Broken links, code that doesn't run, API signatures that changed since the doc was written, jargon that isn't defined. Every one of these is a 5-minute fix if we hear about it.

6. **A performance surprise.** Something ran much slower than you expected (or much faster, if that's suspicious). Include cohort size and hardware.

---

## Where to report

- **GitHub issues** (preferred): https://github.com/DenysOliinyk3007/alphaPhos/issues — one issue per report. Include the info from §"What feedback we most want" above.
- **Email**: `oliinyk@biochem.mpg.de` — for things you'd rather not have public, or if GitHub is too heavy for the specific report.

**What to include** in a bug report:
- alphaPhos version (`ap.__version__`)
- OS + Python version (`python --version`; on Windows also whether via `py` or `python`)
- Exact command that failed
- Full error message (all lines of the traceback)
- Whether you can reproduce with the bundled EGF fixture, or only with your own data
- If only with your own data: the first 3 rows of your PSM report (columns only, no data if sensitive)

---

## What's known-rough (please don't report as bugs)

These are all on the roadmap; reporting them individually just adds noise:

- **No `viz` submodule yet.** Publication-quality volcano / heatmap / PCA scatter don't ship yet (planned). `ap.qc.generate_dashboard(adata, "qc.html")` and the `ap.qc.panel_*` Plotly renderers cover the exploratory side.
- **Error messages are inconsistent.** Some are polished (Wilson threshold validation); others let a Python traceback bubble up. If you hit an unclear one, telling us *which* one is helpful — the audit uses your list.
- **KSEA and UMAP require numpy < 2.4.** Upstream numba issue. If you need them, pin numpy in your environment.
- **`ap.dose_response.fit_dose_response` needs the optional `curve_curator` dep.** Install with `pip install "alphaphos[dose_response]"`. Not blocking for standard DIA phospho.
- **Documentation coverage is uneven** between the top-level tutorial (`how-to-use-alphaphos.md` — refreshed) and the per-module deep-dives under `docs/modules/` (some still lag). If the module docs disagree with the top-level, trust the top-level.
- **Memory scaling beyond `n = 500` not validated.** Runs on `n = 383` (uPhosHT full plate); we haven't stress-tested at `n = 1000+`.
- **API can still change.** Anything with an underscore prefix (`_internal`) is off-limits by convention; everything else is public but not yet contract-stable. Breaking changes will land in minor versions during 0.x. We'll write a stability contract at 1.0.

---

## What's out of scope for this alpha

Things we don't want feedback on (yet):

- **Comparison to non-Python tools' UX.** We know Perseus is more clickable; that's not the axis we're competing on.
- **Additional imputation methods.** We benchmarked shift_rsn / KNN / hybrid / PIMMS-DAE / PIMMS-VAE across 5 datasets; PIMMS-DAE + KNN are the shipped defaults. If you have a strong preference for a specific other imputer (msqrob2's minprob, Amelia II, missForest), file an issue with your reasoning — but expect us to say "not yet, maybe post-1.0."
- **The precursor-informed imputation line.** We tried three variants on EGF and dropped them (details in the CHANGELOG). Not planning to revisit unless someone has a concrete counter-benchmark.
- **Full R-limma compatibility.** Our limma is a clean-room Python implementation of Smyth 2004. It matches R limma on our validation benchmark at Pearson `r = 1.0000` on logFC, but there will be small numerical differences (e.g. `t_stat` down to the 4th decimal). If you rely on bitwise identity with R, alphaPhos is not that tool.
- **Publication plots / a `viz` submodule** (as noted above; we know).

---

## The one thing we can't fix without your data

**Scale.** The largest cohort our defaults are validated on is `n = 383` (uPhosHT full plate). If you run alphaPhos on `n > 500` and something breaks, misbehaves, or takes surprisingly long — that's the exact signal we need to fix the scale-out story. Please report it even if you're not sure it's a bug.

---

## What we'll do with your feedback

Reasonable turnaround:

- **Same day**: acknowledge that we saw it
- **This week**: triage — bug / feature request / expected behaviour / out-of-scope
- **This month**: fix bugs, incorporate doc improvements
- **Post-alpha**: features + polish drive the beta scope

You are not signing up for open-ended support. If we go quiet, feel free to nudge.

Thanks again — the pilot phase of a tool is where it gets shaped most. Every note you send lands.
