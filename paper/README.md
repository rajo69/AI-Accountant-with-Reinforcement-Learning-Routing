# Paper Companion

A 1-page technical note summarising the research extension, designed to be
read in ~60 seconds by someone who won't open the full repository.

## Files

- `one_pager.tex` — LaTeX source. Compiles to a single A4 page.
- `one_pager.md` — Markdown mirror for GitHub readers. Same content.

## Compile the LaTeX to PDF

Requires a TeX distribution (TeX Live, MiKTeX, or MacTeX). No bibliography
pass is needed — references are inline.

```bash
# From the repository root:
pdflatex -output-directory paper paper/one_pager.tex
```

Output: `paper/one_pager.pdf`.

A full LaTeX installation is ~4 GB. If you only want to view or share the
content, `one_pager.md` renders directly on GitHub and contains the same text.

## Updating the numbers

All quantitative claims in the 1-pager are sourced from:

- `experiments/results/evaluation_results.json` (from `python -m agent.evaluate`)
- `experiments/results/statistical_summary.json` (from `python -m experiments.statistical_analysis`)

If either of those files change, update the table and per-tier numbers in
both `one_pager.tex` and `one_pager.md` by hand — they are currently not
auto-generated. (A future improvement: a small Jinja-based template that
fills both files from the JSON.)

## Arxiv submission (future)

The current 1-pager is intentionally short for a portfolio / application
companion. To submit to arXiv as a short technical note:

1. Expand to ~4–6 pages: add a related-work paragraph (Mozannar & Sontag
   2020 learning-to-defer; Kadavath et al. 2022 confidence elicitation;
   Guo et al. 2017 calibration; RouteLLM 2025), a proper Methods section
   with reward-table and full hyperparameters, and a Discussion section
   covering the calibration-probe follow-up.
2. Register for an arXiv account and obtain an endorsement (cs.LG or cs.AI).
3. Submit as source (LaTeX + `experiments/results/figures/*.png`) rather
   than PDF.

The calibration probe and regime probe have since shipped and are
summarised in the current 1-pager; any arXiv expansion should lead with the
regime-probe result (EV-invariance confirmed on both raw and calibrated
signals) as the positive-result component, and treat the natural-regime
convergence as the motivating negative result rather than the headline.
