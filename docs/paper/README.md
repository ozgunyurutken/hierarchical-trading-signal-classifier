# Paper — Build Instructions

LaTeX source for the BBL514E term paper:
**"A Three-Stage Hierarchical Soft-Fusion Framework for Cryptocurrency Trading Signal Classification"**

## Files

```
docs/paper/
├── paper.tex          # main IEEE conference document
├── references.bib     # bibliography (24 entries)
├── figures/           # PNG figures referenced by paper.tex
│   ├── fig_eng_demo.png
│   ├── fig_stage2_fsm.png
│   ├── fig_zigzag_lookahead.png
│   ├── fig_stage1_confusion.png
│   ├── fig_stage1_truth_vs_pred_btc.png
│   ├── fig_arch_heatmap.png
│   └── fig_arch_equity.png
└── README.md          # this file
```

## Build (local, MacTeX or TeX Live)

```bash
cd docs/paper
pdflatex paper
bibtex paper
pdflatex paper
pdflatex paper      # second run resolves cross-references
```

Output: `paper.pdf`.

## Build (Overleaf)

1. Create a new project; upload `paper.tex`, `references.bib`, and the entire `figures/` directory.
2. Compiler: **pdfLaTeX**.
3. Bibliography: **BibTeX** (default).
4. Press **Recompile**; click again after the first run to resolve `\cite` links.

## Class file

The paper uses `IEEEtran.cls` (IEEE conference style). MacTeX, TeX Live, and Overleaf all bundle it; no manual download required.

## Page count target

IEEE conference style, 4--6 pages including references. The current draft is engineered to land near 6 pages with all figures inline; if the rendered PDF overruns, the easiest reductions are:

1. Drop Fig.~`fig_stage1_confusion` (already implied by Section IV-A text).
2. Convert Table I caption text into in-line prose; trim the bullet list in Contributions.
3. Move the "Inner-CV Meta-Overfitting Case Study" subsection to a footnote.

If the rendered PDF underruns, expand the related-work paragraphs and add a small algorithm box for the Stage-2 FSM transition rules.

## Reproducibility

All experimental results referenced in the paper are reproducible from the repository scripts:

- Figs. 1, 2: `scripts/v5_plot_phase1_5_engineered.py`, `scripts/v5_plot_phase_timelines.py`
- Fig. 3:    `scripts/v5_plot_phase3_zigzag_lookahead_test.py`
- Figs. 4, 5: `scripts/v5_plot_phase3_stage1_results.py --variant tuned`, `scripts/v5_plot_phase3_truth_vs_pred.py --variant tuned`
- Figs. 6, 7: `scripts/v5_plot_phase5_arch_ablation.py` (Phase 5.1 overnight script)

Numerical values cited in Tables and prose are stored in:

- `reports/Phase3.5_after_tune/v5_p3_stage1_overall_tuned.csv`
- `reports/Phase4.5_after_tune/v5_p4_stage3_overall_tuned.csv`
- `reports/Phase5.1_arch_ablation/v5_p5_arch_metrics.csv`
- `reports/Phase5/v5_p5_backtest_summary.csv`
