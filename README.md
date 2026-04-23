# Trust-Aware Context Selection: Learned Signal Combination for Multi-Hop Evidence Retrieval

We study paragraph-level context selection as a trust-aware scoring problem on HotpotQA's distractor setting, combining six heterogeneous signals spanning relevance, authority, and cross-source consistency, and show that a small supervised combiner beats every single-signal baseline while a naive uniform average does worse than BM25.

## Paper

- **PDF**: [tex/main.pdf](tex/main.pdf)
- **Source**: [tex/main.tex](tex/main.tex); compile with `tectonic tex/main.tex`
- **Sealed-PDF peer review**: [review.md](review.md) (iter 2 recommendation: accept, score 8/10)

## Primary result

**Precision@2 (paragraph selection): 0.345 ± 0.032** across 3 seeds, a **+3.3 absolute-point** improvement over the strongest single-signal baseline (SBERT at 0.312 ± 0.035). The per-seed sign of the gap is consistent on every seed. The gold-versus-distractor score gap more than doubles, from 0.170 to 0.349.

The fitted logistic-regression weights recover two anti-signals: paragraph log-length (-0.98) and cross-paragraph consistency (-0.88) are learned with negative coefficients in HotpotQA's distractor setting, because distractors are length-matched to gold and topically close enough that the naive authority and corroboration heuristics point toward distractors rather than gold.

## How to reproduce

```bash
python -m pip install "datasets>=2.14" numpy scikit-learn rank-bm25 sentence-transformers tqdm
SEEDS=0,1,2 SUBSET_SIZE=300 python experiments/02-learned-trustscore/experiment.py
```

The script loads HotpotQA distractor through the HuggingFace hub, subsets 300 questions per seed, splits 100/200 into train/test, fits a 6-feature logistic regression on the 100 training questions, and evaluates all seven conditions on the held-out 200. End-to-end wall-clock under fifteen minutes on a single CPU core.

## Figures

| Figure | Section | What it shows |
| --- | --- | --- |
| `figures/fig-architecture.png` | Method | Pipeline: query + 10 candidates → 6-signal feature extraction → logistic-regression combiner → top-2 selection. |
| `figures/fig-signal-taxonomy.png` | Method | Three trust dimensions grouping the six signals: relevance, authority, consistency. |
| `figures/fig-main-results.png` | Results | Precision@2 across all seven methods with error bars. |
| `figures/fig-weights.png` | Results | Fitted logistic-regression weights; log-length and consistency are learned as anti-signals. |
| `figures/fig-score-gap.png` | Results | Gold-versus-distractor score gap; Learned-TrustScore's gap is 2x SBERT's. |
| `figures/fig-per-seed.png` | Results | Per-seed Precision@2 for SBERT vs Learned-TrustScore; learned wins on every seed. |

## Recommended venues

- **NeurIPS** — top ML venue; natural home for retrieval-augmentation and trust-aware context selection work. Abstract deadline around late May.
- **EMNLP** — HotpotQA's original venue; primary natural home for this paper. Submission around mid-June.
- **ACL Findings** — safe mid-tier fallback with 35-40% acceptance; ARR rolling cycle.
- **TMLR** — rolling-submission journal that rewards correctness over novelty-above-SOTA; excellent archival fit for the honest single-benchmark scope of this paper.
- **ACM TOIS** — IR-leaning archival journal if extending to a full-length treatment with cross-benchmark validation.

## Authors

Vikash Chandra Mishra, Vizuara AI Labs (<vikash@vizuara.com>)

## Repository layout

```
.
├── README.md               this file
├── LICENSE                 MIT
├── review.md               sealed-PDF peer review (iter 2)
├── state.json              session state
├── log.md                  per-stage audit log
├── tex/                    LaTeX source + compiled PDF
│   ├── main.tex
│   ├── main.pdf
│   ├── references.bib
│   └── *.tex               per-section inputs
├── figures/                all 6 figure PNGs
├── experiments/
│   ├── 01-uniform-baseline/
│   │   ├── run_1.log
│   │   ├── summary.json    iteration 1 (naive uniform; judged not-worthy)
│   │   └── judge_notes.md
│   └── 02-learned-trustscore/
│       ├── experiment.py
│       ├── experiment_plan.md
│       ├── run_2.log
│       ├── summary.json    iteration 2 (learned; judged worthy, score 8)
│       └── metrics_seed*.json
└── docs/                   (populated by the site build)
```

## Layout audit

Full mechanical validator (`scripts/pdf_layout_check.py` in the upstream pipeline repo) was run on the final PDF with all six checks enabled:

- `overfull_hbox` (compile log scan, >=5pt = blocking)
- `every_page_render` (all 11 pages at 140 dpi)
- `orphan_pages` (<300 chars = blocking)
- `column_fill` (two-column balance ratio)
- `float_in_bibliography` (figure on bibliography page = blocking)
- `missing_legend_block` (any figure prompt without a LEGEND: block = blocking)

Result: **PASS, 0 blocking, 0 warnings**. Full JSON report at [visual_audit_final/report.json](visual_audit_final/report.json); per-page PNG renders at [visual_audit_final/](visual_audit_final/).

## Provenance

Session id: `20260422-100735-94d7`. See [log.md](log.md) and [state.json](state.json) for the full audit trail.
