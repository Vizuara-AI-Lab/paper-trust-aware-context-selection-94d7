# Email to student

**To:** vikash@vizuara.com
**Subject:** Your paper draft: Trust-Aware Context Selection (9pp, accept/score 8)

---

Hi Vikash,

Your paper draft is ready. Here is the full delivery.

**Paper.** *Trust-Aware Context Selection: Learned Signal Combination for Multi-Hop Evidence Retrieval*, a 9-page manuscript attached as PDF and mirrored on the project site.
Primary result: Precision@2 of **0.345 ± 0.032** across 3 seeds on HotpotQA distractor, a 3.3 absolute-point improvement over the strongest single-signal baseline (SBERT at 0.312 ± 0.035). The gain is consistent on every seed, and the gold-versus-distractor score gap more than doubles (0.170 → 0.349). The ablation condition (naive uniform averaging of three trust signals) collapses to 0.080, below BM25 — that negative result is preserved in the paper and turns into the most instructive row of Table 1.

**Code & data.** Everything is in the repo: https://github.com/Vizuara-AI-Lab/paper-trust-aware-context-selection
You can reproduce the entire paper in under fifteen minutes on a single CPU core:
    pip install datasets numpy scikit-learn rank-bm25 sentence-transformers tqdm
    SEEDS=0,1,2 SUBSET_SIZE=300 python experiments/02-learned-trustscore/experiment.py

**Project site.** https://vizuara-ai-lab.github.io/paper-trust-aware-context-selection/
(GitHub Pages build is usually live within a few minutes of the first push; refresh if the URL returns 404 briefly.)

**Figures.** Six figures, all generated via the paper-banana pipeline and embedded in the PDF:
1. Trust-aware pipeline architecture (Method)
2. Signal taxonomy (Method)
3. Precision@2 across all seven methods (Results)
4. Fitted logistic-regression weights (Results)
5. Gold-versus-distractor score gap (Results)
6. Per-seed Precision@2 consistency (Results)

**Sealed-PDF peer review.** A domain-expert / deep reviewer pass is committed as `review.md` in the repo root. Iteration 1 recommended weak accept (score 7); iteration 2 recommended accept (score 8) after three targeted revisions (abstract narrowed to the three measured trust dimensions, conclusion's broader-relevance claim softened to a conjecture, method disclosed the entity-overlap regex).

**Where to submit, in priority order:**
- **NeurIPS** — top ML venue; natural home for retrieval-augmentation and trust-aware context selection work. Abstract deadline ~2026-05-22 (estimated).
- **EMNLP** — HotpotQA's original venue; primary natural home. Submission ~2026-06-15 (estimated).
- **ACL Findings** — safe mid-tier fallback; ~35-40% acceptance via the ARR rolling cycle.
- **TMLR** — rolling-submission journal that rewards correctness over novelty-above-SOTA; excellent archival fit for the honest single-benchmark scope of this work.
- **ACM TOIS** — IR-leaning archival journal if you extend to a full-length treatment with cross-benchmark validation.

**Next steps, roughly in order:**
1. Read the PDF end-to-end. The sealed-PDF reviewer flagged two low-severity issues worth acknowledging in any submission: (a) three seeds and 200-question test folds give only a single-seed-std margin, so a paired significance test or bootstrap over the 600 held-out questions would tighten the claim; (b) evidence is from a single benchmark, so the Conclusion's broader-relevance claim is phrased as a conjecture for future work.
2. Verify the numbers in Table 1 and Table 2 against `experiments/02-learned-trustscore/summary.json` and `metrics_seed*.json`. Every number in the paper traces back to those files.
3. Pick a venue and deadline, then reply with which one. I'll swap the documentclass to the official style file (e.g., neurips_2026.sty), run the final compile check, and prepare any venue-specific artifacts (reproducibility checklist, camera-ready checklist, broader-impact statement).
4. Optional but recommended before any main-track submission: run the pipeline on one additional distractor-rich benchmark (FEVER, RGB, or MuSiQue) to test whether the sign-flip of log-length and consistency is HotpotQA-specific or general.

**Heads up — nothing blocking, flagged for transparency:**
- No downstream QA experiment. The paper measures retrieval-side selection quality, not answer correctness under the selected paragraphs. Listed as immediate future work; the reviewer did not gate acceptance on it.
- Entity-overlap signal uses a pure-Python regex rather than a trained NER tagger. The paper discloses the regex in Method and acknowledges a spaCy baseline would be preferable for a longer version.

Happy to iterate on any section. Reply with specifics and I'll return a revision.

Best,
Research assistant
