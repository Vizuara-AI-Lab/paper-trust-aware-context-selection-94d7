# Experiment Plan v2

## Revision 2 (changelog)

Iteration 1 produced a clean experimental scaffold (3 seeds, 5 methods, 300 ex/seed, 21-point SBERT vs. random margin) but the proposed uniform-weight TrustScore lost to SBERT by 21.8 absolute P@2 points. Root cause: in HotpotQA distractor, (a) paragraph-length-based authority is non-discriminative across Wikipedia candidates, and (b) cross-paragraph consistency is *anti-*correlated with being gold, because distractors are topically close to the query by construction.

Revision 2 addresses this by:

1. Replacing the length-based authority signal with a **question-to-paragraph entity-overlap** signal (capitalized-token / numeric-token overlap; pure-Python regex; no spaCy dependency).
2. Adding a **Learned-TrustScore** variant: a 6-feature logistic regression fit on a 1/3 train split per seed, with features = [sbert_rel, bm25, tfidf, log_len, consistency, entity_overlap]. Scored with predicted gold probability; evaluated on the remaining 2/3 test split.
3. Keeping **uniform-TrustScore** as an explicit `trust_uniform` ablation condition so the negative result is preserved.
4. Reporting a **score-gap diagnostic** (mean score on gold - mean score on distractor) for every method, exposing *which* features calibrate correctly and which are anti-signals.

## Research directive (revised)

We investigate whether combining orthogonal provenance / authority / consistency signals via a simple learned weighting yields better retrieval of gold supporting paragraphs in HotpotQA's distractor setting than the best single-signal baseline. Hypothesis: a logistic-regression Learned-TrustScore, trained on 1/3 of each 300-example seed with six per-paragraph features, will improve Precision@2 over SBERT by at least 3 absolute points while (a) uniform-weighting continues to *degrade* performance, exposing the importance of signal-calibrated combination and (b) the score-gap diagnostic identifies which individual signals are discriminative vs. anti-correlated.

## Question

Given a question and 10 candidate paragraphs (2 gold + 8 distractors), can a small supervised trust-scoring function, combining six heterogeneous signals, outperform any single-signal retriever? And what does the fitted weight vector reveal about which signals are actually informative in distractor-rich multi-hop QA?

## Dataset + preprocessing

- **HotpotQA** distractor configuration, validation split.
- Seeds `[0, 1, 2]`. Each seed samples 300 examples; of those, 100 train (for Learned-TrustScore fitting) and 200 test (for all metric reporting).
- Drop rare examples whose gold-paragraph count != 2.

## Methods compared

1. **Random** — permute, pick top-2.
2. **BM25** — `rank_bm25.BM25Okapi`.
3. **TF-IDF cosine** — sklearn.
4. **SBERT** — `sentence-transformers/all-MiniLM-L6-v2`.
5. **EntityOverlap** (new) — standalone signal: fraction of question entity tokens found in paragraph.
6. **TrustScore-Uniform (ablation)** — mean of normalized (SBERT, log-length, consistency). Same as iter-1 TrustScore; preserved for negative-result reporting.
7. **TrustScore-Learned (proposed)** — logistic regression over six per-paragraph features; trained per seed on the 100-example train split; scored with predicted gold probability.

## Feature vector (Learned-TrustScore)

Each paragraph gets a 6-dim feature vector, all values min-max normalized within the candidate pool of its question:

- `sbert_rel` — cosine(q, p) with all-MiniLM-L6-v2.
- `bm25` — BM25 score against q.
- `tfidf` — TF-IDF cosine against q.
- `log_len` — log(1 + n_tokens(p)).
- `consistency` — mean SBERT cosine to the other 9 candidates.
- `entity_overlap` — |Q_entities ∩ P_entities| / |Q_entities|, where entities = capitalized n-grams + multi-digit numbers.

## Protocol

- Zero-shot for unsupervised methods.
- Learned-TrustScore trains one logistic-regression classifier per seed on 100 examples x 10 candidates = 1000 paragraph rows (with ~200 positives, 800 negatives).
- Evaluated on the 200-example held-out test split per seed.
- 3 seeds total, aggregated mean/std.

## Evaluation metrics

- **Precision@2** — top-2 picks equal gold pair.
- **Paragraph F1** — F1 over selected vs. gold (2 vs. 2).
- **Supporting-fact F1** — sentence-level F1, top-2 selected.
- **Score gap** — `mean(score | gold) - mean(score | distractor)`, a signal-calibration diagnostic. Positive gap = signal pushes gold up; negative gap = signal is anti-correlated.

## Artifacts

- `metrics_seed{i}.json` — per-seed results for all 7 methods plus LR weights and train/test sizes.
- `summary.json` — aggregated mean/std across seeds plus averaged LR weight vector.
- `RESULT:` / `SUMMARY:` lines for fallback parser.

## Compute budget

- CPU-only. Expected wall-clock: 3-6 min per seed (post-cache) * 3 seeds = 10-18 min total.
- Modal / Runpod backend: `cpu=4.0, memory=8192`. Well inside the $5 session cap.

## Success criterion (iteration 2)

- `trust_learned` P@2 >= SBERT P@2 + 0.03 with non-overlapping std.
- `trust_uniform` continues to underperform SBERT (preserved negative result).
- LR weights + score-gap diagnostics corroborate: sbert_rel weight > 0, consistency weight <= 0, entity_overlap weight > 0.
