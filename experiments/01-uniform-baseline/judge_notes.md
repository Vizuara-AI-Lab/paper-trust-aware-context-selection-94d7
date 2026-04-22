# Iteration 1 — TrustScore-Uniform (not worthy)

**Verdict:** not worthy (score 6/10).

## Primary result

Uniform-weight TrustScore P@2 = 0.094 (3 seeds, 300 examples each);
SBERT baseline = 0.312. Hypothesis that a uniformly-weighted combination
of relevance + log-length + consistency would improve over SBERT was
falsified by 21.8 absolute points in the wrong direction.

## Diagnosis

- Paragraph-length-based authority is non-discriminative across
  Wikipedia candidates of similar length.
- Cross-paragraph consistency is anti-correlated with gold because
  HotpotQA distractors are selected to be topically close to the query.

## Feedback carried to iteration 2

1. Add a Learned-TrustScore variant with supervised weighting.
2. Replace length-based authority with a discriminative entity-overlap signal.
3. Keep TrustScore-Uniform as an explicit ablation condition.
4. Add a score-gap calibration diagnostic.

See `experiments/02-learned-trustscore/` for the revised script.
