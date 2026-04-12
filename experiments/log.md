# Experiment Log

All experiments are logged here in chronological order. Append-only — never delete or retroactively modify entries (except adding the git commit hash after committing).

---

## Experiment 1: Scaled synthetic data with character-level errors

**Date:** 2026-04-12T22:00:00+02:00

**Goal:** Scale from 8 hand-crafted pairs to ~300 synthetic pairs using SST2 sentences corrupted with character-level errors. Test whether more training data produces a more robust classifier with fewer false positives.

**Hypothesis:** With ~225 training pairs (vs 6), the classifier should find more generalizable error-indicator features. Precision should improve significantly while maintaining high recall, because features that were error-only by chance in a small training set will be filtered out.

**Parameters:**
- Source: SST2 sentences, filtered to 8–20 words, ~300 pairs
- Error types: adjacent char swap, char deletion, char insertion (2–4 corrupted words per sentence)
- Layers: 5, 10, 13, 17, 22
- SAE width: 16k
- Train/test split: 75/25, seed=42
- min_pair_ratio: 0.5

**Results:**
- 300 pairs generated (225 train, 75 test)
- 41 error features total across 5 layers (9/9/10/8/5 per layer 5/10/13/17/22)
- Confusion matrix: TP=75, FP=64, TN=11, FN=0
- **Accuracy=57.3%, Precision=54.0%, Recall=100%, F1=70.1%**
- Compared to baseline (8 pairs): features dropped from 204→41 (more selective), F1 improved 66.7%→70.1%
- Precision barely improved (50%→54%) — with threshold=1, almost any clean text triggers at least one error feature
- Recall remains perfect (100%) — all error texts detected

**Conclusions:**
- Scaling data helps: feature set is more selective (41 vs 204), confirming baseline had many spurious features
- The fundamental bottleneck is now the threshold: `min_active_features=1` is too permissive for clean texts
- Next steps: raise the threshold (require more features to fire), or weight features by activation strength, or use a proper classifier on top of the feature hits

**Commit:** 05ede17
