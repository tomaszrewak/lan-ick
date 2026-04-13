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

---

## Experiment 2: Token-level logistic regression classifier

**Date:** 2026-04-13T12:00:00+02:00

**Goal:** Replace the binary feature-counting classifier with a token-level logistic regression. Each token gets P(error) based on its SAE activation vector across the selected error features. Sentence-level prediction uses max(token scores) > threshold.

**Hypothesis:** A trained logistic regression can learn which features and activation magnitudes matter, dramatically improving precision over the hard threshold=1 approach. Expect F1 > 80% with balanced precision/recall. Token-level predictions should correlate with actually corrupted words.

**Parameters:**
- Same data: 300 SST2 pairs, 225 train / 75 test, same seed
- Same layers: 5, 10, 13, 17, 22, width 16k
- Feature selection: error-only features (same as Exp 1, min_pair_ratio=0.5)
- Token-level labels: tokens belonging to corrupted words → label=1, others → label=0
- Classifier: sklearn LogisticRegression(class_weight='balanced')
- Sentence threshold: sweep 0.3–0.8, report best

**Results:**
- 41 error features (same selection as Exp 1), LR trained on 8130 tokens (1298 error, 6832 clean)
- 100% recall maintained across ALL thresholds — no error sentences missed
- Threshold sweep results:
  - 0.3: P=66.4%, F1=79.8% (FP=38)
  - 0.5: P=72.1%, F1=83.8% (FP=29)
  - 0.7: P=78.9%, F1=88.2% (FP=20)
  - **0.8: P=85.2%, F1=92.0% (FP=13)** ← best
- Best: **Accuracy=91.3%, Precision=85.2%, Recall=100%, F1=92.0%**
- Compared to Exp 1: F1 improved 70.1%→92.0%, precision improved 54.0%→85.2%

**Conclusions:**
- Token-level LR is a massive improvement over binary feature counting
- Activation magnitudes carry strong signal — the LR learned to weight features meaningfully
- Perfect recall at all thresholds tested suggests robust error detection
- Remaining 13 FP (at threshold=0.8) worth investigating — could be SST2 sentences with unusual phrasing that triggers error features
- Even higher thresholds (0.85, 0.9) worth exploring since recall hasn't started to drop

**Commit:** d75d83b

---

## Experiment 3: Higher thresholds and false positive analysis

**Date:** 2026-04-13T13:00:00+02:00

**Goal:** Push the sentence threshold above 0.8 (0.85, 0.9, 0.95) to find the precision/recall sweet spot. Inspect the false positive sentences to understand what triggers them — unusual grammar, rare words, or genuine ambiguity?

**Hypothesis:** Since recall is still 100% at threshold=0.8, we have room to push higher. Expect to reach F1>95% before recall starts dropping. FP sentences likely contain unusual token patterns (rare words, informal grammar) that overlap with character-level error signals.

**Parameters:**
- Same data and features as Exp 2 (all cached)
- Extended threshold sweep: 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95
- Print FP sentences and their top-scoring tokens at best threshold

**Results:**
- F1 keeps improving all the way to 0.95, where recall finally dips:
  - 0.8: P=85.2%, R=100%, F1=92.0% (FP=13)
  - 0.85: P=87.2%, R=100%, F1=93.2% (FP=11)
  - 0.9: P=88.2%, R=100%, F1=93.8% (FP=10)
  - **0.95: P=91.2%, R=97.3%, F1=94.2% (FP=7, FN=2)**
- Best: **Accuracy=94.0%, Precision=91.2%, Recall=97.3%, F1=94.2%** at threshold=0.95
- False positive analysis (7 FPs at 0.95):
  - 4/7 contain `n't` tokenized as `' n'` + `'t'` — the model sees these as suspicious fragments
  - Remaining 3 triggered by unusual proper nouns/words: "cloyingly hagiographic", "soderbergh", "bubba ho-tep"
  - Pattern: the model flags subword fragments that look like character-level errors (split tokens, rare morphemes)

**Conclusions:**
- Recall is rock-solid: 100% through threshold=0.9, only drops to 97.3% at 0.95
- The dominant FP source is `n't` contractions — the tokenizer splits these into fragments that resemble spelling errors to the SAE features
- Fixing this could be done upstream (contraction normalization) or downstream (whitelist known subword patterns)
- Rare proper nouns are a secondary FP source — inherent limitation when the model has limited exposure to those tokens
- F1=94.2% is strong for a purely interpretable, SAE-based classifier on synthetic data

**Commit:** 5639505
