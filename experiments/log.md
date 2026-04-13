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

---

## Experiment 4: Multi-class error detection (6 error types)

**Date:** 2026-04-13T14:00:00+02:00

**Goal:** Expand from binary (error/clean) to 6 error categories: spelling, word choice, grammar, word order, missing word, extra/duplicate word. Test whether SAE features can distinguish between error types, not just detect errors vs clean text.

**Hypothesis:** Spelling errors should remain highly detectable (proven in Exp 1-3). Grammar and word choice errors may activate different SAE features (semantic/syntactic vs orthographic). Word order and missing/extra word might require attention-related features from different layers. Expect overall detection F1 lower than 94.2% initially due to harder error types, but expect some type separation in the feature space.

**Parameters:**
- Same layers [5, 10, 13, 17, 22], width 16k
- 300 pairs balanced across 6 types (~50 per type)
- Multi-class token-level LR (7 classes: clean + 6 error types)
- Threshold sweep for binary detection + per-type analysis at best threshold
- New data and features (cache invalidated by new generation logic)

**Results:**
- Feature selection: had to refactor `select_features()` to work per-type (50% threshold per type, then union). Global 50% across 225 pairs was impossible for type-specific features. New result: 43 features (layer 5: 13, layer 10: 12, layer 13: 8, layer 17: 5, layer 22: 5).
- LR convergence warning (1000 iterations hit) — 7-class problem is harder to separate.
- Binary detection (error vs clean) at threshold=0.95: **F1=73.3%, P=62.9%, R=88.0%** — major regression from Exp 3's 94.2%.
- 39 false positives (up from 7) — the multi-class model is much nosier on clean text.
- Per-type detection rates:
  | Type | Total | Detected | Det% | Correct Type | TypeAcc% |
  |------|-------|----------|------|-------------|----------|
  | spelling | 13 | 13 | 100% | 13 | 100% |
  | word_choice | 11 | 9 | 82% | 6 | 67% |
  | grammar | 20 | 18 | 90% | 4 | 22% |
  | word_order | 11 | 9 | 82% | 4 | 44% |
  | missing_word | 9 | 6 | 67% | 1 | 17% |
  | extra_word | 11 | 11 | 100% | 9 | 82% |
- Spelling and extra_word are strong: 100% detection, high type accuracy.
- Grammar is detected (90%) but almost always misclassified (22% type accuracy).
- Missing_word is hardest to detect (67%) and virtually impossible to classify (17%).
- FP analysis: proper nouns, contractions, and unusual names still dominate FPs — same pattern as Exp 3 but amplified because the multi-class model is more trigger-happy.

**Conclusions:**
- Spelling remains the strongest signal — SAE features clearly encode character-level anomalies.
- Extra_word is surprisingly well-detected — duplicated words create a distinctive activation pattern.
- Grammar detection exists but the type classifier confuses it with other types — grammar errors (is→are, a→an) may not produce unique SAE signatures distinct from word choice errors.
- Missing_word is the weakest — absence of a word doesn't produce a distinctive token-level activation (the gap isn't a token).
- The multi-class LR struggles with 7 classes on 43 features with heavy class imbalance (465 error vs 7110 clean tokens). Next steps: more data per type, possibly separate binary classifiers per type instead of a single multi-class model.
- Overall binary detection degraded significantly — the model is now trying to do too much with too little data per type.

**Commit:** 7a8b239

---

## Experiment 5: One-vs-rest binary classifiers per error type

**Date:** 2026-04-13T15:30:00+02:00

**Goal:** Replace the single 7-class LR with 6 independent binary classifiers, one per error type. Each classifier answers "is this token a [type] error?" with its own threshold. This should reduce FP (each classifier is simpler and can be tuned independently) and reveal which error types are genuinely separable in SAE feature space.

**Hypothesis:** Binary classifiers will be easier to fit (2 classes instead of 7) and can each specialize. Spelling and extra_word should improve significantly. Grammar/word_choice may still overlap. Per-type thresholds allow being conservative on weak types (missing_word) and aggressive on strong ones (spelling). Overall binary detection should recover toward Exp 3's F1=94%.

**Parameters:**
- Same cached data and features from Exp 4 (300 pairs, 50 per type)
- Same layers [5, 10, 13, 17, 22], width 16k
- 6 binary LR classifiers (class_weight='balanced'), one per ErrorType
- Per-type threshold sweep: 0.5–0.99
- Report: per-type detection+FP rates, combined binary detection, comparison to Exp 4

**Results:**
- Combined F1=72.0% (Precision=60.4%, Recall=89.3%) — regression from Exp 4 (F1=73.3%)
- Per-type best thresholds (by F1): spelling=0.99, word_choice=0.9, grammar=0.5, word_order=0.99, missing_word=0.5, extra_word=0.6
- Per-type detection at best thresholds:
  - spelling: 92% detected, 8% FP, 92% type accuracy — excellent
  - extra_word: 91% detected, 1.3% FP, 73% type accuracy — excellent
  - word_choice: 82% detected at t=0.9, 23% FP — moderate
  - grammar: 85% detected at t=0.5, 39% FP — high FP, classifier too trigger-happy
  - word_order: 36% detected at t=0.99, 8% FP — weak, high FP at lower thresholds (71% FP at 0.5)
  - missing_word: 89% detected at t=0.5, 76% FP — near-random; the classifier cannot separate this type
- 44 FPs in combined mode. FP analysis: proper nouns (rafael, wertmuller, hong kong), contractions ('s), and subword BPE fragments are main false triggers
- Convergence warning on one classifier (max_iter=2000 insufficient)
- Type accuracy generally poor: word_order 11%, grammar 24% — classifiers fire on wrong types

**Conclusions:**
- OVR approach works very well for **spelling** and **extra_word** (low FP, high detection), confirming these are genuinely separable in SAE space
- **grammar**, **word_order**, and **missing_word** are not currently separable — they produce too many FPs even at high thresholds. These types either need better data (more examples, clearer signal) or operate at different SAE layers/widths
- The combined binary F1 is slightly worse than Exp 4's single multi-class → OVR doesn't help overall, but it provides per-type insight
- Per-type thresholding is valuable: we can ship spelling+extra_word detection confidence and suppress the others
- The F1-optimal threshold selection picks low thresholds for weak classifiers (grammar=0.5, missing_word=0.5), which floods FPs. A better strategy: pick thresholds to keep FP below a budget (e.g., max 5% per type)
- Next steps: FP-constrained threshold selection, better synthetic data (longer/harder examples), or investigate which SAE layers best serve each error type

**Commit:** 78ce733

---

## Experiment 6: Per-type feature sets for OVR classifiers

**Date:** 2026-04-13T16:30:00+02:00

**Goal:** Give each OVR binary classifier its own feature set instead of sharing a global union. Currently all 6 classifiers see the same 43 features (union of per-type selections). This means the missing_word classifier sees spelling-specific features (noise), and vice versa. Per-type feature sets should reduce FP by removing irrelevant features from each classifier's input.

**Hypothesis:** Disjoint per-type feature sets will reduce FP for weak classifiers (grammar, word_order, missing_word) whose signal drowns in features selected for other types. Strong classifiers (spelling, extra_word) should hold steady. May also reveal that some types have very few features (confirming they're not well-represented in SAE space at current resolution).

**Parameters:**
- Same cached data and features from Exp 4 (300 pairs, 50 per type)
- Same layers [5, 10, 13, 17, 22], width 16k
- New: `select_features_per_type()` returns features per ErrorType
- Each binary LR trained only on its own type's features
- Per-type threshold sweep: 0.5–0.99
- Compare against Exp 5 (shared features)

**Results:**
- Per-type feature counts: spelling=30, extra_word=14, word_choice=1, word_order=1, grammar=0, missing_word=0
- Grammar and missing_word have ZERO features → no classifier possible. Confirms these types have no distinctive SAE signal at 16k width.
- Combined F1=68.6% (Precision=73.8%, Recall=64.0%) — regression from Exp 5 (72.0%) due to losing 2 types entirely
- Per-type results:
  - **extra_word**: F1=95.2% (91% det, 0% FP at ANY threshold!) — up from Exp 5's 91% det/1.3% FP. Removing noise features eliminated the last FP.
  - **spelling**: F1=80.0% at t=0.99 (92% det, 6.7% FP) — comparable to Exp 5 (77.4% at 0.99)
  - **word_choice**: F1=47.1% at t=0.5 (73% det, 20% FP, 1 feature) — weak, single feature
  - **word_order**: F1=47.1% at t=0.5 (73% det, 20% FP, 1 feature) — same profile, single feature
  - grammar: no model (0 features)
  - missing_word: no model (0 features)
- Type accuracy dramatically improved where models exist: spelling 100% (was 92%), extra_word 100% (was 73%), word_choice 75% (was 36%)
- FP down from 44 → 17 (61% reduction). Remaining FPs: proper nouns triggering spelling, contractions ('s, n't) triggering word_choice, function words triggering word_choice
- word_choice and word_order share the same single feature (L13) pushing identical FP rates

**Conclusions:**
- Per-type features are a clear win for **extra_word** (perfect 0% FP) and **type accuracy** (100% for both strong types)
- The hypothesis about noise features hurting weak classifiers was partially wrong: grammar and missing_word don't have ANY features, so the issue is not noise — it's that these error types don't produce distinctive SAE activations at 16k width
- The 43 "shared" features from Exp 5 were 70% spelling features. When those leaked into other classifiers, they produced FPs. Isolation fixed that.
- For a practical 2-type detector (spelling + extra_word), per-type features are strictly better
- To rescue grammar/missing_word: need wider SAEs (65k/262k), different layers, or fundamentally different data
- Overall binary F1 dropped because recall tanked (64% vs 89%) — we simply can't detect grammar/missing_word at all now

**Commit:** 5f1faeb
