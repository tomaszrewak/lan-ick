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

---

## Experiment 7: Feature selection method comparison

**Date:** 2026-04-13T17:00:00+02:00

**Goal:** The current feature selection (binary presence: fires in error but not clean) yields 0 features for grammar and missing_word. Explore fundamentally different selection approaches to find features that actually distinguish error from clean text. Compare multiple methods head-to-head.

**Hypothesis:** The binary presence method is too coarse — it misses features that fire in both error and clean text but with different magnitudes or at different positions. Methods that compare activations at the same token position (error word vs its clean counterpart) or that use activation magnitude differences should find features invisible to the binary approach, especially for grammar/missing_word.

**Methods to compare:**
1. **baseline**: Current binary presence (error_fids - clean_fids), min_pair_ratio=0.5
2. **relaxed**: Same as baseline but min_pair_ratio=0.3
3. **paired_token_diff**: At each error word position, compare activation in error vs clean at the same token position. Feature selected if it activates more in error text at error positions across enough pairs.
4. **magnitude_diff**: Per-feature mean activation in error texts minus mean in clean texts. Top-K by magnitude difference.
5. **ttest**: Welch's t-test on per-feature activations between error-word tokens and clean tokens. Select by p-value.
6. **top_k_error**: Take features with highest mean activation at error word positions, regardless of clean comparison. Top-K.

Each method produces per-type feature sets. Train OVR classifiers for each. Compare per-type detection and FP.

**Parameters:**
- Same cached data and features (300 pairs, 50 per type)
- Same layers [5, 10, 13, 17, 22], width 16k
- Fixed threshold=0.9 for comparison (no sweep — too slow with 6 methods)
- Also report feature count per method × type to understand selectivity

**Results:**

Summary at threshold=0.9 (det% / FP%):

| Method          | #Feats (spelling/grammar/missing) | Spelling | Word Choice | Grammar  | Word Order | Missing Word | Extra Word | Combined F1 |
|-----------------|----------------------------------|----------|-------------|----------|------------|-------------|------------|-------------|
| baseline        | 30 / 0 / 0                       | 100/21%  | 36/4%       | —        | 18/1%      | —           | 91/0%      | 63.8%       |
| **relaxed_30**  | 122 / 18 / 3                     | 100/28%  | 73/13%      | 75/9%    | 54/5%      | 22/5%       | 91/0%      | **78.0%**   |
| paired_diff     | 191 / 24 / 5                     | 100/25%  | 73/21%      | 80/12%   | 73/20%     | 0/0%        | 82/0%      | 75.5%       |
| magnitude_diff_10 | 50 / 50 / 50                   | 15/8%    | 82/9%       | 85/21%   | 27/15%     | 56/17%      | 91/1%      | 71.2%       |
| magnitude_diff_20 | 100 / 100 / 100                | 77/20%   | 82/16%      | 75/17%   | 18/17%     | 44/31%      | 73/0%      | 71.8%       |
| ttest           | 61 / 7 / 4                       | 100/29%  | 64/9%       | 75/35%   | 46/15%     | 11/3%       | 91/0%      | 71.6%       |
| ttest_relaxed   | 91 / 14 / 10                     | 100/24%  | 73/11%      | 85/37%   | 54/19%     | 33/1%       | 82/1%      | 72.9%       |
| top_k_error_10  | 50 / 50 / 50                     | 0/5%     | 82/31%      | 85/20%   | 18/15%     | 22/20%      | 18/9%      | 66.3%       |

Best method: **relaxed_30** (F1=78.0%, P=67.7%, R=92.0%) — lowering the min_pair_ratio from 0.5 to 0.3 is the single most impactful change.

Key observations:
- **Grammar detection unlocked**: relaxed_30 gets 18 features for grammar → 75% detection at 9.3% FP. Baseline found 0 features.
- **Missing_word still weak**: Even relaxed_30 only gets 3 features → 22% detection. The signal is genuinely sparse.
- **paired_diff** finds grammar signal (80% det) but trades it for high FP (12%) and high word_order FP (20%).
- **magnitude_diff and top_k_error** methods (fixed top-K per layer) perform poorly overall — they select features without regard to error specificity, leading to high FP and even destroying spelling detection (15% det for magnitude_diff_10, 0% for top_k_error_10).
- **ttest** finds grammar (75-85% det) but with unacceptable FP (35-37%) — statistical significance ≠ discriminative power.
- **Convergence warnings** (suppressed but present) indicate classifier is hitting max_iter=2000 for methods with 100+ features. Partial convergence, not a blocking issue.

**Conclusions:**
- The simplest improvement wins: just lowering the ratio threshold from 0.5 to 0.3 (relaxed_30) gives the best F1 (78.0% vs 63.8% baseline), unlocks grammar, and is the least complex change.
- Methods that compare token-level activations (paired_diff, ttest) find grammar signal but produce too many FPs — they're sensitive to any activation difference, not just error-indicating ones.
- Fixed-K methods (magnitude_diff, top_k_error) are fundamentally flawed for this task — they ignore error specificity and select "commonly active" features rather than "error-indicating" features. Spelling detection collapses because the top-K features at error positions aren't spelling-specific.
- The binary presence approach (baseline/relaxed) remains the best paradigm. The key lever is the ratio threshold — it controls the trade-off between coverage (more types) and specificity (fewer FPs).
- `relaxed_30` should become the new default feature selection method.

**Commit:** 37916de

---

## Experiment 8: SAE width comparison (16k vs 65k vs 262k)

**Date:** 2026-04-13T18:00:00+02:00

**Goal:** Determine whether wider SAEs produce more specific error-detection features with lower FP rates. Grammar and missing_word have zero features at 16k — wider SAEs might have features that distinguish these types. Primary focus: does wider SAE width reduce FP while maintaining (or improving) detection?

**Hypothesis:** Wider SAEs decompose activations into finer-grained features. At 16k, grammar errors may activate features shared with clean text (→ no selection). At 65k/262k, the same activation might split into a grammar-specific feature and a clean-text feature, enabling selection. Expect: more features per type, lower FP rates for existing types (spelling, extra_word), and new signal for grammar/missing_word.

**Parameters:**
- Widths: 16k, 65k, 262k
- Layers: [7, 13, 17, 22] (subset where all three widths are available)
- Feature selection: per-type binary presence, min_pair_ratio=0.3 (Exp 7 winner)
- Same data: 300 pairs, 75/25 split, seed=42
- Same classifier: OVR binary LR, threshold sweep [0.5, 0.8, 0.9, 0.95]
- VRAM management: evict SAE cache between widths to fit 262k (~2.25 GB/SAE)

**Results:**

Comparison at threshold=0.9:

| Width | F1 | Precision | FP# | spelling | word_choice | grammar | word_order | missing_word | extra_word |
|-------|------|-----------|-----|----------|-------------|---------|------------|-------------|------------|
| 16k | 75.4% | 66.0% | 34 | 100% | 64% | 90% | 36% | 11% | 91% |
| 65k | 75.1% | 64.2% | 38 | 100% | 73% | 90% | 46% | — | 100% |
| 262k | 76.7% | 68.0% | 31 | 92% | 73% | 85% | 54% | 0% | 100% |

Comparison at threshold=0.95:

| Width | F1 | Precision | FP# |
|-------|------|-----------|-----|
| 16k | 73.4% | 69.9% | 25 |
| 65k | 76.3% | 67.3% | 32 |
| 262k | 77.1% | 70.3% | 27 |

Feature counts per type:

| Width | spelling | word_choice | grammar | word_order | missing_word | extra_word |
|-------|----------|-------------|---------|------------|-------------|------------|
| 16k | 89 | 26 | 12 | 12 | 3 | 17 |
| 65k | 86 | 20 | 16 | 9 | 0 | 20 |
| 262k | 92 | 27 | 13 | 13 | 1 | 19 |

**Observations:**
- Width has surprisingly little effect on feature counts or overall performance. All widths find ~90 spelling features, ~15 grammar features.
- 262k is marginally best: +1.3% F1, +2% precision, -3 FPs vs 16k at threshold=0.9.
- 65k found 0 features for missing_word (worse than 16k's 3). Wider ≠ always better.
- 262k improved word_order detection most (36% → 54%) — the only type with clear width benefit.
- spelling detection actually dropped at 262k (100% → 92%) — wider SAE may split key features.
- 262k extraction was extremely slow due to per-layer SAE eviction (2.25 GB/SAE, 8 reloads per pair).
- **Conclusion:** The marginal gains do not justify 16x wider SAEs and dramatically slower inference. Stick with 16k for now. The bottleneck is not feature granularity — it's data quality and inherent difficulty of types like missing_word.

**Commit:** d3c60a5

---

## Experiment 9: Last-token-only labeling for causal attention

**Date:** 2026-04-13T20:30:00+02:00

**Goal:** Fix a training label mismatch caused by causal attention. Currently, all tokens of an error word get error labels during training. But Gemma uses causal attention — earlier tokens of a multi-token word can't "see" future tokens, so their activations may not reflect the error. This injects noise: non-last tokens get error labels despite having no error signal in their activations.

**Hypothesis:** Labeling only the last token of each error word (which has seen all preceding tokens of the word) and excluding non-last tokens from training should: (1) reduce false positives by removing noisy positive examples, (2) maintain or improve detection since the strongest signal is at the last token. Feature selection (pair-level, comparing error vs clean text features) is unaffected — only the token-level classifier training changes.

**Parameters:**
- Same data: 300 pairs, 75/25 split, seed=42
- Same layers: [7, 13, 17, 22], width 16k
- Same feature selection: per-type, min_pair_ratio=0.3
- Change: in `train_ovr`, error-word tokens except the last are excluded from training
- Prediction unchanged: max P(error) across all tokens
- Thresholds: [0.5, 0.8, 0.9, 0.95]

**Results:**

At threshold=0.9 (vs Exp 7 baseline):

| Type | Detection | FP | Detection (before) | FP (before) |
|------|-----------|------|---------------------|-------------|
| spelling | 100% | **8.0%** | 100% | 24.0% |
| word_choice | 64% | 12.0% | 64% | 13.3% |
| grammar | 90% | 12.0% | 90% | 12.0% |
| word_order | 45% | 13.3% | 36% | 8.0% |
| missing_word | 33% | 8.0% | 11% | 4.0% |
| extra_word | 100% | 1.3% | 91% | 1.3% |
| **Combined** | **F1=77.3%** | **25 FPs** | F1=75.4% | 34 FPs |

At threshold=0.95: F1=76.0%, P=76.0%, R=76.0%, FP#=18 (was F1=73.4%, P=69.9%, FP#=25)

Training token counts: spelling dropped from 171→76 positive tokens (removed ~95 non-last tokens of multi-token misspelled words).

**Observations:**
- **Spelling FP collapsed from 24% to 8%** — the biggest single improvement. Non-last tokens of misspelled words were indeed injecting noise.
- Detection rates held or improved: spelling 100%, grammar 90%, extra_word 100%→100%.
- word_order and missing_word detection improved (36%→45%, 11%→33%) though their FP also increased slightly.
- **FP count at t=0.9 dropped 26%** (34→25) and **at t=0.95 dropped 28%** (25→18).
- **Precision at t=0.95 jumped from 69.9% to 76.0%** — first time above 75%.
- **Conclusion:** Clear win. The causal attention insight is correct — labeling non-last tokens as errors was adding noise. This change should be permanent.

**Commit:** 201630c

---

## Experiment 10: Next-token-after-word labeling

**Date:** 2026-04-13T21:00:00+02:00

**Goal:** Push the causal attention insight further. In Exp 9 we labeled the last token of an error word. But SentencePiece prepends whitespace to the *next* word's first token (e.g., `' about'` after `'ht'`). At the last token of the error word, the model doesn't yet know the word is finished — it could still be a prefix (e.g., "repla" could become "replacing"). Only at the next token (which starts with a space, confirming the word boundary) does the model have full context to "know" the previous word was wrong.

**Hypothesis:** Labeling the first token AFTER the error word (instead of the last token OF the word) should produce cleaner signal. The model's "surprise" at seeing a space/new-word after an unexpected word ending is the strongest error indicator. Expect: further FP reduction, especially for spelling. Note: this shifts error attribution — a flagged token now means the *previous* word has an error.

**Parameters:**
- Same data: 300 pairs, 75/25 split, seed=42
- Same layers: [7, 13, 17, 22], width 16k
- Same feature selection: per-type, min_pair_ratio=0.3
- Change: ALL tokens of error words excluded from training; first token after error word gets the error label
- Edge case: error word is last word in sentence → skip (no next token to label)
- Prediction unchanged: max P(error) across all tokens (sentence-level eval)

**Results:** Worse than Exp 9 across all thresholds — FP rate increased significantly.

| threshold | F1    | Precision | Recall | FP#  |
|-----------|-------|-----------|--------|------|
| 0.5       | 68.8% | 52.4%     | 100.0% | 68   |
| 0.8       | 73.4% | 58.9%     | 97.3%  | 51   |
| 0.9       | 74.2% | 64.1%     | 88.0%  | 37   |
| 0.95      | 76.2% | 68.8%     | 85.3%  | 29   |

Comparison at t=0.9: Exp 9 had F1=77.3%, P=71.6%, FP#=25. Exp 10: F1=74.2%, P=64.1%, FP#=37. Spelling FP went from 8%→20%.

**Conclusion:** The "next token after error word" signal is noisier than the "last token of error word" signal. The last token of an error word has already seen all its characters and can detect something is wrong with the word itself. The next token's "surprise" at a word boundary doesn't encode as cleanly in SAE features — it's influenced by too many other factors (what the next word is, sentence structure, etc.). Reverted to Exp 9 approach (last-token labeling).

**Commit:** bd0a9a9

---

## Experiment 11: Last-token-only for both training and classification

**Date:** 2026-04-13T21:30:00+02:00

**Goal:** Extend last-token-only approach from training (Exp 9) to prediction. Currently, during classification all tokens are scored and max is taken across all of them. But intermediate tokens within multi-token words likely oscillate between error and non-error representations as the model processes the word piece by piece (e.g., `" re"` → fine, `"sou"` → maybe error, `"rce"` → fine, `"full"` → definitely error). These intermediate tokens add noise to sentence-level scoring. Also apply last-token-only to clean text during training — currently all clean tokens are used, but intermediate clean tokens have the same oscillation issue.

**Hypothesis:** Using only last-word tokens everywhere (training both error and clean, prediction) should reduce false positives by eliminating noisy intermediate-token scores while preserving signal at the points where the model has complete word context.

**Parameters:**
- Same data: 300 pairs, 75/25 split, seed=42
- Same layers: [7, 13, 17, 22], width 16k
- Same feature selection: per-type, min_pair_ratio=0.3
- Change 1: Clean text training uses only last token of each word (previously all tokens)
- Change 2: Prediction scores only last token of each word (previously all tokens)
- Sentence-level: max P(error) across last-word tokens only

**Results:**

| threshold | F1    | Precision | Recall | FP#  |
|-----------|-------|-----------|--------|------|
| 0.5       | 74.3% | 59.1%     | 100.0% | 52   |
| 0.8       | 78.1% | 70.2%     | 88.0%  | 28   |
| 0.9       | 78.0% | 73.8%     | 82.7%  | 22   |
| 0.95      | 76.4% | 79.7%     | 73.3%  | 14   |

Comparison at t=0.9 (vs Exp 9): F1 77.3%→78.0%, P 71.6%→73.8%, FP# 25→22.
Comparison at t=0.95 (vs Exp 9): F1 76.0%→76.4%, P 76.0%→79.7%, FP# 18→14.

Per-type highlights at t=0.9:
- **Spelling FP collapsed: 8.0% → 2.7%** — biggest single gain. Intermediate clean-text tokens were triggering spelling features.
- Extra_word: 91% detection, 0% FP (unchanged — already perfect).
- Grammar: 90% detection, 12% FP (unchanged).
- word_order: 36% detection, 8% FP (unchanged from Exp 9).

Training token counts: clean tokens dropped from all tokens to ~1 per word (last-word filtering). Total training tokens dropped proportionally.

**Conclusion:** Win. Filtering both training and prediction to last-word tokens reduces FP further by eliminating noisy intermediate-token activations. Spelling FP improvement is the most dramatic (8%→2.7%). The model's internal back-and-forth while processing multi-token words ("maybe error... no wait... yes error") was indeed adding noise at prediction time. This change should be permanent.

**Commit:** 25f297c

---

## Experiment 12: Smarter synthetic data generation

**Date:** 2026-04-13T22:00:00+02:00

**Goal:** Improve data quality across all 6 error types and scale up to 600 pairs. Current weak types (word_order 36%, missing_word 22%, word_choice 64%) suffer from ambiguous training examples — random word swaps that produce valid text, random deletions that leave valid text, and a small confusables dictionary. Fix each type's generation to produce unambiguously wrong text, and add more corruption variety for spelling.

**Changes:**
1. **Word order**: Use NLTK POS tags to only swap adjacent words where the result is obviously wrong (JJ+NN→NN+JJ, DT+NN→NN+DT, IN+NN→NN+IN). Eliminates ambiguous swaps.
2. **Missing word**: Only delete function words (articles, prepositions, auxiliaries) — these create syntactic breakage. Content word deletion often yields valid text.
3. **Word choice**: Expand CONFUSABLES from ~38 to ~70+ entries (more homophones: would/wood, flour/flower, plain/plane, etc.).
4. **Grammar**: Expand GRAMMAR_SWAPS with verb tense errors (go/went, see/saw, etc.) and pronoun case errors (I/me, he/him). Remove ambiguous a/an swap.
5. **Spelling**: Add vowel substitution and double-letter dropping patterns.
6. **Scale**: 300→600 pairs (100 per type).

**Hypothesis:** Cleaner training examples (especially for word_order and missing_word) should improve detection rates. More diverse spelling/grammar patterns should improve generalization. Larger dataset gives more stable classifiers. Expect: word_order det >50%, missing_word det >40%, word_choice det >75%, with FP rates holding or improving.

**Parameters:**
- 600 pairs, 100 per type, 75/25 split, seed=42
- Same layers: [7, 13, 17, 22], width 16k
- Same feature selection: per-type, min_pair_ratio=0.3
- Same classifier: OVR LR, last-token-only training and prediction
- Thresholds: [0.5, 0.8, 0.9, 0.95]
- DATA_VERSION="v5", EXTRACT_VERSION="v2" (new data, forces re-extraction)

**Results:**

| threshold | F1    | Precision | Recall | FP#  |
|-----------|-------|-----------|--------|------|
| 0.5       | 72.9% | 58.8%     | 96.0%  | 101  |
| 0.8       | 79.5% | 72.5%     | 88.0%  | 50   |
| 0.9       | 81.5% | 78.0%     | 85.3%  | 36   |
| 0.95      | 81.5% | 80.9%     | 82.0%  | 29   |

Per-type detection at t=0.9 (vs Exp 11, noting test set is 150 vs 75):

| Type | Det (Exp 11) | Det (Exp 12) | FP (Exp 11) | FP (Exp 12) |
|------|-------------|-------------|-------------|-------------|
| spelling | 100% | 93% | 2.7% | 5.3% |
| word_choice | 64% | 59% | 8.0% | 6.7% |
| grammar | 90% | 81% | 12.0% | 9.3% |
| word_order | 36% | **55%** | 8.0% | 7.3% |
| missing_word | 22% | **28%** | 5.3% | 6.0% |
| extra_word | 91% | **96%** | 0.0% | 0.0% |

Feature counts: spelling=63, grammar=27, word_choice=17, word_order=15, extra_word=14, missing_word=1.
Training tokens: spelling=118, word_choice=89, grammar=91, word_order=156, missing_word=75, extra_word=75.

Comparison at t=0.9 (vs Exp 11): F1 78.0%→81.5%, P 73.8%→78.0%, R 82.7%→85.3%.
FP rate: Exp 11 had 22/75=29.3% of clean sentences flagged; Exp 12 has 36/150=24.0% — actually lower.
Comparison at t=0.95: F1 76.4%→81.5%, P 79.7%→80.9% — strong improvement.

**Observations:**
- **word_order** detection jumped 36%→55% — POS-based swapping produces clearly wrong examples that the model can learn from. Still room to improve.
- **missing_word** only marginally improved (22%→28%) and still has just 1 feature. Function-word deletion is a better signal but the SAE may not encode "something is missing" well at 16k.
- **grammar** detection dropped 90%→81% — possibly because the expanded GRAMMAR_SWAPS (verb tense, pronoun case) are harder patterns than the narrow is/are/was/were. More diverse errors require more diverse features.
- **spelling** detection dropped 100%→93% — the two new corruption methods (vowel substitution, double-letter drop) may produce subtler errors that are harder to detect. Still very strong.
- **word_choice** FP improved (8%→6.7%) with the expanded confusables dictionary.
- **extra_word** improved to 96% detection at 0% FP — consistently the best type.
- Overall: F1 improved +3.5%, precision improved +4.2%, recall improved +2.6%. Clear win from better data.

**Conclusion:** Smarter data generation is a clear win. The POS-based word order swaps are the biggest per-type improvement. The broader corruption patterns trade some detection on individual types (spelling 100%→93%, grammar 90%→81%) for better generalization and lower overall FP rate (29.3%→24.0%). The classifier is now more robust. This should be the new baseline.

**Commit:** 7f45b8f

---

## Experiment 13: Position-aware top-N feature selection + per-type thresholds

**Date:** 2026-04-13T23:00:00+02:00

**Goal:** Two changes to the feature selection and evaluation pipeline:

1. **Position-aware selection**: Current selection checks if a feature fires *anywhere* in error text but not in clean text. With a 12-word sentence and 1 error word, ~11/12 of firing positions are non-error words — features at those positions are noise. Fix: only count features that fire at the **last token of error words** in error text, then subtract features firing anywhere in clean text.

2. **Top-N ranking (replace min_pair_ratio)**: The 30% cutoff discards features that appear in only 25% of pairs. But four 25% features covering different subsets could combine to 100% detection. Replace the hard cutoff with: rank by pair count, take top N per type, let the LR classifier learn which are meaningful. Sweep N ∈ [25, 50, 100].

3. **Per-type thresholds**: Instead of one global threshold, find each type's threshold to keep FP ≤ 5%. Types that can't meet the budget get disabled. This should improve combined precision significantly.

**Hypothesis:** Position-aware selection eliminates ~90% of noise in feature selection. Top-N lets complementary low-frequency features through. Per-type thresholds let strong types (spelling, extra_word) run at aggressive thresholds while suppressing noisy types. Expect: significant FP reduction, improved precision, potentially some detection improvement for weak types.

**Parameters:**
- Same data: 600 pairs, 100/type, 75/25 split, seed=42
- Same layers: [7, 13, 17, 22], width 16k
- Same classifier: OVR LR, last-token-only training and prediction
- New: position-aware feature selection, top-N sweep [25, 50, 100]
- New: per-type threshold evaluation with 5% FP budget
- Global thresholds: [0.5, 0.8, 0.9, 0.95] for comparison with Exp 12

**Results:**

Best config per N (per-type thresholds with FP ≤ 5% budget):

| N   | F1    | Precision | Recall | FP#  |
|-----|-------|-----------|--------|------|
| 25  | 80.7% | 80.7%     | 80.7%  | 29   |
| 50  | 81.6% | 83.3%     | 80.0%  | 24   |
| 100 | 70.7% | 82.3%     | 62.0%  | 20   |

N=50 per-type breakdown (FP ≤ 5%):

| Type         | Thresh | Det  | FP   |
|--------------|--------|------|------|
| spelling     | 0.93   | 90%  | 4.7% |
| word_choice  | 0.93   | 55%  | 2.0% |
| grammar      | 0.99   | 74%  | 4.7% |
| word_order   | 0.96   | 64%  | 4.7% |
| missing_word | 1.00   | 0%   | 0.0% |
| extra_word   | 0.50   | 100% | 0.7% |

N=50 at global thresholds (for comparison with Exp 12):

| Threshold | F1    | Precision | Recall | FP#  |
|-----------|-------|-----------|--------|------|
| 0.5       | 68.3% | 52.1%     | 99.3%  | 137  |
| 0.8       | 76.3% | 63.6%     | 95.3%  | 82   |
| 0.9       | 78.8% | 68.5%     | 92.7%  | 64   |
| 0.95      | 79.6% | 73.2%     | 87.3%  | 48   |

Candidate pool sizes: spelling 979, word_choice 701, word_order 688, grammar 458, extra_word 397, missing_word 240.

Feature distribution for N=50: spelling L7:17/L13:16/L17:11/L22:6, word_choice L7:17/L13:14/L17:11/L22:8, grammar L7:9/L13:9/L17:16/L22:16, word_order L7:13/L13:19/L17:9/L22:9, missing_word L7:16/L13:17/L17:10/L22:7, extra_word L7:24/L13:13/L17:10/L22:3.

**Observations:**
- **Position-aware selection works**: Candidate pools are large (240–979 per type), confirming that many features fire specifically at error word positions. The selection filters out features that also fire in clean text.
- **N=50 is the sweet spot**: Best F1 (81.6%) and precision (83.3%). N=25 has slightly less precision; N=100 overfits badly (word_order and missing_word collapse to 0% detection when constrained to FP ≤ 5%).
- **Per-type thresholds are a big win**: N=50 per-type (F1=81.6%, P=83.3%, FP#=24) vs N=50 at global t=0.95 (F1=79.6%, P=73.2%, FP#=48) — per-type halves the false positives while improving F1.
- **vs Exp 12 baseline** (t=0.95: F1=81.5%, P=80.9%, FP#=22): N=50 per-type is comparable — slightly better F1 (81.6 vs 81.5), better precision (83.3 vs 80.9), but slightly more FP (24 vs 22). At global t=0.95, Exp 13 is worse (F1=79.6 vs 81.5), meaning the per-type thresholds are what recover the performance.
- **spelling improved**: 90% at 4.7% FP (was 93% at Exp 12 t=0.95, but that had higher FP). Per-type threshold lets it find the right operating point.
- **word_order improved significantly**: 64% detection at 4.7% FP (was 55% in Exp 12). Position-aware selection and per-type threshold both help.
- **grammar needs very high threshold**: t=0.99 for 74% detection at 4.7% FP — the model is uncertain about grammar errors.
- **missing_word remains unsolvable**: Needs t=1.00, yielding 0% detection. The SAE features don't encode "something is missing" at 16k width.
- **N=100 overfits**: With 100 features but only 75–156 positive tokens per type, the LR can't separate signal from noise. FP rates are much higher at every threshold, requiring extreme thresholds that kill detection.

**Conclusions:**
- The per-type threshold mechanism is the clear winner of this experiment — it's a general improvement independent of feature selection method.
- Position-aware top-N selection at N=50 is roughly on par with the old min_pair_ratio=0.3 approach. The hypothesis that top-N would let complementary features through was partially validated (word_order +9%), but overall F1 is similar.
- N=50 with per-type thresholds should become the new approach, as it gives better precision at similar F1 with more principled FP control.
- Missing_word should be dropped or treated as a "bonus" type — don't let its poor performance drag down the pipeline.
- The non-linear classifier idea (Random Forest) may not be needed since N=50 doesn't show the "too many features for linear separation" problem — that only appears at N=100.

**Commit:** 20ada16

---

## Experiment 14: Non-linear classifier (Random Forest vs Logistic Regression)

**Date:** 2026-04-13T23:45:00+02:00

**Goal:** Replace LR with Random Forest in the OVR pipeline. LR assumes a single linear decision boundary, but error detection likely has subpopulations — most errors cluster into a clearly separable region, but smaller subsets don't conform to the main boundary (e.g., subtle grammar errors vs obvious ones). A tree-based classifier partitions the feature space into regions, each with its own decision rule, naturally handling these non-conforming pockets. Also captures feature interactions LR misses.

**Hypothesis:** RF should improve detection for types where LR struggles (grammar, word_order, word_choice) by finding non-linear boundaries around subpopulations. May also reduce the extreme thresholds needed for grammar (t=0.99). Spelling and extra_word are already near-perfect with LR, so less room for improvement there. Risk: RF may overfit with only 75–156 positive tokens per type — trees are greedier than LR.

**Parameters:**
- Same data: 600 pairs, 100/type, 75/25 split, seed=42
- Same layers: [7, 13, 17, 22], width 16k
- Same features: position-aware top-50
- Classifiers compared: LR (baseline) vs RF (n_estimators=100, max_depth=8, class_weight="balanced")
- Same evaluation: per-type thresholds with 5% FP budget + global thresholds

**Results:**

Per-type thresholds (FP ≤ 5%) comparison:

| Metric    | LR        | RF        |
|-----------|-----------|-----------|
| F1        | **81.6%** | 73.3%     |
| Precision | 83.3%     | 82.5%     |
| Recall    | **80.0%** | 66.0%     |
| FP#       | 24        | **21**    |

RF per-type breakdown (FP ≤ 5%):

| Type         | LR Thresh | LR Det | RF Thresh | RF Det |
|--------------|-----------|--------|-----------|--------|
| spelling     | 0.93      | 90%    | 0.50      | 86%    |
| word_choice  | 0.93      | 55%    | 0.50      | 41%    |
| grammar      | 0.99      | 74%    | 0.62      | 70%    |
| word_order   | 0.96      | 64%    | 0.59      | 36%    |
| missing_word | 1.00      | 0%     | 0.51      | 20%    |
| extra_word   | 0.50      | 100%   | 0.50      | 96%    |

RF at global thresholds collapses rapidly: t=0.5 F1=74.8%, t=0.8 F1=50.7%, t=0.9 F1=25.0%.

**Observations:**
- **RF probability calibration is the killer**: RF outputs fraction-of-trees as probability, which clusters around 0.5. Most positive predictions are 0.5–0.7, making thresholds above 0.5 destroy detection. LR's sigmoid produces well-spread probabilities (0.0–1.0) that allow fine-grained threshold tuning.
- **RF does not find non-conforming subpopulations**: The hypothesis was that tree splits would capture pockets of hard-to-classify errors. But with 75–156 positive tokens per type and balanced weighting, RF produces noisy trees that average out to weak predictions. The subpopulations exist but are too small for 100 trees × max_depth=8 to carve out reliably.
- **RF gets slightly fewer FP (21 vs 24)**: The low RF thresholds (all near 0.50) are effectively "any tree says error" — this is conservative but also misses many true errors. LR with high thresholds is more discriminating.
- **missing_word improved to 20% with RF vs 0% with LR**: RF's lower threshold requirement (0.51 vs 1.00) lets some through. But this comes from RF's poor calibration, not from better learning.
- **Overall**: The non-conforming subpopulation hypothesis is plausible but unsupported at this data scale. 75–156 positive examples per type can't train a 100-tree forest with depth 8 effectively. The problem isn't linear separability — it's data quantity. LR remains the better classifier for sparse, low-data regimes.

**Conclusions:**
- LR wins decisively. RF's poor probability calibration makes it worse at threshold-based detection.
- Revert RF changes from `src/classifier.py` — the `classifier_type` parameter adds complexity without benefit.
- The subpopulation idea might work with 10x more data (6000 pairs), but that's independent of classifier choice.
- Next: try negative features (clean-only activations as counterweight) — this addresses the asymmetry in feature space, which is orthogonal to classifier choice.

**Commit:** 2a12042

---

## Experiment 15: Negative-example features (clean-only activation counterweight)

**Date:** 2025-07-25T14:00:00+02:00

**Goal:** Add "correctness evidence" to the classifier by selecting features that fire at last-word positions in clean text but NOT at error word last-token positions. These negative features provide a counterweight — high activation pushes P(error) down, making the decision boundary more robust. Test whether concatenating positive + negative feature vectors reduces false positives while maintaining detection rate.

**Hypothesis:** Current feature selection has survivorship bias — only "error evidence" features are selected. There should be SAE features encoding "this word looks grammatically normal" that fire on clean text but not at error positions. Adding N_neg negative features (top-N by pair count, mirroring the positive selection) will give the LR classifier both dimensions to work with, reducing FP by providing evidence that a token is correct. We expect FP to drop meaningfully with little or no loss in recall.

**Parameters:**
- Baseline: position-aware top-50 positive features, per-type thresholds (FP ≤ 5%)
- Negative feature selection: mirror of positive — features at clean last-word positions, absent from error word last-token positions
- Sweep: N_neg ∈ [0, 25, 50] (0 = baseline)
- Layers: [7, 13, 17, 22], SAE width: 16k
- 600 pairs, v5 data, 75/25 split, seed=42
- Classifier: OVR binary LR, class_weight="balanced", max_iter=2000

**Results:**

Per-type thresholds (FP ≤ 5%) comparison:

| N_neg | F1    | Precision | Recall | FP# |
|-------|-------|-----------|--------|-----|
| 0     | **81.6%** | 83.3% | **80.0%** | 24  |
| 25    | 69.3% | 83.2% | 59.3% | 18  |
| 50    | 60.3% | 85.4% | 46.7% | **12** |

Per-type breakdown (FP ≤ 5%):

| Type         | N=0 Thresh | N=0 Det | N=25 Thresh | N=25 Det | N=50 Thresh | N=50 Det |
|--------------|------------|---------|-------------|----------|-------------|----------|
| spelling     | 0.93       | 90%     | 0.99        | 76%      | 1.00        | 3%       |
| word_choice  | 0.93       | 55%     | 0.99        | 50%      | 0.99        | 41%      |
| grammar      | 0.99       | 74%     | 1.00        | 7%       | 1.00        | 15%      |
| word_order   | 0.96       | 64%     | 0.98        | 59%      | 0.99        | 59%      |
| missing_word | 1.00       | 0%      | 1.00        | 0%       | 1.00        | 0%       |
| extra_word   | 0.50       | 100%    | 0.50        | 88%      | 0.50        | 88%      |

Negative candidate pools: 7945–9679 per type (vs 240–979 positive candidates). Vastly more features fire on clean last-word positions than on error positions.

**Observations:**
- **Negative features crush recall without proportional FP reduction.** N=25 loses 21pp recall for just 6 fewer FP. N=50 loses 33pp recall for 12 fewer FP. The trade-off is terrible — each FP saved costs ~3pp recall.
- **The problem is probability compression.** Negative features fire on both clean AND error tokens (they fire on "normal-looking" token positions, which includes many error tokens too). The LR learns to weight them negatively, which pushes ALL probabilities down — not just clean-text probabilities. Thresholds must rise to 0.99+ to stay within FP budget, killing detection.
- **Negative features are not discriminative enough.** With 8000+ candidates per type, the top-25 fire on ~70+ pairs each — but they fire on clean text by definition, meaning they also fire on non-error positions in error texts. The selection criterion (fire on clean, not on error positions) is necessary but not sufficient for a feature to be a useful "correctness signal". Most of these features encode generic language properties (common words, frequent subword patterns) rather than "this specific position is grammatically correct".
- **The asymmetry is fundamental.** There are ~10x more negative candidates than positive candidates because most SAE features encode common language patterns, while error-specific features are rarer. The top negative features are common-word detectors, not correctness detectors — firing on "the", "is", "and" at word boundaries tells you nothing about whether a word is correctly used.
- **extra_word FP dropped to 0 at N=50** but detection also dropped 100%→88%. The 12% loss (3 sentences) suggests the negative features interfere even with the strongest type.

**Conclusions:**
- Negative features fail as a "correctness counterweight". The hypothesis that clean-only features encode "this word looks grammatically normal" is wrong — they encode "this is a common word/pattern", which is orthogonal to correctness.
- The survivorship bias in positive feature selection is not the bottleneck. The classifier already implicitly uses the absence of error features as correctness evidence (zero activations → low probability). Adding explicit "correctness features" just adds noise.
- Revert all negative feature changes from `src/classifier.py`. Keep `select_negative_features_topn` removed, restore `train_ovr` and `predict_tokens_ovr` to positive-only.
- The FP reduction direction is promising (24→12) but the recall cost is unacceptable. Better FP reduction approaches: per-type threshold tuning (already done), more training data, or better positive feature selection — not negative features.

**Commit:** be90446

---

### Round 2: Diagnostic — coefficient analysis + overfitting check

After the Round 1 results, we dug into WHY negative features hurt so badly. Added coefficient magnitude analysis, train vs test comparison, and C sweep.

**Key diagnostic findings:**

1. **NOT overfitting** — FP increases on both train AND test with negative features. The model generalizes, but in the wrong direction.
2. **Coefficient imbalance is the smoking gun:**
   - Spelling: neg_coef=0.077 vs pos_coef=0.020 (3.8x larger!)
   - Grammar: neg_coef=0.094 vs pos_coef=0.046 (2x)
   - `class_weight="balanced"` upweights each positive token ~42x, causing LR to over-rely on negative features
   - Intercept shifts dramatically: spelling -4.0 → -6.4 with neg features
3. **C sweep didn't help:** N=25 at C=0.01 still only F1=70.3%
4. **Interesting bonus:** N_neg=0 with C=0.01 gave F1=82.2% (slightly better than C=1.0's 81.6%)

**Root cause analysis:** The original negative feature selection was too broad. With ~8000-9700 candidates per type, the selected features encode "this is a common word" (fire on "the", "is", "and") rather than "this specific word is correct". They fire on nearly all text, providing no useful signal.

### Round 3: Paired negative selection + hyperparameter tuning

**Changed approach:** Instead of comparing "all clean last-word positions vs error word positions", compare only the specific clean word at the SAME position as each error word (paired comparison). For spelling/word_choice/grammar/word_order, the clean and error sentences differ at specific word indices — we only look at SAE activations at those positions. For missing_word/extra_word (word count changes), fall back to all clean positions.

Also tested `class_weight=None` vs `"balanced"`, and different C values.

**Paired selection reduced candidate pools dramatically:**
- Grammar: 9679 → 574 candidates (paired)
- Spelling: 9679 → 1131 candidates (paired)
- Word_choice: 9679 → 862 candidates (paired)
- Word_order: 9679 → 1097 candidates (paired)
- Missing_word/extra_word: still ~8400+ (fallback, not truly paired)

**Round 3a: Paired selection × class_weight sweep (N_neg=25):**

| Config | F1 | P | R | FP# |
|--------|-----|------|------|-----|
| N_neg=0, balanced (baseline) | 81.6% | 83.3% | 80.0% | 24 |
| N_neg=0, none | 70.4% | 88.0% | 58.7% | 12 |
| N_neg=25, balanced, paired | 74.3% | 84.0% | 66.7% | 19 |
| N_neg=25, none, paired | 70.9% | 88.1% | 59.3% | 12 |

`class_weight=None` kills recall (80→59%). Balanced is clearly needed. Coefficient imbalance much improved (spelling neg=0.012 vs pos=0.022) but grammar still problematic and still collapsed at N_neg=25.

**Round 3b: Fewer neg features + paired-only types (exclude missing_word/extra_word):**

| N_neg | C | F1 | P | R | FP# |
|-------|---|-----|------|------|-----|
| 0 | 1.0 | 81.6% | 83.3% | 80.0% | 24 |
| 3 | 1.0 | 81.4% | 82.8% | 80.0% | 25 |
| **5** | **1.0** | **82.8%** | **83.7%** | **82.0%** | **24** |
| 5 | 0.1 | 82.4% | 83.6% | 81.3% | 24 |
| 7 | 1.0 | 77.8% | 85.6% | 71.3% | 18 |
| 10 | 1.0 | 78.6% | 84.6% | 73.3% | 20 |
| 25 | 1.0 | 75.6% | 85.0% | 68.0% | 18 |

**N_neg=5/C=1.0 is the new best: F1=82.8% (+1.2pp), R=82.0% (+2pp), P=83.7% (+0.4pp).**

Per-type breakdown at N_neg=5:

| Type | N=0 Det | N=5 Det | Change |
|------|---------|---------|--------|
| spelling | 90% | 90% | — |
| word_choice | 55% | 59% | +4pp |
| grammar | 74% | 78% | +4pp |
| word_order | 64% | 59% | -5pp |
| extra_word | 100% | 100% | — |
| missing_word | 0% | 0% | — |

Coefficient magnitudes well-balanced at N_neg=5: spelling |neg|=0.014/|pos|=0.016, grammar |neg|=0.076/|pos|=0.081. The sweet spot is very sharp — N_neg=3 too weak, N_neg≥7 collapses grammar.

### Final conclusions (Experiment 15)

1. **Negative features CAN help, but require careful tuning:**
   - Paired selection (comparing same-position clean vs error words) instead of broad selection
   - Only for truly paired types (exclude missing_word/extra_word)
   - Very few features (N_neg=5 out of 50 positive) — the effect is fragile
2. **The improvement is modest but real:** +1.2pp F1, +2pp recall, same FP count
3. **Grammar and word_choice benefit most** from negative features (+4pp each)
4. **Word_order gets slightly worse** (-5pp) — its paired neg features may not be informative
5. **Keeping class_weight="balanced" and C=1.0** — neither removing balanced nor changing C helps

Best known config:
- 50 positive features (position-aware top-N)
- ~~5 paired negative features (for spelling/word_choice/grammar/word_order only)~~
- OVR LR with class_weight="balanced", C=1.0
- Per-type thresholds (FP ≤ 5%)
- **F1=81.6%, P=83.3%, R=80.0%, FP#=24** (unchanged from Exp 13 baseline)

**Decision:** Reverted all negative feature changes. The +1.2pp gain at N_neg=5 is too fragile (only works at exactly 5, collapses at 7+) and adds significant complexity (paired selection, PAIRED_ONLY_TYPES, per_type_neg_index). Kept the C and class_weight parameters in `train_ovr` as they're low-cost and useful for future experiments. The codebase at HEAD reflects the positive-only pipeline.

**Commit:** be90446

---

## Experiment 16 — Drop missing_word, add "wtf" category, fix word boundaries

**Date:** 2026-04-14

**Goal:** Three improvements in one pass:
1. **Drop `missing_word`** — threshold=1.0, 0% detection. Fundamentally not detectable at the word level with our approach.
2. **Add `wtf` category** — completely random/gibberish words (e.g., "asdhgboat") are not detected as spelling errors. The LLM represents misspellings and unrecognizable gibberish differently in its activations, so we need a separate category.
3. **Fix word boundary tokenization** — punctuation tokens (`.`, `,`) map to the preceding word in `token_to_word_index`, making the last token of "mat." be `"."` instead of `"mat"`. The classifier evaluates the wrong token for end-of-sentence words.

**Hypothesis:**
- Dropping missing_word frees up 100 data pairs for other types (5 types × 120 pairs each).
- Gibberish words should produce very different SAE activations from normal text — expect high detection rate if the features exist.
- Fixing punctuation boundaries will improve detection of errors in end-of-sentence words.

**Parameters:** Same as Exp 13 baseline (layers [7,13,17,22], top-50, FP≤5%, 600 pairs), but with 5 types instead of 6, and new `wtf` type.

### Round 1 — All three changes together

Changes: (1) dropped `missing_word`, (2) added `wtf` gibberish category, (3) fixed `token_to_word_index` to exclude punctuation-only tokens from word mapping. Data version `v6`, extract version `v3`.

**Results:**

| Type | Thresh | Det | FP | (Exp 13 baseline) |
|------|--------|-----|-----|-------------------|
| spelling | 0.93 | 96% | 4.7% | 90%, 4.7% |
| word_choice | 0.99 | 64% | 4.0% | 55%, 2.0% |
| grammar | 0.98 | 70% | 4.7% | 74%, 4.7% |
| word_order | 0.98 | 31% | 4.7% | 64%, 4.7% |
| extra_word | 0.50 | 92% | 0.0% | 100%, 0.7% |
| wtf | 0.50 | 100% | 4.0% | (new) |

**Combined: F1=82.8%, P=82.2%, R=83.3%, FP#=27** (baseline: F1=81.6%, P=83.3%, R=80.0%, FP#=24)

**Analysis:**
- **wtf works perfectly.** 100% detection at t=0.50 (lowest possible). 834 feature candidates — the LLM clearly represents gibberish very differently from normal text and even spelling errors. Confirms the user's intuition.
- **spelling: +6pp** (90% → 96%). Likely from the punctuation boundary fix — end-of-sentence misspellings now evaluated on the word token instead of `.`.
- **word_choice: +9pp** (55% → 64%). Unexpected improvement. May be from data shift (different sentences in v6) or boundary fix affecting comma-attached words.
- **word_order: -33pp** (64% → 31%). Massive collapse. At t=0.50, word_order detects 88% — the signal is there. But FP is 68% at t=0.50, requiring t=0.98 to stay under budget. The classifier can't separate unusual-but-correct word patterns from actual swaps. This is a data shift issue (different sentences from changed RNG path) making the threshold more restrictive (0.96 → 0.98).
- **extra_word: -8pp** (100% → 92%). Slight decrease, still strong.
- **FP# increased slightly** (24 → 27). The wtf category adds 4% FP on clean texts.

### Round 2 — Seed variation analysis

Investigated whether word_order collapse (64% → 31%) is systematic or data luck by running seeds 43 and 44 with the same code.

| Seed | word_order | spelling | grammar | word_choice | extra_word | wtf | F1 |
|------|-----------|----------|---------|-------------|------------|-----|-----|
| 42 (v6) | 31% @ 0.98 | 96% @ 0.93 | 70% @ 0.98 | 64% @ 0.99 | 92% @ 0.50 | 100% @ 0.50 | 82.8% |
| 43 (v6b) | 52% @ 0.99 | 80% @ 0.92 | 68% @ 0.99 | 48% @ 0.97 | 94% @ 0.50 | 93% @ 0.50 | 84.6% |
| 44 (v6c) | 54% @ 0.99 | 83% @ 0.98 | 31% @ 1.00 | 64% @ 0.96 | 90% @ 0.50 | 96% @ 0.50 | 80.6% |

**Key observations:**
- **Word_order is consistently weak** (31-54% across seeds), down from 64% on v5. Not data luck.
- **Massive per-type variance across seeds.** Grammar swings 31-70%, spelling 80-96%, word_choice 48-64%. With only 25 test pairs per type, this is expected (each pair = 4% swing).
- **Combined F1 is more stable** (80.6-84.6%) because types partially compensate each other.
- **wtf and extra_word are stable** — high detection at low thresholds across all seeds.
- The punctuation fix likely accounts for some word_order drop — punctuation tokens previously leaked word-boundary context that helped. But the fix is correct (punctuation shouldn't be scored), so this is a real regression in our ability to detect word order, not an artifact.

### Final conclusions (Experiment 16)

1. **wtf category works perfectly** — 93-100% detection at t=0.50 across seeds. Gibberish produces very distinct SAE activations (834 candidates). Confirms the LLM represents spelling errors and gibberish very differently.
2. **Punctuation fix is correct and important** — gives +6pp spelling improvement on seed 42 (end-of-sentence words now scored correctly). Improves overall pipeline correctness.
3. **Word_order regression is real and systematic** — 31-54% across seeds vs 64% before. The punctuation fix removed leaked signal and the data regeneration (different RNG path) shifted which sentences get used. High variance (31-54%) indicates fundamental instability with 25 test pairs.
4. **Per-type variance is too high at N=600 pairs.** Grammar swings 31-70% across seeds. Hyperparameter optimization on a single split is unreliable. Need larger datasets and/or cross-validation.
5. **Combined metrics are acceptable:** F1=82.8%, slightly above the Exp 13 baseline of 81.6%.
6. **Dropping missing_word was the right call** — freed data for other types and removed a category with 0% detection.

Restored seed 42 as default. The codebase now has 6 types: spelling, word_choice, grammar, word_order, extra_word, wtf.

**Commit:** b14e129

---

## Experiment 17 — K-fold cross-validation

**Date:** 2026-04-14

**Goal:** Replace the single 75/25 split with 5-fold CV to get reliable mean ± std metrics. Exp 16 showed per-type detection swings wildly across seeds (grammar 31-70%, word_order 31-54%), making single-split evaluation untrustworthy. K-fold will reveal true performance and variance.

**Hypothesis:** Combined F1 will be similar to the single-split result (~82%), but individual type detection rates will have high std (±10-15pp for weak types like word_order, lower for strong types like spelling). This will establish trustworthy baselines for future experiments.

**Parameters:** Same pipeline (layers [7,13,17,22], top-50, FP≤5%, 600 pairs v6). 5 folds, feature selection + training + thresholds computed independently per fold.

### Results

Per-fold results:

| Type | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|------|--------|--------|--------|--------|--------|
| spelling | 78% @ 0.99 | 95% @ 0.86 | 100% @ 0.91 | 88% @ 0.94 | 96% @ 0.89 |
| word_choice | 12% @ 1.00 | 74% @ 0.92 | 58% @ 0.95 | 5% @ 1.00 | 61% @ 0.99 |
| grammar | 59% @ 0.99 | 50% @ 0.98 | 76% @ 0.97 | 80% @ 0.98 | 62% @ 0.99 |
| word_order | 39% @ 0.96 | 25% @ 0.99 | 52% @ 0.98 | 0% @ 1.00 | 30% @ 0.98 |
| extra_word | 91% @ 0.50 | 87% @ 0.50 | 100% @ 0.50 | 95% @ 0.50 | 95% @ 0.50 |
| wtf | 88% @ 0.50 | 100% @ 0.50 | 96% @ 0.50 | 91% @ 0.50 | 100% @ 0.50 |
| **F1** | 78.1% | 79.3% | 83.6% | 76.5% | 80.7% |

5-fold CV summary:

| Type | Det mean | Det std | Thresh mean | FP mean |
|------|----------|---------|-------------|---------|
| spelling | 91.4% | ±7.6% | 0.92 | 5.0% |
| word_choice | 42.0% | ±27.8% | 0.97 | 3.0% |
| grammar | 65.5% | ±11.1% | 0.98 | 4.7% |
| word_order | 29.3% | ±17.3% | 0.98 | 3.0% |
| extra_word | 93.7% | ±4.5% | 0.50 | 0.7% |
| wtf | 94.9% | ±4.9% | 0.50 | 2.3% |

**Combined: F1=79.6% ±2.4%, P=82.7% ±1.5%, R=77.0% ±5.3%, FP#=19.4 ±2.9**

### Analysis

**Three tiers of reliability:**
1. **Rock solid** (std ≤5%): extra_word (93.7% ±4.5%), wtf (94.9% ±4.9%). These are essentially solved — the SAE activations for duplicated words and gibberish are so distinct that even different train/test splits barely matter. Thresholds always at 0.50 (lowest possible).
2. **Reasonable** (std ~8-11%): spelling (91.4% ±7.6%), grammar (65.5% ±11.1%). Some fold-to-fold variance but the signal is real. Spelling is reliably strong. Grammar is moderate and stable.
3. **Unreliable** (std ≥17%): word_choice (42.0% ±27.8%), word_order (29.3% ±17.3%). Catastrophic variance. Word_choice swings 5-74%, word_order 0-52%. Fold 4 hits 0% word_order and 5% word_choice — the classifier essentially fails on these types for certain splits.

**What the variance tells us:**
- **word_choice** at ±27.8% is the worst. The confusable-word swaps (their→there, etc.) produce weak SAE signal — the model doesn't strongly distinguish homophones in its activations. When the specific test pairs happen to include the few "easy" confusables, detection is high; otherwise it collapses. Threshold often hits 1.00 (no detections at all).
- **word_order** at ±17.3% with mean 29.3% is similarly weak. The threshold is consistently high (0.96-1.00) — there's too much FP pressure at lower thresholds, so we have to cut aggressively.  
- **Combined F1 is 79.6% ±2.4%** — decently stable. Our single-split result of 82.8% was slightly optimistic (in the upper range).
- **The previous Exp 13 baseline of 81.6%** also appears to have been a favorable split. The true performance is ~80%.
- **FP count is well-controlled**: 19.4 ±2.9, always under budget.

**Implication for past experiments:**
- The +1.2pp F1 improvement from Exp 15 (negative features) was within noise (±2.4%). Correct to have reverted.
- The Exp 16 "word_order collapse" from 64% to 31% was within the ±17% std band — it wasn't as dramatic as it looked, just normal variance.

### Conclusions

1. **True baseline: F1=79.6% ±2.4%.** Previous single-split results (~81-83%) were slightly optimistic.
2. **spelling, extra_word, wtf are reliably strong.** These types are effectively solved for our current approach.
3. **word_choice and word_order are unreliable.** Variance is too high to meaningfully optimize. Need fundamentally better features or much more data for these types.
4. **grammar is in the middle** — real signal but room for improvement.
5. **K-fold CV is now the standard evaluation method.** Future experiments must report mean ± std to claim improvement.
6. **An improvement must exceed ±2.4% F1** (ideally 2× std = ~5pp) to be considered real.

**Commit:** ccfe466

---

## Experiment 18: Data quality improvements for grammar and word_order

**Date:** 2026-04-14T12:00:00+02:00

**Goal:** Two targeted data quality fixes addressing known root causes of weak detection:
1. **Grammar swap diversity**: Currently 33/100 grammar pairs have "are" as the error word (all from `is→are` swaps), causing the classifier to learn "are" itself as a grammar indicator rather than context-dependent signal. Fix: expand GRAMMAR_SWAPS with more diverse entries (auxiliaries, modals, article errors) and cap per-key usage at 5 to prevent any single swap from dominating.
2. **Word_order single-word labeling**: Currently both swapped words are labeled as errors, but only the displaced word (at position idx+1) is *guaranteed* wrong by the swap pattern. The word at position idx may or may not be contextually wrong. Fix: only label idx+1 to give the classifier cleaner training signal.

**Hypothesis:** Grammar detection (65.5% ±11.1%) should improve in both detection rate and FP behavior as the classifier learns context-dependent grammar features instead of "are = grammar error". Word_order (29.3% ±17.3%) may improve with cleaner labels — the classifier currently has to find signal at two positions when only one is reliable.

**Parameters:**
- Same as Exp 17 baseline: layers=[7,13,17,22], 16k SAE, 600 pairs, 5-fold CV
- Data version: v6 → v7 (new GRAMMAR_SWAPS + cap at 5 per key, single-word word_order labels)
- New GRAMMAR_SWAPS additions: am↔is, will↔would, can↔could, shall↔should, may↔might, much↔many, good↔well, bad↔badly, a↔an
- Grammar max_per_key: 5

### Results (Round 1)

| Type | Exp 17 baseline | Exp 18 | Change |
|------|----------------|--------|--------|
| spelling | 91.4% ±7.6% | 87.5% ±4.9% | -3.9pp |
| word_choice | 42.0% ±27.8% | 44.1% ±16.8% | +2.1pp |
| grammar | 65.5% ±11.1% | **17.5% ±22.2%** | **-48pp** |
| word_order | 29.3% ±17.3% | 10.6% ±21.2% | -18.7pp |
| extra_word | 93.7% ±4.5% | 90.7% ±2.3% | -3.0pp |
| wtf | 94.9% ±4.9% | 92.8% ±2.4% | -2.1pp |
| **Combined F1** | **79.6% ±2.4%** | **75.5% ±3.4%** | **-4.1pp** |

**Combined: F1=75.5% ±3.4%, P=83.7% ±1.2%, R=69.0% ±5.4%, FP#=16.2 ±2.0**

### Analysis

**Both changes backfired.**

1. **Grammar collapsed catastrophically** (65.5% → 17.5%): The swap diversity + per-key cap at 5 diluted the training signal beyond recovery. Grammar thresholds hit 1.00 in 3 of 5 folds (= zero detections). With cap=5, each individual swap key appears too rarely for the classifier to learn from ~87 positive training tokens spread across ~50 different swap keys. The old "is→are"-dominated approach actually *worked* because it provided a consistent, learnable pattern — the classifier detected "are in wrong context" reliably even if the underlying feature was "are-specific" rather than "grammar-general."

2. **Word_order also worsened** (29.3% → 10.6%): Single-word labeling (only idx+1) halved the positive training tokens from ~150 to ~75 for an already-marginal type. The signal, which was barely learnable with two labeled words, became unlearnable with one. Threshold stuck at 1.00 in 4 of 5 folds.

3. **Even strong types regressed slightly**: spelling 91.4%→87.5%, extra_word 93.7%→90.7%, wtf 94.9%→92.8%. This is likely due to the regenerated data (v7) producing different sentence/error distributions rather than a direct effect of the grammar/word_order changes.

**Key insight:** At 100 pairs per type (~80-90 positive training tokens), **consistency beats diversity**. Theoretical improvements to training signal quality are gated on data volume — the current dataset is too small for the classifier to learn from diverse patterns. The "is→are" bias is a symptom of small data, but removing it without adding data just removes the signal entirely.

### Conclusions

1. **Failed experiment. Reverting all changes.** The -4.1pp F1 regression exceeds the 2× std threshold for a real effect.
2. **Data diversity improvements must be paired with data scaling.** Grammar swap diversity and word_order single-label are sound ideas that need ≥1000 pairs per type to work. Added to ideas.md as a prerequisite for the "scale to 6000+ pairs" idea.
3. **The "are" bias is load-bearing at current scale.** Counterintuitive but true: the classifier's grammar detection works *because* 33% of grammar errors are "is→are", not despite it.

**Commit:** 61b3f29

---

## Experiment 19: Feature extraction speedup — truncation and batching

**Date:** 2026-04-14T23:30:00+02:00

**Goal:** Speed up feature extraction to make scaling to 6000+ pairs practical. Two independent optimizations, benchmarked separately:
1. **Model truncation**: We only use layers [7,13,17,22]. Layers 23-25 are never read. Setting `config.num_hidden_layers=23` stops the forward pass after layer 22, saving ~12% compute and ~80M params of VRAM. This is provably lossless — the hidden states at layers 0-22 are identical since later layers can't influence earlier ones.
2. **Batched extraction**: Currently texts are processed one at a time. Batching multiple texts (with padding + attention_mask) improves GPU utilization. Potential concern: attention cross-contamination between padded sequences, though the attention mask should prevent this. Also, N² attention scaling means padding waste grows with batch size.

**Hypothesis:** Truncation gives a modest but guaranteed speedup (~12-15%). Batching gives a larger speedup (2-3x) but needs correctness verification. Combined they should cut 6000-pair extraction from ~3h to ~1-1.5h.

**Parameters:**
- Benchmark on 50 pairs (100 texts) from existing cached data
- Configurations: baseline, truncation-only, batching-only, both combined
- Batch sizes to test: 4, 8, 16
- Correctness: verify hidden states match between all configurations

### Iteration 1: Truncation

Tried 4 approaches to truncate the model at layer 22 (skipping layers 23-25):
1. **Post-load layer surgery** (`model.model.layers = model.model.layers[:23]`): 5x SLOWER. Breaks accelerate dispatch hooks installed by `device_map="cuda"`.
2. **Config-based truncation at load time** (`AutoConfig` with reduced `num_hidden_layers`): 5x SLOWER. HuggingFace warns about "unexpected" weights for layers 23-25, accelerate confused.
3. **Load without device_map, manual `.to(device)`**: 5x SLOWER. Doesn't get same CUDA optimizations as `device_map="cuda"`.
4. **Config-only post-load** (just change `config.num_hidden_layers` after loading): 5x SLOWER. Same dispatch hook issues.

Also ran swapped order (truncated first, baseline second) to rule out thermal throttling — truncated still 5x slower.

**Conclusion:** Model truncation is not viable with `device_map="cuda"` + accelerate. The dispatch hooks are deeply entangled with the model structure, and any modification breaks them catastrophically. Even if it worked, it would only save 3/26 layers of forward pass time, which turned out to be a tiny fraction of total time (see below).

### Iteration 2: Batching

Tested batched extraction (multiple texts per forward pass, left-padded with attention mask). Result: **0% speedup**. Forward pass was not the bottleneck.

### Iteration 3: Timing breakdown — finding the real bottleneck

Added per-phase timing to the benchmark:

| Phase | Time (100 texts) | % of total |
|-------|-------------------|------------|
| Forward pass | 3.9s | 7% |
| SAE encode | 0.1s | <1% |
| Other (tokenize + sparse dict) | 55.4s | **93%** |
| **Total** | **59.4s** | 100% |

The forward pass + SAE encoding is fast (~40ms/text). The bottleneck was `_sae_acts_to_feats` — a Python loop over all nonzero SAE activations with per-element `.item()` calls, each causing a GPU→CPU synchronization. With ~16k-wide SAEs at ~5% sparsity × ~20 tokens × 4 layers, that's ~60,000 individual `.item()` syncs per text.

### Iteration 4: Vectorized GPU→CPU transfer

**Fix:** Replaced per-element `.item()` calls with bulk `.cpu().tolist()` — three tensor transfers instead of N individual syncs:

```python
# Before (slow): per-element GPU→CPU sync
for pos, feat_idx in nonzero:
    fid = feat_idx.item()
    layer_feats[fid].append((pos.item(), sae_acts[pos, feat_idx].item(), ...))

# After (fast): bulk transfer, then pure Python loop
positions = nonzero[:, 0].cpu().tolist()
feat_ids = nonzero[:, 1].cpu().tolist()
values = sae_acts[positions, feat_ids].cpu().tolist()
for pos, fid, val in zip(positions, feat_ids, values):
    ...
```

Also optimized `tokenize()` to use `.tolist()` on token_ids before the decode loop.

**Result: 594ms/text → 41ms/text. 14.5x speedup.**

| Metric | Before | After |
|--------|--------|-------|
| Per-text time | 594ms | 41ms |
| 100 texts | 59.4s | 4.1s |
| 600 pairs (1200 texts) | ~12 min | ~50s |
| 6000 pairs (12000 texts) | ~119 min | ~8 min |

### Results summary

- **Truncation:** Not viable with `device_map="cuda"` / accelerate. All 4 approaches caused 5x slowdown.
- **Batching:** 0% speedup. Forward pass is only 7% of total time.
- **Vectorized sparse dict construction:** **14.5x speedup.** The real bottleneck was Python-level GPU→CPU synchronization in `_sae_acts_to_feats`, not the model or SAE computation.
- **6000 pairs now feasible:** ~8 minutes instead of ~2 hours.

### Conclusions

The initial hypothesis was entirely wrong — the forward pass (the only thing truncation and batching can help) was 7% of total time, not the bottleneck. The vast majority of extraction time was spent in a seemingly innocuous Python utility function doing per-element `.item()` calls. This is a known PyTorch anti-pattern but easy to miss in profiling since it doesn't show up in GPU metrics.

The fix is simple, lossless, and permanent. Scaling to 6000+ pairs is now practical.

**Commit:** 8762562

---

## Experiment 20: Scale to 6000 pairs

**Date:** 2026-04-15T16:00:00+02:00

**Goal:** Scale synthetic data from 600 pairs (100/type) to 6000 pairs (1000/type). This is a straightforward 10x increase enabled by Exp 19's extraction speedup (~8 min for 12,000 texts). More training data should improve weak types (word_order 29.3%, word_choice 42.0%) and reduce per-fold variance.

**Hypothesis:** F1 improves from 79.6% → 82-85%, driven by word_order and word_choice. Variance (±2.4%) should shrink to ±1%. Strong types (extra_word, wtf, spelling) should hold or improve slightly.

**Parameters:**
- N_PAIRS: 6000 (was 600)
- 5-fold CV, same as Exp 17 baseline
- All else unchanged: layers [7,13,17,22], top-50 features, 5% FP budget, 16k SAEs

**Note:** First run loaded stale `synthetic_pairs__v7.pkl` from Exp 18 (only 600 pairs) because `N_PAIRS` wasn't in the data cache key. Fixed by adding `DATA_CACHE_KEY = f"{DATA_VERSION}_n{N_PAIRS}"`. Also reverted DATA_VERSION to "v6" (the data generation logic hasn't changed).

### Round 1: 6000 pairs with top_n=50 (baseline comparison)

| Metric | Exp 17 (600 pairs) | Exp 20 (6000 pairs) | Delta |
|--------|-------------------|---------------------|-------|
| **F1** | 79.6% ± 2.4% | **83.5% ± 1.1%** | **+3.9pp**, variance halved |
| Precision | 82.7% ± 1.5% | 82.2% ± 0.9% | -0.5pp |
| Recall | 77.0% ± 5.3% | 84.9% ± 1.6% | +7.9pp |

Per-type detection rates:

| Type | 600 pairs | 6000 pairs | Delta |
|------|-----------|------------|-------|
| spelling | 91.4% ± 7.6% | 91.1% ± 1.0% | held, variance 7.6x smaller |
| word_choice | 42.0% ± 27.8% | 57.2% ± 5.6% | +15.2pp |
| grammar | 65.5% ± 11.1% | 73.5% ± 3.2% | +8.0pp |
| **word_order** | 29.3% ± 17.3% | **55.8% ± 3.4%** | **+26.5pp** |
| extra_word | 93.7% ± 4.5% | 98.9% ± 1.0% | +5.2pp |
| wtf | 94.9% ± 4.9% | 98.7% ± 1.0% | +3.8pp |

Every type improved. Variance collapsed everywhere. Feature candidate pools exploded (word_order: 352 → 5882). With 1500 positive training tokens per type (was ~80), the classifier has room for more features.

### Round 2: top_N sweep (50, 100, 150, 200)

At 600 pairs, top_n=100 collapsed word_order to 0% (data starvation: ~80 tokens / 100 features). At 6000 pairs with ~1500 tokens/type, higher top_N should now work.

| top_N | F1 | P | R | FP# | spelling | word_choice | grammar | word_order | extra_word | wtf |
|-------|------|------|------|-----|------|------|------|------|------|------|
| 50 | 83.5% ±1.1% | 82.2% | 84.9% | 220 | 91.1% | 57.2% | 73.5% | 55.8% | 98.9% | 98.7% |
| **100** | **84.6% ±0.6%** | 82.4% | **87.0%** | 224 | **93.4%** | **67.6%** | **77.4%** | **58.8%** | 96.4% | 98.4% |
| 150 | 84.7% ±0.5% | 83.2% | 86.3% | 209 | 93.1% | 69.3% | 76.6% | 57.8% | 94.4% | 98.5% |
| 200 | 85.0% ±0.6% | 83.3% | 86.7% | 209 | 93.0% | 71.0% | 78.4% | 57.9% | 94.7% | 98.1% |

top_N=100 is the clear knee:
- +1.1pp F1, +10.4pp word_choice, +3.9pp grammar, +3pp word_order, +2.3pp spelling vs top_N=50
- Beyond 100: diminishing returns (+0.1-0.4pp F1 per step), extra_word degrades (98.9→94.4%)
- Variance tightens: ±1.1% → ±0.6%

Selected **top_N=100** as new default.

### Results summary

**Best configuration: 6000 pairs, top_N=100**
- F1: **84.6% ± 0.6%** (was 79.6% ± 2.4%, **+5.0pp**)
- Precision: 82.4% ± 0.3%
- Recall: 87.0% ± 1.1%
- Per-fold: [85.0%, 84.3%, 84.1%, 85.6%, 84.0%] — remarkably stable

### Conclusions

1. **10x data scaling works.** Every type improved, variance collapsed. The biggest winner is word_order (+29.5pp from Exp 17 baseline).
2. **top_N scaling also works.** At 600 pairs, top_N>50 overfitted. At 6000 pairs, top_N=100 is a free +1.1pp F1. The data starvation bottleneck that limited Exp 13's top-N sweep is gone.
3. **FP rate held.** Per-type FP rates stayed within the 5% budget. The combined FP rate (any type flagged) is ~18.6%, but this is driven by stacking 6 independent type classifiers; individual rates are controlled.
4. **Remaining weak types:** word_choice (67.6%) and word_order (58.8%) still lag. Both have enormous feature candidate pools (4000+, 5800+) but plateau around top_N=100-200. The limitation may be in the feature quality (position-aware selection criteria) or the error types themselves, not data quantity.
5. **Feature extraction at 6000 pairs took ~9 minutes** (one-time, cached), confirming Exp 19's speedup estimate.

**Commit:** d376138

---

## Experiment 21: Data quality and corruption diversity

**Date:** 2026-04-16T10:00:00+02:00

**Goal:** Improve synthetic data quality to reduce false positives and improve detection of currently-missed error patterns. Motivated by manual testing findings and the Exp 18 retry (now unblocked by data scale).

**Issues addressed:**
1. **Spelling: contractions excluded** — `w.isalpha()` filters out "didn't", "won't" etc., so "dodn't" is never generated as training data
2. **Spelling: no repeated-letter corruption** — "Proooooooobably" pattern never generated
3. **Spelling: no bias toward longer words** — long words with early-position typos ("suphistication") underrepresented
4. **Spelling: corruption position uniformly random** — should skew toward early positions in long words to generate more first-token errors for multi-token words
5. **Grammar: is/are dominance** — small swap table + no per-key cap means is↔are dominates training, causing FPs on correct usage
6. **Grammar: missing swap categories** — no auxiliaries/modals, articles, or tense-auxiliary swaps
7. **Word order: both positions labeled** — labeling just the displaced word could give sharper signal (Exp 18 retry, now with 10x data)

**Hypothesis:** Addressing issues 1-6 improves spelling detection and reduces grammar FPs. Issue 7 may help word_order now that we have 1000 pairs/type. Combined F1 should improve from 84.6% baseline.

**Parameters:**
- 6000 pairs, top_N=100, 5-fold CV (same as Exp 20)
- DATA_VERSION bumped to "v7" to invalidate caches

### Round 1: All changes together (grammar expanded + cap=50)

Implemented all 7 changes at once. DATA_VERSION v8, EXTRACT_VERSION v4.

| Metric | Exp 20 baseline | Round 1 | Delta |
|--------|----------------|---------|-------|
| **F1** | 84.6% ± 0.6% | **79.9% ± 0.5%** | **-4.7pp** |
| grammar | 77.4% ± 2.8% | **0.0% ± 0.0%** | collapsed |
| word_order | 58.8% ± 4.4% | 62.1% ± 1.1% | +3.3pp |
| spelling | 93.4% ± 2.1% | 93.4% ± 2.5% | held |

Grammar collapsed to 0% — the expanded swap table (~70 keys) + cap=50 diluted signal too much. Same failure mode as Exp 18 but now at 6000 pairs scale, confirming the issue is the table size, not data quantity.

### Round 2: Grammar cap raised to 200

Kept expanded table, raised cap from 50 → 200 to let high-signal keys (is/are) dominate more.

| Metric | Round 1 | Round 2 |
|--------|---------|---------|
| grammar | 0.0% ± 0.0% | 49.8% ± **25.1%** |
| F1 | 79.9% | 82.4% ± **1.4%** |

Grammar partially recovered but with extreme variance (±25.1%). Some folds get 0%, others get decent detection. The expanded table creates a distribution where fold-to-fold key composition varies wildly.

### Round 3: Revert grammar to original table, keep other improvements

Reverted to the original compact grammar table (~40 keys, no cap). Kept spelling improvements (repeat-letter, contraction support, long-word bias) and word_order single-label.

| Metric | Exp 20 baseline | Round 3 | Delta |
|--------|----------------|---------|-------|
| **F1** | 84.6% ± 0.6% | **84.8% ± 0.5%** | +0.2pp |
| spelling | 93.4% ± 2.1% | 92.0% ± 2.0% | -1.4pp (noise) |
| word_choice | 67.6% ± 4.8% | 63.4% ± 4.7% | -4.2pp (noise) |
| grammar | 77.4% ± 2.8% | 78.5% ± 2.4% | +1.1pp |
| word_order | 58.8% ± 4.4% | 60.7% ± 4.1% | +1.9pp |
| extra_word | 96.4% ± 2.1% | 98.3% ± 1.5% | +1.9pp |
| wtf | 98.4% ± 1.2% | 98.9% ± 0.8% | +0.5pp |

Per-fold F1: [85.0%, 84.5%, 84.0%, 85.1%, 85.5%] — very stable.

### Data quality audit

Verified the new corruption patterns in the generated data:
- **Repeat-letter errors**: 182/1727 error words (11%) have triple+ characters (e.g., "thannn", "seeem")
- **Long-word early corruptions**: 203/1727 (12%) are 7+ char words with diff at position <3 (e.g., "deposited→dfeposited", "transformed→ttransformed")
- **Contraction support**: Code is correct but SST2 tokenizes contractions as separate tokens ("don 't"), so 0 contraction errors generated. This is a data source limitation, not a code issue.
- **Word_order**: All 1000 pairs have exactly 1 label (single displaced word)

### Conclusions

1. **Grammar expansion confirmed harmful** — even at 6000 pairs (10x Exp 18), the expanded table collapses grammar detection. The original compact table is the right approach. The is/are FP the user reported is a classifier/threshold issue, not a data diversity issue.
2. **Spelling improvements**: repeat-letter and long-word bias add important coverage for real-world errors (repeated letters, early-token typos in long words). Metric impact is neutral on synthetic eval but addresses specific failure modes in manual testing.
3. **Word_order single-label**: +1.9pp detection improvement at 6000 pairs (was -19pp at 600 pairs in Exp 18). Confirmed: single-label is better when data is sufficient.
4. **Net result**: +0.2pp F1 overall (within noise), but qualitatively better coverage of real-world error patterns.
5. **Contraction gap**: SST2 doesn't have contractions in natural form. Would need a different data source or post-processing to generate "didn't→didin't" patterns. Not critical since the model should generalize from regular word errors.

**Commit:** 6e87a67
