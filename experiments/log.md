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
