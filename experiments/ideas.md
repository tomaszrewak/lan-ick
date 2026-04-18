# Experiment Ideas

Future experiment proposals, ordered roughly by priority/importance. Update this file as new ideas emerge or get completed.

---

### ~~Reduce combined FP rate (sentence-level calibration)~~ ✓ Done (Experiment 23)
Completed: Greedy F0.5 coordinate descent over per-type thresholds on a held-out calibration split beats the per-type 5% budget baseline. F0.5 82.4%→85.9%, combined precision 82.4%→88.2%, combined FPs 222→127 (−43%), at the cost of recall 86.5%→77.9%. Greedy F0.5 is now the default calibration in `src/pipeline.py`. `global_max` (single threshold on max-over-types) was nearly identical (85.7%) and simpler; kept greedy because it exposes per-type structure. `agreement2` crushed recall to 32% — types are not redundant.

### ~~Conditional (per-token contrastive) feature selection~~ ✗ Failed (Experiment 25)
Swept ratio threshold R ∈ {2, 3, 5, 10}. None beat baseline F0.5 85.9%. Root cause: reject-and-backfill from an equally-bad pool — grammar has ~2400 candidates, rejecting ~115 (R=3) still fills 100 features with nearly-as-token-keyed replacements. Worse: the filter destabilized other types (word_choice FP 75→192 at R=2). The entire grammar candidate pool is contaminated because the corruption IS a token substitution. Feature-level filtering is a dead end; the fix must come from the data side.

### ~~Per-type token-keyed feature audit (diagnostic, all 6 types)~~ ✓ Done (Experiment 25)
Completed: grammar 7/10 TK, word_order 6/10, word_choice 4/10, spelling 3/10, extra_word 0/10, wtf 0/10. Token-keyed features are pervasive in vocabulary-restricted substitution types. Novel-surface-form types (extra_word, wtf) are clean. Confirms the fix must target grammar/word_order/word_choice corruption diversity, not feature selection.

### ~~Diversify grammar corruption (v2 — structural errors, not token swaps)~~ ✗ Failed (Experiment 26)
Tested three structural strategies: verb agreement (-s add/strip + irregular), wrong preposition, tense change (-ed add/strip + irregular). Dropped demonstrative/pronoun swaps. **Results:** Round 1 (all 3 strategies): preposition swaps dominate (~80%) and are undetectable by SAE → grammar fully disabled (threshold=1.00), F0.5 84.4% (−1.5pp). Round 2 (agreement + tense only, no prepositions): F0.5 85.1% (−0.8pp), grammar threshold back to 0.99 but still below baseline. **Root cause:** the SAE detects surface-level token anomalies, not contextual grammar errors. Data diversification cannot fix this — the old token-swap table is actually the best match for the SAE's capabilities. Grammar corruption is now fully exhausted as an improvement avenue (Exp 18, 21, 25, 26 all failed).

### ~~Reduce per-type feature count for contaminated types~~ ✗ No improvement (Experiment 28)
Swept contaminated top_N ∈ {1, 2, 5, 10, 20, 100} with clean types fixed at 100. Combined F0.5 degrades monotonically as N decreases (87.1% → 83.4%). Precision stays flat at ~88.5% (calibrator compensates), but recall drops. Interesting per-type finding: grammar peaks at N=10 (71% det) then drops to 46% at N=100 — more features actively hurt it. word_choice is the opposite (27% → 68%). But combined metric is best at 100, so no change adopted. Also found and fixed a duplicate-loop bug in `select_features_position_aware_topn`. Layer 7 dominates (4/6 types' top feature).

### ~~Second-stage sentence-level meta-classifier~~ — Skipped
Rejected: a meta-classifier over per-type scores would obscure which error type fired, and we want to show the category to the user. The per-type threshold structure is part of the product, not just an implementation detail.

### ~~Diagnose grammar FPs (feature-level inspection)~~ ✓ Done (Experiment 24)
Completed: Inspected grammar's top 10 LR features on held-out test data. **Hypothesis confirmed strongly** — 6/10 features are token-keyed detectors for `those/these/are/Are/were` and related grammatical-category tokens. Top error tokens match top clean tokens per feature (e.g., feature 3 fires 82% on `those` in clean, 81% on `those` at grammar error positions — no conditional discrimination). Grammar FPs are dominated by exactly these tokens (`Are`×9, `is`×8, `these`×7, `those`×3, `were`×2). Root cause: corruption is a token substitution over a small vocabulary, so position-aware selection with pair-wise same-pair-clean filter passes any feature whose firing token matches the target vocabulary. Follow-ups: conditional per-token feature filter (high importance), all-types diagnostic, diversify grammar corruption.

### ~~Full layer sweep across all 26 layers~~ ✓ Done (Experiment 27)
Completed: Tested 6+4 layer combos across two rounds. Round 1: [3,7,13,17,25] beat baseline [7,13,17,22] by +0.9pp F0.5 (86.8% vs 85.9%, 5-fold validated). Round 2: [3,7,13,25] selected as new default — same 4-layer count as baseline, +1.1pp F0.5 in screening, highest precision (88.6%), and faster. Early layers carry more signal than expected. Per-layer caches exist for all 26 layers.

### Expanded layer combo search (6-7 layers with early emphasis)
**Goal:** Exp 27 showed early layers are surprisingly powerful. Try 5-6 layer combos building on [3,7,13,25]: e.g., [3,5,7,13,25], [3,7,8,13,25], [3,5,7,13,17,25]. Per-layer caches already extracted.
**Importance:** Medium-low — diminishing returns likely since [3,7,13,25] already beats the 5-layer combo on precision.

### ~~Scale up synthetic data (100s–1000s of pairs)~~ ✓ Done (Experiment 1)
Completed: 300 pairs, F1=70.1%. Features dropped from 204→41 (more selective). Bottleneck is now threshold/classifier.

### ~~Tune threshold + use activation magnitudes~~ ✓ Done (Experiment 2)
Completed: Token-level LR with activation magnitudes + threshold sweep. F1=92.0% (up from 70.1%). Perfect recall at all thresholds.

### ~~Explore higher thresholds and FP analysis~~ ✓ Done (Experiment 3)
Completed: Pushed to 0.95 — F1=94.2%, P=91.2%, R=97.3%. Dominant FP source: `n't` contractions (4/7 FPs). Secondary: rare proper nouns. Contraction handling would eliminate most FPs.

### ~~Handle contractions and tokenizer artifacts~~ — Largely resolved by Exp 11
The `n't` split FP was the dominant issue in Exp 3. Last-token-only scoring (Exp 11) skips the intermediate `' n'` fragment, leaving only `'t'` (which is a valid word-end). Spelling FP dropped from 8%→2.7% overall. May still be minor residual FP from contractions but no longer a priority.

### ~~Word order swap and similar-word replacement errors~~ ✓ Done (Experiment 4)
Completed as part of full 6-type expansion. Word order detected 82%, word choice 82%, but type classification accuracy lower (44%, 67%). Grammar detected 90% but misclassified 78% of the time.

### ~~Smarter synthetic data generation~~ ✓ Done (Experiment 12)
Completed: POS-based word order swaps (det→55%), function-word-only deletion for missing_word (22%→28%), expanded CONFUSABLES (~70 entries), expanded GRAMMAR_SWAPS (verb tense + pronoun case), added vowel substitution and double-letter drop for spelling. Overall F1 78%→81.5%, FP rate 29.3%→24.0%. Clear win.

### ~~Scale up multi-class training data~~ ✓ Done (Experiment 12)
Completed: Scaled from 300 to 600 pairs (100 per type). Training tokens roughly doubled. Combined with smarter corruption for a +3.5% F1 improvement.
**Importance:** Very high — data starvation is the likely cause of Exp 4's poor type classification.

### ~~One-vs-rest binary classifiers per error type~~ ✓ Done (Experiment 5)
Completed: 6 binary LRs trained. Spelling (92% det, 8% FP) and extra_word (91% det, 1.3% FP) work well. Grammar/word_order/missing_word have high FP — not separable with current features. Combined F1=72% (no improvement over Exp 4). Per-type insight is the main value.

### ~~Per-type feature sets for OVR classifiers~~ ✓ Done (Experiment 6)
Completed: Each type gets its own features. extra_word hit 0% FP (perfect), spelling 100% type accuracy. grammar and missing_word have ZERO features at 16k — not separable. word_choice and word_order each have 1 feature. FP down 61%. Combined F1=68.6% (lower recall since 2 types dropped entirely).

### ~~Feature selection method comparison~~ ✓ Done (Experiment 7)
Completed: Compared 8 methods (baseline, relaxed, paired_token_diff, magnitude_diff, ttest, top_k_error). **relaxed_30** (min_pair_ratio=0.3) wins with F1=78.0%, unlocking grammar detection (18 features, 75% det, 9% FP). Token-comparison methods (paired_diff, ttest) find grammar but with unacceptable FP. Fixed-K methods (magnitude_diff, top_k_error) are fundamentally unsuitable — destroy spelling detection. The simple binary presence approach with a lower threshold is the best paradigm.

### ~~FP-constrained threshold selection / per-type thresholds~~ ✓ Done (Experiment 13)
Completed: Per-type threshold sweep with 5% FP budget. Clear win over global threshold — N=50 per-type: F1=81.6%, P=83.3%, FP#=24 vs global t=0.95: F1=79.6%, P=73.2%, FP#=48. Halves FP while improving F1. Permanent addition to the pipeline.

### ~~Position-aware feature selection (error-word-only)~~ ✓ Done (Experiment 13)
Completed: Only counts features firing at error word last-token positions, not anywhere in clean text. Candidate pools are large (240–979 per type), confirming many features fire specifically at error positions. Combined with top-N at N=50, produces comparable results to old min_pair_ratio=0.3 approach. word_order improved +9% (55%→64%). Permanent addition.

### ~~Top-N feature selection (replace min_pair_ratio cutoff)~~ ✓ Done (Experiment 13)
Completed: Rank by pair count, take top N globally per type. Swept N ∈ [25, 50, 100]. N=50 is the sweet spot — N=25 too few, N=100 overfits (word_order/missing_word collapse to 0%). Partially validated hypothesis: word_order detection improved +9%, but overall F1 similar to old approach. The LR can learn from 50 features but not 100 with only 75–156 positive tokens.

### ~~Non-linear classifier (Random Forest / gradient boosting)~~ ✗ Failed (Experiment 14)
Tested RF (100 trees, max_depth=8, balanced) as drop-in replacement for LR. RF per-type F1=73.3% vs LR 81.6%. RF probability calibration clusters around 0.5, making threshold tuning impossible — t=0.8 drops recall to 35%. LR wins for sparse, low-data regimes. The subpopulation hypothesis is plausible but can't be validated with 75–156 positive tokens per type.

### ~~Negative-example features (clean-only activation counterweight)~~ ✓ Done (Experiment 15)
Broad negative selection failed badly (Round 1: F1 81.6→69.3%). Root cause: too many candidates (~9000 per type), selecting "common word" features. Diagnostic (Round 2): coefficient imbalance + class_weight="balanced" interaction. Fix (Round 3): paired negative selection (clean word at same position as error word) + only for paired types + only 5 features. Result: **F1=82.8%** (+1.2pp), modest but real improvement. The effect is fragile — only works at exactly N_neg=5. Best config: 50 positive + 5 paired negative features, balanced LR, C=1.0.

### ~~K-fold cross-validation~~ ✓ Done (Experiment 17)
Completed: 5-fold CV. True baseline: F1=79.6% ±2.4%. Revealed three tiers: rock solid (extra_word 93.7% ±4.5%, wtf 94.9% ±4.9%), reasonable (spelling 91.4% ±7.6%, grammar 65.5% ±11.1%), unreliable (word_choice 42.0% ±27.8%, word_order 29.3% ±17.3%). Previous single-split results were slightly optimistic. Future experiments must report mean ± std.

### ~~Scale to 6000+ pairs~~ ✓ Done (Experiment 20)
Completed: 6000 pairs (1000/type), extraction ~9 min (thanks to Exp 19 vectorization). F1: 79.6% → 84.6% (+5.0pp), variance halved (±2.4% → ±0.6%). word_order nearly doubled (29.3% → 58.8%), word_choice +25.6pp (42.0% → 67.6%). Also re-validated top_N sweep: top_N=100 now works (was overfitting at 600 pairs), giving +1.1pp F1 over top_N=50.

### ~~Word_order: only label second swapped word~~ ✓ Done at scale (Experiment 21)
Failed at 600 pairs (Exp 18: -19pp), succeeded at 6000 pairs (Exp 21: +1.9pp). Single-label is now the default.

### ~~Diversify grammar swap table~~ ✗ Failed at all scales (Experiments 18, 21)
Failed at 600 pairs (Exp 18: 65%→17%) AND at 6000 pairs (Exp 21: 77%→0%). Also tried cap=200 (partial recovery 50% but ±25% variance). The expanded table is fundamentally harmful — the compact original table with agreement/tense/pronoun swaps is the right granularity.

### ~~Retry data quality fixes after scaling~~ ✓ Done (Experiment 21)
Retried at 6000 pairs. Grammar diversity: confirmed harmful at any scale (0% detection). Word_order single-label: succeeded (+1.9pp). Also added repeat-letter corruption, long-word bias for spelling. Net F1: 84.8% (+0.2pp, within noise) but qualitatively better error coverage.

### ~~Compare 16k vs 65k vs 262k SAE widths~~ ✓ Done (Experiment 8)
Completed: 262k marginally best (F1=76.7% vs 75.4%, P=68.0% vs 66.0%, 31 vs 34 FPs at t=0.9). Feature counts similar across widths (~90 spelling, ~15 grammar). 65k found 0 features for missing_word. 262k extremely slow (per-layer eviction). **Conclusion: not worth the cost. Stick with 16k.**

### Word-level prediction aggregation
**Goal:** Instead of max-across-all-tokens for sentence-level scoring, aggregate per-word (max or mean of the word's tokens). This gives word-level error highlighting — more useful for a real UI. The last-token-only training (Exp 9) makes this natural since signal concentrates at word boundaries.
**Importance:** Medium — needed for production UX, but sentence-level detection is sufficient for experiments.

### ~~Next-token-after-word labeling~~ ✗ Failed (Experiment 10)
Tested labeling the first token *after* the error word (instead of the last token *of* the word). Hypothesis was that the model only "knows" the word is complete at the next token (space boundary). Results: FP# 25→37 at t=0.9, precision 71.6%→64.1%. The next-token signal is noisier — influenced by what the following word is, not just the error. Last-token labeling (Exp 9) remains best.

### Word_choice: use next-word activations (look-ahead context)
**Goal:** For word_choice errors specifically, the "wrongness" of a confusable word (their→there) may only become apparent from the *following* context — the word itself looks fine in isolation. Inspect activations of both the error word's last token and the next word's last token. Could combine both positions as features for the word_choice classifier, or label the next word as the detection point. This is different from Exp 10 (which tested next-token globally and failed) — here the hypothesis is that it matters specifically for semantic/contextual errors like word_choice, not for surface-level errors like spelling.
**Importance:** Medium — word_choice improved to 67.6% with more data (Exp 20) but still the weakest remaining type. This approach could help further.

### ~~Last-token-only for training and classification~~ ✓ Done (Experiment 11)
Extended Exp 9's last-token-only training to also cover clean text training tokens and prediction. Intermediate tokens within multi-token words oscillate between error/non-error representations — filtering to last-word tokens eliminates this noise. Results: spelling FP 8%→2.7%, overall FP# 25→22 at t=0.9, precision 76%→79.7% at t=0.95. Clear win, permanent change.

### ~~Missing word: label sentence-end punctuation instead of next word~~ — Category dropped (Experiment 16)
The `missing_word` category was removed entirely — 0% detection with threshold=1.0. The category is fundamentally not detectable at the word level with SAEs: the gap is a structural issue, not a token-level anomaly.

### ~~Separate spelling errors from grammar errors~~ ✓ Done (Experiment 4)
Completed: Spelling is 100% detected and classified. Grammar is detected (90%) but misclassified 78% of the time — features overlap with word_choice and word_order. Low detection types (missing_word: 67%) are kept as-is; when they fire correctly it's useful, the priority is keeping FP low rather than maximizing recall.

### Treat punctuation as separate words (tokenization + training)
**Goal:** Currently punctuation attached to words (e.g., "mat.") stays as one word via `text.split()`, and standalone punctuation tokens map to `None` (excluded from scoring/training). Instead, split punctuation into separate word entries — both in `token_to_word_index` and the UI. This enables future punctuation error categories (missing/excessive commas, periods, etc.) and makes the word-level output more granular.
**Importance:** Medium — prerequisite for punctuation error detection. No urgency until we add punctuation error types.

### Test on real-world text (not synthetic)
**Goal:** Run the classifier on naturally occurring errors (e.g., from typo corpora, student essays, social media). Synthetic errors may not represent the distribution the model saw during training.
**Importance:** Medium — critical before claiming real-world applicability, but needs the classifier to be more mature first.

### Attention-out vs MLP-out SAEs
**Goal:** Determine whether error detection is primarily driven by attention (cross-token pattern matching) or MLP (per-token knowledge). GemmaScope 2 provides `att` and `mlp` SAEs for this.
**Importance:** Medium — improves mechanistic understanding and could inform architecture of the fused model.

### Feature activation thresholds for binary classification
**Goal:** Turn the raw feature activations into a yes/no error detector per token. Find optimal threshold(s) for the top error-indicative features to maximize precision/recall.
**Importance:** Medium — bridges the gap between "interesting features" and a usable classifier.

### Pretrained vs instruction-tuned model comparison
**Goal:** Compare error-detection features between `gemma-3-1b-pt` and `gemma-3-1b-it`. The IT model may have stronger error awareness from RLHF.
**Importance:** Low — interesting for the paper but the PT model is more suitable for the fused-model goal.

### Cross-language error detection
**Goal:** Test whether the same features fire on spelling errors in other languages (e.g., German, Polish). Gemma 3 is multilingual.
**Importance:** Low — stretch goal, but could significantly broaden the impact of the research.

### Small neural network classifier
**Goal:** RF failed due to poor probability calibration (Exp 14), but the subpopulation hypothesis remains valid — some error subsets likely don't conform to LR's single linear boundary. A small MLP (e.g., 2 hidden layers, 32–64 units, sigmoid output) could capture non-linear boundaries while producing well-calibrated probabilities via sigmoid, unlike RF's tree-fraction approach. Trains with binary cross-entropy, same OVR structure. With 50 features and ~5k training tokens, a small NN should generalize — just need to regularize (dropout, early stopping).
**Importance:** Medium-low — LR works well and the bottleneck is more likely feature quality than classifier capacity. Worth revisiting after negative features and more data.

### Fuse LLM + SAE into a single optimized model
**Goal:** Truncate the LLM at the best layer and attach the SAE encoder, producing a single smaller model that takes text in and outputs error features directly.
**Importance:** Low (for now) — the end goal, but requires identifying the right layer/features first.
