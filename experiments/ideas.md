# Experiment Ideas

Future experiment proposals, ordered roughly by priority/importance. Update this file as new ideas emerge or get completed.

---

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

### K-fold cross-validation
**Goal:** We use a single 75/25 split. With 25 test pairs per type at 600 pairs, a single pair swinging either way changes detection by 4%. Results may be noisy and we can't tell signal from variance. K-fold CV (e.g., 5-fold) would give mean ± std for all metrics and reveal if we're overfitting feature selection to the training split.
**Importance:** Medium — critical for trusting results. Some past experiment "improvements" may have been noise.

### Scale to 6000+ pairs
**Goal:** With 600 pairs, feature extraction takes ~20 min (cached, one-time). 6000 pairs would be ~3 hours but gives 1000 per type, ~750 training pairs per type, and much more stable feature selection and classifier training. Weak types (missing_word, word_order) are especially starved for data.
**Importance:** Medium — straightforward scaling, but requires the feature selection and classifier improvements first to be worth the compute cost.

### ~~Compare 16k vs 65k vs 262k SAE widths~~ ✓ Done (Experiment 8)
Completed: 262k marginally best (F1=76.7% vs 75.4%, P=68.0% vs 66.0%, 31 vs 34 FPs at t=0.9). Feature counts similar across widths (~90 spelling, ~15 grammar). 65k found 0 features for missing_word. 262k extremely slow (per-layer eviction). **Conclusion: not worth the cost. Stick with 16k.**

### Test with only layers 5–13 (drop upper layers)
**Goal:** Check if restricting to layers 5, 10, 13 (29 features) still matches the full 5-layer result (41 features). Layers 17 and 22 contribute only 13 features and the signal tapers off there. If performance holds, the fused model can be truncated at layer 13 instead of 22 — roughly halving the LLM portion and making classification significantly faster.
**Importance:** High — directly determines how much of the model we can strip for the fused deployment.

### Word-level prediction aggregation
**Goal:** Instead of max-across-all-tokens for sentence-level scoring, aggregate per-word (max or mean of the word's tokens). This gives word-level error highlighting — more useful for a real UI. The last-token-only training (Exp 9) makes this natural since signal concentrates at word boundaries.
**Importance:** Medium — needed for production UX, but sentence-level detection is sufficient for experiments.

### ~~Next-token-after-word labeling~~ ✗ Failed (Experiment 10)
Tested labeling the first token *after* the error word (instead of the last token *of* the word). Hypothesis was that the model only "knows" the word is complete at the next token (space boundary). Results: FP# 25→37 at t=0.9, precision 71.6%→64.1%. The next-token signal is noisier — influenced by what the following word is, not just the error. Last-token labeling (Exp 9) remains best.

### ~~Last-token-only for training and classification~~ ✓ Done (Experiment 11)
Extended Exp 9's last-token-only training to also cover clean text training tokens and prediction. Intermediate tokens within multi-token words oscillate between error/non-error representations — filtering to last-word tokens eliminates this noise. Results: spelling FP 8%→2.7%, overall FP# 25→22 at t=0.9, precision 76%→79.7% at t=0.95. Clear win, permanent change.

### Layer sweep across all 26 layers
**Goal:** Map exactly where error-detection signal emerges and peaks. Current experiment samples 5 layers; a full sweep would reveal the optimal layer(s) to use for the final classifier.
**Importance:** High — needed to decide which layer(s) to bake into the fused model.

### Missing word: label sentence-end punctuation instead of next word
**Goal:** For missing_word errors, the current approach labels the word after the gap — but that word is valid, just in a surprising position. The signal may be too diffuse. Alternative: only generate missing_word errors in positions where the gap is followed by sentence-end punctuation, and label the punctuation token. By the end of the sentence, the model has full context to "know" a function word was missing somewhere. This shifts from local detection to sentence-level detection for this type only.
**Importance:** Low — missing_word is the weakest type (28% det, 1 feature) and may be fundamentally hard for SAEs. Worth trying but not critical.

### ~~Separate spelling errors from grammar errors~~ ✓ Done (Experiment 4)
Completed: Spelling is 100% detected and classified. Grammar is detected (90%) but misclassified 78% of the time — features overlap with word_choice and word_order. Low detection types (missing_word: 67%) are kept as-is; when they fire correctly it's useful, the priority is keeping FP low rather than maximizing recall.

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
