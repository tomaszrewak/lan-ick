# Experiment Ideas

Future experiment proposals, ordered roughly by priority/importance. Update this file as new ideas emerge or get completed.

---

### ~~Scale up synthetic data (100s–1000s of pairs)~~ ✓ Done (Experiment 1)
Completed: 300 pairs, F1=70.1%. Features dropped from 204→41 (more selective). Bottleneck is now threshold/classifier.

### ~~Tune threshold + use activation magnitudes~~ ✓ Done (Experiment 2)
Completed: Token-level LR with activation magnitudes + threshold sweep. F1=92.0% (up from 70.1%). Perfect recall at all thresholds.

### ~~Explore higher thresholds and FP analysis~~ ✓ Done (Experiment 3)
Completed: Pushed to 0.95 — F1=94.2%, P=91.2%, R=97.3%. Dominant FP source: `n't` contractions (4/7 FPs). Secondary: rare proper nouns. Contraction handling would eliminate most FPs.

### Handle contractions and tokenizer artifacts
**Goal:** The `n't` split (`' n'` + `'t'`) causes 4/7 false positives at threshold=0.95. Pre-process contractions before error scoring (expand or whitelist known patterns) to eliminate this systematic FP source.
**Importance:** High — would reduce FP by ~57% with minimal effort, likely pushing F1 above 96%.

### ~~Word order swap and similar-word replacement errors~~ ✓ Done (Experiment 4)
Completed as part of full 6-type expansion. Word order detected 82%, word choice 82%, but type classification accuracy lower (44%, 67%). Grammar detected 90% but misclassified 78% of the time.

### Smarter synthetic data generation
**Goal:** Current corruption functions are naive — e.g., `corrupt_missing_word` randomly deletes any word, but deleting an adjective ("the big dog" → "the dog") is grammatically valid and undetectable. Invest in linguistically-aware generation: delete function words (articles, prepositions, auxiliaries) rather than content words; for grammar errors, target known agreement patterns (subject-verb, determiner-noun); for word choice, ensure the context makes the substitution clearly wrong. Better training data = better classifier, regardless of architecture.
**Importance:** Very high — garbage in, garbage out. Low detection on some types may partly be a data quality issue.

### Scale up multi-class training data
**Goal:** Exp 4 has only 50 pairs/type → 37 training pairs/type → tiny error token counts (grammar: 39 tokens, extra_word: 43). Scale to 150+ pairs per type (900+ total) to give the multi-class LR enough signal, especially for grammar and missing_word.
**Importance:** Very high — data starvation is the likely cause of Exp 4's poor type classification.

### ~~One-vs-rest binary classifiers per error type~~ ✓ Done (Experiment 5)
Completed: 6 binary LRs trained. Spelling (92% det, 8% FP) and extra_word (91% det, 1.3% FP) work well. Grammar/word_order/missing_word have high FP — not separable with current features. Combined F1=72% (no improvement over Exp 4). Per-type insight is the main value.

### FP-constrained threshold selection
**Goal:** Instead of optimizing F1 per type (which picks low thresholds for weak classifiers), set a max FP budget per type (e.g., 5%) and find the highest detection rate within that budget. This would let us ship only the types that are reliably separable (spelling, extra_word) and suppress the rest.
**Importance:** High — directly addresses the FP flood from weak classifiers seen in Exp 5.

### Compare 16k vs 65k vs 262k SAE widths
**Goal:** Determine if wider SAEs produce sharper, more specific error-detection features. The 16k SAE may lump multiple error types into one feature; wider SAEs might separate them.
**Importance:** High — directly affects detection granularity and the eventual classifier quality.

### Test with only layers 5–13 (drop upper layers)
**Goal:** Check if restricting to layers 5, 10, 13 (29 features) still matches the full 5-layer result (41 features). Layers 17 and 22 contribute only 13 features and the signal tapers off there. If performance holds, the fused model can be truncated at layer 13 instead of 22 — roughly halving the LLM portion and making classification significantly faster.
**Importance:** High — directly determines how much of the model we can strip for the fused deployment.

### Layer sweep across all 26 layers
**Goal:** Map exactly where error-detection signal emerges and peaks. Current experiment samples 5 layers; a full sweep would reveal the optimal layer(s) to use for the final classifier.
**Importance:** High — needed to decide which layer(s) to bake into the fused model.

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

### Fuse LLM + SAE into a single optimized model
**Goal:** Truncate the LLM at the best layer and attach the SAE encoder, producing a single smaller model that takes text in and outputs error features directly.
**Importance:** Low (for now) — the end goal, but requires identifying the right layer/features first.
