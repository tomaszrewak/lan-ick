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

### Word order swap and similar-word replacement errors
**Goal:** Add non-character-level error types: word order swaps ("the went she to store") and similar-word substitution ("their" → "there", "affect" → "effect"). Test whether SAE features detect semantic/syntactic errors, not just character-level.
**Importance:** High — current errors are all character-level, limiting generalizability.

### Compare 16k vs 65k vs 262k SAE widths
**Goal:** Determine if wider SAEs produce sharper, more specific error-detection features. The 16k SAE may lump multiple error types into one feature; wider SAEs might separate them.
**Importance:** High — directly affects detection granularity and the eventual classifier quality.

### Layer sweep across all 26 layers
**Goal:** Map exactly where error-detection signal emerges and peaks. Current experiment samples 5 layers; a full sweep would reveal the optimal layer(s) to use for the final classifier.
**Importance:** High — needed to decide which layer(s) to bake into the fused model.

### Separate spelling errors from grammar errors
**Goal:** Test whether SAE features distinguish misspellings (e.g., "teh") from grammatical errors (e.g., "she goed"). Currently our test pairs mix both types.
**Importance:** Medium — important for understanding what the model actually detects and for reporting different error categories.

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
