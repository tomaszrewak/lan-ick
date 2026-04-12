# Experiment Ideas

Future experiment proposals, ordered roughly by priority/importance. Update this file as new ideas emerge or get completed.

---

### Scale up synthetic data (100s–1000s of pairs)
**Goal:** Validate that error-only features generalize beyond 8 hand-crafted pairs. Reduce risk that current findings are artifacts of small sample size.
**Importance:** High — our current 8-pair results are suggestive but not statistically robust. This is the foundation for everything else.

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
