# Lan-ick Project Instructions

## Project Overview

Research project exploring whether LLM internal activations (decoded via Sparse Autoencoders) can detect spelling and grammar errors — detection only, not correction. Uses Gemma 3 1B (pretrained) with GemmaScope 2 SAEs.

**Primary goal:** Detect errors while avoiding false positives. A false positive (underlining correct text) is a worse user experience than a missed error, so given any trade-off, prefer the option with fewer FPs — even at the cost of some recall.

**How to evaluate:** The headline metric is **combined precision** on held-out data (what fraction of sentences we flag as errored are actually errored), reported alongside **combined recall** and **combined F0.5** (which weights precision 2x over recall). Also report per-type detection and per-type FP, but treat per-type FP budgets as internal tuning knobs, not as the summary — users see the combined rate, because any type firing on clean text is a FP to them.

When comparing experiments, an improvement must move combined F0.5 (or combined precision at similar recall) by more than the per-fold std. F1 is still useful as a sanity check but is not the target — it treats precision and recall symmetrically, which doesn't match the UX.

## Architecture

```
src/
  classifier.py      # Reusable error classifier (SAE-based detection)
  model.py           # Model loading and activation extraction
  data.py            # Synthetic data generation and caching
experiments/
  run.py             # Single entry point for the current experiment
  log.md             # Experiment log (append-only, never delete entries)
  ideas.md           # Future experiment ideas (keep up to date)
temp/                # All generated/cached artifacts (gitignored, fully disposable)
  data/              # Synthetic datasets (keyed by generation params)
  results/           # Experiment outputs (JSON, plots, etc.)
  models/            # Downloaded model weights and SAE caches
```

## Experiment Workflow

Every experiment MUST follow this protocol:

1. **Pre-experiment**: Append an entry to `experiments/log.md` with:
   - Date/time (ISO 8601)
   - Experiment title
   - Goal and scope description
   - Hypothesis or what we expect to learn
   - Key parameters (layers, SAE width, dataset size, etc.)

2. **Implement**: Modify `experiments/run.py`, `src/` modules, or any other code as needed. Previous experiment code is NOT sacred — each experiment is preserved as a git commit, so destructive changes to both the experiment runner and source modules are welcomed. Override, rewrite, or delete freely.

3. **Run**: Execute via `uv run python3 -m experiments.run`. All cached data lives in `temp/` and is keyed so that parameter changes auto-regenerate stale data. **Always run synchronously** (e.g., `mode=sync` with a very large timeout like 90000000ms) — never launch experiments in the background/async and poll for results, as this wastes context and makes it harder to react to output.

4. **Critically analyze results**: Before writing conclusions, interrogate the results:
   - **Is this expected?** If not, why not? What assumption was wrong?
   - **Why did it work/fail?** Dig into the numbers — coefficient magnitudes, train vs test gaps, per-type breakdowns. Don't just report aggregate metrics.
   - **Did we try everything?** Could a different hyperparameter, a refined selection criterion, or a variation on the approach change the outcome? Are there obvious follow-ups within the same experiment?
   - **Can we push further?** Even for successful experiments — is there a way to squeeze more out of an idea that's working?
   - If new ideas emerge, **iterate**: go back to step 2, adjust the approach, and re-run. A single experiment entry in the log can have multiple rounds of results. It's fine if one experiment takes 10x longer when that depth leads to real understanding. Log each iteration's results and reasoning in the same log entry.
   - Update `experiments/ideas.md` with any new ideas that emerged, even from successful experiments.

5. **Post-experiment**: Update the log entry with:
   - Results summary (quantitative) for each iteration
   - Analysis and reasoning for each round
   - Final conclusions and observations
   - Git commit hash (added after committing)

6. **Update ideas**: Review `experiments/ideas.md` — add new ideas that emerged, adjust priorities based on findings, remove ideas that were completed or became irrelevant.

7. **Commit**: Stage and commit all source changes. The commit message should reference the experiment title. After committing, go back and add the commit hash to the log entry, then amend the commit.

8. **Cleanup**: Remove experiment-specific complexity from `src/` and `experiments/run.py` that isn't justified by results. If the experiment failed or the gains were marginal, revert `src/` changes and simplify `run.py` back to the best known pipeline. Each experiment is preserved as a git commit, so nothing is lost. Commit the cleanup separately so the codebase at HEAD reflects the best known approach, not accumulated scaffolding from past experiments.

### Experiment Code Style

`experiments/run.py` should read **top-to-bottom** like a script: load data → set up model → run analysis → print/save results. It should compose building blocks from `src/` rather than implementing low-level logic. Keep it easy to scan and understand at a glance — someone reading the experiment should see the "what" without getting lost in the "how".

### Failed Experiments

If an experiment does not bring meaningful improvement, **favor the simpler approach**. Don't hesitate to revert changes to `src/` modules when the added complexity isn't justified by results. Each experiment is preserved as a git commit, so nothing is lost — but the codebase at HEAD should reflect the best known approach, not accumulated complexity from failed ideas.

## Data Caching

All temporary/generated files go in `temp/` — never in the project root. The `temp/` folder is gitignored. Deleting it should have no effect other than requiring regeneration on next run.

Caching uses `src/cache.py` with the interface:

```python
result = cached(name, version, generator_fn)
```

- `name`: Human-readable artifact name (e.g., `"test_pairs"`, `"pair_features"`).
- `version`: String that determines cache validity. When it changes, the old cache is ignored and `generator_fn` re-runs.
- `generator_fn`: Zero-arg callable that produces the data.

Files are stored as `temp/data/{name}__{version}.pkl`.

### Versioning strategy

There are two reasons to invalidate cache:

1. **Logic changes** (the generator function was modified): bump a manual version string, e.g., `"v1"` → `"v2"`.
2. **Parameter changes** (different layers, width, dataset): encode the parameters in the version string so it auto-invalidates.

For example, feature extraction results depend on both logic and parameters:

```python
EXTRACT_VERSION = "v1"  # bump when extract_text_features logic changes
EXTRACT_CACHE_KEY = f"{EXTRACT_VERSION}_layers={'_'.join(map(str, LAYERS))}_w{WIDTH}"
all_features = cached("pair_features", EXTRACT_CACHE_KEY, extract_all)
```

Changing `LAYERS` or `WIDTH` produces a different cache key automatically. Changing the extraction logic requires bumping `EXTRACT_VERSION`.

### What to cache vs. not

- **Cache**: Anything that runs text through the LLM/SAE (expensive, minutes). This includes per-text feature extraction.
- **Don't cache**: Training (finding important features), evaluation (confusion matrix), and any analysis over cached features. These are cheap (seconds) and are the part we iterate on frequently.

## Classifier Design

The classifier in `src/classifier.py` must be abstracted from the experiment harness:
- It should accept raw text and return error detection results.
- Internally it loads the model and SAEs, extracts activations, and applies the detection logic.
- It must be reusable: the experiment calls it, but later a standalone app will too.
- The interface can evolve between experiments (e.g., changing what errors are reported or how results are structured). Since each experiment is committed, breaking changes are fine.

## Future-Proofing

The long-term goal is to bake the pipeline (LLM truncated at the relevant layer + SAE encoder) into a single optimized model. Current code should:
- Keep model loading and activation extraction in `src/model.py` as a separate concern from classification logic.
- Hardcoding layer indices, feature IDs, or thresholds is fine during experimentation — these values will change between experiments. Just keep them in obvious, easy-to-find places (e.g., config dicts, module-level constants) rather than buried in logic.
- Make it easy to later replace the "load full LLM + separate SAE" pipeline with a fused model.

## Technical Stack

- **Runtime**: Python 3.12+, uv (NOT pip) for package management
- **Model**: Gemma 3 1B pretrained (`google/gemma-3-1b-pt`) via HuggingFace transformers
- **SAEs**: GemmaScope 2 via SAELens (`gemma-scope-2-1b-pt-res-all` for all layers, `gemma-scope-2-1b-pt-res` for 4 subset layers with more width options)
- **GPU**: 8GB VRAM budget — model ~2GB, each 16k SAE ~0.15GB
- **No TransformerLens**: We use HuggingFace directly with `output_hidden_states=True`

## Conventions

- Run experiments: `uv run python3 -m experiments.run`
- Use `torch.no_grad()` for all inference — we never train
- Use bfloat16 for the LLM, float32 for SAE encoding
- Keep experiment scripts deterministic where possible (fixed seeds for any randomness)
- Secrets (API keys) go in project root, gitignored — never in `src/` or `experiments/`

## What NOT to Do

- Don't put generated data, JSON results, or model caches in the project root
- Don't create one-off throwaway scripts in the root — exploration code goes in a scratch notebook or gets deleted
- Don't modify `experiments/log.md` entries retroactively (append-only, except for adding the commit hash)
