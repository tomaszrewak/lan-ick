# Lan-ick Project Instructions

## Project Overview

Research project exploring whether LLM internal activations (decoded via Sparse Autoencoders) can detect spelling and grammar errors — detection only, not correction. Uses Gemma 3 1B (pretrained) with GemmaScope 2 SAEs.

## Architecture

```
src/
  classifier.py      # Reusable error classifier (SAE-based detection)
  model.py           # Model loading and activation extraction
  data.py            # Synthetic data generation and caching
experiments/
  run.py             # Single entry point for the current experiment
  log.md             # Experiment log (append-only, never delete entries)
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

3. **Run**: Execute via `experiments/run.py`. All cached data lives in `temp/` and is keyed so that parameter changes auto-regenerate stale data.

4. **Post-experiment**: Update the log entry with:
   - Results summary (quantitative)
   - Conclusions and observations
   - Git commit hash (added after committing)

5. **Commit**: Stage and commit all source changes. The commit message should reference the experiment title. After committing, go back and add the commit hash to the log entry, then amend the commit.

## Data Caching

- All temporary/generated files go in `temp/` — never in the project root.
- Cache keys should be deterministic hashes of the generation parameters (e.g., dataset config, model name, SAE width). When parameters change, stale cache is ignored and regenerated.
- The `temp/` folder is gitignored. Deleting it should have no effect other than requiring regeneration on next run.

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

- Entry point for experiments: `uv run python3 experiments/run.py`
- Use `torch.no_grad()` for all inference — we never train
- Use bfloat16 for the LLM, float32 for SAE encoding
- Keep experiment scripts deterministic where possible (fixed seeds for any randomness)
- Secrets (API keys) go in project root, gitignored — never in `src/` or `experiments/`

## What NOT to Do

- Don't put generated data, JSON results, or model caches in the project root
- Don't create one-off throwaway scripts in the root — exploration code goes in a scratch notebook or gets deleted
- Don't modify `experiments/log.md` entries retroactively (append-only, except for adding the commit hash)
