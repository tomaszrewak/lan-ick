"""Experiment 2: Token-level logistic regression classifier.

Flow: generate pairs → split → extract features (cached) → select features →
      train token-level LR → sweep thresholds → evaluate → save results.
"""

import json
from pathlib import Path

from src.cache import cached, TEMP_DIR
from src.data import generate_synthetic_pairs, split_train_test
from src.model import extract_text_features
from src.classifier import select_features, train, evaluate

# --------------- Experiment parameters ---------------

LAYERS = [5, 10, 13, 17, 22]
WIDTH = "16k"
TRAIN_RATIO = 0.75
SPLIT_SEED = 42
MIN_PAIR_RATIO = 0.5  # fraction of training pairs a feature must be error-only in

N_PAIRS = 300
MIN_WORDS = 8
MAX_WORDS = 20
DATA_SEED = 42

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

DATA_VERSION = "v3"           # bump when data generation changes (now with error_word_indices)
EXTRACT_VERSION = "v1"        # bump when extract_text_features logic changes

RESULTS_DIR = TEMP_DIR / "results"

# Feature extraction depends on text content, not data format.
# Text content is identical to v2 (same seed), so reuse that cache.
EXTRACT_CACHE_KEY = f"{EXTRACT_VERSION}_v2_n{N_PAIRS}_layers={'_'.join(map(str, LAYERS))}_w{WIDTH}"


# --------------- Main ---------------

def main():
    # 1. Generate synthetic pairs (v3 — with error_word_indices)
    all_pairs = cached(
        "synthetic_pairs", DATA_VERSION,
        lambda: generate_synthetic_pairs(N_PAIRS, MIN_WORDS, MAX_WORDS, DATA_SEED),
    )
    print(f"Loaded {len(all_pairs)} pairs")

    # 2. Split train/test (deterministic, instant)
    train_idx, test_idx = split_train_test(all_pairs, TRAIN_RATIO, SPLIT_SEED)
    print(f"Split: {len(train_idx)} train, {len(test_idx)} test")

    # 3. Extract features for ALL pairs (cached — expensive LLM + SAE part)
    def extract_all():
        results = []
        for i, pair in enumerate(all_pairs):
            print(f"  [{i+1}/{len(all_pairs)}] {pair.error[:50]}...")
            clean_feats = extract_text_features(pair.clean, LAYERS, WIDTH)
            error_feats = extract_text_features(pair.error, LAYERS, WIDTH)
            results.append({"clean": clean_feats, "error": error_feats})
        return results

    all_features = cached("pair_features", EXTRACT_CACHE_KEY, extract_all)

    train_features = [all_features[i] for i in train_idx]
    test_features = [all_features[i] for i in test_idx]
    train_pairs = [all_pairs[i] for i in train_idx]

    # 4. Select error-indicative features (same as Exp 1)
    error_feats = select_features(train_features, LAYERS, min_pair_ratio=MIN_PAIR_RATIO)
    total = sum(len(v) for v in error_feats.values())
    print(f"\nSelected {total} error features:")
    for layer in LAYERS:
        fids = sorted(error_feats.get(layer, set()))
        print(f"  Layer {layer}: {len(fids)} features"
              + (f" (top: {fids[:5]}...)" if len(fids) > 5 else f" {fids}"))

    # 5. Train token-level logistic regression
    print("\nTraining token-level classifier...")
    classifier = train(train_features, train_pairs, LAYERS, error_feats)
    print(f"{classifier.summary()}")

    # 6. Sweep thresholds and evaluate
    print(f"\n{'='*60}")
    print("THRESHOLD SWEEP")
    print(f"{'='*60}")

    best_f1 = 0.0
    best_threshold = 0.5
    best_metrics = None
    sweep_results = {}

    for threshold in THRESHOLDS:
        classifier.sentence_threshold = threshold
        metrics = evaluate(test_features, classifier)
        sweep_results[threshold] = metrics.to_dict()
        marker = ""
        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            best_threshold = threshold
            best_metrics = metrics
            marker = " ← best"
        print(f"  threshold={threshold:.1f}: {metrics.summary()}{marker}")

    # Show best result
    classifier.sentence_threshold = best_threshold
    print(f"\n{'='*60}")
    print(f"BEST RESULT (threshold={best_threshold})")
    print(f"{'='*60}")
    print(best_metrics.confusion_str())
    print()
    print(best_metrics.summary())

    # 7. Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "experiment_2_token_lr.json"

    results = {
        "params": {
            "layers": LAYERS, "width": WIDTH,
            "n_pairs": N_PAIRS, "min_words": MIN_WORDS, "max_words": MAX_WORDS,
            "data_seed": DATA_SEED,
            "train_ratio": TRAIN_RATIO, "split_seed": SPLIT_SEED,
            "min_pair_ratio": MIN_PAIR_RATIO,
            "n_train": len(train_idx), "n_test": len(test_idx),
        },
        "feature_selection": {
            "total_features": total,
            "per_layer": {
                layer: sorted(error_feats.get(layer, set()))
                for layer in LAYERS
            },
        },
        "threshold_sweep": {str(k): v for k, v in sweep_results.items()},
        "best_threshold": best_threshold,
        "best_metrics": best_metrics.to_dict(),
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
