"""Experiment: Baseline error-only feature detection with Gemma 3 1B + GemmaScope 2.

Flow: load data → split train/test → extract features (cached) → train → evaluate.
"""

import json
from pathlib import Path

from src.cache import cached, TEMP_DIR
from src.data import generate_test_pairs, split_train_test
from src.model import extract_text_features
from src.classifier import train, evaluate

# --------------- Experiment parameters ---------------

LAYERS = [5, 10, 13, 17, 22]
WIDTH = "16k"
TRAIN_RATIO = 0.75
SPLIT_SEED = 42
MIN_PAIR_RATIO = 0.5  # fraction of training pairs a feature must be error-only in

DATA_VERSION = "v1"           # bump when generate_test_pairs changes
EXTRACT_VERSION = "v1"        # bump when extract_text_features logic changes

RESULTS_DIR = TEMP_DIR / "results"

# Cache key for feature extraction encodes model parameters so
# changing LAYERS or WIDTH auto-invalidates.
EXTRACT_CACHE_KEY = f"{EXTRACT_VERSION}_layers={'_'.join(map(str, LAYERS))}_w{WIDTH}"


# --------------- Main ---------------

def main():
    # 1. Load all pairs
    all_pairs = cached("test_pairs", DATA_VERSION, generate_test_pairs)
    print(f"Loaded {len(all_pairs)} pairs")

    # 2. Split train/test (deterministic, instant)
    train_idx, test_idx = split_train_test(all_pairs, TRAIN_RATIO, SPLIT_SEED)
    print(f"Split: {len(train_idx)} train, {len(test_idx)} test "
          f"(train={train_idx}, test={test_idx})")

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

    # 4. Train: find error-indicative features
    classifier = train(train_features, LAYERS, min_pair_ratio=MIN_PAIR_RATIO)
    print(f"\n{classifier.summary()}")
    for layer in LAYERS:
        fids = sorted(classifier.error_features.get(layer, set()))
        print(f"  Layer {layer}: {len(fids)} error features"
              + (f" (top: {fids[:5]}...)" if len(fids) > 5 else f" {fids}"))

    # 5. Evaluate on held-out test pairs
    metrics = evaluate(test_features, classifier)
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION")
    print(f"{'='*60}")
    print(metrics.confusion_str())
    print()
    print(metrics.summary())

    # 6. Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "baseline_experiment.json"

    results = {
        "params": {
            "layers": LAYERS, "width": WIDTH,
            "train_ratio": TRAIN_RATIO, "split_seed": SPLIT_SEED,
            "min_pair_ratio": MIN_PAIR_RATIO,
            "n_train": len(train_idx), "n_test": len(test_idx),
            "train_indices": train_idx, "test_indices": test_idx,
        },
        "classifier": {
            "total_features": classifier.total_features,
            "per_layer": {
                layer: sorted(classifier.error_features.get(layer, set()))
                for layer in LAYERS
            },
        },
        "metrics": metrics.to_dict(),
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
