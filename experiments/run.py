"""Experiment 3: Higher thresholds and false positive analysis.

Flow: generate pairs → split → extract features (cached) → select features →
      train token-level LR → sweep thresholds → analyze FPs → save results.
"""

import json
from pathlib import Path

from src.cache import cached, TEMP_DIR
from src.data import generate_synthetic_pairs, split_train_test
from src.model import extract_text_features
from src.classifier import select_features, train, evaluate, predict_sentence

# --------------- Experiment parameters ---------------

LAYERS = [5, 10, 13, 17, 22]
WIDTH = "16k"
TRAIN_RATIO = 0.75
SPLIT_SEED = 42
MIN_PAIR_RATIO = 0.5

N_PAIRS = 300
MIN_WORDS = 8
MAX_WORDS = 20
DATA_SEED = 42

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

DATA_VERSION = "v3"
EXTRACT_VERSION = "v1"

RESULTS_DIR = TEMP_DIR / "results"

EXTRACT_CACHE_KEY = f"{EXTRACT_VERSION}_v2_n{N_PAIRS}_layers={'_'.join(map(str, LAYERS))}_w{WIDTH}"


# --------------- Main ---------------

def main():
    # 1. Load data and features (all cached)
    all_pairs = cached(
        "synthetic_pairs", DATA_VERSION,
        lambda: generate_synthetic_pairs(N_PAIRS, MIN_WORDS, MAX_WORDS, DATA_SEED),
    )
    print(f"Loaded {len(all_pairs)} pairs")

    train_idx, test_idx = split_train_test(all_pairs, TRAIN_RATIO, SPLIT_SEED)
    print(f"Split: {len(train_idx)} train, {len(test_idx)} test")

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
    test_pairs = [all_pairs[i] for i in test_idx]

    # 2. Select features and train classifier
    error_feats = select_features(train_features, LAYERS, min_pair_ratio=MIN_PAIR_RATIO)
    total = sum(len(v) for v in error_feats.values())
    print(f"\nSelected {total} error features")

    print("Training token-level classifier...")
    classifier = train(train_features, train_pairs, LAYERS, error_feats)
    print(f"{classifier.summary()}")

    # 3. Extended threshold sweep
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
        print(f"  threshold={threshold}: {metrics.summary()}{marker}")

    classifier.sentence_threshold = best_threshold
    print(f"\n{'='*60}")
    print(f"BEST RESULT (threshold={best_threshold})")
    print(f"{'='*60}")
    print(best_metrics.confusion_str())
    print()
    print(best_metrics.summary())

    # 4. False positive analysis at best threshold
    print(f"\n{'='*60}")
    print(f"FALSE POSITIVE ANALYSIS (threshold={best_threshold})")
    print(f"{'='*60}")

    fp_details = []
    for pf, pair in zip(test_features, test_pairs):
        pred = predict_sentence(pf["clean"], classifier)
        if pred.has_errors:
            # Find top-scoring tokens
            top_tokens = sorted(pred.token_predictions, key=lambda t: t.p_error, reverse=True)[:5]
            top_str = ", ".join(f"'{t.token}'={t.p_error:.3f}" for t in top_tokens)
            print(f"\n  FP: {pair.clean}")
            print(f"    max P(error)={pred.max_p_error:.3f}, top tokens: {top_str}")
            fp_details.append({
                "sentence": pair.clean,
                "max_p_error": round(pred.max_p_error, 4),
                "top_tokens": [{"token": t.token, "p_error": round(t.p_error, 4), "pos": t.position}
                               for t in top_tokens],
            })

    print(f"\n  Total FPs: {len(fp_details)}")

    # 5. Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "experiment_3_thresholds.json"

    results = {
        "params": {
            "layers": LAYERS, "width": WIDTH,
            "n_pairs": N_PAIRS,
            "train_ratio": TRAIN_RATIO, "split_seed": SPLIT_SEED,
            "min_pair_ratio": MIN_PAIR_RATIO,
            "n_train": len(train_idx), "n_test": len(test_idx),
        },
        "threshold_sweep": {str(k): v for k, v in sweep_results.items()},
        "best_threshold": best_threshold,
        "best_metrics": best_metrics.to_dict(),
        "false_positives": fp_details,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
