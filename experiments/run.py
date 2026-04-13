"""Experiment 4: Multi-class error detection (6 error types).

Flow: generate balanced pairs → split → extract features (cached) → select features →
      train multi-class LR → threshold sweep → per-type analysis → save results.
"""

import json
from collections import defaultdict
from pathlib import Path

from src.cache import cached, TEMP_DIR
from src.data import ErrorType, generate_synthetic_pairs, split_train_test
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

DATA_VERSION = "v4"
EXTRACT_VERSION = "v2"

RESULTS_DIR = TEMP_DIR / "results"

EXTRACT_CACHE_KEY = f"{EXTRACT_VERSION}_v4_n{N_PAIRS}_layers={'_'.join(map(str, LAYERS))}_w{WIDTH}"


# --------------- Main ---------------

def main():
    # 1. Load data and features
    all_pairs = cached(
        "synthetic_pairs", DATA_VERSION,
        lambda: generate_synthetic_pairs(N_PAIRS, MIN_WORDS, MAX_WORDS, DATA_SEED),
    )
    print(f"Loaded {len(all_pairs)} pairs")

    # Show type distribution
    type_counts = defaultdict(int)
    for p in all_pairs:
        type_counts[p.error_type.value] += 1
    for et, cnt in sorted(type_counts.items()):
        print(f"  {et}: {cnt}")

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
    error_feats = select_features(train_features, LAYERS, train_pairs, min_pair_ratio=MIN_PAIR_RATIO)
    total = sum(len(v) for v in error_feats.values())
    print(f"\nSelected {total} error features")

    print("Training multi-class classifier...")
    classifier = train(train_features, train_pairs, LAYERS, error_feats)
    print(f"{classifier.summary()}")

    # 3. Threshold sweep (binary: error vs clean)
    print(f"\n{'='*60}")
    print("THRESHOLD SWEEP (binary detection)")
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

    # 4. Per-type analysis at best threshold
    print(f"\n{'='*60}")
    print(f"PER-TYPE ANALYSIS (threshold={best_threshold})")
    print(f"{'='*60}")

    type_stats = defaultdict(lambda: {"total": 0, "detected": 0, "correct_type": 0})

    for pf, pair in zip(test_features, test_pairs):
        et = pair.error_type.value
        type_stats[et]["total"] += 1
        pred = predict_sentence(pf["error"], classifier)
        if pred.has_errors:
            type_stats[et]["detected"] += 1
            if pred.predicted_type == pair.error_type:
                type_stats[et]["correct_type"] += 1

    print(f"\n  {'Type':<15} {'Total':>6} {'Detected':>9} {'Det%':>6} {'CorrectType':>12} {'TypeAcc%':>9}")
    print(f"  {'-'*58}")
    for et in ErrorType:
        s = type_stats[et.value]
        det_pct = s["detected"] / s["total"] * 100 if s["total"] else 0
        type_acc = s["correct_type"] / s["detected"] * 100 if s["detected"] else 0
        print(f"  {et.value:<15} {s['total']:>6} {s['detected']:>9} {det_pct:>5.1f}% {s['correct_type']:>12} {type_acc:>8.1f}%")

    # 5. FP analysis (clean sentences flagged as errors)
    print(f"\n{'='*60}")
    print(f"FALSE POSITIVE EXAMPLES (threshold={best_threshold})")
    print(f"{'='*60}")

    fp_count = 0
    fp_details = []
    for pf, pair in zip(test_features, test_pairs):
        pred = predict_sentence(pf["clean"], classifier)
        if pred.has_errors:
            fp_count += 1
            top_tokens = sorted(pred.token_predictions, key=lambda t: t.p_error, reverse=True)[:3]
            top_str = ", ".join(f"'{t.token}'={t.p_error:.3f}({t.predicted_type.value if t.predicted_type else '?'})" for t in top_tokens)
            if fp_count <= 10:
                print(f"\n  FP: {pair.clean}")
                print(f"    max P(error)={pred.max_p_error:.3f}, predicted={pred.predicted_type.value if pred.predicted_type else '?'}")
                print(f"    top tokens: {top_str}")
            fp_details.append({
                "sentence": pair.clean,
                "max_p_error": round(pred.max_p_error, 4),
                "predicted_type": pred.predicted_type.value if pred.predicted_type else None,
            })

    print(f"\n  Total FPs: {fp_count}")

    # 6. Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "experiment_4_multiclass.json"

    results = {
        "params": {
            "layers": LAYERS, "width": WIDTH,
            "n_pairs": N_PAIRS,
            "train_ratio": TRAIN_RATIO, "split_seed": SPLIT_SEED,
            "min_pair_ratio": MIN_PAIR_RATIO,
            "n_train": len(train_idx), "n_test": len(test_idx),
            "error_types": [et.value for et in ErrorType],
        },
        "threshold_sweep": {str(k): v for k, v in sweep_results.items()},
        "best_threshold": best_threshold,
        "best_metrics": best_metrics.to_dict(),
        "per_type": {k: v for k, v in type_stats.items()},
        "false_positives": fp_details,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
