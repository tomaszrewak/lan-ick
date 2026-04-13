"""Experiment 6: Per-type feature sets for OVR classifiers.

Flow: load cached data+features → select features per type → train 6 binary LRs
      (each with its own features) → per-type threshold sweep → combined evaluation.
"""

import json
from collections import defaultdict

from src.cache import cached, TEMP_DIR
from src.data import ErrorType, generate_synthetic_pairs, split_train_test
from src.model import extract_text_features
from src.classifier import (
    select_features_per_type, train_ovr, predict_tokens_ovr, predict_sentence_ovr,
    evaluate_ovr, OVRClassifier, Metrics,
)

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

DATA_VERSION = "v4"
EXTRACT_VERSION = "v2"

RESULTS_DIR = TEMP_DIR / "results"

EXTRACT_CACHE_KEY = f"{EXTRACT_VERSION}_v4_n{N_PAIRS}_layers={'_'.join(map(str, LAYERS))}_w{WIDTH}"

TYPE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]


# --------------- Main ---------------

def main():
    # 1. Load cached data and features
    all_pairs = cached(
        "synthetic_pairs", DATA_VERSION,
        lambda: generate_synthetic_pairs(N_PAIRS, MIN_WORDS, MAX_WORDS, DATA_SEED),
    )
    print(f"Loaded {len(all_pairs)} pairs")

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

    # 2. Select per-type features and train OVR classifiers
    per_type_feats = select_features_per_type(train_features, LAYERS, train_pairs, min_pair_ratio=MIN_PAIR_RATIO)

    print(f"\nPer-type feature counts:")
    for et in ErrorType:
        feats = per_type_feats.get(et, {})
        total = sum(len(v) for v in feats.values())
        by_layer = ", ".join(f"L{l}:{len(feats.get(l, set()))}" for l in LAYERS)
        print(f"  {et.value}: {total} features ({by_layer})")

    print("\nTraining OVR classifiers (per-type features)...")
    classifier = train_ovr(train_features, train_pairs, LAYERS, per_type_feats)
    print(f"{classifier.summary()}")

    # 3. Per-type threshold sweep
    # For each type, find the threshold that maximizes detection while keeping FP minimal
    print(f"\n{'='*60}")
    print("PER-TYPE THRESHOLD SWEEP")
    print(f"{'='*60}")

    # Group test pairs by type
    test_by_type: dict[ErrorType, list[tuple[dict, dict]]] = defaultdict(list)
    for pf, pair in zip(test_features, test_pairs):
        test_by_type[pair.error_type].append((pf, pair))

    best_thresholds = {}
    type_sweep_results = {}

    for et in ErrorType:
        if et not in classifier.models:
            print(f"\n  {et.value}: NO MODEL (skipped)")
            continue

        print(f"\n  {et.value}:")
        type_pairs = test_by_type[et]
        best_f1 = 0.0
        best_t = 0.5
        sweep = {}

        for threshold in TYPE_THRESHOLDS:
            # Detection: how many error sentences of this type are caught?
            detected = 0
            for pf, pair in type_pairs:
                tpreds = predict_tokens_ovr(pf["error"], classifier)
                max_p = max((tp.error_probs.get(et, 0.0) for tp in tpreds), default=0.0)
                if max_p >= threshold:
                    detected += 1

            # FP: how many clean sentences get flagged by this type's classifier?
            fp = 0
            for pf, pair in zip(test_features, test_pairs):
                tpreds = predict_tokens_ovr(pf["clean"], classifier)
                max_p = max((tp.error_probs.get(et, 0.0) for tp in tpreds), default=0.0)
                if max_p >= threshold:
                    fp += 1

            n = len(type_pairs)
            det_pct = detected / n * 100 if n else 0
            fp_pct = fp / len(test_features) * 100

            # F1 treating this type's detection as TP and clean-flagged as FP
            prec = detected / (detected + fp) if (detected + fp) else 0
            rec = detected / n if n else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

            sweep[threshold] = {"detected": detected, "n": n, "det_pct": round(det_pct, 1),
                                "fp": fp, "fp_pct": round(fp_pct, 1), "f1": round(f1, 4)}

            marker = ""
            if f1 > best_f1:
                best_f1 = f1
                best_t = threshold
                marker = " ← best"
            print(f"    t={threshold}: det={detected}/{n} ({det_pct:.0f}%), FP={fp} ({fp_pct:.1f}%), F1={f1:.1%}{marker}")

        best_thresholds[et] = best_t
        type_sweep_results[et.value] = {"sweep": {str(k): v for k, v in sweep.items()}, "best_threshold": best_t}

    # 4. Combined evaluation with best per-type thresholds
    print(f"\n{'='*60}")
    print("COMBINED EVALUATION (best per-type thresholds)")
    print(f"{'='*60}")

    classifier.thresholds = best_thresholds
    print("  Thresholds:", {et.value: t for et, t in best_thresholds.items()})

    metrics = evaluate_ovr(test_features, classifier)
    print(f"\n{metrics.confusion_str()}")
    print(f"\n{metrics.summary()}")

    # 5. Per-type detection at combined thresholds
    print(f"\n{'='*60}")
    print("PER-TYPE DETECTION (combined thresholds)")
    print(f"{'='*60}")

    type_det = defaultdict(lambda: {"total": 0, "detected": 0, "correct_type": 0})
    for pf, pair in zip(test_features, test_pairs):
        et = pair.error_type
        type_det[et.value]["total"] += 1
        pred = predict_sentence_ovr(pf["error"], classifier)
        if pred.has_errors:
            type_det[et.value]["detected"] += 1
            if pred.predicted_type == et:
                type_det[et.value]["correct_type"] += 1

    print(f"\n  {'Type':<15} {'Total':>6} {'Detected':>9} {'Det%':>6} {'CorrectType':>12} {'TypeAcc%':>9}")
    print(f"  {'-'*58}")
    for et in ErrorType:
        s = type_det[et.value]
        det_pct = s["detected"] / s["total"] * 100 if s["total"] else 0
        type_acc = s["correct_type"] / s["detected"] * 100 if s["detected"] else 0
        print(f"  {et.value:<15} {s['total']:>6} {s['detected']:>9} {det_pct:>5.1f}% {s['correct_type']:>12} {type_acc:>8.1f}%")

    # 6. FP analysis
    print(f"\n{'='*60}")
    print("FALSE POSITIVE EXAMPLES")
    print(f"{'='*60}")

    fp_details = []
    fp_count = 0
    for pf, pair in zip(test_features, test_pairs):
        pred = predict_sentence_ovr(pf["clean"], classifier)
        if pred.has_errors:
            fp_count += 1
            top = sorted(pred.token_predictions, key=lambda t: t.p_error, reverse=True)[:3]
            if fp_count <= 10:
                top_str = ", ".join(f"'{t.token}'={t.p_error:.3f}({t.predicted_type.value if t.predicted_type else '?'})" for t in top)
                print(f"\n  FP: {pair.clean}")
                print(f"    predicted={pred.predicted_type.value if pred.predicted_type else '?'}, top: {top_str}")
            fp_details.append({
                "sentence": pair.clean,
                "max_p_error": round(pred.max_p_error, 4),
                "predicted_type": pred.predicted_type.value if pred.predicted_type else None,
            })
    print(f"\n  Total FPs: {fp_count}")

    # 7. Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "experiment_6_ovr_pertype_feats.json"

    results = {
        "params": {
            "layers": LAYERS, "width": WIDTH,
            "n_pairs": N_PAIRS,
            "train_ratio": TRAIN_RATIO, "split_seed": SPLIT_SEED,
            "min_pair_ratio": MIN_PAIR_RATIO,
            "error_types": [et.value for et in ErrorType],
        },
        "per_type_feature_counts": {
            et.value: sum(len(v) for v in per_type_feats.get(et, {}).values())
            for et in ErrorType
        },
        "best_thresholds": {et.value: t for et, t in best_thresholds.items()},
        "per_type_sweep": type_sweep_results,
        "combined_metrics": metrics.to_dict(),
        "per_type_detection": {k: v for k, v in type_det.items()},
        "false_positives": fp_details,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
