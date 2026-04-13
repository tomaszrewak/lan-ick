"""Experiment 7: Feature selection method comparison.

Flow: load cached data+features → run 6 different feature selection methods →
      train OVR classifiers for each → evaluate all at fixed thresholds → compare.
"""

import json
import warnings
from collections import defaultdict

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.cache import cached, TEMP_DIR
from src.data import ErrorType, generate_synthetic_pairs, split_train_test
from src.model import extract_text_features
from src.classifier import (
    select_features_per_type,
    select_features_relaxed,
    select_features_paired_token_diff,
    select_features_magnitude_diff,
    select_features_ttest,
    select_features_top_k_error,
    train_ovr, predict_tokens_ovr, predict_sentence_ovr,
    evaluate_ovr, Metrics,
)

# --------------- Experiment parameters ---------------

LAYERS = [5, 10, 13, 17, 22]
WIDTH = "16k"
TRAIN_RATIO = 0.75
SPLIT_SEED = 42

N_PAIRS = 300
MIN_WORDS = 8
MAX_WORDS = 20
DATA_SEED = 42

DATA_VERSION = "v4"
EXTRACT_VERSION = "v2"

RESULTS_DIR = TEMP_DIR / "results"

EXTRACT_CACHE_KEY = f"{EXTRACT_VERSION}_v4_n{N_PAIRS}_layers={'_'.join(map(str, LAYERS))}_w{WIDTH}"

EVAL_THRESHOLDS = [0.5, 0.8, 0.9, 0.95]


# --------------- Main ---------------

def main():
    # 1. Load cached data and features
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

    # Group test pairs by type
    test_by_type: dict[ErrorType, list[tuple[dict, dict]]] = defaultdict(list)
    for pf, pair in zip(test_features, test_pairs):
        test_by_type[pair.error_type].append((pf, pair))

    # 2. Define selection methods to compare
    methods = {
        "baseline": lambda: select_features_per_type(
            train_features, LAYERS, train_pairs, min_pair_ratio=0.5),
        "relaxed_30": lambda: select_features_relaxed(
            train_features, LAYERS, train_pairs, min_pair_ratio=0.3),
        "paired_diff": lambda: select_features_paired_token_diff(
            train_features, LAYERS, train_pairs, min_pair_ratio=0.3),
        "magnitude_diff_10": lambda: select_features_magnitude_diff(
            train_features, LAYERS, train_pairs, top_k=10),
        "magnitude_diff_20": lambda: select_features_magnitude_diff(
            train_features, LAYERS, train_pairs, top_k=20),
        "ttest": lambda: select_features_ttest(
            train_features, LAYERS, train_pairs, p_threshold=0.05),
        "ttest_relaxed": lambda: select_features_ttest(
            train_features, LAYERS, train_pairs, p_threshold=0.10),
        "top_k_error_10": lambda: select_features_top_k_error(
            train_features, LAYERS, train_pairs, top_k=10),
        "top_k_error_20": lambda: select_features_top_k_error(
            train_features, LAYERS, train_pairs, top_k=20),
    }

    all_results = {}

    # 3. Run each method
    for method_name, select_fn in methods.items():
        print(f"\n{'='*70}")
        print(f"METHOD: {method_name}")
        print(f"{'='*70}")

        per_type_feats = select_fn()

        # Report feature counts
        feat_counts = {}
        for et in ErrorType:
            feats = per_type_feats.get(et, {})
            total = sum(len(v) for v in feats.values())
            feat_counts[et] = total
            by_layer = ", ".join(f"L{l}:{len(feats.get(l, set()))}" for l in LAYERS)
            print(f"  {et.value}: {total} features ({by_layer})")

        # Train OVR classifiers
        print(f"  Training...")
        try:
            classifier = train_ovr(train_features, train_pairs, LAYERS, per_type_feats)
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results[method_name] = {"error": str(e)}
            continue

        # Evaluate: compute predictions ONCE, then slice by threshold
        method_result = {
            "feature_counts": {et.value: cnt for et, cnt in feat_counts.items()},
            "thresholds": {},
        }

        # Pre-compute max P(error_type) per test pair, for error and clean texts
        # error_scores[i][et] = max P(et) across tokens in error text of pair i
        # clean_scores[i][et] = max P(et) across tokens in clean text of pair i
        error_scores: list[dict[ErrorType, float]] = []
        clean_scores: list[dict[ErrorType, float]] = []

        print(f"  Evaluating on {len(test_features)} test pairs...")
        for pf in test_features:
            # Error text
            tpreds_error = predict_tokens_ovr(pf["error"], classifier)
            e_scores = {}
            for et in classifier.models:
                e_scores[et] = max((tp.error_probs.get(et, 0.0) for tp in tpreds_error), default=0.0)
            error_scores.append(e_scores)

            # Clean text
            tpreds_clean = predict_tokens_ovr(pf["clean"], classifier)
            c_scores = {}
            for et in classifier.models:
                c_scores[et] = max((tp.error_probs.get(et, 0.0) for tp in tpreds_clean), default=0.0)
            clean_scores.append(c_scores)

        # Now evaluate at each threshold from pre-computed scores
        for threshold in EVAL_THRESHOLDS:
            type_metrics = {}
            for et in ErrorType:
                if et not in classifier.models:
                    type_metrics[et.value] = {"detected": 0, "n": 0, "fp": 0, "det_pct": 0, "fp_pct": 0}
                    continue

                # Detection: error texts of this type
                type_test_indices = [i for i, pair in enumerate(test_pairs) if pair.error_type == et]
                detected = sum(1 for i in type_test_indices if error_scores[i].get(et, 0) >= threshold)

                # FP: clean texts flagged by this type
                fp = sum(1 for cs in clean_scores if cs.get(et, 0) >= threshold)

                n = len(type_test_indices)
                type_metrics[et.value] = {
                    "detected": detected, "n": n,
                    "det_pct": round(detected / n * 100, 1) if n else 0,
                    "fp": fp, "fp_pct": round(fp / len(test_features) * 100, 1),
                }

            # Combined binary evaluation (from pre-computed scores)
            tp = sum(1 for es in error_scores if any(
                es.get(et, 0) >= threshold for et in classifier.models))
            fn = len(error_scores) - tp
            fp_total = sum(1 for cs in clean_scores if any(
                cs.get(et, 0) >= threshold for et in classifier.models))
            tn = len(clean_scores) - fp_total
            combined = Metrics(tp=tp, fp=fp_total, tn=tn, fn=fn)

            method_result["thresholds"][str(threshold)] = {
                "per_type": type_metrics,
                "combined": combined.to_dict(),
            }

        # Print summary table at threshold=0.9
        t09 = method_result["thresholds"].get("0.9", method_result["thresholds"].get("0.5"))
        print(f"\n  At threshold=0.9:")
        print(f"  {'Type':<15} {'Feats':>6} {'Det':>5} {'FP':>5}")
        print(f"  {'-'*35}")
        for et in ErrorType:
            n_feats = feat_counts.get(et, 0)
            tm = t09["per_type"][et.value]
            det_str = f"{tm['det_pct']:.0f}%" if tm['n'] else "—"
            fp_str = f"{tm['fp_pct']:.1f}%"
            print(f"  {et.value:<15} {n_feats:>6} {det_str:>5} {fp_str:>5}")
        cm = t09["combined"]
        print(f"  Combined: F1={cm['f1']:.1%}  P={cm['precision']:.1%}  R={cm['recall']:.1%}")

        all_results[method_name] = method_result

    # 4. Summary comparison across methods
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON (threshold=0.9)")
    print(f"{'='*70}")

    header = f"  {'Method':<22}"
    for et in ErrorType:
        header += f" {et.value[:6]:>7}"
    header += f" {'F1':>6} {'FP':>4}"
    print(header)
    print(f"  {'-'*(22 + 7*6 + 11)}")

    for method_name, mr in all_results.items():
        if "error" in mr:
            print(f"  {method_name:<22} FAILED: {mr['error'][:40]}")
            continue
        t09 = mr["thresholds"].get("0.9", mr["thresholds"].get("0.5"))
        row = f"  {method_name:<22}"
        for et in ErrorType:
            tm = t09["per_type"][et.value]
            if tm["n"]:
                row += f" {tm['det_pct']:>6.0f}%"
            else:
                row += f"     —"
        cm = t09["combined"]
        row += f" {cm['f1']:>5.1%} {cm['fp']:>4}"
        print(row)

    # Also print at threshold=0.5 for comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON (threshold=0.5)")
    print(f"{'='*70}")
    print(header)
    print(f"  {'-'*(22 + 7*6 + 11)}")

    for method_name, mr in all_results.items():
        if "error" in mr:
            continue
        t05 = mr["thresholds"]["0.5"]
        row = f"  {method_name:<22}"
        for et in ErrorType:
            tm = t05["per_type"][et.value]
            if tm["n"]:
                row += f" {tm['det_pct']:>6.0f}%"
            else:
                row += f"     —"
        cm = t05["combined"]
        row += f" {cm['f1']:>5.1%} {cm['fp']:>4}"
        print(row)

    # 5. Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "experiment_7_feature_selection.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
